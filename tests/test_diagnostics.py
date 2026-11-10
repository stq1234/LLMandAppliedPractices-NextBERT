from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnostics_nextbert import (  # noqa: E402
    _counterfactual_answer,
    _profile_paths,
    _prepare_probe_training_rows,
    build_parser,
    build_long_case,
    freeze_probe_parameters,
    load_heldout_backgrounds,
    main as diagnostics_main,
    unique_source_questions,
)


class ToyEncoding(dict):
    def __init__(self, *args, sequence_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sequence_ids = sequence_ids or []

    def sequence_ids(self, batch_index: int = 0):
        if self._sequence_ids and isinstance(self._sequence_ids[0], list):
            return self._sequence_ids[batch_index]
        return self._sequence_ids


class ToyTokenizer:
    """Whitespace fast-tokenizer stand-in with pair offsets and overflow windows."""

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.is_fast = True

    def num_special_tokens_to_add(self, pair: bool = True) -> int:
        return 3 if pair else 2

    def _tokens(self, text: str):
        out = []
        for match in re.finditer(r"\S+", text):
            word = match.group(0)
            if word not in self.token_to_id:
                token_id = len(self.token_to_id) + 10
                self.token_to_id[word] = token_id
                self.id_to_token[token_id] = word
            out.append((self.token_to_id[word], (match.start(), match.end())))
        return out

    def decode(self, token_ids, **_kwargs):
        return " ".join(self.id_to_token[int(token_id)] for token_id in token_ids)

    def __call__(self, text, text_pair=None, *, add_special_tokens=True,
                 return_offsets_mapping=False, truncation=False, max_length=None,
                 return_overflowing_tokens=False, stride=0, padding=False, **_kwargs):
        first = self._tokens(str(text))
        if text_pair is None:
            ids = [item[0] for item in first]
            offsets = [item[1] for item in first]
            return {"input_ids": ids, "offset_mapping": offsets}

        second = self._tokens(str(text_pair))
        first_ids = [item[0] for item in first]
        first_offsets = [item[1] for item in first]
        second_ids = [item[0] for item in second]
        second_offsets = [item[1] for item in second]
        capacity = len(second_ids)
        if max_length is not None:
            capacity = max(1, max_length - len(first_ids) - self.num_special_tokens_to_add(pair=True))
        windows = []
        if truncation == "only_second" and len(second_ids) > capacity:
            step = max(1, capacity - int(stride)) if return_overflowing_tokens else capacity
            for start in range(0, len(second_ids), step):
                stop = min(len(second_ids), start + capacity)
                windows.append((second_ids[start:stop], second_offsets[start:stop]))
                if stop >= len(second_ids) or not return_overflowing_tokens:
                    break
        else:
            windows = [(second_ids, second_offsets)]

        result_ids, result_offsets, result_seq = [], [], []
        for ctx_ids, ctx_offsets in windows:
            ids = [1, *first_ids, 2, *ctx_ids, 2]
            offsets = [(0, 0), *first_offsets, (0, 0), *ctx_offsets, (0, 0)]
            seq = [None, *([0] * len(first_ids)), None, *([1] * len(ctx_ids)), None]
            result_ids.append(ids)
            result_offsets.append(offsets)
            result_seq.append(seq)
        if len(result_ids) == 1:
            return ToyEncoding(
                {"input_ids": result_ids[0], "offset_mapping": result_offsets[0]},
                sequence_ids=result_seq[0],
            )
        return ToyEncoding(
            {"input_ids": result_ids, "offset_mapping": result_offsets},
            sequence_ids=result_seq,
        )


def _base(question_id="q1", title="Article A", answer="Paris", context=None):
    context = context or f"The capital is {answer} and this is evidence text."
    return {
        "id": question_id,
        "question_id": question_id,
        "language": "en",
        "title": title,
        "question": "What is the capital?",
        "context": context,
        "answer_text": answer,
        "answer_start_char": context.index(answer),
        "answers": {"text": [answer], "answer_start": [context.index(answer)]},
        "source_id": "qa-source",
        "source_revision": "qa-rev",
    }


def _background(source_id: str, split: str = "dev"):
    prefix = re.sub(r"\W+", "", source_id)
    text = " ".join([f"{prefix}unrelated{index}" for index in range(120)])
    return {"source_id": source_id, "split": split, "text": text}


def test_long_inputs_have_exact_length_answer_and_evidence_masks():
    tokenizer = ToyTokenizer()
    base = _base()
    pool = [
        {"source_id": "wiki-dev-1", "split": "dev", "ids": [item[0] for item in tokenizer._tokens(_background("x")["text"])]},
        {"source_id": "wiki-test-2", "split": "test", "ids": [item[0] for item in tokenizer._tokens(_background("y")["text"])]},
    ]
    fractions = []
    for position in ("first", "middle", "last"):
        case = build_long_case(
            tokenizer, base, pool, target_length=64, position=position,
            condition="intact", seed=13,
        )
        assert case is not None
        assert len(case["input_ids"]) == 64
        assert sum(case["context_mask"]) == len(case["input_ids"]) - len(tokenizer._tokens(base["question"])) - tokenizer.num_special_tokens_to_add(pair=True)
        assert case["context_mask"][case["start_positions"]] == 1
        assert case["context_mask"][case["end_positions"]] == 1
        assert case["evidence_mask"][case["start_positions"]] == 1
        assert case["context"][case["offset_mapping"][case["start_positions"]][0]:case["offset_mapping"][case["end_positions"]][1]] == "Paris"
        assert case["answer_start_char"] == case["context"].index("Paris")
        assert all(source_id.startswith("wiki-") for source_id in case["background_source_ids"])
        assert 0.0 <= case["evidence_fraction"] < 1.0
        fractions.append(case["evidence_fraction"])
    assert case["natural_long_qa"] is False
    assert fractions[0] < fractions[1] < fractions[2]


def test_evidence_deletion_uses_natural_tokens_and_drops_answer_label():
    tokenizer = ToyTokenizer()
    base = _base()
    text = _background("wiki-a")["text"]
    ids = [item[0] for item in tokenizer._tokens(text)]
    pool = [{"source_id": "wiki-dev", "split": "dev", "ids": ids}]
    case = build_long_case(
        tokenizer, base, pool, target_length=64, position="middle",
        condition="evidence_deleted", seed=21,
    )
    assert case is not None
    assert len(case["input_ids"]) == 64
    assert case["start_positions"] == case["end_positions"] == -100
    assert case["answers"] == {"text": [], "answer_start": []}
    assert "Paris" not in case["context"]
    assert any(source_id == "wiki-dev" for source_id in case["background_source_ids"])
    assert len(tokenizer._tokens(case["context"])) == sum(case["context_mask"])


def test_background_control_keeps_gold_and_context_masks():
    tokenizer = ToyTokenizer()
    base = _base()
    pool = [
        {"source_id": "wiki-heldout-a", "split": "test", "ids": [item[0] for item in tokenizer._tokens(_background("z")["text"])]},
        {"source_id": "wiki-heldout-b", "split": "dev", "ids": [item[0] for item in tokenizer._tokens(_background("b")["text"])]},
        {"source_id": "wiki-heldout-c", "split": "test", "ids": [item[0] for item in tokenizer._tokens(_background("c")["text"])]},
    ]
    case = build_long_case(
        tokenizer, base, pool, target_length=64, position="last",
        condition="background_replaced", seed=44,
    )
    intact = build_long_case(
        tokenizer, base, pool, target_length=64, position="last",
        condition="intact", seed=44,
    )
    assert case is not None
    assert intact is not None
    assert case["expected_answer"] == "Paris"
    assert case["context"][case["answer_start_char"]:case["answer_end_char"]] == "Paris"
    assert case["context_mask"][case["start_positions"]] == 1
    assert not set(case["background_source_ids"]) & set(intact["background_source_ids"])


def test_source_groups_and_background_holdout_filter(tmp_path):
    rows = [_base("q1", "same article"), _base("q2", "same article"), _base("q3", "other article")]
    selected = unique_source_questions(rows, max_groups=1)
    assert len(selected) == 1
    assert selected[0]["source_group_id"].startswith("en:")

    path = tmp_path / "selected_en.jsonl"
    path.write_text(
        json.dumps({"source_id": "train", "split": "train", "text": "training"}) + "\n"
        + json.dumps({"source_id": "heldout", "split": "dev", "text": "held out"}) + "\n",
        encoding="utf-8",
    )
    backgrounds = load_heldout_backgrounds(tmp_path, "en")
    assert [row["source_id"] for row in backgrounds] == ["heldout"]
    assert all(row["split"] != "train" for row in backgrounds)


def test_counterfactual_requires_matching_token_length_and_numeric_type():
    tokenizer = ToyTokenizer()
    base = _base(answer="2020", context="Year 2020 ended.")
    candidates = [
        _base("q2", "article2", "2021", "Year 2021 ended."),
        _base("q3", "article3", "Paris", "Capital Paris."),
    ]
    assert _counterfactual_answer(tokenizer, base, candidates) == "2021"
    assert _counterfactual_answer(tokenizer, base, []) is None
    pool = [{"source_id": "wiki", "split": "dev", "ids": [item[0] for item in tokenizer._tokens(_background("w")["text"])]}]
    counterfactual = build_long_case(
        tokenizer, base, pool, target_length=64, position="middle",
        condition="counterfactual", seed=91, alternative_answer="2021",
    )
    assert counterfactual is not None
    assert counterfactual["expected_answer"] == "2021"
    assert counterfactual["context"][counterfactual["answer_start_char"]:counterfactual["answer_end_char"]] == "2021"
    assert counterfactual["answers"] == {"text": ["2021"], "answer_start": [counterfactual["answer_start_char"]]}


def test_probe_freeze_leaves_only_selected_head_trainable():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.encoder = nn.Linear(4, 4)
            self.model.attn = nn.Module()
            self.model.attn.dual_gate_logits = nn.Parameter(torch.zeros(6))
            self.classifier = nn.Linear(4, 3)
            self.qa_outputs = nn.Linear(4, 2)

    model = ToyModel()
    info = freeze_probe_parameters(model, "classification")
    assert info["trainable_parameter_count"] == model.classifier.weight.numel() + model.classifier.bias.numel()
    assert model.classifier.weight.requires_grad
    assert not model.qa_outputs.weight.requires_grad
    assert not model.model.encoder.weight.requires_grad
    assert not model.model.attn.dual_gate_logits.requires_grad


def test_frozen_qa_probe_excludes_long_augmentation_rows():
    rows = [
        {"id": "regular-512", "input_ids": [1] * 512, "window_type": "regular"},
        {"id": "augmentation-short", "input_ids": [1] * 128, "window_type": "long"},
        {"id": "augmentation-4096", "input_ids": [1] * 4096, "window_type": "long"},
        {"id": "oversized-regular", "input_ids": [1] * 513, "window_type": "regular"},
    ]
    filtered, stats = _prepare_probe_training_rows("qa", rows)
    assert [row["id"] for row in filtered] == ["regular-512"]
    assert stats == {
        "applied": True,
        "rule": "input_ids length <= 512 and window_type != 'long'",
        "rows_before": 4,
        "rows_after": 1,
        "rows_removed": 3,
    }


def test_full_67m_profile_defaults_and_explicit_long_paths(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    prepared, wiki = _profile_paths("full_67m")
    assert prepared == root / "data" / "prepared" / "full_67m"
    assert wiki == root / "artifacts" / "raw" / "full_67m" / "wikipedia"
    assert "full_67m" in build_parser()._option_string_actions["--profile"].choices

    captured = []
    monkeypatch.setattr("diagnostics_nextbert.run_long_command", lambda args: captured.append(args))
    diagnostics_main([
        "--mode", "long", "--model-dir", "model", "--variant", "V0",
        "--profile", "full_67m", "--max-source-groups", "4",
    ])
    assert captured[-1].qa_data == prepared / "qa_test.jsonl"
    assert captured[-1].wiki_background_dir == wiki

    custom_qa = root / "custom" / "qa.jsonl"
    custom_wiki = root / "custom" / "wiki"
    diagnostics_main([
        "--mode", "long", "--model-dir", "model", "--variant", "V0",
        "--profile", "full_67m", "--max-source-groups", "4",
        "--qa-data", str(custom_qa), "--wiki-background-dir", str(custom_wiki),
    ])
    assert captured[-1].qa_data == custom_qa
    assert captured[-1].wiki_background_dir == custom_wiki

