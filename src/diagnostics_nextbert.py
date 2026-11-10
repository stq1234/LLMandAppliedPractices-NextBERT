"""Long-context evidence diagnostics and frozen-head retention probes.

These commands run only when explicitly requested. Long-context cases combine
real held-out QA examples with held-out Wikipedia background text; they are
controlled synthetic documents, not naturally occurring long-form QA. Probe
runs freeze the encoder and all attention gates and train only a fresh task
head for the explicitly requested number of updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_nextbert import (  # noqa: E402
    evaluate_classification,
    evaluate_qa,
)
from modeling_nextbert import NextBert  # noqa: E402
from qa_metrics import answer_em, cmrc_normalize, squad_normalize  # noqa: E402
from train_nextbert import (  # noqa: E402
    _amp_context,
    _device,
    _seed_everything,
    collate_sequences,
    load_jsonl,
)
from transformers import AutoTokenizer  # noqa: E402


LENGTHS = (1024, 4096, 8192)
POSITIONS = ("first", "middle", "last")
LONG_CONDITIONS = ("intact", "evidence_deleted", "background_replaced", "counterfactual")
PROBE_TASKS = ("classification", "qa")
QA_BATCH_SIZE = 16
PROBE_STEPS = 100
PROBE_LR = 1e-4
PROBE_WEIGHT_DECAY = 0.01
PROBE_WARMUP_FRACTION = 0.05
SLIDING_STRIDE = 128


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _normalized_answer(text: str, language: str) -> str:
    return cmrc_normalize(text) if language == "zh" else squad_normalize(text)


def _answer_pair(row: Mapping[str, Any]) -> tuple[str, int] | None:
    answer_text = str(row.get("answer_text") or "")
    answer_start = row.get("answer_start_char")
    if answer_text and answer_start is not None:
        return answer_text, int(answer_start)
    answers = row.get("answers") or {}
    if isinstance(answers, Mapping):
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(starts, int):
            starts = [starts]
        for text, start in zip(texts, starts):
            if text and start is not None:
                return str(text), int(start)
    return None


def source_group_id(row: Mapping[str, Any]) -> str:
    """Return an article-level group, falling back to one question per group."""
    language = str(row.get("language", "und"))
    title = " ".join(str(row.get("title") or "").split())
    if title:
        identity = title
    else:
        identity = str(row.get("question_id", row.get("id", "")))
    return f"{language}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def unique_source_questions(rows: Sequence[Mapping[str, Any]], max_groups: int) -> list[dict[str, Any]]:
    """Select one deterministic question per source article and language."""
    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = _answer_pair(row)
        context = str(row.get("context") or "")
        question = str(row.get("question") or "")
        qid = str(row.get("question_id", row.get("id", "")))
        if not pair or not context or not question or not qid:
            continue
        grouped[source_group_id(row)].append(row)
    by_language: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for group, candidates in grouped.items():
        selected = min(candidates, key=lambda row: _stable_int(str(row.get("question_id", row.get("id", "")))))
        by_language[str(selected.get("language", "und"))].append((group, selected))
    result = []
    for language, candidates in sorted(by_language.items()):
        candidates.sort(key=lambda pair: _stable_int(pair[0]))
        for group, row in candidates[:max_groups]:
            item = dict(row)
            item["source_group_id"] = group
            item["question_id"] = str(row.get("question_id", row.get("id")))
            result.append(item)
    return result


def load_heldout_backgrounds(directory: Path, language: str) -> list[dict[str, Any]]:
    path = directory / f"selected_{language}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"held-out Wikipedia background file is missing: {path}")
    rows = _jsonl(path)
    heldout = [
        row for row in rows
        if str(row.get("split", "")) in {"dev", "test"}
        and str(row.get("text") or "").strip()
        and row.get("source_id")
    ]
    if not heldout:
        raise ValueError(f"no held-out Wikipedia documents found in {path}")
    if any(str(row.get("split")) == "train" for row in heldout):
        raise AssertionError("training Wiki text leaked into the diagnostic background pool")
    return heldout


def _tokenize_plain(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    return [int(token) for token in ids], [(int(a), int(b)) for a, b in offsets]


def _background_pool(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pool = []
    for row in rows:
        ids, _ = _tokenize_plain(tokenizer, str(row.get("text") or ""))
        if ids:
            pool.append({"source_id": str(row["source_id"]), "split": str(row["split"]), "ids": ids})
    return pool


def _background_text(
    tokenizer: Any,
    pool: Sequence[Mapping[str, Any]],
    token_count: int,
    *,
    seed: int,
    exclude_source_ids: set[str] | None = None,
) -> tuple[str, list[str], int]:
    if token_count <= 0:
        return "", [], 0
    excluded = exclude_source_ids or set()
    order = sorted(
        (index for index, item in enumerate(pool) if str(item["source_id"]) not in excluded),
        key=lambda index: _stable_int(f"{seed}:{pool[index]['source_id']}"),
    )
    fragments: list[str] = []
    source_ids: list[str] = []
    taken = 0
    for index in order:
        item = pool[index]
        ids = list(item["ids"])
        take = min(len(ids), token_count - taken)
        if take <= 0:
            break
        text = tokenizer.decode(ids[:take], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if text.strip():
            fragments.append(text)
            source_ids.append(str(item["source_id"]))
            taken += take
        if taken >= token_count:
            break
    return "\n\n".join(fragments), source_ids, taken


def _split_background_text(tokenizer: Any, text: str, prefix_tokens: int, total_tokens: int) -> tuple[str, str] | None:
    """Split one natural background stream at a token boundary."""
    ids, offsets = _tokenize_plain(tokenizer, text)
    if len(ids) < total_tokens or not 0 <= prefix_tokens <= total_tokens:
        return None
    prefix = text[:offsets[prefix_tokens - 1][1]] if prefix_tokens else ""
    suffix = text[offsets[prefix_tokens][0]:] if prefix_tokens < total_tokens else ""
    return prefix, suffix


def _find_span(offsets: Sequence[Sequence[int]], sequence_ids: Sequence[int | None], start: int, end: int) -> tuple[int, int] | None:
    starts = [i for i, (a, b) in enumerate(offsets) if sequence_ids[i] == 1 and int(a) <= start < int(b)]
    ends = [i for i, (a, b) in enumerate(offsets) if sequence_ids[i] == 1 and int(a) < end <= int(b)]
    if not starts or not ends or starts[0] > ends[-1]:
        return None
    return starts[0], ends[-1]


def _evidence_region(
    tokenizer: Any,
    context: str,
    answer_text: str,
    answer_start: int,
    max_tokens: int,
) -> tuple[str, int, int, int, int] | None:
    if context[answer_start:answer_start + len(answer_text)] != answer_text:
        answer_start = context.find(answer_text)
    if answer_start < 0:
        return None
    answer_end = answer_start + len(answer_text)
    ids, offsets = _tokenize_plain(tokenizer, context)
    span = _find_span(offsets, [1] * len(offsets), answer_start, answer_end)
    if span is None:
        return None
    token_start, token_end = span
    answer_tokens = token_end - token_start + 1
    cap = min(len(ids), max_tokens)
    if cap < answer_tokens:
        return None
    if len(ids) <= cap:
        left, right = 0, len(ids)
    else:
        left = max(0, token_start - max(0, (cap - answer_tokens) // 2))
        right = min(len(ids), left + cap)
        left = max(0, right - cap)
        if not (left <= token_start <= token_end < right):
            return None
    char_start = offsets[left][0]
    char_end = offsets[right - 1][1]
    if char_start > answer_start or char_end < answer_end:
        return None
    return context[char_start:char_end], char_start, char_end, answer_start - char_start, answer_end - char_start


def _pair_encoding(tokenizer: Any, question: str, context: str, max_length: int | None = None, *, overflow: bool = False) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "truncation": "only_second" if max_length is not None else False,
        "return_offsets_mapping": True,
        "padding": False,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    if overflow:
        kwargs["return_overflowing_tokens"] = True
        kwargs["stride"] = SLIDING_STRIDE
    encoded = tokenizer(question, context, **kwargs)
    input_rows = encoded["input_ids"]
    is_batch = bool(input_rows) and isinstance(input_rows[0], list)
    if not is_batch:
        input_rows = [input_rows]
    outputs = []
    for index, ids in enumerate(input_rows):
        feature_offsets = encoded["offset_mapping"][index] if is_batch else encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids(index) if is_batch else encoded.sequence_ids()
        offsets = [[int(a), int(b)] for a, b in feature_offsets]
        mask = [int(segment == 1) for segment in sequence_ids]
        if not any(mask):
            continue
        outputs.append({"input_ids": [int(token) for token in ids], "context_mask": mask, "offset_mapping": offsets})
    return outputs


def _numeric_type(text: str) -> str:
    value = text.strip().lower()
    if re.search(r"\d", value):
        if re.search(r"\b(?:19|20)\d{2}\b|\d{1,4}[./年-]\d{1,2}(?:[./月-]\d{1,4})?", value):
            return "date_or_year"
        if re.search(r"\d+[.,]\d+", value):
            return "decimal"
        if re.fullmatch(r"[+-]?\d+", value):
            return "integer"
        return "numeric_mixed"
    return "non_numeric"


def _counterfactual_answer(
    tokenizer: Any,
    base: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> str | None:
    pair = _answer_pair(base)
    if pair is None:
        return None
    original, _ = pair
    original_length = len(_tokenize_plain(tokenizer, original)[0])
    original_type = _numeric_type(original)
    valid = []
    for row in candidates:
        if row.get("question_id") == base.get("question_id"):
            continue
        pair = _answer_pair(row)
        if not pair:
            continue
        text = pair[0]
        if answer_em(text, original, str(base.get("language", "und"))) == 1.0:
            continue
        if len(_tokenize_plain(tokenizer, text)[0]) != original_length:
            continue
        if _numeric_type(text) != original_type:
            continue
        valid.append(text)
    if not valid:
        return None
    return min(valid, key=lambda text: _stable_int(f"{base.get('question_id')}:{text}"))


def build_long_case(
    tokenizer: Any,
    base: Mapping[str, Any],
    background_pool: Sequence[Mapping[str, Any]],
    *,
    target_length: int,
    position: str,
    condition: str,
    seed: int,
    alternative_answer: str | None = None,
) -> dict[str, Any] | None:
    """Build one exact-length QA pair with verifiable answer/context masks."""
    if position not in POSITIONS or condition not in LONG_CONDITIONS:
        raise ValueError("invalid evidence position or intervention")
    pair = _answer_pair(base)
    if pair is None:
        return None
    answer_text, answer_start = pair
    question = str(base["question"])
    source_context = str(base["context"])
    question_ids, _ = _tokenize_plain(tokenizer, question)
    pair_specials = int(tokenizer.num_special_tokens_to_add(pair=True))
    context_budget = target_length - len(question_ids) - pair_specials
    if context_budget < 32:
        return None
    region = _evidence_region(tokenizer, source_context, answer_text, answer_start, max_tokens=max(16, context_budget // 2))
    if region is None:
        return None
    evidence_text, _region_start, _region_end, relative_start, relative_end = region
    expected_answer: str | None = answer_text
    intervention_sources: set[str] = set()

    if condition == "counterfactual":
        if not alternative_answer:
            return None
        found = evidence_text.find(answer_text, relative_start, relative_end)
        if found < 0:
            return None
        evidence_text = evidence_text[:found] + alternative_answer + evidence_text[found + len(answer_text):]
        relative_start = found
        relative_end = found + len(alternative_answer)
        expected_answer = alternative_answer
    elif condition == "evidence_deleted":
        region_tokens = len(_tokenize_plain(tokenizer, evidence_text)[0])
        replacement, replacement_sources, replaced_count = _background_text(
            tokenizer, background_pool, region_tokens,
            seed=_stable_int(f"{seed}:{base.get('question_id')}:deleted"),
            exclude_source_ids=set(),
        )
        replacement_token_count = len(_tokenize_plain(tokenizer, replacement)[0])
        if (
            replaced_count != region_tokens
            or replacement_token_count != region_tokens
            or not replacement.strip()
        ):
            return None
        intervention_sources.update(replacement_sources)
        evidence_text = replacement
        relative_start = relative_end = -1
        expected_answer = None

    desired_fraction = {"first": 0.05, "middle": 0.5, "last": 0.95}[position]
    evidence_tokens = len(_tokenize_plain(tokenizer, evidence_text)[0])
    background_budget = max(0, context_budget - evidence_tokens - 4)
    prefix_count = int(round(background_budget * desired_fraction))
    suffix_count = background_budget - prefix_count
    if position == "first":
        prefix_count = min(prefix_count, max(0, evidence_tokens // 8))
        suffix_count = background_budget - prefix_count
    elif position == "last":
        suffix_count = min(suffix_count, max(0, evidence_tokens // 8))
        prefix_count = background_budget - suffix_count

    background_seed = seed
    if condition == "background_replaced":
        background_seed += 100003
    prefix_seed = _stable_int(f"{background_seed}:{base.get('question_id')}:prefix")
    suffix_seed = _stable_int(f"{background_seed}:{base.get('question_id')}:suffix")
    used_sources: set[str] = set()
    baseline_sources: set[str] = set()
    final_context = ""
    prefix_text = suffix_text = ""
    actual_length = 0
    for _ in range(48):
        total_background = prefix_count + suffix_count
        if condition == "background_replaced":
            _base_stream, base_sources, base_taken = _background_text(
                tokenizer, background_pool, total_background,
                seed=_stable_int(f"{seed}:{base.get('question_id')}:prefix"),
            )
            baseline_sources = set(base_sources)
            if base_taken < total_background:
                return None
            baseline_split = _split_background_text(tokenizer, _base_stream, prefix_count, total_background)
            if baseline_split is None:
                return None
            baseline_prefix, baseline_suffix = baseline_split
        else:
            baseline_sources = set()
            baseline_prefix = baseline_suffix = ""
        background_text, background_sources, background_taken = _background_text(
            tokenizer,
            background_pool,
            total_background,
            seed=prefix_seed,
            exclude_source_ids=baseline_sources,
        )
        if background_taken < total_background:
            return None
        split_background = _split_background_text(tokenizer, background_text, prefix_count, total_background)
        if split_background is None:
            return None
        prefix_text, suffix_text = split_background
        used_sources = set(background_sources)
        pieces = [piece for piece in (prefix_text, evidence_text, suffix_text) if piece]
        final_context = "\n\n".join(pieces)
        encoded_len = tokenizer(question, final_context, padding=False)["input_ids"]
        if encoded_len and isinstance(encoded_len[0], list):
            encoded_len = encoded_len[0]
        actual_length = len(encoded_len)
        delta = target_length - actual_length
        if delta == 0:
            break
        if delta > 0:
            if position == "first":
                suffix_count += delta
            elif position == "last":
                prefix_count += delta
            else:
                prefix_count += delta // 2
                suffix_count += delta - delta // 2
        else:
            trim = -delta
            if position == "first":
                reduce_suffix = min(suffix_count, trim)
                suffix_count -= reduce_suffix
                prefix_count = max(0, prefix_count - (trim - reduce_suffix))
            elif position == "last":
                reduce_prefix = min(prefix_count, trim)
                prefix_count -= reduce_prefix
                suffix_count = max(0, suffix_count - (trim - reduce_prefix))
            else:
                reduce_prefix = min(prefix_count, trim // 2)
                reduce_suffix = min(suffix_count, trim - reduce_prefix)
                prefix_count -= reduce_prefix
                suffix_count -= reduce_suffix
    if actual_length != target_length:
        return None
    if condition == "background_replaced" and baseline_sources & used_sources:
        return None

    sep_len = len(prefix_text) + (2 if prefix_text else 0)
    evidence_char_start = sep_len
    evidence_char_end = evidence_char_start + len(evidence_text)
    answer_char_start = evidence_char_start + relative_start if expected_answer is not None else -1
    answer_char_end = answer_char_start + len(expected_answer) if expected_answer is not None else -1
    if expected_answer is not None and final_context[answer_char_start:answer_char_end] != expected_answer:
        return None

    features = _pair_encoding(tokenizer, question, final_context, max_length=target_length, overflow=False)
    if len(features) != 1 or len(features[0]["input_ids"]) != target_length:
        return None
    feature = features[0]
    span = _find_span(feature["offset_mapping"], [1 if m else 0 for m in feature["context_mask"]], answer_char_start, answer_char_end) if expected_answer is not None else None
    if expected_answer is not None and span is None:
        return None
    start_position, end_position = span if span is not None else (-100, -100)
    evidence_mask = [
        int(mask and int(offset[0]) < evidence_char_end and int(offset[1]) > evidence_char_start)
        for mask, offset in zip(feature["context_mask"], feature["offset_mapping"])
    ]
    if not any(feature["context_mask"]) or not any(evidence_mask):
        return None
    context_positions = [i for i, valid in enumerate(feature["context_mask"]) if valid]
    evidence_positions = [i for i, valid in enumerate(evidence_mask) if valid]
    evidence_fraction = (evidence_positions[0] - context_positions[0]) / max(1, len(context_positions))
    qid = str(base.get("question_id", base.get("id")))
    case_id = f"{qid}|{condition}|{target_length}|{position}"
    if expected_answer is None:
        answers = {"text": [], "answer_start": []}
    else:
        answers = {"text": [expected_answer], "answer_start": [answer_char_start]}
    return {
        **feature,
        "id": case_id,
        "question_id": case_id,
        "source_question_id": qid,
        "source_group_id": str(base.get("source_group_id") or source_group_id(base)),
        "language": str(base.get("language", "und")),
        "question": question,
        "context": final_context,
        "answers": answers,
        "answer_text": expected_answer or "",
        "start_positions": start_position,
        "end_positions": end_position,
        "evidence_mask": evidence_mask,
        "condition": condition,
        "evidence_position": position,
        "evidence_fraction": evidence_fraction,
        "input_length": len(feature["input_ids"]),
        "target_length": target_length,
        "expected_answer": expected_answer,
        "original_answer": answer_text,
        "background_source_ids": sorted(used_sources | intervention_sources),
        "answer_start_char": answer_char_start,
        "answer_end_char": answer_char_end,
        "background_split_policy": "Wikipedia split in {dev,test}; training articles excluded",
        "window_type": "controlled_synthetic_long_document",
        "natural_long_qa": False,
    }


def _window_view_rows(tokenizer: Any, case: Mapping[str, Any], view: str) -> list[dict[str, Any]]:
    if view == "full":
        return [dict(case)]
    question, context = str(case["question"]), str(case["context"])
    overflow = view == "sliding512"
    features = _pair_encoding(tokenizer, question, context, max_length=512, overflow=overflow)
    if view == "first512":
        features = features[:1]
    if view not in {"first512", "sliding512"}:
        raise ValueError(f"unknown context view: {view}")
    outputs = []
    for index, feature in enumerate(features):
        outputs.append({
            **case,
            **feature,
            "id": f"{case['question_id']}|{view}|window:{index}",
            "question_id": str(case["question_id"]),
            "window_index": index,
            "window_type": view,
        })
    return outputs


def _case_metrics(case: Mapping[str, Any], prediction: Mapping[str, Any], view: str) -> dict[str, Any]:
    predicted = str(prediction.get("prediction_text", ""))
    expected = case.get("expected_answer")
    original = str(case.get("original_answer", ""))
    language = str(case.get("language", "und"))
    if expected is None:
        factual_correct = answer_em(predicted, original, language) == 0.0
        scored = False
    else:
        factual_correct = answer_em(predicted, str(expected), language) == 1.0
        scored = True
    return {
        "source_group_id": case["source_group_id"],
        "source_question_id": case["source_question_id"],
        "question_id": case["question_id"],
        "language": case["language"],
        "condition": case["condition"],
        "length": case["target_length"],
        "evidence_position": case["evidence_position"],
        "evidence_fraction": case["evidence_fraction"],
        "view": view,
        "input_length": case["input_length"] if view == "full" else None,
        "prediction_text": predicted,
        "expected_answer": expected,
        "original_answer": original,
        "exact_match": prediction.get("exact_match") if scored else None,
        "f1": prediction.get("f1") if scored else None,
        "fact_change_correct": int(factual_correct),
        "scored_against_answer": scored,
    }


def _aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        key = (
            case["language"], case["source_group_id"], case["condition"],
            case["length"], case["evidence_position"], case["view"],
        )
        groups[key].append(case)
    group_rows = []
    for key, rows in sorted(groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        scored = [row for row in rows if row["scored_against_answer"]]
        group_rows.append({
            "language": key[0],
            "source_group_id": key[1],
            "condition": key[2],
            "length": key[3],
            "evidence_position": key[4],
            "view": key[5],
            "count": len(rows),
            "em": sum(float(row["exact_match"] or 0.0) for row in scored) / max(1, len(scored)) if scored else None,
            "f1": sum(float(row["f1"] or 0.0) for row in scored) / max(1, len(scored)) if scored else None,
            "fact_change_accuracy": sum(int(row["fact_change_correct"]) for row in rows) / len(rows),
        })
    by_language: dict[str, dict[str, Any]] = {}
    for language in sorted({str(case["language"]) for case in cases}):
        language_groups = [row for row in group_rows if row["language"] == language]
        scored_groups = [row for row in language_groups if row["em"] is not None]
        by_language[language] = {
            "source_group_count": len({row["source_group_id"] for row in language_groups}),
            "group_em": sum(float(row["em"]) for row in scored_groups) / max(1, len(scored_groups)) if scored_groups else None,
            "group_f1": sum(float(row["f1"]) for row in scored_groups) / max(1, len(scored_groups)) if scored_groups else None,
            "group_fact_change_accuracy": sum(float(row["fact_change_accuracy"]) for row in language_groups) / max(1, len(language_groups)),
            "groups": language_groups,
        }
    return {"by_language": by_language, "source_group_results": group_rows}


def run_long_diagnostics(
    *,
    model: nn.Module,
    tokenizer: Any,
    qa_rows: Sequence[Mapping[str, Any]],
    wiki_rows_by_language: Mapping[str, Sequence[Mapping[str, Any]]],
    max_source_groups: int,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, Any]:
    selected = unique_source_questions(qa_rows, max_source_groups)
    candidates_by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qa_rows:
        if _answer_pair(row):
            candidates_by_language[str(row.get("language", "und"))].append(dict(row))
    pools = {lang: _background_pool(tokenizer, rows) for lang, rows in wiki_rows_by_language.items()}
    selected = [row for row in selected if str(row["language"]) in pools]
    model.eval()
    all_results: list[dict[str, Any]] = []
    skipped_counterfactuals: list[dict[str, str]] = []
    skipped_cases: list[dict[str, str]] = []
    # Bound host RAM by evaluating a small number of independent source groups
    # for each condition/length/position/view combination.
    for target_length in LENGTHS:
        for position in POSITIONS:
            for condition in LONG_CONDITIONS:
                cases: list[dict[str, Any]] = []
                for base in selected:
                    language = str(base["language"])
                    alternative = None
                    if condition == "counterfactual":
                        alternative = _counterfactual_answer(tokenizer, base, candidates_by_language[language])
                        if alternative is None:
                            skipped_counterfactuals.append({
                                "source_group_id": str(base["source_group_id"]),
                                "question_id": str(base["question_id"]),
                                "language": language,
                                "reason": "no answer with matching tokenizer length and numeric type",
                            })
                            continue
                    case = build_long_case(
                        tokenizer, base, pools[language], target_length=target_length,
                        position=position, condition=condition,
                        seed=_stable_int(f"{base['question_id']}:{target_length}:{position}"),
                        alternative_answer=alternative,
                    )
                    if case is None:
                        skipped_cases.append({
                            "source_group_id": str(base["source_group_id"]),
                            "question_id": str(base["question_id"]),
                            "language": language,
                            "condition": condition,
                            "length": str(target_length),
                            "position": position,
                            "reason": "could not construct exact-length context with answer span intact",
                        })
                        continue
                    cases.append(case)
                if not cases:
                    continue
                for first in range(0, len(cases), max(1, batch_size)):
                    case_batch = cases[first:first + batch_size]
                    for view in ("first512", "sliding512", "full"):
                        eval_rows = [
                            row
                            for case in case_batch
                            for row in _window_view_rows(tokenizer, case, view)
                        ]
                        evaluated = evaluate_qa(model, eval_rows, device=device, batch_size=batch_size)
                        by_id = {str(row["id"]): row for row in evaluated["predictions"]}
                        for case in case_batch:
                            prediction = by_id.get(str(case["question_id"]))
                            if prediction is None:
                                skipped_cases.append({
                                    "source_group_id": str(case["source_group_id"]),
                                    "question_id": str(case["source_question_id"]),
                                    "language": str(case["language"]),
                                    "condition": str(case["condition"]),
                                    "length": str(case["target_length"]),
                                    "position": str(case["evidence_position"]),
                                    "reason": f"evaluate_qa returned no prediction for {view}",
                                })
                                continue
                            all_results.append(_case_metrics(case, prediction, view))
    aggregate = _aggregate_cases(all_results)
    return {
        "schema_version": 1,
        "diagnostic": "controlled_long_context_evidence",
        "interpretation": "Wikipedia background plus real QA evidence; controlled synthetic documents, not natural long-context QA",
        "input_lengths": list(LENGTHS),
        "evidence_positions": list(POSITIONS),
        "conditions": list(LONG_CONDITIONS),
        "views": ["first512", "sliding512", "full"],
        "source_group_unit": "QA article title, language-stratified; one deterministic question per article",
        "window_selection": "model span score only; evaluate_qa aggregates every sliding window without gold-based selection",
        "rows_evaluated": len(all_results),
        "by_language": aggregate["by_language"],
        "source_group_results": aggregate["source_group_results"],
        "predictions": all_results,
        "skipped_counterfactuals": skipped_counterfactuals,
        "skipped_cases": skipped_cases,
        "scoring_status": "official SQuAD/CMRC scoring via qa_metrics.answer_em/answer_f1 in evaluate_nextbert",
    }


def freeze_probe_parameters(model: nn.Module, task: str) -> dict[str, Any]:
    if task not in PROBE_TASKS:
        raise ValueError(f"task must be one of {PROBE_TASKS}")
    target_prefix = "classifier." if task == "classification" else "qa_outputs."
    trainable = []
    frozen_backbone = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(target_prefix))
        if not parameter.requires_grad:
            parameter.grad = None
        if parameter.requires_grad:
            trainable.append(name)
        elif name.startswith(("model.",)) or ".dual_gate_logits" in name:
            frozen_backbone += parameter.numel()
    if not trainable:
        raise ValueError(f"model has no {task} head parameters under prefix {target_prefix!r}")
    if any("dual_gate_logits" in name for name in trainable):
        raise AssertionError("attention gates must remain frozen during linear probes")
    return {
        "task": task,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen_backbone_parameter_count": frozen_backbone,
    }


def _reset_probe_head(model: nn.Module, task: str, seed: int) -> None:
    module = model.classifier if task == "classification" else model.qa_outputs
    generator = torch.Generator(device=module.weight.device)
    generator.manual_seed(seed)
    nn.init.normal_(module.weight, mean=0.0, std=float(model.config.initializer_range), generator=generator)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _probe_loss(model: nn.Module, task: str, rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    ids, attention = collate_sequences([row["input_ids"] for row in rows], pad_token_id=0, device=device)
    if task == "classification":
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device)
        valid = attention.bool().clone()
        for special in model.special_token_ids:
            valid &= ids.ne(int(special))
        output = model.forward_classification(ids, attention, labels=labels, content_mask=valid)
        loss = output["loss"]
    else:
        starts = torch.tensor([int(row["start_positions"]) for row in rows], dtype=torch.long, device=device)
        ends = torch.tensor([int(row["end_positions"]) for row in rows], dtype=torch.long, device=device)
        contexts, _ = collate_sequences(
            [row.get("context_mask", [1] * len(row["input_ids"])) for row in rows],
            pad_token_id=0,
            device=device,
        )
        context_mask = contexts.bool() & attention.bool()
        output = model.forward_qa(
            ids, attention, start_positions=starts, end_positions=ends, context_mask=context_mask,
        )
        loss = output["loss"]
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError(f"{task} probe head must return a scalar loss")
    return loss


def _prepare_probe_training_rows(
    task: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Match the original short-task QA probe split used by formal training."""
    if task not in PROBE_TASKS:
        raise ValueError(f"task must be one of {PROBE_TASKS}")
    before = len(rows)
    if task == "qa":
        filtered = [
            row for row in rows
            if len(row.get("input_ids", [])) <= 512 and row.get("window_type") != "long"
        ]
        rule = "input_ids length <= 512 and window_type != 'long'"
    else:
        filtered = list(rows)
        rule = "no filtering"
    return filtered, {
        "applied": task == "qa",
        "rule": rule,
        "rows_before": before,
        "rows_after": len(filtered),
        "rows_removed": before - len(filtered),
    }


def train_frozen_probe(
    *,
    model: nn.Module,
    task: str,
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    steps: int = PROBE_STEPS,
    batch_size: int = QA_BATCH_SIZE,
    seed: int = 17,
) -> dict[str, Any]:
    if steps < 1 or batch_size != QA_BATCH_SIZE:
        raise ValueError("probe requires positive explicit steps and physical batch size 16")
    train_rows_before_filter = len(train_rows)
    train_rows, train_filter = _prepare_probe_training_rows(task, train_rows)
    if task == "qa" and not train_rows:
        raise ValueError("QA frozen probe has no regular training rows with input_ids length <=512")
    if not train_rows or not dev_rows:
        raise ValueError(f"{task} probe requires nonempty train and dev rows")
    _reset_probe_head(model, task, seed)
    freeze_info = freeze_probe_parameters(model, task)
    if hasattr(model, "set_gate_ramp"):
        model.set_gate_ramp(1.0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    warmup_steps = max(1, int(round(steps * PROBE_WARMUP_FRACTION)))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (steps - step) / max(1, steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_rows)).tolist()
    cursor = 0
    losses = []
    # Keep the frozen encoder deterministic: train mode would reactivate its
    # dropout even though none of its weights are being updated.
    model.eval()
    task_head = model.classifier if task == "classification" else model.qa_outputs
    task_head.train()
    if task == "classification" and hasattr(model, "task_dropout"):
        model.task_dropout.train()
    for step in range(steps):
        batch_indices = []
        while len(batch_indices) < batch_size:
            if cursor >= len(order):
                order = rng.permutation(len(train_rows)).tolist()
                cursor = 0
            take = min(batch_size - len(batch_indices), len(order) - cursor)
            batch_indices.extend(order[cursor:cursor + take])
            cursor += take
        batch = [train_rows[index] for index in batch_indices]
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device):
            loss = _probe_loss(model, task, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().float().cpu()))
        if any(
            (not parameter.requires_grad) and parameter.grad is not None
            for parameter in model.parameters()
        ):
            raise AssertionError("frozen encoder or attention-gate parameter received gradients")

    if task == "classification":
        metrics = evaluate_classification(model, dev_rows, device=device, batch_size=batch_size)
    else:
        metrics = evaluate_qa(model, dev_rows, device=device, batch_size=batch_size)
    model.eval()
    return {
        "task": task,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "head_learning_rate": PROBE_LR,
        "weight_decay": PROBE_WEIGHT_DECAY,
        "warmup_fraction": PROBE_WARMUP_FRACTION,
        "mean_training_loss": sum(losses) / len(losses),
        "train_rows_before_filter": train_rows_before_filter,
        "training_filter": train_filter,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        **freeze_info,
        "dev_metrics": metrics,
        "head_state_dict": {key: value.detach().cpu() for key, value in (model.classifier if task == "classification" else model.qa_outputs).state_dict().items()},
    }


def _load_model(model_dir: Path, variant: str, tokenizer: Any, device: torch.device, checkpoint: Path | None) -> nn.Module:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {model_dir}")
    model = NextBert(str(model_dir), variant=variant, tokenizer=tokenizer)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint must contain a model state mapping")
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = ("classifier.", "qa_outputs.", "retrieval_projection.", "reranker.")
        bad_missing = [name for name in missing if not name.startswith(allowed)]
        if bad_missing or unexpected:
            raise ValueError(f"checkpoint backbone mismatch: missing={bad_missing[:5]}, unexpected={unexpected[:5]}")
    return model.to(device)


def _profile_paths(profile: str) -> tuple[Path, Path]:
    prepared = ROOT / "data" / "prepared" / ("full" if profile == "full" else "")
    if profile != "full":
        prepared = ROOT / "data" / "prepared"
    raw_wiki = ROOT / "artifacts" / "raw" / profile / "wikipedia"
    return prepared, raw_wiki


def run_long_command(args: argparse.Namespace) -> None:
    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for exact context offsets")
    model = _load_model(args.model_dir, args.variant, tokenizer, device, args.checkpoint)
    result = run_long_diagnostics(
        model=model,
        tokenizer=tokenizer,
        qa_rows=_jsonl(args.qa_data),
        wiki_rows_by_language={
            language: load_heldout_backgrounds(args.wiki_background_dir, language)
            for language in ("en", "zh")
        },
        max_source_groups=args.max_source_groups,
        device=device,
        batch_size=args.eval_batch_size,
    )
    result["variant"] = args.variant
    result["model_dir"] = str(args.model_dir.resolve())
    result["checkpoint"] = str(args.checkpoint.resolve()) if args.checkpoint else None
    result["max_source_groups_per_language"] = args.max_source_groups
    result["device"] = str(device)
    _write_json(args.output, result)
    predictions_path = args.predictions_output or args.output.with_name(args.output.stem + "_predictions.jsonl")
    _write_jsonl(predictions_path, result["predictions"])
    print(json.dumps({"output": str(args.output), "predictions": str(predictions_path), "rows": result["rows_evaluated"]}))


def run_probe_command(args: argparse.Namespace) -> None:
    if args.steps != PROBE_STEPS:
        raise ValueError(f"the frozen probe protocol uses exactly --steps {PROBE_STEPS}")
    device = _device(args.device)
    _seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, use_fast=True)
    model = _load_model(args.model_dir, args.variant, tokenizer, device, args.checkpoint)
    all_results = {}
    head_states = {}
    for task in args.probe_tasks:
        train_rows = _jsonl(args.classification_train if task == "classification" else args.qa_train)
        dev_rows = _jsonl(args.classification_dev if task == "classification" else args.qa_dev)
        result = train_frozen_probe(
            model=model,
            task=task,
            train_rows=train_rows,
            dev_rows=dev_rows,
            device=device,
            steps=args.steps,
            batch_size=QA_BATCH_SIZE,
            seed=args.seed,
        )
        head_states[task] = result.pop("head_state_dict")
        all_results[task] = result
    payload = {
        "schema_version": 1,
        "diagnostic": "frozen_head_retention_probe",
        "variant": args.variant,
        "model_dir": str(args.model_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "backbone_frozen": True,
        "attention_gates_frozen": True,
        "steps": args.steps,
        "batch_size": QA_BATCH_SIZE,
        "seed": args.seed,
        "tasks": all_results,
    }
    _write_json(args.output, payload)
    head_path = args.output.with_name(args.output.stem + "_heads.pt")
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": args.variant, "seed": args.seed, "heads": head_states}, head_path)
    print(json.dumps({"output": str(args.output), "heads": str(head_path), "tasks": list(all_results)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("long", "probe"))
    parser.add_argument("--model-dir", required=True, type=Path, help="Local pretrained or adapted model directory")
    parser.add_argument("--variant", required=True, choices=("V0", "V1", "V2", "V3"))
    parser.add_argument("--checkpoint", type=Path, help="Optional local trainer checkpoint state")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "artifacts" / "mmbert-base")
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path)

    parser.add_argument("--qa-data", type=Path, help="Prepared QA evaluation JSONL with raw question/context/answers")
    parser.add_argument("--wiki-background-dir", type=Path, help="Directory containing selected_{en,zh}.jsonl")
    parser.add_argument("--max-source-groups", type=int, help="Explicit per-language source group cap")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--predictions-output", type=Path)

    parser.add_argument("--probe-tasks", default="classification,qa", help="Comma-separated classification,qa")
    parser.add_argument("--classification-train", type=Path)
    parser.add_argument("--classification-dev", type=Path)
    parser.add_argument("--qa-train", type=Path)
    parser.add_argument("--qa-dev", type=Path)
    parser.add_argument("--steps", type=int, help="Frozen probe steps; explicitly pass 100")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    prepared_default, wiki_default = _profile_paths(args.profile)
    if args.mode == "long":
        if args.qa_data is None:
            args.qa_data = prepared_default / "qa_test.jsonl"
        if args.wiki_background_dir is None:
            args.wiki_background_dir = wiki_default
        if args.max_source_groups is None or args.max_source_groups < 1:
            parser.error("--mode long requires positive --max-source-groups to bound runtime")
        if args.eval_batch_size < 1:
            parser.error("--eval-batch-size must be positive")
        if args.output is None:
            args.output = ROOT / "results" / "diagnostics" / "long_context.json"
        run_long_command(args)
    else:
        if args.steps is None:
            parser.error("--mode probe requires explicit --steps 100")
        task_names = [name.strip() for name in args.probe_tasks.split(",") if name.strip()]
        if not task_names or any(name not in PROBE_TASKS for name in task_names):
            parser.error(f"--probe-tasks values must be from {PROBE_TASKS}")
        args.probe_tasks = task_names
        for task, attribute in (
            ("classification", "classification_train"), ("classification", "classification_dev"),
            ("qa", "qa_train"), ("qa", "qa_dev"),
        ):
            if task in args.probe_tasks and getattr(args, attribute) is None:
                setattr(args, attribute, prepared_default / f"{task}_{'dev' if attribute.endswith('dev') else 'train'}.jsonl")
        if args.output is None:
            args.output = ROOT / "results" / "diagnostics" / "frozen_probe.json"
        run_probe_command(args)


if __name__ == "__main__":
    main()
