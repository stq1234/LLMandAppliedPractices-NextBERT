"""Focused CPU tests for NextBERT training primitives and resume semantics."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
import copy
from unittest.mock import patch
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_nextbert import (  # noqa: E402
    _load_checkpoint,
    _save_checkpoint,
    _scheduler,
    _epoch_sample_indices,
    adaptation_stage_schedule,
    _encode_retrieval,
    _initialize_backbone_from_checkpoint,
    optimizer_for,
    choose_retrieval_physical_batch,
    corrupt_mlm,
    retrieval_logical_batch,
    retrieval_to_reranking_records,
    pair_tokens,
    qa_phase_steps,
    read_plan,
    collate_sequences,
    run_training,
    train_mlm_step,
    train_retrieval_step,
    train_supervised_step,
    validate_mlm_split_sidecar,
)
from evaluate_nextbert import (  # noqa: E402
    _best_span,
    _candidate_documents,
    _load_checkpoint as _load_evaluation_weights,
    evaluate_qa,
    evaluate_reranking,
    evaluate_retrieval,
)
from qa_metrics import answer_em, answer_f1  # noqa: E402
from nextbert_lab import make_tiny_config  # noqa: E402
from train_nextbert import load_nextbert  # noqa: E402
import run_pilot  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_training_matrix  # noqa: E402


class TinyNextBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.special_token_ids = (0, 1, 2, 4)
        self.embedding = nn.Embedding(64, 12, padding_idx=0)
        self.dropout = nn.Dropout(0.15)
        self.mlm_head = nn.Linear(12, 64)
        self.classification_head = nn.Linear(12, 3)
        self.qa_head = nn.Linear(12, 2)
        self.retrieval_projection = nn.Linear(12, 6)
        self.reranking_head = nn.Linear(12, 1)
        self.dual_gate_logits = nn.Parameter(torch.zeros(2))
        self.gate_ramp = 1.0

    def set_gate_ramp(self, value: float) -> None:
        self.gate_ramp = value

    def _hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(input_ids)) * attention_mask.unsqueeze(-1)

    def forward_mlm(self, input_ids, attention_mask=None, labels=None, return_logits=False):
        hidden = self._hidden(input_ids, attention_mask)
        logits = self.mlm_head(hidden)
        supervised = labels.ne(-100)
        loss = F.cross_entropy(logits[supervised].float(), labels[supervised])
        return {"loss": loss, "logits": logits[supervised] if return_logits else None}

    def forward_classification(self, input_ids, attention_mask=None, content_mask=None):
        mask = content_mask if content_mask is not None else attention_mask.bool()
        hidden = self._hidden(input_ids, attention_mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return {"logits": self.classification_head(pooled)}

    def forward_qa(self, input_ids, attention_mask=None, context_mask=None):
        return {"logits": self.qa_head(self._hidden(input_ids, attention_mask))}

    def encode_retrieval(self, input_ids, attention_mask=None, content_mask=None):
        mask = content_mask if content_mask is not None else attention_mask.bool()
        hidden = self._hidden(input_ids, attention_mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(self.retrieval_projection(pooled), dim=-1)

    def forward_reranking(self, input_ids, attention_mask=None, labels=None, group_size=None):
        hidden = self._hidden(input_ids, attention_mask)
        pooled = hidden.sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1)
        return {"logits": self.reranking_head(pooled).squeeze(-1)}


class TrainingPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261110)
        self.device = torch.device("cpu")

    def test_mlm_corruption_excludes_special_and_padding_positions(self) -> None:
        ids = torch.tensor([[2, 8, 9, 10, 11, 1, 0], [2, 12, 13, 14, 15, 1, 0]])
        mask = ids.ne(0).long()
        corrupted, labels, selected = corrupt_mlm(
            ids, mask, vocab_size=64, special_ids=(0, 1, 2, 4), mask_token_id=4,
            seed=31, mode="span",
        )
        forbidden = torch.tensor([0, 1, 2, 4])
        for special in forbidden:
            self.assertFalse(selected[ids.eq(special)].any().item())
            self.assertTrue(torch.equal(corrupted[ids.eq(special)], ids[ids.eq(special)]))
            self.assertTrue(torch.all(labels[ids.eq(special)] == -100).item())
        self.assertTrue(torch.all(labels[ids.eq(0)] == -100).item())
        self.assertTrue(torch.all(selected.sum(dim=1) > 0).item())

    def test_long_span_corruption_is_exact_bounded_and_seed_reproducible(self) -> None:
        ids = torch.full((1, 8192), 9, dtype=torch.long)
        ids[0, 0], ids[0, -1] = 2, 1
        ids[0, 2048], ids[0, 4096], ids[0, 6144] = 4, 2, 1
        mask = torch.ones_like(ids)
        kwargs = {"vocab_size": 64, "special_ids": (0, 1, 2, 4), "mask_token_id": 4,
                  "seed": 20261111, "mode": "span", "mask_rate": 0.30,
                  "span_mean": 3.0, "span_max": 10}
        first = corrupt_mlm(ids, mask, **kwargs)
        second = corrupt_mlm(ids, mask, **kwargs)
        eligible = mask.bool()
        for special in kwargs["special_ids"]:
            eligible &= ids.ne(special)
        target_count = max(1, int(round(int(eligible.sum()) * kwargs["mask_rate"])))
        selected = first[2]
        self.assertEqual(int(selected.sum()), target_count)
        self.assertFalse((selected & ~eligible).any().item())
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_mlm_step_updates_parameters_and_logs_tokens(self) -> None:
        model = TinyNextBert()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        ids = torch.tensor([[2, 7, 8, 9, 1], [2, 10, 11, 12, 1]])
        mask = ids.ne(0).long()
        watched = model.embedding.weight.detach().clone()
        result = train_mlm_step(
            model, optimizer, [(ids, mask), (ids, mask)], device=self.device,
            vocab_size=64, special_ids=model.special_token_ids, mask_token_id=4,
            seed=72, accumulation=2, gate_ramp_updates=64, optimizer_step=0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)).item())
        self.assertEqual(result.nonpad_tokens, int(mask.sum().item()) * 2)
        self.assertFalse(torch.equal(watched, model.embedding.weight.detach()))
        self.assertIn("dual_gate_logits", " ".join(result.gate_values))

    def test_qa_context_only_batch_trains(self) -> None:
        model = TinyNextBert()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": [[2, 8, 9, 10, 1, 0]],
            "start_positions": [2], "end_positions": [3],
            "context_mask": [[0, 0, 1, 1, 0, 0]], "nonpad_tokens": 5,
        }
        result = train_supervised_step(model, optimizer, "qa", [batch],
                                       device=self.device, step=0, accumulation=1)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)).item())
        self.assertEqual(result.nonpad_tokens, 5)

    def test_retrieval_uses_one_logical_batch_and_replays_gradients(self) -> None:
        model = TinyNextBert()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        rows = [
            {"query_ids": [2, 10 + i, 1], "positive_ids": [2, 20 + i, 1],
             "negative_ids": [2, 40 + i, 1]} for i in range(4)
        ]
        rows[1]["negative_ids"] = rows[0]["positive_ids"]
        logical = retrieval_logical_batch(rows, device=self.device)
        self.assertEqual(tuple(logical["valid_negative"].shape), (4, 8))
        # Negatives occupy columns B..2B-1; row 1's explicit negative is
        # column 5 and is a known positive for row 0.
        self.assertFalse(logical["valid_negative"][1, 5].item())
        self.assertTrue(logical["valid_negative"][1, 1].item())
        watched = model.embedding.weight.detach().clone()
        result = train_retrieval_step(model, optimizer, rows, device=self.device,
                                      step=0, physical_document_batch=2)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)).item())
        self.assertEqual(result.nonpad_tokens, 4 * 3 + 8 * 3)
        self.assertFalse(torch.equal(watched, model.embedding.weight.detach()))

    def test_gradient_cache_matches_direct_logical_batch_gradient_with_rng_replay(self) -> None:
        rows = [
            {"query_ids": [2, 10 + i, 11 + i, 1],
             "positive_ids": [2, 20 + i, 21 + i, 1],
             "negative_ids": [2, 40 + i, 41 + i, 1]} for i in range(4)
        ]
        torch.manual_seed(775)
        base = TinyNextBert()
        direct, cached = copy.deepcopy(base), copy.deepcopy(base)
        direct_optimizer = AdamW(direct.parameters(), lr=2e-3)
        cache_optimizer = AdamW(cached.parameters(), lr=2e-3)
        logical = retrieval_logical_batch(rows, device=self.device)

        torch.manual_seed(991)
        direct_optimizer.zero_grad(set_to_none=True)
        query_vectors = []
        for start in range(0, 4, 2):
            ids, mask = collate_sequences(
                [row["query_ids"] for row in rows[start:start + 2]],
                pad_token_id=0, device=self.device)
            query_vectors.append(_encode_retrieval(direct, ids, mask))
        doc_sequences = [row["positive_ids"] for row in rows] + [row["negative_ids"] for row in rows]
        document_vectors = []
        for start in range(0, 8, 2):
            ids, mask = collate_sequences(doc_sequences[start:start + 2], pad_token_id=0, device=self.device)
            document_vectors.append(_encode_retrieval(direct, ids, mask))
        q = torch.cat(query_vectors)
        d = torch.cat(document_vectors)
        direct_logits = q.float() @ d.float().T / 0.05
        direct_logits = direct_logits.masked_fill(
            ~logical["valid_negative"], torch.finfo(direct_logits.dtype).min)
        direct_loss = F.cross_entropy(direct_logits, logical["positive_columns"])
        direct_loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
        direct_optimizer.step()

        torch.manual_seed(991)
        cached_result = train_retrieval_step(
            cached, cache_optimizer, rows, device=self.device, step=0,
            physical_document_batch=2, physical_query_batch=2, temperature=0.05)
        self.assertAlmostEqual(cached_result.loss, float(direct_loss.detach()), places=6)
        for (direct_name, direct_parameter), (cache_name, cache_parameter) in zip(
            direct.named_parameters(), cached.named_parameters()
        ):
            self.assertEqual(direct_name, cache_name)
            self.assertTrue(torch.allclose(direct_parameter, cache_parameter, atol=2e-6, rtol=1e-5),
                            msg=f"gradient-cache update differs for {direct_name}")

    def test_logical64_gradient_cache_replays_query_chunks_of32_and_matches_direct_update(self) -> None:
        rows = [
            {"query_ids": [2, 3 + (i % 58), 1],
             "positive_ids": [2, 3 + ((i * 3 + 7) % 58), 1],
             "negative_ids": [2, 3 + ((i * 7 + 19) % 58), 1]}
            for i in range(64)
        ]
        torch.manual_seed(614)
        base = TinyNextBert()
        direct, cached = copy.deepcopy(base), copy.deepcopy(base)
        direct_optimizer = AdamW(direct.parameters(), lr=1e-3)
        cache_optimizer = AdamW(cached.parameters(), lr=1e-3)
        logical = retrieval_logical_batch(rows, device=self.device)

        torch.manual_seed(901)
        direct_optimizer.zero_grad(set_to_none=True)
        query_vectors = []
        for start in range(0, 64, 32):
            ids, mask = collate_sequences(
                [row["query_ids"] for row in rows[start:start + 32]],
                                          pad_token_id=0, device=self.device)
            query_vectors.append(_encode_retrieval(direct, ids, mask))
        doc_sequences = [row["positive_ids"] for row in rows] + [row["negative_ids"] for row in rows]
        document_vectors = []
        for start in range(0, len(doc_sequences), 4):
            ids, mask = collate_sequences(doc_sequences[start:start + 4], pad_token_id=0, device=self.device)
            document_vectors.append(_encode_retrieval(direct, ids, mask))
        q, d = torch.cat(query_vectors), torch.cat(document_vectors)
        logits = q.float() @ d.float().T / 0.05
        logits = logits.masked_fill(~logical["valid_negative"], torch.finfo(logits.dtype).min)
        loss = F.cross_entropy(logits, logical["positive_columns"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
        direct_optimizer.step()

        call_sizes: list[int] = []
        original_encode = cached.encode_retrieval
        def spy_encode(*, input_ids, attention_mask=None, content_mask=None):
            call_sizes.append(int(input_ids.shape[0]))
            return original_encode(input_ids=input_ids, attention_mask=attention_mask,
                                   content_mask=content_mask)

        cached.encode_retrieval = spy_encode
        torch.manual_seed(901)
        cache_result = train_retrieval_step(
            cached, cache_optimizer, rows, device=self.device, step=0,
            physical_query_batch=32, physical_document_batch=4, temperature=0.05)
        self.assertEqual(tuple(logical["valid_negative"].shape), (64, 128))
        self.assertEqual(call_sizes[:2], [32, 32])
        self.assertEqual(call_sizes[2:34], [4] * 32)
        self.assertAlmostEqual(cache_result.loss, float(loss.detach()), places=6)
        for (direct_name, direct_parameter), (cache_name, cache_parameter) in zip(
            direct.named_parameters(), cached.named_parameters()
        ):
            self.assertEqual(direct_name, cache_name)
            self.assertTrue(torch.allclose(direct_parameter, cache_parameter, atol=2e-6, rtol=1e-5),
                            msg=f"logical64 gradient-cache update differs for {direct_name}")

    def test_new_plan_drives_mlm_stage_offsets_and_downstream_batches(self) -> None:
        plan = read_plan()
        schedule = adaptation_stage_schedule(plan)
        self.assertEqual([(row["max_length"], row["start_step"], row["end_step"])
                          for row in schedule],
                         [(512, 0, 192), (1024, 192, 384), (4096, 384, 448), (8192, 448, 512)])
        self.assertEqual(qa_phase_steps(plan), {"regular": 625, "long": 63})
        self.assertEqual(run_pilot._task_batch_signature("classification", "all", plan),
                         {"physical_batch": 32, "gradient_accumulation": 2, "effective_batch": 64})
        self.assertEqual(run_pilot._task_batch_signature("qa", "regular", plan)["effective_batch"], 32)
        self.assertEqual(run_pilot._task_batch_signature("qa", "long", plan),
                         {"physical_batch": 8, "gradient_accumulation": 4, "effective_batch": 32})
        self.assertEqual(run_pilot._task_batch_signature("retrieval", "all", plan),
                         {"physical_batch": 32, "gradient_accumulation": 1,
                          "effective_batch": 64, "logical_retrieval_queries": 64,
                          "physical_query_batch": 32, "physical_document_batch": 4})
        self.assertEqual(run_pilot._task_batch_signature("reranking", "all", plan),
                         {"physical_batch": 4, "gradient_accumulation": 2,
                          "effective_batch": 8, "physical_pairs": 12, "effective_pairs": 24})
        planned_qa = run_pilot._planned_task_steps(
            "qa", {"qa_rows": [
                {"question_id": "r0", "input_ids": [1, 2]},
                {"question_id": "r0", "input_ids": [1, 2]},
                {"question_id": "r1", "input_ids": [1, 2]},
                {"question_id": "l0", "input_ids": [1] * 513},
            ]}, plan)
        self.assertEqual(planned_qa["regular"]["planned_optimizer_steps"], 625)
        self.assertEqual(planned_qa["regular"]["target_count_per_epoch"], 10000)
        self.assertEqual(planned_qa["regular"]["source_count"], 2)
        self.assertEqual(planned_qa["long"]["planned_optimizer_steps"], 63)
        self.assertEqual(planned_qa["long"]["target_count_per_epoch"], 1000)
        self.assertEqual(planned_qa["long"]["source_count"], 1)

    def test_v3_balanced_projection_averages_matching_token_and_span_medians(self) -> None:
        plan = read_plan()
        cases = []
        for mode, times in (("token", [2.0, 4.0, 6.0]), ("span", [4.0, 6.0, 8.0])):
            cases.append({
                "variant": "V3", "length": 4096, "mode": mode,
                "physical_microbatch": 8, "gradient_accumulation": 4,
                "effective_sequences": 32,
                "steps": [{"pilot_role": "measured", "mlm_mode": mode,
                           "optimizer_step_seconds": value} for value in times],
            })
        projections = run_pilot._adaptation_projection(cases, plan)
        row = next(item for item in projections
                   if item["variant"] == "V3" and item["length"] == 4096)
        self.assertEqual(row["objective_mode"], "balanced_token_span_50_50")
        self.assertAlmostEqual(row["measured_pilot_step_seconds_median"], 5.0)
        self.assertEqual(row["measured_pilot_sample_count"], 3)
        self.assertEqual(row["measured_mode_sample_counts"], {"token": 3, "span": 3})
        self.assertEqual(row["projected_seconds"], 320.0)

        cases[0]["physical_microbatch"] = 4
        rejected = next(item for item in run_pilot._adaptation_projection(cases, plan)
                        if item["variant"] == "V3" and item["length"] == 4096)
        self.assertEqual(rejected["projection_status"], "unmeasured_batch_mismatch")
        self.assertIsNone(rejected["projected_seconds"])

    def test_training_matrix_uses_dynamic_absolute_mlm_targets_and_profile(self) -> None:
        plan = read_plan()
        commands = run_training_matrix.build_commands(
            Path("model"), None, Path("out"), plan)
        mlm = [cmd for cmd in commands if cmd[cmd.index("--task") + 1] == "mlm"]
        self.assertEqual(len(mlm), 14)
        self.assertTrue(all("--final-weights-only" not in cmd for cmd in mlm))
        v3 = [cmd for cmd in mlm if cmd[cmd.index("--variant") + 1] == "V3"]
        self.assertEqual([cmd[cmd.index("--steps") + 1] for cmd in v3], ["448", "512"])
        self.assertTrue(all("balanced" in cmd for cmd in v3))
        self.assertIn("--shared-prefix-resume", v3[0])
        self.assertIn("--resume-from", v3[0])
        self.assertEqual(run_training_matrix._default_data_dir(plan),
                         run_training_matrix.ROOT / "data" / "prepared" / "full_67m")
        prefix = Path("out/mlm/V2/1024/V2_mlm_checkpoint.pt")
        self.assertFalse(run_training_matrix._should_delete_resumed_checkpoint(
            "V2", prefix, prefix))
        self.assertTrue(run_training_matrix._should_delete_resumed_checkpoint(
            "V3", prefix, prefix))
        qa_commands = [cmd for cmd in commands if cmd[cmd.index("--task") + 1] == "qa"]
        self.assertTrue(qa_commands)
        self.assertTrue(all(cmd[cmd.index("--steps") + 1] == "688" for cmd in qa_commands))
        downstream_commands = [cmd for cmd in commands if cmd[cmd.index("--task") + 1] != "mlm"]
        self.assertTrue(all("--final-weights-only" in cmd for cmd in downstream_commands))

    def test_checkpoint_restores_model_optimizer_and_scheduler(self) -> None:
        model = TinyNextBert()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = _scheduler(optimizer, 8, 0.05)
        parameter = model.embedding.weight
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(0.5)
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            _save_checkpoint(checkpoint, model=model, optimizer=optimizer,
                             scheduler=scheduler, step=5, task="mlm", variant="V2", seed=17)
            with torch.no_grad():
                parameter.zero_()
            step = _load_checkpoint(checkpoint, model=model, optimizer=optimizer,
                                    scheduler=scheduler, task="mlm", variant="V3",
                                    map_location=self.device, allow_shared_variant_alias=True)
        self.assertEqual(step, 5)
        self.assertTrue(torch.equal(parameter.detach(), original + 0.5))

    def test_final_weights_only_checkpoint_initializes_and_evaluates_but_cannot_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "classification.jsonl"
            data.write_text("\n".join(json.dumps({
                "input_ids": [2, 8 + i, 1], "label": i % 3,
            }) for i in range(4)) + "\n", encoding="utf-8")

            final_model = TinyNextBert()
            final_result = run_training(
                model=final_model, task="classification", variant="V0", data_path=data,
                steps=1, output_dir=root / "final", device=self.device, seed=17,
                final_weights_only=True,
            )
            final_path = Path(final_result["checkpoint"])
            final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
            self.assertEqual(final_result["checkpoint_kind"], "weights_only")
            self.assertEqual(final_payload["checkpoint_kind"], "weights_only")
            self.assertNotIn("optimizer", final_payload)
            self.assertNotIn("python_rng", final_payload)

            evaluated = TinyNextBert()
            _load_evaluation_weights(evaluated, final_path, self.device)
            self.assertTrue(torch.equal(evaluated.embedding.weight, final_model.embedding.weight))
            newly_initialized = TinyNextBert()
            fresh_head = newly_initialized.classification_head.weight.detach().clone()
            _initialize_backbone_from_checkpoint(
                newly_initialized, final_path, self.device, variant="V0")
            self.assertTrue(torch.equal(newly_initialized.embedding.weight, final_model.embedding.weight))
            self.assertTrue(torch.equal(newly_initialized.classification_head.weight, fresh_head))

            optimizer = AdamW(evaluated.parameters(), lr=1e-3)
            scheduler = _scheduler(optimizer, 2, 0.05)
            with self.assertRaisesRegex(ValueError, "weights-only checkpoint cannot resume"):
                _load_checkpoint(final_path, model=evaluated, optimizer=optimizer,
                                 scheduler=scheduler, task="classification", variant="V0",
                                 map_location=self.device)

            partial_result = run_training(
                model=TinyNextBert(), task="classification", variant="V0", data_path=data,
                steps=2, output_dir=root / "partial", device=self.device, seed=17,
                stop_after_steps=1, final_weights_only=True,
            )
            partial_path = Path(partial_result["checkpoint"])
            partial_payload = torch.load(partial_path, map_location="cpu", weights_only=False)
            self.assertEqual(partial_result["checkpoint_kind"], "full_state")
            self.assertEqual(partial_payload["checkpoint_kind"], "full_state")
            self.assertIn("optimizer", partial_payload)
            resumed_result = run_training(
                model=TinyNextBert(), task="classification", variant="V0", data_path=data,
                steps=2, output_dir=root / "partial", device=self.device, seed=17,
                resume_from=partial_path, final_weights_only=True,
            )
            self.assertEqual(resumed_result["last_step"], 2)
            self.assertEqual(resumed_result["checkpoint_kind"], "weights_only")

    def test_epoch_sampler_visits_each_row_once_before_repeating(self) -> None:
        first_epoch = _epoch_sample_indices(23, 23, seed=91, offset=0)
        next_epoch = _epoch_sample_indices(23, 23, seed=91, offset=23)
        self.assertEqual(set(first_epoch.tolist()), set(range(23)))
        self.assertEqual(len(set(next_epoch.tolist())), 23)
        self.assertNotEqual(first_epoch.tolist(), next_epoch.tolist())

    def test_mlm_initialization_preserves_fresh_downstream_head(self) -> None:
        source, fresh = TinyNextBert(), TinyNextBert()
        with torch.no_grad():
            source.embedding.weight.add_(2.0)
            source.classification_head.weight.add_(3.0)
        expected_backbone = source.embedding.weight.detach().clone()
        expected_fresh_head = fresh.classification_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapted.pt"
            torch.save({"model": source.state_dict()}, checkpoint)
            result = _initialize_backbone_from_checkpoint(fresh, checkpoint, self.device, variant="V3")
        self.assertTrue(torch.equal(fresh.embedding.weight.detach(), expected_backbone))
        self.assertTrue(torch.equal(fresh.classification_head.weight.detach(), expected_fresh_head))
        self.assertGreater(result["retained_task_head_tensors"], 0)

    def test_task_head_backbone_and_gate_learning_rates_are_separate(self) -> None:
        model = TinyNextBert()
        optimizer = optimizer_for(model, backbone_lr=2e-5, gate_lr=1e-3,
                                  head_lr=1e-4, weight_decay=0.01)
        rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates["backbone_and_heads"], 2e-5)
        self.assertEqual(rates["task_heads"], 1e-4)
        self.assertEqual(rates["gates"], 1e-3)
        for group in optimizer.param_groups:
            if group["group_name"] == "task_heads":
                self.assertTrue(any(parameter is model.qa_head.weight for parameter in group["params"]))

    def test_mlm_sidecar_rejects_validation_or_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wiki_512.npy"
            path.write_bytes(b"npy-placeholder")
            sidecar = path.with_suffix(".jsonl")
            sidecar.write_text(
                json.dumps({"split": "train"}) + "\n" +
                json.dumps({"split": "validation"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-train rows"):
                validate_mlm_split_sidecar(path, row_count=2)
            sidecar.write_text(json.dumps({"split": "train"}) + "\n" * 2, encoding="utf-8")
            result = validate_mlm_split_sidecar(path, row_count=1)
            self.assertEqual(result["status"], "verified_train_only")

    def test_vectorized_qa_span_decode_ignores_noncontext_logits(self) -> None:
        start = torch.tensor([100.0, 1.0, 5.0, 100.0, 4.0, 0.0])
        end = torch.tensor([100.0, 4.0, 10.0, 100.0, 4.0, 3.0])
        context = torch.tensor([0, 1, 1, 0, 1, 1])
        self.assertEqual(_best_span(start, end, context), (2, 2, 15.0))

    def test_qa_evaluation_aggregates_overflow_by_question(self) -> None:
        model = TinyNextBert().eval()
        rows = [
            {"id": "q1-0", "question_id": "q1", "language": "en",
             "input_ids": [2, 8, 9, 1], "context_mask": [0, 1, 1, 0],
             "start_positions": 1, "end_positions": 1,
             "answers": {"text": ["foo"]}, "context": "foo bar",
             "offset_mapping": [None, [0, 3], [4, 7], None]},
            {"id": "q1-1", "question_id": "q1", "language": "en",
             "input_ids": [2, 10, 11, 1], "context_mask": [0, 1, 1, 0],
             "start_positions": -100, "end_positions": -100,
             "answers": {"text": ["foo"]}, "context": "baz qux",
             "offset_mapping": [None, [0, 3], [4, 7], None]},
        ]
        result = evaluate_qa(model, rows, device=self.device, batch_size=2)
        self.assertEqual(result["overflow_windows"], 2)
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["count"], 1)
        self.assertIn("qa_metrics.answer_em", result["scoring_status"])

    def test_official_english_qa_metrics_are_used(self) -> None:
        self.assertEqual(answer_em("The answer.", "answer", "en"), 1.0)
        self.assertEqual(answer_f1("The answer.", "answer", "en"), 1.0)

    def test_retrieval_index_and_bm25_rerank_inputs_are_evaluable(self) -> None:
        model = TinyNextBert().eval()
        corpus = [{"id": f"d{i}", "language": "en", "input_ids": [2, 20 + i, 1]}
                  for i in range(4)]
        # This matches prepare_data's retrieval_queries_{dev,test}.jsonl schema.
        queries = [{"id": "q1", "language": "en", "input_ids": [2, 30, 1],
                    "relevant_doc_ids": ["d0"]}]
        retrieval = evaluate_retrieval(
            model, corpus, queries, device=self.device,
            sample_documents_per_language=4)
        self.assertEqual(retrieval["languages"][0]["documents_indexed"], 4)
        self.assertGreater(retrieval["languages"][0]["vector_storage_bytes"], 0)
        self.assertIsNotNone(retrieval["projected_5000_per_language_index_seconds"]["en"])

        rerank_rows = [{"id": "q1", "language": "en", "query_ids": [2, 30, 1],
                        "candidate_ids": ["d0", "d1"], "labels": [1, 0],
                        "relevant_doc_ids": ["d0"]}]
        rerank = evaluate_reranking(model, rerank_rows, device=self.device, corpus=corpus)
        self.assertEqual(rerank["query_count"], 1)
        self.assertEqual(rerank["predictions"][0]["candidate_Recall@20"], 1.0)

    def test_prepared_bm25_candidate_token_lists_and_docids_evaluate_with_or_without_corpus(self) -> None:
        model = TinyNextBert().eval()
        row = {
            "id": "q1", "language": "en", "query_ids": [2, 30, 1],
            # Actual prepared schema: token sequences are candidate_ids;
            # doc IDs used for corpus lookup and retrieval metrics are separate.
            "candidate_ids": [[2, 55, 1], [2, 56, 1]],
            "candidate_docids": ["d0", "d1"],
            "labels": [1, 0], "relevant_doc_ids": ["d0"],
        }
        corpus = [
            {"id": "d0", "language": "en", "input_ids": [2, 20, 1]},
            {"id": "d1", "language": "en", "input_ids": [2, 21, 1]},
        ]
        docs_with_corpus, labels, candidate_docids = _candidate_documents(
            row, {doc["id"]: doc for doc in corpus})
        self.assertEqual(docs_with_corpus, [doc["input_ids"] for doc in corpus])
        self.assertEqual(labels, [1, 0])
        self.assertEqual(candidate_docids, ["d0", "d1"])

        docs_without_corpus, _, direct_docids = _candidate_documents(row)
        self.assertEqual(docs_without_corpus, row["candidate_ids"])
        self.assertEqual(direct_docids, ["d0", "d1"])

        with_corpus = evaluate_reranking(model, [row], device=self.device, corpus=corpus)
        without_corpus = evaluate_reranking(model, [row], device=self.device)
        self.assertEqual(with_corpus["predictions"][0]["candidate_count"], 2)
        self.assertEqual(without_corpus["predictions"][0]["candidate_count"], 2)
        self.assertEqual(with_corpus["predictions"][0]["candidate_Recall@20"], 1.0)
        self.assertEqual(without_corpus["predictions"][0]["candidate_Recall@20"], 1.0)

    def test_prepared_retrieval_candidates_form_fixed_reranking_groups(self) -> None:
        row = {
            "id": "q-en-425512",
            "query_ids": [2, 30, 1],
            "positive_ids": [2, 40, 1],
            # The retrieval objective uses one explicit negative token list.
            "negative_ids": [2, 50, 1],
            # prepare_data retains two independent candidate docs for reranking.
            "negative_candidates": [
                {"docid": "d-neg-1", "input_ids": [2, 50, 51, 1]},
                {"docid": "d-neg-2", "input_ids": [2, 60, 61, 1]},
            ],
        }
        groups = retrieval_to_reranking_records([row])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["input_ids"], [
            pair_tokens(row["query_ids"], row["positive_ids"]),
            pair_tokens(row["query_ids"], row["negative_candidates"][0]["input_ids"]),
            pair_tokens(row["query_ids"], row["negative_candidates"][1]["input_ids"]),
        ])

        # Reranking conversion must not alter the single-negative InfoNCE input.
        second = {**row, "id": "q2", "negative_ids": [2, 70, 1]}
        logical = retrieval_logical_batch([row, second], device=self.device)
        self.assertEqual(logical["document_tokens"][2], row["negative_ids"])

    def test_retrieval_document_microbatch_auto_selection_and_override(self) -> None:
        short_rows = [{"positive_ids": [2, 40, 1], "negative_ids": [2, 50, 1]}]
        self.assertEqual(choose_retrieval_physical_batch(short_rows), 4)
        long_rows = [{"positive_ids": [2, *([40] * 4094), 1],
                      "negative_ids": [2, 50, 1]}]
        self.assertEqual(choose_retrieval_physical_batch(long_rows), 2)
        self.assertEqual(choose_retrieval_physical_batch(long_rows, override=8), 8)
        self.assertEqual(choose_retrieval_physical_batch(short_rows, override=2), 2)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            choose_retrieval_physical_batch(short_rows, override=1)

    def test_pair_tokens_matches_mmbert_native_wrapped_pair_boundaries(self) -> None:
        self.assertEqual(pair_tokens([2, 10, 11, 1], [2, 20, 21, 1]),
                         [2, 10, 11, 1, 20, 21, 1])
        self.assertEqual(pair_tokens([10, 11], [20, 21]), [10, 11, 1, 20, 21])

    def test_pilot_evaluation_only_skips_training_and_does_not_claim_training_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "evaluation_only"
            argv = [
                "run_pilot.py", "--evaluation-only",
                "--model-dir", f"V0={root / 'model'}",
                "--model-dir", f"V3={root / 'model'}",
                "--variants", "V0", "V3",
                "--evaluation-corpus", str(root / "corpus.jsonl"),
                "--evaluation-retrieval-queries", str(root / "queries.jsonl"),
                "--evaluation-reranking-data", str(root / "reranking.jsonl"),
                "--output-dir", str(output_dir),
            ]
            probe_case = {"task": "retrieval_index_probe", "variant": "V0"}
            projection = {
                "is_partial": True,
                "projected_index_seconds_for_four_variants": None,
                "projected_retrieval_query_encode_seconds_for_four_variants": None,
                "projected_reranking_seconds": None,
            }
            with patch.object(sys, "argv", argv), \
                    patch.object(run_pilot, "_device", return_value=self.device), \
                    patch.object(run_pilot, "_device_memory_context", return_value={"test": True}), \
                    patch.object(run_pilot, "_run_evaluation_probes", return_value=[probe_case]) as evaluate, \
                    patch.object(run_pilot, "_evaluation_cost_projection", return_value=projection), \
                    patch.object(run_pilot, "_mlm_case", side_effect=AssertionError("training probe ran")), \
                    patch.object(run_pilot, "_task_case", side_effect=AssertionError("training probe ran")), \
                    patch.object(run_pilot, "build_time_projection", side_effect=AssertionError("training projection ran")):
                run_pilot.main()
            evaluate.assert_called_once()
            report = json.loads((output_dir / "pilot_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "evaluation_only_completed")
            self.assertTrue(report["configuration"]["evaluation_only"])
            self.assertEqual(report["mlm_calibration_cases"], [])
            self.assertEqual(report["downstream_calibration_cases"], [])
            self.assertEqual(report["time_projection"]["training_calibration_status"],
                             "not_run_evaluation_only")

    def test_reranking_conversion_accepts_legacy_nested_negatives_and_rejects_invalid_groups(self) -> None:
        base = {"id": "q1", "query_ids": [2, 30, 1], "positive_ids": [2, 40, 1]}
        legacy = {**base, "negative_ids": [[2, 50, 1], [2, 60, 1]]}
        self.assertEqual(len(retrieval_to_reranking_records([legacy])[0]["input_ids"]), 3)

        insufficient = {**base, "negative_ids": [2, 50, 1]}
        with self.assertRaisesRegex(ValueError, "two explicit reranking negatives"):
            retrieval_to_reranking_records([insufficient])

        duplicate = {**base, "negative_candidates": [
            {"docid": "d1", "input_ids": [2, 50, 1]},
            {"docid": "d2", "input_ids": [2, 50, 1]},
        ]}
        with self.assertRaisesRegex(ValueError, "one positive and two distinct negatives"):
            retrieval_to_reranking_records([duplicate])

    def test_local_tiny_checkpoint_loads_through_offline_runner(self) -> None:
        from modeling_nextbert import NextBert

        config = make_tiny_config(layers=2, hidden=16, heads=2, intermediate=24,
                                  local_window=8, global_every=3, dropout=0.0)
        source = NextBert(None, "V0", attention_backend="sdpa", config=config,
                          special_token_ids=(0, 1, 2, 4), retrieval_dim=8,
                          query_chunk_size=4, mlm_chunk_size=4)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            loaded = load_nextbert(Path(directory), "V0", self.device, "sdpa")
            self.assertTrue(loaded.model.gradient_checkpointing)
            ids = torch.tensor([[2, 7, 8, 1]])
            mask = ids.ne(0).long()
            labels = torch.tensor([[-100, 7, -100, -100]])
            mlm = loaded.forward_mlm(ids, attention_mask=mask, labels=labels)
            classifier = loaded.forward_classification(ids, attention_mask=mask)
            qa = loaded.forward_qa(ids, attention_mask=mask,
                                   context_mask=torch.tensor([[0, 1, 1, 0]]))
            retrieval = loaded.encode_retrieval(ids, attention_mask=mask)
            rerank = loaded.forward_reranking(ids, attention_mask=mask, group_size=3)
        self.assertTrue(torch.isfinite(mlm["loss"]).item())
        self.assertEqual(tuple(classifier["logits"].shape), (1, 3))
        self.assertEqual(tuple(qa["start_logits"].shape), (1, 4))
        self.assertEqual(tuple(retrieval.shape), (1, 8))
        self.assertEqual(tuple(rerank["logits"].shape), (1,))


if __name__ == "__main__":
    unittest.main()
