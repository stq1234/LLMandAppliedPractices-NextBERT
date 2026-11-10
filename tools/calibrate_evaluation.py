"""Bounded timing calibration for evaluation and frozen-head probes.

This script measures small, real-data samples only. It does not start a full
evaluation, diagnostic, or training run. CUDA visibility is controlled by the
caller (for example with ``CUDA_VISIBLE_DEVICES``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diagnostics_nextbert import (  # noqa: E402
    _probe_loss,
    _reset_probe_head,
    run_long_diagnostics,
    unique_source_questions,
)
from evaluate_nextbert import evaluate_classification, evaluate_qa  # noqa: E402
from modeling_nextbert import NextBert  # noqa: E402
from train_nextbert import _amp_context, _device, load_jsonl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


CALIBRATION_SAMPLE_COUNT = 64
INFERENCE_BATCH_SIZE = 16
QA_FEATURE_COUNT = 64
LONG_EVAL_BATCH_SIZE = 4
PROBE_BATCH_SIZE = 16
PROBE_WARMUP_UPDATES = 2
PROBE_TIMED_UPDATES = 3
PROBE_ESTIMATE_UPDATES = 100
LONG_LENGTHS = (1024, 4096, 8192)
PLAN_PATH = ROOT / "configs" / "experiment_plan.json"


def _stable_key(row: Mapping[str, Any], index: int) -> str:
    identity = row.get("id", row.get("feature_id", row.get("question_id", index)))
    return str(identity)


def _length_bucket(row: Mapping[str, Any]) -> str:
    length = len(row.get("input_ids", []))
    if length <= 128:
        return "0-128"
    if length <= 256:
        return "129-256"
    if length <= 512:
        return "257-512"
    if length <= 1024:
        return "513-1024"
    return ">1024"


def _stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    unique_question: bool = False,
) -> list[dict[str, Any]]:
    """Deterministically balance language and sequence-length strata."""
    grouped: dict[str, dict[str, list[tuple[int, Mapping[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    seen_questions: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not row.get("input_ids"):
            continue
        language = str(row.get("language", "und"))
        if unique_question:
            qid = str(row.get("question_id", row.get("id", index)))
            question_key = (language, qid)
            if question_key in seen_questions:
                continue
            seen_questions.add(question_key)
        grouped[language][_length_bucket(row)].append((index, row))

    for language, buckets in grouped.items():
        for bucket, members in buckets.items():
            members.sort(key=lambda pair: hashlib.sha256(
                f"{language}:{bucket}:{_stable_key(pair[1], pair[0])}".encode("utf-8")
            ).digest())

    selected: list[dict[str, Any]] = []
    cursors = {language: {bucket: 0 for bucket in buckets} for language, buckets in grouped.items()}
    bucket_cursor = {language: 0 for language in grouped}
    languages = sorted(grouped)
    total = sum(len(members) for buckets in grouped.values() for members in buckets.values())
    while len(selected) < min(count, total):
        progressed = False
        for language in languages:
            buckets = sorted(grouped[language])
            for shift in range(len(buckets)):
                bucket_index = (bucket_cursor[language] + shift) % len(buckets)
                bucket = buckets[bucket_index]
                cursor = cursors[language][bucket]
                if cursor < len(grouped[language][bucket]):
                    selected.append(dict(grouped[language][bucket][cursor][1]))
                    cursors[language][bucket] += 1
                    bucket_cursor[language] = (bucket_index + 1) % len(buckets)
                    progressed = True
                    break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "device_free_bytes": None,
            "device_total_bytes": None,
        }
    _sync(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


class ForwardCounter:
    """Temporarily count actual task-head batches and nonpadding tokens."""

    def __init__(self, model: nn.Module, method_name: str):
        self.model = model
        self.method_name = method_name
        self.original: Callable[..., Any] | None = None
        self.calls = 0
        self.examples = 0
        self.nonpad_tokens = 0

    def __enter__(self) -> "ForwardCounter":
        self.original = getattr(self.model, self.method_name)

        def counted(*args: Any, **kwargs: Any) -> Any:
            ids = kwargs.get("input_ids", args[0] if args else None)
            attention = kwargs.get("attention_mask")
            if isinstance(ids, torch.Tensor):
                self.examples += int(ids.shape[0]) if ids.ndim > 1 else 1
                self.nonpad_tokens += int(attention.sum().item()) if isinstance(attention, torch.Tensor) else int(ids.numel())
                self.calls += 1
            return self.original(*args, **kwargs)

        setattr(self.model, self.method_name, counted)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.original is not None:
            setattr(self.model, self.method_name, self.original)

    def as_dict(self) -> dict[str, int]:
        return {
            "forward_calls": self.calls,
            "input_features_or_examples": self.examples,
            "nonpadding_tokens": self.nonpad_tokens,
        }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ResultWriter:
    def __init__(self, output: Path, payload: dict[str, Any]):
        self.output = output
        self.payload = payload
        self.flush()

    def flush(self) -> None:
        _atomic_json(self.output, self.payload)

    def record(self, item: Mapping[str, Any]) -> None:
        self.payload["items"].append(dict(item))
        self.flush()
        print(json.dumps(item, ensure_ascii=False, separators=(",", ":")), flush=True)


def _run_item(writer: ResultWriter, name: str, function: Callable[[], Mapping[str, Any]]) -> dict[str, Any] | None:
    try:
        result = dict(function())
        item = {"name": name, "status": "measured", **result}
    except torch.cuda.OutOfMemoryError as exc:
        item = {"name": name, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # Keep each calibration subitem independently auditable.
        item = {
            "name": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    writer.record(item)
    return item if item["status"] == "measured" else None


def _time_eval(
    *,
    model: nn.Module,
    method_name: str,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    evaluator: Callable[..., Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    warm_rows = list(rows[: min(8, len(rows))])
    if not rows:
        raise ValueError("cannot benchmark evaluation on an empty sample")
    for _ in range(2):
        evaluator(model, warm_rows, device=device, batch_size=min(batch_size, len(warm_rows)))
    _sync(device)
    _reset_peak_memory(device)
    counter = ForwardCounter(model, method_name)
    started = time.perf_counter()
    with counter:
        metrics = dict(evaluator(model, rows, device=device, batch_size=batch_size))
    _sync(device)
    elapsed = time.perf_counter() - started
    return metrics, counter.as_dict(), elapsed


def _derive_split_path(data_path: Path, split: str) -> Path | None:
    match = re.match(r"^(.*)_(train|dev|validation|test)$", data_path.stem)
    if match:
        candidate = data_path.with_name(f"{match.group(1)}_{split}{data_path.suffix}")
        return candidate if candidate.is_file() else None
    candidate = data_path.with_name(f"{data_path.stem}_{split}{data_path.suffix}")
    return candidate if candidate.is_file() else None


def _probe_rows(
    rows: Sequence[Mapping[str, Any]],
    task: str,
    count: int = PROBE_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before = len(rows)
    if task == "qa":
        rows = [
            row for row in rows
            if row.get("input_ids")
            and len(row["input_ids"]) <= 512
            and row.get("window_type") != "long"
        ]
        filter_rule = "input_ids length <= 512 and window_type != 'long'"
    else:
        filter_rule = "no length filter"
    after_length_filter = len(rows)
    valid: list[Mapping[str, Any]] = []
    for row in rows:
        if not row.get("input_ids"):
            continue
        if task == "classification":
            if row.get("label") is None:
                continue
        else:
            if int(row.get("start_positions", -100)) < 0 or int(row.get("end_positions", -100)) < 0:
                continue
            context_mask = row.get("context_mask")
            if context_mask is not None and not any(int(value) for value in context_mask):
                continue
        valid.append(row)
    sample = _stratified_sample(valid, count, unique_question=(task == "qa"))
    if not sample:
        return [], {
            "rule": filter_rule,
            "rows_before": before,
            "rows_after_length_filter": after_length_filter,
            "rows_probe_eligible": len(valid),
            "rows_sampled": 0,
        }
    while len(sample) < count:
        sample.extend(sample[: min(count - len(sample), len(sample))])
    sample = sample[:count]
    return sample, {
        "rule": filter_rule,
        "rows_before": before,
        "rows_after_length_filter": after_length_filter,
        "rows_probe_eligible": len(valid),
        "rows_sampled": len(sample),
    }


def _run_probe(
    *,
    model: nn.Module,
    task: str,
    train_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    rows, train_filter = _probe_rows(train_rows, task)
    if len(rows) != PROBE_BATCH_SIZE:
        raise ValueError(f"need at least one valid {task} training row to fill batch {PROBE_BATCH_SIZE}")
    _reset_probe_head(model, task, seed)
    freeze_info = diagnostics_freeze(model, task)
    if hasattr(model, "set_gate_ramp"):
        model.set_gate_ramp(1.0)
    model.eval()
    head = model.classifier if task == "classification" else model.qa_outputs
    head.train()
    if task == "classification" and hasattr(model, "task_dropout"):
        model.task_dropout.train()
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    def update() -> float:
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device):
            loss = _probe_loss(model, task, rows, device)
        loss.backward()
        optimizer.step()
        return float(loss.detach().float().cpu())

    for _ in range(PROBE_WARMUP_UPDATES):
        update()
    _sync(device)
    _reset_peak_memory(device)
    durations: list[float] = []
    losses: list[float] = []
    counter = ForwardCounter(model, "forward_classification" if task == "classification" else "forward_qa")
    with counter:
        for _ in range(PROBE_TIMED_UPDATES):
            _sync(device)
            started = time.perf_counter()
            losses.append(update())
            _sync(device)
            durations.append(time.perf_counter() - started)
    return {
        "task": task,
        "warmup_updates": PROBE_WARMUP_UPDATES,
        "timed_updates": PROBE_TIMED_UPDATES,
        "total_updates_executed": PROBE_WARMUP_UPDATES + PROBE_TIMED_UPDATES,
        "physical_batch_size": PROBE_BATCH_SIZE,
        "sampled_train_rows": len(rows),
        "training_filter": train_filter,
        "mean_input_length": statistics.mean(len(row["input_ids"]) for row in rows),
        "timed_step_seconds": durations,
        "mean_timed_step_seconds": statistics.mean(durations),
        "estimated_100_update_seconds": statistics.mean(durations) * PROBE_ESTIMATE_UPDATES,
        "estimated_updates": PROBE_ESTIMATE_UPDATES,
        "timed_mean_loss": statistics.mean(losses),
        "forward_counts": counter.as_dict(),
        "freeze": freeze_info,
        "memory": _memory(device),
    }


def diagnostics_freeze(model: nn.Module, task: str) -> dict[str, Any]:
    # Keep this import local so the calibration CLI's public module surface is
    # small while sharing the exact probe freezing invariant.
    from diagnostics_nextbert import freeze_probe_parameters

    return freeze_probe_parameters(model, task)


def _read_background_data(path: Path) -> dict[str, list[dict[str, Any]]]:
    from diagnostics_nextbert import load_heldout_backgrounds

    if path.is_dir():
        result: dict[str, list[dict[str, Any]]] = {}
        for language in ("en", "zh"):
            candidate = path / f"selected_{language}.jsonl"
            if candidate.is_file():
                result[language] = load_heldout_backgrounds(path, language)
        if not result:
            raise FileNotFoundError(f"no selected_en/zh.jsonl files under {path}")
        return result
    if not path.is_file():
        raise FileNotFoundError(path)
    match = re.search(r"selected_(en|zh)", path.stem)
    rows = load_jsonl(path)
    language = match.group(1) if match else None
    if language is None:
        languages = {str(row.get("language")) for row in rows if row.get("language") in {"en", "zh"}}
        if len(languages) != 1:
            raise ValueError("a background JSONL file must be named selected_en/zh or contain one language")
        language = languages.pop()
    heldout = [row for row in rows if str(row.get("split", "")) in {"dev", "test"} and row.get("text") and row.get("source_id")]
    if not heldout:
        raise ValueError(f"{path} has no held-out dev/test Wikipedia background rows")
    return {language: heldout}


def _device_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result["cuda_name"] = torch.cuda.get_device_name(device)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def _row_token_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [len(row.get("input_ids", [])) for row in rows]
    return {
        "feature_count": len(rows),
        "nonpadding_tokens": sum(lengths),
        "mean_input_length": statistics.mean(lengths) if lengths else 0.0,
        "max_input_length": max(lengths, default=0),
        "language_counts": dict(sorted(
            (language, sum(str(row.get("language", "und")) == language for row in rows))
            for language in {str(row.get("language", "und")) for row in rows}
        )),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_jsonl(path)


def _measure_evaluation(
    *,
    name: str,
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    task: str,
    device: torch.device,
) -> dict[str, Any]:
    sample_count = CALIBRATION_SAMPLE_COUNT if task == "classification" else QA_FEATURE_COUNT
    sampled = _stratified_sample(
        rows,
        sample_count,
        unique_question=False,
    )
    if not sampled:
        raise ValueError(f"{name} has no rows with input IDs")
    method = "forward_classification" if task == "classification" else "forward_qa"
    evaluator = evaluate_classification if task == "classification" else evaluate_qa
    metrics, counts, elapsed = _time_eval(
        model=model,
        method_name=method,
        rows=sampled,
        device=device,
        batch_size=INFERENCE_BATCH_SIZE,
        evaluator=evaluator,
    )
    full_rows = len(all_rows)
    full_estimate = elapsed * full_rows / len(sampled)
    return {
        "task": task,
        "official_split_sample": _row_token_stats(sampled),
        "full_split_feature_count": full_rows,
        "physical_batch_size": INFERENCE_BATCH_SIZE,
        "warmup_forward_calls": 2,
        "elapsed_seconds": elapsed,
        "estimated_full_split_seconds_linear": full_estimate,
        "estimate_note": "linear scaling from deterministic, language/length-stratified sample; includes collation, model forward, decoding and scoring",
        "forward_counts": counts,
        "metric_summary": {
            key: metrics[key]
            for key in (
                "count", "question_count", "overflow_windows", "accuracy", "macro_f1",
                "exact_match", "f1", "by_language", "exact_match_by_language", "f1_by_language",
            )
            if key in metrics
        },
        "memory": _memory(device),
    }


def _measure_long(
    *,
    model: nn.Module,
    tokenizer: Any,
    qa_rows: Sequence[Mapping[str, Any]],
    background_by_language: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
    long_source_groups_per_language: int,
) -> dict[str, Any]:
    language_seconds: dict[str, float] = {}
    language_records: dict[str, Any] = {}
    total_forward_counts = {"forward_calls": 0, "input_features_or_examples": 0, "nonpadding_tokens": 0}
    total_predictions = 0
    total_source_groups = 0
    all_skipped_cases = 0
    all_skipped_counterfactuals = 0
    language_memory: list[dict[str, int | None]] = []

    qa_languages = {str(row.get("language", "und")) for row in qa_rows}
    for language in sorted(set(background_by_language) & qa_languages):
        language_rows = [row for row in qa_rows if str(row.get("language", "und")) == language]
        planned_groups = unique_source_questions(language_rows, long_source_groups_per_language)
        if not planned_groups:
            continue
        _sync(device)
        _reset_peak_memory(device)
        counter = ForwardCounter(model, "forward_qa")
        started = time.perf_counter()
        with counter:
            result = run_long_diagnostics(
                model=model,
                tokenizer=tokenizer,
                qa_rows=language_rows,
                wiki_rows_by_language={language: list(background_by_language[language])},
                max_source_groups=long_source_groups_per_language,
                device=device,
                batch_size=LONG_EVAL_BATCH_SIZE,
            )
        _sync(device)
        elapsed = time.perf_counter() - started
        language_seconds[language] = elapsed
        count = counter.as_dict()
        memory = _memory(device)
        language_memory.append(memory)
        for key in total_forward_counts:
            total_forward_counts[key] += count[key]
        total_predictions += int(result.get("rows_evaluated", 0))
        groups = {str(row["source_group_id"]) for row in result.get("predictions", [])}
        total_source_groups += len(groups)
        all_skipped_cases += len(result.get("skipped_cases", []))
        all_skipped_counterfactuals += len(result.get("skipped_counterfactuals", []))
        language_records[language] = {
            "elapsed_seconds": elapsed,
            "planned_source_groups": len(planned_groups),
            "evaluated_source_groups": len(groups),
            "rows_evaluated_case_view_pairs": int(result.get("rows_evaluated", 0)),
            "skipped_cases": len(result.get("skipped_cases", [])),
            "skipped_counterfactuals": len(result.get("skipped_counterfactuals", [])),
            "per_language_metrics": result.get("by_language", {}).get(language, {}),
            "forward_counts": count,
            "memory": memory,
        }
    if not language_seconds:
        raise ValueError("no language has both QA rows and held-out background data")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    target_groups = int(plan["evaluation"]["long_context"]["source_groups_per_language"])
    variant_count = len(plan["variants"])
    projections = {
        language: seconds * target_groups / int(language_records[language]["evaluated_source_groups"])
        for language, seconds in language_seconds.items()
        if int(language_records[language]["evaluated_source_groups"]) > 0
    }
    per_language_group_rates = {
        language: {
            "seconds_per_evaluated_source_group": language_seconds[language] / int(language_records[language]["evaluated_source_groups"])
            if int(language_records[language]["evaluated_source_groups"]) > 0 else None,
            "case_view_pairs_per_evaluated_source_group": int(language_records[language]["rows_evaluated_case_view_pairs"])
            / int(language_records[language]["evaluated_source_groups"])
            if int(language_records[language]["evaluated_source_groups"]) > 0 else None,
            "forward_calls_per_evaluated_source_group": int(language_records[language]["forward_counts"]["forward_calls"])
            / int(language_records[language]["evaluated_source_groups"])
            if int(language_records[language]["evaluated_source_groups"]) > 0 else None,
        }
        for language in language_seconds
    }
    single_variant_200_group_estimate = sum(projections.values()) if projections else None
    overall_memory = _memory(device)
    if language_memory and device.type == "cuda":
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            overall_memory[key] = max(int(memory[key] or 0) for memory in language_memory)
    return {
        "lengths": list(LONG_LENGTHS),
        "target_source_groups_per_language_from_plan": target_groups,
        "variant_count_from_plan": variant_count,
        "source_groups_per_language_requested": long_source_groups_per_language,
        "evaluation_batch_size": LONG_EVAL_BATCH_SIZE,
        "evaluated_source_group_count": total_source_groups,
        "language_elapsed_seconds": language_seconds,
        "language_results": language_records,
        "elapsed_seconds": sum(language_seconds.values()),
        "per_language_group_rates": per_language_group_rates,
        "rows_evaluated_case_view_pairs": total_predictions,
        "forward_counts": total_forward_counts,
        "languages_projected": sorted(projections),
        "estimated_source_groups_current_variant": target_groups * len(projections),
        "estimated_plan_groups_current_variant_seconds": single_variant_200_group_estimate,
        "estimated_plan_groups_all_variants_seconds_if_same_rate": single_variant_200_group_estimate * variant_count if single_variant_200_group_estimate is not None else None,
        "estimate_note": "linear scaling of the complete 1K/4K/8K x position x intervention x view protocol; current-variant per-language rates are projected to the plan's source-group count, then multiplied by its variant count; run V0 and V3 separately to capture their different rates",
        "skipped_cases": all_skipped_cases,
        "skipped_counterfactuals": all_skipped_counterfactuals,
        "memory": overall_memory,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path,
                        help="Local tokenizer directory; defaults to --model-dir")
    parser.add_argument("--variant", choices=("V0", "V3"), required=True)
    parser.add_argument("--classification-data", type=Path, required=True,
                        help="Prepared official XNLI split JSONL, normally classification_test.jsonl")
    parser.add_argument("--qa-data", type=Path, required=True,
                        help="Prepared official QA split feature JSONL, normally qa_test.jsonl")
    parser.add_argument("--background-data", type=Path,
                        help="Optional directory with held-out selected_en/zh.jsonl or one selected_<lang>.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-source-groups", type=int, default=4,
                        help="Bounded source groups per available language for the long-context calibration")
    parser.add_argument("--items", choices=("all", "long"), default="all",
                        help="Run all calibration items or only the long-context item")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.long_source_groups < 1:
        raise SystemExit("--long-source-groups must be positive")
    if args.items == "long" and args.background_data is None:
        raise SystemExit("--items long requires --background-data")
    for name, path in (
        ("model-dir", args.model_dir),
        ("classification-data", args.classification_data),
        ("qa-data", args.qa_data),
    ):
        if not path.exists():
            raise SystemExit(f"--{name} path does not exist: {path}")
    if args.background_data is not None and not args.background_data.exists():
        raise SystemExit(f"--background-data path does not exist: {args.background_data}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "calibration": "bounded_official_inference_long_evidence_and_frozen_probe",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "model_dir": str(args.model_dir.resolve()),
        "tokenizer_path": str((args.tokenizer_path or args.model_dir).resolve()),
        "classification_data": str(args.classification_data.resolve()),
        "qa_data": str(args.qa_data.resolve()),
        "background_data": str(args.background_data.resolve()) if args.background_data else None,
        "items": [],
        "protocol": {
            "classification_and_qa_inference_sample_count": CALIBRATION_SAMPLE_COUNT,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "long_evaluation_batch_size": LONG_EVAL_BATCH_SIZE,
            "qa_sample_unit": "64 deterministic features, balanced by language and round-robin length bucket",
            "probe_warmup_updates": PROBE_WARMUP_UPDATES,
            "probe_timed_updates": PROBE_TIMED_UPDATES,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "probe_full_estimate_updates": PROBE_ESTIMATE_UPDATES,
            "long_lengths": list(LONG_LENGTHS),
            "long_source_groups_per_language": args.long_source_groups,
            "long_protocol": "all diagnostics_nextbert positions, interventions, and first512/sliding512/full views",
        },
    }
    writer = ResultWriter(args.output, payload)

    try:
        device = _device()
        payload["runtime"] = {
            **_device_metadata(device),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        }
        writer.flush()
        model_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path or args.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        if not tokenizer.is_fast:
            raise ValueError("calibration requires the fast tokenizer for context offsets")
        model = NextBert(
            str(args.model_dir),
            variant=args.variant,
            attention_backend="auto",
            tokenizer=tokenizer,
        ).to(device)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        _sync(device)
        payload["model_load_seconds"] = time.perf_counter() - model_started
        writer.flush()

        qa_all = _load_jsonl(args.qa_data)
        if args.items == "all":
            classification_all = _load_jsonl(args.classification_data)
            _run_item(
                writer,
                "classification_official_split_inference",
                lambda: _measure_evaluation(
                    name="classification_official_split_inference",
                    model=model,
                    rows=classification_all,
                    all_rows=classification_all,
                    task="classification",
                    device=device,
                ),
            )
            _run_item(
                writer,
                "qa_official_split_inference",
                lambda: _measure_evaluation(
                    name="qa_official_split_inference",
                    model=model,
                    rows=qa_all,
                    all_rows=qa_all,
                    task="qa",
                    device=device,
                ),
            )

        if args.background_data is None and args.items == "all":
            writer.record({
                "name": "controlled_long_evidence_inference",
                "status": "not_measured",
                "reason": "--background-data was not provided",
                "lengths": list(LONG_LENGTHS),
            })
        elif args.background_data is not None:
            def long_item() -> Mapping[str, Any]:
                backgrounds = _read_background_data(args.background_data)
                return _measure_long(
                    model=model,
                    tokenizer=tokenizer,
                    qa_rows=qa_all,
                    background_by_language=backgrounds,
                    device=device,
                    long_source_groups_per_language=args.long_source_groups,
                )

            _run_item(writer, "controlled_long_evidence_inference", long_item)

        if args.items == "all":
            for task, data_path in (
                ("classification", args.classification_data),
                ("qa", args.qa_data),
            ):
                train_path = _derive_split_path(data_path, "train")
                if train_path is None:
                    writer.record({
                        "name": f"frozen_{task}_probe",
                        "status": "not_measured",
                        "reason": f"could not derive adjacent {task}_train.jsonl from {data_path.name}",
                        "warmup_updates": PROBE_WARMUP_UPDATES,
                        "timed_updates": PROBE_TIMED_UPDATES,
                    })
                    continue
                train_rows = _load_jsonl(train_path)
                _run_item(
                    writer,
                    f"frozen_{task}_probe",
                    lambda task=task, train_rows=train_rows: _run_probe(
                        model=model,
                        task=task,
                        train_rows=train_rows,
                        device=device,
                        seed=args.seed,
                    ) | {"train_data": str(train_path.resolve())},
                )
        payload["status"] = "completed_with_failures" if any(
            item.get("status") == "failed" for item in payload["items"]
        ) else "completed"
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        writer.flush()
        print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}), flush=True)
        return 0 if payload["status"] == "completed" else 1
    except Exception as exc:
        payload["status"] = "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=12)
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        writer.flush()
        print(json.dumps({"output": str(args.output.resolve()), "status": "failed", "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
