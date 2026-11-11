#!/usr/bin/env python3
"""Summarize a completed formal_67m training run to compact JSON on stdout.

This reports optimizer-step training logs only. It does not compute evaluation
metrics, launch training, load model weights, or require PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
JSON_DECODER = json.JSONDecoder()


class AuditError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object in {path}")
    return value


def _controller_summary(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read controller log {path}: {exc}") from exc
    # The controller writes human-readable scheduling lines followed by one
    # unindented JSON result object. Ignore any JSON-looking embedded content.
    for match in reversed(list(re.finditer(r"(?m)^\{", text))):
        try:
            value, _ = JSON_DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "job_count" in value and "status" in value:
            return value
    raise AuditError(f"no final controller JSON object found in {path}")


def _marker(lines: list[str], prefix: str) -> tuple[int, dict[str, Any] | None]:
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                return index, None
            return index, value if isinstance(value, dict) else None
    return -1, None


def _parse_worker_log(path: Path, root: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read worker log {path}: {exc}") from exc
    lines = text.splitlines()
    schedule_index, scheduler = _marker(lines, "__SCHEDULER_JOB__=")
    start_index, started = _marker(lines, "__WORKER_START__=")
    end_index, ended = _marker(lines, "__WORKER_END__=")
    if scheduler is None or started is None or ended is None:
        raise AuditError(f"worker log is missing a valid scheduler/start/end marker: {path}")
    if not (schedule_index < start_index < end_index):
        raise AuditError(f"worker markers are out of order: {path}")

    offsets: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)
    result: dict[str, Any] | None = None
    for index in range(start_index + 1, end_index):
        if lines[index].strip() != "{":
            continue
        try:
            value, _ = JSON_DECODER.raw_decode(text, offsets[index])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"task", "variant", "log"}.issubset(value):
            result = value
            break
    if result is None:
        raise AuditError(f"training result JSON not found in worker log {path}")

    raw_training_log = result.get("log")
    training_log = Path(raw_training_log) if raw_training_log else None
    if training_log is not None and not training_log.is_file():
        # Permit a copied run tree whose serialized paths still name the
        # original remote root, while keeping all resolution under --root.
        marker = "/LLM/nextbert/"
        raw = str(training_log).replace("\\", "/")
        if marker in raw:
            training_log = root / raw.split(marker, 1)[1]
    if training_log is None or not training_log.is_file():
        raise AuditError(f"training JSONL referenced by {path} is missing: {raw_training_log}")

    training_rows: list[dict[str, Any]] = []
    try:
        with training_log.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AuditError(f"non-object JSONL row at {training_log}:{line_number}")
                training_rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read training JSONL {training_log}: {exc}") from exc
    if not training_rows:
        raise AuditError(f"training JSONL is empty: {training_log}")

    return {
        "worker_log": path,
        "scheduler": scheduler,
        "started": started,
        "ended": ended,
        "result": result,
        "training_log": training_log,
        "rows": training_rows,
    }


def _adaptation_schedule(plan: dict[str, Any]) -> list[dict[str, int]]:
    offset = 0
    output = []
    for raw in plan["adaptation"]["stages"]:
        stage = dict(raw)
        steps = int(stage["optimizer_steps"])
        stage["start_step"] = offset
        stage["end_step"] = offset + steps
        output.append(stage)
        offset += steps
    return output


def _qa_phase_steps(plan: dict[str, Any]) -> dict[str, int]:
    downstream = plan["downstream"]
    qa = downstream["qa"]
    long = qa.get("long_augmentation", {})
    epochs = int(downstream.get("shared", {}).get("epochs", 2))
    language_count = int(qa.get("language_count", len(qa.get("sources", ("zh", "en")))))
    regular_target = int(qa.get("regular_target_questions",
                                int(qa.get("train_questions_per_language", 0)) * language_count))
    regular_effective = int(qa.get("effective_batch",
                                   int(qa.get("physical_batch", 16)) * int(qa.get("accumulation", 2))))
    regular_steps = int(qa.get("regular_steps", math.ceil(regular_target * epochs / regular_effective)))
    long_target = long.get("train_questions_total")
    if long_target is None:
        long_target = math.ceil(regular_target * float(long.get("fraction_of_regular_training_windows", 0.0)))
    long_effective = int(long.get("effective_batch",
                                  int(long.get("physical_batch", 1)) * int(long.get("accumulation", 1))))
    long_steps = int(long.get("steps", math.ceil(int(long_target) * epochs / max(1, long_effective))))
    return {"regular": regular_steps, "long": long_steps}


def _expected_jobs(plan: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, int]]]:
    schedule = _adaptation_schedule(plan)
    expected: dict[str, dict[str, Any]] = {}
    shared = plan["adaptation"].get("shared_prefix", {})
    shared_variants = set(shared.get("variants", []))
    shared_owner = next(iter(shared.get("variants", [])), None)
    shared_steps = int(shared.get("steps", 0))
    for variant in plan["variants"]:
        variant_id = str(variant["id"])
        for stage in schedule:
            stage_start = int(stage["start_step"])
            stage_end = int(stage["end_step"])
            owns_shared_prefix = variant_id == shared_owner
            effective_start = (max(stage_start, shared_steps)
                               if variant_id in shared_variants and not owns_shared_prefix
                               else stage_start)
            steps = max(0, stage_end - effective_start)
            if steps:
                job_id = f"{variant_id}-mlm-{int(stage['max_length'])}"
                expected[job_id] = {
                    "job_id": job_id,
                    "variant": variant_id,
                    "task": "mlm",
                    "length": int(stage["max_length"]),
                    "steps": steps,
                    "stage": stage,
                    "seed": int(plan["adaptation"].get("seed", 17)),
                }

    downstream = plan["downstream"]
    qa_steps = _qa_phase_steps(plan)
    task_steps = {
        "classification": int(downstream["classification"]["steps"]),
        "qa": qa_steps["regular"] + qa_steps["long"],
        "retrieval": int(downstream["retrieval"]["steps"]),
        "reranking": int(downstream["reranking"]["steps"]),
    }
    for matrix in downstream.get("run_matrix", []):
        for variant in matrix["variants"]:
            for task in matrix["tasks"]:
                for seed in matrix["seeds"]:
                    job_id = f"{variant}-{task}-{seed}"
                    expected[job_id] = {
                        "job_id": job_id,
                        "variant": str(variant),
                        "task": str(task),
                        "steps": task_steps[str(task)],
                        "seed": int(seed),
                        "stage": None,
                    }
    return expected, schedule


def _rows_in_data_manifest(data_manifest: dict[str, Any], artifact: str) -> int | None:
    item = data_manifest.get("artifacts", {}).get(artifact)
    if not isinstance(item, dict) or item.get("rows") is None:
        return None
    return int(item["rows"])


def _expected_sample_rows(task: str, stage: dict[str, Any] | None,
                          plan: dict[str, Any], data_manifest: dict[str, Any]) -> int | None:
    if task == "mlm" and stage is not None:
        item = data_manifest.get("artifacts", {}).get(f"wiki_{int(stage['max_length'])}", {})
        shape = item.get("shape") if isinstance(item, dict) else None
        return int(shape[0]) if isinstance(shape, list) and shape else None
    keys = {
        "classification": "classification_train",
        "qa": "qa_train",
        "retrieval": "retrieval_train",
        "reranking": "reranking_train",
    }
    if task in keys:
        rows = _rows_in_data_manifest(data_manifest, keys[task])
        if rows is not None:
            return rows
    downstream = plan["downstream"]
    if task == "classification":
        section = downstream[task]
        return int(section["train_per_language"]) * len(section.get("sources", []))
    if task == "qa":
        section = downstream[task]
        long = section.get("long_augmentation", {})
        languages = int(section.get("language_count", len(section.get("sources", []))))
        regular = int(section.get("train_questions_per_language", 0)) * languages
        long_rows = int(long.get("train_questions_total",
                                math.ceil(regular * float(long.get("fraction_of_regular_training_windows", 0.0)))))
        return regular + long_rows
    if task == "retrieval":
        section = downstream[task]
        return int(section["train_queries_per_language"]) * 2
    if task == "reranking":
        return int(downstream[task]["train_queries_total"])
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _batch_key(row: dict[str, Any], task: str) -> tuple[Any, ...]:
    if task == "retrieval":
        fields = ("physical_query_batch", "physical_document_batch",
                  "logical_retrieval_queries", "gradient_accumulation", "effective_batch")
    else:
        fields = ("physical_batch", "gradient_accumulation", "effective_batch")
    return tuple(row.get(name) for name in fields)


def _batch_distribution(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    counts = Counter(_batch_key(row, task) for row in rows)
    fields = (("physical_query_batch", "physical_document_batch", "logical_retrieval_queries",
               "gradient_accumulation", "effective_batch") if task == "retrieval" else
              ("physical_batch", "gradient_accumulation", "effective_batch"))
    output = []
    for values, count in sorted(counts.items(), key=lambda item: str(item[0])):
        output.append({**dict(zip(fields, values)), "steps": count})
    return output


def _batch_check(run: dict[str, Any], expected: dict[str, Any], plan: dict[str, Any],
                 qa_steps: dict[str, int]) -> tuple[bool, list[dict[str, Any]]]:
    task = str(expected["task"])
    downstream = plan["downstream"]
    issues: list[dict[str, Any]] = []

    def compare(index: int, field: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            issues.append({"step": _safe_int(run["rows"][index].get("step")),
                           "field": field, "actual": actual, "expected": wanted})

    for index, row in enumerate(run["rows"]):
        if task == "mlm":
            stage = expected["stage"]
            wanted = {
                "physical_batch": int(stage["physical_microbatch"]),
                "gradient_accumulation": int(stage["gradient_accumulation"]),
                "effective_batch": int(stage["effective_sequences"]),
                "length": int(stage["max_length"]),
                "nonpad_tokens": int(stage["input_tokens"]) // int(stage["optimizer_steps"]),
            }
            for field, value in wanted.items():
                compare(index, field, _safe_int(row.get(field)), value)
        elif task == "classification":
            stage = downstream[task]
            for field, value in {
                "physical_batch": int(stage["physical_batch"]),
                "gradient_accumulation": int(stage["accumulation"]),
                "effective_batch": int(stage["effective_batch"]),
            }.items():
                compare(index, field, _safe_int(row.get(field)), value)
        elif task == "qa":
            stage = downstream[task]
            regular = int(_safe_int(row.get("step")) or 0) <= qa_steps["regular"]
            selected = stage if regular else stage["long_augmentation"]
            for field, value in {
                "physical_batch": int(selected["physical_batch"]),
                "gradient_accumulation": int(selected["accumulation"]),
                "effective_batch": int(selected["effective_batch"]),
            }.items():
                compare(index, field, _safe_int(row.get(field)), value)
        elif task == "retrieval":
            stage = downstream[task]
            for field, value in {
                "physical_query_batch": int(stage["physical_query_batch"]),
                "physical_document_batch": int(stage["physical_document_batch"]),
                "logical_retrieval_queries": int(stage["logical_batch_queries"]),
                "gradient_accumulation": 1,
                "effective_batch": int(stage["logical_batch_queries"]),
            }.items():
                compare(index, field, _safe_int(row.get(field)), value)
        elif task == "reranking":
            stage = downstream[task]
            for field, value in {
                "physical_batch": int(stage["physical_query_groups"]),
                "gradient_accumulation": int(stage["accumulation"]),
                "effective_batch": int(stage["effective_query_groups"]),
            }.items():
                compare(index, field, _safe_int(row.get(field)), value)

    return not issues, issues[:8]


def _mode_check(run: dict[str, Any], expected: dict[str, Any], shared_steps: int) -> tuple[bool, dict[str, int]]:
    if expected["task"] != "mlm":
        return True, {}
    stage = expected["stage"]
    balanced = expected["variant"] == "V3" and int(stage["start_step"]) >= shared_steps
    counts = Counter(str(row.get("mlm_mode")) for row in run["rows"])
    if balanced:
        half = int(expected["steps"]) // 2
        matches = counts.get("token", 0) == half and counts.get("span", 0) == half
    else:
        matches = counts.get("token", 0) == int(expected["steps"]) and len(counts) == 1
    return matches, dict(sorted(counts.items()))


def _loss_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(row["loss"]) for row in rows if row.get("loss") is not None]
    tail = losses[-20:]
    return {
        "final_training_loss": round(losses[-1], 8) if losses else None,
        "last20_mean_training_loss": round(statistics.fmean(tail), 8) if tail else None,
        "last20_count": len(tail),
    }


def _max_memory(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [
        _safe_int(row.get("memory", {}).get(key))
        for row in rows if isinstance(row.get("memory"), dict)
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _round_seconds(value: float | None) -> float | None:
    return round(value, 3) if value is not None and math.isfinite(value) else None


def _utc(unix_seconds: float | None) -> str | None:
    if unix_seconds is None:
        return None
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


def _summarize_run(run: dict[str, Any], expected: dict[str, Any] | None,
                   plan: dict[str, Any], data_manifest: dict[str, Any],
                   qa_steps: dict[str, int], shared_steps: int,
                   controller_return_codes: dict[str, int]) -> dict[str, Any]:
    rows = run["rows"]
    result = run["result"]
    scheduler = run["scheduler"]
    task = str(result["task"])
    variant = str(result["variant"])
    length = (
        int(expected["stage"]["max_length"])
        if task == "mlm" and expected and expected.get("stage") is not None
        else _safe_int(rows[0].get("length")) if rows else _safe_int(result.get("length"))
    )
    samples = result.get("sampler") or {}
    step_ids = [_safe_int(row.get("step")) for row in rows]
    optimizer_seconds = math.fsum(float(row.get("optimizer_step_seconds", 0.0)) for row in rows)
    token_total = sum(_safe_int(row.get("nonpad_tokens")) or 0 for row in rows)
    peak_allocated = _max_memory(rows, "peak_allocated_bytes")
    peak_reserved = _max_memory(rows, "peak_reserved_bytes")
    batch_ok, batch_issues = _batch_check(run, expected, plan, qa_steps) if expected else (False, [])
    mode_ok, mode_counts = _mode_check(run, expected, shared_steps) if expected else (False, {})
    expected_steps = int(expected["steps"]) if expected else None
    actual_steps = len(rows)
    requested_steps = _safe_int(result.get("requested_steps"))
    completed_steps = _safe_int(result.get("completed_this_call"))
    reported_last_step = _safe_int(result.get("last_step"))
    expected_requested_steps = (
        int(expected["stage"]["end_step"])
        if task == "mlm" and expected and expected.get("stage") is not None
        else expected_steps
    )
    expected_last_step = expected_requested_steps
    expected_unique_rows = _expected_sample_rows(task, expected.get("stage") if expected else None,
                                                 plan, data_manifest) if expected else None
    expected_effective_batch = None
    if task == "mlm" and expected:
        expected_effective_batch = int(expected["stage"]["effective_sequences"])
    elif task == "classification":
        expected_effective_batch = int(plan["downstream"][task]["effective_batch"])
    elif task == "qa":
        expected_effective_batch = int(plan["downstream"][task]["effective_batch"])
    elif task == "retrieval":
        expected_effective_batch = int(plan["downstream"][task]["logical_batch_queries"])
    elif task == "reranking":
        expected_effective_batch = int(plan["downstream"][task]["effective_query_groups"])
    expected_draws = expected_steps * expected_effective_batch if expected_steps is not None and expected_effective_batch else None
    actual_draws = _safe_int(samples.get("drawn_rows_this_call"))
    actual_unique = _safe_int(samples.get("unique_rows_this_call"))
    sample_match = (expected_unique_rows is not None and expected_draws is not None
                    and actual_unique == expected_unique_rows and actual_draws == expected_draws)
    steps_match = (expected_steps is not None and actual_steps == expected_steps
                   and requested_steps == expected_requested_steps
                   and completed_steps == expected_steps
                   and reported_last_step == expected_last_step)
    token_match: bool | None = None
    if task == "mlm" and expected:
        token_match = token_total == int(expected["stage"]["input_tokens"])
    controller_rc = controller_return_codes.get(str(scheduler.get("job_id")))
    worker_success = bool(run["ended"].get("success")) and controller_rc == 0
    batch_distribution = _batch_distribution(rows, task)
    mode_match = mode_ok if task == "mlm" else None
    checks = {
        "worker_success": worker_success,
        "steps_match_plan": steps_match,
        "samples_match_manifest_and_batch_budget": sample_match,
        "batch_matches_plan": batch_ok,
        "mlm_tokens_match_plan": token_match,
        "mlm_mode_matches_plan": mode_match,
    }
    return {
        "job_id": str(scheduler.get("job_id", "unknown")),
        "wave": int(scheduler.get("wave", 0)),
        "lane": str(scheduler.get("lane", "?")),
        "task": task,
        "variant": variant,
        "seed": _safe_int(result.get("seed")),
        "length": length,
        "training_log": str(run["training_log"]),
        "worker_elapsed_seconds": _round_seconds(float(run["ended"]["finished_unix"])
                                                  - float(run["started"]["started_unix"])),
        "step_count": actual_steps,
        "requested_steps": requested_steps,
        "completed_steps_this_call": completed_steps,
        "first_step": min((x for x in step_ids if x is not None), default=None),
        "last_step": max((x for x in step_ids if x is not None), default=None),
        "optimizer_step_seconds_sum": _round_seconds(optimizer_seconds),
        "nonpad_tokens_sum": token_total,
        "sample_rows_drawn": actual_draws,
        "sample_rows_unique": actual_unique,
        "sample_rows_repeated": _safe_int(samples.get("repeated_draws_this_call")),
        "row_coverage": samples.get("row_coverage"),
        "training_loss": _loss_summary(rows),
        "step_window_peak_allocated_bytes": peak_allocated,
        "step_window_peak_reserved_bytes": peak_reserved,
        "batch_distribution": batch_distribution,
        "mlm_mode_counts": mode_counts if task == "mlm" else None,
        "plan_checks": checks,
        "batch_mismatch_examples": batch_issues,
        "plan_expected": {
            "steps": expected_steps,
            "unique_sample_rows": expected_unique_rows,
            "sample_rows_drawn": expected_draws,
            "effective_batch": expected_effective_batch,
        },
        "worker_end_peak_since_last_reset": run["ended"].get("after_runpy", {}),
    }


def _group_runs(run_summaries: list[dict[str, Any]], expected_jobs: dict[str, dict[str, Any]],
                shared_info: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in run_summaries:
        groups[f"{run['variant']}/{run['task']}"].append(run)
    output: dict[str, Any] = {}
    for key, phases in sorted(groups.items()):
        phases.sort(key=lambda phase: (phase["length"] or 0, phase["seed"] or 0, phase["wave"]))
        losses = phases[-1]["training_loss"]
        group = {
            "variant": phases[0]["variant"],
            "task": phases[0]["task"],
            "optimizer_steps": sum(phase["step_count"] for phase in phases),
            "optimizer_step_seconds_sum": _round_seconds(math.fsum(
                phase["optimizer_step_seconds_sum"] or 0.0 for phase in phases)),
            "nonpad_tokens_sum": sum(phase["nonpad_tokens_sum"] for phase in phases),
            "sample_rows_drawn_sum": sum(phase["sample_rows_drawn"] or 0 for phase in phases),
            "sample_rows_unique_sum_across_phases": sum(phase["sample_rows_unique"] or 0 for phase in phases),
            "final_phase": phases[-1]["length"] if phases[-1]["task"] == "mlm" else f"seed_{phases[-1]['seed']}",
            "final_training_loss": losses["final_training_loss"],
            "last20_mean_training_loss": losses["last20_mean_training_loss"],
            "last20_count": losses["last20_count"],
            "peak_step_window_allocated_bytes": max(
                (phase["step_window_peak_allocated_bytes"] or 0 for phase in phases), default=0) or None,
            "peak_step_window_reserved_bytes": max(
                (phase["step_window_peak_reserved_bytes"] or 0 for phase in phases), default=0) or None,
            "batch_matches_plan": all(phase["plan_checks"]["batch_matches_plan"] for phase in phases),
            "steps_match_plan": all(phase["plan_checks"]["steps_match_plan"] for phase in phases),
            "samples_match_plan": all(phase["plan_checks"]["samples_match_manifest_and_batch_budget"] for phase in phases),
            "phases": [
                {name: phase[name] for name in (
                    "job_id", "length", "seed", "step_count", "optimizer_step_seconds_sum",
                    "nonpad_tokens_sum", "sample_rows_drawn", "sample_rows_unique",
                    "training_loss", "step_window_peak_allocated_bytes", "step_window_peak_reserved_bytes",
                    "batch_distribution", "mlm_mode_counts", "plan_checks")}
                for phase in phases
            ],
        }
        if group["task"] != "mlm" and group["variant"] in {"V2", "V3"}:
            group["design_status"] = "invalid_design_alpha0_bug"
            group["design_note"] = "Retain as an execution record; these losses do not represent the intended gated design."
        if group["task"] == "mlm" and group["variant"] in shared_info["variants"]:
            group["shared_prefix"] = {
                "steps": shared_info["steps"],
                "recorded_under_variant": shared_info["recorded_under_variant"],
                "optimizer_step_seconds_counted_once": shared_info["optimizer_step_seconds"],
                "nonpad_tokens_counted_once": shared_info["nonpad_tokens"],
                "included_in_group_optimizer_steps": group["variant"] == shared_info["recorded_under_variant"],
            }
            if group["variant"] != shared_info["recorded_under_variant"]:
                group["branch_optimizer_step_seconds_including_shared"] = _round_seconds(
                    (shared_info["optimizer_step_seconds"] or 0.0)
                    + (group["optimizer_step_seconds_sum"] or 0.0))
                group["branch_optimizer_steps_including_shared"] = shared_info["steps"] + group["optimizer_steps"]
        output[key] = group
    return output


def build_summary(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    controller_path = root / "runs" / "formal_67m" / "controller.log"
    controller = _controller_summary(controller_path)
    plan_path = root / "configs" / "experiment_plan.json"
    data_manifest_path = root / "data" / "prepared" / "full_67m" / "manifest.json"
    plan = _read_json(plan_path)
    data_manifest = _read_json(data_manifest_path)
    expected_jobs, schedule = _expected_jobs(plan)
    qa_steps = _qa_phase_steps(plan)
    shared = plan["adaptation"].get("shared_prefix", {})
    shared_steps = int(shared.get("steps", 0))
    shared_variants = list(shared.get("variants", []))

    logs_dir = Path(str(controller.get("logs_dir", "")))
    if not logs_dir.is_dir():
        raise AuditError(f"controller-referenced worker log directory does not exist: {logs_dir}")
    worker_paths = sorted(logs_dir.glob("*.log"))
    if not worker_paths:
        raise AuditError(f"no worker logs found in {logs_dir}")
    workers = [_parse_worker_log(path, root) for path in worker_paths]

    controller_return_codes: dict[str, int] = {}
    controller_jobs: list[dict[str, Any]] = []
    for lane in controller.get("lane_results", []):
        for job in lane.get("jobs", []):
            job_id = str(job.get("job_id", ""))
            controller_return_codes[job_id] = int(job.get("return_code", -1))
            controller_jobs.append(job)

    run_summaries = []
    errors: list[str] = []
    seen_job_ids: Counter[str] = Counter()
    for worker in workers:
        job_id = str(worker["scheduler"].get("job_id", "unknown"))
        seen_job_ids[job_id] += 1
        expected = expected_jobs.get(job_id)
        result = worker["result"]
        if expected is None:
            errors.append(f"unplanned worker job: {job_id}")
        else:
            for field in ("variant", "task", "seed"):
                actual = result.get(field)
                if actual is not None and str(actual) != str(expected[field]):
                    errors.append(f"{job_id}: result {field}={actual!r} differs from plan {expected[field]!r}")
        if seen_job_ids[job_id] > 1:
            errors.append(f"duplicate worker job id: {job_id}")
        run_summaries.append(_summarize_run(worker, expected, plan, data_manifest,
                                            qa_steps, shared_steps, controller_return_codes))

    actual_job_ids = set(seen_job_ids)
    expected_job_ids = set(expected_jobs)
    missing_jobs = sorted(expected_job_ids - actual_job_ids)
    extra_jobs = sorted(actual_job_ids - expected_job_ids)
    if missing_jobs:
        errors.append(f"missing planned jobs: {','.join(missing_jobs)}")
    if extra_jobs:
        errors.append(f"unexpected jobs: {','.join(extra_jobs)}")

    starts = [float(run["started"]["started_unix"]) for run in workers]
    ends = [float(run["ended"]["finished_unix"]) for run in workers]
    wave_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in workers:
        wave_groups[int(run["scheduler"].get("wave", 0))].append(run)
    wave_summaries = []
    for wave, wave_runs in sorted(wave_groups.items()):
        wave_start = min(float(run["started"]["started_unix"]) for run in wave_runs)
        wave_end = max(float(run["ended"]["finished_unix"]) for run in wave_runs)
        wave_summaries.append({
            "wave": wave,
            "start_utc": _utc(wave_start),
            "end_utc": _utc(wave_end),
            "duration_seconds": _round_seconds(wave_end - wave_start),
            "job_count": len(wave_runs),
            "job_ids": sorted(str(run["scheduler"].get("job_id")) for run in wave_runs),
        })

    mlm_runs = [run for run in run_summaries if run["task"] == "mlm"]
    mlm_steps = sum(run["step_count"] for run in mlm_runs)
    mlm_tokens = sum(run["nonpad_tokens_sum"] for run in mlm_runs)
    mlm_seconds = math.fsum(run["optimizer_step_seconds_sum"] or 0.0 for run in mlm_runs)
    recorded_under = next((variant for variant in shared_variants
                           if any(run["variant"] == variant and run["task"] == "mlm"
                                  and run["length"] is not None for run in run_summaries)), None)
    shared_stage_ids = [int(stage["max_length"]) for stage in schedule
                        if int(stage["end_step"]) <= shared_steps]
    shared_runs = [run for run in run_summaries
                   if run["task"] == "mlm" and run["variant"] == recorded_under
                   and run["length"] in shared_stage_ids]
    shared_actual_steps = sum(run["step_count"] for run in shared_runs)
    shared_info = {
        "variants": shared_variants,
        "steps": shared_steps,
        "nonpad_tokens_expected": int(shared.get("input_tokens", 0)),
        "nonpad_tokens": sum(run["nonpad_tokens_sum"] for run in shared_runs),
        "optimizer_step_seconds": _round_seconds(math.fsum(
            run["optimizer_step_seconds_sum"] or 0.0 for run in shared_runs)),
        "recorded_under_variant": recorded_under,
        "actual_steps": shared_actual_steps,
        "counted_once": True,
        "matches_plan": shared_actual_steps == shared_steps
                         and sum(run["nonpad_tokens_sum"] for run in shared_runs)
                         == int(shared.get("input_tokens", 0)),
    }

    groups = _group_runs(run_summaries, expected_jobs, shared_info)
    checkpoint_dir = root / "results" / "formal_67m"
    checkpoints = sorted(checkpoint_dir.rglob("*.pt"))
    expected_mlm_steps = int(plan["adaptation"].get("total_executed_steps", 0))
    expected_mlm_tokens = int(plan["adaptation"].get("total_input_tokens", 0))
    expected_downstream_phases = int(plan["downstream"].get("total_training_phases", 0))
    observed_downstream_phases = sum(1 for run in run_summaries if run["task"] != "mlm")
    all_checks = {
        "controller_completed": controller.get("status") == "completed" and controller.get("failure") is None,
        "controller_job_count_matches_worker_logs": int(controller.get("job_count", -1)) == len(workers),
        "expected_job_matrix_matches": not missing_jobs and not extra_jobs and not any(count > 1 for count in seen_job_ids.values()),
        "all_workers_succeeded": all(run["plan_checks"]["worker_success"]
                                      for run in run_summaries),
        "all_step_counts_match_plan": all(run["plan_checks"]["steps_match_plan"] for run in run_summaries),
        "all_batches_match_plan": all(run["plan_checks"]["batch_matches_plan"] for run in run_summaries),
        "all_sample_counts_match_data_manifest": all(
            run["plan_checks"]["samples_match_manifest_and_batch_budget"] for run in run_summaries),
        "mlm_per_step_tokens_match_plan": all(run["plan_checks"]["mlm_tokens_match_plan"] is True
                                               for run in mlm_runs),
        "mlm_modes_match_plan": all(run["plan_checks"]["mlm_mode_matches_plan"] is True
                                     for run in mlm_runs),
        "unique_mlm_steps_match_plan": mlm_steps == expected_mlm_steps,
        "unique_mlm_tokens_match_plan": mlm_tokens == expected_mlm_tokens,
        "shared_prefix_counted_once_and_matches": shared_info["matches_plan"],
        "downstream_phase_count_matches_plan": observed_downstream_phases == expected_downstream_phases,
        "retained_checkpoint_count_matches_final_outputs": len(checkpoints)
            == expected_downstream_phases + len(plan["variants"]),
        "no_parse_errors": not errors,
    }
    audit_status = "ok" if all(all_checks.values()) else ("incomplete" if errors else "plan_mismatches")
    global_step_peak_allocated = max(
        (run["step_window_peak_allocated_bytes"] or 0 for run in run_summaries), default=0) or None
    global_step_peak_reserved = max(
        (run["step_window_peak_reserved_bytes"] or 0 for run in run_summaries), default=0) or None
    post_run_peak_allocated = max((
        _safe_int(run["worker_end_peak_since_last_reset"].get("peak_allocated_bytes")) or 0
        for run in run_summaries), default=0) or None
    post_run_peak_reserved = max((
        _safe_int(run["worker_end_peak_since_last_reset"].get("peak_reserved_bytes")) or 0
        for run in run_summaries), default=0) or None
    wall_start = min(starts) if starts else None
    wall_end = max(ends) if ends else None
    sum_worker_seconds = math.fsum(float(run["ended"]["finished_unix"])
                                   - float(run["started"]["started_unix"]) for run in workers)
    optimizer_seconds_all = math.fsum(run["optimizer_step_seconds_sum"] or 0.0 for run in run_summaries)
    output = {
        "audit_status": audit_status,
        "interpretation": "loss fields are training loss, not evaluation scores; no evaluation metrics are included",
        "source": {
            "controller_log": str(controller_path),
            "worker_logs_dir": str(logs_dir),
            "experiment_plan": str(plan_path),
            "data_manifest": str(data_manifest_path),
        },
        "controller": {
            "status": controller.get("status"),
            "job_count": controller.get("job_count"),
            "failure": controller.get("failure"),
            "maximum_concurrent_tasks": controller.get("maximum_concurrent_tasks"),
            "gpu_preflight": controller.get("gpu_preflight"),
            "worker_log_count": len(workers),
            "successful_worker_count": sum(run["plan_checks"]["worker_success"]
                                            for run in run_summaries),
            "retained_checkpoint_count": len(checkpoints),
            "removed_intermediate_checkpoint_count": len(controller.get("removed_intermediate_checkpoints", [])),
        },
        "wall_clock": {
            "definition": "earliest __WORKER_START__.started_unix to latest __WORKER_END__.finished_unix; excludes controller preflight and post-run cleanup",
            "start_utc": _utc(wall_start),
            "end_utc": _utc(wall_end),
            "worker_span_seconds": _round_seconds(wall_end - wall_start) if wall_start is not None and wall_end is not None else None,
            "sum_of_worker_elapsed_seconds": _round_seconds(sum_worker_seconds),
            "wave_durations": wave_summaries,
        },
        "totals": {
            "observed_optimizer_updates_all_tasks": sum(run["step_count"] for run in run_summaries),
            "observed_optimizer_updates_unique_mlm": mlm_steps,
            "planned_optimizer_updates_unique_mlm": expected_mlm_steps,
            "observed_mlm_nonpad_tokens_unique": mlm_tokens,
            "planned_mlm_nonpad_tokens_unique": expected_mlm_tokens,
            "observed_nonpad_tokens_all_tasks": sum(run["nonpad_tokens_sum"] for run in run_summaries),
            "sum_optimizer_step_seconds_all_jobs": _round_seconds(optimizer_seconds_all),
            "sum_optimizer_step_seconds_mlm_jobs": _round_seconds(mlm_seconds),
            "shared_prefix": shared_info,
            "planned_downstream_phases": expected_downstream_phases,
            "observed_downstream_phases": observed_downstream_phases,
        },
        "gpu_memory": {
            "step_window_peak_allocated_bytes": global_step_peak_allocated,
            "step_window_peak_allocated_gib": round(global_step_peak_allocated / (1024 ** 3), 4) if global_step_peak_allocated else None,
            "step_window_peak_reserved_bytes": global_step_peak_reserved,
            "step_window_peak_reserved_gib": round(global_step_peak_reserved / (1024 ** 3), 4) if global_step_peak_reserved else None,
            "worker_end_peak_since_last_reset_allocated_bytes": post_run_peak_allocated,
            "worker_end_peak_since_last_reset_reserved_bytes": post_run_peak_reserved,
            "scope_note": "Step-window maxima come from training JSONL records; train_nextbert resets CUDA peak counters at each optimizer-step start. Worker after_runpy peaks are only since the last reset, so neither source proves a process-wide peak during model/checkpoint initialization.",
        },
        "by_variant_task": groups,
        "job_matrix": {
            "planned_jobs": len(expected_jobs),
            "observed_worker_logs": len(workers),
            "missing_job_ids": missing_jobs,
            "extra_job_ids": extra_jobs,
            "duplicate_job_ids": sorted(job_id for job_id, count in seen_job_ids.items() if count > 1),
        },
        "plan_checks": all_checks,
        "issues": errors,
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="project root (default: parent of this tools directory)")
    args = parser.parse_args(argv)
    try:
        output = build_summary(args.root)
    except AuditError as exc:
        output = {"audit_status": "incomplete", "issues": [str(exc)]}
        print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
        return 2
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return 0 if output["audit_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
