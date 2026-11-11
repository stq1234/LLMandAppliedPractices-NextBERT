"""Measure real-batch MLM and downstream training calibration, offline only.

Default calibration performs two warm-up and three measured optimizer updates
for the requested phases.  It writes projections separately from observations
and never starts the full training schedule.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from train_nextbert import (
    PACKAGE_ROOT, _amp_context, _device, _epoch_batch, _epoch_sample_indices,
    _load_checkpoint, _prepare_downstream_batch, _save_checkpoint, _scheduler,
    _seed_everything, _special_ids, adaptation_stage_schedule,
    choose_retrieval_physical_batch, collate_sequences,
    dataset_fingerprint, load_jsonl, load_nextbert, optimizer_for, pair_tokens,
    qa_phase_steps, read_plan, retrieval_to_reranking_records, train_mlm_step,
    train_retrieval_step, train_supervised_step, validate_mlm_split_sidecar,
)
from evaluate_nextbert import evaluate_reranking, evaluate_retrieval


VARIANTS = ("V0", "V1", "V2", "V3")


def _map_assignments(values: Sequence[str], *, required: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH assignment, got {value!r}")
        name, path = value.split("=", 1)
        if name not in required:
            raise ValueError(f"unexpected name {name!r}; expected {list(required)}")
        result[name] = Path(path).expanduser().resolve()
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"missing local assignments for: {', '.join(missing)}")
    return result


def _task_phases(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for entry in plan["downstream"]["run_matrix"]:
        for variant in entry["variants"]:
            for task in entry["tasks"]:
                for seed in entry["seeds"]:
                    phases.append({"variant": variant, "task": task, "seed": int(seed)})
    if len(phases) != int(plan["downstream"]["total_training_phases"]):
        raise ValueError("experiment_plan run_matrix phase count is inconsistent")
    return phases


def _measured_record(record: Any, *, ordinal: int, warmup_steps: int,
                     mode: str | None = None) -> dict[str, Any]:
    result = record.to_dict()
    result["pilot_ordinal"] = ordinal
    result["pilot_role"] = "warmup" if ordinal <= warmup_steps else "measured"
    result["mlm_mode"] = mode
    return result


def _stable_gate_path(model: torch.nn.Module) -> None:
    setter = getattr(model, "set_gate_ramp", None)
    if callable(setter):
        setter(1.0)


def _device_memory_context(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda_memory": "not_available"}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "visible_cuda_ordinal": device.index,
        "free_bytes_before_pilot": int(free),
        "total_bytes": int(total),
        "free_gib_before_pilot": round(free / (1024 ** 3), 3),
        "free_margin_at_least_6gib": free >= 6 * (1024 ** 3),
        "note": "device free memory includes other visible processes at sampling time",
    }


def _save_resume_probe(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Any, task: str, variant: str, seed: int,
                       output_dir: Path, device: torch.device) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / ".pilot_save_resume_probe.pt"
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    watched = min(parameters, key=lambda parameter: parameter.numel())
    expected = watched.detach().clone()
    save_started = time.perf_counter()
    _save_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                     step=5, task=task, variant=variant, seed=seed,
                     metadata={"pilot_save_resume_probe": True})
    save_seconds = time.perf_counter() - save_started
    with torch.no_grad():
        watched.add_(1.0)
    load_started = time.perf_counter()
    restored_step = _load_checkpoint(checkpoint, model=model, optimizer=optimizer,
                                     scheduler=scheduler, task=task, variant=variant,
                                     map_location=device)
    load_seconds = time.perf_counter() - load_started
    restored = torch.equal(expected, watched.detach())
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint.unlink()
    if not restored or restored_step != 5:
        raise AssertionError("pilot save/resume failed to restore model parameters and optimizer step")
    return {"status": "passed", "restored_step": restored_step,
            "parameter_restore_exact": restored, "checkpoint_bytes": checkpoint_bytes,
            "save_seconds": save_seconds, "load_seconds": load_seconds,
            "timed_optimizer_steps_include_checkpoint_io": False}


def _mlm_case(*, variant: str, length: int, mode: str, model_dir: Path,
              data_path: Path, device: torch.device, output_dir: Path,
              warmup_steps: int, measured_steps: int, seed: int,
              save_resume_probe: bool) -> dict[str, Any]:
    rows = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if rows.dtype != np.int32 or rows.ndim != 2 or rows.shape[1] != length or len(rows) == 0:
        raise ValueError(f"{data_path} must be nonempty int32 [N,{length}]")
    split_provenance = validate_mlm_split_sidecar(data_path, len(rows))
    plan = read_plan()
    stage = next(item for item in plan["adaptation"]["stages"] if item["max_length"] == length)
    physical = int(stage["physical_microbatch"])
    accumulation = int(stage["gradient_accumulation"])
    _seed_everything(seed)
    model = load_nextbert(model_dir, variant, device)
    model.train()
    _stable_gate_path(model)
    optimizer = optimizer_for(
        model, backbone_lr=float(plan["adaptation"]["optimizer"]["backbone_lr"]),
        gate_lr=float(plan["adaptation"]["optimizer"]["gate_lr"]),
        weight_decay=float(plan["adaptation"]["optimizer"]["weight_decay"]),
    )
    total = warmup_steps + measured_steps
    scheduler = _scheduler(optimizer, total,
                           float(plan["adaptation"]["optimizer"]["warmup_fraction"]))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config.get("vocab_size", 256000))
    mask_token_id = int(config.get("mask_token_id", 4))
    special_ids = getattr(model, "special_token_ids", None) or _special_ids()
    step_records: list[dict[str, Any]] = []
    sampled_indices: list[int] = []
    save_resume: dict[str, Any] | None = None
    for index in range(total):
        chosen = _epoch_sample_indices(len(rows), physical * accumulation,
                                       seed=seed, offset=index * physical * accumulation)
        sampled_indices.extend(int(row) for row in chosen)
        batches = []
        for offset in range(0, len(chosen), physical):
            ids = torch.from_numpy(np.asarray(rows[chosen[offset:offset + physical]], dtype=np.int64)).to(device)
            batches.append((ids, ids.ne(0).long()))
        effective_mode = ("token" if index % 2 == 0 else "span") if mode == "balanced" else mode
        record = train_mlm_step(
            model, optimizer, batches, device=device, vocab_size=vocab_size,
            special_ids=special_ids, mask_token_id=mask_token_id,
            seed=seed + index * accumulation, mode=effective_mode, accumulation=accumulation,
            gate_ramp_updates=0, optimizer_step=index,
        )
        scheduler.step()
        step_records.append(_measured_record(record, ordinal=index + 1,
                                             warmup_steps=warmup_steps, mode=mode))
        step_records[-1]["objective_mode_this_step"] = effective_mode
    if save_resume_probe:
        save_resume = _save_resume_probe(
            model=model, optimizer=optimizer, scheduler=scheduler, task="mlm",
            variant=variant, seed=seed, output_dir=output_dir, device=device)
    result = {
        "variant": variant, "task": "mlm", "length": length, "mode": mode,
        "physical_microbatch": physical, "gradient_accumulation": accumulation,
        "effective_sequences": physical * accumulation,
        "warmup_steps": warmup_steps, "measured_steps": measured_steps,
        "gate_ramp": 1.0, "data_file": str(data_path),
        "data_rows": int(len(rows)), "data_sha256": dataset_fingerprint(data_path),
        "split_provenance": split_provenance,
        "sampler": {"seeded_epoch_permutation": True,
                    "drawn_rows": len(sampled_indices),
                    "unique_rows": len(set(sampled_indices)),
                    "repeated_draws": len(sampled_indices) - len(set(sampled_indices))},
        "steps": step_records,
    }
    if save_resume is not None:
        result["save_resume_probe"] = save_resume
    del model, optimizer, scheduler, rows
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _phase_optimizer(model: torch.nn.Module, task: str, plan: Mapping[str, Any]):
    task_config = plan["downstream"][task]
    backbone_lr = float(task_config["backbone_lr"])
    return optimizer_for(
        model, backbone_lr=backbone_lr, gate_lr=backbone_lr,
        head_lr=float(task_config.get("head_lr", plan["downstream"]["shared"]["head_lr"])),
        weight_decay=float(task_config.get("weight_decay", plan["downstream"]["shared"]["weight_decay"])),
    )


def _prepare_reranking(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return retrieval_to_reranking_records(records, max_length=1024)


def _qa_groups(rows: Sequence[Mapping[str, Any]], max_regular_length: int = 512) -> dict[str, list[Mapping[str, Any]]]:
    regular, long = [], []
    for row in rows:
        if len(row["input_ids"]) > max_regular_length or row.get("window_type") == "long":
            long.append(row)
        else:
            regular.append(row)
    return {"regular": regular, "long": long}


def _target_count(task: str, subphase: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    downstream = plan["downstream"]
    epochs = int(downstream.get("shared", {}).get("epochs", 2))
    cfg = downstream[task]
    if task == "classification":
        per_language = int(cfg.get("train_per_language", 0))
        languages = int(cfg.get("language_count", len(cfg.get("sources", ("zh", "en")))))
        per_epoch = per_language * languages
        unit = "examples"
    elif task == "qa":
        languages = int(cfg.get("language_count", len(cfg.get("sources", ("zh", "en")))))
        if subphase == "regular":
            per_epoch = int(cfg.get("train_questions_per_language", 0)) * languages
        else:
            long_cfg = cfg.get("long_augmentation", {})
            regular_per_epoch = int(cfg.get("train_questions_per_language", 0)) * languages
            per_epoch = int(long_cfg.get(
                "train_questions_total",
                math.ceil(regular_per_epoch * float(long_cfg.get("fraction_of_regular_training_windows", 0.0))),
            ))
        unit = "questions"
    elif task == "retrieval":
        languages = int(cfg.get("language_count", len(cfg.get("sources", ("zh", "en")))))
        per_epoch = int(cfg.get("train_queries_per_language", 0)) * languages
        unit = "queries"
    elif task == "reranking":
        per_epoch = int(cfg.get("train_queries_total", 0))
        unit = "query_groups"
    else:
        return {"target_count_per_epoch": None, "target_count_total_epochs": None,
                "target_count_unit": None, "epochs": epochs}
    return {"target_count_per_epoch": per_epoch,
            "target_count_total_epochs": per_epoch * epochs,
            "target_count_unit": unit, "epochs": epochs}


def _task_batch_signature(task: str, subphase: str,
                          plan: Mapping[str, Any],
                          retrieval_document_batch_override: int | None = None) -> dict[str, int]:
    cfg = plan["downstream"][task]
    if task == "qa" and subphase == "long":
        cfg = cfg["long_augmentation"]
    if task == "retrieval":
        logical = int(cfg["logical_batch_queries"])
        query_physical = int(cfg["physical_query_batch"])
        document_physical = (retrieval_document_batch_override
                             if retrieval_document_batch_override is not None
                             else int(cfg["physical_document_batch"]))
        return {"physical_batch": query_physical, "gradient_accumulation": 1,
                "effective_batch": logical, "logical_retrieval_queries": logical,
                "physical_query_batch": query_physical,
                "physical_document_batch": document_physical}
    if task == "reranking":
        physical = int(cfg["physical_query_groups"])
        accumulation = int(cfg["accumulation"])
        pairs_per_group = 1 + int(cfg["negatives_per_positive"])
        return {"physical_batch": physical, "gradient_accumulation": accumulation,
                "effective_batch": physical * accumulation,
                "physical_pairs": physical * pairs_per_group,
                "effective_pairs": physical * accumulation * pairs_per_group}
    physical = int(cfg["physical_batch"])
    accumulation = int(cfg["accumulation"])
    return {"physical_batch": physical, "gradient_accumulation": accumulation,
            "effective_batch": int(cfg.get("effective_batch", physical * accumulation))}


def _task_case(*, phase: Mapping[str, Any], subphase: str,
               model_dir: Path, data_path: Path, device: torch.device,
               warmup_steps: int, measured_steps: int,
               save_resume_probe: bool, output_dir: Path,
               sample_seed: int,
               retrieval_physical_document_batch: int | None = None) -> dict[str, Any]:
    plan = read_plan()
    task, variant, seed = str(phase["task"]), str(phase["variant"]), int(phase["seed"])
    source_rows = load_jsonl(data_path)
    batch_signature = _task_batch_signature(task, subphase, plan,
                                            retrieval_physical_document_batch)
    if task == "qa":
        max_regular_length = int(plan["downstream"]["qa"]["max_length"])
        rows = _qa_groups(source_rows, max_regular_length)[subphase]
        physical = batch_signature["physical_batch"]
        accumulation = batch_signature["gradient_accumulation"]
    elif task == "classification":
        rows = source_rows
        physical = batch_signature["physical_batch"]
        accumulation = batch_signature["gradient_accumulation"]
    elif task == "retrieval":
        rows = source_rows
        physical = batch_signature["physical_query_batch"]
        accumulation = batch_signature["gradient_accumulation"]
    else:
        rows = _prepare_reranking(source_rows)
        physical = batch_signature["physical_batch"]
        accumulation = batch_signature["gradient_accumulation"]
    if not rows:
        return {"variant": variant, "task": task, "seed": seed,
                "subphase": subphase, "status": "unmeasured_no_prepared_rows",
                "data_file": str(data_path), "data_sha256": dataset_fingerprint(data_path),
                "warmup_steps": warmup_steps, "measured_steps": 0, "steps": []}
    _seed_everything(seed)
    model = load_nextbert(model_dir, variant, device)
    model.train()
    _stable_gate_path(model)
    optimizer = _phase_optimizer(model, task, plan)
    schedule_steps = warmup_steps + measured_steps
    scheduler = _scheduler(optimizer, schedule_steps,
                           float(plan["downstream"]["shared"]["warmup_fraction"]))
    step_records: list[dict[str, Any]] = []
    sampled_row_indices: list[int] = []
    for index in range(schedule_steps):
        if task == "retrieval":
            logical_queries = batch_signature["logical_retrieval_queries"]
            selected_indices = _epoch_sample_indices(len(rows), logical_queries, seed=sample_seed,
                                                     offset=index * logical_queries)
            sampled_row_indices.extend(int(value) for value in selected_indices)
            logical_rows = [rows[int(value)] for value in selected_indices]
            physical_docs = choose_retrieval_physical_batch(
                logical_rows, override=batch_signature["physical_document_batch"])
            record = train_retrieval_step(
                model, optimizer, logical_rows, device=device, step=index,
                physical_document_batch=physical_docs,
                physical_query_batch=batch_signature["physical_query_batch"], temperature=0.05,
            )
        elif task == "reranking":
            group_count = physical * accumulation
            selected_indices = _epoch_sample_indices(len(rows), group_count, seed=sample_seed,
                                                     offset=index * group_count)
            sampled_row_indices.extend(int(value) for value in selected_indices)
            groups = [rows[int(value)] for value in selected_indices]
            microbatches = []
            for offset in range(0, len(groups), physical):
                selected = groups[offset:offset + physical]
                microbatches.append({
                    "input_ids": [pair for group in selected for pair in group["input_ids"]],
                    "labels": [0] * len(selected),
                    "nonpad_tokens": sum(len(pair) for group in selected for pair in group["input_ids"]),
                })
            record = train_supervised_step(model, optimizer, task, microbatches,
                                           device=device, step=index, accumulation=accumulation)
        else:
            selected_indices = _epoch_sample_indices(len(rows), physical * accumulation,
                                                     seed=sample_seed,
                                                     offset=index * physical * accumulation)
            sampled_row_indices.extend(int(value) for value in selected_indices)
            batch_rows = [rows[int(value)] for value in selected_indices]
            microbatches = [
                _prepare_downstream_batch(task, batch_rows[offset:offset + physical], pad_token_id=0)
                for offset in range(0, len(batch_rows), physical)
            ]
            record = train_supervised_step(model, optimizer, task, microbatches,
                                           device=device, step=index, accumulation=accumulation)
        scheduler.step()
        measured_step = _measured_record(record, ordinal=index + 1,
                                         warmup_steps=warmup_steps)
        if task == "retrieval":
            measured_step["physical_document_batch"] = physical_docs
            measured_step["physical_query_batch"] = batch_signature["physical_query_batch"]
            measured_step["logical_retrieval_queries"] = batch_signature["logical_retrieval_queries"]
        elif task == "reranking":
            measured_step["physical_pairs"] = batch_signature["physical_pairs"]
            measured_step["effective_pairs"] = batch_signature["effective_pairs"]
        step_records.append(measured_step)
    target_info = _target_count(task, subphase, plan)
    if task == "qa":
        question_ids = {str(row["question_id"]) for row in rows if row.get("question_id") is not None}
        source_count = len(question_ids) if question_ids else len(rows)
        source_count_unit = "questions" if question_ids else "feature_rows_without_question_id"
    elif task == "reranking":
        source_count = len(rows)
        source_count_unit = "query_groups"
    elif task == "retrieval":
        source_count = len(rows)
        source_count_unit = "queries"
    else:
        source_count = len(rows)
        source_count_unit = "examples"
    result = {
        "variant": variant, "task": task, "seed": seed, "subphase": subphase,
        "physical_batch": physical, "gradient_accumulation": accumulation,
        "effective_batch": batch_signature["effective_batch"],
        **{key: value for key, value in batch_signature.items()
           if key not in {"physical_batch", "gradient_accumulation", "effective_batch"}},
        "retrieval_physical_document_batch_override": retrieval_physical_document_batch
        if task == "retrieval" else None,
        **target_info, "source_count": source_count,
        "source_count_unit": source_count_unit,
        "source_feature_rows": len(rows) if task == "qa" else None,
        "warmup_steps": warmup_steps, "measured_steps": measured_steps,
        "gate_ramp": 1.0, "data_file": str(data_path),
        "data_rows": len(rows), "data_sha256": dataset_fingerprint(data_path),
        "sampler": {"seeded_epoch_permutation": True,
                    "drawn_rows": len(sampled_row_indices),
                    "unique_rows": len(set(sampled_row_indices)),
                    "repeated_draws": len(sampled_row_indices) - len(set(sampled_row_indices))},
        "steps": step_records,
    }
    if save_resume_probe:
        result["save_resume_probe"] = _save_resume_probe(
            model=model, optimizer=optimizer, scheduler=scheduler, task=task,
            variant=variant, seed=seed, output_dir=output_dir,
            device=device)
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _median_step_seconds(case: Mapping[str, Any], *, mode: str | None = None) -> float | None:
    times = [float(row["optimizer_step_seconds"]) for row in case.get("steps", [])
             if row.get("pilot_role") == "measured" and (mode is None or row.get("mlm_mode") == mode)]
    return statistics.median(times) if times else None


def _adaptation_projection(mlm_cases: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = {(str(row["variant"]), int(row["length"]), str(row["mode"])): row
              for row in mlm_cases}
    stages = adaptation_stage_schedule(plan)
    shared_steps = int(plan["adaptation"].get("shared_prefix", {}).get("steps", 0))
    projections: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for stage in stages:
            length = int(stage["max_length"])
            steps = int(stage["optimizer_steps"])
            shared_stage = (variant in {"V2", "V3"}
                            and int(stage["end_step"]) <= shared_steps)
            if shared_stage:
                execution = "shared_prefix_V2_V3"
                source_variant, requested_mode = "V2", "token"
            else:
                execution = f"{variant}_branch"
                source_variant = variant
                requested_mode = "balanced" if variant == "V3" and int(stage["start_step"]) >= shared_steps else "token"
            expected = {"physical_microbatch": int(stage["physical_microbatch"]),
                        "gradient_accumulation": int(stage["gradient_accumulation"]),
                        "effective_sequences": int(stage.get(
                            "effective_sequences", int(stage["physical_microbatch"])
                            * int(stage["gradient_accumulation"]))) }

            def matching_case(mode_name: str) -> tuple[Mapping[str, Any] | None, bool]:
                candidate = lookup.get((source_variant, length, mode_name))
                if candidate is None:
                    return None, False
                candidate_signature = {key: candidate.get(key) for key in expected}
                return candidate, any(candidate_signature.get(key) != value
                                      for key, value in expected.items())

            if variant == "V3" and not shared_stage and int(stage["start_step"]) >= shared_steps:
                balanced_case, balanced_mismatch = matching_case("balanced")
                token_case, token_mismatch = matching_case("token")
                span_case, span_mismatch = matching_case("span")
                if balanced_case is not None and not balanced_mismatch:
                    mode = "balanced"
                    cases_for_projection = {"balanced": balanced_case}
                    measured = _median_step_seconds(balanced_case, mode="balanced")
                    measured_count = sum(1 for row in balanced_case.get("steps", [])
                                         if row.get("pilot_role") == "measured" and row.get("mlm_mode") == "balanced")
                    batch_mismatch = False
                elif token_case is not None and span_case is not None and not token_mismatch and not span_mismatch:
                    mode = "balanced_token_span_50_50"
                    cases_for_projection = {"token": token_case, "span": span_case}
                    token_median = _median_step_seconds(token_case, mode="token")
                    span_median = _median_step_seconds(span_case, mode="span")
                    measured = ((token_median + span_median) / 2.0
                                if token_median is not None and span_median is not None else None)
                    token_count = sum(1 for row in token_case.get("steps", [])
                                      if row.get("pilot_role") == "measured" and row.get("mlm_mode") == "token")
                    span_count = sum(1 for row in span_case.get("steps", [])
                                     if row.get("pilot_role") == "measured" and row.get("mlm_mode") == "span")
                    measured_count = min(token_count, span_count)
                    batch_mismatch = False
                else:
                    mode = "balanced_token_span_50_50"
                    cases_for_projection = {key: value for key, value in
                                            (("balanced", balanced_case), ("token", token_case), ("span", span_case))
                                            if value is not None}
                    measured = None
                    measured_count = 0
                    batch_mismatch = any((case is not None and mismatch) for case, mismatch in
                                         ((balanced_case, balanced_mismatch),
                                          (token_case, token_mismatch), (span_case, span_mismatch)))
            else:
                mode = requested_mode
                case, batch_mismatch = matching_case(mode)
                cases_for_projection = {mode: case} if case is not None else {}
                measured = None if case is None or batch_mismatch else _median_step_seconds(case, mode=mode)
                measured_count = (sum(1 for row in case.get("steps", [])
                                      if row.get("pilot_role") == "measured" and row.get("mlm_mode") == mode)
                                  if case is not None and not batch_mismatch else 0)
            observed = ({mode_name: {key: case.get(key) for key in expected}
                         for mode_name, case in cases_for_projection.items()}) or None
            projections.append({
                "variant": variant, "length": length, "planned_optimizer_steps": steps,
                "absolute_start_step": int(stage["start_step"]),
                "absolute_end_step": int(stage["end_step"]), "objective_mode": mode,
                "physical_microbatch": int(stage["physical_microbatch"]),
                "gradient_accumulation": int(stage["gradient_accumulation"]),
                "execution_accounting": execution,
                "measured_pilot_step_seconds_median": measured,
                "measured_pilot_sample_count": measured_count,
                "measured_mode_sample_counts": {
                    mode_name: sum(1 for row in case.get("steps", [])
                                   if row.get("pilot_role") == "measured" and row.get("mlm_mode") == mode_name)
                    for mode_name, case in cases_for_projection.items()},
                "batch_signature_match": bool(cases_for_projection) and not batch_mismatch,
                "expected_batch_signature": expected, "observed_batch_signature": observed,
                "projected_seconds": measured * steps if measured is not None else None,
                "projection_status": ("projected_from_measured_pilot" if measured is not None
                                       else "unmeasured_batch_mismatch" if batch_mismatch
                                       else "unmeasured_missing_pilot"),
            })
    return projections


def _task_source_counts(task: str, subphase: str, task_data: Mapping[str, Any],
                        plan: Mapping[str, Any]) -> dict[str, Any]:
    if task == "qa":
        rows = task_data.get("qa_rows", [])
        regular_limit = int(plan["downstream"]["qa"]["max_length"])
        phase_rows = _qa_groups(rows, regular_limit).get(subphase, [])
        ids = {str(row["question_id"]) for row in phase_rows if row.get("question_id") is not None}
        return {"source_count": len(ids) if ids else len(phase_rows),
                "source_count_unit": "questions" if ids else "feature_rows_without_question_id",
                "source_feature_rows": len(phase_rows)}
    if task == "classification":
        rows = task_data.get("classification_rows", [])
        return {"source_count": len(rows), "source_count_unit": "examples"}
    if task == "retrieval":
        rows = task_data.get("retrieval_rows", [])
        return {"source_count": len(rows), "source_count_unit": "queries"}
    if task == "reranking":
        rows = task_data.get("retrieval_rows", [])
        return {"source_count": len(rows), "source_count_unit": "query_groups"}
    return {"source_count": None, "source_count_unit": None}


def _planned_task_steps(task: str, task_data: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    downstream = plan["downstream"]
    epochs = int(downstream.get("shared", {}).get("epochs", 2))
    if task == "qa":
        counts = qa_phase_steps(plan)
        return {phase: {"planned_optimizer_steps": steps,
                        **_target_count(task, phase, plan),
                        **_task_source_counts(task, phase, task_data, plan)}
                for phase, steps in counts.items()}
    cfg = downstream[task]
    target = _target_count(task, "all", plan)
    if "steps" in cfg:
        steps = int(cfg["steps"])
    else:
        physical = int(cfg.get("effective_batch", 1))
        steps = math.ceil(int(target["target_count_total_epochs"] or 0) / physical)
    return {"all": {"planned_optimizer_steps": steps, **target,
                    **_task_source_counts(task, "all", task_data, plan)}}


def _task_case_matches_plan(case: Mapping[str, Any], task: str, subphase: str,
                            plan: Mapping[str, Any]) -> tuple[bool, dict[str, int], dict[str, Any]]:
    expected = _task_batch_signature(task, subphase, plan)
    keys = ["physical_batch", "gradient_accumulation", "effective_batch"]
    if task == "retrieval":
        keys += ["logical_retrieval_queries", "physical_query_batch", "physical_document_batch"]
    elif task == "reranking":
        keys += ["physical_pairs", "effective_pairs"]
    observed = {key: case.get(key) for key in keys}
    matched = all(observed.get(key) == expected.get(key) for key in keys)
    return matched, {key: int(expected[key]) for key in keys}, observed


def _downstream_projection(phases: Sequence[Mapping[str, Any]],
                           task_cases: Sequence[Mapping[str, Any]],
                           task_data: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    lookup = {(case["variant"], case["task"], int(case["seed"]), case["subphase"]):
              case for case in task_cases}
    projections: list[dict[str, Any]] = []
    for phase_id, phase in enumerate(phases, 1):
        task = str(phase["task"])
        substeps = _planned_task_steps(task, task_data, plan)
        for subphase, target in substeps.items():
            case = lookup.get((phase["variant"], task, int(phase["seed"]), subphase))
            batch_match, expected_batch, observed_batch = (
                _task_case_matches_plan(case, task, subphase, plan) if case is not None
                else (False, _task_batch_signature(task, subphase, plan), None))
            measured = _median_step_seconds(case) if case is not None and batch_match else None
            status = ("projected_from_real_batch_pilot" if measured is not None
                      and target["planned_optimizer_steps"] else
                      "unmeasured_batch_mismatch" if case is not None and not batch_match
                      else "unmeasured_missing_data_or_pilot")
            projections.append({
                "phase_id": phase_id, "variant": phase["variant"], "task": task,
                "seed": phase["seed"], "subphase": subphase,
                **target,
                "expected_batch_signature": expected_batch,
                "observed_batch_signature": observed_batch,
                "batch_signature_match": batch_match,
                "measured_pilot_step_seconds_median": measured,
                "measured_pilot_sample_count": sum(
                    1 for case in task_cases
                    if case["variant"] == phase["variant"] and case["task"] == task and
                    int(case["seed"]) == int(phase["seed"]) and case["subphase"] == subphase
                    and _task_case_matches_plan(case, task, subphase, plan)[0]
                    for row in case.get("steps", []) if row.get("pilot_role") == "measured"),
                "projected_seconds": measured * target["planned_optimizer_steps"]
                if measured is not None and target["planned_optimizer_steps"] else None,
                "projection_status": status,
            })
    return projections


def _evaluation_work(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluation = plan["evaluation"]
    return [
        {"stage": "classification_validation_and_official_test", "languages": ["zh", "en"],
         "variants": ["V0", "V3"],
         "runs": sum(phase["task"] == "classification" for phase in _task_phases(plan)),
         "measured_seconds": None,
         "projected_seconds": None, "status": "unmeasured", "gpu_hour_cap": None},
        {"stage": "qa_article_grouped_selection_and_locked_final_eval", "sources": ["SQuAD1.1", "CMRC2018"],
         "variants": ["V0", "V1", "V2", "V3"], "measured_seconds": None,
         "projected_seconds": None, "status": "unmeasured", "gpu_hour_cap": None},
        {"stage": "retrieval_index_build_and_dev_test", "languages": ["zh", "en"],
         "documents_per_language": evaluation["retrieval"]["corpus_documents_per_language"],
         "dev_queries_per_language": evaluation["retrieval"]["dev_queries_per_language"],
         "test_queries_per_language": evaluation["retrieval"]["test_queries_per_language"],
         "variant_indexes": 4, "measured_seconds": None, "projected_seconds": None,
         "status": "unmeasured", "gpu_hour_cap": None},
        {"stage": "reranking_fixed_bm25_top20", "languages": ["zh", "en"],
         "queries_per_language": 200, "variants": ["V0", "V3"],
         "measured_seconds": None, "projected_seconds": None, "status": "unmeasured", "gpu_hour_cap": None},
        {"stage": "long_context_evidence_interventions", "source_groups_per_language": 100,
         "lengths": evaluation["long_context"]["lengths"],
         "evidence_positions": evaluation["long_context"]["evidence_positions"],
         "interventions": evaluation["long_context"]["interventions"],
         "measured_seconds": None, "projected_seconds": None, "status": "unmeasured", "gpu_hour_cap": None},
        {"stage": "retention_probes", "references": ["pretrained", "V0", "V3", "retrieval_adapted"],
         "measured_seconds": None, "projected_seconds": None, "status": "unmeasured", "gpu_hour_cap": None},
        {"stage": "performance_and_index_storage", "lengths": evaluation["performance"]["lengths"],
         "physical_batches": evaluation["performance"]["physical_batches"],
         "measures": evaluation["performance"]["measures"], "measured_seconds": None,
         "projected_seconds": None, "status": "unmeasured", "gpu_hour_cap": None},
    ]


def _evaluation_cost_projection(eval_cases: Sequence[Mapping[str, Any]],
                               plan: Mapping[str, Any]) -> dict[str, Any]:
    index_total = 0.0
    index_rows = []
    query_encoding_total = 0.0
    query_projection_complete = True
    rerank_total = 0.0
    rerank_rows = []
    for case in eval_cases:
        if case.get("task") == "retrieval_index_probe":
            language_results = case.get("result", {}).get("languages", [])
            language_index_projection = [row.get("projected_5000_document_index_seconds_from_bucket_rates")
                                        for row in language_results]
            projection_known = (len(language_results) == 2
                                and all(value is not None for value in language_index_projection))
            projected_index = sum(
                float(row["projected_5000_document_index_seconds_from_bucket_rates"])
                for row in language_results
                if row.get("projected_5000_document_index_seconds_from_bucket_rates") is not None)
            query_count = sum(int(row.get("queries_evaluated", 0)) for row in language_results)
            query_seconds = sum(float(row.get("query_encoding_seconds_mean", 0.0))
                                * int(row.get("queries_evaluated", 0)) for row in language_results)
            per_query = query_seconds / query_count if query_count else None
            target_queries_per_language = (
                int(plan["evaluation"]["retrieval"]["dev_queries_per_language"])
                + int(plan["evaluation"]["retrieval"]["test_queries_per_language"]))
            query_projection = per_query * target_queries_per_language * 2 if per_query is not None else None
            query_projection_complete &= query_projection is not None
            index_total += projected_index
            query_encoding_total += query_projection or 0.0
            index_rows.append({"variant": case["variant"],
                               "measured_index_sample_seconds": case["result"].get("total_index_build_seconds"),
                               "projected_5000_documents_per_language_seconds":
                                   projected_index if projection_known else None,
                               "projected_5000_vector_storage_bytes_per_language":
                                   [row.get("projected_5000_vector_storage_bytes") for row in language_results],
                               "query_encoding_sample_count": query_count,
                               "projected_600_queries_total_encode_seconds": query_projection,
                               "projection_source": "real local corpus/query sample"})
        elif case.get("task") == "reranking_probe":
            result = case.get("result", {})
            by_language = result.get("by_language", {})
            reranking_plan = plan["evaluation"]["reranking"]
            query_target = int(reranking_plan["test_queries_per_language"]) + int(reranking_plan.get("dev_queries_per_language", 0))
            projected = sum(float(value.get("mean_query_end_to_end_seconds", 0.0))
                            * query_target
                            for value in by_language.values())
            rerank_total += projected
            rerank_rows.append({"variant": case["variant"],
                                "sample_queries": result.get("query_count"),
                                "measured_sample_query_latency_seconds": result.get("mean_query_end_to_end_seconds"),
                                "projected_dev_and_test_queries_seconds": projected,
                                "candidate_count_mean": result.get("candidate_count_mean"),
                                "projection_source": "real local BM25 candidate rows"})
    index_complete = len(index_rows) == len(VARIANTS) and all(
        row.get("projected_5000_documents_per_language_seconds") is not None for row in index_rows)
    rerank_complete = len(rerank_rows) == 2
    return {
        "index_build_by_variant": index_rows,
        "index_projection_variant_coverage": len(index_rows) / len(VARIANTS),
        "projected_index_seconds_for_four_variants": index_total if index_complete else None,
        "projected_retrieval_query_encode_seconds_for_four_variants":
            query_encoding_total if index_complete and query_projection_complete else None,
        "reranking_by_variant": rerank_rows,
        "reranking_projection_variant_coverage": len(rerank_rows) / 2,
        "projected_reranking_seconds": rerank_total if rerank_complete else None,
        "is_partial": not (index_complete and rerank_complete),
        "evaluation_costs_not_measured_by_these_probes": [
            "classification and QA official-split inference",
            "long-context evidence interventions",
            "retention probes",
            "512/4096/8192 latency p50/p95 sweep",
            "BM25 candidate generation if not included in supplied prepared files",
        ],
    }


def _run_evaluation_probes(*, variants: Mapping[str, Path], device: torch.device,
                           selected_variants: Sequence[str],
                           corpus_path: Path | None, retrieval_query_path: Path | None,
                           reranking_path: Path | None,
                           sample_documents: int, sample_rerank_queries: int,
                           output_dir: Path) -> list[dict[str, Any]]:
    if corpus_path is None and (retrieval_query_path is not None or reranking_path is not None):
        raise ValueError("evaluation query/reranking probes require --evaluation-corpus")
    if corpus_path is None:
        return []
    corpus = load_jsonl(corpus_path)
    retrieval_queries = load_jsonl(retrieval_query_path) if retrieval_query_path else []
    reranking_rows = load_jsonl(reranking_path) if reranking_path else []
    output: list[dict[str, Any]] = []
    for variant in selected_variants:
        _seed_everything(17)
        model = load_nextbert(variants[variant], variant, device)
        model.eval()
        if retrieval_query_path is not None or reranking_path is None:
            started = time.perf_counter()
            result = evaluate_retrieval(
                model, corpus, retrieval_queries, device=device,
                sample_documents_per_language=sample_documents,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output.append({"task": "retrieval_index_probe", "variant": variant,
                           "sample_documents_per_language": sample_documents,
                           "data_file": str(corpus_path),
                           "query_file": str(retrieval_query_path),
                           "elapsed_seconds": time.perf_counter() - started,
                           "result": result})
        if reranking_path is not None and variant in {"V0", "V3"}:
            started = time.perf_counter()
            result = evaluate_reranking(
                model, reranking_rows, device=device,
                sample_queries=sample_rerank_queries, require_20_candidates=True,
                corpus=corpus,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output.append({"task": "reranking_probe", "variant": variant,
                           "sample_queries_total": sample_rerank_queries,
                           "data_file": str(reranking_path), "corpus_file": str(corpus_path),
                           "elapsed_seconds": time.perf_counter() - started,
                           "result": result})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def build_time_projection(mlm_cases: Sequence[Mapping[str, Any]],
                          phases: Sequence[Mapping[str, Any]],
                          task_cases: Sequence[Mapping[str, Any]],
                          task_data: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    adaptation = _adaptation_projection(mlm_cases, plan)
    downstream = _downstream_projection(phases, task_cases, task_data, plan)
    adaptation_projection_seconds = sum(row["projected_seconds"] or 0 for row in adaptation
                                        if row["execution_accounting"] not in {"shared_prefix_V2_V3"} or
                                        row["variant"] == "V2")
    downstream_projection_seconds = sum(row["projected_seconds"] or 0 for row in downstream)
    caps = plan["resources"]["caps_gpu_hours"]
    unmeasured_caps = {
        "evaluation_and_indexing": float(caps["evaluation"]),
        "calibration": float(caps["calibration"]),
        "recovery": float(caps["recovery"]),
        "hard_cap_physical_gpu_hours": float(plan["resources"]["hard_cap_physical_gpu_hours"]),
    }
    return {
        "adaptation_steps_planned": int(plan["adaptation"]["total_executed_steps"]),
        "adaptation_unique_steps_including_shared_prefix": int(plan["adaptation"]["total_executed_steps"]),
        "adaptation_stage_projections": adaptation,
        "adaptation_projected_seconds_from_pilot": adaptation_projection_seconds,
        "adaptation_projection_coverage": sum(row["measured_pilot_sample_count"] > 0 for row in adaptation)
                                           / max(1, len(adaptation)),
        "adaptation_projection_is_partial": any(row["projection_status"] != "projected_from_measured_pilot"
                                                 for row in adaptation),
        "downstream_phase_count": int(plan["downstream"]["total_training_phases"]),
        "downstream_phase_projections": downstream,
        "downstream_projected_seconds_from_real_batch_pilots": downstream_projection_seconds,
        "downstream_projection_coverage": sum(row["measured_pilot_sample_count"] > 0
                                               for row in downstream) / max(1, len(downstream)),
        "downstream_projection_is_partial": any(row["projection_status"] != "projected_from_real_batch_pilot"
                                                for row in downstream),
        "evaluation_and_indexing_work": _evaluation_work(plan),
        "evaluation_and_indexing_measured_seconds": None,
        "unmeasured_plan_caps_gpu_hours": unmeasured_caps,
        "total_projected_training_seconds": adaptation_projection_seconds + downstream_projection_seconds,
        "total_schedule_wall_seconds": None,
        "interpretation": (
            "Pilot durations are observed on short real optimizer runs. Projected training seconds multiply "
            "their measured median step duration by the planned count; they are projections, not completed-run "
            "measurements. Evaluation and index construction remain unmeasured because no such timed run was made."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", action="append", required=True, metavar="VARIANT=PATH",
                        help="provide one local adapted model directory for each selected variant")
    parser.add_argument("--mlm-data-dir", type=Path,
                        help="prepared MLM profile directory; defaults to the profile selected in experiment_plan.json")
    parser.add_argument("--classification-data", type=Path)
    parser.add_argument("--qa-data", type=Path)
    parser.add_argument("--retrieval-data", type=Path)
    parser.add_argument("--retrieval-physical-document-batch", type=int,
                        help="calibration-only override; defaults to the planned physical document batch")
    parser.add_argument("--tasks", nargs="+", choices=("mlm", "classification", "qa", "retrieval", "reranking"),
                        default=["mlm", "classification", "qa", "retrieval", "reranking"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--lengths", nargs="+", type=int, choices=(512, 1024, 4096, 8192),
                        default=[512, 1024, 4096, 8192])
    parser.add_argument("--mlm-modes", nargs="+", choices=("token", "span"),
                        help="optional MLM objective filter; by default profile token for all and span for V3 4K/8K")
    parser.add_argument("--skip-save-resume", action="store_true",
                        help="omit the separate heavyweight checkpoint I/O probe")
    parser.add_argument("--evaluation-only", action="store_true",
                        help="skip all training calibration and run only the requested index/reranking evaluation probes")
    parser.add_argument("--evaluation-corpus", type=Path,
                        help="optional local corpus JSONL for 64-document index probes")
    parser.add_argument("--evaluation-retrieval-queries", type=Path,
                        help="optional local qrels query JSONL for index/query timing")
    parser.add_argument("--evaluation-reranking-data", type=Path,
                        help="optional local fixed BM25 candidate JSONL; expects 20 candidates")
    parser.add_argument("--evaluation-sample-documents", type=int, default=64)
    parser.add_argument("--evaluation-sample-rerank-queries", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "pilot")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def main() -> None:
    pilot_started = time.perf_counter()
    args = _parse_args()
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("warmup steps must be nonnegative and measured steps positive")
    if args.retrieval_physical_document_batch is not None and args.retrieval_physical_document_batch < 2:
        raise ValueError("retrieval physical document batch must be at least 2")
    if args.evaluation_sample_documents < 1 or args.evaluation_sample_rerank_queries < 1:
        raise ValueError("evaluation probe sample sizes must be positive")
    if args.evaluation_only:
        if args.evaluation_corpus is None:
            raise ValueError("--evaluation-only requires --evaluation-corpus")
        if args.evaluation_retrieval_queries is None and args.evaluation_reranking_data is None:
            raise ValueError("--evaluation-only requires retrieval query data, reranking data, or both")
    plan = read_plan()
    profile = str(plan.get("data_profile", ""))
    if args.mlm_data_dir is None:
        args.mlm_data_dir = (PACKAGE_ROOT / "data" / "prepared" / profile if profile
                             else PACKAGE_ROOT / "data" / "prepared")
    variants = _map_assignments(args.model_dir, required=args.variants)
    data_files: dict[str, Path] = {}
    if not args.evaluation_only:
        for task in args.tasks:
            if task == "mlm":
                for length in (int(stage["max_length"]) for stage in plan["adaptation"]["stages"]):
                    path = args.mlm_data_dir / f"wiki_{length}.npy"
                    if not path.is_file():
                        raise FileNotFoundError(f"prepared MLM array missing: {path}")
                    data_files[f"mlm_{length}"] = path
            elif task == "classification":
                if args.classification_data is None:
                    raise ValueError("--classification-data is required when calibration includes classification")
                data_files[task] = args.classification_data.resolve()
            elif task == "qa":
                if args.qa_data is None:
                    raise ValueError("--qa-data is required when calibration includes QA")
                data_files[task] = args.qa_data.resolve()
            elif task in {"retrieval", "reranking"}:
                if args.retrieval_data is None:
                    raise ValueError(f"--retrieval-data is required when calibration includes {task}")
                data_files[task] = args.retrieval_data.resolve()
                data_files["retrieval"] = args.retrieval_data.resolve()
    device = _device(args.device)
    _seed_everything(args.seed)
    memory_context = _device_memory_context(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_log = args.output_dir / "pilot_cases.jsonl"
    case_log.write_text("", encoding="utf-8")

    def persist_case(case: Mapping[str, Any]) -> None:
        with case_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
            stream.flush()

    mlm_cases: list[dict[str, Any]] = []
    task_cases: list[dict[str, Any]] = []
    save_resume_done = False
    if not args.evaluation_only and "mlm" in args.tasks:
        stages_by_length = {int(stage["max_length"]): stage
                            for stage in adaptation_stage_schedule(plan)}
        shared_prefix_steps = int(plan["adaptation"].get("shared_prefix", {}).get("steps", 0))
        for variant in args.variants:
            for length in args.lengths:
                stage = stages_by_length[length]
                modes = list(args.mlm_modes) if args.mlm_modes else ["token"]
                if (args.mlm_modes is None and variant == "V3"
                        and int(stage["start_step"]) >= shared_prefix_steps):
                    modes.append("span")
                for mode in modes:
                    case = _mlm_case(
                        variant=variant, length=length, mode=mode,
                        model_dir=variants[variant], data_path=data_files[f"mlm_{length}"],
                        device=device, output_dir=args.output_dir,
                        warmup_steps=args.warmup_steps, measured_steps=args.measured_steps,
                        seed=args.seed,
                        save_resume_probe=(not args.skip_save_resume and not save_resume_done),
                    )
                    save_resume_done |= "save_resume_probe" in case
                    mlm_cases.append(case)
                    persist_case({"case_kind": "mlm", **case})
    downstream_phases = _task_phases(plan)
    source_data: dict[str, list[dict[str, Any]]] = {}
    if not args.evaluation_only:
        for task in ("classification", "qa", "retrieval"):
            if task in data_files:
                source_data[task] = load_jsonl(data_files[task])
        for phase in downstream_phases:
            task = str(phase["task"])
            if task not in args.tasks or phase["variant"] not in args.variants:
                continue
            data_path = data_files["retrieval"] if task == "reranking" else data_files[task]
            if task == "qa":
                groups = _qa_groups(source_data["qa"],
                                    int(plan["downstream"]["qa"]["max_length"]))
                subphases = [name for name, rows in groups.items() if rows]
                if not subphases:
                    subphases = ["regular"]
            else:
                subphases = ["all"]
            for subphase in subphases:
                case = _task_case(
                    phase=phase, subphase=subphase,
                    model_dir=variants[str(phase["variant"])], data_path=data_path,
                    device=device, warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    save_resume_probe=(not args.skip_save_resume and not save_resume_done),
                    output_dir=args.output_dir,
                    sample_seed=args.seed + int(phase["seed"]),
                    retrieval_physical_document_batch=args.retrieval_physical_document_batch,
                )
                save_resume_done |= "save_resume_probe" in case
                task_cases.append(case)
                persist_case({"case_kind": "downstream", **case})
    evaluation_cases = _run_evaluation_probes(
        variants=variants, selected_variants=args.variants, device=device,
        corpus_path=args.evaluation_corpus,
        retrieval_query_path=args.evaluation_retrieval_queries,
        reranking_path=args.evaluation_reranking_data,
        sample_documents=args.evaluation_sample_documents,
        sample_rerank_queries=args.evaluation_sample_rerank_queries,
        output_dir=args.output_dir,
    )
    for case in evaluation_cases:
        persist_case({"case_kind": "evaluation", **case})
    evaluation_projection = _evaluation_cost_projection(evaluation_cases, plan)
    if save_resume_done:
        probe_file = args.output_dir / ".pilot_save_resume_probe.pt"
        if probe_file.exists():
            probe_file.unlink()
    task_data = {"qa_rows": source_data.get("qa", []),
                 "classification_rows": source_data.get("classification", []),
                 "retrieval_rows": source_data.get("retrieval", [])}
    if args.evaluation_only:
        projection = {"training_calibration_status": "not_run_evaluation_only"}
    else:
        projection = build_time_projection(mlm_cases, downstream_phases, task_cases, task_data, plan)
    projection["evaluation_cost_probes"] = evaluation_projection
    calibrated_eval_components = [evaluation_projection.get("projected_index_seconds_for_four_variants"),
                                  evaluation_projection.get("projected_retrieval_query_encode_seconds_for_four_variants"),
                                  evaluation_projection.get("projected_reranking_seconds")]
    calibrated_eval_subtotal = sum(float(value) for value in calibrated_eval_components if value is not None)
    calibrated_eval_seconds = (sum(float(value) for value in calibrated_eval_components)
                               if all(value is not None for value in calibrated_eval_components) else None)
    projection["calibrated_evaluation_projection_subtotal_seconds"] = calibrated_eval_subtotal
    projection["calibrated_evaluation_and_index_projection_seconds"] = calibrated_eval_seconds
    projection["training_plus_calibrated_evaluation_projection_seconds"] = (
        float(projection["total_projected_training_seconds"]) + calibrated_eval_seconds
        if calibrated_eval_seconds is not None and not args.evaluation_only else None)
    projection["evaluation_projection_is_partial"] = evaluation_projection["is_partial"]
    report = {
        "status": "evaluation_only_completed" if args.evaluation_only else "pilot_completed",
        "configuration": {"warmup_steps": args.warmup_steps,
                           "measured_steps": args.measured_steps,
                           "device": str(device), "offline_only": True,
                           "evaluation_only": args.evaluation_only,
                           "formal_full_matrix_started": False,
                           "stable_gate_ramp": 1.0,
                           "tasks": [] if args.evaluation_only else args.tasks,
                           "variants": args.variants,
                           "lengths": [] if args.evaluation_only else args.lengths,
                           "skip_save_resume": args.skip_save_resume,
                           "evaluation_cost_probe_count": len(evaluation_cases),
                           "memory_context": memory_context},
        "mlm_calibration_cases": mlm_cases,
        "downstream_calibration_cases": task_cases,
        "evaluation_cost_calibration_cases": evaluation_cases,
        "time_projection": projection,
        "pilot_wall_seconds": time.perf_counter() - pilot_started,
    }
    output = args.output_dir / "pilot_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "pilot_steps.jsonl").open("w", encoding="utf-8") as stream:
        for case in mlm_cases + task_cases:
            for step in case.get("steps", []):
                stream.write(json.dumps({"case": {key: case.get(key) for key in
                    ("variant", "task", "length", "mode", "seed", "subphase")},
                    **step}, ensure_ascii=False) + "\n")
    status = "evaluation_only_completed" if args.evaluation_only else "pilot_completed"
    print(json.dumps({"status": status, "evaluation_only": args.evaluation_only,
                      "mlm_cases": len(mlm_cases),
                      "downstream_calibration_cases": len(task_cases),
                      "evaluation_cost_calibration_cases": len(evaluation_cases),
                      "formal_full_matrix_started": False,
                      "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
