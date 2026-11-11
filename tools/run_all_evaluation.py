"""Run the fixed full-data NextBERT evaluation matrix on at most two lanes.

The default is a dry run. Pass ``--execute`` to launch the local evaluation
CLIs; GPU 7 is exposed offline as the only visible CUDA device.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "src" / "evaluate_nextbert.py"
DIAGNOSTICS = ROOT / "src" / "diagnostics_nextbert.py"
MLM_EVALUATOR = ROOT / "tools" / "evaluate_mlm.py"
VARIANT_ORDER = ("V0", "V1", "V2", "V3")
LANES = {"A": ("V0", "V1"), "B": ("V3", "V2")}
TASKS = {
    "V0": ("classification", "qa", "retrieval", "reranking"),
    "V1": ("qa", "retrieval"),
    "V2": ("qa", "retrieval"),
    "V3": ("classification", "qa", "retrieval", "reranking"),
}
TASK_ORDER = ("classification", "qa", "retrieval", "reranking")
SPLITS = ("dev", "test")
METRICS = ("nDCG@10", "Recall@10", "Recall@100")
STATUS_NAME = "evaluation_status.json"


def _csv_variants(value: str) -> tuple[str, ...]:
    variants = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not variants or any(variant not in VARIANT_ORDER for variant in variants):
        raise argparse.ArgumentTypeError("variants must be a comma-separated subset of V0,V1,V2,V3")
    if len(variants) != len(set(variants)):
        raise argparse.ArgumentTypeError("variants must not contain duplicates")
    return variants


def select_variants(include: Sequence[str] | None = None,
                    exclude: Sequence[str] | None = None) -> tuple[str, ...]:
    excluded = set(exclude or ())
    included = set(include or VARIANT_ORDER)
    selected = tuple(variant for variant in VARIANT_ORDER
                     if variant in included and variant not in excluded)
    if not selected:
        raise ValueError("variant selection is empty")
    return selected


def _checkpoint(training_dir: Path, gated_training_dir: Path, variant: str,
                task: str | None = None) -> Path:
    if task is None:
        return training_dir / "mlm" / variant / "8192" / f"{variant}_mlm_checkpoint.pt"
    root = gated_training_dir if variant in {"V2", "V3"} else training_dir
    return root / "downstream" / variant / task / "17" / f"{variant}_{task}_checkpoint.pt"


def build_task_jobs(*, model_dir: Path, data_dir: Path, training_dir: Path,
                    gated_training_dir: Path, output_dir: Path,
                    variants: Sequence[str]) -> list[dict[str, Any]]:
    """Build the ordinary dev/test matrix with deterministic lane assignment."""
    combined_queries = output_dir / "inputs" / "retrieval_queries_dev_test.jsonl"
    corpus = data_dir / "retrieval_corpus.jsonl"
    jobs: list[dict[str, Any]] = []
    for lane, lane_variants in LANES.items():
        for variant in lane_variants:
            if variant not in variants:
                continue
            for task in TASK_ORDER:
                if task not in TASKS[variant]:
                    continue
                checkpoint = _checkpoint(training_dir, gated_training_dir, variant, task)
                if task == "retrieval":
                    output = output_dir / "tasks" / variant / "retrieval_dev_test.json"
                    raw_output = output_dir / ".tmp" / f"{variant}_retrieval_source.json"
                    command = [
                        sys.executable, str(EVALUATOR), "--task", task, "--variant", variant,
                        "--model-dir", str(model_dir), "--checkpoint", str(checkpoint),
                        "--data", str(combined_queries), "--corpus", str(corpus),
                        "--store-index", str(output_dir / "indexes" / f"{variant}_retrieval_index.npy"),
                        "--output", str(raw_output), "--seed", "17", "--device", "cuda",
                    ]
                    data_paths = (data_dir / "retrieval_queries_dev.jsonl",
                                  data_dir / "retrieval_queries_test.jsonl", corpus)
                    job = {
                        "job_id": f"tasks_{variant}_retrieval_dev_test", "phase": "tasks",
                        "lane": lane, "variant": variant, "task": task, "split": "dev_test",
                        "command": command, "output": output, "raw_output": raw_output,
                        "data_paths": data_paths, "checkpoint": checkpoint,
                    }
                else:
                    for split in SPLITS:
                        data_path = data_dir / f"{task}_{split}.jsonl"
                        output = output_dir / "tasks" / variant / f"{task}_{split}.json"
                        command = [
                            sys.executable, str(EVALUATOR), "--task", task, "--variant", variant,
                            "--model-dir", str(model_dir), "--checkpoint", str(checkpoint),
                            "--data", str(data_path), "--output", str(output),
                            "--seed", "17", "--device", "cuda",
                        ]
                        data_paths = (data_path,)
                        if task == "reranking":
                            command.extend(("--corpus", str(corpus), "--require-20-candidates"))
                            data_paths += (corpus,)
                        job = {
                            "job_id": f"tasks_{variant}_{task}_{split}", "phase": "tasks",
                            "lane": lane, "variant": variant, "task": task, "split": split,
                            "command": command, "output": output,
                            "data_paths": data_paths, "checkpoint": checkpoint,
                        }
                        jobs.append(job)
                    continue
                jobs.append(job)
    return jobs


def build_diagnostic_jobs(*, model_dir: Path, data_dir: Path, training_dir: Path,
                          gated_training_dir: Path, output_dir: Path,
                          variants: Sequence[str]) -> list[dict[str, Any]]:
    """Build the fixed long, performance, retention, and probe diagnostics."""
    raw_wiki = ROOT / "artifacts" / "raw" / "full_67m" / "wikipedia"
    jobs: list[dict[str, Any]] = []
    probe_states = {
        "original_V0": ("V0", None),
        "mlm_V0": ("V0", _checkpoint(training_dir, gated_training_dir, "V0")),
        "mlm_V3": ("V3", _checkpoint(training_dir, gated_training_dir, "V3")),
        "retrieval_V0": ("V0", _checkpoint(training_dir, gated_training_dir, "V0", "retrieval")),
        "retrieval_V3": ("V3", _checkpoint(training_dir, gated_training_dir, "V3", "retrieval")),
    }
    for lane, lane_variants in LANES.items():
        for variant in lane_variants:
            if variant not in variants:
                continue
            qa_checkpoint = _checkpoint(training_dir, gated_training_dir, variant, "qa")
            long_output = output_dir / "diagnostics" / "long" / f"{variant}.json"
            long_predictions = long_output.with_name(f"{variant}_predictions.jsonl")
            long_command = [
                sys.executable, str(DIAGNOSTICS), "--mode", "long", "--model-dir", str(model_dir),
                "--tokenizer-path", str(model_dir), "--variant", variant, "--checkpoint", str(qa_checkpoint),
                "--profile", "full_67m", "--qa-data", str(data_dir / "qa_test.jsonl"),
                "--wiki-background-dir", str(raw_wiki), "--max-source-groups", "50",
                "--eval-batch-size", "4", "--output", str(long_output),
                "--predictions-output", str(long_predictions), "--device", "cuda",
            ]
            jobs.append({
                "job_id": f"diagnostics_long_{variant}", "phase": "diagnostics", "lane": lane,
                "variant": variant, "task": "long", "command": long_command,
                "output": long_output, "extra_artifacts": [long_predictions],
                "result_identity": {"diagnostic": "controlled_long_context_evidence", "variant": variant},
                "data_paths": (data_dir / "qa_test.jsonl", raw_wiki / "selected_en.jsonl",
                               raw_wiki / "selected_zh.jsonl"),
                "source_paths": (DIAGNOSTICS, ROOT / "src" / "qa_metrics.py"),
                "checkpoint": qa_checkpoint,
            })

            mlm_checkpoint = _checkpoint(training_dir, gated_training_dir, variant)
            for length in (512, 4096, 8192):
                data_path = data_dir / f"wiki_{length}_test.npy"
                output = output_dir / "diagnostics" / "performance" / f"{variant}_{length}.json"
                command = [
                    sys.executable, str(EVALUATOR), "--task", "performance", "--variant", variant,
                    "--model-dir", str(model_dir), "--checkpoint", str(mlm_checkpoint),
                    "--data", str(data_path), "--length", str(length), "--warmup-steps", "2",
                    "--measured-steps", "10", "--output", str(output), "--seed", "17", "--device", "cuda",
                ]
                jobs.append({
                    "job_id": f"diagnostics_performance_{variant}_{length}", "phase": "diagnostics",
                    "lane": "performance", "variant": variant, "task": "performance", "command": command,
                    "output": output, "data_paths": (data_path,), "checkpoint": mlm_checkpoint,
                })

            retention_states: list[tuple[str, str, Path | None]] = []
            if variant == "V0":
                retention_states.append(("original_V0", "V0", None))
            retention_states.append((f"mlm_{variant}", variant, mlm_checkpoint))
            if variant in {"V0", "V3"}:
                retention_states.append((f"retrieval_{variant}", variant,
                                         _checkpoint(training_dir, gated_training_dir, variant, "retrieval")))
            for state_name, state_variant, checkpoint in retention_states:
                for length in (512, 8192):
                    data_path = data_dir / f"wiki_{length}_test.npy"
                    output = output_dir / "diagnostics" / "mlm_retention" / state_name / f"wiki_{length}_test.json"
                    command = [
                        sys.executable, str(MLM_EVALUATOR), "--model-dir", str(model_dir),
                        "--variant", state_variant, "--data", str(data_path), "--output", str(output),
                        "--seed", "1701", "--batch-size", "4", "--device", "cuda",
                    ]
                    if checkpoint is not None:
                        command.extend(("--checkpoint", str(checkpoint)))
                    jobs.append({
                        "job_id": f"diagnostics_mlm_retention_{state_name}_{length}",
                        "phase": "diagnostics", "lane": lane, "variant": state_variant,
                        "task": "mlm_retention", "command": command, "output": output,
                        "data_paths": (data_path, data_path.with_suffix(".jsonl")),
                        "source_paths": (MLM_EVALUATOR, ROOT / "src" / "train_nextbert.py"),
                        "checkpoint": checkpoint,
                        "result_identity": {"task": "fixed_mask_heldout_mlm", "variant": state_variant},
                    })

            if variant in {"V0", "V3"}:
                for state_name, (state_variant, checkpoint) in probe_states.items():
                    if state_variant != variant:
                        continue
                    output = output_dir / "diagnostics" / "probes" / f"{state_name}.json"
                    heads = output.with_name(f"{output.stem}_heads.pt")
                    command = [
                        sys.executable, str(DIAGNOSTICS), "--mode", "probe", "--model-dir", str(model_dir),
                        "--tokenizer-path", str(model_dir), "--variant", state_variant,
                        "--profile", "full_67m", "--probe-tasks", "classification,qa",
                        "--classification-train", str(data_dir / "classification_train.jsonl"),
                        "--classification-dev", str(data_dir / "classification_dev.jsonl"),
                        "--qa-train", str(data_dir / "qa_train.jsonl"),
                        "--qa-dev", str(data_dir / "qa_dev.jsonl"), "--steps", "100",
                        "--seed", "17", "--output", str(output), "--device", "cuda",
                    ]
                    if checkpoint is not None:
                        command.extend(("--checkpoint", str(checkpoint)))
                    jobs.append({
                        "job_id": f"diagnostics_probe_{state_name}", "phase": "diagnostics",
                        "lane": lane, "variant": state_variant, "task": "probe", "command": command,
                        "output": output, "extra_artifacts": [heads],
                        "result_identity": {"diagnostic": "frozen_head_retention_probe", "variant": state_variant},
                        "data_paths": (data_dir / "classification_train.jsonl", data_dir / "classification_dev.jsonl",
                                       data_dir / "qa_train.jsonl", data_dir / "qa_dev.jsonl"),
                        "source_paths": (DIAGNOSTICS, ROOT / "src" / "qa_metrics.py"),
                        "checkpoint": checkpoint,
                    })
    return jobs


def combine_retrieval_queries(dev_path: Path, test_path: Path,
                              destination: Path) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    """Write dev then test queries and reject duplicate (language, id) keys."""
    source_map: dict[tuple[str, str], str] = {}
    counts = {"dev": 0, "test": 0}
    lines: list[str] = []
    for split, path in (("dev", dev_path), ("test", test_path)):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise ValueError(f"retrieval row must be an object: {path}:{line_number}")
                key = (str(row.get("language", "")), str(row.get("id", "")))
                if not all(key):
                    raise ValueError(f"retrieval row needs language and id: {path}:{line_number}")
                if key in source_map:
                    raise ValueError(f"duplicate retrieval (language, id) across dev/test: {key}")
                source_map[key] = split
                counts[split] += 1
                lines.append(raw_line.rstrip("\r\n") + "\n")
    if not counts["dev"] or not counts["test"]:
        raise ValueError(f"both retrieval splits must be nonempty: {counts}")
    content = "".join(lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"combined retrieval input already exists with different content: {destination}")
    else:
        with destination.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    return source_map, counts


def aggregate_retrieval_result(source_result: Mapping[str, Any],
                               source_splits: Mapping[tuple[str, str], str],
                               split_counts: Mapping[str, int], *, variant: str) -> dict[str, Any]:
    """Split combined retrieval query metrics without rebuilding the index."""
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    seen: set[tuple[str, str]] = set()
    language_meta: dict[str, dict[str, Any]] = {}
    for language_row in source_result.get("languages", []):
        language = str(language_row.get("language", "und"))
        language_meta[language] = {
            key: value for key, value in language_row.items() if key != "query_predictions"
        }
        for prediction in language_row.get("query_predictions", []):
            key = (str(prediction.get("language", language)), str(prediction.get("id", "")))
            if key not in source_splits:
                raise ValueError(f"retrieval evaluator returned an unknown query: {key}")
            if key in seen:
                raise ValueError(f"retrieval evaluator returned a duplicate query: {key}")
            seen.add(key)
            split = source_splits[key]
            target = grouped.setdefault(key[0], {}).setdefault(split, {metric: [] for metric in METRICS})
            for metric in METRICS:
                target[metric].append(float(prediction["metrics"][metric]))
    if seen != set(source_splits):
        missing = sorted(set(source_splits) - seen)[:5]
        raise ValueError(f"retrieval evaluator missed {len(set(source_splits) - seen)} queries: {missing}")

    by_language: dict[str, Any] = {}
    by_split: dict[str, Any] = {}
    all_languages = sorted(set(language_meta) | set(grouped))
    for language in all_languages:
        by_language[language] = {}
    for split in SPLITS:
        language_values: dict[str, Any] = {}
        pooled: dict[str, list[float]] = {metric: [] for metric in METRICS}
        total = 0
        for language in all_languages:
            rows = grouped.get(language, {}).get(split, {metric: [] for metric in METRICS})
            size = len(rows[METRICS[0]])
            if size:
                metrics = {metric: sum(rows[metric]) / size for metric in METRICS}
                for metric in METRICS:
                    pooled[metric].extend(rows[metric])
            else:
                metrics = {metric: None for metric in METRICS}
            by_language[language][split] = {"queries": size, **metrics}
            if size:
                language_values[language] = {"queries": size, **metrics}
            total += size
        by_split[split] = {
            "queries": total,
            "metrics": {metric: sum(values) / len(values) if values else None
                        for metric, values in pooled.items()},
            "by_language": language_values,
        }
        if total != int(split_counts[split]):
            raise ValueError(f"retrieval {split} query count mismatch: {total} != {split_counts[split]}")
    return {
        "schema_version": 1, "task": "retrieval", "variant": variant,
        "evaluation_scope": "prepared_data_scope", "queries_evaluated": len(seen),
        "queries_by_split": dict(split_counts), "by_split": by_split,
        "by_language": by_language, "languages": [language_meta[name] for name in sorted(language_meta)],
        "total_index_build_seconds": source_result.get("total_index_build_seconds"),
        "projected_5000_per_language_index_seconds": source_result.get(
            "projected_5000_per_language_index_seconds"),
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(job: Mapping[str, Any], model_dir: Path) -> dict[str, Any]:
    paths = list(job["data_paths"]) + list(job.get("source_paths", ()))
    if job.get("checkpoint") is not None:
        paths.append(Path(job["checkpoint"]))
    if not job.get("source_paths"):
        paths.append(EVALUATOR if job["task"] in {*TASK_ORDER, "performance"} else DIAGNOSTICS)
    config = model_dir / "config.json"
    if config.is_file():
        paths.append(config)
    files = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": str(path.resolve()), "sha256": _digest(path), "size_bytes": path.stat().st_size})
    return {"model_dir": str(model_dir.resolve()), "files": files}


def _offline_environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    return environment


def _wait_for_repair_status(path: Path) -> None:
    while not path.is_file():
        print(f"[waiting for completed repair status] {path}; checking again in 30s", flush=True)
        time.sleep(30)
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise RuntimeError(f"gate repair did not complete successfully: {path} status={status.get('status')!r}")


def _write_status(path: Path, state: Mapping[str, Any], lock: threading.Lock) -> None:
    with lock:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _validate_result(path: Path, job: Mapping[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = job.get("result_identity", {"task": job["task"], "variant": job["variant"]})
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"result identity mismatch in {path}")


def _validate_execute_inputs(jobs: Sequence[Mapping[str, Any]], model_dir: Path,
                             output_dir: Path, *, resume: bool) -> Path:
    state_path = output_dir / STATUS_NAME
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"--resume requires runner status file: {state_path}")
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {output_dir}; use --resume or choose a new directory")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    for job in jobs:
        for path in job["data_paths"]:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in job.get("source_paths", ()):
            if not path.is_file():
                raise FileNotFoundError(path)
        if job.get("checkpoint") is not None and not Path(job["checkpoint"]).is_file():
            raise FileNotFoundError(job["checkpoint"])
    return state_path


def _validate_resume(jobs: Sequence[dict[str, Any]], state: dict[str, Any],
                     provenance: Mapping[str, Mapping[str, Any]], output_dir: Path) -> set[str]:
    if state.get("runner") != "run_all_evaluation" or state.get("schema_version") != 1:
        raise ValueError("status file was not written by this evaluation runner")
    stored = state.setdefault("jobs", {})
    skipped: set[str] = set()
    for job in jobs:
        job_id = str(job["job_id"])
        output = Path(job["output"])
        previous = stored.get(job_id)
        index_pattern = f"{job['variant']}_retrieval_index_*.npy" if job["task"] == "retrieval" else None
        indices = list((output_dir / "indexes").glob(index_pattern)) if index_pattern else []
        if previous and previous.get("status") == "succeeded":
            if previous.get("provenance") != provenance[job_id]:
                raise ValueError(f"successful job provenance changed; refusing overwrite: {job_id}")
            if not output.is_file():
                raise FileNotFoundError(f"successful job result is missing: {output}")
            _validate_result(output, job)
            for artifact in previous.get("artifacts", []):
                if not Path(artifact).is_file():
                    raise FileNotFoundError(f"successful job artifact is missing: {artifact}")
            skipped.add(job_id)
        elif output.exists() or indices or any(Path(path).exists() for path in job.get("extra_artifacts", [])):
            raise FileExistsError(f"untracked or incomplete result exists for {job_id}; refusing overwrite")
    return skipped


def _retrieval_split_map(job: Mapping[str, Any], output_dir: Path
                         ) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    return combine_retrieval_queries(
        Path(job["data_paths"][0]), Path(job["data_paths"][1]),
        output_dir / "inputs" / "retrieval_queries_dev_test.jsonl",
    )


def _run_lanes(jobs: Sequence[dict[str, Any]], *, environment: Mapping[str, str],
               output_dir: Path, model_dir: Path, split_maps: Mapping[str, tuple[Mapping[tuple[str, str], str], Mapping[str, int]]],
               state: dict[str, Any], status_path: Path, provenance: Mapping[str, Mapping[str, Any]],
               skipped: set[str], max_workers: int) -> bool:
    cancel = threading.Event()
    active: dict[str, subprocess.Popen[bytes]] = {}
    active_lock = threading.Lock()
    status_lock = threading.RLock()
    failure: list[dict[str, Any]] = []

    def terminate_own_children(exclude_pid: int | None = None) -> None:
        with active_lock:
            children = [process for process in active.values() if process.pid != exclude_pid]
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def run_job(job: dict[str, Any], lane: str) -> None:
        job_id = str(job["job_id"])
        if job_id in skipped or cancel.is_set():
            return
        output = Path(job["output"])
        raw_output = Path(job["raw_output"]) if job.get("raw_output") else None
        output.parent.mkdir(parents=True, exist_ok=True)
        if raw_output:
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            if raw_output.exists():
                raise FileExistsError(f"temporary retrieval result already exists: {raw_output}")
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        base_log = log_dir / f"{job_id}.log"
        log_path = base_log
        attempt = 2
        while log_path.exists():
            log_path = log_dir / f"{job_id}.attempt-{attempt}.log"
            attempt += 1
        with status_lock:
            state["jobs"][job_id] = {
                "status": "running", "lane": lane, "task": job["task"],
                "variant": job["variant"], "output": str(output),
                "log": str(log_path), "provenance": provenance[job_id],
            }
            _write_status(status_path, state, status_lock)
        print(f"[{lane}] {job_id}: {shlex.join(job['command'])}", flush=True)
        return_code = -1
        process: subprocess.Popen[bytes] | None = None
        try:
            with log_path.open("wb") as log_stream:
                process = subprocess.Popen(
                    job["command"], cwd=ROOT, env=dict(environment),
                    stdout=log_stream, stderr=subprocess.STDOUT,
                )
                with active_lock:
                    active[job_id] = process
                return_code = process.wait()
        finally:
            if process is not None:
                with active_lock:
                    active.pop(job_id, None)
        if return_code != 0:
            failure.append({"job_id": job_id, "return_code": return_code, "log": str(log_path)})
            cancel.set()
            terminate_own_children()
            with status_lock:
                state["jobs"][job_id]["status"] = "failed"
                state["jobs"][job_id]["return_code"] = return_code
                state["status"] = "failed"
                _write_status(status_path, state, status_lock)
            return
        if job["task"] == "retrieval":
            assert raw_output is not None
            source = json.loads(raw_output.read_text(encoding="utf-8"))
            _validate_result(raw_output, job)
            source_map, counts = split_maps[job_id]
            result = aggregate_retrieval_result(source, source_map, counts, variant=job["variant"])
            result["data_files"] = [str(path.resolve()) for path in job["data_paths"]]
            result["checkpoint"] = str(Path(job["checkpoint"]).resolve())
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_output.unlink()
            artifacts = [str(path) for language in source.get("languages", [])
                         for key, path in (("stored_vector_file", language.get("stored_vector_file")),)
                         if path]
        else:
            _validate_result(output, job)
            artifacts = []
        for artifact_path in job.get("extra_artifacts", []):
            if not Path(artifact_path).is_file():
                raise FileNotFoundError(f"expected evaluation artifact is missing: {artifact_path}")
            artifacts.append(str(Path(artifact_path)))
        with status_lock:
            state["jobs"][job_id]["status"] = "succeeded"
            state["jobs"][job_id]["return_code"] = 0
            state["jobs"][job_id]["artifacts"] = artifacts
            state["status"] = "running"
            _write_status(status_path, state, status_lock)

    def run_lane(lane: str) -> None:
        for job in jobs:
            if job["lane"] == lane and job["task"] != "performance":
                run_job(job, lane)
                if cancel.is_set():
                    return

    lanes = ("A", "B")
    if max_workers == 1:
        for lane in lanes:
            if cancel.is_set():
                break
            run_lane(lane)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_lane, lane) for lane in lanes]
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    failure.append({"job_id": "runner", "error": repr(exc)})
                    cancel.set()
                    terminate_own_children()
                    with status_lock:
                        state["status"] = "failed"
                        state["failure"] = failure[-1]
                        _write_status(status_path, state, status_lock)
            # Surface worker errors through the machine status and exit code.
    if not cancel.is_set():
        # Throughput and latency measurements are sensitive to concurrent GPU work.
        for job in jobs:
            if job["task"] == "performance":
                run_job(job, "performance")
                if cancel.is_set():
                    break
    if failure:
        state["status"] = "failed"
        state["failure"] = failure[0]
    elif all(state.get("jobs", {}).get(str(job["job_id"]), {}).get("status") == "succeeded"
             for job in jobs):
        state["status"] = "completed"
    else:
        state["status"] = "incomplete"
        state["pending_job_ids"] = [str(job["job_id"]) for job in jobs
                                     if state.get("jobs", {}).get(str(job["job_id"]), {}).get("status") != "succeeded"]
    _write_status(status_path, state, status_lock)
    return state["status"] == "completed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "mmbert-base")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "prepared" / "full_67m")
    parser.add_argument("--training-dir", type=Path, default=ROOT / "results" / "formal_67m",
                        help="original MLM and V0/V1 downstream checkpoint root")
    parser.add_argument("--gated-training-dir", type=Path, default=ROOT / "results" / "formal_67m_gatefix",
                        help="V2/V3 downstream checkpoint root; final MLM checkpoints remain in --training-dir")
    parser.add_argument("--phase", choices=("tasks", "diagnostics", "all"), default="tasks")
    parser.add_argument("--output-dir", type=Path,
                        help="evaluation output directory; defaults to <training-dir>/evaluation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--variants", type=_csv_variants, help="comma-separated variant subset")
    group.add_argument("--exclude-variants", type=_csv_variants, help="comma-separated variants to omit")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true",
                        help="skip only successful results whose data/checkpoint provenance still matches")
    parser.add_argument("--after-repair-status", type=Path,
                        help="with --execute, wait in 30-second intervals for gate-repair status=completed")
    parser.add_argument("--execute", action="store_true", help="launch evaluation subprocesses; default prints a dry run")
    args = parser.parse_args(argv)
    variants = select_variants(args.variants, args.exclude_variants)
    output_dir = args.output_dir or (args.training_dir / "evaluation")
    jobs: list[dict[str, Any]] = []
    if args.phase in {"tasks", "all"}:
        jobs.extend(build_task_jobs(
            model_dir=args.model_dir, data_dir=args.data_dir, training_dir=args.training_dir,
            gated_training_dir=args.gated_training_dir, output_dir=output_dir, variants=variants,
        ))
    if args.phase in {"diagnostics", "all"}:
        jobs.extend(build_diagnostic_jobs(
            model_dir=args.model_dir, data_dir=args.data_dir, training_dir=args.training_dir,
            gated_training_dir=args.gated_training_dir, output_dir=output_dir, variants=variants,
        ))
    if not args.execute:
        print(json.dumps({
            "status": "dry_run", "execute_requires_flag": "--execute", "phase": args.phase,
            "gpu": args.gpu, "maximum_concurrent_tasks": args.max_workers,
            "variants": list(variants), "output_dir": str(output_dir), "job_count": len(jobs),
            "lane_schedule": {lane: [job["job_id"] for job in jobs
                                     if job["lane"] == lane and job["task"] != "performance"]
                              for lane in ("A", "B")},
            "serial_performance_jobs": [job["job_id"] for job in jobs if job["task"] == "performance"],
            "jobs": [{"job_id": job["job_id"], "lane": job["lane"],
                      "command": shlex.join(job["command"])} for job in jobs],
        }, ensure_ascii=False, indent=2))
        return 0

    if args.after_repair_status is not None:
        _wait_for_repair_status(args.after_repair_status)
    state_path = _validate_execute_inputs(jobs, args.model_dir, output_dir, resume=args.resume)
    retrieval_jobs = [job for job in jobs if job["task"] == "retrieval"]
    split_maps = {job["job_id"]: _retrieval_split_map(job, output_dir) for job in retrieval_jobs}
    for job in retrieval_jobs:
        combined = output_dir / "inputs" / "retrieval_queries_dev_test.jsonl"
        job["data_paths"] = tuple(job["data_paths"]) + (combined,)
    state: dict[str, Any]
    if args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {"schema_version": 1, "runner": "run_all_evaluation", "status": "running",
                 "created_utc": datetime.now(timezone.utc).isoformat(), "jobs": {}}
    state.setdefault("jobs", {})
    provenance = {job["job_id"]: _provenance(job, args.model_dir) for job in jobs}
    skipped = _validate_resume(jobs, state, provenance, output_dir) if args.resume else set()
    for job in jobs:
        if job["job_id"] not in skipped:
            state["jobs"].setdefault(job["job_id"], {"status": "pending"})
    state["phase"] = args.phase
    state["gpu"] = args.gpu
    state["maximum_concurrent_tasks"] = args.max_workers
    _write_status(state_path, state, threading.RLock())
    success = _run_lanes(
        jobs, environment=_offline_environment(args.gpu), output_dir=output_dir,
        model_dir=args.model_dir, split_maps=split_maps, state=state, status_path=state_path,
        provenance=provenance, skipped=skipped, max_workers=args.max_workers,
    )
    print(json.dumps({"status": state["status"], "completed_jobs": sum(
        entry.get("status") == "succeeded" for entry in state["jobs"].values()),
        "failed_jobs": [job_id for job_id, entry in state["jobs"].items()
                        if entry.get("status") == "failed"],
        "status_file": str(state_path)}, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
