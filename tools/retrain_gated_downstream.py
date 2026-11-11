"""Rerun only the six affected gated downstream branches on GPUs 5 and 7."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from run_training_matrix import _load_plan, build_commands

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "train_worker.py"
GPU_TASKS = {
    5: (("V2", "retrieval"), ("V3", "retrieval"), ("V3", "qa"), ("V3", "classification")),
    7: (("V2", "qa"), ("V3", "reranking")),
}
RESERVED_GIB = {"qa": 8.34, "retrieval": 10.54, "classification": 5.98, "reranking": 6.71}
MIN_FREE_GIB = {5: 39.4, 7: 17.1}
LOG_RUN = "gatefix_57"


def _flag(command: Sequence[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _utc(unix_time: float | None = None) -> str:
    value = datetime.fromtimestamp(unix_time, timezone.utc) if unix_time is not None else datetime.now(timezone.utc)
    return value.isoformat()


def _file_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_repair_schedule(model_dir: Path, data_dir: Path, original: Path,
                          output: Path, plan: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Copy only the fixed six command lines, changing only output directories."""
    original, output = original.resolve(), output.resolve()
    commands = build_commands(model_dir, data_dir, original, plan or _load_plan())
    by_key: dict[tuple[str, str], list[str]] = {}
    for command in commands:
        variant, task = _flag(command, "--variant"), _flag(command, "--task")
        if (variant, task) in {(variant, task) for pairs in GPU_TASKS.values() for variant, task in pairs}:
            if (variant, task) in by_key:
                raise ValueError(f"duplicate training command for {variant}/{task}")
            by_key[(str(variant), str(task))] = list(command)
    expected = {(variant, task) for pairs in GPU_TASKS.values() for variant, task in pairs}
    if set(by_key) != expected:
        raise ValueError(f"repair matrix mismatch: missing={sorted(expected - set(by_key))}, "
                         f"unexpected={sorted(set(by_key) - expected)}")

    jobs: list[dict[str, Any]] = []
    for gpu, pairs in GPU_TASKS.items():
        for variant, task in pairs:
            train_command = by_key[(variant, task)]
            output_index = train_command.index("--output-dir") + 1
            relative_output = Path(train_command[output_index]).resolve().relative_to(original)
            train_command[output_index] = str(output / relative_output)
            job_output = Path(train_command[output_index])
            job_id = f"{variant}_{task}_seed17"
            worker_command = [sys.executable, str(WORKER), "--gpu", str(gpu), "--", *train_command[2:]]
            jobs.append({
                "job_id": job_id, "gpu": gpu, "variant": variant, "task": task,
                "seed": 17, "steps": int(_flag(train_command, "--steps") or 0),
                "training_command": train_command, "command": worker_command,
                "data": str(Path(_flag(train_command, "--data") or "").resolve()),
                "init_checkpoint": str(Path(_flag(train_command, "--init-from") or "").resolve()),
                "output_dir": str(job_output.resolve()),
                "checkpoint": str((job_output / f"{variant}_{task}_checkpoint.pt").resolve()),
                "step_log": str((job_output / f"{variant}_{task}_training.jsonl").resolve()),
            })
    return jobs


def query_gpu(gpu: int) -> dict[str, float]:
    """Read free and total memory for the requested physical GPU only."""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    values = [int(part.strip()) for part in result.stdout.strip().split(",")]
    if len(values) != 2:
        raise RuntimeError(f"unexpected nvidia-smi response for GPU {gpu}: {result.stdout!r}")
    return {"free_gib": values[0] / 1024, "total_gib": values[1] / 1024}


def wait_for_gpus(*, poll_seconds: int = 30) -> dict[str, dict[str, float]]:
    """Wait for both assigned cards to meet only the planned free-memory floor."""
    while True:
        observed = {str(gpu): query_gpu(gpu) for gpu in GPU_TASKS}
        ready = all(observed[str(gpu)]["free_gib"] >= MIN_FREE_GIB[gpu] for gpu in GPU_TASKS)
        if ready:
            print(f"[GPU ready] {json.dumps(observed, sort_keys=True)}", flush=True)
            return observed
        print(f"[waiting for GPU 5/7 memory] free={observed}; required_gib={MIN_FREE_GIB}; "
              f"checking again in {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)


def _environment(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return env


def _write_status(path: Path, status: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _stop_owned(active: Mapping[str, subprocess.Popen[Any]]) -> None:
    processes = [process for process in active.values() if process.poll() is None]
    for process in processes:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _worker_metadata(log_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    start: dict[str, Any] = {}
    end: dict[str, Any] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("__WORKER_START__="):
            start = json.loads(line.split("=", 1)[1])
        elif line.startswith("__WORKER_END__="):
            end = json.loads(line.split("=", 1)[1])
    return start, end


def _step_metrics(step_log: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in step_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    memory = [row.get("memory", {}) for row in rows]
    return {
        "optimizer_updates": len(rows),
        "nonpad_tokens": sum(int(row.get("nonpad_tokens", 0)) for row in rows),
        "optimizer_step_seconds": sum(float(row.get("optimizer_step_seconds", 0.0)) for row in rows),
        "peak_allocated_bytes": max((int(row.get("peak_allocated_bytes") or 0) for row in memory), default=0),
        "peak_reserved_bytes": max((int(row.get("peak_reserved_bytes") or 0) for row in memory), default=0),
    }


def _gpu_contributions(jobs: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for gpu in GPU_TASKS:
        entries = [(job, state["jobs"][job["job_id"]]) for job in jobs if job["gpu"] == gpu]
        intervals = sorted((float(entry["started_unix"]), float(entry["ended_unix"]))
                           for _job, entry in entries if "started_unix" in entry and "ended_unix" in entry)
        union_seconds = 0.0
        union_start = union_end = None
        for start, end in intervals:
            if union_start is None:
                union_start, union_end = start, end
            elif start <= union_end:
                union_end = max(union_end, end)
            else:
                union_seconds += union_end - union_start
                union_start, union_end = start, end
        if union_start is not None:
            union_seconds += union_end - union_start
        metrics = []
        for job, entry in entries:
            values = entry.get("training_metrics", {})
            metrics.append((job, entry, values))
        result[str(gpu)] = {
            "physical_gpu": gpu,
            "job_count": len(entries),
            "successful_job_count": sum(entry.get("status") == "succeeded" for _job, entry in entries),
            "tasks": [job["job_id"] for job, _entry in entries],
            "optimizer_updates": sum(int(values.get("optimizer_updates", 0)) for _job, _entry, values in metrics),
            "nonpad_tokens": sum(int(values.get("nonpad_tokens", 0)) for _job, _entry, values in metrics),
            "worker_step_seconds_sum": sum(float(values.get("optimizer_step_seconds", 0.0))
                                            for _job, _entry, values in metrics),
            "worker_runtime_seconds_sum": sum(float(entry.get("runtime_seconds", 0.0)) for _job, entry, _values in metrics),
            "wall_clock_union_seconds": union_seconds,
            "interval_start_unix": min((start for start, _end in intervals), default=None),
            "interval_end_unix": max((end for _start, end in intervals), default=None),
            "peak_allocated_bytes": max((int(values.get("peak_allocated_bytes", 0)) for _job, _entry, values in metrics), default=0),
            "peak_reserved_bytes": max((int(values.get("peak_reserved_bytes", 0)) for _job, _entry, values in metrics), default=0),
            "device_name": next((entry.get("gpu_provenance", {}).get("device_name")
                                 for _job, entry in entries if entry.get("gpu_provenance", {}).get("device_name")), None),
        }
    return result


def _validate_execute(jobs: Sequence[Mapping[str, Any]], output: Path) -> Path:
    status_path = output / "repair_status.json"
    logs_path = output / "runs" / LOG_RUN
    if status_path.exists():
        raise FileExistsError(f"repair status already exists: {status_path}")
    if logs_path.exists():
        raise FileExistsError(f"repair log directory already exists: {logs_path}")
    for job in jobs:
        for input_path in (Path(job["data"]), Path(job["init_checkpoint"])):
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
        if Path(job["checkpoint"]).exists() or Path(job["step_log"]).exists():
            raise FileExistsError(f"repair output already exists for {job['job_id']}: {job['output_dir']}")
    return status_path


def _execute_jobs(jobs: Sequence[dict[str, Any]], *, output: Path,
                  initial_gpus: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    status_path = output / "repair_status.json"
    logs_dir = output / "runs" / LOG_RUN
    logs_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": 1, "status": "running", "run_id": LOG_RUN,
        "started_utc": _utc(), "started_unix": time.time(),
        "gpu_memory_at_start": dict(initial_gpus),
        "minimum_free_gib": {str(gpu): MIN_FREE_GIB[gpu] for gpu in GPU_TASKS},
        "reserved_task_gib": RESERVED_GIB,
        "resource_note": "startup free-memory floor includes about 1 GiB CUDA non-torch allowance per worker; no allocator cap is set",
        "jobs": {},
    }
    for job in jobs:
        state["jobs"][job["job_id"]] = {
            "status": "pending", "physical_gpu": job["gpu"], "logical_gpu": 0,
            "task": job["task"], "variant": job["variant"], "seed": job["seed"],
            "requested_optimizer_steps": job["steps"], "input_data": _file_provenance(Path(job["data"])),
            "input_checkpoint": _file_provenance(Path(job["init_checkpoint"])),
            "output_checkpoint": job["checkpoint"], "step_log": job["step_log"],
            "train_command": shlex.join(job["training_command"]),
            "worker_command": shlex.join(job["command"]),
            "log": str((logs_dir / f"{job['job_id']}.log").resolve()),
        }
    _write_status(status_path, state)

    active: dict[str, subprocess.Popen[Any]] = {}
    active_lock = threading.Lock()
    status_lock = threading.Lock()
    cancelled = threading.Event()
    first_failure: dict[str, Any] = {}

    def run(job: dict[str, Any]) -> dict[str, Any]:
        if cancelled.is_set():
            with status_lock:
                state["jobs"][job["job_id"]]["status"] = "cancelled_before_start"
                _write_status(status_path, state)
            return {"job_id": job["job_id"], "status": "cancelled_before_start"}
        entry = state["jobs"][job["job_id"]]
        log_path = Path(entry["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        monotonic_start = time.monotonic()
        process: subprocess.Popen[Any] | None = None
        error = None
        return_code = -1
        with status_lock:
            entry.update({"status": "running", "started_unix": started, "started_utc": _utc(started)})
            _write_status(status_path, state)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("__SCHEDULER_JOB__=" + json.dumps({
                    "job_id": job["job_id"], "physical_gpu": job["gpu"],
                    "train_command": job["training_command"], "worker_command": job["command"],
                }, ensure_ascii=False) + "\n")
                log.flush()
                process = subprocess.Popen(job["command"], cwd=ROOT, env=_environment(job["gpu"]),
                                           stdout=log, stderr=subprocess.STDOUT, text=True)
                with active_lock:
                    active[job["job_id"]] = process
                    stop_now = cancelled.is_set()
                with status_lock:
                    entry["pid"] = process.pid
                if stop_now:
                    process.terminate()
                return_code = process.wait()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                with active_lock:
                    active.pop(job["job_id"], None)
        ended = time.time()
        with status_lock:
            entry.update({"ended_unix": ended, "ended_utc": _utc(ended),
                          "runtime_seconds": time.monotonic() - monotonic_start,
                          "return_code": return_code, "error": error})
            if process is not None:
                entry["pid"] = process.pid
        if error is None and return_code == 0 and not Path(job["checkpoint"]).is_file():
            error = f"worker returned zero but checkpoint is missing: {job['checkpoint']}"
            return_code = 98
            with log_path.open("a", encoding="utf-8") as log:
                log.write("__SCHEDULER_ERROR__=" + error + "\n")
        if error is None and return_code == 0 and not Path(job["step_log"]).is_file():
            error = f"worker returned zero but optimizer-step log is missing: {job['step_log']}"
            return_code = 99
            with log_path.open("a", encoding="utf-8") as log:
                log.write("__SCHEDULER_ERROR__=" + error + "\n")
        metrics = _step_metrics(Path(job["step_log"])) if Path(job["step_log"]).is_file() else {}
        start_meta, end_meta = _worker_metadata(log_path) if log_path.is_file() else ({}, {})
        if error is None and return_code == 0:
            checkpoint_provenance = _file_provenance(Path(job["checkpoint"]))
        else:
            checkpoint_provenance = None
            error = error or f"worker exit status {return_code}"
            cancelled.set()
            with active_lock:
                owned = dict(active)
            _stop_owned(owned)
        with status_lock:
            entry["training_metrics"] = metrics
            entry["worker_start"] = start_meta
            entry["worker_end"] = end_meta
            entry["gpu_provenance"] = {
                "physical_gpu": job["gpu"], "logical_gpu": 0,
                "device_name": start_meta.get("device_name"),
            }
            if checkpoint_provenance is not None:
                entry["output_checkpoint_provenance"] = checkpoint_provenance
                entry["status"] = "succeeded"
            else:
                entry["status"] = "failed"
                entry["error"] = error
                if not first_failure:
                    first_failure.update({"job_id": job["job_id"], "return_code": return_code,
                                          "error": error, "log": str(log_path)})
            state["per_gpu"] = _gpu_contributions(jobs, state)
            _write_status(status_path, state)
        return {"job_id": job["job_id"], "status": entry["status"], "return_code": return_code}

    pool = ThreadPoolExecutor(max_workers=6)
    futures = [pool.submit(run, job) for job in jobs]
    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        cancelled.set()
        with active_lock:
            owned = dict(active)
        _stop_owned(owned)
        pool.shutdown(wait=True, cancel_futures=True)
        state["status"] = "interrupted"
        state["failure"] = {"error": "KeyboardInterrupt; owned child processes terminated"}
        state["ended_utc"] = _utc()
        state["ended_unix"] = time.time()
        state["per_gpu"] = _gpu_contributions(jobs, state)
        _write_status(status_path, state)
        raise
    except Exception as exc:
        cancelled.set()
        with active_lock:
            owned = dict(active)
        _stop_owned(owned)
        state["status"] = "failed"
        state["failure"] = {"error": f"{type(exc).__name__}: {exc}"}
        for entry in state["jobs"].values():
            if entry["status"] in {"pending", "running"}:
                entry["status"] = "cancelled"
        state["ended_unix"] = time.time()
        state["ended_utc"] = _utc(state["ended_unix"])
        try:
            state["per_gpu"] = _gpu_contributions(jobs, state)
            _write_status(status_path, state)
        except Exception:
            pass
        return state
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    all_succeeded = all(state["jobs"][job["job_id"]]["status"] == "succeeded"
                        and Path(job["checkpoint"]).is_file() for job in jobs)
    state["status"] = "completed" if all_succeeded else "failed"
    state["failure"] = first_failure or (None if all_succeeded else {"error": "not all six workers succeeded"})
    state["ended_unix"] = time.time()
    state["ended_utc"] = _utc(state["ended_unix"])
    state["per_gpu"] = _gpu_contributions(jobs, state)
    _write_status(status_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "mmbert-base")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "prepared" / "full_67m")
    parser.add_argument("--original-dir", type=Path, default=ROOT / "results" / "formal_67m")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "formal_67m_gatefix")
    parser.add_argument("--execute", action="store_true", help="run all six workers; default prints the fixed plan")
    args = parser.parse_args(argv)
    model_dir, data_dir = args.model_dir.resolve(), args.data_dir.resolve()
    original, output = args.original_dir.resolve(), args.output_dir.resolve()
    jobs = build_repair_schedule(model_dir, data_dir, original, output)
    required_free = {str(gpu): MIN_FREE_GIB[gpu] for gpu in GPU_TASKS}
    if not args.execute:
        print(json.dumps({"status": "dry_run", "execute_requires_flag": "--execute",
                          "jobs": jobs, "minimum_free_gib": required_free,
                          "log_directory": str(output / "runs" / LOG_RUN)}, ensure_ascii=False, indent=2))
        return 0

    status_path = _validate_execute(jobs, output)
    ready_memory = wait_for_gpus()
    result = _execute_jobs(jobs, output=output, initial_gpus=ready_memory)
    print(json.dumps({"status": result["status"], "status_file": str(status_path),
                      "per_gpu": result.get("per_gpu"), "failure": result.get("failure")},
                     ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
