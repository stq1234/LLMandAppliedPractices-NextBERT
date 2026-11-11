"""Schedule the fixed 12-task training matrix on two lanes of one GPU.

The default mode prints the complete plan. ``--execute`` is required to start
training. Five training waves run at most two independent worker processes at
once; each lane runs its own jobs in order.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Readiness checks only; these do not limit PyTorch's allocations.
STARTUP_MIN_FREE_GIB = {1: 18.0, 2: 30.0}
TRAINING_TASKS = {"mlm", "classification", "qa", "retrieval", "reranking"}


def _flag(command: Sequence[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _resolved(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _memory_mib_to_bytes(value: str) -> int:
    # ``--format=csv,noheader,nounits`` normally emits plain integers. Keep
    # the parser tolerant of unit suffixes for older nvidia-smi versions.
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        raise ValueError(f"invalid nvidia-smi memory value: {value!r}")
    return int(digits) * 1024**2


def query_gpu_memory(gpu: int) -> tuple[int, int]:
    """Return (free_bytes, total_bytes) from nvidia-smi without initializing CUDA."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is required for the GPU memory preflight") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi could not query GPU {gpu}: {details}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("nvidia-smi GPU memory query timed out") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU memory row for GPU {gpu}, got {len(rows)}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"unexpected nvidia-smi memory row: {rows[0]!r}")
    return _memory_mib_to_bytes(fields[0]), _memory_mib_to_bytes(fields[1])


def _gib(value: int | float) -> float:
    return float(value) / 1024**3


def _make_job(index: int, command: list[str]) -> dict[str, Any]:
    task = _flag(command, "--task")
    variant = _flag(command, "--variant")
    if task not in TRAINING_TASKS or variant is None:
        raise ValueError(f"unrecognized training command at index {index}: {command}")
    length_or_seed = _flag(command, "--length") if task == "mlm" else _flag(command, "--seed")
    if length_or_seed is None:
        raise ValueError(f"training command lacks length/seed: {command}")
    job_id = f"{variant}-{task}-{length_or_seed}"
    return {
        "index": index,
        "job_id": job_id,
        "task": task,
        "variant": variant,
        "steps": int(_flag(command, "--steps") or 0),
        "command": list(command),
    }


def _checkpoint_written_by(job: Mapping[str, Any]) -> Path:
    command = job["command"]
    output_dir = _resolved(_flag(command, "--output-dir") or "")
    return output_dir / f"{job['variant']}_{job['task']}_checkpoint.pt"


def _validate_dependencies(waves: Sequence[Mapping[str, Any]]) -> None:
    locations: dict[Path, tuple[int, str, int]] = {}
    ordered_jobs: list[tuple[int, str, int, Mapping[str, Any]]] = []
    for wave_number, wave in enumerate(waves, start=1):
        for lane_name, jobs in wave["lanes"].items():
            for lane_index, job in enumerate(jobs):
                checkpoint = _checkpoint_written_by(job)
                if checkpoint in locations:
                    raise ValueError(f"multiple jobs write the same checkpoint: {checkpoint}")
                locations[checkpoint] = (wave_number, lane_name, lane_index)
                ordered_jobs.append((wave_number, lane_name, lane_index, job))
    for wave_number, lane_name, lane_index, job in ordered_jobs:
        command = job["command"]
        for flag in ("--resume-from", "--init-from"):
            source = _flag(command, flag)
            if source is None:
                continue
            source_path = _resolved(source)
            producer = locations.get(source_path)
            if producer is None:
                raise ValueError(f"{job['job_id']} has an unscheduled {flag} dependency: {source_path}")
            producer_wave, producer_lane, producer_index = producer
            if producer_wave > wave_number or (
                producer_wave == wave_number and
                (producer_lane != lane_name or producer_index >= lane_index)
            ):
                raise ValueError(f"{job['job_id']} runs before its {flag} dependency: {source_path}")


def build_schedule(commands: Sequence[list[str]], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Arrange build_commands output into the fixed five-wave, two-lane plan."""
    jobs = [_make_job(index, list(command)) for index, command in enumerate(commands)]
    by_index = {job["index"]: job for job in jobs}
    mlm: dict[str, list[int]] = {variant: [] for variant in ("V0", "V1", "V2", "V3")}
    downstream: dict[tuple[str, str], list[int]] = {}
    for job in jobs:
        if job["task"] == "mlm":
            mlm[job["variant"]].append(job["index"])
        else:
            downstream.setdefault((job["variant"], job["task"]), []).append(job["index"])

    for variant, indices in mlm.items():
        indices.sort(key=lambda index: int(_flag(by_index[index]["command"], "--steps") or 0))
        if len(indices) != (2 if variant == "V3" else 4):
            raise ValueError(f"expected {2 if variant == 'V3' else 4} MLM stages for {variant}, got {len(indices)}")
    if sum(len(indices) for indices in mlm.values()) != 14:
        raise ValueError("the two-lane scheduler expects the frozen 14-stage MLM branch plan")
    if len(downstream) != 12 or any(len(indices) != 1 for indices in downstream.values()):
        raise ValueError("the two-lane scheduler expects 12 unique downstream variant/task phases")

    def one(variant: str, task: str) -> int:
        indices = downstream.get((variant, task), [])
        if len(indices) != 1:
            raise ValueError(f"expected one {variant}/{task} phase, got {len(indices)}")
        return indices[0]

    shared_steps = int(plan["adaptation"].get("shared_prefix", {}).get("steps", 0))
    v2_shared = [index for index in mlm["V2"]
                 if int(_flag(by_index[index]["command"], "--steps") or 0) <= shared_steps]
    v2_tail = [index for index in mlm["V2"] if index not in v2_shared]
    if not v2_shared or not v2_tail:
        raise ValueError("V2 must have both a shared prefix and a remaining branch")
    if len(v2_shared) + len(v2_tail) != len(mlm["V2"]):
        raise ValueError("V2 MLM stages do not partition at the shared-prefix boundary")

    wave_indices = [
        {"name": "wave1_mlm_v0_v1", "lanes": {"A": mlm["V0"], "B": mlm["V1"]}},
        {"name": "wave2_v2_shared_and_early_tasks", "lanes": {
            "A": v2_shared,
            "B": [one("V0", "classification"), one("V0", "reranking"), one("V1", "qa")],
        }},
        {"name": "wave3_v2_v3_mlm_tails", "lanes": {
            "A": v2_tail,
            "B": mlm["V3"],
        }},
        {"name": "wave4_v0_v3_v1_tasks", "lanes": {
            "A": [one("V0", "qa"), one("V0", "retrieval")],
            "B": [one("V3", "classification"), one("V3", "reranking"), one("V1", "retrieval")],
        }},
        {"name": "wave5_v2_v3_tasks", "lanes": {
            "A": [one("V2", "qa"), one("V2", "retrieval")],
            "B": [one("V3", "qa"), one("V3", "retrieval")],
        }},
    ]
    assigned = [index for wave in wave_indices for lane_jobs in wave["lanes"].values()
                for index in lane_jobs]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_index):
        raise ValueError("training plan contains a missing or duplicated scheduled command")
    waves: list[dict[str, Any]] = []
    for number, wave in enumerate(wave_indices, start=1):
        waves.append({
            "wave": number,
            "name": wave["name"],
            "lanes": {lane: [by_index[index] for index in indices]
                      for lane, indices in wave["lanes"].items()},
        })
    _validate_dependencies(waves)
    return waves


def _job_output_dir(job: Mapping[str, Any]) -> Path:
    return _resolved(_flag(job["command"], "--output-dir") or "")


def _validate_execute_inputs(waves: Sequence[Mapping[str, Any]], model_dir: Path,
                              output_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {model_dir}")
    data_paths = {
        Path(_flag(job["command"], "--data"))
        for wave in waves for lane_jobs in wave["lanes"].values() for job in lane_jobs
    }
    missing = [str(path) for path in sorted(data_paths) if not _resolved(path).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared training files missing: {missing}")
    if output_dir.exists() and any(output_dir.rglob("*_checkpoint.pt")):
        raise FileExistsError(
            f"output contains checkpoints; choose a fresh output directory: {output_dir}"
        )
    for wave in waves:
        for lane_jobs in wave["lanes"].values():
            for job in lane_jobs:
                job_dir = _job_output_dir(job)
                if job_dir.exists() and any(job_dir.glob("*_checkpoint.pt")):
                    raise FileExistsError(
                        f"job output already contains a checkpoint: {job_dir}"
                    )


def query_gpu_processes(gpu: int) -> list[dict[str, Any]]:
    """Read this GPU's compute PIDs and their Linux owners without CUDA."""
    device = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    gpu_uuid = device.stdout.strip()
    if not gpu_uuid or len(gpu_uuid.splitlines()) != 1:
        raise RuntimeError(f"could not identify physical GPU {gpu}")
    applications = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    pids: set[int] = set()
    for row in applications.stdout.splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) == 2 and fields[0] == gpu_uuid:
            pids.add(int(fields[1]))
    processes: list[dict[str, Any]] = []
    for pid in sorted(pids):
        owner = subprocess.run(
            ["ps", "-p", str(pid), "-o", "user:64="],
            capture_output=True, text=True, timeout=15,
        )
        if owner.returncode == 1 and not owner.stdout.strip():
            # The process can exit between nvidia-smi and ps.
            continue
        if owner.returncode != 0 or not owner.stdout.strip():
            raise RuntimeError(f"could not read owner of GPU process {pid}")
        processes.append({"pid": pid, "owner": owner.stdout.strip()})
    return processes


def wait_for_gpu(gpu: int, max_workers: int, poll_seconds: float = 30,
                 allowed_owner: str = "shaotianqi") -> tuple[int, int]:
    """Wait until idle or only one allowed user's process remains, then train."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    required_free_gib = STARTUP_MIN_FREE_GIB[max_workers]
    required = int(required_free_gib * 1024**3)
    while True:
        processes = query_gpu_processes(gpu)
        free, total = query_gpu_memory(gpu)
        if total < required:
            raise RuntimeError(f"GPU {gpu} is smaller than the required {required_free_gib:.0f} GiB")
        process_ready = not processes or (
            len(processes) == 1 and processes[0]["owner"] == allowed_owner
        )
        if process_ready and free >= required:
            print(f"[GPU ready] GPU {gpu}: processes={processes}, free={_gib(free):.2f} GiB; "
                  "starting the fixed training schedule", flush=True)
            return free, total
        print(f"[waiting for GPU {gpu}] processes={processes}, free={_gib(free):.2f} GiB; "
              f"need idle or one {allowed_owner} process and >= {required_free_gib:.0f} GiB free; "
              f"checking again in {poll_seconds:g}s", flush=True)
        time.sleep(poll_seconds)


def _offline_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    })
    return env


def _worker_command(job: Mapping[str, Any], gpu: int) -> list[str]:
    original = job["command"]
    worker = ROOT / "tools" / "train_worker.py"
    return [sys.executable, str(worker), "--gpu", str(gpu), "--",
            *original[2:]]


def _terminate_owned(active: Mapping[int, subprocess.Popen[Any]], *, exclude: int | None = None) -> None:
    owned = [process for pid, process in active.items()
             if pid != exclude and process.poll() is None]
    for process in owned:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    for process in owned:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # These are direct children started by this scheduler; no system
            # process or unrelated user process is ever selected here.
            process.kill()
            process.wait()


def _checkpoint_path(job: Mapping[str, Any]) -> Path:
    return _job_output_dir(job) / f"{job['variant']}_{job['task']}_checkpoint.pt"


def _remove_consumed_checkpoints(waves: Sequence[Mapping[str, Any]], wave_number: int,
                                 output_dir: Path) -> list[str]:
    """Remove only successful, intermediate MLM checkpoints at safe wave boundaries."""
    candidates: list[Mapping[str, Any]] = []
    if wave_number == 1:
        for lane in ("A", "B"):
            # V0 and V1 each retain their 8192-token final checkpoint.
            candidates.extend(waves[0]["lanes"][lane][:-1])
    elif wave_number == 2:
        # V2's 512 state has been consumed by the 1024 shared-prefix stage.
        candidates.extend(waves[1]["lanes"]["A"][:-1])
    elif wave_number == 3:
        # Both V2 and V3 have finished consuming V2's shared 384-step state.
        candidates.append(waves[1]["lanes"]["A"][-1])
        # Each 4096 state has been consumed by its own 8192 continuation.
        for lane in ("A", "B"):
            candidates.extend(waves[2]["lanes"][lane][:-1])

    root = output_dir.resolve()
    removed: list[str] = []
    for job in candidates:
        checkpoint = _checkpoint_path(job).resolve()
        try:
            checkpoint.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"refusing to remove checkpoint outside output directory: {checkpoint}") from exc
        if checkpoint.exists():
            checkpoint.unlink()
            removed.append(str(checkpoint))
    return removed


def _execute_waves(waves: Sequence[Mapping[str, Any]], *, gpu: int, output_dir: Path,
                   env: Mapping[str, str], max_workers: int) -> dict[str, Any]:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs_dir = output_dir / "logs" / run_stamp
    logs_dir.mkdir(parents=True, exist_ok=False)
    cancellation = threading.Event()
    active: dict[int, subprocess.Popen[Any]] = {}
    active_lock = threading.Lock()
    failure: dict[str, Any] | None = None
    failure_lock = threading.Lock()
    completed: list[dict[str, Any]] = []
    completed_lock = threading.Lock()
    removed_checkpoints: list[str] = []

    def run_lane(wave: Mapping[str, Any], lane_name: str) -> dict[str, Any]:
        nonlocal failure
        lane_result = {"wave": wave["wave"], "lane": lane_name, "status": "completed", "jobs": []}
        for sequence, job in enumerate(wave["lanes"][lane_name], start=1):
            if cancellation.is_set():
                lane_result["status"] = "cancelled"
                break
            log_path = logs_dir / f"wave{wave['wave']}_{lane_name}_{sequence}_{job['job_id']}.log"
            command = _worker_command(job, gpu)
            header = {
                "wave": wave["wave"], "lane": lane_name, "job_id": job["job_id"],
                "train_command": shlex.join(job["command"]),
                "worker_command": shlex.join(command),
            }
            print(f"[wave {wave['wave']} lane {lane_name}] starting {job['job_id']} log={log_path}",
                  flush=True)
            process: subprocess.Popen[Any] | None = None
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("__SCHEDULER_JOB__=" + json.dumps(header, ensure_ascii=False) + "\n")
                    log.flush()
                    process = subprocess.Popen(command, cwd=ROOT, env=dict(env), stdout=log,
                                               stderr=subprocess.STDOUT, text=True)
                    with active_lock:
                        active[process.pid] = process
                        should_stop = cancellation.is_set()
                    if should_stop:
                        process.terminate()
                    return_code = process.wait()
                    with active_lock:
                        active.pop(process.pid, None)
                if return_code == 0 and not _checkpoint_path(job).is_file():
                    return_code = 98
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write("__SCHEDULER_ERROR__=worker exited successfully without creating its checkpoint\n")
            except KeyboardInterrupt:
                cancellation.set()
                with active_lock:
                    owned_snapshot = dict(active)
                _terminate_owned(owned_snapshot)
                if process is not None:
                    with active_lock:
                        active.pop(process.pid, None)
                raise
            except Exception as exc:
                lane_result["status"] = "failed"
                with failure_lock:
                    if failure is None:
                        failure = {"wave": wave["wave"], "lane": lane_name,
                                   "job_id": job["job_id"], "error": f"{type(exc).__name__}: {exc}",
                                   "log": str(log_path)}
                cancellation.set()
                with active_lock:
                    owned_snapshot = dict(active)
                _terminate_owned(owned_snapshot)
                if process is not None:
                    with active_lock:
                        active.pop(process.pid, None)
                print(f"[wave {wave['wave']} lane {lane_name}] failed {job['job_id']}: {exc}",
                      flush=True)
                break
            job_result = {"job_id": job["job_id"], "return_code": return_code,
                          "log": str(log_path)}
            lane_result["jobs"].append(job_result)
            print(f"[wave {wave['wave']} lane {lane_name}] finished {job['job_id']} "
                  f"return_code={return_code} log={log_path}", flush=True)
            if return_code != 0:
                if cancellation.is_set():
                    lane_result["status"] = "cancelled"
                    job_result["cancelled_by_peer"] = True
                    break
                lane_result["status"] = "failed"
                with failure_lock:
                    if failure is None:
                        failure = {"wave": wave["wave"], "lane": lane_name,
                                   "job_id": job["job_id"], "return_code": return_code,
                                   "log": str(log_path)}
                cancellation.set()
                with active_lock:
                    owned_snapshot = dict(active)
                _terminate_owned(owned_snapshot, exclude=process.pid)
                break
        with completed_lock:
            completed.append(lane_result)
        return lane_result

    try:
        for wave in waves:
            if cancellation.is_set():
                break
            if max_workers == 1:
                for lane in ("A", "B"):
                    if cancellation.is_set():
                        break
                    run_lane(wave, lane)
            else:
                pool = ThreadPoolExecutor(max_workers=max_workers)
                futures = [pool.submit(run_lane, wave, lane) for lane in ("A", "B")]
                try:
                    for future in futures:
                        future.result()
                except KeyboardInterrupt:
                    cancellation.set()
                    with active_lock:
                        owned_snapshot = dict(active)
                    _terminate_owned(owned_snapshot)
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                else:
                    pool.shutdown(wait=True)
            if failure is not None:
                break
            removed_checkpoints.extend(
                _remove_consumed_checkpoints(waves, int(wave["wave"]), output_dir)
            )
    except KeyboardInterrupt:
        cancellation.set()
        with active_lock:
            _terminate_owned(active)
        raise
    return {"status": "failed" if failure else "completed",
            "failure": failure, "lanes": completed, "logs_dir": str(logs_dir),
            "removed_checkpoints": removed_checkpoints}


def _public_plan(waves: Sequence[Mapping[str, Any]], *, gpu: int, max_workers: int,
                 data_dir: Path, output_dir: Path, model_dir: Path) -> dict[str, Any]:
    public_waves: list[dict[str, Any]] = []
    total_jobs = 0
    for wave in waves:
        lanes: dict[str, list[dict[str, Any]]] = {}
        for lane_name, jobs in wave["lanes"].items():
            lanes[lane_name] = []
            for job in jobs:
                total_jobs += 1
                lanes[lane_name].append({
                    "job_id": job["job_id"], "task": job["task"],
                    "variant": job["variant"], "steps": job["steps"],
                    "command": shlex.join(job["command"]),
                })
        public_waves.append({"wave": wave["wave"], "name": wave["name"], "lanes": lanes})
    return {
        "status": "dry_run",
        "execute_requires_flag": "--execute",
        "startup_wait": "poll every 30s until GPU is idle or only one shaotianqi compute process remains, with sufficient free memory",
        "execution_mode": "serial" if max_workers == 1 else "two_lane_parallel",
        "gpu": gpu,
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "wave_count": len(waves),
        "job_count": total_jobs,
        "mlm_job_count": sum(1 for wave in waves for jobs in wave["lanes"].values()
                              for job in jobs if job["task"] == "mlm"),
        "downstream_job_count": sum(1 for wave in waves for jobs in wave["lanes"].values()
                                     for job in jobs if job["task"] != "mlm"),
        "maximum_concurrent_tasks": max_workers,
        "global_preflight_minimum_free_gib": STARTUP_MIN_FREE_GIB[max_workers],
        "checkpoint_cleanup": "consumed MLM intermediates are removed only after their dependent wave succeeds; final 8192 checkpoints are retained",
        "evaluation": "run separately with the existing evaluation CLI after training",
        "waves": public_waves,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/mmbert-base")
    parser.add_argument("--data-dir", type=Path,
                        help="prepared profile directory; defaults to experiment_plan.json data_profile")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/formal_67m")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2,
                        help="1 runs strictly serially; 2 runs the two planned lanes in parallel")
    parser.add_argument("--execute", action="store_true",
                        help="wait for the GPU, then train automatically; default is dry-run")
    args = parser.parse_args(argv)

    # Waiting uses nvidia-smi and ps only; it does not load torch or acquire CUDA.
    preflight: tuple[int, int] | None = None
    if args.execute:
        preflight = wait_for_gpu(args.gpu, args.max_workers)

    from run_training_matrix import _default_data_dir, _load_plan, build_commands

    plan = _load_plan()
    model_dir = _resolved(args.model_dir)
    data_dir = _resolved(args.data_dir or _default_data_dir(plan))
    output_dir = _resolved(args.output_dir)
    commands = build_commands(model_dir, data_dir, output_dir, plan)
    waves = build_schedule(commands, plan)
    public = _public_plan(waves, gpu=args.gpu, max_workers=args.max_workers, data_dir=data_dir,
                          output_dir=output_dir, model_dir=model_dir)
    if not args.execute:
        print(json.dumps(public, ensure_ascii=False, indent=2))
        return 0

    # This preflight uses only nvidia-smi. It must happen before importing
    # torch or launching a training process.
    assert preflight is not None
    free, total = preflight
    _validate_execute_inputs(waves, model_dir, output_dir)
    env = _offline_env(args.gpu)
    result = _execute_waves(waves, gpu=args.gpu, output_dir=output_dir, env=env,
                            max_workers=args.max_workers)
    report = {
        "status": result["status"], "gpu_preflight": {
            "free_gib": _gib(free), "total_gib": _gib(total),
            "minimum_free_gib": public["global_preflight_minimum_free_gib"],
        },
        "job_count": public["job_count"],
        "maximum_concurrent_tasks": public["maximum_concurrent_tasks"],
        "checkpoint_cleanup": "consumed MLM intermediates are removed only after their dependent wave succeeds; final 8192 checkpoints are retained",
        "failure": result["failure"], "lane_results": result["lanes"],
        "removed_intermediate_checkpoints": result["removed_checkpoints"],
        "logs_dir": result["logs_dir"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if result["status"] != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
