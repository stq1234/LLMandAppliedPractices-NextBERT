"""Apply a per-job PyTorch allocator cap, then run one offline training phase.

GPU free-memory checks use nvidia-smi before importing torch, so a rejected
worker does not initialize CUDA. The allocator cap is not a hard limit on
CUDA context, library, or non-PyTorch allocations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "src" / "train_nextbert.py"


def _query_memory_mib(gpu: int) -> tuple[int, int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is required for the worker memory check") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi could not query GPU {gpu}: {details}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("nvidia-smi worker memory query timed out") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU memory row for GPU {gpu}, got {len(rows)}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"unexpected nvidia-smi memory row: {rows[0]!r}")
    try:
        free_mib, total_mib = (int("".join(ch for ch in field if ch.isdigit())) for field in fields)
    except ValueError as exc:
        raise RuntimeError(f"invalid nvidia-smi memory row: {rows[0]!r}") from exc
    return free_mib, total_mib


def _stats(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi GPU index")
    parser.add_argument("--memory-budget-gib", type=float, required=True,
                        help="PyTorch caching-allocator cap; not a whole-process hard limit")
    parser.add_argument("--minimum-free-gib", type=float, required=True,
                        help="free memory required by nvidia-smi before importing torch")
    parser.add_argument("training_args", nargs=argparse.REMAINDER,
                        help="arguments passed to src/train_nextbert.py after --")
    args = parser.parse_args(argv)
    training_args = list(args.training_args)
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    return args, training_args


def main(argv: list[str] | None = None) -> int:
    args, training_args = _parse_args(argv)
    if not training_args:
        raise ValueError("training arguments are required after --")
    if args.memory_budget_gib <= 0 or args.minimum_free_gib < args.memory_budget_gib:
        raise ValueError("memory budget must be positive and no greater than minimum-free-gib")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    # Do not import torch or call CUDA APIs until this physical-device check
    # has passed. Another scheduler lane may already have consumed its cap.
    free_mib, total_mib = _query_memory_mib(args.gpu)
    free_bytes = free_mib * 1024**2
    total_bytes = total_mib * 1024**2
    required_bytes = int(args.minimum_free_gib * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"GPU {args.gpu} has {free_bytes / 1024**3:.2f} GiB free; "
            f"this job requires {args.minimum_free_gib:.2f} GiB before startup"
        )
    if total_bytes < int(args.memory_budget_gib * 1024**3):
        raise RuntimeError(
            f"GPU {args.gpu} has {total_bytes / 1024**3:.2f} GiB total; "
            f"allocator cap {args.memory_budget_gib:.2f} GiB cannot fit"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for physical GPU {args.gpu}")
    # CUDA_VISIBLE_DEVICES exposes the selected physical GPU as logical 0.
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.memory_budget_gib * 1024**3 / total_bytes), device=0
    )
    device = torch.cuda.get_device_name(0)
    torch.cuda.synchronize(0)
    before = _stats(torch)
    torch.cuda.reset_peak_memory_stats(0)
    meta: dict[str, Any] = {
        "physical_gpu": args.gpu,
        "logical_gpu": 0,
        "device_name": device,
        "pid": os.getpid(),
        "started_unix": time.time(),
        "nvidia_smi_free_before_torch_bytes": free_bytes,
        "nvidia_smi_total_bytes": total_bytes,
        "minimum_free_gib": args.minimum_free_gib,
        "pytorch_allocator_cap_gib": args.memory_budget_gib,
        "pytorch_allocator_cap_is_process_hard_limit": False,
        "before_train_nextbert": before,
        "training_args": training_args,
    }
    print("__WORKER_START__=" + json.dumps(meta, ensure_ascii=False), flush=True)

    success = False
    error: str | None = None
    try:
        sys.path.insert(0, str(ROOT / "src"))
        sys.argv = [str(TRAIN_SCRIPT), *training_args]
        runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
        success = True
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            torch.cuda.synchronize(0)
            after = _stats(torch)
        except Exception as exc:  # Preserve the training exception and log the unavailable stats.
            after = {"stats_error": f"{type(exc).__name__}: {exc}"}  # type: ignore[dict-item]
        print("__WORKER_END__=" + json.dumps({
            "pid": os.getpid(), "finished_unix": time.time(), "success": success,
            "error": error, "after_runpy": after,
            "peak_scope_note": (
                "train_nextbert resets CUDA peak counters during optimizer steps; "
                "after_runpy peak counters may cover only activity since the last reset, "
                "not the whole job"
            ),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
