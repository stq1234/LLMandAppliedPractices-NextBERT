"""Run one normal offline training phase on the selected GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "src" / "train_nextbert.py"


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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for physical GPU {args.gpu}")
    # CUDA_VISIBLE_DEVICES exposes the selected physical GPU as logical 0.
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
