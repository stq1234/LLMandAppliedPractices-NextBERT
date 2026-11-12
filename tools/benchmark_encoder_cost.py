"""Measure encoder-only NextBERT throughput on fixed prepared held-out rows.

The fixed protocol runs V0-V3 at lengths 512/4096/8192, with physical batches
16/4/2, three rounds, five warmups per cell, and a ten-second timed window.
Only one model is resident on the GPU at a time. No tokenizer or MLM head is
included in the timed forward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("V0", "V1", "V2", "V3")
LENGTHS = (512, 4096, 8192)
BATCH_SIZES = {512: 16, 4096: 4, 8192: 2}
ROUNDS = 3
WARMUP_STEPS = 5
TIMED_SECONDS = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checkpoint_path(checkpoint_root: Path, variant: str) -> Path:
    return checkpoint_root / "mlm" / variant / "8192" / f"{variant}_mlm_checkpoint.pt"


def build_plan() -> list[dict[str, Any]]:
    """Return 36 cells with a one-position model rotation each round."""
    plan: list[dict[str, Any]] = []
    for round_index in range(ROUNDS):
        order = VARIANTS[round_index:] + VARIANTS[:round_index]
        for variant in order:
            for length in LENGTHS:
                plan.append({
                    "round": round_index + 1,
                    "variant": variant,
                    "length": length,
                    "physical_batch": BATCH_SIZES[length],
                })
    return plan


def load_prepared_batch(path: Path, length: int, physical_batch: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate one fixed held-out array and return its first physical batch."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != length:
        raise ValueError(f"prepared input must be int32 [N,{length}]: {path}")
    if array.shape[0] < physical_batch:
        raise ValueError(f"prepared input has {array.shape[0]} rows; needs {physical_batch}: {path}")
    batch = np.array(array[:physical_batch], dtype=np.int64, order="C", copy=True)
    metadata = {
        "path": str(path.resolve()),
        "rows": int(array.shape[0]),
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "sha256": sha256_file(path),
        "selected_row_indices": list(range(physical_batch)),
        "selected_shape": [int(dim) for dim in batch.shape],
        "selected_input_sha256": sha256_bytes(batch.tobytes(order="C")),
        "nonpad_tokens_per_batch": int(np.count_nonzero(batch)),
    }
    return batch, metadata


def rotated_orders() -> list[list[str]]:
    return [list(VARIANTS[index:] + VARIANTS[:index]) for index in range(ROUNDS)]


def validate_gate_state(variant: str, gate_state: Mapping[str, Any]) -> None:
    """Refuse to benchmark gated variants unless their restored ramp is active."""
    if variant not in {"V2", "V3"}:
        return
    ramp = float(gate_state.get("gate_ramp", 0.0))
    if not math.isclose(ramp, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{variant} MLM checkpoint did not restore a fully active gate ramp: {ramp}")


def summarize_samples(samples: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("cannot summarize an empty benchmark sample")
    cuda_ms = [float(row["cuda_event_encode_ms"]) for row in samples]
    wall_ms = [float(row["wall_prepared_cpu_gpu_mask_encode_ms"]) for row in samples]
    tok_s = [float(row["nonpad_tokens_per_second"]) for row in samples]
    total_tokens = sum(int(row["nonpad_tokens"]) for row in samples)
    aggregate_tok_s = total_tokens / max(sum(wall_ms) / 1000.0, 1e-12)
    return {
        "forward_count": len(samples),
        "cuda_event_encode_ms_mean": statistics.mean(cuda_ms),
        "cuda_event_encode_ms_p50": float(np.percentile(cuda_ms, 50)),
        "cuda_event_encode_ms_p95": float(np.percentile(cuda_ms, 95)),
        "wall_prepared_cpu_gpu_mask_encode_ms_mean": statistics.mean(wall_ms),
        "wall_prepared_cpu_gpu_mask_encode_ms_p50": float(np.percentile(wall_ms, 50)),
        "wall_prepared_cpu_gpu_mask_encode_ms_p95": float(np.percentile(wall_ms, 95)),
        "nonpad_tokens_per_second": aggregate_tok_s,
        "nonpad_tokens_per_second_sample_mean": statistics.mean(tok_s),
        "nonpad_tokens_per_second_p50": float(np.percentile(tok_s, 50)),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _memory_peak(torch: Any, device: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _hardware_info(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_name": properties.name,
        "logical_device": str(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_processor": platform.processor(),
        "host_machine": platform.machine(),
        "host_logical_cpu_count": os.cpu_count(),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
    }


def _software_info(torch: Any, transformers: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def _configure_cuda(gpu: str) -> tuple[str, str]:
    """Set visibility before importing torch; an external mask takes precedence."""
    external = os.environ.get("CUDA_VISIBLE_DEVICES")
    if external is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return str(gpu), "--gpu"
    if not external.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES is set but empty; expose the benchmark GPU")
    return external, "CUDA_VISIBLE_DEVICES"


def _load_runtime() -> tuple[Any, Any, Any]:
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import torch
    import transformers
    from train_nextbert import load_nextbert, restore_gate_state
    return torch, transformers, (load_nextbert, restore_gate_state)


def _load_model(*, torch: Any, load_nextbert: Any, restore_gate_state: Any,
                model_dir: Path, checkpoint: Path, variant: str, device: Any) -> tuple[Any, dict[str, Any]]:
    """Load model and checkpoint on CPU, then move the single model to CUDA."""
    cpu = torch.device("cpu")
    model_started = time.perf_counter()
    model = load_nextbert(model_dir, variant, cpu, attention_backend="auto")
    model.gradient_checkpointing_disable()
    model.eval()
    model_initialized = time.perf_counter()

    checkpoint_started = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_loaded = time.perf_counter()
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping: {checkpoint}")
    state = payload.get("model", payload)
    model.load_state_dict(state)
    gate_state = restore_gate_state(model, payload, fallback_task="mlm")
    validate_gate_state(variant, gate_state)
    state_loaded = time.perf_counter()
    checkpoint_step = int(payload.get("step", 0))
    checkpoint_task = str(payload.get("task", "unknown"))
    del state, payload
    gc.collect()

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gpu_started = time.perf_counter()
    model.to(device)
    torch.cuda.synchronize(device)
    gpu_loaded = time.perf_counter()
    model.eval()
    config_payload = json.loads(json.dumps(model.config.to_dict(), default=str))
    config_fingerprint = sha256_bytes(json.dumps(
        config_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8"))
    model_metadata = {
        "variant": variant,
        "checkpoint_task": checkpoint_task,
        "checkpoint_step": checkpoint_step,
        "gate_state": dict(gate_state),
        "attention_backend": str(model.attention_backend),
        "local_radius": int(model.local_radius),
        "query_chunk_size": int(model.query_chunk_size),
        "config": config_payload,
        "config_sha256": config_fingerprint,
        "model_initialization_seconds": model_initialized - model_started,
        "checkpoint_cpu_load_seconds": checkpoint_loaded - checkpoint_started,
        "checkpoint_cpu_state_apply_seconds": state_loaded - checkpoint_loaded,
        "model_gpu_transfer_seconds": gpu_loaded - gpu_started,
        "model_gpu_load_peak": _memory_peak(torch, device),
    }
    return model, model_metadata


def _forward(torch: Any, model: Any, cpu_ids: Any, device: Any,
             nonpad_tokens: int) -> dict[str, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    ids = cpu_ids.to(device=device, dtype=torch.long, non_blocking=False)
    attention_mask = ids.ne(0).long()
    start_event.record()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.encode(input_ids=ids, attention_mask=attention_mask)
    end_event.record()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started
    result = {
        "cuda_event_encode_ms": float(start_event.elapsed_time(end_event)),
        "wall_prepared_cpu_gpu_mask_encode_ms": wall_seconds * 1000.0,
        "nonpad_tokens": int(nonpad_tokens),
        "nonpad_tokens_per_second": nonpad_tokens / max(wall_seconds, 1e-12),
    }
    del hidden, attention_mask, ids
    return result


def _run_cell(torch: Any, model: Any, cpu_ids: Any, device: Any,
              *, nonpad_tokens: int, round_index: int) -> dict[str, Any]:
    model.eval()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(WARMUP_STEPS):
        _forward(torch, model, cpu_ids, device, nonpad_tokens)

    samples: list[dict[str, float]] = []
    measured_started = time.perf_counter()
    deadline = measured_started + TIMED_SECONDS
    while time.perf_counter() < deadline:
        sample = _forward(torch, model, cpu_ids, device, nonpad_tokens)
        sample["sample_index"] = len(samples)
        samples.append(sample)
    measured_wall_seconds = time.perf_counter() - measured_started
    peak = _memory_peak(torch, device)
    return {
        "round": round_index,
        "warmup_steps": WARMUP_STEPS,
        "target_timed_seconds": TIMED_SECONDS,
        "actual_timed_seconds": measured_wall_seconds,
        "nonpad_tokens_per_batch": nonpad_tokens,
        "summary": summarize_samples(samples),
        "peak_memory": peak,
        "samples": samples,
    }


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    model_dir = args.model_dir.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"local model artifact/config missing: {model_dir}")
    if not (model_dir / "pytorch_model.bin").is_file() and not (model_dir / "nextbert_model.bin").is_file():
        raise FileNotFoundError(f"local model weights missing: {model_dir}")
    checkpoints = {variant: checkpoint_path(checkpoint_root, variant) for variant in VARIANTS}
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"formal MLM checkpoint missing: {checkpoint}")
    inputs = {str(length): prepared_dir / f"wiki_{length}_test.npy" for length in LENGTHS}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"prepared held-out input missing: {path}")
    hashes = {"model_config_sha256": sha256_file(model_dir / "config.json")}
    hashes.update({f"checkpoint_{variant}_sha256": sha256_file(path)
                   for variant, path in checkpoints.items()})
    source_files = {
        "tools/benchmark_encoder_cost.py": Path(__file__).resolve(),
        "src/modeling_nextbert.py": ROOT / "src" / "modeling_nextbert.py",
        "src/train_nextbert.py": ROOT / "src" / "train_nextbert.py",
    }
    hashes["source_sha256"] = {name: sha256_file(path) for name, path in source_files.items()}
    return inputs, checkpoints, {**hashes, "model_dir": str(model_dir)}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    status_path = output.with_name(output.stem + "_status.json")
    if output.exists() or status_path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output/status: {output}, {status_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    status: dict[str, Any] = {
        "status": "running", "started_utc": started_utc, "updated_utc": started_utc,
        "output": str(output), "completed_cells": 0, "total_cells": ROUNDS * len(VARIANTS) * len(LENGTHS),
        "protocol": {"rounds": ROUNDS, "warmup_steps_per_cell": WARMUP_STEPS,
                     "timed_seconds_per_cell": TIMED_SECONDS, "order_by_round": rotated_orders()},
    }
    atomic_write_json(status_path, status)
    try:
        inputs, checkpoints, provenance = _preflight(args)
        batches: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for length in LENGTHS:
            batches[str(length)] = load_prepared_batch(
                inputs[str(length)], length, BATCH_SIZES[length],
            )

        visible_devices, visibility_source = _configure_cuda(args.gpu)
        status.update({"visible_devices": visible_devices, "visibility_source": visibility_source,
                       "updated_utc": utc_now()})
        atomic_write_json(status_path, status)
        torch, transformers, runtime = _load_runtime()
        load_nextbert, restore_gate_state = runtime
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this benchmark requires the specified GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support BF16 autocast")
        device = torch.device("cuda:0")
        hardware = _hardware_info(torch, device)
        software = _software_info(torch, transformers)
        status.update({"hardware": hardware, "software": software, "updated_utc": utc_now()})
        atomic_write_json(status_path, status)

        result: dict[str, Any] = {
            "status": "completed", "started_utc": started_utc, "finished_utc": None,
            "protocol": {
                "variants": list(VARIANTS), "sequence_lengths": list(LENGTHS),
                "physical_batch_sizes": {str(key): value for key, value in BATCH_SIZES.items()},
                "rounds": ROUNDS, "warmup_steps_per_cell": WARMUP_STEPS,
                "timed_seconds_per_cell": TIMED_SECONDS,
                "model_order_by_round": rotated_orders(), "weights_dtype": "float32",
                "compute_dtype": "bfloat16 autocast", "inference_mode": True,
                "timing_scope_cuda_event": "NextBert.encode only; excludes MLM head and tokenization",
                "timing_scope_wall": "prepared CPU token tensor to GPU copy, attention-mask construction, and NextBert.encode; excludes tokenization and MLM head",
                "throughput_tokens": "nonpadding token IDs (ID != 0) per wall second",
            },
            "provenance": {
                **provenance, "prepared_inputs": {length: metadata for length, (_, metadata) in batches.items()},
                "checkpoint_paths": {variant: str(path) for variant, path in checkpoints.items()},
                "checkpoint_sha256": {variant: provenance[f"checkpoint_{variant}_sha256"]
                                      for variant in VARIANTS},
            },
            "hardware": hardware, "software": software, "model_loads": [], "cells": [],
            "whole_run_gpu_memory_peak": {"max_allocated_bytes": 0, "max_reserved_bytes": 0},
        }
        cell_lookup = {(variant, length): {"variant": variant, "sequence_length": length,
                                          "physical_batch": BATCH_SIZES[length], "rounds": []}
                       for variant in VARIANTS for length in LENGTHS}

        for round_index, order in enumerate(rotated_orders(), start=1):
            for variant in order:
                model, load_metadata = _load_model(
                    torch=torch, load_nextbert=load_nextbert, restore_gate_state=restore_gate_state,
                    model_dir=args.model_dir.expanduser().resolve(), checkpoint=checkpoints[variant],
                    variant=variant, device=device,
                )
                load_metadata.update({
                    "round": round_index, "checkpoint_path": str(checkpoints[variant]),
                    "checkpoint_sha256": provenance[f"checkpoint_{variant}_sha256"],
                })
                result["model_loads"].append(load_metadata)
                for metric in ("max_allocated_bytes", "max_reserved_bytes"):
                    result["whole_run_gpu_memory_peak"][metric] = max(
                        result["whole_run_gpu_memory_peak"][metric],
                        load_metadata["model_gpu_load_peak"][metric],
                    )
                for length in LENGTHS:
                    batch, input_info = batches[str(length)]
                    cpu_ids = torch.from_numpy(batch)
                    round_data = _run_cell(
                        torch, model, cpu_ids, device,
                        nonpad_tokens=int(input_info["nonpad_tokens_per_batch"]), round_index=round_index,
                    )
                    for metric in ("max_allocated_bytes", "max_reserved_bytes"):
                        result["whole_run_gpu_memory_peak"][metric] = max(
                            result["whole_run_gpu_memory_peak"][metric], round_data["peak_memory"][metric],
                        )
                    cell_lookup[(variant, length)]["rounds"].append(round_data)
                    status.update({
                        "current_round": round_index, "current_variant": variant,
                        "current_length": length, "completed_cells": status["completed_cells"] + 1,
                        "last_cell_forward_count": round_data["summary"]["forward_count"],
                        "updated_utc": utc_now(),
                    })
                    atomic_write_json(status_path, status)
                del model
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

        result["cells"] = [cell_lookup[(variant, length)] for variant in VARIANTS for length in LENGTHS]
        result["finished_utc"] = utc_now()
        atomic_write_json(output, result)
        status.update({"status": "completed", "finished_utc": result["finished_utc"],
                       "result_sha256": sha256_file(output), "updated_utc": utc_now()})
        atomic_write_json(status_path, status)
        return result
    except BaseException as exc:
        status.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "finished_utc": utc_now(), "updated_utc": utc_now()})
        atomic_write_json(status_path, status)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "mmbert-base")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "data" / "prepared" / "full_67m")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "results" / "formal_67m")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "formal_67m" / "encoder_cost_benchmark.json")
    parser.add_argument("--gpu", default="7", help="physical GPU exposed if CUDA_VISIBLE_DEVICES is unset")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_benchmark(args)
    except BaseException as exc:
        print(f"encoder benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
