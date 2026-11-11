"""Offline training entry points for the frozen NextBERT experiment plan.

This module intentionally trains only the explicitly requested task and step
count.  It never downloads model or data assets and never expands a command to
the full experiment matrix.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PACKAGE_ROOT / "configs" / "experiment_plan.json"
ADAPTATION_PATH = PACKAGE_ROOT / "configs" / "mmbert_adaptation.json"
TASKS = ("mlm", "classification", "qa", "retrieval", "reranking")
TASK_HEAD_KEY_PARTS = (
    "classification_head", "classifier", "classification", "qa_head", "span_head",
    "qa_outputs", "start_head", "end_head", "retrieval_projection", "retrieval_head",
    "reranking_head", "reranker", "task_head",
)


def read_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adaptation_stage_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return plan stages with cumulative absolute ``start_step``/``end_step``."""
    adaptation = plan.get("adaptation", plan)
    stages = adaptation.get("stages", [])
    if not stages:
        raise ValueError("experiment plan has no adaptation stages")
    schedule: list[dict[str, Any]] = []
    offset = 0
    seen_lengths: set[int] = set()
    for raw_stage in stages:
        stage = dict(raw_stage)
        length = int(stage["max_length"])
        steps = int(stage["optimizer_steps"])
        if length in seen_lengths or steps < 1:
            raise ValueError("adaptation stages must have unique lengths and positive optimizer_steps")
        seen_lengths.add(length)
        stage["start_step"] = offset
        stage["end_step"] = offset + steps
        schedule.append(stage)
        offset += steps
    return schedule


def qa_phase_steps(plan: Mapping[str, Any]) -> dict[str, int]:
    """Compute regular and long QA optimizer targets from plan counts and batches."""
    downstream = plan["downstream"]
    qa = downstream["qa"]
    long = qa.get("long_augmentation", {})
    epochs = int(downstream.get("shared", {}).get("epochs", 2))
    language_count = int(qa.get("language_count", len(qa.get("sources", ("zh", "en")))))
    regular_questions_per_language = int(qa.get("train_questions_per_language", 0))
    regular_target = int(qa.get("regular_target_questions",
                                regular_questions_per_language * language_count))
    regular_effective = int(qa.get("effective_batch", int(qa.get("physical_batch", 16))
                                * int(qa.get("accumulation", 2))))
    regular_steps = int(qa.get("regular_steps", math.ceil(regular_target * epochs / regular_effective)))

    long_target = long.get("train_questions_total")
    if long_target is None:
        fraction = float(long.get("fraction_of_regular_training_windows", 0.0))
        long_target = math.ceil(regular_target * fraction)
    long_effective = int(long.get("effective_batch", int(long.get("physical_batch", 1))
                              * int(long.get("accumulation", 1))))
    long_steps = int(long.get("steps", math.ceil(int(long_target) * epochs / max(1, long_effective))))
    return {"regular": regular_steps, "long": long_steps}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(device_name: str | None = None) -> torch.device:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_nextbert(model_dir: Path, variant: str, device: torch.device,
                  attention_backend: str = "auto") -> nn.Module:
    """Construct a model only from a local directory; set HF offline guards first."""
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {model_dir}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from modeling_nextbert import NextBert
    except ImportError as exc:
        raise RuntimeError("NextBert model module is not available in src/") from exc
    model = NextBert(str(model_dir), variant=variant, attention_backend=attention_backend)
    model.gradient_checkpointing_enable()
    return model.to(device)


def _method(model: nn.Module, name: str):
    method = getattr(model, name, None)
    if method is None:
        raise TypeError(f"model does not implement required method {name}()")
    return method


def _field(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _model_loss(output: Any) -> Tensor:
    loss = _field(output, "loss")
    if loss is None and isinstance(output, (tuple, list)) and output:
        loss = output[0]
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise TypeError("forward_mlm must return a scalar .loss (or scalar first tuple item)")
    return loss


def _logits(output: Any, names: Sequence[str] = ("logits",)) -> Tensor:
    for name in names:
        value = _field(output, name)
        if isinstance(value, Tensor):
            return value
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        values = [value for value in output if isinstance(value, Tensor)]
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return torch.stack(values, dim=-1)
    raise TypeError(f"model output does not contain logits: {type(output).__name__}")


def _embeddings(output: Any) -> Tensor:
    for name in ("embeddings", "pooled", "pooler_output", "last_hidden_state"):
        value = _field(output, name)
        if isinstance(value, Tensor):
            return value
    if isinstance(output, Tensor):
        return output
    raise TypeError(f"model output does not contain embeddings: {type(output).__name__}")


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _memory_stats(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None,
                "peak_reserved_bytes": None, "device_free_bytes": None,
                "device_total_bytes": None}
    torch.cuda.synchronize(device)
    device_free, device_total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_free_bytes": int(device_free),
        "device_total_bytes": int(device_total),
    }


def optimizer_for(model: nn.Module, *, backbone_lr: float, gate_lr: float,
                  weight_decay: float, head_lr: float | None = None) -> AdamW:
    gate_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    ordinary_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        lowered = name.lower()
        if "gate" in lowered:
            gate_params.append(parameter)
        elif head_lr is not None and any(key in lowered for key in (
            "classifier", "classification", "qa_head", "qa_outputs", "span_head", "retrieval_projection",
            "rerank", "task_head", "projection",
        )):
            head_params.append(parameter)
        else:
            ordinary_params.append(parameter)
    if not ordinary_params and not gate_params and not head_params:
        raise ValueError("model has no trainable parameters")
    groups: list[dict[str, Any]] = []
    if ordinary_params:
        groups.append({"params": ordinary_params, "lr": backbone_lr,
                       "weight_decay": weight_decay, "group_name": "backbone_and_heads"})
    if gate_params:
        groups.append({"params": gate_params, "lr": gate_lr,
                       "weight_decay": weight_decay, "group_name": "gates"})
    if head_params:
        groups.append({"params": head_params, "lr": float(head_lr),
                       "weight_decay": weight_decay, "group_name": "task_heads"})
    return AdamW(groups, lr=backbone_lr, weight_decay=weight_decay)


def _scheduler(optimizer: AdamW, total_steps: int, warmup_fraction: float):
    warmup_steps = max(1, math.ceil(total_steps * warmup_fraction)) if total_steps else 0

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


_GATE_RAMP_WARMUP_UPDATES = 64


def gate_state_metadata(model: nn.Module) -> dict[str, int | float]:
    """Return the effective structural-gate ramp state for a checkpoint."""
    return {
        "gate_ramp": float(getattr(model, "gate_ramp", 0.0)),
        "adaptation_updates": int(getattr(model, "adaptation_updates", 0)),
    }


def restore_gate_state(model: nn.Module, payload: Mapping[str, Any], *,
                       fallback_task: str | None = None) -> dict[str, int | float]:
    """Restore effective gate state, inferring only legacy MLM checkpoints.

    Older MLM checkpoints did not persist the non-parameter warmup counter, so
    their saved optimizer step determines the ramp. Legacy downstream states
    did not train with an enabled ramp and therefore restore with gates off.
    """
    raw_metadata = payload.get("metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    task = str(payload.get("task", fallback_task or ""))
    updates_value = metadata.get("adaptation_updates")
    if updates_value is None:
        updates = int(payload.get("step", 0)) if task == "mlm" else 0
    else:
        updates = max(0, int(updates_value))

    set_update = getattr(model, "set_adaptation_step", None)
    if callable(set_update):
        set_update(updates)

    saved_ramp = metadata.get("gate_ramp")
    if saved_ramp is not None:
        ramp = min(1.0, max(0.0, float(saved_ramp)))
    elif "adaptation_updates" in metadata or task == "mlm":
        if callable(set_update):
            ramp = float(getattr(model, "gate_ramp", 0.0))
        else:
            ramp = min(1.0, updates / _GATE_RAMP_WARMUP_UPDATES)
    else:
        ramp = 0.0
    _apply_gate_ramp(model, ramp)
    return {
        "gate_ramp": float(getattr(model, "gate_ramp", ramp)),
        "adaptation_updates": int(getattr(model, "adaptation_updates", updates)),
    }


def _metadata_with_gate_state(model: nn.Module,
                              metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(metadata or {})
    result.update(gate_state_metadata(model))
    return result


def _save_checkpoint(path: Path, *, model: nn.Module, optimizer: AdamW,
                     scheduler: Any, step: int, task: str, variant: str,
                     seed: int, metadata: Mapping[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "checkpoint_kind": "full_state",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "task": task,
        "variant": variant,
        "seed": seed,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metadata": _metadata_with_gate_state(model, metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_weights_only(path: Path, *, model: nn.Module, step: int, task: str,
                       variant: str, seed: int,
                       metadata: Mapping[str, Any] | None = None) -> None:
    """Save inference/initialization weights without optimizer or RNG state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "checkpoint_kind": "weights_only",
        "model": model.state_dict(),
        "step": step,
        "task": task,
        "variant": variant,
        "seed": seed,
        "metadata": _metadata_with_gate_state(model, metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, *, model: nn.Module, optimizer: AdamW,
                     scheduler: Any, task: str, variant: str,
                     map_location: torch.device,
                     allow_shared_variant_alias: bool = False) -> int:
    # Stage the serialized model and Adam states in host RAM. Loading the whole
    # payload on CUDA temporarily duplicates large weights beside the live model.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_kind") == "weights_only":
        raise ValueError("weights-only checkpoint cannot resume optimizer/RNG state; use --init-from")
    saved_variant = payload.get("variant")
    same_variant = saved_variant == variant
    shared_pair = {saved_variant, variant} == {"V2", "V3"}
    if payload.get("task") != task or not (same_variant or (allow_shared_variant_alias and shared_pair)):
        raise ValueError("resume checkpoint task/variant does not match this run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    restore_gate_state(model, payload, fallback_task=task)
    return int(payload["step"])


def _initialize_backbone_from_checkpoint(model: nn.Module, path: Path,
                                         map_location: torch.device,
                                         variant: str | None = None) -> dict[str, Any]:
    """Load adaptation weights but retain freshly initialized task heads."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise TypeError("initialization checkpoint has no model state mapping")
    source_variant = payload.get("variant")
    if variant is not None and source_variant is not None:
        if source_variant != variant and {source_variant, variant} != {"V2", "V3"}:
            raise ValueError(f"initialization variant {source_variant} is incompatible with {variant}")
    filtered = {name: value for name, value in state.items()
                if not any(part in name.lower() for part in TASK_HEAD_KEY_PARTS)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad_missing = [name for name in missing
                   if not any(part in name.lower() for part in TASK_HEAD_KEY_PARTS)]
    if bad_missing or unexpected:
        raise ValueError(f"checkpoint backbone mismatch; missing={bad_missing[:5]}, unexpected={unexpected[:5]}")
    _apply_gate_ramp(model, 1.0)
    return {"checkpoint": str(path.resolve()), "loaded_tensors": len(filtered),
            "source_variant": source_variant, "retained_task_head_tensors": len(missing),
            "retained_task_head_names": list(missing)}


@dataclass
class StepRecord:
    step: int
    loss: float
    optimizer_step_seconds: float
    nonpad_tokens: int
    nonpad_tokens_per_second: float
    memory: dict[str, int | None]
    gate_values: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "loss": self.loss,
            "optimizer_step_seconds": self.optimizer_step_seconds,
            "cuda_synchronized": self.memory.get("allocated_bytes") is not None,
            "nonpad_tokens": self.nonpad_tokens,
            "nonpad_tokens_per_second": self.nonpad_tokens_per_second,
            "memory": self.memory, "gate_values": self.gate_values,
        }


def _gate_values(model: nn.Module) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if "gate" in name.lower():
            values[f"logit/{name}"] = float(parameter.detach().float().mean().item())
    effective = getattr(model, "effective_gate_values", None)
    if callable(effective):
        for name, value in effective().items():
            if isinstance(value, Tensor):
                value = value.detach().float().mean().item()
            values[f"alpha/{name}"] = float(value)
    else:
        ramp = float(getattr(model, "gate_ramp", 1.0))
        for name, parameter in model.named_parameters():
            if "gate" in name.lower():
                values[f"alpha/{name}"] = float(torch.sigmoid(parameter.detach().float()).mean().item()) * ramp
    return values


def _step_start(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _step_end(device: torch.device, started: float) -> tuple[float, dict[str, int | None]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, _memory_stats(device)


def collate_sequences(sequences: Sequence[Sequence[int] | Tensor], *, pad_token_id: int,
                      device: torch.device) -> tuple[Tensor, Tensor]:
    tensors = [torch.as_tensor(sequence, dtype=torch.long) for sequence in sequences]
    if not tensors or any(t.numel() == 0 for t in tensors):
        raise ValueError("a batch must contain nonempty token sequences")
    width = max(int(t.numel()) for t in tensors)
    ids = torch.full((len(tensors), width), pad_token_id, dtype=torch.long)
    for row, tokens in enumerate(tensors):
        ids[row, :tokens.numel()] = tokens
    mask = ids.ne(pad_token_id).long()
    return ids.to(device), mask.to(device)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"prepared data file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
    if not records:
        raise ValueError(f"prepared data file is empty: {path}")
    return records


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mlm_split_sidecar(data_path: Path, row_count: int) -> dict[str, Any]:
    """If an adjacent row manifest exists, reject any non-train row explicitly."""
    sidecar = data_path.with_suffix(".jsonl")
    if not sidecar.is_file():
        return {"sidecar": str(sidecar), "status": "not_present", "verified_train_rows": None}
    seen = 0
    bad_split: list[int] = []
    missing_split: list[int] = []
    with sidecar.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "split" not in row:
                if len(missing_split) < 10:
                    missing_split.append(seen)
            elif row["split"] != "train":
                if len(bad_split) < 10:
                    bad_split.append(seen)
            seen += 1
    if seen != row_count:
        raise ValueError(f"MLM split sidecar has {seen} records for {row_count} array rows")
    if missing_split:
        raise ValueError(f"MLM split sidecar is missing split labels at rows {missing_split[:5]}")
    if bad_split:
        raise ValueError(f"MLM training array includes non-train rows at indices {bad_split[:10]}")
    return {"sidecar": str(sidecar), "status": "verified_train_only",
            "verified_train_rows": seen}


def _special_ids() -> list[int]:
    meta = json.loads(ADAPTATION_PATH.read_text(encoding="utf-8"))
    return [int(value) for value in meta["vocabulary"]["special_token_ids"]["all_special_token_ids"]]


def corrupt_mlm(input_ids: Tensor, attention_mask: Tensor, *, vocab_size: int,
                special_ids: Sequence[int], mask_token_id: int, seed: int,
                mode: str = "token", mask_rate: float = 0.30,
                span_mean: float = 3.0, span_max: int = 10) -> tuple[Tensor, Tensor, Tensor]:
    """Apply 80/10/10 MLM corruption while excluding all special positions and IDs."""
    if mode not in {"token", "span"}:
        raise ValueError("MLM mode must be token or span (balanced is selected per optimizer update)")
    if not 0 < mask_rate <= 1 or span_mean <= 0 or span_max < 1:
        raise ValueError("invalid masking parameters")
    if mask_token_id not in special_ids:
        raise ValueError("mask ID must be listed as a special token")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids_cpu = input_ids.detach().cpu()
    mask_cpu = attention_mask.detach().cpu().bool()
    eligible = mask_cpu.clone()
    for special_id in special_ids:
        eligible &= ids_cpu.ne(special_id)
    selected = torch.zeros_like(eligible)
    for row in range(ids_cpu.shape[0]):
        candidates = torch.nonzero(eligible[row], as_tuple=False).flatten().tolist()
        if not candidates:
            raise ValueError(f"row {row} has no attended non-special MLM positions")
        target_count = max(1, int(round(len(candidates) * mask_rate)))
        if mode == "token":
            perm = torch.randperm(len(candidates), generator=generator).tolist()
            selected[row, [candidates[index] for index in perm[:target_count]]] = True
            continue
        # A random permutation is the same without-replacement ordering as
        # repeatedly drawing a uniform start from the remaining candidates.
        # Pre-sampling one truncated geometric length per candidate avoids
        # rebuilding a shrinking O(L) candidate list for every span.
        permutation = torch.randperm(len(candidates), generator=generator).tolist()
        probability = min(1.0, 1.0 / span_mean)
        if span_max == 1 or probability >= 1.0:
            span_lengths = torch.ones(len(candidates), dtype=torch.long)
        else:
            uniforms = torch.rand(len(candidates), generator=generator, dtype=torch.float64)
            geometric = torch.floor(torch.log1p(-uniforms) / math.log1p(-probability)).long() + 1
            span_lengths = geometric.clamp_(max=span_max)
        chosen: set[int] = set()
        for candidate_index in permutation:
            start = candidates[candidate_index]
            if start in chosen:
                continue
            span_length = int(span_lengths[candidate_index])
            for delta in range(min(span_length, ids_cpu.shape[1] - start)):
                position = start + delta
                if not eligible[row, position] or position in chosen:
                    break
                chosen.add(position)
                if len(chosen) == target_count:
                    break
            if len(chosen) == target_count:
                break
        selected[row, list(chosen)] = True

    labels = torch.full_like(ids_cpu, -100)
    labels[selected] = ids_cpu[selected]
    corrupted = ids_cpu.clone()
    draws = torch.rand(ids_cpu.shape, generator=generator)
    to_mask = selected & draws.lt(0.8)
    to_random = selected & draws.ge(0.8) & draws.lt(0.9)
    corrupted[to_mask] = mask_token_id
    if to_random.any():
        # Uniformly sample from the non-special vocabulary without materializing
        # a 256k-entry allowed-ID table on every batch.
        special_set = set(int(v) for v in special_ids)
        sampled = torch.randint(0, vocab_size - len(special_set), (int(to_random.sum()),), generator=generator)
        for special_id in sorted(special_set):
            sampled += sampled.ge(special_id)
        corrupted[to_random] = sampled
    if torch.any(selected & ~eligible):
        raise AssertionError("MLM selected a special or padding position")
    return corrupted.to(input_ids.device), labels.to(input_ids.device), selected.to(input_ids.device)


def _mask_rate_and_output(model: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                          labels: Tensor) -> Tensor:
    output = _method(model, "forward_mlm")(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    return _model_loss(output)


def train_mlm_step(model: nn.Module, optimizer: AdamW, batches: Sequence[tuple[Tensor, Tensor]], *,
                   device: torch.device, vocab_size: int, special_ids: Sequence[int],
                   mask_token_id: int, seed: int, mode: str = "token",
                   accumulation: int = 1, gate_ramp_updates: int = 64,
                   optimizer_step: int = 0) -> StepRecord:
    if not batches:
        raise ValueError("MLM optimizer step requires at least one microbatch")
    optimizer.zero_grad(set_to_none=True)
    started = _step_start(device)
    if gate_ramp_updates > 0:
        _set_adaptation_update(model, optimizer_step, gate_ramp_updates)
    losses: list[float] = []
    nonpad = 0
    actual_accum = max(1, accumulation)
    for micro_index, (input_ids, attention_mask) in enumerate(batches):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        mode_for_batch = mode
        with _amp_context(device):
            corrupted, labels, _ = corrupt_mlm(
                input_ids, attention_mask, vocab_size=vocab_size, special_ids=special_ids,
                mask_token_id=mask_token_id, seed=seed + micro_index,
                mode=mode_for_batch,
            )
            loss = _mask_rate_and_output(model, corrupted, attention_mask, labels)
            scaled_loss = loss / actual_accum
        scaled_loss.backward()
        losses.append(float(loss.detach().float().item()))
        nonpad += int(attention_mask.sum().item())
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    elapsed, memory = _step_end(device, started)
    mean_loss = float(np.mean(losses))
    return StepRecord(optimizer_step + 1, mean_loss, elapsed, nonpad,
                      nonpad / max(elapsed, 1e-9), memory, _gate_values(model))


def _apply_gate_ramp(model: nn.Module, ramp: float) -> None:
    setter = getattr(model, "set_gate_ramp", None)
    if callable(setter):
        setter(float(ramp))


def _set_adaptation_update(model: nn.Module, update: int, ramp_updates: int) -> None:
    setter = getattr(model, "set_adaptation_step", None)
    if callable(setter):
        setter(int(update) + 1)
    else:
        _apply_gate_ramp(model, min(1.0, float(update + 1) / max(1, ramp_updates)))


def _classification_loss(model: nn.Module, input_ids: Tensor, attention_mask: Tensor,
                         labels: Tensor) -> Tensor:
    content_mask = attention_mask.bool().clone()
    for special_id in getattr(model, "special_token_ids", None) or _special_ids():
        content_mask &= input_ids.ne(int(special_id))
    with _amp_context(input_ids.device):
        output = _method(model, "forward_classification")(
            input_ids=input_ids, attention_mask=attention_mask, content_mask=content_mask
        )
        logits = _logits(output)
        return F.cross_entropy(logits.float(), labels)


def _qa_loss(model: nn.Module, input_ids: Tensor, attention_mask: Tensor,
             starts: Tensor, ends: Tensor, decode_mask: Tensor | None = None) -> Tensor:
    if decode_mask is None:
        decode_mask = attention_mask
    with _amp_context(input_ids.device):
        output = _method(model, "forward_qa")(
            input_ids=input_ids, attention_mask=attention_mask, context_mask=decode_mask
        )
        combined = _field(output, "logits", _field(output, "span_logits"))
        start_value = _field(output, "start_logits")
        end_value = _field(output, "end_logits")
        if isinstance(combined, Tensor) and combined.ndim == 3 and combined.shape[-1] == 2:
            start_logits, end_logits = combined[..., 0], combined[..., 1]
        elif isinstance(start_value, Tensor) and isinstance(end_value, Tensor):
            start_logits, end_logits = start_value, end_value
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            start_logits, end_logits = output[0], output[1]
        else:
            logits = _logits(output, ("logits", "span_logits"))
            if logits.ndim == 3 and logits.shape[-1] == 2:
                start_logits, end_logits = logits[..., 0], logits[..., 1]
            else:
                raise TypeError("forward_qa must return [B,L,2] or start/end logits")
        valid_positions = (decode_mask if decode_mask is not None else attention_mask).bool()
        start_logits = start_logits.masked_fill(~valid_positions, torch.finfo(start_logits.dtype).min)
        end_logits = end_logits.masked_fill(~valid_positions, torch.finfo(end_logits.dtype).min)
        return (F.cross_entropy(start_logits.float(), starts)
                + F.cross_entropy(end_logits.float(), ends)) / 2.0


def _task_loss(model: nn.Module, task: str, batch: Mapping[str, Any], device: torch.device) -> Tensor:
    if task == "classification":
        ids, mask = collate_sequences(batch["input_ids"], pad_token_id=0, device=device)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
        return _classification_loss(model, ids, mask, labels)
    if task == "qa":
        ids, mask = collate_sequences(batch["input_ids"], pad_token_id=0, device=device)
        starts = torch.as_tensor(batch["start_positions"], dtype=torch.long, device=device)
        ends = torch.as_tensor(batch["end_positions"], dtype=torch.long, device=device)
        decode_mask = mask
        if "context_mask" in batch:
            decode_mask = collate_sequences(batch["context_mask"], pad_token_id=0, device=device)[0]
            decode_mask = decode_mask.bool().long()
        return _qa_loss(model, ids, mask, starts, ends, decode_mask)
    raise ValueError(f"task loss {task!r} is not an accumulated single-view task")


def train_supervised_step(model: nn.Module, optimizer: AdamW, task: str,
                          microbatches: Sequence[Mapping[str, Any]], *,
                          device: torch.device, step: int,
                          accumulation: int | None = None) -> StepRecord:
    if task not in {"classification", "qa", "reranking"}:
        raise ValueError(f"unsupported supervised task: {task}")
    if not microbatches:
        raise ValueError("supervised optimizer step requires at least one microbatch")
    optimizer.zero_grad(set_to_none=True)
    started = _step_start(device)
    denominator = accumulation or len(microbatches)
    losses: list[float] = []
    nonpad = 0
    for batch in microbatches:
        if task == "reranking":
            ids, mask = collate_sequences(batch["input_ids"], pad_token_id=0, device=device)
            labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
            with _amp_context(device):
                output = _method(model, "forward_reranking")(
                    input_ids=ids, attention_mask=mask, group_size=3
                )
                scores = _logits(output, ("scores", "logits")).reshape(-1, 3)
                loss = F.cross_entropy(scores.float(), labels)
        else:
            loss = _task_loss(model, task, batch, device)
        (loss / denominator).backward()
        losses.append(float(loss.detach().float().item()))
        nonpad += int(batch.get("nonpad_tokens", 0))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    elapsed, memory = _step_end(device, started)
    return StepRecord(step + 1, float(np.mean(losses)), elapsed, nonpad,
                      nonpad / max(elapsed, 1e-9), memory, _gate_values(model))


def _capture_rng(device: torch.device) -> tuple[Tensor, Tensor | None]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None


def _restore_rng(state: tuple[Tensor, Tensor | None], device: torch.device) -> None:
    torch.set_rng_state(state[0].cpu())
    if device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _sequence_key(sequence: Sequence[int] | Tensor) -> tuple[int, ...]:
    if isinstance(sequence, Tensor):
        return tuple(int(x) for x in sequence.detach().cpu().tolist())
    return tuple(int(x) for x in sequence)


def _normalize_documents(value: Any) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("retrieval document field must be a nonempty token array")
    if isinstance(value[0], (int, np.integer)):
        return [[int(token) for token in value]]
    docs = [[int(token) for token in doc] for doc in value]
    return [doc for doc in docs if doc]


def choose_retrieval_physical_batch(records: Sequence[Mapping[str, Any]],
                                    override: int | None = None) -> int:
    """Use the planned 2/4 document microbatch based on this logical batch."""
    if override is not None:
        if override < 2:
            raise ValueError("physical_document_batch must be at least 2")
        return int(override)
    if not records:
        raise ValueError("cannot choose a retrieval microbatch for an empty logical batch")
    max_doc_len = 0
    for index, row in enumerate(records):
        if "positive_ids" not in row or "negative_ids" not in row:
            raise ValueError(f"retrieval row {index} needs positive_ids and negative_ids")
        documents = (_normalize_documents(row["positive_ids"])
                     + _normalize_documents(row["negative_ids"]))
        max_doc_len = max(max_doc_len, *(len(document) for document in documents))
    return 2 if max_doc_len >= 4096 else 4


def retrieval_logical_batch(records: Sequence[Mapping[str, Any]], *, device: torch.device,
                            pad_token_id: int = 0) -> dict[str, Any]:
    """Materialize B queries, B positives and B explicit negatives for one InfoNCE matrix."""
    if len(records) < 2:
        raise ValueError("retrieval InfoNCE requires at least two queries")
    queries: list[list[int]] = []
    positives: list[list[int]] = []
    negatives: list[list[int]] = []
    all_positive_keys: set[tuple[int, ...]] = set()
    for index, row in enumerate(records):
        if "query_ids" not in row or "positive_ids" not in row or "negative_ids" not in row:
            raise ValueError(f"retrieval row {index} needs query_ids, positive_ids, negative_ids")
        query = [int(token) for token in row["query_ids"]]
        pos_list = _normalize_documents(row["positive_ids"])
        neg_list = _normalize_documents(row["negative_ids"])
        if not query or not neg_list:
            raise ValueError(f"retrieval row {index} has an empty query or no explicit negative")
        queries.append(query)
        positives.append(pos_list[0])
        negatives.append(neg_list[0])
        all_positive_keys.update(_sequence_key(item) for item in pos_list)
    keys = [_sequence_key(doc) for doc in positives + negatives]
    valid_negative = torch.ones((len(records), 2 * len(records)), dtype=torch.bool)
    for row in range(len(records)):
        valid_negative[row, row] = True
        for column in range(2 * len(records)):
            if column != row and keys[column] in all_positive_keys:
                valid_negative[row, column] = False
    valid_negative[:, :len(records)] = False
    valid_negative[torch.arange(len(records)), torch.arange(len(records))] = True
    q_ids, q_mask = collate_sequences(queries, pad_token_id=pad_token_id, device=device)
    d_ids, d_mask = collate_sequences(positives + negatives, pad_token_id=pad_token_id, device=device)
    return {"query_ids": q_ids, "query_mask": q_mask, "document_ids": d_ids,
            "document_mask": d_mask, "positive_columns": torch.arange(len(records), device=device),
            "valid_negative": valid_negative.to(device), "query_count": len(records),
            "query_tokens": queries, "document_tokens": positives + negatives}


def _encode_retrieval(model: nn.Module, ids: Tensor, mask: Tensor) -> Tensor:
    content_mask = mask.bool().clone()
    for special_id in getattr(model, "special_token_ids", None) or _special_ids():
        content_mask &= ids.ne(int(special_id))
    with _amp_context(ids.device):
        return _embeddings(_method(model, "encode_retrieval")(
            input_ids=ids, attention_mask=mask, content_mask=content_mask
        ))


def _encode_microbatches(model: nn.Module, sequences: Sequence[Sequence[int]], *,
                         device: torch.device, microbatch: int, pad_token_id: int,
                         collect_rng: bool) -> tuple[Tensor, list[tuple[Tensor, Tensor | None]]]:
    chunks: list[Tensor] = []
    states: list[tuple[Tensor, Tensor | None]] = []
    for start in range(0, len(sequences), microbatch):
        ids, mask = collate_sequences(sequences[start:start + microbatch],
                                      pad_token_id=pad_token_id, device=device)
        if collect_rng:
            states.append(_capture_rng(device))
        chunks.append(_encode_retrieval(model, ids, mask))
    return torch.cat(chunks, dim=0), states


def train_retrieval_step(model: nn.Module, optimizer: AdamW,
                         records: Sequence[Mapping[str, Any]], *, device: torch.device,
                         step: int, physical_document_batch: int = 4,
                         physical_query_batch: int = 32,
                         temperature: float = 0.05, pad_token_id: int = 0) -> StepRecord:
    """Gradient-cache InfoNCE over the full logical query batch, then RNG replay."""
    if physical_document_batch < 2:
        raise ValueError("physical_document_batch must be at least 2")
    if physical_query_batch < 1:
        raise ValueError("physical_query_batch must be positive")
    logical = retrieval_logical_batch(records, device=device, pad_token_id=pad_token_id)
    q_sequences = logical["query_tokens"]
    d_sequences = logical["document_tokens"]
    optimizer.zero_grad(set_to_none=True)
    started = _step_start(device)
    backbone = getattr(model, "model", None)
    checkpointing_was_enabled = getattr(backbone, "gradient_checkpointing", None)
    if checkpointing_was_enabled is not None:
        backbone.gradient_checkpointing = False
    try:
        with torch.no_grad():
            q_cache, q_states = _encode_microbatches(
                model, q_sequences, device=device, microbatch=physical_query_batch,
                pad_token_id=pad_token_id, collect_rng=True)
            d_cache, d_states = _encode_microbatches(
                model, d_sequences, device=device, microbatch=physical_document_batch,
                pad_token_id=pad_token_id, collect_rng=True)
    finally:
        if checkpointing_was_enabled is not None:
            backbone.gradient_checkpointing = checkpointing_was_enabled
    post_cache_rng = _capture_rng(device)
    q_leaf = q_cache.detach().requires_grad_(True)
    d_leaf = d_cache.detach().requires_grad_(True)
    similarities = q_leaf.float() @ d_leaf.float().T / temperature
    similarities = similarities.masked_fill(~logical["valid_negative"], torch.finfo(similarities.dtype).min)
    loss = F.cross_entropy(similarities, logical["positive_columns"])
    loss.backward()
    q_gradient = q_leaf.grad.detach()
    d_gradient = d_leaf.grad.detach()
    q_offset = d_offset = 0
    for sequence_start in range(0, len(q_sequences), physical_query_batch):
        _restore_rng(q_states[q_offset], device)
        sequences = q_sequences[sequence_start:sequence_start + physical_query_batch]
        ids, mask = collate_sequences(sequences, pad_token_id=pad_token_id, device=device)
        q_live = _encode_retrieval(model, ids, mask)
        gradients = q_gradient[sequence_start:sequence_start + len(sequences)]
        torch.autograd.backward(q_live, gradients)
        q_offset += 1
    for sequence_start in range(0, len(d_sequences), physical_document_batch):
        _restore_rng(d_states[d_offset], device)
        sequences = d_sequences[sequence_start:sequence_start + physical_document_batch]
        ids, mask = collate_sequences(sequences, pad_token_id=pad_token_id, device=device)
        d_live = _encode_retrieval(model, ids, mask)
        gradients = d_gradient[sequence_start:sequence_start + len(sequences)]
        torch.autograd.backward(d_live, gradients)
        d_offset += 1
    _restore_rng(post_cache_rng, device)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    elapsed, memory = _step_end(device, started)
    nonpad = int(logical["query_mask"].sum() + logical["document_mask"].sum())
    return StepRecord(step + 1, float(loss.detach().item()), elapsed, nonpad,
                      nonpad / max(elapsed, 1e-9), memory, _gate_values(model))


def pair_tokens(query_ids: Sequence[int], document_ids: Sequence[int], *,
                sep_token_id: int = 1, max_length: int = 1024) -> list[int]:
    if max_length < 3:
        raise ValueError("max_length must allow query, separator and document tokens")
    query = list(map(int, query_ids))
    document = list(map(int, document_ids))
    # Prepared mmBERT sequences already carry BOS=2/EOS=1. Native pair
    # tokenization emits [BOS, query, EOS, document, EOS], so reuse the query
    # EOS as the separator and drop only the document's redundant BOS.
    bounded_pair = (len(query) >= 2 and len(document) >= 2
                    and query[0] == document[0] == 2
                    and query[-1] == document[-1] == sep_token_id)
    if bounded_pair:
        while len(query) + len(document) - 1 > max_length:
            query_content = len(query) - 2
            document_content = len(document) - 2
            if query_content >= document_content and query_content:
                query.pop(-2)
            elif document_content:
                document.pop(-2)
            else:
                break
        return query + document[1:]
    remaining = max_length - 1
    while len(query) + len(document) > remaining:
        if len(query) > len(document):
            query.pop()
        else:
            document.pop()
    return query + [sep_token_id] + document


def retrieval_to_reranking_records(records: Sequence[Mapping[str, Any]], *,
                                   sep_token_id: int = 1, max_length: int = 1024) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        positives = _normalize_documents(row["positive_ids"])
        if "negative_candidates" in row:
            raw_candidates = row["negative_candidates"]
            if not isinstance(raw_candidates, (list, tuple)):
                raise ValueError(f"retrieval row {row_index} negative_candidates must be a list")
            negatives = []
            for candidate_index, candidate in enumerate(raw_candidates):
                if isinstance(candidate, Mapping):
                    candidate_ids = candidate.get(
                        "input_ids", candidate.get("document_ids", candidate.get("ids")))
                else:
                    candidate_ids = candidate
                if candidate_ids is None:
                    raise ValueError(
                        f"retrieval row {row_index} negative candidate {candidate_index} has no token IDs")
                normalized = _normalize_documents(candidate_ids)
                if len(normalized) != 1:
                    raise ValueError(
                        f"retrieval row {row_index} negative candidate {candidate_index} must contain one token sequence")
                negatives.append(normalized[0])
        elif "negative_ids" in row:
            # Legacy prepared rows may already contain two nested negative
            # sequences. A flat token array remains one negative (the
            # retrieval objective's explicit negative), so it cannot silently
            # stand in for the reranking group's required two negatives.
            negatives = _normalize_documents(row["negative_ids"])
        else:
            negatives = []
        if len(negatives) < 2:
            raise ValueError(f"retrieval row {row_index} must provide two explicit reranking negatives")
        selected = [positives[0], negatives[0], negatives[1]]
        if len({_sequence_key(doc) for doc in selected}) != 3:
            raise ValueError(f"retrieval row {row_index} does not have one positive and two distinct negatives")
        derived.append({"group_id": row.get("id", str(row_index)),
                        "input_ids": [pair_tokens(row["query_ids"], doc,
                                                   sep_token_id=sep_token_id,
                                                   max_length=max_length) for doc in selected],
                        "label": 0, "query_ids": row["query_ids"]})
    return derived


def _fixed_batches(records: Sequence[Mapping[str, Any]], batch_size: int,
                   *, repeat: bool = False) -> Iterator[list[Mapping[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    index = 0
    while True:
        batch = list(records[index:index + batch_size])
        if len(batch) < batch_size:
            if not batch or not repeat:
                return
            batch += list(records[:batch_size - len(batch)])
        yield batch
        index += batch_size
        if index >= len(records):
            if not repeat:
                return
            index %= len(records)


def _cyclic_batch(records: Sequence[Mapping[str, Any]], batch_size: int,
                  step: int) -> list[Mapping[str, Any]]:
    if not records:
        raise ValueError("prepared data contains no records")
    offset = (step * batch_size) % len(records)
    return [records[(offset + index) % len(records)] for index in range(batch_size)]


def _epoch_sample_indices(size: int, count: int, *, seed: int, offset: int) -> np.ndarray:
    """Reconstruct a shuffled, no-replacement epoch cursor from seed and offset."""
    if size < 1 or count < 1 or offset < 0:
        raise ValueError("size/count must be positive and offset nonnegative")
    selected: list[np.ndarray] = []
    cursor = offset
    remaining = count
    while remaining:
        epoch, within_epoch = divmod(cursor, size)
        order = np.random.default_rng(seed + epoch).permutation(size)
        take = min(remaining, size - within_epoch)
        selected.append(order[within_epoch:within_epoch + take])
        cursor += take
        remaining -= take
    return np.concatenate(selected).astype(np.int64, copy=False)


def _epoch_batch(records: Sequence[Mapping[str, Any]], batch_size: int,
                 step: int, *, seed: int) -> list[Mapping[str, Any]]:
    indices = _epoch_sample_indices(len(records), batch_size, seed=seed,
                                    offset=step * batch_size)
    return [records[int(index)] for index in indices]


def _prepare_downstream_batch(task: str, rows: Sequence[Mapping[str, Any]], *,
                              pad_token_id: int) -> dict[str, Any]:
    if task == "classification":
        return {"input_ids": [row["input_ids"] for row in rows],
                "labels": [int(row["label"]) for row in rows],
                "nonpad_tokens": sum(len(row["input_ids"]) for row in rows)}
    if task == "qa":
        if any(int(row["start_positions"]) < 0 or int(row["end_positions"]) < 0 for row in rows):
            raise ValueError("QA training batches must contain answer-bearing windows with valid span labels")
        return {"input_ids": [row["input_ids"] for row in rows],
                "start_positions": [int(row["start_positions"]) for row in rows],
                "end_positions": [int(row["end_positions"]) for row in rows],
                "context_mask": [row.get("context_mask", [1] * len(row["input_ids"])) for row in rows],
                "nonpad_tokens": sum(len(row["input_ids"]) for row in rows)}
    raise ValueError(f"cannot prepare standard batch for {task}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def run_training(*, model: nn.Module, task: str, variant: str, data_path: Path,
                 steps: int, output_dir: Path, device: torch.device,
                 seed: int = 17, resume_from: Path | None = None,
                 batch_size: int | None = None, accumulation: int | None = None,
                 model_dir: Path | None = None, length: int | None = None,
                 mode: str = "token", physical_document_batch: int | None = None,
                 stop_after_steps: int | None = None,
                 checkpoint_every_steps: int = 0,
                 final_weights_only: bool = False,
                 schedule_total_steps: int | None = None,
                 init_from: Path | None = None,
                 shared_prefix_resume: bool = False) -> dict[str, Any]:
    """Execute exactly ``steps`` optimizer updates (or the remaining resumed steps)."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if steps < 1:
        raise ValueError("steps must be explicitly positive; full matrix auto-run is disabled")
    plan = read_plan()
    if task == "mlm":
        if length not in {512, 1024, 4096, 8192}:
            raise ValueError("MLM requires an explicit length in {512,1024,4096,8192}")
        if data_path.suffix.lower() != ".npy" or not data_path.is_file():
            raise FileNotFoundError(f"expected a prepared wiki_{length}.npy file: {data_path}")
        array = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != length:
            raise ValueError(f"{data_path} must be int32 [N,{length}], got {array.dtype} {array.shape}")
        if len(array) < 1:
            raise ValueError("MLM prepared array has no rows")
        mlm_split_provenance = validate_mlm_split_sidecar(data_path, len(array))
        adaptation = plan["adaptation"]
        stage = next(item for item in adaptation["stages"] if int(item["max_length"]) == length)
        stage_schedule = next(item for item in adaptation_stage_schedule(plan)
                              if int(item["max_length"]) == length)
        per_device = batch_size or stage["physical_microbatch"]
        accum = accumulation or stage["gradient_accumulation"]
        if per_device < 2:
            raise ValueError("physical microbatch one is prohibited by the experiment plan")
        if model_dir is None:
            raise ValueError("--model-dir is required for local offline model construction")
        model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        vocab_size = int(model_config.get("vocab_size", 256000))
        mask_token_id = int(model_config.get("mask_token_id", 4))
        special_ids = _special_ids()
        data_records = None
        sampler_stage_start = int(stage_schedule["start_step"])
        qa_groups: dict[str, list[dict[str, Any]]] = {}
        qa_regular_steps = 0
    else:
        records = load_jsonl(data_path)
        vocab_size = 0
        mask_token_id = 4
        special_ids = _special_ids()
        data_records = records
        sampler_stage_start = 0
        qa_groups = {}
        qa_regular_steps = 0
        if task == "retrieval":
            task_plan = plan["downstream"]["retrieval"]
            logical_retrieval_queries = int(task_plan["logical_batch_queries"])
            physical_query_batch = int(task_plan["physical_query_batch"])
            per_device = (physical_document_batch if physical_document_batch is not None
                          else int(task_plan["physical_document_batch"]))
            accum = 1
            if per_device is not None and per_device < 2:
                raise ValueError("physical document batch must be at least 2")
            if batch_size is not None and batch_size != logical_retrieval_queries:
                raise ValueError(f"retrieval logical query batch is {logical_retrieval_queries}; "
                                 "configure document microbatch with --physical-document-batch")
        elif task == "reranking":
            task_plan = plan["downstream"]["reranking"]
            per_device = batch_size or int(task_plan["physical_query_groups"])
            accum = accumulation or int(task_plan["accumulation"])
        elif task == "qa":
            task_plan = plan["downstream"]["qa"]
            long_plan = task_plan.get("long_augmentation", {})
            regular_max_length = int(task_plan["max_length"])
            qa_groups = {
                "regular": [row for row in records
                            if len(row["input_ids"]) <= regular_max_length
                            and row.get("window_type") != "long"],
                "long": [row for row in records
                         if len(row["input_ids"]) > regular_max_length
                         or row.get("window_type") == "long"],
            }
            if not qa_groups["regular"]:
                raise ValueError(f"prepared QA data has no regular <={regular_max_length}-token features")
            qa_steps = qa_phase_steps(plan)
            qa_regular_steps = qa_steps["regular"]
            planned_long_steps = qa_steps["long"]
            per_device = batch_size or int(task_plan["physical_batch"])
            accum = accumulation or int(task_plan["accumulation"])
            if steps > qa_regular_steps and not qa_groups["long"]:
                raise ValueError("requested QA steps include the long augmentation phase, "
                                 "but no long windows are prepared")
            if steps > qa_regular_steps + planned_long_steps:
                raise ValueError(f"requested QA budget {steps} exceeds prepared regular+long schedule "
                                 f"({qa_regular_steps + planned_long_steps})")
        else:
            task_plan = plan["downstream"][task]
            per_device = batch_size or int(task_plan["physical_batch"])
            accum = accumulation or int(task_plan["accumulation"])
    if task != "retrieval" and per_device < 2:
        raise ValueError("physical batch size one is prohibited by the experiment plan")
    model.train()
    task_config = plan.get("adaptation", {}) if task == "mlm" else plan["downstream"][task]
    optimizer = optimizer_for(
        model,
        backbone_lr=float(task_config.get("backbone_lr", plan["adaptation"]["optimizer"]["backbone_lr"])),
        gate_lr=float(plan["adaptation"]["optimizer"]["gate_lr"] if task == "mlm"
                      else task_config.get("backbone_lr", 1e-5)),
        weight_decay=float(task_config.get("weight_decay", 0.01)),
        head_lr=(None if task == "mlm" else float(task_config.get(
            "head_lr", plan["downstream"]["shared"]["head_lr"]))),
    )
    scheduler = _scheduler(optimizer, schedule_total_steps or steps,
                            float(task_config.get("warmup_fraction", 0.05)))
    start_step = 0
    initialization = None
    if init_from is not None:
        if task == "mlm":
            raise ValueError("--init-from for MLM is unsupported; use --resume-from to restore optimizer state")
        initialization = _initialize_backbone_from_checkpoint(model, init_from, device, variant=variant)
    if resume_from:
        start_step = _load_checkpoint(resume_from, model=model, optimizer=optimizer,
                                      scheduler=scheduler, task=task, variant=variant,
                                      map_location=device,
                                      allow_shared_variant_alias=shared_prefix_resume)
        if start_step >= steps:
            raise ValueError(f"resume checkpoint already completed {start_step} of {steps} steps")
    elif task != "mlm":
        # A fresh downstream branch starts from an adapted MLM checkpoint (or
        # the base model) with the structural gates fully active. Resumed
        # downstream jobs retain the checkpoint's saved/legacy ramp instead.
        _apply_gate_ramp(model, 1.0)
    if task == "mlm" and start_step < sampler_stage_start:
        raise ValueError(f"length {length} begins at absolute step {sampler_stage_start}; "
                         "resume from the preceding planned MLM stage")
    run_steps = min(steps - start_step, stop_after_steps or steps)
    if run_steps < 1:
        raise ValueError("no remaining steps requested")
    log_path = output_dir / f"{variant}_{task}_training.jsonl"
    checkpoint_path = output_dir / f"{variant}_{task}_checkpoint.pt"
    data_sha256 = dataset_fingerprint(data_path)
    records_out: list[dict[str, Any]] = []
    sampled_row_indices: list[int] = []
    for local_step in range(run_steps):
        global_step = start_step + local_step
        if task == "mlm":
            total_batch = per_device * accum
            sampler_offset = (global_step - sampler_stage_start) * total_batch
            chosen = _epoch_sample_indices(len(array), total_batch, seed=seed,
                                           offset=max(0, sampler_offset))
            sampled_row_indices.extend(int(index) for index in chosen)
            slices: list[tuple[Tensor, Tensor]] = []
            for chunk_start in range(0, total_batch, per_device):
                rows = np.asarray(array[chosen[chunk_start:chunk_start + per_device]], dtype=np.int64)
                ids = torch.from_numpy(rows).to(device)
                slices.append((ids, ids.ne(0).long()))
            effective_mode = ("token" if global_step % 2 == 0 else "span") if mode == "balanced" else mode
            step_record = train_mlm_step(
                model, optimizer, slices, device=device, vocab_size=vocab_size,
                special_ids=special_ids, mask_token_id=mask_token_id,
                seed=seed + global_step * accum, mode=effective_mode, accumulation=accum,
                optimizer_step=global_step,
            )
        elif task == "retrieval":
            indices = _epoch_sample_indices(len(data_records or []), logical_retrieval_queries,
                                            seed=seed, offset=global_step * logical_retrieval_queries)
            sampled_row_indices.extend(int(index) for index in indices)
            row_batch = [(data_records or [])[int(index)] for index in indices]
            retrieval_physical_batch = choose_retrieval_physical_batch(
                row_batch, override=per_device)
            step_record = train_retrieval_step(
                model, optimizer, row_batch, device=device, step=global_step,
                physical_document_batch=retrieval_physical_batch,
                physical_query_batch=physical_query_batch,
            )
        elif task == "reranking":
            rerank_rows = retrieval_to_reranking_records(data_records or [], max_length=1024)
            groups_per_update = per_device * accum
            indices = _epoch_sample_indices(len(rerank_rows), groups_per_update, seed=seed,
                                            offset=global_step * groups_per_update)
            sampled_row_indices.extend(int(index) for index in indices)
            selected_groups = [rerank_rows[int(index)] for index in indices]
            microbatches: list[dict[str, Any]] = []
            for start in range(0, len(selected_groups), per_device):
                groups = selected_groups[start:start + per_device]
                microbatches.append({
                    "input_ids": [pair for group in groups for pair in group["input_ids"]],
                    "labels": [0] * len(groups),
                    "nonpad_tokens": sum(len(pair) for group in groups for pair in group["input_ids"]),
                })
            step_record = train_supervised_step(
                model, optimizer, task, microbatches, device=device, step=global_step,
                accumulation=accum,
            )
        elif task == "qa":
            if global_step < qa_regular_steps:
                current_rows = qa_groups["regular"]
                current_step = global_step
                current_physical = int(task_plan["physical_batch"])
                current_accumulation = int(task_plan["accumulation"])
                subphase = "regular"
            else:
                current_rows = qa_groups["long"]
                current_step = global_step - qa_regular_steps
                current_physical = int(task_plan["long_augmentation"]["physical_batch"])
                current_accumulation = int(task_plan["long_augmentation"]["accumulation"])
                subphase = "long"
            row_count = current_physical * current_accumulation
            indices = _epoch_sample_indices(len(current_rows), row_count, seed=seed,
                                            offset=current_step * row_count)
            row_base = len(qa_groups["regular"]) if subphase == "long" else 0
            sampled_row_indices.extend(row_base + int(index) for index in indices)
            row_batch = [current_rows[int(index)] for index in indices]
            microbatches = []
            for offset in range(0, len(row_batch), current_physical):
                microbatches.append(_prepare_downstream_batch(
                    task, row_batch[offset:offset + current_physical], pad_token_id=0))
            step_record = train_supervised_step(
                model, optimizer, task, microbatches, device=device, step=global_step,
                accumulation=current_accumulation,
            )
        else:
            indices = _epoch_sample_indices(len(data_records or []), per_device * accum,
                                            seed=seed, offset=global_step * per_device * accum)
            sampled_row_indices.extend(int(index) for index in indices)
            row_batch = [(data_records or [])[int(index)] for index in indices]
            microbatches = []
            for offset in range(0, len(row_batch), per_device):
                microbatches.append(_prepare_downstream_batch(
                    task, row_batch[offset:offset + per_device], pad_token_id=0))
            step_record = train_supervised_step(
                model, optimizer, task, microbatches, device=device, step=global_step,
                accumulation=accum,
            )
        scheduler.step()
        record = step_record.to_dict()
        record.update({"task": task, "variant": variant, "seed": seed,
                       "length": length, "mlm_mode": effective_mode if task == "mlm" else None,
                       "data_sha256": data_sha256,
                       "physical_batch": (current_physical if task == "qa" else
                                          per_device if task != "retrieval" else None),
                       "gradient_accumulation": (current_accumulation if task == "qa" else accum),
                       "effective_batch": (current_physical * current_accumulation if task == "qa" else
                                           logical_retrieval_queries if task == "retrieval" else
                                           per_device * accum),
                       "learning_rates": [float(group["lr"]) for group in optimizer.param_groups]})
        if task == "retrieval":
            record["physical_document_batch"] = retrieval_physical_batch
            record["physical_query_batch"] = physical_query_batch
            record["logical_retrieval_queries"] = logical_retrieval_queries
        records_out.append(record)
        _write_jsonl(log_path, [record])
        is_final = local_step == run_steps - 1
        should_save = is_final or (checkpoint_every_steps > 0 and (global_step + 1) % checkpoint_every_steps == 0)
        if should_save:
            checkpoint_metadata = {
                "data_sha256": data_sha256, "length": length,
                "model_dir": str(model_dir.resolve()) if model_dir else None,
                "planned_total_steps": schedule_total_steps or steps,
                "sampler": {"seed": seed, "next_global_step": global_step + 1,
                            "stage_start_step": sampler_stage_start,
                            "rows_per_optimizer_step": (logical_retrieval_queries if task == "retrieval"
                                                        else per_device * accum),
                            "physical_document_batch": retrieval_physical_batch if task == "retrieval" else None,
                            "physical_query_batch": physical_query_batch if task == "retrieval" else None,
                            "qa_regular_steps": qa_regular_steps if task == "qa" else None,
                            "qa_regular_rows": len(qa_groups.get("regular", [])),
                            "qa_long_rows": len(qa_groups.get("long", [])),
                            "data_rows": len(array) if task == "mlm" else len(data_records or [])},
            }
            reached_requested_steps = is_final and global_step + 1 == steps
            if final_weights_only and reached_requested_steps:
                _save_weights_only(checkpoint_path, model=model, step=global_step + 1,
                                   task=task, variant=variant, seed=seed,
                                   metadata=checkpoint_metadata)
            else:
                _save_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
                                 scheduler=scheduler, step=global_step + 1,
                                 task=task, variant=variant, seed=seed,
                                 metadata=checkpoint_metadata)
    unique_rows = len(set(sampled_row_indices))
    return {"task": task, "variant": variant, "seed": seed, "requested_steps": steps,
            "completed_this_call": run_steps, "last_step": start_step + run_steps,
            "checkpoint_kind": "weights_only" if final_weights_only and start_step + run_steps == steps
            else "full_state",
            "checkpoint": str(checkpoint_path), "log": str(log_path),
            "data_sha256": data_sha256,
            "split_provenance": mlm_split_provenance if task == "mlm" else None,
            "initialization": initialization,
            "sampler": {"seeded_epoch_permutation": True,
                        "drawn_rows_this_call": len(sampled_row_indices),
                        "unique_rows_this_call": unique_rows,
                        "repeated_draws_this_call": len(sampled_row_indices) - unique_rows,
                        "row_coverage": unique_rows / max(1, len(array) if task == "mlm" else len(data_records or []))},
            "records": records_out}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--variant", required=True, choices=("V0", "V1", "V2", "V3"))
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="local model/checkpoint directory; remote identifiers are not accepted")
    parser.add_argument("--data", required=True, type=Path, help="prepared local file")
    parser.add_argument("--steps", required=True, type=int,
                        help="explicit update budget; omitted full-matrix runs are disabled")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--length", type=int, choices=(512, 1024, 4096, 8192))
    parser.add_argument("--mode", choices=("token", "span", "balanced"), default="token")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--physical-document-batch", type=int,
                        help="retrieval document microbatch override; default is 2 for batches containing a >=4096-token document, otherwise 4")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--init-from", type=Path,
                        help="initialize backbone/gates only from an MLM checkpoint; task heads stay newly seeded")
    parser.add_argument("--shared-prefix-resume", action="store_true",
                        help="allow a V2 MLM checkpoint to seed V3's shared-prefix continuation")
    parser.add_argument("--schedule-total-steps", type=int,
                        help="absolute optimizer-update count for a multi-stage LR schedule")
    parser.add_argument("--stop-after-steps", type=int,
                        help="save a resumable partial run after this many updates")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0,
                        help="optional periodic checkpoint interval; stage end is always saved")
    parser.add_argument("--final-weights-only", action="store_true",
                        help="save model weights only at the requested final step; intermediate/early-stop checkpoints remain resumable")
    parser.add_argument("--attention-backend", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    _seed_everything(args.seed)
    model = load_nextbert(args.model_dir, args.variant, device, args.attention_backend)
    result = run_training(
        model=model, task=args.task, variant=args.variant, data_path=args.data,
        steps=args.steps, output_dir=args.output_dir, device=device, seed=args.seed,
        resume_from=args.resume_from, batch_size=args.batch_size,
        accumulation=args.accumulation, model_dir=args.model_dir, length=args.length,
        mode=args.mode, physical_document_batch=args.physical_document_batch,
        stop_after_steps=args.stop_after_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        final_weights_only=args.final_weights_only,
        schedule_total_steps=args.schedule_total_steps,
        init_from=args.init_from,
        shared_prefix_resume=args.shared_prefix_resume,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
