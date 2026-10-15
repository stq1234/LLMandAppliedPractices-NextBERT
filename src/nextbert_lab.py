"""CPU-only architecture accounting and mechanical training checks for NextGenBERT.

The large-model inventory is instantiated on the PyTorch meta device from the
published mmBERT configuration. The tiny-model updates use random token IDs and
only establish that the chosen interfaces and losses execute; they are not NLP
quality experiments.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Sequence

# Keep this process within the assigned CPU budget before importing torch.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "2"

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import ModernBertConfig, ModernBertForMaskedLM


THREAD_LIMIT = 2
CHECKPOINT_REVISION = "8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c"
MMBERT_CONFIG = {
    "vocab_size": 256000,
    "hidden_size": 768,
    "num_hidden_layers": 22,
    "intermediate_size": 1152,
    "num_attention_heads": 12,
    "max_position_embeddings": 8192,
    "local_attention": 128,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000,
    "local_rope_theta": 160000,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "norm_bias": False,
    "pad_token_id": 0,
    "unk_token_id": 3,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "cls_token_id": 1,
    "sep_token_id": 1,
    "mask_token_id": 4,
    "embedding_dropout": 0.0,
    "mlp_dropout": 0.0,
    "attention_dropout": 0.0,
}

_ADAPTATION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mmbert_adaptation.json"
_ADAPTATION_METADATA = json.loads(_ADAPTATION_CONFIG_PATH.read_text(encoding="utf-8"))
_TOKENIZER_METADATA = _ADAPTATION_METADATA["vocabulary"]["special_token_ids"]
PINNED_SPECIAL_TOKEN_IDS = tuple(_TOKENIZER_METADATA["all_special_token_ids"])
PINNED_MASK_TOKEN_ID = int(_TOKENIZER_METADATA["mask"])
if len(set(PINNED_SPECIAL_TOKEN_IDS)) != len(PINNED_SPECIAL_TOKEN_IDS):
    raise ValueError("the pinned tokenizer special-token ID list must not contain duplicates")
if PINNED_MASK_TOKEN_ID not in PINNED_SPECIAL_TOKEN_IDS:
    raise ValueError("the pinned mask-token ID must be included in the special-token ID list")


def configure_cpu() -> None:
    torch.set_num_threads(THREAD_LIMIT)
    torch.set_num_interop_threads(THREAD_LIMIT)
    random.seed(20261015)
    torch.manual_seed(20261015)


def make_mmbert_config(**overrides: Any) -> ModernBertConfig:
    values = dict(MMBERT_CONFIG)
    values.update(overrides)
    return ModernBertConfig(**values)


def _count_unique_parameters(module: nn.Module) -> int:
    unique: dict[int, nn.Parameter] = {}
    for parameter in module.parameters():
        unique[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in unique.values())


def _parameter_inventory() -> dict[str, Any]:
    config = make_mmbert_config()
    # Meta tensors preserve the official module structure and tied weights
    # without allocating or downloading the 307M checkpoint.
    with torch.device("meta"):
        model = ModernBertForMaskedLM(config)

    total = _count_unique_parameters(model)
    embedding_weight = model.get_input_embeddings().weight
    tied = embedding_weight is model.get_output_embeddings().weight
    embedding_parameters = embedding_weight.numel()
    encoder_parameters = _count_unique_parameters(model.model.layers)
    other_parameters = total - embedding_parameters - encoder_parameters
    layout = attention_layout(model)

    return {
        "checkpoint": "jhu-clsp/mmBERT-base",
        "revision": CHECKPOINT_REVISION,
        "instantiated_from_config_without_weights": True,
        "total_parameters": total,
        "token_embedding_parameters": embedding_parameters,
        "transformer_block_parameters": encoder_parameters,
        "embedding_norm_and_mlm_head_parameters": other_parameters,
        "input_output_embeddings_tied": tied,
        "max_position_embeddings": config.max_position_embeddings,
        "config": {key: getattr(config, key) for key in MMBERT_CONFIG},
        "attention_layout": layout,
    }


def attention_layout(model: ModernBertForMaskedLM) -> dict[str, Any]:
    global_layers: list[int] = []
    local_layers: list[int] = []
    for index, layer in enumerate(model.model.layers):
        if tuple(layer.attn.local_attention) == (-1, -1):
            global_layers.append(index)
        else:
            local_layers.append(index)
    return {
        "global_layer_indices_zero_based": global_layers,
        "local_layer_indices_zero_based": local_layers,
        "global_layer_count": len(global_layers),
        "local_layer_count": len(local_layers),
        "local_window_config_tokens": model.config.local_attention,
        "local_layer_symmetric_radius": model.config.local_attention // 2,
        "global_every_n_layers": model.config.global_attn_every_n_layers,
    }


def scratch_blueprint_inventory() -> dict[str, Any]:
    layers = 28
    hidden = 1024
    heads = 16
    branch = 1536
    vocabulary = 48000
    attention_per_layer = 4 * hidden * hidden
    geglu_per_layer = 3 * hidden * branch
    transformer_parameters = layers * (attention_per_layer + geglu_per_layer)
    embedding_parameters = vocabulary * hidden
    # MLM transform dense matrix, decoder bias, prediction norm, two block
    # norms per layer, and embedding/final norms; all norms omit bias.
    head_and_norm_parameters = (
        hidden * hidden
        + vocabulary
        + hidden
        + (2 * hidden * layers)
        + (2 * hidden)
    )
    return {
        "status": "untrained_alternative_blueprint",
        "layers": layers,
        "hidden_size": hidden,
        "attention_heads": heads,
        "geglu_branch_size": branch,
        "vocabulary_size": vocabulary,
        "max_position_embeddings": 8192,
        "attention_parameters_per_layer": attention_per_layer,
        "geglu_parameters_per_layer": geglu_per_layer,
        "transformer_block_parameters": transformer_parameters,
        "tied_token_embedding_parameters": embedding_parameters,
        "mlm_head_and_norm_parameters": head_and_norm_parameters,
        "estimated_total_parameters": transformer_parameters + embedding_parameters + head_and_norm_parameters,
        "tied_embeddings": True,
        "biases": False,
    }


def make_tiny_config(*, layers: int = 4, hidden: int = 64, heads: int = 4,
                     intermediate: int = 96, local_window: int = 8,
                     global_every: int = 3, dropout: float = 0.1) -> ModernBertConfig:
    return ModernBertConfig(
        vocab_size=512,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        max_position_embeddings=8192,
        local_attention=local_window,
        global_attn_every_n_layers=global_every,
        global_rope_theta=160000,
        local_rope_theta=160000,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        norm_bias=False,
        pad_token_id=0,
        unk_token_id=3,
        bos_token_id=2,
        eos_token_id=1,
        cls_token_id=1,
        sep_token_id=1,
        mask_token_id=4,
        embedding_dropout=dropout,
        mlp_dropout=dropout,
        attention_dropout=dropout,
    )


def make_synthetic_batch(batch_size: int = 3, sequence_length: int = 18,
                         vocabulary_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1701)
    input_ids = torch.randint(10, vocabulary_size, (batch_size, sequence_length), generator=generator)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    input_ids[:, 0] = 2  # BOS
    input_ids[:, sequence_length // 2] = 1  # SEP/document boundary
    input_ids[:, -1] = 1  # SEP
    input_ids[0, -2:] = 0  # right padding in one example
    attention_mask[0, -2:] = 0
    return input_ids, attention_mask


def make_special_token_probe(vocabulary_size: int,
                             special_token_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in special_token_ids):
        raise ValueError("the special-token probe requires IDs inside the synthetic vocabulary")
    special_id_set = set(special_token_ids)
    ordinary_ids = [token_id for token_id in range(10, vocabulary_size)
                    if token_id not in special_id_set]
    if len(ordinary_ids) < 2:
        raise ValueError("the special-token probe requires two ordinary token IDs")
    input_ids = torch.tensor([[ordinary_ids[0], *special_token_ids, ordinary_ids[1]]],
                             dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _eligible_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   special_token_ids: Sequence[int]) -> torch.Tensor:
    eligible = attention_mask.bool().clone()
    for special_id in special_token_ids:
        eligible &= input_ids.ne(special_id)
    return eligible


def dynamic_corruption(input_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                       mode: str, mask_rate: float = 0.30,
                       vocabulary_size: int = 512, seed: int = 991,
                       special_token_ids: Sequence[int], mask_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create MLM examples excluding padding and every supplied tokenizer special ID."""
    if mode not in {"token", "span"}:
        raise ValueError("mode must be 'token' or 'span'")
    if mask_token_id not in special_token_ids:
        raise ValueError("mask_token_id must be included in special_token_ids")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eligible = _eligible_mask(input_ids, attention_mask, special_token_ids)
    selected = torch.zeros_like(eligible)

    for row in range(input_ids.shape[0]):
        candidates = torch.nonzero(eligible[row], as_tuple=False).flatten().tolist()
        if not candidates:
            raise ValueError(f"example {row} has no eligible prediction tokens")
        target = max(1, round(len(candidates) * mask_rate))
        if mode == "token":
            order = torch.randperm(len(candidates), generator=generator).tolist()
            selected[row, [candidates[i] for i in order[:target]]] = True
            continue

        chosen: set[int] = set()
        while len(chosen) < target:
            remaining = [position for position in candidates if position not in chosen]
            if not remaining:
                break
            start = remaining[int(torch.randint(len(remaining), (1,), generator=generator).item())]
            span_length = int(torch.randint(1, 5, (1,), generator=generator).item())
            added = 0
            for position in range(start, min(start + span_length, input_ids.shape[1])):
                if not eligible[row, position] or position in chosen:
                    break
                chosen.add(position)
                added += 1
                if len(chosen) >= target:
                    break
            if added == 0:
                chosen.add(start)
        selected[row, list(chosen)] = True

    labels = torch.full_like(input_ids, -100)
    labels[selected] = input_ids[selected]
    corrupted = input_ids.clone()
    replacement_draw = torch.rand(input_ids.shape, generator=generator)
    replace_with_mask = selected & replacement_draw.lt(0.8)
    replace_randomly = selected & replacement_draw.ge(0.8) & replacement_draw.lt(0.9)
    corrupted[replace_with_mask] = mask_token_id
    random_tokens = torch.randint(10, vocabulary_size, input_ids.shape, generator=generator)
    corrupted[replace_randomly] = random_tokens[replace_randomly]

    if torch.any(selected & ~eligible):
        raise AssertionError("a special or padded token was selected for prediction")
    if torch.any(selected.sum(dim=1) == 0):
        raise AssertionError("every example with eligible tokens must select at least one target")
    return corrupted, labels, selected


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denominator


def _assert_finite_gradients(parameters: list[nn.Parameter], name: str) -> float:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        raise AssertionError(f"{name}: no gradients were produced")
    if not all(torch.isfinite(gradient).all().item() for gradient in grads):
        raise AssertionError(f"{name}: non-finite gradient found")
    norm = torch.sqrt(sum(gradient.detach().float().pow(2).sum() for gradient in grads)).item()
    if not norm > 0:
        raise AssertionError(f"{name}: gradients are all zero")
    return norm


def _train_one_step(model: ModernBertForMaskedLM, optimizer: AdamW, loss: torch.Tensor,
                    name: str) -> dict[str, Any]:
    watched = model.model.layers[0].attn.Wqkv.weight
    before = watched.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = _assert_finite_gradients([watched], name)
    optimizer.step()
    changed = not torch.equal(before, watched.detach())
    if not changed:
        raise AssertionError(f"{name}: watched encoder attention weights did not update")
    return {"loss": float(loss.detach().item()), "encoder_gradient_norm": gradient_norm,
            "encoder_parameter_updated": changed}


class TaskHeads(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classification = nn.Linear(hidden_size, 3)
        self.extraction = nn.Linear(hidden_size, 2)
        self.embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reranking = nn.Linear(hidden_size, 1)
        self.view_dropout = nn.Dropout(0.1)


def _make_cross_encoder_pairs(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = input_ids.shape[0]
    pair_length = 11
    pairs = torch.zeros((batch_size, pair_length), dtype=torch.long)
    mask = torch.zeros_like(pairs)
    for row in range(batch_size):
        pairs[row, 0] = 2
        pairs[row, 1:5] = input_ids[row, 1:5]
        pairs[row, 5] = 1
        pairs[row, 6:10] = input_ids[(row + 1) % batch_size, 1:5]
        pairs[row, 10] = 1
        mask[row, :11] = 1
    return pairs, mask


def _span_extraction_loss(token_logits: torch.Tensor, start_positions: torch.Tensor,
                          end_positions: torch.Tensor, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          special_token_ids: Sequence[int]) -> torch.Tensor:
    valid_positions = _eligible_mask(input_ids, attention_mask, special_token_ids)
    start_valid = valid_positions.gather(1, start_positions.unsqueeze(1)).all()
    end_valid = valid_positions.gather(1, end_positions.unsqueeze(1)).all()
    if not start_valid.item() or not end_valid.item():
        raise ValueError("gold extraction spans must point to attended, non-special token positions")
    masked_logits = token_logits.masked_fill(
        ~valid_positions.unsqueeze(-1), torch.finfo(token_logits.dtype).min
    )
    return (
        F.cross_entropy(masked_logits[:, :, 0], start_positions)
        + F.cross_entropy(masked_logits[:, :, 1], end_positions)
    )


def _synthetic_gold_spans(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          special_token_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    valid_positions = _eligible_mask(input_ids, attention_mask, special_token_ids)
    starts: list[int] = []
    ends: list[int] = []
    for row in range(input_ids.shape[0]):
        candidates = torch.nonzero(valid_positions[row], as_tuple=False).flatten().tolist()
        if not candidates:
            raise ValueError(f"example {row} has no attended non-special token for an extraction span")
        contiguous_pair = next(
            ((left, right) for left, right in zip(candidates, candidates[1:]) if right == left + 1),
            None,
        )
        start, end = contiguous_pair if contiguous_pair is not None else (candidates[0], candidates[0])
        starts.append(start)
        ends.append(end)
    return torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long)


def _supervised_checks(model: ModernBertForMaskedLM, heads: TaskHeads,
                       optimizer: AdamW, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       special_token_ids: Sequence[int]) -> dict[str, Any]:
    model.train()
    batch_size = input_ids.shape[0]
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    pooled = masked_mean_pool(hidden, attention_mask)
    class_logits = heads.classification(pooled)
    class_loss = F.cross_entropy(class_logits, labels)

    token_logits = heads.extraction(hidden)
    start_positions, end_positions = _synthetic_gold_spans(
        input_ids, attention_mask, special_token_ids
    )
    extraction_loss = _span_extraction_loss(
        token_logits, start_positions, end_positions, input_ids,
        attention_mask, special_token_ids
    )
    perturbed_padding_logits = token_logits.detach().clone()
    perturbed_padding_logits[~attention_mask.bool()] = 10000.0
    perturbed_extraction_loss = _span_extraction_loss(
        perturbed_padding_logits, start_positions, end_positions, input_ids,
        attention_mask, special_token_ids
    )
    padding_logits_invariant = torch.equal(extraction_loss.detach(), perturbed_extraction_loss)
    if not padding_logits_invariant:
        raise AssertionError("span extraction loss changed after padding logits were perturbed")

    # Two independent dropout views of the same sentence, as in unsupervised
    # SimCSE-style contrastive adaptation. This is a shape/gradient check only.
    view_one = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    view_two = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    view_one = F.normalize(heads.embedding(heads.view_dropout(masked_mean_pool(view_one, attention_mask))), dim=-1)
    view_two = F.normalize(heads.embedding(heads.view_dropout(masked_mean_pool(view_two, attention_mask))), dim=-1)
    if torch.equal(view_one, view_two):
        raise AssertionError("independent dropout views unexpectedly match exactly")
    similarity = view_one @ view_two.transpose(0, 1) / 0.07
    retrieval_loss = F.cross_entropy(similarity, torch.arange(batch_size))

    pair_ids, pair_mask = _make_cross_encoder_pairs(input_ids)
    pair_hidden = model.model(input_ids=pair_ids, attention_mask=pair_mask).last_hidden_state
    pair_pool = masked_mean_pool(pair_hidden, pair_mask)
    rerank_scores = heads.reranking(pair_pool).squeeze(-1)
    rerank_labels = torch.tensor([1.0, 0.0, 1.0])
    reranking_loss = F.binary_cross_entropy_with_logits(rerank_scores, rerank_labels)

    losses = {
        "classification": class_loss,
        "span_extraction": extraction_loss,
        "simcse_retrieval": retrieval_loss,
        "cross_encoder_reranking": reranking_loss,
    }
    total_loss = sum(losses.values())
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    encoder_norm = _assert_finite_gradients([model.model.layers[0].attn.Wqkv.weight], "supervised task mix")
    head_names = ("classification", "extraction", "embedding", "reranking")
    head_gradient_norms = {}
    for name in head_names:
        head_gradient_norms[name] = _assert_finite_gradients(list(getattr(heads, name).parameters()), name)
    encoder_before = model.model.layers[0].attn.Wqkv.weight.detach().clone()
    optimizer.step()
    updated = not torch.equal(encoder_before, model.model.layers[0].attn.Wqkv.weight.detach())
    if not updated:
        raise AssertionError("combined supervised update did not change encoder weights")

    return {
        "losses": {name: float(loss.detach().item()) for name, loss in losses.items()},
        "total_loss": float(total_loss.detach().item()),
        "encoder_gradient_norm": encoder_norm,
        "head_gradient_norms": head_gradient_norms,
        "encoder_parameter_updated": updated,
        "output_shapes": {
            "classification_logits": list(class_logits.shape),
            "extraction_token_logits": list(token_logits.shape),
            "retrieval_similarity_matrix": list(similarity.shape),
            "reranking_scores": list(rerank_scores.shape),
        },
        "dropout_views_independent": True,
        "gold_spans_valid_non_special_and_attended": True,
        "synthetic_gold_start_positions": start_positions.tolist(),
        "synthetic_gold_end_positions": end_positions.tolist(),
        "span_loss_invariant_to_padding_logit_perturbation": padding_logits_invariant,
    }


def run_functional_checks() -> dict[str, Any]:
    torch.set_num_threads(THREAD_LIMIT)
    torch.manual_seed(20261016)
    config = make_tiny_config()
    model = ModernBertForMaskedLM(config).cpu()
    heads = TaskHeads(config.hidden_size).cpu()
    optimizer = AdamW(list(model.parameters()) + list(heads.parameters()), lr=1e-3)
    input_ids, attention_mask = make_synthetic_batch()

    token_input, token_labels, token_selected = dynamic_corruption(
        input_ids, attention_mask, mode="token", seed=101,
        special_token_ids=PINNED_SPECIAL_TOKEN_IDS, mask_token_id=PINNED_MASK_TOKEN_ID
    )
    token_out = model(input_ids=token_input, attention_mask=attention_mask, labels=token_labels)
    token_update = _train_one_step(model, optimizer, token_out.loss, "dynamic token MLM")

    span_input, span_labels, span_selected = dynamic_corruption(
        input_ids, attention_mask, mode="span", seed=202,
        special_token_ids=PINNED_SPECIAL_TOKEN_IDS, mask_token_id=PINNED_MASK_TOKEN_ID
    )
    span_out = model(input_ids=span_input, attention_mask=attention_mask, labels=span_labels)
    span_update = _train_one_step(model, optimizer, span_out.loss, "contiguous-span MLM")

    tiny_layout = attention_layout(model)
    expected_globals = list(range(0, config.num_hidden_layers, config.global_attn_every_n_layers))
    if tiny_layout["global_layer_indices_zero_based"] != expected_globals:
        raise AssertionError("the official ModernBERT module does not have the requested global-layer schedule")
    if tiny_layout["local_layer_count"] == 0 or tiny_layout["global_layer_count"] == 0:
        raise AssertionError("the tiny model must contain both local and global layers")

    pool_probe = torch.randn(2, 6, config.hidden_size)
    pool_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    first_pool = masked_mean_pool(pool_probe, pool_mask)
    changed_padding = pool_probe.clone()
    changed_padding[pool_mask == 0] = 10000
    second_pool = masked_mean_pool(changed_padding, pool_mask)
    padding_invariant = torch.allclose(first_pool, second_pool, atol=1e-6, rtol=0.0)
    if not padding_invariant:
        raise AssertionError("masked mean pooling changed when only padded states changed")

    model.eval()
    with torch.inference_mode():
        first_eval = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        second_eval = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    deterministic_eval = torch.equal(first_eval, second_eval)
    if not deterministic_eval:
        raise AssertionError("evaluation-mode output was not deterministic")

    special_probe_ids, special_probe_mask = make_special_token_probe(
        config.vocab_size, PINNED_SPECIAL_TOKEN_IDS
    )
    special_probe_results: dict[str, dict[str, int]] = {}
    injected_counts = {
        str(special_id): int(((special_probe_ids == special_id) & special_probe_mask.bool()).sum().item())
        for special_id in PINNED_SPECIAL_TOKEN_IDS
    }
    if any(count == 0 for count in injected_counts.values()):
        raise AssertionError("the probe did not inject every pinned special ID at an attended position")
    for mode, seed in (("token", 303), ("span", 404)):
        _, _, probe_selected = dynamic_corruption(
            special_probe_ids, special_probe_mask, mode=mode, seed=seed,
            special_token_ids=PINNED_SPECIAL_TOKEN_IDS, mask_token_id=PINNED_MASK_TOKEN_ID
        )
        selected_counts = {
            str(special_id): int(((special_probe_ids == special_id) & probe_selected).sum().item())
            for special_id in PINNED_SPECIAL_TOKEN_IDS
        }
        if any(count != 0 for count in selected_counts.values()):
            raise AssertionError(f"{mode} masking selected a tokenizer special ID: {selected_counts}")
        special_probe_results[mode] = selected_counts

    task_result = _supervised_checks(
        model, heads, optimizer, input_ids, attention_mask, PINNED_SPECIAL_TOKEN_IDS
    )
    return {
        "model": {
            "class": "transformers.ModernBertForMaskedLM",
            "parameters": _count_unique_parameters(model),
            "synthetic_vocab_size": config.vocab_size,
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "attention_heads": config.num_attention_heads,
            "dropout_for_contrastive_views": config.embedding_dropout,
            "device": "cpu",
        },
        "data": {
            "source": "synthetic random token IDs; no language corpus was loaded",
            "batch_shape": list(input_ids.shape),
            "padding_and_document_boundary_present": True,
            "mask_rate": 0.30,
            "selected_token_counts": token_selected.sum(dim=1).tolist(),
            "selected_span_counts": span_selected.sum(dim=1).tolist(),
            "special_token_ids": list(PINNED_SPECIAL_TOKEN_IDS),
            "special_ids_injected_at_attended_probe_positions": injected_counts,
            "special_ids_selected_for_token_mlm": special_probe_results["token"],
            "special_ids_selected_for_span_mlm": special_probe_results["span"],
            "special_and_padding_tokens_excluded": True,
        },
        "updates": {
            "dynamic_token_mlm": token_update,
            "contiguous_span_mlm": span_update,
        },
        "attention_layout": tiny_layout,
        "pooling_padding_invariant": padding_invariant,
        "evaluation_deterministic": deterministic_eval,
        "downstream_heads": task_result,
        "semantic_quality_evaluated": False,
    }


def run_8k_probe() -> dict[str, Any]:
    """One tiny CPU inference at 8K; this is a runtime check, not a speed claim."""
    torch.set_num_threads(THREAD_LIMIT)
    torch.manual_seed(20261017)
    config = make_tiny_config(layers=3, hidden=16, heads=1, intermediate=24,
                              local_window=128, global_every=3, dropout=0.0)
    model = ModernBertForMaskedLM(config).cpu().eval()
    ids = torch.randint(10, config.vocab_size, (1, 8192), dtype=torch.long)
    mask = torch.ones_like(ids)
    process_peak_mb_before: float | None = None
    process_peak_mb_after: float | None = None
    memory_source: str | None = None
    try:
        import psutil

        process = psutil.Process()
        before_info = process.memory_info()
        process_peak_mb_before = getattr(before_info, "peak_wset", before_info.rss) / (1024 * 1024)
        memory_source = "Windows process peak working set (whole verification process)"
    except Exception:
        process = None

    try:
        with torch.inference_mode():
            # Shape warm-up stays short; the 8K sample is measured once.
            warm_ids = ids[:, :128]
            warm_mask = mask[:, :128]
            _ = model.model(input_ids=warm_ids, attention_mask=warm_mask).last_hidden_state
            started = time.perf_counter()
            output = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            duration = time.perf_counter() - started
        if output.shape != (1, 8192, config.hidden_size):
            raise AssertionError(f"unexpected 8K output shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all().item():
            raise AssertionError("8K output contains non-finite values")
        if process is not None:
            after_info = process.memory_info()
            process_peak_mb_after = getattr(after_info, "peak_wset", after_info.rss) / (1024 * 1024)
        return {
            "status": "passed",
            "sequence_length": 8192,
            "model_parameters": _count_unique_parameters(model),
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "device": "cpu",
            "threads": THREAD_LIMIT,
            "attention_implementation": config._attn_implementation,
            "measured_single_forward_seconds": duration,
            "process_peak_working_set_mb_before": process_peak_mb_before,
            "process_peak_working_set_mb_after": process_peak_mb_after,
            "memory_measurement_scope": memory_source,
            "production_latency_claim": False,
        }
    except (RuntimeError, MemoryError) as exc:
        return {
            "status": "not_completed_resource_limit",
            "sequence_length": 8192,
            "device": "cpu",
            "threads": THREAD_LIMIT,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "production_latency_claim": False,
        }


def collect_evidence(run_long_probe: bool = True) -> dict[str, Any]:
    configure_cpu()
    parameter_counts = _parameter_inventory()
    scratch = scratch_blueprint_inventory()
    functional = run_functional_checks()
    long_probe = run_8k_probe() if run_long_probe else {"status": "not_requested"}
    return {
        "schema_version": 1,
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device_used": "cpu only",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_api_called": False,
        },
        "selected_pretrained_route": parameter_counts,
        "scratch_alternative": scratch,
        "functional_verification": functional,
        "tiny_8k_runtime_probe": long_probe,
        "interpretation": (
            "Configuration counts and synthetic task-head updates verify architecture plumbing only. "
            "No pretrained weights were downloaded and no semantic benchmark was run."
        ),
    }


def write_evidence(evidence: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "verification.json"
    json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        ("mmBERT-base pinned config", "token embedding", evidence["selected_pretrained_route"]["token_embedding_parameters"], "meta instantiation of official Transformers module; tied embedding counted once"),
        ("mmBERT-base pinned config", "transformer blocks", evidence["selected_pretrained_route"]["transformer_block_parameters"], "meta instantiation of official Transformers module"),
        ("mmBERT-base pinned config", "embedding norms and MLM head", evidence["selected_pretrained_route"]["embedding_norm_and_mlm_head_parameters"], "meta instantiation of official Transformers module"),
        ("mmBERT-base pinned config", "total unique trainable parameters", evidence["selected_pretrained_route"]["total_parameters"], "meta instantiation of official Transformers module; tied embedding counted once"),
        ("48K scratch alternative", "token embedding", evidence["scratch_alternative"]["tied_token_embedding_parameters"], "formula estimate; untrained design"),
        ("48K scratch alternative", "transformer blocks", evidence["scratch_alternative"]["transformer_block_parameters"], "formula estimate; untrained design"),
        ("48K scratch alternative", "MLM head and norms", evidence["scratch_alternative"]["mlm_head_and_norm_parameters"], "formula estimate; untrained design"),
        ("48K scratch alternative", "total estimated parameters", evidence["scratch_alternative"]["estimated_total_parameters"], "formula estimate; untrained design"),
    ]
    csv_lines = ["model,component,parameters,evidence_basis"]
    for model_name, component, count, basis in rows:
        csv_lines.append(f'"{model_name}","{component}",{count},"{basis}"')
    (output_directory / "parameter_counts.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
