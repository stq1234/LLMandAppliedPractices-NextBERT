"""NextBERT variants built on the checkpoint-compatible ModernBERT blocks.

The pretrained parameter names for embeddings, transformer blocks and the MLM
head intentionally match ``ModernBertForMaskedLM``.  Only attention execution
is replaced; the QKV/output projections and pretrained rotary modules remain
the original modules.  Local attention is computed in bounded query chunks so
no sequence-length-squared local mask or score tensor is materialized.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers import ModernBertConfig, ModernBertForMaskedLM
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


_GLOBAL_LAYERS = {
    "V0": (0, 3, 6, 9, 12, 15, 18, 21),
    "V1": (0, 6, 12, 18, 21),
    "V2": (0, 6, 12, 18, 21),
    "V3": (0, 6, 12, 18, 21),
}
_DUAL_SCALE = {
    "V0": {},
    "V1": {},
    "V2": {2: 2, 4: 4, 8: 2, 10: 4, 14: 2, 16: 4, 20: 2},
    "V3": {2: 2, 4: 4, 8: 2, 10: 4, 14: 2, 16: 4, 20: 2},
}
_LOCAL_RADIUS = 64
_DUAL_HEADS = 6
_GATE_WARMUP_UPDATES = 64
_DEFAULT_MLM_CHUNK_SIZE = 256
_DEFAULT_QUERY_CHUNK_SIZE = 128

_ADAPTATION_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mmbert_adaptation.json"
try:
    _PINNED_SPECIAL_TOKEN_IDS = tuple(
        json.loads(_ADAPTATION_CONFIG.read_text(encoding="utf-8"))["vocabulary"]["special_token_ids"]["all_special_token_ids"]
    )
except (OSError, KeyError, json.JSONDecodeError):
    _PINNED_SPECIAL_TOKEN_IDS = ()


@lru_cache(maxsize=1)
def _try_flash_attention() -> Any | None:
    try:
        from flash_attn import flash_attn_func

        return flash_attn_func
    except Exception:
        return None


def _prefix_lengths(mask: torch.Tensor) -> torch.Tensor | None:
    """Return prefix lengths only when each row is valid tokens then padding."""
    if mask.ndim != 2:
        return None
    bool_mask = mask.bool()
    lengths = bool_mask.sum(dim=-1)
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    if torch.equal(bool_mask, positions < lengths[:, None]):
        return lengths
    return None


class NextBertAttention(nn.Module):
    """ModernBERT projections with global, local, or gated dilated attention."""

    def __init__(
        self,
        source: nn.Module,
        *,
        kind: str,
        radius: int,
        dilation: int | None = None,
        backend: str = "sdpa",
        query_chunk_size: int = _DEFAULT_QUERY_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        # Reuse the original module objects.  Their names and tensors therefore
        # keep the ModernBERT checkpoint key layout exactly.
        self.config = source.config
        self.layer_id = source.layer_id
        self.attention_dropout = source.attention_dropout
        self.deterministic_flash_attn = source.deterministic_flash_attn
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.all_head_size = source.all_head_size
        self.Wqkv = source.Wqkv
        self.Wo = source.Wo
        self.rotary_emb = source.rotary_emb
        self.out_drop = source.out_drop
        self.pruned_heads = source.pruned_heads

        if kind not in {"global", "local", "dual"}:
            raise ValueError(f"unknown attention kind: {kind}")
        if radius < 0 or (kind != "global" and radius < 1):
            raise ValueError("radius must be nonnegative; local attention requires radius >= 1")
        if kind == "dual" and (dilation is None or dilation < 2):
            raise ValueError("dual-scale attention requires dilation >= 2")
        self.kind = kind
        self.radius = radius
        self.dilation = dilation
        self.backend = backend
        self.query_chunk_size = query_chunk_size
        self.local_attention = (-1, -1) if kind == "global" else (radius, radius)
        if kind == "dual":
            # Name contains 'gate' for optimizer grouping and has exactly six
            # scalars per dual-scale layer.
            self.dual_gate_logits = nn.Parameter(torch.zeros(_DUAL_HEADS))
        else:
            self.register_parameter("dual_gate_logits", None)
        self.adaptation_updates = 0
        self.gate_ramp_override: float | None = None

    def set_adaptation_updates(self, updates: int) -> None:
        self.adaptation_updates = max(0, int(updates))
        self.gate_ramp_override = None

    def set_gate_ramp(self, ramp: float) -> None:
        if not 0.0 <= float(ramp) <= 1.0:
            raise ValueError("gate ramp must be between 0 and 1")
        self.gate_ramp_override = float(ramp)

    def _flash_path(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None,
        *,
        radius: int | None,
    ) -> torch.Tensor | None:
        flash_attn_func = _try_flash_attention()
        if key_mask is None:
            lengths = torch.full((q.shape[0],), q.shape[2], device=q.device, dtype=torch.long)
        else:
            lengths = _prefix_lengths(key_mask)
        if flash_attn_func is None or lengths is None:
            return None
        # FA2 expects [batch, sequence, heads, head_dim]. It handles the
        # window itself and skips right padding by evaluating each valid prefix.
        q_seq = q.transpose(1, 2).contiguous()
        k_seq = k.transpose(1, 2).contiguous()
        v_seq = v.transpose(1, 2).contiguous()
        outputs = []
        dropout_p = self.attention_dropout if self.training else 0.0
        window = (-1, -1) if radius is None else (radius, radius)
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length == 0:
                outputs.append(torch.zeros_like(q_seq[row]))
                continue
            valid_out = flash_attn_func(
                q_seq[row : row + 1, :length],
                k_seq[row : row + 1, :length],
                v_seq[row : row + 1, :length],
                dropout_p=dropout_p,
                window_size=window,
                causal=False,
            )[0]
            if length < q.shape[2]:
                valid_out = F.pad(valid_out, (0, 0, 0, 0, 0, q.shape[2] - length))
            outputs.append(valid_out)
        return torch.stack(outputs).transpose(1, 2).contiguous()

    def _sdpa_path(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None,
        *,
        radius: int | None,
    ) -> torch.Tensor:
        batch, _heads, seq_len, _dim = q.shape
        dropout_p = self.attention_dropout if self.training else 0.0
        if radius is None:
            # Global layers use one fused attention call. The no-padding case
            # takes the fastest SDPA/Flash kernel without allocating a mask.
            attn_mask = None
            if key_mask is not None:
                attn_mask = key_mask[:, None, None, :].bool()
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )

        # Fold all query chunks into the SDPA batch dimension. K/V windows are
        # bounded to query_chunk_size + 2 * radius, and the only attention mask
        # is [B, chunks, 1, query_chunk, window], never [B, L, L].
        chunk = self.query_chunk_size
        num_chunks = (seq_len + chunk - 1) // chunk
        padded_len = num_chunks * chunk
        right_pad = padded_len - seq_len + radius
        if key_mask is None:
            key_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=q.device)
        key_mask_padded = F.pad(key_mask.bool(), (radius, right_pad), value=False)
        key_mask_windows = key_mask_padded.unfold(1, chunk + 2 * radius, chunk)

        k_padded = F.pad(k, (0, 0, radius, right_pad))
        v_padded = F.pad(v, (0, 0, radius, right_pad))
        k_windows = k_padded.unfold(2, chunk + 2 * radius, chunk).permute(0, 2, 1, 4, 3)
        v_windows = v_padded.unfold(2, chunk + 2 * radius, chunk).permute(0, 2, 1, 4, 3)
        q_padded = F.pad(q, (0, 0, 0, padded_len - seq_len))
        q_blocks = q_padded.view(batch, q.shape[1], num_chunks, chunk, q.shape[-1]).permute(0, 2, 1, 3, 4)

        q_offsets = torch.arange(chunk, device=q.device)
        k_offsets = torch.arange(chunk + 2 * radius, device=q.device) - radius
        local_band = (q_offsets[:, None] - k_offsets[None, :]).abs() <= radius
        allowed = key_mask_windows[:, :, None, None, :] & local_band[None, None, None, :, :]
        query_windows = q_blocks.reshape(batch * num_chunks, q.shape[1], chunk, q.shape[-1])
        key_windows = k_windows.reshape(batch * num_chunks, k.shape[1], chunk + 2 * radius, k.shape[-1])
        value_windows = v_windows.reshape(batch * num_chunks, v.shape[1], chunk + 2 * radius, v.shape[-1])
        block_output = F.scaled_dot_product_attention(
            query_windows,
            key_windows,
            value_windows,
            attn_mask=allowed.reshape(batch * num_chunks, 1, chunk, chunk + 2 * radius),
            dropout_p=dropout_p,
            is_causal=False,
        )
        output = block_output.view(batch, num_chunks, q.shape[1], chunk, q.shape[-1])
        output = output.permute(0, 2, 1, 3, 4).reshape(batch, q.shape[1], padded_len, q.shape[-1])
        return output[:, :, :seq_len, :]

    def _path(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None,
        *,
        radius: int | None,
    ) -> torch.Tensor:
        if self.backend == "flash_attention_2":
            flash_out = self._flash_path(q, k, v, key_mask, radius=radius)
            if flash_out is not None:
                return flash_out
        return self._sdpa_path(q, k, v, key_mask, radius=radius)

    def _dilated_path(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.dilation is not None
        seq_len = q.shape[2]
        output = torch.zeros_like(q)
        # RoPE was already applied at each original absolute position before
        # this grouping. Positions in each residue class are exactly r apart.
        for residue in range(self.dilation):
            indices = torch.arange(residue, seq_len, self.dilation, device=q.device)
            if indices.numel() == 0:
                continue
            sub_q = q.index_select(2, indices)
            sub_k = k.index_select(2, indices)
            sub_v = v.index_select(2, indices)
            sub_mask = key_mask.index_select(1, indices) if key_mask is not None else None
            sub_out = self._path(sub_q, sub_k, sub_v, sub_mask, radius=self.radius)
            output = output.index_copy(2, indices, sub_out)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor]:
        if output_attentions:
            raise NotImplementedError("NextBERT chunked attention does not materialize attention matrices")
        batch, seq_len, _hidden = hidden_states.shape
        if attention_mask is None:
            key_mask = None
        else:
            key_mask = attention_mask
            if key_mask.ndim == 4:
                # Accept ModernBERT's additive mask for direct interoperability.
                key_mask = key_mask[:, 0, 0, :] == 0
            key_mask = key_mask.to(device=hidden_states.device, dtype=torch.bool)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        qkv = self.Wqkv(hidden_states).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.transpose(3, 1).unbind(dim=2)
        cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kind == "global":
            mixed = self._path(q, k, v, key_mask, radius=None)
        else:
            if key_mask is None:
                key_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=hidden_states.device)
            local = self._path(q, k, v, key_mask, radius=self.radius)
            if self.kind == "local":
                mixed = local
            else:
                dilated = self._dilated_path(
                    q[:, _DUAL_HEADS:, :, :],
                    k[:, _DUAL_HEADS:, :, :],
                    v[:, _DUAL_HEADS:, :, :],
                    key_mask,
                )
                warmup = (
                    self.gate_ramp_override
                    if self.gate_ramp_override is not None
                    else min(1.0, self.adaptation_updates / _GATE_WARMUP_UPDATES)
                )
                alpha = self.dual_gate_logits.sigmoid() * warmup
                gated_local = local[:, _DUAL_HEADS:, :, :]
                gate = alpha[None, :, None, None].to(dtype=local.dtype)
                mixed = torch.cat(
                    (local[:, :_DUAL_HEADS, :, :], (1.0 - gate) * gated_local + gate * dilated),
                    dim=1,
                )

        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, self.all_head_size)
        output = self.out_drop(self.Wo(mixed))
        return (output,)


class NextBert(nn.Module):
    """Pretrained-compatible NextBERT MLM encoder and downstream task heads.

    Args:
        model_dir: A local ModernBERT checkpoint directory or a directory saved
            by :meth:`save_pretrained`. Pass ``None`` together with ``config``
            to initialize a small random model for tests or controlled starts.
        variant: One of ``V0``, ``V1``, ``V2`` or ``V3``.
        attention_backend: ``auto`` selects FlashAttention2 when importable and
            otherwise the chunked SDPA implementation. ``sdpa`` forces SDPA.
    """

    def __init__(
        self,
        model_dir: str | Path | None,
        variant: str,
        attention_backend: str = "auto",
        *,
        config: ModernBertConfig | None = None,
        tokenizer: Any | None = None,
        special_token_ids: Sequence[int] | None = None,
        num_labels: int = 3,
        retrieval_dim: int = 256,
        local_radius: int = _LOCAL_RADIUS,
        query_chunk_size: int = _DEFAULT_QUERY_CHUNK_SIZE,
        mlm_chunk_size: int = _DEFAULT_MLM_CHUNK_SIZE,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        variant = variant.upper()
        if variant not in _GLOBAL_LAYERS:
            raise ValueError(f"variant must be one of {tuple(_GLOBAL_LAYERS)}, got {variant!r}")
        if attention_backend not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError("attention_backend must be 'auto', 'sdpa', or 'flash_attention_2'")
        flash_available = _try_flash_attention() is not None
        if attention_backend == "flash_attention_2" and not flash_available:
            raise ImportError("attention_backend='flash_attention_2' requires the flash-attn package")
        actual_backend = ("flash_attention_2" if flash_available else "sdpa") if attention_backend == "auto" else attention_backend
        if config is None:
            if model_dir is None:
                raise ValueError("config is required when model_dir is None")
            if (Path(model_dir) / "config.json").is_file():
                config = ModernBertConfig.from_pretrained(model_dir, local_files_only=True)
            else:
                # A Hub identifier is accepted for explicit callers, but all
                # project runners pass an already downloaded local path.
                config = ModernBertConfig.from_pretrained(model_dir, local_files_only=local_files_only)
        # Our attention implementation invokes the rotary module directly and
        # bypasses ModernBERT's default dense attention-mask construction.
        config._attn_implementation = "sdpa"
        config.reference_compile = False

        metadata_path = Path(model_dir) / "nextbert_meta.json" if model_dir is not None else None
        if metadata_path is not None and metadata_path.is_file():
            saved_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved_meta.get("variant") != variant:
                raise ValueError(
                    f"checkpoint variant {saved_meta.get('variant')!r} does not match requested {variant!r}"
                )
            num_labels = int(saved_meta.get("num_labels", num_labels))
            retrieval_dim = int(saved_meta.get("retrieval_dim", retrieval_dim))
            local_radius = int(saved_meta.get("local_radius", local_radius))
            query_chunk_size = int(saved_meta.get("query_chunk_size", query_chunk_size))
            mlm_chunk_size = int(saved_meta.get("mlm_chunk_size", mlm_chunk_size))
            if special_token_ids is None and tokenizer is None:
                special_token_ids = saved_meta.get("special_token_ids")

        custom_state = Path(model_dir) / "nextbert_model.bin" if model_dir is not None else None
        if custom_state is not None and custom_state.is_file():
            base = ModernBertForMaskedLM(config)
        elif model_dir is not None:
            base = ModernBertForMaskedLM.from_pretrained(
                model_dir,
                config=config,
                local_files_only=local_files_only,
                attn_implementation="sdpa",
            )
        else:
            base = ModernBertForMaskedLM(config)

        self.config = base.config
        self.model = base.model
        self.head = base.head
        self.decoder = base.decoder
        self.variant = variant
        self.attention_backend = actual_backend
        self.tokenizer = tokenizer
        self.local_radius = int(local_radius)
        self.query_chunk_size = int(query_chunk_size)
        self.mlm_chunk_size = int(mlm_chunk_size)
        self.adaptation_updates = 0
        self.model.gradient_checkpointing = False
        self._patch_attention_layout()

        if special_token_ids is not None:
            ids = special_token_ids
        elif tokenizer is not None and getattr(tokenizer, "all_special_ids", None) is not None:
            ids = tokenizer.all_special_ids
        else:
            ids = [
                getattr(self.config, name, None)
                for name in ("pad_token_id", "bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "unk_token_id", "mask_token_id")
            ]
            ids.extend(_PINNED_SPECIAL_TOKEN_IDS)
        self.special_token_ids = tuple(
            sorted({int(token_id) for token_id in ids if token_id is not None and 0 <= int(token_id) < self.config.vocab_size})
        )

        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.retrieval_projection = nn.Linear(hidden_size, retrieval_dim)
        self.reranker = nn.Linear(hidden_size, 1)
        self.task_dropout = nn.Dropout(float(getattr(self.config, "classifier_dropout", 0.0) or 0.0))
        self._init_task_heads()
        self.num_labels = num_labels
        self.retrieval_dim = retrieval_dim
        del base

        if custom_state is not None and custom_state.is_file():
            state = torch.load(custom_state, map_location="cpu", weights_only=True)
            incompatible = self.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
            missing_allowed = {"classifier.weight", "classifier.bias", "qa_outputs.weight", "qa_outputs.bias",
                               "retrieval_projection.weight", "retrieval_projection.bias", "reranker.weight", "reranker.bias"}
            unexpected_missing = [key for key in incompatible.missing_keys if key not in missing_allowed]
            if unexpected_missing:
                raise RuntimeError(f"checkpoint is missing model keys: {unexpected_missing}")

    def _patch_attention_layout(self) -> None:
        global_layers = set(_GLOBAL_LAYERS[self.variant])
        dilations = _DUAL_SCALE[self.variant]
        for index, layer in enumerate(self.model.layers):
            if index in global_layers:
                kind, dilation = "global", None
            elif index in dilations:
                kind, dilation = "dual", dilations[index]
            else:
                kind, dilation = "local", None
            layer.attn = NextBertAttention(
                layer.attn,
                kind=kind,
                radius=self.local_radius,
                dilation=dilation,
                backend=self.attention_backend,
                query_chunk_size=self.query_chunk_size,
            )

    def _init_task_heads(self) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        for module in (self.classifier, self.qa_outputs, self.retrieval_projection, self.reranker):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_adaptation_step(self, update: int) -> None:
        """Set the structural-adaptation update used for the 64-step gate ramp."""
        self.adaptation_updates = max(0, int(update))
        for layer in self.model.layers:
            layer.attn.set_adaptation_updates(self.adaptation_updates)

    def set_gate_ramp(self, ramp: float) -> None:
        """Set an explicit gate ramp, for calibrated runs and focused tests."""
        if not 0.0 <= float(ramp) <= 1.0:
            raise ValueError("gate ramp must be between 0 and 1")
        for layer in self.model.layers:
            layer.attn.set_gate_ramp(float(ramp))

    def gate_parameters(self) -> list[nn.Parameter]:
        """Return the learnable dual-scale gates for a separate optimizer group."""
        return [parameter for name, parameter in self.named_parameters() if "gate" in name]

    def gate_values(self) -> dict[str, torch.Tensor]:
        """Return the sigmoid-transformed gate values before the ramp."""
        return {
            f"model.layers.{index}.attn.dual_gate_logits": layer.attn.dual_gate_logits.sigmoid()
            for index, layer in enumerate(self.model.layers)
            if layer.attn.dual_gate_logits is not None
        }

    @property
    def gate_ramp(self) -> float:
        for layer in self.model.layers:
            attention = layer.attn
            if attention.dual_gate_logits is not None:
                if attention.gate_ramp_override is not None:
                    return attention.gate_ramp_override
                return min(1.0, attention.adaptation_updates / _GATE_WARMUP_UPDATES)
        return 0.0

    def effective_gate_values(self) -> dict[str, torch.Tensor]:
        """Return the actual mixing coefficients used by dual-scale attention."""
        ramp = self.gate_ramp
        return {name: values * ramp for name, values in self.gate_values().items()}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Mapping[str, Any] | None = None) -> None:
        kwargs = dict(gradient_checkpointing_kwargs or {})
        use_reentrant = bool(kwargs.get("use_reentrant", False))
        self.model.gradient_checkpointing = True
        self._gradient_checkpoint_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing = False

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, seq_len = input_ids.shape
        if seq_len > int(self.config.max_position_embeddings):
            raise ValueError(f"sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}")
        if attention_mask is None:
            attention_mask_for_layers = None
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same [batch, sequence] shape as input_ids")
        else:
            bool_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            # Evaluate the simple all-valid condition once per encoder call,
            # then let every global layer select its no-mask fused SDPA path.
            attention_mask_for_layers = None if bool(bool_mask.all()) else bool_mask
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)

        hidden = self.model.embeddings(input_ids=input_ids)
        for layer in self.model.layers:
            args = (hidden, attention_mask_for_layers, None, position_ids, None, None, False)
            if self.model.gradient_checkpointing and self.training:
                layer_outputs = checkpoint(
                    layer.__call__,
                    *args,
                    use_reentrant=getattr(self, "_gradient_checkpoint_use_reentrant", False),
                )
            else:
                layer_outputs = layer(
                    hidden,
                    attention_mask=attention_mask_for_layers,
                    sliding_window_mask=None,
                    position_ids=position_ids,
                    cu_seqlens=None,
                    max_seqlen=None,
                    output_attentions=False,
                )
            hidden = layer_outputs[0]
        return self.model.final_norm(hidden)

    def _mlm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(hidden))

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.encode(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if labels is None:
            if return_logits:
                raise ValueError("return_logits=True requires sparse labels to avoid materializing [B,L,V] logits")
            return {"loss": None, "logits": None}
        if labels.shape != input_ids.shape:
            raise ValueError("MLM labels must have the same [batch, sequence] shape as input_ids")
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        target_indices = torch.nonzero(flat_labels != -100, as_tuple=False).flatten()
        if target_indices.numel() == 0:
            return {"loss": hidden.sum() * 0.0, "logits": hidden.new_empty((0, self.config.vocab_size)) if return_logits else None}

        losses: list[torch.Tensor] = []
        logits_chunks: list[torch.Tensor] = []
        for start in range(0, target_indices.numel(), self.mlm_chunk_size):
            indices = target_indices[start : start + self.mlm_chunk_size]
            chunk_hidden = flat_hidden.index_select(0, indices)
            chunk_labels = flat_labels.index_select(0, indices)
            if return_logits:
                chunk_logits = self._mlm_logits(chunk_hidden)
                logits_chunks.append(chunk_logits)
                losses.append(F.cross_entropy(chunk_logits.float(), chunk_labels, reduction="sum"))
            else:
                def chunk_loss(states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    logits = self._mlm_logits(states)
                    return F.cross_entropy(logits.float(), targets, reduction="sum")

                losses.append(checkpoint(chunk_loss, chunk_hidden, chunk_labels, use_reentrant=False))
        loss = torch.stack(losses).sum() / target_indices.numel()
        logits = torch.cat(logits_chunks, dim=0) if return_logits else None
        return {"loss": loss, "logits": logits}

    def _content_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, content_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            valid = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if content_mask is not None:
            if content_mask.shape != input_ids.shape:
                raise ValueError("content_mask must have the same shape as input_ids")
            return valid & content_mask.to(device=input_ids.device, dtype=torch.bool)
        for special_id in self.special_token_ids:
            valid &= input_ids != special_id
        return valid

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward_classification(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.encode(input_ids, attention_mask)
        pooled = self._mean_pool(hidden, self._content_mask(input_ids, attention_mask, content_mask))
        logits = self.classifier(self.task_dropout(pooled))
        loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
        return {"loss": loss, "logits": logits}

    def forward_qa(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if context_mask is None:
            raise ValueError("context_mask is required so QA scores exclude the question, special tokens, and padding")
        hidden = self.encode(input_ids, attention_mask)
        logits = self.qa_outputs(hidden)
        start_logits, end_logits = logits.unbind(dim=-1)
        valid_context = self._content_mask(input_ids, attention_mask, context_mask)
        invalid = ~valid_context
        start_logits = start_logits.masked_fill(invalid, torch.finfo(start_logits.dtype).min)
        end_logits = end_logits.masked_fill(invalid, torch.finfo(end_logits.dtype).min)
        loss = None
        if (start_positions is None) != (end_positions is None):
            raise ValueError("start_positions and end_positions must be provided together")
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.reshape(-1).long()
            end_positions = end_positions.reshape(-1).long()
            loss = (F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)) / 2.0
        return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}

    def encode_retrieval(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        content_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask)
        pooled = self._mean_pool(hidden, self._content_mask(input_ids, attention_mask, content_mask))
        projected = self.retrieval_projection(self.task_dropout(pooled))
        return F.normalize(projected, p=2, dim=-1)

    def forward_reranking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        group_size: int | None = None,
        content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.encode(input_ids, attention_mask)
        pooled = self._mean_pool(hidden, self._content_mask(input_ids, attention_mask, content_mask))
        logits = self.reranker(self.task_dropout(pooled)).squeeze(-1)
        loss = None
        if labels is not None:
            if group_size is None or group_size < 2:
                raise ValueError("group_size >= 2 is required for groupwise reranking loss")
            if logits.numel() % group_size:
                raise ValueError("number of pairs must be divisible by group_size")
            grouped_logits = logits.view(-1, group_size)
            group_labels = labels.reshape(-1).long()
            if group_labels.numel() != grouped_logits.shape[0]:
                raise ValueError("labels must contain one positive-pair index per group")
            loss = F.cross_entropy(grouped_logits, group_labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "nextbert_model.bin")
        metadata = {
            "variant": self.variant,
            "num_labels": self.num_labels,
            "retrieval_dim": self.retrieval_dim,
            "attention_backend": self.attention_backend,
            "local_radius": self.local_radius,
            "query_chunk_size": self.query_chunk_size,
            "mlm_chunk_size": self.mlm_chunk_size,
            "special_token_ids": list(self.special_token_ids),
        }
        (path / "nextbert_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

