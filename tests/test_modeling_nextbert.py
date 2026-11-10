from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertForMaskedLM
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_nextbert import NextBert  # noqa: E402


def tiny_config(*, layers: int = 4) -> ModernBertConfig:
    return ModernBertConfig(
        vocab_size=64,
        hidden_size=48,
        num_hidden_layers=layers,
        intermediate_size=24,
        num_attention_heads=12,
        max_position_embeddings=64,
        local_attention=8,
        global_attn_every_n_layers=3,
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
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        attention_dropout=0.0,
    )


def dense_attention(q, k, v, key_mask, radius):
    scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    seq_len = q.shape[-2]
    qpos = torch.arange(seq_len, device=q.device)
    kpos = torch.arange(seq_len, device=q.device)
    allowed = key_mask[:, None, None, :].bool()
    if radius is not None:
        allowed = allowed & ((qpos[:, None] - kpos[None, :]).abs() <= radius)[None, None]
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(weights, v)


def rotate_qkv(attn, hidden, position_ids=None):
    batch, length, _ = hidden.shape
    qkv = attn.Wqkv(hidden).view(batch, length, 3, attn.num_heads, attn.head_dim)
    q, k, v = qkv.transpose(3, 1).unbind(dim=2)
    if position_ids is None:
        position_ids = torch.arange(length, device=hidden.device).unsqueeze(0)
    cos, sin = attn.rotary_emb(qkv, position_ids=position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def test_chunked_local_matches_dense_reference():
    torch.manual_seed(4)
    model = NextBert(
        None, "V1", config=tiny_config(layers=2), local_radius=2, query_chunk_size=3,
        special_token_ids=[0, 1, 2, 3, 4],
    ).eval()
    attn = model.model.layers[1].attn
    hidden = torch.randn(2, 9, 48)
    key_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0], [1] * 9], dtype=torch.bool)
    q, k, v = rotate_qkv(attn, hidden)
    expected = dense_attention(q, k, v, key_mask, radius=2)
    expected = expected.transpose(1, 2).reshape(2, 9, 48)
    expected = attn.Wo(expected)
    actual = attn(hidden, attention_mask=key_mask)[0]
    torch.testing.assert_close(actual[0, :6], expected[0, :6], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)


def test_global_sdpa_matches_dense_reference():
    torch.manual_seed(14)
    model = NextBert(None, "V0", config=tiny_config(layers=1)).eval()
    attn = model.model.layers[0].attn
    hidden = torch.randn(2, 7, 48)
    key_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1] * 7], dtype=torch.bool)
    q, k, v = rotate_qkv(attn, hidden)
    expected = dense_attention(q, k, v, key_mask, radius=None)
    expected = attn.Wo(expected.transpose(1, 2).reshape(2, 7, 48))
    actual = attn(hidden, attention_mask=key_mask)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_dilated_path_uses_original_rotary_positions_and_matches_dense_reference():
    torch.manual_seed(5)
    model = NextBert(
        None, "V2", config=tiny_config(layers=3), local_radius=2, query_chunk_size=3,
        special_token_ids=[0, 1, 2, 3, 4],
    ).eval()
    attn = model.model.layers[2].attn
    assert attn.kind == "dual" and attn.dilation == 2
    hidden = torch.randn(1, 11, 48)
    key_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    q, k, v = rotate_qkv(attn, hidden)
    actual = attn._dilated_path(q[:, 6:], k[:, 6:], v[:, 6:], key_mask)

    length, dilation, radius = hidden.shape[1], attn.dilation, attn.radius
    positions = torch.arange(length, device=hidden.device)
    distance = (positions[:, None] - positions[None, :]).abs()
    allowed = (distance <= radius * dilation) & (distance.remainder(dilation) == 0)
    allowed = allowed[None, None] & key_mask[:, None, None, :]
    scores = torch.matmul(q[:, 6:], k[:, 6:].transpose(-1, -2)) / (attn.head_dim ** 0.5)
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    expected = torch.matmul(F.softmax(scores.float(), dim=-1).to(q.dtype), v[:, 6:])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    compressed_positions = torch.div(positions, dilation, rounding_mode="floor").unsqueeze(0)
    compressed_q, compressed_k, compressed_v = rotate_qkv(attn, hidden, compressed_positions)
    wrong = attn._dilated_path(compressed_q[:, 6:], compressed_k[:, 6:], compressed_v[:, 6:], key_mask)
    assert not torch.allclose(actual, wrong, atol=1e-4, rtol=1e-4)


def test_padding_tokens_do_not_change_valid_representations():
    torch.manual_seed(6)
    model = NextBert(None, "V1", config=tiny_config(layers=3), local_radius=2, query_chunk_size=3).eval()
    ids = torch.tensor([[2, 11, 12, 13, 14, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    padded = model.encode(ids, mask)[:, :5]
    unpadded = model.encode(ids[:, :5], torch.ones((1, 5), dtype=torch.long))
    torch.testing.assert_close(padded, unpadded, atol=2e-6, rtol=2e-5)


def test_dual_gate_receives_gradient_and_is_optimizer_discoverable():
    torch.manual_seed(7)
    model = NextBert(None, "V2", config=tiny_config(layers=3), local_radius=2, query_chunk_size=3)
    model.set_gate_ramp(0.5)
    ids = torch.tensor([[2, 10, 11, 12, 13, 14, 1]])
    output = model.encode(ids).square().sum()
    output.backward()
    gates = model.gate_parameters()
    assert len(gates) == 1 and gates[0].numel() == 6
    assert gates[0].grad is not None
    assert gates[0].grad.abs().sum().item() > 0


def test_v0_backbone_keys_and_values_match_modernbert_mlm():
    torch.manual_seed(8)
    config = tiny_config()
    reference = ModernBertForMaskedLM(config).eval()
    model = NextBert(None, "V0", config=tiny_config(), local_radius=2, query_chunk_size=3)
    model.load_state_dict(reference.state_dict(), strict=False)
    model_keys = model.state_dict()
    for key, value in reference.state_dict().items():
        assert key in model_keys
        torch.testing.assert_close(model_keys[key], value)
    assert model.model.embeddings.tok_embeddings.weight is model.decoder.weight


def test_save_reload_preserves_state_and_task_interfaces(tmp_path):
    torch.manual_seed(9)
    model = NextBert(
        None, "V2", config=tiny_config(layers=3), local_radius=2, query_chunk_size=3,
        mlm_chunk_size=4, special_token_ids=[0, 1, 2, 3, 4],
    )
    model.save_pretrained(tmp_path)
    loaded = NextBert(tmp_path, "V2")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    assert loaded.local_radius == 2
    assert loaded.special_token_ids == (0, 1, 2, 3, 4)

    ids = torch.tensor([[2, 10, 11, 12, 1], [2, 20, 21, 22, 1]])
    mask = torch.ones_like(ids)
    assert loaded.forward_mlm(ids, mask, labels=torch.tensor([[-100, 10, -100, 12, -100], [-100, 20, -100, 22, -100]]))["loss"].ndim == 0
    assert loaded.forward_classification(ids, mask, labels=torch.tensor([0, 2]))["logits"].shape == (2, 3)
    context = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
    qa = loaded.forward_qa(
        ids, mask, start_positions=torch.tensor([2, 2]), end_positions=torch.tensor([3, 3]), context_mask=context
    )
    assert qa["start_logits"].shape == (2, 5) and qa["loss"].ndim == 0
    assert torch.all(qa["start_logits"][:, [0, 1, 4]] == torch.finfo(qa["start_logits"].dtype).min)
    vectors = loaded.encode_retrieval(ids, mask)
    assert vectors.shape == (2, 256)
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2))
    rerank = loaded.forward_reranking(ids[:1].repeat(3, 1), mask[:1].repeat(3, 1), labels=torch.tensor([1]), group_size=3)
    assert rerank["logits"].shape == (3,) and rerank["loss"].ndim == 0


def test_gate_ramp_validates_range():
    model = NextBert(None, "V2", config=tiny_config(layers=3))
    with pytest.raises(ValueError):
        model.set_gate_ramp(1.1)


def test_production_v2_has_308m_mlm_core_and_exact_gate_count():
    config = ModernBertConfig(
        vocab_size=256000,
        hidden_size=768,
        num_hidden_layers=22,
        intermediate_size=1152,
        num_attention_heads=12,
        max_position_embeddings=8192,
        local_attention=128,
        global_attn_every_n_layers=3,
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
    )
    with torch.device("meta"):
        model = NextBert(None, "V2", config=config)
    task_head_prefixes = ("classifier.", "qa_outputs.", "retrieval_projection.", "reranker.")
    mlm_core_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith(task_head_prefixes)
    )
    assert mlm_core_parameters == 307_786_282
    assert len(model.gate_parameters()) == 7
    assert sum(parameter.numel() for parameter in model.gate_parameters()) == 42
