"""The plaintext prefix: the decode state after a public token prefix, computed in float64 and written into the
caches as encryptions of public vectors (the Llama port's BOS prefix, prefix.py, extended to LFM2's two state kinds).

Per layer: conv -> the last two h = B x rows (the ConvState stores 2h); attention -> every position's K after the
QK-norm (before RoPE: the push applies RoPE at the position) and V, pushed through the cachemir_complex push on
plain slot vectors (one push per position).

The default prefix is BOS alone (public). A longer prefix (`Lfm2Model.start(prefix=...)`) is a TEST shortcut: it
writes a plaintext-computed prompt state, standing in for an encrypted prefill (not part of this port)."""
from __future__ import annotations

import numpy as np

from perseus.impl.attention import ComplexKVCache, cache_kv_push_packed_complex
from perseus.impl.layout import encode_linear_input

from examples.llama_from_primitives.prefix import PlainOps, expanded_heads
from perseus.impl.rt import Rt


def _rms(x, g, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g


def prefix_state(store, tokens, n_layers: int | None = None):
    """[(kind, a, b)] per layer for the rows `tokens` at positions 0..P-1, float64: ("conv", h (P, d), None) or
    ("attn", k (P, H_kv, dh) after the QK-norm, before RoPE, v (P, H_kv, dh))."""
    c = store.config
    eps = float(c["norm_eps"])
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    d = int(c["hidden_size"])
    dh = d // H
    types = c["layer_types"]
    theta = float(c["rope_theta"])
    inv = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    t = lambda n: store.tensor(n).astype(np.float64)
    toks = np.asarray(tokens, dtype=np.int64)
    P = len(toks)
    x = store.tensor("model.embed_tokens.weight")[toks].astype(np.float64)
    pos = np.arange(P, dtype=np.float64)
    ang = pos[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, -1)[:, None, :]
    sin = np.concatenate([np.sin(ang)] * 2, -1)[:, None, :]
    rot = lambda z: np.concatenate([-z[..., dh // 2:], z[..., :dh // 2]], -1)
    out = []
    L = int(c["num_hidden_layers"]) if n_layers is None else int(n_layers)
    for b in range(L):
        p = f"model.layers.{b}."
        n = _rms(x, t(p + "operator_norm.weight"), eps)
        if types[b] == "conv":
            W = t(p + "conv.in_proj.weight")
            B, C, X = (n @ W[i * d:(i + 1) * d].T for i in range(3))
            h = B * X
            k3 = t(p + "conv.conv.weight")[:, 0, :]
            hh = np.concatenate([np.zeros((2, d)), h], 0)
            o = k3[:, 0] * hh[:-2] + k3[:, 1] * hh[1:-1] + k3[:, 2] * hh[2:]
            y = (C * o) @ t(p + "conv.out_proj.weight").T
            out.append(("conv", h, None))
        else:
            q = (n @ t(p + "self_attn.q_proj.weight").T).reshape(P, H, dh)
            k = (n @ t(p + "self_attn.k_proj.weight").T).reshape(P, H_kv, dh)
            v = (n @ t(p + "self_attn.v_proj.weight").T).reshape(P, H_kv, dh)
            q = _rms(q, t(p + "self_attn.q_layernorm.weight"), eps)
            k = _rms(k, t(p + "self_attn.k_layernorm.weight"), eps)
            out.append(("attn", k.copy(), v.copy()))
            qr, kr = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            G = H // H_kv
            Ke, Ve = np.repeat(kr, G, axis=1), np.repeat(v, G, axis=1)
            s = np.einsum("thd,shd->hts", qr, Ke) / np.sqrt(dh)
            s = np.where(np.tril(np.ones((P, P), dtype=bool))[None], s, -np.inf)
            pr = np.exp(s - s.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
            y = np.einsum("hts,shd->thd", pr, Ve).reshape(P, d) @ t(p + "self_attn.out_proj.weight").T
        x = x + y
        n = _rms(x, t(p + "ffn_norm.weight"), eps)
        g = n @ t(p + "feed_forward.w1.weight").T
        x = x + (g / (1.0 + np.exp(-g)) * (n @ t(p + "feed_forward.w3.weight").T)) @ t(p + "feed_forward.w2.weight").T
    return out


def plain_attn_cache(sub_dims, k_rows, v_rows, H_kv: int, rope) -> ComplexKVCache:
    """The ComplexKVCache P pushes of (k, v) at positions 0..P-1 leave, on plain slot vectors (sub-runtime dims:
    the KV heads head-interleaved, prefix.expanded_heads without expansion)."""
    rt = Rt(PlainOps(), sub_dims)
    N, hid = sub_dims.N, sub_dims.hid
    lin = lambda f: encode_linear_input(f, N, hid, hid)
    kv = ComplexKVCache(sub_dims.d_head)
    dh = k_rows.shape[-1]
    for p in range(k_rows.shape[0]):
        kf = expanded_heads(k_rows[p].ravel(), H_kv, H_kv, dh)
        vf = expanded_heads(v_rows[p].ravel(), H_kv, H_kv, dh)
        cache_kv_push_packed_complex(rt, kv, lin(kf) + 1j * lin(vf), rope=rope, pos=p)
    return kv


def conv_state_vecs(dims, h_rows) -> list:
    """The ConvState slot vectors [2h_{P-2}, 2h_{P-1}] (h_{-1} = 0) in token layout (lane 0)."""
    from perseus.impl.layout import lane_vec
    d = h_rows.shape[-1]
    rows = [np.zeros(d)] * max(0, 2 - len(h_rows)) + [r for r in h_rows[-2:]]
    return [lane_vec(np.pad(2.0 * r, (0, dims.hid - d)), dims.N, dims.t) for r in rows]
