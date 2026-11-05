"""LFM2 bf16 safetensors -> the folded matrices of the cachemir packing and their slot vectors (EncodedLinear),
cached on disk as memory-mapped arrays (the Llama port's LlamaWeights cache, keyed by this module's builders).

Per layer (matrices (d_in, d_out), y = x @ W; g1 = operator_norm gamma / inv_out_scale(ln1), g2 = ffn_norm's):
  conv layer       in_proj (6144, 2048) = [B; C; x] rows. `bc` = the fused complex linear (B + iC) * g1 (one
                   prepared input gives 2B and 2C after one conjugation: conv.short_conv), `x` = x * g1; both keep
                   the natural channel order (B, C, x, the kernel and out_proj's rows only need to agree).
                   kernel (2048, 1, 3): kernel[:, j] (tap j multiplies h_{t-2+j}), plaintext vectors (ConvWeights.k).
                   out = out_proj.T.
  attention layer  q = q_proj.T g1, kv = (expand_kv(k_proj.T) + i expand_kv(v_proj.T)) g1 (8 KV heads expanded to
                   the 32 query heads, head-interleaved: the Llama port's GQA path reads KV head h' on sub-lane 0),
                   out = rearrange_wo(out_proj.T); q_layernorm / k_layernorm weights (64) are NOT folded here (the
                   QK-norm's numerator mask carries them, qknorm.py).
  MLP              gate = w1.T (g2 / xmax) (the SiLU's 1/xmax, GELU_FOLD), up = w3.T g2, down = w2.T (E = 8192 =
                   the real width: no padding).
The final norm and the tied head (embed_tokens) are the client's (the server never needs lm tiles)."""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np

from perseus.impl.layout import Dims, rearrange_qkv_weights, rearrange_wo_weights
from perseus.impl.linear import EncodedLinear

from examples.llama_from_primitives.weights import LlamaWeights, SafetensorsStore, expand_kv  # noqa: F401


def lfm2_dims(cfg: dict, N: int = 32768) -> Dims:
    """hid = d = 2048 (power of two), E = the auto-adjusted feed-forward width (8192 for LFM2.5-1.2B)."""
    d = int(cfg["hidden_size"])
    H = int(cfg["num_attention_heads"])
    E = ff_dim(cfg)
    hid = 1 << (d - 1).bit_length()
    return Dims(N, hid, d, H, H, 1 << (E - 1).bit_length(), E)


def ff_dim(cfg: dict) -> int:
    """Lfm2MLP's intermediate width: block_auto_adjust_ff_dim -> int(2 ff / 3) * multiplier, rounded up to
    block_multiple_of."""
    ff = int(cfg.get("block_ff_dim", cfg["intermediate_size"]))
    if cfg.get("block_auto_adjust_ff_dim", False):
        ff = int(2 * ff / 3)
        mult = cfg.get("block_ffn_dim_multiplier")
        if mult is not None:
            ff = int(mult * ff)
        m = int(cfg["block_multiple_of"])
        ff = m * ((ff + m - 1) // m)
    return ff


def pad(W, r, c):
    if W.shape == (r, c):
        return W
    out = np.zeros((r, c), dtype=W.dtype)
    out[:W.shape[0], :W.shape[1]] = W
    return out


@dataclasses.dataclass
class ConvWeights:
    """A conv layer: the fused (B + iC) and x linears, the kernel taps (3, d) and out_proj, plus the MLP."""
    bc: EncodedLinear
    x: EncodedLinear
    kernel: np.ndarray            # (3, d) float64: kernel[j] multiplies h_{t-2+j}
    out: EncodedLinear
    gate: EncodedLinear
    up: EncodedLinear
    down: EncodedLinear
    tag: str = ""


@dataclasses.dataclass
class AttnWeights:
    """An attention layer (the Llama BlockWeights fields mha_gqa reads) plus the q / k RMSNorm weights."""
    q: EncodedLinear
    kv: EncodedLinear
    out: EncodedLinear
    gate: EncodedLinear
    up: EncodedLinear
    down: EncodedLinear
    q_norm: np.ndarray            # (d_head,) float64
    k_norm: np.ndarray
    tag: str = ""
    k: EncodedLinear | None = None
    v: EncodedLinear | None = None


def mlp_matrices(store, cfgs, b: int, dims: Dims):
    p = f"model.layers.{b}."
    t = lambda n: store.tensor(p + n).astype(np.float64)
    _ln1, ln2, _op, act = cfgs.block(b)
    g2 = t("ffn_norm.weight") * ln2.descale
    Gate = pad(t("feed_forward.w1.weight").T * (g2 / act.xmax)[:, None], dims.hid, dims.E)
    Up = pad(t("feed_forward.w3.weight").T * g2[:, None], dims.hid, dims.E)
    Down = pad(t("feed_forward.w2.weight").T, dims.E, dims.hid)
    return Gate, Up, Down


def conv_matrices(store, cfgs, b: int, dims: Dims):
    """(BC complex (hid, hid), X (hid, hid), kernel (3, d), O (hid, hid))."""
    p = f"model.layers.{b}."
    t = lambda n: store.tensor(p + n).astype(np.float64)
    ln1 = cfgs.block(b)[0]
    d = dims.dim
    g1 = t("operator_norm.weight") * ln1.descale
    W = t("conv.in_proj.weight")                       # (3d, d)
    Bm, Cm, Xm = (W[i * d:(i + 1) * d].T * g1[:, None] for i in range(3))
    BC = pad(Bm, dims.hid, dims.hid) + 1j * pad(Cm, dims.hid, dims.hid)
    kernel = t("conv.conv.weight")[:, 0, :].T.copy()  # (3, d)
    O = pad(t("conv.out_proj.weight").T, dims.hid, dims.hid)
    return BC, pad(Xm, dims.hid, dims.hid), kernel, O


def attn_matrices(store, cfgs, b: int, dims: Dims):
    """(Q, KV complex, O), head-interleaved, K / V expanded to the query heads."""
    c = store.config
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    dh = dims.dim // H
    p = f"model.layers.{b}."
    t = lambda n: store.tensor(p + n).astype(np.float64)
    ln1 = cfgs.block(b)[0]
    g1 = t("operator_norm.weight") * ln1.descale
    Q = t("self_attn.q_proj.weight").T * g1[:, None]
    K = expand_kv(t("self_attn.k_proj.weight").T, H, H_kv, dh) * g1[:, None]
    V = expand_kv(t("self_attn.v_proj.weight").T, H, H_kv, dh) * g1[:, None]
    Q, K, V = (rearrange_qkv_weights(pad(M, dims.hid, dims.hid), dims.H) for M in (Q, K, V))
    O = rearrange_wo_weights(pad(t("self_attn.out_proj.weight").T, dims.hid, dims.hid), dims.H)
    return Q, K + 1j * V, O


def _folded_config(cfgs) -> str:
    norms = sorted((k, round(v.descale, 15)) for k, v in cfgs.norm.items())
    return repr((norms, [round(a.xmax, 15) for a in cfgs.act], cfgs.layer_types))


def _builder_sha():
    import inspect

    from perseus.impl import layout, linear
    fns = (mlp_matrices, conv_matrices, attn_matrices, expand_kv, pad, Lfm2Weights.block, LlamaWeights._linear,
           layout.cm_params, layout.interleave_idx, layout.weight_index, layout.encode_weight_matrix_fast,
           layout.rearrange_qkv_weights, layout.rearrange_wo_weights, linear.EncodedLinear.encode)
    return hashlib.sha256("".join(inspect.getsource(f) for f in fns).encode()).hexdigest()[:16]


class Lfm2Weights(LlamaWeights):
    """The slot vectors of every layer, built once per (checkpoint, folded configs, builder sources, dims) into the
    disk cache (PERSEUS_LFM2_CACHE or `cache_dir`) and memory-mapped from there (LlamaWeights._cached)."""

    def __init__(self, store: SafetensorsStore, cfgs, dims: Dims, packing: str = "cachemir_complex",
                 cache_dir: str | None = None, pinned=None):
        if packing != "cachemir_complex":
            raise ValueError("the LFM2 port runs the cachemir_complex packing only")
        self.store, self.cfgs, self.dims, self.packing = store, cfgs, dims, packing
        self.pinned = pinned
        root = cache_dir or os.environ.get("PERSEUS_LFM2_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "perseus", "lfm2_slots")
        key = hashlib.sha256(f"{store.fingerprint()}|{_folded_config(cfgs)}|{_builder_sha()}|{dims}|{packing}"
                             .encode()).hexdigest()[:20]
        self.dir = Path(root) / key
        self.dir.mkdir(parents=True, exist_ok=True)

    def block(self, b: int):
        d = self.dims
        hid, E = d.hid, d.E
        pre = f"b{b}."
        lin = lambda n, f, i, o, op=False: self._linear(pre + n, f, i, o, op)
        m = {}

        def mats(kind):
            if kind not in m:
                m[kind] = {"mlp": mlp_matrices, "conv": conv_matrices, "attn": attn_matrices}[kind](
                    self.store, self.cfgs, b, d)
            return m[kind]
        mlp = dict(gate=lin("gate", lambda: mats("mlp")[0], hid, E, True),
                   up=lin("up", lambda: mats("mlp")[1], hid, E, True),
                   down=lin("down", lambda: mats("mlp")[2], E, hid, True))
        if self.cfgs.is_attn(b):
            t = lambda n: self.store.tensor(f"model.layers.{b}.self_attn.{n}").astype(np.float64)
            return AttnWeights(q=lin("q", lambda: mats("attn")[0], hid, hid),
                               kv=lin("kv", lambda: mats("attn")[1], hid, hid),
                               out=lin("out", lambda: mats("attn")[2], hid, hid),
                               q_norm=t("q_layernorm.weight"), k_norm=t("k_layernorm.weight"), tag=pre, **mlp)
        kernel = self.store.tensor(f"model.layers.{b}.conv.conv.weight").astype(np.float64)[:, 0, :].T.copy()
        return ConvWeights(bc=lin("bc", lambda: mats("conv")[0], hid, hid),
                           x=lin("x", lambda: mats("conv")[1], hid, hid),
                           kernel=kernel,
                           out=lin("out", lambda: mats("conv")[3], hid, hid), tag=pre, **mlp)

    def linear_names(self, n_layers: int):
        out = []
        for b in range(n_layers):
            per = ("q", "kv", "out") if self.cfgs.is_attn(b) else ("bc", "x", "out")
            out += [f"b{b}.{n}" for n in per + ("gate", "up", "down")]
        return out

    def lm_tiles(self, paired: bool = True):
        raise NotImplementedError("LFM2: the tied LM head runs on the client (decode_token(head=False))")

    def embedding(self, token: int) -> np.ndarray:
        if not hasattr(self, "_emb"):
            self._emb = self.store.tensor("model.embed_tokens.weight")
        return np.asarray(self._emb[int(token)], dtype=np.float64)
