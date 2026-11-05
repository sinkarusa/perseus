"""One LFM2 decoder layer on perseus.impl:
operator_norm (RMSNorm, gamma in the next linears) -> ShortConv (conv.short_conv) or GQA attention with the per-head
QK-norm on the push payload (attention.mha_gqa(qk_norm=...), qknorm.qk_norm) -> residual -> ffn_norm -> SwiGLU
(the Llama port's gate/up -> THOR SiLU -> glu -> down) -> residual."""
from __future__ import annotations

from perseus.impl.activation import gelu
from perseus.impl.attention import mha_gqa
from perseus.impl.linear import linear, linear_multi
from perseus.impl.norm import rms_norm

from examples.llama_from_primitives.block import _bottom, _tap

from .conv import short_conv
from .qknorm import qk_norm


def lfm2_block(rt, x, w, cache, cfgs, pos: int, rope, gqa, last: bool = False, tag: str = ""):
    """``w``: weights.ConvWeights / AttnWeights; ``cache``: conv.ConvState or ComplexKVCache; ``cfgs``: the
    Lfm2Configs.block tuple (ln1, ln2, op, act), op None (conv) or (SoftmaxCfg, QKNormCfg); ``gqa``: (sub-runtime,
    G); ``pos``: the token's absolute position."""
    ops = rt.ops
    ln1, ln2, op, act = cfgs
    skip = x
    with rt.step("ln_1"):
        x = rms_norm(rt, x, ln1, pos)
    _tap(rt, "ln_1", x)
    if op is None:
        x = short_conv(rt, x, w, cache, tag)
    else:
        sm, qk = op
        sub, G = gqa
        qkn = lambda srt, X: qk_norm(srt, X, qk, w.q_norm, w.k_norm, G, tag)
        x = mha_gqa(rt, sub, G, x, w, cache, sm, rope=rope, pos=pos, pre_linear=_bottom(rt, "qkv"),
                    pre_out=_bottom(rt, "out_proj"), qk_norm=qkn)
    x = ops.add(x, skip)
    _tap(rt, "op_res", x)
    skip = x
    with rt.step("ln_2"):
        x = rms_norm(rt, x, ln2, pos)
    _tap(rt, "ln_2", x)
    with rt.step("gate_up"):
        ops.bootstrap_hint(x, ops.headroom(-1 if ops.qkv_lanes else 1), True)
        pre = _bottom(rt, "gate_up")
        if pre is not None:
            x = pre(x)
        g, u = linear_multi(rt, x, [w.gate, w.up])
    with rt.step("silu"):
        a = gelu(rt, g, act)
    with rt.step("glu"):
        h = ops.mult(a, u)
    with rt.step("down"):
        ops.bootstrap_hint(h, ops.headroom(1), True)
        pre = _bottom(rt, "down")
        if pre is not None:
            h = pre(h)
        x = linear(rt, h, w.down)
    _tap(rt, "mlp", x)
    out = ops.add(x, skip)
    if ops.qkv_lanes and not last:
        ops.fhe.level_hint(out, ops.headroom(3))
    return out
