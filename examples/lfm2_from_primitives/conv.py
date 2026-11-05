"""LFM2's gated short convolution (Lfm2ShortConv) as one encrypted decode step.

    B, C, x = chunk(in_proj(n), 3);  h_t = B x;  out_t = k0 h_{t-2} + k1 h_{t-1} + k2 h_t (depthwise, per channel);
    y = out_proj(C out_t)

Token layout (the Llama port's: feature r at slot r t, lane 0; the linears leave partial sums in lanes 1..t-1):
  in_proj   one prepared input, two linears: the fused complex (B + iC) (weights.conv_matrices) and the real x.
            P = B + iC; conj(P) once: 2B = P + conj(P), 2C = i (conj(P) - P) (the monomial, no level).
  h         2h_t = (2B) * x, one ct x ct product, slot-aligned (channel r of B, x at the same slot).
  conv      out/2 = sum_j (k_j / 4 on lane 0) * 2h_{t-2+j}: three plaintext-mask products + adds, no rotation. The
            mask also zeroes lanes 1..t-1 (the linears' partial sums), so out and C out are lane-clean for out_proj.
  gate      y = (2C) * (out / 2) = C out, one ct x ct product.
  out_proj  hid -> hid linear.
Levels from the norm output L: L+1 (linears), L+2 (h), L+3 (conv), L+4 (C out), L+5 (out_proj).

State (ConvState): the two last 2h ciphertexts, oldest first; every step uses both (a zero encryption stands in for
h_{-1} at the BOS prefix), so the op sequence is the same at every position (one graph / plan per layer).
State level: 2h_t is refreshed if it sits above `state_level` (limit - 3 units; a hint, the plan re-decides it) and
dropped to it before it is used or stored; the prefix's public encryptions are dropped there too (model.start). A
graph input's level is the captured one in the bootstrap plan (planner seed_consumed) and a level_reduce is a pinned
point, so the state read at any position (BOS prefix, eager, planned) sits at the level the plan was cut for. The
conv taps land at the pin, C out one unit above and out_proj two."""
from __future__ import annotations

import numpy as np

from perseus.impl.layout import lane_vec
from perseus.impl.linear import linear, linear_multi


class ConvState:
    """One conv layer's decode state: [2h_{t-2}, 2h_{t-1}] (ciphertexts in token layout)."""

    def __init__(self):
        self.h = []
        self.count = 0          # tokens pushed (BOS included)

    def reset(self):
        self.h = []
        self.count = 0

    def push(self, h2):
        self.h = (self.h + [h2])[-2:]
        self.count += 1

    def cts(self, prefix: str):
        return list(self.h), [f"{prefix}conv.{i}" for i in range(len(self.h))]


def kernel_mask(d, k_j) -> np.ndarray:
    """Tap j's mask: k_j / 4 per channel on lane 0 (1/2 for the stored 2h, 1/2 for the 2C of the gate)."""
    return lane_vec(0.25 * np.asarray(k_j, dtype=np.float64), d.N, d.t)


def state_level(ops) -> int:
    """The level every stored / read 2h sits at (see the module doc)."""
    return ops.level_limit() - 3 * ops.unit


def short_conv(rt, x, w, st: ConvState, tag: str):
    """One decode step of the ShortConv on the operator_norm output x (gamma folded into the in_proj weights);
    pushes 2h_t into `st`. Returns out_proj(C out)."""
    ops, d = rt.ops, rt.dims
    if len(st.h) != 2:
        raise ValueError("short_conv: the state needs h_{t-2} and h_{t-1} (start the model first)")
    with rt.step("in_proj"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        P, X = linear_multi(rt, x, [w.bc, w.x])
    with rt.step("bx"):
        Pc = ops.conjugate(P)
        B2 = ops.add(P, Pc)                       # 2B
        C2 = ops.mult_i(ops.sub(Pc, P))           # 2C
        h2 = ops.mult(B2, X)                      # 2h_t
        lv = state_level(ops)
        ops.bootstrap_hint(h2, lv, True)          # eager: refresh if it would sit above the pin
        h2 = ops.level_drop(h2, lv)               # pinned (the planner keeps a level_reduce's level)
    _tap(rt, "conv.x", X)
    _tap(rt, "conv.b2", B2)
    _tap(rt, "conv.c2", C2)
    _tap(rt, "conv.h2", h2)
    with rt.step("conv"):
        cts = [st.h[0], st.h[1], h2]
        keys = [("conv.k", tag, j) for j in range(3)]
        builds = [lambda j=j: kernel_mask(d, w.kernel[j]) for j in range(3)]
        out = rt.mult_mask_sum(cts, keys, builds)  # out / 2, lane-clean
    st.push(h2)
    _tap(rt, "conv.out", out)
    with rt.step("gate"):
        y = ops.mult(C2, out)                     # C out
    _tap(rt, "conv.y", y)
    with rt.step("out_proj"):
        ops.bootstrap_hint(y, ops.headroom(1), True)
        return linear(rt, y, w.out)


def _tap(rt, name, ct):
    f = getattr(rt, "tap", None)
    if f is not None:
        f(name, ct)
