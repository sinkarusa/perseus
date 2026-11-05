"""Per-head q / k RMSNorm (LFM2's q_layernorm / k_layernorm, RMSNorm over each 64-feature head, weight, eps)
before RoPE, on the GQA sub-runtime's refreshed K / V / Q payload (attention.cache_kv_push_packed_complex's `X`).

Payload X (sub layout: hid' = 512, t' = 64 lanes, H' = 8 KV heads, tH = 512): in-head feature j of KV head h' on
lane l sits at slot j tH + h' t' + l; lane 0 = K of KV head h', lane 1 = V, lane 2 + g = q of query head 4 h' + g
(mha_gqa's pre-rotated queries). One pass normalizes K and all G query lanes:

  xs  = X * c          c: per (head, lane) centering scale (c_k[h'] on lane 0, c_q[4h'+g] on lane 2+g, 0 elsewhere:
                       V and the idle lanes drop out)
  S   = sum_j xs^2     rotsum(., tH, N): the 6-rotation head ladder q.K^T already uses -> tH-periodic (512-slot
                       sparse route for its refreshes)
      + d_head eps c^2 on the live lanes, d_head * seed_mid elsewhere (keeps the idle lanes in the seed's band)
  inv = Chebyshev seed + Newton of inv_out_scale / sqrt(S / d_head)       (norm.rms_norm's machinery)
  Xn  = (X * c gamma / inv_out_scale) * inv = gamma * x / rms(x) on K and q lanes, 0 elsewhere

The numerator mask c gamma / inv_out_scale is a second plaintext product of X at the same level as xs (no extra
level). Levels: X -> xs (+1) -> square (+1) -> seed (Chebyshev depth) -> Newton (2 per iteration) -> final product
(+1). The scores' 1/sqrt(d_head) stays in q.K^T's group masks (layout.qkt_group_mask).

`roles`: the payload's lane map, ((lane, "k" | "q", g), ...): K lanes take c_k / the k gamma, query lanes of group g
c_q[G h' + g] / the q gamma, every other lane drops out. Default: the decode payload (decode_roles); the multi-lane
verify pass passes its own (K of M tokens, G M query lanes)."""
from __future__ import annotations

import numpy as np

from perseus.impl.poly import eval_chebyshev, inv_sqrt_newton_d2, rotsum

from .config import QKNormCfg


def decode_roles(G: int):
    """The decode push payload: K on lane 0, V on lane 1 (no role), query group g on lane 2 + g."""
    return ((0, "k", 0),) + tuple((2 + g, "q", g) for g in range(G))


def lane_scales(d, G: int, c_q, c_k, roles=None) -> np.ndarray:
    """(t', H') per (lane, KV head) value: c_k[h'] on K lanes, c_q[G h' + g] on group-g query lanes, 0 elsewhere."""
    out = np.zeros((d.t, d.H))
    cq = np.asarray(c_q, dtype=np.float64).reshape(d.H, G)
    for lane, kind, g in roles or decode_roles(G):
        out[lane, :] = c_k if kind == "k" else cq[:, g]
    return out


def slot_vec(d, per_lane_head: np.ndarray, per_feature=None) -> np.ndarray:
    """N-vector: slot j tH + h' t + l = per_lane_head[l, h'] (* per_feature[l][j] when given)."""
    base = per_lane_head.T.reshape(d.H * d.t)                        # index h' t + l
    v = np.tile(base, d.N // d.tH)                                    # j blocks of tH
    if per_feature is not None:
        v = v * np.repeat(per_feature, d.tH)
    return v


def masks(d, G: int, cfg: QKNormCfg, q_gamma, k_gamma, roles=None):
    """(c, numerator, eps) slot vectors of a layer's QK-norm."""
    roles = roles or decode_roles(G)
    c = lane_scales(d, G, cfg.c_q, cfg.c_k, roles)
    live = c != 0
    cvec = slot_vec(d, c)
    # numerator: c gamma / ios; gamma per in-head feature j, the k weight on K lanes, the q weight on the q lanes
    lane_of = (np.arange(d.N) % d.tH) % d.t
    k_lanes = [lane for lane, kind, _ in roles if kind == "k"]
    gam = np.where(np.isin(lane_of, k_lanes), np.repeat(k_gamma, d.tH), np.repeat(q_gamma, d.tH))
    num = cvec * gam / cfg.inv_out_scale
    dh = d.N // d.tH
    eps = np.where(live, dh * cfg.eps * c * c, dh * cfg.seed_mid)
    return cvec, num, slot_vec(d, eps)


def qk_norm(rt, X, cfg: QKNormCfg, q_gamma, k_gamma, G: int, tag: str, roles=None):
    """Xn = per-head RMSNorm(X) * gamma on the K lanes and the query lanes (module docstring); `rt` the sub-runtime.
    `roles` (default decode_roles(G)) also keys the masks."""
    ops, d = rt.ops, rt.dims
    dh = d.N // d.tH
    m = {}
    rk = () if roles is None else (tuple(roles),)

    def mk(i):
        if not m:
            m["v"] = masks(d, G, cfg, q_gamma, k_gamma, roles)
        return m["v"][i]
    xs = rt.mult_mask(X, ("qkn.c", tag) + rk, lambda: mk(0))
    xg = rt.mult_mask(X, ("qkn.num", tag) + rk, lambda: mk(1))
    S = ops.tag_reduce(rotsum(ops, ops.square(xs), d.tH, d.N), d.tH)   # tH-periodic: per (head, lane)
    S = rt.add_mask(S, ("qkn.eps", tag) + rk, lambda: mk(2))
    init = eval_chebyshev(ops, S, cfg.cheb_coeffs, dh * cfg.cheb_lo, dh * cfg.cheb_hi)
    inv = inv_sqrt_newton_d2(ops, S, init, cfg.cheb_nr_iters, 1.0 / (dh * cfg.inv_out_scale * cfg.inv_out_scale))
    ops.bootstrap_hint(inv, ops.headroom(4))
    return ops.mult(xg, inv)


def qk_norm_ref(x, gamma, c, cfg: QKNormCfg):
    """The same arithmetic on plaintext rows x (..., d_head) with per-row centering scale c (broadcastable to
    x[..., :1]): gamma * x / rms(x) up to the seed + Newton error (calibrate / mirror)."""
    from perseus.impl.ops import NumpyOps
    ops = NumpyOps()
    xs = c * x
    z = (xs * xs).mean(-1, keepdims=True) + cfg.eps * c * c
    init = eval_chebyshev(ops, z, cfg.cheb_coeffs, cfg.cheb_lo, cfg.cheb_hi)
    inv = inv_sqrt_newton_d2(ops, z, init, cfg.cheb_nr_iters, 1.0 / cfg.inv_out_scale ** 2)
    return xs * inv * gamma / cfg.inv_out_scale
