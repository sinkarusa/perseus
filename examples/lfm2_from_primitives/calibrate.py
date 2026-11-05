"""Calibrate the LFM2 port's approximation configs from collected site statistics (positions >= 1; position 0 is
the plaintext BOS prefix). The statistics come from a float64 teacher-forced pass over calibration windows
(saved as an .npz); the fits are the Llama port's (calibrate.py):

  norm     layers.<b>.ln1 (operator_norm), layers.<b>.ln2 (ffn_norm), lnf: fit_norm on per-token mean(x^2), max|x|
  silu     layers.<b>.mlp.act: fit_silu on max|gate| (THOR composite, Chebyshev basis)
  softmax  layers.<b>.attn (attention layers): fit_softmax_llama on sampled score rows
  qknorm   layers.<b>.qk (attention layers, new): per head a centering scale c = center_target / (max|x_head| margin)
           for the 32 q heads and the 8 k heads; ONE Chebyshev seed + Newton band [lo, hi] over every
           z = c^2 (mean_head(x^2) + eps) of q and k (qknorm.qk_norm evaluates them in one ciphertext), widened by
           norm_safety; inv_out_scale = inv_y_max sqrt(lo).

    python -m examples.lfm2_from_primitives.calibrate --sites sites.npz --config <model dir>/config.json --out c.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from examples.llama_from_primitives.calibrate import fit_silu, fit_softmax_llama, softmax_knobs


def fit_site(lo, hi, scale, target, max_deg=31, max_nr=8):
    """refit_norm_cheb.fit_site with up to `max_nr` Newton iterations (it stops at 4): LFM2's residual bands span
    up to ~7500x (massive activations on special tokens from layer 7 on), where no degree <= 31 seed is close
    enough for 4 iterations. The shallowest (depth, deg + 3 nr) chain meeting `target`."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "utils"))
    from refit_norm_cheb import depth
    from perseus.impl.ops import NumpyOps
    from perseus.impl.poly import eval_chebyshev, inv_sqrt_newton_d2
    ops = NumpyOps()
    t = np.cos(np.pi * (np.arange(400) + 0.5) / 400)
    xs = (t + 1) / 2 * (hi - lo) + lo
    f = scale / np.sqrt(xs)
    grid = np.geomspace(lo, hi, 4001)
    exact = scale / np.sqrt(grid)
    best = None
    for deg in range(2, max_deg + 1):
        w = 1.0 / f
        c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w)
        for _ in range(30):
            r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
            c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
        for nr in range(0, max_nr + 1):
            y = inv_sqrt_newton_d2(ops, grid, eval_chebyshev(ops, grid, c, lo, hi), nr, 1.0 / (scale * scale))
            err = float(np.max(np.abs(y / exact - 1)))
            if err <= target:
                key = (depth(deg, nr), deg + 3 * nr)
                if best is None or key < best[0]:
                    best = (key, deg, nr, c, err)
                break
    if best is None:
        raise RuntimeError(f"no chain up to degree {max_deg} / {max_nr} Newton meets {target} on [{lo:.3g}, {hi:.3g}]")
    return best


def fit_norm(ms, absmax, eps, a):
    """The Llama port's fit_norm (same fields) on fit_site above."""
    c = a.center_target / float(absmax.max() * a.norm_absmax_margin)
    z = c * c * (ms + eps)
    lo, hi = float(z.min()) / a.norm_safety, float(z.max()) * a.norm_safety
    ios = a.inv_y_max * math.sqrt(lo)
    (d, _), deg, nr, coef, err = fit_site(lo, hi, ios, a.norm_err)
    return {"method": "remez", "kind": "rmsnorm", "eps": eps, "z0": 0.0, "center_scale": c,
            "inv_out_scale": ios, "Ncoeffs": [], "Dcoeffs": [], "lin_alpha": 0.0, "lin_beta": 0.0,
            "gs_lo": 0.0, "gs_hi": 1.0, "gs_iters": 0, "nr_iters": 0,
            "cheb_coeffs": [float(v) for v in coef], "cheb_lo": lo, "cheb_hi": hi, "cheb_nr_iters": nr,
            "fit_err": err, "depth": d, "data_lo": float(z.min()), "data_hi": float(z.max())}


def fit_qknorm(q_ms, q_absmax, k_ms, k_absmax, eps, a):
    """q_ms / q_absmax: (n, H) per token and query head; k_*: (n, H_kv)."""
    c_q = a.center_target / (q_absmax.max(0) * a.norm_absmax_margin)
    c_k = a.center_target / (k_absmax.max(0) * a.norm_absmax_margin)
    z = np.concatenate([(c_q ** 2 * (q_ms + eps)).ravel(), (c_k ** 2 * (k_ms + eps)).ravel()])
    lo, hi = float(z.min()) / a.norm_safety, float(z.max()) * a.norm_safety
    ios = a.inv_y_max * math.sqrt(lo)
    (dep, _), deg, nr, coef, err = fit_site(lo, hi, ios, a.norm_err)
    return {"eps": eps, "c_q": c_q.tolist(), "c_k": c_k.tolist(), "inv_out_scale": ios,
            "cheb_coeffs": [float(v) for v in coef], "cheb_lo": lo, "cheb_hi": hi, "cheb_nr_iters": nr,
            "fit_err": err, "depth": dep, "data_lo": float(z.min()), "data_hi": float(z.max()),
            "q_absmax": q_absmax.max(0).tolist(), "k_absmax": k_absmax.max(0).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help=".npz of the collected statistics")
    ap.add_argument("--config", required=True, help="the checkpoint's config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--center-target", type=float, default=12.0)
    ap.add_argument("--norm-absmax-margin", type=float, default=1.25)
    ap.add_argument("--inv-y-max", type=float, default=3.0)
    ap.add_argument("--norm-safety", type=float, default=2.0)
    ap.add_argument("--norm-err", type=float, default=1e-5)
    ap.add_argument("--silu-safety", type=float, default=1.5)
    ap.add_argument("--silu-d1", type=int, default=31)
    ap.add_argument("--silu-d2", type=int, default=27)
    ap.add_argument("--sm-safety", type=float, default=2.0)
    ap.add_argument("--sm-err", type=float, default=1e-5)
    ap.add_argument("--sm-kc-max", type=int, default=256)
    ap.add_argument("--sm-margin", type=float, default=0.3)
    a = ap.parse_args(argv)
    st = np.load(a.sites)
    with open(a.config) as f:
        cfg = json.load(f)
    L = int(cfg["num_hidden_layers"])
    types = list(cfg["layer_types"])
    H = int(cfg["num_attention_heads"])
    dh = int(cfg["hidden_size"]) // H
    theta = float(cfg["rope_theta"])
    eps = float(cfg["norm_eps"])
    from .weights import ff_dim
    out = {"model": {"name": os.path.basename(os.path.dirname(os.path.abspath(a.config))), "n_layers": L,
                     "layer_types": types, "n_embd": int(cfg["hidden_size"]), "n_head": H,
                     "n_kv_head": int(cfg["num_key_value_heads"]), "n_inner": ff_dim(cfg),
                     "vocab": int(cfg["vocab_size"]),
                     "rope_inv_freq": (1.0 / theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh)).tolist(),
                     "rms_norm_eps": eps},
           "norm": {}, "silu": {}, "softmax": {}, "qknorm": {},
           "meta": {"windows": int(st["n_windows"]), "T": int(st["T"]), "args": vars(a)}}
    sites = [f"layers.{b}.{s}" for b in range(L) for s in ("ln1", "ln2")] + ["lnf"]
    for name in sites:
        n = out["norm"][name] = fit_norm(st[name + ".ms"], st[name + ".absmax"], eps, a)
        print(f"[norm] {name:14s} c={n['center_scale']:.3g} band [{n['cheb_lo']:.3g},{n['cheb_hi']:.3g}] "
              f"ios={n['inv_out_scale']:.3g} depth={n['depth']} nr={n['cheb_nr_iters']} err={n['fit_err']:.1e}",
              flush=True)
    from perseus.calibrate.thor_to_cheb import convert_site
    for b in range(L):
        s = fit_silu(float(st[f"layers.{b}.gate.absmax"].max()), a.silu_safety, a.silu_d1, a.silu_d2)
        convert_site(s, f"layers.{b}.mlp.act")
        out["silu"][f"layers.{b}.mlp.act"] = s
        print(f"[silu] {b:2d} xmax={s['xmax']:.3g} err max={s['fit_err_max']:.2e} med={s['fit_err_med']:.2e}",
              flush=True)
    knobs = softmax_knobs()
    knobs.range_margin = a.sm_margin
    knobs.decode_window = None
    T = int(st["T"])
    for b in range(L):
        if types[b] != "full_attention":
            continue
        q = out["qknorm"][f"layers.{b}.qk"] = fit_qknorm(st[f"layers.{b}.q.ms"], st[f"layers.{b}.q.absmax"],
                                                        st[f"layers.{b}.k.ms"], st[f"layers.{b}.k.absmax"], eps, a)
        print(f"[qknorm] {b:2d} c_q [{min(q['c_q']):.3g},{max(q['c_q']):.3g}] c_k [{min(q['c_k']):.3g},"
              f"{max(q['c_k']):.3g}] band [{q['cheb_lo']:.3g},{q['cheb_hi']:.3g}] ios={q['inv_out_scale']:.3g} "
              f"depth={q['depth']} nr={q['cheb_nr_iters']} err={q['fit_err']:.1e}", flush=True)
        sm = fit_softmax_llama(st[f"layers.{b}.scores"], st[f"layers.{b}.scores_q"], T, knobs, a.sm_safety,
                               a.sm_err, kc_max=a.sm_kc_max)
        out["softmax"][f"layers.{b}.attn"] = sm
        print(f"[softmax] {b:2d} clip [{sm['clip_lo']:.3g},{sm['clip_hi']:.3g}] d1={sm['delta1']} d2={sm['delta2']} "
              f"gs={sm['gs_iters_scaled']} refine={sm['per_step_refine_iters']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
