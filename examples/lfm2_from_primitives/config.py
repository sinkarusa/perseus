"""configs.json of the LFM2 port (calibrate.py) -> perseus.impl config structs.

Sections: ``model`` (n_layers, layer_types, dims, rope_inv_freq, eps), ``norm`` (RMSNorm sites layers.<b>.ln1
(operator_norm) / ln2 (ffn_norm) / lnf, the Llama port's format), ``silu`` (THOR composite per layer),
``softmax`` (THOR softmax per ATTENTION layer), ``qknorm`` (per attention layer, QKNormCfg)."""
from __future__ import annotations

import dataclasses
import json

from perseus.impl.config import GeluCfg, NormCfg, SoftmaxCfg, parse_gelu, parse_softmax

from examples.llama_from_primitives.config import parse_rmsnorm


@dataclasses.dataclass(frozen=True)
class QKNormCfg:
    """One attention layer's q / k RMSNorm (one inverse-sqrt evaluation for all heads of q and k).
    Per head a centering scale (c_q: H values, c_k: H_kv values) maps z = c^2 (mean_head(x^2) + eps) into the
    shared Chebyshev band [cheb_lo, cheb_hi]; the Newton output is inv_out_scale / sqrt(z)."""
    eps: float
    c_q: tuple
    c_k: tuple
    cheb_coeffs: tuple
    cheb_lo: float
    cheb_hi: float
    cheb_nr_iters: int
    inv_out_scale: float

    @property
    def seed_mid(self) -> float:
        return 0.5 * (self.cheb_lo + self.cheb_hi)


def parse_qknorm(j: dict) -> QKNormCfg:
    return QKNormCfg(eps=float(j["eps"]), c_q=tuple(float(c) for c in j["c_q"]),
                     c_k=tuple(float(c) for c in j["c_k"]), cheb_coeffs=tuple(float(c) for c in j["cheb_coeffs"]),
                     cheb_lo=float(j["cheb_lo"]), cheb_hi=float(j["cheb_hi"]),
                     cheb_nr_iters=int(j["cheb_nr_iters"]), inv_out_scale=float(j["inv_out_scale"]))


@dataclasses.dataclass
class Lfm2Configs:
    model: dict
    norm: dict                  # site -> NormCfg
    act: list                   # per layer GeluCfg (the SiLU's THOR composite)
    softmax: dict               # attention layer -> SoftmaxCfg
    qknorm: dict                # attention layer -> QKNormCfg

    @property
    def n_layers(self):
        return int(self.model["n_layers"])

    @property
    def layer_types(self):
        return list(self.model["layer_types"])

    def is_attn(self, b: int) -> bool:
        return self.layer_types[b] == "full_attention"

    def block(self, b: int):
        """(ln1, ln2, op, act): op = (SoftmaxCfg, QKNormCfg) for an attention layer, None for a conv layer."""
        op = (self.softmax[b], self.qknorm[b]) if self.is_attn(b) else None
        return (self.norm[f"layers.{b}.ln1"], self.norm[f"layers.{b}.ln2"], op, self.act[b])

    @property
    def ln_f(self):
        return self.norm["lnf"]


def load_configs(path: str) -> Lfm2Configs:
    with open(path) as f:
        j = json.load(f)
    L = int(j["model"]["n_layers"])
    types = j["model"]["layer_types"]
    attn = [b for b in range(L) if types[b] == "full_attention"]
    return Lfm2Configs(model=j["model"],
                       norm={k: parse_rmsnorm(v) for k, v in j["norm"].items()},
                       act=[parse_gelu(j["silu"][f"layers.{b}.mlp.act"]) for b in range(L)],
                       softmax={b: parse_softmax(j["softmax"][f"layers.{b}.attn"]) for b in attn},
                       qknorm={b: parse_qknorm(j["qknorm"][f"layers.{b}.qk"]) for b in attn})
