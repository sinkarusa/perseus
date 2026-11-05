"""LFM2 on ``perseus.impl``: the Llama port's model (session, stages, masks, plans, GQA sub-runtime) with the layer
schedule of config.layer_types: ShortConv layers (conv.ConvState decode state) and GQA attention layers with the
QK-norm (ComplexKVCache). The server stops after the final RMSNorm (decode_token(head=False) only): the client
decrypts rms_norm(x) * inv_out_scale and applies the final norm's gamma and the tied LM head itself.

Block numbering: 0..n_layers-1 the decoder layers, n_layers the tail (final RMSNorm)."""
from __future__ import annotations

import os

import numpy as np

from perseus.impl.attention import ComplexKVCache, complex_attention_step_masks, gqa_dims, rope_step_masks
from perseus.impl.blocked import SubRt
from perseus.impl.norm import rms_norm

from examples.llama_from_primitives.model import LlamaModel

from . import prefix as P
from .block import lfm2_block
from .conv import ConvState, state_level


def _bos_state_level(ops) -> int:
    """Level of the BOS prefix's public conv state encryptions (`start`): one unit richer than the pinned state level
    (conv.py) by default; LFM2_BOS_STATE_RICH=0: at the pin (captures: graph inputs at the pin). Measured on GPU:
    a fresh encryption dropped to the pin carries its encryption level's CKKS scale, a refreshed 2h dropped
    there the refresh landing's; the conv's adds at EQUAL level and degree do not reconcile the two, so the terms of a
    layer whose 2h is refreshed came out ~27% off at positions 1-2. One unit richer, every add with the state's terms
    meets a level difference and takes the runtime's scale-matching level adjust (as layer 0's unrefreshed 2h already
    does). The plan's level pins on the state's consumers are one-sided, so the richer state binds."""
    lv = state_level(ops)
    if os.environ.get("LFM2_BOS_STATE_RICH", "1") not in ("", "0"):
        lv -= ops.unit
    return lv


class Lfm2Model(LlamaModel):
    def __init__(self, inf, weights, cfgs, *, core=None, n_layers=None, unit=None, profile=False, kv_offload=False,
                 dims=None):
        super().__init__(inf, weights, cfgs, core=core, n_layers=n_layers, packing="cachemir_complex",
                         lmhead_cap=None, unit=unit, profile=profile, kv_offload=kv_offload, dims=dims)
        if not (getattr(self.rt.ops, "kv_lanes", False) and getattr(self.rt.ops, "qkv_lanes", False)):
            raise ValueError("the LFM2 port runs the QK-norm on the K / V / Q push payload: KV_LANES=1 QKV_LANES=1")
        # GQA always (32 / 8): the K / V caches hold each KV head once, on the sub-runtime
        self.G = self.H // self.H_kv
        self.sub = SubRt(self.rt, gqa_dims(self.rt.dims, self.G), f"gqa{self.G}")
        self.attn = [cfgs.is_attn(b) for b in range(self.n_layers)]
        self.kv = [ComplexKVCache(self.sub.dims.d_head) if a else ConvState() for a in self.attn]
        self.prefix_tokens = None

    # ── weights ──
    def block_weights(self, b):
        if b not in self._bw:
            w = self.w.block(b)
            for f in ("q", "kv", "out", "gate", "up", "down", "bc", "x"):
                lin = getattr(w, f, None)
                if lin is not None and hasattr(lin, "pts"):
                    self.w.attach_coeffs(lin, f"b{b}.{f}")
            self._bw[b] = w
        return self._bw[b]

    def preload(self):
        for b in range(self.n_layers):
            self.block_weights(b)

    # ── the plaintext prefix (BOS by default) ──
    def prefix_caches(self, tokens=None):
        """Per layer the plain slot vectors of the state after `tokens` (default [BOS])."""
        toks = tuple(int(t) for t in (tokens if tokens is not None else [int(self.w.store.config["bos_token_id"])]))
        if self._prefix is None or self._prefix[0] != toks:
            st = P.prefix_state(self.w.store, toks, self.n_layers)
            out = []
            for kind, a, b in st:
                if kind == "conv":
                    out.append(P.conv_state_vecs(self.rt.dims, a))
                else:
                    out.append(P.plain_attn_cache(self.sub.dims, a, b, self.H_kv, self.rope))
            self._prefix = (toks, out)
        return self._prefix[1]

    def start(self, prefix=None):
        """Reset every layer's state to the plaintext prefix (BOS; `prefix`: a public token list, test shortcut).
        Encrypted decode then starts at position len(prefix)."""
        enc = self._encrypt_public
        self._parked.clear()
        self.prefix_tokens = list(prefix) if prefix is not None else None
        for i, (st, pk) in enumerate(zip(self.kv, self.prefix_caches(prefix))):
            st.reset()
            if isinstance(st, ComplexKVCache):
                st.k_buckets = [enc(c) for c in pk.k_buckets]
                st.k_pend = [enc(c) for c in pk.k_pend]
                st.v_buckets = [None if c is None else enc(c) for c in pk.v_buckets]
                st.k_count, st.v_count = pk.k_count, pk.v_count
            else:
                ops = self.rt.ops
                st.h = [ops.level_drop(enc(v), _bos_state_level(ops)) for v in pk]
                st.count = len(prefix) if prefix is not None else 1
            if self.kv_offload:
                self.park(i)

    # ── the hooks ──
    def step_masks(self, pos):
        """The attention layers' per-step masks at cache count pos + 1 and RoPE's at `pos` (conv layers: none)."""
        art = self.sub
        d = art.dims
        items = []
        for b in range(self.n_layers):
            if self.attn[b]:
                items += complex_attention_step_masks(art, self.cfgs.softmax[b], pos + 1, pos % d.t)
        if any(self.attn):
            items += rope_step_masks(art, self.rope, pos, "cachemir_complex")
        items = [(art._key(k), b) for k, b in items]
        seen, out = set(), []
        for key, build in items:
            if key not in seen:
                seen.add(key); out.append((key, build))
        return out

    def stage_caches(self, i):
        if i < self.n_layers:
            return self.kv[i].cts(f"impl.b{i}.")
        return None

    def token_stages(self, x, pos, head=False):
        if head:
            raise ValueError("LFM2: the LM head is the client's (decode_token(head=False))")

        def block_fn(b):
            def compute(x):
                if self.taps is not None and b in self.tap_blocks:
                    self.rt.tap = lambda name, ct: self.taps.__setitem__((pos, f"{b}.{name}"), self._decode_any(ct))
                else:
                    self.rt.tap = None
                cap = self.enter_stage(b, pos)
                with self.rt.step(f"blk{b}"):
                    x = lfm2_block(self.rt, x, self.block_weights(b), self.kv[b], self.cfgs.block(b), pos,
                                   self.rope, (self.sub, self.G), last=b == self.n_layers - 1, tag=f"b{b}")
                self.exit_stage(b, cap)
                if self.taps is not None:
                    self.taps[(pos, b)] = self.hidden(x)
                return x
            return compute

        def tail_fn(x):
            b = self.n_layers
            cap = self.enter_stage(b, pos)
            with self.rt.step("tail"):
                h = rms_norm(self.rt, x, self.cfgs.ln_f, pos)
            self.exit_stage(b, cap)
            return h

        stages = [(f"blk{b}", None, block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", None, tail_fn))
        return stages

    def decode_token(self, x, pos, head=False):
        return self.run_stages(x, pos, self.token_stages(x, pos, head))

    def client_hidden(self, out):
        """The client's view of the server output: rms_norm(x) (gamma not applied), from rms_norm * inv_out_scale."""
        return self.hidden(out)[:self.rt.dims.dim] / self.cfgs.ln_f.inv_out_scale

    def run_decode(self, *a, **k):
        raise NotImplementedError("LFM2: drive decode_token per position")


def client_logits(hn, gamma_f, emb):
    """Client side of the cut after the final RMSNorm: hn = rms_norm(x) without gamma; returns (logits, hn gamma)."""
    h = np.asarray(hn) * gamma_f
    return h @ emb.T, h
