"""Session setup of the LFM2 port: the Llama port's n32 decode env (examples/llama_from_primitives/env.py) with
GQA on (LLAMA_GQA=1: 32 query / 8 KV heads on the gqa4 sub-runtime) and the attention head block tH = t H = 16 x 32
= 512 slots (the sub-runtime's t' H' = 64 x 8 = 512 too): SPARSE_BTS_SLOTS=512,1.

Rotation keys: `rot_steps` = the Llama port's set for these dims (linears hid->hid / hid->E / E->hid, the attention
and norm ladders, RoPE's N/2, the push's -1..-(t-1), the GQA sub-geometry's steps, the radix groups) plus
`EXTRA` (none: the ShortConv issues no rotation outside its linears, and the QK-norm's head ladder rotsum(., tH, N)
is q.K^T's). The LM-head tile steps (hid -> N) are in the Llama set but unused here (client-side head).
`audit_steps` compares the steps a run actually issued (fakecap rotation log) with a key set."""
from __future__ import annotations

import os

from examples.llama_from_primitives import env as llama_env
from examples.llama_from_primitives.env import bootstrap_setup, plan_runtime  # noqa: F401

from perseus.impl.layout import Dims

ENV = {**llama_env.ENV, "LLAMA_GQA": "1", "SPARSE_BTS_SLOTS": "512,1"}
EXTRA: frozenset = frozenset()


def _gqa_on():
    os.environ["LLAMA_GQA"] = "1"


def rot_steps(dims: Dims) -> list[int]:
    _gqa_on()
    return sorted(set(llama_env.rot_steps(dims)) | set(EXTRA))


def export_env(device=0, chain="n32", **overrides):
    _gqa_on()
    llama_env.export_env(device, chain=chain, **{"LLAMA_GQA": "1", "SPARSE_BTS_SLOTS": "512,1", **overrides})


def open_session(dims: Dims, device=0, complex_payload=True, chain="n32", extra_rot_steps=(), **overrides):
    _gqa_on()
    over = {"LLAMA_GQA": "1", "SPARSE_BTS_SLOTS": "512,1", **overrides}
    return llama_env.open_session(dims, device, complex_payload=complex_payload, chain=chain,
                                  extra_rot_steps=tuple(set(extra_rot_steps) | set(EXTRA)), **over)


def audit_steps(issued, keys, N: int):
    """(missing, unused): signed steps issued without a key of the same signed value (the device keys -1 and N - 1
    apart), and keys never issued."""
    norm = lambda s: int(s) if int(s) < 0 else int(s) % N
    have = {norm(s) for s in keys}
    used = {norm(s) for s in issued if int(s) % N}
    return sorted(used - have), sorted(have - used)
