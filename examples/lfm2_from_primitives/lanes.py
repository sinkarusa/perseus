"""Lane-batched linears for the multi-lane verify pass.

The verify pass holds its M tokens on lanes 0..M-1 of the token layout: feature i of token m at slot i t + m
(t = N / hid). For M | t, that ciphertext is the M-fold slot interleave of M virtual ciphertexts of N' = N / M slots:
real slot s is slot s // M of token s % M's virtual ciphertext, which holds the token in the decode layout of the
N'-slot ring (t' = t / M lanes; virtual lane 0 = real lane m, virtual lanes >= 1 = real lanes >= M).
- A rotation by k M is a rotation by k of every virtual ciphertext at once.
- Slot-wise products, additions and conjugation act on each virtual ciphertext.
So the cachemir decode linear of the N'-slot ring (layout.cm_params(N', d_in, d_out), linear.py unchanged) runs all M
tokens in one pass: its rotation steps times M, each plaintext the virtual one with every slot repeated M times.
- Plaintext products: d_in d_out / N' = M x the decode's, the same work as M per-lane linears.
- Key switches: ~2 sqrt(M r_i) BSGS steps + the replicate / lane-sum ladders, instead of M x the decode linear plus
  the unpack / pack lane moves.
- Levels: the linear's one product. No unpack / pack masks: rms_norm_lanes leaves lanes >= M at zero (the input
  ladder's precondition), and every consumer of an output masks its lanes >= M (the virtual layout's partial sums),
  as the decode's consumers mask its lanes >= 1.
The MLP width E needs N' >= E for one gate / up ciphertext; past that (M > N / E) gate / up / down split into
S = E / N' slices of width N' (gate / up columns, down rows), each slice its own N'-slot linear, whose outputs are
dense (every virtual slot live): S SiLUs, as the per-lane path's ceil(M / tp_E) MLP chunks.
"""
from __future__ import annotations

import dataclasses
import mmap

import numpy as np

from perseus.impl.layout import Dims, active_expanded_mask, cm_params, linear_rot_steps
from perseus.impl.linear import EncodedLinear, apply_linear, apply_linear_outputpack, prepare_linear_input


class LaneOps:
    """`ops` with every rotation step times M: the virtual N'-slot ring of an M-token ciphertext."""

    def __init__(self, ops, M: int):
        self._ops, self._M = ops, int(M)

    def rotate(self, a, k):
        return self._ops.rotate(a, int(k) * self._M)

    def rotate_many(self, a, steps):
        return self._ops.rotate_many(a, [int(k) * self._M for k in steps])

    def rotate_and_sum(self, a, start, stop):
        return self._ops.rotate_and_sum(a, int(start) * self._M, int(stop) * self._M)

    def __getattr__(self, name):
        return getattr(self._ops, name)


@dataclasses.dataclass(frozen=True)
class _VDims:
    N: int


class _VRt:
    """The two fields linear.py reads (`ops`, `dims.N`) for the virtual ring."""

    def __init__(self, rt, M: int):
        self.ops = LaneOps(rt.ops, M) if M > 1 else rt.ops
        self.dims = _VDims(rt.dims.N // M)


class Repeated:
    """Virtual plaintext rows (n, N') read as the real ones (n, N' M): every slot repeated M times, built at use (the
    slot cache keeps the virtual rows: the decode's storage per M). A memory-mapped cache file's pages are released
    after each read (madvise DONTNEED: they stay in the page cache, out of the process's resident set)."""

    def __init__(self, rows, M: int):
        self.rows, self.M = rows, int(M)

    @property
    def dtype(self):
        return self.rows.dtype

    @property
    def shape(self):
        return (self.rows.shape[0], self.rows.shape[1] * self.M)

    @property
    def ndim(self):
        return 2

    def __len__(self):
        return self.rows.shape[0]

    def _release(self):
        mm = getattr(self.rows, "_mmap", None)
        if mm is not None and hasattr(mm, "madvise"):
            mm.madvise(mmap.MADV_DONTNEED)

    def __getitem__(self, k):
        out = np.repeat(np.asarray(self.rows[k]), self.M, axis=-1)
        self._release()
        return out

    def __array__(self, dtype=None, copy=None):
        out = np.repeat(np.asarray(self.rows), self.M, axis=-1)
        self._release()
        return out if dtype is None else out.astype(dtype)


@dataclasses.dataclass
class LaneLinear:
    """One weight (d_in, d_out) as the N' = N / M slot decode linear, applied to M lanes at once."""
    M: int
    lin: EncodedLinear              # pts: Repeated (M > 1) or the decode's own slot vectors (M = 1)

    @property
    def N_virtual(self):
        return self.lin.pts.shape[1] // self.M


def _outputpack_ok(Nv, d_in, d_out):
    return cm_params(Nv, d_in, d_out).r_o % 2 == 0


def encode_lane_linear(W, N: int, M: int, outputpack: bool, cached=None) -> LaneLinear:
    """W (d_in, d_out) for M lanes of an N-slot ring. `outputpack` where the virtual shape allows it (an even r_o);
    `cached(build)` keeps the virtual rows (Lfm2Weights._cached)."""
    d_in, d_out = W().shape if callable(W) else W.shape
    Nv = N // M
    op = bool(outputpack) and _outputpack_ok(Nv, d_in, d_out)

    def build():
        return EncodedLinear.encode(W() if callable(W) else W, Nv, d_in, d_out, outputpack=op).pts
    rows = cached(build) if cached is not None else build()
    pts = Repeated(rows, M) if M > 1 else rows
    return LaneLinear(M, EncodedLinear(d_in, d_out, pts, None, None, op))


def lane_linear_multi(rt, x, lls):
    """One prepared input (M lanes), one output per LaneLinear (same M, d_in, d_out)."""
    M = lls[0].M
    vr = _VRt(rt, M)
    w0 = lls[0].lin
    prep = prepare_linear_input(vr, x, w0.d_in, w0.d_out)
    return [apply_linear_outputpack(vr, prep, ll.lin) if ll.lin.outputpack else apply_linear(vr, prep, ll.lin)
            for ll in lls]


def lane_linear(rt, x, ll: LaneLinear):
    return lane_linear_multi(rt, x, [ll])[0]


def virtual_dims(dims: Dims, M: int) -> Dims:
    return Dims(dims.N // M, dims.hid, dims.dim, dims.H, dims.H_real, dims.E, dims.E_real)


def mlp_slices(dims: Dims, M: int) -> int:
    """Gate / up / down slices of the MLP width for M lanes (1 while N / M >= E)."""
    Nv = dims.N // M
    return max(1, dims.E // Nv)


def mlp_paired(dims: Dims, M: int) -> bool:
    """Slices whose shape has no output pack (an odd r_o: N' = hid) run in pairs on the complex axis instead: gate /
    up as (W_s + i W_s+1) / 2 on the real input, unpacked by one conjugate; down as (W_s - i W_s+1) / 2 on
    h_s + i h_s+1, the real part (P + conj P) the sum of both slices. The plaintext products are then the per-lane
    path's (two real MACs per slot), not twice them."""
    S = mlp_slices(dims, M)
    w = dims.E // S
    return S >= 2 and not _outputpack_ok(dims.N // M, dims.hid, w)


def silu_mask(dims: Dims, M: int, scale: float) -> np.ndarray:
    """The SiLU output mask of a gate / up slice: the decode's active-expanded mask of the virtual ring, repeated M
    times (one slice); every slot (several: each slice's outputs are dense; padded MLP width unsupported there)."""
    S = mlp_slices(dims, M)
    if S == 1:
        return np.repeat(active_expanded_mask(virtual_dims(dims, M), scale), M)
    if dims.E_real != dims.E:
        raise NotImplementedError("lane-batched MLP slices with a padded MLP width")
    return np.full(dims.N, float(scale))


def lane_rot_steps(dims: Dims, M: int) -> set[int]:
    """Rotation steps (mod N) the lane-batched linears of M lanes issue: hid -> hid, gate / up, down (per slice)."""
    Nv = dims.N // M
    S = mlp_slices(dims, M)
    w = dims.E // S
    steps = set()
    for d_in, d_out in ((dims.hid, dims.hid), (dims.hid, w), (w, dims.hid)):
        steps |= {(k * M) % dims.N for k in linear_rot_steps(Nv, d_in, d_out)}
    return {s for s in steps if s}


@dataclasses.dataclass
class LaneBlock:
    """A block's LaneLinears for M lanes: the operator's (conv: bc, x, out; attention: kv, q, out) and the MLP's
    per slice (gate[s], up[s], down[s]), or per slice pair (`paired`: mlp_paired)."""
    M: int
    op: dict
    gate: list
    up: list
    down: list
    paired: bool = False


def _split(ops, P):
    """(P + conj P, i (conj P - P)): the real and imaginary parts of P times 2; one conjugate, no level."""
    cj = ops.conjugate(P)
    return ops.add(P, cj), ops.mult_i(ops.sub(cj, P))


def mlp_gate_up(rt, y, LB: LaneBlock):
    """[(gate_s, up_s)] of every slice of the M lanes y."""
    out = []
    for g, u in zip(LB.gate, LB.up):
        G, U = lane_linear_multi(rt, y, [g, u])
        if LB.paired:
            (g0, g1), (u0, u1) = _split(rt.ops, G), _split(rt.ops, U)
            out += [(g0, u0), (g1, u1)]
        else:
            out.append((G, U))
    return out


def mlp_down(rt, hs, LB: LaneBlock):
    """sum_s down_s(h_s): the MLP output on the M lanes (lanes >= M: partial sums)."""
    ops = rt.ops
    D = None
    for i, dn in enumerate(LB.down):
        if LB.paired:
            P = lane_linear(rt, ops.add(hs[2 * i], ops.mult_i(hs[2 * i + 1])), dn)
            o = ops.add(P, ops.conjugate(P))
        else:
            o = lane_linear(rt, hs[i], dn)
        D = o if D is None else ops.add(D, o)
    return D


def block_lanes(model, b: int, M: int) -> LaneBlock:
    """Block b's LaneLinears for M lanes from the model's weights (weights.Lfm2Weights): the decode's folded
    matrices (weights.conv_matrices / attn_matrices / mlp_matrices) encoded for the N / M slot ring; the virtual rows
    in the slot cache (`b<b>.lanes<M>.<name>`), coefficient rows attached when the weights are pinned (GPU). M = 1:
    the decode's own linears."""
    from .weights import attn_matrices, conv_matrices, mlp_matrices
    w, d = model.w, model.rt.dims
    if M == 1:
        bw = model.block_weights(b)
        names = ("kv", "q", "out") if model.attn[b] else ("bc", "x", "out")
        one = lambda lin: LaneLinear(1, lin)  # noqa: E731
        return LaneBlock(1, {n: one(getattr(bw, n)) for n in names}, [one(bw.gate)], [one(bw.up)], [one(bw.down)])
    memo = {}

    def mats(kind):
        if kind not in memo:
            memo[kind] = {"mlp": mlp_matrices, "conv": conv_matrices, "attn": attn_matrices}[kind](
                w.store, w.cfgs, b, w.dims)
        return memo[kind]

    def lin(name, W, outputpack):
        key = f"b{b}.lanes{M}.{name}"
        ll = encode_lane_linear(W, d.N, M, outputpack, cached=lambda build: w._cached(key, build))
        if getattr(w, "pinned", None) is not None:
            w.attach_coeffs(ll.lin, key)
        return ll
    if model.attn[b]:
        op = {"q": lin("q", lambda: mats("attn")[0], False), "kv": lin("kv", lambda: mats("attn")[1], False),
              "out": lin("out", lambda: mats("attn")[2], False)}
    else:
        op = {"bc": lin("bc", lambda: mats("conv")[0], False), "x": lin("x", lambda: mats("conv")[1], False),
              "out": lin("out", lambda: mats("conv")[3], False)}
    return LaneBlock(M, op, *_mlp_lanes(lambda: mats("mlp"), lin, d, M))


def _mlp_lanes(mats, lin, d: Dims, M: int):
    """(gate, up, down, paired) for M lanes; `mats()` = (Gate, Up, Down), `lin(name, W, outputpack)`."""
    S = mlp_slices(d, M)
    wd = d.E // S
    sl = lambda s: slice(s * wd, (s + 1) * wd)  # noqa: E731
    if mlp_paired(d, M):
        def cols(k, p):                                  # gate / up: (W_2p + i W_2p+1) / 2 on the columns
            W = mats()[k]
            return 0.5 * (W[:, sl(2 * p)] + 1j * W[:, sl(2 * p + 1)])

        def rows(p):                                     # down: (W_2p - i W_2p+1) / 2 on the rows
            W = mats()[2]
            return 0.5 * (W[sl(2 * p), :] - 1j * W[sl(2 * p + 1), :])
        n = S // 2
        gate = [lin(f"gate.p{p}", lambda p=p: cols(0, p), False) for p in range(n)]
        up = [lin(f"up.p{p}", lambda p=p: cols(1, p), False) for p in range(n)]
        down = [lin(f"down.p{p}", lambda p=p: rows(p), False) for p in range(n)]
        return gate, up, down, True
    suf = lambda s: f".s{s}" if S > 1 else ""  # noqa: E731
    gate = [lin("gate" + suf(s), lambda s=s: mats()[0][:, sl(s)], True) for s in range(S)]
    up = [lin("up" + suf(s), lambda s=s: mats()[1][:, sl(s)], True) for s in range(S)]
    down = [lin("down" + suf(s), lambda s=s: mats()[2][sl(s), :], True) for s in range(S)]
    return gate, up, down, False
