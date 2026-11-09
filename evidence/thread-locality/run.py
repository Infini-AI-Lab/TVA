"""TVA library evidence (library issue L149, rounds 33 and 34): what Triton 3.8.0's OptimizeThreadLocality pass rewrites on
sm_90, on the IR it sees, and what the rewritten loop computes.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/thread-locality/run.py
(Part W runs its loop_unroll_factor forms each in a child process of this script, so that a crash of the compiler is
recorded instead of ending the run.)
Output: evidence/thread-locality/output.txt (README.md says what it shows).

The pass (primitives/triton-3.8.0-OptimizeThreadLocality.cpp:252-372) matches a tt.reduce whose combine region is one
float addf, mulf, maximumf, maxnumf, minimumf or minnumf; whose operand is a tensor of rank >= 2 in a blocked layout
defined by a tt.load, reduced along its last axis with more than one element per thread there; whose result has one
use, an operation with one use, the yield of a scf.for; whose loop-carried value starts from an arith.constant. It
reduces each thread's elements in the loop, folds those partial values across the trips with a clone of the update
operation from the combine's neutral element, reduces once after the loop, and applies the update operation once more
to the initial constant (lines 393-520). Every value below is exact in float32 where the source's trips are computed
in any association, so a stored value other than the source's is not a rounding.

Each line: whether the TTGIR holds the rewrite (a tt.reshape to one more dimension), the values stored (one per row,
all rows alike), the source's value, and, where it does not depend on which thread holds which element, the value of
U(c, R(v)) (v the per-thread partials folded by U from the combine's neutral element).
"""
import re
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEV = "cuda"
M, N, T = 64, 64, 4
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


def rewritten(h):
    g = h.asm["ttgir"]
    return bool(re.search(r"tt\.reshape [^\n]*-> tensor<\d+x\d+x\d+x", g))


@triton.jit
def k_upd(x_ptr, o_ptr, UPD: tl.constexpr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    if UPD == 4 or UPD == 9:
        acc = tl.full([M], 1.0, tl.float32)
    else:
        acc = tl.zeros([M], dtype=tl.float32)
    for t in range(T):
        x = tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :])
        if UPD == 0:
            acc = acc - tl.sum(x, axis=1)
        elif UPD == 1:
            acc = tl.sum(x, axis=1) - acc
        elif UPD == 2:
            acc = acc + tl.max(x, axis=1)
        elif UPD == 3:
            acc = tl.maximum(acc, tl.sum(x, axis=1))
        elif UPD == 4:
            acc = acc * tl.sum(x, axis=1)
        elif UPD == 5:
            acc = acc + tl.min(x, axis=1)
        elif UPD == 6:
            acc = tl.minimum(acc, tl.sum(x, axis=1))
        elif UPD == 7:
            acc = tl.maximum(acc, tl.max(x, axis=1))
        elif UPD == 8:
            acc = acc / tl.sum(x, axis=1)
        elif UPD == 9:
            acc = acc * tl.max(x, axis=1)
        else:
            acc = acc + tl.sum(x, axis=1)
    tl.store(o_ptr + rm, acc)


@triton.jit
def helper_load(p):
    return tl.load(p)


@triton.jit
def k_form(x_ptr, h_ptr, o_ptr, desc, nrt, FORM: tl.constexpr, SCALE: tl.constexpr, T: tl.constexpr,
           M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    if FORM == 105:
        acc = tl.load(o_ptr + rm) * 0.0
    else:
        acc = tl.zeros([M], dtype=tl.float32)
    t = 0
    for t in range(T):
        offs = t * M * N + rm[:, None] * N + rn[None, :]
        if FORM == 0:
            acc = acc + tl.sum(tl.load(x_ptr + offs), axis=1)
        elif FORM == 1:
            acc = acc + tl.sum(tl.load(x_ptr + offs).to(tl.float32), axis=1)
        elif FORM == 2:
            acc = acc + tl.sum(tl.load(x_ptr + offs) * 1.0, axis=1)
        elif FORM == 3:
            acc = acc + tl.sum(tl.load(x_ptr + offs) * SCALE, axis=1)
        elif FORM == 4:
            acc = acc + tl.sum(tl.load(x_ptr + offs) / 1.0, axis=1)
        elif FORM == 5:
            acc = acc + tl.sum(tl.load(x_ptr + offs) - 0.0, axis=1)
        elif FORM == 6:
            acc = acc + tl.sum(tl.load(x_ptr + offs) + (-0.0), axis=1)
        elif FORM == 7:
            bp = tl.make_block_ptr(x_ptr, (T * M, N), (N, 1), (t * M, 0), (M, N), (1, 0))
            acc = acc + tl.sum(tl.load(bp), axis=1)
        elif FORM == 8:
            acc = acc + tl.sum(helper_load(x_ptr + offs), axis=1)
        elif FORM == 9:
            x = tl.load(x_ptr + offs)
            acc = acc + tl.sum(tl.maximum(x, x), axis=1)
        elif FORM == 10:
            acc = acc + tl.sum(tl.trans(tl.trans(tl.load(x_ptr + offs))), axis=1)
        elif FORM == 11:
            acc = acc + tl.sum(tl.reshape(tl.load(x_ptr + offs), (M, N)), axis=1)
        elif FORM == 12:
            acc = acc + tl.sum(tl.where(SCALE == 1.0, tl.load(x_ptr + offs), 0.0), axis=1)
        elif FORM == 13:
            acc = acc + tl.sum(tl.load(x_ptr + offs, mask=rn[None, :] < N, other=0.0), axis=1)
        elif FORM == 14:
            acc = acc + tl.sum(tl.load(x_ptr + offs, cache_modifier=".cg"), axis=1)
        elif FORM == 15:
            acc = acc + tl.sum(tl.load(x_ptr + offs, mask=rn[None, :] < nrt, other=0.0), axis=1)
        # controls (not rewritten)
        elif FORM == 100:
            acc = acc + tl.sum(tl.load(x_ptr + offs) + 0.0, axis=1)
        elif FORM == 101:
            acc = acc + tl.sum(tl.load(x_ptr + offs) * 2.0, axis=1) * 0.5
        elif FORM == 102:
            acc = acc + tl.sum(tl.load(h_ptr + offs), axis=1)
        elif FORM == 103:
            acc = acc + tl.sum(desc.load([t * M, 0]), axis=1)
        elif FORM == 104:
            x = tl.load(x_ptr + offs)
            if tl.load(x_ptr) == tl.load(x_ptr):
                acc = acc + tl.sum(x, axis=1)
        elif FORM == 105:
            acc = acc + tl.sum(tl.load(x_ptr + offs), axis=1)
        elif FORM == 106:
            x = tl.load(x_ptr + offs)
            acc = acc + tl.sum(tl.where(x == x, x, 0.0), axis=1)
        elif FORM == 107:
            x = tl.load(x_ptr + offs)
            s = tl.sum(x, axis=1)
            acc = acc + s + s * 0.0
        elif FORM == 108:
            x = tl.load(x_ptr + offs)
            acc = acc + tl.sum(x * tl.load(h_ptr + offs), axis=1)
    tl.store(o_ptr + rm, acc)


@triton.jit
def k_range(x_ptr, o_ptr, KIND: tl.constexpr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros([M], dtype=tl.float32)
    if KIND == 0:
        for t in tl.range(0, T, num_stages=3):
            acc = acc + tl.sum(tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :]), axis=1)
    elif KIND == 1:
        for t in tl.static_range(T):
            acc = acc + tl.sum(tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :]), axis=1)
    else:
        t = 0
        while t < T:
            acc = acc + tl.sum(tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :]), axis=1)
            t += 1
    tl.store(o_ptr + rm, acc)


@triton.jit
def k_axis0(x_ptr, o_ptr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros([N], dtype=tl.float32)
    for t in range(T):
        acc = acc + tl.sum(tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :]), axis=0)
    tl.store(o_ptr + rn, acc)


@triton.jit
def k_1d(x_ptr, o_ptr, T: tl.constexpr, N: tl.constexpr):
    rn = tl.arange(0, N)
    acc = 0.0
    for t in range(T):
        acc = acc + tl.sum(tl.load(x_ptr + t * N + rn), axis=0)
    tl.store(o_ptr + rn, tl.zeros([N], dtype=tl.float32) + acc)


def stored(o):
    vals = o.cpu().tolist()
    return sorted(set(v for v in vals if v == v)) + (["nan"] if any(v != v for v in vals) else [])


def part_u():
    out("== Part U: the update operation (a plain tl.load operand; x is 1 everywhere but 2 in column 1, so each "
        "trip's row sum is 65, max 2, min 1)")
    x = torch.ones(T, M, N, dtype=torch.float32, device=DEV)
    x[:, :, 1] = 2.0
    s = float(N + 1)
    # the source's value, and U(c, R(v)) where it does not depend on the layout
    cases = {
        0: ("acc = acc - tl.sum(x, 1)", -T * s, T * s),           # v <- v - p from 0: R(v) = -260; 0 - (-260)
        1: ("acc = tl.sum(x, 1) - acc", 0.0, T * s),              # same IR operands: v <- v - p; c - R(v)
        2: ("acc = acc + tl.max(x, 1)", T * 2.0, float("-inf")),  # v from -inf (max's neutral) stays -inf
        3: ("acc = tl.maximum(acc, tl.sum(x, 1))", s, s),         # v <- max(v, p) from 0: p alike each trip, R = 65
        4: ("acc = acc * tl.sum(x, 1)  (acc from 1)", s ** T, 0.0),   # v from 0 (sum's neutral) stays 0; 1 * 0
        5: ("acc = acc + tl.min(x, 1)", T * 1.0, float("inf")),
        6: ("acc = tl.minimum(acc, tl.sum(x, 1))", 0.0, 0.0),        # v <- min(v, p) from 0 stays 0
        7: ("acc = tl.maximum(acc, tl.max(x, 1))  (the combine)", 2.0, 2.0),
        8: ("acc = acc / tl.sum(x, 1)", 0.0, float("nan")),         # v from 0 stays 0; 0 / R(v) = 0 / 0
        9: ("acc = acc * tl.max(x, 1)  (acc from 1)", 2.0 ** T, float("-inf")),  # v from -inf stays -inf
        10: ("acc = acc + tl.sum(x, 1)  (the combine)", T * s, T * s),
    }
    for u, (name, src, pred) in cases.items():
        o = torch.full((M,), -7.0, dtype=torch.float32, device=DEV)
        h = k_upd[(1,)](x, o, UPD=u, T=T, M=M, N=N, num_warps=4)
        p = "layout-dependent" if pred is None else pred
        out(f"U{u:<2d} {name:46s} rewritten {rewritten(h)}; stored {stored(o)}; source {src}; U(c, R(v)) {p}")
    # maximum(acc, sum) with data where the per-thread maxima differ from the maxima of the sums
    y = torch.zeros(2, M, N, dtype=torch.float32, device=DEV)
    y[0, :, 0] = 1.0
    y[1, :, N - 1] = 1.0
    o = torch.full((M,), -7.0, dtype=torch.float32, device=DEV)
    h = k_upd[(1,)](y, o, UPD=3, T=2, M=M, N=N, num_warps=4)
    out(f"U3b acc = tl.maximum(acc, tl.sum(x, 1)), trip 0: a 1 in column 0, trip 1: a 1 in column {N - 1}: rewritten "
        f"{rewritten(h)}; stored {stored(o)}; source 1.0; U(c, R(v)) 2.0 when the two columns are in different threads")


def data_assoc():
    """lr-32b's data: trip 0 has 2^24 in column 0 and 1 in column N-1, trip 1 has -2^24 and 1. Each trip's sum
    rounds to 2^24 and is -2^24 + 1 exactly, so every association within the trips gives 1; folding the trips per
    thread first gives 0 + 2 = 2 (column 0 and column N-1 are held by different threads)."""
    x = torch.zeros(2, M, N, dtype=torch.float32, device=DEV)
    x[0, :, 0] = 2.0 ** 24
    x[0, :, N - 1] = 1.0
    x[1, :, 0] = -(2.0 ** 24)
    x[1, :, N - 1] = 1.0
    return x


def part_f():
    out("== Part F: the operand, on the IR the pass sees (acc = acc + tl.sum(operand, 1) from 0, two trips; every "
        "association within the trips gives 1.0, the rewrite 2.0)")
    x = data_assoc()
    xh = torch.zeros(2, M, N, dtype=torch.float16, device=DEV)
    xh[0, :, 0], xh[0, :, N - 1], xh[1, :, 0], xh[1, :, N - 1] = 2048.0, 1.0, -2048.0, 1.0
    ones = torch.ones(2, M, N, dtype=torch.float32, device=DEV)
    desc = TensorDescriptor(x.view(2 * M, N), shape=[2 * M, N], strides=[N, 1], block_shape=[M, N])
    forms = {0: "tl.load(p)", 1: "tl.load(p).to(tl.float32) of float32", 2: "tl.load(p) * 1.0",
             3: "tl.load(p) * SCALE, SCALE: constexpr = 1.0", 4: "tl.load(p) / 1.0", 5: "tl.load(p) - 0.0",
             6: "tl.load(p) + (-0.0)", 7: "tl.load(block pointer)", 8: "a helper returning tl.load(p)",
             9: "tl.maximum(x, x), x = tl.load(p)", 10: "tl.trans(tl.trans(tl.load(p)))",
             11: "tl.reshape(tl.load(p), its shape)", 12: "tl.where(SCALE == 1.0, tl.load(p), 0.0)",
             13: "tl.load(p, mask=always true, other=0.0)", 14: "tl.load(p, cache_modifier='.cg')",
             15: "tl.load(p, mask=rn < n (runtime n = 64), other=0.0)",
             100: "control: tl.load(p) + 0.0", 101: "control: tl.sum(tl.load(p) * 2.0) * 0.5",
             102: "control: float16 (2048, 1; -2048, 1) summed into a float32 acc (converted before U)", 103: "control: a host-descriptor load",
             104: "control: the update inside a runtime if", 105: "control: acc starts from a load (* 0.0)",
             106: "control: tl.where(x == x, x, 0.0) (a runtime mask)", 107: "control: the sum used twice",
             108: "control: x * tl.load(ones)"}
    for fm, name in forms.items():
        o = torch.zeros((M,), dtype=torch.float32, device=DEV)
        h = k_form[(1,)](x, ones if fm == 108 else xh, o, desc, N, FORM=fm, SCALE=1.0, T=2, M=M, N=N, num_warps=4)
        want = 1.0
        out(f"F{fm:<3d} {name:46s} rewritten {rewritten(h)}; stored {stored(o)}; source {want}")


def part_l():
    out("== Part L: the loop and the layout")
    x = data_assoc()
    for kind, name in ((0, "tl.range(0, T, num_stages=3)"), (1, "tl.static_range(T) (unrolled: no loop)"),
                       (2, "while t < T")):
        o = torch.zeros((M,), dtype=torch.float32, device=DEV)
        h = k_range[(1,)](x, o, KIND=kind, T=2, M=M, N=N, num_warps=4)
        out(f"L{kind} {name:40s} rewritten {rewritten(h)}; stored {stored(o)}; source 1.0")
    xa = torch.zeros(2, M, N, dtype=torch.float32, device=DEV)
    xa[0, 0, :], xa[0, M - 1, :], xa[1, 0, :], xa[1, M - 1, :] = 2.0 ** 24, 1.0, -(2.0 ** 24), 1.0
    o = torch.zeros((N,), dtype=torch.float32, device=DEV)
    h = k_axis0[(1,)](xa, o, T=2, M=M, N=N, num_warps=4)
    out(f"L4 tl.sum(axis=0) of a 2-D load                 rewritten {rewritten(h)}; stored {stored(o)}; source 1.0")
    x1 = torch.zeros(2, 1024, dtype=torch.float32, device=DEV)
    x1[0, 0], x1[0, 1023], x1[1, 0], x1[1, 1023] = 2.0 ** 24, 1.0, -(2.0 ** 24), 1.0
    o = torch.zeros((1024,), dtype=torch.float32, device=DEV)
    h = k_1d[(1,)](x1, o, T=2, N=1024, num_warps=4)
    out(f"L5 a 1-D load                                   rewritten {rewritten(h)}; stored {stored(o)}; source 1.0")
    for nw in (1, 2, 8):
        o = torch.zeros((M,), dtype=torch.float32, device=DEV)
        h = k_form[(1,)](x, x, o, None, N, FORM=0, SCALE=1.0, T=2, M=M, N=N, num_warps=nw)
        out(f"L6 tl.load(p), {nw} warps                            rewritten {rewritten(h)}; stored {stored(o)}; source 1.0")
    for (mm, nn) in ((1, 64), (64, 1), (2, 16), (128, 128)):
        xx = torch.zeros(2, mm, nn, dtype=torch.float32, device=DEV)
        xx[0, :, 0], xx[0, :, nn - 1], xx[1, :, 0], xx[1, :, nn - 1] = 2.0 ** 24, 1.0, -(2.0 ** 24), 1.0
        if nn == 1:
            xx[0, :, 0], xx[1, :, 0] = 1.0, 0.0
        o = torch.zeros((mm,), dtype=torch.float32, device=DEV)
        h = k_form[(1,)](xx, xx, o, None, nn, FORM=0, SCALE=1.0, T=2, M=mm, N=nn, num_warps=4)
        out(f"L7 tl.load(p) of [{mm}, {nn}]                      rewritten {rewritten(h)}; stored {stored(o)}; "
            f"source {1.0 if nn > 1 else 1.0}")


# ---- Part W (round 34): the forms of round 33's library review (lr-33a, otl_wording33.py), and the edges of the
# broader exclusion. Data: data_assoc() (two trips, [64, 64], 4 warps).

@triton.jit
def rsum_helper(x):
    return tl.sum(x, axis=1)


@triton.jit
def upd_helper(acc, x):
    return acc + tl.sum(x, axis=1)


@triton.jit
def add_first(a0, a1, b0, b1):
    return a0 + b0, a1


@triton.jit
def max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def k_w(x_ptr, h_ptr, o_ptr, d_ptr, iv, fv, ki, FORM: tl.constexpr, FLAG: tl.constexpr, T: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    if FORM == 7:
        acc = tl.zeros([M], dtype=tl.float32) + iv  # iv a runtime int argument
    elif FORM == 8:
        acc = tl.zeros([M], dtype=tl.float32) + fv  # fv a runtime float argument
    elif FORM == 12:
        acc = tl.zeros([M], dtype=tl.float32) * 1.0
    elif FORM == 23 or FORM == 25:
        acc = tl.zeros([M], dtype=tl.float16)
    else:
        acc = tl.zeros([M], dtype=tl.float32)
    for t in range(T):
        offs = t * M * N + rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + offs)
        if FORM == 0:
            acc = acc + tl.sum(x, axis=1).to(tl.float32)
        elif FORM == 1:
            acc = acc + tl.sum(x, axis=1) * 1.0
        elif FORM == 2:
            acc = (acc + tl.sum(x, axis=1)) * 1.0
        elif FORM == 3:
            if FLAG:
                acc = acc + tl.sum(x, axis=1)
        elif FORM == 4:
            for _ in tl.static_range(1):
                acc = acc + tl.sum(x, axis=1)
        elif FORM == 5:
            s = tl.sum(x, axis=1)
            acc = acc + s
            dead = s * 3.0  # noqa: F841
        elif FORM == 6:
            acc = upd_helper(acc, x)
        elif FORM == 7 or FORM == 8 or FORM == 12:
            acc = acc + tl.sum(x, axis=1)
        elif FORM == 9:
            acc = acc + rsum_helper(x)
        elif FORM == 10:
            acc = acc + tl.sum(x, axis=1)
            dead2 = acc * 2.0  # noqa: F841
        elif FORM == 11:
            acc = acc + tl.sum(x, axis=1)
            if FLAG:
                tl.store(d_ptr + rm, acc)
        elif FORM == 13:
            for _ in tl.static_range(2):
                acc = acc + tl.sum(x, axis=1) * 0.5
        elif FORM == 14:
            acc = acc + (-(-tl.sum(x, axis=1)))
        elif FORM == 15:
            acc = acc + tl.sum(tl.broadcast_to(x, (M, N)), axis=1)
        elif FORM == 16:
            if t >= 0:
                acc = acc + tl.sum(x, axis=1)
        elif FORM == 21:
            acc = acc + tl.sum(x * ki, axis=1)  # ki a runtime int argument
        elif FORM == 22:
            acc = acc + tl.sum(x.to(tl.float64).to(tl.float32), axis=1)
        elif FORM == 23:
            acc = acc + tl.sum(tl.load(h_ptr + offs), axis=1)  # float16 tile and accumulator (tl.sum does not upcast)
        elif FORM == 25:
            acc = acc + tl.reduce(tl.load(h_ptr + offs), 1, max_combine)  # float16, no upcast (tl.max upcasts)
        elif FORM == 24:
            r0, r1 = tl.reduce((x, tl.load(h_ptr + offs)), 1, add_first)
            acc = acc + r0
        elif FORM == 26:
            acc = acc + tl.sum(x, axis=1) * 2.0
    tl.store(o_ptr + rm, acc)


@triton.jit
def k_unroll(x_ptr, o_ptr, Tr, RUNTIME: tl.constexpr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros([M], dtype=tl.float32)
    if RUNTIME:
        for t in tl.range(0, Tr, loop_unroll_factor=2):
            acc = acc + tl.sum(tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :]), axis=1)
    else:
        for t in tl.range(0, T, loop_unroll_factor=2):
            acc = acc + tl.sum(tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :]), axis=1)
    tl.store(o_ptr + rm, acc)


F32 = np.float32
S_TRIPS = (F32(2.0 ** 24), F32(-(2.0 ** 24) + 1.0))  # each trip's row sum of data_assoc(), along any tree


def src_value(form, iv=0, ki=1):
    """The source's value, evaluated in float32 one operation at a time (each trip's row sum is S_TRIPS[t])."""
    acc = F32(iv) if form == 7 else F32(0.0)
    for s in S_TRIPS:
        if form == 13:
            acc = F32(acc + F32(s * F32(0.5)))
            acc = F32(acc + F32(s * F32(0.5)))
        elif form == 21:
            # the row sum of x * ki: elements 2^24 * ki and ki (trip 0), -2^24 * ki and ki (trip 1)
            e = (F32(2.0 ** 24 * ki), F32(ki)) if s > 0 else (F32(-(2.0 ** 24) * ki), F32(ki))
            acc = F32(acc + F32(e[0] + e[1]))
        elif form == 26:
            acc = F32(acc + F32(s * F32(2.0)))
        else:
            acc = F32(acc + s)
    return float(acc)


def unroll_child(argv):
    """Child process: one loop_unroll_factor=2 form; prints one line."""
    runtime, tt = int(argv[0]), int(argv[1])
    x = torch.zeros(4, M, N, dtype=torch.float32, device=DEV)
    x[:2] = data_assoc()
    o = torch.zeros((M,), dtype=torch.float32, device=DEV)
    h = k_unroll[(1,)](x, o, tt, RUNTIME=runtime, T=tt, M=M, N=N, num_warps=4)
    torch.cuda.synchronize()
    print(f"rewritten {rewritten(h)}; stored {stored(o)}", flush=True)


def w24_child():
    """Child process: W24 (a two-operand tl.reduce), whose compile fails; prints one line."""
    x = data_assoc()
    ones = torch.ones(2, M, N, dtype=torch.float32, device=DEV)
    o = torch.zeros((M,), dtype=torch.float32, device=DEV)
    d = torch.zeros((M,), dtype=torch.float32, device=DEV)
    h = k_w[(1,)](x, ones, o, d, 0, 0.0, 1, FORM=24, FLAG=0, T=2, M=M, N=N, num_warps=4)
    torch.cuda.synchronize()
    print(f"rewritten {rewritten(h)}; stored {stored(o)}", flush=True)


def child(args, timeout=600):
    """Run this script with `args` in a child process; return its one line, or why the compile failed."""
    r = subprocess.run([sys.executable, "-I", __file__] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().splitlines()[-1]
    err = r.stderr.splitlines()
    why = [ln.split("Assertion ", 1)[1] for ln in err if "Assertion " in ln]
    why = [f"the pass's assertion {why[0]}"] if why else []
    why += [ln.split("error: ", 1)[1] for ln in err if "error: " in ln and "Failures have been" not in ln][:1]
    why += [ln.split("[", 1)[1].split("]", 1)[0] for ln in err if "Pipeline failed while executing" in ln][:1]
    return f"rejected on the host (exit code {r.returncode}; " + "; ".join(why) + ")"


def part_w():
    out("== Part W (round 34): round 33's review forms, and the edges of the broader exclusion (two trips of "
        "data_assoc(); 'source' is the source's float32 value, operation by operation)")
    x = data_assoc()
    xh = torch.zeros(2, M, N, dtype=torch.float16, device=DEV)
    xh[0, :, 0], xh[0, :, N - 1], xh[1, :, 0], xh[1, :, N - 1] = 2048.0, 1.0, -2048.0, 1.0
    xh2 = torch.ones(2, M, N, dtype=torch.float16, device=DEV)
    xh2[:, :, 1] = 2.0
    ones = torch.ones(2, M, N, dtype=torch.float32, device=DEV)
    forms = [
        ("W0", 0, 0, {}, "acc + tl.sum(x, 1).to(tl.float32)"),
        ("W1", 1, 0, {}, "acc + tl.sum(x, 1) * 1.0"),
        ("W2", 2, 0, {}, "(acc + tl.sum(x, 1)) * 1.0"),
        ("W3", 3, 1, {}, "if FLAG (constexpr True): acc = acc + tl.sum(x, 1)"),
        ("W4", 4, 0, {}, "for _ in tl.static_range(1): acc = acc + tl.sum(x, 1)"),
        ("W5", 5, 0, {}, "s = tl.sum(x, 1); acc = acc + s; dead = s * 3.0 (unused)"),
        ("W6", 6, 0, {}, "acc = upd_helper(acc, x) (reduction and update in a helper)"),
        ("W7", 7, 0, {"iv": 0}, "acc from tl.zeros + iv, iv a runtime int argument = 0"),
        ("W7b", 7, 0, {"iv": 1}, "acc from tl.zeros + iv, iv a runtime int argument = 1 (specialised)"),
        ("W8", 8, 0, {}, "acc from tl.zeros + fv, fv a runtime float argument = 0.0"),
        ("W9", 9, 0, {}, "acc = acc + rsum_helper(x) (reduction in a helper)"),
        ("W10", 10, 0, {}, "acc = acc + tl.sum(x, 1); dead2 = acc * 2.0 (unused)"),
        ("W11", 11, 0, {}, "acc = acc + tl.sum(x, 1); if FLAG (constexpr False): tl.store(acc)"),
        ("W12", 12, 0, {}, "acc from tl.zeros * 1.0"),
        ("W13", 13, 0, {}, "for _ in tl.static_range(2): acc = acc + tl.sum(x, 1) * 0.5"),
        ("W14", 14, 0, {}, "acc + (-(-tl.sum(x, 1)))"),
        ("W15", 15, 0, {}, "tl.sum(tl.broadcast_to(x, its own shape), 1)"),
        ("W16", 16, 0, {}, "if t >= 0 (runtime): acc = acc + tl.sum(x, 1)"),
        ("W21", 21, 0, {"ki": 1}, "tl.sum(x * k, 1), k a runtime int argument = 1 (specialised)"),
        ("W21b", 21, 0, {"ki": 2}, "tl.sum(x * k, 1), k a runtime int argument = 2"),
        ("W22", 22, 0, {}, "tl.sum(x.to(tl.float64).to(tl.float32), 1)"),
        ("W23", 23, 0, {}, "float16 acc + tl.sum(float16 load, 1) (2048, 1; -2048, 1)"),
        ("W25", 25, 0, {}, "float16 acc + tl.reduce(float16 load, 1, maximum) (1, 2 in column 1)"),
        ("W26", 26, 0, {}, "acc + tl.sum(x, 1) * 2.0"),
    ]
    for tag, fm, flag, kw, name in forms:
        o = torch.zeros((M,), dtype=torch.float16 if fm in (23, 25) else torch.float32, device=DEV)
        d = torch.zeros((M,), dtype=torch.float32, device=DEV)
        iv, ki = kw.get("iv", 0), kw.get("ki", 1)
        # W23: 2048 + 1 rounds to 2048 in float16, then 2048 + (-2047) = 1; W25: max 2 per trip, 2 + 2 = 4
        src = {23: 1.0, 25: 4.0, 24: 1.0}.get(fm, None)
        src = src_value(fm, iv=iv, ki=ki) if src is None else src
        try:
            h = k_w[(1,)](x, {23: xh, 25: xh2}.get(fm, ones), o, d, iv, 0.0, ki, FORM=fm, FLAG=flag, T=2, M=M, N=N,
                          num_warps=4)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            msg = [ln for ln in str(e).splitlines() if "Pipeline failed while executing" in ln]
            why = msg[0].split("[", 1)[1].split("]", 1)[0] if msg else str(e).splitlines()[0][:100]
            out(f"{tag:<5} {name:70s} rejected on the host ({type(e).__name__}; {why})")
            continue
        out(f"{tag:<5} {name:70s} rewritten {rewritten(h)}; stored {stored(o.float())}; source {src}")
    out(f"W24   {'r0, r1 = tl.reduce((x, tl.load(ones)), 1, (a0 + b0, a1)); acc = acc + r0':70s} "
        f"{child(['--w24'])}; source 1.0")
    # loop_unroll_factor=2, each in a child process (trips 2 and 3 are zeros, so the source gives 1.0 for 2 or 3 trips)
    for runtime in (0, 1):
        for tt in (2, 3):
            name = f"tl.range(0, {'Tr (runtime)' if runtime else 'T (constexpr)'}, loop_unroll_factor=2), {tt} trips"
            res = child(["--unroll", str(runtime), str(tt)])
            out(f"W20   {name:70s} {res}; source 1.0")


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    part_u()
    part_f()
    part_l()
    part_w()
    with open(__file__.rsplit("/", 1)[0] + "/output.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unroll":
        unroll_child(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--w24":
        w24_child()
    else:
        main()
