"""Library issue L164: which integer atomic a float `tl.atomic_max` / `tl.atomic_min` runs per lane.

Triton 3.8.0 (`semantic.py:1299-1373`) lowers a float32 (float64) `atomic_max` to two predicated
integer atomics, `max.s32` (`.s64`) under `mask & !signbit(val)` and `min.u32` (`.u64`) under
`mask & signbit(val)`, and `atomic_min` to `min.s32` and `max.u32` with the same predicates. This
script checks, on the device:
  - the PTX: the two `atom`s and their predicates;
  - every (old, operand) pair of a set of special and random bit patterns, NaNs of both signs
    included: the new memory equals the one integer atomic the operand's sign bit picks, applied
    to the old bits (so exactly one ran), the returned value is the old bits, and the new value
    is inside `fBound` (one of the two values, and not below (above) either when neither is NaN);
  - a masked-off lane's cell is unchanged (no atomic ran).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/relaxed-atomics/float_maxmin.py
Output: float_maxmin.out.txt beside this file."""
import math
import os
import random
import re
import struct

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def k(p, v, m, o, MAX: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    msk = tl.load(m + j) != 0
    if MAX:
        r = tl.atomic_max(p + j, tl.load(v + j), mask=msk, sem="relaxed")
    else:
        r = tl.atomic_min(p + j, tl.load(v + j), mask=msk, sem="relaxed")
    tl.store(o + j, r, mask=msk)


FMT = {
    32: dict(fl=torch.float32, it=torch.int32, pk="<f", pi="<I", sign=1 << 31, nan=(0x7fc00000, 0xffc00000)),
    64: dict(fl=torch.float64, it=torch.int64, pk="<d", pi="<Q", sign=1 << 63,
             nan=(0x7ff8000000000000, 0xfff8000000000000)),
}


def bits(w, x):
    return struct.unpack(FMT[w]["pi"], struct.pack(FMT[w]["pk"], x))[0]


def val(w, b):
    return struct.unpack(FMT[w]["pk"], struct.pack(FMT[w]["pi"], b))[0]


def signed(w, b):
    return b - (1 << w) if b >> (w - 1) else b


def unsigned(w, b):
    return b & ((1 << w) - 1)


def expected(w, is_max, o, v):
    """The new bits: the one integer atomic the operand's sign bit picks."""
    if v & FMT[w]["sign"]:   # sign bit set: min.u (max) / max.u (min)
        return min(o, v) if is_max else max(o, v)
    so, sv = signed(w, o), signed(w, v)
    return unsigned(w, max(so, sv) if is_max else min(so, sv))


def in_fbound(w, is_max, o, v, n):
    if n not in (o, v):
        return False
    fo, fv, fn = val(w, o), val(w, v), val(w, n)
    if math.isnan(fo) or math.isnan(fv):
        return True
    return (fo <= fn and fv <= fn) if is_max else (fn <= fo and fn <= fv)


def patterns(w, rng):
    nan_p, nan_n = FMT[w]["nan"]
    sp = [0.0, -0.0, 1.0, -1.0, 2.0, -3.0, 5.0, math.inf, -math.inf, 1e-300 if w == 64 else 1e-40,
          -(1e-300 if w == 64 else 1e-40)]
    b = [bits(w, x) for x in sp] + [nan_p, nan_n, nan_p | 1, nan_n | 1,
                                    (nan_p & ~(1 << (w - 2))) | 1,   # signalling NaN, sign clear
                                    (nan_n & ~(1 << (w - 2))) | 1]   # signalling NaN, sign set
    b += [rng.getrandbits(w) for _ in range(40 - len(b))]
    return b


def run(w, is_max, rng):
    F = FMT[w]
    pats = patterns(w, rng)
    pairs = [(o, v) for o in pats for v in pats]
    N = 1
    while N < len(pairs):
        N *= 2
    pairs += [(pats[0], pats[1])] * (N - len(pairs))
    lane_mask = [1] * len(pairs)
    for i in range(0, len(pairs), 17):   # every 17th lane masked off
        lane_mask[i] = 0
    to_t = lambda xs: torch.tensor([signed(w, x) for x in xs], dtype=torch.int64).to(F["it"]).view(F["fl"]).cuda()
    p = to_t([o for o, _ in pairs])
    v = to_t([x for _, x in pairs])
    m = torch.tensor(lane_mask, dtype=torch.int32, device="cuda")
    o = torch.zeros(N, dtype=F["fl"], device="cuda")
    h = k[(1,)](p, v, m, o, is_max, N, num_warps=4)
    torch.cuda.synchronize()
    name = "atomic_max" if is_max else "atomic_min"
    out(f"## float{w} {name}, sem=relaxed, {N} lanes ({sum(lane_mask)} enabled)")
    out("PTX atoms: " + " | ".join(sorted(set(re.sub(r"%\w+", "%_", a) for a in
                                             re.findall(r"@!?%p\d+ atom\.[^;]*;", h.asm["ptx"])))))
    newv = [unsigned(w, x) for x in p.view(F["it"]).cpu().tolist()]
    ret = [unsigned(w, x) for x in o.view(F["it"]).cpu().tolist()]
    bad_one, bad_ret, bad_fb, bad_masked, nan_ops, nan_written = 0, 0, 0, 0, 0, 0
    for i, (ob, vb) in enumerate(pairs):
        nb = newv[i]
        if not lane_mask[i]:
            bad_masked += nb != ob
            continue
        bad_one += nb != expected(w, is_max, ob, vb)
        bad_ret += ret[i] != ob
        bad_fb += not in_fbound(w, is_max, ob, vb, nb)
        if math.isnan(val(w, vb)):
            nan_ops += 1
            nan_written += (nb == vb and nb != ob)
    out(f"enabled lanes whose new value is not the one atomic the sign bit picks: {bad_one}")
    out(f"enabled lanes whose returned value is not the old value: {bad_ret}")
    out(f"enabled lanes whose new value is outside fBound: {bad_fb}")
    out(f"masked-off lanes whose cell changed: {bad_masked}")
    out(f"enabled lanes with a NaN operand: {nan_ops}; of these, the NaN was written: {nan_written}")
    out("")


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    out("")
    rng = random.Random(164)
    for w in (32, 64):
        for is_max in (True, False):
            run(w, is_max, rng)
    with open(os.path.join(HERE, "float_maxmin.out.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
