"""GPU check of the claim `Claims.SeededDropout`.

Runs the kernel's own host, `seeded_dropout` of `source/04-low-memory-dropout.py` (imported by path;
the module's top level also runs the tutorial's own demonstrations, whose printed tables this script
silences), on the launches of `SETUP.md` and on launches at the edges of `Inst.ok`, and checks
`Post` as it is stated:

* the reference random number of element `j` is computed here from the host's `seed` and the index
  `j` by an independent Philox 4x32-10 in numpy `uint32` (counter `(j, 0, 0, 0)`, key
  `(seed mod 2^32, (seed mod 2^64) >> 32)`), then made non-negative (`r` below `2^31`,
  `2^32 - 1 - r` above), rounded to `float32` and multiplied by the `float32` `4.6566127342e-10`
  (numpy's IEEE `float32` product): `uRef`. Before use, that generator is checked against the
  Random123 known-answer vectors of `philox4x32_10`, against values of the claim's own `rbits`,
  `rint` and `uRef` evaluated in Lean, and against `tl.randint4x` and `tl.rand` on the GPU, bit for
  bit, for seeds of every integer class Triton passes;
* `keep j` is `uRef j > p32` (`p32` the `float32` of the Python `p`), and `dRef` the IEEE `float32`
  `1 - p32`; for each `j < n`: if not kept, the output's bit pattern is `0` (`+0`); if kept, `x[j]`
  finite and `|x[j]/dRef| ≤ 2^126`, the output is finite and within `2^-22·|x[j]/dRef| + 2^-148` of
  the exact `x[j]/dRef` (the worst error / tolerance is printed; elements within `10^-6` of the
  bound are re-checked in exact rational arithmetic). The script also reports how many kept
  outputs equal the IEEE `float32` quotient bit for bit (not part of `Post`);
* the frame: `x` is unchanged (bitwise), and a direct launch of the module's kernel with the host's
  arguments but `output` a slice of a larger sentinel-filled buffer leaves every element outside
  the first `n` unchanged; the same launch with other `BLOCK_SIZE` (covered by `Inst.ok`);
* negative tests: the same checker rejects outputs altered in each of the ways `Post` forbids, and
  references built with a wrong seed, a wrong offset or a wrong round count.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import contextlib
import importlib.util
import io
import os
import warnings
from fractions import Fraction

import numpy as np
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "04-low-memory-dropout.py")

# The module's top level runs the tutorial's demonstrations (lines 76-86 and 143-155) and prints
# their tables; keep its definitions and silence the prints. The launches of lines 145-147 are
# checked below. (The baseline `_dropout` of lines 76-81 passes an `int32` mask to `tl.where`, for
# which Triton warns; that kernel is not part of the claim.)
spec = importlib.util.spec_from_file_location("seeded_dropout_src", SRC)
mod = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CHUNK = 1 << 22
U32 = np.uint32
M32 = 0xFFFFFFFF

# ---------------------------------------------------------------------------------------------
# An independent Philox 4x32 (Salmon et al., 2011), numpy `uint32`, `uint64` for the products.
# ---------------------------------------------------------------------------------------------
ROUND_A, ROUND_B = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
KEY_A, KEY_B = U32(0x9E3779B9), U32(0xBB67AE85)


def philox4x32(c0, c1, c2, c3, k0, k1, rounds=10):
    """`rounds` rounds of Philox 4x32 on `uint32` arrays (counter) and `uint32` scalars (key)."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=U32) for c in (c0, c1, c2, c3))
    k0, k1 = U32(k0), U32(k1)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            p0 = ROUND_A * c0.astype(np.uint64)
            p2 = ROUND_B * c2.astype(np.uint64)
            hi0, lo0 = (p0 >> np.uint64(32)).astype(U32), (p0 & np.uint64(M32)).astype(U32)
            hi2, lo2 = (p2 >> np.uint64(32)).astype(U32), (p2 & np.uint64(M32)).astype(U32)
            c0, c1, c2, c3 = hi2 ^ c1 ^ k0, lo2, hi0 ^ c3 ^ k1, lo0
            k0 = U32(k0 + KEY_A)
            k1 = U32(k1 + KEY_B)
    return c0, c1, c2, c3


def key_of(seed):
    """The key `(seed mod 2^32, (seed mod 2^64) >> 32)`, from the Python integer."""
    s = seed % (1 << 64)
    return s & M32, s >> 32


def rbits_np(seed, j0, j1, rounds=10, shift=0):
    """`rbits seed j` for `j0 ≤ j < j1` (the counter `(j, 0, 0, 0)`)."""
    j = (np.arange(j0, j1, dtype=np.int64) + shift).astype(U32)
    z = np.zeros_like(j)
    k0, k1 = key_of(seed)
    return philox4x32(j, z, z, z, k0, k1, rounds)


SCALE32 = np.float32(4.6566127342e-10)
assert int(np.array(SCALE32).view(U32)) == 805306367  # `Philox.scale32` evaluated in Lean: 0x2FFFFFFF


def uniform_np(r):
    """`uRef` from the bits: non-negative integer (`r`, or `2^32 - 1 - r` from `2^31` on), rounded to
    `float32` (exact in `float64`, then one rounding), times `SCALE32` (IEEE `float32`)."""
    r = r.astype(np.int64)
    z = np.where(r < (1 << 31), r, (1 << 32) - 1 - r)
    return z.astype(np.float64).astype(np.float32) * SCALE32


results = []
# The largest random number: `rint ≤ 2^31 - 1` rounds to `2^31`, and `2^31 * SCALE32 = 1 - 2^-24`,
# so at `p32 = 1` no element is kept.
assert float(uniform_np(np.array([2 ** 31 - 1], dtype=np.uint32))[0]) == 1 - 2.0 ** -24


def report(ok, text):
    results.append(bool(ok))
    print(f"{'ok ' if ok else 'BAD'} {text}", flush=True)


# 1. Random123 known-answer vectors (kat_vectors, philox4x32 10 rounds).
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((M32, M32, M32, M32), (M32, M32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]
for ctr, key, want in KAT:
    got = philox4x32(*[np.array([c], dtype=U32) for c in ctr], *key)
    got = tuple(int(g[0]) for g in got)
    report(got == want, f"Random123 KAT ctr={[hex(c) for c in ctr]}: {[hex(g) for g in got]}")

# 2. The claim's `rbits`, `rint`, `uRef` (bit pattern) and `keep` at `p = 0.5`, evaluated in Lean
#    (`#eval` of the definitions in Claim.lean), for j = 0..3.
LEAN = {
    123: [(287538396, 287538396, 1040784358, False), (2747274318, 1547692977, 1060667350, True),
          (739990181, 739990181, 1051749738, False), (510055638, 510055638, 1047738022, False)],
    -5: [(3360914921, 934052374, 1054781959, False), (1769977607, 1769977607, 1062403949, True),
         (2776887645, 1518079650, 1060435996, True), (2067208600, 2067208600, 1064726066, True)],
    2 ** 40 + 3: [(1440152241, 1440152241, 1059827188, True), (600984846, 600984846, 1049577779, False),
                  (34154909, 34154909, 1015171686, False), (1881843052, 1881843052, 1063277898, True)],
    2 ** 63 + 7: [(842335674, 842335674, 1053348886, False), (3114282146, 1180685149, 1057800102, True),
                  (1176394903, 1176394903, 1057766584, True), (1345058411, 1345058411, 1059084268, True)],
    -2 ** 63: [(2753305574, 1541661721, 1060620231, True), (3801849732, 493117563, 1047208707, False),
               (2411072879, 1883894416, 1063293924, True), (3818099198, 476868097, 1046700911, False)],
}
for seed, rows in LEAN.items():
    r = rbits_np(seed, 0, 4)[0]
    u = uniform_np(r)
    z = [int(v) if v < 2 ** 31 else 2 ** 32 - 1 - int(v) for v in r]
    got = [(int(r[i]), z[i], int(u.view(U32)[i]), bool(u[i] > np.float32(0.5))) for i in range(4)]
    report(got == rows, f"Lean values of rbits/rint/uRef/keep, seed {seed}")


# 3. `tl.randint4x` and `tl.rand` on the GPU against the numpy generator.
@triton.jit
def _probe(out_ptr, u_ptr, n, seed, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    r0, r1, r2, r3 = tl.randint4x(seed, offs)
    tl.store(out_ptr + 4 * offs, r0, mask=m)
    tl.store(out_ptr + 4 * offs + 1, r1, mask=m)
    tl.store(out_ptr + 4 * offs + 2, r2, mask=m)
    tl.store(out_ptr + 4 * offs + 3, r3, mask=m)
    tl.store(u_ptr + offs, tl.rand(seed, offs), mask=m)


SEEDS = [123, 512, 0, 1, -1, 7, -2 ** 31, 2 ** 31 - 1, 2 ** 31, -2 ** 31 - 1, 2 ** 40 + 3,
         2 ** 63 - 1, -2 ** 63, 2 ** 63, 2 ** 63 + 7, 2 ** 64 - 1]
NPROBE = 1 << 20
for seed in SEEDS:
    words = torch.empty(4 * NPROBE, dtype=torch.int32, device=DEVICE)
    us = torch.empty(NPROBE, dtype=torch.float32, device=DEVICE)
    _probe[(triton.cdiv(NPROBE, 1024),)](words, us, NPROBE, seed, BLOCK=1024)
    torch.cuda.synchronize()
    w = words.cpu().numpy().view(U32).reshape(NPROBE, 4)
    ref = rbits_np(seed, 0, NPROBE)
    bad_w = sum(int((w[:, i] != ref[i]).sum()) for i in range(4))
    bad_u = int((us.cpu().numpy().view(U32) != uniform_np(ref[0]).view(U32)).sum())
    report(bad_w == 0 and bad_u == 0,
           f"tl.randint4x / tl.rand, seed {seed}, {NPROBE} offsets: {bad_w} word and {bad_u} "
           f"tl.rand mismatches")
del words, us


# ---------------------------------------------------------------------------------------------
# The kernel against `Post`.
# ---------------------------------------------------------------------------------------------
def p32_of(p):
    return np.float32(p)  # C's (float) of the double: `FloatV.hostF32`


def post_check(x_np, out_np, p, seed, rounds=10, shift=0):
    """Check `Post`'s per-element clauses; returns (violations, worst err/tol, kept, dropped,
    unconstrained, ieee_equal)."""
    n = x_np.size
    p32 = p32_of(p)
    d = np.float32(1.0) - p32  # IEEE `float32` `1 - p32`
    d64 = float(d)
    bad = kept = dropped = unc = ieee = 0
    worst = 0.0
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        u = uniform_np(rbits_np(seed, s, e, rounds, shift)[0])
        keep = u > p32
        a = x_np[s:e]
        c = out_np[s:e]
        cb = c.view(U32)
        # not kept: the bit pattern 0
        bad += int((~keep & (cb != 0)).sum())
        dropped += int((~keep).sum())
        with np.errstate(all="ignore"):
            a64 = a.astype(np.float64)
            q = a64 / d64
            guard = keep & np.isfinite(a) & (np.abs(q) <= 2.0 ** 126)
            unc += int((keep & ~guard).sum())
            kept += int(guard.sum())
            cg = c[guard].astype(np.float64)
            qg = q[guard]
            tol = 2.0 ** -22 * np.abs(qg) + 2.0 ** -148
            ratio = np.where(np.isfinite(cg), np.abs(cg - qg) / tol, np.inf)
            ieee += int((c[guard].view(U32) == (a[guard] / d).view(U32)).sum())
        if ratio.size:
            # every element at or above 1 - 1e-6 of its bound (non-finite outputs included) is
            # re-checked exactly; below that the float64 ratio decides (its error is ~2^-31)
            ag = a[guard]
            for k in np.nonzero(ratio > 1 - 1e-6)[0]:
                ck = float(cg[k])
                if not np.isfinite(ck):
                    bad += 1
                    continue
                qk = Fraction(float(ag[k])) / Fraction(d64)
                if abs(Fraction(ck) - qk) > Fraction(1, 2 ** 22) * abs(qk) + Fraction(1, 2 ** 148):
                    bad += 1
            worst = max(worst, float(np.max(ratio)))
    return bad, worst, kept, dropped, unc, ieee


def direct(x, p, seed, block, guard=4096):
    """The host's launch (lines 135-139) with `output` a slice of a guarded buffer."""
    n = x.numel()
    SENT = -12345.5
    buf = torch.full((guard + n + guard,), SENT, dtype=torch.float32, device=x.device)
    out = buf[guard:guard + n]
    mod._seeded_dropout[(triton.cdiv(n, block),)](x, out, n, p, seed, BLOCK_SIZE=block)
    torch.cuda.synchronize()
    g = torch.cat([buf[:guard], buf[guard + n:]])
    return out, bool((g == SENT).all().item())


def check(name, x, p, seed, blocks=(), host=True):
    xb = x.clone()
    x_np = x.cpu().numpy()
    ok = True
    parts = []
    outs = []
    if host:
        out = mod.seeded_dropout(x, p=p, seed=seed)
        torch.cuda.synchronize()
        outs.append(("host", out.cpu().numpy(), True))
        del out
    for b in blocks:
        out, frame = direct(x, p, seed, b)
        outs.append((f"BLOCK {b}", out.cpu().numpy(), frame))
        del out
    for label, o, frame in outs:
        bad, worst, kept, dropped, unc, ieee = post_check(x_np, o, p, seed)
        ok &= bad == 0 and frame
        parts.append(f"{label}: {bad} violations, worst err/tol {worst:.3f}, kept {kept} "
                     f"({ieee} = IEEE quotient), dropped {dropped}, unconstrained {unc}"
                     + ("" if label == "host" else f", frame {frame}"))
    ok &= torch.equal(x.view(torch.int32), xb.view(torch.int32))
    report(ok, f"{name} (n={x.numel()}, p={p}, seed={seed}): " + "; ".join(parts))
    return outs[0][1] if outs else None


# The file's launches (lines 143-147): `torch.randn(10)`, p = 0.5, seeds 123, 123, 512.
torch.manual_seed(0)
x = torch.randn(size=(10,), device=DEVICE)
o1 = check("file: seed 123", x, 0.5, 123, blocks=(1024,))
o2 = check("file: seed 123 again", x, 0.5, 123)
o3 = check("file: seed 512", x, 0.5, 512)
report(np.array_equal(o1.view(U32), o2.view(U32)), "file: seed 123 twice gives the same output")

# Sizes around the block, larger vectors, other p (1 - p rounded or exact) and every seed class.
g = torch.Generator().manual_seed(1)
CASES = [
    (0, 0.5, 123), (1, 0.5, 123), (2, 0.5, 1), (1023, 0.5, 512), (1024, 0.3, 0), (1025, 0.1, -1),
    (4097, 0.9, 2 ** 31 - 1), (98432, 0.5, 123), (98433, 0.2, -2 ** 31), (1 << 20, 0.5, 2 ** 31),
    ((1 << 20) + 3, 0.7, 2 ** 40 + 3), (1 << 22, 0.25, 2 ** 63 - 1), (3_000_001, 0.6, -2 ** 63),
    (1 << 24, 0.5, 2 ** 63), (1 << 24, 0.1, 2 ** 64 - 1), (1 << 26, 0.5, 512),
    (100_000, 1e-30, 9), (100_000, 2.0 ** -149, 9), (100_000, 0.99999994, 5),  # p32 = 1 - 2^-24
    (100_000, 0.999, 5), (100_000, 1.0 / 3.0, 77),
    # the ends of [0, 1]: p32 = +0, -0 and 1, and doubles that round to 0 or 1 in float32
    (100_000, 0.0, 21), (100_000, -0.0, 22), (100_000, 1.0, 23), (100_000, 1e-50, 24),
    (100_000, 1.0 - 2.0 ** -26, 25), (1 << 24, 0.0, 123), (1 << 24, -0.0, 512), (1 << 24, 1.0, 7),
]
for n, p, seed in CASES:
    x = torch.randn(n, generator=g).to(DEVICE)
    check("randn", x, p, seed, blocks=(1024, 1, 128, 4096) if 0 < n <= 100_000 else ())
    del x
torch.cuda.empty_cache()

# Large block sizes: direct launches with BLOCK_SIZE = 2^16 (about 40 s to compile). BLOCK_SIZE =
# 2^20, the top of Inst.ok's range, did not finish compiling within 25 minutes and is not run.
for n, p, seed, blocks in (((1 << 20) + 5, 0.3, 31, (1 << 16,)), (70_000, 0.5, 32, (1 << 16,))):
    x = torch.randn(n, generator=g).to(DEVICE)
    check("randn, large blocks", x, p, seed, blocks=blocks)
    del x
torch.cuda.empty_cache()

# Every float32 bit pattern class in x: random bit patterns, NaN, infinities, signed zeros,
# subnormals, values whose quotient exceeds 2^126 (unconstrained when kept, +0 when dropped).
for n, p, seed in ((1025, 0.5, 3), (98433, 0.3, 4), (4097, 0.999, -7)):
    bits = torch.randint(-2 ** 31, 2 ** 31, (n,), generator=g, dtype=torch.int64).to(torch.int32)
    check("random bit patterns", bits.view(torch.float32).to(DEVICE), p, seed, blocks=(1024, 16))
special = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0,
                        2.0 ** -126, 2.0 ** -149, -2.0 ** -149, 3 * 2.0 ** -149, 2.0 ** -127,
                        3.4028234663852886e38, 4.2e37, 8.5e37, 1.0 + 2.0 ** -23, 1e-40],
                       dtype=torch.float32)
for p in (0.5, 0.1, 0.99999994):
    check("special values", special.repeat(300).to(DEVICE), p, 11, blocks=(1024, 16))

# ---------------------------------------------------------------------------------------------
# Negative tests: the checker must reject each of these.
# ---------------------------------------------------------------------------------------------
torch.manual_seed(2)
n, p, seed = 98432, 0.3, 123
x = torch.randn(n, device=DEVICE)
x_np = x.cpu().numpy()
out = mod.seeded_dropout(x, p=p, seed=seed).cpu().numpy()
u = uniform_np(rbits_np(seed, 0, n)[0])
keep = u > p32_of(p)
ik = int(np.nonzero(keep & (np.abs(x_np) > 1e-3))[0][0])
idr = int(np.nonzero(~keep)[0][0])
d = np.float32(1.0) - p32_of(p)


def rejects(name, o, **kw):
    bad = post_check(x_np, o, p, kw.pop("seed", seed), **kw)[0]
    report(bad > 0, f"negative: {name} rejected ({bad} violations)")


report(post_check(x_np, out, p, seed)[0] == 0, "negative tests: the unaltered output passes")
o = out.copy(); o[ik] = 0.0
rejects("a kept element set to +0", o)
o = out.copy(); o[idr] = x_np[idr] / d
rejects("a dropped element set to x/(1-p)", o)
o = out.copy(); o[idr] = -0.0
rejects("a dropped element set to -0", o)
o = out.copy(); o[ik] = np.float32(o[ik] * np.float32(1 + 2.0 ** -19))
rejects("a kept element off by 2^-19 relative", o)
o = out.copy(); o[ik] = np.float32(x_np[ik])
rejects("a kept element not rescaled", o)
rejects("reference with seed + 1", out, seed=seed + 1)
rejects("reference with offsets j + 1", out, shift=1)
rejects("reference with 9 rounds", out, rounds=9)
xg = x.clone()
buf_out, frame = direct(xg, p, seed, 1024)
report(frame, "negative tests: the guarded launch keeps its guard")
buf = torch.full((4096 + n + 4096,), -12345.5, dtype=torch.float32, device=DEVICE)
buf[4096:4096 + n] = torch.from_numpy(out).to(DEVICE)
buf[4096 + n] = 0.0  # a write past the first n elements
report(not bool((torch.cat([buf[:4096], buf[4096 + n:]]) == -12345.5).all().item()),
       "negative: a write outside the first n elements is detected by the frame check")

print(f"{sum(results)}/{len(results)} cases pass")
assert all(results)
