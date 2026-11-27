"""GPU check of the claim `Claims.ChunkGlobalCumsum` (Claim.lean).

Runs the kernel's own entry point, `chunk_global_cumsum` of `source/chunk_global_cumsum.py`
(imported by path), on the host's test launches (fixed-length and variable-length, forward and
reverse, without a scale and with `RCP_LN2`; the variable-length ones with `int32` and with
`int64` tables) and, through direct launches of the module's kernel, on launches at the edges of
`Inst.ok` (other block sizes `BT`, empty sequences and grids, long sequences, magnitudes near
`2^126`, non-finite gates, extreme scales), and checks `Post` as it is stated, for every sequence
`n`, position `p` in it and head `h`:

* `InRange`: the terms of `rng n p` finite, `A = Σ|s| ≤ 2^126`, and with a scale the double and
  its `float32` value finite and `|σ₃₂|·A ≤ 2^126`;
* if so, the output is finite and `|o[p, h] − σ·S| ≤ tol`, with `S = Σ s[q, h]` over `rng n p`
  (from the sequence's start to `p`, or from `p` to its end) and `depth`, `tol` exactly as `Post`
  defines them (`depth = max(laneDepth, carryDepth) + 1`, `carryDepth = 0` before the first
  block and `⌈log₂ BT⌉ + before` after it; `tol = (depth + 1)·2^-23·|σ₃₂|·A + |σ − σ₃₂|·A +
  2^-125·(1 + |σ₃₂|)`, `σ = σ₃₂ = 1` without a scale), computed in `float64` from the host's
  inputs: positions are grouped by sequence from `cu_seqlens` (or the rows) alone, never from
  the kernel's indexing. The reference sums are `float64` cumulative sums of at most `2^22` terms,
  an error below `2^-31·A`, far below `tol`'s `2^-22·A`.

It reports the largest error / tolerance over the elements in range and counts those out of
range, and checks that plausible wrong outputs fail `Post`: outputs perturbed by `2^-18` relative,
an exclusive scan, a block-local scan (the running sum dropped), a sign flip, and the running sum
missing its last block's `tl.sum`. Also checked: the host's tables satisfy `TablesOk` (`Pre`);
the inputs are unchanged; and a direct launch of the module's kernel with the host's own
arguments and `o` prefilled with a sentinel leaves every element outside the written set
(`written`: the sequences' positions) unchanged, also on tables whose sequences do not cover
`[0, T)`.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import pathlib

import numpy as np
import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "chunk_global_cumsum.py"
spec = importlib.util.spec_from_file_location("chunk_global_cumsum_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
RCP_LN2 = 1.4426950216
SENT = 7.0e30


def f32(x):
    with np.errstate(over="ignore"):
        return float(np.float32(x))


def clog2(n):
    """`Nat.clog 2 n`."""
    return 0 if n <= 1 else (n - 1).bit_length()


def seqs(B, T, cu):
    """`(lo, hi)` of every sequence in the flat time axis (`nSeq`, `lo`, `hi`)."""
    if cu is None:
        return [(b * T, b * T + T) for b in range(B)]
    return [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]


def tables_ok(cu, T):
    """`TablesOk`, on the host's table."""
    N = len(cu) - 1
    return cu[0] >= 0 and all(cu[i] <= cu[i + 1] for i in range(N)) and cu[N] <= T


def post_check(g, o, BT, reverse, scale, cu):
    """Post's value clause; returns (worst err/tol over InRange, all finite there, # out of range)."""
    B, T, H = g.shape
    s = g.reshape(B * T, H).double()
    out = o.reshape(B * T, H).double()
    if scale is None:
        sig = sig32 = 1.0
        scale_ok = True
    else:
        sig, sig32 = float(scale), f32(scale)
        scale_ok = math.isfinite(sig) and math.isfinite(sig32)
    worst, finite, nout = 0.0, True, 0
    L = clog2(BT)
    for lo, hi in seqs(B, T, cu):
        n = hi - lo
        if n <= 0:
            continue
        x = s[lo:hi]
        bad = (~torch.isfinite(x)).double()
        xz = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if reverse:
            cs = lambda t: t.flip(0).cumsum(0).flip(0)
        else:
            cs = lambda t: t.cumsum(0)
        S, A, nbad = cs(xz), cs(xz.abs()), cs(bad)
        j = torch.arange(n, device=s.device)
        nblk = (n + BT - 1) // BT
        blk = j // BT
        start = blk * BT
        lane = (start + BT - 1 - j) if reverse else (j - start)
        before = (nblk - 1 - blk) if reverse else blk
        carry = torch.where(before == 0, torch.zeros_like(before), L + before)
        depth = (torch.maximum(lane, carry) + 1).double()[:, None]
        in_range = (nbad == 0) & (A <= 2.0**126)
        if not scale_ok:
            in_range &= False
        in_range &= abs(sig32) * A <= 2.0**126
        tol = (depth + 1) * 2.0**-23 * (abs(sig32) * A) + abs(sig - sig32) * A + 2.0**-125 * (1 + abs(sig32))
        got = out[lo:hi]
        err = (got - sig * S).abs()
        finite &= bool(torch.isfinite(got)[in_range].all().item())
        ratio = torch.where(tol > 0, err / torch.where(tol > 0, tol, torch.ones_like(tol)),
                            torch.where(err > 0, torch.full_like(err, math.inf), torch.zeros_like(err)))
        ratio = torch.where(torch.isnan(ratio), torch.full_like(ratio, math.inf), ratio)
        if in_range.any():
            worst = max(worst, ratio[in_range].max().item())
        nout += int((~in_range).sum().item())
    return worst, finite, nout


def post_check_sliced(g, o, BT, reverse, scale, cu, step=8):
    """`post_check` over slices of at most `step` heads (heads are independent), to stay under
    4 GB of device memory."""
    worst, finite, nout = 0.0, True, 0
    for h0 in range(0, g.shape[2], step):
        w, f, n = post_check(g[:, :, h0:h0 + step], o[:, :, h0:h0 + step], BT, reverse, scale, cu)
        worst, finite, nout = max(worst, w), finite and f, nout + n
    return worst, finite, nout


def direct(g, BT, reverse, scale, cu_t):
    """The host's launch (lines 123-138) with `o` prefilled with a sentinel, any `BT`."""
    B, T, H = g.shape
    o = torch.full_like(g, SENT)
    N = B if cu_t is None else len(cu_t) - 1
    mod.chunk_global_cumsum_scalar_kernel[(N * H,)](
        s=g, o=o, scale=scale, cu_seqlens=cu_t, T=T, B=B, H=H, REVERSE=reverse,
        HAS_SCALE=scale is not None, IS_VARLEN=cu_t is not None, BT=BT, num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    return o


def frame_ok(o, B, T, cu):
    w = torch.zeros(B * T, dtype=torch.bool, device=o.device)
    for lo, hi in seqs(B, T, cu):
        w[lo:hi] = True
    rest = o.reshape(B * T, o.shape[2])[~w]
    return bool((rest == SENT).all().item())


results = []
worst_all = 0.0


def check(name, g, cu=None, idx=torch.int32, scales=(None, RCP_LN2), host=True, BT=256):
    global worst_all
    B, T, H = g.shape
    cu_t = None if cu is None else torch.tensor(cu, dtype=idx, device=DEVICE)
    g_before = g.clone()
    line = []
    ok = True
    if cu is not None:
        tok = tables_ok(cu, T)
        ok &= tok
        line.append(f"TablesOk {tok}")
    for reverse in (False, True):
        for scale in scales:
            if host:
                assert BT == 256
                o = mod.chunk_global_cumsum(g, reverse=reverse, scale=scale, cu_seqlens=cu_t)
                torch.cuda.synchronize()
                w, f, nout = post_check_sliced(g, o, BT, reverse, scale, cu)
                ok &= w <= 1.0 and f and o.dtype == torch.float32
                worst_all = max(worst_all, w)
            o2 = direct(g, BT, reverse, scale, cu_t)
            w2, f2, nout2 = post_check_sliced(g, o2, BT, reverse, scale, cu)
            fr = frame_ok(o2, B, T, cu)
            ok &= w2 <= 1.0 and f2 and fr
            worst_all = max(worst_all, w2)
            line.append(f"{'rev' if reverse else 'fwd'}/{'-' if scale is None else scale}: "
                        f"{(w if host else w2):.3f} (out {nout2}) frame {fr}")
    ok &= torch.equal(g.view(torch.int32), g_before.view(torch.int32))
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} B{B} T{T} H{H} BT{BT} "
          f"{'' if cu is None else str(idx)[6:] + ' N' + str(len(cu) - 1)}: " + "; ".join(line), flush=True)


gen = torch.Generator().manual_seed(4)


def gate(B, T, H):
    return torch.nn.functional.logsigmoid(torch.randn((B, T, H), generator=gen)).to(DEVICE)


# The host's test launches (source lines 197-209), int32 and int64 tables.
for B, T, H, cu in [(1, 63, 1, None), (2, 500, 4, None), (2, 1000, 5, None), (3, 1024, 6, None),
                    (4, 2048, 8, None), (1, 15, 2, [0, 15]), (1, 1000, 3, [0, 256, 500, 1000]),
                    (1, 2000, 4, [0, 15, 100, 300, 1200, 2000]),
                    (1, 2048, 4, [0, 1, 100, 300, 1200, 2048]), (1, 1, 32, [0, 1]),
                    (1, 65536, 32, [0, 1, 64, 65, 4096, 30000, 65536])]:
    g = gate(B, T, H)
    for idx in ((torch.int32, torch.int64) if cu is not None else (torch.int32,)):
        check("host", g, cu, idx)

# Negatives: wrong outputs must fail Post (worst error / tol above 1).
g = gate(2, 1000, 5)
for reverse in (False, True):
    for scale in (None, RCP_LN2):
        o = mod.chunk_global_cumsum(g, reverse=reverse, scale=scale)
        sg = 1.0 if scale is None else scale
        blocklocal = o.clone()
        lastmiss = o.clone()
        for b0 in range(0, 1000, 256):
            blk = g[:, b0:b0 + 256].double()
            c = (blk.flip(1).cumsum(1).flip(1) if reverse else blk.cumsum(1)) * sg
            blocklocal[:, b0:b0 + 256] = c.float()
        # the running sum missing one block's tl.sum: shift every later block by that block's sum
        first = 768 if reverse else 0
        span = slice(0, 768) if reverse else slice(256, 1000)
        lastmiss[:, span] = (o[:, span].double() - sg * g[:, first:first + 256].double().sum(1, keepdim=True)).float()
        for what, bad in (("x(1+2^-18)", o * (1 + 2.0**-18)), ("exclusive scan", o - sg * g),
                          ("block-local scan", blocklocal), ("sign flip", -o),
                          ("running sum missing a block", lastmiss)):
            w, _, _ = post_check_sliced(g, bad, 256, reverse, scale, None)
            rej = w > 1.0
            results.append(rej)
            print(f"{'ok ' if rej else 'BAD'} negative {what} {'rev' if reverse else 'fwd'}/"
                  f"{'-' if scale is None else scale}: worst error / tol {w:.3g} (rejected {rej})", flush=True)

# Edges of Inst.ok and of InRange.
g = gate(1, 70001, 6)
cu = [0, 0, 1, 17, 17, 255, 256, 257, 4096, 4097, 70001]  # empty sequences, lengths 1, 16, block-sized ends
for BT in (1, 32, 64, 128, 2048):
    check("edge empty and short sequences", g, cu, torch.int64, host=False, BT=BT)
check("edge empty and short sequences", g, cu, torch.int64)
g = gate(1, 300, 3)
check("edge sequences not covering [0, T)", g, [5, 20, 20, 37, 200], torch.int32, host=False, BT=32)
check("edge no sequence (empty grid)", g, [7], torch.int32)
check("edge fixed T = 0", gate(3, 0, 2), None)
check("edge 2^22 tokens, BT 32", gate(1, 2**22, 2), [0, 2**22], torch.int64, host=False, BT=32)
check("edge 2^22 tokens fixed", gate(1, 2**22, 2), None)
g = (torch.randn((1, 4000, 4), generator=gen) * 2.0**114).to(DEVICE)
check("edge near 2^126", g, [0, 1000, 4000], torch.int32, scales=(None, 1.0, 3.0))
g = gate(1, 600, 2)
g[0, 10, 0] = float("nan"); g[0, 300, 1] = float("inf"); g[0, 301, 1] = -float("inf")
check("edge NaN/inf in some positions", g, [0, 200, 600], torch.int32)
check("edge scales", gate(1, 1000, 3), [0, 999, 1000], torch.int64,
      scales=(-2.5, 0.0, -0.0, 1e-45, 1e-40, 1e30, 1e39))
check("edge 64 heads, 2^20 tokens", gate(1, 2**20, 64), [0, 1, 2**19, 2**20 - 7, 2**20], torch.int64)
check("edge fixed, 64 rows", gate(64, 3000, 8), None)
check("edge fixed, BT 64", gate(5, 3001, 3), None, host=False, BT=64)

# fla's own tests of this kernel (tests/ops/utils/test_cumsum.py): 2050 sequences of 16 tokens with
# 32 heads, and 2050 fixed rows of a `1e8` prefix and a unit suffix with 32 heads.
check("fla test, 2050 sequences", gate(1, 32800, 32), list(range(0, 32801, 16)), torch.int32)
g = torch.cat((torch.full((32,), 1e8), torch.ones(32))).reshape(1, 64, 1).expand(2050, 64, 32)
check("fla test, 1e8 prefix and unit suffix", g.contiguous().to(DEVICE), None)

n_bad = results.count(False)
print(f"peak device memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"{len(results)} cases: worst error / tolerance {worst_all:.4f} (margin {1 / max(worst_all, 1e-300):.1f}x); "
      + ("every in-range element within tol and finite, every table TablesOk, every frame clause held"
         if n_bad == 0 else f"{n_bad} FAILED"))
sys.exit(1 if n_bad else 0)
