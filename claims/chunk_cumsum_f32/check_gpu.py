"""GPU check of the claim `Claims.ChunkCumsum` (Claim.lean).

Runs the kernel's own entry point, `chunk_local_cumsum` of
`source/chunk_cumsum.py` (imported by path), on the host's test launches
(fixed-length and variable-length, forward and reverse, without a scale and with `RCP_LN2`; the
variable-length ones with `int32` and with `int64` tables) and on launches at the edges of
`Inst.ok`, and checks `Post` as it is stated, for every sequence `n`, position `p` in it and head
`h`:

* `InRange`: the terms of `rng n p` finite, `A = Σ|s| ≤ 2^126`, and with a scale the double and
  its `float32` value finite and `|σ₃₂|·A ≤ 2^126`;
* if so, the output is finite and `|o[p, h] − σ·S| ≤ tol`, with `S = Σ s[q, h]` over `rng n p`
  (from the chunk's start to `p`, or from `p` to the chunk's end), `depth` (`p − start`, or
  `start + BT − 1 − p`) and `tol` exactly as `Post` defines them
  (`(depth + 1)·2^-23·|σ₃₂|·A + |σ − σ₃₂|·A + 2^-128·(1 + |σ₃₂|)`, `σ = σ₃₂ = 1` without a scale),
  computed in `float64` from the host's inputs: positions are grouped by chunk from `cu_seqlens`
  (or the rows) alone, never from `chunk_indices`, and each chunk is summed on its own (`float64`,
  at most `BT` terms: an error below `2^-40·A`, far below `tol`'s `2^-23·A` term).

It reports the largest error / tolerance over the elements in range and counts those out of
range, and checks that outputs perturbed by `2^-18` relative (or a sign flip) fail `Post`. Also checked: the host's tables satisfy `TablesOk` (`Pre`); the inputs are unchanged; and
a direct launch of the module's kernel with the host's own arguments and `o` prefilled with a
sentinel leaves every element outside the written set (`written`: the sequences' positions)
unchanged, on tables whose sequences do not cover `[0, T)` and on a permuted `chunk_indices`.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import pathlib
import sys

import numpy as np
import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "chunk_cumsum.py"
spec = importlib.util.spec_from_file_location("chunk_cumsum_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
RCP_LN2 = 1.4426950216
SENT = 7.0e30


def f32(x):
    with np.errstate(over="ignore"):
        return float(np.float32(x))


def seqs(B, T, cu):
    """`(lo, hi)` of every sequence in the flat time axis (`nSeq`, `lo`, `hi`)."""
    if cu is None:
        return [(b * T, b * T + T) for b in range(B)]
    return [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]


def tables_ok(cu, ci, T, BT):
    """`TablesOk`, on the host's tables."""
    N = len(cu) - 1
    if not (cu[0] >= 0 and all(cu[i] <= cu[i + 1] for i in range(N)) and cu[N] <= T):
        return False
    rows = [tuple(r) for r in ci]
    if len(set(rows)) != len(rows):
        return False
    for n, c in rows:
        if not (0 <= n < N and c >= 0 and c * BT < cu[n + 1] - cu[n]):
            return False
    want = {(n, c) for n in range(N) for c in range((cu[n + 1] - cu[n] + BT - 1) // BT)}
    return want <= set(rows)


def post_check(g, o, BT, reverse, scale, cu):
    """Post's value clause; returns (worst err/tol over InRange, all finite there, # out of range)."""
    B, T, H = g.shape
    s = g.reshape(B * T, H).double()
    out = o.reshape(B * T, H).double()
    starts, his = [], []
    for lo, hi in seqs(B, T, cu):
        for c0 in range(lo, hi, BT):
            starts.append(c0)
            his.append(hi)
    if not starts:
        return 0.0, True, 0
    st = torch.tensor(starts, device=s.device)
    hi = torch.tensor(his, device=s.device)
    pos = st[:, None] + torch.arange(BT, device=s.device)[None, :]          # [C, BT]
    valid = pos < hi[:, None]
    x = torch.where(valid[..., None], s[pos.clamp(max=B * T - 1)], torch.zeros((), dtype=torch.float64, device=s.device))
    bad = (~torch.isfinite(x)).double()
    xz = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    if reverse:
        cs = lambda t: t.flip(1).cumsum(1).flip(1)
        depth = (BT - 1 - torch.arange(BT, device=s.device)).double()
    else:
        cs = lambda t: t.cumsum(1)
        depth = torch.arange(BT, device=s.device).double()
    S, A, nbad = cs(xz), cs(xz.abs()), cs(bad)
    depth = depth[None, :, None]
    in_range = (nbad == 0) & (A <= 2.0**126)
    if scale is None:
        sig = sig32 = 1.0
    else:
        sig, sig32 = float(scale), f32(scale)
        if not (math.isfinite(sig) and math.isfinite(sig32)):
            in_range &= False
        in_range &= abs(sig32) * A <= 2.0**126
    tol = (depth + 1) * 2.0**-23 * (abs(sig32) * A) + abs(sig - sig32) * A + 2.0**-128 * (1 + abs(sig32))
    got = out[pos.clamp(max=B * T - 1)]
    v = valid[..., None].expand_as(got)
    m = v & in_range
    err = (got - sig * S).abs()
    finite = bool(torch.isfinite(got)[m].all().item())
    with torch.no_grad():
        ratio = torch.where(tol > 0, err / torch.where(tol > 0, tol, torch.ones_like(tol)),
                            torch.where(err > 0, torch.full_like(err, math.inf), torch.zeros_like(err)))
    worst = ratio[m].max().item() if m.any() else 0.0
    return worst, finite, int((v & ~in_range).sum().item())


def post_check_sliced(g, o, BT, reverse, scale, cu, step=8):
    """`post_check` over slices of at most `step` heads (heads are independent), to stay under
    4 GB of device memory."""
    worst, finite, nout = 0.0, True, 0
    for h0 in range(0, g.shape[2], step):
        w, f, n = post_check(g[:, :, h0:h0 + step], o[:, :, h0:h0 + step], BT, reverse, scale, cu)
        worst, finite, nout = max(worst, w), finite and f, nout + n
    return worst, finite, nout


def direct(g, BT, reverse, scale, cu_t, ci_t):
    """The host's launch (lines 177-205) with `o` prefilled with a sentinel."""
    B, T, H = g.shape
    o = torch.full_like(g, SENT)
    NT = triton.cdiv(T, BT) if cu_t is None else len(ci_t)
    mod.chunk_local_cumsum_scalar_kernel[(NT * B * H,)](
        s=g, o=o, scale=scale, cu_seqlens=cu_t, chunk_indices=ci_t, T=T, B=B, H=H, BT=BT,
        REVERSE=reverse, HAS_SCALE=scale is not None, IS_VARLEN=cu_t is not None, USE_GRAPH=False,
        num_warps=2, num_stages=3)
    torch.cuda.synchronize()
    return o


def frame_ok(o, B, T, cu):
    w = torch.zeros(B * T, dtype=torch.bool, device=o.device)
    for lo, hi in seqs(B, T, cu):
        w[lo:hi] = True
    rest = o.reshape(B * T, -1)[~w]
    return bool((rest == SENT).all().item())


results = []
worst_all = 0.0


def check(name, g, BT, cu=None, idx=torch.int32, scales=(None, RCP_LN2), host=True, ci=None):
    global worst_all
    B, T, H = g.shape
    cu_t = None if cu is None else torch.tensor(cu, dtype=idx, device=DEVICE)
    g_before = g.clone()
    line = []
    ok = True
    if cu is not None:
        ci_t = mod.prepare_chunk_indices(cu_t, BT) if ci is None else torch.tensor(ci, dtype=idx, device=DEVICE)
        tok = tables_ok(cu, ci_t.cpu().tolist(), T, BT)
        ok &= tok
        line.append(f"TablesOk {tok}")
    else:
        ci_t = None
    for reverse in (False, True):
        for scale in scales:
            if host:
                o = mod.chunk_local_cumsum(g, BT, reverse=reverse, scale=scale, cu_seqlens=cu_t,
                                           chunk_indices=None if ci is None else ci_t)
                torch.cuda.synchronize()
                w, f, nout = post_check_sliced(g, o, BT, reverse, scale, cu)
                ok &= w <= 1.0 and f and o.dtype == torch.float32
                worst_all = max(worst_all, w)
            o2 = direct(g, BT, reverse, scale, cu_t, ci_t)
            w2, f2, nout2 = post_check_sliced(g, o2, BT, reverse, scale, cu)
            fr = frame_ok(o2, B, T, cu)
            ok &= w2 <= 1.0 and f2 and fr
            worst_all = max(worst_all, w2)
            line.append(f"{'rev' if reverse else 'fwd'}/{'-' if scale is None else scale}: "
                        f"{(w if host else w2):.3f} (out {nout2}) frame {fr}")
    ok &= torch.equal(g.view(torch.int32), g_before.view(torch.int32))
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} B{B} T{T} H{H} BT{BT} "
          f"{'' if cu is None else str(idx)[6:] + ' N' + str(len(cu) - 1)}: " + "; ".join(line))


gen = torch.Generator().manual_seed(4)


def gate(B, T, H):
    return torch.nn.functional.logsigmoid(torch.randn((B, T, H), generator=gen)).to(DEVICE)


# The host's test launches (source lines 271-285), int32 and int64 tables.
for B, T, H, BT, cu in [(1, 63, 1, 16, None), (2, 500, 4, 32, None), (2, 1000, 5, 64, None),
                        (3, 1024, 6, 64, None), (4, 2048, 8, 128, None),
                        (1, 15, 2, 16, [0, 15]), (1, 15, 2, 32, [0, 15]), (1, 1000, 3, 64, [0, 256, 500, 1000]),
                        (1, 2000, 4, 64, [0, 15, 100, 300, 1200, 2000]),
                        (1, 2048, 4, 128, [0, 1, 100, 300, 1200, 2048]), (1, 1, 32, 64, [0, 1]),
                        (1, 65536, 32, 64, [0, 1, 64, 65, 4096, 30000, 65536])]:
    g = gate(B, T, H)
    for idx in ((torch.int32, torch.int64) if cu is not None else (torch.int32,)):
        check("host", g, BT, cu, idx)

# Negatives: wrong outputs must fail Post (worst error / tol above 1).
g = gate(2, 1000, 5)
for reverse in (False, True):
    for scale in (None, RCP_LN2):
        o = mod.chunk_local_cumsum(g, 64, reverse=reverse, scale=scale)
        sg = 1.0 if scale is None else scale
        for what, bad in (("x(1+2^-18)", o * (1 + 2.0**-18)), ("exclusive scan", o - sg * g),
                          ("sign flip", -o)):
            w, _, _ = post_check_sliced(g, bad, 64, reverse, scale, None)
            rej = w > 1.0
            results.append(rej)
            print(f"{'ok ' if rej else 'BAD'} negative {what} {'rev' if reverse else 'fwd'}/"
                  f"{'-' if scale is None else scale}: worst error / tol {w:.3g} (rejected {rej})")

# Edges of Inst.ok and of InRange.
g = gate(1, 70001, 6)
cu = [0, 0, 1, 17, 17, 4096, 4097, 70001]  # empty sequences, lengths 1 and 16, chunk-sized ends
for BT in (16, 128):
    check("edge empty and short sequences", g, BT, cu, torch.int64)
g = gate(1, 300, 3)
check("edge sequences not covering [0, T)", g, 32, [5, 20, 20, 37, 200], torch.int32, host=False)
ci = mod.prepare_chunk_indices(torch.tensor([5, 20, 20, 37, 200]), 32).tolist()[::-1]
check("edge permuted chunk_indices", g, 32, [5, 20, 20, 37, 200], torch.int64, host=False, ci=ci)
check("edge BT 512", gate(1, 5000, 4), 512, [0, 3000, 4999, 5000], torch.int32)
check("edge BT 2048 fixed", gate(2, 4097, 2), 2048)
g = (torch.randn((1, 4000, 4), generator=gen) * 2.0**118).to(DEVICE)
check("edge near 2^126 (BT 128)", g, 128, [0, 1000, 4000], torch.int32, scales=(None, 1.0, 3.0))
g = gate(1, 600, 2)
g[0, 10, 0] = float("nan"); g[0, 300, 1] = float("inf"); g[0, 301, 1] = -float("inf")
check("edge NaN/inf in some positions", g, 64, [0, 200, 600], torch.int32)
check("edge scales", gate(1, 1000, 3), 64, [0, 999, 1000], torch.int64,
      scales=(-2.5, 0.0, -0.0, 1e-45, 1e-40, 1e30, 1e39))
check("edge 64 heads, 2^20 tokens", gate(1, 2**20, 64), 64, [0, 1, 2**19, 2**20 - 7, 2**20], torch.int64)
check("edge fixed, 64 rows", gate(64, 3000, 8), 64)

n_bad = results.count(False)
print(f"peak device memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"{len(results)} cases: worst error / tolerance {worst_all:.4f} (margin {1 / max(worst_all, 1e-300):.1f}x); "
      + ("every in-range element within tol and finite, every table TablesOk, every frame clause held"
         if n_bad == 0 else f"{n_bad} FAILED"))
sys.exit(1 if n_bad else 0)
