"""GPU check of the claim `Claims.MeanPoolingFwd` (Claim.lean).

Runs the kernel's own entry point, `mean_pooling` of `source/pooling.py` (imported by path), on
the host's launches (fla's pooling test and NSA test shapes, fixed-length and variable-length, the
latter with `int32` and with `int64` tables, all in `bfloat16`) and on launches at the edges of
`Inst.ok`, and checks `Post` as it is stated, for every output row `g`, head `h` and feature `j`:

* the row's positions `[start g, stop g)` from `cu_seqlens` and row `g` of `chunk_indices`
  (variable length) or from `g = b·cdiv(T, BT) + c` (fixed length), `L = stop − start`;
* `InRange`: the row's terms finite and `A = Σ|x| ≤ 2^125`;
* if so, the output is finite and `|o[g, h, j] − μ| ≤ tol = 2^-8·|μ| + (log₂ BT + 5)·2^-24·A/L
  + 2^-132`, `μ = S/L`, `S = Σ x`, computed in `float64` from the host's inputs (each chunk summed
  on its own: at most `BT` terms, an error below `2^-40·A`, far below `tol`'s `2^-24·A/L` term).

It reports the largest error / tolerance over the elements in range and counts those out of
range, and checks that plausible wrong outputs fail `Post`: the sum divided by `BT` in place of `L`,
the output times `1 + 2^-6`, and chunks shifted by one position. Also checked: the host's tables
satisfy `TablesOk` (`Pre`); the inputs are unchanged; and a direct launch of the module's kernel
with the host's arguments (and other `BD` of the autotune list) into `o` placed inside a larger
buffer prefilled with a sentinel leaves every byte outside `o` unchanged, on tables whose
sequences do not cover `[0, T)`, on a permuted `chunk_indices` and on one with repeated rows.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import pathlib

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "pooling.py"
spec = importlib.util.spec_from_file_location("pooling_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
SENT = 1.5e38  # a bfloat16 sentinel
PAD = 4096     # sentinel elements on each side of `o` in a direct launch


def rows_of(B, T, BT, cu, ci):
    """`(start, stop)` of every output row, as `start`, `stop` define them."""
    if cu is None:
        nt = (T + BT - 1) // BT
        return [(b * T + c * BT, min(b * T + c * BT + BT, b * T + T)) for b in range(B) for c in range(nt)]
    return [(cu[n] + c * BT, min(cu[n] + c * BT + BT, cu[n + 1])) for n, c in ci]


def tables_ok(cu, ci, T, BT):
    """`TablesOk`, on the host's tables."""
    N = len(cu) - 1
    if not (cu[0] >= 0 and all(cu[i] <= cu[i + 1] for i in range(N)) and cu[N] <= T):
        return False
    return all(0 <= n < N and c >= 0 and c * BT < cu[n + 1] - cu[n] for n, c in ci)


def post_check(x, o, BT, rows, shift=0):
    """Post's value clause on `o` (`[B, NT, H, D]` or `[1, NT, H, D]`); returns (worst err/tol over
    InRange, all finite there, # out of range). `shift` moves every chunk (a negative test)."""
    B, T, H, D = x.shape
    G = len(rows)
    if G == 0:
        return 0.0, True, 0
    xs = x.reshape(B * T, H, D)
    out = o.reshape(G, H, D)
    e = int(math.log2(BT))
    worst, finite, nout = 0.0, True, 0
    step = max(1, 2**23 // (BT * H * D))  # rows per slice: at most 2^23 gathered elements
    for g0 in range(0, G, step):
        rs = rows[g0:g0 + step]
        st = torch.tensor([a + shift for a, _ in rs], device=x.device)
        L = torch.tensor([b - a for a, b in rs], device=x.device)
        pos = st[:, None] + torch.arange(BT, device=x.device)[None, :]          # [G', BT]
        valid = torch.arange(BT, device=x.device)[None, :] < L[:, None]
        valid &= (pos >= 0) & (pos < B * T)
        v = xs[pos.clamp(0, B * T - 1)].double()                                 # [G', BT, H, D]
        zero = torch.zeros((), dtype=torch.float64, device=x.device)
        v = torch.where(valid[..., None, None], v, zero)
        bad = (~torch.isfinite(v)).any(1)
        vz = torch.where(torch.isfinite(v), v, zero)
        S, A = vz.sum(1), vz.abs().sum(1)
        Lf = L.double()[:, None, None]
        mu, Abar = S / Lf, A / Lf
        in_range = (~bad) & (A <= 2.0**125)
        tol = 2.0**-8 * mu.abs() + (e + 5) * 2.0**-24 * Abar + 2.0**-132
        got = out[g0:g0 + len(rs)].double()
        err = (got - mu).abs()
        finite &= bool(torch.isfinite(got)[in_range].all().item())
        ratio = torch.where(torch.isfinite(err), err / tol, torch.full_like(err, math.inf))
        if in_range.any():
            worst = max(worst, ratio[in_range].max().item())
        nout += int((~in_range).sum().item())
    return worst, finite, nout


def direct(x, BT, cu_t, ci_t, BD=64, num_warps=2):
    """The host's launch (lines 170-186) of the module's kernel, with `o` a view inside a larger
    sentinel-filled buffer; returns `(o, frame_ok)`."""
    B, T, H, D = x.shape
    NT = triton.cdiv(T, BT) if cu_t is None else len(ci_t)
    n = B * NT * H * D
    big = torch.full((n + 2 * PAD,), SENT, dtype=x.dtype, device=x.device)
    o = big[PAD:PAD + n].view(B, NT, H, D)
    grid = (triton.cdiv(D, BD) * NT, B * H)
    mod.mean_pooling_fwd_kernel[grid](x, o, cu_t, ci_t, T=T, H=H, D=D, BT=BT, BD=BD,
                                      IS_VARLEN=cu_t is not None, num_warps=num_warps, num_stages=3)
    torch.cuda.synchronize()
    fr = bool((big[:PAD] == SENT).all().item() and (big[PAD + n:] == SENT).all().item())
    return o, fr


results = []
worst_all = 0.0


def check(name, x, BT, cu=None, idx=torch.int32, host=True, ci=None, bds=((64, 2),)):
    global worst_all
    B, T, H, D = x.shape
    cu_t = None if cu is None else torch.tensor(cu, dtype=idx, device=DEVICE)
    x_before = x.clone()
    ok = True
    line = []
    if cu is not None:
        ci_t = mod.prepare_chunk_indices(cu_t, BT) if ci is None else torch.tensor(ci, dtype=idx, device=DEVICE).reshape(-1, 2)
        ci_l = [tuple(r) for r in ci_t.cpu().tolist()]
        tok = tables_ok(cu, ci_l, T, BT)
        ok &= tok
        line.append(f"TablesOk {tok}")
    else:
        ci_t, ci_l = None, None
    rows = rows_of(B, T, BT, cu, ci_l)
    if host:
        o = mod.mean_pooling(x, BT, cu_seqlens=cu_t) if ci is None else mod.mean_pooling_fwd(x, BT, cu_t, ci_t)
        torch.cuda.synchronize()
        w, f, nout = post_check(x, o, BT, rows)
        ok &= w <= 1.0 and f and o.dtype == torch.bfloat16 and o.shape[1] * o.shape[0] == len(rows)
        worst_all = max(worst_all, w)
        line.append(f"host {w:.3f} (out of range {nout})")
    for BD, nw in bds:
        o2, fr = direct(x, BT, cu_t, ci_t, BD, nw)
        w2, f2, nout2 = post_check(x, o2, BT, rows)
        ok &= w2 <= 1.0 and f2 and fr
        worst_all = max(worst_all, w2)
        line.append(f"direct BD{BD}/w{nw} {w2:.3f} frame {fr}")
    ok &= torch.equal(x.view(torch.int16), x_before.view(torch.int16))
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} B{B} T{T} H{H} D{D} BT{BT} "
          f"{'' if cu is None else str(idx)[6:] + ' N' + str(len(cu) - 1)}: " + "; ".join(line))


gen = torch.Generator().manual_seed(4)


def inp(B, T, H, D, scale=1.0):
    return (torch.randn((B, T, H, D), generator=gen) * scale).to(DEVICE, torch.bfloat16)


# The host's launches (SETUP.md): fla's pooling test, then NSA's tests, then a packed batch.
for B, T, H, D, BT in [(1, 63, 1, 30, 16), (2, 500, 4, 60, 32), (2, 1000, 5, 128, 64),
                       (3, 1024, 6, 500, 64), (4, 2048, 8, 1024, 128),
                       (1, 63, 1, 64, 32), (3, 111, 1, 100, 32), (3, 1024, 2, 60, 32),
                       (3, 1024, 2, 128, 32), (4, 2048, 2, 64, 32)]:
    check("host fixed", inp(B, T, H, D), BT)
for H, D, BT, cu in [(2, 60, 32, [0, 15]), (3, 100, 64, [0, 256, 500, 1000]),
                     (4, 256, 64, [0, 15, 100, 300, 1200, 2000]), (4, 500, 128, [0, 1, 100, 300, 1200, 2048]),
                     (2, 1024, 128, [0, 200, 512, 1200, 2048]),
                     (1, 64, 32, [0, 15]), (1, 64, 16, [0, 15, 205, 550, 800]), (2, 64, 32, [0, 256, 500, 1000]),
                     (2, 100, 32, [0, 15, 100, 300, 1200, 2000]), (4, 128, 64, [0, 1, 64, 65, 4096, 30000, 65536])]:
    x = inp(1, cu[-1], H, D)
    for idx in (torch.int32, torch.int64):
        check("host varlen", x, BT, cu, idx)

# Negatives: wrong outputs must fail Post (worst error / tol above 1).
x = inp(1, 1000, 3, 100)
cu = [0, 256, 500, 1000]
cu_t = torch.tensor(cu, dtype=torch.int64, device=DEVICE)
o = mod.mean_pooling(x, 64, cu_seqlens=cu_t)
rows = rows_of(1, 1000, 64, cu, [tuple(r) for r in mod.prepare_chunk_indices(cu_t, 64).tolist()])
sums = torch.stack([x[0, a:b].double().sum(0) for a, b in rows])[None]
for what, bad, shift in (("sum / BT for sum / L", (sums / 64).to(torch.bfloat16), 0),
                         ("x(1+2^-6)", (o.double() * (1 + 2.0**-6)).to(torch.bfloat16), 0),
                         ("chunks shifted by one position", o, 1)):
    w, _, _ = post_check(x, bad, 64, rows, shift)
    rej = w > 1.0
    results.append(rej)
    print(f"{'ok ' if rej else 'BAD'} negative {what}: worst error / tol {w:.3g} (rejected {rej})")

# Edges of Inst.ok and of InRange.
x = inp(1, 70001, 2, 64)
cu = [0, 0, 1, 17, 17, 4096, 4097, 70001]  # empty sequences, lengths 1 and 16, chunk-sized ends
for BT in (1, 16, 128):
    check("edge empty and short sequences", x, BT, cu, torch.int64, bds=((64, 2), (16, 1), (128, 8)))
x = inp(1, 300, 3, 40)
check("edge sequences not covering [0, T)", x, 32, [5, 20, 20, 37, 200], torch.int32, host=False)
ci = mod.prepare_chunk_indices(torch.tensor([5, 20, 20, 37, 200]), 32).tolist()[::-1]
check("edge permuted chunk_indices", x, 32, [5, 20, 20, 37, 200], torch.int64, ci=ci)
check("edge repeated chunk_indices rows", x, 32, [5, 20, 20, 37, 200], torch.int32, ci=ci + ci[:3])
check("edge all sequences empty (NT = 0)", x, 32, [0, 0, 0], torch.int32)
check("edge T = 0 (empty grid)", inp(2, 0, 3, 40), 16)
check("edge BT 4096, D 1 (BD 16)", inp(1, 20000, 2, 1), 4096, [0, 4096, 4097, 20000], torch.int32,
      host=False, bds=((16, 4),))
check("edge BT 1024 fixed", inp(2, 3000, 2, 200), 1024, bds=((64, 2), (128, 4), (16, 2), (32, 1)))
check("edge near 2^125 (BT 64)", inp(1, 4000, 2, 64, 2.0**117), 64, [0, 1000, 4000], torch.int32)
check("edge near 2^125 (BT 1)", inp(1, 64, 2, 64, 2.0**121), 1)
check("edge subnormal inputs", inp(1, 2000, 2, 64, 2.0**-130), 64, [0, 1, 999, 2000], torch.int64)
x = inp(1, 600, 2, 64)
x[0, 10, 0, 3] = float("nan"); x[0, 300, 1, 5] = float("inf"); x[0, 301, 1, 6] = -float("inf")
check("edge NaN/inf in some positions", x, 64, [0, 200, 600], torch.int32)
check("edge 2^22 tokens", inp(1, 2**22, 1, 64), 64, [0, 1, 2**21, 2**22 - 7, 2**22], torch.int64)
check("edge fixed, 8192 rows", inp(8192, 64, 4, 64), 64)
check("edge D 2^17", inp(1, 100, 1, 2**17), 64, [0, 37, 100], torch.int32)

n_bad = results.count(False)
print(f"peak device memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"{len(results)} cases: worst error / tolerance {worst_all:.4f} (margin {1 / max(worst_all, 1e-300):.1f}x); "
      + ("every in-range element within tol and finite, every table TablesOk, every frame clause held"
         if n_bad == 0 else f"{n_bad} FAILED"))
sys.exit(1 if n_bad else 0)
