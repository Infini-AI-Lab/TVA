"""GPU check of the claim `Claims.MoeSumReduce`.

Runs the kernel's own host, `moe_sum_reduce` of
`source/moe_sum_reduce.py` (imported by path), on the host's test
launches and on launches at the edges of `Inst.ok`, and checks `Post` as it is stated, for every
output element `(t, j)`:

* `InRange`: the `topk_num` inputs and the scale (the double and its `float32` constant) are
  finite, `A = Σ|x| ≤ 2^126` and `|σ₃₂|·A ≤ 2^126`;
* if so, the output is finite and `|out − σ·S| ≤ tol K S A σ σ₃₂`, with `S = Σx`, `σ` the host's
  double and `tol` exactly as `Post` defines it (`accErr`, `Em`, `Q`, the `bfloat16` rounding and
  the scale's conversion), all computed here in `float64` from the host's inputs. (`S` and `A` are
  sums of at most a few hundred `bfloat16` values in `float64`; their own error, below
  `2^-45·A`, is far below the `A`-term of `tol`, `|σ₃₂|·accErr ≥ K·2^-24·|σ₃₂|·A`.)

It reports, over the elements in range, the largest error / tolerance, and counts the elements
out of range. The frame: the input is unchanged, and a second launch of the module's kernel with
the host's own arguments but the output a slice of a larger sentinel-filled buffer leaves every
element outside the `token_num·hidden_dim` outputs unchanged; the same launch is run with other
`BLOCK_M`, `BLOCK_DIM` (covered by `Inst.ok`).

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
SRC = HERE / "source" / "moe_sum_reduce.py"
spec = importlib.util.spec_from_file_location("moe_sum_reduce_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
U32, ETA32, UBF, ETABF = 2.0**-24, 2.0**-150, 2.0**-8, 2.0**-134


def sigma32(s):
    """`FloatV.litF32` of the double: round to nearest even, `+0` for a zero."""
    with np.errstate(over="ignore"):
        v = float(np.float32(s))
    return 0.0 if v == 0.0 else v


def post_check(x, out, scale):
    """Post's value clause on every element: (worst error / tol over InRange, number out of range).
    Computed in row chunks, so that no large float64 copy of the input is held at once."""
    T, K, D = x.shape
    s, s32 = float(scale), sigma32(scale)
    scale_ok = math.isfinite(s) and math.isfinite(s32)
    g = (1 + U32) ** K
    worst, ok_fin, nout = 0.0, True, 0
    rows = max(1, (2 ** 25) // max(1, K * D))          # at most 256 MiB of float64 per chunk
    for r in range(0, T, rows):
        xd = x[r:r + rows].double()
        S = xd.sum(dim=1)
        A = xd.abs().sum(dim=1)
        fin = torch.isfinite(xd).all(dim=1)
        del xd
        in_range = fin & (A <= 2.0**126) & (abs(s32) * A <= 2.0**126)
        if not scale_ok:
            in_range &= False
        E1 = (g - 1) * A + ETA32 * K * g
        Em = U32 * (abs(s32) * (S.abs() + E1)) + ETA32
        Q = abs(s32) * S.abs() + abs(s32) * E1 + Em
        tol = (UBF * Q + ETABF) + Em + abs(s32) * E1 + abs(s - s32) * S.abs()
        o = out[r:r + rows].double()
        ok_fin = ok_fin and torch.isfinite(o)[in_range].all().item()
        ratio = ((o - s * S).abs() / tol)[in_range]
        if ratio.numel():
            worst = max(worst, ratio.max().item())
        nout += int((~in_range).sum().item())
        del S, A, fin, in_range, E1, Em, Q, tol, o, ratio
    return worst, ok_fin, nout


def direct(x, scale, bm, bd, guard=4096):
    """The host's launch (lines 113-146) with `output` a slice of a guarded buffer and the given
    `BLOCK_M`, `BLOCK_DIM`."""
    T, K, D = x.shape
    SENT = torch.tensor([12345.0], dtype=torch.bfloat16)
    buf = torch.full((guard + T * D + guard,), SENT.item(), dtype=torch.bfloat16, device=x.device)
    out = buf[guard:guard + T * D].view(T, D)
    mod._moe_sum_reduce_kernel[(triton.cdiv(T, bm), triton.cdiv(D, bd))](
        x, *x.stride(), out, *out.stride(), token_num=T, topk_num=K, hidden_dim=D,
        routed_scaling_factor=scale, BLOCK_M=bm, BLOCK_DIM=bd, NUM_STAGE=1, USE_GDC=False,
        num_warps=16 if bm * bd >= 2048 else 4, num_stages=3)
    torch.cuda.synchronize()
    sent = SENT.to(DEVICE)
    frame = bool((buf[:guard] == sent).all().item()) and bool((buf[guard + T * D:] == sent).all().item())
    return buf, out, frame


results = []
worst_all = 0.0


def check(name, x, scale, blocks=((1, 2048),), host=True):
    global worst_all
    before = x.clone()
    lines = []
    ok = True
    if host:
        out = mod.moe_sum_reduce(x, scale)
        torch.cuda.synchronize()
        w, f, nout = post_check(x, out, scale)
        del out
        ok &= w <= 1.0 and f
        worst_all = max(worst_all, w)
        lines.append(f"host: worst err/tol {w:.3f}, finite {f}, out of range {nout}")
    for bm, bd in blocks:
        buf, out2, frame = direct(x, scale, bm, bd)
        w, f, nout = post_check(x, out2, scale)
        del buf, out2
        ok &= w <= 1.0 and f and frame
        worst_all = max(worst_all, w)
        lines.append(f"BLOCK ({bm},{bd}): worst {w:.3f}, frame {frame}")
    ok &= torch.equal(x.view(torch.int16), before.view(torch.int16))  # bitwise: NaN inputs
    results.append(ok)
    del before
    torch.cuda.empty_cache()
    T, K, D = x.shape
    print(f"{'ok ' if ok else 'BAD'} {name} ({T},{K},{D}) scale {scale}: " + "; ".join(lines))


gen = torch.Generator().manual_seed(3)


def randn(T, K, D):
    return torch.randn((T, K, D), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)


# The host's test launches (source lines 158-168).
for T, K, D, s in [(1, 8, 7168, 2.5), (7, 9, 4096, 0.3), (32, 8, 2048, 1.0), (33, 6, 5120, 1.0),
                   (17, 8, 2880, 1.0), (1, 1, 1, 1.0), (300, 3, 100, 2.5), (1024, 9, 4096, 0.3),
                   (8192, 8, 7168, 2.5)]:
    check("host", randn(T, K, D), s, blocks=((1, 2048), (4, 256)) if T * D < 2**24 else ((1, 2048),))

# Edges of Inst.ok and of InRange.
x = randn(64, 9, 3000)
x[:, 1::2] = -x[:, 0::2][:, :4]  # cancellation: pairs that sum to zero
check("edge cancellation", x, 2.5, blocks=((1, 2048), (2, 1024)))
x = randn(16, 2, 1000) * 2.0**124  # near the top of the range
check("edge near 2^126", x, 1.0)
check("edge near 2^126, scale 3 (some out of range)", x, 3.0)
x = (randn(16, 3, 777) * 2.0**-130)  # bfloat16 subnormals
check("edge subnormal inputs", x, 1.0)
check("edge tiny scale 1e-40 (float32 subnormal)", randn(8, 4, 2049), 1e-40)
check("edge scale 0", randn(8, 4, 513), 0.0)
check("edge scale -0.0", randn(8, 4, 513), -0.0)
check("edge negative scale", randn(8, 4, 513), -7.75)
check("edge scale 1e30", randn(8, 4, 513), 1e30)
check("edge scale 1e39 (float32 overflow: no element in range)", randn(4, 2, 100), 1e39)
x = randn(32, 5, 300)
x[0, 2, 7] = float("nan"); x[3, 0, 9] = float("inf"); x[5, 4, 11] = -float("inf")
check("edge NaN/inf in some elements", x, 2.5)
check("edge topk 1", randn(100, 1, 7168), 2.5)
check("edge topk 64", randn(16, 64, 2048), 0.5)
check("edge hidden 1", randn(1000, 9, 1), 2.5, blocks=((1, 2048), (8, 64)))
check("edge hidden 2049", randn(10, 9, 2049), 2.5, blocks=((1, 2048), (1, 4096)))
check("edge 2^15 tokens", randn(2**15, 9, 1024), 2.5)

del x
torch.cuda.empty_cache()
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"peak device memory: {peak:.2f} GiB")
if peak >= 3.5:
    results.append(False)
n_bad = results.count(False)
print(f"{len(results)} cases: worst error / tolerance {worst_all:.4f}; "
      + ("every in-range element within tol and finite, every frame clause held" if n_bad == 0
         else f"{n_bad} FAILED"))
sys.exit(1 if n_bad else 0)
