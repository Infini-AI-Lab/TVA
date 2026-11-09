"""GPU check of the prepared claim `Claims.ScaledMmInt8` (Claim.lean).

Runs the kernel's own host, `triton_scaled_mm` of `source/triton_scaled_mm.py` (imported by path;
its test under `__main__` does not run), on the launches SETUP.md lists and on launches at the
edges of `Inst.ok` (vLLM's rounding test, `K = 32`, is outside `Inst.ok`, `K_dim`, and checked as
an extra), and checks `Post` element by element in float64, from the host's inputs, as
`Post` states it:

* for each element `(i, j)` of `C`: `S = sum_k A[i, k] * B[k, j]` (an integer; every partial sum is
  below `2^53`, so the float64 sum is exact), `sa` the scale of row `i` (`scale_a[i]` per token,
  `scale_a[0]` per tensor), `sb` the scale of column `j`, `beta` the bias of column `j` (`0`
  without one), `R = sa * sb * S + beta`, and `E = tol` from `e1 = |S| u + eta`,
  `e2 = |sa| e1 + (|sa S| + |sa| e1) u + eta`, `e3 = |sb| e2 + (|sa sb S| + |sb| e2) u + eta`,
  `E = e3` (no bias) or `e3 + 3u (|sa sb S| + e3 + |beta|) + 4 eta` (bias), with `u = 2^-24`,
  `eta = 2^-150`; `Room32` is checked to hold; then `OutSpec`'s first clause, that the reals
  rounding to `c` in the output format meet `[R - E, R + E]` (some `v` within `E` of `R` rounds to
  `c`, up to `v`'s `float32` grid); and, with room (`|R| + E` at most the format's largest value,
  which every case here meets), that `c` is finite and
  `|c - R| <= E + (|R| + E) u_o + eta_o` (`u_o = 2^-8`, `eta_o = 2^-134` for `bfloat16`;
  `2^-11`, `2^-25` for `float16`). The float64 evaluation of `R` and `E` rounds at `2^-53`, far
  below every margin here.
* negative tests: wrong results computed from the same inputs (the scaled product rounded to the
  output format before the bias is added, on vLLM's rounding test; the last `BLOCK_SIZE_K` products
  of the reduction left out; the per-token scale of the next row; the bias left out) must each
  break `Post` on some element;
* frame: the inputs are unchanged (compared with copies); and direct launches of the source's
  `scaled_mm_kernel` with the host's launch parameters into a `C` placed inside a guarded buffer
  (padded rows, and a column-major layout) leave every byte outside the matrix `C` unchanged;
* outside the claim (informational, `K` not a multiple of the tile's `BLOCK_SIZE_K`): vLLM's test
  shapes with a partial last `K` block, where the result rests on the masked lanes reading `0`.

Every test stays below 4 GB of device memory. Prints, per case, the largest error / tolerance of
the second clause (below 1, with its margin) and the frame results.
Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "triton_scaled_mm.py")
spec = importlib.util.spec_from_file_location("triton_scaled_mm_src", SRC)
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

DEV = "cuda"
U32, ETA32 = 2.0 ** -24, 2.0 ** -150
MAX32 = (2 - 2.0 ** -23) * 2.0 ** 127
FMT = {
    torch.bfloat16: dict(u=2.0 ** -8, eta=2.0 ** -134, max=(2 - 2.0 ** -7) * 2.0 ** 127, int=torch.int16),
    torch.float16: dict(u=2.0 ** -11, eta=2.0 ** -25, max=65504.0, int=torch.int16),
}
GEN = torch.Generator(device=DEV)
GEN.manual_seed(0)
FAILS = []
WORST = [0.0, ""]


def exact_S(a, b, chunk=2048):
    """`a @ b` of int8 matrices, exactly, in float64 (column chunks to bound memory)."""
    M, N = a.shape[0], b.shape[1]
    out = torch.empty((M, N), dtype=torch.float64, device=DEV)
    ad = a.double()
    for j in range(0, N, chunk):
        out[:, j:j + chunk] = ad @ b[:, j:j + chunk].double()
    return out


def neighbours(c):
    """The float64 values of the next lower and next higher numbers of `c`'s format (16-bit)."""
    bits = c.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    neg = (bits & 0x8000) != 0
    mag = bits & 0x7FFF
    # a positive number's successor is mag + 1; for a negative one, the larger value is mag - 1
    up_bits = torch.where(neg, torch.where(mag == 0, torch.ones_like(mag), (mag - 1) | 0x8000), mag + 1)
    dn_bits = torch.where(neg, (mag + 1) | 0x8000, torch.where(mag == 0, 0x8001 * torch.ones_like(mag), mag - 1))

    def val(bb):
        return bb.to(torch.int16).view(c.dtype).double()
    return val(dn_bits), val(up_bits)


def post_rows(out, a, b, scale_a, scale_b, bias, od):
    """`Post`'s element clauses on the rows of `out` (the rows of `a` given): returns, per element,
    clause 1 and the ratio of clause 2."""
    f = FMT[od]
    M, N = out.shape
    S = exact_S(a, b)
    sa = scale_a.double().reshape(-1, 1)
    sb = scale_b.double().reshape(1, -1)
    beta = bias.double().reshape(1, -1) if bias is not None else torch.zeros((1, 1), dtype=torch.float64, device=DEV)
    assert torch.isfinite(sa).all() and torch.isfinite(sb).all() and torch.isfinite(beta).all()
    P = sa * sb * S
    R = P + beta
    e1 = S.abs() * U32 + ETA32
    e2 = sa.abs() * e1 + ((sa * S).abs() + sa.abs() * e1) * U32 + ETA32
    del e1
    e3 = sb.abs() * e2 + (P.abs() + sb.abs() * e2) * U32 + ETA32
    assert ((sb.abs() * ((sa * S).abs() + e2) <= MAX32)
            & (sa.abs() * (S.abs() + S.abs() * U32 + ETA32) <= MAX32)
            & (P.abs() + e3 + beta.abs() <= MAX32 / 2 - 2 * ETA32)).all(), "Room32 fails"
    del e2, S
    E = e3 + 3 * U32 * (P.abs() + e3 + beta.abs()) + 4 * ETA32 if bias is not None else e3
    del e3, P
    assert (R.abs() + E <= f["max"]).all(), "output room fails"
    c = out.double()
    lo, hi = neighbours(out)
    # first clause: [R - E, R + E] meets the reals rounding to c (between the midpoints)
    c1 = (R - E <= (c + hi) / 2) & (R + E >= (c + lo) / 2) & torch.isfinite(c)
    del lo, hi
    # second clause
    tol2 = E + (R.abs() + E) * f["u"] + f["eta"]
    ratio = torch.where(torch.isfinite(c), (c - R).abs() / tol2, torch.full_like(c, float("inf")))
    return c1, ratio


def check_post(name, out, a, b, scale_a, scale_b, bias, od, *, expect_fail=False, rows=None):
    """Checks `Post`'s element clauses for the host's inputs, in chunks of rows; reports the largest
    ratio of the second clause (with `expect_fail`, whether some element breaks `Post`)."""
    M, N = out.shape
    rows = rows or max(1, (1 << 22) // N)
    per_token = scale_a.numel() > 1
    c1_all, worst, broken = True, 0.0, 0
    for r0 in range(0, M, rows):
        r1 = min(M, r0 + rows)
        c1, ratio = post_rows(out[r0:r1], a[r0:r1], b, scale_a[r0:r1] if per_token else scale_a,
                              scale_b, bias, od)
        c1_all = c1_all and bool(c1.all().item())
        worst = max(worst, ratio.max().item())
        broken += (~c1).sum().item() + (ratio > 1).sum().item()
    ok = c1_all and worst <= 1.0
    if expect_fail:
        status = "rejected" if not ok else "NOT REJECTED"
        print(f"  negative {name}: {status} ({broken} element clauses broken, largest ratio {worst:.3g})")
        if ok:
            FAILS.append("negative " + name)
        return ok
    print(f"  {name}: clause 1 {'ok' if c1_all else 'FAILS'}, largest error / tolerance "
          f"{worst:.4f} (margin {1 - worst:.4f})")
    if not ok:
        FAILS.append(name)
    if worst > WORST[0]:
        WORST[0], WORST[1] = worst, name
    return ok


def inputs(M, N, K, od, per_token, per_channel, use_bias, bT, *, lo=-128, hi=128, sa_scale=1e-2, sb_scale=1e-3):
    a = torch.randint(lo, hi, (M, K), dtype=torch.int8, device=DEV, generator=GEN)
    if bT:
        b = torch.randint(lo, hi, (N, K), dtype=torch.int8, device=DEV, generator=GEN).t()
    else:
        b = torch.randint(lo, hi, (K, N), dtype=torch.int8, device=DEV, generator=GEN)
    scale_a = torch.rand((M, 1) if per_token else (1, 1), device=DEV, generator=GEN) * sa_scale
    scale_b = torch.rand((N, 1) if per_channel else (1, 1), device=DEV, generator=GEN) * sb_scale
    bias = torch.randn((N,), device=DEV, generator=GEN).to(od) if use_bias else None
    return a, b, scale_a, scale_b, bias


def run_host(name, a, b, scale_a, scale_b, od, bias, **kw):
    copies = [t.clone() for t in (a, b, scale_a, scale_b) + ((bias,) if bias is not None else ())]
    out = sm.triton_scaled_mm(a, b, scale_a, scale_b, od, bias, **kw)
    torch.cuda.synchronize()
    after = [a, b, scale_a, scale_b] + ([bias] if bias is not None else [])
    if not all(torch.equal(x, y) for x, y in zip(copies, after)):
        print(f"  {name}: INPUTS CHANGED")
        FAILS.append(name + " inputs")
    check_post(name, out, a, b, scale_a, scale_b, bias, od)
    return out


FLAGS = [  # (per_token, per_channel, use_bias, b transposed)
    (True, True, False, True),
    (False, False, True, False),
    (True, False, True, True),
    (False, True, False, False),
]


def main():
    torch.manual_seed(0)
    print("vLLM's test, the shapes whose K is a multiple of the tile's BLOCK_SIZE_K "
          "(tests/kernels/quantization/test_triton_scaled_mm.py:66-123, its draws)")
    for M, N, K in [(64, 971, 1024), (512, 20486, 1024)]:
        for od in (torch.bfloat16, torch.float16):
            for sa_s in (True, False):
                for sb_s in (True, False):
                    for ub in (True, False):
                        a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=DEV, generator=GEN)
                        b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device=DEV, generator=GEN)
                        scale_a = (torch.rand((1, 1), device=DEV, generator=GEN) if sa_s
                                   else 0.25 * torch.rand((M, 1), device=DEV, generator=GEN))
                        scale_b = (torch.rand((1, 1), device=DEV, generator=GEN) if sb_s
                                   else 0.25 * torch.rand((N, 1), device=DEV, generator=GEN))
                        bias = torch.rand((N,), device=DEV, generator=GEN).to(od) if ub else None
                        run_host(f"test M {M} N {N} K {K} {str(od)[6:]} scalar a {sa_s} scalar b {sb_s} bias {ub}",
                                 a, b, scale_a, scale_b, od, bias)

    print("vLLM's descriptor test, plain path (test_triton_scaled_mm.py:150-178)")
    for M, N, K in [(1, 4096, 4096), (64, 4096, 4096), (256, 2048, 4096)]:
        for sa_s in (True, False):
            for ub in (True, False):
                a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=DEV, generator=GEN)
                b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device=DEV, generator=GEN)
                scale_a = (torch.rand((1, 1), device=DEV, generator=GEN) if sa_s
                           else 0.25 * torch.rand((M, 1), device=DEV, generator=GEN))
                scale_b = 0.25 * torch.rand((N, 1), device=DEV, generator=GEN)
                bias = torch.rand((N,), device=DEV, generator=GEN).to(torch.bfloat16) if ub else None
                run_host(f"plain M {M} N {N} K {K} scalar a {sa_s} bias {ub}",
                         a, b, scale_a, scale_b, torch.bfloat16, bias)

    print("vLLM's rounding test with use_heuristic=False (test_triton_scaled_mm.py:126-142; "
          "outside Inst.ok: K = 32 is not a multiple of 128, checked as an extra)")
    a = torch.ones((1, 32), device=DEV).to(torch.int8)
    col = torch.tensor([64.0] * 16 + [2.0, 1.0] + [0.0] * 14, device=DEV)
    b = col[:, None].expand(32, 16).to(torch.int8).contiguous()
    scale_a = torch.ones((1, 1), device=DEV)
    scale_b = torch.full((1, 1), 2 ** -10, device=DEV)
    bias = torch.full((16,), 2 ** -8, device=DEV, dtype=torch.bfloat16)
    out = run_host("rounding M 1 N 16 K 32 tile (32, 32, 32)", a, b, scale_a, scale_b, torch.bfloat16, bias,
                   use_heuristic=False)
    assert (out.double() == 1 + 2 ** -7).all()
    # the wrong order: the scaled product rounded to bfloat16 first, then the bias added
    wrong = ((scale_a * scale_b * exact_S(a, b).float()).to(torch.bfloat16).float() + bias.float()).to(torch.bfloat16)
    check_post("bias added after rounding", wrong, a, b, scale_a, scale_b, bias, torch.bfloat16, expect_fail=True)

    print("Llama-class layers (benchmarks/kernels/weight_shapes.py, int8 GEMM benchmark batches), "
          "full int8 range")
    layers = [  # (name, K, N)
        ("Llama-3-8B qkv TP1", 4096, 6144), ("Llama-3-8B o TP1", 4096, 4096),
        ("Llama-3-8B gate_up TP1", 4096, 28672), ("Llama-3-8B down TP1", 14336, 4096),
        ("Llama-3-8B down TP8", 1792, 4096), ("Llama-3.2-1B gate_up TP1", 2048, 16384),
        ("Llama-2-70B down TP1", 28672, 8192), ("Llama-3.1-405B down TP8", 6656, 16384),
    ]
    for (lname, K, N), M, od in zip(layers, [1, 16, 64, 128, 256, 1024, 37, 4096],
                                    [torch.bfloat16, torch.float16] * 4):
        for flags in FLAGS:
            a, b, scale_a, scale_b, bias = inputs(M, N, K, od, *flags)
            run_host(f"{lname} M {M} {str(od)[6:]} flags {flags}", a, b, scale_a, scale_b, od, bias)
    print("the BLOCK_SIZE_K = 128 family (Llama-2-7B down at TP2, K = 5504) and the largest K, N")
    for lname, M, K, N in [("Llama-2-7B down TP2", 128, 5504, 4096), ("Llama-2-7B down TP2", 2000, 5504, 4096),
                           ("Llama-3.1-405B down TP1", 16, 53248, 16384),
                           ("Llama-3.1-405B gate_up TP2", 4, 16384, 53248)]:
        a, b, scale_a, scale_b, bias = inputs(M, N, K, torch.bfloat16, True, True, True, True)
        run_host(f"{lname} M {M}", a, b, scale_a, scale_b, torch.bfloat16, bias)

    print("edges of Inst.ok")
    # K = 2^17 - 256, every value -128: S = K * 2^14 = 2^31 - 2^22, the largest accumulator
    K = 2 ** 17 - 256
    a = torch.full((1, K), -128, dtype=torch.int8, device=DEV)
    b = torch.full((K, 64), -128, dtype=torch.int8, device=DEV)
    run_host("K 2^17 - 256 all -128", a, b, torch.full((1, 1), 2.0 ** -20, device=DEV),
             torch.full((1, 1), 2.0 ** -10, device=DEV), torch.bfloat16, None)
    for M, N, K in [(1, 1, 256), (33, 129, 512), (65, 8191, 256), (31, 8192, 768)]:
        for od in (torch.bfloat16, torch.float16):
            a, b, scale_a, scale_b, bias = inputs(M, N, K, od, True, True, True, False)
            run_host(f"M {M} N {N} K {K} {str(od)[6:]}", a, b, scale_a, scale_b, od, bias)
    # large and small scales; a bias that cancels the product
    a, b, scale_a, scale_b, _ = inputs(64, 512, 1024, torch.bfloat16, True, True, False, True)
    run_host("scales 2^20", a, b, scale_a * 2.0 ** 20, scale_b * 2.0 ** 10, torch.bfloat16, None)
    run_host("scales 2^-60", a, b, scale_a * 2.0 ** -60, scale_b, torch.bfloat16, None)
    S = exact_S(a, b)
    cancel = (-(scale_a.double() * scale_b.double().t() * S)[0]).to(torch.bfloat16)
    run_host("bias cancelling row 0", a, b, scale_a, scale_b, torch.bfloat16, cancel)

    print("negative tests (wrong results from the same inputs)")
    M, N, K = 128, 4096, 4096
    a, b, scale_a, scale_b, bias = inputs(M, N, K, torch.bfloat16, True, True, True, True)
    good = sm.triton_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, bias)
    check_post("reference", good, a, b, scale_a, scale_b, bias, torch.bfloat16)
    sa64, sb64 = scale_a.double(), scale_b.double().t()
    short = (sa64 * sb64 * exact_S(a[:, :K - 128], b[:K - 128]) + bias.double()).to(torch.bfloat16)
    check_post("last BLOCK_SIZE_K products left out", short, a, b, scale_a, scale_b, bias, torch.bfloat16,
               expect_fail=True)
    shifted = (torch.roll(sa64, 1, 0) * sb64 * exact_S(a, b) + bias.double()).to(torch.bfloat16)
    check_post("scale of the next row", shifted, a, b, scale_a, scale_b, bias, torch.bfloat16, expect_fail=True)
    nobias = (sa64 * sb64 * exact_S(a, b)).to(torch.bfloat16)
    check_post("bias left out", nobias, a, b, scale_a, scale_b, bias, torch.bfloat16, expect_fail=True)

    print("frame: direct launches into a guarded C (host's launch parameters)")
    for layout in ("padded rows", "column-major"):
        M, N, K = 100, 300, 512
        od = torch.bfloat16
        a, b, scale_a, scale_b, bias = inputs(M, N, K, od, True, True, True, True)
        BM, BN, BK = 64, 128, 128  # the heuristic's tile for M = 100
        guard = torch.full(((M + 3) * (N + 7) + 64,), -1234.0, dtype=od, device=DEV)
        if layout == "padded rows":
            base, scm, scn = 17, N + 7, 1
        else:
            base, scm, scn = 17, 1, M + 3
        idx = (base + torch.arange(M, device=DEV)[:, None] * scm + torch.arange(N, device=DEV)[None, :] * scn)
        before = guard.clone()
        c_view = guard[base:]
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        sm.scaled_mm_kernel[grid](a, b, scale_a, scale_b, c_view, bias, M, N, K,
                                  a.stride(0), a.stride(1), b.stride(0), b.stride(1), scm, scn,
                                  tl.int32, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                                  BLOCK_SIZE_SCALE_A=BM, BLOCK_SIZE_SCALE_B=BN, USE_TD=False, B_T=True)
        torch.cuda.synchronize()
        inside = torch.zeros_like(guard, dtype=torch.bool)
        inside[idx.reshape(-1)] = True
        frame_ok = torch.equal(guard[~inside].view(torch.int16), before[~inside].view(torch.int16))
        print(f"  {layout}: bytes outside the matrix C unchanged: {frame_ok}")
        if not frame_ok:
            FAILS.append("frame " + layout)
        check_post(f"guarded C, {layout}", guard[idx], a, b, scale_a, scale_b, bias, od)

    print("outside the claim (K not a multiple of BLOCK_SIZE_K; informational)")
    for M, N, K in [(1, 256, 128), (33, 256, 496), (512, 256, 496)]:
        a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=DEV, generator=GEN)
        b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device=DEV, generator=GEN)
        scale_a = 0.25 * torch.rand((M, 1), device=DEV, generator=GEN)
        scale_b = 0.25 * torch.rand((N, 1), device=DEV, generator=GEN)
        out = sm.triton_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)
        ref = sm.reference(a, b, scale_a, scale_b)
        print(f"  M {M} N {N} K {K}: max |C - R| / (|R| + 2^-126) = "
              f"{((out.double() - ref).abs() / (ref.abs() + 2.0 ** -126)).max().item():.3g}")

    print(f"largest error / tolerance over the launches checked: {WORST[0]:.4f} ({WORST[1]})")
    print(f"peak device memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    if FAILS:
        print("FAIL:", FAILS)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
