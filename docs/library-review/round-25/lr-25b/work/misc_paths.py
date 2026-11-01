"""lr-25b: code-generation options the dot evidence does not vary, for the 16-bit and fp8 tl.dot facts.

  * input_precision on fp8 operands (rules/num.md says "any precision" for the chunk facts; the 16-bit
    case is evidence/num-dot-sub16-round25 Part G1): every value against the default, bit for bit;
  * a tutorial-03 K loop with `tl.range(..., warp_specialize=True)` and with num_ctas = 2 (a cluster),
    against the default loop, bit for bit, for float16, bfloat16 and the fp8 pairs, on inputs with
    subnormal operands that set the window.

Run from this folder:  CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I misc_paths.py > misc_paths.out
"""
import time

import torch
import triton
import triton.language as tl

torch.manual_seed(25125)


@triton.jit
def dotp(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c, input_precision=PREC))


@triton.jit
def kloop(a_ptr, b_ptr, c_ptr, o_ptr, KT, M: tl.constexpr, N: tl.constexpr, BK: tl.constexpr,
          WS: tl.constexpr):
    pid = tl.program_id(0)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BK)
    acc = tl.load(c_ptr + pid * M * N + rm[:, None] * N + rn[None, :])
    for kk in tl.range(0, KT, BK, warp_specialize=WS):
        a = tl.load(a_ptr + pid * M * KT + rm[:, None] * KT + (kk + rk)[None, :])
        b = tl.load(b_ptr + pid * KT * N + (kk + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(o_ptr + pid * M * N + rm[:, None] * N + rn[None, :], acc)


def operands(dt, shape_a, shape_b):
    """Values with subnormal operands against large partners, and products near the window's unit."""
    fi = {torch.float16: (2.0 ** -24, 2.0 ** 14), torch.bfloat16: (2.0 ** -133, 2.0 ** 100),
          torch.float8_e5m2: (2.0 ** -16, 2.0 ** 14), torch.float8_e4m3fn: (2.0 ** -9, 2.0 ** 8)}[dt]
    a = torch.randn(shape_a) * torch.exp2(torch.randint(-8, 4, shape_a).float())
    b = torch.randn(shape_b) * torch.exp2(torch.randint(-8, 4, shape_b).float())
    ma = torch.rand(shape_a) < 0.1
    a[ma] = fi[0] * torch.randint(1, 4, (int(ma.sum()),)).float() * torch.sign(torch.randn(int(ma.sum())))
    mb = torch.rand(shape_b) < 0.05
    b[mb] = fi[1] * torch.sign(torch.randn(int(mb.sum())))
    return a.to(dt), b.to(dt)


def path(h):
    ptx = h.asm["ptx"]
    return "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "other")


def main():
    t0 = time.time()
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("input_precision on fp8 operands, against the default (bit for bit)\n")
    print("| operands | M x N x K, warps | path | precision | equal elements |")
    print("|---|---|---|---|---|")
    pairs = [(torch.float8_e4m3fn, torch.float8_e4m3fn), (torch.float8_e5m2, torch.float8_e5m2),
             (torch.float8_e4m3fn, torch.float8_e5m2), (torch.float8_e5m2, torch.float8_e4m3fn)]
    for da, db in pairs:
        for (M, N, K, nw) in [(64, 64, 128, 4), (32, 32, 64, 2)]:
            a, _ = operands(da, (M, K), (K, N))
            _, b = operands(db, (M, K), (K, N))
            a, b = a.cuda(), b.cuda()
            c = (torch.randn(M, N) * 0.01).cuda()
            ref = torch.empty(M, N, device="cuda")
            h = dotp[(1,)](a, b, c, ref, M, N, K, "tf32", num_warps=nw)
            for prec in ("tf32x3", "ieee", "bf16x3", "bf16x6"):
                o = torch.empty(M, N, device="cuda")
                try:
                    h2 = dotp[(1,)](a, b, c, o, M, N, K, prec, num_warps=nw)
                    eq = int((o.view(torch.int32) == ref.view(torch.int32)).sum())
                    res = f"{eq}/{M * N}"
                    p2 = path(h2)
                except Exception as e:  # a launch rejected on the host
                    res, p2 = "rejected: " + type(e).__name__, "-"
                print(f"| {da} x {db} | {M} x {N} x {K}, {nw} | {path(h)} / {p2} | {prec} | {res} |", flush=True)

    print("\nK loop with warp_specialize and with num_ctas = 2, against the default loop (bit for bit)\n")
    print("| operands | M x N, BK, K, warps | variant | path | equal elements |")
    print("|---|---|---|---|---|")
    dts = [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)] + pairs
    for da, db in dts:
        for (M, N, BK, nw) in [(128, 128, 64, 4), (128, 64, 64, 8), (64, 128, 128, 4)]:
            KT, G = 1024, 4
            a, _ = operands(da, (G, M, KT), (G, KT, N))
            _, b = operands(db, (G, M, KT), (G, KT, N))
            a, b = a.cuda().contiguous(), b.cuda().contiguous()
            c = (torch.randn(G, M, N) * 0.01).cuda()
            ref = torch.empty(G, M, N, device="cuda")
            h = kloop[(G,)](a, b, c, ref, KT, M, N, BK, False, num_warps=nw)
            for variant in ("warp_specialize", "num_ctas=2"):
                o = torch.empty(G, M, N, device="cuda")
                try:
                    if variant == "warp_specialize":
                        h2 = kloop[(G,)](a, b, c, o, KT, M, N, BK, True, num_warps=nw)
                    else:
                        h2 = kloop[(G,)](a, b, c, o, KT, M, N, BK, False, num_warps=nw, num_ctas=2)
                    eq = int((o.view(torch.int32) == ref.view(torch.int32)).sum())
                    res = f"{eq}/{G * M * N}"
                    p2 = path(h2)
                except Exception as e:
                    res, p2 = "rejected: " + type(e).__name__, "-"
                print(f"| {da} x {db} | {M} x {N}, {BK}, {KT}, {nw} | {variant} | {path(h)} / {p2} | {res} |",
                      flush=True)
    print(f"\n(run time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
