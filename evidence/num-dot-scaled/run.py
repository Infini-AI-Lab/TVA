"""TVA library evidence (library issue L4): `tl.dot_scaled` on sm_90 (Triton's emulation).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-scaled/run.py
Output: evidence/num-dot-scaled/out.txt (README.md says what it shows).

`tl.dot_scaled(a, sa, "e4m3", b, sb, fmt, acc, fast_math=FM)` with `e8m0` scales (groups of 32
along K, uint8), `a` stored as uint8 bits [M, K], `b` stored as uint8 bits [K, N] ("e4m3") or as
packed e2m1 [K/2, N] (two elements per byte along K, the first in the low nibble). Each output
element is checked against the fact the library states (docs/rules/num.md, `DotScaledE4m3`,
`DotScaledE4m3E2m1`): when every scaled value of its row of a and column of b is 0 or has a
magnitude in [2^-126, bfloat16 max], and every scale byte is in 1..254,

  |r - (acc + sum va vb)| <= (K+2) 2^-22 (|acc| + sum |va vb|) + (K+2) 2^-126,

with room in float32; the result must then be finite. The exact values are decoded from the bits
(e4m3 through torch's float8_e4m3fn, e2m1 by its table, e8m0 as 2^(s-127)) and summed in float64;
the reference's rounding error, at most (K+1) 2^-52 mag, is added to the error. It also prints
what the emulation does with the excluded scale bytes 0x00 and 0xFF, and the PTX instructions.
"""
import re
import torch
import triton
import triton.language as tl

torch.manual_seed(20261028)
DEV = "cuda"
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def dsk(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        KB: tl.constexpr, RF: tl.constexpr, FM: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    rkb = tl.arange(0, KB)
    rs = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    sa = tl.load(sa_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b = tl.load(b_ptr + rkb[:, None] * N + rn[None, :])
    sb = tl.load(sb_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    o = tl.dot_scaled(a, sa, "e4m3", b, sb, RF, c, fast_math=FM)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


def e4m3_val(bits):
    return bits.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)


def e2m1_val(nib):
    t = torch.tensor(E2M1, dtype=torch.float64, device=DEV)
    v = t[(nib & 7).long()]
    return torch.where((nib & 8) != 0, -v, v)


def inputs(kind, M, N, K, rf):
    # element bits: avoid e4m3 NaN (0x7f, 0xff)
    a = torch.randint(0, 256, (M, K), device=DEV)
    a = torch.where((a & 0x7f) == 0x7f, a - 1, a)
    if rf == "e4m3":
        b = torch.randint(0, 256, (K, N), device=DEV)
        b = torch.where((b & 0x7f) == 0x7f, b - 1, b)
    else:
        b = torch.randint(0, 16, (K, N), device=DEV)
    if kind == "mid":
        sa = torch.randint(110, 145, (M, K // 32), device=DEV); sb = torch.randint(110, 145, (N, K // 32), device=DEV)
    elif kind == "wide":
        sa = torch.randint(10, 230, (M, K // 32), device=DEV); sb = torch.randint(10, 230, (N, K // 32), device=DEV)
    elif kind == "low":   # near the bottom of bfloat16's normal range
        sa = torch.randint(8, 30, (M, K // 32), device=DEV); sb = torch.randint(100, 140, (N, K // 32), device=DEV)
    elif kind == "high":  # products near the top of float32's range
        sa = torch.randint(160, 185, (M, K // 32), device=DEV); sb = torch.randint(160, 185, (N, K // 32), device=DEV)
    elif kind == "accbig":
        sa = torch.randint(120, 135, (M, K // 32), device=DEV); sb = torch.randint(120, 135, (N, K // 32), device=DEV)
    elif kind == "allones":  # same-sign elements with every fraction bit set, one scale per row/column
        a = torch.full((M, K), 0x3f, device=DEV)            # e4m3 1.875
        b = torch.full((K, N), 0x3f if rf == "e4m3" else 0x7, device=DEV)   # 1.875 / e2m1 6.0
        sa = torch.randint(120, 135, (M, 1), device=DEV).repeat(1, K // 32)
        sb = torch.randint(120, 135, (N, 1), device=DEV).repeat(1, K // 32)
    c = torch.randn(M, N, device=DEV)
    if kind in ("accbig", "allones"):
        c = torch.exp2(torch.randint(0, 40, (M, N), device=DEV).float()) * (2 - 2.0 ** -23)
    return a, b, sa, sb, c


def run(a, b, sa, sb, c, rf, fm, nw):
    M, K = a.shape
    N = b.shape[1]
    if rf == "e2m1":
        packed = (b[0::2] | (b[1::2] << 4)).to(torch.uint8)
        KB = K // 2
    else:
        packed = b.to(torch.uint8)
        KB = K
    o = torch.empty(M, N, dtype=torch.float32, device=DEV)
    h = dsk[(1,)](a.to(torch.uint8).contiguous(), sa.to(torch.uint8).contiguous(), packed.contiguous(),
                  sb.to(torch.uint8).contiguous(), c, o, M, N, K, KB, rf, fm, num_warps=nw)
    return o, h


def check(a, b, sa, sb, c, o, rf, K):
    va = e4m3_val(a) * torch.exp2(sa.double() - 127).repeat_interleave(32, dim=1)
    vb_el = e4m3_val(b) if rf == "e4m3" else e2m1_val(b)
    vb = vb_el * torch.exp2(sb.double() - 127).repeat_interleave(32, dim=1).T
    bf16max = 3.3895313892515355e38
    ina = (va == 0) | ((va.abs() >= 2.0 ** -126) & (va.abs() <= bf16max))
    inb = (vb == 0) | ((vb.abs() >= 2.0 ** -126) & (vb.abs() <= bf16max))
    rowok = ina.all(dim=1)[:, None] & inb.all(dim=0)[None, :]
    exact = va @ vb + c.double()
    mag = va.abs() @ vb.abs() + c.double().abs()
    rel = (K + 2) * 2.0 ** -22
    ab = (K + 2) * 2.0 ** -126
    room = rowok & (mag + rel * mag + ab <= 3.4028234663852886e38)
    err = (o.double() - exact).abs() + (K + 1) * 2.0 ** -52 * mag
    bound = rel * mag + ab
    fin = torch.isfinite(o)
    viol = int((room & fin & (err > bound)).sum()) + int((room & ~fin).sum())
    ratio = torch.where(room & fin, err / bound, torch.zeros_like(err)).max().item()
    return viol, ratio, int(room.sum()), int(rowok.numel())


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| rhs format | fast_math | M x N, num_warps | K | elements checked / all | outside the bound | largest error / bound |")
    print("|---|---|---|---|---|---|---|")
    total = 0
    for rf in ["e4m3", "e2m1"]:
        for fm in [False, True]:
            for (M, N, nw) in [(16, 16, 4), (64, 64, 4), (128, 128, 8), (128, 64, 4)]:
                for K in [32, 64, 128, 256, 512]:
                    agg = [0, 0.0, 0, 0]
                    try:
                        for kind in ["mid", "wide", "low", "high", "accbig", "allones"]:
                            for rep in range(2):
                                a, b, sa, sb, c = inputs(kind, M, N, K, rf)
                                o, h = run(a, b, sa, sb, c, rf, fm, nw)
                                v, r, n, tot = check(a, b, sa, sb, c, o, rf, K)
                                agg[0] += v; agg[1] = max(agg[1], r); agg[2] += n; agg[3] += tot
                    except Exception as e:
                        print(f"| {rf} | {fm} | {M} x {N}, {nw} | {K} | rejected ({type(e).__name__}) | | |")
                        break
                    total += agg[0]
                    print(f"| {rf} | {fm} | {M} x {N}, {nw} | {K} | {agg[2]} / {agg[3]} | {agg[0]} | {agg[1]:.4f} |", flush=True)
    print(f"\nTotal elements outside the bound: {total}")
    # the excluded scale bytes
    print("\n## Scale bytes outside 1..254 (one element: a = 1.0 (0x38), b = 1.0, other elements 0, acc 0)")
    print("| scale of a | fast_math | result |")
    print("|---|---|---|")
    M = N = 64; K = 32
    for s in [0, 1, 127, 254, 255]:
        for fm in [False, True]:
            a = torch.zeros(M, K, dtype=torch.int64, device=DEV); a[0, 0] = 0x38
            b = torch.zeros(K, N, dtype=torch.int64, device=DEV); b[0, 0] = 0x38
            sa = torch.full((M, 1), 127, device=DEV); sa[0, 0] = s
            sb = torch.full((N, 1), 127, device=DEV)
            c = torch.zeros(M, N, device=DEV)
            o, h = run(a, b, sa, sb, c, "e4m3", fm, 4)
            print(f"| 0x{s:02x} (OCP value {'NaN' if s == 255 else repr(2.0 ** (s - 127))}) | {fm} | {o[0, 0].item()!r} |")
    ptx = h.asm["ptx"]
    ops = sorted(set(m for m in re.findall(r"\b(wgmma[a-z0-9._]*|mma\.[a-z0-9._]+|mul\.[a-z0-9.]*bf16[a-z0-9.]*|fma\.[a-z0-9.]*bf16[a-z0-9.]*|cvt\.[a-z0-9.]*bf16[a-z0-9.]*)", ptx)))
    print("\nPTX instructions of the product (64 x 64, e4m3 x e4m3):", ", ".join(f"`{o}`" for o in ops))


if __name__ == "__main__":
    main()
