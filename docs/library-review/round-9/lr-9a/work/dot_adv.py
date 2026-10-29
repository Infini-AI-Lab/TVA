"""lr-9a: adversarial tl.dot inputs the evidence (evidence/num-dot) does not construct.

A. Subnormal operands against large partners, so the products are large: if the tensor cores
   flushed subnormal inputs, the error would be the whole product. (evidence/num-dot's "tiny"
   kind pairs subnormal bfloat16 values with tiny partners, so their products are below 2^-200
   and a flush would be invisible.)
B. The truncation's worst case: an accumulator 2^E exactly and every product just below the
   window's step (2^-23 of 2^E for 16-bit operands, 2^-13 for fp8 on wgmma), so every product
   is cut away. The chunk derivation says this reaches about half of the bound.
C. The A operand computed in registers (wgmma's register-operand form, as in an attention
   kernel's `tl.dot(p.to(f16), v)`), against the same dot with A loaded, bit for bit.
D. The ieee float32 dot with enable_fp_fusion=False (a launch option Compiler.Faithful admits):
   the instructions, and the fact's bound on adversarial inputs.

Each element with room is checked against the library's fact (docs/rules/num.md), as in
evidence/num-dot: err + (K+1) 2^-52 mag <= rel(K) mag + abs(K), the result finite.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I dot_adv.py
"""
import re
import torch
import triton
import triton.language as tl

torch.manual_seed(928)
DEV = "cuda"
DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float8e4nv": torch.float8_e4m3fn,
      "float8e5": torch.float8_e5m2, "float32": torch.float32}


@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    o = tl.dot(a, b, c, input_precision=PREC)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


@triton.jit
def dotk_reg(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # A is loaded as float32 and converted in registers (exact for values of the 16-bit format)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    o = tl.dot((a * 1.0).to(b.dtype), b, c)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


def path_of(ptx):
    return "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "fma")


def run(a, b, c, nw, prec="tf32", **kw):
    M, K = a.shape
    N = b.shape[1]
    o = torch.empty(M, N, dtype=torch.float32, device=DEV)
    h = dotk[(1,)](a, b, c, o, M, N, K, prec, num_warps=nw, **kw)
    return o, path_of(h.asm["ptx"]), h


def fact(da, K):
    if da == "float32":
        return (K + 1) * 2.0 ** -23, (K + 1) * 2.0 ** -149
    if da in ("float16", "bfloat16"):
        return (K + 2) * 2.0 ** -22, (K + 2) * 2.0 ** -126
    n = (K + 31) // 32
    return (1 + 2.0 ** -7) ** n - 1, n * (1 + 2.0 ** -7) ** n * 2.0 ** -126


def check(a, b, c, o, da, K):
    rel, ab = fact(da, K)
    ad, bd, cd = a.double(), b.double(), c.double()
    exact = ad @ bd + cd
    mag = ad.abs() @ bd.abs() + cd.abs()
    room = mag + rel * mag + ab <= 3.4028234663852886e38
    err = (o.double() - exact).abs() + (K + 1) * 2.0 ** -52 * mag
    bound = rel * mag + ab
    fin = torch.isfinite(o)
    viol = int((room & fin & (err > bound)).sum()) + int((room & ~fin).sum())
    ratio = torch.where(room & fin, err / bound, torch.zeros_like(err)).max().item()
    relmax = torch.where(room & fin & (mag > 0), (o.double() - exact).abs() / mag, torch.zeros_like(err)).max().item()
    return viol, ratio, relmax, int(room.sum())


SHAPES = [(16, 16, 1), (64, 64, 4)]


def sign(shape):
    return torch.where(torch.rand(shape, device=DEV) < 0.5, -1.0, 1.0)


def subnormal_inputs(da, M, N, K):
    """a subnormal (random significands, random signs), b large; c zero or tiny."""
    if da == "float16":
        a = torch.randint(1, 1024, (M, K), device=DEV).float() * 2.0 ** -24
        b = (torch.rand(K, N, device=DEV) + 1) * 2.0 ** 14
    elif da == "bfloat16":
        a = torch.randint(1, 128, (M, K), device=DEV).float() * 2.0 ** -133
        b = (torch.rand(K, N, device=DEV) + 1) * 2.0 ** 100
    elif da == "float8e4nv":
        a = torch.randint(1, 8, (M, K), device=DEV).float() * 2.0 ** -9
        b = torch.full((K, N), 448.0, device=DEV)
    else:
        a = torch.randint(1, 4, (M, K), device=DEV).float() * 2.0 ** -16
        b = torch.full((K, N), 57344.0, device=DEV)
    a = a * sign((M, K)); b = b * sign((K, N))
    c = torch.zeros(M, N, device=DEV)
    return a.to(DT[da]).contiguous(), b.to(DT[da]).contiguous(), c


def window_inputs(da, M, N, K, path):
    """acc = 2^(w+1) exactly, so the window's step is 2; every product just below 2, so every
    product is cut away if the window truncates (w = 23 for 16-bit operands and for fp8 on
    mma.sync, which upcasts to float16; w = 13 for fp8 on wgmma)."""
    w = 13 if (da.startswith("float8") and path == "wgmma") else 23
    sa, sb = {"float16": (2 - 2.0 ** -10, 1.0), "bfloat16": (2 - 2.0 ** -7, 1.0),
              "float8e4nv": (1.875, 1.0625), "float8e5": (1.5, 1.25)}[da]
    a = torch.full((M, K), sa, device=DEV)
    b = torch.full((K, N), sb, device=DEV)
    c = torch.full((M, N), 2.0 ** (w + 1), device=DEV)
    return a.to(DT[da]).contiguous(), b.to(DT[da]).contiguous(), c.contiguous()


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("## A. Subnormal operands against large partners")
    print("| operands | M x N, warps | path | K | elements in range | outside the fact | largest error / bound | largest error / mag |")
    print("|---|---|---|---|---|---|---|---|")
    for da in ["float16", "bfloat16", "float8e4nv", "float8e5"]:
        for (M, N, nw) in SHAPES:
            for K in ([16, 64, 256] if not da.startswith("float8") else [32, 64, 256]):
                agg = [0, 0.0, 0.0, 0]
                for rep in range(4):
                    a, b, c = subnormal_inputs(da, M, N, K)
                    o, path, _ = run(a, b, c, nw)
                    v, r, rm, n = check(a, b, c, o, da, K)
                    agg[0] += v; agg[1] = max(agg[1], r); agg[2] = max(agg[2], rm); agg[3] += n
                print(f"| {da} | {M} x {N}, {nw} | {path} | {K} | {agg[3]} | {agg[0]} | {agg[1]:.4f} | {agg[2]:.3g} |")
    print("\n## B. The truncation's worst case: acc = 2^(w+1) (step 2), every product just below 2")
    print("| operands | M x N, warps | path | K | product / step | elements in range | outside the fact | largest error / bound | result - acc (one element) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for da in ["float16", "bfloat16", "float8e4nv", "float8e5"]:
        for (M, N, nw) in SHAPES:
            for K in ([16, 32, 256, 1024] if not da.startswith("float8") else [32, 64, 256, 1024]):
                # compile once to learn the path
                a0, b0, c0 = window_inputs(da, M, N, K, "mma.sync")
                try:
                    _, path, _ = run(a0, b0, c0, nw)
                except Exception as e:
                    print(f"| {da} | {M} x {N}, {nw} | rejected | {K} | | | | | ({type(e).__name__}) |")
                    continue
                a, b, c = window_inputs(da, M, N, K, path)
                o, path, _ = run(a, b, c, nw)
                v, r, rm, n = check(a, b, c, o, da, K)
                w = 13 if (da.startswith("float8") and path == "wgmma") else 23
                p = (a[0, 0].double() * b[0, 0].double()).item() / 2.0
                print(f"| {da} | {M} x {N}, {nw} | {path} | {K} | {p:.4f} | {n} | {v} | {r:.4f} | {o[0, 0].item() - 2.0 ** (w + 1):.6g} (sum of products {K * p * 2:.6g}) |")
    print("\n## C. A computed in registers, against A loaded, bit for bit")
    print("| operands | M x N, warps | K | wgmma form (register A?) | bits equal to the loaded-A dot |")
    print("|---|---|---|---|---|")
    for da in ["float16", "bfloat16", "float8e4nv", "float8e5"]:
        for (M, N, nw) in [(64, 64, 4), (128, 64, 4)]:
            for K in ([16, 64] if not da.startswith("float8") else [32, 128]):
                e = 20 if not da.startswith("float8") else 4
                ad = (torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-e, e, (M, K), device=DEV).float())).to(DT[da])
                b = (torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-e, e, (K, N), device=DEV).float())).to(DT[da]).contiguous()
                c = torch.randn(M, N, device=DEV) * 2.0 ** 10
                o1, p1, _ = run(ad.contiguous(), b, c, nw)
                o2 = torch.empty_like(o1)
                h = dotk_reg[(1,)](ad.float().contiguous(), b, c, o2, M, N, K, num_warps=nw)
                ptx = h.asm["ptx"]
                forms = sorted(set(re.findall(r"wgmma\.mma_async[a-z0-9._]*", ptx)))
                regA = bool(re.search(r"wgmma\.mma_async[^\n]*\{[^}]*\}\s*,\s*\{", ptx))
                print(f"| {da} | {M} x {N}, {nw} | {K} | {', '.join(forms) or path_of(ptx)} (register A: {regA}) | {bool(torch.equal(o1.view(torch.int32), o2.view(torch.int32)))} |")
    print("\n## D. ieee float32 dot with enable_fp_fusion=False")
    print("| M x N, warps | K | enable_fp_fusion | float ops in the PTX | elements in range | outside the fact | largest error / bound |")
    print("|---|---|---|---|---|---|---|")
    for (M, N, nw) in [(16, 16, 1), (64, 64, 4)]:
        for K in [8, 64, 512]:
            for fus in [True, False]:
                agg = [0, 0.0, 0]
                ops = ""
                for kind in range(3):
                    if kind == 0:   # all fraction bits set, same sign, against a large accumulator
                        a = torch.full((M, K), 2 - 2.0 ** -23, device=DEV); b = torch.full((K, N), 1 + 2.0 ** -23 * 3, device=DEV)
                        c = torch.exp2(torch.randint(0, 20, (M, N), device=DEV).float()) * (2 - 2.0 ** -23)
                    elif kind == 1:  # products below the subnormal range
                        a = torch.randint(-3, 4, (M, K), device=DEV).float() * 2.0 ** -80
                        b = torch.randint(-3, 4, (K, N), device=DEV).float() * 2.0 ** -75
                        c = torch.randint(-3, 4, (M, N), device=DEV).float() * 2.0 ** -149
                    else:            # spread
                        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-30, 30, (M, K), device=DEV).float())
                        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-30, 30, (K, N), device=DEV).float())
                        c = torch.randn(M, N, device=DEV)
                    a = a.contiguous(); b = b.contiguous(); c = c.contiguous()
                    try:
                        o, path, h = run(a, b, c, nw, "ieee", enable_fp_fusion=fus)
                    except triton.runtime.errors.OutOfResources:
                        ops = "rejected (out of resources)"
                        break
                    v, r, rm, n = check(a, b, c, o, "float32", K)
                    agg[0] += v; agg[1] = max(agg[1], r); agg[2] += n
                    ops = ", ".join(sorted(set(re.findall(r"^\s*((?:fma|mul|add)(?:\.[a-z0-9]+)*)\s", h.asm["ptx"], re.M))))
                print(f"| {M} x {N}, {nw} | {K} | {fus} | {ops} | {agg[2]} | {agg[0]} | {agg[1]:.4f} |")


if __name__ == "__main__":
    main()
