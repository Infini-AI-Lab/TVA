"""TVA library evidence (library issue L4): the accuracy of `tl.dot` into float32 on sm_90.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot/run.py
Output: evidence/num-dot/out.txt (README.md says what it shows).

Part 1 (the accumulation window). One output element of `tl.dot(a, b, acc)` with chosen
products shows how the tensor cores add: how many bits below the largest term survive, and
whether the accumulator itself is cut. Run on both instruction paths Triton picks (`mma.sync`,
`wgmma`).

Part 2 (the facts). For each operand dtype pair, block shape, `num_warps` (which picks the
instruction) and `K` that compiles, adversarial inputs are run through `tl.dot(a, b, acc)` and
every output element is checked against the fact the library states (docs/rules/num.md):

  DotF16, DotBF16  (float16 / bfloat16 operands, max_num_imprecise_acc = 0):
      |r - (acc + sum a b)| <= (K+2) 2^-22 (|acc| + sum |a b|) + (K+2) 2^-126
  DotE4m3, DotE5m2, DotE4m3E5m2, DotE5m2E4m3  (fp8 operands, max_num_imprecise_acc = 2^30,
  Triton's default on sm_90):
      |r - (acc + sum a b)| <= ((1 + 2^-7)^n - 1) (|acc| + sum |a b|) + n (1 + 2^-7)^n 2^-126,
      n = ceil(K / 32)
  DotF32Ieee  (float32 operands, input_precision="ieee"):
      |r - (acc + sum a b)| <= (K+1) 2^-23 (|acc| + sum |a b|) + (K+1) 2^-149

whenever the room condition holds (mag + rel mag + abs <= the float32 maximum); the result must
then be finite. The exact value is computed in float64 (products of these operands are exact
there); its rounding error is at most (K+1) 2^-52 mag, which the script adds to the error.

Part 3 (input_precision). Whether `input_precision` changes the result for 16-bit and fp8
operands: every allowed value against the default, bit for bit.

Part 4 (the ieee float32 dot). Whether the result is the sequential `fma` over k = 0..K-1
starting from acc, bit for bit, computed exactly with fractions.
"""
import math
import re
from fractions import Fraction
import torch
import triton
import triton.language as tl

torch.manual_seed(20261028)
DEV = "cuda"
U = 2.0 ** -24


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


DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float8e4nv": torch.float8_e4m3fn,
      "float8e5": torch.float8_e5m2, "float32": torch.float32}
CACHE = {}


def run(a, b, c, nw, prec="tf32"):
    M, K = a.shape
    N = b.shape[1]
    o = torch.empty(M, N, dtype=torch.float32, device=DEV)
    h = dotk[(1,)](a, b, c, o, M, N, K, prec, num_warps=nw)
    ptx = h.asm["ptx"]
    path = "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "fma")
    return o, path


def fact(da, db, K):
    if da == "float32":
        return "DotF32Ieee", (K + 1) * 2.0 ** -23, (K + 1) * 2.0 ** -149
    if da in ("float16", "bfloat16"):
        return ("DotF16" if da == "float16" else "DotBF16"), (K + 2) * 2.0 ** -22, (K + 2) * 2.0 ** -126
    n = (K + 31) // 32
    name = {("float8e4nv", "float8e4nv"): "DotE4m3", ("float8e5", "float8e5"): "DotE5m2",
            ("float8e4nv", "float8e5"): "DotE4m3E5m2", ("float8e5", "float8e4nv"): "DotE5m2E4m3"}[(da, db)]
    return name, (1 + 2.0 ** -7) ** n - 1, n * (1 + 2.0 ** -7) ** n * 2.0 ** -126


def maxval(name):
    return {"float16": 65504.0, "bfloat16": 3.3895313892515355e38, "float8e4nv": 448.0, "float8e5": 57344.0,
            "float32": 3.4028234663852886e38}[name]


def ones_frac(name):
    """The largest value below 2 in the format (all fraction bits set)."""
    m = {"float16": 10, "bfloat16": 7, "float8e4nv": 3, "float8e5": 2, "float32": 23}[name]
    return 2.0 - 2.0 ** -m


def gen(kind, M, N, K, da, db):
    ta, tb = DT[da], DT[db]
    sa = math.sqrt(maxval(da)) / 4; sb = math.sqrt(maxval(db)) / 4
    if kind == "normal":
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV); c = torch.randn(M, N, device=DEV)
    elif kind == "spread":
        ea = 10 if da == "float16" else (4 if da.startswith("float8") else 30)
        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-ea, ea, (M, K), device=DEV).float())
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-ea, ea, (K, N), device=DEV).float())
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-ea, ea, (M, N), device=DEV).float())
    elif kind == "accbig":
        a = torch.rand(M, K, device=DEV) + 1; b = torch.rand(K, N, device=DEV) + 1
        c = torch.exp2(torch.randint(4, 40, (M, N), device=DEV).float()) * (2 - 2.0 ** -23)
    elif kind == "allones":
        # same-sign products with all fraction bits set, against an accumulator with all bits set
        a = torch.full((M, K), ones_frac(da), device=DEV); b = torch.full((K, N), ones_frac(db), device=DEV)
        c = torch.exp2(torch.randint(0, 30, (M, N), device=DEV).float()) * (2 - 2.0 ** -23)
        c = c * torch.where(torch.rand(M, N, device=DEV) < 0.5, -1.0, 1.0)
        a = a * torch.exp2(-torch.randint(0, 3, (M, 1), device=DEV).float())
    elif kind == "onebig":
        t = torch.randint(1, 30, (M, 1), device=DEV).float()
        a = torch.full((M, K), ones_frac(da), device=DEV) * torch.exp2(-t).clamp(min=2.0 ** -14)
        a[:, 0] = ones_frac(da) * 4
        b = torch.full((K, N), ones_frac(db), device=DEV) * torch.where(torch.rand(K, N, device=DEV) < 0.1, -1.0, 1.0)
        c = torch.zeros(M, N, device=DEV)
    elif kind == "cancel":
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV)
        a[:, 1::2] = -a[:, 0::2]
        b[1::2, :] = b[0::2, :] * (1 + 2.0 ** -3)
        c = torch.zeros(M, N, device=DEV)
    elif kind == "tiny":
        small = {"float16": 2.0 ** -24, "bfloat16": 2.0 ** -130, "float8e4nv": 2.0 ** -9, "float8e5": 2.0 ** -16, "float32": 2.0 ** -140}
        a = torch.randint(-3, 4, (M, K), device=DEV).float() * small[da]
        b = torch.randint(-3, 4, (K, N), device=DEV).float() * small[db] * 2 ** 10
        c = torch.randint(-3, 4, (M, N), device=DEV).float() * 2.0 ** -140
    elif kind == "huge":
        a = (torch.rand(M, K, device=DEV) + 0.5) * sa; b = (torch.rand(K, N, device=DEV) + 0.5) * sb
        c = torch.zeros(M, N, device=DEV)
        if da in ("bfloat16", "float32"):
            a = a * 2.0 ** -40; b = b * 2.0 ** -40
    else:
        raise ValueError(kind)
    a = a.to(ta); b = b.to(tb)
    return a.contiguous(), b.contiguous(), c.contiguous()


KINDS = ["normal", "spread", "accbig", "allones", "onebig", "cancel", "tiny", "huge"]


def check(a, b, c, o, rel, ab, K):
    ad, bd, cd = a.double(), b.double(), c.double()
    exact = ad @ bd + cd
    mag = ad.abs() @ bd.abs() + cd.abs()
    room = mag + rel * mag + ab <= 3.4028234663852886e38
    err = (o.double() - exact).abs() + (K + 1) * 2.0 ** -52 * mag
    bound = rel * mag + ab
    fin = torch.isfinite(o)
    viol = int((room & fin & (err > bound)).sum()) + int((room & ~fin).sum())
    ratio = torch.where(room & fin, err / bound, torch.zeros_like(err)).max().item()
    relu = torch.where(room & fin & (mag > 0), (o.double() - exact).abs() / mag, torch.zeros_like(err)).max().item() / U
    return viol, ratio, relu, int(room.sum())


def part1():
    print("## Part 1: the accumulation window (one output element)")
    print("Row of `a`: products as listed, `b` all ones, `acc` as listed. `kept(t)`: the result minus the large "
          "term, in units of the small terms' sum (1 = every small term kept, 0 = all dropped).\n")
    print("| operands | path | K | small terms 2^-t against one term 1 (largest t kept fully) | K terms 1 against acc 2^s (largest s kept) | acc 1+2^-23, no products: result - 1 |")
    print("|---|---|---|---|---|---|")
    for da, K in [("float16", 16), ("bfloat16", 16), ("float8e4nv", 32), ("float8e5", 32)]:
        for (M, N, nw) in [(16, 16, 1), (64, 64, 4)]:
            def one(row, accv):
                a = torch.zeros(M, K, device=DEV); a[0] = torch.tensor(row, device=DEV)
                b = torch.zeros(K, N, device=DEV); b[:, 0] = 1.0
                c = torch.zeros(M, N, device=DEV); c[0, 0] = accv
                o, path = run(a.to(DT[da]).contiguous(), b.to(DT[da]).contiguous(), c, nw)
                return o[0, 0].item(), path
            tmax = {"float16": 24, "bfloat16": 30, "float8e4nv": 9, "float8e5": 16}[da]
            last_t = None
            for t in range(1, tmax + 1):
                r, path = one([1.0] + [2.0 ** -t] * (K - 1), 0.0)
                if abs((r - 1) / ((K - 1) * 2.0 ** -t) - 1) < 1e-9:
                    last_t = t
            last_s = None
            for s in range(1, 40):
                r, _ = one([1.0] * K, 2.0 ** s)
                if r - 2.0 ** s == K:
                    last_s = s
            r1, _ = one([0.0] * K, 1 + 2.0 ** -23)
            print(f"| {da} | {path} | {K} | t <= {last_t} | s <= {last_s} | {r1 - 1:.3g} |")


def part2():
    print("\n## Part 2: adversarial inputs against the facts")
    print("`ratio`: the largest (error + reference error) / bound over all elements and inputs (must be below 1); "
          "`err/u`: the largest error relative to |acc| + sum |a b|, in units of 2^-24.\n")
    print("| operands | fact | M x N, num_warps | path | K | elements in range | outside the bound | ratio | err/u (worst input kind) |")
    print("|---|---|---|---|---|---|---|---|---|")
    pairs = [("float16", "float16"), ("bfloat16", "bfloat16"), ("float8e4nv", "float8e4nv"), ("float8e5", "float8e5"),
             ("float8e4nv", "float8e5"), ("float8e5", "float8e4nv"), ("float32", "float32")]
    shapes = [(16, 16, 1), (32, 32, 2), (16, 64, 4), (64, 64, 4), (128, 64, 4), (64, 128, 8), (128, 128, 8)]
    total_viol = 0
    for da, db in pairs:
        ks = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if not da.startswith("float8") else [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        if da == "float32":
            ks = [8, 16, 32, 64, 128, 256, 512, 1024]
        prec = "ieee" if da == "float32" else "tf32"
        for (M, N, nw) in shapes:
            for K in ks:
                if M * N > 32 * 32 and K > 2048:
                    break
                name, rel, ab = fact(da, db, K)
                agg = dict(viol=0, ratio=0.0, relu=0.0, worst=None, n=0)
                path = None
                try:
                    for kind in KINDS:
                        for rep in range(2):
                            a, b, c = gen(kind, M, N, K, da, db)
                            o, path = run(a, b, c, nw, prec)
                            v, r, ru, n = check(a, b, c, o, rel, ab, K)
                            agg["viol"] += v; agg["n"] += n
                            agg["ratio"] = max(agg["ratio"], r)
                            if ru > agg["relu"]:
                                agg["relu"], agg["worst"] = ru, kind
                except triton.runtime.errors.OutOfResources as e:
                    print(f"| {da} x {db} | {name} | {M} x {N}, {nw} | rejected | {K} | (out of resources: {str(e).split(',')[1].strip()}) | | | |")
                    break
                except Exception as e:
                    print(f"| {da} x {db} | {name} | {M} x {N}, {nw} | rejected | {K} | ({type(e).__name__}) | | | |")
                    break
                total_viol += agg["viol"]
                print(f"| {da} x {db} | {name} | {M} x {N}, {nw} | {path} | {K} | {agg['n']} | {agg['viol']} | {agg['ratio']:.4f} | {agg['relu']:.1f} ({agg['worst']}) |", flush=True)
    print(f"\nTotal elements outside the bound: {total_viol}")


def part3():
    print("\n## Part 3: input_precision with 16-bit and fp8 operands")
    print("| operands | M x N, num_warps | K | value | result bits equal to the default (tf32) |")
    print("|---|---|---|---|---|")
    for da in ["float16", "bfloat16", "float8e4nv", "float8e5"]:
        for (M, N, nw) in [(16, 16, 1), (64, 64, 4)]:
            K = 64
            a, b, c = gen("spread", M, N, K, da, da)
            o0, _ = run(a, b, c, nw, "tf32")
            for p in ["ieee", "tf32x3", "bf16x3", "bf16x6"]:
                try:
                    o, _ = run(a, b, c, nw, p)
                    same = bool(torch.equal(o.view(torch.int32), o0.view(torch.int32)))
                    print(f"| {da} | {M} x {N}, {nw} | {K} | {p} | {same} |")
                except Exception as e:
                    print(f"| {da} | {M} x {N}, {nw} | {K} | {p} | rejected: {type(e).__name__} |")


def rne32(q):
    """Round the rational q to nearest even binary32 (finite, no overflow), as a Fraction."""
    if q == 0:
        return Fraction(0)
    s = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    qe = max(e, -126) - 23
    x = a / Fraction(2) ** qe
    n = x.numerator // x.denominator
    rem = x - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return s * n * Fraction(2) ** qe


def part4():
    print("\n## Part 4: the ieee float32 dot against the sequential fma, bit for bit")
    print("| M x N, num_warps | path | K | elements compared | equal to fma over k = 0..K-1 from acc |")
    print("|---|---|---|---|---|")
    for (M, N, nw) in [(16, 16, 1), (64, 64, 4), (128, 128, 8)]:
        for K in [8, 32, 64]:
            a, b, c = gen("spread", M, N, K, "float32", "float32")
            o, path = run(a, b, c, nw, "ieee")
            A = a.cpu().double().tolist(); B = b.cpu().double().tolist(); C = c.cpu().double().tolist(); O = o.cpu().double().tolist()
            eq = 0; cnt = 0
            for i in range(0, M, max(1, M // 8)):
                for j in range(0, N, max(1, N // 8)):
                    acc = Fraction(C[i][j])
                    for k in range(K):
                        acc = rne32(Fraction(A[i][k]) * Fraction(B[k][j]) + acc)
                    cnt += 1
                    eq += int(acc == Fraction(O[i][j]))
            print(f"| {M} x {N}, {nw} | {path} | {K} | {cnt} | {eq} |")


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    part1()
    part3()
    part4()
    part2()


if __name__ == "__main__":
    main()
