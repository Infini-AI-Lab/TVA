"""TVA library evidence (library issue L4): `/` on float32 (`div.full.f32`), the fact `DivFull`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-div-f32/run.py
Output: evidence/num-div-f32/out.txt (README.md says what it shows).

`DivFull`: for finite x, y with y != 0 and |x/y| <= 2^126, the result of `x / y` is finite and
within 2^-22 |x/y| + 2^-148 of x/y. The reference is the float64 quotient on the GPU (IEEE
division, within 2^-53 |x/y| of the real quotient, never subnormal for float32 operands); the
script checks err (1 + 2^-52) + 2^-52 |q| <= bound.

Inputs, about 3.1 * 10^10 pairs:
  A. every divisor mantissa (2^23) at each divisor exponent of a list, against 32 dividends whose
     exponents put the quotient in each binade of interest (subnormal, near 2^-126, 1, 2^100,
     2^125, just below 2^126 and at it), with mantissas 1, 2 - 2^-23, 1.5 and random ones;
  B. subnormal divisors (all 2^23 patterns) and subnormal dividends;
  C. 2^31 random pairs of finite bit patterns, kept when |x/y| <= 2^126.
It also prints the quotient's error in ulps of its binade, and the PTX instruction.
"""
import re
import torch
import triton
import triton.language as tl

torch.manual_seed(20261028)


@triton.jit
def div(x_ptr, y_ptr, o_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x / y, mask=m)


class Stats:
    def __init__(self):
        self.pairs = 0; self.viol = 0; self.nonfin = 0; self.ratio = 0.0; self.at = None; self.ulp = 0.0; self.ulpat = None

    def add(self, x, y):
        o = torch.empty_like(x)
        div[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), B=1024)
        xd, yd, od = x.double(), y.double(), o.double()
        q = xd / yd
        keep = torch.isfinite(xd) & torch.isfinite(yd) & (yd != 0) & (q.abs() <= 2.0 ** 126)
        aq = q.abs()
        err = (od - q).abs()
        lhs = err * (1 + 2.0 ** -52) + 2.0 ** -52 * aq
        bound = 2.0 ** -22 * aq + 2.0 ** -148
        fin = torch.isfinite(od)
        self.pairs += int(keep.sum())
        self.nonfin += int((keep & ~fin).sum())
        self.viol += int((keep & fin & (lhs > bound)).sum())
        r = torch.where(keep & fin, lhs / bound, torch.zeros_like(lhs))
        i = int(r.argmax())
        if r[i].item() > self.ratio:
            self.ratio = r[i].item(); self.at = (x[i].item(), y[i].item())
        # error in ulps of the quotient's binade (normal quotients)
        nq = keep & fin & (aq >= 2.0 ** -126)
        e = torch.floor(torch.log2(torch.where(nq, aq, torch.ones_like(aq))))
        u = torch.where(nq, err / torch.exp2(e - 23), torch.zeros_like(err))
        j = int(u.argmax())
        if u[j].item() > self.ulp:
            self.ulp = u[j].item(); self.ulpat = (x[j].item(), y[j].item())


def f32(sign, exp, mant):
    """float32 from fields (tensors or ints), as a float32 tensor."""
    b = (torch.as_tensor(sign, dtype=torch.int64) << 31) | (torch.as_tensor(exp, dtype=torch.int64) << 23) | torch.as_tensor(mant, dtype=torch.int64)
    b = torch.where(b >= 2 ** 31, b - 2 ** 32, b).to(torch.int32)
    return b.view(torch.float32)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    dev = "cuda"
    allm = torch.arange(1 << 23, dtype=torch.int64, device=dev)
    mants = [0, (1 << 23) - 1, 1 << 22] + torch.randint(0, 1 << 23, (29,)).tolist()
    S = {k: Stats() for k in "ABC"}
    # A: divisor exponent ey (biased), dividend exponent chosen so that the quotient's exponent is qe
    for ey in [1, 2, 50, 127, 200, 253, 254]:
        for qe in [-149, -140, -127, -126, -125, -60, 0, 1, 64, 100, 124, 125, 126]:
            ex = qe + ey
            if not (1 <= ex <= 254):
                continue
            ys = f32(0, ey, allm).to(dev)
            for mx in mants:
                for sx in (0, 1):
                    xs = f32(sx, ex, torch.full_like(allm, mx)).to(dev)
                    S["A"].add(xs.contiguous(), ys.contiguous())
    # B: subnormal divisors and dividends
    ysub = f32(0, 0, allm[1:]).to(dev)
    for ex in [0, 1, 2, 10, 100, 120, 126, 127, 140, 150]:
        for mx in mants[:8]:
            xs = f32(0, ex, torch.full_like(allm[1:], mx)).to(dev)
            S["B"].add(xs.contiguous(), ysub.contiguous())
    xsub = f32(0, 0, allm[1:]).to(dev)
    for ey in [1, 2, 10, 100, 127, 140, 200, 254]:
        for my in mants[:8]:
            ys = f32(0, ey, torch.full_like(allm[1:], my)).to(dev)
            S["B"].add(xsub.contiguous(), ys.contiguous())
    # C: random finite pairs
    for _ in range(32):
        n = 1 << 26
        xb = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=dev).to(torch.int32)
        yb = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=dev).to(torch.int32)
        S["C"].add(xb.view(torch.float32), yb.view(torch.float32))
    print("\n| inputs | pairs with abs(x/y) <= 2^126 | outside the bound | non-finite | largest error / bound (x, y) | largest error in ulps of the quotient's binade (x, y) |")
    print("|---|---|---|---|---|---|")
    for k, name in [("A", "A: every divisor mantissa"), ("B", "B: subnormal operands"), ("C", "C: random pairs")]:
        s = S[k]
        print(f"| {name} | {s.pairs} | {s.viol} | {s.nonfin} | {s.ratio:.4f} {s.at} | {s.ulp:.3f} {s.ulpat} |")
    total = sum(s.pairs for s in S.values())
    print(f"\nTotal pairs: {total}; outside the bound: {sum(s.viol for s in S.values())}")
    xx = torch.ones(16, device=dev); oo = torch.empty_like(xx)
    h = div[(1,)](xx, xx, oo, 16, B=16)
    print("PTX division instructions:", sorted(set(re.findall(r"\b(?:div|rcp)\.[a-z0-9.]+", h.asm["ptx"]))))


if __name__ == "__main__":
    main()
