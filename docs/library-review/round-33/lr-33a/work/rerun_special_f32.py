"""`tl.exp` past its overflow and `/` by an infinite divisor, on float32 (library issue L154 (3)).

  E1  `tl.exp(x)` for every float32 pattern from `+64.0` (0x42800000) to `+inf` (0x7f800000): which inputs give
      `+inf`, the smallest input from which every larger one gives `+inf`, and the largest finite result
  E2  `tl.exp(x)` for every float32 pattern from `-0.0`'s negative side down to `-inf` below `-64`
      (0xc2800000 .. 0xff800000): every result `+0`, a subnormal, or a normal (for the record)
  D1  `x / y` (`tl.fdiv`, `div.full.f32`) for every finite float32 `x` (both signs, subnormals included) and
      `y = +inf`, `y = -inf` (loaded, so nothing folds): is the result the zero of sign `sign(x) xor sign(y)`?

Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-special-f32/run.py
"""
import torch
import triton
import triton.language as tl

CH = 1 << 26


@triton.jit
def k_exp(lo, n, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    bits = (lo + i).to(tl.uint32)
    x = bits.to(tl.float32, bitcast=True)
    y = tl.exp(x)
    tl.store(o_ptr + i, y.to(tl.uint32, bitcast=True), mask=m)


@triton.jit
def k_div(lo, n, yp, o_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    bits = (lo + i).to(tl.uint32)
    x = bits.to(tl.float32, bitcast=True)
    y = tl.load(yp)
    tl.store(o_ptr + i, (x / y).to(tl.uint32, bitcast=True), mask=m)


def run_exp(lo, hi):
    """Yields (start, results as int64 patterns) for patterns lo..hi inclusive."""
    s = lo
    out = torch.empty(CH, dtype=torch.int32, device="cuda")
    while s <= hi:
        n = min(CH, hi - s + 1)
        k_exp[(triton.cdiv(n, 1024),)](s, n, out, BLOCK=1024)
        torch.cuda.synchronize()
        yield s, out[:n].to(torch.int64) & 0xffffffff
        s += n


def e1():
    lo, hi = 0x42800000, 0x7f800000
    INF = 0x7f800000
    last_noninf = None
    n_inf = 0
    max_fin = 0
    nan = 0
    for s, r in run_exp(lo, hi):
        isinf = r == INF
        n_inf += int(isinf.sum())
        nan += int(((r & 0x7fffffff) > INF).sum())
        nz = (~isinf).nonzero()
        if nz.numel():
            last_noninf = s + int(nz[-1])
            max_fin = max(max_fin, int(r[~isinf].max()))
    T = last_noninf + 1
    tf = torch.tensor([T], dtype=torch.int64).to(torch.int32).view(torch.float32).item()
    lf = torch.tensor([last_noninf], dtype=torch.int64).to(torch.int32).view(torch.float32).item()
    mf = torch.tensor([max_fin], dtype=torch.int64).to(torch.int32).view(torch.float32).item()
    return [f"E1 tl.exp on {hi - lo + 1} patterns from +64 to +inf: {n_inf} give +inf, {nan} NaN; the largest input "
            f"with a result other than +inf is 0x{last_noninf:08x} ({lf!r}); every input from 0x{T:08x} ({tf!r}) "
            f"to +inf gives +inf; the largest finite result is 0x{max_fin:08x} ({mf!r})"]


def e2():
    lo, hi = 0xc2800000, 0xff800000
    zero = sub = norm = other = 0
    for s, r in run_exp(lo, hi):
        zero += int((r == 0).sum())
        sub += int(((r > 0) & (r < 0x00800000)).sum())
        norm += int(((r >= 0x00800000) & (r < 0x7f800000)).sum())
        other += int((r >= 0x7f800000).sum())
    return [f"E2 tl.exp on {hi - lo + 1} patterns from -64 to -inf: {zero} give +0, {sub} a subnormal, {norm} a "
            f"normal, {other} anything else"]


def d1():
    lines = []
    out = torch.empty(CH, dtype=torch.int32, device="cuda")
    for ybits, yname in ((0x7f800000, "+inf"), (0xff800000, "-inf")):
        yp = torch.tensor([ybits], dtype=torch.int64).to(torch.int32).view(torch.float32).cuda()
        ysign = ybits >> 31
        tot = bad = 0
        for lo, hi in ((0x00000000, 0x7f7fffff), (0x80000000, 0xff7fffff)):
            s = lo
            while s <= hi:
                n = min(CH, hi - s + 1)
                k_div[(triton.cdiv(n, 1024),)](s, n, yp, out, BLOCK=1024)
                torch.cuda.synchronize()
                r = out[:n].to(torch.int64) & 0xffffffff
                xs = (s >> 31) & 1
                want = ((xs ^ ysign) << 31)
                bad += int((r != want).sum())
                tot += n
                s += n
        lines.append(f"D1 x / {yname} for every finite float32 x ({tot}): {bad} differ from the zero of sign "
                     f"sign(x) xor sign(y)")
    return lines


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for f in (e1, e2, d1):
        for l in f():
            print(l, flush=True)


if __name__ == "__main__":
    main()
