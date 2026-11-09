"""TVA evidence for library issue L168 (round 38; library review lr-37a's work/fmt_probe.py, round 37):
tl.sigmoid and tl.softmax on the fp8 and integer formats.
For each input format: does it compile, the result's dtype, the PTX divisions, and (sigmoid) whether
a following `+ c` is fused (FFMA fed by MUFU.RCP) by comparing with fusion off.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import contextlib, os, re, sys
import torch, triton, triton.language as tl


@triton.jit
def ksig(x, o, c, N: tl.constexpr):
    j = tl.arange(0, N)
    s = tl.sigmoid(tl.load(x + j))
    tl.store(o + j, s)
    tl.store(c + j, s.dtype.primitive_bitwidth + 0 * j)


@triton.jit
def ksig_add(x, c, o, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.store(o + j, tl.sigmoid(tl.load(x + j)) + tl.load(c + j))


@triton.jit
def ksm(x, o, w, R: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    s = tl.softmax(tl.load(x + r), dim=1, keep_dims=True, ieee_rounding=R)
    tl.store(o + r, s.to(tl.float32))
    tl.store(w + r, s.dtype.primitive_bitwidth + 0 * r)


@contextlib.contextmanager
def quiet():
    sys.stdout.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for f in saved + [null]:
            os.close(f)


def divs(ptx):
    return sorted(set(re.findall(r"\b(?:div|rcp)\.[a-z0-9.]+", ptx)))


def last(e):
    s = str(e).strip()
    s = re.sub(r"(/[^\s:]+)+", "<path>", s.splitlines()[-1]) if s else ""
    return f"{type(e).__name__}: {s[:100]}"


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    N = 1024
    g = torch.Generator(device="cuda").manual_seed(37)
    print("## tl.sigmoid")
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.float8_e5m2, torch.float8_e4m3fn):
        x = (torch.randn(N, device="cuda", generator=g) * 4).to(dt)
        o = torch.empty(N, device="cuda", dtype=torch.float64)
        bw = torch.empty(N, device="cuda", dtype=torch.int32)
        try:
            with quiet():
                h = ksig[(1,)](x, o, bw, N)
                torch.cuda.synchronize()
            print(f"{dt}: result width {int(bw[0])} bits; {divs(h.asm['ptx'])}")
        except Exception as e:
            print(f"{dt}: rejected ({last(e)})"); continue
        # fusion of the following addition: compare fusion on and off
        res = {}
        for fu in (True, False):
            c = -(o.to(torch.float32)) * (1 + 2 ** -12)
            oo = torch.empty(N, device="cuda", dtype=torch.float32)
            try:
                with quiet():
                    hh = ksig_add[(1,)](x, c, oo, N, enable_fp_fusion=fu)
                    torch.cuda.synchronize()
                res[fu] = oo.clone()
                ffma = len(re.findall(r"\bFFMA\b", hh.asm.get('sass', ''))) if 'sass' in hh.asm else None
            except Exception as e:
                res[fu] = None
                print(f"  + c, fusion={fu}: rejected ({last(e)})")
        if res.get(True) is not None and res.get(False) is not None:
            d = int((res[True].view(torch.int32) != res[False].view(torch.int32)).sum())
            print(f"  sigmoid(x) + c (float32 c): fusion on vs off differ on {d} of {N}")
    print("## tl.softmax")
    M, K = 64, 16
    for dt in (torch.float8_e5m2, torch.float8_e4m3fn, torch.int32, torch.int16):
        for R in (False, True):
            x = (torch.randn(M * K, device="cuda", generator=g) * 4).to(dt)
            o = torch.empty(M * K, device="cuda", dtype=torch.float32)
            w = torch.empty(M * K, device="cuda", dtype=torch.int32)
            try:
                with quiet():
                    h = ksm[(1,)](x, o, w, R, M, K)
                    torch.cuda.synchronize()
                print(f"{dt} ieee_rounding={R}: result width {int(w[0])} bits; {divs(h.asm['ptx'])}")
            except Exception as e:
                print(f"{dt} ieee_rounding={R}: rejected ({last(e)})")


if __name__ == "__main__":
    main()
