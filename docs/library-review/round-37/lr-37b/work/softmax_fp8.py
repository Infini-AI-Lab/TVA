"""lr-37b, round 37: tl.softmax on the fp8 formats and on int32 (the formats SEMANTICS.md "`/` on
floats" does not list): does it compile, the result's dtype, the PTX division.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>"""
import contextlib, os, re, sys
import torch, triton, triton.language as tl


@triton.jit
def ksm(x, o, R: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    r = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    s = tl.softmax(tl.load(x + r), dim=1, keep_dims=True, ieee_rounding=R)
    tl.static_print("result dtype:", s.dtype)
    tl.store(o + r, s.to(tl.float32))


@contextlib.contextmanager
def quiet():
    sys.stdout.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for f in saved + [null]:
            os.close(f)


def last(e):
    s = str(e).strip()
    s = re.sub(r"(/[^\s:]+)+", "<path>", s.splitlines()[-1]) if s else ""
    return f"{type(e).__name__}: {s[:110]}"


print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
M, N = 64, 16
for dt in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int32):
    for R in (False, True):
        x = (torch.randn(M * N, device="cuda") * 4).to(dt)
        o = torch.empty(M * N, device="cuda")
        try:
            with quiet():
                h = ksm[(1,)](x, o, R, M, N)
                torch.cuda.synchronize()
            ref = torch.softmax(x.float().view(M, N), 1).flatten()
            print(f"{dt} ieee_rounding={R}: compiles; {sorted(set(re.findall(r'div[.][a-z0-9.]+', h.asm['ptx'])))}; "
                  f"max |result - torch softmax of the float32 values| = {float((o - ref).abs().max()):.2e}")
        except Exception as e:
            print(f"{dt} ieee_rounding={R}: rejected ({last(e)})")
