"""lr-35a (library round 35): does a float32 math function's result fuse into a following addition?

Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
For each function f, launch 1 stores f(x); launch 2 (same inputs, same options) stores f(x) + c.
SEMANTICS.md "Float + and -" counts only `*`, `tl.dot`, `tl.rand` and a float32 `/` as products, so
f(x) + c is the IEEE form: r must equal fl(q + c) with q the stored f(x). Counted: results that are
not that IEEE sum. c is -q moved by up to 3 ulps (so an unrounded product would show), plus random c.
"""
import os, re, subprocess, struct
import torch, triton, triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
CUOBJDUMP = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")


@triton.jit
def k(x_ptr, c_ptr, q_ptr, F: tl.constexpr, ADD: tl.constexpr, N: tl.constexpr):
    o = tl.arange(0, N)
    x = tl.load(x_ptr + o)
    if F == 0:
        q = tl.exp(x)
    elif F == 1:
        q = tl.exp2(x)
    elif F == 2:
        q = tl.log(x)
    elif F == 3:
        q = tl.log2(x)
    elif F == 4:
        q = tl.sqrt(x)
    elif F == 5:
        q = tl.rsqrt(x)
    elif F == 6:
        q = tl.sin(x)
    elif F == 7:
        q = tl.cos(x)
    elif F == 8:
        q = tl.erf(x)
    elif F == 9:
        q = tl.sqrt_rn(x)
    elif F == 10:
        q = tl.math.div_rn(x, x + 3.0)
    elif F == 11:
        q = tl.extra.cuda.libdevice.tanh(x)
    else:
        q = tl.extra.cuda.libdevice.pow(x, 1.5)
    if ADD:
        c = tl.load(c_ptr + o)
        q = q + c
    tl.store(q_ptr + o, q)


NAMES = ["tl.exp", "tl.exp2", "tl.log", "tl.log2", "tl.sqrt", "tl.rsqrt", "tl.sin", "tl.cos", "tl.erf",
         "tl.sqrt_rn", "tl.div_rn (control)", "libdevice.tanh", "libdevice.pow(x, 1.5)"]


def sass_ops(h):
    path = os.path.join(HERE, "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(h.asm["cubin"])
    try:
        s = subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)
    return sorted({m for m in re.findall(r"\b((?:[DHF](?:ADD|FMA|MUL)2?|MUFU)(?:\.[A-Z0-9_]+)*)\b", s)})


def main():
    torch.manual_seed(0)
    N = 4096
    dev = "cuda"
    regimes = {
        "moderate": torch.rand(N, device=dev) * 4 + 0.25,
        "wide": torch.exp2(torch.rand(N, device=dev) * 60 - 30),
        "exp-subnormal": -(torch.rand(N, device=dev) * 20 + 88.0),   # exp(x) subnormal or tiny
        "exp2-subnormal": -(torch.rand(N, device=dev) * 22 + 126.0),
    }
    for F, name in enumerate(NAMES):
        for rname, x in regimes.items():
            if rname.startswith("exp") and F not in (0, 1):
                continue
            x = x.float().contiguous()
            q = torch.empty_like(x)
            k[(1,)](x, x, q, F=F, ADD=False, N=N)
            ulp = torch.where(q == 0, torch.full_like(q, 1e-45), torch.abs(q) * 2.0 ** -23)
            sh = torch.randint(-3, 4, (N,), device=dev).float()
            c = (-(q + sh * ulp)).contiguous()
            c[N // 2:] = torch.randn(N - N // 2, device=dev) * q[N // 2:].abs()
            r = torch.empty_like(x)
            h = k[(1,)](x, c, r, F=F, ADD=True, N=N)
            ieee = (q.double() + c.double()).float()   # fl(q + c): exact in double for these, then rounded
            fin = torch.isfinite(r) & torch.isfinite(ieee)
            diff = int(((r != ieee) & fin).sum())
            print(f"| `{name}(x) + c` | {rname} | {N} | {diff} | {', '.join(sass_ops(h))} |", flush=True)


if __name__ == "__main__":
    main()
