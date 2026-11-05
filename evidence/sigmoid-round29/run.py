"""`tl.sigmoid(x)`: Triton's `@triton.jit` body `1 / (1 + math.exp(-x))`.

Library evidence for library issue L137 (round 29). `primitives/triton-3.8.0-standard.py` 44-50
defines `tl.sigmoid` (and the tensor method `x.sigmoid()`, `_tensor_member_fn`) as a `@jit` function
whose body is `1 / (1 + math.exp(-x))`, `math` being `triton.language.math`. The front end inlines it
like any helper (`primitives/triton-3.8.0-code-generator.py` 1358-1396), so it means that expression:
a negation, `tl.exp` (whose `_check_dtype` admits only fp32 and fp64,
`primitives/triton-3.8.0-math.py` 8-30, 95-100), an addition of the Python number `1`, and the
default float division `/` of the Python number `1` by the sum.

  S1  x an fp32 block loaded from memory: EVERY fp32 bit pattern (2^32 inputs), tl.sigmoid(x) against
      1 / (1 + tl.exp(-x)) written out in the kernel, compared bit for bit (NaNs included)
  S2  the method form x.sigmoid() on the same inputs, against the same expression (a sample of 2^28)
  S3  x an fp64 block: 2^24 random bit patterns and edge values, bit for bit against the expression
  S4  x a bf16, fp16 or int32 block: does the front end accept tl.sigmoid(x)?
  S5  x a compile-time float (tl.constexpr): tl.sigmoid(C) stored, against the expression on C

Each case runs in its own process with its own empty Triton cache.
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/sigmoid-round29/run.py
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_both(x_ptr, a_ptr, b_ptr, n, METHOD: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if METHOD:
        s = x.sigmoid()
    else:
        s = tl.sigmoid(x)
    e = 1 / (1 + tl.exp(-x))
    tl.store(a_ptr + offs, s, mask=m)
    tl.store(b_ptr + offs, e, mask=m)


@triton.jit
def k_one(x_ptr, a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(a_ptr + offs, tl.sigmoid(x), mask=m)


@triton.jit
def k_const(a_ptr, b_ptr, C: tl.constexpr):
    tl.store(a_ptr, tl.sigmoid(C))
    tl.store(b_ptr, 1 / (1 + tl.exp(-C)))


def run_both(xbits, fdt, idt, method):
    x = xbits.view(fdt)
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    n = x.numel()
    k_both[(triton.cdiv(n, 1024),)](x, a, b, n, METHOD=method, BLOCK=1024)
    return int((a.view(idt) != b.view(idt)).sum().item())


def s1():
    diff, total = 0, 0
    step = 2**28
    for lo in range(-2**31, 2**31, step):
        bits = torch.arange(lo, lo + step, dtype=torch.int64, device="cuda").to(torch.int32)
        diff += run_both(bits, torch.float32, torch.int32, False)
        total += step
    return f"S1: tl.sigmoid on every fp32 bit pattern: {diff} of {total} differ from 1 / (1 + tl.exp(-x))"


def s2():
    bits = torch.arange(-2**31, -2**31 + 2**28, dtype=torch.int64, device="cuda").to(torch.int32)
    bits = torch.cat([bits, torch.randint(-2**31, 2**31 - 1, (2**27,), dtype=torch.int32, device="cuda")])
    d = run_both(bits, torch.float32, torch.int32, True)
    return f"S2: x.sigmoid() on {bits.numel()} fp32 bit patterns: {d} differ"


def s3():
    g = torch.Generator(device="cuda").manual_seed(29)
    bits = torch.randint(-2**63, 2**63 - 1, (2**24,), dtype=torch.int64, device="cuda", generator=g)
    edges = torch.tensor([0.0, -0.0, 1.0, -1.0, 700.0, -700.0, 745.0, -745.0, 1e-300, -1e-300,
                          float("inf"), float("-inf"), float("nan"), 2.0**1023, -2.0**1023],
                         dtype=torch.float64, device="cuda").view(torch.int64)
    bits = torch.cat([bits, edges])
    d = run_both(bits, torch.float64, torch.int64, False)
    return f"S3: fp64, {bits.numel()} bit patterns ({edges.numel()} edges): {d} differ"


def s4():
    out = []
    for name, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16), ("int32", torch.int32)):
        x = torch.ones(16, dtype=dt, device="cuda")
        a = torch.empty_like(x)
        try:
            k_one[(1,)](x, a, 16, BLOCK=16)
            torch.cuda.synchronize()
            out.append(f"{name}: accepted")
        except Exception as e:
            chain, c = [], e
            while c is not None:
                chain.append(str(c))
                c = c.__cause__ or c.__context__
            why = [l.strip() for t in chain for l in t.splitlines() if "Expected dtype" in l]
            out.append(f"{name}: rejected ({type(e).__name__}: {why[0] if why else chain[-1].strip().splitlines()[-1]})")
    return "S4: " + "; ".join(out)


def s5():
    out = []
    for c in (0.0, 1.0, -3.5, 20.0, -100.0):
        a = torch.empty(1, dtype=torch.float32, device="cuda")
        b = torch.empty(1, dtype=torch.float32, device="cuda")
        k_const[(1,)](a, b, C=c)
        same = a.view(torch.int32).item() == b.view(torch.int32).item()
        out.append(f"C={c}: {a.item():.9g} ({'same bits' if same else 'DIFFERENT'})")
    return "S5: " + "; ".join(out)


CASES = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(CASES[sys.argv[1]]())
        sys.exit(0)
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for c in CASES:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, TRITON_CACHE_DIR=cache)
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), c], env=env,
                               capture_output=True, text=True)
            print(r.stdout.strip() or f"{c}: failed: {r.stderr.strip().splitlines()[-1:]}")
    print("Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/sigmoid-round29/run.py` from TVA's root.")
