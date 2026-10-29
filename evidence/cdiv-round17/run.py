"""`tl.cdiv(x, div)`: Triton's `@triton.jit` body `(x + (div - 1)) // div`, not ceiling division.

Library evidence for library issue L88 (round 17). `primitives/triton-3.8.0-standard.py` 32-43
defines `tl.cdiv` (and the tensor method `x.cdiv(div)`, `_tensor_member_fn`) as a `@jit` function
whose body is `(x + (div - 1)) // div`. The front end inlines it like any helper
(`primitives/triton-3.8.0-code-generator.py` 1358-1396), so its meaning is that expression's: on two
compile-time values, Python's arithmetic at compile time, and the call returns a compile-time value
(`decide_return_type`, 556-585, keeps a `constexpr` return type); with a tensor operand, the tensor
operators, which wrap (`+`, `-`) and truncate (`//`) in the promoted integer dtype.

  C1  x an int32 block loaded from memory, div a tl.constexpr in {1, 2, 3, 4, 7, 128, 1024, -4}:
      EVERY int32 value of x, for each div (2^32 inputs per divisor), against
      wrap32(x + (div - 1)) truncated-divided by div
  C2  x and div two int32 blocks loaded from memory (both runtime): 2^24 random pairs and every pair of
      a set of edge values (div != 0, and not x = -2^31 with div = -1), against the same formula
  C3  x a runtime int32 scalar argument, div a tl.constexpr: edge values of x (the scalar form)
  C4  x an int64 block, div = 128 (tl.constexpr): edge values, against the formula wrapped at 64 bits
  C5  the method form x.cdiv(4) on C1's int32 block (a sample), against tl.cdiv(x, 4)'s formula
  C6  both arguments compile-time values: tl.cdiv(A, B) stored, against Python's (A + (B - 1)) // B,
      and tl.static_assert(tl.cdiv(A, B) == Python's value), which the front end evaluates only on a
      compile-time condition; the result used as tl.arange's bound (only a compile-time value is
      accepted there)

Each case runs in its own process with its own empty Triton cache.
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/cdiv-round17/run.py
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl

I32_MIN, I32_MAX = -2**31, 2**31 - 1


@triton.jit
def k_block_const(x_ptr, out_ptr, n, D: tl.constexpr, METHOD: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if METHOD:
        q = x.cdiv(D)
    else:
        q = tl.cdiv(x, D)
    tl.store(out_ptr + offs, q, mask=m)


@triton.jit
def k_block_both(x_ptr, d_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    d = tl.load(d_ptr + offs, mask=m, other=1)
    tl.store(out_ptr + offs, tl.cdiv(x, d), mask=m)


@triton.jit(do_not_specialize=["x"])
def k_scalar(x, out_ptr, D: tl.constexpr):
    tl.store(out_ptr, tl.cdiv(x, D))


@triton.jit
def k_const(out_ptr, A: tl.constexpr, B: tl.constexpr, E: tl.constexpr):
    tl.static_assert(tl.cdiv(A, B) == E)
    tl.store(out_ptr, tl.cdiv(A, B))


@triton.jit
def k_arange(out_ptr, A: tl.constexpr, B: tl.constexpr):
    r = tl.arange(0, tl.cdiv(A, B))
    tl.store(out_ptr, tl.sum(r))


def wrap(v, w):
    return ((v + 2**(w - 1)) % 2**w) - 2**(w - 1)


def trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def formula(x, d, w=32):
    return wrap(trunc_div(wrap(x + wrap(d - 1, w), w), d), w)


def expected_t(x, d):
    """The formula on int64 torch tensors holding int32 values (d a tensor or an int)."""
    s = x + (d - 1)
    s = ((s + 2**31) % 2**32) - 2**31
    q = torch.div(s, d, rounding_mode="trunc")
    return ((q + 2**31) % 2**32) - 2**31


def c1():
    dev = "cuda"
    res = []
    CH = 2**24
    BLOCK = 1024
    for D in (1, 2, 3, 4, 7, 128, 1024, -4):
        bad = 0
        for start in range(I32_MIN, I32_MAX + 1, CH):
            x = torch.arange(start, start + CH, device=dev, dtype=torch.int64)
            x32 = x.to(torch.int32)
            out = torch.empty(CH, device=dev, dtype=torch.int32)
            k_block_const[(CH // BLOCK,)](x32, out, CH, D=D, METHOD=False, BLOCK=BLOCK)
            exp = expected_t(x, D)
            bad += int((out.to(torch.int64) != exp).sum().item())
            del x, x32, out, exp
        torch.cuda.synchronize()
        res.append(f"div={D}: {bad} of 2^32 differ")
    return "; ".join(res)


EDGES = [I32_MIN, I32_MIN + 1, -2**30, -1025, -1024, -129, -128, -127, -8, -5, -4, -3, -2, -1, 0, 1, 2, 3,
         4, 5, 7, 8, 127, 128, 129, 1000, 1024, 2**30, I32_MAX - 1024, I32_MAX - 128, I32_MAX - 127,
         I32_MAX - 1, I32_MAX]


def c2():
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(17)
    n = 2**24
    x = torch.randint(I32_MIN, I32_MAX + 1, (n,), generator=g, dtype=torch.int64)
    d = torch.randint(I32_MIN, I32_MAX + 1, (n,), generator=g, dtype=torch.int64)
    small = torch.randint(-300, 301, (n // 2,), generator=g, dtype=torch.int64)
    d[: n // 2] = small
    pairs = [(a, b) for a in EDGES for b in EDGES if b != 0 and not (a == I32_MIN and b == -1)]
    x = torch.cat([x, torch.tensor([p[0] for p in pairs], dtype=torch.int64)])
    d = torch.cat([d, torch.tensor([p[1] for p in pairs], dtype=torch.int64)])
    keep = (d != 0) & ~((x == I32_MIN) & (d == -1))
    x, d = x[keep], d[keep]
    N = x.numel()
    out = torch.empty(N, device=dev, dtype=torch.int32)
    k_block_both[(triton.cdiv(N, 1024),)](x.to(torch.int32).to(dev), d.to(torch.int32).to(dev), out, N, BLOCK=1024)
    torch.cuda.synchronize()
    exp = expected_t(x, d)
    bad = int((out.cpu().to(torch.int64) != exp).sum().item())
    return f"{N} pairs ({len(pairs)} edge pairs): {bad} differ"


def c3():
    dev = "cuda"
    res = []
    for D in (4, 128):
        bad, shown = 0, []
        for v in EDGES:
            out = torch.zeros(1, device=dev, dtype=torch.int32)
            k_scalar[(1,)](v, out, D=D)
            torch.cuda.synchronize()
            got = int(out.item())
            if got != formula(v, D):
                bad += 1
            if v in (I32_MAX, I32_MAX - 127, -5, -1, 0, 1000):
                shown.append(f"x={v}: {got}")
        res.append(f"div={D}: {bad} of {len(EDGES)} differ ({', '.join(shown)})")
    return "; ".join(res)


def c4():
    dev = "cuda"
    vals = [-2**63, -2**63 + 1, -2**40, -129, -128, -5, -1, 0, 1, 127, 128, 129, 2**31 - 1, 2**31, 2**32,
            2**62, 2**63 - 128, 2**63 - 127, 2**63 - 1]
    x = torch.tensor(vals, device=dev, dtype=torch.int64)
    out = torch.empty(len(vals), device=dev, dtype=torch.int64)
    k_block_const[(1,)](x, out, len(vals), D=128, METHOD=False, BLOCK=32)
    torch.cuda.synchronize()
    got = out.cpu().tolist()
    bad = [f"x={v}: {g}" for v, g in zip(vals, got) if g != formula(v, 128, 64)]
    shown = [f"x={v}: {g}" for v, g in zip(vals, got) if v in (2**31 - 1, 2**31, 2**63 - 1, -5)]
    return f"int64, div=128: {len(bad)} of {len(vals)} differ ({', '.join(shown)})"


def c5():
    dev = "cuda"
    x = torch.tensor(EDGES, device=dev, dtype=torch.int32)
    out = torch.empty(len(EDGES), device=dev, dtype=torch.int32)
    k_block_const[(1,)](x, out, len(EDGES), D=4, METHOD=True, BLOCK=64)
    torch.cuda.synchronize()
    bad = sum(1 for v, g in zip(EDGES, out.cpu().tolist()) if g != formula(v, 4))
    return f"x.cdiv(4): {bad} of {len(EDGES)} differ"


def c6():
    dev = "cuda"
    res = []
    for A, B in ((-5, 4), (5, 4), (7, -2), (2**31 - 1, 128), (0, 3), (-7, -2)):
        E = (A + (B - 1)) // B
        out = torch.zeros(1, device=dev, dtype=torch.int32)
        k_const[(1,)](out, A=A, B=B, E=E)
        torch.cuda.synchronize()
        res.append(f"tl.cdiv({A}, {B}) = {int(out.item())} (Python's {E})")
    out = torch.zeros(1, device=dev, dtype=torch.int32)
    k_arange[(1,)](out, A=13, B=4)
    torch.cuda.synchronize()
    res.append(f"tl.arange(0, tl.cdiv(13, 4)) accepted, its sum {int(out.item())}")
    return "; ".join(res)


CASES = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = CASES[sys.argv[2]]()
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in CASES:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = "process failed: " + (err[-1][:160] if err else "(no output)")
        print(f"{name}: {res}")


if __name__ == "__main__":
    main()
