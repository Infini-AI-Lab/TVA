"""lr-37b, round 37: adversarial checks around evidence/atomic-dtypes (library issue L168).

  S. SASS of a few accepted cases (cuobjdump of the cubin): f16/bf16 add with the result used
     and unused, f32 atomic_max, f64 atomic_and, f16 cas.
  U. the result unused (tensor, 64 contiguous lanes): the PTX instruction and the new values,
     for f16/bf16/f32 add and the 32-bit integer ops; f16/bf16 add with the operand's values
     chosen so that each lane's rounding matters.
  C. scalar atomics (one pointer, one value) on each element type: accepted or rejected, as for
     tensors.
  V. operand conversions: atomic_add of a float32 tensor into float16/bfloat16 cells (the
     operand is converted with semantic.cast, round to nearest even, before the add), of a
     float32 into int32 cells; atomic_cas with operands of another type than the cell (no cast
     in semantic.atomic_cas), a tensor pointer with scalar cmp/val.
  B. float32 atomic_cas compares bit patterns: +0 vs -0, NaN vs the same NaN.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import contextlib
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch
import triton
import triton.language as tl

N = 64


@triton.jit
def k_used(p, v, o, OP: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(v + j)
    if OP == "add":
        r = tl.atomic_add(p + j, x)
    elif OP == "max":
        r = tl.atomic_max(p + j, x)
    elif OP == "and":
        r = tl.atomic_and(p + j, x)
    else:
        r = tl.atomic_cas(p + j, tl.load(o + j), x)
    tl.store(o + j, r)


@triton.jit
def k_unused(p, v, OP: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(v + j)
    if OP == "add":
        tl.atomic_add(p + j, x)
    elif OP == "max":
        tl.atomic_max(p + j, x)
    elif OP == "min":
        tl.atomic_min(p + j, x)
    elif OP == "and":
        tl.atomic_and(p + j, x)
    elif OP == "or":
        tl.atomic_or(p + j, x)
    elif OP == "xor":
        tl.atomic_xor(p + j, x)
    else:
        tl.atomic_xchg(p + j, x)


@triton.jit
def k_scalar(p, v, o, OP: tl.constexpr):
    x = tl.load(v)
    if OP == "add":
        r = tl.atomic_add(p, x)
    elif OP == "max":
        r = tl.atomic_max(p, x)
    elif OP == "min":
        r = tl.atomic_min(p, x)
    elif OP == "and":
        r = tl.atomic_and(p, x)
    elif OP == "or":
        r = tl.atomic_or(p, x)
    elif OP == "xor":
        r = tl.atomic_xor(p, x)
    elif OP == "xchg":
        r = tl.atomic_xchg(p, x)
    else:
        r = tl.atomic_cas(p, x, x)
    tl.store(o, r)


@triton.jit
def k_conv(p, v, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.atomic_add(p + j, tl.load(v + j))


@triton.jit
def k_cas_mixed(p, o, CASE: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    if CASE == 0:      # int64 cells, Python int operands (int32 scalars)
        r = tl.atomic_cas(p, 0, 1)
        tl.store(o, r)
    elif CASE == 1:    # tensor pointer, scalar operands
        r = tl.atomic_cas(p + j, 0, 1)
        tl.store(o + j, r)
    elif CASE == 2:    # float16 cells, Python float operands (float32 scalars)
        r = tl.atomic_cas(p, 0.0, 1.0)
        tl.store(o, r)
    else:              # int32 cells, int64 operands
        r = tl.atomic_cas(p, tl.full([], 0, tl.int64), tl.full([], 1, tl.int64))
        tl.store(o, r)


@triton.jit
def k_cas_bits(p, c, v, o, N: tl.constexpr):
    j = tl.arange(0, N)
    r = tl.atomic_cas(p + j, tl.load(c + j), tl.load(v + j))
    tl.store(o + j, r)


@contextlib.contextmanager
def quiet():
    """Drop what the compiler writes to file descriptors 1 and 2 (a rejected module's MLIR dump)."""
    sys.stdout.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for f in saved + [null]:
            os.close(f)


def last(e):
    s = str(e).strip()
    s = s.splitlines()[-1] if s else ""
    s = re.sub(r"(/[^\s:]+)+", "<path>", s)
    return f"{type(e).__name__}: {s[:100]}"


def sass(h):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "k.cubin")
        with open(f, "wb") as fh:
            fh.write(h.asm["cubin"])
        out = subprocess.run([triton.knobs.nvidia.cuobjdump.path, "-sass", f], capture_output=True,
                             text=True).stdout
    ins = set()
    for line in out.splitlines():
        m = re.search(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
        if m and re.match(r"(ATOM|RED|ATOMG|REDG)", m.group(1)):
            ins.add(m.group(1))
    return sorted(ins)


def ptx_atoms(h):
    return sorted(set(re.findall(r"\b(?:atom|red)\.[a-z0-9._]+", h.asm["ptx"])))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    dev = "cuda"
    print("## S. SASS of a few accepted cases")
    for op, dt in [("add", torch.float16), ("add", torch.bfloat16), ("max", torch.float32),
                   ("and", torch.float64), ("cas", torch.float16), ("add", torch.float32)]:
        p = torch.zeros(N, device=dev, dtype=dt)
        v = torch.ones(N, device=dev, dtype=dt)
        o = torch.zeros(N, device=dev, dtype=dt)
        h = k_used[(1,)](p, v, o, op, N)
        line = f"{op} {dt} used: PTX {ptx_atoms(h)}; SASS {sass(h)}"
        if op != "cas":
            h2 = k_unused[(1,)](p, v, op, N)
            line += f"; unused: PTX {ptx_atoms(h2)}; SASS {sass(h2)}"
        print(line)
    torch.cuda.synchronize()

    print("## U. the result unused: new values")
    rng = np.random.default_rng(37)
    for op in ["add", "max", "min", "and", "or", "xor", "xchg"]:
        for dt in [torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32]:
            if dt in (torch.float16, torch.bfloat16) and op != "add":
                continue
            if dt == torch.float32 and op in ("and", "or", "xor"):
                continue
            if dt.is_floating_point:
                old = torch.from_numpy(rng.standard_normal(N) * 100).to(dt).to(dev)
                val = torch.from_numpy(rng.standard_normal(N) * 3).to(dt).to(dev)
            else:
                old = torch.from_numpy(rng.integers(-1000, 1000, N)).to(dt).to(dev)
                val = torch.from_numpy(rng.integers(-1000, 1000, N)).to(dt).to(dev)
            p = old.clone()
            h = k_unused[(1,)](p, val, op, N)
            torch.cuda.synchronize()
            if dt.is_floating_point:
                o64, v64 = old.double(), val.double()
                exp = {"add": (o64 + v64), "max": torch.maximum(o64, v64), "min": torch.minimum(o64, v64),
                       "xchg": v64}[op].to(dt)  # o + v exact in double, one rounding to dt
            else:
                exp = {"add": old + val, "max": torch.maximum(old, val), "min": torch.minimum(old, val),
                       "and": old & val, "or": old | val, "xor": old ^ val, "xchg": val}[op]
            bad = int((p.view(torch.int16 if dt.itemsize == 2 else (torch.int32 if dt.itemsize == 4 else torch.int64))
                       != exp.view(torch.int16 if dt.itemsize == 2 else (torch.int32 if dt.itemsize == 4 else torch.int64))).sum())
            print(f"{op} {str(dt)[6:]} unused: {ptx_atoms(h)}; new value wrong on {bad} of {N}")

    print("## C. scalar atomics: accepted or rejected")
    dts = [torch.bool, torch.int8, torch.uint8, torch.int16, torch.uint16, torch.int32, torch.uint32,
           torch.int64, torch.uint64, torch.float8_e4m3fn, torch.float8_e5m2, torch.float16,
           torch.bfloat16, torch.float32, torch.float64]
    for op in ["add", "max", "min", "and", "or", "xor", "xchg", "cas"]:
        acc = []
        for dt in dts:
            p = torch.zeros(1, device=dev, dtype=dt)
            v = torch.zeros(1, device=dev, dtype=dt)
            o = torch.zeros(1, device=dev, dtype=dt)
            try:
                k_scalar[(1,)](p, v, o, op)
                torch.cuda.synchronize()
                acc.append(str(dt)[6:])
            except Exception:
                pass
        print(f"atomic_{op} scalar accepts: {acc}")

    print("## V. operand conversions")
    x = torch.from_numpy(rng.standard_normal(N) * 1000 + 0.3).to(torch.float32).to(dev)
    x[0] = 65520.0   # rounds to +inf in float16 (RTNE)
    x[1] = 1.0 + 2.0 ** -11  # a tie in float16: RTNE keeps 1.0
    for dt in (torch.float16, torch.bfloat16, torch.int32):
        p = torch.zeros(N, device=dev, dtype=dt)
        try:
            h = k_conv[(1,)](p, x, N)
            torch.cuda.synchronize()
            exp = x.to(dt)   # torch: float->half/bf16 RTNE, float->int truncation
            bad = int((p.double() != exp.double()).sum()) if dt != torch.int32 else int((p != exp).sum())
            print(f"float32 operand into {str(dt)[6:]} cells (0 + cast(x)): {ptx_atoms(h)}; "
                  f"!= torch's conversion on {bad} of {N}; lanes 0,1: {p[:2].tolist()}")
        except Exception as e:
            print(f"float32 operand into {dt}: rejected ({last(e)})")
    for case, dt in [(0, torch.int64), (1, torch.int64), (2, torch.float16), (3, torch.int32)]:
        p = torch.zeros(N, device=dev, dtype=dt)
        o = torch.full((N,), 7, device=dev, dtype=dt)
        try:
            with quiet():
                h = k_cas_mixed[(1,)](p, o, case, N)
                torch.cuda.synchronize()
            print(f"cas case {case} ({dt}): accepted, {ptx_atoms(h)}; cells {p[:4].tolist()}, results {o[:4].tolist()}")
        except Exception as e:
            print(f"cas case {case} ({dt}): rejected ({last(e)})")

    print("## B. float32 atomic_cas compares bit patterns")
    pats = np.array([0x00000000, 0x80000000, 0x7fc00000, 0x7fc00001, 0xffc00000, 0x3f800000],
                    dtype=np.uint32)
    olds, cmps = np.meshgrid(pats, pats)
    olds, cmps = olds.ravel(), cmps.ravel()
    n = olds.size
    pad = 64 - n
    olds = np.concatenate([olds, np.zeros(pad, np.uint32)])
    cmps = np.concatenate([cmps, np.ones(pad, np.uint32)])
    t = lambda a: torch.from_numpy(a.view(np.int32).copy()).to(dev).view(torch.float32)
    p, c = t(olds), t(cmps)
    v = torch.full((64,), 2.0, device=dev)
    o = torch.empty(64, device=dev)
    k_cas_bits[(1,)](p, c, v, o, 64)
    torch.cuda.synchronize()
    newb = p.view(torch.int32).cpu().numpy().view(np.uint32)
    expb = np.where(olds == cmps, np.uint32(0x40000000), olds)
    retb = o.view(torch.int32).cpu().numpy().view(np.uint32)
    print(f"new value != bitwise-cas on {int((newb != expb).sum())} of 64; returned != old on "
          f"{int((retb != olds).sum())} of 64; swapped where +0 vs -0: "
          f"{int(((olds == 0) & (cmps == 0x80000000) & (newb == 0x40000000)).sum())}, where NaN == same NaN: "
          f"{int(((olds == 0x7fc00000) & (cmps == 0x7fc00000) & (newb == 0x40000000)).sum())}")


if __name__ == "__main__":
    main()
