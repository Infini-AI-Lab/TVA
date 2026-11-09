"""Library issue L168 (2): which element types Triton 3.8.0 accepts for each `tl.atomic_*`.

For each op (add, max, min, and, or, xor, xchg, cas) and each pointer element type a torch tensor
argument gives (bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8e4nv,
float8e5, float16, bfloat16, float32, float64): one tensor call of 64 lanes, each lane on its own
cell, its result stored. Reports whether the launch is rejected (and the error's class and last
line), the PTX atomic instructions, and for an accepted launch the lanes whose returned value is
not the old bit pattern and whose new value is not the expected one (integers wrap; small
integers for floats, so every float result is exact; a float and/or/xor against the op on the bit
patterns; cas compares with the old value on even lanes and with another value on odd lanes).
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I evidence/atomic-dtypes/run.py
"""
import re

import numpy as np
import torch
import triton
import triton.language as tl

N = 64


@triton.jit
def k(p, v, c, o, OP: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(v + j)
    if OP == "add":
        r = tl.atomic_add(p + j, x)
    elif OP == "max":
        r = tl.atomic_max(p + j, x)
    elif OP == "min":
        r = tl.atomic_min(p + j, x)
    elif OP == "and":
        r = tl.atomic_and(p + j, x)
    elif OP == "or":
        r = tl.atomic_or(p + j, x)
    elif OP == "xor":
        r = tl.atomic_xor(p + j, x)
    elif OP == "xchg":
        r = tl.atomic_xchg(p + j, x)
    else:
        r = tl.atomic_cas(p + j, tl.load(c + j), x)
    tl.store(o + j, r)


DTYPES = [torch.bool, torch.int8, torch.uint8, torch.int16, torch.uint16, torch.int32, torch.uint32,
          torch.int64, torch.uint64, torch.float8_e4m3fn, torch.float8_e5m2, torch.float16,
          torch.bfloat16, torch.float32, torch.float64]
OPS = ["add", "max", "min", "and", "or", "xor", "xchg", "cas"]
SIGNED = {torch.int8: np.int8, torch.int16: np.int16, torch.int32: np.int32, torch.int64: np.int64}
UNSIGNED = {torch.uint8: np.uint8, torch.uint16: np.uint16, torch.uint32: np.uint32,
            torch.uint64: np.uint64}
BITS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(t):
    """The raw bit patterns of a CUDA tensor, as unsigned numpy integers."""
    w = t.element_size()
    a = t.view(BITS[w]).cpu().numpy() if t.dtype != torch.bool else t.view(torch.int8).cpu().numpy()
    return a.view(f"u{w}")


def inputs(dt, rng):
    """old, operand, cmp as torch tensors on the device, and numpy arrays to compute with."""
    if dt in SIGNED or dt in UNSIGNED:
        nd = SIGNED.get(dt) or UNSIGNED.get(dt)
        info = np.iinfo(nd)
        lo, hi = (0, 200) if dt in UNSIGNED else (-100, 100)
        old = rng.integers(lo, hi, N).astype(nd)
        val = rng.integers(lo, hi, N).astype(nd)
        old[0], val[0] = info.max, nd(1)          # add wraps
        old[1], val[1] = info.min, info.max
        w = old.itemsize
        mk = lambda a: torch.from_numpy(a.view(f"i{w}").copy()).cuda().view(dt)
        cmp = old.copy()
        cmp[1::2] = old[1::2] + nd(1)
        return mk(old), mk(val), mk(cmp), old, val, cmp
    old = rng.integers(-8, 8, N).astype(np.float64)
    val = rng.integers(-8, 8, N).astype(np.float64)
    cmp = old.copy()
    cmp[1::2] = old[1::2] + 1
    if dt == torch.bool:
        old, val, cmp = old > 0, val > 0, cmp > 0
    mk = lambda a: torch.from_numpy(a).cuda().to(dt)
    return mk(old), mk(val), mk(cmp), old, val, cmp


def expected(op, dt, old, val, cmp):
    if dt in SIGNED or dt in UNSIGNED:
        with np.errstate(over="ignore"):
            return {"add": old + val, "max": np.maximum(old, val), "min": np.minimum(old, val),
                    "and": old & val, "or": old | val, "xor": old ^ val, "xchg": val,
                    "cas": np.where(old == cmp, val, old)}[op]
    return {"add": old + val, "max": np.maximum(old, val), "min": np.minimum(old, val),
            "xchg": val, "cas": np.where(old == cmp, val, old)}.get(op)


def last(e):
    s = str(e).strip()
    s = s.splitlines()[-1] if s else ""
    s = re.sub(r"(/[^\s:]+)+", "<path>", s)
    return f"{type(e).__name__}: {s[:100]}"


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    rng = np.random.default_rng(168)
    for op in OPS:
        print(f"## atomic_{op}")
        for dt in DTYPES:
            name = str(dt).replace("torch.", "")
            p, v, c, oldn, valn, cmpn = inputs(dt, rng)
            p0 = p.clone()
            o = torch.empty_like(p)
            try:
                h = k[(1,)](p, v, c, o, op, N)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{name}: rejected ({last(e)})")
                continue
            ptx = h.asm["ptx"]
            ins = sorted(set(re.findall(r"\b(?:atom|red)\.[a-z0-9._]+", ptx)))
            ret_bad = int(np.sum(bits(o) != bits(p0)))
            exp = expected(op, dt, oldn, valn, cmpn)
            if exp is None:
                # a float and/or/xor: the bitwise op on the bit patterns
                ob, vb = bits(p0), bits(v)
                eb = {"and": ob & vb, "or": ob | vb, "xor": ob ^ vb}[op]
                new_bad = int(np.sum(bits(p) != eb))
                new = f"new value not the {op} of the bit patterns on {new_bad} of {N}"
            else:
                e = torch.from_numpy(np.ascontiguousarray(exp).view(f"i{exp.itemsize}")).view(dt) \
                    if (dt in SIGNED or dt in UNSIGNED) else torch.from_numpy(exp).to(dt)
                new_bad = int(np.sum(bits(p) != bits(e.cuda())))
                new = f"new value wrong on {new_bad} of {N}"
            print(f"{name}: accepted, {ins}; returned value not the old one on {ret_bad} of {N}; {new}")


if __name__ == "__main__":
    main()
