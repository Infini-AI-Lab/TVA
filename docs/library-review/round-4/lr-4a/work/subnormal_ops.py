"""lr-4a: SEMANTICS.md says float32 operations keep subnormals ("no flushing of subnormals", S5),
and `tl.maximum`/`tl.minimum` follow `FloatV.MaxRel` (the reference max/min up to NaN payload, or
either operand when both are zeros). Do the comparison-like float32 operations keep subnormals on
the device: `tl.maximum`, `tl.minimum`, `tl.clamp`, `tl.abs`, unary `-`, `<`, `==`, `tl.where`, and the
reductions `tl.max`, `tl.min`, `tl.sum`, and float `tl.atomic_max`/`tl.atomic_min`?

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/subnormal_ops.py
"""
import re

import torch
import triton
import triton.language as tl


@triton.jit
def k(a_ptr, b_ptr, o_ptr, B: tl.constexpr):
    i = tl.arange(0, B)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(o_ptr + 0 * B + i, tl.maximum(a, b))
    tl.store(o_ptr + 1 * B + i, tl.minimum(a, b))
    tl.store(o_ptr + 2 * B + i, tl.clamp(a, b, b + 1.0))
    tl.store(o_ptr + 3 * B + i, tl.abs(a))
    tl.store(o_ptr + 4 * B + i, -a)
    tl.store(o_ptr + 5 * B + i, (a < b).to(tl.float32))
    tl.store(o_ptr + 6 * B + i, (a == b).to(tl.float32))
    tl.store(o_ptr + 7 * B + i, tl.where(a < b, a, b))
    tl.store(o_ptr + 8 * B + i, tl.max(a, axis=0) + 0 * a)
    tl.store(o_ptr + 9 * B + i, tl.min(a, axis=0) + 0 * a)
    tl.store(o_ptr + 10 * B + i, tl.sum(a, axis=0) + 0 * a)


@triton.jit
def k_atomic(p_ptr, v_ptr, q_ptr, B: tl.constexpr):
    i = tl.arange(0, B)
    v = tl.load(v_ptr + i)
    tl.atomic_max(p_ptr + i, v)
    tl.atomic_min(q_ptr + i, v)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    B = 4
    s1, s2 = 2.0 ** -140, 2.0 ** -145
    a = torch.tensor([s1, -s1, s2, 0.0], dtype=torch.float32, device="cuda")
    b = torch.tensor([s2, -s2, s1, -s2], dtype=torch.float32, device="cuda")
    o = torch.empty(11 * B, dtype=torch.float32, device="cuda")
    h = k[(1,)](a, b, o, B=B)
    torch.cuda.synchronize()
    ins = sorted({l.strip().split()[0] for l in h.asm["ptx"].splitlines()
                  if re.match(r"^\s*(max|min|abs|neg|setp|selp|add)\.\S*f32", l)})
    print("PTX float32 instructions:", ins)
    A, Bv = a.cpu().tolist(), b.cpu().tolist()
    ref = [
        [max(x, y) for x, y in zip(A, Bv)],
        [min(x, y) for x, y in zip(A, Bv)],
        [min(max(x, y), y + 1.0) for x, y in zip(A, Bv)],
        [abs(x) for x in A],
        [-x for x in A],
        [float(x < y) for x, y in zip(A, Bv)],
        [float(x == y) for x, y in zip(A, Bv)],
        [x if x < y else y for x, y in zip(A, Bv)],
        [max(A)] * B,
        [min(A)] * B,
        None,
    ]
    names = ["tl.maximum", "tl.minimum", "tl.clamp(a, b, b+1)", "tl.abs", "-a", "a < b", "a == b",
             "tl.where(a < b, a, b)", "tl.max", "tl.min", "tl.sum"]
    got = o.cpu().view(11, B).tolist()
    for n, r, g in zip(names, ref, got):
        if r is None:
            exact = sum(A)
            print(f"{n:24s} {g}  (exact sum {exact!r}; any tree's IEEE sum of these is exact)")
            continue
        same = all((x == y) and (str(x)[0] == str(y)[0] or x != 0) for x, y in zip(r, g))
        print(f"{n:24s} {g}  reference {r}: {'same' if same else 'DIFFERENT'}")
    p = torch.full((B,), -1.0, dtype=torch.float32, device="cuda")
    q = torch.full((B,), 1.0, dtype=torch.float32, device="cuda")
    v = torch.tensor([s1, s2, -s1, -s2], dtype=torch.float32, device="cuda")
    k_atomic[(1,)](p, v, q, B=B)
    print(f"tl.atomic_max(-1.0, v) for v = {v.cpu().tolist()}: {p.cpu().tolist()}")
    print(f"tl.atomic_min(+1.0, v) for v = {v.cpu().tolist()}: {q.cpu().tolist()}")


if __name__ == "__main__":
    main()
