"""TVA library evidence: the order in which Triton's front end evaluates a call's arguments, what a
conditional expression with a runtime condition runs, and what `@triton.jit(launch_metadata=...)` does.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-order/run.py
Output: evidence/front-end-order/out.txt (README.md says what it shows).

Part K compiles calls whose positional and keyword arguments each hold an atomic of a different kind,
and prints the order of the atomics in the TTIR (the order the front end emitted them). Part C runs a
conditional expression `a if flag > 0 else b` with a runtime `flag`, each side an atomic increment of
a counter by a different amount, and prints the counter. Part M launches the same kernel with and
without `launch_metadata=`, with no launch hook and then with one, and prints how often the metadata
function ran and whether the compiled PTX and the results are the same.
"""
import datetime
import re

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def kw_store(c_ptr, out_ptr):
    # positional: atomic_add; keyword: atomic_xor
    tl.store(out_ptr, tl.atomic_add(c_ptr, 1), mask=tl.atomic_xor(c_ptr + 1, 3) >= 0)


@triton.jit
def kw_where(c_ptr, out_ptr):
    # a positional argument, a keyword argument and a keyword that is a constant, in two calls
    v = tl.load(out_ptr)
    tl.store(out_ptr + tl.atomic_max(c_ptr + 2, 7), v, mask=tl.atomic_and(c_ptr + 3, 1) >= 0,
             cache_modifier=".cg")
    tl.atomic_add(c_ptr, tl.atomic_or(c_ptr + 4, 2), mask=tl.atomic_xchg(c_ptr + 5, 9) >= 0)


@triton.jit
def cond_expr(c_ptr, out_ptr, flag):
    pid = tl.program_id(0)
    v = tl.atomic_add(c_ptr, 1) if flag > 0 else tl.atomic_add(c_ptr, 100)
    tl.store(out_ptr + pid, v)


def body(x_ptr, out_ptr, n, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0 + 1.0, mask=m)


CALLS = []


def meta(grid, metadata, args):
    CALLS.append(grid)
    return {"name": "probe", "flops": 2 * args["n"]}


plain = triton.jit(body)
with_meta = triton.jit(launch_metadata=meta)(body)


def rmw_order(ttir):
    return [m.group(1) for m in re.finditer(r"tt\.atomic_rmw (\w+)", ttir)]


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"{datetime.date.today().isoformat()}")
    print("\n## K: the order of a call's arguments (atomics in the TTIR, in emission order)")
    c = torch.zeros(8, dtype=torch.int32, device=DEV)
    out = torch.zeros(16, dtype=torch.int32, device=DEV)
    k = kw_store[(1,)](c, out)
    print("K1 tl.store(out, <atomic_add>, mask=<atomic_xor> >= 0):", rmw_order(k.asm["ttir"]))
    k = kw_where[(1,)](c, out)
    print("K2 tl.store(out + <atomic_max>, v, mask=<atomic_and> >= 0, cache_modifier=...); "
          "tl.atomic_add(c, <atomic_or>, mask=<atomic_xchg> >= 0):", rmw_order(k.asm["ttir"]))

    print("\n## C: x if flag > 0 else y, flag a runtime int32, 4 programs; each side adds to one counter")
    for flag in (5, -5):
        c = torch.zeros(1, dtype=torch.int32, device=DEV)
        out = torch.zeros(4, dtype=torch.int32, device=DEV)
        k = cond_expr[(4,)](c, out, flag)
        torch.cuda.synchronize()
        print(f"C flag={flag}: counter {c.item()} (only the taken side: {4 if flag > 0 else 400}; both sides: 404); "
              f"runtime branch in TTIR: {'scf.if' in k.asm['ttir']}")

    print("\n## M: launch_metadata=")
    n, B = 1000, 128
    x = torch.randn(n, device=DEV)
    o1 = torch.empty(n, device=DEV)
    o2 = torch.empty(n, device=DEV)
    k1 = plain[(triton.cdiv(n, B),)](x, o1, n, B=B)
    k2 = with_meta[(triton.cdiv(n, B),)](x, o2, n, B=B)
    torch.cuda.synchronize()
    print(f"M1 without a launch hook: metadata function ran {len(CALLS)} times")
    print(f"M2 PTX with and without launch_metadata identical: {k1.asm['ptx'] == k2.asm['ptx']}; "
          f"results identical: {torch.equal(o1, o2)}")
    seen = []

    def hook(md):
        seen.append(md.get())

    triton.knobs.runtime.launch_enter_hook.add(hook)
    try:
        with_meta[(triton.cdiv(n, B),)](x, o2, n, B=B)
        torch.cuda.synchronize()
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(hook)
    print(f"M3 with a launch-enter hook that reads the metadata: metadata function ran {len(CALLS)} times; "
          f"the hook saw name={seen[0].get('name')!r}, flops={seen[0].get('flops')}")
    with_meta[(triton.cdiv(n, B),)](x, o2, n, B=B)
    torch.cuda.synchronize()
    print(f"M4 hook removed again: metadata function ran {len(CALLS)} times in all")


if __name__ == "__main__":
    main()
