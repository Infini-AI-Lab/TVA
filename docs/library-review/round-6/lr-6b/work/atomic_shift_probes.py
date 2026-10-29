"""Library review round 6 (lr-6b), survey area `semantics`: atomics and shifts as SEMANTICS.md states them.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-6/lr-6b/work/atomic_shift_probes.py
Output: atomic_shift_probes.out.txt next to this file.

Cases:
  A1  "A tensor atomic is one rmw per enabled lane": 16 lanes on one address add 16; a [16] block with
      4 warps (each element held by several threads) adds 1 per element, not once per thread; a
      [16, 16] block whose 16 columns repeat 16 addresses adds 16 per address.
  A2  "`atomic_cas` compares bit patterns": memory -0.0, cmp +0.0 (loaded): no swap; memory NaN, cmp
      the same NaN bits: swap.
  A3  `>>` (SEMANTICS.md "Integer `//`, `%`, `<<`, `>>`"): `int8 >> uint8` is an arithmetic shift of the
      promoted `uint8` values (-128 >> 1 gives 0xC0); `0xF0 >> s` with `s : uint8` is logical (0x78).
"""
import os
import tempfile

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr6b-atom-")

import torch
import triton
import triton.language as tl

print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")


@triton.jit
def a1_kernel(c_ptr, d_ptr, e_ptr):
    r = tl.arange(0, 16)
    tl.atomic_add(c_ptr + r * 0, 1)                  # 16 lanes, one address
    tl.atomic_add(d_ptr + r, 1)                      # 16 lanes, 16 addresses, 4 warps
    cols = r[None, :] + r[:, None] * 0               # [16, 16]: each address 16 times
    tl.atomic_add(e_ptr + cols, tl.full([16, 16], 1, tl.int32))


def a1():
    c = torch.zeros(1, dtype=torch.int32, device="cuda")
    d = torch.zeros(16, dtype=torch.int32, device="cuda")
    e = torch.zeros(16, dtype=torch.int32, device="cuda")
    a1_kernel[(1,)](c, d, e, num_warps=4)
    torch.cuda.synchronize()
    print(f"A1 one address x16 lanes: {c.item()} (16); [16] block, 4 warps: {sorted(set(d.tolist()))} (1); "
          f"[16,16] repeating 16 addresses: {sorted(set(e.tolist()))} (16)")


@triton.jit
def a2_kernel(m_ptr, cmp_ptr, val_ptr, old_ptr):
    i = tl.program_id(0)
    old = tl.atomic_cas(m_ptr + i, tl.load(cmp_ptr + i), tl.load(val_ptr + i))
    tl.store(old_ptr + i, old)


def a2():
    nan_bits = 0x7fc12345
    m = torch.tensor([-0.0, 0.0], dtype=torch.float32, device="cuda")
    m.view(torch.int32)[1] = nan_bits
    cmpv = torch.tensor([0.0, 0.0], dtype=torch.float32, device="cuda")
    cmpv.view(torch.int32)[1] = nan_bits
    val = torch.tensor([5.0, 7.0], dtype=torch.float32, device="cuda")
    old = torch.zeros(2, dtype=torch.float32, device="cuda")
    a2_kernel[(2,)](m, cmpv, val, old)
    torch.cuda.synchronize()
    mb = [hex(x & 0xffffffff) for x in m.view(torch.int32).tolist()]
    print(f"A2 atomic_cas: memory -0.0 vs cmp +0.0 -> memory {mb[0]} (0x80000000 = no swap); "
          f"memory NaN vs the same NaN bits -> memory {mb[1]} (0x40e00000 = 7.0, swapped)")


@triton.jit
def a3_kernel(x_ptr, s_ptr, o_ptr):
    x = tl.load(x_ptr)      # int8 -128
    s = tl.load(s_ptr)      # uint8 1
    tl.store(o_ptr, (x >> s).to(tl.uint8))
    tl.store(o_ptr + 1, (0xF0 >> s).to(tl.uint8))


def a3():
    x = torch.tensor([-128], dtype=torch.int8, device="cuda")
    s = torch.tensor([1], dtype=torch.uint8, device="cuda")
    o = torch.zeros(2, dtype=torch.uint8, device="cuda")
    a3_kernel[(1,)](x, s, o)
    torch.cuda.synchronize()
    v = o.tolist()
    print(f"A3 int8 -128 >> uint8 1 = {hex(v[0])} (0xc0: arithmetic on uint8); 0xF0 >> uint8 1 = {hex(v[1])} (0x78: logical)")


for f in (a1, a2, a3):
    try:
        f()
    except Exception as ex:  # noqa: BLE001
        print(f"{f.__name__}: FAILED {type(ex).__name__}: {str(ex).strip().splitlines()[-1][:200]}")
