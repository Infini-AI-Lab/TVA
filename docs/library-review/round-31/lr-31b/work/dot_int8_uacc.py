"""lr-31b: a `uint32` accumulator of an int8 `tl.dot` (library issue L141's SEMANTICS text says an
accumulator must be `int32` or the kernel is rejected on the host).

  U1  acc = tl.load of a uint32 buffer (values >= 2^31 included), tl.dot(a, b, acc) and
      tl.dot(a, b, acc, out_dtype=tl.uint32): accepted? the result's dtype (static_print), and the
      values stored: as acc's bits read as int32 plus the exact sum, or as the uint32 value plus the sum
  U2  the result compared with 0 and converted to float32 in the kernel (where int32 and uint32 differ)

Run from TVA's root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(a_ptr, b_ptr, c0_ptr, c_ptr, neg_ptr, f_ptr, OUT: tl.constexpr):
    r = tl.arange(0, 16)
    rk = tl.arange(0, 32)
    a = tl.load(a_ptr + r[:, None] * 32 + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * 16 + r[None, :])
    acc = tl.load(c0_ptr + r[:, None] * 16 + r[None, :])
    tl.static_assert(acc.dtype == tl.uint32)
    if OUT:
        c = tl.dot(a, b, acc, out_dtype=tl.uint32)
    else:
        c = tl.dot(a, b, acc)
    tl.static_print("RESULT", c.dtype)
    tl.store(c_ptr + r[:, None] * 16 + r[None, :], c)
    tl.store(neg_ptr + r[:, None] * 16 + r[None, :], (c < 0).to(tl.int32))
    tl.store(f_ptr + r[:, None] * 16 + r[None, :], c.to(tl.float32))


def main():
    g = torch.Generator(device="cuda").manual_seed(317)
    a = torch.randint(-4, 5, (16, 32), dtype=torch.int8, device="cuda", generator=g)
    b = torch.randint(-4, 5, (32, 16), dtype=torch.int8, device="cuda", generator=g)
    # uint32 accumulator values, half of them >= 2^31 (torch has uint32; build from int64)
    vals = torch.randint(0, 2**32 - 1, (16, 16), dtype=torch.int64, device="cuda", generator=g)
    vals[0, :] = 3_000_000_000
    vals[1, :] = 5
    c0 = vals.to(torch.uint32) if hasattr(torch, "uint32") else None
    s = (a.to(torch.float64) @ b.to(torch.float64)).to(torch.int64)
    as_i32 = torch.where(vals >= 2**31, vals - 2**32, vals) + s  # acc's bits read as int32, plus the sum
    as_u32 = vals + s
    for out in (False, True):
        c = torch.empty(16, 16, dtype=torch.int32, device="cuda")
        neg = torch.empty(16, 16, dtype=torch.int32, device="cuda")
        f = torch.empty(16, 16, dtype=torch.float32, device="cuda")
        k[(1,)](a, b, c0, c, neg, f, OUT=out, num_warps=1)
        torch.cuda.synchronize()
        ci = c.to(torch.int64)
        print(f"U1 out_dtype={'tl.uint32' if out else 'omitted'}: accepted; values equal acc's bits read as int32 "
              f"plus the sum on {int((ci == as_i32).sum())}/256 (also equal mod 2^32 to the uint32 value plus the "
              f"sum on {int(((ci - as_u32) % 2**32 == 0).sum())}/256); row 0 (acc = 3000000000): stored "
              f"{int(c[0, 0])}, as uint32 + sum {int(as_u32[0, 0])}")
        print(f"U2 out_dtype={'tl.uint32' if out else 'omitted'}: (c < 0) on row 0: {int(neg[0, 0])}; "
              f"c.to(tl.float32) on row 0: {float(f[0, 0])} (int32 reading {float(as_i32[0, 0])}, uint32 reading "
              f"{float(as_u32[0, 0])})")


if __name__ == "__main__":
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    main()
