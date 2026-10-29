"""Which value a Python number takes in a comparison, in `tl.where` and in `tl.maximum`.

Library issue L45 (round 8). A comparison makes a Python number a tensor of its own type first
(`core.py` 900-1105, `to_tensor`), so the constant is not built in the comparison's dtype: with a
`float64` tensor, `x < 0.1` compares against the `float32` constant `0.1`, widened. `tl.where`
types its two values like an arithmetic operator, so its constant is built in the result's dtype.

Run: .venv/bin/python -I evidence/promotion/constants.py  (from TVA's root; CUDA_VISIBLE_DEVICES=0)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, out_ptr, f_ptr, MODE: tl.constexpr):
    offs = tl.arange(0, 4)
    x = tl.load(x_ptr + offs)
    if MODE == 0:
        tl.store(out_ptr + offs, (x < 0.1).to(tl.int32))
        tl.store(out_ptr + 4 + offs, (x == 0.1).to(tl.int32))
        tl.store(f_ptr + offs, tl.where(x > 0, 0.1, x))
    elif MODE == 1:
        tl.store(out_ptr + offs, (x == 1e-7).to(tl.int32))
    elif MODE == 2:
        tl.store(out_ptr + offs, (x < 2**31).to(tl.int32))
    else:
        m = tl.maximum(x, 0.1)
        tl.static_assert(m.dtype == tl.float32)
        tl.store(f_ptr + offs, m)


def lines(ck):
    return [l.strip().split(" loc(")[0] for l in ck.asm["source"].splitlines()
            if "arith.constant" in l or "arith.extf" in l or "arith.cmp" in l or "arith.bitcast" in l]


def main():
    dev = "cuda"
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    out = torch.zeros(8, dtype=torch.int32, device=dev)

    x = torch.full((4,), 0.1, dtype=torch.float64, device=dev)
    f64 = torch.zeros(4, dtype=torch.float64, device=dev)
    ck = k[(1,)](x, out, f64, MODE=0)
    print("\nK1: x float64 = 0.1 (the double)")
    print("  x < 0.1 :", out[:4].tolist(), " x == 0.1 :", out[4:].tolist())
    print("  tl.where(x > 0, 0.1, x) == the double 0.1:", bool((f64 == 0.1).all().item()))
    print("  front end:", lines(ck))

    xb = torch.full((4,), 1e-7, dtype=torch.float32, device=dev).to(torch.bfloat16)
    out.zero_()
    ck = k[(1,)](xb, out, f64, MODE=1)
    print(f"\nK2: x bfloat16 = {xb[0].item()!r}")
    print("  x == 1e-7 :", out[:4].tolist(), "(compared in float32 with float32(1e-7) =",
          repr(torch.tensor(1e-7, dtype=torch.float32).item()), ")")
    print("  front end:", lines(ck))

    xi = torch.tensor([-1, 0, 5, 2**31 - 1], dtype=torch.int32, device=dev)
    out.zero_()
    ck = k[(1,)](xi, out, f64, MODE=2)
    print("\nK3: x int32 =", xi.tolist())
    print("  x < 2**31 :", out[:4].tolist(), "(2**31 is uint32, so the comparison is unsigned)")
    print("  front end:", lines(ck))

    xh = torch.tensor([0.0, 0.0999755859375, 0.1, 1.0], dtype=torch.float16, device=dev)
    f32 = torch.zeros(4, dtype=torch.float32, device=dev)
    ck = k[(1,)](xh, out, f32, MODE=3)
    print("\nK4: x float16 =", xh.tolist())
    print("  tl.maximum(x, 0.1) (float32):", [repr(v) for v in f32.tolist()],
          "float32(0.1) =", repr(torch.tensor(0.1, dtype=torch.float32).item()))
    print("  front end:", lines(ck))


if __name__ == "__main__":
    main()
