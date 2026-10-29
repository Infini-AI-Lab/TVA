"""lr-14a: are the approximate math functions rejected on the host for float16 and bfloat16
(L66's corrected "What the library does not state")? Run: CUDA_VISIBLE_DEVICES=0 python -I math_f16.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, o_ptr, FN: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    if FN == 0:
        y = tl.exp(x)
    elif FN == 1:
        y = tl.math.exp2(x)
    elif FN == 2:
        y = tl.log(x)
    elif FN == 3:
        y = tl.sqrt(x)
    elif FN == 4:
        y = tl.math.rsqrt(x)
    elif FN == 5:
        y = tl.sin(x)
    else:
        y = tl.erf(x)
    tl.store(o_ptr + offs, y)


print(f"Triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
print("| function | float16 | bfloat16 |")
print("|---|---|---|")
for fn, name in enumerate(["tl.exp", "tl.math.exp2", "tl.log", "tl.sqrt", "tl.math.rsqrt", "tl.sin", "tl.erf"]):
    row = [name]
    for dt in (torch.float16, torch.bfloat16):
        x = torch.ones(16, dtype=dt, device="cuda"); o = torch.empty_like(x)
        try:
            k[(1,)](x, o, FN=fn); torch.cuda.synchronize()
            row.append("compiles and runs")
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else ""
            row.append(f"rejected: {type(e).__name__} ({msg[:70]})")
    print("| " + " | ".join(row) + " |")
