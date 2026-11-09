"""lr-34b: as div_fused.py, for `tl.div_rn` on float32 and `/` on float64 (both IEEE in the library):
does `x / y - q` stay exactly 0 (q = the same quotient from an earlier launch)?"""
import torch
import triton
import triton.language as tl


@triton.jit
def kq(x_ptr, y_ptr, q_ptr, RN: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(x_ptr + j)
    y = tl.load(y_ptr + j)
    tl.store(q_ptr + j, tl.div_rn(x, y) if RN else x / y)


@triton.jit
def kr(x_ptr, y_ptr, q_ptr, r_ptr, RN: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(x_ptr + j)
    y = tl.load(y_ptr + j)
    tl.store(r_ptr + j, (tl.div_rn(x, y) if RN else x / y) - tl.load(q_ptr + j))


print(f"Triton {triton.__version__}, {torch.cuda.get_device_name()}")
torch.manual_seed(2)
N = 1 << 16
for dt, rn in ((torch.float32, True), (torch.float64, False)):  # tl.div_rn is float32 only
    x = torch.rand(N, device="cuda", dtype=dt) + 0.5
    y = torch.rand(N, device="cuda", dtype=dt) + 0.5
    q = torch.empty_like(x)
    r = torch.empty_like(x)
    kq[(1,)](x, y, q, rn, N)
    kr[(1,)](x, y, q, r, rn, N)
    torch.cuda.synchronize()
    print(f"{dt} {'div_rn' if rn else '/'}: x/y - q nonzero on {(r != 0).sum().item()} of {N}")
