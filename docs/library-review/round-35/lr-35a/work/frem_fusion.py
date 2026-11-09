"""lr-35a (round 35, L163): is a float32 `%` (not a product in SEMANTICS.md) fused into a following
addition? Launch 1 stores x % y; launch 2 stores x % y + c (c near -(x % y)); counted: results that
are not fl(q + c). Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import torch, triton, triton.language as tl


@triton.jit
def k(x_ptr, y_ptr, c_ptr, q_ptr, ADD: tl.constexpr, N: tl.constexpr):
    o = tl.arange(0, N)
    x = tl.load(x_ptr + o)
    y = tl.load(y_ptr + o)
    q = x % y
    if ADD:
        q = q + tl.load(c_ptr + o)
    tl.store(q_ptr + o, q)


N = 4096
torch.manual_seed(1)
for name, x, y in [("moderate", torch.rand(N, device="cuda") * 100, torch.rand(N, device="cuda") * 3 + 0.1),
                   ("wide", torch.exp2(torch.rand(N, device="cuda") * 40 - 10), torch.exp2(torch.rand(N, device="cuda") * 20 - 10))]:
    q = torch.empty_like(x)
    k[(1,)](x, y, x, q, ADD=False, N=N)
    ulp = torch.abs(q) * 2.0 ** -23 + 1e-45
    c = -(q + torch.randint(-3, 4, (N,), device="cuda").float() * ulp)
    r = torch.empty_like(x)
    h = k[(1,)](x, y, c, r, ADD=True, N=N)
    ieee = (q.double() + c.double()).float()
    fin = torch.isfinite(r) & torch.isfinite(ieee)
    print(f"x % y + c, {name}: {int(((r != ieee) & fin).sum())} of {N} not the IEEE sum")
