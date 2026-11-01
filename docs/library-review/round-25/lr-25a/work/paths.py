"""lr-25a: which instruction path Triton 3.8.0 picks for a tl.dot on sm_90, at small and unusual block shapes
and num_warps, for float16, bfloat16 and fp8 operands (the facts DotF16, DotBF16 and DotE4m3Chunk, ... are
stated for every M, N; Triton's min_dot_size allows M, N down to 1). Prints the path read from the PTX."""
import re
import sys
import torch
import triton
import triton.language as tl

@triton.jit
def kdot(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c))

def path(ptx):
    if "wgmma.mma_async" in ptx:
        return "wgmma " + ",".join(sorted(set(re.findall(r"wgmma\.mma_async\.\w+\.\w+\.(m\d+n\d+k\d+)", ptx))))
    if "mma.sync" in ptx:
        return "mma.sync " + ",".join(sorted(set(re.findall(r"mma\.sync\.aligned\.(m\d+n\d+k\d+)", ptx))))
    n = len(re.findall(r"\bfma\.rn\.f32", ptx))
    return f"no mma ({n} fma.rn.f32)"

DT = {"f16": torch.float16, "bf16": torch.bfloat16, "e5": torch.float8_e5m2, "e4": torch.float8_e4m3fn}
for dt in ["f16", "bf16", "e5", "e4"]:
    K = 32 if dt in ("e5", "e4") else 16
    for (M, N, nw) in [(1, 16, 1), (2, 16, 1), (4, 16, 1), (8, 16, 1), (16, 1, 1), (16, 2, 1), (16, 4, 1), (16, 8, 1),
                       (8, 8, 1), (1, 1, 1), (16, 16, 1), (16, 16, 8), (32, 32, 2), (32, 128, 4), (64, 64, 2),
                       (64, 8, 4), (64, 16, 4), (64, 64, 4), (64, 256, 4), (64, 64, 8), (128, 256, 8), (256, 64, 8),
                       (256, 128, 8)]:
        try:
            a = torch.zeros(M, K, device="cuda").to(DT[dt]); b = torch.zeros(K, N, device="cuda").to(DT[dt])
            c = torch.zeros(M, N, device="cuda"); o = torch.empty(M, N, device="cuda")
            h = kdot[(1,)](a, b, c, o, M, N, K, num_warps=nw)
            print(dt, M, N, K, nw, path(h.asm["ptx"]), flush=True)
        except Exception as e:
            print(dt, M, N, K, nw, "rejected:", type(e).__name__, str(e).splitlines()[0][:100], flush=True)
