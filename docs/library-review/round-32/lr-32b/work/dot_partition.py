"""lr-32b, round 32, L150: which k each mma.sync instruction of a float16 tl.dot adds, when its `a`
operand is converted in registers from float8e5 or int8 (kWidth = 4 in the TTGIR), and when it is
not (kWidth = 2).

Probe: product 1 at k = BIG, every other k of the row a product 2^-27 (float16 a = 2^-14, b = 2^-13;
int8 source: a = 64, b = 2^15 at BIG, a = 1, b = 2^-6 elsewhere). Ks already found are zeroed (a zero
product is no term). The result is above the big product iff some earlier instruction held 16 small
products and no big one, so scanning BIG over the remaining k finds the first instruction's k: those
with the plain result. Repeating with them zeroed finds the next.

Run from TVA's root: CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I docs/library-review/round-32/lr-32b/work/dot_partition.py
"""
import os, tempfile
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="lr32b-"))
import torch, triton, triton.language as tl


@triton.jit
def kdot(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, SRC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    if SRC == 1:
        a = a.to(tl.float16)
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))


def once(src, M, N, K, nw, big, zero):
    if src == "int8":
        a = torch.full((M, K), 1.0); b = torch.full((K, N), 2.0 ** -6)
        a[:, big] = 64.0; b[big, :] = 2.0 ** 15
        base = 2.0 ** 21
    else:
        a = torch.full((M, K), 2.0 ** -14); b = torch.full((K, N), 2.0 ** -13)
        a[:, big] = 1.0; b[big, :] = 1.0
        base = 1.0
    for z in zero:
        a[:, z] = 0.0
    if src == "fp8e5":
        ad = a.to(torch.float8_e5m2).cuda(); S = 1
    elif src == "int8":
        ad = a.to(torch.int8).cuda(); S = 1
    else:
        ad = a.to(torch.float16).cuda(); S = 0
    bd = b.to(torch.float16).cuda()
    o = torch.empty(M, N, dtype=torch.float32, device="cuda")
    kdot[(1,)](ad, bd, o, M, N, K, S, num_warps=nw)
    vals = set(o.flatten().tolist())
    assert len(vals) == 1, vals
    v = vals.pop()
    return v == base


def partition(src, M, N, K, nw):
    left = list(range(K)); zero = []; parts = []
    while left:
        first = [k for k in left if once(src, M, N, K, nw, k, zero)]
        if len(first) == len(left) or not first:
            parts.append(left); break
        parts.append(first)
        zero += first
        left = [k for k in left if k not in first]
    return parts


if __name__ == "__main__":
    for src in ["fp16", "fp8e5", "int8"]:
        for (M, N, nw, K) in [(16, 16, 1, 64), (32, 32, 2, 64), (16, 64, 1, 128)]:
            p = partition(src, M, N, K, nw)
            consec = all(sorted(c) == list(range(c[0], c[0] + 16)) and c[0] % 16 == 0 for c in p)
            print(f"src={src} M={M} N={N} K={K} warps={nw}: instructions in order: {p}; "
                  f"16 consecutive k in increasing order: {consec}", flush=True)
