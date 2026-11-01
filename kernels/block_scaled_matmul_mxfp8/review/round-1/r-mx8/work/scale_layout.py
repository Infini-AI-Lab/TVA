"""r-mx8, round 1: which scale byte meets which element, three ways, on the CPU.

1. The host's reference (source lines 411-417): `unpack_scale` of the `[R // 128, K // 128, 32, 16]`
   scales, `repeat_interleave(32, dim=1)`.
2. The kernel's path (lines 223-227): the `[1, R // 128, K // 128, 2, 256]` view (lines 401-402),
   the block `[1, rep, 1, 2, 256]` at `[0, q * rep, t, 0, 0]` (zero padding outside), then
   `reshape(rep, 1, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rep * 128, 4)`, entry `(j, k // 32)`
   for element `(q * rep * 128 + j, 128 t + k)`.
3. The claim's `sIdx K r k` (Defs.lean).

All three must give the same byte for every element inside the matrix.
"""
import torch


def sidx(K, r, k):
    return (r // 128 * (K // 128) + k // 128) * 512 + 16 * (r % 32) + 4 * (r % 128 // 32) + k % 128 // 32


def unpack_scale(packed):  # source lines 411-414, verbatim
    packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
    num_chunk_m, num_chunk_k, _, _, _ = packed.shape
    return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()


def check(R, K, rep, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (R // 128, K // 128, 32, 16), generator=g, dtype=torch.int64)
    flat = packed.reshape(-1)
    # 1. host reference
    ref = unpack_scale(packed).repeat_interleave(32, dim=1)[:R, :K]
    # 3. sIdx
    rr = torch.arange(R)[:, None]
    kk = torch.arange(K)[None, :]
    idx = (rr // 128 * (K // 128) + kk // 128) * 512 + 16 * (rr % 32) + 4 * (rr % 128 // 32) + kk % 128 // 32
    by_sidx = flat[idx]
    ok_ref = bool((ref == by_sidx).all())
    # 2. kernel path
    view = packed.reshape(1, R // 128, K // 128, 2, 256)
    nblk = (R + 128 * rep - 1) // (128 * rep)
    ker = torch.full((nblk * rep * 128, K), -1, dtype=torch.int64)
    for q in range(nblk):
        for t in range(K // 128):
            blk = torch.zeros((1, rep, 1, 2, 256), dtype=torch.int64)  # padding "zero"
            hi = min((q + 1) * rep, R // 128)
            n_in = hi - q * rep
            blk[:, :n_in] = view[:, q * rep:hi, t:t + 1]
            s = blk.reshape(rep, 1, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rep * 128, 4)
            rows = slice(q * rep * 128, (q + 1) * rep * 128)
            ker[rows, 128 * t:128 * (t + 1)] = s.repeat_interleave(32, dim=1)
    ok_ker = bool((ker[:R] == by_sidx).all())
    return ok_ref, ok_ker


if __name__ == "__main__":
    for R, K, rep in [(128, 128, 1), (256, 384, 1), (640, 512, 1), (256, 256, 2), (384, 640, 2), (1152, 256, 2)]:
        ok_ref, ok_ker = check(R, K, rep, seed=R * 7 + K + rep)
        print(f"R={R:5d} K={K:4d} rep={rep}: host reference == sIdx: {ok_ref}; kernel path == sIdx: {ok_ker}")
