"""Reviewer r-mx8b: which scale byte meets which group of 32 elements (CPU, no GPU).

Simulates, with torch on the CPU, the kernel's scale path (source lines 223-227: the 5-D block
loads of the [1, R/128, K/128, 2, 256] view, reshape, trans(0, 3, 2, 1, 4), reshape) and compares
it with (a) the host's reference unpacking (unpack_scale, lines 411-417) and (b) the claim's
`sIdx K r k` (Defs.lean). torch's permute has Triton's trans(dims) meaning: dimension k of the
result is dimension dims[k] of the input (SEMANTICS.md, Shape changes).
"""
import torch

torch.manual_seed(3)


def unpack_scale(packed):  # lines 411-414, verbatim
    packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
    num_chunk_m, num_chunk_k, _, _, _ = packed.shape
    return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()


def sIdx(K, r, k):  # Defs.lean
    return (r // 128 * (K // 128) + k // 128) * 512 + 16 * (r % 32) + 4 * (r % 128 // 32) + k % 128 // 32


def check(R, K, rep):
    """rep = rep_m (1, BLOCK_M = 128) for a, rep_n (2, BLOCK_N = 256) for b."""
    sc = torch.randint(0, 256, (R // 128, K // 128, 32, 16), dtype=torch.uint8)  # line 370/371 layout
    view = sc.reshape(1, R // 128, K // 128, 2, 256)  # lines 401-402
    flat = sc.flatten()
    ref = unpack_scale(sc)  # [R, K/32]; column g = k // 32 (repeat_interleave(32), line 416)
    BL = 128 * rep
    bad_ref = bad_idx = 0
    n_tiles = (R + BL - 1) // BL
    for p in range(n_tiles):
        for t in range(K // 128):
            # the descriptor load: block [1, rep, 1, 2, 256] at [0, p*rep, t, 0, 0], zero padding
            blk = torch.zeros((1, rep, 1, 2, 256), dtype=torch.uint8)
            hi = min(p * rep + rep, R // 128)
            blk[0, :hi - p * rep, 0] = view[0, p * rep:hi, t]
            s = blk.reshape(rep, 1, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(BL, 4)  # lines 226/227
            for r in range(BL):
                row = p * BL + r
                if row >= R:
                    continue  # padding rows: never meet a stored element
                for kk in range(128):
                    k = 128 * t + kk
                    v = int(s[r, kk // 32])  # tl.dot_scaled: scale of (r, k) is column k // group
                    bad_ref += v != int(ref[row, k // 32])
                    bad_idx += v != int(flat[sIdx(K, row, k)])
    return bad_ref, bad_idx


for name, R, K, rep in [("a", 256, 384, 1), ("b", 512, 256, 2), ("b, R/128 odd", 384, 256, 2)]:
    print(name, "R =", R, "K =", K, "mismatches vs unpack_scale, vs sIdx:", check(R, K, rep))
