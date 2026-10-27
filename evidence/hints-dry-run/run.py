"""Does `tl.multiple_of` on a loop-carried pointer reach the compiled code?

Tutorial 08's `grouped_matmul_kernel` (lines 93-169 of tutorials/08-grouped-gemm.py, verbatim)
is compiled twice with Triton 3.8.0: as written, and without the two hints (lines 148-149). The
script prints the `tt.divisibility` attributes in the TTIR and the global-load instructions in
the PTX. Run: ~/fpv/tva/.venv/bin/python -I run.py
"""
import re, sys, torch, triton, triton.language as tl

@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


@triton.jit
def grouped_matmul_kernel_nohint(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def build(kernel, BM, BN, BK, M, N, K, lda, ldb, ldc):
    A = torch.randn(M * lda + 64, device="cuda", dtype=torch.float16)
    B = torch.randn(K * ldb + 64, device="cuda", dtype=torch.float16)
    C = torch.zeros(M * ldc + 64, device="cuda", dtype=torch.float16)
    a_ptrs = torch.tensor([A.data_ptr()], device="cuda", dtype=torch.int64)
    b_ptrs = torch.tensor([B.data_ptr()], device="cuda", dtype=torch.int64)
    c_ptrs = torch.tensor([C.data_ptr()], device="cuda", dtype=torch.int64)
    sizes = torch.tensor([M, N, K], device="cuda", dtype=torch.int32)
    lds = torch.tensor([lda, ldb, ldc], device="cuda", dtype=torch.int32)
    k = kernel[(4,)](a_ptrs, b_ptrs, c_ptrs, sizes, lds, 1, NUM_SM=4, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
    torch.cuda.synchronize()
    ref = (A[: M * lda].view(M, lda)[:, :K].float() @ B[: K * ldb].view(K, ldb)[:, :N].float())
    out = C[: M * ldc].view(M, ldc)[:, :N].float()
    return k, (out - ref).abs().max().item()


def census(k, name):
    ttir, ptx = k.asm["ttir"], k.asm["ptx"]
    print(f"== {name}")
    for l in ttir.splitlines():
        if "tt.divisibility" in l and ("addptr" in l or "scf.for" in l or "tt.func" in l):
            print("  ttir:", re.sub(r"\s+", " ", l.strip())[:200])
    loads = {}
    for m in re.finditer(r"^\s*(ld\.global[\w.]*|cp\.async[\w.]*)\s", ptx, re.M):
        loads[m.group(1)] = loads.get(m.group(1), 0) + 1
    print("  ptx loads:", dict(sorted(loads.items())))


if __name__ == "__main__":
    print("triton", triton.__version__, "|", torch.cuda.get_device_name(0))
    mode = sys.argv[1] if len(sys.argv) > 1 else "census"
    if mode == "census":
        for kern, name in ((grouped_matmul_kernel, "as written"), (grouped_matmul_kernel_nohint, "without lines 148-149")):
            k, err = build(kern, 64, 64, 32, 64, 64, 64, 64, 64, 64)
            census(k, name)
            print(f"  aligned run (lda = ldb = ldc = 64): max |C - A@B| = {err:.3g}")
    else:  # one misaligned run, in its own process (a fault poisons the CUDA context)
        kern = grouped_matmul_kernel if mode == "misaligned" else grouped_matmul_kernel_nohint
        try:
            k, err = build(kern, 64, 64, 32, 64, 64, 32, 36, 64, 64)
            print(f"{mode}: lda = 36: completed, max |C - A@B| = {err:.3g}")
        except Exception as e:
            print(f"{mode}: lda = 36: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
