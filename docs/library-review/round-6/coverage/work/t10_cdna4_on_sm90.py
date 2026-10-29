"""Does tutorial 10's `block_scaled_matmul_kernel_cdna4` (written for AMD CDNA4) run on the H200?

Its host passes `matrix_instr_nonkdim`, an AMD option. The CUDA backend keeps only `CUDAOptions`
fields (`parse_options`), so on NVIDIA the option is dropped silently. This script runs the
tutorial's own AMD host path (`validate_block_scaled_amd` pieces) on sm_90, with the option (as the
tutorial passes it) and without it, and compares with the tutorial's torch reference.

Usage: CUDA_VISIBLE_DEVICES=0 ~/fpv/tva/.venv/bin/python -I t10_cdna4_on_sm90.py <tutorials dir>
"""
import importlib.util
import os
import sys

import torch

sys.dont_write_bytecode = True  # leave no cache in the tutorials folder

tut = os.path.join(sys.argv[1], "10-block-scaled-matmul.py")
spec = importlib.util.spec_from_file_location("t10", tut)
t10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t10)


def e8m0_to_f32(x):
    x_f32 = 2**((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales):
    x_f32 = x.to(torch.float32) * e8m0_to_f32(x_scales.repeat_interleave(32, dim=1).to(torch.float32))
    w_f32 = w.to(torch.float32) * e8m0_to_f32(w_scales.repeat_interleave(32, dim=1).to(torch.float32))
    return torch.mm(x_f32, w_f32.T)


M = N = K = 512
for nonk in [16, 32]:
    for with_opt in [True, False]:
        try:
            x4, w4, xs, ws, xs_t, ws_t, cfg = t10.initialize_block_scaled_amd(M, N, K, nonk)
            x = x4.to_packed_tensor(dim=1)
            w = w4.to_packed_tensor(dim=1).T
            out = torch.empty((M, N), device="cuda")
            kw = {"matrix_instr_nonkdim": nonk} if with_opt else {}
            grid = (((M + cfg["BLOCK_M"] - 1) // cfg["BLOCK_M"]) * ((N + cfg["BLOCK_N"] - 1) // cfg["BLOCK_N"]), 1)
            t10.block_scaled_matmul_kernel_cdna4[grid](
                x, w, out, xs_t, ws_t, M, N, x.shape[1], x.stride(0), x.stride(1), w.stride(0), w.stride(1), 0,
                out.stride(0), out.stride(1), xs_t.stride(0), xs_t.stride(1), ws_t.stride(0), ws_t.stride(1),
                cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"], nonk, num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"], **kw)
            torch.cuda.synchronize()
            ref = run_torch(x4, w4, xs, ws)
            err = (ref - out).abs().max().item()
            print(f"mfma_nonkdim={nonk} option={'given' if with_opt else 'absent'}: ran, max abs err {err:.3e}")
        except Exception as e:
            print(f"mfma_nonkdim={nonk} option={'given' if with_opt else 'absent'}: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:300]}")
