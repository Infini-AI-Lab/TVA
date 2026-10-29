"""Does tutorial 10's `block_scaled_matmul_kernel` run on the H200 (sm_90)?

The tutorial's own main refuses to run without block-scaling hardware (sm_100/110 or CDNA4).
TVA's facts are for sm_90. This script launches the kernel on sm_90 through the tutorial's own
host code, for each of its formats, and compares with the tutorial's reference.

Usage: CUDA_VISIBLE_DEVICES=0 ~/fpv/tva/.venv/bin/python -I t10_on_sm90.py <tutorials dir>
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

print("capability", torch.cuda.get_device_capability())
M, N, K = 256, 256, 512
for fmt in ["mxfp8", "mixed", "mxfp4", "nvfp4"]:
    try:
        res = t10.initialize_block_scaled(M, N, K, fmt, compute_reference=True)
        a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference = res[:9]
        out = t10.block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m,
                                      rep_n, rep_k, configs)
        torch.cuda.synchronize()
        err = (reference - out.to(torch.float32)).abs().max().item()
        ok = torch.allclose(reference, out.to(torch.float32), atol=1e-3, rtol=1e-3)
        print(f"{fmt}: ran, max abs err vs reference {err:.3e}, max |reference| "
              f"{reference.abs().max().item():.3e}, tutorial's tolerance (atol=rtol=1e-3) met: {ok}")
    except Exception as e:  # report and go on
        print(f"{fmt}: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
