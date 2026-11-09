"""Which formats of tutorial 10's `block_scaled_matmul_kernel` run on the H200 (library issue L153).

For each `--format` the tutorial accepts (mxfp4, nvfp4, mxfp8, mixed), at M = N = K = 1024 and at
the tutorial's own validation size 8192^3 for `mixed`, this builds the tutorial's inputs and host
descriptors with its own `initialize_block_scaled` (with the reference), launches its own
`block_scaled_matmul` (the kernel, its config, `torch.float16` output), and prints whether the launch
ran or was rejected on the host (the exception's type and first line), and for a launch that ran the
largest absolute error against the tutorial's reference and whether the tutorial's own check
(`torch.testing.assert_close`, atol = rtol = 1e-3) passes. The tutorial file is imported, not edited.
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/tut10-formats/run.py
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TUT = os.path.join(HERE, "..", "..", "tutorials", "10-block-scaled-matmul.py")


def load():
    spec = importlib.util.spec_from_file_location("tut10", TUT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one(t, fmt, M, N, K):
    torch.manual_seed(42)
    try:
        res = t.initialize_block_scaled(M, N, K, fmt, compute_reference=True)
        a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, ref = res[:9]
        out = t.block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K,
                                    rep_m, rep_n, rep_k, configs)
        torch.cuda.synchronize()
    except Exception as e:  # a rejection on the host
        msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else ""
        return f"{fmt} {M}x{N}x{K}: rejected on the host: {type(e).__name__}: {msg}"
    err = (ref - out.to(torch.float32)).abs().max().item()
    try:
        torch.testing.assert_close(ref, out.to(torch.float32), atol=1e-3, rtol=1e-3)
        ok = "passes"
    except AssertionError:
        ok = "fails"
    kname = ""
    try:
        kname = ", BLOCK_M/N/K = %d/%d/%d" % (configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"])
    except Exception:
        pass
    return (f"{fmt} {M}x{N}x{K}: ran{kname}; max |ref - out| = {err:.6g}; "
            f"the tutorial's assert_close (1e-3, 1e-3) {ok}; output finite: {bool(torch.isfinite(out).all())}")


def main():
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0),
          "capability", torch.cuda.get_device_capability(0))
    import triton
    print("triton", triton.__version__)
    t = load()
    for fmt in ("mxfp4", "nvfp4", "mxfp8", "mixed"):
        print(one(t, fmt, 1024, 1024, 1024), flush=True)
    print(one(t, "mixed", 8192, 8192, 8192), flush=True)


if __name__ == "__main__":
    sys.exit(main())
