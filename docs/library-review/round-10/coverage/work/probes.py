"""Coverage round 10: three probes on the H200 (Triton 3.8.0), run from this folder.

P1  tutorial 10's block_scaled_matmul_kernel through the tutorial's own host code
    (initialize_block_scaled, block_scaled_matmul) for each format: does the launch run on
    sm_90, and how far is it from the tutorial's reference?
P2  Python's builtins in a kernel (`float`, `isinstance`, as tutorials 02 and 06 use them): is the
    result a compile-time value (accepted by tl.static_assert), and which value is stored?
P3  tutorial 02's `softmax_kernel.warmup(..., num_stages=..., num_warps=8)`: does `num_stages`
    reach the compile options, or only the kernel's constexpr parameter of that name?

Paths are printed relative to the tutorials folder only.
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True  # loading a tutorial must leave no cache in tutorials/

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

HERE = os.path.dirname(os.path.abspath(__file__))
TUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "tutorials"))


def load_tutorial(name, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(TUT, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short(e):
    import re
    s = f"{type(e).__name__}: {e}"
    s = re.sub(r"(/[^\s:'\"]+)+/", "", s)  # keep file names, drop the folders
    s = " ".join(s.split())
    return s[:240]


def p1():
    print("== P1: tutorial 10 block_scaled_matmul_kernel on sm_90 ==")
    print("capability", torch.cuda.get_device_capability())
    t10 = load_tutorial("10-block-scaled-matmul.py", "t10")
    print("supports_block_scaling() =", t10.supports_block_scaling())
    M = N = 512
    K = 512
    for fmt in ["mxfp8", "mixed", "mxfp4", "nvfp4"]:
        torch.manual_seed(42)
        try:
            res = t10.initialize_block_scaled(M, N, K, fmt, compute_reference=True)
            a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference = res[:9]
            out = t10.block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K,
                                          rep_m, rep_n, rep_k, configs)
            torch.cuda.synchronize()
            err = (reference - out.to(torch.float32)).abs().max().item()
            ok = torch.allclose(reference, out.to(torch.float32), atol=1e-3, rtol=1e-3)
            print(f"{fmt:6s} configs BLOCK_K={configs['BLOCK_SIZE_K']} VEC={configs['VEC_SIZE']} "
                  f"num_stages={configs['num_stages']}: RAN, max |ref - out| = {err:.3e}, "
                  f"tutorial tolerance met: {ok}")
        except Exception as e:  # a launch rejected on the host
            print(f"{fmt:6s}: REJECTED on the host before any program ran: {short(e)}")


@triton.jit
def builtins_kernel(out_ptr, desc_or_ptr):
    # `isinstance` and `float` are Python's own, evaluated at compile time:
    tl.static_assert(isinstance(out_ptr, tl.tensor))
    tl.static_assert(float("inf") > 1e308)
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        tl.store(out_ptr, 1.0)
    else:
        tl.store(out_ptr, 2.0)
    tl.store(out_ptr + 1, -float('inf'))
    tl.store(out_ptr + 2, tl.zeros([1], dtype=tl.float32).sum(0) - float("inf"))


def p2():
    print("== P2: Python builtins in a kernel ==")
    out = torch.zeros(3, device="cuda", dtype=torch.float32)
    x = torch.zeros((16, 16), device="cuda", dtype=torch.float16)
    builtins_kernel[(1, )](out, x)
    torch.cuda.synchronize()
    print("pointer argument   :", out.tolist())
    out.zero_()
    d = TensorDescriptor.from_tensor(x, [16, 16])
    builtins_kernel[(1, )](out, d)
    torch.cuda.synchronize()
    print("host TensorDescriptor:", out.tolist())
    k = builtins_kernel.warmup(out, x, grid=(1, ))
    ttir = k.asm["ttir"]
    lines = [l.strip() for l in ttir.splitlines() if "tt.store" in l or "arith.constant" in l]
    print("TTIR constants and stores (pointer argument):")
    for l in lines:
        print("   ", l.split(" loc(")[0])


def p3():
    print("== P3: tutorial 02's warm-up keyword num_stages ==")

    @triton.jit
    def k_ns(x_ptr, n, BLOCK: tl.constexpr, num_stages: tl.constexpr):
        for i in tl.range(0, n, 1, num_stages=num_stages):
            tl.store(x_ptr + i + tl.arange(0, BLOCK), tl.zeros([BLOCK], dtype=tl.float32))

    x = torch.zeros(64, device="cuda")
    k = k_ns.warmup(x, 4, BLOCK=16, num_stages=4, num_warps=8, grid=(1, ))
    print("warm-up(num_stages=4, num_warps=8): compiled metadata num_stages =", k.metadata.num_stages,
          ", num_warps =", k.metadata.num_warps)
    k2 = k_ns.warmup(x, 4, BLOCK=16, num_stages=2, num_warps=4, grid=(1, ))
    print("warm-up(num_stages=2, num_warps=4): compiled metadata num_stages =", k2.metadata.num_stages,
          ", num_warps =", k2.metadata.num_warps)
    print("constexprs of the compiled kernel:", {kk: v for kk, v in k.src.constants.items()})


if __name__ == "__main__":
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    for f in (p1, p2, p3):
        try:
            f()
        except Exception:
            print("probe failed:", short(sys.exc_info()[1]))
