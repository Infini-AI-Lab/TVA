"""Options and arguments of a precompiled kernel (`CompiledKernel`).

Library evidence for library issues L72 and L74 (round 11). A `CompiledKernel` is compiled by
`kernel.warmup(…)` (`primitives/triton-3.8.0-jit.py` lines 362-363: the same keywords as a launch) or
by `triton.compile(src, target, options)` (`primitives/triton-3.8.0-compiler.py` lines 226-243), and
its launch `compiled[grid](…)` takes no option. A host tensor descriptor is specialised by its dtype
and block shape (`primitives/triton-3.8.0-specialize.cc` lines 165-226), and the launcher builds the
TMA descriptor with the compiled kernel's block shape and element type, not the argument's
(`primitives/triton-3.8.0-nvidia-driver.py` lines 209-264). Each case runs on GPU 0 in its own
process.

  O1  a JIT launch with the keyword enable_reflect_ftz=False: tl.math.exp2(-130) (2^-130 without
      flush to zero, 0 with it, the default)
  O2  k.warmup(…, enable_reflect_ftz=False), then compiled[grid](…) with no keyword
  O3  k.warmup(…) with the defaults, then compiled[grid](…) (control)
  O4  triton.compile(ASTSource(k, …), options={"enable_reflect_ftz": False}), then compiled[grid](…)
  O5  triton.compile for another target (GPUTarget("cuda", 80, 32)), launched on the H200
  O6  a kernel copying its host TensorDescriptor's block, warmed up with a [32, 32] block, then
      launched through the CompiledKernel with a [16, 16] descriptor: which block is copied?
  O7  the same kernel through the JIT with the [16, 16] descriptor (control: recompiled)

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/compiled-options/run.py  (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def k_exp2(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.math.exp2(tl.load(x_ptr + offs, mask=m)), mask=m)


@triton.jit
def k_desc(xd, y_ptr):
    BM: tl.constexpr = xd.block_shape[0]
    BN: tl.constexpr = xd.block_shape[1]
    v = xd.load([0, 0])
    tl.store(y_ptr + tl.arange(0, BM)[:, None] * 32 + tl.arange(0, BN)[None, :], v)


CASES = {
    "O1": "JIT launch with enable_reflect_ftz=False",
    "O2": "warmup with enable_reflect_ftz=False, then the CompiledKernel with no keyword",
    "O3": "warmup with the defaults, then the CompiledKernel (control)",
    "O4": "triton.compile with options enable_reflect_ftz=False, then the CompiledKernel",
    "O5": "triton.compile for target cuda sm_80, launched on the H200",
    "O6": "CompiledKernel warmed up with a [32, 32] descriptor, launched with a [16, 16] one",
    "O7": "the JIT with the [16, 16] descriptor (control)",
}


def run_case(name):
    dev = "cuda"
    if name in ("O1", "O2", "O3", "O4", "O5"):
        x = torch.full((16,), -130.0, device=dev, dtype=torch.float32)
        y = torch.full_like(x, 7.0)
        if name == "O1":
            k_exp2[(1,)](x, y, 16, BLOCK=16, enable_reflect_ftz=False)
        elif name in ("O2", "O3"):
            kw = {"enable_reflect_ftz": False} if name == "O2" else {}
            compiled = k_exp2.warmup(x, y, 16, BLOCK=16, grid=(1,), **kw)
            compiled[(1, 1, 1)](x, y, 16)
        else:
            src = ASTSource(k_exp2, {"x_ptr": "*fp32", "y_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
                            constexprs={"BLOCK": 16})
            if name == "O4":
                compiled = triton.compile(src, options={"enable_reflect_ftz": False})
            else:
                compiled = triton.compile(src, target=GPUTarget("cuda", 80, 32))
            compiled[(1, 1, 1)](x, y, 16)
        torch.cuda.synchronize()
        v = y[0].item()
        return f"exp2(-130) = {v!r} ({'2^-130' if v == 2.0 ** -130 else ('0' if v == 0.0 else 'other')})"
    x = torch.arange(32 * 32, device=dev, dtype=torch.float32).view(32, 32)
    y = torch.full((32 * 32,), -1.0, device=dev)
    d16 = TensorDescriptor(x, shape=[32, 32], strides=[32, 1], block_shape=[16, 16])
    if name == "O6":
        d32 = TensorDescriptor(x, shape=[32, 32], strides=[32, 1], block_shape=[32, 32])
        compiled = k_desc.warmup(d32, y, grid=(1,))
        compiled[(1, 1, 1)](d16, y)
    else:
        k_desc[(1,)](d16, y)
    torch.cuda.synchronize()
    written = int((y != -1.0).sum().item())
    ok = all(y.view(32, 32)[r, c].item() == x[r, c].item() for r in range(16) for c in range(16))
    return f"elements written: {written} (a [16, 16] block is 256, a [32, 32] block 1024); top-left 16 x 16 {'equal to x' if ok else 'not x'}"


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "error: " + type(e).__name__ + ": " + (msg[0] if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else "process failed: " + p.stderr.strip()[-160:]
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()
