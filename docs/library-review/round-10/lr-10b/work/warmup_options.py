"""lr-10b, round 10 (rules survey): a compile option given at `warmup`, not at the launch, reaches a
`CompiledKernel` launch (Rules.Compiler.Faithful, scope 4 and 6).

`k.warmup(..., enable_reflect_ftz=False)` compiles with libdevice's flush-to-zero off; the returned
CompiledKernel is then launched as tutorial 02 does (`compiled[grid](...)`), with no keyword at the launch.
`tl.math.exp2` at -130 is a subnormal (2^-130) without flush-to-zero and 0 with it.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-10/lr-10b/work/warmup_options.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, tl.math.exp2(x), mask=m)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    x = torch.full((16,), -130.0, device="cuda", dtype=torch.float32)
    for label, kw in (("warmup with the defaults", {}), ("warmup with enable_reflect_ftz=False", {"enable_reflect_ftz": False})):
        y = torch.full_like(x, 7.0)
        compiled = k.warmup(x, y, 16, BLOCK=16, grid=(1,), **kw)
        compiled[(1, 1, 1)](x, y, 16)  # the launch itself gives no option
        torch.cuda.synchronize()
        print(f"- {label}: compiled enable_reflect_ftz = {compiled.metadata.enable_reflect_ftz}; "
              f"y[0] = {y[0].item()!r} (2^-130 = {2.0 ** -130!r})")


if __name__ == "__main__":
    main()
