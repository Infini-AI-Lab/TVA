"""lr-11b, round 11: a compile option given by `@triton.heuristics` (library issue L72, scope 4).

`Heuristics.run` (primitives/triton-3.8.0-autotuner.py lines 480-483) sets `kwargs[name]` for each
heuristic and calls the JIT's `run` with them, which parses options out of the keywords
(primitives/triton-3.8.0-jit.py lines 726-741). So a heuristic whose name is a compile option gives
that option to every launch, although the host's call `kernel[grid](...)` gives no keyword.

  H1  @triton.heuristics({"enable_reflect_ftz": lambda a: False}): tl.math.exp2(-130), launched with
      no keyword (2^-130 without libdevice's flush to zero; 0 with it, the default)
  H2  the same kernel without the heuristic (control)
  H3  @triton.heuristics({"enable_fp_fusion": lambda a: False}): x * y + z on float32, launched with no
      keyword; the PTX holds an fma exactly when fusion is on
  H4  the same kernel without the heuristic (control)
  H5  @triton.heuristics({"enable_fp_fusion": lambda a: True}), launched with the keyword
      enable_fp_fusion=False: the heuristic's value replaces the launch's

Run (from TVA's root): CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_exp2_body(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(y_ptr + offs, tl.math.exp2(tl.load(x_ptr + offs)))


@triton.heuristics({"enable_reflect_ftz": lambda a: False})
@triton.jit
def k_exp2_h(x_ptr, y_ptr, BLOCK: tl.constexpr):
    k_exp2_body(x_ptr, y_ptr, BLOCK)


@triton.jit
def k_exp2(x_ptr, y_ptr, BLOCK: tl.constexpr):
    k_exp2_body(x_ptr, y_ptr, BLOCK)


@triton.jit
def k_fma_body(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) * tl.load(y_ptr + offs) + tl.load(z_ptr + offs))


@triton.heuristics({"enable_fp_fusion": lambda a: False})
@triton.jit
def k_fma_h(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK: tl.constexpr):
    k_fma_body(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK)


@triton.heuristics({"enable_fp_fusion": lambda a: True})
@triton.jit
def k_fma_on(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK: tl.constexpr):
    k_fma_body(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK)


@triton.jit
def k_fma(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK: tl.constexpr):
    k_fma_body(x_ptr, y_ptr, z_ptr, o_ptr, BLOCK)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    x = torch.full((16,), -130.0, device="cuda")
    for name, k in (("H1", k_exp2_h), ("H2", k_exp2)):
        y = torch.full_like(x, 7.0)
        ck = k[(1,)](x, y, BLOCK=16)
        torch.cuda.synchronize()
        v = y[0].item()
        print(f"{name}: exp2(-130) = {v!r} ({'2^-130' if v == 2.0 ** -130 else ('0' if v == 0.0 else 'other')}); "
              f"compiled with enable_reflect_ftz={ck.metadata.enable_reflect_ftz}")
    a = torch.full((16,), 1.0 + 2.0 ** -12, device="cuda")
    c = torch.full((16,), -(1.0 + 2.0 ** -11), device="cuda")
    for name, k, kw in (("H3", k_fma_h, {}), ("H4", k_fma, {}), ("H5", k_fma_on, {"enable_fp_fusion": False})):
        o = torch.empty_like(a)
        ck = k[(1,)](a, a, c, o, BLOCK=16, **kw)
        torch.cuda.synchronize()
        ptx = ck.asm["ptx"]
        print(f"{name}: x*y+z = {o[0].item()!r} (fused: 2^-24 = {2.0 ** -24!r}; unfused: 0.0); "
              f"fma in PTX: {'fma.rn.f32' in ptx}; compiled with enable_fp_fusion={ck.metadata.enable_fp_fusion}")


if __name__ == "__main__":
    main()
