"""Library review round 3 (lr-3b), survey area `transcription`: TRANSCRIPTION.md [itof-narrow] says
a narrowing of an integer-to-float value is one `tl.itof` to the final format, because MLIR folds
`truncf(sitofp x)` into `sitofp x`. That fold needs the default rounding mode
(primitives/llvm-ArithCanonicalization.td:453-463), and a narrowing with
`fp_downcast_rounding="rtz"` is not an `arith.truncf` at all but `tt.fp_to_fp`
(primitives/triton-3.8.0-semantic.py:822-845). This kernel computes
`x.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")` for every int32 x in [0, 2^16) and
compares it with the values `tl.itof` allows (one rounding to nearest even into bfloat16, or two,
through float32 or float64: `tl.ItofRel`), which for these x are all the bf16 nearest value.

Triton 3.8.0 (~/fpv/tva/.venv), H200 NVL (sm_90), GPU 0. Run from the TVA root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3b/work/itof_rtz.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_rtz(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz"))


@triton.jit
def k_rn(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x.to(tl.float32).to(tl.bfloat16))


def main():
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    n, B = 1 << 16, 1024
    x = torch.arange(n, dtype=torch.int32, device="cuda")
    o_rtz = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    o_rn = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    k_rtz[(n // B,)](x, o_rtz, BLOCK=B)
    k_rn[(n // B,)](x, o_rn, BLOCK=B)
    # every int below 2^16 is exact in float32 and float64, so all of tl.ItofRel's roundings
    # (direct, through fp32, through fp64) give the bf16 nearest of x
    near = x.to(torch.float64).to(torch.bfloat16)
    print("rn  chain: elements not the bf16 nearest:", (o_rn != near).sum().item(), "of", n)
    print("rtz chain: elements not the bf16 nearest:", (o_rtz != near).sum().item(), "of", n)
    i = 259
    print(f"x = {i}: rtz chain gives {o_rtz[i].item()}, tl.itof allows only {near[i].item()}")
    ttir = k_rtz.warmup(x, o_rtz, BLOCK=B, grid=(n // B,)).asm["ttir"]
    print("TTIR of the rtz chain has tt.fp_to_fp:", "tt.fp_to_fp" in ttir, "| sitofp:", "arith.sitofp" in ttir)


if __name__ == "__main__":
    main()
