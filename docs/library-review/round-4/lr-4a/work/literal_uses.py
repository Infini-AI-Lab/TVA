"""lr-4a, L23: the uses of a Python float that evidence/literals does not run: a stored value,
`tl.maximum`, `tl.where`, `/` on bfloat16 (computed in float32), and `+` on bfloat16, each with
c = 1e-7 (bfloat16 constant +0 when built in bfloat16, a nonzero bfloat16 when built in float32 and
converted). The bfloat16 output bits are compared with what `[literal]` says.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/literal_uses.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, o_ptr, WHICH: tl.constexpr):
    i = tl.arange(0, 2)
    x = tl.load(x_ptr + i)            # bfloat16 zeros (the second lane is 1.0 for `/`)
    if WHICH == 0:
        tl.store(o_ptr + i, 1e-7)     # a stored value: its own type (float32), then converted
    elif WHICH == 1:
        tl.store(o_ptr + i, tl.maximum(x, 1e-7))  # tl.maximum: float32 tensor, compared in float32
    elif WHICH == 2:
        tl.store(o_ptr + i, tl.where(i < 0, x, 1e-7))  # tl.where: built in bfloat16
    elif WHICH == 3:
        tl.store(o_ptr + i, (x + 1.0) / 1e7)      # `/`: computed in float32, constant litF32(1e7)
    else:
        tl.store(o_ptr + i, x + 1e-7)             # `+` on bfloat16: built in bfloat16 (+0)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    x = torch.zeros(2, dtype=torch.bfloat16, device="cuda")
    ref_f32_then_bf16 = torch.tensor([1e-7], dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
    div_ref = (torch.tensor([1.0], dtype=torch.float32) / torch.tensor([1e7], dtype=torch.float32)).to(torch.bfloat16)
    expect = {0: ref_f32_then_bf16, 1: ref_f32_then_bf16, 2: 0, 3: div_ref.view(torch.int16).item() & 0xFFFF, 4: 0}
    names = {0: "tl.store(p, 1e-7)", 1: "tl.maximum(x, 1e-7)", 2: "tl.where(False, x, 1e-7)",
             3: "(x + 1.0) / 1e7", 4: "x + 1e-7"}
    for w in range(5):
        o = torch.empty(2, dtype=torch.bfloat16, device="cuda")
        k[(1,)](x, o, WHICH=w)
        got = o.view(torch.int16)[0].item() & 0xFFFF
        print(f"{names[w]:28s} on bfloat16: bits {got:#06x}; [literal] says {expect[w]:#06x}: "
              f"{'as stated' if got == expect[w] else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
