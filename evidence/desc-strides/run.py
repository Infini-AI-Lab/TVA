"""Host tensor descriptors with a zero leading stride, or one shorter than a row (library issue L154 (4)).

`TensorDesc.Valid` (`lean/TVA/Lang/Desc.lean`) asks of a leading stride only that it be a multiple of 16
bytes, as Triton's host `TensorDescriptor` does (`triton/tools/tensor_descriptor.py`), so it admits a stride of
`0` (every row the same memory) and one shorter than the row (rows that overlap). The library's meaning of an
access is the formula `base + Σ_k (offs_k + i_k)·stride_k` for in-bounds lanes and the padding for the others
(`TensorDesc.elemAt`, `tl.desc_load`, `tl.desc_atomic`). Does the H200 compute that?

  Z1  `desc.load` of every block of a 2-D float32 descriptor of shape [R, C] = [24, 40] with strides [0, 1],
      [4, 1], [8, 1], [16, 1] and [40, 1] (a control), blocks [8, 16] and [16, 32] at every block offset that
      tiles the shape and past it (padding), against the formula; also float16 and int8 (strides of 16 bytes)
  Z2  a 3-D float32 descriptor of shape [4, 6, 32] with strides [0, 32, 1] and [32, 8, 1]
  Z3  `desc.atomic_add` of a block of ones through strides [0, 1] and [8, 1] (lanes on the same element):
      against the formula, each lane adding once to its element

Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-strides/run.py
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def k_load2(desc, out, R0, C0, BR: tl.constexpr, BC: tl.constexpr):
    v = desc.load([R0, C0])
    r = tl.arange(0, BR)
    c = tl.arange(0, BC)
    tl.store(out + r[:, None] * BC + c[None, :], v)


@triton.jit
def k_load3(desc, out, BA: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    v = desc.load([0, 0, 0])
    a = tl.arange(0, BA)
    r = tl.arange(0, BR)
    c = tl.arange(0, BC)
    tl.store(out + a[:, None, None] * BR * BC + r[None, :, None] * BC + c[None, None, :], v)


@triton.jit
def k_add2(desc, BR: tl.constexpr, BC: tl.constexpr):
    desc.atomic_add([0, 0], tl.full([BR, BC], 1.0, tl.float32))


def formula2(mem, R, C, s0, r0, c0, BR, BC, pad):
    out = torch.full((BR, BC), pad, dtype=mem.dtype, device=mem.device)
    for i in range(BR):
        for j in range(BC):
            r, c = r0 + i, c0 + j
            if 0 <= r < R and 0 <= c < C:
                out[i, j] = mem[r * s0 + c]
    return out


def z1():
    lines = []
    R, C = 24, 40
    for dt, name in ((torch.float32, "float32"), (torch.float16, "float16"), (torch.int8, "int8")):
        e = torch.tensor([], dtype=dt).element_size()
        unit = 16 // e
        for s0 in (0, unit, 2 * unit, 4 * unit, C if (C * e) % 16 == 0 else 48):
            n = (R - 1) * s0 + C
            g = torch.Generator(device="cuda").manual_seed(s0 + e)
            mem = (torch.randint(-100, 100, (n,), device="cuda", generator=g)).to(dt)
            bad = tot = 0
            for (BR, BC) in ((8, 16), (16, 32)):
                if BC * e < 16:
                    continue
                desc = TensorDescriptor(mem, [R, C], [s0, 1], [BR, BC])
                for r0 in range(0, R + BR, BR):
                    for c0 in range(0, C + BC, BC):
                        out = torch.empty(BR, BC, dtype=dt, device="cuda")
                        k_load2[(1,)](desc, out, r0, c0, BR=BR, BC=BC)
                        torch.cuda.synchronize()
                        ref = formula2(mem.cpu(), R, C, s0, r0, c0, BR, BC, 0).to("cuda")
                        bad += int((out != ref).sum())
                        tot += out.numel()
            lines.append(f"Z1 {name} strides [{s0}, 1]: {bad} of {tot} lanes differ from the formula")
    return lines


def z2():
    lines = []
    for strides in ((0, 32, 1), (32, 8, 1)):
        n = 3 * strides[0] + 5 * strides[1] + 32
        mem = torch.arange(n, dtype=torch.float32, device="cuda")
        desc = TensorDescriptor(mem, [4, 6, 32], list(strides), [4, 8, 32])
        out = torch.empty(4, 8, 32, dtype=torch.float32, device="cuda")
        k_load3[(1,)](desc, out, BA=4, BR=8, BC=32)
        torch.cuda.synchronize()
        ref = torch.zeros(4, 8, 32, dtype=torch.float32)
        m = mem.cpu()
        for a in range(4):
            for r in range(8):
                for c in range(32):
                    if r < 6:
                        ref[a, r, c] = m[a * strides[0] + r * strides[1] + c]
        lines.append(f"Z2 float32 [4, 6, 32] strides {list(strides)}: {int((out.cpu() != ref).sum())} of {out.numel()} "
                     f"lanes differ from the formula")
    return lines


def z3():
    lines = []
    R, C, BR, BC = 8, 32, 8, 32
    for s0 in (0, 8):
        n = (R - 1) * s0 + C
        mem = torch.zeros(n, dtype=torch.float32, device="cuda")
        desc = TensorDescriptor(mem, [R, C], [s0, 1], [BR, BC])
        k_add2[(1,)](desc, BR=BR, BC=BC)
        torch.cuda.synchronize()
        ref = torch.zeros(n, dtype=torch.float32)
        for i in range(R):
            for j in range(C):
                ref[i * s0 + j] += 1.0
        lines.append(f"Z3 atomic_add of ones, strides [{s0}, 1]: {int((mem.cpu() != ref).sum())} of {n} elements differ "
                     f"from one addition per lane (max count {int(ref.max())})")
    return lines


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for f in (z1, z2, z3):
        try:
            for l in f():
                print(l, flush=True)
        except Exception as ex:
            print(f"{f.__name__}: {type(ex).__name__}: {str(ex).splitlines()[0][:150]}")


if __name__ == "__main__":
    main()
