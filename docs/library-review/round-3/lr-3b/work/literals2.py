"""Library review round 3 (lr-3b): the same question as literals.py, for the other places a Python
float literal enters a kernel: `tl.full(shape, c, dtype)`, `tl.where(m, x, c)` and
`tl.load(p, mask, other=c)`. Compares the device's bits with [literal] + [implicit] (nearest
float32 of `c`, then rounded to the dtype).

Triton 3.8.0 (~/fpv/tva/.venv), H200 NVL (sm_90), GPU 0. Run from the TVA root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3b/work/literals2.py
"""
import importlib.util
import os
import struct
import tempfile

import torch
import triton
import triton.language as tl

B = 64
SRC = '''
import triton
import triton.language as tl

@triton.jit
def k_full(o_ptr, DT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.full([BLOCK], C, DT))

@triton.jit
def k_where(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, tl.where(offs < 0, x, C))

@triton.jit
def k_other(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < 0, other=C)
    tl.store(o_ptr + offs, x)
'''


def bits(t):
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def main():
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    tmp = tempfile.mkdtemp()
    tdt = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32, torch.float64: tl.float64}
    for n, c in enumerate((1e-7, -1e-7, 1.23456e-5, 0.1)):
        path = os.path.join(tmp, f"lit2_{n}.py")
        with open(path, "w") as f:
            f.write(SRC.replace("C)", f"{c!r})").replace("C, DT", f"{c!r}, DT").replace("other=C", f"other={c!r}"))
        spec = importlib.util.spec_from_file_location(f"lit2_{n}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for dt in (torch.bfloat16, torch.float64):
            model = torch.tensor(struct.unpack("<f", struct.pack("<f", c))[0], dtype=torch.float32).to(dt)
            for kind in ("full", "where", "other"):
                o = torch.empty(B, dtype=dt, device="cuda")
                x = torch.ones(B, dtype=dt, device="cuda")
                if kind == "full":
                    mod.k_full[(1,)](o, DT=tdt[dt], BLOCK=B)
                else:
                    getattr(mod, f"k_{kind}")[(1,)](x, o, BLOCK=B)
                differ = (bits(o) != bits(model.expand(B).contiguous().cuda())).sum().item()
                print(f"c = {c!r:<11} {str(dt)[6:]:>8} {kind:>5}: device {o[0].item()!r:<24} "
                      f"[literal] model {model.item()!r:<24} differing {differ}/{B}")


if __name__ == "__main__":
    main()
