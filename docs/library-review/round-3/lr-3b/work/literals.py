"""Library review round 3 (lr-3b), survey area `transcription`: what value a Python float literal
takes in a Triton 3.8.0 kernel, against TRANSCRIPTION.md [literal] ("A Python float constant that
fits float32 is the nearest float32") together with [implicit] (the conversion to the operation's
dtype is written).

Each kernel computes `x <op> c` (or `tl.full`) for a Python float literal `c`, on a dtype `d`, and
stores the result. The script compares the stored bits with what [literal] + [implicit] give: the
nearest float32 of `c`, converted (round to nearest even) to `d`, then the IEEE operation in `d`.

Triton 3.8.0 (~/fpv/tva/.venv), H200 NVL (sm_90), GPU 0. Run from the TVA root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3b/work/literals.py
"""
import struct

import torch
import triton
import triton.language as tl

DEV = "cuda"
B = 128


def f32_nearest(c):
    return struct.unpack("<f", struct.pack("<f", c))[0]


def make_kernels():
    ks = {}
    for name, cval in (("1e-7", 1e-7), ("m1e-7", -1e-7), ("1.23456e-5", 1.23456e-5), ("0.1", 0.1),
                       ("3.14159265", 3.14159265), ("1e-3", 1e-3), ("0.3", 0.3)):
        src = f'''
@triton.jit
def k_mul(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x * {cval!r})

@triton.jit
def k_add(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(o_ptr + offs, x + {cval!r})
'''
        ks[name] = (cval, src)
    return ks


def run():
    import tempfile, importlib.util, os, sys
    tmpdir = tempfile.mkdtemp()
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    total_bad = {}
    for name, (cval, src) in make_kernels().items():
        path = os.path.join(tmpdir, f"lit_{name.replace('.', '_').replace('-', '_')}.py")
        with open(path, "w") as f:
            f.write("import triton\nimport triton.language as tl\n" + src)
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for dt in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            x = torch.tensor([1.0, 2.0, -3.0, 1e3, 0.5, 7.0] * (B // 6) + [1.0] * (B % 6), dtype=dt, device=DEV)
            for op in ("mul", "add"):
                o = torch.empty_like(x)
                getattr(mod, f"k_{op}")[(1,)](x, o, BLOCK=B)
                # [literal] + [implicit]: nearest float32 of c, then rounded to dt
                c_model = torch.tensor(f32_nearest(cval), dtype=torch.float32).to(dt)
                ref = (x * c_model) if op == "mul" else (x + c_model)
                bad = (o.view(torch.int16 if dt.itemsize == 2 else torch.int32 if dt.itemsize == 4 else torch.int64)
                       != ref.view(torch.int16 if dt.itemsize == 2 else torch.int32 if dt.itemsize == 4 else torch.int64)).sum().item()
                tag = f"{name:>11} {str(dt).replace('torch.', ''):>8} x {'*' if op == 'mul' else '+'} c"
                print(f"{tag}: device x[0] {'*' if op == 'mul' else '+'} c = {o[0].item()!r:<24} "
                      f"[literal] model = {ref[0].item()!r:<24} differing elements {bad}/{B}")
                total_bad[(name, str(dt), op)] = bad
    print()
    print("cases with differences:", sorted(k for k, v in total_bad.items() if v))


if __name__ == "__main__":
    run()
