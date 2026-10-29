"""TVA library evidence (library issue L26): which spellings of a reduction on axis 1 of
broadcast products Triton 3.8.0's combine pass rewrites into a `tl.dot`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/bcast-mul/run.py
Output: evidence/bcast-mul/out.txt (README.md says what it shows).

Each kernel computes `r[i, j] = sum_k x[i, k] * y[k, j]` on float32 (or the dtype named) with
`M = N = 16`, `K = 2`, in one spelling, and runs with `enable_fp_fusion` True and False. Reported:
whether the TTIR after the combine pass has a `tt.dot` (the rewrite), and how many of the 256
outputs no tree of IEEE additions of the IEEE products gives (for K = 2 the only such value is
`round(round(x0 y0) + round(x1 y1))`): with fusion off and no rewrite, that count must be 0.
The products are inexact in float32, so a fused or dot result differs from it often.
"""
import torch
import triton
import triton.language as tl

M, K, N = 16, 2, 16


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _add3(a, b):
    return a + b + 0.0


def make(body):
    src = f'''
@triton.jit
def k(x_ptr, y_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    y = tl.load(y_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    r = {body}
    tl.store(o_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], tl.reshape(r, (M, N)))
'''
    return src


SPELLINGS = [
    ("index", "tl.sum(x[:, :, None] * y[None, :, :], axis=1)"),
    ("expand_dims", "tl.sum(tl.expand_dims(x, 2) * tl.expand_dims(y, 0), axis=1)"),
    ("broadcast_to", "tl.sum(tl.broadcast_to(x[:, :, None], (M, K, N)) * tl.broadcast_to(y[None, :, :], (M, K, N)), axis=1)"),
    ("one side broadcast_to", "tl.sum(x[:, :, None] * tl.broadcast_to(y[None, :, :], (M, K, N)), axis=1)"),
    ("tl.reduce with a + b", "tl.reduce(x[:, :, None] * y[None, :, :], 1, _add)"),
    ("method x.sum", "(x[:, :, None] * y[None, :, :]).sum(axis=1)"),
    ("keep_dims=True", "tl.sum(x[:, :, None] * y[None, :, :], axis=1, keep_dims=True)"),
    ("factors swapped", "tl.sum(y[None, :, :] * x[:, :, None], axis=1)"),
    ("reshape instead of expand", "tl.sum(tl.reshape(x, (M, K, 1)) * tl.reshape(y, (1, K, N)), axis=1)"),
    ("tl.reduce with a + b + 0.0", "tl.reduce(x[:, :, None] * y[None, :, :], 1, _add3)"),
    ("dtype=tl.float32 on the sum", "tl.sum(x[:, :, None] * y[None, :, :], axis=1, dtype=tl.float32)"),
]


def main():
    import importlib.util
    import os
    import tempfile
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    g = torch.Generator(device="cuda").manual_seed(20261027)
    x = (1 + torch.rand(M, K, device="cuda", generator=g)).float()
    y = (1 + torch.rand(K, N, device="cuda", generator=g)).float()
    ieee = x[:, 0:1] * y[0:1, :] + x[:, 1:2] * y[1:2, :]
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, body) in enumerate(SPELLINGS):
            path = os.path.join(tmp, f"bm{i}.py")
            open(path, "w").write("import triton\nimport triton.language as tl\n"
                                  "from __main__ import _add, _add3\n" + make(body))
            spec = importlib.util.spec_from_file_location(f"bm{i}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for fusion in (True, False):
                o = torch.empty(M, N, device="cuda")
                try:
                    c = mod.k[(1,)](x, y, o, M, K, N, enable_fp_fusion=fusion)
                    torch.cuda.synchronize()
                    differ = int((o != ieee).sum())
                    print(f"{name}, fusion={fusion}: tt.dot after combine: {'tt.dot' in c.asm['ttir']}; "
                          f"outputs no IEEE tree gives: {differ}/{M * N}")
                except Exception as e:  # noqa: BLE001
                    print(f"{name}, fusion={fusion}: rejected: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        for dt, tdt in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
            path = os.path.join(tmp, f"bm_{dt}.py")
            open(path, "w").write("import triton\nimport triton.language as tl\n" + make(SPELLINGS[0][1]))
            spec = importlib.util.spec_from_file_location(f"bm_{dt}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            o = torch.empty(M, N, device="cuda", dtype=tdt)
            try:
                c = mod.k[(1,)](x.to(tdt), y.to(tdt), o, M, K, N, enable_fp_fusion=False)
                torch.cuda.synchronize()
                print(f"index spelling on {dt}, fusion=False: tt.dot after combine: {'tt.dot' in c.asm['ttir']}")
            except Exception as e:  # noqa: BLE001
                print(f"index spelling on {dt}, fusion=False: rejected on the host: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:80]}")
        for mm, nn in ((8, 16), (16, 8)):
            path = os.path.join(tmp, f"bm_{mm}_{nn}.py")
            open(path, "w").write("import triton\nimport triton.language as tl\n" + make(SPELLINGS[0][1]).replace(
                "tl.reshape(r, (M, N))", "r"))
            spec = importlib.util.spec_from_file_location(f"bm_{mm}_{nn}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            o = torch.empty(mm, nn, device="cuda")
            c = mod.k[(1,)](x[:mm].contiguous(), y[:, :nn].contiguous(), o, mm, K, nn, enable_fp_fusion=False)
            torch.cuda.synchronize()
            print(f"index spelling with M = {mm}, N = {nn}: tt.dot after combine: {'tt.dot' in c.asm['ttir']}")


if __name__ == "__main__":
    main()
