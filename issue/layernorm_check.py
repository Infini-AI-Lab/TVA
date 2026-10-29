"""Library issue L15: does tutorial 05 (layer norm) read global memory its own program wrote?

The kernels are taken verbatim from the pinned tutorials/05-layer-norm.py (lines 1-297: the
kernels and the host `LayerNorm` function; the module-level test and benchmark are not run).
Each case runs forward and backward many times and compares with torch in float32. The
`dirty` variant allocates the partial-sum buffers `_dw`/`_db` with `torch.empty` instead of
`torch.zeros`, so the result depends on the `Count` flag that each program plain-loads and then
atomically sets (tutorial lines 179-182).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I issue/layernorm_check.py
"""
import os

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "tutorials", "05-layer-norm.py")

lines = open(SRC).read().splitlines(keepends=True)
end = next(i for i, l in enumerate(lines) if l.startswith("layer_norm = LayerNorm.apply")) + 1
src = "".join(lines[:end])
NS = {"__name__": "tutorial05", "__file__": SRC}
exec(compile(src, SRC, "exec"), NS)
layer_norm = NS["layer_norm"]


def run(M, N, dtype, reps, dirty):
    zeros = torch.zeros
    if dirty:
        # the backward allocates _dw/_db with torch.zeros; make them garbage instead
        NS["torch"] = type("T", (), {**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")},
                                     "zeros": staticmethod(lambda *a, **k: zeros(*a, **k) if k.get("dtype") == torch.int32
                                                           else torch.full(*a, 1e3, **k))})
    else:
        NS["torch"] = torch
    bad_y = bad_dx = bad_dw = bad_db = 0
    for _ in range(reps):
        x = -2.3 + 0.5 * torch.randn(M, N, dtype=dtype, device="cuda")
        w = torch.rand(N, dtype=dtype, device="cuda", requires_grad=True)
        b = torch.rand(N, dtype=dtype, device="cuda", requires_grad=True)
        dy = 0.1 * torch.randn_like(x)
        x.requires_grad_(True)
        y = layer_norm(x, (N,), w, b, 1e-5)
        y.backward(dy)
        got = [y.detach(), x.grad, w.grad, b.grad]
        x.grad = w.grad = b.grad = None
        xf, wf, bf = (t.detach().float().requires_grad_(True) for t in (x, w, b))
        yr = torch.nn.functional.layer_norm(xf, (N,), wf, bf, 1e-5)
        yr.backward(dy.float())
        ref = [yr.detach(), xf.grad, wf.grad, bf.grad]
        tol = [1e-2, 1e-2, 1e-2 * max(1, M / 256), 1e-2 * max(1, M / 256)]
        bad = [int(((g.float() - r).abs() > t).any()) for g, r, t in zip(got, ref, tol)]
        bad_y += bad[0]; bad_dx += bad[1]; bad_dw += bad[2]; bad_db += bad[3]
    NS["torch"] = torch
    return bad_y, bad_dx, bad_dw, bad_db


def main():
    print(f"triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
    for dtype in (torch.float16, torch.float32):
        for M, N in ((4096, 16), (4096, 64), (2048, 1024), (1151, 8192)):
            for dirty in (False, True):
                reps = 50
                r = run(M, N, dtype, reps, dirty)
                print(f"{str(dtype):14s} M={M:5d} N={N:5d} {'dirty buffers' if dirty else 'as written  '}: "
                      f"wrong runs out of {reps}: y {r[0]}, dx {r[1]}, dw {r[2]}, db {r[3]}", flush=True)


if __name__ == "__main__":
    main()
