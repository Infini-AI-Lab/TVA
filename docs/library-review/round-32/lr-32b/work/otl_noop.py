"""lr-32b, round 32, L149: does OptimizeThreadLocality also rewrite a loop whose reduced operand is,
in the source, a load followed by an operation that folds away (a same-dtype `.to`, `* 1.0` from a
constexpr, `/ 1.0`, `- 0.0`)? Same data as evidence/reduce-tree L: every association of the source
gives 1.0.

Run from TVA's root: CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I docs/library-review/round-32/lr-32b/work/otl_noop.py
"""
import os, tempfile
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="lr32b-"))
import torch, triton, triton.language as tl


@triton.jit
def k(x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr, FORM: tl.constexpr,
      SCALE: tl.constexpr):
    rows = tl.arange(0, M)
    acc = tl.zeros([M], tl.float32)
    for t in range(T):
        x = tl.load(x_ptr + t * M * N + rows[:, None] * N + tl.arange(0, N)[None, :])
        if FORM == 0:
            x = x.to(tl.float32)
        elif FORM == 1:
            x = x * SCALE
        elif FORM == 2:
            x = x / 1.0
        elif FORM == 3:
            x = x - 0.0
        elif FORM == 4:
            x = x + 0.0
        acc += tl.sum(x, axis=1)
    tl.store(o_ptr + rows, acc)


for FORM, name in ((0, "tl.load(p).to(tl.float32) on float32"), (1, "x * SCALE, SCALE: constexpr = 1.0"),
                   (2, "x / 1.0"), (3, "x - 0.0"), (4, "x + 0.0 (control)")):
    M, N, T = 16, 64, 2
    x = torch.zeros(T, M, N)
    x[0, :, 0] = 2.0 ** 24; x[0, :, N - 1] = 1.0
    x[1, :, 0] = -(2.0 ** 24); x[1, :, N - 1] = 1.0
    xd = x.cuda(); o = torch.empty(M, device="cuda")
    h = k[(1,)](xd, o, M, N, T, FORM, 1.0, num_warps=4)
    g = h.asm["ttgir"]
    print(f"{name}: rewritten {g.count('tt.reduce') >= 2 and 'tt.reshape' in g}; stored "
          f"{sorted(set(o.cpu().tolist()))}; every association of the source gives 1.0", flush=True)
