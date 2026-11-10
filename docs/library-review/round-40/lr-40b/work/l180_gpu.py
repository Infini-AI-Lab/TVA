"""L180 on the device: a 2-D tile whose masked-off lanes' int32 offsets wrap.

lr-39b's numbers: M = 1000 rows, BM = 64, row stride s0 = 2149000, N = BN = 1, s1 = 1, int8 elements.
The matrix bound (M - 1) * s0 < 2^31 holds; the last tile (pid 15, rows 960..1023) has rows 1000..1023
masked off, and their offsets (r * s0, r >= 1000) wrap in int32 to negative values. The library's `_on`
lemmas and the footprint rules say such lanes are never used: the load returns the enabled rows and the
store writes them only. Checked here: no fault, the loaded values are the enabled rows', and the store
changes exactly the enabled rows' bytes, nothing below the buffer's start (a guard region) or after it.
Run: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I l180_gpu.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def tile_copy(src, dst, M, s0, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    offs = rows[:, None] * s0 + cols[None, :]
    mask = (rows < M)[:, None] & (cols < 1)[None, :]
    x = tl.load(src + offs, mask=mask)
    tl.store(dst + offs, x + 1, mask=mask)


def main():
    M, BM, BN, s0 = 1000, 64, 1, 2149000
    assert (M - 1) * s0 < 2 ** 31 and 1023 * s0 >= 2 ** 31
    n = (M - 1) * s0 + 1
    guard = 2 ** 31 + 2 ** 20   # room below and after, so a wrapped address would land in our own memory
    big = torch.zeros(2 * guard + n, dtype=torch.int8, device="cuda")
    src = big[guard:guard + n]
    g = torch.Generator(device="cpu").manual_seed(0)
    rows_val = torch.randint(-100, 100, (M,), generator=g, dtype=torch.int8)
    src[torch.arange(M, device="cuda") * s0] = rows_val.cuda()
    dbig = torch.full((2 * guard + n,), 7, dtype=torch.int8, device="cuda")
    dst = dbig[guard:guard + n]
    print("triton", triton.__version__, torch.cuda.get_device_name(0))
    grid = (triton.cdiv(M, BM),)
    tile_copy[grid](src, dst, M, s0, BM=BM, BN=BN)
    torch.cuda.synchronize()
    idx = torch.arange(M, device="cuda") * s0
    ok_vals = bool((dst[idx].cpu() == (rows_val + 1)).all())
    changed = (dbig != 7).nonzero().flatten() - guard
    expect = set(range(0, n, s0))
    ok_set = set(changed.tolist()) <= expect and len(changed) <= M
    # every enabled row whose value + 1 != 7 must have changed; rows whose value is 6 keep 7
    want = {r * s0 for r in range(M) if int(rows_val[r]) + 1 != 7}
    print("last tile has wrapped masked-off offsets:", 1000 * s0 - 2 ** 32, "..", 1023 * s0 - 2 ** 32)
    print("stored values are the enabled rows' + 1:", ok_vals)
    print("bytes changed only at enabled rows:", ok_set and set(changed.tolist()) == want)
    print("bytes changed:", len(changed), "of", M, "rows (rows holding 6 store 7 unchanged)")


if __name__ == "__main__":
    main()
