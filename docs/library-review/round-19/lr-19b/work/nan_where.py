"""lr-19b, library review round 19 (L98): NaN constants the batch's evidence does not run.

T1. `tl.where(<false>, x, E)` and `tl.where(<true>, E, x)` with `x` an `int32` or `int64` tensor and
    `E` a Python NaN: `whereTyping?` gives `float32` (a Python float with an integer tensor), so the
    constant is `FloatV.litF32` of the double, `0x7fc00000` / `0xffc00000`, stored as `float32` bits.
T2. The same with `x` a `float16`/`bfloat16` tensor and the constant on the *true* side (N1 of
    evidence/literals ran only the false side): `litF16`/`litBF16`.
T3. A NaN that Python computes from `tl.constexpr` arguments that are not NaN (`C - C` with
    `C = inf`, `C * 0.0`): the cache key is `inf`, so the pattern is fixed; Python's double is
    `0xfff8000000000000` on this host (x86-64).
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def k_int(xi_ptr, xl_ptr, o_ptr):
    i = tl.arange(0, 1)
    xi = tl.load(xi_ptr + i)
    xl = tl.load(xl_ptr + i)
    tl.store(o_ptr + 0 + i, tl.where(i < 0, xi, float('nan')))
    tl.store(o_ptr + 1 + i, tl.where(i < 0, xi, -float('nan')))
    tl.store(o_ptr + 2 + i, tl.where(i >= 0, float('nan'), xi))
    tl.store(o_ptr + 3 + i, tl.where(i >= 0, -float('nan'), xl))
    tl.store(o_ptr + 4 + i, tl.where(i < 0, xl, float('inf') - float('inf')))


@triton.jit
def k_half(xh_ptr, xb_ptr, oh_ptr, ob_ptr):
    i = tl.arange(0, 1)
    xh = tl.load(xh_ptr + i)
    xb = tl.load(xb_ptr + i)
    tl.store(oh_ptr + 0 + i, tl.where(i >= 0, float('nan'), xh))
    tl.store(oh_ptr + 1 + i, tl.where(i >= 0, -float('nan'), xh))
    tl.store(ob_ptr + 0 + i, tl.where(i >= 0, float('nan'), xb))
    tl.store(ob_ptr + 1 + i, tl.where(i >= 0, -float('nan'), xb))


@triton.jit
def k_cexpr(o_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + 0 + i, tl.full([1], C - C, tl.float32))
    tl.store(o_ptr + 1 + i, tl.full([1], C * 0.0, tl.float32))
    tl.store(o_ptr + 2 + i, tl.full([1], -(C - C), tl.float32))


def bits(t, w):
    iv = {16: torch.int16, 32: torch.int32}[w]
    return [v & ((1 << w) - 1) for v in t.view(iv).tolist()]


def main():
    print(f"triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
    xi = torch.ones(1, dtype=torch.int32, device="cuda")
    xl = torch.ones(1, dtype=torch.int64, device="cuda")
    o = torch.zeros(5, dtype=torch.float32, device="cuda")
    k_int[(1,)](xi, xl, o)
    torch.cuda.synchronize()
    want = [0x7fc00000, 0xffc00000, 0x7fc00000, 0xffc00000, 0xffc00000]
    got = bits(o, 32)
    print("T1 int x, float32 result:", " ".join(f"{g:08x}" for g in got),
          "| litF32:", " ".join(f"{w:08x}" for w in want), "| differ:", sum(g != w for g, w in zip(got, want)))
    xh = torch.ones(1, dtype=torch.float16, device="cuda")
    xb = torch.ones(1, dtype=torch.bfloat16, device="cuda")
    oh = torch.zeros(2, dtype=torch.float16, device="cuda")
    ob = torch.zeros(2, dtype=torch.bfloat16, device="cuda")
    k_half[(1,)](xh, xb, oh, ob)
    torch.cuda.synchronize()
    gh, gb = bits(oh, 16), bits(ob, 16)
    wh, wb = [0x7e00, 0xfe00], [0x7fc0, 0xffc0]
    print("T2 float16 (true side):", " ".join(f"{g:04x}" for g in gh), "| litF16:", " ".join(f"{w:04x}" for w in wh),
          "| differ:", sum(g != w for g, w in zip(gh, wh)))
    print("T2 bfloat16 (true side):", " ".join(f"{g:04x}" for g in gb), "| litBF16:", " ".join(f"{w:04x}" for w in wb),
          "| differ:", sum(g != w for g, w in zip(gb, wb)))
    oc = torch.zeros(3, dtype=torch.float32, device="cuda")
    k_cexpr[(1,)](oc, C=float('inf'))
    torch.cuda.synchronize()
    inf = float('inf')
    py = [struct.unpack("<Q", struct.pack("<d", v))[0] for v in (inf - inf, inf * 0.0, -(inf - inf))]
    print("T3 Python doubles:", " ".join(f"{p:016x}" for p in py))
    gc = bits(oc, 32)
    wc = [0xffc00000, 0xffc00000, 0x7fc00000]
    print("T3 C = inf constexpr, float32:", " ".join(f"{g:08x}" for g in gc), "| litF32:",
          " ".join(f"{w:08x}" for w in wc), "| differ:", sum(g != w for g, w in zip(gc, wc)))


if __name__ == "__main__":
    main()
