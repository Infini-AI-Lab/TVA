"""lr-1, library review round 1, L3: are launches rejected before they run reported on the host, with no
program run and memory unchanged? And does every launch that runs no program report an error?

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-1/lr-1/work/l3_rejections.py
(Triton 3.8.0, NVIDIA H200 NVL, sm_90.)

Each case fills an output buffer with a sentinel, launches, and reports: the exception the host got
(or none), whether the device is still usable (a synchronize succeeds), and whether the sentinel is
unchanged. A kernel that runs writes 1.0 into out[0] from program (0, 0, 0) and 2.0 into out[1]
from the last program.
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def mark(out_ptr):
    # program (0,0,0) writes out[0] = 1; the last program writes out[1] = 2
    x = tl.program_id(0)
    y = tl.program_id(1)
    z = tl.program_id(2)
    if (x == 0) & (y == 0) & (z == 0):
        tl.store(out_ptr, 1.0)
    if (x == tl.num_programs(0) - 1) & (y == tl.num_programs(1) - 1) & (z == tl.num_programs(2) - 1):
        tl.store(out_ptr + 1, 2.0)


@triton.jit
def big_dot(a_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    tl.store(out_ptr, 1.0)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(a_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b)
    tl.store(out_ptr + 2 + rm[:, None] * N + rn[None, :], c)


@triton.jit
def piped_dot(a_ptr, out_ptr, KT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # a pipelined K loop: each stage keeps an M x K and a K x N float16 tile in shared memory
    tl.store(out_ptr, 1.0)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k in range(0, KT):
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :] + k * 16)
        b = tl.load(a_ptr + rk[:, None] * N + rn[None, :] + k * 16)
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + 2 + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def small_dot(a_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    tl.store(out_ptr, 1.0)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(a_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b)
    tl.store(out_ptr + 2 + rm[:, None] * N + rn[None, :], c)


@triton.jit
def to_fnuz(x_ptr, out_ptr):
    tl.store(out_ptr, 1.0)
    x = tl.load(x_ptr + tl.arange(0, 16))
    y = x.to(tl.float8e4b8)
    tl.store(out_ptr + 2 + tl.arange(0, 16), y.to(tl.float32))


@triton.jit
def failing_static_assert(out_ptr, B: tl.constexpr):
    tl.store(out_ptr, 1.0)
    tl.static_assert(B % 16 == 0, "B must be a multiple of 16")


@triton.jit
def plain(out_ptr):
    tl.store(out_ptr, 1.0)


def sentinel(n=4 + 256 * 256):
    return torch.full((n,), -7.0, device=DEV, dtype=torch.float32)


def report(name, fn, out, expect_first=None):
    t0 = time.time()
    try:
        fn()
        exc = "none"
    except Exception as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}"
    try:
        torch.cuda.synchronize()
        usable = "yes"
    except Exception as e:  # noqa: BLE001
        usable = f"no ({type(e).__name__})"
    dt = time.time() - t0
    head = out[:2].tolist()
    unchanged = bool((out == -7.0).all().item())
    print(f"{name}\n    host error: {exc}\n    device usable after: {usable}; memory unchanged: {unchanged}; "
          f"out[0:2] = {head}; {dt:.2f} s")


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}")
    a = torch.randn(256 * 256, device=DEV, dtype=torch.float16)

    print("\n== Rejections before any program runs (L3: no outcome, reported on the host) ==")
    out = sentinel()
    report("R0 (not a rejection) a single tl.dot of 256x256x256 float16 operands: compiles and runs",
           lambda: big_dot[(1,)](a, out, 256, 256, 256), out)
    out = sentinel()
    report("R1 shared memory: pipelined 256x256x128 float16 tl.dot loop, num_stages=6",
           lambda: piped_dot[(1,)](a, out, 64, 256, 256, 128, num_stages=6, num_warps=8), out)
    out = sentinel()
    report("R2 threads: num_warps=64 (2048 threads per program)", lambda: plain[(1,)](out, num_warps=64), out)
    out = sentinel()
    report("R3 tl.dot below its minimum size: float16, K = 8", lambda: small_dot[(1,)](a, out, 16, 16, 8), out)
    out = sentinel()
    x = torch.randn(16, device=DEV, dtype=torch.float32)
    report("R4 a dtype the target lacks: x.to(tl.float8e4b8) on CUDA", lambda: to_fnuz[(1,)](x, out), out)
    out = sentinel()
    report("R5 tl.static_assert that fails (B = 24)", lambda: failing_static_assert[(1,)](out, 24), out)
    out = sentinel()
    report("R6 grid y = 65536 (above CUDA's limit; Launch.Valid excludes it)", lambda: mark[(1, 65536, 1)](out), out)

    print("\n== Launches that run no program ==")
    out = sentinel()
    report("Z1 grid (0, 1, 1): no program, by the semantics an ok end with memory unchanged",
           lambda: mark[(0, 1, 1)](out), out)
    out = sentinel()
    report("C1 control, grid (2^20, 2, 1) = 2^21 programs: both marks written",
           lambda: mark[(2 ** 20, 2, 1)](out), out)
    out = sentinel()
    report("C2 control, grid (2^30, 1, 1) = 2^30 programs: both marks written",
           lambda: mark[(2 ** 30, 1, 1)](out), out)
    out = sentinel()
    report("W1 grid (2^30, 2, 1) = 2^31 programs, Launch.Valid holds: x < 2^31, y < 2^16",
           lambda: mark[(2 ** 30, 2, 1)](out), out)
    out = sentinel()
    report("W2 grid (2^16, 2^15, 2) = 2^32 programs, Launch.Valid holds",
           lambda: mark[(2 ** 16, 2 ** 15, 2)](out), out)


if __name__ == "__main__":
    main()
