"""TVA library evidence: which launches end with an error on the host, and which return without one.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/launch-outcomes/run.py
Output: evidence/launch-outcomes/out.txt (README.md says what it shows).

Every case launches `count`, or a kernel that also increments the counters, onto a fresh buffer of
4096 int64 counters set to a sentinel. `count` makes every program add 1 to counter `pid_0 % 4096`, so
after the launch the counters' sum minus the sentinels is the number of programs that ran. For each
case the script prints the exception the host got (or none), whether the device is still usable (a
synchronize and a control launch succeed), and how many programs ran.
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
SLOTS = 4096
SENTINEL = 7


@triton.jit
def count(cnt_ptr):
    x = tl.program_id(0)
    tl.atomic_add(cnt_ptr + x % 4096, 1, sem="relaxed")


@triton.jit
def count_big_smem(cnt_ptr, a_ptr, KT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # a pipelined tl.dot loop whose stages need more shared memory than an SM has
    tl.atomic_add(cnt_ptr + tl.program_id(0) % 4096, 1, sem="relaxed")
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k in range(0, KT):
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :] + k * 16)
        b = tl.load(a_ptr + rk[:, None] * N + rn[None, :] + k * 16)
        acc = tl.dot(a, b, acc)
    tl.store(a_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


@triton.jit
def count_small_dot(cnt_ptr, a_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # tl.dot with K = 8, below the minimum size on sm_90
    tl.atomic_add(cnt_ptr + tl.program_id(0) % 4096, 1, sem="relaxed")
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(a_ptr + rk[:, None] * N + rn[None, :])
    tl.store(a_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b).to(tl.float16))


@triton.jit
def count_fnuz(cnt_ptr, x_ptr):
    # float8e4b8 is an AMD format; the NVIDIA backend has no conversion into it
    tl.atomic_add(cnt_ptr + tl.program_id(0) % 4096, 1, sem="relaxed")
    x = tl.load(x_ptr + tl.arange(0, 16))
    tl.store(x_ptr + tl.arange(0, 16), x.to(tl.float8e4b8).to(tl.float32))


@triton.jit
def count_static_assert(cnt_ptr, B: tl.constexpr):
    tl.atomic_add(cnt_ptr + tl.program_id(0) % 4096, 1, sem="relaxed")
    tl.static_assert(B % 16 == 0, "B must be a multiple of 16")


def counters():
    return torch.full((SLOTS,), SENTINEL, device=DEV, dtype=torch.int64)


def ran(cnt):
    return int((cnt - SENTINEL).sum().item())


def case(name, launch, expect):
    cnt = counters()
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        launch(cnt)
        exc = "none"
    except Exception as e:  # noqa: BLE001 - the point is which exception the host gets
        lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
        text = lines[0][:100] if lines else ""
        if len(lines) > 1:  # a CompilationError gives the cause on its last line
            text += " ... " + lines[-1][:140]
        exc = f"{type(e).__name__}: {text}"
    try:
        torch.cuda.synchronize()
        probe = counters()
        count[(3,)](probe)
        torch.cuda.synchronize()
        usable = "yes" if ran(probe) == 3 else "no (control launch wrong)"
    except Exception as e:  # noqa: BLE001
        usable = f"no ({type(e).__name__})"
    n = ran(cnt)
    ok = (n == expect)
    print(f"{name}\n    host error: {exc}\n    programs run: {n} (expected {expect}): {'as expected' if ok else 'UNEXPECTED'}; "
          f"memory unchanged: {n == 0}; device usable after: {usable}; {time.time() - t0:.2f} s", flush=True)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    a = torch.randn(256 * 256, device=DEV, dtype=torch.float16)
    x = torch.randn(16, device=DEV, dtype=torch.float32)
    count[(1,)](counters())  # compile once
    torch.cuda.synchronize()

    print("\n== A. Launches the host rejects with an error, before any program runs ==", flush=True)
    case("A1 shared memory: pipelined 256x256x128 float16 tl.dot loop, num_stages=6",
         lambda c: count_big_smem[(1,)](c, a, 64, 256, 256, 128, num_stages=6, num_warps=8), 0)
    case("A2 threads: num_warps=64 (2048 threads per program)", lambda c: count[(1,)](c, num_warps=64), 0)
    case("A3 tl.dot below its minimum size: float16 operands, K = 8",
         lambda c: count_small_dot[(1,)](c, a, 16, 16, 8), 0)
    case("A4 a dtype the target lacks: x.to(tl.float8e4b8)", lambda c: count_fnuz[(1,)](c, x), 0)
    case("A5 a tl.static_assert that fails (B = 24)", lambda c: count_static_assert[(1,)](c, 24), 0)
    case("A6 grid (1, 65536, 1): y above CUDA's limit", lambda c: count[(1, 2 ** 16, 1)](c), 0)
    case("A7 grid (1, 1, 65536): z above CUDA's limit", lambda c: count[(1, 1, 2 ** 16)](c), 0)
    case("A8 grid (2^31, 1, 1): x does not fit the launcher's C int", lambda c: count[(2 ** 31, 1, 1)](c), 0)

    print("\n== B. Launches with no program: they return without an error ==", flush=True)
    case("B1 grid (0, 1, 1)", lambda c: count[(0, 1, 1)](c), 0)
    case("B2 grid (5, 0, 3)", lambda c: count[(5, 0, 3)](c), 0)

    print("\n== C. The program count: the launcher's guard gridX * gridY * gridZ > 0 is computed in C int ==",
          flush=True)
    case("C1 grid (2^31 - 1, 1, 1) = 2^31 - 1 programs", lambda c: count[(2 ** 31 - 1, 1, 1)](c), 2 ** 31 - 1)
    case("C2 grid (65535, 32768, 1) = 2^31 - 2^15 programs", lambda c: count[(65535, 2 ** 15, 1)](c),
         65535 * 2 ** 15)
    case("C3 grid (2^14, 2^14, 2) = 2^29 programs", lambda c: count[(2 ** 14, 2 ** 14, 2)](c), 2 ** 29)
    case("C4 grid (2^30, 2, 1) = 2^31 programs (wraps to -2^31)", lambda c: count[(2 ** 30, 2, 1)](c), 0)
    case("C5 grid (2^15, 2^15, 2) = 2^31 programs (wraps to -2^31)", lambda c: count[(2 ** 15, 2 ** 15, 2)](c), 0)
    case("C6 grid (2^16 + 1, 2^15, 1) = 2^31 + 2^15 programs (wraps negative)",
         lambda c: count[(2 ** 16 + 1, 2 ** 15, 1)](c), 0)
    case("C7 grid (2^16, 2^15, 2) = 2^32 programs (wraps to 0)", lambda c: count[(2 ** 16, 2 ** 15, 2)](c), 0)
    case("C8 grid (6700417, 641, 1) = 2^32 + 1 programs (wraps to 1, so the guard passes)",
         lambda c: count[(6700417, 641, 1)](c), 2 ** 32 + 1)


if __name__ == "__main__":
    main()
