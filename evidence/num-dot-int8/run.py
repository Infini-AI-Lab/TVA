"""The integer `tl.dot`: `int8` by `int8` into `int32` on sm_90.

Library evidence for library issue L141 (round 31). Sources, pinned in `primitives/`:
`triton-3.8.0-semantic.py` 1429-1520 (`dot`: integer operands must both be `int8`, the result is `int32`
whatever `out_dtype` says, an accumulator must have the `out_dtype`, `max_num_imprecise_acc` is `0`
unless given), `triton-3.8.0-core.py` 2355-2408 (`tl.dot`'s `input_precision` default),
`triton-3.8.0-nvidia-compiler.py` 19-31 (`min_dot_size`: `K >= 32` for 8-bit operands, any `M`, `N`).

  I1  random int8 operands and random int32 accumulators with room (|acc| + sum|a*b| <= 2^31 - 1), a K loop
      of BK-wide tiles, on 14 tile configurations (M, N from 1 to 128, num_warps 1 to 8: mma.sync and
      wgmma), each in four spellings (tl.dot(a, b, acc); out_dtype=tl.int32; acc=acc; acc += tl.dot(a, b)),
      against the exact integer result
  I2  every input_precision the front end accepts (and allow_tf32=False), on three configurations
  I3  the room at its edge: |acc| + sum|a*b| = 2^31 - 1 exactly, K = 131040, all products +16384 or -16256
  I4  beyond the room (not stated by the library): a final sum past 2^31 - 1, and a partial sum past it
      that the later chunks bring back in range, on both instruction paths
  I5  the front end: int8 x uint8, uint8 x uint8, K = 16, out_dtype=tl.float16 without an accumulator,
      an int32 accumulator with out_dtype=tl.float32, a float32 accumulator, max_num_imprecise_acc=32
  I6  the instructions: the PTX of each configuration of I1 (mma.sync / wgmma, .satfinite or not)

Each case runs in its own process with its own empty Triton cache. The exact result is computed in
float64 on the GPU (every partial sum is an integer below 2^53, so it is exact).
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-int8/run.py
"""
import os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_dot(a_ptr, b_ptr, c0_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          MODE: tl.constexpr, PREC: tl.constexpr, NOTF32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.load(c0_ptr + rm[:, None] * N + rn[None, :])
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None].to(tl.int64) * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None].to(tl.int64) * N + rn[None, :])
        if NOTF32:
            acc = tl.dot(a, b, acc, allow_tf32=False)
        elif MODE == 0:
            acc = tl.dot(a, b, acc, input_precision=PREC)
        elif MODE == 1:
            acc = tl.dot(a, b, acc, out_dtype=tl.int32, input_precision=PREC)
        elif MODE == 2:
            acc = tl.dot(a, b, acc=acc, input_precision=PREC)
        else:
            acc += tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


# (BM, BN, BK, num_warps)
CONFIGS = [(16, 16, 32, 1), (16, 16, 32, 4), (32, 32, 64, 4), (64, 64, 32, 4), (64, 128, 64, 4),
           (128, 128, 128, 8), (128, 64, 32, 4), (64, 64, 256, 8), (16, 64, 32, 2), (64, 16, 64, 4),
           (1, 16, 32, 1), (2, 2, 32, 1), (8, 8, 32, 4), (1, 1, 32, 1)]


def exact(a, b, c0):
    return (a.to(torch.float64) @ b.to(torch.float64)) + c0.to(torch.float64)


def run(a, b, c0, cfg, mode=0, prec=None, notf32=False, num_stages=3):
    BM, BN, BK, nw = cfg
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, dtype=torch.int32, device="cuda")
    ck = k_dot[(M // BM, N // BN)](a, b, c0, c, M, N, K, BM=BM, BN=BN, BK=BK, MODE=mode, PREC=prec,
                                   NOTF32=notf32, num_warps=nw, num_stages=num_stages)
    return c, ck


def rand_case(M, N, K, g, room=True):
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device="cuda", generator=g)
    # extremes in some rows / columns
    a[0, :] = -128
    b[:, 0] = -128
    if M > 1:
        a[1, :] = 127
    if N > 1:
        b[:, 1] = -128
    # the magnitudes in float64 (an int8 abs of -128 is -128)
    mag = (a.to(torch.float64).abs() @ b.to(torch.float64).abs())
    lim = (2**31 - 1) - mag
    assert bool((lim >= 0).all().item())
    u = torch.rand(M, N, dtype=torch.float64, device="cuda", generator=g) * 2 - 1
    c0 = torch.trunc(u * lim).to(torch.int32)
    # every instance has room: |acc| + sum|a*b| <= 2^31 - 1
    assert bool(((c0.to(torch.float64).abs() + mag) <= 2**31 - 1).all().item())
    return a, b, c0


def i1():
    g = torch.Generator(device="cuda").manual_seed(141)
    out, bad, tot = [], 0, 0
    for cfg in CONFIGS:
        BM, BN, BK, nw = cfg
        M, N = max(BM, 1) * 2, max(BN, 1) * 2
        for K in (BK, 4 * BK, 2048):
            a, b, c0 = rand_case(M, N, K, g)
            ref = exact(a, b, c0)
            for mode in (0, 1, 2, 3):
                c, _ = run(a, b, c0, cfg, mode)
                d = int((c.to(torch.float64) != ref).sum().item())
                bad += d
                tot += c.numel()
        out.append(f"{cfg}")
    return f"I1: {len(CONFIGS)} configurations (BM, BN, BK, num_warps) x K in {{BK, 4BK, 2048}} x 4 spellings, {tot} outputs: {bad} differ from the exact result"


def i2():
    g = torch.Generator(device="cuda").manual_seed(142)
    out, bad, tot = [], 0, 0
    for cfg in ((16, 16, 32, 1), (64, 128, 64, 4), (1, 16, 32, 1)):
        BM, BN, BK, nw = cfg
        a, b, c0 = rand_case(BM * 2, BN * 2, 1024, g)
        ref = exact(a, b, c0)
        for prec in (None, "tf32", "tf32x3", "ieee", "bf16x3", "bf16x6", "NOTF32"):
            try:
                c, _ = run(a, b, c0, cfg, 0, None if prec == "NOTF32" else prec, notf32=(prec == "NOTF32"))
                d = int((c.to(torch.float64) != ref).sum().item())
                bad += d
                tot += c.numel()
            except Exception as e:
                out.append(f"{cfg} {prec}: rejected ({type(e).__name__})")
    return f"I2: input_precision in {{omitted, tf32, tf32x3, ieee, bf16x3, bf16x6}} and allow_tf32=False, 3 configurations, {tot} outputs: {bad} differ" + (("; " + "; ".join(out)) if out else "")


def i3():
    out = []
    K = 131040
    for cfg in ((64, 64, 32, 4), (16, 16, 32, 1), (128, 128, 128, 8)):
        BM, BN, BK, nw = cfg
        M, N = BM, BN
        a = torch.full((M, K), -128, dtype=torch.int8, device="cuda")
        bp = torch.full((K, N), -128, dtype=torch.int8, device="cuda")
        bn = torch.full((K, N), 127, dtype=torch.int8, device="cuda")
        c0p = torch.full((M, N), (2**31 - 1) - K * 16384, dtype=torch.int32, device="cuda")
        c0n = torch.full((M, N), -(2**31 - 1) + K * 16256, dtype=torch.int32, device="cuda")
        cp, _ = run(a, bp, c0p, cfg)
        cn, _ = run(a, bn, c0n, cfg)
        okp = bool((cp == 2**31 - 1).all().item())
        okn = bool((cn == -(2**31 - 1)).all().item())
        out.append(f"{cfg}: +{2**31 - 1} {'exact' if okp else 'WRONG ' + str(cp.flatten()[:2].tolist())}, "
                   f"-{2**31 - 1} {'exact' if okn else 'WRONG ' + str(cn.flatten()[:2].tolist())}")
    return f"I3: K = {K}, |acc| + sum|a*b| = 2^31 - 1: " + "; ".join(out)


def i4():
    out = []
    for cfg in ((16, 16, 32, 1), (64, 64, 32, 4)):
        BM, BN, BK, nw = cfg
        M, N, K = BM, BN, 64
        # final overflow: acc = 2^31 - 1 - 100, 64 products of 127*127 = 16129
        a = torch.full((M, K), 127, dtype=torch.int8, device="cuda")
        b = torch.full((K, N), 127, dtype=torch.int8, device="cuda")
        c0 = torch.full((M, N), 2**31 - 1 - 100, dtype=torch.int32, device="cuda")
        c, _ = run(a, b, c0, cfg, num_stages=1)
        ex = 2**31 - 1 - 100 + 64 * 16129
        v = c[0, 0].item()
        kind = "wrapped" if v == ex - 2**32 else ("saturated" if v == 2**31 - 1 else "other")
        out.append(f"{cfg} final sum {ex} > 2^31-1: {v} ({kind})")
        # transient: first 32 k give +16129 each, the next 32 give -16256 each
        a2 = torch.full((M, K), 127, dtype=torch.int8, device="cuda")
        b2 = torch.full((K, N), 127, dtype=torch.int8, device="cuda")
        b2[32:, :] = -128
        c02 = torch.full((M, N), 2**31 - 1 - 1000, dtype=torch.int32, device="cuda")
        c2, _ = run(a2, b2, c02, cfg, num_stages=1)
        ex2 = 2**31 - 1 - 1000 + 32 * 16129 - 32 * 16256
        v2 = c2[0, 0].item()
        kind2 = "exact" if v2 == ex2 else ("as if the first chunk saturated" if v2 == 2**31 - 1 - 32 * 16256 else "other")
        out.append(f"{cfg} partial sum past 2^31-1, final {ex2}: {v2} ({kind2})")
    return "I4 (beyond the room): " + "; ".join(out)


@triton.jit
def k_fe(a_ptr, b_ptr, c_ptr, CASE: tl.constexpr, KK: tl.constexpr):
    r = tl.arange(0, 32)
    rk = tl.arange(0, KK)
    a = tl.load(a_ptr + r[:, None] * KK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * 32 + r[None, :])
    if CASE == 0:
        c = tl.dot(a, b)
    elif CASE == 1:
        c = tl.dot(a, b, out_dtype=tl.float16)
        tl.static_assert(c.dtype == tl.int32)
    elif CASE == 2:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.int32), out_dtype=tl.float32)
    elif CASE == 3:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.float32))
    else:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.int32), max_num_imprecise_acc=32)
    tl.static_assert(c.dtype == tl.int32)
    tl.store(c_ptr + r[:, None] * 32 + r[None, :], c)


def i5():
    out = []
    g = torch.Generator(device="cuda").manual_seed(145)
    s8 = lambda *s: torch.randint(-128, 128, s, dtype=torch.int8, device="cuda", generator=g)
    u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device="cuda", generator=g)
    cases = [("int8 x uint8", 0, s8(32, 32), u8(32, 32), 32), ("uint8 x uint8", 0, u8(32, 32), u8(32, 32), 32),
             ("int8 x int8, K = 16", 0, s8(32, 16), s8(16, 32), 16),
             ("out_dtype=tl.float16, no accumulator", 1, s8(32, 32), s8(32, 32), 32),
             ("int32 accumulator, out_dtype=tl.float32", 2, s8(32, 32), s8(32, 32), 32),
             ("float32 accumulator", 3, s8(32, 32), s8(32, 32), 32),
             ("max_num_imprecise_acc=32", 4, s8(32, 64), s8(64, 32), 64)]
    for name, case, a, b, kk in cases:
        c = torch.zeros(32, 32, dtype=torch.int32, device="cuda")
        try:
            k_fe[(1,)](a, b, c, case, kk)
            torch.cuda.synchronize()
            ref = a.to(torch.float64) @ b.to(torch.float64)
            ok = bool((c.to(torch.float64) == ref).all().item())
            out.append(f"{name}: accepted, an int32 result, {'exact' if ok else 'NOT exact'}")
        except Exception as e:
            chain, x = [], e
            while x is not None:
                chain.append(str(x))
                x = x.__cause__ or x.__context__
            lines = [l.strip() for l in chain[-1].splitlines() if l.strip()]
            msg = re.sub(r"(/[^\s:'\"]+)+", "<path>", (lines or [type(e).__name__])[0])[:140]
            out.append(f"{name}: rejected ({type(e).__name__}: {msg})")
    return "I5: " + "; ".join(out)


def i6():
    g = torch.Generator(device="cuda").manual_seed(146)
    out = []
    for cfg in CONFIGS:
        BM, BN, BK, nw = cfg
        a, b, c0 = rand_case(BM, BN, BK, g)
        _, ck = run(a, b, c0, cfg)
        ptx = ck.asm["ptx"]
        ins = sorted(set(re.findall(r"(wgmma\.mma_async[\w.]*|mma\.sync[\w.]*|mad\.lo\.s32|dp4a[\w.]*)", ptx)))
        out.append(f"{cfg}: {', '.join(ins) if ins else 'none of mma/wgmma/dp4a/mad'}")
    return "I6: " + "; ".join(out)


CASES = {"I1": i1, "I2": i2, "I3": i3, "I4": i4, "I5": i5, "I6": i6}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(CASES[sys.argv[1]]())
        sys.exit(0)
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for c in CASES:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, TRITON_CACHE_DIR=cache)
            r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), c], env=env,
                               capture_output=True, text=True)
            last = [re.sub(r"(/[^\s:'\"]+)+", "<path>", l) for l in r.stderr.strip().splitlines()[-3:]]
            mine = [l for l in r.stdout.splitlines() if l.startswith(c + ":") or l.startswith(c + " ")]
            print(mine[-1] if mine else f"{c}: failed: {last}")
    print("Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-int8/run.py` from TVA's root.")
