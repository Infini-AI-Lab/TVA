"""lr-31b: further tests of DotI8 (library issue L141), beyond evidence/num-dot-int8.

  X1  operand layouts: b given as [N, K] and transposed in the kernel (tl.trans, .T), a as [K, M]
      transposed, both transposed; mma.sync and wgmma configurations
  X2  operands made in registers (int32 loads narrowed with .to(tl.int8)), not loaded as int8
  X3  masked K tail (K not a multiple of BK, other=0), pipelined (num_stages 4), and wide tiles
      (wgmma n256, n8), num_warps 8
  X4  the room's edge with mixed signs on transposed operands, K large, both paths
  X5  accumulators of other integer dtypes (uint32, int16, int64), with and without out_dtype,
      and out_dtype=tl.uint32 with an int32 accumulator: accepted or rejected on the host, and what
      the result's dtype is
  X6  acc += tl.dot(a, b) with acc near 2^31 - 1 (the combine pass's dot_add): which result

Each case runs in its own process with its own empty Triton cache. Exact results in float64.
Run from TVA's root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_lay(a_ptr, b_ptr, c0_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          AT: tl.constexpr, BT: tl.constexpr, REG: tl.constexpr, MASK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.load(c0_ptr + rm[:, None] * N + rn[None, :])
    for k in range(0, K, BK):
        kk = k + rk
        if AT:  # a stored as [K, M]
            at = tl.load(a_ptr + kk[:, None].to(tl.int64) * M + rm[None, :], mask=(kk[:, None] < K) | (not MASK), other=0)
            a = tl.trans(at)
        else:
            a = tl.load(a_ptr + rm[:, None].to(tl.int64) * K + kk[None, :], mask=(kk[None, :] < K) | (not MASK), other=0)
        if BT:  # b stored as [N, K]
            bt = tl.load(b_ptr + rn[:, None].to(tl.int64) * K + kk[None, :], mask=(kk[None, :] < K) | (not MASK), other=0)
            b = bt.T
        else:
            b = tl.load(b_ptr + kk[:, None].to(tl.int64) * N + rn[None, :], mask=(kk[:, None] < K) | (not MASK), other=0)
        if REG:
            a = (a.to(tl.int32) * 1).to(tl.int8)
            b = (b.to(tl.int32) + 0).to(tl.int8)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def exact(a, b, c0):
    return (a.to(torch.float64) @ b.to(torch.float64)) + c0.to(torch.float64)


def launch(a, b, c0, cfg, at=False, bt=False, reg=False, mask=False, stages=3):
    BM, BN, BK, nw = cfg
    M, K = a.shape
    N = b.shape[1]
    ap = a.t().contiguous() if at else a.contiguous()
    bp = b.t().contiguous() if bt else b.contiguous()
    c = torch.empty(M, N, dtype=torch.int32, device="cuda")
    ck = k_lay[(M // BM, N // BN)](ap, bp, c0, c, M, N, K, BM=BM, BN=BN, BK=BK, AT=at, BT=bt, REG=reg,
                                   MASK=mask, num_warps=nw, num_stages=stages)
    return c, ck


def rand_case(M, N, K, g):
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device="cuda", generator=g)
    a[0, :] = -128
    b[:, 0] = -128
    mag = a.to(torch.float64).abs() @ b.to(torch.float64).abs()
    lim = (2**31 - 1) - mag
    assert bool((lim >= 0).all().item())
    u = torch.rand(M, N, dtype=torch.float64, device="cuda", generator=g) * 2 - 1
    c0 = torch.trunc(u * lim).to(torch.int32)
    return a, b, c0


def instr(ck):
    return ",".join(sorted(set(re.findall(r"(wgmma\.mma_async[\w.]*|mma\.sync[\w.]*|mad\.lo\.s32|dp4a[\w.]*)",
                                          ck.asm["ptx"])))) or "none"


CFGS = [(16, 16, 32, 1), (32, 64, 64, 4), (64, 64, 64, 4), (128, 128, 64, 8), (64, 256, 32, 4), (64, 8, 32, 4)]


def x1():
    g = torch.Generator(device="cuda").manual_seed(311)
    bad = tot = 0
    ins = set()
    for cfg in CFGS:
        BM, BN, BK, nw = cfg
        a, b, c0 = rand_case(2 * BM, 2 * BN, 512, g)
        ref = exact(a, b, c0)
        for at, bt in ((False, True), (True, False), (True, True)):
            c, ck = launch(a, b, c0, cfg, at=at, bt=bt)
            bad += int((c.to(torch.float64) != ref).sum().item())
            tot += c.numel()
            ins.add(f"{cfg}{'AT' if at else ''}{'BT' if bt else ''}:{instr(ck)}")
    return f"X1: {len(CFGS)} configurations x 3 transposed layouts, {tot} outputs: {bad} differ; instructions " + "; ".join(sorted(ins))


def x2():
    g = torch.Generator(device="cuda").manual_seed(312)
    bad = tot = 0
    ins = []
    for cfg in CFGS:
        BM, BN, BK, nw = cfg
        a, b, c0 = rand_case(2 * BM, 2 * BN, 256, g)
        ref = exact(a, b, c0)
        for bt in (False, True):
            c, ck = launch(a, b, c0, cfg, bt=bt, reg=True)
            bad += int((c.to(torch.float64) != ref).sum().item())
            tot += c.numel()
            ins.append(f"{cfg}:{instr(ck)}")
    return f"X2: operands narrowed in registers, {len(CFGS)} configurations x 2 layouts, {tot} outputs: {bad} differ; " + "; ".join(sorted(set(ins)))


def x3():
    g = torch.Generator(device="cuda").manual_seed(313)
    bad = tot = 0
    out = []
    for cfg in CFGS:
        BM, BN, BK, nw = cfg
        for K in (BK * 7 + 5, 1000, 2 * BK + 1):
            a, b, c0 = rand_case(2 * BM, 2 * BN, K, g)
            ref = exact(a, b, c0)
            for st in (1, 4):
                c, ck = launch(a, b, c0, cfg, mask=True, stages=st)
                d = int((c.to(torch.float64) != ref).sum().item())
                bad += d
                tot += c.numel()
                if d:
                    out.append(f"{cfg} K={K} stages={st}: {d}")
    return f"X3: masked K tail (K in {{7BK+5, 1000, 2BK+1}}), num_stages 1 and 4, {len(CFGS)} configurations, {tot} outputs: {bad} differ" + ("; " + "; ".join(out) if out else "")


def x4():
    out = []
    K = 65536
    for cfg in ((16, 16, 32, 1), (64, 64, 64, 4), (64, 256, 32, 4)):
        BM, BN, BK, nw = cfg
        M, N = BM, BN
        # alternating signs: a = -128 everywhere, b alternates -128 / 127 in blocks of 32 along k
        a = torch.full((M, K), -128, dtype=torch.int8, device="cuda")
        b = torch.full((K, N), -128, dtype=torch.int8, device="cuda")
        b.view(-1, 64, N)[:, 32:, :] = 127
        s = exact(a, b, torch.zeros(M, N, dtype=torch.int32, device="cuda"))
        mag = a.to(torch.float64).abs() @ b.to(torch.float64).abs()
        # acc so that |acc| + sum|a*b| = 2^31 - 1 exactly
        c0 = ((2**31 - 1) - mag).to(torch.int32)
        ref = s + c0.to(torch.float64)
        res = []
        for at, bt in ((False, False), (True, True)):
            c, ck = launch(a, b, c0, cfg, at=at, bt=bt)
            res.append("exact" if bool((c.to(torch.float64) == ref).all().item()) else "WRONG")
        out.append(f"{cfg}: {res}")
    return f"X4: K = {K}, |acc| + sum|a*b| = 2^31 - 1, alternating signs, plain and transposed: " + "; ".join(out)


@triton.jit
def k_acc(a_ptr, b_ptr, c_ptr, CASE: tl.constexpr):
    r = tl.arange(0, 32)
    a = tl.load(a_ptr + r[:, None] * 32 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 32 + r[None, :])
    if CASE == 0:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.uint32))
    elif CASE == 1:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.uint32), out_dtype=tl.uint32)
    elif CASE == 2:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.int16))
    elif CASE == 3:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.int64))
    elif CASE == 4:
        c = tl.dot(a, b, tl.zeros([32, 32], dtype=tl.int32), out_dtype=tl.uint32)
    elif CASE == 5:
        c = tl.dot(a, b, out_dtype=tl.uint32)
    else:
        c = tl.dot(a, b, out_dtype=tl.int8)
    tl.store(c_ptr + r[:, None] * 32 + r[None, :], c.to(tl.int32))
    tl.static_print("RESULT_DTYPE", c.dtype)


def x5():
    out = []
    g = torch.Generator(device="cuda").manual_seed(315)
    names = ["uint32 acc", "uint32 acc, out_dtype=uint32", "int16 acc", "int64 acc",
             "int32 acc, out_dtype=uint32", "no acc, out_dtype=uint32", "no acc, out_dtype=int8"]
    for case, name in enumerate(names):
        a = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device="cuda", generator=g)
        b = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device="cuda", generator=g)
        c = torch.zeros(32, 32, dtype=torch.int32, device="cuda")
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "X5one", str(case)],
                           env=os.environ, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).strip().splitlines()
        res = [l for l in lines if l.startswith("X5one")]
        if res:
            out.append(f"{name}: {res[-1][6:]}")
        else:
            errs = [l for l in lines if "Error" in l or "error" in l]
            msg = re.sub(r"(/[^\s:'\"]+)+", "<path>", (errs[-1] if errs else (lines[-1] if lines else "?")))[:160]
            out.append(f"{name}: rejected ({msg})")
    return "X5: " + "; ".join(out)


def x5one(case):
    g = torch.Generator(device="cuda").manual_seed(316)
    a = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device="cuda", generator=g)
    b = torch.randint(-128, 128, (32, 32), dtype=torch.int8, device="cuda", generator=g)
    c = torch.zeros(32, 32, dtype=torch.int32, device="cuda")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        k_acc[(1,)](a, b, c, case)
    torch.cuda.synchronize()
    dt = [l for l in buf.getvalue().splitlines() if "RESULT_DTYPE" in l]
    ok = bool((c.to(torch.float64) == a.to(torch.float64) @ b.to(torch.float64)).all().item())
    print(f"X5one accepted, result dtype {dt[-1].split('RESULT_DTYPE')[-1].strip() if dt else '?'}, "
          f"{'exact' if ok else 'not exact'}")


@triton.jit
def k_fuse(a_ptr, b_ptr, c0_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.load(c0_ptr + rm[:, None] * BN + rn[None, :])
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], acc)


def x6():
    out = []
    for cfg in ((16, 16, 32, 1), (64, 64, 32, 4)):
        BM, BN, BK, nw = cfg
        a = torch.full((BM, BK), 1, dtype=torch.int8, device="cuda")
        b = torch.full((BK, BN), 3, dtype=torch.int8, device="cuda")  # sum = 96
        c0 = torch.full((BM, BN), 2**31 - 1 - 10, dtype=torch.int32, device="cuda")
        c = torch.empty_like(c0)
        ck = k_fuse[(1,)](a, b, c0, c, BM=BM, BN=BN, BK=BK, num_warps=nw)
        v = c[0, 0].item()
        wrap = 2**31 - 1 - 10 + 96 - 2**32
        kind = "wrapped (unfused, or fused on wgmma)" if v == wrap else ("saturated (fused into mma.sync)" if v == 2**31 - 1 else "other")
        fused = "tt.dot" in ck.asm["ttir"] and not re.search(r"arith\.addi %\w+, %\w+ : tensor<\d+x\d+xi32", ck.asm["ttir"])
        out.append(f"{cfg}: {v} ({kind}; TTIR {'one tt.dot, no tensor addi' if fused else 'tt.dot then arith.addi'})")
    return "X6 (beyond the room; acc += tl.dot(a, b), acc = 2^31 - 11, sum 96): " + "; ".join(out)


CASES = {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "X6": x6}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "X5one":
        x5one(int(sys.argv[2]))
        sys.exit(0)
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
