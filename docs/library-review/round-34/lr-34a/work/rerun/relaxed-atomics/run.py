"""TVA library evidence (library issue L157): atomics whose result is unused, with `sem` other than `acq_rel`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/relaxed-atomics/run.py
Output: evidence/relaxed-atomics/out.txt (README.md says what it shows).

Part P, what Triton 3.8.0 compiles them to. For each `tl.atomic_*` (add on int32, float32, float16,
bfloat16, float64; max, min on int32 and float32; and, or, xor, xchg on int32; xchg on float32; cas
on int32), as a tensor atomic and as a scalar one, with `sem` relaxed, acquire, release and acq_rel and
`scope` gpu and sys, the result unused and the result stored: the memory instructions of the PTX
(`atom`, `red`, `ld`, `st`, `cp.reduce`) on the atomic's pointer, and whether a barrier was added.

Part A, atomicity under contention. G programs, each of B lanes; lane j of program p updates cell
(p * 7 + j * 13) % C of a small array (so programs collide on every cell), with the result unused,
for each op and `sem` (acq_rel as the control). The value each lane gives is chosen so that the final
memory is the same for every order of the updates (integers; floats with small integer values, whose
every partial sum is exact): any lost, doubled or torn update shows as a different value. For `xchg`
and `cas` (whose final value depends on the order) the check is that each cell ends as one of the
values written to it (`cas`: compare with 0, initial 0, so exactly one succeeds per cell). Each case
is launched R times.

Part M, other memory in the same programs. Each program writes its own block with plain stores
before and after its relaxed atomics (to many shared cells, in a loop of T trips), and loads an input
it alone reads; the final memory is checked in full: its own block, and the shared cells.

Part L, the split-K pattern of vLLM's `_lora_shrink_kernel` (claims/lora_shrink_bf16, line 305):
`bfloat16` tiles, `tl.dot` per part of K, `accumulator *= scaling`, then
`tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")` into a zeroed `float32` output, for
SPLIT_K 8 and 64. With integer data (every value in {-2, ..., 2}, `scaling = 1`) every partial sum
is exact, so the output must equal the exact product; with random data, the output is compared with
the float64 product, and with the same launch at `acq_rel`.
"""
import re
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


SEMS = ["relaxed", "acquire", "release", "acq_rel"]

# ---------------------------------------------------------------- Part P


@triton.jit
def p_tensor(ptr, val_ptr, res_ptr, OP: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr,
             USE: tl.constexpr, B: tl.constexpr):
    j = tl.arange(0, B)
    v = tl.load(val_ptr + j)
    if OP == "add":
        r = tl.atomic_add(ptr + j, v, mask=j < B, sem=SEM, scope=SCOPE)
    elif OP == "max":
        r = tl.atomic_max(ptr + j, v, sem=SEM, scope=SCOPE)
    elif OP == "min":
        r = tl.atomic_min(ptr + j, v, sem=SEM, scope=SCOPE)
    elif OP == "and":
        r = tl.atomic_and(ptr + j, v, sem=SEM, scope=SCOPE)
    elif OP == "or":
        r = tl.atomic_or(ptr + j, v, sem=SEM, scope=SCOPE)
    elif OP == "xor":
        r = tl.atomic_xor(ptr + j, v, sem=SEM, scope=SCOPE)
    elif OP == "xchg":
        r = tl.atomic_xchg(ptr + j, v, sem=SEM, scope=SCOPE)
    else:
        r = tl.atomic_cas(ptr + j, v * 0, v, sem=SEM, scope=SCOPE)
    if USE:
        tl.store(res_ptr + j, r)


@triton.jit
def p_scalar(ptr, val_ptr, res_ptr, OP: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr,
             USE: tl.constexpr):
    v = tl.load(val_ptr)
    if OP == "add":
        r = tl.atomic_add(ptr, v, sem=SEM, scope=SCOPE)
    elif OP == "max":
        r = tl.atomic_max(ptr, v, sem=SEM, scope=SCOPE)
    elif OP == "min":
        r = tl.atomic_min(ptr, v, sem=SEM, scope=SCOPE)
    elif OP == "and":
        r = tl.atomic_and(ptr, v, sem=SEM, scope=SCOPE)
    elif OP == "or":
        r = tl.atomic_or(ptr, v, sem=SEM, scope=SCOPE)
    elif OP == "xor":
        r = tl.atomic_xor(ptr, v, sem=SEM, scope=SCOPE)
    elif OP == "xchg":
        r = tl.atomic_xchg(ptr, v, sem=SEM, scope=SCOPE)
    else:
        r = tl.atomic_cas(ptr, v * 0, v, sem=SEM, scope=SCOPE)
    if USE:
        tl.store(res_ptr, r)


P_CASES = [("add", torch.int32), ("add", torch.float32), ("add", torch.float16), ("add", torch.bfloat16),
           ("add", torch.float64), ("max", torch.int32), ("max", torch.float32), ("min", torch.int32),
           ("min", torch.float32), ("and", torch.int32), ("or", torch.int32), ("xor", torch.int32),
           ("xchg", torch.int32), ("xchg", torch.float32), ("cas", torch.int32)]

MEM_RE = re.compile(r"^\s*(?:@%p\d+\s+)?((?:atom|red|cp\.reduce)[\w.:]*)", re.M)
BAR_RE = re.compile(r"\bbar\.sync\b|\bbarrier\.sync\b")


def instrs(ptx):
    body = ptx.split(".entry", 1)[1]
    return sorted(set(MEM_RE.findall(body))), bool(BAR_RE.search(body))


def ld_scope(ptx):
    body = ptx.split(".entry", 1)[1]
    return sorted(set(re.findall(r"\b(ld\.global\.(?:gpu|sys|cta)\.[\w.]+)", body)))


def part_p():
    out("== Part P: the instructions (B = 128 lanes, 4 warps)")
    out("| op | dtype | form | result | sem | scope | atomic instructions | strong loads | barrier |")
    out("|---|---|---|---|---|---|---|---|---|")
    B = 128
    for op, dt in P_CASES:
        for form in ["tensor", "scalar"]:
            for use in [False, True]:
                for sem in SEMS:
                    for scope in ["gpu", "sys"]:
                        if form == "scalar" or use:
                            # scalar and result-used forms: relaxed and acq_rel at gpu are enough
                            if scope == "sys" or sem in ("acquire", "release"):
                                continue
                        ptr = torch.zeros(B, device=DEV, dtype=dt)
                        val = torch.ones(B, device=DEV, dtype=dt)
                        res = torch.zeros(B, device=DEV, dtype=dt)
                        if form == "tensor":
                            h = p_tensor[(1,)](ptr, val, res, op, sem, scope, use, B)
                        else:
                            h = p_scalar[(1,)](ptr, val, res, op, sem, scope, use)
                        ins, bar = instrs(h.asm["ptx"])
                        lds = ld_scope(h.asm["ptx"])
                        out(f"| {op} | {str(dt)[6:]} | {form} | {'used' if use else 'unused'} | {sem} | {scope} | "
                            f"{', '.join('`' + i + '`' for i in ins) or 'none'} | "
                            f"{', '.join('`' + i + '`' for i in lds) or 'none'} | {bar} |")
    # the scalar add of a constant zero (Triton's LoadAcquireOp path)
    for sem in ["relaxed", "acquire", "acq_rel"]:
        for use in [False, True]:
            ptr = torch.full((1,), -0.0, device=DEV)
            res = torch.zeros(1, device=DEV)
            h = p_zero[(1,)](ptr, res, sem, use)
            torch.cuda.synchronize()
            ins, bar = instrs(h.asm["ptx"])
            out(f"scalar atomic_add(p, 0.0, sem={sem}), result {'used' if use else 'unused'}: atomic "
                f"{ins or 'none'}, strong loads {ld_scope(h.asm['ptx']) or 'none'}; a cell holding -0.0 "
                f"ends as {'-0.0' if torch.signbit(ptr)[0] else '+0.0'}")


@triton.jit
def p_zero(ptr, res_ptr, SEM: tl.constexpr, USE: tl.constexpr):
    r = tl.atomic_add(ptr, 0.0, sem=SEM)
    if USE:
        tl.store(res_ptr, r)


# ---------------------------------------------------------------- Part A


@triton.jit
def a_kernel(ptr, OP: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
             FLOAT: tl.constexpr, CONTIG: tl.constexpr):
    p = tl.program_id(0)
    j = tl.arange(0, B)
    if CONTIG:
        cell = (p * 8 + j) % C
    else:
        cell = (p * 7 + j * 13) % C
    w = ((p * 31 + j * 17) % 5 - 2)  # in {-2, ..., 2}
    if FLOAT:
        v = w.to(ptr.dtype.element_ty)
    else:
        v = w
    if OP == "add":
        tl.atomic_add(ptr + cell, v, sem=SEM, scope=SCOPE)
    elif OP == "max":
        tl.atomic_max(ptr + cell, p * B + j, sem=SEM, scope=SCOPE)
    elif OP == "min":
        tl.atomic_min(ptr + cell, p * B + j, sem=SEM, scope=SCOPE)
    elif OP == "and":
        tl.atomic_and(ptr + cell, ~(1 << ((p + j) % 31)), sem=SEM, scope=SCOPE)
    elif OP == "or":
        tl.atomic_or(ptr + cell, 1 << ((p * 3 + j) % 31), sem=SEM, scope=SCOPE)
    elif OP == "xor":
        tl.atomic_xor(ptr + cell, (p * 40503 + j * 9973) & 0x7FFFFFFF, sem=SEM, scope=SCOPE)
    elif OP == "xchg":
        tl.atomic_xchg(ptr + cell, p * B + j + 1, sem=SEM, scope=SCOPE)
    else:
        tl.atomic_cas(ptr + cell, cell * 0, p * B + j + 1, sem=SEM, scope=SCOPE)


def a_expected(op, G, B, C, contig):
    p = torch.arange(G, dtype=torch.int64).repeat_interleave(B)
    j = torch.arange(B, dtype=torch.int64).repeat(G)
    cell = (p * 8 + j) % C if contig else (p * 7 + j * 13) % C
    if op == "add":
        w = ((p * 31 + j * 17) % 5 - 2)
        e = torch.zeros(C, dtype=torch.int64).index_add_(0, cell, w)
        return e, None
    if op == "max":
        return torch.full((C,), -(2 ** 31), dtype=torch.int64).scatter_reduce_(0, cell, p * B + j, "amax"), None
    if op == "min":
        return torch.full((C,), 2 ** 31 - 1, dtype=torch.int64).scatter_reduce_(0, cell, p * B + j, "amin"), None
    if op == "and":
        e = torch.full((C,), -1, dtype=torch.int64)
        m = ~(1 << ((p + j) % 31))
        for c in range(C):
            acc = -1
            for x in m[cell == c].tolist():
                acc &= x
            e[c] = acc
        return e, None
    if op == "or":
        e = torch.zeros(C, dtype=torch.int64)
        m = 1 << ((p * 3 + j) % 31)
        for c in range(C):
            acc = 0
            for x in m[cell == c].tolist():
                acc |= x
            e[c] = acc
        return e, None
    if op == "xor":
        e = torch.zeros(C, dtype=torch.int64)
        m = (p * 40503 + j * 9973) & 0x7FFFFFFF
        for c in range(C):
            acc = 0
            for x in m[cell == c].tolist():
                acc ^= x
            e[c] = acc
        return e, None
    # xchg, cas: the set of values written to each cell
    vals = p * B + j + 1
    return None, [set(vals[cell == c].tolist()) for c in range(C)]


A_CASES = [("add", torch.int32), ("add", torch.float32), ("add", torch.float16), ("add", torch.bfloat16),
           ("add", torch.float64), ("max", torch.int32), ("min", torch.int32), ("and", torch.int32),
           ("or", torch.int32), ("xor", torch.int32), ("xchg", torch.int32), ("cas", torch.int32)]


def a_init(op, C, dt):
    if op == "max":
        return torch.full((C,), -(2 ** 31), device=DEV, dtype=dt)
    if op == "min":
        return torch.full((C,), 2 ** 31 - 1, device=DEV, dtype=dt)
    if op == "and":
        return torch.full((C,), -1, device=DEV, dtype=dt)
    return torch.zeros(C, device=DEV, dtype=dt)


def part_a(B=256, C=64, R=50):
    out(f"== Part A: atomicity under contention (G programs x B = {B} lanes onto C = {C} cells, R = {R} launches "
        f"per case, 4 warps; G = 4096, 16384 updates per cell, except float16 and bfloat16: G = 32, 128 updates per "
        f"cell, so that every partial sum is exact; lanes scattered (cell (7p + 13j) mod C) or contiguous "
        f"(cell (8p + j) mod C, which Triton may pack or vectorise))")
    out("| op | dtype | lanes | sem | scope | instruction | launches | cells checked | cells wrong |")
    out("|---|---|---|---|---|---|---|---|---|")
    total_bad = 0
    for op, dt in A_CASES:
      for contig in [False, True]:
        G = 32 if dt in (torch.float16, torch.bfloat16) else 4096
        exp, sets = a_expected(op, G, B, C, contig)
        for sem in SEMS:
            for scope in (["gpu", "sys"] if sem == "relaxed" else ["gpu"]):
                bad = 0
                for r in range(R):
                    buf = a_init(op, C, dt)
                    h = a_kernel[(G,)](buf, op, sem, scope, C, B, dt.is_floating_point, contig)
                    torch.cuda.synchronize()
                    if exp is not None:
                        got = buf.to(torch.float64).to(torch.int64).cpu() if dt.is_floating_point else buf.cpu().to(
                            torch.int64)
                        e = exp
                        if not dt.is_floating_point:
                            e = ((exp + 2 ** 31) % 2 ** 32) - 2 ** 31  # int32 wraps
                        bad += int((got != e).sum())
                    else:
                        got = buf.cpu().tolist()
                        bad += sum(1 for c in range(C) if got[c] not in sets[c])
                total_bad += bad
                ins = ", ".join("`" + i + "`" for i in instrs(h.asm["ptx"])[0])
                out(f"| {op} | {str(dt)[6:]} | {'contiguous' if contig else 'scattered'} | {sem} | {scope} | {ins} | {R} | "
                    f"{R * C} | {bad} |")
    out(f"Part A: cells wrong in all cases: {total_bad}")


# ---------------------------------------------------------------- Part M


@triton.jit
def m_kernel(own_ptr, in_ptr, acc_ptr, SEM: tl.constexpr, B: tl.constexpr, C: tl.constexpr, T: tl.constexpr):
    p = tl.program_id(0)
    j = tl.arange(0, B)
    x = tl.load(in_ptr + p * B + j)
    tl.store(own_ptr + p * (2 * B) + j, x * 3 + 1)
    for t in range(T):
        cell = (p * 5 + j * 11 + t * 3) % C
        tl.atomic_add(acc_ptr + cell, x + t, sem=SEM)
        tl.atomic_add(acc_ptr + C + (cell * 7) % C, 1, sem=SEM)
    tl.store(own_ptr + p * (2 * B) + B + j, x - 5)


def part_m(G=2048, B=128, C=128, T=8, R=50):
    out(f"== Part M: plain stores and loads beside the atomics (G = {G}, B = {B}, C = {C} cells x 2, T = {T} "
        f"trips, R = {R} launches)")
    torch.manual_seed(1)
    x = torch.randint(-1000, 1000, (G * B,), device=DEV, dtype=torch.int32)
    p = torch.arange(G, dtype=torch.int64).repeat_interleave(B)
    j = torch.arange(B, dtype=torch.int64).repeat(G)
    xc = x.cpu().to(torch.int64)
    e_acc = torch.zeros(2 * C, dtype=torch.int64)
    for t in range(T):
        cell = (p * 5 + j * 11 + t * 3) % C
        e_acc.index_add_(0, cell, xc + t)
        e_acc.index_add_(0, C + (cell * 7) % C, torch.ones_like(cell))
    e_own = torch.empty(G, 2 * B, dtype=torch.int64)
    e_own[:, :B] = (xc * 3 + 1).view(G, B)
    e_own[:, B:] = (xc - 5).view(G, B)
    for sem in ["relaxed", "acq_rel"]:
        bad_acc = bad_own = 0
        for r in range(R):
            own = torch.full((G * 2 * B,), -7, device=DEV, dtype=torch.int32)
            acc = torch.zeros(2 * C, device=DEV, dtype=torch.int32)
            m_kernel[(G,)](own, x, acc, sem, B, C, T)
            torch.cuda.synchronize()
            bad_acc += int((acc.cpu().to(torch.int64) != e_acc).sum())
            bad_own += int((own.cpu().to(torch.int64).view(G, 2 * B) != e_own).sum())
        out(f"sem={sem}: shared cells wrong {bad_acc} of {R * 2 * C}; private elements wrong {bad_own} of "
            f"{R * G * 2 * B}")


# ---------------------------------------------------------------- Part L


@triton.jit
def l_kernel(a_ptr, b_ptr, c_ptr, M, N, K, scaling, SEM: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_sk_m_n = tl.program_id(0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_m_n // num_pid_n
    pid_n = pid_m_n % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    STEP_K = BLOCK_K * SPLIT_K
    for k in range(0, tl.cdiv(K, STEP_K)):
        kk = k * STEP_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + kk < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] + kk < K), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += STEP_K
        b_ptrs += STEP_K
    accumulator *= scaling
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem=SEM)


def part_l(R=20):
    out(f"== Part L: the split-K pattern of _lora_shrink_kernel (R = {R} launches per case)")
    torch.manual_seed(2)
    for (M, N, K, BK, SK) in [(16, 16, 4096, 256, 64), (128, 64, 4096, 32, 8), (100, 48, 14336, 256, 64),
                              (256, 32, 14336, 32, 8)]:
        BM, BN = 32, 16
        grid = (SK * triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        # integer data: every partial sum exact
        a = torch.randint(-2, 3, (M, K), device=DEV).to(torch.bfloat16)
        b = torch.randint(-2, 3, (N, K), device=DEV).to(torch.bfloat16)
        ref = (a.double() @ b.double().T)
        bad = 0
        for r in range(R):
            c = torch.zeros(M, N, device=DEV, dtype=torch.float32)
            h = l_kernel[grid](a, b, c, M, N, K, 1.0, "relaxed", BM, BN, BK, SK, num_warps=4, num_stages=2)
            torch.cuda.synchronize()
            bad += int((c.double() != ref).sum())
        out(f"M={M} N={N} K={K} BLOCK_K={BK} SPLIT_K={SK} ({grid[0]} programs), integer data, relaxed: "
            f"elements not exact {bad} of {R * M * N}")
        # random data: compare relaxed and acq_rel against float64
        a = torch.randn(M, K, device=DEV).to(torch.bfloat16)
        b = torch.randn(N, K, device=DEV).to(torch.bfloat16)
        ref = (a.double() @ b.double().T) * 0.5
        mags = (a.double().abs() @ b.double().abs().T) * 0.5
        for sem in ["relaxed", "acq_rel"]:
            worst = 0.0
            for r in range(R):
                c = torch.zeros(M, N, device=DEV, dtype=torch.float32)
                l_kernel[grid](a, b, c, M, N, K, 0.5, sem, BM, BN, BK, SK, num_warps=4, num_stages=2)
                torch.cuda.synchronize()
                worst = max(worst, float(((c.double() - ref).abs() / mags).max()))
            out(f"    random data, {sem}: largest |c - ref| / sum|a*b|*scaling over {R} launches {worst:.3e}")
    ins, bar = instrs(h.asm["ptx"])
    out(f"Part L instructions: {ins}; barrier {bar}")


def main():
    t0 = time.time()
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    part_p()
    part_a()
    part_m()
    part_l()
    out(f"time {time.time() - t0:.0f} s")
    with open(__file__.rsplit("/", 1)[0] + "/out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
