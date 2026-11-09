"""lr-32b, round 32, L149: adversarial cases for SumDepth that evidence/reduce-tree did not run.

Probe (as evidence/reduce-tree F): every leaf 1 + 2^-23 (float32), separately 1 - 2^-24; a sum of
n = 2^k such leaves is exact along a perfect tree and rounded at the first node of any other tree.
Cases:
  R  operands in layouts the evidence did not reach: a 1-D load reshaped to 2-D (and 3-D) and
     reduced along each axis; tl.permute of a 3-D tile; tl.join / tl.interleave results; a 3-D tile
     reduced along axis 0 and 1; tl.trans of a wgmma result with N = 256; a sum of the operand of a
     tl.dot (converted in registers)
  B  1-D axes of 32768 to 131072 positions; num_warps 32
  C  lanes agree: tl.sum(x * y, axis=1) with x, y loaded in a layout that holds one element of the
     axis per thread, its result broadcast back and stored by every thread (fusion on): do all the
     copies of one row's result agree?
  L  OptimizeThreadLocality: (a) acc = acc - tl.sum(x, 1); (b) acc = tl.maximum(acc, tl.sum(x, 1));
     (c) tl.sum(tl.load(p) * 1.0, 1); (d) acc = tl.sum(x, 1) + acc; against every association of the
     source

Run from TVA's root: CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I docs/library-review/round-32/lr-32b/work/sum_depth.py
"""
import os, re, tempfile
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="lr32b-"))
import torch, triton, triton.language as tl

LEAVES = [1 + 2.0 ** -23, 1 - 2.0 ** -24]
FAIL = []


def layouts(h):
    g = h.asm["ttgir"]
    return sorted(set(re.findall(r"#(blocked|linear|mma|slice|dot_op)\d*\s*=", g)))


def check(name, out, n, leaf):
    want = torch.tensor(n * leaf, dtype=torch.float64)
    ok = bool(torch.all(out.double() == want))
    if not ok:
        FAIL.append(name)
    return ok


@triton.jit
def k_reshape(x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, AXIS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, A * B))
    y = tl.reshape(x, [A, B])
    s = tl.sum(y, axis=AXIS)
    L: tl.constexpr = B if AXIS == 0 else A
    tl.store(o_ptr + tl.arange(0, L), s)


@triton.jit
def k_reshape3(x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, AXIS: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, A * B * C))
    y = tl.reshape(x, [A, B, C])
    s = tl.sum(y, axis=AXIS)
    s = tl.reshape(s, [A * B * C // (A if AXIS == 0 else (B if AXIS == 1 else C))])
    tl.store(o_ptr + tl.arange(0, A * B * C // (A if AXIS == 0 else (B if AXIS == 1 else C))), s)


@triton.jit
def k_permute(x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, AXIS: tl.constexpr):
    o = (tl.arange(0, A)[:, None, None] * (B * C) + tl.arange(0, B)[None, :, None] * C
         + tl.arange(0, C)[None, None, :])
    x = tl.load(x_ptr + o)
    y = tl.permute(x, (2, 0, 1))  # [C, A, B]
    s = tl.sum(y, axis=AXIS)
    if AXIS == 0:
        tl.store(o_ptr + tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :], s)
    elif AXIS == 1:
        tl.store(o_ptr + tl.arange(0, C)[:, None] * B + tl.arange(0, B)[None, :], s)
    else:
        tl.store(o_ptr + tl.arange(0, C)[:, None] * A + tl.arange(0, A)[None, :], s)


@triton.jit
def k_join(x_ptr, o_ptr, A: tl.constexpr, B: tl.constexpr, AXIS: tl.constexpr, INTER: tl.constexpr):
    o = tl.arange(0, A)[:, None] * B + tl.arange(0, B)[None, :]
    x = tl.load(x_ptr + o)
    y = tl.load(x_ptr + A * B + o)
    if INTER:
        z = tl.interleave(x, y)  # [A, 2B]
        s = tl.sum(z, axis=AXIS)
        if AXIS == 0:
            tl.store(o_ptr + tl.arange(0, 2 * B), s)
        else:
            tl.store(o_ptr + tl.arange(0, A), s)
    else:
        z = tl.join(x, y)  # [A, B, 2]
        s = tl.sum(z, axis=AXIS)
        if AXIS == 2:
            tl.store(o_ptr + o, s)
        elif AXIS == 1:
            tl.store(o_ptr + tl.arange(0, A)[:, None] * 2 + tl.arange(0, 2)[None, :], s)
        else:
            tl.store(o_ptr + tl.arange(0, B)[:, None] * 2 + tl.arange(0, 2)[None, :], s)


@triton.jit
def k_dott(a_ptr, b_ptr, x_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           AXIS: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    d = tl.dot(a, b)  # zeros
    x = tl.load(x_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    t = tl.trans(d + x)  # [N, M]
    s = tl.sum(t, axis=AXIS)
    L: tl.constexpr = M if AXIS == 0 else N
    tl.store(o_ptr + tl.arange(0, L), s)


@triton.jit
def k_dotop(a_ptr, b_ptr, o_ptr, d_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]).to(tl.float16)
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    s = tl.sum(a, axis=1)  # float16 sum of float16 leaves 1 + 2^-10
    tl.store(o_ptr + tl.arange(0, M), s)
    d = tl.dot(a, b)
    tl.store(d_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], d)


@triton.jit
def k_big(x_ptr, o_ptr, N: tl.constexpr):
    tl.store(o_ptr, tl.sum(tl.load(x_ptr + tl.arange(0, N))))


@triton.jit
def k_lanes(x_ptr, y_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, SM: tl.constexpr):
    o = tl.arange(0, M)[:, None] * SM + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + o)
    y = tl.load(y_ptr + o)
    s = tl.sum(x * y, axis=1)
    z = tl.broadcast_to(s[:, None], [M, N])
    tl.store(o_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], z)


@triton.jit
def k_loop(x_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr, FORM: tl.constexpr):
    rows = tl.arange(0, M)
    if FORM == 1:
        acc = tl.full([M], float("-inf"), tl.float32)
    else:
        acc = tl.zeros([M], tl.float32)
    for t in range(T):
        x = tl.load(x_ptr + t * M * N + rows[:, None] * N + tl.arange(0, N)[None, :])
        if FORM == 0:
            acc = acc - tl.sum(x, 1)
        elif FORM == 1:
            acc = tl.maximum(acc, tl.sum(x, 1))
        elif FORM == 2:
            acc = acc + tl.sum(x * 1.0, 1)
        else:
            acc = tl.sum(x, 1) + acc
    tl.store(o_ptr + rows, acc)


def dev(t):
    return t.cuda()


def run_R():
    n_ok = n = 0
    for leaf in LEAVES:
        for nw in (1, 4, 8):
            for (A, B) in ((16, 64), (64, 16), (4, 1024), (128, 128), (2, 4096)):
                for AXIS in (0, 1):
                    x = dev(torch.full((A * B,), leaf, dtype=torch.float32))
                    L = B if AXIS == 0 else A
                    o = dev(torch.empty(L, dtype=torch.float32))
                    h = k_reshape[(1,)](x, o, A, B, AXIS, num_warps=nw)
                    n += 1; n_ok += check(f"reshape {A}x{B} ax{AXIS} w{nw} leaf{leaf}", o, A if AXIS == 0 else B, leaf)
            for (A, B, C) in ((8, 16, 32), (32, 4, 64), (4, 64, 4)):
                for AXIS in (0, 1, 2):
                    tot = A * B * C
                    x = dev(torch.full((tot,), leaf, dtype=torch.float32))
                    red = (A, B, C)[AXIS]
                    o = dev(torch.empty(tot // red, dtype=torch.float32))
                    k_reshape3[(1,)](x, o, A, B, C, AXIS, num_warps=nw)
                    n += 1; n_ok += check(f"reshape3 {A}x{B}x{C} ax{AXIS} w{nw}", o, red, leaf)
                    x = dev(torch.full((tot,), leaf, dtype=torch.float32))
                    red = (C, A, B)[AXIS]
                    o = dev(torch.empty(tot // red, dtype=torch.float32))
                    k_permute[(1,)](x, o, A, B, C, AXIS, num_warps=nw)
                    n += 1; n_ok += check(f"permute {A}x{B}x{C} ax{AXIS} w{nw}", o, red, leaf)
            for (A, B) in ((16, 64), (64, 32), (128, 8)):
                for INTER, axes in ((0, (0, 1, 2)), (1, (0, 1))):
                    for AXIS in axes:
                        x = dev(torch.full((2 * A * B,), leaf, dtype=torch.float32))
                        if INTER:
                            L = 2 * B if AXIS == 0 else A
                            red = A if AXIS == 0 else 2 * B
                        else:
                            L = A * B if AXIS == 2 else (2 * A if AXIS == 1 else 2 * B)
                            red = (A, B, 2)[AXIS]
                        o = dev(torch.empty(L, dtype=torch.float32))
                        k_join[(1,)](x, o, A, B, AXIS, INTER, num_warps=nw)
                        n += 1; n_ok += check(f"join{INTER} {A}x{B} ax{AXIS} w{nw}", o, red, leaf)
        for (M, N, K, nw) in ((64, 256, 16, 4), (128, 256, 16, 8), (64, 128, 32, 4), (16, 256, 16, 1)):
            for AXIS in (0, 1):
                a = dev(torch.zeros(M, K, dtype=torch.float16)); b = dev(torch.zeros(K, N, dtype=torch.float16))
                x = dev(torch.full((M, N), leaf, dtype=torch.float32))
                L = M if AXIS == 0 else N
                o = dev(torch.empty(L, dtype=torch.float32))
                h = k_dott[(1,)](a, b, x, o, M, N, K, AXIS, num_warps=nw)
                path = "wgmma" if "wgmma" in h.asm["ptx"] else "mma.sync"
                n += 1; n_ok += check(f"dot-trans {M}x{N} {path} ax{AXIS} w{nw}", o, N if AXIS == 0 else M, leaf)
        for (M, N, K, nw) in ((16, 16, 64, 1), (64, 64, 128, 4), (128, 64, 256, 8)):
            a = dev(torch.full((M, K), leaf, dtype=torch.float32))
            a8 = dev(torch.full((M, K), 1.0).to(torch.float8_e5m2))
            b = dev(torch.zeros(K, N, dtype=torch.float16))
            o = dev(torch.empty(M, dtype=torch.float32)); d = dev(torch.empty(M, N, dtype=torch.float32))
            # float16 leaf 1 + 2^-10 for the converted operand
            a16 = dev(torch.full((M, K), 1 + 2.0 ** -10, dtype=torch.float32))
            k_dotop[(1,)](a16, b, o, d, M, N, K, num_warps=nw)
            n += 1; n_ok += check(f"dot-operand {M}x{K} w{nw}", o, K, float(torch.tensor(1 + 2.0 ** -10).half()))
    print(f"R: {n} reductions, {n_ok} exact (a perfect tree over exactly n leaves)", flush=True)


def run_B():
    n_ok = n = 0
    for leaf in LEAVES:
        for N in (32768, 65536, 131072):
            for nw in (4, 8, 16, 32):
                x = dev(torch.full((N,), leaf, dtype=torch.float32))
                o = dev(torch.empty(1, dtype=torch.float32))
                try:
                    k_big[(1,)](x, o, N, num_warps=nw)
                except Exception as ex:
                    print(f"B N={N} w{nw}: {type(ex).__name__}", flush=True); continue
                n += 1; n_ok += check(f"big N={N} w{nw}", o, N, leaf)
        for (M, N) in ((32, 32), (1, 1024), (64, 64)):
            x = dev(torch.full((M * N,), leaf, dtype=torch.float32))
            o = dev(torch.empty(M, dtype=torch.float32))
            k_reshape[(1,)](x, o, M, N, 1, num_warps=32)
            n += 1; n_ok += check(f"w32 {M}x{N}", o, N, leaf)
    print(f"B: {n} reductions, {n_ok} exact", flush=True)


def run_C():
    g = torch.Generator().manual_seed(3)
    bad = tot = 0
    for (M, N, nw) in ((32, 32, 1), (128, 32, 4), (64, 64, 8), (8, 256, 4), (256, 128, 8)):
        for SMx in (N, N + 1):  # contiguous and padded rows
            x = dev(torch.randn(M * SMx, generator=g)); y = dev(torch.randn(M * SMx, generator=g))
            o = dev(torch.empty(M, N, dtype=torch.float32))
            h = k_lanes[(1,)](x, y, o, M, N, SMx, num_warps=nw)
            ptx = h.asm["ptx"]
            nf = len(re.findall(r"\bfma\.rn\.f32", ptx))
            rows_bad = int((o != o[:, :1]).any(dim=1).sum())
            bad += rows_bad; tot += M
            print(f"C {M}x{N} stride {SMx} w{nw}: fma in PTX {nf}; rows whose copies differ {rows_bad} of {M}",
                  flush=True)
    print(f"C: {tot} rows, {bad} with copies that differ", flush=True)


def run_L():
    for FORM, name in ((0, "acc - sum"), (1, "max(acc, sum)"), (2, "acc + sum(x * 1.0)"), (3, "sum + acc")):
        for (M, N, nw) in ((16, 64, 4), (32, 128, 8)):
            T = 2
            x = torch.zeros(T, M, N)
            if FORM == 1:
                # trip 0: row sum 4, trip 1: row sum 1  -> max over trips 4; sum over threads of each
                # thread's max over trips differs
                x[0, :, : N // 2] = 8.0 / (N // 2); x[0, :, N // 2:] = -4.0 / (N // 2)
                x[1, :, : N // 2] = -2.0 / (N // 2); x[1, :, N // 2:] = 3.0 / (N // 2)
                want = 4.0
            else:
                x[0, :, 0] = 2.0 ** 24; x[0, :, N - 1] = 1.0
                x[1, :, 0] = -(2.0 ** 24); x[1, :, N - 1] = 1.0
                want = -1.0 if FORM == 0 else 1.0
            xd = dev(x.contiguous()); o = dev(torch.empty(M, dtype=torch.float32))
            h = k_loop[(1,)](xd, xd, o, M, N, T, FORM, num_warps=nw)
            g = h.asm["ttgir"]
            rewritten = g.count("tt.reduce") >= 2 and "tt.reshape" in g
            vals = sorted(set(o.cpu().tolist()))
            print(f"L {name} [{M},{N}] w{nw}: rewritten {rewritten}; stored {vals}; every association of "
                  f"the source gives {want}", flush=True)


if __name__ == "__main__":
    run_R()
    run_B()
    run_C()
    run_L()
    print("failures:", FAIL[:40], len(FAIL), flush=True)
