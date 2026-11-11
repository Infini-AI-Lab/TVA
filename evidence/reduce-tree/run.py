"""The trees of Triton 3.8.0's reductions on sm_90, and the reduction rewritten across loop iterations.

Library evidence for library issue L149 (round 32). Sources, pinned in `primitives/`:
`triton-3.8.0-ReduceOpToLLVM.cpp` (the lowering of `tt.reduce`: `treeReduce` within a thread, a butterfly
over the lanes, a layout conversion through shared memory and the butterfly again over the warps),
`triton-3.8.0-TargetInfoBase.h` (`getReductionTreeArity`: 2 unless a target overrides it; NVIDIA does not),
`triton-3.8.0-ReduceScanCommon.h`, `triton-3.8.0-ScanOpToLLVM.cpp` (scans),
`triton-3.8.0-OptimizeThreadLocality.cpp` and `triton-3.8.0-nvidia-compiler.py` (the pass that rewrites a
reduction inside a loop).

  T  the shape of the tree: a tuple `tl.reduce` of (depth, count) leaves (0, 1) with the combine
     (d1, c1), (d2, c2) -> (max(d1, d2) + 1, c1 + c2), whose result is the tree's depth and leaf count, in
     int32, int16 and int64 (the layouts of 32-, 16- and 64-bit loads)
  F  `tl.sum` itself, on floats: every leaf 1 + 2^-p (p the format's fraction bits) and, separately, every
     leaf 1 - 2^-(p+1). A partial sum of m such leaves, m(1 + 2^-p) or m(1 - 2^-(p+1)), is a float exactly
     when m is a power of two, so the sum of n = 2^k leaves is exact when every node of the tree combines two
     subtrees of equal size (a perfect tree, depth k) and is rounded at a node of any other tree. Formats
     float32, float16, bfloat16, float64; also `tl.sum(x * y)` with y = 1 (a sum of products, fusion on)
  Layouts: 1-D loads of n = 1 ... 16384 with num_warps 1, 2, 4, 8, 16, contiguous (vectorised) and strided
     (scalar) loads; 2-D [M, N] loads along axis 0 and 1, row- and column-major, keep_dims both; the result
     of a tl.dot (mma.sync and wgmma layouts); tl.trans of a loaded tile; a 3-D tile reduced along axis 2 and
     then along axis 1 (a reduction of a reduction). Each reduction's operand layout is read from the TTGIR.
  L  the loop rewrite: acc = acc + tl.sum(x, axis=1) in a loop over 2-D tiles x loaded directly, from a
     constant accumulator. Rows chosen so that every association of the source (each trip's sum along any
     tree, then added to acc) gives 1: trip 0 holds 2^24 at column 0 and 1 at the last column, trip 1 holds
     -2^24 at column 0 and 1 at the last column. The TTGIR of the loop and the stored result; also the same
     loop on a 1-D tile, and with a loaded (not constant) initial accumulator, which the pass leaves alone.

Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/reduce-tree/run.py
"""
import math, os, re, sys, tempfile

os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="reduce-tree-cache-"))

import torch
import triton
import triton.language as tl

DEV = "cuda"
OUT = []


def log(s):
    OUT.append(s)
    print(s, flush=True)


@triton.jit
def comb(d1, c1, d2, c2):
    return tl.maximum(d1, d2) + 1, c1 + c2


# ---------------------------------------------------------------- 1-D
@triton.jit
def k1(d_ptr, c_ptr, x_ptr, y_ptr, o_ptr, s_ptr, stride, N: tl.constexpr, STRIDED: tl.constexpr,
       PROD: tl.constexpr):
    offs = tl.arange(0, N)
    if STRIDED:
        offs = offs * stride
    d = tl.load(d_ptr + offs)
    c = tl.load(c_ptr + offs)
    rd, rc = tl.reduce((d, c), 0, comb)
    tl.store(o_ptr, rd)
    tl.store(o_ptr + 1, rc)
    x = tl.load(x_ptr + offs)
    if PROD:
        y = tl.load(y_ptr + offs)
        tl.store(s_ptr, tl.sum(x * y))
    else:
        tl.store(s_ptr, tl.sum(x))


# ---------------------------------------------------------------- 2-D
@triton.jit
def k2(d_ptr, c_ptr, x_ptr, o_ptr, s_ptr, sm, sn, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr,
       KEEP: tl.constexpr, TRANS: tl.constexpr):
    offs = tl.arange(0, M)[:, None] * sm + tl.arange(0, N)[None, :] * sn
    d = tl.load(d_ptr + offs)
    c = tl.load(c_ptr + offs)
    x = tl.load(x_ptr + offs)
    if TRANS:
        d = tl.trans(d)
        c = tl.trans(c)
        x = tl.trans(x)
    rd, rc = tl.reduce((d, c), AXIS, comb, keep_dims=KEEP)
    rs = tl.sum(x, axis=AXIS, keep_dims=KEEP)
    if TRANS:
        if AXIS == 0:
            L: tl.constexpr = M
        else:
            L: tl.constexpr = N
    else:
        if AXIS == 0:
            L: tl.constexpr = N
        else:
            L: tl.constexpr = M
    o = tl.arange(0, L)
    if KEEP:
        rd = tl.reshape(rd, (L,))
        rc = tl.reshape(rc, (L,))
        rs = tl.reshape(rs, (L,))
    tl.store(o_ptr + 2 * o, rd)
    tl.store(o_ptr + 2 * o + 1, rc)
    tl.store(s_ptr + o, rs)


# ---------------------------------------------------------------- tl.dot results
@triton.jit
def kdot(a8_ptr, b8_ptr, a16_ptr, b16_ptr, d_ptr, c_ptr, x_ptr, o_ptr, s_ptr, M: tl.constexpr,
         N: tl.constexpr, AXIS: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, 32)
    a8 = tl.load(a8_ptr + rm[:, None] * 32 + rk[None, :])
    b8 = tl.load(b8_ptr + rk[:, None] * N + rn[None, :])
    a16 = tl.load(a16_ptr + rm[:, None] * 32 + rk[None, :])
    b16 = tl.load(b16_ptr + rk[:, None] * N + rn[None, :])
    offs = rm[:, None] * N + rn[None, :]
    d = tl.dot(a8, b8, tl.load(d_ptr + offs))
    c = tl.dot(a8, b8, tl.load(c_ptr + offs))
    x = tl.dot(a16, b16, tl.load(x_ptr + offs))
    rd, rc = tl.reduce((d, c), AXIS, comb)
    rs = tl.sum(x, axis=AXIS)
    if AXIS == 0:
        L: tl.constexpr = N
    else:
        L: tl.constexpr = M
    o = tl.arange(0, L)
    tl.store(o_ptr + 2 * o, rd)
    tl.store(o_ptr + 2 * o + 1, rc)
    tl.store(s_ptr + o, rs)


# ---------------------------------------------------------------- a reduction of a reduction
@triton.jit
def k3(d_ptr, c_ptr, x_ptr, o_ptr, s_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    offs = (tl.arange(0, A)[:, None, None] * (B * C) + tl.arange(0, B)[None, :, None] * C
            + tl.arange(0, C)[None, None, :])
    d = tl.load(d_ptr + offs)
    c = tl.load(c_ptr + offs)
    x = tl.load(x_ptr + offs)
    rd, rc = tl.reduce((d, c), 2, comb)
    rd, rc = tl.reduce((rd, rc), 1, comb)
    rs = tl.sum(tl.sum(x, 2), 1)
    o = tl.arange(0, A)
    tl.store(o_ptr + 2 * o, rd)
    tl.store(o_ptr + 2 * o + 1, rc)
    tl.store(s_ptr + o, rs)


# ---------------------------------------------------------------- the loop rewrite
@triton.jit
def kloop(x_ptr, a0_ptr, o_ptr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr, LOADED0: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    if LOADED0:
        acc = tl.load(a0_ptr + rm)
    else:
        acc = tl.zeros([M], dtype=tl.float32)
    for t in range(T):
        x = tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :])
        acc = acc + tl.sum(x, axis=1)
    tl.store(o_ptr + rm, acc)


@triton.jit
def kloop1(x_ptr, o_ptr, T: tl.constexpr, N: tl.constexpr):
    rn = tl.arange(0, N)
    acc = 0.0
    for t in range(T):
        x = tl.load(x_ptr + t * N + rn)
        acc = acc + tl.sum(x)
    tl.store(o_ptr, acc)


TD = {"int32": torch.int32, "int16": torch.int16, "int64": torch.int64}
FL = {"float32": (torch.float32, 23), "float16": (torch.float16, 10), "bfloat16": (torch.bfloat16, 7),
      "float64": (torch.float64, 52)}

STATS = {"cases": 0, "tree_ok": 0, "tree_bad": 0, "float_ok": 0, "float_bad": 0}
ENCODINGS = set()
BAD = []


def reduce_encodings(ttgir):
    alias = dict(re.findall(r"^(#\w+) = (#ttg\.\w+<.*>)$", ttgir, re.M))
    encs = []
    for m in re.finditer(r'"tt\.reduce"\(([^)]*)\).*?\}\) : \(([^)]*)\)', ttgir, re.S):
        for ty in re.findall(r"tensor<[^>]*?(#\w+)>", m.group(2)):
            e = alias.get(ty, ty)
            e = re.sub(r"#ttg\.(\w+)<", r"\1<", e)
            encs.append(e)
    return encs


def leaf(dt, p, which):
    return 1.0 + 2.0 ** -p if which == 0 else 1.0 - 2.0 ** -(p + 1)


def check_tree(name, rd, rc, n):
    k = int(round(math.log2(n)))
    STATS["cases"] += 1
    ok = bool(((rd == k) & (rc == n)).all())
    if ok:
        STATS["tree_ok"] += 1
    else:
        STATS["tree_bad"] += 1
        BAD.append(f"T {name}: depth {sorted(set(rd.tolist()))} count {sorted(set(rc.tolist()))} n {n}")
    return ok


def check_float(name, rs, n, dt, p, which):
    exact = n * leaf(dt, p, which)
    ok = bool((rs.to(torch.float64) == exact).all())
    if ok:
        STATS["float_ok"] += 1
    else:
        STATS["float_bad"] += 1
        BAD.append(f"F {name}: got {sorted(set(rs.to(torch.float64).tolist()))[:4]} exact {exact}")
    return ok


def part_1d():
    log("== 1-D: n = 1 ... 16384, num_warps 1, 2, 4, 8, 16, contiguous and strided loads")
    for nw in (1, 2, 4, 8, 16):
        for k in range(0, 15):
            n = 2 ** k
            for strided in (False, True):
                stride = 3 if strided else 1
                for (iname, fname, prod) in (("int32", "float32", False), ("int16", "float16", False),
                                             ("int16", "bfloat16", False), ("int64", "float64", False),
                                             ("int32", "float32", True), ("int16", "float16", True)):
                    for which in (0, 1):
                        it = TD[iname]
                        ft, p = FL[fname]
                        d = torch.zeros(n * stride, dtype=it, device=DEV)
                        c = torch.ones(n * stride, dtype=it, device=DEV)
                        x = torch.full((n * stride,), leaf(ft, p, which), dtype=ft, device=DEV)
                        y = torch.ones(n * stride, dtype=ft, device=DEV)
                        o = torch.zeros(2, dtype=it, device=DEV)
                        s = torch.zeros(1, dtype=ft, device=DEV)
                        h = k1[(1,)](d, c, x, y, o, s, stride, N=n, STRIDED=strided, PROD=prod, num_warps=nw)
                        name = f"1d n={n} nw={nw} strided={strided} {iname}/{fname} prod={prod} leaf={which}"
                        check_tree(name, o[0:1].cpu(), o[1:2].cpu(), n)
                        check_float(name, s.cpu(), n, ft, p, which)
                        for e in reduce_encodings(h.asm["ttgir"]):
                            ENCODINGS.add(e)
        log(f"  num_warps={nw}: {STATS}")


def part_2d():
    log("== 2-D: [M, N], M, N in 1 ... 256 (M*N <= 16384), axis 0 and 1, row/column-major, keep_dims, tl.trans")
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for nw in (1, 4, 8):
        for M in sizes:
            for N in sizes:
                if M * N > 16384:
                    continue
                for axis in (0, 1):
                    for colmajor in (False, True):
                        for trans in (False, True):
                            keep = (M + N + axis) % 2 == 0
                            for (iname, fname) in (("int32", "float32"), ("int16", "float16"),
                                                   ("int16", "bfloat16")):
                                it = TD[iname]
                                ft, p = FL[fname]
                                which = (M * N + axis) % 2
                                sm, sn = (1, M) if colmajor else (N, 1)
                                d = torch.zeros(M * N, dtype=it, device=DEV)
                                c = torch.ones(M * N, dtype=it, device=DEV)
                                x = torch.full((M * N,), leaf(ft, p, which), dtype=ft, device=DEV)
                                if trans:
                                    L, n = (M if axis == 0 else N), (N if axis == 0 else M)
                                else:
                                    L, n = (N if axis == 0 else M), (M if axis == 0 else N)
                                o = torch.zeros(2 * L, dtype=it, device=DEV)
                                s = torch.zeros(L, dtype=ft, device=DEV)
                                h = k2[(1,)](d, c, x, o, s, sm, sn, M=M, N=N, AXIS=axis, KEEP=keep, TRANS=trans,
                                             num_warps=nw)
                                name = (f"2d [{M},{N}] axis={axis} colmajor={colmajor} trans={trans} keep={keep} "
                                        f"nw={nw} {iname}/{fname} leaf={which}")
                                check_tree(name, o[0::2].cpu(), o[1::2].cpu(), n)
                                check_float(name, s.cpu(), n, ft, p, which)
                                for e in reduce_encodings(h.asm["ttgir"]):
                                    ENCODINGS.add(e)
        log(f"  num_warps={nw}: {STATS}")


def part_dot():
    log("== tl.dot results: [M, N] = int8 dot and float16 dot (K = 32) plus a loaded accumulator, both axes")
    for nw in (1, 2, 4, 8):
        for M in (16, 32, 64, 128):
            for N in (16, 32, 64, 128):
                for axis in (0, 1):
                    for which in (0, 1):
                        a8 = torch.zeros(M * 32, dtype=torch.int8, device=DEV)
                        b8 = torch.zeros(32 * N, dtype=torch.int8, device=DEV)
                        a16 = torch.zeros(M * 32, dtype=torch.float16, device=DEV)
                        b16 = torch.zeros(32 * N, dtype=torch.float16, device=DEV)
                        d = torch.zeros(M * N, dtype=torch.int32, device=DEV)
                        c = torch.ones(M * N, dtype=torch.int32, device=DEV)
                        x = torch.full((M * N,), leaf(None, 23, which), dtype=torch.float32, device=DEV)
                        L, n = (N, M) if axis == 0 else (M, N)
                        o = torch.zeros(2 * L, dtype=torch.int32, device=DEV)
                        s = torch.zeros(L, dtype=torch.float32, device=DEV)
                        h = kdot[(1,)](a8, b8, a16, b16, d, c, x, o, s, M=M, N=N, AXIS=axis, num_warps=nw)
                        name = f"dot [{M},{N}] axis={axis} nw={nw} leaf={which}"
                        check_tree(name, o[0::2].cpu(), o[1::2].cpu(), n)
                        check_float(name, s.cpu(), n, torch.float32, 23, which)
                        encs = reduce_encodings(h.asm["ttgir"])
                        for e in encs:
                            ENCODINGS.add(e)
                        if which == 0 and axis == 0 and N == 64:
                            kinds = sorted(set(re.sub(r"<.*", "", e) for e in encs))
                            ptx = h.asm["ptx"]
                            ins = "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "?")
                            log(f"  [{M},{N}] nw={nw}: {ins}; reduce operand layouts {kinds}")
    log(f"  {STATS}")


def part_3d():
    log("== a reduction of a reduction: [A, B, C] along axis 2, then axis 1")
    for nw in (1, 4, 8):
        for A in (1, 4, 16):
            for B in (1, 2, 8, 32):
                for C in (1, 4, 16, 64):
                    for which in (0, 1):
                        n = B * C
                        d = torch.zeros(A * n, dtype=torch.int32, device=DEV)
                        c = torch.ones(A * n, dtype=torch.int32, device=DEV)
                        x = torch.full((A * n,), leaf(None, 23, which), dtype=torch.float32, device=DEV)
                        o = torch.zeros(2 * A, dtype=torch.int32, device=DEV)
                        s = torch.zeros(A, dtype=torch.float32, device=DEV)
                        h = k3[(1,)](d, c, x, o, s, A=A, B=B, C=C, num_warps=nw)
                        name = f"3d [{A},{B},{C}] nw={nw} leaf={which}"
                        check_tree(name, o[0::2].cpu(), o[1::2].cpu(), n)
                        check_float(name, s.cpu(), n, torch.float32, 23, which)
                        for e in reduce_encodings(h.asm["ttgir"]):
                            ENCODINGS.add(e)
    log(f"  {STATS}")


def part_loop():
    log("== L: acc = acc + tl.sum(x, axis=1) in a loop (T = 2 trips), the source's value is 1 in every association")
    for (M, N, nw) in ((16, 64, 4), (32, 128, 4), (64, 256, 8), (4, 1024, 4), (128, 32, 4)):
        x = torch.zeros(2, M, N, dtype=torch.float32, device=DEV)
        x[0, :, 0] = 2.0 ** 24
        x[0, :, N - 1] = 1.0
        x[1, :, 0] = -(2.0 ** 24)
        x[1, :, N - 1] = 1.0
        for loaded0 in (False, True):
            a0 = torch.zeros(M, dtype=torch.float32, device=DEV)
            o = torch.zeros(M, dtype=torch.float32, device=DEV)
            h = kloop[(1,)](x, a0, o, T=2, M=M, N=N, LOADED0=loaded0, num_warps=nw)
            ttgir = h.asm["ttgir"]
            nred = len(re.findall(r'"tt\.reduce"', ttgir))
            rank3 = bool(re.search(r"tt\.reshape .*tensor<\d+x\d+x\d+x", ttgir))
            for_line = [l.strip() for l in ttgir.splitlines() if "scf.for" in l][:1]
            vals = sorted(set(o.cpu().tolist()))
            log(f"  [M={M}, N={N}] nw={nw} initial acc {'loaded' if loaded0 else 'constant 0'}: stored {vals}; "
                f"TTGIR: {nred} tt.reduce, 3-D reshape {rank3}; loop {for_line[0][:160] if for_line else '-'}")
    for (N, nw) in ((256, 4), (1024, 4)):
        x = torch.zeros(2, N, dtype=torch.float32, device=DEV)
        x[0, 0] = 2.0 ** 24
        x[0, N - 1] = 1.0
        x[1, 0] = -(2.0 ** 24)
        x[1, N - 1] = 1.0
        o = torch.zeros(1, dtype=torch.float32, device=DEV)
        h = kloop1[(1,)](x, o, T=2, N=N, num_warps=nw)
        ttgir = h.asm["ttgir"]
        rank2 = bool(re.search(r"tt\.reshape .*tensor<\d+x\d+x", ttgir))
        log(f"  1-D [N={N}] nw={nw}: stored {o.item()}; TTGIR reshape {rank2}")


def main():
    log(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    part_loop()
    part_dot()
    part_3d()
    part_1d()
    part_2d()
    log("== Summary")
    log(f"  {STATS}")
    log(f"  distinct reduce operand layouts seen: {len(ENCODINGS)}")
    kinds = {}
    for e in ENCODINGS:
        kk = re.sub(r"<.*", "", e)
        kinds[kk] = kinds.get(kk, 0) + 1
    log(f"  by kind: {sorted(kinds.items())}")
    for e in sorted(ENCODINGS):
        log(f"    {e}")
    log(f"  failures: {len(BAD)}")
    for b in BAD[:200]:
        log(f"    {b}")


if __name__ == "__main__":
    main()
