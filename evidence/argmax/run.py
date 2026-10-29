"""TVA library evidence (library issue L17): `tl.argmax`, `tl.argmin` and `tl.max`/`tl.min` with
`return_indices=True`, against the library's definition.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/argmax/run.py
Output: evidence/argmax/out.txt (README.md says what it shows).

The library's definition (`tl.argCombine`, after Triton's `_argmax_combine`/`_argmin_combine`):
combining `(v1, i1)` and `(v2, i2)`, `tie = tieLeft and v1 == v2 and i1 < i2`,
`win = (v1 > v2 if isMax else v1 < v2) or tie`, and the result is `(v1, i1)` if `win`, else
`(v2, i2)`. The reduction applies it along SOME binary tree whose leaves are all positions, in any
order, leaf `j` being `(x[j], j)`. Comparisons are Triton's: ordered for floats (false with a NaN).

Each row of a 2-D block is reduced along axis 1, or each column along axis 0, by `tl.max(...,
return_indices=True)`, `tl.argmax`, `tl.min(..., return_indices=True)` or `tl.argmin`, with
`tie_break_left` true or false. For each device result the script checks:
(a) the index is in range and the value is `x[index]` (bit for bit);
(b) without NaN and with `tie_break_left`, the index is the first position whose value `==` the
    maximum (minimum);
(c) without NaN and without `tie_break_left`, the value `==` the maximum (minimum) and the index is a
    position where it is attained;
(d) for rows of at most 8 elements, the index is one that some tree gives (all trees and leaf orders
    are enumerated).
Check (a) is counted apart for rows with a NaN. A later part measures, for float16 and bfloat16
along axis 0, how often the value and the index come from different positions when the row holds a
NaN; another tries fp8 operands; another puts one NaN at each position of a row. The last part
(`value_index`) counts, for float16, float32 and bfloat16, on [16, 64] blocks along axis 0 and
[32, 256] blocks along axis 1, with and without NaN, every way the value and the index can disagree,
whether the index is in range, and whether `tl.argmax` (`tl.argmin`) gives the same index as
`tl.max` (`tl.min`) with `return_indices=True`. Values are compared bit for bit, except that any NaN
equals any NaN.
Rows: every row of length 4 over a 7-value float alphabet (NaN, -inf, -1, -0, +0, 1, +inf) or a
5-value integer alphabet (the type's extremes, -1, 0, 1); random rows of length 8 over the same
alphabets; random rows of length 1024 with ties and NaNs.
"""
import itertools
import math
import re
import time

import numpy as np
import torch
import triton
import triton.language as tl

DEV = "cuda"
RB = 64  # rows per program


@triton.jit
def red(x_ptr, v_ptr, i_ptr, R, N: tl.constexpr, RB: tl.constexpr, AXIS: tl.constexpr, MODE: tl.constexpr,
        TIE: tl.constexpr):
    pid = tl.program_id(0)
    if AXIS == 1:
        rows = pid * RB + tl.arange(0, RB)[:, None]
        cols = tl.arange(0, N)[None, :]
        x = tl.load(x_ptr + rows * N + cols, mask=rows < R, other=0)
    else:
        rows = tl.arange(0, N)[:, None]
        cols = pid * RB + tl.arange(0, RB)[None, :]
        x = tl.load(x_ptr + rows * R + cols, mask=cols < R, other=0)
    if MODE == 0:
        v, i = tl.max(x, AXIS, return_indices=True, return_indices_tie_break_left=TIE)
        out = pid * RB + tl.arange(0, RB)
        tl.store(v_ptr + out, v, mask=out < R)
    elif MODE == 1:
        i = tl.argmax(x, AXIS, tie_break_left=TIE)
    elif MODE == 2:
        v, i = tl.min(x, AXIS, return_indices=True, return_indices_tie_break_left=TIE)
        out = pid * RB + tl.arange(0, RB)
        tl.store(v_ptr + out, v, mask=out < R)
    else:
        i = tl.argmin(x, AXIS, tie_break_left=TIE)
    out = pid * RB + tl.arange(0, RB)
    tl.store(i_ptr + out, i, mask=out < R)


@triton.jit
def fp8(x_ptr, v_ptr, i_ptr, N: tl.constexpr, M: tl.constexpr, CODE: tl.constexpr):
    rows = tl.arange(0, N)[:, None]
    cols = tl.arange(0, M)[None, :]
    raw = tl.load(x_ptr + rows * M + cols)
    if CODE == 0:
        x = raw.to(tl.float8e5, bitcast=True)
    else:
        x = raw.to(tl.float8e4nv, bitcast=True)
    v, i = tl.max(x, 0, return_indices=True)
    tl.store(v_ptr + tl.arange(0, M), v.to(tl.float32))
    tl.store(i_ptr + tl.arange(0, M), i)


@triton.jit
def pair(x_ptr, v_ptr, i_ptr, a_ptr, N: tl.constexpr, M: tl.constexpr, AXIS: tl.constexpr, ISMAX: tl.constexpr,
         TIE: tl.constexpr):
    # one [N, M] block per program; the value and index of `tl.max`/`tl.min(..., return_indices=True)`
    # and the index of `tl.argmax`/`tl.argmin`, along AXIS
    pid = tl.program_id(0)
    rows = tl.arange(0, N)[:, None]
    cols = tl.arange(0, M)[None, :]
    x = tl.load(x_ptr + pid * N * M + rows * M + cols)
    if ISMAX:
        v, i = tl.max(x, AXIS, return_indices=True, return_indices_tie_break_left=TIE)
        a = tl.argmax(x, AXIS, tie_break_left=TIE)
    else:
        v, i = tl.min(x, AXIS, return_indices=True, return_indices_tie_break_left=TIE)
        a = tl.argmin(x, AXIS, tie_break_left=TIE)
    if AXIS == 0:
        out = pid * M + tl.arange(0, M)
    else:
        out = pid * N + tl.arange(0, N)
    tl.store(v_ptr + out, v)
    tl.store(i_ptr + out, i)
    tl.store(a_ptr + out, a)


MODES = {0: ("tl.max(x, axis, return_indices=True)", True), 1: ("tl.argmax(x, axis)", True),
         2: ("tl.min(x, axis, return_indices=True)", False), 3: ("tl.argmin(x, axis)", False)}
TORCH = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "int32": torch.int32, "int8": torch.int8,
         "uint8": torch.uint8}
BITS = {"float32": torch.int32, "float16": torch.int16, "bfloat16": torch.int16, "int32": torch.int32, "int8": torch.int8, "uint8": torch.uint8}


def is_float(dtype):
    """A float dtype (`bfloat16` included: an earlier version of this script tested
    `dtype.startswith("float")`, which sent `bfloat16` inputs through `int64`, so NaN became -2^63)."""
    return dtype in ("float32", "float16", "bfloat16")


def win(is_max, tie_left, x, a, b):
    """Whether `(x[a], a)` wins against `(x[b], b)` in the library's combine."""
    va, vb = x[a], x[b]
    tie = tie_left and va == vb and a < b
    return (va > vb if is_max else va < vb) or tie


def tree_results(is_max, tie_left, x):
    """The indices some binary tree over all positions (any shape, any leaf order) can give."""
    n = len(x)
    W = [[win(is_max, tie_left, x, a, b) for b in range(n)] for a in range(n)]
    poss = {}
    for size in range(1, n + 1):
        for S in itertools.combinations(range(n), size):
            m = 0
            for j in S:
                m |= 1 << j
            if size == 1:
                poss[m] = 1 << S[0]
                continue
            res = 0
            sub = (m - 1) & m
            while sub:
                L, Rm = sub, m ^ sub
                pa, pb = poss[L], poss[Rm]
                for a in range(n):
                    if pa >> a & 1:
                        for b in range(n):
                            if pb >> b & 1:
                                res |= 1 << (a if W[a][b] else b)
                sub = (sub - 1) & m
            poss[m] = res
    full = (1 << n) - 1
    return {j for j in range(n) if poss[full] >> j & 1}


def alphabet(dtype):
    if is_float(dtype):
        return [math.nan, -math.inf, -1.0, -0.0, 0.0, 1.0, math.inf]
    lo, hi = {"int32": (-2**31, 2**31 - 1), "int8": (-128, 127), "uint8": (0, 255)}[dtype]
    return sorted({lo, hi, 0, 1} | ({-1} if lo < 0 else {254}))


def run_device(dtype, X, axis, mode, tie):
    R, N = X.shape
    t = torch.tensor(X if is_float(dtype) else X.astype(np.int64), dtype=TORCH[dtype])
    if axis == 0:
        t = t.t().contiguous()
    t = t.to(DEV)
    v = torch.zeros(R, dtype=TORCH[dtype], device=DEV)
    i = torch.zeros(R, dtype=torch.int32, device=DEV)
    k = red[(triton.cdiv(R, RB), )](t, v, i, R, N, RB, axis, mode, tie)
    torch.cuda.synchronize()
    tys = re.findall(r"\}\) : \([^)]*\) -> (\([^)]*\))", k.asm["ttir"])
    return v.cpu(), i.cpu().numpy(), tys[0] if tys else "?"


def check(dtype, X, axis, mode, tie, enumerate_trees):
    is_max = MODES[mode][1]
    v, idx, ty = run_device(dtype, X, axis, mode, tie)
    R, N = X.shape
    xt = torch.tensor(X if is_float(dtype) else X.astype(np.int64), dtype=TORCH[dtype])
    fails = {"a": 0, "a_nan": 0, "b": 0, "c": 0, "d": 0}
    nan_rows = 0
    nan_idx = {}
    for r in range(R):
        row = [float(z) for z in xt[r]] if is_float(dtype) else [int(z) for z in X[r]]
        j = int(idx[r])
        ok_a = 0 <= j < N
        if ok_a and mode in (0, 2):
            ok_a = bool(v[r:r + 1].view(BITS[dtype]).item() == xt[r, j:j + 1].view(BITS[dtype]).item())
        has_nan = any(z != z for z in row)
        fails["a_nan" if has_nan else "a"] += not ok_a
        if has_nan:
            nan_rows += 1
            key = "NaN" if ok_a and row[j] != row[j] else "not NaN"
            nan_idx[key] = nan_idx.get(key, 0) + 1
        else:
            best = max(row) if is_max else min(row)
            first = next(p for p, z in enumerate(row) if z == best)
            if tie:
                fails["b"] += j != first
            else:
                fails["c"] += not (ok_a and row[j] == best)
        if enumerate_trees:
            fails["d"] += j not in tree_results(is_max, tie, row)
    return fails, ty, nan_rows, nan_idx


def rows_exhaustive(dtype, n):
    return np.array(list(itertools.product(alphabet(dtype), repeat=n)), dtype=np.float64)


def rows_random(dtype, n, count, rng):
    A = alphabet(dtype)
    return np.array([[A[k] for k in rng.integers(0, len(A), size=n)] for _ in range(count)], dtype=np.float64)


def rows_large(dtype, n, count, rng, nan):
    if is_float(dtype):
        X = rng.integers(-4, 5, size=(count, n)).astype(np.float64)  # many ties
        X[X == 0] = -0.0
        X[:, ::7] = 0.0
        if nan:
            m = rng.random((count, n)) < 0.01
            X[m] = math.nan
    else:
        lo, hi = {"int32": (-2**31, 2**31 - 1), "int8": (-128, 127), "uint8": (0, 255)}[dtype]
        X = rng.integers(lo, hi + 1, size=(count, n)).astype(np.float64)
        X[:, rng.integers(0, n, size=4)] = hi  # ties at the maximum
    return X


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}")
    rng = np.random.default_rng(20261027)
    total = {"a": 0, "a_nan": 0, "b": 0, "c": 0, "d": 0}
    for dtype in ("float32", "float16", "int32", "int8", "uint8"):
        print(f"\n== {dtype} ==")
        sets = [("n=4, every row over the alphabet", rows_exhaustive(dtype, 4), True),
                ("n=8, 512 random rows over the alphabet", rows_random(dtype, 8, 512, rng), True),
                ("n=1024, 128 random rows with ties" + (", 1% NaN" if is_float(dtype) else ""),
                 rows_large(dtype, 1024, 128, rng, True), False)]
        if is_float(dtype):
            sets.append(("n=1024, 128 random rows with ties, no NaN", rows_large(dtype, 1024, 128, rng, False), False))
        for label, X, enum in sets:
            if not is_float(dtype):
                X = X.astype(np.int64)
            for mode in range(4):
                for tie in (True, False):
                    for axis in (1, 0):
                        fails, ty, nan_rows, nan_idx = check(dtype, X, axis, mode, tie, enum)
                        for k in total:
                            total[k] += fails[k]
                        nan_note = (f"; {nan_rows} rows with NaN: result value NaN in {nan_idx.get('NaN', 0)}, "
                                    f"not NaN in {nan_idx.get('not NaN', 0)}") if nan_rows else ""
                        print(f"{label:45} | {MODES[mode][0]:37} tie_break_left={tie!s:5} axis={axis} | "
                              f"TTIR {ty:12} | failed (a) {fails['a']} (rows with NaN: {fails['a_nan']}), (b) {fails['b']}, (c) {fails['c']}"
                              + (f", (d) {fails['d']}" if enum else ", (d) not enumerated") + nan_note)
    print(f"\nTotal failures: (a) on rows without NaN {total['a']}, (a) on rows with NaN {total['a_nan']}, "
          f"(b) {total['b']}, (c) {total['c']}, (d) {total['d']}")

    # The value and the index of a 16-bit float reduction along axis 0, on rows with NaN. Bits are
    # compared, except that any NaN equals any NaN (bfloat16 is reduced in float32 and converted back).
    for dtype in ("float16", "bfloat16"):
        print(f"\n== {dtype}, axis 0, tie_break_left=True: is the value x[index]? 20 x 256 rows of length 16, "
              f"5% NaN ==")
        for mode in (0, 2):
            cnt = {"rows": 0, "nan rows": 0, "same position": 0, "value in row, index in range": 0, "other": 0,
                   "no-NaN rows not same": 0}
            for rep in range(20):
                X = rng.integers(-4, 5, size=(256, 16)).astype(np.float64)
                X[rng.random((256, 16)) < 0.05] = math.nan
                v, idx, _ = run_device(dtype, X, 0, mode, True)
                xt = torch.tensor(X, dtype=TORCH[dtype])
                for r in range(256):
                    cnt["rows"] += 1
                    j = int(idx[r])
                    bits = v[r:r + 1].view(torch.int16).item()
                    vnan = bool(torch.isnan(v[r].float()))
                    row_bits = [xt[r, q:q + 1].view(torch.int16).item() for q in range(16)]
                    row_nan = [math.isnan(z) for z in X[r]]

                    def eq(q):
                        return (vnan and row_nan[q]) or (not vnan and not row_nan[q] and bits == row_bits[q])
                    same = 0 <= j < 16 and eq(j)
                    if not any(row_nan):
                        cnt["no-NaN rows not same"] += not same
                        continue
                    cnt["nan rows"] += 1
                    if same:
                        cnt["same position"] += 1
                    elif 0 <= j < 16 and any(eq(q) for q in range(16)):
                        cnt["value in row, index in range"] += 1
                    else:
                        cnt["other"] += 1
            print(f"{MODES[mode][0]}: {cnt}")

    # fp8 operands with indices.
    print("\n== float8e5 / float8e4nv operands, tl.max(x, 0, return_indices=True) ==")
    for name, tdt, code in (("float8e5", torch.float8_e5m2, 0), ("float8e4nv", torch.float8_e4m3fn, 1)):
        x = torch.zeros(64, 64, dtype=tdt).view(torch.int8).to(DEV)
        v = torch.zeros(64, dtype=torch.float32, device=DEV)
        i = torch.zeros(64, dtype=torch.int32, device=DEV)
        try:
            fp8[(1, )](x, v, i, 64, 64, code)
            torch.cuda.synchronize()
            res = "compiles"
        except Exception as e:  # noqa: BLE001
            res = f"rejected on the host ({type(e).__name__})"
        print(f"{name}: {res}")

    # How a NaN's position decides the result: one row, NaN at each position, the rest distinct.
    print("\n== float32, n = 8: values 1..8 with one NaN at position p (tl.argmax, tie_break_left=True, axis 1) ==")
    for p in range(8):
        row = [float(k + 1) for k in range(8)]
        row[p] = math.nan
        X = np.array([row], dtype=np.float64)
        _, idx, _ = run_device("float32", X, 1, 1, True)
        print(f"NaN at {p}: device index {int(idx[0])}; indices some tree gives: "
              f"{sorted(tree_results(True, True, row))}")
    value_index(rng)


def value_index(rng):
    """Is the value of `tl.max`/`tl.min(..., return_indices=True)` the element at its index?

    Blocks [N, M] = [16, 64] reduced along axis 0 (64 slices of 16) and [32, 256] along axis 1
    (32 slices of 256), 256 programs per launch, 8 launches per configuration, values in -4..4
    (many ties), with NaNs at a rate that puts a NaN in about half of the slices, and again with
    no NaN. Comparison: bit for bit, except that any NaN equals any NaN (bfloat16 is reduced in
    float32 and converted back, which may change a NaN's payload). Also: is the index in range,
    and is `tl.argmax`'s (`tl.argmin`'s) index the same as the index `tl.max` (`tl.min`) returns?
    """
    print("\n== value and index of the same result, with and without NaN ==")
    P, reps = 256, 8
    for dtype in ("float16", "float32", "bfloat16"):
        tdt, bdt = TORCH[dtype], BITS[dtype]
        for N, M, axis in ((16, 64, 0), (32, 256, 1)):
            n = N if axis == 0 else M
            slices = M if axis == 0 else N
            rate = 1 - 0.5 ** (1 / n)
            for nan in (True, False):
                for is_max in (True, False):
                    for tie in (True, False):
                        c = {"slices": 0, "with NaN": 0, "index out of range": 0, "argmax index differs": 0,
                             "same (no NaN)": 0, "differ (no NaN)": 0, "same (NaN)": 0,
                             "value NaN, x[index] not NaN": 0, "value not NaN, x[index] NaN": 0,
                             "both numbers, differ (NaN)": 0, "value not in slice": 0}
                        for _ in range(reps):
                            X = rng.integers(-4, 5, size=(P, N, M)).astype(np.float64)
                            if nan:
                                X[rng.random((P, N, M)) < rate] = math.nan
                            xt = torch.tensor(X, dtype=tdt)
                            v = torch.zeros(P * slices, dtype=tdt, device=DEV)
                            i = torch.zeros(P * slices, dtype=torch.int32, device=DEV)
                            a = torch.zeros(P * slices, dtype=torch.int32, device=DEV)
                            pair[(P, )](xt.to(DEV), v, i, a, N, M, axis, is_max, tie)
                            torch.cuda.synchronize()
                            vb = v.cpu().view(bdt).numpy()
                            vn = torch.isnan(v.cpu().float()).numpy()
                            ii, aa = i.cpu().numpy(), a.cpu().numpy()
                            xs = (xt if axis == 1 else xt.transpose(1, 2)).reshape(P * slices, n)
                            xb = xs.contiguous().view(bdt).numpy()
                            xn = torch.isnan(xs.float()).numpy()
                            ar = np.arange(P * slices)
                            c["slices"] += P * slices
                            c["argmax index differs"] += int((aa != ii).sum())
                            inr = (ii >= 0) & (ii < n)
                            c["index out of range"] += int((~inr).sum())
                            jc = np.clip(ii, 0, n - 1)
                            xjb, xjn = xb[ar, jc], xn[ar, jc]
                            same = (vn & xjn) | (~vn & ~xjn & (vb == xjb))
                            has_nan = xn.any(axis=1)
                            ok = inr & ~has_nan
                            c["same (no NaN)"] += int((ok & same).sum())
                            c["differ (no NaN)"] += int((ok & ~same).sum())
                            w = inr & has_nan
                            c["with NaN"] += int(w.sum())
                            c["same (NaN)"] += int((w & same).sum())
                            c["value NaN, x[index] not NaN"] += int((w & ~same & vn).sum())
                            c["value not NaN, x[index] NaN"] += int((w & ~same & ~vn & xjn).sum())
                            c["both numbers, differ (NaN)"] += int((w & ~same & ~vn & ~xjn).sum())
                            c["value not in slice"] += int((w & ~vn & ~(xb == vb[:, None]).any(axis=1)).sum())
                        name = "tl.max" if is_max else "tl.min"
                        print(f"{dtype:8} [{N}, {M}] axis {axis} {'NaN' if nan else 'no NaN':6} {name} "
                              f"tie_break_left={tie!s:5}: " + ", ".join(f"{k} {v}" for k, v in c.items()), flush=True)


if __name__ == "__main__":
    main()
