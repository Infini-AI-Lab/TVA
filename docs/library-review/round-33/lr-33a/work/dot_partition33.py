"""lr-33a, library review round 33, L150: which k each instruction of a 16-bit tl.dot adds on sm_90, for operand
origins the author's Part P did not probe, and whether each instruction's k lie in one aligned block of 64, blocks in
increasing order (the premise of DotF16Chunk / DotBF16Chunk as restated in round 33).

Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>

Probe (vectorised over columns): b[k, n] = 2^-12 except b[big_n, n] = 2^15, a = 1 (0 at the k already found). A
column's result is exactly 2^15 iff no earlier instruction held 16 nonzero products 2^-12 (their sum 2^-8 is one ulp
of 2^15 and survives only from an earlier instruction); so the k with result 2^15 are the first instruction's, which
are zeroed before the next round. For a b converted from float8e5 the roles swap (vectorised over rows).
"""
import re
import sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEV = "cuda"
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


# SA: 0 fmt, 1 e5, 2 e4, 3 i8, 4 u8, 5 bool ptr, 6 join(i8), 7 join+permute(i8), 8 nested join(i8),
#     9 trans of an i8 [K, M] load, 10 masked i8 load, 11 join(e5) on bf16/f16, 12 f16 load .to(f32) join .to(fmt)
# SB: 0 fmt, 1 e5, 2 trans of a fmt [N, K] load, 3 i8 join on b
@triton.jit
def kp(a_ptr, b_ptr, o_ptr, Kv, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, SA: tl.constexpr,
       SB: tl.constexpr, BF: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if SA == 6:
        rk2 = tl.arange(0, K // 2)
        lo = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :]).to(tl.float16)
        hi = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :] + 1).to(tl.float16)
        a = tl.join(lo, hi).reshape(M, K)
    elif SA == 11:
        rk2 = tl.arange(0, K // 2)
        lo = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :]).to(tl.float16)
        hi = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :] + 1).to(tl.float16)
        a = tl.join(lo, hi).reshape(M, K)
    elif SA == 12:
        rk2 = tl.arange(0, K // 2)
        lo = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :]).to(tl.float32)
        hi = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :] + 1).to(tl.float32)
        a = tl.join(lo, hi).reshape(M, K)
    elif SA == 7:
        rk2 = tl.arange(0, K // 2)
        lo = tl.load(a_ptr + rm[:, None] * K + rk2[None, :]).to(tl.float16)
        hi = tl.load(a_ptr + rm[:, None] * K + K // 2 + rk2[None, :]).to(tl.float16)
        a = tl.permute(tl.join(lo, hi), (0, 2, 1)).reshape(M, K)
    elif SA == 8:
        rk4 = tl.arange(0, K // 4)
        x0 = tl.load(a_ptr + rm[:, None] * K + 4 * rk4[None, :]).to(tl.float16)
        x1 = tl.load(a_ptr + rm[:, None] * K + 4 * rk4[None, :] + 1).to(tl.float16)
        x2 = tl.load(a_ptr + rm[:, None] * K + 4 * rk4[None, :] + 2).to(tl.float16)
        x3 = tl.load(a_ptr + rm[:, None] * K + 4 * rk4[None, :] + 3).to(tl.float16)
        a = tl.join(tl.join(x0, x1), tl.join(x2, x3)).reshape(M, K)
    elif SA == 9:
        at = tl.load(a_ptr + rk[:, None] * M + rm[None, :])
        a = tl.trans(at)
    elif SA == 10:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rk[None, :] < Kv, other=0)
    else:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    if BF:
        a = a.to(tl.bfloat16)
    else:
        a = a.to(tl.float16)
    if SB == 2:
        b = tl.trans(tl.load(b_ptr + rn[:, None] * K + rk[None, :]))
    elif SB == 3:
        rk2 = tl.arange(0, K // 2)
        lo = tl.load(b_ptr + (2 * rk2[:, None]) * N + rn[None, :]).to(tl.float16)
        hi = tl.load(b_ptr + (2 * rk2[:, None] + 1) * N + rn[None, :]).to(tl.float16)
        b = tl.permute(tl.join(lo, hi), (0, 2, 1)).reshape(K, N)
    else:
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if BF:
        b = b.to(tl.bfloat16)
    else:
        b = b.to(tl.float16)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))


@triton.jit
def kdesc(a_desc, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BF: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = a_desc.load([0, 0])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if BF:
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)
    else:
        a = a.to(tl.float16)
        b = b.to(tl.float16)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b))


@triton.jit
def kloop(a_ptr, b_ptr, o_ptr, KT, M: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, KP: tl.constexpr,
          SA: tl.constexpr, BF: tl.constexpr):
    """acc fed back over a K loop of BK, the last block masked (KT need not be a multiple of BK); a converted."""
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BK)
    acc = tl.zeros((M, N), dtype=tl.float32)
    for k0 in range(0, KT, BK):
        km = (k0 + rk) < KT
        a = tl.load(a_ptr + rm[:, None] * KP + k0 + rk[None, :], mask=km[None, :], other=0.0)
        b = tl.load(b_ptr + (k0 + rk[:, None]) * N + rn[None, :], mask=km[:, None], other=0)
        if BF:
            acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        else:
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)


TSRC = {0: None, 1: torch.float8_e5m2, 2: torch.float8_e4m3fn, 3: torch.int8, 4: torch.uint8, 5: torch.bool,
        6: torch.int8, 7: torch.int8, 8: torch.int8, 9: torch.int8, 10: torch.int8, 11: torch.float8_e5m2,
        12: None}
SMALL, BIG = 2.0 ** -12, 2.0 ** 15


def info(h):
    kw = sorted(set(int(x) for x in re.findall(r"kWidth\s*=\s*(\d+)", h.asm["ttgir"])))
    ptx = h.asm["ptx"]
    path = "wgmma" if "wgmma.mma_async" in ptx else ("mma.sync" if "mma.sync" in ptx else "other")
    return path, kw


def to_mem_a(A, sa, fdt):
    """A: logical [M, K] float32 (values 0 / 1). Returns the memory tensor the kernel loads."""
    M, K = A.shape
    if sa == 8:
        mem = torch.empty_like(A)
        for j in range(4):
            mem[:, j::4] = A[:, [4 * i + 2 * (j % 2) + j // 2 for i in range(K // 4)]]
        A = mem
    if sa == 9:
        A = A.T
    t = TSRC[sa]
    if sa in (0, 12):
        t = fdt
    return A.contiguous().to(t)


def probe(sa, sb, M, N, K, nw, bf, mode="ptr", BK=None, KT=None):
    fdt = torch.bfloat16 if bf else torch.float16
    Kp = KT if KT else K
    found, left, parts, meta = [], list(range(Kp)), [], {}
    while left:
        first = []
        swap = sb in (1, 3)
        for g0 in range(0, len(left), M if swap else N):
            grp = left[g0:g0 + (M if swap else N)]
            A = torch.ones(M, Kp, device=DEV)
            A[:, found] = 0
            if swap:  # b converted (e5, or i8 joined) holds 1s; a carries the magnitudes, one big per row
                Bv = torch.ones(Kp, N, device=DEV)
                Bv[found, :] = 0
                Af = torch.full((M, Kp), SMALL, device=DEV)
                for i, k in enumerate(grp):
                    Af[i, k] = BIG
                Af[:, found] = 0
                o = torch.empty(M, N, device=DEV)
                bt = torch.float8_e5m2 if sb == 1 else torch.int8
                h = kp[(1,)](Af.to(fdt), Bv.to(bt), o, Kp, M, N, K, 0, sb, int(bf), num_warps=nw)
                res = o[:, 0]
                hit = [k for i, k in enumerate(grp) if res[i].item() == BIG]
            else:
                Bv = torch.full((Kp, N), SMALL, device=DEV)
                for n, k in enumerate(grp):
                    Bv[k, n] = BIG
                o = torch.empty(M, N, device=DEV)
                if mode == "desc":
                    at = A.to(TSRC[sa] or fdt).contiguous()
                    d = TensorDescriptor.from_tensor(at, [M, K])
                    h = kdesc[(1,)](d, Bv.to(fdt), o, M, N, K, int(bf), num_warps=nw)
                elif mode == "loop":
                    h = kloop[(1,)](A.to(TSRC[sa]).contiguous(), Bv.to(fdt), o, KT, M, N, BK, Kp, sa, int(bf),
                                    num_warps=nw, num_stages=3)
                else:
                    bm = Bv.T.contiguous() if sb == 2 else Bv
                    bm = bm.to(fdt)
                    h = kp[(1,)](to_mem_a(A, sa, fdt), bm, o, Kp, M, N, K, sa, sb, int(bf), num_warps=nw)
                col = o[0]
                assert bool((o == o[0:1]).all())
                hit = [k for n, k in enumerate(grp) if col[n].item() == BIG]
            meta["path"], meta["kw"] = info(h)
            first += hit
        if not first or len(first) == len(left):
            parts.append(left)
            break
        parts.append(sorted(first))
        found += first
        left = [k for k in left if k not in first]
    return parts, meta.get("path"), meta.get("kw")


def check(parts):
    sizes = sorted(set(len(p) for p in parts))
    blocks = [min(p) // 64 for p in parts]
    in64 = all(min(p) // 64 == max(p) // 64 for p in parts)
    incr = all(b1 <= b2 for b1, b2 in zip(blocks, blocks[1:]))
    span = max(max(p) - min(p) + 1 for p in parts)
    return in64 and incr, sizes, span


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    shapes = [(16, 16, 1), (16, 64, 2), (32, 64, 4), (16, 128, 4), (64, 64, 2), (64, 64, 4), (128, 64, 8)]
    names = {0: "fmt", 1: "e5", 2: "e4", 3: "i8", 4: "u8", 5: "bool", 6: "join-i8", 7: "joinperm-i8",
             8: "join2-i8", 9: "trans-i8", 10: "mask-i8", 11: "join-e5", 12: "f32join"}
    bad = 0
    total = 0
    cfgs = []
    for bf in (0, 1):
        for sa in names:
            if sa == 2 and bf:
                continue
            for (M, N, nw) in shapes:
                for K in (32, 128):
                    cfgs.append(("ptr", sa, 0, M, N, K, nw, bf))
        for sb in (1, 2, 3):
            for (M, N, nw) in shapes:
                for K in (64, 128):
                    cfgs.append(("ptr", 0, sb, M, N, K, nw, bf))
        for sa in (3, 5, 0):
            for (M, N, nw) in [(16, 64, 1), (16, 64, 2), (64, 64, 4)]:
                for K in (64, 128):
                    cfgs.append(("desc", sa, 0, M, N, K, nw, bf))
        for sa in (3, 1):
            for (M, N, nw) in [(16, 32, 1), (32, 64, 2), (64, 64, 4)]:
                for BK, KT in ((64, 96), (32, 96), (64, 192), (128, 160)):
                    cfgs.append(("loop", sa, 0, M, N, BK, nw, bf, KT))
    for c in cfgs:
        mode, sa, sb, M, N, K, nw, bf = c[:8]
        KT = c[8] if len(c) > 8 else None
        tag = f"{mode} {'bf16' if bf else 'f16'} a={names[sa]} b={['fmt', 'e5', 'trans', 'join-i8'][sb]} " \
              f"{M}x{N} nw{nw} K{K}" + (f" KT{KT}" if KT else "")
        try:
            parts, path, kw = probe(sa, sb, M, N, K, nw, bf, mode=mode, BK=K if mode == "loop" else None, KT=KT)
        except Exception as e:  # noqa: BLE001
            out(f"{tag}: not run ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
            continue
        ok, sizes, span = check(parts)
        total += 1
        bad += not ok
        out(f"{tag}: {path} kW{kw} instrs {len(parts)} sizes {sizes} widest-span {span} in-aligned-64-and-ordered {ok}"
            + ("" if ok else f" PARTS {parts}"))
    out(f"SUMMARY: {total} configurations probed, {bad} with an instruction outside one aligned block of 64 or "
        f"out of order")
    with open(__file__.rsplit("/", 1)[0] + "/dot_partition33.out", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
