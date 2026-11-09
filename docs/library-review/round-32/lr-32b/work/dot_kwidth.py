"""lr-32b, round 32, L150: are a 16-bit tl.dot's chunks 16 consecutive k when an operand is converted
in registers from a narrower dtype (fp8 or int8 to float16)?

Triton picks the dot operand's kWidth from the operand's original bit width; with kWidth > the
instruction's per-thread k-run, one instruction's 16 k may not be consecutive. Probe: row i of a has
one product 1 at k = BIG and the rest 2^-27 (2^-14 * 2^-13), K = 32. The per-instruction model
(evidence/num-dot-sub16-round25) gives 1 if the chunk holding BIG comes first and holds 15 of the tiny
products, 1 + 2^-23 if a chunk of 16 tiny products comes first. Also: random inputs compared with the
model chunked as 16 consecutive k, and as the interleaved sets.

Run from TVA's root: CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I docs/library-review/round-32/lr-32b/work/dot_kwidth.py
"""
import os, re, tempfile
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="lr32b-"))
from fractions import Fraction as Fr
import math
import torch, triton, triton.language as tl


@triton.jit
def kdot(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         SRC: tl.constexpr, ACC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    if SRC == 1:
        a = a.to(tl.float16)          # fp8e5 or int8 -> float16 in registers
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if ACC:
        acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
        r = tl.dot(a, b, acc)
    else:
        r = tl.dot(a, b)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], r)


def efield(x, emin):
    return max(math.floor(math.log2(abs(x))), emin)


def trunc_f32(S):
    if S == 0:
        return Fr(0)
    sgn = 1 if S > 0 else -1
    a = abs(S)
    e = max(math.floor(math.log2(a)), -126)
    # fix floor for Fractions
    while Fr(2) ** e > a:
        e -= 1
    while Fr(2) ** (e + 1) <= a:
        e += 1
    e = max(e, -126)
    q = Fr(2) ** (e - 23)
    return sgn * (a // q) * q


def instr(s, prods, emin=-14):
    """one instruction: s (Fraction) and list of (a, b) Fractions."""
    terms, fields = [], []
    if s != 0:
        terms.append(s)
        fields.append(max(math.floor(math.log2(abs(s))), -126))
    for a, b in prods:
        p = a * b
        if p != 0:
            terms.append(p)
            fields.append(efield(a, emin) + efield(b, emin))
    if not terms:
        return Fr(0)
    E = max(fields)
    u = Fr(2) ** (E - 25)
    tot = Fr(0)
    for t in terms:
        q = abs(t) // u
        tot += (q * u) if t > 0 else -(q * u)
    return trunc_f32(tot)


def model(acc, a_row, b_col, chunks):
    s = Fr(acc)
    for ch in chunks:
        s = instr(s, [(Fr(a_row[k]), Fr(b_col[k])) for k in ch])
    return s


def consec(K):
    return [list(range(16 * j, 16 * j + 16)) for j in range(K // 16)]


def inter8(K):
    # kWidth 8 over two instructions: each thread c holds 8 consecutive k (8c..8c+7) of a 32-block,
    # an instruction takes 4 of them
    out = []
    for blk in range(K // 32):
        base = 32 * blk
        out.append([base + 8 * c + i for c in range(4) for i in range(4)])
        out.append([base + 8 * c + 4 + i for c in range(4) for i in range(4)])
    return out


def run(src, M, N, K, nw, acc_on, a_vals, b_vals, c_vals):
    dev = "cuda"
    if src == "fp8e5":
        a = a_vals.to(torch.float8_e5m2).to(dev)
        SRC = 1
    elif src == "int8":
        a = a_vals.to(torch.int8).to(dev)
        SRC = 1
    else:
        a = a_vals.to(torch.float16).to(dev)
        SRC = 0
    b = b_vals.to(torch.float16).to(dev)
    c = c_vals.to(torch.float32).to(dev)
    o = torch.empty(M, N, dtype=torch.float32, device=dev)
    h = kdot[(1,)](a, b, c, o, M, N, K, SRC, acc_on, num_warps=nw)
    ttgir = h.asm["ttgir"]
    ptx = h.asm["ptx"]
    kw = sorted(set(re.findall(r"kWidth\s*=\s*(\d+)", ttgir)))
    path = "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "other")
    return o.cpu(), kw, path


def probe(src, M, N, K, nw, big):
    a = torch.full((M, K), 2.0 ** -14)
    b = torch.full((K, N), 2.0 ** -13)
    if src == "int8":
        a = torch.full((M, K), 1.0)
        b = torch.full((K, N), 2.0 ** -6)
        a[:, big] = 64.0
        b[big, :] = 2.0 ** 15
    else:
        a[:, big] = 1.0
        b[big, :] = 1.0
    c = torch.zeros(M, N)
    o, kw, path = run(src, M, N, K, nw, False, a, b, c)
    got = set(o.flatten().tolist())
    ar = [Fr(x) for x in a[0].tolist()]
    bc = [Fr(x) for x in b[:, 0].tolist()]
    mc = float(model(0, ar, bc, consec(K)))
    mi = float(model(0, ar, bc, inter8(K)))
    print(f"probe src={src} M={M} N={N} K={K} warps={nw} path={path} kWidth={kw} big={big}: "
          f"device={sorted(got)} model(consecutive)={mc!r} model(interleaved8)={mi!r}", flush=True)


def randtest(src, M, N, K, nw, trials=20):
    g = torch.Generator().manual_seed(7)
    bad_c = bad_i = tot = 0
    kw = path = None
    for t in range(trials):
        if src == "int8":
            a = torch.randint(-127, 128, (M, K), generator=g).float()
            e = torch.randint(-30, -10, (K, N), generator=g).float()
            b = (torch.rand(K, N, generator=g) + 1) * torch.pow(2.0, e)
            b = b.to(torch.float16).float()
        else:
            ea = torch.randint(-14, 4, (M, K), generator=g).float()
            a = torch.pow(2.0, ea) * torch.where(torch.rand(M, K, generator=g) < 0.5, -1.0, 1.0)
            a = a.to(torch.float8_e5m2).float()
            eb = torch.randint(-14, 4, (K, N), generator=g).float()
            b = (torch.rand(K, N, generator=g) + 1) * torch.pow(2.0, eb)
            b = b.to(torch.float16).float()
        c = torch.zeros(M, N)
        o, kw, path = run(src, M, N, K, nw, False, a, b, c)
        for i in range(0, M, max(1, M // 8)):
            ar = [Fr(x) for x in a[i].tolist()]
            for n in range(0, N, max(1, N // 8)):
                bc = [Fr(x) for x in b[:, n].tolist()]
                d = Fr(o[i, n].item())
                tot += 1
                if model(0, ar, bc, consec(K)) != d:
                    bad_c += 1
                if model(0, ar, bc, inter8(K)) != d:
                    bad_i += 1
    print(f"random src={src} M={M} N={N} K={K} warps={nw} path={path} kWidth={kw}: {tot} elements, "
          f"differ from model(consecutive) {bad_c}, from model(interleaved8) {bad_i}", flush=True)


if __name__ == "__main__":
    for src in ["fp16", "fp8e5", "int8"]:
        for (M, N, nw) in [(16, 16, 1), (32, 32, 2), (64, 64, 4), (128, 64, 4)]:
            for K in [32, 64]:
                for big in [4, 20]:
                    try:
                        probe(src, M, N, K, nw, big)
                    except Exception as ex:
                        print(f"probe src={src} M={M} N={N} K={K} warps={nw}: {type(ex).__name__}", flush=True)
            try:
                randtest(src, M, N, 64, nw)
            except Exception as ex:
                print(f"random src={src} M={M}: {type(ex).__name__} {ex}", flush=True)
