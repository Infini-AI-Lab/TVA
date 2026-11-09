"""GPU check of the prepared claim `Claims.RecomputeWU` (Claim.lean).

Runs the kernel's own host (`source/wy_fast.py`, imported by path) on the launches SETUP.md names
and on launches at the edges of `Inst.ok`, and checks `Post` element by element, computed in
float64 from the host's inputs exactly as `Post` states it. For token `t` of a sequence starting at
flat token `s0`, value head `h`, the columns `j < 64` of its row of `A` with
`t // 64 * 64 + j < len` (the chunk's tokens `f_j = s0 + t // 64 * 64 + j` inside the sequence):

* `u`: `vb_j = bf16(fl32(v[f_j, h, c] * beta[f_j, h]))` for a `float32` `beta` (the IEEE `float32`
  product, then the rounding to nearest even; torch computes both exactly so), the IEEE `bfloat16`
  product `v[f_j, h, c] * beta[f_j, h]` for a `bfloat16` one (its exact value fits `float32`, then one
  rounding), `uRef = sum_j A_j * vb_j`,
  `uMag = sum_j liftMag(A_j) * liftMag(vb_j)` (`liftMag q = 0` at `0`, else `max(|q|, 2^-126)`),
  `tolU = 2^-8 |uRef| + (1 + 2^-8) (66 * 2^-22 * uMag + 66 * 2^-126) + 2^-134`;
* `w`: `x_j = k[f_j, h // (HV // H), c] * beta[f_j, h]`, `e_j = 2^g[f_j, h]`,
  `wRef = sum_j A_j x_j e_j`, `wMag = sum_j liftMag(A_j) |x_j| e_j`,
  `wFloor = sum_j liftMag(A_j) (|x_j| + 1)`,
  `tolW = 2^-8 |wRef| + cW wMag + 2^-124 wFloor + 2^-119`, `cW = 2^-8 + 2^-14` (`float32` `beta`) or
  `2^-7 + 2^-13` (`bfloat16` `beta`);

and every output is finite. The frame: the inputs (and the host's `chunk_indices`, passed in) are
compared with copies after each launch, and one direct launch writes `w` and `u` into views placed
between guard regions, which must keep their pattern.

Negative tests: outputs changed in ways the claim excludes (no gate factor, `A` transposed within
the chunk, the wrong key head, chunks not restarted at sequence starts, a 2% error in one element)
must be rejected. The launch with `B * HV = 65536` must fail (CUDA's grid axis 1), outside the claim.

Each edge runs with a `float32` and with a `bfloat16` `beta`, except one built for a `bfloat16`
`beta` (the roundings of lines 129 and 132 all upward, adding up). Prints, per case, the largest error / tolerance of each output. Usage:
CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "wy_fast.py")
spec = importlib.util.spec_from_file_location("wy_fast_src", SRC)
wy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wy)

DEV = "cuda"
BT = 64

# SETUP.md's launches (the `setup_*` instances of Claim.lean, except `edge_bhv65535`, outside Inst.ok:
# B = 13107 > 8192, `b_batch`; Post is checked there as well): name, cu_seqlens dtype, beta bfloat16
# (else float32), B, T, H, HV, K, V, cu_seqlens (None: padded).
SETUPS = [
    ("chunk_b4_t1024_h4_d128", torch.int64, False, 4, 1024, 4, 4, 128, 128, None),
    ("chunk_b2_t75_h4_d64", torch.int64, False, 2, 75, 4, 4, 64, 64, None),
    ("chunk_b2_t500_h3_d60", torch.int64, False, 2, 500, 3, 3, 60, 60, None),
    ("chunk_b2_t1000_h3_d64", torch.int64, False, 2, 1000, 3, 3, 64, 64, None),
    ("chunk_b3_t1024_h4_d100", torch.int64, False, 3, 1024, 4, 4, 100, 100, None),
    ("chunk_b2_t1500_h4_d128", torch.int64, False, 2, 1500, 4, 4, 128, 128, None),
    ("chunk_b4_t2048_h8_d64", torch.int64, False, 4, 2048, 8, 8, 64, 64, None),
    ("chunk_b2_t256_h2_hv4_d64", torch.int64, False, 2, 256, 2, 4, 64, 64, None),
    ("chunk_b2_t512_h2_hv8_d64", torch.int64, False, 2, 512, 2, 8, 64, 64, None),
    ("chunk_b2_t1024_h4_hv8_d128", torch.int64, False, 2, 1024, 4, 8, 128, 128, None),
    ("wu_b2_t128_h2_d64", torch.int64, True, 2, 128, 2, 2, 64, 64, None),
    ("wu_b2_t128_h2_hv4_d64", torch.int64, True, 2, 128, 2, 4, 64, 64, None),
    ("wu_b1_t256_h4_d32", torch.int64, True, 1, 256, 4, 4, 32, 32, None),
    ("qwen3next_b2_t4096", torch.int64, False, 2, 4096, 16, 32, 128, 128, None),
    ("fla_gdn_b2_t4096", torch.int64, False, 2, 4096, 6, 6, 256, 512, None),
    ("doc_b4_t2048_h4_hv8_d512", torch.int64, True, 4, 2048, 4, 8, 512, 512, None),
    ("varlen_h4_d60_n1", torch.int64, False, 1, 15, 4, 4, 60, 60, [0, 15]),
    ("varlen_h4_d64_n3", torch.int64, False, 1, 1000, 4, 4, 64, 64, [0, 256, 500, 1000]),
    ("varlen_h4_d100_n5", torch.int64, False, 1, 2000, 4, 4, 100, 100, [0, 15, 100, 300, 1200, 2000]),
    ("varlen_h2_hv4_d64_n3", torch.int64, False, 1, 1000, 2, 4, 64, 64, [0, 256, 500, 1000]),
    ("varlen_h2_hv8_d64_n3", torch.int64, False, 1, 1000, 2, 8, 64, 64, [0, 256, 500, 1000]),
    ("wu_varlen_h2_d64", torch.int64, True, 1, 128, 2, 2, 64, 64, [0, 64, 128]),
    ("wu_varlen_h2_hv4_d64", torch.int64, True, 1, 256, 2, 4, 64, 64, [0, 128, 256]),
    ("wu_varlen_h1_d64", torch.int64, True, 1, 128, 1, 1, 64, 64, [0, 64, 128]),
    ("doc_varlen_h4_hv8_d512", torch.int64, True, 1, 8192, 4, 8, 512, 512, [0, 2048, 4096, 6144, 8192]),
    ("qwen3next_varlen_i32", torch.int32, False, 1, 16384, 16, 32, 128, 128, [0, 1, 777, 2048, 4097, 8192, 8256, 12000, 16384]),
    ("edge_t1_k1_v1", torch.int64, False, 1, 1, 1, 1, 1, 1, None),
    ("edge_bhv65535", torch.int64, False, 13107, 1, 1, 5, 1, 1, None),
    ("edge_varlen_empty", torch.int32, False, 1, 70, 2, 2, 64, 64, [0, 0, 5, 5, 70]),
    ("edge_varlen_132x1", torch.int64, False, 1, 132, 2, 4, 128, 64, list(range(133))),
    ("gdp_b2_t4096_h6_k256_v512", torch.int64, True, 2, 4096, 6, 6, 256, 512, None),
    ("gdp_varlen_h6_k256_v512", torch.int64, True, 1, 4096, 6, 6, 256, 512, [0, 1000, 2600, 4096]),
    ("precond_b2_t2048_h6_k256_v512", torch.int64, True, 2, 2048, 6, 6, 256, 512, None),
    ("mom_varlen_h4_k256_v512", torch.int64, True, 1, 4096, 4, 4, 256, 512, [0, 700, 1500, 2048, 2900, 4096]),
]


def seqs_of(B, T, cu):
    """The sequences (s0, len): padded `(b T, T)`, packed `(cu[n], cu[n+1] - cu[n])`."""
    if cu is None:
        return [(b * T, T) for b in range(B)]
    return [(cu[n], cu[n + 1] - cu[n]) for n in range(len(cu) - 1)]


def chunk_rows(seqs):
    """Per (sequence, chunk): the flat tokens of its 64 rows/columns and which are inside the
    sequence (`t // 64 * 64 + j < len`)."""
    starts, ends = [], []
    for s0, ln in seqs:
        for c0 in range(0, ln, BT):
            starts.append(s0 + c0)
            ends.append(s0 + ln)
    st = torch.tensor(starts, dtype=torch.int64, device=DEV)
    en = torch.tensor(ends, dtype=torch.int64, device=DEV)
    idx = st[:, None] + torch.arange(BT, device=DEV)[None, :]
    valid = idx < en[:, None]
    return idx.clamp_max(max(int(en.max().item()) - 1, 0) if len(ends) else 0), valid


def lift(x):
    """liftMag .bf16 on float64 values of bfloat16 numbers."""
    a = x.abs()
    return torch.where(a == 0, a, a.clamp_min(2.0 ** -126))


def make_inputs(B, T, H, HV, K, V, cu, gen, kind="real"):
    """Inputs as the gated DeltaNet passes them: L2-normalised `k`, Gaussian `v`, `beta` in (0, 1),
    `g` the chunk-local cumulative sum of `logsigmoid` in base 2 (non-positive), `A` the chunk's
    unit lower-triangular matrix with small entries below the diagonal (fla's test generator).
    `kind` selects an edge instead."""
    f32, bf = torch.float32, torch.bfloat16
    N = B * T
    if kind == "real":
        k = F.normalize(torch.randn(B, T, H, K, device=DEV, generator=gen), p=2, dim=-1).to(bf)
        v = torch.randn(B, T, HV, V, device=DEV, generator=gen).to(bf)
        beta = torch.rand(B, T, HV, device=DEV, generator=gen).sigmoid()
        x = F.logsigmoid(torch.randn(N, HV, device=DEV, generator=gen, dtype=torch.float64)) / math.log(2)
        idx, valid = chunk_rows(seqs_of(B, T, cu))
        g = torch.zeros(N, HV, dtype=torch.float64, device=DEV)
        xs = torch.where(valid[..., None], x[idx], 0).cumsum(1)
        g[idx[valid]] = xs[valid]
        g = g.to(f32).reshape(B, T, HV)
        A = torch.randn(N, HV, BT, device=DEV, generator=gen) * 0.1
        pos = torch.zeros(N, dtype=torch.int64, device=DEV)
        pos[idx[valid]] = torch.arange(BT, device=DEV).expand_as(idx)[valid]
        j = torch.arange(BT, device=DEV)
        tril = (j[None, :] <= pos[:, None]).to(f32)[:, None, :]
        eye = (j[None, :] == pos[:, None]).to(f32)[:, None, :]
        A = (A * tril + eye).to(bf).reshape(B, T, HV, BT)
    elif kind == "dense":
        # every entry of A of order 1, any sign; |k| of order 1
        k = torch.randn(B, T, H, K, device=DEV, generator=gen).to(bf)
        v = torch.randn(B, T, HV, V, device=DEV, generator=gen).to(bf)
        beta = torch.rand(B, T, HV, device=DEV, generator=gen) * 2
        g = -torch.rand(B, T, HV, device=DEV, generator=gen) * 30
        A = torch.randn(B, T, HV, BT, device=DEV, generator=gen).to(bf)
    elif kind == "coherent":
        # every term's rounding to bfloat16 (line 132) has the same sign and A is all +1: the
        # roundings add up
        k = torch.full((B, T, H, K), 1 + 2 ** -7, dtype=bf, device=DEV)
        v = torch.full((B, T, HV, V), 1 + 2 ** -7, dtype=bf, device=DEV)
        beta = torch.full((B, T, HV), 1 + 2 ** -9 + 2 ** -12, dtype=f32, device=DEV)
        g = torch.zeros(B, T, HV, dtype=f32, device=DEV)
        A = torch.ones(B, T, HV, BT, dtype=bf, device=DEV)
    elif kind == "coherent16":
        # for a bfloat16 beta: k * beta = 1.15625 * 1.75 rounds up to bfloat16 by 0.99 of the largest
        # relative step (line 129), and that times 2^-0.0055 rounds up again (line 132); A is all +1,
        # so the 64 terms' roundings add up
        k = torch.full((B, T, H, K), 1.15625, dtype=bf, device=DEV)
        v = torch.full((B, T, HV, V), 1.15625, dtype=bf, device=DEV)
        beta = torch.full((B, T, HV), 1.75, dtype=f32, device=DEV)
        g = torch.full((B, T, HV), -0.0055, dtype=f32, device=DEV)
        A = torch.ones(B, T, HV, BT, dtype=bf, device=DEV)
    elif kind == "max":
        # the largest magnitudes Pre allows: |A| = 2^64, |k|, |v| = 2^32, beta = 2^8, g = 8
        sgn = lambda *s: torch.randint(0, 2, s, device=DEV, generator=gen).to(f32) * 2 - 1
        k = (sgn(B, T, H, K) * 2.0 ** 32).to(bf)
        v = (sgn(B, T, HV, V) * 2.0 ** 32).to(bf)
        beta = torch.full((B, T, HV), 2.0 ** 8, dtype=f32, device=DEV)
        g = torch.full((B, T, HV), 8.0, dtype=f32, device=DEV)
        A = (torch.ones(B, T, HV, BT, device=DEV) * 2.0 ** 64).to(bf)
    elif kind == "tiny":
        # subnormal bfloat16 values, tiny beta, gates far below (exp2 flushes to 0)
        k = (torch.randn(B, T, H, K, device=DEV, generator=gen) * 2.0 ** -128).to(bf)
        v = (torch.randn(B, T, HV, V, device=DEV, generator=gen) * 2.0 ** -128).to(bf)
        beta = torch.rand(B, T, HV, device=DEV, generator=gen) * 2.0 ** -20
        g = -torch.rand(B, T, HV, device=DEV, generator=gen) * 300
        A = (torch.randn(B, T, HV, BT, device=DEV, generator=gen) * 2.0 ** -120).to(bf)
    else:
        raise ValueError(kind)
    return k.contiguous(), v.contiguous(), beta.contiguous(), A.contiguous(), g.contiguous()


def evaluate(k, v, beta, A, g, H, HV, cu, u, w):
    """Post on the outputs `u`, `w`: its references and tolerances in float64 from the host's
    inputs, over the rows inside a sequence. Returns the largest |u - uRef| / tolU, whether every
    `u` is finite, and the same for `w`. Computed a few value heads at a time."""
    B, T, _, K = k.shape
    V = v.shape[-1]
    N = B * T
    rows, valid = chunk_rows(seqs_of(B, T, cu))
    if rows.numel() == 0:
        return 0.0, True, 0.0, True
    Q = rows.shape[0]
    kf = k.reshape(N, H, K)
    vf = v.reshape(N, HV, V)
    bf = beta.reshape(N, HV)
    gf = g.reshape(N, HV)
    Af = A.reshape(N, HV, BT)
    uo = u.reshape(N, HV, V)
    wo = w.reshape(N, HV, K)
    cm = valid.to(torch.float64)[:, None, None, :]       # columns inside the sequence
    rm = valid[:, :, None, None]                          # rows inside the sequence
    hg = max(1, (1 << 24) // (Q * BT * max(K, V, BT)))
    # cW: 2^-8 + 2^-14 for a float32 beta, 2^-7 + 2^-13 for a bfloat16 one
    cw = 2.0 ** -7 + 2.0 ** -13 if beta.dtype == torch.bfloat16 else 2.0 ** -8 + 2.0 ** -14
    ru = rw = 0.0
    fu = fw = True
    for h0 in range(0, HV, hg):
        hs = torch.arange(h0, min(HV, h0 + hg), device=DEV)
        Ar = Af[:, hs].double()[rows]                     # [Q, 64 r, h, 64 j]
        Arow = Ar * cm
        Lrow = lift(Ar) * cm
        del Ar
        # vb_j = bf16(fl32(v * beta)) (float32 beta; torch's float32 product and conversion are IEEE,
        # to nearest) or the IEEE bfloat16 product v * beta (bfloat16 beta): computed here as the
        # float32 product rounded to bfloat16, which is that product whenever the exact product is
        # at least 2^-126; below, the float32 product is subnormal and rounds twice, a difference
        # under 2^-133 that tolU's floors absorb
        vb = (vf[:, hs].float() * bf[:, hs].float()[..., None]).to(torch.bfloat16).double()[rows]
        uRef = torch.einsum('qrhj,qjhc->qrhc', Arow, vb)
        uMag = torch.einsum('qrhj,qjhc->qrhc', Lrow, lift(vb))
        del vb
        tolU = 2.0 ** -8 * uRef.abs() + (1 + 2.0 ** -8) * (66 * 2.0 ** -22 * uMag + 66 * 2.0 ** -126) + 2.0 ** -134
        del uMag
        o = uo[:, hs].double()[rows]
        fu = fu and bool(torch.isfinite(o)[rm.expand_as(o)].all().item())
        r = torch.where(rm, (o - uRef).abs() / tolU, 0.0)
        ru = max(ru, float(r.max().item()))
        del uRef, tolU, o, r
        kh = hs // (HV // H)
        x = kf[:, kh].double() * bf[:, hs].double()[..., None]   # k[f, h // G, c] * beta[f, h]
        e = torch.exp2(gf[:, hs].double())[..., None]
        wRef = torch.einsum('qrhj,qjhc->qrhc', Arow, (x * e)[rows])
        wMag = torch.einsum('qrhj,qjhc->qrhc', Lrow, (x.abs() * e)[rows])
        tolW = 2.0 ** -8 * wRef.abs() + cw * wMag
        del wMag
        tolW += 2.0 ** -124 * torch.einsum('qrhj,qjhc->qrhc', Lrow, (x.abs() + 1)[rows]) + 2.0 ** -119
        del x, e, Arow, Lrow
        o = wo[:, hs].double()[rows]
        fw = fw and bool(torch.isfinite(o)[rm.expand_as(o)].all().item())
        r = torch.where(rm, (o - wRef).abs() / tolW, 0.0)
        rw = max(rw, float(r.max().item()))
        del wRef, tolW, o, r
    return ru, fu, rw, fw


def run_case(name, idt, bbf, B, T, H, HV, K, V, cu, kind, seed, negatives=False):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    k, v, beta, A, g = make_inputs(B, T, H, HV, K, V, cu, gen, kind)
    if bbf:
        beta = beta.to(torch.bfloat16)
    cu_t = torch.tensor(cu, dtype=idt, device=DEV) if cu is not None else None
    ci = wy.prepare_chunk_indices(cu_t, BT) if cu_t is not None else None
    copies = [t.clone() for t in (k, v, beta, A, g)] + ([cu_t.clone(), ci.clone()] if cu_t is not None else [])
    w, u = wy.recompute_w_u_fwd(k, v, beta, A, g, cu_seqlens=cu_t, chunk_indices=ci)
    torch.cuda.synchronize()
    after = [k, v, beta, A, g] + ([cu_t, ci] if cu_t is not None else [])
    frame = all(torch.equal(a, b) for a, b in zip(copies, after))
    ru, fu, rw, fw = evaluate(k, v, beta, A, g, H, HV, cu, u, w)
    ok = ru <= 1 and rw <= 1 and fu and fw and frame
    print(f"{'PASS' if ok else 'FAIL'} {name:30s} {kind:9s} {'bf16' if bbf else 'f32 '} u {ru:.4f}  w {rw:.4f}  finite {fu and fw}  frame {frame}")
    neg_ok = True
    if negatives:
        neg_ok = run_negatives(k, v, beta, A, g, H, HV, cu, w, u)
    torch.cuda.empty_cache()
    return ok and neg_ok


def run_negatives(k, v, beta, A, g, H, HV, cu, w, u):
    """Outputs the claim must reject."""
    B, T = k.shape[:2]
    cuf = (lambda: None) if cu is None else (lambda: torch.tensor(cu, device=DEV))
    U = lambda uu: evaluate(k, v, beta, A, g, H, HV, cu, uu, w)[0]
    W = lambda ww: evaluate(k, v, beta, A, g, H, HV, cu, u, ww)[2]
    res = []
    # 1. w without the gate factor 2^g (the kernel's USE_G = False path)
    res.append(("w without 2^g", W(wy.recompute_w_u_fwd(k, v, beta, A, None, cu_seqlens=cuf())[0])))
    # 2. u with A transposed within each chunk
    rows, valid = chunk_rows(seqs_of(B, T, cu))
    At = A.clone()
    Af = A.reshape(B * T, -1, BT)
    Atf = At.reshape(B * T, -1, BT)
    Atf[rows[valid]] = Af[rows].permute(0, 3, 2, 1)[valid]
    res.append(("u with A transposed", U(wy.recompute_w_u_fwd(k, v, beta, At, g, cu_seqlens=cuf())[1])))
    # 3. w from key head h % H instead of h // (HV // H) (HV key heads given, each value head reads
    #    its own)
    if HV > H:
        kw = k[:, :, torch.arange(HV, device=DEV) % H].contiguous()
        res.append(("w with the wrong key head", W(wy.recompute_w_u_fwd(kw, v, beta, A, g, cu_seqlens=cuf())[0])))
    # 4. chunks not restarted at sequence starts (the packed batch launched as one padded sequence)
    if cu is not None and any(c % BT for c in cu):
        res.append(("packed launched as padded (u)", U(wy.recompute_w_u_fwd(k, v, beta, A, g)[1])))
    # 5. one element of u off by 2%
    u5 = u.clone().reshape(-1)
    i = int(torch.argmax(u5.abs().float()).item())
    u5[i] = (u5[i].float() * 1.02).to(torch.bfloat16)
    res.append(("u with one element off by 2%", U(u5.reshape(u.shape))))
    # 6. w scaled by 1 + 2^-6
    res.append(("w scaled by 1 + 2^-6", W((w.float() * (1 + 2 ** -6)).to(torch.bfloat16))))
    ok = True
    for label, r in res:
        rej = r > 1
        ok = ok and rej
        print(f"  {'REJECTED' if rej else 'ACCEPTED (bad)'} {label}: worst {r:.3f}")
    return ok


def run_guarded():
    """A direct launch whose w and u are views between guard regions: every guard byte keeps its
    pattern (nothing outside w and u is written)."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    B, T, H, HV, K, V = 2, 100, 2, 4, 96, 80
    k, v, beta, A, g = make_inputs(B, T, H, HV, K, V, None, gen)
    G = 4096
    nw, nu = B * T * HV * K, B * T * HV * V
    wbuf = torch.full((nw + 2 * G,), 1234.5, dtype=torch.bfloat16, device=DEV)
    ubuf = torch.full((nu + 2 * G,), -77.25, dtype=torch.bfloat16, device=DEV)
    w = wbuf[G:G + nw].view(B, T, HV, K)
    u = ubuf[G:G + nu].view(B, T, HV, V)
    wy.recompute_w_u_fwd_kernel[(triton_cdiv(T, BT), B * HV)](
        k=k, v=v, beta=beta, w=w, u=u, A=A, g=g, cu_seqlens=None, chunk_indices=None, T=T,
        H=H, HV=HV, K=K, V=V, BT=BT, BK=64, BV=64, USE_G=True, IS_VARLEN=False,
        num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    gw = bool((wbuf[:G] == 1234.5).all() and (wbuf[G + nw:] == 1234.5).all())
    gu = bool((ubuf[:G] == -77.25).all() and (ubuf[G + nu:] == -77.25).all())
    ru, fu, rw, fw = evaluate(k, v, beta, A, g, H, HV, None, u, w)
    ok = gw and gu and ru <= 1 and rw <= 1 and fu and fw
    print(f"{'PASS' if ok else 'FAIL'} guarded direct launch: guards intact {gw and gu}, u {ru:.4f}, w {rw:.4f}")
    return ok


def triton_cdiv(a, b):
    return (a + b - 1) // b


def run_grid_limit():
    """`B * HV = 65536`: grid axis 1 beyond CUDA's 65535, the launch must fail (outside Inst.ok)."""
    B, T, H, HV, K, V = 16384, 1, 1, 4, 1, 1
    gen = torch.Generator(device=DEV).manual_seed(3)
    k, v, beta, A, g = make_inputs(B, T, H, HV, K, V, None, gen)
    try:
        wy.recompute_w_u_fwd(k, v, beta, A, g)
        torch.cuda.synchronize()
    except Exception as ex:  # the launcher reports the invalid grid
        print(f"PASS B*HV = 65536 fails to launch: {type(ex).__name__}: {str(ex).splitlines()[0][:80]}")
        return True
    print("FAIL B*HV = 65536 launched")
    return False


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    ok = True
    seed = 1
    for (name, idt, bbf, B, T, H, HV, K, V, cu) in SETUPS:
        neg = name in ("chunk_b2_t1024_h4_hv8_d128", "varlen_h4_d100_n5", "wu_b2_t128_h2_hv4_d64",
                       "mom_varlen_h4_k256_v512")
        ok &= run_case(name, idt, bbf, B, T, H, HV, K, V, cu, "real", seed, negatives=neg)
        seed += 1
    # edges of Inst.ok / Pre
    edges = [
        ("dense_b2_t500_h2_hv4_d96", torch.int64, 2, 500, 2, 4, 96, 96, None, "dense"),
        ("dense_varlen", torch.int64, 1, 1000, 2, 4, 64, 128, [0, 3, 300, 1000], "dense"),
        ("coherent_b1_t128_d64", torch.int64, 1, 128, 1, 1, 64, 64, None, "coherent"),
        ("max_b1_t130_d64", torch.int64, 1, 130, 1, 2, 64, 64, None, "max"),
        ("tiny_b2_t100_d64", torch.int64, 2, 100, 2, 2, 64, 64, None, "tiny"),
        ("max_varlen", torch.int32, 1, 200, 2, 2, 70, 33, [0, 63, 64, 65, 200], "max"),
        ("coherent16_b1_t128_d64", torch.int64, 1, 128, 1, 1, 64, 64, None, "coherent16"),
    ]
    for (name, idt, B, T, H, HV, K, V, cu, kind) in edges:
        for bbf in ((True,) if kind == "coherent16" else (False, True)):
            ok &= run_case(name, idt, bbf, B, T, H, HV, K, V, cu, kind, seed)
        seed += 1
    ok &= run_guarded()
    ok &= run_grid_limit()
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
