"""GPU check of the prepared claim `Claims.FusedMoe` (Claim.lean).

Runs the kernel's own host (`source/fused_moe.py`, imported by path) on the
launches SETUP.md names and on launches at the edges of `Inst.ok`, and checks `Post` element by
element, computed in float64 from the host's inputs exactly as `Post` states it:

* a routed row `(t, j)` (local expert `e` of `topk_ids[t, j]`, through the expert map when there is
  one; the host's input, not what the kernel reads) and column `c`, with `a` row `t` of `A` for the
  first GEMM (`fused_moe_gemm1`) and row `t·top_k + j` for the second (`fused_moe_gemm2`):
  `S = Σ_k a[k] · B[e, c, k]`, `E = tol BK K a B[e, c]` (`accErr` of Claim.lean: the chunk
  recursion of `Rules.Num.DotBF16Chunk` over the exact partial sums of the 64-chunks, with the
  lifted products `liftMag q = 0` at `0`, else `max(|q|, 2^-126)`; `BK` the tile's
  `BLOCK_SIZE_K`), `tol_chunk` below. Without `MUL_ROUTED_WEIGHT`:
  `|C - S| <= E + (|S| + E)·2^-8 + 2^-134`; with it, `w = topk_weights[t, j]`,
  `tolW = |w|·E + ((|S| + E)·|w|)·2^-24 + 2^-150` and `|C - S·w| <= tolW + (|S·w| + tolW)·2^-8 + 2^-134`
  (`outFin`, under its room conditions, which every case here meets);
* a row whose block has expert `-1`: the `bfloat16` `+0.0` (bit for bit);
* a row not routed to a local expert: unchanged (the host's `C` is filled with a sentinel pattern
  before the launch, by wrapping `torch.empty` during the host call);
* every byte outside the matrix `C` unchanged: the inputs and alignment buffers are compared with
  copies, and one direct launch through the host's `invoke_fused_moe_triton_kernel` writes into a
  `C` placed between two guard regions.

Prints, per case, the tile and assignment the host chose, the largest error / tolerance of the
output, `tol / |reference|` (median, 90th percentile) and the exactness and frame results; on some
cases, negative tests (wrong outputs, counted by the elements `Post` rejects; one rejected element
rejects the output). Each case stays under 4 GiB of device memory. Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "fused_moe.py")
spec = importlib.util.spec_from_file_location("fused_moe_src", SRC)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

DEV = "cuda"



RHO, SIG, ALP = 2.0 ** -13, 2.0 ** -20, 2.0 ** -143   # rhoChunk16, sigmaChunk16, alphaChunk16


def tol_chunk(a, w, K, BK):
    """`tol BK K va vb` (Claim.lean: `accErr`, `tripErr`, `chainE`, `chunkStep`) for every row of
    `a` [r, K] and row of `w` [n, K] (float64 values of bfloat16 numbers): [r, n]. Positions past
    `K` are zeros; the chunks are the 64 consecutive positions from `64·g`."""
    r, n = a.shape[0], w.shape[0]
    Kp = -(-K // BK) * BK
    G, q, T = Kp // 64, BK // 64, Kp // BK
    out = torch.empty((r, n), dtype=torch.float64, device=a.device)
    nb = max(1, (2 ** 25) // max(1, r * G))
    ap = torch.zeros((r, Kp), dtype=torch.float64, device=a.device)
    ap[:, :K] = a
    la = torch.zeros_like(ap)
    la[:, :K] = lift(a)
    ap, la = ap.view(r, G, 64), la.view(r, G, 64)
    for n0 in range(0, n, nb):
        wp = torch.zeros((min(nb, n - n0), Kp), dtype=torch.float64, device=a.device)
        wp[:, :K] = w[n0:n0 + nb]
        lw = torch.zeros_like(wp)
        lw[:, :K] = lift(w[n0:n0 + nb])
        c = torch.einsum("rgk,ngk->rng", ap, wp.view(-1, G, 64))     # chunkSum
        m = torch.einsum("rgk,ngk->rng", la, lw.view(-1, G, 64))     # chunkLift
        P = torch.cat([torch.zeros_like(c[..., :1]), c.cumsum(-1)], -1)   # preSum

        def step(e, Pg, mg, Pg1):
            return (1 + RHO + SIG) * e + RHO * torch.maximum(Pg.abs(), mg) + SIG * Pg1.abs() + ALP

        E = torch.zeros_like(c[..., 0])
        for t in range(T):
            g0 = t * q
            ea, L = E, torch.zeros_like(E)
            for j in range(q):
                ea = step(ea, P[..., g0 + j], m[..., g0 + j], P[..., g0 + j + 1])
                L = step(L, P[..., g0 + j] - P[..., g0], m[..., g0 + j], P[..., g0 + j + 1] - P[..., g0])
            eb = E + L + 3 * 2.0 ** -24 * ((P[..., g0 + q] - P[..., g0]).abs() + L + P[..., g0].abs() + E) \
                + 4 * 2.0 ** -150
            E = torch.maximum(ea, eb)
        out[:, n0:n0 + nb] = E
        del c, m, P, wp, lw
    return out


def lift(x):
    """liftMag .bf16 on float64 values of bfloat16 numbers."""
    a = x.abs()
    return torch.where(a == 0, a, a.clamp_min(2.0 ** -126))


def fingerprint(x):
    """A cheap, allocation-light fingerprint of a tensor's bytes: the sum of its 16-bit words
    (summed in chunks, so that no large temporary is allocated) and the words at a fixed sample of
    positions."""
    w = x.contiguous().view(torch.int16).reshape(-1)
    total = 0
    for i in range(0, w.numel(), 1 << 24):
        total += w[i:i + (1 << 24)].sum(dtype=torch.int64).item()
    idx = torch.arange(0, w.numel(), max(1, w.numel() >> 20), device=w.device)
    return (total, w[idx].clone())


def same_fp(f, g):
    return f[0] == g[0] and torch.equal(f[1], g[1])


class Sentinel:
    """Wraps torch.empty during the host call: the host's C starts as a known bit pattern."""

    PAT = -12345  # int16 bit pattern 0xCFC7: the bfloat16 -6.6e9, never a kernel result here

    def __enter__(self):
        self.orig = torch.empty

        def empty(*a, **k):
            t = self.orig(*a, **k)
            if t.dtype == torch.bfloat16:
                t.view(torch.int16).fill_(self.PAT)
            return t

        torch.empty = empty
        return self

    def __exit__(self, *exc):
        torch.empty = self.orig


def make_inputs(M, N, K, E, top_k, local_experts, gemm, scale, seed):
    """The host's inputs: for the first GEMM `A = hidden_states` `[M, K]` and `w1` `[E_local, N, K]`;
    for the second `A = intermediate_cache2` `[M·top_k, K]` and `w2` `[E_local, N, K]`."""
    gen = torch.Generator().manual_seed(seed)
    rows = M if gemm == 1 else M * top_k
    a = (torch.randn((rows, K), generator=gen) * scale).to(torch.bfloat16).to(DEV)
    score = torch.randn((M, E), generator=gen).to(DEV)
    topk_weights, topk_ids = torch.topk(torch.softmax(score, dim=-1), top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    expert_map = None
    E_local = E if local_experts is None else local_experts
    w = (torch.randn((E_local, N, K), generator=gen) * scale).to(torch.bfloat16).to(DEV)
    if local_experts is not None:
        expert_map = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        expert_map[:local_experts] = torch.arange(local_experts, dtype=torch.int32, device=DEV)
    return a, w, topk_weights, topk_ids, expert_map, E_local


NEGATIVES = ("drop the last K block", "the next expert's weights", "a missing or wrong scale",
             "a bfloat16 accumulator", "all zero")
# Post must reject each of them on some element of every case that runs the negatives.
MUST_REJECT = NEGATIVES


def run_case(name, M, N, K, E, top_k, on_input, local_experts=None, ignore_invalid=True,
             scale=0.1, seed=0, gemm=1, negatives=False):
    """One launch of the host (`fused_moe_gemm1` or `fused_moe_gemm2`), `Post` checked element by
    element. `on_input` is `apply_router_weight_on_input`; the kernel's `MUL_ROUTED_WEIGHT` is it
    for the first GEMM and its negation for the second."""
    a, w1, topk_weights, topk_ids, expert_map, E_local = make_inputs(
        M, N, K, E, top_k, local_experts, gemm, scale, seed)
    copies = [fingerprint(x) for x in (a, w1, topk_weights, topk_ids)]
    # the host's configuration: vLLM's tuned H200 file for w2's shape [E_local, ·, N_w2], its entry
    # nearest M, else get_default_config (try_get_optimal_moe_config)
    n_w2 = N // 2 if gemm == 1 else K
    config = fm.try_get_optimal_moe_config(None, (E_local, 0, n_w2), top_k, None, M)
    BM, BK = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]
    naive = expert_map is None and M * top_k * 4 <= E
    rw = on_input if gemm == 1 else not on_input
    g = E if expert_map is not None else -1

    if ignore_invalid:
        with Sentinel():
            host = fm.fused_moe_gemm1 if gemm == 1 else fm.fused_moe_gemm2
            out = host(a, w1, topk_weights, topk_ids, on_input, global_num_experts=g,
                       expert_map=expert_map)
    else:
        # vLLM's other expert-parallel mode: the alignment over global ids, then `expert_ids`
        # mapped through the expert map (blocks of other ranks' experts get -1: the zeros path).
        s, e_ids, ntp = fm.moe_align_block_size(topk_ids, BM, E, expert_map,
                                                ignore_invalid_experts=False)
        out = torch.empty((M, top_k, N), device=DEV, dtype=torch.bfloat16)
        out.view(torch.int16).fill_(Sentinel.PAT)
        fm.invoke_fused_moe_triton_kernel(a, w1, out, topk_weights, s, e_ids, ntp, rw,
                                          top_k if gemm == 1 else 1, config,
                                          compute_type=fm.tl.bfloat16)
    torch.cuda.synchronize()
    inputs_same = all(same_fp(f, fingerprint(x))
                      for f, x in zip(copies, (a, w1, topk_weights, topk_ids)))

    local = topk_ids.long() if expert_map is None else expert_map[topk_ids.long()].long()
    ad = a.double()
    worst = 0.0
    ratios = []           # tol / |reference| per element (a sample)
    neg_fail = {n: 0 for n in NEGATIVES}
    n_el = 0
    outd = out.double()
    out_bits = out.view(torch.int16)
    for e in range(E_local):
        t, j = torch.nonzero(local == e, as_tuple=True)
        if t.numel() == 0:
            continue
        rows = t if gemm == 1 else t * top_k + j
        wd = w1[e].double()
        S = ad[rows] @ wd.t()                    # [rows, N], exact products, float64 sum
        Et = tol_chunk(ad[rows], wd, K, BK)
        wt = topk_weights[t, j].double()[:, None]

        def tol_of(S, Et):
            if rw:
                ref = S * wt
                tolW = wt.abs() * Et + ((S.abs() + Et) * wt.abs()) * 2.0 ** -24 + 2.0 ** -150
                return ref, tolW + (ref.abs() + tolW) * 2.0 ** -8 + 2.0 ** -134
            return S, Et + (S.abs() + Et) * 2.0 ** -8 + 2.0 ** -134

        ref, tol = tol_of(S, Et)
        n_el += ref.numel()
        got = outd[t, j]
        assert torch.isfinite(got).all(), name
        r = ((got - ref).abs() / tol).max().item()
        worst = max(worst, r)
        rr = (tol / ref.abs().clamp_min(1e-30)).reshape(-1)
        ratios.append(rr[:: max(1, rr.numel() // 20000)].float().cpu())
        if negatives:
            # wrong results the kernel could give; each must break Post somewhere
            Kd = ((K - 1) // BK) * BK
            S_drop = ad[rows][:, :Kd] @ wd[:, :Kd].t()
            e2 = (e + 1) % E_local
            S_next = ad[rows] @ w1[e2].double().t()
            bad = {
                "drop the last K block": S_drop * wt if rw else S_drop,
                "the next expert's weights": S_next * wt if rw else S_next,
                "a missing or wrong scale": S if rw else S * 0.5,
                "a bfloat16 accumulator": None,
                "all zero": torch.zeros_like(S),
            }
            for n, v in bad.items():
                if v is None:
                    continue
                v = v.to(torch.bfloat16).double()
                neg_fail[n] += int(((v - ref).abs() > tol).sum().item())
            # a bfloat16 accumulator: each BK step rounded to bfloat16
            acc = torch.zeros_like(S)
            for k0 in range(0, K, BK):
                acc = (acc + ad[rows][:, k0:k0 + BK] @ wd[:, k0:k0 + BK].t()).to(torch.bfloat16).double()
            v = (acc * wt if rw else acc).to(torch.bfloat16).double()
            neg_fail["a bfloat16 accumulator"] += int(((v - ref).abs() > tol).sum().item())
            del S_drop, S_next, acc, v
        del S, Et, ref, tol, got, wd
    rat = torch.cat(ratios)
    med, p90 = rat.median().item(), rat.quantile(0.9).item()
    # rows not routed to a local expert
    unrouted = ~((local >= 0) & (local < E_local))
    if ignore_invalid:
        exact_ok = True
        pat = 0 if (gemm == 2 and expert_map is not None) else Sentinel.PAT  # the host zeroes cache3
        frame_rows = bool((out_bits[unrouted] == pat).all().item())
    else:
        # rows routed to another rank's expert sit in blocks whose expert id is -1: bf16 +0.0
        routed_other = (topk_ids.long() >= 0) & (topk_ids.long() < E) & unrouted
        exact_ok = bool((out_bits[routed_other] == 0).all().item())
        frame_rows = True
    neg_ok = all(neg_fail[n] > 0 for n in MUST_REJECT) if negatives else True
    print(f"{name}: GEMM {gemm} M={M} N={N} K={K} E={E} E_local={E_local} top_k={top_k} "
          f"rw={rw} naive={naive} tile={(BM, config['BLOCK_SIZE_N'], BK, config['GROUP_SIZE_M'])}: "
          f"max err/tol {worst:.4f}, tol/|ref| median {med:.4f} p90 {p90:.3f}, zeros exact "
          f"{exact_ok}, unrouted rows unchanged {frame_rows}, inputs unchanged {inputs_same}")
    if negatives:
        print("    negatives (elements outside tol): " +
              ", ".join(f"{n} {v}" for n, v in neg_fail.items()) +
              f" (of {n_el}); all rejected: {neg_ok}")
    ok = worst < 1 and exact_ok and frame_rows and inputs_same and neg_ok
    del a, w1, out, outd, copies
    torch.cuda.empty_cache()
    return ok, worst


def guard_case():
    """A direct launch through the host's launcher into a `C` between two guard regions: every
    byte outside the matrix `C`, the alignment buffers included, unchanged."""
    M, N, K, E, top_k = 100, 1022, 511, 8, 2
    gen = torch.Generator().manual_seed(7)
    a = (torch.randn((M, K), generator=gen) / 10).to(DEV, torch.bfloat16)
    w1 = (torch.randn((E, N, K), generator=gen) / 10).to(DEV, torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn((M, E), generator=gen).to(DEV), -1), top_k, -1)
    ti = ti.to(torch.int32)
    config = fm.try_get_optimal_moe_config(None, (E, 0, N // 2), top_k, None, M)
    s, e_ids, ntp = fm.moe_align_block_size(ti, config["BLOCK_SIZE_M"], E, None,
                                            ignore_invalid_experts=True)
    G = 4096
    store = torch.empty(G + M * top_k * N + G, device=DEV, dtype=torch.bfloat16)
    store.view(torch.int16).fill_(Sentinel.PAT)
    C = store[G:G + M * top_k * N].view(M, top_k, N)
    keep = [x.clone() for x in (a, w1, tw, s, e_ids, ntp)]
    fm.invoke_fused_moe_triton_kernel(a, w1, C, tw, s, e_ids, ntp, True, top_k, config,
                                      compute_type=fm.tl.bfloat16)
    torch.cuda.synchronize()
    bits = store.view(torch.int16)
    guards = bool((bits[:G] == Sentinel.PAT).all().item() and (bits[-G:] == Sentinel.PAT).all().item())
    same = all(torch.equal(x, y) for x, y in zip(keep, (a, w1, tw, s, e_ids, ntp)))
    print(f"guard: bytes before and after C unchanged {guards}, inputs and alignment buffers "
          f"unchanged {same}")
    return guards and same


def main():
    T = [  # the host's own test cases (source lines 865-873): (name, M, N, K, E, top_k, on_input, local)
        ("m1_n256_k128_e8_top2", 1, 256, 128, 8, 2, False, None),
        ("m33_n4096_k128_e64_top6", 33, 4096, 128, 64, 6, False, None),
        ("m100_n1022_k511_e8_top2", 100, 1022, 511, 8, 2, False, None),
        ("m222_n512_k7168_e256_top8", 222, 512, 7168, 256, 8, False, None),
        ("m1024_n1536_k2048_e128_top8", 1024, 1536, 2048, 128, 8, False, None),
        ("m2048_n3584_k4096_e8_top2", 2048, 3584, 4096, 8, 2, False, None),
        ("m64_n2048_k5120_e16_top1_rw", 64, 2048, 5120, 16, 1, True, None),
        ("m300_n1024_k1024_e64_top6_ep16", 300, 1024, 1024, 64, 6, False, 16),
        ("m2_n512_k7168_e256_top8", 2, 512, 7168, 256, 8, False, None),
    ]
    cases = []
    for (n, M, N, K, E, tk, oi, loc) in T:
        cases.append(("gemm1_" + n, M, N, K, E, tk, oi, loc))
        # its second GEMM: w2 [E, K, N / 2]
        cases.append(dict(name="gemm2_" + n, M=M, N=K, K=N // 2, E=E, top_k=tk, on_input=oi,
                          local_experts=loc, gemm=2))
    cases += [
        # SETUP's models (sizes cut to stay under 4 GB of device memory)
        dict(name="qwen3_m4096_gemm1_rw", M=4096, N=1536, K=2048, E=128, top_k=8, on_input=True),
        dict(name="qwen3_m4_naive_gemm2", M=4, N=2048, K=768, E=128, top_k=8, on_input=False,
             gemm=2, negatives=True),
        dict(name="qwen3_m80_gemm1", M=80, N=1536, K=2048, E=128, top_k=8, on_input=False),
        dict(name="deepseek_v3_tp8_m2048_gemm1", M=2048, N=512, K=7168, E=256, top_k=8,
             on_input=False, negatives=True),
        dict(name="deepseek_v3_tp8_m1_naive_gemm1", M=1, N=512, K=7168, E=256, top_k=8,
             on_input=False, negatives=True),
        dict(name="deepseek_v3_tp8_m8_naive_gemm2", M=8, N=7168, K=256, E=256, top_k=8,
             on_input=False, gemm=2),
        dict(name="deepseek_v3_tp8_m9_gemm1", M=9, N=512, K=7168, E=256, top_k=8, on_input=False),
        dict(name="mixtral_tp2_m2048_gemm1", M=2048, N=14336, K=4096, E=8, top_k=2, on_input=False),
        dict(name="mixtral_m1_naive_gemm2_k14336", M=1, N=4096, K=14336, E=8, top_k=2,
             on_input=False, gemm=2, negatives=True),
        dict(name="mixtral_m512_gemm2_k14336", M=512, N=2048, K=14336, E=8, top_k=2,
             on_input=False, gemm=2, negatives=True),
        dict(name="mixtral_8x22b_m256_gemm2_k16384", M=256, N=1024, K=16384, E=8, top_k=2,
             on_input=False, gemm=2, negatives=True),
        dict(name="grok1_m128_gemm2_k32768", M=128, N=512, K=32768, E=8, top_k=2, on_input=False,
             gemm=2, negatives=True),
        # tiles of the tuned H200 files: BLOCK_SIZE_K = 256, GROUP_SIZE_M = 64, BLOCK_SIZE_N = 32 and 256
        dict(name="mixtral_m16_gemm2_k14336_bk256", M=16, N=4096, K=14336, E=8, top_k=2,
             on_input=False, gemm=2, negatives=True),
        dict(name="mixtral_m64_gemm2_k14336_gm64", M=64, N=4096, K=14336, E=8, top_k=2,
             on_input=False, gemm=2),
        dict(name="mixtral_tp8_m1_naive_gemm1_bk256", M=1, N=3584, K=4096, E=8, top_k=2,
             on_input=False, negatives=True),
        dict(name="mixtral_tp2_m1024_gemm2_bn256_gm64", M=1024, N=4096, K=7168, E=8, top_k=2,
             on_input=False, gemm=2),
        dict(name="llama4_scout_tp8_m1_naive_gemm1_bn32", M=1, N=2048, K=5120, E=16, top_k=1,
             on_input=True),
        dict(name="m40000_n256_k128_e8_top2_gemm1", M=40000, N=256, K=128, E=8, top_k=2,
             on_input=False),
        # edges of Inst.ok: K = 2^15 - 1 with N not a multiple of the tile, one expert with
        # GROUP_SIZE_M = 16 and the tile (64, 128, 64), N = 129
        dict(name="edge_k32767_n1000_rw", M=96, N=1000, K=32767, E=8, top_k=2, on_input=True),
        dict(name="edge_e1_top1_m300", M=300, N=384, K=1024, E=1, top_k=1, on_input=False),
        dict(name="edge_k64_n129", M=777, N=129, K=64, E=16, top_k=6, on_input=True),
        # larger inputs (|x| up to ~20): the tolerance is relative, the margin the same
        dict(name="edge_scale4", M=256, N=512, K=4096, E=8, top_k=2, on_input=True, scale=4.0),
        # the expert -1 path (vLLM's expert map applied to `expert_ids`): zeros for other ranks' rows
        dict(name="ep_minus1_m300_e64_local16", M=300, N=1024, K=1024, E=64, top_k=6,
             on_input=False, local_experts=16, ignore_invalid=False),
    ]
    ok = True
    worst = 0.0
    peak = 0
    for c in cases:
        torch.cuda.reset_peak_memory_stats()
        r, w = run_case(*c) if isinstance(c, tuple) else run_case(**c)
        p_case = torch.cuda.max_memory_allocated()
        if p_case >= 2 ** 31:
            print(f"    peak device memory {p_case / 2 ** 30:.2f} GiB")
        peak = max(peak, p_case)
        ok &= r
        worst = max(worst, w)
    ok &= guard_case()
    peak = max(peak, torch.cuda.max_memory_allocated())
    print(f"peak device memory of a case: {peak / 2 ** 30:.2f} GiB")
    ok &= peak < 4 * 2 ** 30
    print(f"overall: max err/tol {worst:.4f} (margin {1 - worst:.4f}); "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
