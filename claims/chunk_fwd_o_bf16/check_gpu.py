"""GPU check of the `chunk_fwd_o_bf16` claim (Claim.lean's `Post`), on an H200 with Triton 3.8.0.

For each launch of SETUP.md (and the edges of `Inst.ok`), the script builds the inputs, runs the
host `chunk_fwd_o` of `source/chunk_o.py`, and runs the same launch directly into an output buffer
prefilled with a sentinel and guarded on both sides. It then checks, exactly as `Post` states them:

* the host's output and the direct launch's agree bit for bit on every written element;
* for every checked chunk `r`, token `t`, value head `j` and column `e` whose inputs are `InRange`,
  the output is finite and `|o - ref| <= tol`, with `ref`, `Mabs`, `Mag`, `Flo` and `tol` computed
  in float64 from the inputs as Claim.lean defines them (chunks: all of them when there are at most
  64, else the first, the last, up to 16 partial ones and 64 drawn at random);
* the frame: every element of the guarded buffer outside the written elements keeps the sentinel,
  and every input is unchanged;
* on a variable-length launch, the tables are `TablesOk`.

Negative tests: outputs that differ from the kernel's meaning (no causal mask, `e` for `2` in the
gates, the wrong query head, the state read untransposed, the scale squared, the previous chunk's
state, one element moved by twice its tolerance, and with `g_gamma` the gate taken at the global
token position or without its `+ 1`) must be rejected by the same check. The kernel's
grid limit (`B * HV >= 65536` on a fixed-length batch) is checked to fail on the host.

Each launch stays below 4 GB of device memory.
"""
import sys
sys.dont_write_bytecode = True

import importlib.util
import os
import math

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("chunk_o", os.path.join(HERE, "source", "chunk_o.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

DEV = "cuda"
BT = 64
SENT = -12345.0          # the sentinel (exact in bfloat16)
GUARD = 4096             # guard elements on each side of the direct launch's output

def liftmag(x):
    """`Rules.Num.liftMag .bf16`: 0 for 0, else max(|x|, 2^-126)."""
    a = x.abs()
    return torch.where(x == 0, torch.zeros_like(a), torch.clamp(a, min=2.0 ** -126))


def bf16_finite(x):
    return torch.isfinite(x.float())


# ----------------------------------------------------------------------------------------------
# inputs

def make_inputs(B, T, H, HV, K, V, use_g, svf, cu, idx_dtype, gen, style="gdn", NT=None):
    q = torch.randn((B, T, H, K), generator=gen)
    k = torch.randn((B, T, H, K), generator=gen)
    if style == "gdn":   # gated DeltaNet's: L2-normalised q, k
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = torch.randn((B, T, HV, V), generator=gen)
    if NT is None:
        NT = triton.cdiv(T, BT) if cu is None else None
    g = None
    if use_g:
        if style == "gdn":   # chunk-local cumulative sums of nonpositive base-2 log decays
            a = -torch.rand((B, T, HV), generator=gen) * 0.5
            bounds = [(b, 0, T) for b in range(B)] if cu is None else \
                [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
            g = torch.zeros_like(a)
            for b, lo, hi in bounds:
                for c in range(lo, hi, BT):
                    g[b, c:min(c + BT, hi)] = a[b, c:min(c + BT, hi)].cumsum(0)
        else:                # fla's test: 0.1 * randn
            g = 0.1 * torch.randn((B, T, HV), generator=gen)
    return q, k, v, g


RCP_LN2 = 1.4426950216  # fla/ops/utils/constant.py:10


def make_gamma(kind, HV, layer_idx=0, num_layers=12):
    """g_gamma as chunk_simple_gla passes it (scaled by RCP_LN2 in float32): RetNet's
    log(1 - 2^(-5-h)), lightning attention's -(8/H)(1 - layer_idx/num_layers) h, or a given list."""
    h = torch.arange(HV, dtype=torch.float, device=DEV)
    if kind == "retnet":
        gm = (1 - 2.0 ** (-5.0 - h)).log()
    elif kind == "lightning":
        gm = -(8 / HV * (1 - layer_idx / num_layers)) * h
    else:
        return torch.tensor(kind, dtype=torch.float, device=DEV)
    return gm * RCP_LN2


def to_dev(q, k, v, g):
    return (q.to(DEV, torch.bfloat16), k.to(DEV, torch.bfloat16), v.to(DEV, torch.bfloat16),
            None if g is None else g.to(DEV, torch.float32))


# ----------------------------------------------------------------------------------------------
# the launch, through the host and directly into a guarded, prefilled buffer

WARPS = {32: 2, 64: 4, 128: 8}   # the source's list on Hopper: BK = BV and its num_warps


def direct(q, k, v, h, g, scale, svf, cu_seqlens, chunk_indices, o, g_gamma=None, bk=64):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K ** -0.5
    grid = (triton.cdiv(V, bk) * NT, B * HV)
    mod.chunk_fwd_kernel_o[grid](
        q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma, o=o, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=bk, BV=bk,
        USE_G=g is not None, USE_G_GAMMA=g_gamma is not None, IS_VARLEN=cu_seqlens is not None,
        STATE_V_FIRST=svf, num_warps=WARPS[bk], num_stages=3)


# ----------------------------------------------------------------------------------------------
# the claim's chunks, reference and tolerance

def chunks_of(B, T, cu_list, ci_list):
    """(r, b, lo, hi): chunk r's flat tokens [lo, hi) (Claim.lean's rowLo, rowHi)."""
    if cu_list is None:
        NT = triton.cdiv(T, BT)
        return [(b * NT + c, b, b * T + c * BT, min(b * T + c * BT + BT, b * T + T))
                for b in range(B) for c in range(NT)]
    return [(r, 0, cu_list[n] + c * BT, min(cu_list[n] + c * BT + BT, cu_list[n + 1]))
            for r, (n, c) in enumerate(ci_list)]


def tables_ok(T, cu_list, ci_list):
    N = len(cu_list) - 1
    if not (cu_list[0] >= 0 and all(cu_list[i] <= cu_list[i + 1] for i in range(N)) and cu_list[N] <= T):
        return False
    seen = set()
    for n, c in ci_list:
        if not (0 <= n < N and c >= 0 and c * BT < cu_list[n + 1] - cu_list[n]):
            return False
        if (n, c) in seen:
            return False
        seen.add((n, c))
    need = {(n, c) for n in range(N) for c in range(triton.cdiv(cu_list[n + 1] - cu_list[n], BT))}
    return need == seen


def chunk_terms(qf, kf, vf, hf, gf, r, lo, hi, H, HV, K, V, svf, sig, variant=None, gam=None):
    """float64 ref, tol and the InRange mask of chunk r's tokens [lo, hi), all heads and columns.
    qf, kf: [Ttot, H, K]; vf: [Ttot, HV, V]; hf: [R, HV, K, V] (as stored); gf: [Ttot, HV] or None."""
    G = HV // H
    n = hi - lo
    heads = torch.arange(HV, device=DEV) // G
    if variant == "head_mod":
        heads = torch.arange(HV, device=DEV) % H
    Q = qf[lo:hi][:, heads].permute(1, 0, 2).double()          # [HV, n, K]
    Kc = kf[lo:hi][:, heads].permute(1, 0, 2).double()
    Vc = vf[lo:hi].permute(1, 0, 2).double()                    # [HV, n, V]
    rr = r - 1 if variant == "prev_state" and r > 0 else r
    S = hf[rr].double()                                # as stored: [HV, K, V] or [HV, V, K]
    if svf and variant != "untransposed":
        S = S.transpose(-1, -2)
    elif svf and variant == "untransposed":
        S = S.reshape(HV, -1)[:, :K * V].reshape(HV, K, V)
    tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=DEV))
    if variant == "no_causal":
        tri = torch.ones(n, n, dtype=torch.bool, device=DEV)
    base = math.e if variant == "exp_gates" else 2.0
    if gf is not None:
        gc = gf[lo:hi].transpose(0, 1).double()        # [HV, n]
        e = base ** gc
        D = gc[:, :, None] - gc[:, None, :]
        w = torch.where(tri, base ** torch.where(tri, D, torch.full_like(D, -math.inf)), torch.zeros_like(D))
    elif gam is not None:                              # g_gamma: gamma_j * (local position + 1)
        gm = gam.double()[:, None]                     # [HV, 1]
        pos = torch.arange(n, dtype=torch.float64, device=DEV)[None, :]
        if variant == "gamma_global":
            pos = pos + lo
        e = base ** (gm * (pos + (0 if variant == "gamma_noshift" else 1)))
        D = gm[:, :, None] * (pos[:, :, None] - pos[:, None, :])
        w = torch.where(tri, base ** torch.where(tri, D, torch.full_like(D, -math.inf)), torch.zeros_like(D))
    else:
        e = torch.ones((HV, n), dtype=torch.float64, device=DEV)
        w = tri.to(torch.float64).expand(HV, n, n)
    LQ, LK, LV, LS = liftmag(Q), liftmag(Kc), liftmag(Vc), liftmag(S)
    QS = Q @ S
    PQS = LQ @ LS
    zero = torch.zeros((), dtype=torch.float64, device=DEV)
    A = torch.where(tri, Q @ Kc.transpose(-1, -2), zero)          # the sums run over s <= t only
    PA = torch.where(tri, LQ @ LK.transpose(-1, -2), zero)
    trif = tri.to(torch.float64)
    s_eff = sig * sig if variant == "scale_sq" else sig
    ref = s_eff * (e[..., None] * QS + (A * w) @ Vc)
    Mabs = (A.abs() * w) @ Vc.abs()
    Mag = e[..., None] * PQS + (w * PA) @ LV
    Flo = 1 + e[..., None] + PQS + ((1 + w + PA) * trif) @ LV
    c = 2.0 ** -13 if gam is None else (2.0 ** -13 + 2.0 ** -15 * gam.double().abs())[:, None, None]
    tol = 2.0 ** -8 * ref.abs() + (1 + 2.0 ** -8) * sig * (2.0 ** -8 * Mabs + c * Mag
                                                         + 2.0 ** -100 * Flo) + 2.0 ** -130
    # InRange
    fin_q = torch.isfinite(Q).all(-1)                                  # [HV, n]
    fin_h = torch.isfinite(S).all(-2)                                  # [HV, V] (column e over x)
    fin_k_upto = torch.cumprod(torch.isfinite(Kc).all(-1).to(torch.int64), dim=1).bool()  # [HV, n]
    fin_v = torch.isfinite(Vc).all(1)                                  # [HV, V] over the chunk
    ok = fin_q[..., None] & fin_h[:, None, :] & fin_k_upto[..., None] & fin_v[:, None, :]
    if gf is not None:
        gfin = torch.isfinite(gc) & (gc.abs() <= 2.0 ** 64)
        gfin_upto = torch.cumprod(gfin.to(torch.int64), dim=1).bool()
        Dl = torch.where(tri, gc[:, :, None] - gc[:, None, :], torch.full_like(A, -math.inf))
        dok = (Dl <= 64).all(-1)
        ok &= (gfin_upto & (gc <= 64) & dok)[..., None]
    if gam is not None:
        gd = gam.double()
        ok &= (torch.isfinite(gd) & (gd.abs() <= 64) & (gd <= 1))[:, None, None]
    room_pa = torch.where(tri, (PA <= 2.0 ** 100) & (w * PA <= 2.0 ** 100), torch.ones_like(tri)).all(-1)
    ok &= room_pa[..., None]
    ok &= (PQS <= 2.0 ** 100) & (e[..., None] * PQS <= 2.0 ** 100)
    ok &= (Mag + Flo <= 2.0 ** 100) & (sig * (Mag + Flo) <= 2.0 ** 100)
    assert bool(torch.isfinite(ref[ok]).all() and torch.isfinite(tol[ok]).all())
    return ref.permute(1, 0, 2), tol.permute(1, 0, 2), ok.permute(1, 0, 2)   # [n, HV, V]


def check_post(case, o, variant=None, sample=64):
    """Checks Post's accuracy clause on o ([B, T, HV, V], bfloat16); returns (worst err/tol,
    elements checked, in range, violations)."""
    qf, kf, vf, hf, gf = case["flat"]
    H, HV, K, V, svf, sig = case["H"], case["HV"], case["K"], case["V"], case["svf"], case["sig"]
    of = o.reshape(-1, HV, V).double()
    chs = case["chunks"]
    if len(chs) > sample:
        idx = sorted(set([0, len(chs) - 1] + [i for i, c in enumerate(chs) if c[3] - c[2] < BT][:16] +
                         torch.randperm(len(chs), generator=torch.Generator().manual_seed(1))[:sample].tolist()))
        chs = [chs[i] for i in idx]
    worst, nchk, nin, bad = 0.0, 0, 0, 0
    rel = []
    for r, b, lo, hi in chs:
        if hi <= lo:
            continue
        ref, tol, ok = chunk_terms(qf, kf, vf, hf, gf, r, lo, hi, H, HV, K, V, svf, sig, variant,
                                   case.get("gam"))
        c = of[lo:hi]
        nchk += c.numel()
        nin += int(ok.sum())
        err = (c - ref).abs()
        good = torch.isfinite(c) & (err <= tol)
        bad += int((ok & ~good).sum())
        ratio = torch.where(ok, err / tol, torch.zeros_like(err))
        worst = max(worst, float(ratio.max()))
        sel = ok & (ref.abs() > 0)
        if bool(sel.any()):
            rel.append((tol[sel] / ref[sel].abs()).float().cpu())
    case["tol_rel_median"] = float(torch.cat(rel).median()) if rel else float("nan")
    return worst, nchk, nin, bad


# ----------------------------------------------------------------------------------------------
# one launch

def run_case(name, B, T, H, HV, K, V, use_g, svf, cu=None, idx_dtype=torch.int32, scale=None,
             style="gdn", seed=0, permute=False, edit=None, keep=False, gamma=None, bk=64):
    gen = torch.Generator().manual_seed(seed)
    q, k, v, g = make_inputs(B, T, H, HV, K, V, use_g, svf, cu, idx_dtype, gen, style)
    cu_seqlens = None if cu is None else torch.tensor(cu, dtype=idx_dtype, device=DEV)
    chunk_indices = None
    if cu is not None:
        chunk_indices = mod.prepare_chunk_indices(cu_seqlens, BT)
        if permute:
            perm = torch.randperm(len(chunk_indices), generator=gen).to(DEV)
            chunk_indices = chunk_indices[perm].contiguous()
    R = B * triton.cdiv(T, BT) if cu is None else len(chunk_indices)
    hs = (V, K) if svf else (K, V)
    h = 0.1 * torch.randn((B if cu is None else 1, (R // B) if cu is None else R, HV, *hs), generator=gen)
    if edit is not None:
        q, k, v, h, g = edit(q, k, v, h, g)
    q, k, v, g = to_dev(q, k, v, g)
    h = h.to(DEV, torch.bfloat16)
    gg = None if gamma is None else make_gamma(gamma, HV)
    bits = lambda x: x.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    snap = [bits(x).clone() if x is not None else None for x in (q, k, v, h, g, cu_seqlens, chunk_indices, gg)]
    torch.cuda.reset_peak_memory_stats()
    o = mod.chunk_fwd_o(q, k, v, h, g=g, g_gamma=gg, scale=scale, state_v_first=svf,
                        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, BK=bk)
    numel = o.numel()
    big = torch.full((numel + 2 * GUARD,), SENT, dtype=torch.bfloat16, device=DEV)
    od = big[GUARD:GUARD + numel].view(o.shape)
    direct(q, k, v, h, g, scale, svf, cu_seqlens, chunk_indices, od, gg, bk)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    cu_list = None if cu is None else cu_seqlens.tolist()
    ci_list = None if cu is None else chunk_indices.tolist()
    if cu is not None:
        assert tables_ok(T, cu_list, ci_list), f"{name}: tables not TablesOk"
    chunks = chunks_of(B, T, cu_list, ci_list)
    # the written elements
    written = torch.zeros(B * T, dtype=torch.bool, device=DEV)
    for r, b, lo, hi in chunks:
        written[lo:hi] = True
    wmask = written.view(B, T, 1, 1).expand(o.shape)
    assert torch.equal(o.view(torch.int16)[wmask], od.view(torch.int16)[wmask]), f"{name}: host != direct"
    # the frame
    sent = torch.tensor(SENT, dtype=torch.bfloat16)
    assert bool((big[:GUARD] == SENT).all()) and bool((big[GUARD + numel:] == SENT).all()), f"{name}: guard"
    assert bool((od[~wmask] == SENT).all()), f"{name}: unwritten element changed"
    for x, y in zip((q, k, v, h, g, cu_seqlens, chunk_indices, gg), snap):
        if x is not None:
            assert torch.equal(bits(x), y), f"{name}: an input changed"
    sig = (K ** -0.5) if scale is None else scale
    qf = q.reshape(-1, H, K)
    kf = k.reshape(-1, H, K)
    vf = v.reshape(-1, HV, V)
    hf = h.reshape(-1, HV, *hs)
    gf = None if g is None else g.reshape(-1, HV)
    case = dict(flat=(qf, kf, vf, hf, gf), H=H, HV=HV, K=K, V=V, svf=svf, sig=sig, chunks=chunks,
                gam=gg)
    worst, nchk, nin, bad = check_post(case, od)
    status = "ok" if bad == 0 and worst <= 1.0 else "FAIL"
    print(f"{status:4} {name}: BK=BV {bk} B {B} T {T} H/HV {H}/{HV} K/V {K}/{V} g {use_g} "
          f"g_gamma {gamma is not None} svf {svf} "
          f"chunks {len(chunks)}: worst err/tol {worst:.4f}, median tol/|ref| "
          f"{case['tol_rel_median']:.4f}, checked {nchk}, in range {nin}, peak {peak:.2f} GiB")
    assert status == "ok", name
    if keep:
        return case, od
    del q, k, v, h, g, gg, o, od, big, case, snap, qf, kf, vf, hf, gf
    torch.cuda.empty_cache()
    return worst


def negative_tests(bk=64):
    """Outputs of other meanings, rounded to bfloat16, must be rejected."""
    case, od = run_case(f"negative-base-bk{bk}", 1, 700, 4, 8, 128, 128, True, True,
                        cu=[0, 1, 64, 65, 300, 700], style="fla", keep=True, bk=bk)
    qf, kf, vf, hf, gf = case["flat"]
    HV, V = case["HV"], case["V"]
    rejected = []
    for variant in ("no_causal", "exp_gates", "head_mod", "untransposed", "scale_sq", "prev_state"):
        alt = torch.zeros_like(od, dtype=torch.float64).reshape(-1, HV, V)
        for r, b, lo, hi in case["chunks"]:
            ref, _, _ = chunk_terms(qf, kf, vf, hf, gf, r, lo, hi, case["H"], HV, case["K"], V,
                                    case["svf"], case["sig"], variant)
            alt[lo:hi] = ref
        alt = alt.reshape(od.shape).to(torch.bfloat16)
        worst, _, _, bad = check_post(case, alt)
        rejected.append((variant, bad > 0, worst))
    # one element moved by twice its tolerance
    r, b, lo, hi = case["chunks"][3]
    ref, tol, ok = chunk_terms(qf, kf, vf, hf, gf, r, lo, hi, case["H"], HV, case["K"], V, case["svf"], case["sig"])
    moved = od.clone().reshape(-1, HV, V)
    t0 = lo + 5
    moved[t0, 1, 7] = (ref[5, 1, 7] + 2 * tol[5, 1, 7]).to(torch.bfloat16)
    worst, _, _, bad = check_post(case, moved.reshape(od.shape))
    rejected.append(("moved by 2 tol", bad > 0, worst))
    # g_gamma: the gate at the global token position, or without the `+ 1`
    gcase, god = run_case(f"negative-gamma-base-bk{bk}", 1, 300, 4, 4, 64, 64, False, False,
                          cu=[0, 70, 300], gamma="lightning", keep=True, bk=bk)
    gq, gk, gv, gh, gg = gcase["flat"]
    for variant in ("gamma_global", "gamma_noshift"):
        alt = torch.zeros_like(god, dtype=torch.float64).reshape(-1, gcase["HV"], gcase["V"])
        for r, b, lo, hi in gcase["chunks"]:
            ref, _, _ = chunk_terms(gq, gk, gv, gh, gg, r, lo, hi, gcase["H"], gcase["HV"], gcase["K"],
                                    gcase["V"], gcase["svf"], gcase["sig"], variant, gcase["gam"])
            alt[lo:hi] = ref
        worst, _, _, bad = check_post(gcase, alt.reshape(god.shape).to(torch.bfloat16))
        rejected.append((variant, bad > 0, worst))
    for variant, rej, worst in rejected:
        print(f"{'ok' if rej else 'FAIL':4} negative (BK=BV {bk}): {variant}: rejected {rej} (worst err/tol {worst:.1f})")
        assert rej, variant


def grid_limit():
    """The kernel's limit on fixed-length batches: grid axis 1 = B * HV < 65536."""
    for B, expect in ((2047, True), (2048, False)):
        q = torch.randn((B, 1, 16, 16), device=DEV).to(torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn((B, 1, 32, 16), device=DEV).to(torch.bfloat16)
        h = torch.randn((B, 1, 32, 16, 16), device=DEV).to(torch.bfloat16)
        try:
            o = mod.chunk_fwd_o(q, k, v, h)
            torch.cuda.synchronize()
            ran = True
        except RuntimeError as exc:
            ran = False
            msg = str(exc).splitlines()[0]
        print(f"{'ok' if ran == expect else 'FAIL':4} grid limit: B {B} HV 32 (B*HV = {B * 32}): "
              f"{'runs' if ran else 'fails on the host: ' + msg}")
        assert ran == expect


def main():
    torch.manual_seed(0)
    worst = 0.0
    W = lambda *a, **kw: max(worst, run_case(*a, **kw))
    # SETUP.md's launches
    worst = W("setup_test_g", 2, 128, 2, 2, 64, 64, True, False, style="fla")
    worst = W("setup_test_gva", 2, 128, 2, 4, 64, 64, True, False, style="fla")
    worst = W("setup_test_nog", 2, 128, 2, 2, 64, 64, False, False, style="fla")
    worst = W("setup_det_varlen", 1, 8192, 8, 8, 128, 128, True, False, cu=[0, 5000, 8192])
    worst = W("setup_qwen3next", 1, 32768, 16, 32, 128, 128, True, True, cu=list(range(0, 32769, 4096)))
    worst = W("setup_gdn_default", 1, 2000, 6, 6, 256, 512, True, True, cu=[0, 15, 100, 300, 1200, 2000],
              idx_dtype=torch.int64)
    worst = W("setup_bench_h96", 1, 8192, 96, 96, 128, 128, True, False)
    worst = W("setup_bench_h64_nog", 4, 4096, 64, 64, 128, 128, False, False)
    worst = W("setup_bench_d256", 8, 2048, 32, 32, 256, 256, True, False)
    worst = W("setup_dims_60_100", 2, 500, 4, 4, 60, 100, True, False, style="fla")
    worst = W("setup_retnet", 2, 4096, 8, 8, 128, 256, False, False, gamma="retnet")
    worst = W("setup_retnet_varlen", 1, 8192, 8, 8, 128, 256, False, False, cu=[0, 1000, 3000, 8000, 8192],
              idx_dtype=torch.int64, gamma="retnet")
    worst = W("setup_lightning_h96", 1, 8192, 96, 96, 128, 128, False, False, gamma="lightning")
    worst = W("setup_lightning_h16", 4, 2048, 16, 16, 128, 128, False, False, gamma="lightning")
    worst = W("setup_grid_edge", 2047, 1, 16, 32, 128, 128, True, True)
    # edges of Inst.ok
    worst = W("edge_dims_1_1", 2, 130, 1, 1, 1, 1, True, False)
    worst = W("edge_dims_17_200_svf", 1, 300, 2, 6, 17, 200, True, True, cu=[0, 0, 1, 63, 64, 65, 200, 200],
              idx_dtype=torch.int64)
    worst = W("edge_permuted_tables", 1, 1000, 4, 8, 128, 128, True, True, cu=[0, 129, 500, 513, 900],
              permute=True)
    worst = W("edge_unwritten_tail", 1, 1000, 2, 4, 64, 64, True, False, cu=[0, 70, 600])
    worst = W("edge_scale_small", 1, 256, 2, 2, 128, 128, True, False, scale=2.0 ** -64)
    worst = W("edge_scale_large", 1, 256, 2, 2, 128, 128, True, False, scale=2.0 ** 64)
    worst = W("edge_gamma_svf_edges", 1, 400, 2, 4, 64, 64, False, True, cu=[0, 1, 64, 65, 400],
              gamma=[0.0, 1.0, -64.0, -0.015625])
    worst = W("edge_fla_gates", 2, 300, 2, 4, 128, 128, True, True, style="fla")

    def big_gates(q, k, v, h, g):        # gates from -3000 (w, 2^g underflow) to 60 (2^60)
        g = g.clone()
        g[0, :64] = 60.0
        g[0, 64:128] = torch.linspace(0, -3000, 64)[:, None]
        return q, k, v * 2.0 ** -70, h * 2.0 ** -70, g
    worst = W("edge_large_gates", 1, 256, 2, 2, 64, 64, True, False, edit=big_gates)

    def large_vals(q, k, v, h, g):       # |v|, |h| near 2^60, |q|, |k| near 2^10
        return q * 2.0 ** 10, k * 2.0 ** 10, v * 2.0 ** 60, h * 2.0 ** 60, g
    worst = W("edge_large_values", 1, 256, 2, 2, 64, 64, True, False, style="fla", edit=large_vals)

    def subnormals(q, k, v, h, g):       # bfloat16 subnormal values
        v = v.clone(); v[0, ::3] *= 2.0 ** -130
        q = q.clone(); q[0, 1::5] *= 2.0 ** -128
        return q, k, v, h, g
    worst = W("edge_subnormals", 1, 256, 2, 2, 64, 64, True, False, style="fla", edit=subnormals)

    def nonfinite(q, k, v, h, g):        # a NaN key, an inf value, a NaN gate: elements stay in range or not
        k = k.clone(); k[0, 70, 0, 3] = float("nan")
        v = v.clone(); v[0, 200, 1, 5] = float("inf")
        g = g.clone(); g[0, 150, 0] = float("nan")
        return q, k, v, h, g
    worst = W("edge_nonfinite_inputs", 1, 256, 2, 2, 64, 64, True, False, edit=nonfinite)
    # the configurations BK = BV = 32 and 128 (Claim.lean's setup_*_bk32, setup_*_bk128)
    worst = W("setup_test_g_bk32", 2, 128, 2, 2, 64, 64, True, False, style="fla", bk=32)
    worst = W("setup_det_varlen_bk32", 1, 8192, 8, 8, 128, 128, True, False, cu=[0, 5000, 8192], bk=32)
    worst = W("setup_dims_60_100_bk32", 2, 500, 4, 4, 60, 100, True, False, style="fla", bk=32)
    worst = W("setup_bench_h96_bk128", 1, 8192, 96, 96, 128, 128, True, False, bk=128)
    worst = W("setup_bench_h64_nog_bk128", 4, 4096, 64, 64, 128, 128, False, False, bk=128)
    worst = W("setup_bench_d256_bk128", 8, 2048, 32, 32, 256, 256, True, False, bk=128)
    worst = W("setup_qwen3next_bk128", 1, 32768, 16, 32, 128, 128, True, True,
              cu=list(range(0, 32769, 4096)), bk=128)
    worst = W("setup_gdn_default_bk128", 1, 2000, 6, 6, 256, 512, True, True,
              cu=[0, 15, 100, 300, 1200, 2000], idx_dtype=torch.int64, bk=128)
    worst = W("setup_dims_60_100_bk128", 2, 500, 4, 4, 60, 100, True, False, style="fla", bk=128)
    worst = W("setup_retnet_bk128", 2, 4096, 8, 8, 128, 256, False, False, gamma="retnet", bk=128)
    worst = W("setup_lightning_h96_bk128", 1, 8192, 96, 96, 128, 128, False, False, gamma="lightning",
              bk=128)
    for bk in (32, 128):
        worst = W(f"edge_dims_1_1_bk{bk}", 2, 130, 1, 1, 1, 1, True, False, bk=bk)
        worst = W(f"edge_dims_17_200_svf_bk{bk}", 1, 300, 2, 6, 17, 200, True, True,
                  cu=[0, 0, 1, 63, 64, 65, 200, 200], idx_dtype=torch.int64, bk=bk)
        worst = W(f"edge_permuted_tables_bk{bk}", 1, 1000, 4, 8, 128, 128, True, True,
                  cu=[0, 129, 500, 513, 900], permute=True, bk=bk)
        worst = W(f"edge_gamma_svf_edges_bk{bk}", 1, 400, 2, 4, 64, 64, False, True, cu=[0, 1, 64, 65, 400],
                  gamma=[0.0, 1.0, -64.0, -0.015625], bk=bk)
        worst = W(f"edge_large_gates_bk{bk}", 1, 256, 2, 2, 64, 64, True, False, edit=big_gates, bk=bk)
        worst = W(f"edge_large_values_bk{bk}", 1, 256, 2, 2, 64, 64, True, False, style="fla",
                  edit=large_vals, bk=bk)
        worst = W(f"edge_subnormals_bk{bk}", 1, 256, 2, 2, 64, 64, True, False, style="fla",
                  edit=subnormals, bk=bk)
    print(f"worst err/tol over all launches: {worst:.4f}")
    for bk in (32, 64, 128):
        negative_tests(bk)
    grid_limit()
    print("all checks passed")


if __name__ == "__main__":
    main()
