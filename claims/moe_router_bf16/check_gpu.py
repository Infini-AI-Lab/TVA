"""GPU check of the claim `Claims.MoeRouter` (Claim.lean).

Runs the kernel's own host `fused_moe_router_cudacore` (source/
moe_router.py, imported by path) on SETUP.md's launches (the source's eight Grok test cases) and on
launches at the edges of `Inst.ok` and `Pre`, and checks `Pre` and `Post` as the claim states them,
in float64:

* `Pre`: the inputs finite, every `absS <= 2^120`, every `mu + capTol <= 1320`;
* `Post`: `Post` asks for reals `C e` (one per expert) within `capTol` of `capRef(mu)`, of which
  the ids are the first argmaxes and the weights the softmax. The check takes as the witness `C` the
  capped logits computed by a probe kernel, `probe_kernel` below: the source's lines 66-93 copied
  verbatim, storing `logits_softcapped` (launched with the host's `BLOCK_SIZE` and `num_warps`).
  Any `C` meeting every clause shows `Post`; the check verifies that this one does: each `C e`
  within `capTol` of `capRef(mu)`, the kernel's ids the first argmax of `C` and the first argmax of
  `C` without `i0`, the kernel's weights finite and within `wRel * softRef + wFloor`;
* the byte frame: the inputs unchanged, and guard bands of 4096 elements on each side of
  `topk_weights` and `topk_ids` (the host's two `torch.empty` allocations are placed inside
  sentinel-filled buffers during the call) unchanged; every output element is checked.

`mu` is the exact logit computed in float64 from the bfloat16 and float32 inputs (relative error
about 1e-16 of `absS`, negligible against `tolL`). Each line reports the largest error / tolerance
of the capped logits (against `capRef`) and of the two weights (against the softmax of `C`).

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "moe_router.py")
spec = importlib.util.spec_from_file_location("moe_router_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DEV = torch.device("cuda")
U = 2.0 ** -24
ETA = 2.0 ** -150
WREL0, WREL1, WFLOOR = 2.0 ** -14, 2.0 ** -14, 2.0 ** -130
GUARD = 4096                    # guard elements on each side of each output
SENTINEL = 0x7FC0DEAD           # a pattern no store of the kernel writes (a NaN, or an id >= 8)
worst = {"cap": 0.0, "w0": 0.0, "w1": 0.0}
failures = []


@triton.jit
def probe_kernel(input_ptr, moe_router_weight_ptr, out_ptr, num_experts: tl.constexpr,
                 moe_softcapping: tl.constexpr, hidden_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # the source's lines 66-93, verbatim, then a store of logits_softcapped
    pid = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # moe_router_weight is k major
    expert_offsets = tl.arange(0, num_experts)[:, None]
    router_mask = mask[None, :]
    w_router = tl.load(
        moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :],
        mask=router_mask,
        other=0.0,
    )

    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)

    # todo: tl.dot?
    logits = tl.sum((w_router.to(tl.float32) * x[None, :].to(tl.float32)), axis=-1)

    # logit softcap
    if moe_softcapping == 0:
        logits_softcapped = logits
    else:
        logits_scaled = logits / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        top = exped - 1
        bottom = exped + 1
        logits_softcapped = top / bottom * moe_softcapping
    tl.store(out_ptr + pid * num_experts + tl.arange(0, num_experts), logits_softcapped)


def tol_l(B, hidden, S, depth=None):
    """`tolL`: the reduction along a tree of depth `⌈log₂ B⌉` (`SumDepth`); `depth=B-1` gives
    the bound for any tree, which the negative tests report for comparison."""
    D = (B - 1).bit_length() if depth is None else depth      # ⌈log₂ B⌉
    g = (1 + 3 * U) ** D
    return (g - 1) * ((1 + U) * S + hidden * ETA) + 4 * ETA * (B - 1) * g + U * S + hidden * ETA


def cap_tol(B, hidden, mu, S, depth=None):
    return (1 + 2.0 ** -21) * tol_l(B, hidden, S, depth) + 2.0 ** -21 * mu.abs() + 2.0 ** -14


def soft_range(r, t, j):
    """The range of `softRef C i0 j = exp(C j) / Σ_k exp(C k)` over every `C` with
    `|C k − r k| ≤ t k` (per token; `j` the expert of each token): increasing in `C j`, decreasing
    in the others."""
    lo_c, hi_c = r - t, r + t
    m = hi_c.max(1, keepdim=True).values
    el, eh = torch.exp(lo_c - m), torch.exp(hi_c - m)
    elj, ehj = el.gather(1, j[:, None])[:, 0], eh.gather(1, j[:, None])[:, 0]
    hi = ehj / (el.sum(1) - elj + ehj)
    lo = elj / (eh.sum(1) - ehj + elj)
    return lo, hi


def negatives(name, x, w):
    """Wrong results `Post` must reject (it is a statement about every token, so one rejected token
    rejects the run), and the size of the tolerances, at the depth `⌈log₂ B⌉` (`SumDepth`) and, for
    comparison, along any tree (depth `B − 1`):
    * a kernel without the soft cap: the ids and weights of the raw logits (the ids are the same, the
      weights the softmax of `mu`); rejected on a token when a weight lies outside the range of
      `softRef` over every `C` within `capTol` (widened by `wRel` and `wFloor`);
    * swapped ids `(i1, i0)`: rejected on a token when no `C` within `capTol` puts `i1` first,
      `capRef(i0) − capTol(i0) > capRef(i1) + capTol(i1)`."""
    bs, hidden = x.shape
    B = triton.next_power_of_2(hidden)
    xd, wd = x.double(), w.double()
    mu = xd @ wd.T
    S = xd.abs() @ wd.abs().T
    ref = 30 * torch.tanh(mu / 30)
    top = ref.topk(2, dim=1).indices
    i0, i1 = top[:, 0], top[:, 1]
    raw = torch.softmax(mu, dim=1)
    out = []
    for label, depth in (("depth log2 B", None), ("any tree", B - 1)):
        ct = cap_tol(B, hidden, mu, S, depth)
        rel = (ct / ref.abs()).flatten()
        lo0, hi0 = soft_range(ref, ct, i0)
        lo1, hi1 = soft_range(ref, ct, i1)
        a0 = raw.gather(1, i0[:, None])[:, 0]
        a1 = raw.gather(1, i1[:, None])[:, 0]
        nocap = ((a0 < lo0 * (1 - WREL0) - WFLOOR) | (a0 > hi0 * (1 + WREL0) + WFLOOR) |
                 (a1 < lo1 * (1 - WREL1) - WFLOOR) | (a1 > hi1 * (1 + WREL1) + WFLOOR))
        r0, r1 = ref.gather(1, i0[:, None])[:, 0], ref.gather(1, i1[:, None])[:, 0]
        t0, t1 = ct.gather(1, i0[:, None])[:, 0], ct.gather(1, i1[:, None])[:, 0]
        swap = r0 - t0 > r1 + t1
        wtol = torch.exp(2 * ct.max(1).values) - 1 + WREL1
        out.append((label, nocap, swap))
        print(f"    {name}, {label}: capTol/|capRef| median {rel.median().item():.3g} "
              f"p90 {rel.quantile(0.9).item():.3g}; composed weight tolerance (relative) median "
              f"{wtol.median().item():.3g}; tokens rejecting a kernel without the soft cap "
              f"{nocap.double().mean().item():.3f}, rejecting swapped ids {swap.double().mean().item():.3f}")
    ok = bool(out[0][1].any()) and bool(out[0][2].any())
    if not ok:
        failures.append(name + " negatives")
    return ok


def first_argmax(v, excl=None):
    best = None
    for e in range(len(v)):
        if e == excl:
            continue
        if best is None or v[e] > v[best]:
            best = e
    return best


def check(name, x, w):
    bs, hidden = x.shape
    B = triton.next_power_of_2(hidden)
    xs, ws = x.clone(), w.clone()
    # the host allocates topk_weights and topk_ids with torch.empty (lines 167-168): during the
    # call each allocation is placed inside a larger buffer whose guard bands (GUARD elements on
    # each side) hold a sentinel, so that Post's byte frame is checked around the outputs too
    orig_empty = torch.empty
    bufs = []

    def empty_guarded(shape, device=None, dtype=None):
        n = math.prod(shape)
        big = orig_empty((n + 2 * GUARD,), device=device, dtype=dtype)
        big.view(torch.int32).fill_(SENTINEL)
        bufs.append(big)
        return big[GUARD:GUARD + n].view(shape)

    torch.empty = empty_guarded
    try:
        weights, ids = mod.fused_moe_router_cudacore(x, w, 2, 30.0, None)
    finally:
        torch.empty = orig_empty
    torch.cuda.synchronize()
    frame_ok = torch.equal(x, xs) and torch.equal(w, ws) and len(bufs) == 2
    for big in bufs:
        g = big.view(torch.int32)
        frame_ok = frame_ok and bool((g[:GUARD] == SENTINEL).all()) and bool((g[-GUARD:] == SENTINEL).all())
    # the witness C: the probe's capped logits (the host's BLOCK_SIZE and num_warps)
    C = torch.empty((bs, 8), dtype=torch.float32, device=DEV)
    nw = max(min(triton.next_power_of_2(triton.cdiv(hidden, 256)), 32), 4)
    probe_kernel[(bs,)](x, w, C, num_experts=8, moe_softcapping=30.0, hidden_dim=hidden, BLOCK_SIZE=B,
                        num_warps=nw)
    torch.cuda.synchronize()
    # Pre and the reference, in float64
    wd = w.double()
    rows = max(1, 2 ** 25 // hidden)
    mu = torch.empty((bs, 8), dtype=torch.float64, device=DEV)
    S = torch.empty((bs, 8), dtype=torch.float64, device=DEV)
    xfin = True
    for r in range(0, bs, rows):
        xd = x[r:r + rows].double()
        xfin = xfin and bool(torch.isfinite(xd).all())
        mu[r:r + rows] = xd @ wd.T                   # the exact logits (float64)
        S[r:r + rows] = xd.abs() @ wd.abs().T
        del xd
    ct = cap_tol(B, hidden, mu, S)
    pre_ok = xfin and bool(torch.isfinite(wd).all())
    pre_ok = pre_ok and bool((S <= 2.0 ** 120).all()) and bool((mu + ct <= 1320).all())
    ref = 30 * torch.tanh(mu / 30)
    Cd = C.double()
    c_ok = bool(torch.isfinite(Cd).all())
    e_cap = ((Cd - ref).abs() / ct).max().item()
    # the ids: the first argmaxes of C (on the CPU, exactly)
    Cl = Cd.cpu().tolist()
    il = ids.long().cpu().tolist()
    ids_ok = True
    for t in range(bs):
        i0 = first_argmax(Cl[t])
        i1 = first_argmax(Cl[t], excl=i0)
        if il[t] != [i0, i1]:
            ids_ok = False
            break
    # the weights against the softmax of C, normalised at i0
    i0 = ids[:, 0].long()
    i1 = ids[:, 1].long()
    c0 = Cd.gather(1, i0[:, None])
    den = torch.exp(Cd - c0).sum(-1)
    s0 = 1.0 / den
    s1 = torch.exp(Cd.gather(1, i1[:, None])[:, 0] - c0[:, 0]) / den
    wv = weights.double()
    w_ok = bool(torch.isfinite(wv).all())
    e0 = ((wv[:, 0] - s0).abs() / (WREL0 * s0 + WFLOOR)).max().item()
    e1 = ((wv[:, 1] - s1).abs() / (WREL1 * s1 + WFLOOR)).max().item()
    worst["cap"] = max(worst["cap"], e_cap)
    worst["w0"] = max(worst["w0"], e0)
    worst["w1"] = max(worst["w1"], e1)
    good = pre_ok and c_ok and ids_ok and w_ok and frame_ok and e_cap <= 1 and e0 <= 1 and e1 <= 1
    print(f"{'ok ' if good else 'BAD'} {name}: bs {bs} hidden {hidden} BLOCK {B}: err/tol capped logits "
          f"{e_cap:.2e}, weight0 {e0:.3f}, weight1 {e1:.3f}; ids {ids_ok}, Pre {pre_ok}, frame {frame_ok}; "
          f"max S {S.max().item():.3g}, max |mu| {mu.abs().max().item():.3g}, max capTol {ct.max().item():.3g}")
    if not good:
        failures.append(name)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)

    def grok_like(bs, hidden):
        # the source's generator (lines 219-222)
        x = (torch.randn((bs, hidden), generator=gen) / 16).to(device=DEV, dtype=torch.bfloat16)
        w = torch.randn((8, hidden), generator=gen) / 16
        w[:, 0] += torch.linspace(-0.03, 0.03, 8)
        return x, w.to(DEV)

    # SETUP.md's launches: the source's Grok cases (line 215)
    for bs in (1, 7, 128, 511):
        for h in (6144, 8192):
            check(f"setup_{bs}_{h}", *grok_like(bs, h))
    # edges of Inst.ok: other hidden sizes and block sizes, many tokens
    for bs, h in [(17, 1), (17, 3), (17, 5), (128, 64), (128, 257), (33, 4097), (9, 8191), (2048, 8192),
                  (32768, 8192), (262144, 1024)]:
        check(f"hidden {h}, bs {bs}", *grok_like(bs, h))
    # edges of Pre and Post
    x = torch.zeros((5, 6144), dtype=torch.bfloat16, device=DEV)
    w = torch.zeros((8, 6144), device=DEV)
    check("zero inputs (all ties)", x, w)
    x, w = grok_like(64, 6144)
    w[3] = w[0]
    w[0] += 0.05 * x.float().mean(0)
    w[3] = w[0]
    check("tied rows 0 and 3", x, w)
    x = torch.ones((4, 8192), dtype=torch.bfloat16, device=DEV)
    w = torch.full((8, 8192), 0.1, device=DEV)
    w[2] = 1300.0 / 8192
    w[5] = -12.0
    check("logit near 1300, and near -98000", x, w)
    x = torch.full((3, 4096), 2.0 ** -126, dtype=torch.bfloat16, device=DEV)
    w = torch.randn((8, 4096), generator=gen).to(DEV) * 2.0 ** 120
    check("subnormal-scale x, huge w", x, w)
    x, w = grok_like(256, 8192)
    check("x * 64 (Grok-scale hidden states)", (x.float() * 64).to(torch.bfloat16), w)
    # negative tests at Grok's shapes: the source's data, and logits of standard deviation 3
    print("negative tests and the size of the tolerances:")
    for h in (6144, 8192):
        x, w = grok_like(511, h)
        check(f"source data bs 511 hidden {h}", x, w)
        negatives(f"source data, hidden {h}", x, w)
        x = torch.randn((511, h), generator=gen).to(device=DEV, dtype=torch.bfloat16)
        w = (torch.randn((8, h), generator=gen) * (3 / h ** 0.5)).to(DEV)
        check(f"logits std 3, bs 511 hidden {h}", x, w)
        negatives(f"logits std 3, hidden {h}", x, w)
    print(f"worst err/tol: capped logits {worst['cap']:.2e}, weight0 {worst['w0']:.3f}, weight1 {worst['w1']:.3f}")
    if failures:
        print("FAILED:", ", ".join(failures))
        raise SystemExit(1)
    print("all checks hold")
