"""GPU check of the claim of `extend_attn_bf16`.

Runs the kernel's own host (`source/extend_attention.py`, imported by path) on the code base's
test shapes, on SETUP.md's model shapes (extend batches and target verifications, with and without
a sliding window) and on launches at the edges of `Inst.ok` and `Pre`; computes `Post`'s reference
and tolerance from the inputs in float64 exactly as `Claim.lean` states them (`keyK`/`keyV`
through `kv_indices`, `score`, `scoreMag` with `liftMag`, `attends` with `maskBit`/`maskIdx`/`wOff`
and the window, `keys`, `Xrow`, `attnRef`, `attnAbs`, `vSum`, `trips`, `dlt`, `rho`, `gam`, `errE`,
`tolRel`, `tolAbs`);
and reports the largest error / tolerance over every written element, that every written element
is finite, and the frame (every input unchanged; every element of `o_extend` outside the requests'
new tokens keeps its initial bits: the host's `torch.empty_like` is made to fill it with a
sentinel pattern). It also reports tol / |attnRef| (median and largest) on the model shapes.
Negative tests: outputs that are wrong (no prefix, not causal, heads swapped, one element off, a
NaN, a write outside the requests; at the longest requests, the all-zero output and the output
without the prefix's first quarter; with a window, the output without it; with the custom mask,
the causal output, the output without the prefix, and with a window too, the output without the
window or without the window offset) must be rejected by the same checks.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "extend_attention.py")
spec = importlib.util.spec_from_file_location("extend_attention_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
BN = 64
SENTINEL = -12345  # an int16 pattern (a finite bfloat16) the fresh `o_extend` starts with


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def block_m(D):
    return 128 if D <= 128 else 64


def tol_rel(BD, Tp, Te, X):
    """`tolRel BD Tp Te X` of Claim.lean, in float64 (Tp a number, Te, X tensors)."""
    T = Tp + Te
    dlt = (BD + 8) * 2.0 ** -22 * X + 2.0 ** -40
    rho = torch.expm1(2.0 ** -8 + (T + 8) * 2.0 ** -20 + (X + dlt) * 2.0 ** -17)
    gam = torch.expm1((66 + 2 * Tp + 72 * Te) * 2.0 ** -22)
    e2 = torch.exp(2 * dlt)
    errE = e2 * (rho + gam * (1 + rho)) / (1 - rho) + torch.expm1(2 * dlt)
    return 2.0 ** -8 + (1 + 2.0 ** -8) * errE


def lift(x):
    """`Device.liftMag .bf16`: 0 for 0, else max(|x|, 2^-126)."""
    a = x.abs()
    return torch.where(x == 0, torch.zeros_like(a), torch.clamp(a, min=2.0 ** -126))


def requests(inp):
    """(q0, ext, k0, pre) per request, read from the index tensors (`qsOf`, `extOf`, `ksOf`,
    `preOf`)."""
    qo, kvp = inp["qo"].tolist(), inp["kvp"].tolist()
    return [(qo[b], qo[b + 1] - qo[b], kvp[b], kvp[b + 1] - kvp[b]) for b in range(len(qo) - 1)]


def keys_of(inp, b_req, g):
    """keyK, keyV of request `b_req` for KV head `g`: the prefix slots, then the new tokens."""
    q0, n, k0, pre = b_req
    slots = inp["kvi"][k0:k0 + pre]
    K = torch.cat([inp["kb"][slots, g, :], inp["k"][q0:q0 + n, g, :]]).double()
    V = torch.cat([inp["vb"][slots, g, :], inp["v"][q0:q0 + n, g, :]]).double()
    return K, V


def allowed_rows(inp, bi, b_req, i, use_prefix=True, causal=True, window=True, use_mask=True,
                 offset=True):
    """`attends` of Claim.lean for the rows `i` of request `bi` over its `pre + n` positions (the
    prefix first): the prefix, then the new tokens up to the row (causal) or as its row of the
    custom mask allows (`maskBit` at `maskIdx`, with `wOff`), and with a window `W > 0` only
    positions `p` with `pre + i <= p + W`. The keyword flags give the wrong references of the
    negative tests (not Post's): no prefix, not causal, no window, the causal mask in place of the
    custom mask, the mask read without the window offset."""
    q0, n, k0, pre = b_req
    npre = pre if use_prefix else 0
    j = torch.arange(npre + n, device=DEV)
    row = npre + i
    sw = inp["sw"]
    if inp["mask"] is not None and use_mask:
        off = int(inp["wko"][bi]) if sw > 0 else 0
        off_read = off if offset else 0
        m0 = int(inp["mip"][bi])
        L = pre + n + off
        idx = m0 + i[:, None] * L + off_read + pre + torch.arange(n, device=DEV)[None, :]
        new = inp["mask"][idx.clamp(max=inp["mask"].numel() - 1)]
        allowed = torch.cat([torch.ones(len(i), npre, dtype=torch.bool, device=DEV), new], 1)
    elif causal:
        allowed = j[None, :] <= row[:, None]
    else:
        allowed = torch.ones(len(i), npre + n, dtype=torch.bool, device=DEV)
    if window and sw > 0:
        allowed &= row[:, None] <= j[None, :] + sw
    return allowed


def post_ratio(inp, o, rows=256, report=None):
    """The largest |o - attnRef| / (tolRel·attnAbs + tolAbs) over every written element, and
    whether every written element is finite; `report` (a list) collects tol / |attnRef|."""
    q = inp["q"]
    H, HKV, D = q.shape[1], inp["k"].shape[1], q.shape[2]
    group = H // HKV
    sigma = inp["sigma"]
    BD, BM = next_pow2(D), block_m(D)
    worst, at, finite, tolrel_max = 0.0, None, True, 0.0
    for bi, b_req in enumerate(requests(inp)):
        q0, n, k0, pre = b_req
        if n == 0:
            continue
        for h in range(H):
            K, V = keys_of(inp, b_req, h // group)
            lk = lift(K)
            out = o[q0:q0 + n, h, :]
            finite = finite and bool(torch.isfinite(out.float()).all())
            qh = q[q0:q0 + n, h, :].double()
            lq = lift(qh)
            for r0 in range(0, n, rows):
                r1 = min(n, r0 + rows)
                i = torch.arange(r0, r1, device=DEV)
                allowed = allowed_rows(inp, bi, b_req, i)
                s = sigma * (qh[r0:r1] @ K.T)
                mag = sigma * (lq[r0:r1] @ lk.T)
                s = torch.where(allowed, s, torch.full_like(s, -math.inf))
                # Xrow: over keys and the row's own position pre + i
                own = torch.zeros_like(allowed)
                own[torch.arange(r1 - r0, device=DEV), pre + i] = True
                mag = torch.where(allowed | own, mag, torch.full_like(mag, -math.inf))
                X = mag.max(dim=1).values
                w = torch.softmax(s, dim=1)
                ref = w @ V
                absw = w @ V.abs()
                vsum = allowed.double() @ V.abs()
                Tp = (pre + BN - 1) // BN  # tripsPre
                Te = ((torch.clamp((i // BM + 1) * BM, max=n) + BN - 1) // BN).double()  # tripsExt
                T = Tp + Te
                tr = tol_rel(BD, Tp, Te, X)
                tol = tr[:, None] * absw + 2.0 ** -80 * (1 + vsum)
                err = (out[r0:r1].double() - ref).abs()
                ratio = err / tol
                if report is not None:
                    report.append((tol / ref.abs()).flatten().float().cpu())
                r = ratio.max().item()
                if not math.isfinite(r):
                    r = math.inf
                tolrel_max = max(tolrel_max, float(tr.max()))
                if r > worst:
                    worst = r
                    ii = int(ratio.max(dim=1).values.argmax())
                    e = int(ratio[ii].argmax())
                    at = (float(err[ii, e]), float(tol[ii, e]), float(X[ii]), int(T[ii]),
                          float(absw[ii, e]))
    return worst, finite, at, tolrel_max


def frame_ok(inp, o, snaps):
    ok = all(torch.equal(a, inp[nm]) for nm, a in snaps.items())
    covered = torch.zeros(o.shape[0], dtype=torch.bool, device=DEV)
    for q0, n, _, _ in requests(inp):
        covered[q0:q0 + n] = True
    ok_out = bool((o.view(torch.int16)[~covered] == SENTINEL).all())
    return ok, ok_out


def make_case(H, HKV, D, pre, ext, gen, extra_slots=37, lead=2, trail=3, scale_qk=1.0,
              v_scale=1.0, mean=0.0, std=1.0, fused=False, uniform_v=None, sm_scale=None,
              max_ext=None, sw=-1, verify=False, seq=None):
    """One host launch: requests with prefix lengths `pre` and new-token lengths `ext`; the
    prefix slots a random permutation of a cache with `extra_slots` unused slots; `lead` and
    `trail` new-token rows in no request (before the first and after the last). `sw` the window
    (`-1`: none). `verify`: a target verification: `int32` `qo_indptr` and a random tree mask
    (each new token sees itself and a random chain of earlier ones), request `b`'s block of
    `ext_b` rows of `seq_b + ext_b` (`seq_b >= pre_b` the full prefix; with a window the window
    prefix is its last `pre_b` positions, `window_kv_offsets[b] = seq_b - pre_b`)."""
    B = len(pre)
    S = sum(pre) + extra_slots
    T = lead + sum(ext) + trail
    qo = torch.tensor([lead + sum(ext[:b]) for b in range(B + 1)],
                      dtype=torch.int32 if verify else torch.int64)
    kvp = torch.tensor([sum(pre[:b]) for b in range(B + 1)], dtype=torch.int32)
    kvi = torch.randperm(S, generator=gen)[: sum(pre)].to(torch.int64)

    def rnd(*shape, scale=1.0):
        return ((torch.randn(shape, generator=gen) * std + mean) * scale).to(DEV, torch.bfloat16)

    q = rnd(T, H, D, scale=scale_qk)
    k = rnd(T, HKV, D, scale=scale_qk)
    v = rnd(T, HKV, D, scale=v_scale)
    kb = rnd(S, HKV, D, scale=scale_qk)
    vb = rnd(S, HKV, D, scale=v_scale)
    if fused:
        # SETUP.md's strided q: a view of a fused [T, (H + 2·HKV)·D] projection
        qkv = rnd(T, (H + 2 * HKV) * D)
        q = qkv[:, :H * D].view(T, H, D)
    if uniform_v is not None:
        # q = 0: every score 0, uniform weights; v constant along the tokens, one value per
        # component: the bfloat16 roundings of the weights all push the same way
        q = torch.zeros_like(q)
        c = torch.tensor(uniform_v, dtype=torch.float32)[:D].to(torch.bfloat16).to(DEV)
        v = c[None, None, :].expand(T, HKV, D).contiguous()
        vb = c[None, None, :].expand(S, HKV, D).contiguous()
    inp = dict(q=q, k=k, v=v, kb=kb, vb=vb, qo=qo.to(DEV), kvp=kvp.to(DEV), kvi=kvi.to(DEV))
    inp["sw"] = sw
    inp["mask"] = inp["mip"] = inp["wko"] = None
    seq = list(pre) if seq is None else list(seq)
    if sw > 0:
        inp["wko"] = torch.tensor([sq - p for sq, p in zip(seq, pre)], dtype=torch.int64).to(DEV)
    if verify:
        blocks = []
        for b in range(B):
            n = ext[b]
            parent = [-1] + [int(torch.randint(0, t, (1,), generator=gen)) for t in range(1, n)]
            tree = torch.zeros(n, n, dtype=torch.bool)
            for t in range(n):
                a = t
                while a >= 0:
                    tree[t, a] = True
                    a = parent[a]
            blocks.append(torch.cat([torch.ones(n, seq[b], dtype=torch.bool), tree], 1).flatten())
        lens = [len(x) for x in blocks]
        inp["mask"] = torch.cat(blocks + [torch.zeros(5, dtype=torch.bool)]).to(DEV)
        inp["mip"] = torch.tensor([sum(lens[:b]) for b in range(B + 1)], dtype=torch.int64).to(DEV)
    inp["max_ext"] = max(ext) if max_ext is None else max_ext
    inp["sm_scale"] = sm_scale
    inp["sigma"] = (1.0 / (D ** 0.5)) if sm_scale is None else sm_scale
    return inp


def run_host(inp):
    real_empty_like = torch.empty_like

    def sentinel_empty_like(x, *a, **kw):
        y = real_empty_like(x, *a, **kw)
        y.view(torch.int16).fill_(SENTINEL)
        return y

    torch.empty_like = sentinel_empty_like
    try:
        o = mod.extend_attention_fwd(inp["q"], inp["k"], inp["v"], inp["kb"], inp["vb"],
                                     inp["qo"], inp["kvp"], inp["kvi"], inp["max_ext"],
                                     sm_scale=inp["sm_scale"], custom_mask=inp["mask"],
                                     mask_indptr=inp["mip"], sliding_window_size=inp["sw"],
                                     window_kv_offsets=inp["wko"])
    finally:
        torch.empty_like = real_empty_like
    torch.cuda.synchronize()
    return o


TENSORS = ("q", "k", "v", "kb", "vb", "qo", "kvp", "kvi", "mask", "mip", "wko")


def snapshot(inp):
    return {nm: inp[nm].clone() for nm in TENSORS if inp[nm] is not None}


def run_case(name, inp, W, rows=256, report=False):
    snaps = snapshot(inp)
    o = run_host(inp)
    ok_in, ok_out = frame_ok(inp, o, snaps)
    rep = [] if report else None
    r, finite, at, trm = post_ratio(inp, o, rows=rows, report=rep)
    ok = ok_in and ok_out and finite and r < 1
    W["tolrel_max"] = max(W["tolrel_max"], trm)
    if r > W["out"]:
        W["out"], W["at"], W["case"] = r, at, name
    W["ok"] = W["ok"] and ok
    reqs = requests(inp)
    q = inp["q"]
    mode = (f"  window {inp['sw']}" if inp["sw"] > 0 else "") + \
        ("  custom mask" if inp["mask"] is not None else "")
    print(f"{name}: heads {q.shape[1]}/{inp['k'].shape[1]} D {q.shape[2]} requests {len(reqs)} "
          f"max prefix {max(p for *_, p in reqs)} max extend {max(n for _, n, _, _ in reqs)}: "
          f"err/tol {r:.3e}{'' if finite else '  NOT FINITE'}{'' if ok_in else '  INPUT CHANGED'}"
          f"{'' if ok_out else '  OUTSIDE WRITTEN'}{'' if ok else '  FAILED'}{mode}")
    if report:
        t = torch.cat(rep)
        t = t[torch.isfinite(t)]
        print(f"  tol/|attnRef|: median {t.median().item():.3e}, largest {t.max().item():.3e}")
    return o


def test_shape(gen, D, B=19, N_CTX=12331):
    """test_triton_attention_kernels.py:172-243: lengths drawn from [1, N_CTX // 2), values
    N(0.1, 0.2), the backend's index dtypes; each request's prefix slots consecutive in the
    cache (b_start_loc), its new tokens also in the cache after them."""
    pre = torch.randint(1, N_CTX // 2, (B,), generator=gen).tolist()
    ext = torch.randint(1, N_CTX // 2, (B,), generator=gen).tolist()
    inp = make_case(12, 4, D, pre, ext, gen, extra_slots=sum(ext), lead=0, trail=0, mean=0.1,
                    std=0.2)
    start = [sum(pre[:b]) + sum(ext[:b]) for b in range(B)]
    inp["kvi"] = torch.cat([torch.arange(s, s + p) for s, p in zip(start, pre)]).to(DEV)
    # the new tokens' keys and values are the cache's slots after each prefix (the test copies
    # them from k_buffer, v_buffer)
    q0 = 0
    for s, p, n in zip(start, pre, ext):
        inp["k"][q0:q0 + n] = inp["kb"][s + p:s + p + n]
        inp["v"][q0:q0 + n] = inp["vb"][s + p:s + p + n]
        q0 += n
    return inp


def adversarial(gen):
    """Every bfloat16 rounding pushes the same way: one prefix key with score 0 (p = 1) and 4299
    keys with p just above the midpoint between the bfloat16 values 0.5 + 2^-9 and 0.5 + 2^-8
    (score ln(0.5 + 2^-9 + 2^-20), from q = e_0, k = -8·e_0 and a chosen sm_scale), so every
    p.to(bfloat16) rounds up by almost 2^-8; v constant 1 + 2^-7, so acc / deno lands just above
    a bfloat16 midpoint and the store rounds up too: the output is 1 + 2^-6, one bfloat16 step
    above the exact attention 1 + 2^-7."""
    D = 128
    sigma = -math.log(0.5 + 2.0 ** -9 + 2.0 ** -20) / 8
    inp = make_case(1, 1, D, [4000], [300], gen, sm_scale=sigma)
    e0 = torch.zeros(D, device=DEV, dtype=torch.bfloat16)
    e0[0] = 1
    inp["q"].copy_(e0.expand_as(inp["q"]))
    inp["k"].copy_((-8 * e0).expand_as(inp["k"]))
    inp["kb"].copy_((-8 * e0).expand_as(inp["kb"]))
    inp["kb"][inp["kvi"][0]] = 0  # the prefix's first key: score 0
    c = 1 + 2.0 ** -7
    inp["v"].fill_(c)
    inp["vb"].fill_(c)
    return inp


def ref_output(inp, drop_frac=0.0, rows=128):
    """A wrong output for the negative tests: each new token's attention over its request's
    positions without the first `drop_frac` of the prefix, in float64, rounded to bfloat16."""
    o = torch.zeros(inp["q"].shape, dtype=torch.bfloat16, device=DEV)
    H, HKV = inp["q"].shape[1], inp["k"].shape[1]
    for b_req in requests(inp):
        q0, n, k0, pre = b_req
        cut = int(pre * drop_frac)
        for h in range(H):
            K, V = keys_of(inp, b_req, h // (H // HKV))
            K, V = K[cut:], V[cut:]
            for r0 in range(0, n, rows):
                r1 = min(n, r0 + rows)
                i = torch.arange(r0, r1, device=DEV)
                j = torch.arange(K.shape[0], device=DEV)
                s = inp["sigma"] * (inp["q"][q0 + r0:q0 + r1, h].double() @ K.T)
                s = s.masked_fill(j[None, :] > (pre - cut + i)[:, None], -math.inf)
                o[q0 + r0:q0 + r1, h] = (torch.softmax(s, 1) @ V).to(torch.bfloat16)
    return o


def long_case(name, gen, pre, ext, W, quarter_required=True, shift_first_quarter=0.0, H=2, HKV=1):
    """The longest requests: the kernel's output must meet Post, and an all-zero output and the
    output without the first quarter of the prefix must be rejected. With `randn` values, dropping a
    quarter of the prefix changes the exact attention by about `attnAbs/sqrt(n)`, which at 2^18
    positions is below the bound (`quarter_required=False`: reported, not required);
    `shift_first_quarter` adds a constant to the values of the prefix's first quarter (a context
    whose early part differs), where dropping it must be rejected."""
    inp = make_case(H, HKV, 128, [pre], [ext], gen)
    if shift_first_quarter:
        first = inp["kvi"][: pre // 4]
        inp["vb"][first] = (inp["vb"][first].float() + shift_first_quarter).to(torch.bfloat16)
    run_case(name, inp, W, rows=128)
    fails = []
    for nm, bad, required in (
            ("all-zero", torch.zeros(inp["q"].shape, dtype=torch.bfloat16, device=DEV), True),
            ("quarter-prefix-dropped", ref_output(inp, drop_frac=0.25), quarter_required)):
        r, finite, _, _ = post_ratio(inp, bad, rows=128)
        rejected = (r >= 1) or not finite
        verdict = "rejected" if rejected else ("NOT REJECTED" if required else
                                               "within the bound (not required)")
        print(f"  negative {name} {nm}: err/tol {r:.3e}: {verdict}")
        if required and not rejected:
            fails.append(f"{name} {nm}")
    return fails


def adversarial_small(gen, D=16, H=8):
    """The same pushes with the smallest score bound and one trip: 4 requests of 64 new tokens, no
    prefix; each request's first key has score `+0.345` and `v = 0`, every other key score
    `-0.345` (a gap of `ln 2`: `p = 0.5 + 2^-9 + 2^-22`, just above a bfloat16 midpoint) and values
    in `[1, 2)`, so every weighted value rounds up, then the store."""
    sigma = -math.log(0.5 + 2.0 ** -9 + 2.0 ** -22) / 8
    inp = make_case(H, H, D, [0] * 4, [64] * 4, gen, sm_scale=sigma, lead=0, trail=0,
                    extra_slots=0)
    e0 = torch.zeros(D, device=DEV, dtype=torch.bfloat16)
    e0[0] = 1
    inp["q"].copy_(e0.expand_as(inp["q"]))
    inp["k"].copy_((-4 * e0).expand_as(inp["k"]))
    cvals = torch.tensor([1 + i / 128 for i in range(128)])
    for bi, (q0, n, _, _) in enumerate(requests(inp)):
        idx = torch.arange(H * D) + bi * 7
        inp["v"][q0:q0 + n] = cvals[idx % 128].view(H, D).to(DEV, torch.bfloat16)
        inp["k"][q0] = 4 * e0
        inp["v"][q0] = 0
    return inp


def wrong_attention(inp, o, **kw):
    """The exact attention under `allowed_rows(..., **kw)` (a wrong question), rounded to
    bfloat16, in place of `o`'s requests."""
    bad = o.clone()
    H, HKV = o.shape[1], inp["k"].shape[1]
    for bi, b_req in enumerate(requests(inp)):
        q0, n, k0, pre = b_req
        if n == 0:
            continue
        i = torch.arange(n, device=DEV)
        allowed = allowed_rows(inp, bi, b_req, i, **kw)
        for h in range(H):
            K, V = keys_of(inp, b_req, h // (H // HKV))
            if not kw.get("use_prefix", True):
                K, V = K[pre:], V[pre:]
            s = inp["sigma"] * (inp["q"][q0:q0 + n, h].double() @ K.T)
            s = s.masked_fill(~allowed, -math.inf)
            bad[q0:q0 + n, h] = (torch.softmax(s, 1) @ V).to(torch.bfloat16)
    return bad


def mode_negatives(name, inp, o, tests):
    """Wrong outputs of a window or custom-mask launch (`tests`: name -> allowed_rows flags)."""
    fails = []
    for nm, kw in tests:
        r, finite, _, _ = post_ratio(inp, wrong_attention(inp, o, **kw))
        rejected = (r >= 1) or not finite
        print(f"  negative {name} {nm}: err/tol {r:.3e}: {'rejected' if rejected else 'NOT REJECTED'}")
        if not rejected:
            fails.append(f"{name} {nm}")
    return fails


def negatives(gen):
    """Wrong outputs the checks must reject."""
    inp = make_case(8, 2, 128, [300, 0, 64], [200, 129, 1], gen)
    o = run_host(inp)
    good, _, _, _ = post_ratio(inp, o)
    fails = []

    def wrong(name, o_bad):
        r, finite, _, _ = post_ratio(inp, o_bad)
        rejected = (r >= 1) or not finite
        print(f"  negative {name}: err/tol {r:.3e}{'' if finite else ' (not finite)'}: "
              f"{'rejected' if rejected else 'NOT REJECTED'}")
        if not rejected:
            fails.append(name)

    # the right answer to the wrong question, judged against Post
    for nm, kw in (("no-prefix", dict(use_prefix=False)), ("not-causal", dict(causal=False))):
        wrong(nm, wrong_attention(inp, o, **kw))
    sw = o.clone()
    sw[:, [0, 1]] = o[:, [1, 0]]
    wrong("heads-swapped", sw)
    one = o.clone()
    one[inp["qo"][0] + 5, 3, 7] += 0.25
    wrong("one-element-off", one)
    nan = o.clone()
    nan[inp["qo"][0] + 1, 0, 0] = float("nan")
    wrong("nan", nan)
    outside = o.clone()
    outside[0, 0, 0] = 1.0  # row 0 is in no request (lead rows)
    snaps = snapshot(inp)
    _, ok_out = frame_ok(inp, outside, snaps)
    print(f"  negative write-outside: frame {'rejected' if not ok_out else 'NOT REJECTED'}")
    if ok_out:
        fails.append("write-outside")
    print(f"  (the kernel's own output on this case: err/tol {good:.3e})")
    return fails


def main():
    gen = torch.Generator().manual_seed(0)
    W = {"out": 0.0, "ok": True, "tolrel_max": 0.0, "at": None, "case": None}
    # the code base's test (test_triton_attention_kernels.py:317-325): 19 requests, 12/4 heads
    for D in (256, 128, 96, 80, 13):
        run_case(f"test-D{D}", test_shape(gen, D), W, report=(D == 128))
    # SETUP.md's model shapes, scattered prefix slots, prefix 0 and extend 1 included
    pre = [0, 1, 1000, 4096, 63, 64, 65, 3000]
    ext = [1, 2048, 127, 128, 129, 1000, 1, 300]
    run_case("llama-32/8", make_case(32, 8, 128, pre, ext, gen), W, report=True)
    run_case("qwen-28/4", make_case(28, 4, 128, pre, ext, gen), W)
    run_case("qwen-0.5b-14/2-D64", make_case(14, 2, 64, pre, ext, gen), W)
    run_case("gemma3-16/8-D256", make_case(16, 8, 256, pre, ext, gen), W)
    run_case("mqa-8/1", make_case(8, 1, 128, [5, 0, 777, 2], [64, 65, 1, 500], gen), W)
    run_case("strided-q", make_case(32, 8, 128, pre[:4], ext[:4], gen, fused=True), W)
    # the longest covered request: prefix 2^17 - 8192, 8192 new tokens
    # the longest requests (128K and the cap 2^18 = 256K positions), with negative tests
    long_fails = long_case("long-128K", gen, 2 ** 17 - 8192, 8192, W)
    long_fails += long_case("long-256K", gen, 2 ** 18 - 2048, 2048, W, quarter_required=False)
    long_fails += long_case("long-256K-shifted", gen, 2 ** 18 - 2048, 2048, W,
                            shift_first_quarter=0.25)
    # edges of Inst.ok: head dims 9 and 200, a custom sm_scale, a request with no new tokens, a
    # grid larger than the longest request; head dims up to 8 are rejected on the host
    for D in (1, 8):
        try:
            run_host(make_case(2, 1, D, [3], [5], gen))
            print(f"edge-D{D}: UNEXPECTEDLY COMPILED")
            W["ok"] = False
        except Exception as ex:  # triton.compiler.errors.CompilationError
            print(f"edge-D{D}: rejected on the host as expected ({type(ex).__name__})")
    run_case("edge-D9", make_case(2, 1, 9, [70, 0], [1, 300], gen), W)
    run_case("edge-D200", make_case(4, 2, 200, [129, 0, 64], [65, 300, 64], gen), W)
    run_case("edge-smscale", make_case(4, 2, 64, [333, 7], [64, 200], gen, sm_scale=0.5), W)
    run_case("edge-empty", make_case(4, 4, 128, [10, 50, 0], [0, 130, 0], gen, max_ext=400), W)
    # edges of Pre: large scores, large and tiny values, uniform attention over a long prefix
    run_case("pre-bigscore", make_case(8, 2, 128, [600, 0], [129, 300], gen, scale_qk=6.0), W)
    run_case("pre-bigv", make_case(4, 4, 64, [300], [300], gen, v_scale=1e30), W)
    run_case("pre-smallv", make_case(4, 4, 64, [300], [300], gen, v_scale=1e-20), W)
    # subnormal `bfloat16` values: below the absolute floor `2^-80`, which the check meets
    # trivially (the library's absolute `tl.dot` term alone is about `2^-108`)
    run_case("pre-subnormalv", make_case(4, 4, 64, [300], [300], gen, v_scale=1e-37), W)
    cs = [1.0 + i / 128 for i in range(64)]  # every bfloat16 in [1, 1.5), then their negatives
    run_case("pre-uniform", make_case(2, 1, 128, [4000, 0, 255], [600, 129, 2], gen,
                                      uniform_v=cs + [-c for c in cs]), W)
    run_case("pre-adversarial", adversarial(gen), W)
    run_case("pre-adversarial-small", adversarial_small(gen), W)
    # sliding-window layers: the code base's test (window 127, 64/8 heads, the whole prefix passed)
    # at a reduced draw; Gemma 3's local layers (window 1023, 16/8 heads, head dim 256) with window
    # prefixes; a window wider than the block, and the window's edges at the tiles (127, 64, 1)
    mode_fails = []
    pre = [5000, 1, 0, 700, 3000]
    ext = [1200, 300, 129, 1, 64]
    inp = make_case(64, 8, 128, pre, ext, gen, sw=127, mean=0.1, std=0.2)
    o = run_case("test-swa-127", inp, W, report=True)
    mode_fails += mode_negatives("test-swa-127", inp, o, [("no-window", dict(window=False))])
    gpre = [1023, 1023, 500, 0, 1023, 1023, 1023, 1023]
    gext = [8192 - 7 * 600] + [600] * 7
    inp = make_case(16, 8, 256, gpre, gext, gen, sw=1023)
    o = run_case("gemma3-local-1023", inp, W, rows=128, report=True)
    mode_fails += mode_negatives("gemma3-local-1023", inp, o, [("no-window", dict(window=False))])
    for w in (64, 1, 4096):
        run_case(f"swa-{w}", make_case(8, 2, 128, [300, 63, 0, 4100], [200, 129, 65, 300], gen,
                                       sw=w), W)
    run_case("swa-D64", make_case(14, 2, 64, [127, 0], [500, 129], gen, sw=127), W)
    # target verification: int32 qo_indptr, tree masks of 4, 8 and 64 draft tokens, with and
    # without a window (window prefixes, the offsets of the windows in the mask)
    inp = make_case(32, 8, 128, [3000, 1, 0, 600, 2047, 64], [64] * 6, gen, verify=True,
                    lead=0, trail=0)
    o = run_case("verify-llama-64", inp, W, report=True)
    mode_fails += mode_negatives("verify-llama-64", inp, o, [
        ("causal-not-tree", dict(use_mask=False)), ("no-prefix", dict(use_prefix=False))])
    run_case("verify-chain-4", make_case(28, 4, 128, [100, 5, 0, 4000] * 4, [4] * 16, gen,
                                         verify=True, lead=0, trail=0), W)
    run_case("verify-D256", make_case(16, 8, 256, [500, 70], [8, 8], gen, verify=True,
                                      lead=0, trail=0), W)
    seq = [5000, 1023, 300, 0]
    inp = make_case(16, 8, 256, [min(x, 1023) for x in seq], [8] * 4, gen, verify=True, sw=1023,
                    seq=seq, lead=0, trail=0)
    o = run_case("verify-gemma3-local", inp, W, report=True)
    mode_fails += mode_negatives("verify-gemma3-local", inp, o, [
        ("causal-not-tree", dict(use_mask=False)), ("no-window", dict(window=False))])
    seq = [700, 64, 9]
    inp = make_case(8, 2, 128, [min(x, 60) for x in seq], [64, 64, 8], gen, verify=True, sw=60,
                    seq=seq, lead=0, trail=0)
    o = run_case("verify-swa-60", inp, W)
    mode_fails += mode_negatives("verify-swa-60", inp, o, [
        ("no-window", dict(window=False)), ("mask-without-offset", dict(offset=False))])
    print("negative tests:")
    fails = negatives(gen) + long_fails + mode_fails
    print(f"WORST: err/tol {W['out']:.3e} (margin {1 / max(W['out'], 1e-300):.2f}x) in {W['case']} at "
          f"(err, tol, Xrow, trips, attnAbs) = {W['at']}; largest tolRel {W['tolrel_max']:.3e}; "
          f"finiteness and frame {'hold' if W['ok'] else 'FAIL'}; negatives "
          f"{'all rejected' if not fails else 'NOT REJECTED: ' + ', '.join(fails)}")
    sys.exit(0 if W["ok"] and W["out"] < 1 and not fails else 1)


if __name__ == "__main__":
    main()
