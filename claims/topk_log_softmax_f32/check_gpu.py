"""GPU check of the claim `Claims.TopkLogSoftmax`: the kernel, run on the GPU, meets `Post` on the launches checked.

Runs the kernel's own host, `compute_token_logprobs` (and `compute_topk_scores`, which builds the
sampler's token-id matrix) of `source/logprob.py` (imported by path), on the launches of
`SETUP.md` and on launches at the edges of `Inst.ok`, plus direct launches of
`_topk_log_softmax_kernel`, and checks `Post` as it is stated, row by row:

* if the row is `RowOk` (every logit finite or `-inf`, one finite, the finite ones within `2^127` of
  each other), with `L_j = x_id - m - log(sum_k e^(x_k - m))` the exact log-softmax computed in
  `float64` from the host's input (`m` the row's largest logit, `e^(-inf) = 0`): an output whose
  token's logit is `-inf` is `-inf` exactly; every other output is finite, and there is one shift
  `sh` with `|sh| <= lseTol = 2^-34*vocab_size + 2^-16` and `|c_j - (L_j - sh)| <= T_j` for every
  `j`, `T_j = elemTol*|L_j| + floorTol` (`elemTol = 2^-22`, `floorTol = 2^-37`). The check computes,
  per row, the smallest `lam` such that some `sh` with `|sh| <= lam*lseTol` has
  `|c_j - (L_j - sh)| <= lam*T_j` for every `j` (bisection), and reports the largest `lam` (the
  error / tolerance; at most 1 means `Post` holds);
* that the tolerance is meaningful: the host's output rounded to `bfloat16`, a log-sum-exp that
  drops the row's last block of `1024` logits, the logit of the next token id, and the exact values
  shifted by `2*lseTol` fail it;
* the frame: the logits and the ids are unchanged (bitwise), and direct launches with the output
  inside a sentinel-filled buffer leave every element outside the output elements unchanged.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "logprob.py")
spec = importlib.util.spec_from_file_location("logprob_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
ELEM = 2.0 ** -22       # Claim.lean, `elemTol`
FLOOR = 2.0 ** -37      # `floorTol`
NINF = -float("inf")


def lse_tol(V):
    """`lseTol vocab_size`."""
    return 2.0 ** -34 * V + 2.0 ** -16


def post_rows(x, ids, out, chunk=32):
    """(worst lam over RowOk rows, -inf outputs exact and others finite, rows not RowOk)."""
    B, V = x.shape
    S = lse_tol(V)
    worst, exact, notok = 0.0, True, 0
    for r0 in range(0, B, chunk):
        xr = x[r0:r0 + chunk].double()
        ir = ids[r0:r0 + chunk].long()
        c = out[r0:r0 + chunk].double()
        isfin = torch.isfinite(xr)
        neginf = xr == NINF
        big = torch.full_like(xr, float("inf"))
        fmax = torch.where(isfin, xr, -big).max(dim=1).values
        fmin = torch.where(isfin, xr, big).min(dim=1).values
        rowok = (isfin | neginf).all(dim=1) & isfin.any(dim=1) & (fmax - fmin <= 2.0 ** 127)
        notok += int((~rowok).sum().item())
        if not rowok.any() or ir.shape[1] == 0:
            continue
        xr, ir, c, m = xr[rowok], ir[rowok], c[rowok], fmax[rowok].unsqueeze(1)
        A = torch.exp(xr - m).sum(dim=1, keepdim=True)          # -inf gives 0
        xg = xr.gather(1, ir)
        L = xg - m - torch.log(A)
        ninf_tok = xg == NINF
        exact &= bool((c[ninf_tok] == NINF).all().item())
        fin_tok = ~ninf_tok
        exact &= bool(torch.isfinite(c[fin_tok]).all().item())
        Lf = torch.where(fin_tok, L, torch.zeros_like(L))
        cf = torch.where(fin_tok & torch.isfinite(c), c, torch.zeros_like(c))
        d = Lf - cf                                             # sh must be within lam*T_j of d
        T = ELEM * Lf.abs() + FLOOR
        inf_ = torch.full_like(d, float("inf"))

        def feasible(lam):
            lam_ = lam.unsqueeze(1)
            lo = torch.where(fin_tok, d - lam_ * T, -inf_).max(dim=1).values
            hi = torch.where(fin_tok, d + lam_ * T, inf_).min(dim=1).values
            lo = torch.maximum(lo, -lam * S)
            hi = torch.minimum(hi, lam * S)
            return lo <= hi

        lo_l = torch.zeros(c.shape[0], dtype=torch.float64, device=c.device)
        hi_l = torch.full_like(lo_l, 1e12)
        for _ in range(80):
            mid = (lo_l + hi_l) / 2
            f = feasible(mid)
            hi_l = torch.where(f, mid, hi_l)
            lo_l = torch.where(f, lo_l, mid)
        worst = max(worst, hi_l.max().item())
    return worst, exact, notok


def direct(x, ids, TB, guard=4096):
    """`_topk_log_softmax_kernel` on `x` and `ids` with `TOPK_BLOCK_SIZE = TB`, the output inside a
    sentinel buffer."""
    B, V = x.shape
    K = ids.shape[1]
    SENT = -777.25
    buf = torch.full((guard + B * K + guard,), SENT, device=DEV)
    out = buf[guard:guard + B * K].view(B, K)
    mod._topk_log_softmax_kernel[(B,)](out, x, x.stride(0), ids, ids.stride(0), K, V,
                                       BLOCK_SIZE=1024, TOPK_BLOCK_SIZE=TB)
    torch.cuda.synchronize()
    mask = torch.ones_like(buf, dtype=torch.bool)
    mask[guard:guard + B * K] = False
    frame = bool((buf[mask].view(torch.int32) == torch.tensor(SENT, device=DEV).view(torch.int32)).all().item())
    return out, frame


results = []
worst_all = 0.0


def check(name, x, ids, host=True, extra=()):
    """`extra`: `TOPK_BLOCK_SIZE` values for direct launches."""
    global worst_all
    xb, ib = x.clone(), ids.clone()
    parts, ok = [], True
    if host:
        out = mod.compute_token_logprobs(x, ids)
        torch.cuda.synchronize()
        w, ex, nok = post_rows(x, ids, out)
        ok &= w <= 1.0 and ex
        worst_all = max(worst_all, w)
        TB = min(triton.next_power_of_2(ids.shape[1]), 1024)
        parts.append(f"host TB={TB}: worst {w:.4f}, exact {ex}, rows not RowOk {nok}")
        del out
    for TB in extra:
        out, frame = direct(x, ids, TB)
        w, ex, nok = post_rows(x, ids, out)
        ok &= w <= 1.0 and ex and frame
        worst_all = max(worst_all, w)
        parts.append(f"direct TB={TB}: worst {w:.4f}, frame {frame}")
        del out
    ok &= torch.equal(x.view(torch.int32), xb.view(torch.int32)) and torch.equal(ids, ib)
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} {tuple(x.shape)} K={ids.shape[1]}: " + "; ".join(parts),
          flush=True)


def logits(B, V, g, scale=4.0):
    return torch.randn(B, V, device=DEV, generator=g) * scale


def sampled(x, g):
    """A token per row, sampled from the row's softmax (as the sampler's tokens)."""
    p = torch.softmax(x.double(), dim=1)
    return torch.multinomial(p.float(), 1, generator=g).squeeze(1)


def sampler_ids(x, num_logprobs, g):
    """The fast path's token-id matrix, built by the host's `compute_topk_scores`."""
    ids, _ = mod.compute_topk_scores(x, num_logprobs, sampled(x, g))
    return ids


def custom_ids(x, num_logprobs, per_req, g):
    """The custom path's matrix `[B, 1 + max(num_logprobs, max(per_req))]`: the sampled token, then
    a request's own ids (or the top-k), zeros after them (`_fill_logprob_token_ids_kernel`)."""
    B, V = x.shape
    cols = max(num_logprobs, max(per_req))
    ids = torch.zeros(B, 1 + cols, dtype=torch.int64, device=DEV)
    ids[:, 0] = sampled(x, g)
    top = torch.topk(x, num_logprobs, dim=-1).indices if num_logprobs else None
    for r in range(B):
        n = per_req[r]
        if n > 0:
            ids[r, 1:1 + n] = torch.randint(0, V, (n,), device=DEV, generator=g)
        elif num_logprobs:
            ids[r, 1:1 + num_logprobs] = top[r]
    return ids


g = torch.Generator(device=DEV).manual_seed(0)

# The launches of SETUP.md.
x = logits(1024, 151936, g)
check("sample, no logprobs (Qwen)", x, sampler_ids(x, 0, g), extra=(1,))
del x
x = logits(256, 128256, g)
check("sample, logprobs=20 (Llama 3)", x, sampler_ids(x, 20, g), extra=(32, 4))
del x
x = logits(128, 262208, g)
check("sample, logprobs=5 (Gemma 3)", x, sampler_ids(x, 5, g), extra=(8,))
del x
x = logits(64, 129280, g)
per = [128 if r % 3 == 0 else (17 if r % 3 == 1 else 0) for r in range(64)]
check("custom ids (DeepSeek-V3)", x, custom_ids(x, 20, per, g), extra=(256, 64))
del x
x = logits(5120, 50257, g)
check("speculative decoding, logprobs=1 (GPT-2)", x, sampler_ids(x, 1, g), extra=(2,))
del x
x = logits(1024, 32000, g)
check("prompt logprobs, whole vocabulary (Llama 2)", x, sampler_ids(x, 32000, g), extra=(1024, 128))
del x
x = logits(1024, 151936, g)
check("prompt label ids (Qwen)", x, torch.randint(0, 151936, (1024, 8), device=DEV, generator=g),
      extra=(8,))
del x
wide = logits(256, 152064, g)
x = wide[:, :151936]
check("logits column slice (Qwen, stride 152064)", x, sampler_ids(x, 20, g), extra=(32,))
del x, wide
x = logits(0, 151936, g)
check("empty batch", x, torch.zeros(0, 21, dtype=torch.int64, device=DEV))
del x
torch.cuda.empty_cache()

# Edges of Inst.ok and of RowOk.
pinf, nan = float("inf"), float("nan")
B, V = 48, 3000
x = logits(B, V, g, 30.0)
x[0, :] = 0.0                                    # all equal: the sum is V
x[1, :] = NINF
x[1, 7] = 1.0                                    # one finite logit, the rest -inf
x[2, ::2] = NINF                                 # half the vocabulary masked
x[3, :] = NINF                                   # not RowOk: all -inf
x[4, 5] = nan                                    # not RowOk
x[5, 9] = pinf                                   # not RowOk
x[6, :] = torch.linspace(-8.5e37, 8.5e37, V)     # finite logits within 2^127 of each other
x[7, :] = 1.7e38
x[7, 1] = 0.0                                    # gap just below 2^127
x[8, :] = 2.0 ** -149                            # subnormal logits
x[9, :] = torch.linspace(-1000, 0, V)            # gaps up to 1000
x[10, :] = -3.0e38                               # all near the bottom of the range
x[11, :1500] = NINF                              # the first blocks masked
x[12, :] = 1e30                                  # large equal logits
x[13, :] = torch.linspace(-1.7e38, 1.7e38, V)    # not RowOk: x - max overflows
xs, _ = torch.sort(x[14], descending=True)
x[14] = xs
x[15, 2999] = 50.0                               # the max in the last, partial block
ids = torch.randint(0, V, (B, 40), device=DEV, generator=g)
ids[1, :] = torch.arange(40, device=DEV)         # -inf tokens and the finite one
ids[1, 0] = 7
ids[2, :] = torch.arange(40, device=DEV)         # every other token -inf
ids[11, :] = torch.arange(1480, 1520, device=DEV)
ids[15, 0] = 2999
ids[16, :] = 5                                   # repeated ids
ids[17, :] = V - 1
check("edge values", x, ids, extra=(1, 2, 64, 1024))
masked = logits(256, 151936, g)                   # processed logits: all but the top 50 masked
kth = torch.topk(masked, 50, dim=1).values[:, -1:]
masked = torch.where(masked >= kth, masked, torch.full_like(masked, NINF))
check("top-k masked logits (-inf outside the top 50)", masked, sampler_ids(masked, 20, g), extra=(32,))
del masked
for Vs in (1, 2, 1023, 1024, 1025, 2047, 4097):
    x = logits(37, Vs, g)
    check(f"vocab {Vs}", x, torch.randint(0, Vs, (37, 3), device=DEV, generator=g), extra=(1, 4))
x = logits(4, 2 ** 19, g, 10.0)
check("vocab 2^19", x, sampler_ids(x, 20, g), extra=(32,))
x = logits(8, 2 ** 19, g, 0.0)
check("vocab 2^19, all equal", x, sampler_ids(x, 3, g))
x = logits(3, 70000, g)
# The host cannot launch with no ids (`next_power_of_2(0) = 0` makes `tl.arange(0, 0)`, which Triton
# rejects); a direct launch with `TOPK_BLOCK_SIZE = 1` can.
check("no ids (topk = 0), direct", x, torch.zeros(3, 0, dtype=torch.int64, device=DEV), host=False,
      extra=(1,))
x = logits(5, 5000, g)
check("one id (topk = 1, TOPK_BLOCK_SIZE = 1)", x, torch.randint(0, 5000, (5, 1), device=DEV, generator=g),
      extra=(1, 1024))
idw = torch.randint(0, 5000, (5, 90), device=DEV, generator=g)
check("ids as a column slice (stride 90)", x, idw[:, :33], extra=(1, 16, 64))
x = logits(1, 262208, g, 2.0)
check("one row, Gemma 3 vocabulary, 3000 ids", x, torch.randint(0, 262208, (1, 3000), device=DEV, generator=g),
      extra=(1024, 256))
del x, idw
torch.cuda.empty_cache()

# Meaningfulness: plausible wrong outputs must fail Post.
for V in (32000, 151936):
    x = logits(256, V, g)
    ids = sampler_ids(x, 20, g)
    out = mod.compute_token_logprobs(x, ids)
    w, _, _ = post_rows(x, ids, out.to(torch.bfloat16).float())
    results.append(w > 1.0)
    print(f"{'ok ' if w > 1.0 else 'BAD'} output rounded to bfloat16 V={V}: worst {w:.1f} (must exceed 1)",
          flush=True)
    xd = x.double()
    m = xd.max(dim=1, keepdim=True).values
    e = torch.exp(xd - m)
    nb = (V - 1) // 1024 * 1024
    wrong = (xd.gather(1, ids) - m - torch.log(e[:, :nb].sum(dim=1, keepdim=True))).float()
    w, _, _ = post_rows(x, ids, wrong)
    results.append(w > 1.0)
    print(f"{'ok ' if w > 1.0 else 'BAD'} lse drops the last block V={V}: worst {w:.1f} (must exceed 1)",
          flush=True)
    lse = torch.log(e.sum(dim=1, keepdim=True))
    wrong = (xd.gather(1, (ids + 1) % V) - m - lse).float()
    w, _, _ = post_rows(x, ids, wrong)
    results.append(w > 1.0)
    print(f"{'ok ' if w > 1.0 else 'BAD'} logit of the next id V={V}: worst {w:.1f} (must exceed 1)",
          flush=True)
    wrong = (xd.gather(1, ids) - m - lse + 2 * lse_tol(V)).float()
    w, _, _ = post_rows(x, ids, wrong)
    results.append(w > 1.0)
    print(f"{'ok ' if w > 1.0 else 'BAD'} shifted by 2*lseTol V={V}: worst {w:.2f} (must exceed 1)",
          flush=True)
    del x, xd, e, out, wrong

print(f"worst error / tolerance over every RowOk row: {worst_all:.4f}")
print(f"{sum(results)}/{len(results)} cases pass")
assert all(results)
