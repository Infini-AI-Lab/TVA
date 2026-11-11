"""GPU check of the claim for `rejection_greedy_i32`.

Runs the kernel's own host (`source/rejection_greedy.py`) on the
setup's launches (the host's test cases, the random batches both with `is_greedy = None` and with
an `is_greedy` mask), on launches at the edges of `Inst.ok` and `Pre`, and on two launches beyond
them ("beyond Inst.ok": more than 8192 requests, and more than 2^22 packed drafts, which the claim
leaves out of scope; they check the kernel past the claim), and checks `Post` exactly as
`Claim.lean` states it:

* for every request `r < B` and column `j < S + 1`, the output holds `expect d init r j`
  (for a greedy request, the target's `int32` truncation for `j < min(k_r + 1, n_r)`, the bonus
  token for `j = n_r = k_r`), and its initial value otherwise (every column of a request that is
  not greedy);
* the frame: every byte outside the output's `B * (S + 1)` elements is unchanged (the inputs, and
  guard regions around the output in a direct launch with the host's own launch line).

Every value is an integer, so the check is exact (no tolerance). Negative tests: wrong outputs
(the mask ignored, a dropped bonus token, an all-zero output, a shifted start, a missing mismatch
column) that `Post` must reject. Run with
`CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "rejection_greedy.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("rejection_greedy_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")


def trunc32(t):
    """`Int.bmod t (2^32)`: the `int32` two's-complement truncation of an `int64` tensor."""
    lo = t.to(torch.int64).bitwise_and(0xFFFFFFFF)
    return torch.where(lo >= 2 ** 31, lo - 2 ** 32, lo)


def expected(cu, draft, target, bonus, B, S, init_out, greedy=None):
    """`Post`'s first clause, vectorised: the output `B x (S + 1)` the claim requires, given the
    initial output `init_out` (int64) and the mask `greedy` (`None`, or its bytes: `greedyAt` is
    `byte != 0`). Computed from the host's inputs (`cu`, drafts, targets, bonus, mask), not from
    what the kernel reads."""
    cu = cu.to(torch.int64).cpu()
    draft = draft.to(torch.int64).cpu()
    tgt = trunc32(target.cpu())
    bonus = bonus.reshape(-1).to(torch.int64).cpu()
    start = torch.cat([torch.zeros(1, dtype=torch.int64), cu[:-1]])  # startOf
    n = cu - start  # nOf
    p = torch.arange(S + 1, dtype=torch.int64)
    in_req = p[None, :] < n[:, None]  # p < n_r
    idx = (start[:, None] + p[None, :]).clamp(max=max(len(draft) - 1, 0))
    if len(draft) > 0:
        d_at = draft[idx]
        t_at = tgt[idx]
    else:
        d_at = torch.zeros(B, S + 1, dtype=torch.int64)
        t_at = torch.zeros(B, S + 1, dtype=torch.int64)
    mism = in_req & (d_at != t_at)
    # k_r: the first mismatching position, n_r when none (firstDiff / matchLen)
    big = torch.full((B, S + 1), S + 2, dtype=torch.int64)
    first = torch.where(mism, p[None, :].expand(B, S + 1), big).min(dim=1).values
    k = torch.where(first > S + 1, n, first)
    out = init_out.clone().to(torch.int64).cpu()
    write_t = p[None, :] < torch.minimum(k + 1, n)[:, None]
    out = torch.where(write_t, t_at, out)
    write_b = (p[None, :] == n[:, None]) & (k == n)[:, None]
    out = torch.where(write_b, bonus[:, None].expand(B, S + 1), out)
    if greedy is not None:
        g = greedy.cpu().view(torch.uint8).to(torch.int64) != 0  # greedyAt
        out = torch.where(g[:, None], out, init_out.to(torch.int64).cpu())
    return out


def negatives(cu, draft, target, bonus, B, S, init, greedy, exp):
    """Wrong outputs `Post` must reject (each differs from `expect` in some cell); returns how
    many were rejected and how many were tried."""
    wrong = []
    if greedy is not None and bool((greedy.cpu().view(torch.uint8) == 0).any()):
        wrong.append(expected(cu, draft, target, bonus, B, S, init))  # the mask ignored
    cu64 = cu.to(torch.int64).cpu()
    start = torch.cat([torch.zeros(1, dtype=torch.int64), cu64[:-1]])
    n = cu64 - start
    bonus_cells = (exp == bonus.reshape(-1).to(torch.int64).cpu()[:, None]) & (
        torch.arange(S + 1)[None, :] == n[:, None])
    if bool(bonus_cells.any()):
        w = exp.clone()
        w[bonus_cells] = init[bonus_cells]  # bonus token dropped
        wrong.append(w)
    if bool((exp != 0).any()):
        wrong.append(torch.zeros_like(exp))  # all-zero output
    some_greedy = greedy is None or bool((greedy.cpu().view(torch.uint8) != 0).any())
    if B > 1 and draft.shape[0] > 1 and some_greedy:
        # a shifted packing (request r's drafts read from one position later)
        wrong.append(expected(cu, torch.roll(draft, -1), torch.roll(target, -1), bonus, B, S, init, greedy))
    written = exp != init
    if bool(written.any()):
        w = exp.clone()
        idx = written.nonzero()[0]
        w[idx[0], idx[1]] = init[idx[0], idx[1]]  # one written column missing
        wrong.append(w)
    rejected = sum(1 for w in wrong if not torch.equal(w, exp))
    return rejected, len(wrong)


def check_pre(cu, n, S, N):
    """`Pre`: nonnegative, nondecreasing running sums, last at most `N`, every `n_r <= S`."""
    cu = cu.to(torch.int64).cpu()
    assert cu[0] >= 0 and bool((cu[1:] >= cu[:-1]).all()) and cu[-1] <= N
    assert max(n) <= S


results = []
neg_results = []


def run_case(name, draft, n, S, cu, target, bonus, compare_host_reference=True, greedy=None):
    """The host's launch; with a mask `greedy`, also vLLM's mixed-batch `uniform_probs` (a
    `float32` tensor of `num_tokens` values, not read with `SYNTHETIC_MODE = False`)."""
    B = len(n)
    check_pre(cu, n, S, draft.shape[0])
    inputs = [x.to(DEV) for x in (draft, cu, target, bonus)]
    g_dev = None if greedy is None else greedy.to(DEV)
    if g_dev is not None:
        inputs.append(g_dev)
        uniform = torch.rand(draft.shape[0], dtype=torch.float32, device=DEV)
        inputs.append(uniform)
    else:
        uniform = None
    before = [x.clone() for x in inputs]
    got = mod.rejection_greedy_sample(inputs[0], n, S, inputs[1], inputs[2], inputs[3], g_dev, uniform).cpu()
    assert got.shape == (B, S + 1) and got.dtype == torch.int32
    init = torch.full((B, S + 1), -1, dtype=torch.int64)  # the host's torch.full(..., -1)
    exp = expected(cu, draft, target, bonus, B, S, init, greedy)
    ok = torch.equal(got.to(torch.int64), exp)
    frame_ok = all(torch.equal(a, b) for a, b in zip(before, inputs))
    if compare_host_reference:
        ok_ref = torch.equal(got, mod.reference(draft, n, S, target, bonus,
                                                None if greedy is None else greedy.cpu()))
    else:
        ok_ref = None
    rej, tried = negatives(cu, draft, target, bonus, B, S, init, greedy, exp)
    neg_results.append((name, rej, tried))
    results.append((name, ok, frame_ok, ok_ref, B * (S + 1)))
    print(f"{name}: B={B} S={S} N={draft.shape[0]} is_greedy {'None' if greedy is None else 'mask'}: "
          f"output exact {ok} ({B * (S + 1)} cells), inputs unchanged {frame_ok}, host reference {ok_ref}, "
          f"wrong outputs rejected {rej}/{tried}")


def run_guarded(name, draft, n, S, cu, target, bonus, guard=4096, greedy=None):
    """The host's launch line (source lines 164-177) on an output carved from a sentinel pool
    whose initial contents are arbitrary (not the host's -1): checks the `none` cells keep their
    initial value and that no byte around the output changes."""
    B = len(n)
    check_pre(cu, n, S, draft.shape[0])
    g = torch.Generator().manual_seed(7)
    pool = torch.randint(-2 ** 31, 2 ** 31, (guard + B * (S + 1) + guard,), generator=g, dtype=torch.int64)
    pool = pool.to(torch.int32).to(DEV)
    pool0 = pool.clone()
    out = pool[guard:guard + B * (S + 1)].view(B, S + 1)
    init = out.clone().to(torch.int64).cpu()
    d, c, t, b = (x.to(DEV) for x in (draft, cu, target, bonus))
    gd = None if greedy is None else greedy.to(DEV)
    mod.rejection_greedy_sample_kernel[(B,)](
        out, c, d, t, b, gd, S, None, None, SYNTHETIC_MODE=False, num_warps=4, num_stages=3)
    exp = expected(cu, draft, target, bonus, B, S, init, greedy)
    ok = torch.equal(out.cpu().to(torch.int64), exp)
    frame_ok = torch.equal(pool[:guard], pool0[:guard]) and torch.equal(pool[guard + B * (S + 1):],
                                                                        pool0[guard + B * (S + 1):])
    results.append((name, ok, frame_ok, None, B * (S + 1)))
    print(f"{name} (guarded direct launch): B={B} S={S}: output exact {ok}, guards unchanged {frame_ok}")


def cu_of(n):
    return torch.tensor(n, dtype=torch.int64).cumsum(0).to(torch.int32)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup: vLLM's hand-written cases (source lines 220-237).
    for spec_, out_toks in [
        ([[1, 2, 3]], [[1, 2, 3, 4]]), ([[1, 2, 3]], [[1, 5, 3, 4]]), ([[1, 2], [3]], [[1, 2, 5], [3, 4]]),
        ([[1]], [[1, 2]]), ([[]], [[5]]), ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3, 4], [4, 7, 6, 8]]),
    ]:
        n = [len(s) for s in spec_]
        draft = torch.tensor(sum(spec_, []), dtype=torch.int32)
        target = torch.tensor(sum([o[:-1] for o in out_toks], []), dtype=torch.int64)
        bonus = torch.tensor([[o[-1]] for o in out_toks], dtype=torch.int32)
        run_case(f"setup vLLM {spec_}", draft, n, max(n), cu_of(n), target, bonus)
    # The setup: the random batches (source lines 241-265), same generator, is_greedy = None.
    for batch_size, k, vocab, accept, mixed, pad in [
        (1, 1, 128256, 0.7, False, False), (8, 3, 128256, 0.7, False, False),
        (64, 5, 151936, 0.8, True, False), (256, 4, 32000, 0.5, True, True),
        (1024, 8, 151936, 0.9, True, False), (37, 128, 128256, 0.97, True, False),
        (16, 2, 32000, 1.0, False, False), (16, 2, 32000, 0.0, True, False),
    ]:
        n = torch.randint(0, k + 1, (batch_size,), generator=gen).tolist() if mixed else [k] * batch_size
        draft, cu, target, bonus = mod.make_batch(n, vocab, accept, gen, pad)
        mask = torch.rand(batch_size, generator=gen) < 0.5  # the test's is_greedy mask
        run_case(f"setup batch {batch_size} k {k} accept {accept}", draft, n, max(n), cu, target, bonus)
        run_case(f"setup batch {batch_size} k {k} accept {accept}", draft, n, max(n), cu, target, bonus,
                 greedy=mask)

    g = torch.Generator().manual_seed(1)
    # Edges of Inst.ok / Pre.
    # MAX_SPEC_LEN = 128 drafts for every request, all accepted (bonus at column 128).
    n = [128] * 512
    draft, cu, target, bonus = mod.make_batch(n, 151936, 1.0, g)
    run_case("edge S=128 all accepted", draft, n, 128, cu, target, bonus)
    # S larger than every n_r (Pre allows S >= max n_r), zero-draft requests mixed in.
    n = torch.randint(0, 4, (300,), generator=g).tolist()
    draft, cu, target, bonus = mod.make_batch(n, 1000, 0.6, g, pad_drafts=True)
    run_case("edge S=9 > max n_r", draft, n, 9, cu, target, bonus)
    # Targets with high bits: the kernel compares the int32 truncation (2^32 + t matches draft t,
    # negative int64 values truncate), as Post states.
    n = [4] * 1000
    draft, cu, target, bonus = mod.make_batch(n, 1000, 0.8, g)
    hi = torch.randint(-2 ** 20, 2 ** 20, target.shape, generator=g, dtype=torch.int64) * 2 ** 32
    target_hi = target + hi
    target_hi[::7] = torch.randint(-2 ** 62, 2 ** 62, target_hi[::7].shape, generator=g, dtype=torch.int64)
    run_case("edge int64 targets with high bits", draft, n, 4, cu, target_hi, bonus,
             compare_host_reference=False)
    # Draft ids -1 and int32 extremes; S = 0 (every request has no draft: only the bonus).
    n = [0] * 70000
    draft, cu, target, bonus = mod.make_batch(n, 10, 0.5, g)
    run_case("beyond Inst.ok: S=0, 70000 requests", draft, n, 0, cu, target, bonus)
    # A large batch: 2^21 requests of up to 8 drafts (beyond Inst.ok and Pre: B > 8192, and about
    # 2^23 packed drafts).
    n = torch.randint(0, 9, (2 ** 21,), generator=g).tolist()
    draft, cu, target, bonus = mod.make_batch(n, 151936, 0.9, g, pad_drafts=True)
    draft[::1001] = 2 ** 31 - 1
    run_case("beyond Inst.ok: 2^21 requests, k 8", draft, n, 8, cu, target, bonus, compare_host_reference=False)
    # Frame and arbitrary initial output values (guarded direct launch).
    n = torch.randint(0, 6, (777,), generator=g).tolist()
    draft, cu, target, bonus = mod.make_batch(n, 32000, 0.7, g, pad_drafts=True)
    run_guarded("frame k 5", draft, n, 6, cu, target, bonus)
    n = [0, 0, 3, 0]
    draft, cu, target, bonus = mod.make_batch(n, 5, 1.0, g)
    run_guarded("frame zero-draft requests", draft, n, 3, cu, target, bonus)
    # Mixed batches: vLLM's is_greedy = (temperature == 0), a bool tensor.
    n = [128] * 1024
    draft, cu, target, bonus = mod.make_batch(n, 151936, 0.95, g)
    temperature = torch.where(torch.rand(1024, generator=g) < 0.3, 0.0, 0.7).to(torch.float32)
    run_case("edge serving mixed (temperature == 0), S=128", draft, n, 128, cu, target, bonus,
             greedy=temperature == 0)
    n = torch.randint(0, 9, (5000,), generator=g).tolist()
    draft, cu, target, bonus = mod.make_batch(n, 32000, 0.8, g, pad_drafts=True)
    run_case("edge mask all True", draft, n, 8, cu, target, bonus, greedy=torch.ones(5000, dtype=torch.bool))
    run_case("edge mask all False", draft, n, 8, cu, target, bonus, greedy=torch.zeros(5000, dtype=torch.bool))
    # Mask bytes other than 0 and 1 (Pre allows any byte; greedyAt is byte != 0).
    raw = torch.randint(0, 256, (5000,), generator=g, dtype=torch.int64).to(torch.uint8)
    raw[::3] = 0
    run_case("edge mask bytes 0..255", draft, n, 8, cu, target, bonus, greedy=raw.view(torch.bool),
             compare_host_reference=False)
    n = torch.randint(0, 6, (777,), generator=g).tolist()
    draft, cu, target, bonus = mod.make_batch(n, 32000, 0.7, g, pad_drafts=True)
    run_guarded("frame mask", draft, n, 6, cu, target, bonus, greedy=torch.rand(777, generator=g) < 0.5)

    bad = [r for r in results if not (r[1] and r[2] and r[3] in (None, True))]
    bad += [r for r in neg_results if r[1] != r[2] or r[2] == 0]
    print(f"negative tests: {sum(r[1] for r in neg_results)} of {sum(r[2] for r in neg_results)} wrong "
          f"outputs rejected")
    print(f"{len(results)} launches, {sum(r[4] for r in results)} output cells checked; "
          f"worst error / tolerance: 0 / 0 (exact) on every output cell; frame held on every launch"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
