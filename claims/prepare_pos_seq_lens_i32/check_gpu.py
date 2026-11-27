"""GPU check of the claim for `prepare_pos_seq_lens_i32`.

Runs the kernel's own host (`source/prepare_pos_seq_lens.py`) on the setup's launches (the host's
test cases and vLLM's serving sizes), on launches at the edges of `Inst.ok` and `Pre`, and on
direct launches of the kernel with the host's launch line, and checks `Post` exactly as
`Claim.lean` states it:

* for every request `r < num_reqs`, `seq_lens[r] = computed r + qlen r` and
  `pos[query_start_loc[r] + j] = computed r + j` for `j < qlen r`, where
  `computed r = num_computed_tokens[idx_mapping[r]]` and
  `qlen r = query_start_loc[r + 1] - query_start_loc[r]`;
* `seq_lens[r] = 0` for `num_reqs <= r < max_num_reqs`;
* the frame: every other element of `pos` and `seq_lens` keeps its initial (random) value, every
  input is unchanged, and the sentinel-filled guard regions around both outputs are unchanged.

Every value is an integer, so the check is exact (no tolerance). Negative tests: wrong outputs
(`seq_lens` off by one, positions that ignore the computed tokens, padding entries left unzeroed,
the slot ignored, a request's positions shifted, a token dropped) that `Post` must reject. Run with
`CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "prepare_pos_seq_lens.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("prepare_pos_seq_lens_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
GUARD = 4096


def check_pre(idx, qsl, nct, N, T):
    """`Pre`: `query_start_loc` nondecreasing on `[0, N]` from a nonnegative start, the last entry
    at most `T` and `2^22`; every slot in `num_computed_tokens`, its value in `[0, 2^22]`."""
    q = qsl[:N + 1].to(torch.int64)
    assert q[0] >= 0 and bool((q[1:] >= q[:-1]).all()) and q[N] <= T and q[N] <= 2 ** 22
    if N > 0:
        s = idx[:N].to(torch.int64)
        assert bool((s >= 0).all()) and bool((s < nct.shape[0]).all())
        c = nct.to(torch.int64)[s]
        assert bool((c >= 0).all()) and bool((c <= 2 ** 22).all())


def expected(idx, qsl, nct, pos0, sl0, N, M):
    """`Post` as a function: the final `pos` and `seq_lens` (int64) from the initial memory, with
    every element `Post` does not name kept at its initial value."""
    pos = pos0.to(torch.int64).clone()
    sl = sl0.to(torch.int64).clone()
    q = qsl[:N + 1].to(torch.int64)
    if N > 0:
        c = nct.to(torch.int64)[idx[:N].to(torch.int64)]  # computed r
        ql = q[1:] - q[:-1]  # qlen r
        sl[:N] = c + ql
        total = int(q[N] - q[0])
        if total > 0:
            req = torch.repeat_interleave(torch.arange(N), ql)  # the request of each token
            k = torch.arange(int(q[0]), int(q[N]))  # its element of pos
            pos[k] = c[req] + (k - q[req])  # computed r + j, j = k - query_start_loc[r]
    if M > N:
        sl[N:M] = 0
    return pos, sl


def post_holds(cand_pos, cand_sl, exp_pos, exp_sl):
    return torch.equal(cand_pos, exp_pos) and torch.equal(cand_sl, exp_sl)


def negatives(idx, qsl, nct, pos0, sl0, N, M, exp_pos, exp_sl):
    """Wrong outputs `Post` must reject; returns (rejected, tried)."""
    wrong = []
    q = qsl[:N + 1].to(torch.int64)
    if N > 0:
        w = exp_sl.clone(); w[0] += 1; wrong.append((exp_pos, w))  # seq_lens off by one
        c = nct.to(torch.int64)[idx[:N].to(torch.int64)]
        if bool((c != 0).any()) and int(q[N] - q[0]) > 0:
            # positions that ignore the computed tokens (0 .. qlen - 1)
            zeros_nct = torch.zeros_like(nct)
            wrong.append(expected(idx, qsl, zeros_nct, pos0, sl0, N, M))
            # the slot ignored: num_computed_tokens[r] instead of [idx_mapping[r]]
            ident = torch.arange(N, dtype=torch.int64)
            if nct.shape[0] >= N and not torch.equal(nct.to(torch.int64)[ident], c):
                wrong.append(expected(ident, qsl, nct, pos0, sl0, N, M))
        written = (exp_pos != pos0.to(torch.int64)).nonzero()
        if written.numel() > 0:
            w = exp_pos.clone(); w[written[-1, 0]] = pos0[written[-1, 0]]  # a token dropped
            wrong.append((w, exp_sl))
            w = exp_pos.clone(); i = written[0, 0]; w[i] += 1  # a position shifted by one
            wrong.append((w, exp_sl))
    if M > N and bool((sl0[N:M] != 0).any()):
        w = exp_sl.clone(); w[N:M] = sl0[N:M].to(torch.int64); wrong.append((exp_pos, w))  # padding left
    rejected = sum(1 for p, s in wrong if not post_holds(p, s, exp_pos, exp_sl))
    return rejected, len(wrong)


results = []
neg_results = []


def run(name, idx, qsl, nct, pos0, sl0, N, M, direct=False, M_arg=None):
    """One launch on outputs carved from sentinel-filled pools (`GUARD` elements on each side):
    through the host (`prepare_pos_seq_lens`, which passes `max_num_reqs = seq_lens.shape[0]`),
    or, with `direct`, the host's launch line with `max_num_reqs = M_arg`."""
    T = pos0.shape[0]
    check_pre(idx, qsl, nct, N, T)
    g = torch.Generator().manual_seed(11)
    pp = torch.randint(-2 ** 62, 2 ** 62, (2 * GUARD + T,), generator=g, dtype=torch.int64)
    pp[GUARD:GUARD + T] = pos0
    sp = torch.randint(-2 ** 31, 2 ** 31, (2 * GUARD + sl0.shape[0],), generator=g,
                       dtype=torch.int64).to(torch.int32)
    sp[GUARD:GUARD + sl0.shape[0]] = sl0
    pp_d, sp_d = pp.to(DEV), sp.to(DEV)
    pos_d = pp_d[GUARD:GUARD + T]
    sl_d = sp_d[GUARD:GUARD + sl0.shape[0]]
    ins = [x.to(DEV) for x in (idx, qsl, nct)]
    before = [x.clone() for x in ins]
    if direct:
        mod._prepare_pos_seq_lens_kernel[(N + 1,)](
            pos_d, sl_d, ins[0], ins[1], ins[2], M_arg, BLOCK_SIZE=1024)
        M = M_arg
    else:
        assert idx.shape[0] == N and sl0.shape[0] == M
        mod.prepare_pos_seq_lens(ins[0], ins[1], ins[2], pos_d, sl_d)
    torch.cuda.synchronize()
    got_pos = pos_d.cpu().to(torch.int64)
    got_sl = sl_d.cpu().to(torch.int64)
    exp_pos, exp_sl = expected(idx, qsl, nct, pos0, sl0, N, M)
    ok = post_holds(got_pos, got_sl, exp_pos, exp_sl)
    pc, sc = pp_d.cpu(), sp_d.cpu()
    guards_ok = (torch.equal(pc[:GUARD], pp[:GUARD]) and torch.equal(pc[GUARD + T:], pp[GUARD + T:])
                 and torch.equal(sc[:GUARD], sp[:GUARD])
                 and torch.equal(sc[GUARD + sl0.shape[0]:], sp[GUARD + sl0.shape[0]:]))
    inputs_ok = all(torch.equal(a, b) for a, b in zip(before, ins))
    rej, tried = negatives(idx, qsl, nct, pos0, sl0, N, M, exp_pos, exp_sl)
    neg_results.append((name, rej, tried))
    cells = int(qsl[N].item() - qsl[0].item()) + max(N, M)
    results.append((name, ok, guards_ok and inputs_ok, cells))
    print(f"{name}: N={N} M={M} T={T} tokens={int(qsl[N] - qsl[0])}{' (direct)' if direct else ''}: "
          f"Post exact {ok}, inputs and guards unchanged {guards_ok and inputs_ok}, "
          f"wrong outputs rejected {rej}/{tried}")


def host_case(name, query_lens, max_num_reqs, max_num_tokens, max_context, gen):
    idx, qsl, nct, pos, sl = mod.make_batch(query_lens, max_num_reqs, max_num_tokens, max_context, gen)
    N = len(query_lens)
    # the model runner passes query_start_loc[: num_reqs_padded + 1]; the kernel reads N + 1
    run(name, idx, qsl[:N + 1].contiguous(), nct, pos, sl, N, max_num_reqs)
    return idx, qsl, nct, pos, sl


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup: the host's test cases (source lines 145-164), same generator.
    for name, query_lens, M, T, ctx in [
        ("decode", [1] * 8, 256, 8192, 4096),
        ("prefill", [1, 1023, 1024, 1025, 3000, 0, 17], 16, 8192, 2048),
        ("full batch", [2] * 128, 128, 512, 1000),
        ("one request", [5], 1, 16, 100),
        ("long padding", [3, 1, 2], 4096, 64, 100),
        ("spec decode", [4] * 64, 1024, 8192, 32768),
        ("serving", [1] * 1000 + [299] * 24, 1024, 8192, 131072),
        ("long prefill", [2 ** 20], 256, 2 ** 20, 2 ** 21),
    ]:
        host_case(f"setup {name}", query_lens, M, T, ctx, gen)
    g = torch.Generator().manual_seed(1)
    # vLLM's large serving sizes: 8192 requests, 2^22 positions (setup_serving_max).
    lens = torch.randint(0, 1024, (8192,), generator=g)
    lens = (lens * (2 ** 22) // max(int(lens.sum()), 1)).tolist()
    host_case("setup serving max", lens, 8192, 2 ** 22, 2 ** 22 + 1, g)

    # Edges of Inst.ok and Pre.
    # No request: one program, the zeroing one (N = 0).
    host_case("edge no request", [], 300, 16, 10, g)
    host_case("edge no request, max_num_reqs 1", [], 1, 16, 10, g)
    # Every request empty (qlen 0): only seq_lens.
    host_case("edge empty queries", [0] * 50, 64, 16, 1 << 22, g)
    # The largest values Pre allows: computed tokens 2^22 and 2^22 tokens in the batch.
    idx = torch.zeros(1, dtype=torch.int64)
    nct = torch.tensor([2 ** 22], dtype=torch.int32)
    qsl = torch.tensor([0, 2 ** 22], dtype=torch.int32)
    pos0 = torch.randint(-2 ** 40, 2 ** 40, (2 ** 22,), generator=g, dtype=torch.int64)
    sl0 = torch.randint(-2 ** 31, 2 ** 31, (4,), generator=g, dtype=torch.int64).to(torch.int32)
    run("edge computed 2^22, 2^22 tokens", idx, qsl, nct, pos0, sl0, 1, 4)
    # Many trips of the zeroing loop: max_num_reqs = 2^24 (within Inst.ok, beyond real sizes).
    idx, qsl, nct, pos0, sl0 = mod.make_batch([7, 0, 2000], 2 ** 24, 4096, 5000, g)
    run("edge max_num_reqs 2^24", idx, qsl[:4].contiguous(), nct, pos0, sl0, 3, 2 ** 24)
    # Slots shared by several requests, and a larger num_computed_tokens.
    idx = torch.tensor([3, 3, 0, 7, 3], dtype=torch.int64)
    nct = torch.randint(0, 2 ** 22, (100,), generator=g, dtype=torch.int32)
    qsl = torch.tensor([0, 5, 1030, 1030, 3000, 3001], dtype=torch.int32)
    pos0 = torch.randint(-2 ** 40, 2 ** 40, (3500,), generator=g, dtype=torch.int64)
    sl0 = torch.randint(-2 ** 31, 2 ** 31, (9,), generator=g, dtype=torch.int64).to(torch.int32)
    run("edge shared slots", idx, qsl, nct, pos0, sl0, 5, 9)
    # Direct launches with the host's launch line: query_start_loc starting above 0, a longer
    # idx_mapping and query_start_loc than the batch, max_num_reqs below the buffer's length.
    idx = torch.randperm(64, generator=g)[:40].to(torch.int64)
    nct = torch.randint(0, 2 ** 20, (64,), generator=g, dtype=torch.int32)
    q = torch.randint(0, 600, (21,), generator=g).cumsum(0) + 777
    qsl = torch.cat([q, torch.full((5,), int(q[-1]))]).to(torch.int32)
    pos0 = torch.randint(-2 ** 40, 2 ** 40, (int(q[-1]) + 100,), generator=g, dtype=torch.int64)
    sl0 = torch.randint(-2 ** 31, 2 ** 31, (3000,), generator=g, dtype=torch.int64).to(torch.int32)
    run("direct start 777", idx, qsl, nct, pos0, sl0, 20, 3000, direct=True, M_arg=2500)
    # max_num_reqs below num_reqs (Inst.ok allows it when seq_lens holds the batch): the zeroing
    # loop runs no trip.
    run("direct max_num_reqs < num_reqs", idx, qsl, nct, pos0, sl0, 20, 3000, direct=True, M_arg=7)
    run("direct max_num_reqs 1", idx, qsl, nct, pos0, sl0, 20, 3000, direct=True, M_arg=1)
    run("direct max_num_reqs 0", idx, qsl, nct, pos0, sl0, 20, 3000, direct=True, M_arg=0)

    bad = [r for r in results if not (r[1] and r[2])]
    bad += [r for r in neg_results if r[1] != r[2]]
    if sum(r[2] for r in neg_results) == 0:
        bad.append("no negative test")
    print(f"negative tests: {sum(r[1] for r in neg_results)} of {sum(r[2] for r in neg_results)} wrong "
          f"outputs rejected")
    print(f"{len(results)} launches, {sum(r[3] for r in results)} output cells checked; "
          f"worst error / tolerance: 0 / 0 (exact) on every output cell; frame held on every launch"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
