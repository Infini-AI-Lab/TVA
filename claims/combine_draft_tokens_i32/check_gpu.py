"""GPU check of the claim for `combine_draft_tokens_i32`.

Runs the kernel's own host (`source/combine_draft_tokens.py`) on the setup's launches (decoding
without drafts, MTP, EAGLE, n-gram and long drafts with `num_new_sampled_tokens = 1`, and
launches with `num_new_sampled_tokens = 0`, on batches that mix decoding requests, prefill chunks,
prompt-tail chunks and padded query slots), on direct launches at the edges of `Inst.ok` and `Pre`
(a `BLOCK_SIZE` above `next_power_of_2(S + N)`, `num_logits = BLOCK_SIZE`, repeated
request-state rows, `int64` tokens with high bits, negative `prefill_len` entries, `seq_lens`
near `2^31`, stride `1`, an empty batch), and checks `Post` exactly as `Claim.lean` states it:

* `logits_indices[cu_num_logits[b] + j] = query_start_loc[b + 1] - nlog + j` for `j < nlog`;
* for a request past its prefill (`seq_lens[b] > prefill_len[idx_mapping[b]]`): with `N = 1` and
  `seq_lens[b] - nlog >= prefill_len[idx_mapping[b]]`, `input_ids[qend - nlog]` holds the `int32`
  truncation of `last_sampled_tokens[idx_mapping[b]]`, and `input_ids[qend - ndraft + j]` the
  truncation of `draft_tokens[idx_mapping[b], j]` for `j < ndraft`;
* the frame: every other `input_ids` slot keeps its initial (random) value, the inputs are
  unchanged and, in the guarded direct launches, no byte around the two outputs changes.

Every value is an integer, so the check is exact (no tolerance). Negative tests: wrong outputs
(prefill requests treated as decoding, the prompt-tail condition ignored, the last sampled token
dropped, logits indices shifted by one, drafts read from the wrong row, the `int64` tokens not
truncated) that `Post` must reject. Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "combine_draft_tokens.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("combine_draft_tokens_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")


def trunc32(x):
    """`Int.bmod x (2^32)` of a Python int: the `int32` two's-complement truncation."""
    x &= 0xFFFFFFFF
    return x - 2 ** 32 if x >= 2 ** 31 else x


class Case:
    """A launch's inputs on the CPU (Python ints), with the `Inst` fields `B`, `N`, `dstride`."""

    def __init__(self, idx, last, qsl, seq, pl, draft, cu, ids0, N):
        self.idx, self.last, self.qsl, self.seq, self.pl = idx, last, qsl, seq, pl
        self.draft, self.cu, self.ids0, self.N = draft, cu, ids0, N  # draft: [R, S] int64 tensor
        self.B = len(idx)

    def check_pre(self, BLOCK):
        """`Pre` (`ReqOk` for every request, and the ends of `cu_num_logits`, `query_start_loc`)."""
        R, S = self.draft.shape
        dstride = self.draft.stride(0)
        assert 0 <= self.cu[0] and 0 <= self.qsl[0] and self.qsl[self.B] <= len(self.ids0)
        for b in range(self.B):
            i = self.idx[b]
            nl = self.cu[b + 1] - self.cu[b]
            nd = nl - self.N
            assert 0 <= i < len(self.pl) and i < len(self.last) and self.seq[b] >= 0
            assert self.N <= nl <= BLOCK and self.qsl[b] + nl <= self.qsl[b + 1]
            assert nd <= 0 or i * dstride + nd <= R * S

    def expected(self):
        """`Post`'s values: `logits_indices` (a list) and `input_ids` (a list, the initial values
        where the launch writes nothing)."""
        dstride = self.draft.stride(0)
        flat = self.draft.reshape(-1).tolist() if self.draft.numel() else []
        logits = [None] * self.cu[self.B]
        ids = list(self.ids0)
        for b in range(self.B):
            i = self.idx[b]
            nl = self.cu[b + 1] - self.cu[b]
            nd = nl - self.N
            qe = self.qsl[b + 1]
            for j in range(nl):
                logits[self.cu[b] + j] = qe - nl + j
            if self.pl[i] < self.seq[b]:  # Decode
                if self.N > 0 and self.pl[i] <= self.seq[b] - nl:  # LastOn
                    ids[qe - nl] = trunc32(self.last[i])
                for j in range(max(nd, 0)):
                    ids[qe - nd + j] = trunc32(flat[i * dstride + j])
        return logits, ids

    def tensors(self):
        R = len(self.pl)
        return dict(
            idx=torch.tensor(self.idx, dtype=torch.int64, device=DEV),
            last=torch.tensor(self.last, dtype=torch.int64, device=DEV).reshape(R, 1),
            qsl=torch.tensor(self.qsl, dtype=torch.int32, device=DEV),
            seq=torch.tensor(self.seq, dtype=torch.int32, device=DEV),
            pl=torch.tensor(self.pl, dtype=torch.int32, device=DEV),
            draft=self.draft.to(DEV),
            cu=torch.tensor(self.cu, dtype=torch.int32, device=DEV),
        )


def make_case(g, B, R, S, N, T_extra=37, pad_reqs=3, kinds=("decode", "prefill", "tail", "pad"),
              high_bits=True, repeat_rows=False):
    """A batch as vLLM's `prepare_inputs` builds it: request `b` (row `idx[b]`, distinct rows
    unless `repeat_rows`) is a decoding request (query `nd + N` tokens, its drafts after its last
    sampled token), a prefill chunk (`seq_lens <= prefill_len`), a prompt-tail chunk (past the
    prefill, but its first logit's position before `prefill_len`), or a chunk with padded query
    slots (more query rows than logits); `query_start_loc` is padded with `pad_reqs` entries
    equal to the token count, as for full graphs; `input_ids` has `T_extra` slots more."""
    def rint(lo, hi):
        return int(torch.randint(lo, hi + 1, (1,), generator=g))

    if repeat_rows:
        idx = torch.randint(0, R, (B,), generator=g).tolist()
    else:
        idx = torch.randperm(R, generator=g)[:B].tolist()
    pl = torch.randint(1, 4096, (R,), generator=g).tolist()
    qsl, cu, seq = [0], [0], []
    for b in range(B):
        kind = kinds[rint(0, len(kinds) - 1)]
        nd = rint(0, S)
        nl = nd + N
        i = idx[b]
        if kind == "decode":
            qlen = nl
            seq_b = pl[i] + rint(nl, 3000)  # past the prefill, first logit past it too
        elif kind == "prefill":
            qlen = max(nl, rint(1, 600))
            seq_b = rint(min(qlen, pl[i]), pl[i])  # seq_lens <= prefill_len
        elif kind == "tail":
            qlen = max(nl, rint(1, 600))
            seq_b = pl[i] + rint(1, max(1, nl - 1)) if nl >= 2 else pl[i] + 1
            # seq - nl < pl < seq: the last sampled token's slot is a prompt slot
        else:  # padded query slots
            qlen = nl + rint(1, 20)
            seq_b = pl[i] + rint(nl, 3000)
        qsl.append(qsl[-1] + qlen)
        cu.append(cu[-1] + nl)
        seq.append(max(seq_b, 0))
    T = qsl[-1]
    qsl += [T] * pad_reqs
    seq += [0] * pad_reqs
    hi = 2 ** 40 if high_bits else 1
    last = (torch.randint(0, 200000, (R,), generator=g)
            + hi * torch.randint(-1000, 1000, (R,), generator=g)).tolist()
    draft = (torch.randint(0, 200000, (R, S), generator=g, dtype=torch.int64)
             + hi * torch.randint(-1000, 1000, (R, S), generator=g, dtype=torch.int64))
    ids0 = torch.randint(-2 ** 31, 2 ** 31, (T + T_extra,), generator=g, dtype=torch.int64).tolist()
    return Case(idx, last, qsl, seq, pl, draft, cu, ids0, N)


results = []
neg_results = []


def negatives(c, exp_logits, exp_ids):
    """Wrong outputs `Post` must reject; returns how many were rejected and how many tried."""
    wrong = []
    flat = c.draft.reshape(-1).tolist() if c.draft.numel() else []
    dstride = c.draft.stride(0)
    # (1) every request treated as past its prefill; (2) the prompt-tail condition ignored
    for variant in ("all_decode", "no_tail_check"):
        ids = list(c.ids0)
        for b in range(c.B):
            i = c.idx[b]
            nl = c.cu[b + 1] - c.cu[b]
            nd = nl - c.N
            qe = c.qsl[b + 1]
            dec = True if variant == "all_decode" else c.pl[i] < c.seq[b]
            if dec:
                if c.N > 0 and (variant == "no_tail_check" or c.pl[i] <= c.seq[b] - nl):
                    ids[qe - nl] = trunc32(c.last[i])
                for j in range(max(nd, 0)):
                    ids[qe - nd + j] = trunc32(flat[i * dstride + j])
        wrong.append((exp_logits, ids))
    # (3) the last sampled token dropped
    ids = list(exp_ids)
    for b in range(c.B):
        i = c.idx[b]
        nl = c.cu[b + 1] - c.cu[b]
        if c.pl[i] < c.seq[b] and c.N > 0 and c.pl[i] <= c.seq[b] - nl:
            ids[c.qsl[b + 1] - nl] = c.ids0[c.qsl[b + 1] - nl]
    wrong.append((exp_logits, ids))
    # (4) logits indices shifted by one
    wrong.append(([v + 1 for v in exp_logits], exp_ids))
    # (5) drafts and tokens not truncated to int32 (the int64 values)
    ids = list(exp_ids)
    for b in range(c.B):
        i = c.idx[b]
        nl = c.cu[b + 1] - c.cu[b]
        nd = nl - c.N
        qe = c.qsl[b + 1]
        if c.pl[i] < c.seq[b]:
            for j in range(max(nd, 0)):
                ids[qe - nd + j] = flat[i * dstride + j]
    wrong.append((exp_logits, ids))
    # (6) drafts read from the next row
    if c.draft.shape[0] > 1:
        ids = list(exp_ids)
        R = c.draft.shape[0]
        for b in range(c.B):
            i = c.idx[b]
            nl = c.cu[b + 1] - c.cu[b]
            nd = nl - c.N
            qe = c.qsl[b + 1]
            if c.pl[i] < c.seq[b]:
                for j in range(max(nd, 0)):
                    ids[qe - nd + j] = trunc32(flat[((i + 1) % R) * dstride + j])
        wrong.append((exp_logits, ids))
    return [w for w in wrong if w != (exp_logits, exp_ids)]


def rejected(wrong, got_logits, got_ids):
    """How many wrong outputs the exact check of `Post` rejects on the kernel's result (each
    must differ from what the kernel wrote), and how many were tried."""
    return sum(1 for w in wrong if w != (got_logits, got_ids)), len(wrong)


def run_host(name, c):
    """The host's launch, `combine_sampled_and_draft_tokens`, with vLLM's arguments."""
    S = c.draft.shape[1]
    BLOCK = triton.next_power_of_2(S + c.N)
    c.check_pre(BLOCK)
    t = c.tensors()
    ids = torch.tensor(c.ids0, dtype=torch.int64).to(torch.int32).to(DEV)
    before = {k: v.clone() for k, v in t.items()}
    L = c.cu[c.B]
    got = mod.combine_sampled_and_draft_tokens(ids, t["idx"], t["last"], t["qsl"], t["seq"], t["pl"],
                                               t["draft"], t["cu"], L, c.N)
    exp_logits, exp_ids = c.expected()
    got_l, got_i = got.cpu().tolist(), ids.cpu().to(torch.int64).tolist()
    ok_l = got.dtype == torch.int64 and got_l == exp_logits
    ok_i = got_i == exp_ids
    frame = all(torch.equal(before[k], t[k]) for k in t)
    neg = rejected(negatives(c, exp_logits, exp_ids), got_l, got_i)
    neg_results.append((name,) + neg)
    results.append((name, ok_l and ok_i, frame, L + len(exp_ids)))
    nwritten = sum(1 for a, b in zip(exp_ids, c.ids0) if a != b)
    print(f"{name}: B={c.B} S={S} N={c.N} BLOCK={BLOCK}: logits_indices exact {ok_l} ({L}), "
          f"input_ids exact {ok_i} ({len(exp_ids)} slots, {nwritten} written), inputs unchanged {frame}, "
          f"wrong outputs rejected {neg[0]}/{neg[1]}")


def run_guarded(name, c, BLOCK, guard=1024):
    """A direct launch with the host's launch line (source line 141) and a given `BLOCK_SIZE`, on
    outputs carved from sentinel pools: checks `Post` and that no byte around them changes."""
    c.check_pre(BLOCK)
    t = c.tensors()
    g = torch.Generator().manual_seed(11)
    L = c.cu[c.B]
    T = len(c.ids0)
    pool_i = torch.tensor([0] * guard + c.ids0 + [0] * guard, dtype=torch.int64)
    pool_i[:guard] = torch.randint(-2 ** 31, 2 ** 31, (guard,), generator=g)
    pool_i[guard + T:] = torch.randint(-2 ** 31, 2 ** 31, (guard,), generator=g)
    pool_i = pool_i.to(torch.int32).to(DEV)
    pool_l = torch.randint(-2 ** 62, 2 ** 62, (2 * guard + L,), generator=g, dtype=torch.int64).to(DEV)
    pi0, pl0 = pool_i.clone(), pool_l.clone()
    ids = pool_i[guard:guard + T]
    logits = pool_l[guard:guard + L]
    before = {k: v.clone() for k, v in t.items()}
    mod._combine_sampled_and_draft_tokens_kernel[(c.B,)](
        ids, t["idx"], t["last"], t["qsl"], t["seq"], t["pl"], t["draft"], t["draft"].stride(0),
        t["cu"], logits, NUM_NEW_SAMPLED_TOKENS=c.N, BLOCK_SIZE=BLOCK)
    exp_logits, exp_ids = c.expected()
    got_l, got_i = logits.cpu().tolist(), ids.cpu().to(torch.int64).tolist()
    ok = got_l == exp_logits and got_i == exp_ids
    frame = (torch.equal(pool_i[:guard], pi0[:guard]) and torch.equal(pool_i[guard + T:], pi0[guard + T:])
             and torch.equal(pool_l[:guard], pl0[:guard]) and torch.equal(pool_l[guard + L:], pl0[guard + L:])
             and all(torch.equal(before[k], t[k]) for k in t))
    neg = rejected(negatives(c, exp_logits, exp_ids), got_l, got_i)
    neg_results.append((name,) + neg)
    results.append((name, ok, frame, L + T))
    print(f"{name} (guarded direct launch, BLOCK={BLOCK}): B={c.B} S={c.draft.shape[1]} N={c.N}: "
          f"outputs exact {ok}, guards and inputs unchanged {frame}, wrong outputs rejected {neg[0]}/{neg[1]}")


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    # The setup's launches (SETUP.md): (name, B, R, S, N).
    for name, B, R, S, N, kinds in [
        ("setup decode b256", 256, 1024, 0, 1, ("decode", "prefill", "tail", "pad")),
        ("setup decode b1024", 1024, 1024, 0, 1, ("decode",)),
        ("setup single", 1, 1024, 0, 1, ("prefill",)),
        ("setup mtp1", 128, 1024, 1, 1, ("decode", "prefill", "tail", "pad")),
        ("setup eagle3", 64, 1024, 3, 1, ("decode", "prefill", "tail", "pad")),
        ("setup eagle7", 256, 1024, 7, 1, ("decode", "prefill", "tail", "pad")),
        ("setup ngram8", 512, 1024, 8, 1, ("decode", "tail")),
        ("setup spec128", 32, 256, 128, 1, ("decode", "prefill", "tail", "pad")),
        ("setup n0 s4", 64, 1024, 4, 0, ("decode", "prefill", "tail", "pad")),
        ("setup n0 s7", 128, 1024, 7, 0, ("decode", "prefill", "tail", "pad")),
        ("setup large", 8192, 8192, 3, 1, ("decode", "prefill", "tail", "pad")),
        ("setup empty", 0, 1024, 3, 1, ("decode",)),
    ]:
        run_host(name, make_case(g, B, R, S, N, kinds=kinds))

    # Edges of Inst.ok / Pre (direct launches).
    g = torch.Generator().manual_seed(1)
    # BLOCK_SIZE above next_power_of_2(S + N), and num_logits = BLOCK_SIZE exactly.
    run_guarded("edge BLOCK 64 > S + N", make_case(g, 300, 512, 5, 1), 64)
    c = make_case(g, 200, 512, 7, 1, kinds=("decode",))
    q = [0]
    for b in range(c.B):
        q.append(q[-1] + 8 + (b % 3))  # query rows: nl = 8, then 0 to 2 more
    ids0 = torch.randint(-2 ** 31, 2 ** 31, (q[-1] + 10,), generator=g, dtype=torch.int64).tolist()
    seq = [c.pl[c.idx[b]] + 10 + b for b in range(c.B)]
    c = Case(c.idx, c.last, q + [q[-1]] * 3, seq + [0] * 3, c.pl, c.draft,
             [8 * b for b in range(c.B + 1)], ids0, 1)  # every request has nl = 8 = BLOCK_SIZE
    run_guarded("edge nl = BLOCK 8", c, 8)
    # Repeated request-state rows (Pre allows them: rows are only read), N = 0 with nl = 0.
    run_guarded("edge repeated rows", make_case(g, 500, 64, 4, 1, repeat_rows=True), 8)
    run_guarded("edge N=0 nl may be 0", make_case(g, 400, 512, 3, 0), 4)
    # S = 1: stride 1, the compile-time constant 1.
    run_guarded("edge stride 1 (S=1)", make_case(g, 300, 512, 1, 1), 2)
    # S = 0: a [R, 0] draft table (stride 1), N = 1, no draft read.
    run_guarded("edge S=0", make_case(g, 300, 512, 0, 1), 1)
    # Negative prefill_len entries and seq_lens near 2^31 (Pre: seq_lens >= 0 only).
    c = make_case(g, 300, 512, 6, 1)
    for i in range(0, len(c.pl), 3):
        c.pl[i] = -int(torch.randint(1, 2 ** 31, (1,), generator=g))
    for b in range(0, c.B, 5):
        c.seq[b] = 2 ** 31 - 1 - (b % 4)
    c.pl[c.idx[1]] = 2 ** 31 - 1  # seq <= pl for request 1 whatever its seq
    run_guarded("edge prefill_len extremes", c, 8)
    # seq_lens = 0 (seq <= pl when pl >= 0)
    c = make_case(g, 100, 128, 2, 1)
    for b in range(c.B):
        c.seq[b] = 0
    run_guarded("edge seq_lens 0", c, 4)
    # An empty batch through a direct launch.
    run_guarded("edge empty batch", make_case(g, 0, 16, 3, 1), 4)

    bad = [r for r in results if not (r[1] and r[2])]
    bad += [r for r in neg_results if r[1] != r[2] or (r[2] == 0 and "empty" not in r[0])]
    print(f"negative tests: {sum(r[1] for r in neg_results)} of {sum(r[2] for r in neg_results)} wrong "
          f"outputs rejected")
    print(f"{len(results)} launches, {sum(r[3] for r in results)} output cells checked; "
          f"worst error / tolerance: 0 / 0 (exact) on every output cell; frame held on every launch"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
