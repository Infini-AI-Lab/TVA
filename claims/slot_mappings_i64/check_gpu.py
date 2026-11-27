"""GPU check of the claim for `slot_mappings_i64`.

Runs the kernel's own host (`source/block_table.py`, `BlockTables.compute_slot_mappings`) on the
setup's launches (vLLM's model runner with one group and with a hybrid model's groups, decode
context parallelism, a dummy batch, a speculator's wider output) and direct launches at the edges
of `Inst.ok` and `Pre`, and checks `Post` exactly as `Claim.lean` states it:

* for every group `g`, request `b` and token `t` of `b`, element `(g, t)` of `slot_mappings` is
  `slot d init g b t`: `block_table[g][idx_mapping[b], p' // B] * B + p' % B` (`p'` the token's
  local position) when the group uses slot mappings, the request is real and the token is the
  rank's, and `-1` otherwise;
* every element `(g, t)` from `query_start_loc[num_reqs]` to `max_num_tokens` is `-1`;
* the frame: every other byte is unchanged (the inputs; the output's elements before
  `query_start_loc[0]`; in direct launches, the columns past `max_num_tokens` of a wider row and
  guard regions around the output, all from random contents).

Negative tests: outputs a wrong kernel would leave (the context-parallel local position ignored,
the rank's filter dropped, dummy requests mapped through a row, the groups without slot mappings
mapped, the padding started one token late) must fail `Post`. Integers only: exact, no tolerance.
Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "block_table.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("block_table_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
PAD = mod.PAD_SLOT_ID
I32 = 2 ** 31


def local_pos(p, B, C, I, r, sh):
    """`cpLocal` and `cpPos` of `Claim.lean`, on a tensor of nonnegative positions (any
    positions: the caller masks the others)."""
    if C == 1 or not sh:
        return torch.ones_like(p, dtype=torch.bool), p
    V = B * C
    vbo = torch.remainder(p, V)
    local = torch.div(vbo, I, rounding_mode="floor") % C == r
    lp = torch.div(p, V, rounding_mode="floor") * B + torch.div(vbo, I * C, rounding_mode="floor") * I + vbo % I
    return local, lp


def expected(t, idx, qsl, pos, init_rows, n, wrong=None):
    """`Post`: the output rows (`[G, S]`, `S` the row stride) the claim requires, from their
    initial contents. `wrong` names a negative test: the output of a kernel with that mistake."""
    out = init_rows.clone().cpu()
    idx, qsl, pos = idx.cpu().long(), qsl.cpu().long(), pos.cpu().long()
    R = idx.numel()
    q0, qR = int(qsl[0]), int(qsl[R])
    tok = torch.arange(q0, qR)
    req = torch.searchsorted(qsl[: R + 1], tok, right=True) - 1
    C, I, r = t.cp_size, t.cp_interleave, t.cp_rank
    for g in range(t.num_kv_cache_groups):
        B = t.block_sizes[g]
        en = t._slot_mapping_enabled[g] or wrong == "groups without slot mappings mapped"
        sh = bool(t.dcp_sharded[g]) and wrong != "local position ignored"
        bt = t.block_tables[g].cpu().reshape(-1).long()
        st = t.block_tables[g].stride(0)
        if tok.numel() > 0:
            p = pos[tok]
            ix = idx[req]
            real = ix >= 0
            if wrong == "dummy requests through row 0":
                ix = torch.clamp(ix, min=0)
                real = torch.ones_like(real)
            local, lp = local_pos(torch.where(real, p, torch.zeros_like(p)), B, C, I, r, sh)
            if wrong == "rank filter dropped":
                local = torch.ones_like(local)
            has = real & local & bool(en)
            k = torch.where(has, ix * st + torch.div(lp, B, rounding_mode="floor"), torch.zeros_like(ix))
            if wrong is not None:  # a wrong kernel's index may leave the table: any entry will do
                k = torch.clamp(k, 0, bt.numel() - 1)
            slot = torch.where(has, bt[k] * B + lp % B, torch.full_like(ix, PAD))
            out[g, q0:qR] = slot
        start = qR + (1 if wrong == "padding one late" else 0)
        if start < n:
            out[g, start:n] = PAD
    return out


def check_pre(t, idx, qsl, pos, n):
    """`Inst.ok`'s per-group conditions and `Pre`, from the host's inputs."""
    C, I, r = t.cp_size, t.cp_interleave, t.cp_rank
    assert C >= 1 and I >= 1 and I * C < I32 and 0 <= r < I32
    assert all(1 <= B and B * C < I32 for B in t.block_sizes)
    idx, qsl, pos = idx.cpu().long(), qsl.cpu().long(), pos.cpu().long()
    R = idx.numel()
    assert R <= 8192 and n <= 2 ** 22
    q = qsl[: R + 1]
    assert int(q[0]) >= 0 and bool((q[1:] >= q[:-1]).all()) and int(q[R]) <= n and int(q[R]) <= pos.numel()
    for b in range(R):
        if idx[b] < 0 or q[b] == q[b + 1]:
            continue
        for g in range(t.num_kv_cache_groups):  # RowOk
            assert int(idx[b]) * t.block_tables[g].stride(0) < t.block_tables[g].numel()
        p = pos[q[b]: q[b + 1]]
        for g in range(t.num_kv_cache_groups):  # TokOk
            if not t._slot_mapping_enabled[g]:
                continue
            B = t.block_sizes[g]
            assert bool((p >= 0).all())
            local, lp = local_pos(p, B, C, I, r, bool(t.dcp_sharded[g]))
            k = int(idx[b]) * t.block_tables[g].stride(0) + torch.div(lp, B, rounding_mode="floor")
            k = k[local]
            assert bool((k < t.block_tables[g].numel()).all())
            e = t.block_tables[g].cpu().reshape(-1).long()[k] * B
            assert bool((e >= -I32).all()) and bool((e < I32).all())


def make_tables(block_sizes, cols, rows, en, sh, n, C=1, r=0, I=1, gen=None):
    """The host's `BlockTables`, its block tables random over the whole `int32` range."""
    t = mod.BlockTables(block_sizes, rows, n, cols, DEV, cp_size=C, cp_rank=r, cp_interleave=I,
                        slot_mapping_enabled=en, dcp_sharded=sh)
    for g, bt in enumerate(t.block_tables):
        bt.copy_(torch.randint(-I32, I32, bt.shape, generator=gen, dtype=torch.int32).to(DEV))
    return t


def make_batch(t, rows, lens, gen, dummy=(), q0=0):
    """A batch: request `b` has `lens[b]` tokens at positions `ctx_b, …, ctx_b + lens[b] - 1`
    (`ctx_b` random, the request's blocks within every group's row), its row of each block table
    filled with block ids whose slots fit `int32`; the requests in `dummy` get `idx_mapping = -1`
    and random positions over the whole `int64` range."""
    R = len(lens)
    C = t.cp_size
    cap = min(t.block_tables[g].shape[1] * t.block_sizes[g] *
              (C if (C > 1 and bool(t.dcp_sharded[g])) else 1)
              for g in range(t.num_kv_cache_groups))
    reqs = torch.randperm(rows, generator=gen)[:R]
    idx = reqs.clone().long()
    for b in dummy:
        idx[b] = -1
    pos = []
    for b, L in enumerate(lens):
        if b in dummy:
            pos.append(torch.randint(-2 ** 62, 2 ** 62, (L,), generator=gen))
            continue
        assert L <= cap
        c = int(torch.randint(0, cap - L + 1, (1,), generator=gen))
        pos.append(torch.arange(c, c + L))
        for g, bt in enumerate(t.block_tables):
            B = t.block_sizes[g]
            hi = (I32 - 1) // B
            bt[int(reqs[b])] = torch.randint(0, hi, (bt.shape[1],), generator=gen,
                                             dtype=torch.int64).to(DEV, torch.int32)
            ext = torch.tensor([0, hi - 1], dtype=torch.int32)[: bt.shape[1]]
            bt[int(reqs[b]), : ext.numel()] = ext.to(DEV)  # the extremes
    qsl = torch.tensor([q0] + [q0 + s for s in torch.tensor(lens, dtype=torch.long).cumsum(0).tolist()],
                       dtype=torch.int32)
    pos = torch.cat([torch.zeros(q0, dtype=torch.long)] + pos) if (pos or q0) else torch.zeros(0, dtype=torch.long)
    return idx.to(DEV), qsl.to(DEV), pos.to(DEV)


results = []
negatives = []


def check_negatives(name, got, t, idx, qsl, pos, init, n):
    for what in ["local position ignored", "rank filter dropped", "dummy requests through row 0",
                 "groups without slot mappings mapped", "padding one late"]:
        w = expected(t, idx, qsl, pos, init, n, wrong=what)
        if torch.equal(w, expected(t, idx, qsl, pos, init, n)):
            continue  # the mistake changes nothing on this launch
        negatives.append((f"{name}: {what}", not torch.equal(got, w)))


def inputs_of(t, idx, qsl, pos):
    return [idx, qsl, pos, t.block_table_ptrs, t.block_table_strides, t.block_sizes_tensor,
            t.slot_mapping_enabled, t.dcp_sharded] + list(t.block_tables)


def run_host(name, t, idx, qsl, pos, n_padded, out=None, gen=None):
    """`compute_slot_mappings` on the host's own output (`self.slot_mappings`, or `out`), its
    initial contents random."""
    sm = t.slot_mappings if out is None else out
    sm.copy_(torch.randint(-2 ** 62, 2 ** 62, sm.shape, generator=gen).to(DEV))
    init = sm.clone()
    n = sm.shape[1]
    check_pre(t, idx, qsl, pos, n)
    ins = inputs_of(t, idx, qsl, pos)
    before = [x.clone() for x in ins]
    t.compute_slot_mappings(idx, qsl, pos, n_padded, out=out)
    torch.cuda.synchronize()
    exp = expected(t, idx, qsl, pos, init, n)
    got = sm.cpu()
    check_negatives(name, got, t, idx, qsl, pos, init, n)
    ok = torch.equal(got, exp)
    frame = all(torch.equal(a, b) for a, b in zip(before, ins))
    G = t.num_kv_cache_groups
    written = G * (n - int(qsl[0]))
    results.append((name, ok, frame, written))
    print(f"{name}: {G} groups, {idx.numel()} requests, {int(qsl[idx.numel()])} tokens, row {n}, "
          f"CP {t.cp_size}/{t.cp_rank}/{t.cp_interleave}: exact {ok}, inputs unchanged {frame}")


def run_direct(name, t, idx, qsl, pos, n, S, guard=4096, gen=None):
    """The host's launch line (source lines 242-259) on an output `[G, n]` with row stride
    `S ≥ n`, carved from a random-filled pool with guards."""
    G = t.num_kv_cache_groups
    pool = torch.randint(-2 ** 62, 2 ** 62, (guard + G * S + guard,), generator=gen).to(DEV)
    pool0 = pool.clone()
    rows = pool[guard: guard + G * S].view(G, S)
    out = rows[:, :n]
    init = rows.clone()
    check_pre(t, idx, qsl, pos, n)
    ins = inputs_of(t, idx, qsl, pos)
    before = [x.clone() for x in ins]
    R = idx.shape[0]
    mod._compute_slot_mappings_kernel[(G, R + 1)](
        out.shape[1], idx, qsl, pos, t.block_table_ptrs, t.block_table_strides,
        t.block_sizes_tensor, t.slot_mapping_enabled, t.dcp_sharded, out, out.stride(0),
        t.cp_rank, CP_SIZE=t.cp_size, CP_INTERLEAVE=t.cp_interleave, PAD_ID=PAD,
        TRITON_BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    exp = expected(t, idx, qsl, pos, init, n)
    got = rows.cpu()
    check_negatives(name, got, t, idx, qsl, pos, init, n)
    ok = torch.equal(got, exp)
    frame = (torch.equal(pool[:guard], pool0[:guard]) and torch.equal(pool[guard + G * S:], pool0[guard + G * S:])
             and all(torch.equal(a, b) for a, b in zip(before, ins)))
    results.append((name, ok, frame, G * (n - int(qsl[0]))))
    print(f"{name} (guarded direct launch): {G} groups, {R} requests, row {n} of stride {S}, "
          f"start {int(qsl[0])}, CP {t.cp_size}/{t.cp_rank}/{t.cp_interleave}: exact {ok} (other "
          f"elements kept), inputs and guards unchanged {frame}")


def rand_lens(gen, R, hi, decode_frac=0.5):
    lens = torch.randint(1, hi + 1, (R,), generator=gen)
    dec = torch.rand(R, generator=gen) < decode_frac
    lens[dec] = 1
    return lens.tolist()


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup. vLLM's model runner (model_runner.py:718-730, 1609-1614), one attention group:
    # block size 16, 1024 rows of 8192 blocks (128K-token contexts), 256 requests, 8192 tokens.
    t = make_tables([16], [8192], 1024, [True], [True], 8192, gen=gen)
    lens = rand_lens(gen, 256, 64)
    lens[0] = 8192 - sum(lens[1:])  # one long prefill chunk fills the step
    run_host("setup runner, one group", t, *make_batch(t, 1024, lens, gen), 8192, gen=gen)
    # A hybrid model's groups: full attention (16), sliding window (32), a recurrent group
    # without slot mappings (block size 1056, one block per request).
    t = make_tables([16, 32, 1056], [2048, 1024, 1], 512, [True, True, False], [True, True, False],
                    16384, gen=gen)
    lens = rand_lens(gen, 300, 100)
    run_host("setup runner, hybrid groups", t, *make_batch(t, 512, lens, gen), 16384, gen=gen)
    # Decode context parallelism (cp_size = dcp_size, cp_rank = dcp_rank, cp_interleave =
    # cp_kv_cache_interleave_size): two ranks, interleave 1; eight ranks, interleave 64 with a
    # replicated (unsharded) group; four ranks, interleave 16, rank 1 (a constant `1`).
    for C, r, I, sh in [(2, 0, 1, [True]), (2, 1, 1, [True]), (8, 5, 64, [True, False]),
                        (4, 1, 16, [True, True])]:
        G = len(sh)
        t = make_tables([64] * G, [512] * G, 256, [True] * G, sh, 8192, C=C, r=r, I=I, gen=gen)
        lens = rand_lens(gen, 100, 150)
        run_host(f"setup dcp {C} rank {r} interleave {I}", t, *make_batch(t, 256, lens, gen), 8192, gen=gen)
    # A dummy batch (InputBatch.make_dummy, input_batch.py:140-200): idx_mapping = arange,
    # positions 0, lengths split evenly; and a padded batch with dummy requests (-1).
    t = make_tables([16, 64], [256, 64], 128, [True, True], [True, True], 4096, gen=gen)
    for g, bt in enumerate(t.block_tables):  # block ids the runner left in the rows: valid ids
        bt.copy_(torch.randint(0, (I32 - 1) // t.block_sizes[g], bt.shape, generator=gen).to(DEV, torch.int32))
    idx = torch.arange(100, dtype=torch.int64, device=DEV)
    lens = [40] * 96 + [41] * 4
    qsl = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device=DEV)
    pos = torch.zeros(4004, dtype=torch.int64, device=DEV)
    run_host("setup dummy batch", t, idx, qsl, pos, 4096, gen=gen)
    lens = rand_lens(gen, 64, 30)
    run_host("setup dummy requests (-1)", t, *make_batch(t, 128, lens, gen, dummy=(0, 5, 63)), 4096, gen=gen)
    # A speculator's output wider than the runner's (standalone_ar/speculator.py:94-98, 210-216):
    # [G, max_num_tokens + max_num_reqs].
    t = make_tables([16, 32], [1024, 512], 256, [True, True], [True, True], 4096, gen=gen)
    out = torch.empty(2, 4096 + 256, dtype=torch.int64, device=DEV)
    lens = rand_lens(gen, 200, 40, decode_frac=0.0)
    run_host("setup speculator expanded output", t, *make_batch(t, 256, lens, gen), sum(lens), out=out, gen=gen)
    # The source's own test (source lines 299-326).
    for C, r, I in [(1, 0, 1), (2, 1, 1), (4, 2, 8)]:
        t = make_tables([16, 64], [64, 16], 8, [True, True], [True, False], 4096, C=C, r=r, I=I, gen=gen)
        run_host(f"setup source test CP {C}", t, *make_batch(t, 8, [5, 0, 1000, 3], gen, dummy=(3,)), 2048,
                 gen=gen)

    # Edges: no request (only the padding programs); the batch filling the row (no padding);
    # max_num_tokens 1 (a constant `1`); 8192 requests; 2^22 tokens; block size 1;
    # a nonzero query_start_loc[0]; a wider output row; a large CP group.
    t = make_tables([16, 32], [64, 64], 16, [True, True], [True, True], 3000, gen=gen)
    run_direct("edge no request", t, torch.zeros(0, dtype=torch.int64, device=DEV),
               torch.tensor([0], dtype=torch.int32, device=DEV), torch.zeros(0, dtype=torch.int64, device=DEV),
               3000, 3000, gen=gen)
    run_direct("edge batch fills the row", t, *make_batch(t, 16, [1000, 24, 976], gen), 2000, 2000, gen=gen)
    run_direct("edge max_num_tokens 1", t, *make_batch(t, 16, [1], gen), 1, 1, gen=gen)
    run_direct("edge query_start_loc[0] 777, wider rows", t, *make_batch(t, 16, [300, 5, 1], gen, q0=777),
               2100, 2400, gen=gen)
    t = make_tables([16], [8], 8192, [True], [True], 8192, gen=gen)
    run_direct("edge 8192 requests", t, *make_batch(t, 8192, [1] * 8192, gen), 8192, 8192, gen=gen)
    t = make_tables([256], [2 ** 14], 4, [True], [True], 2 ** 22, gen=gen)
    run_direct("edge 2^22 tokens", t, *make_batch(t, 4, [2 ** 21, 2 ** 20, 2 ** 20 - 5], gen), 2 ** 22,
               2 ** 22, gen=gen)
    t = make_tables([1, 7], [4096, 4096], 4, [True, True], [True, True], 4096, C=3, r=2, I=1, gen=gen)
    run_direct("edge block sizes 1 and 7, CP 3", t, *make_batch(t, 4, [1000, 2000], gen, dummy=(1,)), 4096,
               4096, gen=gen)
    t = make_tables([128], [64], 32, [True], [True], 2048, C=64, r=63, I=128, gen=gen)
    run_direct("edge CP 64 interleave 128 rank 63", t, *make_batch(t, 32, [2000], gen), 2048, 2048, gen=gen)

    bad = [r for r in results if not (r[1] and r[2])] + [n for n in negatives if not n[1]]
    print(f"{len(results)} launches, {sum(r[3] for r in results)} written elements checked; worst "
          f"error / tolerance: 0 / 0 (exact) on every element; frame held on every launch; "
          f"{len(negatives)} wrong outputs, every one rejected by Post"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
