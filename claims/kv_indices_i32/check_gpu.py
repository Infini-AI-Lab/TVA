"""GPU check of the prepared claim for `kv_indices_i32` (claims/README.md, check 4).

Runs the kernel's own host (`source/kv_indices.py`) on the setup's launches
(the host's test cases, with the same generator, and FlashInfer's sliding-window launches with
`kv_start_idx`) and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post` exactly as
`Claim.lean` states it:

* for every request `b` and `j < seq_lens[b]`, element `kv_indptr[b] + j` of `kv_indices` is
  element `req_pool_indices[b] * max_context_len + s[b] + j` of the table's storage, `s[b]` the
  request's `kv_start_idx[b]` (`0` without it);
* the frame: every other byte is unchanged (the inputs; in direct launches with the host's own
  launch line, the output's other elements and guard regions around it, from random contents).

Negative tests: outputs a wrong kernel would leave (the start ignored, the start off by one, the
last block of a request dropped) must fail `Post`. Integers only: exact, no tolerance. Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "kv_indices.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("kv_indices_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")


def starts(lens, start):
    """`sidx`: each request's first position (`0` without `kv_start_idx`)."""
    return torch.zeros_like(lens.long()) if start is None else start.long()


def check_pre(table, rpi, lens, indptr, out_len, start=None):
    """`Inst.ok`'s `kv_start_idx` conditions and `Pre`: `ReqOk` for every request, `kv_indptr` the
    running sums from a nonnegative start, within the output."""
    ctx = table.shape[1]
    if start is not None:
        assert start.dtype in (torch.int64, torch.int32) and start.numel() >= lens.numel()
        assert start.data_ptr() != 0
    s = starts(lens, start).cpu()
    rpi, lens, indptr = rpi.long().cpu(), lens.long().cpu(), indptr.long().cpu()
    assert bool((lens >= 0).all()) and bool((lens + 511 < 2 ** 31).all())
    assert bool((s >= 0).all()) and bool((s < 2 ** 31).all())
    nz = lens > 0
    assert bool((rpi[nz] * ctx + s[nz] >= 0).all())
    assert bool((rpi[nz] * ctx + s[nz] + lens[nz] <= table.numel()).all())
    assert bool((indptr[1:] == indptr[:-1] + lens).all())
    assert bool((indptr >= 0).all()) and bool((indptr <= out_len).all())


def expected(table, rpi, lens, indptr, init_out, start=None, drop_last_block=False):
    """`Post`, from the host's inputs: the output the claim requires, given its initial contents.
    With `drop_last_block`, the output of a kernel that skips each request's last block of 512
    (a negative test)."""
    ctx = table.shape[1]
    flat = table.reshape(-1)
    out = init_out.clone()
    st = starts(lens, start).cpu().tolist()
    rpi, lens, indptr = rpi.long().cpu().tolist(), lens.long().cpu().tolist(), indptr.long().cpu().tolist()
    for b in range(len(lens)):
        n = lens[b]
        if drop_last_block:
            n = ((n - 1) // 512) * 512 if n > 0 else 0
        if n > 0:
            s = rpi[b] * ctx + st[b]
            a = indptr[b]
            out[a:a + n] = flat[s:s + n]
    return out


negatives = []


def check_negatives(name, out, table, rpi, lens, indptr, init_out, start):
    """Wrong outputs `Post` must reject (each differs from `out` on some written element)."""
    wrong = []
    if start is not None and bool((start[:lens.numel()].long() != 0).any()) and int(lens.sum()) > 0:
        wrong.append(("start ignored", expected(table, rpi, lens, indptr, init_out, None)))
    if start is not None and int(lens.sum()) > 0:
        nz = (lens.long() > 0) & (start[:lens.numel()].long() + lens.long() < table.shape[1])
        if bool(nz.any()):
            wrong.append(("start + 1", expected(table, rpi, lens, indptr, init_out, start.long() + nz.long())))
    if bool((lens.long() > 512).any()):
        wrong.append(("last block dropped", expected(table, rpi, lens, indptr, init_out, start, True)))
    for what, w in wrong:
        rejected = not torch.equal(out, w)
        negatives.append((f"{name}: {what}", rejected))


results = []


def run_host(name, table, rpi, lens, start=None):
    ins = [table, rpi, lens] + ([] if start is None else [start])
    before = [x.clone() for x in ins]
    kv_indptr, kv_indices = mod.create_kv_indices(table, rpi, lens, start)
    total = int(lens.sum())
    check_pre(table, rpi, lens, kv_indptr, kv_indices.numel(), start)
    # every element of the host's output is written (kvOut covers `[0, total)`), so the initial
    # contents do not matter: compare all of it
    zero = torch.zeros_like(kv_indices)
    exp = expected(table, rpi, lens, kv_indptr, zero, start)
    check_negatives(name, kv_indices, table, rpi, lens, kv_indptr, zero, start)
    ok = torch.equal(kv_indices, exp) and kv_indices.numel() == total
    frame = all(torch.equal(a, b) for a, b in zip(before, ins))
    results.append((name, ok, frame, total))
    print(f"{name}: {lens.numel()} requests, {total} indices, kv_start_idx "
          f"{'no' if start is None else str(start.dtype).split('.')[-1]}: exact {ok}, inputs unchanged {frame}")


def run_guarded(name, table, rpi, lens, start=0, extra=777, guard=4096, kv_start=None):
    """The host's launch line (source lines 172-185) with `kv_indptr` from `start` and an output
    carved from a random-filled pool, `extra` elements longer than needed, with guards."""
    bs = lens.numel()
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    kv_indptr += start
    n_out = start + int(lens.sum()) + extra
    g = torch.Generator(device=DEV).manual_seed(5)
    pool = torch.randint(-2 ** 31, 2 ** 31 - 1, (guard + n_out + guard,), generator=g, device=DEV,
                         dtype=torch.int32)
    pool0 = pool.clone()
    out = pool[guard:guard + n_out]
    init = out.clone()
    check_pre(table, rpi, lens, kv_indptr, n_out, kv_start)
    mod.create_flashinfer_kv_indices_triton[(bs,)](
        table, rpi, lens, kv_indptr, kv_start, out, table.shape[1], ENTRY_PAGE_SIZE=1,
        TOKEN_BLOCK_PARALLEL=False, token_mapping=None, num_warps=4, num_stages=3)
    exp = expected(table, rpi, lens, kv_indptr, init, kv_start)
    check_negatives(name, out, table, rpi, lens, kv_indptr, init, kv_start)
    ok = torch.equal(out, exp)
    frame = torch.equal(pool[:guard], pool0[:guard]) and torch.equal(pool[guard + n_out:], pool0[guard + n_out:])
    results.append((name, ok, frame, int(lens.sum())))
    print(f"{name} (guarded direct launch): {bs} requests, start {start}: output exact {ok} (other "
          f"elements kept), guards unchanged {frame}")


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup: the host's test cases (source lines 203-235), same generator.
    cases = [
        (4096, 4096, 1, None, torch.int32, False),
        (4096, 4096, 37, None, torch.int32, False),
        (4096, 4096, 1786, None, torch.int64, False),
        (513, 8192, 6, [1, 511, 512, 513, 1025, 8192], torch.int64, False),
        (9, 131072, 8, [131072, 1, 70000, 4097, 0, 33, 99999, 512], torch.int64, False),
        (2049, 32768, 256, None, torch.int64, False),
        (300, 4096, 100, None, torch.int64, True),
    ]
    for num_slots, ctx, bs, lens, idt, with_start in cases:
        table = torch.randperm(num_slots * ctx, generator=gen).to(torch.int32).reshape(num_slots, ctx)
        rpi = torch.randperm(num_slots, generator=gen)[:bs]
        lens = torch.randint(0, ctx, (bs,), generator=gen) if lens is None else torch.tensor(lens)
        start = None
        if with_start:  # the host test's sliding window: the last `window` positions
            window = torch.randint(1, 1024, (bs,), generator=gen)
            gather = torch.minimum(lens, window)
            start = (lens - gather).to(DEV, idt)
            lens = gather
        run_host(f"setup slots {num_slots} ctx {ctx} batch {bs} {str(idt).split('.')[-1]}"
                 f"{' window' if with_start else ''}",
                 table.to(DEV), rpi.to(DEV, idt), lens.to(DEV, idt), start)
        del table

    g = torch.Generator().manual_seed(1)

    def rand_table(slots, ctx):
        return torch.randint(-2 ** 31, 2 ** 31 - 1, (slots, ctx), generator=g, dtype=torch.int32).to(DEV)

    # Edges: mixed index dtypes; lengths at the block boundaries; rows reused by several requests;
    # max_context_len = 1 (the stride argument a compile-time constant), 16 (marked divisible), 0
    # lengths only; a 2^20-token row; 65536 short requests.
    t = rand_table(64, 1024)
    for dp, dl in [(torch.int64, torch.int32), (torch.int32, torch.int64)]:
        lens = torch.tensor([0, 1, 511, 512, 513, 1023, 1024, 1000, 2, 0, 0, 1024], dtype=dl)
        rpi = torch.randint(0, 64, (12,), generator=g).to(dp)
        run_host(f"edge mixed dtypes rows {str(dp).split('.')[-1]} lens {str(dl).split('.')[-1]}",
                 t, rpi.to(DEV), lens.to(DEV))
    rpi = torch.tensor([3, 3, 3, 7, 3], dtype=torch.int64)
    run_host("edge one row for several requests", t, rpi.to(DEV), torch.tensor([10, 600, 1024, 5, 0]).to(DEV))
    t1 = rand_table(5000, 1)
    run_host("edge max_context_len 1", t1, torch.randperm(5000, generator=g)[:3000].to(DEV),
             torch.randint(0, 2, (3000,), generator=g).to(DEV))
    t16 = rand_table(300, 16)
    run_host("edge max_context_len 16", t16, torch.randint(0, 300, (200,), generator=g).to(DEV, torch.int32),
             torch.randint(0, 17, (200,), generator=g).to(DEV, torch.int32))
    run_host("edge all lengths 0", t, torch.randint(0, 64, (50,), generator=g).to(DEV),
             torch.zeros(50, dtype=torch.int64, device=DEV))
    tl_ = rand_table(4, 2 ** 20)
    run_host("edge 2^20-token rows", tl_, torch.tensor([2, 0, 3, 1]).to(DEV),
             torch.tensor([2 ** 20, 2 ** 20 - 1, 2 ** 19 + 1, 1]).to(DEV))
    tb = rand_table(70000, 40)
    run_host("edge 65536 requests", tb, torch.randperm(70000, generator=g)[:65536].to(DEV),
             torch.randint(0, 41, (65536,), generator=g).to(DEV))
    # FlashInfer's planners with kv_start_idx (flashinfer_backend.py:1687-1697, 2045-2071):
    # sliding-window decode, kv_start_idx = seq_lens - min(seq_lens, window + 1); the prefill's
    # full-attention wrapper, kv_start_idx = seq_lens - seq_lens (all 0); the ragged prefill's
    # window over the prefix, max(prefix - window, 0); int32 lengths and starts.
    tw = rand_table(64, 131072)
    seq = torch.randint(1, 131073, (64,), generator=g)
    seq[:4] = torch.tensor([1, 4097, 4098, 131072])
    paged = torch.clamp(seq, max=4097)
    run_host("swa decode window 4096, contexts to 131072", tw, torch.randperm(64, generator=g).to(DEV),
             paged.to(DEV), (seq - paged).to(DEV))
    seq = torch.randint(0, 131073, (32,), generator=g)
    run_host("prefill full attention, kv_start_idx 0", tw, torch.randint(0, 64, (32,), generator=g).to(DEV),
             seq.to(DEV), (seq - seq).to(DEV))
    prefix = torch.randint(0, 131073, (48,), generator=g)
    eff = torch.clamp(prefix - 4096, min=0)
    run_host("ragged prefill window over the prefix", tw, torch.randint(0, 64, (48,), generator=g).to(DEV),
             (prefix - eff).to(DEV), eff.to(DEV))
    del tw
    t8 = rand_table(128, 8192)
    seq = torch.randint(0, 8193, (32,), generator=g)
    paged = torch.clamp(seq, max=1025)
    run_host("int32 lengths and kv_start_idx, int64 rows", t8, torch.randint(0, 128, (32,), generator=g).to(DEV),
             paged.to(DEV, torch.int32), (seq - paged).to(DEV, torch.int32))
    run_host("int64 lengths, int32 kv_start_idx and rows", t8,
             torch.randint(0, 128, (32,), generator=g).to(DEV, torch.int32), paged.to(DEV),
             (seq - paged).to(DEV, torch.int32))
    # a window past its row (start + length beyond max_context_len, inside the table: Pre allows it)
    run_host("edge window past its row", t8, torch.tensor([3, 100, 0]).to(DEV),
             torch.tensor([600, 513, 2000]).to(DEV), torch.tensor([8000, 8191, 7000]).to(DEV))
    # Frame: a nonzero start of kv_indptr, a longer output, random initial contents.
    run_guarded("frame start 0", t, torch.randint(0, 64, (40,), generator=g).to(DEV),
                torch.randint(0, 1025, (40,), generator=g).to(DEV))
    run_guarded("frame start 1234, int32", t, torch.randint(0, 64, (17,), generator=g).to(DEV, torch.int32),
                torch.randint(0, 1025, (17,), generator=g).to(DEV, torch.int32), start=1234)
    run_guarded("frame start 99 with kv_start_idx", t, torch.randint(0, 64, (23,), generator=g).to(DEV),
                torch.randint(0, 513, (23,), generator=g).to(DEV), start=99,
                kv_start=torch.randint(0, 512, (23,), generator=g).to(DEV))

    bad = [r for r in results if not (r[1] and r[2])] + [n for n in negatives if not n[1]]
    print(f"{len(results)} launches, {sum(r[3] for r in results)} written indices checked; worst "
          f"error / tolerance: 0 / 0 (exact) on every element; frame held on every launch; "
          f"{len(negatives)} wrong outputs, every one rejected by Post"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
