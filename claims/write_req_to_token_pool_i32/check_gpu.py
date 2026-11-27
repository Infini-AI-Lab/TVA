"""GPU check of the claim for `write_req_to_token_pool_i32`.

Runs the kernel's own host (`source/write_req_to_token_pool.py`, `write_cache_indices`) on the
setup's launches (the host's test cases, with the same generator, and serving shapes) and on
launches at the edges of `Inst.ok` and `Pre`, and checks `Post` exactly as `Claim.lean` states it:

* for every request `b`, element `j` of row `req_pool_indices[b]` of the table holds the low 32
  bits of `prefix_b[j]` for `j < prefix_lens[b]`, and of
  `out_cache_loc[c[b] + j - prefix_lens[b]]` for `prefix_lens[b] <= j < seq_lens[b]`, where
  `c[b] = extend_lens[0] + ... + extend_lens[b - 1]`;
* the frame: every other byte is unchanged: the rest of the table (filled with random values
  first), guard regions around the table, and every input (the index buffers, `out_cache_loc`,
  the prefix tensors and the pointer tensor).

The test locations are random `int64` values over the whole range (not only valid slot numbers),
so that the `int32` cast is checked on values that do not fit `int32`. Negative tests: outputs a
wrong kernel would leave (the prefix not written, the new locations one element late, every
request reading `out_cache_loc` from `0`, a request's last block of 512 dropped) must fail
`Post`. Integers only: exact, no tolerance. Run with
`CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "write_req_to_token_pool.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("write_req_to_token_pool_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
EMPTY = torch.empty((0,), dtype=torch.int64)  # the host's prefix of a request without one
GUARD = 4096


def low32(x):
    """The `int32` cast of `int64` values (`IntV.cast`): the low 32 bits, two's complement."""
    return (((x.long() + 2 ** 31) % 2 ** 32) - 2 ** 31).to(torch.int32)


def check_pre(table, rows, pre, seq, ext, prefixes, out):
    """`Inst.ok` and `Pre` on the host's inputs."""
    bs = rows.numel()
    mcl = table.shape[1]
    assert bs <= 8192 and mcl <= 2 ** 22 and table.is_contiguous()
    for t in (rows, pre, seq, ext, out):
        assert t.dtype == torch.int64
    rows, pre, seq, ext = (t.cpu().tolist() for t in (rows, pre, seq, ext))
    c = 0
    writing = []
    for b in range(bs):
        assert 0 <= pre[b] <= seq[b] <= mcl and rows[b] >= 0
        assert rows[b] * mcl + seq[b] <= table.numel()
        if pre[b] > 0:
            assert prefixes[b].dtype == torch.int64 and prefixes[b].numel() >= pre[b]
            assert prefixes[b].is_cuda
        assert 0 <= ext[b] <= 2 ** 22
        assert c + seq[b] - pre[b] <= out.numel()
        if seq[b] > 0:
            writing.append(rows[b])
        c += ext[b]
    assert len(set(writing)) == len(writing)


def expected(init, rows, pre, seq, ext, prefixes, out, mode=None):
    """`Post`, from the host's inputs: the table the claim requires, given its initial contents.
    `mode` gives the table of a wrong kernel instead (a negative test)."""
    exp = init.clone()
    rows, pre, seq, ext = (t.cpu().tolist() for t in (rows, pre, seq, ext))
    c = 0
    for b in range(len(rows)):
        r, p, s = rows[b], pre[b], seq[b]
        if p > 0 and mode != "prefix not written":
            exp[r, :p] = low32(prefixes[b][:p])
        start = {"one late": c + 1, "start 0": 0}.get(mode, c)
        n = s - p
        if mode == "last block dropped":
            n = ((n - 1) // 512) * 512 if n > 0 else 0
        if n > 0:
            src = out[start:start + n]
            exp[r, p:p + src.numel()] = low32(src)
        c += ext[b]
    return exp


results = []
negatives = []


def run(name, rows_n, mcl, rows, pre, seq, ext, out_len=None, gen=None, direct=False,
        empty_prefix=EMPTY):
    """One launch: the table `[rows_n, mcl]` carved from a random pool with guards, random
    prefixes of the given lengths and random `int64` locations; the host's `write_cache_indices`
    (or, with `direct`, its launch line called with the same arguments)."""
    g = gen if gen is not None else torch.Generator().manual_seed(7)
    bs = len(rows)
    rows, pre, seq, ext = (torch.tensor(x, dtype=torch.int64) for x in (rows, pre, seq, ext))
    n_out = int(ext.sum()) if out_len is None else out_len
    gd = torch.Generator(device=DEV).manual_seed(11)
    pool = torch.randint(-2 ** 31, 2 ** 31 - 1, (GUARD + rows_n * mcl + GUARD,), generator=gd,
                         device=DEV, dtype=torch.int32)
    pool0 = pool.clone()
    table = pool[GUARD:GUARD + rows_n * mcl].view(rows_n, mcl)
    init = pool0[GUARD:GUARD + rows_n * mcl].view(rows_n, mcl)
    prefixes = [torch.randint(-2 ** 63, 2 ** 63 - 1, (int(p),), generator=gd, device=DEV,
                              dtype=torch.int64) if int(p) > 0 else empty_prefix for p in pre]
    out = torch.randint(-2 ** 63, 2 ** 63 - 1, (n_out,), generator=gd, device=DEV,
                        dtype=torch.int64)
    args = [t.to(DEV) for t in (rows, pre, seq, ext)]
    check_pre(table, *args, prefixes, out)
    inputs = args + [out] + [p for p in prefixes if p.numel() > 0]
    before = [x.clone() for x in inputs]
    if direct:
        ptrs = torch.tensor([t.data_ptr() for t in prefixes], dtype=torch.uint64).to(DEV)
        mod.write_req_to_token_pool_triton[(bs,)](
            table, args[0], ptrs, args[1], args[2], args[3], out, table.shape[1],
            num_warps=4, num_stages=3)
        ptrs0 = ptrs.clone()
    else:
        ptrs = mod.write_cache_indices(out, *args, prefixes, table)
        ptrs0 = torch.tensor([t.data_ptr() for t in prefixes], dtype=torch.uint64).to(DEV)
    torch.cuda.synchronize()
    exp = expected(init, *args, prefixes, out)
    ok = torch.equal(table, exp)
    frame = (torch.equal(pool[:GUARD], pool0[:GUARD]) and
             torch.equal(pool[GUARD + rows_n * mcl:], pool0[GUARD + rows_n * mcl:]) and
             all(torch.equal(a, b) for a, b in zip(before, inputs)) and torch.equal(ptrs, ptrs0))
    for mode in ("prefix not written", "one late", "start 0", "last block dropped"):
        w = expected(init, *args, prefixes, out, mode)
        if not torch.equal(w, exp):  # a wrong output that differs on this launch
            negatives.append((f"{name}: {mode}", not torch.equal(table, w)))
        del w
    written = int(seq.sum())
    results.append((name, ok, frame, written))
    print(f"{name}{' (direct launch)' if direct else ''}: pool {rows_n} x {mcl}, {bs} requests, "
          f"{written} elements written: exact {ok}, frame (rest of table, guards, inputs) {frame}")
    del pool, pool0, table, init, exp, prefixes, out, inputs, before
    torch.cuda.empty_cache()


def batch(gen, size, mcl, bs, max_pre, max_ext, min_ext=1):
    """A random batch as `alloc_for_extend` makes it: distinct rows of the pool (row 0 is the
    padding row), prefixes and extends with `prefix + extend <= mcl`, `extend_lens` the extends."""
    rows = (torch.randperm(size, generator=gen) + 1)[:bs]
    pre = torch.randint(0, max_pre + 1, (bs,), generator=gen)
    ext = torch.randint(min_ext, max_ext + 1, (bs,), generator=gen)
    ext = torch.minimum(ext, mcl - pre)
    return rows.tolist(), pre.tolist(), (pre + ext).tolist(), ext.tolist()


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup: the host's test cases (source lines 158-184), the same batches.
    for size, ctx, bs, max_pre, max_ext in [
        (64, 4096, 1, 0, 4096),
        (256, 4096, 37, 2048, 2048),
        (4096, 4096, 512, 1024, 32),
        (64, 32768, 8, 20000, 8192),
        (8, 131072, 4, 100000, 16384),
    ]:
        mcl = ctx + 4
        rows, pre, seq, ext = batch(gen, size, mcl, bs, max_pre, max_ext)
        run(f"setup test pool {size} ctx {ctx} batch {bs}", size + 1, mcl, rows, pre, seq, ext)
        torch.randperm(sum(ext) + 1000, generator=gen)  # the host test's slot draw
        for p in pre:  # and its prefix draws, to keep the generator in step
            if p > 0:
                torch.randint(1, 2 ** 31, (p,), generator=gen)

    g = torch.Generator().manual_seed(1)
    # Serving shapes (SETUP.md): a chunked prefill of 8192 new tokens over 1024 requests of a
    # 128K-context pool (prefixes cut to 16384 to keep the case under 4 GB); a 1M-token row; 8192
    # requests; the row length of speculative decoding (`ctx + 4 + 8`).
    rows, pre, seq, ext = batch(g, 2048, 131076, 1024, 16384, 15)
    run("setup serve ctx 128K, 1024 requests", 2049, 131076, rows, pre, seq, ext)
    rows, pre, seq, ext = batch(g, 8, 2 ** 20 + 4, 8, 2 ** 20 - 2048, 2048)
    pre[0], seq[0], ext[0] = 2 ** 20, 2 ** 20 + 4, 4  # a full row
    run("setup serve ctx 1M", 9, 2 ** 20 + 4, rows, pre, seq, ext)
    rows, pre, seq, ext = batch(g, 8192, 4100, 8192, 64, 4)
    run("setup serve 8192 requests", 8193, 4100, rows, pre, seq, ext)
    rows, pre, seq, ext = batch(g, 256, 32780, 64, 32768, 512)
    run("setup spec decode ctx 32K + 8", 257, 32780, rows, pre, seq, ext)

    # Edges: lengths at the block boundaries; no prefixes; no new tokens; whole rows; rows 0 and
    # last; max_context_len 1; empty requests among others (rows reused by requests that write
    # nothing); the batch of one; prefixes longer than prefix_lens.
    L = [0, 1, 511, 512, 513, 1023, 1024, 1025, 1536, 2047]
    rows = list(range(1, 1 + len(L) ** 2))
    pre = [a for a in L for _ in L]
    ext = [b for _ in L for b in L]
    seq = [a + b for a, b in zip(pre, ext)]
    run("edge block boundaries (prefix x extend)", 1 + len(L) ** 2, 4100, rows, pre, seq, ext, gen=g)
    run("edge no prefixes", 65, 4100, list(range(1, 41)), [0] * 40, [100 * i for i in range(40)],
        [100 * i for i in range(40)], gen=g)
    run("edge no new tokens", 65, 4100, list(range(1, 41)), [100 * i for i in range(40)],
        [100 * i for i in range(40)], [0] * 40, gen=g)
    run("edge whole rows, rows 0 and last", 5, 3000, [0, 4, 2], [1000, 0, 3000], [3000, 3000, 3000],
        [2000, 3000, 0], gen=g)
    s1 = [1 if (i % 3 or i % 2) else 0 for i in range(300)]
    run("edge max_context_len 1", 300, 1, list(range(300)), [i % 2 for i in range(300)], s1,
        [s - i % 2 for i, s in enumerate(s1)], gen=g)
    run("edge empty requests on a shared row", 9, 2048, [3, 3, 5, 3, 3], [0, 0, 100, 0, 0],
        [0, 700, 900, 0, 0], [0, 700, 800, 0, 0], gen=g)
    run("edge one request", 2, 4100, [1], [4000], [4100], [100], gen=g)
    # Direct launches with the host's launch line: extend_lens other than seq - prefix (the claim
    # reads c[b] from extend_lens alone), a longer out_cache_loc, an empty prefix pointer of 0,
    # and the empty grid.
    run("direct extend_lens larger than the extends", 65, 4100, [7, 9, 11, 13], [5, 0, 600, 1],
        [600, 1000, 1200, 3000], [1000, 3, 2000, 4], out_len=6100, gen=g, direct=True)
    run("direct prefix pointer 0 for empty prefixes", 65, 4100, [1, 2, 3], [0, 10, 0],
        [513, 20, 1], [513, 10, 1], gen=g, direct=True,
        empty_prefix=torch.empty((0,), dtype=torch.int64, device=DEV))
    run("direct empty batch", 9, 4100, [], [], [], [], gen=g, direct=True)

    bad = [r for r in results if not (r[1] and r[2])] + [n for n in negatives if not n[1]]
    print(f"{len(results)} launches, {sum(r[3] for r in results)} written elements checked; worst "
          f"error / tolerance: 0 / 0 (exact) on every element; frame held on every launch; "
          f"{len(negatives)} wrong outputs, every one rejected by Post"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
