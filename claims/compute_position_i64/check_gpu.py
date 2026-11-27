"""GPU check of the claim for `compute_position_i64`.

Runs the kernel's own host, `compute_position_triton` of `source/position.py` (imported by path),
on the setup's launches (the code base's test cases, prefill batches, speculative draft extends,
the host without prefixes, an empty batch) and on launches at the edges of `Inst.ok` and `Pre`,
and checks `Post` exactly as `Claim.lean` states it:

* for every request `b` and `j < extend_seq_lens[b]`, element `start(b) + j` of `positions` is
  the `int64` value `p(b) + j`, `start(b)` the sum of the lengths before request `b` and `p(b)`
  the request's prefix (`0` without prefixes); computed here from the host's inputs, in `int64`
  on the GPU, independently of the kernel;
* element `b` of `extend_start_loc` is `start(b)`;
* the frame: the inputs are unchanged, and, in direct launches with the host's own launch line
  whose outputs are slices of random-filled buffers with guards, every other element and every
  guard keeps its initial value.

Negative tests: outputs a wrong kernel would leave (the prefix ignored, the inclusive running sum
as the start, each request's last block of 512 dropped, positions off by one) must fail `Post`.
Integers only: exact, no tolerance. Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "position.py")
spec = importlib.util.spec_from_file_location("position_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")


def check_pre(prefix, lens, has_prefix, n_pos):
    """`ReqOk` for every request and the length of `positions` (`Pre`), from the host's inputs."""
    l = lens.long().cpu()
    p = prefix.long().cpu() if has_prefix else torch.zeros_like(l)
    start = torch.cumsum(l, 0) - l
    assert lens.dtype == torch.int32 and (not has_prefix or prefix.dtype == torch.int32)
    assert bool((l >= 0).all()) and bool((l + 511 < 2 ** 31).all())
    assert bool((p + l <= 2 ** 31).all()) and bool((start < 2 ** 31).all())
    assert int(l.sum()) <= n_pos


def expected(prefix, lens, has_prefix, init_pos, init_start, variant=None):
    """`Post`, from the host's inputs: the outputs the claim requires, given their initial contents.
    `variant` gives the outputs of a wrong kernel instead (a negative test)."""
    bs = lens.numel()
    l = lens.long()
    p = prefix[:bs].long() if has_prefix else torch.zeros_like(l)
    if variant == "prefix ignored":
        p = torch.zeros_like(l)
    start = torch.cumsum(l, 0) - l
    pos = init_pos.clone()
    sl = init_start.clone()
    total = int(l.sum())
    if total > 0:
        req = torch.repeat_interleave(torch.arange(bs, device=l.device), l)
        j = torch.arange(total, device=l.device) - start[req]
        vals = p[req] + j
        if variant == "off by one":
            vals = vals + 1
        keep = torch.ones(total, dtype=torch.bool, device=l.device)
        if variant == "last block dropped":
            keep = j < ((l[req] - 1) // 512) * 512
        pos[:total][keep] = vals[keep]
    sl[:bs] = (torch.cumsum(l, 0) if variant == "inclusive start" else start).to(torch.int32)
    return pos, sl


negatives = []


def check_negatives(name, pos, sl, prefix, lens, has_prefix, init_pos, init_start):
    """Wrong outputs `Post` must reject (each differs from the right one on some written element)."""
    l = lens.long()
    wrong = []
    if has_prefix and bool((prefix[:lens.numel()].long() != 0).any()) and int(l.sum()) > 0:
        wrong.append("prefix ignored")
    if bool((l[:-1] > 0).any()) or (lens.numel() > 0 and int(l[0]) > 0):
        wrong.append("inclusive start")
    if bool((l > 512).any()):
        wrong.append("last block dropped")
    if int(l.sum()) > 0:
        wrong.append("off by one")
    for v in wrong:
        wp, ws = expected(prefix, lens, has_prefix, init_pos, init_start, v)
        negatives.append((f"{name}: {v}", not (torch.equal(pos, wp) and torch.equal(sl, ws))))


results = []


def run_host(name, prefix, lens):
    """The host `compute_position_triton` (source lines 38-61) with `extend_seq_lens_sum` the sum of
    the lengths."""
    bs = lens.numel()
    has_prefix = prefix.shape[0] == bs
    before = [prefix.clone(), lens.clone()]
    total = int(lens.long().sum())
    pos, sl = mod.compute_position_triton(prefix, lens, total)
    torch.cuda.synchronize()
    check_pre(prefix, lens, has_prefix, pos.numel())
    # every element of both outputs is written (`posOut` is `[0, total)`), so the initial contents
    # do not matter
    zp, zs = torch.zeros_like(pos), torch.zeros_like(sl)
    ep, es = expected(prefix, lens, has_prefix, zp, zs)
    check_negatives(name, pos, sl, prefix, lens, has_prefix, zp, zs)
    ok = torch.equal(pos, ep) and torch.equal(sl, es) and pos.dtype == torch.int64
    frame = torch.equal(before[0], prefix) and torch.equal(before[1], lens)
    results.append((name, ok, frame, total))
    print(f"{'ok ' if ok and frame else 'BAD'} {name}: {bs} requests, {total} positions, prefixes "
          f"{'yes' if has_prefix else 'no'}: exact {ok}, inputs unchanged {frame}")


def run_guarded(name, prefix, lens, extra=777, guard=4096):
    """The host's launch line (source lines 53-59) with outputs carved from random-filled buffers,
    `positions` `extra` elements longer than needed, both with guards."""
    bs = lens.numel()
    has_prefix = prefix.shape[0] == bs
    before = [prefix.clone(), lens.clone()]
    n_pos = int(lens.long().sum()) + extra
    g = torch.Generator(device=DEV).manual_seed(5)
    pbuf = torch.randint(-2 ** 62, 2 ** 62, (guard + n_pos + guard,), generator=g, device=DEV,
                         dtype=torch.int64)
    sbuf = torch.randint(-2 ** 31, 2 ** 31 - 1, (guard + bs + guard,), generator=g, device=DEV,
                         dtype=torch.int32)
    pbuf0, sbuf0 = pbuf.clone(), sbuf.clone()
    pos, sl = pbuf[guard:guard + n_pos], sbuf[guard:guard + bs]
    ip, is_ = pos.clone(), sl.clone()
    check_pre(prefix, lens, has_prefix, n_pos)
    mod.compute_position_kernel[(bs,)](pos, sl, prefix, lens, has_prefix, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    ep, es = expected(prefix, lens, has_prefix, ip, is_)
    check_negatives(name, pos, sl, prefix, lens, has_prefix, ip, is_)
    ok = torch.equal(pos, ep) and torch.equal(sl, es)
    frame = (torch.equal(pbuf[:guard], pbuf0[:guard]) and torch.equal(pbuf[guard + n_pos:], pbuf0[guard + n_pos:])
             and torch.equal(sbuf[:guard], sbuf0[:guard]) and torch.equal(sbuf[guard + bs:], sbuf0[guard + bs:])
             and torch.equal(before[0], prefix) and torch.equal(before[1], lens))
    results.append((name, ok, frame, int(lens.long().sum())))
    print(f"{'ok ' if ok and frame else 'BAD'} {name} (guarded direct launch): {bs} requests: outputs "
          f"exact {ok} (other elements kept), guards and inputs unchanged {frame}")


def i32(x):
    return torch.as_tensor(x).to(device=DEV, dtype=torch.int32)


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    # The code base's test (test_next_verify_prep.py:107-134): bs in {1, 3, 129}, width in
    # {1, 4, 7}, every prefix the value in {0, 1, 131000}.
    for bs in [1, 3, 129]:
        for width in [1, 4, 7]:
            for value in [0, 1, 131000]:
                run_host(f"test bs {bs} width {width} prefix {value}",
                         i32(torch.full((bs,), value)), i32(torch.full((bs,), width)))
    # Prefill batches (ForwardBatch.init_new, two-batch overlap): random lengths and prefixes.
    lens = torch.randint(1, 257, (64,), generator=g)
    run_host("prefill 64 requests", i32(torch.randint(0, 2 ** 20, (64,), generator=g)), i32(lens))
    run_host("prefill one long request", i32([2 ** 20]), i32([2 ** 20]))
    run_host("prefill 4096 requests, zeros among them",
             i32(torch.randint(0, 2 ** 17, (4096,), generator=g)),
             i32(torch.randint(0, 33, (4096,), generator=g)))
    run_host("prefill lengths at the block edges",
             i32(torch.randint(0, 4096, (12,), generator=g)),
             i32([0, 1, 511, 512, 513, 1023, 1024, 1025, 0, 2047, 8192, 3]))
    # Speculative draft extend (eagle_worker_common.py, DFlash, DSpark): fixed widths.
    run_host("draft extend 256 x 8", i32(torch.randint(0, 2 ** 17, (256,), generator=g)),
             i32(torch.full((256,), 8)))
    # The host without prefixes (`extend_prefix_lens` of another length), and an empty batch.
    run_host("no prefixes, 16 requests", i32(torch.zeros(0)), i32(torch.randint(0, 600, (16,), generator=g)))
    run_host("no prefixes, prefix tensor of another length", i32(torch.arange(5)),
             i32(torch.randint(0, 600, (9,), generator=g)))
    run_host("empty batch", i32(torch.zeros(0)), i32(torch.zeros(0)))
    # Edges of ReqOk: a prefix with prefix + length = 2^31 (the largest position 2^31 - 1), a
    # negative prefix, 8192 requests (every program sums up to 8191 lengths).
    run_host("edge prefix + length = 2^31", i32([2 ** 31 - 1000, 5, 2 ** 31 - 1]), i32([1000, 7, 1]))
    run_host("edge negative prefixes", i32([-5, -2 ** 31, -1]), i32([700, 3, 1]))
    run_host("edge 8192 requests", i32(torch.randint(0, 2 ** 20, (8192,), generator=g)),
             i32(torch.randint(0, 65, (8192,), generator=g)))
    # Frame: outputs inside random-filled buffers with guards.
    run_guarded("frame prefixes", i32(torch.randint(0, 2 ** 20, (40,), generator=g)),
                i32(torch.randint(0, 1100, (40,), generator=g)))
    run_guarded("frame no prefixes", i32(torch.zeros(0)), i32(torch.randint(0, 1100, (23,), generator=g)))
    run_guarded("frame all lengths 0", i32(torch.randint(0, 99, (17,), generator=g)), i32(torch.zeros(17)))

    bad = [r for r in results if not (r[1] and r[2])] + [n for n in negatives if not n[1]]
    print(f"{len(results)} launches, {sum(r[3] for r in results)} positions checked; worst error / "
          f"tolerance: 0 / 0 (exact) on every element; frame held on every launch; "
          f"{len(negatives)} wrong outputs, every one rejected by Post"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
