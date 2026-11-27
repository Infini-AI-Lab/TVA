"""GPU check of the claim for `ep_src2dst_i32`.

Runs the kernel's own host (`source/ep_src2dst.py`, `cutlass_w4_run_moe_ep_preproess`) on the
setup's launches (the host's test cases, with the same generator, and serving shapes) and direct
launches of `compute_src2dst_triton_kernel` with the host's launch line at the edges of `Inst.ok`
and `Pre`, and checks `Post` exactly as `Claim.lean` states it:

* for every sorted position `i < n`, element `reorder_ids[i]` of `src2dst` is `-1` when
  `sorted_topk_ids[i] < 0` and `i - nneg` otherwise, `nneg` the number of negative ids among the
  first `n` sorted ids;
* the frame: every other byte is unchanged (the inputs; in the direct launches, the output's
  elements no `reorder_ids[i]` names and guard regions around it, from random contents).

`Pre` is checked on every launch: the negative ids come first, the first `n` entries of
`reorder_ids` are distinct elements of `src2dst`. Negative tests: outputs a wrong kernel would
leave (the packed row without the negative prefix removed, the prefix off by one, `-1` not
written for the negative ids, the row stored at the sorted position instead of at `reorder_ids[i]`) must fail `Post`. Integers only: exact, no
tolerance. Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "ep_src2dst.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("ep_src2dst_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")


def check_pre(sorted_ids, reorder, n, out_len):
    """`Inst.ok` and `Pre`: `n < 2^31`, the buffers hold `n` entries, the negative ids come first,
    the first `n` entries of `reorder_ids` are distinct elements of `src2dst`."""
    assert n < 2 ** 31 and sorted_ids.numel() >= n and reorder.numel() >= n
    assert sorted_ids.dtype == torch.int32 and reorder.dtype == torch.int64
    neg = (sorted_ids[:n] < 0).to(torch.int8)
    # negatives first: the mask never rises from 0 to 1
    assert n < 2 or bool((neg[1:] <= neg[:-1]).all())
    r = reorder[:n]
    assert bool((r >= 0).all()) and bool((r < out_len).all())
    assert torch.unique(r).numel() == n


def expected(sorted_ids, reorder, n, init_out, mode="post"):
    """`Post`, from the kernel's inputs: the output the claim requires, given the output's initial
    contents. The other modes are wrong outputs (negative tests)."""
    out = init_out.clone()
    if n == 0:
        return out
    pos = torch.arange(n, device=DEV, dtype=torch.int64)
    neg = sorted_ids[:n] < 0
    nneg = int(neg.sum())
    if mode == "post":
        val = torch.where(neg, -1, pos - nneg)
    elif mode == "prefix kept":
        val = torch.where(neg, -1, pos)
    elif mode == "prefix + 1":
        val = torch.where(neg, -1, pos - nneg - 1)
    elif mode == "no -1":
        val = pos - nneg
    elif mode == "sorted position":
        out[:n] = torch.where(neg, -1, pos - nneg).to(torch.int32)
        return out
    out[reorder[:n]] = val.to(torch.int32)
    return out


negatives = []


def check_negatives(name, out, sorted_ids, reorder, n, init_out):
    """Wrong outputs `Post` must reject (each that differs from the required one on a written
    element)."""
    want = expected(sorted_ids, reorder, n, init_out)
    for what in ("prefix kept", "prefix + 1", "no -1", "sorted position"):
        w = expected(sorted_ids, reorder, n, init_out, what)
        if not torch.equal(w, want):
            negatives.append((f"{name}: {what}", not torch.equal(out, w)))


results = []


def run_host(name, topk_ids):
    """The host on `topk_ids`; `sorted_topk_ids`, `reorder_ids` recomputed by the same stable sort
    (deterministic) to state `Post` on the kernel's inputs."""
    before = topk_ids.clone()
    src2dst = mod.cutlass_w4_run_moe_ep_preproess(topk_ids)
    n = topk_ids.numel()
    sorted_ids, reorder = torch.sort(topk_ids.view(-1), stable=True)
    check_pre(sorted_ids, reorder, n, src2dst.numel())
    # every element of the host's output is named by the permutation, so the initial contents do
    # not matter: compare all of it
    zero = torch.zeros_like(src2dst)
    exp = expected(sorted_ids, reorder, n, zero)
    ok = torch.equal(src2dst, exp)
    if n <= 2 ** 20:
        ok = ok and torch.equal(src2dst.cpu(), mod.reference(topk_ids))
    check_negatives(name, src2dst, sorted_ids, reorder, n, zero)
    frame = torch.equal(before, topk_ids)
    results.append((name, ok, frame, n))
    print(f"{name}: {n} slots, {int((topk_ids < 0).sum())} negative: exact {ok}, inputs unchanged {frame}")


def run_guarded(name, sorted_ids, reorder, n, out_len, guard=4096, seed=5):
    """The host's launch line (source lines 68-72) on given inputs, with `src2dst` carved from a
    random-filled pool, guards around it."""
    check_pre(sorted_ids, reorder, n, out_len)
    g = torch.Generator(device=DEV).manual_seed(seed)
    pool = torch.randint(-2 ** 31, 2 ** 31 - 1, (guard + out_len + guard,), generator=g, device=DEV,
                         dtype=torch.int32)
    pool0 = pool.clone()
    out = pool[guard:guard + out_len]
    init = out.clone()
    ins = [sorted_ids, reorder]
    before = [x.clone() for x in ins]
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    mod.compute_src2dst_triton_kernel[grid](sorted_ids, reorder, out, n, BLOCK_SIZE)
    exp = expected(sorted_ids, reorder, n, init)
    check_negatives(name, out, sorted_ids, reorder, n, init)
    ok = torch.equal(out, exp)
    frame = (torch.equal(pool[:guard], pool0[:guard]) and torch.equal(pool[guard + out_len:], pool0[guard + out_len:])
             and all(torch.equal(a, b) for a, b in zip(before, ins)))
    results.append((name, ok, frame, n))
    print(f"{name} (guarded direct launch): {n} slots, output {out_len}, "
          f"{int((sorted_ids[:n] < 0).sum())} negative: output exact {ok} (other elements kept), "
          f"inputs and guards unchanged {frame}")


def topk_like(gen, num_tokens, topk, num_experts, neg_share):
    ids = torch.randint(0, num_experts, (num_tokens, topk), generator=gen, dtype=torch.int32)
    drop = torch.rand((num_tokens, topk), generator=gen) < neg_share
    ids[drop] = -1
    return ids


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup: the host's test cases (source lines 93-100), same generator.
    cases = [
        (1, 1, 8, 0.0),
        (7, 8, 16, 0.5),
        (512, 8, 32, 0.0),
        (1000, 8, 64, 0.75),
        (4096, 6, 160, 0.25),
        (300, 4, 8, 1.0),
    ]
    for num_tokens, topk, num_experts, neg_share in cases:
        run_host(f"setup test {num_tokens}x{topk}", topk_like(gen, num_tokens, topk, num_experts, neg_share).to(DEV))

    g = torch.Generator().manual_seed(1)
    # Serving shapes: decode (one token, top-8); a 16384-token prefill chunk with 256 experts on EP 8
    # (7/8 of the slots to no local expert); 2^22 tokens, top-8 (2^25 slots).
    run_host("serving decode 1x8", topk_like(g, 1, 8, 32, 0.875).to(DEV))
    run_host("serving prefill 16384x8 ep8", topk_like(g, 16384, 8, 32, 0.875).to(DEV))
    big = torch.randint(-1, 32, (2 ** 22, 8), device=DEV, dtype=torch.int32)
    run_host("serving 2^22 tokens x8", big)
    del big
    # Edges of the host: a single negative slot; no negative slot; numel at block boundaries.
    run_host("edge 1 slot negative", torch.tensor([[-1]], dtype=torch.int32, device=DEV))
    for n in (511, 512, 513, 1024, 1025):
        run_host(f"edge {n} slots", topk_like(g, n, 1, 4, 0.3).to(DEV))

    # Direct launches at the edges of Pre: ids whose negatives come first but are not sorted
    # after them (any values below -1 too), a reorder that is an injection into a longer output,
    # n = 1 (the compile-time `num_toks`), all negative, the negative prefix ending at a block
    # boundary.
    def direct(name, n, num_neg, out_len, seed):
        gg = torch.Generator().manual_seed(seed)
        neg = -torch.randint(1, 2 ** 31, (num_neg,), generator=gg, dtype=torch.int64)
        pos = torch.randint(0, 2 ** 31, (n - num_neg,), generator=gg, dtype=torch.int64)
        sorted_ids = torch.cat([neg, pos]).to(torch.int32).to(DEV)
        reorder = torch.randperm(out_len, generator=gg)[:n].to(DEV)
        run_guarded(name, sorted_ids, reorder, n, out_len, seed=seed)

    direct("edge n 1, non-negative", 1, 0, 1, 11)
    direct("edge n 1, negative", 1, 1, 7, 12)
    direct("edge unsorted after the prefix, injection", 5000, 1234, 9000, 13)
    direct("edge all negative", 2049, 2049, 2049, 14)
    direct("edge prefix ending at a block", 4096, 1024, 4096, 15)
    direct("edge prefix 511 of 513", 513, 511, 600, 16)
    direct("edge no negative, longer output", 70000, 0, 100000, 17)
    direct("edge large, half negative", 2 ** 24 + 3, 2 ** 23, 2 ** 24 + 100, 18)

    bad = [r for r in results if not (r[1] and r[2])] + [x for x in negatives if not x[1]]
    print(f"{len(results)} launches, {sum(r[3] for r in results)} written elements checked; worst "
          f"error / tolerance: 0 / 0 (exact) on every element; frame held on every launch; "
          f"{len(negatives)} wrong outputs, every one rejected by Post"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
