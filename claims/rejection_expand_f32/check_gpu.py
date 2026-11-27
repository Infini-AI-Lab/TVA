"""GPU check of the claim for `rejection_expand_f32`.

Runs the kernel's own host (`source/rejection_expand.py`, `expand_batch_to_tokens`) on the setup's
launches (the docstring's example, the host test's random batches with the three calls of vLLM's
`apply_sampling_constraints`: `temperature` `float32` `0 -> 1`, `top_k` `int32` `0 -> 0`, `top_p`
`float32` `0 -> 0`, and the serving sizes), on launches at the edges of `Inst.ok` and `Pre`
(8192 requests, every count `0` to `128`, NaNs of several payloads, infinities, signed zeros,
subnormals, `replace_from`/`replace_to` at the `int32` extremes and at values `float32` rounds),
on guarded direct launches with the host's launch line (an output inside a sentinel pool, longer
than the number of tokens), and on one launch beyond `Inst.ok` (more than 8192 requests), and
checks `Post` exactly as `Claim.lean` states it:

* for every request `r < B` and position `start_r <= j < cu[r]`, output element `j` holds
  `expandVal dt x[r] replace_from replace_to`, bit for bit (on `float32`: `float32(replace_to)`
  when `x[r] == float32(replace_from)` under the ordered float comparison, else `x[r]` with its
  bits, NaN payloads included; on `int32`: `replace_to` when `x[r] == replace_from`, else `x[r]`);
* the frame: every byte outside the output's first `cu[B-1]` elements is unchanged (the inputs,
  and in a guarded launch the output's tail and the guards around it).

Every value is exact, so the check is bit for bit (no tolerance). Negative tests: wrong outputs
(the replacement ignored, a bitwise comparison in place of the float one, `replace_to` left
unconverted, a shifted packing, a request's last token missing, an all-zero output) that `Post`
must reject. Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "rejection_expand.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("rejection_expand_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")


def bits(t):
    """The 32-bit patterns of a `float32` or `int32` tensor, as `int32`."""
    return t.contiguous().view(torch.int32)


def f32_nearest(z):
    """`FloatV.ofRat z : F32`: the `float32` nearest to the integer `z` (ties to even); `z` is an
    `int32`, exact in `float64`, so the `float64 -> float32` conversion rounds once."""
    return torch.tensor(float(z), dtype=torch.float64).to(torch.float32)


def expand_val(x, rf, rt, wrong=None):
    """`expandVal` on every element of `x` (CPU). `wrong` selects a plausible wrong reading for the
    negative tests."""
    if x.dtype == torch.float32:
        a, b = f32_nearest(rf), f32_nearest(rt)
        if wrong == "bitwise":
            hit = bits(x) == bits(a.reshape(1))  # bit equality: -0.0 and +0.0 differ
        else:
            hit = x == a  # ordered IEEE comparison: NaN never, -0.0 == +0.0
        if wrong == "ignore":
            hit = torch.zeros_like(hit)
        if wrong == "unconverted":
            b = torch.tensor(rt, dtype=torch.int32).view(torch.float32)  # rt's int32 bits as float32
        return torch.where(hit, b.expand_as(x), x)
    assert x.dtype == torch.int32
    hit = x.to(torch.int64) == rf
    if wrong == "ignore":
        hit = torch.zeros_like(hit)
    return torch.where(hit, torch.tensor(rt, dtype=torch.int32).expand_as(x), x)


def expected(x, n, rf, rt, init, wrong=None):
    """`Post`'s first clause on the whole output: positions `start_r .. cu[r]-1` hold
    `expand_val(x[r])`; the others (`j >= cu[B-1]`) keep `init`. Computed from the host's inputs."""
    vals = expand_val(x.cpu(), rf, rt, wrong)
    counts = torch.tensor(n, dtype=torch.int64)
    rep = torch.repeat_interleave(vals, counts)
    out = init.clone()
    if wrong == "shift" and rep.numel() > 1:
        rep = torch.roll(rep, 1)
    out[: rep.numel()] = rep
    if wrong == "drop_last":
        cu = counts.cumsum(0)
        last = cu[counts > 0] - 1
        out[last] = init[last]
    if wrong == "zeros":
        out[: rep.numel()] = 0
    return out


def check_pre(cu, n, out_len):
    """`Pre`: each `n_r` in `[0, 128]`, `cu` their running sums, the total within the output."""
    cu = cu.to(torch.int64).cpu()
    assert all(0 <= k <= 128 for k in n)
    assert torch.equal(cu, torch.tensor(n, dtype=torch.int64).cumsum(0))
    assert int(cu[-1]) <= out_len


results = []
neg_results = []


def negatives(x, n, rf, rt, init, got_bits):
    """Wrong outputs `Post` must reject (those that differ from the expected one on this input);
    returns how many were rejected and how many were tried."""
    exp_bits = bits(expected(x, n, rf, rt, init))
    tried = rejected = 0
    for w in ["ignore", "bitwise", "unconverted", "shift", "drop_last", "zeros"]:
        wb = bits(expected(x, n, rf, rt, init, wrong=w))
        if torch.equal(wb, exp_bits):
            continue  # not a wrong output on this input
        tried += 1
        rejected += int(not torch.equal(wb, got_bits))
    return rejected, tried


def run_case(name, x, n, rf=0, rt=0, compare_host_reference=True, in_claim=True):
    """The host's launch (`expand_batch_to_tokens`), `num_tokens = sum(n)`."""
    B = len(n)
    N = sum(n)
    cu = torch.tensor(n, dtype=torch.int64).cumsum(0).to(torch.int32)
    if in_claim:
        check_pre(cu, n, N)
        assert 1 <= B <= 8192
    xd, cud = x.to(DEV), cu.to(DEV)
    before = [xd.clone(), cud.clone()]
    got = mod.expand_batch_to_tokens(xd, cud, N, replace_from=rf, replace_to=rt).cpu()
    assert got.shape == (N,) and got.dtype == x.dtype
    init = got.clone()  # every element is written (N = cu[B-1]); init is not read
    exp = expected(x, n, rf, rt, init)
    ok = torch.equal(bits(got), bits(exp))
    frame_ok = all(torch.equal(bits(a), bits(b)) for a, b in zip(before, [xd, cud]))
    ok_ref = torch.equal(bits(got), bits(mod.reference(x, n, rf, rt))) if compare_host_reference else None
    rej, tried = negatives(x, n, rf, rt, init, bits(got))
    neg_results.append((name, rej, tried))
    results.append((name, ok, frame_ok, ok_ref, N))
    print(f"{name}: {str(x.dtype)[6:]} B={B} N={N} {rf}->{rt}: output exact {ok} ({N} elements), "
          f"inputs unchanged {frame_ok}, host reference {ok_ref}, wrong outputs rejected {rej}/{tried}")


def run_guarded(name, x, n, rf=0, rt=0, tail=1000, guard=4096):
    """The host's launch line (source lines 139-148) on an output of `sum(n) + tail` elements carved
    from a sentinel pool: the tail (`j >= cu[B-1]`) and the guards must be unchanged."""
    B = len(n)
    N = sum(n)
    cu = torch.tensor(n, dtype=torch.int64).cumsum(0).to(torch.int32)
    check_pre(cu, n, N + tail)
    g = torch.Generator().manual_seed(7)
    pool = torch.randint(-2 ** 31, 2 ** 31, (guard + N + tail + guard,), generator=g, dtype=torch.int64)
    pool = pool.to(torch.int32).view(x.dtype).to(DEV)
    pool0 = pool.clone()
    out = pool[guard:guard + N + tail]
    init = out.clone().cpu()
    xd, cud = x.to(DEV), cu.to(DEV)
    mod.expand_kernel[(B,)](out, xd, cud, rf, rt, MAX_NUM_TOKENS=mod.MAX_SPEC_LEN, num_warps=4, num_stages=3)
    exp = expected(x, n, rf, rt, init)
    ok = torch.equal(bits(out.cpu()), bits(exp))
    frame_ok = torch.equal(bits(pool[:guard]), bits(pool0[:guard])) and torch.equal(
        bits(pool[guard + N + tail:]), bits(pool0[guard + N + tail:])) and torch.equal(
        bits(out[N:].cpu()), bits(init[N:]))
    rej, tried = negatives(x, n, rf, rt, init, bits(out.cpu()))
    neg_results.append((name, rej, tried))
    results.append((name, ok, frame_ok, None, N + tail))
    print(f"{name} (guarded direct launch): {str(x.dtype)[6:]} B={B} N={N} tail={tail} {rf}->{rt}: "
          f"output exact {ok}, tail and guards unchanged {frame_ok}, wrong outputs rejected {rej}/{tried}")


def specials_f32(B, g, rf=0.0):
    """`float32` values with every special kind: NaNs of several payloads and signs, infinities,
    signed zeros, subnormals, the largest finite, values equal and next to `float32(rf)`."""
    x = (torch.rand(B, generator=g) * 4 - 2).float()
    pats = torch.tensor([0x7FC00000, 0x7F800001, 0xFFC00001, 0x7FFFFFFF, 0x7F800000, 0xFF800000,
                         0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x7F7FFFFF, 0x3F800000],
                        dtype=torch.int64).to(torch.int32).view(torch.float32)
    a = f32_nearest(rf)
    near = torch.stack([a, torch.nextafter(a, torch.tensor(float("inf"))),
                        torch.nextafter(a, torch.tensor(float("-inf"))), -a])
    k = torch.randint(0, len(pats) + len(near) + 2, (B,), generator=g)
    allv = torch.cat([pats, near])
    sel = k < len(allv)
    x[sel] = allv[k[sel]]
    return x


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # The setup: the docstring's example (source lines 165-171).
    run_case("setup docstring example", torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32), [2, 3, 1])
    # The setup: the random batches (source lines 173-205), same generator, the three calls.
    for batch_size, k, mixed in [(1, 1, False), (8, 3, False), (64, 5, True), (256, 4, True),
                                 (1024, 8, True), (37, 128, True), (16, 2, False)]:
        n = torch.randint(0, k + 1, (batch_size,), generator=gen).tolist() if mixed else [k] * batch_size
        greedy = torch.rand(batch_size, generator=gen) < 0.3
        temperature = torch.where(greedy, 0.0, torch.rand(batch_size, generator=gen) * 2).float()
        top_k = torch.randint(0, 100, (batch_size,), generator=gen, dtype=torch.int32)
        top_p = torch.where(torch.rand(batch_size, generator=gen) < 0.2, 0.0,
                            torch.rand(batch_size, generator=gen)).float()
        run_case(f"setup batch {batch_size} k {k} temperature", temperature, n, 0, 1)
        run_case(f"setup batch {batch_size} k {k} top_k", top_k, n, 0, 0)
        run_case(f"setup batch {batch_size} k {k} top_p", top_p, n, 0, 0)
    # The setup: serving sizes, 1024 requests of MAX_SPEC_LEN = 128 tokens.
    g = torch.Generator().manual_seed(1)
    n = [128] * 1024
    temperature = torch.where(torch.rand(1024, generator=g) < 0.3, 0.0, 0.7).float()
    run_case("setup serving max temperature", temperature, n, 0, 1)
    run_case("setup serving max top_k", torch.randint(-1, 200, (1024,), generator=g, dtype=torch.int32), n, 0, 0)
    run_case("setup serving max top_p", torch.where(torch.rand(1024, generator=g) < 0.3, 0.0, 0.9).float(), n, 0, 0)

    # Edges of Inst.ok / Pre: 8192 requests, counts 0..128 (zero-token requests, the first one too).
    n = torch.randint(0, 129, (8192,), generator=g).tolist()
    n[0] = 0
    n[1:40] = [0] * 39
    run_case("edge B=8192 temperature, specials", specials_f32(8192, g, 0), n, 0, 1)
    run_case("edge B=8192 top_p, specials", specials_f32(8192, g, 0), n, 0, 0)
    run_case("edge B=8192 top_k", torch.randint(-2 ** 31, 2 ** 31, (8192,), generator=g,
                                                 dtype=torch.int64).to(torch.int32), n, 0, 0)
    # replace_from / replace_to at the int32 extremes, at 1 (do_not_specialize) and where float32
    # rounds (2^24 + 1 -> 2^24, 2^31 - 1 -> 2^31).
    n = torch.randint(0, 129, (3000,), generator=g).tolist()
    for rf, rt in [(1, 1), (1, 0), (-1, 2 ** 31 - 1), (2 ** 31 - 1, -2 ** 31), (-2 ** 31, 7),
                   (2 ** 24 + 1, 2 ** 24 + 3), (16777217, -5)]:
        x = specials_f32(3000, g, rf)
        run_case(f"edge float32 {rf}->{rt}", x, n, rf, rt)
        xi = torch.randint(-2 ** 31, 2 ** 31, (3000,), generator=g, dtype=torch.int64).to(torch.int32)
        xi[::3] = torch.tensor(rf, dtype=torch.int64).to(torch.int32)
        run_case(f"edge int32 {rf}->{rt}", xi, n, rf, rt)
    # One request; requests of 128 tokens each with the value matching.
    run_case("edge B=1 N=128", torch.tensor([0.0]), [128], 0, 1)
    run_case("edge B=1 N=0", torch.tensor([3], dtype=torch.int32), [0], 0, 0)
    # Frame: output longer than cu[B-1], initial contents arbitrary (guarded direct launches).
    n = torch.randint(0, 129, (777,), generator=g).tolist()
    run_guarded("frame float32", specials_f32(777, g, 0), n, 0, 1)
    run_guarded("frame int32", torch.randint(-5, 5, (777,), generator=g, dtype=torch.int32), n, 0, -9)
    n = [0, 0, 5, 0, 128, 0]
    run_guarded("frame zero-token requests", torch.tensor([0.0, -0.0, 0.0, 1.0, -0.0, 2.0]), n, 0, 1, tail=37)
    # Beyond Inst.ok: more than 8192 requests (the claim covers batch sizes up to 8192).
    n = torch.randint(0, 9, (70000,), generator=g).tolist()
    run_case("beyond Inst.ok: 70000 requests", specials_f32(70000, g, 0), n, 0, 1, in_claim=False)

    bad = [r for r in results if not (r[1] and r[2] and r[3] in (None, True))]
    bad += [r for r in neg_results if r[1] != r[2]]
    total_tried = sum(r[2] for r in neg_results)
    if total_tried == 0:
        bad.append(("no negative test", 0, 0))
    print(f"negative tests: {sum(r[1] for r in neg_results)} of {total_tried} wrong outputs rejected")
    print(f"{len(results)} launches, {sum(r[4] for r in results)} output elements checked; "
          f"worst error / tolerance: 0 / 0 (exact, bit for bit) on every element; frame held on every launch"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
