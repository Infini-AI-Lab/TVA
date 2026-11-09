"""GPU check of the prepared claim `Claims.RouteHistogram`.

Runs the kernel's own host, `route_histogram` of
`source/route_histogram.py` (imported by path), on the host's test
launches and the real routes of `SETUP.md` on both paths (`BINS == 0`, one atomic per pair, and
the `tl.histogram` path) and on launches at the edges of `Inst.ok`, and checks `Post` exactly as
it is stated:

* for every bucket `b < NUM_BUCKETS`, `counts[b] = init[b] + pairsIn d b` (the wrapping `int32`
  sum), with `pairsIn` the number of pairs `p < num_pairs` whose `bucketOf` is `b`, computed here
  from the host's inputs in Python integers (independently of the module's `reference`);
* the frame: the inputs are unchanged, and, in a second launch of the module's kernel with the
  host's own arguments but `counts` a slice of a larger buffer (sentinel guards, nonzero initial
  counts, some near the `int32` maximum so that the sum wraps), every element outside the first
  `NUM_BUCKETS` keeps its value.

Launches the host does not make but `Inst.ok` covers (`BINS > 0` with fewer than 16384 pairs,
`BINS == 0` with more than 16384 pairs and fewer than 512 buckets, a `BINS` above the host's,
`num_pairs` of 0 and 1) are run by that second launch only.

Negative tests: on every launch, wrong outputs (all counts left at their initial value, one pair
moved to another bucket, the last block's pairs missing, the invalid pairs not counted) must fail
`Post`.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import pathlib
import sys

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "route_histogram.py"
spec = importlib.util.spec_from_file_location("route_histogram_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
SENT = -987654321


def wrap32(x):
    return (x + 2**31) % 2**32 - 2**31


def pairs_in(slots, groups, gps, ml):
    """`pairsIn d b` for every bucket, from the host's inputs (`bucketOf`, Post)."""
    nb = gps * ml + 1
    s = slots.cpu().long()
    if groups is None:
        width, n = 1, s.numel()
        sl = s[:n]
        live = (sl >= 0) & (sl < ml)
        b = sl
    else:
        width, n = groups.shape[1], groups.numel()
        sl = s[torch.arange(n) // width]
        g = groups.reshape(-1).cpu().long()
        live = (sl >= 0) & (sl < ml) & (g >= 0)
        if gps == 1:
            b = sl
        else:
            live &= g < gps
            b = sl * gps + g
    b = torch.where(live, b, torch.full_like(b, nb - 1))
    return torch.bincount(b, minlength=nb).tolist()


def direct(slots, groups, gps, ml, bins, init, block=512, guard=32):
    """The host's launch (lines 159-184) with `counts` a guarded slice holding `init`."""
    nb = gps * ml + 1
    g = slots if groups is None else groups
    n = g.numel()
    buf = torch.full((guard + nb + guard,), SENT, dtype=torch.int32, device=DEVICE)
    buf[guard:guard + nb] = torch.tensor(init, dtype=torch.int32)
    mod._route_histogram_kernel[(triton.cdiv(max(n, 1), block),)](
        g, slots, buf[guard:guard + nb], n, NUM_BUCKETS=nb, GROUPS_PER_SLOT=gps, MAX_LORAS=ml,
        WIDTH=1 if groups is None else groups.shape[1], HAS_GROUPS=groups is not None, BLOCK=block,
        BINS=bins, USE_PDL=False, num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    out = buf.cpu().tolist()
    return out[:guard] + out[guard + nb:], out[guard:guard + nb], guard


def post(init, ref, out):
    """`Post` on `counts`' first `NUM_BUCKETS` elements: `init[b] + pairsIn d b`, wrapping."""
    return out == [wrap32(i + r) for i, r in zip(init, ref)]


def negatives(init, ref, n, slots, groups, gps, ml, block):
    """Wrong outputs that `Post` must reject (those that differ from the right one)."""
    right = [wrap32(i + r) for i, r in zip(init, ref)]
    wrong = [list(init)]  # nothing added
    if n > 0:
        moved = list(right)
        b = next(i for i, r in enumerate(ref) if r > 0)
        moved[b] = wrap32(moved[b] - 1)
        moved[(b + 1) % len(moved)] = wrap32(moved[(b + 1) % len(moved)] + 1)
        wrong.append(moved)  # one pair in the wrong bucket
        last = (n - 1) // block * block  # the pairs of the last program
        if groups is None:
            ref_drop = pairs_in(slots[:last], None, gps, ml)
        else:
            w = groups.shape[1]
            keep = groups.reshape(-1)[:last]
            pad = (-last) % w
            g2 = torch.cat([keep, torch.full((pad,), -1, dtype=keep.dtype, device=keep.device)])
            ref_drop = pairs_in(slots, g2.reshape(-1, w), gps, ml)
            ref_drop[-1] -= pad  # the padding pairs are not pairs
        wrong.append([wrap32(i + r) for i, r in zip(init, ref_drop)])  # last block missing
        wrong.append(right[:-1] + [init[-1]])  # invalid pairs not counted
    wrong = [w for w in wrong if w != right]
    return all(not post(init, ref, w) for w in wrong), len(wrong)


results = []
gen = torch.Generator().manual_seed(2)


def check(name, slots, groups, gps, ml, host=True, bins=None, block=512):
    nb = gps * ml + 1
    n = slots.numel() if groups is None else groups.numel()
    hb = mod._histogram_bins(nb, n)
    bins = hb if bins is None else bins
    assert bins == 0 or bins > nb, (name, bins, nb)
    ref = pairs_in(slots, groups, gps, ml)
    before = (slots.clone(), None if groups is None else groups.clone())
    ok_host = True
    if host:
        got = mod.route_histogram(slots, groups, groups_per_slot=gps, max_loras=ml)
        torch.cuda.synchronize()
        ok_host = got.cpu().tolist() == ref
    init = torch.randint(-2**31, 2**31 - 1, (nb,), generator=gen, dtype=torch.int64).tolist()
    init[0] = 2**31 - 1 - (ref[0] // 2)  # wraps when the count is at least 2
    guards, got2, _ = direct(slots, groups, gps, ml, bins, init, block=block)
    ok_direct = post(init, ref, got2)
    frame = all(v == SENT for v in guards) and torch.equal(slots, before[0]) and (
        groups is None or torch.equal(groups, before[1]))
    ok_neg, n_neg = negatives(init, ref, n, slots, groups, gps, ml, block)
    ok = ok_host and ok_direct and frame and ok_neg
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: pairs {n}, buckets {nb}, BINS {bins}, BLOCK {block}: "
          f"host exact {ok_host if host else '-'}, init+count exact {ok_direct}, frame {frame}, "
          f"{n_neg} wrong outputs rejected {ok_neg}")


# The host's test launches (source lines 213-224), then the real routes of SETUP.md on the path
# BINS == 0: MoE LoRA with 128 experts x 4 adapters (513 buckets) and 256 x 8, at least 8192 route
# buckets with few pairs, and a route of fewer than 16384 pairs.
for name, nt, width, gps, ml, ng in [
        ("dense", 16384, None, 1, 1, None), ("dense", 16385, None, 1, 4, None),
        ("dense", 100000, None, 1, 8, None), ("dense", 32768, None, 1, 16, None),
        ("moe per expert", 2048, 8, 64, 4, 128), ("moe per expert", 8193, 2, 8, 8, 8),
        ("moe per expert", 16384, 8, 32, 8, 32), ("moe shared outer", 3000, 6, 1, 8, 64),
        ("pairs (heads)", 1024, 1, 32, 4, 32),
        ("dense, one atomic per pair", 1000, None, 1, 8, None),
        ("moe, one atomic per pair", 4096, 8, 128, 4, 128),
        ("moe lora 128 experts x 4 adapters", 16384, 8, 128, 4, 128),
        ("moe lora 256 experts x 8 adapters", 65536, 8, 256, 8, 256),
        ("moe lora 384 experts x 32 adapters", 96, 8, 384, 32, 384),
        ("moe shared outer, few pairs", 2000, 6, 1, 8, 64)]:
    slots = torch.randint(-1, ml + 1, (nt,), generator=gen, dtype=torch.int32)
    if name.startswith("pairs"):
        slots = slots.repeat_interleave(gps)
        groups = torch.arange(gps, dtype=torch.int32).repeat(nt).reshape(-1, 1)
    elif width is None:
        groups = None
    else:
        groups = torch.randint(-1, ng, (nt, width), generator=gen, dtype=torch.int32)
    check(f"host {name} {nt}", slots.to(DEVICE), None if groups is None else groups.to(DEVICE), gps, ml)


def wild(shape):
    x = torch.randint(-2**31, 2**31 - 1, shape, generator=gen, dtype=torch.int64)
    return x.to(torch.int32)


# Edges of Inst.ok: any int32 values (slots and groups of every sign and size), NUM_BUCKETS = 511
# (BINS = 512, the host's largest), one group per slot with any groups, a large launch, and the
# launches only Inst.ok covers (fewer than 16384 pairs, num_pairs 0 and 1, a larger BINS).
s = wild((50001,)); s[::3] = torch.randint(-2, 9, (s[::3].numel(),), generator=gen, dtype=torch.int32)
check("edge any slots, dense", s.to(DEVICE), None, 1, 7)
s = torch.randint(-1, 6, (20000,), generator=gen, dtype=torch.int32)
g = wild((20000, 3)); g[:, 0] = torch.randint(-2, 90, (20000,), generator=gen, dtype=torch.int32)
check("edge any groups, 85 groups x 6 slots (511 buckets)", s.to(DEVICE), g.to(DEVICE), 85, 6)
check("edge any groups, one group per slot", s.to(DEVICE), g.to(DEVICE), 1, 5)
s = torch.randint(-1, 9, (2**26,), generator=gen, dtype=torch.int32)
check("edge 2^26 pairs", s.to(DEVICE), None, 1, 8)
del s
s = torch.randint(-1, 5, (3000,), generator=gen, dtype=torch.int32)
g = torch.randint(-1, 9, (3000, 4), generator=gen, dtype=torch.int32)
check("edge 12000 pairs, BINS > 0 (not the host's BINS)", s.to(DEVICE), g.to(DEVICE), 8, 4, host=False, bins=64)
check("edge BINS above the host's", s.to(DEVICE), g.to(DEVICE), 8, 4, host=False, bins=1024)
check("edge BLOCK 128", s.to(DEVICE), g.to(DEVICE), 8, 4, host=False, bins=64, block=128)
check("edge BLOCK 2048", s.to(DEVICE), g.to(DEVICE), 8, 4, host=False, bins=64, block=2048)
s = wild((30000,)); s[::2] = torch.randint(-2, 7, (s[::2].numel(),), generator=gen, dtype=torch.int32)
g = wild((30000, 8)); g[:, :5] = torch.randint(-2, 130, (30000, 5), generator=gen, dtype=torch.int32)
check("edge BINS 0, any slots and groups, 128 groups x 5 slots", s.to(DEVICE), g.to(DEVICE), 128, 5)
s = torch.randint(-1, 9, (2**25,), generator=gen, dtype=torch.int32)
check("edge BINS 0 (not the host's BINS), 2^25 pairs, 9 buckets", s.to(DEVICE), None, 1, 8, host=False, bins=0)
del s
s = torch.randint(-1, 5, (3000,), generator=gen, dtype=torch.int32)
g = torch.randint(-1, 9, (3000, 4), generator=gen, dtype=torch.int32)
check("edge BINS 0, BLOCK 128", s.to(DEVICE), g.to(DEVICE), 8, 4, host=False, bins=0, block=128)
check("edge BINS 0, one pair", torch.tensor([2], dtype=torch.int32, device=DEVICE), None, 1, 3, host=False, bins=0)
check("edge BINS 0, no pair", torch.zeros((0,), dtype=torch.int32, device=DEVICE), None, 1, 3, host=False, bins=0)
check("edge one pair", torch.tensor([0], dtype=torch.int32, device=DEVICE), None, 1, 3, host=False, bins=8)
check("edge no pair", torch.zeros((0,), dtype=torch.int32, device=DEVICE), None, 1, 3, host=False, bins=8)

n_bad = results.count(False)
print(f"{len(results)} launches: every count exact and every frame clause held" if n_bad == 0
      else f"{n_bad} of {len(results)} launches FAILED")
sys.exit(1 if n_bad else 0)
