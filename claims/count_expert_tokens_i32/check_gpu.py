"""GPU check of the prepared claim `Claims.CountExpertTokens` (claims/README.md, check 4).

Runs the kernel's own host, `count_expert_num_tokens` of
`source/count_expert_tokens.py` (imported by path), on the host's
test launches and on launches at the edges of `Inst.ok`, and checks `Post` exactly as it is
stated:

* `countOf d e` for every local expert `e`: the number of positions `j < numel` with
  `routed d j = e`, where `routed` is the entry itself without a map, and `expert_map[entry]` for
  an entry `>= 0` (`-1` for a negative one) with a map; computed here on the CPU from the host's
  inputs, in Python integers, independently of the module's own `reference`;
* the frame: the inputs are unchanged, and, in a second launch of the module's kernel with the
  host's own arguments but an output that is a slice of a larger buffer filled with a sentinel,
  every element outside the first `num_local_experts` keeps the sentinel.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import pathlib
import sys

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "source" / "count_expert_tokens.py"
spec = importlib.util.spec_from_file_location("count_expert_tokens_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
SENTINEL = -123456789


def post_counts(topk_ids, num_local, expert_map):
    """`countOf d e` for `e < num_local`, from the host's inputs (Post)."""
    ids = topk_ids.flatten().cpu().tolist()
    emap = None if expert_map is None else expert_map.cpu().tolist()
    counts = [0] * num_local
    for a in ids:
        r = a if emap is None else (emap[a] if a >= 0 else -1)
        if 0 <= r < num_local:
            counts[r] += 1
    return counts


def frame_launch(topk_ids, num_local, expert_map, guard=64):
    """The host's launch (source lines 87-105) with the output a slice of a guarded buffer."""
    buf = torch.full((guard + num_local + guard,), SENTINEL, dtype=torch.int32, device=topk_ids.device)
    out = buf[guard:guard + num_local]
    B = triton.next_power_of_2(min(topk_ids.numel(), 1024))
    mod._count_expert_num_tokens[(num_local,)](
        topk_ids, out, num_local, topk_ids.numel(), expert_map,
        HAS_EXPERT_MAP=expert_map is not None, BLOCK_SIZE=B, num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    return buf, guard


results = []


def check(name, topk_ids, num_local, expert_map):
    ids_before = topk_ids.clone()
    map_before = None if expert_map is None else expert_map.clone()
    got = mod.count_expert_num_tokens(topk_ids, num_local, expert_map)
    torch.cuda.synchronize()
    ref = post_counts(topk_ids, num_local, expert_map)
    exact = got.cpu().tolist() == ref
    inputs_kept = torch.equal(topk_ids, ids_before) and (
        expert_map is None or torch.equal(expert_map, map_before))
    buf, g = frame_launch(topk_ids, num_local, expert_map)
    fb = buf.cpu().tolist()
    frame = all(v == SENTINEL for v in fb[:g] + fb[g + num_local:]) and fb[g:g + num_local] == ref
    ok = exact and inputs_kept and frame
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name}: numel {topk_ids.numel()} {str(topk_ids.dtype)[6:]} "
          f"local {num_local} map {expert_map is not None}: counts exact {exact}, frame {frame and inputs_kept}")


gen = torch.Generator().manual_seed(1)

# The host's test launches (source lines 146-180).
cases = []
for nt in [1, 4, 11, 127, 128, 3333, 7317]:
    for topk, ep in [(2, 1), (6, 2), (8, 4)]:
        cases.append((nt, topk, 64, ep, torch.int64, "linear", 0.0))
for numel in [1, 112, 1024, 1025, 2047, 8191]:
    cases.append((numel, 1, 32, 2, torch.int64, "linear", 0.0))
cases += [(4096, 8, 256, 8, torch.int32, "linear", 0.0), (16384, 8, 256, 16, torch.int32, "round_robin", 0.0),
          (2000, 8, 128, 1, torch.int32, "linear", 0.05), (513, 8, 128, 3, torch.int32, "linear", 0.05),
          (777, 2, 8, 1, torch.int32, "linear", 0.0), (1, 8, 256, 8, torch.int32, "round_robin", 0.0)]
for nt, topk, E, ep, dt, placement, inv in cases:
    ids = mod.make_topk_ids(nt, topk, E, dt, gen, inv)
    for rank in sorted({0, ep - 1}):
        nl, emap = mod.make_expert_map(E, ep, rank, placement)
        check(f"host {nt}x{topk} E{E} ep{rank}/{ep}", ids.to(DEVICE), nl, emap.to(DEVICE))
        local = torch.where(ids >= 0, emap.long()[ids.long().clamp(min=0)], -1).to(dt)
        check(f"host {nt}x{topk} E{E} ep{rank}/{ep} local ids", local.to(DEVICE), nl, None)

# Edges of Inst.ok: any values without a map (int64 values whose low 32 bits are a local id,
# negatives other than -1, values past num_local), any int32 map entries (negative, past
# num_local), entries of every sign with a map; numel = 1 (a compile-time `topk_numel`),
# num_local = 1, numel not a multiple of BLOCK_SIZE, a large numel and many programs.
for dt in (torch.int32, torch.int64):
    lo, hi = (-2**31, 2**31 - 1) if dt == torch.int32 else (-2**62, 2**62)
    wild = torch.randint(lo, hi, (5000,), generator=gen, dtype=torch.int64)
    wild[::7] = torch.randint(-3, 40, (wild[::7].numel(),), generator=gen)
    if dt == torch.int64:
        wild[1::11] = 2**32 + torch.randint(0, 40, (wild[1::11].numel(),), generator=gen)
    check(f"edge any values, no map", wild.to(dt).to(DEVICE), 37, None)
    G = 300
    ent = torch.randint(-5, G, (3001, 3), generator=gen).to(dt)
    emap = torch.randint(-2**31, 2**31 - 1, (G,), generator=gen, dtype=torch.int64).to(torch.int32)
    emap[::3] = torch.randint(-2, 20, (emap[::3].numel(),), generator=gen, dtype=torch.int32)
    check(f"edge map with any entries", ent.to(DEVICE), 17, emap.to(DEVICE))
    check(f"edge numel 1", torch.tensor([[0]], dtype=dt, device=DEVICE), 1, None)
    check(f"edge numel 1 map", torch.tensor([[2]], dtype=dt, device=DEVICE), 3, torch.tensor([5, 1, 0], dtype=torch.int32, device=DEVICE))
    check(f"edge one expert", torch.randint(-1, 2, (999, 2), generator=gen).to(dt).to(DEVICE), 1, None)
# large: 2^26 int32 entries (256 MiB), 8 programs; many programs: 4096 local experts.
big = torch.randint(-1, 8, (2**26,), generator=gen, dtype=torch.int32)
check("edge numel 2^26", big.to(DEVICE), 8, None)
del big
check("edge 4096 programs", torch.randint(-1, 4096, (40000, 8), generator=gen, dtype=torch.int32).to(DEVICE), 4096, None)

n_bad = results.count(False)
print(f"{len(results)} launches: every count exact and every frame clause held" if n_bad == 0
      else f"{n_bad} of {len(results)} launches FAILED")
sys.exit(1 if n_bad else 0)
