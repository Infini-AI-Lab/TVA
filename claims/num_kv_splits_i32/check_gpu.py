"""GPU check of the claim for `num_kv_splits_i32`.

Runs the kernel's own host (`source/metadata.py`) on the setup's launches (the backend's call
sites with model shapes, and the source's test cases) and direct launches with the host's launch
line on outputs carved from random-filled pools with guards, at the edges of `Inst.ok` and `Pre`,
and checks `Post` as `Claim.lean` states it:

* there is one integer `E` with `CoreOk` (`C <= E`, `E <= v + C * 2^-17`, `v - C * 2^-17 < E + 1`,
  `v = C * max(log2(L / 64), 1)`) such that element `j < num_seq * num_group` of `num_kv_splits`
  is `numSplits(E, j // num_group)`, exactly;
* with `uneven_batch`, its element `0` is `1` when `minLen < maxLen` and `0` otherwise;
* the frame: the inputs, the output's other elements and the guards are unchanged.

`v` is computed in `float64` (`math.log2`), whose error (about `1e-15` relative) is far below the
tolerance. The script also reports which `E` the device used. Negative tests: outputs a wrong
kernel would leave (every split count off by one, `E` far outside the set, the 80 % rule
ignored, a floor in place of a ceiling) must fail `Post`. Run with
`CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import math
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "metadata.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("num_kv_splits_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
H200_CORES = 132


def cdiv(a, b):
    return (a + b - 1) // b


def spec_values(lens, G, H, Hkv, K, C, floor_chunk=False, no_rule=False):
    """`maxLen`, `minLen`, `chunk1`, `tokenGrid` and a function `E -> numSplits(E, .)` of
    `Claim.lean` (with `floor_chunk` or `no_rule`, those of a wrong kernel)."""
    S = len(lens)
    L = max(lens)
    m0 = min(lens)
    mn = m0 if no_rule else (L if L * 8 < m0 * 10 else m0)
    div = (lambda a, b: a // b) if floor_chunk else cdiv
    k1 = min(cdiv(L, mn), K)
    c1 = max(div(L, k1), 1)
    g = H // Hkv
    tg = S * G * H if g == 1 else S * G * cdiv(H, min(16, g))

    def splits(E):
        k2 = min(cdiv(E, tg), K)
        c2 = max(div(L, k2), 1)
        return [max(cdiv(x, c1), cdiv(x, c2)) for x in lens]

    return L, mn, splits


def core_set(L, C):
    """The integers `E` with `CoreOk`."""
    v = C * max(math.log2(L / 64.0), 1.0)
    tau = C * 2.0 ** -17
    lo = max(C, math.floor(v - tau - 1) + 1)
    hi = math.floor(v + tau)
    return v, [E for E in range(lo, hi + 1) if v - tau < E + 1 and E <= v + tau]


def check_pre(lens, S, G, H, Hkv, K, C, MAX):
    assert 1 <= S <= 8192 and S <= MAX <= 2 ** 20 and MAX & (MAX - 1) == 0
    assert G >= 1 and MAX * G <= 2 ** 31 and 1 <= Hkv <= H and S * G * H <= 2 ** 30
    assert 1 <= K <= 2 ** 22 and 1 <= C <= 2 ** 16
    assert all(1 <= x <= 2 ** 22 for x in lens)


results = []
negatives = []
used = {}


def post_holds(out_vals, unev, lens, G, H, Hkv, K, C, **wrong):
    """`Post`'s values: some `E` of `CoreOk` gives every output; the uneven flag. Returns the
    matching `E`s (empty when `Post` fails)."""
    L, mn, splits = spec_values(lens, G, H, Hkv, K, C, **wrong)
    v, Es = core_set(L, C)
    good = []
    for E in Es:
        exp = [x for x in splits(E) for _ in range(G)]
        if out_vals == exp:
            good.append(E)
    if unev is not None and unev != (1 if mn < L else 0):
        return []
    return good


def run(name, lens, dt, G, H, Hkv, K, C=H200_CORES, uneven=True, MAX=None, host=True,
        guard=1024, extra=37):
    """One launch: through the host when `host` (its `SCHEDULE_SEQ`), else the host's launch line
    with `MAX_NUM_SEQ = MAX`, the output (`extra` elements longer than needed) and `uneven_batch`
    carved from random-filled pools with guards."""
    S = len(lens)
    if MAX is None:
        MAX = 256 if S < 256 else triton.next_power_of_2(S)
    check_pre(lens, S, G, H, Hkv, K, C, MAX)
    gen = torch.Generator().manual_seed(S * 7919 + G * 31 + H + K)
    seq = torch.tensor(lens, dtype=dt, device=DEV)
    seq_before = seq.clone()
    n = S * G
    pool = torch.randint(-2 ** 31, 2 ** 31, (guard + n + extra + guard,), generator=gen,
                         dtype=torch.int64).to(torch.int32).to(DEV)
    pool_before = pool.clone()
    out = pool[guard:guard + n] if host else pool[guard:guard + n + extra]
    upool = torch.randint(-2 ** 31, 2 ** 31, (2 * guard + 1,), generator=gen,
                          dtype=torch.int64).to(torch.int32).to(DEV)
    upool_before = upool.clone()
    ub = upool[guard:guard + 1] if uneven else None
    if host:
        mod.get_num_kv_splits(out, seq, H, Hkv, K, C, ub)
    else:
        mod.get_num_kv_splits_triton[(1,)](
            out, seq, S, G, H, Hkv, K, C, MAX_NUM_SEQ=MAX, uneven_batch_ptr=ub,
            num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    vals = pool[guard:guard + n].tolist()
    unev = int(upool[guard]) if uneven else None
    Es = post_holds(vals, unev, lens, G, H, Hkv, K, C)
    # the frame: everything but the written elements (and the flag) is unchanged
    fpool = pool.clone(); fpool[guard:guard + n] = pool_before[guard:guard + n]
    fupool = upool.clone()
    if uneven:
        fupool[guard] = upool_before[guard]
    frame = (torch.equal(seq, seq_before) and torch.equal(fpool, pool_before)
             and torch.equal(fupool, upool_before))
    L = max(lens)
    v, allowed = core_set(L, C)
    results.append((name, bool(Es), frame))
    used[name] = (Es, allowed, math.floor(v))
    # negative tests: each must fail Post
    wrong = [("every split + 1", [x + 1 for x in vals], unev, {}),
             ("80 % rule ignored", vals, unev, {"no_rule": True}),
             ("floor for ceiling", vals, unev, {"floor_chunk": True})]
    for what, wv, wu, kw in wrong:
        L_, mn_, sp_ = spec_values(lens, G, H, Hkv, K, C, **kw)
        L0, mn0, sp0 = spec_values(lens, G, H, Hkv, K, C)
        if kw and all(sp_(E) == sp0(E) for E in allowed) and (mn_ < L_) == (mn0 < L0):
            continue  # the wrong formula agrees on this batch: no test
        if kw:
            # the wrong kernel's output: its own formula at the device's E
            E0 = Es[0] if Es else allowed[0]
            wv = [x for x in sp_(E0) for _ in range(G)]
            wu = (1 if mn_ < L_ else 0) if uneven else None
        negatives.append((f"{name}: {what}", not post_holds(wv, wu, lens, G, H, Hkv, K, C)))
    far = [x for x in spec_values(lens, G, H, Hkv, K, C)[2](max(allowed) + 2 * G * H * S)
           for _ in range(G)]
    if far != vals:
        negatives.append((f"{name}: E far above the set", not post_holds(far, unev, lens, G, H, Hkv, K, C)))
    print(f"{name}: batch {S} group {G} heads {H}/{Hkv} max splits {K} cores {C} MAX {MAX} "
          f"{str(dt).split('.')[-1]} L {L}: Post {bool(Es)} (E {Es}, allowed {allowed}, "
          f"float64 floor {math.floor(v)}), frame {frame}")


gen = torch.Generator().manual_seed(0)


def rnd(bs, top, lo=1):
    return torch.randint(lo, top + 1, (bs,), generator=gen).tolist()


i64, i32 = torch.int64, torch.int32

# the backend's call sites (SETUP.md) with model shapes, through the host
run("decode llama8b bs1", [1], i64, 1, 32, 8, 8)
run("decode llama8b bs7", rnd(7, 4096), i64, 1, 32, 8, 8)
run("decode llama70b-tp8 bs256", rnd(256, 32768), i64, 1, 8, 1, 8)
run("decode dsv3-mla-tp8 bs300 cap256", rnd(300, 163840), i64, 1, 16, 1, 256)
run("decode qwen7b bs8192", rnd(8192, 4096), i64, 1, 28, 4, 8)
run("decode tile256 ctx131072 bs64", rnd(64, 131072), i64, 1, 32, 8, 512)
run("window llama8b bs100", [min(x, 4096) for x in rnd(100, 131072)], i64, 1, 32, 8, 8, uneven=False)
run("draft-window int32 bs33", rnd(33, 4097), i32, 1, 32, 8, 8, uneven=False)
run("multistep topk4 bs64", rnd(64, 8192), i64, 4, 32, 8, 8, uneven=False)
run("multistep topk8 bs1024", rnd(1024, 65536), i64, 8, 32, 8, 8, uneven=False)
# the source's test cases
run("test 1", [1], i64, 1, 32, 8, 8)
run("test 2", [4096] * 8, i64, 1, 32, 32, 8)
run("test 3", rnd(64, 32768), i64, 1, 128, 1, 8)
run("test 4", rnd(300, 131072), i64, 1, 64, 8, 16, uneven=False)
run("test 5", rnd(7, 2000), i32, 4, 32, 8, 8, uneven=False)
# edges of Inst.ok and Pre, direct launches with guards
for k in range(0, 16):  # L = 64 * 2^k: v = C * k is an integer (k >= 1)
    run(f"edge L=64*2^{k}", [64 * 2 ** k, 1 + k], i64, 1, 32, 8, 8, host=False, MAX=256)
for L in [1, 63, 64, 65, 127, 128, 129, 2 ** 22 - 1, 2 ** 22]:
    run(f"edge L={L}", [L], i32, 1, 32, 8, 8, host=False, MAX=1)
run("edge 80% rule just above", [1000, 801], i64, 1, 32, 8, 8, host=False, MAX=2)
run("edge 80% rule at", [1000, 800], i64, 1, 32, 8, 8, host=False, MAX=2)
run("edge all ones (S=G=H=Hkv=K=C=1)", [1], i32, 1, 1, 1, 1, C=1, host=False, MAX=1)
run("edge C=1 K=1", rnd(5, 9000), i64, 1, 32, 8, 1, C=1, host=False, MAX=8)
run("edge C=2^16", rnd(9, 2 ** 22), i64, 1, 32, 8, 2 ** 22, C=2 ** 16, host=False, MAX=16)
run("edge K=2^22", rnd(3, 2 ** 22), i32, 1, 1, 1, 2 ** 22, host=False, MAX=4)
run("edge group 64 heads", rnd(11, 50000), i64, 2, 128, 2, 64, host=False, MAX=16)
run("edge heads 20/2", rnd(11, 50000), i64, 1, 20, 2, 64, host=False, MAX=16)
run("edge heads 3/2", rnd(11, 50000), i32, 3, 3, 2, 64, host=False, MAX=16)
run("edge S=8192 MAX=8192", rnd(8192, 2 ** 22), i32, 1, 4, 4, 32, host=False, MAX=8192)
# `MAX_NUM_SEQ` up to `2^16` (a block of `2^20` lanes compiles for many minutes)
run("edge S=1 MAX=2^16", [777], i64, 1, 32, 8, 8, host=False, MAX=2 ** 16)
run("edge MAX*G=2^31", rnd(3, 1000), i64, 2 ** 15, 2, 1, 8, host=False, MAX=2 ** 16, guard=64,
    extra=5)
run("edge S*G*H=2^30", [5000, 6000], i64, 2 ** 13, 2 ** 16, 2 ** 16, 8, host=False, MAX=2,
    guard=64, extra=3)
run("edge uneven batch", [10, 20000, 5], i32, 1, 32, 8, 8, host=False, MAX=4)

bad = [r for r in results if not (r[1] and r[2])]
neg_bad = [n for n, ok in negatives if not ok]
dev_floor = sum(1 for k, (Es, allowed, fl) in used.items() if fl in Es)
print(f"{len(results)} launches; Post holds on {len(results) - len(bad)}; the outputs match the "
      f"float64 floor of v on {dev_floor}; {len(negatives)} wrong outputs, rejected "
      f"{len(negatives) - len(neg_bad)}")
if bad or neg_bad:
    for r in bad:
        print("FAIL", r)
    for n in neg_bad:
        print("NEGATIVE NOT REJECTED", n)
    sys.exit(1)
print("all OK")
