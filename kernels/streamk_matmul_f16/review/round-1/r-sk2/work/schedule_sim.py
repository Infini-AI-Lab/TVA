# Simulate the kernel's schedule (lines 36-64) with int32 semantics (truncating // and %,
# wrapping + - *, each checked for wrap) on instances that meet Inst.ok, and check:
#  - every while loop ends, every iteration advances `it`;
#  - every tile's T iterations are done exactly once over all programs;
#  - a tile is stored (line 61) by exactly one program and touched by no other, or else it is
#    atomically added (line 63) by every program that touches it and stored by none;
#  - at most T programs touch a tile; no int32 operation wraps.
import random

W = 2 ** 31

def chk(v):
    assert -W <= v < W, v
    return v

def tdiv(a, b):
    q = abs(a) // abs(b)
    return chk(q if (a >= 0) == (b >= 0) else -q)

def trem(a, b):
    return chk(a - tdiv(a, b) * b)

def cdiv(a, b):
    return (a + b - 1) // b

def run(M, N, K, np_, ipp, total):
    num_pid_n = tdiv(chk(N + 127), 128)
    T = tdiv(chk(K + 63), 64)
    stores, adds, done = {}, {}, {}
    for pid in range(np_):
        it = chk(pid * ipp)
        end = min(chk(it + ipp), total)
        trips = 0
        while it < end:
            tile = tdiv(it, T)
            k_first = chk(it - chk(tile * T))
            k_last = min(T, chk(k_first + chk(end - it)))
            pid_m = tdiv(tile, num_pid_n)
            pid_n = trem(tile, num_pid_n)
            chk(pid_m * 128 + 127); chk(pid_n * 128 + 127)
            assert pid_m * 128 < M and pid_n * 128 < N  # the tile has an element of c
            assert k_last > k_first
            for kk in range(k_first, k_last):
                chk(kk * 64 + 63)
                done[(tile, kk)] = done.get((tile, kk), 0) + 1
            if k_first == 0 and k_last == T:
                stores.setdefault(tile, []).append(pid)
            else:
                adds.setdefault(tile, []).append(pid)
            it = chk(it + chk(k_last - k_first))
            trips += 1
            assert trips <= ipp
    ntiles = cdiv(M, 128) * cdiv(N, 128)
    assert len(done) == ntiles * T and all(v == 1 for v in done.values())
    for t in range(ntiles):
        s, a = stores.get(t, []), adds.get(t, [])
        assert (len(s) == 1 and not a) or (not s and len(a) >= 2), (t, s, a)
        assert len(s) + len(a) <= T
    return sum(1 for t in range(ntiles) if t in stores), sum(1 for t in range(ntiles) if t in adds)

tests = [(512, 512, 512, 132), (1000, 1000, 1000, 132), (3072, 4096, 1024, 132),
         (257, 129, 4097, 7), (64, 64, 64, 3), (8192, 8192, 4096, 132)]
for M, N, K, np_ in tests:
    total = cdiv(M, 128) * cdiv(N, 128) * cdiv(K, 64)
    ipp = cdiv(total, np_)
    st, ad = run(M, N, K, np_, ipp, total)
    print(f"M={M} N={N} K={K} np={np_}: total={total} ipp={ipp} np*ipp={np_*ipp} stored tiles={st} split tiles={ad}")

random.seed(1)
n = 0
for _ in range(3000):
    M = random.choice([1, 2, 127, 128, 129, 300, 1000, random.randint(1, 3000)])
    N = random.choice([1, 2, 127, 128, 129, 300, 1000, random.randint(1, 3000)])
    K = random.choice([1, 63, 64, 65, 128, 1000, 4097, 8192, random.randint(1, 8192)])
    total = cdiv(M, 128) * cdiv(N, 128) * cdiv(K, 64)
    np_ = random.choice([1, 2, 3, 7, 132, total, total + 5, random.randint(1, 2 * total + 2)])
    # any ipp that covers (cover), not only the host's cdiv
    lo = cdiv(total, np_)
    ipp = random.choice([lo, lo, lo + 1, lo + random.randint(0, 50), total])
    if np_ * ipp >= W:
        continue
    run(M, N, K, np_, ipp, total)
    n += 1
print("random instances checked:", n)
