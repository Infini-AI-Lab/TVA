# The schedule's int32 arithmetic (lines 37-49, 52, 64) at the largest instances Inst.ok allows,
# for the first, last and some middle programs, every value checked against [-2^31, 2^31).
W = 2 ** 31

def chk(v):
    assert -W <= v < W, v
    return v

def cdiv(a, b):
    return (a + b - 1) // b

def prog(M, N, K, ipp, total, pid):
    num_pid_n = chk(N + 127) // 128
    T = chk(K + 63) // 64
    it = chk(pid * ipp)
    end = min(chk(it + ipp), total)
    n = 0
    while it < end:
        tile = it // T
        k_first = chk(it - chk(tile * T))
        k_last = min(T, chk(k_first + chk(end - it)))
        pid_m, pid_n = tile // num_pid_n, tile % num_pid_n
        chk(pid_m * 128 + 127); chk(pid_n * 128 + 127)
        chk((k_last - 1) * 64 + 63)
        assert pid_m * 128 < M and pid_n * 128 < N
        it = chk(it + chk(k_last - k_first))
        n += 1
    return n

cases = [
    # M at its bound (M + 128 <= 2^31), N = 1, K = 2^13: total just below 2^31
    (2 ** 31 - 128, 1, 2 ** 13, 1),
    (2 ** 31 - 128, 1, 2 ** 13, 132),
    (1, 2 ** 31 - 128, 2 ** 13, 7),
    (2 ** 31 - 128, 1, 1, 2 ** 24 - 1),   # one iteration per program, many programs
]
for M, N, K, np_ in cases:
    total = cdiv(M, 128) * cdiv(N, 128) * cdiv(K, 64)
    ipp = cdiv(total, np_)
    assert np_ * ipp < W, "progs_lt fails"
    pids = sorted({0, 1, np_ // 2, np_ - 2, np_ - 1} & set(range(np_)))
    if total > 10 ** 7 and np_ == 1:
        pids = [0]
    its = {p: prog(M, N, K, ipp, total, p) for p in pids}
    print(f"M={M} N={N} K={K} np={np_} total={total} ipp={ipp} np*ipp={np_*ipp}: while-iterations {its}")
