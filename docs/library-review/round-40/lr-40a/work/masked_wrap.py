"""L180 device check (lr-40a, round 40): a 2-D tile whose masked-off lanes' int32 offsets wrap.

M = 1000, BM = 64, s0 = 2149000, N = BN = 16, s1 = 1 (lr-39b's numbers). The matrix bound
(M-1)*s0 + (N-1)*s1 < 2^31 holds; rows 1000..1023 of the last tile (pid 15) wrap to negative
offsets. The buffer X sits 2^31 bytes into a larger int8 allocation, so a wrapped address is
mapped memory of ours (a canary), not a fault. Checks: the load gives each enabled lane its
element and `other` elsewhere; the masked store writes the enabled lanes and leaves the canaries
(the wrapped addresses) and everything else untouched. Also prints the LLVM IR flags on the
offset arithmetic and the address computation (nsw/nuw on mul/add, inbounds on getelementptr).
"""
import re, sys, datetime
import torch, triton, triton.language as tl

M, BM, S0, N, BN, S1 = 1000, 64, 2149000, 16, 16, 1
assert (M - 1) * S0 + (N - 1) * S1 < 2**31
assert (960 + BM - 1) * S0 >= 2**31


@triton.jit
def load_k(X, Y, M, N, s0, s1, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    offs = rows[:, None] * s0 + cols[None, :] * s1
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    x = tl.load(X + offs, mask=mask, other=-1)
    tl.store(Y + rows[:, None] * BN + cols[None, :], x)


@triton.jit
def store_k(X, M, N, s0, s1, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    offs = rows[:, None] * s0 + cols[None, :] * s1
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    v = ((rows[:, None] + cols[None, :]) % 50 + 2).to(tl.int8)
    tl.store(X + offs, v, mask=mask)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0),
          datetime.date.today().isoformat())
    base = 2**31
    big = torch.zeros(base + 2**31 + 2**20, dtype=torch.int8, device="cuda")
    X = big[base:]
    r = torch.arange(M, device="cuda", dtype=torch.int64)
    c = torch.arange(N, device="cuda", dtype=torch.int64)
    elem = (r[:, None] * S0 + c[None, :] * S1).reshape(-1)
    val = ((r[:, None] + c[None, :]) % 100 + 1).to(torch.int8).reshape(-1)
    X[elem] = val
    rw = torch.arange(M, 1024, device="cuda", dtype=torch.int64)
    wrapped = rw[:, None] * S0 + c[None, :] * S1
    wrapped = ((wrapped + 2**31) % 2**32 - 2**31).reshape(-1)  # int32 two's complement
    assert int(wrapped.max()) < 0 and int(wrapped.min()) + base >= 0
    big[base + wrapped] = 77
    before = big.clone()
    grid = (triton.cdiv(M, BM),)
    Y = torch.zeros(grid[0] * BM, BN, dtype=torch.int8, device="cuda")
    h = load_k[grid](X, Y, M, N, S0, S1, BM=BM, BN=BN)
    torch.cuda.synchronize()
    exp = torch.full((grid[0] * BM, BN), -1, dtype=torch.int8, device="cuda")
    exp[:M] = val.reshape(M, N)
    print("load: enabled lanes = element, masked-off lanes = other:", bool(torch.equal(Y, exp)))
    print("load: memory unchanged:", bool(torch.equal(big, before)))
    store_k[grid](X, M, N, S0, S1, BM=BM, BN=BN)
    torch.cuda.synchronize()
    sv = ((r[:, None] + c[None, :]) % 50 + 2).to(torch.int8).reshape(-1)
    print("store: enabled lanes written:", bool(torch.equal(X[elem], sv)))
    print("store: canaries at wrapped addresses kept:", bool((big[base + wrapped] == 77).all()))
    after = big.clone(); after[base + elem] = before[base + elem]
    print("store: nothing else changed:", bool(torch.equal(after, before)))
    for name, kern in (("load_k", h),):
        ll = kern.asm["llir"]
        flags = sorted(set(re.findall(r"= (mul|add|getelementptr|shl)( [a-z ]*?)? ", ll)))
        print(name, "llir op/flags:", flags)
        print(name, "nsw" in ll, "nuw" in ll, "inbounds" in ll)


if __name__ == "__main__":
    main()
