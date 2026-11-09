"""lr-34b (round 34, L157): contended `tl.atomic_*` whose result is unused, every `sem`, scopes gpu and
sys (cta for information), with a random per-lane mask whose disabled lanes point at sentinel cells
that no enabled lane touches.

For each case: G programs of 256 lanes; enabled lane j of program p updates cell (7p + 13j + 5t) mod 64
(t = launch index); a disabled lane's pointer is a sentinel cell (64 + j mod 64), holding a pattern
(for floats -0.0 and a NaN with a payload) that must stay bit for bit. The final value of each
cell is checked against every order of the enabled updates (add: exact sums, integers wrap; and,
or, xor: the fold; max, min: the extreme, a float zero of either sign accepted when the extreme is
a zero; xchg: some enabled lane's value written to that cell, or the initial value if none; cas from
0 with nonzero values: some lane's value, or 0 if none). The PTX is scanned: the number of `atom`
instructions, no `red`, no `bar`.
"""
import re
import sys

import numpy as np
import torch
import triton
import triton.language as tl

C = 64
B = 256


@triton.jit
def k(ptr, val_ptr, msk_ptr, cmp_ptr, T, OP: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr, BLOCK: tl.constexpr,
      NC: tl.constexpr):
    p = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = tl.load(msk_ptr + p * BLOCK + j) != 0
    v = tl.load(val_ptr + p * BLOCK + j)
    cell = (7 * p + 13 * j + 5 * T) % NC
    sent = NC + j % NC
    ptrs = ptr + tl.where(m, cell, sent)
    if OP == 0:
        tl.atomic_add(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    elif OP == 1:
        tl.atomic_max(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    elif OP == 2:
        tl.atomic_min(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    elif OP == 3:
        tl.atomic_and(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    elif OP == 4:
        tl.atomic_or(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    elif OP == 5:
        tl.atomic_xor(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    elif OP == 6:
        tl.atomic_xchg(ptrs, v, mask=m, sem=SEM, scope=SCOPE)
    else:  # cas has no mask: disabled lanes compare with a value the sentinel never holds
        cmpv = tl.load(cmp_ptr + p * BLOCK + j)
        tl.atomic_cas(ptrs, cmpv, v, sem=SEM, scope=SCOPE)


OPS = {0: "add", 1: "max", 2: "min", 3: "and", 4: "or", 5: "xor", 6: "xchg", 7: "cas"}
CASES = [(0, torch.int32), (0, torch.int64), (0, torch.float32), (0, torch.float16), (0, torch.bfloat16),
         (0, torch.float64), (1, torch.int32), (1, torch.int64), (1, torch.float32), (2, torch.int32),
         (2, torch.float32), (2, torch.int64), (3, torch.int32), (3, torch.int64), (4, torch.int32), (5, torch.int64),
         (6, torch.int32), (6, torch.float32), (6, torch.int64), (7, torch.int32), (7, torch.int64)]


def gen(op, dt, G, rng):
    n = G * B
    if dt.is_floating_point:
        if op == 0:
            lo, hi = (-1, 2) if dt == torch.bfloat16 else (-2, 3)
            v = torch.from_numpy(rng.integers(lo, hi, n).astype(np.float64)).to(dt)
        elif op in (1, 2):
            base = rng.standard_normal(n) * 100
            base[rng.random(n) < 0.05] = 0.0
            base[rng.random(n) < 0.05] = -0.0
            v = torch.from_numpy(base).to(dt)
        else:
            v = torch.from_numpy(rng.standard_normal(n) * 100 + 1000.5).to(dt)
    else:
        info = torch.iinfo(dt)
        if op == 7:
            v = torch.from_numpy(rng.integers(1, 1 << 30, n)).to(dt)
        else:
            v = torch.from_numpy(rng.integers(info.min, info.max, n, dtype=np.int64)).to(dt)
    msk = torch.from_numpy((rng.random(n) < 0.7).astype(np.int32))
    return v, msk


def bits(t):
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]).tolist()


def check(op, dt, G, T, v, msk, init, final):
    """Returns the number of wrong cells (data cells and sentinels); numpy, every order of the updates."""
    fl = dt.is_floating_point
    vb = v.double().numpy() if fl else v.long().numpy()
    ml = msk.numpy() != 0
    p = np.repeat(np.arange(G), B)
    j = np.tile(np.arange(B), G)
    cell = (7 * p + 13 * j + 5 * T) % C
    xs, cs = vb[ml], cell[ml]
    fin = final.double().numpy() if fl else final.long().numpy()
    ini = init.double().numpy() if fl else init.long().numpy()
    acc = ini[:C].copy()
    if op == 0:
        np.add.at(acc, cs, xs)
        if not fl:
            w_ = dt.itemsize * 8
            if w_ < 64:
                acc = ((acc + (1 << (w_ - 1))) % (1 << w_)) - (1 << (w_ - 1))
        ok = fin[:C] == acc
    elif op in (1, 2):
        (np.maximum if op == 1 else np.minimum).at(acc, cs, xs)
        ok = fin[:C] == acc  # a float zero of either sign compares equal
    elif op in (3, 4, 5):
        (np.bitwise_and, np.bitwise_or, np.bitwise_xor)[op - 3].at(acc, cs, xs)
        ok = fin[:C] == acc
    else:
        ok = np.zeros(C, dtype=bool)
        for c in range(C):
            vals = xs[cs == c]
            ok[c] = np.isin(fin[c], vals) if len(vals) else fin[c] == (ini[c] if op == 6 else 0)
    w = int((~ok).sum())
    w += sum(1 for a, b in zip(bits(init[C:]), bits(final[C:])) if a != b)
    return w


def main():
    print(f"Triton {triton.__version__}, {torch.cuda.get_device_name()}")
    rng = np.random.default_rng(157)
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"{L} launches per case; G = 4096 programs (64 for float16/bfloat16 add)")
    print("| op | dtype | sem | scope | atom in PTX | red | bar | wrong cells (of 64 + 64 sentinels, all launches) |")
    print("|---|---|---|---|---|---|---|---|")
    tot = 0
    for op, dt in CASES:
        G = 64 if (op == 0 and dt in (torch.float16, torch.bfloat16)) else 4096
        for sem, scope in [("relaxed", "gpu"), ("acquire", "gpu"), ("release", "gpu"), ("relaxed", "sys"),
                           ("acq_rel", "gpu"), ("relaxed", "cta")]:
            wrong = 0
            ptx = None
            for T in range(L):
                v, msk = gen(op, dt, G, rng)
                cmpv = torch.zeros(G * B, dtype=dt)
                if op == 7:
                    # disabled lanes: compare with 1, which the sentinel (0x7ead...) never holds
                    cmpv = torch.where(msk != 0, torch.zeros_like(cmpv), torch.ones_like(cmpv))
                init = torch.zeros(2 * C, dtype=dt)
                if dt.is_floating_point:
                    if op in (1, 2):
                        init[:C] = torch.from_numpy(rng.standard_normal(C) * 100).to(dt)
                    elif op == 6:
                        init[:C] = 7.0
                    s16 = {torch.float16: 0x7E5A, torch.bfloat16: 0x7FA5}
                    sent = torch.empty(C, dtype=dt)
                    if dt in s16:
                        sent.view(torch.int16)[:] = s16[dt]
                        sent.view(torch.int16)[::2] = -32768  # -0.0
                    elif dt == torch.float32:
                        sent.view(torch.int32)[:] = 0x7FC0BEEF
                        sent.view(torch.int32)[::2] = -2 ** 31
                    else:
                        sent.view(torch.int64)[:] = 0x7FF8DEADBEEF0001
                        sent.view(torch.int64)[::2] = -2 ** 63
                    init[C:] = sent
                else:
                    if op in (1, 2, 3, 4, 5, 6):
                        info = torch.iinfo(dt)
                        init[:C] = torch.from_numpy(rng.integers(info.min, info.max, C, dtype=np.int64)).to(dt)
                    init[C:] = 0x7EAD0123
                cells = init.cuda()
                h = k[(G,)](cells, v.cuda(), msk.cuda(), cmpv.cuda(), T, op, sem, scope, B, C)
                torch.cuda.synchronize()
                if ptx is None:
                    ptx = h.asm["ptx"]
                wrong += check(op, dt, G, T, v, msk, init, cells.cpu())
            body = ptx.split(".entry", 1)[1]
            atoms = sorted(set(re.findall(r"\batom\.[a-z0-9.]+", body)))
            n_atom = len(re.findall(r"\batom\.", body))
            n_red = len(re.findall(r"\bred\.", body))
            n_bar = len(re.findall(r"\bbar\.|\bbarrier\.", body))
            if scope != "cta":
                tot += wrong
            print(f"| {OPS[op]} | {str(dt)[6:]} | {sem} | {scope} | {n_atom} ({', '.join(atoms)}) | {n_red} | {n_bar} | "
                  f"{wrong} |", flush=True)
    print(f"wrong cells at gpu/sys scope, all cases: {tot}")


if __name__ == "__main__":
    main()
