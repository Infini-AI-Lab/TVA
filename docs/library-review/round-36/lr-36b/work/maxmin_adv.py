"""lr-36b, library issue L164: adversarial checks of float tl.atomic_max / tl.atomic_min.

  A. Which dtypes Triton 3.8.0 accepts (float16, bfloat16, int8, int16, uint16, float32, float64).
  B. PTX for sem in {relaxed, acq_rel, acquire, release} x scope {gpu, cta, sys}, tensor and scalar.
  C. Scalar (one-lane) calls on every (old, operand) pair of special patterns, acq_rel: new bits =
     the sign-bit-picked integer atomic, returned value = old bits (scalar result path).
  D. Contention: 64 programs x 256 lanes, all on 8 cells, operands of mixed sign (no NaN):
     final value = float max/min of all operands and the initial value, bit-exact for zeros as
     the total order -0 < +0 the integer atomics give; with NaN operands: final value is one of
     the written values or the initial.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import math
import random
import re
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def kt(p, v, o, MAX: tl.constexpr, SEM: tl.constexpr, SC: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    if MAX:
        r = tl.atomic_max(p + j, tl.load(v + j), sem=SEM, scope=SC)
    else:
        r = tl.atomic_min(p + j, tl.load(v + j), sem=SEM, scope=SC)
    tl.store(o + j, r)


@triton.jit
def ks(p, v, o, MAX: tl.constexpr, SEM: tl.constexpr):
    i = tl.program_id(0)
    if MAX:
        r = tl.atomic_max(p + i, tl.load(v + i), sem=SEM)
    else:
        r = tl.atomic_min(p + i, tl.load(v + i), sem=SEM)
    tl.store(o + i, r)


@triton.jit
def kc(p, v, o, MAX: tl.constexpr, N: tl.constexpr, C: tl.constexpr):
    j = tl.program_id(0) * N + tl.arange(0, N)
    if MAX:
        r = tl.atomic_max(p + (j % C), tl.load(v + j), sem="relaxed")
    else:
        r = tl.atomic_min(p + (j % C), tl.load(v + j), sem="relaxed")
    tl.store(o + j, r)


def part_a():
    print("## A. dtypes accepted")
    for dt in (torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.float32, torch.float64,
               torch.int32, torch.uint32 if hasattr(torch, "uint32") else torch.int32):
        for mx in (True, False):
            try:
                p = torch.zeros(32, dtype=dt, device="cuda")
                kt[(1,)](p, p.clone(), p.clone(), mx, "relaxed", "gpu", 32)
                torch.cuda.synchronize()
                res = "compiles"
            except Exception as e:
                res = "rejected: " + str(e).strip().splitlines()[-1][:90]
            print(f"{dt} {'max' if mx else 'min'}: {res}")


def atoms(ptx):
    return sorted(set(re.sub(r"%\w+", "%_", a) for a in re.findall(r"(?:@!?%p\d+ )?atom\.[^;]*;", ptx)))


def part_b():
    print("## B. PTX")
    for dt in (torch.float32, torch.float64):
        for sem in ("relaxed", "acq_rel", "acquire", "release"):
            for sc in ("gpu", "cta", "sys"):
                p = torch.zeros(128, dtype=dt, device="cuda")
                h = kt[(1,)](p, p.clone(), p.clone(), True, sem, sc, 128)
                print(dt, sem, sc, "tensor max:", " | ".join(atoms(h.asm["ptx"])))
        for sem in ("relaxed", "acq_rel"):
            p = torch.zeros(4, dtype=dt, device="cuda")
            h = ks[(4,)](p, p.clone(), p.clone(), False, sem)
            a = atoms(h.asm["ptx"])
            print(dt, sem, "scalar min:", " | ".join(a), "| bar.sync" if "bar.sync" in h.asm["ptx"] else "")


W = {32: (torch.float32, torch.int32, "<f", "<I"), 64: (torch.float64, torch.int64, "<d", "<Q")}


def sg(w, b):
    return b - (1 << w) if b >> (w - 1) else b


def us(w, b):
    return b & ((1 << w) - 1)


def fv(w, b):
    return struct.unpack(W[w][2], struct.pack(W[w][3], b))[0]


def fb(w, x):
    return struct.unpack(W[w][3], struct.pack(W[w][2], x))[0]


def pick(w, mx, o, v):
    if v >> (w - 1):
        return min(o, v) if mx else max(o, v)
    a, b = sg(w, o), sg(w, v)
    return us(w, max(a, b) if mx else min(a, b))


def pats(w):
    nanp = 0x7fc00000 if w == 32 else 0x7ff8000000000000
    s = 1 << (w - 1)
    xs = [0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, 3.5e-45 if w == 32 else 5e-324]
    b = [fb(w, x) for x in xs] + [s | fb(w, xs[-1])]
    b += [nanp, nanp | s, nanp | 0x1234, nanp | s | 0x1234, (s - 1), (1 << w) - 1,  # max-payload NaNs
          (nanp ^ (1 << (w - 2))) | 1, ((nanp ^ (1 << (w - 2))) | 1) | s]          # signalling
    b += [fb(w, 1.0) - 1, fb(w, -1.0) + 1]
    return b


def part_c():
    print("## C. scalar calls, acq_rel, every pair")
    for w in (32, 64):
        fl, it = W[w][0], W[w][1]
        P = pats(w)
        pairs = [(o, v) for o in P for v in P]
        to = lambda xs: torch.tensor([sg(w, x) for x in xs], dtype=torch.int64).to(it).view(fl).cuda()
        for mx in (True, False):
            p, v = to([a for a, _ in pairs]), to([b for _, b in pairs])
            o = torch.zeros_like(p)
            ks[(len(pairs),)](p, v, o, mx, "acq_rel")
            torch.cuda.synchronize()
            nb = [us(w, x) for x in p.view(it).cpu().tolist()]
            rb = [us(w, x) for x in o.view(it).cpu().tolist()]
            bad = sum(nb[i] != pick(w, mx, a, b) for i, (a, b) in enumerate(pairs))
            badr = sum(rb[i] != a for i, (a, _) in enumerate(pairs))
            # zeros: which wins (keyed by the sign bits of old and operand)
            zs = {(a >> (w - 1), b >> (w - 1)): nb[i] >> (w - 1) for i, (a, b) in enumerate(pairs)
                  if fv(w, a) == 0 and fv(w, b) == 0}
            print(f"float{w} {'max' if mx else 'min'}: {len(pairs)} pairs, new != picked: {bad},"
                  f" ret != old: {badr}, zero sign bits (old, operand) -> new: {zs}")


def part_d():
    print("## D. contention, 64 programs x 256 lanes on 8 cells")
    rng = random.Random(36)
    for w in (32, 64):
        fl = W[w][0]
        for mx in (True, False):
            for withnan in (False, True):
                bad = 0
                for t in range(20):
                    N, G, C = 256, 64, 8
                    xs = [rng.choice([0.0, -0.0, rng.uniform(-1e3, 1e3), -math.inf, math.inf, 1e-45])
                          for _ in range(N * G)]
                    if withnan:
                        for i in rng.sample(range(N * G), 50):
                            xs[i] = math.nan if rng.random() < .5 else -math.nan
                    init = [rng.uniform(-10, 10) for _ in range(C)]
                    p = torch.tensor(init, dtype=fl, device="cuda")
                    v = torch.tensor(xs, dtype=fl, device="cuda")
                    o = torch.zeros_like(v)
                    kc[(G,)](p, v, o, mx, N, C)
                    torch.cuda.synchronize()
                    vb = [us(w, x) for x in v.view(W[w][1]).cpu().tolist()]
                    fin = [us(w, x) for x in p.view(W[w][1]).cpu().tolist()]
                    for c in range(C):
                        cur = us(w, fb(w, init[c]))
                        ops = [vb[j] for j in range(N * G) if j % C == c]
                        if not withnan:
                            for x in ops:
                                cur = pick(w, mx, cur, x)   # order-independent without NaN
                            bad += fin[c] != cur
                        else:
                            bad += fin[c] not in set(ops) | {cur}
                print(f"float{w} {'max' if mx else 'min'} nan={withnan}: cells wrong {bad} of 160")


if __name__ == "__main__":
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    part_a()
    part_b()
    part_c()
    part_d()
