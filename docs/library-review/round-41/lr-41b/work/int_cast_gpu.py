"""lr-41b, round 41 (L185): `x.to(d')` between integer dtypes other than int1, on the device,
against the library's `IntV.cast` (`IntV.cast_eq_ofInt`: the source's value modulo 2^w', read
in d'). 8- and 16-bit sources exhaustively; 32- and 64-bit sources adversarially (edges, powers
of two +-1, random). Also `(pid*BM + tl.arange(0, BM)) % M` (L184's `irem` rows) in int32 and
int64, BM > M included, against (r0 + i) % M."""
import random, sys
import torch, triton, triton.language as tl

TY = {"int8": (torch.int8, tl.int8, True, 8), "int16": (torch.int16, tl.int16, True, 16),
      "int32": (torch.int32, tl.int32, True, 32), "int64": (torch.int64, tl.int64, True, 64),
      "uint8": (torch.uint8, tl.uint8, False, 8), "uint16": (torch.uint16, tl.uint16, False, 16),
      "uint32": (torch.uint32, tl.uint32, False, 32), "uint64": (torch.uint64, tl.uint64, False, 64)}

@triton.jit
def cast_k(x_ptr, y_ptr, n, DST: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, x.to(DST), mask=m)

def model(v, w2, s2):
    r = v % (1 << w2)
    if s2 and r >= (1 << (w2 - 1)):
        r -= 1 << w2
    return r

def bits_to_val(b, w, s):
    b %= 1 << w
    return b - (1 << w) if s and b >= (1 << (w - 1)) else b

def sources(w, s):
    if w <= 16:
        return [bits_to_val(b, w, s) for b in range(1 << w)]
    vals = set()
    for k in range(w + 1):
        for d in (-1, 0, 1):
            vals.add(bits_to_val((1 << k) + d, w, s))
            vals.add(bits_to_val(-(1 << k) + d, w, s))
    rng = random.Random(41)
    for _ in range(20000):
        vals.add(bits_to_val(rng.getrandbits(w), w, s))
    return sorted(vals)

def to_tensor(vals, tt, w, s):
    # build from raw bits through a same-width signed view (torch's uint32/64 have limited ops)
    sv = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}[w]
    raw = [bits_to_val(v, w, True) for v in vals]
    return torch.tensor(raw, dtype=sv).view(tt).cuda()

def from_tensor(t, w, s):
    sv = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}[w]
    raw = t.view(sv).cpu().tolist()
    return [bits_to_val(r, w, s) for r in raw]

total = bad = 0
for sn, (st, _, ss, sw) in TY.items():
    vals = sources(sw, ss)
    x = to_tensor(vals, st, sw, ss)
    for dn, (dt, dtl, ds, dw) in TY.items():
        y = torch.empty(len(vals), dtype=dt, device="cuda")
        cast_k[(triton.cdiv(len(vals), 1024),)](x, y, len(vals), DST=dtl, BLOCK=1024)
        got = from_tensor(y, dw, ds)
        nb = sum(1 for v, g in zip(vals, got) if g != model(v, dw, ds))
        total += len(vals); bad += nb
        print(f"{sn:>6} -> {dn:<6} {len(vals):6d} values, mismatches {nb}")

@triton.jit
def irem_k(out_ptr, M, BM: tl.constexpr, WIDE: tl.constexpr):
    pid = tl.program_id(0)
    if WIDE:
        rows = (pid.to(tl.int64) * BM + tl.arange(0, BM)) % M
    else:
        rows = (pid * BM + tl.arange(0, BM)) % M
    tl.store(out_ptr + pid * BM + tl.arange(0, BM), rows.to(tl.int64))

nirem = birem = 0
for wide in (False, True):
    for BM in (4, 16, 64):
        for M in (1, 3, 5, 17, 63, 100, 1000, 2**31 - 1):
            grid = 40
            out = torch.empty(grid * BM, dtype=torch.int64, device="cuda")
            irem_k[(grid,)](out, M, BM=BM, WIDE=wide)
            got = out.cpu().tolist()
            exp = [(p * BM + i) % M for p in range(grid) for i in range(BM)]
            nb = sum(1 for a, b in zip(got, exp) if a != b)
            nirem += len(exp); birem += nb
print(f"irem rows: {nirem} lanes, mismatches {birem}")
print(f"casts: {total} conversions, mismatches {bad}")
print("triton", triton.__version__, "torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
sys.exit(1 if bad or birem else 0)
