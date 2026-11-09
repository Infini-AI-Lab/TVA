"""The accumulator of an int8 `tl.dot` (library issue L141, round 32).

  A1  which accumulator dtypes the front end and the compiler accept for `tl.dot(a, b, acc)` with int8
      operands (`acc` loaded from a buffer of that dtype), with `out_dtype` omitted, equal to the
      accumulator's dtype, and `tl.int32`: int32, uint32, int8, int16, int64, uint8, uint16, uint64,
      float16, float32; accepted or rejected on the host, and the result's dtype.
  A2  a uint32 accumulator, on both paths (mma.sync at (16, 16) with 4 warps, wgmma at (64, 64) with 4
      warps), K = 32, 64 and 256, accumulator values in [0, 2^32) with every value >= 2^31 included,
      operands in [-128, 127], with room for the int32 reading (|bits read as int32| + sum|a*b| <= 2^31 - 1):
      is the result, an int32, equal to the accumulator's bits read as int32 plus the exact sum, on every
      element? With out_dtype omitted and out_dtype=tl.uint32.
  A3  the result's dtype in the kernel: `c < 0` and `c.to(tl.float32)` where the int32 and uint32
      readings differ.

Run from TVA's root (MLIR prints its diagnostics of the rejected cases to stdout; output.txt keeps the result lines):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-int8-round32/run.py 2>/dev/null | grep -E "^(Triton|A[0-9] )"
"""
import torch
import triton
import triton.language as tl

TL = {"int32": tl.int32, "uint32": tl.uint32, "int8": tl.int8, "int16": tl.int16, "int64": tl.int64,
      "uint8": tl.uint8, "uint16": tl.uint16, "uint64": tl.uint64, "float16": tl.float16, "float32": tl.float32}
TORCH = {"int32": torch.int32, "uint32": torch.uint32, "int8": torch.int8, "int16": torch.int16,
         "int64": torch.int64, "uint8": torch.uint8, "uint16": torch.uint16, "uint64": torch.uint64,
         "float16": torch.float16, "float32": torch.float32}


@triton.jit
def kdot(a_ptr, b_ptr, c0_ptr, c_ptr, neg_ptr, f_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         OUT: tl.constexpr, OUTD: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
    acc = tl.load(c0_ptr + rm[:, None] * BN + rn[None, :])
    if OUT == 0:
        c = tl.dot(a, b, acc)
    else:
        c = tl.dot(a, b, acc, out_dtype=OUTD)
    tl.static_assert(c.dtype == tl.int32)
    tl.store(c_ptr + rm[:, None] * BN + rn[None, :], c)
    tl.store(neg_ptr + rm[:, None] * BN + rn[None, :], (c < 0).to(tl.int32))
    tl.store(f_ptr + rm[:, None] * BN + rn[None, :], c.to(tl.float32))


def launch(a, b, c0, BM, BN, BK, out, outd, nw):
    c = torch.empty(BM, BN, dtype=torch.int32, device="cuda")
    neg = torch.empty(BM, BN, dtype=torch.int32, device="cuda")
    f = torch.empty(BM, BN, dtype=torch.float32, device="cuda")
    kdot[(1,)](a, b, c0, c, neg, f, BM=BM, BN=BN, BK=BK, OUT=out, OUTD=outd, num_warps=nw)
    torch.cuda.synchronize()
    return c, neg, f


def a1():
    g = torch.Generator(device="cuda").manual_seed(1)
    a = torch.randint(-3, 4, (16, 32), dtype=torch.int8, device="cuda", generator=g)
    b = torch.randint(-3, 4, (32, 16), dtype=torch.int8, device="cuda", generator=g)
    s = (a.double() @ b.double()).to(torch.int64)
    lines = []
    for name in TL:
        c0 = torch.full((16, 16), 5, dtype=TORCH[name], device="cuda")
        for out, outd, label in ((0, tl.int32, "out_dtype omitted"), (1, TL[name], f"out_dtype=tl.{name}"),
                                 (1, tl.int32, "out_dtype=tl.int32")):
            if out == 1 and name == "int32" and label == "out_dtype=tl.int32":
                continue
            try:
                c, _, _ = launch(a, b, c0, 16, 16, 32, out, outd, 4)
                ok = bool((c.to(torch.int64) == s + 5).all())
                lines.append(f"A1 acc {name}, {label}: accepted, result int32, equal to 5 + the exact sum: {ok}")
            except Exception as e:
                msg = str(e).strip().splitlines()
                first = next((m for m in msg if "assert" in m.lower() or "error" in m.lower() or "expected" in m),
                             msg[0] if msg else "")
                lines.append(f"A1 acc {name}, {label}: rejected on the host ({type(e).__name__}: {first.strip()[:110]})")
    return lines


def a2():
    lines = []
    g = torch.Generator(device="cuda").manual_seed(2)
    for BM, BN, nw, path in ((16, 16, 4, "mma.sync"), (64, 64, 4, "wgmma")):
        for BK in (32, 64, 256):
            tot = good = 0
            for out in (0, 1):
                for trial in range(8):
                    a = torch.randint(-128, 128, (BM, BK), dtype=torch.int8, device="cuda", generator=g)
                    b = torch.randint(-128, 128, (BK, BN), dtype=torch.int8, device="cuda", generator=g)
                    s = (a.double() @ b.double()).to(torch.int64)
                    room = 2**31 - 1 - (a.double().abs() @ b.double().abs()).to(torch.int64)  # per output
                    # an int32 reading with |r| <= room, as uint32 bits: half >= 2^31 (negative readings)
                    u = torch.rand(BM, BN, device="cuda", generator=g, dtype=torch.float64)
                    r = ((u * 2 - 1) * room.double()).to(torch.int64)
                    r[0, :] = -room[0, :]           # the edge, negative reading: bits >= 2^31
                    r[1, :] = room[1, :]            # the edge, positive reading
                    bits = torch.where(r < 0, r + 2**32, r)
                    c0 = bits.to(torch.uint32)
                    c, neg, f = launch(a, b, c0, BM, BN, BK, out, tl.uint32, nw)
                    exp = r + s
                    good += int((c.to(torch.int64) == exp).sum())
                    tot += BM * BN
            lines.append(f"A2 {path} ({BM}, {BN}, {nw} warps) K = {BK}: {good}/{tot} results equal the accumulator's "
                         f"bits read as int32 plus the exact sum (out_dtype omitted and tl.uint32, 8 trials each)")
    return lines


def a3():
    a = torch.ones(16, 32, dtype=torch.int8, device="cuda")
    b = torch.ones(32, 16, dtype=torch.int8, device="cuda")
    c0 = torch.full((16, 16), 3_000_000_000, dtype=torch.int64, device="cuda").to(torch.uint32)
    c, neg, f = launch(a, b, c0, 16, 16, 32, 0, tl.int32, 4)
    return [f"A3 acc = 3000000000 (uint32), 32 products 1: stored {int(c[0, 0])} "
            f"(int32 reading {3_000_000_000 - 2**32 + 32}); c < 0: {int(neg[0, 0])}; c.to(tl.float32): {float(f[0, 0])}"]


def main():
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    for fn in (a1, a2, a3):
        for line in fn():
            print(line, flush=True)


if __name__ == "__main__":
    main()
