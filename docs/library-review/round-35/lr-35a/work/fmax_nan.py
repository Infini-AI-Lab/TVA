"""lr-35a (library round 35, L164): float32/float64 atomic_max/atomic_min, every pair of special and
ordinary operands, the result used. The semantics (maxUpd/fBound, one read-modify-write per enabled
lane): the returned value is the old bit pattern, the new value is the old or the operand (bit
pattern), and with neither NaN not below (above) either. Lists the atom instructions of the PTX.
Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import re, struct, itertools
import torch, triton, triton.language as tl


@triton.jit
def k(P, V, R, MAX: tl.constexpr, N: tl.constexpr):
    o = tl.arange(0, N)
    v = tl.load(V + o)
    if MAX:
        r = tl.atomic_max(P + o, v)
    else:
        r = tl.atomic_min(P + o, v)
    tl.store(R + o, r)


def vals32():
    b = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
         0x7f800001, 0xff800001, 0x7fffffff, 0x00000001, 0x80000001, 0x7f7fffff, 0x40400000, 0xc0400000]
    return b


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    for dt, it, bits in [(torch.float32, torch.int32, vals32())]:
        pairs = list(itertools.product(bits, bits))
        N = 1 << (len(pairs) - 1).bit_length()
        pairs += [(bits[2], bits[2])] * (N - len(pairs))
        def t(xs):
            return torch.tensor([x - (1 << 32) if x >= 1 << 31 else x for x in xs], dtype=it, device="cuda").view(dt)
        for MAX in (True, False):
            P = t([a for a, _ in pairs]); V = t([b for _, b in pairs]); R = torch.empty_like(P)
            h = k[(1,)](P, V, R, MAX=MAX, N=N)
            torch.cuda.synchronize()
            pb = (P.view(it).cpu().numpy().astype("int64") & 0xffffffff).tolist()
            rb = (R.view(it).cpu().numpy().astype("int64") & 0xffffffff).tolist()
            bad_ret = bad_new = 0
            ex = []
            for (o, v), n, r in zip(pairs, pb, rb):
                if r != o:
                    bad_ret += 1
                    ex.append(("ret", hex(o), hex(v), hex(r)))
                fo, fv, fn = (struct.unpack("<f", struct.pack("<I", x))[0] for x in (o, v, n))
                ok = n in (o, v)
                if ok and fo == fo and fv == fv:
                    ok = (fn >= fo and fn >= fv) if MAX else (fn <= fo and fn <= fv)
                if not ok:
                    bad_new += 1
                    ex.append(("new", hex(o), hex(v), hex(n)))
            atoms = sorted(set(re.findall(r"(?:@%p\d+ )?atom\.[a-z0-9.]+", h.asm["ptx"])))
            print(f"{'atomic_max' if MAX else 'atomic_min'} f32: {len(pairs)} lanes; returned value not the old bits: "
                  f"{bad_ret}; new value outside fBound: {bad_new}; examples {ex[:4]}; PTX {atoms}")


if __name__ == "__main__":
    main()
