"""TVA library evidence, L40 (sm_89): `/` on float64 is IEEE division (`DivRn P .binary64`).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/l40/num-div-f64/run.py
Output (stdout): kept as evidence/l40/num-div-f64/run.out.

The H200's script (`evidence/num-div-f64/run.py`) checks the device against the library's own
`FloatV.divRn`, compiled by a driver that v3 no longer builds. Here the reference is the CPU's float64
division (NumPy, IEEE round to nearest even, no flush of subnormals), which `FloatV.divRn` equals: the
H200's run agreed with both on about `4.5·10^8` pairs. The inputs are the H200 script's families
(imported from it, the same seeds and sizes), and `x / c` for its constant divisors; the device runs the
same kernel (`x / y`, `tl.fdiv(x, y)`, `x / c`). A result is compared bit for bit, a NaN up to payload.
"""
import os
import re
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "num-div-f64"))
import run as H  # noqa: E402  (the H200 script: kernel, families, constants)

NAN = np.uint64(H.NAN)
CH = 1 << 22


def canon(a):
    """NaN patterns replaced by the canonical NaN magnitude (as the H200 script does)."""
    e = (a >> np.uint64(52)) & np.uint64(0x7FF)
    f = a & H.FRAC
    return np.where((e == np.uint64(0x7FF)) & (f != np.uint64(0)), NAN, a)


def check(x, y, mode, c=0.0):
    dev = H.device(x, y, mode, c).cpu().numpy().view(np.uint64)
    with np.errstate(all="ignore"):
        ref = (x.view(np.float64) / (np.float64(c) if mode == "const" else y.view(np.float64))).view(np.uint64)
    bad = np.nonzero(canon(dev) != canon(ref))[0]
    return len(x), bad, dev, ref


def main():
    import triton
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| family | mode | pairs | differ from IEEE division (up to NaN payload) | first difference (x, y, device, IEEE) |")
    print("|---|---|---|---|---|")
    total = diff = 0
    runs = [(n, g, sz, m, None) for (n, g, sz, m) in H.FAMILIES] + \
           [(f"x / {c!r}", H.fam_uniform, 1 << 22, "const", c) for c in H.CONSTS]
    for name, gen, n, mode, c in runs:
        t0 = time.time()
        seed = int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") + \
            (0 if c is None else int(np.float64(c).view(np.uint64)))
        rng = np.random.default_rng(seed)
        cnt, nbad, ex = 0, 0, None
        if n is None:
            parts = [gen(rng, 0)]
        else:
            parts = (gen(rng, min(CH, n - lo)) for lo in range(0, n, CH))
        for x, y in parts:
            k, bad, dev, ref = check(x, y, mode, 0.0 if c is None else c)
            cnt += k; nbad += len(bad)
            if ex is None and len(bad):
                i = bad[0]
                ex = (float(x.view(np.float64)[i]), float(y.view(np.float64)[i]) if mode != "const" else c,
                      hex(int(dev[i])), hex(int(ref[i])))
        total += cnt; diff += nbad
        print(f"| {name} | {mode} | {cnt} | {nbad} | {ex} |", flush=True)
        print(f"{name}: {time.time() - t0:.0f} s", file=sys.stderr)
    print(f"\nTotal pairs: {total}; differing from IEEE division: {diff}")
    print("\nPTX division instructions:")
    for (mode, c), h in sorted(H.COMPILED.items(), key=lambda t: (t[0][0], str(t[0][1]))):
        ops = sorted(set(re.findall(r"\b(?:div|rcp|fma)\.[a-z0-9.]+", h.asm["ptx"])))
        print(f"- {mode}{'' if mode != 'const' else ' ' + repr(c)}: {', '.join(ops)}")


if __name__ == "__main__":
    main()
