"""lr-12b: the float layer against the device on every pair or triple of a set of edge values.

The set, per format: both zeros, the smallest and largest subnormals, the smallest normal (and
its neighbours), 1 and its neighbours, 1/2, 3/2, 2, 3, values near the square root of the
smallest subnormal (products that underflow, tiny exact results that round to a signed zero),
values near the square root of maxVal (products that overflow, sums that round up to infinity),
maxVal and its predecessor, the infinities, quiet and signalling NaNs, all with both signs.
Every pair (binary operations) or triple (`tl.fma`) of the set goes through the device (the
kernel of evidence/float-layer/run.py) and through the library's definitions (the driver that
evidence/float-layer/build.sh builds), and the outputs are compared one by one, NaN up to payload.

Usage, from this folder: CUDA_VISIBLE_DEVICES=0 python -I edge.py <build folder>
"""
import importlib.util
import itertools
import os
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True  # leave no cache in the evidence folder
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "evidence", "float-layer", "run.py"))
spec = importlib.util.spec_from_file_location("flrun", RUN)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def edge_values(t):
    e, mb, sp = R.FMT[t]
    bias = (1 << (e - 1)) - 1
    emax = (1 << e) - 1
    s = 1 << (e + mb)
    one = bias << mb
    top = emax - 1 if sp == "ieee" else emax
    maxmag = ((top << mb) | ((1 << mb) - 1)) if sp == "ieee" else ((emax << mb) | ((1 << mb) - 2))
    # exponent fields near half of the subnormal exponent, and near half of the top exponent
    emin = 1 - bias
    lo_e = max(1, (emin - mb) // 2 + bias)
    hi_e = (top + bias) // 2 + 1
    mags = [0, 1, 2, 3, (1 << mb) - 1, 1 << mb, (1 << mb) + 1, (2 << mb) - 1,
            one, one + 1, one - 1, one | (1 << (mb - 1)), (bias - 1) << mb, (bias + 1) << mb,
            ((bias + 1) << mb) | (1 << (mb - 1)),
            lo_e << mb, (lo_e << mb) | 1, ((lo_e - 1) << mb) | ((1 << mb) - 1), (lo_e << mb) | (1 << (mb - 1)),
            hi_e << mb, (hi_e << mb) | ((1 << mb) - 1), ((hi_e - 1) << mb) | (1 << (mb - 1)),
            maxmag, maxmag - 1]
    if sp == "ieee":
        mags += [emax << mb, (emax << mb) | 1, (emax << mb) | (1 << (mb - 1))]
    vals = []
    for mg in mags:
        vals += [mg, mg | s]
    return sorted(set(vals))


def run(build, name, test, recs):
    x, y, z = (np.array(c, dtype=np.uint64) for c in zip(*recs))
    work = os.path.join(os.path.abspath(build), "work")
    os.makedirs(work, exist_ok=True)
    p = os.path.join(work, "edge.bin")
    np.stack([x, y, z], axis=1).astype("<u8").tofile(p)
    dev = test.device(0, len(x), files=(x, y, z)).tolist()
    out = subprocess.run([os.path.join(os.path.abspath(build), "driver"), test.spec, "dump-file", p, "0"],
                         capture_output=True, text=True, check=True).stdout
    lib = [int(b, 16) for a, b in (l.split() for l in out.splitlines())]
    os.remove(p)
    bad = [i for i in range(len(x)) if (lib[i] & R.M64) != (dev[i] & R.M64)]
    lines = [f"{name:16s} {test.spec:20s} {len(x):7d} inputs  " + ("agree" if not bad else f"DIFFER on {len(bad)}")]
    for i in bad[:8]:
        inp = ",".join(f"{v:x}" for v in recs[i][:test.arity])
        lines.append(f"    input {inp}: library {lib[i]:x}, device {dev[i] & R.M64:x}")
    return "\n".join(lines)


def main():
    build = sys.argv[1]
    T = R.tests()
    todo = []
    for f in ("f16", "bf16", "f32", "f64"):
        ev = edge_values(f)
        pairs = [(a, b, 0) for a, b in itertools.product(ev, ev)]
        triples = list(itertools.product(ev, ev, ev))
        for op in ("add", "sub", "mul", "div", "max", "min", "maxnan", "minnan", "lt", "le", "eq", "ne"):
            name = f"B.{op}.{f}"
            if name in T:
                todo.append((name, T[name], pairs))
        todo.append((f"B.fma.{f}", T[f"B.fma.{f}"], triples))
        for op in ("neg", "abs"):
            todo.append((f"C.{op}.{f}", T[f"C.{op}.{f}"], [(a, 0, 0) for a in ev]))
    todo.append(("B.sqrt.f32", T["B.sqrt.f32"], [(a, 0, 0) for a in edge_values("f32")]))
    for name in ("A.cvt.f32.f16.rn", "A.cvt.f32.f16.rz", "A.cvt.f32.bf16.rn", "A.cvt.f32.bf16.rz",
                 "A.cvt.f32.e4m3.sat1", "A.cvt.f32.e5m2.sat1", "A.cvt.f32.e5m2.rz.sat0", "A.cvt.f64.f32",
                 "A.cvt.f64.f16", "A.cvt.f64.bf16", "A.cvt.f32.f64"):
        t = T[name]
        todo.append((name, t, [(a, 0, 0) for a in edge_values(t.src)]))
    with open(os.path.join(HERE, "edge.out"), "w") as out:
        for name, t, recs in todo:
            line = run(build, name, t, recs)
            print(line, flush=True)
            out.write(line + "\n")


if __name__ == "__main__":
    main()
