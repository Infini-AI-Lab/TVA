"""TVA library evidence (library issue L64): the fp8 constants Triton 3.8.0 builds for a Python number,
against the library's `FloatV.litF8` (`lean/TVA/Lang/Scalar.lean`, used by `Operand.constIn?`), bit for
bit, and which of them are rejected on the host.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/fp8-constants/run.py
Output: evidence/fp8-constants/out.txt (README.md says what it shows).

A Python number built directly in an fp8 dtype (`tl.full(shape, c, d)`, a value of `tl.where` beside an
fp8 tensor) is `scalar_constant`'s `arith.truncf` of the `float32` constant, which MLIR folds only when
the conversion loses nothing (`primitives/triton-3.8.0-semantic.py:601-613`,
`primitives/llvm-ArithOps.cpp:1588-1611, 1788-1825`).

F1. Every value of `float8e5` and of `float8e4nv` (each pattern that is a number, both zeros included),
    written as a literal in the source: one kernel per format stores `tl.full([1], c, d)` for all of
    them, another `tl.where(i < 0, x, c)` with `x` an fp8 tensor.
F2. Values neither format holds exactly, each in a kernel of its own: the `float32` neighbours of a
    sample of fp8 values, the midpoints between neighbouring fp8 values, values beyond `maxVal`, values
    below the smallest subnormal, `0.1`, `1e6`; `tl.full` (all) and `tl.where` (a sample).
F3. The special values: `float('inf')`, `-float('inf')`, a value whose `float32` overflows (`1e39`,
    `-3.5e38`), `float('nan')`, `-float('nan')`, and NaN doubles with payloads (their `float32` with
    and without fraction bits below the top two), passed as `tl.constexpr` arguments, each launch in a
    kernel of its own and a process of its own with an empty cache (every NaN shares Triton's cache key).
F4. The run-time paths, for comparison: a stored Python number (`tl.store(p, c)`), `other=c` of a
    masked-off `tl.load`, and `tl.full([1], c, tl.float32).to(d)`: the `float32` constant converted on
    the device (`tl.cvt8`), against `FloatV.cvtSat .rn (litF32 c)` (the `CvtSat…` facts).

The library side is evaluated by Lean (`lake env lean` on a scratch file importing `TVA.Lang.Scalar`)
on the same doubles.
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(HERE))
LAKE = os.path.expanduser("~/.elan/bin/lake")
# (tag, Triton dtype, torch dtype name, Lean format, exponent bits, fraction bits, ieee)
FMTS = [("e5m2", "tl.float8e5", "float8_e5m2", "Math.Float.Format.e5m2", 5, 2, True),
        ("e4m3", "tl.float8e4nv", "float8_e4m3fn", "Math.Float.Format.e4m3", 4, 3, False)]


def bits64(c):
    return struct.unpack("<Q", struct.pack("<d", c))[0]


def double(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def f32(c):
    """The float32 nearest to the double c (as Python's struct does: round to nearest even)."""
    try:
        return struct.unpack("<f", struct.pack("<f", c))[0]
    except OverflowError:
        return math.copysign(math.inf, c)


def fp8_value(p, e, m, ieee):
    """The value of the fp8 pattern p, or None for a NaN or an infinity."""
    s = -1.0 if p >> (e + m) else 1.0
    ex, fr = (p >> m) & ((1 << e) - 1), p & ((1 << m) - 1)
    bias = (1 << (e - 1)) - 1
    if ieee and ex == (1 << e) - 1:
        return None
    if not ieee and ex == (1 << e) - 1 and fr == (1 << m) - 1:
        return None
    if ex == 0:
        return s * fr * 2.0 ** (1 - bias - m)
    return s * ((1 << m) + fr) * 2.0 ** (ex - bias - m)


def exact_values(fmt):
    _, _, _, _, e, m, ieee = fmt
    vals = []
    for p in range(256):
        v = fp8_value(p, e, m, ieee)
        if v is not None:
            vals.append(-0.0 if p == 1 << (e + m) else v)
    return vals


def inexact_values(fmt):
    """F2's values for one format: not held exactly (checked against the format's values)."""
    exact = sorted(set(v for v in exact_values(fmt)))
    held = set(exact)
    pos = [v for v in exact if v > 0]
    out = []
    for v in pos[::5] + [pos[-1]]:
        for d in (1, -1):
            n = struct.unpack("<f", struct.pack("<f", v))[0]
            b = struct.unpack("<I", struct.pack("<f", n))[0] + d
            out.append(struct.unpack("<f", struct.pack("<I", b))[0])
    for a, b in list(zip(pos, pos[1:]))[::7]:
        out.append((a + b) / 2)
    mx, mn = pos[-1], pos[0]
    out += [mx * 1.0625, mx * 2, mn / 2, mn / 3, 0.1, -0.1, 1e6, -1e6, 3e38]
    return [c for c in out if f32(c) not in held and not math.isinf(f32(c))]


SPECIALS = [float("inf"), -float("inf"), 1e39, -3.5e38, float("nan"), -float("nan")]
NAN_DOUBLES = [0x7ff8000000000000, 0xfff8000000000000, 0x7ffc000000000000, 0xfffc000000000000,
               0x7ff4000000000000, 0x7ff8000000000001, 0x7ff8000020000000, 0x7ff8000400000000,
               0xfffa000000000000, 0x7fffffffffffffff]
RUNTIME = [0.0, -0.0, 1.0, 0.1, 460.0, 1000.0, 1e6, float("inf"), -float("inf"), float("nan")]


# ---------------------------------------------------------------------------------------------
# device side (children, so that each has its own Triton cache)

def kernel_file(tmp, name, body, params):
    lines = ["import triton", "import triton.language as tl", "", "@triton.jit",
             f"def {name}({', '.join(params)}):"] + ["    " + l for l in body]
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def fmt_of(tag):
    return next(f for f in FMTS if f[0] == tag)


def run_store(tmp, name, tag, exprs, form, params=(), kwargs=None):
    """One kernel storing `form` (full / where / store / other / to) of each expression in fp8; the bytes,
    or None when the launch is rejected on the host."""
    import torch
    _, dt, tname, *_ = fmt_of(tag)
    body = ["i = tl.arange(0, 1)", "x = tl.load(x_ptr + i)"]
    for k, e in enumerate(exprs):
        if form == "full":
            body.append(f"tl.store(o_ptr + {k} + i, tl.full([1], {e}, {dt}))")
        elif form == "where":
            body.append(f"tl.store(o_ptr + {k} + i, tl.where(i < 0, x, {e}))")
        elif form == "store":
            body.append(f"tl.store(o_ptr + {k} + i, {e})")
        elif form == "other":
            body.append(f"tl.store(o_ptr + {k} + i, tl.load(x_ptr + i, mask=i < 0, other={e}))")
        else:
            body.append(f"tl.store(o_ptr + {k} + i, tl.full([1], {e}, tl.float32).to({dt}))")
    kern = kernel_file(tmp, name, body, ["x_ptr", "o_ptr"] + list(params))
    tdt = getattr(torch, tname)
    x = torch.zeros(1, dtype=torch.uint8, device="cuda").view(tdt)
    o = torch.full((len(exprs),), 0x55, dtype=torch.uint8, device="cuda").view(tdt)
    try:
        kern[(1,)](x, o, **(kwargs or {}))
        torch.cuda.synchronize()
    except Exception as ex:   # rejected on the host
        return {"rejected": type(ex).__name__}
    return {"bytes": o.view(torch.uint8).tolist()}


def child(mode, arg):
    tmp = tempfile.mkdtemp(prefix="tva-fp8c-")
    a = json.loads(arg)
    out = []
    if mode == "batch":          # F1, F4: many expressions, one kernel
        out = run_store(tmp, a["name"], a["tag"], a["exprs"], a["form"])
    elif mode == "single":       # F2: one kernel per expression
        for k, e in enumerate(a["exprs"]):
            out.append(run_store(tmp, f"{a['name']}_{k}", a["tag"], [e], a["form"]))
    else:                        # F3: one constexpr value per kernel, in this process
        out = run_store(tmp, a["name"], a["tag"], ["C"], a["form"], params=["C: tl.constexpr"],
                        kwargs={"C": double(a["bits"])})
    print("RESULT " + json.dumps(out))


def run_child(mode, arg):
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="tva-fp8c-cache-")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", mode, json.dumps(arg)],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise SystemExit(f"child {mode} failed: {r.stderr[-2000:]}")


# ---------------------------------------------------------------------------------------------
# library side

def lean_values(doubles):
    """For each double and format: `FloatV.litF8` (a pattern, or -1 for `none`) and the run-time
    conversion `FloatV.cvtSat .rn (FloatV.litF32 c)` (a pattern)."""
    src = ["import TVA.Lang.Scalar", "open TVA", "", "def cs : List Nat := ["]
    src.append(",\n".join(f"  {b}" for b in doubles) + "]")
    for tag, _, _, lf, *_ in FMTS:
        src.append(f"#eval IO.println (String.intercalate \" \" (cs.map fun n => "
                   f"match (FloatV.litF8 (g := {lf}) (.ofPat n)) with | some y => toString y.pat | none => \"-1\"))")
        src.append(f"#eval IO.println (String.intercalate \" \" (cs.map fun n => "
                   f"toString (FloatV.cvtSat (g := {lf}) .rn (FloatV.litF32 (.ofPat n))).pat))")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "Fp8Check.lean")
        with open(path, "w") as fh:
            fh.write("\n".join(src) + "\n")
        r = subprocess.run([LAKE, "env", "lean", path], cwd=os.path.join(TVA, "lean"), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("lean failed: " + (r.stderr[-800:] + r.stdout[-800:]).replace(tmp, "<tmp>"))
    rows = [list(map(int, l.split())) for l in r.stdout.strip().splitlines()]
    return {"e5m2": (rows[0], rows[1]), "e4m3": (rows[2], rows[3])}


def show(b):
    return "rejected" if b is None or b < 0 else f"{b:02x}"


def compare(label, tag, doubles, device, lib):
    """Device results (bytes, or None for a rejection) against litF8 (-1 for none); the differences."""
    diff = []
    for c, d, w in zip(doubles, device, lib):
        want = None if w < 0 else w
        if d != want:
            diff.append(f"{double(c)!r} (double {c:016x}): device {show(d)}, library {show(w)}")
    return diff


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    total = 0
    for fmt in FMTS:
        tag = fmt[0]
        # F1
        ex = exact_values(fmt)
        dbl = [bits64(v) for v in ex]
        lib = lean_values(dbl)[tag][0]
        for form in ("full", "where"):
            r = run_child("batch", {"name": f"f1_{tag}_{form}", "tag": tag, "form": form, "exprs": [repr(v) for v in ex]})
            dev = r.get("bytes") or [None] * len(ex)
            diff = compare("F1", tag, dbl, dev, lib)
            total += len(diff)
            print(f"F1 {tag} {form}: {len(ex)} values of the format, " +
                  ("rejected: " + r["rejected"] if "rejected" in r else f"{len(diff)} differ from litF8") +
                  "".join(f"\n    {d}" for d in diff[:8]))
        print(f"   e.g. " + ", ".join(f"{v!r} -> {b:02x}" for v, b in list(zip(ex, lib))[::37]))
        # F2
        inx = inexact_values(fmt)
        dbl = [bits64(v) for v in inx]
        lib = lean_values(dbl)[tag][0]
        for form, sub in (("full", inx), ("where", inx[::4])):
            r = run_child("single", {"name": f"f2_{tag}_{form}", "tag": tag, "form": form, "exprs": [repr(v) for v in sub]})
            dev = [(x["bytes"][0] if "bytes" in x else None) for x in r]
            lsub = lib if form == "full" else lib[::4]
            diff = compare("F2", tag, [bits64(v) for v in sub], dev, lsub)
            total += len(diff)
            rej = sum(1 for x in dev if x is None)
            print(f"F2 {tag} {form}: {len(sub)} values the format does not hold, {rej} rejected on the host; "
                  f"{len(diff)} differ from litF8" + "".join(f"\n    {d}" for d in diff[:8]))
        print(f"   e.g. " + ", ".join(repr(v) for v in inx[::9]))
        # F3
        sp = [bits64(v) for v in SPECIALS] + NAN_DOUBLES
        lib = lean_values(sp)[tag][0]
        dev = []
        for k, b in enumerate(sp):
            r = run_child("constexpr", {"name": f"f3_{tag}_{k}", "tag": tag, "form": "full", "bits": b})
            dev.append(r["bytes"][0] if "bytes" in r else None)
        diff = compare("F3", tag, sp, dev, lib)
        total += len(diff)
        print(f"F3 {tag} full, special values as tl.constexpr arguments (float32 constant in brackets): "
              f"{len(diff)} differ from litF8")
        for b, d, w in zip(sp, dev, lib):
            c = double(b)
            fb = struct.unpack("<I", struct.pack("<f", f32(c)))[0] if not math.isnan(c) else None
            print(f"    {b:016x} ({c!r}{'' if fb is None else f', float32 {fb:08x}'}): device {show(d)}, "
                  f"library {show(w)}")
        # F4
        rt = [bits64(v) for v in RUNTIME]
        sat = lean_values(rt)[tag][1]
        for form in ("store", "other", "to"):
            r = run_child("batch", {"name": f"f4_{tag}_{form}", "tag": tag, "form": form,
                                    "exprs": [repr(v) if not math.isnan(v) and not math.isinf(v) else
                                              ("float('nan')" if math.isnan(v) else
                                               ("float('inf')" if v > 0 else "-float('inf')")) for v in RUNTIME]})
            if "rejected" in r:
                print(f"F4 {tag} {form}: rejected ({r['rejected']})")
                continue
            dev = r["bytes"]
            e, m = fmt[4], fmt[5]

            def nan8(p):
                return fp8_value(p, e, m, fmt[6]) is None and (fmt[6] is False or p & ((1 << m) - 1) != 0)
            bad = [f"{v!r}: device {d:02x}, cvtSat {s:02x}" for v, d, s in zip(RUNTIME, dev, sat)
                   if not (d == s or (nan8(d) and nan8(s)))]
            total += len(bad)
            print(f"F4 {tag} {form}: " + ", ".join(f"{v!r} -> {d:02x}" for v, d in zip(RUNTIME, dev)) +
                  f"; {len(bad)} differ from FloatV.cvtSat .rn (litF32 c), up to NaN payload" +
                  "".join(f"\n    {x}" for x in bad))
    print(f"differences in all: {total}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
