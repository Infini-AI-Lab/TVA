"""TVA library evidence (library issue L98): the bit patterns Triton 3.8.0 builds for a Python NaN
constant, per dtype, against the library's `FloatV.litF64`, `litF32`, `litF16` and `litBF16`
(`lean/TVA/Lang/Scalar.lean`), compared bit for bit, both signs.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/literals/nan.py
Output: evidence/literals/nan.out.txt (README.md, "NaN constants").

N1. NaNs written in the kernel's source: `float('nan')`, `-float('nan')`, `float('-nan')`,
    `float('inf') - float('inf')` and other Python expressions the front end evaluates, and the
    infinities. For each and each dtype `d` (`bfloat16`, `float16`, `float32`, `float64`), one
    kernel stores `tl.full([1], E, d)` and `tl.where(<false>, x, E)` with `x` a `d` tensor (the
    constant built in `d`), compared bit for bit with the library's function of the double that
    Python computes for `E`; and `tl.load(p, mask=False, other=E)` (the `float32` constant, then
    converted to `d`), compared up to NaN payload (a conversion, `tl.convert`) and printed.
N2. NaN doubles with payloads, quiet and signalling, both signs (every single payload bit, all
    ones, random payloads), passed as `tl.constexpr` arguments. Triton's cache keys write every NaN
    as `nan`, so each batch of 16 runs in a process of its own with an empty `TRITON_CACHE_DIR`:
    one kernel stores `tl.full([1], C_k, d)` for its 16 constants and the four dtypes.
N3. The cache: in one process with one empty cache folder, the N2 kernel launched with
    `C = float('nan')` and then `C = -float('nan')`; then a second process on the same folder
    launched with `-float('nan')` first; and the same with an empty folder.
N4. Runtime scalar arguments holding a NaN (not constants; for `docs/SEMANTICS.md`, "Scalar
    arguments"): `x` without annotation and `x: tl.float32`, `tl.float64`, `tl.float16`,
    `tl.bfloat16`, stored as passed, against `FloatV.hostF32` (C's `(float)`) and the double.

The library side is evaluated by Lean (`lake env lean` on a scratch file importing
`TVA.Lang.Scalar`), on the same doubles.
"""
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(HERE))
LAKE = os.path.expanduser("~/.elan/bin/lake")
DTYPES = [("bf16", "tl.bfloat16", 16), ("f16", "tl.float16", 16), ("f32", "tl.float32", 32),
          ("f64", "tl.float64", 64)]
EXPRS = ["float('nan')", "-float('nan')", "float('-nan')", "float('NaN')", "-(-float('nan'))",
         "float('inf') - float('inf')", "-(float('inf') - float('inf'))", "float('inf') * 0.0",
         "0.0 * float('-inf')", "float('nan') + 1.0", "-float('nan') * 2.0", "float('inf') / float('inf')",
         "float('inf')", "-float('inf')"]
BATCH = 16


def bits64(c):
    return struct.unpack("<Q", struct.pack("<d", c))[0]


def double(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def is_nan_tag(tag, v):
    e, m = {"bf16": (8, 7), "f16": (5, 10), "f32": (8, 23), "f64": (11, 52)}[tag]
    return (v >> m) & ((1 << e) - 1) == (1 << e) - 1 and v & ((1 << m) - 1) != 0


def payload_doubles():
    """N2's NaN doubles: quiet (`0x7ff8…`) and signalling (`0x7ff0…`, payload nonzero) with each
    single payload bit, all-ones payloads, and random payloads, with both signs."""
    rng = random.Random(20261028)
    out = []
    for sign in (0, 1 << 63):
        for b in range(51):
            out.append(sign | 0x7ff8000000000000 | (1 << b))
            out.append(sign | 0x7ff0000000000000 | (1 << b))
        out += [sign | 0x7ff8000000000000, sign | 0x7fffffffffffffff, sign | 0x7ff7ffffffffffff]
        for _ in range(24):
            out.append(sign | 0x7ff8000000000000 | rng.getrandbits(51))
            out.append(sign | 0x7ff0000000000000 | (rng.getrandbits(51) or 1))
    seen, uniq = set(), []
    for b in out:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


# ---------------------------------------------------------------------------------------------
# device side (in a child process, imported only there, so each child has its own cache)

def kernel_file(tmp, name, body_lines, params):
    lines = ["import triton", "import triton.language as tl", "", "@triton.jit",
             f"def {name}({', '.join(params)}):"] + ["    " + l for l in body_lines]
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def stored(t, w):
    import torch
    iv = {16: torch.int16, 32: torch.int32, 64: torch.int64}[w]
    return [v & ((1 << w) - 1) for v in t.view(iv).tolist()]


def child_source(tmp):
    """N1: one kernel per dtype, three stores per expression."""
    import torch
    import triton
    res = {}
    tdt = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32, "f64": torch.float64}
    for tag, dt, w in DTYPES:
        body = ["i = tl.arange(0, 1)", "x = tl.load(x_ptr + i)"]
        for k, e in enumerate(EXPRS):
            body.append(f"tl.store(full_ptr + {k} + i, tl.full([1], {e}, {dt}))")
            body.append(f"tl.store(where_ptr + {k} + i, tl.where(i < 0, x, {e}))")
            body.append(f"tl.store(other_ptr + {k} + i, tl.load(p_ptr + i, mask=i < 0, other={e}))")
        kern = kernel_file(tmp, f"nan_src_{tag}", body, ["x_ptr", "p_ptr", "full_ptr", "where_ptr", "other_ptr"])
        x = torch.ones(1, dtype=tdt[tag], device="cuda")
        p = torch.zeros(1, dtype=tdt[tag], device="cuda")
        bufs = [torch.zeros(len(EXPRS), dtype=tdt[tag], device="cuda") for _ in range(3)]
        kern[(1,)](x, p, *bufs)
        torch.cuda.synchronize()
        res[tag] = [stored(b, w) for b in bufs]
    return res


def child_const(tmp, doubles):
    """N2: one kernel, `tl.full([1], C_k, d)` for each constexpr `C_k` and dtype `d`."""
    import torch
    body = ["i = tl.arange(0, 1)"]
    for j, (tag, dt, w) in enumerate(DTYPES):
        for k in range(len(doubles)):
            body.append(f"tl.store(o{j}_ptr + {k} + i, tl.full([1], C{k}, {dt}))")
    params = [f"o{j}_ptr" for j in range(len(DTYPES))] + [f"C{k}: tl.constexpr" for k in range(len(doubles))]
    kern = kernel_file(tmp, "nan_const", body, params)
    tdt = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32, "f64": torch.float64}
    bufs = [torch.zeros(len(doubles), dtype=tdt[tag], device="cuda") for tag, _, _ in DTYPES]
    kern[(1,)](*bufs, **{f"C{k}": double(b) for k, b in enumerate(doubles)})
    torch.cuda.synchronize()
    return {tag: stored(bufs[j], w) for j, (tag, _, w) in enumerate(DTYPES)}


def child_cache(tmp, order):
    """N3: the one-constant kernel launched once per double of `order`, in this process."""
    import torch
    kern = kernel_file(tmp, "nan_cache", ["tl.store(o_ptr + tl.arange(0, 1), tl.full([1], C, tl.float32))"],
                       ["o_ptr", "C: tl.constexpr"])
    out = []
    for b in order:
        o = torch.zeros(1, dtype=torch.float32, device="cuda")
        kern[(1,)](o, C=double(b))
        torch.cuda.synchronize()
        out.append(stored(o, 32)[0])
    return out


def child_args(tmp, doubles):
    """N4: runtime scalar arguments, stored as passed."""
    import torch
    out = {}
    for tag, ann, td, w in (("plain", "", torch.float32, 32), ("f32", ": tl.float32", torch.float32, 32),
                            ("f64", ": tl.float64", torch.float64, 64), ("f16", ": tl.float16", torch.float16, 16),
                            ("bf16", ": tl.bfloat16", torch.bfloat16, 16)):
        kern = kernel_file(tmp, f"nan_arg_{tag}", ["tl.store(o_ptr, x)"], ["o_ptr", f"x{ann}"])
        vals = []
        for b in doubles:
            o = torch.zeros(1, dtype=td, device="cuda")
            kern[(1,)](o, double(b))
            torch.cuda.synchronize()
            vals.append(stored(o, w)[0])
        out[tag] = vals
    return out


def child(mode, arg):
    tmp = tempfile.mkdtemp(prefix="tva-nan-")
    if mode == "source":
        r = child_source(tmp)
    elif mode == "const":
        r = child_const(tmp, json.loads(arg))
    elif mode == "cache":
        r = child_cache(tmp, json.loads(arg))
    else:
        r = child_args(tmp, json.loads(arg))
    print("RESULT " + json.dumps(r))


def run_child(mode, arg="", cache=None):
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = cache or tempfile.mkdtemp(prefix="tva-nan-cache-")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", mode, arg],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise SystemExit(f"child {mode} failed: {r.stderr[-2000:]}")


# ---------------------------------------------------------------------------------------------
# library side

def lean_values(doubles):
    """For each double: litF64, litF32, litF16, litBF16 and hostF32 patterns, by Lean."""
    src = ["import TVA.Lang.Scalar", "open TVA", "", "def cs : List Nat := ["]
    src.append(",\n".join(f"  {b}" for b in doubles) + "]")
    src.append("def row {f : Math.Float.Format} (g : Nat → FloatV f) : IO Unit :=")
    src.append("  IO.println (String.intercalate \" \" (cs.map fun n => toString (g n).pat))")
    for fn in ("litF64", "litF32", "litF16", "litBF16", "hostF32"):
        src.append(f"#eval row (fun n => FloatV.{fn} (.ofPat n))")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "NanCheck.lean")
        with open(path, "w") as fh:
            fh.write("\n".join(src) + "\n")
        r = subprocess.run([LAKE, "env", "lean", path], cwd=os.path.join(TVA, "lean"), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("lean failed: " + (r.stderr[-800:] + r.stdout[-800:]).replace(tmp, "<tmp>"))
    rows = [list(map(int, l.split())) for l in r.stdout.strip().splitlines()]
    return dict(zip(("f64", "f32", "f16", "bf16", "host32"), rows))


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    # N1
    dbl = [bits64(eval(e)) for e in EXPRS]   # what Python computes for each expression
    lib = lean_values(dbl)
    dev = run_child("source")
    print("N1. NaNs and infinities written in the source (tl.full and tl.where: bit for bit; "
          "other=: up to NaN payload)")
    bad = 0
    for k, e in enumerate(EXPRS):
        cells = []
        for tag, _, w in DTYPES:
            full, where, other = (dev[tag][q][k] for q in range(3))
            want = lib[tag][k]
            ok = full == want and where == want
            own = lib["f32"][k]   # the float32 constant, then converted (up to NaN payload)
            ok_other = is_nan_tag(tag, other) if is_nan_tag("f32", own) else other == want
            bad += (not ok) + (not ok_other)
            cells.append(f"{tag} {full:x}{'' if ok else ' (library ' + format(want, 'x') + ', where ' + format(where, 'x') + ')'}"
                         f" other {other:x}{'' if ok_other else ' DIFFERS'}")
        print(f"  {e:34s} double {dbl[k]:016x}: " + "; ".join(cells))
    print(f"  N1 differences: {bad}")
    # N2
    pds = payload_doubles()
    lib2 = lean_values(pds)
    dev2 = {tag: [] for tag, _, _ in DTYPES}
    for k in range(0, len(pds), BATCH):
        r = run_child("const", json.dumps(pds[k:k + BATCH]))
        for tag, _, _ in DTYPES:
            dev2[tag] += r[tag]
    print(f"N2. {len(pds)} NaN doubles (quiet and signalling, both signs) as tl.constexpr arguments, "
          f"batches of {BATCH}, each in a process with an empty cache; tl.full, bit for bit")
    for tag, _, _ in DTYPES:
        diff = [i for i in range(len(pds)) if dev2[tag][i] != lib2[tag][i]]
        print(f"  {tag}: {len(diff)} of {len(pds)} differ from FloatV.lit{tag.upper() if tag != 'bf16' else 'BF16'}"
              + "".join(f"; {pds[i]:016x}: device {dev2[tag][i]:x}, library {lib2[tag][i]:x}" for i in diff[:6]))
    for b in (0x7ff8000000000000, 0xfff8000000000000, 0x7ff4000000000000, 0x7ff0000000000001,
              0xfff0000020000000, 0x7fffffffffffffff):
        i = pds.index(b)
        print(f"  e.g. {b:016x}: " + ", ".join(f"{tag} {dev2[tag][i]:x}" for tag, _, _ in DTYPES))
    # N3
    pn, nn = 0x7ff8000000000000, 0xfff8000000000000
    shared = tempfile.mkdtemp(prefix="tva-nan-shared-")
    a = run_child("cache", json.dumps([pn, nn]), cache=shared)
    b = run_child("cache", json.dumps([nn]), cache=shared)
    c = run_child("cache", json.dumps([nn, pn]))
    print("N3. the cache (tl.full([1], C, tl.float32), C a tl.constexpr)")
    print(f"  one process, empty cache: C = float('nan') stores {a[0]:x}, then C = -float('nan') stores {a[1]:x}")
    print(f"  a second process, the same cache folder: C = -float('nan') stores {b[0]:x}")
    print(f"  one process, empty cache: C = -float('nan') stores {c[0]:x}, then C = float('nan') stores {c[1]:x}")
    # N4
    ads = [pn, nn, 0x7ff4000000000000, 0xfff4000000000000, 0x7ff0000000000001, 0x7ff8000000012345,
           0xfff80000a0000000, 0x7fffffffffffffff]
    lib4 = lean_values(ads)
    r4 = run_child("args", json.dumps(ads))
    print("N4. runtime scalar arguments holding a NaN, stored as passed")
    for tag in ("plain", "f32", "f64", "f16", "bf16"):
        if tag in ("plain", "f32"):
            want, what = lib4["host32"], "FloatV.hostF32"
        elif tag == "f64":
            want, what = ads, "the double"
        elif tag == "bf16":
            want, what = [v >> 16 for v in lib4["host32"]], "the top 16 bits of FloatV.hostF32"
        else:
            want, what = None, None
        vals = r4[tag]
        line = f"  x{'' if tag == 'plain' else ': tl.' + {'f32': 'float32', 'f64': 'float64', 'f16': 'float16', 'bf16': 'bfloat16'}[tag]}: "
        line += ", ".join(f"{d:016x} -> {v:x}" for d, v in zip(ads, vals))
        if want is not None:
            line += f"; differs from {what}: {sum(1 for v, x in zip(vals, want) if v != x)}"
        print(line)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        main()
