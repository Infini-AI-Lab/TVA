"""lr-20f, round 20, L64: fp8 constants beyond evidence/fp8-constants' cases, against FloatV.litF8
(evaluated by Lean). Cases: doubles whose float32 underflows to a signed zero, Python ints and bools,
literal infinities and NaNs written in the source, and int/float values in tl.where beside an fp8 tensor.
Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>"""
import os, sys, struct, subprocess, tempfile, importlib.util, json, math

TVA = os.getcwd()
LAKE = os.path.expanduser("~/.elan/bin/lake")
# (source expression for c, the Python value it evaluates to)
CASES = ["1e-50", "-1e-50", "5e-324", "-5e-324", "448", "-3", "2**70", "2**200", "True", "57344",
         "float('inf')", "-float('inf')", "float('nan')", "-float('nan')", "0.5", "1.5e-5",
         "2.0**-16", "2.0**-17", "2.0**-9", "2.0**-10", "-0.0", "0"]
FMTS = [("e5m2", "tl.float8e5", "float8_e5m2", "Math.Float.Format.e5m2"),
        ("e4m3", "tl.float8e4nv", "float8_e4m3fn", "Math.Float.Format.e4m3")]


def bits64(c):
    return struct.unpack("<Q", struct.pack("<d", float(c)))[0]


def lean_litF8(dbls):
    src = ["import TVA.Lang.Scalar", "open TVA", "def cs : List Nat := [" + ", ".join(map(str, dbls)) + "]"]
    for _, _, _, lf in FMTS:
        src.append(f"#eval IO.println (String.intercalate \" \" (cs.map fun n => match (FloatV.litF8 (g := {lf}) (.ofPat n)) with | some y => toString y.pat | none => \"-1\"))")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "L.lean")
        open(p, "w").write("\n".join(src) + "\n")
        r = subprocess.run([LAKE, "env", "lean", p], cwd=os.path.join(TVA, "lean"), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:] + r.stdout[-500:]
    rows = [list(map(int, l.split())) for l in r.stdout.strip().splitlines()]
    return {"e5m2": rows[0], "e4m3": rows[1]}


def child(tag, tl_dt, tdt, form, expr):
    import torch
    tmp = tempfile.mkdtemp()
    name = f"k_{form}"
    body = (f"    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], {expr}, {tl_dt}))" if form == "full" else
            f"    i = tl.arange(0, 1)\n    x = tl.load(x_ptr + i)\n    tl.store(o_ptr + i, tl.where(i < 0, x, {expr}))")
    src = f"import triton\nimport triton.language as tl\n\n@triton.jit\ndef {name}(x_ptr, o_ptr):\n{body}\n"
    p = os.path.join(tmp, name + ".py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    x = torch.zeros(1, dtype=torch.uint8, device="cuda").view(getattr(torch, tdt))
    o = torch.full((1,), 0x55, dtype=torch.uint8, device="cuda").view(getattr(torch, tdt))
    try:
        getattr(mod, name)[(1,)](x, o); torch.cuda.synchronize()
        print("RESULT", o.view(torch.uint8).item())
    except Exception as ex:
        print("RESULT", -1, type(ex).__name__)


def main():
    import triton, torch
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, Python {sys.version.split()[0]}")
    vals = [eval(e) for e in CASES]
    lib = lean_litF8([bits64(v) for v in vals])
    diffs = 0
    for tag, tl_dt, tdt, _ in FMTS:
        for form in ("full", "where"):
            out = []
            for k, e in enumerate(CASES):
                env = dict(os.environ); env["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
                r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", tag, tl_dt, tdt, form, e],
                                   capture_output=True, text=True, env=env)
                line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
                dev = int(line[0].split()[1]) if line else None
                want = lib[tag][k]
                ok = dev == want
                diffs += not ok
                out.append(f"{e}: device {'rejected' if dev == -1 else (format(dev, '02x') if dev is not None else 'CRASH')}, "
                           f"litF8 {'none' if want < 0 else format(want, '02x')}{'' if ok else '  <-- DIFFERS'}")
            print(f"{tag} {form}:\n  " + "\n  ".join(out))
    print("differences:", diffs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:7])
    else:
        main()
