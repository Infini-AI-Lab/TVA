"""TVA library evidence (library issues L98, L103): compile-time values that Triton 3.8.0's caches and its
helper names do not tell apart.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/compile-keys/run.py
Output: evidence/compile-keys/out.txt (README.md says what it shows).

Every launch sequence runs in a process of its own; "fresh" means an empty `TRITON_CACHE_DIR` of its
own, "shared" one cache folder that two processes use one after the other. Each value is also launched
alone, fresh, as the control: what the launch's own compile-time values compile to.

K1. The in-process cache equates values Python's `==` equates (`compute_cache_key`,
    `primitives/triton-3.8.0-jit.py:592-614`): one process launches a kernel with `C` one value and
    then another (`1` and `1.0`, `1` and `True`, `0` and `0.0`, `0` and `False`, both orders). The
    kernel stores `x * C` for the `int32` `x = 2^24 + 1` (`16777217` when `C` is an `int` or a `bool`,
    `16777216` through `float32` when it is a `float`) and whether `isinstance(C, float)` and
    `C is True or C is False` hold.
K2. The in-process cache keys a kernel by the text `str` writes, where every NaN is `nan`: `C` a NaN,
    then another NaN; the kernel stores `tl.full([1], C, tl.float32)`.
K3. The disk cache keys a kernel by each value's `str`, joined with `-` (`ASTSource.hash`,
    `primitives/triton-3.8.0-compiler.py:71-76`): a first process launches with `MODE` one value, a
    second process on the same cache folder with another (`1` and `"1"`, `None` and `"None"`, `2.5` and
    `"2.5"`, `True` and `"True"`, `float('nan')` and `"nan"`, both orders); the kernel stores `5` for
    `None`, `20` for a string and `10` otherwise. Also in one process (`1` then `"1"`): the in-process
    keys differ (`str` writes the string with its quotes), and the second launch, compiling, finds the
    first's code in the process's own cache folder.
K4. The disk key's `-`: `(A, B) = ("a-b", "c")` and then `("a", "b-c")` in two processes on one cache
    folder; the kernel stores `10·len(A) + len(B)`.
K5. Helper copies are named by `mangle_fn` (`primitives/triton-3.8.0-code-generator.py:32-41`): the
    `repr` of each compile-time argument, with `'` written `_sq_` and `[`, `]` written `_`
    (`primitives/triton-3.8.0-core.py:196-201`). One kernel calls a helper twice: with `float('nan')` and
    `-float('nan')` (the helper stores `tl.full([1], C, tl.float32)`), both orders; with `"x[y"` and
    `"x_y"` (the helper stores `1` when its argument is `"x_y"`, else `2`), both orders; and the controls
    `float('inf')`/`-float('inf')`, `1`/`1.0`/`True` (one helper called three times), `"a"`/`"x_y"`.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

SRC = '''import triton
import triton.language as tl


@triton.jit
def k1(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * C)
    if isinstance(C, float):
        tl.store(o_ptr + 1 + i, tl.full([1], 1, tl.int32))
    if C is True or C is False:
        tl.store(o_ptr + 2 + i, tl.full([1], 1, tl.int32))


@triton.jit
def k2(o_ptr, C: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], C, tl.float32))


@triton.jit
def k3(o_ptr, MODE: tl.constexpr):
    i = tl.arange(0, 1)
    if MODE is None:
        tl.store(o_ptr + i, tl.full([1], 5, tl.int32))
    elif MODE == "1" or MODE == "None" or MODE == "2.5" or MODE == "True" or MODE == "nan":
        tl.store(o_ptr + i, tl.full([1], 20, tl.int32))
    else:
        tl.store(o_ptr + i, tl.full([1], 10, tl.int32))


@triton.jit
def k4(o_ptr, A: tl.constexpr, B: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], len(A) * 10 + len(B), tl.int32))


@triton.jit
def store_c(p, C):
    tl.store(p + tl.arange(0, 1), tl.full([1], C, tl.float32))


@triton.jit
def pick(p, MODE):
    if MODE == "x_y":
        tl.store(p + tl.arange(0, 1), tl.full([1], 1, tl.int32))
    else:
        tl.store(p + tl.arange(0, 1), tl.full([1], 2, tl.int32))


@triton.jit
def k5_nan(o_ptr):
    store_c(o_ptr, float('nan'))
    store_c(o_ptr + 1, -float('nan'))


@triton.jit
def k5_nan_rev(o_ptr):
    store_c(o_ptr, -float('nan'))
    store_c(o_ptr + 1, float('nan'))


@triton.jit
def k5_inf(o_ptr):
    store_c(o_ptr, float('inf'))
    store_c(o_ptr + 1, -float('inf'))


@triton.jit
def k5_num(o_ptr):
    store_c(o_ptr, 1)
    store_c(o_ptr + 1, 1.0)
    store_c(o_ptr + 2, True)


@triton.jit
def k5_str(o_ptr):
    pick(o_ptr, "x[y")
    pick(o_ptr + 1, "x_y")


@triton.jit
def k5_str_rev(o_ptr):
    pick(o_ptr, "x_y")
    pick(o_ptr + 1, "x[y")


@triton.jit
def k5_str_ctl(o_ptr):
    pick(o_ptr, "a")
    pick(o_ptr + 1, "x_y")
'''

VALS = {"1": 1, "1.0": 1.0, "True": True, "0": 0, "0.0": 0.0, "False": False, "-0.0": -0.0,
        "'1'": "1", "None": None, "'None'": "None", "2.5": 2.5, "'2.5'": "2.5", "'True'": "True",
        "nan": float("nan"), "-nan": -float("nan"), "'nan'": "nan", "'a-b'": "a-b", "'c'": "c",
        "'a'": "a", "'b-c'": "b-c"}


def load_module(tmp):
    path = os.path.join(tmp, "keys_mod.py")
    with open(path, "w") as fh:
        fh.write(SRC)
    import importlib.util
    spec = importlib.util.spec_from_file_location("keys_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def launch(mod, kern, args):
    """One launch of kernel `kern` with the compile-time values `args` (names of VALS); its stored words."""
    import torch
    import struct
    if kern == "k1":
        o = torch.zeros(3, dtype=torch.int32, device="cuda")
        x = torch.full((1,), 2 ** 24 + 1, dtype=torch.int32, device="cuda")
        mod.k1[(1,)](o, x, C=VALS[args[0]])
        torch.cuda.synchronize()
        return o.tolist()
    if kern == "k2":
        o = torch.zeros(1, dtype=torch.float32, device="cuda")
        mod.k2[(1,)](o, C=VALS[args[0]])
        torch.cuda.synchronize()
        return [f"{v & 0xffffffff:08x}" for v in o.view(torch.int32).tolist()]
    if kern == "k3":
        o = torch.zeros(1, dtype=torch.int32, device="cuda")
        mod.k3[(1,)](o, MODE=VALS[args[0]])
        torch.cuda.synchronize()
        return o.tolist()
    if kern == "k4":
        o = torch.zeros(1, dtype=torch.int32, device="cuda")
        mod.k4[(1,)](o, A=VALS[args[0]], B=VALS[args[1]])
        torch.cuda.synchronize()
        return o.tolist()
    fn = getattr(mod, kern)
    if kern in ("k5_nan", "k5_nan_rev", "k5_inf", "k5_num"):
        o = torch.zeros(3, dtype=torch.float32, device="cuda")
        fn[(1,)](o)
        torch.cuda.synchronize()
        n = 3 if kern == "k5_num" else 2
        return [f"{v & 0xffffffff:08x}" for v in o.view(torch.int32).tolist()[:n]]
    o = torch.zeros(2, dtype=torch.int32, device="cuda")
    fn[(1,)](o)
    torch.cuda.synchronize()
    return o.tolist()


def child(plan):
    tmp = tempfile.mkdtemp(prefix="tva-keys-")
    mod = load_module(tmp)
    print("RESULT " + json.dumps([launch(mod, kern, args) for kern, args in json.loads(plan)]))


def run(plan, cache=None):
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = cache or tempfile.mkdtemp(prefix="tva-keys-cache-")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", json.dumps(plan)],
                       capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise SystemExit(f"child failed: {r.stderr[-2000:]}")


def alone(kern, args):
    return run([[kern, args]])[0]


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    wrong = 0

    def report(label, seq, got, ctl):
        nonlocal wrong
        bad = [i for i, (g, c) in enumerate(zip(got, ctl)) if g != c]
        wrong += len(bad)
        print(f"  {label}: " + "; ".join(f"{a} -> {g}{'' if g == c else f' (alone: {c}) DIFFERS'}"
                                      for a, g, c in zip(seq, got, ctl)))

    print("K1. one process, fresh cache, kernel k1 (x * C, isinstance(C, float), C is True or C is False), C =")
    for a, b in (("1", "1.0"), ("1.0", "1"), ("1", "True"), ("True", "1"), ("0", "0.0"), ("0.0", "0"),
                 ("0", "False"), ("-0.0", "0")):
        got = run([["k1", [a]], ["k1", [b]]])
        report(f"{a} then {b}", [a, b], got, [alone("k1", [a]), alone("k1", [b])])
    print("K2. one process, fresh cache, kernel k2 (tl.full([1], C, tl.float32)), C =")
    for a, b in (("nan", "-nan"), ("-nan", "nan")):
        got = run([["k2", [a]], ["k2", [b]]])
        report(f"{a} then {b}", [a, b], got, [alone("k2", [a]), alone("k2", [b])])
    print("K3. kernel k3 (5 for None, 20 for a string, 10 otherwise); MODE in a first process, then MODE in a "
          "second process on the same cache folder:")
    for a, b in (("1", "'1'"), ("'1'", "1"), ("None", "'None'"), ("'None'", "None"), ("2.5", "'2.5'"),
                 ("True", "'True'"), ("nan", "'nan'")):
        shared = tempfile.mkdtemp(prefix="tva-keys-shared-")
        got = run([["k3", [a]]], cache=shared) + run([["k3", [b]]], cache=shared)
        report(f"{a} then {b}", [a, b], got, [alone("k3", [a]), alone("k3", [b])])
    got = run([["k3", ["1"]], ["k3", ["'1'"]]])
    report("1 then '1' in one process (its own cache folder)", ["1", "'1'"], got, [alone("k3", ["1"]), alone("k3", ["'1'"])])
    print("K4. kernel k4 (10·len(A) + len(B)); (A, B) in a first process, then in a second on the same cache folder:")
    shared = tempfile.mkdtemp(prefix="tva-keys-shared-")
    got = run([["k4", ["'a-b'", "'c'"]]], cache=shared) + run([["k4", ["'a'", "'b-c'"]]], cache=shared)
    report("('a-b', 'c') then ('a', 'b-c')", ["('a-b', 'c')", "('a', 'b-c')"], got,
           [alone("k4", ["'a-b'", "'c'"]), alone("k4", ["'a'", "'b-c'"])])
    print("K5. one kernel calling a helper twice (fresh cache); the stored words, against each call alone:")
    single = {"nan": "7fc00000", "-nan": "ffc00000", "inf": "7f800000", "-inf": "ff800000", "1": "3f800000",
              "1.0": "3f800000", "True": "3f800000"}
    for kern, args, want in (("k5_nan", ["nan", "-nan"], ["7fc00000", "ffc00000"]),
                             ("k5_nan_rev", ["-nan", "nan"], ["ffc00000", "7fc00000"]),
                             ("k5_inf", ["inf", "-inf"], ["7f800000", "ff800000"]),
                             ("k5_num", ["1", "1.0", "True"], ["3f800000"] * 3),
                             ("k5_str", ["'x[y'", "'x_y'"], [2, 1]),
                             ("k5_str_rev", ["'x_y'", "'x[y'"], [1, 2]),
                             ("k5_str_ctl", ["'a'", "'x_y'"], [2, 1])):
        got = run([[kern, []]])[0]
        report(f"{kern}: helper({', '.join(args)})", args, got, want)
    print(f"launches that ran code compiled for other values than their own: {wrong}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
