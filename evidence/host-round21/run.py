"""TVA library evidence (library issues L103, L109, L110): what a launch runs when the code was compiled earlier, in
its own process or in another one through the cache directory, and when the host process itself is in another
state (`python -O`, a changed floating-point environment).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/host-round21/run.py
Output: evidence/host-round21/out.txt (README.md says what it shows).

Every case runs in processes of their own (each a new interpreter, all of Triton's variables removed from the
environment), on a cache directory made for the case; the kernel module is written to a temporary file and imported
from the child (another module, as a host program would). No path is printed.

K1-K5 (L103, scope 11's check): `kc` stores `x * C` for the `int32` `x = 2^24 + 1` and a `tl.constexpr` `C` (the
    integer product `C·x` for an `int` `C`, the `float32` product, `2^24·C`, for a `float`):
    K1 one process on a fresh cache directory, `C = 2` then `C = 2.0` (and the reverse);
    K2 one process: `C = 1` on a first cache directory, then `triton.knobs.cache.dir` set to a second, fresh one,
       then `C = 1.0`;
    K3 one process on a fresh cache directory, `kab` (stores `10·len(A) + len(B)`) with `(A, B) = ("a-b", "c")`,
       then `("a", "b-c")`;
    K4 one process on a fresh cache directory: `triton.autotune` over the configs `{"C": 1}` and `{"C": 1.0}` of a
       copy of `kc`, with a `do_bench` that runs each config once and times the second config faster;
    K5 the controls: each value launched alone, the first launch of its process on a fresh cache directory.
X1-X5 (L109): a first process A, outside the scope, launches on a cache directory S; then B, inside every scope,
    launches the same kernel on S; C, as B, on a fresh directory; D, as B, on S with `TRITON_ALWAYS_COMPILE=1`:
    X1 A assigns `tl.exp = tl.log`; the kernel stores `tl.exp(x)` for `x = 2`;
    X2 A sets `triton.knobs.propagate_env = False` and `triton.knobs.language.fp32_default = "ieee"`; the kernel
       stores `tl.dot(a, I)[0, 0]` of `16 x 16` `float32` blocks with `a = 1 + 2^-12` (tf32 keeps `1.0`);
    X3 A calls `fesetround(FE_TOWARDZERO)`; the kernel stores `tl.full([1], 0.1, tl.float32)`;
    X4 A runs as `python -I -O`; the kernel `kd` (below) is rejected on the host without `-O`;
    X5 one process: the knob of X2 set for a first launch, then deleted (its default) for a second launch, with and
       without `TRITON_ALWAYS_COMPILE=1` in the process's environment from its start.
H1-H3 (L110): the host process's own state, each case on a fresh cache directory:
    H1 `kd`: `d = tl.float16`, then a `for` loop over `range(n)` whose body assigns `d = tl.float32` (a carried
       name given another type: rejected on the host, SEMANTICS "Names after a loop"), then
       `tl.store(o, tl.full([1], 1 + 2**-11, d).to(tl.float32))`, with `n = 3`; under `python -I`, `python -I -O`,
       `python -I -OO`, and `python -s` with `PYTHONOPTIMIZE=1`;
    H2 `fesetround(FE_TOWARDZERO)` (glibc's `libm`): `envk` stores `tl.full([1], 0.1, tl.float32)`, the `float32`
       argument `y = 0.1`, `tl.full([1], 1e-40, tl.float32)` and `tl.full([1], A + B, tl.float64)` for the
       constexprs `A = 0.1`, `B = 0.2`;
    H3 flush-to-zero and denormals-are-zero (MXCSR bits 15 and 6): set by a C function calling `_mm_setcsr`, and by
       loading a shared library built with `gcc -shared -ffast-math` (its start-up code); `envk` as in H2.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

KMOD = '''import triton
import triton.language as tl


@triton.jit
def kc(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * C)


@triton.jit
def kc_t(o_ptr, x_ptr, C: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.load(x_ptr + i) * C)


@triton.jit
def kab(o_ptr, A: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o_ptr + i, tl.full([1], 10 * len(A) + len(B), tl.int32))


@triton.jit
def expk(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    tl.store(o_ptr + i, tl.exp(tl.load(x_ptr + i)))


@triton.jit
def dotk(a_ptr, b_ptr, o_ptr):
    i = tl.arange(0, 16)
    a = tl.load(a_ptr + i[:, None] * 16 + i[None, :])
    b = tl.load(b_ptr + i[:, None] * 16 + i[None, :])
    tl.store(o_ptr + i[:, None] * 16 + i[None, :], tl.dot(a, b))


@triton.jit
def kd(o_ptr, n):
    d = tl.float16
    for i in range(n):
        d = tl.float32
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], 1 + 2**-11, d).to(tl.float32))


@triton.jit
def envk(o32_ptr, o64_ptr, y, A: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, 1)
    tl.store(o32_ptr + i, tl.full([1], 0.1, tl.float32))
    tl.store(o32_ptr + 1 + i, y + tl.zeros([1], tl.float32))
    tl.store(o32_ptr + 2 + i, tl.full([1], 1e-40, tl.float32))
    tl.store(o64_ptr + i, tl.full([1], A + B, tl.float64))
'''

CSRC = '''#include <xmmintrin.h>
void set_ftz_daz(void) { _mm_setcsr(_mm_getcsr() | 0x8040); }
unsigned get_csr(void) { return _mm_getcsr(); }
'''
FMSRC = '''double fm_identity(double x) { return x; }
'''

FE_TOWARDZERO = 0xC00  # x86-64 glibc


def child(plan, kdir):
    """Run one process's plan: its setup, then its launches; return each launch's result."""
    import ctypes
    import importlib.util
    import torch
    import triton
    import triton.language as tl
    plan = json.loads(plan)
    for step in plan.get("setup", []):
        if step == "patch_exp":
            tl.exp = tl.log
        elif step == "knob_ieee":
            triton.knobs.propagate_env = False
            triton.knobs.language.fp32_default = "ieee"
        elif step == "fesetround_rz":
            assert ctypes.CDLL("libm.so.6").fesetround(FE_TOWARDZERO) == 0
        elif step == "ftz_daz":
            ctypes.CDLL(os.path.join(kdir, "libftz.so")).set_ftz_daz()
        elif step == "load_fast_math_lib":
            ctypes.CDLL(os.path.join(kdir, "libfm.so"))
        else:
            raise SystemExit(f"unknown step {step}")
    spec = importlib.util.spec_from_file_location("kmod_r21", os.path.join(kdir, "kmod_r21.py"))
    kmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kmod)
    out = []
    for launch in plan["launches"]:
        kind = launch[0]
        try:
            if kind == "kc":
                x = torch.full((1,), 2 ** 24 + 1, dtype=torch.int32, device="cuda")
                o = torch.zeros(1, dtype=torch.int32, device="cuda")
                kmod.kc[(1,)](o, x, launch[1])
                out.append(int(o.item()))
            elif kind == "kab":
                o = torch.zeros(1, dtype=torch.int32, device="cuda")
                kmod.kab[(1,)](o, launch[1], launch[2])
                out.append(int(o.item()))
            elif kind == "tuned":
                calls = []

                def bench(kernel_call, quantiles):
                    kernel_call()
                    torch.cuda.synchronize()
                    calls.append(1)
                    t = 2.0 if len(calls) == 1 else 1.0
                    return [t, t, t]

                # `triton.Config` is hashed and compared by its keywords and its `pre_hook`
                # (`primitives/triton-3.8.0-autotuner.py:393-405`): two configs `{"C": 1}` and `{"C": 1.0}` are one
                # key of the autotuner's timings, so the second gets a `pre_hook` that does nothing
                tuned = triton.autotune(configs=[triton.Config({"C": 1}),
                                                 triton.Config({"C": 1.0}, pre_hook=lambda nargs: None)], key=[],
                                        do_bench=bench)(kmod.kc_t)
                x = torch.full((1,), 2 ** 24 + 1, dtype=torch.int32, device="cuda")
                o = torch.zeros(1, dtype=torch.int32, device="cuda")
                tuned[(1,)](o, x)
                torch.cuda.synchronize()
                out.append({"chosen": repr(tuned.best_config.kwargs["C"]), "stored": int(o.item()),
                            "benchmark runs": len(calls)})
            elif kind == "expk":
                x = torch.full((16,), 2.0, device="cuda")
                o = torch.zeros(16, device="cuda")
                kmod.expk[(1,)](x, o)
                out.append(float(o[0].item()))
            elif kind == "dotk":
                a = torch.full((16, 16), 1.0 + 2.0 ** -12, device="cuda")
                b = torch.eye(16, device="cuda")
                o = torch.zeros(16, 16, device="cuda")
                kmod.dotk[(1,)](a, b, o)
                out.append(o[0, 0].item().hex())
            elif kind == "kd":
                o = torch.full((1,), -7.0, device="cuda")
                kmod.kd[(1,)](o, launch[1])
                out.append(repr(o.item()))
            elif kind == "envk":
                o32 = torch.zeros(3, dtype=torch.float32, device="cuda")
                o64 = torch.zeros(1, dtype=torch.float64, device="cuda")
                kmod.envk[(1,)](o32, o64, 0.1, 0.1, 0.2)
                w32 = [format(v & 0xFFFFFFFF, "08x") for v in o32.cpu().view(torch.int32).tolist()]
                w64 = format(o64.cpu().view(torch.int64).item() & (2 ** 64 - 1), "016x")
                out.append({"0.1": w32[0], "y=0.1": w32[1], "1e-40": w32[2], "A+B": w64})
            elif kind == "knob_reset":
                del triton.knobs.language.fp32_default
                out.append(f"knob deleted, now {triton.knobs.language.fp32_default!r}")
            elif kind == "cache_dir":
                triton.knobs.cache.dir = os.path.join(kdir, launch[1])
                out.append("cache directory set to a fresh one")
            elif kind == "csr":
                csr = ctypes.CDLL(os.path.join(kdir, "libftz.so")).get_csr()
                out.append(f"MXCSR FTZ {bool(csr & 0x8000)}, DAZ {bool(csr & 0x40)}")
            else:
                raise SystemExit(f"unknown launch {kind}")
            torch.cuda.synchronize()
        except Exception as ex:  # a rejection on the host
            out.append(f"rejected on the host ({type(ex).__name__})")
    return out


def run(plan, kdir, cache, flags=("-I",), env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_") and not k.startswith("PYTHON")}
    env["TRITON_CACHE_DIR"] = os.path.join(kdir, cache)
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, *flags, os.path.abspath(__file__), "child", json.dumps(plan), kdir],
                       env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    last = (p.stderr.strip().splitlines() or ["?"])[-1]
    return "child failed (exit %d): %s" % (p.returncode, re.sub(r"(?<![\w.])/[^\s:]+", "<path>", last)[:160])


def main():
    import torch
    import triton
    kdir = tempfile.mkdtemp(prefix="tva-r21-host-")
    with open(os.path.join(kdir, "kmod_r21.py"), "w") as fh:
        fh.write(KMOD)
    with open(os.path.join(kdir, "ftz.c"), "w") as fh:
        fh.write(CSRC)
    with open(os.path.join(kdir, "fm.c"), "w") as fh:
        fh.write(FMSRC)
    subprocess.run(["/usr/bin/gcc", "-shared", "-fPIC", "-O1", "-o", os.path.join(kdir, "libftz.so"),
                    os.path.join(kdir, "ftz.c")], check=True)
    subprocess.run(["/usr/bin/gcc", "-shared", "-fPIC", "-O1", "-ffast-math", "-o", os.path.join(kdir, "libfm.so"),
                    os.path.join(kdir, "fm.c")], check=True)
    gcc = subprocess.run(["/usr/bin/gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"python {sys.version.split()[0]}, machine {os.uname().machine}, {gcc}")
    n = [0]

    def fresh():
        n[0] += 1
        return f"cache-{n[0]}"

    other = []  # launches that ran code compiled for other values, or outside the scope

    print("\nK. scope 11: compile-time values that share Triton's key (one process each line)")
    # keyed by `repr`: a dictionary keyed by the values themselves would equate `1` and `1.0`, as Triton's does
    ctl = {repr(v): run({"launches": [["kc", v]]}, kdir, fresh())[0] for v in (1, 1.0, 2, 2.0)}
    print(f"K5 controls, each value the first launch of its process on a fresh cache directory: "
          f"C = 1 -> {ctl['1']}, C = 1.0 -> {ctl['1.0']}, C = 2 -> {ctl['2']}, C = 2.0 -> {ctl['2.0']}")
    for a, b in ((2, 2.0), (2.0, 2)):
        r = run({"launches": [["kc", a], ["kc", b]]}, kdir, fresh())
        bad = r[1] != ctl[repr(b)]
        other += [bad]
        print(f"K1 fresh cache directory, C = {a!r} then C = {b!r}: {r} (C = {b!r} alone: {ctl[repr(b)]}): "
              f"{'the second ran the first value' + chr(39) + 's code' if bad else 'its own code'}")
    r = run({"launches": [["kc", 1], ["cache_dir", fresh()], ["kc", 1.0]]}, kdir, fresh())
    bad = r[2] != ctl["1.0"]
    other += [bad]
    print(f"K2 C = 1, then a fresh cache directory set in the process, then C = 1.0: {r}: "
          f"{'C = 1.0 ran C = 1' + chr(39) + 's code' if bad else 'its own code'}")
    alone = run({"launches": [["kab", "a", "b-c"]]}, kdir, fresh())[0]
    r = run({"launches": [["kab", "a-b", "c"], ["kab", "a", "b-c"]]}, kdir, fresh())
    bad = r[1] != alone
    other += [bad]
    print(f"K3 fresh cache directory, (A, B) = ('a-b', 'c') then ('a', 'b-c'): {r} (('a', 'b-c') alone: {alone}): "
          f"{'the second ran the first' + chr(39) + 's code' if bad else 'its own code'}")
    r = run({"launches": [["tuned"]]}, kdir, fresh())[0]
    bad = isinstance(r, dict) and r["chosen"] == "1.0" and r["stored"] != ctl["1.0"]
    other += [bad]
    print(f"K4 fresh process and cache directory, autotune over C = 1 and C = 1.0: {r} (C = 1.0 alone: {ctl['1.0']}): "
          f"{'the chosen config ran the benchmarked C = 1' + chr(39) + 's code' if bad else 'its own code'}")

    print("\nX. scope 12: code compiled outside the scope, found later (A outside; B, C, D inside every scope)")

    def leak(label, a_plan, plain, a_flags=("-I",), a_env=None):
        s = fresh()
        ra = run(a_plan, kdir, s, a_flags, a_env)
        rb = run(plain, kdir, s)
        rc = run(plain, kdir, fresh())
        rd = run(plain, kdir, s, env_extra={"TRITON_ALWAYS_COMPILE": "1"})
        bad = rb != rc
        other.append(bad)
        print(f"{label}: A -> {ra}; B (same directory) -> {rb}; C (fresh directory) -> {rc}; "
              f"D (same directory, TRITON_ALWAYS_COMPILE=1) -> {rd}: "
              f"{'B ran A' + chr(39) + 's code' if bad else 'B ran its own code'}"
              f"{', D its own' if rd == rc else ', D NOT its own'}")

    leak("X1 tl.exp = tl.log in A; tl.exp(2.0)", {"setup": ["patch_exp"], "launches": [["expk"]]},
         {"launches": [["expk"]]})
    leak("X2 fp32_default = 'ieee' from Python in A; tl.dot(1 + 2^-12, I)[0, 0]",
         {"setup": ["knob_ieee"], "launches": [["dotk"]]}, {"launches": [["dotk"]]})
    leak("X3 fesetround(FE_TOWARDZERO) in A; tl.full([1], 0.1, tl.float32) and the others of envk",
         {"setup": ["fesetround_rz"], "launches": [["envk"]]}, {"launches": [["envk"]]})
    leak("X4 A runs as python -I -O; kd with n = 3", {"launches": [["kd", 3]]}, {"launches": [["kd", 3]]},
         a_flags=("-I", "-O"))
    for env_extra in (None, {"TRITON_ALWAYS_COMPILE": "1"}):
        r = run({"setup": ["knob_ieee"], "launches": [["dotk"], ["knob_reset"], ["dotk"]]}, kdir, fresh(),
                env_extra=env_extra)
        c = run({"launches": [["dotk"]]}, kdir, fresh())[0]
        bad = r[2] != c
        other.append(bad)
        print(f"X5 one process{' with TRITON_ALWAYS_COMPILE=1' if env_extra else ''}: knob 'ieee', dotk, knob deleted, "
              f"dotk: {r} (the default's own code: {c}): "
              f"{'the second launch ran the ieee code' if bad else 'the second launch ran its own code'}")

    print("\nH. the host process's own state (each line in a process of its own, fresh cache directory)")
    for label, flags, env_extra in (("python -I", ("-I",), None), ("python -I -O", ("-I", "-O"), None),
                                    ("python -I -OO", ("-I", "-OO"), None),
                                    ("python -s, PYTHONOPTIMIZE=1", ("-s",), {"PYTHONOPTIMIZE": "1"})):
        r = run({"launches": [["kd", 3]]}, kdir, fresh(), flags, env_extra)
        print(f"H1 {label}: kd with n = 3 -> {r[0] if isinstance(r, list) else r}")
    plain = run({"launches": [["csr"], ["envk"]]}, kdir, fresh())
    print(f"H2/H3 plain process: {plain}")
    for label, setup in (("H2 fesetround(FE_TOWARDZERO)", ["fesetround_rz"]), ("H3 _mm_setcsr(FTZ | DAZ)", ["ftz_daz"]),
                         ("H3 a shared library built with -ffast-math loaded", ["load_fast_math_lib"])):
        r = run({"setup": setup, "launches": [["csr"], ["envk"]]}, kdir, fresh())
        diff = {k: f"{v} (plain {plain[1][k]})" for k, v in r[1].items() if v != plain[1][k]} \
            if isinstance(r, list) and isinstance(r[1], dict) else r
        print(f"{label}: {r[0] if isinstance(r, list) else ''}; envk words that differ from the plain process: {diff}")

    print(f"\nlaunches that ran code compiled for other values or outside the scope (K1-K4, X1-X5): "
          f"{sum(other)} of {len(other)}")
    shutil.rmtree(kdir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print("RESULT " + json.dumps(child(sys.argv[2], sys.argv[3])))
    else:
        main()
