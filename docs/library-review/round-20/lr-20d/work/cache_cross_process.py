"""lr-20d, library review round 20 (survey area: rules).

Question: Compiler.Faithful's scopes 2 (knobs), 3 (hooks) and 10 (Triton and the kernel's module as
installed and written) speak of the claimed launch's own process. Triton's disk cache is keyed by
`get_cache_key` (`triton_key()`, the source's hash, the backend, the options and the environment
variables in `CACHE_INVALIDATING_ENV_VARS`, `primitives/triton-3.8.0-GetEnv.h`). Does code compiled by
an EARLIER process that was outside those scopes reach a LATER process that is inside all of them,
through a shared cache directory (the default one, under the user's home directory, is shared by every
process of a user)?

Each case: process A (outside the scope) launches the kernel on a fresh cache folder S; process B (all
of Triton at its defaults, no change to Triton, no hook) then launches the same kernel on S; process C
(the same as B) launches it on a fresh folder of its own, as the control. Every process is a new
Python interpreter.

- X1: A assigns `tl.exp = tl.log` (scope 10 in A); the kernel stores `tl.exp(x)` for x = 2.
- X2: A sets `triton.knobs.propagate_env = False` and `triton.knobs.language.fp32_default = "ieee"`
  from Python (scope 2 in A); the kernel stores `tl.dot(a, I)` of 16x16 float32 blocks, with
  a = 1 + 2^-12 (tf32 keeps 1.0, ieee keeps 1 + 2^-12).
- X3: A runs with `TRITON_KERNEL_OVERRIDE=1` and an override folder whose `.ttir` adds 2.0 where the
  source adds 1.0 (scope 2 in A: `TRITON_KERNEL_OVERRIDE` is not among the exceptions, and is not in
  `CACHE_INVALIDATING_ENV_VARS`); the IR is first dumped by a process with `TRITON_KERNEL_DUMP=1`
  (an exception of scope 2) on a cache folder of its own.
- X4 (control of the key): A sets `TRITON_F32_DEFAULT=ieee` in the environment (in
  `CACHE_INVALIDATING_ENV_VARS`), then B as above.

Run from the TVA root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
No path is printed.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

KSRC = '''import triton
import triton.language as tl


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
def addk(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    tl.store(o_ptr + i, tl.load(x_ptr + i) + 1.0)
'''


def child(mode, kdir):
    import torch
    import triton
    import triton.language as tl
    if mode == "exp_patch":
        tl.exp = tl.log
    if mode == "dot_knob":
        triton.knobs.propagate_env = False
        triton.knobs.language.fp32_default = "ieee"
    sys.path.insert(0, kdir)
    import kmod_lr20d as kmod
    if mode.startswith("exp"):
        x = torch.full((16,), 2.0, device="cuda")
        o = torch.zeros(16, device="cuda")
        kmod.expk[(1,)](x, o)
        return float(o[0].item())
    if mode.startswith("dot"):
        a = torch.full((16, 16), 1.0 + 2.0 ** -12, device="cuda")
        b = torch.eye(16, device="cuda")
        o = torch.zeros(16, 16, device="cuda")
        kmod.dotk[(1,)](a, b, o)
        return o[0, 0].item().hex()
    if mode.startswith("add"):
        x = torch.full((16,), 5.0, device="cuda")
        o = torch.zeros(16, device="cuda")
        kmod.addk[(1,)](x, o)
        return float(o[0].item())
    raise SystemExit(mode)


def run(mode, kdir, cache, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
    env["TRITON_CACHE_DIR"] = cache
    env.update(extra or {})
    p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "child", mode, kdir],
                       env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return "ERROR " + (p.stderr.strip().splitlines() or ["?"])[-1][:160]


def main():
    import torch
    import triton
    tmp = tempfile.mkdtemp(prefix="lr20d-x-")
    kdir = os.path.join(tmp, "k")
    os.makedirs(kdir)
    with open(os.path.join(kdir, "kmod_lr20d.py"), "w") as fh:
        fh.write(KSRC)
    fresh = lambda: tempfile.mkdtemp(prefix="c-", dir=tmp)
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"python {sys.version.split()[0]}")
    rows = []

    def case(name, a_mode, plain_mode, a_extra=None):
        s = fresh()
        ra = run(a_mode, kdir, s, a_extra)
        rb = run(plain_mode, kdir, s)
        rc = run(plain_mode, kdir, fresh())
        verdict = "B ran A's code" if (rb == ra and rb != rc) else ("B ran its own code" if rb == rc else "?")
        rows.append(verdict)
        print(f"{name}: A (outside the scope) -> {ra}; B (defaults, shared cache) -> {rb}; "
              f"C (defaults, fresh cache) -> {rc}: {verdict}")

    case("X1 tl.exp = tl.log in A; tl.exp(2.0)", "exp_patch", "exp_plain")
    case("X2 fp32_default = 'ieee' from Python, propagate_env False, in A; tl.dot(1 + 2^-12, I)[0, 0]",
         "dot_knob", "dot_plain")
    # X3: dump the IR, edit it, override it in A.
    dump = os.path.join(tmp, "dump")
    over = os.path.join(tmp, "over")
    r0 = run("add_plain", kdir, fresh(), {"TRITON_KERNEL_DUMP": "1", "TRITON_DUMP_DIR": dump})
    edited = 0
    for f in glob.glob(os.path.join(dump, "*", "addk.ttir")):
        d = os.path.join(over, os.path.basename(os.path.dirname(f)))
        os.makedirs(d, exist_ok=True)
        text = open(f).read()
        new = text.replace("dense<1.000000e+00>", "dense<2.000000e+00>")
        edited += new != text
        with open(os.path.join(d, "addk.ttir"), "w") as fh:
            fh.write(new)
    print(f"X3 setup: dump run -> {r0}; .ttir files edited: {edited}")
    case("X3 TRITON_KERNEL_OVERRIDE=1 in A (the .ttir adds 2.0); 5.0 + 1.0", "add_plain", "add_plain",
         {"TRITON_KERNEL_OVERRIDE": "1", "TRITON_OVERRIDE_DIR": over})
    case("X4 (control) TRITON_F32_DEFAULT=ieee in A's environment; tl.dot(1 + 2^-12, I)[0, 0]",
         "dot_plain", "dot_plain", {"TRITON_F32_DEFAULT": "ieee"})
    print(f"cases where B, inside every scope, ran code A compiled outside it: "
          f"{sum(r == 'B ran A' + chr(39) + 's code' for r in rows)} of {len(rows)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        print("RESULT " + json.dumps(child(sys.argv[2], sys.argv[3])))
    else:
        main()
