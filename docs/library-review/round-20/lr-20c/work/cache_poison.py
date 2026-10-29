"""lr-20c, round 20: does a launch whose own process is inside every scope point of `Compiler.Faithful` (no knob,
no hook, no option, nothing changes Triton or the kernel's module, compile-time values that collide with no
earlier launch's) run code that an earlier process, outside the scope, wrote into the same Triton cache folder?

C1. Process A runs `tl.exp = tl.log` (scope 10, in A only) and launches `y = tl.exp(x)` on cache folder C.
    Process B, clean, launches the same kernel with the same arguments on C. Control: B alone on a fresh folder.
C2. Process A sets TRITON_KERNEL_OVERRIDE=1 with an override folder holding the kernel's `.ttir` edited from
    `x + 1` to `x + 2` (scope 2, in A only) and launches `y = x + 1` on cache folder C. Process B, clean (no
    Triton variable but TRITON_CACHE_DIR, an exception of scope 2), launches it on C.
C3. One process, a fresh cache folder: `triton.knobs.compilation.override = True` (TRITON_OVERRIDE_DIR, which only
    names a folder, set) for a first launch of `y = x + 1`, then the knob back to its default and a second launch:
    is the knob at its default at the second launch enough?
The key Triton finds a compiled kernel by: `get_cache_key` (triton/runtime/cache.py) hashes Triton's version, the
source key (`ASTSource.hash`: the kernel's `cache_key`, whose `DependenciesFinder` skips `triton.language`
attributes, primitives/triton-3.8.0-jit.py visit_Attribute), the backend, the options and the variables of
`CACHE_INVALIDATING_ENV_VARS` (primitives/triton-3.8.0-GetEnv.h), which does not list TRITON_KERNEL_OVERRIDE.

Usage: CUDA_VISIBLE_DEVICES=0 <TVA's venv python> -I cache_poison.py SCRATCH   (SCRATCH: an empty folder outside
TVA; its path is never printed)
"""
import glob
import json
import os
import shutil
import subprocess
import sys

SRC = '''import triton
import triton.language as tl


@triton.jit
def expk(x_ptr, y_ptr):
    i = tl.arange(0, 4)
    tl.store(y_ptr + i, tl.exp(tl.load(x_ptr + i)))


@triton.jit
def add1(x_ptr, y_ptr):
    i = tl.arange(0, 4)
    tl.store(y_ptr + i, tl.load(x_ptr + i) + 1.0)
'''


def child(mode, kern, moddir):
    sys.path.insert(0, moddir)
    import torch
    import triton
    import triton.language as tl
    import kmod
    if mode == "patch":
        tl.exp = tl.log
    x = torch.tensor([1.0, 2.0, 4.0, 8.0], device="cuda")
    y = torch.zeros(4, device="cuda")
    out = []
    if mode == "knob-then-reset":
        # the override knob set from Python for a first launch, then put back to its default
        k = triton.knobs.compilation
        default = k.override
        k.override = True
        getattr(kmod, kern)[(1,)](x, y)
        torch.cuda.synchronize()
        out.append([round(v, 4) for v in y.tolist()])
        k.override = default
        y.zero_()
    getattr(kmod, kern)[(1,)](x, y)
    torch.cuda.synchronize()
    out.append([round(v, 4) for v in y.tolist()])
    print("RESULT " + json.dumps(out if len(out) > 1 else out[0]))


def run(scratch, mode, kern, cache, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
    env["TRITON_CACHE_DIR"] = cache
    env.update(extra or {})
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", mode, kern,
                        os.path.join(scratch, "mod")], capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return "failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]


def main():
    scratch = os.path.realpath(sys.argv[1])
    assert os.path.isdir(scratch) and not os.listdir(scratch)
    os.makedirs(os.path.join(scratch, "mod"))
    open(os.path.join(scratch, "mod", "kmod.py"), "w").write(SRC)
    d = lambda name: os.path.join(scratch, name)  # noqa: E731
    import triton
    import torch
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    print("x = [1, 2, 4, 8]")
    print("C1. tl.exp")
    print(f"  B alone, fresh cache folder: {run(scratch, 'clean', 'expk', d('c1-fresh'))}")
    print(f"  A (tl.exp = tl.log) on folder C: {run(scratch, 'patch', 'expk', d('c1'))}")
    print(f"  B, clean, then on folder C: {run(scratch, 'clean', 'expk', d('c1'))}")
    print("C2. TRITON_KERNEL_OVERRIDE")
    print(f"  B alone, fresh cache folder: {run(scratch, 'clean', 'add1', d('c2-fresh'))}")
    # dump the kernel's IR, then make an override folder whose .ttir adds 2.0
    run(scratch, "clean", "add1", d("c2-dump-cache"), {"TRITON_KERNEL_DUMP": "1", "TRITON_DUMP_DIR": d("dump")})
    ttirs = glob.glob(os.path.join(d("dump"), "*", "add1.ttir"))
    assert len(ttirs) == 1, ttirs
    sub = os.path.basename(os.path.dirname(ttirs[0]))
    os.makedirs(os.path.join(d("override"), sub))
    text = open(ttirs[0]).read()
    assert "1.000000e+00" in text
    open(os.path.join(d("override"), sub, "add1.ttir"), "w").write(text.replace("1.000000e+00", "2.000000e+00"))
    print(f"  A (TRITON_KERNEL_OVERRIDE=1, the .ttir edited to + 2.0) on folder C: "
          f"{run(scratch, 'clean', 'add1', d('c2'), {'TRITON_KERNEL_OVERRIDE': '1', 'TRITON_OVERRIDE_DIR': d('override')})}")
    print(f"  B, clean, then on folder C: {run(scratch, 'clean', 'add1', d('c2'))}")
    print("C3. one process, a fresh cache folder: knobs.compilation.override = True (TRITON_OVERRIDE_DIR naming the "
          "edited .ttir) for a first launch, then the knob back to its default and a second launch")
    print(f"  [first launch, second launch]: "
          f"{run(scratch, 'knob-then-reset', 'add1', d('c3'), {'TRITON_OVERRIDE_DIR': d('override')})}")
    shutil.rmtree(scratch + os.sep + "mod", ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
