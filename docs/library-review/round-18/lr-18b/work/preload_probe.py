"""lr-18b: `JITFunction.preload` puts a kernel compiled with options of the host's choosing into the JIT's cache under
the key a plain launch computes (`triton-3.8.0-jit.py:824-875`: the key and the options are read from the JSON, and
`_do_compile` stores the kernel at that key, `:902-903`), so a later `k[grid](…)` with no option runs that code.
`Compiler.Faithful`'s scope 4 names the launch, `warmup`, `triton.compile` and a `triton.Config` as the places an
option is given, not `preload`.

Step G (a first process): with a `jit_cache_hook` that records the specialization data and lets the compile go on
(it returns False), launch the kernel `y = tl.math.exp2(x)` once with `enable_reflect_ftz=False` and once plainly,
and keep the first's data with the second's key.
Step P (a second process): `k.preload(<that data>)`, then a plain `k[(1,)](x, y)` on `x = -130` (no option given
anywhere). The model (`Launch` with Triton's defaults, `enable_reflect_ftz` at its default `True`) is libdevice's
flush to zero: `exp2(-130) = 0`; with `enable_reflect_ftz=False` it is `2^-130` (evidence/compiled-options O1).
C is the control: the plain launch in a fresh process with no preload.

Run: `CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I preload_probe.py > preload_probe.out`.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

KMOD = '''import triton
import triton.language as tl


@triton.jit
def k(x_ptr, y_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(y_ptr + i, tl.math.exp2(x))
'''


def load(d):
    p = os.path.join(d, "pmod.py")
    if not os.path.exists(p):
        open(p, "w").write(KMOD)
    spec = importlib.util.spec_from_file_location("pmod", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pmod"] = m
    spec.loader.exec_module(m)
    return m


def step(name, d):
    import torch
    import triton
    m = load(d)
    x = torch.full((16,), -130.0, dtype=torch.float32, device="cuda")
    y = torch.full((16,), -7.0, dtype=torch.float32, device="cuda")
    if name == "G":
        seen = []

        def hook(*args, **kw):
            seen.append((kw["key"], kw["compile"]["specialization_data"]))
            return False
        triton.knobs.runtime.jit_cache_hook = hook
        m.k[(1,)](x, y, enable_reflect_ftz=False)
        m.k[(1,)](x, y)
        torch.cuda.synchronize()
        triton.knobs.runtime.jit_cache_hook = None
        data = json.loads(seen[0][1])
        data["key"] = seen[1][0]
        open(os.path.join(d, "spec.json"), "w").write(json.dumps(data))
        print(f"- G: recorded {len(seen)} compiles; the first's options name enable_reflect_ftz = "
              f"{data['options'].get('enable_reflect_ftz')}; the key is the plain launch's")
        return
    if name == "P":
        m.k.preload(open(os.path.join(d, "spec.json")).read())
    m.k[(1,)](x, y)
    torch.cuda.synchronize()
    v = float(y[0])
    print(f"- {name}: a plain k[(1,)](x, y) stored exp2(-130) = {v!r} (0.0 is the model's; 2^-130 = {2.0 ** -130!r})")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        step(sys.argv[1], sys.argv[2])
        sys.exit(0)
    import torch
    import triton
    print(f"Triton {triton.__version__}, {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    d = tempfile.mkdtemp()
    for s in ("G", "P", "C"):
        # every step a fresh process, with a fresh Triton cache, so that nothing compiled earlier is reused
        env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(d, f"cache-{s}"))
        r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), s, d], capture_output=True, text=True, env=env)
        print(r.stdout.strip() or f"- {s}: exit {r.returncode}: " + r.stderr.strip().splitlines()[-1][:200].replace(d, "<tmp>"))
