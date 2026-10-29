"""lr-3a, library review round 3 (area `rules`): does `Rules.Compiler.Faithful`'s scope
("Triton's environment at its defaults", `Launch.Valid`, `debug = false`) exclude every way the
host can change what Triton 3.8.0 compiles?

Each case runs in a fresh subprocess with every `TRITON_*` variable removed and a new empty
`TRITON_CACHE_DIR`, so no environment variable the rule names is set. The case then changes the
compilation only from Python: a launch keyword (a field of the NVIDIA backend's `CUDAOptions`,
`primitives/triton-3.8.0-nvidia-compiler.py:108-133`) or a `triton.knobs` attribute
(`primitives/triton-3.8.0-knobs.py`). Each case prints whether a Triton environment variable is
set when the kernel compiles, and the result.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3a/work/options_knobs.py

Cases.
- dot: `tl.dot(a, b)` on float32 16x16 without `input_precision`; every `a` element is
  `1 + 2^-12` (not a tf32 value), `b` is ones: tf32 gives 16, ieee gives `16 + 2^-8`.
  - D0: control, nothing changed (expected tf32, as TRANSCRIPTION `[dot]` says).
  - D1: launch keyword `allowed_dot_input_precisions=("ieee",)`.
  - D2: `triton.knobs.propagate_env = False`, then `triton.knobs.language.fp32_default = "ieee"`.
  - D3: `triton.knobs.language.fp32_default = "ieee"` (default `propagate_env`): does Python set
    the environment variable?
- add: `y = x + 1.0` on float32.
  - A0: control.
  - A1: `ir_override=<file>.ptx`, the kernel's own PTX with the constant 1.0 replaced by 2.0.
  - A2: `triton.knobs.runtime.add_stages_inspection_hook`, a hook that rewrites the PTX stage
    (`add` becomes `sub`).
- exp2: `tl.math.exp2(x)` at `x = -130` (exact result `2^-130`, subnormal in float32).
  - E0: control (libdevice with flush-to-zero: expected 0).
  - E1: launch keyword `enable_reflect_ftz=False`.
- assert: `tl.device_assert(x > 0)` with `x = -1`, no `debug` argument.
  - S0: control (no `debug`: the assert is not compiled in).
  - S1: `triton.knobs.runtime.debug = True` (a plain attribute: no environment variable).
"""
import json
import os
import subprocess
import sys
import tempfile

CASE = r'''
import json, os, sys
import torch, triton, triton.language as tl
case = sys.argv[1]
res = {"case": case}

@triton.jit
def dotk(a_ptr, b_ptr, c_ptr):
    r = tl.arange(0, 16)
    a = tl.load(a_ptr + r[:, None] * 16 + r[None, :])
    b = tl.load(b_ptr + r[:, None] * 16 + r[None, :])
    tl.store(c_ptr + r[:, None] * 16 + r[None, :], tl.dot(a, b))

@triton.jit
def addk(x_ptr, y_ptr, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=o < n) + 1.0, mask=o < n)

@triton.jit
def exp2k(x_ptr, y_ptr, n, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.math.exp2(tl.load(x_ptr + o, mask=o < n)), mask=o < n)

@triton.jit
def assertk(x_ptr, y_ptr):
    x = tl.load(x_ptr)
    tl.device_assert(x > 0, "x must be positive")
    tl.store(y_ptr, x + 1.0)

def env_set():
    return sorted(k for k in os.environ if k.startswith("TRITON_") and k != "TRITON_CACHE_DIR")

kw = {}
if case in ("D0", "D1", "D2", "D3"):
    if case == "D1":
        kw["allowed_dot_input_precisions"] = ("ieee",)
    if case == "D2":
        triton.knobs.propagate_env = False
        triton.knobs.language.fp32_default = "ieee"
    if case == "D3":
        triton.knobs.language.fp32_default = "ieee"
    a = torch.full((16, 16), 1 + 2 ** -12, device="cuda", dtype=torch.float32)
    b = torch.ones((16, 16), device="cuda", dtype=torch.float32)
    c = torch.empty((16, 16), device="cuda", dtype=torch.float32)
    res["env_at_compile"] = env_set()
    k = dotk[(1,)](a, b, c, **kw)
    torch.cuda.synchronize()
    v = c[0, 0].item()
    res["c00"] = v
    res["c00_minus_16_times_2^8"] = (v - 16) * 2 ** 8
    m = k.asm["ttgir"]
    res["ttgir_inputPrecision"] = [p for p in ("tf32x3", "tf32", "ieee") if f"inputPrecision = {p}" in m] or ["(default: ieee)"]
elif case in ("A0", "A1", "A2"):
    n, B = 1024, 1024
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    if case == "A1":
        k0 = addk.warmup(x, y, n, B, grid=(1,))
        ptx = k0.asm["ptx"]
        assert "0f3F800000" in ptx, "constant 1.0 not found in PTX"
        path = os.path.join(os.environ["LR3A_TMP"], "addk_override.ptx")
        open(path, "w").write(ptx.replace("0f3F800000", "0f40000000"))
        kw["ir_override"] = path
    if case == "A2":
        def hook(*args):
            if not args:
                return "lr3a-hook", "lr3a-hook-1"
            backend, stages, options, language, capability = args
            orig = stages["ptx"]
            def ptx_stage(src, metadata):
                out = orig(src, metadata)
                return out.replace("add.f32", "sub.f32").replace("add.rn.f32", "sub.rn.f32")
            stages["ptx"] = ptx_stage
        triton.knobs.runtime.add_stages_inspection_hook = hook
    res["env_at_compile"] = env_set()
    addk[(1,)](x, y, n, B, **kw)
    torch.cuda.synchronize()
    d = (y - x)
    res["y_minus_x_distinct"] = sorted(set(d.tolist()))
elif case in ("E0", "E1"):
    if case == "E1":
        kw["enable_reflect_ftz"] = False
    x = torch.tensor([-130.0, -126.0, -149.0, 0.0], device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    res["env_at_compile"] = env_set()
    k = exp2k[(1,)](x, y, 4, 4, **kw)
    torch.cuda.synchronize()
    res["x"] = x.tolist()
    res["exp2"] = [float.hex(v) for v in y.tolist()]
    p = k.asm["ptx"]
    res["ptx_ex2"] = sorted(set(l.split()[0] for l in p.splitlines() if "ex2" in l))
elif case in ("S0", "S1"):
    if case == "S1":
        triton.knobs.runtime.debug = True
    x = torch.tensor([-1.0], device="cuda")
    y = torch.zeros(1, device="cuda")
    res["env_at_compile"] = env_set()
    try:
        k = assertk[(1,)](x, y)
        torch.cuda.synchronize()
        res["compiled_debug"] = bool(k.metadata.debug)
        res["result"] = f"returned, y = {y.item()}"
    except Exception as e:
        res["result"] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:120]}"
print(json.dumps(res))
'''


def main():
    import triton  # only for the version line
    import torch
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    work = os.path.dirname(os.path.abspath(__file__))
    for case in ("D0", "D1", "D2", "D3", "A0", "A1", "A2", "E0", "E1", "S0", "S1"):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
            env["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
            env["LR3A_TMP"] = tmp
            script = os.path.join(tmp, "case.py")
            open(script, "w").write(CASE)
            p = subprocess.run([sys.executable, "-I", script, case], env=env, cwd=tmp,
                               capture_output=True, text=True, timeout=600)
            out = p.stdout.strip().splitlines()
            if p.returncode == 0 and out:
                print(out[-1])
            else:
                tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-3:]
                print(json.dumps({"case": case, "exit": p.returncode, "stderr_tail": tail,
                                  "stdout_tail": out[-2:]}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
