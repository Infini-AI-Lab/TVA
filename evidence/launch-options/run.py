"""TVA library evidence (library issues L18, L19): which host-side settings change what Triton 3.8.0
compiles, besides the environment variables of `evidence/env-knobs`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/launch-options/run.py
Output: evidence/launch-options/out.txt (README.md says what it shows).

Every case runs in a fresh subprocess with every `TRITON_*` variable removed and a new empty
`TRITON_CACHE_DIR`; it prints the `TRITON_*` variables set when the kernel compiles (none, except
where a knob assignment sets one) and the result.

Part A, settings `Rules.Compiler.Faithful` excludes (each against a control):
- dot: `tl.dot(a, b)` on float32 16x16 without `input_precision`, every `a` element `1 + 2^-12`
  (not a tf32 value), `b` all ones: tf32 gives 16, ieee `16 + 2^-8`.
  D1 the launch keyword `allowed_dot_input_precisions=("ieee",)`; D2 the launch keyword
  `default_dot_input_precision="ieee"`; D3 `triton.knobs.propagate_env = False`, then
  `triton.knobs.language.fp32_default = "ieee"` (no environment variable); D4 the same knob with
  the default `propagate_env` (it sets the environment variable).
- add: `y = x + 1.0` on float32. A1 the launch keyword `ir_override=` the kernel's own PTX with the
  constant `1.0` replaced by `2.0`; A2 `triton.knobs.runtime.add_stages_inspection_hook`, a hook
  that rewrites the PTX stage (`add` becomes `sub`); A3 `triton.knobs.runtime.jit_cache_hook`
  returning true: the call returns and nothing runs (`y` keeps its initial `-7`).
- fma: `y = a * b + c` with `a = b = 1 + 2^-13`, `c = -(1 + 2^-12)`: fused `2^-26`, unfused `0`.
  F1 `triton.knobs.propagate_env = False`, then `triton.knobs.language.default_fp_fusion = False`.
- exp2: `tl.math.exp2(x)` at `x = -130, -126, -149, 0`. E1 the launch keyword
  `enable_reflect_ftz=False`.
- assert: `tl.device_assert(x > 0)` with `x = -1`, no `debug` argument. S1
  `triton.knobs.runtime.debug = True` (a plain attribute, no environment variable).

Part B, the options `Rules.Compiler.Faithful` lets vary (`num_warps`, `num_stages`, `maxnreg`,
`launch_cooperative_grid`): a kernel with exact results (an `int8` `tl.dot` into `int32`, an
integer `tl.sum`, a float32 `x * 2 + 1` with fusion, 64 programs) under each value, compared bit
for bit with the defaults.
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
def fmak(a_ptr, b_ptr, c_ptr, y_ptr):
    tl.store(y_ptr, tl.load(a_ptr) * tl.load(b_ptr) + tl.load(c_ptr))

@triton.jit
def exp2k(x_ptr, y_ptr, n, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.math.exp2(tl.load(x_ptr + o, mask=o < n)), mask=o < n)

@triton.jit
def assertk(x_ptr, y_ptr):
    x = tl.load(x_ptr)
    tl.device_assert(x > 0, "x must be positive")
    tl.store(y_ptr, x + 1.0)

@triton.jit
def exactk(a_ptr, b_ptr, x_ptr, c_ptr, s_ptr, y_ptr):
    pid = tl.program_id(0)
    r = tl.arange(0, 32)
    a = tl.load(a_ptr + pid * 1024 + r[:, None] * 32 + r[None, :])
    b = tl.load(b_ptr + pid * 1024 + r[:, None] * 32 + r[None, :])
    tl.store(c_ptr + pid * 1024 + r[:, None] * 32 + r[None, :], tl.dot(a, b, out_dtype=tl.int32))
    v = tl.load(x_ptr + pid * 1024 + tl.arange(0, 1024))
    tl.store(s_ptr + pid, tl.sum(v.to(tl.int32), axis=0))
    tl.store(y_ptr + pid * 1024 + tl.arange(0, 1024), v * 2.0 + 1.0)

def env_set():
    return sorted(k for k in os.environ if k.startswith("TRITON_") and k != "TRITON_CACHE_DIR")

kw = {}
if case[0] == "D":
    if case == "D1":
        kw["allowed_dot_input_precisions"] = ("ieee",)
    if case == "D2":
        kw["default_dot_input_precision"] = "ieee"
    if case == "D3":
        triton.knobs.propagate_env = False
        triton.knobs.language.fp32_default = "ieee"
    if case == "D4":
        triton.knobs.language.fp32_default = "ieee"
    a = torch.full((16, 16), 1 + 2 ** -12, device="cuda", dtype=torch.float32)
    b = torch.ones((16, 16), device="cuda", dtype=torch.float32)
    c = torch.empty((16, 16), device="cuda", dtype=torch.float32)
    res["env_at_compile"] = env_set()
    try:
        dotk[(1,)](a, b, c, **kw)
        torch.cuda.synchronize()
        res["c00_minus_16_in_2^-8"] = (c[0, 0].item() - 16) * 2 ** 8
    except Exception as e:
        res["result"] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:100]}"
elif case[0] == "A":
    n, B = 1024, 1024
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, -7.0)
    if case == "A1":
        k0 = addk.warmup(x, y, n, B, grid=(1,))
        ptx = k0.asm["ptx"]
        assert "0f3F800000" in ptx, "constant 1.0 not found in PTX"
        path = os.path.join(os.environ["CASE_TMP"], "addk_override.ptx")
        open(path, "w").write(ptx.replace("0f3F800000", "0f40000000"))
        kw["ir_override"] = path
    if case == "A2":
        def hook(*args):
            if not args:
                return "inspect", "inspect-1"
            backend, stages, options, language, capability = args
            orig = stages["ptx"]
            def ptx_stage(src, metadata):
                return orig(src, metadata).replace("add.f32", "sub.f32").replace("add.rn.f32", "sub.rn.f32")
            stages["ptx"] = ptx_stage
        triton.knobs.runtime.add_stages_inspection_hook = hook
    if case == "A3":
        triton.knobs.runtime.jit_cache_hook = lambda *args, **kwargs: True
    res["env_at_compile"] = env_set()
    r = addk[(1,)](x, y, n, B, **kw)
    torch.cuda.synchronize()
    res["call_returned_kernel"] = r is not None
    res["y_minus_x_distinct"] = sorted(set((y - x).tolist()))[:4]
    res["y_is_initial"] = bool((y == -7.0).all())
elif case[0] == "F":
    if case == "F1":
        triton.knobs.propagate_env = False
        triton.knobs.language.default_fp_fusion = False
    a = torch.tensor([1 + 2 ** -13], device="cuda")
    c = torch.tensor([-(1 + 2 ** -12)], device="cuda")
    y = torch.empty(1, device="cuda")
    res["env_at_compile"] = env_set()
    k = fmak[(1,)](a, a, c, y)
    torch.cuda.synchronize()
    res["y_in_2^-26"] = y.item() * 2 ** 26
    res["ptx_fma"] = k.asm["ptx"].count("fma.rn.f32")
elif case[0] == "E":
    if case == "E1":
        kw["enable_reflect_ftz"] = False
    x = torch.tensor([-130.0, -126.0, -149.0, 0.0], device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    res["env_at_compile"] = env_set()
    k = exp2k[(1,)](x, y, 4, 4, **kw)
    torch.cuda.synchronize()
    res["exp2"] = [float.hex(v) for v in y.tolist()]
    res["ptx_ex2"] = sorted(set(l.split()[0] for l in k.asm["ptx"].splitlines() if "ex2" in l))
elif case[0] == "S":
    if case == "S1":
        triton.knobs.runtime.debug = True
    x = torch.tensor([-1.0], device="cuda")
    y = torch.zeros(1, device="cuda")
    res["env_at_compile"] = env_set()
    try:
        k = assertk[(1,)](x, y)
        torch.cuda.synchronize()
        res["result"] = f"returned, y = {y.item()}, compiled debug = {bool(k.metadata.debug)}"
    except Exception as e:
        res["result"] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:80]}"
elif case == "B":
    g = torch.Generator(device="cuda").manual_seed(20261027)
    P = 64
    a = torch.randint(-128, 128, (P * 1024,), device="cuda", generator=g, dtype=torch.int8)
    b = torch.randint(-128, 128, (P * 1024,), device="cuda", generator=g, dtype=torch.int8)
    x = torch.randn(P * 1024, device="cuda", generator=g) * 1e3
    x[::97] = float("nan")
    def run(**opts):
        c = torch.empty(P * 1024, device="cuda", dtype=torch.int32)
        s = torch.empty(P, device="cuda", dtype=torch.int32)
        y = torch.empty(P * 1024, device="cuda")
        exactk[(P,)](a, b, x.to(torch.float32), c, s, y, **opts)
        torch.cuda.synchronize()
        return torch.cat([c.view(torch.int32), s, y.view(torch.int32)])
    base = run()
    res["env_at_compile"] = env_set()
    res["variants"] = {}
    for name, opts in [("num_warps=1", {"num_warps": 1}), ("num_warps=2", {"num_warps": 2}),
                       ("num_warps=8", {"num_warps": 8}), ("num_stages=1", {"num_stages": 1}),
                       ("num_stages=2", {"num_stages": 2}), ("num_stages=4", {"num_stages": 4}),
                       ("maxnreg=64", {"maxnreg": 64}), ("maxnreg=128", {"maxnreg": 128}),
                       ("launch_cooperative_grid=True", {"launch_cooperative_grid": True})]:
        try:
            res["variants"][name] = int((run(**opts) != base).sum())
        except Exception as e:
            res["variants"][name] = f"raised {type(e).__name__}"
print(json.dumps(res))
'''


def main():
    import torch
    import triton
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    cases = ["D0", "D1", "D2", "D3", "D4", "A0", "A1", "A2", "A3", "F0", "F1", "E0", "E1", "S0", "S1", "B"]
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
            env["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
            env["CASE_TMP"] = tmp
            script = os.path.join(tmp, "case.py")
            open(script, "w").write(CASE)
            p = subprocess.run([sys.executable, "-I", script, case], env=env, cwd=tmp,
                               capture_output=True, text=True, timeout=900)
            out = p.stdout.replace(tmp, "<tmp>").strip().splitlines()
            if p.returncode == 0 and out:
                print(out[-1])
            else:
                tail = (p.stderr.replace(tmp, "<tmp>").strip().splitlines() or ["(no stderr)"])[-2:]
                print(json.dumps({"case": case, "exit": p.returncode, "stderr_tail": tail}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
