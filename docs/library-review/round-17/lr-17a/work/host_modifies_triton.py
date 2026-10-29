"""lr-17a: host code that modifies Triton changes the compiled kernel (survey: what a launch means).

`Compiler.Faithful`'s scope (docs/rules/compiler-faithful.md, "Scope") lists knobs, hooks, options,
heuristics, descriptors, precompiled kernels, the target and graph capture, but says nothing of host
or module code that assigns to Triton's own objects. Two such routes, each with no knob, hook or
option:

  U1  `k._unsafe_update_src(src)` (`primitives/triton-3.8.0-jit.py:561-569`, a public method of
      the kernel's `JITFunction`): the kernel's source `x + 1` replaced by `x + 2` before the launch
  U2  `tl.arange = fake` on the `triton.language` module, `fake` a `@triton.jit` function returning
      `tl.zeros`: the kernel's `tl.arange(0, 64)` then means `fake`'s body
  U0  the same kernel with neither (control)

The kernel stores `x + 1` at `out + tl.arange(0, 64)`. Each case runs in its own process with its
own Triton cache. GPU 0. Run from TVA's root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-17/lr-17a/work/host_modifies_triton.py
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, out_ptr):
    offs = tl.arange(0, 64)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)


@triton.jit
def fake(a: tl.constexpr, b: tl.constexpr):
    return tl.zeros([b], tl.int32)


SRC2 = '''def k(x_ptr, out_ptr):
    offs = tl.arange(0, 64)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 2)
'''


def run_case(name):
    x = torch.arange(64, device="cuda", dtype=torch.int32) * 10
    out = torch.full((64,), -1, device="cuda", dtype=torch.int32)
    if name == "U1":
        k._unsafe_update_src(SRC2)
    if name == "U2":
        tl.arange = fake
    k[(1,)](x, out)
    torch.cuda.synchronize()
    o = out.cpu().tolist()
    return {"first 4 stored": o[:4], "equals x + 1": o == (x.cpu() + 1).tolist()}


CASES = {"U0": "control", "U1": "k._unsafe_update_src(source with x + 2)", "U2": "tl.arange = fake (returns tl.zeros)"}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = "process failed: " + (err[-1][:160] if err else "(no output)")
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()
