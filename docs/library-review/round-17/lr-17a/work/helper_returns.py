"""lr-17a: what a helper with a `return` inside a runtime `if` returns (survey, area `semantics`).

The supported subset admits, in a helper, `return e` inside a runtime `if` and statements after that
`if` when only one branch returns (SEMANTICS.md "The supported subset", "`return`"). SEMANTICS says
that such a `return` ends the inlined helper for the programs that take it, but not what value the
call then has when the helper's `return`s give different kinds of values. The front end gives the
function one return type, the common type of every `return` it visited (`decide_return_type`,
`primitives/triton-3.8.0-code-generator.py:556-585`), and converts each returned value to it
(`cast_to`, 587-604): two equal compile-time values stay a compile-time value; otherwise a Python
number becomes a scalar tensor of its own type (`to_tensor_type`), and the call's result is a tensor.
A path that reaches the end of the helper without a `return` gets a poison value
(`handle_returns`, 606-626: `builder.ret` of `create_poison`), or, when the return type is a
compile-time value, that value.

  R1  h(c): `if c > 0: return -7`, then `return -9`; the kernel stores `h(c) // 2` (c = 3, c = -1,
      runtime int32 scalars)
      (a compile-time -7 // 2 is Python's -4, a tensor's is -3; -9: Python -5, tensor -4)
  R2  h(c): `if c > 0: return -7`, then `return -7` (the same value)
  R3  h(c): `if c > 0: return 5`, and nothing after (the other path falls off the end); the kernel
      stores h(c) and checks `tl.static_assert(h(c) == 5)` (passes only on a compile-time 5)
  R4  h(c, x): `if c > 0: return x` (x a loaded int32 block), nothing after; the kernel stores
      h(c, x) + 1 for c = -1
  R5  h(c): `if c > 0: return 0.1`, then `return 0.2`; the kernel stores `xb * h(c)` for a
      bfloat16 xb into float32: a Python 0.1 would compute in bfloat16 (SEMANTICS "Constants"),
      a float32 scalar tensor in float32

Each case runs in its own process with its own Triton cache. GPU 0.
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-17/lr-17a/work/helper_returns.py
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def h1(c):
    if c > 0:
        return -7
    return -9


@triton.jit
def h2(c):
    if c > 0:
        return -7
    return -7


@triton.jit
def h3(c):
    if c > 0:
        return 5


@triton.jit
def h4(c, x):
    if c > 0:
        return x


@triton.jit
def h5(c):
    if c > 0:
        return 0.1
    return 0.2


@triton.jit
def k_int(out_ptr, c, CASE: tl.constexpr):
    if CASE == 1:
        tl.store(out_ptr, h1(c) // 2)
    elif CASE == 2:
        tl.store(out_ptr, h2(c) // 2)
    else:
        tl.static_assert(h3(c) == 5)
        tl.store(out_ptr, h3(c))


@triton.jit
def k_blk(x_ptr, out_ptr, c):
    i = tl.arange(0, 64)
    x = tl.load(x_ptr + i)
    tl.store(out_ptr + i, h4(c, x) + 1)


@triton.jit
def k_flt(x_ptr, out_ptr, c):
    i = tl.arange(0, 64)
    xb = tl.load(x_ptr + i)
    tl.store(out_ptr + i, (xb * h5(c)).to(tl.float32))


CASES = {
    "R1": "if c > 0: return -7; return -9; store h(c) // 2",
    "R2": "if c > 0: return -7; return -7; store h(c) // 2",
    "R3": "if c > 0: return 5 (falls off otherwise); static_assert(h(c) == 5); store h(c)",
    "R4": "if c > 0: return x (falls off otherwise); store h(c, x) + 1 with c = -1",
    "R5": "if c > 0: return 0.1; return 0.2; store xb * h(c) for bfloat16 xb",
}


def run_case(name):
    dev = "cuda"
    if name in ("R1", "R2", "R3"):
        res = {}
        # c = 3 takes the branch, c = -1 does not; neither is the specialised value 1
        for cc in (3, -1):
            out = torch.full((1,), 99, device=dev, dtype=torch.int32)
            k_int[(1,)](out, cc, CASE=int(name[1:]))
            torch.cuda.synchronize()
            res[f"c={cc}"] = int(out.item())
        return res
    if name == "R4":
        x = torch.arange(64, device=dev, dtype=torch.int32) * 10
        out = torch.full((64,), 99, device=dev, dtype=torch.int32)
        ck = k_blk[(1,)](x, out, -1)
        torch.cuda.synchronize()
        o = out.cpu().tolist()
        return {"c=-1 first 6 stored": o[:6], "equals x + 1": o == (x.cpu() + 1).tolist()}
    if name == "R5":
        xb = (torch.arange(64, device=dev, dtype=torch.float32) * 0.37 + 1.0).to(torch.bfloat16)
        res = {}
        for cc in (3, -1):
            out = torch.zeros(64, device=dev, dtype=torch.float32)
            k_flt[(1,)](xb, out, cc)
            torch.cuda.synchronize()
            v = 0.1 if cc > 0 else 0.2
            f32 = (xb.float() * torch.tensor(v, dtype=torch.float32, device=dev)).cpu()
            b16 = (xb * torch.tensor(v, dtype=torch.bfloat16, device=dev)).float().cpu()
            o = out.cpu()
            res[f"c={cc}"] = {"as float32 product": bool(torch.equal(o, f32)),
                              "as bfloat16 product": bool(torch.equal(o, b16))}
        return res


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
