"""lr-15b: code after a `return` reached through a compile-time `if`: visited, not run.

SEMANTICS "`return`": "Code after a `return` at the top level is not compiled." A `return` inside a
compile-time `if` that is taken is visited as part of the branch (`visit_If` visits only that
branch; `visit_Return` opens a new, unreachable block, code_generator.py:544-554), and the
statements after the `if` are still visited (`visit_compound_statement` stops only at a `return` of
its own list, :508-517): they are emitted into the unreachable block. What they emit does not run,
but what visiting them does at compile time stays: a hint marks the operation that defined its
argument (SEMANTICS "Compiler hints"), and a `tl.static_assert` is evaluated.

  D1  load through offs = base + arange; if FLAG: return; then the hint pair (16, 16) on offs   (FLAG True)
  D2  the same with FLAG False (the hint is reached; control)
  D3  the same with no hint at all (control)
  D4  load; return (top level, no if); then the hint pair                         (not visited)
  D5  load; if FLAG: return; then tl.static_assert(False)                          (FLAG True)
  D6  load; return; then tl.static_assert(False)                                    (top level)
  D7  offs = tl.where(i < m, 16 + i, 0); load; if FLAG: return; then the hint pair on offs  (m = 100: false
      past the first 100 lanes, as on tutorial 09's partial tiles)
  D8  the same with no hint (control)
  D9  the same load; for j in tl.static_range(2): if j == 1: the hint pair; if j == 0: return
      (copy 0 returns; copy 1, after it, holds the hint)
int8 data, 256 elements, one warp, base = 4 (so the hint is false); `base` is not specialised
(do_not_specialize), so the hint is the only source of divisibility. For each case: the loads in the
PTX of `warmup`, then a launch: the values, or the error.
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import json, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_where(x_ptr, out_ptr, m, FLAG: tl.constexpr, HINT: tl.constexpr):
    i = tl.arange(0, 256)
    offs = tl.where(i < m, 16 + i, 0)
    tl.store(out_ptr + i, tl.load(x_ptr + offs))
    if FLAG:
        return
    if HINT:
        offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)


@triton.jit
def k_static(x_ptr, out_ptr, m):
    i = tl.arange(0, 256)
    offs = tl.where(i < m, 16 + i, 0)
    tl.store(out_ptr + i, tl.load(x_ptr + offs))
    for j in tl.static_range(2):
        if j == 1:
            offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
        if j == 0:
            return


@triton.jit(do_not_specialize=["base"])
def k(x_ptr, out_ptr, base, FLAG: tl.constexpr, CASE: tl.constexpr):
    offs = base + tl.arange(0, 256)
    tl.store(out_ptr + tl.arange(0, 256), tl.load(x_ptr + offs))
    if CASE == 4:
        return
        offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
    elif CASE == 6:
        return
        tl.static_assert(False, "visited")
    else:
        if FLAG:
            return
        if CASE == 1 or CASE == 2:
            offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
        elif CASE == 5:
            tl.static_assert(False, "visited")


CASES = {"D1": (True, 1), "D2": (False, 2), "D3": (True, 3), "D4": (True, 4), "D5": (True, 5), "D6": (True, 6),
         "D7": (True, 7), "D8": (True, 8), "D9": (True, 9)}


def run_where(name):
    x = torch.arange(1024, device="cuda", dtype=torch.int32).to(torch.int8)
    out = torch.zeros(256, device="cuda", dtype=torch.int8)
    m = 100
    hint = name == "D7"
    if name == "D9":
        ck = k_static.warmup(x, out, m, grid=(1,), num_warps=1)
        k_static[(1,)](x, out, m, num_warps=1)
    else:
        ck = k_where.warmup(x, out, m, FLAG=True, HINT=hint, grid=(1,), num_warps=1)
        k_where[(1,)](x, out, m, FLAG=True, HINT=hint, num_warps=1)
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)*", ck.asm["ptx"])))
    torch.cuda.synchronize()
    i = torch.arange(256, device="cuda")
    expect = x[torch.where(i < m, 16 + i, 0)]
    bad = int((out != expect).sum().item())
    return f"loads {loads}; values {'correct' if bad == 0 else f'WRONG in {bad} of 256'}"


def run_case(name):
    if name in ("D7", "D8", "D9"):
        return run_where(name)
    flag, case = CASES[name]
    x = torch.arange(1024, device="cuda", dtype=torch.int32).to(torch.int8)
    out = torch.zeros(256, device="cuda", dtype=torch.int8)
    base = 4
    try:
        ck = k.warmup(x, out, base, FLAG=flag, CASE=case, grid=(1,), num_warps=1)
    except Exception as e:
        msg = [l for l in str(e).strip().splitlines() if l.strip()]
        return "rejected on the host: " + type(e).__name__ + ": " + (msg[-1].strip() if msg else "")[:160]
    loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)*", ck.asm["ptx"])))
    try:
        k[(1,)](x, out, base, FLAG=flag, CASE=case, num_warps=1)
        torch.cuda.synchronize()
    except Exception as e:
        return f"loads {loads}; launch: {type(e).__name__}: {str(e).strip().splitlines()[0][:120]}"
    expect = x[base:base + 256]
    bad = int((out != expect).sum().item())
    return f"loads {loads}; values {'correct' if bad == 0 else f'WRONG in {bad} of 256'}"


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            res = "error: " + type(e).__name__ + ": " + str(e).strip().splitlines()[-1][:200]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in CASES:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=cache)
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                               capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[len("RESULT "):])
        else:
            err = [l for l in p.stderr.strip().splitlines() if l.strip()]
            res = f"process failed (exit {p.returncode}): " + (err[-1][:200] if err else "(no output)")
        print(f"{name}: {res}")


if __name__ == "__main__":
    main()
