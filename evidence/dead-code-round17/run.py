"""Code the front end visits after a `return`: compiled, never run.

Library evidence for library issue L89 (round 17). `primitives/triton-3.8.0-code-generator.py`:
`visit_compound_statement` (508-517) stops only at the statements after a `return` in its own
statement list; `visit_Return` (544-554) records the returned value and opens a new block that no
program reaches, into which the front end visits whatever comes after the statement holding the
`return`; `visit_if_top_level` (893-929) joins a runtime `if`'s two branches in a block that, when
both branches return, only unreachable blocks reach; `decide_return_type` (556-585) gives a helper
the common type of every `return` it visited. lr-15b's
`docs/library-review/round-15/lr-15b/work/dead_after_return.py` (D1-D9) showed the compile-time `if`
and `tl.static_range` cases; these are the others.

  E1  offs = tl.where(i < m, 16 + i, 0); a load through offs; a runtime if whose two branches
      return; then the pair hint (16, 16) on offs (false past lane m = 100)
  E2  the same without the hint (control)
  E3  a compile-time `if FLAG: return` (FLAG true), then tl.atomic_add(c, 1): the atomic's count
  E4  a helper `if FLAG: return -7` then `return 0`, called with FLAG true: h() // 2 stored
  E5  a helper `if FLAG: return -7` then `return -7` (the same value): h() // 2 stored
  E6  a helper `return -7` alone (control): h() // 2 stored
  E7  a runtime if whose two branches return, then tl.static_assert(False)

int8 data, 256 elements, one warp, unless the case says otherwise. Each case runs in its own process
with its own empty Triton cache; for E1 and E2 the loads of the PTX are listed with the result.
Run from TVA's root: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/dead-code-round17/run.py
"""
import json, os, re, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_rt(x_ptr, out_ptr, m, n, HINT: tl.constexpr, ASSERT: tl.constexpr):
    i = tl.arange(0, 256)
    offs = tl.where(i < m, 16 + i, 0)
    tl.store(out_ptr + i, tl.load(x_ptr + offs))
    if n > 0:
        return
    else:
        return
    if HINT:
        offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
    if ASSERT:
        tl.static_assert(False, "visited")


@triton.jit
def k_atomic(c_ptr, FLAG: tl.constexpr):
    if FLAG:
        return
    tl.atomic_add(c_ptr, 1)


@triton.jit
def _h_diff(FLAG: tl.constexpr):
    if FLAG:
        return -7
    return 0


@triton.jit
def _h_same(FLAG: tl.constexpr):
    if FLAG:
        return -7
    return -7


@triton.jit
def _h_one():
    return -7


@triton.jit
def k_ret(out_ptr, CASE: tl.constexpr):
    if CASE == 4:
        tl.store(out_ptr, _h_diff(True) // 2)
    elif CASE == 5:
        tl.store(out_ptr, _h_same(True) // 2)
    else:
        tl.store(out_ptr, _h_one() // 2)


CASES = {
    "E1": "runtime if, both branches return; then the pair hint on offs",
    "E2": "the same without the hint (control)",
    "E3": "if FLAG: return (FLAG true); then tl.atomic_add(c, 1), 8 programs",
    "E4": "helper: if FLAG: return -7; return 0 (FLAG true); h() // 2",
    "E5": "helper: if FLAG: return -7; return -7 (FLAG true); h() // 2",
    "E6": "helper: return -7 (control); h() // 2",
    "E7": "runtime if, both branches return; then tl.static_assert(False)",
}


def run_case(name):
    dev = "cuda"
    if name in ("E1", "E2", "E7"):
        x = torch.arange(1024, device=dev, dtype=torch.int32).to(torch.int8)
        out = torch.zeros(256, device=dev, dtype=torch.int8)
        m = 100
        ck = k_rt[(1,)](x, out, m, 1, HINT=(name == "E1"), ASSERT=(name == "E7"), num_warps=1)
        torch.cuda.synchronize()
        loads = sorted(set(re.findall(r"ld\.global(?:\.[a-z0-9]+)*", ck.asm["ptx"])))
        i = torch.arange(256, device=dev)
        expect = x[torch.where(i < m, 16 + i, 0)]
        bad = int((out != expect).sum().item())
        return f"loads {loads}; values {'correct' if bad == 0 else f'WRONG in {bad} of 256'}"
    if name == "E3":
        c = torch.zeros(1, device=dev, dtype=torch.int32)
        k_atomic[(8,)](c, FLAG=True)
        torch.cuda.synchronize()
        return f"c = {int(c.item())} after 8 programs"
    out = torch.zeros(1, device=dev, dtype=torch.int32)
    k_ret[(1,)](out, CASE=int(name[1]))
    torch.cuda.synchronize()
    return f"stored {int(out.item())}"


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
