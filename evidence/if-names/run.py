"""Names after a runtime `if`: which ones the front end accepts.

Library evidence for library issue L61 (round 11). `visit_then_else_blocks` and `visit_if_top_level`
(`primitives/triton-3.8.0-code-generator.py` lines 834-966) merge, after a runtime `if`, the names
defined before it that a branch assigns, and the names new to the `if` that both branches define; a
name new to the `if` that only one branch defines is left out of the scope after it. Each case runs
on GPU 0 in its own process, with a condition `flag > 0` on a runtime argument (`flag = 2` or `-2`,
not 1, which the launcher would make a constant), 4 programs.

  N1  y defined in the then branch only and used there only (stored in the branch)
  N2  y defined in the then branch only and used after the if
  N3  y = 1 before; y = y + 10 in the then branch only (the same dtype); y stored after
  N4  y = 1 before; y = 1.5 in both branches (another dtype in both)
  N5  y new, defined in both branches with one dtype; y stored after
  N6  y new, int32 in one branch and float32 in the other

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/if-names/run.py  (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k(out_ptr, flag, CASE: tl.constexpr):
    pid = tl.program_id(0)
    if CASE == 1:
        if flag > 0:
            y = pid + 100
            tl.store(out_ptr + pid, y)
    elif CASE == 2:
        if flag > 0:
            y = pid + 100
        tl.store(out_ptr + pid, y)
    elif CASE == 3:
        y = pid + 1
        if flag > 0:
            y = y + 10
        tl.store(out_ptr + pid, y)
    elif CASE == 4:
        y = pid + 1
        if flag > 0:
            y = 1.5
        else:
            y = 2.5
        tl.store(out_ptr + pid, y)
    elif CASE == 5:
        if flag > 0:
            y = pid + 100
        else:
            y = pid + 200
        tl.store(out_ptr + pid, y)
    else:
        if flag > 0:
            y = pid + 100
        else:
            y = pid + 2.5
        tl.store(out_ptr + pid, y)


CASES = {
    "N1": "a name only the then branch defines, used only there",
    "N2": "a name only the then branch defines, used after the if",
    "N3": "a name defined before, reassigned (same dtype) in one branch",
    "N4": "a name defined before (int32), reassigned to a float in both branches",
    "N5": "a new name defined in both branches, one dtype",
    "N6": "a new name, int32 in one branch and float32 in the other",
}


def run_case(name):
    case = int(name[1:])
    res = []
    for flag in (2, -2):
        out = torch.full((4,), -1, device="cuda", dtype=torch.int32)
        k[(4,)](out, flag, CASE=case)
        torch.cuda.synchronize()
        res.append(f"flag {flag}: {out.tolist()}")
    return "ran; " + "; ".join(res)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = str(e).strip().splitlines()
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1] if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, desc in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else "process failed: " + p.stderr.strip()[-160:]
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()
