"""Runtime `if` that reassigns a name holding a compile-time value (lr-8b, round 8).

`visit_then_else_blocks` (code-generator.py:855-874) merges only live-in names that hold Triton
values; a live-in `tl.constexpr` (a parameter, or a local annotated `tl.constexpr`) is skipped,
and `enter_sub_region` restores the scope after the `if`. Program p takes the branch iff p is
odd (a runtime condition). Python's meaning: the odd programs see the new value.
"""
import os, subprocess, sys
import torch
import triton
import triton.language as tl


@triton.jit
def k_param(out, B: tl.constexpr):         # a constexpr parameter reassigned in a runtime branch
    pid = tl.program_id(0)
    if pid % 2 == 1:
        B = 7
    tl.store(out + pid, B)


@triton.jit
def k_param_else(out, B: tl.constexpr):    # the same, both branches
    pid = tl.program_id(0)
    if pid % 2 == 1:
        B = 7
    else:
        B = 5
    tl.store(out + pid, B)


@triton.jit
def k_local(out):                          # an annotated constexpr local reassigned
    pid = tl.program_id(0)
    x: tl.constexpr = 3
    if pid % 2 == 1:
        x = 7
    tl.store(out + pid, x)


@triton.jit
def k_loop(out, B: tl.constexpr):          # the same in a loop body (control: _find_carries asserts)
    pid = tl.program_id(0)
    for i in range(0, pid % 2):
        B = 7
    tl.store(out + pid, B)


CASES = {"param": k_param, "param_else": k_param_else, "local": k_local, "loop": k_loop}


def run(name):
    out = torch.full((8,), -777, device="cuda", dtype=torch.int32)
    try:
        if name == "local":
            CASES[name][(8,)](out)
        else:
            CASES[name][(8,)](out, B=3)
        torch.cuda.synchronize()
        print(name, "stored", out.cpu().tolist())
    except Exception as e:
        msg = [l for l in str(e).splitlines() if l.strip()]
        print(name, "rejected:", type(e).__name__, (msg[-1] if msg else "")[:200])


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("triton", triton.__version__, torch.cuda.get_device_name(0))
        print("Python's meaning: param [3,7,3,7,...]; param_else [5,7,5,7,...]; local [3,7,3,7,...]; loop [3,7,3,7,...]")
        for n in CASES:
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), n], capture_output=True, text=True)
            sys.stdout.write(p.stdout or (n + " no output: " + p.stderr[-300:] + "\n"))
