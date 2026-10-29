"""Runtime `if` on a name that holds a compile-time value before the `if` (lr-8b, round 8).

Each case: 8 programs, program p takes the branch iff p % 2 == 1 (a runtime condition).
The source's meaning (Python's, and SEMANTICS.md "A runtime if") stores the taken branch's
value; the script prints what each program stored, and the front end's error if any.
"""
import os, subprocess, sys
import torch
import triton
import triton.language as tl


@triton.jit
def k_a(out):                      # constexpr live-in, assigned in the then branch only
    pid = tl.program_id(0)
    x = 0
    if pid % 2 == 1:
        x = 1
    tl.store(out + pid, x)


@triton.jit
def k_b(out):                      # constexpr live-in, different constexprs in both branches
    pid = tl.program_id(0)
    x = 0
    if pid % 2 == 1:
        x = 1
    else:
        x = 2
    tl.store(out + pid, x)


@triton.jit
def k_c(out, v_ptr):               # constexpr live-in, a tensor assigned in the then branch
    pid = tl.program_id(0)
    x = 0
    if pid % 2 == 1:
        x = tl.load(v_ptr + pid)
    tl.store(out + pid, x)


@triton.jit
def k_d(out, v_ptr):               # constexpr live-in, tensors of one type in both branches
    pid = tl.program_id(0)
    x = 0
    if pid % 2 == 1:
        x = tl.load(v_ptr + pid)
    else:
        x = tl.load(v_ptr + pid) + 100
    tl.store(out + pid, x)


@triton.jit
def k_e(out):                      # new name, different constexprs in the two branches
    pid = tl.program_id(0)
    if pid % 2 == 1:
        y = 5
    else:
        y = 6
    tl.store(out + pid, y)


@triton.jit
def k_f(out, FLAG: tl.constexpr):  # constexpr live-in changed inside a runtime if, read by a compile-time if
    pid = tl.program_id(0)
    use = False
    if pid % 2 == 1:
        use = True
    if use:
        tl.store(out + pid, 9)
    else:
        tl.store(out + pid, 3)


@triton.jit
def k_g(out):                      # the same, conditional expression form for reference
    pid = tl.program_id(0)
    x = 1 if pid % 2 == 1 else 0
    tl.store(out + pid, x)


CASES = {"A": k_a, "B": k_b, "C": k_c, "D": k_d, "E": k_e, "F": k_f, "G": k_g}


def run(name):
    out = torch.full((8,), -777, device="cuda", dtype=torch.int32)
    v = torch.arange(10, 18, device="cuda", dtype=torch.int32)
    fn = CASES[name]
    try:
        if name in ("C", "D"):
            fn[(8,)](out, v)
        elif name == "F":
            fn[(8,)](out, FLAG=True)
        else:
            fn[(8,)](out)
        torch.cuda.synchronize()
        print(name, "stored", out.cpu().tolist())
    except Exception as e:
        print(name, "error", type(e).__name__, str(e).splitlines()[-1][:200] if str(e) else "")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("triton", triton.__version__, torch.cuda.get_device_name(0))
        print("expected (source meaning): A [0,1,0,1,...]  B [2,1,2,1,...]  C [0,11,0,13,...]  D [110,11,112,13,...]  E [6,5,6,5,...]  F [3,9,3,9,...]  G [0,1,0,1,...]")
        for n in CASES:
            p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), n], capture_output=True, text=True)
            sys.stdout.write(p.stdout or (n + " no output; stderr tail: " + p.stderr[-300:] + "\n"))
