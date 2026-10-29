"""A name defined in one branch of a runtime `if` only (SEMANTICS "A runtime if", L28).

lr-8a, round 8. SEMANTICS: "After the if, a name has the taken branch's value when it was defined
before the if or is defined in both branches, with the same dtype and shape; otherwise the kernel
is rejected on the host (R5)." Is a kernel whose name is defined in one branch only, and not used
after the `if`, rejected? And one whose live-in name is redefined, in both branches, with another
dtype than before the `if`?
Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>   (from TVA's root)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_one_branch(x_ptr, out_ptr, n):
    pid = tl.program_id(0)
    if pid % 2 == 0:
        t = tl.load(x_ptr + pid) * 2      # `t` is defined in this branch only, never used after
        tl.store(out_ptr + pid, t)
    else:
        tl.store(out_ptr + pid, -1.0)
    tl.store(out_ptr + n + pid, 5.0)


@triton.jit
def k_both_other_dtype(x_ptr, out_ptr):
    pid = tl.program_id(0)
    y = pid                               # int32 before the `if`
    if pid % 2 == 0:
        y = tl.load(x_ptr + pid)          # float32 in both branches
    else:
        y = tl.load(x_ptr + pid) + 1.0
    tl.store(out_ptr + pid, y)


def main():
    n = 8
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    out = torch.zeros(2 * n, device="cuda")
    try:
        k_one_branch[(n,)](x, out, n)
        torch.cuda.synchronize()
        ref = [float(2 * i) if i % 2 == 0 else -1.0 for i in range(n)] + [5.0] * n
        print("one-branch name, unused after: ran;", "correct" if out.cpu().tolist() == ref else "WRONG")
    except Exception as e:
        print("one-branch name, unused after: rejected:", type(e).__name__, str(e).splitlines()[0][:120])
    out = torch.zeros(n, device="cuda")
    try:
        k_both_other_dtype[(n,)](x, out)
        torch.cuda.synchronize()
        print("live-in redefined in both branches with another dtype: ran;", out.cpu().tolist())
    except Exception as e:
        msg = [l for l in str(e).splitlines() if l.strip()]
        print("live-in redefined in both branches with another dtype: rejected:", type(e).__name__,
              (msg[-1] if msg else "")[:160])
    print("triton", triton.__version__, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
