"""Library review round 3 (lr-3b), survey area `transcription`: the drop reason [config] ("a
branch not taken in the covered configuration (a branch on constexprs)") asks nothing of the
branch's condition. The front end evaluates the condition of every `if` (visit_If,
code_generator.py:967-996) and of a conditional expression (visit_IfExp, :998-1049) before it
decides, so an operation inside a condition that is a compile-time bool still runs. Each kernel
below has an atomic in such a condition; a grid of 4 programs; the counter shows whether it ran.

Triton 3.8.0 (~/fpv/tva/.venv), H200 NVL (sm_90), GPU 0. Run from the TVA root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3b/work/config_cond.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_if(cnt_ptr, o_ptr):
    # a constexpr `if`: the condition is the compile-time bool `... is not None`
    if tl.atomic_add(cnt_ptr, 1) is not None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def k_ifexp(cnt_ptr, o_ptr):
    v = 1 if tl.atomic_add(cnt_ptr, 1) is None else 2
    tl.store(o_ptr + tl.program_id(0), v)


@triton.jit
def k_flag(cnt_ptr, o_ptr, FLAG: tl.constexpr):
    # a constexpr `if` whose condition is pure: control
    if FLAG:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


def main():
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    for name, k, kw in (("if (atomic in condition)", k_if, {}), ("conditional expression", k_ifexp, {}),
                        ("control: if FLAG", k_flag, {"FLAG": True})):
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        o = torch.zeros(4, dtype=torch.int32, device="cuda")
        k[(4,)](cnt, o, **kw)
        torch.cuda.synchronize()
        ttir = k.warmup(cnt, o, grid=(4,), **kw).asm["ttir"] if hasattr(k, "warmup") else ""
        print(f"{name}: counter {cnt.item()} (4 = the atomic ran in every program); outputs {o.tolist()}; "
              f"atomics in TTIR: {ttir.count('tt.atomic_rmw')}; scf.if in TTIR: {ttir.count('scf.if')}")


if __name__ == "__main__":
    main()
