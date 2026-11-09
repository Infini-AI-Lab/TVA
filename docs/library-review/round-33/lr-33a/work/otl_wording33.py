"""lr-33a, library review round 33, L149: loops whose source does not have the excluded pattern as docs/exclusions.md
and docs/SEMANTICS.md ("Float + and -") word it, but which Triton 3.8.0's OptimizeThreadLocality pass rewrites (on the
IR after inlining, constant folding, dead-code removal and constexpr specialisation).

Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>

Data (two trips, [64, 64] tiles, 4 warps): trip 0 has 2^24 in column 0 and 1 in column 63, trip 1 has -2^24 and 1;
each trip's row sum is 2^24 and -2^24 + 1 along any tree, so the source gives 1.0 (or the value named); folding each
thread's elements across the trips first gives 2.0. "rewritten" = the TTGIR holds a tt.reshape to rank 3.
"""
import re

import torch
import triton
import triton.language as tl

DEV = "cuda"
M, N, T = 64, 64, 2
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


def rewritten(h):
    return bool(re.search(r"tt\.reshape [^\n]*-> tensor<\d+x\d+x\d+x", h.asm["ttgir"]))


@triton.jit
def rsum_helper(x):
    return tl.sum(x, axis=1)


@triton.jit
def upd_helper(acc, x):
    return acc + tl.sum(x, axis=1)


@triton.jit
def k(x_ptr, o_ptr, d_ptr, iv, fv, FORM: tl.constexpr, FLAG: tl.constexpr, T: tl.constexpr, M: tl.constexpr,
      N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    if FORM == 7:
        acc = tl.zeros([M], dtype=tl.float32) + iv  # iv a runtime int argument
    elif FORM == 8:
        acc = tl.zeros([M], dtype=tl.float32) + fv  # fv a runtime float argument
    elif FORM == 12:
        acc = tl.zeros([M], dtype=tl.float32) * 1.0
    else:
        acc = tl.zeros([M], dtype=tl.float32)
    for t in range(T):
        x = tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :])
        if FORM == 0:  # the update's other operand converted to its own dtype after the reduction
            acc = acc + tl.sum(x, axis=1).to(tl.float32)
        elif FORM == 1:  # the reduction times 1.0 before the update
            acc = acc + tl.sum(x, axis=1) * 1.0
        elif FORM == 2:  # the update's result times 1.0
            acc = (acc + tl.sum(x, axis=1)) * 1.0
        elif FORM == 3:  # the reduction under a compile-time `if` of the body
            if FLAG:
                acc = acc + tl.sum(x, axis=1)
        elif FORM == 4:  # the reduction inside a one-trip tl.static_range nested in the for
            for _ in tl.static_range(1):
                acc = acc + tl.sum(x, axis=1)
        elif FORM == 5:  # the reduction's result has a second, dead use
            s = tl.sum(x, axis=1)
            acc = acc + s
            dead = s * 3.0  # noqa: F841
        elif FORM == 6:  # reduction and update in a helper
            acc = upd_helper(acc, x)
        elif FORM == 7 or FORM == 8 or FORM == 12:
            acc = acc + tl.sum(x, axis=1)
        elif FORM == 9:  # the reduction in a helper
            acc = acc + rsum_helper(x)
        elif FORM == 10:  # the update's result has a second, dead use
            acc = acc + tl.sum(x, axis=1)
            dead2 = acc * 2.0  # noqa: F841
        elif FORM == 11:  # the update's result has a second use under a compile-time false `if`
            acc = acc + tl.sum(x, axis=1)
            if FLAG:
                tl.store(d_ptr + rm, acc)
        elif FORM == 13:  # the reduction inside a two-trip tl.static_range nested in the for
            for _ in tl.static_range(2):
                acc = acc + tl.sum(x, axis=1) * 0.5
        elif FORM == 14:  # -(-sum)
            acc = acc + (-(-tl.sum(x, axis=1)))
        elif FORM == 15:  # sum of tl.load(p) after a no-op broadcast_to its own shape
            acc = acc + tl.sum(tl.broadcast_to(x, (M, N)), axis=1)
        elif FORM == 16:  # a runtime `if` on the trip whose condition is always true (control)
            if t >= 0:
                acc = acc + tl.sum(x, axis=1)
    tl.store(o_ptr + rm, acc)


FORMS = {
    0: ("acc + tl.sum(x, 1).to(tl.float32)", 0),
    1: ("acc + tl.sum(x, 1) * 1.0", 0),
    2: ("(acc + tl.sum(x, 1)) * 1.0", 0),
    3: ("if FLAG (constexpr True): acc = acc + tl.sum(x, 1)", 1),
    4: ("for _ in tl.static_range(1): acc = acc + tl.sum(x, 1)", 0),
    5: ("s = tl.sum(x, 1); acc = acc + s; dead = s * 3.0 (unused)", 0),
    6: ("acc = upd_helper(acc, x) (reduction and update in a helper)", 0),
    7: ("acc starts at tl.zeros + iv, iv a runtime int argument = 0", 0),
    70: ("acc starts at tl.zeros + iv, iv a runtime int argument = 1", 0),
    8: ("acc starts at tl.zeros + fv, fv a runtime float argument = 0.0", 0),
    9: ("acc = acc + rsum_helper(x) (reduction in a helper)", 0),
    10: ("acc = acc + tl.sum(x, 1); dead2 = acc * 2.0 (unused)", 0),
    11: ("acc = acc + tl.sum(x, 1); if FLAG (constexpr False): tl.store(acc)", 0),
    12: ("acc starts at tl.zeros * 1.0", 0),
    13: ("for _ in tl.static_range(2): acc = acc + tl.sum(x, 1) * 0.5", 0),
    14: ("acc + (-(-tl.sum(x, 1)))", 0),
    15: ("tl.sum(tl.broadcast_to(x, its shape), 1)", 0),
    16: ("if t >= 0 (runtime): acc = acc + tl.sum(x, 1)  [control]", 0),
}


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    x = torch.zeros(T, M, N, device=DEV)
    x[0, :, 0] = 2.0 ** 24
    x[0, :, 63] = 1.0
    x[1, :, 0] = -2.0 ** 24
    x[1, :, 63] = 1.0
    for f, (desc, flag) in FORMS.items():
        form = 7 if f == 70 else f
        iv = 1 if f == 70 else 0
        o = torch.empty(M, device=DEV)
        d = torch.empty(M, device=DEV)
        try:
            h = k[(1,)](x, o, d, iv, 0.0, form, flag, T, M, N, num_warps=4)
        except Exception as e:  # noqa: BLE001
            out(f"F{f:<3} {desc}: not compiled ({type(e).__name__}: {str(e).splitlines()[0][:100]})")
            continue
        torch.cuda.synchronize()
        vals = sorted(set(o.tolist()))
        src = 1.0 if f != 70 else 2.0
        out(f"F{f:<3} {desc}: rewritten {rewritten(h)}; stored {vals}; source gives {src}")
    with open(__file__.rsplit("/", 1)[0] + "/otl_wording33.out", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
