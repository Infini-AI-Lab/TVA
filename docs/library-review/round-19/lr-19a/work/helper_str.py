"""lr-19a: two string constants that `mangle_fn` writes alike. `mangle_fn`
(primitives/triton-3.8.0-code-generator.py:32-41) replaces `'` by `_sq_` and `[`, `]` by `_` in the
`repr` of each constexpr argument, so `'x[y'` and `'x_y'` both become `_sq_x_y_sq_`: one helper
function for both calls, and the second call runs the first's body.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I helper_str.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def pick(p, MODE):
    if MODE == "x_y":
        tl.store(p + tl.arange(0, 1), tl.full([1], 1, tl.int32))
    else:
        tl.store(p + tl.arange(0, 1), tl.full([1], 2, tl.int32))


@triton.jit
def k_a(o):
    pick(o, "x[y")
    pick(o + 1, "x_y")


@triton.jit
def k_b(o):
    pick(o, "x_y")
    pick(o + 1, "x[y")


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name, k, want in (("S1 pick(o, 'x[y'); pick(o+1, 'x_y')", k_a, [2, 1]),
                          ("S2 pick(o, 'x_y'); pick(o+1, 'x[y')", k_b, [1, 2])):
        o = torch.zeros(2, dtype=torch.int32, device="cuda")
        k[(1,)](o)
        torch.cuda.synchronize()
        got = o.tolist()
        print(f"{name}: stored {got}, Python's meaning {want}: {'same' if got == want else 'DIFFERS'}")


if __name__ == "__main__":
    main()
