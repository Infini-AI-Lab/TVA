"""lr-13a's probe of the map's coverage (library issue L50): a JIT function reads values bound at module
level, outside every function, and the map check does not ask for their lines.

Usage (Triton 3.8.0, GPU 0): CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-13/lr-13a/work/globals.py
  > docs/library-review/round-13/lr-13a/work/globals.out

Part 1 runs a kernel on the H200 that reads a module-level `tl.constexpr` global (`SCALE`) and a
module-level dtype (`OUT_T`); Triton's front end resolves both at compile time
(`triton-3.8.0-code-generator.py:387-420`: a constexpr global, a `language.dtype`, a namedtuple, a
module are allowed). Part 2 gives the same source to the gate's `map_problems` with the kernel lines
holding the kernel only: every function is classed, every code line of the ranges is mapped, and the two
module-level lines that decide what the kernel computes lie in no class.
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
SRC = '''import torch
import triton
import triton.language as tl

SCALE = tl.constexpr(3)
OUT_T = tl.float16


@triton.jit
def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, (x * SCALE).to(OUT_T))


def run():
    x = torch.arange(16, dtype=torch.float32, device="cuda") + 0.25
    o = torch.empty(16, dtype=torch.float32, device="cuda")
    k[(1,)](x, o)
    return x, o
'''


def part1():
    import triton
    print(f"# Part 1: Triton {triton.__version__} on {__import__('torch').cuda.get_device_name(0)}, 2026-10-28\n")
    ns = {"__name__": "globals_probe"}
    path = os.path.join(HERE, "globals_src.py")
    open(path, "w").write(SRC)
    sys.path.insert(0, HERE)
    import globals_src as m
    x, o = m.run()
    print(f"- x[:4] = {x[:4].tolist()}; o[:4] = {o[:4].tolist()} (x * SCALE, rounded to OUT_T = float16 and stored as float32)")
    want = (x * 3).half().float()
    print(f"- equal to (x * 3).half().float() on all 16 lanes: {bool((o == want).all())}")
    m.SCALE = triton.language.constexpr(5)
    try:
        m.run()
        print("- after `SCALE = tl.constexpr(5)`: the launch ran")
    except RuntimeError as e:
        print(f"- after `SCALE = tl.constexpr(5)` in the same process: {str(e)[:120]} (the value is part of the "
              "compiled kernel, and checked at each launch)")
    os.remove(path)
    print()


def part2():
    sys.path.insert(0, os.path.join(TVA, "tools"))
    import tvr
    print("# Part 2: the gate's map check on the same source\n")
    code = tvr.code_lines(SRC)[0]
    ranges = [[9, 13]]
    mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if 9 <= i <= 13]}
    man = {"source": {"ranges": ranges, "host": [[16, 20]], "unused": []}}
    print(f"- the source's functions: {tvr.source_functions(SRC)}")
    print(f"- kernel lines {ranges}, host lines [[16, 20]]: map problems {tvr.map_problems(man, mm, SRC, 'x')}")
    print("- lines 5 (`SCALE = tl.constexpr(3)`) and 6 (`OUT_T = tl.float16`) are code lines in no range")
    print()


if __name__ == "__main__":
    part1()
    part2()
