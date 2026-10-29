"""lr-18a, round 18 (library issue L79): which lines the map check takes as an unused function's header.

Calls `def_headers`, `module_lines` and `map_problems` (tools/tvr.py) on small sources, each with a `@triton.jit`
helper `double` that the kernel `k` (lines 5-8, the kernel lines with the imports) calls, and an unused function
whose header could rebind it: a one-line `def`, comments and blank lines between decorators, an `async def`, a
signature over several lines with a docstring body, a `def` inside the `__main__` block, and a decorator that calls
the function it decorates (so its body runs at import, though it is named in `source.unused`).

Usage: python3 -I headers.py > headers.out
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

KERNEL = '''import triton
import triton.language as tl
@triton.jit
def double(x): return x * 2
@triton.jit
def k(x_ptr):
    x = tl.load(x_ptr)
    tl.store(x_ptr, double(x))
@triton.jit
def triple(x): return x * 3
'''
TAILS = {
    "a one-line def": "def unused(f=(double := triple)): return f\n",
    "comments and a blank line between its decorators": "@staticmethod\n# a comment\n\n@staticmethod\ndef unused(\n"
                                                         "    f=(double := triple),\n):\n    return f\n",
    "an async def": "async def unused(f=(double := triple)):\n    return f\n",
    "a signature over three lines and a docstring": 'def unused(a,\n        f=(double := triple),\n        b=1):\n'
                                                    '    """A docstring."""\n    return f\n',
    "a def inside the `__main__` block": 'if __name__ == "__main__":\n    def unused(f=(double := triple)):\n'
                                         '        return f\n    print(1)\n',
    "a decorator that calls the function (its body runs at import)":
        "def run_now(f):\n    f()\n    return f\n@run_now\ndef unused():\n    global double\n    double = triple\n",
}


def show(label, value):
    print(f"- {label}: {value}")


print(f"headers.py (lr-18a): tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}; "
      f"Python {sys.version.split()[0]}\n")
for label, tail in TAILS.items():
    src = KERNEL + tail
    n0 = KERNEL.count("\n")
    print(f"## {label} (lines {n0 + 1}–{src.count(chr(10))})\n")
    unused = ["triple", "unused"]
    heads = tvr.def_headers(src, set(unused))
    show("def_headers", sorted(heads) if heads is not None else heads)
    show("module_lines", tvr.module_lines(src, unused))
    code = tvr.code_lines(src)[0]
    host = [i for i in range(n0 + 1, src.count("\n") + 1)]
    others = sorted(i for i in host if i not in (heads or set()))
    funcs = [(n, a, b) for n, a, b in tvr.source_functions(src) if n not in ("double", "k", "triple", "unused")]
    host_fns = [[a, b] for _, a, b in funcs]
    for what, ranges, hostr in (
            ("the unused function's lines in no class but source.unused", [[1, 8]], host_fns + [[i, i] for i in others]),
            ("its header, and `triple`'s (lines 9-10), in the host lines too", [[1, 8]],
             host_fns + [[9, 10]] + [[i, i] for i in host])):
        m = {"source": {"ranges": ranges, "host": hostr, "unused": unused}}
        mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if i <= 8]}
        show(what, tvr.map_problems(m, mm, src, "x")[0] or "passes")
    print()
