"""lr-18b, L79: the header lines the map check takes for an unused function (`def_headers`, `module_lines`), on
forms lr-16a's sources do not have, and what Python 3.13 evaluates when it runs such a `def` at import.

Part A: `def_headers` and `module_lines` on a source whose unused functions have a multi-line decorator, a
default on a continuation line, a return annotation, a one-line body, an `async def`, a `def` inside a
module-level `if`, and a `def` inside a module-level `try`.
Part B: Python itself (no Triton): which of a `def`'s decorator, default, annotation and return annotation rebind a
module-level name at import, with a `:=` in each.

Run: `python3 -I headers_probe.py > headers_probe.out` (Python 3.13, TVA's Triton environment's version).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = HERE
while not os.path.isfile(os.path.join(TVA, "tools", "tvr.py")):
    TVA = os.path.dirname(TVA)
sys.path.insert(0, os.path.join(TVA, "tools"))
sys.dont_write_bytecode = True
import tvr  # noqa: E402

SRC = '''import triton
import triton.language as tl


@triton.jit
def k(x_ptr):
    tl.store(x_ptr, 1)


@deco(
    a=(k := None),
)
def u1(
    x,
    y=(k := None),
) -> (k := None):
    return x


def u2(z=(k := None)): return z


async def u3(w=(k := None)):
    return w


if True:
    def u4(v=(k := None)):
        return v

try:
    def u5(t=(k := None)):
        return t
except Exception:
    pass
'''

print(f"Python {sys.version.split()[0]}; tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}\n")
print("# Part A: the headers of unused functions\n")
for i, l in enumerate(SRC.split("\n"), 1):
    if l.strip():
        print(f"    {i:2d}  {l}")
unused = ["u1", "u2", "u3", "u4", "u5"]
print(f"\n- def_headers(unused = {unused}): {sorted(tvr.def_headers(SRC, set(unused)))}")
print(f"- module_lines: {tvr.module_lines(SRC, unused)}")
m = {"source": {"ranges": [[1, 7]], "host": [], "unused": unused}}
mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(tvr.code_lines(SRC)[0]) if i <= 7]}
print(f"- map_problems with kernel lines [[1, 7]] and nothing else classed: {tvr.map_problems(m, mm, SRC, 'x')[0]}")

print("\n# Part B: what a def evaluates at import (Python itself)\n")
for label, code in [
    ("decorator", "def deco(f):\n    return f\n@(deco if (n := 1) else deco)\ndef f():\n    pass\n"),
    ("default", "def f(a=(n := 1)):\n    pass\n"),
    ("annotation", "def f(a: (n := 1) = 0):\n    pass\n"),
    ("return annotation", "def f() -> (n := 1):\n    pass\n"),
    ("annotation, with `from __future__ import annotations`", "from __future__ import annotations\ndef f(a: (n := 1) = 0):\n    pass\n"),
    ("body (never called)", "def f():\n    global n\n    n = 1\n"),
]:
    g = {"n": 0}
    try:
        exec(compile(code, "<probe>", "exec"), g)
        print(f"- {label}: after the def, n = {g['n']}")
    except SyntaxError as e:
        print(f"- {label}: SyntaxError ({e.msg})")
