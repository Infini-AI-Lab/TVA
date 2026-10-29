"""Library review round 6 (lr-6b): which source lines the gate's `map` check requires the map to cover.

Run from the TVA root:  python3 -I docs/library-review/round-6/lr-6b/work/map_coverage.py
Output: map_coverage.out.txt next to this file.

The loop below is copied from `check` in tools/tvr.py (the lines from `src_lines = src_text.splitlines()` to
the end of the `for a, b in man["source"]["ranges"]` loop); this script checks that the copy is still the
tool's text, then runs it on a small kernel source with an empty map. Every line it does not list as
uncovered is a line the gate lets the map leave out.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))))

COPY = '''    src_lines = src_text.splitlines()
    missing, in_doc = [], False
    for a, b in man["source"]["ranges"]:
        for i in range(a, b + 1):
            t = src_lines[i - 1].strip() if i <= len(src_lines) else ""
            q = t.count('"""')
            if in_doc or q:
                if q % 2 == 1:
                    in_doc = not in_doc
                continue
            if t and not t.startswith("#") and i not in covered:
                missing.append(i)
'''

tool = open(os.path.join(ROOT, "tools", "tvr.py")).read()
print("copy matches tools/tvr.py:", COPY in tool)

SRC = '''@triton.jit
def k(x_ptr, c_ptr, n, BLOCK: tl.constexpr):
    """A docstring."""
    pid = tl.program_id(0)
    tl.device_print("""v""", tl.atomic_add(c_ptr, 1))
    x = tl.load(x_ptr + pid) + tl.atomic_add(c_ptr, 2)  ; s = """"""
    tl.store(x_ptr + pid, x)
'''


def run(src_text, man, covered):
    ns = {"src_text": src_text, "man": man, "covered": covered}
    exec(compile("def f():\n" + COPY + "    return missing\n", "copy", "exec"), ns)
    return ns["f"]()


lines = SRC.splitlines()
missing = run(SRC, {"source": {"ranges": [[1, len(lines)]]}}, set())
for i, l in enumerate(lines, 1):
    print(f"{i:2} {'must be covered' if i in missing else 'exempt         '}  {l}")
