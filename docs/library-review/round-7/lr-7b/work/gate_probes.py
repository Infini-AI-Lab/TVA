"""Probes of the round-7 gate functions (L38, L39, L40), on small inputs, with the tree's own tools/tvr.py.
Nothing is written. Lean's own reading of the two Lean probes is in interp.out (`lake env lean`).

Usage: python3 -I docs/library-review/round-7/lr-7b/work/gate_probes.py
"""
import io
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

TQ = '"' * 3


def l39():
    print("## L39: code lines (tvr.code_lines) on further cases")
    cases = {
        "P1 a bare f-string holding an atomic": ["def k(c):", "    f'{tl.atomic_add(c, 1)}'", "    'a bare string'"],
        "P2 a docstring-like string with .format": ["def k(c):", f"    {TQ}doc", f"    {{}}{TQ}.format(tl.atomic_add(c, 1))"],
        "P3 implicit concatenation over lines": ["def k(c):", "    ('a'", "     'b')", "    tl.atomic_add(c, 1)"],
        "P4 a line continuation": ["def k(c):", "    x = 1 + \\", "        tl.atomic_add(c, 1)"],
        "P5 a string statement then code after `;`": ["def k(c):", "    'x'; tl.atomic_add(c, 1)"],
    }
    for name, lines in cases.items():
        code, why = tvr.code_lines("\n".join(lines) + "\n")
        print(f"{name}: code lines {sorted(code) if code is not None else why}")
    print()


def numbering():
    print("## Line numbers: Python's tokenizer and Lean count line breaks; str.splitlines counts more")
    # claim.md quotes source lines with `src_text.splitlines()` and Lean declarations with
    # `open(p).read().splitlines()[a - 1:b]`, by numbers that Python's tokenizer and Lean's ranges give
    for sep_name, sep in (("form feed U+000C", "\x0c"), ("U+001C", "\x1c"), ("line separator U+2028", " "),
                          ("next line U+0085", "\x85")):
        src = "\n".join(["@triton.jit", f"def k(c, x_ptr):  # a comment{sep}with one more character",
                         "    x = tl.load(x_ptr)", "    tl.atomic_add(c, 1)", "    tl.store(x_ptr, x)"]) + "\n"
        code, _ = tvr.code_lines(src)
        toks = {t.start[0]: t.line.rstrip("\n") for t in tokenize.generate_tokens(io.StringIO(src).readline)
                if t.type == tokenize.NAME and t.string == "atomic_add"}
        line = next(iter(toks))
        shown = src.splitlines()[line - 1].strip()
        print(f"{sep_name} in a comment: the atomic is on line {line} for Python (a code line: {line in code}); "
              f"claim.md would show line {line} as {shown!r}")
    lean = "-- a comment with one more character\ndef x : Nat := 1\ndef model : Nat := 2\n"
    print(f"a Lean file with U+2028 in a comment: Lean puts `model` on line 3; "
          f"splitlines()[2] is {lean.splitlines()[2]!r}")
    print()


def l40():
    print("## L40: the library hygiene's reading of interpolated strings (tvr._uncommented)")
    for f in ("InterpStr.lean", "InterpOpen.lean"):
        text = open(os.path.join(HERE, f), encoding="utf-8").read()
        code = tvr._uncommented(text)
        hits = sorted({w for w, p in tvr.LIBRARY_HYGIENE if re.search(p, code, re.M)})
        toks = tvr._TOKEN.findall(code)
        print(f"{f}: library hygiene flags {hits or 'nothing'}; `#eval` left in the code: {'#eval' in code}; "
              f"`open` left: {'open' in toks}")
    print("library files with an interpolated string (s!, m!, f!):",
          [tvr.rel(p) for p in tvr.library_lean_files() if re.search(r'[smf]!"', open(p, encoding='utf-8').read())] or "none")
    print()


def l38():
    print("## L38: names the manifest check accepts (keywords are plain names too)")
    man = tvr.manifest("vec_add_f32")
    for ns in ("Kernels.VecAdd", "end", "sorry", "Kernels.K'", "_root_.X", "Kernels.K\n#exit"):
        m = {**man, "lean": {**man["lean"], "namespace": ns}}
        print(f"namespace {ns!r}: {tvr.manifest_problems('vec_add_f32', m) or 'accepted'}")
    print()


if __name__ == "__main__":
    l39()
    numbering()
    l40()
    l38()
