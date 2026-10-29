"""lr-6a, round 6 (survey area `gate`): which source lines does the gate's map check require to be
covered?

`check` in tools/tvr.py (the loop after `src_lines = src_text.splitlines()`) skips a line of the
kernel's ranges when it is blank, starts with `#`, or holds `\"\"\"` (it toggles a docstring state on an
odd count, and skips the line itself whatever the count). This script copies that loop verbatim from
the tool (by reading its text), applies it to three small kernels, and compares the lines it requires
with the lines Python's own tokenizer reads as code (a token other than a comment, a newline, an
indent, or a string statement standing alone).

Usage: python3 -I map_coverage.py  (no arguments; prints to stdout)
"""
import io
import os
import re
import tokenize

TVA = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
TQ = '"' * 3

CASES = {
    "K1: a call whose string argument is triple-quoted": [
        "@triton.jit",
        "def k(c, x_ptr):",
        "    x = tl.load(x_ptr)",
        f"    tl.device_print({TQ}x{TQ}, tl.atomic_add(c, 1))",
        "    tl.store(x_ptr, x)",
    ],
    "K2: triple quotes in two comments": [
        "@triton.jit",
        "def k(c, x_ptr):",
        f"    x = tl.load(x_ptr)  # {TQ}",
        "    tl.atomic_add(c, 1)",
        f"    tl.store(x_ptr, x)  # {TQ}",
    ],
    "K3: an honest docstring (control)": [
        "@triton.jit",
        "def k(c, x_ptr):",
        f"    {TQ}Adds one.",
        "    tl.atomic_add(c, 1) is how it would read",
        f"    {TQ}",
        "    tl.atomic_add(c, 1)",
    ],
}


def tool_loop():
    """The coverage loop of tools/tvr.py, as text, made into a function of (src_lines, ranges)."""
    s = open(os.path.join(TVA, "tools", "tvr.py")).read()
    start = s.index("    missing, in_doc = [], False\n")
    end = s.index("    add(\"map\", not missing and not problems,")
    body = s[start:end]
    assert "for a, b in man[\"source\"][\"ranges\"]:" in body
    body = body.replace("man[\"source\"][\"ranges\"]", "ranges")
    ns = {}
    exec("def required(src_lines, ranges, covered=frozenset()):\n" + body + "    return missing\n", ns)
    return ns["required"], body


def python_code_lines(lines):
    """Lines that hold a token of code, by Python's tokenizer: a docstring (a string that is the whole
    statement) and comments do not count."""
    src = "\n".join(lines) + "\n"
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    code = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    for i, t in enumerate(toks):
        if t.type in skip:
            continue
        if t.type == tokenize.STRING:
            prev = next((p for p in reversed(toks[:i]) if p.type not in (tokenize.NL, tokenize.COMMENT)), None)
            nxt = next((p for p in toks[i + 1:] if p.type not in (tokenize.NL, tokenize.COMMENT)), None)
            if (prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)) and \
                    (nxt is None or nxt.type in (tokenize.NEWLINE, tokenize.DEDENT, tokenize.ENDMARKER)):
                continue
        code.update(range(t.start[0], t.end[0] + 1))
    return code


def main():
    required, body = tool_loop()
    print("the tool's loop (tools/tvr.py, check):")
    for l in body.rstrip().splitlines():
        print("  " + l)
    print()
    for name, lines in CASES.items():
        need = set(required(lines, [[1, len(lines)]]))
        code = python_code_lines(lines)
        print(f"## {name}")
        for i, l in enumerate(lines, 1):
            mark = ("required" if i in need else "not required") + ("" if (i in need) == (i in code) else
                                                                     ", but Python reads code here")
            print(f"  {i}: {l:<60} {mark}")
        print()


if __name__ == "__main__":
    main()
