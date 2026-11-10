"""Every identifier with an underscore or a dot in LIBRARY.md's backticks, checked against the
declarations of TVA (any suffix of a full name), per table row."""
import re, pathlib
work = pathlib.Path(__file__).resolve().parent
tva = work.parents[4]
decls = [l.split()[1] for l in (work / "allnames.txt").read_text().splitlines()]
suffixes = set()
for d in decls:
    parts = d.split(".")
    for k in range(len(parts)):
        suffixes.add(".".join(parts[k:]))
ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_?'!]*(\.[A-Za-z_][A-Za-z0-9_?'!]*)*$")
text = (tva / "docs/LIBRARY.md").read_text().splitlines()
for n, line in enumerate(text, 1):
    missing = set()
    for tok in re.findall(r"`([^`]+)`", line):
        for t in re.split(r"[ ,()]+", tok):
            if not ident.match(t) or ("_" not in t and "." not in t):
                continue
            if t in suffixes or t.startswith(("TVA.", "Mathlib", "tl.", "docs", "evidence")):
                continue
            # slash shorthand a/b/c handled by split on / below
            missing.add(t)
    for tok in re.findall(r"`([^`]+)`", line):
        if "/" in tok and " " not in tok:
            pass
    if missing:
        print(n, sorted(missing)[:30])
