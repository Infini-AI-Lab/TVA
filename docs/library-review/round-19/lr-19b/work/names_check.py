"""lr-19b: every backticked name in the given documents (for a .lean file, its docstrings only)
that looks like a Lean identifier and is not a suffix of a declaration under `TVA`.
Usage: python3 -I names_check.py <names file, from `lake env lean Names.lean`> <document> ...
The misses it prints are variables, module names, Python names and PTX names, read by hand."""
import os
import re
import sys

names = open(sys.argv[1]).read().split()
suffixes = set()
for n in names:
    parts = n.split(".")
    for i in range(len(parts)):
        suffixes.add(".".join(parts[i:]))
ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_'?!₀-₉]*(\.[A-Za-z_][A-Za-z0-9_'?!₀-₉]*)*$")
print(f"{len(names)} declarations")
for path in sys.argv[2:]:
    text = open(path).read()
    if path.endswith(".lean"):
        text = "\n".join(re.findall(r"/-[-!](.*?)-/", text, re.S))
    text = re.sub(r"```.*?```", "", text, flags=re.S)   # fenced code blocks would misalign the pairs
    toks = {m.group(1) for m in re.finditer(r"`([^`]+)`", text) if ident.match(m.group(1))}
    miss = sorted(t for t in toks if t not in suffixes)
    print(f"{os.path.relpath(path)}: {len(toks)} names, not declarations: {' '.join(miss)}")
