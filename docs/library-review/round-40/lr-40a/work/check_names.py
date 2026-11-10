"""lr-40a, round 40: every identifier-like token in backticks in docs/LIBRARY.md and in the
module headers and docstrings of lean/TVA/Lemmas/*.lean that names no TVA declaration (as any
dotted suffix of a full name). Shorthand `a/b/c` is expanded on the last component; tokens
without `_` or `.` are skipped (mostly variables). Run from this folder with `python3 -I`."""
import re, pathlib, sys
work = pathlib.Path(__file__).resolve().parent
tva = work.parents[4]
decls = [l.split()[1] for l in (work / "allnames.txt").read_text().splitlines()]
suf = set()
for d in decls:
    p = d.split(".")
    for k in range(len(p)):
        suf.add(".".join(p[k:]))
ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_?'!]*(\.[A-Za-z_][A-Za-z0-9_?'!]*)*$")

def expand(tok):
    if "/" not in tok:
        return [tok]
    parts = tok.split("/")
    head = parts[0]
    out = [head]
    pre = head.rsplit("_", 1)[0] if "_" in head else ""
    for q in parts[1:]:
        out.append(pre + "_" + q if pre and "_" not in q and "." not in q else q)
    return out

def check(path, text):
    for n, line in enumerate(text.splitlines(), 1):
        miss = []
        for tok in re.findall(r"`([^`]+)`", line):
            for t in re.split(r"[ ,()\[\]:]+", tok):
                for u in expand(t):
                    u = u.strip(".")
                    if not ident.match(u) or ("_" not in u and "." not in u):
                        continue
                    if u in suf or u.startswith(("tl.", "Mathlib", "Real.", "Finset.", "Nat.", "Int.")):
                        continue
                    miss.append(u)
        if miss:
            print(f"{path}:{n}: {sorted(set(miss))}")

check("docs/LIBRARY.md", (tva / "docs/LIBRARY.md").read_text())
for f in sorted((tva / "lean/TVA/Lemmas").glob("*.lean")):
    s = f.read_text()
    doc = "\n".join(m.group(0) for m in re.finditer(r"/-[-!].*?-/", s, re.S))
    check(f"lean/TVA/Lemmas/{f.name}", doc)
