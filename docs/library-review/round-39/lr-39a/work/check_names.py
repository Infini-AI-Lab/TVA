"""Check every identifier-looking backticked name in LIBRARY.md's rows for the five round-39
modules, in the module docstrings and in the author's notes, against the declarations Lean has."""
import re, sys, pathlib
work = pathlib.Path(__file__).resolve().parent
tva = work.parents[4]
decls = [l.split()[1] for l in (work / "names.txt").read_text().splitlines()]
suffixes = set()
for d in decls:
    parts = d.split(".")
    for k in range(len(parts)):
        suffixes.add(".".join(parts[k:]))
rows = [l for l in (tva / "docs/LIBRARY.md").read_text().splitlines()
        if re.match(r"\| `(IntW|Tile2|FloatFacts|OnlineSoftmax|Recurrence)`", l)]
srcs = {"LIBRARY.md rows": "\n".join(rows),
        "notes.md": (tva / "docs/library-review/round-39/author/notes.md").read_text()}
for m in ["IntW", "Tile2", "FloatFacts", "OnlineSoftmax", "Recurrence"]:
    t = (tva / f"lean/TVA/Lemmas/{m}.lean").read_text()
    srcs[m + " docstring"] = t.split("-/", 1)[0]
ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_?']*(\.[A-Za-z_][A-Za-z0-9_?']*)*$")
for name, text in srcs.items():
    missing = set()
    for tok in re.findall(r"`([^`]+)`", text):
        for t in re.split(r"[/ ,]+", tok):
            t = t.strip("()")
            if not ident.match(t) or "_" not in t:
                continue
            if t not in suffixes:
                missing.add(t)
    print(name, "->", sorted(missing))
