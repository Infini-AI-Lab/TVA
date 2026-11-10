"""lr-41b, round 41: every name the Tile2 and IntW rows of docs/LIBRARY.md list resolves in Lean
(`#check`, with the namespaces the rows imply open). Usage: python3 -I check_names.py <tva>."""
import re, subprocess, sys, os
tva = sys.argv[1]
rows = [l for l in open(os.path.join(tva, "docs/LIBRARY.md"), encoding="utf-8") if l.startswith(("| `Tile2`", "| `IntW`"))]
names = set()
for l in rows:
    for t in re.findall(r"`([A-Za-z_][A-Za-z0-9_.'?/…]*)`", l):
        if any(c in t for c in "…/ ") or t in ("Tile2", "IntW", "simp", "int32", "int64", "IntV", "Tensor"):
            continue
        names.add(t)
lean = ["import TVA", "open TVA TVA.Tile2 TVA.IntV TVA.Buf TVA.ConvRel TVA.Tensor TVA.IntV.I32 TVA.IntV.I64"]
order = sorted(names)
for n in order:
    lean.append(f"#check @{n}")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Names.lean")
open(path, "w").write("\n".join(lean) + "\n")
r = subprocess.run([os.path.expanduser("~/.elan/bin/lake"), "env", "lean", path], cwd=os.path.join(tva, "lean"),
                   capture_output=True, text=True)
errs = [l for l in r.stdout.splitlines() if "error" in l]
print(f"{len(order)} names checked, {len(errs)} errors")
for e in errs:
    m = re.search(r"Names.lean:(\d+):", e)
    print(order[int(m.group(1)) - 3] if m else "", "|", e.split("error", 1)[1][:120])
