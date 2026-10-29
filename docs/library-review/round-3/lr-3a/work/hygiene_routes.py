"""lr-3a, round 3: the gate's hygiene list (`HYGIENE` in tools/tvr.py) applied to the two route files
in work/lean, and the namespace check's rule (a declaration whose root is `TVA`) applied to their
declarations. Run from the TVA root: python3 -I docs/library-review/round-3/lr-3a/work/hygiene_routes.py"""
import importlib.util
import os
import re

spec = importlib.util.spec_from_file_location("tvr", os.path.join(os.getcwd(), "tools", "tvr.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
W = os.path.join("docs", "library-review", "round-3", "lr-3a", "work", "lean")
for f in ("NotationRoute.lean", "ShadowRoute.lean"):
    text = open(os.path.join(W, f), encoding="utf-8").read()
    hits = [(what, m.group(0)) for what, pat in t.HYGIENE for m in re.finditer(pat, text, re.M)]
    decls = re.findall(r"^(?:def|theorem)\s+(\S+)", text, re.M)
    ns = re.findall(r"^namespace\s+(\S+)", text, re.M)
    full = [f"{ns[0]}.{d}" for d in decls]
    print(f"{f}: hygiene hits: {hits or 'none'}; declarations: {full}; any with root TVA: "
          f"{any(n.split('.')[0] == 'TVA' for n in full)}")
