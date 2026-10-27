"""lr-2a, round 2, survey `gate`: run tvr's exact `hygiene` regex over the GateHygiene probe.
Run from the TVA root:  python3 -I docs/library-review/round-2/lr-2a/work/hygiene_regex.py
"""
import re

# copied verbatim from tools/tvr.py check() (the `hygiene` gate check)
pat = re.compile(r"\bsorry\b|\badmit\b|\bnative_decide\b|^\s*axiom\b|implemented_by|@\[extern|\bunsafe\b|set_option\s+debug\.")
p = "docs/library-review/round-2/lr-2a/work/lean/GateHygiene.lean"
hits = [(i, l.rstrip()) for i, l in enumerate(open(p), 1) if pat.search(l.split("--")[0])]
print("hygiene regex hits:", hits if hits else "NONE -> the hygiene check passes")
