"""List the identifiers docs/LIBRARY.md's rows for the five round-39 modules name."""
import re, sys, json
rows = {}
for line in open("docs/LIBRARY.md"):
    m = re.match(r"\| `(IntW|Tile2|FloatFacts|OnlineSoftmax|Recurrence)` \|", line)
    if not m:
        continue
    toks = set()
    for t in re.findall(r"`([^`]+)`", line):
        for part in re.split(r"[ ,/]+", t):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'?.]*", part) and ("_" in part or "." in part):
                toks.add(part)
    rows["LIBRARY.md:" + m.group(1)] = sorted(toks)
json.dump(rows, sys.stdout, indent=0)
