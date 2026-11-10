"""List the identifiers each round-39 module's header names (backticked, identifier-shaped)."""
import re, sys, json
mods = ["IntW", "Tile2", "FloatFacts", "Recurrence", "OnlineSoftmax"]
out = {}
for m in mods:
    src = open(f"lean/TVA/Lemmas/{m}.lean").read()
    head = re.sub(r"```.*?```", "", src.split("-/", 1)[0], flags=re.S)
    toks = set()
    for t in re.findall(r"`([^`]+)`", head):
        for part in re.split(r"[ ,/]+", t):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'?.]*", part) and ("_" in part or "." in part):
                toks.add(part)
    out[m] = sorted(toks)
json.dump(out, sys.stdout, indent=0)
