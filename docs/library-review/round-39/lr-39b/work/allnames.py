"""List the identifiers named by every row of docs/LIBRARY.md's module tables and by every
lean/TVA/Lemmas module header (backticked, identifier-shaped, with a dot or an underscore)."""
import re, sys, json, glob, os
def toks(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    out = set()
    for t in re.findall(r"`([^`]+)`", text):
        for part in re.split(r"[ ,/()\[\]]+", t):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_'?.!]*", part) and ("_" in part or "." in part):
                out.add(part)
    return sorted(out)
res = {}
for n, line in enumerate(open("docs/LIBRARY.md"), 1):
    if line.startswith("| `"):
        res[f"LIBRARY.md:{n}"] = toks(line)
for f in sorted(glob.glob("lean/TVA/Lemmas/*.lean")):
    src = open(f).read()
    m = re.search(r"/-!(.*?)-/", src, flags=re.S)
    if m:
        res[os.path.basename(f)] = toks(m.group(1))
json.dump(res, sys.stdout, indent=0)
