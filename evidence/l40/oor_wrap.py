"""Make an evidence script's per-shape loops skip block shapes that do not fit this GPU's shared memory.

  oor_wrap.py FILE [FILE ...]

For each loop `for (M, N, nw...) in ...:` the body is wrapped in `try:` / `except OutOfResources:`, which
records the shape as rejected (a row naming it) and goes on with the next shape. On a GPU where every shape
fits (the H200) the script behaves exactly as before. The original is kept as FILE.orig."""
import re
import shutil
import sys

HEAD = re.compile(r"^(\s*)for \(M, N, nw[^)]*\) in .*:\s*$")
HANDLER = """{ind}    except triton.runtime.errors.OutOfResources:
{ind}        _r = "| rejected: out of resources on this device | " + ", ".join(f"{{k}} = {{locals()[k]}}" for k in ("f", "M", "N", "nw", "K", "D", "T", "pk") if k in locals()) + " |"
{ind}        (locals().get("out") or print)(_r)
{ind}        continue
"""


def wrap(src):
    lines = src.split("\n")
    out, i, n = [], 0, 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if not m:
            out.append(lines[i]); i += 1; continue
        ind = m.group(1)
        out.append(lines[i]); i += 1
        body = []
        while i < len(lines) and (lines[i].strip() == "" or len(lines[i]) - len(lines[i].lstrip()) > len(ind)):
            body.append(lines[i]); i += 1
        while body and body[-1].strip() == "":       # trailing blank lines stay outside the loop
            i -= 1; body.pop()
        out.append(ind + "    try:")
        out += [("    " + l) if l.strip() else l for l in body]
        out += HANDLER.format(ind=ind).rstrip("\n").split("\n")
        n += 1
    return "\n".join(out), n


for p in sys.argv[1:]:
    shutil.copy(p, p + ".orig")
    with open(p) as f:
        src = f.read()
    new, n = wrap(src)
    compile(new, p, "exec")
    with open(p, "w") as f:
        f.write(new)
    print(f"{p}: {n} loops wrapped")
