"""Library review round 3 (lr-3b), L7, end to end: copy the TVA tree (without .venv, .lake, .git,
__pycache__) to SCRATCH, append one line with a shell redirection into a sibling folder of TVA
(`>` + a /home path, no space) to docs/SEMANTICS.md in the copy, and run the copy's `tvr audit`.
Paths are joined at run time. Usage, from the TVA root:
  python3 -I docs/library-review/round-3/lr-3b/work/l7_e2e.py SCRATCH"""
import os
import shutil
import subprocess
import sys

TVA = os.getcwd()
dst = os.path.join(sys.argv[1], "tva")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(TVA, dst, symlinks=True, ignore=shutil.ignore_patterns(".venv", ".lake", ".git", "__pycache__"))
S = "/"
for name, line in (("redirect >/home", "python3 run.py >" + S + "home" + S + "someone" + S + "fpv" + S + "other" + S + "out.txt"),
                   ("redirect >~", "echo x >~" + S + "fpv" + S + "other" + S + "x"),
                   ("control: same with a space", "python3 run.py > " + S + "home" + S + "someone" + S + "fpv" + S + "other" + S + "out.txt")):
    p = os.path.join(dst, "docs", "SEMANTICS.md")
    old = open(p).read()
    with open(p, "a") as f:
        f.write("\n" + line + "\n")
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "audit"], cwd=dst, capture_output=True, text=True)
    first = (r.stdout.strip().splitlines() or [""])[0]
    # print the kind of the result only, never the path the audit quotes
    what = "independent" if r.returncode == 0 else ("a path outside TVA" if "a path outside TVA" in first else "other problem")
    print(f"{name}: audit exit {r.returncode}: {what}")
    open(p, "w").write(old)
shutil.rmtree(dst)
