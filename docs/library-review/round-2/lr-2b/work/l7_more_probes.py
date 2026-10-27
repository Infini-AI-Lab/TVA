"""lr-2b, library review round 2, L7: probes that the author's 53 and round 1's 21 do not try.

Usage, from the TVA root:
  python3 -I docs/library-review/round-2/lr-2b/work/l7_more_probes.py SCRATCH

SCRATCH is an empty scratch directory outside TVA. Each probe copies the tree there (without .venv,
.lake, .git, __pycache__), appends one line to one file, and runs the copy's own `tools/tvr.py audit`.
Each probe text is a path that leads out of TVA into a made-up directory beside it, spelled in a way
the shell or a browser resolves (no project is named). Paths are joined at run time, so this file
holds no such path. Controls must pass.
"""
import os
import shutil
import subprocess
import sys

TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))))
SCRATCH = sys.argv[1]
S = "/"
D = "$"
OUT_DIR = S.join(["fpv", "archive", "notes.md"])      # a made-up directory beside TVA
USER = S.join(["", "home", "someone"])

PROBES = [
    # (id, file, line, must be flagged)
    ("H1", "docs/SEMANTICS.md", f"See {D}HOME{S}{OUT_DIR}.", True),
    ("H2", "docs/SEMANTICS.md", f"See {D}{{HOME}}{S}{OUT_DIR}.", True),
    ("H3", "docs/SEMANTICS.md", f"See file:{S}{S}{USER}{S}{OUT_DIR}.", True),
    ("H4", "docs/SEMANTICS.md", f"See <file:{S}{S}{USER}{S}{OUT_DIR}>.", True),
    ("H5", "docs/SEMANTICS.md", f"See ~someone{S}{OUT_DIR}.", True),
    ("H6", "docs/SEMANTICS.md", f"See {USER}{S}{OUT_DIR}.", True),        # control: caught today
    ("C1", "docs/SEMANTICS.md", "See https://triton-lang.org/main/index.html.", False),
]


def fresh():
    dst = os.path.join(SCRATCH, "tva")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(TVA, dst, symlinks=True,
                    ignore=shutil.ignore_patterns(".venv", ".lake", ".git", "__pycache__"))
    return dst


def main():
    wrong = 0
    for pid, f, line, must in PROBES:
        dst = fresh()
        with open(os.path.join(dst, f), "a", encoding="utf-8") as h:
            h.write("\n" + line + "\n")
        r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "audit"], cwd=dst, capture_output=True, text=True)
        flagged = r.returncode != 0 or "independent" not in r.stdout
        ok = flagged == must
        wrong += not ok
        print(f"{pid}: {'flagged' if flagged else 'passed'} "
              f"({'as the rule asks' if ok else 'NOT as the rule asks'})", flush=True)
    shutil.rmtree(os.path.join(SCRATCH, "tva"), ignore_errors=True)
    print(f"{len(PROBES)} probes, {wrong} not as the rule asks")


if __name__ == "__main__":
    main()
