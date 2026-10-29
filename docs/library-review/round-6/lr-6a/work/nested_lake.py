"""lr-6a, round 6 (survey area `gate`): does the text scan read every file TVA ships?

The scan (`text_files` in tools/tvr.py) skips every folder named `.lake`, `.venv`, `.git` or
`__pycache__`, at any depth. TVA's .gitignore ignores `.venv/` and `__pycache__/` at any depth, but
`.lake` only as `lean/.lake/`. So a folder `.lake` elsewhere (in `docs/`, or in a kernel's `lean/`
folder, where running lake makes one) is shipped (`git ls-files --others --exclude-standard`, the
release check's list) and never scanned.

Method, on a copy of the tree (with its git metadata) in a scratch directory outside the home
directory: add `docs/.lake/note.md` and `kernels/vec_add_f32/lean/.lake/note.md`, each holding a
path into a sibling folder of TVA (assembled at run time here); then report whether git lists them
among the files TVA ships and what the copy's `tvr audit` says. Control: the same note at
`docs/note.md`. The real tree is never written.

Usage: python3 -I nested_lake.py  (no arguments; prints to stdout; the scratch directory is removed)
"""
import os
import shutil
import subprocess
import sys
import tempfile

TVA = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
NOTE = "Measured with the script in " + "~" + "/" + "fpv" + "/" + "sibx" + "/bench.py.\n"


def run(root, rels):
    for r in rels:
        p = os.path.join(root, r)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(NOTE)
    ls = subprocess.run(["git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard"],
                        capture_output=True, text=True).stdout.splitlines()
    a = subprocess.run([sys.executable, "-I", "tools/tvr.py", "audit"], cwd=root, capture_output=True, text=True)
    for r in rels:
        print(f"  {r}: shipped (git lists it): {r in ls}; named by tvr audit: {r in a.stdout}")
    print(f"  tvr audit: {'PASS (independent)' if a.returncode == 0 else 'FAIL'}")
    for r in rels:
        os.remove(os.path.join(root, r))


def main():
    tmp = tempfile.mkdtemp(prefix="lr6a-lake-", dir=os.sep + os.path.join("var", "tmp"))
    root = os.path.join(tmp, "tva")
    try:
        shutil.copytree(TVA, root, symlinks=True, ignore=shutil.ignore_patterns(".venv", ".lake", "__pycache__"))
        print("## baseline (the copy as it is)")
        a = subprocess.run([sys.executable, "-I", "tools/tvr.py", "audit"], cwd=root, capture_output=True, text=True)
        print(f"  tvr audit: {'PASS (independent)' if a.returncode == 0 else 'FAIL: ' + a.stdout[:200]}")
        print("## N1: the note in two nested .lake folders")
        run(root, [os.path.join("docs", ".lake", "note.md"),
                   os.path.join("kernels", "vec_add_f32", "lean", ".lake", "note.md")])
        print("## N2 (control): the note in docs/")
        run(root, [os.path.join("docs", "note.md")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
