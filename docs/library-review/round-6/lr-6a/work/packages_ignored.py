"""lr-6a, round 6 (L7, L36): what does the release check's `packages` step see in the Mathlib packages it
borrows?

`package_problems` in tools/tvr.py accepts a package when `git rev-parse HEAD` is the manifest's
revision and `git status --porcelain --untracked-files=all` prints nothing. That command does not list
files git ignores: the package's own .gitignore (Mathlib's ignores `**/__pycache__/` and `.lake` at any
depth), the checkout's `.git/info/exclude`, and the user's global excludes. Every built `.olean` is
such a file. The sandbox binds the packages folder whole, so lake builds and imports whatever it holds.

Parts (no file of TVA or of its packages is written; the probes live in a scratch directory):
  P1  in the real Mathlib checkout, read-only: which of three candidate module files git ignores
      (`git check-ignore -v`; the files need not exist).
  P2  in a scratch git repository with Mathlib's ignore rule and one excluded path in
      `.git/info/exclude`: the exact command of `package_problems` on two added module files.
  P3  TVA's import check (`import_violations` with the library's roots) on a header importing
      `Mathlib.__pycache__.Lemmas`.
  P4  Lean resolves that import: a module `Mathlib.__pycache__.Lemmas` compiled to an `.olean` in the
      scratch directory, imported by a second file with the scratch directory on LEAN_PATH, both run
      with `lake env lean` from TVA's `lean/` folder (no lake build).

Usage: python3 -I packages_ignored.py SCRATCH  (an empty scratch directory outside TVA)
"""
import os
import subprocess
import sys

TVA = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
LAKE = os.path.expanduser(os.path.join("~", ".elan", "bin", "lake"))
MATHLIB = os.path.join(TVA, "lean", ".lake", "packages", "mathlib")
CANDIDATES = [os.path.join("Mathlib", "__pycache__", "Lemmas.lean"),
              os.path.join("Mathlib", "Analysis", "FloatErrLocal.lean"),
              os.path.join("Mathlib", "Logic", "Basic.lean")]


def sh(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def main():
    print("## P1: the real Mathlib checkout (read-only)")
    for c in CANDIDATES:
        r = sh(["git", "-C", MATHLIB, "check-ignore", "-v", c])
        why = r.stdout.strip().split("\t")[0] if r.returncode == 0 else ""
        print(f"  {c}: {'ignored by ' + why if why else 'not ignored'}")
    st = sh(["git", "-C", MATHLIB, "status", "--porcelain", "--untracked-files=all"])
    print(f"  git status --porcelain --untracked-files=all prints {len(st.stdout.splitlines())} line(s)")

    print("## P2: a scratch repository, the command of package_problems")
    repo = os.path.join(SCRATCH, "repo")
    os.makedirs(os.path.join(repo, "Mathlib", "Analysis"))
    sh(["git", "init", "-q", repo])
    open(os.path.join(repo, ".gitignore"), "w").write("**" + "/" + "__pycache__" + "/" + "\n" + ".lake" + "\n")
    open(os.path.join(repo, "Mathlib", "Basic.lean"), "w").write("theorem basic : True := trivial\n")
    sh(["git", "-C", repo, "add", "-A"])
    sh(["git", "-C", repo, "-c", "user.name=p", "-c", "user.email=p@example.invalid", "commit", "-qm", "base"])
    os.makedirs(os.path.join(repo, "Mathlib", "__pycache__"))
    open(os.path.join(repo, CANDIDATES[0]), "w").write("theorem probe1 : True := trivial\n")
    open(os.path.join(repo, CANDIDATES[1]), "w").write("theorem probe2 : True := trivial\n")
    open(os.path.join(repo, ".git", "info", "exclude"), "a").write("/" + CANDIDATES[1].replace(os.sep, "/") + "\n")
    st = sh(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all"])
    print(f"  two module files added, one under __pycache__, one listed in .git/info/exclude: "
          f"git status prints {len(st.stdout.splitlines())} line(s) {st.stdout.strip()!r}")
    st2 = sh(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all", "--ignored"])
    print(f"  with --ignored it prints: {st2.stdout.strip().splitlines()}")

    print("## P3: TVA's import check")
    hdr = os.path.join(SCRATCH, "Header.lean")
    open(hdr, "w").write("import Mathlib.__pycache__.Lemmas\nimport TVA\n")
    print(f"  import_violations (roots {tvr.LEAN_ROOTS}): {tvr.import_violations([hdr], tvr.LEAN_ROOTS) or 'none'}")

    print("## P4: Lean resolves the import")
    lib = os.path.join(SCRATCH, "olean")
    os.makedirs(os.path.join(lib, "Mathlib", "__pycache__"))
    srcroot = os.path.join(SCRATCH, "src")
    src = os.path.join(srcroot, CANDIDATES[0])
    os.makedirs(os.path.dirname(src))
    open(src, "w").write("/-- a lemma from a file a fresh Mathlib checkout does not have -/\n"
                         "theorem lr6aProbeLemma : (2 : Nat) + 2 = 4 := rfl\n")
    lean_dir = os.path.join(TVA, "lean")
    r = sh([LAKE, "env", "lean", "--root=" + srcroot, "-o",
            os.path.join(lib, "Mathlib", "__pycache__", "Lemmas.olean"), src], cwd=lean_dir)
    print(f"  compiled Mathlib.__pycache__.Lemmas: exit {r.returncode} {r.stderr.strip()[:200]}")
    use = os.path.join(SCRATCH, "Use.lean")
    open(use, "w").write("import Mathlib.__pycache__.Lemmas\n#check lr6aProbeLemma\n")
    r = sh([LAKE, "env", "sh", "-c", 'LEAN_PATH="$0:$LEAN_PATH" exec lean "$1"', lib, use], cwd=lean_dir)
    print(f"  importing file: exit {r.returncode}; output {r.stdout.strip()[:200]!r} {r.stderr.strip()[:200]}")


if __name__ == "__main__":
    main()
