"""`tvr audit` on a copy of the tree (no Lean build): the round-7 cases L40 (lake options), L41 (shipped files
the scan skips), L43 (rule pages in subfolders), and two cases the round-7 checks do not cover: a
`lean/lean-toolchain` that names a toolchain other than an official Lean release, and a library module
that `lean/TVA.lean` does not import (so `lake build TVA` does not build it and `leanchecker TVA` in the
release check does not replay it, though a kernel may import it).

The copy is made without `.lake`, `.venv` and `__pycache__`, with the git metadata; the tree is not written.

Usage: python3 -I docs/library-review/round-7/lr-7b/work/audit_probes.py <empty scratch folder outside the home directory>
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
SIBLING = "fpv" + "/" + "sibling" + "/notes"  # a path into a sibling of TVA, built at run time


def run(root, *args):
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py"] + list(args), cwd=root, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
    return r.returncode, " | ".join(lines[:3])[:400].replace(SIBLING, "<a sibling of TVA>")


def main():
    d = os.path.realpath(sys.argv[1])
    assert not os.listdir(d)
    root = os.path.join(d, "tva")
    shutil.copytree(TVA, root, symlinks=True,
                    ignore=lambda _, names: [n for n in names if n in (".lake", ".venv", "__pycache__", ".git")])
    shutil.copytree(os.path.join(TVA, ".git"), os.path.join(root, ".git"), symlinks=True)
    for who in ("lr-7a", "lr-7b"):  # this round's reports in progress, not part of the cases
        shutil.rmtree(os.path.join(root, "docs", "library-review", "round-7", who), ignore_errors=True)
    p = lambda *x: os.path.join(root, *x)
    print("the copy as it is: audit exit %d (%s)" % run(root, "audit"))

    print("\n## L43: rule pages")
    os.makedirs(p("lean", "TVA", "Rules", "Num"))
    open(p("lean", "TVA", "Rules", "Num", "Exp2.lean"), "w").write("import TVA.Rules.Num\n")
    print("lean/TVA/Rules/Num/Exp2.lean: audit exit %d (%s)" % run(root, "audit"))
    os.makedirs(p("lean", "TVA", "Rules", "Extra"))
    open(p("lean", "TVA", "Rules", "Extra", "Y.lean"), "w").write("import TVA.Rules.Num\n")
    print("lean/TVA/Rules/Extra/Y.lean: audit exit %d (%s)" % run(root, "audit"))
    shutil.rmtree(p("lean", "TVA", "Rules", "Num"))
    shutil.rmtree(p("lean", "TVA", "Rules", "Extra"))

    print("\n## L40: lean/lakefile.toml leanOptions")
    lake = p("lean", "lakefile.toml")
    base = open(lake).read()
    for opt in ("debug.skipKernelTC = true", "maxHeartbeats = 0", "trace.profiler = true", "pp.all = true",
                "[leanOptions.debug]\nskipKernelTC = true", "linter.all = \"yes\"", "Elab.async = false"):
        text = base.replace("autoImplicit = false\n", "autoImplicit = false\n" + opt + "\n", 1)
        if opt.startswith("[leanOptions.debug]"):
            text = base + "\n" + opt + "\n"
        open(lake, "w").write(text)
        print(f"{opt!r}: audit exit %d (%s)" % run(root, "audit"))
    open(lake, "w").write(base)

    print("\n## L41: a note in a nested .lake folder")
    os.makedirs(p("kernels", "vec_add_f32", "lean", ".lake"))
    open(p("kernels", "vec_add_f32", "lean", ".lake", "note.md"), "w").write(f"see ~/{SIBLING}\n")
    ls = subprocess.run(["git", "-C", root, "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout
    print(f"untracked kernels/vec_add_f32/lean/.lake/note.md shipped: {'.lake/note.md' in ls}; audit exit %d (%s)"
          % run(root, "audit"))
    subprocess.run(["git", "-C", root, "add", "-f", "kernels/vec_add_f32/lean/.lake/note.md"], capture_output=True)
    print("after git add -f: audit exit %d (%s)" % run(root, "audit"))
    subprocess.run(["git", "-C", root, "rm", "-q", "--cached", "kernels/vec_add_f32/lean/.lake/note.md"], capture_output=True)
    shutil.rmtree(p("kernels", "vec_add_f32", "lean", ".lake"))

    print("\n## Not covered: the toolchain lean/lean-toolchain names")
    tc = p("lean", "lean-toolchain")
    saved = open(tc).read()
    for name in ("someone/lean4:v4.35.0-patched", "leanprover/lean4:nightly"):
        open(tc, "w").write(name + "\n")
        print(f"lean-toolchain {name!r}: audit exit %d (%s)" % run(root, "audit"))
    open(tc, "w").write(saved)

    print("\n## Not covered: a library module outside lean/TVA.lean's imports")
    open(p("lean", "TVA", "Lemmas", "Orphan.lean"), "w").write(
        "import TVA.Claim\n\n/-! A module no other library module imports. -/\nnamespace TVA\n\ntheorem orphan : True := trivial\n\nend TVA\n")
    print("lean/TVA/Lemmas/Orphan.lean (not imported by lean/TVA.lean): audit exit %d (%s)" % run(root, "audit"))
    shutil.rmtree(d + os.sep + "tva")


if __name__ == "__main__":
    main()
