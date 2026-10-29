"""lr-7a, library review round 7: does the release check's `packages` step accept a package commit that
is not Mathlib's (or its dependency's) upstream commit?

The round-7 fix (L7) makes each package of lean/lake-manifest.json afresh "from the working tree's
checkout of the package if it has the commit, else from the manifest's URL". This script builds a
scratch TVA root (never the tree) whose lean/.lake/packages/mathlib and .../batteries are clones of the
tree's checkouts, each with one extra local commit that adds a module file (a stand-in for another
project's lemma under a new name), and whose lake-manifest.json pins those local commits. It then calls
the tree's own `tools/tvr.py` functions on that root:
  P1  `lake_violations()`  (the audit's lake check)
  P2  `make_packages()`    (the release check's `packages` step) into a fresh copy, and whether the
                           extra module file is in the package it made
  P3  `import_violations()` on a TVA file that imports the extra Mathlib module
  P4  `lake_violations()` with an inherited package's URL pointing at a repository other than the
                           Lean community's
Usage: python3 -I work/packages_local_commit.py SCRATCH   (an empty folder outside TVA; never printed)
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True  # import the tree's modules without writing caches into it

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
scratch = os.path.abspath(sys.argv[1])
assert not os.listdir(scratch), "SCRATCH must be empty"

spec = importlib.util.spec_from_file_location("tvr", os.path.join(TVA, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
ENV.update(GIT_AUTHOR_NAME="probe", GIT_AUTHOR_EMAIL="probe@invalid", GIT_COMMITTER_NAME="probe",
           GIT_COMMITTER_EMAIL="probe@invalid")


def git(*args, cwd=None, inp=None, extra=None):
    env = dict(ENV, **(extra or {}))
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, input=inp, env=env)
    if r.returncode != 0:
        raise SystemExit(f"git {args[0]} failed: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def local_commit(src, dst, base, path, text):
    """Clone `src` (shared objects, no checkout) to `dst` and add one commit on `base` that adds `path`."""
    git("clone", "-q", "--shared", "--no-checkout", src, dst)
    idx = {"GIT_INDEX_FILE": os.path.join(dst, ".git", "probe-index")}
    git("read-tree", base, cwd=dst, extra=idx)
    blob = git("hash-object", "-w", "--stdin", cwd=dst, inp=text)
    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", cwd=dst, extra=idx)
    tree = git("write-tree", cwd=dst, extra=idx)
    c = git("commit-tree", tree, "-p", base, "-m", "local commit", cwd=dst)
    git("update-ref", "refs/heads/probe", c, cwd=dst)
    return c


root = os.path.join(scratch, "root")
os.makedirs(os.path.join(root, "lean", ".lake", "packages"))
for f in ("lakefile.toml", "lean-toolchain", "lake-manifest.json"):
    shutil.copy(os.path.join(TVA, "lean", f), os.path.join(root, "lean", f))
# the scratch lakefile keeps TVA's library only (the kernel's library would point outside the scratch root)
lf = os.path.join(root, "lean", "lakefile.toml")
text = open(lf).read().split("# One library per kernel folder")[0].replace(', "vec_add_f32"', "")
open(lf, "w").write(text)
man = json.load(open(os.path.join(root, "lean", "lake-manifest.json")))
pk = {p["name"]: p for p in man["packages"]}
tree_pk = os.path.join(TVA, "lean", ".lake", "packages")
lemma = "theorem borrowed_result : True := trivial\n"
c_m = local_commit(os.path.join(tree_pk, "mathlib"), os.path.join(root, "lean", ".lake", "packages", "mathlib"),
                   pk["mathlib"]["rev"], "Mathlib/Extra/Borrowed.lean", lemma)
c_b = local_commit(os.path.join(tree_pk, "batteries"), os.path.join(root, "lean", ".lake", "packages", "batteries"),
                   pk["batteries"]["rev"], "Batteries/Extra/Borrowed.lean", lemma)
pk["mathlib"]["rev"], pk["batteries"]["rev"] = c_m, c_b
json.dump(man, open(os.path.join(root, "lean", "lake-manifest.json"), "w"), indent=2)
print(f"local commits: mathlib {c_m[:12]} (upstream pin {json.load(open(os.path.join(TVA, 'lean', 'lake-manifest.json')))['packages'][0]['rev'][:12]}), "
      f"batteries {c_b[:12]}")

tvr.ROOT = root          # the scratch root
json.dump(json.load(open(os.path.join(TVA, "lean", "lake-manifest.json"))), open(os.path.join(root, "lean", "lake-manifest.json"), "w"))
print("P0 lake_violations with the tree's manifest (control):", tvr.lake_violations() or "[] (accepted)")
json.dump(man, open(os.path.join(root, "lean", "lake-manifest.json"), "w"), indent=2)
print("P1 lake_violations with both local commits pinned:", tvr.lake_violations() or "[] (accepted)")

copy_lean = os.path.join(scratch, "copy", "lean")
os.makedirs(copy_lean)
bad = tvr.make_packages(copy_lean)
print("P2 make_packages:", bad or "[] (the packages step passes)")
for name, f in (("mathlib", "Mathlib/Extra/Borrowed.lean"), ("batteries", "Batteries/Extra/Borrowed.lean")):
    p = os.path.join(copy_lean, ".lake", "packages", name, f)
    head = git("rev-parse", "HEAD", cwd=os.path.join(copy_lean, ".lake", "packages", name))
    print(f"   made {name}: HEAD {head[:12]}; {f} present: {os.path.isfile(p)}")

probe = os.path.join(scratch, "Probe.lean")
open(probe, "w").write("import Mathlib.Extra.Borrowed\n\ntheorem t : True := borrowed_result\n")
print("P3 import_violations on `import Mathlib.Extra.Borrowed`:", tvr.import_violations([probe], tvr.LEAN_ROOTS) or "[] (accepted)")

pk["batteries"]["url"] = "https://example.invalid/someone/batteries"
json.dump(man, open(os.path.join(root, "lean", "lake-manifest.json"), "w"), indent=2)
print("P4 lake_violations with an inherited package from another URL:", tvr.lake_violations() or "[] (accepted)")
