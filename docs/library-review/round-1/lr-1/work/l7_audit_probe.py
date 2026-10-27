"""lr-1, library review round 1, L7: does `tvr audit` catch each kind of reference it promises to catch?

Usage: python3 -I l7_audit_probe.py NAME SCRATCH
  NAME     the other project's name (one whose sha256 is in tvr's FOREIGN set); it is read from the
           command line so that this file does not contain it
  SCRATCH  an empty scratch directory outside TVA

For each probe, a fresh copy of the TVA tree (without .venv, .lake, .git) is made in SCRATCH, one file
is mutated, and the copy's own `tools/tvr.py audit` is run. The output names each probe and whether
the audit flagged it. No probe text is printed, so the output names no project and no path.
"""
import hashlib
import os
import shutil
import subprocess
import sys

TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))))
NAME, SCRATCH = sys.argv[1], sys.argv[2]
HOME = "~"  # joined at run time, so that this file holds no home path
SEP = "/"


def fresh():
    dst = os.path.join(SCRATCH, "tva")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(TVA, dst, ignore=shutil.ignore_patterns(".venv", ".lake", ".git", "__pycache__"))
    return dst


def audit(root):
    r = subprocess.run([sys.executable, "-I", os.path.join(root, "tools", "tvr.py"), "audit"],
                       capture_output=True, text=True, cwd=root)
    return r.returncode, r.stdout.strip().splitlines()


def append(root, relpath, text):
    p = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(text)


def prepend(root, relpath, text):
    p = os.path.join(root, relpath)
    old = open(p).read()
    with open(p, "w") as f:
        f.write(text + old)


LIB = "lean/TVA/Claim.lean"
DOC = "docs/SEMANTICS.md"
PROBES = [
    # (id, what, should be caught by the rule, mutation)
    ("I1", "library Lean file: plain `import Aesop`", True,
     lambda r: prepend(r, LIB, "import Aesop\n")),
    ("I2", "library Lean file: `public import Aesop` (module system; elaborates in Lean 4.35, see ImportSyntax.lean)", True,
     lambda r: prepend(r, LIB, "module\npublic import Aesop\n")),
    ("I3", "library Lean file: `meta import Aesop`", True,
     lambda r: prepend(r, LIB, "module\nmeta import Aesop\n")),
    ("K1", "lakefile: [[require]] of another package, `name` first", True,
     lambda r: append(r, "lean/lakefile.toml", '\n[[require]]\nname = "other"\npath = "../../other"\n')),
    ("K2", "lakefile: [[require]] of another package, `path` before `name`", True,
     lambda r: append(r, "lean/lakefile.toml", '\n[[require]]\npath = "../../other"\nname = "other"\n')),
    ("K3", "lakefile: [[lean_lib]] whose srcDir is outside TVA", True,
     lambda r: append(r, "lean/lakefile.toml", '\n[[lean_lib]]\nname = "Other"\nsrcDir = "../../other/lean"\n')),
    ("K4", "lakefile: a comment naming the other project", True,
     lambda r: append(r, "lean/lakefile.toml", f"\n# see {NAME}\n")),
    ("N1", "docs: the other project's name as a word", True,
     lambda r: append(r, DOC, f"\nAs in {NAME}.\n")),
    ("N2", "docs: the name in upper case", True,
     lambda r: append(r, DOC, f"\nAs in {NAME.upper()}.\n")),
    ("N3", "docs: the name as the prefix of an identifier (name + `Lemmas`)", True,
     lambda r: append(r, DOC, f"\nAs in `{NAME.upper()}Lemmas.foo`.\n")),
    ("N4", "docs: the name joined by an underscore (name + `_v2`)", True,
     lambda r: append(r, DOC, f"\nAs in {NAME}_v2.\n")),
    ("N5", "Lean docstring: the name as a word", True,
     lambda r: append(r, LIB, f"\n/-! From {NAME}. -/\n")),
    ("P1", "docs: a home path into a sibling directory of TVA (a made-up name)", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P2", "docs: a home path into an archive directory (a made-up name)", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "archive" + SEP + "otherproj" + SEP + "x.lean.\n")),
    ("P3", "docs: a home path to a directory whose name starts with `tva` (`tva-old`, a made-up name)", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "tva-old" + SEP + "x.lean.\n")),
    ("P4", "docs: a relative path that climbs out of TVA (a made-up target)", True,
     lambda r: append(r, DOC, "\nSee ../../otherproj/lean/X.lean.\n")),
    ("P5", "docs: an absolute path outside /home", True,
     lambda r: append(r, DOC, "\nSee " + SEP + "scratch" + SEP + "other" + SEP + "x.lean.\n")),
    ("F1", "evidence: the name inside a `.log` output file", True,
     lambda r: append(r, "evidence/hints-dry-run/run.log", f"{NAME}\n")),
    ("F2", "evidence: the name inside a `.csv` output file", True,
     lambda r: append(r, "evidence/hints-dry-run/data.csv", f"{NAME},1\n")),
    ("F3", "evidence: the name inside a `.sh` script", True,
     lambda r: append(r, "evidence/hints-dry-run/run.sh", f"# {NAME}\n")),
    ("F4", "top-level kernels/README.md: the name", True,
     lambda r: append(r, "kernels/README.md", f"\n{NAME}\n")),
]


def main():
    assert hashlib.sha256(NAME.lower().encode()).hexdigest() in open(os.path.join(TVA, "tools", "tvr.py")).read(), \
        "NAME must be one of tvr's hashed names"
    os.makedirs(SCRATCH, exist_ok=True)
    root = fresh()
    code, out = audit(root)
    print(f"baseline: exit {code}, {'independent' if code == 0 else 'flagged'}")
    misses = 0
    for pid, what, should, mutate in PROBES:
        root = fresh()
        mutate(root)
        code, out = audit(root)
        caught = code != 0
        if should and not caught:
            misses += 1
        print(f"{pid}: {'CAUGHT' if caught else 'MISSED'} - {what}")
    print(f"{misses} of {len(PROBES)} probes missed")


if __name__ == "__main__":
    main()
