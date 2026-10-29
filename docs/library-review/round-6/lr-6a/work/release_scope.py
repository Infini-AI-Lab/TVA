"""lr-6a, round 6: does the release check (`tvr release`) hide everything outside TVA, or only the home and
temporary directories?

The release check copies what TVA ships into a sandbox and, with bwrap, mounts the whole root
read-only (`--ro-bind / /`) and puts an empty tmpfs on the home and temporary directories only.
Every other top-level directory of the machine (var, scratch, data, opt, srv, mnt, ...) stays
readable inside the sandbox.

Method (the same shape as evidence/release-check R2, with the outside file moved from the
temporary directory to var/tmp): copy the TVA tree (with its git metadata; Mathlib's packages linked, as the author's
script does) to a scratch directory outside the home directory; put a file in a fresh directory
under var/tmp (from the root); make the copy's `tvr audit` read that file first, at a path computed at run time
from its components (no path is written in the text); run the copy's `tvr release` and print each
step. If the audit step passes inside the sandbox, the dependency on a file outside TVA was not
caught. The real tree is never written.

Usage: python3 -I release_scope.py  (no arguments; prints to stdout; the scratch directories are removed)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

TVA = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
PY = sys.executable


def copy(dst):
    shutil.copytree(TVA, dst, symlinks=True, ignore=shutil.ignore_patterns(".venv", ".lake", "__pycache__"))
    os.makedirs(os.path.join(dst, "lean", ".lake"))
    os.symlink(os.path.join(TVA, "lean", ".lake", "packages"), os.path.join(dst, "lean", ".lake", "packages"))


def main():
    base = tempfile.mkdtemp(prefix="lr6a-release-", dir=os.sep + os.path.join("var", "tmp"))
    outside = tempfile.mkdtemp(prefix="lr6a-outside-", dir=os.sep + os.path.join("var", "tmp"))
    try:
        open(os.path.join(outside, "data.txt"), "w").write("probe\n")
        c = os.path.join(base, "copy")
        copy(c)
        pieces = [p for p in outside.split(os.sep) if p]
        t = os.path.join(c, "tools", "tvr.py")
        s = open(t).read()
        anchor = "    if len(argv) >= 2 and argv[1] == \"audit\":\n"
        assert anchor in s
        probe = anchor + f"        open(os.sep + os.sep.join({pieces!r}) + os.sep + \"data.txt\").read()  # probe\n"
        open(t, "w").write(s.replace(anchor, probe, 1))
        print(f"outside file: a fresh directory under the top-level directory {pieces[0]!r}, then {pieces[1]!r}")
        a = subprocess.run([PY, "-I", "tools/tvr.py", "audit"], cwd=c, capture_output=True, text=True)
        print(f"outside the sandbox: tvr audit of the copy: {'PASS' if a.returncode == 0 else 'FAIL'}")
        d = os.path.join(base, "sandbox")
        r = subprocess.run([PY, "-I", "tools/tvr.py", "release", "--dir", d], cwd=c, capture_output=True, text=True)
        rep = json.load(open(os.path.join(d, "release.json")))
        print(f"release check: {'PASS' if rep['ok'] else 'FAIL'} (masked by bwrap: {rep['masked']}, {rep['files']} files)")
        for st in rep["steps"]:
            det = st["detail"] if st["ok"] else st["detail"][:160]
            if "Traceback" in det:
                det = "the tool stopped with an exception"
            print(f"  {'PASS' if st['ok'] else 'FAIL'}  {st['step']}: {det}")
        # what else the sandbox shows: the top-level directories that are neither hidden nor system ones
        system = {"usr", "bin", "sbin", "lib", "lib64", "etc", "dev", "proc", "sys", "home", "tmp", "boot", "run"}
        visible = sorted(x for x in os.listdir(os.sep) if x not in system and os.path.isdir(os.sep + x))
        print("top-level directories bound read-only into the sandbox besides the system ones: " + ", ".join(visible))
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


if __name__ == "__main__":
    main()
