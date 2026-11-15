#!/usr/bin/env python3
"""Run a test of validation/tests/ against a claim's model.

  run_test.py KERNEL [BUILD_DIR]

compiles claims/KERNEL/Claim.lean and validation/{Solve,Solvers,Exec}.lean against the built
library (lean/, after `lake build`) into BUILD_DIR (default /tmp/tva-validation), then elaborates
validation/tests/KERNEL.lean, whose `#guard`s are the test. Standard library only."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEAN = os.path.join(ROOT, "lean")
k = sys.argv[1]
bd = os.path.join(sys.argv[2] if len(sys.argv) > 2 else "/tmp/tva-validation", k)
os.makedirs(bd, exist_ok=True)
base = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=LEAN, capture_output=True, text=True, check=True).stdout.strip()
lib_mtime = os.path.getmtime(os.path.join(LEAN, ".lake", "build", "lib", "lean", "TVA.olean"))
env = dict(os.environ, LEAN_PATH=bd + ":" + base)
def lean(root, src, out=None, after=lib_mtime):
    """Compile `src` unless `out` is newer than it and than `after` (its imports' oleans).
    Returns the olean's mtime."""
    if out and os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(src), after):
        return os.path.getmtime(out)
    cmd = ["lean", "-R", root] + (["-o", out] if out else []) + [src]
    r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"run_test.py: {os.path.relpath(src, ROOT)} failed")
    return os.path.getmtime(out) if out else 0
lean(os.path.join(ROOT, "claims", k), os.path.join(ROOT, "claims", k, "Claim.lean"), os.path.join(bd, "Claim.olean"))
dep = lib_mtime
for m in ["Solve", "Solvers", "Exec"]:  # each may import the ones before it
    dep = lean(HERE, os.path.join(HERE, m + ".lean"), os.path.join(bd, m + ".olean"), dep)
lean(os.path.join(HERE, "tests"), os.path.join(HERE, "tests", k + ".lean"))
print(f"{k}: ok")
