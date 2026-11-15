#!/usr/bin/env python3
"""Run the validation tests, and report their verdicts.

  run_test.py [--build DIR] TEST ...
  run_test.py [--build DIR] --ops CASES.json (DEVICE.json | --self) RESULTS.json

The first form compiles validation/{Solve,Solvers,Exec,Verdict}.lean against the built library
(lean/, after `lake build`; or the one in $TVA_LEAN, a built lean/ directory elsewhere) into DIR
(default /tmp/tva-validation), then for each TEST the claim claims/TEST/Claim.lean (if there is
one) and validation/tests/TEST.lean, whose `#guard`s are the test. A test may print verdict lines
(`Verdict.line`: `verdict {"test": ..., "result": {"verdict": ...}}`).

The second form runs ops/Ops.lean (single operations against a device's outputs, or the model's
own with --self) and summarises RESULTS.json: the verdict of each case, and the mutants not
caught.

Verdicts (validation/Verdict.lean): certified and conditional are agreements (the second resting
on assumed platform answers); discrepancy is a candidate finding; inconclusive and unsupported say
nothing about the model. The exit status is 1 if a test fails to compile or a `#guard` fails, or
if some verdict is a discrepancy; 0 otherwise. Standard library only."""
import collections, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEAN = os.environ.get("TVA_LEAN") or os.path.join(ROOT, "lean")
LIB = ["Solve", "Solvers", "Exec", "Verdict"]  # each may import the ones before it

args = sys.argv[1:]
build = "/tmp/tva-validation"
if args[:1] == ["--build"]:
    build, args = args[1], args[2:]
if not args:
    sys.exit(__doc__)
base = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=LEAN, capture_output=True, text=True, check=True).stdout.strip()
lib_mtime = os.path.getmtime(os.path.join(LEAN, ".lake", "build", "lib", "lean", "TVA.olean"))
libdir = os.path.join(build, "_lib")
os.makedirs(libdir, exist_ok=True)

def env(*dirs):
    return dict(os.environ, LEAN_PATH=":".join(list(dirs) + [libdir, base]))

def lean(root, src, out=None, after=lib_mtime, dirs=()):
    """Compile `src` unless `out` is newer than it and than `after` (its imports' oleans).
    Returns the olean's mtime, and the output (None on failure)."""
    if out and os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(src), after):
        return os.path.getmtime(out), ""
    cmd = ["lean", "-R", root] + (["-o", out] if out else []) + [src]
    r = subprocess.run(cmd, cwd=root, env=env(*dirs), capture_output=True, text=True)
    text = r.stdout + r.stderr
    if r.returncode != 0:
        sys.stdout.write(text)
        print(f"run_test.py: {os.path.relpath(src, ROOT)} failed")
        return 0, None
    return (os.path.getmtime(out) if out else 0), text

dep = lib_mtime
for m in LIB:
    dep, text = lean(HERE, os.path.join(HERE, m + ".lean"), os.path.join(libdir, m + ".olean"), dep)
    if text is None:
        sys.exit(1)

def verdicts(text):
    """The verdict lines a test printed."""
    out = []
    for line in text.splitlines():
        if line.startswith("verdict "):
            out.append(json.loads(line[len("verdict "):]))
    return out

def run_tests(tests):
    failed, found = [], []
    for k in tests:
        kd = os.path.join(build, k)
        os.makedirs(kd, exist_ok=True)
        claim = os.path.join(ROOT, "claims", k, "Claim.lean")
        after = dep
        if os.path.exists(claim):
            after, text = lean(os.path.dirname(claim), claim, os.path.join(kd, "Claim.olean"), dirs=(kd,))
            if text is None:
                failed.append(k); continue
            after = max(after, dep)
        _, text = lean(os.path.join(HERE, "tests"), os.path.join(HERE, "tests", k + ".lean"), dirs=(kd,))
        if text is None:
            failed.append(k); continue
        vs = verdicts(text)
        found += vs
        print(f"{k}: ok" + "".join(f"; {v['test']}: {v['result']['verdict']}" for v in vs))
    return failed, found

def summarise(vs):
    """Print the count of each verdict, and the discrepancies. True if there is none."""
    count = collections.Counter(v["result"]["verdict"] for v in vs)
    print("verdicts: " + (", ".join(f"{n} {k}" for k, n in sorted(count.items())) or "none"))
    for v in vs:
        if v["result"]["verdict"] == "discrepancy":
            print(f"  discrepancy: {v['test']}: {v['result'].get('what', '')}")
    return count["discrepancy"] == 0

def run_ops(cases, device, out):
    r = subprocess.run(["lean", "--run", os.path.join(HERE, "ops", "Ops.lean"), cases, device, out],
                       cwd=os.path.join(HERE, "ops"), env=env(), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout + r.stderr)
        sys.exit("run_test.py: Ops.lean failed")
    res = json.load(open(out))
    vs = [{"test": c["id"], "result": c["result"]} for c in res["cases"] if "result" in c]
    other = collections.Counter(c["status"] for c in res["cases"] if "result" not in c)
    if other:
        print("not run: " + ", ".join(f"{n} {k}" for k, n in sorted(other.items())))
    for c in res["cases"]:
        if c.get("status") in ("inconclusive", "unsupported"):
            print(f"  {c['status']}: {c['id']}: {c['result'].get('what', '')}")
    missed = [m for m in res["mutants"] if not m["caught"]]
    print(f"mutants: {len(res['mutants']) - len(missed)} of {len(res['mutants'])} caught" +
          "".join(f"\n  not caught: {m['id']}: {m['mutant']} ({m['status']})" for m in missed))
    return summarise(vs)

if args[0] == "--ops":
    if len(args) != 4:
        sys.exit(__doc__)
    sys.exit(0 if run_ops(*args[1:]) else 1)
failed, vs = run_tests(args)
ok = summarise(vs)
if failed:
    print("failed: " + ", ".join(failed))
sys.exit(0 if ok and not failed else 1)
