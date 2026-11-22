#!/usr/bin/env python3
"""Check claims/<k>/Claim.lean and its oracle proof against the built library in lean/ (after `lake build TVA TVA.Proof`), without the sandbox.
  check_oracles.py OUT_DIR [KERNEL ...]   (J=jobs, default 6)   (all claims if none given)
Each claim compiles in its own build dir (OUT_DIR/<k>), modules in import order; also checks
that `main` has type `claim.Holds` and uses only the standard axioms. Exits 1 if any claim fails."""
import os, re, subprocess, sys, json
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAN = os.path.join(ROOT, "lean")
# `lean` runs outside lean/, where elan would not see lean-toolchain: pin it here.
with open(os.path.join(LEAN, "lean-toolchain")) as f:
    os.environ.setdefault("ELAN_TOOLCHAIN", f.read().strip())
AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
if len(sys.argv) < 2: sys.exit(__doc__)
out = sys.argv[1]
ks = sys.argv[2:] or sorted(k for k in os.listdir(os.path.join(ROOT, "claims")) if os.path.isfile(os.path.join(ROOT, "claims", k, "oracle", "Proof.lean")))
lp = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=LEAN, capture_output=True, text=True)
if lp.returncode != 0 or not lp.stdout.strip(): sys.exit("check_oracles.py: `lake env` failed in lean/ (run `lake build TVA TVA.Proof` first)\n" + lp.stderr)
base_path = lp.stdout.strip()

def read(p):
    with open(p) as f: return f.read()

def mods(k):
    src = os.path.join(ROOT, "claims", k, "oracle")
    files = {"Claim": os.path.join(ROOT, "claims", k, "Claim.lean"), "Proof": os.path.join(src, "Proof.lean")}
    pd = os.path.join(src, "Proof")
    for dp, _, fs in os.walk(pd):
        for f in fs:
            if f.endswith(".lean"):
                rel = os.path.relpath(os.path.join(dp, f), src)[:-5].replace(os.sep, ".")
                files[rel] = os.path.join(dp, f)
    return files

def run(k):
    files = mods(k)
    deps = {m: [i for i in re.findall(r"^import\s+(\S+)", read(p), re.M) if i in files] for m, p in files.items()}
    order, seen = [], set()
    def visit(m):
        if m in seen: return
        seen.add(m)
        for d in deps[m]: visit(d)
        order.append(m)
    for m in files: visit(m)
    bd = os.path.join(out, k)
    os.makedirs(bd, exist_ok=True)
    env = dict(os.environ, LEAN_PATH=bd + ":" + base_path)
    for m in order:
        ol = os.path.join(bd, *m.split(".")) + ".olean"
        os.makedirs(os.path.dirname(ol), exist_ok=True)
        root = os.path.join(ROOT, "claims", k) if m == "Claim" else os.path.join(ROOT, "claims", k, "oracle")
        r = subprocess.run(["lean", "-R", root, "-o", ol, files[m]], cwd=root, env=env, capture_output=True, text=True)
        errs = [l for l in (r.stdout + r.stderr).splitlines() if ": error" in l or l.startswith("error")]
        if r.returncode != 0 or errs:
            return k, False, f"{m}: " + "\n".join((r.stdout + r.stderr).splitlines()[:40])
    ns = json.loads(read(os.path.join(ROOT, "claims", k, "claim.json")))["namespace"]
    chk = os.path.join(bd, "Check.lean")
    with open(chk, "w") as f:
        f.write(f"import Proof\nexample : {ns}.claim.Holds := {ns}.main\n#print axioms {ns}.main\n")
    r = subprocess.run(["lean", "-R", bd, chk], cwd=bd, env=env, capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip()
    m = re.search(r"depends on axioms: \[([^\]]*)\]", r.stdout)
    used = {a.strip() for a in m.group(1).split(",")} if m else set()
    ok = (r.returncode == 0 and "error" not in msg
          and (m is not None or "does not depend on any axioms" in r.stdout) and used <= AXIOMS)
    if r.returncode == 0 and not used <= AXIOMS:
        msg += "\nunexpected axioms: " + ", ".join(sorted(used - AXIOMS))
    return k, ok, msg

failed = []
with ThreadPoolExecutor(int(os.environ.get("J", "6"))) as ex:
    for k, ok, msg in ex.map(run, ks):
        if not ok: failed.append(k)
        print(("OK   " if ok else "FAIL ") + k + ("" if ok else "\n" + msg), flush=True)
        if ok: print("     " + msg.replace("\n", " "), flush=True)
if failed: sys.exit(f"{len(failed)} of {len(ks)} failed: " + ", ".join(failed))
print(f"all {len(ks)} OK")
