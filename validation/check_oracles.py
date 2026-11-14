#!/usr/bin/env python3
"""Check claims/<k>/Claim.lean and its oracle proof against the built library in lean/ (after `lake build TVA TVA.Proof`), without the sandbox.
  check_oracles.py OUT_DIR [KERNEL ...]   (J=jobs, default 6)   (all claims if none given)
Each claim compiles in its own build dir (OUT_DIR/<k>), modules in import order; also checks
that `main` has type `claim.Holds` and reports axioms."""
import os, re, subprocess, sys, json
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAN = os.path.join(ROOT, "lean")
out = sys.argv[1]
ks = sys.argv[2:] or sorted(k for k in os.listdir(os.path.join(ROOT, "claims")) if os.path.isfile(os.path.join(ROOT, "claims", k, "oracle", "Proof.lean")))
base_path = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=LEAN, capture_output=True, text=True).stdout.strip()

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
    deps = {m: [i for i in re.findall(r"^import\s+(\S+)", open(p).read(), re.M) if i in files] for m, p in files.items()}
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
    ns = json.load(open(os.path.join(ROOT, "claims", k, "claim.json")))["namespace"]
    chk = os.path.join(bd, "Check.lean")
    open(chk, "w").write(f"import Proof\nexample : {ns}.claim.Holds := {ns}.main\n#print axioms {ns}.main\n")
    r = subprocess.run(["lean", "-R", bd, chk], cwd=bd, env=env, capture_output=True, text=True)
    ok = r.returncode == 0 and "sorryAx" not in r.stdout
    return k, ok, (r.stdout + r.stderr).strip()

with ThreadPoolExecutor(int(os.environ.get("J", "6"))) as ex:
    for k, ok, msg in ex.map(run, ks):
        print(("OK   " if ok else "FAIL ") + k + ("" if ok else "\n" + msg), flush=True)
        if ok: print("     " + msg.replace("\n", " "), flush=True)
