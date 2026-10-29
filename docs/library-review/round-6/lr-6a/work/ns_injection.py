"""lr-6a, round 6 (survey area `gate`): can a kernel's manifest change what the claim reader reports?

`tvr check` builds the claim reader from tools/claim_reader.lean.in by replacing `@NS@` with the
`lean.namespace` string of the kernel's `kernel.json` (written by the prover), unchecked, and then
trusts the first output line that starts with `TVR_JSON `. Nothing reads `kernel.json` for Lean
syntax: the hygiene list reads only the kernel's `.lean` files.

Method, on a copy of the tree in a scratch directory outside the home directory (the real tree is
never written):
  A  the copy's `vec_add_f32` brought up to date as in evidence/gate-digests (its `update_kernel`);
     the gate runs; the honest claim reader's JSON line is saved.
  B  the kernel's claim made wrong: `main` and `consistent` renamed `main_honest` and
     `consistent_honest` (their proofs kept), and new `theorem main : True` and
     `theorem consistent : True` added. The gate runs with the manifest as it is (control).
  C  as B (B's gate.json removed, see below), with `lean.namespace` in kernel.json set to a string that closes the claim reader's first
     `example`, prints A's JSON line, and ends the file with `#exit`. The gate runs.
It prints each gate's checks, and for C the first lines of claim.md section 1.

Usage: python3 -I ns_injection.py  (no arguments; prints to stdout; the scratch directory is removed)
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

TVA = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5)))
LAKE = os.path.expanduser(os.path.join("~", ".elan", "bin", "lake"))
spec = importlib.util.spec_from_file_location("gate_digests", os.path.join(TVA, "evidence", "gate-digests", "run.py"))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def gate(root, name):
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "check", "vec_add_f32"], cwd=root,
                       capture_output=True, text=True)
    g = json.load(open(os.path.join(root, "kernels", "vec_add_f32", "gate.json")))
    print(f"## {name}: gate {'PASS' if g['ok'] else 'FAIL'}")
    for c in g["checks"]:
        print(f"  {'pass' if c['ok'] else 'FAIL'}  {c['name']}: {c['detail'][:150]}")
    return g


def main():
    tmp = tempfile.mkdtemp(prefix="lr6a-ns-", dir=os.sep + os.path.join("var", "tmp"))
    root = os.path.join(tmp, "tva")
    try:
        gd.copy_tree(root)
        gd.update_kernel(root)
        gate(root, "A (the kernel updated in the copy)")
        reader = os.path.join(root, "lean", ".lake", "tvr", "vec_add_f32", "claim_reader.lean")
        r = subprocess.run([LAKE, "env", "lean", reader], cwd=os.path.join(root, "lean"), capture_output=True, text=True)
        line = next(l for l in r.stdout.splitlines() if l.startswith("TVR_JSON "))
        print(f"  saved the honest claim reader's JSON line ({len(line)} characters)\n")

        lean = os.path.join(root, "kernels", "vec_add_f32", "lean", "VecAdd.lean")
        gd.edit(lean, "theorem main : ∀ (K : Claim.Behaviour Inst)", "theorem main_honest : ∀ (K : Claim.Behaviour Inst)")
        gd.edit(lean, "theorem consistent : Claim.Consistent Assumptions", "theorem consistent_honest : Claim.Consistent Assumptions")
        gd.edit(lean, "end Proof\n", "/-- The wrong claim. -/\ntheorem main : True := trivial\n\n"
                "/-- The wrong consistency. -/\ntheorem consistent : True := trivial\n\nend Proof\n")
        gate(root, "B (main : True, manifest unchanged)")
        print()

        # B's failing gate wrote the claim reader's error, with the scratch copy's absolute path, into
        # gate.json, which the next gate's text scan would flag (an artifact of running outside the
        # home directory); remove it before C
        os.remove(os.path.join(root, "kernels", "vec_add_f32", "gate.json"))
        man_path = os.path.join(root, "kernels", "vec_add_f32", "kernel.json")
        man = json.load(open(man_path))
        lit = line.replace("\\", "\\\\").replace('"', '\\"')
        man["lean"]["namespace"] = f"Nat), True := fun _ => trivial\n#eval IO.println \"{lit}\"\n#exit\n--"
        json.dump(man, open(man_path, "w"), indent=2, ensure_ascii=False)
        gate(root, "C (main : True, kernel.json namespace replaced)")
        claim = open(os.path.join(root, "kernels", "vec_add_f32", "claim.md")).read()
        sec = claim.split("## 1. The claim", 1)[1].split("## 2.", 1)[0]
        shown = [l if len(l) < 140 else l[:140] + " …" for l in sec.strip().splitlines()]
        print("  claim.md section 1, as a reviewer reads it:")
        for l in shown[:8]:
            print("    | " + l)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
