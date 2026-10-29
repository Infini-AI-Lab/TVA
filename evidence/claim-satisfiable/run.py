"""The claim reader's new shape check: `satisfiable : Claim.Satisfiable Inst.ok Pre`.

Library evidence for library issue L71 (round 11). Before round 11 a claim whose conditions cover
no instance (`Inst.ok _ := False`) passed the gate's `shape`, `axioms`, `usage`, `namespace` and
`kinds` checks with a real `consistent`, since `Claim.Meets` then holds of every behaviour. The
claim reader (`tools/claim_reader.lean.in`) now also elaborates
`example : TVA.Claim.Satisfiable NS.Inst.ok NS.Pre := NS.satisfiable` and reports the axioms of
`satisfiable`; the gate's `shape` check fails when the reader does not elaborate.

This script writes two small claims into a temporary folder, compiles each against the built
library (`lake env lean`), fills in the claim reader for it as `tools/tvr.py` does
(`claim_reader_text`), and runs it:

  S1  a claim with `Inst.ok d := d.n = 1` and `satisfiable := ⟨⟨1⟩, …⟩`: the reader prints one
      result, with the axioms of `main`, `consistent` and `satisfiable`
  S2  the same claim with `Inst.ok _ := False` and no `satisfiable` (none can be proved): the reader
      does not elaborate, so the gate's `shape` check fails

Run: python3 -I evidence/claim-satisfiable/run.py  (from TVA's root; needs the built library,
`cd lean && lake build TVA`)
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEAN_DIR = os.path.join(ROOT, "lean")
LAKE = os.path.expanduser("~/.elan/bin/lake")

CLAIM = """import TVA

namespace {ns}
open TVA

structure Inst where
  n : Nat

def Inst.ok (d : Inst) : Prop := {ok}
def Pre (_ : Inst) (_ : Mem) : Prop := True
def Post (_ : Inst) (_ _ : Mem) : Prop := True
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] (a stand-in for this test) -/
  spec : Spec K

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=
  fun _ _ h => h.spec

theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ init o => o = .ok init, Platform.any, fun _ init => ⟨.ok init, rfl⟩,
    ⟨fun _ _ _ _ _ h => ⟨_, h, trivial⟩⟩⟩
{sat}
end {ns}
"""


def main():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import tvr  # the tool's own template filler
    env_lp = subprocess.run([LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True,
                            text=True).stdout.strip()
    cases = {
        "S1": ("RSat1", "d.n = 1", "theorem satisfiable : Claim.Satisfiable Inst.ok Pre :=\n"
                                   "  ⟨⟨1⟩, fun _ => none, rfl, trivial⟩\n"),
        "S2": ("RSat2", "False", ""),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, (ns, ok, sat) in cases.items():
            src = os.path.join(tmp, ns + ".lean")
            with open(src, "w", encoding="utf-8") as f:
                f.write(CLAIM.format(ns=ns, ok=ok, sat=sat))
            c = subprocess.run([LAKE, "env", "lean", "--root=" + tmp, "-o", os.path.join(tmp, ns + ".olean"), src],
                               cwd=LEAN_DIR, capture_output=True, text=True)
            reader = os.path.join(tmp, "reader_" + ns + ".lean")
            with open(reader, "w", encoding="utf-8") as f:
                f.write(tvr.claim_reader_text(ns, ns, {"TVA"}))
            r = subprocess.run([LAKE, "env", "env", "LEAN_PATH=" + tmp + ":" + env_lp, "lean", reader],
                               cwd=LEAN_DIR, capture_output=True, text=True)
            js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
            errs = [l.split("error", 1)[1][:140] for l in r.stdout.splitlines() if ": error" in l]
            print(f"{name}: Inst.ok := {ok}; satisfiable {'given' if sat else 'absent'}; the claim compiles: "
                  f"{c.returncode == 0}")
            if r.returncode == 0 and len(js) == 1:
                ax = json.loads(js[0][len("TVR_JSON "):])["axioms"]
                print(f"  the reader elaborates and prints one result; axioms {ax}")
            else:
                print(f"  the reader does not elaborate (exit {r.returncode}): the gate's shape check fails; "
                      f"error{errs[0] if errs else ''}")


if __name__ == "__main__":
    main()
