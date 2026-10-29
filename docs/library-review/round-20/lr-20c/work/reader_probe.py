"""lr-20c, round 20 (L106, gate survey): the claim reader and the gate's shape and kinds decisions on claims the
round's evidence did not try. Each kernel is compiled with `lean -o` into the scratch folder against the
library's built modules (never `lake build`); the tree's claim reader runs on it, and its output goes through
`tvr.reader_result` (the shape step) and `tvr.kinds_problems` / the usage decision, as `tvr check` does.

R1. a `private` field of `Assumptions` that holds the specification ([kernel]): is the field read, or is the
    claim refused?
R2. a `[kernel]` field whose statement is an earlier field (`hq : self.Q`, a field `Q : Prop` before it): does
    the reader see `K`/`P` in it (the projection's parameters)?
R3. a field docstring holding the word PANIC (a usability check: the shape step reads every output line).

Usage: python3 -I reader_probe.py SCRATCH   (SCRATCH: an empty folder outside TVA; its path is never printed)
"""
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = HERE
for _ in range(5):
    TVA = os.path.dirname(TVA)
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH)
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")

HEAD = """import TVA

namespace Kernels.Demo
open TVA

structure Inst where
  n : Nat
def Inst.ok (_ : Inst) : Prop := True
def model (_ : Inst) : Kernel := { body := fun _ => pure () }
def launch (_ : Inst) : Launch := Launch.grid1 1 [[1]]
def Pre (_ : Inst) (_ : Mem) : Prop := True
/-- Memory is unchanged. -/
def Post (_ : Inst) (init fin : Mem) : Prop := fin = init
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

"""
TAIL = """
theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ init o => o = .ok init, Platform.any, fun _ init => ⟨.ok init, rfl⟩,
    PROOF⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩

end Kernels.Demo
"""
MEETS = "(fun _ _ init _ o h => ⟨init, h, rfl⟩)"
CASES = [
    ("Priv", "R1: a private field holding the specification",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] The kernel meets its specification. -/
  private spec : Claim.Meets Inst.ok Pre Post K
""", "fun _ _ h => h.spec", f"⟨{MEETS}⟩"),
    ("Dep", "R2: a [kernel] field that is an earlier field",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] Some proposition. -/
  Q : Prop
  /-- [kernel] It holds. -/
  hq : Q
""", "fun _ _ h => (h.hq.elim : False).elim", "⟨True, trivial⟩"),
    ("Panic", "R3: the word PANIC in a field's docstring",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] A fact whose docstring says PANIC. -/
  t : True
""", "fun _ _ _ => sorry", "⟨trivial⟩"),
]


def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True,
                         text=True).stdout.strip()
    return OLEAN + os.pathsep + pre


def main():
    os.makedirs(OLEAN)
    env = dict(os.environ, LEAN_PATH=lean_path())
    for name, label, decl, main_, proof in CASES:
        print(f"## {label}")
        text = (HEAD + decl + "\ntheorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=\n  "
                + main_ + "\n" + TAIL.replace("PROOF", proof))
        f = os.path.join(LDIR, name + ".lean")
        open(f, "w").write(text)
        r = subprocess.run([LEAN, f"--root={LDIR}", "-o", os.path.join(OLEAN, name + ".olean"), f], cwd=LEAN_DIR,
                           capture_output=True, text=True, env=env)
        errs = [l.split(": error: ", 1)[1] for l in (r.stdout + r.stderr).splitlines() if ": error: " in l]
        if r.returncode:
            print(f"- the kernel does not compile: {errs[:1]}")
            print()
            continue
        print("- the kernel compiles")
        g = os.path.join(LDIR, f"reader{name}.lean")
        open(g, "w").write(tvr.claim_reader_text(name, "Kernels.Demo", set()))
        r2 = subprocess.run([LEAN, g], cwd=LEAN_DIR, capture_output=True, text=True, env=env)
        surf, why = tvr.reader_result(r2.returncode, r2.stdout, r2.stderr)
        print(f"- the shape step: {'PASS' if surf else 'FAIL: ' + why[:160].replace(SCRATCH, '<scratch>')}")
        if surf:
            print(f"- the reader's fields: {surf['fields']}; field_types: {json.dumps(surf['field_types'], ensure_ascii=False)}")
            print(f"- field_uses (K, P): {[(k, v['K'], v['P']) for k, v in surf['field_uses'].items()]}")
            consts = tvr.const_table(surf)
            print(f"- kinds: {tvr.kinds_problems(surf, consts, 'Kernels.Demo') or 'PASS'}")
            unused = [k for k, u in surf['usage'].items() if not u]
            print(f"- usage: {'unused: ' + ', '.join(unused) if unused else 'PASS'}; axioms of main: {surf['axioms']['main']}")
        print()


main()
