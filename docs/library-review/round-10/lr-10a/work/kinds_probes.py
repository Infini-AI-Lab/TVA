"""lr-10a, round 10 (gate survey): can a `[rule]` assumption carry a constant of the kernel's own?

Protocol §5: a `[rule]` may mention the library's constants, `K`, `P`, and the kernel's `Inst`, `Inst.ok`,
`model`, `launch`, and no other kernel constant; "the gate checks the kinds and these limits, so a kernel
cannot slip its own constants into a library rule". This compiles small kernels with `lean` (not
`lake build`) into SCRATCH, each with a `[rule]` field citing `TVA.Rules.Num.DivAcc` with a relative bound
that the kernel defines itself (`2 ^ (-40)`, 2^18 times tighter than `DivFull`'s), and a real `consistent`
proof (a platform whose division has no result). It runs the tree's claim reader on each and the gate's
own checks of its output (`axioms`, `usage`, `namespace`, `kinds`), as `tvr check` runs them:
  V0  the library's constants written out (control: passes)
  V1  `def tol` in the kernel's namespace (control: `kinds` refuses it)
  V2  `private def tol`
  V3  `def _root_.Kernels._tol` (an internal name outside the namespace)
  V4  `def _root_._tol`
The tree is never written.

Usage: python3 -I work/kinds_probes.py SCRATCH   (SCRATCH: an empty folder; its path is never printed)
"""
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.realpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
NS = "Kernels.Demo"

DEMO = """import TVA

{decl}
namespace Kernels.Demo
open TVA

/-- The model. -/
def model : Nat -> Nat := fun x => x + 1
/-- The launch. -/
def launch : Nat := 0
/-- The instances. -/
structure Inst where
  n : Nat
/-- None is covered. -/
def Inst.ok (_ : Inst) : Prop := False
def Pre (_ : Inst) (_ : Mem) : Prop := True
def Post (_ : Inst) (_ _ : Mem) : Prop := True
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K
structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] the division fact. -/
  div : TVA.Rules.Num.DivAcc P .binary32 (2 ^ 126) {rel} (2 ^ (-148 : Int))
theorem main : forall (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P -> Spec K := by
  intro K P h d hd
  have := h.div
  exact absurd hd id
theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ _ o => o = .hang, {{ Platform.any with div := fun _ _ _ _ => False }}, fun _ _ => ⟨.hang, rfl⟩,
    ⟨by intro x y r h; exact h.elim⟩⟩
end Kernels.Demo
"""
VARIANTS = {
    "V0": ("", "(2 ^ (-22 : Int))"),
    "V1": ("namespace Kernels.Demo\n/-- A bound of the kernel's own. -/\ndef tol : ℚ := 2 ^ (-40 : Int)\nend Kernels.Demo\n",
           "Kernels.Demo.tol"),
    "V2": ("namespace Kernels.Demo\n/-- A bound of the kernel's own. -/\nprivate def tol : ℚ := 2 ^ (-40 : Int)\nend Kernels.Demo\n",
           "tol"),
    "V3": ("/-- A bound of the kernel's own. -/\ndef _root_.Kernels._tol : ℚ := 2 ^ (-40 : Int)\n", "Kernels._tol"),
    "V4": ("/-- A bound of the kernel's own. -/\ndef _root_._tol : ℚ := 2 ^ (-40 : Int)\n", "_tol"),
}


def main():
    lean_dir = os.path.join(TVA, "lean")
    d = os.path.join(SCRATCH, "lean")
    os.makedirs(os.path.join(d, "olean"))
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=lean_dir, capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, LEAN_PATH=pre + os.pathsep + os.path.join(d, "olean"))
    for s, (decl, rel) in VARIANTS.items():
        f = os.path.join(d, f"Kinds{s}.lean")
        text = DEMO.replace("{decl}", decl).replace("{rel}", rel).replace("{{", "{").replace("}}", "}")
        open(f, "w").write(text)
        r = subprocess.run([tvr.LAKE, "env", "lean", f"--root={d}", "-o", os.path.join(d, "olean", f"Kinds{s}.olean"), f],
                           cwd=lean_dir, capture_output=True, text=True)
        if r.returncode:
            print(f"- {s}: does not compile: {(r.stdout + r.stderr).replace(SCRATCH, '<scratch>')[:600]}")
            continue
        reader = os.path.join(d, f"reader{s}.lean")
        open(reader, "w").write(tvr.claim_reader_text(f"Kinds{s}", NS, {"TVA"}))
        r = subprocess.run([tvr.LAKE, "env", "lean", reader], cwd=lean_dir, capture_output=True, text=True, env=env)
        js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
        if r.returncode or len(js) != 1:
            print(f"- {s}: the reader fails: {(r.stdout + r.stderr).replace(SCRATCH, '<scratch>')[:600]}")
            continue
        surf = json.loads(js[0][len("TVR_JSON "):])
        consts = tvr.const_table(surf)
        axioms = all(set(v) <= tvr.STD_AXIOMS for v in surf["axioms"].values())
        usage = all(surf["usage"].values()) and not surf["destructured"]
        bad_ns = (surf["shadow"] + [i["name"] for i in surf["instances"] if not i.get("internal")] +
                  tvr.binder_problems(surf))
        print(f"- {s}: [rule] div : {surf['field_types']['div']}")
        print(f"  constants the statement mentions (in scope): {surf['field_uses']['div']['consts']}")
        print(f"  axioms {'PASS' if axioms else 'FAIL'}; usage {'PASS' if usage else 'FAIL'}; "
              f"namespace {'PASS' if not bad_ns else 'FAIL ' + '; '.join(bad_ns)[:200]}; "
              f"kinds {'PASS' if not tvr.kinds_problems(surf, consts, NS) else 'FAIL ' + '; '.join(tvr.kinds_problems(surf, consts, NS))}")


if __name__ == "__main__":
    main()
