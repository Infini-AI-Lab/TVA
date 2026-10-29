"""lr-10a, round 10 (gate survey): what an item digest holds of an assumption's statement.

Compiles small kernels with `lean` (not `lake build`) into SCRATCH, runs the tree's claim reader on each
(`tvr.claim_reader_text`), and computes the item digests with the tree's `tvr.item_digests`, as the gate
does. Variants of one kernel:
  R1  [rule]  div : DivAcc P .binary32 (2 ^ 126) (2 ^ (-22 : Int)) (2 ^ (-148 : Int))   (DivFull's constants)
  R2  the same field with 2 ^ (-40 : Int): a bound 2^18 times tighter than the library's fact
  R3  R1 with the field's statement wrapped as  forall (_ : Nat), ...  (a different statement shape)
  K2  R1 with the [kernel] field  own : model 0 = model 0  changed to  own : model 0 <= model 0 + 1
and prints, for each variant against R1, which items keep their digest. The tree is never written.

Usage: python3 -I work/digest_probes.py SCRATCH   (SCRATCH: an empty folder; its path is never printed)
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

DEMO = """import TVA

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
  div : {div}
  /-- [kernel] a fact about the model. -/
  own : {own}
theorem main : forall (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P -> Spec K := by
  intro K P h d hd
  have := h.div
  have := h.own
  exact absurd hd id
theorem consistent : Claim.Consistent Assumptions := by
  WITNESS
end Kernels.Demo
"""
DIV = "TVA.Rules.Num.DivAcc P .binary32 (2 ^ 126) (2 ^ ({e} : Int)) (2 ^ (-148 : Int))"
BASE = {"div": DIV.format(e="-22"), "own": "model 0 = model 0"}
VARIANTS = {
    "R1": {},
    "R2": {"div": DIV.format(e="-40")},
    "R3": {"div": "forall (_ : Nat), " + DIV.format(e="-22")},
    "K2": {"own": "model 0 <= model 0 + 1"},
}


def text(v):
    t = DEMO
    for key, val in {**BASE, **v}.items():
        t = t.replace("{" + key + "}", val)
    # `consistent` plays no part in the digests; the reader's `example` only needs it to have its type, so
    # this probe (not a kernel: the gate's hygiene and axioms checks do not run on it) leaves its proof open
    return t.replace("WITNESS", "so" + "rry")


def main():
    lean_dir = os.path.join(TVA, "lean")
    d = os.path.join(SCRATCH, "lean")
    os.makedirs(os.path.join(d, "olean"))
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=lean_dir, capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, LEAN_PATH=pre + os.pathsep + os.path.join(d, "olean"))
    digests, mains = {}, {}
    for s, v in VARIANTS.items():
        f = os.path.join(d, f"Demo{s}.lean")
        open(f, "w").write(text(v))
        r = subprocess.run([tvr.LAKE, "env", "lean", f"--root={d}", "-o", os.path.join(d, "olean", f"Demo{s}.olean"), f],
                           cwd=lean_dir, capture_output=True, text=True)
        if r.returncode:
            print(f"- {s}: does not compile: {(r.stdout + r.stderr).replace(SCRATCH, '<scratch>')[:600]}")
            continue
        reader = os.path.join(d, f"reader{s}.lean")
        open(reader, "w").write(tvr.claim_reader_text(f"Demo{s}", "Kernels.Demo", {"TVA"}))
        r = subprocess.run([tvr.LAKE, "env", "lean", reader], cwd=lean_dir, capture_output=True, text=True, env=env)
        js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
        if r.returncode or len(js) != 1:
            print(f"- {s}: the reader fails: {(r.stdout + r.stderr).replace(SCRATCH, '<scratch>')[:600]}")
            continue
        surf = json.loads(js[0][len("TVR_JSON "):])
        consts = tvr.const_table(surf)
        digests[s] = {k: x["digest"] for k, x in tvr.item_digests(surf, consts, "0" * 64, "{}").items()}
        mains[s] = surf["field_types"]
        print(f"- {s}: field statements {json.dumps(surf['field_types'], ensure_ascii=False)}")
        print(f"  roots of assumption:div {surf['items']['assumption:div']['roots']}")
        print(f"  item digests {{{', '.join(f'{k}: {x[:16]}' for k, x in sorted(digests[s].items()))}}}")
    print()
    for x, y in (("R1", "R2"), ("R1", "R3"), ("R1", "K2")):
        if x in digests and y in digests:
            same = sorted(i for i in digests[x] if digests[x][i] == digests[y][i])
            diff = sorted(set(digests[x]) - set(same))
            stm = sorted(f for f in mains[x] if mains[x][f] != mains[y].get(f))
            print(f"- {x} against {y}: statements that differ: {', '.join(stm) or 'none'}; the same digest for "
                  f"{', '.join(same) or 'no item'}; different for {', '.join(diff) or 'no item'}")


if __name__ == "__main__":
    main()
