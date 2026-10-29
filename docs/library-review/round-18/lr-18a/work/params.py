"""lr-18a, round 18 (library issue L94 and the claim's shape): which `Assumptions` the claim reader accepts.

Each case is a small kernel compiled with `lean -o` (never `lake build`) into SCRATCH against the library's built
modules, then read with the tree's claim reader (all four `example`s, as the gate runs it, `lake env lean` exit code
and its one `TVR_JSON` line), then the gate's axioms, usage, namespace and kinds checks on the reader's output, as
`tvr check` computes them. The fields are evidence/tool-round18's Part K fields (lr-16a's): a `[kernel]` field that
is the specification itself and one that is a device fact.

Cases:
- control: `structure Assumptions (K : Claim.Behaviour Inst) (P : Platform)`;
- K with a default value (`optParam`): the places of K and P are right, so the flags should be;
- K and P as `variable`s: the same structure;
- an `inductive Assumptions (K …) (P …) : Prop` that is not a structure, whose one constructor holds the
  specification, used by `main` directly (destructured) and through a `private` lemma.

Usage: python3 -I params.py SCRATCH > params.out
"""
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True,
                         text=True).stdout.strip()
    return OLEAN + os.pathsep + pre


def compile_and_read(name, text):
    f = os.path.join(LDIR, name + ".lean")
    os.makedirs(OLEAN, exist_ok=True)
    open(f, "w").write(text)
    env = dict(os.environ, LEAN_PATH=lean_path())
    r = subprocess.run([LEAN, f"--root={LDIR}", "-o", os.path.join(OLEAN, name + ".olean"), f], cwd=LEAN_DIR,
                       capture_output=True, text=True, env=env)
    if r.returncode:
        errs = [l for l in (r.stdout + r.stderr).splitlines() if "error" in l]
        return None, "the kernel does not compile: " + (errs[0] if errs else (r.stdout + r.stderr)[:300])
    reader = os.path.join(LDIR, f"reader{name}.lean")
    open(reader, "w").write(tvr.claim_reader_text(name, "Kernels.Demo", set()))
    r = subprocess.run([LEAN, reader], cwd=LEAN_DIR, capture_output=True, text=True, env=env)
    js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
    other = [l[:160] for l in (r.stdout + r.stderr).splitlines() if not l.startswith("TVR_JSON ")]
    note = f"reader exit code {r.returncode}, {len(js)} TVR_JSON line(s); other output: {other[:3] or 'none'}"
    if r.returncode or len(js) != 1:
        return None, "shape FAIL: " + note
    return json.loads(js[0][len("TVR_JSON "):]), "shape PASS (as `tvr check` judges it: exit code 0, one result line); " + note


def gate_lean_checks(surf):
    consts = tvr.const_table(surf)
    out = {}
    out["axioms"] = all(set(v) <= tvr.STD_AXIOMS for v in surf["axioms"].values()) or surf["axioms"]
    unused = [f for f, u in surf["usage"].items() if not u]
    out["usage"] = (not unused and not surf["destructured"]) or {"unused": unused, "destructured": surf["destructured"]}
    bad_ns = surf["shadow"] + [i["name"] for i in surf["instances"] if not i.get("internal")] + tvr.binder_problems(surf)
    out["namespace"] = bad_ns or True
    out["kinds"] = tvr.kinds_problems(surf, consts, "Kernels.Demo") or True
    return out


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

/-- A platform whose `/` on `float32` returns its dividend. -/
def dividend : Platform := { Platform.any with div := fun _ x _ r => r = x }

"""
FIELDS = """  /-- [kernel] The kernel meets its specification. -/
  spec : Claim.Meets Inst.ok Pre Post K
  /-- [kernel] `/` on `float32` returns its dividend (a device fact, false on the H200). -/
  div : ∀ x y r : FloatV .binary32, P.div .binary32 x y r → r = x
"""
STRUCT_TAIL = """
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=
  fun _ _ h => (fun _ => h.spec) h.div

theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ init o => o = .ok init, dividend, fun _ init => ⟨.ok init, rfl⟩,
    ⟨fun _ _ init _ o h => ⟨init, h, rfl⟩, fun _ _ _ h => h⟩⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩

end Kernels.Demo
"""
INDUCTIVE = """/-- Not a structure: one constructor, which holds the specification itself. -/
inductive Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  | intro : Claim.Meets Inst.ok Pre Post K → Assumptions K P
"""
IND_TAIL = """
{main}

theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ init o => o = .ok init, Platform.any, fun _ init => ⟨.ok init, rfl⟩,
    .intro (fun _ _ init _ o h => ⟨init, h, rfl⟩)⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩

end Kernels.Demo
"""
MAIN_DIRECT = """theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=
  fun _ _ h => match h with | .intro s => s"""
MAIN_PRIVATE = """private theorem fromAssumptions {K : Claim.Behaviour Inst} {P : Platform} (h : Assumptions K P) : Spec K :=
  match h with | .intro s => s

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=
  fun _ _ h => fromAssumptions h"""

CASES = [
    ("Plain", "control: `structure Assumptions (K : Claim.Behaviour Inst) (P : Platform)`",
     HEAD + "structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where\n" + FIELDS + STRUCT_TAIL),
    ("OptK", "K with a default: `structure Assumptions (K : Claim.Behaviour Inst := fun _ _ _ => True) (P : Platform)`",
     HEAD + "structure Assumptions (K : Claim.Behaviour Inst := fun _ _ _ => True) (P : Platform) : Prop where\n"
     + FIELDS + STRUCT_TAIL),
    ("Vars", "K and P as `variable (K : Claim.Behaviour Inst) (P : Platform)`, then `structure Assumptions : Prop`",
     HEAD + "variable (K : Claim.Behaviour Inst) (P : Platform)\nstructure Assumptions : Prop where\n" + FIELDS
     + STRUCT_TAIL),
    ("IndDirect", "an `inductive Assumptions (K …) (P …) : Prop` (not a structure) holding the specification; "
     "`main` matches on it", HEAD + INDUCTIVE + IND_TAIL.format(main=MAIN_DIRECT)),
    ("IndPrivate", "the same inductive; `main` applies a `private` lemma that matches on it",
     HEAD + INDUCTIVE + IND_TAIL.format(main=MAIN_PRIVATE)),
]


if __name__ == "__main__":
    print(f"params.py (lr-18a): tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}\n")
    for name, label, text in CASES:
        surf, why = compile_and_read(name, text)
        show(label, why)
        if not surf:
            continue
        show("  fields the reader lists", surf["fields"])
        for f in surf["fields"]:
            u = surf["field_uses"][f]
            show(f"  field {f}: flags", {"K": u["K"], "P": u["P"], "head": u["head"]})
        for k, v in gate_lean_checks(surf).items():
            show(f"  {k}", "PASS" if v is True else f"FAIL {v}")
        print()
