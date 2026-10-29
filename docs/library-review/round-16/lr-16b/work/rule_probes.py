"""lr-16b: more `[rule]` fields for the gate's `kinds` check (library issue L62), and the fact page's reader.

Part A: `Compiler.Faithful` fields beyond the round-16 evidence (evidence/tool-round16 Part F): another
platform than `P`, a field over every platform, a conjunction, an existential, a double negation, a
`let`, a beta-redex, a nested binder, `@`-application, a launch with a field updated, `id (model d)`,
and a hypothesis that is itself a `Faithful` of another instance's model.
Part B: facts of docs/rules/num.md cited in further forms (another platform, a default argument, a
bound format, a kernel abbreviation of a listed constant, an `if`).
Part C: the reader of num.md's section "Facts the library states" (`stated_facts`) on scratch copies:
a second table under a `###` heading inside the section, a code span in another column, a row whose
first cell has two spans and text.

The kernels are compiled with `lean` (not `lake build`) into the scratch folder, against the tree's
built library, and read with the tree's claim reader (`tvr.claim_reader_text`), without its
`consistent` example: the fields test `kinds`, not consistency. The tree is not written.

Usage: python3 -I rule_probes.py SCRATCH     (SCRATCH: an empty folder outside TVA; never printed)
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
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True,
                         text=True).stdout.strip()
    os.makedirs(OLEAN, exist_ok=True)
    return OLEAN + os.pathsep + pre


def compile_and_read(name, text):
    f = os.path.join(LDIR, name + ".lean")
    o = os.path.join(OLEAN, name + ".olean")
    os.makedirs(LDIR, exist_ok=True)
    open(f, "w").write(text)
    env = dict(os.environ, LEAN_PATH=lean_path())
    r = subprocess.run([LEAN, f"--root={LDIR}", "-o", o, f], cwd=LEAN_DIR, capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"- {name}: does not compile: {mask(r.stdout + r.stderr)[:2500]}")
        return None
    reader = os.path.join(LDIR, f"reader{name}.lean")
    rt = tvr.claim_reader_text(name, "Kernels.Demo", set())
    rt = "\n".join(l for l in rt.split("\n") if not l.startswith("example : TVA.Claim.Consistent"))
    open(reader, "w").write(rt)
    r = subprocess.run([LEAN, reader], cwd=LEAN_DIR, capture_output=True, text=True, env=env)
    js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
    if r.returncode or len(js) != 1:
        print(f"- {name}: the reader fails: {mask(r.stdout + r.stderr)[:2500]}")
        return None
    return json.loads(js[0][len("TVR_JSON "):])


HEAD = """import TVA
namespace Kernels.Demo
open TVA

structure Inst where
  n : Nat
def Inst.ok (_ : Inst) : Prop := True
def model (_ : Inst) : Kernel := { body := fun _ => pure () }
def launch (_ : Inst) : Launch := Launch.grid1 1 [[1]]
def Pre (_ : Inst) (_ : Mem) : Prop := True
def Post (_ : Inst) (_ _ : Mem) : Prop := True
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K
/-- A kernel abbreviation of a listed constant. -/
abbrev f64 : TVA.Math.Float.Format := .binary64
"""
MAIN = """theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P h
  exact meets_independent h.faithful (fun _ _ => rfl)
    (fun _ _ => Launch.valid_grid1 (by norm_num) (by simp [Shape.TritonOk]; exact ⟨0, rfl⟩))
    (fun _ _ _ => ∅) (fun _ _ _ => ∅) (fun _ _ _ _ => True)
    (fun _ _ _ _ _ => safe_ret trivial) (fun _ _ _ _ _ _ _ => by simp) (fun _ _ _ _ _ => by simp)
    (fun _ _ _ _ _ _ _ => trivial)
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩
end Kernels.Demo
"""
STD = ("faithful", "[rule] `Compiler.Faithful`: the standard form (the control).",
       "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)")

A_FIELDS = [
    STD,
    ("otherP", "[rule] the rule on another platform than `P` (the most accurate one).",
     "∀ d, d.ok → Rules.Compiler.Faithful Platform.sm90ref (model d) (launch d) (K d)"),
    ("everyP", "[rule] the rule for every platform (false: some platform computes exactly).",
     "∀ (Q : Platform) (d : Inst), d.ok → Rules.Compiler.Faithful Q (model d) (launch d) (K d)"),
    ("conj", "[rule] a conjunction.",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d) ∧ True"),
    ("exist", "[rule] an existential.",
     "∃ d : Inst, Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
    ("dneg", "[rule] a double negation.",
     "∀ d, d.ok → ¬¬ Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
    ("letx", "[rule] a `let` of the instance.",
     "∀ d : Inst, let e := d; Rules.Compiler.Faithful P (model e) (launch e) (K e)"),
    ("beta", "[rule] a beta-redex.",
     "∀ d : Inst, (fun x => Rules.Compiler.Faithful P (model x) (launch x) (K x)) d"),
    ("nested", "[rule] the standard form under a second binder of the same type (sound: it is implied by the rule).",
     "∀ d : Inst, d.ok → ∀ e : Inst, Rules.Compiler.Faithful P (model e) (launch e) (K e)"),
    ("explicit", "[rule] the standard form with `@`.",
     "∀ d, d.ok → @Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
    ("updated", "[rule] the launch with a field updated (debug off whatever the launch says).",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model d) { launch d with debug := false } (K d)"),
    ("idModel", "[rule] `id (model d)`.",
     "∀ d, d.ok → Rules.Compiler.Faithful P (id (model d)) (launch d) (K d)"),
    ("hypOther", "[rule] the standard form under a hypothesis about another instance's model (sound: a hypothesis only weakens).",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model ⟨0⟩) (launch d) (K d) → Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
]

B_FIELDS = [
    STD,
    ("expOtherP", "[rule] `ExpF32` on another platform.", "Rules.Num.ExpF32 Platform.sm90ref"),
    ("cvtDefault", "[rule] `CvtTotal` with its default rounding argument.", "Rules.Num.CvtTotal P .binary32 .e4m3"),
    ("mathBound", "[rule] `MathTotal` at every format.", "∀ f : TVA.Math.Float.Format, Rules.Num.MathTotal P .exp f"),
    ("d64abbrev", "[rule] `DivRn` on the kernel's abbreviation of `binary64`.", "Rules.Num.DivRn P f64"),
    ("d64if", "[rule] `DivRn` on an `if` that is `binary64`.",
     "Rules.Num.DivRn P (if True then .binary64 else .binary32)"),
    ("divK", "[rule] a listed fact under a hypothesis that mentions K.",
     "∀ d : Inst, K d = K d → Rules.Num.DivFull P"),
]


def kernel_text(fields):
    body = "".join(f"  /-- {doc} -/\n  {name} : {ty}\n" for name, doc, ty in fields)
    return HEAD + "structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where\n" + body + MAIN


def verdicts(label, fields):
    print(f"# {label}\n")
    surf = compile_and_read("Probe" + label.split()[1].rstrip(":"), kernel_text(fields))
    if not surf:
        print()
        return
    consts = tvr.const_table(surf)
    bad = tvr.kinds_problems(surf, consts, "Kernels.Demo")
    for name, _, ty in fields:
        mine = [x[len(name) + 2:] for x in bad if x.startswith(name + ": ")]
        u = surf["field_uses"][name]
        print(f"- `{name} : {ty}`")
        print(f"  head {u.get('head') or '(none)'}; arguments: " + json.dumps(
            [{k: v for k, v in a.items() if v not in ("", -1, False)} for a in u["args"]], ensure_ascii=False))
        print(f"  kinds: {'; '.join(mine) if mine else 'PASS'}")
    print()


def part_c():
    print("# Part C: the reader of num.md's section \"Facts the library states\"\n")
    root = os.path.join(SCRATCH, "c-root")
    os.makedirs(os.path.join(root, "docs", "rules"))
    page = os.path.join(root, "docs", "rules", "num.md")
    text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
    marker = "Every fact chooses its constants with margin"
    sub = ("### Shapes the facts use (not citable)\n\n| Shape | Says |\n|---|---|\n"
           "| `MathAcc` | an accuracy bound with the field's constants |\n| `DivRn P .binary32` | IEEE division |\n\n")
    variants = [
        ("as in the tree", text),
        ("a `###` subsection with a table of shapes inside the section", text.replace(marker, sub + marker)),
        ("a code span `CvtRn` in the second column of the ExpF32 row",
         text.replace("| `ExpF32` | `tl.exp`, `float32` |", "| `ExpF32` | `tl.exp`, `float32`, unlike `CvtRn` |")),
    ]
    real = tvr.ROOT
    tvr.ROOT = root
    try:
        for label, t in variants:
            open(page, "w", encoding="utf-8").write(t)
            f = tvr.stated_facts() or {}
            names = sorted(n[len("TVA.Rules.Num."):] + ("" if p == [None] else " as " + " / ".join(" ".join(x) for x in p))
                           for n, p in f.items())
            extra = [n for n in names if n.split(" ")[0] in ("MathAcc", "CvtRn") or "binary32" in n]
            print(f"- num.md {label}: {len(f)} facts; shapes among them: {extra or 'none'}")
            u = {"head": "TVA.Rules.Num.MathAcc", "args": [{"platform": True}] + [{}] * 5}
            print(f"  a field `MathAcc P .exp .binary32 Set.univ (2^-24) (2^-150)`: "
                  f"{tvr.rule_fact_problem(u, 'Kernels.Demo', tvr.stated_facts()) or 'PASS'}")
    finally:
        tvr.ROOT = real
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}\n")
    verdicts("Part A: Compiler.Faithful fields", A_FIELDS)
    verdicts("Part B: listed facts in further forms", B_FIELDS)
    part_c()
