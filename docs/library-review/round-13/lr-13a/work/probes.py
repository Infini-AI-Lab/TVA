"""lr-13a's probes of the gate, round 13 (survey area `gate`; issues L52, L62, L70).

Usage: python3 -I docs/library-review/round-13/lr-13a/work/probes.py SCRATCH > .../work/probes.out
  SCRATCH  an empty folder outside TVA; its path is never printed.

Part F (L62): which `Compiler.Faithful` fields `kinds` accepts. F1 is a whole claim (the claim reader
with its `consistent` example, as the gate runs it) whose `faithful` field relates every instance's
behaviour to the model of one fixed instance; F2 compiles several such fields for `kinds` only.
Part R (L52): a reviewer's filing, its blocking finding and its assignment removed from a kernel's ledger,
in a scratch copy, with the reviewer's folder left in place.
Part C (L70): citations the check reads as placeholders.
Kernels are compiled with `lean -o` into SCRATCH (never `lake build`); the tree is never written.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
REAL = {k: getattr(tvr, k) for k in ("ROOT", "KROOT", "LREVIEW", "LIBRARY_LEDGER")}
VEC = "vec_add_f32"
LEAN_DIR = os.path.join(TVA, "lean")
OLEAN = os.path.join(SCRATCH, "lean", "olean")


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


def use_root(root):
    if root is None:
        for k, v in REAL.items():
            setattr(tvr, k, v)
        return
    tvr.ROOT, tvr.KROOT = root, os.path.join(root, "kernels")
    tvr.LREVIEW = os.path.join(root, "docs", "library-review")
    tvr.LIBRARY_LEDGER = os.path.join(tvr.LREVIEW, "ledger.jsonl")


def quiet(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, (out.getvalue() + err.getvalue()).strip()


def compile_and_read(name, text, with_consistent=True):
    """Compile `text` as module `name` and run the tree's claim reader on it (with its `consistent`
    example unless `with_consistent` is False). Returns the reader's JSON, or None."""
    d = os.path.join(SCRATCH, "lean")
    os.makedirs(OLEAN, exist_ok=True)
    f = os.path.join(d, f"{name}.lean")
    open(f, "w").write(text)
    r = subprocess.run([tvr.LAKE, "env", "lean", f"--root={d}", "-o", os.path.join(OLEAN, f"{name}.olean"), f],
                       cwd=LEAN_DIR, capture_output=True, text=True)
    if r.returncode:
        print(f"- {name}: does not compile: {mask(r.stdout + r.stderr)[:1500]}")
        return None
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True, text=True).stdout.strip()
    reader = os.path.join(d, f"reader{name}.lean")
    text = tvr.claim_reader_text(name, "Kernels.Demo", set())
    if not with_consistent:
        text = "\n".join(l for l in text.split("\n") if not l.startswith("example : TVA.Claim.Consistent"))
    open(reader, "w").write(text)
    r = subprocess.run([tvr.LAKE, "env", "lean", reader], cwd=LEAN_DIR, capture_output=True, text=True,
                       env=dict(os.environ, LEAN_PATH=pre + os.pathsep + OLEAN))
    js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
    if r.returncode or len(js) != 1:
        print(f"- {name}: the reader fails: {mask(r.stdout + r.stderr)[:1500]}")
        return None
    return json.loads(js[0][len("TVR_JSON "):])


def gate_lean_checks(surf):
    """The gate's checks on the reader's output, as `check` computes them (axioms, usage, namespace, kinds)."""
    consts = tvr.const_table(surf)
    out = {}
    out["axioms"] = all(set(v) <= tvr.STD_AXIOMS for v in surf["axioms"].values()) or surf["axioms"]
    unused = [f for f, u in surf["usage"].items() if not u]
    out["usage"] = (not unused and not surf["destructured"]) or unused
    bad_ns = surf["shadow"] + [i["name"] for i in surf["instances"] if not i.get("internal")] + tvr.binder_problems(surf)
    out["namespace"] = bad_ns or True
    out["kinds"] = tvr.kinds_problems(surf, consts, "Kernels.Demo") or True
    return out


# ---------------------------------------------------------------- Part F

# A whole claim. The instances are `Bool` (`true` is the size the host uses); the model of `true` is the
# kernel the source runs there, and loops forever (a hang, which `Meets` excludes); the model of `false`
# does nothing. The `faithful` field says that the device at every instance behaves as the model of
# `false`: the library's rule (`Faithful P (model d) (launch d) (K d)`) is about the model of the
# instance launched. `main` proves `Post` (memory unchanged) from the field.
F1 = """import TVA

namespace Kernels.Demo
open TVA

/-- The instances: `true` is the one the host launches. -/
abbrev Inst := Bool
/-- Every instance is covered. -/
def Inst.ok (_ : Inst) : Prop := True
/-- What the source runs at `true`: a loop that never ends. -/
def busy : Kernel := { body := fun _ => do let _ ← tl.whileLoop () (fun _ => pure true) pure; pure () }
/-- What the source runs at `false`: nothing. -/
def idle : Kernel := { body := fun _ => pure () }
/-- The model: the source at each instance. -/
def model (d : Inst) : Kernel := match d with | true => busy | false => idle
/-- The launch: one program. -/
def launch (_ : Inst) : Launch := Launch.grid1 1 [[1]]
def Pre (_ : Inst) (_ : Mem) : Prop := True
/-- Memory is unchanged. -/
def Post (_ : Inst) (init fin : Mem) : Prop := fin = init
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful` -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model false) (launch d) (K d)

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P h
  exact meets_independent (model := fun _ => model false) h.faithful (fun _ _ => rfl)
    (fun _ _ => Launch.valid_grid1 (by norm_num) (by simp [Shape.TritonOk]; exact ⟨0, rfl⟩))
    (fun _ _ _ => ∅) (fun _ _ _ => ∅) (fun _ _ _ _ => True)
    (fun _ _ _ _ _ => safe_ret trivial) (fun _ _ _ _ _ _ _ => by simp) (fun _ _ _ _ _ => by simp)
    (fun _ _ _ _ _ _ hfr => funext fun x => hfr x (by simp))

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any (fun _ : Inst => model false) launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨true, fun _ => none, trivial, trivial⟩

end Kernels.Demo
"""

# Fields for `kinds` only, on a structure of instances (the round-13 evidence's kernel).
F2_HEAD = """import TVA

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
"""
F2_MAIN = """theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P h
  exact meets_independent h.faithful (fun _ _ => rfl)
    (fun _ _ => Launch.valid_grid1 (by norm_num) (by simp [Shape.TritonOk]; exact ⟨0, rfl⟩))
    (fun _ _ _ => ∅) (fun _ _ _ => ∅) (fun _ _ _ _ => True)
    (fun _ _ _ _ _ => safe_ret trivial) (fun _ _ _ _ _ _ _ => by simp) (fun _ _ _ _ _ => by simp)
    (fun _ _ _ _ _ _ _ => trivial)
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩
end Kernels.Demo
"""
F2_FIELDS = [
    ("faithful", "[rule] `Compiler.Faithful`: the standard form.",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
    ("otherModel", "[rule] `Compiler.Faithful` with another instance's model.",
     "∀ d e : Inst, d.ok → Rules.Compiler.Faithful P (model e) (launch d) (K d)"),
    ("otherLaunch", "[rule] `Compiler.Faithful` with another instance's launch.",
     "∀ d e : Inst, d.ok → Rules.Compiler.Faithful P (model d) (launch e) (K d)"),
    ("chosen", "[rule] `Compiler.Faithful` with the model of one chosen covered instance.",
     "∀ (h : ∃ d : Inst, d.ok) (d : Inst), d.ok → Rules.Compiler.Faithful P (model (Classical.choose h)) (launch d) (K d)"),
    ("noOk", "[rule] `Compiler.Faithful` at every instance, covered or not.",
     "∀ d, Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
]


def part_f():
    print("# Part F (L62): the arguments of a `Compiler.Faithful` field\n")
    surf = compile_and_read("WrongPair", F1)
    if surf:
        show("F1: a whole claim, `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model false) (launch d) (K d)`",
             "the claim reader elaborates main, consistent, satisfiable and Spec (shape PASS)")
        show("  field as claim.md prints it", surf["field_types"]["faithful"])
        show("  head; arguments", (surf["field_uses"]["faithful"]["head"],
                                    [("P" if a["platform"] else a["const"] or "…") for a in surf["field_uses"]["faithful"]["args"]]))
        for k, v in gate_lean_checks(surf).items():
            show(f"  {k}", "PASS" if v is True else f"FAIL {v}")
    body = "".join(f"  /-- {doc} -/\n  {name} : {ty}\n" for name, doc, ty in F2_FIELDS)
    text = F2_HEAD + "structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where\n" + body + F2_MAIN
    surf = compile_and_read("FaithfulArgs", text, with_consistent=False)
    if surf:
        consts = tvr.const_table(surf)
        bad = tvr.kinds_problems(surf, consts, "Kernels.Demo")
        for name, _, ty in F2_FIELDS:
            mine = [x[len(name) + 2:] for x in bad if x.startswith(name + ": ")]
            show(f"F2 `{name} : {ty}`: kinds", "; ".join(mine) if mine else "PASS")
    print()


# ---------------------------------------------------------------- Part R

def scratch_root(name):
    root = os.path.join(SCRATCH, name)
    ign = shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv")
    for d in ("tools", "docs", "lean"):
        shutil.copytree(os.path.join(TVA, d), os.path.join(root, d), ignore=ign)
    shutil.copytree(os.path.join(TVA, "kernels", VEC), os.path.join(root, "kernels", VEC), ignore=ign)
    return root


def part_r():
    print("# Part R (L52): a reviewer's lines removed from a kernel's ledger, the folder left in place\n")
    root = scratch_root("r")
    use_root(root)
    # a passing gate made by hand (vec-add does not build against this library), as the round-13 evidence's U3
    items = {"model": "a" * 64, "spec": "b" * 64}
    gd = tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items())))
    tvr.dump_json(tvr.kpath(VEC, "gate.json"), {"protocol": "tvr-2", "kernel": VEC, "time": "t", "ok": True, "digest": gd,
                                                "checks": [], "items": items, "kinds": {}, "files": tvr.kernel_files(VEC),
                                                "tool": tvr.tool_digest(), "build": tvr.build_inputs(VEC)})
    tvr.dump_json(tvr.kpath(VEC, "review", "snapshots", gd[:16] + ".json"),
                  {"items": {i: {"digest": d, "cone": {"packages": tvr.packages_digest()}} for i, d in items.items()}})
    report = ("# Review of {k}, round 2, by {w}\n\n## Verdict\n\n{v}\n{why}\n\n## Findings\n\n{f}\n\n## Accepted\n\n{a}\n\n"
              "## Checked\n\n- the probe\n\n## Not checked\n\nNothing.\n")
    # a second reviewer, r-two, files a blocking finding in round 2 and accepts nothing
    show("tvr assign r-two", quiet(tvr.cmd_assign, VEC, "r-two", "all")[0] or "ok")
    d2 = tvr.kpath(VEC, "review", "round-2", "r-two")
    tvr.dump_json(os.path.join(d2, "filing.json"), {
        "reviewer": "r-two", "kernel": VEC, "round": 2, "accept": [], "release": [], "reraise": [],
        "findings": [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "the tolerance says nothing",
                      "detail": "a blocking finding"}]})
    open(os.path.join(d2, "report.md"), "w").write(report.format(
        k=VEC, w="r-two", v="reject", why="A blocking finding.", f="- **SPEC_TOO_WEAK** `item:spec` — the tolerance says nothing",
        a="None."))
    show("tvr file r-two (a blocking finding)", quiet(tvr.cmd_file, VEC, "r-two")[1].splitlines()[0])
    # r-vecadd files again in round 2, accepting every item at the gate
    d1 = tvr.kpath(VEC, "review", "round-2", "r-vecadd")
    tvr.dump_json(os.path.join(d1, "filing-2.json"), {
        "reviewer": "r-vecadd", "kernel": VEC, "round": 2, "accept": "all", "release": [], "reraise": [], "findings": []})
    open(os.path.join(d1, "report-2.md"), "w").write(report.format(
        k=VEC, w="r-vecadd", v="accept", why="Every item.", f="None.", a="All items."))
    show("tvr file r-vecadd (filing-2, accept all)", quiet(tvr.cmd_file, VEC, "r-vecadd")[1].splitlines()[0])
    S = tvr.state(VEC)
    show("certificate with r-two's lines", tvr.cert_state(VEC, S, tvr.load_json(tvr.kpath(VEC, "gate.json")))[0])
    # remove r-two's assign, file and raise events, renumber, and keep r-two's folder
    led = tvr.kpath(VEC, "review", "ledger.jsonl")
    evs = [json.loads(l) for l in open(led) if l.strip()]
    kept = [e for e in evs if e.get("actor") != "r-two"]
    for i, e in enumerate(kept, 1):
        e["seq"] = i
    open(led, "w").write("".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in kept))
    show("events removed", len(evs) - len(kept))
    show("r-two's folder still holds", sorted(os.listdir(d2)))
    S = tvr.state(VEC)
    show("after: kernel_record_problems (the certificate's ledger condition)", tvr.kernel_record_problems(VEC) or "none")
    show("after: certificate", tvr.cert_state(VEC, S, tvr.load_json(tvr.kpath(VEC, "gate.json")))[0])
    show("after: the records step, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("after: what the release check lints there", [c for c in tvr.release_records(root) if c[1] == VEC])
    code, out = quiet(tvr.cmd_lint, ["tvr", "lint", VEC, "reviewer", "r-two"])
    show("after: tvr lint vec_add_f32 reviewer r-two", (code, out.splitlines()[0] if out else ""))
    use_root(None)
    print()


# ---------------------------------------------------------------- Part C

def part_c():
    print("# Part C (L70): citations of files that do not exist, read as placeholders\n")
    root = os.path.join(SCRATCH, "c")
    os.makedirs(os.path.join(root, "docs"))
    os.makedirs(os.path.join(root, "evidence", "num-div-f32"))
    use_root(root)
    page = os.path.join(root, "docs", "page.md")
    cases = ["`docs/DESIGN`", "`docs/DESIGNS.md`", "`evidence/NUM-DIV-F64/README.md`", "`evidence/num-div-F64/README.md`",
             "`evidence/num-div-f64/README.md`", "[the record](../evidence/num-div-F64/README.md)"]
    for c in cases:
        open(page, "w").write(f"See {c}.\n")
        show(f"a page citing {c}", tvr.citation_problems([page]) or "passes")
    use_root(None)
    show("the tree's evidence READMEs (skipped by the audit), cited files that do not exist",
         tvr.citation_problems(sorted(os.path.join(TVA, "evidence", d, "README.md") for d in os.listdir(os.path.join(TVA, "evidence"))
                                      if os.path.isfile(os.path.join(TVA, "evidence", d, "README.md")))) or "none")
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"Python {sys.version.split()[0]}; 2026-10-28\n")
    parts = sys.argv[2:] or ["f", "r", "c"]
    if "f" in parts:
        part_f()
    if "r" in parts:
        part_r()
    if "c" in parts:
        part_c()
