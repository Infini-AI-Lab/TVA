"""The tool after round 16 (library issues L7, L62, L79, L81, L82): git reads no repository's configuration but
the one it is given (L7); a `[rule]` field cites `Compiler.Faithful` in its one form, and other facts only as
docs/rules/num.md lists them (L62); every code line outside a function is classed (L79); every role folder of a
kernel's review/ follows from its ledger (L81); a component in capitals is a cited name, not a placeholder, and
the records step re-derives old events with today's defaults (L82).

Part F and Part L compile small kernels with `lean` (not `lake build`) into the scratch folder and run the tree's
claim reader on them (F1 with its `consistent` example, as the gate runs it; the others without). Part L also
compiles a scratch library module `TVA.Rules.Num.Extra` (helpers a library author might add) and calls the tool's
functions on scratch copies of docs/rules/num.md. Parts M, R and C call the tool's functions on small inputs and
on scratch copies of the parts of the tree they read (tools/, docs/, lean/ without its build,
kernels/vec_add_f32), with the module's root pointed at the copy; Part R pins the copy's vec-add ledger at its
events 1-9, the records round 16 tested (`pin_ledger`, library issue L97). Part G reads Mathlib's upstream refs (it
needs the network). The tree is never written, except for out.txt. No GPU and no Triton run.

Usage: python3 -I evidence/tool-round16/run.py SCRATCH > evidence/tool-round16/out.txt
  SCRATCH  an empty folder outside TVA; its path is never printed.
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
TVA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
REAL = {k: getattr(tvr, k) for k in ("ROOT", "KROOT", "LREVIEW", "LIBRARY_LEDGER")}
VEC = "vec_add_f32"
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
# Lean itself, as elan picks it for lean/lean-toolchain (run in lean/), with the LEAN_PATH `lean_path` gives
# (`lake env lean` would put the build's own paths first)
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    return (str(text).replace(tvr._git_home(), "<git home>").replace(SCRATCH, "<scratch>").replace(TVA, "<tva>")
            .replace(HOME, "~"))


def show(label, value):
    print(f"- {label}: {mask(value)}")


def use_root(root):
    """Point the tool's module at a scratch root (or back at the tree, with None)."""
    if root is None:
        for k, v in REAL.items():
            setattr(tvr, k, v)
        return
    tvr.ROOT, tvr.KROOT = root, os.path.join(root, "kernels")
    tvr.LREVIEW = os.path.join(root, "docs", "library-review")
    tvr.LIBRARY_LEDGER = os.path.join(tvr.LREVIEW, "ledger.jsonl")


def scratch_root(name):
    """The parts of the tree the tool reads, copied to a fresh folder (no Lean build, no git metadata)."""
    root = os.path.join(SCRATCH, name)
    ign = shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv")
    for d in ("tools", "docs", "lean"):
        shutil.copytree(os.path.join(TVA, d), os.path.join(root, d), ignore=ign)
    shutil.copytree(os.path.join(TVA, "kernels", VEC), os.path.join(root, "kernels", VEC), ignore=ign)
    return root


def quiet(fn, *args):
    """Run a tool function that may print or exit; return (exit code or None, what it printed)."""
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, (out.getvalue() + err.getvalue()).strip()


# ---------------------------------------------------------------- Lean kernels (Parts F and L)

def lean_path():
    """The build's LEAN_PATH with the scratch folder's oleans first. The scratch folder mirrors the library's
    built modules (`TVA.olean` and `TVA/`, file by file, as symbolic links), so that a scratch module under
    `TVA` (Part L's `TVA.Rules.Num.Extra`) is found beside them: Lean looks for a module's file in the first
    entry that holds its root."""
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True, text=True).stdout.strip()
    build = os.path.join(LEAN_DIR, ".lake", "build", "lib", "lean")
    if not os.path.exists(os.path.join(OLEAN, "TVA.olean")):
        os.makedirs(OLEAN, exist_ok=True)
        for dirpath, _, names in os.walk(os.path.join(build, "TVA")):
            os.makedirs(os.path.join(OLEAN, os.path.relpath(dirpath, build)), exist_ok=True)
            for n in names:
                os.symlink(os.path.join(dirpath, n), os.path.join(OLEAN, os.path.relpath(dirpath, build), n))
        for n in os.listdir(build):
            if n.startswith("TVA.") and not os.path.lexists(os.path.join(OLEAN, n)):
                os.symlink(os.path.join(build, n), os.path.join(OLEAN, n))
    return OLEAN + os.pathsep + pre


def compile_module(name, text):
    """Compile `text` as module `name` (dotted) into the scratch folder; True when it compiles."""
    rel = name.replace(".", os.sep)
    f = os.path.join(LDIR, rel + ".lean")
    o = os.path.join(OLEAN, rel + ".olean")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    open(f, "w").write(text)
    r = subprocess.run([LEAN, f"--root={LDIR}", "-o", o, f], cwd=LEAN_DIR, capture_output=True,
                       text=True, env=dict(os.environ, LEAN_PATH=lean_path()))
    if r.returncode:
        print(f"- {name}: does not compile: {mask(r.stdout + r.stderr)[:1500]}")
        return False
    return True


def compile_and_read(name, text, with_consistent=False):
    """Compile `text` as module `name` and run the tree's claim reader on it (with its `consistent` example only
    if `with_consistent`). Returns the reader's JSON, or None."""
    if not compile_module(name, text):
        return None
    reader = os.path.join(LDIR, f"reader{name}.lean")
    text = tvr.claim_reader_text(name, "Kernels.Demo", set())
    if not with_consistent:
        text = "\n".join(l for l in text.split("\n") if not l.startswith("example : TVA.Claim.Consistent"))
    open(reader, "w").write(text)
    r = subprocess.run([LEAN, reader], cwd=LEAN_DIR, capture_output=True, text=True,
                       env=dict(os.environ, LEAN_PATH=lean_path()))
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


def field_verdicts(surf, fields):
    consts = tvr.const_table(surf)
    bad = tvr.kinds_problems(surf, consts, "Kernels.Demo")
    for name, _, ty in fields:
        mine = [x[len(name) + 2:] for x in bad if x.startswith(name + ": ")]
        u = surf["field_uses"][name]
        print(f"- `{name} : {ty}`")
        print(f"  arguments as the reader gives them: " + json.dumps(
            [{k: v for k, v in a.items() if v != "" and v != -1 and v is not False} for a in u["args"]], ensure_ascii=False))
        print(f"  kinds: {'; '.join(mine) if mine else 'PASS'}")
    other = [x for x in bad if not any(x.startswith(n + ": ") for n, _, _ in fields)]
    if other:
        show("kinds problems not about a field", other)


# lr-13a's F1 (round-13/lr-13a/work/probes.py): a whole claim whose `faithful` field relates every instance's
# behaviour to the model of one fixed instance; the model of `true`, the instance the host launches, never ends.
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

HEAD = """import TVA
{imports}
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
FAITHFUL = ("faithful", "[rule] `Compiler.Faithful`: the standard form.",
            "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)")


def kernel_text(fields, imports=""):
    body = "".join(f"  /-- {doc} -/\n  {name} : {ty}\n" for name, doc, ty in fields)
    return (HEAD.replace("{imports}", imports) + "structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where\n"
            + body + MAIN)


F2_FIELDS = [
    FAITHFUL,
    ("otherModel", "[rule] lr-13a F2: another instance's model.",
     "∀ d e : Inst, d.ok → Rules.Compiler.Faithful P (model e) (launch d) (K d)"),
    ("otherLaunch", "[rule] lr-13a F2: another instance's launch.",
     "∀ d e : Inst, d.ok → Rules.Compiler.Faithful P (model d) (launch e) (K d)"),
    ("otherK", "[rule] another instance's behaviour.",
     "∀ d e : Inst, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K e)"),
    ("chosen", "[rule] lr-13a F2: the model of one chosen covered instance.",
     "∀ (h : ∃ d : Inst, d.ok) (d : Inst), d.ok → Rules.Compiler.Faithful P (model (Classical.choose h)) (launch d) (K d)"),
    ("fixed", "[rule] the model of the instance 0 (lr-13a F1's pattern on this Inst).",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model ⟨0⟩) (launch d) (K d)"),
    ("written", "[rule] the launch written out.",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (Launch.grid1 1 [[1]]) (K d)"),
    ("eta", "[rule] `K d` written as a lambda (fails closed).",
     "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (fun init o => K d init o)"),
    ("noOk", "[rule] lr-13a F2: at every instance, covered or not (the library's rule holds of every launch).",
     "∀ d, Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
    ("renamed", "[rule] the standard form on another bound variable, with an extra hypothesis.",
     "∀ (d e : Inst), d.ok → d = e → Rules.Compiler.Faithful P (model e) (launch e) (K e)"),
]


def part_f():
    print("# Part F (L62): the arguments of a `Compiler.Faithful` field\n")
    surf = compile_and_read("WrongPair", F1, with_consistent=True)
    if surf:
        show("F1 (lr-13a): a whole claim, `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model false) (launch d) (K d)`",
             "the claim reader elaborates main, consistent, satisfiable and Spec (shape PASS)")
        for k, v in gate_lean_checks(surf).items():
            show(f"  {k}", "PASS" if v is True else f"FAIL {v}")
    surf = compile_and_read("FaithfulArgs", kernel_text(F2_FIELDS))
    if surf:
        field_verdicts(surf, F2_FIELDS)
    print()


# ---------------------------------------------------------------- Part L: the list of facts

# Helpers a library author might add under TVA.Rules.Num (lr-13a's new gap): of type `Platform → Prop`, or named
# with the suffix `Total`, but no fact docs/rules/num.md lists. `DivRnF32` is false on the H200 (evidence/num-div-f32).
EXTRA = """import TVA

namespace TVA.Rules.Num
/-- A helper: `/` on float32 is IEEE (false on NVIDIA). -/
def DivRnF32 (P : Platform) : Prop := DivRn P .binary32
/-- A helper named like a totality rule: an accuracy shape. -/
def DivAccTotal (P : Platform) (e : ℚ) : Prop := DivAcc P .binary32 (2 ^ 126) e (2 ^ (-148 : Int))
end TVA.Rules.Num
"""
L_FIELDS = [
    FAITHFUL,
    ("div", "[rule] `Num.DivFull`.", "Rules.Num.DivFull P"),
    ("exp2", "[rule] `Num.Exp2F32`, under binders.", "∀ d : Inst, d.ok → Rules.Num.Exp2F32 P"),
    ("dot", "[rule] `Num.DotF16`.", "Rules.Num.DotF16 P"),
    ("tot", "[rule] `Num.MathTotal` (listed alone: any arguments).", "Rules.Num.MathTotal P .exp .binary32"),
    ("d64", "[rule] `Num.DivRn` on float64 (listed as `DivRn P .binary64`).", "Rules.Num.DivRn P .binary64"),
    ("d64full", "[rule] the same with the format's full name.", "Rules.Num.DivRn P TVA.Math.Float.Format.binary64"),
    ("d32", "[rule] `Num.DivRn` on float32 (false on NVIDIA).", "Rules.Num.DivRn P .binary32"),
    ("exp9a", "[rule] lr-9a's field: `MathAcc` for `tl.exp` with one rounding's bound.",
     "Rules.Num.MathAcc P .exp .binary32 Set.univ (2 ^ (-24 : Int)) (2 ^ (-150 : Int))"),
    ("fam", "[rule] the family `DotF8` at operands of the field's choice.", "Rules.Num.DotF8 P .f8e4nv .f8e5"),
    ("helper", "[rule] lr-13a: the helper `DivRnF32` (type `Platform → Prop`, not listed).", "Rules.Num.DivRnF32 P"),
    ("helperTot", "[rule] lr-13a: the helper `DivAccTotal` (named `…Total`, not listed).",
     "Rules.Num.DivAccTotal P (2 ^ (-40 : Int))"),
]


def part_l():
    print("# Part L (L62): a [rule] field cites a fact docs/rules/num.md lists\n")
    facts = tvr.stated_facts()
    show(f"the facts the tree's docs/rules/num.md lists ({len(facts)})",
         ", ".join(n[len("TVA.Rules.Num."):] + ("" if p == [None] else " as " + " / ".join(" ".join(x) for x in p))
                   for n, p in facts.items()))
    if compile_module("TVA.Rules.Num.Extra", EXTRA):
        surf = compile_and_read("ListDemo", kernel_text(L_FIELDS, "import TVA.Rules.Num.Extra\n"))
        if surf:
            field_verdicts(surf, L_FIELDS)
    # the list's page in scratch copies: renamed parenthetical, a full name, the section or its table missing
    root = os.path.join(SCRATCH, "l-root")
    os.makedirs(os.path.join(root, "docs", "rules"))
    page = os.path.join(root, "docs", "rules", "num.md")
    text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
    heading = next(l for l in text.split("\n") if l.startswith("## Facts the library states"))
    u_exp = {"head": "TVA.Rules.Num.ExpF32", "platform_fact": True, "args": [{"platform": True}]}
    variants = [
        ("the heading's parenthetical changed", text.replace(heading, "## Facts the library states (sm_90 and sm_100)")),
        ("`ExpF32` written as `TVA.Rules.Num.ExpF32`", text.replace("| `ExpF32` |", "| `TVA.Rules.Num.ExpF32` |")),
        ("the section renamed", text.replace(heading, "## Facts")),
        ("the table removed", "\n".join(l for l in text.split("\n") if not (l.startswith("|") and "`" in l.split("|")[1]))),
        ("the page missing", None),
    ]
    use_root(root)
    for label, t in variants:
        if t is None:
            os.remove(page)
        else:
            open(page, "w", encoding="utf-8").write(t)
        f = tvr.stated_facts()
        show(f"num.md with {label}: facts listed", len(f) if f else f"none ({f})")
        show("  `ExpF32 P`", tvr.rule_fact_problem(u_exp, "Kernels.Demo", f) or "PASS")
        show("  the audit's fact-page check about it", [x for x in tvr.fact_page_problems() if "num.md" in x and "section" in x]
             or "nothing")
    use_root(None)
    print()


# ---------------------------------------------------------------- Part M: module-level lines (L79)

GLOBALS = '''import torch
import triton
import triton.language as tl

SCALE = tl.constexpr(3)
OUT_T = tl.float16


@triton.jit
def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, (x * SCALE).to(OUT_T))


def run():
    x = torch.arange(16, dtype=torch.float32, device="cuda") + 0.25
    o = torch.empty(16, dtype=torch.float32, device="cuda")
    k[(1,)](x, o)
    return x, o


class Wrapper:
    """A docstring, not code."""
    scale = 2

    def __call__(self, x):
        return run()


if __name__ == "__main__":
    run()
'''


def part_m():
    print("# Part M (L79): every code line outside a function is classed\n")
    show("lr-13a's source (work/globals.py, extended with a class and a main block): code lines outside every function",
         tvr.module_lines(GLOBALS))
    code = tvr.code_lines(GLOBALS)[0]

    def case(label, ranges, host):
        mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if any(a <= i <= b for a, b in ranges)]}
        show(label, tvr.map_problems({"source": {"ranges": ranges, "host": host, "unused": []}}, mm, GLOBALS, "x")[0]
             or "passes")
    case("lr-13a's manifest: kernel lines [[9, 13]], host lines [[16, 20], [23, 28]]", [[9, 13]], [[16, 20], [23, 28]])
    case("the module's lines in the kernel lines [[1, 13]], the rest host", [[1, 13]], [[16, 32]])
    case("the module's lines in the host lines (the gate passes; the reviewer judges SCALE and OUT_T)", [[9, 13]], [[1, 6], [16, 32]])
    case("all but the main block classed", [[1, 13]], [[16, 28]])
    man = tvr.manifest(VEC)
    src = open(tvr.kpath(VEC, man["source"]["file"]), encoding="utf-8", newline="").read()
    mm = tvr.load_json(tvr.kpath(VEC, "model-map.json"))
    lean = " ".join(open(p, encoding="utf-8").read() for p in tvr.kernel_lean_files(VEC))
    show("vec-add's code lines outside every function", tvr.module_lines(src))
    show(f"vec-add's manifest as shipped (ranges {man['source']['ranges']}, host {man['source'].get('host', [])})",
         tvr.map_problems(man, mm, src, lean)[0] or "passes")
    # claim.md §2, as `tvr check` renders it (a scratch copy; the gate's other parts are stand-ins)
    root = scratch_root("m")
    use_root(root)
    m = json.loads(json.dumps(man))
    m["source"]["ranges"] = [[21, 26], [29, 54]]
    m["source"]["host"] = [[62, 135]]
    gate = {"time": "t", "ok": False, "digest": "0" * 64, "checks": []}
    items = {i: {"digest": "0" * 64, "cone": [], "files": {}} for i in ("conditions", "spec", "model", "map")}
    surf = {"main_type": "…", "fields": [], "field_docs": {}, "field_types": {}, "field_uses": {}, "usage": {}}
    tvr.render_claim(VEC, m, gate, surf, {}, items, {"entries": []}, src.split("\n"), tvr.jit_functions(src),
                     tvr.source_functions(src))
    text = open(tvr.kpath(VEC, "claim.md"), encoding="utf-8").read()
    para = next(p for p in text.split("\n\n") if p.startswith("The code outside every function"))
    show("claim.md §2 with kernel lines [[21, 26], [29, 54]] and host lines [[62, 135]]", para)
    use_root(None)
    print()


# ---------------------------------------------------------------- Part R: role folders and the ledger (L81)

REPORT = ("# Review of {k}, round 2, by {w}\n\n## Verdict\n\n{v}\n{why}\n\n## Findings\n\n{f}\n\n## Accepted\n\n{a}\n\n"
          "## Checked\n\n- the probe\n\n## Not checked\n\nNothing.\n")


def hand_gate():
    """A passing gate made by hand (vec-add does not build against this library), as the round-13 evidence's U3."""
    items = {"model": "a" * 64, "spec": "b" * 64}
    gd = tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items())))
    tvr.dump_json(tvr.kpath(VEC, "gate.json"), {"protocol": "tvr-2", "kernel": VEC, "time": "t", "ok": True, "digest": gd,
                                                "checks": [], "items": items, "kinds": {}, "files": tvr.kernel_files(VEC),
                                                "tool": tvr.tool_digest(), "build": tvr.build_inputs(VEC)})
    tvr.dump_json(tvr.kpath(VEC, "review", "snapshots", gd[:16] + ".json"),
                  {"items": {i: {"digest": d, "cone": {"packages": tvr.packages_digest()}} for i, d in items.items()}})


def cert():
    return tvr.cert_state(VEC, tvr.state(VEC), tvr.load_json(tvr.kpath(VEC, "gate.json")))[0]


def write_ledger(evs):
    for i, e in enumerate(evs, 1):
        e["seq"] = i
    open(tvr.kpath(VEC, "review", "ledger.jsonl"), "w").write(
        "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in evs))


def pin_ledger(n):
    """Keep the first `n` events of vec-add's ledger in the scratch copy, and remove the files only later events
    record (and the folders left empty): its records as round 16 tested them (library issue L97: vec-add's review
    went on after round 16, so Part R pins the ledger at the events it tested, 1-9: round 2 open, r-vecadd's first
    filing of it recorded)."""
    led = tvr.kpath(VEC, "review", "ledger.jsonl")
    evs = [json.loads(l) for l in open(led) if l.strip()]
    kept = {p for e in evs[:n] for p in e.get("files", {})}
    for p in sorted({p for e in evs[n:] for p in e.get("files", {})} - kept):
        if os.path.isfile(os.path.join(tvr.ROOT, p)):
            os.remove(os.path.join(tvr.ROOT, p))
    for dirpath, _, _ in sorted(os.walk(tvr.kpath(VEC, "review")), reverse=True):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    write_ledger(evs[:n])


def part_r():
    print("# Part R (L81): every role folder of a kernel's review/ follows from its ledger\n")
    show("vec-add as shipped: role folders the events do not account for", tvr.kernel_folder_problems(TVA, VEC) or "none")
    root = scratch_root("r")
    use_root(root)
    pin_ledger(9)
    hand_gate()
    show("tvr start vec_add_f32 reviewer r-two before r-two is assigned",
         quiet(tvr.cmd_start, ["tvr", "start", VEC, "reviewer", "r-two"]))
    show("tvr lint vec_add_f32 reviewer r-two before r-two is assigned",
         quiet(tvr.cmd_lint, ["tvr", "lint", VEC, "reviewer", "r-two"]))
    # lr-13a's Part R: r-two files a blocking finding in round 2; r-vecadd files again, accepting every item
    show("tvr assign r-two", quiet(tvr.cmd_assign, VEC, "r-two", "all")[0] or "ok")
    d2 = tvr.kpath(VEC, "review", "round-2", "r-two")
    tvr.dump_json(os.path.join(d2, "filing.json"), {
        "reviewer": "r-two", "kernel": VEC, "round": 2, "accept": [], "release": [], "reraise": [],
        "findings": [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "the tolerance says nothing",
                      "detail": "a blocking finding"}]})
    open(os.path.join(d2, "report.md"), "w").write(REPORT.format(
        k=VEC, w="r-two", v="reject", why="A blocking finding.", f="- **SPEC_TOO_WEAK** `item:spec` — the tolerance says nothing",
        a="None."))
    show("tvr file r-two (a blocking finding)", quiet(tvr.cmd_file, VEC, "r-two")[1].splitlines()[0])
    d1 = tvr.kpath(VEC, "review", "round-2", "r-vecadd")
    tvr.dump_json(os.path.join(d1, "filing-2.json"), {
        "reviewer": "r-vecadd", "kernel": VEC, "round": 2, "accept": "all", "release": [], "reraise": [], "findings": []})
    open(os.path.join(d1, "report-2.md"), "w").write(REPORT.format(
        k=VEC, w="r-vecadd", v="accept", why="Every item.", f="None.", a="All items."))
    show("tvr file r-vecadd (filing-2, accept all)", quiet(tvr.cmd_file, VEC, "r-vecadd")[1].splitlines()[0])
    show("certificate with r-two's lines", cert())
    led = tvr.kpath(VEC, "review", "ledger.jsonl")
    evs = [json.loads(l) for l in open(led) if l.strip()]
    # R1 (lr-13a): r-two's assign, file and raise events removed, the folder kept
    write_ledger([dict(e) for e in evs if e.get("actor") != "r-two"])
    show("R1 (lr-13a): r-two's 3 events removed; kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("R1: certificate", cert())
    show("R1: the records step, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("R1: tvr lint vec_add_f32 reviewer r-two", quiet(tvr.cmd_lint, ["tvr", "lint", VEC, "reviewer", "r-two"]))
    # R2: r-two's file and raise events removed, its assignment kept (the current round's folder is in progress)
    write_ledger([dict(e) for e in evs if not (e.get("actor") == "r-two" and e["type"] != "assign")])
    show("R2: r-two's file and raise removed, its assign kept; kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("R2: the current round's folders not yet recorded (unrecorded_current)", tvr.unrecorded_current(VEC) or "none")
    show("R2: certificate", cert())
    show("R2: the records step, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("R2: what the release check lints there", [c for c in tvr.release_records(root) if c[1] == VEC])
    write_ledger([dict(e) for e in evs])
    show("the ledger restored: kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    # R3: a folder of an earlier round that no event records, of an assigned reviewer
    os.makedirs(tvr.kpath(VEC, "review", "round-1", "r-two"))
    open(tvr.kpath(VEC, "review", "round-1", "r-two", "filing.json"), "w").write("{}\n")
    show("R3: review/round-1/r-two/ (r-two assigned in round 2, nothing recorded there)", tvr.kernel_record_problems(VEC))
    shutil.rmtree(tvr.kpath(VEC, "review", "round-1", "r-two"))
    # R4: entries of review/ that are no role folder
    os.makedirs(tvr.kpath(VEC, "review", "round-02", "r-two"))
    open(tvr.kpath(VEC, "review", "notes.md"), "w").write("x\n")
    open(tvr.kpath(VEC, "review", "round-2", "stray.json"), "w").write("{}\n")
    show("R4: review/round-02/, review/notes.md and review/round-2/stray.json", tvr.kernel_record_problems(VEC))
    use_root(None)
    print()


# ---------------------------------------------------------------- Part C: citations and old events (L82)

def part_c():
    print("# Part C (L82): cited names in capitals, and old events re-derived with today's defaults\n")
    root = os.path.join(SCRATCH, "c")
    os.makedirs(os.path.join(root, "docs"))
    os.makedirs(os.path.join(root, "evidence", "num-div-f32"))
    os.makedirs(os.path.join(root, "docs", "library-review"))
    os.makedirs(os.path.join(root, "docs", "rules"))
    open(os.path.join(root, "docs", "DESIGN.md"), "w").write("x\n")
    use_root(root)
    page = os.path.join(root, "docs", "page.md")
    cases = ["`docs/DESIGN`", "`docs/DESIGN.md`", "`docs/DESIGNS.md`", "`evidence/NUM-DIV-F64/README.md`",
             "`docs/library-review/round-N/author/notes.md`", "`evidence/<topic>/README.md`", "`docs/rules/*.md`",
             "`evidence/num-div-f32/...`"]
    for c in cases:
        open(page, "w").write(f"See {c}.\n")
        show(f"a page citing {c}", tvr.citation_problems([page]) or "passes")
    use_root(None)
    show("the tree's citations", tvr.citation_problems() or "every cited file exists")
    show("vec-add's ledger as shipped: events no recorded file gives", tvr.kernel_event_problems(TVA, VEC) or "none")
    old = tvr.KINDS["DOC"]
    tvr.KINDS["DOC"] = (True,) + old[1:]
    show("the same with DOC's default made blocking (its round-1 DOC finding gave no `blocking`)",
         tvr.kernel_event_problems(TVA, VEC) or "none")
    tvr.KINDS["DOC"] = old
    print()


# ---------------------------------------------------------------- Part G: git and the repository it runs in (L7)

def part_g():
    print("# Part G (L7): git reads no repository's configuration but the one it is given\n")
    man = json.load(open(os.path.join(TVA, "lean", "lake-manifest.json")))
    url, rev = man["packages"][0]["url"], man["packages"][0]["rev"]
    plain = os.path.join(SCRATCH, "g-plain")
    repo = os.path.join(SCRATCH, "g-repo")
    os.makedirs(plain)
    env = {"PATH": "/usr/bin:/bin", "HOME": plain, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["/usr/bin/git", "init", "-q", repo], check=True, env=env)
    # lr-13a's work/gitcfg.py: the repository's own configuration reads github.com from another host
    subprocess.run(["/usr/bin/git", "-C", repo, "config", "url.https://example.invalid/.insteadOf", "https://github.com/"],
                   check=True, env=env)
    os.makedirs(os.path.join(repo, "sub", "deeper"))
    show("the repository's .git/config", open(os.path.join(repo, ".git", "config")).read().strip().replace("\n", " | "))
    r = subprocess.run(["/usr/bin/git", "ls-remote", url, "refs/tags/v4.35.0"], cwd=repo, capture_output=True, text=True, env=env)
    show("control: a plain `git ls-remote` run inside it", (r.returncode, (r.stdout or r.stderr).strip()[:150]))
    here = os.getcwd()
    for where, cwd in (("outside any repository", plain), ("inside the repository", repo),
                       ("in a subfolder of the repository", os.path.join(repo, "sub", "deeper"))):
        os.chdir(cwd)
        r = tvr._git("ls-remote", url, "refs/tags/v4.35.0")
        show(f"{where}: the tool's _git ls-remote {url} refs/tags/v4.35.0",
             (r.returncode, (r.stdout.decode() or r.stderr.decode()).strip()[:150]))
        os.makedirs(os.path.join(SCRATCH, "g-up-" + where[:6]))
        show(f"{where}: upstream_problem(Mathlib's URL, v4.35.0, its pin, tag)",
             tvr.upstream_problem(url, "v4.35.0", rev, os.path.join(SCRATCH, "g-up-" + where[:6]), "tag") or "accepted")
    os.chdir(here)
    r = tvr._git("-C", os.path.join(repo, "sub"), "rev-parse", "--git-dir")
    show("_git -C <a folder inside the repository, not one itself> rev-parse --git-dir",
         (r.returncode, (r.stdout.decode() or r.stderr.decode()).strip()[:150]))
    r = tvr._git("-C", repo, "config", "--get", "url.https://example.invalid/.insteadof")
    show("_git -C <the repository> config --get its insteadOf (a git given a repository reads it)",
         (r.returncode, r.stdout.decode().strip()))
    r = tvr._git("-C", TVA, "rev-parse", "--show-toplevel")
    show("_git -C <tva> rev-parse --show-toplevel (TVA's own repository, for shipped_files)",
         (r.returncode, r.stdout.decode().strip()))
    show("the folder every git runs in (<git home>/cwd) holds", os.listdir(os.path.join(tvr._git_home(), "cwd")) or "nothing")
    os.chdir(repo)
    os.makedirs(os.path.join(SCRATCH, "g-up-master"))
    show("inside the repository: upstream_problem(Mathlib's URL, master, its pin, branch) (a clone of master's history)",
         tvr.upstream_problem(url, "master", rev, os.path.join(SCRATCH, "g-up-master"), "branch") or "accepted")
    os.chdir(here)
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"{subprocess.run(['/usr/bin/git', '--version'], capture_output=True, text=True).stdout.strip()}; "
          f"Python {sys.version.split()[0]}\n")
    parts = sys.argv[2:] or ["f", "l", "m", "r", "c", "g"]
    for p, fn in (("f", part_f), ("l", part_l), ("m", part_m), ("r", part_r), ("c", part_c), ("g", part_g)):
        if p in parts:
            fn()
