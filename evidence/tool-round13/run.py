"""The tool after round 13 (library issues L7, L50, L52, L62, L63, L68, L69, L70): the programs the release
check runs and Mathlib's revision (L7), every function of the source classed (L50), the ledgers held to their
records (L52), `[rule]` fields that state a fact the library states (L62) with no constant of the kernel's own
(L63), assumption statements in their items' digests (L68), the build's inputs in the gate (L69), and the audit
of kernels/ and of cited paths (L70).

Part U imports the tree's own tools/tvr.py and calls its functions on small inputs, and on scratch copies of the
parts of the tree they read (tools/, docs/, lean/ without its build, kernels/vec_add_f32), with the module's
root pointed at the copy. Part K and Part D compile small kernels with `lean` (not `lake build`) into the scratch
folder and run the tree's claim reader on them; the reader's `consistent` example is left out (these kernels
test the kinds, the namespace check and the digests, not consistency). Each kernel covers every instance, proves
`satisfiable` (the reader asks for it since library issue L71) and proves `main` from its `faithful` field
(`meets_independent`, its programs doing nothing). Part N fetches from Mathlib's upstream
repository (it needs the network). The tree is never written, except for out.txt. No GPU and no Triton run.

Usage: python3 -I evidence/tool-round13/run.py SCRATCH > evidence/tool-round13/out.txt
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


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


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
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, (out.getvalue() + err.getvalue()).strip()


# ---------------------------------------------------------------- Lean kernels (Parts K and D)

LEAN_DIR = os.path.join(TVA, "lean")
OLEAN = os.path.join(SCRATCH, "lean", "olean")


def compile_and_read(name, text):
    """Compile `text` as module `name` into the scratch folder and run the tree's claim reader on it
    (without its `consistent` example). Returns the reader's JSON, or None."""
    d = os.path.join(SCRATCH, "lean")
    os.makedirs(OLEAN, exist_ok=True)
    f = os.path.join(d, f"{name}.lean")
    open(f, "w").write(text)
    r = subprocess.run([tvr.LAKE, "env", "lean", f"--root={d}", "-o", os.path.join(OLEAN, f"{name}.olean"), f],
                       cwd=LEAN_DIR, capture_output=True, text=True)
    if r.returncode:
        print(f"- {name}: does not compile: {mask(r.stdout + r.stderr)[:400]}")
        return None
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True, text=True).stdout.strip()
    reader = os.path.join(d, f"reader{name}.lean")
    text = tvr.claim_reader_text(name, "Kernels.Demo", set())
    text = "\n".join(l for l in text.split("\n") if not l.startswith("example : TVA.Claim.Consistent"))
    open(reader, "w").write(text)
    r = subprocess.run([tvr.LAKE, "env", "lean", reader], cwd=LEAN_DIR, capture_output=True, text=True,
                       env=dict(os.environ, LEAN_PATH=pre + os.pathsep + OLEAN))
    js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
    if r.returncode or len(js) != 1:
        print(f"- {name}: the reader fails: {mask(r.stdout + r.stderr)[:400]}")
        return None
    return json.loads(js[0][len("TVR_JSON "):])


HEAD = """import TVA

namespace Kernels.Demo
open TVA

/-- The instances. -/
structure Inst where
  n : Nat
/-- Every instance is covered. -/
def Inst.ok (_ : Inst) : Prop := True
/-- The model (a stand-in for a kernel's). -/
def model (_ : Inst) : Kernel := { body := fun _ => pure () }
/-- The launch (a stand-in). -/
def launch (_ : Inst) : Launch := Launch.grid1 1 [[1]]
def Pre (_ : Inst) (_ : Mem) : Prop := True
def Post (_ : Inst) (_ _ : Mem) : Prop := True
def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K
"""
MAIN = """theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
{proof}
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩
end Kernels.Demo
"""
# `main` from the `faithful` field: one program that does nothing, with empty footprints
PROOF_1 = """  intro K P h
  exact meets_independent h.faithful (fun _ _ => rfl)
    (fun _ _ => Launch.valid_grid1 (by norm_num) (by simp [Shape.TritonOk]; exact ⟨0, rfl⟩))
    (fun _ _ _ => ∅) (fun _ _ _ => ∅) (fun _ _ _ _ => True)
    (fun _ _ _ _ _ => safe_ret trivial) (fun _ _ _ _ _ _ _ => by simp) (fun _ _ _ _ _ => by simp)
    (fun _ _ _ _ _ _ _ => trivial)"""
# the same proof, written otherwise (Part D's proof edit)
PROOF_2 = """  intro K P h
  refine meets_independent h.faithful (fun _ _ => rfl) (fun _ _ => Launch.valid_grid1 (by decide) ?_)
    (fun _ _ _ => ∅) (fun _ _ _ => ∅) (fun _ _ _ _ => True) (fun _ _ _ _ _ => safe_ret trivial)
    (fun _ _ _ _ _ _ _ => disjoint_bot_left) (fun _ _ _ _ _ => disjoint_bot_left)
    (fun _ _ _ _ _ _ _ => trivial)
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  exact ⟨fun k hk => ⟨0, by simpa using hk⟩, by decide⟩"""

K_DECLS = """/-- A bound of the kernel's own, private. -/
private def tol : ℚ := 2 ^ (-40 : Int)
/-- The same bound, declared outside the kernel's namespace under an internal-looking name. -/
def _root_.Kernels._tol : ℚ := 2 ^ (-40 : Int)
/-- The same bound, at the root. -/
def _root_._tol : ℚ := 2 ^ (-40 : Int)
/-- An instance with a name `Name.isInternal` holds of. -/
instance _hdiv : HDiv Nat Nat Nat := ⟨fun a _ => a⟩
"""
K_FIELDS = [
    ("faithful", "[rule] `Compiler.Faithful`.", "∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)"),
    ("div", "[rule] `Num.DivFull`.", "Rules.Num.DivFull P"),
    ("exp2", "[rule] `Num.Exp2F32`, under binders.", "∀ d : Inst, d.ok → Rules.Num.Exp2F32 P"),
    ("tot", "[rule] `Num.MathTotal`.", "Rules.Num.MathTotal P .exp .binary32"),
    ("d64", "[rule] `Num.DivRn` on float64.", "Rules.Num.DivRn P .binary64"),
    ("d32", "[rule] `Num.DivRn` on float32 (false on NVIDIA).", "Rules.Num.DivRn P .binary32"),
    ("accLib", "[rule] `DivAcc` with DivFull's constants written out.",
     "Rules.Num.DivAcc P .binary32 (2 ^ 126) (2 ^ (-22 : Int)) (2 ^ (-148 : Int))"),
    ("exp9a", "[rule] lr-9a's field: `MathAcc` for `tl.exp` with one rounding's bound.",
     "Rules.Num.MathAcc P .exp .binary32 Set.univ (2 ^ (-24 : Int)) (2 ^ (-150 : Int))"),
    ("fam", "[rule] the family `DotF8` at operands of the field's choice.", "Rules.Num.DotF8 P .f8e4nv .f8e5"),
    ("tolPriv", "[rule] a private bound of the kernel's own.", "Rules.Num.DivAcc P .binary32 (2 ^ 126) tol (2 ^ (-148 : Int))"),
    ("tolRoot", "[rule] `Kernels._tol`.", "Rules.Num.DivAcc P .binary32 (2 ^ 126) Kernels._tol (2 ^ (-148 : Int))"),
    ("tolRoot2", "[rule] `_tol` at the root.", "Rules.Num.DivAcc P .binary32 (2 ^ 126) _root_._tol (2 ^ (-148 : Int))"),
    ("kOut", "[rule] `K` outside `Compiler.Faithful`.", "(K = K) → Rules.Num.DivFull P"),
    ("conj", "[rule] two facts in one field.", "Rules.Num.DivFull P ∧ Rules.Num.Exp2F32 P"),
    ("own", "[kernel] a fact about the model.", "model = model"),
]


def kernel_text(fields, decls="", proof=PROOF_1):
    body = "".join(f"  /-- {doc} -/\n  {name} : {ty}\n" for name, doc, ty in fields)
    return (HEAD.replace("open TVA\n", "open TVA\n\n" + decls) +
            "structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where\n" + body +
            MAIN.replace("{proof}", proof))


def part_k():
    print("# Part K (L62, L63): what a [rule] field may state, and whose constants\n")
    surf = compile_and_read("KindsDemo", kernel_text(K_FIELDS, K_DECLS))
    if not surf:
        return
    consts = tvr.const_table(surf)
    bad = tvr.kinds_problems(surf, consts, "Kernels.Demo")
    for name, _, ty in K_FIELDS:
        u = surf["field_uses"][name]
        mine = [x[len(name) + 2:] for x in bad if x.startswith(name + ": ")]
        print(f"- `{name} : {ty}`")
        print(f"  head {u['head'] or '(none)'}; Platform → Prop: {u['platform_fact']}; "
              f"arguments {[('P' if a['platform'] else a['const'] or '…') for a in u['args']]}")
        print(f"  kinds: {'; '.join(mine) if mine else 'PASS'}")
    other = [x for x in bad if not any(x.startswith(n + ": ") for n, _, _ in K_FIELDS)]
    show("kinds problems not about a field", other or "none")
    show("namespace: names declared outside the kernel's namespace", [x for x in surf["shadow"] if "outside" in x] or "none")
    show("namespace: instances (internal = the one Lean makes)", [(i["name"], i["class"], i["internal"]) for i in surf["instances"]])
    # which constants of the library the second form accepts: every TVA.Rules constant of type Platform → Prop
    lister = os.path.join(SCRATCH, "lean", "PlatformFacts.lean")
    open(lister, "w").write(PLATFORM_FACTS)
    r = subprocess.run([tvr.LAKE, "env", "lean", lister], cwd=LEAN_DIR, capture_output=True, text=True)
    facts = sorted(l.strip() for l in r.stdout.splitlines() if l.strip())
    show(f"the TVA.Rules constants of type Platform → Prop ({len(facts)})", facts or mask(r.stdout + r.stderr)[:300])
    # a control: the names Lean makes for an ordinary kernel stay inside its namespace, so exempting only names
    # with a numeric component refuses none of them
    surf = compile_and_read("NamesDemo", kernel_text([K_FIELDS[0], K_FIELDS[1], K_FIELDS[-1]], NAMES_DECLS))
    if surf:
        show("control (structures with defaults, recursion, a private def, simp, omega, decide, norm_num): namespace "
             "problems", surf["shadow"] or "none")
        show("control: instances", [(i["name"], i["class"], i["internal"]) for i in surf["instances"]])
    print()


PLATFORM_FACTS = """import Lean
import TVA
open Lean Meta
#eval show MetaM Unit from do
  for (n, ci) in (← getEnv).constants.map₁.toList do
    if (`TVA.Rules).isPrefixOf n then
      match ci.type with
      | .forallE _ (.const p []) (.sort .zero) _ => if p == ``TVA.Platform then IO.println n
      | _ => pure ()
"""

NAMES_DECLS = """/-- A structure with a default. -/
structure Cfg where
  a : Nat
  b : Nat := 3
/-- Recursion with a match. -/
def f (x : Nat) : Nat := match x with
  | 0 => 1
  | k + 1 => f k + 2
theorem f_pos (x : Nat) : 0 < f x := by
  induction x <;> simp [f]
theorem t1 (a b : Nat) (h : a < b) : a + 1 ≤ b := by omega
theorem t2 : (37 : Nat) * 3 = 111 := by decide
private def secret : Nat := 4
/-- A definition with a `let` and an `if`. -/
def g (x : Nat) : Nat := if x < 3 then x + secret else let y := x * 2; y + 1
theorem g_eq : g 5 = 11 := by unfold g; norm_num
"""


D_BASE = [("div", "[rule] the division shape (a digest probe; `kinds` refuses it).",
           "Rules.Num.DivAcc P .binary32 (2 ^ 126) (2 ^ (-22 : Int)) (2 ^ (-148 : Int))"),
          ("own", "[kernel] a fact about the model.", "model (Inst.mk 0) = model (Inst.mk 0)")]
D_FAITHFUL = K_FIELDS[0]  # every variant has it, unchanged: `main` uses it


def part_d():
    print("# Part D (L68): an assumption item's digest holds its statement\n")
    print("lr-10a's digest probes: R2 tightens `div`'s relative bound to 2^-40, R3 wraps `div` in `∀ _ : Nat`, K2")
    print("changes the [kernel] field `own`; C is R1 with a proof edit.\n")
    variants = {
        "R1": (D_BASE, PROOF_1),
        "R2": ([(D_BASE[0][0], D_BASE[0][1], D_BASE[0][2].replace("(-22 : Int)", "(-40 : Int)")), D_BASE[1]], PROOF_1),
        "R3": ([(D_BASE[0][0], D_BASE[0][1], "∀ _ : Nat, " + D_BASE[0][2]), D_BASE[1]], PROOF_1),
        "K2": ([D_BASE[0], ("own", D_BASE[1][1], "model (Inst.mk 0) = model (Inst.mk 0) ∧ True")], PROOF_1),
        "C": (D_BASE, PROOF_2),
    }
    digests = {}
    for v, (fields, proof) in variants.items():
        surf = compile_and_read(f"Digest{v}", kernel_text([D_FAITHFUL] + fields, "", proof))
        if not surf:
            continue
        consts = tvr.const_table(surf)
        items = tvr.item_digests(surf, consts, "0" * 64, "{}")
        digests[v] = {k: x["digest"][:16] for k, x in items.items()}
        print(f"- {v}: statements {json.dumps(surf['field_types'], ensure_ascii=False)}")
        print(f"  item digests {digests[v]}")
    for x, y in (("R1", "R2"), ("R1", "R3"), ("R1", "K2"), ("R1", "C")):
        if x in digests and y in digests:
            same = sorted(i for i in digests[x] if digests[x][i] == digests[y][i])
            print(f"- {x} against {y}: different for {', '.join(sorted(set(digests[x]) - set(same))) or 'no item'}; "
                  f"the same for {', '.join(same) or 'no item'}")
    print()


# ---------------------------------------------------------------- Part U: the tool's functions

def part_u():
    print("# Part U: the tool's functions\n")

    print("## U1 (L50): every function of the source is classed")
    man = tvr.manifest(VEC)
    src = open(tvr.kpath(VEC, man["source"]["file"]), encoding="utf-8", newline="").read()
    mm = tvr.load_json(tvr.kpath(VEC, "model-map.json"))
    lean = " ".join(open(p, encoding="utf-8").read() for p in tvr.kernel_lean_files(VEC))
    show("vec-add's functions", tvr.source_functions(src))
    show("vec-add's manifest (ranges [[29, 54]], host [[62, 78], [84, 89], [107, 129]])", tvr.map_problems(man, mm, src, lean))
    m = json.loads(json.dumps(man))
    m["source"]["host"] = [[62, 78]]
    show("vec-add with host [[62, 78]] only (benchmark and its lambdas unclassed)", tvr.map_problems(m, mm, src, lean)[0])
    cases = {
        "lr-10a G6: `h = triton.jit(_h)`, ranges hold the kernel only":
            ("import triton\nimport triton.language as tl\n\n\ndef _h(x):\n    return x + 1\n\n\nh = triton.jit(_h)\n\n\n"
             "@triton.jit\ndef k(p):\n    tl.store(p, h(tl.load(p)))\n", [[12, 14]], []),
        "an aliased decorator (`from triton import jit as J`)":
            ("import triton\nfrom triton import jit as J\nimport triton.language as tl\n\n\n@J\ndef h(x):\n    return x + 1\n\n\n"
             "@triton.jit\ndef k(p):\n    tl.store(p, h(tl.load(p)))\n", [[11, 13]], []),
        "a decorator bound to a name (`deco = triton.jit`)":
            ("import triton\nimport triton.language as tl\n\ndeco = triton.jit\n\n\n@deco\ndef h(x):\n    return x + 1\n\n\n"
             "@triton.jit\ndef k(p):\n    tl.store(p, h(tl.load(p)))\n", [[12, 14]], []),
        "lr-10a G5: a `@triton.constexpr_function` helper":
            ("import triton\nimport triton.language as tl\n\n\n@triton.constexpr_function\ndef width(n):\n    return 2 * n\n\n\n"
             "@triton.jit\ndef k(p, N: tl.constexpr):\n    tl.store(p + tl.arange(0, width(N)), 0)\n", [[10, 12]], []),
        "a lambda made a constexpr function":
            ("import triton\nimport triton.language as tl\n\nwidth = triton.constexpr_function(lambda n: 2 * n)\n\n\n"
             "@triton.jit\ndef k(p, N: tl.constexpr):\n    tl.store(p + tl.arange(0, width(N)), 0)\n", [[7, 9]], []),
        "the same lambda in the host lines (control: the reviewer judges the class)":
            ("import triton\nimport triton.language as tl\n\nwidth = triton.constexpr_function(lambda n: 2 * n)\n\n\n"
             "@triton.jit\ndef k(p, N: tl.constexpr):\n    tl.store(p + tl.arange(0, width(N)), 0)\n", [[7, 9]], [[4, 4]]),
        "a `@triton.jit` helper in the host lines":
            ("import triton\nimport triton.language as tl\n\n\n@triton.jit\ndef h(x):\n    return x + 1\n\n\n"
             "@triton.jit\ndef k(p):\n    tl.store(p, h(tl.load(p)))\n", [[10, 12]], [[5, 7]]),
        "a nested helper inside the kernel (control)":
            ("import triton\nimport triton.language as tl\n\n\n@triton.jit\ndef k(p):\n    def inner(x):\n        return x\n"
             "    tl.store(p, tl.load(p))\n", [[5, 9]], []),
    }
    for label, (text, ranges, host) in cases.items():
        hman = {"source": {"ranges": ranges, "host": host, "unused": []}}
        code = tvr.code_lines(text)[0]
        hmm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if any(a <= i <= b for a, b in ranges)]}
        show(label, tvr.map_problems(hman, hmm, text, "x")[0] or "passes")
    text, ranges, _ = cases["lr-10a G6: `h = triton.jit(_h)`, ranges hold the kernel only"]
    code = tvr.code_lines(text)[0]
    hmm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if 12 <= i <= 14]}
    show("lr-10a G6 with source.unused = [_h] (the reviewer judges that `k` never runs it)",
         tvr.map_problems({"source": {"ranges": ranges, "host": [], "unused": ["_h"]}}, hmm, text, "x")[0] or "passes")
    show("source.unused naming no function of the source",
         tvr.map_problems({"source": {"ranges": ranges, "host": [], "unused": ["nope"]}}, hmm, text, "x")[0])
    print()

    print("## U2 (L52): kernel ledgers derived from their recorded files")
    show("vec-add's ledger as shipped: events no recorded file gives", tvr.kernel_event_problems(TVA, VEC) or "none")
    show("vec-add's ledger as shipped: recorded files", tvr.ledger_problems(TVA, f"kernels/{VEC}/review/ledger.jsonl", "the ledger") or "as recorded")
    root = scratch_root("u2")
    use_root(root)
    led = os.path.join(root, "kernels", VEC, "review", "ledger.jsonl")
    events = [json.loads(l) for l in open(led) if l.strip()]
    # lr-10a G1: a hand-written accept at another gate, appended
    with open(led, "a") as f:
        f.write(json.dumps({"seq": len(events) + 1, "time": "t", "type": "accept", "actor": "r-vecadd", "round": 2,
                            "items": {"model": "c" * 64}, "gate": "d" * 64}) + "\n")
    show("lr-10a G1: a hand-written accept appended", tvr.kernel_event_problems(root, VEC))
    show("  the certificate's ledger condition (kernel_record_problems)", len(tvr.kernel_record_problems(VEC)) > 0)
    open(led, "w").write("".join(json.dumps(e) + "\n" for e in events))
    # the accept of round 2 changed in place to other digests
    ev2 = json.loads(json.dumps(events))
    ev2[-1]["items"] = {k: "e" * 64 for k in ev2[-1]["items"]}
    open(led, "w").write("".join(json.dumps(e) + "\n" for e in ev2))
    show("round 2's accept edited to other digests", tvr.kernel_event_problems(root, VEC))
    # a release with no filing
    open(led, "w").write("".join(json.dumps(e) + "\n" for e in events) + json.dumps(
        {"seq": len(events) + 1, "time": "t", "type": "release", "actor": "r-vecadd", "round": 2, "id": "F1", "note": "x"}) + "\n")
    show("a release appended with no filing", tvr.kernel_event_problems(root, VEC))
    # a decide on a finding that is not escalated
    open(led, "w").write("".join(json.dumps(e) + "\n" for e in events) + json.dumps(
        {"seq": len(events) + 1, "time": "t", "type": "decide", "actor": "user", "round": 2, "id": "F1", "decision": "x"}) + "\n")
    show("a decide on a finding that is not escalated", tvr.kernel_event_problems(root, VEC))
    open(led, "w").write("".join(json.dumps(e) + "\n" for e in events))
    # lr-10b P2 for a kernel: an edited recorded report, re-frozen by an appended line with its new sha256
    rep = os.path.join(root, "kernels", VEC, "review", "round-2", "r-vecadd", "report.md")
    open(rep, "a").write("\nedited after recording\n")
    with open(led, "a") as f:
        f.write(json.dumps({"seq": len(events) + 1, "time": "t", "type": "note", "actor": "coordinator", "text": "x",
                            "files": tvr.archive_entry(rep)}) + "\n")
    show("a recorded report edited, and a line appended that records it again",
         tvr.ledger_problems(root, f"kernels/{VEC}/review/ledger.jsonl", "the ledger"))
    use_root(None)
    print()

    print("## U3 (L52): a round trip through `tvr file` and `tvr respond`")
    root = scratch_root("u3")
    use_root(root)
    # a passing gate as `tvr check` would write it now (the kernel does not build against this library, so the
    # gate is made by hand: its files, tool and build digests as they are, one item, and its snapshot)
    items = {"model": "a" * 64, "spec": "b" * 64}
    gd = tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items())))
    tvr.dump_json(tvr.kpath(VEC, "gate.json"), {"protocol": "tvr-2", "kernel": VEC, "time": "t", "ok": True, "digest": gd,
                                                "checks": [], "items": items, "kinds": {}, "files": tvr.kernel_files(VEC),
                                                "tool": tvr.tool_digest(), "build": tvr.build_inputs(VEC)})
    tvr.dump_json(tvr.kpath(VEC, "review", "snapshots", gd[:16] + ".json"),
                  {"items": {i: {"digest": d, "cone": {"packages": tvr.packages_digest()}} for i, d in items.items()}})
    show("the hand-made gate: stale", tvr.gate_staleness(VEC, tvr.load_json(tvr.kpath(VEC, "gate.json"))) or "no")
    d = tvr.kpath(VEC, "review", "round-2", "r-vecadd")
    tvr.dump_json(os.path.join(d, "filing-2.json"), {
        "reviewer": "r-vecadd", "kernel": VEC, "round": 2, "accept": ["item:spec"], "release": [], "reraise": [],
        "findings": [{"kind": "DOC", "target": "item:model", "summary": "a remark"}]})
    open(os.path.join(d, "report-2.md"), "w").write(
        f"# Review of {VEC}, round 2, by r-vecadd\n\n## Verdict\n\naccept-partly\nA remark.\n\n## Findings\n\n"
        "- **DOC** `item:model` — a remark\n\n## Accepted\n\n- `item:spec`\n\n## Checked\n\n- the probe\n\n## Not checked\n\nNothing.\n")
    code, out = quiet(tvr.cmd_file, VEC, "r-vecadd")
    show("tvr file (a second filing in round 2)", (code, out.splitlines()[0] if out else ""))
    show("  events no recorded file gives, after it", tvr.kernel_event_problems(root, VEC) or "none")
    pd = tvr.kpath(VEC, "review", "round-2", "prover")
    tvr.dump_json(os.path.join(pd, "response.json"), {"kernel": VEC, "round": 2, "responses": [
        {"id": "F2", "response": "CLARIFIED", "text": "noted"}]})
    open(os.path.join(pd, "notes.md"), "w").write(
        f"# Prover's notes on {VEC}, round 2\n\n## Changes\n\nNone.\n\n## Responses\n\n- **F2** CLARIFIED — noted\n\n"
        f"## Library requests\n\nNone.\n\n## Gate\n\nPASS, digest `{gd[:16]}`\n")
    code, out = quiet(tvr.cmd_respond, VEC)
    show("tvr respond", (code, out.splitlines()[0] if out else ""))
    show("  events no recorded file gives, after it", tvr.kernel_event_problems(root, VEC) or "none")
    show("  recorded files", tvr.kernel_record_problems(VEC) or "as recorded")
    use_root(None)
    print()

    print("## U4 (L52): the library's ledger")
    show("the tree's library ledger: lines", tvr.library_ledger_problems(TVA) or "every line records one role folder, once")
    root = scratch_root("u4")
    use_root(root)
    lled = os.path.join(root, "docs", "library-review", "ledger.jsonl")
    lines = open(lled).read()
    before = tvr.ledger_digests(root)[tvr.LIBRARY_LEDGER_PATH]
    rep = os.path.join(root, "docs", "library-review", "round-7", "lr-7a", "report.md")
    text = open(rep).read()
    assert "**L7** REOPEN" in text
    open(rep, "w").write(text.replace("**L7** REOPEN", "**L7** ACCEPT"))
    show("lr-10b P2: round 7's lr-7a report changed (REOPEN to ACCEPT)", tvr.records_problems(root))
    evs = [json.loads(l) for l in lines.splitlines() if l.strip()]
    r7 = next(e for e in evs if e["dir"] == "docs/library-review/round-7/lr-7a")
    again = {**r7, "seq": len(evs) + 1, "files": {**r7["files"], "docs/library-review/round-7/lr-7a/report.md": tvr.file_sha(rep)}}
    open(lled, "w").write(lines + json.dumps(again) + "\n")
    show("route 1: one line appended that records the folder again", tvr.records_problems(root))
    r7e = {**r7, "files": {**r7["files"], "docs/library-review/round-7/lr-7a/report.md": tvr.file_sha(rep)}}
    open(lled, "w").write("".join(json.dumps(r7e if e is r7 else e) + "\n" for e in evs))
    show("route 2: the original line's sha256 edited", tvr.records_problems(root) or "passes (a ledger's lines rest on "
         "the git history or an outside digest)")
    after = tvr.ledger_digests(root)[tvr.LIBRARY_LEDGER_PATH]
    first = b"".join(open(lled, "rb").readlines()[:before["lines"]])
    show("  release.json's earlier digest of the ledger against the first lines now",
         "the same" if tvr.sha(first) == before["sha256"] else f"differs ({before['sha256'][:12]} then, {tvr.sha(first)[:12]} now)")
    open(lled, "w").write(lines)
    open(rep, "w").write(open(os.path.join(TVA, "docs", "library-review", "round-7", "lr-7a", "report.md")).read())
    # lr-10a G2: --no-lint for a later round
    os.makedirs(os.path.join(root, "docs", "library-review", "round-10", "lr-x"))
    open(os.path.join(root, "docs", "library-review", "round-10", "lr-x", "report.md"), "w").write("not a report\n")
    code, out = quiet(tvr.cmd_record, ["tvr", "record", "library", "10", "lr-x", "--no-lint"])
    show("lr-10a G2: tvr record library 10 lr-x --no-lint", (code, out))
    hand = {"seq": len(evs) + 1, "time": "t", "type": "record", "actor": "coordinator", "round": 10, "who": "lr-x",
            "dir": "docs/library-review/round-10/lr-x", "linted": False,
            "files": tvr.archive_entry(os.path.join(root, "docs", "library-review", "round-10", "lr-x", "report.md"))}
    open(lled, "w").write(lines + json.dumps(hand) + "\n")
    show("the same line written by hand", tvr.library_ledger_problems(root))
    use_root(None)
    print()

    print("## U5 (L69): the build's inputs make a gate stale")
    root = scratch_root("u5")
    use_root(root)
    items = {"model": "a" * 64}
    gd = tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items())))
    gate = {"ok": True, "digest": gd, "items": items, "files": tvr.kernel_files(VEC), "tool": tvr.tool_digest(),
            "build": tvr.build_inputs(VEC)}
    tvr.dump_json(tvr.kpath(VEC, "review", "snapshots", gd[:16] + ".json"),
                  {"items": {"model": {"digest": items["model"], "cone": {"packages": tvr.packages_digest()}}}})
    show("the library files the kernel imports, transitively (count)", len([p for p in gate["build"] if p.endswith(".lean")]))
    show("a gate recorded now", tvr.gate_staleness(VEC, gate) or "not stale")
    edits = [("lr-10a G4: lean/lean-toolchain to v4.36.0", "lean/lean-toolchain", lambda t: t.replace("v4.35.0", "v4.36.0")),
             ("a leanOptions key in lean/lakefile.toml", "lean/lakefile.toml", lambda t: t.replace("autoImplicit = false", "autoImplicit = false\nrelaxedAutoImplicit = false")),
             ("a comment in lean/TVA.lean (a module that holds no constant)", "lean/TVA.lean", lambda t: t + "\n-- a comment\n"),
             ("a comment in lean/TVA/Lemmas/Sm90.lean", "lean/TVA/Lemmas/Sm90.lean", lambda t: t + "\n-- a comment\n")]
    for label, p, fn in edits:
        f = os.path.join(root, p)
        old = open(f).read()
        open(f, "w").write(fn(old))
        show(label, tvr.gate_staleness(VEC, gate) or "not stale")
        open(f, "w").write(old)
    show("a gate without `build` (round 10's)", tvr.gate_staleness(VEC, {k: v for k, v in gate.items() if k != "build"}))
    show("gate_mismatch: a shipped gate whose build inputs differ",
         tvr.gate_mismatch({**gate, "build": {"lean/lean-toolchain": "1"}}, gate))
    use_root(None)
    print()

    print("## U6 (L70): the audit of kernels/, and cited paths")
    root = scratch_root("u6")
    use_root(root)
    os.makedirs(os.path.join(root, "kernels", "notes"))
    os.makedirs(os.path.join(root, "evidence"))  # the copy's evidence folder (its files are not needed here)
    sib = "~/" + "fpv/" + "sibling/x.md"  # a path into a sibling of TVA, written in parts so this script holds none
    open(os.path.join(root, "kernels", "notes", "readme.md"), "w").write(f"see {sib}\n")
    code, out = quiet(tvr.main, ["tvr", "audit"])
    show("lr-10a G8: kernels/notes/readme.md (no kernel.json) holding a sibling path, named by the audit",
         "kernels/notes/readme.md" in out)
    page = os.path.join(root, "docs", "probe.md")
    open(page, "w").write("Cites `evidence/nope/README.md`, [a link](../evidence/nope-too/README.md), `docs/SEMANTICS.md:12`,\n"
                          "[the protocol](REVIEW_PROTOCOL.md#4-claim-items-and-digests), `docs/library-review/round-N/author/notes.md`,\n"
                          "`evidence/<topic>/README.md`, `docs/rules/*.md`, `compiled[grid](…)` in code.\n")
    show("a page citing two missing files, and six that exist, end in a placeholder or are code", tvr.citation_problems([page]))
    lean = os.path.join(root, "lean", "TVA", "Probe.lean")
    open(lean, "w").write("/-! Evidence: `docs/evidence/conformance/cvt.md`. -/\n")
    show("lr-9a: a Lean docstring citing a missing record", tvr.citation_problems([lean]))
    use_root(None)
    show("the tree's citations", tvr.citation_problems() or "every cited file exists")
    print()


# ---------------------------------------------------------------- Part N: the release check's programs and Mathlib

def part_n():
    print("# Part N (L7): the programs the release check runs, and Mathlib's revision\n")
    fake = os.path.join(SCRATCH, "fakebin")
    os.makedirs(fake)
    marker = os.path.join(SCRATCH, "fake-ran.txt")
    for name in ("git", "bwrap", "find"):
        p = os.path.join(fake, name)
        open(p, "w").write(f"#!/bin/sh\necho {name} >> {marker}\nexit 0\n")
        os.chmod(p, 0o755)
    show("system_tool(git), (bwrap), (find)", [tvr.system_tool(x) for x in ("git", "bwrap", "find")])
    code = ("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, 'tools'); import tvr; "
            "print('git:', tvr._git('--version').stdout.decode().strip()); print('bwrap:', tvr.system_tool('bwrap')); "
            "print('find:', tvr.system_problems(['/usr/share/doc']) or 'nothing user-placed'); tvr.shipped_files()")
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=TVA, capture_output=True, text=True,
                       env=dict(os.environ, PATH=fake + os.pathsep + os.environ.get("PATH", "")))
    show("lr-10b P3: with git, bwrap and find wrappers first on PATH, the tool ran", r.stdout.strip().replace("\n", "; "))
    show("  the wrappers that ran", open(marker).read().split() if os.path.exists(marker) else "none")
    show("  the PATH git gets", tvr._git_env()["PATH"])
    show("Mathlib's allowed inputRevs for the tree's toolchain", tvr.mathlib_ref(open(os.path.join(TVA, "lean", "lean-toolchain")).read()))
    show("the tree's lake configuration", tvr.lake_violations() or "accepted")
    root = scratch_root("n")
    use_root(root)
    lf = os.path.join(root, "lean", "lakefile.toml")
    lm = os.path.join(root, "lean", "lake-manifest.json")
    text = open(lf).read()
    open(lf, "w").write(text.replace('rev = "v4.35.0"', 'rev = "feat/abel-sym-module"'))
    man = json.load(open(lm))
    man["packages"][0]["inputRev"] = "feat/abel-sym-module"
    json.dump(man, open(lm, "w"), indent=2)
    show("lr-10a G7: a feature branch as Mathlib's rev (the audit)", tvr.lake_violations())
    show("lr-10a G7: the same in the release check's packages step (before any fetch)",
         tvr.make_packages(os.path.join(root, "lean"), os.path.join(SCRATCH, "n-scratch")))
    use_root(None)
    url, rev = man["packages"][0]["url"], json.load(open(os.path.join(TVA, "lean", "lake-manifest.json")))["packages"][0]["rev"]
    os.makedirs(os.path.join(SCRATCH, "n-up"))
    show("upstream: Mathlib's pin as the tag v4.35.0", tvr.upstream_problem(url, "v4.35.0", rev, os.path.join(SCRATCH, "n-up", "a"), "tag") or "accepted")
    show("upstream: Mathlib's pin as a commit of master", tvr.upstream_problem(url, "master", rev, os.path.join(SCRATCH, "n-up", "b"), "branch") or "accepted")
    show("upstream: `master` asked for as a tag", tvr.upstream_problem(url, "master", rev, os.path.join(SCRATCH, "n-up", "c"), "tag"))
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"{subprocess.run(['/usr/bin/git', '--version'], capture_output=True, text=True).stdout.strip()}; "
          f"Python {sys.version.split()[0]}\n")
    part_k()
    part_d()
    part_u()
    part_n()
