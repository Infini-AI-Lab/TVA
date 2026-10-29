"""lr-16a's probes (library review round 16, survey area `gate`).

Part K: the claim reader finds `K` and `P` by their place among the binders of an `Assumptions` field's
projection (`xs[0]`, `xs[1]`). Does a structure with two instance-implicit parameters before `K` and `P`
(`structure Assumptions [Inhabited Unit] [Inhabited Bool] (K …) (P …)`) still pass `shape`, and do `[kernel]`
fields that mention `K` (the specification itself) or `P` (a device fact) then pass `kinds`?
Part U: an unused function's `def` statement runs when Python executes the module (its defaults and
decorators are evaluated), so it can bind a module-level name the kernel reads. Does the map check pass a source
whose unused function's default rebinds a name the kernel lines bind: `SCALE = tl.constexpr(2)` to 3, or the
`@triton.jit` helper `double` to the unused `@triton.jit` function `triple`? (Sources: `probes_src.py`; the device
half is `unused_gpu.py`.)
Part R: a kernel's role folders against its ledger (L52, L81), in a scratch copy of the tree: vec-add's ledger has
`file` and `response` events that name no `dir` (written before the tool recorded folders); is a folder they record
frozen? Does lr-13a's route (a reviewer's `file` and `raise` lines removed, the filing kept) grant the certificate on
such a folder (L1), or on a folder that a coordinator's `tvr note … --files` also records (N)?
Part L: does a second table inside num.md's section "Facts the library states" make its names citable?
Part C: the citation check and the release check's lint list on the tree.

Parts K and U compile small kernels with `lean -o` (never `lake build`) into the scratch folder, against the
library's built modules, and run the tree's claim reader on them, as evidence/tool-round16/run.py does. Part R
calls the tool's functions on a scratch copy (tools/, docs/, lean/ without its build, kernels/vec_add_f32) with a
passing gate made by hand. The tree is never written. No GPU.

Usage: python3 -I probes.py SCRATCH [K U R L C] > probes.out
  SCRATCH  an empty folder outside TVA; its path is never printed.
"""
import json
import os
import re
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
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True, text=True).stdout.strip()
    return OLEAN + os.pathsep + pre


def compile_and_read(name, text):
    """Compile `text` as module `name` into the scratch folder and run the tree's claim reader on it (all four
    `example`s, as the gate runs it). Returns the reader's JSON, or None."""
    f = os.path.join(LDIR, name + ".lean")
    o = os.path.join(OLEAN, name + ".olean")
    os.makedirs(OLEAN, exist_ok=True)
    open(f, "w").write(text)
    r = subprocess.run([LEAN, f"--root={LDIR}", "-o", o, f], cwd=LEAN_DIR, capture_output=True, text=True,
                       env=dict(os.environ, LEAN_PATH=lean_path()))
    if r.returncode:
        print(f"- {name}: does not compile: {mask(r.stdout + r.stderr)[:1500]}")
        return None
    reader = os.path.join(LDIR, f"reader{name}.lean")
    open(reader, "w").write(tvr.claim_reader_text(name, "Kernels.Demo", set()))
    r = subprocess.run([LEAN, reader], cwd=LEAN_DIR, capture_output=True, text=True, env=dict(os.environ, LEAN_PATH=lean_path()))
    js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
    if r.returncode or len(js) != 1:
        print(f"- {name}: the reader fails (shape FAIL): {mask(r.stdout + r.stderr)[:600]}")
        return None
    return json.loads(js[0][len("TVR_JSON "):])


def gate_lean_checks(surf):
    """The gate's checks on the reader's output, as `check` computes them."""
    consts = tvr.const_table(surf)
    out = {"shape": True}
    out["axioms"] = all(set(v) <= tvr.STD_AXIOMS for v in surf["axioms"].values()) or surf["axioms"]
    unused = [f for f, u in surf["usage"].items() if not u]
    out["usage"] = (not unused and not surf["destructured"]) or unused
    bad_ns = surf["shadow"] + [i["name"] for i in surf["instances"] if not i.get("internal")] + tvr.binder_problems(surf)
    out["namespace"] = bad_ns or True
    out["kinds"] = tvr.kinds_problems(surf, consts, "Kernels.Demo") or True
    return out


# ---------------------------------------------------------------- Part K

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
TAIL = """
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=
  fun _ _ h => (fun _ => h.spec) h.div

theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ init o => o = .ok init, dividend, fun _ init => ⟨.ok init, rfl⟩,
    ⟨fun _ _ init _ o h => ⟨init, h, rfl⟩, fun _ _ _ h => h⟩⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩

end Kernels.Demo
"""
SHIFTED = HEAD + ("structure Assumptions [Inhabited Unit] [Inhabited Bool] (K : Claim.Behaviour Inst) (P : Platform) : Prop "
                  "where\n") + FIELDS + TAIL
PLAIN = HEAD + "structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where\n" + FIELDS + TAIL


def part_k():
    print("# Part K: the claim reader's K and P, with parameters before them\n")
    for name, text, label in (("Plain", PLAIN, "control: `structure Assumptions (K …) (P …)`"),
                              ("Shifted", SHIFTED, "`structure Assumptions [Inhabited Unit] [Inhabited Bool] (K …) (P …)`")):
        print(f"## {label}\n")
        print("Fields: `[kernel] spec : Claim.Meets Inst.ok Pre Post K` (the specification itself) and `[kernel] div : ∀ x y r "
              ": FloatV .binary32, P.div .binary32 x y r → r = x` (a device fact); `main` is `h.spec`.\n")
        hyg = [what for what, pat in tvr.HYGIENE if re.search(pat, text, re.M)] + \
            (["a string holding {"] if tvr.brace_string_problems(text) else [])
        show("hygiene (the gate's patterns on the module's text)", hyg or "PASS")
        surf = compile_and_read(name, text)
        if not surf:
            continue
        prefix = subprocess.run([tvr.LAKE, "env", "lean", "--print-prefix"], cwd=LEAN_DIR, capture_output=True,
                                text=True).stdout.strip()
        r = subprocess.run([os.path.join(prefix, "bin", "leanchecker"), name], cwd=LEAN_DIR, capture_output=True, text=True,
                           env=dict(os.environ, LEAN_PATH=lean_path(), LEAN_SYSROOT=prefix))
        show("replay (leanchecker on the compiled module)", "PASS" if r.returncode == 0 else f"FAIL {(r.stdout + r.stderr)[:300]}")
        for f in surf["fields"]:
            u = surf["field_uses"][f]
            show(f"field {f}: the reader's flags", {"K": u["K"], "P": u["P"], "head": u["head"]})
        for k, v in gate_lean_checks(surf).items():
            show(k, "PASS" if v is True else f"FAIL {v}")
        print()


# ---------------------------------------------------------------- Part U

sys.path.insert(0, HERE)
from probes_src import MANIFEST, SOURCES  # noqa: E402


def part_u():
    print("# Part U: an unused function's def statement binds a module-level name the kernel reads\n")
    man = MANIFEST
    show("the manifest's classes (probes_src.MANIFEST)", man["source"])
    for name, src in SOURCES.items():
        print(f"## {name}_SRC\n")
        show("the source's functions (name, first, last)", tvr.source_functions(src))
        show("code lines outside every function", tvr.module_lines(src))
        code = tvr.code_lines(src)[0]
        mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code)
                          if any(a <= i <= b for a, b in man["source"]["ranges"])]}
        show("map_problems (every code line of the kernel lines mapped)", tvr.map_problems(man, mm, src, "x")[0] or "passes")
        print()


# ---------------------------------------------------------------- Part C

def part_l():
    print("# Part L: a second table in num.md's section \"Facts the library states\"\n")
    root = os.path.join(SCRATCH, "l")
    os.makedirs(os.path.join(root, "docs", "rules"))
    text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
    anchor = "Every fact chooses its constants with margin"
    assert anchor in text
    extra = ("The shapes behind these facts, which a field never cites:\n\n| Shape | Says |\n|---|---|\n"
             "| `MathAcc` | accuracy of a function |\n| `DotAcc` | accuracy of `tl.dot` |\n\n")
    open(os.path.join(root, "docs", "rules", "num.md"), "w", encoding="utf-8").write(text.replace(anchor, extra + anchor))
    show("the tree's page: facts listed", len(tvr.stated_facts() or {}))
    f = tvr.stated_facts(root)
    show("the page with a second table (of shapes, \"never cited\") inside the section, before its next heading: facts "
         "listed", f"{len(f)}; MathAcc listed alone: {f.get('TVA.Rules.Num.MathAcc') == [None]}, "
         f"DotAcc: {f.get('TVA.Rules.Num.DotAcc') == [None]}")
    u = {"head": "TVA.Rules.Num.MathAcc", "args": [{"platform": True}] + [{"pp": x} for x in
                                                                          (".exp", ".binary32", "Set.univ", "2^-24", "2^-150")]}
    show("then lr-9a's field `MathAcc P .exp .binary32 Set.univ (2^-24) (2^-150)`: rule_fact_problem",
         tvr.rule_fact_problem(u, "Kernels.Demo", f) or "PASS")
    print()


def part_c():
    print("# Part C: citations, and the release check's lint list\n")
    show("the tree's citations (citation_problems)", tvr.citation_problems() or "every cited file exists")
    show("the facts docs/rules/num.md lists (stated_facts)", len(tvr.stated_facts() or {}))
    show("the audit's fact-page check", tvr.fact_page_problems() or "nothing")
    show("the records the release check would lint now (release_records on the tree)", tvr.release_records(TVA))
    print()


# ---------------------------------------------------------------- Part R

VEC = "vec_add_f32"
REAL = {k: getattr(tvr, k) for k in ("ROOT", "KROOT", "LREVIEW", "LIBRARY_LEDGER")}
REPORT = ("# Review of {k}, round 2, by {w}\n\n## Verdict\n\n{v}\n{why}\n\n## Findings\n\n{f}\n\n## Accepted\n\n{a}\n\n"
          "## Checked\n\n- the probe\n\n## Not checked\n\nNothing.\n")


def use_root(root):
    if root is None:
        for k, v in REAL.items():
            setattr(tvr, k, v)
        return
    tvr.ROOT, tvr.KROOT = root, os.path.join(root, "kernels")
    tvr.LREVIEW = os.path.join(root, "docs", "library-review")
    tvr.LIBRARY_LEDGER = os.path.join(tvr.LREVIEW, "ledger.jsonl")


def quiet(fn, *args):
    import contextlib
    import io
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue().strip()


def read_ledger():
    return [json.loads(l) for l in open(tvr.kpath(VEC, "review", "ledger.jsonl")) if l.strip()]


def write_ledger(evs):
    evs = [dict(e) for e in evs]
    for i, e in enumerate(evs, 1):
        e["seq"] = i
    open(tvr.kpath(VEC, "review", "ledger.jsonl"), "w").write(
        "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in evs))


def cert():
    return tvr.cert_state(VEC, tvr.state(VEC), tvr.load_json(tvr.kpath(VEC, "gate.json")))[0]


def file_as(who, name, findings, accept, verdict):
    """Write `who`'s next filing of round 2 (`name` is filing.json, filing-2.json, …) and record it with tvr file."""
    d = tvr.kpath(VEC, "review", "round-2", who)
    os.makedirs(d, exist_ok=True)
    tvr.dump_json(os.path.join(d, name), {"reviewer": who, "kernel": VEC, "round": 2, "accept": accept, "release": [],
                                          "reraise": [], "findings": findings})
    f = "None." if not findings else "\n".join(f"- **{x['kind']}** `{x['target']}` — {x['summary']}" for x in findings)
    open(os.path.join(d, name.replace("filing", "report").replace(".json", ".md")), "w").write(REPORT.format(
        k=VEC, w=who, v=verdict, why="The probe.", f=f, a="All items." if accept == "all" else "None."))
    return quiet(tvr.cmd_file, VEC, who)[1].splitlines()[0]


BLOCKING = [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "the tolerance says nothing",
             "detail": "a blocking finding"}]


def part_r():
    import shutil
    print("# Part R: role folders and recorded folders of a kernel's ledger (L52, L81)\n")
    evs = read_ledger()
    legacy = [e["seq"] for e in evs if e["type"] in ("file", "response") and "dir" not in e]
    show("vec-add's ledger as it is in the tree: file and response events that name no `dir`", legacy)
    rec = {p for e in evs for p in e.get("files", {})}
    folders = {os.path.dirname(p) for e in evs if e["type"] in ("file", "response") for p in e.get("files", {})}
    loose = sorted(os.path.relpath(f, TVA) for d in folders for f in tvr.text_files(os.path.join(TVA, d))
                   if os.path.relpath(f, TVA) not in rec)
    show("files in the folders of those events that no event records", loose)
    show("ledger_problems on the tree (a file added to a recorded folder?)",
         tvr.ledger_problems(TVA, f"kernels/{VEC}/review/ledger.jsonl", "the ledger") or "none")
    # A scratch copy: vec-add's ledger up to r-vecadd's round-2 acceptance (events 1-9), the folders in progress
    # removed, and a passing gate made by hand at the item digests r-vecadd accepted (its snapshot kept, with the
    # packages' digest as its only cone), so that the certificate is granted on r-vecadd's acceptance.
    root = os.path.join(SCRATCH, "r")
    ign = shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv")
    for d in ("tools", "docs", "lean"):
        shutil.copytree(os.path.join(TVA, d), os.path.join(root, d), ignore=ign)
    shutil.copytree(os.path.join(TVA, "kernels", VEC), os.path.join(root, "kernels", VEC), ignore=ign)
    use_root(root)
    base = [e for e in evs if e["seq"] <= 9]
    write_ledger(base)
    shutil.rmtree(tvr.kpath(VEC, "review", "round-2", "r-vecadd2"), ignore_errors=True)
    for n in ("filing-2.json", "report-2.md"):
        if os.path.exists(tvr.kpath(VEC, "review", "round-2", "r-vecadd", n)):
            os.remove(tvr.kpath(VEC, "review", "round-2", "r-vecadd", n))
    acc = [e for e in base if e["type"] == "accept"][-1]
    items, gd = acc["items"], acc["gate"]
    assert tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items()))) == gd
    tvr.dump_json(tvr.kpath(VEC, "gate.json"), {"protocol": "tvr-2", "kernel": VEC, "time": "t", "ok": True, "digest": gd,
                                                "checks": [], "items": items, "kinds": {"faithful": "rule"},
                                                "files": tvr.kernel_files(VEC), "tool": tvr.tool_digest(),
                                                "build": tvr.build_inputs(VEC)})
    tvr.dump_json(tvr.kpath(VEC, "review", "snapshots", gd[:16] + ".json"),
                  {"items": {i: {"digest": d, "cone": {"packages": tvr.packages_digest()}} for i, d in items.items()}})
    show("scratch, events 1-9 and a hand-made gate at r-vecadd's accepted digests: kernel_record_problems",
         tvr.kernel_record_problems(VEC) or "none")
    show("  certificate", cert())
    # L1: lr-13a's Part R on vec-add's own folder: r-vecadd files again in round 2 (a blocking finding), then the
    # filing's `file` and `raise` lines are removed; its folder is also recorded by event 7, which names no `dir`.
    show("L1: r-vecadd files filing-2.json in round 2 with a blocking finding", file_as("r-vecadd", "filing-2.json", BLOCKING, [], "reject"))
    show("L1: the file event names its folder as `dir`", "dir" in read_ledger()[-2])
    show("L1: certificate with its lines", cert())
    write_ledger(base)
    show("L1: its file and raise lines removed: kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("L1: unrecorded_current", tvr.unrecorded_current(VEC) or "none")
    show("L1: the records step, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("L1: certificate (filing-2.json, a blocking SPEC_TOO_WEAK, still in the folder)", cert())
    # N: the same on a folder the current tool recorded (with `dir`), where a coordinator's note records the files
    for n in ("filing-2.json", "report-2.md"):
        os.remove(tvr.kpath(VEC, "review", "round-2", "r-vecadd", n))
    show("N: r-vecadd files filing-2.json, accepting every item", file_as("r-vecadd", "filing-2.json", [], "all", "accept"))
    show("N: certificate", cert())
    mid = read_ledger()
    show("N: r-vecadd files filing-3.json with a blocking finding", file_as("r-vecadd", "filing-3.json", BLOCKING, [], "reject"))
    files = [f"kernels/{VEC}/review/round-2/r-vecadd/filing-3.json", f"kernels/{VEC}/review/round-2/r-vecadd/report-3.md"]
    quiet(tvr.cmd_note, VEC, "memo: r-vecadd's third filing, for the record", files)
    show("N: the coordinator's `tvr note … --files` the filing and its report; certificate", cert())
    write_ledger(mid + [read_ledger()[-1]])
    show("N: filing-3's file and raise lines removed (the note kept): kernel_record_problems",
         tvr.kernel_record_problems(VEC) or "none")
    show("N: the records step, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("N: certificate (filing-3.json, a blocking SPEC_TOO_WEAK, still in the folder)", cert())
    use_root(None)
    print()


if __name__ == "__main__":
    parts = sys.argv[2:] or ["K", "U", "R", "L", "C"]
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"Python {sys.version.split()[0]}\n")
    for p, fn in (("K", part_k), ("U", part_u), ("R", part_r), ("L", part_l), ("C", part_c)):
        if p in parts:
            fn()
