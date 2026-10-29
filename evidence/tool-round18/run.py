"""The tool after round 18 (library issues L79, L81, L94, L95, L96): the claim reader refuses an `Assumptions` with
other parameters than `K` and `P` (L94); the gate reads only the first table of docs/rules/num.md's facts section,
and the audit names a second one (L95); the header of an unused function is module code (L79); a kernel's role
folder is frozen by its own filing or response events, whether or not a line names it, a note accounts for none,
and `tvr freeze` records once the work files an early line left out (L81); the release check compares the pages
with those the copy shipped, read before the sandbox runs (L96).

Part K compiles small kernels with `lean -o` (never `lake build`) into the scratch folder, against the library's
built modules, and runs the tree's claim reader on them with all four `example`s, as the gate does. Parts L, U, R
and P call the tool's functions on small inputs and on scratch copies of the parts of the tree they read (tools/,
docs/, lean/ without its build, kernels/vec_add_f32), with the module's root pointed at the copy; Part R pins the
copy's vec-add ledger at a given number of events. Part V runs the tool itself (`tvr check`, `tvr freeze`, `tvr
status`) in a scratch copy, as the coordinator would after the merge: its `lean/.lake` holds a copy of the tree's
build and a link to the tree's packages (Mathlib, read only when its build is current). The tree is never written,
except for out.txt. No GPU and no Triton run.

Usage: python3 -I evidence/tool-round18/run.py SCRATCH [--semantics FILE] [PARTS] > evidence/tool-round18/out.txt
  SCRATCH          an empty folder outside TVA; its path is never printed.
  --semantics FILE Part V runs a second time with FILE as the copy's docs/SEMANTICS.md (its sha256 is printed, and
                   compared with the one vec-add's accepted snapshot records), so that the gate's digests can be those
                   its reviewers accepted.
  PARTS            any of k l u r p v (all by default).
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

ARGS = sys.argv[1:]
SEM = None
if "--semantics" in ARGS:
    i = ARGS.index("--semantics")
    SEM = os.path.realpath(ARGS[i + 1])
    del ARGS[i:i + 2]
SCRATCH = os.path.realpath(ARGS[0])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
REAL = {k: getattr(tvr, k) for k in ("ROOT", "KROOT", "LREVIEW", "LIBRARY_LEDGER")}
VEC = "vec_add_f32"
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    text = str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>")
    if SEM:
        text = text.replace(SEM, "<the given SEMANTICS.md>")
    return text.replace(HOME, "~")


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
    shutil.copy2(os.path.join(TVA, "kernels", "README.md"), os.path.join(root, "kernels", "README.md"))
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


def write_ledger(evs):
    for i, e in enumerate(evs, 1):
        e["seq"] = i
    open(tvr.kpath(VEC, "review", "ledger.jsonl"), "w").write(
        "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in evs))


def read_ledger():
    return [json.loads(l) for l in open(tvr.kpath(VEC, "review", "ledger.jsonl")) if l.strip()]


def pin_ledger(n):
    """Keep the first `n` events of vec-add's ledger in the scratch copy, and remove the files only later events
    record (and the folders left empty). A file a later `freeze` event records is kept: `tvr freeze` records work
    files that were there before it (vec-add's round-1 work files, frozen by its event 15), so that the copy pinned
    before the freeze is the tree as it was then (library issue L108: lr-18a, lr-18b)."""
    evs = read_ledger()
    kept = {p for e in evs[:n] for p in e.get("files", {})}
    kept |= {p for e in evs[n:] if e.get("type") == "freeze" for p in e.get("files", {})}
    for p in sorted({p for e in evs[n:] for p in e.get("files", {})} - kept):
        if os.path.isfile(os.path.join(tvr.ROOT, p)):
            os.remove(os.path.join(tvr.ROOT, p))
    for dirpath, _, _ in sorted(os.walk(tvr.kpath(VEC, "review")), reverse=True):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    write_ledger(evs[:n])


# ---------------------------------------------------------------- Part K: the parameters of Assumptions (L94)

def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True, text=True).stdout.strip()
    return OLEAN + os.pathsep + pre


def compile_and_read(name, text):
    """Compile `text` as module `name` into the scratch folder and run the tree's claim reader on it (all four
    `example`s, as the gate runs it). Returns (the reader's JSON or None, what failed)."""
    f = os.path.join(LDIR, name + ".lean")
    o = os.path.join(OLEAN, name + ".olean")
    os.makedirs(OLEAN, exist_ok=True)
    open(f, "w").write(text)
    r = subprocess.run([LEAN, f"--root={LDIR}", "-o", o, f], cwd=LEAN_DIR, capture_output=True, text=True,
                       env=dict(os.environ, LEAN_PATH=lean_path()))
    if r.returncode:
        errs = [l for l in (r.stdout + r.stderr).splitlines() if "error" in l]
        return None, "the kernel does not compile: " + (errs[0] if errs else (r.stdout + r.stderr)[:300])
    reader = os.path.join(LDIR, f"reader{name}.lean")
    open(reader, "w").write(tvr.claim_reader_text(name, "Kernels.Demo", set()))
    r = subprocess.run([LEAN, reader], cwd=LEAN_DIR, capture_output=True, text=True, env=dict(os.environ, LEAN_PATH=lean_path()))
    js = [l for l in r.stdout.splitlines() if l.startswith("TVR_JSON ")]
    if r.returncode or len(js) != 1:
        errs = [l for l in r.stdout.splitlines() if "error" in l]
        return None, "the reader fails (shape FAIL): " + " | ".join(errs)[:500]
    return json.loads(js[0][len("TVR_JSON "):]), ""


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


# lr-16a's Part K (docs/library-review/round-16/lr-16a/work/probes.py): two [kernel] fields that a correct reader
# refuses in `kinds`, the specification itself (it mentions K) and a device fact (it mentions P).
K_HEAD = """import TVA

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
K_FIELDS = """  /-- [kernel] The kernel meets its specification. -/
  spec : Claim.Meets Inst.ok Pre Post K
  /-- [kernel] `/` on `float32` returns its dividend (a device fact, false on the H200). -/
  div : ∀ x y r : FloatV .binary32, P.div .binary32 x y r → r = x
"""
K_TAIL = """
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=
  fun _ _ h => (fun _ => h.spec) h.div

theorem consistent : Claim.Consistent Assumptions :=
  ⟨fun _ init o => o = .ok init, dividend, fun _ init => ⟨.ok init, rfl⟩,
    ⟨fun _ _ init _ o h => ⟨init, h, rfl⟩, fun _ _ _ h => h⟩⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨⟨0⟩, fun _ => none, trivial, trivial⟩

end Kernels.Demo
"""
K_CASES = [
    ("Plain", "(K : Claim.Behaviour Inst) (P : Platform)", "control: the two parameters"),
    ("Shifted", "[Inhabited Unit] [Inhabited Bool] (K : Claim.Behaviour Inst) (P : Platform)",
     "lr-16a: two instance parameters before K and P"),
    ("Trailing", "(K : Claim.Behaviour Inst) (P : Platform) [Inhabited Unit]",
     "an instance parameter after them (`Claim.Consistent Assumptions` does not elaborate)"),
    ("Defaulted", "(K : Claim.Behaviour Inst) (P : Platform) (n : Nat := 0)",
     "a third parameter with a default, which `Assumptions K P` fills in"),
]


def part_k():
    print("# Part K (L94): Assumptions has exactly the parameters K and P\n")
    print("Fields: `[kernel] spec : Claim.Meets Inst.ok Pre Post K` (the specification itself) and `[kernel] div : ∀ x y "
          "r : FloatV .binary32, P.div .binary32 x y r → r = x` (a device fact); `main` is `h.spec`.\n")
    for name, params, label in K_CASES:
        text = K_HEAD + f"structure Assumptions {params} : Prop where\n" + K_FIELDS + K_TAIL
        surf, why = compile_and_read(name, text)
        show(f"{label}: `structure Assumptions {params}`", why or "the reader elaborates (shape PASS)")
        if not surf:
            continue
        for f in surf["fields"]:
            u = surf["field_uses"][f]
            show(f"  field {f}: the reader's flags", {"K": u["K"], "P": u["P"], "head": u["head"]})
        for k, v in gate_lean_checks(surf).items():
            show(f"  {k}", "PASS" if v is True else f"FAIL {v}")
    print()


# ---------------------------------------------------------------- Part L: the facts section's one table (L95)

def part_l():
    print("# Part L (L95): the gate reads the first table of the facts section only, and the audit names a second\n")
    f = tvr.stated_facts()
    show("the tree's docs/rules/num.md: facts listed", len(f) if f else f)
    show("  its section's tables", len(tvr.facts_section_tables() or []))
    show("  the audit's fact-page check", [x for x in tvr.fact_page_problems() if "num.md" in x] or "nothing")
    root = os.path.join(SCRATCH, "l-root")
    os.makedirs(os.path.join(root, "docs", "rules"))
    page = os.path.join(root, "docs", "rules", "num.md")
    text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
    first = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
    end = next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))
    extra = ["| Shape | Says |", "|---|---|", "| `MathAcc` | an accuracy shape |", "| `DivRn P .binary32` | IEEE division |"]
    variants = [
        ("lr-16b's: a `### Shapes the facts use (not citable)` subsection with a table",
         lines[:end] + ["", "### Shapes the facts use (not citable)", ""] + extra + lines[end:]),
        ("a second table after a line of text, no heading", lines[:end] + ["", "More:", ""] + extra + lines[end:]),
        ("the table split in two by a blank line", lines[:first + 4] + [""] + lines[first + 4:]),
        ("the only table moved under a `###` heading", lines[:first] + ["### The table", ""] + lines[first:]),
    ]
    use_root(root)
    u_acc = {"head": "TVA.Rules.Num.MathAcc", "platform_fact": False, "args": [{"platform": True}] + [{}] * 5}
    u_d32 = {"head": "TVA.Rules.Num.DivRn", "platform_fact": False,
             "args": [{"platform": True}, {"type": "TVA.Math.Float.Format", "const": "TVA.Math.Float.Format.binary32"}]}
    u_exp = {"head": "TVA.Rules.Num.ExpF32", "platform_fact": True, "args": [{"platform": True}]}
    for label, ls in variants:
        open(page, "w", encoding="utf-8").write("\n".join(ls))
        f = tvr.stated_facts()
        show(f"num.md with {label}: facts listed", len(f) if f else f"none ({f})")
        show("  its section's tables", len(tvr.facts_section_tables() or []))
        for what, u in (("`MathAcc P …`", u_acc), ("`DivRn P .binary32`", u_d32), ("`ExpF32 P`", u_exp)):
            show(f"  {what}", tvr.rule_fact_problem(u, "Kernels.Demo", f) or "PASS")
        show("  the audit's fact-page check", [x for x in tvr.fact_page_problems() if "num.md" in x] or "nothing")
    use_root(None)
    print()


# ---------------------------------------------------------------- Part U: an unused function's header (L79)

# lr-16a's Part U sources (docs/library-review/round-16/lr-16a/work/probes_src.py): the unused function's default
# rebinds, when Python executes its `def` at import, a name the kernel reads (on the H200 both kernels store 3·x,
# lr-16a's work/unused_gpu.out).
sys.path.insert(0, os.path.join(TVA, "docs", "library-review", "round-16", "lr-16a", "work"))
from probes_src import MANIFEST, SOURCES  # noqa: E402

NESTED = '''import triton
import triton.language as tl


@triton.jit
def k(x_ptr):
    tl.store(x_ptr, 1)


def outer(a=1):
    def inner(b=(k := None)):
        return b
    return inner


class Holder:
    """A docstring, not code."""

    def method(self, c=2):
        return c
'''


def part_u():
    print("# Part U (L79): the header of an unused function is module code\n")
    man = json.loads(json.dumps(MANIFEST))
    show("lr-16a's manifest (probes_src.MANIFEST)", man["source"])
    for name, src in SOURCES.items():
        print(f"## lr-16a's {name}_SRC\n")
        unused = man["source"]["unused"]
        show("the headers of its unused functions (def_headers)", sorted(tvr.def_headers(src, set(unused))))
        show("its module lines (module_lines with source.unused)", tvr.module_lines(src, unused))
        code = tvr.code_lines(src)[0]

        def case(label, ranges, host):
            m = {"source": {"ranges": ranges, "host": host, "unused": unused}}
            mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if any(a <= i <= b for a, b in ranges)]}
            show(label, tvr.map_problems(m, mm, src, "x")[0] or "passes")
        case("lr-16a's manifest, every code line of the kernel lines mapped", [[1, 17]], [[27, 31]])
        case("the headers in the host lines (the reviewer then judges what they bind)", [[1, 17]], [[18, 19], [23, 23], [27, 31]])
        case("the headers in the kernel lines, mapped", [[1, 19], [23, 23]], [[27, 31]])
    print("## A nested unused function, and a method\n")
    unused = ["outer", "inner", "method"]
    show("headers of `outer`, `inner` (inside `outer`) and `method` (in a class)", sorted(tvr.def_headers(NESTED, set(unused))))
    show("module lines", tvr.module_lines(NESTED, unused))
    code = tvr.code_lines(NESTED)[0]
    m = {"source": {"ranges": [[1, 7]], "host": [[10, 10], [16, 16], [19, 19]], "unused": unused}}
    mm = {"entries": [{"src": [i], "model": "x"} for i in sorted(code) if 1 <= i <= 7]}
    show("kernel lines [[1, 7]], host lines 10, 16, 19 (the headers of outer and method, the class line)",
         tvr.map_problems(m, mm, NESTED, "x")[0] or "passes")
    man = tvr.manifest(VEC)
    src = open(tvr.kpath(VEC, man["source"]["file"]), encoding="utf-8", newline="").read()
    mm = tvr.load_json(tvr.kpath(VEC, "model-map.json"))
    lean = " ".join(open(p, encoding="utf-8").read() for p in tvr.kernel_lean_files(VEC))
    show(f"vec-add's manifest as shipped (unused {man['source'].get('unused', [])})", tvr.map_problems(man, mm, src, lean)[0] or "passes")
    # claim.md §2 as `tvr check` renders it, for lr-16a's JIT source as a kernel of a scratch copy
    root = scratch_root("u")
    use_root(root)
    m = json.loads(json.dumps(tvr.manifest(VEC)))
    m["source"].update({"ranges": [[1, 17]], "host": [[18, 19], [23, 23], [27, 31]], "unused": MANIFEST["source"]["unused"]})
    src = SOURCES["JIT"]
    gate = {"time": "t", "ok": False, "digest": "0" * 64, "checks": []}
    items = {i: {"digest": "0" * 64, "cone": [], "files": {}} for i in ("conditions", "spec", "model", "map")}
    surf = {"main_type": "…", "fields": [], "field_docs": {}, "field_types": {}, "field_uses": {}, "usage": {}}
    tvr.render_claim(VEC, m, gate, surf, {}, items, {"entries": []}, src.split("\n"), tvr.jit_functions(src),
                     tvr.source_functions(src))
    text = open(tvr.kpath(VEC, "claim.md"), encoding="utf-8").read()
    para = next(p for p in text.split("\n\n") if p.startswith("The code outside every function"))
    show("claim.md §2 for JIT_SRC, the headers in the host lines", para)
    use_root(None)
    print()


# ---------------------------------------------------------------- Part R: role folders and their files (L81)

REPORT = ("# Review of {k}, round 2, by {w}\n\n## Verdict\n\n{v}\n{why}\n\n## Findings\n\n{f}\n\n## Accepted\n\n{a}\n\n"
          "## Checked\n\n- the probe\n\n## Not checked\n\nNothing.\n")


def accepted_gate(n_accept):
    """A passing gate made by hand at the digests of the ledger's accept event `n_accept` (its snapshot rewritten to
    record no library file, so that it is current), so that the certificate turns on the records alone."""
    e = read_ledger()[n_accept - 1]
    assert e["type"] == "accept", e
    items = dict(e["items"])
    gd = tvr.sha("\n".join(f"{i} {d}" for i, d in sorted(items.items())))
    assert gd == e["gate"], "the accepted items are the gate's"
    tvr.dump_json(tvr.kpath(VEC, "gate.json"), {"protocol": "tvr-2", "kernel": VEC, "time": "t", "ok": True, "digest": gd,
                                                "checks": [], "items": items, "kinds": {}, "files": tvr.kernel_files(VEC),
                                                "tool": tvr.tool_digest(), "build": tvr.build_inputs(VEC)})
    tvr.dump_json(tvr.kpath(VEC, "review", "snapshots", gd[:16] + ".json"),
                  {"items": {i: {"digest": d, "cone": {"packages": tvr.packages_digest()}} for i, d in items.items()}})


def cert():
    c, blocking, full = tvr.cert_state(VEC, tvr.state(VEC), tvr.load_json(tvr.kpath(VEC, "gate.json")))
    return "GRANTED" if c else f"not granted ({len(blocking)} blocking; records: {len(tvr.kernel_record_problems(VEC))} problem(s))"


def part_r():
    print("# Part R (L81): a role folder is frozen by its own filing or response, and only by it\n")
    n = 14
    root = scratch_root("r0")
    use_root(root)
    pin_ledger(n)
    print(f"## vec-add's ledger at its first {n} events (as shipped before the coordinator's freeze)\n")
    show("the events naming no folder (`dir`)", [e["seq"] for e in read_ledger() if e["type"] in ("file", "response") and not e.get("dir")])
    show("kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("the records step, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("legacy_folders (what tvr freeze would record)", tvr.legacy_folders(VEC) or "none")
    show("tvr freeze vec_add_f32 --dry-run", quiet(tvr.cmd_freeze, VEC, True))
    show("the ledger unchanged by --dry-run (events)", len(read_ledger()))
    code, out = quiet(tvr.cmd_freeze, VEC)
    show("tvr freeze vec_add_f32", (code, out.splitlines()[0]))
    e = read_ledger()[-1]
    show("the event it appended", {x: v for x, v in e.items() if x != "time"})
    show("kernel_record_problems after it", tvr.kernel_record_problems(VEC) or "none")
    show("the records step after it, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    show("tvr freeze vec_add_f32 again", quiet(tvr.cmd_freeze, VEC))
    review = open(tvr.kpath(VEC, "REVIEW.md"), encoding="utf-8").read()
    show("REVIEW.md's line for it", next((l for l in review.split("\n") if "froze" in l), "none")[26:])
    # freeze lines written by hand: each must be one `tvr freeze` writes
    evs = read_ledger()
    w1 = "kernels/vec_add_f32/review/round-1/r-vecadd"
    w2 = "kernels/vec_add_f32/review/round-2/r-vecadd"
    some = {f"{w1}/work/Cover.lean": tvr.file_sha(os.path.join(root, w1, "work", "Cover.lean"))}
    hand = [
        ("a second freeze of round-1/r-vecadd", {"actor": "coordinator", "dir": w1, "files": some}),
        ("a freeze of round-2/r-vecadd, which event 11 names as `dir`", {"actor": "coordinator", "dir": w2, "files": {
            f"{w2}/work/Names.lean": tvr.file_sha(os.path.join(root, w2, "work", "Names.lean"))}}),
        ("a freeze by r-vecadd", {"actor": "r-vecadd", "dir": w1, "files": some}),
        ("a freeze of round-1/r-two, which no event records", {"actor": "coordinator", "dir": "kernels/vec_add_f32/review/round-1/r-two",
                                                               "files": {}}),
        ("a note naming a folder as `dir`", {"type": "note", "actor": "coordinator", "dir": w1, "text": "x", "files": {}}),
    ]
    for label, extra in hand:
        write_ledger([dict(x) for x in evs] + [{"type": "freeze", "round": 3, "time": "t", **extra}])
        show(f"a hand-written line, {label}", [x for x in tvr.kernel_event_problems(root, VEC)] or "none")
    # the first freeze written by hand to record a filing, before the tool's own
    write_ledger([dict(x) for x in evs[:n]] + [{"type": "freeze", "round": 3, "time": "t", "actor": "coordinator", "dir": w1,
                                                "files": {f"{w1}/filing.json": tvr.file_sha(os.path.join(root, w1, "filing.json"))}}])
    show("a hand-written freeze of round-1/r-vecadd recording its filing.json", tvr.kernel_event_problems(root, VEC) or "none")
    write_ledger([dict(x) for x in evs])
    use_root(None)
    print()

    print("## lr-16a's routes (round-16/lr-16a/work/probes.out, Part R): vec-add at events 1–9, a gate at r-vecadd's "
          "accepted digests\n")
    root = scratch_root("r1")
    use_root(root)
    pin_ledger(9)
    accepted_gate(9)
    show("certificate at events 1–9", cert())
    show("tvr freeze vec_add_f32 (the round-1 folder's four work files)", quiet(tvr.cmd_freeze, VEC)[1].splitlines()[0])
    show("certificate after it", cert())
    d1 = tvr.kpath(VEC, "review", "round-2", "r-vecadd")
    tvr.dump_json(os.path.join(d1, "filing-2.json"), {
        "reviewer": "r-vecadd", "kernel": VEC, "round": 2, "accept": [], "release": [], "reraise": [],
        "findings": [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "the tolerance says nothing",
                      "detail": "a blocking finding"}]})
    open(os.path.join(d1, "report-2.md"), "w").write(REPORT.format(
        k=VEC, w="r-vecadd", v="reject", why="A blocking finding.", f="- **SPEC_TOO_WEAK** `item:spec` — the tolerance says nothing",
        a="None."))
    show("L1: r-vecadd files filing-2.json in round-2/r-vecadd (event 7 names no `dir`), a blocking SPEC_TOO_WEAK",
         quiet(tvr.cmd_file, VEC, "r-vecadd")[1].splitlines()[0])
    show("L1: certificate", cert())
    evs = read_ledger()
    write_ledger([dict(e) for e in evs if not (e["seq"] > 10 and e["type"] in ("file", "raise"))])
    show("L1: the filing's `file` and `raise` lines removed; kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("L1: certificate", cert())
    show("L1: tvr freeze vec_add_f32", quiet(tvr.cmd_freeze, VEC))
    # N: a coordinator's note of the filing and its report
    show("N: tvr note vec_add_f32 … --files <the filing> <its report>", quiet(
        tvr.cmd_note, VEC, "the filing", [f"kernels/{VEC}/review/round-2/r-vecadd/filing-2.json",
                                          f"kernels/{VEC}/review/round-2/r-vecadd/report-2.md"]))
    note = {"type": "note", "actor": "coordinator", "round": 2, "time": "t", "text": "the filing",
            "files": tvr.archive_entry(os.path.join(d1, "filing-2.json"), os.path.join(d1, "report-2.md"))}
    write_ledger([dict(e) for e in read_ledger()] + [note])
    show("N: the same note line written by hand; kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("N: certificate", cert())
    write_ledger([dict(e) for e in evs])
    show("the filing's lines restored: kernel_record_problems", tvr.kernel_record_problems(VEC) or "none")
    show("certificate (the blocking finding)", cert())
    use_root(None)
    print()


# ---------------------------------------------------------------- Part P: the pages the copy shipped (L96)

def part_p():
    print("# Part P (L96): the release check compares the regenerated pages with those the copy shipped\n")
    tree = scratch_root("p-tree")   # the live tree
    copy = scratch_root("p-copy")   # the release check's copy
    claim = f"kernels/{VEC}/claim.md"
    shipped = tvr.shipped_records(copy)
    show("pages and gates read from the copy before the sandbox", (sorted(shipped["pages"]), sorted(shipped["gates"])))

    def regenerate(root, time):
        """What `tvr check` does to the pages' gate time: a new time in gate.json and in claim.md."""
        g = json.load(open(os.path.join(root, "kernels", VEC, "gate.json")))
        old = g["time"]
        g["time"] = time
        json.dump(g, open(os.path.join(root, "kernels", VEC, "gate.json"), "w"))
        p = os.path.join(root, claim)
        text = open(p).read()
        open(p, "w").write(text.replace(old, time))
    regenerate(copy, "2026-10-28T23:59:59Z")
    show("the sandbox regenerates the same pages at another gate time: old (the live tree) / new (the shipped copy)",
         (tvr.page_mismatch(tree, copy) or "none", tvr.page_mismatch(shipped, copy) or "none"))
    regenerate(tree, "2026-10-29T00:00:01Z")
    open(os.path.join(tree, claim), "a").write("\nA line a later `tvr check` added in the live tree.\n")
    show("A: the live tree's claim.md rewritten during the run: old / new",
         (tvr.page_mismatch(tree, copy) or "none", tvr.page_mismatch(shipped, copy) or "none"))
    copy2 = scratch_root("p-copy2")
    stale = open(os.path.join(copy2, claim)).read()
    open(os.path.join(copy2, claim), "w").write(stale + "\nA stale line the copy shipped.\n")
    shipped2 = tvr.shipped_records(copy2)
    open(os.path.join(copy2, claim), "w").write(stale)  # the sandbox regenerates the right page in the copy
    tree2 = scratch_root("p-tree2")  # the live tree, its page fixed during the run (the regenerated one)
    show("B: the copy shipped a stale claim.md, fixed in the live tree during the run: old / new",
         (tvr.page_mismatch(tree2, copy2) or "none", tvr.page_mismatch(shipped2, copy2) or "none"))
    print()


# ---------------------------------------------------------------- Part V: the coordinator's commands (L81)

def tvr_cli(root, *args):
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", *args], cwd=root, capture_output=True, text=True,
                       env=dict(os.environ, TMPDIR=os.path.join(SCRATCH, "tmp")))
    return r.returncode, (r.stdout + r.stderr).strip()


def part_v():
    print("# Part V (L81): the coordinator's commands on a copy of the tree: `tvr check`, then `tvr freeze`\n")
    os.makedirs(os.path.join(SCRATCH, "tmp"), exist_ok=True)
    for sem in [None] + ([SEM] if SEM else []):
        print(f"## With {'the given' if sem else 'the tree' + chr(39) + 's'} docs/SEMANTICS.md\n")
        copy_v(sem)


def copy_v(sem):
    root = scratch_root("v-given" if sem else "v-tree")
    use_root(root)
    pin_ledger(14)
    use_root(None)
    lake = os.path.join(root, "lean", ".lake")
    os.makedirs(lake)
    shutil.copytree(os.path.join(LEAN_DIR, ".lake", "build"), os.path.join(lake, "build"), symlinks=True)
    os.symlink(os.path.join(LEAN_DIR, ".lake", "packages"), os.path.join(lake, "packages"))
    snap = json.load(open(os.path.join(root, "kernels", VEC, "review", "snapshots", "401e1fa92db421d7.json")))
    accepted = snap["items"]["model"]["cone"].get("file:docs/SEMANTICS.md")
    if sem:
        shutil.copy2(sem, os.path.join(root, "docs", "SEMANTICS.md"))
    have = tvr.file_sha(os.path.join(root, "docs", "SEMANTICS.md"))
    whose = "the given file" if sem else "the tree's"
    show("docs/SEMANTICS.md in the copy", f"{whose}, sha256 {have[:16]}; the one vec-add's accepted gate 401e1fa92db421d7 "
         f"records: {accepted[:16]} ({'the same' if have == accepted else 'another'})")
    code, out = tvr_cli(root, "check", VEC)
    show("tvr check vec_add_f32", (code, [l for l in out.splitlines() if l.startswith(("FAIL", "gate"))]))
    for cmd in (["status", VEC], ["freeze", VEC, "--dry-run"], ["freeze", VEC], ["status", VEC]):
        code, out = tvr_cli(root, *cmd)
        lines = out.splitlines()
        show("tvr " + " ".join(cmd), (code, [l for l in lines if l.startswith(("**Certificate", "froze", "would", "nothing"))
                                             or "freeze" in l][:3]))
    new = os.path.join(root, "kernels", VEC, "review", "certificate.json")
    new = json.load(open(new)) if os.path.exists(new) else None
    gate = json.load(open(os.path.join(root, "kernels", VEC, "gate.json")))
    show("certificate.json after it", "none" if new is None else
         f"gate {new['gate'][:16]} (the gate's: {new['gate'] == gate['digest']}), items {sorted(new['items'])} at the gate's "
         f"digests ({new['items'] == gate['items']}), reviewers {new['reviewers']}")
    use_root(root)
    show("the records step on the copy, about kernels/", [x for x in tvr.records_problems(root) if x.startswith("kernels/")] or "none")
    use_root(None)
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"Python {sys.version.split()[0]}\n")
    parts = ARGS[1:] or ["k", "l", "u", "r", "p", "v"]
    for p, fn in (("k", part_k), ("l", part_l), ("u", part_u), ("r", part_r), ("p", part_p), ("v", part_v)):
        if p in parts:
            fn()
