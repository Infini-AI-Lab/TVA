"""The tool after round 20 (library issues L95, L106, L108): the gate reads docs/rules/num.md's facts section as
the page renders it, or accepts no fact (L95); the claim reader refuses an `Assumptions` that is not a structure,
and the gate's shape step fails on a reader that panicked (L106); a recorded path that is not normalised, a freeze
of `<folder>/work/../filing-2.json` included, is refused (L108).

Part L calls `facts_section`, `stated_facts`, `facts_section_tables`, `rule_fact_problem` and the audit's
`fact_page_problems` on the tree's num.md and on scratch copies of it with one change each. Part S compiles small
kernels with `lean -o` (never `lake build`) into the scratch folder, against the library's built modules, and runs
on each the tree's claim reader and the reader of the round's start (the same template without its `isStructure`
check, the one line round 20 added), each through `tvr.reader_result`, the gate's shape decision, and through the
round's-start decision (exit code 0 and one result line). Part F calls `is_work_file`, `freeze_problem` and
`ledger_problems` on vec-add's ledger as the tree holds it and on scratch copies with one path changed. The tree
is never written, except for out.txt. No GPU and no Triton run.

Usage: python3 -I evidence/tool-round20/run.py SCRATCH [PARTS] > evidence/tool-round20/out.txt
  SCRATCH  an empty folder outside TVA; its path is never printed.
  PARTS    any of l s f (all by default).
"""
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
SCRATCH = os.path.realpath(ARGS[0])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"
HOME = os.path.realpath(os.path.expanduser("~"))
REAL_ROOT = tvr.ROOT
VEC = "vec_add_f32"
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


# ---------------------------------------------------------------- Part L: the facts section as it renders (L95)

def part_l():
    print("# Part L (L95): the facts section holds only forms a line reader reads as the page renders them\n")
    got = tvr.facts_section()
    f = tvr.stated_facts()
    show("the tree's docs/rules/num.md: facts listed", len(f) if f else f)
    show("  the section's problems", got[1] if got and got[1] else "none")
    show("  the audit's fact-page check", [x for x in tvr.fact_page_problems() if "num.md" in x] or "nothing")
    root = os.path.join(SCRATCH, "l-root")
    os.makedirs(os.path.join(root, "docs", "rules"))
    page = os.path.join(root, "docs", "rules", "num.md")
    text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
    first = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
    end = next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))
    rows = lines[first:end]
    hdr, body = rows[:2], rows[2:]
    fake = ["| Fact (`TVA.Rules.Num`) | Says |", "|---|---|", "| `DivRn P .binary32` | IEEE division (false on the "
            "H200 for `tl.fdiv`) |"]
    # Each variant: what a reader of the rendered page sees, against what the gate read before round 20.
    variants = [
        ("control: a line of text between the heading and the table", lines[:first] + ["Text.", ""] + lines[first:]),
        ("the real table inside an HTML comment, a fake one after it (the page shows only the fake)",
         lines[:first] + ["<!--"] + rows + ["-->", ""] + fake + lines[end:]),
        ("a fenced block holding a fake table before the real one (the page shows it as code)",
         lines[:first] + ["```"] + fake + ["```", ""] + lines[first:]),
        ("the table indented by four spaces (the page shows code, no table)",
         lines[:first] + ["    " + r for r in rows] + lines[end:]),
        ("the table in a block quote", lines[:first] + ["> " + r for r in rows] + lines[end:]),
        ("the table in a list item", lines[:first] + ["- " + rows[0]] + ["  " + r for r in rows[1:]] + lines[end:]),
        ("a setext heading (text over `---`) before the table, which the page shows under it",
         lines[:first] + ["Not citable", "---", ""] + lines[first:]),
        ("a rule (`***`) before the table", lines[:first] + ["***", ""] + lines[first:]),
        ("text against the table (no blank line), which the page reads as the table's header",
         lines[:first] + ["Fact | Says"] + ["|---|---|"] + body + lines[end:]),
        ("a second `## Facts the library states` heading later on the page",
         lines + ["", "## Facts the library states", ""] + fake),
        ("the section's heading inside a fenced block, with the real one removed",
         lines[:start] + ["```", lines[start], "```"] + lines[start + 1:]),
    ]
    tvr_root = tvr.ROOT
    tvr.ROOT = root
    u_exp = {"head": "TVA.Rules.Num.ExpF32", "platform_fact": True, "args": [{"platform": True}]}
    u_d32 = {"head": "TVA.Rules.Num.DivRn", "platform_fact": False,
             "args": [{"platform": True}, {"type": "TVA.Math.Float.Format", "const": "TVA.Math.Float.Format.binary32"}]}
    try:
        for label, ls in variants:
            open(page, "w", encoding="utf-8").write("\n".join(ls))
            got = tvr.facts_section()
            f = tvr.stated_facts()
            print(f"## {label}\n")
            show("facts listed", len(f) if f else f"none ({f})")
            show("the section's problems", (got[1][:3] if got else "no section") if (got is None or got[1]) else "none")
            for what, u in (("`ExpF32 P`", u_exp), ("`DivRn P .binary32`", u_d32)):
                show(f"a `[rule]` field citing {what}", tvr.rule_fact_problem(u, "Kernels.Demo", f) or "PASS")
            show("the audit's fact-page check names num.md",
                 len([x for x in tvr.fact_page_problems() if "num.md" in x]) or "no")
            print()
    finally:
        tvr.ROOT = tvr_root


# ---------------------------------------------------------------- Part S: Assumptions is a structure (L106)

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
    ("Struct", "a structure with a `[kernel]` field that is the specification itself (control)",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] The kernel meets its specification. -/
  spec : Claim.Meets Inst.ok Pre Post K
""", "fun _ _ h => h.spec", f"⟨{MEETS}⟩"),
    ("Induct", "lr-18a's: an `inductive` with one constructor whose argument is the specification",
     """inductive Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  | mk (spec : Claim.Meets Inst.ok Pre Post K)
""", "fun _ _ h => match h with | .mk s => s", f"Assumptions.mk {MEETS}"),
    ("Def", "a `def` whose body is the specification",
     """def Assumptions (K : Claim.Behaviour Inst) (_ : Platform) : Prop := Claim.Meets Inst.ok Pre Post K
""", "fun _ _ h => h", MEETS),
]


def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True,
                         text=True).stdout.strip()
    return OLEAN + os.pathsep + pre


def old_reader(text):
    """The claim reader of the round's start: the tree's template without the `isStructure` check round 20 added
    (its comment and its two lines)."""
    lines = text.split("\n")
    i = next(i for i, l in enumerate(lines) if "`Assumptions` must be a `structure`" in l)
    j = next(j for j in range(i, len(lines)) if "TVR: Assumptions must be a structure" in lines[j])
    out = "\n".join(lines[:i] + lines[j + 1:])
    assert "isStructure" not in out
    return out


def part_s():
    print("# Part S (L106): the claim reader refuses an `Assumptions` that is not a structure, and the gate's shape "
          "step a reader that panicked\n")
    os.makedirs(OLEAN, exist_ok=True)
    env = dict(os.environ, LEAN_PATH=lean_path())
    for name, label, decl, main, proof in CASES:
        print(f"## {label}\n")
        text = (HEAD + decl + "\ntheorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K :=\n  "
                + main + "\n" + TAIL.replace("PROOF", proof))
        f = os.path.join(LDIR, name + ".lean")
        open(f, "w").write(text)
        r = subprocess.run([LEAN, f"--root={LDIR}", "-o", os.path.join(OLEAN, name + ".olean"), f], cwd=LEAN_DIR,
                           capture_output=True, text=True, env=env)
        if r.returncode:
            errs = [l for l in (r.stdout + r.stderr).splitlines() if "error" in l]
            show("the kernel", "does not compile: " + (errs[0] if errs else (r.stdout + r.stderr)[:300]))
            print()
            continue
        show("the kernel", "compiles")
        tree = tvr.claim_reader_text(name, "Kernels.Demo", set())
        for who, reader in (("the tree's reader", tree), ("the round's-start reader", old_reader(tree))):
            g = os.path.join(LDIR, f"reader{name}{'Old' if reader is not tree else ''}.lean")
            open(g, "w").write(reader)
            r2 = subprocess.run([LEAN, g], cwd=LEAN_DIR, capture_output=True, text=True, env=env)
            js = [l for l in r2.stdout.splitlines() if l.startswith("TVR_JSON ")]
            panic = [l for l in (r2.stdout + "\n" + r2.stderr).splitlines() if "PANIC" in l]
            errs = [l.split(": error: ", 1)[1] for l in r2.stdout.splitlines() if ": error: " in l]
            show(f"{who}", f"exit {r2.returncode}, {len(js)} result line(s), "
                 f"{'a PANIC line: ' + panic[0][:110] if panic else 'no PANIC line'}"
                 + (f"; error: {errs[0][:110]}" if errs else ""))
            surf, why = tvr.reader_result(r2.returncode, r2.stdout, r2.stderr)
            show("  the gate's shape step (`tvr.reader_result`)", ("PASS" if surf else "FAIL") + (f": {why[:150]}" if not surf else ""))
            if r2.returncode == 0 and len(js) == 1:
                s0 = json.loads(js[0][len("TVR_JSON "):])
                show("  the round's-start shape step (exit 0, one result line)", f"PASS, the claim's fields: {s0['fields']}")
            else:
                show("  the round's-start shape step (exit 0, one result line)", "FAIL")
        print()


# ---------------------------------------------------------------- Part F: recorded paths are normalised (L108)

def part_f():
    print("# Part F (L108): a recorded path that is not normalised is refused\n")
    evs = [json.loads(l) for l in open(os.path.join(TVA, "kernels", VEC, "review", "ledger.jsonl")) if l.strip()]
    fr = next(i for i, e in enumerate(evs) if e["type"] == "freeze")
    e = evs[fr]
    d = e["dir"]
    show("vec-add's freeze event", f"event {e['seq']}, folder {d}, {len(e['files'])} files")
    show("freeze_problem on it, as recorded", tvr.freeze_problem(VEC, evs[:fr], e) or "none (the tool's own)")
    for bad in (f"{d}/work/../filing.json", f"{d}/work/./Cover.lean", f"{d}//work/Cover.lean"):
        show(f"is_work_file({bad!r})", tvr.is_work_file(d, bad))
        e2 = dict(e, files=dict(e["files"], **{bad: "0" * 64}))
        show("  freeze_problem with it added", tvr.freeze_problem(VEC, evs[:fr], e2) or "none")
    # the ledger check on a scratch copy of vec-add's folder whose freeze records the filing through `work/..`
    root = os.path.join(SCRATCH, "f-root")
    shutil.copytree(os.path.join(TVA, "kernels", VEC), os.path.join(root, "kernels", VEC),
                    ignore=shutil.ignore_patterns(".lake", "__pycache__"))
    filing = f"{d}/filing.json"
    sneaky = f"{d}/work/../filing.json"
    e3 = dict(e, files=dict(e["files"], **{sneaky: tvr.file_sha(os.path.join(TVA, filing))}))
    path = f"kernels/{VEC}/review/ledger.jsonl"
    with open(os.path.join(root, path), "w") as fh:
        fh.write("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in evs[:fr] + [e3] + evs[fr + 1:]))
    show(f"ledger_problems on a copy whose freeze also records {sneaky!r} (the filing's own sha256)",
         [p for p in tvr.ledger_problems(root, path, "the copy's ledger", folders=False)] or "none")
    show("ledger_problems on the tree's vec-add ledger", tvr.ledger_problems(TVA, path, "the ledger", folders=False) or "none")
    lib = os.path.relpath(tvr.LIBRARY_LEDGER, TVA)
    show("ledger_problems on the tree's library ledger, about paths",
         [p for p in tvr.ledger_problems(TVA, lib, "the library ledger") if "normalised" in p] or "none")
    print()


if __name__ == "__main__":
    print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}, "
          f"tools/claim_reader.lean.in sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'claim_reader.lean.in'))[:16]}; "
          f"Python {sys.version.split()[0]}\n")
    parts = ARGS[1:] or ["l", "s", "f"]
    for p, fn in (("l", part_l), ("s", part_s), ("f", part_f)):
        if p in parts:
            fn()
