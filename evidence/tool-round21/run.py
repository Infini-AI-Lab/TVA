"""The tool after round 21 (library issues L95, L112): the gate reads docs/rules/num.md's facts section as a
CommonMark and GFM renderer shows it, or accepts no fact (L95); the claim reader refuses a `private` field of
`Assumptions` by name, and the gate's shape step reads the word PANIC only on lines other than the reader's result
line (L112).

Part L calls `facts_section`, `stated_facts`, `rule_fact_problem` and the audit's `fact_page_problems` on the
tree's num.md and on scratch copies of it with one change each: the forms of round 20's review
(`docs/library-review/round-20/lr-20c/work/facts_probe.py`, where the round's-start tool read 38 facts from each) and
four more. Part R compiles small kernels with `lean -o` (never `lake build`) into the scratch folder, against the
library's built modules, runs the tree's claim reader on each and gives its output to `tvr.reader_result` (the
gate's shape decision) and the kinds and usage decisions; it also gives `tvr.reader_result` two made-up outputs.
The tree is never written, except for out.txt. No GPU and no Triton run.

Usage: python3 -I evidence/tool-round21/run.py SCRATCH [PARTS] > evidence/tool-round21/out.txt
  SCRATCH  an empty folder outside TVA; its path is never printed.
  PARTS    any of l r (both by default).
"""
import hashlib
import json
import os
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
PARTS = set(ARGS[1:]) or {"l", "r"}
HOME = os.path.realpath(os.path.expanduser("~"))
LEAN_DIR = os.path.join(TVA, "lean")
LDIR = os.path.join(SCRATCH, "lean")
OLEAN = os.path.join(LDIR, "olean")
LEAN = os.path.join(os.path.dirname(tvr.LAKE), "lean")


def mask(text):
    return str(text).replace(SCRATCH, "<scratch>").replace(TVA, "<tva>").replace(HOME, "~")


def show(label, value):
    print(f"- {label}: {mask(value)}")


def part_l():
    print("# Part L (L95): what the gate reads as the facts table is what the page shows as it\n")
    got = tvr.facts_section()
    f = tvr.stated_facts()
    show("the tree's docs/rules/num.md: facts listed", len(f) if f else f)
    show("  the section's problems", got[1] if got and got[1] else "none")
    show("  the audit's fact-page check", [x for x in tvr.fact_page_problems() if "num.md" in x] or "nothing")
    print()
    root = os.path.join(SCRATCH, "l-root")
    os.makedirs(os.path.join(root, "docs", "rules"))
    page = os.path.join(root, "docs", "rules", "num.md")
    lines = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
    first = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
    end = next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))
    nxt = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    row = "| `DivRn P .binary32` | x | x | x | x |"
    variants = [
        ("control: a line of text between the heading and the table", lines[:first] + ["Text.", ""] + lines[first:],
         True),
        ("control: a tilde fence and a backtick fence, each closed, before the section",
         lines[:start] + ["~~~", "| `DivRn P .binary32` |", "~~~", "", "````text", "```", "````", ""] + lines[start:],
         True),
        ("a ```` fence, a ``` line inside it (too short to close it), the section, then ```` (lr-20c 1)",
         lines[:start] + ["````", "```"] + lines[start:nxt + 1] + ["````"] + lines[nxt + 1:], False),
        ("a ``` fence, a ```python line inside it (a closer has no info string), the section, then ``` (lr-20c 2)",
         lines[:start] + ["```", "```python"] + lines[start:nxt + 1] + ["```"] + lines[nxt + 1:], False),
        ("a line ```a`b (no fence: its info string holds a backtick), then a ``` fence around the section (lr-20c 3)",
         lines[:start] + ["```a`b", "```"] + lines[start:nxt + 1] + ["```"] + lines[nxt + 1:], False),
        ("<script> before the heading, </script> after the next ## heading (lr-20c 4)",
         lines[:start] + ["<script>", ""] + lines[start:nxt + 1] + ["</script>"] + lines[nxt + 1:], False),
        ("<pre> before the heading, </pre> after the next ## heading (lr-20c 5)",
         lines[:start] + ["<pre>", ""] + lines[start:nxt + 1] + ["</pre>"] + lines[nxt + 1:], False),
        ("a row that starts `||` (GFM: an empty first cell, the fact in the second column) (lr-20c 6)",
         lines[:end] + ["|" + row] + lines[end:], None),
        ("inline HTML: `<details>` in a text line before the table (lr-20c 7)",
         lines[:first] + ["Older facts: <details><summary>show</summary>", ""] + lines[first:], False),
        ("inline HTML: `<sub>` in a cell (lr-20c 7)",
         lines[:end] + ["| <sub>`DivRn P .binary32`</sub> | x | x | x | x |"] + lines[end:], False),
        ("a struck-through first cell (lr-20c's probe, its last case)",
         lines[:end] + ["| ~~`DivRn P .binary32`~~ (withdrawn) | x | x | x | x |"] + lines[end:], False),
        ("a header row with one cell more than the delimiter row (GFM shows no table)",
         lines[:first] + [lines[first] + " Extra |"] + lines[first + 1:], False),
        ("a list item's fence (two spaces), then a ``` line at the margin opening a fence around the section",
         lines[:start] + ["- item", "  ```", "```"] + lines[start:nxt + 1] + ["```"] + lines[nxt + 1:], False),
        ("a fence indented by one space around the section",
         lines[:start] + [" ```"] + lines[start:nxt + 1] + [" ```"] + lines[nxt + 1:], False),
    ]
    u_d32 = {"head": "TVA.Rules.Num.DivRn", "platform_fact": False,
             "args": [{"platform": True}, {"type": "TVA.Math.Float.Format", "const": "TVA.Math.Float.Format.binary32"}]}
    u_exp = {"head": "TVA.Rules.Num.ExpF32", "platform_fact": True, "args": [{"platform": True}]}
    saved = tvr.ROOT
    tvr.ROOT = root
    try:
        for label, ls, keeps in variants:
            open(page, "w", encoding="utf-8").write("\n".join(ls))
            got = tvr.facts_section()
            f = tvr.stated_facts()
            print(f"## {label}\n")
            show("facts listed", len(f) if f else f"none ({f})")
            show("the section's problems", (got[1][:2] if got else "no section") if (got is None or got[1]) else "none")
            for what, u in (("`ExpF32 P`", u_exp), ("`DivRn P .binary32`", u_d32)):
                show(f"a `[rule]` field citing {what}", tvr.rule_fact_problem(u, "Kernels.Demo", f) or "PASS")
            show("the audit names num.md", len([x for x in tvr.fact_page_problems() if "num.md" in x]) or "no")
            print()
    finally:
        tvr.ROOT = saved


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
    ("Struct", "a public `[kernel]` field that is the specification (control)",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] The kernel meets its specification. -/
  spec : Claim.Meets Inst.ok Pre Post K
""", "fun _ _ h => h.spec", f"⟨{MEETS}⟩"),
    ("Priv", "a `private` `[kernel]` field that is the specification (lr-20c R1)",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] The kernel meets its specification. -/
  private spec : Claim.Meets Inst.ok Pre Post K
""", "fun _ _ h => h.spec", f"⟨{MEETS}⟩"),
    ("Panic", "the word PANIC in a field's docstring (lr-20c R3)",
     """structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [kernel] The kernel meets its specification; no PANIC here. -/
  spec : Claim.Meets Inst.ok Pre Post K
""", "fun _ _ h => h.spec", f"⟨{MEETS}⟩"),
]


def lean_path():
    pre = subprocess.run([tvr.LAKE, "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True,
                         text=True).stdout.strip()
    return OLEAN + os.pathsep + pre


def part_r():
    print("# Part R (L112): a private field is refused by name; PANIC is read off the result line only\n")
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
        g = os.path.join(LDIR, f"reader{name}.lean")
        open(g, "w").write(tvr.claim_reader_text(name, "Kernels.Demo", set()))
        r2 = subprocess.run([LEAN, g], cwd=LEAN_DIR, capture_output=True, text=True, env=env)
        js = [l for l in r2.stdout.splitlines() if l.startswith("TVR_JSON ")]
        errs = [l.split(": error: ", 1)[1] for l in r2.stdout.splitlines() if ": error: " in l]
        show("the reader", f"exit {r2.returncode}, {len(js)} result line(s)"
             + (f"; error: {errs[0][:170]}" if errs else ""))
        surf, why = tvr.reader_result(r2.returncode, r2.stdout, r2.stderr)
        show("the gate's shape step (`tvr.reader_result`)", ("PASS" if surf else "FAIL") + (f": {why[:170]}" if not surf else ""))
        if surf:
            consts = tvr.const_table(surf)
            show("the claim's fields", surf["fields"])
            show("kinds", tvr.kinds_problems(surf, consts, "Kernels.Demo") or "PASS")
            unused = [k for k, u in surf["usage"].items() if not u]
            show("usage", "unused: " + ", ".join(unused) if unused else "PASS")
        print()
    print("## made-up reader outputs (the shape decision alone)\n")
    ok = 'TVR_JSON {"fields": ["spec"], "note": "a docstring that says PANIC"}'
    for label, code, out, err in (
            ("a result line that holds the word PANIC, nothing else", 0, ok + "\n", ""),
            ("the same, and a panic of Lean's on stderr", 0, ok + "\n", "PANIC at Lean.getStructureInfo Lean.Structure:1:2: structure expected\n"),
            ("the same, and a panic of Lean's on stdout before the result line", 0,
             "PANIC at Lean.getStructureInfo Lean.Structure:1:2: structure expected\n" + ok + "\n", "")):
        surf, why = tvr.reader_result(code, out, err)
        show(label, ("PASS" if surf else "FAIL") + (f": {why[:120]}" if not surf else ""))
    print()


def main():
    tools = {p: hashlib.sha256(open(os.path.join(TVA, p), "rb").read()).hexdigest()[:16]
             for p in ("tools/tvr.py", "tools/claim_reader.lean.in")}
    print(f"tool files: {tools}\n")
    if "l" in PARTS:
        part_l()
    if "r" in PARTS:
        part_r()


main()
