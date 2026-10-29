"""lr-18b, L95: what the gate reads as docs/rules/num.md's facts table, against what a reader of the rendered page
sees. Each case copies the tree's num.md into a scratch root, changes the section "Facts the library states", and
calls the tool's `facts_section_tables`, `stated_facts` and (with the tool's root pointed at the scratch root) the
audit's `fact_page_problems`.

- S1: a setext heading (a line of text, then a line of `-`), which Markdown renders as a `##` heading, put between
  the section's heading and its table: a reader sees the table under "Shapes the facts use (not citable)".
- S2: S1 with a table of `MathAcc` and `DivRn P .binary32` under the setext heading, and the real table after a
  `##` heading of its own (so the gate's section holds only the shapes' table).
- C1: the table inside an HTML comment (`<!--` … `-->`), which the rendered page does not show.
- B1: the table inside a fenced code block (```), which the page shows as code, not as a table.
- H1: a fenced code block holding a `# comment` line before the table (a heading to the tool, not to Markdown).

Run: `python3 -I facts_probe.py SCRATCH > facts_probe.out` (SCRATCH: an empty folder outside TVA, never printed).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = HERE
while not os.path.isfile(os.path.join(TVA, "tools", "tvr.py")):
    TVA = os.path.dirname(TVA)
sys.path.insert(0, os.path.join(TVA, "tools"))
sys.dont_write_bytecode = True
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
lines = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read().split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
first = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
end = next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))
table = lines[first:end]
shapes = ["| Shape | Says |", "|---|---|", "| `MathAcc` | an accuracy shape |", "| `DivRn P .binary32` | IEEE division |"]
setext = ["Shapes the facts use (not citable)", "----------------------------------", ""]
cases = {
    "S1": lines[:first] + setext + lines[first:],
    "S2": lines[:first] + setext + shapes + ["", "## The facts (citable)", ""] + table + lines[end:],
    "C1": lines[:first] + ["<!--"] + table + ["-->"] + lines[end:],
    "B1": lines[:first] + ["```"] + table + ["```"] + lines[end:],
    "H1": lines[:first] + ["```", "# a comment in a code block", "```", ""] + lines[first:],
}
u_acc = {"head": "TVA.Rules.Num.MathAcc", "platform_fact": False, "args": [{"platform": True}] + [{}] * 5}
u_exp = {"head": "TVA.Rules.Num.ExpF32", "platform_fact": True, "args": [{"platform": True}]}
print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}\n")
print(f"- the tree's num.md: {len(tvr.stated_facts() or {})} facts, {len(tvr.facts_section_tables() or [])} table(s)")
real = tvr.ROOT
for name, ls in cases.items():
    root = os.path.join(SCRATCH, name)
    os.makedirs(os.path.join(root, "docs", "rules"))
    for p in ("compiler-faithful.md",):
        open(os.path.join(root, "docs", "rules", p), "w").write("x")
    open(os.path.join(root, "docs", "rules", "num.md"), "w", encoding="utf-8").write("\n".join(ls))
    facts = tvr.stated_facts(root)
    tables = tvr.facts_section_tables(root) or []
    tvr.ROOT = root
    audit = [x for x in tvr.fact_page_problems() if "num.md" in x]
    tvr.ROOT = real
    print(f"- {name}: facts listed {len(facts) if facts else 0}, tables {len(tables)}; "
          f"`MathAcc P …`: {tvr.rule_fact_problem(u_acc, 'K', facts) or 'PASS'}; "
          f"`ExpF32 P`: {tvr.rule_fact_problem(u_exp, 'K', facts) or 'PASS'}; the audit about num.md: {audit or 'nothing'}")
