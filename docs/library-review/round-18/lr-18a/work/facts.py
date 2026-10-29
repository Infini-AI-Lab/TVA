"""lr-18a, round 18 (library issue L95): what the gate reads as docs/rules/num.md's table of facts, against what the
rendered Markdown shows.

`facts_section_tables` reads the section line by line: a run of lines that start with `|` is a table, and a line that
starts with `#` is a heading. Markdown also has headings underlined with `---` or `===` (setext), HTML comments, which
the rendered page does not show, and fenced code blocks, which it shows as code, not as a table. Each case below is a
scratch copy of the tree's num.md with the tree's table first moved or hidden and a two-row table (`MathAcc`, a shape
the field would choose constants for, and `DivRn P .binary32`) in its place, or added; the tool's `stated_facts`,
`rule_fact_problem` and the audit's `fact_page_problems` are called on it, as evidence/tool-round18 Part L does.

Usage: python3 -I facts.py SCRATCH > facts.out
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"


def show(label, value):
    print(f"- {label}: {str(value).replace(SCRATCH, '<scratch>').replace(TVA, '<tva>')}")


text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
lines = text.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
first = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
end = next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))
table = lines[first:end]
shapes = ["| Shape (not citable) | Says |", "|---|---|", "| `MathAcc` | an accuracy shape |",
          "| `DivRn P .binary32` | IEEE division on float32 |"]
head, tail = lines[:first], lines[end:]
CASES = [
    ("control: the tree's page", lines),
    ("the shapes' table under a setext heading (`Shapes, not citable` underlined with `---`) right after the "
     "section's heading, in place of the facts' table, which moves to a later `##` section",
     head + ["Shapes, not citable", "-------------------", ""] + shapes + tail + ["", "## The facts", ""] + table),
    ("the shapes' table inside an HTML comment (the rendered page shows no table) in place of the facts' table",
     head + ["<!--"] + shapes + ["-->"] + tail),
    ("the shapes' table inside a fenced code block (the rendered page shows code, not a table) in place of the "
     "facts' table", head + ["```"] + shapes + ["```"] + tail),
    ("the shapes' table inside an HTML comment before the facts' table", head + ["<!--"] + shapes + ["-->", ""] + table + tail),
]
U_ACC = {"head": "TVA.Rules.Num.MathAcc", "platform_fact": False, "args": [{"platform": True}] + [{}] * 5}
U_D32 = {"head": "TVA.Rules.Num.DivRn", "platform_fact": False,
         "args": [{"platform": True}, {"type": "TVA.Math.Float.Format", "const": "TVA.Math.Float.Format.binary32"}]}
U_EXP = {"head": "TVA.Rules.Num.ExpF32", "platform_fact": True, "args": [{"platform": True}]}

root = os.path.join(SCRATCH, "root")
os.makedirs(os.path.join(root, "docs", "rules"))
page = os.path.join(root, "docs", "rules", "num.md")
real_root = tvr.ROOT
print(f"facts.py (lr-18a): tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}\n")
for label, ls in CASES:
    open(page, "w", encoding="utf-8").write("\n".join(ls))
    tvr.ROOT = root
    f = tvr.stated_facts()
    show(label + ": facts listed", len(f) if f else f"none ({f})")
    show("  the section's tables (facts_section_tables)", len(tvr.facts_section_tables() or []))
    for what, u in (("`MathAcc P …`", U_ACC), ("`DivRn P .binary32`", U_D32), ("`ExpF32 P`", U_EXP)):
        show(f"  a field citing {what}", tvr.rule_fact_problem(u, "Kernels.Demo", f) or "PASS")
    show("  the audit's fact-page check about num.md", [x for x in tvr.fact_page_problems() if "num.md" in x] or "nothing")
    tvr.ROOT = real_root
    print()
