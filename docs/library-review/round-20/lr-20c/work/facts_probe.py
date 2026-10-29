"""lr-20c, round 20 (L95): pages on which what tvr reads as the facts table differs from what a Markdown
renderer (CommonMark / GitHub) shows, beyond the forms round 20's `facts_section` refuses.

Each case writes a scratch copy of docs/rules/num.md with one change and reports what the gate's functions
(`facts_section`, `stated_facts`, `rule_fact_problem`, `fact_page_problems`) make of it. The tree is never written.

Usage: python3 -I facts_probe.py SCRATCH   (SCRATCH: an empty folder outside TVA; its path is never printed)
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = HERE
for _ in range(5):
    TVA = os.path.dirname(TVA)
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH)

text = open(os.path.join(TVA, "docs", "rules", "num.md"), encoding="utf-8").read()
lines = text.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
first = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
end = next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))
nxt = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
rows = lines[first:end]
fake = ["| Fact (`TVA.Rules.Num`) | Says |", "|---|---|", "| `DivRn P .binary32` | (a non-fact) |"]
sec_fake = ["## Facts the library states", "", *fake, ""]

cases = [
    ("control: the tree's page", lines),
    # the section (its heading, its table, the next ## heading) inside a fence that tvr's fence tracker takes as
    # closed before the heading, so the page shows the section as code
    ("a ```` (4 backticks) fence, a ``` line inside it (CommonMark: no closing fence, too short), the section, "
     "the next ## heading, then ```` closing it",
     lines[:start] + ["````", "```"] + lines[start:nxt + 1] + ["````"] + lines[nxt + 1:]),
    ("a ``` fence, a ```python line inside it (CommonMark: a closing fence has no info string), the section, "
     "the next ## heading, then ``` closing it",
     lines[:start] + ["```", "```python"] + lines[start:nxt + 1] + ["```"] + lines[nxt + 1:]),
    ("a line ```a`b (CommonMark: no fence, a backtick fence's info string holds no backtick), then ``` opening a "
     "fence, the section, the next ## heading, ``` closing it",
     lines[:start] + ["```a`b", "```"] + lines[start:nxt + 1] + ["```"] + lines[nxt + 1:]),
    # an HTML block of CommonMark's type 1 runs to its end tag across blank lines and headings
    ("<script> before the heading, </script> after the next ## heading (the page shows neither heading nor table)",
     lines[:start] + ["<script>", ""] + lines[start:nxt + 1] + ["</script>"] + lines[nxt + 1:]),
    ("<pre> before the heading, </pre> after the next ## heading (the page shows them as preformatted text)",
     lines[:start] + ["<pre>", ""] + lines[start:nxt + 1] + ["</pre>"] + lines[nxt + 1:]),
    # cell splitting
    ("a row starting `||`: GFM reads an empty first cell, tvr strips every leading `|`",
     lines[:end] + ["|| `DivRn P .binary32` | x | x | x | x |"] + lines[end:]),
    ("inline HTML: a text line opening <details> before the table, and a cell with <sub> (protocol §2: HTML refused)",
     lines[:first] + ["Older facts, not citable: <details><summary>show</summary>", ""] + lines[first:end]
     + ["| <sub>`DivRn P .binary32`</sub> | x | x | x | x |"] + lines[end:]),
    ("a struck-through first cell (GFM shows it struck out)",
     lines[:end] + ["| ~~`DivRn P .binary32`~~ (withdrawn) | x | x | x | x |"] + lines[end:]),
]

root = os.path.join(SCRATCH, "root")
os.makedirs(os.path.join(root, "docs", "rules"))
page = os.path.join(root, "docs", "rules", "num.md")
u_d32 = {"head": "TVA.Rules.Num.DivRn", "platform_fact": False,
         "args": [{"platform": True}, {"type": "TVA.Math.Float.Format", "const": "TVA.Math.Float.Format.binary32"}]}
saved = tvr.ROOT
tvr.ROOT = root
try:
    for label, ls in cases:
        open(page, "w", encoding="utf-8").write("\n".join(ls))
        got = tvr.facts_section()
        f = tvr.stated_facts()
        print(f"## {label}")
        print(f"- facts listed: {len(f) if f else f}")
        print(f"- the section's problems: {(got[1][:2] if got else 'no section') if (got is None or got[1]) else 'none'}")
        print(f"- a [rule] field citing `DivRn P .binary32`: {tvr.rule_fact_problem(u_d32, 'Kernels.Demo', f) or 'PASS'}")
        print(f"- the audit names num.md: {len([x for x in tvr.fact_page_problems() if 'num.md' in x]) or 'no'}")
        print()
finally:
    tvr.ROOT = saved
