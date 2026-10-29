"""lr-20f, round 20, L95: forms of the facts section beyond evidence/tool-round20 Part L, through
tvr.facts_section / stated_facts on scratch copies of docs/rules/num.md. Run from the TVA root:
python3 -I <this file> <scratch dir>"""
import os, sys, re
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import tvr

scratch = sys.argv[1]
page = open(os.path.join(os.getcwd(), "docs/rules/num.md"), encoding="utf-8").read()
lines = page.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("## Facts the library states"))
tab = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
FAKE_PIPELESS = ["Fact | Operation", "--- | ---", "`MathAcc` | anything", ""]
FAKE_LEADING_PIPE_DELIM = ["Fact | Operation", "|---|---|", "`MathAcc` | anything", ""]
CASES = {
    "control (the tree's page)": lambda L: L,
    "a pipe-less table before the real one": lambda L: L[:tab] + FAKE_PIPELESS + L[tab:],
    "a table whose header has no leading pipe": lambda L: L[:tab] + FAKE_LEADING_PIPE_DELIM + L[tab:],
    "<details> around the real table": lambda L: L[:tab] + ["<details>", ""] + L[tab:],
    "a text line with inline HTML before the table": lambda L: L[:tab] + ["Note <b>old</b> list:", ""] + L[tab:],
    "the real table's rows indented by 1 space": lambda L: L[:tab] + [" " + l if l.startswith("|") else l for l in L[tab:tab + 50]] + L[tab + 50:],
    "a '#Heading' (no space) before the table": lambda L: L[:tab] + ["#Old facts", ""] + L[tab:],
    "a tab-indented line": lambda L: L[:tab] + ["\tcode", ""] + L[tab:],
}
for name, f in CASES.items():
    d = os.path.join(scratch, re.sub(r"\W+", "_", name))
    os.makedirs(os.path.join(d, "docs/rules"), exist_ok=True)
    open(os.path.join(d, "docs/rules/num.md"), "w", encoding="utf-8").write("\n".join(f(list(lines))))
    sec = tvr.facts_section(d)
    facts = tvr.stated_facts(d)
    print(f"{name}: problems {len(sec[1]) if sec else 'no section'}"
          f"{(' (' + sec[1][0][:90] + ')') if sec and sec[1] else ''}; facts the gate accepts: "
          f"{len(facts) if facts else 0}; MathAcc citable: {bool(facts) and 'TVA.Rules.Num.MathAcc' in facts}")
