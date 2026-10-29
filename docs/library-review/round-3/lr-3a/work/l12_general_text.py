"""lr-3a, round 3 (L12): does the tool's split of docs/TRANSCRIPTION.md into rules and general text
match what the page says ("The text outside the rules (this introduction, §1, the last paragraph of
§3, §9) and exclusions.md apply to every model")?

Run from the TVA root: python3 -I docs/library-review/round-3/lr-3a/work/l12_general_text.py

1. Lists every unindented paragraph (after a blank line, not a list item, table row or heading)
   that the tool's parser puts inside a rule.
2. On a copy of the page in a temporary folder, changes the last paragraph of §3 and reports which
   digest parts change for a map that names only [mask] (as a kernel without atomics might) and for
   one that names [atomic].
"""
import importlib.util
import os
import tempfile

ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

g, rules, refs, problems = t.transcription_rules()
print(f"rules: {len(rules)}, page problems: {problems}")
print("1. unindented paragraphs the parser puts inside a rule:")
for rid, text in rules.items():
    lines = text.split("\n")
    for i, l in enumerate(lines[1:], 1):
        if l and not l[0].isspace() and not l.startswith(("- ", "|", "#")) and lines[i - 1].strip() == "":
            print(f"   [{rid}]: {l[:90]!r}")

page = open(t.TRANSCRIPTION, encoding="utf-8").read()
old = "No construct of [exclusions.md](exclusions.md) is used."
assert old in page
before = {ids: t.rule_parts(list(ids)) for ids in (("mask",), ("atomic",))}
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "TRANSCRIPTION.md")
    open(p, "w", encoding="utf-8").write(page.replace(old, "Constructs of [exclusions.md](exclusions.md) may be used."))
    t.TRANSCRIPTION = p
    after = {ids: t.rule_parts(list(ids)) for ids in (("mask",), ("atomic",))}
print("2. the last paragraph of §3 changed ('No construct of exclusions.md is used' -> 'may be used'):")
for ids in before:
    changed = sorted(k for k in set(before[ids]) | set(after[ids]) if before[ids].get(k) != after[ids].get(k))
    print(f"   map naming {list(ids)}: digest parts that change: {changed or 'none'}")
