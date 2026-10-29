"""Library review round 3 (lr-3b): how tools/tvr.py splits docs/TRANSCRIPTION.md into rules and
general text (library issue L12). Reads the real tree, writes nothing.
Run from the TVA root: python3 -I docs/library-review/round-3/lr-3b/work/rule_parse.py"""
import importlib.util
import os
import sys

ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

gtext, rules, refs, problems = tvr.transcription_rules()
print(f"rules: {len(rules)}; problems: {problems}")
print("references per rule:")
for i in sorted(refs):
    if refs[i]:
        print(f"  [{i or '(general)'}] -> {sorted(refs[i])}")
# Where does the last paragraph of section 3 land?
probe = "No construct of [exclusions.md](exclusions.md) is used."
print()
print("last paragraph of section 3 in the general text:", probe in gtext)
print("last paragraph of section 3 in rule texts:", [i for i, t in rules.items() if probe in t])
print()
print("---- the text of rule [atomic] as the digest folds it:")
print(rules["atomic"])
print()
# Which section headings and paragraphs make up the general text of the page itself
page_general = gtext.split("\n\n# Exclusions")[0]
print("---- general text of the page (non-blank lines):")
for l in page_general.splitlines():
    if l.strip():
        print("  " + l[:110])
