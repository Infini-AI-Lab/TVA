"""lr-4b (library review round 4, L12/L27): how tools/tvr.py's `transcription_rules` splits
docs/TRANSCRIPTION.md, and where that may differ from what the page shows (CommonMark list-item
rules). Read-only: imports the tool from the TVA root and prints. Run from the TVA root:
  python3 -I docs/library-review/round-4/lr-4b/work/gate_rule_split.py
"""
import importlib.util, os, re, sys
ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tvr)
gtext, rules, refs, problems = tvr.transcription_rules()
print("problems:", problems)
print("rules:", len(rules))
for i, t in rules.items():
    lines = t.splitlines()
    print(f"[{i}] {len(lines)} lines; names {sorted(refs[i])}")
page = open(tvr.TRANSCRIPTION).read().splitlines()
# the general text lines of the page itself
gen_page = gtext.split("\n\n" + open(tvr.EXCLUSIONS).read())[0].splitlines()
print("\n== general text of the page (non-blank lines, first 80 chars):")
for l in gen_page:
    if l.strip():
        print("  |", l[:80])
print("\n== suspicious lines inside rules: indented by 1 space or a tab, or a code fence")
for i, t in rules.items():
    for l in t.splitlines():
        if (l.startswith(" ") and not l.startswith("  ")) or l.startswith("\t") or l.strip().startswith("```"):
            print(f"  [{i}] {l[:80]!r}")
print("\n== general text names these rules (closure added to every map):", sorted(refs[""]))
print("\n== general text includes exclusions.md:", open(tvr.EXCLUSIONS).read()[:40] in gtext,
      "; SEMANTICS.md:", open(tvr.SEMANTICS).read()[-60:] in gtext)
