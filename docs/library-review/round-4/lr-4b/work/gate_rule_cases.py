"""lr-4b, library review round 4 (L12): tools/tvr.py's rule split and page check on variants of
docs/TRANSCRIPTION.md written to a temporary file (the tool's page path is pointed at it in this
process only). Each case: does the split agree with how CommonMark shows the page, and does the
check report a disagreement? Run from the TVA root:
  python3 -I docs/library-review/round-4/lr-4b/work/gate_rule_cases.py
"""
import importlib.util, os, tempfile
spec = importlib.util.spec_from_file_location("tvr", os.path.join(os.getcwd(), "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tvr)
page = open(tvr.TRANSCRIPTION).read()
ANCHOR = "- **[broadcast]** `[:, None]`, `[None, :]` and scalar–tensor operations keep their axes.\n"
assert ANCHOR in page
CASES = [
    # (name, text inserted right after the [broadcast] line, CommonMark: is the extra text part of [broadcast]?)
    ("lazy continuation (unindented, no blank line)", "Lazy text here.\n", True),
    ("blank line, then unindented paragraph", "\nGeneral text here.\n", False),
    ("blank line, then 2-space indented paragraph", "\n  Rule text here.\n", True),
    ("blank line, then 1-space indented paragraph", "\n Para at one space.\n", False),
    ("ordered item `2.` right after (cannot interrupt a paragraph)", "2. Lazy item text.\n", True),
    ("bullet `- ` right after", "- A plain list item.\n", False),
]
for name, extra, cm_in_rule in CASES:
    text = page.replace(ANCHOR, ANCHOR + extra)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(text)
    tvr.TRANSCRIPTION = f.name
    g, rules, refs, problems = tvr.transcription_rules()
    os.unlink(f.name)
    word = extra.strip().split()[-1]
    in_rule = word in rules["broadcast"]
    flagged = any("broadcast" in p for p in problems)
    agree = in_rule == cm_in_rule
    print(f"{name:62} tool: {'rule' if in_rule else 'general'}; CommonMark: {'rule' if cm_in_rule else 'general'}; "
          f"{'agree' if agree else ('DISAGREE, reported' if flagged else 'DISAGREE, not reported')}")
