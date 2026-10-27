"""lr-2a, round 2, L7: path forms that lead out of TVA, given to the tool's own `foreign_refs`.

Run from the TVA root:  python3 -I docs/library-review/round-2/lr-2a/work/l7_path_forms.py

Every path below leads to a sibling of TVA under the same home directory (a made-up project folder
`otherproj`, which is not a hashed name), so the protocol (§2, "Text") asks the audit to flag each.
The paths are joined at run time, so that this file holds none of them. The output prints only the
probe's label and whether it was flagged.
"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    ROOT = os.path.dirname(ROOT)
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

S = "/"
HOME_ABS = S + "home" + S + "someone"
SIB = "fpv" + S + "otherproj" + S + "evidence" + S + "card.md"

PROBES = [
    ("control: the home form", "~" + S + SIB, True),
    ("control: the absolute form", HOME_ABS + S + SIB, True),
    ("$HOME form", "$HOME" + S + SIB, True),
    ("${HOME} form", "${HOME}" + S + SIB, True),
    ("file URL, three slashes", "file:" + S + S + HOME_ABS + S + SIB, True),
    ("path after a Markdown link bracket", "[card](" + HOME_ABS + S + SIB + ")", True),
    ("path inside a shell assignment", "P=" + HOME_ABS + S + SIB, True),
    ("path glued to a semicolon", "cd x;" + HOME_ABS + S + SIB, True),
]

where = os.path.join(ROOT, "docs")
for label, text, want in PROBES:
    got = bool(tvr.foreign_refs("see " + text + " for the card", where))
    print(f"{label}: {'flagged' if got else 'NOT flagged'} ({'as the rule asks' if got == want else 'MISSED'})")
