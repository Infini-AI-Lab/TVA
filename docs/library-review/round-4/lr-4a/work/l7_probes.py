"""lr-4a: spellings of a path into a sibling folder of TVA, against `foreign_refs` of tools/tvr.py.

Run from the TVA root:  python3 -I docs/library-review/round-4/lr-4a/work/l7_probes.py
The sibling's name is a made-up one (`otherproj`), and every path is joined at run time, so this
file holds none of them literally. Each probe prints whether the scanner flags it.
"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("tvr", os.path.join("tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
sys.argv = ["tvr"]
spec.loader.exec_module(tvr)

F = "f" + "pv"
S = "other" + "proj"
H = "~"
Q, A = '"', "'"
probes = [
    ("control: sibling, plain", f"cat {H}/{F}/{S}/README.md", True),
    ("control: TVA itself", f"cat {H}/{F}/tva/README.md", False),
    ("double-quoted component", f"cat {H}/{Q}{F}{Q}/{S}/README.md", True),
    ("single-quoted component", f"cat {H}/{A}{F}{A}/{S}/README.md", True),
    ("empty quotes inside a name", f"cat {H}/{F}{Q}{Q}/{S}/README.md", True),
    ("whole path quoted after ~/", f"cat {H}/{Q}{F}/{S}/README.md{Q}", True),
    ("brace expansion", f"cat {H}/{F}/{{tva,{S}}}/README.md", True),
    ("glob over siblings", f"cat {H}/{F}/*/README.md", True),
    ("bracket glob", f"cat {H}/{F}/[o]{S[1:]}/README.md", True),
    ("cd to the parent, then relative", f"cd {H}/{F} && cat {S}/README.md", True),
    ("backslash-escaped letter (shell)", f"cat {H}/{F}/\\{S}/README.md", True),
    ("variable for the parent", f"P={H}/{F}; cat $P/{S}/README.md", True),
]
wrong = 0
for name, text, should in probes:
    got = bool([r for r in tvr.foreign_refs(text) if "path" in r])
    mark = "flagged" if got else "passed "
    ok = got == should
    wrong += not ok
    print(f"{mark}  {'as the rule asks' if ok else 'NOT as a reader resolves it'}  {name}")
print(f"{len(probes)} probes, {wrong} not flagged where a shell resolves the path into a sibling")
