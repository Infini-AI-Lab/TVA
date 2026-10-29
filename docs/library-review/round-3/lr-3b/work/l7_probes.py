"""Library review round 3 (lr-3b), library issue L7: home paths to a sibling folder of TVA, in
spellings the round-3 tool may miss, tested against the tool's own `foreign_refs` (tools/tvr.py).
Each probe should be flagged ("a path outside TVA" or "a relative path that leaves TVA"); the
controls should not. The sibling folder is called `other`, so no project is named. Paths are joined
at run time, so this file holds no path out of TVA. Reads the real tree, writes nothing.
Run from the TVA root: python3 -I docs/library-review/round-3/lr-3b/work/l7_probes.py"""
import importlib.util
import os

ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

S = "/"
H = S + "home" + S + "someone"          # an absolute home directory
T = "~"                                   # the home directory
SIB = S + "fpv" + S + "other" + S + "out.txt"   # a sibling of TVA under the home directory
WHERE = os.path.join(ROOT, "docs", "library-review", "round-3", "lr-3b", "work")

PROBES = [
    # (id, text, should be flagged)
    ("Q1", "python3 run.py >" + H + SIB, True),            # shell redirection, no space
    ("Q2", "python3 run.py 2>" + H + SIB, True),
    ("Q3", "echo x >" + T + SIB, True),
    ("Q4", "echo x >>" + T + SIB, True),
    ("Q5", "cd $(dirname $0)" + (S + "..") * 6 + S + "other", True),  # six levels up from work/ leaves TVA
    ("Q13", "python3 run.py >" + ".." + (S + "..") * 5 + S + "other" + S + "out.txt", True),
    ("Q6", "see (the notes)" + H + SIB, True),             # after a closing bracket
    ("Q7", "x]" + T + SIB, True),
    ("Q8", "f\"{os.environ['HOME']}" + SIB + "\"", True),  # a Python f-string
    ("Q9", "f\"{Path.home()}" + SIB + "\"", True),
    ("Q10", "PATH=$PATH:" + T + SIB, True),                # control-like, should be flagged (after ':')
    ("Q11", "a-" + H + SIB, True),                          # after a hyphen
    ("Q12", "cat <" + H + SIB, True),                       # input redirection
    # controls: should not be flagged
    ("C1", "round-<N>" + S + "author" + S + "notes.md", False),
    ("C2", "kernels" + S + "{kernel}" + S + "review" + S + "x", False),
    ("C3", "python3 run.py >out.txt", False),
    ("C4", "echo x >" + T + S + "fpv" + S + "tva" + S + "x", False),
]

wrong = 0
for pid, text, want in PROBES:
    got = tvr.foreign_refs(text, WHERE)
    ok = bool(got) == want
    wrong += not ok
    print(f"{pid}: {'flagged' if got else 'passed'} ({'as the rule asks' if ok else 'NOT as the rule asks'})")
print(f"{len(PROBES)} probes, {wrong} not as the rule asks")
