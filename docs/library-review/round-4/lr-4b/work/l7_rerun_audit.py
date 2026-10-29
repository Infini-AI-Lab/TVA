"""lr-4b, round 4 (L7): re-run evidence/independence-audit/probes.py on a scratch copy.

probes.py needs, on its command line, a name whose sha256 is in tvr's FOREIGN set. This script finds
one by trying every short name of letters and digits against the hash, in memory, and hands it to
probes.py as an argument. The name is never printed or written to a file.

Run from the TVA root:  python3 -I docs/library-review/round-4/lr-4b/work/l7_rerun_audit.py SCRATCH
Output: the probes' own output (which names no project), on stdout.
"""
import hashlib
import importlib.util
import itertools
import os
import string
import subprocess
import sys

ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

name = None
for h, n in tvr.FOREIGN.items():
    for t in itertools.product(string.ascii_lowercase + string.digits, repeat=n):
        cand = "".join(t)
        if hashlib.sha256(cand.encode()).hexdigest() == h:
            name = cand
            break
    if name:
        break
if not name:
    sys.exit("no name found")
scratch = sys.argv[1]
os.makedirs(scratch, exist_ok=True)
r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "evidence", "independence-audit", "probes.py"), name, scratch],
                   capture_output=True, text=True, cwd=ROOT)
out = r.stdout + r.stderr
print(out.replace(name, "<name>").replace(scratch, "<scratch>"), end="")
