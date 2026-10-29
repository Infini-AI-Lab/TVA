"""lr-18b, L81: hand-written `freeze` lines after vec-add's shipped ledger (15 events, the coordinator's freeze being
event 15), judged by the tool's own `freeze_problem` and `own_records` in memory (nothing is written), and the
events' accounting of each role folder. Can a freeze account for a filing, a report, a response or notes?

Run: `python3 -I freeze_probe.py > freeze_probe.out`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = HERE
while not os.path.isfile(os.path.join(TVA, "tools", "tvr.py")):
    TVA = os.path.dirname(TVA)
sys.path.insert(0, os.path.join(TVA, "tools"))
sys.dont_write_bytecode = True
import tvr  # noqa: E402

K = "vec_add_f32"
ev = tvr.ledger(K)
base = f"kernels/{K}/review"
print(f"tvr: tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}; vec-add's ledger: {len(ev)} events\n")
print(f"- event 15 as shipped: {tvr.freeze_problem(K, ev[:14], ev[14]) or 'what tvr freeze writes'}")
for f, o in sorted(tvr.own_records(K, ev).items()):
    print(f"- {f[len(base) + 1:]}: dir {o['dir']}, frozen {o['frozen']}, records {sorted(p[len(f) + 1:] for p in o['files'])}")
x = "0" * 64
cases = [
    ("round-1/r-vecadd again, a new work file", "coordinator", "round-1/r-vecadd", ["work/new.py"]),
    ("round-1/prover (event 5, no dir, not frozen): its notes.md", "coordinator", "round-1/prover", ["notes.md"]),
    ("round-1/prover: a response-2.json at its top", "coordinator", "round-1/prover", ["response-2.json"]),
    ("round-1/prover: a report under a subfolder", "coordinator", "round-1/prover", ["work/report-2.md"]),
    ("round-1/prover: a file of round-1/r-vecadd", "coordinator", "round-1/prover", ["../r-vecadd/filing-2.json"]),
    ("round-2/r-vecadd (event 11 names its dir): a work file", "coordinator", "round-2/r-vecadd", ["work/new.py"]),
    ("round-2/r-vecadd2: its filing-2.json", "coordinator", "round-2/r-vecadd2", ["filing-2.json"]),
    ("round-1/prover by r-vecadd", "r-vecadd", "round-1/prover", ["work/new.py"]),
    ("round-3/r-vecadd, which no event records", "coordinator", "round-3/r-vecadd", ["work/new.py"]),
]
for label, actor, d, files in cases:
    dd = f"{base}/{d}"
    e = {"type": "freeze", "actor": actor, "round": 2, "dir": dd,
         "files": {os.path.normpath(f"{dd}/{p}"): x for p in files}}
    print(f"- a hand-written freeze of {label}: {tvr.freeze_problem(K, ev, e) or 'accepted (records no event)'}")
