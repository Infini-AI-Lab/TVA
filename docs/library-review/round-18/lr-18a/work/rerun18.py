"""Re-run evidence/tool-round18/run.py's Parts R and P on the tree as it stands after the coordinator's
`tvr freeze vec_add_f32` (event 15 of vec-add's ledger), lr-18a, round 18.

run.py's `pin_ledger(n)` keeps the first `n` events and removes from the scratch copy every file that only a later
event records. The four round-1 work files of r-vecadd are now recorded by event 15, the freeze, so `pin_ledger(14)`
removes them, although they lay in the folder long before (the freeze records files already there): Part R then no
longer starts from the ledger "as shipped before the coordinator's freeze", and its hand-written freeze lines fail
with FileNotFoundError (rerun18.out, first run). This wrapper loads run.py unchanged and replaces `pin_ledger` by
one that also keeps the files a later `freeze` records, then runs the given parts (second run).

Usage: python3 -I rerun18.py SCRATCH [--plain] [PARTS]   (PARTS: any of r p; both by default)
  SCRATCH  an empty folder outside TVA; its path is never printed (run.py masks it).
  --plain  use run.py's own pin_ledger (to show the failure).
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
args = sys.argv[1:]
plain = "--plain" in args
args = [a for a in args if a != "--plain"]
scratch, parts = args[0], args[1:] or ["r", "p"]
sys.argv = [os.path.join(TVA, "evidence", "tool-round18", "run.py"), scratch]
spec = importlib.util.spec_from_file_location("run18", sys.argv[0])
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
tvr = run.tvr


def pin_ledger(n):
    """run.py's pin_ledger, but a file a later `freeze` records stays: it was in the folder before the freeze."""
    evs = run.read_ledger()
    kept = {p for e in evs[:n] for p in e.get("files", {})}
    frozen = {p for e in evs[n:] if e.get("type") == "freeze" for p in e.get("files", {})}
    for p in sorted({p for e in evs[n:] for p in e.get("files", {})} - kept - frozen):
        if os.path.isfile(os.path.join(tvr.ROOT, p)):
            os.remove(os.path.join(tvr.ROOT, p))
    for dirpath, _, _ in sorted(os.walk(tvr.kpath(run.VEC, "review")), reverse=True):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    run.write_ledger(evs[:n])


if not plain:
    run.pin_ledger = pin_ledger
print(f"rerun18: run.py's pin_ledger {'as shipped' if plain else 'keeping the files a later freeze records'}; "
      f"the tree's vec-add ledger has {len(tvr.ledger(run.VEC))} events\n")
for p, fn in (("r", run.part_r), ("p", run.part_p)):
    if p in parts:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - shown, not hidden
            print(f"- Part {p.upper()} stopped: {type(e).__name__}: {run.mask(e)}\n")
