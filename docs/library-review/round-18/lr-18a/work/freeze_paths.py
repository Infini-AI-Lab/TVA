"""lr-18a, round 18 (library issue L81): a hand-written `freeze` line with a path the tool never writes.

`tvr freeze` writes normalised paths. `freeze_problem` (the release check's test of a freeze line) takes a work file
to be any path that starts with the folder and has a `/` after it, so `<folder>/work/../filing-2.json` passes as a
work file. Does such a line let a filing at the top of the folder count as recorded? A scratch copy of the tree
(tools/, docs/, kernels/vec_add_f32; evidence/tool-round18/run.py's `scratch_root`) has vec-add's ledger pinned at
its 14 events before the coordinator's freeze (the four round-1 work files kept); a `filing-2.json` and `report-2.md`
are put at the top of `round-1/r-vecadd`, and the freeze line is written by hand.

Usage: python3 -I freeze_paths.py SCRATCH > freeze_paths.out
"""
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
scratch = sys.argv[1]
sys.argv = [os.path.join(TVA, "evidence", "tool-round18", "run.py"), scratch]
spec = importlib.util.spec_from_file_location("run18", sys.argv[0])
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
tvr = run.tvr
VEC = run.VEC


def pin_ledger(n):  # rerun18.py's: a file a later freeze records stays
    evs = run.read_ledger()
    kept = {p for e in evs[:n] for p in e.get("files", {})}
    frozen = {p for e in evs[n:] if e.get("type") == "freeze" for p in e.get("files", {})}
    for p in sorted({p for e in evs[n:] for p in e.get("files", {})} - kept - frozen):
        if os.path.isfile(os.path.join(tvr.ROOT, p)):
            os.remove(os.path.join(tvr.ROOT, p))
    run.write_ledger(evs[:n])


root = run.scratch_root("fp")
run.use_root(root)
pin_ledger(14)
w1 = f"kernels/{VEC}/review/round-1/r-vecadd"
for name, body in (("filing-2.json", json.dumps({"reviewer": "r-vecadd", "findings": [
        {"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "s", "detail": "d"}]})), ("report-2.md", "# r\n")):
    open(os.path.join(root, w1, name), "w").write(body)
evs = run.read_ledger()
work = {f"{w1}/work/{n}": tvr.file_sha(os.path.join(root, w1, "work", n))
        for n in ("Cover.lean", "Print.lean", "Spec.lean", "offsets.py")}
odd = {f"{w1}/work/../{n}": tvr.file_sha(os.path.join(root, w1, n)) for n in ("filing-2.json", "report-2.md")}
run.write_ledger([dict(e) for e in evs] + [{"type": "freeze", "actor": "coordinator", "round": 2, "time": "t", "dir": w1,
                                            "files": {**work, **odd}}])
print(f"freeze_paths.py (lr-18a): tools/tvr.py sha256 {tvr.file_sha(os.path.join(TVA, 'tools', 'tvr.py'))[:16]}\n")
run.show("the hand-written freeze records", sorted(work) + sorted(odd))
run.show("freeze_problem on it (is it one `tvr freeze` writes?)", tvr.freeze_problem(VEC, evs, run.read_ledger()[-1]) or "none")
run.show("kernel_event_problems", tvr.kernel_event_problems(root, VEC) or "none")
run.show("ledger_problems", tvr.ledger_problems(root, f"kernels/{VEC}/review/ledger.jsonl", "the ledger", folders=False) or "none")
run.show("kernel_folder_problems", tvr.kernel_folder_problems(root, VEC) or "none")
run.use_root(None)
