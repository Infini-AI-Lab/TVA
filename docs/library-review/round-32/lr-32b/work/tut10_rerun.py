"""lr-32b, round 32, L153: evidence/tut10-formats/run.py at 1024^3 only (each format), not 8192^3.
Run from TVA's root: CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I docs/library-review/round-32/lr-32b/work/tut10_rerun.py
"""
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "..", "..", "..", "..", "evidence", "tut10-formats", "run.py")
spec = importlib.util.spec_from_file_location("tut10run", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t = m.load()
for fmt in ("mxfp4", "nvfp4", "mxfp8", "mixed"):
    print(m.one(t, fmt, 1024, 1024, 1024), flush=True)
