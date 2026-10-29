"""lr-7a: once my release check (work/release.out) has passed its build step, compute the digest of the
packages' build outputs in its sandbox with evidence/release-check/run.py's own `build_digest`, before
the check removes the sandbox, to compare with the author's R0 (TVA at its own path).
Usage: python3 -I work/build_digest_watch.py SANDBOX   (the --dir of the release check; never printed)
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True  # import the tree's modules without writing caches into it
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
sys.path.insert(0, os.path.join(TVA, "tools"))
spec = importlib.util.spec_from_file_location("release_ev", os.path.join(TVA, "evidence", "release-check", "run.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
d = sys.argv[1]
log = os.path.join(HERE, "release.out")
while True:
    text = open(log).read()
    if "  build: " in text or "release check" in text:
        break
    time.sleep(10)
if "PASS  build: " not in text:
    print("the build step did not pass; no digest")
    sys.exit(0)
try:
    t0 = time.time()
    print(f"build outputs of the packages (count, digest): {ev.build_digest(d)} ({time.time() - t0:.0f} s)")
except OSError as e:
    print(f"the sandbox was removed before the digest finished ({type(e).__name__})")
