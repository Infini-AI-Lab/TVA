#!/usr/bin/env python3
"""Write the oracle proof of a fixed claim, claims/<kernel>/oracle/, where a task's solution or the worked example
needs it.

  oracle.py KERNEL OUT_DIR

writes OUT_DIR/Proof.lean (and OUT_DIR/Proof/**.lean) from claims/KERNEL/oracle/. Every fixed claim keeps its
complete proof there, passing `tva-check grade`. A prover never sees one: make_task.py copies only the claim's own
files into a task's images and puts the proof only in the task's `solution/` (--oracle), and the claim image holds
one, the worked example vec_add_f32, which it shows at /app/examples/vec_add_f32 and checks at build (`tva-check
check`). It refuses when OUT_DIR's claim.json names a namespace that no file of the proof opens, so a changed claim
fails here with a clear message rather than in the build. Standard library only.
"""
import glob
import json
import os
import re
import shutil
import sys

CLAIMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "claims")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    k, out = sys.argv[1], sys.argv[2]
    src = os.path.join(CLAIMS, k, "oracle")
    if not os.path.isfile(os.path.join(src, "Proof.lean")):
        sys.exit(f"oracle.py: no proof of {k} (claims/{k}/oracle/Proof.lean)")
    cj_path = os.path.join(out, "claim.json")
    if os.path.isfile(cj_path):
        ns = json.load(open(cj_path))["namespace"]
        files = [os.path.join(src, "Proof.lean")] + glob.glob(os.path.join(src, "Proof", "**", "*.lean"), recursive=True)
        if not any(re.search(rf"^namespace {re.escape(ns)}$", open(f).read(), re.M) for f in files):
            sys.exit(f"oracle.py: the proof of {k} is not in the claim's namespace {ns}")
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(src, "Proof.lean"), os.path.join(out, "Proof.lean"))
    if os.path.isdir(os.path.join(src, "Proof")):
        shutil.copytree(os.path.join(src, "Proof"), os.path.join(out, "Proof"), dirs_exist_ok=True)
    print(f"{k}: proof written to {out}")


if __name__ == "__main__":
    main()
