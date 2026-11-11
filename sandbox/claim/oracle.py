#!/usr/bin/env python3
"""Write the complete proof of a fixed claim (claims/<kernel>/) that the sandbox keeps.

  oracle.py KERNEL OUT_DIR

writes OUT_DIR/Proof.lean (and OUT_DIR/Proof/**.lean) from oracle/KERNEL/. The sandbox keeps one: the worked
example vec_add_f32, which the claim image shows at /app/examples/vec_add_f32 and checks at build
(`tva-check check`). It refuses when OUT_DIR's claim.json names another namespace than the proof's, so a changed
claim fails here with a clear message rather than in the build. Standard library only.
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    k, out = sys.argv[1], sys.argv[2]
    src = os.path.join(HERE, "oracle", k)
    if not os.path.isfile(os.path.join(src, "Proof.lean")):
        have = sorted(os.listdir(os.path.join(HERE, "oracle")))
        sys.exit(f"oracle.py: no proof of {k} (the sandbox has proofs of {', '.join(have)})")
    cj_path = os.path.join(out, "claim.json")
    if os.path.isfile(cj_path):
        ns = json.load(open(cj_path))["namespace"]
        if not re.search(rf"^namespace {re.escape(ns)}$", open(os.path.join(src, "Proof.lean")).read(), re.M):
            sys.exit(f"oracle.py: the proof of {k} is not in the claim's namespace {ns}")
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(src, "Proof.lean"), os.path.join(out, "Proof.lean"))
    if os.path.isdir(os.path.join(src, "Proof")):
        shutil.copytree(os.path.join(src, "Proof"), os.path.join(out, "Proof"), dirs_exist_ok=True)
    print(f"{k}: proof written to {out}")


if __name__ == "__main__":
    main()
