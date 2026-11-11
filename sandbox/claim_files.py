#!/usr/bin/env python3
"""Copy what a prover may see of a fixed claim (claims/<kernel>/) into a task or an image.

  claim_files.py SRC DST

copies only the claim's own files, an allow list: `Claim.lean`, `claim.json`, `README.md`, `SETUP.md` and
`source/`. Nothing else in the folder reaches a prover: not the GPU check (`check_gpu.py`; the containers have no
GPU), and not the claim's oracle proof (`oracle/`), which make_task.py puts only in a task's `solution/`. claim.json
gets Claim.lean's sha256 (`claim_sha256`), which the checker pins. The copy is checked afterwards: a file outside
the list, a Lean file other than `Claim.lean`, or a file with the content of any claim's oracle file stops it.
make_task.py imports `copy_claim`; claim/build.sh runs this file for the worked example. Standard library only.
"""
import hashlib
import json
import os
import shutil
import sys

FILES = ("Claim.lean", "claim.json", "README.md", "SETUP.md")
SOURCE = "source"


def oracle_hashes(claims):
    """The sha256 of every file of every claim's oracle proof (claims/*/oracle/**)."""
    hs = set()
    for k in os.listdir(claims):
        for root, _, files in os.walk(os.path.join(claims, k, "oracle")):
            for f in files:
                hs.add(sha256(os.path.join(root, f)))
    return hs


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def copy_claim(src, dst):
    """Copy the allow list of claim folder `src` into `dst` (a new folder); return claim.json, with `claim_sha256`."""
    for f in FILES + (SOURCE,):
        if not os.path.exists(os.path.join(src, f)):
            sys.exit(f"{src}: no {f}")
    os.makedirs(dst)
    for f in FILES:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    shutil.copytree(os.path.join(src, SOURCE), os.path.join(dst, SOURCE),
                    ignore=shutil.ignore_patterns("__pycache__", "*.lean", "oracle", "Proof*"))
    proofs = oracle_hashes(os.path.dirname(os.path.abspath(src)))
    for root, _, files in os.walk(dst):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), dst)
            ok = rel in FILES or (rel.startswith(SOURCE + os.sep) and not rel.endswith(".lean"))
            if not ok:
                sys.exit(f"claim_files.py: {rel} is not a claim file: refusing to copy {src}")
            if sha256(os.path.join(root, f)) in proofs:
                sys.exit(f"claim_files.py: {rel} is a file of an oracle proof: refusing to copy {src}")
    p = os.path.join(dst, "claim.json")
    cj = json.load(open(p))
    if cj.get("lean") != "Claim.lean":
        sys.exit(f"{src}/claim.json: `lean` must be Claim.lean")
    cj["claim_sha256"] = sha256(os.path.join(dst, "Claim.lean"))
    with open(p, "w") as fh:
        fh.write(json.dumps(cj, indent=2) + "\n")
    return cj


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    copy_claim(sys.argv[1], sys.argv[2])
