#!/usr/bin/env python3
"""The content hash of TVA's library source, which tags the sandbox's images.

  lean_hash.py LEAN_DIR      prints 12 hex digits

It covers lakefile.toml, lake-manifest.json, lean-toolchain, TVA.lean and every TVA/**.lean (path and bytes),
nothing under .lake. Standard library only.
"""
import hashlib
import os
import sys


def lean_hash(lean):
    files = ["lakefile.toml", "lake-manifest.json", "lean-toolchain", "TVA.lean"]
    for dirpath, dirnames, names in os.walk(os.path.join(lean, "TVA")):
        dirnames.sort()
        files += [os.path.relpath(os.path.join(dirpath, n), lean) for n in names if n.endswith(".lean")]
    h = hashlib.sha256()
    for rel in sorted(files):
        data = open(os.path.join(lean, rel), "rb").read()
        h.update(f"{rel}\0{len(data)}\0".encode() + data)
    return h.hexdigest()[:12]


if __name__ == "__main__":
    print(lean_hash(sys.argv[1]))
