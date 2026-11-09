"""lr-34b: print the lines of atomics_unused.py's add/int32/relaxed PTX that contain `bar`."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomics_unused import k  # noqa: E402

G, B, C = 4, 256, 64
h = k[(G,)](torch.zeros(2 * C, dtype=torch.int32, device="cuda"), torch.ones(G * B, dtype=torch.int32, device="cuda"),
            torch.ones(G * B, dtype=torch.int32, device="cuda"), torch.zeros(G * B, dtype=torch.int32, device="cuda"), 0, 0,
            "relaxed", "gpu", B, C)
body = h.asm["ptx"].split(".entry", 1)[1]
L = body.splitlines()
for i, l in enumerate(L):
    if "bar.sync" in l:
        print(" | ".join(x.strip() for x in L[max(0, i - 2):i + 3]))
