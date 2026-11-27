"""Dump the L40's float32 -> fp8 results on a sample of inputs, for CvtCheck.lean (the library's own
functions against the device). Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/l40/num-cvt8/dump.py OUT.txt
One line per input: `pattern e4m3 e5m2` (decimal bit patterns). The sample: every 4093rd float32
pattern; the first 200000 inputs where one rounding and two differ (double.py); the 2^16 patterns
around float16's largest value, float8's maxVal and the infinities; every NaN-adjacent edge."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "num-cvt8"))
from run import cvt  # noqa: E402
from double import rz16, sat_bits  # noqa: E402

pats = set(range(0, 1 << 32, 4093))
for c in (0x477FE000, 0x477FF000, 0x43E00000, 0x47600000, 0x7F800000, 0x7F7FFFFF, 0x00800000, 0x00000001,
          0x3A800000, 0x37000000, 0x33800000, 0x387FC000):
    for s in (0, 1 << 31):
        pats |= {(s | (c + d)) & 0xFFFFFFFF for d in range(-4096, 4096)}
# inputs where one rounding and two differ
total, CH, extra = 1 << 32, 1 << 26, []
for lo in range(0, total, CH):
    if len(extra) >= 200000:
        break
    b64 = torch.arange(lo, lo + CH, dtype=torch.int64, device="cuda")
    xf = torch.where(b64 >= (1 << 31), b64 - total, b64).to(torch.int32).view(torch.float32).to(torch.float64)
    d = ((sat_bits(xf, "float8e4nv") != sat_bits(rz16(xf), "float8e4nv")) |
         (sat_bits(xf, "float8e5") != sat_bits(rz16(xf), "float8e5"))) & ~torch.isnan(xf)
    extra += (b64[d][:200000 - len(extra)]).tolist()
pats |= set(extra)
pats = sorted(p for p in pats if 0 <= p < total)
b = torch.tensor(pats, dtype=torch.int64, device="cuda")
bits = torch.where(b >= (1 << 31), b - total, b).to(torch.int32)
n = len(pats); e4 = torch.empty(n, dtype=torch.int8, device="cuda"); e5 = torch.empty(n, dtype=torch.int8, device="cuda")
cvt[((n + 1023) // 1024,)](bits, e4, e5, n, SRC=0, B=1024)
with open(sys.argv[1], "w") as f:
    for p, a, c in zip(pats, (e4.to(torch.int64) & 0xFF).tolist(), (e5.to(torch.int64) & 0xFF).tolist()):
        f.write(f"{p} {a} {c}\n")
print(f"{n} inputs ({len(extra)} where one rounding and two differ)")
