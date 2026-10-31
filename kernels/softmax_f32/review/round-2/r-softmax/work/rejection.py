"""Reviewer r-softmax, round 2: the new Post against wrong kernels, with the claim's constants.

Run with `python3 -I`. Per-element bound after the common factor: (elemTol + gapTol·gap)·σ +
floorTol·σ_max; two outputs of a row must have relative errors within the sum of their bounds
(work/Check2.lean proves that step).
"""
import math
import struct

elemTol = 2**-21 + 2**-30
gapTol = 3 * 2**-24 + 2**-30
floorTol = 2**-133


def sumTol(B):
    return (B + 86) * 2**-24


def f16(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def bf16(x):
    b = struct.unpack('<I', struct.pack('<f', x))[0]
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000            # round to nearest even
    return struct.unpack('<f', struct.pack('<I', b))[0]


def bound(xs, j):
    m = max(xs)
    A = sum(math.exp(x - m) for x in xs)
    s = math.exp(xs[j] - m) / A
    smax = 1 / A
    return elemTol + gapTol * (m - xs[j]) + floorTol * smax / s


print("1. The row (0, -1): the largest difference of relative errors Post allows")
row = [0.0, -1.0]
A = sum(math.exp(x) for x in row)
sig = [math.exp(x) / A for x in row]
allow = bound(row, 0) + bound(row, 1)
print(f"   sigma = {sig}; allowed |d0 - d1| <= {allow:.3e}")
for name, rnd in (("float16", f16), ("bfloat16", bf16)):
    d = [rnd(s) / s - 1 for s in sig]
    print(f"   {name}: d0 = {d[0]:+.3e}, d1 = {d[1]:+.3e}, |d0 - d1| = {abs(d[0] - d[1]):.3e}"
          f" -> rejected: {abs(d[0] - d[1]) > allow}")

print("2. A sum that leaves out one input (k = the last), on a row whose last input is the largest")
for N in (781, 12672):
    row = [0.0] * (N - 1) + [10.0]
    m = max(row)
    a = [math.exp(x - m) for x in row]
    A = sum(a)
    sc_needed = A / (A - a[-1])                      # every other output is scaled by this
    B = 1 << (N - 1).bit_length()
    print(f"   N = {N}, B = {B}: outputs scaled by {sc_needed:.4f}; sumTol = {sumTol(B):.2e}"
          f" -> rejected: {abs(sc_needed - 1) > sumTol(B) + 2 * elemTol}")

print("3. other=0.0 instead of -inf (padding lanes add e^-x_max each), randn-like rows")
for N, xmax in ((781, 3.2), (12672, 4.0)):
    B = 1 << (N - 1).bit_length()
    A = N * math.exp(0.5) * math.exp(-xmax)          # expected exact sum of e^(x - max)
    extra = (B - N) * math.exp(-xmax)
    print(f"   N = {N}, B = {B}: common factor {A / (A + extra):.4f}, sumTol = {sumTol(B):.2e}"
          f" -> rejected: {abs(A / (A + extra) - 1) > sumTol(B) + 2 * elemTol}")

print("4. The common factor at the host's block sizes, against the source's test rtol 1e-5")
for B in (256, 1024, 2**14):
    print(f"   B = {B}: sumTol = {sumTol(B):.2e}; library tree bound (B-1)*2^-24 = {(B - 1) * 2**-24:.2e}")
