# Reviewer r-vecadd, round 2: the three excluded size ranges of the revised README
# (Narrowings, bullet on n + BLOCK_SIZE <= 2^31), checked for every covered BLOCK_SIZE = 2^e <= 2^20.
# Run with: python3 -I ranges.py

def i32(v):
    return (v + 2 ** 31) % 2 ** 32 - 2 ** 31

ok = True
for e in range(21):
    B = 2 ** e
    # (1) n > 2^31: the last program's pid*B is >= 2^31 and wraps (negative offsets).
    for n in (2 ** 31 + 1, 2 ** 31 + B, 2 ** 32):
        g = (n + B - 1) // B
        p = g - 1
        ok = ok and p >= 2 ** 31 // B and p * B >= 2 ** 31
        # every lane of that program has a negative int32 offset when n <= 2^32
        ok = ok and all(i32(i32(p * B) + l) < 0 for l in (0, B - 1))
    # from n > 2^32 on, some program's pid*B wraps to a non-negative offset (another program's)
    n = 2 ** 32 + 1
    p = (n + B - 1) // B - 1
    ok = ok and i32(p * B) == 0
    # (2) n = 2^31: the largest offset is 2^31 - 1, nothing wraps
    n = 2 ** 31
    g = (n + B - 1) // B
    ok = ok and (g - 1) * B + B - 1 == 2 ** 31 - 1 and i32((g - 1) * B) == (g - 1) * B
    # (3) 2^31 - B < n < 2^31: B - 1 sizes; their last program has p*B + B = 2^31
    ok = ok and len(range(2 ** 31 - B + 1, 2 ** 31)) == B - 1
    for n in ((2 ** 31 - B + 1, 2 ** 31 - 1) if B > 1 else ()):
        p = (n + B - 1) // B - 1
        ok = ok and p * B + B == 2 ** 31          # strict Fits (< 2^31) fails, <= holds
        ok = ok and n + B > 2 ** 31               # outside n_fits
# the host's B = 1024 bound: n <= 2^31 - 1024
ok = ok and (2 ** 31 - 1024) + 1024 <= 2 ** 31 and (2 ** 31 - 1023) + 1024 > 2 ** 31
print("all three ranges as the README states, every B = 2^e <= 2^20:", ok)
