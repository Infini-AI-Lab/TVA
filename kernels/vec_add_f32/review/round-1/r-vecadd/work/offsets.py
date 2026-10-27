# Reviewer r-vecadd, round 1: the int32 offsets of add_kernel at the n = 2^31 boundary.
# Run with: python3 -I offsets.py
#
# n = 2^31: for every BLOCK_SIZE = 2^e <= 2^20, the largest offset pid*B + l is 2^31 - 1, so
# nothing wraps (n_elements is then passed as int64, which the model's I32 does not cover).
# n = 2^31 + 1, B = 1024: the last program's pid*B wraps to -2^31 (out-of-bounds accesses).

n = 2 ** 31
ok = True
for e in range(21):
    B = 2 ** e
    g = (n + B - 1) // B
    mx = (g - 1) * B + B - 1
    ok = ok and mx == 2 ** 31 - 1
print("n = 2^31: largest offset is 2^31 - 1 for every B = 2^e <= 2^20:", ok)

B = 1024
n = 2 ** 31 + 1
g = (n + B - 1) // B
p = g - 1
wrapped = (p * B + 2 ** 31) % 2 ** 32 - 2 ** 31
print("n = 2^31 + 1, B = 1024: last pid", p, "pid*B as int32", wrapped)
