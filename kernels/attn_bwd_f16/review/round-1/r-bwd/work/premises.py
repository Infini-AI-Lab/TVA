"""r-bwd, round 1: do the host's inputs satisfy `HeadV.Ok`, and how large are the tolerances?

One head of the source's host data, simulated on the CPU in float64 (no GPU, no Triton):
q, k, v, do as the benchmark (randn, std 1, sm_scale = 1.3) or the test (std 0.5,
sm_scale = 0.5) make them, rounded to float16; the forward's `M` (the row's base-2
log-sum-exp of `q.k * qk_scale`, `qk_scale = fp32(fp32(sm_scale) * fp32(1.44269504))`, causal
mask as the forward's), its output `o` (rounded to float16) and `delta = sum_j o * do`;
`arg_k = fp16(k * (sm_scale * RCP_LN2))` as the host computes it.

Then, by `Spec.lean`'s formulas, over every `Allowed` pair: the largest `xE + eX` (`HeadV.Ok.fits`
asks <= 15) and the largest `pBr * yB` (asks <= 2^15); and, at a few rows, `dVTol`, `dKTol`,
`dQTol` against the exact values and against the sums of magnitudes they are relative to.

Usage: python3 -I premises.py N HEAD_DIM causal(0|1) mode(bench|test)
"""

import sys

import numpy as np

N = int(sys.argv[1])
HD = int(sys.argv[2])
CAUSAL = sys.argv[3] == "1"
MODE = sys.argv[4]

rng = np.random.default_rng(20)
std, sm = (1.0, 1.3) if MODE == "bench" else (0.5, 0.5)


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


q = f16(rng.standard_normal((N, HD)) * std)
k = f16(rng.standard_normal((N, HD)) * std)
v = f16(rng.standard_normal((N, HD)) * std)
dO = f16(rng.standard_normal((N, HD)))

smF = float(f32(sm))
qk_scale = float(f32(smF * float(f32(1.44269504))))
RCP_LN2 = 1.4426950408889634
argk = f16(f32(k) * f32(sm * RCP_LN2))
ln2F = 1453635 / 2097152

u32, eta32, u16, eta16 = 2.0**-24, 2.0**-150, 2.0**-11, 2.0**-25
rH, aH = (HD + 2) * 2.0**-22, (HD + 2) * 2.0**-126
rho = 34 * 2.0**-22 + 3 * 2.0**-24 * (1 + 34 * 2.0**-22)
alpha = 34 * 2.0**-126 * (1 + 3 * 2.0**-24) + 4 * 2.0**-150


def accB(T, B):
    return ((1 + rho) ** T - 1) * B + T * (1 + rho) ** T * alpha


def eSub(s, a, x):
    return rH * a + aH + 3 * u32 * (np.abs(s) + rH * a + aH + np.abs(x)) + 4 * eta32


CH = 1024
idx = np.arange(N)

# the forward pass: M, o (float16), then delta
M = np.empty(N)
o = np.empty((N, HD))
for m0 in range(0, N, CH):
    s = (q[m0:m0 + CH] @ k.T) * qk_scale
    if CAUSAL:
        s = np.where(idx[None, :] <= idx[m0:m0 + CH, None], s, -np.inf)
    mx = s.max(axis=1)
    w = np.exp2(s - mx[:, None])
    M[m0:m0 + CH] = mx + np.log2(w.sum(axis=1))
    o[m0:m0 + CH] = (w / w.sum(axis=1)[:, None]) @ v
o = f16(o)
M = f32(M)
Dl = f32((o * dO).sum(axis=1))

# the premises of `HeadV.Ok` over every Allowed pair (query rows m in chunks, all keys n)
max_fit1 = -np.inf
max_fit2 = 0.0
max_xE = -np.inf
max_aQK = 0.0
for m0 in range(0, N, CH):
    qs, ds = q[m0:m0 + CH], dO[m0:m0 + CH]
    sQK = qs @ argk.T
    aQK = np.abs(qs) @ np.abs(argk).T
    xE = sQK - M[m0:m0 + CH, None]
    eX = eSub(sQK, aQK, M[m0:m0 + CH, None])
    sVD = ds @ v.T
    aVD = np.abs(ds) @ np.abs(v).T
    yE = sVD - Dl[m0:m0 + CH, None]
    eY = eSub(sVD, aVD, Dl[m0:m0 + CH, None])
    pBr = (1 + 2.0**-22) * np.exp2(xE + eX) + 2.0**-126
    al = (idx[None, :] <= idx[m0:m0 + CH, None]) if CAUSAL else np.ones_like(xE, dtype=bool)
    max_fit1 = max(max_fit1, (xE + eX)[al].max())
    max_fit2 = max(max_fit2, (pBr * (np.abs(yE) + eY))[al].max())
    max_xE = max(max_xE, xE[al].max())
    max_aQK = max(max_aQK, aQK[al].max())

print(f"N={N} HEAD_DIM={HD} CAUSAL={CAUSAL} mode={MODE} (std {std}, sm_scale {sm})")
print(f"  max xE = {max_xE:.4f}, max aQK = {max_aQK:.1f}")
print(f"  fits.1: max (xE + eX) = {max_fit1:.4f}  (needs <= 15)")
print(f"  fits.2: max pBr*yB   = {max_fit2:.2f}  (needs <= 32768)")


def pair_terms(n_rows, m_rows):
    """Spec's per-pair quantities for keys n_rows x queries m_rows."""
    sQK = argk[n_rows] @ q[m_rows].T
    aQK = np.abs(argk[n_rows]) @ np.abs(q[m_rows]).T
    xE = sQK - M[m_rows][None, :]
    eX = eSub(sQK, aQK, M[m_rows][None, :])
    sVD = v[n_rows] @ dO[m_rows].T
    aVD = np.abs(v[n_rows]) @ np.abs(dO[m_rows]).T
    yE = sVD - Dl[m_rows][None, :]
    eY = eSub(sVD, aVD, Dl[m_rows][None, :])
    yB = np.abs(yE) + eY
    pBr = (1 + 2.0**-22) * np.exp2(xE + eX) + 2.0**-126
    al = (n_rows[:, None] <= m_rows[None, :]) if CAUSAL else np.ones_like(xE, dtype=bool)
    two = np.exp2(xE)
    pE = np.where(al, two, 0.0)
    dsE = pE * yE
    p16Err = np.where(al, pBr - two + u16 * pBr + eta16, 0.0)
    p16B = np.where(al, (1 + u16) * pBr + eta16, 0.0)
    dsErr = (pBr - two) * yB + two * eY + u32 * pBr * yB + eta32
    dsB = (1 + u32) * pBr * yB + eta32
    ds16Err = np.where(al, dsErr + u16 * dsB + eta16, 0.0)
    ds16B = np.where(al, (1 + u16) * dsB + eta16, 0.0)
    return pE, dsE, p16Err, p16B, ds16Err, ds16B


def stepsKV(n):
    return 8 + (N - 128 * (n // 128) - 128) // 32 if CAUSAL else N // 32


def stepsQ(m):
    return 8 + 128 * (m // 128) // 32 if CAUSAL else N // 32


rows = sorted({0, 128, N // 2, N - 129, N - 1})
print("  dV, dK at key rows n (all queries m), column j = 0..HD-1; ratios over j (median, max):")
for n in rows:
    pE, dsE, p16Err, p16B, ds16Err, ds16B = pair_terms(np.array([n]), idx)
    T = stepsKV(n)
    dVE = pE[0] @ dO
    dVTol = p16Err[0] @ np.abs(dO) + accB(T, p16B[0] @ np.abs(dO))
    magV = np.abs(pE[0]) @ np.abs(dO)
    dKS = dsE[0] @ q
    dKAcc = ds16Err[0] @ np.abs(q) + accB(T, ds16B[0] @ np.abs(q))
    dKE = smF * dKS
    dKTol = abs(smF) * dKAcc + u32 * abs(smF) * (np.abs(dKS) + dKAcc) + eta32
    magK = abs(smF) * (np.abs(dsE[0]) @ np.abs(q))
    print(f"   n={n:5d} T={T:4d}  dVTol/sum|P dO| = {np.median(dVTol / magV):.2e} {np.max(dVTol / magV):.2e}"
          f"  dVTol/|dV| = {np.median(dVTol / np.abs(dVE)):.2e}"
          f"  |dV| median {np.median(np.abs(dVE)):.3g}"
          f"  dKTol/sum|dS Q| = {np.median(dKTol / magK):.2e}  dKTol/|dK| = {np.median(dKTol / np.abs(dKE)):.2e}"
          f"  |dK| median {np.median(np.abs(dKE)):.3g}")

print("  dQ at query rows m (all keys n):")
for m in rows:
    pE, dsE, p16Err, p16B, ds16Err, ds16B = pair_terms(idx, np.array([m]))
    T = stepsQ(m)
    dQS = dsE[:, 0] @ argk
    dQAcc = ds16Err[:, 0] @ np.abs(argk) + accB(T, ds16B[:, 0] @ np.abs(argk))
    dQE = ln2F * dQS
    dQTol = ln2F * dQAcc + u32 * ln2F * (np.abs(dQS) + dQAcc) + eta32
    magQ = ln2F * (np.abs(dsE[:, 0]) @ np.abs(argk))
    print(f"   m={m:5d} T={T:4d}  dQTol/sum|dS K| = {np.median(dQTol / magQ):.2e} {np.max(dQTol / magQ):.2e}"
          f"  dQTol/|dQ| = {np.median(dQTol / np.abs(dQE)):.2e}  |dQ| median {np.median(np.abs(dQE)):.3g}")
