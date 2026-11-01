# attn_bwd_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/06-fused-attention.py`](source/06-fused-attention.py). Lean: [`lean/`](lean/),
the module `AttnBwd` (namespace `Kernels.AttnBwd`) and its parts:

- `AttnBwd.Model`: the model (`attn_bwd_dkdv`, `attn_bwd_dq`, `attn_bwd`, `model`, `launch`) and
  the conditions (`Inst`, `Inst.ok`, `Pre`);
- `AttnBwd.Spec`: the specification (`HeadV`, `HeadV.Ok`, `OutSpec`, `Post`, `Spec`) and the
  assumptions;
- `AttnBwd.Lane`, `AttnBwd.Acc`, `AttnBwd.Basic`, `AttnBwd.Step`: one element of `pT`/`dsT`
  (`lane`), one accumulation step (`accI_dotAdd`), the loops' invariants (`KVInv`, `QInv`) and
  one iteration of each loop (`kv_step`, `q_step`);
- `AttnBwd.DkDv`, `AttnBwd.Dq`: the two helpers, loop included (`safe_dkdv`, `safe_dq`);
- `AttnBwd.Out`, `AttnBwd.Program`: the stored values (`out_dv`, `out_scaled`, `qp_final`) and
  one program (`safe_program`);
- `AttnBwd`: `main`, `consistent`, `satisfiable`.

The kernel is `_attn_bwd` (lines 366–500) with its two `@triton.jit` helpers `_attn_bwd_dkdv`
(266–315) and `_attn_bwd_dq` (319–363), which it calls by name and the front end inlines. They
read the module-level names `triton` (their decorators) and `tl`, bound by the imports of lines
20–21, which are kernel lines and are mapped (dropped: no step), and the helpers' names, bound only
by their own `def`s. The host's launch is in `_attention.backward` (lines 582–623): it checks
`do.is_contiguous()`, `q.stride() == k.stride() == v.stride() == o.stride() == do.stride()` and
`N_CTX % 128 == 0`, allocates `dq`, `dk`, `dv` with `torch.empty_like`, computes
`arg_k = k * (sm_scale * RCP_LN2)` and `delta` (by `_attn_bwd_preprocess`, another folder's
kernel), and launches `_attn_bwd[grid]` with `grid = (N_CTX // 128, 1, BATCH * N_HEAD)`,
`BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2`,
`HEAD_DIM=ctx.HEAD_DIM`, `CAUSAL=ctx.causal`, `num_warps=4`, `num_stages=5`. The script runs it in
the test (`Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, both
`causal`, `sm_scale = 0.5`, lines 631–695) and in the benchmark (`BATCH = 4`, `N_HEADS = 32`,
`N_CTX = 2^10 … 2^14`, `HEAD_DIM ∈ {64, 128}`, `sm_scale = 1.3`, lines 705–775).

Every other line is a host line or unused (`kernel.json`): the imports of `pytest`, `torch`, `os`
and `TensorDescriptor`, `DEVICE`, the device queries `is_hip` … `is_hopper`, the forward's
`_host_descriptor_pre_hook`, the configs and `keep`/`prune_invalid_configs` (which nothing calls in
this source), `_attention` with its `forward` and `backward`, `attention`, the test and the
benchmark. The `@triton.jit` functions `_attn_fwd_inner`, `_maybe_make_tensor_desc`, `_attn_fwd`
and `_attn_bwd_preprocess` are not run by this launch (`source.unused`; `_attn_fwd` and
`_attn_bwd_preprocess` are other folders' launches); their headers are host lines, and bind only
their own names, which `_attn_bwd` does not read.

## Specification

`Post d init fin` (`AttnBwd.Spec`) says two things.

**The outputs.** Write, for head `b < BATCH·N_HEAD`, `Q`, `K`, `V`, `dO` for its
`N_CTX × HEAD_DIM` blocks of `q`, `arg_k`, `v`, `do`, and `L`, `Δ` for its rows of `M` and
`delta`, all as `float16`/`float32` values in the initial memory (`HeadV`, `HeadV.Holds`). For a key
row `n` and a query row `m`, let `Allowed n m` be `CAUSAL = false ∨ n ≤ m`, and (in ℝ, exactly):

- `P[n, m] = 2^(Σ_c K[n, c]·Q[m, c] − L[m])` when `Allowed`, else `0` (`pE`);
- `dS[n, m] = P[n, m]·(Σ_c V[n, c]·dO[m, c] − Δ[m])` (`dsE`);
- `dV[n, j] = Σ_m P[n, m]·dO[m, j]` (`dVE`);
- `dK[n, j] = s · Σ_m dS[n, m]·Q[m, j]`, `s` the `float32` `sm_scale` the kernel receives (`dKE`);
- `dQ[m, j] = LN2 · Σ_n dS[n, m]·K[n, j]`, `LN2` the kernel's `float32` constant
  `0x3f317218 = 1453635/2097152` (`dQE`, `ln2_val`).

For every `b`, row `r < N_CTX` and column `j < HEAD_DIM`, each of the three outputs `dv`, `dk`, `dq`
holds at element `(b, r, j)` some `float16` value `c` that can be read in the final memory
(`OutSpec`). And whenever head `b`'s values are covered (`HeadV.Ok`, below), `c` is the `float16`
rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` with
`|v − E| ≤ T` (`E` the exact value above and `T` its tolerance), and when
`|E| + T ≤ 65504`, `c` is finite and `|c − E| ≤ T + (|E| + T)·2^-11 + 2^-25`.

With the host's inputs these are the gradients of attention: `arg_k` is `k·sm_scale/ln 2`
(rounded to `float16`), the forward's `M` is the row's base-2 log-sum-exp of the scaled scores, so
`P` is the softmax probability, and `delta = Σ_j O·dO`; `LN2·K = sm_scale·k`. The claim does not
assume those relations: it speaks of the values in memory, whatever they are.

**The tolerances** are explicit real expressions in the inputs (`dVTol`, `dKTol`, `dQTol`):

- per pair, the error bound of the `float16` `P` (`p16Err`: the exponent's dot over `HEAD_DIM`
  products, `Rules.Num.relDot16 HEAD_DIM = (HEAD_DIM + 2)·2^-22` relative to the lifted magnitude
  `aQK = Σ lm K·lm Q` plus
  `absDot16`, the `float32` subtraction, `tl.math.exp2`'s `2^-22·2^x + 2^-126`, one `float16`
  rounding), and of the `float16` `dS` (`ds16Err`: the same for `dP − Δ`, the product with `P`, two
  roundings), weighted by `|dO|`, `|Q|` or `|K|`;
- for the accumulation over `T` steps of `acc += tl.dot(a, b)` (at most 32 products each),
  `accB T B = ((1 + ρ)^T − 1)·B + T·(1 + ρ)^T·α`, with `ρ = DotLoop.stepRel relDot16 32`
  (`≈ 8.29·10^-6`), `α = stepAbs absDot16 32` (`≈ 34·2^-126`) and `B` the sum of the bounds of the
  products' lifted magnitudes (`p16L·lm dO`, `ds16L·lm Q`, `ds16L·lm K`); `T` is
  `stepsKV`/`stepsQ`, the steps the program takes for that row:
  `N_CTX/32`, or with `CAUSAL` `8` masked steps plus the rest;
- for `dk` and `dq`, the product by `s` or `LN2` scales that, plus one `float32` rounding.

**How large the tolerances are** depends on the inputs, not only on the sizes. For `dv`,
`dVTol` is about `r·Σ_m P[n, m]·|dO[m, j]|` plus an absolute floor, with

`r ≈ ln 2·(HEAD_DIM + 2)·2^-22·A + 2^-11 + ((1 + ρ)^T − 1)`:

- `ln 2·(HEAD_DIM + 2)·2^-22·A` is the exponent's error carried through `2^x` (`p16Err`), where `A`
  is the `P·|dO|`-weighted mean over the row of `aQK = Σ_c lm K[n, c]·lm Q[m, c]` (`Σ_c |K·Q|`
  when no operand is subnormal). It grows with the scores'
  magnitudes, which `HeadV.Ok` bounds only through `xE + eX ≤ 15`, so no fixed percentage holds for
  every covered input;
- `2^-11` is the `float16` rounding of `P`;
- `(1 + ρ)^T − 1` is the accumulation: `≈ 0.11%` at `T ≤ 132` (`N_CTX = 4096`, the test's
  largest), `≈ 0.43%` at `T ≤ 516` (`N_CTX = 2^14`, the largest covered);
- the floor is mainly `2^-25` per allowed pair (the `float16` rounding of a small `P`), about
  `N_CTX·2^-25·mean|dO|` (`≈ 4·10^-4` at `N_CTX = 2^14` with `|dO| ~ 1`); the `α` term is below
  `10^-33`. It matters only on rows that get little attention. Since library round 25 the
  accumulation's `B` also counts a `float16` `P` or `dS` that may be a nonzero subnormal at `2^-14`
  (`lift16`: one certainly below `2^-24` is `0` and counts `0`); on the host's inputs this changes
  the figures below by at most `0.07` percentage points (most by `0.01` or less).

`dKTol` and `dQTol` have the same relative part times `|s|·Σ|dS·Q|` (`LN2·Σ|dS·K|`), and also
carry the error of the computed `dP − Δ` (`eY ≈ (HEAD_DIM + 2)·2^-22·Σ_c |V·dO|` per pair, times
`P`), which does not shrink when `dP ≈ Δ`. So on a row where `dS` nearly cancels (a query whose
softmax one key dominates; query row `0` with `CAUSAL`, where `dS` is `0` up to rounding) the
tolerance is large next to `Σ|dS·x|`, though small in absolute terms.

The figures below come from evaluating `Spec.lean`'s formulas, in float64, on one head drawn from
the host's input distributions, with the script below. These are numpy's draws, not torch's, so
they are not the host's exact values.

- **Inputs.** One head (`b = 0`), with `N_CTX × HEAD_DIM` blocks `q`, `k`, `v` drawn from
  `numpy.random.default_rng(20)` as standard normals times `std` and rounded to `float16`, in that
  order, then `do` as standard normals (the host's `randn_like`). The test uses `N_CTX = 4096`,
  `std = 0.5`, `sm_scale = 0.5` (lines 645–648); the benchmark uses `N_CTX = 2^14`, `std = 1`,
  `sm_scale = 1.3` (lines 744–753). Each runs with `HEAD_DIM ∈ {64, 128}` and both `CAUSAL`.
- **Derived values.** `arg_k = float16(k·float32(sm_scale·RCP_LN2))` (line 597). The forward's
  `M` is the exact base-2 log-sum-exp of `q·kᵀ·sm_scale·RCP_LN2` over the allowed keys, rounded to
  `float32`. `o` is the exact softmax times `v`, rounded to `float16`. `delta = Σ_j o·do`, rounded
  to `float32`. These stand in for the forward's and the preprocess's own roundings.
- **The pair quantities.** `aQK`, `aVD`, `xE`, `eX`, `yE`, `eY`, `yB`, `pBr`, `pE`, `dsE`,
  `p16Err`, `p16B`, `p16L`, `dsErr`, `dsB`, `ds16Err`, `ds16B` and `ds16L` are `Spec.lean`'s
  formulas, with `lm`, `lift16`, `u32`, `eta32`, `u16`, `eta16`, `rH`, `aH`, `rhoA`, `alphaA` and
  `accB` as there.
- **The tolerances.** `stepsKV` and `stepsQ` give `T`, and `dVTol`, `dKTol` (with `dKS`, `dKAcc`)
  and `dQTol` (with `dQS`, `dQAcc`) are `Spec.lean`'s formulas.
- **The rows.** The sampled rows are `0, N_CTX/32, 2·N_CTX/32, …` together with
  `127, 128, N_CTX/2 + 5, N_CTX − 129, N_CTX − 1`: as key rows `n` for `dV` and `dK`, and as query
  rows `m` for `dQ`.
- **The figures.** For each sampled row, the figure is the tolerance summed over `j`, divided by
  the magnitude sum summed over `j`: `Σ_m pE·|dO|` for `dV`, `|s|·Σ_m |dsE·Q|` for `dK`,
  `LN2·Σ_n |dsE·K|` for `dQ`. The table gives its median / 90th percentile (numpy's `median` and
  `quantile`, linear) over the sampled rows, non-causal; causal. For `dV` and `dK` only the rows with
  attention mass `Σ_m pE ≥ 1/8` count; for `dQ` every row counts. `A` is the median, over the
  sampled key rows, of the `pE·Σ_j|dO|`-weighted mean of `aQK` over `m`.

```python
import numpy as np                       # numpy 2.3.5, float64 throughout
f16, f32, f64 = np.float16, np.float32, np.float64
u32, eta32, u16, eta16 = 2.0**-24, 2.0**-150, 2.0**-11, 2.0**-25         # Spec: u32 eta32 u16 eta16
rho = 34 * 2.0**-22 + 3 * 2.0**-24 * (1 + 34 * 2.0**-22)                  # Spec: rhoA (rhoA_eq)
alpha = 34 * 2.0**-126 * (1 + 3 * 2.0**-24) + 4 * 2.0**-150               # Spec: alphaA (alphaA_eq)
RCP_LN2, LN2F = 1.4426950408889634, 1453635 / 2097152                    # line 595; ln2F (ln2_val)

def accB(T, B):                                                           # Spec: accB
    return ((1 + rho)**T - 1) * B + T * (1 + rho)**T * alpha

def lm(x):                               # Spec: lm (Rules.Num.liftMag .f16), on float64 values
    return np.where(x == 0, 0.0, np.maximum(np.abs(x), 2.0**-14))

def lift16(B):                                                            # Spec: lift16
    return np.where(B < 2.0**-24, 0.0, np.maximum(B, 2.0**-14))

def run(name, N, HD, causal, sd, sm, seed=20):
    rng = np.random.default_rng(seed)   # one head: (N_CTX, HEAD_DIM) blocks, the host's distributions
    q, k, v = [(rng.standard_normal((N, HD)) * sd).astype(f16) for _ in range(3)]
    do = rng.standard_normal((N, HD)).astype(f16)                         # randn_like: std 1
    K = (k.astype(f32) * f32(sm * RCP_LN2)).astype(f16).astype(f64)       # arg_k (line 597)
    Q, V, dO = q.astype(f64), v.astype(f64), do.astype(f64)
    s = float(f32(sm))                                                    # smF
    rH, aH = (HD + 2) * 2.0**-22, (HD + 2) * 2.0**-126                    # Spec: rH, aH
    def eSub(sv, av, x):                                                  # Spec: eSub
        return rH * av + aH + 3 * u32 * (np.abs(sv) + rH * av + aH + np.abs(x)) + 4 * eta32
    # the forward's M (base-2 log-sum-exp of the scaled scores, float32) and delta = sum_j o*do
    M, Dl = np.empty(N), np.empty(N)
    for a in range(0, N, 1024):
        S = (Q[a:a + 1024] * (sm * RCP_LN2)) @ k.astype(f64).T
        if causal:
            S[np.arange(a, a + 1024)[:, None] < np.arange(N)[None, :]] = -np.inf
        mx = S.max(1, keepdims=True); P = np.exp2(S - mx); l = P.sum(1, keepdims=True)
        M[a:a + 1024] = (mx + np.log2(l))[:, 0]
        o = ((P / l) @ V).astype(f16).astype(f64)
        Dl[a:a + 1024] = (o * dO[a:a + 1024]).sum(1)
    M, Dl = M.astype(f32).astype(f64), Dl.astype(f32).astype(f64)
    def pair(sQK, aQK, L, sVD, aVD, Dm, al):                             # one row of pairs
        xE, eX = sQK - L, eSub(sQK, aQK, L)                               # Spec: xE, eX
        yE, eY = sVD - Dm, eSub(sVD, aVD, Dm)                             # Spec: yE, eY
        yB = np.abs(yE) + eY                                              # Spec: yB
        pBr = (1 + 2.0**-22) * np.exp2(xE + eX) + 2.0**-126               # Spec: pBr
        p2 = np.exp2(xE)
        pE = np.where(al, p2, 0.0)                                        # Spec: pE
        dsErr = (pBr - p2) * yB + p2 * eY + u32 * (pBr * yB) + eta32      # Spec: dsErr
        dsB = (1 + u32) * (pBr * yB) + eta32                              # Spec: dsB
        return (pE, pE * yE,                                              # Spec: pE, dsE
                np.where(al, pBr - p2 + u16 * pBr + eta16, 0.0),          # Spec: p16Err
                lift16(np.where(al, (1 + u16) * pBr + eta16, 0.0)),       # Spec: p16L (lift16 p16B)
                np.where(al, dsErr + u16 * dsB + eta16, 0.0),             # Spec: ds16Err
                lift16(np.where(al, (1 + u16) * dsB + eta16, 0.0)))       # Spec: ds16L (lift16 ds16B)
    rows = sorted(set(range(0, N, N // 32)) | {127, 128, N // 2 + 5, N - 129, N - 1})
    out = {"dV": [], "dK": [], "dQ": []}; A = []
    for n in rows:                                                        # key row n: dV, dK
        al = np.logical_or(not causal, n <= np.arange(N))                 # Spec: Allowed n m
        aQK = lm(Q) @ lm(K[n])                                            # Spec: aQK n m
        pE, dsE, p16Err, p16L, ds16Err, ds16L = pair(Q @ K[n], aQK, M, dO @ V[n],
                                                     lm(dO) @ lm(V[n]), Dl, al)
        T = 8 + (N - 128 * (n // 128) - 128) // 32 if causal else N // 32  # Spec: stepsKV
        tolV = p16Err @ np.abs(dO) + accB(T, p16L @ lm(dO))               # Spec: dVTol n j
        out["dV"].append((pE.sum(), tolV.sum() / (pE @ np.abs(dO)).sum(), tolV.max()))
        dKS = dsE @ Q; acc = ds16Err @ np.abs(Q) + accB(T, ds16L @ lm(Q))       # Spec: dKS, dKAcc
        tolK = abs(s) * acc + u32 * (abs(s) * (np.abs(dKS) + acc)) + eta32      # Spec: dKTol
        out["dK"].append((pE.sum(), tolK.sum() / (abs(s) * (np.abs(dsE) @ np.abs(Q))).sum(), tolK.max()))
        w = pE * np.abs(dO).sum(1); A.append((w @ aQK) / w.sum())
    for m in rows:                                                        # query row m: dQ
        al = np.logical_or(not causal, np.arange(N) <= m)
        pE, dsE, p16Err, p16L, ds16Err, ds16L = pair(K @ Q[m], lm(K) @ lm(Q[m]), M[m],
                                                     V @ dO[m], lm(V) @ lm(dO[m]), Dl[m], al)
        T = 8 + 128 * (m // 128) // 32 if causal else N // 32             # Spec: stepsQ
        dQS = dsE @ K; acc = ds16Err @ np.abs(K) + accB(T, ds16L @ lm(K))       # Spec: dQS, dQAcc
        tolQ = LN2F * acc + u32 * (LN2F * (np.abs(dQS) + acc)) + eta32          # Spec: dQTol
        out["dQ"].append((1.0, tolQ.sum() / (LN2F * (np.abs(dsE) @ np.abs(K))).sum(), tolQ.max()))
    print(f"{name} N_CTX={N} HEAD_DIM={HD} CAUSAL={causal}  A = {np.median(A):.0f}")
    for key in ("dV", "dK", "dQ"):
        r = np.array([x[1] for x in out[key] if x[0] >= 1 / 8])          # attention mass >= 1/8
        worst = max(out[key], key=lambda x: x[1])
        print(f"  {key}: median {100 * np.median(r):.2f}%  90th {100 * np.quantile(r, 0.9):.2f}%  "
              f"max {100 * r.max():.2f}%  (all rows: worst {100 * worst[1]:.3g}%, abs {worst[2]:.2g})")

for name, N, sd, sm in [("test", 4096, 0.5, 0.5), ("benchmark", 16384, 1.0, 1.3)]:
    for HD in (64, 128):
        for causal in (False, True):
            run(name, N, HD, causal, sd, sm)
```

| inputs | `A` | `dVTol` | `dKTol` | `dQTol` |
|---|---|---|---|---|
| test, `N_CTX = 4096`, `HEAD_DIM = 64` | 7 | 0.18 / 0.18; 0.13 / 0.16% | 0.18 / 0.18; 0.13 / 0.17% | 0.18 / 0.18; 0.13 / 0.18% |
| test, `N_CTX = 4096`, `HEAD_DIM = 128` | 15 | 0.21 / 0.21; 0.15 / 0.19% | 0.22 / 0.22; 0.17 / 0.22% | 0.22 / 0.22; 0.17 / 0.22% |
| benchmark, `N_CTX = 2^14`, `HEAD_DIM = 64` | 82 | 0.61 / 0.85; 0.50 / 0.61% | 0.61 / 0.66; 0.49 / 0.59% | 0.62 / 2.1; 0.54 / 3.3% |
| benchmark, `N_CTX = 2^14`, `HEAD_DIM = 128` | 166–168 | 0.90 / 0.95; 0.79 / 0.87% | 0.96 / 1.12; 0.85 / 1.51% | 1.0 / 4.8; 0.95 / 16% |

These match the formula for `r` (benchmark, `HEAD_DIM = 128`: `0.36% + 0.05% + 0.43% ≈ 0.84%`).
The largest `dV`/`dK` ratio among the counted rows is `1.04%` / `1.19%` (non-causal) and `0.92%` /
`5.2%` (causal: a row where `dS` partly cancels). The large `dQ` ratios come from `dS`
cancelling: the worst sampled rows (query `0` with `CAUSAL`; one non-causal benchmark row at each
`HEAD_DIM`) have tolerances of `2.6·10^-4` to `9.2·10^-3` in absolute terms. On the benchmark's
inputs the softmax is sharp (`A ≈ 167`), so many key rows get attention mass below `1/8`, and on
those the floor is most of `dVTol`. A kernel that drops a block, masks the wrong side of the
diagonal, uses the wrong row of `M` or `delta`, or skips a scale breaks `Post` on some inputs. The
bound is relative to the sum of magnitudes, not to `|E|`.

**`HeadV.Ok`**, the premise of the accuracy statement (not of `Pre`: safety and the frame hold for
any values, NaNs included): `Q`, `K`, `V`, `dO`, `L`, `Δ` finite, `|L|, |Δ| ≤ 2^126`, and for every
`Allowed` pair the exponent with its error bound at most `15` and the bound on `|P·(dP − Δ)|` at
most `2^15`. These keep every `float32` and `float16` intermediate in range; on the host's inputs
`P ≤ 1` (the exponent is at most about `0`) and `dS` is small, so they hold with room.

**The frame.** Every byte outside the three outputs' elements (`outBytes`) is unchanged.

## Conditions

`Inst` holds the shapes (`Z`, `H`, `N_CTX`, `HEAD_DIM`, as variables), `CAUSAL`, the double
`sm_scale`, and the nine buffers. The layout is the contiguous one: element `(z, h, r, c)` of a 4-D
tensor is element `((z·H + h)·N_CTX + r)·HEAD_DIM + c` (`eIdx`), of `M`/`delta` element
`(z·H + h)·N_CTX + r` (`rIdx`); the model passes the strides `H·N_CTX·HEAD_DIM`,
`N_CTX·HEAD_DIM`, `HEAD_DIM`, `1`.

`Inst.ok`:

- `hd`: `HEAD_DIM ∈ {16, 32, 64, 128, 256}`, the forward's assertion (line 512), which made `M`;
- `n_mod`: `N_CTX % 128 = 0`, the host's assertion (line 599);
- `n_le`: `N_CTX ≤ 2^14`, the benchmark's largest (Narrowings);
- `g_lt`: `BATCH·N_HEAD < 2^16`, the grid's axis 2 (`Launch.Valid`);
- `size_lt`: `BATCH·N_HEAD·N_CTX·HEAD_DIM < 2^31`, so the `int32` offsets `bhid * N_CTX` and `adj`
  (lines 384–385), the strides and the pointer offsets do not wrap;
- `sm`: `sm_scale` as `float32` is finite with `|s| ≤ 2^16` (the host's `0.5`, `1.3`);
- `q_len … d_len`: each storage holds its tensor;
- `out_in`, `dq_dk`, `dq_dv`, `dk_dv`: the outputs share no byte with the inputs or with each
  other (`torch.empty_like` allocates them).

`Pre d init`: the nine buffers are valid (aligned, inside `[0, 2^64)`, every byte present).
Values are unconstrained.

`launch d`: the grid `(N_CTX/128, 1, BATCH·N_HEAD)`, `fusion = true` (Triton's default), the
defaults `num_ctas = 1`, `launch_pdl = False`, `debug = false`, and the source's block shapes
(`blocks`). `num_warps=4` and `num_stages=5` are the only launch options, both of the four that do
not change the meaning (`Compiler.Faithful` scope 4). The launch is a direct `kernel[grid](…)` (no
`@triton.autotune`, no `@triton.heuristics`), builds no descriptor and takes no host descriptor
(scope 5, 8), is not a `CompiledKernel` launch (scope 6), and the source captures no CUDA graph
(scope 9: `do_bench`, not `do_bench_cudagraph`).

## Assumptions

All `[rule]`, no `[kernel]` assumption:

- `faithful`: `Rules.Compiler.Faithful P (model d) (launch d) (K d)` on every covered instance;
- `dot`: `Rules.Num.DotF16 P`, for the seven `tl.dot`s of `float16` operands into `float32`
  (lines 293, 303, 307, 310, 345, 353, 358). Since library round 25 it is a `DotLiftAcc` fact:
  the bound is relative to `|acc| + Σ liftMag a·liftMag b`, a subnormal operand counted at
  `2^-14`;
- `dot_total`: `Rules.Num.DotTotal P` for the model's `DotCfg`;
- `exp2`: `Rules.Num.Exp2F32 P`, `tl.math.exp2` on `float32` (lines 294, 346);
- `exp2_total`: `Rules.Num.MathTotal P .exp2 .binary32`.

## Proof

`main` is `TVA.meets_independent` with, for program `i` (`p = i % (N_CTX/128)`,
`b = i / (N_CTX/128)`): the read footprint `inBytes d`, the write footprint `W d b p` (the program's
`128 × HEAD_DIM` tiles of `dv`, `dk`, `dq`, rows `128p …`), and `Qp` (`OutSpec` at those rows).
Different programs' tiles are different elements (`tileG_inj`, `W_disj`); `W ⊆ outBytes`, disjoint
from the inputs (`out_in`). `Post` follows by reading each element from the program that wrote it
(`mem_tile`, `outSpec_congr`) and the frame from the footprints.

`safe_program` follows the kernel line by line with the library's `Safe` rules (`safe_irem1`,
`safe_idiv1`, `safe_load_buf`, `safe_store_buf`, `safe_mul`, `safe_cvt`, `safe_forRange`, …),
proving that every `int32` value is the intended natural number (no wrap), every pointer is
`base + element·size` of an element inside its buffer, and the memory is unchanged until the
stores. The helpers' loops are `safe_dkdv` and `safe_dq`: by `safe_forRange` with the invariant
`KVState`/`QState` (the carried names, the step count, and `KVInv`/`QInv`: the accumulators are
`AccI`, within `accB` of exact sums over the rows covered so far). One step is `kv_step`/`q_step`:
each lane of `pT`/`dsT` (`lane`, by `Exp2F32`, `FloatV.AddRel.err`, `mul_err`, `cvt_err`), masked
lanes exactly `0` (`mask_kv`, `mask_q`), and the accumulation (`accI_dotAdd`, by `DotF16` and
`DotLift.dotAdd_step`, with each product's lifted magnitude bounded by `p16L·lm dO`, `ds16L·lm Q`
or `ds16L·lm K`: `lm_le_lift16`, from `|P16| ≤ p16B`, `|dS16| ≤ ds16B` and `f16_small_zero`). The two phases of each output (masked and unmasked) cover disjoint row sets
whose union with the not-`Allowed` rows is all `N_CTX` rows (`kv_cover_sub`, `kv_vanish`,
`q_cover_sub`, `q_vanish`); at those rows the exact terms vanish (`vanish`), so the sums are over
all rows (`accI_range`). `out_dv`/`out_scaled`/`qp_final` give `OutSpec`, with the sizes of
`scaled_size` (every sum below `2^47`, so the products by `s` and `LN2` are in range).

`consistent`: `exists_total_faithful Platform.sm90ref` with `sm90ref.dotF16`, `ref.dotTotal`,
`sm90ref.exp2F32` and `ref.mathTotal`. `satisfiable`: the test's shape `Z = 1`, `H = 2`,
`N_CTX = 128`, `HEAD_DIM = 64`, not causal, `sm_scale = 0.5`, nine buffers at `k·2^21`.

The proofs use only `propext`, `Classical.choice` and `Quot.sound`. No library request.

## What changed (library round 25)

Library round 25 (issue L126) restated `Rules.Num.DotF16`. The tensor core reads a subnormal
`float16` operand's exponent field as the smallest normal's, so the fact is now a `DotLiftAcc`
fact: its bound is relative to `|acc| + Σ liftMag a·liftMag b`, with a subnormal operand counted at
`2^-14`. The claim is re-proved on the restated fact, and subnormal operands stay covered: an input
may be subnormal, and the kernel makes subnormal `P` and `dS` (`p.to(float16)`).

- **Specification.** The magnitudes the tolerances use for a `tl.dot`'s operands are the lifted
  ones:
  - `aQK` and `aVD` are `Σ_c lm·lm`, with `lm x = Rules.Num.liftMag .f16 x` (`0` for `0`, `|x|` for
    a normal value, `2^-14` for a subnormal one).
  - The accumulations' `B` in `dVTol`, `dKAcc`, `dQAcc` is `Σ p16L·lm dO`, `Σ ds16L·lm Q`,
    `Σ ds16L·lm K`. Here `p16L = lift16 p16B` and `ds16L = lift16 ds16B`, with
    `lift16 B = 0` when `B < 2^-24` (the `float16` value is then `0`), else `max B 2^-14`.

  On a zero or normal `float16` value, `lm` is the old magnitude `|x|`: so `aQK` and `aVD`, and
  `lm dO`, `lm Q`, `lm K`, are unchanged on inputs with no subnormal element. `p16L` and `ds16L`
  lift the bounds `p16B` and `ds16B`, not the `float16` values. They equal the old bounds when
  these are at least `2^-14`. They are `2^-14` when a bound lies in `[2^-24, 2^-14)`, even where the
  `float16` `P` or `dS` turns out to be `0`, and `0` when a bound is below `2^-24`.

  The definitions of the per-pair errors (`p16Err`, `ds16Err`) and of `HeadV.Ok` are unchanged.
  Through `eX` and `eY` (and so `pBr` and `yB`), though, they read the lifted `aQK` and `aVD`. So on
  inputs with subnormal elements the per-pair errors are slightly larger and `HeadV.Ok.fits` is
  slightly stronger. Each subnormal operand adds at most `2^-14` times its partner's `lm` to `aQK`
  or `aVD`. The conditions and the model are unchanged.
- **Proofs.** `dot0` and `accI_dotAdd` use `DotLift.dotAdd_step` and the lifted sums. New lemmas
  cover lifted magnitudes: `lm_le_lift16`, `f16_small_zero`, `lm_le`, `abs_mul_le_lm`,
  `sum_bd_le_lm`, `p16L_bd`, `ds16L_bd`, `vanishL`.
- **Figures.** The tolerances moved by at most `0.07` percentage points. The cells of the table
  that changed, from the script's output before and after:
  - test, `HEAD_DIM = 128`, non-causal, `dVTol`: `0.20 / 0.20%` to `0.21 / 0.21%`;
  - benchmark, `HEAD_DIM = 64`, non-causal, `dVTol` 90th percentile: `0.83` to `0.85%`;
  - benchmark, `HEAD_DIM = 64`, causal, `dVTol`: `0.49 / 0.60%` to `0.50 / 0.61%`;
  - benchmark, `HEAD_DIM = 128`, causal, `dKTol` 90th percentile: `1.50` to `1.51%`.

  The largest causal `dV` ratio in the text after the table went from `0.91` to `0.92%`. Some
  figures moved without changing the table's rounded cells. The `dQTol` 90th percentiles went from
  `2.06` to `2.13%` (benchmark, `HEAD_DIM = 64`, non-causal: the largest move), from `3.27` to
  `3.30%` (`HEAD_DIM = 64`, causal), from `4.75` to `4.76%` (`HEAD_DIM = 128`, non-causal) and from
  `15.92` to `15.93%` (`HEAD_DIM = 128`, causal). The README's statements of what `Post` rejects
  still hold.

## Narrowings and modelling decisions

**Narrowings.**

- `N_CTX ≤ 2^14`: the benchmark's largest. The host allows larger `N_CTX`. The bound keeps the
  step counts at most `516`, so the accumulation part of the tolerance stays at most `0.43%`; the
  per-pair part depends on the inputs (Specification). With `size_lt`, it also keeps the `int32`
  arithmetic exact.
- The layout: the claim covers only the canonical contiguous strides, which the model passes:
  `H·N_CTX·HEAD_DIM`, `N_CTX·HEAD_DIM`, `HEAD_DIM`, `1`. The kernel addresses all seven 4-D tensors
  with `q`'s strides. The host asserts `do.is_contiguous()` (line 585) and
  `q.stride() == k.stride() == v.stride() == o.stride() == do.stride()` (line 586), so these
  tensors have the canonical stride along every dimension of size greater than 1
  (`is_contiguous` ignores the stride of a size-1 dimension). On a non-empty launch
  `N_CTX ≥ 128` and `HEAD_DIM ≥ 16`, so the host's checks still allow two non-canonical strides:
  `stride(0)` when `BATCH = 1`, and `stride(1)` when `N_HEAD = 1`. The kernel would multiply such
  a stride by `bhid // H = 0` or `bhid % H = 0`, but the claim does not cover those launches.
  The claim also covers an empty launch (`N_CTX = 0`, which `n_mod` allows, or `BATCH = 0` or
  `N_HEAD = 0`), but only with the canonical strides. Such a tensor has no elements, so
  `is_contiguous` holds for any strides and the host's checks allow others. These launches with
  other strides are the only other ones left out, and they run no program (the grid has
  `N_CTX/128 · BATCH·N_HEAD = 0` programs), so nothing is lost. `M` and `delta` come from
  `torch.empty` and `torch.empty_like`, so they are contiguous.
- `BATCH·N_HEAD < 2^16` and `BATCH·N_HEAD·N_CTX·HEAD_DIM < 2^31`: larger launches exceed the grid
  limit or wrap `adj` in `int32`, so the kernel would address other elements; they are not covered.
- `sm_scale` is a Python `float`: `Inst.sm_scale` is the double the host passes, and the
  kernel receives it as a `float32`. The test passes `0.5` and the benchmark `1.3`. The claim
  requires that `float32` to be finite, with magnitude at most `2^16`. An `int` `sm_scale` gives
  the kernel another parameter type (`1` is a compile-time constant, another `int` an `int32`;
  SEMANTICS.md, "Scalar arguments"), so such launches are not covered.
- `HeadV.Ok` limits the accuracy statement, not safety or the frame (Specification).

**Modelling decisions.**

- Each helper is a Lean function applied at its calls ("Helpers are inlined"), the shape
  constexprs first; parameters the body does not read are `_`-prefixed. The two calls of each
  helper have different copy names (`16`/`True` and `32`/`False`).
- A `if CAUSAL:` (compile-time) is a Lean `if` on the parameter, parenthesised and bound once, with
  the names its branch assigns as its tuple result. `_attn_bwd_dq`'s `if MASK:` (lines 348–351),
  which assigns `offs_n` and `p`, is two `let`s on `MASK`. The loops carry the names defined before
  them that their bodies assign; the dq loop carries `offs_n` also when not `MASK` (unchanged, the
  same values).
- Names assigned Python numbers (`start_m = 0`, `step_m`, `num_steps = BLOCK_N1 // MASK_BLOCK_M1`)
  hold `int32` scalars; annotated names (`LN2`, `MASK_BLOCK_M1`, `MASK_BLOCK_N2`) are compile-time
  values; `LN2` is built in `float32` at line 499 (`FloatV.litF32` of its double).
- `stride_d = 1` (and `H` when it is `1`) is specialised to the compile-time `1`; every use builds
  it in `int32`, so the model's `int32` `1` gives the same values.
- `sm_scale` is the double rounded to `float32` (`FloatV.hostF32`).
- `tl.dot(…) − x` is the fusable `tl.subc`; `acc += tl.dot(a, b)` is `tl.dot_add`; `.to(tl.float32)`
  of a `float32` dot is `tl.convert` (same values); `tl.where(mask, x, 0.0)` uses `+0.0`.
