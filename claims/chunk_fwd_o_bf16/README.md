# `chunk_fwd_o_bf16`

**The kernel.** `chunk_fwd_kernel_o` of flash-linear-attention (fla) v0.6.0
(`source/chunk_o.py`, kernel lines 83–181, with the `@triton.jit` helpers `unflatten_program_id`,
lines 69–76, and `exp2`, lines 79–80; host `chunk_fwd_o`, lines 223–273, and fla's
`prepare_chunk_indices`). The output of chunked linear attention: one program per (block of `BV`
value columns, 64-token chunk, batch row and value head) loops over `K` in steps of `BK`,
accumulating `q·S` (the chunk's state from `h`) and the scores `q·kᵀ` with `tl.dot`, applies the
gates (`g` or `g_gamma`, through `tl.exp2`), masks the scores causally, rounds them to `bfloat16`,
multiplies them by `v` with `tl.dot`, scales, and stores the `[64, BV]` tile of `o` in `bfloat16`.
The claim is `Claims.ChunkFwdO.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
chunk `r` of the batch, token `t` in it, value head `j < HV` and column `e < V`:
- `o[t, j, e]` holds a `bfloat16` value `c`;
- when the element is `InRange`, `c` is finite and `|c − ref| ≤ tol`, with
  `ref = σ·(G·QS + Σ_{s ≤ t in the chunk} A_s·w_s·v[s, j, e])` exact in the reals: `σ` the scale,
  `QS = Σ_x q[t, qh j, x]·S_r[j, x, e]`, `A_s = Σ_x q[t, qh j, x]·k[s, qh j, x]`,
  `qh j = j // (HV // H)`, and the gates `G = 2^g[t, j]`, `w_s = 2^(g[t, j] − g[s, j])` with `g`,
  `G = 2^(γ_j·(i+1))`, `w_s = 2^(γ_j·(t − s))` with `g_gamma` (`i` the local position of `t`), `1`
  with neither;
- every byte outside the written elements of `o` is unchanged.

`tol = 2^-8·|ref| + (1 + 2^-8)·σ·(2^-8·Mabs + (2^-13 + Γ)·Mag + 2^-100·Flo) + 2^-130`, with
`Mabs = Σ_s |A_s|·w_s·|v_s|`, `Mag = G·PQS + Σ_s w_s·PA_s·L(v_s)`,
`Flo = 1 + G + PQS + Σ_s (1 + w_s + PA_s)·L(v_s)`, `PQS = Σ_x L(q)·L(S_r)`, `PA_s = Σ_x L(q)·L(k_s)`,
`L = Device.liftMag .bf16` (a subnormal counted at `2^-126`), and `Γ = 2^-15·|γ_j|` with `g_gamma`,
`0` otherwise. Its main terms are the kernel's two `bfloat16` roundings (the store's and the
scores').

`InRange` (per element): the values the element reads are finite (the query row and state column,
the keys up to `t`, the value column over the whole chunk); with `g`, `|g[s, j]| ≤ 2^64`,
`g[t, j] ≤ 64` and `g[t, j] − g[s, j] ≤ 64` for `s ≤ t`; with `g_gamma`, `γ_j` finite with
`|γ_j| ≤ 64` and `γ_j ≤ 1`; `PQS`, `G·PQS`, each `PA_s`, `w_s·PA_s`, `Mag + Flo` and `σ·(Mag + Flo)`
at most `2^100`. Outside `InRange`, only "holds a `bfloat16` value" is claimed.

## Conditions

`Inst.ok` and `Pre`:
- `BK = BV ∈ {32, 64, 128}`, the source's configurations on Hopper (whichever the autotuner picks);
- `B, H, HV ≥ 1`, `H ∣ HV`; `1 ≤ K ≤ 256`, `1 ≤ V ≤ 512`;
- `T < 2^31` (an `int32` argument); on a fixed-length batch of two rows or more also
  `T + 63 < 2^31` (see Exclusions);
- `T ≤ 2^22`, `B ≤ 8192` and, on variable length, `N ≤ 8192` (see Exclusions);
- at most one of `g`, `g_gamma`; `B = 1` on variable length;
- the grid `(cdiv(V, BV)·NT, B·HV)` within CUDA's limits (`B·HV < 2^16`) and fewer than `2^31`
  programs;
- each buffer holds its tensor (`g_gamma` at least `HV` entries) and is valid in the initial memory
  with the instance's contents; `o` shares no byte with `q`, `k`, `v`, `h`, `g`, `g_gamma` or the
  tables;
- variable length: `int32` or `int64` tables, `N < 2^31`, the tables' lengths, and `TablesOk`:
  `cu_seqlens` nonnegative, nondecreasing and at most `T`, each row of `chunk_indices` a chunk that
  starts inside its sequence, no two rows equal, every chunk a row (the table
  `prepare_chunk_indices` builds, in any row order);
- `scale` a finite double in `[2^-64, 2^64]` (by default `K^-0.5 ∈ [1/16, 1]`).

The values are covered element by element (`InRange`).

**The launches covered** (`SETUP.md`, `setup_*_ok`): fla's `test_chunk_fwd_o` cases, its
deterministic varlen test, Qwen3-Next's heads (packed `8 × 4096` and `2^20` tokens), gated
DeltaNet's defaults (`K = 256`, `V = 512`, `int64` tables), fla's benchmark shapes, head dims 60
and 100, RetNet's and lightning attention's `g_gamma`, the largest fixed-length batch below the
grid limit (`B = 2047` at 32 value heads), and the other two configurations (`BK = BV = 32`, `128`)
on a selection of these. `mk_ok` covers the host's launch at every size within `Inst.ok` with
tensors of at most `2^40` bytes.

**Device facts** the tolerance rests on (all in `Device P`): `Device.DotBF16` (`tl.dot` of
`bfloat16` into `float32`: relative `relDot16 n = (n + 2)·2^-22`, absolute
`absDot16 n = (n + 2)·2^-126`, on the lifted magnitude), `Device.Exp2F32` (`tl.exp2` on `float32`:
relative `2^-22`, absolute `2^-126`), and the totality facts for `tl.dot` and `exp2`. The other
float operations are IEEE.

## Tolerances

For an `InRange` element, `u = 2^-24`, `η = 2^-150`; `n_K = cdiv(K, BK) ≤ 256/BK`;
`ρ = relDot16 BK`, `α = absDot16 BK` for the two accumulating dots, `ρ_64 = 66·2^-22`,
`α_64 = 66·2^-126` for the last (`K`-dimension 64).

1. **The accumulating dots** (`n_K` steps): the error is at most `((1+ρ)^{n_K} − 1)·(P + α/ρ)`,
   `≤ δ = 6.49·10^-5` relative in every configuration (`bk = 32`: `6.49·10^-5`; `64`: `6.30·10^-5`;
   `128`: `6.20·10^-5`), so `|ô − QS| ≤ δ·PQS + 2^-117`, `|Â_s − A_s| ≤ δ·PA_s + 2^-117`.
2. **The gates.** With `g`: `|Ĝ − G| ≤ 2^-22·G + 2^-126`; `ŵ_s = exp2(fl(g[t] − g[s]))` is within
   `2^-17·w_s + 2^-124` of `w_s` (the IEEE `−` moves the exponent by at most `126u` on
   `[−126, 64]`; below `−126` both are under `2^-124`). With `g_gamma`: `b_i = fl(γ·(i+1))` is
   within `2^-18·|γ| + η` of `γ(i+1)`, and `tl.subc` (fused or not) gives an exponent within
   `2.5·2^-17·|γ| + 5η` of `γ(i − i')`; with `|γ| ≤ 64` the gates' extra relative errors carried to
   `c` stay below `1.76·2^-17·|γ| ≤ Γ`.
3. **The gated values**: `|õ − G·QS| ≤ 6.52·10^-5·G·PQS + floors`,
   `|Ã_s − A_s·w_s| ≤ 7.26·10^-5·w_s·PA_s + floors`.
4. **The scores' rounding** to `bfloat16`: `2^-8·|Ã_s| + 2^-134`.
5. **The last dot** (`K = 64`): `|D − Σ_s A_s·w_s·v_s| ≤ 2^-8·Mabs + 8.87·10^-5·Σ_s w_s·PA_s·L(v_s)
   + floors`.
6. **The scale and `tl.addc`** (three roundings at most on either side; `hostF32(σ)` within `u·σ`):
   the value `x` before the store has `|x − ref| ≤ E = σ·(2^-8·Mabs + c·Mag + f) + 2^-147` with
   `c ≤ 8.90·10^-5 ≤ 2^-13` (`2^-13 + Γ` with `g_gamma`) and `f ≤ 2^-110·Flo`.
7. **The store** (nearest even): `|c − ref| ≤ 2^-8·|ref| + (1 + 2^-8)·E + 2^-134 ≤ tol`; `c` is
   finite since `|x| ≤ 2^101`.

Room: `InRange`'s `2^100` bounds keep every dot, product and the result below `2^128`. The store's
rounding alone reaches `2^-8/(1 + 2^-8)·|x|`, so the leading `2^-8·|ref|` cannot be smaller; on
the GPU check the median tolerance is `0.008` to `0.030` of `|ref|`, and the worst element is at
`0.906` of `tol`.

## Exclusions

- **Fixed-length batches with `B·HV ≥ 65536`**: a limit of the kernel. `B·HV` is grid axis 1,
  which CUDA limits to `65535`; the launch fails on the host. Packed batches have `B = 1`.
- **`2^31 − 63 ≤ T < 2^31` on a fixed-length batch of two rows or more**: a limit of the kernel.
  Line 119's `int32` `T + 63` wraps, `NT` is negative and rows `b ≥ 1` read the wrong chunk of `h`.
  With one row or on variable length the wrap does not matter, and these are covered. `T ≥ 2^31`
  makes `T` an `int64` (another compiled kernel).
- **A grid of `2^31` programs or more**: Triton's launcher rejects it.
- **Contexts longer than `2^22` tokens, batches of more than `8192`**: the claim covers real model
  shapes only (`T_le`, `B_le`, `N_le`). `K`, `V` are head dims and `H`, `HV` head counts, so no
  field is a model dimension.
- **Elements not `InRange`**: such an element is still a `bfloat16` value.
- **Tables not `TablesOk`, `B > 1` with `cu_seqlens`, both `g` and `g_gamma`, `g_gamma` with
  fewer than `HV` entries, `K > 256`, `V > 512`, `chunk_size` 16 or 32, `float16`/`float32`
  inputs**: outside real use or the setup.

`check_gpu.py` (one H200) checks `Post` on 52 launches (29 at `BK = BV = 64`, 23 at `32` or
`128`), checks `TablesOk` and `InRange` on the inputs, rejects wrong outputs, and shows the grid
limit failing at `B·HV = 65536`.
