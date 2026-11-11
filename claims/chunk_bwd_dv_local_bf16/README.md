# `chunk_bwd_dv_local_bf16`

**The kernel.** `chunk_bwd_kernel_dv_local` of flash-linear-attention (fla) v0.6.0
(`source/chunk_o.py`, kernel lines 74–149 with the `@triton.jit` helper `exp2`, lines 70–71; host
`chunk_bwd_dv_local`, lines 191–241). The intra-chunk value gradient of chunked (gated) linear
attention: one program per (chunk, batch row and value head) accumulates the `[BT, BT]` matrix
`k·qᵀ·scale` with `tl.dot` over `K` in steps of `BK`, applies the gate (`tl.exp2` of the gate
differences), masks it causally, rounds it to `bfloat16`, and for each block of `BV` columns stores
`tl.dot(A, do)` rounded to `bfloat16` into `dv`. The claim is `Claims.ChunkBwdDvLocal.claim` in
`Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
sequence `n` (a row of a fixed-length batch, or `[cu_seqlens[n], cu_seqlens[n+1])`), token `p` of
it, value head `hv < HV` and column `e < V`:
- `dv[p, hv, e]` is a finite `bfloat16` `c` with `|c − ref| ≤ tol`, where
  `ref = Σ_{j=p}^{chunkEnd−1} Aex(p, j)·do[j, hv, e]`, `Aex = σ·kq·wt` exact in the reals:
  `kq(p, j) = Σ_c k[p, h, c]·q[j, h, c]`, `h = hv // (HV // H)`, `wt = 2^{g[j, hv] − g[p, hv]}` with a
  gate and `1` without, `σ` the host's double `scale`, `chunkEnd` the end of `p`'s chunk of `BT`
  tokens counted from the sequence's start;
- every byte outside the written elements of `dv` is unchanged.

`tol = 2^-7·Σ_j |Aex|·|do_j| + 2^-13·Σ_j aMag·liftMag do_j + 2^-100` over `j ∈ [p, chunkEnd)`, with
`aMag = σ·(kqMag + 2^-94)·(wt + 2^-112) + 2^-86`, `kqMag = Σ_c liftMag k·liftMag q`
(`Device.liftMag .bf16`). The first term is the kernel's two `bfloat16` roundings (of each entry of
`A`, line 140, and of the stored result, line 149); the second the `float32` errors of the entries
and of the second `tl.dot`, relative to the products' magnitudes. The reference is computed from
`cu_seqlens` (or the rows), `q`, `k`, `g` and `do` alone, never from `chunk_indices`.

## Conditions

`Inst.ok` and `Pre`:
- `BT = chunk_size ∈ {16, 32, 64}`; `BK`, `BV` powers of two in `[16, 128]` (the host's values);
- `1 ≤ K ≤ 256`, `1 ≤ V ≤ 2^16`, `HV·V < 2^31` (every constexpr of the kernel an `int32` constant);
  `H ≥ 1` divides `HV ≥ 1`; `B ≥ 1`, and `B = 1` on variable length;
- `T < 2^31` (an `int32` argument), `T ≤ 2^22`, `B ≤ 8192` and, on variable length, `N ≤ 8192`
  (see Exclusions);
- the grid: `NT < 2^31`, `B·HV < 2^16` (see Exclusions), fewer than `2^31` programs;
- `scale` a finite double in `[2^-32, 2^32]`;
- the buffers hold the contiguous tensors and are valid; `dv` shares no byte with `q`, `k`, `do`,
  `g` and the tables;
- variable length: `int32` or `int64` tables, `N < 2^31`, the tables' lengths, the tables holding
  the instance's contents and `TablesOk` (`cu_seqlens` nonnegative, nondecreasing and at most `T`;
  each row of `chunk_indices` a distinct (sequence, chunk) whose chunk starts inside its sequence;
  every such chunk a row: what `prepare_chunk_indices` builds, in any row order);
- `ValsOk`: in every sequence, every `q`, `k`, `do` component finite, `|do| ≤ 2^100`, every `g`
  finite (with a gate); every pair `p ≤ j` of one chunk with `g[j] − g[p] ≤ 64` (with a gate),
  `kqMag ≤ 2^100` and `σ·kqMag·(1 + wUp) ≤ 2^100`; every output with
  `Σ_j σ·kqMag·wUp·liftMag do ≤ 2^120`.

**The launches covered** (`SETUP.md`, `setup_*_ok`, each with and without a gate): the host's test
cases (fixed and variable length), fla's benchmark shapes, gated DeltaNet's default layer, a packed
batch of `2^16` tokens with Qwen3-Next's heads (`int32` and `int64` tables), and the largest grid
`y` (`B·HV = 65535`). `setup_var3` carries its real tables, which `setup_var3_tables` proves
`TablesOk`. `canon_ok` covers the host's launch at every size within `Inst.ok` with tensors of at
most `2^38` elements.

**Device facts** the tolerance rests on (all in `Device P`): `Device.DotBF16` (`tl.dot` of
`bfloat16` into `float32`: relative `relDot16 n = (n + 2)·2^-22`, absolute
`absDot16 n = (n + 2)·2^-126`, on the lifted magnitude), `Device.Exp2F32` (`tl.exp2` on `float32`:
relative `2^-22`, absolute `2^-126`), `Device.Exp2F32NegInf` (`exp2(−∞) = +0`, for a gate
difference that overflows), and the totality facts for `tl.dot` and `exp2`.

## Tolerances

Fix a pair `p ≤ j` of one chunk, `L = kqMag(p, j)`, `S = kq(p, j)`; `u = 2^-24`, `η = 2^-150`;
`nK = cdiv(K, BK)`, `εA = (BK + 8 + 2·nK)·2^-22`; `wErr`, `wUp` the gate's error and upper bound
(`Claim.lean`); the entry's error bound `eA = σ·L·(εA·wUp + wErr) + wUp·(σ·2^-110 + 2^-140) + 2^-140`.

1. **The `K` loop** (`nK` trips of a `tl.dot` then `tl.mul_add` with `hostF32(σ)`):
   `|P − σ₃₂·S| ≤ σ₃₂·L·((BK+2)·2^-22 + (nK+1)·u)·(1 + 2^-20) + nK·(σ₃₂·(BK+2)·2^-126 + 2^-148)`.
2. **The gate**: the IEEE `fl(g[j] − g[p])`, then `Exp2F32` (or `Exp2F32NegInf` on an overflow to
   `−∞`): `|G − 2^{dg}| ≤ wErr`, `G ≤ wUp`.
3. **The entry before rounding**: `|Ā − Aex| ≤ eA`.
4. **Line 140**: masked lanes are `+0`; a valid entry rounds to `bfloat16`:
   `|Â − Aex| ≤ 2^-8·|Aex| + (1 + 2^-8)·eA + 2^-134`.
5. **Line 148** (`tl.dot`, `K = BT`): within `(BT+2)·2^-22·Σ_j liftMag Â·liftMag do + (BT+2)·2^-126`.
6. **The store**: `|c − ref| ≤ tol₀ = 2^-8·|ref| + (1 + 2^-8)·(2^-8·B1 + (1 + 2^-8)·Be +
   (BT + 3)·2^-22·Bl) + 2^-100·(1 + Dl)`, with `B1 = Σ_j |Aex|·|do_j|`, `Be = Σ_j eA·|do_j|`,
   `Bl = Σ_j (|Aex| + eA)·liftMag do_j`, `Dl = Σ_j liftMag do_j`; `c` is finite.
7. **`tol₀ ≤ tol`**: `wErr ≤ 2^-14·wt + 2^-125`, `εA ≤ 140·2^-22`, so `eA ≤ 8.0·10^-5·aMag' + 2^-140`
   (`aMag' = aMag − 2^-86`), and
   `tol₀ ≤ 2^-7·B1 + 1.12·10^-4·Σ_j aMag'·d' + 2^-99·Dl + 2^-100 ≤ tol`.

On the H200 the worst error is `0.934` of `tol`; over `SETUP.md`'s launches `tol / |ref|` has median
`2.4·10^-2`.

## Exclusions

- **`B·HV ≥ 65536` on fixed-length batches**: a limit of the kernel. The host puts `B·HV` on grid
  axis `y`, which CUDA limits to `65535`; the launch fails. Variable-length batches (`B = 1`) never
  reach it.
- **More than `2^31` programs or chunks**: CUDA's and the launcher's limit.
- **Gate values**: a gate rising by more than about 128 within a chunk overflows `exp2`, a limit of
  the kernel; `ValsOk` keeps `g[j] − g[p] ≤ 64` (fla's gates are nonincreasing within a chunk).
- **Input magnitudes** beyond `ValsOk`'s room bounds: `float32` overflows near `2^128`.
- **`scale` outside `[2^-32, 2^32]`**, or an `int` `scale` (another specialisation).
- **`V > 2^16` or `HV·V ≥ 2^31`**: outside real use; at `V ≥ 2^31` the kernel is not correct (the
  constexpr `V` becomes a `uint32` constant).
- **Contexts longer than `2^22` tokens, batches of more than `8192`**: the claim covers real model
  shapes only (`T_ctx`, `B_le`, `N_le`). `K`, `V` are head dims and `H`, `HV` head counts, so no
  field is a model dimension.
- **`K > 256`, chunk sizes other than 16, 32, 64, `USE_G_GAMMA`, `USE_A`, `float16`/`float32`
  inputs, tables not `TablesOk`, `dv` overlapping the inputs**: outside real use or the setup.

`check_gpu.py` (one H200) checks `Post` on every `SETUP.md` launch with and without a gate and on
the edges of `Inst.ok` and `Pre`, checks `ValsOk` and `TablesOk` on the inputs, rejects wrong
outputs, and shows `B·HV = 65536` failing to launch.
