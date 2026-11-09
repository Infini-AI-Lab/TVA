# Prepared claim: `fused_moe_bf16`

The kernel: `source/fused_moe.py`, vLLM's `fused_moe_kernel` (lines 109–420)
with its helper `write_zeros_to_output` (lines 89–106), on the unquantised `bfloat16` path with the
tile vLLM picks on an H200 (`try_get_optimal_moe_config`, lines 560–570: the tuned file's entry for
the batch, else `get_default_config`; `SETUP.md`, The config), launched by
the hosts `fused_moe_gemm1` (lines 704–766) and `fused_moe_gemm2` (lines 769–833), vLLM's two expert
GEMMs, through `invoke_fused_moe_triton_kernel` (lines 609–690), with the expert assignment of
`_prepare_expert_assignment` (lines 573–606): `moe_align_block_size`, or for decode-sized batches
the naive assignment. The claim is `Claims.FusedMoe.Statement` in `Claim.lean`:

```lean
def Statement : Prop := ∀ K P, Assumptions K P → Spec K
```

`Claim.lean` proves `consistent`, `satisfiable` and the coverage facts `setup_*_ok`; it does not prove
`Statement`.

The model map is `model-map.json` (59 entries, every
code line of the kernel lines `[85, 86], [89, 106], [109, 420]`).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.FusedMoe.main : Statement` (`Statement`: `Claim.lean`
line 715). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 423), and every initial memory `init` that `Pre` accepts (line 557, which is `PreC`,
line 513), every outcome of the launch is a normal end (no fault, no hang) whose final memory `fin`
satisfies `Post` (line 656).

**What `Post` asks** (line 656). Write `ntp = ntpOf d init` (line 376), `entryOf d init pos` (line
386) for the flat index at position `pos`, `eOf d init pid_m` (line 396) for the expert of block
`pid_m`, `nv d = M·top_k` (line 363), and `stride_cm d = N` (line 367). For every flat index
`f < nv d` and column `c < N`, the element `f·N + c` of `C`:
1. **not routed**: if no position `pos < ntp` has `entryOf d init pos = f`, the element is unchanged:
   both of its bytes (`fin z = init z` for every `z ∈ d.c.elems {f·N + c}`, `Buf.elems`, the bytes of
   that one element);
2. **routed**: for every position `pos < ntp` with `entryOf d init pos = f`, `outElem d init fin pos f c`
   holds (line 634): the element reads in `fin` as some `bfloat16` `cv`, and
   - if the block's expert `eOf d init (pos / BM)` is `-1`: `cv` is `+0.0` up to NaN payload
     (`FloatV.NaNEq (FloatV.ofPat 0) cv`; exact);
   - otherwise the expert is some `e < E`, and for all `va vb : Nat → BF16` such that row `f / top_k`
     of `A` (elements `f / top_k · K + k`) reads as finite `va k` and column `(e, c)` of `B` (elements
     `e·stride_be + c·stride_bn + k`) reads as finite `vb k` for every `k < K`, and
     `dotMag va vb K ≤ 2^120` (line 569, the sum of `Rules.Num.liftMag .bf16` products): there is a
     finite `float32` `u` with `|u.toRat − S| ≤ T`, `T = tol BK K va vb = 2^-12·pathMag va vb BK K +
     2^-133` (lines 608, 599), and `outFin d init f cv u S T` (line 619), where `S = dotSum va vb K`
     (line 611) is the exact dot product in the rationals. `pathMag` sums, over the chunks of 64
     positions the loop's `tl.dot`s cover, the largest of the exact partial sum before the chunk
     (`preSum`), the exact partial sum before the chunk's trip, and the chunk's lifted magnitude
     (`chunkLift`). `outFin` says:
     - without `rw`: `cv` is `FloatV.cvt u` up to NaN payload (the `bfloat16` rounding of `u`), and
       if `|S| + T ≤ bfloat16.maxVal`, `cv` is finite;
     - with `rw`: the weight `wv` at element `f` of `w` reads as finite, some `r` is
       `FloatV.mul u wv` up to NaN payload, `cv` is `FloatV.cvt r` up to NaN payload, and if
       `(|S| + T)·|wv| ≤ 2^126`, `r` and `cv` are finite.

   With a non-finite input along the row or column, or `dotMag > 2^120`, only "holds a `bfloat16`" is
   asked;
3. the frame: every byte outside `d.c.elems (cIdx d)` (line 370, the matrix `C`) is unchanged.

**What you are given** (`Assumptions`, line 693):
- `faithful` (line 699, `Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's
  launch, when the launch is valid, has `debug = false`, and the model's launch is well behaved from
  `init`;
- `dot` (line 706, `Rules.Num.DotBF16Chunk`, a `Dot16BitChunk`): the `tl.dot` of source line 373
  (line 238), chunk by chunk of 64 `k`: chunk `j` within
  `min (2^-13·Mj) (2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| + 2^-143` of `s j + c_j`, with room;
- `dot_total` (line 708, `Rules.Num.DotTotal`): that `tl.dot` always has a result.

**The kernel, as the model runs it** (`fused_moe_kernel`, line 92; `model` line 676; `launch` line
688): a 1-D grid of `cdiv(EM, BM)·cdiv(N, BN)` programs. Program `pid` maps to a block `pid_m` and a
column tile `pid_n` by the grouped ordering (lines 101–125). It loads `ntp` (line 129) and
returns if `pid_m·BM ≥ ntp` (line 132). Otherwise it builds `offs_token` (line 137: loaded from
`sorted_token_ids`, or with `naive` built from `pid_m`), masks lanes with `offs_token < nv`, and loads
the block's expert (line 158). Expert `-1`: `write_zeros_to_output` (line 45) stores `bfloat16` zeros
(line 165). Otherwise: a `tl.forRange` of `cdiv(K, BK)` trips (line 214), each loading a masked
`[BM, BK]` tile of `A` (rows `offs_token / top_k`, line 225) and a `[BK, BN]` tile of `B` (columns
`offs_bn = (pid_n·BN + lane) % N`, line 230), with `other = 0`, then `tl.dot_add` into the `float32`
accumulator (line 238); then, with `rw`, a masked load of the weights and a `tl.mul` (lines 258–260);
`tl.cvt` to `bfloat16` and a masked store (lines 263–285). Programs write disjoint sets of `C`'s
elements and use no atomics: the route is `TVA.meets_independent`; the loop is `safe_forRange` with
`fun _ => ∅` as the trip footprints (`iterW_empty`) (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: fewer than `2^31` programs from
  `grid_lt`; the block shapes from `cfg`);
- each program is `Safe`: the scalar loads (`safe_load1`; `ntp_len`, `elen`, `PreC.ntpEx`), the
  `sorted_token_ids` load without `naive` (`s_len`, `ntpEM`), the `A`, `B` and weight loads and the store of `C`
  (`safe_load`, `safe_store`; `a_len`, `b_len`, `w_len`, `c_len`, `PreC.s1`, `erange`, the masks),
  no division by zero (`safe_idiv`, `safe_irem`; `cfg`, `topk_pos`, `N_pos`), `int32` arithmetic
  that does not wrap (`K_le`, `N_le` (also from `N_le17`), `EM_le`, `N_group`, `nv_le`, `be`, `bn`, `cm`) and `int64`
  sums of `b_ptrs` that do not wrap (`b_off`), and the loop's `tl.RangeOk`;
- footprints disjoint: two programs write different elements of `C`, from the grouped ordering being
  injective, the column tiles of `pid_n`, `PreC.s2` (each valid flat index once) and `ntpBM` (every
  lane of a running block is below `ntp`); no program reads `C`'s matrix (`c_a`, `c_b`, `c_w`, `c_s`,
  `c_e`, `c_n`), which also gives `hRW`;
- the value part, `|u − S| ≤ 2^-12·pathMag + 2^-133`: every block below `ntp` and every column
  tile is run by some program (the ordering covers the grid); derive the fine bound of README
  `## Tolerances`: the loop invariant bounds the accumulator by `accErr` after each trip
  (`safe_dot_add` gives either combined form: the chunk steps from `dot`,
  `Dot16Chunk.DotChunk16Loss.errB` and `Dot8Chunk.chain_err`, or the separate product with
  `AccAdd`, `FloatV.AddRel.err`), the room from `dotMag ≤ 2^120`; then relax `accErr` to
  `2^-12·pathMag + 2^-133` (README, `## Tolerances`, Relaxation). The weight's product
  (`safe_mul`) and the conversion (`safe_cvt`) give `outFin`, the finiteness from the room
  conditions (`|r| ≤ 2^126` with the weight);
- the frame: no program writes outside `cIdx d`, and unrouted elements are written by no program.

**Already in `Claim.lean`** (do not prove again): `consistent` (line 719), `satisfiable` (line 1252),
`hostInst_ok` and the `setup_*_ok` theorems; `row_lt` (line 823), `succ_mul_le` (line 829) and
`bytes_disjoint` (line 834) can be reused.

## Specification

`Post d init fin` (`Claim.lean`, `Post`, `outElem`, `outFin`) says, for every flat index
`f < M·top_k` (`M` the rows of `A`; for the first GEMM the token–expert pair
`(f / top_k, f % top_k)`; for the second GEMM, where `A` is the `[M_tok·top_k_ids, K]`
`intermediate_cache2` and the kernel's `top_k` is `1`, the pair `(f / top_k_ids, f % top_k_ids)`,
which is also `A`'s row `f`) and column `c < N`, of the element `f·N + c` of `C` (`C` is the
contiguous output, `stride_cm = C.stride(1) = N`). A position `pos` below the padded total
`ntp = num_tokens_post_padded` holds the flat index `entryOf pos`: entry `pos` of
`sorted_token_ids`, or with the naive assignment `pos / BLOCK_SIZE_M` in the block's first lane
(`pos % BLOCK_SIZE_M = 0`) and `M·top_k` (masked) in the others, as lines 222–226 build it.

- **Not routed** (no position below `ntp` holds `f`: the pair's expert is not local, as vLLM
  leaves it): the element is unchanged.
- **Routed, block expert `-1`** (the position's block has expert `-1`, vLLM's other expert-parallel
  mode): the element is the `bfloat16` `+0.0` (`write_zeros_to_output`).
- **Routed, block expert `e ∈ [0, E)`**: when row `f / top_k` of `A` and column `(e, c)` of `B` hold
  finite values `va k`, `vb k` (`k < K`) and `dotMag ≤ 2^120`, the element is the `bfloat16` rounding
  of a `float32` value `u` (of the `float32` product `u ⊗ w` with the routing weight `w =
  topk_weights` at `f`, finite, with `MUL_ROUTED_WEIGHT`) with

  `|u − S| ≤ T = 2^-12·pathMag + 2^-133`,

  `S = Σ_k va k · vb k` the exact dot product and `pathMag` the magnitude the chunked
  accumulation carries: over the chunks of 64 positions of the loop's `tl.dot`s (`BK` the tile's
  `BLOCK_SIZE_K`, `BK/64` chunks per trip), the largest of the exact partial sum before the chunk,
  the exact partial sum before the chunk's trip, and the sum of the chunk's lifted products
  (`liftMag`: `|q|`, a nonzero subnormal counted as `2^-126`), summed (Tolerances). `pathMag ≥
  dotMag ≥ |S|`; `dotMag = Σ_k liftMag(va k)·liftMag(vb k)`. With room (`|S| + T` within the
  `bfloat16` range, or `(|S| + T)·|w| ≤ 2^126` with the weight) the output is finite.

  With a NaN or infinity on the row or the column, or `dotMag > 2^120`, it says only that the
  element holds some `bfloat16`.
- **Frame:** every byte outside the matrix `C` (`cIdx`) is unchanged: `A`, `B`, `topk_weights`, the
  alignment buffers, and any storage of `C`'s buffer outside the matrix.

The reference is the exact dot product of row `f / top_k` of `A` (the flat index's token for the
first GEMM, the flat index itself for the second) and the column of the expert of the block holding
`f`. The kernel's interface is the expert assignment: `sorted_token_ids` (or none), `expert_ids` and
`num_tokens_post_padded` are the kernel's inputs, built by the host's `_prepare_expert_assignment`,
and `Pre` states what it guarantees (below). Under the host's assignment (each routed flat index
`t·top_k + j` placed once, in the blocks of its local expert; with the naive assignment block `f`
is flat index `f`, of expert `topk_ids.view(-1)[f]`), `Post` gives, for the first GEMM,
`C[t, j] ≈ A[t]·w1[local(topk_ids[t, j])]ᵀ`, and for the second
`C[t, j] ≈ intermediate_cache2[t·top_k + j]·w2[local(topk_ids[t, j])]ᵀ` (times `topk_weights[t, j]`
with `MUL_ROUTED_WEIGHT`), for every pair routed to a local expert, within the tolerance below, and
leaves the other rows unchanged. `check_gpu.py` computes its reference from the host's own
`topk_ids` (and expert map), which checks the assignment's routing too.

`Spec K := Claim.Meets Inst.ok Pre Post K`: every launch on a covered instance and input ends normally
with a memory satisfying `Post`.

## Conditions

`Inst` (one launch): `M` (`A`'s rows), `top_k` (the kernel's), `E` (the local experts,
`B.size(0)`), `N`, `K`, `EM` (lines 628–639: the `sorted_token_ids` length, its `min` with
`M·top_k·BLOCK_SIZE_M` when `M < BLOCK_SIZE_M`, or `M·top_k·BLOCK_SIZE_M` with the naive
assignment), `B`'s strides `stride_be`, `stride_bn`, the flags `MUL_ROUTED_WEIGHT` (`rw`) and
`naive_block_assignment` (`naive`), the tile `BM`, `BN`, `BK`, `GM`, `topk_weights`' `size(1)` and
`stride(0)` (`w_cols`, `w_stride0`), and seven buffers: `a` (`A`, contiguous `[M, K]`,
`stride_am = K`, `stride_ak = 1`), `b` (`B`, `stride_bk = 1`), `c` (`C`), `w` (`topk_weights`), `s`
(`sorted_token_ids`; unused with `naive`), `eid` (`expert_ids`), `ntpb` (`num_tokens_post_padded`).

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `cfg` | `(BM, BN, BK, GM)` is one of the 56 tiles of vLLM's tuned `bfloat16` H200 files, which include `get_default_config`'s seven (`cfgs`) | the host's configuration (`SETUP.md`, The config) |
| `topk_pos`, `E_pos`, `N_pos`, `K_pos` | positive | the kernel divides by `top_k` (line 283) and by `N` (line 252) |
| `topk_le` | `top_k < 2^31` | the constexpr built in `int64` for `//` |
| `K_le`, `N_le`, `EM_le` | `K + BK`, `N + BN`, `EM + BM ≤ 2^31` | `int32` arguments; `tl.cdiv` adds `B − 1` in `int32` (lines 199, 200, 330) |
| `N_le17` | `N ≤ 2^17` | `N` is a model dimension (twice the intermediate size per rank for the first GEMM, the hidden size for the second): real model shapes (Exclusions); the largest real `N` is `65536` (Grok-1's first GEMM at TP1) |
| `K_le15` | `K ≤ 2^15` | outside real use beyond it (the largest real `K` is `32768`, Grok-1's second GEMM at TP1); the tolerance grows with `K`, to a median `0.094` of the output at `2^15` (Tolerances) |
| `N_group` | `cdiv(N, BN)·GM < 2^31` | `num_pid_in_group` (line 201); implied by `N_le` |
| `grid_lt` | `cdiv(EM, BM)·cdiv(N, BN) < 2^31` | `Launch.Valid` |
| `nv_le` | `M·top_k < 2^31` | `num_valid_tokens` is an `int32` argument |
| `be`, `bn`, `cm` | `stride_be`, `stride_bn`, `N < 2^31` | `int32` arguments (the model's integer class) |
| `b_off` | `e·stride_be + c·stride_bn + k < 2^63` (`k < K + BK`) | the `int64` sums of `b_ptrs` (lines 285–289) |
| `w_contig` | `topk_weights.stride(0) = topk_weights.size(1)` (`w_stride0 = w_cols`) | the kernel reads the weight of flat index `f` at storage element `f` (line 404): `topk_weights[t, j]` only for a contiguous `[M_tok, top_k_ids]` (the host asserts only `stride(1) = 1`) |
| `a_len` … `ntp_len` | the buffers hold the matrices (`s`: `EM` entries, without `naive`) | the kernel reads them |
| `c_a` … `c_n` | the matrix `C` shares no byte with the other six buffers | the host's `C` is fresh |

`Pre` (`PreC`): the seven buffers are valid (`Buf.Valid`); the expert assignment is as
`_prepare_expert_assignment` builds it: `ntp` (`ntpEx`) a multiple of `BLOCK_SIZE_M` (`ntpBM`) and at
most `EM` (`ntpEM`); the entries (`entryOf`) below `ntp` nonnegative (`s1`), the valid ones (below
`M·top_k`) pairwise distinct (`s2`; both hold by definition with the naive assignment); `expert_ids`
covering the blocks below `ntp` (`elen`), each `-1` or in `[0, E)` (`erange`); with
`MUL_ROUTED_WEIGHT`, the weights of the valid flat indices finite (`wfin`). Any `bfloat16` values in
`A` and `B`.

What `Pre` leaves open, because the kernel does not depend on it: the entries of
`sorted_token_ids` below `ntp` are bounded only from below (an entry at or above `M·top_k` fails
`offs_token < num_valid_tokens` and is masked like the padding in the loads of `A`, the weight load
and the store, lines 342–346, 404, 420; a negative entry would pass the mask); the entries past
`ntp` are unconstrained (programs with `pid_m·BM ≥ ntp` return at line 217, and `ntp` is a multiple
of `BLOCK_SIZE_M`); the weights need be finite only with `MUL_ROUTED_WEIGHT` (without the flag the kernel
reads none). The offsets of `a_ptrs` are `int64` (`offs_token` is converted to `int64` at line
229, so `offs_token // top_k * stride_am`, line 283, does not wrap); `a_len` and validity bound
them.

The hosts meet `Pre`: `moe_align_block_size` (lines 423–482) fills `sorted_token_ids` with
`M_tok·top_k_ids`, places each routed flat index once at its expert's padded offset, pads each
expert to a multiple of `BLOCK_SIZE_M`, and writes the expert of each block below the total (local
ids only, since the host passes `ignore_invalid_experts=True`). `ntp ≤ EM` also when
`A.size(0) < BLOCK_SIZE_M`: each nonempty expert's padded count `⌈c_e/BM⌉·BM` is at most `c_e·BM`, so
`ntp ≤ M_tok·top_k_ids·BM`. The naive assignment (no expert map, `M_tok·top_k_ids·4 ≤ E`) writes
`ntp = M_tok·top_k_ids·BM = EM`, and `expert_ids = topk_ids.view(-1)` holds an expert in `[0, E)` for
each of the `M_tok·top_k_ids` blocks. The routing weights are softmax probabilities.

**Part (A) of `Compiler.Faithful`'s scope** (docs/rules/compiler-faithful.md, "Scope"): the launch of
lines 646–690 is a direct launch of the `@triton.jit` kernel (no `@triton.autotune`, no
`@triton.heuristics`), with only `num_warps` (`4` or `8`) and `num_stages` (`2` to `5`) of the
configuration (the config dict, spread at line 689) besides the grid; no `debug`, `num_ctas`,
`launch_pdl` or `enable_fp_fusion` (the defaults: `debug = false`, `num_ctas = 1`,
`enable_fp_fusion = true`), no `warmup`/`triton.compile`, no descriptors (`USE_TD = False`).
`launch d` is `Launch.grid1 (cdiv(EM, BM)·cdiv(N, BN))` with the source's block shapes. **Part (B)**, the environment (points 1,
2, 3, the environment's part of 4, 7, 10, 11), is assumed: the claim relies on it and TVA does not
check it.

The integer arguments: `N`, `K`, `EM`, `num_valid_tokens`, `A.stride(0) = K`, `B.stride(0)`,
`B.stride(1)`, `C.stride(1) = N` are `int32` (covered values below `2^31`); `A.stride(1)`,
`B.stride(2)`, `C.stride(2)` are `1`, which Triton passes as the compile-time constant `1`; the model's
uses give the same values (the `fused_moe_kernel` docstring in `Claim.lean`). The nine quantisation
strides are `0` and unused; `b_bias_ptr`, `a_scale_ptr`, `b_scale_ptr` are `None` and unused, and so
is `sorted_token_ids_ptr` with the naive assignment.

**The supported subset** (docs/SEMANTICS.md): every form of the covered path is the subset's;
`compute_type`, a `tl.constexpr` whose argument is the dtype `tl.bfloat16` ("A dtype as a
`tl.constexpr` argument"), is used as `tl.zeros`'s `dtype=` (line 102) and `.to`'s target (line
413), where the model writes `.bf16`.

## Assumptions

All `[rule]`, cited as docs/PROVER_GUIDE.md section 3 and docs/rules/num.md give them:

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`;
- `dot : Rules.Num.DotBF16Chunk P`: `tl.dot` of `bfloat16` into `float32`, `maxImpreciseAcc = 0`,
  any precision, chunk by chunk (64 consecutive `k` per chunk; the model's line 373, `tl.dot_add` of
  `BLOCK_SIZE_K` products with the source's default `tf32`);
- `dot_total : Rules.Num.DotTotal P {a := .bf16, b := .bf16, acc := .f32, precision := .tf32,
  maxImpreciseAcc := 0}`.

No `[kernel]` assumption. The other operations (`int32`/`int64` arithmetic, `tl.convert`, the
`float32` product by the weight, the `bfloat16` conversion) are exact or IEEE in the library and need
no fact.

## Coverage

Every launch `SETUP.md` asks for, of both GEMMs: `A` contiguous (`[M, K]`, or the second GEMM's
`[M·top_k, K]` `intermediate_cache2`), `B` `[E_local, N, K]` with `stride(-1) = 1`, `C` fresh
`[M, top_k, N]`, `topk_ids` `int32`, the expert assignment of `_prepare_expert_assignment`
(`moe_align_block_size` with or without an expert map, or the naive assignment), the tile the host
picks on an H200 for every batch (any of the 56 tiles of `cfgs`: `BLOCK_SIZE_M` 16–128,
`BLOCK_SIZE_N` 32–256, `BLOCK_SIZE_K` 64, 128 or 256, `GROUP_SIZE_M` 1, 16, 32 or 64),
`MUL_ROUTED_WEIGHT` either way, `E` 1–256, `top_k` 1–8, `K` up to `2^15`, `N` up to
`65536`. `Inst.ok` covers all of them: the binding bounds are `K ≤ 2^15`, `N·K < 2^31`
(`stride_be`), `M·top_k < 2^31`, above the real sizes (the largest: `M = 40000`, `top_k = 8`,
`EM ≤ 320000 + 256·127`).

Proved instances (64 `setup_*_ok`; `gemm1Inst` and `gemm2Inst` compute the assignment, the buffer
sizes and `EM` as the hosts do for the tile each names, the one the host picks on an H200,
`hostInst_ok`):

| Instances | Launches | Source |
|---|---|---|
| `setup_gemm1_*`, `setup_gemm2_*` (18) | the host's nine test cases `(M, N, K, E, top_k)` from `(1, 256, 128, 8, 2)` to `(2, 512, 7168, 256, 8)`, with their second GEMMs: the naive assignment (`M = 1, 2`), `get_default_config`'s tiles and three tuned files' (`E=128,N=768`, `E=8,N=1792`: `(128, 256, 64, 16)`; `E=16,N=1024`: `(16, 64, 128, 1)`), the weight on the input, an expert map | source lines 865–873 |
| `setup_mixtral_8x7b_m{1,16,64,512,40000}_gemm{1,2}` | Mixtral-8x7B TP1 (file `E=8,N=14336`): decode (naive, `(16, 128, 128, 16)`), `(16, 64, 256, 16)` at `M = 16`, `(32, 128, 128, 64)` at `64`, `(128, 256, 64, 16)` from `512`; the second GEMM's `K = 14336` | SETUP |
| `setup_mixtral_8x7b_tp2_m1024_*`, `setup_mixtral_8x7b_tp8_m1_*` | Mixtral-8x7B at TP2 (`(128, 256, 64, 64)`) and TP8 (naive, `(16, 64, 256, 1)`) | SETUP |
| `setup_mixtral_8x22b_m{1,512}_gemm{1,2}` | Mixtral-8x22B TP1 (no file: `get_default_config`); the second GEMM's `K = 16384` | SETUP |
| `setup_grok1_m{1,512}_gemm{1,2}` | Grok-1 TP1 (no file): the first GEMM's `N = 65536`, the second's `K = 2^15` | SETUP |
| `setup_deepseek_v3_m{1,8,9,512,40000}_gemm{1,2}` | DeepSeek-V3 TP1 (no file): naive up to `M = 8`, aligned from `M = 9` | SETUP |
| `setup_qwen3_30b_m{1,4,80,32768}_gemm{1,2}`, `setup_qwen3_30b_m32768_rw_gemm1` | Qwen3-30B-A3B TP1 (file `E=128,N=768`): naive at `M = 1, 4` (`(16, 64, 64, 1)`, `(16, 64, 128, 16)`), `(16, 256, 128, 1)` at `80`, `(128, 256, 64, 16)` at `32768`, the weight on the input | SETUP |
| `setup_llama4_scout_tp8_m{1,64}_gemm{1,2}` | Llama-4-Scout TP8 (file `E=16,N=1024`), the weight on the input: naive at `M = 1` with `(16, 32, 64, 1)` | SETUP |
| `setup_e1_top1_m300_gemm1` | one expert, `top_k = 1`: `GROUP_SIZE_M = 16` with the tile `(64, 128, 64)` | edge |

`satisfiable` uses the first instance with the all-zero memory (`ntp = 0`: every program returns).

## Tolerances

The tolerance is provable under these conditions: the library facts give a fine bound `E`, computed
from the host's values along the exact partial sums of the reduction, chunk by chunk, and `E` is at
most `T = 2^-12·pathMag + 2^-133`.

1. **The chunks.** Position `k` of the reduction holds the product `p_k = va k · vb k` (`0` for
   `k ≥ K`: the last trip's lanes past `K` load `other=0.0`, `+0`), lifted `ℓ_k = liftMag(va k)·
   liftMag(vb k) ≥ |p_k|`. Chunk `g` is the 64 positions from `64·g`; trip `t` of the loop
   (`q = BK/64` chunks) holds the chunks `t·q … t·q + q − 1`, which are exactly the chunks
   `Rules.Num.chunk16` of that trip's `tl.dot` (`BK` is `64`, `128` or `256`). `c_g = Σ_chunk p_k`
   (`chunkSum`), `m_g = Σ_chunk ℓ_k ≥` every lifted product of the chunk and `≥ |c_g|`
   (`chunkLift`), `P_g = Σ_{h<g} c_h` (`preSum`). There are `G = cdiv(K, BK)·q ≤ 512` chunks
   (`K ≤ 2^15`, and `BK` divides `2^15`).
2. **One chunk step** (`Rules.Num.DotBF16Chunk`, its first arm: within
   `2^-13·Mj + 2^-20·|s_j + c_j| + 2^-143` of `s_j + c_j`, for `Mj ≥ |s_j|` and every lifted
   product of the chunk). With `ρ = 2^-13`, `σ = 2^-20`, `α = 2^-143`: from a partial value within
   `e` of `P_g`, take `Mj = max(|P_g| + e, m_g) ≤ max(|P_g|, m_g) + e` and
   `|s_j + c_j| ≤ |P_{g+1}| + e`: the next one is within
   `step(e) = (1 + ρ + σ)·e + ρ·max(|P_g|, m_g) + σ·|P_{g+1}| + α` of `P_{g+1}` (the recursion of
   `Dot8Chunk.chain_err`, `errB`).
3. **One trip** (line 373, `tl.dot_add`): the combine pass may leave either form.
   (i) `tl.dot(a, b, acc)` continues the accumulator's chain: `q` chunk steps from the error `e`
   against `P`. (ii) `tl.dot(a, b)` and a `float32` addition (`AccAdd`, fusable): the product's own
   chain from `+0`, within `L` of the trip's exact sum (the same steps against the trip's partial
   sums `D_j = P_{tq+j} − P_{tq}`, `DotChunk16Loss.errB`), then `AddRel.err`,
   `3u·(|d| + |acc|) + 4η` with `|d| ≤ |D_q| + L`, `|acc| ≤ |P_{tq}| + e` (`u = 2^-24`,
   `η = 2^-150`). The trip's error `tripErr(e)` is the larger of the two.
4. **The K loop**, `T = cdiv(K, BK)` trips from the exact `+0` of `tl.zeros`:
   `E = accErr(T)`, `accErr(0) = 0`, `accErr(t + 1) = tripErr(accErr(t))`; the last partial sum is
   `P_{Tq} = S`.
5. **The weight** (line 409): one `float32` rounding of `u·w` (`FloatV.mul`), **the store** (line
   413): one `bfloat16` rounding (`FloatV.cvt`); `Post` states both exactly (`outFin`), and the
   room: `|u| ≤ |S| + T`, so with `(|S| + T)·|w| ≤ 2^126` the product rounds to at most `2^126`
   (a `float32` and a `bfloat16` value), and without the weight `|u| ≤ |S| + T ≤ bfloat16.maxVal`.

Room: `Post` asks `dotMag ≤ 2^120`, so every partial value and its bound stays far inside
`float32`, as the chunk fact's room and `AddRel.err`'s ask.

**Relaxation: `E ≤ T`.** Write `A_g = max(|P_g|, |P_{tq}|, m_g)` for chunk `g` of trip `t`
(`pathMag = Σ_{g<G} A_g`), `M_t = Σ_{j<q} A_{tq+j}` (so `Σ_t M_t = pathMag`) and `p_t = P_{tq}`.
- Form (i): each step's added term is `ρ·max(|P_g|, m_g) + σ·|P_{g+1}| + α ≤ (ρ + 2σ)·A_g + α`
  (`|P_{g+1}| ≤ |P_g| + m_g ≤ 2·A_g`), so after `q` steps the error is at most
  `(1 + ρ + σ)^q·(e + (ρ + 2σ)·M_t + q·α)`.
- Form (ii): `D_0 = 0`, so the first step adds `ρ·m_{tq} + …`; for `j ≥ 1`,
  `max(|D_j|, m_j) ≤ |P_{tq+j}| + |p_t| ≤ A_{tq+j} + |p_t|`; and `|D_{j+1}| ≤ 2·A_{tq+j} + |p_t| ≤
  3·A_{tq+j}`. Since every `A_{tq+j} ≥ |p_t|`, `(q − 1)·|p_t| ≤ (q − 1)/q·M_t`, so
  `L ≤ (1 + ρ + σ)^q·((ρ·(2q − 1)/q + 3σ)·M_t + q·α)`; with `|D_q| ≤ Σ_j m_{tq+j} ≤ M_t` and
  `|p_t| ≤ M_t`, the addition adds `3u·(2·M_t + L + e) + 4η`.
- Both forms: with `γ = (1 + 3u)·(1 + ρ + σ)^q`, `tripErr(e) ≤ γ·e + γ·(c_q·M_t + q·α) + 4η`,
  `c_q = ρ·(2q − 1)/q + 3σ + 6u ≤ 1.7763·ρ` (`q ≤ 4`: `(2q − 1)/q ≤ 1.75`, `3σ = 0.0234·ρ`,
  `6u = 0.0029·ρ`).
- Unrolling over `T` trips: `E ≤ γ^T·(c_q·pathMag + G·α + 4T·η)`, and
  `γ^T ≤ (1 + 3u)^512·(1 + ρ + σ)^512 ≤ e^0.0630 ≤ 1.0651` (`T ≤ G ≤ 512`). So
  `E ≤ 1.0651·1.7763·ρ·pathMag + 1.0651·(512·2^-143 + 2048·2^-150) ≤ 1.892·2^-13·pathMag + 2^-133
  ≤ 2^-12·pathMag + 2^-133 = T`.

The weight's product and the `bfloat16` rounding have no tolerance of their own in `Post`: it
states them exactly (`FloatV.mul`, `FloatV.cvt`, up to NaN payload), so the output is pinned by
`u`; to first order the output is within `|w|·T` plus half a `bfloat16` gap (`2^-9` relative) of
`S·w`.

Size: the bound charges `2^-12` of the larger of the running partial sums and each chunk's lifted
magnitude, along the exact partial sums, not `K·Σ|a·b|`. On the GPU check's launches (random data,
`|x| ≈ 0.1`), `T/|S|` against the fine bound `E/|S|` (medians; the `bfloat16` output adds its
rounding, at most `2^-9` relative), and the elements of the check where `T < E` (none: `T/E` lies
in `[1.91, 2.15]` on every element):

| `K` | `E/|S|` median | `T/|S|` median | `(E + (|S| + E)·2^-8)/|S|` median (with the output's rounding) |
|---|---|---|---|
| 64 | 0.00092 | 0.0018 | 0.0048 |
| 128 (test cases) | 0.0013 | 0.0026 | 0.0052 |
| 2048 (Qwen3) | 0.0054 | 0.011 | 0.0093 |
| 7168 (DeepSeek-V3) | 0.0128 | 0.025 | 0.0167 |
| 14336 (Mixtral-8x7B, second GEMM) | 0.023 | 0.045 | 0.027 |
| 16384 (Mixtral-8x22B, second GEMM) | 0.026 | 0.050 | 0.030 |
| 32768 (Grok-1, second GEMM) | 0.048 | 0.094 | 0.052 |

Over every sampled element of the check, `T/|S|` has median `0.013` (`E/|S|`: `0.0065`); its
worst, `3.8·10^5` (`1.9·10^5` for `E`), is at an element whose sum cancels (`|S|` near `0`). The
bound is about `2×` the fine bound at every size.

What `Post` rejects (`check_gpu.py`, negative tests, each on some element of every case that runs
them, from `K = 768` to `K = 32768`; an output is rejected when no value within `T` of `S` rounds
to it): the next expert's weights, a missing or wrong scale (the routing weight, or half the
result), an all-zero output (on almost every element); a result that drops the last `K` block (on
18% of the elements at `K = 32768`, 54% at `16384`, 60–85% at `14336` and `7168`); and a
`bfloat16` accumulator, each trip rounded to `bfloat16` (on 0.0008–5.4% of the elements). One
rejected element rejects the output.

## Exclusions

Nothing `SETUP.md` asks for is left out. Conditions that exclude launches outside SETUP's use:

- **Out of scope: dimensions that are not multiples of `128` or exceed `2^17`, contexts longer than
  `2^22` tokens, and batches of more than `8192` requests: the claim covers real model shapes
  only.** The field is `N` (`N_le17`: `N ≤ 2^17`); `K` is already at most `2^15` (`K_le15`). `N`
  and `K` are not required to be multiples of `128`: one `Inst` covers both GEMMs, and the second
  GEMM's `K` and the first's `N` are the intermediate size per tensor-parallel rank, which vLLM's
  own tuned H200 files take below or off multiples of `128` (`N=96`, `N=320`, `N=64`:
  Qwen3-30B-A3B at TP 8, …). `M` (the tokens for the first GEMM, tokens times `top_k` for the
  second) and `EM` are not bounded by `2^22` for the same reason (`nv_le` bounds `M·top_k`); no
  field counts requests.

- `K > 2^15`: outside real use (the largest `K` of the models above is `32768`); the tolerance's
  relative part grows with `K`. The kernel is believed correct there.
- `stride_be = N·K ≥ 2^31` or `N ≥ 2^31`: outside SETUP (`N·K ≤ 1.2·10^8`); Triton would pass the
  stride as `int64`, a different model, not a kernel fault.
- Non-finite routing weights with `MUL_ROUTED_WEIGHT`: outside SETUP (softmax probabilities).
- The batch-invariant tile `(64, 64, 32, 8)` (`VLLM_BATCH_INVARIANT`), tuned files of a user's
  `VLLM_TUNED_CONFIG_FOLDER` or another device, an override configuration, `int64` `topk_ids` with
  the naive assignment (then `expert_ids` is an
  `int64` tensor), quantised paths, bias, `SWAP_AB`, `USE_TD`: they are not part of the setup
  (`SETUP.md`).

Kernel limits (the kernel is wrong there under the library's semantics; none reached by real use):

- `EM + BM > 2^31`, `N + BN > 2^31`, `K + BK > 2^31`: `tl.cdiv` (`x + (B − 1)` in `int32`, lines
  199, 200, 330) wraps. The first launch that hits it: `sorted_token_ids` of `2^31 − 127` entries
  with the tile `BM = 128` (8 GiB of `int32`, `M·top_k ≈ 2^31`). Real use does not reach it
  (`EM ≤ 4·10^5` in SETUP).
- A non-contiguous `topk_weights` with `MUL_ROUTED_WEIGHT` (`stride(0) ≠ top_k`, e.g. the view
  `w[:, :top_k]` of a wider `[M, 4]` tensor, strides `(4, 1)`, which the host's
  `stride(1) == 1` assertion accepts): the kernel's flat read (line 404) takes the wrong weights
  (those of other token–expert pairs). Excluded by `w_contig`.
  Real use does not reach it: vLLM's routers return a contiguous `[M, top_k]` tensor, as the host's
  test does (`torch.topk`).
- `M·top_k ≥ 2^31`: `num_valid_tokens` no longer an `int32`, and `sorted_token_ids`' `int32` entries
  cannot hold the flat indices. First launch: `M = 2^28`, `top_k = 8`. Not reached.

## Library requests

None.

## Checks

1. **Builds**: `lake env lean ../claims/fused_moe_bf16/Claim.lean` (from `lean/`): no errors or
   warnings, no `sorry`, no `axiom`; `consistent`, `satisfiable` and every
   `setup_*_ok` depend only on `propext`, `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (above).
3. **Coverage**: sixty-four `setup_*_ok` (table above).
4. **GPU** (`check_gpu.py`, H200, Triton 3.8.0; 41 launches of the hosts plus one direct launch,
   at most 3.7 GiB of device memory each, each with the tile the host picks): the nine host test
   cases and their second GEMMs, model-sized cases (Qwen3, DeepSeek-V3 at TP 8 with the naive
   assignment at `M = 1, 8`, Mixtral with the second GEMM at `K = 14336`, Mixtral-8x22B's at
   `16384`, Grok-1's at `32768`, Mixtral at TP 2 and TP 8, Llama-4-Scout at TP 8), the tuned files'
   tiles with `BLOCK_SIZE_K = 256`, `GROUP_SIZE_M = 64`, `BLOCK_SIZE_N = 32` and `256`, edges
   (`K = 32767` with `N = 1000` and the weight, `E = 1`/`top_k = 1` with `GROUP_SIZE_M = 16`,
   `K = 64` with `N = 129`, inputs scaled by 4), the `-1` expert path (vLLM's expert map applied to
   `expert_ids`), and a guarded `C`. Every output is the `bfloat16` rounding of a value within `T`
   of `S` (the check's `err`, the distance from `S` to the values that round to the output, over
   `T`: largest **0.0006**); zeros exact on the `-1` path; unrouted rows unchanged; inputs,
   alignment buffers and the guard bytes around `C` unchanged; the negative tests as in Tolerances.
   PASS.
