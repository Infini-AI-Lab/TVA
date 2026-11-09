# `ssd_bmm_chunk_bf16`: the prepared claim

Kernel: `ssd_bmm_chunk_bf16` (SGLang's `_bmm_chunk_fwd_kernel`, step 4 of Mamba-2's chunked scan:
per chunk and group, `C·Bᵀ` over `dstate`, with the pairs of tokens of different requests set to
zero), on `bfloat16` inputs into a `float32` output, with the configuration of `SETUP.md`
(`BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 32`, `num_stages = 4`,
`num_warps = 4`; `IS_CAUSAL = False`, `HAS_SEQ_IDX = True`), launched by the host `bmm_chunk_fwd`
(`source/ssd_bmm.py`, lines 193–282). Claim: [`Claim.lean`](Claim.lean), namespace
`Claims.SsdBmm`; the statement to prove is `Statement : ∀ K P, Assumptions K P → Spec K`.
`Claim.lean` proves `consistent`, `satisfiable`, `tol_rel_lt` and the coverage facts `setup_*_ok`;
it does not prove `Statement`. Model map: [`model-map.json`](model-map.json) (every code line of
the kernel lines `[57, 190]`). GPU check: [`check_gpu.py`](check_gpu.py).

The kernel reads two module-level names, `triton` (its decorator) and `tl`, bound by the imports of
lines 57–58, which are kernel lines and are mapped (as imports that add no step). The other
module-level lines are host lines: `import math`, `import torch` (53–55), the host `bmm_chunk_fwd`,
`make_inputs`, `reference` and the test (193–351).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.SsdBmm.main : Statement` (`Statement`: `Claim.lean`
line 498). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 290), and every initial memory `init` that `Pre` accepts (line 376: the four
buffers valid, every `seq_idx` entry a non-negative `int32`), every outcome of the launch is a
normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 469).

**What `Post` asks** (line 469), two parts:
1. for every `c < nch d`, `g < G`, `m < C`, `n < C`: `OutElem d init fin c g m n` (line 454).
   Element `oIdx d c g m n` (line 286) of `out` reads in `fin` as some `float32` `v`, and:
   - if `sameReq d init c m n` (line 445: tokens `c·C + m` and `c·C + n` are both below `T` and
     have equal `seq_idx` in `init`): for all `va vb : Nat → BF16` such that, for every `k < K`,
     `init` reads `a`'s element `aIdx d (c·C + m) g k` as a finite `va k` and `b`'s element
     `bIdx d (c·C + n) g k` as a finite `vb k`, and `dotMag va vb d.K ≤ 2^120` (line 399): `v` is
     finite and `|v.toRat - dotSum va vb d.K| ≤ tol va vb d.K`. With a non-finite input or no
     room, only "holds a value" is asked;
   - otherwise, on every input: `FloatV.NaNEq (FloatV.ofPat 0) v`, the `float32` `+0`.

   The reference `dotSum` (line 394) is the exact `Σ_{k<K} va k · vb k` in the rationals. `tol`
   (line 427) is `errAt (prodV va vb K) (liftV va vb K) (steps K)`: the recursion `errAt`
   (line 421), `E (t+1) = (1 + rho)·E t + rho·(|partSum p t| + stepMag q t) + alpha`, over
   `steps K = cdiv(K, 32)` (line 383), with `prodV`, `liftV`, `partSum`, `stepMag` (lines
   404–414) and `rho`, `alpha` (lines 388, 391: `DotLoop.stepRel`, `stepAbs` of `relDot16 32`,
   `absDot16 32`);
2. the frame: every byte outside `out`'s first `oB d` elements (line 268) is unchanged.

**What you are given** (`Assumptions`, line 477):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, when the launch is
  valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotBF16`, a `DotLiftAcc`): the `tl.dot` of source line 161 is within
  `relDot16 K · (|acc| + Σ liftMag a · liftMag b) + absDot16 K` of `acc + Σ a·b`, with room
  below the `float32` range;
- `dot_total` (`Rules.Num.DotTotal P cfgBF16`, `cfgBF16` line 36): that `tl.dot` has a result.

**The kernel, as the model runs it** (`bmm_chunk_fwd_kernel`, line 82; `model` line 350, `launch`
line 363): a 3-D grid `(gx d, 1, gz d)` (lines 275, 278). Program `(pid, ·, pid_ch)` owns one
`64 × 128` tile (`pid_m`, `pid_n` from `pid`) of chunk `pid_c = pid_ch / G`, group
`pid_h = pid_ch − pid_c·G`. It runs `tl.forRange 0 (cdiv(K, 32)) 1` of two masked loads (rows and
columns below `chunk_size_limit`, `k` below `K`; `other` the `bfloat16` `+0`) and `tl.dot_add`
into a zero accumulator. After the loop it loads the tile's `seq_idx` rows (`other = -1`) and
columns (`other = -2`), zeroes the pairs that differ (`tl.where`), and stores the tile (masked to
`m, n < C`). The batch strides enter through `batchOff` (line 61), `int32` or `int64` by value
(`intArg`, line 55). Programs write disjoint tiles and use no atomics: the route is
`TVA.meets_independent` (PROVER_GUIDE §4); the loop writes nothing (`safe_forRange` with an
invariant and `fun _ => ∅`).

**What the proof must establish.**
- each launch is valid and has `debug = false`: the grid is 3-D, so `Launch.Valid` is proved
  directly (`x = gx d` is small since `C·C < 2^31` by `oC_lt`; `z` from `gz_lt`, the count `gx d · gz d` from `grid_lt`;
  the six listed block shapes);
- each program is `Safe`: no `int32` or `int64` offset wraps (`T_lt`, `aB_lt` … `oC_lt`, `a_fit`,
  `b_fit`, `aK_step`, `bK_step`, `gz_lt`); every enabled load is in bounds (`a_len`, `b_len`,
  `seq_len`); the store is in bounds (`out_len`) on distinct lanes; `safe_idiv`, `safe_irem`,
  `safe_convert`, `safe_dot_add` (`dot_total`) for the steps;
- footprints: the programs' tiles of `out` are disjoint, and `out` is disjoint from every read
  (`out_a`, `out_b`, `out_s`; `a` and `b` may overlap, both only read);
- the value part of `Post`: the `tl.where` gives `+0` exactly when `sameReq` fails (`Pre`'s
  non-negative `seq_idx`, so `-1`, `-2` never match a token); on `sameReq` the loop is within
  `errAt` of `dotSum`, step by step (`DotLift.dotAdd_step` with `dot`, by induction on the
  recursion; room from `dotMag ≤ 2^120`; README `## Tolerances`);
- the frame: the bytes no program writes.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `tol_rel_lt`
(line 431), `mk_ok` and the `setup_*_ok` theorems; `row_lt` (line 535), `bytes_disjoint`
(line 541), `zero_i32` (line 651) and `valid_zero` (line 658) can be reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance, from every memory `Pre`
allows, every launch ends normally, without fault or hang, with a final memory satisfying `Post`.

`Post d init fin` says, for every chunk `c < nchunks`, group `g < ngroups` and positions
`m, n < chunk_size`, with the tokens `tm = c·chunk_size + m` and `tn = c·chunk_size + n`
(`OutElem`):

- element `(0, c, g, m, n)` of `out` holds a `float32` value `v`;
- **same request** (`sameReq`: `tm < seqlen`, `tn < seqlen` and `seq_idx[0, tm] = seq_idx[0, tn]`):
  when row `tm` of `a` and row `tn` of `b` in group `g` hold finite values `va k`, `vb k`
  (`k < dstate`) and their lifted magnitudes leave room in `float32` (`dotMag ≤ 2^120`), `v` is
  finite and `|v − S| ≤ tol`, where `S = Σ_k va k · vb k` is the exact dot product and `tol` the
  tolerance below, computed from the same values;
- **otherwise** (the tokens belong to different requests, or one of them lies past `seqlen`, in the
  last chunk's padding): `v` is the `float32` `+0.0`, bit for bit, for every input, NaN and
  infinities included;
- **nothing else changes**: every byte outside `out`'s `nchunks·ngroups·chunk_size²` elements is the
  same in `fin` as in `init` (`a`, `b`, `seq_idx`, and the rest of `out`'s storage).

So every element of every `chunk_size × chunk_size` block is pinned: there is no causally masked
part (the launch has `IS_CAUSAL = False`, so every tile is computed and stored), the cross-request
pairs and the padding are exact zeros, and the rest is the product. With a NaN or an infinity in one
of the two rows, a same-request element holds some `float32` value.

**The reference** `S` is the per-chunk `C·Bᵀ` of the host's inputs: the rows of `a` and `b` at the
two token positions `c·chunk_size + m`, `c·chunk_size + n` (element `t·aS + g·aH + k·aK` of `a`'s
storage, `aIdx`; the same for `b`), and the request test reads `seq_idx` at the same two positions,
as the wrapper's docstring states it ("`out[i, j]` for `seq_idx[i] != seq_idx[j]` will be zeroed
out"). Nothing in it is what the kernel reads or how it indexes: not the tiles, the `-1`/`-2`
padding values of the masked `seq_idx` loads, `chunk_size_limit` or the pointer arithmetic. A kernel
that drops the zeroing, uses the wrong chunk's rows, misses a `tl.dot` step or leaves a tile
unwritten fails `Post` (Checks).

**The tolerance** `tol va vb K = errAt (prodV va vb K) (liftV va vb K) (steps K)`: with `T =
steps K = cdiv(K, 32)` loop steps, `p e = va e · vb e` for `e < K` and `0` from `K` on (`prodV`),
`l e = liftMag(va e)·liftMag(vb e)` likewise (`liftV`; `liftMag q` is `|q|` for zero and normal
values and `2^-126` for a nonzero subnormal one), `S_t = Σ_{e < 32t} p e` (`partSum`) and
`L_t = Σ_{32t ≤ e < 32t+32} l e` (`stepMag`):

`E_0 = 0`, `E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`, `tol = E_T`,

with `ρ = DotLoop.stepRel relDot16 32 = 34·2^-22 + 3·2^-24·(1 + 34·2^-22)` and
`α = DotLoop.stepAbs absDot16 32 = 34·2^-126·(1 + 3·2^-24) + 4·2^-150` (Tolerances).

## Conditions

An instance `Inst` is the host's choice: `seqlen` (`T`), `ngroups` (`G`), `dstate` (`K`),
`chunk_size` (`C`), the eight strides of `a` and `b` (`aB, aS, aH, aK`, `bB, bS, bH, bK`, in
elements), and the four buffers `a`, `b` (`bfloat16`), `seq_idx` (`int32`) and `out` (`float32`)
by their storage (`Buf`). `seq_idx` is contiguous (`stride = (seqlen, 1)`), `out` is the host's
fresh contiguous `[1, nchunks, ngroups, C, C]` (strides `oB, oC, oH, C, 1`), `batch = 1`.
`launch` is the grid `(cdiv(C, 64)·cdiv(C, 128), 1, nchunks·ngroups)`, the source's block shapes,
`enable_fp_fusion` at its default.

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `T_pos`, `G_pos`, `K_pos` | `seqlen, ngroups, dstate ≥ 1` | real use; `ngroups` is a divisor (line 117) |
| `C_pow2`, `C_ge` | `chunk_size` a power of two, `≥ 2` | the code base asserts a power of two (`ssd_combined.py:77`); `1` is outside real use (Exclusions) |
| `K_le` | `dstate ≤ 2^10` | keeps the tolerance below `2.7·10^-4` of the products' magnitudes (`tol_rel_lt`), which also bounds the accumulator for the `float32` room condition (Tolerances); every Mamba-2 model's `dstate` is at most `256` |
| `gz_lt` | `nchunks·ngroups < 2^16` | CUDA's grid `z` limit (`Launch.Valid`); a kernel limit (Exclusions) |
| `grid_lt` | `gx·gz = cdiv(C, 64)·cdiv(C, 128)·nchunks·ngroups < 2^31` | fewer than `2^31` programs (`Launch.Valid`); a kernel limit (Exclusions) |
| `T_lt`, `aS_lt` … `aK_lt`, `bS_lt` … `bK_lt`, `oC_lt` | `seqlen`, the strides of `a` and `b` other than `stride(0)`, and `out.stride(1) = ngroups·C²` below `2^31` | each is an `int32` argument in the model (`seq_idx.stride(0) = seqlen`; `oC` bounds `out`'s later strides, `C²` and the store offsets `m·C + n`) |
| `T_le` | `seqlen ≤ 2^22` | `seqlen` is a context length, at most `2^22` tokens: real model shapes (Exclusions) |
| `aB_lt`, `bB_lt` | `a.stride(0)`, `b.stride(0)` below `2^63` | an `int32` argument below `2^31`, an `int64` one from `2^31` (`intArg`); the kernel only multiplies each by `pid_b` (`batchOff`). `out.stride(0) = nchunks·oC` is below `2^47` by `gz_lt` and `oC_lt`, of either class too |
| `a_fit`, `b_fit` | `(C − 1)·aS + 31·aK < 2^31`, `31·bK + (C − 1)·bS < 2^31` | the `int32` lane offsets of lines 143–144 on the lanes the loads enable (rows below `C`, `k`-lanes below 32) do not wrap; disabled lanes may wrap and are not accessed |
| `aK_step`, `bK_step` | `32·aK < 2^31`, `32·bK < 2^31` | the `int32` step of lines 162–163 does not wrap (the pointers themselves are 64-bit) |
| `a_len`, `b_len`, `seq_len`, `out_len` | every element of each tensor lies in its storage | the kernel accesses them |
| `out_a`, `out_b`, `out_s` | `out` shares no byte with `a`, `b`, `seq_idx` | the host allocates `out` (lines 220–228); `a` and `b` may overlap each other in any way (both are only read; the Mamba-2 mixer's `C` and `B` are views of one `xBC` tensor) |

The kernel's own offsets need no other condition at large `seqlen`: the chunk and group offsets
(`pid_c·chunk_size·stride`, `pid_h·stride`, `pid_c·stride_out_chunk`, lines 125–138 and 182–184)
are `int64` (`pid_ch` is sign-extended at line 116), and `pid_b·stride_*_batch` is `0` (`batch = 1`),
an `int32` product sign-extended when the stride is an `int32` argument, an `int64` product when
it is an `int64` one (`batchOff`; the model takes the class Triton gives each batch stride's
value, `intArg`).

`Pre d init`: the four buffers are valid (`Buf.Valid`: aligned, inside `[0, 2^64)`, every byte
present), and every `seq_idx` entry is a non-negative `int32` (real use: request indices
`0, 1, …`; the kernel pads `seq_idx` with `-1` and `-2`, so a negative entry could match the
padding). Any order of the entries is allowed (not only non-decreasing), and any `bfloat16` values
in `a` and `b`.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of `SETUP.md` is a direct JIT launch `_bmm_chunk_fwd_kernel[grid](…)`
(lines 246–281) of the `@triton.jit` function, from `bmm_chunk_fwd`:
- **4 (options).** The launch gives the constexprs `IS_CAUSAL`, `dot_dtype`, `HAS_SEQ_IDX`,
  `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` and two options, `num_stages=4` and `num_warps=4`,
  which do not change the meaning; no `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or other
  option; no `warmup` or `triton.compile`; the kernel has no `@triton.autotune` (removed,
  `SETUP.md`) and no `@triton.heuristics`; the decorator is a bare `@triton.jit`. So `launch` keeps
  every default (`enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, no PDL).
- **5 (descriptor scratch).** The kernel builds no tensor descriptor.
- **6 (precompiled kernel).** No `CompiledKernel` is launched.
- **8 (host descriptors).** No host `TensorDescriptor`: the arguments are four tensors (`seq_idx`
  included), integers and the constexprs.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`, with four pointer parameters, nineteen
unannotated integers and six `tl.constexpr`s, none with a default: `IS_CAUSAL` and `HAS_SEQ_IDX`
(`bool`), the three block sizes (`int`) and `dot_dtype`, whose argument is the dtype `tl.bfloat16`:
it holds that dtype and means at its one use, `.to(dot_dtype)`, what `.to(tl.bfloat16)` means
(SEMANTICS.md, "A dtype as a `tl.constexpr` argument"). Its statements are assignments, `name += e` (lines 125, 130, 136, 161–163,
182), compile-time `if`s on `IS_CAUSAL` (not taken: its body, a runtime `if` with a `return`, is
not visited) and `HAS_SEQ_IDX` (taken),
one `for k in range(0, …)` with two positional arguments (its body's names `a`, `b` and `k` are
not used after it), and the call `tl.store`. Its expressions are `tl.program_id(axis=…)`,
`.to(tl.int64)`, `.to(dot_dtype)`, `.to(out_ptr.dtype.element_ty)`, `tl.cdiv`, `tl.arange`,
`tl.zeros(…, dtype=tl.float32)`, `tl.load(…, mask=…, other=…)`, `tl.dot(a, b)`, `tl.where`,
Python's `min` and `range`, the operators `+ - * // % & < ==`, and the subscripts `[:, None]`,
`[None, :]`.

## Assumptions

Three assumptions, all of kind `[rule]`, each cited as the library states it
(`docs/PROVER_GUIDE.md` section 3, [num.md](../../docs/rules/num.md)):

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`;
- `dot : Rules.Num.DotBF16 P` — `tl.dot` of `bfloat16` operands into `float32` (line 161), any
  `input_precision`, `maxImpreciseAcc = 0` (`Rules.Num.DotLiftAcc`): each output element whose row
  of `a`, column of `b` and accumulator have values is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, `L = |acc| + Σ liftMag a·liftMag b`, when that leaves
  room below the `float32` range;
- `dot_total : Rules.Num.DotTotal P cfgBF16` — the `tl.dot` of line 161 always has a result.

There is no `[kernel]` assumption. The `other=0.0` conversion to `bfloat16` (lines 153, 159) is an
IEEE rounding the library defines (`tl.convert`), exact for `0`; `tl.where` and the stores are
exact.

`consistent` instantiates them at `Platform.sm90ref`.

## Coverage

`SETUP.md` asks for the host's launches with `IS_CAUSAL = False`, `HAS_SEQ_IDX = True`, `bfloat16`
`C`, `B` (contiguous, or views of a packed `xBC` tensor with seqlen stride `conv_dim`), any number of
requests of any lengths, and the code base's sizes. `mk T G K C cd` builds the host's instance
(strides `(T·cd, cd, K, 1)` for `a` and `b`, the buffers at aligned addresses `k·2^36`), and
`mk_ok` reduces `Inst.ok` to numeric bounds, proved by `norm_num`:

| Instance | `seqlen`, `ngroups`, `dstate`, `chunk_size`, seqlen stride | Source |
|---|---|---|
| `setup_single_g3_k5_ok`, `setup_single_g32_k128_ok` | 112, 3 / 32, 5 / 128, 16, contiguous | the single-example test |
| `setup_single_g11_k19_c32_ok` | 128, 11, 19, 32, contiguous | the same test outside CI |
| `setup_inter_g16_k64_c64_ok` | 256, 16, 64, 64, contiguous | the intermediate-states test |
| `setup_cont_c8_g4_k5_ok`, `setup_cont_c256_g13_k32_ok`, `setup_cont_c128_g13_k32_ok` | 64 / 64 / 768, 4 / 13 / 13, 5 / 32 / 32, 8 / 256 / 128 | the continuous-batch test |
| `setup_chunking_c8_ok`, `setup_chunking_c256_ok` | 773, 16, 64, 8 / 256 | the prefill-chunking test |
| `setup_mamba2_g8_c256_ok`, `setup_mamba2_g8_c128_ok` | 32768, 8, 128, 256 / 128, `conv_dim = 10240` | the mixer, 8 groups (Mamba-2, Nemotron-H at TP 1) |
| `setup_g1_k128_ok`, `setup_g1_k256_ok` | 32768, 1, 128 / 256, 256, `conv_dim = 4352 / 4608` | one group per rank (Granite-4.0-H, Falcon-H1) |
| `setup_long_g8_c256_ok`, `setup_longest_g8_c256_ok` | 262,144 / 2,096,896, 8, 128, 256, `conv_dim = 10240` | the mixer on one long prefill (chunked prefill disabled, `python/sglang/srt/managers/scheduler.py:1372`): `a.stride(0)`, `b.stride(0)` `int64` arguments, and at 2,096,896 tokens (8191 chunks, the largest grid `z` below `2^16`) `out.stride(0)` too |

`satisfiable` uses `setup_single_g3_k5` with the all-zero memory. The request pattern is memory
(`seq_idx`), not instance: every pattern of non-negative entries is covered. The SETUP sizes beyond
these (chunks of 8 to 256, up to 8 groups, `dstate` up to 256, prefills of tens of thousands of
tokens) are inside `Inst.ok`: with `ngroups = 8`, chunks of 256 and `conv_dim = 10240` the binding
bound is the grid's `z`, `nchunks·ngroups < 2^16`, up to 2,096,896 tokens. A batch stride
`seqlen·conv_dim` or `nchunks·ngroups·C²` from `2^31` on (from 209,716 tokens at
`conv_dim = 10240`; from 1,048,321 tokens for `out` at 8 groups and chunks of 256) is an `int64`
argument, covered (`intArg`).

## Tolerances

From the library's facts only. Step `t` of the loop (line 161) is `acc ← tl.dot_add(a, b, acc)`, a
`bfloat16` dot of 32 products followed by the addition of `acc`, or, after the combine pass,
`tl.dot(a, b, acc)`. `DotLift.dotAdd_step` (`lean/TVA/Lemmas/DotLift.lean`) turns
`Rules.Num.DotBF16` (`DotLiftAcc` with `relDot16 32 = 34·2^-22`, `absDot16 32 = 34·2^-126`) into a
bound for either form: from a finite `acc`, the result is finite and within
`ρ·(|acc| + L_t) + α` of `acc + c_t`, `c_t = S_{t+1} − S_t` the step's exact sum, with
`ρ = stepRel relDot16 32` and `α = stepAbs absDot16 32` (the separate `float32` addition's room
included). For a same-request element both rows are below `seqlen`, so the loads' masks disable
only the `k`-lanes past `dstate`, which hold `+0`: they add nothing to `c_t` or `L_t` (`prodV`,
`liftV`). If `acc` is within `E_t` of `S_t` then `|acc| ≤ |S_t| + E_t`, so the new accumulator is
within `E_t + ρ·(|S_t| + E_t + L_t) + α = E_{t+1}` of `S_{t+1}`; by induction from `acc = +0`
(`E_0 = 0`) the loop leaves a finite `float32` within `E_T` of `S`. The room condition of
`dotAdd_step` holds at every step: `|S_t| ≤ dotMag ≤ 2^120` and `E_t ≤ 3·10^-4·dotMag + 10^-34`, far
below `2^127`. `tl.where` then keeps `acc` (the two `seq_idx` loads read the two tokens' entries,
which are equal) and the store writes it unchanged (`.to(tl.float32)` of `float32` converts nothing).
For any other element the two loaded `seq_idx` values differ (a real entry `≥ 0` against the
padding `-1` or `-2`, the two paddings against each other, or two different entries), and
`tl.where` gives the constant `+0.0`.

Since `|S_t| + L_t ≤ A = dotMag`, `tol ≤ ((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α`, with
`(1 + ρ)^T − 1 < 2.7·10^-4` for `dstate ≤ 2^10` (`tol_rel_lt`), `8.3·10^-6` at `dstate = 32`,
`3.3·10^-5` at `128` and `6.6·10^-5` at `256`.

**Meaningful.** The bound follows the exact partial sums, so on zero-mean data it is far below
`Σ|a·b|` and far below the outputs: on `randn` inputs its median is `8.2·10^-4` against a median
`|S|` of `7.6` at `dstate = 128` (`1.1·10^-4` of `|S|`; `1.6·10^-4` at `dstate = 256`).
`check_gpu.py` measures the device's error at up to `0.016` of the tolerance (`0` where a single
product is exact, `dstate = 1`): the library's 16-bit dot fact allows `(K + 2)·2^-22` per step of 32
products, about 60 times what the tensor core loses here (Library requests). It still separates
wrong results: the exact product rounded to `bfloat16` breaks it by a factor up to `162`, a missing
`tl.dot` step by `4·10^4`.

## Exclusions

What `SETUP.md` allows and the claim leaves out:

- **`nchunks·ngroups ≥ 2^16`** (`gz_lt`): a limit of the kernel's launch. The grid's `z` is
  `nchunks·ngroups`, and CUDA's limit is `65535`; the launch fails before any program runs. First
  launch that hits it with SGLang's sizes: `ngroups = 8`, chunks of 256, `seqlen = 2,096,897`
  tokens (8192 chunks); with one group, chunks of 256, `16,776,961` tokens. Real use (prefills of
  thousands to tens of thousands of tokens) does not reach it.
- **`cdiv(C, 64)·cdiv(C, 128)·nchunks·ngroups ≥ 2^31`** (`grid_lt`): a kernel limit. A launch of
  `2^31` or more programs is not a valid CUDA launch (Triton's launcher skips it without running a
  program). The host's grid reaches it only with a very large `chunk_size`: the first such launch is
  `chunk_size = 2^15`, `ngroups = 1`, `seqlen = 2^29` (`2^17 × 2^14` programs), whose `out` holds
  `2^14·2^30` `float32` values (64 TiB); since `gx ≈ C²/2^13`, every such launch has an `out` of at
  least about `2^44` elements. Real use (chunks of 8 to 256, so `gx ≤ 8` and fewer than `2^19`
  programs) does not reach it.
- **Strides other than `stride(0)` from `2^31` on** (`aS_lt` … `bK_lt`, `oC_lt`; `T_lt`):
  outside real use. Real use has `a.stride(1) = b.stride(1) = conv_dim` (about 10^4),
  `stride(2) = dstate`, `stride(3) = 1` and `out.stride(1) = ngroups·C² ≤ 2^19`; Triton would
  pass such an argument as `int64`, a specialisation the model does not describe. `T_lt` follows
  from `gz_lt` and `oC_lt` (`seqlen ≤ nchunks·C < 2^16·2^15`).
- **`int32` lane offsets** (`a_fit`, `b_fit`, `aK_step`, `bK_step`): a limit of the kernel. Lines
  143–144 compute `offs_m·stride_seqlen + offs_k·stride_k` and lines 162–163 `32·stride_k` in
  `int32`, which wrap once `(chunk_size − 1)·stride_seqlen + 31·stride_k ≥ 2^31`. Real use has
  `stride_k = 1` and `stride_seqlen = conv_dim` (about 10^4), so the offsets stay below
  `chunk_size·conv_dim`; the first launch through the host that hits it has a token-major `a`
  (`stride(1) = 1`, which the host does not copy) with `stride(-1) = seqlen ≥ 2^26` and
  `dstate > 32`, where the step `32·stride(-1)` wraps (a 4.1 GiB `a` at one group, `dstate = 33`). Not reached by real use.
- **Contexts longer than `2^22` tokens (out of scope).** Dimensions that are not multiples of 128
  or exceed `2^17`, and contexts longer than `2^22` tokens: the claim covers real model shapes only.
  The field is `seqlen ≤ 2^22` (`T_le`). The kernel has no model dimension (`dstate` is a per-head
  state size with real values 16 to 256, `ngroups` a group count, `chunk_size` a block size). Every
  launch of `SETUP.md` meets it (`seqlen ≤ 2,096,896`).
- **`dstate > 2^10`** (`K_le`): outside real use (every Mamba-2 model has `dstate ≤ 256`); the
  tolerance grows with `dstate`, and the bound on `E_t` it gives is part of the `float32` room
  argument (Tolerances). The kernel is believed correct there.
- **`chunk_size = 1`**: outside real use (8 to 256); the argument would be the compile-time
  constant `1`.
- **A negative `seq_idx` entry** (`Pre`): outside real use (request indices). An entry `-1` or
  `-2` could equal the kernel's padding of the masked loads, and the padding of the last chunk would
  then not be zeroed.
- **Values** (a hypothesis of `Post`'s numeric clause only; the zeros, the frame and termination
  hold on every input): rows with a NaN or an infinity, and `dotMag > 2^120`, where the `float32`
  accumulation may overflow (a limit of the kernel's `float32` arithmetic; real use, activations of
  order 1, is far inside).
- **`causal=True`, `seq_idx=None`, 3-D `a`/`b`, `batch > 1`, other dtypes**: not part of the
  setup (`SETUP.md`). Real use never sets `causal` (`ssd_combined.py:165`), so every element of
  every block is computed, stored and pinned by `Post`. With `IS_CAUSAL = True` the programs whose
  tile's columns all come after its rows (`n > m` throughout) return at line 124 and leave their part of `out` as the
  allocation left it (`check_gpu.py`'s negative test).

## Checks

1. **Builds**: `lake env lean ../claims/ssd_bmm_chunk_bf16/Claim.lean`: no errors or warnings, no
   `sorry`, no `axiom`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (the single-example instance,
   the all-zero memory).
3. **Coverage**: 17 `setup_*_ok` (the tests' shapes and the Mamba-2 mixer's, two with `int64`
   batch strides).
4. **GPU** (`check_gpu.py`, H200 NVL, Triton 3.8.0): 38 launches through the host, all holding:
   the tests' 17 patterns in `bfloat16` with `seq_idx`; five mixer launches with `C`, `B` as views
   of a packed `xBC` tensor (`ngroups` 8, 4, 1; `dstate` 128, 256; chunks of 128 and 256; up to
   32768 tokens); edges (`dstate` 1, 33, 1024; chunks of 2 and 2048; one token; 200 one-token
   requests; 2^20 tokens; a grid `z` of 65535; a random non-monotone `seq_idx`; token-major `a`,
   `b`; `int64` batch strides `a.stride(0) = b.stride(0)` of `2^31 + 7` and `262144·10240`; values
   near `2^54`; subnormal operands; NaN and infinities, where the zeros stay exact).
   Largest error / tolerance **0.0160**; every element outside a same-request pair is `+0.0` bit
   for bit; the inputs (and the whole `xBC` storage) unchanged and guard bands of 4096 elements on
   each side of `out` unchanged. Negative tests: the exact product rounded to `bfloat16`, the
   result missing the last `tl.dot` step, a launch without the cross-request zeroing (`seq_idx` all
   zero), and the causal launch (tiles left unwritten) each break `Post`. PASS.

## Library requests

None needed for this claim. The tolerance is about 60 times the device's error here because the claim
cites `DotBF16` (`relDot16 K = (K + 2)·2^-22` per step); the library also states `DotBF16Chunk` (64
consecutive `k` per chunk), with which it could be tighter.
