# Trust inventory of TVA v3

Phase 0 of [PLAN.md](PLAN.md) (Yi Zhou, 2026-11-11). For each part of what a reader of a proved
claim trusts (`TVA/Claim.lean`: the transcription, the compiler with the meaning `TVA.Triton`
gives each operation, the device facts), what validates it today against Triton and the GPU.

Made by reading the files of branch `v3` at `68e4318`; nothing was run. Items marked ✓ were
re-checked by hand against the source; the rest come from a systematic read and should be
spot-checked before anyone relies on one. Citations of Triton's source were checked against the
pinned sources in `primitives/`, which are on `main` and not on `v3`.

Validation sources, in the tables:

| Tag | Meaning |
|---|---|
| **FL** | `evidence/float-layer`: the library's own float definitions, compiled to native code, against the device, bit for bit |
| **E(x)** | an `evidence/x` folder in the v3 tree: a device measurement |
| **OLD(x)** | an evidence folder of commit `5ea2115` (2026-10-29) that is not in the v3 tree |
| **SRC** | a docstring cites Triton, LLVM or PTX source (checked by reading) |
| **CG** | only the claims' `check_gpu.py`: the whole kernel against `Post`, not the definition alone |
| **none** | nothing |

## Summary

1. **The float layer's evidence carries over to v3.** ✓ `float-layer` tested the previous
   library's `TVA/Math/Defs.lean`. With comments removed, 61 of the 63 definitions of v3's
   `Base/Float/{Format,Round,Arith}` are the same text (the other two, `Std` and `F8`, are new
   predicates), and the `FloatV` wrappers, `Literal`, `Int`, `DType`, `Bytes`, `Promote`,
   `Random` and the hints match the old `Lang/*` and `Math/*`. The evidence is valid; its
   README's module names are not.
2. **Evidence was dropped from the tree.** Commit `5ea2115` has fifteen evidence folders that v3
   does not: `literals`, `fp8-constants`, `itof-rtz`, `cdiv-round17`, `promotion`, `permute`,
   `int-reductions`, `python-numbers`, `scalar-args` and six `hint-*`. v3's definitions of
   literals (`litF32` in 31 claims, `hostF32` in 13), integer `//` and `%`, promotion and hints
   rest on them. `float-layer`'s README links to `evidence/literals`, which is gone. Also
   missing and cited: `num-dot-sub16`, `num-dot-int8-round32`, `thread-locality`,
   `docs/rules/num.md`, and the `primitives/` sources that docstrings cite by line.
3. **No device fact is contradicted by its evidence.** Every measured bound is the v3
   constant or tighter. Three soft spots:
   - ✓ `Dot16Chunk` (`dotF16Chunk`, `dotBF16Chunk`) cites `num-dot-chunk16` for its constants,
     but that folder tested an earlier statement (chunks of 16, other constants) which
     `num-dot-chunk16-order` shows false on some layouts. The v3 statement (chunks of 64) is
     tested only on a sample of 32 elements per launch (`num-dot-chunk16-order`, Part M).
   - ✓ `exp2F32` has no slack in its absolute term: the largest error over the bound is
     `1.0000` (`num-math-f32/out.txt:7`), below 1 only through flush to zero.
     `Device.lean:22-24` says facts have "typically a factor of 2" of slack.
   - The six `*Total` facts (each platform relation has a result for every operand) cite no
     evidence, though `Device.lean:26` says each fact cites a folder.
4. **The launch semantics is untested.** Sequential consistency of race-free launches,
   masked-off lanes not accessing memory, duplicate stores, atomics' order (`Step.defer`,
   `MemSem.unused`, `Event.Strong`): PTX reading only.
5. **The most used operations have the least direct evidence.** `tl.load` / `tl.store` masking
   (38 / 35 claims), pointer arithmetic and address validity (all), integer `//` / `%` (27 /
   22, only `OLD(cdiv-round17)`), `forRange` (28, source only), the fusable forms `addc`,
   `subc`, `mul_add` (17 claims, no evidence), Philox (no device test, old or new).
6. **`check_gpu.py` is weaker than `Post` in places.** ✓ Three that are plain bugs or gaps:
   - `softmax_f32`: the 98 benchmark shapes run only through the host's `softmax`, which
     launches through `warmup`, the path the claim excludes since `abfc23d`
     (`check_gpu.py:121-157`). The covered (direct) launch is run only on the test shape
     (`1823 × 781`, outside `Inst.ok`) and on edges.
   - `lora_shrink_bf16`: a NaN output passes: `worst = max(worst, nan)` keeps `worst`
     (`check_gpu.py:232`).
   - `gdn_gating_bf16`: the input-unchanged check is vacuous for finite values
     (`... or torch.equal(t.isnan(), c.isnan())`, `check_gpu.py:188`).
   In many claims the sentinel frame is checked on separate direct launches only, and in
   several (softmax, persistent and TMA matmul, moe_router, vocab_embedding) most "edges" lie
   outside `Inst.ok`. No `check_gpu.py` run is recorded with a date, driver or claim hash.

## 1. Device facts (`Device P`, 38 fields)

All evidence: Triton 3.8.0, torch 2.11.0+cu130, H200 NVL; dates 2026-10-28 to 2026-11-08 (before
v3); only `float-layer` records a driver (610.57.04); none records CUDA or `ptxas`.

"Ratio": largest error / bound.

| Field(s) | Evidence | Measured vs v3 | Coverage |
|---|---|---|---|
| `mathTotal`, `divTotal`, `cvtTotal`, `dotTotal`, `atomicAddTotal`, `dotScaledTotal` | none (indirect: every run produced results) | not measured; `sin`, `cos`, `erf`, non-`f32` math, `f16`/`bf16` division, `e4m3` `.rz` never run | — |
| `expF32`, `exp2F32`, `logF32`, `log2F32`, `sqrtF32`, `rsqrtF32` | E(num-math-f32) | same constants; ratios 0.60, **1.00**, 0.34, 0.45, 0.42, 0.52 | exhaustive, all `2^32` |
| `expF32NegInf`, `exp2F32NegInf` | E(num-math-f32) | match | one input each |
| `expF32Overflow` | E(num-special-f32) | match: cutoff exactly `0x42b17218` | exhaustive from 64 to ∞ |
| `divFull` | E(num-div-f32) | same; ratio 0.65 | adversarial + random, `3·10^10` pairs |
| `divF32InfDivisor` | E(num-special-f32) | match | exhaustive over finite `x` |
| `divF64` | E(num-div-f64) | match, against the old `Format.div` (identical text to v3's) | `4.5·10^8` pairs |
| `cvtSat{F32,F16,BF16}{E4m3,E5m2}` (6) | E(num-cvt8), FL | match, 0 differ | exhaustive |
| `dotF16`, `dotBF16` | E(num-dot16-model), E(num-dot) | ratio 0.11; only `kWidth = 2` layouts measured against the whole-dot bound | adversarial, `K` 16–2048 |
| `dotF32Ieee` | E(num-dot) | ratio 0.50 | `K` 8–1024 |
| `dotI8` | E(num-dot-int8) | exact, 0 differ | to `K = 131040` |
| `dotF16Chunk`, `dotBF16Chunk` | E(num-dot-chunk16-order) Part M | ratio 0.38 / 0.31, **sampled** (32 elements per launch); `num-dot-chunk16` tested a superseded statement | `K` 16–256, `kWidth` 2, 4, 8 |
| `dot{E4m3,E5m2,E4m3E5m2,E5m2E4m3}ChunkLoss` | E(num-dot-chunk-loss), E(num-dot-chunk) | ratios ≤ 0.49 | adversarial, to `K = 16384` |
| `dotScaledE4m3`, `dotScaledE4m3E2m1` | E(num-dot-scaled) | ratios 0.11 / 0.10 | `K` 32–512 only |
| `atomicAddF32` | E(num-atomic-add) | ratio 0.50; every run compiled to `.v4` atomics, the scalar `atom.add.f32` never observed | `2^24` lanes, 12 `sem`/`scope` |
| `sumDepth` | E(reduce-tree) | depth `log₂ n` in all 7960 configurations; the loop rewrite that changes results is excluded only in docs (`EXCLUSIONS.md`) | `n = 2^k ≤ 16384` |

Folders that back no v3 field as stated: `num-dot-chunk16` (superseded statement); `num-dot`'s
fp8 rows (withdrawn facts) and 16-bit rows (older bound).

## 2. The `tl.*` definitions

Kinds: **pure** (Lean computation), **guard** (`pick` of `Unit`: undefined behaviour unless a
condition), **hint**, **partial** (undefined behaviour on some inputs), **free** (a choice not
of the platform: NaN payload, tree, order, padding), **platform** (reads `Env`: a `Platform`
relation or `enable_fp_fusion`), **effect** (load, store, atomic). "Uses": claims whose
`Claim.lean` uses it in code.

### Values (`Base/`)

| Definitions | Kind | Validated by | Uses |
|---|---|---|---|
| float formats, decoding, rounding (`Format`, `Round`) | pure | FL (all `f32` patterns, all smaller formats) | indirect, all float claims |
| IEEE `add sub mul fma div sqrt neg abs`, `maxNum`/`minNum`, comparisons, `cvt` | pure | FL (16-bit formats: all pairs; `f32`/`f64` sampled; `sqrt`: `f32` only) | indirect |
| `cvtSat` into fp8 | pure | FL, E(num-cvt8) | 1 |
| `litF32`, `hostF32`, `litBF16`, `litF16`, `litF64`, `litF8` (Python numbers into constants and arguments) | pure | OLD(literals, fp8-constants, scalar-args) | 31, 13, 2, 0, 0, 0 |
| `leBytes`, `DType.toBits`/`ofBits` (memory encoding) | pure | FL (15 dtypes, store and load) | 10, 2, 14 |
| integer `+ - *`, comparisons, `max`/`min`, `cast` | pure | SRC | ~all, `ltS` 35 |
| integer `div?`, `rem?` | partial | SRC, OLD(cdiv-round17) | via `idiv` 27, `irem` 22 |
| `shl?`, `ashr?`, `lshr?` | partial | SRC | 0 |

### Arithmetic, conversion, promotion (`Triton/Arith`, `Convert`, `Promote`)

| Definitions | Kind | Validated by | Uses |
|---|---|---|---|
| `add`, `sub`, `mul`, `fma`, `div_rn`, `sqrt_rn`, `neg`, `abs` (`upToNaN` of the IEEE result) | free | FL | 9, 14, 24, 0, 0, 0, 1, 0 |
| `addc`, `subc`, `mul_add`, `AddRel` (a product may be fused) | platform (fusion) | **none** | 11, 8, 8 |
| `div` (`/`) | platform | E(num-div-f32, num-div-f64, num-special-f32) | 13 |
| `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt` | platform | E(num-math-f32), `f32` only | 10, 6, 2, 1, 3, 0 |
| `sin`, `cos`, `erf`, `sigmoid` | platform | none | 0 |
| `maximum`, `minimum` (`MaxRel`) | free | FL (looser than measured: either zero allowed) | 5, 0 |
| `idiv`, `irem`, `cdiv` | partial | SRC, OLD(cdiv-round17) | 27, 22, 0 |
| `shl`, `ashr`, `lshr` | partial | SRC | 0 |
| `cvt` (standard formats), `cvtFrom8` | guard + free | FL | 27, 0 |
| `cvt8` (into fp8) | guard + platform | E(num-cvt8), FL | 1 |
| `itof`, `ftoi` | guard + free / partial | FL, OLD(itof-rtz) | 2, 0 |
| `convert` (the default `.to`) | platform (fp8) | parts FL / E; int and float → `int1` none | 30 |
| `bitcast`, `intToPtr` | pure | SRC | 0, 1 |
| dtype promotion (`Promote`) | pure | SRC, OLD(promotion, python-numbers) | used through the models' explicit casts |

### Core, loops, random (`Triton/Core`, `Loop`, `Random`)

| Definitions | Kind | Validated by | Uses |
|---|---|---|---|
| `program_id`, `num_programs`, `arange`, `zeros`, `full`, `where` | pure | SRC | 41, 3, 38, 26, 24, 22 |
| `reshape`, `permute`, `trans`, `join`, `split`, `interleave`, `broadcast_to` | pure | OLD(permute) for `permute`/`trans`; others SRC | ≤ 10 each |
| `reshape_any` (`can_reorder=True`) | free | none | 0 |
| `assume`, `static_assert` | guard | — | 2, 0 |
| hints (`multiple_of`, `max_contiguous`, …) | hint | SRC, OLD(six `hint-*` folders) | 3 |
| `forRange` (`range`, `tl.range`) | guard | SRC | 28 |
| `whileLoop` | loop | none | 1 |
| Philox, `randint`, `rand` | pure / free | SRC; **no device test** (`check_gpu.py` of `seeded_dropout_f32` compares Philox with Random123) | 1 |

### Memory, block pointers, descriptors (`Triton/Memory`, `BlockPtr`, `Desc`, `Memory.lean`)

| Definitions | Kind | Validated by | Uses |
|---|---|---|---|
| `tl.load`, `tl.store`, `load1`, `store1` | effect | FL (byte encoding only); SRC; CG | 38, 35, 19, 5 |
| pointer arithmetic, `Mem.read`/`write`, `Aligned`, `Writable` (what makes an access undefined) | pure | SRC | all |
| `atomic_add` | platform + effect | E(num-atomic-add), `f32` only | 3 |
| `atomic_add_unused` | effect, order made `acq_rel` | none (`MemSem.unused`'s argument) | 1 |
| other atomics (`xchg`, `cas`, `max`, `min`, `and`, `or`, `xor`), `fBound` | effect / free | SRC | 0 |
| `tl.pad` (padding option) | free | none | through descriptors |
| all of `BlockPtr` | guard + effect | SRC | 0 |
| `desc_load`, `desc_store`, `TensorDesc.Valid`/`Access` | guard + effect | SRC; CG | 3 |
| `desc_atomic`, `desc_gather`, `desc_scatter`, `make_tensor_descriptor` | guard + effect | SRC / none | 0 |

### Dot, reductions, scans (`Triton/Dot`, `Reduce`, `Scan`, `Tensor`)

| Definitions | Kind | Validated by | Uses |
|---|---|---|---|
| `tl.dot` | platform | E(num-dot*): the facts that bound `P.dot` | 19 |
| `tl.dot_add` (`dot(a, b) + c`, maybe rewritten) | platform + free | SRC; E(num-dot-int8) for `int8` only | 8 |
| `tl.dot_scaled`, unpacking, `e8m0Value` | guard + platform | E(num-dot-scaled), two configurations | 1 |
| `sum_bcast_mul`, `dot_scaled_add` | free + platform | SRC | 0 |
| `sum`, `sum1`, `sumc`, `sum1c`, `sum0c` | free (tree) + platform | E(reduce-tree) | 10, 4, 1, 5, 1 |
| `max1`, `maxAll`, `max0` (float `tl.max`) | free (tree) | none (FL tests `maximum` elementwise) | 3, 3, 0 |
| `argmaxAll`, `argmax1`, `argmax0` | free | SRC | 1, 0, 0 |
| `histogram` | partial | SRC | 1 |
| `cumsum`, `cumprod`, `associative_scan` | free (tree) | SRC | 1, 0, 0 |
| `sort` | free | SRC | 0 |
| integer reductions (`foldAll`, `foldRows`) | pure | SRC, OLD(int-reductions) | 2, 1 |

## 3. Assumptions of the launch semantics

All are trusted for every claim; none has a device test.

| Assumption | Where | Basis |
|---|---|---|
| A race-free launch behaves as an interleaving of whole steps (each `tl.load` / `tl.store` of a tensor one step) | `LStep`, `InterRace` | argument |
| A program's own conflicting accesses are excluded, so in-place kernels are outside every claim | `IntraRace` | argument |
| A masked-off lane accesses nothing and cannot fault | `Step.load`, `Step.store` | none |
| Duplicate addresses in one store leave one lane's whole element | `Step.store` | none |
| One element's atomic read-modify-write is indivisible | `Step.rmw` | PTX ISA §8.10.3 |
| A program's atomics may be performed later, past steps that do not depend on them | `Step.defer`, `UsesSharedAtomicResult` | reading |
| An atomic with unused result behaves as `acq_rel` at `gpu`/`sys` scope | `MemSem.unused` | PTX ISA §8.4, §8.10.3 |
| `acq_rel` atomics on the same bytes at `gpu`/`sys` scope do not race | `Event.Strong` | reading |
| Atomics create no synchronisation between programs | `HB` | SRC `LoadStoreOpToLLVM.cpp` |
| Program ids, and grid limits (`2^31` to `2^32` programs run nothing) | `Launch.ctx`, `Launch.Valid` | SRC `nvidia-driver.c` |

## 4. The claims

`check_gpu.py` runs the real kernel through its own host code, on the `SETUP.md` launches and
on edges of `Inst.ok`, and checks `Post`. "Neg": tests that the tolerance rejects wrong outputs.
"Exec": how hard an exact comparison of the model with the device would be (phase 4): **easy**
(no platform operation, no free float choice), **exact-inputs** (free reductions or `tl.dot`,
made exact by the inputs), **calibrated** (needs `P_dev`), **hard** (atomics' order with floats,
or several of the above).

| Claim | `Post` | Device facts | Neg | Weaker than `Post` / notes | Exec |
|---|---|---|---|---|---|
| attn_bwd_f16 | tol | dotF16, exp2F32 | no | bench cut to 1 batch, 2 heads; large tests check 4 heads | calibrated |
| attn_bwd_pre_f16 | tol | sumDepth | yes | frame on direct launches only | exact-inputs |
| attn_fwd_f16 | tol | dotF16, exp2F32, log2F32, divFull | no | heads subsampled | hard |
| block_scaled_matmul_mxfp8 | tol | dotScaled* | yes | bench K 10240–14336 and `2^17` at full size not run | exact-inputs |
| chunk_bwd_dv_local_bf16 | tol | dotBF16, exp2F32 | yes | frame on 3 direct launches | calibrated |
| chunk_cumsum_f32 | tol | none | yes | `cu_seqlens` / `chunk_indices` not checked unchanged | exact-inputs |
| chunk_delta_h_bf16 | tol | dotBF16, exp2F32 | yes | the step bound is shown on an instrumented copy of the kernel | calibrated |
| chunk_fwd_o_bf16 | tol | dotBF16, exp2F32 | yes | two `2^20`-token setups not run | calibrated |
| count_expert_tokens_i32 | exact | — | n/a | — | easy |
| extend_attn_bf16 | tol | dotBF16, expF32, divFull | yes | inputs not checked against `Pre`; no guards outside the output | calibrated |
| fused_moe_bf16 | tol | dotBF16Chunk | yes | model sizes cut under 4 GB; inputs fingerprinted, not compared; no NaN/∞ | exact-inputs |
| gdn_gating_bf16 | tol | expF32, logF32, divFull | **no** | ✓ input-unchanged check vacuous (`:188`) | calibrated |
| kv_cache_write_bf16 | exact / tol (fp8) | divFull, cvtSatF32E4m3 | fp8 only | serving shapes cut | easy / calibrated |
| kv_indices_i32 | exact | — | yes | — | easy |
| l2norm_fwd_bf16 | tol | sqrtF32, divFull | yes | the `T ≥ 2^31` case checks 2 programs | calibrated |
| layer_norm_fwd_bf16 | exact / tol | sqrtF32, divFull, sumDepth | yes | frame on a separate direct launch | exact-inputs |
| layer_norm_fwd_f16 | tol | sqrtF32, divFull, sumDepth | yes | ✓ `SETUP.md:42` says every `N` in 2–32768 is covered; `Claim.lean:198` requires `N % 128 = 0` | exact-inputs |
| lora_shrink_bf16 | tol | dotBF16(Chunk), atomicAddF32 | yes | ✓ NaN output passes (`:232`); `rand`/`randn` inputs only | hard |
| matmul_e5m2 | tol | dotE5m2ChunkLoss | yes | NaN/∞ and `M = 0` claimed, not run; frame launches outside `Inst.ok` | calibrated |
| matmul_f16 | tol | dotF16 | yes | — | exact-inputs |
| merge_attn_states_bf16 | tol / exact | expF32, logF32, divFull | yes | no output sentinel | calibrated |
| moe_router_bf16 | tol | expF32, divFull, sumDepth | yes | edges use hidden sizes outside `Inst.ok`; bound skipped for `B < 128` | calibrated |
| moe_sum_reduce_bf16 | tol | none | yes | covered `(65536, 9, 7168)` not run | easy |
| paged_attention_f16 | tol | expF32, divFull, sumDepth | yes | `setup_bigcache` not run; 256-element guard | calibrated |
| persistent_matmul_f16 | tol | dotF16 | yes | most edges outside `Inst.ok` | exact-inputs |
| prefill_attn_varlen_bf16 | tol | dotBF16, expF32, divFull, sumDepth | yes | all-masked programs never run | calibrated |
| recompute_wu_bf16 | tol | dotBF16, exp2F32 | yes | reference rounds twice on subnormal products | calibrated |
| rejection_greedy_i32 | exact | — | yes | — | easy |
| rotary_varlen_bf16 | tol | none | yes | — | easy |
| route_histogram_i32 | exact | — | yes | integer atomics across programs | easy (needs an atomic order) |
| scaled_mm_int8 | tol | dotI8 | yes | one `M` per layer on 8 of 87 shapes; `SETUP.md` and README disagree on the rounding test | exact-inputs |
| seeded_dropout_f32 | exact mask / tol | divFull | yes | `BLOCK_SIZE = 2^20` not run | calibrated |
| softmax_f32 | tol | expF32, divFull, sumDepth | yes | ✓ setup shapes run only through the excluded `warmup` path | calibrated |
| ssd_bmm_chunk_bf16 | tol / exact zeros | dotBF16 | yes | — | exact-inputs |
| ssd_chunk_scan_bf16 | tol | dotBF16, expF32 | yes | tolerance copied by hand into Python | calibrated |
| ssd_chunk_state_bf16 | tol | expF32, dotBF16 | yes | `setup_e7` not run | calibrated |
| ssd_state_passing_f32 | exact / tol | expF32 | yes | — | calibrated |
| streamk_matmul_f16 | tol | dotF16, atomicAddF32 | yes | float atomics across programs; edges outside `Inst.ok` | hard |
| tma_persistent_matmul_f16 | tol | dotF16 | yes | one in-claim direct launch; `f16` overflow edge never reached | exact-inputs |
| vec_add_f32 | exact | — | n/a | — | easy |
| vocab_embedding_bf16 | exact | — | n/a | output frame never checked on a covered launch | easy |

Atomics across programs: `lora_shrink_bf16`, `streamk_matmul_f16` (float), `route_histogram_i32`
(integer). `count_expert_tokens_i32` uses none. TMA descriptors: `attn_fwd_f16`,
`block_scaled_matmul_mxfp8`, `tma_persistent_matmul_f16`. Block pointers: none.

## 5. Gaps, and the phase that addresses each

| Gap | Phase |
|---|---|
| masked loads and stores, address validity, disabled lanes (38 claims) | 2 (sanitizer), 4 |
| integer `//`, `%`, shifts, loops' bounds | 1 |
| fusion (`addc`, `subc`, `mul_add`; 17 claims) | 1, 4 (with fusion on and off) |
| Philox | 1 (bit for bit) |
| float `tl.max` reductions, scans, `argmax`, `histogram` | 1 |
| literals, promotion, hints: evidence only in commit `5ea2115` | restore the folders, or re-run (1) |
| `Dot16Chunk` constants: sampled; `kWidth` 4 and 8 layouts for `dotF16` | rerun of `num-dot-chunk16-order` at full size |
| scalar `atom.add.f32` never observed | 1 |
| `*Total` facts | 1 (results exist for every operand tried) |
| the transcriptions | 4, with mutation checks |
| the launch semantics | 2 (memcheck), 5 (litmus) |
| `check_gpu.py` gaps (section 4) | for zhuominc |

## 6. For zhuominc

Checked, with a fix that is probably small:

1. ✓ `softmax_f32/check_gpu.py` tests the setup shapes only on the `warmup` path, excluded since
   `abfc23d`; direct launches on those shapes are missing.
2. ✓ `lora_shrink_bf16/check_gpu.py:232`: a NaN output passes.
3. ✓ `gdn_gating_bf16/check_gpu.py:188`: the input check is vacuous for finite values.
4. ✓ `layer_norm_fwd_f16`: `SETUP.md:42` contradicts `Inst.ok` (`Claim.lean:198`).
5. ✓ `Triton/Arith.lean:139-141`: the docstring of `fdiv` says the default compiles to
   `tt.precise_divf`; `semantic.py:337-340` uses `create_fdiv` unless `ieee_rounding`. The
   definition follows the source.
6. ✓ `Device/Dot16Chunk.lean` cites `num-dot-chunk16`, which tested the earlier statement.
7. Evidence READMEs name modules of the previous library; fifteen folders of `5ea2115` and the
   `primitives/` sources are not in v3, though v3 relies on and cites them.
8. `check_gpu.py` runs are not recorded (date, driver, claim sha256).

Questions about meaning (not verdicts):

9. `AddRel` (`Arith.lean:67`) was tightened in v3: a fused addition adds at most one unrounded
   product ("sm_90 has no instruction that adds two unrounded products", `:59-60`). Nothing
   measures it.
10. `forRange` (`Loop.lean:41`) requires `0 < step`; Triton 3.8.0 allows a negative constexpr
    step. Such loops are undefined behaviour in the model (incomplete, not unsound).
11. Loop bounds: Triton casts each bound to the induction variable's type with the
    destination's signedness (`code_generator.py:1272-1274`). Can a claim reach a case where a
    model's `tl.convert` (which extends by the source's signedness) differs?
12. `tl.dot_add` allows the rewritten and the plain form for every configuration; its docstring
    says the rewrite happens only when `max_num_imprecise_acc` is 0. Wider than stated: intended?
13. fp8 ↔ `f64` conversion is undefined behaviour in the model (`Convert.lean:67, 132`);
    `semantic.cast` routes it to `create_fp_to_fp`. Was the rejection observed?
14. `tl.maximum(…, propagate_nan=ALL)` has no model, although `FloatV.maxNaN` exists and is
    tested by FL.
15. `atomicAddF32`'s evidence saw only `.v4` atomics; the scalar `atom.global.add.f32` the
    docstring also covers was never observed.
