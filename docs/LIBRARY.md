# Library map

Build: `cd lean && ~/.elan/bin/lake build` (targets `TVA` and one library per kernel folder, `kernels/<kernel>/lean`).

## Language (`TVA/Lang/`)

| Module | Contents |
|---|---|
| `Scalar` | `IntV sg w` (wrapping arithmetic, `toInt`, comparisons, `div?`/`rem?`/`shl?`/`ashr?`/`lshr?`, bitwise); `FloatV f` (IEEE `add sub mul fma divRn sqrtRn`, `max min`, `cvt`, `toRat`, `isNaN`…); `DType` with `Val`, `bytes`, `toBits`/`ofBits` |
| `Tensor` | `Shape`, `Idx`, `Tensor s α` (`get`, `map`, `map2`, `of1`/`of2`, `get1`/`get2`, `col`/`row`, `bcastCols/Rows`, `trans`, `arange`, `where_`), integer tensor arithmetic and comparisons (`lt`, `ltS`, …), the exact reductions `foldAll`/`foldRows`/`foldCols` (with Triton's integer widening in their docstrings), `Tree` (`Covers`, `Eval`, `seq`); any-axis indexing (`Shape.dim`, `Shape.dim_eq_getD`, `Idx.coord`, `Idx.set`, `Idx.ofCoords`, `Idx.snoc`/`unsnoc`, `Idx.pos`/`ofPos`); `join`, `split`, `interleave`, `reshape` (row-major), `permute` (any rank: `get_permute`, `permute_trans`) with their lemmas |
| `Mem` | `Mem := Int → Option Byte`, `Ptr d`, `Mem.read`/`write`/`Writable`/`Aligned`/`span`, `ptr + offsets` |
| `Platform` | `MathFn`, `InputPrecision`, `DotCfg` (with `maxImpreciseAcc`), `MXFmt`, `ScaledCfg`, `Platform` (relations `math div dot cvt atomicAdd dotScaled`), `Env` |
| `Prog` | `Prog` constructors; monad laws (`bind_assoc`, `bind_ret`, …) |
| `Ops` | the `tl.*` API (see [SEMANTICS.md](SEMANTICS.md)); `FloatV.NaNEq`, `FloatV.MaxRel`, `AddRel`, `AccAdd`, `tl.MulRel`; conversions (`tl.cvt` with `RoundOk`, `tl.cvt8`, `tl.cvtFrom8`, `tl.itof` with `ItofRel`); `tl.ashr`/`lshr`; `tl.RangeOk`; `tl.max0`/`maxAll`; index reductions (`CmpKey`, `tl.argCombine`, `tl.argmaxAll`, `tl.argmax1`, `tl.argmax0`: `tl.max`/`tl.min` with `return_indices`, `tl.argmax`/`tl.argmin`); `tl.dot_add`, `tl.sum_bcast_mul` (combine-pass rewrites); `tl.intToPtr` (`int → pointer`); `Padding`, `tl.pad`; hints for any rank (`HintVal`, `Idx.SameGroup`, `…N`); `tl.reshape_any` |
| `Desc` | tensor descriptors (TMA): `TensorDesc` (`Valid`, `Access`, `inbAt`, `elemAt`, `ptrsAt`, `maskAt`, `padding`), `RedOp`; `tl.desc_load`, `desc_store`, `desc_atomic`, `desc_gather`, `desc_scatter` |
| `BlockPtr` | block pointers: `BlockPtr` (`coord`, `elem`, `ptrs`, `inb`, `mask`, `Valid`, `CheckOk`); `tl.make_block_ptr`, `advance`, `load_block`, `store_block` |
| `Random` | `Philox.impl`, `Philox.randint4x`, `tl.randint`, `tl.randint4x`, `tl.uint_to_uniform_float`, `tl.rand` (Philox 4x32-10, exact) |
| `Scan` | `tl.associative_scan` (`ScanLeaves`), `cumsum`/`cumsumc`/`cumprod`, `icumsum`/`icumprod`; `tl.sort` (`SortKey`, `PermAlong`, `SortedAlong`); `tl.histogram` (`enabledBy`) |
| `Scaled` | `tl.dot_scaled`, `tl.dot_scaled_add`: `MXFmt.value`, `e8m0Value`, `ScaleFmt.value`/`Stores`, `scaledValue`, unpacking (`lhsBits`, `rhsBits`, `unpackL/R`, `scalesL/R`), `ScaledCfg.ShapeOk` |

## Semantics (`TVA/Sem/`, `TVA/Claim.lean`, `TVA/Rules/`)

| Module | Contents |
|---|---|
| `Sem.Step` | `Event`, `Out`, `enabled`, `accBytes`, `LoadOk`/`LoadBad`/`StoreOk`, `writeAll`, `Step` |
| `Sem.Launch` | `Kernel`, `Shape.TritonOk`, `Launch` (`grid1`, `ctx`, `size`, `env`, `Valid`, `numCtas`, `pdl`, `blocks`), `Config`, `LStep`, `Reach`, `HB`, `InterRace`, `WellBehaved`, `Outcomes` |
| `Claim` | `Outcome`, `Behaviour`, `Meets`, `Total`, `Consistent` |
| `Rules.Compiler` | `Faithful` ([page](rules/compiler-faithful.md)) |
| `Rules.Num` | cards about `P`: `MathFn.natDom`, `MathAt`, `MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`, `DotScaledTotal`, `MathAcc`, `DivAcc`, `DivFull`, `DivRn` (with `DivRn.divAcc`, `DivFull.of_divRn`), `CvtAcc`, `AtomicAddAcc`, `DotAcc` (with `DType.val?`, `DType.bound`), `DotScaledAcc` ([page](rules/num.md)) |

## Proof kit (`TVA/Lemmas/`)

| Module | Main results |
|---|---|
| `Mem` | read/write algebra: `read_write_self`, `read_write_other`, `write_frame`, `read_congr`, `writeAll_frame`, `writeAll_agree` |
| `Buf` | `Buf d` (`ptr`, `bytes`, `Valid`, `Holds`); `read_some`, `writable`, `span_disjoint`; `writeAll_read` (a store to distinct lanes reads back), `writeAll_present` |
| `Num` | `Format.maxVal`, `u`, `eta`, `Regular` (+ instances for f16/bf16/f32/f64/fp8); `roundRat_rn_err`; `FloatV.add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err`, `cvt_err` (half an ulp, finite, for in-range results) |
| `Tree` | `depth`, `nodes`, `sumOf`, `absSumOf`, `errBound`; `eval_err` (a reduction tree with a per-node bound is within `((1+δ)^d − 1)·Σ|x| + η·N·(1+δ)^d`) |
| `Sum` | `sum_err` (a given tree), `sum_err_covers` (every tree over `0…n−1`) for IEEE-node reductions |
| `Int` | `IntV.SFits`; `toInt_add/sub/mul` without overflow; `toInt_ofInt/ofNat`; `lt_eq`, `le_eq`; `I32.*` |
| `Safe` | `Safe E R W Q m p` (per-program total correctness with footprints); `bind`, `seq`, `mono`, `step`, `not_ub`, `transfer`, `acc` |
| `Ops` | one rule per operation: `safe_load` (`LoadRes`), `safe_store` (`StoreRes`), `safe_store_any`, `safe_add`…`safe_dot`, `safe_dot_add`, `safe_sum_bcast_mul`, `safe_sum`/`sum1`/`sum0`, `safe_sumc`/`sum1c`, `safe_max1`, `safe_max0`, `safe_maxAll`, `safe_argmaxAll`/`argmax1`/`argmax0` (with `tl.argCombine_eq_or`, `tl.argCombine_total`, `tl.argCombine_tieLeft`, `Tree.eval_argCombine`: the result is a leaf and its value, `Tree.eval_argCombine_first`: with `tie_break_left` on integers, the first index of the maximum or minimum; `IntV.cmpKey_lt`, `IntV.cmpKey_eq`), `safe_cvt` (standard formats), `safe_cvtFrom8`, `safe_itof`, `safe_ftoi`, integer `//`/`%`/shifts (`safe_shl`, `safe_ashr`, `safe_lshr`), hints (1-D and `…N`), `safe_reshape_any`, `safe_forN(_bind)`, `safe_forRange`, `safe_while`, `safe_ret`, `safe_last` |
| `Access` | `Buf.elems`, `Buf.img`, `Buf.Addrs` (lane `i` touches element `g i` when `en i`); `addrs_of_get` (from per-lane pointers and mask), `Addrs.accBytes`, `Addrs.distinct`; `safe_load_buf`, `safe_store_buf`; `safe_pad`, `Padding.exists_holds`; `safe_atomic_buf` (a tensor atomic on owned elements); `elems_disjoint(_of)`, `elems_sub_bytes` |
| `Desc` (lemmas) | `TensorDesc.addrs_buf(_all)`, `elemAt_nonneg`; `safe_desc_load`, `safe_desc_store`, `safe_desc_atomic`, `safe_desc_gather`, `safe_desc_scatter`; `RedOp.upd_total_int`, `upd_total_minmax` |
| `BlockPtr` (lemmas) | `BlockPtr.addrs_buf`, `inb_iff`, `coord_toInt`, `elem_eq` (no overflow), `advance_offsets`; `safe_load_block_ptr`, `safe_store_block_ptr` |
| `Scan` (lemmas) | `safe_associative_scan`, `safe_cumsum(With/c)`, `safe_cumprod`, `safe_icumsum` (prefix sums), `safe_icumprod`; `ScanLeaves.perm`/`exists`; `Tree.eval_fun_assoc`, `Tree.eval_add_sum`, `Tree.mapLeaves`; `sum_err_list` (float sum along any leaf list); `SortKey.Lawful` (int and float keys), `sorted_row_exists`, `safe_sort`; `safe_histogram` |
| `Scaled` (lemmas) | `safe_dot_scaled`, `safe_dot_scaled_add` |
| `Layout` | `Idx.pos_snoc`; `Tensor.interleave_eq_reshape_join` (the library's `interleave` is Triton's documented definition) |
| `Block` | the 1-D pattern `offs p B`, `mask p B n`, `Tile`, `Fits` (`p*B + B ≤ 2^31`): `addrs_lane`, `lane_toInt`, `Buf.tile` (+ disjointness), `safe_load_block`, `safe_store_block`, `fits_of_grid` (`n + B ≤ 2^31`), `fits_of_grid_dvd` (`n < 2^31`, `B ∣ 2^31`), `fits_of_grid_pow2`, `tile_div` |
| `Independent` | `independent`: safe programs with non-conflicting footprints ⇒ well behaved, no hang, final memory determined |
| `Shared` | programs sharing a region `A` only through `acq_rel` `gpu`/`sys` atomics on equal-or-disjoint cells: `SafeA` (per program, with a log of `AtomicStep`s; `bind`, `mono`, `of_safe`, `atomic1`), `shared` (well behaved, no hang, `A` = `Replay` of the interleaved logs); integer accumulation: `AddsTo`, `addsTo_add`, `addsTo_other`, `replay_add`, `sum_proj` |
| `Claim` | `meets_of_faithful`, `meets_independent` (both need `hdebug` and `hvalid`); `Launch.size_grid1`, `Launch.valid_grid1` (a 1-D grid of fewer than `2^31` programs with accepted block shapes, and the defaults `num_ctas = 1`, `pdl = false`, is valid) |
| `Consistency` | `Step.progress` (no stuck state), `WellBehaved.outcome`, `modelBehaviour`, `exists_total_faithful`; `Platform.any`; `FloatV.exists_nearest` (half an ulp from any real), `FloatV.exists_near`, `FloatV.exists_near_normal`, `Platform.ref` (`mathAcc`, `mathAcc_of_eta`, `mathAcc_of_normal`, `mathTotal`, `mathAt`, `divRn`, `divTotal`, `divFull`, `cvtAcc`, `cvtTotal`, `atomicAddAcc`, `atomicAddTotal`, `dotTotal`, `dotAcc`, `dotAcc_int`, `dotScaledTotal`, `dotScaledAcc`) |
| `Random` (lemmas) | `safe_uint_to_uniform_float`, `safe_rand` |
| `Fused` | errors of contractible additions: `Format.roundRat_rn_err_of_finite` (a finite rounding is within half an ulp, also above `maxVal`), `Format.magRat_le_maxVal`, `FloatV.abs_toRat_le_maxVal`; `RoundsTo.err`, `RoundsTo.abs_le`; `FloatV.AddRel.err` (`3u·(|x|+|y|) + 4η`), `FloatV.mulAdd_err` |
| `DotLoop` | `acc += tl.dot(a, b)` over `K`: `dotAdd_step` (either form of `tl.dot_add`, from `DotAcc`, with `stepRel`/`stepAbs`), `accum_step`/`accum` (the running bound `bound ρ α t S`), `bound_le` (`≤ 2t·(ρS + α)` for `tρ ≤ 1/2`), `pow_le` |
| `Scalar` | `safe_load1`, `safe_load1_buf` (scalar loads); `IntV.div?_of_nonneg`, `rem?_of_nonneg`; `safe_idiv1`, `safe_irem1` (scalar `//`, `%`); `IntV.I32.add_nat`, `mul_nat`, `sub_nat`, `ofNat_nat` |
| `SoftmaxMax`, `Softmax` | online softmax: `float32` `−∞` and max facts (`maxRel`, subtraction with `−∞`); real lemmas for the rescaled running sums (`2^a` perturbation, rescale-and-accumulate errors, `ratio_err`, `logb_err`) |

## Math (`TVA/Math/`)

Float formats, `roundRat`, binary32 bridges, exactness and saturation
lemmas, and rounding-error bounds (`Format.Error`). `TVA.Lemmas.Num` lifts the bounds to
`FloatV`.

## Kernels (`kernels/<kernel>/`, one folder each; index: [kernels/README.md](../kernels/README.md))

| Kernel | What it exercises |
|---|---|
| (none yet) | the kernels were cleared on 2026-10-27 and are rewritten under the protocol, beginning with `vec_add_f32` |
