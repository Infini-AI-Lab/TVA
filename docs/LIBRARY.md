# A map of `lean/TVA/`

Every module opens with a docstring that says what it defines, how that maps to Triton, and what
to read next; this page is a map of the modules. To find a definition or lemma by name or topic,
use [INDEX.md](INDEX.md) and the generated [index/](index/). The library has two parts:

- **`import TVA`** is the trusted part (`lean/TVA.lean`): everything a claim is stated in, and
  everything a reader of a proved claim trusts. Claims import it.
- **`import TVA.Proof`** is the proof kit (`lean/TVA/Proof.lean`): lemmas for proving
  `claim.Holds`. Nothing in it is trusted; every statement is proved from the definitions in `TVA`.
  Proofs import it.

`TVA.Core` gathers `Base`, `Tensor`, `Memory`, `Platform` and `Program`, the layer below the
`tl.*` vocabulary.

## The trusted part

### Values: `TVA.Base`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Base.Bytes` | `Byte`, `leBytes`, `leValue`: the little-endian encoding of an element in memory. |
| `TVA.Base.Int` | `IntV signed w` (`I8` … `I64`, `U8` … `U64`) with LLVM's meaning: wrapping `+ - *`, truncating `div?`/`rem?` (undefined as `none`), shifts, bitwise operations, comparisons, `cast`. |
| `TVA.Base.Float` | Umbrella of the float model; its docstring gives the reading order. |
| `TVA.Base.Float.Format` | `Format` (`binary16`, `bfloat16`, `binary32`, `binary64`, `e5m2`, `e4m3`, `e2m1`), what a bit pattern means (`toRat`, `isNaN`, `isInf`), the constants `maxVal`, `u`, `eta`. |
| `TVA.Base.Float.Round` | `RoundMode` and `roundRat`, the one rounding every operation and constant uses. |
| `TVA.Base.Float.Arith` | The IEEE operations on patterns: `add`, `mul`, `fma`, `div`, `sqrt`, `maxNum`, `cmp`, `cvt`, with their special cases. |
| `TVA.Base.Float.Value` | `FloatV f` (`F16`, `BF16`, `F32`, `F64`, `F8E5`, `F8E4`) with the operations rounding to nearest even, `ofRat`, `cvt`, `cvtSat`, `toRat`, `toReal`, `NaNEq`. |
| `TVA.Base.Float.Literal` | The constants Triton builds for Python numbers in each dtype (`litF32`, `litF16`, `litBF16`, `litF64`, `litF8`) and the host's conversion `hostF32`. Use when a model writes a Python float. |
| `TVA.Base.DType` | `DType` (Triton's dtypes), `DType.Val` (the Lean type of a value), `bytes`, `toBits`, `ofBits`, `val?`. |

### Tensors, memory, programs

| Module | What it holds; when to use it |
|---|---|
| `TVA.Tensor` | Umbrella of the block values. |
| `TVA.Tensor.Basic` | `Shape`, `Idx`, `Tensor s α`, elementwise maps, rank-1 and rank-2 forms (`get1`, `get2`, `col`, `row`, `trans`), `bc`, `where_`, integer arithmetic and comparisons, float comparisons, the exact folds. Each operation has a `@[simp]` `get_*` lemma. |
| `TVA.Tensor.Index` | Coordinates of an index along any axis (`Idx.coord`, `Idx.set`, `Idx.ofCoords`) and row-major positions; used by scans, permutes and block pointers. |
| `TVA.Tensor.Layout` | `join`, `split`, `interleave`, `reshape`, `permute`, `broadcastTo`, with element lemmas. |
| `TVA.Tensor.Tree` | Reduction trees (`Tree`, `Tree.Covers`, `Tree.Eval`): the order in which a float reduction combines. |
| `TVA.Memory` | `Mem` (a partial map from addresses to bytes), `Ptr d` and pointer arithmetic, aligned reads and writes, `Buf d` (a host buffer: `ptr`, `elems`, `Valid`, `Holds`). Claims state footprints and specifications with `Buf`. |
| `TVA.Platform` | `Platform`: the relations Triton leaves to the device (`math`, `div`, `dot`, `cvt`, `atomicAdd`, `dotScaled`, `sumDepth`), `DotCfg`, `ScaledCfg`, `Env` (the platform with the fusion flag), and `PlatQ` (one question to the platform on concrete operands, with `PlatQ.Rel`: what `Prog.plat` picks among). |
| `TVA.Program` | `Prog α`, one program as a value whose effects are constructors (`load`, `store`, `rmw`, `pick`, `ub`, `loop`, `env`); `Prog.plat` (`env` then `pick`); `Ctx`; `MemSem`, `Scope`, `MemSem.unused`; `Solve`, `Ans`, `Query`, `Hints`, the solvers a `pick` or an `rmw` carries for running a model. Its docstring says why a program is not sequential for global memory. |

### The `tl.*` vocabulary: `TVA.Triton`

A claim's model is written with these; [SEMANTICS.md](SEMANTICS.md) explains each.

| Module | What it holds |
|---|---|
| `TVA.Triton` | Umbrella, with the table of families. |
| `TVA.Triton.Core` | The conventions of a model; `tl.program_id`, `tl.num_programs`, `tl.arange`, `tl.full`, `tl.zeros`, `tl.where`, shape changes, `tl.reshape_any`, compile-time arithmetic (`tl.py`), `tl.assume`, `tl.static_assert`, the compiler hints. |
| `TVA.Triton.Loop` | `tl.forRange` (with `tl.RangeOk`, `tl.tripCount`), `tl.forN`, `tl.whileLoop`. |
| `TVA.Triton.Arith` | Float arithmetic (`tl.add`, `tl.mul_add`, `tl.addc`, `tl.div`, `tl.div_rn`, `tl.math`, `tl.exp`, `tl.maximum`, `tl.sigmoid`, …), `AddRel` and the fusion rule; integer `tl.idiv`, `tl.irem`, shifts, `tl.cdiv`. |
| `TVA.Triton.Convert` | `tl.convert` (`ConvRel`), `tl.cvt`, `tl.cvt8`, `tl.cvtFrom8`, `tl.itof`, `tl.ftoi`, `tl.bitcast`, `tl.intToPtr`. |
| `TVA.Triton.Promote` | Triton's dtype promotion (`computationType?`, `BinOp.typing?`, `whereTyping?`, …) and the dtype and constant of a Python number (`Operand.ownType?`, `Operand.constIn?`). A prover evaluates them with `decide`. |
| `TVA.Triton.Memory` | `tl.load`, `tl.store`, the scalar forms, `tl.addrs`, `Buf.Addrs`, padding, and the atomics with their `_unused` forms. |
| `TVA.Triton.BlockPtr` | `BlockPtr`, `tl.make_block_ptr`, `tl.advance`, `tl.load_block`, `tl.store_block`. |
| `TVA.Triton.Desc` | `TensorDesc` and its requirements (`TensorDesc.Access`), `tl.desc_load`, `tl.desc_store`, `tl.desc_atomic`, `tl.desc_gather`, `tl.desc_scatter`. |
| `TVA.Triton.Reduce` | Float sums along trees (`tl.sum`, `tl.sumc` and the axis forms, `tl.sumBound`), `tl.max`, index reductions (`tl.argmaxAll`, `tl.ArgRes`), `tl.histogram`. |
| `TVA.Triton.Scan` | `tl.associative_scan` (`ScanLeaves`), `tl.cumsum`, `tl.cumsumc`, `tl.cumprod`, the integer scans, `tl.sort`. |
| `TVA.Triton.Dot` | `tl.dot`, `tl.dot_add`, `tl.sum_bcast_mul` (the combine pass's rewrites), `tl.dot_scaled` and its unpacking (`ScaledCfg.lhsBits`, `scaledValue`). |
| `TVA.Triton.Random` | Philox (`Philox.impl`, `Philox.randint4x`), `tl.randint`, `tl.rand`. |

### Launches: `TVA.Launch`

| Module | What it holds |
|---|---|
| `TVA.Launch` | Umbrella; its docstring is the semantics of a launch in prose: how it runs, what makes it well behaved and why, what `Correct` states. |
| `TVA.Launch.Step` | `Step`: one step of one program, its `Event` (bytes read and written), undefined behaviour, `Step.defer` for atomics performed later. |
| `TVA.Launch.Run` | `Kernel`, `Launch` (grid, options, block shapes), `Launch.grid1`, `Launch.ctx`, `Launch.Valid`, `Config`, `LStep`, `Reach` (interleaving). |
| `TVA.Launch.WellBehaved` | `InterRace`, `IntraRace`, `UsesSharedAtomicResult`, `WellBehaved`, `Outcomes`, `Correct`. |
| `TVA.Launch.Progress` | `Step.progress`, `WellBehaved.exists_outcome`: a well-behaved launch always has an outcome. |

### Device facts: `TVA.Device`

| Module | What it holds |
|---|---|
| `TVA.Device` | `Device P`, the conjunction of every device fact; the facts implied by others (`Device.cvtF32E4m3`, `Device.dotE4m3Chunk`, …); `Device.ref_ok`. |
| `TVA.Device.Shapes` | The shapes a fact takes: totality (`MathTotal`, …), accuracy (`MathAcc`, `DivAcc`, `DotAcc`, `DotLiftAcc`, `DotScaledAcc`, `AtomicAddAcc`), exact results (`MathAt`, `DivRn`, `CvtRn`, `CvtSat`); `liftMag`. |
| `TVA.Device.Sm90` | The H200 facts with their constants and evidence: `ExpF32`, `DivFull`, `CvtSatF32E4m3`, `DotF16`, `DotI8`, `AtomicAddF32`, `SumDepth`, …; `relDot16`, `absDot16`. |
| `TVA.Device.Dot8Chunk` | The fp8 `tl.dot` facts, chunk by chunk (`DotE4m3ChunkLoss`, …; shapes `DotChunkAcc`, `DotChunkLoss`). |
| `TVA.Device.Dot16Chunk` | The 16-bit `tl.dot` facts, chunk by chunk (`DotF16Chunk`, `DotBF16Chunk`; shape `DotChunk16Loss`). |
| `TVA.Device.Tolerance` | Tolerance formulas specifications name: `DotLoop.bound`, `DotLoop.stepRel`, `DotLoop.stepAbs`, `Dot8Chunk.psum`, `Dot8Chunk.errB`, `Dot8Chunk.errM`. Arithmetic only. |
| `TVA.Device.Weaken` | Proved: weaker forms of a fact (`MathAcc.mono`, `DotAcc.mono`, `CvtSat.cvtRn`, `DotAcc.liftAcc`, …). Use when a proof needs larger constants or a smaller domain. |
| `TVA.Device.Ref.Platform` | Proved: `Platform.ref`, IEEE arithmetic rounded once, and its general properties. |
| `TVA.Device.Ref.Ops` | Proved: `Platform.ref` meets each shape with one rounding's constants. |
| `TVA.Device.Ref.Chunk` | Proved: `Platform.ref` meets the chunk shapes. |
| `TVA.Device.Ref.Sm90` | Proved: `Platform.ref` meets each fact; `TVA.Device` assembles them into `Device.ref_ok`. |

### Claims

| Module | What it holds |
|---|---|
| `TVA.Claim` | `TVA.Claim` (`Inst`, `ok`, `kernel`, `launch`, `Pre`, `Post`), `Claim.Holds`, `Claim.Satisfiable`; its docstring lists everything a reader of a proved claim trusts. |

## The proof kit: `TVA.Proof`

[PROVING.md](PROVING.md) says how the parts fit together.

### The core: `TVA.Proof.Basic`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Basic` | Umbrella of the core, with the three-step route (one program at a time, footprints, the endpoint) and a table of files. |
| `TVA.Proof.Mem` | Locality of reads and writes (`Mem.read_congr`, `Mem.write_frame`), read after write, presence, `DistinctLanes`, `writeAll_read`. Use to move facts between memories. |
| `TVA.Proof.Buf` | Valid buffers can be read and written (`Buf.read_some`, `Buf.writable`); `Buf.Addrs.accBytes`, `Buf.Addrs.distinct`. |
| `TVA.Proof.Step` | `Step.frame`, `Step.transfer`, `Mem.mix`: one step in two memories (used by the composition). |
| `TVA.Proof.Safe` | `Safe` (one program run alone, with footprints), `Safe.bind`, `Safe.seq`, `Safe.bind_ro`, `Safe.mono`; `Safe.stepT`, `Safe.no_ub`, `Safe.acc` for the composition. |
| `TVA.Proof.Rules.Core` | `safe_ret`, `safe_last`, `safe_assume`, `safe_static_assert`, the hint rules, `safe_reshape_any`. |
| `TVA.Proof.Rules.Memory` | Loads and stores (`safe_load`, `safe_store`, `safe_store_any`, `safe_load_buf`, `safe_store_buf`, scalar forms), padding (`safe_pad`), atomics on owned elements (`safe_atomic_buf`). |
| `TVA.Proof.Rules.Arith` | Float operations (`safe_add`, …, `safe_mul_add`, `safe_addc`), platform `safe_div`, `safe_fdiv`, `safe_math`, `safe_maximum`, integer `safe_idiv`, `safe_irem`, shifts. |
| `TVA.Proof.Rules.Convert` | `safe_convert`, `safe_cvt`, `safe_cvt8`, `safe_cvtFrom8`, `safe_itof`, `safe_ftoi`. |
| `TVA.Proof.Rules.Reduce` | Sums along trees of bounded depth (`safe_sum`, `safe_sumc`, axis forms), `safe_max1`, `safe_maxAll`, `safe_max0`; `Tree.bal` shows a tree exists. |
| `TVA.Proof.Rules.ArgReduce` | `safe_argmaxAll`, `safe_argmax1`, `safe_argmax0`, and what `tl.ArgRes` allows (`tl.ArgRes.index_lt`, `tl.ArgRes.value`, `Tree.eval_argCombine_first`). |
| `TVA.Proof.Rules.Dot` | `safe_dot`, `safe_dot_add`, `safe_sum_bcast_mul`, `safe_dot_scaled`, `safe_dot_scaled_add`. |
| `TVA.Proof.Rules.Loop` | `safe_forN`, `safe_forN_bind`, `safe_forRange`, `safe_while`, with per-trip write footprints (`iterW`). |
| `TVA.Proof.Compose` | `independent`: a launch of `Safe` programs with disjoint footprints is well behaved and ends as each program says. |
| `TVA.Proof.Correct` | `correct_of_safe`, the endpoint of almost every proof (its docstring shows the skeleton of `main`); `Launch.valid_grid1`. |

### Memory access patterns: `TVA.Proof.Pattern`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Pattern` | Umbrella, with one line per pattern. |
| `TVA.Proof.Pattern.Block` | 1-D blocks `pid * B + tl.arange(0, B)`, `mask = offs < n`: `Block.Fits` (`fits_of_grid_pow2`), `safe_load_block`, `safe_store_block`, footprint `Buf.tile`, `Buf.tile_disjoint`. |
| `TVA.Proof.Pattern.TileOffset` | 2-D offsets `rows[:, None] * s0 + cols[None, :] * s1` in `int32` or `int64`: lane values (`off_toInt`, the `…_on` forms for enabled lanes only), pointers (`ptr_lane`, `ptr_lane_on`), rows wrapped by `% M`. |
| `TVA.Proof.Pattern.Tile` | Masked 2-D tiles: `Tile2.MaskAt`, `safe_load_tileAt`, `safe_store_tileAt`, footprints `Buf.tileAt`, `Buf.tile2`, `Buf.matrix`, non-aliasing layouts (`Tile2.Inj`), disjoint tiles. |
| `TVA.Proof.Pattern.BlockPtr` | `safe_load_block_ptr`, `safe_store_block_ptr`; `BlockPtr.elem_eq`, `BlockPtr.inb_iff`, `BlockPtr.advance_offsets`. |
| `TVA.Proof.Pattern.Desc` | `safe_desc_load`, `safe_desc_store`, `safe_desc_atomic`, `safe_desc_gather`, `safe_desc_scatter`, `safe_make_tensor_descriptor`. |

### Integers: `TVA.Proof.Int`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Int` | Umbrella; its docstring states the conventions (values by `toInt`, `SFits`, tracked values, `omega`). |
| `TVA.Proof.Int.Basic` | `SFits`, the value of constants (`toInt_ofNat`, `toInt_ofNat_lit`), equality and comparisons (`lt_eq`, `le_eq`, `eq_eq`, `toInt_max`). |
| `TVA.Proof.Int.Arith` | `+ - *` without wrapping (`I32.add_nat`, `I32.mul_nat`, …), `//` and `%` of nonnegative values. |
| `TVA.Proof.Int.Cast` | `IntV.cast` keeps values in range; `tl.convert` between integer dtypes is the cast (`ConvRel.iff_cast_<d>_<d'>`). |
| `TVA.Proof.Int.Lanes` | Lanes of integer vectors (`Tensor.get1_add`, …), `tl.arange` (`I32.toInt_sadd_arange`), `idiv_ok`, `irem_ok`. |

### Floats: `TVA.Proof.Float`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Float` | Umbrella; its docstring explains the idea of every bound (one rounding: `|q|·u + η`) and the file order. |
| `TVA.Proof.Float.Encoding` | What a pattern means: sign, magnitude, order keys, `maxVal`; every Triton format is `Regular`. |
| `TVA.Proof.Float.Constants` | `u`, `η`, `maxVal` of each format, in `ℚ` and `ℝ` (`binary32_u`, …); infinities and NaN. |
| `TVA.Proof.Float.Quantum` | The steps of `roundRat` (quantum, significand, packing), proved correct. |
| `TVA.Proof.Float.Round` | The error of one rounding: `roundRat_rn_err`, `roundRat_rz_err`, `roundRat_rn_err_of_finite`. |
| `TVA.Proof.Float.Ops` | The pattern operations on finite operands; negation; comparisons; exact widening (`Wider`). |
| `TVA.Proof.Float.Sqrt` | The rounded square root is IEEE rounding (`sqrt_rn_err`). |
| `TVA.Proof.Float.Value` | `FloatV` and its pattern, `toReal`, `NaNEq` (`NaNEq.exact_of_isFinite`). |
| `TVA.Proof.Float.Error` | The bounds a kernel proof uses: `add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err`, `sqrtRn_err`, `cvt_err`, `cvtSat_rn_err`, and the `_real` forms. |
| `TVA.Proof.Float.Max` | `tl.maximum`, `tl.max` reductions, maxima of approximations (`FloatV.max_approx`, `Tree.eval_maxRel_approx`), `−∞`. |
| `TVA.Proof.Float.Facts` | Zeros, bounds of a value, finiteness from magnitude, `DType.val?` of float elements. |
| `TVA.Proof.Float.Widen` | Exact widening conversions (`cvt_f16_f32_exact`, …) and what `x.to(…)` may return (`FloatRel`, `ConvRel`). |
| `TVA.Proof.Float.Tree` | The error of a reduction tree from a per-node bound (`Tree.errBound`, `eval_err_abs`, `eval_err`). |
| `TVA.Proof.Float.Fused` | Fusable additions: `FloatV.AddRel.err`, `FloatV.mulAdd_err`, `RoundsTo.err`. |
| `TVA.Proof.Float.Sum` | Float reductions along any tree: `sum_err_depth`, `sum_err_covers`, `sumc_err_depth`, `sumc_err_list`. |
| `TVA.Proof.Float.SumBcast` | Rounded `float32` products summed along a tree (`sum_bcast_mul_depth_err`). |

### `tl.dot`: `TVA.Proof.Dot`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Dot` | Umbrella, with the route of a GEMM proof. |
| `TVA.Proof.Dot.Loop` | `acc += tl.dot(a, b)` over `K` blocks: `DotLoop.dotAdd_step` (one trip), `DotLoop.accum` (the loop against `DotLoop.bound`), `DotLoop.bound_le`. |
| `TVA.Proof.Dot.Lift` | The same trip from a fact relative to lifted magnitudes (`DotLift.dotAdd_step`, for `DotF16`, `DotBF16`); lifted against plain magnitudes. |
| `TVA.Proof.Dot.Chunk` | The arithmetic of a dot added chunk by chunk (`Dot8Chunk.chain_err`, `Dot8Chunk.chain_errM`). |
| `TVA.Proof.Dot.Chunk8` | An fp8 dot, chunks of 32: `Device.DotF8ChunkLoss.err`, `Device.DotF8Chunk.liftAcc`. |
| `TVA.Proof.Dot.Chunk16` | A 16-bit dot, chunks of 64: `Device.Dot16BitChunk.err`, `Device.Dot16BitChunk.liftAcc`. |
| `TVA.Proof.Dot.Scaled` | `acc += tl.dot_scaled(…)` (`Device.DotScaledAcc.addStep`) and the exact scaled values. |
| `TVA.Proof.Dot.Sm90` | The facts in the shapes a proof wants, from `h : Device P` (`h.dotE5m2Lift`, `h.dotF16Acc`, `h.dotI8_eq`), and `sum_bcast_mul_err`. |

### Shared atomics: `TVA.Proof.Shared`

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Shared` | Umbrella, with the route for programs that meet through atomics on shared cells (split-K, stream-K, histograms). |
| `TVA.Proof.Shared.Log` | `AtomicStep`, `Replay`, `proj`, `cellBytes`, `PermInv`: the record of a program's shared atomic steps. |
| `TVA.Proof.Shared.SafeA` | `SafeA`: `Safe` with atomics on shared cells, logged; `SafeA.bind`, `SafeA.mono`, `SafeA.of_safe`, `SafeA.perm`. |
| `TVA.Proof.Shared.Acc` | `SafeA.acc`: a `SafeA` program ends whatever the others write. |
| `TVA.Proof.Shared.Rules` | `SafeA.atomic_add`, `SafeA.atomic_add_unused`, `SafeA.atomic`, `SafeA.atomic1`. |
| `TVA.Proof.Shared.Launch` | The launch invariant `InvA`, `no_interRaceA`. |
| `TVA.Proof.Shared.Correct` | `shared` and `correct_of_safeA`, the endpoint for such kernels. |
| `TVA.Proof.Shared.Replay` | What a replay leaves in a cell: `replay_add` with `sum_proj`, `Paired`, `replay_fold`, `replay_max`, `replay_min`. |
| `TVA.Proof.Shared.FloatAcc` | Float `atomic_add` from several programs: `replay_err`, `replay_err_shared`, `accStep_atomicAdd`. |

### Scans, random numbers, softmax, recurrences

| Module | What it holds; when to use it |
|---|---|
| `TVA.Proof.Scan` | `safe_associative_scan`, `safe_cumsum`, `safe_cumsumc`, `safe_cumprod`, `safe_icumsum` (an integer cumulative sum is the prefix sum), `safe_icumprod`; `sum_err_list`; `safe_sort`; `safe_histogram`. |
| `TVA.Proof.Random` | `safe_rand`, `safe_uint_to_uniform_float`; `tl.itofRel_binary32` (the conversion of a Philox integer is one rounding). |
| `TVA.Proof.Softmax` | Umbrella of the online softmax (mathematics over `ℝ`, namespace `TVA.Softmax`). |
| `TVA.Proof.Softmax.Exp` | Bounds on `exp` of a rounded argument (`abs_exp_round_le`); `exp2` as `exp` (`two_rpow_eq_exp`). |
| `TVA.Proof.Softmax.Exact` | `softmax`, `wavg`, the exact invariant of the running maximum, sum and accumulator (`l_eq`, `acc_eq`, `ratio_eq`), keys in blocks (`wavg_blocks`). |
| `TVA.Proof.Softmax.Error` | The same with rounding: `accum`, `l_err`, `acc_err`, one trip (`step_err`, `l_trip`), the output (`err_final_mono`). |
| `TVA.Proof.Recurrence` | A state carried across trips with a decay, `S (t+1) = g t · S t + U t`: closed forms (`closed_cumsum`, `closed_cumsum_mat`), error bounds (`perturb_le_sum`, `perturb_mat_le_sum`). Namespace `TVA.Recurrence`. |
