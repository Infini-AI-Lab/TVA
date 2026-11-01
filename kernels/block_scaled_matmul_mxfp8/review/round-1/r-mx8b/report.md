# Review of block_scaled_matmul_mxfp8, round 1, by r-mx8b

## Verdict

accept-partly
The model, the map, the specification and the three `[rule]` assumptions are right, and I accept them. On every covered instance the model does what lines 174–240 do by SEMANTICS.md. The scales' path is correct: the 5-D `uint8` blocks, their offsets, the reshape, `trans(0, 3, 2, 1, 4)` and reshape give each group of 32 elements the scale that the host's own reference (`unpack_scale`, lines 411–417) and the claim's `sIdx` give it, checked by hand, in Lean (`scaleA_get`, `scaleB_get`, `sIdx_a`, `sIdx_b`) and by a CPU simulation. `Post` determines every element of `c` within a tolerance. At the largest covered `K = 2^14` that tolerance still rejects a misplaced or wrongly scaled group of 32 products, and the README's figures for it match my runs. One blocking finding is on the conditions item. The README counts the benchmark's launches as covered, but the script makes them under proton's launch hook (line 756), which `Compiler.Faithful`'s point 3 excludes. So the certificate covers the test's launch but not the benchmark's, and the write-up must say so.

## Findings

- **CONDITION_WRONG** `item:conditions` — The benchmark's launches run under proton's launch hook (line 756), outside Compiler.Faithful's point 3, yet the README counts them among the covered launches

## Accepted

- `item:model`
- `item:map`
- `item:spec`
- `item:assumption:faithful`
- `item:assumption:dot`
- `item:assumption:dot_total`

## Checked

- Every Lean file of the kernel, in full (`BlockScaledMxfp8.lean`, `Defs.lean`, `Basic.lean`, `Program.lean`). I found no notation, instance, `open … renaming`, `export`, `set_option` or look-alike name. `work/Names.lean` prints the model, `launch`, `cfgMx`, the descriptors, `Assumptions`, `Post`, `OutSpec`, `sIdx` and `InDom` with full names. Every operation is `TVA.*` or a kernel definition I read (`cfgMx`'s omitted fields are `ScaledCfg`'s defaults: `e8m0`, group 32, `K`-packed, `fastMath := false`). `*` and `+` on `I32` are the library's wrapping `IntV` instances. `tl.cvt`'s default rounding is `.rn`. `#print axioms` of `main`, `consistent` and `satisfiable` shows only `propext`, `Classical.choice` and `Quot.sound`. `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)`.
- Each model line and map entry against lines 123–124 and 174–240, by SEMANTICS.md:
  - `tl.cdiv` of constexprs is computed at compile time, and a name assigned it holds an `int32` scalar;
  - `%` and `//` of `int32` scalars are `tl.irem` and `tl.idiv`;
  - the constexpr operands are built as `int32`, and the Python `0`s of the offset lists are `int32` constants;
  - the loop is `tl.range` with `int32` bounds and carries `accumulator`, `offs_k_a`, `offs_k_b` and `offs_scale_k`. `k` is not used after the loop. `num_stages` and `disallow_acc_multi_buffer` change no outcome;
  - each `desc.load` has its host descriptor's block shape;
  - `reshape`, `trans(dims)` and `.T` are `Tensor.reshape`, `permute` and `trans`;
  - `tl.dot_scaled` takes the positional `accumulator` as `acc` and the defaults `fast_math=False`, `k_pack=True`; group `128 / 4 = 32`; storage shapes `[128, 128]` and `[128, 256]`; scales `[128, 4]` and `[256, 4]`;
  - `.to(tl.float16)` is one rounding to nearest even, and the store drops out-of-bounds lanes.
- The drops (decorator, imports, lines 197–202, 216, 229–233) under `output_type = 1` and `ELEM_PER_BYTE_A = ELEM_PER_BYTE_B = 1`. These are the only values the file passes for `"mxfp8"` (lines 247–248, 336–337, 458, 484, 492). `launch_metadata=` is admitted by the supported subset and changes no result without a hook.
- The supported subset. The kernel uses only listed forms: an annotated constexpr `name: tl.constexpr = e`, compile-time `if` chains, `tl.range` with its keywords, the method forms of reshape and trans, and `bool` and `int` constexpr arguments. `triton`, `tl` and `block_scaled_matmul_kernel` are each bound only by their own statement: line 125 binds `proton`, and lines 126–127 and 148 bind other names. Lines 515–521 (the unused AMD kernel's header) bind nothing the launch reads. The kernel calls no function other than `tl.*`.
- The launch:
  - the grid `cdiv(M, 128)·cdiv(N, 256)` on axis 0;
  - every block shape the source uses is listed: `[128, 256]`, `[128, 128]`, `[256, 128]`, the four scale shapes of each operand;
  - fusion, `num_ctas` and `debug` keep their defaults;
  - for part (A): no option but kernel arguments (`NUM_STAGES` is positional and `disallow_acc_multi_buffer` is a parameter keyword), no heuristics or autotune, no in-kernel descriptor, no `CompiledKernel`, no `round_f32_to_tf32`, no graph capture.
- The host's tensors against `Inst`, `aDesc`, `bDesc`, `aScaleDesc`, `bScaleDesc` and `cDesc`: shapes, strides (including the 5-D view of lines 401–402), block shapes (lines 367–368, 399–400, 256) and the default padding.
- `Inst.ok` against the host:
  - multiples of 128, as the scale layout of lines 370–371 and the reference need;
  - `int32` bounds;
  - 16-byte bases;
  - lengths;
  - a fresh `c`;
  - `K ≤ 2^14`, a stated narrowing.

  The test (8192³) and the benchmark's instances meet it.
- The case `N / 128` odd (the satisfiable instance has `N = 128`). The second scale slice and the rows of `b` past `N` read zero padding, and they meet only columns that are never stored. The library's `DotScaledAcc` is per output element.
- `Post`:
  - every element of `c` is determined within an explicit `tol`;
  - `tol` follows the library's per-step bound;
  - the rest of memory is unchanged;
  - the domain (`InDom`, room) holds on all of the host's data, a stated narrowing.

  I evaluated the tolerance at `K = 512`, `8192` and `16384`. It rejects a dropped, repeated or misplaced group, a wrong scale or bias, and another tile's element, and it rejects `bfloat16` output rounding at small `K` and, at `K = 2^14`, on inputs that do not cancel.
- The `[rule]` assumptions:
  - `DotScaledE4m3 P` is the listed fact for `"e4m3"` by `"e4m3"` with `e8m0` scales, group 32, `K`-packed operands and either `fast_math`;
  - `DotScaledTotal P` is cited for the model's configuration;
  - `Faithful P (model d) (launch d) (K d)` is stated under `∀ d, d.ok`.
- The README against the Lean: specification, conditions, proof outline, closed form (`tol_le_closed`, `closed_rel_le`). I also checked its figures by CPU runs, recorded in `work/host_data.py` and `work/host_data.out`, and they agree within sampling. The scale path was simulated in `work/scale_path.py` and `work/scale_path.out`, with no mismatch.

## Not checked

- Whether proton's `deactivate()` removes the launch hooks during the benchmark's warm-up launches (lines 483–485). The finding rests on the timed launches, which run with profiling active, and on the README's own words. Checking it would mean reading proton's source or running it on the GPU, and both are outside a kernel reviewer's scope.
- Part (B) of `Compiler.Faithful`'s scope (environment, caches, target) beyond what the source visibly does, as the protocol says.
- The library's facts, definitions and lemmas, which I trust.
