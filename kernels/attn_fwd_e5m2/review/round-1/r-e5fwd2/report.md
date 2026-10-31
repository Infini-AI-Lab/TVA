# Review of attn_fwd_e5m2, round 1, by r-e5fwd2

## Verdict

accept-partly
The model, the map, the launch, the conditions and the twelve `[rule]` assumptions are right: the
model does what `_attn_fwd`, `_attn_fwd_inner` and `_maybe_make_tensor_desc` do on every covered
instance by SEMANTICS.md, both descriptor kinds included, and the conditions cover the test's and
the benchmark's shapes and inputs. The `vEff` reading is right too (checked independently): only the
heads with `bh·HEAD_DIM < Z·H` read `v`, the others read the descriptor's zero padding, and Post
pins their output to `±0`. I do not accept `item:spec`: on the test's own data the output tolerance
`tolO` admits an all-zero output on every sampled element at `HEAD_DIM = 128` (88–100% at 64), it is
about `10^4` on the benchmark's data at `HEAD_DIM = 128`, and about `10^6` times the values at the
covered `N_CTX = 2^16`, while the write-up calls it meaningful on the test's shapes and data. Most of
this comes from `Num.DotE5m2`'s constants (a library report), so a plainly stated limitation would
settle the finding.

## Findings

- **SPEC_TOO_WEAK** `item:spec` — `tolO` does not tell `o` from an all-zero output on the test's own data (every sampled element at D=128, 88–100% at D=64), is ~1e4 on benchmark data at D=128 and ~1e6 at N=2^16
- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotE5m2` — `relDot8` charges every fp8 `tl.dot` `2^-7` per chunk of `|acc|` too; fed back once per block it compounds as `(1 + relDot8 64)^T`, so fp8 attention claims get no usable tolerance
- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — say up front (README's opening, kernel.json's title) that the claim certifies output `0` for every head with `bh·HEAD_DIM ≥ Z·H`, all but `⌈Z·H/HEAD_DIM⌉` heads

## Accepted

- `item:model`
- `item:map`
- `item:conditions`
- `item:assumption:faithful`
- `item:assumption:dot`
- `item:assumption:dot_total`
- `item:assumption:cvt`
- `item:assumption:cvt_total`
- `item:assumption:exp2`
- `item:assumption:exp2_ninf`
- `item:assumption:exp2_total`
- `item:assumption:log2`
- `item:assumption:log2_total`
- `item:assumption:div_full`
- `item:assumption:div_total`

## Checked

- The source (`source/06-fused-attention.py`) against SETUP.md: no `@triton.autotune`, the
  pre-hook done by the host (line 557), the config's `BLOCK_M=128, BLOCK_N=64, num_stages=2,
  num_warps=4` at the launch.
- Every model line and every map entry against the source by SEMANTICS.md: the compile-time `if`s
  (`STAGE`, `dtype == tl.float8e5`, `FP8_OUTPUT`, `not IS_HOPPER and …`, `isinstance(…,
  tl.tensor_descriptor)`), `lo, hi = 0, …` as `int32` tensors, the hint of line 59 checked where `lo`
  is defined and line 70's hint on the induction variable dropped, the `float32` constants (the bit
  patterns of `1.44269504`, `-1e6`, `+∞`, `1.0`, `0.5` checked), `tl.where`'s values built in
  `float32`, the fusable forms (`tl.mul_add` at lines 76 and 106, `tl.subc` over the IEEE product at
  line 81, IEEE elsewhere), the mask's signed comparison, the loop's carried names, the two
  `_attn_fwd_inner` calls with `4 - STAGE` and `2`, the epilogue, the unmasked store of `M`, the
  descriptors' shapes, strides, block shapes and padding for both `DescArg` kinds, and the
  equivalence of `Z` or `H` equal to `1` (a compile-time constant) with the `int32` model.
- The drops (imports, decorators, line 70, lines 66–67, 88–94, 99–100, 187, 188, 200–202), the
  supported subset (helpers, defaults, returns, names after the loop, the helper copies' names), and
  the ranges: the kernel lines hold every function the launch runs; `_host_descriptor_pre_hook`,
  `grid`, `alloc_fn` run on the host only; the unused functions and their headers bind nothing the
  kernel reads; nothing rebinds `triton`, `tl` or the two helpers (the module-level `for` loops bind
  `HEAD_DIM`, `mode`, `causal`, `warp_specialize`, which the kernel reads only as parameters).
- The launch: grid `(cdiv(N_CTX, 128), Z·H, 1)`, `program_id` axes, all eight block shapes,
  `fusion`, `debug`, `num_ctas`; `Compiler.Faithful` part (A): no options beyond `num_stages` and
  `num_warps` on the H200, a fresh allocator buffer, no `CompiledKernel`, no `round_f32_to_tf32`,
  `do_bench` without CUDA graphs.
- What the names mean: Model.lean and the root module read in full; Facts, Tile, Inner, Program,
  Real, Lane and RowMax read; the rest in part (Not checked). No `open …
  renaming`, notation, instance or option other than `maxHeartbeats` and a linter; the model printed
  with full names (`work/names.lean`, `work/names2.lean`): every operation is `TVA.*` or a Model.lean
  helper, the `+`/`*` instances are TVA's; `#print axioms` of `main`, `consistent`, `satisfiable`
  (`work/axioms.lean`).
- `vEff`/`vIn` against line 98, the `[HEAD_DIM, Z·H·N_CTX]` descriptor and `tl.desc_load`'s
  per-dimension bounds and zero padding: all or nothing per head, and the in-bounds lanes are `v`'s
  own elements.
- The conditions against the test and the benchmark (shapes, `sm_scale`, layouts, alignment,
  disjointness, both `causal`, both `warp_specialize`), the stated narrowings, satisfiability; the
  score cap `Xh` on the host's inputs by a Chernoff bound (`work/score_cap.py`, `work/score_cap.out`).
- The tolerance: `tolO` and `tolM` evaluated row by row on test-like, benchmark-like and `q = 0,
  v = 1` data up to `N_CTX = 2^16`, against `|attnRef|` and against a floor of the fp8 roundings
  alone (`work/tolerance.py`, `work/tolerance.out`); Post's frame condition.
- Each `[rule]` field against num.md's table for the operation the model uses (`cfg8` written out in
  `dot_total`, `.rn` in `cvt_total`); there is no `[kernel]` field. The README against the Lean.

## Not checked

- Step.lean, and parts of RowAcc.lean, Final.lean and Row.lean, read only for their module
  docstrings and the names of their declarations (the gate's `namespace` check and my grep found no
  instance, notation, renaming or option there; Lean checks the proofs; no declaration there enters
  a claim item's cone).
- Whether a sharper fp8 `tl.dot` fact is true of the device (a library matter, raised as a report).
- The host's environment, part (B) of `Compiler.Faithful`'s scope, by the protocol.
