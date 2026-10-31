# Review of attn_bwd_f16, round 1, by r-bwd2

## Verdict

accept
The model follows `_attn_bwd` and its two helpers line by line, by SEMANTICS.md, on every covered
instance (the compile-time `if`s, the names that hold `int32` tensors, the `tl.subc` after each
`tl.dot`, the four `tl.dot_add`s, the `+0.0` of `tl.where`, `LN2` built in `float32`, the loops'
carried names, the helper copies), and the map's drops are sound. The conditions cover the test's
and the benchmark's shapes and values; each narrowing has its reason. `Post` states each element
of `dv`, `dk`, `dq` against the exact sums with an explicit tolerance, about 0.8% of the magnitude
sum at the benchmark's largest size on its inputs, and the frame. Its value premise `HeadV.Ok`
holds with wide margin on what the host passes (the forward's `M`, `_attn_bwd_preprocess`'s
`delta`, normal inputs). The five `[rule]` fields are the library's facts for the operations the
model uses. I raise two wording findings on the README, neither blocking.

## Findings

- **DOC** `doc:kernels/attn_bwd_f16/README.md` — the summary "within about 0.5% … at every covered size" drops the input-dependent per-pair exponent term (about 0.8% in all at the benchmark's HEAD_DIM = 128, N_CTX = 2^14)
- **DOC** `doc:kernels/attn_bwd_f16/README.md` — Narrowings: "the host asserts only that the strides are equal" omits `do.is_contiguous()`; a Python-float `sm_scale` is a narrowing the list does not state

## Accepted

All items.

## Checked

- Every Lean file of the kernel, in full (Model, Spec, Lane, Acc, Basic, Step, DkDv, Dq, Out, Program, AttnBwd): no instance, notation, attribute, export, `set_option` or `open … renaming`; only `open TVA` (and `open TVA.Math.Float` in the proof files Lane and Acc). The model, `launch`, `Inst.ok`, `Pre`, `Post`, `OutSpec`, `HeadV.Ok` and `Assumptions` printed with `pp.fullNames` (work/print_names.lean): every operation is `TVA.*` or a kernel definition I read; the `DotCfg` defaults are `tf32` and `0`; the axioms of `main`, `consistent`, `satisfiable` are `propext`, `Classical.choice`, `Quot.sound`.
- The model against lines 266–500, entry by entry of the map: argument order of all four helper calls (positional, `MASK=` by keyword); the loops' carried tuples (`dk, dv, offs_m, curr_m, qT_ptrs, do_ptrs`; `dq, offs_n, curr_n, kT_ptrs, vT_ptrs`, `offs_n` carried unchanged when not `MASK`); `row`/`col`/`bcastCols`/`unsqueeze1` against `[None, :]`, `[:, None]` (library `Tensor.row`, `Tensor.col`); the masks as signed `int32` `le` with the operands swapped; `tl.subc` for `qkT - m[None, :]`, `dpT - Di`, `qk - m`, `dp - Di[:, None]` (a `tl.dot` feeds each); `tl.dot_add` for the four `+=` of a `tl.dot` with the default zero accumulator; `.to(tl.float32)` of a `float32` dot as `tl.convert` (identity by `FloatRel`); the stores' `float16` conversions; `dk *= sm_scale` and `dq *= LN2` as IEEE products; `LN2`'s double `0x3FE62E4300000000` and its `float32` `0x3f317218` (python); the names assigned Python numbers (`start_m = 0`, `step_m`, `num_steps = 8`) as `int32` tensors and the annotated `LN2`, `MASK_BLOCK_M1`, `MASK_BLOCK_N2` as compile-time values; `stride_d`, and `H` when `1`, as the compile-time `1` built in `int32` at every use; the two copies of each helper have different names (`16`/`True`, `32`/`False`); the `tl.static_assert`s hold.
- The phases' row coverage: `dkdv` rows `[start_n, start_n + 128)` masked then `[start_n + 128, N_CTX)` (causal) or `[0, N_CTX)`; `dq` keys `[128p, 128p + 128)` masked then `[0, 128p)` (causal) or `[0, N_CTX)`; the rows left out are not `Allowed`; `stepsKV`, `stepsQ` match the trip counts; every pointer stays inside its tensor and no `int32` product wraps under `size_lt`, `n_le`, `g_lt`.
- The supported subset: every form `_attn_bwd` and the helpers use is listed; the module-level names read are `triton`, `tl` (lines 20–21) and the two helpers, bound only by their own `import`/`def`; the module-level `HEAD_DIM`, `BATCH`, `causal`, … that lines 705–737 bind are shadowed by parameters or not read; the unused functions' headers bind only their own names; `kernel.json`'s classes (host functions run only on the host; `_attn_fwd_inner`, `_maybe_make_tensor_desc`, `_attn_fwd`, `_attn_bwd_preprocess` not run by this launch).
- The launch: grid `(N_CTX // 128, 1, BATCH * N_HEAD)`, `program_id` axes 0 and 2, `fusion = true`, defaults for `debug`, `num_ctas`, `pdl`; every block shape the source builds is in `blocks`; only `num_warps`, `num_stages` given; no descriptors, no `CompiledKernel`, no host descriptor, no graph capture (`do_bench`).
- The conditions against the host: the test (`Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ≤ 4096`, `HEAD_DIM ∈ {64, 128}`, `sm_scale = 0.5`) and the benchmark (`4 × 32`, `N_CTX ≤ 2^14`, `sm_scale = 1.3`, its fp8 provider also runs the backward in `float16`) lie in `Inst.ok`; contiguity from lines 585–586 and `empty_like`; disjoint outputs; satisfiable.
- `Post`: what each output is; the tolerance at `N_CTX = 2^14` (`(1 + rhoA)^516 − 1 ≈ 0.43%`, the α term `≈ 2·10^-34`) and the per-pair parts (work/tol.py); it rejects dropped blocks, the wrong mask side, a missing scale or `Δ`, the wrong row of `M`, a coarser rounding of `P` or `dS` (inputs concentrated on the affected terms); the frame; `HeadV.Ok` satisfiable, and true with margin on the host's inputs (xE + eX about 0.08, pBr·yB about 10^4 at most).
- The five `[rule]` fields against rules/num.md's table and compiler-faithful.md: `DotF16` for the seven `float16` dots into `float32` with `maxImpreciseAcc = 0`, `DotTotal` for the model's `dotCfg`, `Exp2F32` and `MathTotal .exp2 .binary32` for `tl.math.exp2` on `float32`, `Faithful` in the cited form; their docstrings match. No `[kernel]` field.
- The write-up against the Lean (two wording findings above).

## Not checked

- The proofs' internal steps beyond reading them (Lean checks them).
- The tolerance on actual random draws: I evaluated it with the expected value of `Σ|K·Q|` for the host's inputs, not on sampled tensors.
