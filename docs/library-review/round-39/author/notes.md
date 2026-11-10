# Library batch, round 39: author's notes

Issues L175 to L179 ([ISSUES.md](../../ISSUES.md)): lemmas that kernel provers re-derived by hand across the
fixed claims, now in five new modules of the proof kit. Five authors each wrote one module, from TVA and
Mathlib alone (L7, L115). The coordinator merged them: the imports in `TVA.lean` and the rows in `LIBRARY.md`.
Every addition is a proved lemma or a definition over `ℝ` or TVA's own types: no semantics, rule, device fact
or document that a kernel reviewer reads changed meaning, and no existing declaration changed. `lake build TVA`
and the twelve kernel libraries with Lean files build with the new imports (the modules add `simp` lemmas;
no kernel proof broke). Every declaration of the five modules depends only on `propext`, `Classical.choice`
and `Quot.sound`, and none uses `sorry`. No kernel or claim folder was edited.

## L175: Integer-width facts for any width

- **What changed.** New module `TVA.Lemmas.IntW` (130 theorems):
  - tracked values for any width, with no `0 < w` premise:
    - `add_nat`, `sub_nat`, `mul_nat`, `add_int`, `sub_int`, `mul_int`;
    - a constant operand: `add_ofNat`, `mul_ofNat`, `ofNat_mul`, `sub_ofNat`;
    - the `I64` names, and `I32/I64.toInt_add/sub/mul` with the bounds written out;
  - `SFits` helpers: `sfits_natCast`, `sfits_of_nonneg`, `SFits.mono`, `sfits_nat_add/mul(_of_le)`, the
    `I32`/`I64` forms;
  - literal `toInt` as `simp` lemmas (`0`, `1`, `-1`, numerals in range, `IntV.ofNat n`, unsigned numerals);
  - wrapping `ofNat`/`ofInt` arithmetic (`simp`, always true);
  - `ext_toInt`, `toInt_inj`, `eq_eq`, `not_eq_eq`;
  - a lane of `tl.idiv`/`tl.irem` on non-negative values (`toInt_of_div?`, `toInt_of_rem?`);
  - every widening integer `ConvRel` keeps `toInt` (`int8…int64`, `uint8…uint64`), and `i64→i32` in range;
  - lane forms: `Tensor.get1_add/…/castInt` (`simp`), `Tensor.toInt_get1_add/sub/mul/sadd/muls`,
    `I32.toInt_arange`, `I32.toInt_sadd_arange`.
- **Why.** Provers wrote 908 such lemmas in 39 of the 40 fixed claims (about 8,100 lines), while the library
  stated the tracked forms for `int32` only.
- **Evidence.** None needed (proved lemmas about `IntV`, `SFits` and `ConvRel` as `Lang/Scalar.lean` and
  `Lang/Promote.lean` define them).
- **Constants whose meaning changed.** None (new declarations only).

## L176: The 2-D strided tile

- **What changed.** New module `TVA.Lemmas.Tile2` (53 declarations), covering the pattern
  `base + rows[:, None]*s0 + cols[None, :]*s1` under `(rows < M)[:, None] & (cols < N)[None, :]` (either factor,
  or none) at any signed width:
  - lane offsets without overflow from the largest offset (`Tile2.off_toInt` and the per-term lemmas,
    `pid*BM + arange` in `int32` and `int64`);
  - lane pointers (`ptr_lane*`, both orders of the terms; `Tensor.get2_ptrAdd`);
  - masks (`MaskOk`, `maskOk_and/rows/cols/none`);
  - the footprint (`addrs_tile2`, `img_tile2`, `accBytes_tile2`, `Buf.tile2`);
  - `safe_load_tile2`, `safe_store_tile2`;
  - no aliasing `Inj` (row-major `N*s1 ≤ s0`, column-major `M*s0 ≤ s1`) and disjointness of tiles at
    different `(r0, c0)` or different grid programs (`elem_disjoint`, `elem_disjoint_grid`,
    `Buf.tile2_disjoint(_grid)`).
- **Why.** Every claim with a matrix operand walks such tiles, and `TVA.Lemmas.Block` covered the 1-D block
  only. Provers wrote footprint lemmas in all 40 claims, mostly this pattern again.
- **Evidence.** None needed (proved from `Block`, `Access`, `Buf` and `Tensor`).
- **Constants whose meaning changed.** None (new declarations only). Left out: offsets converted inside the
  program, negative strides, ranks other than 2; for other strides a kernel proves `Inj` itself.

## L177: Float constants, exact conversions, approximations of a maximum

- **What changed.** New module `TVA.Lemmas.FloatFacts` (198 theorems):
  - zeros of every format: `ofPat 0`, `lit*`, `cvt`, `tl.zeros`, `DType.ofBits`, `val?`;
  - `u`, `η`, `maxVal` of `binary32`, `binary16`, `bfloat16`, `binary64` and the fp8 formats as closed
    numbers in `ℝ`;
  - finiteness from magnitude (`isFinite_of_abs_toReal_le_maxVal` and its converse);
  - exact widening conversions (`cvt_*_exact`, `ConvRel.f16_f32` … `f8e4nv_f32`: `r = cvt x`, finite, same
    `toRat` and `toReal`; `ConvRel.cvt_*`: `cvt x` is an allowed result);
  - `NaNEq` as an equivalence and `NaNEq.exact_of_isFinite`;
  - `FloatErr`'s bounds on `toReal` (`add/sub/mul/fma/divRn/cvt/cvtWith_rz_err_real`);
  - the maximum and minimum of approximations (`FloatV.max_approx`, `MaxRel.max_approx`,
    `Tree.eval_maxRel_approx`, and the `min` forms).
- **Why.** Provers wrote 673 such lemmas in 35 claims (about 10,400 lines).
- **Evidence.** None needed (proved from `TVA.Math`, `FloatErr`, `FloatMax` and `ConvRel`). `ConvRel` widening
  is not exact by definition: between standard formats it is `cvt x` up to a NaN's payload, and out of fp8 it is
  `cvt x` on finite values only (a non-finite fp8 value may convert to any value), so the per-pair lemmas take
  a finite `x`.
- **Constants whose meaning changed.** None (new declarations only).

## L178: Online softmax

- **What changed.** New module `TVA.Lemmas.OnlineSoftmax` (pure mathematics over `ℝ`, 65 theorems, 7
  definitions), written from TVA and Mathlib alone (L115). It has three parts.
  - **The exact invariant** (`l_eq`, `acc_eq`, `ratio_eq`):
    - Trip `u` adds a block of lanes `B u`; a masked lane is left out of the block or has weight `0`.
    - For any references `m_t` (not only the true maximum), the rescaled recurrence from an empty state
      equals the closed sums `lsum`/`asum`. `α 0` is free, which is how the first trip from `m = −∞` fits.
    - `acc/l` is the softmax-weighted sum `wavg`, whatever the references.
    - Keys in blocks under a mask: `ratio_blocks`, `wavg_blocks`.
  - **The perturbed form:**
    - `accum` is a running bound in `DotLoop`'s style. `absErr_alpha` is its closed form for any references,
      and `bound_le` gives `≤ 2tε·M + 2Σ η` when `α ≤ 1` and `tε ≤ 1/2`.
    - `l_err` and `acc_err` apply it to `l` and `acc`.
    - `ratio_err` bounds the final ratio: `|â/l̂ − A/L| ≤ (eA + V·eL)/(L − eL)`.
    - `err_final` and `err_final_mono` combine them; with a running maximum the result is within
      `4n(εa + εl)·V + 4(Σ ηa + V·Σ ηl)/L`.
    - One trip from its parts: `scale_err`, `block_err`, `step_err`, `l_trip`.
  - **Exponential helpers:** `|exp a − exp b| ≤ |a − b|·exp(max a b)`, `|exp d − 1| ≤ 2|d|`, `2^x` as
    `exp(x·log 2)`, `exp x ≤ 2^k` when `x ≤ k·log 2`, and a rounded-argument bound.
- **Why.** None of the five fixed claims with this loop was proved. Their provers stopped at the invariant.
- **Evidence.** None needed (mathematics over `ℝ`).
- **Constants whose meaning changed.** None (new declarations only). Left out:
  - the connection to float values, which kernels make with `FloatErr`, `FusedErr` and the device facts;
  - a fully masked row;
  - the log-sum-exp output;
  - the backward pass.

## L179: A recurrence carried across loop trips with a decay

- **What changed.** New module `TVA.Lemmas.Recurrence` (pure mathematics over `ℝ`, 30 theorems):
  - the closed form of `S (t+1) = g t·S t + U t` (`closed`, `closed_Ico`);
  - a running bound `run` in `DotLoop`'s style (`run_eq`, `le_run`, `run_le_sum`, `run_le_pow`);
  - the perturbed state (`perturb`, `perturb'`, `perturb_le_sum`: `|Ŝ n − S n| ≤ |Ŝ 0 − S 0| + Σ (e + ε + γ·|S|)`
    when `|ĝ| ≤ 1`; `perturb_le_uniform`, `perturb_le_pow`);
  - the exact state's size (`abs_le_sum_of_le`);
  - exponential gates (`prod_exp_sub` telescopes, `exp_sub_mem_Ioc` puts each factor in `(0, 1]`,
    `closed_cumsum` is the chunked form `exp(cs_last)·S + Σ exp(cs_last − cs_i)·U_i`);
  - a matrix state with a scalar decay (`closed_mat`, `closed_cumsum_mat`, `perturb_mat`, `perturb_mat_le_sum`).
- **Why.** One of the five fixed claims of this family (a state carried across chunks with a decay) was proved.
- **Evidence.** None needed (mathematics over `ℝ`).
- **Constants whose meaning changed.** None (new declarations only). Left out: float rounding (kernel-side) and
  a matrix-valued decay such as the delta rule's `(I − β k kᵀ)`, which needs operator norms.
