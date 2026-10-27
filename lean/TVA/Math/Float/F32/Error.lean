import TVA.Math.Float.F32.Exact

/-!
# Rounding-error theorems for the IEEE `f32` arithmetic of `TVA.Math.Float.F32` (index)

`TVA.Math.Float.F32` (`TVA.Math.Float.F32`) defines binary32 arithmetic as the exact rational result rounded to
nearest-even into the binary32 grid (`roundMag` / `roundRat`). This file proves
the classical error bounds *about those definitions*:

* `roundRat_err_normal`: `|round q − q| ≤ u·|q|` with `u = 2^-24` on the normal
  range `minNormal ≤ |q| ≤ maxNormal`; `roundRat_err_sub`: `≤ 2^-150` below it;
* `add_err` / `mul_err` / `fma_err`: one rounding per operation;
* `fmaChain_err`: an FMA chain of `n` products (the tensor-core twin's
  accumulation order) is within `γ_n = (1+u)^n − 1 ≤ n·2^-23` of the exact sum,
  relative to `|c| + Σ|aᵢ||bᵢ|`;
* `addChain_err`: a chain of `m` additions (the atomic fix-up) likewise;
* `gemmTwin_err`: the composition — `K` products split into `s` chains whose
  results are added — is within `2^-23·(K + s + 1)·Σ|a||b|`, the bound the
  `streamk_gemm_bf16` specification states;
* `bf16_round_err`: rounding `f32 → bf16` (RNE) perturbs a normal value by at
  most `2^-8` relatively (the `P` rounding of `varlen_flash_attn_bf16`).

The hypotheses are the usual ones: finite operands and exact results in the
normal range (or exactly zero); nothing is assumed about the rounding beyond the
definitions in `TVA.Math.Float.F32`.

The analysis is split by topic:

* `TVA.Math.Float.F32.Bounds`: the definitions (constants, `rne`, the chains, `γ`, `δ`, `roundBf16`);
* `TVA.Math.Float.F32.Round`: `log2Floor`, round-to-nearest-even, `roundMag`, `roundRat_err_*`;
* `TVA.Math.Float.F32.OpErr`: one rounding per operation (`add_err`, `mul_err`, `fma_err`);
* `TVA.Math.Float.F32.ChainErr`: FMA and addition chains (`fmaChain_err`, `addChain_err`), `γ`;
* `TVA.Math.Float.F32.GemmErr`: the GEMM twin (`gemmTwin_err`);
* `TVA.Math.Float.F32.Bf16Round`: rounding `f32 → bf16` (`bf16_round_err`);
* `TVA.Math.Float.F32.AbsErr`: the same bounds with an absolute term, no subnormal hypothesis;
* `TVA.Math.Float.F32.Exact`: rounding a representable value is exact.

Import this module for all of them.
-/
