# `tl.exp` past its overflow, and `/` by an infinite divisor, on `float32` (library issue L154 (3))

**Question.** `ExpF32` (`MathAcc` with room) and `DivFull` (`|x/y| ≤ 2^126`, a finite divisor) say nothing
where `tl.exp` overflows or where the divisor is infinite. What does the H200 give there?

**Method** (`run.py`), exhaustive over the inputs named:
- E1: `tl.exp(x)` (`mul.f32` by `log2 e`, then `ex2.approx.f32`) for every `float32` pattern from `+64.0`
  (`0x42800000`) to `+∞` (`0x7f800000`), `1023410177` inputs: which give `+∞`, the smallest input from which
  every larger one gives `+∞`, and the largest finite result.
- E2: `tl.exp(x)` for every pattern from `-64.0` (`0xc2800000`) to `-∞` (`0xff800000`): `+0`, subnormal or
  normal (for the record).
- D1: `x / y` (`tl.fdiv`, `div.full.f32`) for every finite `float32` `x` (`4278190080` patterns, both signs,
  subnormals and zeros included) and `y = +∞`, `y = -∞`, loaded from memory so nothing folds: is the result
  the zero of sign `sign x xor sign y`?

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-special-f32/run.py` from TVA's root (a few
minutes).

**Triton version.** 3.8.0 (torch 2.11.0+cu130). **GPU.** NVIDIA H200 NVL (sm_90), GPU 2. **Date.** 2026-11-08.

**Result** (`output.txt`).
- E1: `1020169705` of the inputs give `+∞` and none NaN. The largest input with another result is
  `0x42b17217` (`88.72283172607422`); every input from `0x42b17218` (`88.72283935546875`) to `+∞` gives `+∞`.
  The largest finite result is `0x7f7fff4f`. (`ln` of the largest `float32` is about `88.722839052`, so from
  `0x42b17218` on the exact `e^x` is beyond the format, where `ExpF32` states nothing.)
- E2: every input gives `+0`, a subnormal or a normal; nothing else.
- D1: by `+∞` and by `-∞`, 0 of the `4278190080` results differ from the zero of sign `sign x xor sign y`.

**Conclusions.**
1. On `float32`, `x / ±∞` is the signed zero `IEEE` division gives, for every finite `x`: the fact
   `DivF32InfDivisor` (`lean/TVA/Rules/Num/Sm90.lean`), exact.
2. `tl.exp` gives `+∞` exactly on the inputs from `0x42b17218` to `+∞` among the non-negative ones. Round 33
   (library issue L154) states this as the fact `Rules.Num.ExpF32Overflow` (`lean/TVA/Rules/Num/Sm90.lean`:
   on `float32`, every input from `1453635/16384 = 0x42b17218` up, and `+∞`, gives `+∞`), on this evidence,
   re-run that round with identical output. For it the reference platform changed first: `Platform.ref`
   (`lean/TVA/Lemmas/Consistency.lean`) now gives its table's value also at a finite input where no float is
   within the requested bound (`Platform.ref.mathAt_far`), and `sm90Special` gives `exp x = +∞` there, so
   `Platform.sm90ref.expF32Overflow` proves the fact of that platform (library issue L156 corrected this
   conclusion, which still said the library states no fact from it).
