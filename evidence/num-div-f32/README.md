# `/` on `float32` (`DivFull`)

Library evidence (not a kernel's), for library issue L4. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** The library states `DivFull` ([rules/num.md](../../docs/rules/num.md), "Division"):
for finite `x`, `y` with `y ≠ 0` and `|x/y| ≤ 2^126`, the result of `x / y` on `float32` is finite
and within `2^-22·|x/y| + 2^-148` of `x/y`. Triton lowers `/` to `llvm.nvvm.div.full`, which is
`div.full.f32` (the PTX in `out.txt`). Does the device meet the bound?

**Method.** The division space has `2^64` pairs, so the test is adversarial: about `3.1·10^10`
pairs.
- **A.** Every divisor mantissa (`2^23`) at divisor exponents `1, 2, 50, 127, 200, 253, 254`
  (biased), against 32 dividend mantissas (`1`, `2 − 2^-23`, `1.5` and 29 random ones), both
  signs, with the dividend's exponent chosen so that the quotient lies in each binade of interest:
  subnormal quotients (`2^-149`, `2^-140`), around `2^-126`, `2^-60`, `1`, `2`, `2^64`, `2^100`,
  and `2^124`–`2^126`.
- **B.** Every subnormal divisor against dividends at ten exponents, and every subnormal
  dividend against divisors at eight exponents.
- **C.** `2^31` random pairs of finite bit patterns.

The reference is the `float64` quotient on the GPU (IEEE division: within `2^-53·|x/y|` of the
real quotient, never subnormal for `float32` operands). The script checks
`err·(1 + 2^-52) + 2^-52·|q| ≤ 2^-22·|q| + 2^-148` for every pair with `|q| ≤ 2^126`, and that
the result is finite.

**Result** (`out.txt`).

| Inputs | Pairs with `\|x/y\| ≤ 2^126` | Outside the bound | Non-finite | Largest error / bound | Largest error, ulps of the quotient's binade |
|---|---|---|---|---|---|
| A: every divisor mantissa | 27942544904 | 0 | 0 | 0.650 | 2.106 |
| B: subnormal operands | 957190703 | 0 | 0 | 0.645 | 2.060 |
| C: random pairs | 1858035562 | 0 | 0 | 0.648 | 2.106 |

**Conclusions.**
1. `DivFull` holds on every pair tested, with margin: no error above `0.651` of the bound.
2. The error reaches 2.11 ulp of the quotient's binade (above the PTX ISA's "2 ulp"), near the
   top of a binade, where the bound `2^-22·|x/y|` is almost 4 ulp. So the fact is the relative
   bound, as stated, not "2 ulp".
3. Subnormal operands and quotients are within the bound too (the absolute term `2^-148`).
