# `tl.sigmoid`: Triton's `1 / (1 + math.exp(-x))`

**Question.** SEMANTICS.md listed `sigmoid` among the `tl.*` functions Triton writes in Triton, which the
front end inlines, but stated no body for it, so a model of `tl.sigmoid(x)` could not be judged against
the page (library issue L137, from a sandbox test-set run on `gdn_gating_bf16`, whose reviewer filed it
as `UNSUPPORTED` and the prover referred it). What does `tl.sigmoid(x)` compute, and on which dtypes is
it accepted? Round 29.

**Sources** (Triton 3.8.0, pinned in `primitives/`).
- `triton-3.8.0-standard.py` lines 44-50: `sigmoid` is a `@jit` function, also a tensor method
  (`_tensor_member_fn`), whose body is `return 1 / (1 + math.exp(-x))`, `math` being
  `triton.language.math`.
- `triton-3.8.0-math.py` lines 8-30 and 95-100: `math.exp` is `tl.exp`, under `_check_dtype(["fp32",
  "fp64"])`, which raises on a tensor of any other dtype.
- `triton-3.8.0-code-generator.py` lines 1358-1396 (`call_JitFunction`): a call of a `@jit` function is
  a helper call, inlined; its body is compiled by the page's rules.

**Method.** `run.py`, each case in its own process with its own Triton cache:
- S1: `tl.sigmoid(x)` on an `fp32` block loaded from memory, for **every** `fp32` bit pattern (`2^32`
  inputs), against `1 / (1 + tl.exp(-x))` written out in the same kernel, bit for bit (NaNs included);
- S2: the method form `x.sigmoid()` on `2^28 + 2^27` `fp32` bit patterns, against the same expression;
- S3: `fp64`, `2^24` random bit patterns and 15 edge values (`±0`, `±∞`, NaN, `±745`, `±2^1023`, …);
- S4: `bfloat16`, `float16` and `int32` blocks: whether the front end accepts the call;
- S5: a compile-time float `C` (`tl.constexpr`): `tl.sigmoid(C)` stored, against the expression on `C`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/sigmoid-round29/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 3.
**Date.** 2026-11-05.

**Result** (`output.txt`).

| case | what | result |
|---|---|---|
| S1 | every `fp32` `x` | 0 of `2^32` differ |
| S2 | `x.sigmoid()`, 402653184 `fp32` patterns | 0 differ |
| S3 | `fp64`, 16777231 patterns | 0 differ |
| S4 | `bfloat16`, `float16`, `int32` | each rejected at compile time: `Expected dtype ['fp32', 'fp64']` |
| S5 | compile-time `C` in `{0, 1, -3.5, 20, -100}` | the same bits as the expression |

**Conclusions**, used by `docs/SEMANTICS.md` ("`tl.sigmoid`"):

1. `tl.sigmoid(x)` and `x.sigmoid()` are the expression `1 / (1 + tl.exp(-x))` by the rules of the page
   (S1-S3, S5): a float negation, `tl.exp`, the addition of the Python number `1` and the default float
   division `/` of the Python number `1` by the sum, each in `x`'s dtype; a kernel's model writes these
   operations, and their accuracy is the page's for `tl.exp` and `/` (`Rules.Num.MathAcc`,
   `Rules.Num.DivFull`).
2. It is accepted only on `fp32` and `fp64` (S4): a `bfloat16` or `float16` argument is rejected on the
   host, so a kernel converts first (`tl.sigmoid(b.to(tl.float32))`), and the model writes that
   conversion.
