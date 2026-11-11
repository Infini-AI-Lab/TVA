# Conversions into fp8 on sm_90

Library evidence (not a kernel's), for library issues L4 and L101. Triton 3.8.0 (TVA's `.venv`,
torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28. Scripts: [run.py](run.py) (finite
inputs), [specials.py](specials.py) (infinities and NaNs, round 20); outputs: [out.txt](out.txt),
[specials.out](specials.out).

**Question.** What does `x.to(tl.float8e4nv)` and `x.to(tl.float8e5)` (the default rounding,
`"rtne"`; the model's `tl.cvt8 x .rn`) give from `float32`, `float16` and `bfloat16`? The library
states `CvtRn` for the six pairs (`CvtF32E4m3`, …, [rules/num.md](../../docs/rules/num.md)):
a finite value with `|x| ≤ maxVal` of the fp8 format (448, 57344) rounds to nearest even.

**What Triton compiles them to** (`primitives/triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:190-246,
414-444`, and the PTX in `out.txt`): on sm_89 and later, `cvt.rn.satfinite.e4m3x2.f32` /
`e5m2x2.f32` from `float32`, the `f16x2` forms from `float16`, and from `bfloat16` an exact
widening (`cvt.f32.bf16`) followed by the `float32` form.

**Method.** Every input bit pattern: all `2^32` of `float32` (64 chunks of `2^26`), all `2^16` of
`float16` and of `bfloat16`, each converted on the device into both fp8 formats. The reference is
round-to-nearest-even computed exactly on the GPU in integer arithmetic (the input is `M·2^E` with
an integer `M < 2^24`; the quantum is `2^(max(⌊log₂|x|⌋, emin) − p)`; ties go to the even
significand; the sign is kept, also for values that round to zero), and compared bit for bit for
every finite input with `|x| ≤ maxVal`. (Before the run, the reference agreed with torch's own
fp8 conversion on 10^6 random values.)

**Result** (`out.txt`).

| From | To | Finite inputs with `\|x\| ≤ maxVal` | Differ from round-to-nearest-even | Results above `maxVal` |
|---|---|---|---|---|
| `float32` | `float8e4nv` | 2277507074 | 0 | `0x7e`/`0xfe` (±448) |
| `float32` | `float8e5` | 2394947586 | 0 | `0x7b`/`0xfb` (±57344) |
| `float16` | `float8e4nv` | 48642 | 0 | ±448 |
| `float16` | `float8e5` | 62978 | 0 | ±57344 |
| `bfloat16` | `float8e4nv` | 34754 | 0 | ±448 |
| `bfloat16` | `float8e5` | 36546 | 0 | ±57344 |

Of the non-finite inputs, `run.py` converts only `float32`'s `+inf`, `-inf` and its canonical
NaN (`0x7fc00000`): `±inf` gives the largest finite value of the same sign and NaN gives `0x7f`.

**Infinities and NaNs** ([specials.py](specials.py), [specials.out](specials.out); round 20,
library issue L101). Every non-finite bit pattern of each source format: `±inf` and all
16,777,214 NaN patterns of `float32`, `±inf` and all 2,046 NaN patterns of `float16`, `±inf` and
all 254 NaN patterns of `bfloat16`, each converted into both fp8 formats. Every infinity gives
the largest finite value of its sign (`0x7e`/`0xfe`, `0x7b`/`0xfb`), and every NaN, of either
sign and any payload, gives `0x7f`, a NaN of both formats: `rows not as CvtSat says: 0`.

**Conclusions.**
1. `CvtRn` holds for all six pairs on every input it covers, bit for bit (zeros keep their sign),
   so `CvtAcc` does too (`CvtRn.cvtAcc`).
2. Finite values above `maxVal` and infinities saturate to the largest finite value of their
   sign (`.satfinite`), and NaN gives the NaN `0x7f`. Since round 14 (library issue L64) the
   library states this as `CvtSat` (`CvtSatF32E4m3`, …, `CvtSatBF16E5m2`): every input converts to
   `FloatV.cvtSat .rn x`, the saturating rounding, up to NaN payload. This folder shows the
   results beyond the range on every input: `run.py` the finite values above `maxVal` of every
   format, `specials.py` every infinity and NaN pattern of every format (the NaN's sign is not
   kept: every NaN gives `0x7f`, which `CvtSat` allows, since it holds up to NaN payload and
   sign of a NaN); `evidence/float-layer` (part A, tests
   `A.cvt.*.e4m3.sat1`, `A.cvt.*.e5m2.sat1`) compares the library's definition itself with the
   device on every `float16`, `bfloat16` and `float32` input, bit for bit up to NaN payload.
