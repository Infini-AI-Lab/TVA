# fp8 constants: an exact conversion of the `float32` constant, or a host error

Library evidence (not a kernel's), for library issue L64. Round 20.

**Question.** What does Triton 3.8.0 build for a Python number used directly in an fp8 dtype
(`tl.full(shape, c, tl.float8e5)`, the value `c` of `tl.where(m, x, c)` beside a `float8e4nv`
tensor), and does the library's `FloatV.litF8` (`lean/TVA/Lang/Scalar.lean`, which
`Operand.constIn?` uses for the two fp8 dtypes) give the same bits? Round 14 modelled these
constants as the run-time conversion `tl.cvt8` of the `float32` constant, the `CvtSat…` facts,
and `docs/rules/num.md` said so; lr-14a and lr-14b found that they are not (`-inf` in `float8e5`
stored `0xfc`, where the model gave `0xfb`).

**Sources** (pinned in `primitives/`). `triton-3.8.0-semantic.py:601-613` (`scalar_constant`): a
Python number in an fp8 dtype is built as a `float32` constant and then `arith.truncf` to the fp8
type. `llvm-ArithOps.cpp:1588-1611` (`TruncFOp::fold`) and `1788-1825` (`convertFloatValue`):
MLIR folds the `truncf` only when APFloat's conversion loses no information; an unfolded `truncf`
of a constant has no lowering in this path, and compilation fails on the host.

**Method** ([run.py](run.py), output [out.txt](out.txt)). On the device, each kernel's stored fp8
byte, or the host error, against `FloatV.litF8` evaluated by Lean (`lake env lean` on a scratch
file importing `TVA.Lang.Scalar`) on the same doubles:
- F1: every value of each format (each pattern that is a number, both zeros included: 248 of
  `float8e5`, 254 of `float8e4nv`), as a source literal in `tl.full` and in `tl.where`;
- F2: values neither format holds exactly (the `float32` neighbours of a sample of fp8 values, the
  midpoints between neighbouring fp8 values, values beyond `maxVal`, values below the smallest
  subnormal, `0.1`, `1e6`), each in a kernel of its own;
- F3: `±inf`, values whose `float32` overflows (`1e39`, `-3.5e38`), `±nan` and NaN doubles with
  payloads, passed as `tl.constexpr` arguments, each in a process of its own with an empty cache
  (every NaN shares Triton's cache key: [compile-keys](../compile-keys/README.md));
- F4, for comparison, the run-time paths: `tl.store(p, c)`, `other=c` of a masked-off load and
  `tl.full([1], c, tl.float32).to(d)`, against `FloatV.cvtSat .rn (litF32 c)` (the `CvtSat…`
  facts, [rules/num.md](../../docs/rules/num.md)).

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`out.txt`: `differences in all: 0`).

| case | `float8e5` | `float8e4nv` |
|---|---|---|
| F1, every value of the format | 248 of 248 exact, in both forms | 254 of 254 exact, in both forms |
| F2, values the format does not hold | 79 (`tl.full`) and 20 (`tl.where`) values, all rejected on the host | 81 and 21, all rejected on the host |
| F3, `±inf` and overflowing doubles | `0x7c`/`0xfc` (the format's infinities) | rejected on the host |
| F3, NaN | `±nan` gives `0x7e`/`0xfe` (sign, quiet bit, the next fraction bit of the `float32`); `0x7ffc…` gives `0x7f`; a NaN with `float32` fraction bits below the top two set is rejected | every NaN rejected on the host |
| F4, run-time conversions | `CvtSat`: `1e6` and `inf` give `0x7b`, NaN gives `0x7f` | `CvtSat`: `1000.0`, `inf` give `0x7e`, NaN gives `0x7f` |

`FloatV.litF8` agrees with the device on every case, rejections included (`none` where the host
rejects).

**Conclusions.**
1. An fp8 constant is the exact, non-saturating conversion of the `float32` constant `litF32 c`:
   a value the format holds is kept, an infinity becomes the format's infinity (`float8e5`), a
   NaN keeps its sign, quiet bit and next fraction bit (`float8e5`) when no lower `float32`
   fraction bit is set; every other value (an inexact one, one beyond `maxVal`, an infinity or
   NaN of `float8e4nv`) is a compile error on the host, so no launch runs. This is what
   `FloatV.litF8` says, and what SEMANTICS "Constants" states (library issue L64).
2. The `CvtSat…` facts are about run-time conversions only (F4: the same constants stored, given
   as `other`, or converted with `.to`, saturate); round 14's account in `docs/rules/num.md`, that
   constants convert as `tl.cvt8`, was wrong and is corrected.
