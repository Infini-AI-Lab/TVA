# Python number constants, per dtype

Library evidence (not a kernel's), for library issue L23. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, Lean 4.35.0 for the library's side, 2026-10-27.
Script: [run.py](run.py); output: [out.txt](out.txt).

**Question.** What value does a Python `int` or `float` literal take when it meets a tensor in an
operation, or fills `tl.full`, for each float dtype? Do the library's definitions
`FloatV.litF64`, `litF32`, `litF16` and `litBF16` (`lean/TVA/Lang/Scalar.lean`) give that value?
And what does `other=` of `tl.load` give?

**What the library reads.**
- A Python number that meets a tensor in a binary operation becomes
  `scalar_constant(c, dtype of the operation)` (`primitives/triton-3.8.0-semantic.py:205-206`);
  `tl.full` does the same (`:622-623`, `make_scalar`). A zero becomes the null value `+0`; another
  value is built by `get_<dtype>` (`:601-613`).
- `get_fp64(double)`, `get_fp32(float)`, `get_fp16(float)` (`getF16FloatAttr`) and
  `get_bf16(float)`, which builds the constant from the text `std::to_string(v)`: C's `%f`, six
  digits after the point (`primitives/triton-3.8.0-ir.cc:952-972`). A pybind `float` parameter is
  C's conversion of the double, to nearest even.
- `tl.load(…, other=c)` first makes `c` a tensor of its own type (`to_tensor`, `float32` in
  float32's normal range or for `0`, `±∞`, NaN, else `float64`; an `int` is `int32` or wider,
  `:118-158`; `load`, `primitives/triton-3.8.0-core.py:2534-2535`), then converts it to the
  pointer's element type (`semantic.py:1045`).

**Method.** 9364 literals, each with both signs: an adversarial list (values around the
six-decimal cut, `5·10⁻⁷`, `10⁻⁶`, `1.23456e-5`; ties of the cut at odd multiples of `2⁻⁷`; values
within `10⁻⁹`–`6·10⁻⁷` of `bfloat16` rounding midpoints from `2⁻²⁰` to `2⁵`; the edges of float16 and
float32; values outside float32's range; `±0`; integers up to `2^53 + 1`) and 4000 log-uniform
random values in `[10⁻⁹, 10⁶]`. For each float dtype, kernels of 256 literals each store, per
literal, `tl.full([1], c, d)`, `x * c` with `x = 1` of dtype `d` (exactly `c`), and
`tl.load(p, mask=False, other=c)`. The script evaluates the library's functions on the same
doubles with Lean (`#eval`, in a scratch file importing `TVA.Lang.Scalar`) and compares bit for
bit (`other=`, a conversion, up to NaN payload). It also compares C's `snprintf("%f")` with
Python's `"%f"` on the float32 values. The literals are written with `repr`, so the list has no
NaN or infinity (they have no literal); they are in "NaN constants" below. Round 19: `run.py`
compares `tl.full` and `x * c` bit for bit (before, any NaN equalled any NaN, which no literal of
the list reached), and its re-run on the round's library printed exactly `out.txt` (2026-10-28).

**Result** (`out.txt`).

| dtype | `tl.full` vs `FloatV.lit…` | `x * c` vs `FloatV.lit…` | `other=` vs own type, then one rounding | literals where `other=` and the operation's constant differ |
|---|---|---|---|---|
| `bfloat16` | 0 of 9364 differ | 0 | 0 | 3202 |
| `float16` | 0 | 0 | 0 | 0 |
| `float32` | 0 | 0 | 0 | 0 |
| `float64` | 0 | 0 | 0 | 9294 |

Examples (patterns): `1e-7` is `bfloat16` `+0`, `-1e-7` is `-0`, `1.23456e-5` is `0x3749`
(`1.198e-5`, the `bfloat16` nearest to `0.000012`); in `float64` each literal is the Python double
itself (`0.1` is `0x3fb999999999999a`, not the double of float32's `0.1`). C's and Python's `%f`
agree on every float32 value tested.

**Conclusions.**
1. The constant in an operation of dtype `d`, and in `tl.full(…, c, d)`, is `FloatV.litF64`,
   `litF32`, `litF16` or `litBF16` of the Python double, on every literal tested: the double itself
   in `float64`, the nearest `float32`, the nearest `float16` of that, and the `bfloat16` nearest to
   the six-decimal text of that `float32`. The earlier rule (the nearest `float32`, then converted)
   was wrong for `float64` (almost every literal) and `bfloat16` (a third of these literals).
2. `other=` follows the number's own type, then one rounding to the element type: in these tests,
   the nearest `float32`, converted (or the double, converted, for a number outside float32's
   normal range, and the integer itself for an `int`).

## Integer constants (round 5, library issue L35)

**Question.** `[literal]` said that in an integer dtype the constant is "`c` itself". What does
`tl.full(shape, c, d)` hold for an `int` `c` outside `d`'s range, and what does an operator do with
one? Script: [int_full.py](int_full.py); output: [int_full.out.txt](int_full.out.txt); Triton 3.8.0,
H200 NVL, GPU 0, 2026-10-28.

**Method.** For each of the eight integer dtypes, 50 constants (the range's ends and one past them,
`±2^w`, `2^w ± 1`, `±300`, and 40 random integers in `[-2^(w+8), 2^(w+8)]`) stored with `tl.full([1],
c, d)` and compared with `c` modulo `2^w`; then `x + c` for `c` one past each end.

**Result.** Every constant that compiles is stored as `c` modulo `2^w`, in all eight dtypes (0
otherwise): `300` in `int8` is `44`, `2^31` in `int32` is `-2^31`. Rejected on the host: a negative
`c` in an unsigned dtype, and a `c` outside the 64-bit range in `int64`/`uint64`. Every `x + c` with
`c` out of `d`'s range is rejected ("Scalar … is out of range", or a negative scalar with an unsigned
tensor). So the constant is `IntV.ofInt c` (`c` modulo `2^w`) wherever the launch is not rejected,
as [SEMANTICS.md](../../docs/SEMANTICS.md) ("Constants") says since round 6 (`[literal]` in round 5).

## NaN constants (round 19, library issue L98)

**Question.** Which bit pattern does Triton 3.8.0 build for a Python NaN in each float dtype, and
do `FloatV.litF64`, `litF32`, `litF16` and `litBF16` give it, bit for bit, sign included? The
constants are exact patterns, unlike the operations on them (up to NaN payload). Script:
[nan.py](nan.py); output: [nan.out.txt](nan.out.txt); Triton 3.8.0, torch 2.11.0+cu130, Python
3.13.9, H200 NVL, GPU 0, 2026-10-28.

**What the library reads.** The builders take the Python `float` as a C `float`
(`get_fp32(float)`, `get_fp16(float)`, `get_bf16(float)`) or a `double` (`get_fp64`)
(`primitives/triton-3.8.0-ir.cc:952-972`): C's `(float)` of the double, as the launcher's
`extractFP32` writes it (`primitives/triton-3.8.0-nvidia-driver.c:1152-1157`), which on x86-64
keeps a NaN's sign and the top 22 bits of its payload and sets the quiet bit (`FloatV.hostF32`,
`nanNarrow`); `float16` converts that `float` with LLVM's `APFloat` (sign and top payload bits
kept, quiet bit set); `bfloat16` parses `std::to_string(v)`, which is `nan` or `-nan`, a quiet NaN
of that sign with no payload. The device evidence below is what establishes the patterns.

**Method.**
- N1: NaNs and infinities written in the kernel's source, 14 expressions the front end evaluates
  in Python (`float('nan')`, `-float('nan')`, `float('-nan')`, `float('inf') - float('inf')`,
  `float('inf') * 0.0`, `float('nan') + 1.0`, …, `±float('inf')`). For each dtype a kernel stores
  `tl.full([1], E, d)` and `tl.where(<false>, x, E)` (the constant built in `d`), compared bit for
  bit with the library's function of the double Python computes for `E`, and
  `tl.load(p, mask=False, other=E)` (the `float32` constant, then converted), compared up to NaN
  payload.
- N2: 306 NaN doubles: quiet and signalling, each single payload bit, all-ones payloads and random
  payloads, both signs, passed as `tl.constexpr` arguments, in batches of 16, each batch in a
  process of its own with an empty `TRITON_CACHE_DIR` (see N3); `tl.full([1], C, d)` in the four
  dtypes, bit for bit.
- N3: the cache: one process, one empty cache folder, `C = float('nan')` then `C = -float('nan')`;
  a second process on the same folder with `-float('nan')`; and the reverse order in an empty
  folder.
- N4: runtime scalar arguments holding a NaN (`x` without annotation, `x: tl.float32`,
  `tl.float64`, `tl.float16`, `tl.bfloat16`), stored as passed, for SEMANTICS "Scalar arguments".

**Result** ([nan.out.txt](nan.out.txt)).
- N1: 0 differences. `float('nan')` is `0x7fc0`, `0x7e00`, `0x7fc00000`, `0x7ff8000000000000` in
  `bfloat16`, `float16`, `float32`, `float64`; `-float('nan')`, `float('-nan')`, `inf - inf`,
  `inf * 0`, `0 * -inf` and `inf / inf` are Python's `0xfff8000000000000` (x86's default NaN is
  negative) and give `0xffc0`, `0xfe00`, `0xffc00000`, `0xfff8000000000000`. `other=` is the same
  pattern in every dtype (the compiler folds the conversion), within the library's "up to NaN
  payload".
- N2: 0 of 306 differ in each dtype: a NaN keeps its sign everywhere; `float32` keeps the top 22
  payload bits and sets the quiet bit (`0x7ff4000000000000` is `0x7fe00000`, the signalling
  `0x7ff0000000000001` is `0x7fc00000`, `0xfff0000020000000` is `0xffc00001`); `float16` keeps the
  top 9 (`0x7f00` for `0x7ff4…`); `bfloat16` is `0x7fc0` or `0xffc0` whatever the payload;
  `float64` is the double itself, a signalling NaN included.
- N3: after `C = float('nan')`, a launch with `C = -float('nan')` stored `0x7fc00000`, in the same
  process and in a second process on the same cache folder; in the reverse order both stored
  `0xffc00000`. Triton's cache keys write every NaN as `nan` (`primitives/triton-3.8.0-jit.py:611`,
  `primitives/triton-3.8.0-compiler.py:73-75`), so a `tl.constexpr` NaN runs whichever NaN's kernel
  was compiled first.
- N4: an unannotated `float` and `tl.float32` argument are `FloatV.hostF32` of the double on all 8
  NaNs; `tl.float64` is the double; `tl.bfloat16` is the top 16 bits of `FloatV.hostF32`;
  `tl.float16` is `0x7e00` or `0xfe00` (the payload dropped).

**Conclusions.**
1. The constant of a Python NaN written in the source is `FloatV.litF64`, `litF32`, `litF16` or
   `litBF16` of its double, bit for bit, both signs: the device's pattern, not the library's
   canonical NaN (library issue L98). With payloads (N2) the functions are right on every double
   tested.
2. A NaN passed as a `tl.constexpr` argument has no fixed pattern: the kernel the launch runs may
   be another NaN's (N3). `docs/exclusions.md` excludes it; SEMANTICS "Constants" says so.
3. A runtime `float` argument holding a NaN is `FloatV.hostF32` of the double (SEMANTICS "Scalar
   arguments").
