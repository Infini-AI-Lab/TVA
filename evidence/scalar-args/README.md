# The types and values of scalar kernel arguments

**Question.** When a launch passes a Python scalar (an `int`, `bool`, `float`, a numpy scalar, a
0-d tensor, `None`) to a `@triton.jit` parameter, what type does the kernel's parameter get, and
what value does the kernel receive? How do an annotation (`n: tl.int32`), `do_not_specialize` and
`do_not_specialize_on_alignment` change that? Library issue L29 (round 8).

Where Triton 3.8.0 decides it:

- **Specialisation** (the type, and whether the value becomes a compile-time constant), in
  `primitives/triton-3.8.0-specialize.cc`: `handle_long_type` lines 264-306
  (an `int` equal to `1` becomes a constexpr, 274-276; `i32` in [-2^31, 2^31), else `i64`, 281;
  `u64` in [2^63, 2^64), 297; beyond, `OverflowError`, 286-296; the key `D` when the value is a
  multiple of 16, 283 and 299); `handle_bool_type` 377-382 (`u1`); `handle_float_type` 384-388
  (`fp32`); `handle_tensor` 308-375 (a tensor is a pointer, `D` when its address is a multiple of
  16, 363); the dispatch `specialize_arg` 510-569 (exact Python types first, 469-472; `None` a
  constexpr, 522-524; anything with `data_ptr` a pointer, 557-559; subclasses of `int` and `float`,
  562-567; anything else is refused, 600-606: `failed to specialize argument of type`).
- **Annotations and options**, in `primitives/triton-3.8.0-jit.py`: `KernelParam.annotation_type`
  lines 317-332; `create_function_from_signature` 396-458 (an annotated parameter takes the
  annotated type, 415-424: an annotated float or bool is not specialised at all, an annotated
  integer keeps only the `D` key, so `1` is not a constexpr; `do_not_specialize` passes
  `specialize=False`, `do_not_specialize_on_alignment` passes `align=False`, 412-413); `_pack_args`
  702-724 (the constexprs, 717-718, and the attributes, 720-722). The key `D` becomes
  `tt.divisibility = 16` in `BaseBackend.parse_attr` (`primitives/triton-3.8.0-backends-compiler.py` lines
  75-80). Annotation names: `primitives/triton-3.8.0-utils.py` lines 77-115 (`int`
  is `i32`, 103; `bool` and `int1` are `u1`, 79-82).
- **The value**, in the launcher's extractors, `primitives/triton-3.8.0-nvidia-driver.c` lines
  1090-1163, chosen by the signature's type name (1229-1275; `i1` uses the `int32` extractor, 1243,
  `u1` the `uint32` one, 1255): `i8`/`i16`/`i32` take `PyLong_AsLong` and narrow it in C (wraps),
  `i64` `PyLong_AsLongLong`, `u8`/`u16`/`u32` `PyLong_AsUnsignedLong` (negative values are refused,
  wider values wrap), `u64` `PyLong_AsUnsignedLongLong`; `fp16` packs the double with
  `PyFloat_Pack2` (1130-1142), `bf16` converts the double to `float` and keeps its top 16 bits
  (1144-1150), `fp32` converts the double to `float` (1152-1157), `fp64` keeps the double
  (1159-1163).
- Arithmetic on a constexpr is Python's (`constexpr.__floordiv__`, `primitives/triton-3.8.0-core.py`
  lines 267-268: floor division), on a runtime integer Triton's (`floordiv`,
  `primitives/triton-3.8.0-semantic.py` lines 317-329: `sdiv`, truncation).

**Method.** `run.py` compiles and launches small kernels on GPU 0, one process per case, with a
fresh Triton cache. Each kernel stores its parameter into a buffer: integers into an `int64` buffer
(the store converts by the parameter's signedness), floats bitcast to an integer of their width,
so the host reads the exact value and bits. It reports the parameter's type in the JIT's signature
(`CompiledKernel.src.signature`, Triton's type with signedness) and in the front end's IR
(`asm['source']`, `tt.func`; MLIR integers are signless, so `u32` shows as `i32` there), with its
attributes, and compares the value read back with a host reference: the exact integer, the
round-to-nearest-even conversion of the double (numpy) for `float32` and `float16`, and both the
truncation of the nearest `float32` and torch's round-to-nearest-even for `bfloat16`.

- S1: the int `1` unannotated, annotated `n: tl.int32`, and with `do_not_specialize=['n']`; and `3`.
  The kernel stores `n`, `n + n`, `(n - 2) // 2` and `n * 65536 * 65536`.
- S2: integers across the `i32`, `i64` and `u64` ranges and beyond, multiples of 16, and the two
  options.
- S3: `True`, `False`; an annotated `b: tl.int1` given `True`, `1`, `2`, `0`.
- S4: a Python float to an unannotated parameter: 23 edge values (ties, overflow, underflow,
  subnormals, `-0.0`, infinities, NaNs) and 4000 random doubles (random sign, exponent in
  [-160, 135], 52 random mantissa bits, seed 20261028).
- S5: annotated `tl.float16` (16 edge values, 4000 random doubles with exponents in [-26, 15]),
  `tl.bfloat16` (11 edge values, 4000 random doubles, exponents in [-130, 127]), `tl.float64`
  (2005 doubles, all exponents, with `-0.0`, infinities, NaN, the smallest subnormal), `tl.float32`
  given ints, and annotated `tl.int64`, `tl.uint32`, `tl.int8`, `tl.int32` and `int` given values
  in and out of range.
- S6: Python `int`, `float`, `bool`; numpy `float64`, `float32`, `int64`, `int32`, `bool_`; a 0-d
  CUDA tensor and a 0-d CPU tensor; `None` and `0` tested with `x is None`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/scalar-args/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, numpy 2.3.5).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | argument | parameter | value received |
|---|---|---|---|
| S1 | `1`, unannotated | constexpr (absent from the IR) | Python's arithmetic: `(n-2)//2 = -1`, `n*65536*65536 = 4294967296` |
| S1 | `1`, annotated `tl.int32` or `do_not_specialize` | `i32` | `int32` arithmetic: `(n-2)//2 = 0`, `n*65536*65536 = 0` |
| S2 | int in [-2^31, 2^31) other than 1 | `i32` (`n+n` wraps: `2^30+2^30 = -2^31`) | exact |
| S2 | int in [-2^63, -2^31) or [2^31, 2^63) | `i64` | exact |
| S2 | int in [2^63, 2^64) | `u64` | exact (as unsigned) |
| S2 | int ≥ 2^64 or < -2^63 | rejected on the host: `OverflowError: integer to be specialized too large to represent` | - |
| S2 | multiple of 16 (also `0`, `-2^31`, `2^31`, `2^63`) | `tt.divisibility = 16`; none with `do_not_specialize` or `do_not_specialize_on_alignment` | exact |
| S3 | `True` / `False` | `u1` (IR `i1`) | 1 / 0 |
| S3 | annotated `tl.int1` given `2` | `u1` | 0 (the low bit) |
| S4 | Python float | `fp32` | the double rounded to nearest-even `float32`: 23 of 23 edge values, 4000 of 4000 random; overflow gives ±inf, `-0.0` kept, NaN quiet with its sign |
| S5 | annotated `tl.float16` | `fp16` | the double rounded to nearest-even `float16` directly (not through `float32`: `1+2^-11+2^-40` gives `0x3c01`); values that round to ±inf (`65520`, `70000`, 1 random) are rejected on the host: `OverflowError: float too large to pack with e format`; `inf` itself is accepted |
| S5 | annotated `tl.bfloat16` | `bf16` | the top 16 bits of the nearest `float32` (truncation, not rounding): 4000 of 4000 random; only 1960 equal round-to-nearest-even; `FLT_MAX` gives `0x7f7f`, not inf |
| S5 | annotated `tl.float64` | `fp64` | the double's bits, 2005 of 2005 |
| S5 | annotated `tl.float32` given `3`, `2^24+1` | `fp32` | `3.0`, `16777216.0` (the int's double, rounded) |
| S5 | annotated `tl.int64` | `i64` | exact; `2^63` rejected: `OverflowError: int too big to convert` |
| S5 | annotated `tl.uint32` | `u32` | wraps modulo 2^32 (`2^32` gives 0); `-1` rejected: `OverflowError: can't convert negative value to unsigned int` |
| S5 | annotated `tl.int8` | `i8` | wraps modulo 2^8 (`128` gives -128, `300` gives 44); `2^63` rejected: `OverflowError: Python int too large to convert to C long` |
| S5 | annotated `tl.int32` or `int` | `i32` | wraps modulo 2^32 (`2^31` gives -2^31; `2^40` gives 0) |
| S6 | `numpy.float64` | `fp32` | as a Python float |
| S6 | `numpy.float32`, `numpy.int64`, `numpy.int32`, `numpy.bool_` | rejected on the host: `TypeError: failed to specialize argument of type: numpy.float32` (etc.) | - |
| S6 | 0-d CUDA tensor | pointer `*i64` | the tensor's address, not its value |
| S6 | 0-d CPU tensor | rejected on the host: `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)` | - |
| S6 | `None` | constexpr (absent from the IR) | `x is None` is a compile-time `True` |

Every rejection is raised by the launch call on the host, before any program runs (the buffer
still holds its sentinel).

**Conclusions**, used by `docs/SEMANTICS.md`:

1. An unannotated integer argument equal to `1` is a compile-time constant (Python's integer
   arithmetic, unbounded, floor division) unless the parameter is annotated or listed in
   `do_not_specialize`; any other integer is a runtime `int32` if it lies in [-2^31, 2^31), a
   runtime `int64` if in [-2^63, 2^63), a runtime `uint64` if in [2^63, 2^64), and is rejected on
   the host otherwise. The argument's type thus depends on its value.
2. A multiple of 16 (including `0`) also gives the parameter `tt.divisibility = 16` (a fact the
   compiler may use), unless `do_not_specialize` or `do_not_specialize_on_alignment` lists it.
3. A `bool` is a runtime `int1` (`u1`).
4. An unannotated Python `float` (or `numpy.float64`) is a runtime `float32`: the double rounded to
   nearest-even, with overflow to ±inf, `-0.0` kept and NaN quiet with its sign.
5. An annotated scalar parameter takes the annotated type: `tl.float64` gets the double exactly;
   `tl.float16` gets the double rounded to nearest-even `float16`, and a value that rounds to ±inf
   (other than `inf` itself) is rejected on the host; `tl.bfloat16` gets the *truncation* of the
   nearest `float32` to its top 16 bits; `tl.int8` and `tl.int32` (and `int`) get the value modulo
   2^8 and 2^32 (as signed), `tl.uint32` the value modulo 2^32 (negative values rejected),
   `tl.int64` the exact value (out of range rejected); `tl.int1` given `2` gets `0`. (`tl.int16`,
   `tl.uint8`, `tl.uint16` and `tl.uint64` were not run; they use the extractors of the same
   kind, lines 1095-1128.) Annotated floats and bools are never specialised; annotated integers
   are never constants but still get `tt.divisibility = 16`.
6. numpy scalars other than `float64` are rejected on the host; a 0-d tensor is passed as a
   pointer to its data, not as its value; `None` is a compile-time constant.
