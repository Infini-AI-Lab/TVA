# A dtype as a `tl.constexpr` argument, `tl.broadcast_to`, a pointer's `.type` and `if p:`, `tl.cast`

Library evidence (not a kernel's), for library issues L139, L140, L142 and L143. Round 31.

**Question.** Writing TVA's fixed claims (`claims/`) found four forms of real kernels that SEMANTICS.md did not
state:
- L139: a dtype as the argument of a `tl.constexpr` parameter (vLLM's `scaled_mm_kernel` takes
  `ACCUMULATOR_DTYPE: tl.constexpr`, passed `tl.int32`, used as `tl.zeros`'s `dtype=` and `tl.dot`'s
  `out_dtype=`; SGLang's `_bmm_chunk_fwd_kernel` `dot_dtype`, the target of `.to(dot_dtype)`). Does the
  parameter mean the dtype wherever it is used? Which arguments does the launcher accept?
- L140: `tl.broadcast_to` and its method form (`scale_a.broadcast_to((BLOCK_SIZE_M, 1))`). What does it give,
  and which shapes does Triton reject?
- L142: `c_ptr.type.element_ty` (the subset listed `.dtype` and a dtype's `.element_ty`), and `if bias_ptr:` on a
  pointer argument that may be `None` or a tensor.
- L143: `tl.cast(x, dtype=…)` (SGLang's `_moe_sum_reduce_kernel` writes `tl.cast(input_stride_0,
  dtype=tl.int64)`), its relation to `.to`, `fp_downcast_rounding` and `bitcast=`.

**Sources** (Triton 3.8.0, pinned in `primitives/`). `triton-3.8.0-core.py` 667-705 (`pointer_type`, its
`element_ty`), 707-756 (`block_type`: `element_ty` is the block's scalar type), 884-890 (`tensor.type`,
`tensor.dtype = type.scalar`), 1130-1135 (`tensor.to` calls `cast`), 2049-2066 (`broadcast_to`), 2322-2345
(`cast`: `to_tensor`, then `semantic.bitcast` or `semantic.cast`); `triton-3.8.0-semantic.py` 629-634 (`splat`),
711-725 (`broadcast_impl_shape`), 793-809 (`bitcast`), 811-916 (`cast`; 900-906, a pointer to `int1` is
`ptrtoint` then `!= 0`); `triton-3.8.0-code-generator.py` 967-996 (`visit_If`: a tensor condition is converted
`.to(tl.int1)`), 1535-1544 (`get_Attribute`); `triton-3.8.0-utils.py` 63-74 (`validate_block_shape`);
`triton-3.8.0-specialize.cc` 515-560 and `triton-3.8.0-jit.py` 396-410 (a `tl.constexpr` parameter's argument
is passed as it is; another parameter's argument is specialised by its type, and a dtype has no handler);
`triton-3.8.0-nvidia-driver.c` 1055-1078 (a pointer whose `data_ptr()` is `0` is passed as the null pointer).

**Method** ([run.py](run.py), output [output.txt](output.txt)). Each case runs in its own process with its own
empty Triton cache; D1 and D2 write each kernel to a file of its own in a temporary folder and import it from
there. "The same TTIR" is the compiled `ttir` text with its source locations removed.
- D1: a kernel with `D: tl.constexpr` using `D` in `.to(D)`, `tl.cast(x, D)`, `tl.zeros([16], dtype=D)`,
  `tl.full([16], 3, D)`, `d = D` then `.to(d)`, `tl.static_assert(y.dtype == D)` and `if D == tl.float32:`,
  against the same kernel with the literal dtype (`tl.int32`, …) written in each place, for `int8`, `int32`,
  `int64`, `float16`, `bfloat16`, `float32`, `float64`: the TTIR, and the stored values.
- D2: as D1, `acc = tl.zeros([32, 32], dtype=D); acc = tl.dot(a, b, acc, out_dtype=D)`, for `int8` operands
  into `int32`, `float16` into `float32` and into `float16`.
- D3: `D = tl.int16` given by position and by keyword.
- D4: one process, the same kernel storing `x.to(D)` of `x = 1.5`, launched with `tl.int32`, `tl.float32`,
  `tl.int32`.
- D5: `tl.int32` passed to a parameter without annotation; `torch.int32` passed to a `tl.constexpr` parameter
  used as `.to(D)`.
- B1: `tl.broadcast_to(x, (T0, T1))`, `x.broadcast_to((T0, T1))` and `x.broadcast_to(T0, T1)` of loaded rank-2
  blocks, seven pairs of shapes (`[1, 8]`, `[4, 1]`, `[1, 1]`, `[4, 8]` to `[4, 8]`; `[1, 1]` to `[1, 1]`;
  `[16, 1]`, `[1, 1]` to `[16, 1]`), and the method on rank-3 blocks (`[1, 4, 1]`, `[2, 1, 8]`, `[1, 1, 1]`
  to `[2, 4, 8]`), against torch's `expand`.
- B2: a loaded scalar, and a runtime `float32` argument, broadcast to `[4, 8]`.
- B3: `[4, 1]` to `[8, 1]` and to `[2, 1]`; `[4]` to `[4, 4]`; `[4, 1]` to `[4]`; `[4, 1]` to `[4, 3]`;
  `tl.broadcast_to(1.0, (4,))`.
- P1: for a pointer argument `c_ptr` of eight dtypes, `tl.static_assert(c_ptr.type.element_ty ==
  c_ptr.dtype.element_ty)` and `tl.static_assert(c_ptr.type == c_ptr.dtype)`, and the store of
  `x.to(c_ptr.type.element_ty)` (the float ones compared with torch's `.to`).
- P2: `x.type.element_ty == x.dtype == tl.float32` for a `float32` block; `ptrs.type.element_ty == ptrs.dtype
  == x_ptr.type` and `!= tl.float32` for a block of pointers; `n.type.element_ty` of an `int32` argument.
- P3: `tl.store(o_ptr, 1); if b_ptr: tl.store(o_ptr + 1, 2)` with `b_ptr` a tensor, an empty tensor
  (`data_ptr()` is `0`), and `None` (with `tl.static_assert(False)` in the branch); the tensor's TTIR.
- C1: `tl.cast(x, dtype=D, fp_downcast_rounding=R, bitcast=BIT)`, `x.to(D, …)` and `x.cast(D, …)` on 24
  conversions (float narrowings with `"rtne"` and `"rtz"`, widenings, float to integer, integer to float,
  integer to integer and `int1`, into fp8, to the same dtype with `"rtz"`, and six bit casts): the TTIR.
- C2: `tl.cast(s, dtype=tl.int64)` of a scalar argument `s` (`1`, `7`, `2^40`), `tl.cast(1, tl.int64)`,
  `tl.cast(2**40 + 5, tl.int32)`, `tl.cast(1.75, tl.int32)`, `tl.cast(-2.5, tl.int32)`.
- C3: `fp_downcast_rounding=` on `float16` to `float32` (`"rtne"`), `float32` to `float64`, `int32` to
  `float32`, `float32` to `int32`, `float32` to `float32` (`"rtz"`), `float16` to `float16` (`"rtne"`),
  `float32` to `float16` (`"rtne"`, and `"rd"`).
- C4: `x.to(D, bitcast=True)` of a loaded block, stored and compared bit for bit with the input: every pattern
  of seven 8-bit pairs (`int8`, `uint8`, `float8e4nv`, `float8e5`) and six 16-bit pairs (`float16`, `int16`,
  `bfloat16`, `uint16`), and `2^24` random patterns with edge patterns (zeros, infinities, NaNs with payloads)
  of four 32-bit and three 64-bit pairs; and three pairs of different widths.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round31/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-11-08.

**Result** (`output.txt`).

| case | result |
|---|---|
| D1 | for all seven dtypes: the same TTIR and the same values as with the literal dtype |
| D2 | for the three dots: the same TTIR and the same values |
| D3 | by position and by keyword: the same TTIR and values |
| D4 | `1.0`, `1.5`, `1.0`: each launch its own dtype's result |
| D5 | a dtype to a parameter without annotation: rejected on the host (`failed to specialize argument of type: dtype`); a `torch` dtype used as a dtype: rejected (`'torch.dtype' object has no attribute 'scalar'`) |
| B1 | 10 pairs of shapes, rank 2 in three forms and rank 3: 0 differ from `expand` |
| B2 | a loaded scalar and a runtime argument: every element the scalar |
| B3 | all six rejected on the host: "the expanded size … must match the existing size … at non-singleton dimension" (`[8, 1]`, `[2, 1]`), "rank mismatch" (both rank changes), "Shape element 1 must be a power of 2", and for the Python number `'constexpr_type' object has no attribute 'is_block'` |
| P1 | for all eight dtypes both assertions hold, and the float stores equal torch's `.to` |
| P2 | the first two accepted (the assertions hold); `n.type.element_ty` rejected on the host (`'dtype' object has no attribute 'element_ty'`) |
| P3 | a tensor: the branch ran; an empty tensor: it did not; `None`: no error (the branch was not visited) and it did not run; the TTIR has `tt.ptr_to_int` to `i64`, `arith.cmpi ne` with `0` and `scf.if` |
| C1 | 0 of 24 differ (the bit cast of `float32` to `float16` is rejected in all three forms) |
| C2 | `1`, `7`, `1099511627776`; `1`; `5`; `1`; `-2` |
| C3 | the four conversions that are not float narrowings rejected (`fp_downcast_rounding should be set only for truncating fp conversions`), `"rd"` rejected (`Invalid rounding mode`); the two to the same dtype and `float32` to `float16` with `"rtne"` accepted |
| C4 | 0 patterns differ in all 20 pairs; the three pairs of different widths rejected (`Cannot bitcast data-type of size …`) |

**Conclusions**, used by `docs/SEMANTICS.md`:

1. A `tl.constexpr` parameter given a dtype of `tl` holds that dtype and means, at each use, what the dtype
   written there means (D1–D3), and launches with different dtypes compile apart (D4). A dtype to a parameter
   without annotation, and a `torch` dtype used as a dtype, are rejected on the host (D5). ("A dtype as a
   `tl.constexpr` argument", "Parameters and arguments".)
2. `tl.broadcast_to(x, shape)` and its two method forms repeat each size-1 dimension of a block (`expand`), and
   of a scalar give the scalar at every index (B1, B2); a target that is not of the same rank, or differs from
   the block in a dimension of size other than 1, or is not a power of two, is rejected on the host, and so is
   a Python number (B3). (`Tensor.broadcastTo`, `Shape.bcastOk`; "`tl.broadcast_to`".)
3. A tensor's `.type` is its dtype for a scalar and its block type for a block, whose `.element_ty` is the
   block's dtype; so `p.type.element_ty` of a pointer argument is the dtype of its buffer, and of a block of
   pointers is the pointer type; `.element_ty` of a non-pointer scalar's type is rejected (P1, P2). ("Values",
   "A tensor's `.type`".)
4. `if p:` on a pointer is a runtime `if` on `p != 0` (as a 64-bit address); a pointer given `None` is the
   compile-time `None` (P3). ("A runtime `if`".)
5. `tl.cast(x, d, r, b)`, `x.to(d, r, b)` and `x.cast(d, r, b)` are one operation (C1); `tl.cast` makes a Python
   number a tensor of its own type first (C2); `fp_downcast_rounding` is accepted only on float narrowings and to
   the same dtype (C3); `bitcast=True` reads each element's bits in the target dtype, exactly, and needs the same
   width (C4). ("`x.to(d)`, `tl.cast(x, d)` and `x.cast(d)`"; `tl.bitcast`.)
