# Front-end forms of round 32

**Question.** Library issues L151 (a pointer minus an integer), L152 (the predicates of a dtype, such as
`is_fp8()`, and the conversion of a value stored to an fp8 pointer), L147 (which float conversions take a
`fp_downcast_rounding`) and L148 (a block's `.type` where a dtype goes): what does Triton 3.8.0's front end
build for each, and what does the H200 compute?

**Sources** (pinned in `primitives/`). `triton-3.8.0-semantic.py`: `sub` (261-276: `ptr - offset` is
`add(ptr, minus(offset))`), `minus` (478-485: `0 - x` in `x`'s own type), `add` (230-259: an unsigned offset
narrower than 64 bits is zero-extended, another is taken signed, then `tt.addptr`), `cast` (811-916: a rounding
mode only on a float conversion to fewer bits; a block's target type is replaced by its element type when the
input is a block), `store` (1205-1243: the value is converted with `cast`). `triton-3.8.0-core.py`: the dtype
names (391-394: `int1` is among the unsigned types) and predicates (449-524).

**Method** (`run.py`; each case in its own process with an empty Triton cache).
- S1: `p - i` with `i` a runtime `int32` argument (`0, 1, -3, 7, 2^31 - 1, -2^31`; `1` is a compile-time
  constant), a loaded `int32`, `int64` and `uint32` scalar, a Python literal (`3`, `2**40`), an `int32` block, and
  a block of pointers minus a block, on `float32`, `float16` and `float64` buffers: the address, read with
  `.to(tl.int64)`, against `p + (−i)·size` with `−i` wrapping in `i`'s type; and the TTIR.
- S2: `i - p`, `p - q`, `p - 1.0`, `1 - p`.
- S3: a load and a store through `cu + req - 1` (vLLM's `cu_num_draft_tokens_ptr + req_idx - 1`).
- T1: 23 predicates (`is_fp8`, `is_fp8e4nv`, `is_fp8e5`, `is_fp16`, `is_bf16`, `is_fp32`, `is_fp64`,
  `is_floating`, `is_standard_floating`, `is_int`, `is_int_signed`, `is_int_unsigned`, `is_bool`, `is_int1`,
  `is_int8` … `is_uint64`, `is_ptr`) on a loaded block of each of 13 dtypes, used in `tl.static_assert`, in a
  compile-time `if` and in a conditional expression, against the predicate on the dtype's name (299 kernels).
- T2: `y = x if x.dtype.is_fp8() else x / s` stored to a `float8e4nv` and a `float8e5` pointer from a `bfloat16`
  `x` (with `±10^6` and `±0`), and from a `float8e4nv` `x`: the stored bits against the saturating rounding to
  nearest even of the `float32` value.
- R1: `fp_downcast_rounding="rtne"` and `"rtz"` on `bfloat16 → float16`, `float16 → bfloat16` (the same width),
  and on `float32 → float16`, `float32 → bfloat16` (a control).
- Y1: a block's and a scalar's `.type` where a dtype goes: `x.to(y.type)`, `v.to(x.type)`, `v.to(y.type)`,
  `v.to(z.type)`, `tl.zeros([16], dtype=x.type)`, `tl.zeros([16], dtype=v.type)`, `x.to(y.type.element_ty)`,
  `x.to(v.type)`, `tl.full([16], 1, x.type)` (`x` a `float32` block of 16, `y` an `int32` block of 32, `z` an
  `int32` block of 16, `v` a loaded `float32` scalar).

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-round32/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130). **GPU.** NVIDIA H200 NVL (sm_90). **Date.** 2026-11-08.

**Result** (`output.txt`).
- S1: every address equals `p + (−i)·size`, `−i` computed in `i`'s type: for a signed `i`, `i` elements back,
  except `−2^31` in `int32`, which is its own negation (`p − 2^31·size`); for a `uint32` `i`, `2^32 − i` elements
  forward (`i = 1`, `2^31`); a literal is typed as a constant is (`p - 2**40` is an `int64` offset); blocks lane by
  lane. The TTIR is `arith.subi` from `0`, then `tt.addptr`.
- S2: all four are rejected on the host (`IncompatibleTypeErrorImpl`).
- S3: `cu + req - 1` read `cu[req − 1]` and wrote `out[req − 1]` for `req = 1, 2, 3`.
- T1: 0 of 299 kernels differ: each predicate is a compile-time `bool`, the predicate on the dtype's name
  (`is_fp8()` is `'fp8' in name`, true of `float8e4nv` and `float8e5`; `is_int_unsigned()` is true of `int1`,
  which is among Triton's unsigned types, and `is_int_signed()` false of it; `is_ptr()` false of every element
  dtype).
- T2: 0 of 1024 stored bits differ from the saturating rounding to nearest even (`10^6` stored as `448` in
  `float8e4nv` and `57344` in `float8e5`), and the `is_fp8()` branch stored `x` itself bit for bit.
- R1: on `bfloat16 → float16` and `float16 → bfloat16` both `"rtne"` and `"rtz"` are rejected on the host
  ("fp_downcast_rounding should be set only for truncating fp conversions"); on `float32 → float16` and
  `float32 → bfloat16` both are accepted.
- Y1: accepted: `x.to(y.type)` (converts to `y`'s element dtype, `int32`, whatever `y`'s shape: `1.0`),
  `v.to(x.type)` and `x.to(v.type)` (the same dtype: unchanged), `tl.zeros([16], dtype=v.type)`,
  `x.to(y.type.element_ty)`. Rejected on the host: `v.to(y.type)` and `v.to(z.type)` (a scalar to a block type of
  another dtype), `tl.zeros([16], dtype=x.type)` and `tl.full([16], 1, x.type)` (a block type as a dtype).

**Conclusions**, used by `docs/SEMANTICS.md` ("Pointers", "A dtype's predicates", "`x.to(d)`, `tl.cast(x, d)`",
"A tensor's `.type`") and `lean/TVA/Lang/Mem.lean` (`ptr - k`):
1. `p - i` is `p + (−i)`, the negation wrapping in `i`'s own type, then added as `+` adds (L151).
2. A dtype's predicates are compile-time `bool`s given by the dtype's name; a value stored to an fp8 pointer is
   converted by the saturating rounding to nearest even (L152).
3. A rounding mode is accepted only on a float conversion to a float of fewer bits (L147).
4. Only a dtype, a scalar's `.type` and a block type's `.element_ty` stand for a dtype everywhere; a block's
   `.type` is rejected as `tl.zeros`'s or `tl.full`'s dtype and as a scalar's target, and as the target of a
   block's `.to` it means its element type (L148).
