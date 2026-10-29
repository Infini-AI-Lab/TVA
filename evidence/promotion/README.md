# Operator types and dtype promotion

**Question.** For each operator a kernel can write on tensors and Python numbers, what dtype does
Triton 3.8.0's front end give the result, which operand pairs does it reject, and what does the
device compute? Library issue L45 (round 8): `/` on integers, `and`/`or`/`not` on tensors, the
promotion of mixed-dtype operators, `tl.where` with a condition that is not `int1`, Python's
builtin `min`/`max` in a kernel, and integer `tl.minimum`/`tl.maximum`.

**Method.**

- `table.py` (front end, exhaustive over its operand kinds, no GPU). It sets up the front end as
  `compile` does (an `ir.context` with the CUDA backend's dialects, options, `codegen_fns` and
  `module_map`, a `CodeGenerator` of `triton.compiler.code_generator`), opens a function body,
  binds the operands to `a`, `b` (and `c`) in the generator's scope and visits the parsed
  expression (`a + b`, `a < b`, `a and b`, `tl.minimum(a, b)`, `min(a, b)`, `tl.where(c, a, b)`,
  `-a`, ...), so the dispatch is the front end's own. Operands: a block tensor `[4]` (`t:`) and a
  scalar tensor (`s:`) of each of the 15 dtypes `int1 int8 int16 int32 int64 uint8 uint16 uint32
  uint64 fp8e5 fp8e4nv fp16 bf16 fp32 fp64`, the Python ints `0, 1, -1, 2^31-1, 2^31, -2^31-1,
  2^63, 2^64-1`, the float `1.5` and `True`: 40 operands. Operators: `+ - * / // % & | ^ << >>`,
  `< <= > >= == !=`, `tl.minimum`, `tl.maximum`, `and`, `or`, builtin `min`, `max` (all 1600
  ordered pairs each); unary `-`, `~`, `not`, `+` (40 each); `tl.where` with every condition and
  the values `(t:fp32, t:fp32)` and `(s:fp32, s:fp32)`, and with the conditions `t:int1`, `s:int1`
  and every pair of values. 40240 cases, in `table.tsv`: `<op> <lhs> <rhs> <result>`, the result
  `t:<dtype>`, `s:<dtype>`, `py:<value>` (a compile-time Python value; `py:big` for an int of 128
  bits or more) or `error` with the exception's class and first line.
- `rules.py` restates the rules below without Triton and compares them with every line of
  `table.tsv` (the result, and the class of each rejection).
- `device.py` (GPU 0, one process per case P1-P11) launches kernels that check their result's dtype
  with `tl.static_assert(z.dtype == Z)` and compares the values with references computed on the
  host: exhaustively over all `int8`/`uint8` pairs where that is 65536 cases, over edge values
  and random values for wider integers, over all `float16` values against chosen values of the
  other operand for mixed floats.

Run, from TVA's root:
`.venv/bin/python -I evidence/promotion/table.py > evidence/promotion/table.tsv`,
`.venv/bin/python -I evidence/promotion/rules.py`,
`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/promotion/device.py > evidence/promotion/device-output.txt`.

**Triton version.** 3.8.0 (torch 2.11.0+cu130); the cited sources are the pinned
`primitives/triton-3.8.0-semantic.py`, `-core.py` and `-code-generator.py`, identical to the
installed files.
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result.** `rules.py`: 40240 lines compared, 0 differ. `table.py` gives the same table on a second
run. `device-output.txt`: every value check agrees with the rules below; the device's choices
where LLVM leaves the result open are listed under "Undefined results".

## Conclusions

### 1. Dispatch (`code-generator.py`)

- A literal is a `constexpr` (`visit_Constant`, 1478-1479); `-1` is `constexpr(1)` negated.
- `a ∘ b` (`_apply_binary_method`, 795-808): if `a` is a tensor, `a.__op__(b)`; else if `b` is a
  tensor, `b.__rop__(a)`; else Python's own operator on the two values (the result is a
  compile-time value). Comparisons likewise (`visit_Compare`, 1076-1091).
- Unary (`visit_UnaryOp`, 1097-1114): a tensor's `__neg__`, `__invert__`, `__not__`; a tensor has
  no `__pos__`, so `+x` on a tensor is rejected (AttributeError). On a Python value: Python's.
- `a and b`, `a or b` (`visit_BoolOp`, 1481-1531), left to right: a Python value that decides
  the result (falsy for `and`, truthy for `or`) is returned as it is; any other Python value is
  dropped; if no tensor remains the last value is returned; otherwise the tensors are folded with
  `logical_and`/`logical_or`. So `x and True` is `x` itself, whatever its dtype, and `x and 0`
  is the constant `0`.
- `min`, `max` are `core.builtin_min`/`builtin_max` (357-364; `core.py` 3882-3929): Python's own
  when every argument is a compile-time value, else a fold of `tl.minimum`/`tl.maximum`.

### 2. Promotion (`semantic.py`)

A **Python number** operand (`isinstance(v, numbers.Number)`: an int, a float or a bool, not a
tensor of any shape) is first given its own type by `to_tensor_type` (129-158): `bool` → `int1`;
an int in `[-2^31, 2^31)` → `int32`, `[2^31, 2^32)` → `uint32`, `[-2^63, 2^63)` → `int64`,
`[2^63, 2^64)` → `uint64`, else rejected; a float → `fp32` if it is `±inf`, `±0`, NaN or of
magnitude in `[2^-126, (2-2^-23)·2^127]`, else `fp64`.

`binary_op_type_checking_impl` (175-210) computes the common type `T =
computation_type(ta, a_is_py, tb, b_is_py, div_or_mod)` (68-116), checked in this order:

0. exactly one operand is a Python number, and its kind is at most the tensor's kind
   (`int1` < other integers < floats; `core.py` `kind`, 524-532): `T` is the tensor's dtype, except
   `fp32` when `div_or_mod` and the tensor is `fp16` or `bf16`;
1. either is `fp64` → `fp64`;
2. either is `fp32` → `fp32`;
3. either is `fp16` → `fp32` if `div_or_mod`, else `fp16` (so `fp16` with `bf16` is **`fp16`**);
4. both `bf16` → `fp32` if `div_or_mod`, else `bf16`; one `bf16` → `fp32`;
5. both fp8: the same fp8 type if equal (even for `/` and `%`), else `fp16`;
6. otherwise, if either is not an integer (fp8 with an integer) → rejected (TypeError);
7. both integers: if `div_or_mod` and their signedness differs → rejected (TypeError); else
   `integer_promote` (53-66): equal signedness → the wider (the second on a tie); otherwise the
   unsigned one if its width is at least the signed one's, else the signed one. `int1` is
   unsigned of width 1.

Then (194-206): a Python operand that is negative while `T` is unsigned, or (for an integer `T`)
outside `T`'s range, is rejected (ValueError); a Python operand becomes the constant
`scalar_constant(v, T)`, a tensor is `cast` to `T` (811-916: integers sign-extend from a signed
type, zero-extend from an unsigned one or `int1`; integer to float rounds to nearest; float to float
is exact upward, to nearest downward). Shapes broadcast. A scalar tensor (a scalar argument, a
loaded scalar) is not a Python number: it takes part in the promotion like a block.

`div_or_mod` is set for `/`, `//`, `%` only. Each operator, after that:

| operator | result dtype | rejected when |
|---|---|---|
| `+ - *` (230-290) | `T` | the promotion rejects |
| `/` (292-315) | `T` if `T` is a float, else **`fp32`** (both integer operands converted to `fp32` first, `int64` too) | the promotion rejects (e.g. `int8 / uint8`) |
| `//` (317-329) | `T`; signed or unsigned division by `T` | `T` is a float |
| `%` (343-360) | `T`; `srem`/`urem`, or `frem` for a float `T` | the promotion rejects |
| `& \| ^ << >>` (415-469) | `T`; `>>` is arithmetic exactly when the operand whose method runs (the left one if it is a tensor, else the right one) has a signed dtype *before* promotion (`core.py` 1009-1026), so `x_int8 >> y_uint8` shifts the `uint8` value arithmetically (P9) | `T` is not an integer |
| `< <= > >= == !=` (501-577) | `int1` | the promotion rejects |
| `tl.minimum`, `tl.maximum`, builtin `min`/`max` on tensors | `T` | the promotion rejects |
| `and`, `or` on two tensors (440-452) | `int1` | an operand is not `int1` (its bitcast to `int1` fails) |
| `not x` (454-457) | `int1` | `x` is not `int1` |
| `-x` (478-485) | `x`'s dtype (`0 - x` for integers, `fneg` for floats) | |
| `~x` (487-492) | `x`'s dtype (`x ^ all-ones`) | `x` is a float |
| `tl.where(c, x, y)` (`core.py` 2837-2856, `semantic.py` 1638-1652) | `T` of `x, y` with the Python-number rule, broadcast with `c` | `c` is fp8 (no cast to `int1`), or the promotion of `x, y` rejects |

**Own-type rule.** Comparisons (`core.py` 1028-1095), `tl.minimum`/`tl.maximum` (2912-2954, and so
builtin `min`/`max` on tensors) and the condition of `tl.where` first make a Python number a tensor
of its own type (`to_tensor`), so step 0 never applies to them: `x_uint8 < -1` is computed in
`int32` (always false), `x_int8 < 200` in `int32` (always true), `x_uint32 == -1` in `uint32`
(`-1` is `2^32-1`), `x_int32 < 2**31` in `uint32` (false for negative `x`), `x_fp16 < 0.1` in
`fp32`, and `tl.minimum(x_uint8, -1)` is `int32`. The same operands in `+` are rejected
(`x_uint8 + (-1)`, `x_int8 + 200`) or stay in the tensor's dtype (`x_fp16 + 0.1` is `fp16`).
`tl.minimum` and `tl.maximum` also convert each `bf16` operand to `fp32` before the promotion
(`_promote_bfloat16_to_float32`, 3083-3089): `tl.minimum(bf16, bf16)` is `fp32`.

**Conditions.** `tl.where` casts a condition that is not `int1` to `int1` (with a deprecation
warning) by `c != 0` (`cast`, 874-877 and 883-886): for a float, `fcmp une`, so NaN is true and
`-0.0` false.

### 3. Device results (`device-output.txt`)

- P1: `/` on `int32`, `int64`, `uint32`, `int8` converts both to the nearest `fp32` first:
  `16777217 / 1 = 16777216`, `(2^53+1) / 1 = 2^53`.
- P2: `//` and `%` are C's truncating division and remainder, signed or unsigned by `T`: 0
  differences over all 65279 `int8` and 65280 `uint8` pairs with a nonzero divisor, and over edge
  pairs of `int16`, `int32`, `int64`, `uint32`, `uint64`.
- P3: `int32 + int64` sign-extends; `uint32 + int32` and `int8 + uint8` (all 65536 pairs) are
  unsigned, modulo `2^w`; `int1 + int1` is `int1` (true + true = false), `int1 + int8` zero-extends.
- P4: `fp16 + bf16` is `fp16(x + fp16(y))` (all 65536 `fp16` × 68 `bf16`: 0 differences; a
  `bf16` `1e38` becomes `inf`); `/` with them is `fp32`; `fp16 * fp32`, `fp8 + fp16`,
  `fp8e5 + fp8e4nv` (all pairs) are one IEEE operation after exact conversion; `int32 + fp32`
  and `int32 + fp16` round the integer to nearest (`65520` → `inf` in `fp16`).
- P5: an `int8` (all 256) or `int32` condition is `c != 0`; an `fp32` condition: `0.0`, `-0.0`
  false; NaN, `±1e-45`, `inf` true.
- P6: `and`/`or`/`not` on `int32` tensors are rejected on the host; on `int1` they are elementwise.
- P7: builtin `min` on `int32` arguments equals `tl.minimum`; integer `tl.minimum` is signed or
  unsigned by `T`; float `tl.minimum` (default `propagate_nan=NONE`) returns the other operand for
  one NaN and NaN for two, with `propagate_nan=ALL` NaN for any NaN; `min(-0, +0) = -0`,
  `max(-0, +0) = +0` in both orders.
- P8: the own-type rule's examples above, as stated.
- P9: shifts by `0 … w-1` are `x << s mod 2^w` and `x >> s` (arithmetic or logical as above) on
  `int8`, `uint8`, `int32`, `uint32`; `[-1, -128, 64]` (`int8`) `>> 1` (`uint8`) is
  `[255, 192, 32]`, arithmetic on the `uint8` values.
- P10: `-x` on `uint8` is `(-x) mod 256` (all 256); `-(-2^31) = -2^31`; `~` and `-` on `int1`.
- P11: `x_int32 * 0.1` is `fp32(x) * fp32(0.1)`, rounded.

**Rejected after the front end.** `fp8e5 + fp8e5` and `fp8e4nv + fp8e4nv` type-check as fp8
(step 5) but fail in the backend (`ConvertTritonGPUToLLVM`, "PassManager::run failed"): an
arithmetic operation on two operands of the same fp8 type is rejected on the host.

**Undefined results** (the device's choices; LLVM leaves these open, so a model should not rely on
them):
- `x // 0`, `x % 0`: all ones for every width (`-1`, `255`, `2^32-1`), but `0 // 0` in `int64` and
  `uint64` is `2^32-1`;
- `MIN // -1 = MIN`, `MIN % -1 = 0`;
- a shift by a tensor amount of at least the width, or negative: `<<` and logical `>>` give 0,
  arithmetic `>>` the sign; a constant amount of the width (`x << 32` on `int32`, which the front
  end only warns about, `core.py` 376-382) gave `16776640` for both `x = 1` and `x = 5`.

### 4. The library's definitions against the table (`lean_compare.py`)

`lean_compare.py` evaluates the library's definitions (`TVA.Lang.Promote`: `BinOp.typing?`,
`UnOp.typing?`, `whereTyping?`, `boolOp?`) in Lean on the operands of `table.tsv` and compares each
result with the front end's (`lean-compare-output.txt`): 40236 distinct cases, 37896 compared, **0
differ**; the 2340 others have only Python-number operands (Python's own arithmetic, which the
library does not model). Run it after `lake build TVA`:
`.venv/bin/python -I evidence/promotion/lean_compare.py`.

### 5. Constants in comparisons, `tl.where` and `tl.maximum` (`constants.py`)

`constants.py` (GPU 0; `constants-output.txt`) checks the values, not only the dtypes, of the
own-type rule:
- K1: with a `float64` tensor `x = 0.1`, `x < 0.1` is true and `x == 0.1` false: the constant is
  `float32(0.1)` widened (`arith.constant 1.0e-01 : f32`, `arith.extf`), not the double. In
  `tl.where(x > 0, 0.1, x)` the constant is the double (`arith.constant 1.0e-01 : f64`).
- K2: a `bfloat16` tensor compared with `1e-7` is converted to `float32` and compared with
  `float32(1e-7)`, not with the `bfloat16` constant an arithmetic operator would build (`+0`).
- K3: `x < 2**31` on an `int32` tensor compares unsigned (`arith.cmpi ult`): false for `-1`.
- K4: `tl.maximum(x, 0.1)` with a `float16` tensor is `float32`, and its constant `float32(0.1)`.
