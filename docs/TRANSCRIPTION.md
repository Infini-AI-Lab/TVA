# Transcription rules

How a Triton source becomes a TVA model. This page is part of the library:
- the **kernel prover** follows it;
- the **kernel reviewer** checks the model against the source by it, and trusts that each rule is
  what Triton 3.8.0 does;
- **library review** checks that ([LIBRARY_REVIEW.md](LIBRARY_REVIEW.md)).

What each `tl.*` operation means is in [SEMANTICS.md](SEMANTICS.md).

A construct this page does not cover cannot be transcribed yet. The prover asks the library for
a rule, and a kernel reviewer who finds one files `UNSUPPORTED`. A rule marked **(open: L<n>)** is
under a library issue ([library-review/ISSUES.md](library-review/ISSUES.md)): a kernel that needs
it waits for the issue's resolution.

## 1. The model's shape

- The kernel is a `Kernel` whose `body : Ctx → Prog Unit` is a `do` block. It keeps the source's
  line structure and names, one model line per source line where possible.
- A `tl.constexpr` argument is a Lean parameter (`BLOCK_SIZE : Nat`). A branch on constexprs
  that the covered configuration does not take is dropped (`config`, with the condition).
- An inlined `@triton.jit` helper is transcribed where it is called, or as a Lean function the
  call site applies.
- Every operation is a `TVA.tl.*` or another `TVA.*` definition. The model declares nothing in
  the `TVA` or `tl` namespaces and no instance of a library class (`SortKey`, `HintVal`, …): such
  a declaration would change an operation's meaning without changing the model's text.
- An operation missing from `TVA.Lang.Ops` is not defined in the kernel: the library adds it.

## 2. Values and types

- **Implicit conversions are written.** Dtype promotion, `.to(...)`, scalar–tensor arithmetic
  and broadcasting are written out (`tl.cvt`, `bc`, `.col`/`.row`).
- **Conversions** use the operation for their formats:
  - `tl.cvt` between the standard float formats, with the source's `fp_downcast_rounding`;
  - `tl.cvt8` into fp8;
  - `tl.cvtFrom8` out of fp8;
  - `tl.itof` from integers.
- **A narrowing of an integer-to-float value is one `tl.itof` to the final format.** That covers
  an explicit chain `x.to(f32).to(bf16)`, and also the implicit conversion when such a value is
  stored into a narrower buffer, atomically added, or stored through a descriptor.
- **`>>`** is `tl.ashr` when the operand Python dispatches on is signed *before* promotion, and
  `tl.lshr` otherwise. That operand is the left one when it is a tensor, and the right one when
  the left is a Python scalar. Shift amounts have the promoted type.
- **Literals.** A float zero literal is `+0.0`, even when the source writes `-0.0`. A Python
  float constant that fits `float32` is the nearest `float32` (`FloatV.ofPat`, with the bit
  pattern in a comment).
- **Index arithmetic** is `int32` where the source's is (`program_id`, `arange`, scalar
  arguments below `2^31`), and `int64` where the source casts.
- **Pointers.** Element types match the buffers. A pointer built from a loaded integer
  (`.to(tl.pointer_type(…))`) is `tl.intToPtr`.
- **Float operations are `Prog`s** (`let y ← tl.neg x`), because a NaN result may be any NaN.

## 3. Memory

- **`mask` and `other` are the source's.** A masked load with no `other` leaves the disabled
  lanes unspecified, and the model never uses them as `0`.
- **Broadcasting.** `[:, None]`, `[None, :]` and scalar–tensor operations keep their axes.
- **Block pointers.** `tl.make_block_ptr` takes each `shape`/`strides`/`offsets` entry cast to
  `int64`, with the source's `block_shape`, `boundary_check` and `padding_option` (the default
  padding is undefined).
- **Descriptors (TMA).** A `TensorDesc` has the shape, strides, `block_shape` and padding the
  kernel builds or the host's `TensorDescriptor` gives. Building a descriptor inside the kernel
  (`tl.make_tensor_descriptor`) is **(open: L5)**.
- **Atomics** keep the source's `sem` and `scope`. The result of a masked-off atomic lane never
  decides control flow or an address.
- **Exclusions.** No construct of [exclusions.md](exclusions.md) is used. In particular: no
  `inline_asm_elementwise` replaced by "equivalent" code, no pointers to pointers, no access to the
  same bytes through both pointers and descriptors.

## 4. Float addition: IEEE or fusable

An addition, subtraction, reduction or scan that a product may feed is written in its fusable
form. A product is an operand computed by `*`, `tl.dot` or `tl.rand`, through any dataflow,
inlined helpers and library functions included. Otherwise the IEEE form is used.

| IEEE form | fusable form |
|---|---|
| `tl.add` | `tl.mul_add a b c` for `a * b + c`; `tl.addc` otherwise |
| `tl.sub` | `tl.subc` |
| `tl.sum`, `sum1`, `sum0` | `tl.sumc`, `sum1c`, `sum0c` |
| `tl.cumsum` | `tl.cumsumc` |
| an `AddRel false` scan relation | `AddRel E.fusion` |

The launch's `enable_fp_fusion` is the source's (Triton's default: `true`).

## 5. Shapes the compiler rewrites

These are transcribed in their rewritten form, whatever `enable_fp_fusion` says:

- `tl.dot(a, b) + c` or `c + tl.dot(a, b)`, where the dot's accumulator is a constant zero and
  the product has no other use, is `tl.dot_add` (with the source's zero as `z`).
- The same for `tl.dot_scaled`: `tl.dot_scaled_add`.
- A float `tl.sum(x[:, :, None] * y[None, :, :], axis=1)` is `tl.sum_bcast_mul`.
- `tl.reshape(x, s, can_reorder=True)` and `tl.view` are `tl.reshape_any`. So is a reshape to
  rank 1 whose only users are reductions or histograms. No index (`tl.argmax`, a
  `(value, index)` reduction) is paired with such a reshape as if the order were kept.

## 6. Reductions, `tl.dot`, sorting, scans

- **Reductions** keep the source's axis.
- **`tl.dot`** keeps the source's `out_dtype` and `input_precision`, with the resolved
  `max_num_imprecise_acc` in its `DotCfg`: the source's value, or `2^30` for an fp8 × fp8 dot on
  sm_90 and `0` otherwise.
- **`tl.dot_scaled`.** Its `ScaledCfg` has the source's formats, `k_pack` flags and
  `fast_math`:
  - `lhsScale`/`rhsScale` are `e8m0` for `uint8`/`int8` scale tensors, and `float d` for float
    ones;
  - `group` is `K` divided by the scale tensors' last dimension (`16` or `32`);
  - the operands are the storage tensors as they are (packed fp4 as `uint8`).
- **`tl.sort`, `tl.histogram`.** `descending` and the dimension (the last) match; `bins` and
  `mask` match.
- **Scans.**
  - `combine_fn` is written as a relation that covers every result its body can give. For float
    bodies, use the library's relations: `AddRel`, `NaNEq`, `MaxRel`. For integers, write
    `fun a b c => c = a + b`.
  - The axis and `reverse` match.
  - With `reverse=True` the left operand of `combine_fn` covers the higher indices, so a
    non-commutative relation is stated that way.
  - The promotions `tl.cumsum`/`tl.cumprod` do are written.

## 7. Control flow

- `range(lo, hi, step)` is `tl.forRange`, and a fixed count is `tl.forN`. A loop's carried
  variables form a tuple state, and the induction variable has the promotion of the bounds'
  types.
- A `range` with unsigned bounds must satisfy `tl.RangeOk` (a signed range).
- `tl.static_range` is unrolled with the same count.
- A `while` loop is `tl.whileLoop`, with the source's condition.
- `and`/`or` of scalar tensors evaluate both sides.

## 8. Compiler hints

`tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy` and `tl.assume` are checks, not
no-ops:
- **Where.** The check sits where the hinted value is defined, before any use of it. Triton
  applies a hint to the value everywhere, also before the call and on paths that never reach it.
- **Rank ≥ 2.** One value per dimension, in the source's order. What a per-dimension value
  promises, and the `contig` argument of `multiple_of`, are **(open: L2)**.
- **Inside a loop body.** A hint on a name the loop carries, or on its induction variable, is
  **(open: L1)**. The earlier rule that dropped such a hint as a no-op is withdrawn: Triton
  applies it to the value the name has before the loop.

## 9. The launch

- `launch d` gives the host's grid formula, and `program_id` reads the right axis.
- `blocks` lists every block shape of the source on the instance. `Launch.Valid` checks only the
  shapes listed, so an empty or partial list passes it.
- `num_ctas` is the source's (`1`), and `debug` is `false`.
- With `@triton.autotune`, `Inst` ranges over the configs the autotuner may pick.

## 10. What may be dropped

The model map's drop reasons ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §7). Anything else that
is dropped is a mismatch.

| Reason | Source |
|---|---|
| `no-op:debug_barrier` | `tl.debug_barrier()` |
| `no-op:print` | `tl.device_print`, `tl.device_assert`, `tl.static_print` |
| `no-op:cache` | cache, eviction and `volatile` modifiers |
| `no-op:range-kw` | `tl.range` scheduling keywords (`num_stages`, `loop_unroll_factor`, `warp_specialize`, …) |
| `decl` | a decorator or signature line, covered by the model's definition |
| `config` | a branch not taken in the covered configuration (the note names the condition) |
| `compile-check` | `tl.static_assert` that holds on every covered instance (the note names the condition) |
| `dead` | a pure value no covered path uses (the note says why) |
| `comment` | a comment or blank line inside a range |
