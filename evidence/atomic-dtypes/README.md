# The element types of `tl.atomic_*` (library issue L168)

Library evidence (not a kernel's), for library issue L168 part (2) (from library review lr-36b,
round 36). Triton 3.8.0 (TVA's `.venv`, torch 2.11.0+cu130, numpy 2.3.5), NVIDIA H200 NVL (sm_90),
GPU 0, 2026-11-08. Script: [run.py](run.py); output: [out.txt](out.txt) (about a minute). Round 38
(library issues L172, L173, from library review lr-37b's `work/atomic_adv.py`, round 37): script
[adv.py](adv.py), output [adv.out.txt](adv.out.txt) (GPU 0, 2026-11-08, about a minute; identical to
lr-37b's output).

**Question.** docs/SEMANTICS.md, "Atomics", gave each `tl.atomic_*` a meaning on every element
type of the library (`addUpd`, `maxUpd`, `bitUpd` in `lean/TVA/Lang/Ops.lean`) and said that
`atomic_and`/`or`/`xor` are integer-only, but not which element types Triton accepts. Which does it
accept, which does it reject, and what do the accepted ones compile to?

**The source.** `semantic.py:1257-1263`: `atomic_cas` accepts an element of 16, 32 or 64 bits and
raises `ValueError` otherwise. `semantic.py:1265-1290` (`atom_red_typechecking_impl`, used by every
other op): `float16` and `bfloat16` are rejected for every op but `add`; `int16`, `uint16` and every
type narrower than 16 bits (`int1`, `int8`, `uint8`, the `float8` formats) for every op. Nothing
there rejects a `float32` or `float64` element for `and`, `or` or `xor`.

**Method** (`run.py`). For each op (`add`, `max`, `min`, `and`, `or`, `xor`, `xchg`, `cas`) and each
pointer element type a torch tensor argument gives (`bool`, `int8`, `uint8`, `int16`, `uint16`,
`int32`, `uint32`, `int64`, `uint64`, `float8_e4m3fn`, `float8_e5m2`, `float16`, `bfloat16`,
`float32`, `float64`): one tensor call of 64 lanes (default `sem` and scope), each lane on its own
cell, its result stored. For an accepted launch: the PTX atomic instructions; the lanes whose
returned value is not the cell's old bit pattern; the lanes whose new value is not the expected
one (integers wrap, lane 0 adds 1 to the maximum; small integers for floats, so every float result
is exact; a float `and`/`or`/`xor` against the operation on the bit patterns; `cas` compares with
the old value on even lanes and with another value on odd lanes).

**Result** (`out.txt`).
- Rejected (a `CompilationError` from the front end, on the host): every op on `bool`, `int8`,
  `uint8`, `float8_e4m3fn`, `float8_e5m2`; every op but `cas` on `int16`, `uint16`; every op but
  `add` and `cas` on `float16`, `bfloat16`.
- Accepted: `add` on `int32`, `uint32`, `int64`, `uint64` (`add.u32`, `add.u64`), `float16`,
  `bfloat16` (`add.noftz.f16`, `add.noftz.bf16`), `float32`, `float64`; `max`, `min` on the four
  32- and 64-bit integers (`.s32`, `.u32`, `.s64`, `.u64`) and on `float32`, `float64` (two
  predicated integer atomics, library issue L164); `and`, `or`, `xor`, `xchg` on the four integers
  and on `float32`, `float64` (`and.b32`, …, `exch.b64`); `cas` on `int16`, `uint16`, `float16`,
  `bfloat16` (`cas.b16`) and on the four integers and `float32`, `float64` (`cas.b32`, `cas.b64`).
- In every accepted case the returned value is the old bit pattern on 64 of 64 lanes, and the new
  value is the expected one on 64 of 64; a float `and`, `or` or `xor` acts on the bit patterns.

**Conclusions.** Triton 3.8.0 accepts `atomic_add` on the 32- and 64-bit integers and on `float16`,
`bfloat16`, `float32`, `float64`; `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`,
`atomic_xor` and `atomic_xchg` on the 32- and 64-bit integers and on `float32`, `float64`;
`atomic_cas` on every 16-, 32- and 64-bit element. It rejects every other element type on the host,
so such a launch has no outcome. `atomic_and`, `atomic_or` and `atomic_xor` on `float32`/`float64`
compile to the integer operation on the bit patterns; the library gives them no meaning (`bitUpd`
has no new value for a float, so the model's atomic is undefined behaviour), and
[docs/exclusions.md](../../docs/exclusions.md) excludes them. The library's meanings on the
rejected types (`addUpd` on 8- and 16-bit integers, `maxUpd` on 16-bit floats, …) are never used.

**Round 38: operands, SASS and scalars** (`adv.py`, library issues L172, L173).
- *Question.* SEMANTICS.md "Atomics" did not say that an atomic's operand is converted to the cell's
  element type, nor that `atomic_cas` converts nothing; review-questions.md Q7 described pre-sm_90
  16-bit atomics. What do the accepted cases compile to in SASS, and what happens to operands of
  another type than the cell?
- *The source.* `semantic.py:1280`: `atom_red_typechecking_impl`, used by every op but `cas`,
  converts the operand with `self.cast(val, ptr.type.scalar.element_ty)` (`semantic.cast`'s default
  rounding, as `tl.store`); `semantic.py:1257-1263` and `core.py:2746-2751`: `atomic_cas` makes
  tensors of `cmp` and `val` and converts neither.
- *Method.* S: the PTX and SASS atomics (`cuobjdump -sass`) of a `float16` and `bfloat16` add, a
  `float32` `atomic_max`, a `float64` `atomic_and`, a `float16` `atomic_cas` and a `float32` add, 64
  lanes. U: each op with its result unused, the new values on 64 lanes. C: scalar atomics (one
  pointer, one value) on each of the 15 element types, accepted or rejected. V: a `float32` operand
  added into zeroed `float16`, `bfloat16` and `int32` cells (lane 0 `65520`, lane 1 `1 + 2^-11`, the
  rest random), against torch's conversion; `atomic_cas` with operands of another type than the cell
  (Python `int`s on `int64` cells, scalar and tensor pointer; Python floats on `float16` cells;
  `int64` values on `int32` cells). B: `float32` `atomic_cas` on 36 pairs of `±0`, NaNs and `1.0`.
- *Result* (`adv.out.txt`). S: `float16`/`bfloat16` add and `float16` cas are `ATOMG.E.CAS` (a
  compare-and-swap loop, evidence/half-atomics); `float32` max is `ATOMG.E.MAX.S32` and
  `ATOMG.E.MIN`; `float64` and is `ATOMG.E.AND.64`; `float32` add `ATOMG.E.ADD.F32.FTZ.RN`. U: 0 of 64
  new values wrong for every op. C: scalar atomics accept and reject the same types as tensor ones.
  V: the converted operands matched torch's conversion (round to nearest even into `float16` and
  `bfloat16`, `65520` to `+∞`; truncation into `int32`) on 64 of 64 lanes each; the four mismatched
  `atomic_cas` calls were rejected on the host ("error encountered during parsing"; the
  compiler's own message, checked by hand for the first case, is "'tt.atomic_cas' op failed to verify
  that ptr type matches cmp type"). B: the new value is the bitwise
  compare-and-swap on 64 of 64 lanes and the returned value the old one; `+0` against `-0` does not
  swap, a NaN against the same NaN bits swaps.
- *Conclusions.* The operand of every `tl.atomic_*` but `atomic_cas` is converted to the cell's
  element type with `semantic.cast`'s default rounding, as `tl.store`'s value is; `atomic_cas`
  converts nothing, so operands of another type than the cell are rejected on the host and such a
  launch has no outcome (SEMANTICS.md "Atomics", "The operand"). `atomic_cas` compares bit patterns.
