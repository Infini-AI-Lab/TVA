# Review of vec_add_f32, round 2, by r-vecadd

## Verdict

accept
I judged the claim in full at gate `401e1fa92db421d7`, as if for the first time. I read the
source, the whole Lean file, the map, the write-up and the library pages the claim reaches. The
model is the source line by line under SEMANTICS.md: `program_id` on axis 0; the constexpr
`BLOCK_SIZE` built in `int32` for the wrapping `int32` product; the scalar–tensor `+` with
`tl.arange`; a signed `int32` `<` against `n_elements`, which is the same at `n = 1`, where it is a
constant; masked loads without `other`; the IEEE `tl.add`, since no product feeds it; and a masked
store with no conversion. Every name resolves to `TVA.*` or to Lean's own `Nat` operations. The
launch is the host's: a 1-D grid of `cdiv(n, BLOCK_SIZE)` programs, the one block shape
`[BLOCK_SIZE]`, and the defaults for fusion, `num_ctas`, `pdl` and `debug`. The conditions cover
the script's real use and every `int32` size. Each narrowing (float32, `n < 2^31`, contiguous
buffers, `y`'s length, no output/input overlap within the first `n` elements, the H200) is stated
with its reason. `Post` is exact up to NaN payload, with a full frame. The one assumption is
`Compiler.Faithful`, cited exactly as the library states it. I have no finding.

## Findings

None.

## Accepted

All items.

## Checked

- `model`, line by line against SEMANTICS.md:
  - Line 39: `tl.program_id ctx 0` is the `int32` axis-0 id.
  - Line 44: `pid * (IntV.ofNat BLOCK_SIZE : I32)`. The constexpr is an operand of `*` with an `int32` tensor, so it is built in `int32` ("Constants"), and the product is `IntV.instMul`, which wraps.
  - Line 45: `Tensor.instHAddIntV` adds the scalar to every lane of `tl.arange 0 BLOCK_SIZE`.
  - Line 47: `ltS` against `IntV.ofNat n` is the signed `int32` `<`. For `n` in `[2, 2^31)` or `0`, `n_elements` is a runtime `int32` ("Scalar arguments"). At `n = 1` it is the constant `1`, which a comparison first makes an `int32` tensor ("Constants", everywhere else), so the comparison is the same.
  - Lines 50–51: masked loads without `other`, through `instHAddPtrTensorIntV`, which sign-extends each offset and steps 4 bytes per element.
  - Line 52: `tl.add` (IEEE), since both operands are loads and no product feeds them.
  - Line 54: a masked store of `float32` into `*fp32`, with no conversion.
- `model`, names: work/Names.lean prints `add_kernel`, `model` and `launch` with `pp.fullNames` and `pp.explicit`. Every operation and instance is `TVA.*` or Lean's own `Nat` instances (`Nat.instDiv`, `instSubNat`, `instAddNat`). The kernel's Lean has only `open TVA` and, after the model, `open Block` in the proof section. It has no notation, instance, `export`, renaming or option, and no binder named like a library namespace.
- `model`, launch: `Launch.grid1 ((n + B − 1) / B)` is `triton.cdiv` of host lines 70 and 75. `blocks = [[BLOCK_SIZE]]` is the shape of every block value (`arange`, `offsets`, `mask`, the pointer tensors, `x`, `y`, `output`). Fusion is `true` (the default; it does not matter here), and `num_ctas = 1`, `pdl = false`, `debug = false`. The host gives no launch option but the constexpr, and uses no autotune.
- `map`, the ranges and the code outside functions:
  - The kernel lines are 23–24 and 29–54. The launch runs only `add_kernel` and calls no helper. `add`, `benchmark` and the lambdas run only on the host.
  - The kernel reads only `triton` (line 29) and `tl` (lines 34, 39–54), bound by the kernel lines 23–24.
  - The host lines bind `torch`, `DEVICE`, `add`, `size`, `x`, `y`, `output_torch`, `output_triton` and `benchmark`. The kernel's `x` and `y` are its own names, assigned at lines 50–51 before their use at line 52, so it reads none of these.
  - Each drop (the two imports and the bare `@triton.jit`) adds no step. The other lines of 29–54 are comments.
- `map`, the supported subset: a top-level `def` under a bare `@triton.jit`; pointer, unannotated integer and `tl.constexpr` parameters with no defaults; `name = e` assignments and one call statement; `tl.*` calls with arguments of the kinds they state; the operators `*`, `+` and one `<`.
- The host is within `Compiler.Faithful`'s scope: one stream; no knob, hook, option, heuristic, `CompiledKernel` or descriptor; the active device taken to be the H200, as the README states.
- `conditions`: work/Cover2.lean proves `Inst.ok` for the host's launch shape (fresh, disjoint, 4-byte-aligned `float32` buffers):
  - at `BLOCK_SIZE = 1024`, for `n = 0`, `1`, `98432`, `2^27` and `2^31 − 1`;
  - at `n = 2^31 − 1`, for `BLOCK_SIZE = 2^20` and `1`;
  - for `add(x, x)`.

  The same file shows that `n = 2^31` (an `int64` `n_elements`) and an in-place output fail `Inst.ok`. `Pre` allows every `float32` value. The README states each narrowing with its reason, and matches `Inst.ok` and `Buf.Valid`.
- `spec`: `Post` fixes each output element exactly (`FloatV.add` up to `NaNEq`) and keeps every byte outside `output.elems {j | j < n}`, which covers the inputs, `output` from `n` on, and every other byte. work/Cover2.lean evaluates `FloatV.add` on `float32` patterns, and each result is the IEEE one: ties to even, rounding up, overflow to `+∞`, `∞ + −∞` NaN, signed zeros, subnormals kept, and a NaN input giving NaN. `1 + 2^-10` is exact, so a kernel that adds in `bfloat16` fails `Post`. There is no tolerance to evaluate at large sizes.
- `assumption:faithful`: `∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)` is the citation form of docs/rules/compiler-faithful.md. No approximate operation is used, so no `Num` rule is needed, and there is no `[kernel]` assumption.
- Axioms: `main`, `consistent` and `satisfiable` depend only on `propext`, `Classical.choice` and `Quot.sound` (work/Names.lean).
- The write-up against the Lean: the Specification, Conditions, Assumptions, Proof and Narrowings sections, the docstrings of `add_kernel`, `model`, `launch`, `Inst.ok` and `Post`, and the `satisfiable` instance (`testInst`, lines 85–89).

## Not checked

- The proofs beyond their statements and axioms (Lean checks them).
- The library (`tl.*` meanings, `Compiler.Faithful`, the device facts, Triton's specialisation of scalar and pointer arguments), which kernel review trusts.
