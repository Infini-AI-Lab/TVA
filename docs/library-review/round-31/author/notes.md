# Library batch, round 31: author's notes

Issues L139–L143 ([ISSUES.md](../../ISSUES.md)), all gaps found while writing TVA's fixed claims (`claims/`).
The claim sources each issue names were read, not edited. `lake build TVA` is green, `tvr library-check` was
run after it, and the new theorems use only the standard axioms (`propext`, `Classical.choice`,
`Quot.sound`).

## L139: A dtype as the argument of a `tl.constexpr` parameter

- **What changed.** `docs/SEMANTICS.md`: "The supported subset", "Parameters and arguments", admits a
  `tl.constexpr` parameter whose argument is a dtype of `tl`; a new item "A dtype as a `tl.constexpr` argument"
  in "How the front end evaluates the source" states that the parameter holds the dtype, a compile-time value,
  and means at each use what the dtype written there means, so it may be used wherever the page admits a dtype
  of `tl` (the model writes the dtype, a `DType` parameter or the instance's dtype); that launches with
  different dtypes do not share code; and that a dtype passed to a parameter without annotation, and a `torch`
  dtype used as a dtype, are rejected on the host.
- **Why.** The page listed `tl.constexpr` arguments that are a Python `int`, `float`, `bool`, `str` or `None`
  only, so `scaled_mm_kernel`'s `ACCUMULATOR_DTYPE` (`tl.zeros`'s `dtype=`, `tl.dot`'s `out_dtype=`),
  `_bmm_chunk_fwd_kernel`'s `dot_dtype` and `fused_moe_kernel`'s `compute_type` were outside the subset.
- **Evidence.** [evidence/front-end-round31](../../../../evidence/front-end-round31/README.md) D1–D5: for seven
  dtypes the same TTIR and values with the parameter as with the literal dtype in eight uses, `tl.dot`'s
  `out_dtype=` with a `tl.zeros(dtype=D)` accumulator included; by position and by keyword alike; each launch
  its own dtype in one process; the two rejections. Sources: `specialize.cc:515-560`, `jit.py:396-410`.
- **Constants whose meaning changed.** None: no Lean constant states this.

## L140: `tl.broadcast_to` and `x.broadcast_to(shape)`

- **What changed.** `lean/TVA/Lang/Tensor.lean`: new `Shape.bcastOk` (Triton's check of the shapes), with
  `Shape.bcastOk_length` and `Shape.bcastOk_dim`; `Tensor.broadcastTo x t h` (any rank; the default proof `h`
  is found for literal shapes), with `Tensor.bcast_coord_lt`, `Tensor.get_broadcastTo`, `Tensor.broadcastTo_cols`
  and `Tensor.broadcastTo_rows` (a `[m, 1]` or `[1, n]` block's broadcast is `bcastCols`, `bcastRows`); the
  module docstring names it. `docs/SEMANTICS.md`: an entry "`tl.broadcast_to(x, shape)`" with its meaning, both
  method forms, the scalar case (`Tensor.full`, `bc`), and the targets Triton rejects (a rank change, a
  differing dimension of size other than 1, a dimension that is not a power of two, a Python number).
  `docs/LIBRARY.md`'s `Tensor` row.
- **Why.** The page named `tl.broadcast_to` only inside other entries, with no meaning of its own, so
  `scaled_mm_kernel`'s `scale_a.broadcast_to((BLOCK_SIZE_M, 1))` rested on a fact only `primitives/` held.
- **Evidence.** evidence/front-end-round31 B1–B3: 10 pairs of shapes of ranks 2 and 3, in the three forms, the
  same values as torch's `expand`; a loaded scalar and a runtime argument; six rejected targets. Sources:
  `core.py:2049-2066`, `semantic.py:629-634, 711-725`, `_utils.py:63-74`.
- **Constants whose meaning changed.** None; new constants only (`TVA.Shape.bcastOk`, `TVA.Tensor.broadcastTo`
  and the lemmas above). `lean/TVA/Lang/Tensor.lean` changed, so every item whose cone reaches it is re-opened.

## L141: The integer `tl.dot`

- **What changed.** `lean/TVA/Rules/Num/Sm90.lean`: a new fact `Rules.Num.DotI8`:
  `∀ p, DotAcc P ⟨.i8, .i8, .i32, p, 0⟩ (fun _ => True) (fun _ => 0) (fun _ => 0)`, exact while
  `|acc| + Σ|a·b| ≤ 2^31 - 1`. `lean/TVA/Lemmas/Sm90.lean`: `Platform.sm90ref.dotI8` (from
  `Platform.ref.dotAcc_int`), so it is satisfiable with the other sm_90 facts. `docs/rules/num.md`: a row in
  the kernel reviewer's table and in "Facts the library states" (`DotI8`, listed alone: it is of type
  `Platform → Prop`), a section "The integer `tl.dot`" (the lowering, why the room makes it exact for every
  `K`, the evidence), "What the library does not state" (a given `max_num_imprecise_acc`, and any sum beyond the
  room, where the two paths differ), and "Satisfiability". `docs/SEMANTICS.md` (`tl.dot`): the only integer
  dot, its result dtype whatever `out_dtype` says, `K ≥ 32`, the accumulator's dtype, its `cfg` and fact. The
  docstring of `Rules.Num.DotAcc` names `DotI8`. `docs/LIBRARY.md`. `docs/TRANSCRIPTION.md`: a pitfall.
- **Why.** num.md listed the integer `tl.dot` as not stated, so `scaled_mm_int8` wrote `DotAcc` with zero error
  itself, constants of its own.
- **Evidence.** [evidence/num-dot-int8](../../../../evidence/num-dot-int8/README.md), on an H200: exact on
  2142192 outputs with room (14 tile configurations, `mma.sync` and `wgmma`, `M`, `N` from 1 to 128, `K` up to
  2048, four spellings with and without an accumulator) and on 236992 more under all six precisions and
  `allow_tf32=False`; exact at the edge of the room (`K = 131040`, `±(2^31 - 1)`); beyond it `mma.sync`
  (`.satfinite`) saturates and `wgmma` wraps, also for a partial sum the later chunks bring back in range; the
  front end's rejections.
- **Constants whose meaning changed.** None; new constants only (`TVA.Rules.Num.DotI8`,
  `TVA.Platform.sm90ref.dotI8`). `TVA.Rules.Num.DotAcc`'s docstring changed, not its value.

## L142: A pointer's `.type.element_ty`; a pointer as an `if` condition

- **What changed.** `docs/SEMANTICS.md`: the subset's attributes admit a tensor's `.type` and `.element_ty` of a
  pointer type or a block's type; a new item "A tensor's `.type`" in "Values" (for a scalar its `.dtype`, for a
  block its block type, whose `.element_ty` is the block's `.dtype`: so `p.type.element_ty` of a pointer
  argument is its buffer's dtype, of a block of pointers the pointer type, and of a non-pointer scalar
  rejected); "Expressions with no effect" lists `.type`; "A runtime `if`" states that a pointer condition is
  `p.addr ≠ 0`, with the empty tensor (address `0`) and a pointer given `None` (a compile-time `if`).
  `docs/TRANSCRIPTION.md`: a pitfall.
- **Why.** `scaled_mm_kernel` writes `accumulator.to(c_ptr.type.element_ty)` and `if bias_ptr:`, neither of
  which the page stated.
- **Evidence.** evidence/front-end-round31 P1–P3: `c_ptr.type.element_ty == c_ptr.dtype.element_ty` and
  `c_ptr.type == c_ptr.dtype` for eight dtypes, the stores; the block cases and the rejection; `if b_ptr:` taken
  for a tensor, not for an empty tensor, not visited for `None`, and its TTIR (`tt.ptr_to_int`,
  `arith.cmpi ne`, `scf.if`). Sources: `core.py:667-756, 884-890`, `semantic.py:900-906`,
  `code-generator.py:967-996`, `nvidia-driver.c:1055-1078`.
- **Constants whose meaning changed.** None: no Lean constant changed.

## L143: `tl.cast(x, dtype=…)`

- **What changed.** `lean/TVA/Lang/Promote.lean`: new `DType.bitwidth`, `DType.ofBits_toBits`, `tl.bitcast d d'
  x h` (each element's bits read in `d'`; the same width required) and `tl.bitcast_self`; the module docstring
  names it. `docs/SEMANTICS.md`: an entry "`x.to(d)`, `tl.cast(x, d)` and `x.cast(d)`" after "Float
  conversions": one operation; `tl.cast` makes a Python number a tensor of its own type first; without
  `bitcast`, `semantic.cast` (its own dtype gives `x` whatever the rounding; otherwise `tl.convert` and its
  parts, with `"rtne"` and `"rtz"` as "Float conversions" states; a rounding on a conversion that is not a
  float narrowing, and any other mode, rejected); with `bitcast=True`, `tl.bitcast`, the rounding ignored, the
  same width required, and a pointer operand or target converted as without `bitcast`. `docs/LIBRARY.md`'s
  `Promote` row.
- **Why.** The page did not state `tl.cast`, which `_moe_sum_reduce_kernel` (`claims/moe_sum_reduce_bf16`) and
  `causal_conv1d_fwd_bf16` write, nor `bitcast=`.
- **Evidence.** evidence/front-end-round31 C1–C4: the same TTIR for the three forms on 24 conversions; `tl.cast`
  of Python numbers and of a scalar argument specialised to `1`; which roundings are accepted; `bitcast=True`
  bit for bit on every pattern of 13 pairs of 8 and 16 bits and on `2^24` random and edge patterns of seven
  pairs of 32 and 64 bits, and the three rejected widths. Sources: `core.py:1130-1135, 2322-2345`,
  `semantic.py:793-916`.
- **Constants whose meaning changed.** None; new constants only (`TVA.DType.bitwidth`,
  `TVA.DType.ofBits_toBits`, `TVA.tl.bitcast`, `TVA.tl.bitcast_self`). `lean/TVA/Lang/Promote.lean` changed, so
  every item whose cone reaches it is re-opened.
