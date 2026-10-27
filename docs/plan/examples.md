# Plan: proofs of vec-add, grouped GEMM and flash-attention forward

User goal (2026-10-27): "write proofs for vec-add, flash-attn-fwd and grouped-gemm. we need to
clearly state the specification, assumption, condition and proof."

The sources are Triton 3.8.0's tutorials `01-vector-add.py`, `08-grouped-gemm.py`
(`grouped_matmul_kernel`, the pointer version) and `06-fused-attention.py` (`_attn_fwd`).
Each bundle states four things, and its kernel folder's `README.md` (`kernels/<kernel>/`) repeats them in words:

- **Specification** (`Post`): what the outputs are.
- **Conditions** (`Inst.ok`, `Pre`): the instances and inputs covered.
- **Assumptions** (`Assumptions`): `Compiler.Faithful` and the numeric rules, with constants.
- **Proof** (`main`, and `consistent`).

## Library work first

Status (2026-10-27): items 1-4 are done (`Fused`, `tl.intToPtr` in `Lang/Ops`, `Scalar`,
`DotLoop`). Full-tile 2-D accesses use `safe_load_buf`/`safe_store_buf` directly, so no separate
tile module was needed. Item 5 is in `TVA.Lemmas.Softmax*` (flash attention).

1. `TVA.Lemmas.Fused`:
   - the error of a rounding that comes out finite (IEEE formats), also just above `maxVal`;
   - `RoundsTo` and the `AddRel` error bound.

   Both kernels have contractible additions.
2. `tl.int_to_ptr`: grouped GEMM loads its matrix addresses as `int64` and casts them.
3. A 2-D tile kit: row-major matrices with a leading dimension, full tiles.
4. Accumulation over `K` with `tl.dot_add`, from `Rules.Num.DotAcc`.
5. Online softmax: the numerical invariant of the flash-attention loop.

## Conditions chosen (narrowings, stated in the bundles)

- **Grouped GEMM.** The kernel has no masks ("assume full tile"), so every `M`, `N`, `K` is a
  multiple of its block size. The matrices are row-major fp16.
- **Flash attention.**
  - Non-causal (`STAGE = 1`), fp16 (`FP8_OUTPUT = False`), Hopper (`IS_HOPPER`).
  - `N_CTX` is a multiple of `BLOCK_M` and `BLOCK_N`; otherwise tiles cross into the next
    head.
  - `q` and `k` entries are bounded.

## Status

| Bundle | Lean | Page | State |
|---|---|---|---|
| vec-add | [`kernels/vec_add_f32/`](../../kernels/vec_add_f32/) | [README](../../kernels/vec_add_f32/README.md) | proved; restated with labelled conditions/specification/assumptions/proof |
| grouped GEMM | [`kernels/grouped_gemm_f16/`](../../kernels/grouped_gemm_f16/) | [README](../../kernels/grouped_gemm_f16/README.md) | proved (`main`, `consistent`; standard axioms only) |
| flash attention fwd | [`kernels/flash_attn_fwd_f16/`](../../kernels/flash_attn_fwd_f16/) | [README](../../kernels/flash_attn_fwd_f16/README.md) | proved (`main`, `consistent`; standard axioms only) |

Library rule added on the way: a hint on a block argument (a loop-carried value or an
induction variable) is a no-op in Triton 3.8.0 and is dropped (`SEMANTICS.md`,
`REVIEWER_GUIDE.md`, `Lang/Ops.lean`; `primitives/triton-3.8.0-ir.cc`, `…-AxisInfo.cpp` pinned).
