# Index of the TVA library

Where to find a definition or lemma of the library, and the ones a proof usually needs. Every
declaration of `lean/TVA/` is listed in [`index/`](index/), generated from the built library by
`tools/make_index.py`. Each one comes with its signature, its docstring and its `file:line`.

## How to search

* **By name or word:** [`index/NAMES.tsv`](index/NAMES.tsv) has one line per declaration: name,
  kind, `file:line` and the docstring's first sentence.

  ```bash
  grep -i 'tl.load' docs/index/NAMES.tsv          # everything about tl.load
  grep -P '^TVA\.safe_' docs/index/NAMES.tsv      # every rule for Safe
  grep -i 'disjoint' docs/index/NAMES.tsv          # footprint disjointness lemmas
  ```

* **By part of the library:** one file per part, in source order, with full signatures and
  docstrings:

  | Trusted (`import TVA`) | Proof kit (`import TVA.Proof`) |
  |---|---|
  | [Claim](index/Claim.md): `Claim`, `Holds`, `Satisfiable` | [Proof.Basic](index/Proof.Basic.md): `Safe`, memory and footprint lemmas, `correct_of_safe` |
  | [Base](index/Base.md): bytes, integers, float formats, dtypes | [Proof.Rules](index/Proof.Rules.md): one rule per `tl.*` operation |
  | [Core](index/Core.md): tensors, memory, platform, `Prog` | [Proof.Pattern](index/Proof.Pattern.md): blocks, tiles, block pointers, descriptors |
  | [Triton](index/Triton.md): the meaning of each `tl.*` operation | [Proof.Shared](index/Proof.Shared.md): atomics on shared cells, `correct_of_safeA` |
  | [Launch](index/Launch.md): launches, outcomes, `WellBehaved`, `Correct` | [Proof.Int](index/Proof.Int.md): widths, casts, `//`, `%`, lanes |
  | [Device](index/Device.md): `Device P`, the reference platform, tolerances | [Proof.Float](index/Proof.Float.md): rounding, operation errors, trees, sums |
  | | [Proof.Dot](index/Proof.Dot.md): `acc += tl.dot` loops, chunked accumulation |
  | | [Proof.Math](index/Proof.Math.md): scans, sorts, histograms, `tl.rand`, softmax, recurrences |

* **In Lean:** `#check @TVA.safe_load` in a scratch file prints a signature. `exact?` and
  `apply?` search the imported library for a goal.

## What a proof usually needs

A proof of `theorem main : claim.Holds` follows the worked example (`claims/vec_add_f32/oracle/Proof.lean`,
explained in [PROVING.md](PROVING.md)). The table lists, for each step, the lemmas to reach for first.
Everything is in namespace `TVA`.

| Step | Lemmas | Index |
|---|---|---|
| Finish `main`: per-program facts give `Correct` | `correct_of_safe`, `Launch.valid_grid1` | [Proof.Basic](index/Proof.Basic.md) |
| Finish `main` when programs meet only through atomics | `correct_of_safeA`, `SafeA`, `SafeA.atomic_add`, `replay_add`, `sum_proj` | [Proof.Shared](index/Proof.Shared.md) |
| Walk one program: sequence, weaken, run a step | `Safe`, `Safe.bind`, `Safe.seq`, `Safe.mono`, `Safe.stepT`, `safe_ret` | [Proof.Basic](index/Proof.Basic.md), [Proof.Rules](index/Proof.Rules.md) |
| `tl.load`, `tl.store`, `tl.atomic_*` | `safe_load`, `safe_store`, `safe_load1`, `safe_store1`, `safe_load_buf`, `safe_store_buf`, `safe_atomic_buf` | [Proof.Rules](index/Proof.Rules.md) |
| `offs = pid*B + tl.arange(0, B)`, `mask = offs < n` | `Block.mask`, `Block.Fits`, `Block.fits_of_grid`, `Block.fits_of_grid_pow2`, `safe_load_block`, `safe_store_block`, `Buf.tile` | [Proof.Pattern](index/Proof.Pattern.md) |
| 2-D tiles `base + rows[:, None]*s0 + cols[None, :]*s1` | `safe_load_tileAt`, `safe_store_tileAt`, `Tile2.add_inj` | [Proof.Pattern](index/Proof.Pattern.md) |
| Block pointers, tensor descriptors | `safe_load_block_ptr`, `safe_store_block_ptr`, and the rules in `Pattern/Desc.lean` | [Proof.Pattern](index/Proof.Pattern.md) |
| Footprints do not overlap; reads after writes | `Buf.elems_disjoint_of`, `Buf.elems_sub_bytes`, `Mem.read_write_self`, `Mem.read_congr` | [Core](index/Core.md), [Proof.Basic](index/Proof.Basic.md) |
| Float arithmetic: `+`, `*`, `/`, `fma`, `tl.math.*` | `safe_add`, `safe_mul`, `safe_div`, `safe_fdiv`, `safe_fma`, `safe_math`, then `FloatV.add_err`, `FloatV.mul_err` for the error | [Proof.Rules](index/Proof.Rules.md), [Proof.Float](index/Proof.Float.md) |
| Integer arithmetic, `//`, `%` | `safe_idiv1`, `safe_irem1`, `IntV.div?_of_nonneg`, `IntV.I32.toInt_ofNat`, `IntV.I32.add_nat`, `irem_ok` | [Proof.Rules](index/Proof.Rules.md), [Proof.Int](index/Proof.Int.md) |
| Conversions (`.to(dtype)`, fp8) | `safe_convert`, `safe_cvt`, `safe_cvt8`, `safe_itof`, `safe_ftoi` | [Proof.Rules](index/Proof.Rules.md) |
| `tl.sum`, `tl.max`, `tl.argmax` and their errors | `safe_sum`, `safe_sum1`, `safe_max1`, `safe_argmax1`, `sum_err`, `Tree.eval_err` | [Proof.Rules](index/Proof.Rules.md), [Proof.Float](index/Proof.Float.md) |
| `tl.dot` and `acc += tl.dot(a, b)` loops | `safe_dot`, `safe_dot_add`, `DotLoop.dotAdd_step`, `DotLoop.accum_step`, `DotLoop.accum`, `sum_bcast_mul_err` | [Proof.Rules](index/Proof.Rules.md), [Proof.Dot](index/Proof.Dot.md) |
| `for` loops over `range` / `tl.range`, `while` | `safe_forN`, `safe_forN_bind`, `safe_forRange`, `safe_while` | [Proof.Rules](index/Proof.Rules.md) |
| `tl.cumsum`, `tl.associative_scan`, `tl.sort`, `tl.histogram`, `tl.rand` | `safe_cumsum`, `safe_associative_scan`, `safe_histogram`, `safe_rand` | [Proof.Math](index/Proof.Math.md) |
| Online softmax (running max and sum) | `Softmax.err_final`, `Softmax.err_final_mono` | [Proof.Math](index/Proof.Math.md) |
| A state carried across loop trips with a decay | `Recurrence.run`, `Recurrence.le_run`, `Recurrence.closed_cumsum` | [Proof.Math](index/Proof.Math.md) |
| Device facts (accuracy of `exp`, `tl.dot`, …) | `Device` (its fields: `hP.<field>`), `Device.ref_ok` | [Device](index/Device.md) |

## Keeping the index current

`docs/index/` is generated: never edit it by hand. After changing the library, rebuild it
(`cd lean && lake build && lake build TVA.Proof`) and run `python3 -I tools/make_index.py`.
`python3 -I tools/make_index.py --check` fails when the index no longer matches the library, and
the sandbox's image build runs that check.
