# Library batch, round 40: author's notes

Issues L180 to L183 ([ISSUES.md](../../ISSUES.md)), all in the proof kit (`lean/TVA/Lemmas`): `Tile2`, `Block`,
`Recurrence`, and one docstring of `IntW`, with the `Block` and `Tile2` rows of `docs/LIBRARY.md`. No semantics,
rule, device fact or document that a kernel reviewer reads changed. `lake build TVA` is green, and every changed
or new declaration depends only on `propext`, `Classical.choice` and `Quot.sound` (or fewer), with no `sorry`. No
kernel or claim folder holds a Lean file that names a `Tile2` declaration, so no kernel needs a change. The
probes in [work/probes.lean](work/probes.lean) check each reviewer's case with `lake env lean` from `lean/`.

## L180: Lane bounds for enabled lanes only

- **What changed.**
  - `Tile2`'s footprint, load and store rules (`addrs_tile2`, `accBytes_tile2`, `safe_load_tile2`,
    `safe_store_tile2`) take the pointers on the enabled lanes only:
    `hp : ∀ i j, r0 + i < M → c0 + j < N → ptrs.get2 i j = b.ptr (e + (r0 + i) * s0 + (c0 + j) * s1)`.
  - New lane lemmas for the enabled lanes, from the matrix's largest offset
    `(M - 1) * s0 + (N - 1) * s1 < 2^(w-1)`:
    - offsets: `off_toInt_on`, `add_toInt_on`, and the terms `colMul_toInt_on`, `mulCol_toInt_on`,
      `rowMul_toInt_on`, `mulRow_toInt_on`;
    - pointers: `ptr_lane_on`, `ptr_lane_seq_on`, `ptr_lane_sum_of_on`, `ptr_lane_seq_of_on`,
      `ptr_lane_sum_swap_of_on`, `ptr_lane_seq_swap_of_on`. Each concludes in the exact form of `hp`.
  - The every-lane lemmas stay: `fun i j _ _ => ptr_lane … i j` gives `hp` from them.
  - In `Block`, the existing statements are unchanged. Beside them:
    - `fits_of_dvd` and `fits_of_pow2`: `Fits p B` holds once the program's first offset `p * B` is
      below `2^31`;
    - `fits_of_lt_pow2`: every program with an element below `n ≤ 2^31` fits (`p * B < n`), the last
      partial block included.
  - The `Block` header says why the 1-D block keeps a bound on every lane.
- **Why.**
  - In a 2-D tile the mask is computed from `rows` and `cols`, not from the offset. So a masked-off
    lane's wrapped offset is never used: `tl.addrs` gives that lane no address.
  - A claim that bounds only the matrix could not use the old rules on the last partial tile.
    lr-39b's example: `M = 1000`, `BM = 64`, `s0 = 2149000`.
  - The 1-D block's mask compares the offsets themselves, so a lane that wraps to a negative offset
    would be enabled. Its bound must therefore cover every lane. For a power-of-two block (every
    `tl.arange` length, SEMANTICS.md) this costs nothing: a multiple of `B` below `2^31` is at most
    `2^31 - B`. So `n ≤ 2^31` suffices for every program with an element, and that is the 1-D form
    the issue asks for.
- **Evidence.** [work/probes.lean](work/probes.lean):
  - lr-39b's numbers satisfy the matrix bound and fail the every-lane bound on the tile at `r0 = 960`;
  - `ptr_lane_on`, `maskOk_and` and `safe_load_tile2` then load that tile;
  - `fits_of_lt_pow2` gives `Fits p 1024` from `p * 1024 < n ≤ 2^31`.
- **Constants whose meaning changed.**
  - `TVA.addrs_tile2`, `TVA.accBytes_tile2`, `TVA.safe_load_tile2`, `TVA.safe_store_tile2`: weaker
    hypothesis `hp`.
  - Dropped the `hw : 0 < w` premise, which `IntV.mul_nat` and `IntV.add_nat` do not need:
    - offsets: `TVA.Tile2.colMul_toInt`, `mulCol_toInt`, `rowMul_toInt`, `mulRow_toInt`, `add_toInt`,
      `off_toInt`;
    - pointers: `ptr_lane`, `ptr_lane_seq`, `ptr_lane_sum_of`, `ptr_lane_sum_swap_of`.
  - New declarations: the twelve `…_on` lemmas, and `TVA.Block.fits_of_dvd`, `fits_of_pow2`,
    `fits_of_lt_pow2`.

## L181: An empty tile at the end of its buffer

- **What changed.** `Tile2.elem_lt`, `Buf.tile2_sub_bytes`, `Buf.tile2_disjoint_of` (both buffers),
  `safe_load_tile2` and `safe_store_tile2` take `M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len`
  where they took the last disjunct alone.
- **Why.**
  - An empty matrix has no element, so its tile reads and writes nothing, wherever it starts.
    For example, the last sequence of a varlen batch, of length `0`, at `e = b.len`.
  - With `M = 0` the old premise was false, although the access is safe (lr-39a `work/tile2_empty.lean`).
  - The 1-D `safe_load_block` already admits `n = 0` (`n ≤ b.len`).
  - A tile past a non-empty matrix (`r0 ≥ M`) was already covered, since the premise is on the
    matrix.
- **Evidence.** [work/probes.lean](work/probes.lean): lr-39a's case (`M = 0`, `e = b.len`) now goes through
  `safe_load_tile2` and `safe_store_tile2`. The disjunct is `.inl rfl`, and `hp` holds vacuously because no
  lane is enabled.
- **Constants whose meaning changed.** `TVA.Tile2.elem_lt`, `TVA.Buf.tile2_sub_bytes`,
  `TVA.Buf.tile2_disjoint_of`, `TVA.safe_load_tile2` and `TVA.safe_store_tile2`: each premise is weaker.
  A caller with the old premise `h` passes `.inr (.inr h)`.

## L182: `exp_sub_mem_Ioc`'s docstring

- **What changed.** The docstring of `Recurrence.exp_sub_mem_Ioc` now says what the lemma bounds: the
  exact gate `g t = Real.exp(…)`. It points to:
  - `perturb'`, whose `|g t| ≤ G t` is on the exact gate, with `G t = 1`, followed by `run_le_sum`;
  - `abs_le_sum`.

  It also says that `perturb_le_sum` bounds the computed decay `ĝ t`, which an approximate `exp`
  may round above `1` (`perturb_le_pow`).
- **Why.** The old text sent a kernel to `perturb_le_sum` with a bound on the wrong quantity.
- **Evidence.** None needed (wording; the statements of `perturb'`, `run_le_sum` and `abs_le_sum`
  are as cited).
- **Constants whose meaning changed.** None.

## L183: `start_arange64_toInt` and repeated integer lemmas

- **What changed.**
  - `Tile2.start_arange64_toInt` is restated for the form a faithful transcription reaches. The
    `arange` is converted by `tl.convert .i32 .i64`: `x` with
    `∀ i, ConvRel P .i32 .i64 ((tl.arange 0 n).get i) (x.get i)`, the lanes `safe_convert` gives.
  - New `Tile2.arange64_toInt` gives such an `x`'s lane `i` as `i`, through
    `ConvRel.toInt_i32_i64` and `IntV.I32.toInt_tl_arange`.
  - `start_add_toInt` takes lanes of value `i`, the form of `I32.toInt_tl_arange` and
    `arange64_toInt`, and has no `0 < w` premise.
  - Removed four repeats; each first one is the `Tile2` lemma, the second the one to use:
    - `Tile2.toInt_mul_nat`: `IntV.mul_nat`;
    - `toInt_add_nat`: `IntV.add_nat`;
    - `toInt_mul_ofNat`: `IntV.mul_ofNat`;
    - `arange_toInt`: `IntV.I32.toInt_tl_arange`.
  - Also removed `start_arange_toInt`, which repeated `IntV.I32.toInt_sadd_arange`.
  - Also removed `castInt_toInt`, which no transcription of this pattern reaches.
  - `Tensor.get1_castInt`'s docstring (`IntW`) no longer calls `castInt` `x.to(tl.int64)`. It says
    that `.to(…)` and the implicit promotion are `tl.convert`.
  - `Tile2` now imports `TVA.Lemmas.IntW`.
- **Why.**
  - SEMANTICS.md ("Operators and promotion", and `tl.cast` without `bitcast`) models both the
    implicit promotion and `.to(tl.int64)` as `tl.convert`. So the `castInt` form was unreachable.
  - The issue asks for one lemma of each kind.
- **Evidence.** [work/probes.lean](work/probes.lean): `pid.to(tl.int64) * BM + tl.arange(0, BM)` through
  `safe_convert` and `start_arange64_toInt`.
- **Constants whose meaning changed.**
  - Restated: `TVA.Tile2.start_arange64_toInt`, `TVA.Tile2.start_add_toInt`.
  - New: `TVA.Tile2.arange64_toInt`.
  - Removed: `TVA.Tile2.toInt_mul_nat`, `toInt_add_nat`, `toInt_mul_ofNat`, `arange_toInt`,
    `start_arange_toInt`, `castInt_toInt`.
  - `TVA.Tensor.get1_castInt`: docstring only.
  - lr-39a's round-39 probes `work/tile2_use.lean` and `work/tile2_empty.lean` used the old signatures. They
    are records, not library files.
