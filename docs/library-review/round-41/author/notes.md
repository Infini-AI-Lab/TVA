# Library batch, round 41: author's notes

Issues L184 to L188 ([ISSUES.md](../../ISSUES.md)). L184 and L185 are in the proof kit (`lean/TVA/Lemmas/Tile2.lean`,
`lean/TVA/Lemmas/IntW.lean`), with three docstrings of `TVA.Lang` (`Tensor.castInt`, `IntV.cast`) and
`TVA.Lemmas.IntW` (`Tensor.get1_castInt`) reworded and one row added to `docs/TRANSCRIPTION.md`. L186 to L188 are
in `tools/tvr.py`, `lean/lakefile.toml` and the pages that describe `kernels/`. No definition or statement of
`TVA.Lang`, `TVA.Sem`, `TVA.Claim` or `TVA.Rules` changed, and no semantics, rule or device fact changed.

Checks, all from `lean/` after `lake build TVA` (green):
- [work/probes.lean](work/probes.lean), output [work/probes.out](work/probes.out): the probes below, and
  `#print axioms` of all 102 new or changed declarations: each depends on `propext`, `Classical.choice` and
  `Quot.sound` at most.
- The coordinator's constraint for this round: the 41 fixed claims must stay unaffected.
  - Every `claims/<k>/Claim.lean` was elaborated with `lake env lean` against the new library
    ([work/claims.out](work/claims.out), one line per claim). 40 of the 41 claims elaborate with no error.
  - `claims/layer_norm_fwd_f16/Claim.lean` reports one error, at line 547 (`zero16_facts`: `simp` leaves
    `(FloatV.ofPat (leValue [0, 0])).toRat = 0` unsolved). This error comes from before this round. The
    same file gives the same error against the library as committed (HEAD, before this round's changes), in
    a scratch copy built with `lake build TVA`. Nothing this round changed reaches it: the round added no
    `simp` lemma or instance, and changed no definition of `TVA.Lang`. It is for the coordinator (the
    claim is fixed by its sha256).
  - `git diff --stat -- claims/` is empty. Nothing under `claims/` was edited.
  - Coordinator's addendum: the error came from round 39's `simp` lemmas `DType.ofBits_*`, not from before
    round 39 (L189); after the fix below, all 41 claims elaborate with no error.
- The round-40 probes (author, lr-40a, lr-40b), which use the `Tile2` lemmas in their `r0 + i` form, still
  elaborate with no error, unchanged.

**`kernels/` re-rendered.** While testing L187 I ran `python3 -I tools/tvr.py index` once (about 15:04 on
2026-11-10). It re-renders every `kernels/<k>/REVIEW.md` and `kernels/README.md` (`render_review`,
`render_index`): their gates now show as stale, naming this round's changed library files. No other kernel file
was written: I ran no `tvr check` and no `tvr new`. `tvr index` regenerates those pages from each kernel's
ledger and `gate.json`, which it does not change, so the next `tvr check` or `tvr index` the coordinator runs
writes them the same way. Building `VecAdd` and `Softmax` through the generated configuration (L187, below)
wrote only to `lean/.lake/build`.

## L184: `Tile2` for rows that are not `r0 + i`

- **What changed.** All in `lean/TVA/Lemmas/Tile2.lean`.
  - The twelve enabled-lane lemmas take row and column lane values of any form. Row lane `i` has the value
    `ρ i` and column lane `j` the value `κ j` (`ρ : Fin BM → Nat`, `κ : Fin BN → Nat`, implicit). These
    replace `r0 + i` and `c0 + j`. The lemmas are `colMul_toInt_on`, `mulCol_toInt_on`, `rowMul_toInt_on`,
    `mulRow_toInt_on`, `add_toInt_on`, `off_toInt_on`, `ptr_lane_sum_of_on`, `ptr_lane_sum_swap_of_on`,
    `ptr_lane_seq_of_on`, `ptr_lane_seq_swap_of_on`, `ptr_lane_on` and `ptr_lane_seq_on`. A caller's
    `hr : ∀ i, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)` still fits: Lean takes `ρ = fun i => r0 + i`.
  - New, for rows and columns of any form:
    - masks: `MaskAt mk ρ κ M N`, with `maskAt_of_get`, `maskAt_none`, `maskAt_and`, `maskAt_rows` and
      `maskAt_cols`. `maskOk_iff_maskAt` (`Iff.rfl`): `MaskOk mk r0 c0 M N` is `MaskAt` with `r0 + i`,
      `c0 + j`;
    - elements: `ElemAt`, `elem_iff_elemAt` (`Elem` is the `r0 + i` instance), `elemAt_matrix` (a tile's
      element is an element of the whole matrix) and `elemAt_lt`;
    - bytes: `Buf.tileAt`, `Buf.tile2_eq_tileAt`, `Buf.tileAt_sub_matrix` (a read footprint holding the
      matrix holds the tile), `Buf.tileAt_sub_bytes` and `Buf.span_sub_tileAt`;
    - accesses: `addrs_tileAt`, `tileAt_lane_elem`, `img_tileAt`, `accBytes_tileAt`, `tileAt_inj`,
      `safe_load_tileAt` and `safe_store_tileAt`. The store also needs distinct enabled rows (`hρ`) and
      columns (`hκ`). `add_inj` gives both for `r0 + i`;
    - rows wrapped by `% M`: `Tile2.irem_ok` (every lane of `tl.irem x y` is defined), `Tile2.irem_toInt`
      (lane `i` has the value `f i % M`) and `Tile2.safe_irem_lanes` (the rule). Each takes `x`'s lanes of
      value `f i` and `y`'s lanes of value `M > 0`.
  - The existing `r0 + i` rules (`MaskOk`, `Elem`, `Buf.tile2`, `addrs_tile2`, …, `safe_store_tile2`) are
    unchanged.
  - The module docstring and the `Tile2` row of `docs/LIBRARY.md` say all this.
- **Why.**
  - Tutorial 03's `offs_am = (pid_m * BM + tl.arange(0, BM)) % M` has no row mask, and its row `i` is
    `(r0 + i) % M`. The test-set kernels lr-40a names use the same pattern: `% N`, `offset_n % N`.
  - The enabled-lane lemmas never needed `r0 + i`. Their bound is `ρ i < M`, and that is all they use.
  - The load needs no injectivity: `% M` rows repeat when `BM > M`, and a load may read an element twice.
    The store does need it, so `safe_store_tileAt` takes `hρ` and `hκ`. The `r0 + i` store keeps them
    hidden (`add_inj`).
- **Evidence.** [work/probes.lean](work/probes.lean):
  - Tutorial 03's `a` tile on the first trip goes through `safe_irem_lanes`, `ptr_lane_on` (with
    `ρ i = (r0 + i) % M`), `maskAt_cols` (`Nat.mod_lt`), `Buf.tileAt_sub_matrix` and `safe_load_tileAt`. It
    starts from the `int32` lanes of `pid_m * BM + tl.arange(0, BM)`, and its only bound is the matrix's.
  - An `r0 + i` store goes through the general rule from a `MaskOk` proof, `ptr_lane_on` and `add_inj`.
  - With `BM = 4` and `M = 3`, the `% M` rows' `hρ` is false, as it must be.
  - No device fact is involved: `tl.irem` is `rem?` lane by lane (SEMANTICS.md), and the rest is arithmetic.
- **Constants whose meaning changed.**
  - Statements generalised (implicit `ρ`, `κ` in place of `r0 + i`, `c0 + j`): `TVA.Tile2.colMul_toInt_on`,
    `mulCol_toInt_on`, `rowMul_toInt_on`, `mulRow_toInt_on`, `add_toInt_on`, `off_toInt_on`,
    `ptr_lane_sum_of_on`, `ptr_lane_sum_swap_of_on`, `ptr_lane_seq_of_on`, `ptr_lane_seq_swap_of_on`,
    `ptr_lane_on` and `ptr_lane_seq_on`.
  - New: `TVA.Tile2.MaskAt`, `ElemAt`, `TVA.Buf.tileAt` and the lemmas listed above.
  - No other constant changed. No kernel or claim Lean file names a `Tile2` declaration (grep).

## L185: One meaning for the integer cast

- **What changed.**
  - New in `lean/TVA/Lemmas/IntW.lean`:
    - `IntV.cast_eq_ofInt`: `(IntV.cast a : IntV s' w') = IntV.ofInt a.toInt`, for any signedness and
      width;
    - `ConvRel.iff_cast_<d>_<d'>` for all 64 pairs of `int8`…`int64` and `uint8`…`uint64`:
      `ConvRel P d d' x r ↔ r = IntV.cast x`;
    - `safe_convert_iff`: if every element has exactly one result, the element of `y`
      (`ConvRel E.P d d' (x.get i) r ↔ r = y.get i`), then `Safe … (tl.convert d d' x >>= k) ↔ Safe … (k y)`.
      With `iff_cast_i32_i64`, `y` is `x.castInt`.
  - Docstrings reworded, with no change to a definition or statement: `Tensor.castInt`
    (`lean/TVA/Lang/Tensor.lean`), `IntV.cast` (`lean/TVA/Lang/Scalar.lean`) and `Tensor.get1_castInt`
    (`IntW`). All three now say the same thing:
    - the model writes `x.to(d')` and an implicit promotion as `tl.convert` (SEMANTICS.md);
    - between integer dtypes other than `int1`, that conversion has exactly one result, `x.castInt`, so
      either form transcribes it;
    - `castInt` is also the model's own widening step.
  - `docs/TRANSCRIPTION.md`: a pattern row for `offs.to(tl.int64)` and the promoted `int32` `arange`.
  - The `IntW` row of `docs/LIBRARY.md` and the module docstring list the new lemmas.
- **Why.**
  - `castInt` and `IntV.cast` said `x.to(tl.int64)`; round 40's `get1_castInt` said it is not. Both are true
    in value: `ofIntRel` gives the value modulo `2^w'`, and the cast sign- or zero-extends by the source and
    then truncates, which is the same value.
  - The lemmas let a prover move between the two forms, and let a reviewer accept the claims written with
    `castInt` (`rotary_varlen_bf16`, `recompute_wu_bf16`, `ssd_chunk_scan_bf16`) as written. Per the
    coordinator's constraint, `castInt`, `IntV.cast`, `tl.convert` and `ConvRel` keep their definitions.
- **Evidence.** [work/probes.lean](work/probes.lean):
  - `safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64` proves
    `Safe (tl.convert .i32 .i64 x >>= k) ↔ Safe (k x.castInt)`;
  - a narrowing pair (`i64_i32`) and an unsigned pair (`u32_i64`) also check;
  - the cast of `int32` `-1` is `2^32 - 1` in `uint32` and `-1` in `int64` (`decide`).
- **Constants whose meaning changed.**
  - None changed in meaning. `TVA.Tensor.castInt`, `TVA.IntV.cast` and `TVA.Tensor.get1_castInt` have new
    docstrings only, but their files changed (`TVA/Lang/Tensor.lean`, `TVA/Lang/Scalar.lean`), so every
    kernel item whose cone reaches them is re-opened (§4).
  - New: `TVA.IntV.cast_eq_ofInt`, `TVA.ConvRel.iff_cast_*` (64) and `TVA.safe_convert_iff`.

## L186: The release check's audit and `git`

- **What changed.** `tools/tvr.py`:
  - New `git_checkout()`: whether TVA's root has `.git`.
  - New `shipped_set()`: what ships, folders included, in a git checkout, and `None` outside one.
  - `cited_files` filters by what ships only in a git checkout. Outside one it reads every Markdown and
    Lean file (`CITE_SKIP` and `review/` aside), as `main` already did for `shipped_problems`. So the audit
    runs no git in a release copy.
  - REVIEW_PROTOCOL §2 ("Citations") says so.
- **Why.**
  - The release copy has no `.git` and holds only what ships. There `git ls-files` failed ("not a git
    repository"), and inside the sandbox `system_tool("git")` refused `/usr/bin/git` first.
- **Evidence.** [work/audit_copy.py](work/audit_copy.py), output [work/audit_copy.out](work/audit_copy.out).
  It copies the shipped files into an empty folder. Case C1, with no `.git`: `independent`, exit 0, where
  before this round it stopped at `git ls-files`. The release check (`tvr release`, log in [work/release.log](work/release.log)) passes 8 of 8 steps, `audit` (`independent`) included. It ran on a copy taken before one last edit, the usage line of `tvr new` in `tools/tvr.py` (text only); `tvr audit` passes after that edit too.
- **Constants whose meaning changed.** None (tool only).

## L187: The shipped TVA and its audit

- **What changed.**
  - `lean/lakefile.toml` declares `TVA` alone. The 37 kernel `lean_lib`s and their comment are gone.
  - `lake_violations` refuses any `lean_lib` other than `TVA`.
  - `tvr new` no longer edits the lakefile (`add_lake_lib` is removed).
  - New `kernel_lakefile()` writes `lean/.lake/tvr/lakefile.toml`, which is git-ignored (`.lake/`). It holds
    the shipped configuration, then one library per local kernel folder (the same `srcDir` and root rules as
    before). `tvr check` builds with `lake -f <that file> build <module>` from `lean/`, so the package,
    manifest and build folder are the library's. `tvr new`, `tvr check` and `tvr index` write it afresh.
  - `citation_problems`: in a git checkout, a cited TVA file or folder must also ship, not only exist.
  - `README.md` and `docs/LIBRARY.md` no longer link to `kernels/README.md`. They point to `claims/` and
    describe `kernels/` as local.
  - The kernel prover's prompt (`docs/prompts/kernel-prover.md`) builds with `lake -f .lake/tvr/lakefile.toml`.
  - REVIEW_PROTOCOL §2 ("Lake", "Citations"), §3 and §8's `build` row say all this.
- **Why.** A fresh copy failed its own audit with 39 problems: 37 `lean_lib`s with no kernel folder, and 2
  links to a file that does not ship. The local audit could not see them, because the git-ignored `kernels/`
  exists in the working tree.
- **Evidence.**
  - Before the lakefile change, the new checks in the working tree listed exactly those 39 problems.
  - Afterwards, `tvr audit` prints `independent` in the working tree.
  - [work/audit_copy.out](work/audit_copy.out):
    - C2, a fresh git copy: `independent`;
    - C3, a git-ignored `kernels/README.md` that exists and is cited from a shipped page: refused ("which
      TVA does not ship");
    - C4, a kernel `lean_lib` in the lakefile: refused.
  - `lake -f .lake/tvr/lakefile.toml build VecAdd` and `build Softmax` succeed. `lake build VecAdd` with the
    shipped lakefile fails with "unknown target", as intended.
- **Constants whose meaning changed.** None (tool and configuration).
- **Left open.**
  - `sandbox/` (exempt from `tvr`, L135) still runs `lake build <module>` for a kernel task (`tvakit.build`)
    and needs the generated configuration (`-f`) to build a kernel. That is for the sandbox's owner.
  - Kernels no longer appear in the shipped lakefile. A kernel that no longer builds (none found) is its
    prover's to fix.

## L188: Documents that describe `kernels/` as shipped

- **What changed.**
  - REVIEW_PROTOCOL.md §2:
    - the release check now reads nothing of `kernels/` and covers no kernel. Its `records` and `lint`
      read the library's records alone, and `lake build TVA` builds the library alone. A local kernel is
      checked where it lives;
    - "Every file under `kernels/` is scanned" now applies only where a local `kernels/` exists;
    - the "Lake" bullet describes the lakefile as above.
  - REVIEW_PROTOCOL.md §3: `kernels/` is local and does not ship.
  - LIBRARY_REVIEW.md's `gate` row: of `kernels/` the release check reads nothing (L187, L188); of `claims/`
    it reads the text, in the audit.
  - LIBRARY.md: the build line, and the "Kernels" section (local, not shipped, with the user's decision quoted;
    "the first twelve kernels" for the table).
  - README.md: the "Kernels" line and the build line.
- **Why.** The user's decision of 2026-11-10 ("let's just git ignore kernels and remove on the remote site"):
  the release copy has no kernel, so these pages described a check that no longer happens.
- **Constants whose meaning changed.** None (documents only).

## L189: A `simp` lemma that broke a fixed claim

- **What changed.** (The coordinator's fix, after the author's notes.) `DType.ofBits_f16`, `ofBits_bf16`,
  `ofBits_f32`, `ofBits_f64`, `ofBits_f8e5` and `ofBits_f8e4nv` (`lean/TVA/Lemmas/FloatFacts.lean`) are no longer
  `simp` lemmas; their statements are unchanged, and a proof cites them by name.
- **Why.** As `simp` lemmas they rewrite `DType.ofBits .f16 …` to `ofPat …` before a hypothesis stated on
  `ofBits` can apply. `claims/layer_norm_fwd_f16/Claim.lean` (`zero16_facts`, `simp [FloatV.toReal, h]`) stopped
  elaborating from round 39 on. The same file elaborates against the library before round 39 (the claim
  image built from that revision, `lake env lean` inside it), and fails against round 39's library. With the
  attributes removed, `simp [FloatV.toReal, h]` closes the goal again.
- **Evidence.** `lake build TVA` is green. All 41 `claims/*/Claim.lean` elaborate with `lake env lean` against
  the fixed library, each with exit code 0 and no error. `git diff --stat -- claims/` is empty. No other
  declaration of the library used these lemmas as `simp` lemmas.
- **Constants whose meaning changed.** None (attributes only; no statement or definition changed).
