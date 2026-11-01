# Library batch, round 25: author's notes

Issue: L126 ([ISSUES.md](../../ISSUES.md)), soundness, from `grouped_matmul_tma_f16` F5 (r-gmm) and the
round-24 re-acceptances of r-gmm2, r-fwd2, r-bwd, r-bwd2 and r-matmul2, with the coordinator's
evidence (`evidence/num-dot-sub16`), taken up on the user's decision of 2026-11-01: "please fix L126
after the current run." No review follows this batch (coordinator, as for rounds 21 to 24): it is
checked by `lake build TVA`, `tvr library-check`, `tvr lint library 25 author`, `tvr audit`, the
coordinator's release check, and the evidence. Worked in TVA's tree itself.

**Build and checks.** `lake build TVA` is green (3195 jobs). The new and changed theorems use only
`propext`, `Classical.choice` and `Quot.sound` (`#print axioms` of `Rules.Num.DotLiftAcc.mono`,
`Rules.Num.DotAcc.liftAcc`, `DotLoop.dotAdd_step_of`, `DotLoop.dotAdd_step`, `DotLift.dotAdd_step`,
`DotLift.liftMag_mul_le`, `DotLift.sum_liftMag_mul_le`, `DotLift.DotLiftAcc.dotAcc`,
`Platform.ref.dotLiftAcc`, `Platform.sm90ref.dotF16`, `.dotBF16`, `Dot8Chunk.DotF8Chunk.liftAcc`);
there is no `sorry` or `axiom`. `tvr library-check` passes (hygiene, `lean/TVA.lean` reaches every
module, `leanchecker TVA`), `tvr lint library 25 author` prints `regular` and `tvr audit` prints
`independent`. The release check was not run (the coordinator's instruction). No kernel folder was
edited and `tvr check` was not run. Of the nine kernel libraries, `vec_add_f32`, `softmax_f32`,
`attn_bwd_pre_f16`, `matmul_e5m2` and `attn_fwd_e5m2` build (`lake build <lib>` under the lock); the
four that cite `DotF16` stop building, as listed under "Constants whose meaning changed".

## L126: `DotF16` and `DotBF16` fail when a product with a subnormal operand sets the window: restated relative to lifted magnitudes

- **What changed.**
  - **The facts** (`lean/TVA/Rules/Num/Sm90.lean`). `DotF16` is now
    `∀ p, DotLiftAcc P ⟨.f16, .f16, .f32, p, 0⟩ relDot16 absDot16`, and `DotBF16` the same with `.bf16`
    (before: `DotAcc … (fun _ => True) relDot16 absDot16`). `relDot16` and `absDot16` are unchanged.
    The docstrings give the model, the derivation and the evidence; `DotScaledMx`'s docstring adds that
    on `scaledDomBF16` its `bfloat16` dot has no subnormal operand.
  - **The shape** (`lean/TVA/Rules/Num.lean`). New `DotLiftAcc P cfg rel abs`: `DotAcc` on every value,
    with `L = |vc| + Σ liftMag cfg.a (va k) · liftMag cfg.b (vb k)` in place of `|vc| + Σ|va k · vb k|` in
    the room and in the bound. It is round 24's `Dot8Chunk.LiftAcc`, the same statement, moved into
    `TVA.Rules` because a fact is stated there. `minNormal8` and `liftMag` moved here from
    `Rules/Num/Dot8Chunk.lean` (same names, same namespace), and `minNormal8` gained `.f16 => 2^-14`
    and `.bf16 => 2^-126` (unchanged at `.f8e5`, `.f8e4nv`; `0` elsewhere). New weakening lemmas:
    `DotLiftAcc.mono`, and `DotAcc.liftAcc` (a `DotAcc` fact on every value, with `rel ≥ 0`, gives
    `DotLiftAcc` with the same constants), with two private helpers. The module docstring lists them.
  - `lean/TVA/Rules/Num/Dot8Chunk.lean`: the two moved definitions are gone (a comment says where they
    are); the module docstring names `DotLiftAcc` and the `float16` window of the `mma.sync` path.
  - `lean/TVA/Lemmas/Dot8Chunk.lean`: `Dot8Chunk.LiftAcc` is gone (a comment points to
    `Rules.Num.DotLiftAcc`), and `DotF8Chunk.liftAcc` now states `DotLiftAcc P ⟨da, db, .f32, p, 2^30⟩
    relLift8 absLift8`, the same proposition. Docstrings: the module's, `liftMag_of_le`'s.
  - `lean/TVA/Lemmas/DotLoop.lean`: new `DotLoop.dotAdd_step_of`, one step of `acc += tl.dot(a, b)` (either
    form of `tl.dot_add`) for any magnitude `A ≥ |S|` of the products, given a per-element bound
    `ρ·(|vc| + A) + α` for every accumulator; `dotAdd_step` is now its instance (statement unchanged).
  - `lean/TVA/Lemmas/DotLift.lean` (new, imported by `lean/TVA.lean`): `DotLift.dotAdd_step` (the step
    from a `DotLiftAcc` fact, `DotLoop.dotAdd_step`'s statement with `Σ liftMag a·liftMag b` in place
    of `Σ|a·b|` and no domain), `liftMag_mul_le` (`liftMag da x·liftMag db y ≤ |x·y| + mb·|x| + ma·|y|
    + ma·mb`), `sum_liftMag_mul_le` (summed over `k`), `minNormal8_f16`, `minNormal8_bf16`,
    `DotLiftAcc.dotAcc` (on zero and normal values a `DotLiftAcc` fact is `DotAcc`), and
    `Platform.ref.dotLiftAcc` (satisfiable for `rel K ≥ u`, `abs K ≥ η`).
  - `lean/TVA/Lemmas/Sm90.lean`: `Platform.sm90ref.dotF16`, `.dotBF16` now through `DotAcc.liftAcc`
    (a private `relDot16_nonneg`); the module docstring says so.
  - `docs/rules/num.md`: the header (the lemma module); the weakening lists; the definitions table (a
    `DotLiftAcc` row; `DotAcc`'s typical use); the date line and the `DotF16`, `DotBF16` row of "Facts
    the library states" (the row's first column, `DotF16`, `DotBF16`, is unchanged); in "`tl.dot` (the
    chunk model)" the 16-bit bullet is rewritten (what each instruction computes, the constants'
    derivation, why the lift, how a kernel uses the facts, `input_precision`) and the intro names both
    evidence folders; in the fp8 chunk section, the `mma.sync` path (a `float16` instruction, 25 bits
    below its largest field, truncating), its constants ("largest lifted term"), and `DotLiftAcc` in
    "A bound on the whole dot"; a sentence in "`tl.dot_scaled` on sm_90"; "Device questions" (the
    alignment for 16-bit operands); "Satisfiability" (`DotLiftAcc`, `.dotF16`, `.dotBF16`).
  - `docs/LIBRARY.md`: the rows `Rules.Num`, `Rules.Num.Sm90`, `Rules.Num.Dot8Chunk`, `Sm90` (lemmas),
    `DotLoop`, `Dot8Chunk` (lemmas), and a new row `DotLift`.
  - `docs/PROVER_GUIDE.md`: the stale `tl.dot` cell of its table ("`DotF16`, `DotBF16`, `DotE4m3`, …,
    `DotF32Ieee`; each is a `DotAcc`") now names the facts' shapes as they are.
  - `evidence/num-dot/README.md`: a paragraph "Restated (round 25, library issue L126)".
    `evidence/num-dot-chunk/README.md`: a note in conclusion 4 (the `mma.sync` derivation's "largest
    term" is the largest lifted term; it stands).
  - `evidence/num-dot-sub16-round25/` (new): `README.md`, `run.py`, `out.txt`. The coordinator's
    `evidence/num-dot-sub16` is unchanged.
- **Why.** The user's decision (2026-11-01): "please fix L126 after the current run." `DotF16` and
  `DotBF16` were false of the device: on both instruction paths the tensor core aligns a chunk's terms
  to the largest exponent field and reads a subnormal 16-bit operand's field as the smallest normal's,
  so a product of a subnormal operand and a large partner sets the window while far smaller, and the
  other products lose their bits below the window's unit (`53` times the old bound for `float16`, `13`
  times for `bfloat16`, `evidence/num-dot-sub16`; `113.7` and `14.2` times in the new evidence). No
  listed fact may be false, and subnormal operands must stay covered (tutorial 06's `p.to(float16)`
  makes them, and `randn` cast to `float16` contains them).

  **The route: restate, not mirror round 24.** I restated the two facts with lifted magnitudes
  (`DotLiftAcc`, the coordinator's first route), keeping their names and constants, rather than state
  the 16-bit dot chunk by chunk:
  - The 16-bit window keeps 25 bits, so a chunk-by-chunk statement would buy a kernel little (the
    whole-dot bound is within a factor of about 6 of the derivation's worst case), while the four
    kernels would have to move to a new shape and a composition lemma. With the restated facts a
    kernel cites the same fact, gets the whole-dot bound in `DotAcc`'s shape against
    `|acc| + Σ liftMag a·liftMag b` with `relDot16`, `absDot16` directly (no lemma needed), and takes a
    `K` step from `DotLift.dotAdd_step` where it used `DotLoop.dotAdd_step`, with the same arguments
    less the domain.
  - It stays one device fact per format, as before, evidenced by a bit-exact model and a derivation
    valid for every `K`.
  - Restricting the domain to zero and normal values would drop the subnormal operands the kernels
    meet.

  I reused round 24's `liftMag` instead of a second definition: it moved with `minNormal8` into
  `TVA.Rules.Num`, and `minNormal8` was extended to `float16` and `bfloat16`. I kept the name
  `minNormal8` (now a misnomer, which its docstring says) so that `attn_fwd_e5m2`'s
  `simp [Rules.Num.minNormal8]` and the `Dot8Chunk.liftMag_*` lemmas apply unchanged; the fp8 chunk
  facts state the same as before, since `minNormal8` is unchanged at the fp8 dtypes. The shape
  `Dot8Chunk.LiftAcc` became `Rules.Num.DotLiftAcc`, since a fact must be stated in `TVA.Rules`. No
  kernel names `Dot8Chunk.LiftAcc`, and `DotF8Chunk.liftAcc` states the same proposition.

  **The derivation of the constants** (checkable against the model of
  `evidence/num-dot-sub16-round25`, which equals the device bit for bit on about `3.4·10^8` element
  results). Per instruction, the terms are the accumulator `s` and the chunk's 16 exact products; a
  product's field is `e(a) + e(b)` with `e(x) = max(⌊log₂|x|⌋, emin)`, the accumulator's is
  `max(⌊log₂|s|⌋, −126)`; with `E` the largest and `u = 2^(E−25)`, every term is truncated toward zero
  to a multiple of `u`, the terms are summed exactly, and the sum `S'` is truncated toward zero to
  `float32`.
  - Each of the at most 17 terms loses less than `u` and no more than itself, so `|S' − (s + c)| < 17u`
    (`c` the chunk's exact sum) and `|S'| ≤ |s| + Σ_chunk|a·b|`.
  - The cut to `float32` loses less than `2^-23·|S'| + 2^-149` (an ulp of a normal `S'` is at most
    `2^-23·|S'|`; below `2^-126` the grid is `2^-149`).
  - `2^e(x) ≤ liftMag x` for `x ≠ 0`, so a product's `2^(e(a)+e(b)) ≤ liftMag a·liftMag b`, and the
    accumulator's `2^field ≤ max(|s|, 2^-126)`. So `u ≤ 2^-25·Mj + 2^-151`, with `Mj` the larger of
    `|s|` and the chunk's largest lifted product; `Mj ≤ |s| + A_j`, with `A_j` the chunk's sum of
    `liftMag a·liftMag b`, which is at least `Σ_chunk|a·b|`.
  - So a chunk's error is below `17·2^-25·(|s| + A_j) + 17·2^-151 + 2^-23·(|s| + A_j) + 2^-149`, that
    is `5.25·2^-23·(|s| + A_j) + 5.25·2^-149`. The evidence checks every chunk of the model's chain
    against this: at most `0.902` of it.
  - Truncation toward zero never increases a magnitude, so the accumulator before chunk `j` is at most
    `|acc| + Σ_{i<j}Σ_chunk|a·b|`. Hence `|s| + A_j ≤ L = |acc| + Σ liftMag a·liftMag b`, with no
    compounding.
  - The result's error is the sum of the `⌈K/16⌉` chunks' errors, below
    `⌈K/16⌉·(5.25·2^-23·L + 5.25·2^-149)`. Since `⌈K/16⌉ ≤ (K+15)/16` and `5.25·(K+15)/16 ≤ 2·(K+2)`
    for every `K ≥ 1` (that is `14.75 ≤ 26.75·K`), this is at most `relDot16 K·L + absDot16 K`, with
    `relDot16 K = 2·(K+2)·2^-23`. The relative term keeps a factor `6.9` at `K = 16` and about `6.1` for
    large `K`; the absolute term `(K+2)·2^-126` is far above `⌈K/16⌉·5.25·2^-149`. The measured ratio
    reaches `0.111`, against the derivation's worst case `0.146` at `K = 16`.
  - Room: every partial accumulator and every `|S'|` is at most `|acc| + Σ|a·b| ≤ L`, so with the
    fact's room nothing overflows, and the result has a value.
  - Satisfiable: `Platform.sm90ref` (one rounding of the exact result) meets `DotAcc` with these
    constants, hence `DotLiftAcc` (`DotAcc.liftAcc`).

  **Other facts checked for the same gap.**
  - `DotScaledE4m3`, `DotScaledE4m3E2m1`: their domain `scaledDomBF16` admits only `0` and normal
    `bfloat16` scaled values (a subnormal `e4m3` element, `2^-9`, is a normal `bfloat16`), so the
    emulated `bfloat16` dot has no subnormal operand, its lifted magnitudes are `|a·b|`, and `DotBF16`
    gives their bound as stated. No gap; evidence Part G3 (at most `0.0057` of the bound on window
    inputs).
  - The fp8 chunk facts on `mma.sync`: fp8 is converted to `float16`, a subnormal `float8e5` value to a
    subnormal `float16`, read as `2^-14`, the same as `liftMag`'s `float8e5` threshold; every
    `float8e4nv` value is a normal `float16`. Round 24's per-chunk derivation ("within `17·2^-23` of its
    largest term") holds with "largest lifted term", which `Mj` bounds, and its `2^-23` is above the
    window's `2^-25`. No gap; Part G2 (a subnormal `float8e5` operand with `2^15`, the products below the
    `float16` window lost: at most `10^-4` of the facts' bound). The text in `docs/rules/num.md` and in
    num-dot-chunk's README now says "lifted".
  - `DotF32Ieee`: no tensor core, `fma.rn.f32`, which keeps subnormals (num-dot Part 4, bit for bit with
    `tiny` inputs). No window, no gap.
  - A `tf32`, `tf32x3`, `bf16x3` or `bf16x6` dot of `float32` operands is not stated by the library.
    For 16-bit operands every `input_precision` gives the default's bits, subnormal operands included
    (Part G1), so `∀ p` stands.
- **Evidence.** [evidence/num-dot-sub16-round25](../../../../evidence/num-dot-sub16-round25/README.md)
  (Triton 3.8.0, H200 NVL, GPU 0, 2026-11-01; about 7 minutes).
  - **Part A**, the window, on both paths and in both formats: every rule of the model, each against
    exact rationals, and every device result equal to the model.
    - The window keeps 25 bits below the largest field: `1 − 1 + 2^-25` keeps `2^-25`, and `2^-26` is
      lost. The same holds for `1.5·1.5` and for the largest significands' product, whose fields are
      `0`.
    - Each term is truncated toward zero, of either sign, with no partial sums.
    - The accumulator is in the same window, with its own field (`1`, `3`).
    - The sum is truncated toward zero to `float32` (`1 + 3·2^-25` gives `1`).
    - A subnormal operand, in `a` or in `b`, reads as the smallest normal: `float16` `2^-23·2^14` has
      `E = 0`, so `2^-28` is lost, which the same product from normals keeps; `bfloat16`
      `2^-133·2^124` has `E = −2`.
    - L126's two cases are reproduced.
    - Subnormal accumulators (field `−126`) and subnormal results (kept, on the grid `2^-149`).
  - **Parts B and C**, both formats, the seven block shapes of num-dot Part 2, every `K` that compiles
    (16 to 2048 on `16 × 16`, to 256 on `128 × 128`; the next rejected on the host, recorded).
    - Fifteen kinds of inputs: the old ones of num-dot Part 2 and `softmax`, `sub`, `subwin`,
      `winworst`, `subrand`, `trunc24`, `bits`. Kinds with subnormal operands are drawn 16 times per
      configuration.
    - 70778880 elements: none outside the new bound; every one equal to the model.
    - The largest new ratio is `0.1111` (`float16`) and `0.1107` (`bfloat16`), both `winworst` at
      `K = 16`.
    - The old bound fails on 1492340 elements, all `subwin` but 49 `subrand`, up to `113.67` times
      (`float16`) and `14.17` times (`bfloat16`).
  - **Part D**, the kernels' shapes. D1: `[128, D] × [D, 64]` with `b` transposed, with and without an
    accumulator, and `[128, 64] × [64, D]` with `a` converted in registers, for `D ∈ {64, 128, 256}`;
    the `128 × 128 × 64` and `128 × 256 × 64` tiles; 4 and 8 warps.
    - 99614720 elements, none outside, all equal to the model; largest ratio `0.0897`.
    - The old bound fails on 2058216 of them (`subwin`), up to `30.1` times.
    - D2: 264 chains (attention's `p.to(…)` from a softmax or `subwin` blocks, with and without a
      rescale, and tutorial 03's `K` loop; `T` up to 256). Every dot equals the model (169951232
      element results), and so does each whole chain run as one dot (3022848).
    - Largest ratio `0.0489` per dot and `0.0292` for a whole chain. The softmax chains stay inside even
      the old bound; the `subwin` chains exceed it up to `29.1` times.
  - **Part E**: the chunks are 16 consecutive `k` in increasing order on both paths (168 of 168, 3000
    of 3000).
  - **Part F**: L126's cases are at `53.18` and `13.12` times the old bound and `0.104`, `0.1025` of the
    new one.
  - **Part G**: G1, `input_precision` changes no bit, in 48 configurations. G2 and G3 as above.
  - **Zero results**: all 577048 that equal the model's value have its sign too.
- **Constants whose meaning changed.**
  - **Meaning changed:**
    - `TVA.Rules.Num.DotF16`, `TVA.Rules.Num.DotBF16`: now `DotLiftAcc`, relative to lifted
      magnitudes, with room on `L`. They are weaker than before on inputs with a subnormal operand,
      and the same for zero and normal operands.
    - `TVA.Rules.Num.minNormal8`: now `2^-14` at `.f16` and `2^-126` at `.bf16` (was `0`), so
      `TVA.Rules.Num.liftMag` changed at `.f16` and `.bf16`. Both moved to `Rules/Num.lean`, same full
      names; they are unchanged at the fp8 dtypes, so the fp8 chunk facts (`DotChunkAcc`,
      `DotF8Chunk`, `DotE4m3Chunk`, …) state the same as before.
  - **Withdrawn:** `TVA.Dot8Chunk.LiftAcc`, whose statement is now `TVA.Rules.Num.DotLiftAcc`.
  - **Statement re-expressed, same proposition:** `TVA.Dot8Chunk.DotF8Chunk.liftAcc` (now
    `DotLiftAcc …`).
  - **New:** `TVA.Rules.Num.DotLiftAcc`, `DotLiftAcc.mono`, `DotAcc.liftAcc`;
    `TVA.DotLoop.dotAdd_step_of`; `TVA.DotLift.dotAdd_step`, `.liftMag_mul_le`,
    `.sum_liftMag_mul_le`, `.minNormal8_f16`, `.minNormal8_bf16`, `.DotLiftAcc.dotAcc`;
    `TVA.Platform.ref.dotLiftAcc`.
  - **Proofs changed, statements unchanged:** `TVA.DotLoop.dotAdd_step`, `TVA.Platform.sm90ref.dotF16`,
    `.dotBF16`.
  - **Docstrings changed, statements unchanged:** `TVA.Rules.Num.DotScaledMx`,
    `TVA.Dot8Chunk.liftMag_of_le`, `TVA.Dot8Chunk.DotF8Chunk.liftAcc`, and the module docstrings of
    `Rules/Num.lean`, `Rules/Num/Dot8Chunk.lean`, `Lemmas/Dot8Chunk.lean`, `Lemmas/DotLoop.lean` and
    `Lemmas/Sm90.lean`.
  - `docs/rules/num.md` changed, so every kernel item whose cone reaches it (each num-fact item)
    re-opens; the `assumption:dot` items of the four kernels below change meaning.
  - **Kernel modules that stop building** (found with `lake build <lib>` under the lock for every
    `[[lean_lib]]`, and `grep` of the kernels' Lean; no kernel folder edited, `tvr check` not run).
    Each kernel's failing module applies `DotF16` as a `DotAcc` (with a domain argument, or through
    `DotLoop.dotAdd_step`). Each must take its dot steps from the restated fact (`DotLift.dotAdd_step`,
    or `DotF16` applied directly) and bound `Σ liftMag a·liftMag b`, for example with
    `DotLift.sum_liftMag_mul_le` (`2^-14` for `float16`). `relDot16`, `absDot16`, `DotLoop.stepRel`,
    `stepAbs` and `Platform.sm90ref.dotF16` keep their names and statements.
    - `matmul_f16`: `Matmul.Basic` (line 237), and through it `Matmul.Program` and the root `Matmul`.
      `Matmul.Defs` and `Matmul.Arith` build.
    - `attn_fwd_f16`: `AttnFwd.Inner` (`qk_lane`, `pv_lane`, lines 79 and 112), and through it
      `AttnFwd.Program` and the root `AttnFwd`. `Model`, `Real`, `Lane`, `RowMax`, `RowAcc`, `Step`,
      `Final`, `Facts`, `Row` and `Tile` build.
    - `attn_bwd_f16`: `AttnBwd.Acc` (`dot0`, `accI_dotAdd`, lines 83 and 256), and through it `Basic`,
      `Step`, `DkDv`, `Dq`, `Out`, `Program` and the root `AttnBwd`. `Model`, `Spec` and `Lane` build.
    - `grouped_matmul_tma_f16`: `GroupedMatmulTma.Basic` (line 290), and through it `Tile`, `Program`
      and the root `GroupedMatmulTma`. `Defs` and `Sched` build.
    - The kernels' `[rule]` docstrings of `dot` still describe the old bound (`relDot16 K·(|acc| +
      Σ|a·b|)`); their provers update them.

