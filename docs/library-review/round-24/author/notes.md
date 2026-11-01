# Library batch, round 24: author's notes

Issues: L124 ([ISSUES.md](../../ISSUES.md)), from `attn_fwd_e5m2`'s round-1 reviewer r-e5fwd2 (F2,
a library report), taken up on the user's request of 2026-10-31: "please tighten the fp8 kernel
error bound"; and L125 (soundness), which the coordinator filed during the round from L124's
evidence (Part F) and added to the batch. No review follows this batch (coordinator, as for rounds
21 to 23): it is checked by `lake build TVA`, `tvr library-check`, `tvr lint library 24 author`,
`tvr audit` and the evidence. Worked in TVA's tree itself.

**Build and checks.** `lake build TVA` is green (3194 jobs). The new theorems use only `propext`,
`Classical.choice` and `Quot.sound` (`#print axioms` of `Dot8Chunk.DotF8Chunk.err`,
`Dot8Chunk.errB_le`, `Platform.ref.dotChunkAcc`, `Platform.sm90ref.dotE5m2Chunk`,
`DotChunkAcc.mono`); there is no `sorry` or `axiom`. `tvr library-check` passes (hygiene, `lean/TVA.lean`
reaches every module, `leanchecker TVA`). `tvr lint library 24 author` prints `regular` and
`tvr audit` prints `independent`. The release check was not run (the coordinator's instruction: it
needs every kernel re-gated first). No kernel folder was edited and `tvr check` was not run. L124
alone changed no kernel's build; L125 withdraws `DotE5m2`, so `matmul_e5m2` and `attn_fwd_e5m2` stop
building (listed under L125), and the other seven kernel libraries build. Every kernel item whose
cone reaches `docs/rules/num.md` (each num-fact item) re-opens through its digest, and its reviewer
re-accepts it.

## L124: an fp8 `tl.dot` chunk by chunk, new facts `DotE4m3Chunk` … beside `DotE5m2`

- **What changed.** A new fact for each fp8 pair, beside the old ones, which are unchanged:
  - `lean/TVA/Rules/Num/Dot8Chunk.lean` (new, namespace `TVA.Rules.Num`): `minNormal8` (`2^-14` for
    `float8e5`, `2^-6` for `float8e4nv`), `liftMag d q` (`0` for `0`, else `max |q| (minNormal8 d)`),
    `chunk8 K j` (the `k : Fin K` with `k / 32 = j`), `chunkSum8`, the shape `DotChunkAcc P cfg rho
    sigma alpha` (per output element, with values and the room `DotF8` asks: the result has a value
    `vr`, and there are `s : ℕ → ℚ` with `s 0 = vc`, `s (chunks8 K) = vr` and, for each `j <
    chunks8 K` and every `Mj ≥ |s j|` that bounds each `liftMag cfg.a (va k) * liftMag cfg.b (vb k)` of
    chunk `j`, `|s (j+1) − (s j + chunkSum8 va vb j)| ≤ rho·Mj + sigma·|s j + chunkSum8 va vb j| +
    alpha`), the constants `rhoChunk8 = 2^-7`, `sigmaChunk8 = 2^-12`, `alphaChunk8 = 2^-126`, the
    family `DotF8Chunk P a b` (every `input_precision`, `maxImpreciseAcc = 2^30`, accumulator
    `float32`), the facts `DotE4m3Chunk`, `DotE5m2Chunk`, `DotE4m3E5m2Chunk`, `DotE5m2E4m3Chunk`
    (each of type `Platform → Prop`), and `DotChunkAcc.mono`.
  - `lean/TVA/Lemmas/Dot8Chunk.lean` (new): `Dot8Chunk.sum_chunkSum8`, `psum` (exact partial sums),
    `errB` (the composed bound from the exact inputs), `chain_err` (any chain of steps bounded as the
    facts bound them stays within `errB` of the exact partial sums), `errB_le` (the closed form
    `(1+ρ+σ)^n·Σ_j (ρ·max |S j| (m j) + σ·|S (j+1)| + α)`), `chunkLift8` (a chunk's largest lifted
    product), `DotChunkAcc.err` and `DotF8Chunk.err` (the result within `errB` of `acc + Σ a·b`, for any
    bounds `m j` on the chunks' lifted products); `Platform.ref.dotChunkAcc` and
    `Platform.sm90ref.dotF8Chunk`, `.dotE4m3Chunk`, …, `.dotE5m2E4m3Chunk` (satisfiability).
  - `lean/TVA.lean` imports both.
  - `docs/rules/num.md`: the module in the header; `DotChunkAcc` in the table of definitions and
    `DotChunkAcc.mono` among the weakening lemmas; the chunk facts in "What a kernel reviewer checks"
    (the fp8 row); four rows in "Facts the library states" (one per fact, `DotE5m2Chunk` written out,
    the others "as `DotE5m2Chunk`", each its subnormal threshold), and the date of their measurement;
    a new section "`tl.dot` of fp8, chunk by chunk" (what the facts say, how a kernel uses them,
    what Triton runs and the window, the derivation of the constants, what they do not state); in
    "`tl.dot` (the chunk model)", a paragraph on the subnormal operand that breaks `relDot8`'s
    derivation (below); in "Device questions", the alignment by exponent fields; in "Satisfiability",
    `Platform.sm90ref.dotE4m3Chunk` … and `Platform.ref.dotChunkAcc`.
  - `docs/LIBRARY.md`: a row for each new module.
  - `evidence/num-dot-chunk/` (new): `README.md`, `run.py`, `out.txt`.
- **Why.** The user's goal (2026-10-31): "please tighten the fp8 kernel error bound." `attn_fwd_e5m2`
  can only cite `DotE5m2`, which bounds each fp8 `tl.dot` by `relDot8 K·(|acc| + Σ|a·b|)`: inside the
  softmax's exponent a score's `relDot8 D·Σ|q·k|` moves each weight by up to `2^(1.5·relDot8 D·X)`, and
  `tl.dot(p, v, acc)` compounds `relDot8 64` of `|acc|` per block. The device does much less: each
  `wgmma` keeps 13 bits below the chunk's largest term, so a chunk's error is bounded by that term, not
  by the magnitudes' sum. The new facts state that, per chunk, so a kernel can bound a score by its
  chunks' largest terms and partial sums (`Dot8Chunk.DotF8Chunk.err` with `chunkLift8`), and each
  chunk of `tl.dot(p, v, acc)` by `2^-7·max(|acc|, the chunk's largest lifted product) + 2^-12·|new
  value| + 2^-126`. The derivation of each constant, checkable against the evidence's model:
  - **The model** (evidence Part A, equal to the device bit for bit in Parts B and D, about `2·10^8`
    element results). Per `wgmma` chunk, with the accumulator `s` and 32 exact products: exponent
    fields `max(⌊log₂|s|⌋, −126)` and `e(a) + e(b)`, `e(x) = max(⌊log₂|x|⌋, emin)`; `E` the largest,
    `u = 2^(E−13)`; each term truncated toward zero to a multiple of `u`; the exact sum truncated toward
    zero to 14 significant bits.
  - **The error of a chunk.** Each term loses less than `u` and no more than itself: the truncated sum
    `S'` is within `33u` of `s + c_j`. The cut to 14 bits loses less than `2^-13·|S'|` and
    `|S'| ≤ |s + c_j| + 33u`. So the error is below `33·(1 + 2^-13)·u + 2^-13·|s + c_j|`.
  - **`u` against `Mj`.** `2^e(x) = max(2^⌊log₂|x|⌋, 2^emin) ≤ liftMag x` for `x ≠ 0`, so a product's
    `2^(e(a)+e(b)) ≤ liftMag a · liftMag b ≤ Mj`; the accumulator's `2^max(⌊log₂|s|⌋, −126) ≤ |s| ≤ Mj`
    when `s` is normal, and `2^-126` when it is subnormal. So `u ≤ 2^-13·Mj + 2^-139`.
  - **The constants.** The error is below `33·(1 + 2^-13)·2^-13·Mj + 2^-13·|s + c_j| +
    33·(1 + 2^-13)·2^-139`: `33·(1 + 2^-13)·2^-13 = 0.00403 < rhoChunk8 = 2^-7 = 0.00781` (a factor
    `1.94`, the margin `relDot8` keeps over its per-chunk `33·2^-13`); `2^-13 < sigmaChunk8 = 2^-12` (a
    factor 2); `34·2^-139 < alphaChunk8 = 2^-126`. The measured worst chunk is `0.468` of the bound
    (Part C's `chunk ratio`), consistent with these margins.
  - **`mma.sync`** (smaller blocks): two `float16` instructions per chunk, each keeping every term's
    bits down to at least `2^-23` of its largest term and rounding into `float32` (evidence/num-dot
    Part 1; Part A2). The first's error is at most `17·2^-23·Mj + 2^-23·|s'|` with `|s'| ≤ 17·Mj` (plus
    that error), the second's at most `17·2^-23·max(|s'|, Mj) + 2^-23·|s''|` with `|s''| ≤ 33·Mj`: in all
    about `360·2^-23·Mj ≈ 2^-14.5·Mj`, far below `2^-7·Mj`. Accumulators pass unchanged, subnormals
    included, and fp8 products are at least `2^-32`, so there is no absolute term (Part A2).
  - **The order.** Increasing `k`, chunk by chunk, on both paths (Part E), so the facts pin it.
  - **The room** is `DotF8`'s, which every kernel citing `DotE5m2` already proves; truncation toward
    zero never increases a magnitude, so the device's partial values stay within `|acc| + Σ|a·b|`.
  - **Satisfiable.** `Platform.ref.dotChunkAcc`: with the exact partial sums as the partial values and
    the one rounding as the last step, any `rho ≥ 0`, `sigma ≥ u`, `alpha ≥ η` hold; so the chunk facts
    hold of `Platform.sm90ref` with every other sm_90 fact.
- **Evidence.** [evidence/num-dot-chunk](../../../../evidence/num-dot-chunk/README.md) (Triton 3.8.0,
  H200 NVL, GPU 0, 2026-10-31). Part A: 27 structured single elements establish each rule of the model
  (the unit against the largest term, truncation toward zero, one window for the chunk with the
  accumulator in it, the 14-bit cut, a subnormal operand read as the smallest normal for each pair,
  subnormal accumulators); every device result equals the model. Part A2: the `mma.sync` path. Parts B
  and C: the four pairs, the seven block shapes of evidence/num-dot Part 2 at every `K` that compiles
  (32 up to 8192 on `16 × 16`; larger `K` rejected on the host), and `_attn_fwd`'s shapes with
  `num_warps = 4` (`[128, D] × [D, 64]`, `D ∈ {64, 128, 256}`, `b` transposed, with and without an
  accumulator; `[128, 64] × [64, D]` with `a` converted in registers); thirteen kinds of adversarial
  inputs (the window's worst cases among them); about `3.2·10^7` elements, each against the interval
  the facts imply for it (the per-chunk bound composed with the exact partial sums and the worst case
  of the unknown partial values, exact in `float64` or rationals): none outside; the largest ratio
  `0.4678` on `wgmma` (`float8e4nv` operands, `subwin`: a subnormal operand setting the window, 31
  products just below `u`, an accumulator just below `u`), `0.0002` on `mma.sync`; every `wgmma`
  element equal to the model. Part D: 304 chains of `tl.dot(p, v, acc)` with `p` converted in
  registers, `T` up to 256 blocks, with and without a rescale: none outside, the model exact on every
  dot and on each whole chain; largest ratio `0.4591` per dot, `0.4514` for a whole chain. Part E: on
  both paths every order-dependent outcome agrees with increasing `k`, chunk by chunk.
- **Constants whose meaning changed.** None of the existing constants: `DotE4m3`, `DotE5m2`,
  `DotE4m3E5m2`, `DotE5m2E4m3`, `DotF8`, `relDot8`, `absDot8`, `chunks8` and every other existing Lean
  file are unchanged, statements and docstrings. New constants: `TVA.Rules.Num.minNormal8`, `liftMag`,
  `chunk8`, `chunkSum8`, `DotChunkAcc`, `rhoChunk8`, `sigmaChunk8`, `alphaChunk8`, `DotF8Chunk`,
  `DotE4m3Chunk`, `DotE5m2Chunk`, `DotE4m3E5m2Chunk`, `DotE5m2E4m3Chunk`, `DotChunkAcc.mono`; and
  `TVA.Dot8Chunk.*` (lemmas), `TVA.Platform.ref.dotChunkAcc`, `TVA.Platform.sm90ref.dotF8Chunk`,
  `.dotE4m3Chunk`, `.dotE5m2Chunk`, `.dotE4m3E5m2Chunk`, `.dotE5m2E4m3Chunk`. `docs/rules/num.md` changed
  (above), so every kernel item that cites a fact of `TVA.Rules.Num` re-opens; no `[rule]` field's
  meaning changed. No kernel stops building.
- **Found on the way (for a new library issue, soundness).** `DotE4m3`, `DotE5m2`, `DotE4m3E5m2` and
  `DotE5m2E4m3` are false of the device. The `wgmma` window follows exponent fields, and a subnormal
  fp8 operand's field reads as the smallest normal: the product `2^-9 · 2^8 = 2^-1` of `float8e4nv`
  values sets `u = 2^-11` (from normal operands, `2^-14`), so 31 products `1.875·2^-12` and an
  accumulator `2^-11·(1 − 2^-23)` are lost whole. The error is then `3.65` times `relDot8 32·(|acc| +
  Σ|a·b|) + absDot8 32` (Part F; `1.79` times at `K = 64`; `1.73` times for `float8e5` at `K = 32`,
  whose subnormal `2^-16 · 2^15 = 2^-1` sets `u = 2^-12`; both mixed pairs likewise), and Part C
  counts about 6.6·10^5 such elements (every one of kind `subwin`, at `K = 32` and `64`). `relDot8`'s derivation
  (`33·2^-13` of the magnitudes' sum per chunk) assumed the window follows the largest term. Following
  this batch's instruction the facts are left as they are, and `docs/rules/num.md` says so where it
  derives them ("`tl.dot` (the chunk model)") and in "Device questions". Kernels that cite them:
  `matmul_e5m2` and `attn_fwd_e5m2` (`DotE5m2`). A fix could restrict `dom` to zero and normal
  values, or state the bound relative to `|acc| + Σ liftMag a·liftMag b` (derivable from the chunk facts,
  with `(1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1` in place of `relDot8`); the new facts already hold on these
  inputs (ratio below `0.47`). The coordinator filed this as L125 in the same round; its fix (below)
  withdraws the four facts and `DotF8`, so the statements above that they are "left as they are" and
  that "no kernel stops building" describe L124 alone.

## L125: the fp8 whole-dot facts `DotE4m3` … are false on a subnormal operand: withdrawn, with a whole-dot bound by lemma

- **What changed.**
  - **Withdrawn**, removed from the Lean and from "Facts the library states": `TVA.Rules.Num.DotF8`,
    `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3` (`lean/TVA/Rules/Num/Sm90.lean`, a comment in
    their place saying so), and their satisfiability lemmas `TVA.Platform.sm90ref.dotE4m3`, `.dotE5m2`,
    `.dotE4m3E5m2`, `.dotE5m2E4m3` with the private `dot8` and `chunks8_pos` (`lean/TVA/Lemmas/Sm90.lean`).
  - **Moved, statements unchanged**: `chunks8`, `relDot8`, `absDot8`, from `Rules/Num/Sm90.lean` to
    `Rules/Num/Dot8Chunk.lean` (same namespace `TVA.Rules.Num`, same names). They are now only the
    chunk facts' room, and their docstrings say so.
  - **New lemmas** in `lean/TVA/Lemmas/Dot8Chunk.lean` (which now imports `TVA.Lemmas.DotLoop`):
    `Dot8Chunk.LiftAcc P cfg rel abs`, `DotAcc`'s shape without `dom`, with
    `L = |vc| + Σ liftMag cfg.a (va k) * liftMag cfg.b (vb k)` in place of `|vc| + Σ|va k * vb k|` in
    both the room and the bound. `DotChunkAcc.liftErr`: under `DotChunkAcc P cfg ρ σ α` with
    `ρ, σ, α ≥ 0`, the result is within `DotLoop.bound (ρ + σ) α (chunks8 K) L`. `relLift8 K =
    (1 + (rhoChunk8 + sigmaChunk8))^chunks8 K − 1` and `absLift8 K = chunks8 K·(1 + (rhoChunk8 +
    sigmaChunk8))^chunks8 K·alphaChunk8`. `DotF8Chunk.liftAcc : DotF8Chunk P da db → ∀ p, LiftAcc P
    ⟨da, db, .f32, p, 2^30⟩ relLift8 absLift8`. Helpers: `chunkLiftSum8`, `sum_chunkLiftSum8`,
    `liftMag_nonneg`, `abs_le_liftMag`, `liftMag_zero`, `liftMag_of_le` (`liftMag d q = |q|` when
    `minNormal8 d ≤ |q|`), `liftMag_le` (`liftMag d q ≤ |q| + minNormal8 d`), `minNormal8_nonneg`,
    `abs_mul_le_liftMag`, `relDot8_nonneg`, `relDot8_le_relLift8`, `absDot8_le_absLift8`,
    `relLift8_nonneg`, `absLift8_nonneg`.
  - `docs/rules/num.md`:
    - the fp8 row is removed from "Facts the library states";
    - the fp8 row of "What a kernel reviewer checks" lists only the chunk facts;
    - in "`tl.dot` (the chunk model)", the list of facts drops the fp8 ones, and the fp8 bullet now says
      the four facts were derived, are false, and are withdrawn, and that num-dot's Part 2 had no input
      where a subnormal operand sets the window;
    - in the chunk section, the intro is new, the room is written with `relDot8` and `absDot8`, a new
      paragraph "A bound on the whole dot" covers `DotF8Chunk.liftAcc`, and the comparison with
      `relDot8`'s margin is gone;
    - `DotChunkAcc`'s row in the definitions (its room, and the new lemma as a use);
    - "Device questions".
  - `docs/LIBRARY.md`: the rows `Rules.Num.Sm90`, `Rules.Num.Dot8Chunk` and `Dot8Chunk` (lemmas).
  - The docstrings of `Rules/Num/Dot8Chunk.lean` (the module's, `DotChunkAcc`'s) no longer cite
    `DotF8`.
  - `evidence/num-dot/README.md`: a paragraph "Withdrawn" says the fp8 facts in its table are no
    longer stated and that Part 2 had no input where a product with a subnormal operand sets the
    alignment window, and points to evidence/num-dot-chunk Part F. `evidence/num-dot-chunk/README.md`:
    the question and conclusion 6 say the facts are withdrawn, and conclusion 6 names the lemma.
- **Why.** The four facts are false of the device (evidence/num-dot-chunk Part F, up to `3.65` times
  their bound). The coordinator reproduced this independently: at `K = 32`, with `a[0] = 2^-16`
  (subnormal `float8e5`), `b[0] = 2^15` and 31 products `1.75·2^-13`, `wgmma` gives `0.5` where the
  exact sum is `0.50662`, `1.67` times `DotE5m2`'s bound. The cause: a subnormal operand's exponent
  field reads as the smallest normal's, so a small product can set the window. I chose to withdraw
  them and derive the whole-dot bound by a lemma (the coordinator's preference), rather than restate
  them with lifted magnitudes as device facts:
  - A restated fact would be a second device claim about the same instructions. It would need its
    own evidence and review, though it follows from the chunk facts. With one device fact per pair,
    what reviewers must trust stays smaller.
  - Restricting `dom` to zero and normal values would drop the subnormal operands that tutorials 03
    and 06 produce (`randn` cast to fp8). `liftMag` keeps them covered.
  - The lemma costs little. Per chunk it charges `2^-7 + 2^-12` in place of `2^-7` (a factor
    `1.03`). For zero and normal operands `L = |acc| + Σ|a·b|`, so the bound has the old shape.

  The derivation. Take chunk `j` with `Mj = |s j| + A j`, where `A j = Σ_chunk liftMag a_k · liftMag
  b_k`. That `Mj` bounds `|s j|` and every lifted product of the chunk (each is at most `A j`, the
  terms being nonnegative). And `|c_j| ≤ A j`, because `|a·b| ≤ liftMag a · liftMag b`. So
  `|s j + c_j| ≤ |s j| + A j`, and the fact's step is at most `(ρ + σ)·(|s j| + A j) + α`. That is
  `DotLoop.accum_step`'s hypothesis with `ρ + σ` in place of `ρ`, so by induction on `j` the result is
  within `DotLoop.bound (ρ + σ) α n (|vc| + Σ_j A j)`. Since `Σ_j A j = Σ_k liftMag a_k · liftMag b_k`
  (`sum_chunkLiftSum8`), this is `relLift8 K·L + absLift8 K`. The room: `L ≥ |vc| + Σ|a·b|`,
  `relLift8 ≥ relDot8` and `absLift8 ≥ absDot8`, so `LiftAcc`'s room implies the chunk facts' room.
  The chunk facts keep `relDot8` and `absDot8` as their room, unchanged. A room is a hypothesis of the
  fact, so it cannot make the fact false; and changing it would change an evidenced statement.

  A kernel with `tl.dot(a, b, acc)` per `K` step (tutorial 03, `matmul_e5m2`) uses
  `DotF8Chunk.liftAcc` where it used `DotE5m2`. Each step's magnitude becomes `Σ liftMag a·liftMag b`,
  which `liftMag_le` bounds by `Σ (|a| + 2^-14)·(|b| + 2^-14)` for `float8e5`.
- **Evidence.** This fix states no new device fact, so there is no new run.
  - `Dot8Chunk.DotF8Chunk.liftAcc` is a theorem of the chunk facts (`#print axioms`: `propext`,
    `Classical.choice`, `Quot.sound`; likewise `DotChunkAcc.liftErr`, `liftMag_le`, `liftMag_of_le`).
  - The chunk facts' evidence is evidence/num-dot-chunk (L124, above). It covers the inputs where a
    subnormal operand sets the window: the `subwin` kind at every pair, and Part F. There the chunk
    facts hold, with a ratio of at most `0.4678`.
  - `lake build TVA` is green (3194 jobs). `lake build` of `Matmul`, `AttnFwd`, `AttnBwd`,
    `AttnBwdPre`, `GroupedMatmulTma`, `Softmax` and `VecAdd`, each under the lock: every one completes.
    `MatmulE5m2.Arith` builds. `MatmulE5m2.Defs` and `AttnFwdE5m2.Model` fail with "Unknown identifier
    `Rules.Num.DotE5m2`", as expected (below).
- **Constants whose meaning changed.**
  - **Withdrawn:**
    - `TVA.Rules.Num.DotF8`, `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3`;
    - `TVA.Platform.sm90ref.dotE4m3`, `.dotE5m2`, `.dotE4m3E5m2`, `.dotE5m2E4m3` (and the private
      `dot8` and `chunks8_pos`).
  - **Moved, with unchanged statements and new docstrings:** `TVA.Rules.Num.chunks8`, `relDot8`,
    `absDot8`.
  - **Docstrings changed, statements unchanged:** `TVA.Rules.Num.DotChunkAcc` and the module
    docstring of `Rules/Num/Dot8Chunk.lean`.
  - **New:** `TVA.Dot8Chunk.LiftAcc`, `relLift8`, `absLift8`, `DotChunkAcc.liftErr`,
    `DotF8Chunk.liftAcc` and the helpers above.
  - **Kernel modules that stop building.** I found these by grep, then followed the import closure;
    the kernel folders were not edited and `tvr check` was not run.
    - `matmul_e5m2`: `MatmulE5m2.Defs` (its field `dot : Rules.Num.DotE5m2 P`), and through it
      `MatmulE5m2.Basic` (`acc_step` takes `Rules.Num.DotE5m2 P`), `MatmulE5m2.Program` and
      `MatmulE5m2.Reject`, and the root `MatmulE5m2` (`Platform.sm90ref.dotE5m2`).
      `MatmulE5m2.Arith` still builds.
    - `attn_fwd_e5m2`: `AttnFwdE5m2.Model` (its field `dot : Rules.Num.DotE5m2 P`), and through it
      every module of the kernel: `Real`, `Lane`, `RowMax`, `RowAcc`, `Step`, `Final`, `Mag`,
      `Facts`, `Row`, `Tile`, `Inner` (`qk_lane`, `pv_lane` and the step lemmas take
      `Rules.Num.DotE5m2 P`), `Program`, and the root `AttnFwdE5m2` (`Platform.sm90ref.dotE5m2`).

    Both must cite `DotE5m2Chunk` and re-prove their dot steps, from `DotF8Chunk.liftAcc` or, for
    tighter bounds, `DotF8Chunk.err`. Their uses of `relDot8`, `absDot8` and `chunks8` still resolve
    (same names, same statements).
  - `docs/rules/num.md` changed, so every num-fact item re-opens. No other `[rule]` field's meaning
    changed.
  - **Stale wording in files I did not edit (for the wording backlog):**
    - `tools/tvr.py:2297` and `docs/REVIEW_PROTOCOL.md:569` give `DotF8` as the example of a family
      that a `[rule]` field may not cite; `DotF8Chunk` is now the family to name.
    - `evidence/tool-round13/run.py` and `evidence/tool-round16/run.py` test the field
      `Rules.Num.DotF8 P .f8e4nv .f8e5` as a refused citation. It is still refused, now also as an
      unknown name.

