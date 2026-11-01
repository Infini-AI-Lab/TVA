# Library batch, round 26: author's notes

Issues: L124 (reopened by lr-24b) and L127 to L133 (wording, from lr-24a, lr-24b, lr-25a and lr-25b)
([ISSUES.md](../../ISSUES.md)). The user's goal (2026-11-01): "run one round of library review and
fix." The round 24/25 library review is `round-24/lr-24a`, `round-24/lr-24b`, `round-25/lr-25a`,
`round-25/lr-25b` (reports and `work/` read in full); this batch is the fix. No review follows it
(coordinator): it is checked by `lake build TVA`, `tvr library-check`, `tvr lint library 26 author`,
`tvr audit`, the coordinator's release check, and the evidence. Worked in TVA's tree itself.

**Build and checks.** `lake build TVA` is green (3195 jobs). The new theorems use only `propext`,
`Classical.choice` and `Quot.sound` (`#print axioms` of each theorem named under L124); there is no
`sorry` or `axiom`. `tvr library-check` passes (hygiene, `lean/TVA.lean` reaches every module,
`leanchecker TVA`), `tvr lint library 26 author` prints `regular` and `tvr audit` prints
`independent`. The release check was not run (the coordinator's instruction). No kernel folder was
edited and `tvr check` was not run. All nine kernel libraries build (`lake build <lib>` under the
lock, for every `[[lean_lib]]` of `lean/lakefile.toml`): `vec_add_f32`, `softmax_f32`, `matmul_f16`,
`matmul_e5m2`, `attn_fwd_f16`, `attn_fwd_e5m2`, `attn_bwd_pre_f16`, `attn_bwd_f16`,
`grouped_matmul_tma_f16`.

## L124: the fp8 chunk facts compound over a fed-back accumulator: new loss facts, each chunk within the smaller of two bounds

- **What changed.**
  - **New facts** (`lean/TVA/Rules/Num/Dot8Chunk.lean`, section "Each chunk's error, the smaller of
    two bounds"): `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss`,
    `DotE5m2E4m3ChunkLoss`, each `DotF8ChunkLoss P a b`, that is
    `∀ p, DotChunkLoss P ⟨a, b, .f32, p, 2^30⟩ rhoChunk8 sigmaChunk8 alphaChunk8 kappaChunk8 rhoLossChunk8`.
    The shape `DotChunkLoss P cfg ρ σ α κ ρ'` is `DotChunkAcc` (the same hypotheses, room and partial
    values `s 0 = vc, …, s (chunks8 K) = vr`) with each step's bound replaced by
    `min (ρ·Mj) (κ·chunkAbs8 va vb j + ρ'·Mj) + σ·|s j + chunkSum8 va vb j| + α`, where the new
    `chunkAbs8 va vb j = Σ_{k ∈ chunk8 K j} |va k · vb k|` is the chunk's exact magnitude (not lifted).
    New constants `kappaChunk8 = 2` and `rhoLossChunk8 = 2^-12`. New theorems in the same file:
    `DotChunkLoss.chunkAcc` (the first arm: `DotChunkLoss P cfg ρ σ α κ ρ' → DotChunkAcc P cfg ρ σ α`,
    on the same partial values), `DotChunkLoss.mono`, and `DotF8ChunkLoss.chunk`
    (`DotF8ChunkLoss P a b → DotF8Chunk P a b`, so each loss fact gives the chunk fact of its pair).
    The module docstring explains the compounding and the second bound.
  - **New lemmas** (`lean/TVA/Lemmas/Dot8Chunk.lean`, section "Each chunk's error, the smaller of
    two bounds"): `Dot8Chunk.chunkAbs8_nonneg`, `abs_chunkSum8_le`, `sum_chunkAbs8`; `errM` (the
    composed bound with the minimum: `errM 0 = 0`,
    `errM (j+1) = (1+σ)·errM j + min (ρ·(X j + errM j)) (κ·a j + ρ'·(X j + errM j)) + σ·|psum v c (j+1)| + α`,
    `X j = max |psum v c j| (m j)`), `errM_nonneg`; `chain_errM` (a chain whose steps satisfy
    `DotChunkLoss`'s bound stays within `errM` of the exact partial sums); `errM_le_errB` (never worse
    than the round-24 `errB`); `errL` (the second arm composed alone:
    `errL (j+1) = (1+ρ'+σ)·errL j + κ·a j + ρ'·X j + σ·|psum v c (j+1)| + α`), `errM_le_errL`, and
    the closed form `errL_le` (`errL j ≤ (1+ρ'+σ)^j·Σ_{i<j} (κ·a i + ρ'·X i + σ·|psum (i+1)| + α)`);
    `DotChunkLoss.err` and `DotF8ChunkLoss.err` (the result within
    `errM … vc (chunkSum8 va vb) (chunkAbs8 va vb) m (chunks8 K)` of `vc + Σ va·vb`, for any bounds
    `m j` on the chunks' lifted products); `kappaChunk8_nonneg`, `rhoLossChunk8_nonneg`;
    `DotChunkLoss.of_chunkAcc_zero` (`DotChunkAcc` with `ρ = 0` gives `DotChunkLoss` for any
    `ρ, κ, ρ' ≥ 0`); satisfiability: `Platform.ref.dotChunkLoss`, `Platform.sm90ref.dotF8ChunkLoss`,
    `.dotE4m3ChunkLoss`, `.dotE5m2ChunkLoss`, `.dotE4m3E5m2ChunkLoss`, `.dotE5m2E4m3ChunkLoss`. The
    module docstring lists them. `DotChunkAcc`, `DotF8Chunk`, the four chunk facts, `errB`,
    `chain_err`, `errB_le`, `DotChunkAcc.err` and `DotF8Chunk.err` are unchanged.
  - `docs/rules/num.md`: the weakening lists and the definitions table (a `DotChunkLoss` row); the
    reviewer's table (the fp8 row also names the four loss facts); "Facts the library states" (four
    rows, `DotE5m2ChunkLoss` written out and the others "as `DotE5m2ChunkLoss`", and the date line);
    in "`tl.dot` of fp8, chunk by chunk": the opening, the corrected compounding sentence (L127), a new
    paragraph "Each chunk's error, the smaller of two bounds" (statement and use), and a new paragraph
    "The loss facts' constants" (derivation and evidence); "Satisfiability".
  - `docs/LIBRARY.md` (the `Rules.Num.Dot8Chunk` and `Dot8Chunk` rows), `docs/PROVER_GUIDE.md` (the
    `tl.dot` cell), `evidence/num-dot-chunk/README.md` (a round-26 note in conclusion 2, L127).
  - New evidence folder `evidence/num-dot-chunk-round26/` (`README.md`, `run.py`, `out.txt`).
- **Why.** lr-24b reopened L124: under `DotChunkAcc` each step charges `2^-7·Mj` with `Mj ≥ |s j|`, so a
  fed-back accumulator still grows by about `1 + 2^-7 + 2^-12` per chunk, while by the facts' own
  window model a truncated product loses at most `min(|p|, u)` (lr-24b's `work/fp8_alt.out`: every
  chunk within `B_alt`, largest `0.82`).

  **The route: new facts, not a restatement.** I added the sharper bound as facts of their own, as
  the coordinator preferred, rather than restate `DotChunkAcc`:
  - the minimum must bound the same partial values as the present bound, so each new fact states
    both arms of one chain (`DotChunkLoss`), not the second arm alone; its first arm is exactly
    `DotChunkAcc`'s bound, so a loss fact gives the chunk fact of its pair (`DotF8ChunkLoss.chunk`);
  - `matmul_e5m2` and `attn_fwd_e5m2` cite `DotE5m2Chunk` and use `errB`, `chain_err`,
    `DotF8Chunk.err` and `chunkLift8`, all unchanged, so every kernel builds unchanged; a kernel that
    wants the sharper bound cites `DotE5m2ChunkLoss` and takes `DotF8ChunkLoss.err` (or `chain_errM`).

  The second arm shares `σ = 2^-12` and `α = 2^-126` with the first, so the step bound reads
  `min (2^-7·Mj) (2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126`: one minimum, on the window terms.
  A dominating accumulator is then charged `2^-12 + 2^-12 = 2^-11` of itself per chunk, and `errL`
  grows by `1 + 2^-11` per chunk (the coordinator's example had `2^-11·Mj` and so `1 + 3·2^-12`;
  the derivation supports `2^-12`, with the same factor of about 2 as the other constants).

  **The derivation of the new constants** (checkable against the bit-level `wgmma` model of
  `evidence/num-dot-chunk`, equal to the device bit for bit there and on every `wgmma` element of
  the new evidence). Per chunk, the terms are the accumulator `s` and 32 exact products; `E` is the
  largest exponent field and `u = 2^(E−13)`; every term is truncated toward zero to a multiple of
  `u`, the terms are summed exactly to `S'`, and `S'` is truncated toward zero to 14 significant bits.
  - A product loses at most `min(|p|, u)`: all of it when `|p| < u`, less than `u ≤ |p|` otherwise.
    The accumulator loses less than `u`. So `|S' − (s + c_j)| ≤ A_j + u`, `A_j = Σ_chunk |a·b|`.
  - The cut loses less than `2^-13·|S'|` (the kept grid is `2^(⌊log₂|S'|⌋−13)`), and
    `|S'| ≤ |s + c_j| + A_j + u`. So the chunk's error is below
    `(1 + 2^-13)·(A_j + u) + 2^-13·|s + c_j|`.
  - `u ≤ 2^-13·Mj + 2^-139` (round 24's step: `2^e(x) ≤ liftMag x`, so a product's
    `2^(e(a)+e(b)) ≤ Mj`; the accumulator's `2^max(⌊log₂|s|⌋, −126) ≤ Mj + 2^-126`).
  - So the error is below
    `(1 + 2^-13)·A_j + (1 + 2^-13)·2^-13·Mj + 2^-13·|s + c_j| + (1 + 2^-13)·2^-139`.
    `kappaChunk8 = 2 = 1.9998·(1 + 2^-13)`; `rhoLossChunk8 = 2^-12 = 1.9998·(1 + 2^-13)·2^-13`;
    `sigmaChunk8 = 2^-12 = 2·2^-13`; `alphaChunk8 = 2^-126`, about `8191·(1 + 2^-13)·2^-139`. (I checked
    each inequality with `norm_num` against the Lean constants.)
  - The worst case: an accumulator setting the window with every bit below `u` set and magnitude just
    above `2^E` (so `Mj = |s|` is smallest against `u`), and 32 products of its sign just below `u`,
    lost whole: an error of about `33u` against `2·32u + 2u + 2u = 68u`, a ratio of about `0.485`
    (measured `0.4843`, kind `lossacc`). Where a product sets the window, `|p| ≥ 2^(E−3)` (a
    subnormal operand is at least `2^-3` of the smallest normal), so `2·A_j` alone is far above `u`.
  - `mma.sync` (two `float16` instructions per chunk, 25 bits below each one's largest field, terms
    and sum truncated toward zero; `evidence/num-dot-sub16-round25`): each instruction loses at most
    the products' magnitudes, `2^-25` of its largest lifted term (at most `Mj` for the first and
    `Mj + A_j` for the second, whose accumulator is at most `|s| + A_j`), and `2^-23` of its
    truncated sum (at most `|s| + A_j`) plus `2^-149` in the cut to `float32`. In all, below
    `(1 + 2^-25 + 2^-22)·A_j + (2^-24 + 2^-22)·Mj + 2·2^-151 + 2·2^-149`, far inside the second arm.
  - The first arm, `2^-7·Mj`, is round 24's derivation, on the same partial values, so the minimum
    holds. The room is `DotChunkAcc`'s and suffices: no partial value exceeds `|acc| + Σ|a·b|`.
  - Satisfiable: `Platform.sm90ref` (one `float32` rounding of the exact result) meets `DotChunkAcc`
    with `ρ = 0`, whose step bound is below both arms (`Platform.ref.dotChunkLoss`).
- **Evidence.** [evidence/num-dot-chunk-round26](../../../../evidence/num-dot-chunk-round26/README.md)
  (Triton 3.8.0, H200 NVL, GPU 0, 2026-11-01; about 2 minutes), all four pairs, both paths.
  - **Part S**, single chunks (`K = 32`) checked on the device's own values (with one chunk the partial
    values are the accumulator and the result: no model), seven block shapes and operand forms
    (`b` transposed and `a` converted in registers, as tutorial 06), eighteen kinds of inputs
    (num-dot-chunk's thirteen and five where the accumulator sets the window: `accone`, `accdom`,
    `accsmall`, `lossacc`, `lossmix`), two draws each: 4460544 elements, none outside.
  - **Part B**, every `K` that compiles on four blocks: on `wgmma` the model equals the device on all
    6193152 elements and each chunk of its chain is within the step bound; every element of both paths
    is within the interval the facts imply (exact, since the step is increasing in `s`).
  - **Part C**, tutorial 06's chains (`acc = tl.dot(p.to(fp8), v, acc)`, softmax, `accdom` and
    `subwin` inputs, `T` up to 256, with and without a rescale): 78913536 per-dot elements and 958464
    whole-chain elements, none outside, every `wgmma` dot equal to the model.
  - **The largest ratio** to the new bound is `0.4872`: a whole dot of kind `lossacc` at `K = 1024`
    (`float8e4nv` left operand, `64 × 64`, 4 warps). Per chunk it is `0.4843` (Part S, `lossacc`,
    `128 × 64` with `a` converted in registers: accumulator `262175.96875 = 2^18 + 2^5 − 2^-5`, unit
    `u = 2^5`, 32 products `30` lost whole, exact `263135.96875`, device `262144`), and `0.4833` with a
    `float8e5` left operand. `accone` (lr-24b's case): `0.4558` (round-24 bound `0.2348`).
    `mma.sync`: at most `0.0005`. No result exceeds `|acc| + Σ|a·b|` in magnitude.
  - **The sharpening**: the new composed bound is `0.29` to `0.78` of the round-24 one on the softmax
    chains at `T = 256`, `0.35` to `1` on the `accdom` chains at `T = 64`, and down to `0.06` on single
    dots where the accumulator dominates.
- **Constants whose meaning changed.**
  - **Meaning changed:** none. `DotChunkAcc`, `DotF8Chunk`, `DotE4m3Chunk`, `DotE5m2Chunk`,
    `DotE4m3E5m2Chunk`, `DotE5m2E4m3Chunk`, `rhoChunk8`, `sigmaChunk8`, `alphaChunk8`, `errB`,
    `chain_err`, `errB_le`, `DotChunkAcc.err`, `DotF8Chunk.err` and every other existing constant keep
    their statements.
  - **New:** `TVA.Rules.Num.chunkAbs8`, `DotChunkLoss`, `kappaChunk8`, `rhoLossChunk8`,
    `DotF8ChunkLoss`, `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss`,
    `DotE5m2E4m3ChunkLoss`, `DotChunkLoss.chunkAcc`, `DotChunkLoss.mono`, `DotF8ChunkLoss.chunk`;
    `TVA.Dot8Chunk.chunkAbs8_nonneg`, `abs_chunkSum8_le`, `sum_chunkAbs8`, `errM`, `errM_nonneg`,
    `chain_errM`, `errM_le_errB`, `errL`, `errM_le_errL`, `errL_le`, `DotChunkLoss.err`,
    `DotF8ChunkLoss.err`, `kappaChunk8_nonneg`, `rhoLossChunk8_nonneg`, `DotChunkLoss.of_chunkAcc_zero`;
    `TVA.Platform.ref.dotChunkLoss`, `TVA.Platform.sm90ref.dotF8ChunkLoss`, `.dotE4m3ChunkLoss`,
    `.dotE5m2ChunkLoss`, `.dotE4m3E5m2ChunkLoss`, `.dotE5m2E4m3ChunkLoss`.
  - **Docstrings changed, statements unchanged:** the module docstrings of `Rules/Num/Dot8Chunk.lean`
    and `Lemmas/Dot8Chunk.lean`.
  - **Files of the digests that changed:** `docs/rules/num.md`, `lean/TVA/Rules/Num/Dot8Chunk.lean`,
    `lean/TVA/Lemmas/Dot8Chunk.lean`. Every kernel item whose cone reaches `TVA.Rules.Num` (each
    num-fact item of every kernel) or either Lean file (the fp8 kernels' items) re-opens; no statement
    a kernel cites changed.
  - **Kernel modules that stop building:** none (all nine kernel libraries build).
  - **Left open:** the new facts do not state the device's no-growth (`|s j| ≤ |acc| + Σ_{i<j} A_i`,
    which the evidence checks and which would remove the remaining `1 + 2^-11` per chunk, as lr-24a
    suggested). A kernel with a long fed-back chain may ask for it.

## L127: the fed-back accumulator's compounding is the facts' form, not the device's behaviour

- **What changed.** `docs/rules/num.md`, "`tl.dot` of fp8, chunk by chunk", "How a kernel uses
  them": the sentence that an accumulator fed back once per block compounds by up to about
  `1 + 2^-7 + 2^-12` per chunk "as the device's window allows (below), not by slack in the fact" is
  replaced. It now says that a bound composed from these steps grows so; that the compounding comes
  from the facts' form (each step relative to `Mj ≥ |s j|`) and margin (`2^-7` is about twice the
  window's worst chunk), not from the device; that the device truncates every term and the sum toward
  zero, so its partial values never exceed `|acc| + Σ_{i<j} Σ_chunk |a·b|` in magnitude, and a
  truncated product loses at most its own magnitude, so its error grows at most linearly in the
  chunks; and that the loss facts (L124) state a second bound from this.
  `evidence/num-dot-chunk/README.md` had no such sentence; conclusion 2 gains a round-26 note saying
  the same and pointing to the loss facts and `evidence/num-dot-chunk-round26`. The module docstring
  of `lean/TVA/Rules/Num/Dot8Chunk.lean` says it too (L124). `ISSUES.md`'s round-24 entry is the
  coordinator's to correct (coordinator's note).
- **Why.** lr-24a and lr-24b: by the page's own model a truncated term loses at most `min(|term|, u)`,
  and truncation toward zero never increases a magnitude, so the device's error grows at most
  linearly; the facts' bound compounds only because each step is relative to the partial value, and
  `2^-7` keeps a factor `1.94` over the window's `33·(1 + 2^-13)·2^-13`.
- **Evidence.** `evidence/num-dot-chunk-round26`: no result of about `9.1·10^7` exceeds
  `|acc| + Σ|a·b|` in magnitude (`magnitude`, every row), and the device's chunks stay within the
  second arm (L124).
- **Constants whose meaning changed.** None (wording). `docs/rules/num.md` changed (as for L124).

## L128: `input_precision` on fp8 operands: cite what covers the mixed pairs

- **What changed.** `evidence/num-dot-chunk/README.md`, "What Triton runs": num-dot Part 3 is now
  cited for what it covered (the results bit for bit for `float8e4nv × float8e4nv` and
  `float8e5 × float8e5` only, random inputs, both paths), and the four pairs, the two mixed ones
  included, are covered by the new Part P of `evidence/num-dot-chunk-round26`.
- **Why.** lr-24a: Part 3 ran only the two unmixed pairs, so it said nothing of `DotE4m3E5m2Chunk` and
  `DotE5m2E4m3Chunk`, which state every `input_precision`. The coordinator asked me to run it myself
  if I cite it.
- **Evidence.** `evidence/num-dot-chunk-round26`, Part P: for each of the four pairs, on `64 × 64`
  (4 warps) and `128 × 64` (8 warps) on `wgmma` and `16 × 16` (1 warp) and `32 × 32` (2 warps) on
  `mma.sync`, `K = 64`, every `input_precision` (`tf32`, `tf32x3`, `ieee`, `bf16x3`, `bf16x6`) gives
  the PTX of the default (`None`), the default's PTX equals that of a kernel that omits the argument
  (both without debug line information), and every result equals the default's bit for bit on
  `normal`, `sub`, `subwin` and `bits` inputs: 80 of 80 rows. So the facts' `∀ p` holds for every
  pair, and so do the new loss facts'.
- **Constants whose meaning changed.** None.

## L129: the scaled-dot evidence rests on the 25-bit window, not num-dot's per-chunk derivation

- **What changed.** `evidence/num-dot-scaled/README.md`, conclusion 2: the facts take `DotBF16`'s
  constants; on `scaledDomBF16` every scaled value is `0` or a normal `bfloat16` value, so the emulated
  dot has no subnormal operand and its lifted magnitudes are `|a·b|`; the 25-bit window of
  `evidence/num-dot-sub16-round25` (its bit-level model and its derivation, which covers every `K`)
  bounds that dot by `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K`, and its Part G3 checks these facts on
  window inputs (at most `0.0057` of the bound). A parenthesis records what the conclusion cited
  before.
- **Why.** lr-25a: the conclusion extended the facts to every `K` by num-dot's per-chunk derivation
  ("every term down to `2^-23` of the largest"), which round 25 found holds only for normal operands.
  The conclusion still holds, because on `scaledDomBF16` the operands are normal, as
  `docs/rules/num.md` ("`tl.dot_scaled` on sm_90") already says.
- **Evidence.** `evidence/num-dot-sub16-round25` (Part G3 and its derivation), unchanged.
- **Constants whose meaning changed.** None (an evidence README, in no digest).

## L130: the opening of "`tl.dot` (the chunk model)"

- **What changed.** `docs/rules/num.md`, the opening of "`tl.dot` (the chunk model)": the parenthesis
  now reads "evidence `evidence/num-dot`, whose "Part 1" … "Part 4" are cited by name below, and, for
  the 16-bit facts, `evidence/num-dot-sub16-round25`; the fp8 facts are in the next section". The
  sentence on chunks now says that each instruction adds 16 products for 16-bit operands on both
  paths, and, for fp8 operands, 32 on `wgmma` and 16 on `mma.sync`, where Triton converts fp8 to
  `float16` and runs two instructions per chunk of 32 whose `k` interleave.
- **Why.** lr-25a and lr-25b: the parenthesis did not parse, and "32 for fp8" holds on `wgmma` only
  (evidence/num-dot-chunk Part E: on `mma.sync` a chunk of 32 is two `k16` instructions).
- **Evidence.** None needed (wording; evidence/num-dot-chunk Parts A2 and E).
- **Constants whose meaning changed.** None (wording). `docs/rules/num.md` changed (as for L124).

## L131: `docs/LIBRARY.md` lists the kernels

- **What changed.** `docs/LIBRARY.md`, "Kernels": the "(none yet)" row is replaced by the nine
  kernels, each with its tutorial function and what it exercises, and a pointer to
  `kernels/README.md`, the index with each kernel's gate, review phase and certificate.
- **Why.** lr-25b: nine kernel folders exist (`kernels/README.md`).
- **Evidence.** None needed (`kernels/README.md`, `lean/lakefile.toml`).
- **Constants whose meaning changed.** None (`docs/LIBRARY.md` is in no digest).

## L132: num-dot-sub16's conclusion 1 is about the facts as stated before round 25

- **What changed.** `evidence/num-dot-sub16/README.md` (the coordinator's folder; the coordinator
  allowed this edit), conclusion 1: "`DotF16` and `DotBF16`, as stated before round 25 (relative to
  `|acc| + Σ|a·b|`), do not hold …", and a parenthesis: the bound there and in the table's
  `error / bound` is the facts' as stated before round 25; since round 25 the names denote the facts
  restated relative to lifted magnitudes (`DotLiftAcc`), which hold on these inputs at `0.104` and
  `0.1025` of their bound (`evidence/num-dot-sub16-round25`, Part F). The scripts and outputs are
  unchanged.
- **Why.** lr-25b: since round 25 the names `DotF16` and `DotBF16` denote the restated facts, which
  hold on the same inputs.
- **Evidence.** `evidence/num-dot-sub16-round25`, Part F (`53.181` and `13.122` times the old bound,
  `0.1040` and `0.1025` of the new one), unchanged.
- **Constants whose meaning changed.** None (an evidence README, in no digest).

## L133: round 25's missing reach entry

- **What changed.** Nothing in the library; this entry completes round 25's list of constants whose
  meaning changed (round 25's notes are recorded and frozen). Round 25 gave `TVA.Rules.Num.minNormal8`
  the values `2^-14` at `.f16` and `2^-126` at `.bf16` (it was `0` there), so `TVA.Rules.Num.liftMag`
  changed at those dtypes, and with it every constant that applies `liftMag` or `minNormal8` at a
  dtype it does not fix:
  - `TVA.Rules.Num.DotChunkAcc` (generic in `cfg`: its `Mj` must bound `liftMag cfg.a … · liftMag
    cfg.b …`) and the theorems stated with it, `DotChunkAcc.mono`, `TVA.Dot8Chunk.DotChunkAcc.err`,
    `DotChunkAcc.liftErr` and `TVA.Platform.ref.dotChunkAcc`;
  - the `liftMag_*` lemmas of `TVA.Dot8Chunk`, stated for every dtype: `liftMag_nonneg`,
    `abs_le_liftMag`, `liftMag_zero`, `liftMag_of_le`, `liftMag_le`, `minNormal8_nonneg`,
    `abs_mul_le_liftMag`;
  - `TVA.Dot8Chunk.chunkLift8`, `chunkLiftSum8` (generic in the dtypes) and `le_chunkLift8`,
    `chunkLift8_nonneg`, `sum_chunkLiftSum8`.

  Each means at `.f16` and `.bf16` what it says with a subnormal operand counted at the smallest
  normal, instead of at `|q|`. At the fp8 dtypes `minNormal8` is unchanged, so the fp8 facts
  (`DotF8Chunk`, `DotE4m3Chunk`, …) and every use at those dtypes state the same as before. No fact
  applies `DotChunkAcc` at `.f16` or `.bf16`; the 16-bit kernels apply the `liftMag_*` lemmas at `.f16`
  in their proofs (for example `attn_bwd_f16`'s `AttnBwd.Acc`), which round 25's change to `DotF16`
  required in any case, and the digests re-open the items whose cones reach these files.
- **Why.** lr-25b: round 25's list named `DotF16`, `DotBF16`, `minNormal8`, `liftMag` and `LiftAcc`,
  but not the generic `DotChunkAcc` and the `liftMag_*` lemmas, which change meaning at `.f16` and
  `.bf16` through `liftMag`.
- **Evidence.** None needed (the definitions; round 25's notes and lr-25b's report).
- **Constants whose meaning changed.** In round 26, none; the entry above is round 25's.

