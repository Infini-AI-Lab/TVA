# Library issues

The library's open questions and defects, from kernel review (`LIBRARY_REPORT`s and `REFERRED`
findings, collected in [REPORTS.md](REPORTS.md)), from library review, and from the author.
The process is in [../LIBRARY_REVIEW.md](../LIBRARY_REVIEW.md); kernel findings are cited as
`<kernel>#F<n>` (each kernel's `REVIEW.md`).

States: `OPEN` (to be fixed in the round named), `ACCEPTED` (fixed, and accepted by library
review), `FIXED` (fixed in a round that no library review followed, by the user's decision under
L36: rounds 21 to 23 and 26 to 28; the fixes of rounds 24 and 25 were reviewed later; and round 38, the
final fix, by the user's decision of 2026-11-08: "let's run up to round37, and file a final fix without further
review"), `CLOSED` (settled by the coordinator, as its section says).

The kernels of the first round (vec-add, grouped GEMM, flash attention) were cleared on
2026-10-27 at the user's request, to restart from scratch. The issues they raised stand on their
own; the cleared kernels' records are in the git history (commit `089d4c5`).

| Issue | State | Round | Source | Summary |
|---|---|---|---|---|
| [L1](#l1) | ACCEPTED | 8 | cleared round 1: grouped GEMM F1, F2 | A hint inside a loop body on a loop-carried name is not a no-op |
| [L2](#l2) | ACCEPTED | 8 | cleared round 1: grouped GEMM F2 | What a hint on a rank ≥ 2 tensor promises, in terms a kernel can check |
| [L3](#l3) | ACCEPTED | 2 | cleared round 1: vec-add F1, grouped GEMM F3, flash attention F4 | A launch rejected before it runs is outside the claim (user, 2026-10-27) |
| [L4](#l4) | ACCEPTED | 9 | cleared round 1: flash attention F1, F2, F7, grouped GEMM F6 | Device facts with constants, owned by the library, on TVA's own evidence |
| [L5](#l5) | ACCEPTED | 11 | cleared round 1: flash attention F5 | A descriptor built inside the kernel writes launch scratch memory |
| [L6](#l6) | ACCEPTED | 2 | cleared round 1: flash attention F8 | A drop reason for `tl.static_assert` |
| [L7](#l7) | ACCEPTED | 16 | user, 2026-10-27 | TVA cites no other project's results, lemmas or measurements, and the tool enforces it |
| [L8](#l8) | ACCEPTED | 2 | cleared vec-add prover, round 1 | `Block.Fits` is stricter than `int32` indexing needs |
| [L9](#l9) | ACCEPTED | 2 | library review round 1 (lr-1) | `Launch.Valid` admits grids the launcher silently skips (program count ≥ 2^31) |
| [L10](#l10) | ACCEPTED | 2 | library review round 1 (lr-1) | `no-op:print` drops arguments that may have effects |
| [L11](#l11) | ACCEPTED | 2 | library review round 1 (lr-1) | `rules/num.md` asks kernel reviewers to check device truth |
| [L12](#l12) | ACCEPTED | 4 | library author, round 2 | No digest covers the transcription rules, so a change to them re-opens no kernel `map` |
| [L13](#l13) | ACCEPTED | 3 | library review round 2 (lr-2a) | soundness: a kernel's Lean can add `main` without the kernel checking it |
| [L14](#l14) | ACCEPTED | 3 | library review round 2 (lr-2a) | soundness: "pure" is unsound for aggregate operands (overloaded operators run code) |
| [L15](#l15) | ACCEPTED | 3 | library review round 2 (lr-2b) | soundness: one program is not sequential for global memory; user's decision: no conflicting accesses within a program |
| [L16](#l16) | ACCEPTED | 3 | library review round 2 (lr-2b) | gap: `tl.max`/`tl.min` widen small integers to `int32` |
| [L17](#l17) | ACCEPTED | 4 | library review round 2 (lr-2b) | gap: no `tl.permute` (rank ≥ 3), PDL (`gdc_*`, `launch_pdl`) or `tl.argmax` |
| [L18](#l18) | ACCEPTED | 4 | library review round 2 (lr-2b) | gap: `TRITON_F32_DEFAULT` changes `tl.dot`'s default precision |
| [L19](#l19) | ACCEPTED | 4 | library review round 3 (lr-3a) | soundness: launch keywords outside `Launch` change the compiled code |
| [L20](#l20) | ACCEPTED | 6 | library review round 3 (lr-3a) | soundness: model text can mean another operation (a kernel's own `tl.add`, `local notation`) |
| [L21](#l21) | ACCEPTED | 8 | library review round 3 (lr-3a, lr-3b) | gap: autotuning and hooks run the kernel and host code around the claimed launch |
| [L22](#l22) | ACCEPTED | 6 | library review round 3 (lr-3a) | wording: descriptor and TMA text written before L15 |
| [L23](#l23) | ACCEPTED | 4 | library review round 3 (lr-3b) | soundness: `[literal]` is wrong for fp64 and bf16 constants |
| [L24](#l24) | ACCEPTED | 4 | library review round 3 (lr-3b) | soundness: `[itof-narrow]` is wrong under `fp_downcast_rounding="rtz"` |
| [L25](#l25) | ACCEPTED | 6 | library review round 3 (lr-3b) | soundness: `[config]` drops branch conditions that may run operations |
| [L26](#l26) | ACCEPTED | 4 | library review round 3 (lr-3b) | soundness: `[sum-bcast-mul]` names one spelling of what the compiler rewrites |
| [L27](#l27) | ACCEPTED | 4 | library review round 3 (lr-3b) | soundness: SEMANTICS.md, which `[call]` relies on, is outside every digest |
| [L28](#l28) | ACCEPTED | 8 | library review round 3 (lr-3b) | gap: no rule for a runtime `if`/`else` or an early `return` |
| [L29](#l29) | ACCEPTED | 8 | library review round 3 (lr-3b) | gap: no rule for the types of scalar arguments |
| [L30](#l30) | ACCEPTED | 8 | library review round 3 (lr-3b) | gap: block pointers are aggregates, which exclusions.md excludes |
| [L31](#l31) | ACCEPTED | 7 | library review round 3 (lr-3b) | wording: stale hint text, `[no-op:debug_barrier]`'s reason, `[dot-add]`'s "constant zero" |
| [L32](#l32) | ACCEPTED | 6 | library review round 4 (lr-4a, lr-4b) | soundness: text kernels rely on is outside the digests (module docstrings, unused field docstrings, the citable facts of rules/num.md) |
| [L33](#l33) | ACCEPTED | 9 | library review round 4 (lr-4b) | gap: no error bound for reductions with fusable additions (`tl.sumc`, `tl.cumsumc`, tree `sum_bcast_mul`) |
| [L34](#l34) | ACCEPTED | 14 | library review round 4 (lr-4b) | gap: the shared-atomics kit has no masked tensor atomic and no accumulation beyond `int32` addition |
| [L35](#l35) | ACCEPTED | 6 | library review round 4 (lr-4a, lr-4b) | wording: seven items (DivFull's "2 ulp", `tl.maximum` in Scalar's docstring, `tl.full` wrap, ItofRel, `sum_bcast_mul`, Num's "reviewer judges", shell expansions in §2) |
| [L36](#l36) | ACCEPTED | 10 | user, 2026-10-28 | Leverage judgment: mechanical checks only where complete; transcription as guidance; advisory scans |
| [L37](#l37) | ACCEPTED | 11 | user, 2026-10-28 | Cover every pinned tutorial kernel, except tutorial 05's `_layer_norm_bwd_dx_fused` and tutorial 11 |
| [L38](#l38) | ACCEPTED | 7 | library review round 6 (lr-6a) | soundness: a kernel's `kernel.json` namespace is pasted unchecked into the claim reader |
| [L39](#l39) | ACCEPTED | 7 | library review round 6 (lr-6a, lr-6b) | gap: the map coverage check skips lines with `"""` and the lines between two of them |
| [L40](#l40) | ACCEPTED | 10 | library review round 6 (lr-6a) | gap: library modules are not replayed with `leanchecker`; any `leanOptions` key is accepted |
| [L41](#l41) | ACCEPTED | 7 | library review round 6 (lr-6a) | gap: `.lake` folders outside `lean/` ship, but the text scan skips them |
| [L42](#l42) | ACCEPTED | 7 | library review round 6 (lr-6a) | gap: a library change leaves kernel gates stale and certificates GRANTED |
| [L43](#l43) | ACCEPTED | 7 | library review round 6 (lr-6a) | gap: a `TVA.Rules` module in a subfolder gets no rule page, and the audit is silent |
| [L44](#l44) | ACCEPTED | 8 | library review round 6 (lr-6a, lr-6b, coverage) | gap: `tl.store` cache modifiers have no stated meaning |
| [L45](#l45) | ACCEPTED | 8 | library review round 6 (lr-6b, coverage) | gap: operator and promotion semantics not stated (integer `/`, `and`/`or`/`not`, mixed dtypes, `tl.where` on int32, builtin `min`, integer `tl.minimum`) |
| [L46](#l46) | ACCEPTED | 7 | library review round 6 (lr-6a, lr-6b, coverage) | wording: eight items (release check's reach in §2, evaluation order of keywords, conditional expressions, Desc and L5, `launch_metadata=`) |
| [L47](#l47) | ACCEPTED | 8 | library review round 6 (coverage) | gap: launching a precompiled `CompiledKernel` skips the JIT's argument checks (tutorial 02) |
| [L48](#l48) | ACCEPTED | 10 | library review round 7 (lr-7a) | soundness: item digests rest on 64-bit non-cryptographic hashes; two different kernel values share one digest |
| [L49](#l49) | ACCEPTED | 8 | library review round 7 (lr-7b) | gap: `Compiler.Faithful`'s "runs alone" forbids every other writer of global memory, more than claims need |
| [L50](#l50) | ACCEPTED | 13 | library review round 7 (lr-7a) | gap: the map's coverage rests on `source.ranges`, which nothing checks |
| [L51](#l51) | ACCEPTED | 10 | library review round 7 (lr-7a, lr-7b) | gap: claim.md numbers lines with `splitlines()`, so a form feed or U+2028 shifts the quoted source and declarations |
| [L52](#l52) | ACCEPTED | 13 | library review round 7 (lr-7a) | gap: the release check does not compare certificates and review records with the ledger |
| [L53](#l53) | ACCEPTED | 10 | library review round 7 (lr-7a, lr-7b) | gap: gate staleness uses file times, leaves out the source and the write-up, and ignores changes to the gate |
| [L54](#l54) | ACCEPTED | 10 | library review round 7 (lr-7b) | gap: `lean/lean-toolchain` is unchecked |
| [L55](#l55) | ACCEPTED | 10 | library review round 7 (lr-7a) | gap: a `TVA.Rules.*` constant declared outside `lean/TVA/Rules` gets no rule page |
| [L56](#l56) | ACCEPTED | 10 | library review round 7 (lr-7a, lr-7b) | wording: the toolchain's trust, the release check mid-round, `[kernel]` assumptions, skipped grids, DESIGN.md |
| [L57](#l57) | ACCEPTED | 11 | library review round 8 (lr-8a) | soundness: PROVER_GUIDE still builds a comparison's constant in the operation's dtype (the rule L45 corrected) |
| [L58](#l58) | ACCEPTED | 17 | library review round 8 (lr-8a, lr-8b) | gap: two hint readings are stronger than Triton relies on (`multiple_of` alone on an arange; `max_contiguous` with `multiple_of`) |
| [L59](#l59) | ACCEPTED | 11 | library review round 8 (lr-8b) | soundness: a name assigned a Python number holds a scalar tensor of its own dtype, which SEMANTICS never says |
| [L60](#l60) | ACCEPTED | 11 | library review round 8 (lr-8b) | soundness: a hint on one result of an operation with several results applies to every result |
| [L61](#l61) | ACCEPTED | 15 | library review round 8 (lr-8a, lr-8b) | wording: index arithmetic, one-branch names, builtin `min`/`max`, hooks and arguments, masked lanes, PROVER_GUIDE's hints |
| [L62](#l62) | ACCEPTED | 16 | library review round 9 (lr-9a) | soundness: a `[rule]` field can state a device fact with the kernel's own constants (`MathAcc` with literals) |
| [L63](#l63) | ACCEPTED | 13 | library review round 10 (lr-10a) | soundness: a `private` or `_root_` kernel constant passes `kinds` in a `[rule]` |
| [L64](#l64) | ACCEPTED | 20 | library review round 9 (lr-9a) | gap: fp8 conversion beyond `maxVal` (saturation) is not stated; four tutorials need it |
| [L65](#l65) | ACCEPTED | 12 | library review round 9 (lr-9a); coordinator | independence: `lean/TVA/Math` and parts of three lemma modules are another project's code; user: rewrite independently |
| [L66](#l66) | ACCEPTED | 14 | library review round 9 (lr-9a, lr-9b) | wording: evidence accounts of two met bounds, "classical bound", a chunk "within", "plain load", the not-stated list, two names for −∞ |
| [L67](#l67) | ACCEPTED | 14 | library review round 9 (lr-9b) | gap: `DivRn .binary64` has no device evidence of TVA's own |
| [L68](#l68) | ACCEPTED | 13 | library review round 10 (lr-10a) | soundness: an assumption item's digest omits the assumption's statement |
| [L69](#l69) | ACCEPTED | 13 | library review round 10 (lr-10a) | gap: gates don't go stale on `lean-toolchain`, `lakefile.toml` or an imported module with no constant in a cone |
| [L70](#l70) | ACCEPTED | 13 | library review rounds 9 and 10 (lr-9a, lr-10a) | gap: the audit skips `kernels/` folders without `kernel.json`, and cited paths that do not exist |
| [L71](#l71) | ACCEPTED | 11 | library review round 10 (lr-10a) | gap: the gate passes a claim whose conditions cover no instance (`Inst.ok := False`) |
| [L72](#l72) | ACCEPTED | 11 | library review round 10 (lr-10b) | gap: Faithful's scope 4 misses options given at `warmup` or `triton.compile` |
| [L73](#l73) | ACCEPTED | 11 | library review round 10 (lr-10b) | gap: no claim can say the launch leaves memory outside `init` alone |
| [L74](#l74) | ACCEPTED | 11 | library review round 10 (lr-10b) | wording: Faithful names no target, its grid sentence, scope 6's host descriptors |
| [L75](#l75) | ACCEPTED | 11 | library coverage round 10 | gap: SEMANTICS never says what a call of one of Python's builtins in a kernel gives (`float`, `isinstance`, `int`, `len`, …) |
| [L76](#l76) | ACCEPTED | 15 | library author, round 11 | soundness: a host `TensorDescriptor` built with `round_f32_to_tf32=True` makes its loads round `float32` to tf32, which `tl.desc_load` does not model |
| [L77](#l77) | ACCEPTED | 15 | library review round 11 (lr-11a, lr-11b) | gap: `Faithful` covers a `CompiledKernel` from `triton.compile` of an IR file, which replaces the source |
| [L78](#l78) | ACCEPTED | 15 | library review round 11 (lr-11b) | soundness: `@triton.heuristics` sets compile options, a route the library never names |
| [L79](#l79) | ACCEPTED | 18 | library review round 13 (lr-13a) | gap: module-level values a JIT function reads (a `tl.constexpr` global, a dtype) are in no class of the map |
| [L80](#l80) | ACCEPTED | 15 | library review round 13 (lr-13b, lr-13a) | gap: SEMANTICS gives no meaning to constexpr functions, plain Python calls, helper defaults or module globals |
| [L81](#l81) | ACCEPTED | 18 | library review round 13 (lr-13a) | gap: a kernel ledger shows nothing of a reviewer whose lines are removed while their folder stays |
| [L82](#l82) | ACCEPTED | 16 | library review round 13 (lr-13a, lr-13b) | wording: what ledger lines and a gate in the tree rest on; old events re-derived with today's defaults; capital path components |
| [L83](#l83) | ACCEPTED | 15 | library review round 13 (lr-13b) | soundness: TRANSCRIPTION's `tl.where(keep, x, 0.0)` row puts the constant in `x`'s dtype, false for an integer `x` |
| [L84](#l84) | ACCEPTED | 15 | library review round 13 (lr-13b) | soundness: the rule "unless the target is annotated `: tl.constexpr`" is false under either reading |
| [L85](#l85) | ACCEPTED | 15 | library review round 13 (lr-13b) | soundness: `for i in range(e)` with one argument evaluates `e` twice |
| [L86](#l86) | ACCEPTED | 15 | library review rounds 11 and 13 (lr-11a, lr-13b) | wording: several-result hints, index promotion by narrow unsigned types, and seven transcription points |
| [L87](#l87) | ACCEPTED | 15 | library review round 11 (lr-11a) | gap: `tl.cumsum`'s `dtype=` argument is not stated |
| [L88](#l88) | ACCEPTED | 17 | user, 2026-10-28 | TVA gives meaning to a supported subset of Triton's Python front end; every other form is excluded |
| [L89](#l89) | ACCEPTED | 17 | library review round 15 (lr-15b) | soundness: code after a `return` reached through a compile-time `if` or a `tl.static_range` copy is still compiled, and its hints apply |
| [L90](#l90) | ACCEPTED | 17 | library review round 15 (lr-15b) | gap: the subset admits `range(…)` with any arguments; the front end ignores keywords and a fourth argument |
| [L91](#l91) | ACCEPTED | 20 | library review round 16 (lr-16b) | soundness: a launch captured into a CUDA graph runs no program when called and runs at each replay; nothing says what a claim covers there |
| [L92](#l92) | ACCEPTED | 17 | library review round 15 (lr-15a, lr-15b) | wording: names a runtime `if` does not merge, `Launch.fusion` and autotune configs, annotated and default values, names after a loop, the hint bound, a garbled checklist item |
| [L93](#l93) | ACCEPTED | 17 | library review round 16 (lr-16b) | wording: the guides' lists of `Faithful`'s host conditions leave some out |
| [L94](#l94) | ACCEPTED | 18 | library review round 16 (lr-16a) | soundness: the claim reader finds `K` and `P` by position, so instance parameters before them shift the `[kernel]`/`[rule]` checks |
| [L95](#l95) | FIXED | 21 | library review round 16 (lr-16a, lr-16b) | gap: `stated_facts` reads every table under "Facts the library states", `###` subsections included, not only its table |
| [L96](#l96) | ACCEPTED | 18 | library review round 16 (lr-16a) | gap: the release check's `pages` step compares the regenerated pages with the live tree, not the checked copy |
| [L97](#l97) | ACCEPTED | 18 | library review round 16 (lr-16a, lr-16b) | wording: evidence/tool-round16's output was written before the protocol's text was fixed |
| [L98](#l98) | ACCEPTED | 20 | library review round 12 (lr-12a) | soundness: a Python NaN constant gets the canonical NaN, not the pattern Triton builds (sign and quiet payload) |
| [L99](#l99) | ACCEPTED | 19 | library review round 12 (lr-12a, lr-12b) | wording: the float layer's account of ±∞, NaN, overflow, `u`, `sqrt_rn`, and its evidence README and build script |
| [L100](#l100) | ACCEPTED | 19 | library review round 12 (lr-12b) | wording: names and docstrings in the proof kit that do not match their statements |
| [L101](#l101) | ACCEPTED | 20 | library review round 14 (lr-14a, lr-14b) | wording: three evidence READMEs, fp8's missing `u`/`eta` constants, `replay_err`'s operands |
| [L102](#l102) | FIXED | 21 | library review round 17 (lr-17a, lr-17b) | soundness: a helper whose returns differ, and a runtime conditional expression, make Python numbers scalar tensors of the front end's type |
| [L103](#l103) | FIXED | 21 | library review rounds 17 and 19 (lr-17a, lr-19a) | soundness: Triton's caches and helper names do not tell some different compile-time values apart (`1`/`1.0`/`True`, `1`/`"1"`, NaNs, strings) |
| [L104](#l104) | FIXED | 21 | library review rounds 17 and 18 (lr-17a, lr-18a, lr-18b) | soundness: `Faithful`'s scope never says that Triton and the kernel's module are used as installed and written |
| [L105](#l105) | ACCEPTED | 20 | library review round 18 (lr-18b) | soundness: `kernel.debug = True` and `JITFunction.preload` set compile options no scope point names |
| [L106](#l106) | ACCEPTED | 20 | library review round 18 (lr-18a) | soundness: an `Assumptions` declared as an `inductive` passes the gate; the claim reader panics, reads no field and exits 0 |
| [L107](#l107) | ACCEPTED | 20 | library review round 19 (lr-19b) | soundness: SEMANTICS states host-dependent values (CPython's float packing, x86-64's default NaN) without naming the host |
| [L108](#l108) | ACCEPTED | 20 | library review rounds 17, 18 and 19 (lr-17a, lr-17b, lr-18a, lr-18b, lr-19a, lr-19b) | wording: `tl.cdiv`, two general SEMANTICS sentences, round 18's evidence, `freeze` paths, `|` in tables, half an ulp, `FloatV.nan`, `Int.lean`, LIBRARY.md, float-layer conclusion 4 |
| [L109](#l109) | FIXED | 21 | library review round 20 (lr-20c, lr-20d) | soundness: a launch may run code compiled earlier, in its process or another, outside scopes 2, 3 and 10; Triton's keys do not record it |
| [L110](#l110) | FIXED | 21 | library review round 20 (lr-20b, lr-20d) | soundness: the host scope 7 names admits `python -O` (which removes Triton's `assert` rejections) and a changed floating-point environment |
| [L111](#l111) | FIXED | 21 | library review round 20 (lr-20e) | soundness: SEMANTICS says arithmetic in an fp8 dtype is always rejected, but MLIR folds `x * 1.0`, `x - 0.0` and arithmetic on fp8 constants, which then run |
| [L112](#l112) | FIXED | 21 | library review round 20 (lr-20c) | gap: the claim reader misreports two claims, both failing closed (a `PANIC` substring; a `private` field) |
| [L113](#l113) | FIXED | 21 | library review round 20 (lr-20f) | gap: the proof kit has no rule for `tl.convert` or `tl.sum0c`, and none named for `tl.store1` |
| [L114](#l114) | FIXED | 21 | library review round 20 (lr-20a, lr-20b, lr-20e, lr-20f) | wording: the fp8 constants' prose and PROVER_GUIDE's list, num.md's "exact" division facts, four lemma docstrings |
| [L115](#l115) | CLOSED | — | coordinator, 2026-10-29 (independence check) | independence: `TVA.Lemmas.Softmax` was written by an agent that had read another project's lemmas; removed |
| [L116](#l116) | FIXED | 22 | user, 2026-10-29 | gap: the autotuner is not verified; for an autotuned tutorial kernel we pick one config when we set up its folder |
| [L117](#l117) | FIXED | 23 | vec_add_f32#F3, vec_add_f32#F4 (r-vecadd, r-vecadd2) | soundness: condition 11's reuse clause does not ask for the same specialisation of the runtime arguments |
| [L118](#l118) | FIXED | 23 | vec_add_f32#F2 (r-vecadd) | wording: the `int32` docstring sits on `I8` in `Lang/Scalar.lean`; two pages still say only "without `-O`" |
| [L119](#l119) | CLOSED | — | matmul_e5m2#F3 (r-e5m2), matmul_e5m2#F4 (r-e5m2b) | accuracy: `relDot8`'s per-chunk constant `2^-7` is about twice the `33·2^-13` that num-dot's derivation gives |
| [L120](#l120) | CLOSED | — | grouped_matmul_tma_f16#F2 (r-gmm2) | gap: nothing states that a descriptor the kernel builds again in a loop is rewritten only after the accesses through its old contents (a TMA store may still be in flight) |
| [L121](#l121) | CLOSED | — | grouped_matmul_tma_f16#F4 (r-gmm) | gap: scope 5 does not ask the allocator to return the `alignment` the launcher asks for the descriptor scratch |
| [L122](#l122) | OPEN | — | attn_fwd_f16#F3 (r-fwd) | wording: "Performance-only arguments" says `num_stages` and `warp_specialize` are given by a model, but `Launch` and `tl.forRange` have no place for them and "Loops" says they change no outcome |
| [L123](#l123) | OPEN | — | attn_fwd_f16#F5 (r-fwd2) | wording: the fusion rule does not say whether a `tl.maximum` or `tl.where` result that carries a product's value is a product |
| [L124](#l124) | FIXED | 26 | attn_fwd_e5m2#F2 (r-e5fwd2) | sharpness: `DotE5m2`'s `relDot8` charges each fp8 `tl.dot` `2^-7` per chunk of `\|acc\|` too, so an accumulator fed back once per block compounds as `(1 + relDot8 64)^T` |
| [L125](#l125) | ACCEPTED | 24 | library author, round 24 (evidence/num-dot-chunk Part F) | soundness: `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3` fail on `wgmma` when a product with a subnormal operand sets the window (up to 3.65 times `relDot8`'s bound); `matmul_e5m2` and `attn_fwd_e5m2` cite `DotE5m2` |
| [L126](#l126) | ACCEPTED | 25 | grouped_matmul_tma_f16#F5, #F6; attn_fwd_f16#F6; attn_bwd_f16#F5, #F6; matmul_f16 (r-matmul2) | soundness: `DotF16` and `DotBF16` fail when a product with a subnormal operand sets the window (up to 53 and 13 times their bound); `matmul_f16`, `attn_fwd_f16`, `attn_bwd_f16` and `grouped_matmul_tma_f16` cite `DotF16` |
| [L127](#l127) | FIXED | 26 | round 24 review (lr-24a), with L124 (lr-24b) | wording: `docs/rules/num.md` ("How a kernel uses them") and L124 call the fed-back accumulator's compounding the device's behaviour, "not slack in the fact"; the window's own model loses at most each truncated term, so the device's error grows at most linearly |
| [L128](#l128) | FIXED | 26 | round 24 review (lr-24a) | wording: `evidence/num-dot-chunk/README.md` cites num-dot Part 3 for `input_precision` on fp8, but Part 3 ran only the two unmixed pairs |
| [L129](#l129) | FIXED | 26 | round 25 review (lr-25a) | wording: `evidence/num-dot-scaled/README.md` conclusion 2 extends the scaled-dot facts by num-dot's per-chunk derivation, which round 25 found holds only for normal operands |
| [L130](#l130) | FIXED | 26 | round 25 review (lr-25a, lr-25b) | wording: the opening of `docs/rules/num.md`'s "`tl.dot` (the chunk model)" does not parse, and its "32 products per instruction for fp8" holds on `wgmma` only |
| [L131](#l131) | FIXED | 26 | round 25 review (lr-25b) | wording: `docs/LIBRARY.md`'s "Kernels" section still says "(none yet)" |
| [L132](#l132) | FIXED | 26 | round 25 review (lr-25b) | wording: `evidence/num-dot-sub16/README.md` conclusion 1 says `DotF16` and `DotBF16` do not hold, without "as stated before round 25" |
| [L133](#l133) | FIXED | 26 | round 25 review (lr-25b) | wording: round 25's list of constants whose meaning changed leaves out the generic `DotChunkAcc` and the `liftMag_*` lemmas, which change meaning at `float16` and `bfloat16` (no kernel uses them there) |
| [L134](#l134) | OPEN | — | library author, round 26 | sharpness: the fp8 chunk facts do not state that a partial value never grows past `|acc| + Σ|a·b|`; stating it (with evidence) would remove the remaining `1 + 2^-11` per chunk of a fed-back accumulator |
| [L135](#l135) | FIXED | 27 | user, 2026-11-02 | tool: the sandbox's files (`sandbox/`, agents proving and reviewing a kernel in containers) name the containers' own paths, which the text scan read as paths out of TVA, so no gate could pass with the folder in the tree |
| [L136](#l136) | FIXED | 28 | user, 2026-11-04 (sandbox pilot, paged_attention_f16) | protocol: a write-up that states a claim other than the Lean's (a constant, tolerance or condition its definition does not have) could only be filed as `DOC`, which does not block, so a kernel was certified with a README stating a looser tolerance and a wider covered range than its Lean proves |
| [L137](#l137) | FIXED | 29 | sandbox test-set run, 2026-11-05 (`gdn_gating_bf16`, a REFERRED `UNSUPPORTED` finding) | gap: SEMANTICS.md names `tl.sigmoid` among the `tl.*` functions Triton writes in Triton but states no body, so a model of it cannot be judged against the page |
| [L138](#l138) | FIXED | 30 | sandbox test-set run, 2026-11-05 (`l2norm_fwd_bf16`: every submission after round 1 rejected) | tool: the generated REVIEW.md writes a `|` of a finding's summary as `/`, so a reviewer's `|x|=2^30` becomes `/x/` followed directly by `=2^30`, which the kernel's independence check reads as an absolute path outside TVA; the prover cannot edit the page, so its gate fails on the service's own record |
| [L139](#l139) | ACCEPTED | 31 | fixed claims, 2026-11-08 (`claims/scaled_mm_int8`), `claims/ssd_bmm_chunk_bf16`, `claims/fused_moe_bf16` | gap: a dtype as the argument of a `tl.constexpr` parameter has no stated meaning |
| [L140](#l140) | ACCEPTED | 31 | fixed claims, 2026-11-08 (`claims/scaled_mm_int8`) | gap: `tl.broadcast_to` and its method form `x.broadcast_to(shape)` are not stated as an operation |
| [L141](#l141) | ACCEPTED | 32 | fixed claims, 2026-11-08 (`claims/scaled_mm_int8`) | gap: the integer `tl.dot` (`int8 × int8 → int32`) is exact but no device fact states it |
| [L142](#l142) | ACCEPTED | 31 | fixed claims, 2026-11-08 (`claims/scaled_mm_int8`) | gap: a pointer's `.type.element_ty`, and a pointer as the condition of a runtime `if`, are not in the subset |
| [L143](#l143) | ACCEPTED | 31 | fixed claims, 2026-11-08 (`claims/moe_sum_reduce_bf16`) | gap: `tl.cast(x, dtype=…)` has no stated meaning |
| [L144](#l144) | ACCEPTED | 33 | user, 2026-11-08 | rules: the framework names CUDA graphs; state `Compiler.Faithful` about an execution of the kernel instead, so no dispatch mechanism is mentioned |
| [L145](#l145) | CLOSED | — | user, 2026-11-08 | semantics: treat programmatic dependent launch as scheduling, like `warp_specialize`: allow `launch_pdl`, drop `gdc_wait`/`gdc_launch_dependents` from models |
| [L146](#l146) | ACCEPTED | 33 | user, 2026-11-08 | scope: descriptors built inside a kernel (`tl.make_tensor_descriptor`) are excluded; host `TensorDescriptor`s stay covered (settles L120, L121) |
| [L147](#l147) | ACCEPTED | 32 | library review round 31 (lr-31a) | wording: the `tl.cast` entry does not say what a float narrowing is (`bfloat16 ↔ float16` is rejected) |
| [L148](#l148) | ACCEPTED | 32 | library review round 31 (lr-31a) | wording: a block's `.type` is not a dtype; only a scalar's `.type` and a block type's `.element_ty` stand where a dtype goes |
| [L149](#l149) | ACCEPTED | 34 | fixed-claims recheck, 2026-11-08 | gap: the reduction and scan facts allow any tree; Triton's `tl.sum`/`tl.cumsum` tree depth on sm_90 is not stated, so some claims' tolerances accept wrong kernels |
| [L150](#l150) | ACCEPTED | 33 | fixed-claims recheck, 2026-11-08 | gap: the 16-bit `tl.dot` facts (`DotF16`, `DotBF16`) charge `(K+2)·2^-22` per step, about 6× their own per-chunk derivation, and the whole `|acc|` on a fed-back accumulator |
| [L151](#l151) | ACCEPTED | 32 | fixed-claims recheck, 2026-11-08 | gap: a pointer minus an integer (`p - i`) is not stated |
| [L152](#l152) | ACCEPTED | 32 | fixed-claims recheck, 2026-11-08 | gap: `dtype.is_fp8()` (and the other dtype predicates) are not in the subset |
| [L153](#l153) | ACCEPTED | 32 | fixed-claims recheck, 2026-11-08 | soundness: exclusions.md says tutorial 10's fp4 launches are rejected on the host, but `--format mixed` (e4m3 × e2m1) runs on the H200 |
| [L154](#l154) | ACCEPTED | 33 | fixed-claims recheck, 2026-11-08 | sharpness: small precision items (`tl.ItofRel .binary32` has one value for |z| < 2^53; `AddRel` widens a non-product operand; `tl.exp` overflow and `1/∞`; `TensorDesc.Valid` zero or short strides without evidence) |
| [L155](#l155) | ACCEPTED | 34 | library author, round 33 (from L154 (2)) | sharpness: `AddRel` lets both operands of a fusable addition enter unrounded; an `fma` takes one product |
| [L156](#l156) | ACCEPTED | 34 | library review round 33 (lr-33a) | wording: evidence/num-special-f32's conclusion 2 still says the library states no `tl.exp` overflow fact |
| [L157](#l157) | ACCEPTED | 34 | claim fix (lora_shrink_bf16), user decision 2026-11-08 | coverage: relaxed read-modify-write atomics on memory other programs touch race (Q10), which excludes split-K accumulation (vLLM's default LoRA shrink) |
| [L158](#l158) | ACCEPTED | 34 | claim re-review (rejection_greedy_i32) | wording: SEMANTICS.md does not say a `torch.bool` argument is `*i1` and that loading an `*i1` element reads one byte and tests ≠ 0 |
| [L159](#l159) | ACCEPTED | 34 | claim fix (attn_bwd_pre_f16) | lemma: no depth form of `sum_bcast_mul_tree_err` (a prover must compose `sumc_err_depth` with the products' roundings) |
| [L160](#l160) | ACCEPTED | 36 | user decision 2026-11-08 | tooling: the release check recomputes every kernel's gate and certificate, which is no longer needed (the kernels are becoming benchmark tasks) |
| [L161](#l161) | ACCEPTED | 35 | library review round 34 (lr-34a) | soundness: `Step.defer` treats a whole tensor load or store as one step, so a store that uses an atomic's result holds back every later atomic of the program; the device orders each thread's accesses only (a well-behaved launch's H200 outcome has no model outcome) |
| [L162](#l162) | ACCEPTED | 35 | library review round 34 (lr-34a) | wording: the source and IR definitions of an atomic's "unused" result disagree (a value read only by a dead pure operation; returned from a helper called as a statement) |
| [L163](#l163) | ACCEPTED | 35 | library review round 34 (lr-34b) | soundness: ptxas fuses the final multiply of float32 `/` (`div.full.f32`) with a following add into one `FFMA`, but SEMANTICS.md counts only `*`, `tl.dot` and `tl.rand` as products, so it calls that add IEEE |
| [L164](#l164) | ACCEPTED | 36 | library review round 34 (lr-34b) | wording: float `atomic_max`/`atomic_min` compile to two predicated integer atoms (at most one runs per lane), not "one `atom`" |
| [L165](#l165) | ACCEPTED | 37 | library review round 35 (lr-35a) | wording: three sentences around the atomic-result exclusion (the barrier sentence, the `relaxed` form of the counterexample, scope 12's rule when a reviewer cannot tell whether a cell is shared) |
| [L166](#l166) | ACCEPTED | 36 | library review round 35 (lr-35b) | wording: `tl.fdiv(..., ieee_rounding=True)` has no meaning in the library, and `tl.softmax(ieee_rounding=True)` is called a product though it ends in an IEEE division |
| [L167](#l167) | ACCEPTED | 37 | library review round 36 (lr-36a) | wording: SEMANTICS.md "Programs" says without qualification that the semantics performs a program's atomics in every order their dependences allow (false inside the atomic-result exclusion); LIBRARY_REVIEW.md:106 still asks whether a claim can "pass `tvr release`" |
| [L168](#l168) | FIXED | 38 | library review round 36 (lr-36b) | wording: `tl.softmax` of float16/bfloat16 compiles (it divides in float32) though SEMANTICS.md can be read as saying it fails; "the result is one of the two values" in "Atomics" means the new value; the atomic dtypes Triton accepts are not listed |
| [L169](#l169) | FIXED | 38 | library review round 37 (lr-37a) | wording: SEMANTICS.md "Programs", "Atomics" and review-questions Q7 say a step using any lane of an atomic's result waits for the whole atomic; `Step.defer` is finer (a store masked to lane 0 waits only for lane 0) |
| [L170](#l170) | FIXED | 38 | library review round 37 (lr-37a) | wording: evidence/gate-round7 and evidence/tool-round18 describe the gate and pages comparisons removed in round 35 in the present tense |
| [L171](#l171) | FIXED | 38 | library review round 37 (lr-37a) | wording: LIBRARY_REVIEW.md's gate row omits the release check's shipped, packages and sandbox steps; the audit also reads the kernels' imports |
| [L172](#l172) | FIXED | 38 | library review round 37 (lr-37b) | wording: review-questions Q7 describes the wrong sm_90 instructions for 16-bit atomics (a single float16/bfloat16 add and a 16-bit atomic_cas are a 32-bit CAS loop; contiguous lanes are `.v8` atomics, not `f16x2`) |
| [L173](#l173) | FIXED | 38 | library review round 37 (lr-37b) | wording: SEMANTICS.md "Atomics" does not say the operand is converted to the pointer's element type, or that `atomic_cas` converts nothing (mismatched operands are rejected on the host) |
| [L174](#l174) | FIXED | 38 | library review round 37 (lr-37b) | wording: "signed max or unsigned min" (S3) holds for `atomic_max` only; its line references are off |

## L1

**A hint inside a loop body on a loop-carried name is not a no-op.**

- **Problem.** The library said such a hint is a no-op, and tutorial 08 relied on that. But to
  find the loop-carried names, Triton's front end first visits a `for` or `while` body once with
  every name still bound to its pre-loop value. So the hint marks the pre-loop value, and the
  compiled loads then assume 16-byte alignment. With `lda = 36` the kernel faults.
- **Evidence.** [evidence/hints-dry-run](../../evidence/hints-dry-run/README.md).
- **To do.** State the meaning (a check on the pre-loop value at the loop's entry, plus no effect
  on the block argument) and the transcription rule. Cover the induction variable, and loops
  inside loops.
- **Affects.** The hint operations' documentation, [TRANSCRIPTION.md](../TRANSCRIPTION.md) §8,
  and the protocol's drop reasons (`no-op:block-arg-hint` is withdrawn).
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md).

## L2

**What a hint on a rank ≥ 2 tensor promises.**

- **Problem.** `tl.multiple_of(a_ptrs, [16, 16])` on a 2-D pointer tensor, read per element,
  would claim every element is 16-byte aligned, which is false for tutorial 08. Triton 3.8.0
  uses the value along the fastest-varying dimension only.
- **Problem.** The `contig` argument of `multiple_of` is defined by "Triton's contiguity", which
  a kernel reviewer cannot check.
- **To do.** Define both in terms of the tensor's values.
- **Coverage round 6 ([coverage](round-6/coverage/matrix.md), N4):** tutorial 09's hints are false on partial tiles, so under the
  library's reading (a hint is a check; false is undefined behaviour) those launches are undefined,
  while the device computes them correctly (the tutorial's own `validate(32, 32, 32)`). Decide what a
  false N-D hint means, or say kernels must narrow to full tiles.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md). Both found the readings no stronger than Triton 3.8.0's documentation and the
  compiler's belief; two are stronger than what the compiler relies on (L58).

## L3

**A launch rejected before it runs is outside the claim** (user decision, 2026-10-27).

- **Problem.** Too much shared memory, too many registers, or a `tl.dot` below its minimum
  size: Triton reports these on the host before any program runs. The rule page asked kernel
  reviewers to check that every covered instance compiles.
- **To do.** State in `TVA.Claim` and `Rules.Compiler.Faithful` (docstrings and the rule page)
  that such a launch has no outcome. Remove the compile-log check from the rule page.
- **Fixed in round 1** ([notes](round-1/author/notes.md)); **reopened** by library review ([lr-1](round-1/lr-1/report.md)): a rejection must be *reported to the host with an error*, and stale text remains. Round 2.

## L4

**Device facts with constants, owned by the library.**

- **Problem.** Kernels chose their own accuracy constants, and justified them with another
  project's measurements. Under the protocol, device facts are the library's, on TVA's own
  evidence (Triton 3.8.0, the H200), and kernels cite them by name.
- **To do,** for sm_90 with Triton 3.8.0:
  - `tl.math.exp2` on `float32`: the lowering (libdevice, with flush-to-zero), the constants,
    and `exp2(-∞)`;
  - `tl.math.log2` on `float32`: the lowering (libdevice), the constants;
  - `/` on `float32` (`DivFull`): TVA's own measurement;
  - `tl.dot` with `float16` and `bfloat16` operands into `float32`: the constants;
  - each as a named fact with a page and evidence.
- **Round 4 ([lr-4a](round-4/lr-4a/report.md)):** the list of facts to add also needs `tl.exp`, `tl.sqrt`, the fp32→fp8
  conversion and `dot_scaled`, which the pinned tutorials use. The device survey found no stated
  fact false on the H200 (exact operations IEEE with subnormals, exhaustive where possible;
  `DivFull` within its bound, at most 0.654 of it on about 3·10^9 pairs). Round 7.
- **Coverage round 6 ([coverage](round-6/coverage/matrix.md)):** also fp16 and fp8 `tl.dot`, and `dot_scaled` as Triton emulates it
  on sm_90 for mxfp8 and the mixed format (tutorial 10, user's decision in L37: H200 only).
- **Round 9** ([notes](round-9/author/notes.md)): accepted by [lr-9a](round-9/lr-9a/report.md) and [lr-9b](round-9/lr-9b/report.md). Every evidence folder re-ran identical
  (num-div-f32 sampled by lr-9b), adversarial tests found nothing outside a fact, and
  `Platform.sm90ref` satisfies the facts together. Follow-ups: L64, L66, L67; the float layer the
  facts are stated in is rewritten under L65.

## L5

**A descriptor built inside the kernel writes launch scratch memory.**

- **Problem.** With `tl.make_tensor_descriptor`, each program writes its descriptors into
  memory the launcher allocates. The semantics has no such memory, so "every other byte is
  unchanged" would be false.
- **To do.** Model launch scratch, or exclude in-kernel descriptors (with host-built
  descriptors still covered).
- **Round 8** ([notes](round-8/author/notes.md)): pure `tl.make_tensor_descriptor`, and scope 5's descriptor scratch.
  Accepted by [lr-8b](round-8/lr-8b/report.md); **reopened** by [lr-8a](round-8/lr-8a/report.md): `tl.make_tensor_descriptor` admits a zero shape entry,
  and on the H200 every load or store through such an in-kernel descriptor faults ("an illegal
  instruction was encountered", `round-8/lr-8a/work/desc_edges.py` E2, E3, E5), while the model
  gives the padding or writes nothing. So `Faithful`'s "never faults" is false for that
  well-behaved model. Building one without accessing it is harmless (E7), and a host-built one is
  rejected on the host (E6). Make an access through a descriptor with a zero shape entry undefined
  (`0 < shape`), say so in SEMANTICS "Tensor descriptors", and add the cases to the evidence.
  Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md).

## L6

**A drop reason for `tl.static_assert`.**

- **Problem.** None of the drop reasons fit a compile-time assertion.
- **Fixed in round 1.** The drop reason `compile-check`, whose note names the condition of
  `Inst.ok` that makes the assertion hold: in the protocol (§7), the tool's drop list and
  [TRANSCRIPTION.md](../TRANSCRIPTION.md) §10.
- **Reopened** by library review ([lr-1](round-1/lr-1/report.md)): the condition is evaluated as an
  ordinary expression, so an operation inside it runs. The drop applies only to conditions free of
  operations. Round 2.

## L7

**TVA cites no other project's results, lemmas or measurements** (user, 2026-10-27: "let's make
sure the independence of tva, provers and reviewers cannot cite other project's results or
lemmas").

- **Rule.** TVA's results come from TVA and Mathlib, and its evidence is its own. No library
  file, kernel file or review record refers to another project.
- **Enforced** by `tvr audit` and the gate's `independence` check:
  - Lean imports only `TVA`, the kernel's own modules and `Mathlib`;
  - the lake configuration requires only Mathlib;
  - no text names another project or a path into one.

  `tvr file` and `tvr respond` refuse a filing or a response that does.
- **Fixed in round 1** ([notes](round-1/author/notes.md)). One consequence is open: `DivFull`
  had rested on another project's measurement, and now waits for TVA's own (L4).
- **Fixed in round 2**; **reopened** by library review round 2 ([lr-2a](round-2/lr-2a/report.md), [lr-2b](round-2/lr-2b/report.md)): home paths written as `$HOME/…`, `${HOME}/…`, `file:///home/…` or `~<user>/…` pass the audit. Round 3.
- **Reopened** by library review ([lr-1](round-1/lr-1/report.md)): 14 of 21 probes passed the audit.
  Examples: `public import`/`meta import`, a `[[require]]` whose `name` is not its first key, a
  `[[lean_lib]]` outside TVA, unscanned file types, names inside identifiers, `fpv/tva*` and `../`
  paths. Round 2.
- **Fixed in round 3; reopened** by library review round 3 ([lr-3a](round-3/lr-3a/report.md), [lr-3b](round-3/lr-3b/report.md)): `${HOME:-}/…`,
  `${HOME%/}/…`, `$(echo ~)/…`, JSON `\/` escapes, and a home path right after `>`, `)`, `]`, `}` or
  `-` (a shell redirect with no space) still pass. Round 4. **Coordinator's note:** this is the
  third reopen of a pattern-by-pattern scanner. Prefer a rule that does not depend on what precedes
  the path: flag any path segment `fpv/<name>` with `<name>` other than `tva`, wherever it occurs,
  besides the hashed names. State in protocol §2 that the text scan is lexical (paths computed at run
  time are not checked), while the guarantee for results and lemmas is Lean's imports and the lake
  configuration.
- **Fixed in round 4 (accepted by [lr-4a](round-4/lr-4a/report.md)); reopened** by [lr-4b](round-4/lr-4b/report.md): files pinned by sha (`upstream_files()`),
  which include each kernel's source, are checked for names only, so a kernel source comment
  `# from ~/fpv/<sibling>/…` passes `tvr audit`; quoted spellings such as `~/"fpv"/<x>`, and `&sol;`,
  pass too. Round 5. Scan kernel sources for paths as well (they are the prover's files, not
  upstream's).
- **Round 6** (the release check, L36): accepted by [lr-6b](round-6/lr-6b/report.md); **reopened** by [lr-6a](round-6/lr-6a/report.md):
  - the sandbox hides only the home and temporary folders, so the system's other temporary and
    optional folders, data folders and the whole Python prefix stay visible: a file placed in the
    system's second temporary folder, read by a path built at run time, passed `tvr audit` inside
    the sandbox;
  - the `packages` step sees only files git does not ignore (Mathlib ignores `**/__pycache__/`, and
    `.git/info/exclude` can hide any path): a module `Mathlib.__pycache__.X` passes the import
    check and builds, and built `.olean` files are borrowed unchecked.

  Round 7. Make the sandbox show only what TVA ships plus an explicit list of system paths the
  toolchain needs (read-only), and check package trees including ignored files, or rebuild them.
- **Round 7** ([notes](round-7/author/notes.md)): the sandbox is an allowlist, and Mathlib's packages are made
  afresh from their pinned commits and built inside it. Both reviewers found the allowlist as
  described. **Reopened** by both for the packages:
  - [lr-7a](round-7/lr-7a/report.md): nothing checks that the pinned commit is Mathlib's. `make_packages` fetches it from
    the working tree's own checkout first, and only the direct package's URL is checked. A local
    commit adding `Mathlib/Extra/Borrowed.lean` passed the lake check, the packages step and the
    import check. Fetch only from upstream, require the commit to be reachable from the manifest's
    `inputRev`, and match inherited packages to Mathlib's own manifest;
  - [lr-7b](round-7/lr-7b/report.md): the `packages` step runs git outside the sandbox with the user's home, so the user's
    git configuration decides what the checkout writes. With a global attributes file and a smudge
    filter, every `.lean` file differed from the commit's blobs and `make_packages` reported nothing.
    Run git with no user or system configuration, and check each file against its blob.

  Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): packages are fetched from upstream only, Mathlib's pin is checked
  against the upstream refs, inherited packages must match Mathlib's own manifest, git runs without
  user or system configuration, and every checked-out file is compared with its blob. Accepted by
  [lr-10a](round-10/lr-10a/report.md); **reopened** by [lr-10b](round-10/lr-10b/report.md): the check runs `git`, `bwrap` and `find` from the caller's PATH,
  which here lists four home folders before `/usr/bin`. Wrappers placed there ran as all three
  (`round-10/lr-10b/work/probes_batch.py` P3), so a user's git decides the packages and the refs
  they are checked against, and a user's bwrap decides the sandbox. Run `/usr/bin/git`,
  `/usr/bin/bwrap` and `/usr/bin/find` (resolving under `/usr`), give git a PATH of system folders
  only, and say so in protocol §2. Also ([lr-10a](round-10/lr-10a/report.md), filed as a new gap): any branch head of Mathlib's
  repository passes as `inputRev` (71 branches, some personal or for CI; G7). Require a release tag
  `v<version>` matching the toolchain, or `master`. Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): the tool runs only `/usr/bin/git`, `/usr/bin/bwrap` and `/usr/bin/find`, git
  gets a system-only PATH, and only `v4.35.0` (as a tag) or `master` (as a branch) passes as
  Mathlib's `inputRev`. Accepted by [lr-13b](round-13/lr-13b/report.md); **reopened** by [lr-13a](round-13/lr-13a/report.md): `upstream_problem` runs
  `git ls-remote` (and, for `master`, `git clone`) without `-C`, in the caller's folder, which is
  TVA's root, a git repository. `_git_env` removes the user's and the system's configuration, not
  that repository's `.git/config`, which does not ship, is not scanned and the user can write. In
  `round-13/lr-13a/work/gitcfg.py`, a `url.<host>.insteadOf = https://github.com/` line in it sent the
  tool's ref lookup for Mathlib's tag to another host, so that file decides which commit counts as
  Mathlib's. Run every git that needs no repository with `cwd` an empty scratch folder and
  `GIT_CEILING_DIRECTORIES` set to it, and say in §2 that no repository's configuration is read
  but the new package repository's own. Round 16.
- **Round 16** ([notes](round-16/author/notes.md)): accepted by [lr-16a](round-16/lr-16a/report.md) and [lr-16b](round-16/lr-16b/report.md).

## L8

**`Block.Fits` is stricter than `int32` indexing needs** (a library request by the cleared
vec-add kernel's prover).

- **Problem.** `TVA.Block.Fits p B` asks `p*B + B < 2^31`. The lane lemmas need only the largest
  offset, `p*B + B − 1`, to fit `int32`, that is `p*B + B ≤ 2^31`. So a kernel on `n` elements
  cannot cover `2^31 − B < n < 2^31`, though its indexing is exact there.
- **To do.** Relax `Fits` to `p*B + B ≤ 2^31`. Add a `fits_of_grid` form for `n < 2^31` when
  `B` divides `2^31`.

## L9

**`Launch.Valid` admits grids the launcher silently skips** (library review, round 1).

- **Problem.** Triton 3.8.0's launcher guards the launch with `gridX * gridY * gridZ > 0` computed
  in C `int` (`driver.c`). A program count in `[2^31, 2^32]` wraps to a value `≤ 0`, and the launch
  returns normally with no program run. Grids `(2^30, 2, 1)` and `(2^16, 2^15, 2)` satisfy
  `Launch.Valid`, so `Compiler.Faithful` is false there for any kernel that writes.
- **To do.** Add `L.size < 2^31` to `Launch.Valid` (docstring, SEMANTICS.md, the rule page), pin
  `driver.c`, and put the test in `evidence/`.

## L10

**`no-op:print` drops arguments that may have effects** (library review, round 1).

- **Problem.** `tl.static_print`, `tl.device_assert` and `tl.device_print` evaluate their arguments
  as ordinary expressions. With `debug` off, an atomic inside any of them still runs. A Python
  `assert` statement lowers to `device_assert`, and the drop list does not name it.
- **To do.** Allow the drop only for pure arguments (the same test as L6); otherwise transcribe the
  argument operations. Name `assert`.

## L11

**`rules/num.md` asks kernel reviewers to check device truth** (library review, round 1).

- **Problem.** Its section "What the reviewer judges" asks for the PTX or a test. Kernel reviewers
  read that page, and the protocol forbids them that check.
- **To do.** Address the page's device questions to library review, and tell kernel reviewers what
  they do check (the right fact for the operation).

## L12

**No digest covers the transcription rules** (the library author's notes, round 2).

- **Problem.** A kernel item's digest covers the Lean constants it reaches, and the claim reader
  hashes the docstrings of `TVA.Claim`, `TVA.Rules` and `TVA.Platform`. Nothing covers
  `docs/TRANSCRIPTION.md` or the drop list in `tools/tvr.py`. So when a rule or a drop reason
  changes (as L6 and L10 did), no kernel's `map` or `model` re-opens, and a map accepted under the
  old rule stays accepted.
- **To do.** Give each transcription rule and drop reason an identity, have the map entries name
  the ones they use, and fold their text into the `map` (and `model`) digests.
- **Fixed in round 3; reopened** ([lr-3a](round-3/lr-3a/report.md), [lr-3b](round-3/lr-3b/report.md)): the parser folds the last paragraph of
  TRANSCRIPTION §3, which the page calls general text, into `[atomic]`. Round 4, with L27.

## L13

**A kernel's Lean can add `main` without the kernel checking it** (soundness; library review round 2,
[lr-2a](round-2/lr-2a/report.md)).

- **Problem.** Meta code in a kernel file (`run_cmd`, `#eval`, `elab`, `macro`, `initialize`) can
  call `addDeclCore … (doCheck := false)` and add `main` and `consistent` unchecked. A bundle with
  `Post = False` passes build, shape, axioms, usage and hygiene (`work/lean/GateHygiene.lean`).
- **To do.** Re-check the kernel's compiled output with an independent kernel checker, and/or
  forbid meta-level environment changes in kernel files.

## L14

**"Pure" is unsound for aggregate operands** (soundness; [lr-2a](round-2/lr-2a/report.md)).

- **Problem.** On an aggregate value, `+`, `==`, `<` and the like dispatch to its `@triton.jit`
  dunder methods, which are inlined and may run atomics. So `compile-check`, `no-op:print` and `dead`
  can drop a line with an effect (`work/l6_operator_dispatch.py`: 4 of 4 programs).
- **To do.** Restrict purity to operands of constexpr, dtype, shape, tuple or plain scalar type, or
  exclude aggregates.

## L15

**One program is not sequential for global memory** (soundness; [lr-2b](round-2/lr-2b/report.md)).

- **Problem.** The library says a program's later steps see its earlier writes (SEMANTICS.md,
  `Prog`/`Launch`, S1), and drops `tl.debug_barrier` and `num_stages`. On the H200:
  - a program that stores a block and loads it back reversed reads stale data in 24–40% of
    elements;
  - a loop that loads the row the previous iteration stored reads stale data when Triton pipelines
    it (`num_stages = 3`, or a `tl.dot` in the body at the default).

  A barrier, the same offsets (without pipelining), or `num_stages = 1` gives the sequential result.
  So `Compiler.Faithful` is false for such models.
- **Decision (user, 2026-10-27): Option 1, no conflicting accesses within a program.** Two global
  accesses of one program conflict when they touch the same bytes and at least one writes them (a
  store or an atomic), unless both are atomics (two atomics do not race, but their order within the program is unspecified: the semantics must
  allow every order, which matters for returned old values, `xchg` and `cas`). A model with a conflict anywhere in a program, across loop iterations
  included, is not well behaved, like a race between programs, and no claim covers it.
  `tl.debug_barrier` keeps no meaning, and `num_stages` can stay dropped, since any reuse it could
  reorder is excluded. Correct SEMANTICS.md ("Programs", "No barrier semantics", "Granularity"), the
  `Prog`/`Launch` docstrings, `Compiler.Faithful`'s well-behaved condition, and S1. TVA's own evidence
  of the hazard is the reproduction in `issue/stale_readback.py` and lr-2b's `work/`; it shows why
  the rule exists, and the rule promises no ordering. Kernels that write and read back their own
  data, or update an input in place, are out of scope: say so in `exclusions.md`. A barrier-phase
  model (Option 2) may be added later without breaking claims made under this rule.

## L16

**`tl.max`/`tl.min` widen small integers to `int32`** (gap; [lr-2b](round-2/lr-2b/report.md)).

- **To do.** State it in SEMANTICS.md and by the integer reductions (to `int32` for any
  signedness; `uint8`/`uint16` zero-extended).

## L17

**Missing operations** (gap; [lr-2b](round-2/lr-2b/report.md)).

- **Problem.** These are missing:
  - `tl.permute`, and `trans` of rank 3 or more;
  - PDL: `gdc_wait`, `gdc_launch_dependents` and `launch_pdl`, which `Launch.Valid` does not
    exclude;
  - `tl.argmax`, which the docs name but nothing defines.
- **To do.** Add `permute`. Define PDL, or require `launch_pdl = false` and list it in
  exclusions.md. Define the index reductions, or drop the references to them.
- **Fixed in round 3** for `permute` and the index reductions; **reopened** ([lr-3a](round-3/lr-3a/report.md), [lr-3b](round-3/lr-3b/report.md)):
  `[no-op:gdc]` is false. A kernel launched without PDL that calls `gdc_launch_dependents()` lets the
  next kernel, launched with `launch_pdl=True`, run while it is still running (`evidence/pdl` part C;
  [lr-3a](round-3/lr-3a/report.md) `work/pdl_successor.py`: its input overwritten in 20 of 20 runs). Round 4, with L18 and L19.

## L18

**The environment changes defaults the model fixes** (gap; [lr-2b](round-2/lr-2b/report.md)).

- **Problem.** `TRITON_F32_DEFAULT` sets `tl.dot`'s `input_precision` when the source omits it;
  `DotCfg.precision` defaults to `tf32`.
- **To do.** Record it in the launch, or state that claims assume it unset.
- **Fixed in round 3** for environment variables; **reopened** ([lr-3a](round-3/lr-3a/report.md)): the same switches are
  set from Python with no environment variable set: `knobs` (`propagate_env=False` with
  `fp32_default`), `knobs.runtime.debug`, `add_stages_inspection_hook` (rewrites the PTX) and a
  `jit_cache_hook` that skips the launch (`work/options_knobs.py`). Round 4, with L17 and L19.
  **Coordinator's note:** L17, L18 and L19 are one class. One scope statement in `Faithful` would
  close it: the launch is made through Triton's defaults, with no knob set (by the environment or by
  Python), no hook installed, only the launch keywords `Launch` models, and no programmatic dependent
  launch on either side of it. List the rest in exclusions.md.

## L19

**Launch keywords outside `Launch` change the compiled code** (soundness; [lr-3a](round-3/lr-3a/report.md)).

- **Problem.** With the environment at its defaults, keywords such as `ir_override` (gave `x+2` for
  `x+1`), `allowed_dot_input_precisions` (switched `tl.dot` to `ieee`), `enable_reflect_ftz=False`
  (turned FTZ off), `extern_libs`, `ptx_options` and `arch` change what runs. `Faithful` does not
  exclude them.
- **To do.** See L18's note: cover only the launch keywords `Launch` models.

## L20

**Model text can mean another operation** (soundness; [lr-3a](round-3/lr-3a/report.md)).

- **Problem.** A kernel can declare `tl.add` in its own namespace, or a `local notation "tl.add"`,
  that is the fusable `addc` by `rfl`. The model reads as the library's operation but is not. Neither
  the hygiene check nor the namespace check catches it (`work/lean/ShadowRoute.lean`,
  `NotationRoute.lean`).
- **To do.** The gate checks that every operation the model mentions resolves to the library's
  constant (for example from the elaborated model's constants), and forbids notation, macros and
  declarations in the kernel that shadow library names.
- **And (user, 2026-10-27): give kernel reviewers the full Lean.** "this can also fix by providing
  the reviewer agents with full context to pass the lean proof. this cheating behavior can easily
  caught." The kernel reviewer prompt and REVIEWER_GUIDE.md now say the reviewer need not read the
  Lean files. Instead the reviewer reads every one of them, checking what each name means: every
  `def`, `notation`, `open`, `namespace`, `instance`, `attribute`, `export` and `set_option`, and
  anything that changes what a library name or a model line means. Lean checks the proofs; what the
  names mean is the reviewer's to judge. Such a change is a `MODEL_MISMATCH`. The gate's mechanical
  check stays as a backstop.
- **Fixed in round 4 (accepted by [lr-4a](round-4/lr-4a/report.md)); reopened** by [lr-4b](round-4/lr-4b/report.md): two shadowing routes pass on a copy:
  `def Launch.mk` in the kernel namespace (the claim reader skips names ending in `mk`, `ext`, … as
  if generated), and `open TVA` with its names on the next line (`opened_names` reads one line).
  Also ([lr-4a](round-4/lr-4a/report.md), filed as a gap): a local binder named `tl` makes `tl.add` mean `addc`, and the gate
  passes it. Backstops [lr-4b](round-4/lr-4b/report.md) suggests: flag non-library instances, binders named `tl`, and look-alike
  names (`tℓ.add`). Round 5.

## L21

**Autotuning and hooks around the claimed launch** (gap; [lr-3a](round-3/lr-3a/report.md), [lr-3b](round-3/lr-3b/report.md)).

- **Problem.** An autotuned call runs the kernel several times on the real arguments, and
  `reset_to_zero`, `restore_value` and the pre/post hooks run host code that writes its buffers. The
  library does not say how this relates to a claim's `init`. `[decl]` drops the decorator as
  "covered by the model's definition".
- **To do.** Say that a claim covers one launch, from the memory at that launch, and that hooks and
  benchmark runs are host code that `Pre` must account for.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md). Hooks change the launch's arguments too (L61).

## L22

**Stale text after L15** (wording; [lr-3a](round-3/lr-3a/report.md)). The descriptor docstring and the pointer/TMA exclusion
still read as if a program were sequential.

## L23

**`[literal]` is wrong for fp64 and bf16** (soundness; [lr-3b](round-3/lr-3b/report.md)).

- **Problem.** A Python scalar operand becomes `scalar_constant(c, dtype of the operation)`. In
  fp64 that is the exact double; in bf16 the constant is built from `std::to_string` (six
  decimals), so `x * 1e-7` multiplies by `+0` (`work/literals.py`, `literals2.py`: 128 of 128).
- **To do.** Give the constant per the operation's dtype; keep nearest-fp32-then-convert only for
  `other=` and `.to`.

## L24

**`[itof-narrow]` under `rtz`** (soundness; [lr-3b](round-3/lr-3b/report.md)). With `fp_downcast_rounding="rtz"` the
narrowing is `tt.fp_to_fp`, not a fold into one `sitofp`: 32128 of 65536 ints differ
(`work/itof_rtz.py`). Restrict the rule to default rounding; with `rtz`, model `tl.itof`, then
`tl.cvt … .rtz`.

## L25

**`[config]` drops conditions that may run operations** (soundness; [lr-3b](round-3/lr-3b/report.md)). The front end
evaluates every `if` condition, so `if tl.atomic_add(c, 1) is not None:` runs the atomic in every
program (`work/config_cond.py`). Allow the drop only for a `[pure]` condition.
- **Fixed in round 4; reopened** ([lr-4a](round-4/lr-4a/report.md), [lr-4b](round-4/lr-4b/report.md)): the rule makes every operation of an impure
  condition a model line, but the front end skips the right side of a compile-time `and`/`or`, the
  untaken side of a compile-time conditional expression, and an `elif` after a taken compile-time
  `if` (device: atomic runs in 0 of 4 programs where the rule's model runs it; the author's case C4
  agrees). `[pure]`'s rule for impure arguments of `[compile-check]` and `[no-op:print]` has the same
  defect ([lr-4a](round-4/lr-4a/report.md): `FLAG and <atomic>` in `tl.static_print` runs no atomic). Round 5: transcribe the
  operations the front end evaluates, by its short-circuit order.

## L26

**`[sum-bcast-mul]` names one spelling** (soundness; [lr-3b](round-3/lr-3b/report.md)). The combine pass matches the IR, so
`tl.expand_dims` spellings (and a `tl.reduce` with `+`) are rewritten into a dot too, even with
fusion off (`work/bcast_mul.py`: 58 of 256). State the rule by the computation.

## L27

**SEMANTICS.md is outside every digest** (soundness; [lr-3b](round-3/lr-3b/report.md)). `[call]` and the conversions a model
must write rest on it, but `rule_parts` folds in only TRANSCRIPTION.md and exclusions.md
(`work/l12_digests.py`, X3/X4). Fold it into the `map`/`model` digests, or move each obligation
into a transcription rule. With L12.

## L28

**No rule for a runtime `if`/`else` or an early `return`** (gap; [lr-3b](round-3/lr-3b/report.md)). Tutorial 08 needs one.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md). Wording follow-up in L61.

## L29

**No rule for scalar argument types** (gap; [lr-3b](round-3/lr-3b/report.md)). A Python `float` is passed as fp32, a `bool`
as `int1`, an int at or above `2^31` as `int64`, and `1` as a constexpr.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md).

## L30

**Block pointers are aggregates** (gap; [lr-3b](round-3/lr-3b/report.md)). `tl.make_block_ptr` returns `_block_ptr`, an
`@_aggregate` class, so exclusions.md excludes block pointers while `[block-ptr]` covers them.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md).

## L31

**Wording** ([lr-3b](round-3/lr-3b/report.md)): SEMANTICS "Compiler hints" still states the withdrawn hint rule (L1);
`[no-op:debug_barrier]`'s reason ignores the order of atomics; `[dot-add]` says "constant zero" for
"±0 after folding".
- **Round 6:** reopened by [lr-6b](round-6/lr-6b/report.md) (and noted by the [coverage](round-6/coverage/matrix.md)): `lean/TVA/Lang/Ops.lean` "Compiler hints"
  (lines 685–701) still says a hint on a loop variable has no effect and is dropped, and the
  `multiple_of`/`multiple_ofN` docstrings still define `contig` by Triton's contiguity; both
  contradict SEMANTICS.md (L1, L2 open). Round 7.

## L32

**Text kernels rely on is outside the digests** (soundness; [lr-4a](round-4/lr-4a/report.md), [lr-4b](round-4/lr-4b/report.md)).

- **Problem.** Module docstrings, and docstrings of structure fields whose projection no term uses,
  are not hashed: L18 changed `DotCfg.precision`'s docstring and re-opened no kernel item ([lr-4b](round-4/lr-4b/report.md)). The
  list of citable device facts in `docs/rules/num.md` is outside every digest too, so changing or
  withdrawing a fact there re-opens nothing ([lr-4a](round-4/lr-4a/report.md)).
- **To do.** Hash every library docstring a kernel item's cone can reach, including module and
  field docstrings, and fold `rules/num.md` (or each fact's section) into the digests of the
  assumptions that cite it.

## L33

**No error bound for fusable-add reductions** (gap; [lr-4b](round-4/lr-4b/report.md)). `tl.sumc`, `tl.cumsumc` and the tree form
of `sum_bcast_mul` have no error lemma, and `Tree.eval_err`'s per-node condition cannot be met for
them.
- **Round 9** ([notes](round-9/author/notes.md)): accepted by [lr-9a](round-9/lr-9a/report.md) and [lr-9b](round-9/lr-9b/report.md).

## L34

**The shared-atomics kit is narrow** (gap; [lr-4b](round-4/lr-4b/report.md)). No rule covers a masked tensor atomic, and the only
accumulation lemma is `int32` addition.
- **Round 9** ([notes](round-9/author/notes.md)): `SafeA.atomic`, integer accumulation at any width, max/min in any order,
  and a float replay bound. Accepted by [lr-9a](round-9/lr-9a/report.md) (on the device, one read-modify-write per enabled lane,
  116 configurations); **reopened** by [lr-9b](round-9/lr-9b/report.md): `replay_err` and the max/min folds weight each logged
  step by a function of the step, and a logged step records only the cell and its old and new
  values. Two programs adding `1.0` and `0.5` to a `float32` cell holding `2^24` log the same step,
  so no weighting gives each its own operand, and nothing bounds the cell against `x₀` plus the
  programs' operands. State the accumulation per position (step–operand pairs), with a lemma that
  merges the programs' operand lists along the interleaving, and the same for max/min; or record
  the operand in the step. Round 14 (after L65's float layer).
- **Round 12** ([lr-12b](round-12/lr-12b/report.md), filed as a gap): `replay_err` bounds a cell with `n` the length of the
  whole replayed log, so each step on another cell enters as an approximate addition of `0`; for
  split-K over `M·N` cells the bound carries `(1+2^-23)^(M·N·S) − 1`. Count only the steps on the
  cell, or add a lemma restricting a replay to them. The round-14 batch, in review, restates
  `replay_err` this way and adds `replay_err_shared`.
- **Round 14** ([notes](round-14/author/notes.md)): `Paired` (operands per position, merged along the interleaving), `replay_err`
  counting only the steps on the cell, `replay_err_shared`, `replay_max`, `replay_min`. Accepted by [lr-14a](round-14/lr-14a/report.md)
  and [lr-14b](round-14/lr-14b/report.md); [lr-14b](round-14/lr-14b/report.md) applied the lemmas as a kernel would, in `round-14/lr-14b/work/Usability.lean`.
  Float `atomic_max`/`atomic_min` stay uncovered (no tutorial needs them). Wording follow-up in L101.

## L35

**Wording** ([lr-4a](round-4/lr-4a/report.md), [lr-4b](round-4/lr-4b/report.md)):
- `DivFull` says "2 ulp", but up to 2.11 ulp was measured; the formal bound holds;
- `TVA.Lang.Scalar`'s module docstring lists `tl.maximum` among the operations that build a constant
  in their own dtype;
- `tl.full` wraps an out-of-range integer constant (300 in int8 gives 44), not "`c` itself";
- the `tl.ItofRel` docstring still says MLIR folds any narrowing (false for `rtz`, L24);
- the `sum_bcast_mul` docstring admits an `a + a` combine, which Triton also rewrites into a dot;
- `Num.lean`'s "the reviewer judges" should name library review;
- protocol §2 should name shell expansions (braces, globs, `cd` then a relative path) among what
  the lexical scan does not check.

## L36

**Leverage the provers' and reviewers' judgment** (user, 2026-10-28: "our transcription is not
based on a fixed rule, we provide the guidance and the reviewer needs to check whether the model
matches the kernel"; "basically we need to assume the provers and reviewers have intelligence and
properly leverage that"). Round 6, before the gaps. Round 5's fixes (L7, L20, L22, L25, L31, L32, L35) are
reviewed with round 6, since L36 reshapes L7, L20, L25 and L32.

- **Mechanical, because complete, cheap, or invisible to a reader:**
  - the build, `leanchecker` replay and the standard axioms;
  - structural independence: imports only `TVA`, Mathlib and the kernel's own modules, and lake
    requires only Mathlib;
  - the claim's shape, and regular output formats;
  - digests, as bookkeeping of what changed. Make them complete by construction: hash whole
    library files (Lean modules and the doc pages that state facts) by the modules an item's cone
    reaches, instead of a curated list of docstrings (subsumes L12, L27, L32);
  - the gate's name-resolution check (L20) as a backstop for the common route, per the user's
    "keep both".
- **Judgment, with context:**
  - model against kernel, including every name, drop and shadowing trick (reviewers read the full
    Lean);
  - transcription: TRANSCRIPTION.md becomes guidance (worked patterns and pitfalls). Every Triton
    fact it states moves to SEMANTICS as a trusted library fact: the literals per dtype (L23), `rtz`
    narrowing (L24), condition evaluation and short-circuiting (L25), the dot rewrite (L26), the
    aggregate operators (L14). Rule IDs are no longer required in map entries; the gate does not
    check them; drop reasons are judged, with a short list of common reasons as guidance. A change to
    the guidance re-opens nothing;
  - references to other projects: **stay blocking** (user, 2026-10-28: "we need to keep the project
    independent. since tva will be released without others."). Make the guarantee complete by
    construction rather than by spelling:
    - **release check (new, blocking):** copy exactly the files TVA ships (`git ls-files` plus
      untracked files that would ship) into an empty sandbox, with `HOME` set to a directory that
      holds only the Lean toolchain, the venv TVA documents, and Triton's cache. There, build, gate
      every kernel, run `tvr lint` on every record and `tvr audit`. A dependency on anything outside
      TVA fails there, because nothing outside exists;
    - **names (blocking, kept):** other projects' names as sha256 hashes, matched in every spelling
      the scanner decodes; the other folders that share TVA's parent directory were checked by the
      coordinator (2026-10-28): the only other project is the one already hashed, and the rest are an
      archive, scratch space and two Triton installs, so no names are added;
    - **paths (blocking, kept as round 5 built it):** not extended spelling by spelling. A path
      spelling that names no project, and that the release check shows is no dependency, is a
      wording-level finding, not a soundness reopen;
  - the remaining L20 routes are reviewer territory.
- **Prompts and guides:** give rationale and stakes rather than checklists. Ask reviewers how the
  claim could be wrong.
- **Round 6:** accepted by [lr-6b](round-6/lr-6b/report.md); reopened by [lr-6a](round-6/lr-6a/report.md) only for the release check (tracked in L7).
  The digests, the guidance, the map check and the prompts hold.
- **Round 7:** accepted by [lr-7b](round-7/lr-7b/report.md); reopened by [lr-7a](round-7/lr-7a/report.md) only for the release check (tracked in L7).
  Both reviewers judged that building Mathlib from source in each run holds. lr-7a corrects the
  author's reasoning: 4 of 5 builds gave the same `.olean` digest, including one at TVA's own path,
  so the builds do not depend on the path and one run is an outlier of unknown cause. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md). The release check's remaining route is L7's.
- **User, 2026-10-28, on holes reviews find in the gate: "accept your recommendations".** For
  each hole, make the check complete when that is cheap and it catches what a reader cannot see;
  otherwise state exactly what the gate guarantees and leave the rest to the kernel reviewer, rather
  than patching one spelling at a time. As applied:
  - L50, complete: every function the source defines is classed (kernel lines, host or unused), and
    the reviewer checks that the ranges hold every function the launch runs;
  - L52, a stated limit plus cheap checks: a ledger shows that each recorded file is unchanged since
    its line; the lines themselves are trusted as the coordinator keeps them, with no commit at each
    recording. The cheap checks: `--no-lint` only for rounds 1–7, no file recorded with two sha256s,
    no folder recorded twice;
  - L71, complete: a claim proves that its conditions admit an instance, and whether they cover the
    launches the kernel is meant for stays the reviewer's judgment.
- **User, 2026-10-28, on a malicious prover: "the reviewer can challenge malicious provers and can
  access the entire proof".** A gate hole that only a prover writing a misleading claim or map can
  use (L79's unused-function header, L94's parameters before `K` and `P`) needs no new machinery: the
  docs state exactly what the gate does not check, and REVIEWER_GUIDE names the check the kernel
  reviewer makes. A cheap complete check already written stays.
- **User, 2026-10-28, on the process: "after round 20, let's stop round after round. let's have a
  one-time fix attempt".** Rounds 14 and 17–19 are reviewed as planned; round 20 is one fix
  attempt for every issue still open. Then (user, 2026-10-28: "run the final record-only review and
  then do a final fix") one final review of round 20 that records its findings, and one final fix
  batch for them, checked by the build, lint, audit and the release check, with no review after it.
- **User, 2026-10-29, on Triton's caches: "To make the verifier easy, we do not verify triton cache"
  and "we assume the kernel is new".** `Faithful`'s scope assumes that the code a launch runs was
  compiled for that launch (its source, its compile-time values, the settings of the other scope
  conditions), not taken from Triton's in-process cache or cache directory as code compiled earlier.
  TVA does not verify Triton's caches; no gate, prover or reviewer check covers this assumption.
  This settles the cache halves of L103 and L104, and L109. Evidence/compile-keys and the round-20
  reports record why the assumption is needed.
- **User, 2026-10-29, on what is verified: "So basically, we want to verify the kernel itself,
  rather than triton installation or other code programmed by users".** `Faithful`'s scope has two
  parts. (A) The kernel and its launches (the `@triton.jit` source with its helpers and decorators,
  the launch call, and what the kernel needs from its host to run, such as the allocator and the
  descriptors it is given) is checked by the kernel reviewer. (B) The environment (Triton 3.8.0 as
  installed and unmodified, the host and its process settings, knobs, environment variables and
  hooks, other host code, and the kernel assumed new) is assumed, and no gate, prover or reviewer
  check covers it. This settles how L104, L105, L107, L109 and L110 are stated: as part (B).
- **User, 2026-10-29, on autotuning: "actually we do not neccessarily cover autotune. we do
  verification basically for the kernel. that is to say, for autotune, we actually should prove the
  selected hyper-parameters. for the kernels in the tutorial with autotune, we can select a
  reasonable autotune config to provide proof."** The autotuner (its benchmark runs, its key,
  pruning and choice) is not verified. Correction (user, 2026-10-29): "our principle is, we are
  verifying kernel itself. it is not correct that 'For an autotuned tutorial kernel, the prover
  picks a reasonable config from the tutorial's list and says which. The reviewer checks that the
  claim fixes it'. actually not the prover picks a config, it is we pick a config. prover is not
  aware of the autotune." So we (the user and the coordinator) pick the config when we set up the
  kernel's folder: its source is the tutorial's without `@triton.autotune`, launched with that
  config's values; the prover and the reviewer see only that kernel (L116). Condition
  11 ("the kernel is new") is read as: the code the launch runs was compiled for it, or earlier for
  the same source, compile-time values and settings (a repeated launch of the same instance).
- **User, 2026-10-29, on the vec-add reviewers' three non-blocking wording remarks: "Fix and
  re-review".**
- **User, 2026-10-29, on L117 (condition 11 and the specialisation of runtime arguments): "for the
  viriable n_elements, its value should be merged into the assumptions of the prover. e.g., x.len >=
  n_elements, etc."** A runtime argument's value is part of the claim's instance, and the prover's
  conditions constrain it (`vec_add_f32`'s `Inst.ok` has `n ≤ x.len`, `n ≤ y.len`, `n ≤
  output.len`). So condition 11 assumes that the code a launch runs is the code Triton compiles for
  that launch's own argument values (compile-time and runtime, as the instance gives them) and
  settings, with no clause about reusing other code: code a cache returns counts when it is that
  code, as for a repeated launch.

## L37

**The library covers every pinned tutorial kernel** (user, 2026-10-28: "we need to be able to provide
proof for every kernel in the tutorial. although we will not write the proof now"; "we can exclude
layer norm backward and programmatic dependent launch. but we might want to have others").

- **Target.** Every `@triton.jit` function of the pinned tutorials (`tutorials/inventory.json`, 27
  functions in 10 files). For each one, every construct it uses has a library meaning (SEMANTICS
  and, where accuracy matters, a device fact on TVA's evidence) and guidance a prover can follow, or
  is excluded by the user's decision.
- **Excluded by the user.**
  - Tutorial 05's `_layer_norm_bwd_dx_fused`, which hands partial sums between programs under a spin
    lock (S11) and updates them in place within a program (L15).
  - Tutorial 11 (`launch_pdl`; L17).

  Say so in `exclusions.md`, citing this decision. Tutorial 05's `_layer_norm_bwd_dwdb` uses neither
  pattern and stays in scope (coordinator's reading; the user may say otherwise).
- **In scope, to be covered rather than excluded:** everything else, including tutorials 06, 09
  and 10 where they use tensor descriptors, warp specialisation or `dot_scaled`.
- **Coverage matrix.** `docs/plan/tutorial-coverage.md` lists, per function and construct: covered
  (where), gap (issue), or excluded (decision). Rounds 7 and 8 are set from it, and the library is
  settled when a round reopens nothing, raises no new soundness issue, and the matrix has no gap.
- **Coverage round 6** ([coverage](round-6/coverage/matrix.md)): 27 functions; 2 excluded, 1 fully covered, 24 with gaps (221
  covered rows, 108 gap rows, 7 excluded rows); no soundness issue found.
- **User decisions, 2026-10-28:**
  - tutorial 10's `block_scaled_matmul_kernel_cdna4` (AMD CDNA4 only) is **excluded**: TVA targets the
    H200;
  - tutorial 10's `block_scaled_matmul_kernel` is covered **on the H200 only**. mxfp8 and the mixed
    format run through Triton's emulation, which `dot_scaled`'s device fact must describe (L4). fp4
    launches are rejected on the H200 and have no outcome. Blackwell would be a later target.
- **User, 2026-10-28: "certain proof steps can be done in runtime (proof time) as well".** The
  library need not pre-state a fact for every case. Where the meaning is defined, a prover
  establishes the step while proving the kernel: by evaluating the library's definitions on the
  kernel's values (a literal's float, a promoted dtype, a shape), by deciding a finite condition on
  the covered instances (a hint, tile fullness), or by reasoning about that kernel. A construct is
  then **covered**. A gap means a proof cannot be written from what the library states. Device
  facts are the exception: what the hardware does comes from the library, on TVA's evidence.
  Performance-only primitives need no model (L44).
- **Round 8** ([notes](round-8/author/notes.md)): exclusions.md records the user's decisions. Accepted by [lr-8a](round-8/lr-8a/report.md);
  **reopened** by [lr-8b](round-8/lr-8b/report.md): tutorial 08's `grouped_matmul_kernel` has no claim on any launch.
  `tl.multiple_of(a_ptrs, [16, 16])` is false on every launch under Triton 3.8.0's documentation
  (`core.py:3307-3330`: along a dimension of contiguity 1, every element must be a multiple), under
  the compiler's belief and under the library. Tutorial 09's pointer kernels' hints are false on
  partial tiles, where the device reads other elements (evidence/hint-rank2 T2). Both reviewers
  found no sound reading of the values that makes either true.
- **User decisions, 2026-10-28:**
  - tutorial 08's `grouped_matmul_kernel` is **excluded**: its hint is false by Triton's
    documentation, so it has no defined meaning. Tutorial 08's `grouped_matmul_tma_kernel` stays in
    scope;
  - tutorial 09's pointer kernels are covered on **full tiles only**: `M` and `N` multiples of the
    block sizes of every config the autotuner may pick.

  Record both in exclusions.md, citing this decision. Round 11.
- **Coverage round 10** ([coverage](round-10/coverage/matrix.md)): 27 functions; 4 excluded whole, 276 covered rows, 14 gap rows
  (L61: 3, L64: 8, L75: 3) and 10 excluded rows; no soundness issue found. Every in-scope function
  of tutorials 01, 03, 04 and 05 is covered.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md). exclusions.md rows 46 and 47 record both decisions.
- **User decision, 2026-11-08 (L146):** descriptors built inside a kernel are **excluded** ("let's
  exclude this case. we only cover descriptor outside of triton kernels"). This removes from the
  target tutorial 08's `grouped_matmul_tma_kernel` and tutorial 09's
  `matmul_kernel_descriptor_persistent`, which build every descriptor in the kernel, and tutorial
  06's `_attn_fwd` on the launches where its host passes tensors (the Hopper warp-specialised path,
  where `_maybe_make_tensor_desc` builds them); `_attn_fwd` with host `TensorDescriptor`s, tutorial
  09's `matmul_kernel_tma_persistent` and tutorial 10 stay in scope. Recorded in exclusions.md
  (round 32, [notes](round-32/author/notes.md)).
- **Tutorial 10's formats on the H200 (L153, round 32):** the decision above says fp4 launches are
  rejected on the H200. At the tutorial's config `mxfp4` and `nvfp4` are (out of shared memory), but
  `mixed` (`e4m3` by `e2m1`) runs, through `DotScaledE4m3E2m1`, and is covered with `mxfp8`
  (evidence/tut10-formats).

## L38

**A kernel's namespace is pasted unchecked into the claim reader** (soundness; [lr-6a](round-6/lr-6a/report.md)). The prover's
`kernel.json` `lean.namespace` is substituted into `claim_reader.lean.in` as text, so a crafted value
changes what the reader checks. On a copy, `main : True` passed every gate check this way. Validate
the namespace as a Lean identifier (and the module name likewise) before substitution, or pass them
as data, not source text.

## L39

**The map coverage check skips `"""` lines** (gap; [lr-6a](round-6/lr-6a/report.md), [lr-6b](round-6/lr-6b/report.md)). A code line containing `"""`, and every
line between two such lines, is treated as a docstring, so an atomic there needs no map entry. Parse
the source with Python's `ast` to find docstrings and comments.

## L40

**Library modules are not replayed; `leanOptions` is unchecked** (gap; [lr-6a](round-6/lr-6a/report.md)). `leanchecker` replays
only kernel modules; it replays the library cleanly in about 2 minutes. Any `leanOptions` key is
accepted in the lakefile. Replay the library in the release check, and allow only known options.
- **Round 7:** accepted by [lr-7a](round-7/lr-7a/report.md); **reopened** by [lr-7b](round-7/lr-7b/report.md). Replay and the option check hold. Open:
  - the comment reader `_uncommented` misreads an interpolated string that holds a nested string
    (`s!"{"--"}"`, also `m!`, `f!`, `throwError`): Lean runs an `#eval` after it on the same line,
    the library hygiene flags nothing, and an `open … in` there is missed by the namespace check
    (lr-7a filed the same as a gap). Read `{…}` inside a string as nested code, as Lean's lexer
    does, or fail closed;
  - `lake build TVA` builds, and `leanchecker TVA` replays, only what `lean/TVA.lean` imports, and
    the audit accepts a library module it does not import (latent: none exists today). Require every
    library module to be imported by `lean/TVA.lean`, or build and replay each one.

  Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md).

## L41

**`.lake` folders outside `lean/` ship unscanned** (gap; [lr-6a](round-6/lr-6a/report.md)).

## L42

**A library change leaves gates stale** (gap; [lr-6a](round-6/lr-6a/report.md)). `current_gate` looks only at kernel files, so a
certificate stays GRANTED after the library changes, and the release check does not compare the
gates it recomputes with the shipped ones. Make gate freshness include the library's digests, and
make the release check fail on a mismatch.

## L43

**A rules module in a subfolder gets no rule page** (gap; [lr-6a](round-6/lr-6a/report.md)), and the audit does not notice.

## L44

**`tl.store` cache modifiers have no stated meaning** (gap; [lr-6a](round-6/lr-6a/report.md), [lr-6b](round-6/lr-6b/report.md), [coverage](round-6/coverage/matrix.md) N7). SEMANTICS.md should say
what they do (nothing to results in a well-behaved launch, if that is true, with evidence).
- **User, 2026-10-28: "we do not need to provide models for performance-only primitives (e.g.,
  L44)".** SEMANTICS states once that performance-only arguments do not change results, per
  Triton's documentation (trusted, `Compiler.Faithful`), and that models omit them. Examples: cache
  and eviction modifiers on loads and stores, `num_warps`, `maxnreg`, `launch_metadata=`, and the
  like. No model and no per-primitive evidence. Keep out of that list whatever changes results or
  ordering: `enable_fp_fusion`, `input_precision` and `max_num_imprecise_acc` (results), and
  `num_stages` (ordering, already handled by L15's rule).
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md).

## L45

**Operator and promotion semantics** (gap; [lr-6b](round-6/lr-6b/report.md), [coverage](round-6/coverage/matrix.md) N1–N3). SEMANTICS.md does not state:
- `/` on two integers (both converted to float32, int64 included: int32 `16777217/1` gives
  `16777216`);
- `and`/`or`/`not` on tensors (rejected unless `int1`);
- the dtype promotion of mixed-dtype operators;
- `tl.where` with an `int32` condition (cast as `x != 0`);
- Python's builtin `min` in a kernel, and integer `tl.minimum`.
- **User, 2026-10-28: "I think L45 should be built in the library."** All of the above is defined in
  the library: Lean definitions and SEMANTICS.md, with the promotion rules as one definition that
  every mixed-dtype operator uses, not left for provers to work out at proof time. Proof time is for
  evaluating these definitions on a kernel's values (L37's note). Round 8.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md). Follow-ups: PROVER_GUIDE still states the old comparison rule (L57), and a name
  assigned a Python number holds a tensor (L59).

## L46

**Wording** ([lr-6a](round-6/lr-6a/report.md), [lr-6b](round-6/lr-6b/report.md), [coverage](round-6/coverage/matrix.md)):
- protocol §2 overstates what the release check runs (evidence scripts are not run in it), and its
  "nothing outside exists" (only `/home` and `/tmp` are hidden; see L7);
- keyword arguments are evaluated before positional ones, not left to right;
- a conditional expression with a tensor condition runs only the taken side, not "both sides";
- the `TVA.Lang.Desc` docstring treats in-kernel descriptors as covered, but L5 is open;
- `launch_metadata=` on `@triton.jit` (tutorial 09) needs a word.

## L47

**Launching a precompiled `CompiledKernel`** (gap; [coverage](round-6/coverage/matrix.md) N6). Tutorial 02 launches a compiled kernel
directly, which skips the JIT's argument checks and specialisation. Say what a claim covers then.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md).

## L48

**Item digests rest on 64-bit non-cryptographic hashes** (soundness; [lr-7a](round-7/lr-7a/report.md)). A kernel constant's
hash in the claim reader (`constJson`'s `th`, `vh`) comes from `Expr.hash` (32 bits), `String.hash`
and `mixHash`, and no kernel file is hashed whole. A 21-minute collision search on 64 threads found
`fun x => x + 4536642005803190292` and `fun x => x + 3061831643823180776`, which get the same hash
line and so the same item digest. A prover could have one model reviewed and ship the other, and the
gate, REVIEW.md, the certificate and the release check would all agree. Hash a canonical text of
each kernel constant with sha256 (for example its printed type and value, with universe levels,
literals and binder kinds), or hash the kernel's Lean files whole into the items. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md). Follow-up: an assumption item's digest omits the assumption's statement (L68).

## L49

**`Compiler.Faithful`'s "runs alone" is stronger than claims need** (gap; [lr-7b](round-7/lr-7b/report.md)). It forbids every
other writer of global memory, on any stream, device or host, while the launch runs. That excludes
every launch that shares the GPU, and a kernel reviewer cannot check it from host code. A write to
bytes the launch is not given cannot change its outcome. Say instead that from its start to its
end, nothing else writes the bytes `init` gives the launch (the buffers of `Pre`), and keep the PDL
case as an instance. Where: `lean/TVA/Rules/Compiler.lean` (Scope 1), `docs/rules/compiler-faithful.md`,
`lean/TVA/Claim.lean`, SEMANTICS.md "Launches", `exclusions.md`. Round 8.
- **Round 8** ([notes](round-8/author/notes.md)): accepted by [lr-8a](round-8/lr-8a/report.md) and [lr-8b](round-8/lr-8b/report.md). Follow-up: a claim cannot say the rest of memory is left alone (L73).

## L50

**The map's coverage rests on unchecked `source.ranges`** (gap; [lr-7a](round-7/lr-7a/report.md)). With `output = x + y` left
out of vec-add's map, ranges that skip its line, or no ranges at all, pass, while claim.md says
"every code line of the source is mapped". Require the ranges to cover every line of each
`@triton.jit` function (found with `ast`) unless the manifest names a function as unused, and have
claim.md, protocol §7 and the reviewer guide say what is covered. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): the ranges must hold every `@triton.jit` function, except those that
  `source.unused` names. Accepted by [lr-10b](round-10/lr-10b/report.md); **reopened** by [lr-10a](round-10/lr-10a/report.md) (and filed by [lr-10b](round-10/lr-10b/report.md) as a new gap):
  `jit_functions` finds only functions under a decorator spelled `jit` or `….jit`. A JIT function
  made by a call (`h = triton.jit(_h)`) or under an aliased decorator (`from triton import jit as
  J`) runs on the H200 (`round-10/lr-10a/work/jit_spellings.py`), but it can lie outside the ranges
  and the map check still passes (lr-10a G6, lr-10b P1). A `@triton.constexpr_function` helper is
  not required either (G5). Make the rule complete: every function the source defines, at any
  depth, lies in the kernel ranges, `source.host` or `source.unused`; or fail on a source that names
  `jit` in any other way. Have REVIEWER_GUIDE ask reviewers to check that the ranges hold every
  function the launch runs. Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): accepted by [lr-13a](round-13/lr-13a/report.md) and [lr-13b](round-13/lr-13b/report.md). Follow-up: module-level values a kernel reads are in no class (L79).

## L51

**claim.md numbers lines with `splitlines()`** (gap; [lr-7a](round-7/lr-7a/report.md), [lr-7b](round-7/lr-7b/report.md)). It also breaks at form feeds,
U+2028 and other separators, which Python's tokenizer, Triton and Lean do not. One such character in
a comment shifts every later row of the map table and every later quoted declaration. Split at line
feeds only, or refuse such characters in sources and kernel Lean. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md).

## L52

**The release check does not compare review records with the ledger** (gap; [lr-7a](round-7/lr-7a/report.md)). A
`certificate.json` that names the recomputed gate and items passes with no reviewer or acceptance
behind it; a recorded filing, report or response changed after recording ships; library review
records have no recorded hash, so "earlier rounds' records are frozen" rests on nothing checked.
Require each regenerated certificate to equal the shipped one, fail when a recorded file differs from
its sha256 in the ledger, and keep a ledger for library rounds too. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): certificates are compared with the ledger, a `records` step, and a
  library ledger. Accepted by [lr-10b](round-10/lr-10b/report.md); **reopened** by [lr-10a](round-10/lr-10a/report.md) ([lr-10b](round-10/lr-10b/report.md) filed the second part as a new
  gap):
  - `tvr record library N WHO --no-lint` records a folder of any round without linting it, and
    `release_records` then skips it (lr-10a G2);
  - nothing ties a ledger line to the tool or to its history. A hand-written `accept` line grants a
    certificate that `ledger_problems` and `certificate_mismatch` both confirm (G1). An appended
    line re-freezes an edited record, and an edited sha256 hides a changed one (G3, lr-10b P2);
  - the rounds 1–7 entries froze those files as they stood in round 10, under the coordinator's
    name although the author ran them (round 2's notes had had a path spelling masked).

  Allow `--no-lint` only for rounds 1–7, and have the release check lint any later record made
  without it. Derive each kernel event from the recorded file it belongs to, and fail on an event
  that no recorded file supports. Refuse a file recorded with two sha256s and a folder recorded
  twice. Either check that the ledgers are append-only across the git history (committing at each
  recording), or say in protocol §3 and §10 and in LIBRARY_REVIEW.md that a ledger shows a file is
  unchanged since its line, not that the line is unchanged. Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): accepted by [lr-13a](round-13/lr-13a/report.md) and [lr-13b](round-13/lr-13b/report.md). Follow-ups: a reviewer's ledger lines removed with the folder left (L81), and what the
  docs can promise about ledger lines (L82).

## L53

**Gate staleness uses file times and misses files** (gap; [lr-7a](round-7/lr-7a/report.md), [lr-7b](round-7/lr-7b/report.md)). `gate_staleness` compares
mtimes of the kernel's Lean files, the map and `kernel.json`, so an edit of the source or the
write-up, or of a Lean file whose old mtime is restored, leaves the certificate GRANTED. Nothing
records which gate code made a gate, so a gate fix leaves old gates current. REVIEW.md and
`kernels/README.md` are not compared in the release check (today vec-add's pages say GRANTED while
its certificate has lapsed). Record the sha256 of every kernel file and of the gate's own code (or a
gate version) in gate.json, compare contents, and regenerate and compare the generated pages in the
release check. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md). Follow-up: inputs that staleness still misses (L69).

## L54

**`lean/lean-toolchain` is unchecked** (gap; [lr-7b](round-7/lr-7b/report.md)). The audit accepts any toolchain name, and the
release check binds whatever folder it names, following links. Require
`leanprover/lean4:v<version>`, and refuse a toolchain folder that is a link. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md).

## L55

**A `TVA.Rules.*` constant outside `lean/TVA/Rules` gets no rule page** (gap; [lr-7a](round-7/lr-7a/report.md)). A rule's page
follows its module, while the `[rule]` kind follows its name, so such a constant passes `kinds` with
no rule page in its digest (latent: none exists today). Require every `TVA.Rules.*` constant to be
declared under `lean/TVA/Rules`, or take the page from the name. Round 10.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md).

## L56

**Wording** ([lr-7a](round-7/lr-7a/report.md), [lr-7b](round-7/lr-7b/report.md)):
- protocol §2's "trusted as installed" for the toolchain: say that it supplies `Init`, `Std` and
  `Lean`, which every module reaches, and that it sits unchecked in the user's home (the author's
  notes say instead that it "can supply no module TVA imports");
- the release check prints, and writes into `release.json`, the absolute path of its folder, which
  fails the audit when kept in `work/` (and a gate run then copies it into `gate.json`); and a run
  mid-round always fails `lint` on that round's records in progress. Say both in the prompts, or
  print the folder masked;
- a `[kernel]` assumption is proved by `consistent`, not "unproved" (protocol §2, §4, §5, §9,
  REVIEWER_GUIDE §4, PROVER_GUIDE);
- protocol §6 says the launcher skips grids of `2^31` programs or more, but TVA's evidence
  (launch-outcomes C8) runs a grid of `2^32 + 1` programs; say `2^31` to `2^32` are skipped, a larger
  grid may run, and `Launch.Valid` excludes both;
- DESIGN.md's decision row still says a dependency fails "whatever its spelling"; L46 limited this
  in §2 to the steps run in the sandbox.
- **Round 10** ([notes](round-10/author/notes.md)): accepted by [lr-10a](round-10/lr-10a/report.md) and [lr-10b](round-10/lr-10b/report.md). Follow-up: one sentence of the same kind in Faithful's docstring (L74).

## L57

**PROVER_GUIDE still gives the constant rule that L45 corrected** (soundness; [lr-8a](round-8/lr-8a/report.md)).
`docs/PROVER_GUIDE.md` ("Write each Python number as the constant Triton builds") says a number
"with an operator or `tl.where` … in the operation's dtype", and an operator includes a comparison.
SEMANTICS "Constants" and evidence/promotion K1 and K3 say a comparison first makes the number a
tensor of its own type: `x < 0.1` on `float64` compares with `float32(0.1)` widened, and is true for
`x = 0.1`; `x < 2**31` on `int32` compares in `uint32`. A prover who follows the guide writes a
model that differs from the device. Say "with an arithmetic or bitwise operator, or as a value of
`tl.where`", as SEMANTICS does. Round 11 (a change to `docs/PROVER_GUIDE.md`).
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md).

## L58

**Two hint readings are stronger than Triton 3.8.0 relies on** (gap; [lr-8a](round-8/lr-8a/report.md), [lr-8b](round-8/lr-8b/report.md)). Correct launches
then have no claim.
- `multiple_of` alone on `base + tl.arange(0, B)` is false on every launch under the "every element"
  reading, yet it is the common 1-D idiom. 3.8.0 pairs `v` with the contiguity it infers for an
  arange, and the result is correct (`round-8/lr-8a/work/hint_weaker.py` W1). evidence/hint-loops'
  "promise true for" column calls such cases true.
- With a `multiple_of` on the same value, `max_contiguous` needs only `ContigAlong`, because the
  pair replaces the compiler's belief (`AxisInfo.cpp:1244-1259`). W3, W4, W5 and W7 are correct with
  a `C` below the value's own contiguity, which `ContigIs` makes undefined; W8 shows `ContigIs` is
  needed only for `max_contiguous` alone. lr-8b notes the same of `ContigIs`'s "not groups of `2·C`".

State the combined reading `ContigAlong ∧ FirstsDiv`. State, as a library fact, the contiguity the
analysis gives the shapes kernels use (`tl.arange`, plus a scalar or a value constant along the
axis), so that `FirstsDiv x k C v` with such a lower bound `C` covers the compiler's belief. Correct
evidence/hint-loops' column. Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): the readings `FirstsDiv` with a lower bound of the analysis' contiguity, and
  `ContigAlong ∧ FirstsDiv` for the pair, with a list of lower bounds. Accepted by [lr-11b](round-11/lr-11b/report.md);
  **reopened** by [lr-11a](round-11/lr-11a/report.md): the list is not a lower bound when a sub-expression carries a contiguity
  hint. `tl.max_contiguous(tl.multiple_of(a, v), C)` may set `C` below `a`'s own contiguity, the
  analysis then gives `a` contiguity `C`, and `a + s` inherits it (`AxisInfo.cpp:265-279`), while the
  list gives `tl.arange(0, B) + s` the bound `B`. In `round-11/lr-11a/work/hint_bound_subexpr.py`
  the loads narrowed from 16 to 4 bytes (no wrong result: the vectoriser takes the smaller of
  divisibility and contiguity). Say that a value carrying a `max_contiguous`, alone or in the pair,
  has the hinted `C` as its bound along each dimension, whatever operation defines it, and that the
  list applies to values without one. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): a value carrying a `max_contiguous`, alone or in the pair, has the hinted `C`
  as its bound, whatever operation defines it. Accepted by [lr-15b](round-15/lr-15b/report.md); **reopened** by [lr-15a](round-15/lr-15a/report.md): the
  library itself says the compiler does not read the mark on a loop's or a runtime `if`'s result
  (evidence/hint-multi M7–M10), loses it on `tl.split` (M1) and drops it on an inlined call (M3,
  M12, M13). For those values the stated bound is above the compiler's: in
  `round-15/lr-15a/work/probes.py`, with `r` pair-hinted `C = v = 16`, loads through
  `x + tl.multiple_of(r + s, 16)` were 16 bytes when an add defines `r` (B1), and 1 byte, as with no
  hint (B0), when `r` is a `for` loop's result (B2), a runtime `if`'s (B3), an inlined helper's (B4)
  or the first result of `tl.split` (B5). No wrong value arose. Limit the bullet to values whose
  defining operation's mark the compiler reads (not a loop, a runtime `if`, an inlined call or an
  operation with several results), give the others the list's bound without the hint, and say the
  same in TRANSCRIPTION's pitfall. Round 17.
- **Round 17** ([notes](round-17/author/notes.md)): accepted by [lr-17a](round-17/lr-17a/report.md) and [lr-17b](round-17/lr-17b/report.md). The bound is the smaller of the hinted `C` and the list's bound without the hint
  (evidence/hint-unread-round17: the canonicalizer also drops a mark when it folds an operation).

## L59

**A name assigned a Python number holds a tensor** (soundness; [lr-8b](round-8/lr-8b/report.md)). `visit_Assign`'s
`_sanitize_value` (`code-generator.py:727-750`) turns a Python number assigned to a name into a
scalar tensor of its own dtype, unless the target is annotated `: tl.constexpr`. This covers a
literal, the value of a `tl.constexpr` and each element of a tuple; a literal passed directly to a
helper stays a compile-time value there (`:1358-1365`). SEMANTICS's rules for Python numbers
("Constants", "Operators and promotion") and compile-time values ("Order", the compile-time `if`),
and TRANSCRIPTION's rows built on them, read as if the name still held a Python value. On the device
(`round-8/lr-8b/work/t_assign_literal.out`): `e = 1e-7; x * e` with a `bfloat16` `x` gives `1e-7·x`
in `float32`, where the literal in place gives `0`; after `F = False`, `if F and tl.atomic_add(c, 1)
> 0:` ran the atomic in 64 of 64 programs; `a = -7; b = 2; a // b` stores `-3`, where the literal in
place gives `-4`. Tutorials 05, 06, 08 and 10 rely on this for literal-initialised loop variables.
State it in SEMANTICS "How the front end evaluates the source" and "Constants" and in the `Operand`
docstring of `TVA.Lang.Promote`, and add a TRANSCRIPTION pitfall. Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md). Follow-up: the annotation rule it ends with is not exact (L84).

## L60

**A hint on one result of an operation with several results is a hint on all of them**
(soundness; [lr-8b](round-8/lr-8b/report.md)). `tt.split` has no AxisInfo visitor, so each result gets
`getPessimisticValueState`, which reads the hint off the defining operation
(`AxisInfo.cpp:1234-1259, 1318-1340`). By the same code this covers a tuple `tl.reduce` or
`tl.associative_scan`, `tl.max`/`tl.min` with `return_indices`, and a helper Triton does not inline
that returns a tuple; only `tt.split` was run. On the device (`round-8/lr-8b/work/t_hint_multiresult.out`),
with `a, b = tl.split(z)`, the true hint `tl.multiple_of(a, 4)` made the loads through `b` vectorise,
and they faulted. The model the library prescribes, a check on `a`, is well behaved, so `Faithful`
is false for it. Say in SEMANTICS "Compiler hints" and the `TVA.Lang.Ops` hint section that such a
hint promises the same of each result (the model checks each), or exclude such hints. No pinned
tutorial hints such a value. Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md). Wording follow-up in L86.

## L61

**Wording** ([lr-8a](round-8/lr-8a/report.md), [lr-8b](round-8/lr-8b/report.md)):
- SEMANTICS "Index arithmetic": "arithmetic on them stays `int32` … until the source casts it" is
  false when an operand is a runtime `int64` or `uint64` scalar, which promotion widens to; and
  "below `2^31` in magnitude" leaves out `-2^31` ("Scalar arguments" says `[-2^31, 2^31)`);
- SEMANTICS "A runtime `if`": "otherwise the kernel is rejected" reads as if a name defined in one
  branch only were rejected; it is rejected only when used after the `if`
  (`round-8/lr-8a/work/if_names.py`, evidence/runtime-if R5);
- builtin `min`/`max` of compile-time values reject NaN and `-0.0` (`core.py:3884-3886`);
- SEMANTICS "One launch" and `Faithful`'s "Which launch" speak only of memory, but hooks also change
  the launch's arguments (tutorials 06 and 09 set a host `TensorDescriptor`'s `block_shape` in a
  config's `pre_hook`): the claimed launch has the arguments as the last hook leaves them;
- TRANSCRIPTION "Masked-off lanes": "gives unspecified values, never `0`" reads as "is never 0";
  say "not necessarily `0`";
- PROVER_GUIDE still calls L1 and L2 open and tells provers to write every hint, but a hint on a
  parameter or an induction variable is now dropped.

Round 11 (PROVER_GUIDE is a change outside the round's files).
- **Round 11** ([notes](round-11/author/notes.md)): items 1 and 3–6 fixed. Accepted by [lr-11b](round-11/lr-11b/report.md); **reopened** by [lr-11a](round-11/lr-11a/report.md) for item 2,
  at soundness level: the rewritten "A runtime `if`" text is false for a name whose value before
  the `if` is not a Triton value. `visit_then_else_blocks` skips such a live-in
  (`code_generator.py:858-861`). In `round-11/lr-11a/work/if_compile_time_livein.py`, after
  `y = None; if n > 0: y = tl.load(p)`, `y` is still `None` after the `if` (I2); after
  `d = tl.float16; if n > 0: d = tl.float32`, `d` is still `float16`, and `1 + 2^-12` was stored as
  `1.0` (I6); and a `None` that both branches assign merges and runs (I3), where the text says the
  kernel is rejected. A model that follows the text differs from the device. State it (a name
  holding `None` or a dtype before a runtime `if` merges only when both branches define it, and
  otherwise keeps its value), or exclude it under L88. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md). The exclusion is conservative for tuples and descriptors, which the front end merges.

## L62

**A `[rule]` field can state a device fact with constants the kernel chose** (soundness; [lr-9a](round-9/lr-9a/report.md)).
The gate's `kinds` check refuses only constants in the kernel's own namespace, so a field such as
`MathAcc P .exp .binary32 Set.univ (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ))` passes it, `Platform.ref`
satisfies it, so `consistent` passes too, and it is false on the device (`tl.exp` reaches
`2^-17.99` relative error, evidence/num-math-f32). The guidance leads there: PROVER_GUIDE's table
gives the shapes `Rules.Num.MathAcc` and `Rules.Num.DotAcc`, not the named facts, says the constants
of `Platform.ref` are the bundle's (`:199`), and never mentions `Platform.sm90ref`. REVIEWER_GUIDE §4
(`:121`) says the gate ensures the statement adds no constant of the kernel's own. Have `kinds`
accept in a `[rule]` field only a stated fact applied to `P` and `RULE_ARGS` (a `TVA.Rules.Num.Sm90`
fact, `DivFull`, `DivRn .binary64`, a `…Total` rule with enumeration arguments, `Compiler.Faithful`),
never the shapes themselves (`MathAcc`, `MathAt`, `DivAcc`, `CvtAcc`, `CvtRn`, `DotAcc`,
`DotScaledAcc`, `AtomicAddAcc`). Correct both guides: cite the named fact, and prove `consistent`
with `Platform.sm90ref`. Round 13 (with the fact names as L65's rewrite leaves them).
- **Round 13** ([notes](round-13/author/notes.md)): a `[rule]` field passes only as a fact the library states applied to `P`
  (`Compiler.Faithful`, a `Platform → Prop` fact, a `…Total` rule, `DivRn` on `binary64`). Accepted
  by [lr-13b](round-13/lr-13b/report.md); **reopened** by [lr-13a](round-13/lr-13a/report.md):
  - `Compiler.Faithful` is accepted with any arguments after `P`. In `round-13/lr-13a/work/probes.py`
    F1, a claim with `Inst := Bool`, a model whose instance `true` never ends and whose `false` does
    nothing, and `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model false) (launch d) (K d)`
    passes `shape`, `axioms`, `usage`, `namespace` and `kinds`: `main` is proved from the idle
    model, while the launched instance's model never ends (F2: `model e` for another bound `e`, and
    `model (Classical.choose h)`, pass too). Accept `Faithful` only as `P (model x) (launch x) (K x)`
    for one variable `x` the statement binds, and say so in §5 and §8;
  - (filed by lr-13a as a new gap) facts are accepted by their type and name (any `TVA.Rules`
    constant of type `Platform → Prop`, any `TVA.Rules.Num` name ending in `Total`), not by the
    library's list, so a helper such as `def DivRnF32 := DivRn P .binary32` (false on the H200) would
    pass as a stated fact. Accept a head only if `docs/rules/num.md`'s "Facts the library states"
    names it, and say in the library author's prompt that a fact is citable once listed there.

  The check depends on names (`binary64`, the `Total` suffix); a rename fails closed. Round 16.
- **Round 16** ([notes](round-16/author/notes.md)): accepted by [lr-16a](round-16/lr-16a/report.md) and [lr-16b](round-16/lr-16b/report.md). Follow-ups next to it: the claim reader finds `K` and `P` by position (L94), and the
  fact list reads every table of its section (L95).

## L63

**A `[rule]` can carry a constant of the kernel's own** (soundness; [lr-10a](round-10/lr-10a/report.md)). `kinds_problems` counts
as the kernel's own only names that start with its namespace, and the reader's namespace check
exempts every `isInternal` name. In `round-10/lr-10a/work/kinds_probes.py`, complete kernels with a
real `consistent` have a `[rule]` `DivAcc … tol …` whose `tol = 2^(-40)` the kernel defines: a public
`def tol` fails `kinds` (the control), but `private def tol`, `def _root_.Kernels._tol` and
`def _root_._tol` all pass `axioms`, `usage`, `namespace` and `kinds`. So protocol §5's "a kernel
cannot slip its own constants into a library rule" is false. Count as the kernel's own every
in-scope constant not rooted at `TVA`, other than `RULE_ARGS`, and exempt from the namespace check
only the names Lean generates, not a user-written `_x`. Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): accepted by [lr-13a](round-13/lr-13a/report.md) and [lr-13b](round-13/lr-13b/report.md).

## L64

**fp8 conversion of values beyond `maxVal` is not stated** (gap; [lr-9a](round-9/lr-9a/report.md)). The fp8 facts hold only
for `|x| ≤ maxVal` (`CvtRn`), so a claim can say nothing about larger values. In-scope tutorials
convert unbounded `float32` accumulators into fp8: tutorial 08's `grouped_matmul_tma_kernel` with
`FP8`, tutorial 09's four kernels (`float8e4nv`), tutorial 06's `acc.to(float8e5)` and tutorial 10's
`output_type == 2`. evidence/num-cvt8 already shows on every input that finite values beyond
`maxVal` and `±∞` give `±maxVal`, and NaN gives `0x7f`. State a named fact for each of the six pairs
on non-NaN inputs, citing num-cvt8, and add it to the table in `docs/rules/num.md`. Round 14 (stated
on L65's float layer).
- **Coverage round 10** ([coverage](round-10/coverage/matrix.md)): also tutorial 09's `matmul_kernel_descriptor_persistent`
  (`.to(dtype)` into `float8e4nv`), and tutorial 06's `_attn_fwd`, whose `acc` can exceed `maxVal`
  (`p` is rounded to fp8 after `l_i` is summed): 8 rows in all.
- **Round 14** ([notes](round-14/author/notes.md)): the shape `CvtSat` and six facts `CvtSatF32E4m3` … `CvtSatBF16E5m2`, with
  `Platform.ref` saturating on `.rn`. **Reopened by [lr-14a](round-14/lr-14a/report.md) and [lr-14b](round-14/lr-14b/report.md)** (soundness): the facts are true of a
  conversion at run time (both re-ran float-layer's six `sat1` tests on every input; NaN gives `0x7f`,
  `±∞` gives `±maxVal`), but the batch says they cover fp8 constants too ("folded by the compiler only
  when the conversion is exact … so the facts cover it too", `docs/rules/num.md` "Conversions into
  fp8"), and SEMANTICS "Constants" models an fp8 constant as `FloatV.litF32` then `tl.cvt8`
  (`Operand.constIn?`). `scalar_constant` builds it as `arith.truncf` of the `float32` constant
  (`triton-3.8.0-semantic.py:601-613`), which MLIR folds whenever the conversion is lossless, and `±∞`
  into `float8e5` is: `tl.full([16], float('-inf'), tl.float8e5)` and `tl.where(m, x8, float('-inf'))`
  store `0xfc` (`round-14/lr-14a/work/fp8_const.out`, `fp8_store.out`), while the model with
  `CvtSatF32E5m2` gives `0xfb` (`work/Fp8Const.lean`; [lr-14b](round-14/lr-14b/report.md)'s U10 proves it), so a wrong claim is
  provable. An inexact fp8 constant is rejected on the host; a stored Python number and `other=` go
  through the device's conversion and saturate, as modelled. Model an fp8 constant as the exact,
  non-saturating conversion of its `float32` (`FloatV.cvt (litF32 c)`), with a host rejection when it
  is inexact, in SEMANTICS, `TVA.Lang.Scalar` and `constIn?`, and correct num.md and the notes'
  account. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md). Wording follow-up (the fp8 prose of SEMANTICS and PROVER_GUIDE) in L114.

## L65

**TVA's float layer is another project's code** (independence; [lr-9a](round-9/lr-9a/report.md), coordinator). lr-9a found that
docstrings in `lean/TVA/Math/Float` cite device validation that TVA does not have: `F32.lean`'s
"Validation record (ported from v1)" (nvcc 13.2, 2026-10-20), and `Convert.lean`'s and
`Format.lean`'s "conformance records `docs/evidence/conformance/cvt.md`" and "findings F1, F2".
`tvr audit` still prints `independent`, since the text names no other project. The coordinator then
compared TVA's Lean with the project those records come from. `lean/TVA/Math` (Float and Bytes,
about 3,700 lines) is that project's layer with the namespace renamed, and
`lean/TVA/Lemmas/SoftmaxMax.lean` and `Fused.lean` reuse some of its proofs. Those two files and
`lean/TVA/Lemmas/Num.lean` are written in its namespace. Lean re-checks every proof, so no result is
trusted, but the code and the lemmas are reused, which protocol §2 (Independence) forbids. lr-9a's
own tests found the definitions true of Triton 3.8.0 on the H200 (`round-9/lr-9a/work/cvt_std.py`,
`exact_ops.py`, `cvt_lean.py`: 0 differences).
- **User decision, 2026-10-28: rewrite independently.** An author who has never seen that project
  writes TVA's own float layer from IEEE 754 and Mathlib: the formats, rounding, conversions, the
  meaning of the exact operations, the error lemmas the library needs, and the byte encoding. The
  copied modules are deleted, the rest of the library is adapted to the new layer, and what
  `SoftmaxMax`, `Fused` and `Num` provide is proved afresh on it. The definitions' agreement with
  the device stands on TVA's evidence in `evidence/` (lr-9a's scripts can serve), and the
  docstrings cite only that.
- The audit part, refusing a cited `docs/…` or `evidence/…` path that does not exist, is in L70.

Round 12.
- **Round 12** ([notes](round-12/author/notes.md)): the float layer rewritten from the standards alone (`TVA.Math.Defs`,
  `Encoding`, `Rounding`, `Arith`, `LittleEndian`; lemma modules `FloatErr`, `FusedErr`, `FloatMax`),
  with evidence/float-layer. Accepted by [lr-12a](round-12/lr-12a/report.md) and [lr-12b](round-12/lr-12b/report.md), who read it against IEEE 754-2019 and the
  OCP formats and re-ran 142 and 88 of the evidence's tests on the device, with tests of their own.
  Follow-ups: a Python NaN constant's pattern (L98), and the layer's wording (L99, L100). [lr-12b](round-12/lr-12b/report.md)'s
  gap on `replay_err` is under L34.

## L66

**Wording** ([lr-9a](round-9/lr-9a/report.md), [lr-9b](round-9/lr-9b/report.md)):
- evidence/num-dot (conclusion 4, and "(tiny inputs)" in its table) and the round-9 notes attribute
  `DotF32Ieee`'s ratio `0.499` to the absolute term on tiny inputs, "by construction". lr-9a finds
  it is the relative term on the `allones` input, `(K−1)/(2(K+1))`; lr-9b gives `K·2^-150` against
  `(K+1)·2^-149`. Re-derive it and state one account. Part 2's "worst input kind" column names the
  kind with the largest `err/u`, not the largest ratio;
- `docs/rules/num.md` ("`tl.dot`") calls `(K+1)·2^-23` "the classical bound for `K` roundings"; it
  is about twice `γ_K ≈ K·2^-24`;
- evidence/num-math-f32 (conclusion 3) and `docs/rules/num.md` say exp2's bound is "reached
  exactly"; it is approached from below and cannot be reached;
- `docs/rules/num.md` "What the library does not state": "any approximate function on `float16`,
  `bfloat16`" names functions Triton rejects (`triton-3.8.0-math.py`: `fp32`/`fp64` only), and the
  `dot_scaled` entry does not name `e2m1` by `e2m1`, which compiles and runs on sm_90 at a small
  tile (`round-9/lr-9a/work/probes.out` part 2);
- the `DotF16`/`DotF8` docstrings and `docs/rules/num.md` say a chunk is "within `j·2^-23` of their
  magnitudes' sum", meaning that many times the sum;
- `docs/rules/num.md` ("Float `atomic_add`") calls the lowering of `tl.atomic_add(p, 0)` with
  `relaxed` or `acquire` a "plain load"; it is a strong load at the atomic's scope
  (`ld.global.gpu.relaxed.b32`, evidence/num-atomic-add);
- `float32` `-∞` has two names, `Rules.Num.negInfF32` and `Softmax.negInf32`, equal only by
  unfolding: use one, or add a lemma equating them.

Round 14.
- **Round 14** ([notes](round-14/author/notes.md)): accepted by [lr-14a](round-14/lr-14a/report.md) and [lr-14b](round-14/lr-14b/report.md). Wording follow-up (`num-dot`'s "four digits") in L101.

## L67

**`DivRn P .binary64` has no device evidence** (gap; [lr-9b](round-9/lr-9b/report.md)). `docs/rules/num.md`'s table gives "the
lowering" as its evidence; TVA has no run of `float64` `/` on the device. lr-9b's
`round-9/lr-9b/work/div_f64.py` found 0 of 2.7·10^8 pairs differing from IEEE division, so the fact
looks true. Add an evidence folder (random patterns, subnormal dividends and quotients, divisors
near 1), cited by the table and by the `DivRn` docstring. Round 14.
- **Round 14** ([notes](round-14/author/notes.md)): evidence/num-div-f64 (452,986,128 pairs in 24 families against `FloatV.divRn`).
  Accepted by [lr-14a](round-14/lr-14a/report.md) and [lr-14b](round-14/lr-14b/report.md), each also comparing against the host's IEEE division (218,103,808 and
  671 million pairs of their own): no difference.

## L68

**An assumption item's digest does not hold the assumption's statement** (soundness; [lr-10a](round-10/lr-10a/report.md)). In
`tools/claim_reader.lean.in` (`fitems`) and `item_digests`, no item's cone reaches `Assumptions`,
so only the set of TVA and kernel constants a field mentions, plus docstrings, enters its digest.
`round-10/lr-10a/work/digest_probes.py` makes three edits, and each keeps every item digest: a
`[rule]` `DivAcc P .binary32 (2^126) (2^(-22)) (2^(-148))` changed to `2^(-40)`; the same field
wrapped in `∀ _ : Nat`; and a `[kernel]` `model 0 = model 0` changed to `model 0 ≤ model 0 + 1`. So
`tvr check` alone restores the certificate, and REVISED cannot name the changed item. Protocol §4
("the statement of field `f`") and LIBRARY_REVIEW.md ("no two different meanings share a digest")
are false for assumption items. Have the reader write the canonical text of each field's type, with
`K` and `P` bound, into that field's item. Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): accepted by [lr-13a](round-13/lr-13a/report.md) and [lr-13b](round-13/lr-13b/report.md).

## L69

**A gate does not go stale on every input that changes what the kernel's Lean elaborates to**
(gap; [lr-10a](round-10/lr-10a/report.md)). After `lean/lean-toolchain` changes to `v4.36.0`, which the audit accepts, the gate is
not stale and the certificate's gate condition still holds (`round-10/lr-10a/work/gate_probes.out`
G4), though every item digest holds Lean's version. By the code, the same holds for
`lean/lakefile.toml` (`leanOptions`) and for an imported library module that holds no constant of
any cone (notation, macros). Only the release check catches these. Record in gate.json the sha256
of these files and of every library file the kernel's modules import. Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): accepted by [lr-13a](round-13/lr-13a/report.md) and [lr-13b](round-13/lr-13b/report.md).

## L70

**The audit misses two things** (gap; [lr-10a](round-10/lr-10a/report.md), [lr-9a](round-9/lr-9a/report.md)):
- a folder of `kernels/` with no `kernel.json` is not scanned: `kernels/notes/readme.md` holding a
  path into a sibling of TVA is not named, though it would ship (`round-10/lr-10a/work/gate_probes_g8.out`).
  Scan every file under `kernels/`, or refuse a folder there that is not a kernel;
- a cited `docs/…` or `evidence/…` path that does not exist passes, as L65's citations did. Refuse
  it.

Round 13.
- **Round 13** ([notes](round-13/author/notes.md)): accepted by [lr-13a](round-13/lr-13a/report.md) and [lr-13b](round-13/lr-13b/report.md). A path component in capitals only (`docs/DESIGN`) is read as a placeholder (L82).

## L71

**The gate passes a claim whose conditions cover no instance** (gap; [lr-10a](round-10/lr-10a/report.md)). `Consistent` bounds
only the assumptions. A kernel with `Inst.ok _ := False` and a real `consistent` passes `axioms`,
`usage`, `namespace`, `kinds` and the reader's shape examples (`round-10/lr-10a/work/kinds_probes.py`
V0). The reviewer guide asks whether the conditions are satisfiable, but an unsatisfiable
combination can be invisible to a reader. Protocol §8, DESIGN.md "No vacuous claims" and
`TVA.Claim`'s module docstring ("so the claim is not vacuous") say otherwise. Require, for example,
`nonempty : ∃ d init, Inst.ok d ∧ Pre d init` in the claim and check it in the reader, or say in
those places that this is the reviewer's judgment. Round 11 (`TVA.Claim`; a reader check is a
change to `tools/` outside the round's files).
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md). The witness may be degenerate (an empty grid); whether the conditions cover real use
  stays the reviewer's judgment, and the docs say so.

## L72

**`Compiler.Faithful`'s scope 4 misses the options of a precompiled kernel** (gap; [lr-10b](round-10/lr-10b/report.md)). It
excludes compile options given "as a launch keyword" or "in an autotune `triton.Config`"
(`lean/TVA/Rules/Compiler.lean`, docs/rules/compiler-faithful.md, exclusions.md row 39, SEMANTICS
"Launches"). A `CompiledKernel` (scope 6, tutorial 02) gets its options at `warmup(…)`
(`triton-3.8.0-jit.py:362-363`) or at `triton.compile(…, options=…)` (`triton-3.8.0-compiler.py:226`),
and its launch takes none. In `round-10/lr-10b/work/warmup_options.py`,
`k.warmup(…, enable_reflect_ftz=False)` and then a launch with no keyword gave
`tl.math.exp2(-130) = 2^-130` instead of the default's `0`, with scope 4 as written satisfied. Say
"given to the call that compiles the kernel (a JIT launch, `warmup`, `triton.compile`'s options) or
in an autotune `triton.Config`", and have the reviewer's host check name `warmup` (REVIEWER_GUIDE,
outside the round's files). Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md). Follow-ups: `@triton.heuristics` (L78) and `triton.compile` of an IR file (L77).

## L73

**A claim cannot say that the launch leaves the rest of memory alone** (gap; [lr-10b](round-10/lr-10b/report.md)). Round 8's
"Memory" paragraph of `Faithful` (`lean/TVA/Rules/Compiler.lean`, the rule page) puts what the
device does to bytes `init` does not give outside the outcome, and `Outcome.ok fin` holds only
`init`'s bytes. So no certificate excludes a launch that writes past its buffers into other
tensors, which a host that composes launches needs excluded. Trusting the compiler implies it: the
model touches no other byte, and only scope 5's descriptor scratch is written outside. State in
`Faithful` that a valid, well-behaved launch in scope writes no byte the host allocated outside
those `init` gives, its descriptor scratch aside, and say in `TVA.Claim` what a claim then implies
about other memory; or say plainly there that a claim carries no such frame. Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md). Both reviewers widened the evidence (11 and 12 more kernels): no byte outside the
  written buffers changed, and only `tl.make_tensor_descriptor` uses global scratch.

## L74

**Wording of `Faithful`'s scope** ([lr-10b](round-10/lr-10b/report.md)):
- Faithful's docstring, its rule page and `TVA.Claim` speak of "the device" but never say which.
  exclusions.md row 45 limits TVA to the H200 (sm_90), and the model's defaults are sm_90's
  (`DotCfg`'s `tf32`, the fp8 `maxImpreciseAcc = 2^30`). Name the target as a scope condition;
- Faithful's docstring still gives the reason for `2^31` as "Triton's launcher skips larger grids
  without an error"; say that `2^31` to `2^32` are skipped and a larger grid may run, as L56 made
  protocol §6 say;
- scope 6's list of argument classes does not name host tensor descriptors, whose specialisation is
  their dtype and block shape (`triton-3.8.0-specialize.cc:165-226`).

Round 11.
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md).

## L75

**SEMANTICS never says what a call of one of Python's builtins in a kernel gives** (gap; [coverage](round-10/coverage/matrix.md)).
The front end calls `float`, `isinstance`, `list`, `getattr` and `hasattr` as Python's own on the
unwrapped arguments (`code-generator.py:357-365, 1444-1448`), and `int` and `len` through its static
executor (`:1639-1659`, a `constexpr` result), at compile time, emitting no operation. `range`,
`min`, `max` and `print` it maps to Triton's own, which SEMANTICS treats. Yet "Expressions with no
effect" counts every call as possibly effectful, and "Order" does not count such a result among the
compile-time values. In `round-10/coverage/work/probes.out` P2, `tl.static_assert(float("inf") >
1e308)` and `tl.static_assert(isinstance(out_ptr, tl.tensor))` compile, `-float('inf')` is stored as
`0xFF800000`, and only the branch `isinstance` picks is emitted: a host `TensorDescriptor` argument
is a `tl.tensor_descriptor`, and a pointer is not. Tutorial 02's `softmax_kernel` and tutorial 06's
`_maybe_make_tensor_desc` and `_attn_fwd` need it. State it in SEMANTICS "How the front end
evaluates the source", with guidance in TRANSCRIPTION. Round 11 (added after the round started).
- **Round 11** ([notes](round-11/author/notes.md)): accepted by [lr-11a](round-11/lr-11a/report.md) and [lr-11b](round-11/lr-11b/report.md).

## L76

**A host descriptor can round `float32` loads to tf32** (soundness; the round-11 author, in their
final report). Triton 3.8.0's host `TensorDescriptor` (`triton/tools/tensor_descriptor.py`) has a
field `round_f32_to_tf32`, `False` by default and allowed only for `float32` tensors, which the
driver passes into the TMA descriptor (`triton/backends/driver.py`). With it set, the author saw
`desc.load` return `1.0` for a stored `1.0000009537`. `tl.desc_load`'s model returns the stored
values, so a model that follows the library differs from the device. No pinned tutorial sets it.
Either exclude it (exclusions.md, `Faithful`'s scope and the reviewer's host check: a host
descriptor with `round_f32_to_tf32=True`), or model it with TVA's evidence of the rounding. Round 15
(the coordinator proposes the exclusion: nothing in scope uses it).
- Moved to round 15 (a semantics batch; the coordinator proposes the exclusion).
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md).

## L77

**`triton.compile` of an IR file replaces the kernel's source** (gap; [lr-11a](round-11/lr-11a/report.md), [lr-11b](round-11/lr-11b/report.md)). Scopes 4 and 6 of
`Compiler.Faithful` (`lean/TVA/Rules/Compiler.lean`, the rule page, SEMANTICS "Launches") present any
`CompiledKernel` from `triton.compile` as the kernel's code, but `triton.compile(src)` with `src` a
file path compiles that IR file (`IRSource`, `primitives/triton-3.8.0-compiler.py:87-112, 233-240`:
`.ttir`, `.ttgir`, `.ptx`, …), as `ir_override` does. Say that a `CompiledKernel` from
`triton.compile` is covered only when `src` is an `ASTSource` of the kernel's `@triton.jit`
function, and add the IR-file route to exclusions.md's options row and to the reviewer's host
check. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md).

## L78

**`@triton.heuristics` sets compile options** (soundness; [lr-11b](round-11/lr-11b/report.md)). `Heuristics.run`
(`primitives/triton-3.8.0-autotuner.py:480-483`) sets `kwargs[name]` for each heuristic, replacing a
launch keyword of the same name, and the JIT's `run` parses options out of those keywords
(`jit.py:726-741`). In `round-11/lr-11b/work/heur_options.py`, with no launch keyword, a heuristic
`enable_reflect_ftz: False` gave `exp2(-130) = 2^-130` (H1); and a heuristic `enable_fp_fusion: True`
overrode a launch's `enable_fp_fusion=False`, so the device fused and `x*y+z` gave `2^-24` (H5). A
model that follows `Launch.fusion`'s docstring ("the source's or the launch's value") gives the
unfused `0`, so `Faithful` fails for a valid, well-behaved model. `debug`, `num_ctas`, `launch_pdl`
and the excluded options reach the compile the same way. No in-scope tutorial uses
`@triton.heuristics`. Either exclude it under L88, or say that a field of `Launch` is the value the
JIT receives (the launch keyword, an autotune config's, or a heuristic's, which replaces both) and
name heuristics in scope 4, the `Launch` docstrings, SEMANTICS, exclusions.md and the reviewer's
checks. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md). Wording follow-up on `Launch.fusion`'s docstring in L92.

## L79

**The map's coverage misses values a JIT function reads from module level** (gap; [lr-13a](round-13/lr-13a/report.md)). The front
end resolves a name in a kernel to a module-level `tl.constexpr(…)`, a dtype, a namedtuple or a
module's attribute at compile time (`triton-3.8.0-code-generator.py:387-420`), and the value is part
of the compiled kernel (`triton-3.8.0-jit.py:532-536, 761-766`). Such a line is in no function, so
nothing classes it: in `round-13/lr-13a/work/globals.py` a kernel reading `SCALE = tl.constexpr(3)` and
`OUT_T = tl.float16` from lines outside its kernel lines passes `map_problems`, and on the H200 stores
`x·3` rounded to `float16`. Class every module-level code line outside a function as kernel or host
lines, and have the reviewer judge that every module-level value the launch reads is in the kernel
lines (the meaning of such names is L80's). Round 16.
- **Round 16** ([notes](round-16/author/notes.md)): every code line no function spans must lie in the kernel or the host lines.
  Accepted by [lr-16b](round-16/lr-16b/report.md); **reopened** by [lr-16a](round-16/lr-16a/report.md): a function in `source.unused` is exempt from classing,
  but its `def` statement runs at import, evaluating its decorators, defaults and annotations, which
  can bind a module-level name the launch reads. In `round-16/lr-16a/work/probes_src.py`, an unused
  `_unused_helper(x=(SCALE := tl.constexpr(3)))`, or `_unused_helper(f=(double := triple))` with the
  `@triton.jit` helper `double` that the kernel calls, rebinds the name; `map_problems` passes both
  (Part U), and on the H200 both kernels store `3·x` (`work/unused_gpu.out`). Class the header of
  every `source.unused` function (its decorators and `def` lines up to the body) as module code that
  must lie in the kernel or the host lines; say in protocol §7 and the guides that a `def` evaluates
  its decorators, defaults and annotations at import; and admit in SEMANTICS' supported subset a
  module-level name only as its own `def` or `import` binds it.
- **Wording** ([lr-16b](round-16/lr-16b/report.md)): the guidance's examples of module-level values to class and map (REVIEWER_GUIDE
  §1, PROVER_GUIDE §1, the kernel reviewer's and prover's prompts: "a `tl.constexpr(…)` global, a
  dtype, a namedtuple") are forms the supported subset excludes. Say that the module-level values a
  launch may read are the imports of `triton` and `triton.language as tl` and the module's JIT
  functions, and that a kernel reading any other is `UNSUPPORTED` (L88).

Round 18.
- **User, 2026-10-28** (under L36): the unused-function header case is a malicious prover's; a stated
  limit and a reviewer check suffice (told to round 18's author).
- **Round 18** ([notes](round-18/author/notes.md)): accepted by [lr-18a](round-18/lr-18a/report.md) and [lr-18b](round-18/lr-18b/report.md). Follow-up: Triton and the kernel module rebound by other host code (L104).

## L80

**Compile-time evaluation SEMANTICS does not state** (gap; [lr-13b](round-13/lr-13b/report.md), [lr-13a](round-13/lr-13a/report.md)). Kernels and the guidance
(L50's REVIEWER_GUIDE §1, claim.md §2, the kernel reviewer's prompt) rely on:
- **constexpr and plain Python functions:** Python runs a `@triton.constexpr_function` at compile
  time (`triton-3.8.0-jit.py:1153-1169`), and any plain Python function reached through a module
  (`code_generator.py:1444-1446`); the result is a compile-time value (`round-13/lr-13b/work/fe_probes.py`
  F1), and arithmetic on a tensor argument inside one is rejected on the host (F2);
- **helper defaults:** a helper parameter's default is bound and kept as a compile-time value
  (`call_JitFunction`, `:1358-1365`): with `e=1e-7`, `h(x)` computes as `h(x, 1e-7)`, a `bfloat16`
  product of 0 (D1, D2);
- **module globals:** only some are visible (`:383-418`): a `tl.constexpr(…)` global is a
  compile-time value (G1), a plain number is rejected on the host (G2).

Under L88, state those in the supported subset in "How the front end evaluates the source", and
exclude the rest. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md). Wording follow-up (a helper's string default) in L92.

## L81

**A reviewer's ledger lines can be removed while their folder stays** (gap; [lr-13a](round-13/lr-13a/report.md)). Every kernel
event must follow from a recorded file, but no filed folder must have its events. In
`round-13/lr-13a/work/probes.py` Part R (a scratch copy of vec-add with a hand-made passing gate),
`r-two` is assigned and files a blocking `SPEC_TOO_WEAK`; with `r-two`'s `assign`, `file` and `raise`
lines removed and the rest renumbered, the records step names nothing, `tvr lint` prints `regular`,
and the certificate is granted, with `review/round-2/r-two/filing.json` still in the tree. The
library's ledger has the counterpart (an unrecorded role folder of an earlier round fails). By the
user's rule (L36), the cheap complete check: the records step and the certificate fail on a
`review/round-N/<who>/` folder of a reviewer no `assign` names, or of an earlier round no event
records, and `tvr lint` refuses a reviewer not assigned. Round 16.
- **Round 16** ([notes](round-16/author/notes.md)): the records step and the certificate fail on a folder of a reviewer no
  `assign` names, or of an earlier round no event records; `tvr start` and `tvr lint` refuse an
  unassigned reviewer. Accepted by [lr-16b](round-16/lr-16b/report.md); **reopened** by [lr-16a](round-16/lr-16a/report.md): a folder counts as recorded by any
  event naming one of its files, and only folders an event names as `dir` are frozen.
  1. vec-add's ledger has `file`/`response` events from before `dir` existed (2, 5, 7): their
     folders are not frozen (`review/round-1/r-vecadd/work/` holds four files no event records). In
     a scratch copy at events 1–9 with a hand-made passing gate, r-vecadd files `filing-2.json` with
     a blocking `SPEC_TOO_WEAK`; with that filing's `file` and `raise` lines removed, the records
     step names nothing and the certificate is granted (`round-16/lr-16a/work/probes.out` Part R,
     L1).
  2. On a frozen folder, a coordinator's `tvr note … --files` of a filing and its report accounts
     for both, so removing the filing's lines again grants the certificate (N).

  Freeze the folder of every `file` and `response` event's path whether or not the line names
  `dir`, and let only a role's own `file`/`response` events account for a role folder, never a
  `note`. Give a supported way to bring an existing ledger such as vec-add's (its four round-1 work
  files) to a passing state, without editing ledger lines by hand. Round 18.
- **Round 18** ([notes](round-18/author/notes.md)): accepted by [lr-18a](round-18/lr-18a/report.md) and [lr-18b](round-18/lr-18b/report.md). The coordinator ran `tvr freeze vec_add_f32` after the merge (four round-1 work
  files). Wording follow-up in L108.

## L82

**Wording of what the tool can promise** ([lr-13a](round-13/lr-13a/report.md), [lr-13b](round-13/lr-13b/report.md)):
- LIBRARY_REVIEW.md ("Rounds", step 4), protocol §10 and §3, and `ledger_digests` say ledger lines
  "rest on the git history, or on the ledger's digest that the release check keeps". The user
  decided that the lines are trusted as the coordinator keeps them, with no commit at each
  recording (L36); the library ledger has never been committed, and no step reads an earlier
  `release.json` back. Say that the lines are trusted as the coordinator keeps them, and that an
  edit shows only against a copy kept elsewhere, compared by hand. (The round-13 notes' request to
  commit after each recording is declined by that decision.);
- §3: `filing_events` re-derives old events with the tool's current `KINDS` defaults,
  `ESCALATE_AFTER` and escalation memo, so changing any of them makes every older ledger fail the
  records step; say so, or derive those from what the event recorded;
- §8 and §9: gate.json, its snapshot and claim.md are as the last `tvr check` wrote them in its
  caller's environment (`TVR_LAKE`, elan's toolchain, the local `.lake/packages`) or as anyone edited
  them since; only the release check recomputes them in a clean sandbox. Say so, as §3 does for
  ledger lines;
- `citation_problems` reads a path component in capitals only (`docs/DESIGN`) as a placeholder.

Round 16.
- **Round 16** ([notes](round-16/author/notes.md)): accepted by [lr-16a](round-16/lr-16a/report.md) and [lr-16b](round-16/lr-16b/report.md). Follow-up: evidence/tool-round16's output predates the protocol's fix (L97).

## L83

**TRANSCRIPTION's `tl.where(keep, x, 0.0)` row is false for an integer `x`** (soundness; [lr-13b](round-13/lr-13b/report.md)).
TRANSCRIPTION.md §2's row says "`Tensor.where_` with the constant in `x`'s dtype". That holds only
for a float `x`, as in tutorial 04. For an integer `x`, SEMANTICS "Constants" and `whereTyping?` give
`float32`: on the device the result is `fp32`, and stored back into an `int32` buffer, 16777217
became 16777216 (`round-13/lr-13b/work/fe_probes.py` W1; W2 is the float control). A model that
follows the row keeps the integer values. Limit the row to a float `x`, and give the integer case,
or say "in the dtype `whereTyping?` gives". Round 15.
- **User, 2026-10-28: "tl.where guidance should be fixed anyway".** `tl.where` is in the supported
  subset (L88), so this is fixed, not excluded: the guidance gives the right dtype for every `x` the
  subset admits, integer and float, resting on SEMANTICS "Constants" and `whereTyping?`.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md).

## L84

**The `: tl.constexpr` annotation rule is not exact** (soundness; [lr-13b](round-13/lr-13b/report.md)). SEMANTICS "A name assigned a
Python number holds a tensor" ends with "unless the target is annotated `: tl.constexpr`"
(`code_generator.py:694-760` keeps `pending_annotations`). In `round-13/lr-13b/work/fe_probes.py`:
- after a bare `e: tl.constexpr` line, the next assignment keeps the Python value though its own
  statement has no annotation: `e = 1e-7` then `x * e` with a `bfloat16` `x` gives 0 (C4; C5 without
  the bare line gives `float32`);
- a later assignment to a name annotated earlier, or to a `tl.constexpr` parameter, makes a tensor:
  `c: tl.constexpr = -7; c += 0; c // 2` gives `-3` (C2, against `-4`), and after `BLOCK = BLOCK * 1`,
  `tl.arange(0, BLOCK)` is rejected on the host (C1);
- an annotation other than `tl.constexpr` is ignored: `a: tl.int64 = 2**31 - 1` is an `int32`, and
  `a + 1` wraps to `-2^31` (C6).

The in-scope tutorials use only `x: tl.constexpr = …` on one statement. Under L88, state that form
exactly (and the rule for every other assignment), exclude the others or state them, and add the
pitfall to TRANSCRIPTION. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md). Wording follow-up (an annotation other than `tl.constexpr` evaluates its value twice) in
  L92.

## L85

**A single-argument `range(e)` evaluates `e` twice** (soundness; [lr-13b](round-13/lr-13b/report.md)). `code_generator.py:1206`
visits every argument, and `:1242` visits the single argument again for the upper bound, so an
operation in `e` runs twice in every program: `range(tl.atomic_add(c, 1) * 0)` ran 16 atomics in 8
programs (`round-13/lr-13b/work/fe_probes.py` L4), against 8 for `range(0, …)` (L5) and `tl.range(…)`
(L6). SEMANTICS "Order" and "Loops" and TRANSCRIPTION's `range` row imply one evaluation. The in-scope
tutorials pass a plain name. State the double evaluation (one sentence), or, under L88, restrict a
single-argument `range` to an argument without effects. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md). Follow-up: `range`'s keyword and fourth arguments (L90).

## L86

**Wording** ([lr-11a](round-11/lr-11a/report.md), [lr-13b](round-13/lr-13b/report.md)):
- SEMANTICS "Compiler hints: an operation with several results" and `TVA.Lang.Ops`'s hint section:
  "the same promise about each result" is undefined for a float result or one of another rank (a
  tuple reduce, a loop carrying an accumulator or a scalar beside a block). Say "each integer or
  pointer result of the hinted value's rank", and that a hint after a loop or a runtime `if` is
  never read (evidence/hint-multi M7–M10);
- SEMANTICS "Index arithmetic": an "unsigned" operand promotes `int32`, but `uint8` and `uint16` do
  not; say "a wider type, or an unsigned type at least 32 bits wide";
- TRANSCRIPTION §3 "Python's builtins" leaves out `range`, then says any other builtin is rejected;
- §3 "A scalar argument's type": below `-2^31` is also `int64`, and from `2^63` `uint64`;
- the §2 row "`lo, hi = 0, N`": `hi` holds `N`'s own type (`int64`, `uint32` or `uint64` for a large
  `N`), not always `int32`;
- the §2 row "`x * w + b`": the addition the product feeds is `tl.mul_add`;
- §5: since L50 the gate also checks that every function is classed;
- SEMANTICS' opening ("What runs on the device is what it evaluates"): the loop body's dry visit is
  evaluated and its operations discarded (an atomic in a zero-trip body ran in no program; only the
  marks of its hints stay);
- SEMANTICS' `tl.load` does not say that `other` is converted to the pointer's dtype
  (`semantic.py:1043-1045`), through `float32` for `float16` (`work/fe_probes.py` O1).

Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md).

## L87

**`tl.cumsum`'s `dtype=` argument is not stated** (gap; [lr-11a](round-11/lr-11a/report.md)). SEMANTICS (Scans) gives `tl.cumsum`'s
promotions but not `dtype=`: Triton first widens `bfloat16` to `float32`, then converts the input to
`dtype` (`standard.py:334-343`). State it as for `tl.sum`, or exclude it under L88: no pinned
tutorial uses it. Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): accepted by [lr-15a](round-15/lr-15a/report.md) and [lr-15b](round-15/lr-15b/report.md).

## L88

**A supported subset of the Python front end** (user, 2026-10-28, choosing "a supported subset"
over "model every form"). Most new soundness findings came from corners of how Triton's front end
evaluates Python, which the in-scope tutorials barely use.
- **SEMANTICS lists the Python forms TVA gives meaning to:** those the in-scope tutorial kernels use
  (`tutorials/inventory.json`, L37), plus generalisations that are cheap to state. Every other form
  is excluded (exclusions.md), and the kernel reviewer checks that the kernel stays inside the list
  (REVIEWER_GUIDE). A finding about a form outside the list closes by excluding it, not by modelling
  it. A kernel that uses an excluded form has no claim until the library adds the form.
- **To settle under it now:** L61 (a name holding `None` or a dtype before a runtime `if`), L76
  (`round_f32_to_tf32`), L78 (`@triton.heuristics`), L80 (constexpr functions, plain Python calls,
  helper defaults, module globals), L84 (annotations other than `x: tl.constexpr = …`), L85 (a
  single-argument `range` with an effectful argument), L87 (`tl.cumsum`'s `dtype=`): for each, state
  it when it is in the subset or cheap, and otherwise exclude it.
- A syntactic check of the subset in the gate is for a later tool batch, if it is cheap and
  complete (L36).

Round 15.
- **Round 15** ([notes](round-15/author/notes.md)): SEMANTICS "The supported subset", with the forms L61, L76, L78, L80, L84,
  L85 and L87 state or exclude. Accepted by [lr-15b](round-15/lr-15b/report.md), whose AST scans of the 23 in-scope functions find
  only listed forms (follow-ups: L89, L90); **reopened** by [lr-15a](round-15/lr-15a/report.md): the list misses `tl.cdiv`, which
  the in-scope kernels of tutorials 03, 06, 08, 09 and 10 call 23 times. The subset admits only the
  `tl.*` functions SEMANTICS and `TVA.Lang` give meaning to, and neither gives `tl.cdiv` one; it is
  not ceiling division but Triton's `@jit` body `(x + (div - 1)) // div` (`standard.py:34-43`): on
  the H200, `tl.cdiv(n, 128)` with `int32` `n = 2^31 - 1` stored `-16777215`, and `tl.cdiv(-5, 4)`
  stored `0` (`round-15/lr-15a/work/probes.py` K1, K2). State `tl.cdiv` (that body, with the
  operator rules it implies) and list it, or add a `TVA.tl.cdiv`. Also check the list once more
  for any other `tl.*` call the in-scope kernels make that has no meaning in SEMANTICS or
  `TVA.Lang`. Round 17.
- **Round 17** ([notes](round-17/author/notes.md)): accepted by [lr-17a](round-17/lr-17a/report.md) and [lr-17b](round-17/lr-17b/report.md). `tl.cdiv` is stated as Triton's body (evidence/cdiv-round17), and evidence/subset-calls-round17
  finds no other call of the in-scope kernels without a meaning. Wording follow-up in L108.

## L89

**Code after a `return` taken through a compile-time `if` is still compiled** (soundness; [lr-15b](round-15/lr-15b/report.md)).
SEMANTICS "`return`" says only that code after a `return` at the top level is not compiled; the
natural reading is that a compile-time `if` whose taken branch returns ends the function. The front
end stops only at the statements after the `return` in its own statement list
(`code_generator.py:508-517`), and `visit_Return` opens an unreachable block (`:544-554`): the code
after the enclosing `if`, and every later `tl.static_range` copy, is visited into it. It never runs,
but its compile-time effects stay: a hint marks the operation that defined its argument, even one
defined before the `return`, and a `tl.static_assert` is evaluated. In
`round-15/lr-15b/work/dead_after_return.py`, a pair hint after `if FLAG: return` (`FLAG` true) made an
earlier load `ld.global.v2.b32` and read wrong values in 137 of 256 lanes (D7; D8 without the hint
was correct), faulted with a misaligned address (D1), did the same through `tl.static_range` (D9),
and a `tl.static_assert(False)` after the `if` rejected the kernel (D5). A model that drops that code
is well behaved, so `Faithful` is false for it. State it in SEMANTICS "`return`" and TRANSCRIPTION
§3/§4 (the rest of the function is visited but does not run; its hints are checks where their values
are defined; its `tl.static_assert`s are evaluated), or exclude statements after such an `if` or
copy under L88. No in-scope tutorial has one. Round 17.
- **User, 2026-10-28: "L89, L91 can be excluded".** Excluded, not stated (told to round 17's author).
- **Round 17** ([notes](round-17/author/notes.md)): accepted by [lr-17a](round-17/lr-17a/report.md) and [lr-17b](round-17/lr-17b/report.md). Excluded (user's decision).

## L90

**`range`'s keyword and fourth arguments** (gap; [lr-15b](round-15/lr-15b/report.md)). The subset's "Statements" item admits `for name
in range(…)` with any arguments, and SEMANTICS gives meaning only to one, two or three positional
arguments. `visit_For` evaluates the keywords (`code_generator.py:1207`) but uses only
`iter_args[0..2]` for Python's `range` (`:1237-1243`): in `round-15/lr-15b/work/fe_edges.py` (`n = 8`),
`range(0, n, step=2)` and `range(n, step=2)` ran 8 trips (K1, K2), `range(0, n, 2, 99)` ran 4 (K4),
and `tl.range(0, n, step=2)` ran 4 (K3). Make the item "Python's `range` with one, two or three
positional arguments", and add the keyword and fourth-argument forms to the subset's excluded forms
and to exclusions.md. Round 17.
- **Round 17** ([notes](round-17/author/notes.md)): accepted by [lr-17a](round-17/lr-17a/report.md) and [lr-17b](round-17/lr-17b/report.md).

## L91

**CUDA graph capture** (soundness; [lr-16b](round-16/lr-16b/report.md)). `TVA.Claim`, `Launch.Valid`'s docstring,
compiler-faithful.md ("What it says"), SEMANTICS "Launches" and protocol §6 say that a valid launch
that returns without an error runs its whole grid, from `init`, the memory when it starts. Under
stream capture (`torch.cuda.graph`, `triton.testing.do_bench_cudagraph`), `k[grid](…)` returns
without an error, compiling first if needed, and runs no program; the programs run at each
`graph.replay()` with the captured arguments, on the memory at the replay. In
`round-16/lr-16b/work/graph_capture.py` the output kept its sentinel after the captured call, held
`x + 1` after the first replay and `11` after the host set `x = 10` and replayed; a captured `n = 16`
wrote 16 elements whatever `n` was later. If the captured call is the launch, `Faithful` makes "ok,
memory unchanged" a model outcome, false for any kernel that writes; if each replay is, nothing says
so. The coordinator proposes excluding capture (exclusions.md, `Faithful`'s scope, `TVA.Claim`,
SEMANTICS "One launch", `Launch.Valid`'s docstring, the reviewer's host check): a claim covers no
call made while its stream is capturing and no replay. No in-scope tutorial kernel's host captures
(tutorial 11, which benchmarks with it, is excluded). Stating capture instead (a replay is a launch
of the recorded instance from the memory at the replay) needs TVA's own evidence. Round 17.
- **User, 2026-10-28: "L89, L91 can be excluded".** Excluded, not stated (told to round 17's author).
- **Round 17** ([notes](round-17/author/notes.md)): excluded as `Faithful`'s scope 9 (user's decision). Accepted by [lr-17a](round-17/lr-17a/report.md);
  **reopened** by [lr-17b](round-17/lr-17b/report.md): the exclusion is clear, but the reason it gives ("a call made while a stream
  capture is in progress … runs no program", in `TVA.Claim`, `Launch.Valid`, the rule page, scope 9,
  SEMANTICS, exclusions.md and protocol §6) is false: in `round-17/lr-17b/work/probes.py` G1–G3 (all
  three capture modes), a call on another stream during a capture ran at once, and the replay ran
  nothing; only a call on the capturing stream is recorded. Keep the exclusion as it is, and state
  the reason truly. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md).

## L92

**Wording in the front end and transcription** ([lr-15a](round-15/lr-15a/report.md), [lr-15b](round-15/lr-15b/report.md)):
- SEMANTICS "A runtime `if`" merges "a name new to the `if` that both branches define", but a name
  both branches bind with `name: tl.constexpr = e` is not merged (`visit_AnnAssign` writes `lscope`,
  not `local_defs`, `code_generator.py:704-710`), so a use after the `if` is rejected on the host
  (`round-15/lr-15a/work/probes.py` A2). Say that such a name belongs to its branch;
- `TVA.Launch.fusion`'s docstring gives "the source's or the launch's value", but scope 4 also admits
  `enable_fp_fusion` in an autotune `triton.Config` (`autotuner.py:365-375`): name the chosen config's
  value;
- exclusions.md row 52 and SEMANTICS "Annotated names": an annotation other than `tl.constexpr` is
  not only ignored; its value is evaluated twice (`:698`, `:712`, `:757`;
  `round-15/lr-15b/work/fe_edges.py` A1: 16 atomics in 8 programs);
- exclusions.md row 54 and SEMANTICS "Names after a loop": the placeholder sentence holds for `range`
  and `tl.range`, not `tl.static_range`, after which the variable holds its last value (S1) and a
  name the body defines is defined (S2); say which of these uses the subset admits;
- SEMANTICS "Helpers are inlined": the front end also visits each default as an assignment
  (`:634-649`), so a default it cannot make a tensor (a string; an integer beyond 64 bits) rejects
  the helper even when the call passes the argument (H1);
- TRANSCRIPTION §2's `tl.multiple_of` row and §3's hint pitfall take the bound "for `x`'s shape";
  SEMANTICS' bound depends on how `x` is computed: say so, citing "The contiguity the analysis
  infers";
- `docs/rules/compiler-faithful.md`, the reviewer's checklist, third item: "and the arguments those
  the last hook leaves" is garbled.

Round 17.
- **Round 17** ([notes](round-17/author/notes.md)): accepted by [lr-17a](round-17/lr-17a/report.md) and [lr-17b](round-17/lr-17b/report.md).

## L93

**The guides' lists of the host conditions are incomplete** ([lr-16b](round-16/lr-16b/report.md)). REVIEWER_GUIDE §1 ("The host
code must stay within … scope") leaves out scope 5 (an allocator whose buffer holds none of the
launch's bytes, when the kernel builds descriptors), scope 6's condition that a `CompiledKernel` gets
arguments of its specialisation, and that an autotuned launch's `init` and arguments are those after
the last hook. PROVER_GUIDE §2 also leaves out `@triton.heuristics`, the `ASTSource` of
`triton.compile`, the H200 target and `round_f32_to_tf32`. compiler-faithful.md's "How a bundle cites
it" lists them all. Give the full list in both guides, or say "among them" and point to the page's
checklist. Round 17.
- **Round 17** ([notes](round-17/author/notes.md)): accepted by [lr-17a](round-17/lr-17a/report.md) and [lr-17b](round-17/lr-17b/report.md).

## L94

**The claim reader finds `K` and `P` by position** (soundness; [lr-16a](round-16/lr-16a/report.md)). `mentions 0`/`mentions 1` and
the `platform` and `K` flags read `xs[0]` and `xs[1]` of the field projection's binders
(`tools/claim_reader.lean.in:395-431`), and nothing requires `Assumptions` to have only those two
parameters. With `structure Assumptions [Inhabited Unit] [Inhabited Bool] (K : Claim.Behaviour Inst)
(P : Platform)`, the flags read the instances: a field `[kernel] spec : Claim.Meets Inst.ok Pre Post
K` (`main := h.spec`), and a field `[kernel] div : ∀ x y r, P.div .binary32 x y r → r = x` (a device
fact false on the H200, satisfied in `consistent` by a platform built for it), pass every check
through `kinds` (`round-16/lr-16a/work/probes.out` Part K). Protocol §5 and REVIEWER_GUIDE §4 tell the
reviewer the gate guarantees what it does not. Have the reader fail unless `Assumptions` has exactly
the two parameters `K : Claim.Behaviour Inst` and `P : Platform`, and say so in §8's `shape`. Round
18.
- **User, 2026-10-28: "the reviewer can challenge malicious provers and can access the entire
  proof".** A stated limit and a reviewer check suffice (under L36; told to round 18's author).
- **Round 18** ([notes](round-18/author/notes.md)): accepted by [lr-18a](round-18/lr-18a/report.md) and [lr-18b](round-18/lr-18b/report.md). Follow-up: an `Assumptions` that is not a structure (L106).

## L95

**The fact list is read from every table of its section** (gap; [lr-16a](round-16/lr-16a/report.md), [lr-16b](round-16/lr-16b/report.md)). `stated_facts`
(`tools/tvr.py`) takes every table row from "Facts the library states" to the next `#` or `##`
heading, so a second table there, even under a `### … (not citable)` subsection, makes its
first-column names citable alone: with `MathAcc` in such a table, lr-9a's field `MathAcc P .exp
.binary32 Set.univ (2^-24) (2^-150)` passes `kinds` (`round-16/lr-16a/work/probes.out` Part L,
`round-16/lr-16b/work/rule_probes.py` Part C). Protocol §5 says "the first column of its table". The
page has one table today. Read only the section's first table (stop at its first line that is not a
table row, or at any heading), have `fact_page_problems` name a section with more than one table,
and say so in §5 and the author's prompt. Round 18.
- **Round 18** ([notes](round-18/author/notes.md)): only the section's first table is read, and the audit names a second.
  Accepted by [lr-18a](round-18/lr-18a/report.md), who filed the rest as a new gap; **reopened** by [lr-18b](round-18/lr-18b/report.md): a setext heading (text
  over a `---` line) makes a "not citable" table the citable one with the audit silent, and a table
  inside an HTML comment or a fenced code block is read as the facts table
  (`round-18/lr-18a/work/facts.py`, and [lr-18b](round-18/lr-18b/report.md)'s report). Treat setext headings as headings, and have
  the gate read only a table outside comments and fences (or the audit refuse comments and fences in
  the section). Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): the section is read through an allowlist, and every form of tool-round20 Part L
  makes the gate accept no fact. Accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md); **reopened** by [lr-20c](round-20/lr-20c/report.md): what
  comes before the heading is judged by a fence tracker that does not follow CommonMark, and cells are
  split unlike GFM, so the gate still reads 38 facts, with the audit silent, from a page whose section
  renders as code or not at all (`round-20/lr-20c/work/facts_probe.py`): a four-backtick fence with a
  three-backtick line inside, a fence with a ```` ```python ```` line inside, a line ```` ```a`b ````
  before a real fence, `<script>` or `<pre>` before the heading closed after the next `##` heading; a
  row starting `||` makes `DivRn P .binary32` citable; inline HTML passes. Fix in `facts_section` and
  `stated_facts`: track fences as CommonMark does (a closer uses the opener's character, is at least
  as long, and has nothing after it but spaces; a backtick opener whose info string holds a backtick
  opens nothing); make it a problem when a line before the heading, outside a fence, starts with `<`
  (after up to three spaces), and when the section holds `<` followed by a letter, `/`, `!` or `?`;
  split a row as GFM does, or refuse a row starting `||`; and make protocol §2 say exactly this.
  Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L96

**The release check's `pages` step reads the live tree** (gap; [lr-16a](round-16/lr-16a/report.md)). It compares the pages the
sandbox regenerates with `ROOT`'s (`page_mismatch(ROOT, copy)`), not the copy's, since the sandbox
overwrites the copy's pages. A page fixed in the tree during the run passes though the copy shipped it
stale, and one regenerated in the tree during the run fails (the coordinator's run of the merged tree
named vec-add's `claim.md`, which its prover's `tvr check` rewrote meanwhile). Protocol §2 says each
page "must be the one regenerated from the shipped files". Read the copy's generated pages (and gate
times) before the sandbox runs, and compare with those. Round 18.
- **Round 18** ([notes](round-18/author/notes.md)): accepted by [lr-18a](round-18/lr-18a/report.md) and [lr-18b](round-18/lr-18b/report.md).

## L97

**evidence/tool-round16's `out.txt` predates the protocol's fix** ([lr-16a](round-16/lr-16a/report.md), [lr-16b](round-16/lr-16b/report.md)). Part C, line 139,
names a `docs/DESIGN` citation in REVIEW_PROTOCOL.md, while the README says the tree cites no missing
file; the protocol changed after the run, and on the merged tree the check finds none. Part M's
vec-add line also changes now that vec-add's prover has classed the module lines. Re-run the
evidence so that its output matches its README. Round 18.
- **Round 18** ([notes](round-18/author/notes.md)): accepted by [lr-18a](round-18/lr-18a/report.md) and [lr-18b](round-18/lr-18b/report.md). Follow-up: the round-18 evidence no longer re-runs after the freeze (L108).

## L98

**A Python NaN constant does not have the pattern the library states** (soundness; [lr-12a](round-12/lr-12a/report.md)).
`FloatV.litF32`, `litF16` and `litBF16` (`lean/TVA/Lang/Scalar.lean`, used by `Operand.constIn?` and
SEMANTICS "Constants", which says the constant "takes the value Triton builds for it") give
`0x7fffffff`, `0x7fff`, `0x7fff` for `float('nan')` and `-float('nan')` alike
(`round-12/lr-12a/work/litnan.out`). Triton 3.8.0 builds `0x7fc00000`, `0x7e00`, `0x7fc0`, and
`0xffc00000`, `0xfe00`, `0xffc0` for the negated NaN (`work/nan_probe.out`, `tl.full`). These
constants are exact, not up to NaN payload, so a model storing `tl.full(…, float('nan'),
tl.float32)` claims bits the device does not write; evidence/literals counts any NaN equal to any
other, so it does not support the patterns either. Give the three functions the device's patterns
(sign kept, the quiet NaN with the double's top payload bits), or state in SEMANTICS and `constIn?`
that a NaN constant is some NaN; and compare NaN bit for bit, with both signs, in
evidence/literals. Round 19.
- **Round 19** ([notes](round-19/author/notes.md)): `litF32`, `litF16`, `litBF16` give the device's NaN patterns (new `FloatV.qNaN`,
  `nanNarrow`, `hostF32`), evidence/literals compares bit for bit, and a `tl.constexpr` argument
  holding a NaN is excluded (Triton's cache keys write every NaN as `nan`). Accepted by [lr-19b](round-19/lr-19b/report.md);
  **reopened** by [lr-19a](round-19/lr-19a/report.md): a NaN passed to a helper is not "covered, with its exact pattern": Triton
  names each helper copy by the `repr` of its compile-time arguments, and every NaN's is `nan`, so
  `h(o, float('nan')); h(o+1, -float('nan'))` stored `0x7fc00000` twice, in either order, through
  constexpr names, by keyword, in `bfloat16` and with `tl.where` (`round-19/lr-19a/work/helper_nan.py`).
  This is one case of L103 (compile-time values Triton's caches and helper names do not tell apart);
  settle it there. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md). Settled as a case of L103; the reviewer's check for L103 is reopened.

## L99

**Wording of the float layer and its evidence** ([lr-12a](round-12/lr-12a/report.md), [lr-12b](round-12/lr-12b/report.md)):
- `toRat` of `±∞` is said to be unused or meaningless (`TVA.Math.Encoding`'s module docstring, the
  round-12 notes, `FloatV.toRat`, `Format.toRat`), but `maxNum`, `minNum`, `cmp`, `tl.maximum`,
  `FloatV.sortKey`, `le_iff`, `lt_iff`, `eq_iff`, `toRat_max`, `negInf_toRat_le` and `Tree.eval_maxRel`
  order the infinities through it. Say it is `±2^(emax+1)`, beyond `maxVal`, which orders them; that
  only a NaN's value is meaningless; and that every error bound also concludes finiteness;
- `FloatV.nan` is "the canonical NaN NVIDIA arithmetic produces"; not for `float64`, where invalid
  operations give `0xfff8000000000000` and a NaN operand keeps its payload. Say "the library's
  canonical NaN", true of the device's `float32`/`float16`/`bfloat16` arithmetic and fp8 conversions;
- `TVA.Lemmas.FusedErr`: `u ≤ 1/16` holds for "every format the library covers", but `float8e5` has
  `u = 1/8` and `e2m1` `1/4`; name the formats. `FloatV.ofRat`: "`±∞` on overflow", but it gives NaN in
  `e4m3` and `±maxVal` in `e2m1`; give the overflow by `Specials`;
- `TVA.Lemmas.FloatMax`'s module docstring lists `le_max_left`, `le_max_right`, which resolve to
  Mathlib's lattice lemmas; name what it provides;
- `docs/rules/num.md` "Exact operations need no rule" names `tl.sqrt_rn` among operations whose error
  bounds are proved; none bounds `FloatV.sqrtRn`. Add `sqrtRn_err` (with a `roundRat`
  characterisation of `sqrtMag`), or say it has no error lemma yet;
- evidence/float-layer's README and the round-12 notes say "the 7 that differ are the
  non-saturating variants"; one, `A.cvt.f32.e5m2.rz.sat1.sample`, is saturating, and its "listed"
  differences cover 4 of 64 chunks. Say six non-saturating `"rtne"` tests and one saturating `"rtz"`
  test, and give the argument for "exactly";
- evidence/float-layer's `build.sh` calls `lean` and `leanc` from `PATH` and never uses the `BIN` it
  computes; run both steps through the toolchain `lean/lean-toolchain` names, or say so in the README.

Round 19.
- **Round 19** ([notes](round-19/author/notes.md)): accepted by [lr-19a](round-19/lr-19a/report.md) and [lr-19b](round-19/lr-19b/report.md). Wording follow-ups in L108.

## L100

**Proof-kit docstrings that do not match the code** ([lr-12b](round-12/lr-12b/report.md)):
- `lean/TVA/Lemmas/Access.lean:12` cites `accBytes_of` (it is `Buf.Addrs.accBytes`), and `:17`
  promises "a row of a matrix, a 2-D tile" in `TVA.Lemmas.Block`, which has only the 1-D pattern;
- `lean/TVA/Lemmas/Shared.lean:36` cites `safeA_atomic1` (it is `SafeA.atomic1`), `:32` says
  "bitwise" updates are determined with no lemma for `atomic_and`/`or`/`xor`, and `SafeA.atomic1`,
  `SafeA.atomic` and `safe_atomic_buf` say they apply to `tl.atomic_add`, which needs an `env` step
  first;
- `safe_upToNaN` (`lean/TVA/Lemmas/Ops.lean`) lists `cvt`, but `tl.cvt` is a `pick` around
  `upToNaN` (`safe_cvt` is the rule); `safe_dot_add`'s docstring lists two hypotheses its statement
  does not take; `safe_sum_bcast_mul` says "each element" chooses its form, while the statement
  chooses one for the whole tensor;
- `lean/TVA/Lemmas/Int.lean`'s docstring says "casts keep their value" (no lemma about `IntV.cast`)
  and that `lt_eq`, `le_eq` hold "(always)" (signed only);
- `lean/TVA/Lemmas/Claim.lean`'s docstring says "both lemmas need validity" and omits `hdebug`;
- `docs/LIBRARY.md`'s `Independent` row says "final memory determined" (only as far as each
  program's `Q` says).

(`foldl_max_toInt`'s docstring, also filed, went with round 14, which removed the lemma.) Round 19.
- **Round 19** ([notes](round-19/author/notes.md)): accepted by [lr-19a](round-19/lr-19a/report.md) and [lr-19b](round-19/lr-19b/report.md). Wording follow-up in L108.

## L101

**Wording from round 14's reviews** ([lr-14a](round-14/lr-14a/report.md), [lr-14b](round-14/lr-14b/report.md)):
- `evidence/num-cvt8/README.md` says it shows the results beyond the range "on every input", but
  `run.py` converts `±∞` and NaN only from `float32`, and only the canonical NaN; the statement is
  true (`round-14/lr-14a/work/fp8_nan.out`: every `float16`/`bfloat16` NaN, 139,262 `float32` NaN
  patterns, `±∞` of each format). Say which inputs the folder ran, or add those inputs to `run.py`;
- `evidence/num-dot/README.md`: the largest ratio "equals `(K−1)/(2(K+1))` to four digits", but at
  `K = 512` `0.498035` prints as `0.4980` against `0.4981`. Say "within `2·10^-5` of, and below";
- `AtomicAddF32` covers every `sem` and `scope`, while `evidence/num-atomic-add` runs only `acq_rel`
  and `relaxed` at `gpu`: add the other combinations (all 12 bit-identical in
  `round-14/lr-14a/work/atomic_sem.out`), and say in its README that the fact covers them all;
- `docs/LIBRARY.md`'s `Encoding` row promises "the constants of each format", but `float8e5` and
  `float8e4nv` have only `maxVal`, no `u` or `eta`, which `FloatV.cvtSat_rn_err`'s `|c|·u + η` needs
  (`round-14/lr-14b/work/Usability.lean` U6 derives them by `rfl`): add them, or say so;
- `replay_err`'s docstring names "the operands `W`"; the statement's operands are
  `L.filterMap Prod.snd`.

Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md).

## L102

**A helper's returns and a runtime conditional expression are typed by the front end** (soundness;
[lr-17a](round-17/lr-17a/report.md), [lr-17b](round-17/lr-17b/report.md)). SEMANTICS says a helper call "means its body" and a conditional expression "means
the runtime `if`", which reads as keeping a Python number a Python number. When a helper's visited
`return`s differ (or a path falls off the end), `decide_return_type` gives the call the common type
(`code_generator.py:556-626`); a conditional expression with a runtime condition is a runtime `if`
whose two values are typed the same way (`visit_IfExp`). Python numbers become scalar tensors: on
the H200, `return -7` / `return -9` behind a runtime `if` gave `h(c) // 2 = -3, -4` (Python: `-4, -5`;
`round-17/lr-17a/work/helper_returns.py`); `(-1 if c > 0 else -3) // 2` stored 0, and `x *
_h_f(c)` on `bfloat16` was computed in `float32`, differing on 13,076 of 65,280 inputs
(`round-17/lr-17b/work/probes.py`); a helper `if c > 0: return 5` returned 5 on every path. Both forms
are in the subset; no in-scope kernel uses either. State the rule in SEMANTICS (and a TRANSCRIPTION
pitfall), or exclude both forms under L88. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): the runtime conditional expression is stated; helper `return` forms nobody uses
  are excluded. Accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md); **reopened** by [lr-20b](round-20/lr-20b/report.md): the admitted form
  "`return`s that all give one compile-time value, equal values of one Python type" admits `(1, 2)`
  and `(1.0, 2)` (equal tuples), but the front end gives the first visited `return` on every path:
  with `if c > 0: return (1, 2)` then `return (1.0, 2)` and a runtime `c`, `x * th(c)[0]` stored
  `16777217` for `c = -5`, where the `return` taken gives `16777216` (`round-20/lr-20b/work/probes.out`
  T1–T3). Fix: make the form "the same `None`, `int`, `float` (not a NaN), `bool`, string or dtype, of
  one Python type; for a tuple, element by element", or exclude tuples from it, in SEMANTICS "The
  supported subset" (`return`) and "A helper's value", exclusions.md's row, TRANSCRIPTION's pitfall
  "A helper's value", and REVIEWER_GUIDE §1's list. Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L103

**Compile-time values Triton does not tell apart** (soundness; [lr-17a](round-17/lr-17a/report.md), [lr-19a](round-19/lr-19a/report.md)). A launch may run a kernel
compiled for another compile-time value:
- in one process, `1`, `1.0` and `True` share a cache entry, as do `0` and `0.0`: `C=1.0` after `C=1`
  ran the integer kernel and stored 16777217, not 16777216 (`round-19/lr-19a/work/cache_num.py`);
- Triton's disk cache keys a `tl.constexpr` argument by `str`: `"1"` and `1`, `"None"` and `None`,
  `"2.5"` and `2.5` share one compiled kernel, across processes and within one, in either order
  (`round-17/lr-17a/work/constexpr_cache.py`, `round-19/lr-19a/work/cache_str.py`);
- each helper copy is named by the `repr` of its compile-time arguments, so two NaNs (L98) and two
  strings that differ only by `[`/`]` against `_` run the first call's body
  (`round-19/lr-19a/work/helper_nan.py`, `helper_str.py`).

Settle the family at once, not value by value: either narrow the compile-time argument values the
subset admits (of a kernel and of a helper) to those Triton keys apart (for example integers, and
dtypes; no float, `bool`, string or `None` where another admitted value of the same parameter has the
same `str` or `repr`), or state a scope condition of `Faithful` that the compiled code a launch runs
is the one compiled for its own compile-time values (a fresh cache, or no earlier launch in the
cache's lifetime whose values collide), with the reviewer's host check. Round 19's NaN exclusion
becomes a case of the rule. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): new scope 11 of `Faithful` (a launch whose compile-time values share Triton's key
  with an earlier launch is outside the rule), checked by the kernel reviewer; two calls of one
  helper whose arguments give one copy name are outside the subset. Accepted by [lr-20d](round-20/lr-20d/report.md) and [lr-20e](round-20/lr-20e/report.md);
  **reopened** by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md) and [lr-20f](round-20/lr-20f/report.md): the rule's condition is right, but the check the reviewer is
  given is false. REVIEWER_GUIDE item 11 ("A fresh cache directory, or values Triton keys apart,
  meets it"), the rule page's checklist item 11 and scope 11's last sentence (Lean docstring and rule
  page, "on a fresh cache, or with each compile-time parameter given values that Triton keys apart"):
  - the process keeps its own cache: in one process on a fresh directory, even one set just before
    the launch, `C = 1.0` after `C = 1` ran the integer code (`16777217`, not `16777216`;
    evidence/compile-keys K1, `round-20/lr-20a/work/probes.out` P2a, `round-20/lr-20f/work/fresh_cache.py`);
  - values are keyed as one tuple, not per parameter: `("a-b", "c")` then `("a", "b-c")` stored `31`,
    not `13` (`round-20/lr-20a/work/probes.out` P2b), though each parameter's two values are apart;
  - the lists name only `1`/`1.0`/`True` and `0`/`0.0`/`-0.0`/`False`, but every pair `==` equates
    collides: `C = 2` then `C = 2.0`, and `3`, `3.0` ([lr-20b](round-20/lr-20b/report.md)'s probe K, [lr-20d](round-20/lr-20d/report.md)'s `work/keys_extra.py` Y1);
  - an autotuner's benchmark runs are earlier launches, even in a fresh process ([lr-20b](round-20/lr-20b/report.md)).

  Fix, in REVIEWER_GUIDE item 11, the checklist item 11, PROVER_GUIDE (11) and scope 11's last
  sentence: lead with the rules (values Python's `==` equates, such as `2` and `2.0`, `True` and `1`,
  `0.0` and `-0.0`; values with the same `repr` in the process, every NaN; on disk, the `str`s of all
  compile-time values joined with `-`, so `1` and `"1"`, and a `-` moved between strings), compared
  over all compile-time parameters as one tuple; and replace the shortcut with: "met when the claimed
  launch is the kernel's first launch in its process (warm-ups and autotuner runs included) on a fresh
  cache directory, or when every launch of the kernel in the process and through its cache directory
  gave values Triton keys apart from the claimed launch's; a fresh cache directory alone does not meet
  it". Also add to item 11's list an integer `1` and `None` passed to a parameter without annotation
  (`specialize.cc` makes both compile-time, [lr-20a](round-20/lr-20a/report.md)). Round 21.
- **User decision, 2026-10-29** (under L36): the kernel is assumed new and Triton's caches are not
  verified. Round 21 states scope 11 as that assumption, with no reviewer check, instead of the fix
  above; the helper copy-name rule of the subset stays.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate. Stated as part (B) of `Faithful`'s scope, by the user's decisions of 2026-10-29
  (under L36).

## L104

**Triton and the kernel's module may be changed by other host code** (soundness; [lr-18b](round-18/lr-18b/report.md), [lr-18a](round-18/lr-18a/report.md),
[lr-17a](round-17/lr-17a/report.md)). Host code in another module, before the first launch, changed what the launch computes: on
the H200, `kmod.double = kmod.triple` gave `3·x` where the model says `2·x`
(`round-18/lr-18a/work/rebind_ext.py`, `round-18/lr-18b/work/host_mutation.py`), `tl.exp = tl.log` made
`tl.exp` compute a logarithm, `tl.arange = fake` changed the stored values, and
`kernel._unsafe_update_src` gave `5·x` (`round-17/lr-17a/work/host_modifies_triton.py`). Neither
SEMANTICS nor `Faithful`'s scope excludes this. By the user's decision on a malicious prover (under
L36), a stated limit and the reviewer's check suffice: add a scope condition (Triton 3.8.0's
installed modules unmodified, and the kernel's module's names bound only as its source binds them,
nothing else assigning to them or to the JIT functions' sources), and name the check in REVIEWER_GUIDE's
host section and the rule page's checklist. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): new scope 10 of `Faithful` (Triton and the kernel's module as installed and
  written), checked by the kernel reviewer. Accepted by [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md) and [lr-20e](round-20/lr-20e/report.md) (a reviewer following item 10
  catches M1–M4); **reopened** by [lr-20a](round-20/lr-20a/report.md) and [lr-20d](round-20/lr-20d/report.md): scope 10 ("no code … changes a module, class or object
  of Triton", REVIEWER_GUIDE "assigns to a Triton module, class or object") also forbids what in-scope
  hosts must do: tutorials 06 and 09 set `block_shape` on host `TensorDescriptor`s in their `pre_hook`s
  ("Which launch" covers it), tutorials 06, 08 and 09 call `triton.set_allocator` (scope 5 requires
  it), scope 2 allows assigning a knob, and Triton's own code changes its objects at every launch
  (`round-20/lr-20a/work/scope10.out`). In the other direction "assigns to" misses changes made by a
  call (`setattr`, `triton.runtime.driver.set_active`, `importlib.reload`), and nothing says the
  imported `triton` is the installed one or that the cache directory holds only what Triton wrote
  ([lr-20a](round-20/lr-20a/report.md); the cache is L109). Fix, the same text in `lean/TVA/Rules/Compiler.lean` scope 10, the rule
  page's scope and checklist item 10, REVIEWER_GUIDE and PROVER_GUIDE item 10, SEMANTICS "Names" and
  "Launches" and the exclusions row: (a) the process imports Triton 3.8.0's installed package,
  unmodified (no other `triton` on `sys.path`, no substitute in `sys.modules`); (b) no code but
  Triton's own changes an attribute of a Triton module, a class Triton defines, or the kernel's or a
  helper's `JITFunction`, by assignment, `setattr`/`delattr`, or a call made to change it
  (`_unsafe_update_src`, `driver.set_active`, `importlib.reload`); (c) other than through Triton's API
  as the in-scope tutorials use it: launches, `warmup`, `triton.autotune` with its `Config`s and hooks,
  the `JITFunction`'s `debug` (scope 4), `triton.set_allocator` (scope 5), the knobs scope 2 allows,
  and the fields of a host `TensorDescriptor` the launch is given (a hook may set its `block_shape`);
  (d) the clause on the kernel module's names, unchanged. Round 21.
- **User decision, 2026-10-29** (under L36): the cache-directory clause of the fix above is dropped
  (the kernel is assumed new, scope 11).
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate. Stated as part (B) of `Faithful`'s scope, by the user's decisions of 2026-10-29
  (under L36).

## L105

**Two routes to compile options that nothing names** (soundness; [lr-18b](round-18/lr-18b/report.md)). Setting the attribute
`kernel.debug = True` made an `int32` overflow trap, so `Launch.debug`'s docstring ("False
otherwise") is false; and after `JITFunction.preload` of a kernel compiled with
`enable_reflect_ftz=False`, a plain launch ran that code (`round-18/lr-18b/work/preload_probe.py`).
Scope 4 lists neither. Name both in scope 4 (as excluded, or as giving `Launch`'s fields), in
`Launch.debug`'s docstring, and in the reviewer's host check. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md).

## L106

**An `Assumptions` that is not a structure** (soundness; [lr-18a](round-18/lr-18a/report.md)). An `inductive Assumptions` whose
constructor holds the specification passes `shape`, `axioms`, `usage` (through a `private` lemma),
`namespace` and `kinds`: the claim reader panics, returns no fields, and still exits 0, so the claim
is vacuous for the reviewer who trusts the gate's field list. By L36 this is cheap and complete: have
the reader require `isStructure`, and fail `shape` on any `PANIC` or non-zero exit, and say so in §8.
Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md). Follow-up: two misreports of the claim reader, both failing closed (L112).

## L107

**Host-dependent values without a named host** (soundness; [lr-19b](round-19/lr-19b/report.md)). A NaN passed as a `tl.float16`
argument is `0x7e00`/`0xfe00` only under CPython 3.13: under CPython 3.14, which TVA's `pyproject.toml`
allows (`>=3.10`), `0x7ff4…` becomes `0x7d00` (`round-19/lr-19b/work/host_pack2.py`); and "`inf - inf`
is `0xfff8…`" on the host is x86-64's default NaN. State the host (x86-64, CPython 3.13, as TVA's
evidence ran) as part of the target in SEMANTICS, exclusions.md and `Faithful`'s scope, and pin
`pyproject.toml`'s Python to it. Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md). Follow-up: `-O` and the floating-point environment of the host (L110).

## L108

**Wording from the reviews of rounds 17–19**:
- SEMANTICS' `tl.cdiv`: `div - 1` wraps only for a runtime `div`; for a compile-time `div` it is
  Python's ([lr-17a](round-17/lr-17a/report.md));
- name the exceptions of two general SEMANTICS sentences: "what runs is what it evaluates" (code
  visited after an `if` that returns runs nowhere, L89) and "every argument is evaluated"
  (`tl.static_assert`'s `msg` only when the assertion fails) ([lr-17b](round-17/lr-17b/report.md));
- `evidence/tool-round18/run.py` no longer re-runs after the freeze: `pin_ledger` deletes the four
  frozen work files, Part R crashes and Part V finds nothing to freeze ([lr-18a](round-18/lr-18a/report.md), [lr-18b](round-18/lr-18b/report.md));
- `freeze_problem` accepts a non-normalised path the tool never writes (harmless; [lr-18a](round-18/lr-18a/report.md));
- an unescaped `|` inside code spans breaks table rows in `docs/rules/num.md` (8 rows) and
  `docs/LIBRARY.md:40` ([lr-18a](round-18/lr-18a/report.md));
- `docs/rules/num.md` and `Format.sqrtMag_rn_spec` say "within half an ulp", but the lemma proves
  `√x·u + η`, which can approach a whole ulp ([lr-19b](round-19/lr-19b/report.md));
- `FloatV.nan`'s docstring lists `div_rn` and `sqrt_rn` for `float16`/`bfloat16`, which Triton accepts
  on `float32` only, and omits that `float64` `propagate_nan=ALL` gives `0x7ff8…` ([lr-19b](round-19/lr-19b/report.md));
- `lean/TVA/Lemmas/Int.lean`'s docstring promises an unsigned `<=` lemma that does not exist, and
  LIBRARY.md's `Block` row omits `addrs_block` ([lr-19b](round-19/lr-19b/report.md));
- evidence/float-layer's conclusion 4 still says "The library states neither yet (L64)"; the `CvtSat`
  facts state the saturation (round 19's author, [lr-19a](round-19/lr-19a/report.md), [lr-19b](round-19/lr-19b/report.md)).

Round 20.
- **Round 20** ([notes](round-20/author/notes.md)): accepted by [lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md), [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md), [lr-20e](round-20/lr-20e/report.md) and [lr-20f](round-20/lr-20f/report.md).

## L109

**Code compiled outside the scope** (soundness; [lr-20c](round-20/lr-20c/report.md), [lr-20d](round-20/lr-20d/report.md)). Scopes 2 (knobs), 3 (hooks) and 10 (Triton
unmodified) are stated of the claimed launch's own process at the launch, but the launch may run
code compiled earlier and found in the process's cache or the cache directory, under keys that hold
neither a knob set from Python (`knobs.language.fp32_default`), nor `TRITON_KERNEL_OVERRIDE` /
`knobs.compilation.override` (absent from `CACHE_INVALIDATING_ENV_VARS`), nor the state of
`triton.language` in memory. On the H200 (`round-20/lr-20c/work/cache_poison.py`,
`round-20/lr-20d/work/cache_cross_process.py`, `knob_history.py`): a clean process on a cache directory
an earlier process had used ran that process's code (`log x` for `tl.exp(x)` after `tl.exp = tl.log`
there; the `ieee` `tl.dot` after a Python-set `fp32_default = "ieee"` there; `x + 2` after an
override there); within one process, a knob set for an earlier launch and reset to its default left
its code in use. `TRITON_F32_DEFAULT` in the environment, which is in the key, did not leak; a fresh
directory gave the model's values. No reviewer reading the claimed host can see another process. Fix,
a stated limit with the reviewer's check (the user's rule on a malicious host): scopes 2, 3 and 10
hold at every compile whose code the launch may run, at any time before it in its process and in
every process that has written the cache directory it uses; a process that keeps them from its start,
with a cache directory no other process writes (or a fresh one), meets it, and so does one that sets
`TRITON_ALWAYS_COMPILE=1` (an exception of scope 2; it ignores the directory's entries,
`round-20/lr-20d/work/keys_extra.py` Y2). State it in the Lean docstring, the rule page's scopes and
checklist items 2, 3, 10 (or a new scope 12), REVIEWER_GUIDE and PROVER_GUIDE, SEMANTICS "Launches",
exclusions.md and `TVA.Claim`'s docstring. Round 21.
- **User decision, 2026-10-29** (under L36): the kernel is assumed new and Triton's caches are not
  verified. Code compiled earlier and reused through either cache is outside that assumption; round
  21 states it in scope 11, with no new scope condition or reviewer check, instead of the fix above.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate. Stated as part (B) of `Faithful`'s scope, by the user's decisions of 2026-10-29
  (under L36).

## L110

**The host process: `-O` and the floating-point environment** (soundness; [lr-20b](round-20/lr-20b/report.md), [lr-20d](round-20/lr-20d/report.md)). Scope 7 names
the host as x86-64 and CPython 3.13 (L107), which admits two host states that change outcomes:
- `-O`/`-OO` or `PYTHONOPTIMIZE` removes the Python `assert`s by which Triton's front end makes some
  host rejections (`primitives/triton-3.8.0-code-generator.py:497, 872, 886, 1020, 1117-1121`): a loop
  body assigning `tl.float32` to a name holding `tl.float16`, rejected under `python -I` ("Names after
  a loop"), compiled and ran under `python -I -O` and stored `1.0` where Python's value gives
  `1.00048828125` (`round-20/lr-20b/work/probes3.out` D);
- the process's rounding mode and MXCSR flush-to-zero / denormals-are-zero bits change `litF32`,
  `hostF32`, `litBF16` and compile-time float arithmetic: after `fesetround(FE_TOWARDZERO)`,
  `tl.full([1], 0.1, tl.float32)` and the argument `0.1` stored `0x3dcccccc` (the library gives
  `0x3dcccccd`); with FTZ/DAZ set, `tl.full([1], 1e-40, tl.float32)` stored `0`
  (`round-20/lr-20d/work/host_env.py`). Host code sets these with `fesetround`, `_mm_setcsr`, or by
  loading a library built with `-ffast-math`.

Fix, as a stated limit with the reviewer's check: in scope 7 (`lean/TVA/Rules/Compiler.lean`, the
rule page and checklist item 7), both guides' item 7, SEMANTICS' introduction and "Launches",
exclusions.md's targets row, TRANSCRIPTION's host pitfall and `FloatV.hostF32`'s docstring, name the
host as "CPython 3.13 run without `-O`/`-OO` and with `PYTHONOPTIMIZE` unset, in the default
floating-point environment (round to nearest even; flush-to-zero and denormals-are-zero off: no
`fesetround`, no `_mm_setcsr`, no loaded library that changes them)", with the reasons. Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate. Stated as part (B) of `Faithful`'s scope, by the user's decisions of 2026-10-29
  (under L36).

## L111

**Arithmetic in an fp8 dtype is not always rejected** (soundness; [lr-20e](round-20/lr-20e/report.md)). SEMANTICS "Constants" (the fp8
bullet: "With an fp8 tensor an arithmetic operator is rejected anyway") and "Operators and promotion"
(last item: "the backend rejects it on the host") are false: MLIR's `arith` folders run first. With an
fp8 tensor `x` of either dtype, `x * 1.0`, `1.0 * x`, `x * 1`, `x / 1.0` and `x - 0.0` compile and run,
storing `x` on all 256 patterns, and arithmetic on two fp8 constants is computed at compile time and
runs (`tl.full([256], 1.5, d) * tl.full([256], 2.0, d)` stores `3.0`); `x + 0.0`, `x * 2.0`, `-x`,
`x + x` and others are rejected (`round-20/lr-20e/work/fp8_arith.py`, `.out`). A claim about
`x * SCALE` with an fp8 `x` that models every `SCALE` as rejected is false at `SCALE = 1.0`. Fix: put
every operator whose computation dtype (`computationType?`) is an fp8 dtype, and unary `-` on an fp8
tensor, outside the supported subset (its Operators item and list of excluded forms), with a row in
exclusions.md citing this work file; replace the two sentences with "the backend rejects it on the
host, except where MLIR's folders remove the operation first (`x * 1`, `1 * x`, `x / 1` and `x - 0`
give `x`; an operation on two fp8 constants is computed at compile time); none of these forms is in
the subset". No in-scope tutorial computes in fp8. Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L112

**Two misreports of the claim reader, both failing closed** (gap; [lr-20c](round-20/lr-20c/report.md)). `reader_result` takes any
output line holding `PANIC` as a panic, the `TVR_JSON` line included, so a field docstring or name
holding the word fails `shape` as "the claim reader panicked"; and a `private` field of `Assumptions`
is listed in `fields` but its statement is not read (the reader looks it up as
`<ns>.Assumptions.<f>`, and its projection is private), so `kinds` says "not tagged" of a tagged field
(`round-20/lr-20c/work/reader_probe.py` R1, R3). Fix: skip the `TVR_JSON ` line, or match Lean's
`PANIC at ` at a line's start; read each field through `StructureFieldInfo.projFn`, or refuse a private
field in the reader with a named error. Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L113

**Three model operations without a rule** (gap; [lr-20f](round-20/lr-20f/report.md)). `tl.convert` (integer `.to(…)`, implicit
promotions and `tl.where`'s `int1` condition, which TRANSCRIPTION's worked patterns write as
`← tl.convert …`) has no rule, so each kernel must build the result tensor by choice; `tl.sum0c` has
none, though `safe_sum1c` exists; `tl.store1` has none, against LIBRARY.md's "one rule per operation".
Fix: add `safe_convert` (premise `∀ i, ∃ r, ConvRel E.P d d' (x.get i) r`; continuation for every
`r` with `∀ i, ConvRel E.P d d' (x.get i) (r.get i)`) and `safe_sum0c` (as `safe_sum1c`) to
`lean/TVA/Lemmas/Ops.lean`, both proved as stated in `round-20/lr-20f/work/MissingRules.lean`; add
`safe_store1`, or say in the module docstring that `safe_store` applies to `tl.store1`; list them in
LIBRARY.md's `Ops` row. Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L114

**Wording from the final review of round 20**:
- SEMANTICS "Values" (the fp8 row: "a Python number built in them is its exact value, or the kernel
  is rejected"), "Constants" (the fp8 bullet: "is never rounded"; "every other `c` … is rejected") and
  `litF8`'s docstring speak of `c` itself, but Triton first rounds `c` to `float32`, as `litF8` tests:
  `1 + 2^-30` and `1 + 2^-40` build `1.0`, `1e39` builds `+∞`, `-1e-50` builds `-0`, none rejected
  (`round-20/lr-20a/work/probes.out` P1, `round-20/lr-20b/work/probes2.out`). Say "the exact value of its
  `float32` constant (`litF32 c`), or the kernel is rejected", and drop "is never rounded" ([lr-20a](round-20/lr-20a/report.md), [lr-20b](round-20/lr-20b/report.md));
- PROVER_GUIDE's "Write each Python number as the constant Triton builds" lists `litF64`, `litF32`,
  `litF16`, `litBF16` only; add "`FloatV.litF8` in an fp8 dtype (not `tl.cvt8`)" or `Operand.constIn?` ([lr-20b](round-20/lr-20b/report.md));
- `docs/rules/num.md` under the facts table: "The conversion and division facts are exact", but
  `DivFull` is a bound (at most 0.651 of it). Say "The conversion facts (`Cvt…`, `CvtSat…`) and
  `DivRn P .binary64` are exact (bit for bit, up to NaN payload)" ([lr-20e](round-20/lr-20e/report.md));
- three places still call `u·|y| + η` "half an ulp": the module docstrings of
  `lean/TVA/Lemmas/FusedErr.lean` (11-12) and `Consistency.lean` (44-45), and `base` in
  `lean/TVA/Lemmas/Sm90.lean:38`; say "one rounding's bound" ([lr-20f](round-20/lr-20f/report.md));
- `ofRat_err` (`lean/TVA/Lemmas/FloatErr.lean:100`) calls `FloatV.ofRat` "`tl` constants of exact
  values"; say "the nearest float to an exact rational" ([lr-20f](round-20/lr-20f/report.md));
- `lean/TVA/Lemmas/Shared.lean:23` says an `AtomicStep` records the update; it records the cell and
  the old and new values ([lr-20f](round-20/lr-20f/report.md));
- LIBRARY.md's `Consistency` row says `FloatV.exists_nearest` holds "from any real"; it needs
  `|y| ≤ maxVal` ([lr-20f](round-20/lr-20f/report.md)).

Round 21.
- **Round 21** ([notes](round-21/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L115

**`TVA.Lemmas.Softmax` removed for its provenance** (independence; coordinator, 2026-10-29, for the
user's goal "make sure the independence of tva, provers and reviewers cannot cite other project's
results or lemmas"). The coordinator checked provenance by authorship, not only by text: every tool
call of every agent of the project's sessions, for what each read outside TVA, and what each then
wrote into TVA. One file of the library had been written, on 2026-10-27, by an agent that had just
read another project's float `max`/compare lemmas and exponential notes: `lean/TVA/Lemmas/Softmax.lean`
(online-softmax error lemmas over the reals). Its text shares no theorem name or statement with that
project, but a lemma derived from another project's work is not cleared by a text comparison. No
kernel and no other library module used it, so it is removed (with its `TVA.lean` import, its
`LIBRARY.md` row and a line of `docs/plan/examples.md`), and is to be written again, from TVA and
Mathlib alone, when a softmax kernel needs it. The same agent's order-key `max` lemmas had already
been replaced by `TVA.Lemmas.FloatMax`, which round 12's author wrote with no access to any other
project. No review followed this removal (after the final batch, round 21); it is checked by the
build, `tvr library-check`, the audit and the release check, and leaves `vec_add_f32`'s certificate
as it was.
- **The rest of the project, checked the same day** (coordinator):
  - *Authorship.* Apart from that agent, only two early kernel reviewers (2026-10-27, on kernels
    since deleted, whose filings never reached the repository) and the coordinator's own session
    (2026-10-26, while building TVA's second version) read the other project's content; library
    reviewers that listed folder names read no file. No prover, kernel reviewer or library author
    did so after the independence rules (L7). Nothing those agents wrote survives in the library
    except through later rewrites: the copied float layer was replaced clean-room in round 12
    (L65: its author worked in a tree without the copied files or their history, and its
    transcript shows it never looked for them).
  - *Declarations.* Of TVA's 1,498 declaration names, 276 also occur in the other project; their
    statements and bodies were compared. The float layer shares names and types that the rest of
    TVA used, as L65's clean room intended, but its definitions are written differently (fields by
    `2^`, division and remainder rather than bit operations; another rounding algorithm). The only
    theorems with nearly the same statement are elementary (`u_pos`: `0 < u`; `binary32_emin`:
    `emin = -126`; `abs_toRat`: `|value| = magnitude`). Of 8,058 distinctive Lean lines, 27 occur
    there, all generic tactic steps or standard definitions.
  - *Results.* Every measured fact kernels may cite (32 of the 38; the six `…Total` facts only say
    a result exists) names a TVA evidence folder, and each was re-run by library reviewers; the
    constants an early kernel reviewer had cited from the other project (`Exp2F32`'s `2^-126`)
    stand on TVA's own test of all `2^32` inputs (evidence/num-math-f32).
  - *History.* The repository's first commit still holds the copied float layer; the user chose,
    2026-10-29, to keep the history as it is.

## L116

**The autotuner is not verified: a claim fixes the selected config** (gap; the user's decision of
2026-10-29, under L36). The library now covers an autotuned call: `Compiler.Faithful`'s "Which
launch" treats a call of a `@triton.autotune`d kernel as host code that benchmarks every config on
a key it has not timed and then launches the chosen one, and part (A) of the scope counts
`triton.autotune` with its `Config`s and hooks. By the user's decision TVA verifies the kernel
itself, not the autotuner, and (the user's correction, same day) "it is we pick a config. prover is
not aware of the autotune":
- `@triton.autotune` is outside the supported subset of a kernel's source, with a row in
  exclusions.md; the autotuner's benchmark runs, key, pruning, `reset_to_zero`/`restore_value`,
  `do_bench` and choice are outside every claim;
- for a kernel the tutorial autotunes, we (the user and the coordinator) pick one config when we set
  up its folder, before any prover starts: its `source/` is the tutorial's source with the decorator
  removed, launched with that config's meta-parameters and launch options (`num_warps`,
  `num_stages`, …), the host doing what the config's `pre_hook` does; the folder records the choice;
- the prover is not told about autotuning and proves the kernel as given; the kernel reviewer judges
  that kernel, not the choice of config, and a source that still holds `@triton.autotune` is
  `UNSUPPORTED`;
- `docs/REVIEW_PROTOCOL.md` says how a kernel's folder is set up, the config choice included;
- "Which launch" and part (A) cover a direct launch of the JIT function; condition 11 reads as the
  user's decision says.
Round 22.
- **Round 22** ([notes](round-22/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate.

## L117

**Condition 11's reuse clause is too loose** (soundness of an assumption's wording;
`vec_add_f32` F3 by r-vecadd and F4 by r-vecadd2, independently, 2026-10-29, in
`kernels/vec_add_f32/review/round-2/`). Condition 11 of `Compiler.Faithful`'s scope (part (B),
assumed) lets the launch run code compiled "earlier for the same source, compile-time values and
settings", but does not ask for the same specialisation of the runtime arguments (point 6's classes:
an integer's divisibility by 16 and equality to 1, a pointer's 16-byte alignment). Read literally, a
launch with `n = 98433` may run code compiled for `n = 98432`, and point 6's own evidence
(evidence/compiled-launch K2) shows code of another specialisation writing past `n`. Its example, "a
repeated launch of the same instance", can also be read as the only reuse allowed, which would leave
a benchmark's later launches at other sizes outside the rule. Fix: in the Lean docstring, the rule
page, SEMANTICS, exclusions.md and `TVA.Claim`, say "the same source, compile-time values,
specialisation of the runtime arguments (point 6) and settings", and give the example as one case
("for example, a repeated launch"). `vec_add_f32`'s launches are all in one class (every size the
script uses is a multiple of 16), so its claim is not affected.
- **User decision, 2026-10-29** (under L36): the runtime arguments' values are the instance's, and
  the prover's conditions constrain them; condition 11 is stated for the launch's own argument
  values, with no reuse clause. Round 23.
- **Round 23** ([notes](round-23/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate. The kernel reviewers who reported it released their findings (`vec_add_f32`
  F2, F3, F4) at gate `5f65eb1f7a961315`.

## L118

**Two wording slips** (`vec_add_f32` F2 by r-vecadd, and a remark of r-vecadd2, 2026-10-29). In
`lean/TVA/Lang/Scalar.lean:103` the docstring "`tl.int32` values, the type of `tl.program_id` …"
sits on `abbrev I8 := IntV true 8`, and `I32` has none; and the rule page and `TVA.Claim`'s docstring
still say only "without `-O`" where condition 7 says without `-O` or `-OO` and with
`PYTHONOPTIMIZE` unset. Fix: move the docstring to `I32` (and give `I8` its own), and say what
condition 7 says. Round 23.
- **Round 23** ([notes](round-23/author/notes.md)): fixed; no library review followed, by the user's
  decision (under L36). Checked by `lake build TVA`, `tvr library-check`, `tvr lint`, `tvr audit`, the
  release check and `vec_add_f32`'s gate. The kernel reviewers who reported it released their findings (`vec_add_f32`
  F2, F3, F4) at gate `5f65eb1f7a961315`.

## L119

**`relDot8` keeps a margin of about two over its derivation** (accuracy, not soundness;
`matmul_e5m2` F3 by r-e5m2 and the second remedy of F4 by r-e5m2b, 2026-10-31). `relDot8 K =
(1 + 2^-7)^⌈K/32⌉ − 1` (`lean/TVA/Rules/Num/Sm90.lean`) rounds up, to `2^-7`, the per-chunk
`33·2^-13` that [evidence/num-dot](../../evidence/num-dot/README.md) derives for `wgmma`'s fp8
accumulation (13 bits kept below the largest term). Compounded over `⌈K/32⌉` chunks, the constant
passes `1` from `K = 2817`, so a tolerance built from `DotLoop.bound` over all of `Σ|a·b|` says
nothing from there on; with the unrounded constant it would be about `0.67` at `K = 4096`.
- **Settled by the coordinator** (2026-10-31). The fact is a bound with margin, as every fact of
  [rules/num.md](../rules/num.md) is: no measured error was above `0.39` of it, and the ratio falls
  as `K` grows. Making it tighter changes a device fact, which needs library review against the
  evidence, and library rounds have stopped (the user's decision under L36). The kernel does not
  need it: r-e5m2's F1 shows that the same fact, applied step by step against the exact partial
  sums, gives a tolerance that rejects a zero result and a dropped step at every covered `K`, and
  `matmul_e5m2` is revised that way. A tighter fp8 fact is left to a later library round, if a
  kernel needs one.

## L120

**A descriptor the kernel builds again in a loop, while an earlier TMA store through it may be in
flight** (gap; `grouped_matmul_tma_f16` F2 by r-gmm2, 2026-10-31). Tutorial 08's
`grouped_matmul_tma_kernel` calls `tl.make_tensor_descriptor` for `a`, `b` and `c` once per group,
inside `for g in range(group_size)`. Each program therefore rewrites its three scratch slots once
per group, and the previous group's last `c_desc.store` may still be in flight when `c`'s slot is
rewritten. SEMANTICS ("Tensor descriptors") gives a built descriptor its arguments' values and keeps
the scratch outside the claim's memory, and [evidence/desc-scratch](../../evidence/desc-scratch/README.md)
tests kernels that build each descriptor once. Nothing in the library states that a rebuild comes
after the accesses through the old contents.
- **What the coordinator saw** (2026-10-31; an observation, not library evidence):
  - **The PTX.** In the PTX Triton 3.8.0 compiles for that kernel (config 128/128/64, 3 stages,
    4 warps, sm_90a), each rebuild copies the new tensormap to global memory
    (`tensormap.cp_fenceproxy.global.shared::cta…release.gpu`) and then runs
    `cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0`. `a`'s and `b`'s slots are
    rewritten and waited on first, so before `c`'s slot changes, every earlier bulk store has
    finished reading its source. PTX does not say in so many words that the store's tensormap
    has been read by then.
  - **A stress run.** On an H200 NVL, one to three programs ran 64 to 256 groups of one 64- or
    128-step tile each, 80 launches in all. No element was wrong or missing.
- **Status.** The issue stays open for a later library round (a statement in SEMANTICS, with an
  evidence folder), if the user starts one: library rounds have stopped (L36). Kernel review
  trusts the library as it stands, so the `LIBRARY_REPORT` does not block the kernel.
- **Closed** (round 32, [notes](round-32/author/notes.md)): settled by L146. Descriptors built
  inside a kernel are excluded by the user's decision (2026-11-08), so no covered launch rebuilds
  one (`Compiler.Faithful` scope 5, `docs/exclusions.md`).

## L121

**Scope 5 does not ask the allocator to honour the alignment the launcher passes** (gap;
`grouped_matmul_tma_f16` F4 by r-gmm, 2026-10-31). For a kernel that builds descriptors, the
launcher calls the allocator set by `triton.set_allocator` as `alloc_fn(size, alignment, stream)`,
and uses the buffer it returns as the descriptors' scratch. For `grouped_matmul_tma_kernel`, the
compiled kernel's metadata gives `global_scratch_align = 128`. Scope 5 of
[rules/compiler-faithful.md](../rules/compiler-faithful.md) and SEMANTICS ("Tensor descriptors")
ask only two things of the scratch: that it holds none of the launch's bytes, and that nothing else
writes it during the launch. Neither asks that it have the alignment the launcher asks for.
- **The tutorials' allocators.** Tutorials 06 and 08 return
  `torch.empty(size, dtype=torch.int8, device="cuda")` and ignore `alignment`. PyTorch's caching
  allocator returns blocks aligned to at least 512 bytes, so their buffers are aligned in practice,
  but the rule does not require an allocator to be.
- **Status.** Left open for a later library round (a condition in scope 5), if the user starts
  one (L36). The `LIBRARY_REPORT` does not block the kernel.
- **Closed** (round 32, [notes](round-32/author/notes.md)): settled by L146. No covered launch
  builds a descriptor in the kernel, so the launcher allocates no scratch and calls no allocator
  (`Compiler.Faithful` scope 5, `docs/exclusions.md`).

## L122

**`num_stages` and `warp_specialize` are said to be "given by a model", but no model gives them**
(wording; `attn_fwd_f16` F3 by r-fwd, 2026-10-31). SEMANTICS's "Performance-only arguments" lists
`num_stages` (of a launch, a `triton.Config` or `tl.range`) and `warp_specialize` among the
arguments that are "not performance-only, and given by a model". `Launch` and `tl.forRange` have no
field for either, and SEMANTICS's "Loops" says they change no outcome of a well-behaved model, so a
model leaves both out (as `attn_fwd_f16`'s does, rightly). The sentence should say what the two
paragraphs mean together: they are not performance-only, because they change the order of a
program's accesses, and a model omits them because a well-behaved model has no conflict within a
program. Left open for a later library round, if the user starts one (L36); the `LIBRARY_REPORT`
does not block.

## L123

**The fusion rule does not say whether a value carried through `tl.maximum` or `tl.where` is a
product** (wording or gap; `attn_fwd_f16` F5 by r-fwd2, 2026-10-31). SEMANTICS's fusion rule says
that an addition may fuse when an operand is a product, "computed by `*` … through any dataflow".
It does not say whether the result of a `tl.maximum` or a `tl.where` whose value is a product's
counts. In `_attn_fwd_inner`, `m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)` feeds
`m_i - m_ij` (line 84), and the epilogue's `m_i += tl.math.log2(l_i)` adds to a value carried from
it. `attn_fwd_f16`'s model writes both as plain IEEE operations, which r-fwd2 judges right under
its reading (a `tl.maximum` result is not a product). The library should say which reading holds,
for `tl.maximum`, `tl.minimum` and `tl.where` (a select whose arm is a product may be folded into
the addition before contraction). Left open for a later library round, if the user starts one (L36); the `LIBRARY_REPORT`
does not block.

## L124

**The fp8 `tl.dot` facts compound over a fed-back accumulator** (sharpness; `attn_fwd_e5m2` F2 by
r-e5fwd2, 2026-10-31). Not a doubt of the facts' truth. `DotE5m2` (and the other `DotE4m3…`
facts) gives `DotAcc` with `relDot8 K = (1 + 2^-7)^⌈K/32⌉ − 1` of `|acc| + Σ|a·b|`, as
`docs/rules/num.md`'s chunk model bounds each `wgmma` instruction by `33·2^-13` of the magnitudes'
sum, the accumulator included. A kernel that feeds its accumulator back through
`tl.dot(p, v, acc)` once per key block (tutorial 06's `_attn_fwd`; any fp8 K loop with
`maxImpreciseAcc = 2^30`) can then prove only a bound growing as `(1 + relDot8 64)^T`: about `2.7`
at `T = 64` blocks and `10^6` at `N_CTX = 2^16` in `attn_fwd_e5m2`'s tolerance. A score's
`relDot8 D·Σ|q·k|`, inside an exponent, also lets each softmax weight move by a factor
`2^(1.5·relDot8 D·X)`. r-e5fwd2 suggests, if the chunk model's evidence supports it, a per-instruction
form such as the smaller of the present bound and `Σ|a·b| + 2^-13·|acc|` (a truncated product loses
at most its own magnitude, the accumulator only its own truncation), which would bound such loops by
about the terms' magnitudes. Left open for a later library round, if the user starts one (L36); the
`LIBRARY_REPORT` does not block, and `attn_fwd_e5m2`'s write-up states the limit it puts on that
claim's tolerance.
- **Round 24 (user, 2026-10-31): "please tighten the fp8 kernel error bound."** A one-time fix,
  as rounds 21 to 23, with no library review after it. A new fact, not a change to `DotE5m2`
  (which `matmul_e5m2` cites): per output element, the fp8 `tl.dot` as its `⌈K/32⌉` chunks, each
  within a constant times the chunk's largest term, the accumulator included (the 13-bit window
  that `evidence/num-dot` Part 1 measures), with its own evidence. It lets `attn_fwd_e5m2` bound
  each score by the chunks' largest terms instead of `Σ|q·k|`. The accumulator fed back through
  `tl.dot(p, v, acc)` still compounds under that fact. The coordinator wrote here that this was the
  device's behaviour under the measured window, not slack in the fact; the round 24/25 library review
  showed that this is wrong (L127, below): the window loses at most each truncated term.
- **Fixed in round 24** (no library review follows, as for rounds 21 to 23): the facts `DotE4m3Chunk`,
  `DotE5m2Chunk`, `DotE4m3E5m2Chunk`, `DotE5m2E4m3Chunk` (`lean/TVA/Rules/Num/Dot8Chunk.lean`,
  `DotChunkAcc`, `DotF8Chunk`, `rhoChunk8 = 2^-7`, `sigmaChunk8 = 2^-12`, `alphaChunk8 = 2^-126`,
  `liftMag`, `chunk8`, `chunkSum8`), the lemmas of `lean/TVA/Lemmas/Dot8Chunk.lean`, `docs/rules/num.md`'s
  section and rows, and `evidence/num-dot-chunk` (a bit-exact model of `wgmma` on about `2·10^8`
  results; largest ratio to the bound `0.468`). The author's notes: `round-24/author/notes.md`.
- **Round 24/25 library review (2026-11-01, user: "run one round of library review and fix."):**
  lr-24a ACCEPT, lr-24b **REOPEN**. The chunk facts are true (both reviewers' adversarial tests:
  about `3.5·10^7` more elements, none outside, largest ratio `0.44`), but a fed-back accumulator
  still grows by `1 + 2^-7 + 2^-12` per chunk, and the sentence above that this is the device's
  behaviour is wrong (L127). By the facts' own model a truncated term loses at most `min(|term|, u)`,
  so each chunk is also within `B_alt = (1+2^-13)·Σ_chunk|a·b| + 2^-12·Mj + 2^-12·|s_j + c_j| + 2^-126`
  (lr-24b's `work/fp8_alt.out`: every chunk within it, largest `0.82` on single-chunk dots checked
  on the device). Round 26: bound each step by the smaller of the two (with a margin), extend the
  evidence's per-chunk check to it, and give the lemmas (`errB`, `chain_err`) the minimum.
- **Fixed in round 26** (no review follows; user, 2026-11-01: "run one round of library review and
  fix."): new facts `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss`,
  `DotE5m2E4m3ChunkLoss` (`DotF8ChunkLoss`, shape `DotChunkLoss`, `lean/TVA/Rules/Num/Dot8Chunk.lean`):
  the same partial values, each step within `min(2^-7·Mj, 2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126`
  (`A_j = chunkAbs8`, the chunk's exact `Σ|a·b|`), so a fed-back accumulator grows by at most
  `1 + 2^-11` per chunk. Each implies the old chunk fact (`DotF8ChunkLoss.chunk`); the existing facts
  and lemmas are unchanged, and no kernel stops building. Lemmas `errM`, `chain_errM`,
  `errM_le_errB`, `errL`, `DotChunkLoss.err`, `DotF8ChunkLoss.err`. Evidence
  `evidence/num-dot-chunk-round26`: about `9.1·10^7` elements, none outside, largest ratio `0.487`.

## L125

**The fp8 `tl.dot` facts are false when a subnormal operand sets the window** (soundness; library
author, round 24, `evidence/num-dot-chunk` Part F; 2026-10-31). On `wgmma` the tensor core aligns a
chunk's terms to the largest exponent field, and a subnormal fp8 operand's field reads as the
smallest normal's. A product with a subnormal operand can then set the window while its magnitude is
far below it, and the other products, just below the window's unit, are lost whole. `DotE4m3`,
`DotE5m2`, `DotE4m3E5m2` and `DotE5m2E4m3` bound the error by `relDot8 K` times
`|acc| + Σ|a·b|`, which assumed the window follows the largest term: the error reaches `3.65` times
that bound with a `float8e4nv` left operand (`K = 32`; `1.79` at `K = 64`) and `1.73` times with a
`float8e5` one (`K = 32`). The coordinator reproduced it independently on the H200 (GPU 0): with
`a[0] = 2^-16` (a subnormal `float8e5`), `b[0] = 2^15` and 31 products `1.75·2^-13`, `K = 32`, the
`wgmma` result is `0.5` against the exact `0.50662`, an error of `1.67` times `DotE5m2`'s bound; with
normal operands of the same product (`2^-14 · 2^13`) the error is `0.24` times it.
`matmul_e5m2` and `attn_fwd_e5m2` cite `DotE5m2`: both certificates rest on a false fact until the
fix and their re-proofs. The new chunk facts (L124) count a subnormal operand as the smallest normal
(`liftMag`) and hold on the same inputs (ratio `0.47`).
- **Round 24 (coordinator).** Fix in the same batch: no listed fact may be false. Keep subnormal
  operands covered (tutorial 03's and 06's inputs, `randn` cast to fp8, include them), for example by
  counting each operand's magnitude as `liftMag` does, or by deriving the whole-dot bound from the
  chunk facts by a lemma instead of a second device fact. `matmul_e5m2` and `attn_fwd_e5m2` are then
  re-proved and re-reviewed.
- **Fixed in round 24** (no library review follows): the four facts and `DotF8` are withdrawn from the
  Lean and from "Facts the library states", with their `Platform.sm90ref` lemmas; `chunks8`, `relDot8`
  and `absDot8` move to `lean/TVA/Rules/Num/Dot8Chunk.lean` unchanged. A whole-dot bound is now a
  lemma of the chunk facts, `Dot8Chunk.DotF8Chunk.liftAcc` (`relLift8`, `absLift8`, against
  `|acc| + Σ liftMag·liftMag`), not a device fact. `matmul_e5m2` and `attn_fwd_e5m2` no longer build
  until their provers cite `DotE5m2Chunk`; both are re-proved and re-reviewed.
- **Accepted** by the round 24 library review (lr-24a and lr-24b, 2026-11-01).

## L126

**`DotF16` and `DotBF16` fail when a product with a subnormal operand sets the window**
(soundness; asked by r-gmm as `grouped_matmul_tma_f16` F5, and by r-gmm2, r-fwd2, r-bwd, r-bwd2 and
r-matmul2 in their round-24 re-acceptances; 2026-11-01). L125's mechanism holds for 16-bit operands
too. The tensor core reads a subnormal operand's exponent field as the smallest normal's. So a product
`a·b` with a subnormal `a` and a large `b` can set the chunk's window while being far smaller, and the
other products lose their bits below the window's unit. `evidence/num-dot`'s Part 2 had no such input.
The coordinator measured it on the H200 (`evidence/num-dot-sub16`, `probe.py`), at `K = 16` on both
`wgmma` and `mma.sync`, with the large product at any position:
- `float16` with `a = 2^-23` (subnormal) and `b = 2^14`, and 15 products `(2 − 2^-10)·2^-24`, gives
  an error of `53` times `DotF16`'s bound;
- `bfloat16` with `a = 2^-133` and `b = 2^124` gives `13` times;
- normal operands of the same products give `0.007` and `0.014`.

A randomized sweep (`run.py`) reaches `106` times for `float16`. A first probe by the coordinator
showed no loss, because its smaller products were whole multiples of the window's unit; it was
wrong to call the issue closed on it.
- **Kernels.** `matmul_f16`, `attn_fwd_f16`, `attn_bwd_f16` and `grouped_matmul_tma_f16` cite
  `DotF16`. Their inputs, or values they make (`p.to(float16)` in attention), can be subnormal, so
  their certificates rest on a fact that fails on such inputs. The failure needs a subnormal operand
  whose partner is large enough for the product's apparent magnitude to exceed the chunk's real
  largest term; on the host's data (`randn`, `rand`) the chunk's other products are far larger. Their
  reviewers filed it as non-blocking `LIBRARY_REPORT`s.
- **Round 25 (user, 2026-11-01): "please fix L126 after the current run."** After `attn_fwd_e5m2`'s
  re-review (L124) is done. A one-time fix, as rounds 21 to 24.
- **To fix.** Follow L125: count a subnormal operand at the
  smallest normal (`liftMag`) in the facts' magnitudes, or state the 16-bit dot chunk by chunk as
  L124 does, with evidence for the window's width (25 bits below the field, by `probe.py`'s losses).
  Then re-prove and re-review the four kernels.
- **Fixed in round 25** (no library review follows, as for rounds 21 to 24): `DotF16` and `DotBF16`
  keep their names and `relDot16`, `absDot16`, and are now `DotLiftAcc` (`lean/TVA/Rules/Num.lean`):
  `DotAcc`'s bound with `|acc| + Σ liftMag a·liftMag b` in place of `|acc| + Σ|a·b|`, a subnormal
  operand counted at the smallest normal (`minNormal8` now also gives `2^-14` for `float16` and
  `2^-126` for `bfloat16`; `liftMag` and `minNormal8` move to `Rules/Num.lean` under the same names;
  round 24's `Dot8Chunk.LiftAcc` becomes `DotLiftAcc`). Evidence: `evidence/num-dot-sub16-round25`, a
  bit-exact model of the 16-bit tensor-core dot (25 bits below the largest exponent field, the
  accumulator aligned to it, terms and sum truncated toward zero, a subnormal read as the smallest
  normal) equal to the device on about `3.4·10^8` results; largest ratio to the new bound `0.111`;
  the old bound fails up to `113.7` (`float16`) and `14.2` (`bfloat16`) times. `matmul_f16`,
  `attn_fwd_f16`, `attn_bwd_f16` and `grouped_matmul_tma_f16` no longer build until re-proved.
- **Accepted** by the round 25 library review (lr-25a and lr-25b, 2026-11-01).

## L127

**`docs/rules/num.md` ("How a kernel uses them") and L124 call the fed-back accumulator's compounding the device's behaviour, "not slack in the fact"; the window's own model loses at most each truncated term, so the device's error grows at most linearly** (wording; round 24 review (lr-24a), with L124 (lr-24b)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L128

**`evidence/num-dot-chunk/README.md` cites num-dot Part 3 for `input_precision` on fp8, but Part 3 ran only the two unmixed pairs** (wording; round 24 review (lr-24a)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L129

**`evidence/num-dot-scaled/README.md` conclusion 2 extends the scaled-dot facts by num-dot's per-chunk derivation, which round 25 found holds only for normal operands** (wording; round 25 review (lr-25a)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L130

**The opening of `docs/rules/num.md`'s "`tl.dot` (the chunk model)" does not parse, and its "32 products per instruction for fp8" holds on `wgmma` only** (wording; round 25 review (lr-25a, lr-25b)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L131

**`docs/LIBRARY.md`'s "Kernels" section still says "(none yet)"** (wording; round 25 review (lr-25b)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L132

**`evidence/num-dot-sub16/README.md` conclusion 1 says `DotF16` and `DotBF16` do not hold, without "as stated before round 25"** (wording; round 25 review (lr-25b)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L133

**Round 25's list of constants whose meaning changed leaves out the generic `DotChunkAcc` and the `liftMag_*` lemmas, which change meaning at `float16` and `bfloat16` (no kernel uses them there)** (wording; round 25 review (lr-25b)). Round 26. Fixed in round 26 (`round-26/author/notes.md`).

## L134

**The fp8 chunk facts leave a fed-back accumulator `1 + 2^-11` per chunk** (sharpness; library
author, round 26). By the window model a partial value never grows past `|acc| + Σ|a·b|` (no result
in round 26's evidence did). Stating that, with evidence, would remove the remaining compounding.
Open; not scheduled.

## L135

**The sandbox's files name the containers' own paths, which the text scan read as paths out of TVA**
(tool; user, 2026-11-02). The user's goal: the sandbox (`sandbox/`: a review service, an agent harness,
the images and the verifier, in which agents prove and review one kernel in containers) is part of TVA.
Its files run in those containers and name their paths: the test harness's folders, the review
service's folder, the agents' homes and scratch folders, the toolchain's folder. The gate's text scan
(`library_independence`, run by every `tvr check` and by `tvr audit`) read 60 of them, in 14 of the
folder's 26 files, as paths out of TVA, so with the folder in the tree no kernel's gate could pass.
Round 27: the files under `sandbox/` skip the scan's generic absolute-path, relative-path and home-path
tests; the name test and the `fpv/<name>` test apply to them as to every file. Fixed in round 27
(`round-27/author/notes.md`).

## L136

**A write-up that states a claim other than the Lean's could not block the certificate** (protocol;
user, 2026-11-04). In a sandbox run on `paged_attention_f16`, the prover tightened its tolerance in
round 1 (`Xcap` from `2^6` to `2^4`, `tolR`'s last term from `2^-9` to `2^-10`) and left four sentences
of `README.md` with the old constants. Both reviewers found them in round 2, but the only kind for them
was `DOC` (non-blocking), so the certificate was granted with a write-up that told its reader the bound
covers scores up to 64 where the Lean covers 16. `CONDITION_WRONG` covers a condition "not what the
write-up says", but nothing covered the specification's constants, a narrowing, or an assumption.
Round 28: a new kind `DOC_WRONG` (blocking; targets `item:`, `decl:`, `doc:`) for a write-up, or a
docstring of a constant in an item's cone, that states a constant, tolerance, condition, narrowing or
assumption other than its definition; `DOC` stays for wording. Fixed in round 28
(`round-28/author/notes.md`).

## L137

**`tl.sigmoid` has no stated meaning** (gap; sandbox test-set run, 2026-11-05). SEMANTICS.md lists
`sigmoid` among the `tl.*` functions Triton writes in Triton and inlines ("Compiler hints", "Which
value"), but, unlike `tl.cdiv` (L88), states no body for it. In a sandbox run on `gdn_gating_bf16`, a
reviewer filed the model of `tl.sigmoid(blk_b.to(tl.float32))` as `UNSUPPORTED` (blocking): the model
wrote the body `1 / (1 + math.exp(-x))` from `primitives/triton-3.8.0-standard.py`, a fact the page does
not state; the prover referred it to the library. The run on `act_and_mul_bf16` modelled SiLU's
`x * tl.sigmoid(x)` the same way, unflagged. Round 29: SEMANTICS.md states `tl.sigmoid(x)` and
`x.sigmoid()` as the expression, by the page's rules, and that only `fp32` and `fp64` are accepted
(evidence/sigmoid-round29: every `fp32` input bit for bit, the method form, `fp64`, compile-time values;
`bfloat16`, `float16` and `int32` rejected on the host). No Lean constant changed. Fixed in round 29
(`round-29/author/notes.md`).

## L138

**The generated REVIEW.md fails its own kernel's independence check** (tool; sandbox test-set run,
2026-11-05). `tvr` writes each finding's summary into a Markdown table of `REVIEW.md` (and of
`docs/library-review/REPORTS.md`, and the gate's detail into the review packet) with every `|` replaced
by `/`, so that the cell does not split the row. In a sandbox run on `l2norm_fwd_bf16`, a reviewer's
summary held `|x|=2^30`; the page then held `/x/` followed directly by `=2^30`, which `foreign_refs` reads as an absolute path of
two components outside TVA, and the kernel folder's text scan (the gate's `independence` check) failed.
The page is generated from the review service's ledger, which the prover may not edit, so every
submission after the reviewers filed round 1 was rejected on that line. Round 30: the three tables
escape `|` as `\|` (`md_cell`), which Markdown renders as `|` and the scan does not read as a path. The
gate's own code changed, so every gate was stale; `tvr check` was re-run on the twelve kernels with a
gate (all PASS, digests unchanged). No Lean file and no fact page changed. Fixed in round 30
(`round-30/author/notes.md`).

## L139

**A dtype as the argument of a `tl.constexpr` parameter** (gap; fixed claims, 2026-11-08). SEMANTICS.md
("Parameters and arguments") lists `tl.constexpr` arguments that are a Python `int`, `float`, `bool`, `str` or
`None`. vLLM's `scaled_mm_kernel` takes `ACCUMULATOR_DTYPE: tl.constexpr` (`claims/scaled_mm_int8/source/triton_scaled_mm.py:72`),
passed `tl.int32` by the host (line 292) and used as `tl.zeros`'s `dtype=` (line 89) and `tl.dot`'s `out_dtype=`
(line 167). SGLang's `_bmm_chunk_fwd_kernel` takes `dot_dtype: tl.constexpr` (`claims/ssd_bmm_chunk_bf16/source/ssd_bmm.py:109`,
`tl.bfloat16`, the target of `.to(dot_dtype)`), and vLLM's `fused_moe_kernel` `compute_type: tl.constexpr`
(`claims/fused_moe_bf16/source/fused_moe.py:157`). The library should state that such a parameter holds the dtype,
a compile-time value, and where it may be used. Round 31.

## L140

**`tl.broadcast_to` and `x.broadcast_to(shape)`** (gap; fixed claims, 2026-11-08). `scaled_mm_kernel`
writes `scale_a.broadcast_to((BLOCK_SIZE_M, 1))` and `scale_b.broadcast_to((BLOCK_SIZE_N, 1))`
(`claims/scaled_mm_int8/source/triton_scaled_mm.py:179, 184`). SEMANTICS.md mentions `tl.broadcast_to` only inside
other entries (shape rules), with no entry of its own giving its meaning, its method form, and when a shape is
rejected. Round 31.

## L141

**The integer `tl.dot`** (gap; fixed claims, 2026-11-08). `scaled_mm_kernel` computes
`tl.dot(a, b, acc, out_dtype=ACCUMULATOR_DTYPE)` with `int8` operands into an `int32` accumulator
(`claims/scaled_mm_int8/source/triton_scaled_mm.py:167`). docs/rules/num.md lists the integer `tl.dot` under "What the
library does not state" (exact, but not stated: `dotAcc_int` shows it is satisfiable), so a claim must write
`DotAcc` with zero error directly instead of citing a fact. The library should state the exact `int8 × int8 → int32`
`tl.dot` on sm_90 (with or without an accumulator, every `input_precision` the front end accepts for it, the
wrap of the `int32` sum if any) with TVA's own evidence, and list it in "Facts the library states". Round 31.

**Round 31 review** (lr-31a ACCEPT, lr-31b REOPEN): `DotI8` holds (0 wrong of about 2.2 million outputs in both
reviews), but SEMANTICS.md and rules/num.md ("The integer `tl.dot`") say an accumulator other than `int32` is
rejected on the host, and a `uint32` accumulator is accepted: the result is `int32`, the accumulator's bits read as
`int32` plus the sum (3000000000 + 9 gave −1294967287). State that case (a `tl.bitcast .u32 .i32` of the accumulator,
then the dot with ⟨.i8, .i8, .i32, p, 0⟩) with evidence, or exclude it in exclusions.md; correct the two sentences
and evidence/num-dot-int8 (I5, conclusion 1). Round 32.

## L142

**A pointer's `.type.element_ty`; a pointer as an `if` condition** (gap; fixed claims, 2026-11-08).
`scaled_mm_kernel` writes `accumulator.to(c_ptr.type.element_ty)` (line 196; the subset lists `.dtype` and a
dtype's `.element_ty`) and `if bias_ptr:` on a pointer argument that may be `None` or a tensor (line 188;
`semantic.cast` to `int1` is `ptrtoint` then `!= 0`, `semantic.py:900-906`, or a compile-time `None`)
(`claims/scaled_mm_int8/source/triton_scaled_mm.py`). The library should state both forms. Round 31.

## L143

**`tl.cast(x, dtype=…)`** (gap; fixed claims, 2026-11-08). SGLang's `_moe_sum_reduce_kernel` writes
`tl.cast(input_stride_0, dtype=tl.int64)` (`claims/moe_sum_reduce_bf16/source/moe_sum_reduce.py:68-70`). SEMANTICS.md
does not state `tl.cast`; the claim writes it as `tl.convert`. The library should state `tl.cast`'s meaning (its
relation to `.to`, `fp_downcast_rounding`, `bitcast=`) for the dtypes it accepts. Round 31.

## L144

**The framework should not mention CUDA graphs** (rules; user, 2026-11-08: "we verify the kernel itself";
"I do not think we even need to mention CUDA Graph in our claims or in our framework"). `Compiler.Faithful`
is stated about a host call that launches the kernel through Triton, so its scope point 9
(`lean/TVA/Rules/Compiler.lean:118-128`, `docs/rules/compiler-faithful.md`) excludes calls made during CUDA-graph
capture (which run no program) and replays (which are not calls), and `lean/TVA/Claim.lean:30`,
`lean/TVA/Sem/Launch.lean`, SEMANTICS.md, exclusions.md, PROVER_GUIDE.md, REVIEWER_GUIDE.md, REVIEW_PROTOCOL.md and
TRANSCRIPTION.md repeat it. What TVA verifies is the kernel: an execution of its grid of programs on a memory with
given arguments. State the rule (and the claim's notion of a launch) about such an execution, however a caller
dispatches it, so that no dispatch mechanism needs a scope point; remove point 9 and every mention of CUDA graphs,
keeping the rule true (a host call that runs no program is not an execution). Round 32.

**Round 32 review** (lr-32a REOPEN, lr-32b ACCEPT): the restated rule is true and usable, and no framework page names CUDA graphs, but the new sentence "a host call that does not run the grid … (later, or never) … has no outcome" contradicts the one kept beside it, "a launch that returns without an error has an outcome, even if no program ran": the launcher's silent skip of a grid of 2^31 to 2^32 programs falls under both, and the new reading gives it a vacuous `Meets` (the L9 hazard). The contradiction is in `lean/TVA/Claim.lean`, `lean/TVA/Rules/Compiler.lean`, rules/compiler-faithful.md, SEMANTICS.md and REVIEW_PROTOCOL.md §6; and "returns while its programs run later" describes every asynchronous launch. Drop "or never": the skipped grid ends `ok`, and `Launch.Valid` excludes it. Round 33.

## L145

**Programmatic dependent launch as scheduling** (semantics; user, 2026-11-08: "Let's deal with PDL just like warp
specialization. since we trust triton compiler."). Today `Launch.Valid` requires `pdl = false` (exclusions.md,
programmatic dependent launch), and SEMANTICS.md gives `tl.extra.cuda.gdc_wait()` and
`gdc_launch_dependents()` a meaning only without `launch_pdl` (evidence/pdl). Treat PDL as SEMANTICS.md treats
`num_stages` and `warp_specialize`: a launch option and two calls that change when a program's accesses happen, not
what the kernel computes, dropped from the model, with the condition that keeps that true stated as for pipelining
(a launch with `launch_pdl` reads the claim's initial memory, the previous kernel's final memory, in every access
after the program's `gdc_wait()`; what the kernel needs of its accesses before that wait, if any, the author decides
and states). Real uses: SGLang's `_moe_sum_reduce_kernel` on the H200 (`USE_GDC=True`, `launch_pdl=True`;
`gdc_wait()` before any access, `claims/moe_sum_reduce_bf16/source/moe_sum_reduce.py:85-87`) and the LoRA
`_route_histogram_kernel` (`launch_pdl=True`, `gdc_launch_dependents()` at the end and no `gdc_wait()`,
`claims/route_histogram_i32/source/route_histogram.py:143-144`). Closed by the user's decision (2026-11-08: "let's
consider PDL is turned off"): programmatic dependent launch stays outside (`Launch.Valid` requires `pdl = false`), and
the claims cover the launches with it off.

## L146

**Exclude descriptors built inside a kernel** (scope; user, 2026-11-08: "let's exclude this case. we only cover
descriptor outside of triton kernels"). A kernel that calls `tl.make_tensor_descriptor` needs the launcher's
global scratch from the allocator set by `triton.set_allocator` (Compiler.Faithful scope 5), which raised L120 (a
descriptor rebuilt in a loop) and L121 (the allocator's alignment). Exclude every launch whose kernel builds a
descriptor (docs/exclusions.md, SEMANTICS.md "Tensor descriptors", scope 5), keep host `TensorDescriptor` arguments
(scope 8), and close L120 and L121 as settled by the exclusion. Affected claims: `grouped_matmul_tma_f16` (replaced by
tutorial 09's `matmul_kernel_tma_persistent` with host descriptors) and `attn_fwd_f16`'s `warp_specialize = True`
path (narrowed to the host-descriptor path). Round 32.

**Round 32 review** (lr-32a REOPEN, lr-32b ACCEPT): the exclusion is true, complete and consistent (scope 5, SEMANTICS.md, exclusions.md, L37), and host descriptors stay covered; every in-scope feature compiled asked no allocator. Leftover: scope 10(c) no longer names `triton.set_allocator`, but exclusions.md (row 61) still lists it, and the in-scope host path of `claims/attn_fwd_f16` calls it (source line 548): say whether a host that sets an allocator the launch never calls is within scope 10. Round 33.

## L147

**`tl.cast`: what a float narrowing is** (wording; lr-31a). Triton's test is a float target with fewer bits, so `bfloat16 ↔ float16` is rejected with either rounding mode. Say "to a float of fewer bits" and add that pair to the examples. Round 32.

## L148

**A block's `.type` is not a dtype** (wording; lr-31a). `block.to(x.type)` works, but `scalar.to(x.type)` and `tl.zeros(..., dtype=x.type)` are rejected on the host. Say that only a scalar's `.type` and a block type's `.element_ty` may stand where a dtype goes. Round 32.

## L149

**The depth of Triton's reduction and scan trees** (gap; fixed-claims recheck, 2026-11-08). `sum_err_covers`, `sumc_err_covers` and `ScanLeaves` allow any tree over `n` leaves (depth up to `n − 1`). Claims that compose them: `claims/layer_norm_fwd_f16` (at N ≥ 8192 `Post` accepts a mean with a term dropped, a variance with a term dropped, and the unbiased variance) and `claims/moe_router_bf16` (`capTol` about 1000× the kernel's error; a kernel without the soft cap passes on 95–100% of tokens), and, looser than needed, `softmax_f32`, `attn_bwd_pre_f16`, `layer_norm_fwd_bf16`, `chunk_cumsum_f32`, `l2norm_fwd_bf16`. State, with TVA's evidence, the order and depth of Triton 3.8.0's `tl.sum` (each thread's elements in sequence, then the warp's shuffle tree, then across warps through shared memory, by the layout) and `tl.cumsum` on sm_90, as a fact a claim can cite with a depth that depends on the layout, and lemmas that compose it. Round 32.

**Round 32 review** (lr-32a and lr-32b REOPEN): `SumDepth` holds (the reduce-tree evidence re-ran identical; 310 further reductions exact over reshape/permute/join/interleave layouts, 3-D tiles on every axis, transposed wgmma results, axes up to 131072, 32 warps). The `OptimizeThreadLocality` exclusion is wrong in two ways. (1) The pass is not a reassociation when the loop's update is not the reduction's combine: with exact values, `acc - tl.sum` stored +260 for −260 (and 2.0 for −1.0), `max(acc, tl.sum)` stored 11 for 4, `acc + tl.max` stored −inf, `acc * tl.sum` stored 0 — a miscompile, not a rounding. (2) "Produced directly by `tl.load`" is too narrow: the pass reads the IR after folding, and also rewrote loops whose tile was `tl.load(p).to(own dtype)`, `tl.load(p).to(tl.float32)` on float32 data, `x * 1.0` (a constexpr), `x / 1.0`, `x - 0.0`, a block-pointer load, or a load in a helper. State the excluded pattern on the IR the pass sees (a loop-carried value updated by an elementwise operation with a reduction of a tile that folds to a load), call it a miscompile, and keep TVA's evidence for each case; lr-32a checked all 40 claims and none has the pattern on its covered path. Round 33.

**Round 33 review** (lr-33a REOPEN): `SumDepth` stands; the restated operand wording catches every round-32 case and its "folds to a load" test is checkable, but Triton also rewrote (2.0 stored where the source gives 1.0) loops the wording says it does not: an identity between the reduction and the update or after the update (`.to(tl.float32)`, `* 1.0`, `-(-s)`); a second use of the result that is dead (a store under a constexpr-false `if`); a reduction under a compile-time `if` or in a nested `tl.static_range(1)`; a reduction or update inside a helper; a runtime int argument equal to 1 as the initial value (specialised: 3.0 stored for 2.0). State the whole pattern on the IR after folding, specialisation and inlining, or exclude it more broadly (round-33 lr-33a `work/otl_wording33.py`). A static scan of the claims' sources finds no affected claim. Round 34.

## L150

**Sharper 16-bit `tl.dot` facts** (gap; fixed-claims recheck, 2026-11-08). `relDot16 K = (K+2)·2^-22` per step is about 6× the `16·5.25·2^-23` per 256-chunk that the fact's own docstring derives, and a dot that accumulates charges it on the whole `|acc|` every trip. Claims: `claims/lora_shrink_bf16` (on signed data at K = 14336 a dropped K block passes for 95% of outputs), `claims/extend_attn_bf16` (no-prefix bound 0.087 at 2^18 tokens against 0.0078 measured), `claims/streamk_matmul_f16` (a float16-rounded result passes from K ≈ 2048), and `fused_moe_bf16`, `ssd_bmm_chunk_bf16`, `attn_fwd_f16`. State per-chunk facts for float16 and bfloat16 `tl.dot` on sm_90 (both `mma.sync` and `wgmma`) with the accumulator's share bounded per chunk, with TVA's evidence, alongside the present facts. Round 32.

**Round 32 review** (lr-32b REOPEN: soundness; lr-32a ACCEPT on the quick re-run): `DotF16Chunk` and `DotBF16Chunk` are false on `mma.sync` when an operand was converted in registers from fp8 or int8 (the dequantising-matmul pattern): the layout's `kWidth` is then 4 and each instruction adds the k ≡ 0, 1 (mod 4) of a 32-k block, not 16 consecutive k; a per-instruction model with that partition matches the device bit for bit, and a K = 32 counterexample has device result 0 where no partial values satisfy the fact (exact rational check). State the chunks as 32 consecutive k (or the facts with a stated condition on the operands' origin, with evidence), and check whether `kWidth = 8` occurs for 16-bit dots. The fp8 chunk facts are unaffected. Round 33.

## L151

**Pointer minus integer** (gap; fixed-claims recheck, 2026-11-08). vLLM's `rejection_greedy_sample_kernel` writes `cu_num_draft_tokens_ptr + req_idx - 1` (`claims/rejection_greedy_i32/source`, line 91), which Triton's `semantic.sub` builds as `addptr(p, -i)`; `lean/TVA/Lang/Mem.lean` has only `HAdd (Ptr d) (IntV ..)` and SEMANTICS.md states only `+`. State `p - i`. Round 32.

## L152

**`dtype.is_fp8()` and the other dtype predicates** (gap; fixed-claims recheck, 2026-11-08). vLLM's `reshape_and_cache_kernel_flash` branches on `key_load.dtype.is_fp8()` on its fp8 KV-cache path (`claims/kv_cache_write_bf16/source`, lines 167, 176). In Triton 3.8.0 `is_fp8()` is `'fp8' in self.name` (`core.py:449`), a compile-time bool. State it (and `is_fp8e4nv`, `is_fp16`, `is_floating`, …, as the subset needs), and check that the fp8 conversion of an implicit `tl.store` to an fp8 pointer is stated. Round 32.

## L153

**Tutorial 10's `mixed` format runs** (soundness: a false statement; fixed-claims recheck, 2026-11-08). docs/exclusions.md (line 45) says tutorial 10's fp4 launches are rejected on the host. That is true of mxfp4 and nvfp4 (out of shared memory on the H200) but `--format mixed` (e4m3 × e2m1, the `MIXED_PREC` branch) runs at 1024³ with a maximum error of 0.0625, through the stated `DotScaledE4m3E2m1`. Correct the entry. Round 32.

## L154

**Small precision items** (sharpness; fixed-claims recheck, 2026-11-08). (1) A lemma that the alternatives of `tl.ItofRel .binary32 z` coincide for |z| < 2^53 (`claims/seeded_dropout_f32`'s `uRef`; provable now). (2) `AddRel` lets an operand that is not a product be replaced by any exact value rounding to it; on sm_90 only a product enters an `fma` unrounded, so a narrower relation would let `l2norm_fwd_bf16` and `layer_norm_fwd_bf16` cover `eps ∈ [2^-126, 2^-125)`. (3) `tl.exp` overflowing to `+∞` on `float32` and `DivFull` at an infinite divisor (`1/∞ = +0`): `gdn_gating_bf16` could cover every `b`. (4) `TensorDesc.Valid` admits a zero row stride or one shorter than the row; the H200 handled both, but no evidence folder records it. Round 32.

**Round 32 review** (lr-32b REOPEN, lr-32a ACCEPT): the itof lemma, `DivF32InfDivisor` (exhaustive) and the descriptor-stride evidence are correct; narrowing `AddRel` for a non-product operand and the `tl.exp` overflow fact were left undone with no issue tracking them. Do them (the overflow fact after the `Platform.ref` change it needs), or file each as its own issue with the reason. Round 33.

## L155

**`AddRel` with one operand unrounded** (sharpness; library author, round 33, split from L154 (2) as round 32's
review asked). `AddRel fusion x y r` (`lean/TVA/Lang/Ops.lean`) lets, with fusion, *both* operands be replaced by
exact values that round to them, whether or not they are products. On sm_90 contraction makes an addition an `fma`,
which takes one product unrounded and its other operand as it is, so at most one operand of each addition enters
unrounded. A relation with one operand replaced (and the infinite cases as now) is narrower and still covers every
fusion; it would let `l2norm_fwd_bf16` and `layer_norm_fwd_bf16` cover `eps ∈ [2^-126, 2^-125)` (a variance that
rounds to `0` and an `eps` just above `2^-126` cannot then both move down). Not done in round 33 because it changes
the meaning of every fusable float addition (`tl.addc`, `tl.subc`, `tl.sumc` and its axis forms, `tl.cumsumc`, the
scaled dot's addition, `AccAdd`), the library lemmas that take the relation apart (`Lemmas/FusedErr.lean`,
`SumFused.lean`, `Ops.lean`, `Scan.lean`, `Scaled.lean`, `DotLoop.lean`), the kernel proofs that do
(`attn_fwd_f16`, `attn_fwd_e5m2`, `attn_bwd_f16`, `attn_bwd_pre_f16`, `paged_attention_f16`, `softmax_f32`) and the
claims whose models or posts name it (`attn_bwd_pre_f16`, `chunk_delta_h_bf16`, `scaled_mm_int8`); and the narrowing
needs its own evidence that Triton 3.8.0's lowering never feeds two unrounded products into one addition (sm_90 has
no fused two-product instruction for these formats; LLVM's NVPTX contraction and the combine pass's `dot_add`, read
for it). Do it in a round of its own: define the narrow relation, prove it implies today's (`AddRel.wide`), restate
the lemmas through it, and name the kernels that no longer build. Round 34.

## L156

**num-special-f32's conclusion is out of date** (wording; lr-33a). `evidence/num-special-f32/README.md` conclusion 2 says the library states no overflow fact for `tl.exp`; round 33 added `ExpF32Overflow` and the `Platform.ref` change (`mathAt_far`). Name both. Round 34.

## L157

**Relaxed atomics that only accumulate** (coverage; user decision 2026-11-08: a library rule rather than narrowing the claim). `docs/exclusions.md` and review-questions.md Q10 make `sem="relaxed"` (and acquire-only / release-only) atomics on memory other programs touch a race, so the launch is not well behaved and `Compiler.Faithful` constrains nothing. vLLM's LoRA shrink (`_lora_shrink_kernel`, split-K, its default config: split_k 64 below 128 tokens, 8 from 128) adds each part into the output with `tl.atomic_add(..., sem="relaxed")`, a pattern common in hand-tuned kernels. State a rule (Q10's proposal): relaxed read-modify-write atomics whose returned values are unused, on cells that no plain load or store of the launch touches, behave like `acq_rel` ones (PTX atomicity holds for relaxed operations; only their ordering with other accesses differs, and no access observes it). Give its exact conditions (scope `gpu`, the ops covered, what "unused" means on the IR), the evidence (PTX memory model citation; device runs), and update exclusions.md and Q10. `AtomicAddF32`'s numerical part already covers every `sem`. Round 34.

## L158

**`torch.bool` arguments** (wording; claim re-review). SEMANTICS.md does not state that a `torch.bool` tensor argument is a `*i1` pointer, one byte per element, and that `tl.load` of an `*i1` element reads the byte and tests ≠ 0 (Triton's `semantic.py` loads `int8` then compares with 0); the library states it only in the Lean docstrings of `DType.bytes` and `DType.ofBits`. Say it in SEMANTICS.md (arguments; `tl.load`/`tl.store` of `i1`). Round 34.

## L159

**Depth form of the broadcast-product sum lemma** (lemma; claim fix). `sum_bcast_mul_tree_err` takes the any-tree bound; with `SumDepth` a prover must rebuild its composition from `sumc_err_depth` and the products' roundings. State `sum_bcast_mul_depth_err` (depth `Nat.clog 2 n`) beside it. Round 34.

## L160

**The release check without the kernels' gates** (tooling; user decision 2026-11-08: "we do not need 12 certified kernels' shipped gates now. since they are becoming tasks in our benchmark." and "I do not think we need recomputing the gates of all 38 kernels in kernels/"). `tvr release` recomputes the gate of every kernel under `kernels/` and compares it and the certificate with what the kernel ships, one kernel after another; that step is most of the check's time and is no longer needed. Drop the kernel gate and certificate steps from `tvr release` (and the pages and records steps' parts that only follow from them, if any), keeping the library build, `tvr library-check`, the library's records, lint and audit; update REVIEW_PROTOCOL.md's description of the release check. Say what of `kernels/` the check still reads. Round 35.

**Round 35 review** (lr-35a REOPEN, lr-35b ACCEPT): `cmd_release` is right and nothing it still runs needs the dropped steps, but the docs still describe them: REVIEW_PROTOCOL.md §2 says a scratch path fails "the `independence` check of every kernel's gate" (no gate runs there now); the same clause is in docs/prompts/library-reviewer.md:64 and docs/prompts/library-author.md:91; DESIGN.md:159 lists "the gates" among the sandbox's steps. Round 36.

## L161

**A tensor store waits for every lane's atomic** (soundness; lr-34a, round 34, `work/crosswarp_nowait.py`). `Step.defer` treats a whole tensor `tl.load`/`tl.store` as one step, so a store that uses any lane of an atomic's result holds back every later atomic of that program, while on the device each thread orders only its own accesses. Example (a well-behaved launch: only `acq_rel` atomics touch the shared cells, no program waits on another): program 0 does `r = atomic_add(Y + x, 0)`, `store(OUT + x, r)`, `atomic_add(C + x, 1)`; program 1 does `q = atomic_add(C + 511, 0)`, `atomic_add(Y, q)`. The semantics gives no outcome with `q = 1` and `OUT[0] = 1`; the H200 gave exactly that in 200 of 200 launches, with both `acq_rel` and `relaxed` (so `Compiler.Faithful` is false there; the exclusion argument is by hand, no Lean proof yet). The defect predates L157 and is in the `acq_rel` semantics; L157's docstring premise relies on it. No current claim uses an atomic's result. Fix: per-lane, deferrable load and store steps, or exclude launches that use an atomic's result on a cell another program also accesses; prove the counterexample's exclusion in Lean either way. Round 35.

**Decision** (user, 2026-11-08: "let's exclude with Fix B"): keep the model; exclude, stated on the source and checkable like `MemSem.unused`, every launch in which a program uses an atomic's result (in a store, an address, a branch, or any other operation) on a cell that another program of the launch also accesses. Prove in Lean that the counterexample falls under the exclusion; correct L157's docstring premise; update exclusions.md, SEMANTICS.md and review-questions.md (a new question for the per-lane model, to do when a kernel needs atomic results across programs: work queues, tickets, locks). None of the 41 claims is affected (stream-K, the histogram and the LoRA shrink discard their atomics' results; the persistent matmuls use none; the tutorial layer norm's lock is in the backward kernel, outside its claim).

**Decision** (user, 2026-11-08, on round 35's widening): keep the wider exclusion — a used atomic result on a cell that another program, or another atomic operation of the same program (another call, or the same call in a loop), also accesses ("Keep wider").

## L162

**"Unused" on the source and on the IR** (wording; lr-34a). The source and IR definitions of an atomic's unused result disagree for a value read only by a dead pure operation, or returned from a helper whose call is a statement. Both are sound; say which governs `MemSem.unused`. Round 35.

## L163

**Float `/` fused into a following add** (soundness; lr-34b, round 34, `work/`). On float32, and on float16/bfloat16 `/` (computed in float32), ptxas fuses the final multiply of `div.full.f32` (reciprocal times dividend) with a following addition into one `FFMA`. SEMANTICS.md counts only `*`, `tl.dot` and `tl.rand` as products whose following addition may fuse, so it gives that addition its IEEE meaning. On the device, `x / y - q`, with `q` the same quotient stored by an earlier launch, is nonzero on 65536 of 65536 elements, at values no model outcome gives; with fusion off it is 0. `tl.div_rn` and float64 `/` are not affected. It reaches claims: the layer-norm pattern's `var + eps` compiles to `FFMA(rcp(N), sum, eps)`, which `claims/layer_norm_fwd_f16` models as an IEEE addition. Fix: count float32 `/` (and 16-bit `/`) as a product for the fusion rule (the add after it has `AddRel` meaning, with the quotient's own `DivFull` error), with evidence for each spelling (`a / b + c`, `c + a / b`, `- c`, through `tl.where`, `.to`), and list the claims that use `/` followed by an addition so they can be re-checked (about 13 use `tl.div`). Round 35.

## L164

**"One `atom`" per atomic** (wording; lr-34b). Float `atomic_max`/`atomic_min` compile to two predicated integer atomics, at most one of which runs per lane. evidence/relaxed-atomics and SEMANTICS.md should say "one atomic read-modify-write per enabled lane". Round 35.

**Round 35 review** (lr-35a and lr-35b REOPEN): the restated sentence is still wrong. Triton 3.8.0 picks which of the two integer atomics runs from the operand's sign bit (`semantic.py:1326-1333`), so exactly one runs per enabled lane, NaN operands included ("neither with a NaN operand" and "or none" are false; on the H200 `atomic_max(1.0, +NaN)` stored NaN, 7 of 16 NaN-operand lanes wrote the NaN); for `atomic_min` the pair is `min.s32`/`max.u32`. `fBound` still holds (0 of 128 new values outside). Fix SEMANTICS.md "Atomics" and the Part P note of evidence/relaxed-atomics/README.md. Round 36.

## L165

**Wording around the atomic-result exclusion** (lr-35a, round 35). (1) The "No barrier semantics" sentence ("a barrier could only remove behaviours") holds only outside the L161 exclusion: say so. (2) The `relaxed` form of lr-34a's launch is not a counterexample: the semantics already counts it as a race; say which form the exclusion is for. (3) Scope point 12 does not say what a reviewer does when they cannot tell whether a cell is shared: state the rule (treat it as shared). Round 36.

**Round 36 review** (lr-36a REOPEN, lr-36b ACCEPT): parts (2) and (3) are right, but part (1) is fixed only in SEMANTICS.md: its twin, the `tl.debug_barrier` row of docs/exclusions.md (line 8), still says the semantics allows "every order" of two atomics on the same bytes, false inside the exclusion; add the same qualification there. Round 37.

## L166

**`ieee_rounding=True` divisions** (wording; lr-35b, round 35). The library gives `tl.fdiv(x, y, ieee_rounding=True)` no meaning; Triton compiles it to the same IEEE `div.rn.f32` as `tl.div_rn`, which is not fused into a following addition on the device. SEMANTICS.md calls `tl.softmax` a product unconditionally, but with `ieee_rounding=True` it ends in that IEEE division (sound but loose). State both. Round 36.

## L167

**Unqualified "every order"** (wording; lr-36a, round 36). SEMANTICS.md lines 145-157 ("Programs") say without qualification that the semantics performs a program's atomics in every order their dependences allow, the premise L161 found false across threads: qualify it with the Atomics exclusion. Also LIBRARY_REVIEW.md:106 still asks whether a claim can "pass `tvr release`", which no longer reads any claim. Round 37.

## L168

**Softmax formats and atomic wording** (wording; lr-36b, round 36). (1) `tl.softmax` of float16/bfloat16 compiles and divides in float32 (`tl.max` widens 16-bit floats); with `ieee_rounding=True` it is IEEE (0 of 1024 off); SEMANTICS.md's "fails to compile on other formats … So `tl.softmax` …" can be read as saying it fails: state the result for each input format. (2) In "Atomics", "the result is one of the two values" means the new value, not the returned one: say so; and list the atomic dtypes Triton accepts (float16/bfloat16 `atomic_max`/`atomic_min` are rejected on the host). Round 37.

**Round 37 review** (lr-37a REOPEN, lr-37b ACCEPT with the same point as a new item): the atomic dtype lists, the "new value" wording and the float `atomic_and`/`or`/`xor` exclusion are right, but the per-format `tl.softmax` statement leaves out the two fp8 formats, which compile and return `float32` (ending in `div.full.f32` by default, `div.rn.f32` with `ieee_rounding=True`; lr-37a `work/fmt_probe.py`); integer softmax is rejected. Add fp8 there and to the product lists of "Float `+` and `-`" and evidence/div-fusion. Round 38 (the final fix, with no review after it: user decision 2026-11-08).

**Round 38 fix** (FIXED, no review followed): SEMANTICS.md "`/` on floats" and "Float `+` and `-`", evidence/div-fusion and evidence/fdiv-ieee (new Part F, `fmt_probe.py`) give `tl.softmax` on `float8e5`/`float8e4nv` with the 16-bit formats (`float32` result, `div.full.f32` by default, `div.rn.f32` with `ieee_rounding=True`) and integer inputs as rejected.

## L169

**Lane granularity of an atomic's result** (wording; lr-37a, round 37). SEMANTICS.md "Programs" says a step using any lane of an atomic's result "depends on every lane"; "Atomics" and review-questions.md Q7 say it holds the atomic back "as a whole". `Step.defer` is finer: a store masked to lane 0 waits only for lane 0. Use the "Granularity"/Q17 wording. Round 38 (the final fix, no review after it).

**Round 38 fix** (FIXED, no review followed): SEMANTICS.md "Programs" and "Atomics", review-questions.md Q7 and `Step.defer`'s docstring say that a tensor step waits for the atomics of the lanes whose result it uses and, being one step, holds back every later atomic of the program.

## L170

**Evidence describing removed release steps** (wording; lr-37a, round 37). evidence/gate-round7 and evidence/tool-round18 still describe the gate and pages comparisons, removed from `tvr release` in round 35, in the present tense: add a note that the release check no longer runs them. Round 38 (the final fix, no review after it).

**Round 38 fix** (FIXED, no review followed): "Since round 35" notes in evidence/gate-round7, evidence/tool-round18 and evidence/tool-round10, and their conclusions put in the past tense.

## L171

**The release check's steps in LIBRARY_REVIEW.md** (wording; lr-37a, round 37). The gate row lists the build, library, records, lint and audit steps but not the shipped, packages and sandbox steps; and the audit reads the kernels' imports, not only "the text". List them as `cmd_release` runs them. Round 38 (the final fix, no review after it).

**Round 38 fix** (FIXED, no review followed): LIBRARY_REVIEW.md's `gate` row lists the release check's steps as `cmd_release` runs them (shipped, packages, sandbox, build, library, records, lint, audit) and says the audit reads the text and the kernels' Lean imports.

## L172

**16-bit atomics on sm_90** (wording; lr-37b, round 37). review-questions.md Q7 describes the wrong instructions: a single `float16`/`bfloat16` atomic add and a 16-bit `atomic_cas` compile to a 32-bit compare-and-swap loop that rewrites the neighbouring half; contiguous lanes become `.v8` atomics, not `f16x2`. A stress test found no wrong value in 40960 cells (lr-37b `work/`). Correct Q7. Round 38 (the final fix, no review after it).

**Round 38 fix** (FIXED, no review followed): review-questions.md Q7 describes the sm_90 forms (per-lane 32-bit compare-and-swap loop; vector `.v8` atomics; `f16x2` only before sm_90), with new evidence/half-atomics; the trust is unchanged.

## L173

**Atomic operand conversion** (wording; lr-37b, round 37). SEMANTICS.md "Atomics" does not say that an atomic's operand is converted to the pointer's element type (as `tl.store` does), or that `atomic_cas` converts nothing, so mismatched operands are rejected on the host. Say both. Round 38 (the final fix, no review after it).

**Round 38 fix** (FIXED, no review followed): SEMANTICS.md "Atomics" gains "The operand" (converted to the cell's type as by `tl.store`; `atomic_cas` converts nothing, mismatched operands rejected on the host), with evidence/atomic-dtypes `adv.py` part V; the transcription row says the same.

## L174

**S3's max/min pair** (wording; lr-37b, round 37). The sentence "signed max or unsigned min" (S3) is true for `atomic_max` only (`atomic_min` is signed min or unsigned max), and its line references into Triton's `semantic.py` are off. Correct both. Round 38 (the final fix, no review after it).

**Round 38 fix** (FIXED, no review followed): S3 gives both pairs (`atomic_max`: signed max or unsigned min; `atomic_min`: signed min or unsigned max) and cites `semantic.py:1299-1374` and `1271-1275`.
