# Review of attn_fwd_f16, round 1, by r-fwd

## Verdict

accept
The model matches `_attn_fwd`, `_attn_fwd_inner` and `_maybe_make_tensor_desc` line by line on every covered instance, by SEMANTICS.md: the compile-time branches on `STAGE`, `dtype`, `FP8_OUTPUT`, `IS_HOPPER` and `isinstance`; the Python numbers `0`, `1.0`, `float("inf")`, `1.44269504` and `tl.where(mask, 0, -1.0e6)` built in `float32` (bit patterns checked); the three additions a product feeds (lines 76, 81, 106) written as `tl.mul_add`/`tl.subc`, every other `+` and `-` IEEE; the hints of lines 59 and 70; both kinds of descriptor argument; the launch's grid, block shapes and defaults. Every drop holds on covered instances, the kernel lines hold every function and module-level name the launch reads, and the host lines bind none of them. The conditions cover every shape, dtype, layout and value of the test and the benchmark, and each narrowing is stated with its reason. `Post` says what the kernel is for (the causal or full attention output, the base-2 log-sum-exp, nothing else written), and its tolerance stays below 1 at the largest covered sizes (23.6% of `attnAbs` at `N = 2^16`, `D = 256`, `X = 2^10`; 3.1–3.5% at the benchmark, 0.3–0.5% at the test). The ten `[rule]` fields are the library's facts for the operations used. My findings are two wording points and one library wording report; none blocks.

## Findings

- **DOC** `doc:kernels/attn_fwd_f16/lean/AttnFwd.lean` — the module docstring gives wrong source ranges for the test (600–650, really 631–695) and the benchmark (655–705, really 698–775).
- **DOC** `doc:kernels/attn_fwd_f16/README.md` — the estimates of `σ·Σ|q·k|` for the host's inputs differ from `HeadOk`'s docstring and neither gives an argument; a Cauchy–Schwarz and χ² bound gives at most 36 (test) and 375 (benchmark).
- **LIBRARY_REPORT** `lib:docs/SEMANTICS.md` — "Performance-only arguments" calls `num_stages` and `warp_specialize` "given by a model", which `Launch`, `tl.forRange` and "Loops" contradict; the library should say that a model omits them.

## Accepted

All items.

## Checked

- The whole source against the model map: every entry of `model-map.json` (76) against SEMANTICS.md, including each drop (lines 20, 21, 47, 64–66, 70, 87–94, 97–99, 168, 176, 187, 188, 197–200) and the condition that makes it hold.
- Every Lean file of the kernel: `Model.lean` and `AttnFwd.lean` in full, and every declaration, `open`, `namespace` and `set_option` of the eleven proof files (no instance, notation, attribute, export or renaming anywhere; the only option is `linter.unusedVariables` on `attn_fwd`). The proof files import `Model.lean` and cannot change what the claim's names mean.
- With `pp.fullNames` (`work/names.lean`): every operation of `attn_fwd_inner`, `attn_fwd`, `maybe_make_tensor_desc`, `model`, `launch` and `descArg` is `TVA.*` or a kernel helper; `cfg16` and the `dot_total` configuration take the defaults `tf32` and `0`; `Launch` keeps `fusion`, `debug`, `numCtas` and `pdl` at their defaults; `Assumptions` is a structure with exactly `(K, P)`; `main`'s axioms are `propext`, `Classical.choice`, `Quot.sound`.
- The float constants' bit patterns (`-1e6`, `+∞`, `1.0`, `1.44269504`, `0.5`) against Python's doubles.
- The fusion forms: which additions a product feeds (lines 76, 81, 106) and which not (lines 78, 84, 85, 214, 215, 241); no `tl.dot_add` rewrite applies (each dot's result is scaled before any addition, and line 103 has its own accumulator).
- The launch: grid `(cdiv(N_CTX, 128), Z·H, 1)`, the eight block shapes (complete for the source), `num_stages=2`, `num_warps=4`, no `maxnreg` on the H200, a bare `@triton.jit`; `Compiler.Faithful` scope (A) points 4, 5 (the allocator's fresh `torch.empty`), 6, 8 and 9 (`do_bench`, no graph capture).
- The scalar arguments: `Z = 1` (the test) and `H = 1` as the compile-time `1` give the same model; `N_CTX ≥ 128`; `sm_scale` as `hostF32` of the double.
- The supported subset and the module-level names: no `@triton.autotune` left, no rebinding of `triton`, `tl`, the helpers, `float` or `isinstance`; the unused functions' headers bind only their own names; the module-level loop's `HEAD_DIM` and `warp_specialize` are shadowed by parameters.
- The conditions against the test (`Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, both `causal`, `warp_specialize = False`) and the benchmark (`Z = 4`, `H = 32`, `N_CTX = 2^10 … 2^14`, both `causal`, `warp_specialize ∈ {False, True}` for the non-causal forward on Hopper); `HeadOk`'s value premises against their inputs (`work/headok.py`).
- The tolerances evaluated exactly with fractions at the test's, the benchmark's and the largest covered sizes, and the README's table of values (`work/tol.py`, `work/tol.out`).
- Each `[rule]` field against `docs/rules/num.md`'s "Facts the library states" and the operation the model uses (`DotF16`/`DotTotal` at lines 73 and 103, `Exp2F32` and `Exp2F32NegInf` at lines 82 and 84, `Log2F32` at line 241, `DivFull` at line 242, the totality rules).
- The write-up against the Lean: the Specification, the Conditions table, the narrowings, the assumptions and the outline of the proof.

## Not checked

- The proofs line by line (Lean checks them); I read their statements and declarations only.
- Whether a covered launch compiles (shared memory at `HEAD_DIM = 256`, warp specialisation on Hopper), which is outside every claim.
