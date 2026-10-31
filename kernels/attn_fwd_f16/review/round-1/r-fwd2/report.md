# Review of attn_fwd_f16, round 1, by r-fwd2

## Verdict

accept
I found no line where the model and the source part ways. I read the source next to the model, all
76 map entries and the drops, and printed the model with full names. The conditions cover every
shape and input the wrapper, the test and the benchmark use, and each narrowing is stated with its
reason. `Post` fixes both outputs and the frame, with tolerances that I recomputed and that still
mean something at the largest covered case. The ten `[rule]` fields are the library's facts for the
model's operations. My two findings do not block: wording in the write-up, and a library report on
how the fusion rule should be read.

## Findings

- **DOC** `doc:kernels/attn_fwd_f16/lean/AttnFwd.lean` — stale line numbers for the test and the benchmark in the module docstring; `testInst` called the test's shape with `H = 1`; two different estimates of the benchmark's scores.
- **LIBRARY_REPORT** `lib:docs/SEMANTICS.md` — "through any dataflow" does not say whether a `tl.maximum`/`tl.where` result that carries a product's value counts as a product (lines 84 and 241 rely on the reading that it does not).

## Accepted

All items.

## Checked

- The supported subset, line by line, for every line the launch runs (`_attn_fwd`, `_attn_fwd_inner`, `_maybe_make_tensor_desc`, their decorators, lines 20–21). The kernel and helpers are bare `@triton.jit`. There is no autotune left. The `tl.constexpr` annotations on helper parameters that receive tensors (`offs_m`, `offs_n`, `N_CTX`) change nothing (SEMANTICS, "Helpers are inlined"). `dtype` holds a dtype. The two calls of `_attn_fwd_inner` have different `STAGE` values, so their copies have different names. Each helper has one visited `return`. Neither the loop variable nor a body-only name is used after the loop.
- The module-level names: the launch reads only `triton`, `tl` and the two helpers, each bound once. The module-level `for HEAD_DIM … warp_specialize` loop (lines 709–714) binds globals, but the kernel reads parameters of those names. The headers of the unused functions bind nothing the kernel reads. The function classes in claim.md §2 are right: the pre-hook, `grid` and `alloc_fn` run on the host only.
- The model against SEMANTICS, line by line. The `int32` index arithmetic and its bounds. Lines 55–62: the Python `0` becomes an `int32` tensor. The constants: I checked the bit patterns of `inf`, `1.0`, `1.44269504` and `-1e6`. `tl.where`'s constants are `float32`. The fusable forms are at lines 76, 81 and 106, and the IEEE forms at 78, 80, 84, 85, 95, 214, 215, 219 and 241. `cfg16` uses the default precision and `maxImpreciseAcc = 0`. The order of the four descriptor calls matches the source, and so does `descArg` (the host descriptor with the pre-hook's block shape, or the in-kernel one). Hints: line 59 is checked where `lo` is defined, and line 70, on the induction variable, is dropped. The dropped dead or config lines are 64–66, 87–94, 97–99, 187, 188 and 197–200. Integer arguments equal to `1` (`Z = 1` in the test) give the same values.
- The launch: the grid formula, the program-id mapping, every source block shape in `blocks`, and the defaults for fusion, debug, `num_ctas` and `pdl`. Scope (A) of `Compiler.Faithful`. Point 4: only `num_stages=2` and `num_warps=4`, and `extra_kern_args` is empty on the H200. Point 5: `alloc_fn` returns a fresh `torch.empty`. Points 6, 8 and 9 also hold: `do_bench` captures no graph.
- The full-name print of the model, `launch`, `descArg`, `Inst.ok`, `Pre`, `Post` and `Assumptions` (`work/print_names.lean`). `#print axioms` for `main`, `consistent` and `satisfiable`. Every command of every kernel Lean file: no instance, notation, attribute or option beyond the linter.
- The conditions against the host. `HEAD_DIM` is 64 or 128; 16 and 32 are rejected by line 188. The `N_CTX` values all meet the condition. `Z·H ≤ 192` and `Z·H·N ≤ 2^21`. `sm_scale` is 0.5 or 1.3. Both values of `causal` and `warp_specialize` are covered, and the tensors are fresh and contiguous. For `HeadOk` on the host's inputs, a Chernoff bound gives `σ·Σ|q·k|` at most about 23 (test) and 240 (benchmark), against `2^10` (`work/scores.py`).
- `Post`: both outputs, the frame outside `outBytes`, and the reference (exact softmax over `keys`, causal `j ≤ n`). `M` is the base-2 log-sum-exp. I recomputed the tolerances exactly (`work/tol.py`). The README's table is right; at the largest case `tolO` is 0.236 and `tolM` is 0.194. I checked that `Post` rejects plausible wrong kernels: a dropped block, the wrong base of `exp`, natural-log `M`, no `alpha` rescale.
- Each `[rule]` field against num.md's table: `DotF16`, `DotTotal cfg16`, `Exp2F32`, `Exp2F32NegInf`, `MathTotal exp2`/`log2` on `binary32`, `Log2F32`, `DivFull`, `DivTotal binary32`. There is no `[kernel]` field. `Assumptions` has exactly `(K) (P)`.
- The README and the docstrings against the Lean.

## Not checked

- The proof steps of `Real.lean`, `RowAcc.lean`, `RowMax.lean`, `Step.lean` and `Final.lean`, which I read for their declarations and commands but not step by step. Lean checks them, and none of them enters the claim's statement: `Model.lean` imports only `TVA`.
- The host input estimates assume iid normal entries. They are not a run of the script's random streams.
