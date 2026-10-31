# Review of attn_bwd_pre_f16, round 1, by r-bwdpre2

## Verdict

accept
The model is the source of `_attn_bwd_preprocess` (lines 248-262) as SEMANTICS.md reads it, line by line, and every name in the kernel's three Lean files means the library's operation (printed with full names and explicit instances). The launch is the host's (lines 600-607): the 2-tuple grid with axis 2 of size 1, no option, so Triton's defaults, and every block shape of the source. The conditions cover every launch the script makes (the test's largest `(Z·H − 1)·HEAD_DIM·N_CTX = 191·2^19`, the benchmark's `127·2^21`), and each narrowing (power-of-two `BLOCK_M` dividing `N_CTX`, `HEAD_DIM ≤ 256`, the `int32` limits, which are the kernel's own, the grid limits, no overlap of `delta` with `o` or `do`, finite rows only for the value bound) is stated with its reason. `Post` fixes every element of `delta` within a tolerance of at most `4.57·10^-5·Σ|o·do|` at the largest covered `HEAD_DIM`, which rejects a float16 rounding, a 16-bit accumulation and a dropped column, and says every other byte is unchanged. The one assumption is `Compiler.Faithful` in the form the rule page gives, and the launches are within part (A) of its scope. One wording remark only.

## Findings

- **DOC** `doc:kernels/attn_bwd_pre_f16/README.md` — Narrowings, 'The grid': 'the host's largest has Z·H = 192 and 128 · 128 programs' joins two different launches' maxima into one sentence.

## Accepted

All items.

## Checked

- Every Lean file of the kernel read in full (`AttnBwdPre.lean`, `Model.lean`, `Err.lean`, `Proof.lean`): no notation, instance, `open … renaming`, `export` or option that changes a name; `open TVA` (and `TVA.Math.Float`, `Block` in the proof files only); the definitions `R`, `W`, `Q`, `gx`, `gIn`, `gOut` in `Proof.lean` come after the claim's definitions and change none of them. `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)` and one field. `#print axioms main`: `propext`, `Classical.choice`, `Quot.sound`.
- `#print` with `pp.fullNames`, `pp.explicit` and `pp.notation false` of `_attn_bwd_preprocess` (work/Names.lean): every operator resolves to `TVA.IntV.instMul`, `TVA.Tensor.instHAddIntV` (scalar + block), `TVA.Tensor.instHMulIntV_1` (block × scalar), `TVA.instHAddPtrIntV`, `TVA.instHAddPtrTensorIntV`, `TVA.instHAddTensorPtrIntV`; every call is `TVA.tl.*`.
- Line 254: `int32` `program_id(0)` times the constexpr built in `int32` (`BinOp.typing? .mul (.tensor .i32 false) (.int 128)` is `int32`), plus `tl.arange(0, BLOCK_M)` broadcast; lines 255-256.
- Lines 258-259: Python's parse `((O + (off_hz*HEAD_DIM)*N_CTX) + off_m[:, None]*HEAD_DIM) + off_n[None, :]`, each `+` a separate pointer advance by a sign-extended `int32`; the model's `.col`/`.row` broadcast before `*` and `+` gives the same lanes; no mask or `other`; line 259's `.to(tl.float32)` is `tl.cvt` (exact widening) applied to the load's result.
- Line 260: `float16 × float32` computes in `float32` (`BinOp.typing?` evaluated: `opnd = res = f32`, block), so `tl.convert .f16 .f32 o` then `tl.mul`; `tl.sum(…, axis=1)` of a product is `tl.sum1c` with the launch's fusion flag; not the combine pass's `sum_bcast_mul` (the factors are loaded `[BLOCK_M, HEAD_DIM]` blocks, not operands expanded along new axes); a float `tl.sum` does not widen.
- Line 262: `Delta + off_hz*N_CTX + off_m`, no conversion (`delta` is `float32`), no mask.
- Scalar arguments: `Z`, `H` never read, so their class (constant `1`, `int32`, `int64`, `uint64`) is immaterial; `N_CTX` a runtime `int32` (`ctx_lt`), and the constant `1` only when `BLOCK_M = 1`, built in `int32` as an operand of `*` with an `int32` scalar, the same value.
- The map: all 11 entries and the drops of lines 20, 21, 248 (imports binding only `triton` and `tl`, a bare `@triton.jit`). The kernel calls no helper; the source binds `triton`, `tl` and `_attn_bwd_preprocess` nowhere else (searched for assignments, `:=`, `import … as`, `setattr`, `__globals__`); the module-level `HEAD_DIM` bound by the benchmark's loop (line 709) is shadowed by the kernel's parameter; the six other `@triton.jit` functions are run by nothing this launch does, and their headers (no defaults) bind nothing the kernel reads; `kernel.json`'s host and unused classes.
- The source against the pinned tutorial: the only differences are in `_attn_fwd` and its launch, as SETUP.md says; `_attn_bwd_preprocess` and its launch are the tutorial's; no `@triton.autotune` or `@triton.heuristics` on the kernel.
- The launch: grid `![N_CTX / BLOCK_M, Z*H, 1]`; `fusion`, `debug`, `numCtas`, `pdl` at Triton's defaults (the launch gives no option, the decorator no `debug`); the five block shapes are all the source's blocks. Part (A) of `Compiler.Faithful`'s scope: point 4 (no option, no config, no heuristics), 5 (no descriptor built), 6 (no `CompiledKernel`), 8 (no host descriptor given), 9 (the test calls `backward` directly; the benchmark uses `do_bench`, not `do_bench_cudagraph`; no capture in the script).
- The conditions against the host: `o` from `torch.empty_like(q)` and `do` share strides and `do.is_contiguous()` is asserted, so both are contiguous `(Z, H, N_CTX, HEAD_DIM)` with the layout `Inst` states; `delta = torch.empty_like(M)` is a fresh contiguous `float32` `(Z, H, N_CTX)`; `HEAD_DIM` in `{16, …, 256}` (line 512), `N_CTX % 128 == 0` (line 599); every test and benchmark size meets `row_fit`, `head_fit`, `bh_lt`, `grid_lt`; the `int32` limits are the kernel's own (its products wrap beyond them); `satisfiable` is an instance of the test.
- The specification: `OutSpec` indexes row `bh·N_CTX + m` of `O`, `DO` at `r·HEAD_DIM + k` and element `r` of `Delta`, as the kernel addresses them; `Post` covers every `(bh, m)` and keeps every other byte; the tolerance re-evaluated exactly (work/tol.py): `46u, 94u, 190u, 382u, 766u` relative and `76η … 1276η` absolute for `HEAD_DIM = 16 … 256`, matching the README's table, and below `2^-14·B + 2^-138`; the README's float16-rounding example (`9.77·10^-4` error against `7.25·10^-4` tolerance) and a dropped column are rejected; finite inputs give `Σ|o·do| ≤ 2^40`, so `c` finite is meaningful.
- The assumption `faithful`: `∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`, the rule page's form; its docstring says what the rule says.
- The write-up against the Lean: the conditions table, the specification, the tolerance table, the proof outline and the narrowings; line references to the source checked.

## Not checked

- Part (B) of `Compiler.Faithful`'s scope (the environment), which the protocol assumes and the reviewer does not check.
- The library's definitions, lemmas (`sum_bcast_mul_tree_err`, `meets_independent`) and SEMANTICS.md's facts, which kernel review trusts; Lean checked the proofs, which I did not re-read for correctness beyond what the names mean.
