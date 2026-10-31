# Review of attn_bwd_f16, round 1, by r-bwd

## Verdict

accept
I looked for the line where the model leaves the source, the use the conditions leave out and the size where the tolerance stops meaning anything, and found no blocking problem. The model follows `_attn_bwd`, `_attn_bwd_dkdv` and `_attn_bwd_dq` line by line by SEMANTICS.md (the fusable `tl.subc` after each `tl.dot`, `tl.dot_add` for the three accumulations, the compile-time `if`s, the names assigned numbers as `int32` tensors, the annotated `LN2` built in `float32`, the compile-time `1` of `stride_d` and of `N_HEAD = 1`), every name resolves to the library's operation, and the launch is the host's. The conditions cover the test's and the benchmark's shapes, with each narrowing stated. `Post` bounds every element of `dv`, `dk` and `dq` and keeps the rest of memory; its value premises hold with wide room on the host's own inputs (the forward's `M`, the preprocess's `delta`, `arg_k`), and its tolerance stays at about 1% of the magnitude sums at `N_CTX = 2^14`. The two findings are about wording in the write-up.

## Findings

- **DOC** `doc:kernels/attn_bwd_f16/README.md` — The README's tolerance figures understate Post: about 1%, not 0.5%, of the magnitude sums at the benchmark's largest size, and the per-pair term depends on the inputs
- **DOC** `doc:kernels/attn_bwd_f16/README.md` — The README's layout narrowing misdescribes the host's checks: do.is_contiguous() and equal strides already fix the contiguous strides on every dimension of size > 1

## Accepted

All items.

## Checked

- The model against the source, every mapped line and every drop of the model map (imports, decorators, the two `tl.static_assert`s, which hold at `16`/`32` against `128`), by SEMANTICS.md: the supported subset (statements, annotated names, helpers called by name and keyword, `range(num_steps)` on a name, no name used after a loop that its body defines), the fusable `-` after each `tl.dot`, `tl.dot_add` for `dv +=`, `dk +=`, `dq +=`, the `+0.0` of `tl.where`, the stores' conversions, `.to(tl.float32)` of a `float32` dot, `int32` `//`, `%` and the `int64` casts, the carried names of both loops (with `offs_n` carried unchanged when not `MASK`), and the two copies of each helper (different compile-time arguments, different names).
- Every Lean file of the kernel in full; the model, the launch and `Assumptions` printed with full names (`work/names.lean`, `work/names.out`, `work/names2.out`): every operation is `TVA.*` or the kernel's own helper, the `HAdd`/`HMul` instances are the library's, `dotCfg` keeps the defaults `tf32` and `0`, and `main` uses only `propext`, `Classical.choice` and `Quot.sound`. The proof files cannot change the claim's definitions (they import `AttnBwd.Spec`, which comes first) and declare no instance or notation.
- The kernel lines, host lines and unused functions of §2: the launch runs only the three JIT functions and reads only `triton`, `tl` and the two helpers; the module-level names the host lines bind (`TensorDescriptor`, `configs`, the benchmark's loop names such as `HEAD_DIM`) are not read, `HEAD_DIM` being a parameter of each function.
- The launch: the grid `(N_CTX // 128, 1, BATCH * N_HEAD)` and the `program_id` axes, `fusion = true`, every block shape of the source in `blocks`, and part (A) of `Compiler.Faithful`'s scope (only `num_warps` and `num_stages`, no descriptors, no `CompiledKernel`, no graph capture: `do_bench`).
- The conditions against the host: the asserts of lines 512, 585-586 and 599, `torch.empty_like` outputs, the test's and the benchmark's shapes (largest `G * N_CTX * HEAD_DIM` is `2^28`), `sm_scale` `0.5` and `1.3`, and the `int32` bounds behind `size_lt` and `g_lt`.
- `Post` and its tolerance at the largest covered sizes, by simulating one head of the host's data in float64 with `Spec.lean`'s formulas (`work/premises.py`, `work/premises.out`): the premises of `HeadV.Ok` hold on every `Allowed` pair (largest `xE + eX` 0.044, largest `pBr * yB` 16.2), and at `N_CTX = 2^14` the tolerances are 0.7 to 2.2% of the magnitude sums and 0.6 to 4% of the values (medians over the columns) on rows with real attention, with absolute floors near `4·10^-4` on rows that get little.
- Each `[rule]` field against rules/num.md and rules/compiler-faithful.md: `DotF16` and `DotTotal` with the model's own `DotCfg`, `Exp2F32` and `MathTotal` for `exp2` on `binary32`, `Faithful` for the instance launched.
- The write-up against the Lean (two wording findings).

## Not checked

- Whether the library's facts are true (trusted): `DotF16`'s constants, `Exp2F32`, the fusable forms, the combine pass's rewrites.
- The device run itself: the simulation of the host's inputs is on the CPU in float64, with the forward's `M` computed exactly rather than in `float32`, which moves `xE` by far less than the room the premises leave.
- The proofs' steps, which Lean checked; I read them for names only.
