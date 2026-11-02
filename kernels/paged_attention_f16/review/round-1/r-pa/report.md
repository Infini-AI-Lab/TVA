# Review of paged_attention_f16, round 1, by r-pa

## Verdict

accept-partly
The conditions, the specification and the six `[rule]` assumptions are right, and I accept them. The conditions cover the wrapper's launches as the test makes them, and state each narrowing. The specification reads every token `t < seq_lens[s]` from page `block_tables[s, t // PAGE_SIZE]`, row `t % PAGE_SIZE`, of key/value head `h // QUERIES_PER_KV`, as memory holds it at launch. Its tolerance is meaningful: `1.25e-3·attnAbs + 3e-8` at the test's sizes, `0.174` at the largest covered sizes. The model matches the source on every line except one covered case. At `PAGE_SIZE = 1`, which `Inst.ok` admits, line 68's one-lane `tl.sum` is the product `p·v` itself. The `+` then adds two products, which the library writes `tl.addc`, not `tl.mul_add`. Requiring `PAGE_SIZE >= 2` settles it at no cost to real use. Until then I do not accept `model` and `map`. The two DOC findings and the library report do not block.

## Findings

- **MODEL_MISMATCH** `map:68` — At PAGE_SIZE = 1 (covered: page_size admits b = 0) line 68 adds two products, which the library writes tl.addc; the model's tl.mul_add leaves out a fusion the device may do.
- **DOC** `decl:Kernels.PagedAttention.Inst` — Inst and the README take seq_lens contiguous (element s at offset s), but the wrapper does not assert seq_lens's stride; state the narrowing.
- **DOC** `decl:Kernels.PagedAttention.model` — The stated reason why a stride equal to 1 changes nothing is wrong for line 59's page * stride_kp, whose other operand is int64; the conclusion still holds.
- **LIBRARY_REPORT** `lib:docs/SEMANTICS.md` — The fusion rule's 'a product ... through any dataflow' does not say whether a tl.where of a product, or a reduction over one element, is still a product.

## Accepted

- `item:conditions`
- `item:spec`
- `item:assumption:faithful`
- `item:assumption:exp_acc`
- `item:assumption:exp_ninf`
- `item:assumption:exp_total`
- `item:assumption:div_full`
- `item:assumption:div_total`

## Checked

- **Lean files.** I read every Lean file of the kernel in full: `PagedAttention.lean`, and `Model`, `Real`, `Lane`, `RowMax`, `RowAcc`, `Step`, `Final`, `Facts` and `Program` under `PagedAttention/`.
  - There is no `instance`, `notation`, `attribute`, `export`, `set_option`, renaming or local binder named like a library name.
  - Every `def` is in `Kernels.PagedAttention`, with a docstring that says what it is.
  - `work/names.lean` prints the model, `model`, `launch`, `Assumptions` and `Spec` with full names: every operation is `TVA.*` or one of the kernel's own definitions (output in `work/names.out`).
  - `Assumptions` is a structure with exactly the parameters `(K : Claim.Behaviour Inst) (P : Platform)`.
  - `#print axioms` shows only `propext`, `Classical.choice` and `Quot.sound` for `main`, `consistent` and `satisfiable`.
- **The model against the source**, line by line by SEMANTICS.md, every map entry and drop:
  - Lines 26–27 are the only bindings of `triton` and `tl`, in the kernel lines. Line 24's `torch` and the host functions bind nothing the kernel reads. The kernel reads no other module-level value, and uses only `float` and `range` among the builtins.
  - Every form in lines 30–72 is in the supported subset: a bare `@triton.jit`, no defaults, a two-argument `range`, `float("-inf")`, the `[:, None]` and `[None, :]` subscripts, and no name used after the loop that was first defined in it.
  - Line 44: `//` by the constexpr, built in `int32`, is `tl.idiv`. Line 45 is `tl.load1`.
  - Lines 51–52: a name assigned a Python float holds a `float32` scalar (`-inf`, `+0`).
  - Line 54: `tl.cdiv` is `(x + (PAGE_SIZE - 1)) // PAGE_SIZE` in `int32`, evaluated once. The induction variable is `int32`, and `RangeOk` holds (bound at most `2^16`).
  - Line 57: the `int32` entry is sign-extended to `int64`.
  - Line 59: Python's left-to-right order. `page * stride_kp` is `int64` (stride widened). The two `int32` products are widened for the additions, and broadcasting is written early with the same values.
  - Lines 60 and 67: `mask[:, None]`. `other=0.0` is a `float32` constant converted to `float16`, then widened (exact).
  - Line 61: the product feeds the reduction (`sum1c`), then the IEEE product by the `float32` scale. Line 62: the `-inf` of `tl.where` is built in `float32`.
  - Line 63: `tl.maxAll true`, then `tl.maximum` (`MaxRel`).
  - Lines 64–65: IEEE `tl.sub`, under the library's tl.add docstring: a select's result is not a multiplication's (see the LIBRARY_REPORT). Then `tl.exp`.
  - Line 66: IEEE `tl.sum` of `p` (an `exp` result), then `tl.mul_add`.
  - Line 68: `sum0c` and `tl.mul_add`, right for `PAGE_SIZE >= 2` (finding).
  - Line 71 is `tl.div`. Line 72 is `tl.cvt` (round to nearest even) and `tl.store` without a mask.
  - The masked lanes: `s = -inf`, `p = exp(-inf) = +0`, and `v = 0` by `other`, so `p·v = 0`.
  - The first trip: `alpha = exp(-inf - m) = +0`, `l_i = +0`, `acc = 0`.
  - Every trip has lane 0 unmasked, since `j·PAGE_SIZE < seq_len`, so `m_new` is finite on covered values.
  - A stride equal to `1` is a compile-time constant, and the model's value is the same at every use (DOC on the stated reason). The `combine` rewrite `sum_bcast_mul` does not apply: these are rank-2 products of a loaded block and a broadcast vector.
- **The launch.**
  - The grid is `(num_seqs, num_heads, 1)`, with `program_id` axes 0 and 1.
  - `blocks` lists every block shape of the source: `[P]`, `[D]`, `[P,1]`, `[1,D]`, `[P,D]`.
  - Fusion is Triton's default; `num_ctas = 1`; `debug = false`; `num_warps = 4` is performance-only.
  - Part (A) of Faithful's scope, points 4, 5, 6, 8 and 9: no other option, no descriptors, no `CompiledKernel`, no host descriptor, no graph capture.
- **The integer arithmetic at the largest covered sizes.**
  - `prods_lt` bounds every `int32` product of lines 49, 57, 59 and 72: `seq·stride_qs`, `head·stride_qh`, `seq·stride_bs`, `r·stride_kt`, `kv_head·stride_kh`, `seq·stride_os` and `head·stride_oh`.
  - The other `int32` values stay below `2^24 + 2^8`: `j·PAGE_SIZE + offs_t` (line 58) and `seq_len + PAGE_SIZE - 1` (line 54).
  - The `int64` offset is below `num_pages·stride_kp + …`, which is below `k_cache.len`, which is below `2^63` (`kv_len`, `Buf.Valid`). So the cache may hold more than `2^31` elements.
  - The strides stay below `2^31`, so they are `int32` arguments.
- **The conditions against real use.**
  - The wrapper's asserts (lines 78–82) and `empty_like(q)` (`out_inj`, `disj`) match the conditions.
  - The test's three configurations and `make_cache` (distinct pages below `num_pages`, the rest of the table `0`) meet `Inst.ok` and `Pre`.
  - `SeqOk` lets pages be shared between sequences, and puts no constraint on entries past a sequence's last page. It bounds `seq_lens[s]` to `[0, max_pages·PAGE_SIZE]`, which in-bounds block-table reads need.
  - The narrowings are stated with their reasons: `HEAD_DIM` and `PAGE_SIZE` at most `256`, `max_pages <= 2^16`, `prods_lt`, the grid limits, the range of `scale`, and `SeqVals`'s finite values, `|v| <= 2^15` and `X <= 2^10`. `seq_lens`'s stride is the one gap (DOC).
- **The specification.**
  - `attnRef` sums over `t < seqLen` exactly, the last partial page included, with keys and values through `tokElem`.
  - Every element of `out` holds a `float16` value. Every byte outside `out`'s written elements is unchanged.
  - A sequence with `seq_lens[s] = 0` gets only "holds a value". The kernel writes `0/0` there, and the README says so.
  - Plausible wrong kernels are rejected: a wrong page or row, a wrong key/value-head map, an unmasked partial page, a missing `alpha` rescale.
- **The tolerance.** `work/tol.py` evaluates the definitions with 80-digit decimals (`work/tol.out`).
  - Every README table row reproduces: `1.2450e-3`, `1.2488e-3`, `8.607e-4`, `1.900e-3`, `0.17429`; `tolF = 2.98e-8`.
  - At `2^16` pages with `X = 11`, `tolO = 5.8e-2`. With `X = 50` it is `6.0e-2` (note in the filing).
- **The test's data.** `work/test_data.py` regenerates the test's inputs on the CPU (seed 0; output in `work/test_data.out`).
  - The largest `scale·scoreAbs` is `10.960`. The mean is `7.156` at configuration 1 and `6.845` at configuration 3 (head_dim 128), and `5.39` at head_dim 64, whose largest is `9.04`, so the README's `X = 10` row holds.
  - The largest `attnAbs` is `3.869`, and the largest number of pages is `256`. These match the README's `10.96`, `7.2`, `3.87` and `256`.
  - With each head's own `X`, the per-element bound is at most `3.42e-3`. With `X = 11`, `1.245e-3·3.87 + 3e-8 = 4.82e-3`, at most the README's `4.9e-3`.
  - A `float32` emulation of the kernel's loop (sequential sums, no `fma`) differs from exact attention through the block table by `4.88e-4`, `4.86e-4` and `2.43e-4`. That is at most `0.69` of the bound on every element, and it agrees with SETUP.md's GPU figures (`4.9e-4`, `4.9e-4`, `2.4e-4`).
- **The rules.** I checked each `[rule]` against `docs/rules/num.md`: `ExpF32`, `ExpF32NegInf` and `MathTotal .exp .binary32` for `tl.exp` on `float32` (lines 64–65), and `DivFull` and `DivTotal .binary32` for `/` on `float32` (line 71). `Compiler.Faithful` is cited as `Faithful P (model d) (launch d) (K d)` under `d.ok`. There is no `[kernel]` assumption.
- **The write-up.** README.md matches the Lean: the specification, the conditions table, the assumptions, the proof outline, the narrowings and the modelling decisions. The two DOC findings are the exceptions.

## Not checked

- **The proofs' inner steps.** I read the files for what their names mean, and left the bounds of `RowMax`, `RowAcc`, `Step`, `Final` and `Real` to Lean.
- **The GPU figures.** SETUP.md's GPU run and the coordinator's test with more than `2^31` elements: no GPU, by the protocol. I compared them only with the CPU emulation.
- **The library.** Whether the cited library facts are true (`ExpF32`'s constants, `DivFull`), and part (B) of `Compiler.Faithful`'s scope. Both are outside a kernel reviewer's scope.
