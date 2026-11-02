# Review of paged_attention_f16, round 1, by r-pa2

## Verdict

accept-partly
The conditions, the specification and the six `[rule]` assumptions are right, and I accept them.
I do not accept `model` and `map`, for one reason. `Inst.ok` covers `PAGE_SIZE = 1`. There the
reduction at line 68 has a single leaf, so its value is the product `p*v` itself, and line 68 adds
two products. The library's fusion rule then allows the device to contract either product, but
`tl.mul_add` allows only one of the two contractions. Every other line, the drops, the launch, the
token mapping through the block table, the masked last page and the integer arithmetic match the
source on the covered instances. The fix is cheap: narrow to `PAGE_SIZE >= 2`, which the test's
16 and 32 meet, or write line 68 with `tl.addc`. The other findings do not block: a library report
that ties this kernel to the open issue L123, and two wording points.

## Findings

- **MODEL_MISMATCH** `map:68` — At PAGE_SIZE = 1 (covered: b = 0) line 68 adds two products, since the one-leaf tl.sum is p*v itself: the addition is tl.addc, not tl.mul_add
- **LIBRARY_REPORT** `src:65` — Open library issue L123 (is a tl.where or tl.maximum of a product a product?) decides lines 64 and 65, which the model writes as IEEE tl.sub
- **DOC** `decl:Kernels.PagedAttention.model` — The stride-equal-to-1 argument says every use of a stride multiplies an int32 tensor; line 59's `page * stride_kp` multiplies the int64 `page`
- **DOC** `doc:kernels/paged_attention_f16/README.md` — The write-up calls the length seq_lens[s], but the kernel reads element s of seq_lens's storage, and the wrapper does not assert seq_lens is contiguous

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

- Every Lean file of the kernel, read in full: `PagedAttention.lean`, `Model.lean`, `Facts.lean`, `Program.lean`, `Real.lean`, `Lane.lean`, `RowMax.lean`, `Final.lean`; in `RowAcc.lean` and `Step.lean`, every definition (`psum`, `wt`, `relE`, `psiPr`, `psiLr`, `psiAr`, `stepR`, `absLr`, `absAr`, `ffac`, `dAl`, `bw1`, `bwg`, `LInv`, `AInv`) and the theorems' statements. They contain no instance, notation, `open … renaming`, `export`, local attribute or option beyond printing. The only `open` is `TVA TVA.Math.Float`, and no kernel name shadows a library operation. `work/names.out` prints the model, `model`, `launch`, `Post`, `SeqOk`, `Pre` and `Assumptions` with full names: every operation is `TVA.*` or a kernel definition I read. `#print axioms` of `main`, `consistent` and `satisfiable` gives `propext`, `Classical.choice`, `Quot.sound`.
- The model against the source, every map entry, by SEMANTICS.md:
  - Lines 42–45. `program_id` axes; `head // QUERIES_PER_KV` is `tl.idiv` with the constexpr built in `int32`; the scalar load of `seq_lens`.
  - Line 49. Two pointer additions of `int32` products (no wrap by `prods_lt`), then the block; the exact `.to(float32)`.
  - Lines 51–52. Names assigned `float("-inf")` and `0.0` hold `float32` scalar tensors (`litF32`); the loop carries them as `float32`.
  - Line 54. `range(0, tl.cdiv(...))` is `(seq_len + (PAGE_SIZE − 1)) // PAGE_SIZE` in `int32`, evaluated once, with an `int32` induction variable. It carries `acc`, `l_i`, `m_i`; neither `j` nor a body-defined name is used after the loop.
  - Line 57. One `int32` element, sign-extended to `int64`.
  - Line 58. The signed `int32` comparison.
  - Line 59. Python's left-to-right order and promotions (`page * stride_kp` in `int64`, the `int32` products converted), broadcasting written at `[PAGE_SIZE, HEAD_DIM]`.
  - Lines 60 and 67. `mask[:, None]`; `other=0.0` as the `float32` `+0` converted to `float16`.
  - Line 61. `tl.sum1c` of the product, then the IEEE product by the runtime `float32` `scale` (`hostF32`).
  - Lines 62–66 and 68–69. `tl.where` with the `float32` `−∞`, `maxAll`, `maximum`, `sub`, `exp`, the IEEE `tl.sum` of `p` (an `exp`, not a product), `mul_add`, `sum0c` (the finding concerns `PAGE_SIZE = 1` only).
  - Lines 71–72. `tl.div`; `tl.cvt` to `float16`; the unmasked store.
  - The drops of lines 26, 27 and 30 are right. The ranges are right: the launch runs only `paged_attention_kernel`, and reads only the module-level names `triton` and `tl` (lines 26–27). The host lines bind no name it reads (its `q` and `out` are locals assigned before use). Nothing is outside the supported subset.
- The launch: grid `(num_seqs, num_heads, 1)`, `fusion = true`, `debug = false`, `num_ctas = 1`, `pdl = false`. The blocks list every block shape of the source (`[P]`, `[D]`, `[P,1]`, `[1,D]`, `[P,D]`). The wrapper stays within part (A) of `Compiler.Faithful`'s scope. Strides equal to `1` become constants with the same value at each use.
- The conditions against the wrapper and the test:
  - The three configurations (lengths 1–8191, contiguous tensors, distinct pages, unused table entries `0`) meet `Inst.ok` and `Pre`, with the README's stride and product figures.
  - `SeqOk` allows shared pages and unset entries past a sequence's last page, as real use needs.
  - The `int64` page offset needs only `kv_len`, and the `int32` products and `seq_len + PAGE_SIZE − 1 ≤ 2^24 + 255` do not wrap.
  - Each narrowing is stated with its reason.
- The specification:
  - Token `t < seq_lens[s]` is row `t % PAGE_SIZE` of page `block_tables[s, t // PAGE_SIZE]`, key/value head `h // (num_heads // num_kv_heads)`, read from `init`. So the last, partial page counts only its tokens.
  - The reference is exact softmax attention with the double `scale`.
  - The output is finite under `SeqVals`, and within `tolO·attnAbs + tolF` for every `X ≤ 2^10` that bounds `scale·Σ|q·k|`.
  - Every byte outside `out`'s elements is unchanged.
  - It rejects a wrong page, row, head or mask, or a dropped partial page, unless the effect is below about `10^-3·attnAbs`.
- The tolerances, recomputed (`work/tol.py`, `work/tol.out`):
  - `tolO` at the README's rows: `1.2450e-3`, `1.2488e-3`, `0.8607e-3`, `1.9000e-3`, `0.17429`; `tolF = 2^-25`.
  - `5.81e-2` at `T = 2^16`, `X = 11`.
  - Below 1 at every covered size.
- The test's inputs, rebuilt on the CPU (`work/testdata.py`, `work/testdata.out`): largest `scale·scoreAbs` `10.9601` (mean `7.156` at 128), largest `attnAbs` `3.8691`, at most 256 pages, bound at most `4.83e-3`. All as the README states.
- Each `[rule]` assumption is the fact docs/rules/num.md lists for the operation used:
  - `ExpF32`, `ExpF32NegInf`, `MathTotal .exp .binary32`: `tl.exp` on `float32`, lines 64–65;
  - `DivFull`, `DivTotal .binary32`: `/` on `float32`, line 71;
  - `Faithful` under `∀ d, d.ok →`.

  There is no `[kernel]` assumption, and `Assumptions` has exactly the parameters `(K, P)`.
- The write-up against the Lean: it agrees, apart from the two DOC findings.

## Not checked

- The proofs' internal steps beyond Lean's check (I relied on the gate's build and replay, and printed the axioms).
- Whether the bounds are tight; only that they match the README and are meaningful.
- The library (SEMANTICS.md, `Compiler.Faithful`, the device facts), which I trusted, and part (B) of the rule's scope.
- Nothing on a GPU: the kernel's outputs on the test were not recomputed (the README's `4.9e-4` is the coordinator's run).
