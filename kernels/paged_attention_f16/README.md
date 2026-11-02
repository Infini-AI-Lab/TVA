# paged_attention_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/paged_attention.py`](source/paged_attention.py). Lean: [`lean/`](lean/).

The kernel is `paged_attention_kernel` (lines 30–72): decode-phase attention, one program per
(sequence, query head), over a paged key/value cache that it reads through each sequence's block
table (indirect memory access), with an online softmax in `float32`. The host is `paged_attention`
(lines 75–97); `make_cache`, `reference` and the test (lines 100–155) are host code too.

## Specification

`Spec K = Claim.Meets Inst.ok Pre Post K`: on every covered instance and initial memory, every
outcome of a launch that runs is normal (no undefined behaviour, no error), and its final memory
satisfies `Post`.

`Post d init fin` (`Model.lean`) says:

1. Every element `(s, h, e)` of `out` (`s < num_seqs`, `h < num_heads`, `e < HEAD_DIM`, element
   `outIdx d s h e`) holds a `float16` value `c` in `fin`.
2. When the values of sequence `s` and head `h` are covered (`SeqVals d init s h`, below), `c` is
   finite, and for every rational `X ≤ 2^10` that bounds the sequence's scaled score magnitudes
   (`scale · scoreAbs ≤ X` on each of its tokens),
   `|c − attnRef| ≤ tolO(HEAD_DIM, PAGE_SIZE, T, X) · attnAbs + tolF(HEAD_DIM, PAGE_SIZE, T, X)`,
   where `T = cdiv(seq_lens[s], PAGE_SIZE)` (`nblocks`), the number of pages the kernel visits.
3. Every byte outside the elements of `out` that the launch writes (`outElems`) is unchanged.

The reference is exact attention over the input values as memory holds them at launch:

- token `t < seq_lens[s]` of sequence `s` is in page `block_tables[s, t // PAGE_SIZE]` at row
  `t % PAGE_SIZE` (`pageOf`); query head `h` reads key/value head `h // (num_heads // num_kv_heads)`
  (`tokElem`); the block-table entries and the lengths are read from `init` (`val32`), as are `q`,
  `k_cache` and `v_cache` (`val16`);
- `score t = Σ_e q[s, h, e] · k[t, e]` and `scoreAbs t = Σ_e |q[s, h, e] · k[t, e]|`, exact
  rationals;
- `weight t = exp(σ · score t)` with `σ = scale`, the double's exact value (`sigma`);
- `attnRef e = Σ_t weight t · v[t, e] / Σ_t weight t`, and `attnAbs e = Σ_t weight t · |v[t, e]| /
  Σ_t weight t`, the same weighted mean of `|v|`, which scales the relative tolerance.

`SeqVals d init s h`: the sequence has at least one token; `q`'s row and every key and value of its
tokens are finite `float16`; every `|v| ≤ 2^15`; and every `scale · scoreAbs ≤ 2^10`.

The tolerances (`Model.lean`, rationals) are built from the library's constants only: one rounding
of `float32` (`2^-24` relative, `2^-150` absolute, from `FloatV.mul_err`, `mulAdd_err`, the sum
lemmas), one of `float16` (`2^-11`, `2^-25`, `FloatV.cvt_err`), `Num.ExpF32` (`2^-22 + 2^-23·|x|`
relative, `2^-140` absolute) and `Num.DivFull` (`2^-22`, `2^-148`):

- `rhoD D = (1 + u)(1 + 3u)^(D−1) − 1` (`u = 2^-24`): a score's row sum of `D` products along a
  fusable tree; `expErr D X = (rhoD D + 8u)·X + 2^-50`: each exponent `s − m_new` against
  `σ·score − m_new`;
- `psiP` (each `p` against its weight), `psiL` (a page's `tl.sum(p)`), `psiA` (a page's
  `tl.sum(p[:, None] * v)`), `step` (what a page can scale the earlier pages' error by: `tl.mul_add`
  and `alpha`), `drift` (`alpha`'s input-dependent error over the loop);
- `fL = (1 + psiL)·drift·step^T` and `fA = (1 + psiA)·drift·step^T`, the relative bounds on `l_i`
  and `acc` after `T` pages, with absolute parts `T·2^-100` and `T·2^-90`;
- `betaL`, `relQ`, `absQ`: `acc / l_i` against the weighted mean, using `Σ weights ≥ 1 − expErr`
  (`wlow`);
- `tolO = lastRel·(1 + relQ) + relQ` and `tolF = absQ·(1 + lastRel) + 2^-11·2^-148 + 2^-148 +
  2^-25`, `lastRel = 2^-11·(1 + 2^-22) + 2^-22` (the division and the `float16` rounding).

Sizes (computed from the definitions; `tolF` is about `2^-25 ≈ 3.0·10^-8` in every row):

| `HEAD_DIM` | `PAGE_SIZE` | `T` | `X` | `tolO` |
|---|---|---|---|---|
| 128 | 16 | 256 | 11 | `1.245·10^-3` |
| 128 | 32 | 256 | 11 | `1.249·10^-3` |
| 64 | 16 | 129 | 10 | `0.861·10^-3` |
| 128 | 16 | 512 | 20 | `1.90·10^-3` |
| 128 | 16 | 4096 | 11 | `4.46·10^-3` |
| 128 | 16 | `2^16` | 11 | `5.81·10^-2` |
| 256 | 256 | `2^16` | `2^10` | `0.174` |

On the test's data (its generator, seed 0, recomputed on the CPU) the largest `scale · scoreAbs`
over all sequences, heads and tokens is `10.96` (mean `7.2` at `head_dim = 128`), at most `256`
pages are visited, and `attnAbs ≤ 3.87` (near `E|v| ≈ 0.8` when many tokens carry weight); so the
bound is at most `1.25·10^-3 · attnAbs + 3·10^-8`, at most `4.9·10^-3`. The
test itself asserts `2·10^-3` against a `float64` reference and measured `4.9·10^-4`. The
tolerance grows mainly with the number of pages `T` (about `T·9·10^-7` from `step^T`): `0.058`
at `2^16` pages and the test's `X`, and below `0.18·attnAbs` at the largest covered sizes.

Nothing is said of an element of a sequence whose values are not covered beyond its holding a
`float16` value; in particular, a sequence with `seq_lens[s] = 0` gives `0/0`.

## Conditions

An instance `Inst` is the host's choice: `num_seqs`, `num_heads`, `num_kv_heads`, `HEAD_DIM`,
`PAGE_SIZE`, `num_pages`, `max_pages` (the block table's second dimension), `scale` (the double), the
eight strides in elements, and six buffers (`Buf`: a base address and a length in elements):
`q`, `k_cache`, `v_cache`, `out` (`float16`), `block_tables`, `seq_lens` (`int32`). The last
stride of `q`, `k_cache`, `v_cache`, `block_tables` and `out` is `1` (line 82 asserts it of `q`,
`k_cache` and `block_tables`; `v_cache` has `k_cache`'s strides, line 79; `out = empty_like(q)`).
`seq_lens` is taken contiguous: the kernel reads element `s` of its storage (line 45), so the claim
is about the `num_seqs` consecutive `int32` values from `seq_lens`'s data pointer, which are the
lengths `seq_lens[s]` only when `seq_lens` is contiguous; the wrapper does not check this (it
asserts the last stride of `q`, `k_cache` and `block_tables` only), and the test's `seq_lens`
(`torch.tensor`, line 115) is contiguous. Throughout, `seq_lens[s]` means element `s` of that
storage.

`Inst.ok d`:

| Field | Condition | Why |
|---|---|---|
| `seqs_pos`, `heads_pos` | `num_seqs ≥ 1`, `num_heads ≥ 1` | a grid with an empty extent runs no program; excluded to keep the claim simple |
| `kv_pos`, `kv_dvd` | `num_kv_heads ≥ 1` divides `num_heads` | line 80 asserts it; `QUERIES_PER_KV ≥ 1` |
| `heads_lt`, `grid_lt` | `num_heads < 2^16`, `num_seqs·num_heads < 2^31` | CUDA's grid limits (`Launch.Valid`) |
| `head_dim`, `page_size` | powers of two, at most `256`; `PAGE_SIZE ≥ 2` | `tl.arange` needs a power of two; `≤ 256` lanes per reduction (the test uses `64`, `128` and `16`, `32`); at `PAGE_SIZE = 1` line 68 adds two products (Narrowings) |
| `max_pages_le` | `max_pages ≤ 2^16` | at most `2^24` tokens per sequence; keeps the tolerance below `0.18` |
| `strides_lt` | every stride `< 2^31` | an `int32` argument |
| `prods_lt` | `(num_seqs−1)·stride_qs`, `(num_heads−1)·stride_qh`, the same for `out`, `(num_seqs−1)·stride_bs`, `(PAGE_SIZE−1)·stride_kt`, `(num_kv_heads−1)·stride_kh` below `2^31` | the `int32` products of lines 49, 57, 59 and 72 do not wrap; the page offset `page·stride_kp` is `int64`, so the cache may hold more than `2^31` elements |
| `q_len`, `out_len`, `kv_len`, `bt_len`, `sl_len` | every element the indices name lies in its buffer | the kernel accesses them (`kv_len` for every page below `num_pages`) |
| `out_inj` | `out`'s elements are distinct | no two lanes or programs store to one element |
| `disj` | `out` shares no byte with the five inputs | a fresh `torch.empty_like(q)` |
| `scale` | finite, in `[2^-64, 2^64]` | the host passes `head_dim**-0.5` |

`Pre d init`: the six buffers are valid in `init` (`Buf.Valid`: aligned, inside `[0, 2^64)`, every
byte present), and every sequence `s` is as `SeqOk` says: `seq_lens[s]` is in
`[0, max_pages·PAGE_SIZE]`, and each block-table entry the kernel reads (`j·PAGE_SIZE < seq_lens[s]`)
is a page number in `[0, num_pages)`. Pages need not be distinct, and entries the kernel does not
read may hold anything. Every `float16` value is allowed in `q`, `k_cache` and `v_cache`.

**The host's use is covered.** The test's three configurations (`num_heads/num_kv_heads/head_dim/
page` `32/8/128/16`, `8/8/64/16`, `16/2/128/32`, lengths `1` to `8191`) meet `Inst.ok` and `Pre`:
the tensors are contiguous, so `stride_qs = num_heads·head_dim ≤ 4096`, `stride_kt =
num_kv_heads·head_dim ≤ 1024`, `stride_kh = head_dim`, `stride_kp = page·stride_kt ≤ 16384`,
`stride_bs = max_pages ≤ 256`, and every product of `prods_lt` is below `2^22`; `make_cache` gives
each sequence distinct pages below `num_pages` and fills the rest of the table with `0`; `out` is a
fresh tensor; `scale = head_dim**-0.5`. The coordinator's run with a cache of more than `2^31`
elements is covered as well (`kv_len` asks only that the elements lie in the buffer, which is below
`2^63` elements by `Buf.Valid`).

`launch d` is the host's launch (lines 84–96): the grid `(num_seqs, num_heads, 1)`, Triton's default
`enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, no `launch_pdl`, and the source's block
shapes; `num_warps = 4` changes no outcome and is not recorded.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope.** The only
launch is the JIT launch `paged_attention_kernel[grid](…)` of `paged_attention`:

- **4 (options).** It gives the three constexprs and `num_warps=4`, one of the options that do not
  change the meaning; no other option. Nothing calls `warmup` or `triton.compile`; the kernel has
  no `@triton.autotune` or `@triton.heuristics`, and a bare `@triton.jit`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor.
- **6 (precompiled kernel).** No `CompiledKernel` is launched.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`.
- **9 (graph capture).** The script captures no CUDA graph.

Part (B), the environment (nothing else writes the launch's bytes, Triton's knobs at their
defaults, no hooks, and the others), is assumed by the rule, as for every kernel. The strides and
the other integers are runtime arguments whose values are part of the instance; the rule assumes
that the launch runs the code Triton compiles for them, specialisations included.

## Assumptions

`Assumptions K P` (a structure, every field a `[rule]`, each used by `main`):

| Field | Rule | Used for |
|---|---|---|
| `faithful` | `Compiler.Faithful P (model d) (launch d) (K d)` on every covered instance | the compiled kernel's outcomes are the model's (`meets_independent`) |
| `exp_acc` | `Num.ExpF32` | `tl.exp` of the exponents (line 65) and of `alpha`'s input (line 64) |
| `exp_ninf` | `Num.ExpF32NegInf` | `exp(−∞) = +0`: `alpha` on the first page, `p` on masked lanes |
| `exp_total` | `Num.MathTotal .exp .binary32` | `tl.exp` has a result for every input (no undefined behaviour) |
| `div_full` | `Num.DivFull` | `acc / l_i` (line 71) |
| `div_total` | `Num.DivTotal .binary32` | `/` has a result for every pair |

There is no `[kernel]` assumption. `consistent` shows that all of them hold together, of
`Platform.sm90ref` with a total faithful behaviour.

## Proof

`main` applies `TVA.meets_independent` with program `i` (sequence `pS d i = i % num_seqs`, head
`pH d i = i / num_seqs`) reading the five inputs' bytes (`Rd`) and writing its row of `out` (`Wr`):
different programs write different elements (`Wr_disj`, from `out_inj`), and `out` shares no byte
with the inputs (`disj`). The modules:

- `Real`, `Lane`: real-number lemmas, and the library's float facts in the form the loop uses
  (products, `tl.mul_add`, `float16` rounding, `tl.exp` at an input `≤ 1`, IEEE and fusable sums
  of at most `256` lanes).
- `RowMax`: one lane's score (`score_lane`: the row sum and the product by `float32` `scale`,
  within `Es` of `σ·score`), and the running maximum (`MInv`: within `E` of some token's
  `σ·score`, and at least every token's minus `E`); `mstep` gives the new maximum, each lane's
  exponent (`−∞` on a masked lane) and `alpha`'s input.
- `RowAcc`, `Step`: the weights `wt j = exp(σ·score j − m)` against the running maximum, each `p`
  and `alpha` against them, and the invariants `LInv`, `AInv` for `l_i` and each element of `acc`
  after `t` pages, kept by `lstep` and `astep` (masked lanes contribute exactly `+0`; the first
  page starts from `l_i = +0`, `acc = 0`, `alpha = +0`).
- `Final`: the tolerances as real numbers and their sizes (`tol_facts`); `final_o`: after the loop,
  `acc / l_i` is within `relQ·attnAbs + absQ` of the weighted mean (`ratio_err`; the weights'
  common factor `exp(−m)` cancels, so the weighted mean is `attnRef` exactly), then `DivFull` and
  the `float16` rounding.
- `Facts`: `int32`/`int64` arithmetic, the constants (`other=0.0` in `float16`, `-inf`, `scale`),
  the row's state `RowSt`/`StInv` (for covered values and every `X`), and `row_step`, one page.
- `Program`: `safe_program` walks the model line by line with the library's rules
  (`safe_load1`, `safe_load_buf` with `Buf.addrs_of_get` for the masked loads of `k` and `v`,
  `safe_convert`, `safe_forRange` with the invariant `StInv`, `safe_store_buf`). Every address is
  checked from `Pre` alone: the block-table entry of trip `j` is read and is a page below
  `num_pages` (`SeqOk`, since `j·PAGE_SIZE < seq_len`), and a lane's `int64` offset is the element
  `kvIdx` (`kv_ptr`); so no program has undefined behaviour whatever the float values are.
  `row_out` gives `Post`'s clause from the state after the loop.

`satisfiable` exhibits a covered instance (one sequence, one head, `HEAD_DIM = 128`,
`PAGE_SIZE = 16`, `scale = 128^-0.5`) in a memory of zeros. The proofs use only `propext`,
`Classical.choice` and `Quot.sound`.

## Narrowings and modelling decisions

Narrowings (each a field of `Inst.ok` or `SeqVals`, with its reason in the tables above):

- `HEAD_DIM`, `PAGE_SIZE ≤ 256`; `max_pages ≤ 2^16` (the tolerance's growth with the number of
  pages); non-empty grids.
- `PAGE_SIZE ≥ 2`. At `PAGE_SIZE = 1` the reduction `tl.sum(p[:, None] * v, axis=0)` of line 68
  has one lane, so its value is the rounded product `p·v` itself, and line 68's addition has two
  products as operands: the library writes it `tl.addc`, which lets the device contract either
  product (for example `fma(p, v, round(acc·alpha))`), while the model's `tl.mul_add` allows only
  the contraction of `acc * alpha`. From two lanes on, the reduction ends in an addition, which is
  no product, and `tl.mul_add` is right. A one-token page is not this kernel's use: the test's
  pages are `16` and `32` tokens. So the one-element case of the fusion rule (library issue L123's
  question, for a reduction) does not arise in this claim.
- `seq_lens` is contiguous (Conditions): the claim speaks of the `int32` values the kernel reads,
  element `s` of `seq_lens`'s storage.
- The `int32` products of the source do not wrap (`prods_lt`). The sum of the `int64` page offset
  needs no condition beyond the buffers.
- The float statement is for covered values (`SeqVals`): a non-empty sequence, finite inputs,
  `|v| ≤ 2^15` (so `acc` and the output stay in range) and `scale · scoreAbs ≤ 2^10` (so `tl.exp`'s
  inputs lie where `Num.ExpF32` applies). Outside them the claim still says the launch is safe and
  writes every element.

Modelling decisions:

- `m_i - m_new` (line 64) and `s - m_new` (line 65) are IEEE `tl.sub`: no product feeds either
  (`s` at line 65 is the result of `tl.where`, line 62, and `m_new` of `tl.maximum`, line 63). This
  is the reading of the library's `tl.add` docstring (an operand that is a select or a maximum is
  not the result of a multiplication); library issue L123 asks whether the library means it, and
  this kernel is recorded against it.
- A stride argument equal to `1` is the compile-time constant `1`, which each use builds in the
  dtype of the value it multiplies (`int32`, or `int64` for line 59's `page * stride_kp`); either
  way it is the value `1` the model passes.
- `tl.sum(k * q[None, :], axis=1)` is `tl.sum1c` and `tl.sum(p[:, None] * v, axis=0)` is
  `tl.sum0c` (the product may be fused into the reduction's additions); `tl.sum(p, axis=0)` is the
  IEEE `tl.sum`; `l_i * alpha + …` and `acc * alpha + …` are `tl.mul_add`.
- The Python floats `-inf` and `0.0` assigned to `m_i` and `l_i` are `float32` scalars
  (`FloatV.litF32`); `other=0.0` is a `float32` constant converted to `float16`; the `-inf` of
  `tl.where` is a `float32` constant.
- `tl.cdiv(seq_len, PAGE_SIZE)` is `(seq_len + PAGE_SIZE − 1) // PAGE_SIZE` in `int32`, evaluated
  once before the loop; `page` is widened with `tl.convert .i32 .i64`, and the `int32` terms of
  line 59 are converted to `int64` by the promotion.
- Broadcasting is written in the final shape (`col`, `row`, `bc`).
