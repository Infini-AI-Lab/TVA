# `paged_attention_f16`: the prepared claim

Kernel: `source/paged_attention.py`, `paged_attention_kernel` (lines 30–72; the kernel lines are
26–27 and 30–72, the host lines as [`claim.json`](claim.json) gives them). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.PagedAttention`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py). The launches to cover: [`SETUP.md`](SETUP.md).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.PagedAttention.main : Statement` (`Statement`:
`Claim.lean` line 477). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 182), and every initial memory `init` that `Pre` accepts (line 262, with `SeqOk`,
line 252), every outcome of the launch is a normal end (no fault, no hang) whose final memory `fin`
satisfies `Post` (line 438).

**What `Post` asks** (line 438), two parts:
1. for every `s < num_seqs`, `h < num_heads`, `e < HEAD_DIM`: element `outIdx d s h e` of `out`
   reads in `fin` as some `float16` `c`. This holds for every element. Only when
   `SeqVals d init s h` (line 315: at least one token, `q`'s row and the sequence's keys and values
   finite, `|v| ≤ 2^15`, `sigma d * scoreAbs ≤ 2^10`) it also asks that `c` is finite and, for
   every rational `X ≤ 2^10` with `sigma d * scoreAbs d init s h t ≤ X` on every token
   `t < seqLen d init s`,
   `|c.toReal - attnRef d init s h e| ≤ tolO D P T X * attnAbs d init s h e + tolR D P T X * |attnRef d init s h e| + tolF D P T X`
   with `D = HEAD_DIM`, `P = PAGE_SIZE`, `T = nblocks d init s` (line 288). `attnRef` (line 329)
   is exact attention in the reals over the tokens `t < seq_lens[s]`, read through `block_tables`
   in `init` (`tokElem`, `pageOf`); `attnAbs` (line 334) the same weighted mean of `|v|`; `tolO`,
   `tolR`, `tolF` are lines 417, 421, 425, built from the definitions of lines 349–414;
2. the frame: every byte outside `d.out.elems (outElems d)` (line 429) is unchanged.

**What you are given** (`Assumptions`, line 453):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, for a valid launch with
  `debug = false` that is well behaved from `init`;
- `exp_acc` (`Rules.Num.ExpF32`): on `[-X, X]`, `tl.exp` (source lines 64, 65) is within
  `(2^-22 + 2^-23·X)·e^x + 2^-140`;
- `exp_ninf` (`Rules.Num.ExpF32NegInf`): `tl.exp(-∞) = +0` (`alpha` on the first page, `p` on
  masked lanes);
- `exp_total` (`Rules.Num.MathTotal P .exp .binary32`): `tl.exp` has a result for every input;
- `div_full`, `div_total` (`Rules.Num.DivFull`, `Rules.Num.DivTotal`): `/` (source line 71) has a result,
  within `2^-22` relative and `2^-148` absolute;
- `sum_depth` (`Rules.Num.SumDepth`): the reductions of source lines 61, 66, 68 are trees of depth at most
  `⌈log₂ n⌉`.

**The kernel, as the model runs it** (`paged_attention_kernel` line 37, `model` line 232, `launch`
line 242): a 2-D grid of `num_seqs × num_heads` programs. Program `(s, h)` computes
`kv_head = h // qpk`, loads `seq_lens[s]` and the unmasked `q` row (converted to `float32`), and
runs a `tl.forRange` loop over `j ∈ [0, cdiv(seq_len, PAGE_SIZE))` (the trip count is
`nblocks`, from the loaded length). Each trip loads one block-table entry, the masked
`[PAGE_SIZE, HEAD_DIM]` key and value tiles of that page, and does the online softmax: the scores
(`tl.mul`, `tl.sum1c`, `· scale`), a `where` to `-∞` on masked rows, `tl.maxAll`, `tl.maximum`,
`alpha` and `p` by `tl.sub` and `tl.exp`, `l_i` by `tl.mul_add` with `tl.sum`, `acc` by
`tl.sum0c` with `tl.mul_add` (or `tl.addc` when `PAGE_SIZE = 1`). After the loop it divides,
rounds to `float16` and stores the unmasked `out` row. No dot, no atomic. Programs write disjoint
`out` rows and only read inputs: the route is `TVA.meets_independent`; the loop is
`safe_forRange` with an invariant on `(acc, l_i, m_i)` and `Wt := fun _ => ∅` (the body writes
nothing; `iterW_empty`) (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch has `debug = false` and is `Launch.Valid` directly (2-D grid): axis 0 below `2^31`
  from `seqs_lt`, the program count from `grid_lt`, axis 1 below `2^16`
  from `heads_lt`, the five block shapes from `head_dim`, `page_size`;
- each program is `Safe`: `safe_idiv` (`kv_pos`), the `int32` products do not wrap (`strides_lt`,
  `prods_lt`), the loop bound and the mask `j·PAGE_SIZE + r < seq_len` do not wrap (`SeqOk`:
  `seq_len ≤ max_pages·PAGE_SIZE`, `max_pages_le`, and `seq_len ≤ 2^22`, so at most `2^22` pages
  are visited and `j·PAGE_SIZE + r < 2^22 + 2^8`); every load and store is in bounds (`q_len`,
  `out_len`, `sl_len`, `bt_len` for `j < max_pages`, `kv_len` with the page from `SeqOk` and
  `h // qpk < num_kv_heads` from `kv_dvd`); the store's lanes are distinct (`out_inj`);
- the write footprints (the program's `HEAD_DIM` elements of `out`) are disjoint across programs
  (`out_inj`), and no program reads what another writes (`disj`);
- the value part of part 1, under `SeqVals`: `sumc_err_depth`, `sum_err_depth` with
  `SumDepth.depth_le`, `FloatV.mul_err`, `FloatV.mulAdd_err`, `AddRel.err`, `exp_acc` (weakened by
  `MathAcc.mono`), `exp_ninf`, `div_full`, `FloatV.cvt_err`, carried through the loop invariant.
  `## Tolerances` gives the derivation;
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `canon_ok`
(line 550) and the `setup_*_ok` theorems; reusable: `mix_lt` (line 519), `mix_inj` (line 525),
`bytes_disj` (line 537).

## Specification

`Post d init fin`:

1. Every element `(s, h, e)` of `out` (`s < num_seqs`, `h < num_heads`, `e < HEAD_DIM`, element
   `outIdx d s h e`) holds a `float16` value `c` in `fin`.
2. When the values of sequence `s` and head `h` are covered (`SeqVals`), `c` is finite, and for
   every rational `X ≤ 2^10` that bounds the sequence's scaled score magnitudes
   (`scale·scoreAbs ≤ X` on each of its tokens),
   `|c − attnRef| ≤ tolO·attnAbs + tolR·|attnRef| + tolF`, each tolerance a function of
   `(HEAD_DIM, PAGE_SIZE, T, X)`, `T = cdiv(seq_lens[s], PAGE_SIZE)` (`nblocks`), the pages the
   kernel visits.
3. Every byte outside the elements of `out` the launch writes (`outElems`) is unchanged.

The reference is exact attention over the values in `init`: token `t < seq_lens[s]` is in page
`block_tables[s, t // PAGE_SIZE]` at row `t % PAGE_SIZE` (`pageOf`); query head `h` reads key/value
head `h // (num_heads // num_kv_heads)` (`tokElem`); `score t = Σ_e q[s, h, e]·k[t, e]`,
`scoreAbs t = Σ_e |q·k|`; `weight t = exp(σ·score t)`, `σ = scale` (the double); `attnRef e =
Σ_t weight t·v[t, e] / Σ_t weight t`, and `attnAbs e` the same weighted mean of `|v|`.

`SeqVals d init s h`: at least one token; `q`'s row and every key and value of its tokens finite;
every `|v| ≤ 2^15`; every `scale·scoreAbs ≤ 2^10`.

The tolerances (rationals, `Claim.lean`, "The tolerances") compose the library's constants only:

| `HEAD_DIM` | `PAGE_SIZE` | `T` | `X` | `tolO` | `tolR` |
|---|---|---|---|---|---|
| 128 | 16 | 256 | 11 | `1.41·10^-4` | `6.29·10^-4` |
| 128 | 32 | 256 | 11 | `1.41·10^-4` | `6.29·10^-4` |
| 64 | 16 | 129 | 10 | `0.83·10^-4` | `5.71·10^-4` |
| 128 | 16 | 512 | 20 | `2.75·10^-4` | `7.63·10^-4` |
| 128 | 16 | 2048 | 11 | `8.90·10^-4` | `1.38·10^-3` |
| 128 | 16 | 4096 | 11 | `1.75·10^-3` | `2.24·10^-3` |
| 128 | 1 | 4097 | 11 | `1.99·10^-3` | `2.48·10^-3` |
| 128 | 16 | `2^16` | 11 | `2.86·10^-2` | `2.91·10^-2` |
| 256 | 256 | `2^16` | `2^10` | `3.21·10^-2` | `3.26·10^-2` |

`tolF ≈ 2^-25 ≈ 3.0·10^-8` in every row (the output's `float16` rounding near zero). On the test's
data the largest `scale·scoreAbs` is about `11` and at most `256` pages are visited, so the bound is
at most `1.41·10^-4·attnAbs + 6.29·10^-4·|attnRef| + 3·10^-8`: mostly the output's own `float16`
rounding (`2^-11·|attnRef|`).

## Conditions

`Inst`: `num_seqs`, `num_heads`, `num_kv_heads`, `HEAD_DIM`, `PAGE_SIZE`, `num_pages`,
`max_pages`, `scale` (the double), the eight strides in elements, and six buffers: `q`, `k_cache`,
`v_cache`, `out` (`float16`), `block_tables`, `seq_lens` (`int32`). The last stride of `q`,
`k_cache`, `v_cache`, `block_tables` and `out` is `1` (line 82 asserts it of `q`, `k_cache`,
`block_tables`; `v_cache` has `k_cache`'s strides, line 79; `out = empty_like(q)`); `seq_lens` is
read as contiguous (the kernel reads element `s` of its storage; the test's `torch.tensor` is).

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `kv_pos`, `kv_dvd` | `num_kv_heads ≥ 1` divides `num_heads` | line 80 asserts it |
| `heads_lt`, `grid_lt`, `seqs_lt` | `num_heads < 2^16`, `num_seqs·num_heads < 2^31`, `num_seqs < 2^31` | CUDA's grid limits (`Launch.Valid`); `seqs_lt` bounds axis 0 also when `num_heads = 0` |
| `seqs_le` | `num_seqs ≤ 8192` | `num_seqs`, the decode batch's sequences, is a batch size: real model shapes only (Exclusions) |
| `head_dim`, `page_size` | powers of two, at most `256` | `tl.arange` |
| `max_pages_le` | `max_pages ≤ 2^16` | `2^24` tokens per sequence; see Exclusions |
| `strides_lt`, `prods_lt` | every stride `< 2^31`; the `int32` products of lines 49, 57, 59, 72 below `2^31` | `int32` arguments and products (a kernel limit); the page offset `page·stride_kp` is `int64` |
| `q_len`, `out_len`, `kv_len`, `bt_len`, `sl_len` | every element the indices name lies in its buffer | the kernel accesses them |
| `out_inj`, `disj` | `out`'s elements distinct; `out` shares no byte with the inputs | a fresh `torch.empty_like(q)` |
| `scale` | finite, in `[2^-64, 2^64]` | the host passes `head_dim**-0.5` |

Empty grids (`num_seqs = 0`, or `num_heads = 0` with `num_seqs ≤ 8192`) are covered: no program runs and nothing changes.

`Pre`: the six buffers valid, and for every sequence `s` (`SeqOk`) `seq_lens[s] ∈ [0,
max_pages·PAGE_SIZE]`, `seq_lens[s] ≤ 2^22` (a context length: at most `2^22` tokens, real model
shapes, Exclusions), and each block-table entry the kernel reads (`j·PAGE_SIZE < seq_lens[s]`) a
page in `[0, num_pages)`. Pages need not be distinct; entries the kernel does not read may hold
anything; any `float16` values.

The model follows lines 42–72: `head // QUERIES_PER_KV` and `tl.cdiv` as `int32` `tl.idiv`; the
Python floats `-inf` and `0.0` assigned to `m_i`, `l_i` as `float32` scalars; `other=0.0` a
`float32` constant converted to `float16`; `page` widened with `tl.convert .i32 .i64` and the
`int32` terms of line 59 converted to `int64` by promotion; `tl.sum(k * q[None, :], axis=1)` as
`tl.sum1c` and `tl.sum(p[:, None] * v, axis=0)` as `tl.sum0c` (sums of products); `tl.sum(p)` the
IEEE `tl.sum`; `l_i * alpha + …` and, with `PAGE_SIZE ≥ 2`, `acc * alpha + …` as `tl.mul_add`; with
`PAGE_SIZE = 1` line 68's column sum has one lane, its value is the product `p·v`, and
`acc * alpha + …` adds two products: `tl.addc` of the IEEE product `acc * alpha` and the sum (an
`if` on the compile-time `PAGE_SIZE`); `m_i - m_new`, `s - m_new`
the IEEE `tl.sub` (no product feeds them: a `tl.where` and a `tl.maximum`); a stride equal to `1`
the compile-time `1`, built in the dtype of the value it multiplies.

`launch d`: the grid `(num_seqs, num_heads, 1)`, `fusion = true`, `debug = false`, `num_ctas = 1`,
no `launch_pdl`, the source's block shapes; `num_warps = 4` changes no outcome. A direct JIT launch
with only `num_warps` among the options, no `@triton.autotune`/`@triton.heuristics`, no
`warmup`/`triton.compile`, no descriptors: within part (A) of
`Compiler.Faithful`'s scope. The claim relies on part (B) (the environment) as the rule states it.

## Assumptions

All `[rule]`: `faithful` (`Compiler.Faithful`), `exp_acc` (`Num.ExpF32`, lines 64–65), `exp_ninf`
(`Num.ExpF32NegInf`: `alpha` on the first page, `p` on masked lanes), `exp_total`
(`Num.MathTotal P .exp .binary32`), `div_full` (`Num.DivFull`, line 71), `div_total`
(`Num.DivTotal P .binary32`), `sum_depth` (`Num.SumDepth`: the reductions of lines 61, 66 and 68
combine along trees of depth `⌈log₂ n⌉`). `consistent` uses `Platform.sm90ref`.

## Coverage

`canon S H KV D P NP MP sc` is the host's launch on contiguous tensors; `canon_ok` proves
`Inst.ok` from numeric facts. `SETUP.md`'s launches: `setup_t1_ok` (`32/8`, head dim 128, pages of
16, 7 sequences, 395 pages, `max_pages = 256`), `setup_t2_ok` (`8/8`, 64, 16, 3 sequences, 184
pages, 129), `setup_t3_ok` (`16/2`, 128, 32, 4 sequences, 310 pages, 256), each with the host's
`head_dim**-0.5` (`sc128`, `sc64`); `setup_bigcache_ok`: a cache of more than `2^31` elements;
`setup_page1_ok`: pages of one token (`8/2` heads, head dim 128, sequences up to 4097 tokens).
`Pre` holds for the test's caches: `make_cache` gives each sequence distinct pages below
`num_pages` and fills the rest of the table with `0`, and `seq_lens[s] ≤ max_pages·PAGE_SIZE`;
its longest sequence, 8191 tokens, is below `2^22`.

## Tolerances

From the library's facts (`docs/rules/num.md`): one `float32` rounding (`2^-24`, `2^-150`:
`FloatV.mul_err`, `mulAdd_err`, `AddRel.err`), one `float16` rounding (`2^-11`, `2^-25`,
`FloatV.cvt_err`), `Num.ExpF32` (`2^-22 + 2^-23·|x|` relative, `2^-140` absolute on `[-X', X']`),
`Num.ExpF32NegInf`, `Num.DivFull` (`2^-22`, `2^-148`), and `Num.SumDepth` with `sum_err_depth`,
`sumc_err_depth` (a reduction over `n` positions along a tree of depth `⌈log₂ n⌉`, `Nat.clog 2 n`):

- `rhoD D = (1 + u)(1 + 3u)^⌈log₂ D⌉ − 1`: a score's row sum of `D` rounded products along a
  fusable tree; `expErr D X = (rhoD D + 8u)·X + 2^-50`: each exponent `s − m_new` against
  `σ·score − m_new` (the product by the `float32` `scale`, its rounding from the double, the
  subtraction);
- `psiP` (each `p` against its weight: `expErr` through `exp`, and `ExpF32` at `|x| ≤ 2X + 1`),
  `psiL` (a page's `tl.sum(p)`, IEEE additions of depth `⌈log₂ P⌉`), `psiA` (a page's
  `tl.sum(p[:, None] * v)`, the rounded products along a fusable tree of depth `⌈log₂ P⌉`),
  `step P` (what a page scales the earlier pages' error by: the addition's `3u` (`tl.mul_add`, or
  at `P = 1` `tl.addc`, whose operand `acc * alpha` is rounded first: one more factor `1 + u`),
  `alpha`'s `2^-22` and `2^-140`), `drift` (`alpha`'s input-dependent error over the loop: the
  running maximum only grows, so its arguments sum to at most `2(X + E)`);
- `fL`, `fA = (1 + psi)·drift·step P^T`, the relative bounds on `l_i` (against the sum of the
  weights `W`) and on `acc` (against `Σ w·v`, relative to `Σ w·|v|`) after `T` pages, with
  absolute parts `T·2^-100` and `T·2^-90`; `betaL` folds `l_i`'s absolute part in
  (`W ≥ 1 − expErr` against the maximum, `wlow`);
- the ratio: `acc/l − ΣwV/W = (acc − ΣwV)/l + (ΣwV/W)·(W − l)/l` with `l ≥ (1 − betaL)·W`, so
  `|acc/l − attnRef| ≤ relA·attnAbs + relR·|attnRef| + absQ`, `relA = (fA − 1)/denQ`,
  `relR = betaL/denQ`, `denQ = 1 − betaL` (the weights' common factor `exp(−m)` cancels). The
  accumulator's error is relative to the weighted mean of `|v|`, and `l_i`'s error scales the
  output itself: an error common to `acc` and `l_i` (each `p`, each `alpha`) is charged once to
  each, which is what a relative change of the weights costs a weighted mean
  (`Σ w·ε·(v − attnRef) / Σ w·(1 + ε)`);
- the last roundings (`/`, then `.to(float16)`; `lastRel = 2^-11·(1 + 2^-22) + 2^-22`) are relative
  to `|acc/l| ≤ |attnRef| + |acc/l − attnRef|`: `tolO = relA·(1 + lastRel)`,
  `tolR = relR·(1 + lastRel) + lastRel`, `tolF = absQ·(1 + lastRel) + 2^-11·2^-148 + 2^-148 + 2^-25`.

Provable: each factor is one cited fact or rounding lemma, and `SeqVals` keeps every intermediate
in range (`|v| ≤ 2^15` and `T ≤ 2^16` keep `acc` finite; `X ≤ 2^10` keeps `tl.exp`'s inputs where
`ExpF32` applies). Meaningful: at the test's sizes `tolO ≈ 1.4·10^-4` and `tolR ≈ 6.3·10^-4`, about
`1.3` times the output's `float16` rounding; the GPU check reaches `0.88` of the bound on the
test's launches, where `tol / |attnRef|` has median `7·10^-4`. The bound grows with `T` (about
`T·4.2·10^-7` in each of `tolO`, `tolR`, from `step P^T`: `T` sequential updates, each with
`alpha`'s `2^-22` and the addition's `3·2^-24`): at 32K to 64K tokens (pages of 16, `T` = 2048 to
4096) `tol / |attnRef|` has median `0.21` (`0.11` with pages of 32), while the error reaches at most
`0.02` of the bound. Such a worst-case bound cannot tell a dropped last page from the kernel's
rounding there (dropping one page of 16 out of 64K tokens moves the output by about
`2.4·10^-4·attnAbs`, below `tolO`); `Post` still rejects it on some rows of each such launch, and
rejects an all-zero output, a neighbouring key/value head and missing `scale` on every row
(Checks). With pages of one token, on zero-mean `randn` values, `tol / |attnRef|` has median `0.09` at
4K tokens, `0.27` at 8K, `0.76` at 16K, `2.05` at 32K and `6.6` at 64K: it passes `1` near 20K
tokens, and at 64K an all-zero output meets `Post`. With values of mean `0.3` it stays at most
`0.125`.

## Exclusions

* **`int32` products (a kernel limit)**: `seq * stride_qs`, `head * stride_qh`,
  `seq * stride_bs`, `offs_t * stride_kt`, `kv_head * stride_kh`, `seq * stride_os`,
  `head * stride_oh` are `int32`; a `q` or `out` of `2^31` elements or more, or a block table of
  `2^31` entries, wraps them (for example `num_seqs·num_heads·head_dim ≥ 2^31`: `2^19` sequences at
  `32 × 128`); real decode batches are far below. The cache itself may exceed `2^31` elements.
* **`HEAD_DIM`, `PAGE_SIZE > 256`, `max_pages > 2^16`** (more than `2^24` tokens per sequence at
  pages of 256): outside real use (the test's head dims are 64 and 128, its pages 16 and 32, its
  longest sequence 8191 tokens); the tolerance grows with the pages.
* **Contexts longer than `2^22` tokens, batches of more than `8192`** (out of scope): dimensions
  that are not multiples of 128 or exceed `2^17`, contexts longer than `2^22` tokens, and batches
  of more than `8192` requests: the claim covers real model shapes only. Fields: `seq_lens[s]`
  (`SeqOk`, `seq_lens[s] ≤ 2^22`) and `num_seqs` (`seqs_le`, the batch size, `≤ 8192`). No field of
  this kernel is a model dimension (`HEAD_DIM` is a head dim, `num_heads`, `num_kv_heads` head
  counts, `PAGE_SIZE` a page size), so none is bounded by the first rule. Every launch of
  `SETUP.md` meets them (at most `7` sequences).
* **Values outside `SeqVals`** (an empty sequence, which gives `0/0`; non-finite inputs; `|v| >
  2^15`; `scale·scoreAbs > 2^10`): covered with a weaker `Post` (a `float16` value is stored,
  nothing else changes); outside real use.
* **`num_seqs ≥ 2^31`** (with `num_heads ≥ 1` already outside `grid_lt`, so only with
  `num_heads = 0`; beyond `seqs_le` too): outside real use (a model has heads; the host's
  `num_seqs` is a decode batch). The grid `(num_seqs, 0, 1)` runs no program, but its axis 0 is
  beyond CUDA's `2^31 − 1` limit, so it is not a launch the claim covers.
* **A non-contiguous `seq_lens`**: the kernel reads element `s` of its storage; outside real use.
* **`out` overlapping an input**: the wrapper allocates a fresh `out`.

## Library requests

None.

## Checks

* **Builds**: `lake env lean ../claims/paged_attention_f16/Claim.lean`: no errors or warnings;
  `consistent`, `satisfiable` and the coverage facts use only `propext`, `Classical.choice`,
  `Quot.sound`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (one sequence, one head,
  `HEAD_DIM = 128`, `PAGE_SIZE = 16`, a memory of zeros).
* **Covers the setup**: `canon_ok`, `setup_t1_ok`, `setup_t2_ok`, `setup_t3_ok`,
  `setup_bigcache_ok`, `setup_page1_ok`.
* **Model map**: every code line covered, every model text found in `Claim.lean` (the gate's
  `map_problems`).
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): the test's three configurations from its
  generator, and edges (head dim 1 with pages of 2 and of 1, pages of 1 at head dim 128, head dim
  and pages of 256, pages of 2, 64 heads on one KV head, 200 sequences, pages shared by every
  sequence, a strided `q`, `2^16` tokens with pages of 16 and of 1, sequences of 32K to 64K tokens
  at `32/8` heads with pages of 16 and at `16/2` with pages of 32, empty sequences, `q` scaled 8×,
  `|v|` near `2^15` and near `2^-20`). Worst error / tolerance `0.88` on the test's launches,
  `0.997` overall (the `|v| ≈ 2^-20` case, an output near `10^-7`, where `tolF`'s `2^-25`, the
  output's `float16` rounding, is most of the bound and the rounding reaches it); every covered
  element finite; inputs unchanged bit for bit, the bytes past `out` keep their sentinel.
  `tol / |attnRef|`: median `7·10^-4` on the test's launches, `0.21` and `0.11` on the 32K–64K
  launches. Wrong outputs: on the test's launches, pages of 1 and the 32K–64K launches, `Post`
  rejects an all-zero output, the last page dropped, the first page dropped, the neighbouring
  key/value head, and the scores without `scale` (on the test's launches every row of each but
  `no-scale`, which a one-token sequence cannot show; at 32K–64K tokens a dropped page fails on 7 of
  96 and 17 of 32 rows).
