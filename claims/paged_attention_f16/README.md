# `paged_attention_f16`: the prepared claim

Kernel: `source/paged_attention.py`, `paged_attention_kernel` (lines 30–72; the kernel lines are
26–27 and 30–72, the host lines as [`claim.json`](claim.json) gives them). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.PagedAttention`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py). The launches to cover: [`SETUP.md`](SETUP.md).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.PagedAttention.main : Statement` (`Statement`:
`Claim.lean` line 400). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 182), and every initial memory `init` that `Pre` accepts (line 262, with `SeqOk`,
line 252), every outcome of the launch is a normal end (no fault, no hang) whose final memory `fin`
satisfies `Post` (line 362).

**What `Post` asks** (line 362), two parts:
1. for every `s < num_seqs`, `h < num_heads`, `e < HEAD_DIM`: element `outIdx d s h e` of `out`
   reads in `fin` as some `float16` `c`. This holds for every element. Only when
   `SeqVals d init s h` (line 315: at least one token, `q`'s row and the sequence's keys and values
   finite, `|v| ≤ 2^15`, `sigma d * scoreAbs ≤ 2^10`) it also asks that `c` is finite and, for
   every rational `X ≤ 2^10` with `sigma d * scoreAbs d init s h t ≤ X` on every token
   `t < seqLen d init s`,
   `|c.toReal - attnRef d init s h e| ≤ tolA T X * (attnAbs d init s h e + |attnRef d init s h e|) + 1 / 2^11 * |attnRef d init s h e| + 1 / 2^24`
   with `T = nblocks d init s` (line 288) and `tolA T X = (9·T/8 + 8·X + 8)·2^-21` (line 349).
   `attnRef` (line 329) is exact attention in the reals over the tokens `t < seq_lens[s]`, read
   through `block_tables` in `init` (`tokElem`, `pageOf`); `attnAbs` (line 334) the same weighted
   mean of `|v|`;
2. the frame: every byte outside `d.out.elems (outElems d)` (line 352) is unchanged.

**What you are given** (`Assumptions`, line 376):
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
  `MathAcc.mono`), `exp_ninf`, `div_full`, `FloatV.cvt_err`, carried through the loop invariant:
  derive the fine bound `tolO·attnAbs + tolR·|attnRef| + tolF` of README `## Tolerances

From the library's facts (`docs/rules/num.md`): one `float32` rounding (`u = 2^-24`, `2^-150`:
`FloatV.mul_err`, `mulAdd_err`, `AddRel.err`), one `float16` rounding (`2^-11`, `2^-25`,
`FloatV.cvt_err`), `Num.ExpF32` (`2^-22 + 2^-23·|x|` relative, `2^-140` absolute on `[-X', X']`),
`Num.ExpF32NegInf`, `Num.DivFull` (`2^-22`, `2^-148`), and `Num.SumDepth` with `sum_err_depth`,
`sumc_err_depth` (a reduction over `n` positions along a tree of depth `⌈log₂ n⌉`, `Nat.clog 2 n`).
With `D = HEAD_DIM`, `P = PAGE_SIZE`, `T` the pages and `X` the score bound, the fine bound is
`|c − attnRef| ≤ tolO·attnAbs + tolR·|attnRef| + tolF`, composed as follows (all rationals):

- `rhoD = (1 + u)(1 + 3u)^⌈log₂ D⌉ − 1`: a score's row sum of `D` rounded products along a
  fusable tree; `E = expErr = (rhoD + 8u)·X + 2^-50`: each exponent `s − m_new` against
  `σ·score − m_new` (the product by the `float32` `scale`, its rounding from the double, the
  subtraction);
- `relExp = 2^-22 + (2X + 1)·2^-23` (`ExpF32` at `|x| ≤ 2X + 1`); `psiP = (1 + E + E²)(1 + relExp)
  − 1` (each `p` against its weight); `psiL = (1 + u)^⌈log₂ P⌉ (1 + psiP) − 1` (a page's
  `tl.sum(p)`, IEEE additions); `psiA = (1 + 3u)^⌈log₂ P⌉ (1 + u)(1 + psiP) − 1` (a page's
  `tl.sum(p[:, None] * v)`, rounded products along a fusable tree);
- `step = (1 + 3u)·(1 + u if P = 1, else 1)·(1 + 2^-140)·(1 + 2^-22)`: what a page scales the
  earlier pages' error by (the addition's `3u`, `tl.mul_add`, or at `P = 1` `tl.addc`, whose operand
  `acc * alpha` is rounded first; `alpha`'s `2^-22` and `2^-140`); `drift = 1 + 4·2^-22·(X + E)`
  (`alpha`'s input-dependent error over the loop: the running maximum only grows, so its arguments
  sum to at most `2(X + E)`);
- `fL = (1 + psiL)·drift·step^T`, `fA = (1 + psiA)·drift·step^T`: the relative bounds on `l_i`
  (against the sum of the weights `W`) and on `acc` (against `Σ w·v`, relative to `Σ w·|v|`)
  after `T` pages, with absolute parts `T·2^-100` and `T·2^-90`; `wlow = 1 − E` (`W ≥ 1 − E`
  against the maximum); `betaL = (fL − 1) + fL·T·2^-100/wlow` folds `l_i`'s absolute part in;
- the ratio: `acc/l − ΣwV/W = (acc − ΣwV)/l + (ΣwV/W)·(W − l)/l` with `l ≥ (1 − betaL)·W`, so
  `|acc/l − attnRef| ≤ relA·attnAbs + relR·|attnRef| + absQ`, `denQ = 1 − betaL`,
  `relA = (fA − 1)/denQ`, `relR = betaL/denQ`, `absQ = fA·T·2^-90/(wlow·denQ)` (the weights'
  common factor `exp(−m)` cancels). The accumulator's error is relative to the weighted mean of
  `|v|`, and `l_i`'s error scales the output itself: an error common to `acc` and `l_i` (each `p`,
  each `alpha`) is charged once to each, which is what a relative change of the weights costs a
  weighted mean (`Σ w·ε·(v − attnRef) / Σ w·(1 + ε)`);
- the last roundings (`/`, then `.to(float16)`; `lastRel = 2^-11·(1 + 2^-22) + 2^-22`) are relative
  to `|acc/l| ≤ |attnRef| + |acc/l − attnRef|`: `tolO = relA·(1 + lastRel)`,
  `tolR = relR·(1 + lastRel) + lastRel`, `tolF = absQ·(1 + lastRel) + 2^-11·2^-148 + 2^-148 + 2^-25`.

Provable: each factor is one cited fact or rounding lemma, and `SeqVals` keeps every intermediate
in range (`|v| ≤ 2^15` and `T ≤ 2^16` keep `acc` finite; `X ≤ 2^10` keeps `tl.exp`'s inputs where
`ExpF32` applies).

**The relaxation to `Post`'s bound.** `tolO ≤ ε` and `tolR ≤ ε + 2^-11` with
`ε = tolA T X = (9T/8 + 8X + 8)·2^-21`, and `tolF ≤ 2^-24`; then `tolO·attnAbs + tolR·|attnRef| +
tolF ≤ ε·(attnAbs + |attnRef|) + 2^-11·|attnRef| + 2^-24`. To first order `tolO` and `tolR − 2^-11`
are `T·(step − 1) + (psi + drift − 1)`: `step − 1 = 7u = 0.875·2^-21` per page (`8u = 2^-21` at
`P = 1`), and the `X` part `(rhoD + 8u)·X + (2X + 1)·2^-23 + 4·2^-22·X ≤ (33 + 4 + 16)·u·X ≤
6.7·2^-21·X` at `D ≤ 256`; `8X` and `8` take that part and the constants (`rhoD`, `relExp`'s `2^-22`,
`lastRel − 2^-11 = 2^-22 + 2^-33`). The compounding (`step^T`, at most `e^{2^-5} = 1.032` at
`T = 2^16`, `P = 1`) and `1/denQ` (`denQ ≥ 0.96`) are what `9T/8` leaves room for: at `P = 1`,
`T = 2^16`, `X = 0`, `tolO = 0.0328 ≤ 0.0352 = ε`. Evaluated exactly (decimal arithmetic, 60
digits) over `D ∈ {1, 2, 3, 64, 128, 129, 256}`, `P ∈ {1, 2, 3, 16, 17, 256}`, `T` from `1` to
`2^16` and `X` from `0` to `2^10` (each part is monotone in `⌈log₂ D⌉`, `⌈log₂ P⌉`, `T` and `X`,
and linear or convex in `T` and `X`), `max(tolO, tolR − 2^-11)/ε ≤ 0.933`, reached at `P = 1`,
`T = 2^16`, `X = 0`. `absQ ≤ 2^16·2^-90·1.1`, so `tolF ≤ 2^-25 + 2^-73 ≤ 2^-24`.

Meaningful: at the test's sizes `ε ≈ 1.8·10^-4` and `ε + 2^-11 ≈ 6.7·10^-4` (`tolO ≈ 1.4·10^-4`,
`tolR ≈ 6.3·10^-4`), about `1.4` times the output's `float16` rounding; the GPU check reaches
`0.85` of the bound on the test's launches, where `tol / |attnRef|` has median `8·10^-4`
(`7·10^-4` for the fine bound). The bound grows with `T` (about `T·5.4·10^-7` in `ε`: `T`
sequential updates, each with `alpha`'s `2^-22` and the addition's `3·2^-24`): at 32K to 64K tokens
(pages of 16, `T` = 2048 to 4096) `tol / |attnRef|` has median `0.27` (`0.14` with pages of 32;
`0.21` and `0.11` for the fine bound), while the error reaches at most `0.015` of the bound. Such
a worst-case bound cannot tell a dropped last page from the kernel's rounding there (dropping one
page of 16 out of 64K tokens moves the output by about `2.4·10^-4·attnAbs`, below `ε`); `Post`
still rejects it on some rows of each such launch, and rejects an all-zero output, a neighbouring
key/value head and missing `scale` on every row (Checks). With pages of one token, on zero-mean
`randn` values, the fine bound's `tol / |attnRef|` has median `0.09` at 4K tokens, `0.27` at 8K,
`0.76` at 16K, `2.05` at 32K and `6.6` at 64K (`tol` is at most `1.13` times the fine bound
there): it passes `1` near 20K tokens, and at 64K an all-zero output meets `Post`. With values of
mean `0.3` the fine bound's ratio stays at most `0.125`.

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
  `|v|` near `2^15` and near `2^-20`). Worst error / tolerance `0.85` on the test's launches,
  `0.89` overall (pages of 2 at head dim 64; on the `|v| ≈ 2^-20` case, outputs near `10^-7`, the
  output's `float16` rounding reaches `0.50` of the `2^-24` floor); every covered
  element finite; inputs unchanged bit for bit, the bytes past `out` keep their sentinel.
  `tol / |attnRef|`: median `8·10^-4` on the test's launches, `0.27` and `0.14` on the 32K–64K
  launches (`7·10^-4`, `0.21`, `0.11` for the fine bound of Tolerances). Wrong outputs: on the test's launches, pages of 1 and the 32K–64K launches, `Post`
  rejects an all-zero output, the last page dropped, the first page dropped, the neighbouring
  key/value head, and the scores without `scale` (on the test's launches every row of each but
  `no-scale`, which a one-token sequence cannot show; at 32K–64K tokens a dropped page fails on 2
  to 4 of 96 and 15 to 16 of 32 rows).
