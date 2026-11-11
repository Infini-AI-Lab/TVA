# `paged_attention_f16`

**The kernel.** `paged_attention_kernel`, a decode-phase paged attention written for TVA
(`source/paged_attention.py`, kernel lines 30–72; host `paged_attention` lines 75–97). A 2-D grid
of `num_seqs × num_heads` programs; program `(s, h)` loads the query row, then loops over the
sequence's `cdiv(seq_len, PAGE_SIZE)` pages, reading each page's keys and values through the
block table, and runs an online softmax in `float32`; it divides, rounds to `float16` and stores
the output row. The claim is `Claims.PagedAttention.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- every element `(s, h, e)` of `out` holds a `float16` value `c`;
- if the sequence's values are covered (`SeqVals`: at least one token, `q`'s row and the
  sequence's keys and values finite `float16`, every `|v| ≤ 2^15`, every
  `scale · Σ_e |q·k| ≤ 2^10`), `c` is finite and, for every `X ≤ 2^10` bounding the sequence's
  scaled score magnitudes,
  `|c − attnRef| ≤ tolA T X · (attnAbs + |attnRef|) + 2^-11·|attnRef| + 2^-24`, where `attnRef`
  is the exact attention in the reals over the tokens `t < seq_lens[s]` (keys and values read
  through `block_tables` in the initial memory, weights `exp(scale · q·k)` with the host's double
  `scale`), `attnAbs` the same weighted mean of `|v|`, `T = cdiv(seq_len, PAGE_SIZE)` and
  `tolA T X = (9·T/8 + 8·X + 8)·2^-21`;
- every byte outside `out`'s elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `num_kv_heads ≥ 1` divides `num_heads`; `num_heads < 2^16`, `num_seqs·num_heads < 2^31`,
  `num_seqs < 2^31` (CUDA's grid limits), `num_seqs ≤ 8192`;
- `HEAD_DIM` and `PAGE_SIZE` are powers of two at most `256`; `max_pages ≤ 2^16`;
- the strides and the kernel's `int32` products (lines 49, 57, 59, 72) are below `2^31`;
- every element the indices name lies in its buffer (the cache may hold more than `2^31`
  elements); `out`'s elements are distinct and its storage shares no byte with the inputs';
- `scale` is a finite double in `[2^-64, 2^64]`;
- the buffers are valid, and each sequence's length is in `[0, max_pages·PAGE_SIZE]` and at most
  `2^22`, with each block-table entry the kernel reads a page below `num_pages` (`SeqOk`; pages
  need not be distinct).

Every `float16` value is covered; outside `SeqVals` only that a value is stored is claimed.

**The launches covered** (`SETUP.md`): the test's three configurations (`setup_t1_ok`,
`setup_t2_ok`, `setup_t3_ok`), a cache of more than `2^31` elements (`setup_bigcache_ok`) and pages
of one token (`setup_page1_ok`); `canon_ok` covers the host's launch on contiguous tensors at every
shape the conditions allow.

**Device facts** the tolerance rests on (all in `Device P`): `Device.ExpF32` and
`Device.ExpF32NegInf` (`tl.exp` on `float32`, lines 64–65), `Device.DivFull` (`/` on `float32`,
line 71), `Device.SumDepth` (the trees of the `tl.sum`s of lines 61, 66, 68), and the totality
facts for `exp` and `/`.

## Tolerances

With `u = 2^-24`, `T` the pages and `X` the score bound: each page scales the earlier pages' error
by at most `(1 + 3u)·(1 + u if PAGE_SIZE = 1)·(1 + 2^-140)·(1 + 2^-22)` (the online-softmax
rescaling by `alpha`); the exponents' arguments carry the score's rounding (a `⌈log₂ HEAD_DIM⌉`-deep
sum of rounded products, the product by `scale`, the subtraction), relative `≈ 33u·X`, and
`ExpF32`'s `2^-22 + 2^-23·|x|`; `l_i` and `acc` are bounded relatively against the exact sums of
the weights and of `w·v`; the division (`DivFull`) and the `float16` rounding (`2^-11`) are
relative to `|acc / l|`. To first order the relative error is `T·7u + 6.7·2^-21·X + O(2^-21)`,
which `tolA T X = (9·T/8 + 8·X + 8)·2^-21` covers on `HEAD_DIM, PAGE_SIZE ≤ 256`, `T ≤ 2^16`,
`X ≤ 2^10` (worst ratio `0.933`, at `PAGE_SIZE = 1`, `T = 2^16`, `X = 0`); the absolute parts stay
below `2^-24`. The bound grows with `T`: on contexts of 32K–64K tokens it is loose.

## Exclusions

- **`int32` products** (a kernel limit): a `q` or `out` of `2^31` elements or more, or a block
  table of `2^31` entries, wraps the `int32` offsets. Real decode batches are far below.
- **`HEAD_DIM` or `PAGE_SIZE > 256`, `max_pages > 2^16`**: outside real use; the tolerance grows
  with the pages.
- **Contexts longer than `2^22` tokens, batches of more than `8192` sequences**: the claim covers
  real model shapes only. Every launch of `SETUP.md` meets them.
- **Values outside `SeqVals`** (an empty sequence, which gives `0/0`; non-finite inputs; large
  values or scores): only that a `float16` value is stored is claimed.
- **`num_seqs ≥ 2^31`** (only possible with `num_heads = 0`), **a non-contiguous `seq_lens`**,
  **`out` overlapping an input**: outside real use.

`check_gpu.py` (one H200) checks `Post` on the test's launches and on edge cases.
