# `topk_log_softmax_f32`

**The kernel.** `_topk_log_softmax_kernel` of vLLM (`source/logprob.py`, kernel lines 30–72; host
`compute_token_logprobs` lines 75–99), the kernel behind the logprobs vLLM's sampler returns. A 1-D
grid of `batch_size` programs; program `r` takes the max of row `r` of a `float32` logits matrix in
blocks of `1024` (masked lanes `-∞`), then the sum of `tl.exp(logit − max)` in blocks of `1024`
(masked lanes `0`), then `lse = tl.log(sum)`, and for each of the row's `num_logprobs` token ids
(an `int64` matrix, in blocks of `TOPK_BLOCK_SIZE`) stores `(logits[r, id] − max) − lse`. The
claim is `Claims.TopkLogSoftmax.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
row `r < batch_size`, with `xs` the row's initial logits and `id_j` its initial token ids:
- every output element `out[r, j]`, `j < num_logprobs`, holds a `float32` value;
- if the row is `RowOk` (every logit a finite `float32` or `-∞`, one finite, and the finite ones
  within `2^127` of each other), there is one shift `sh` for the whole row with
  `|sh| ≤ lseTol = 2^-34·vocab_size + 2^-16` such that every output `c_j` is `-∞` exactly when
  `xs[id_j] = -∞`, and otherwise finite with `|c_j − (L_j − sh)| ≤ elemTol·|L_j| + floorTol`,
  `elemTol = 2^-22`, `floorTol = 2^-37`, where `L_j = xs[id_j] − log Σ_k e^(xs[k])` is the exact
  log-softmax in the reals (`e^(−∞) = 0`);
- every byte outside the output elements is unchanged.

The shift is the error of `lse`, common to the row: differences of a row's outputs are accurate to
`2^-22` relative. A row that is not `RowOk` (a NaN, a `+∞`, only `-∞`, or finite logits more than
`2^127` apart) has outputs that need only hold values.

**The conditions** (`Inst.ok`, `Pre`):
- `0 < vocab_size ≤ 2^19` (see Exclusions);
- `TOPK_BLOCK_SIZE` a power of two at most `1024`;
- `num_logprobs + TOPK_BLOCK_SIZE ≤ 2^31` (the `int32` lanes `j + l` and the last increment of the
  `int32` loop over the ids do not overflow: a kernel limit);
- fewer than `2^31` rows (the grid), strides below `2^31` (`int32` arguments);
- `logits` holds every row's `vocab_size` logits, `token_ids` every id read, the output
  `batch_size · num_logprobs` elements; the output shares no byte with the logits and ids read (the
  host's output is fresh); the three buffers are valid;
- every id read is in `[0, vocab_size)`.

Any strides (a column slice of wider logits, an id matrix with row stride above `num_logprobs`,
even overlapping rows, which are only read), `batch_size = 0`, `num_logprobs = 0` (with any
`TOPK_BLOCK_SIZE`), repeated ids and every `float32` logit, NaN and infinities included, are
covered.

**The launches covered** (`SETUP.md`): every launch the callers of `compute_token_logprobs` make
on `float32` logits: the sampler without and with logprobs, the custom-id path (`1 + 128`
columns, zeros in unused columns), speculative decoding's rows, prompt logprobs over the whole
vocabulary (`1 + vocab_size` columns) and prompt-token scoring, for vocabularies from `32000` to
Gemma 3's `262208`, contiguous or a column slice; `mk_ok` covers a `B × V` launch with row stride
`S ≥ V` for every `TOPK_BLOCK_SIZE = 2^e`, `e ≤ 10`, and the `setup_*_ok` theorems the instances
of `SETUP.md`. Launches on `bfloat16` or `float16` logits are not covered (see Exclusions).

**Device facts** the tolerances rest on (all in `Device P`): `Device.ExpF32` (`tl.exp` on
`float32`: relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute `2^-140`), `Device.ExpF32NegInf`
(`tl.exp(−∞) = +0`), `Device.LogF32` (`tl.log` on `float32`: `2^-22·|ln x| + 2^-149`),
`Device.SumDepth` (the tree of a float `tl.sum` over `1024` positions has depth at most `10`), and
the totality facts for `exp` and `log`. The max, the subtractions and the additions are IEEE
operations.

## Tolerances

Per `RowOk` row, with `V = vocab_size ≤ 2^19`, `T = ⌈V/1024⌉ ≤ 512` trips of the vocabulary
loops, `u = 2^-24`, `m` the row's max and `Z = Σ_k e^(x_k − m) ∈ [1, V]`:
- the max (IEEE maxima along the loop and each block's tree) is a finite logit, at least every
  logit;
- each lane's `s = fl(x − m) ≤ 0` is one IEEE rounding (or `−∞` when `x = −∞` or `x − m`
  overflows, where `tl.exp` gives `+0` or its result is tiny); `ExpF32` at `X = |s|` with the
  rounding of `s` gives `e^(x−m)` within `(4u·|x − m| + 11u)·e^(x−m) + 2^-137`, and since
  `|t|·e^t ≤ 14·e^t + 2^-15` for `t ≤ 0` the lanes' errors add up to at most `132u·Z`;
- the loop's additions `se += tl.sum(e)`, with each block's tree, form one tree of IEEE additions
  of depth at most `T + 10` over the row's lanes and the initial `0.0`: `|se − Z| ≤ (T + 143)·u·Z`;
- `|log se − log Z| ≤ ρ + 2ρ²`, `ρ = (T + 143)·u`, and `LogF32` adds at most
  `2^-22·|log se| + 2^-149 ≤ 53u` (`log Z ≤ log 2^19 < 13.2`): the shift `sh = lse − log Z` has
  `|sh| ≤ (T + 196)·u ≤ (V/1024 + 197)·u`, relaxed to `lseTol = (V/1024 + 256)·u =
  2^-34·V + 2^-16`;
- an output of a finite logit is two IEEE roundings, `d = fl(x − m)` (`|x − m| ≤ 2^127`: no
  overflow) and `fl(d − lse)`: `|c − (L − sh)| ≤ (2u + u²)·|x − m| + u·log Z + u·|sh| + 3·2^-150`;
  with `|L| = |x − m| + log Z` this is at most `2^-22·|L| + 2^-37` (`elemTol`, `floorTol`);
- an output of a `-∞` logit is `−∞ − m = −∞`, then `−∞ − lse = −∞`, exactly.

An output rounded to `bfloat16`, a `lse` that drops the row's last block of `1024` logits, the logit
of the next id, and the exact log-softmax shifted by `2·lseTol` fail `Post`.

## Exclusions

- **Logits of another dtype.** The model loads `float32` logits; vLLM also passes the model's raw
  `bfloat16` (or `float16`) logits when no logits processing copies them to `float32` and the head
  is not `float32`. Those launches compile the kernel for another pointer dtype (the loads, the
  `tl.max` and the `.to(tl.float32)` conversions differ) and are not covered by this claim.
- **`vocab_size > 2^19`**: above the vocabularies of real models (the largest, Gemma 3's `262208`
  and Gemma 3n's `262400`, are above `2^18`); the tolerance uses `T ≤ 512` and `log 2^19`.
- **`num_logprobs + TOPK_BLOCK_SIZE > 2^31`**: a kernel limit (the `int32` lanes of the ids wrap).
- **Strides or `batch_size` at `2^31` or above**: outside real use (Triton would pass `int64`
  strides; the grid has fewer than `2^31` programs).
- **`num_logprobs = 0` through the host**: the host computes `TOPK_BLOCK_SIZE =
  next_power_of_2(0) = 0`, and Triton rejects `tl.arange(0, 0)` before launching; a direct launch
  with `TOPK_BLOCK_SIZE ≥ 1` is covered.
- **Rows that are not `RowOk`**: the log-softmax is undefined (NaN, `+∞`, only `-∞`), or
  `x − max` may overflow to `-∞` (finite logits more than `2^127` apart); only that outputs hold
  values is claimed.
- **Ids outside `[0, vocab_size)`**, **output overlapping the inputs**: outside the host's
  launches (ids are sampled tokens, top-k indices or ids checked against the vocabulary; the
  output is fresh).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, one H200 NVL) checks `Post` row by row on the 9 launches of
`SETUP.md` (host calls, and direct launches with other `TOPK_BLOCK_SIZE`s and the output inside a
sentinel buffer) and on edges of `Inst.ok` and `RowOk`: rows with `-∞` (top-k masked logits, one
finite logit, half the vocabulary, the first blocks), all-equal rows, logits near `±1.7·10^38`
within `2^127` of each other, subnormal logits, gaps up to `1000`, the max in the last partial
block, rows that are not `RowOk`, repeated ids, vocabularies `1`, `2`, `1023`, `1024`, `1025`,
`2047`, `4097` and `2^19`, `num_logprobs = 0` (direct) and `1`, an id matrix with a wider row
stride, and `3000` ids of one Gemma 3 row. The worst error / tolerance over every `RowOk` row is
`0.49`, every `-∞` output is exact, the
inputs are unchanged and no element outside the output elements is written. The four wrong outputs
above fail `Post` on vocabularies `32000` and `151936`. 32 of 32 cases pass.
