# `ssd_chunk_state_bf16`

**The kernel.** SGLang's Mamba-2 SSD `_chunk_state_fwd_kernel` (`source/ssd_chunk_state.py`,
kernel lines 77–224; host `chunk_state_fwd` lines 227–303), on a packed prefill batch
(`HAS_SEQ_IDX = True`, `batch = 1`, `seq_idx` the request of each token), `bfloat16` `x` and `B`,
`float32` `dt`, `dA_cumsum` and states, `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`,
`BLOCK_SIZE_K = 64`. The grid is `(cdiv(headdim, 64)·cdiv(dstate, 128), nchunks, nheads)`; each
program loops over its chunk in steps of `64` tokens, scales `B` by `exp(dA_last − dA_t)·dt`
(zero for tokens of another request than the chunk's last), rounds it to `bfloat16`, accumulates
`x^T·B` with `tl.dot` into `float32`, and stores its `64 × 128` tile of the chunk's state. The
claim is `Claims.SsdChunkState.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- every element `states[0, c, h, p, n]` holds a `float32` value;
- when the input values are covered (`Vals`), it is finite and within `tol` of the exact chunk
  state `sRef = Σ_{t ∈ toks c} x[t, h, p]·B[t, grp h, n]·exp(dRef)·dt[h, c, t]`, where `toks c` are
  chunk `c`'s tokens of the request running at its end and `dRef` is `dA_cumsum` at the chunk's
  end less `dA_cumsum` at `t`;
- every byte outside `states` is unchanged.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the source's ten test cases (`setup_1 …
setup_10`) and seven edges (`setup_e1 … setup_e7`: 32768 tokens at 128 heads × 64 × 128, 200
one-token requests, chunks of `1` and `2`, `headdim = dstate = 1`, `headdim = 65`, `dstate = 129`,
one request of 180,000 tokens). `mk` builds the host's instance from the request lengths, sizes and
layout (contiguous, or SGLang's layer's rows of `conv_dim` elements).

**Device facts** the tolerance rests on (all in `Device P`): `Device.ExpF32` (`tl.exp` on
`float32`: relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute `2^-140`), `Device.DotBF16` (`tl.dot`
of `bfloat16` operands into `float32`: within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`,
`L = |acc| + Σ liftMag a·liftMag b`), and the totality facts for `exp` and `tl.dot`.

## Conditions

`Inst.ok`:
- positive sizes, `ngroups ∣ nheads`, `chunk_size` a power of two, at most `2^16`;
- the grid: `nchunks < 2^16`, `nheads < 2^16`, fewer than `2^31` programs;
- the packed batch: at least one request, `qsl 0 = 0`, every request at least one token,
  `qsl num_requests = seqlen`; `seqlen ≤ 2^22`, `num_requests ≤ 8192`;
- the integer arguments: the strides of `x` and `B` other than `stride(0)` are `int32`; the five
  batch strides are `int32` or `int64`; the kernel's `int32` products and tile offsets do not wrap
  (`(nheads − 1)·sXh`, `(ngroups − 1)·sBg`, the tile offsets of `x`, `B` and `states`,
  `(nheads − 1)·nchunks·chunk_size`, `nheads·headdim·dstate`);
- the buffers hold their tensors, and `states` shares no byte with an input (`x` and `B` may share
  storage, as in SGLang's layer).

`Pre`: the six buffers are valid, and `seq_idx[0, t]` is the request of token `t` (`MetaOk`). Any
input values are covered; `Vals`, the hypothesis of `Post`'s numeric clause, asks: `x` and `B`
finite; `dt` and `dA_cumsum` finite, `dA_cumsum ≤ 0`; for each summed token `dRef ≤ 64`,
`eHat·|dt| ≤ 2^126` and `|B|·sHat ≤ 2^126`; and `dotMag ≤ 2^120`.

## Tolerances

`tol = 2^-8·(1 + steps·2^-6)·dotMag + 2^-100·inMag`, with `dotMag` the products' lifted magnitudes
`Σ liftMag x·max(bHat, 2^-126)` over `toks c`, `steps = cdiv(chunk_size_limit, 64)`, and
`inMag = Σ (1 + |x|)(1 + |B|)(1 + |dt|)`. Per token and element:
1. the decay exponent's IEEE subtraction errs by `dErr = 2^-24·|dRef| + 2^-150`;
2. `tl.exp` (`Device.ExpF32` at `X = |dRef| + dErr`) is within `errE` of `e^{dRef}`, also for very
   negative exponents;
3. `scale = ẽ·dt`, then `B·scale`, each an IEEE product (`errS`, `errP`);
4. the rounding to `bfloat16` adds `2^-8·pHat + 2^-134` (`errB`);
5. the dot's other lanes hold exact zeros (another request's tokens get `scale = 0`, lanes past
   the chunk load zeros), so the exact sum is `Σ_{toks c} x·b̃`, within `Σ |x|·errB` of `sRef`;
6. the loop: each step is one `tl.dot` of `64` products (`Device.DotBF16`, `relDot16 64 = 66·2^-22`);
   `chunk_size ≤ 2^16` bounds the steps by `1024`, so the accumulated error is at most
   `((1 + ρ)^T − 1)·dotMag + T·(1 + ρ)^T·α` and every step stays inside `float32`'s range.

Relaxed: `errE ≤ 2^-16·e^{dRef} + 2^-134`, so `Σ|x|·errB ≤ (2^-8 + 1.545·10^-5)·dotMag +
2^-133·inMag`; `(1 + ρ)^T − 1 ≤ T·1.605·10^-5`; and `1.545·10^-5 + T·1.605·10^-5 ≤ 2^-8·T·2^-6`.
The leading term is one `bfloat16` rounding of each scaled `B`; on the source's test cases the
median `tol` is `0.7%` to `1.5%` of `|sRef|`, and a kernel that drops a middle token of each chunk
fails it.

## Exclusions

- **`(nheads − 1)·nchunks·chunk_size ≥ 2^31`**: a kernel limit (`pid_h·stride_dt_head` is an
  `int32` product and wraps); real use does not reach it.
- **`x` or `B` with strides that make the `int32` head or tile offsets wrap**: a kernel limit;
  SGLang's layer's layout is far below.
- **`chunk_size > 2^16`**: outside real use (256 in the code base); the accumulation bound needs a
  bound on the steps to keep the accumulator in range.
- **`nchunks ≥ 2^16`** (CUDA's grid `y`), **`nheads ≥ 2^16`**, sizes at `2^31`: outside real use.
- **`seqlen > 2^22`, more than `8192` requests**: real shapes only.
- **Values outside `Vals`**: non-finite inputs, positive `dA_cumsum`, and the kernel's `float32`
  range (a decay exponent above `64`, scaled values above `2^126`, product magnitudes above
  `2^120`). The frame and termination hold on every input.
- **`HAS_SEQ_IDX = False`, `batch > 1`, other dtypes, `states_in_fp32 = False`**: not part of
  `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the setup launches and edge cases.
