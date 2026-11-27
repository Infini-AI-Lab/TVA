# `rejection_expand_f32`

**The kernel.** `expand_kernel` of vLLM's rejection sampler (`source/rejection_expand.py`, kernel
lines 107–128; host `expand_batch_to_tokens`, lines 66–104, launching through `_expand`, lines
131–148, on a 1-D grid of `batch_size` programs). Program `r` handles request `r`: it reads the
request's first position `start_r` (`0` for the first request, else `cu[r-1]`) and its end
`cu[r]` in the `int32` running sums `cu_num_tokens`, reads the request's value `x[r]`, replaces it
by `replace_to` when it equals `replace_from`, and stores it at the output positions
`start_r … cu[r] - 1` with one store of `MAX_NUM_TOKENS = 128` lanes, masked by
`offset < cu[r] - start_r`. vLLM uses it to expand the per-request `temperature` (`float32`,
`0 → 1`), `top_k` (`int32`, `0 → 0`) and `top_p` (`float32`, `0 → 0`) to the draft tokens. The
claim is `Claims.RejectionExpand.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). For each
request `r < B` and each position `j` with `start_r ≤ j < cu[r]`, output element `j` holds
`expandVal(x[r])`, bit for bit:
- on `float32`: the `float32` nearest to `replace_to` (ties to even) when `x[r] == a`, where `a` is
  the `float32` nearest to `replace_from` and `==` is the ordered float comparison (`-0.0` and
  `+0.0` are equal, a NaN equals nothing); otherwise `x[r]` itself, NaN payloads, infinities and
  signed zeros kept;
- on `int32`: `replace_to` when `x[r] = replace_from`, otherwise `x[r]`.

Every byte outside the output's first `cu[B-1]` elements is unchanged (the inputs, and any tail of
an output longer than the number of tokens).

**The conditions** (`Inst.ok`, `Pre`):
- `x` and the output are `float32` or `int32` (the dtypes vLLM's three calls pass);
- `1 ≤ B ≤ 8192` requests (one program each);
- `replace_from` and `replace_to` in `[-2^31, 2^31)`: the kernel's `int32` arguments;
- `cu` and `x` hold `B` entries; all three buffers valid in the initial memory;
- the output shares no byte with an input (the inputs may overlap one another);
- `cu` holds the inclusive running sums of the per-request token counts, each count
  `n_r = cu[r] - start_r` in `[0, 128]`, and `cu[B-1]` at most the output's length.

Every value of `x` and every initial output is covered.

**The launches covered** (`SETUP.md`): the docstring's example, the source test's seven random
batches of 1 to 1024 requests (0 to 128 tokens each) with the three calls of
`apply_sampling_constraints`, and the serving size (1024 requests of `MAX_SPEC_LEN = 128` tokens)
with the three calls (`setup_*_ok`; 25 instances). `mk_ok` covers every `B ≤ 8192`, any number of
tokens and either dtype.

**Tolerances.** None: the result is exact. The comparison and the selection are exact
operations, and on `float32` the two conversions of the `int32` arguments (`sitofp`) round once to
nearest (an `int32` is exact in `float64`, so the alternatives `tl.ItofRel` allows coincide). No
device fact is used.

## Exclusions

- **Token counts above 128.** The store has `MAX_NUM_TOKENS = MAX_SPEC_LEN = 128` lanes, so a
  request with more tokens would get only its first 128 positions written. vLLM's caller asserts
  `max_spec_len ≤ MAX_SPEC_LEN` (`RejectionSampler.forward`), so its launches have none.
- **More than 8192 requests**: the claim covers real batch sizes only.
- **`replace_from` or `replace_to` outside `[-2^31, 2^31)`**: Triton would pass it as an `int64`, a
  different compilation of the kernel; vLLM passes `0` and `1`.
- **Other dtypes** (the warm-up's `int64` inputs; no caller passes one), **an empty batch
  (`B = 0`)** and **an output overlapping an input**: outside the file's launches.

`check_gpu.py` (one H200) checks `Post` bit for bit on the `SETUP.md` launches and on edges (8192
requests, NaNs of several payloads, infinities, signed zeros, subnormals, `int32`-extreme and
`float32`-rounded replacement values, guarded direct launches with an output longer than the number
of tokens), and runs one launch beyond `Inst.ok` (70000 requests).
