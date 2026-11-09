# `ssd_chunk_scan_bf16`: the prepared claim

Kernel: `ssd_chunk_scan_bf16` (SGLang's `_chunk_scan_fwd_kernel`, the output of Mamba-2's chunked
scan over a packed prefill batch, with or without initial states, `bfloat16` activations). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.SsdChunkScan`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance, from every memory `Pre`
allows, every launch ends normally and leaves `Post`. For each token `t < seqlen`, head
`h < nheads` and channel `p < headdim`:

* `out[0, t, h, p]` holds a `bfloat16` value; when the inputs' values are covered (`Vals`), it is
  finite and `|out[0, t, h, p] − yRef t h p| ≤ tol t h p`;
* every byte outside `out`'s `seqlen·nheads·headdim` elements is unchanged.

**The reference** `yRef` is Mamba-2's chunked-scan output computed from the host's inputs alone: the
request boundaries `query_start_loc` (`Inst.qsl`) and the values of `cb`, `x`, `dt`, `dA_cumsum`,
`C`, `states`, `initial_states` and `D`, read as exact values from the initial memory. Token `t`
belongs to request `r = reqOf t` (the requests that start at or before `t`) and physical chunk
`c = t / chunk_size`, at position `i = t mod chunk_size`; its logical chunk starts at token
`l = max(c·chunk_size, qsl r)` (`lTok`), at offset `o = l − c·chunk_size` (`lOff`); its group is
`g = h / (nheads / ngroups)`. Then

`yRef t h p = exp(dSt)·Σ_{n<dstate} C[0,t,g,n]·S[p,n] + Σ_{s=l}^{t} cb[0,c,g,i,s−c·cs]·exp(dA_{c,i} − dA_{c,s−c·cs})·dt[0,h,c,s−c·cs]·x[0,s,h,p] + D[h]·x[0,t,h,p]`

where `dA_{c,j} = dA_cumsum[0, h, c, j]`, `dSt = dA_{c,i} − (dA_{c,o−1} if o > 0 else 0)`, and the
entering state `S` (`Sv`) is the state of request `r` entering its logical chunk:
* when `r ≥ 1` and request `r` starts in chunk `c` (`fromInit`), its initial state:
  `initial_states[r, h]` with initial states, `0` without (the request starts from the zero
  state, as SGLang runs a request without a cached prefix);
* otherwise `states[0, c, h]`, the state entering physical chunk `c` of the request running there:
  `states` is state passing's output, whose `out[0]` is request 0's initial state
  (`ssd_state_passing.py:110-112`; `0` without initial states) and whose `out[c]` for `c ≥ 1` is
  the state after chunk `c − 1` of the request that continues into chunk `c`.

`exp` is the real exponential and the arithmetic is exact. Nothing in the reference is what the
kernel reads or how it indexes: the request starts come from `query_start_loc`, not from
`seq_idx`, `chunk_indices` or `chunk_offsets`, and `cb` is read only on the pairs `l ≤ s ≤ t` of
one logical chunk. (Without initial states the kernel's loop runs over the whole physical chunk;
`Pre` gives the fact it relies on, that `cb` is a zero across requests, below.) A kernel that
reads the wrong metadata entry, takes the wrong entering state, misses the decay boundary or the
`D` term, or adds across requests fails `Post` (`check_gpu.py`'s negative tests).

**The tolerance** `tol` (`Claim.lean`, "The tolerance"; derivation under Tolerances) is

`tol = (1 + 2^-8)·errPre + 2^-8·|yRef| + 2^-134`,

the store's `bfloat16` rounding of a `float32` value within `errPre` of `yRef`, where `errPre =
errSig + errW + errDot + errMA` sums the error of the state term (the `bfloat16` rounding of the
states, line 360, the `tl.dot`, the computed decay `exp`, the product), of the intra-chunk weights
(the `exp`, two products, the `bfloat16` rounding of line 414), of the loop's `tl.dot`s, and of
the `D` term's `tl.mul_add`. Each is a magnitude computed from the inputs (`mSig = Σ|C||S|`,
`wMag = |cb|·eHat·|dt|`, …) times a few unit roundoffs.

## Conditions

`Inst`: `HAS_INITSTATES` (`hasInit`: `initial_states`, `chunk_indices` and `chunk_offsets` given,
or all `None`), `nheads` (`H`), `headdim` (`P`), `dstate` (`N`), `ngroups` (`G`), `chunk_size`
(`cs`), `seqlen` (`T`), `num_requests`, `query_start_loc` (`qsl`), the grid's `y` (`nmeta`:
`N = len(chunk_offsets)` with initial states, `nchunks` without), `BLOCK_SIZE_DSTATE` (`BSD`), the
token strides `xT` of `x` and `cT` of `C`, and the twelve tensors by their storage (`Buf`; without
initial states the three `None`s are empty). The other flags are `HAS_SEQ_IDX = HAS_D = IS_CAUSAL
= IS_TRITON_22 = True`, `D_HAS_HDIM = HAS_Z = False`, `batch = 1`, `BLOCK_SIZE_M = BLOCK_SIZE_N =
64`, `BLOCK_SIZE_K = 32` (the host's line 544). `model` is `chunk_scan_kernel` with initial states
and `chunk_scan_kernel_noinit` without: there `c_idx` is the `int64` program index, `c_off = 0`,
the loop runs over the physical chunk, and line 340 zeroes the state term of the rows whose
`seq_idx` differs from the request at the chunk's previous token. `x` and `C` have strides `(T·xT, xT, P, 1)` and
`(T·cT, cT, N, 1)`; every other tensor is contiguous with the strides the host passes (`model`;
note the launch passes `dt.stride(0), stride(2), stride(1), stride(3)`, and the same for
`dA_cumsum`). `launch` is the grid `(cdiv(cs, 64)·cdiv(P, 64), nmeta, H)`, the source's block shapes,
`enable_fp_fusion` at its default.

`Inst.ok`:
* `1 ≤ H < 2^16` (the grid's `z`), `P ≥ 1`, `N ≥ 1`, `G ≥ 1` dividing `H` (asserted);
  `BSD = max(next_power_of_2(N), 16) ≤ 128` (stated as: a power of two, `16 ≤ BSD ≤ 128`,
  `N ≤ BSD`, `BSD = 16` or `BSD < 2N`); `cs` a power of two (asserted), `cs ≤ 2^14`;
* the packed batch: `num_requests ≥ 1`, `qsl 0 = 0`, every request at least one token,
  `qsl num_requests = T`; `nmeta` the host's `hostN` (`nchunks` without initial states), `< 2^16`
  (the grid's `y`); fewer than `2^31` programs;
* `H·P ≤ xT`, `G·N ≤ cT` (the slices of a wider row);
* the `int32` arguments and arithmetic: `x.stride(0) = T·xT`, `C.stride(0) = T·cT` and
  `out.stride(0) = T·H·P` below `2^31` (`x_arg`, `c_arg`, `out_arg`: `int32` arguments, the class
  the model takes; then line 193's `int32` product `c_idx·cs·xT ≤ (T − 1)·xT` does not wrap, nor do
  lines 204 and 478's); the in-chunk offsets `(2cs + 64)·xT + 64 < 2^31` (`x_off`, likewise
  `c_off`, `out_off`); `nchunks·G·cs² < 2^31` (`cb.stride(0)`), `H·nchunks·cs < 2^31` (`dt`,
  `dA_cumsum`), `nchunks·H·P·N < 2^31` (`states.stride(0)`); with initial states
  `(num_requests − 1)·H·P·N < 2^31` (line 255's product, a kernel limit: Exclusions);
* the buffers hold their tensors; `out` shares no byte with the inputs (inputs may overlap each
  other: `x` and `C` are slices of one buffer in SGLang).

`Pre`: the twelve buffers are valid (aligned, present), the metadata are the host's (`MetaOk`):
`seq_idx[0, t] = reqOf t` (`repeat_interleave(arange(nreq), seqlens)`), and with initial states
`chunk_indices[k] = hostIdx k`, `chunk_offsets[k] = hostOff k`, where `hostMeta` transcribes the
host's loop (`query_start_loc_to_chunk_indices_offsets`, lines 647–679: `chunk_indices` starts as
`arange(N)`; for each later request `p` counts the starts not on a chunk boundary,
`chunk_indices[s//cs + p : e//cs + p + (e%cs > 0)] -= p`, `chunk_offsets[s//cs + p] = s % cs`);
and without initial states `cb` is a zero across requests (`CbZero`): for tokens `s ≤ t` of one
physical chunk in different requests, `cb[0, c, g, t mod cs, s mod cs]` is a finite zero. SGLang's
`_bmm_chunk_fwd` with `seq_idx` (`ssd_combined.py:165`) writes `+0` there (`ssd_bmm.py:179`); the
kernel's loop relies on it (line 406's comment). No other condition on the float values in
`Pre`.

`Vals` (the hypothesis of `Post`'s numeric clause): every value the reference reads is finite
(`C`, `x`, `D`, `dt`, `dA_cumsum`, and `cb` on the pairs `l ≤ s ≤ t` of a logical chunk; the other
`cb` entries may hold anything, NaN included, as SGLang leaves them past `seqlen`); every
`states` entry and, with initial states, every `initial_states` entry finite and at most `2^120` in
magnitude (each may be read and rounded to `bfloat16` at line 360, and a `float32` value above
`bfloat16`'s overflow threshold rounds to infinity); every `dA_cumsum` entry `≤ 0` (real use:
cumulative sums of `A·dt`, `A < 0`, `dt ≥ 0`); and the kernel's `float32` range, for every output
row `t` in range: `dSt ≤ 64`; on every pair `s ≤ t` the loop computes (`kToks`: the logical chunk
with initial states, the whole physical chunk without) `dS ≤ 64`, `|cb|·eHat(dS) ≤ 2^120` (line
408) and `wMag = |cb|·eHat·|dt| ≤ 2^120` (line 410); the read-out room of the state the kernel
reads (`lSigL`, line 361: on a row whose state term line 340 zeroes, `states[0, c]`) `≤ 2^120`;
and `hSt·(mSig + eSig)`, `aSig + lW`, `aAcc + |x·D|` at most `2^120` (Tolerances, Exclusions).
Within a chunk `dA_cumsum` need not be monotone, so a decay exponent may be slightly positive; the
tolerance covers it.

The launches are within part (A) of `Compiler.Faithful`'s scope: the kernel is the `@triton.jit`
function of the source (without its autotuner, `SETUP.md`); the launch gives `num_warps = 4`,
`num_stages = 1` and no other option, no `@triton.heuristics`, no descriptors, no `CompiledKernel`,
and no host `TensorDescriptor`. Part (B), the environment, is assumed and not checked.

## Assumptions

* `[rule] faithful`: `Rules.Compiler.Faithful P (model d) (launch d) (K d)` on every covered `d`;
* `[rule] dot`: `Rules.Num.DotBF16 P` (`tl.dot` of `bfloat16` operands into `float32`, lines 361,
  420);
* `[rule] dot_total`: `Rules.Num.DotTotal P cfgBF16`;
* `[rule] exp_acc`: `Rules.Num.ExpF32 P` (`tl.exp` on `float32`, lines 340, 344, 408);
* `[rule] exp_total`: `Rules.Num.MathTotal P .exp .binary32`.

`consistent` instantiates them at `Platform.sm90ref`.

## Coverage

`SETUP.md` asks for the host's launches with the flags above, `HAS_INITSTATES` either way,
`batch = 1`, any number of requests of any lengths ≥ 1, and the code base's sizes. Each of the
source's eleven test cases (lines 781–793), which the test runs with and without initial states,
is two instances, `setup_1 … setup_11` (with) and `setup_12 … setup_22` (without); two model
ranks the GPU check runs at 128 heads and 8192 tokens, Nemotron-H (`headdim` 64) and Nemotron Nano
2 (`headdim` 80), are four more, `setup_23 … setup_26` (with and without). Each is built by `mk`
from the flag, its request lengths, heads, `headdim`, `dstate`, groups, `chunk_size` and
`BLOCK_SIZE_DSTATE` (the host's shapes, `x` and `C` with the token stride `conv_dim` of
`make_inputs`, the grid's `y`, the tensors at distinct aligned addresses), with
`setup_i_ok : setup_i.ok` proved by `decide`. `satisfiable` uses `setup_1` in a memory of zeros.
The SETUP sizes beyond the tests are inside `Inst.ok`: for example a Nemotron Nano 2 rank (128
heads × 80, `dstate` 128, 8 groups, `conv_dim` 12288, chunks of 128) up to 174,762 tokens, a
Nemotron-H rank (`conv_dim` 10240) up to 209,715 tokens.

## Tolerances

Per output element, from the library's facts (`FloatV.sub_err`, `mul_err`, `add_err`,
`fma_err`, `cvt_err`: `float32` `u = 2^-24`, `η = 2^-150`, `bfloat16` `u = 2^-8`, `η = 2^-134`;
`Rules.Num.ExpF32`; `Rules.Num.DotBF16`: `relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`
of `|acc| + Σ liftMag a·liftMag b`; `docs/rules/num.md`):

1. **The entering state's decay** (line 344 with initial states): the exponent
   `dA_cs_m − dA_cs_m_boundary`; the boundary is the `+0` of line 269 when `o = 0`, so the IEEE `-`
   is exact, and otherwise errs by at most `dErrSt = 2^-24·|dSt| + 2^-150` (`sub_err`; two finite
   non-positive values do not overflow). Without initial states (line 340) `tl.exp(dA_cs_m)` runs
   on every row and `tl.where` keeps it on the rows whose `seq_idx` equals the request at the
   chunk's previous token (`0` in chunk 0): exactly the rows that do not start their request
   (`gated` is false), where `o = 0` and the exponent `dA_cs_m = dSt` is exact; on the others it
   keeps `0.0`. `tl.exp`: as in the state-passing derivation, `ExpF32` at `X = |dSt| + dErrSt` gives
   `|ẽ − e^{dSt}| ≤ errE(dSt, dErrSt) = (1 + rel)·e^{dSt + δ} − e^{dSt} + 2^-140`
   (`rel = 2^-22 + 2^-23·X`) and `|ẽ| ≤ eHat = e^{dSt} + errE` (`eSt`, `hSt`).
2. **The state term's `tl.dot`** (lines 355–361): `S̃ = bfloat16(S)` with `|S̃ − S| ≤ 2^-8|S| +
   2^-134` (`cvt_err`), so `|Σ C·(S̃ − S)| ≤ 2^-8·mSig + 2^-134·cSum`; the dot over `K = BSD`
   (padding lanes `n ≥ dstate` load `0` and add nothing; `liftMag 0 = 0`) is within
   `relDot16 BSD·L + absDot16 BSD` of `Σ C·S̃`, with `L ≤ lSig = Σ (|C| + 2^-126)·((1 + 2^-8)|S| +
   2^-134 + 2^-126)` (`liftMag q ≤ |q| + 2^-126`). So `|σ̃ − σ| ≤ eSig`. The product by `ẽ` (an
   IEEE `*`): `|acc₀ − e^{dSt}·σ| ≤ (1 + 2^-24)·eSig·hSt + 2^-24·mSig·hSt + mSig·eSt + 2^-150 =
   errSig`, and `|acc₀| ≤ aSig`. On a row whose state term line 340 zeroes, the kernel reads out
   `states[0, c]` all the same (`SvL`, with the room `lSigL`), finite by `Vals`, and multiplies it
   by `0.0`: `acc₀ = ±0`, the reference's state term is `0` (`S = 0`, `mSig = 0`), and
   `errSig ≥ 2^-150` covers it.
3. **Each intra-chunk weight** (lines 408–414, a pair `s ≤ t`): the exponent's `-` errs by
   `dErrS = 2^-24·|dS| + 2^-150`, `tl.exp` by `errE(dS, dErrS)`; then two `float32` products and a
   `bfloat16` rounding. With `W' = |cb|·|ẽ|·|dt| ≤ wMag`, `|w̃ − w| ≤ |cb||dt|·errE +
   ((1 + 2^-8)(1 + 2^-24)² − 1)·W' + (abs)`, and `(1 + 2^-8)(1 + 2^-24)² − 1 ≤ 2^-8 + 3·2^-24`; the
   absolute terms `(1 + 2^-8)(1 + 2^-24)·2^-150·|dt| + (1 + 2^-8)·2^-150 + 2^-134` are below `wAbs =
   2^-133 + 2^-148·|dt|`. So `|w̃ − w| ≤ ewS` and `|w̃| ≤ wHat`; pairs `s > t` give `+0` (line 413),
   whatever `cb`, `exp` or `dt` hold there. Without initial states the loop also computes the pairs
   `c·cs ≤ s < l` of earlier requests in the chunk: `cb` is a zero there (`Pre`), `exp` finite
   (`dS ≤ 64`), so the weight is `±0` and adds nothing to the dot (`liftMag 0 = 0`).
4. **The loop's `tl.dot`s** (line 420, `K = 32` per trip, at most `nIt = cdiv(cs, 32)` trips:
   `K_MAX ≤ chunk_size_limit ≤ cs`). `acc += tl.dot(cb, x)` is `tl.dot_add` (`Ops.lean:651-655`):
   either the dot into the accumulator, within `relDot16 32·(|acc_j| + L_j) + absDot16 32`, or the
   dot with a zero accumulator (within `relDot16 32·L_j + absDot16 32`) followed by the fusable
   addition, whose operands may enter unrounded (`AddRel`): that addition costs at most
   `2.01·2^-24·(|acc_j| + (1 + relDot16 32)·L_j + absDot16 32) + 3.01·2^-150`, a few ulps. Both
   branches are within `(relDot16 32 + 2^-21)·(|acc_j| + L_j) + absDot16 32 + 2^-140` per trip.
   `|acc_j| + L_j ≤ aSig + lW + E` with `lW = Σ_s (wHat + 2^-126)(|x| + 2^-126)` and `E` the
   errors so far (`|Σ_block w̃·x| ≤ L_block`); as `nIt·(relDot16 32 + 2^-21) ≤ 512·36·2^-22 ≤ 1/2`,
   the total is `E ≤ 2·nIt·((relDot16 32 + 2^-21)·(aSig + lW) + absDot16 32 + 2^-140) = errDot`. The rows `x[s]` with `s ≥ limit`
   load `0`; the `x` values met are exact. So `|acc − (acc₀ + Σ_s w̃·x)| ≤ errDot` and `|acc| ≤ aAcc
   = aSig + mW + errDot`.
5. **The `D` term** (line 442, `tl.mul_add`): either `fl(fl(x·D) + acc)` or `fma`, within
   `(2^-23 + 2^-47)·(|x·D| + |acc|) + 3·2^-150 ≤ errMA` (room: `Vals`).
6. **Composition and the store**: the `float32` value `r` before the store is within
   `errPre = errSig + errW + errDot + errMA` of `yRef` (`errW = Σ_s ewS·|x_s|`), and the store's
   `bfloat16` rounding errs by at most `2^-8·|r| + 2^-134 ≤ 2^-8·(|yRef| + errPre) + 2^-134`. So
   `|out − yRef| ≤ tol`.
7. **No overflow.** Each `float32` value the kernel computes for a stored output is bounded by a
   `Vals` quantity, `2^120` with at most a few relative errors of `2^-5` or less on top, a factor
   about `2^8` below the largest finite `float32` and `bfloat16` values (`≈ 2^128`): line 360's
   roundings of `|S| ≤ 2^120` are finite; line 361's dot has the room `lSigL ≤ 2^120` (on every
   row, the state the kernel reads); the product by `ẽ` is at most `(1 + 2^-24)·hSt·(mSig + eSig)`
   (or `0` on a zeroed row); the decays' arguments are at most `64 + 2^-17`, so `tl.exp` is below
   `2^93`; line 408's product is at most `(1 + 2^-24)·|cb|·eHat ≤ 2^120(1 + 2^-24)` and line 410's
   at most `(1 + 2^-24)·wMag`, on every pair `s ≤ t` the loop computes, so line 414's rounding is
   finite; line 420's dots have the room `aSig + lW + errDot`; line 442's operands are at most
   `|x·D| ≤ 2^120` and `aAcc ≤ 2^120`, and the store rounds `|r| ≤ 2^121`. Every other lane is
   discarded before it meets a stored value: pairs `s > t` become `+0` at line 413 (an infinity or
   NaN there included), rows past the logical chunk or `seqlen` are never stored, and a `tl.dot`'s
   output row depends only on its own row of the left operand (padding lanes load `0`). The
   launches `check_gpu.py` runs outside `Vals` show each bound is needed: the `float32` maximum in a
   state (line 360 rounds it to infinity, and `0·∞` at line 361 gives NaN), in `cb` with `dt = 0`
   (line 408 overflows, line 410 gives `∞·0`), a cross-request decay above `64` without initial
   states (`exp` overflows times the zero `cb`), a decay boundary above `64`, a state read-out
   above the `float32` range, and `x` near the `bfloat16` maximum: each gives non-finite outputs.

**Meaningful at the covered sizes.** The margin is the store's half-ulp: the tolerance's largest
term is the store's rounding `2^-8·|yRef|`, a bound on half a `bfloat16` ulp that a value just
below a power of two nearly reaches, so an element's error / tolerance can come within about
`0.996` of `1` from that rounding alone. The launches of `check_gpu.py` sample this: the worst
covered launch reaches `0.984` (0.86 to 0.98 on most launches), so the
tolerance is the kernel's own worst case, not a loose bound. The next terms are the `bfloat16` roundings of the states and of the weights, which the edge "states rounding one way" meets (`0.965`). The median
tolerance is 0.5% to 5% of `|yRef|` (the sums cancel: `Σ|C||S|` exceeds `|Σ C·S|`). On subnormal
inputs (`x`, `C` near `10^-39`) the absolute floor of `DotBF16` (`absDot16 = (K+2)·2^-126` per
`tl.dot`) dominates, about `10^-35`, four orders above those outputs.

## Exclusions

What `SETUP.md` allows and the claim leaves out:

* **`x.stride(0)`, `C.stride(0)` or `out.stride(0)` at least `2^31`** (`x_arg`, `c_arg`,
  `out_arg`): the host then passes an `int64` argument, another specialisation, outside the
  model's argument class. First launch outside: a Nemotron Nano 2 rank on one GPU (128 heads × 80,
  `ngroups` 8, `dstate` 128: `conv_dim` 12288) with 174,763 prefill tokens; 209,716 tokens at
  `conv_dim` 10240 (Nemotron-H 8B), 254,201 at 8448 (Granite 4.0-H). The kernel itself is right up
  to 174,848 tokens at `conv_dim` 12288. From 174,849 tokens (physical chunk 1366 with chunks of
  128), with initial states, line 193's `int32` product `c_idx * chunk_size * stride_x_seqlen`
  (`1366·128·12288 ≥ 2^31`) wraps and the kernel reads and writes the wrong rows, a limit of the
  kernel; without initial states `c_idx` is `int64` and the product does not wrap. Only that
  product wraps: the in-chunk offsets (lines 329–331, 387–389, 436–438, 481–483) add to a 64-bit
  pointer, and wrap only at `(2·cs + 64)·conv_dim ≥ 2^31` (`x_off`, `c_off`, `out_off`), far
  outside. Real use reaches the limit only with chunked prefill off: on the H200 it is on, with a
  budget of 8192 tokens per forward pass (`memory_hook.py:146-153`); with it off, only a batch's
  first request may exceed the input-token cap (`schedule_policy.py:1626-1632`), and one request
  stays below the model's context length, 131,072 tokens for Nemotron Nano 2, below its limit.
* **`(num_requests − 1)·nheads·headdim·dstate ≥ 2^31`** with initial states (`init_i32`): a limit
  of the kernel. Line 255 computes `seq_idx_m * stride_init_states_batch` in `int32`, which wraps;
  the kernel then reads the wrong initial state. First launch that hits it: 128 heads × 64 × 128
  (`2^20` per request) and 2049 requests, or 128 × 80 × 128 and 1640 requests (an `initial_states`
  tensor of 8 GiB). Real use reaches it only with a prefill batch of that many requests at that
  head size.
* **`nchunks·nheads·headdim·dstate ≥ 2^31`** (`st_i32`), **`nchunks·ngroups·cs² ≥ 2^31`**
  (`cb_i32`), **`nheads·nchunks·cs ≥ 2^31`** (`dt_i32`): `states.stride(0)` (`cb.stride(0)`,
  `dt.stride(0)`) would be an `int64` argument, a different specialisation, and `c_idx *
  stride_states_chunk` (line 214) an `int32` product that wraps with initial states, a kernel
  limit. These need an 8 GiB `states` tensor (524,288 tokens at `2^20` per chunk-head set with
  chunks of 256; 209,664 for Nemotron Nano 2 with chunks of 128), or 1 million tokens (`cb`, 8
  groups, chunks of 256): beyond the token limit above. Likewise `N ≥ 2^16` and `nheads ≥ 2^16`
  (CUDA's grid `y` and `z`), `chunk_size > 2^14` (real use: 8 to 256): outside real use.
* **Values** (`Vals`, a hypothesis of `Post`'s numeric clause; the frame and termination hold on
  every input): non-finite inputs; a state above `2^120` in magnitude; a positive `dA_cumsum` entry
  (outside real use: `A < 0`, `dt ≥ 0`); and, a limit of the kernel's `float32` arithmetic, a
  decay exponent above `64` or an intermediate above `2^120` (Tolerances, step 7), where a
  rounding, `exp`, a product or the accumulator would overflow. Real use (activations and states
  of order 1 to `10^3`, decays `≤ 0`, `dA_cumsum` non-increasing along a chunk) is far inside.
* **`dstate > 128`** (`BLOCK_SIZE_DSTATE > 128`, e.g. the larger Falcon-H1 models' `dstate` 256):
  the other compile-time branch (lines 362–381, a loop over `dstate`), not part of the setup.
* **`z` given, `D` absent or `[nheads, headdim]`, `batch > 1`**: other specialisations, not part of
  the setup (`SETUP.md`).

## Checks

* **Build**: `Claim.lean` builds with no error, no `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (at `Platform.sm90ref`) and `satisfiable` (`setup_1` in a zero
  memory) are proved.
* **Coverage**: `setup_1_ok … setup_26_ok` (the source's eleven test cases and the Nemotron-H and
  Nemotron Nano 2 ranks, with and without
  initial states) are proved.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 76 covered launches hold: the 22 test-case
  launches and 27 edges, each with and without initial states (chunk sizes 2, 16 and 1024;
  `dstate` 1, 17, 100; `headdim` 1, 65, 128; one group for 128 heads; 200 one-token requests;
  chunk-aligned starts; seven starts in one chunk; contiguous `x` and `C`; Nemotron-H and
  Nemotron Nano 2 ranks at 8192 tokens and 128 heads; 24 heads in one group at 8192 tokens; 32,768
  tokens; strong and no decay; a non-monotone `dA_cumsum`; values near `10^15`; states at `2^120`;
  subnormal `x` and `C`; states whose `bfloat16` roundings all go one way; `cb` at the `float32`
  maximum and NaN on the pairs `s > t`; `dt` at the `float32` maximum where `cb` is `0`). Worst
  error / tolerance `0.984` (a sample; the store's half-ulp allows about `0.996`). `Pre` (the host's metadata equal `reqOf`/`hostIdx`/`hostOff`/`hostN`;
  without initial states `cb` is `0` across requests), `Vals` and the byte frame hold (the inputs
  unchanged bit for bit, and guard bands of 4096 elements on each side of `out`, placed in a
  sentinel-filled buffer, unchanged); `cb` holds NaN on entries past `seqlen`. Eight launches
  outside `Vals` (step 7) fail `Vals` as they must, and give non-finite outputs with the frame
  intact. `Post` rejects a launch with `chunk_offsets` all `0`, an output moved by twice its
  tolerance (with and without initial states), and the readings "the first request's chunk-0
  state from `initial_states`", "no decay boundary", "no `D` term" and, without initial states,
  "a request that starts in a chunk continues `states[0, c]`" (`6·10^4`, `2.2`, `136`,
  `2.6·10^5`, `1.1·10^4` and `87` times the tolerance).

## Library requests

None. (The `int32` edges near `2^31` need tensors of several GiB and are not run on the GPU.)
