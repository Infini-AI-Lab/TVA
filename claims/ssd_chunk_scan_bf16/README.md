# `ssd_chunk_scan_bf16`

**The kernel.** `_chunk_scan_fwd_kernel` of SGLang's Mamba-2 SSD (`source/ssd_chunk_scan.py`,
kernel lines 88–89 and 92–488; host `chunk_scan_fwd` lines 491–644, with the metadata builder
`query_start_loc_to_chunk_indices_offsets`, lines 647–679, and `make_inputs`, lines 682–732). It
computes the output of Mamba-2's chunked scan over a packed prefill batch (`batch = 1`, several
requests), with or without initial states, on `bfloat16` activations. A grid
`(cdiv(chunk_size, 64)·cdiv(headdim, 64), N, nheads)`; a program owns a `64 × 64` tile (rows of one
logical chunk, channels) of one head: it reads the entering state out through `C` and decays it,
adds the causal intra-chunk sum over `cb·exp(ΔdA)·dt·x` (a loop of `tl.dot`s over `K = 32` tokens)
and `D·x`, and stores `out` as `bfloat16`. The claim is `Claims.SsdChunkScan.claim` in
`Claim.lean`; the model is `chunk_scan_kernel` (with initial states) and
`chunk_scan_kernel_noinit` (without).

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for each
token `t < seqlen`, head `h < nheads`, channel `p < headdim`:
- `out[0, t, h, p]` holds a `bfloat16` value; when the inputs' values are covered (`Vals`), it is
  finite and `|out[0, t, h, p] − yRef| ≤ tol`;
- every byte outside `out`'s `seqlen·nheads·headdim` elements is unchanged.

`yRef` is Mamba-2's chunked-scan output, exact in the reals, from the host's inputs alone: with
`r` the request of token `t`, `c` its physical chunk, `l` the first token of its logical chunk
(the later of the chunk's and the request's first tokens) and `g` its group,
`yRef = exp(dSt)·Σ_n C[t, g, n]·S[p, n] + Σ_{s=l}^{t} cb·exp(dA_t − dA_s)·dt_s·x[s, h, p] +
D[h]·x[t, h, p]`, where the entering state `S` is the request's initial state (`0` without initial
states) when the request starts in chunk `c`, else `states[0, c]`. The request boundaries come from
`query_start_loc`, not from the kernel's metadata. `tol = (nIt + 256)·2^-15·mag + 2·expErr +
2^-110·(1 + hSt)` (Tolerances).

## Conditions

`Inst.ok`:
- `1 ≤ nheads < 2^16`, `headdim ≥ 1`, `dstate ≥ 1`, `ngroups ≥ 1` dividing `nheads`;
  `BLOCK_SIZE_DSTATE = max(next_power_of_2(dstate), 16) ≤ 128`; `chunk_size` a power of two at
  most `2^14`;
- the packed batch: `1 ≤ num_requests ≤ 8192`, `query_start_loc` from `0` to `seqlen`, each request
  at least one token, `seqlen ≤ 2^22`; the grid's `y` the host's `N` (`nchunks` without initial
  states), below `2^16`; fewer than `2^31` programs;
- `x`'s and `C`'s token strides hold the heads and the groups;
- the `int32` arguments and arithmetic: the batch strides of `x`, `C`, `out`, `cb`, `dt`,
  `dA_cumsum`, `states` below `2^31`, the in-chunk offsets below `2^31`, and with initial states
  `(num_requests − 1)·nheads·headdim·dstate < 2^31` (line 255's product, a kernel limit);
- the buffers hold their tensors; `out` shares no byte with the inputs.

`Pre`: the twelve buffers valid; the metadata the host's (`MetaOk`: `seq_idx`, and with initial
states `chunk_indices`, `chunk_offsets` as `hostMeta` transcribes the host's loop); without initial
states `cb` a finite zero across requests within a chunk (`CbZero`, as SGLang's `_bmm_chunk_fwd`
writes it).

`Vals` (the hypothesis of `Post`'s numeric clause): every value the reference reads finite; every
chunk state and initial state at most `2^120` in magnitude; every `dA_cumsum` entry `≤ 0`; and the
kernel's `float32` range: decay exponents at most `64`, and the intermediates of Tolerances step 7
at most `2^120`.

**The launches covered** (`SETUP.md`): `mk_ok` covers the host's launch built by `mk` from the flag,
the request lengths and the sizes; `setup_1 … setup_22` are the source's eleven test cases with and
without initial states, `setup_23 … setup_26` a Nemotron-H and a Nemotron Nano 2 rank at 128 heads
and 8192 tokens. `claim_satisfiable` uses `setup_1` in a zero memory.

**Device facts** the tolerance rests on (all in `Device P`): `Device.DotBF16` (`tl.dot` of
`bfloat16` operands into `float32`), `Device.ExpF32`, and the totality facts for `tl.dot` and
`exp`.

## Tolerances

Per output element, from the library's facts (`FloatV.sub_err`, `mul_err`, `add_err`,
`fma_err`, `cvt_err`: `float32` `u = 2^-24`, `η = 2^-150`, `bfloat16` `u = 2^-8`, `η = 2^-134`;
`Device.ExpF32`; `Device.DotBF16`: `relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`
of `|acc| + Σ liftMag a·liftMag b`), steps 1–7 give a fine bound, and step 8
relaxes it to `tol`. The names `errSig`, `ewS`, `errW`, `errMA`, `errPre` are this derivation's;
the others are `Claim.lean`'s:

1. **The entering state's decay** (line 344 with initial states): the exponent
   `dA_cs_m − dA_cs_m_boundary`; the boundary is the `+0` of line 269 when `o = 0`, so the IEEE `-`
   is exact, and otherwise errs by at most `dErrSt = 2^-24·|dSt| + 2^-150` (`sub_err`; two finite
   non-positive values do not overflow). Without initial states (line 340) `tl.exp(dA_cs_m)` runs
   on every row and `tl.where` keeps it on the rows whose `seq_idx` equals the request at the
   chunk's previous token (`0` in chunk 0): exactly the rows that do not start their request
   (`gated` is false), where `o = 0` and the exponent `dA_cs_m = dSt` is exact; on the others it
   keeps `0.0`. `tl.exp`: `ExpF32` at `X = |dSt| + dErrSt` gives
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
   `K_MAX ≤ chunk_size_limit ≤ cs`). `acc += tl.dot(cb, x)` is `tl.dot_add` (`TVA.Triton.Dot`):
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
   `(2^-23 + 2^-47)·(|x·D| + |acc|) + 3·2^-150 ≤ errMA` (room: `Vals`;
   `errMA = (2^-23 + 2^-47)·(|x·D| + aAcc) + 3·2^-150`).
6. **Composition and the store**: the `float32` value `r` before the store is within
   `errPre = errSig + errW + errDot + errMA` of `yRef` (`errW = Σ_s ewS·|x_s|`), and the store's
   `bfloat16` rounding errs by at most `2^-8·|r| + 2^-134 ≤ 2^-8·(|yRef| + errPre) + 2^-134`. So
   `|out − yRef| ≤ fine = (1 + 2^-8)·errPre + 2^-8·|yRef| + 2^-134`.
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

8. **The relaxation `fine ≤ tol`**, on every covered instance and `Vals` input (`u = 2^-8`,
   `f = 2^-24`; `BSD ≤ 128`, so `relDot16 BSD ≤ 130·2^-22` and `absDot16 BSD < 2^-118.9`;
   `cs ≤ 2^14`, so `nIt ≤ 512`). Lifts: `lSig ≥ (1 + u)·mSig + cSum·(2^-134 + 2^-126)`, so
   `u·mSig + 2^-134·cSum ≤ u·lSig`, and `eSig ≤ (u + relDot16 BSD)·lSig + absDot16 BSD`; `mW ≤ lW`
   and `Σ_s wMag·|x_s| ≤ lW`; `aSig ≤ 1.00004·hSt·lSig + (1 + f)·hSt·absDot16 BSD + 2^-150`;
   `|x·D| ≤ |yRef| + hSt·lSig + lW` (`hSt ≥ e^{dSt}`, `wHat ≥ |wTS|`).
   - `(1 + u)·errSig ≤ 0.003953·hSt·lSig + (1 + u)·mSig·eSt + (1 + u)(1 + f)·hSt·absDot16 BSD +
     (1 + u)·2^-150`.
   - `(1 + u)·errW ≤ 0.003922·lW + (1 + u)·Σ_s (|cb|·|dt|·errE + wAbs)·|x_s|`.
   - `(1 + u)·errDot ≤ nIt·1.7234·10^-5·(hSt·lSig + lW) + 2^-110.9 + 2^-118·hSt`
     (`2·nIt·(relDot16 32 + 2^-21) = nIt·72·2^-22`; `(1 + u)·2·nIt·(absDot16 32 + 2^-140) ≤
     1.004·1024·35·2^-126`).
   - `(1 + u)·errMA ≤ 2.5·10^-7·(|yRef| + hSt·lSig + lW) + 2^-140` (`aAcc ≤ 1.01·(hSt·lSig + lW)`
     plus floors).
   - `u·|yRef| + 2^-134`.

   Summed: the coefficients of `|yRef|`, `hSt·lSig` and `lW` are at most `0.003953 +
   nIt·1.73·10^-5 + 2.5·10^-7 ≤ 2^-7 + nIt·2^-15 = (nIt + 256)·2^-15`; the `tl.exp` and `wAbs`
   terms, at most `(1 + u)·expErr ≤ 2·expErr`; the rest at most `2^-110·(1 + hSt)`. So
   `fine ≤ (nIt + 256)·2^-15·mag + 2·expErr + 2^-110·(1 + hSt) = tol`. `check_gpu.py` evaluates
   both on every covered launch: `tol ≥ fine` everywhere, `tol/fine` at most `2.06` on the test
   launches and real ranks (median `2.0`), up to `193` on the edges where the lifts dominate
   (subnormal `x` and `C`, states at `2^120` with `C` near `2^-130`).

**Meaningful at the covered sizes.** The tolerance is about twice the fine bound, whose largest
term is the store's rounding `2^-8·|yRef|`; on the test launches it is a median 0.9% to 9% of
`|yRef|`, and the worst covered launch `check_gpu.py` runs reaches `0.48` of it.

## Exclusions

- **`x.stride(0)`, `C.stride(0)` or `out.stride(0)` at least `2^31`** (`x_arg`, `c_arg`, `out_arg`):
  the host then passes an `int64` argument, another specialisation outside the model's argument
  class (first reached at about 175K prefill tokens on a Nemotron Nano 2 rank, with chunked prefill
  off). With initial states, line 193's `int32` product `c_idx * chunk_size * stride_x_seqlen`
  would also wrap there, a limit of the kernel.
- **`(num_requests − 1)·nheads·headdim·dstate ≥ 2^31`** with initial states (`init_i32`): line
  255's `int32` product wraps and the kernel reads the wrong initial state, a limit of the kernel.
- **`cb`, `dt`, `dA_cumsum` or `states` batch strides at least `2^31`** (`cb_i32`, `dt_i32`,
  `st_i32`), `N ≥ 2^16`, `nheads ≥ 2^16`, `chunk_size > 2^14`: outside real use.
- **Values** outside `Vals`: non-finite inputs, a state above `2^120`, a positive `dA_cumsum`
  entry, a decay exponent above `64` or an intermediate above `2^120` (a limit of the kernel's
  `float32` arithmetic, step 7). The frame and termination hold on every input.
- **Contexts longer than `2^22` tokens, batches of more than `8192` requests**: real model shapes
  only.
- **`dstate > 128`** (the other compile-time branch, lines 362–381), **`z` given, `D` absent or
  `[nheads, headdim]`, `batch > 1`**: not `SETUP.md`'s launches.

`check_gpu.py` (one H200) checks `Post`, `Pre`, `Vals` and the frame on the 22 test-case launches
and 27 edges, each with and without initial states, and `Post`'s rejection of wrong outputs.
