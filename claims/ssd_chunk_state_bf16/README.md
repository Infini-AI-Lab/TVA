# `ssd_chunk_state_bf16`: the prepared claim

Kernel: `ssd_chunk_state_bf16` (SGLang's `_chunk_state_fwd_kernel`, the Mamba-2 SSD chunk states of
a packed prefill batch: `bfloat16` `x` and `B`, `float32` `dt`, `dA_cumsum` and states). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.SsdChunkState`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.SsdChunkState.main : Statement` (`Statement`:
`Claim.lean` line 637). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 337), and every initial memory `init` that `Pre` accepts (line 449: the six
buffers valid, `seq_idx` the host's, `MetaOk` line 443), every outcome of the launch is a normal end
(no fault, no hang) whose final memory `fin` satisfies `Post` (line 603).

**What `Post` asks** (line 603), two parts:
1. for every `c < nc d`, `h < nheads`, `p < headdim`, `n < dstate`: element `stIdx d c h p n`
   (line 324) of `states` holds some `float32` `v` (on every input); and when `Vals init d` holds
   (line 584), `v` is finite and `|v.toReal - sRef init d c h p n| ≤ tol init d c h p n`. Without
   `Vals`, only "holds a value" is asked. The reference `sRef` (line 494) is exact, in the reals:
   `Σ_{t ∈ toks d c} xV t h p · bV t h n · wRef c h t`, over the chunk's tokens of the request at
   its end (`toks`, line 465; `endTok`, `eReq`, lines 458, 461, from `qsl`), with
   `wRef = exp(dRef) · dtV` (lines 488, 484, 479) and `bV` read at group `grp d h` (lines 475, 469).
   `tol` (line 575) is `Σ_{t ∈ toks d c} |xV t h p| · errB c h t n + dotTol (steps d c)
   (dotMag c h p n)`: the per-token operand error `errB` (line 542, built from `dErr`, `expRel`,
   `errE`, `eHat`, `errS`, `sHat`, `errP`, `pHat`, lines 506–537) and the accumulation's
   `dotTol` (line 570, with `rhod`, `alphad`, lines 564–566; `dotMag`, line 555; `steps`, line
   560);
2. the frame: every byte outside `states`' first `sSb d` elements (line 304) is unchanged.

**What you are given** (`Assumptions`, line 616):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, when the launch is
  valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `exp_acc` (`Rules.Num.ExpF32`): for every `X ≥ 0`, `tl.exp` on `float32` (source line 200) on
  `[−X, X]` is within relative `2^-22 + 2^-23·X`, absolute `2^-140`;
- `exp_total` (`Rules.Num.MathTotal P .exp .binary32`): that `tl.exp` has a result;
- `dot` (`Rules.Num.DotBF16`, a `DotLiftAcc`): the `tl.dot` of source line 204 is within
  `relDot16 K · (|acc| + Σ liftMag a · liftMag b) + absDot16 K` of `acc + Σ a·b`, with room;
- `dot_total` (`Rules.Num.DotTotal P dotCfg`, `dotCfg` line 614): that `tl.dot` has a result.

**The kernel, as the model runs it** (`ssd_chunk_state_kernel`, line 50; `model` line 407,
`launch` line 425): a 3-D grid `(gx d, nc d, nheads)` (`gx`, `nc`, lines 293, 290). Program
`(pid, pid_c, pid_h)` owns one `64 × 128` tile (`pid_m`, `pid_n` from `pid`) of chunk `pid_c`'s
state for head `pid_h`. It loads `dA_cumsum` at the chunk's end and `seq_idx` at the chunk's last
token, then runs `tl.forRange 0 chunk_size_limit 64` (`int64` bounds, `steps d c` trips): five
masked loads (`x`, `B`, `dA_cumsum`, `seq_idx`, `dt`), `tl.sub`, `tl.exp`, `tl.mul`, `tl.where`
(scale `+0` for tokens of another request), `tl.mul` by `B`, `tl.cvt` to `bfloat16`, and
`tl.dot_add` into the accumulator. After the loop, one masked store of the tile. Programs write
disjoint tiles and use no atomics: the route is `TVA.meets_independent` (PROVER_GUIDE §4); the
loop writes nothing (`safe_forRange` with an invariant and `fun _ => ∅`).

**What the proof must establish.**
- each launch is valid and has `debug = false`: the grid is 3-D, so `Launch.Valid` is proved
  directly (`grid_y`, `grid_z`, `grid_lt`, with `gx d < 2^31` from `grid_lt`; the six listed block
  shapes);
- each program is `Safe`: no `int32` or `int64` offset wraps (`headdim_lt`, `dstate_lt`,
  `tok_lt`, `sX_lt`, `sB_lt`, `batch_lt`, `xhead_lt`, `bhead_lt`, `xtile_lt`, `btile_lt`, `dt_lt`,
  `st_lt`, `sttile_lt`); every enabled load is in bounds (`x_len`, `b_len`, `dt_len`, `dA_len`,
  `seq_len`, with `qsl_*` for the chunk's last token); the store is in bounds (`states_len`) on
  distinct lanes; `safe_idiv`, `safe_irem`, `safe_convert`, `safe_sub`, `safe_math` (`exp_total`),
  `safe_mul`, `safe_cvt`, `safe_dot_add` (`dot_total`) for the steps;
- footprints: the programs' tiles are disjoint, and `states` is disjoint from every read (`disj`);
- the kernel's `scale` is nonzero exactly on `toks d c`: `seq_idx_k` equals `seq_idx_last` on the
  chunk's tokens of request `eReq d c` (`MetaOk`, `reqOf`), and the masked lanes load `-1`;
- the value part of `Post`: each token's operand `b` is within `errB` of `B·wRef` (`FloatV.sub_err`,
  `exp_acc` at `X = |dRef| + dErr`, `FloatV.mul_err`, `FloatV.cvt_err`, room from `Vals`); the
  loop is within `dotTol` of the computed products' sum (`DotLift.dotAdd_step` with `dot`, then
  `DotLoop.accum`; each step's room from `dotMag ≤ 2^120` in `Vals` and `steps d c ≤ 1024` from
  `chunk_le`, via `DotLoop.bound_le`); README `## Tolerances` gives the composition;
- the frame: the bytes no program writes.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable` and the `setup_*_ok`
theorems; `bytes_disj` (line 676), `zero_i32` (line 804) and `valid_zero` (line 811) can be
reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance, from every memory `Pre`
allows, every launch ends normally and leaves `Post`:

* for each chunk `c < nchunks`, head `h < nheads`, `p < headdim`, `n < dstate`,
  `states[0, c, h, p, n]` holds a `float32` value; when the inputs' values are covered (`Vals`), it
  is finite and `|states[0, c, h, p, n] − sRef| ≤ tol`;
* every byte outside `states`' `nchunks·nheads·headdim·dstate` elements is unchanged.

**The reference** `sRef` is the Mamba-2 chunk state computed from the host's inputs alone: the
request boundaries `query_start_loc` (`Inst.qsl`), `x`, `B`, `dt` and `dA_cumsum`, read as exact
values from the initial memory. With `endTok c = min((c+1)·chunk_size, seqlen) − 1` the chunk's
last token and `reqOf t` the request of token `t` (the number of requests after the first that
start at or before `t`), chunk `c` sums the tokens `toks c = {t : c·chunk_size ≤ t ≤ endTok c,
reqOf t = reqOf (endTok c)}` (its own tokens of the request running at its end):

`sRef = Σ_{t ∈ toks c} x[0, t, h, p] · B[0, t, h // (nheads // ngroups), n] · exp(dRef t) · dt[0, h, c, i]`,

`dRef t = dA_cumsum[0, h, c, chunk_size − 1] − dA_cumsum[0, h, c, i]`, `i = t − c·chunk_size`: each
token decayed to the end of the chunk (position `chunk_size − 1`, the chunk's cumulative total,
which past `seqlen` is constant). `exp` is the real exponential and the arithmetic is exact.
Nothing in it is what the kernel reads or how it indexes: the request boundaries come from
`query_start_loc`, not from `seq_idx`; the tensors are read at the elements their shapes and
strides name (`xIdx`, `bIdx`, `dIdx`), not through the kernel's pointer arithmetic, `BLOCK_SIZE_*`
or loop. A kernel that sums another request's tokens, reads the wrong group of `B`, misses a token
(the last of a chunk, one in its middle), decays to the wrong position (`chunk_size − 2`) or reads
`dt` one position off fails `Post` (`check_gpu.py`'s negative tests). A missed token whose decayed
weight is far below the tolerance's share passes: under SGLang-like decay the first token of each
chunk's request segment (`toks c`) in chunks of 256 is weighted `exp(dA[255] − dA[i])`, in the
check `e^-20.1` to `e^-9.5` in chunks 0 and 1 (`i = 0`) and `e^-12.8` to `e^-7.5` in chunk 2
(token 600, `i = 88`), and a kernel that drops it is accepted; under a weak decay (`dA × 0.01`) it
is rejected.

**The tolerance** (`Claim.lean`, "The tolerance"; derivation under Tolerances), per element:

`tol = Σ_{t ∈ toks c} |x[0, t, h, p]| · errB t n + dotTol (steps c) (dotMag)`

where `errB t n` bounds the error of the kernel's `bfloat16` operand `b` against `B·exp(dRef)·dt`
(the subtraction, `tl.exp`, the two `float32` products and the rounding to `bfloat16`), `dotMag =
Σ_{t ∈ toks c} liftMag(x) · max(bHat, 2^-126)` the products' lifted magnitudes, `steps c =
cdiv(chunk_size_limit, 64)` the loop's `tl.dot` steps, and `dotTol T A = ((1 + ρ)^T − 1)·A +
T·(1 + ρ)^T·α` with `ρ = DotLoop.stepRel relDot16 64`, `α = DotLoop.stepAbs absDot16 64`.

## Conditions

`Inst`: `seqlen`, `nheads`, `headdim`, `dstate`, `ngroups`, `chunk_size`, `num_requests`,
`query_start_loc` (`qsl`), the strides of `x` (`sXb`, `sXs`, `sXh`, `sXp`) and of `B` (`sBb`,
`sBs`, `sBg`, `sBn`), and the six tensors by their storage (`Buf`). The flags are `HAS_SEQ_IDX =
True`, `batch = 1`, `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 64` (the host's line
253); `dt`, `dA_cumsum` and `seq_idx` are contiguous and `states` is fresh, with the strides the
host passes (`sSb`, …; the launch passes `dt.stride(0), stride(2), stride(1), stride(3)`, and the
same for `dA_cumsum`). `launch` is the grid `(cdiv(headdim, 64)·cdiv(dstate, 128), nchunks,
nheads)`, the source's block shapes, `enable_fp_fusion` at its default.

`Inst.ok`:
* positive sizes, `ngroups ∣ nheads` (asserted by the host), `chunk_size` a power of two (asserted
  by the code base) and at most `2^16` (`chunk_le`, Exclusions);
* the grid: `nchunks < 2^16` (CUDA's `y`), `nheads < 2^16` (`z`), fewer than `2^31` programs;
* the packed batch: `num_requests ≥ 1`, `qsl 0 = 0`, every request at least one token,
  `qsl num_requests = seqlen`; `nchunks = cdiv(seqlen, chunk_size)` (`nc`);
* the integer arguments and arithmetic: `headdim + 63`, `dstate + 127`, `nchunks·chunk_size` below
  `2^31`; the strides of `x` and `B` other than `stride(0)` below `2^31` (`int32` arguments); the
  five batch strides (`x.stride(0)`, `B.stride(0)`, `states.stride(0)`, `dt.stride(0)`,
  `dA_cumsum.stride(0)`) below `2^63`, `int32` or `int64` arguments (`batch_lt`: the kernel only
  multiplies each by the `int64` `pid_b = 0`, so the model takes their `int64` values and covers
  both classes); the `int32` products `(nheads − 1)·sXh`, `(ngroups − 1)·sBg`, the tile offsets
  `(headdim + 63)·sXp + 64·sXs` and `(dstate + 127)·sBn + 64·sBs` (every lane, masked ones
  included, and the per-step increments), `(nheads − 1)·nchunks·chunk_size`
  (`pid_h·stride_dt_head`), `nheads·headdim·dstate` (`states.stride(1)`) and the `states` tile
  offsets `(headdim + 63)·dstate + dstate + 127` below `2^31` (Exclusions);
* the buffers hold their tensors (the largest index of `x` and `B` inside their storage);
  `states` shares no byte with the inputs (the host allocates it). `x` and `B` may share storage,
  as in SGLang's layer.

`Pre`: the six buffers are valid (aligned, present), and `seq_idx[0, t] = reqOf t` for every
token (`MetaOk`: the host's `repeat_interleave(arange(num_requests), seqlens)`). No condition on
the `bfloat16` and `float32` values in `Pre`.

`Vals` (the hypothesis of `Post`'s numeric clause): every `x` and `B` entry of the packed tokens is
a finite `bfloat16` (a token of another request is multiplied by `0`, which needs a finite value);
every `dt` and `dA_cumsum` entry is finite, and every `dA_cumsum` entry is `≤ 0` (real use: sums of
`A·dt` with `A < 0` and `dt ≥ 0`, non-positive in any order); and the kernel's `float32` range: for
each summed token `dRef ≤ 64`, `eHat·|dt| ≤ 2^126` and `|B|·sHat ≤ 2^126`, and `dotMag ≤ 2^120`.
`dRef` may be slightly positive (a parallel scan's rounding can break the monotonicity of
`dA_cumsum` within a chunk); `dt` may have any sign.

The launches are within part (A) of `Compiler.Faithful`'s scope: the kernel is the `@triton.jit`
function of the source (without its autotuner, `SETUP.md`); the launch gives `num_warps = 4`,
`num_stages = 2` and no other option, no `@triton.heuristics`, no descriptors, no `CompiledKernel`,
no host `TensorDescriptor`. Part (B), the environment, is assumed and not checked.

## Assumptions

* `[rule] faithful`: `Rules.Compiler.Faithful P (model d) (launch d) (K d)` on every covered `d`;
* `[rule] exp_acc`: `Rules.Num.ExpF32 P` (`tl.exp` on `float32`, line 200);
* `[rule] exp_total`: `Rules.Num.MathTotal P .exp .binary32`;
* `[rule] dot`: `Rules.Num.DotBF16 P` (`tl.dot` of `bfloat16` into `float32`, line 204);
* `[rule] dot_total`: `Rules.Num.DotTotal P dotCfg`.

`consistent` instantiates them at `Platform.sm90ref`.

## Coverage

`SETUP.md` asks for the host's launches with `HAS_SEQ_IDX = True`, `batch = 1`, `bfloat16` `x` and
`B`, `float32` states, any number of requests of any lengths ≥ 1, `x` and `B` of any strides, and
the code base's sizes. Each of the source's ten test cases (lines 364-375) is an instance
`setup_1 … setup_10` built by `mk` from its request lengths, `nheads`, `headdim`, `dstate`,
`ngroups`, `chunk_size` and layout (the host's shapes and strides; for the layer's layout `x` and
`B` are views of one buffer of `conv_dim`-element rows), and the six edges of `check_gpu.py` and one long request are
`setup_e1 … setup_e7` (32768 tokens at 128 heads × 64 × 128; 200 one-token requests; chunks of
`1` and `2`; `headdim = dstate = 1`; `headdim = 65`, `dstate = 129`; 180,000 tokens); each `setup_*_ok` is proved by
deciding every condition. `satisfiable` uses `setup_1` (one request, so `seq_idx` is all zeros) in
a memory of zeros. The SETUP sizes beyond these (128 heads, `dstate` 256, `ngroups` up to `nheads`,
tens of thousands of tokens) are inside `Inst.ok`, and so are long requests whose `x.stride(0) = seqlen·conv_dim` is an
`int64` argument (from about 210,000 tokens; `setup_e7`, one request of 180,000 tokens at 128
heads × 80 × 128, is proved covered): for SGLang's layout at 128 heads × 64 × 128 with eight
groups and `chunk_size` 256, up to the grid's limit, `nchunks < 2^16` (16,776,960 tokens).

## Tolerances

For one token `t ∈ toks c` and one output element, from the library's facts (`docs/rules/num.md`;
`FloatV.sub_err`, `Rules.Num.ExpF32`, `FloatV.mul_err`, `FloatV.cvt_err`, `DotLift.dotAdd_step`,
`DotLoop.accum`):

1. The decay exponent (line 200): `dA_cs_last` and `dA_cs_k` are loaded exactly; their IEEE `-`
   (two finite non-positive values: no overflow) errs by at most `dErr = 2^-24·|dRef| + 2^-150`.
2. `tl.exp`: the computed input `d̃` lies in `[−X, X]` for `X = |dRef| + dErr`; `ExpF32` at that
   `X` gives the result `ẽ` within `rel·e^{d̃} + 2^-140`, `rel = 2^-22 + 2^-23·X` (`expRel`), with
   room since `d̃ ≤ 64 + dErr`. Then `|ẽ − e^{dRef}| ≤ (1 + rel)·e^{dRef + dErr} − e^{dRef} + 2^-140
   = errE`: for `d̃ ≥ dRef` directly; for `d̃ < dRef` and `rel ≤ 1`, the error is at most `e^{dRef}
   − (1 − rel)·e^{dRef − δ} + 2^-140` (`δ = dRef − d̃ ≤ dErr`), below `errE` because `2 ≤ (1 +
   rel)·e^δ + (1 − rel)·e^{−δ}`; for `d̃ < dRef` and `rel > 1` (only when `|dRef|` exceeds about
   `2^23`, which `Vals` allows), it is at most `rel·e^{d̃} + 2^-140 + e^{dRef} − e^{d̃} ≤
   rel·e^{dRef} + 2^-140`, below `errE`. In every case `|ẽ| ≤ eHat = e^{dRef} + errE`. `ExpF32` is
   instantiated per token at its `X`, so very negative exponents (strong decay) are covered: there
   `rel` grows but `rel·e^{dRef}` vanishes.
3. `scale = ẽ·dt` (line 200, the IEEE product; `tl.where` picks it since `seq_idx[t]` is the
   request of the chunk's last token, `MetaOk`): within `|dt|·errE + 2^-24·eHat·|dt| + 2^-150 =
   errS` of `wRef = e^{dRef}·dt`, and `|scale| ≤ sHat = |wRef| + errS` (room: `eHat·|dt| ≤ 2^126`).
4. `B·scale` (line 202, the IEEE product of the exactly widened `B`): within `|B|·errS +
   2^-24·|B|·sHat + 2^-150 = errP` of `B·wRef` (room: `|B|·sHat ≤ 2^126`), magnitude at most
   `pHat = |B·wRef| + errP`.
5. The rounding to `bfloat16` (line 203; `pHat` is below `bfloat16`'s range): within `2^-8·pHat +
   2^-134` more; so the operand `b̃` is within `errB = errP + 2^-8·pHat + 2^-134` of `B·wRef`, and
   `|b̃| ≤ bHat = |B·wRef| + errB`, its lifted magnitude at most `max(bHat, 2^-126)`.
6. The other lanes of each `tl.dot` hold exact zeros: a token of another request gets `scale = 0`
   (`tl.where`; `B` finite, so `b̃ = ±0`), a lane past `chunk_size_limit` loads `B = 0`, `x = 0` and
   `seq_idx = −1` (`≠ seq_idx_last ≥ 0`), and `x` is finite. So the exact sum of the dot products
   over the loop is `Σ_{t ∈ toks c} x·b̃`, within `Σ |x|·errB` of `sRef`.
7. The loop (line 204): each step is `tl.dot_add` of `64` products; `DotLift.dotAdd_step` (from
   `Num.DotBF16`, `relDot16 64 = 66·2^-22`, `absDot16 64 = 66·2^-126`, whichever form the combine
   pass gives) bounds a step by `ρ·(|acc| + A_j) + α` with `A_j` the step's lifted magnitudes, and
   `DotLoop.accum` gives `|acc − Σ x·b̃| ≤ DotLoop.bound ρ α T (Σ A_j) ≤ dotTol T dotMag` after the
   `T = steps c` steps. The room of each step: `chunk_size ≤ 2^16` (`chunk_le`) gives `T ≤
   2^16/64 = 1024`, and `ρ = 66·2^-22 + 3·2^-24·(1 + 66·2^-22) < 1.6·10^-5`, so `T·ρ < 0.017 ≤ 1/2`
   and `DotLoop.bound_le` gives `DotLoop.bound ρ α t A ≤ 2t·(ρ·A + α) ≤ 0.034·A + 2^-100` for every
   `t ≤ T`. With `A = dotMag ≤ 2^120` (`Vals`; `Σ A_j ≤ dotMag`, the zero lanes adding nothing), the
   accumulator before a step has `|acc| ≤ A + 0.034·A + 2^-100`, so `2·((1 + relDot16 64)·(|acc| +
   A_j) + absDot16 64 + 2η) ≤ 2·1.0001·2.04·2^120 + 2^-90 < 2^123 ≤ maxVal`: the condition of
   `DotLift.dotAdd_step` holds at every step, and each step's result is finite. (Without a bound on
   `T`, `(1 + ρ)^T` grows without limit and `dotMag ≤ 2^120` no longer gives this room.) The store
   (line 224) writes the `float32` accumulator unchanged.

Together: `|states − sRef| ≤ Σ |x|·errB + dotTol = tol`.

**Meaningful at the covered sizes.** The leading term is one `bfloat16` rounding of each scaled
`B`, `2^-8·Σ|x·B·wRef|`; the `tl.dot` part is about `66·2^-22·T` of `dotMag` (below `10^-4` of it
at 256 tokens) and the exponential's part `2^-22` relative. On the GPU (`check_gpu.py`) the error
reaches `0.992` of `tol` on the coherent worst case (every scaled `B` rounded down by just under
half an ulp) and `0.98` on random inputs, at elements dominated by one token; over all elements of
the 8192-token launches the median error is `0.10` of `tol` (the 99th percentile `0.44`), and the
median `tol` is `1.4%` of `|sRef|`. The bound is the worst case of the kernel's own roundings: a
sum of random-sign roundings over many tokens stays below it by about the square root of their
count. At values near `10^-20` (products below `float32`'s normal range) the absolute terms
dominate and `tol` exceeds the outputs, as any `float32` result does there.

## Exclusions

What `SETUP.md` allows and the claim leaves out:

* **`(nheads − 1)·nchunks·chunk_size ≥ 2^31`**: a limit of the kernel. Line 140 (and lines
  141-145) computes `pid_h * stride_dt_head` (`stride_dt_head = nchunks·chunk_size`) as an `int32`
  product of two `int32` values; from `2^31` it wraps to a negative offset, and the kernel reads
  before the storage of `dt` and `dA_cumsum`. `Inst.ok` keeps it below `2^31` (`dt_lt`). First
  launch that hits it: 128 heads, `chunk_size` 512 (so that `nchunks < 2^16`), 16,909,313 tokens.
  Real use (prefills of up to tens of thousands of tokens, a few hundred thousand for one long
  request) does not reach it.
* **`x` with a large head or feature stride**: a limit of the kernel. `pid_h * stride_x_head` (line
  138) and `offs_m * stride_x_hdim + offs_k * stride_x_seqlen` (line 155) are `int32` and wrap when
  `(nheads − 1)·sXh` or `(headdim + 63)·sXp + 64·sXs` reaches `2^31` (`xhead_lt`, `xtile_lt`; the
  same for `B`, `bhead_lt`, `btile_lt`). SGLang's layer passes rows of `conv_dim` elements (`sXh =
  headdim`, `sXp = 1`, `sXs = conv_dim`), far below; the first launch that hits it is a token-major
  `x` (`x.stride(1) = 1`, `x.stride(3) = seqlen`, which the caller accepts) with `(nheads −
  1)·headdim·seqlen ≥ 2^31`, e.g. 128 heads × 64 at 264,209 tokens, which `xhead_lt` excludes. Not
  part of real use.
* **`chunk_size > 2^16`**: outside real use. The loop runs up to `chunk_size / 64` `tl.dot`
  steps, and the error bound of the `float32` accumulation, `(1 + ρ)^T − 1` of the products' magnitudes,
  needs a bound on `T` to keep the accumulator inside `float32`'s range (Tolerances, step 7); with
  `chunk_size` up to `2^30` (allowed by `nchunks·chunk_size < 2^31` alone; up to `2^24` steps) an
  accumulator that takes the largest error `tl.dot` allows at every step can overflow `float32`
  although `dotMag ≤ 2^120`. `Inst.ok` keeps `chunk_size ≤
  2^16` (`chunk_le`); the code base's model configs use `chunk_size` 256 and SGLang's tests 8 to
  256 (`SETUP.md`).
* **`nchunks ≥ 2^16`** (CUDA's grid `y`; the launch fails) and **`nheads ≥ 2^16`**, `headdim + 63`,
  `dstate + 127`, `nchunks·chunk_size` or `nheads·headdim·dstate` at `2^31`: outside real use
  (16.8 million tokens at `chunk_size` 256).
* **Values** (`Vals`, a hypothesis of `Post`'s numeric clause; the frame and termination hold on
  every input): non-finite inputs; a positive `dA_cumsum` entry (outside real use: `A < 0`, `dt ≥
  0`); and, a limit of the kernel's `float32` arithmetic, a decay exponent above `64`, a scale or
  scaled `B` above `2^126`, or lifted product magnitudes above `2^120`, where `exp`, the products or
  the accumulator would overflow. Real use (activations of order 1 to `10^3`, decays `≤ 0`) is far
  inside.
* The launch without `seq_idx` (`HAS_SEQ_IDX = False`): SGLang's only serving caller,
  `python/sglang/srt/layers/attention/mamba/mamba.py:616`, always passes `seq_idx`
  (`mamba2_metadata.py:268-275`); only SGLang's unit tests compile that variant. Also `batch > 1`,
  `x` and `B` in `float16` or `float32`, and `states_in_fp32 = False`: not part of the setup
  (`SETUP.md`).

## Checks

* **Build**: `Claim.lean` builds with no error, no `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (at `Platform.sm90ref`) and `satisfiable` (`setup_1` in a zero
  memory) are proved.
* **Coverage**: `setup_1_ok … setup_10_ok` (the source's ten test cases) and `setup_e1_ok …
  setup_e7_ok` (the edges and one long request) are proved.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 24 launches (and the negative tests' two) hold: the ten test cases, six edges
  of `Inst.ok`, one with `int64` `x.stride(0)` and `B.stride(0)`, and seven of `Vals` (strong
  decay `dA × 200`; no decay; `dt × 1000`; a non-monotone
  `dA_cumsum` with decay exponents up to `+3`; values near `10^15`; near `10^-20`; the coherent
  `bfloat16` worst case). Worst error / tolerance: `0.992`. Every element is finite, `Vals` holds,
  the byte frame holds (the inputs unchanged, and guard bands of 4096 elements on each side of
  `states`, in which the check places the host's allocation, unchanged), and the host's `seq_idx`
  equals `reqOf` on every launch. The negative tests are rejected: one element moved by `1.01 × tol`
  (error / tolerance `1.01`), the launch without `seq_idx` (`271`), `B`'s groups swapped (`6590`),
  and, on a launch of 768 tokens in chunks of 256, the states of kernels that drop the last token of
  each chunk (`202`), decay to position `chunk_size − 2` (`76.7`), read `dt` one position off
  (`1190`), drop a middle token of each chunk (`1.43`), or drop the first token of each chunk's
  request segment under a weak decay `dA × 0.01` (`36.4`). Dropping that token under the inputs'
  own decay (weight `e^-20.1` to `e^-9.5` in chunks 0 and 1, `e^-12.8` to `e^-7.5` in chunk 2,
  where it is token 600 at position 88) is accepted (`0.040`).

## Library requests

None.
