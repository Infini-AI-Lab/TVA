# `attn_fwd_f16`

**The kernel.** `_attn_fwd` of Triton 3.8.0's tutorial 06, "Fused Attention"
(`source/06-fused-attention.py`, kernel lines 176–245, helpers `_attn_fwd_inner` lines 47–110 and
`_maybe_make_tensor_desc` lines 168–173; host `_attention.forward` lines 505–580), in the `float16`
run with the config of `SETUP.md` (`BLOCK_M = 128`, `BLOCK_N = 64`, the autotune decorator removed
and its `pre_hook` done by the host). The grid is `(cdiv(N_CTX, 128), Z·H)`; each program takes a
`128`-row query tile of one head, loops over the keys in blocks of `64` with an online softmax
(`tl.math.exp2`, running max and row sum), and stores the `float16` output tile `o` and the
`float32` base-2 log-sum-exp `M`. The inputs are read through host `TensorDescriptor`s
(`warp_specialize = False`). The claim is `Claims.AttnFwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- every element `(bh, n, e)` of `o` holds a `float16` value `c`; when head `bh`'s values are
  covered (`HeadOk`), `c` is finite and, for every bound `X ≤ 2^10` on the row's scaled score
  magnitudes, `|c − attnRef| ≤ epsO·attnMag + 2^-24`;
- every element `(bh, n)` of `M` holds a `float32` value `c`; for a covered head, `c` is finite and
  `|c − lseRef| ≤ epsM`;
- every byte outside `o` and `M` is unchanged.

`attnRef` is the exact attention `Σ_j w_j·v_j / Σ_j w_j` with `w_j = exp(sm_scale·q·k_j)` over the
row's keys (`j ≤ n` when causal), `lseRef = log₂ Σ_j w_j`, `attnMag` the larger of the weighted and
the plain mean of `|v|`; `epsO = 2^-20·(D·X + N + 2^10)` and `epsM = 2^-20·(D·X + 2^10)`, with
`D = HEAD_DIM`, `N = N_CTX`. `HeadOk`: every input of the head is a finite `float16`, every
`|v| ≤ 2^15`, and every `sm_scale·scoreMag ≤ 2^10` (`scoreMag = Σ_e liftMag q·liftMag k`).

## Conditions

`Inst.ok` and `Pre`:
- `HEAD_DIM ∈ {64, 128, 256}`;
- `N_CTX` is a positive multiple of `128`, at most `2^16`;
- `0 < Z ≤ 8192`, `0 < H`, `Z·H < 2^16`, `Z·H·N_CTX < 2^31`;
- `sm_scale` is a finite double in `[2^-64, 2^64]`;
- each tensor holds its elements; `q`, `k`, `v`, `o` are 16-byte aligned; `o` and `M` share no byte
  with each other or with `q`, `k`, `v`;
- the five buffers are valid in the initial memory. Any input values (`Post` says what holds for
  covered heads).

**The launches covered** (`SETUP.md`): every test launch (`setup_test_ok`) and every `float16`
benchmark launch with `warp_specialize = False` (`setup_bench_ok`), through `canon_ok`, which covers
the host's contiguous layout for every shape `Inst.ok` allows.

**Device facts** the tolerances rest on (in `Device P`): `Device.DotF16` (a `float16` `tl.dot` into
`float32`, relative to the lifted magnitudes), `Device.Exp2F32` and `Device.Exp2F32NegInf`
(`tl.math.exp2` on `float32`, and `exp2(−∞) = +0`), `Device.Log2F32` (`tl.math.log2`),
`Device.DivFull` (`/` on `float32`), and the totality of `tl.dot`, `exp2`, `log2` and `/`.

## Tolerances

The fine bounds compose the device constants (`relDot16 D = (D + 2)·2^-22` for the score's
`tl.dot`, `relDot16 64` for `p·v`, `2^-22` for `exp2`, `/`, `log2`) with `float32` and `float16`
roundings, step by step: the exponents (within `(relDot16 D + 2^-20)·(3X/2) + 2^-50`), the weights
`p = exp2(·)` and their rounding to `float16`, the row sums `l_i` (compounding over the `N/64`
blocks), the accumulator (`tl.dot(p, v, acc)` per block), and the end (`acc / l_i`, the change from
base 2 to `exp`, the rounding to `float16`; `M = m_i + log₂ l_i`). The result is
`|o − attnRef| ≤ tolO·attnAbs + tolS·vAbsSum + tolF` and `|M − lseRef| ≤ tolM + 2^-24·|lseRef|`.

The relaxation: each fine bound is convex in `X` and `epsO`, `epsM` are affine in `X`, so
`fine ≤ stated` on `[0, 2^10]` follows from the two ends; evaluated exactly for every `D` and every
`N` a multiple of `128` up to `2^16`, the ratios are at most `0.927` (`o`, relative part), `0.709`
(`o`, the `2^-24` floor) and `0.889` (`M`).

## Exclusions

- **`N_CTX` not a multiple of `128`**: a limit of the kernel; the last query tile of a head runs
  into the next head's rows (a race on `o`, and wrong keys).
- **`Z·H·N_CTX ≥ 2^31`**: a limit of the kernel; the `int32` row offsets wrap.
- **`N_CTX > 2^16`**: the kernel is believed correct, but the bound grows with `N` and the file's
  largest is `2^14`.
- **`Z > 8192`**: the claim covers real batch sizes only.
- **`Z·H ≥ 2^16`** (beyond the grid's `y` limit, no launch), **`HEAD_DIM ∈ {16, 32}`** (rejected
  on the host by line 188's `tl.static_assert`).
- **Inputs outside `HeadOk`**: only that `o` and `M` hold values is claimed.
- **`sm_scale` outside `[2^-64, 2^64]`**: outside real use.
- **`warp_specialize = True`**: the kernel builds its own tensor descriptors; not covered (its
  `False` twin is).
- **The `float8e5` run, other configs, other devices**: not part of this claim.

`check_gpu.py` (one H200) checks `Post` on the file's launches and adversarial inputs.
