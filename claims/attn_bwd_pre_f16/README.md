# `attn_bwd_pre_f16`

**The kernel.** `_attn_bwd_preprocess` of Triton 3.8.0's tutorial 06, "Fused Attention"
(`source/06-fused-attention.py`, kernel lines 248–262; host `_attention.backward` lines 582–623,
which launches it at lines 600–607). A grid `(N_CTX // BLOCK_M, Z·H)`; program `(i, bh)` loads the
`BLOCK_M × HEAD_DIM` tile of rows `i·BLOCK_M …` of head `bh` of `o` and `do` (`float16`), widens
both to `float32`, multiplies them and sums each row (`tl.sum(o * do, axis=1)`, whose additions may
be fused with the products), and stores the `BLOCK_M` sums in `delta` (`float32`). The claim is
`Claims.AttnBwdPre.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
row `r = bh·N_CTX + m` (`bh < Z·H`, `m < N_CTX`):
- element `r` of `delta` holds a `float32` value `c`;
- when row `r` of `o` and of `do` holds finite values `o_k`, `do_k` (`k < HEAD_DIM`), `c` is finite
  and `|c − Σ_k o_k·do_k| ≤ tol B = 2^-19·B + 2^-138`, `B = Σ_k |o_k·do_k|`, the sums exact;
- every byte outside the elements `r < Z·H·N_CTX` of `delta` is unchanged.

The bound is relative to `Σ|o·do|`, not to `|Σ o·do|`; a row with a NaN or `∞` input only holds a
value.

**The conditions** (`Inst.ok`, `Pre`):
- `BLOCK_M` and `HEAD_DIM` are powers of two, `HEAD_DIM ≤ 256`, `BLOCK_M·HEAD_DIM ≤ 2^20`;
- `BLOCK_M ∣ N_CTX` (the host asserts it), `N_CTX < 2^31`, `N_CTX ≤ 2^22`, `Z ≤ 8192`;
- the grid within CUDA's limits (`Z·H < 2^16`, fewer than `2^31` programs);
- the kernel's `int32` products do not wrap: `(N_CTX − 1)·HEAD_DIM < 2^31` and
  `(Z·H − 1)·HEAD_DIM·N_CTX < 2^31`;
- each buffer holds its tensor and is valid in the initial memory; the written elements of `delta`
  share no byte with the read elements of `o` or `do`.

Every `float16` input is covered, NaN and infinities included.

**The launches covered** (`SETUP.md`): all 24 shapes of the test and all 10 of the benchmark
(`setup_test_ok`, `setup_bench_ok`), and `HEAD_DIM` 16, 32 and 256 at the two largest shapes
(`setup_headdims_ok`). `canon_ok` covers the host's launch (`BLOCK_M = 128`, contiguous `o`, `do`,
fresh `delta`) at every shape meeting the numeric conditions.

**Device facts** the tolerance rests on: `Device.SumDepth` (the tree of the float `tl.sum` over
`HEAD_DIM` positions has depth at most `⌈log₂ HEAD_DIM⌉`). The widenings are exact, the products
IEEE, the sum's additions IEEE or fused.

## Tolerances

With `u = 2^-24`, `η = 2^-150`, `D = HEAD_DIM`: each product is one `float32` rounding
(`u·|a·b| + η`), and the fused sum along a tree of depth `⌈log₂ D⌉` with `D − 1` additions (`3u`,
`4η` each) is within `errBound(3u, 4η, ⌈log₂ D⌉, D − 1, (1 + u)·B + D·η)` of the products' sum,
`Tree.errBound δ η d N A = ((1 + δ)^d − 1)·A + η·N·(1 + δ)^d`. The fine bound
`F(D, B) = errBound(…) + u·B + D·η` grows with `D`; at `D = 256` it is at most
`25.0001·u·B + 1277·η ≤ 2^-19·B + 2^-138 = tol B` (`fine_le`). `tol/B = 2^-19 ≈ 1.9·10^-6`, at
most `2.5×` the fine bound (at `D = 16`). A result rounded to `float16`, a missing or doubled
column, or the wrong row of `do` fails `Post`.

## Exclusions

- **`int32` offsets that wrap** (`(Z·H − 1)·HEAD_DIM·N_CTX ≥ 2^31`): a kernel limit; the kernel then
  reads the wrong rows. The file's launches are far below.
- **`N_CTX` not a multiple of `BLOCK_M`**: the host asserts it away.
- **`HEAD_DIM > 256`, other dtypes**: the forward rejects them.
- **The grid beyond CUDA's limits**: no launch.
- **`N_CTX > 2^22`, `Z > 8192`**: the claim covers real model shapes only.
- **`delta` overlapping `o` or `do`**: the host allocates a fresh `delta`.
- **Non-finite inputs**: covered with a weaker `Post` (the output only holds a value).

`check_gpu.py` (one H200) checks `Post` on the script's launches.
