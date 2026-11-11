# `recompute_wu_bf16`

**The kernel.** `recompute_w_u_fwd_kernel` of fla's `wy_fast.py` (`source/wy_fast.py`, kernel lines
69–133 with the helper `exp2`, lines 65–66; host `recompute_w_u_fwd`, lines 175–220). It computes
the WY representation of the gated DeltaNet's chunks: on a grid `(NT, B·HV)`, program `(i_t, i_bh)`
takes one 64-token chunk of one sequence and one value head, loads the chunk's rows of `A`
(`[64, 64]`) and `beta`, and, in two loops over 64-column tiles, computes `u = A · (beta ⊙ v)` and
`w = A · (beta · 2^g ⊙ k)` with `bfloat16` `tl.dot`s into `float32`, rounded to `bfloat16`. The
batch is padded (`cu_seqlens = None`) or packed (`cu_seqlens`, `chunk_indices`, `int32` or
`int64`); `g` is given (`USE_G = True`); `beta` is `float32` or `bfloat16`, and the model is the
kernel compiled for that dtype (`recompute_w_u_fwd_kernel_b32`, `_b16`). The claim is
`Claims.RecomputeWU.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
token `t` of every sequence (padded: `b·T … b·T + T − 1`; packed: `cu_seqlens[n] … cu_seqlens[n+1] − 1`,
read from the initial memory), every value head `h`, the sums running over the columns `j < 64` of
the token's row of `A` whose token `f_j = s0 + t // 64 · 64 + j` is inside the sequence:
- `u[s0 + t, h, c]` (`c < V`) is a finite `bfloat16` within
  `tolU = 2^-8·|uRef| + 2^-15·uMag + 2^-119` of `uRef = Σ_j A_j · vb_j`, where `vb_j` is the kernel's
  own operand of line 115 (`bf16(fl32(v · beta))` for a `float32` `beta`, the IEEE `bfloat16` product
  for a `bfloat16` one) and `uMag = Σ_j liftMag(A_j) · liftMag(vb_j)`;
- `w[s0 + t, h, c]` (`c < K`) is a finite `bfloat16` within
  `tolW = 2^-6·wMag + 2^-124·wFloor + 2^-119` of the exact `wRef = Σ_j A_j · x_j · e_j`, with
  `x_j = k[f_j, h // (HV // H), c] · beta[f_j, h]`, `e_j = 2^(g[f_j, h])`,
  `wMag = Σ_j liftMag(A_j) · |x_j| · e_j` and `wFloor = Σ_j liftMag(A_j) · (|x_j| + 1)`;
- every byte outside the elements of `u` and `w` is unchanged.

`liftMag` is `Device.liftMag .bf16`: `|q|`, a nonzero value counted at least `2^-126`.

**The conditions** (`Inst.ok`, `Pre`):
- the host's own requirements: `H, HV, K, V ≥ 1`, `HV` a multiple of `H`; the index dtype `int32`
  or `int64`; a packed batch is `[1, T, ·, ·]`; the buffers hold their tensors; `w` and `u` fresh
  (disjoint from every input and from each other);
- `T < 2^31`, `K + 64 ≤ 2^31`, `V + 64 ≤ 2^31` (the kernel's `int32` values), and
  `(B·T + 64)·HV·(K + V + 64) < 2^62` (every element offset far inside `int64`);
- kernel limits: `B·HV < 2^16` (grid axis 1), grid axis 0 below `2^31`, fewer than `2^31` programs,
  `N < 2^31`;
- real shapes: `T ≤ 2^22`, `B ≤ 8192` and, packed, `N ≤ 8192`;
- every input element present and in range: `k`, `v` finite with `|·| ≤ 2^32`; `A` finite with
  `|·| ≤ 2^64` (all 64 columns of every row); `beta` finite with `|·| ≤ 2^8`; `g` finite with `g ≤ 8`;
- packed: `cu_seqlens` non-decreasing from `0` to `T`, and `chunk_indices` listing each
  (sequence, chunk of 64 tokens) pair exactly once, in any order (`SeqOk`), as
  `prepare_chunk_indices` builds it.

**The launches covered** (`SETUP.md`): 33 launches (`setup_*_ok`): fla's tests (`test_chunk`,
`test_chunk_varlen`, the kernel's own tests), Qwen3-Next's and fla's layer shapes, padded and packed,
`float32` and `bfloat16` `beta`, and edges (one token; empty sequences; 132 one-token sequences).
`SETUP.md`'s edge `B·HV = 65535` (`B = 13107`) is not covered (`b_batch`).

**Device facts** the tolerances use (all in `Device P`): `Device.DotBF16` (a `bfloat16` `tl.dot`
into `float32`, at either input precision, is within `relDot16 K·L + absDot16 K` of the exact sum,
`L` the lifted magnitude), `Device.Exp2F32` (`tl.math.exp2` on `float32`: within
`2^-22·2^x + 2^-126`), and the totality facts for `tl.dot` and `exp2`. The products and the
conversions are IEEE operations.

## Tolerances

With `u = 2^-24`, `η = 2^-150` (`float32`), `u_b = 2^-8`, `η_b = 2^-134` (`bfloat16`),
`relDot16 64 = 66·2^-22`, `absDot16 64 = 66·2^-126`:
- **`u`.** The dot (line 116) of the row's `A_j` and the column of `vb_j` (`+0` outside the
  sequence) is within `E = relDot16 64·uMag + absDot16 64` of `uRef`; the store rounds once:
  `|o − uRef| ≤ 2^-8·|uRef| + (1 + 2^-8)·E + 2^-134 ≤ tolU`.
- **`w`, `float32` `beta`.** `p1 = fl32(k·beta)`, `e = exp2(g)` (room since `g ≤ 8`),
  `p2 = fl32(p1·e)`, `kb = bf16(p2)`: `|kb − x·2^g| ≤ (2^-8 + 2^-20)·y + 2^-125·(|x| + 1)` with
  `y = |x|·2^g`. With the dot (line 132) and the store, `|o − wRef| ≤ 2^-8·|wRef| + c·wMag +
  2^-124·wFloor + 2^-119`, `c = 2^-8 + 2^-14`.
- **`w`, `bfloat16` `beta`.** Line 129 rounds `k·beta` to `bfloat16` once more: the same bound with
  `c = 2^-7 + 2^-13`.
- **Relaxed to `tolW`.** `|wRef| ≤ wMag`, so `2^-8·|wRef| + c·wMag ≤ (2^-8 + 2^-7 + 2^-13)·wMag ≤
  2^-6·wMag` for either `c`.

The bounds rest on `DotBF16`, `Exp2F32` and IEEE rounding; the dots' room and the outputs'
finiteness come from `Pre`'s magnitudes. On `check_gpu.py`'s launches the device reaches `0.988` of
`tolU` and `0.488` (`0.719` with a `bfloat16` `beta`) of `tolW`; both reject a 2% error in one
element.

## Exclusions

- **`B·HV ≥ 65536`** (a kernel limit): CUDA allows at most `65535` on grid axis 1, so such a padded
  launch fails with no outcome (`grid_bh`).
- **`2^31` sequences or more** (a kernel limit): line 94 truncates the sequence index to `int32`
  (`n_lt`).
- **Input values** outside `Pre`'s ranges, non-finite values: outside real use (`k` L2-normalised,
  `beta` in `(0, 2)`, `g ≤ 0`, `A` of order 1); the ranges keep the products, dots and outputs inside
  their formats.
- **Real shapes only**: contexts longer than `2^22` tokens, batches of more than `8192` sequences;
  `T ≥ 2^31`, `K` or `V` above `2^31 − 64`, tensors of `2^62` elements.
- **Not among the launches**: `g = None`, `float16` `beta`, `float16`/`float32` `k`, `v`, `A`, chunk
  sizes 16 and 32; overlapping outputs, non-contiguous tensors.

`check_gpu.py` (one H200) checks `Post` on the setup's launches, a direct launch with guard regions,
and negative tests.
