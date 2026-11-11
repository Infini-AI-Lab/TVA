# `ssd_bmm_chunk_bf16`

**The kernel.** `_bmm_chunk_fwd_kernel` of SGLang's Mamba-2 SSD (`source/ssd_bmm.py`, kernel
lines 80–190; host `bmm_chunk_fwd` lines 193–282), launched as the chunked scan does:
`CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)`, with
`bfloat16` `C`, `B` (`[1, seqlen, ngroups, dstate]`), `int32` `seq_idx`, `IS_CAUSAL = False`,
`HAS_SEQ_IDX = True`, blocks `64 × 128 × 32` (`SETUP.md`). Each program computes a tile of one
chunk's `C·Bᵀ` for one group by a loop of `bfloat16` `tl.dot`s over `dstate` into a `float32`
accumulator, zeroes the pairs of positions whose `seq_idx` differ (different requests, or past the
end of the sequence), and stores the tile of the `float32` `out`. The claim is
`Claims.SsdBmm.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
element `(0, c, g, m, n)` of `out` (`nchunks × ngroups × chunk_size × chunk_size`):
- it holds a `float32` value `v`;
- if positions `tm = c·C + m` and `tn = c·C + n` are tokens of the sequence with the same `seq_idx`
  (`sameReq`), then whenever row `tm` of `a` and row `tn` of `b` (group `g`) are finite and
  `dotMag ≤ 2^120`, `v` is finite and `|v − S| ≤ tol = T·2^-16·dotMag + 2^-112`, with `S` the exact
  dot product of the two rows, `dotMag = Σ_k liftMag(a)·liftMag(b)` and `T = cdiv(dstate, 32)`;
- otherwise `v` is exactly `+0.0`, for every input;
- every byte outside `out`'s elements is unchanged.

## Conditions

`Inst.ok` and `Pre`:
- `seqlen`, `ngroups`, `dstate` positive; `chunk_size` a power of two, at least `2`;
- `dstate ≤ 2^10`; `seqlen ≤ 2^22`;
- the grid's `z = nchunks·ngroups < 2^16`, and fewer than `2^31` programs;
- `seqlen`, the strides other than the batch strides, and `out.stride(1)` are `int32`; the batch
  strides are below `2^63` (an `int32` or an `int64` argument by value);
- the `int32` lane offsets of lines 143–144 and the steps of lines 162–163 do not wrap (`a_fit`,
  `b_fit`, `aK_step`, `bK_step`);
- each buffer holds its tensor; `out` shares no byte with `a`, `b` or `seq_idx` (`a` and `b` may
  overlap: both are views of one tensor in the mixer);
- the four buffers are valid in the initial memory, and every `seq_idx` entry is a non-negative
  `int32`. Any `bfloat16` values in `a` and `b`, NaN and infinities included.

**The launches covered** (`SETUP.md`): SGLang's tests (single example, intermediate states,
continuous batching, prefill chunking) and the mixer's real shapes (8 groups or one group per rank,
`dstate` 128 or 256, chunks of 128 or 256, prefills up to 2,096,896 tokens, where the batch strides
become `int64` arguments), each a `setup_*_ok` through `mk_ok`. Every request pattern (`seq_idx`)
is covered: it is memory, not instance.

**Device facts** the tolerance rests on (in `Device P`): `Device.DotBF16` (a `bfloat16` `tl.dot`
into `float32`: each finite output within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with
`L = |acc| + Σ liftMag a·liftMag b`, a subnormal operand counted at `2^-126`) and the totality of
`tl.dot`.

## Tolerances

Step `t` of the loop (line 161) is `acc += tl.dot(a, b)`: a dot of `32` products followed by a
`float32` addition (or the fused form). With `DotBF16` at `K = 32` (`relDot16 32 = 34·2^-22`,
`absDot16 32 = 34·2^-126`) and the addition's rounding, the step is within `ρ·(|acc| + L_t) + α`
of `acc + c_t`, with `ρ = relDot16 32 + 3·2^-24·(1 + relDot16 32) = 0.543·2^-16` and
`α = absDot16 32·(1 + 3·2^-24) + 4·2^-150`. The masked `k`-lanes past `dstate` hold `+0`. So the
error follows `E_{t+1} = E_t + ρ·(|S_t| + E_t + L_t) + α` from `E_0 = 0`, and since
`|S_t| + L_t ≤ dotMag`, `E_T ≤ ((1 + ρ)^T − 1)·dotMag + T·(1 + ρ)^T·α`. With `T ≤ 32`,
`(1 + ρ)^T − 1 ≤ 0.544·T·2^-16` and `T·(1 + ρ)^T·α ≤ 2^-112`, so `E_T ≤ tol`. `dotMag ≤ 2^120`
keeps every step inside the `float32` range. The other elements: the two loaded `seq_idx` values
differ (a real entry `≥ 0` against the masked loads' padding `-1` or `-2`, or two different
entries), and `tl.where` gives `+0.0`.

A result rounded to `bfloat16` or a missing `tl.dot` step breaks `Post`.

## Exclusions

- **`nchunks·ngroups ≥ 2^16`**, **`2^31` or more programs**: limits of the kernel's launch (CUDA's
  grid limits); real use does not reach them.
- **Strides other than the batch strides from `2^31` on**: outside real use (Triton would pass an
  `int64`).
- **`int32` wrap of the lane offsets**: a limit of the kernel; real use (`stride_k = 1`,
  `stride_seqlen ≈ 10^4`) does not reach it.
- **`seqlen > 2^22`**: the claim covers real context lengths only.
- **`dstate > 2^10`**: outside real use (Mamba-2 has `dstate ≤ 256`); the tolerance is stated for
  `dstate ≤ 2^10`.
- **`chunk_size = 1`**: outside real use (8 to 256).
- **A negative `seq_idx` entry**: outside real use; `-1` or `-2` could equal the masked loads'
  padding.
- **Values** (a hypothesis of the numeric clause only): rows with a NaN or an infinity, and
  `dotMag > 2^120`.
- **`causal=True`, `seq_idx=None`, 3-D `a`/`b`, `batch > 1`, other dtypes**: not part of the
  setup.

`check_gpu.py` (one H200) checks `Post` on the setup's launches, edge cases and wrong kernels.
