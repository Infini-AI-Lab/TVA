# `rotary_varlen_bf16`

**The kernel.** `rotary_embedding_kernel` of fla's rotary embedding (`source/rotary.py`, kernel
lines 63–154; host `rotary_embedding_fwd` lines 200–272, which launches the JIT function directly).
It rotates the first `2 R` channels of every head of `bfloat16` queries or keys by the `cos`/`sin`
tables at each token's position, over a padded batch (`cu_seqlens = None`) or packed
variable-length sequences (`cu_seqlens` and `chunk_indices`, `int32` or `int64`), interleaved
(GPT-J) or not (GPT-NeoX), with one integer offset or a tensor of one offset per sequence (`int32`
or `int64`, negative under left padding), `conjugate = False`. A grid `(NT, B, H)`; each program
handles `BT` tokens of one head of one sequence, with no loop and no atomics. The claim is
`Claims.RotaryVarlen.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). For every
sequence `n`, token `l` of it, head `h` and rotated channel `c < 2 R`, with flat token `t`
(`n·T + l` padded, `cu_seqlens[n] + l` packed) and position `p = l + offset[n]`:
- if `p ≥ 0`: `y[t, h, c]` is a finite `bfloat16` `v` with
  `|v − ref| ≤ 2^-8·|ref| + 2^-22·(|a·cs| + |b·sn|) + 2^-132`, where `a = x[t, h, c]`, `b` is the
  partner channel's value (`c ± R`, or `c ± 1` by parity when interleaved), `cs`, `sn` the table
  entries of the pair at row `p`, and `ref = a·cs − b·sn` for the first channel of a pair,
  `a·cs + b·sn` for the second, exactly, in the rationals, on the initial values;
- if `p < 0` (left padding under a tensor of offsets): the element's bytes are unchanged;
- every byte outside the rotated channels of `y` is unchanged (no input is written; `y`'s channels
  from `2 R` on keep what the host put there).

**The conditions** (`Inst.ok`, `Pre`):
- the host's own requirements: `D ≤ 256`, `2 R ≤ D`, `T ≤ TR` padded, `off + T ≤ TR` for an `int`
  offset, a tensor of offsets `int32` or `int64` with one entry per sequence and `len + v ≤ TR`;
  `R ≥ 1`, index dtype `int32` or `int64`, packed batches `[1, T, H, D]`;
- `BT`, `BD` powers of two, `2 R ≤ BD`, a tile of at most `2^20` elements;
- the buffers hold their tensors and are valid; `y` shares no byte with an input;
- packed: `cu_seqlens` non-decreasing from `0` to `T`, and `chunk_indices` lists each (sequence,
  chunk of `BT` tokens) pair exactly once (any order);
- integer bounds: `T`, `off < 2^31`; `TR·R`, `B·T·H·D < 2^62`; a tensor offset `≥ −2^31`;
- real-use bounds: `T ≤ 2^22`, every offset `≤ 2^22`, `B ≤ 8192`, packed `N ≤ 8192`;
- kernel limits: `B`, `H < 2^16`, `NT·B·H < 2^31` programs, packed `N < 2^31`;
- inputs: the rotated channels of `x` finite with `|x| ≤ 2^125`; the table rows read finite with
  `|·| ≤ 1`. Anything the kernel does not read is unconstrained.

**The launches covered** (`SETUP.md`): 35 instances, each with `setup_*_ok`: the host's 22 test
cases (both layouts; padded and packed, `int32` and `int64` `cu_seqlens`), decoding with offset
4095, head dim 64 with 32 heads, one key head, partial rotary 64 of `D = 256`, one packed sequence
with `int64` `cu_seqlens`, and tensors of offsets (`int32` left-padded prefill and decoding,
`int64` padded and packed). `claim_satisfiable` uses the one-token instance with a zero memory.

**Device facts.** None: the kernel uses only `tl.mul`, `tl.addc`, `tl.subc`, `tl.cvt`,
`tl.convert` and integer arithmetic, all exact up to NaN payload in TVA.

## Tolerances

With `S = |a·cs| + |b·sn|` (`u = 2^-24`, `η = 2^-150` for `float32`; `2^-8`, `2^-134` for
`bfloat16`):
1. the products of the exactly widened inputs are `float32` roundings:
   `|p − a·cs| ≤ 2^-24·|a·cs| + 2^-150`;
2. the addition, IEEE or with one product contracted into an `fma` (each operand then within one
   more half-ulp), gives `|r − ref| ≤ (2^-23 + 2^-24)·S·(1 + 2^-22) + 2^-146 =: e32`;
3. the `bfloat16` store: `|v − r| ≤ 2^-8·|r| + 2^-134`.

So `|v − ref| ≤ 2^-8·|ref| + 1.51·2^-23·S + 2^-133`, inside the tolerance. No overflow: `|a| ≤ 2^125`
and `|cs|, |sn| ≤ 1` give `S ≤ 2^126`, so every step stays finite in both formats. The interleaved
path computes the same products and one fusable addition per channel. The `bfloat16` rounding can
reach `0.996` of the `2^-8·|ref|` term, a structural margin.

## Exclusions

- **`B ≥ 65536` padded, `H ≥ 65536`**: kernel limits (grid axes 1 and 2; CUDA allows `65535`).
- **`NT·B·H ≥ 2^31` programs**, **`N ≥ 2^31` sequences** (line 88 truncates the sequence index to
  `int32`): kernel limits, needing tensors of hundreds of GB.
- **`T`, `off ≥ 2^31`, `TR·R`, `B·T·H·D ≥ 2^62`, a tensor offset below `−2^31`**: outside real use.
- **Contexts longer than `2^22` tokens (`T`, every offset) and batches of more than `8192`
  requests (`B`, packed `N`)**: the claim covers real model shapes only.
- **Inputs**: non-finite or `|x| > 2^125` in a rotated channel of `x`, or a table entry read that
  is non-finite or above `1` in magnitude: outside real use (the bounds are a sufficient
  no-overflow condition).
- Not `SETUP.md`'s launches: `conjugate = True`, the in-place path, a negative `int` offset.

`check_gpu.py` (one H200) checks `Post` on the setup's launches and on edges of `Inst.ok`; the worst
`|y − ref| / tol` is `0.996`.
