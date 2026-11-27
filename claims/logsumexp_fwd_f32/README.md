# `logsumexp_fwd_f32`

**The kernel.** `logsumexp_fwd_kernel` of fla (`source/ops.py`, kernel lines 67–91, with the
`@triton.jit` helpers `exp` and `log` of lines 55–60; host `logsumexp_fwd` lines 94–108). The host
views a contiguous `float32` `x` as `[N, D]`, sets `BD = min(next_power_of_2(D), 65536)` and
`ND = cdiv(D, BD)`, and launches the grid `(N, ND)` with `HAS_SCALE = scale is not None` and
`HAS_SOFTCAPPING = softcapping is not None`. Program `(n, i)` loads the columns
`[i·BD, min(D, (i+1)·BD))` of row `n` (`int64` offsets, masked lanes `-inf`), multiplies them by
`scale` if there is one, takes the block's maximum `m`, and stores
`z[n, i] = log(Σ_j exp(y_j − m)) + m` (`float32`). The host then combines the `ND` values of a row
with `torch.logsumexp`, outside the kernel. The claim is `Claims.LogsumexpFwd.claim` in
`Claim.lean`; it covers `HAS_SOFTCAPPING = False` and `HAS_SCALE` both ways.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). For each row
`n < N` and block `i < ND`, with `xs j` the row's initial `float32` inputs, `σ` the host's double
`scale` (or `1` without a scale) and `y_j = xs j · σ`:
- `z[n, i]` holds a `float32` value (for every input, NaN and infinities included);
- when the block is `BlockOk` (every `xs j` of the block finite, and `|y_j| ≤ 2^126` without a
  scale, `|y_j| ≤ 2^20` with one), `z[n, i]` is finite and, for every bound `M` of the block's
  `|y_j|`, `|z[n, i] − log Σ_{j ∈ block} e^(y_j)| ≤ 2^-20·M + 2^-16`, the log-sum-exp exact in
  the reals;
- every byte outside the first `N·ND` elements of `z` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ D < 2^31`; `BD` a power of two at most `2^16` (the host's `min(next_power_of_2(D),
  65536)`, and every other power of two for a direct launch);
- `ND < 2^16` (CUDA's limit on the grid's axis 1) and fewer than `2^31` programs (`N·ND`); `N = 0`
  is covered;
- `x` holds at least `N·D` elements and `z` at least `N·ND`, both valid in the initial memory; the
  written elements of `z` share no byte with the read elements of `x` (the host allocates `z`
  fresh);
- with a scale: a finite double of magnitude at most `2^126` (`0` and scales whose `float32` is
  subnormal or `0` included).

The kernel limits among these: `ND < 2^16` and `N·ND < 2^31` (the launch fails or is skipped
beyond), and `D ≥ 1`. The others (`D < 2^31`, `BD ≤ 2^16`, `|scale| ≤ 2^126`, the buffers'
sizes and disjointness) are what the host does or what the proof needs.

**The launches covered** (`SETUP.md`): fla's test shape `2 × 1025`, the host test's nine shapes,
and logits chunks over vocabularies of `32000`, `128256`, `151936` and `262144` columns, with and
without a scale (`setup_*_ok`). `mk_ok` covers the host's launch for every `N`, `D` and
power-of-two `BD ≤ 2^16` within the grid's limits, with `x` below `2^40` bytes, and every scale in
range.

**Device facts** the tolerances rest on (all in `Device P`): `Device.ExpF32` (`tl.exp` on
`float32`: relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute `2^-140`), `Device.ExpF32NegInf`
(`exp(−∞) = +0`), `Device.LogF32` (`tl.log` on `float32`: relative `2^-22`, absolute `2^-149`, on
positive inputs), `Device.SumDepth` (a `tl.sum` of `B` lanes is a tree of depth at most
`⌈log₂ B⌉`), and the totality of `exp` and `log`.

## Tolerances

With `u = 2^-24`, `η = 2^-150` (one `float32` rounding), `n ≤ 2^16` the block's valid lanes and `M`
a bound of its `|y_j|`:
- The lanes: without a scale `v_j = xs j` exactly; with one, the double `scale` rounded to
  `float32` and the product `xs j · scale32` give `|v_j − y_j| ≤ 3u·|y_j| + 5u`.
- The subtraction `v_j − m`: IEEE without a scale (error `u·|v_j − m| + η`); with a scale the
  product feeds it, so it is fusable (`AddRel`, error `3u·(|v_j| + |m|) + 4η`). Every exponent
  `s_j` is then at most `κ` above `0` and the maximum's lane at least `−κ`, `κ = η` without a scale
  and `κ ≤ 1/2` with one (from `|y_j| ≤ 2^20`). A masked lane's exponent is `−∞`, or (fused) a
  finite value below `−2^126`; its `tl.exp` is at most `2^-100`.
- The block: `tl.exp` on `[−X, X]` (relative `4u + 2u·|s|`, and `Σ |s|·e^s ≤ 16·Σ e^s` plus a
  tiny term), the sum along a tree of depth at most `16` (relative `16.001u`), the masked lanes,
  and `tl.log` (relative `2^-22` of a value at most `12`) give `L` within `100u` of
  `log Σ_j e^(s_j)`, with `|L| ≤ 12`. Since log-sum-exp is 1-Lipschitz in the exponents, the
  lanes' errors move it by at most their largest error.
- The last addition `L + m`: IEEE without a scale (`u·|L + m| + η`), fusable with one
  (`3u·(|L| + |m|) + 4η`).

Totals: `112u + 3u·M` without a scale and below `141u + 12u·M` with one, both at most
`16u·M + 256u = 2^-20·M + 2^-16`.

## Exclusions

- **Softcapping** (`HAS_SOFTCAPPING = True`, `softcapping` not `None`): `tanh` (libdevice) has no
  device fact, so these launches are not covered. fla's own test of this host uses
  `softcapping = 0.5`; the claim covers its shape only without softcapping.
- **Blocks that are not `BlockOk`**: a NaN or `+inf` input gives `z = NaN`, and so does a row
  whose block is all `−inf` (without a scale `−inf − (−∞)` is NaN). The claim only says `z[n, i]`
  holds a `float32` value. The mask never empties a block: `ND = cdiv(D, BD)`, so each block has
  at least one column below `D`.
- **`|y_j| > 2^126` without a scale, `|y_j| > 2^20` with one, and `|scale| > 2^126`**: the kernel
  is accurate there on the GPU (`x = 2^127` gives `2^127`, `|y| ≈ 2^26` with a scale gives values
  near `1.7·10^8` correctly), but the subtraction needs room below `2^127`, and with a scale the
  fusable subtraction's error `3u·(|v| + |m|)` must stay below `1/2` for the exponent's bound.
  Real logits are far below `2^20`.
- **A non-contiguous `x`**: the host's `x.view(-1, D)` keeps a row stride other than `D` when one
  is viewable, and the kernel reads rows with stride `D`. fla passes contiguous logits.
- **`x` of another dtype** (`bfloat16` logits): outside the launches; the claim's `x` and `z` are
  `float32`.
- **`D ≥ 2^31`, `ND ≥ 2^16`, `N·ND ≥ 2^31`**: the constexpr `D` is not an `int32`-range constant,
  CUDA rejects the grid, or Triton's launcher skips it.

`check_gpu.py` (one H200) checks `Post` on the `SETUP.md` launches, on direct launches at the
edges of `Inst.ok` and of `BlockOk`, and rejects plausible wrong outputs; it also prints what the
kernel stores on blocks that are not `BlockOk`.
