# `block_scaled_matmul_mxfp8`

**The kernel.** `block_scaled_matmul_kernel` of Triton 3.8.0's tutorial 10, "Block Scaled Matrix
Multiplication" (`source/10-block-scaled-matmul.py`, kernel lines 174–240; host
`block_scaled_matmul` lines 243–281 with the config `initialize_block_scaled`, lines 330–449),
for `block_scale_type = "mxfp8"` (`e4m3` by `e4m3`) and `"mixed"` (`e4m3` by `e2m1`, packed two per
byte along `K`), `e8m0` scales in groups of `32` in the 5-D preshuffled layout, and a `float16`
output. A 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs; each loads its tiles through tensor
descriptors, runs `K / 128` steps of `tl.dot_scaled` into a `float32` accumulator, converts it to
`float16` and stores its `128 × 256` tile of `c`. The claim is `Claims.BlockScaledMxfp8.claim` in
`Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
output element `(i, j)`, with `S = Σ_k scaled(a[i,k])·scaled(b[j,k])` the exact scaled dot product
(`ref`, each element times `2^(s − 127)` of its scale byte `s`) and `μ = stepMag` the magnitude the
loop's steps carry (`Σ|p| + Σ_t |partSum t|`):
- `c[i, j]` is the `float16` rounding (up to NaN payload) of a finite `float32` value within
  `vTol = 2^-14·μ + 2^-108` of `S`;
- when there is room in `float16` (`|S| + tolOut ≤ 65504`), `c[i, j]` is finite and within
  `cTol = 2^-11·|S| + 2^-14·μ + 2^-24` of `S`;
- every byte outside `c` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `M`, `N`, `K` are positive multiples of `128` (the host's scale layout), `M ≤ 2^22` (a token
  count), `N ≤ 2^17`, `K ≤ 2^17` (model dimensions; see Exclusions), and the grid has fewer than
  `2^31` programs;
- every tensor starts at a `16`-byte-aligned address and holds the host's sizes, and is valid in
  the initial memory; `c` shares no byte with an input;
- every element of `a` and `b` is finite, with its scale byte in `1 … 254` and its scaled value `0`
  or a normal `bfloat16` magnitude (`InDom`), and every output has room in `float32`:
  `Σ_k |p k| + tolOut ≤ 2^127` (`ValsOk`).

**The launches covered** (`SETUP.md`, `setup_*_ok`): for `mxfp8`, the test (`8192³`) and the
benchmark at `K = 512, 8192, 10240, 12288, 14336, 16384` and `2^17`; for `mixed`, the test and the
benchmark at `K = 512` and `2^17`. `hostInst_ok` covers every size meeting `Inst.ok`'s bounds in
both formats.

**Device facts** the tolerances rest on (all in `Device P`): `Device.DotScaledE4m3` and
`Device.DotScaledE4m3E2m1` (one `tl.dot_scaled` step of `K = 128` products on the domain above:
within `relDot16 128·(|acc| + Σ|a·b|) + absDot16 128` of `acc + Σ a·b`, with
`relDot16 128 = 130·2^-22`, `absDot16 128 = 130·2^-126`), and the totality of `tl.dot_scaled`. The
final conversion to `float16` is IEEE rounding to nearest even.

## Tolerances

Per output element, by induction over the loop's `T = K / 128` steps (`acc_t` the accumulator
before step `t`, `P_t = partSum t`, `C_t = chunkAbs t`):
1. **One step**: the device fact gives `acc_{t+1}` finite and within `ρ·(|acc_t| + C_t) + α` of
   `acc_t + Σ_chunk p` (`ρ = relDot16 128`, `α = absDot16 128`). The scales' permute and reshape
   put byte `sIdx K r k` at entry `(r, k / 32)`; for `mixed`, element `k` of `b.T` is nibble `k % 2`
   of byte `k / 2` (`xbBits`).
2. **Induction**: `|acc_t − P_t| ≤ tol p t`, the recursion `tol p (t+1) = tol p t +
   ρ·(|P_t| + tol p t + C_t) + α`; the room condition keeps every step below `2^127`.
3. **The store**: one rounding to `float16`, `|c − v| ≤ 2^-11·|v| + 2^-25` when `|v| ≤ 65504`.

**Relaxation.** Unrolled, `tol p T ≤ (1 + ρ)^T·(ρ·μ + T·α)`; `K ≤ 2^17` gives `T ≤ 2^10`,
`(1 + ρ)^T ≤ 1.0323`, so `tolOut ≤ 2^-14·μ + 2^-108 = vTol`, and
`tolOut + (|S| + tolOut)·2^-11 + 2^-25 ≤ cTol`. The `float16` rounding stays relative to `|S|`: on
the host's data `μ` is about `100` to `170` times `|S|`.

On the host's data, `cTol / |S|` has median `1.6·10^-3` at `K = 512`, `6.9·10^-3` at `8192`,
`1.1·10^-2` at `16384` and `5.9·10^-2` at `2^17`; the tolerance still rejects an omitted or
repeated group of 32 products, a wrong scale or scale layout, swapped nibbles, or another tile's
element.

## Exclusions

- **`K > 2^17`**: outside real use; not a kernel limit, but the tolerance grows with the number of
  steps (about `25%` of the typical output at `2^20`).
- **`M`, `N`, `K` not positive multiples of `128`**: outside the file's launches (the host's scale
  tensors have `M // 128`, `N // 128`, `K // 128` blocks; the benchmark asserts `K % 128 == 0`).
- **`N > 2^17`, `M > 2^22`**: real model shapes only.
- **Sizes or grids at or above `2^31`**: a limit of the kernel's `int32` offsets and of TMA.
- **Input values outside `InDom` or without room**: NaN elements, scale bytes `0x00` and `0xFF`,
  scaled values outside `bfloat16`'s normal range, products reaching `2^127`; the device facts
  state nothing there, and the host never produces them.
- **`c` overlapping an input, other output dtypes, the `mxfp4` and `nvfp4` configurations, the AMD
  kernel**: not asked by `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the setup launches and negative tests.
