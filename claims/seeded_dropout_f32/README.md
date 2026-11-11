# `seeded_dropout_f32`

**The kernel.** `_seeded_dropout` of Triton 3.8.0's tutorial 04, "Low-Memory Dropout"
(`source/04-low-memory-dropout.py`, kernel lines 110–131 with the imports of lines 38–39; host
`seeded_dropout` lines 134–140). Each program of a 1-D grid of `cdiv(n, BLOCK_SIZE)` programs loads
a masked tile of `x`, draws `tl.rand(seed, offsets)` for each lane, keeps the lanes whose random
number is above `p` and divides them by `1 - p`, sets the others to `0`, and stores the tile of
`output`. The claim is `Claims.SeededDropout.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
`j < n`, with `a = x[j]` initially and `c = output[j]` finally:
- `keep j` is `p32 < uRef seed j`, where `p32` is the `float32` the kernel receives for `p` and
  `uRef seed j` is Triton's generator computed from the host's `seed` and the index `j`: the first
  word of Philox 4x32-10 on the counter `(j mod 2^32, 0, 0, 0)` and the key `(seed mod 2^32, high 32
  bits of seed mod 2^64)`, complemented when negative as an `int32`, rounded to `float32` and
  multiplied by `Philox.scale32` (one IEEE product);
- if `j` is not kept, `c` is `+0` exactly (bit pattern `0`);
- if `j` is kept and `a` is finite with `|a/dRef| ≤ 2^126`, `c` is finite and
  `|c − a/dRef| ≤ 2^-22·|a/dRef| + 2^-148`, where `dRef` is the IEEE `float32` `1 − p32`;
- every byte outside the first `n` elements of `output` is unchanged.

The keep/drop decision is exact, bit for bit. A kept element with a NaN or infinite `a`, or a larger
quotient, need only be readable.

**The conditions** (`Inst.ok`, `Pre`):
- `BLOCK_SIZE` is a power of two at most `2^20` (every block size Triton accepts);
- `n < 2^31` (`n_elements` is an `int32`; the `int32` offsets do not wrap); `n = 0` is covered;
- `p32` is finite and `0 ≤ p32 ≤ 1` (`-0.0` included);
- `-2^63 ≤ seed < 2^64` (Triton rejects other integers on the host); the model passes the seed as an
  `int32`, `int64` or `uint64` according to its value, as Triton does;
- `x` and `output` hold at least `n` elements each and are valid in the initial memory;
- the first `n` elements of `output` share no byte with the first `n` elements of `x`.

Every `float32` value of `x` is covered, NaN and infinities included.

**The launches covered** (`SETUP.md`): the file's launches of lines 145–147 (`n = 10`, `p = 0.5`,
`BLOCK_SIZE = 1024`, seeds `123` and `512`; `setup_*_ok`). `mk_ok` covers the host's launch with
`p = 0.5` at every `n < 2^31`, every block size `2^e ≤ 2^20` and every seed in `[-2^63, 2^64)`.

**Device facts** the tolerance rests on (in `Device P`): `Device.DivFull` (`/` on `float32`:
relative `2^-22`, absolute `2^-148`, for finite operands with a quotient up to `2^126`) and the
totality of `/`. The generator is integer arithmetic; its conversion and product, and `1 − p`, are
IEEE operations; the comparison is exact.

## Exclusions

- **`p` outside `[0, 1]`** (as its `float32`: negative, above `1`, infinite or NaN): `p` is a
  dropout probability, and the file uses `0.5`.
- **Kept elements with a NaN or infinite `x`, or with `|x/(1−p)| > 2^126`**: `Device.DivFull`
  promises nothing there, so `Post` asks only that the output be readable.
- **`n ≥ 2^31`**: at `2^31` Triton passes an `int64`, which the model does not take; beyond, the last
  program's `int32` product `pid * BLOCK_SIZE` wraps.
- **Seeds outside `[-2^63, 2^64)`**: rejected by Triton on the host.
- **Dtypes other than `float32`**, **`output` overlapping `x`** (in place), **strided tensors**:
  outside the file's launches.
- **The baseline `_dropout`** (lines 44–63): another kernel of the file.

`check_gpu.py` (one H200) checks the generator against Philox known-answer vectors and the GPU bit
for bit, and checks `Post` on the file's launches and on edge cases of `Inst.ok`.
