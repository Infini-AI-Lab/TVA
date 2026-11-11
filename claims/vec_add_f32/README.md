# `vec_add_f32`

**The kernel.** `add_kernel` of Triton 3.8.0's tutorial 01, "Vector Addition"
(`source/01-vector-add.py`, kernel lines 29–54; host `add` lines 62–78). Each program of a 1-D grid
of `cdiv(n, BLOCK_SIZE)` programs loads a masked tile of `x` and `y`, adds them and stores the tile
of `output`. The claim is `Claims.VecAdd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and in the
final memory element `j < n` of `output` is the IEEE `float32` sum `x[j] + y[j]` of the initial
inputs, rounded to nearest even, up to NaN payload (`FloatV.add`, `FloatV.NaNEq`). It is exact:
there is no tolerance. Every byte outside the first `n` elements of `output` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `BLOCK_SIZE` is a power of two at most `2^20` (every block size Triton accepts, not only the
  host's `1024`);
- `n < 2^31`: `n_elements` is an `int32` argument, and the `int32` offsets
  `pid * BLOCK_SIZE + arange` do not wrap;
- `x`, `y` and `output` each hold at least `n` elements, and are valid in the initial memory
  (aligned, every byte present);
- the first `n` elements of `output` share no byte with the first `n` elements of `x` or of `y`
  (the host's `output` is fresh). `x` and `y` may overlap (`add(x, x)` is covered); `n = 0` is
  covered.

Every `float32` input value is covered, NaN and infinities included.

**The launches covered** (`SETUP.md`): the test (`n = 98432`) and the benchmark
(`n = 2^12 … 2^27`), all with `BLOCK_SIZE = 1024` (`setup_*_ok`); `mk_ok` covers the host's launch
at every `n < 2^31` and every block size `2^e ≤ 2^20`.

**Device facts.** None: the only float operation is an IEEE `+`.

**Not covered.** `n ≥ 2^31` (then `n_elements` is an `int64` argument, and the last program's
`int32` product `pid * BLOCK_SIZE` wraps); dtypes other than `float32`; an `output` overlapping the
first `n` elements of `x` or `y` (an in-place `add`); strided tensors (the kernel ignores strides).
None of these is among the file's launches.

`check_gpu.py` (one H200) checks `Post` bit for bit on the script's launches and on the edges of
`Inst.ok`.
