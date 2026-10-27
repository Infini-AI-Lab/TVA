# vec_add_f32

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/01-vector-add.py`](source/01-vector-add.py). Lean: [`lean/`](lean/), one module
`VecAdd` (namespace `Kernels.VecAdd`).

The kernel is `add_kernel` (source lines 29–54). The host function `add` (lines 62–78) allocates
`output = torch.empty_like(x)`, sets `n_elements = output.numel()`, and launches a 1-D grid of
`triton.cdiv(n_elements, BLOCK_SIZE)` programs with `BLOCK_SIZE=1024`. The script calls it on
`float32` vectors of 98432 elements (lines 84–89) and, in the benchmark, of `2^12 … 2^27`
elements (lines 107–129).

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `j < n`, element `j` of `x` and of `y` can be read in the initial
  memory (values `a`, `b`), element `j` of `output` can be read in the final memory (value `c`),
  and `c` is the IEEE `float32` sum `a + b`, rounded to nearest even (`FloatV.add`), up to NaN
  payload (`FloatV.NaNEq`: equal, or both NaN). There is no tolerance: the result is exact.
- **Nothing else changes.** Every byte outside the first `n` elements of `output` is the same
  in `fin` as in `init`. That covers the inputs (unless `output` is one of them, see below), the
  elements of `output` from `n` on, and all other memory.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `n` (`n_elements`), `BLOCK_SIZE`, and three
contiguous `float32` buffers `x`, `y`, `output` (`Buf .f32`: a base address and a length in
elements).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `block_pow2`, `block_le` | `BLOCK_SIZE` is a power of two, at most `2^20` | the block sizes Triton accepts for `tl.arange`; `Launch.Valid` needs them. The host's `1024` is one |
| `n_fits` | `n + BLOCK_SIZE ≤ 2^31` | the `int32` index arithmetic `pid * BLOCK_SIZE + tl.arange(…)` wraps for `n > 2^31`, `n_elements` is an `int32` only for `n < 2^31`, and library issue L8 excludes the last `BLOCK_SIZE − 1` sizes below `2^31` (see Narrowings) |
| `x_len`, `y_len`, `output_len` | each buffer has at least `n` elements | the kernel accesses elements `0 … n−1` of each |
| `output_x`, `output_y` | `output` does not overlap `x` (resp. `y`), or starts at the same address (in place) | a partial overlap can make programs race (see Narrowings) |

`x` and `y` may overlap each other in any way (both are only read), and may be the same tensor
(`add(x, x)`). `n = 0` is covered: the grid is empty and memory is unchanged.

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 4 bytes, the buffer inside `[0, 2^64)`, every byte present). Every
`float32` value is allowed in the inputs, NaN and infinities included.

## Assumptions

One assumption, of kind `[rule]`:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: the
  compiled kernel, launched as `launch d`, ends as some interleaving of the model's launch does,
  provided the model's launch is well behaved (no undefined behaviour, no race). It is cited as
  the library states it ([rule page](../../docs/rules/compiler-faithful.md)).

The kernel uses no approximate operation (its only float operation is an IEEE `+`), so it needs
no fact about the platform `P`, and there is no `[kernel]` assumption.

`consistent` proves the assumptions satisfiable: `TVA.exists_total_faithful` with
`Platform.any` gives a total behaviour that satisfies `faithful`.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` (Triton's default) by definition of `launch`.
- `hvalid` (`launch_valid`): the grid `cdiv(n, BLOCK_SIZE) ≤ n + BLOCK_SIZE − 1 < 2^31`, the
  other grid dimensions are `1`, `num_ctas = 1`, and the one block shape `[BLOCK_SIZE]` is a
  power of two of at most `2^20` elements.
- Footprints: program `p` reads its tiles of `x` and `y` (`R`) and writes its tile of `output`
  (`W`); a tile is the elements `p·BLOCK_SIZE + l` below `n` (`Block.Tile`, `Buf.tile`).
- `hsafe` (`safe_program`): `body_eq` (by `rfl`) puts program `p`'s body in the form of the
  library's 1-D block pattern, with `program_id` equal to `p` (`launch_pid`). Then one rule per
  line: `safe_load_block` for the loads of lines 50 and 51 (each enabled lane holds its
  element of the initial memory), `safe_add` for line 52 (the IEEE sum up to NaN payload),
  `safe_last` and `safe_store_block` for the store of line 54 (the tile holds the sums), and
  `safe_ret`. `Block.fits_of_grid` turns `n + BLOCK_SIZE ≤ 2^31` into the `Block.Fits` each
  block rule needs.
- `hdisj`: for programs `i ≠ j`, `output`'s tile `i` is disjoint from tile `j` of `x`, `y` and
  `output`: tiles of one buffer by `Buf.tile_disjoint`; of disjoint buffers by
  `Buf.tile_disjoint_of`; and when `output` starts where `x` (or `y`) does, its tiles are that
  buffer's tiles (`tile_disjoint_in`).
- `hpost`: element `j < n` is in the tile of program `j / BLOCK_SIZE` (`Block.tile_div`), which
  is in the grid (`div_lt_grid`); the final memory agrees with that program's memory on its
  footprint, so `Mem.read_congr` gives the value. A byte outside `output`'s first `n` elements
  is in no write footprint, so it keeps its initial value.

`main` and `consistent` use only the axioms `propext`, `Classical.choice` and `Quot.sound` (the
gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float32` only.** The kernel's source is dtype-generic, but its host code and benchmark use
  `float32` (`torch.rand` without a dtype, and `dtype=torch.float32`). Other dtypes, and mixed
  dtypes (which would add promotions), are not covered.
- **`n + BLOCK_SIZE ≤ 2^31`.** The bound excludes three ranges of sizes, each for its own
  reason. With the host's `BLOCK_SIZE = 1024` it is `n ≤ 2^31 − 1024`; the script's largest
  size is `2^27`.
  - `n > 2^31`: the indexing wraps. Every covered `BLOCK_SIZE` divides `2^31`, so the last
    program has `pid ≥ 2^31 / BLOCK_SIZE`, and its `int32` product `pid * BLOCK_SIZE` is at
    least `2^31` and wraps. For example, with `BLOCK_SIZE = 1024` and `n = 2^31 + 1`, the last
    `pid` is `2^21` and `pid * BLOCK_SIZE` wraps to `−2^31`. The enabled lanes of such a program
    then access memory before the buffers (out of bounds), and from `n > 2^32` on, some programs'
    offsets wrap back onto other programs' elements. The kernel is not meant for such sizes.
  - `n = 2^31`: nothing wraps. The grid is `2^31 / BLOCK_SIZE` programs and the largest offset
    is `(grid − 1)·BLOCK_SIZE + BLOCK_SIZE − 1 = 2^31 − 1`, so the kernel's indexing is exact at
    this size. It is excluded because Triton passes `n_elements = 2^31` as `int64`, while the
    model takes `n_elements : I32` (`IntV.ofNat n`; map entry for lines 30–36). Covering it
    would need another model, with an `int64` `n_elements` and the `int64` comparison
    `offsets < n_elements`.
  - `2^31 − BLOCK_SIZE < n < 2^31`: the kernel's indexing is exact (`BLOCK_SIZE` divides
    `2^31`) and `n_elements` is an `int32`, but the library's `Block.Fits` asks
    `p·BLOCK_SIZE + BLOCK_SIZE < 2^31`, which excludes the last program there; this is library
    issue L8.
- **Contiguous buffers.** Each tensor is a contiguous `Buf`, as `torch.rand` and
  `torch.empty_like` of a contiguous tensor give. The kernel ignores strides, so it is not meant
  for strided tensors.
- **`y` has at least `n` elements.** The host takes `n` from `output` (that is, from `x`) and
  does not check `y`'s size; a shorter `y` would be read out of bounds, so the condition is
  needed.
- **Aliasing.** `output` is disjoint from `x` and `y` or equal to one of them (in place); the
  host always allocates a fresh `output`, so its use is the disjoint case. Partly overlapping
  `output` and inputs are not covered. An overlap within the first `n` elements can make
  programs race; the conditions are stated on whole buffers (`Buf.bytes`), so they also exclude
  an overlap that lies only in elements from `n` on, which would be harmless.

Modelling decisions:

- `BLOCK_SIZE` ranges over every block size Triton accepts, not only the host's `1024`.
- `n_elements` is an `int32` (`IntV.ofNat n`): Triton passes an integer argument below `2^31` as
  `int32`. Triton's specialisation of arguments (pointers and integers divisible by 16, an
  integer equal to `1`) is the compiler's, covered by `Compiler.Faithful`.
- `pid * BLOCK_SIZE` multiplies two `int32` values: the constexpr becomes an `int32` constant.
- The loads have the source's `mask` and no `other`, so disabled lanes are unspecified; the
  proof uses only the enabled lanes, and the store has the same `mask`.
- `x + y` is `tl.add` (IEEE): its operands are loads, not products, so no contraction applies.
- The launch lists one block shape, `[BLOCK_SIZE]`, the shape of every block value of the
  source (`tl.arange`, `offsets`, `mask`, `x`, `y`, `output`); `enable_fp_fusion` is written as
  Triton's default `true` (it does not matter here).
