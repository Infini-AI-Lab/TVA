# `vec_add_f32`: prepared claim

Triton tutorial 01's `add_kernel` (`source/01-vector-add.py`, kernel lines 29–54 with the imports of
lines 23–24; host `add` 62–78). `Claim.lean`, namespace `Claims.VecAdd`; the claim to prove is
`Statement`. The launches to cover are in `SETUP.md`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.VecAdd.main : Statement` (`Statement`: `Claim.lean`
line 134). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 69), and every initial memory `init` that `Pre` accepts (line 105), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 113).

**What `Post` asks** (line 113), two parts:
1. for every `j < n`: elements `j` of `x` and `y` read in `init` as `a`, `b`, element `j` of
   `output` reads in `fin` as `c`, and `c` is the IEEE `float32` sum `FloatV.add a b` up to NaN
   payload (`FloatV.NaNEq`). It is exact: no tolerance;
2. the frame: every byte outside the first `n` elements of `output` is unchanged.

**What you are given** (`Assumptions`, line 125): one field, `faithful` (`Compiler.Faithful`): the
device's outcomes `K d` are outcomes of the model's launch, when the launch is valid, has
`debug = false`, and the model's launch is well behaved from `init` (no out-of-bounds access, no race).
No numeric fact: the sum is the IEEE operation.

**The kernel, as the model runs it** (`add_kernel`, `model` line 93, `launch` line 100): a 1-D grid
of `cdiv(n, BLOCK_SIZE)` programs; program `p` loads the masked tile
`p·BLOCK_SIZE + [0, BLOCK_SIZE)` of `x` and `y`, adds, and stores the tile of `output`. Programs
write disjoint tiles and share nothing: the route is `TVA.meets_independent` (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: `block_pow2`, `block_le`, and
  fewer than `2^31` programs from `n_lt`);
- each program is `Safe`: its loads and its store are in bounds (`x_len`, `y_len`, `output_len`;
  `int32` offsets do not wrap: `Block.fits_of_grid_pow2` from `n_lt`), and it writes only its tile;
- the programs' write tiles are disjoint, and no program reads what another writes (`output_x`,
  `output_y`);
- the stored values give part 1 (`safe_add`), and the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `elems_disjoint_of_lt` (line 152) can be reused.

## Specification

`Post d init fin`:
- for every `j < n`, elements `j` of `x` and `y` can be read in `init` (values `a`, `b`), element `j`
  of `output` can be read in `fin` (value `c`), and `c` is the IEEE `float32` sum `a + b` rounded to
  nearest even (`FloatV.add`), up to NaN payload (`FloatV.NaNEq`). The result is exact: there is no
  tolerance;
- frame: every byte outside the first `n` elements of `output` is unchanged (the inputs, the rest of
  `output`, every other byte).

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch that
runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

`Inst`: `n` (`n_elements`), `BLOCK_SIZE`, and the contiguous `float32` buffers `x`, `y`, `output`.

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `block_pow2`, `block_le` | `BLOCK_SIZE` a power of two, at most `2^20` | the block sizes Triton accepts; `Launch.Valid`. Every one is covered, not only the host's `1024` |
| `n_lt` | `n < 2^31` | `n_elements` is an `int32` argument (a compile-time `1` at `n = 1`, the same model); the `int32` offsets of the grid `cdiv(n, BLOCK_SIZE)` do not wrap (`Block.fits_of_grid_pow2`) |
| `x_len`, `y_len`, `output_len` | each buffer holds at least `n` elements | the kernel accesses elements `0 … n−1` of each; the host does not check `y`'s size |
| `output_x`, `output_y` | the first `n` elements of `output` share no byte with the first `n` of `x` (of `y`) | a program that reads and writes the same bytes, or programs touching bytes another writes, are not well behaved; the host's `output` is fresh |

`x` and `y` may overlap in any way (`add(x, x)` is covered); `n = 0` is covered (empty grid).

`Pre`: the three buffers are valid (`Buf.Valid`). Every `float32` input value is covered, NaN and
infinities included.

The launch: `Launch.grid1 (cdiv(n, BLOCK_SIZE)) [[BLOCK_SIZE]]`, `enable_fp_fusion = true`,
`debug = false`, `num_ctas = 1`, no `launch_pdl`. Part (A) of `Compiler.Faithful`'s scope
(`docs/rules/compiler-faithful.md`): (4) the launch gives no option, only the constexpr; no
`warmup`, `triton.compile`, `@triton.autotune` or `@triton.heuristics`, and a bare `@triton.jit`;
(5) no tensor descriptor; (6) no `CompiledKernel`; (8) no host `TensorDescriptor`. The claim relies on part (B), the
environment, which is assumed and not checked.

The kernel is inside the supported subset (`docs/SEMANTICS.md`): a top-level `def` under a bare
`@triton.jit`, three pointer parameters, one unannotated integer, one `tl.constexpr`; it reads only
the module-level names `triton` and `tl`, each bound only by its own `import`.

## Assumptions

One `[rule]` field, `faithful` (`Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every
covered `d`). The only float operation is an IEEE `+`, which needs no device fact. No `[kernel]`
field.

## Coverage

`setup_test_98432` and `setup_bench_2p12` … `setup_bench_2p27`, with `setup_*_ok`: the test and the
sixteen benchmark sizes of `SETUP.md`, `BLOCK_SIZE = 1024`, fresh buffers (`mk`). `mk_ok` covers the
host's launch at every `n < 2^31` and every block size `2^e ≤ 2^20`.

## Tolerances

None: `Post` states the exact IEEE result. It is provable because the model's only float operation is
`tl.add`, whose meaning is the IEEE sum up to NaN payload (`safe_add`).

## Exclusions

- **`n ≥ 2^31`** — outside the file's launches (largest `2^27`), and from `2^31` on `n_elements`
  is an `int64` argument (another specialisation). Beyond `2^31` it is also a kernel limit: the last
  program's `int32` product `pid * BLOCK_SIZE` wraps (first launch: `n = 2^31 + 1` with
  `BLOCK_SIZE = 1024`), and its lanes would access memory before the buffers. Real use does not reach it.
- **Dtypes other than `float32`** — outside the file's launches (the source is dtype-generic, the
  file uses `float32` only).
- **`output` overlapping the first `n` elements of `x` or `y`** (an in-place `add`) — outside the
  file's launches (`output` is always fresh); it would also be a conflict within one program, which
  the library does not order.
- **Strided tensors** — outside the file's launches; the kernel ignores strides.

## Library requests

None.

## Checks

- `Claim.lean` builds with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`
  and every `setup_*_ok` use only `propext`, `Classical.choice`, `Quot.sound`.
- `check_gpu.py` (H200): the test, the sixteen benchmark sizes, random bit patterns at
  `n = 1, 2, 1023, 1025, 4097, 98433`, 169 pairs of special values, ties, `add(x, x)`; direct launches
  with `BLOCK_SIZE` `1, 16, 64, 128, 1024, 32768`. Every output element is the IEEE sum bit for bit
  (both NaN counted equal), the inputs are unchanged bitwise, and a sentinel-guarded output is
  untouched outside its first `n` elements: 26 of 26 cases pass.
