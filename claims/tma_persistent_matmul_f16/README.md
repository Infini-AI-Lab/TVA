# Prepared claim: `tma_persistent_matmul_f16`

The kernel: `source/09-persistent-matmul.py`, Triton 3.8.0's tutorial 09, the TMA kernel
`matmul_kernel_tma_persistent` (lines 407–463) with its helper `_compute_pid` (lines 291–298), on
`float16` matrices with the config of `SETUP.md` (`BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`,
`BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`, `EPILOGUE_SUBTILE = True`, `num_stages = 4`,
`num_warps = 8`, and `NUM_SMS`, `132` on the H200), launched by the host
`matmul_tma_persistent(a, b, False)` (lines 466–503) with three host `TensorDescriptor`s, on a grid
of `min(NUM_SMS, tiles)` programs, each looping over the tiles `start_pid, start_pid + NUM_SMS, …`
of `c = a · bᵀ` and storing each tile as two `128 × 128` halves. The claim is
`Claims.TmaPersistentMatmul.Statement` in `Claim.lean`:

```lean
def Statement : Prop := ∀ K P, Assumptions K P → Spec K
```

`Claim.lean` proves `consistent`, `satisfiable` and the coverage facts `setup_*_ok`;
it does not prove `Statement`. The model map is `model-map.json` (every code line of the kernel
lines `[44, 45], [291, 298], [407, 463]`).

The kernel reads three module-level names: `triton` (its decorators) and `tl`, bound by the imports
of lines 44–45, which are kernel lines and are mapped (as `import` lines that add no step), and the
`@triton.jit` helper `_compute_pid`, bound only by its `def` (line 292). Every other module-level
statement is a host line (`claim.json`'s `source.host`): the other imports, the device helpers,
`_matmul_launch_metadata` (lines 89–100, the host function the decorator's `launch_metadata=`
names), the config lists, the pre-hook `matmul_tma_set_block_size_hook`, the file's other kernels
and their hosts, `matmul_tma_persistent`, the test, the benchmark and `__main__`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.TmaPersistentMatmul.main : Statement` (`Statement`:
`Claim.lean` line 372). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 199), and every initial memory `init` that `Pre` accepts (line 283: the three
buffers valid, any values), every outcome of the launch is a normal end (no fault, no hang) whose
final memory `fin` satisfies `Post` (line 342).

**What `Post` asks** (line 342), two parts:
1. for every `i < M`, `j < N`, `OutSpec d init fin i j` (line 328): element `i * ldc + j` of `c`
   reads in `fin` as some `float16` `c` (on every input); and for all `va vb : Nat → F16` such that,
   for every `k < K`, `init` reads `a`'s element `i * lda + k` as a finite `va k` and `b`'s element
   `j * ldb + k` as a finite `vb k`, there is a finite `float32` `v` with
   - `|v.toRat - dotSum va vb d.K| ≤ tol va vb d.K`, and
   - `FloatV.NaNEq (FloatV.cvt v) c`: `c` is `v` rounded to `float16`, up to NaN payload;
   - and, when `|dotSum| + tol ≤ binary16.maxVal`: `c` is finite.

   The reference `dotSum` (line 292) is the exact `Σ_{k<K} va k · vb k` in the rationals. The
   tolerance `tol` (line 321) is `2^-15·pathMag va vb K + 2^-112`, with `pathMag` (line 315) the
   sum over the inner loop's `steps K` (line 289) steps of `|partSum p t| + stepMag q t`, the
   exact partial sum before step `t` plus the step's lifted magnitude; the helpers are `prodV`
   (line 296), `liftV` (line 302), `partSum` (line 306), `stepMag` (line 309). With a non-finite
   input, only "holds a value" is asked;
2. the frame: every byte outside the matrix `c` (`c.elems (cIdx d)`, `cIdx` line 193) is unchanged.

**What you are given** (`Assumptions`, line 350):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, when the launch is
  valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotF16`, a `DotLiftAcc`): the `tl.dot` of source line 442 is within
  `relDot16 K · (|acc| + Σ liftMag a · liftMag b) + absDot16 K` of `acc + Σ a·b` when the
  operands and `acc` are finite, with room below the `float32` range;
- `dot_total` (`Rules.Num.DotTotal` for the model's configuration): that `tl.dot` has a result.

**The kernel, as the model runs it** (`matmul_kernel_tma_persistent`, line 71, with `compute_pid`,
line 34; `model` line 267, `launch` line 276): a 1-D grid of `min NUM_SMS (tiles d)` programs
(`tiles`, line 184). Program `start_pid` runs `tl.forRange start_pid num_tiles NUM_SMS` over tile
ids. For each tile id it computes `(pid_m, pid_n)` by the grouped order (`GROUP_SIZE_M = 8`),
runs an inner `tl.forRange 0 k_tiles 1` of two `tl.desc_load`s (`aDesc`, `bDesc`, lines 247, 250;
padding `+0`) and one `tl.dot(a, b.trans, acc)` into a zero `128 × 256` accumulator, then
recomputes the pid from the carried `tile_id_c` (equal to the tile id on each trip), reshapes,
permutes and splits the accumulator into two `128 × 128` halves, converts each to `float16`
(`tl.cvt`) and stores them with two `tl.desc_store`s (`cDesc`, line 254) at columns `pid_n·256`
and `pid_n·256 + 128`. Each tile is written by one program, once; programs read only `a` and `b`
and use no atomics. The route is `TVA.meets_independent` (PROVER_GUIDE §4), with
`safe_forRange` for both loops: the outer one with per-trip write footprints (the trip's tile,
disjoint across trips), the inner one with `fun _ => ∅`.

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: `min NUM_SMS (tiles d) <
  2^31` from `sms`; the seven listed block shapes are powers of two);
- each program is `Safe`: no `int32` value wraps and the tile loop's bounds meet `tl.RangeOk`
  (`M_le`, `N_le`, `sms`, `tiles_le`); each descriptor is `TensorDesc.Valid` over its buffer
  (`pos`, `ld8`, `base_al`, `a_len`, `b_len`, `c_len`) for `safe_desc_load`, `safe_desc_store`;
  the stored lanes are distinct elements (`N_ldc`); `safe_idiv`, `safe_irem`, `safe_dot`
  (`dot_total`), `safe_cvt` for the steps;
- footprints: the grouped order maps the tile ids `[0, tiles d)` one to one onto the tiles, so
  the tiles of different programs (ids `≡ start_pid` mod `NUM_SMS`) and of different trips are
  disjoint; the two halves of a tile are disjoint columns; `c`'s matrix is disjoint from the
  reads (`c_a`, `c_b`);
- the value part of `Post`, `|v − dotSum| ≤ 2^-15·pathMag + 2^-112`: derive the fine bound of
  README `## Tolerances` (the inner loop leaves a finite `float32` within `E_T` of `dotSum`: `dot`,
  step by step, by induction on the recursion), then relax it (Tolerances, Relaxation); the
  reshape, permute and split move each element unchanged to its column (`Tensor.get_reshape`,
  `Tensor.get_permute`, `Tensor.join_split` with `Tensor.get_join_snoc`); the conversion is
  `FloatV.cvt`, finite with room (`|v| ≤ |S| + tol ≤ 65504`);
- the frame: the bytes no program writes.

`N_dim`, `K_dim` and `M_ctx` restrict the instances to real model shapes. With `128 ∣ K` every
inner step loads a full `64`-column block (`64·steps K = K`: `prodV` has no padded lane), and with
`128 ∣ N` each `128`-column store half lies wholly inside or wholly past `N`; the partial row tile
(any `M`) remains, and the last column tile's second half lies past `N` when `N` is an odd
multiple of `128`.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `hostInst_ok` and
the `setup_*_ok` theorems; `row_lt` (line 402) and `bytes_disjoint` (line 408) can be reused.

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `c` (element `i·ldc + j` of
  `c`'s storage) can be read in the final memory, with some value `c` (`OutSpec`). When, moreover,
  row `i` of `a` and row `j` of `b` hold finite `float16` values `va k` and `vb k` (`k < K`) in the
  initial memory:
  - `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
    `float32` value `v` with

    `|v − S| ≤ E = 2^-15·pathMag + 2^-112`,

    where `S = Σ_k va k · vb k` is the exact dot product and `pathMag = Σ_{t<T} (|S_t| + L_t)` the
    magnitude the inner loop carries over its `T = cdiv(K, 64)` steps: the exact partial sum
    `S_t` before step `t` plus the step's lifted magnitude `L_t` (`Σ liftMag(va k)·liftMag(vb k)`
    over the step's 64 `k`); `pathMag ≥ Σ_k |va k·vb k| ≥ |S|`;
  - when `|S| + E ≤ 65504` (the largest `float16`), `c` is finite.

  With a NaN or an infinity in row `i` of `a` or row `j` of `b`, the claim says only that element
  `(i, j)` holds some `float16` value.
- **Nothing else changes.** Every byte outside the elements of the matrix `c` is the same in `fin`
  as in `init`: `a`, `b`, and the elements of `c`'s storage that are not in the matrix (the padding
  of a row, when `ldc > N`).

The reference `S` is the exact product of the host's matrices, row `i` of `a` by row `j` of `b`
(`c = a · bᵀ`, as `matmul_tma_persistent(a, b)` means it, `b` being the `N × K` transpose); it does
not depend on which program computes which tile, on the tile order of `_compute_pid`, on how the
epilogue splits a tile, nor on `NUM_SMS`.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `M`, `N`, `K`, `NUM_SMS`, the row strides `lda`, `ldb`,
`ldc` (in elements; the last stride is `1`, as `TensorDescriptor` requires), and three `float16`
buffers `a`, `b`, `c` (`Buf .f16`: a base address and a length in elements); element `(i, k)` of the
matrix `a` is element `i·lda + k` of the buffer `a`, element `(j, k)` of `b` element `j·ldb + k`,
element `(i, j)` of `c` element `i·ldc + j` (`aIdx`, `bIdx`, `cIdx`). The model's descriptors are
the host's: `aDesc` (shape `[M, K]`, strides `[lda, 1]`, block `[128, 64]`), `bDesc` (`[N, K]`,
`[ldb, 1]`, `[256, 64]`), `cDesc` (`[M, N]`, `[ldc, 1]`, `[128, 128]`), each with the padding
`"zero"`. `tiles d = cdiv(M, 128)·cdiv(N, 256)`.

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `pos` | `M, N, K ≥ 1` | the host's `TensorDescriptor` rejects a zero shape entry (no launch), and an access through a descriptor needs positive ones (`TensorDesc.Valid`) |
| `M_le`, `N_le` | `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31` | `M` and `N` are `int32` arguments, and `tl.cdiv` computes `M + 127` and `N + 255` in `int32` (lines 421–422); the tile offsets `pid_m·128`, `pid_n·256` and `offs_bn_c + 128` (line 460) then stay below `2^31`, and the descriptors' shape entries are below `2^31` |
| `sms` | `0 < NUM_SMS < 2^31` | the tile loop's step (line 432) is positive (`tl.RangeOk`), and the constexpr is built as an `int32` (lines 426, 432, 444) |
| `tiles_le` | `tiles + NUM_SMS ≤ 2^31` | the `int32` `num_tiles` (line 424) does not wrap, and `num_tiles + NUM_SMS − 1`, the tile loop's bound (`tl.RangeOk`), is an `int32` |
| `K_le` | `K ≤ 2^13` | keeps the tolerance meaningful (Tolerances); the setup's largest `K` is `1024` |
| `N_dim`, `K_dim` | `N % 128 = 0`, `K % 128 = 0`, both `≤ 2^17` | real model shapes: `N` and `K` (the output and input features) are model dimensions (Exclusions) |
| `M_ctx` | `M ≤ 2^22` | real model shapes: `M` (the tokens) is a context length (Exclusions) |
| `ld8` | `lda`, `ldb`, `ldc` multiples of `8` | a descriptor's leading stride is a multiple of 16 bytes (`TensorDesc.Valid`; `TensorDescriptor` asserts it) |
| `N_ldc` | `N ≤ ldc` | the rows of `c` do not overlap, so no two lanes or programs store to one element |
| `base_al` | the three bases are 16-byte aligned | `TensorDesc.Valid` (`TensorDescriptor` asserts it; PyTorch's allocations are) |
| `a_len`, `b_len`, `c_len` | every element of each matrix is inside its buffer | the descriptors' accesses reach them |
| `c_a`, `c_b` | the matrix `c` shares no byte with the matrix `a`, nor with the matrix `b` | a program that reads bytes another program writes races; `matmul_tma_persistent` always allocates a fresh `c` |

The descriptors' accesses compute each lane's element as `i·ld + k` in exact integers (`TensorDesc`,
no `int32` offsets), so the claim has no bound on the size of a matrix beyond `M_le`, `N_le`,
`K_le`, `N_dim`, `K_dim` and `M_ctx`. `M`, `N`, `K` are the kernel's runtime arguments (or, at the
value `1`, compile-time constants, which give the same model: `model`'s docstring). Their values
are part of the instance, and the conditions above constrain them; `Compiler.Faithful`'s point 11
assumes that the launch runs the code Triton compiles for these values. Every `K` block is full
(`64 ∣ 128 ∣ K`). The conditions allow a partial row tile (any `M` from `1`: the loads pad with
zeros past the tensors' edges and the stores drop the lanes outside `c`), a last column tile whose
second `128`-column half lies wholly past `N` (`N` an odd multiple of `128`), any row strides that
are multiples of `8` (padded rows of `a`, `b`, `c`), any `NUM_SMS` within the bounds, and any
addresses meeting the alignment; `a` and `b` may overlap each other in any way (both are only
read).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 2 bytes, the buffer inside `[0, 2^64)`, every byte present). Every
`float16` value is allowed in the inputs, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of `SETUP.md` is a direct launch of the `@triton.jit` function, the
JIT launch `matmul_kernel_tma_persistent[grid](…)` of `matmul_tma_persistent` (lines 496–502),
called by `run_test` from `validate` (line 735) and, in the benchmark, by `bench_fn` (line 694).
Point by point:

- **4 (options).** The launch gives the constexprs `FP8_OUTPUT`, `NUM_SMS`, `WARP_SPECIALIZE`,
  `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `EPILOGUE_SUBTILE` and two
  options, `num_stages=4` and `num_warps=8`, two of the four that do not change the meaning; no
  `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any other option. Nothing calls
  `matmul_kernel_tma_persistent.warmup(…)` or `triton.compile`; the kernel has no
  `@triton.autotune` (so no `triton.Config`) and no `@triton.heuristics`, and its decorator
  `@triton.jit(launch_metadata=_matmul_launch_metadata)` sets no `debug` (`launch_metadata=` only
  names a host function for a launch-enter hook). So `launch` keeps every default:
  `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor (no
  `tl.make_tensor_descriptor`), so no allocator is involved.
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is the JIT launch above.
- **8 (host descriptors).** The launch is given three host `TensorDescriptor`s, built by
  `TensorDescriptor.from_tensor(x, [1, 1])` (lines 481–483) with the default
  `round_f32_to_tf32=False`, their block shapes then set by the pre-hook (line 495), as SEMANTICS.md
  ("One launch") allows; their shapes, strides, block shapes and padding are `aDesc`, `bDesc`,
  `cDesc`.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it. Programmatic dependent launch
is off: no launch passes `launch_pdl`, and the kernel calls no `gdc_*` function.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under `@triton.jit(launch_metadata=…)`; its helper `_compute_pid` a
top-level `def` under a bare `@triton.jit`, called by name from the kernel. The kernel's parameters
are three host `TensorDescriptor`s, three unannotated integers (`int` arguments) and eight
`tl.constexpr`s with `int` or `bool` arguments, none with a default; the helper's five parameters
are unannotated, with no default, and both calls pass three `int32` scalars and the same two
constexprs. The kernel's statements are assignments `name = e`, a conditional expression on a
compile-time condition (line 419), three tuple assignments `a, b = e` (the helper's tuple result
twice, `tl.split` once), one `name += e` (line 444), expression statements (the descriptors'
`store`), a `for tile_id in tl.range(…)` with three positional arguments and the keywords `flatten`
and `warp_specialize`, a nested `for ki in range(k_tiles)` with one positional argument, and an
`if`/`else` on a compile-time condition; the helper's are assignments and one
`return pid_m, pid_n`. The expressions are `tl.program_id(axis=0)`, `tl.cdiv`,
`tl.zeros(…, dtype=tl.float32)`, the descriptor methods `load` and `store` with a list of offsets,
`tl.dot(a, b.T, acc)` with the attribute `.T`, `tl.reshape`, `tl.permute`, `tl.split`,
`x.to(dtype)`, Python's builtins `min` and `range`, the operators `+ - * // %`, the dtype names
`tl.float8e4nv` and `tl.float16`, and tuples and lists of integers. After each loop no name is
used: the tile loop carries `tile_id_c`, the inner loop `accumulator`, and neither loop's variable
nor a name first defined in a body is read after it. The module-level names the kernel reads are
`triton` and `tl`, Triton's modules, each bound only by its own `import` (lines 44 and 45), and
`_compute_pid`, bound only by its `def`; no other statement of the module binds `triton`, `tl`,
`_compute_pid` or `matmul_kernel_tma_persistent`.

## Assumptions

Three assumptions, all of kind `[rule]`, each cited as the library states it
(`docs/PROVER_GUIDE.md` section 3, [num.md](../../docs/rules/num.md)):

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`;
- `dot : Rules.Num.DotF16 P` — `tl.dot` of `float16` operands into `float32` (line 442), for every
  `input_precision`, with `maxImpreciseAcc = 0` (`Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at `2^-14`), when that leaves room below the `float32` range;
- `dot_total : Rules.Num.DotTotal P {a := .f16, b := .f16, acc := .f32, precision := .tf32,
  maxImpreciseAcc := 0}` — the `tl.dot` of line 442 always has a result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversions of lines 457 and 459 are
IEEE roundings the library defines (`tl.cvt`), the reshape, permute and split of lines 454–456 are
exact (`Tensor.reshape`, `permute`, `split`), and the descriptors' loads and stores are the
library's `tl.desc_load` and `tl.desc_store`; none needs a platform fact. The tile loop's
`flatten=True` and the launch's `num_stages=4` are scheduling choices with no meaning for a
well-behaved model, and `warp_specialize` is `False` (SEMANTICS.md, "Performance-only arguments",
"Loops"): each program writes each of its tiles once (the two halves of a tile are disjoint
columns) and reads only `a` and `b`, so no program accesses bytes it wrote.

## Coverage

Every launch of `SETUP.md` ("The launches"): `a` contiguous `[M, K]` (row stride `K`), `b`
contiguous `[N, K]` (row stride `K`), the fresh `c` (row stride `N`), `NUM_SMS = 132`;
`M = N = 8192` with `K = 128, 256, …, 1024`. `Inst.ok` covers them: the sizes are positive,
`K` and `N` are multiples of `128` (so the row strides are multiples of `8`) and at most `2^17`,
`M ≤ 2^22`, `tiles = 2048` and `tiles + 132 ≤ 2^31`, `N ≤ ldc = N`, and `K ≤ 1024 ≤ 2^13`. The
test's first call, `validate(32, 32, 32)` (`M = N = K = 32`), is outside `Inst.ok`: `N` and `K` are
not multiples of `128` (`N_dim`, `K_dim`; Exclusions).

Proved instances (`hostInst M N K S`: the host's strides, `NUM_SMS = S`, the three buffers at fixed
aligned addresses; `hostInst_ok`):

| Instance | `M = N` | `K` | Source |
|---|---|---|---|
| `setup_k128_ok` | 8192 | 128 | the test `validate(8192, 8192, 128)` (line 771) with `--K_range 128 1024 --K_step 128` (line 35), and `bench(128)` (lines 674–698) |
| `setup_k512_ok` | 8192 | 512 | the default `-K 512` (line 753): `validate(8192, 8192, 512)` and `bench(512)` |
| `setup_k256_ok`, `setup_k384_ok`, `setup_k640_ok`, `setup_k768_ok`, `setup_k896_ok`, `setup_k1024_ok` | 8192 | 256, 384, 640, 768, 896, 1024 | `bench(K)` with `--K_range 128 1024 --K_step 128` |

`satisfiable` uses the `K = 512` instance with the all-zero memory.

## Tolerances

The fine bound is `E_T`, the error bound of the inner loop's `T = steps K = cdiv(K, 64)` steps
against the exact partial sums. With `p e = va e · vb e` for `e < K` and `0` from `K` on (`prodV`:
the lanes past the tensors' last column load the padding `+0`), `l e` its lifted magnitude
`liftMag(va e)·liftMag(vb e)` (`liftV`; `liftMag q` is `|q|` for zero and normal values, `2^-14`
for a nonzero subnormal one), `S_t = Σ_{e < 64t} p e` (`partSum`) and
`L_t = Σ_{64t ≤ e < 64t+64} l e` (`stepMag`):

`E_0 = 0`, `E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`,

that is `E_T = Σ_{t<T} (1 + ρ)^(T−1−t)·(ρ·(|S_t| + L_t) + α)`, with the library's constants for one
fp16 `tl.dot` of `K = 64`: `ρ = relDot16 64 = 66·2^-22`, `α = absDot16 64 = 66·2^-126`.

**Derivation** (library facts only). For a tile, the inner loop (lines 438–442) starts from
`accumulator = 0` (line 437) and its step `t` (`ki = t`) is `acc ← tl.dot(a, b.T, acc)`, a
`float16` dot of `64` products with the accumulator as its own (no separate addition): row `i` of
`a`'s columns `64t … 64t + 63` (`tl.desc_load` of `aDesc` at `[pid_m·128, 64t]`) and row `j` of
`b`'s columns `64t … 64t + 63` (`bDesc` at `[pid_n·256, 64t]`, transposed), the lanes `≥ K` holding
the padding `+0` (and the lanes of rows `≥ M` or `≥ N` too, which reach only outputs the stores
drop). By `Rules.Num.DotF16` (`DotLiftAcc` with `K = 64`) each output element of it is within
`relDot16 64·(|acc| + L_t) + absDot16 64` of `acc + c_t`, `c_t = S_{t+1} − S_t` the step's exact
sum (the padded lanes add nothing to `c_t` or `L_t`). If `acc` is within `E_t` of `S_t` then
`|acc| ≤ |S_t| + E_t`, so the new accumulator is within `E_t + ρ·(|S_t| + E_t + L_t) + α = E_{t+1}`
of `S_{t+1}`; by induction the loop leaves a `float32` value within `E_T` of `S`. The fact's room
condition holds at every step: finite `float16` products are below `2^32`, so with `K ≤ 2^13` every
`|acc| + L_t` and its bound stay far inside the `float32` range. The epilogue's reshape, permute and
split (lines 454–456) move element `(m, n)` of the accumulator to `acc0[m, n]` for `n < 128` and to
`acc1[m, n − 128]` otherwise, unchanged, and the two stores (lines 458, 460) put them at columns
`pid_n·256 + n`; each `.to(tl.float16)` (lines 457, 459) is one rounding (`FloatV.cvt`), which
`Post` states exactly. The tile loop only chooses which program
computes which tile (`tile_id_c` equals `tile_id` on each iteration, so the stores go to the tile
the loads read), and each tile's computation is the same, so the bound holds for every element
whatever `NUM_SMS`.

**Relaxation: `E_T ≤ tol`.** From the closed form, `E_T ≤ (1 + ρ)^T·(ρ·Σ_{t<T}(|S_t| + L_t) + T·α)
= (1 + ρ)^T·(ρ·pathMag + T·α)`. On a covered instance `K ≤ 2^13`, so `T ≤ 128` and
`(1 + ρ)^128 ≤ 1.0021` (`ρ = 1.574·10^-5`). So `(1 + ρ)^T·ρ ≤ 1.0021·66·2^-22 = 66.14·2^-22 ≤
128·2^-22 = 2^-15`, and `(1 + ρ)^T·T·α ≤ 1.0021·128·66·2^-126 ≤ 8466·2^-126 ≤ 2^14·2^-126 = 2^-112`:
`E_T ≤ 2^-15·pathMag + 2^-112 = tol`.

**Meaningful.** The bound follows the exact partial sums, so on zero-mean data it is far below
`Σ|a·b|`. It is `1.94×` the fine bound `E_T` on every element of the setup's launches (`E_T` is
`(1 + ρ)^T·ρ·pathMag` up to `0.2%`). On the setup's `torch.randn` inputs (`M = N = 8192`, `K = 128`
to `1024`), `tol/|S|` has median `8.3·10^-4` and 90th percentile `4.2·10^-3` (`E_T/|S|`:
`4.3·10^-4`, `2.2·10^-3`); per `K`, about `3.5·10^-4` (median) at `K = 128` and `1.2·10^-3` at
`K = 1024`. Its worst, `189`, is at an element whose sum cancels (`|S|` near `0`). `check_gpu.py`
measures, as the error, the distance from `S` to the reals that round to the output, at most
`0.0042` of the tolerance (the device's `float32` value is not observed; its rounding to `float16`
covers `S` on most elements), and it checks that four wrong results break `Post` at
`K = 32, 128, 512, 1024` and on partial tiles: the exact result rounded to `bfloat16` (largest
ratio `8` to `62`), a result missing the last `K` block's products (`1500` and more), two row
blocks of 128 swapped (a wrong tile mapping), and the two `128`-column halves of each tile swapped
(a wrong epilogue; both `6900` and more).

## Exclusions

Launches `SETUP.md` names that the claim leaves out:

- **Out of scope**: dimensions that are not multiples of 128 or exceed 2^17, contexts longer than
  2^22 tokens, and batches of more than 8192 requests: the claim covers real model shapes only.
  Here: `N` or `K` not a multiple of `128` or above `2^17` (`N_dim`, `K_dim`; the test's
  `validate(32, 32, 32)` among them), and `M > 2^22` (`M_ctx`). The claim has no batch-size field.

Conditions that exclude launches outside `SETUP.md`'s launches:

- `matmul_tma_persistent(a, b, True)` (outside real use): the file never runs it on the H200
  (lines 693 and 733); the model has `warp_specialize = False`.
- `M`, `N` or `K` equal to `0`, a row stride not a multiple of `8`, a base not 16-byte aligned
  (`pos`, `ld8`, `base_al`; outside real use): the host's `TensorDescriptor` rejects them on the
  host (assertions in its constructor), and no program runs.
- `K > 2^13` (outside real use: the setup's largest `K` is `1024`): the tolerance grows with `K`
  and the room argument of Tolerances is stated for `K ≤ 2^13`. The kernel is believed correct
  there.
- `NUM_SMS = 0` or `NUM_SMS ≥ 2^31` (`sms`; outside real use: the host passes the device's
  multiprocessor count, `132`).
- `float8e4nv` operands (`--prec fp8`): not part of the setup, a scope choice of this `float16`
  claim (a different instantiation of the kernel).
- Other block configurations and the file's other kernels: not part of the setup.

Kernel limits (the kernel is wrong there under the library's semantics; real use does not reach
them):

- `int32` wrap of `tl.cdiv` (`M_le`, `N_le`): with `int32` arguments, `M + 127` (line 421) wraps
  from `M = 2^31 − 127` on, and `N + 255` (line 422) from `N = 2^31 − 255` on; the first launch that
  hits it is `matmul_tma_persistent` with `M = 2^31 − 127` rows (`N = K = 8`: a `32 GiB` `a` and
  `c`). The setup's largest `M` and `N` are `8192`.
- `tiles + NUM_SMS > 2^31` (`tiles_le`): `num_tiles` (line 424) or the tile loop's bound wraps;
  with `M_le` and `N_le` this needs `cdiv(M, 128)·cdiv(N, 256)` near `2^31`, about `2^46` elements
  of `c`. Not reached.
- An output whose rows overlap in storage (`N_ldc`), or that overlaps `a` or `b` (`c_a`, `c_b`):
  two stores to one element, or a read of what another program writes. Not reached:
  `matmul_tma_persistent` allocates a fresh contiguous `c`.

## Library requests

None.

## Checks

1. **Builds**: `lake env lean ../claims/tma_persistent_matmul_f16/Claim.lean`: no errors or
   warnings, no `sorry`, no `axiom`; `consistent`, `satisfiable` and every
   `setup_*_ok` depend only on `propext`, `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (the `K = 512` instance, the
   all-zero memory).
3. **Coverage**: 8 `setup_*_ok` (`M = N = 8192`, `K = 128, 256, …, 1024`; `NUM_SMS = 132`).
4. **GPU** (`check_gpu.py`, H200 NVL, Triton 3.8.0, `NUM_SMS = 132`, under 2 GB of device memory):
   `matmul_tma_persistent(a, b, False)` on the nine sizes of `SETUP.md` with the file's draws and
   layout (`M = N = K = 32` and the launches with `N` or `K` not a multiple of `128` below are
   outside `Inst.ok`, run as extras);
   edges (`K = 8192`; `M = 1`, `N = K = 8`; partial tiles `M = N = K = 1000`; `N = 96`, each tile's
   second half wholly past `N`; `M = 129`, `N = 136`, `K = 72`; inputs scaled by 8 and by `2^-12`,
   the last with subnormal operands; NaN and infinities in `a` and `b`, the other elements checked);
   four direct launches with `NUM_SMS = 1, 7, 132, 1000` (grids `1`, `7`, `132`, `40`), `a` and `c`
   with padded rows, `c` in a guarded buffer. `OutSpec` checked as: some real `v` within `E` of
   `S` rounds to `c` in `float16` (the error is the distance from `S` to the reals rounding to
   `c`; the check does not snap `v` to the `float32` grid, whose spacing is far below every `E`
   here): largest error / tolerance **0.0042** (`K = 1024`); `0.0035–0.0042` on the setup's
   `8192` sizes, `0.0019` with subnormal operands. Negative tests at `K = 32, 128, 512, 1024` and on `M = N = K = 1000`: a
   `bfloat16`-rounded result, a missing `K` block, two swapped row blocks and swapped tile halves
   each break `Post` (Tolerances). Inputs unchanged; no byte outside the matrix `c` written. PASS.
