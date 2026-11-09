# Prepared claim: `scaled_mm_int8`

The kernel: `source/triton_scaled_mm.py`, vLLM's `scaled_mm_kernel` (lines 55–206), on the `int8`
path (`ACCUMULATOR_DTYPE = tl.int32`, `USE_TD = False`) with the tile the host's heuristic picks
from `M` and `N` (`SETUP.md`, The config), launched by the host `triton_scaled_mm` (lines 211–302).
It computes `C = (scale_a · scale_bᵀ) ⊙ (A @ B) (+ bias)` for `int8` `A` (`M × K`) and `B` (`K × N`),
`float32` scales per token or per tensor (`scale_a`) and per channel or per tensor (`scale_b`), an
optional bias, into a `bfloat16` or `float16` `C`. The claim is `Claims.ScaledMmInt8.Statement` in
`Claim.lean`:

```lean
def Statement : Prop := ∀ K P, Assumptions K P → Spec K
```

`Claim.lean` proves `consistent`, `satisfiable` and the coverage facts `setup_*_ok`; it does not
prove `Statement`. The model map is `model-map.json` (56 entries, every code line of the kernel lines
`[43, 44], [55, 206]`). The kernel reads two module-level names, `triton` (its decorator) and `tl`,
bound by the imports of lines 43–44, which are kernel lines and are mapped. Every other module-level
line is a host line: `import torch` (41), `is_weak_contiguous` (47–52), `triton_scaled_mm`
(211–302), `reference` (305–313) and the test (316–356).

## Specification

`Post d init fin` (`Claim.lean`, `OutSpec`, `Post`) says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `C` (element
  `i·stride_cm + j·stride_cn` of `C`'s storage) can be read in the final memory, with some value `c`
  of the output's dtype. When, moreover, in the initial memory, row `i` of `A` and column `j` of `B`
  hold the `int8` values `va k`, `vb k` (`k < K`; every `int8` is a value), the scale of row `i`
  (`sa`: `scale_a[i, 0]` per token, `scale_a[0, 0]` per tensor) and of column `j` (`sb`) are finite
  `float32`s, the bias of column `j` is a finite `β` (`β = 0` without a bias), and `float32` has room
  (`Room32`, below): with `S = Σ_k va k · vb k` (an exact integer), `R = sa·sb·S + β` and
  `E = tol hasBias sa sb S β` (Tolerances),
  - `c` is the rounding to nearest even (up to NaN payload) of a finite `float32` value `v` with
    `|v − R| ≤ E` (`OutIs`);
  - when `|R| + E` is at most the output format's largest value, `c` is finite and
    `|c − R| ≤ E + (|R| + E)·u + η`, with `u = 2^-8`, `η = 2^-134` for `bfloat16` and
    `u = 2^-11`, `η = 2^-25` for `float16` (`outU`, `outEta`: one more rounding).

  With a non-finite scale or bias, or without room, it says only that the element holds some value
  of the output's dtype.
- **Nothing else changes.** Every byte outside the elements of the matrix `C` is the same in `fin`
  as in `init`: `A`, `B`, the scales, the bias, and the elements of `C`'s storage outside the matrix.

The reference `R` is computed from the host's tensors: `A[i, k]`, `B[k, j]` by their strides,
`scale_a[i, 0]` and `scale_b[j, 0]` by theirs (`saIdx`, `sbIdx`), `bias[j]` (`BiasIs`), exactly as
vLLM's test computes its reference (`torch_scaled_mm`,
`tests/kernels/quantization/test_triton_scaled_mm.py:24-37`, in exact arithmetic). It pins each
output within about one rounding of the output format: the `int32` accumulation is exact, so the
tolerance is the `float32` scale products, the bias addition and the output's rounding only.

`Spec K := Claim.Meets Inst.ok Pre Post K`: every launch on a covered instance and input ends
normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

`Inst` (one launch): `M`, `N`, `K`, the six strides of `A`, `B`, `C`, the tile `BM`, `BN`, `BK`, the
flags `saVec` (`BLOCK_SIZE_SCALE_A = BLOCK_SIZE_M`), `sbVec`, `hasBias`, the output dtype `od`, the
scales' and the bias's strides, and six buffers (`a`, `b` `int8`; `sa`, `sb` `float32`; `c`, `bias`
of `od`).

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `od_ok` | `od` is `bfloat16` or `float16` | the activations' dtype of the W8A8 linear (`out_dtype = x.dtype`) |
| `tile_ok` | `(BM, BN, BK)` is `heurTile M N` or `(32, 32, 32)` | the host's tile (lines 249–263): the heuristic, or the defaults with `use_heuristic=False` |
| `M_pos`, `N_pos`, `K_pos` | positive | the host's assertion (line 226) |
| `K_div` | `K % BK = 0` | **kernel limit**: the masked loads of lines 161 and 164 have no `other`, so a partial last `K` block feeds unspecified values into every output's `tl.dot` (Exclusions) |
| `K_lt` | `K < 2^17` | **kernel limit**: with `|a|, |b| ≤ 128`, `K·2^14 < 2^31` keeps every partial sum of the `int32` accumulator in range (Exclusions) |
| `M_le`, `N_le` | `M + BM`, `N + BN ≤ 2^31` | `int32` arguments; `tl.cdiv(N, BN)` adds `BN − 1` in `int32` (line 83); `pid_m·BM` (lines 96, 111, 199) |
| `grid_lt` | `cdiv(M, BM)·cdiv(N, BN) < 2^31` | `Launch.Valid` |
| `strides` | each stride `< 2^31` | `int32` arguments |
| `steps` | `BK·stride_ak`, `BK·stride_bk < 2^31` | the `int32` products of lines 170–171 |
| `c_inj` | distinct elements of `C` are distinct storage elements | otherwise two lanes or programs store to one element |
| `a_len`, `b_len`, `c_len`, `sa_len`, `sb_len`, `bias_len` | the buffers hold the matrices, the `M` (or `1`) and `N` (or `1`) scales, the `N` bias values | the kernel accesses them |
| `scales_contig` | a vector scale and the bias have `stride(0) = 1` | **kernel limit**: the kernel reads row `i`'s scale at element `i` of the storage (lines 109–125, 189–192), with no stride |
| `bias_nz` | the bias's address is not `0` | line 188 tests the pointer (a CUDA tensor's data pointer is never null) |
| `c_a`, `c_b`, `c_sa`, `c_sb`, `c_bias` | the matrix `C` shares no byte with what the kernel reads | the host's `C` is fresh |

`Pre`: the buffers the kernel accesses are valid (`Buf.Valid`); any values in them.

The integer arguments `M`, `N`, `K` and the six strides are `int32` (covered values below `2^31`); at
`1` Triton passes the compile-time constant `1`, which gives the same values at every use (the
docstring of `model`): in particular `M = 1` (a decode step) and `stride_ak = stride_bk = stride_cn
= 1`, the host's layouts.

**Part (A) of `Compiler.Faithful`'s scope** (docs/rules/compiler-faithful.md, "Scope"): the launch of
lines 276–300 is a direct JIT launch of the `@triton.jit` kernel (no `@triton.autotune`, no
`@triton.heuristics`) with its grid function, the constexprs as keywords, and no launch option
(point 4: `num_warps`, `num_stages` at their defaults; no `debug`, `num_ctas`, `launch_pdl` or
`enable_fp_fusion`, so `debug = false`, `num_ctas = 1`, `enable_fp_fusion = true`); no
`warmup`/`triton.compile` (6); the kernel builds no descriptor on a covered launch (`USE_TD = False`:
the `tl.make_tensor_descriptor` calls are not visited, so no allocator is involved, 5) and is passed
no host `TensorDescriptor` (8). **Part (B)**, the
environment, is assumed: the claim relies on it, and TVA does not check it.

**The supported subset** (docs/SEMANTICS.md). Every form of the covered path is the subset's,
among them: `ACCUMULATOR_DTYPE`, a `tl.constexpr` parameter whose argument is the dtype `tl.int32`
("A dtype as a `tl.constexpr` argument"), used as `tl.zeros`'s `dtype=` (line 89) and `tl.dot`'s
`out_dtype=` (line 167), where the model writes `.i32`; `c_ptr.type.element_ty` ("Values": a pointer
argument's `.type.element_ty` is its buffer's dtype, `od`); the method `x.broadcast_to((m, 1))`
(lines 179, 184; `tl.broadcast_to`, the model's `Tensor.broadcastTo`, accepted for a `[1, 1]` or
`[m, 1]` block, as on every covered launch); and `if bias_ptr:` ("A runtime `if`": a pointer argument
is a runtime condition, `p.addr ≠ 0`; `None` a compile-time one).

## Assumptions

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)` (`[rule]`);
- `dot : Rules.Num.DotI8 P` (`[rule]`; docs/rules/num.md, "Facts the library states"): the `int8`
  `tl.dot` into `int32` (line 167, any `precision`, `maxImpreciseAcc = 0`) is exact, `acc + Σ a·b`,
  when `|acc| + Σ|a·b| ≤ 2^31 − 1`;
- `dot_total : Rules.Num.DotTotal P {a := .i8, b := .i8, acc := .i32, precision := .tf32,
  maxImpreciseAcc := 0}` (`[rule]`): the dot always has a result.

No `[kernel]` assumption. The other operations (`int32`/`int64` arithmetic, `tl.itof`, the `float32`
products, the contractible addition, `tl.convert` to and from the output dtype) are exact or IEEE in
the library and need no fact. `consistent` uses `Platform.sm90ref` (`Platform.sm90ref.dotI8`).

## Coverage

`hostInst M N K od hasBias saVec sbVec bT heur` is the host's instance (contiguous `A`; `B`
contiguous or, with `bT`, the transposed view of a contiguous `[N, K]` weight; contiguous scales and
bias; the fresh contiguous `C`; the tile of `tileOf heur M N`); `hostInst_ok` proves it covered when
`0 < M, N ≤ 2^20`, `0 < K < 2^17` and `K % BK = 0`.

| Theorem | Launches | Source |
|---|---|---|
| `setup_llama_ok` | every `1 ≤ M ≤ 16384`, every `(K, N)` of `llamaShapes` (87 shapes, `K` a multiple of 256), either output dtype, every flag, either layout of `B` | SETUP: Llama-class layers at TP 1–8, the int8 GEMM benchmark's batches |
| `setup_llama128_ok` | `64 < M ≤ 16384`, the 4 shapes of `llamaShapes128` (`K` a multiple of 128 only) | SETUP: Llama-2-7B down TP2, Llama-2-13B down TP4 and attention output TP8, TinyLlama down TP4 |
| `setup_vllm_test_ok` | `(64, 971, 1024)`, `(512, 20486, 1024)`, every flag, either dtype and layout | `test_triton_scaled_mm.py:66-123` |
| `setup_vllm_plain_ok` | `(1, 4096, 4096)`, `(64, 4096, 4096)`, `(256, 2048, 4096)` | `test_triton_scaled_mm.py:150-178` (`use_td=False`) |
| `setup_bias_rounding_ok` | `(1, 16, 32)`, tile `(32, 32, 32)`, per-tensor scales, a bias | `test_triton_scaled_mm.py:126-142` (`use_heuristic=False`) |

`satisfiable` uses `setup_bias_rounding` with the all-zero memory.

## Tolerances

`tol hasBias sa sb S β` (`Claim.lean`), with `u = 2^-24`, `η = 2^-150` (`float32`):

- `e1 = |S|·u + η`: `accumulator.to(tl.float32)` (line 180) of the exact integer `S` (`tl.itof`,
  `ItofRel`: one rounding of `S`, directly or through the exact `float64`; `FloatV.ofRat_err`,
  `cvt_err`);
- `e2 = |sa|·e1 + (|sa·S| + |sa|·e1)·u + η`: the product by `scale_a` (line 180, `FloatV.mul_err`;
  `|sa·x| ≤ |sa|·(|S| + e1)`);
- `e3 = |sb|·e2 + (|sa·sb·S| + |sb|·e2)·u + η`: the product by `scale_b` (line 185);
- without a bias, `tol = e3`; with one, `tol = e3 + 3u·(|sa·sb·S| + e3 + |β|) + 4η`: the addition
  of line 193, contractible with line 185's product (`tl.addc`, `FloatV.AddRel.err`:
  `3u·(|x| + |β|) + 4η` with `|x| ≤ |sa·sb·S| + e3`; it covers the IEEE sum and the `fma`);
- the output's rounding (line 196, `tl.convert` to `od`, `FloatV.cvt_err`): `(|R| + tol)·u_o + η_o`
  in the second clause.

`Room32` is what these steps need: `|sa|·(|S| + e1)` and `|sb|·(|sa·S| + e2)` at most `float32`'s
largest value, and `|sa·sb·S| + e3 + |β| ≤ maxVal/2 − 2η` (`AddRel.err`'s room). The `int32`
accumulator is exact by `dot` (`DotI8`) at every trip: on a covered instance the accumulator is the exact
partial sum and `|acc| + Σ|a·b| ≤ K·2^14 < 2^31 − 1` (`K_lt`), so `DotI8`'s room condition holds,
and the masked lanes are only rows `≥ M` or columns `≥ N`, which no stored element reads (`K_div`).

**Meaningful.** `tol` is about `3·2^-24` of `|sa·sb·S| + |β|`, so the bound is the output's rounding
plus a few `float32` roundings: about `2^-8·|R|` for `bfloat16` and `2^-11·|R|` for `float16`, the
device's own rounding error. `check_gpu.py` measures the device's error at up to `0.9993` of the
bound (rounding to nearest reaches it near the bottom of a binade), and wrong results break it: the
scaled product rounded to `bfloat16` before the bias is added (vLLM's rounding test data: ratio
`1.74`), the last `BLOCK_SIZE_K` products left out, the next row's scale, the bias left out (ratios
`10^6` and more). The bound is provable: it composes the facts above, each step's error by the
library's lemma for its operation.

## Exclusions

Kernel limits (the kernel is wrong there under the library's semantics):

- **`K` not a multiple of the tile's `BLOCK_SIZE_K`** (`K_div`). The last trip of the `K` loop loads
  `A` and `B` with the lanes `k ≥ K` masked and no `other` (lines 161, 164), so those lanes hold
  unspecified values (SEMANTICS.md, `tl.load`), and their products enter the `tl.dot` of every
  output. The first launch that hits it: vLLM's test `(M, N, K) = (1, 256, 128)` (tile
  `(64, 64, 256)`). Real use reaches it: vLLM's kernel tests `(1, 256, 128)`, `(33, 256, 496)`,
  `(64, 20486, 128)`, `(512, 256, 496)`, the rounding test with the heuristic (`K = 32`, tile
  `BLOCK_SIZE_K = 256`), Llama-2-7B's down projection at TP2 for `M ≤ 64` (`K = 5504`) and at TP4,
  TP8 (`K = 2752`, `1376`), Llama-2-13B's at TP4 for `M ≤ 64` (`K = 3456`) and at TP8 (`K = 1728`),
  Llama-2-13B's attention output at TP8 for `M ≤ 64` (`K = 640`), TinyLlama's down projection at TP4
  for `M ≤ 64` (`K = 1408`) and at TP8 (`K = 704`). No Llama-3 or Mistral layer at TP 1–8 reaches it. On the
  device, Triton 3.8.0 happens to load `0` on those lanes, and `check_gpu.py` sees the three kernel
  tests it runs match the reference (outside the claim).
- **`K ≥ 2^17`** (`K_lt`). The `int32` accumulator can wrap: the first launch that hits it is
  `K = 2^17` with every value of `A` and `B` equal to `−128` (`S = 2^31`). Real use does not reach it
  (the largest `K` is `53248`, Llama-3.1-405B's down projection).
- **A non-contiguous vector scale or bias** (`scales_contig`): the kernel reads row `i`'s scale and
  column `j`'s scale and bias at element `i` or `j` of the storage, so a strided view (for example a
  `scale_b` of shape `[N, 1]` sliced from a wider tensor, strides `(2, 1)`) gives other elements; the
  host checks only the shapes. Real use does not reach it: `scaled_int8_quant` returns a fresh
  `[M, 1]` scale and the weight scales and the bias are parameters.
- **`int32` limits** (`M_le`, `N_le`, `grid_lt`, `strides`, `steps`): `M` or `N` within a block of
  `2^31`, `2^31` programs or more, a stride of `2^31` or more, or a contiguous `B` with
  `N ≥ 2^31 / BLOCK_SIZE_K` (line 171's `int32` product): `tl.cdiv` and the products wrap. The first
  launch that hits one through the host is `N = 2^23` with a contiguous `B` (`BLOCK_SIZE_K·N = 2^31`).
  Real use does not reach them (`N ≤ 106496`, `M ≤ 16384`).
- **A bias at address `0`** (`bias_nz`): line 188 would skip the bias. Not reached (a CUDA tensor's
  data pointer is not null).

Outside real use:

- non-finite scales or bias values, or values without room in `float32` (`Room32`) or in the output
  format: `Post` then says only that each element holds some value (quantisation scales are finite
  and small);
- an output dtype other than `bfloat16` or `float16` (`od_ok`: the W8A8 linear passes the
  activations' dtype; `cutlass_scaled_mm` asserts one of the two), and explicit block sizes with
  `use_heuristic=False` other than the defaults (no caller passes them).

Not part of the setup: the tensor-descriptor path (`use_td=True`, XPU's default), the `fp8` path of
the same kernel, and other kernels of the W8A8 linear (Cutlass, AITER).

## Library requests

None.

## Checks

1. **Builds**: `lake env lean ../claims/scaled_mm_int8/Claim.lean`: no errors or warnings, no `sorry`,
   no `axiom`; `consistent`, `satisfiable` and every `setup_*_ok` depend only on `propext`,
   `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (above).
3. **Coverage**: five `setup_*_ok` theorems, each over a family of launches (table above).
4. **GPU** (`check_gpu.py`, H200 NVL, Triton 3.8.0): 94 host launches and two guarded direct
   launches (`SETUP.md`, Checked on the H200). Both clauses of `OutSpec` checked (the first as: some
   real `v` within `E` of `R` rounds to `c`); largest error / tolerance **0.9993** (margin 0.0007,
   Llama-3.1-405B's down projection at TP8, `float16`); four wrong results each break `Post`; inputs
   unchanged; no byte outside the matrix `C` written. PASS.
