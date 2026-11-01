# block_scaled_matmul_mxfp8

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/10-block-scaled-matmul.py`](source/10-block-scaled-matmul.py). Lean: [`lean/`](lean/), the
module `BlockScaledMxfp8` (namespace `Kernels.BlockScaledMxfp8`) and its parts
`BlockScaledMxfp8.Defs` (model, conditions, specification, assumptions), `BlockScaledMxfp8.Basic`
(`int32` arithmetic, descriptor lanes, the scales' reshape and permute, the dot's operands, the
tolerance, one accumulation step, the last rounding) and `BlockScaledMxfp8.Program` (the schedule,
the footprints, one program).

The kernel is `block_scaled_matmul_kernel` (source lines 174–240), unchanged from tutorial 10
(SETUP.md). It reads two module-level names, `triton` (its decorator) and `tl`, bound by the
imports of lines 123–124, which are kernel lines too and are mapped (dropped: they add no step).
It is a tiled GEMM through host TMA descriptors: a 1-D grid of `cdiv(M, 128)·cdiv(N, 256)`
programs; program `p` computes the `128 × 256` tile of `c` at row block `p % cdiv(M, 128)` and
column block `p // cdiv(M, 128)` (lines 204–209), looping over `K` in steps of `128` (line 219):
each step loads a `128 × 128` block of `a`, a `256 × 128` block of `b`, and the matching 5-D
blocks of their `e8m0` scales, rearranges the scales to `128 × 4` and `256 × 4` (lines 226–227),
and accumulates `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)` into a
`float32` accumulator (line 234); the tile is rounded to `float16` and stored (line 240).

One host function launches it, `block_scaled_matmul` (lines 243–281), with the tensors and config
`initialize_block_scaled(M, N, K, "mxfp8")` builds (lines 330–449): `a` (`M × K`) and `b`
(`N × K`) of `torch.float8_e4m3fn`, `e8m0` scales (`MXScaleTensor(…).data`, `uint8`) of shape
`[M // 128, K // 128, 32, 16]` and `[N // 128, K // 128, 32, 16]` viewed as
`[1, M // 128, K // 128, 2, 256]`, a fresh `float16` output `c` (`M × N`), and the constexprs
`BLOCK_M = 128`, `BLOCK_N = 256`, `BLOCK_K = 128`, `VEC_SIZE = 32`, `ELEM_PER_BYTE_A =
ELEM_PER_BYTE_B = 1`, `rep_m = 1`, `rep_n = 2`, `rep_k = 1`, `output_type = 1`, `NUM_STAGES = 4`,
`disallow_acc_multi_buffer = True` (SETUP.md). The test calls it once,
`validate_block_scaled(8192, 8192, 8192, "mxfp8")` (lines 452–469, 749); the benchmark
`bench_block_scaled(K, block_scale_type="mxfp8")` calls it at `M = N = 8192` with `K` a multiple of
`128` (lines 472–502, 755–760: default `K = 512`, the module docstring's `mxfp8` command runs `K`
from `8192` to `16384`), but only after `proton.start(…, hook="triton")` (line 756) has installed
Triton launch hooks, so the claim does not cover the benchmark's launches as the script makes them;
it covers the test's launch, and launches of the benchmark's sizes made with no hook installed
(Conditions).

Every other function and module-level line is a host line: the imports of `argparse`, `torch`,
`triton.profiler`, `TensorDescriptor`, `MXFP4Tensor`, `MXScaleTensor` (120, 122, 125–127),
`is_cuda`, `is_hip_cdna4`, `supports_block_scaling`, `is_rubin` (130–144), the cuBLAS setup
(147–152), `_matmul_launch_metadata` (155–171, the host function the decorator names, which Triton
calls only to hand metadata to an installed launch-enter hook),
`block_scaled_matmul`, `cublas_block_scaled_matmul` (284–327), `initialize_block_scaled`, the test,
the benchmark, `show_profile` (505–512), the AMD helpers (596–727) and `__main__` (730–765). The
file's other `@triton.jit` function, `block_scaled_matmul_kernel_cdna4` (the AMD kernel), is out of
scope ([exclusions.md](../../docs/exclusions.md): targets other than the H200): it is in
`source.unused`, and its header (lines 515–521) is a host line (it binds a name the launch does not
read).

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `c` (element `i·N + j` of
  `c`'s storage) can be read in the final memory, with some value `c` (`OutSpec`). Let row `i` of
  `a` hold `xa k` (element `i·K + k`), row `j` of `b` hold `xb k` (element `j·K + k`), and let
  their scale bytes be `sa k`, `sb k`, the bytes at `sIdx K i k` of `a_scale` and `sIdx K j k` of
  `b_scale` (`k < K`), where

  `sIdx K r k = (r / 128·(K / 128) + k / 128)·512 + 16·(r % 32) + 4·((r % 128) / 32) + (k % 128) / 32`

  is the place of the scale of element `(r, k)` in the `[R / 128, K / 128, 32, 16]` layout, exactly
  as the host's own reference unpacks it (`unpack_scale`, lines 411–417: `reshape(…, 32, 4, 4)`,
  `permute(0, 3, 2, 1, 4)`, `reshape(R, K / 32)`, `repeat_interleave(32)`). The element's value is
  `scaled x s = x · 2^(s − 127)` (the `e8m0` value of the byte `s`), and the products are
  `p k = scaled (xa k) (sa k) · scaled (xb k) (sb k)` (`prodV`), with the exact dot product
  `S = Σ_{k<K} p k` (`dotSum`). When, in the initial memory,
  - every element is in the domain of the library's fact (`InDom`): `xa k`, `xb k` finite (not
    NaN), each scale byte in `1 … 254` (`Rules.Num.e8m0Dom`: not `0x00`, not the NaN `0xFF`), each
    scaled value `0` or of a normal `bfloat16` magnitude (`Rules.Num.scaledDomBF16`:
    `2^-126 ≤ |·| ≤ bfloat16.maxVal`), and
  - there is room: `Σ_k |p k| + tol ≤ 2^127` (`dotAbs`),

  then `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
  `float32` value `v` with `|v − S| ≤ tol`, and when moreover `|S| + tol ≤ 65504` (the largest
  `float16`), `c` is finite and `|c − S| ≤ tol + (|S| + tol)·2^-11 + 2^-25` (one more rounding:
  `binary16.u = 2^-11`, `binary16.eta = 2^-25`).

  **The tolerance** `tol = tol p (K / 128)` is computed from the exact products by the recursion
  `tol p 0 = 0`,
  `tol p (t + 1) = tol p t + ρ·(|partSum p t| + tol p t + chunkAbs p t) + α`, where
  `partSum p t = Σ_{k < 128 t} p k` is the exact partial sum after `t` steps and
  `chunkAbs p t = Σ_{128 t ≤ k < 128 t + 128} |p k|` the magnitude of step `t`'s products. Each step
  adds the library's bound for one `tl.dot_scaled` of `128` products into the accumulator
  (`Rules.Num.DotScaledE4m3`: `rel·(|acc| + Σ|a·b|) + abs`, with `|acc|` bounded by
  `|partSum| + tol`), `ρ = relDot16 128 = 130·2^-22` (about `3.10·10^-5`) and
  `α = absDot16 128 = 130·2^-126`. It is explicit and computable (a rational for any given
  inputs). In closed form (`tol_le_closed`, proved, not used by `main`):
  `tol p T ≤ ((1 + ρ)^T − 1)·Σ_{k<128T} |p k| + T·(1 + ρ)^T·α`, and at the largest covered
  `K = 2^14` (`T = 128`) `(1 + ρ)^128 − 1 ≤ 1/250` (`closed_rel_le`) and
  `128·(1 + ρ)^128·α ≤ 2^-110` (`closed_abs_le`): at every covered `K`, `tol` is at most `0.4%` of
  `Σ|p k|` plus `2^-110`. The recursion is tighter than the closed form when the partial sums
  cancel, as they do on the host's data.

  **Its size on the host's data.** The host's operands are `MXFP4Tensor(…).random()` values cast
  to `e4m3` (exactly): a sign and one of `0, 0.5, 1, 1.5, 2, 3, 4, 6`, uniform; its scale bytes are
  `floor(log2(torch.rand + 1e-8)) + 127`, in `100 … 126` (`2^-1` with probability `1/2`, `2^-2` with
  `1/4`, …). So every host element is in `InDom` (scaled values are `0` or at least `2^-28`, far
  inside `bfloat16`'s normal range; scale bytes in `1 … 254`) and the room condition holds
  (`Σ|p k| ≤ 9·K`), so `OutSpec` gives the bound on every element. The products have
  `E|p| ≈ 0.56` and `E[p²] ≈ 1.50`. Sampling 200 output elements per `K` with these distributions
  (a CPU computation of `tol` by the recursion):

  | `K` | median `|S|` | median `Σ|p|` | median `tol` | `tol / Σ|p|` (max) | median `tol / |S|` | median whole bound `/ |S|` (with the `float16` rounding) |
  |---|---|---|---|---|---|---|
  | `512` (benchmark default) | 17.5 | 279 | 0.0099 | 4.8·10^-5 | 0.052% | 0.10% |
  | `8192` (test) | 77.8 | 4615 | 0.247 | 1.0·10^-4 | 0.30% | 0.35% |
  | `16384` (largest covered) | 101 | 9154 | 0.55 | 1.4·10^-4 | 0.57% | 0.62% |

  At `K = 16384` the bound (about `0.55` plus the `float16` rounding, on outputs of typical
  magnitude `100`) still rejects a kernel that drops, repeats or misplaces one group of `32`
  products or applies one wrong scale (a group contributes about `√(32·1.50) ≈ 6.9` in RMS), that
  reads the scales in another layout or with another bias (`2^(s − 126)` doubles every product), or
  that computes another tile's element; it does not reject a single dropped product of magnitude
  below about `0.5`. It is looser than the test's own check (`assert_close(…, atol=1e-3,
  rtol=1e-3)`, about `0.08` at `K = 8192`) because the library's per-step bound is relative to
  `|acc| + Σ|a·b|`, and on this data `Σ|p|` is `60` to `90` times the typical `|S|` at `K = 8192` to
  `16384` (the products cancel); at `K = 512` and `8192` the
  whole bound is also below a `bfloat16` rounding's largest relative error (`0.39%`) on typical
  outputs.

  Outside the domain (a NaN element, a scale byte `0x00` or `0xFF`, a scaled value below `2^-126`
  or above `bfloat16`'s range), the claim says only that element `(i, j)` of `c` holds some
  `float16` value.
- **Nothing else changes.** Every byte outside the matrix `c` (`cEl d`, its first `M·N` elements)
  is the same in `fin` as in `init`: the inputs, and any element of `c`'s storage past `M·N`.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: the sizes `M`, `N`, `K` and five buffers (`Buf`: a base
address and a length in elements): `a`, `b` of `float8e4nv`, `a_scale`, `b_scale` of `uint8`, `c`
of `float16`. The descriptors are the host's (`aDesc`, `bDesc`, `aScaleDesc`, `bScaleDesc`, `cDesc`,
from `TensorDescriptor.from_tensor` on the contiguous tensors, lines 256, 367–368, 401–404): shape
`[M, K]`, strides `[K, 1]`, block `[128, 128]` for `a`; `[N, K]`, `[K, 1]`, `[256, 128]` for `b`;
`[1, M / 128, K / 128, 2, 256]`, strides `[(M / 128)·(K / 128)·512, (K / 128)·512, 512, 256, 1]`,
block `[1, 1, 1, 2, 256]` for `a_scale` (the same with `N` and block `[1, 2, 1, 2, 256]` for
`b_scale`); `[M, N]`, `[N, 1]`, `[128, 256]` for `c`; padding `"zero"` (the default).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `M_ok` | `0 < M`, `M % 128 = 0`, `M < 2^31` | the scales come in blocks of `128` rows (`M // 128`, line 370; the host's reference needs all of them, lines 416–418); `M` is a descriptor shape entry (`int32`); a zero shape entry is rejected on the host |
| `N_ok` | the same for `N` | line 371; the last column of tiles may be half outside `c` (`BLOCK_N = 256`): those lanes read the padding and are not stored |
| `K_ok` | `0 < K`, `K % 128 = 0`, `K ≤ 2^14` | `K // VEC_SIZE // 4` (line 370) and the benchmark's `assert K % 128 == 0` (line 473); `2^14 = 16384` is the module docstring's largest `mxfp8` `K` and keeps the tolerance meaningful (Narrowings) |
| `grid_lt` | `cdiv(M, 128)·cdiv(N, 256) < 2^31` | the grid's size (`Launch.Valid`) |
| `base_al` | every buffer starts at a 16-byte-aligned address | a descriptor's base (`TensorDesc.Valid`) |
| `a_len`, `b_len`, `as_len`, `bs_len`, `c_len` | `M·K ≤ a.len`, `N·K ≤ b.len`, `(M/128)·(K/128)·512 ≤ a_scale.len`, `(N/128)·(K/128)·512 ≤ b_scale.len`, `M·N ≤ c.len` | every element a lane inside the tensor touches is inside its buffer |
| `c_in` | `c`'s matrix shares no byte with `a`, `b` or the scales | a program reading bytes another writes races |

**Which of the host's launches are covered.** Every launch the script makes for `"mxfp8"` meets
`Inst.ok`: the test `M = N = K = 8192` (`64·32 = 2048` programs), the benchmark `M = N = 8192` with
`K` a positive multiple of `128` up to `16384`. The tensors are contiguous (`a_ref.to(…)`,
`MXScaleTensor(…).data` reshaped as a view, `torch.empty`), fresh PyTorch CUDA allocations used
from their start, so 16-byte aligned, of exactly the lengths above, and `c` (`torch.empty` at
line 244, made while the inputs are alive) overlaps none of them. But meeting `Inst.ok` is not
enough: `Compiler.Faithful` covers a launch only with no Triton hook installed (scope point 3,
part (B); [exclusions.md](../../docs/exclusions.md), the row "a hook"), and the script installs
one before the benchmark. `__main__` calls `proton.start("block_scaled_matmul", hook="triton")`
(line 756) before every `bench_block_scaled` call (line 760); that registers Triton's launch-enter,
launch-exit and kernel-load hooks (`knobs.runtime.*_hook`), the hook `launch_metadata=` (line 174)
is there for. `proton.deactivate()` (lines 501, 757) only pauses the profiling and leaves the hooks
installed; only `proton.finalize()` (line 764) removes them. So every benchmark launch, the
warm-up ones (line 484) and the timed ones (line 492, after `proton.activate()`, line 490), runs
with a hook installed, and the claim does not cover them. The claim covers:
- the test's launch, `validate_block_scaled(8192, 8192, 8192, "mxfp8")` (line 749), made before
  line 756 installs any hook;
- launches of the benchmark's sizes (`M = N = 8192`, `K` a positive multiple of `128` up to
  `2^14`) made with no hook installed, for instance `bench_block_scaled`'s launches without the
  profiling session of lines 755–765.

The conditions are more general: any sizes within them, longer buffers, and inputs overlapping each
other (they are only read).

`Pre d init`: the five buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to the element size, the buffer inside `[0, 2^64)`, every byte present).
Any values in `a`, `b` and the scales, NaN and every scale byte included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). The script's only launch of this kernel is the direct launch
`block_scaled_matmul_kernel[grid](…)` in `block_scaled_matmul` (lines 258–280), called by the test
(line 458) and the benchmark (lines 484, 492; outside the claim as the script makes them, because of
the hook, above and part (B) below). Point by point:

- **4 (options).** The launch passes the constexprs positionally, `NUM_STAGES` positionally and
  `disallow_acc_multi_buffer=` as a keyword: both are kernel parameters (`tl.constexpr`), not
  launch options. No `num_warps`, `num_stages`, `enable_fp_fusion`, `debug`, `num_ctas`,
  `launch_pdl` or other option is given; nothing calls `warmup` or `triton.compile`; there is no
  `@triton.autotune` and no `@triton.heuristics` (SETUP.md); the decorator gives only
  `launch_metadata=` (a host function for a launch-enter hook), not `debug`. So `launch` keeps
  every default:
  `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no descriptor (`tl.make_tensor_descriptor` does not
  occur): every descriptor is a host `TensorDescriptor` argument, so no allocator is needed and the
  script sets none.
- **6 (precompiled kernel).** No `CompiledKernel` is launched.
- **8 (host descriptors).** The five host descriptors come from `TensorDescriptor.from_tensor(…)`
  with no `round_f32_to_tf32` (its default is `False`; none is `float32` anyway).
- **9 (CUDA graphs).** No call is made during a stream capture and nothing is replayed: the test
  and the benchmark call `block_scaled_matmul` directly, and the script captures no graph. (The
  benchmark's profiling installs hooks, which point 9 is not about but point 3 is: below.)

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it. The script's
own benchmark visibly breaks point 3 (no hook installed): its launches run after
`proton.start(…, hook="triton")` (line 756) has installed Triton's launch and kernel-load hooks,
which stay installed until `proton.finalize()` (line 764). So those launches are outside the
rule's scope and the claim does not cover them; the test's launch (line 749, before line 756) is
not affected, and launches of the benchmark's sizes made with no hook installed are covered. Among
part (B),
point 7: the H200, where `supports_block_scaling()` is false and the script's `__main__` stops before
the test (SETUP.md); the claim covers the host functions' `mxfp8` launches as if they ran there,
with `tl.dot_scaled` lowered by Triton's sm_90 emulation (exclusions.md).

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under `@triton.jit(launch_metadata=…)`, not under `@triton.autotune`.
Its parameters are five host tensor descriptors and fifteen `tl.constexpr`s with `int` arguments
(`disallow_acc_multi_buffer` with the `bool` `True`), none with a default. Its statements are
assignments `name = e`, one annotated assignment of a constexpr (line 216), three `name += e`
(lines 236–238), an expression statement (`c_desc.store(…)`), compile-time `if`/`elif`/`else` chains
(lines 197–202, 229–234) and one `for k in tl.range(0, …, num_stages=…,
disallow_acc_multi_buffer=…)`. Its expressions are `tl.program_id(axis=0)`, `tl.cdiv` of
constexprs, `desc.load`, `desc.store`, `tl.zeros(…, dtype=tl.float32)`, `.reshape(…)`,
`.trans(0, 3, 2, 1, 4)`, `.T`, `tl.dot_scaled(…)` with the formats `"e4m3"`, `x.to(dtype)`, the
dtypes `tl.float32`, `tl.float16`, `tl.float8e4nv`, the operators `+ * // %`, `==` and `and` on
constexprs. The only module-level names it reads are `triton` and `tl`, Triton's modules, each
bound only by its own `import` (lines 123–124); no other statement binds `triton`, `tl` or
`block_scaled_matmul_kernel`.

## Assumptions

Three assumptions, all of kind `[rule]`, each cited as the library states it:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved, every outcome of the compiled kernel launched as
  `launch d` is one of the model's. The rule covers a launch only within its scope: part (A), which
  the launches meet (Conditions), and part (B), the environment, which the claim relies on
  ([rule page](../../docs/rules/compiler-faithful.md)).
- `dot` — `Rules.Num.DotScaledE4m3 P`, the library's fact for `tl.dot_scaled` of `"e4m3"` by
  `"e4m3"` with `e8m0` scales, groups of `32`, `K`-packed operands, either `fast_math`
  ([num.md](../../docs/rules/num.md), "`tl.dot_scaled` on sm_90"): each output element whose row
  of `lhs` and column of `rhs` have scaled values in `scaledDomBF16`, scale bytes in `e8m0Dom` and a
  finite accumulator is finite and within `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K` of
  `acc + Σ a·b`, when that leaves room below the `float32` range. The model's dot (line 234) has
  `K = 128`.
- `dot_total` — `Rules.Num.DotScaledTotal P cfg` for the model's configuration (`cfgMx`): the
  `tl.dot_scaled` of line 234 always has a result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 240 is `tl.cvt`,
which the library defines (one rounding to nearest even).

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` by definition of `launch`.
- `hvalid` (`launch_valid`, by `Launch.valid_grid1`): the grid has fewer than `2^31` programs
  (`grid_lt`), and each of the eleven block shapes is a list of powers of two with at most `2^20`
  elements.
- Footprints (`BlockScaledMxfp8.Program`): every program reads within the inputs (`R = inEl`);
  program `p` writes the elements `(pm·128 + i)·N + pn·256 + j` of `c` with `i < 128`, `j < 256`,
  `pn·256 + j < N` (`tileIdx`, `W`), where `pm = p % cdiv(M, 128)`, `pn = p / cdiv(M, 128)`. Two
  programs' tiles are disjoint (`W_disj`, by `c_inj`), and every tile is inside `c`'s matrix
  (`W_sub`), so disjoint from the inputs (`R_W`, from `c_in`).
- `hsafe` (`safe_program`): one rule per line, from `program_id = p` (`launch_pid`).
  - Lines 205–213 (`safe_irem1`, `safe_idiv1`, `IntV.I32.mul_nat`): `pid_m = pm`, `pid_n = pn`,
    `offs_am = 128 pm`, `offs_bn = 256 pn`, `offs_scale_m = pm`, `offs_scale_n = 2 pn` (no `int32`
    wrap: `M`, `N` below `2^31`).
  - Line 219 (`safe_forRange`, `K / 128` trips) with the invariant: memory unchanged, `offs_k_a =
    offs_k_b = 128 t`, `offs_scale_k = t`, and `AccInv t`: every element `(i, j)` of the tile inside
    `c` whose rows and scales are in `OutSpec`'s domain is finite and within `tol p t` of
    `partSum p t`; it holds at `t = 0` (`acc0`: `+0.0`, exact).
  - Lines 221–222 (`safe_desc_load`, `access2`, `lane2`): each access meets the descriptor's
    requirements (positive shape entries below `2^31`, `16`-byte-aligned base, row stride and inner
    offset of multiples of `16` bytes); lane `(i, k)` reads element `(128 pm + i)·K + 128 t + k` of
    `a` (all lanes are inside, `M % 128 = 0`), lane `(j, k)` element `(256 pn + j)·K + 128 t + k` of
    `b` when `256 pn + j < N` (the padding otherwise, never used for a stored element).
  - Lines 223–224 (`safe_desc_load`, `access5`, `lane5`): lane `(0, l1, 0, l3, l4)` of a scale block
    reads byte `(q + l1)·(K/128)·512 + t·512 + 256 l3 + l4` (`q = pm`, resp. `2 pn`), inside the
    scale tensor (`scale_lt`).
  - Lines 226–227 (`scaleA_get`, `scaleB_get`): entry `(i, c)` of the rearranged `128 × 4` block is
    lane `(0, 0, 0, q / 256, q % 256)` of the loaded block with `q = 16 (i % 32) + 4 (i / 32) + c`
    (for `b`, lane `(0, j / 128, 0, …)` with `q = 16 (j % 32) + 4 ((j % 128) / 32) + c`), so entry
    `(i, k / 32)` is the byte at `sIdx K (128 pm + i) (128 t + k)` (`sIdx_a`, `scaleA_val`; `sIdx_b`,
    `scaleB_val`).
  - Line 234 (`safe_dot_scaled`, with `dot_total`; the shape and dtype checks by `decide`), then
    `acc_step`: the dot's operands are the loaded bits (`unpackL_get`, `scalesL_get`,
    `unpackR_get`, `scalesR_get`), their scaled values `scaled x s` (`scaledValue_eq`), and `dot`
    at `K = 128` gives the step: finite, within `ρ·(|acc| + chunkAbs) + α` of `acc + Σ` the step's
    products, which with `|acc| ≤ |partSum t| + tol t` is `AccInv (t + 1)`; the room condition
    (`Σ|p| + tol ≤ 2^127`) keeps every step inside the `float32` range.
  - Lines 236–238 (`IntV.I32.add_nat`): the offsets become `128 (t + 1)`, `t + 1` (no wrap,
    `K ≤ 2^14`).
  - Line 240 (`safe_cvt`, `safe_desc_store`): `c` is the rounding of the accumulator to `float16`,
    up to NaN payload; lanes `(i, j)` with `256 pn + j < N` store to element
    `(128 pm + i)·N + 256 pn + j` of `c`, inside `W p`, distinct lanes to distinct elements
    (`c_inj`). After `K / 128` trips the partial sum is `S` (`partSum_end`), so `OutSpec` follows
    from `AccInv` and the last rounding (`out_f16`, by `FloatV.cvt_err`): `Q`.
- `hdisj`, `hRW`: `W_disj`, `R_W`.
- `hpost`: element `(i, j)` of `c` is in the tile of program `(j / 256)·cdiv(M, 128) + i / 128` at
  lane `(i % 128, j % 256)` (`tile_of`); its `Q` gives `OutSpec` in its memory, and the final memory
  agrees with it on its footprint, so `Mem.read_congr` gives the value. A byte outside `c`'s matrix
  is in no write footprint (`W_sub`), so it keeps its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref` gives a total behaviour that
satisfies `faithful`; `Platform.sm90ref.dotScaledE4m3` gives `dot` and
`Platform.ref.dotScaledTotal` gives `dot_total`.

`satisfiable`: `M = N = K = 128` (`testInst`), the buffers back to back at the 16-byte-aligned
addresses `0`, `16384`, `32768`, `33280`, `33792`, in a memory where every byte is present
(`testMem`), meets `Inst.ok` (`test_ok`) and `Pre` (`test_pre`).

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`mxfp8` only.** `a`, `b` are `float8e4nv` with `e8m0` scales, `ELEM_PER_BYTE_A =
  ELEM_PER_BYTE_B = 1`, `VEC_SIZE = 32`, and the output is `float16` (`output_type = 1`, the only
  output dtype the file asks for). The `mixed`, `mxfp4` and `nvfp4` configurations are other folders
  or rejected on the H200 (SETUP.md).
- **The launch's configuration.** `BLOCK_M = 128`, `BLOCK_N = 256`, `BLOCK_K = 128`, `rep_m = 1`,
  `rep_n = 2`, `rep_k = 1`, the values `initialize_block_scaled` fixes for `"mxfp8"` on the H200
  (SETUP.md); other configurations are not covered.
- **`M`, `N`, `K` multiples of `128`** (`M_ok`, `N_ok`, `K_ok`): the host's scale layout has
  `M // 128`, `N // 128`, `K // 128` blocks and its reference needs them to cover the matrices; the
  benchmark asserts `K % 128 == 0`. Positive sizes: a zero size gives a zero descriptor shape entry,
  which the host rejects.
- **`K ≤ 2^14`.** The tolerance grows with `K` against the typical output (the table above:
  `0.052%`, `0.30%`, `0.57%` of the typical `|S|` at `K = 512`, `8192`, `2^14`); beyond the module
  docstring's largest `16384` the claim is not stated. The proof does not need the bound: it needs
  only the `int32` bounds of the loop and offsets and the room condition, which keeps the
  accumulator inside the `float32` range. It is kept so that the tolerance stays meaningful at
  every covered `K` (the closed form's `0.4%` of `Σ|p|`, `closed_rel_le`, is proved for `K ≤ 2^14`).
- **`int32` bounds** (`M`, `N` below `2^31`, `grid_lt`): the descriptor shape entries are `int32`,
  and the offsets `pid_m * 128`, `pid_n * 256`, `offs_k_a + 128` are computed in `int32`; the host's
  sizes are far inside.
- **Contiguous tensors, 16-byte-aligned bases** (the descriptors' strides and `base_al`): the
  host's tensors are contiguous fresh allocations.
- **No overlap** of `c` with an input (`c_in`). The host allocates `c` fresh.
- **Accuracy in the library fact's domain only.** Elements outside `InDom` (a NaN, a scale byte
  `0x00` or `0xFF`, a scaled value below `2^-126` or above `bfloat16`'s range) and inputs without
  room leave only "some `float16` value". The host's data is always inside (Specification).
- **`block_scaled_matmul_kernel_cdna4`** (lines 515–593) is excluded
  ([exclusions.md](../../docs/exclusions.md)); `cublas_block_scaled_matmul` launches no Triton
  kernel.
- **The benchmark's launches as the script makes them are not covered.** They run after
  `proton.start(…, hook="triton")` (line 756) has installed Triton's launch and kernel-load hooks,
  which `proton.deactivate()` leaves installed until `proton.finalize()` (line 764): outside
  `Compiler.Faithful`'s scope point 3 (part (B), no hook installed; exclusions.md, "a hook"). The
  claim covers the test's launch (line 749, made before any hook) and launches of the benchmark's
  sizes (`M = N = 8192`, `K` a positive multiple of `128` up to `2^14`) when made with no hook
  installed. `Inst.ok` covers those sizes; no condition changes.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions), among it point 7's target (the H200)
  and host, and point 3 (no hook), which the script's benchmark breaks (above).

Modelling decisions:

- The constexprs `output_type`, `ELEM_PER_BYTE_A`, `ELEM_PER_BYTE_B` are no parameters of the
  model: fixed on every covered launch, they are read only by compile-time conditions (lines
  197–202, 216, 229–233, dropped in the map, `config`) and by the Python arithmetic
  `BLOCK_K // ELEM_PER_BYTE_*` (lines 236–237, written as `BLOCK_K`). `output_dtype` is the
  compile-time `tl.float16`, written as `F16` at its use (line 240). `N` is read by no line of the
  body; `NUM_STAGES` and `disallow_acc_multi_buffer` only by the `tl.range` scheduling keywords,
  which change no outcome; they are no parameters either.
- `tl.cdiv` of constexprs (lines 205, 219) is Python's `(x + (B − 1)) // B`, a compile-time integer;
  `num_pid_m`, assigned to a name, is an `int32` scalar, as are the Python `0`s of lines 210–211,
  214 that the loop carries; every constexpr operand of an `int32` operation (`BLOCK_M`, `BLOCK_N`,
  `rep_m`, `rep_n`, `BLOCK_K`, `rep_k`) and the `0`s of the offset lists are built as `int32`
  constants; `%`, `//` of `int32` scalars are `tl.irem`, `tl.idiv`.
- The descriptor arguments are `TensorDesc`s with the host's shapes, strides, block shapes and
  padding; each `desc.load` has its descriptor's block shape.
- `reshape` is `Tensor.reshape` (row-major, `can_reorder=False`), `trans(0, 3, 2, 1, 4)` is
  `Tensor.permute [0, 3, 2, 1, 4]`, `b.T` is `Tensor.trans`; the model takes the element-count
  equalities of the second reshapes as hypotheses `hA`, `hB` (Triton rejects the kernel on the host
  otherwise; `rfl` at the host's values).
- `tl.dot_scaled(…)` is `tl.dot_scaled cfgMx 128 256 128` with both scales present, the defaults
  `fast_math=False`, `lhs_k_pack=rhs_k_pack=True`, `out_dtype=tl.float32`, and the accumulator as
  `acc`; `.to(tl.float16)` is one rounding to nearest even (`tl.cvt`).
- The launch lists the source's block shapes (the accumulator, `b.T`, `c`'s block `[128, 256]`;
  `a`'s `[128, 128]`; `b`'s `[256, 128]`; the scale blocks and their rearrangements
  `[1, 1, 1, 2, 256]`, `[1, 1, 32, 4, 4]`, `[1, 4, 32, 1, 4]`, `[128, 4]`, `[1, 2, 1, 2, 256]`,
  `[2, 1, 32, 4, 4]`, `[2, 4, 32, 1, 4]`, `[256, 4]`); `enable_fp_fusion` is written as Triton's
  default `true`.
