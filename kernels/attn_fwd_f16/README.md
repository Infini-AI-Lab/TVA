# attn_fwd_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/06-fused-attention.py`](source/06-fused-attention.py) (tutorial 06 as set up in
[SETUP.md](SETUP.md): the `@triton.autotune` decorator removed, the config `BLOCK_M=128,
BLOCK_N=64, num_stages=2, num_warps=4` passed by the launch, the config's `pre_hook` called by the
host). Lean: [`lean/`](lean/), the module `AttnFwd` (namespace `Kernels.AttnFwd`) and its parts:

| Module | What it holds |
|---|---|
| `AttnFwd.Model` | the model (`attn_fwd_inner`, `maybe_make_tensor_desc`, `attn_fwd`, `model`, `launch`), the conditions (`Inst`, `Inst.ok`, `Pre`), the specification (`Post`, `Spec`, the tolerances), the assumptions |
| `AttnFwd.Real`, `.Lane`, `.RowMax`, `.RowAcc`, `.Step` | one query row through the loop, in real numbers: the running maximum (`MInv`), `l_i` (`LInv`) and `acc` (`AInv`) |
| `AttnFwd.Final` | the end of a row (lines 241–245: `final_o`, `final_M`) and the tolerances' numeric facts (`tol_facts`) |
| `AttnFwd.Facts` | the grid, `int32` arithmetic, the descriptors (`descOf`), the constants, `qk_scale` (`qk_scale_facts`) |
| `AttnFwd.Row`, `.Tile`, `.Inner` | a row's state (`RowSt`, `row_step_plain`, `row_step_masked`), the tile loads and stores, the loop of `_attn_fwd_inner` and its three calls (`safe_inner1`, `safe_inner2`, `safe_inner3`) |
| `AttnFwd.Program` | one program (`safe_program`) |
| `AttnFwd` | `main`, `consistent`, `satisfiable` |

**The kernel** is `_attn_fwd` (source lines 176–245) with its two `@triton.jit` helpers
`_attn_fwd_inner` (lines 47–110) and `_maybe_make_tensor_desc` (lines 168–173), in the `float16`
run (`FP8_OUTPUT = False`). It reads the module-level names `triton` (the decorators) and `tl`,
bound by the imports of lines 20–21, which are kernel lines and are mapped (dropped: they add no
step), and the two helpers, bound by their `def`s.

**The host.** `_attention.forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True)` (lines
505–580) takes `q`, `k`, `v` of shape `[Z, H, N_CTX, HEAD_DIM]`, asserts one `HEAD_DIM` in
`{16, 32, 64, 128, 256}` (lines 508–512), allocates `o = torch.empty_like(q)` and the `float32`
`M = torch.empty((Z, H, N_CTX))`, and passes `STAGE = 3` when `causal` and `1` otherwise. On the
H200 (`supports_host_descriptor()` and `is_hopper()` are true), when `warp_specialize` is false it
builds host `TensorDescriptor`s over `q`, `k`, `v`, `o` (`shape = [Z·H·N_CTX, HEAD_DIM]`,
`strides = [HEAD_DIM, 1]`, lines 525–536), whose `block_shape` `_host_descriptor_pre_hook` (line
557) sets to `[128, HEAD_DIM]` for `q` and `o` and `[64, HEAD_DIM]` for `k` and `v`; when it is true
it passes the tensors themselves (lines 538–541), and the kernel builds the descriptors (line 173),
with the allocator of line 546. It launches the grid `(cdiv(N_CTX, 128), Z·H, 1)` (lines 548–549)
with `N_CTX`, `HEAD_DIM`, `BLOCK_M=128`, `BLOCK_N=64`, `FP8_OUTPUT=False`, `STAGE`,
`warp_specialize`, `IS_HOPPER=is_hopper()` (true) and `num_stages=2, num_warps=4`
(`extra_kern_args` is empty on the H200: lines 517 and 552 are false).

The script calls it on `float16` inputs in the test (`test_op`, lines 631–695: `Z ∈ {1, 4}`,
`H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, both `causal`,
`warp_specialize = False` off Blackwell, inputs `normal(0, 0.5)`, `sm_scale = 0.5`; the `float16`
forward runs in the `bwd` mode, line 677) and in the benchmark (lines 698–780: its configs and
`bench_flash_attention`, lines 739–775: `Z = 4`, `H = 32`, `N_CTX = 2^10 … 2^14`, `HEAD_DIM ∈ {64, 128}`, both `causal`,
`warp_specialize ∈ {False, True}` for the non-causal forward on Hopper, inputs `torch.randn`,
`sm_scale = 1.3`, the `triton-fp16` line). Its `float8e5` run (the `triton-fp8` provider) is not
this folder's (Narrowings).

**Host lines and unused functions.** Every other module-level line is a host line (`source.host`):
the imports of `pytest`, `torch`, `os` and `TensorDescriptor` (16–18, 22), `DEVICE` (24), `is_hip`,
`is_cuda`, `supports_host_descriptor`, `is_blackwell`, `is_hopper` (27–44, called by the host),
`_host_descriptor_pre_hook` (113–125, called at line 557), the config list (128–146), the class
`_attention` (503–623), `attention`, `TORCH_HAS_FP8`, the test, the benchmark's configs and
function and the `__main__` block (626–780). The functions in `source.unused` run nothing: `keep`
and `prune_invalid_configs` (nothing calls them once the decorator is removed, SETUP.md), and the
backward kernels `_attn_bwd_preprocess`, `_attn_bwd_dkdv`, `_attn_bwd_dq`, `_attn_bwd` (other
folders' kernels; `_attn_fwd` does not call them). Their headers (149, 156, 248–253, 266–278,
319–330, 366–380) are host lines: they bind `keep`, `prune_invalid_configs` and the four backward
names, which the launch does not read.

## What changed (library round 25)

Library round 25 (issue L126) restated `Rules.Num.DotF16` as `DotLiftAcc`: a `tl.dot`'s error is now
relative to `|acc| + Σ liftMag a·liftMag b`, a subnormal `float16` operand counted at `2^-14`, since
the tensor core aligns it there. The claim was re-proved on it:

- **The score's `tl.dot` (line 73).** The score magnitude is now `scoreMag`, the lifted sum
  (formerly `scoreAbs = Σ_e |q·k|`; equal for zero and normal values). It is what `HeadOk`'s bound
  `σ·scoreMag ≤ 2^10` and `Post`'s premise `σ·scoreMag ≤ X` bound. The host's inputs still meet
  them: the lift adds below `0.031` to `σ·Σ_e|q·k|` (Specification).
- **The `p·v` `tl.dot` (line 103).** `p.to(float16)` is subnormal for every weight below `2^-14`, and
  `v` may be. A subnormal `p` adds `66·2^-36·|v|` per key (`omega0`: `2^-25 + 2^-125` before, now
  `2^-25 + 2^-125 + 66·2^-36`). A subnormal `v` adds `66·2^-36·|p|` per key, and `|p|` is at most
  `(1 + psi16)·weight + 2^-25 + 2^-125`: over the keys, at most `kapA = 2^-29` times `l_i`'s exact
  value, which adds about `kapA` to `acc / l_i`, and the rest `absA` per block (`2^-85` before, now
  `2^-37`).
- **The tolerances.** `tolO` and `tolM` are unchanged. `tolS` grows about `3%` (`3.0·10^-8` to
  `3.1·10^-8` at the test's sizes). `tolF` grows from `3.0·10^-8` to `3.2·10^-8` at the test's
  sizes, and to at most `4.2·10^-8` (`N = 2^16`, `X = 2^10`).
- **Review round 3 (F7).** The first re-proof bounded a subnormal `v`'s lift with `p ≤ 2.3` on all
  `64` keys of every block (`5·2^-9` per block, `absA = 2^-22`), so `tolF` grew as `N/64·2^-22`, to
  `3.3·10^-4` at `N = 2^16`: a figure of the analysis, not of the device. The lift is now bounded
  through each block's weights, which `l_i` collects: `AInv`'s term `kapAr·Σ 2^(…)`, carried from
  block to block with the rescaling by `alpha` (`acc_step'`), and `ratio_err`'s `κ·W`, which
  becomes `kapA` relative to the weighted mean.

## Specification

Write `D = HEAD_DIM`, `N = N_CTX`, `σ = sm_scale` (the double, as a rational: `sigma`). Element
`(bh, n, e)` of `q`, `k`, `v`, `o` (for the head `bh = z·H + h`) is element
`idx d bh n e = (bh·N + n)·D + e` of its buffer (`Inst.q` etc., contiguous), and element `(bh, n)`
of `M` is element `bh·N + n`. For query `n`, its keys are `keys d n`: every `j < N`, and only
`j ≤ n` when `causal`. With the input values in `init` (`val16`: a `float16` element's value), the
exact score is `score = Σ_e q[bh, n, e]·k[bh, j, e]` and its magnitude
`scoreMag = Σ_e liftMag q[bh, n, e]·liftMag k[bh, j, e]`, where `liftMag` (`Rules.Num.liftMag`) is
`|x|` for a zero or normal `float16` value and the smallest normal magnitude `2^-14` for a subnormal
one: the magnitude the score's `tl.dot` error is relative to (`DotF16`, library round 25). For zero
and normal values it is `Σ_e |q·k|`, and in general at most
`Σ_e |q·k| + 2^-14·(Σ_e |q| + Σ_e |k|) + D·2^-28`. The exact weight is `weight = exp(σ·score)` (real
`exp`), and

- `attnRef d init bh n e = Σ_j weight_j·v[bh, j, e] / Σ_j weight_j` (over `keys d n`), the exact
  attention output;
- `attnAbs` the same weighted mean of `|v[bh, j, e]|`, and `vAbsSum = Σ_j |v[bh, j, e]|`;
- `lseRef d init bh n = log₂ Σ_j weight_j`, the exact log-sum-exp in base 2, which `M` holds
  (the kernel computes it as `m_i + log₂ l_i` with base-2 exponents).

A head is covered (`HeadOk d init bh`) when every element of `q`, `k`, `v` in it is a finite
`float16`, every `|v| ≤ 2^15`, and every `σ·scoreMag ≤ 2^10` (over all `n, j < N`).

`Post d init fin` says three things.

- **The output `o`.** Every element `(bh, n, e)` of `o` (`bh < Z·H`, `n < N`, `e < D`) holds some
  `float16` value `c` in `fin`. When head `bh` is covered, `c` is finite and, for every `X ≤ 2^10`
  that bounds the row's scaled magnitudes (`σ·scoreMag(n, j) ≤ X` for every key `j` of `n`),
  `|c − attnRef| ≤ tolO D N X · attnAbs + tolS D N X · vAbsSum + tolF D N X`.
- **The log-sum-exp `M`.** Every element `(bh, n)` of `M` holds some `float32` value `c`. When head
  `bh` is covered, `c` is finite and, for every such `X`, `|c − lseRef| ≤ tolM D N X + 2^-24·|lseRef|`.
- **Nothing else changes.** Every byte outside the elements of `o` and `M` (`outBytes`: the first
  `Z·H·N·D` elements of `o`, the first `Z·H·N` of `M`) is the same in `fin` as in `init`.

**The tolerances** are explicit rational expressions of `D`, `N` and `X` (`AttnFwd.Model`, "The
tolerances"), built only from the library's constants: `relDot16 D = (D + 2)·2^-22` (the score's
`tl.dot`, `Rules.Num.DotF16`), `relDot16 64` (the `p·v` `tl.dot`), `2^-22` (`Exp2F32`, `DivFull`,
`Log2F32`), one `float32` rounding `2^-24` and one `float16` rounding `2^-11`. The main term is the
error of each computed exponent, `expErr D X = (relDot16 D + 2^-20)·(3X/2) + 2^-50`: the score's
`tl.dot` error relative to `scoreMag`, times `qk_scale ≈ σ·log₂e ≤ 3σ/2`. It moves each weight by a
factor `2^±expErr`, and the weighted mean by about `expErr`. On top come the `float16` rounding of
`p` (`2^-11` relative and `2^-25` absolute per key), the `p·v` `tl.dot` relative to its lifted
magnitudes (a subnormal `p` counts at `2^-14`, adding `66·2^-36·|v|` per key: with the rounding, the
`omega0` of the `tolS·vAbsSum` term; a subnormal `v` counts at `2^-14`, adding `66·2^-36·|p|` per
key: over the keys at most `kapA = 2^-29` times `l_i`, and `absA = 2^-37` per block, the `absQ0`
part of `tolF`), the `float32` accumulation over `N/64` blocks (`stepAcc^(N/64)`, `stepL^(N/64)`),
the change from `qk_scale·ln 2` to `σ` (`thetaW X = 2^-20·X`), and the last division and rounding
(`lastRel`; near zero, the `float16` rounding's `2^-25`, most of `tolF`). Some values:

| `D` | `N` | `X` | `tolO` | `tolS` | `tolF` | `tolM` |
|---|---|---|---|---|---|---|
| 64 | 128 | 16 | `1.8·10^-3` | `3.1·10^-8` | `3.2·10^-8` | `8.3·10^-4` |
| 128 | 4096 | 16 | `3.6·10^-3` | `3.1·10^-8` | `3.2·10^-8` | `1.6·10^-3` |
| 128 | 2^14 | 256 | `3.1·10^-2` | `3.2·10^-8` | `3.4·10^-8` | `2.5·10^-2` |
| 256 | 2^16 | 16 | `2.1·10^-2` | `3.1·10^-8` | `3.9·10^-8` | `3.9·10^-3` |
| 256 | 2^16 | 1024 | `0.24` | `4.2·10^-8` | `4.2·10^-8` | `0.19` |

On the host's inputs, `σ·Σ_e|q_e·k_e| = σ·s²·Σ_e|X_e·Y_e|` with `X`, `Y` independent standard
normals (the test: `s = 0.5`, `σ = 0.5`; the benchmark: `s = 1`, `σ = 1.3`). Its mean at
`D = 128` is `σ·s²·128·2/π`, about `10` (test) and `106` (benchmark); a Chernoff bound on the sum
of the `128` independent terms, with a union bound over every pair of the largest shape
(`3.2·10^9` pairs in the test, `2^35` in the benchmark), keeps it below `22.7` (test) and `239.9`
(benchmark) except with probability below `10^-6` (and lower at `D = 64`). The lift adds at most
`σ·(2^-14·(Σ_e|q| + Σ_e|k|) + 128·2^-28)`; with `Σ_e|q| ≤ √128·|q|` and the `χ²` tail of `|q|²`
(every row's `|q|² ≤ 288·s²` except with probability below `3·10^-6`), that is below `0.006`
(test) and `0.031` (benchmark). So every `σ·scoreMag` is below `23` (test) and `240` (benchmark),
except with probability below `4·10^-6`. So at the test
(`X = 23`, `D = 128`, `N = 4096`) the output is within about `0.43%` of the weighted mean of `|v|`
(about nine `float16` roundings), at the benchmark (`X = 240`, `N = 2^14`) within about `2.9%`, and at the largest
covered values (`X = 2^10`, `D = 256`, `N = 2^16`) within `24%`: still below `1`, so the bound
fixes the output to well within its own magnitude, up to the two absolute terms. `tolS·vAbsSum` is
`tolS·N` times the mean of `|v|` over the keys (about `2^-25` per key, the `float16` rounding of a
small `p` and the lift of a subnormal one): at most `2.8·10^-3` of that mean at `N = 2^16`,
`5·10^-4` at the benchmark's `N = 2^14`. `tolF` is mostly the output's own `float16` rounding near
zero (`2^-25`, half `float16`'s smallest subnormal `2^-24 ≈ 6.0·10^-8`); the lift of subnormal `v`
in the `p·v` `tl.dot` adds about `kapA = 2^-29 ≈ 1.9·10^-9` and `N/64·2^-37`: `3.2·10^-8` at the
test's `N = 4096`, `3.4·10^-8` at the benchmark's `N = 2^14`, at most `4.2·10^-8` at `N = 2^16`.
It outweighs `tolO·attnAbs` only when `attnAbs` is below `tolF/(1 − tolO)`, at most `5.5·10^-8`
(at `N = 2^16`, `X = 2^10`), below `2^-24`: there the exact output rounds to `0` or `±2^-24`, and
`tolO·attnAbs + tolF` is below `2^-24`, so the bound still fixes the output to within one `float16`
subnormal step. The bound is relative to `attnAbs`, not to `|attnRef|`: for zero-mean `v`,
`|attnRef|` can be much smaller than `attnAbs`.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `Z`, `H`, `N_CTX`, `HEAD_DIM`, `sm_scale` (the Python
`float`, a double), `causal`, `warp_specialize`, and the storage of `q`, `k`, `v`, `o` (`Buf .f16`)
and `M` (`Buf .f32`), each a base address and a length in elements, contiguous.

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `head_dim` | `HEAD_DIM ∈ {64, 128, 256}` | the wrapper allows `16` and `32` too (line 512), which line 188's `tl.static_assert(BLOCK_N <= HEAD_DIM)` rejects on the host (no outcome); the host uses `64` and `128` |
| `n_mul` | `0 < N_CTX` and `N_CTX % 128 = 0` | otherwise the last query tile of a head runs into the next head's rows (two programs store the same rows of `o`: a race) and its last key block reads the next head's keys; every host size is a multiple of `128` |
| `n_le` | `N_CTX ≤ 2^16` | keeps the tolerance meaningful and the proof's numeric bounds valid (`N/64 ≤ 1024` blocks); the host's largest is `2^14` |
| `zh_pos` | `0 < Z`, `0 < H` | `off_hz // H` (line 191) needs `H ≠ 0`; an empty batch is not covered |
| `zh_lt` | `Z·H < 2^16` | the grid's `y` is `Z·H` (line 549), within CUDA's limit; the host's largest is `192` |
| `rows_lt` | `Z·H·N_CTX < 2^31` | `y_dim` (line 194), the descriptors' row count and every row offset (lines 208–209, 243) are `int32` values that do not wrap; the host's largest is `2^21` |
| `scale` | `sm_scale` finite, in `[2^-64, 2^64]` | keeps `qk_scale` (line 219) a finite positive `float32` within `2^-22` relative of `sm_scale·log₂e`; the host passes `0.5` and `1.3` |
| `lens` | `Z·H·N_CTX·HEAD_DIM` elements in `q`, `k`, `v`, `o`, `Z·H·N_CTX` in `M` | the kernel accesses them (`torch.empty_like`, `torch.empty`) |
| `align` | the four tensors' base addresses are multiples of 16 | TMA's requirement for a descriptor's base (`TensorDesc.Valid`); PyTorch's allocations are |
| `disj` | `o` shares no byte with `q`, `k`, `v`, `M`, and `M` none with `q`, `k`, `v` | the outputs are fresh tensors (lines 513, 521); an overlap would make programs race |

The descriptors (`descArg`): when `warp_specialize` is false, host `TensorDescriptor`s
`⟨b.ptr 0, [Z·H·N_CTX, HEAD_DIM], [HEAD_DIM, 1], block_shape, "zero"⟩` with the block shapes the
pre-hook sets; when it is true, the tensors' pointers. The model's `maybe_make_tensor_desc` returns
the host descriptor, or builds the same one (`make_desc`), so both kinds are covered.

`Pre d init`: the five buffers are valid in the initial memory (`Buf.Valid`: base non-negative and
aligned to the element size, the buffer inside `[0, 2^64)`, every byte present). Any input values
are allowed: `Post` says what holds for covered heads, and only that `o` and `M` hold values for
the others.

**The host's use is covered.** Every test and benchmark shape meets `Inst.ok`: `HEAD_DIM ∈ {64,
128}`, `N_CTX ∈ {128, 1024, 4096, 2^10, …, 2^14}` (multiples of `128`, at most `2^16`),
`Z·H ≤ 192`, `Z·H·N_CTX ≤ 2^21`, `sm_scale ∈ {0.5, 1.3}`, fresh contiguous tensors from
`torch.empty`/`torch.randn`/`normal_` (16-byte aligned, disjoint), both values of `causal` and of
`warp_specialize`. Their values are finite, far below `2^15`, and have `σ·scoreMag` far below `2^10`
(Specification), so every head the script runs is covered.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch the script makes is the direct JIT launch `_attn_fwd[grid](…)` of
lines 562–574, through `attention(…)` (line 677 in the test; line 754, timed by
`triton.testing.do_bench` at line 759, in the benchmark). Point by point:

- **4 (options).** The launch gives the constexprs and the two options `num_stages=2` and
  `num_warps=4`, two of the four that do not change the meaning; `extra_kern_args` is empty on the
  H200 (`is_hip()` and `is_blackwell()` are false), so no `maxnreg`, `waves_per_eu` or
  `allow_flush_denorm`. No `warmup`, `triton.compile`, `triton.Config` (the decorator is removed:
  SETUP.md; `configs`, `keep` and `prune_invalid_configs` are unused) or `@triton.heuristics`; the
  decorator is a bare `@triton.jit`, so `debug` is not set. So `launch` keeps every default:
  `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, no `launch_pdl`.
- **5 (descriptor scratch).** When `warp_specialize` is true the kernel builds descriptors (line
  173). The allocator (lines 543–546) returns a fresh `torch.empty(size, dtype=torch.int8)`, which
  holds no byte of `q`, `k`, `v`, `o` or `M` (all alive during the launch), and nothing else
  writes it during the launch.
- **6 (precompiled kernel).** No `CompiledKernel` is launched.
- **8 (host descriptors).** The host descriptors (lines 528–536) leave `round_f32_to_tf32` at its
  default, false.
- **9 (CUDA graphs).** No call is made during a stream capture and nothing is replayed: the test
  calls `attention` directly, the benchmark through `triton.testing.do_bench` (plain calls timed with
  CUDA events, not `do_bench_cudagraph`); the script opens no `torch.cuda.graph`.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`; the helpers are top-level `def`s under a
bare `@triton.jit`, called by name. The kernel's parameters are a `float` (`sm_scale`), a pointer
(`M`), three integers, four descriptor arguments (host `TensorDescriptor`s or tensors) and seven
`tl.constexpr`s with `int`/`bool` arguments, none with a default. The statements are assignments
(`name = e`, tuple assignments `lo, hi = …` and `acc, l_i, m_i = _attn_fwd_inner(…)`), `op=`
(lines 78, 108–109, 219, 241), `for … in tl.range(…)`, `if`/`elif`/`else` on compile-time
conditions (`STAGE`, `dtype`, `FP8_OUTPUT`, `IS_HOPPER`, `isinstance`, `STAGE & 1`), `return`s in
the helpers (one visited per call), and the calls `tl.static_assert` and `tl.store`. The
expressions are `tl.program_id`, `tl.arange`, `tl.zeros`, `tl.dot`, `tl.where`, `tl.maximum`,
`tl.max`, `tl.sum`, `tl.math.exp2`, `tl.math.log2`, `tl.multiple_of`, `tl.make_tensor_descriptor`,
the descriptor methods `load` and `store`, `.T`, `.to(…)`, Python's `float` and `isinstance`, the
operators `+ - * / // % &`, the comparisons `>=`, `<=`, `==`, `and`/`not` on constexprs,
`a if c else b` (line 187) and the subscripts `[:, None]`, `[None, :]`. The lines the front end does
not visit (65, 88–93, 98, 198–199) are dead on every covered launch. The module-level names it reads
are `triton`, `tl` and the two helpers, each bound only by its own `import` or `def`.

## Assumptions

Ten assumptions, all of kind `[rule]`, each cited as the library states it
([num.md](../../docs/rules/num.md), "Facts the library states"):

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid, then from every initial memory from which the
  model's launch is well behaved, every outcome of the compiled kernel launched as `launch d` is one
  of the model's. The rule covers a launch only within its scope: part (A), which the launches meet
  (Conditions), and part (B), the environment, which the claim relies on
  ([rule page](../../docs/rules/compiler-faithful.md)).
- `dot` — `Rules.Num.DotF16 P`: `tl.dot` of `float16` operands into `float32` (lines 73 and 103),
  each output element within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with the lifted
  magnitude `L = |acc| + Σ liftMag a·liftMag b` (a subnormal operand counted at `2^-14`;
  `Rules.Num.DotLiftAcc`, since library round 25), for finite operands and room below the `float32`
  range.
- `dot_total` — `Rules.Num.DotTotal P cfg16` (the configuration written out): the model's `tl.dot`
  always has a result.
- `exp2` — `Rules.Num.Exp2F32 P`: `tl.math.exp2` on `float32` (lines 82, 84), within `2^-22`
  relative and `2^-126` absolute of `2^x`.
- `exp2_ninf` — `Rules.Num.Exp2F32NegInf P`: `tl.math.exp2(−∞) = +0` (line 84 at the first block,
  where `m_i = −∞`).
- `exp2_total` — `Rules.Num.MathTotal P .exp2 .binary32`: `exp2` has a result for every input.
- `log2` — `Rules.Num.Log2F32 P`: `tl.math.log2` on `float32` (line 241), within `2^-22` relative and
  `2^-149` absolute of `log₂ x`.
- `log2_total` — `Rules.Num.MathTotal P .log2 .binary32`.
- `div_full` — `Rules.Num.DivFull P`: `/` on `float32` (line 242) is `div.full.f32`: for finite
  operands, a nonzero divisor and `|x/y| ≤ 2^126`, the result is finite and within
  `2^-22·|x/y| + 2^-148` of `x/y`.
- `div_total` — `Rules.Num.DivTotal P .binary32`.

There is no `[kernel]` assumption. The IEEE operations (`+`, `-`, `*`, the fused `mul_add` and
`subc`, `tl.max`, `tl.maximum`, `tl.sum`), the conversions to `float16` (lines 101, 245) and the
constants are the library's definitions and need no platform fact.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` by the definition of `launch`; `hvalid` (`launch_valid`): the grid
  `(N/128, Z·H, 1)` is within CUDA's limits with fewer than `2^31` programs (`Z·H·N/128 < 2^31`), and
  the eight block shapes are lists of powers of two with at most `2^20` elements.
- Footprints: every program reads `q`, `k`, `v` (`Rd`); program `i`, whose query tile is
  `(start_m, off_hz) = (i mod g0, i / g0)` with `g0 = N/128` (`pm`, `pbh`, `pid0`, `pid1`), writes
  rows `128·i … 128·i + 127` of the `[Z·H·N, D]` view of `o` and the same elements of `M` (`Wr`,
  `oTile`, `mTile`): `off_hz·N + 128·start_m = 128·i` (`row0`).
- `hsafe` (`safe_program`, in `AttnFwd.Program`), line by line:
  - Lines 189–192 (`safe_idiv1`, `safe_irem1`): `off_z = bh / H`, `off_h = bh mod H`.
  - Lines 195–206 (`safe_maybe_desc`): each descriptor is `descOf d b BM`, both kinds of argument
    (a built descriptor passes `safe_make_tensor_descriptor`, its shape and strides in range, and
    equals the host one, `make_desc`); every access at `[O, 0]` through it meets Triton's requirements
    (`access_descOf`: alignment, strides, block shape) and lane `(r, e)` is element `(O + r)·D + e`
    (`lane2`).
  - Lines 208–219: `offset_y = bh·N` and `qo_offset_y = 128·i` (no `int32` wrap: `rows_lt`);
    `m_i = −∞`, `l_i = 1.0` (`m_init_eq`, `l_init_eq`); `qk_scale` is a finite `float32` with
    `0 < qk_scale ≤ 3σ/2` and `|qk_scale·ln 2 − σ| ≤ 2^-22·σ` (`qk_scale_facts`, from one rounding of
    `σ` and of `1.44269504` to `float32` and of their product).
  - Line 221 (`safe_load_rows`): `q` holds the rows `128·start_m + r` of head `bh` (`QReads`).
  - Lines 225–239 (`safe_inner1`, `safe_inner2`, `safe_inner3`): causal launches run the helper with
    `STAGE = 1` (blocks `0 … 2·start_m − 1`, every key below the tile's rows) and then `STAGE = 2`
    (blocks `2·start_m`, `2·start_m + 1`, masked); non-causal launches run it with `STAGE = 3`
    (all `N/64` blocks). Each loop (`safe_loop_plain`, `safe_loop_masked`, by `safe_forRange` with
    the memory unchanged and the offsets `bh·N + 64·t`) keeps `StInv`: for a covered head and every
    `X ≤ 2^10` bounding the row, every row's `RowSt` after `t` blocks: before the first block
    `m_i = −∞`, `|l_i| ≤ 1`, `acc = 0`; afterwards `MInv` (`m_i` is within `E = expErr D X` of some
    attended key's `qk_scale·s_j`, and no attended key's is above `m_i + E`), `LInv` (`l_i` is
    within a relative factor `ffac − 1` and an absolute `t·2^-100` of `Σ_j 2^(qk_scale·s_j − m_i)`
    over the attended keys of the first `t` blocks) and `AInv` (the same for `acc` and
    `Σ_j 2^(…)·v_j`, with `omega0·Σ|v_j|`, `2^-29·Σ_j 2^(…)` and `t·2^-37` absolute). One block
    (`row_step_plain`, `row_step_masked`): `Num.DotF16` at each score (`qk_lane`, relative to
    `scoreMag`) and at each `p·v` element (`pv_lane`; in `astep` the lifted magnitudes are at most
    `Σ|p·v|` plus `2^-14·(|v| + |p|) + 2^-28` per attended key, by `DotLift.liftMag_mul_le`, with
    `|p| ≤ (1 + psi16r)·2^(…) + omR`, so the `|p|` part stays relative to the weights, and a masked
    key's `p` is exactly `0`), the running maximum (`mstep_plain`, `mstep_masked`; a masked lane's exponent is
    below `−2^19`, so its `p` is at most about `2^-126` and enters `l_i` and `acc` only through
    their absolute error terms), `Exp2F32` at each `p` and at
    `alpha` (`Exp2F32NegInf` at the first block), the roundings of `acc·alpha`, of `l_i·alpha + l_ij`
    and of `p` to `float16` (`lstep`, `astep`).
  - Lines 241–245 (`safe_epi`): `safe_math` (`log2_total`), `safe_add`, `safe_div` (`div_total`),
    `safe_store_buf` for `M` (lanes `bh·N + 128·start_m + r`, distinct), `safe_cvt`, and
    `safe_store_rows` for `o`. Per row (`row_out`): after `T` blocks that hold all its keys
    (`T = N/64`, or `2·start_m + 2` when causal), `final_o` (`DivFull`, the `float16` rounding,
    `ratio_err`, and the change from `2^(qk_scale·s)` to `exp(σ·s)`, `wmean_err`) and `final_M`
    (`Log2F32`, `|log₂ l_i| ≤ 17`) give `OutO` and `OutM` at every `X`; `tol_facts` checks the
    numeric side conditions (`expErr`, `facc`, `fl`, `betaL`, `relQ` and the absolute terms within
    fixed bounds, such as `facc ≤ 6/5` and `betaL ≤ 1/4`) at `D ≤ 256`, `X ≤ 2^10`, `N ≤ 2^16`.
- `hdisj`, `hRW`: a program's tile of `o` and `M` is inside `outBytes` (`Wr_sub`), which is
  disjoint from `q`, `k`, `v` (`outBytes_disj`, from `disj`); two programs' tiles are disjoint
  (`Wr_disj`: rows `128·i + r`, `r < 128`, determine `i`).
- `hpost`: element `(bh, n, e)` of `o` and `(bh, n)` of `M` are in the tile of program
  `bh·g0 + n/128` (`prog_of`), whose `Qp` gives them; the final memory agrees with that program's on
  its footprint (`Mem.read_congr`). A byte outside `outBytes` is in no write footprint, so it keeps
  its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref` gives a total behaviour that
satisfies `faithful`; `Platform.sm90ref.dotF16`, `exp2F32`, `exp2F32NegInf`, `log2F32`, `divFull`
and `Platform.ref.dotTotal`, `mathTotal`, `divTotal` give the others.

`satisfiable`: a covered instance with the test's smallest `N_CTX` and `HEAD_DIM` and its
`sm_scale`, one head (`Z = H = 1`, which the test does not use: its `H` is `2` or `48`),
`N_CTX = 128`, `HEAD_DIM = 64`, `sm_scale = 0.5`, not causal, host descriptors; the four tensors at `2^21`, `2·2^21`, `3·2^21`,
`4·2^21` and `M` at `5·2^21` (`testInst`), in a memory where every byte is present (`testMem`).

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float16` only.** This folder is `_attn_fwd`'s `float16` run (`FP8_OUTPUT = False`): `q`, `k`,
  `v`, `o` are `float16` and `M` is `float32`. The script's `float8e5` run (`FP8_OUTPUT = True`,
  the `triton-fp8` provider of the test and the benchmark, with a transposed `v`) is excluded here;
  it is the folder `attn_fwd_e5m2`.
- **The H200.** `IS_HOPPER = True` (`is_hopper()` on the H200), so line 87's Blackwell branch is
  never taken and the host passes no `maxnreg`. Other devices are outside `Compiler.Faithful`'s
  point 7.
- **The launch's configuration.** `BLOCK_M = 128`, `BLOCK_N = 64`, `num_stages = 2`,
  `num_warps = 4` (the config SETUP.md chose); other configs are not covered.
- **`HEAD_DIM ∈ {64, 128, 256}`.** `16` and `32`, which the wrapper accepts, are rejected on the
  host by line 188 (no outcome to cover).
- **`N_CTX` a positive multiple of `128`, at most `2^16`.** A ragged last tile would race on `o`
  and read another head's keys; every host size is a multiple of `128`. The bound `2^16` keeps the
  accumulated rounding (`N/64` blocks) and the tolerance small; the host's largest is `2^14`.
- **`Z·H < 2^16` and `Z·H·N_CTX < 2^31`.** The grid's `y` limit and `int32` row offsets that do not
  wrap; the host's largest are `192` and `2^21`.
- **`sm_scale` in `[2^-64, 2^64]`, finite.** The host passes `0.5` and `1.3`.
- **Covered heads** (`HeadOk`): finite inputs, `|v| ≤ 2^15`, `σ·scoreMag ≤ 2^10`. For another head
  the claim says only that `o` and `M` hold some value. The bound on `v` leaves room for `o` in
  `float16`; the bound on the scores keeps the causal mask's `−10^6` below every unmasked exponent
  and the tolerance below `1`. The host's inputs are far inside (Specification).
- **The tolerance's form.** It is relative to `attnAbs` (the weighted mean of `|v|`), plus a term
  in `Σ|v|` and an absolute one (`tolF`, below `4.3·10^-8`, mostly the output's `float16` rounding
  near zero), and it grows with `X`: it follows the library's `tl.dot` rule,
  which bounds a score's error by `relDot16 D·scoreMag`, and each weight is an exponential of a
  score.
- **Alignment and disjointness** (`align`, `disj`): the host's tensors are fresh PyTorch
  allocations; an output overlapping an input would race.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions).

Modelling decisions:

- `Z`, `H`, `N_CTX` are `int32` (`IntV.ofNat`): Triton passes an integer argument other than `1` as
  an `int32` (all are below `2^31` on covered instances), and at `1` (the test's `Z = 1`) the
  constant `1`, which gives the same model (`model`'s docstring). `sm_scale` is the `float32`
  rounding of the double (`FloatV.hostF32`).
- `dtype`, `FP8_OUTPUT`, `IS_HOPPER` and `warp_specialize` are no parameters of the model: the
  first three are fixed on covered launches, and `warp_specialize` is read only by `tl.range`'s
  scheduling keyword (which changes no outcome of a well-behaved model) and by line 87's false
  condition; it still chooses the descriptor kind (`descArg`).
- `_maybe_make_tensor_desc`'s `isinstance` is decided by Python at compile time: the model
  `maybe_make_tensor_desc` matches on the argument's kind (`DescArg`).
- The compile-time `if`s on `STAGE`, `STAGE & 1`, `STAGE & 2` are Lean `if`s on the `Nat`
  parameter; `4 - STAGE` is computed by Python.
- `float("inf")` is the Python `+∞`, built in `float32` as an operand of `-` (line 214); `1.0`,
  `1.44269504`, `0` and `-1.0e6` are built in `float32` (`FloatV.litF32` of the double).
- Floating-point fusion: `qk * qk_scale + where(…)` (line 76) and `l_i * alpha + l_ij` (line 106)
  are `tl.mul_add`, and `qk * qk_scale - m_ij[:, None]` (line 81) is `tl.subc`: an addition or
  subtraction that a product feeds, which `enable_fp_fusion` may contract to an `fma`; the other
  operations are IEEE (`tl.mul`, `tl.sub`, `tl.add`).
- `tl.max(qk, 1)` is a row maximum in any tree (`tl.max1`), `tl.sum(p, 1)` a row sum in any tree
  (`tl.sum1`); `/` is the platform's division (`P.div`); `.to(tl.float16)` is one rounding to
  nearest even (`tl.cvt`).
- The hint of line 59 is a check at the definition of `lo` (`tl.multiple_of1`); the hint of line
  70 on the induction variable has no effect and is dropped (SEMANTICS.md, "Compiler hints").
- The launch lists the source's block shapes (`[128]`, `[64]`, `[128, 1]`, `[1, 64]`, `[128, 64]`,
  `[HEAD_DIM, 64]`, `[64, HEAD_DIM]`, `[128, HEAD_DIM]`); `enable_fp_fusion` is Triton's default
  `true`; `num_warps = 4` and `num_stages = 2` change no result and are not recorded.
