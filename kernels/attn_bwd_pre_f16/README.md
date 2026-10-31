# attn_bwd_pre_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/06-fused-attention.py`](source/06-fused-attention.py) (tutorial 06, fused
attention; [SETUP.md](SETUP.md) says how it was set up). Lean: [`lean/`](lean/), the module
`AttnBwdPre` (namespace `Kernels.AttnBwdPre`) and its parts `AttnBwdPre.Model` (the model, the
conditions, the specification, the assumptions), `AttnBwdPre.Err` (the error of one output, the
tolerance) and `AttnBwdPre.Proof` (the proof, `main`, `consistent`, `satisfiable`).

The kernel is `_attn_bwd_preprocess` (source lines 248–262). It reads two module-level names,
`triton` (its decorator) and `tl`, bound by the imports of lines 20–21, which are kernel lines too
and are mapped (dropped: they add no step). The host's `_attention.backward` (lines 583–623) takes
the tensors `q, k, v, o, M` saved by the forward pass and the incoming gradient `do`, asserts
`do.is_contiguous()` and that `q, k, v, o, do` have the same strides, sets
`BATCH, N_HEAD, N_CTX = q.shape[:3]` and `PRE_BLOCK = 128`, asserts `N_CTX % PRE_BLOCK == 0`,
allocates `delta = torch.empty_like(M)` (a fresh `float32` tensor of shape
`(BATCH, N_HEAD, N_CTX)`), and launches (lines 600–607)

```python
_attn_bwd_preprocess[(N_CTX // PRE_BLOCK, BATCH * N_HEAD)](o, do, delta, BATCH, N_HEAD, N_CTX,
                                                           BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM)
```

with `o` (allocated by the forward as `torch.empty_like(q)`) and `do` `float16` tensors of shape
`(BATCH, N_HEAD, N_CTX, HEAD_DIM)`, contiguous. `ctx.HEAD_DIM` is the forward's `HEAD_DIM_K`, which
the forward asserts is one of `16, 32, 64, 128, 256` (line 512). The script runs the backward pass
from its test (`test_op`, lines 631–695, through `tri_out.backward(dout)`, line 682) with
`Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, and from its
benchmark (`bench_flash_attention`, lines 739–775, through `o.backward(do, retain_graph=True)` timed by
`triton.testing.do_bench`, lines 755–759) with `BATCH = 4`, `H = 32`, `N_CTX = 2^10 … 2^14` and
`HEAD_DIM ∈ {64, 128}` (lines 706–736). The backward with `float8e5` is not supported by the
tutorial (line 643), so `o` and `do` are always `float16` here.

The file's other `@triton.jit` functions (`_attn_fwd_inner`, `_maybe_make_tensor_desc`,
`_attn_fwd`, `_attn_bwd_dkdv`, `_attn_bwd_dq`, `_attn_bwd`) are other folders' kernels and
helpers: this launch runs none of them, so they are named in `source.unused`, and their headers
(decorators and `def` lines, which Python runs at import) are host lines. Every other line outside
the kernel (the imports of `pytest`, `torch`, `os` and `TensorDescriptor`, `DEVICE`, the five
device predicates, `_host_descriptor_pre_hook`, the autotune configs and their filters, the class
`_attention`, `attention`, the test, the benchmark and the `__main__` block) is a host line: the
kernel reads none of what they bind.

## Specification

The kernel computes, for every row `r` of `o` (one batch, head and position), the `float32` dot
product of that row with the same row of `do`, `Delta[r] = Σ_k o[r, k] · do[r, k]` over the
`HEAD_DIM` columns. `Post d init fin` says two things.

- **The outputs.** For every `(batch, head)` pair `bh < Z·H` and every position `m < N_CTX`, with
  `r = bh·N_CTX + m` (element `r` of `Delta` is `delta[b, h, m]` for `bh = b·H + h`), `OutSpec`
  holds:
  - element `r` of `Delta` can be read in the final memory, as some `float32` value `c`;
  - whenever row `r` of `O` and of `DO`, the elements `r·HEAD_DIM + k` for `k < HEAD_DIM`, hold
    **finite** `float16` values `vo k` and `vdo k` in the initial memory, `c` is finite and

    `|c − Σ_k vo k · vdo k| ≤ tol(HEAD_DIM, Σ_k |vo k · vdo k|)`,

    the sums exact (in the rationals: `dotSum`, `dotAbs`).
- **Nothing else changes.** Every byte outside the elements `r < Z·H·N_CTX` of `Delta` is the same
  in `fin` as in `init`: `O`, `DO`, and every other byte the launch is given.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch that
runs ends normally, without fault or hang, with a final memory satisfying `Post`.

**The tolerance** (`tol`, a computable rational function). With `u = 2^-24` and `η = 2^-150` (one
`float32` rounding's relative and absolute bounds), `D = HEAD_DIM` and `B = Σ_k |vo k · vdo k|`,

`tol(D, B) = errBound(3u, 4η, D − 1, D − 1, (1 + u)·B + D·η) + (u·B + D·η)`,

where `Tree.errBound δ η d N A = ((1 + δ)^d − 1)·A + η·N·(1 + δ)^d` is the library's bound for a
sum along a tree of depth `d` with `N` nodes. The second term covers the `float32` products
(`u·|a·b| + η` each; the products of two `float16` values are in fact exact in `float32`, but the
library's bound counts a rounding); the first, the sum of the products along any tree of `D` leaves
(depth and node count at most `D − 1`, whatever tree Triton chooses) whose additions may be
contracted with the products (`3u` relative and `4η` absolute per addition, `FloatV.AddRel.err`).
It is the library's `bcastTreeBound D B` (`tol_eq`), the bound of `sum_bcast_mul_tree_err` for
exactly this computation. Its values, to first order `(3D − 2)·2^-24·B`:

| `HEAD_DIM` | `tol / B` | absolute part |
|---|---|---|
| 16 | `2.74·10^-6` (`46·2^-24`) | `76·2^-150` |
| 64 | `1.13·10^-5` (`190·2^-24`) | `316·2^-150` |
| 128 | `2.28·10^-5` (`382·2^-24`) | `636·2^-150` |
| 256 (the largest covered) | `4.57·10^-5` (`766·2^-24`) | `1276·2^-150` |

and `tol_le` proves `tol(D, B) ≤ 2^-14·B + 2^-138` (`2^-14 ≈ 6.1·10^-5`) for every covered
`HEAD_DIM ≤ 256`. The bound is relative to `Σ|o·do|`, not to `|Σ o·do|`: a sum with cancellation can
lose every digit of its result in `float32` too, and no bound relative to the result holds.

**What it rejects.** At every covered size the tolerance is far below a `float16` rounding
(`2^-11 ≈ 4.9·10^-4` relative): a result rounded to `float16` or `bfloat16`, an accumulation in a
16-bit format, or a dropped or doubled column does not meet `Post` in general. For example, with
`HEAD_DIM = 64` and every `o[r, k] = do[r, k] = 1` except `o[r, 0] = 1 + 2^-10`, the exact sum is
`64 + 2^-10` (also the `float32` result), `tol ≈ 1.13·10^-5 · 64 ≈ 7.3·10^-4`, and the `float16`
rounding `64` is `2^-10 ≈ 9.8·10^-4` away: `Post` rejects it. A missing column of `1·1` is `1` away.

Rows with a non-finite input (`±∞`, NaN in `o` or `do`) have no exact finite sum; for them `Post`
says only that a `float32` value is stored (Narrowings).

## Conditions

An instance `Inst` is the host's choice: `Z`, `H`, `N_CTX` (the integers `BATCH`, `N_HEAD`,
`N_CTX`), the constexprs `HEAD_DIM` and `BLOCK_M`, and the tensors `o` (`O`), `do` (`DO`), both
`float16`, and `delta` (`Delta`), `float32`, as buffers (`Buf`: a base address and a length in
elements). Element `(b, h, m, k)` of `o` and `do` is element `((b·H + h)·N_CTX + m)·HEAD_DIM + k`
of the buffer, and element `(b, h, m)` of `delta` is `(b·H + h)·N_CTX + m`, as the contiguous
tensors the host passes are laid out and as the kernel's own index arithmetic reads them. `outSet d`
is the elements `r < Z·H·N_CTX` of `Delta`, `inSet d` the elements `e < Z·H·N_CTX·HEAD_DIM` of
`O` and `DO`.

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `block_pow2` | `BLOCK_M` is a power of two | Triton requires it of `tl.arange`; the host's `128` |
| `head_pow2` | `HEAD_DIM` is a power of two | Triton requires it of `tl.arange`; the host's `16 … 256` |
| `head_le` | `HEAD_DIM ≤ 256` | the host's largest (line 512); the tolerance is stated up to it (`tol_le`) |
| `block_le` | `BLOCK_M · HEAD_DIM ≤ 2^20` | Triton's largest block (`[BLOCK_M, HEAD_DIM]`); the host's at most `128 · 256 = 2^15` |
| `ctx_dvd` | `BLOCK_M` divides `N_CTX` | the host asserts `N_CTX % PRE_BLOCK == 0` (line 599); the grid `N_CTX // BLOCK_M` then covers every position, and the kernel has no mask |
| `ctx_lt` | `N_CTX < 2^31` | Triton passes `N_CTX` as an `int32` |
| `bh_lt` | `Z · H < 2^16` | CUDA's limit on the grid's axis 1, `BATCH * N_HEAD` (`Launch.Valid`); a larger grid is rejected on the host |
| `grid_lt` | `(N_CTX // BLOCK_M) · Z · H < 2^31` | fewer than `2^31` programs (`Launch.Valid`) |
| `row_fit` | `(N_CTX − 1) · HEAD_DIM < 2^31` | the `int32` products `off_m * HEAD_DIM` (lines 258–259) do not wrap |
| `head_fit` | `(Z·H − 1) · HEAD_DIM · N_CTX < 2^31` | the `int32` products `off_hz * HEAD_DIM * N_CTX` (lines 258–259) and `off_hz * N_CTX` (line 262) do not wrap |
| `O_len`, `DO_len` | `Z·H·N_CTX·HEAD_DIM` elements at least | `O` and `DO` hold every element the launch reads |
| `Delta_len` | `Z·H·N_CTX` elements at least | `Delta` holds every element the launch writes |
| `Delta_O`, `Delta_DO` | the elements of `Delta` the launch writes share no byte with the elements of `O` (of `DO`) it reads | the host allocates a fresh `delta`; an overlap would make a program read what another writes, or itself writes (Narrowings) |

`row_fit` and `head_fit` are the kernel's own limits, not only the proof's: beyond them the `int32`
index arithmetic wraps (the column offset `off_n` and the three terms are added to the 64-bit
pointer one by one, so only the products must fit). `O` and `DO` may overlap each other (both are
only read).

The three integer arguments are part of the instance, and the conditions constrain them, as the
user's decision of 2026-10-29 on library issue L117 asks. `Z` and `H` are never read by the kernel,
so their class (the constant `1`, as in the test's `Z = 1`, or an `int32`, …) changes nothing, and
the model takes them as `int32`. `N_CTX` is a runtime `int32` (`ctx_lt`); it can be `1` only when
`BLOCK_M = 1`, and then Triton passes the compile-time constant `1`, whose two uses (as an operand
of `*` with an `int32` scalar) build it in `int32`, the value the runtime `1` has, so the model is
the same (`model`'s docstring). A multiple of 16 is marked divisible by 16 for the compiler, which
is true of it.

Covered in particular: every launch the script makes. The test's largest is `4 · 48` heads of
`4096 × 128` (`(Z·H − 1)·HEAD_DIM·N_CTX = 191 · 2^19 < 2^31`), the benchmark's `4 · 32` heads of
`16384 × 128` (`127 · 2^21 < 2^31`, `2^28` elements of `o`); with `BLOCK_M = 128`, `HEAD_DIM` any of
`16 … 256`. Also covered: every other power-of-two `BLOCK_M` and `HEAD_DIM ≤ 256` within Triton's
block limit, every `N_CTX` that `BLOCK_M` divides within the `int32` limits (`N_CTX = 0` included:
no program runs and nothing changes), any `Z` and `H` with `Z·H < 2^16` within those limits (`Z·H = 0`
included), and buffers longer than the tensors.

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to the element size, inside `[0, 2^64)`, every byte present). The input
values are not restricted: any `float16` values, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of the kernel the script makes is the one of lines 602–607, a
direct launch of the `@triton.jit` function, in `_attention.backward`, which autograd runs from
`tri_out.backward(dout)` (line 682) and, in the benchmark, from `o.backward(do, retain_graph=True)`
(lines 757–759). Point by point:

- **4 (options).** The launch gives the constexprs `BLOCK_M` and `HEAD_DIM` and no option at all:
  no `num_warps`, `num_stages`, `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any other.
  There is no `kernel.warmup`, no `triton.compile`, no `@triton.autotune` on this kernel (so no
  `triton.Config`: the list of lines 135–146 decorates nothing in the source) and no
  `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so its `debug` is not set. So
  `launch` keeps every default: `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`,
  `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor (no
  `tl.make_tensor_descriptor`), so it asks the allocator for nothing; the allocator the forward sets
  (`triton.set_allocator(alloc_fn)`, line 546) is not involved.
- **6 (precompiled kernel).** No `CompiledKernel`: the launch is the JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are three
  tensors, three integers and two constexprs (the descriptors of lines 528–536 go to `_attn_fwd`).
- **9 (CUDA graphs).** No call is made during a CUDA stream capture, and nothing is replayed: the
  test calls `backward` directly (line 682), and the benchmark through `triton.testing.do_bench`
  (line 759), which times plain calls of its function, not `triton.testing.do_bench_cudagraph`;
  the script opens no `torch.cuda.graph` and makes no capture of its own.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it. The script's
own code breaks none of it visibly: the kernels before and after this launch on the stream (the
forward that wrote `o`, and `_attn_bwd`, launched next at lines 609–621, which reads `delta`) are
ordered by the stream and use no `launch_pdl`, and nothing else writes `o`, `do` or `delta` during
the launch. The script's `DEVICE` is whatever device is active, and its `is_hip()` branches are for
AMD devices, outside the target (an H200).

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`; its parameters are three pointers (tensor
arguments), three unannotated integers (`int` arguments) and two `tl.constexpr` with `int`
arguments, none with a default; its statements are assignments `name = e` and one call
(`tl.store`). Its expressions are `tl.program_id(0)`, `tl.program_id(1)`, `tl.arange`,
`tl.load(…)` without mask or `other`, the method `.to(tl.float32)`, `tl.sum(…, axis=1)`,
`tl.store(…)` without mask, the subscripts `[:, None]` and `[None, :]`, and the operators: `*` of
an `int32` scalar and a constexpr (lines 254, 258–259), of two `int32` scalars (lines 258–259,
262), of an `int32` block and a constexpr (lines 258–259) and of a `float16` and a `float32` block
(line 260); `+` of an `int32` scalar and an `int32` block (line 254), and of a pointer (scalar or
block) and an `int32` scalar or block (lines 258–259, 262), which advances the pointer by that
many elements ([SEMANTICS.md](../../docs/SEMANTICS.md), "Values"). The only module-level names it
reads are `triton` (its decorator) and `tl`, Triton's modules, each bound only by its own `import`
(lines 20 and 21): no other statement of the module binds `triton`, `tl` or
`_attn_bwd_preprocess`. The names it uses otherwise are its parameters and its own assignments; in
particular its `HEAD_DIM` is its parameter, not the module-level `HEAD_DIM` the benchmark's loop
binds (line 709).

## Assumptions

One assumption, of kind `[rule]`, cited as the library states it:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved (no undefined behaviour, no race between programs, no
  conflict within a program), every outcome of the compiled kernel launched as `launch d` is one of
  the model's. The rule covers a launch only within its scope: part (A), which the launches meet
  (Conditions), and part (B), the environment, which the claim relies on
  ([rule page](../../docs/rules/compiler-faithful.md)).

There is no device fact and no `[kernel]` assumption. Every operation of the model is exact or
has its meaning in the library: the conversions `float16 → float32` are exact widenings, the
product is the IEEE `float32` product, and the sum's additions are the library's `AddRel` (the IEEE
sum, or a sum contracted with a product), none of which depends on the platform `P`.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false`, by definition of `launch`.
- `hvalid` (`launch_valid`): axis 0 has `N_CTX // BLOCK_M ≤ N_CTX < 2^31` programs, axis 1
  `Z·H < 2^16` (`bh_lt`), axis 2 one; the grid has fewer than `2^31` programs (`grid_lt`); and the
  five block shapes are powers of two with at most `2^20` elements (`block_pow2`, `head_pow2`,
  `block_le`).
- Program `p < gx·Z·H` (`gx = N_CTX // BLOCK_M`, `launch_size`) has `program_id(0) = p mod gx` and
  `program_id(1) = p / gx` (`launch_pid0`, `launch_pid1`). Since `N_CTX = gx·BLOCK_M` (`ctx_eq`),
  its lane `i` handles row `p·BLOCK_M + i = (p / gx)·N_CTX + (p mod gx)·BLOCK_M + i`.
- Footprints: every program reads the elements `inSet` of `O` and `DO` (`R`) and writes its tile of
  `Delta`, the rows `p·BLOCK_M + i` (`W`, a `Buf.tile`).
- `hsafe` (`safe_program`), line by line:
  - lines 258–259: `safe_load_buf` with `Buf.addrs_of_get_all` (no mask): lane `(i, k)` points at
    element `(p·BLOCK_M + i)·HEAD_DIM + k` (`ptr_in`; the `int32` arithmetic does not wrap by
    `ctx_lt`, `row_fit`, `head_fit`, `prog_bounds`), below `Z·H·N_CTX·HEAD_DIM ≤ O.len`; each lane
    gets its element of the initial memory;
  - line 259's conversion: `safe_cvt` (`float16` and `float32` are standard formats, `.rn`);
  - line 260: `safe_convert` (a result exists, `conv_ex`), `safe_mul`, `safe_sum1c` (`HEAD_DIM > 0`),
    each giving what the result may be;
  - line 262: `safe_last` and `safe_store_buf`: lane `i` writes element `p·BLOCK_M + i` of `Delta`
    (`ptr_out`), distinct lanes distinct elements, inside the program's tile;
  - each row is then done by `row_bound` (`Err.lean`): the converted values are finite with the
    inputs' values (`conv_widen`, `widen`, from `FloatV.toRat_cvt_of_wider`), each product is an IEEE
    `float32` product (`tl.MulRel`), and the reduction's tree covers the `HEAD_DIM` products, so
    `sum_bcast_mul_tree_err` gives a finite result within `bcastTreeBound`, which is `tol`
    (`tol_eq`); the room it needs holds because a finite `float16` is at most `65504`, so
    `Σ|o·do| ≤ 256 · 2^32 = 2^40` (`room`, with `tol_le`).
- `hdisj`: program `p`'s tile of `Delta` is disjoint from every other program's tile
  (`Buf.tile_disjoint`) and from `O`'s and `DO`'s elements (`W_R`, from `Delta_O`, `Delta_DO`).
- `hRW`: the same, `W_R`.
- `hpost`: row `r = bh·N_CTX + m < Z·H·N_CTX` lies in the tile of program `r / BLOCK_M`
  (`tile_div`), a program of the grid; the final memory agrees with that program's memory on its
  footprint, so `OutSpec` carries over (`outSpec_of_agree`). A byte outside `Delta`'s elements
  `outSet` is in no write footprint (`W_sub`), so it keeps its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.any` gives a total behaviour satisfying
`faithful`, the only field.

`satisfiable`: an instance of the source's test (lines 631–647), `Z = 1`, `H = 2`, `N_CTX = 128`,
`HEAD_DIM = 64`, `BLOCK_M = 128`, with `o` and `do` of `16384` elements and `delta` of `256` at the
addresses `2^21`, `2^22` and `3·2^21` (`testInst`), in a memory where every byte is present and
`0` (`testMem`), meets `Inst.ok` and `Pre`.

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float16` inputs, `float32` output.** The folder is for the tutorial's backward pass, whose
  `o` and `do` are `float16` and `delta` `float32`; the tutorial does not run the backward pass in
  `float8e5` (line 643).
- **`HEAD_DIM ≤ 256`.** The host admits `16 … 256` (line 512). The tolerance grows with
  `HEAD_DIM` (the sum's tree depth, `(3·HEAD_DIM − 2)·2^-24` relative to first order), and it is
  stated and bounded up to `256` (`tol_le`); larger head dimensions are not covered.
- **`BLOCK_M` divides `N_CTX`.** The host asserts it. Otherwise the grid `N_CTX // BLOCK_M` leaves
  the last positions without a program (the kernel has no mask), and `delta` there would not be
  computed: the claim would be false, so the condition is the kernel's contract.
- **The `int32` limits** (`ctx_lt`, `row_fit`, `head_fit`). Triton passes `N_CTX` as an `int32`
  only below `2^31` (as `int64` from `2^31` on, another model), and the kernel's `int32` products
  wrap beyond `row_fit` and `head_fit`. The host's largest instances are far inside them
  (Conditions).
- **The grid** (`bh_lt`, `grid_lt`). CUDA rejects a grid whose axis 1 is `2^16` or more, and
  `Launch.Valid` requires fewer than `2^31` programs; the host's largest `Z·H` is `192` (the
  test's `Z = 4`, `H = 48`, with `N_CTX = 4096`: `32 · 192` programs), and its largest grid has
  `128 · 128` programs (the benchmark's `N_CTX = 16384`, with `Z·H = 4 · 32 = 128`).
- **No overlap of `delta` with `o` or `do`** (`Delta_O`, `Delta_DO`). The host always allocates a
  fresh `delta`. An overlap would make a program write bytes that it or another program reads, a
  conflict the library does not order and no claim covers
  ([SEMANTICS.md](../../docs/SEMANTICS.md), "Programs"; [exclusions.md](../../docs/exclusions.md);
  library issue L15). The conditions are stated on the elements the launch accesses, so overlap
  elsewhere is allowed.
- **Non-finite inputs.** For a row with `±∞` or NaN in `o` or `do`, the exact sum is infinite or
  undefined, and `Post` says only that a `float32` value is stored there (it is still covered: the
  launch ends normally and changes nothing else). The host's inputs are finite (`o` is the forward's
  output on normally distributed `q, k, v`; `do` is `torch.randn_like(…)`).
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host (an
  H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment).

Modelling decisions:

- `BLOCK_M` ranges over every power of two and `HEAD_DIM` over every power of two up to `256`,
  within Triton's block limit, not only the host's `128` and `16 … 256`; `Z`, `H` and `N_CTX` over
  every value the conditions allow, not only the script's.
- The three integers are `int32` (`IntV.ofNat`): `N_CTX` as Triton passes it below `2^31` (and the
  constant `1` gives the same model, Conditions); `Z` and `H` are never read, so their type is
  immaterial. The code a launch runs is the code Triton 3.8.0 compiles for that launch's own argument
  values (`Compiler.Faithful`'s point 11, in part (B), assumed).
- Line 258's pointer expression is written as the source evaluates it, left to right:
  `O + (off_hz * HEAD_DIM) * N_CTX`, then `+ off_m[:, None] * HEAD_DIM`, then `+ off_n[None, :]`,
  each `+` advancing the pointer by its `int32` offset; the broadcast of `[BLOCK_M, 1]` against
  `[1, HEAD_DIM]` is written on the operands (`off_m.col HEAD_DIM`, `off_n.row BLOCK_M`), which gives
  the same lanes since `*` and the pointer `+` act lane by lane. Each constexpr in an operation with
  an `int32` tensor is built in `int32` (SEMANTICS.md, "Constants"). Line 259 repeats the arithmetic,
  as the source does.
- The loads have no mask and no `other`, as in the source: every lane is enabled and in bounds on
  covered instances. The store has no mask either.
- `.to(tl.float32)` is `tl.cvt` with the default rounding (exact here); in `o * do` the
  `float16` block `o` is converted to `float32` (`tl.convert .f16 .f32`, the dtype `BinOp.typing?`
  gives), then the IEEE `float32` product. `tl.sum(o * do, axis=1)` is `tl.sum1c`: it reduces a
  product, so the compiler may contract its first additions with the products (`enable_fp_fusion`
  is on). It is not the combine pass's broadcast-multiply-sum (`tl.sum_bcast_mul`), whose operands
  are expanded along new axes: here two blocks of one shape are multiplied lane by lane.
- The source's local name `do` is a keyword in Lean, so the model writes `«do»`.
- The launch lists the source's block shapes `[BLOCK_M]`, `[HEAD_DIM]`, `[BLOCK_M, 1]`,
  `[1, HEAD_DIM]` and `[BLOCK_M, HEAD_DIM]`; the grid is the host's `(N_CTX // PRE_BLOCK,
  BATCH * N_HEAD)` with axis 2 of size `1`; `enable_fp_fusion` is Triton's default `true`.
