# Tutorial coverage, round 10

## 01-vector-add.py: add_kernel

| Construct | Status | Where |
|---|---|---|
| `pid = tl.program_id(axis=0)` on a 1-D grid | covered | SEMANTICS "Launches" (`program_id`), "Index arithmetic"; `TVA.tl.program_id`; TRANSCRIPTION §2 |
| `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` (`int32`, wrapping) | covered | SEMANTICS "Index arithmetic", "Constants" (the constexpr computes in `int32`); `TVA.tl.arange`; `TVA.Block` (`Fits`, `fits_of_grid`) |
| `mask = offsets < n_elements`, with `n_elements` an `int32` scalar or the constexpr `1` | covered | SEMANTICS "Operators and promotion" (a comparison makes a Python number an `int32` tensor first); `BinOp.typing?`, `Tensor.ltS` |
| `tl.load(x_ptr + offsets, mask=mask)` without `other`, `float32` | covered | SEMANTICS "tl.load" (masked-off lanes unspecified); `TVA.tl.load`, `Tensor.ptrAdd`; `safe_load_block` |
| `output = x + y`, no product feeds it | covered | SEMANTICS "Float + and -: IEEE or fusable"; `TVA.tl.add` |
| `tl.store(output_ptr + offsets, output, mask=mask)`, same dtype | covered | SEMANTICS "tl.store"; `TVA.tl.store`; `safe_store_block` |
| pointer arguments (`p + i` advances `i` elements); the output a buffer of its own | covered | SEMANTICS "Values" (`Ptr`), "Pointers"; `Ptr.offset`; `TVA.independent`, `Buf` |
| `n_elements`: `int32` below `2^31`, the constexpr `1`, `int64` from `2^31` | covered | SEMANTICS "Scalar arguments" (L29); `Inst.ok` fixes the class |
| `BLOCK_SIZE: tl.constexpr` (`1024`) | covered | SEMANTICS "Values" (constexprs are Lean values); `Launch.blocks`, `Shape.TritonOk` |
| launch `add_kernel[grid](…, BLOCK_SIZE=1024)`, grid `(cdiv(n_elements, BLOCK_SIZE),)`, no option | covered | SEMANTICS "Launches"; `Launch.grid1`, `Launch.valid_grid1`; rules/compiler-faithful.md "Scope" |

## 02-fused-softmax.py: softmax_kernel

| Construct | Status | Where |
|---|---|---|
| `row_start = tl.program_id(0)`, `row_step = tl.num_programs(0)` | covered | SEMANTICS "Launches"; `TVA.tl.program_id`, `TVA.tl.num_programs` |
| `for row_idx in tl.range(row_start, n_rows, row_step, …)`: a runtime start and step | covered | SEMANTICS "Loops" (`tl.forRange`, `tl.RangeOk`: the step is at least `1` in a program that runs); `safe_forRange` |
| `num_stages=num_stages` of `tl.range`, from the kernel's constexpr parameter of that name | covered | SEMANTICS "Performance-only arguments", "Loops" (pipelining reorders only a program's own accesses; the program reads `input` and writes `output`, distinct buffers, so it has no conflict) |
| `input_ptr + row_idx * input_row_stride + col_offsets` (`int32`) | covered | SEMANTICS "Index arithmetic", "Pointers"; `Ptr` `+` `IntV`, `Tensor.ptrAdd` |
| `-float('inf')`: Python's builtin `float` called in the kernel | gap | NEW: SEMANTICS does not say what a call of one of Python's builtins in a kernel gives (`float`, `int`, `isinstance`, `len`, …: Python's own result on the unwrapped arguments, a compile-time value, no operation emitted; code-generator.py:357-365, 1444-1448, 1639-1659), and "Expressions with no effect" counts every call as possibly effectful (work/probes.out P2) |
| `tl.load(input_ptrs, mask=mask, other=-inf)` | covered | SEMANTICS "tl.load", "Constants" (`other=`: a `float32` tensor of its own type, `FloatV.litF32`) |
| `tl.max(row, axis=0)` of a rank-1 `float32` tensor | covered | SEMANTICS "tl.sum, tl.max, tl.min (reductions)"; `TVA.tl.maxAll`, `FloatV.MaxRel`; `safe_maxAll` |
| `row - tl.max(row, axis=0)`: a scalar broadcast, no product feeds it | covered | SEMANTICS "Operators on tensors" (`bc`), "Float + and -" (IEEE); `TVA.tl.sub` |
| `tl.exp(row_minus_max)` on `float32`, and `exp(-inf) = +0` on the masked-off lanes | covered | SEMANTICS "tl.exp, exp2, …" (`P.math`); rules/num.md `ExpF32` (on `[-X, X]`), `ExpF32NegInf`; `TVA.tl.exp` |
| `tl.sum(numerator, axis=0)`: IEEE (the result of `tl.exp` is not a product) | covered | SEMANTICS reductions, "Float + and -"; `TVA.tl.sum`; `TVA.Lemmas.Sum` |
| `numerator / denominator` on `float32` | covered | SEMANTICS "/ on floats"; rules/num.md `DivFull`, `DivTotal`; `TVA.tl.div` |
| `tl.store(output_ptrs, softmax_output, mask=mask)` | covered | SEMANTICS "tl.store"; `TVA.tl.store` |
| the strides, `n_rows`, `n_cols`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch through a `CompiledKernel`: `softmax_kernel.warmup(…, num_stages=…, num_warps=8, grid=(1,))`, then `kernel[(num_programs, 1, 1)](…)` with the warm-up's own arguments, constexprs passed by position | covered | SEMANTICS "Launches" (a precompiled kernel's specialisation, L47); rules/compiler-faithful.md Scope 6. The warm-up's options are `num_warps` and `num_stages` (the keyword reaches both the parameter and the option, work/probes.out P3), both allowed by Scope 4, so L72 (options given at `warmup`) does not bear on this host |
| grid `(min(NUM_SM · occupancy, n_rows), 1, 1)`, possibly `0` | covered | SEMANTICS "Launches" (a zero grid ends `ok` with memory unchanged); `Launch.Valid` |

## 03-matrix-multiplication.py: matmul_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])`: benchmark runs on the caller's buffers, then one launch | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md "Which launch"; `Inst` ranges over the configs |
| `num_warps`, `num_stages` of each CUDA `triton.Config` | covered | rules/compiler-faithful.md Scope 4; SEMANTICS "Performance-only arguments" (`num_stages` reorders only; A and B are read, C written) |
| the HIP configs' `matrix_instr_nonkdim` | excluded | exclusions.md (targets other than the H200) |
| `tl.cdiv(M, BLOCK_SIZE_M)`: Triton's `@jit` `(x + (div - 1)) // div` | covered | SEMANTICS "Helpers are inlined" (primitives/triton-3.8.0-standard.py:32-43), "Integer //, %, <<, >>" (`tl.idiv`); `safe_idiv1` |
| `GROUP_SIZE_M * num_pid_n`, `pid // num_pid_in_group`, `pid % num_pid_in_group`, `% group_size_m` | covered | SEMANTICS "Constants" (a constexpr with an `int32` scalar computes in `int32`), "Integer //, %, <<, >>"; `tl.idiv`, `tl.irem`; `safe_irem1` |
| builtin `min(num_pid_m - first_pid_m, GROUP_SIZE_M)` | covered | SEMANTICS "Operators and promotion" (Python's `min` folds `tl.minimum`; on integers `tl.iminimum`); TRANSCRIPTION §2 |
| `tl.assume(pid_m >= 0)` … `tl.assume(stride_cn > 0)`, also on a stride that is the constexpr `1` | covered | SEMANTICS "Compiler hints" (`tl.assume`: undefined behaviour if false); `TVA.tl.assume`; `safe_assume` |
| `(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M` | covered | SEMANTICS "Integer //, %, <<, >>" (`tl.irem`, undefined for a zero divisor) |
| `a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)`; `c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]` | covered | SEMANTICS "Operators on tensors" (`x[:, None]`, `.col`/`.row`, broadcasting written in the model), "Index arithmetic", "Pointers"; `Tensor.ptrAddT` |
| `accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)` | covered | `TVA.tl.zeros` |
| `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))` carrying `accumulator`, `a_ptrs`, `b_ptrs` | covered | SEMANTICS "Loops" (`tl.forRange`, the carried tuple); `safe_forRange` |
| `tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)`: a `[1, BK]` mask broadcast to the block, `other` converted to `float16` | covered | SEMANTICS "tl.load", "Constants" (`other=`), "Float conversions"; `safe_load_buf` |
| the same loads with `float8e5` operands (the tutorial's fp8 run): `other=0.0` through `tl.cvt8` | covered | SEMANTICS "Float conversions" (`tl.cvt8`); rules/num.md `CvtF32E5m2` (`0` gives `+0`) |
| `tl.dot(a, b, accumulator)`, `float16` operands into `float32` | covered | SEMANTICS "tl.dot" (`DotCfg`: `tf32` by default, `maxImpreciseAcc = 0`); rules/num.md `DotF16`; `TVA.tl.dot`; `safe_dot`; `TVA.Lemmas.DotLoop` |
| `tl.dot` with `float8e5` operands (`maxImpreciseAcc = 2^30` by default) | covered | SEMANTICS "tl.dot"; rules/num.md `DotE5m2` |
| `a_ptrs += BLOCK_SIZE_K * stride_ak`, `b_ptrs += BLOCK_SIZE_K * stride_bk` | covered | SEMANTICS "Pointers" |
| `if ACTIVATION == "leaky_relu":` on a constexpr string | covered | SEMANTICS "if, elif, else on a compile-time condition"; TRANSCRIPTION §4 `config` |
| helper `leaky_relu(accumulator)` | covered | SEMANTICS "Helpers are inlined" (its section below) |
| `accumulator.to(tl.float16)` | covered | SEMANTICS "Float conversions" (`tl.cvt`, nearest even) |
| `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)`; `tl.store(c_ptrs, c, mask=c_mask)` | covered | SEMANTICS "Operators on tensors" (`Tensor.and`), "tl.store" |
| `M`, `N`, `K`, the strides: `int32`, or the constexpr `1` (the unit strides; `stride_bk = 1` in the fp8 run, where `b` is a transpose) | covered | SEMANTICS "Scalar arguments" |
| launch: 1-D grid `cdiv(M, BM) · cdiv(N, BN)`, `ACTIVATION=` a constexpr | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 03-matrix-multiplication.py: leaky_relu

| Construct | Status | Where |
|---|---|---|
| `x >= 0` on a `float32` tensor with the literal `0` | covered | SEMANTICS "Operators and promotion" (a comparison makes `0` an `int32` tensor, then `float32` `+0`); `Tensor.fle`. PROVER_GUIDE's stale rule (L57) gives the same `+0` here |
| `0.01 * x` | covered | SEMANTICS "Constants" (`FloatV.litF32`); `TVA.tl.mul` |
| `tl.where(x >= 0, x, 0.01 * x)` | covered | `TVA.tl.where` (both sides computed); SEMANTICS "Operators and promotion" (`whereTyping?`) |
| `return` as the last statement | covered | SEMANTICS "Helpers are inlined", "`return`" |

## 04-low-memory-dropout.py: _dropout

| Construct | Status | Where |
|---|---|---|
| `pid`, `offsets`, `mask` as in tutorial 01 | covered | SEMANTICS "Index arithmetic", "Operators and promotion"; `TVA.Block` |
| `tl.load(x_ptr + offsets, mask=mask)` (`float32`), `tl.load(x_keep_ptr + offsets, mask=mask)` (`int32`) | covered | SEMANTICS "tl.load"; `safe_load_block` |
| `p`, a Python `float` argument: a runtime `float32` | covered | SEMANTICS "Scalar arguments" (L29) |
| `1 - p`: the `int` `1` built in `float32`, an IEEE subtraction | covered | SEMANTICS "Constants", "Operators and promotion", "Float + and -"; `TVA.tl.sub` |
| `x / (1 - p)`: a scalar broadcast | covered | SEMANTICS "/ on floats"; rules/num.md `DivFull`, `DivTotal`; `TVA.tl.div` |
| `tl.where(x_keep, …, 0.0)` with the `int32` tensor `x_keep` as the condition | covered | SEMANTICS "Operators and promotion" (`tl.where`: `x_keep != 0`); `whereTyping?`, `tl.convert .i32 .i1`; TRANSCRIPTION §2 |
| `0.0` as a value of `tl.where` (`float32` `+0`) | covered | SEMANTICS "Constants" |
| `tl.store(output_ptr + offsets, output, mask=mask)` | covered | SEMANTICS "tl.store" |
| `n_elements`: `int32`, the constexpr `1`, or `int64` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(cdiv(n_elements, 1024),)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 04-low-memory-dropout.py: _seeded_dropout

| Construct | Status | Where |
|---|---|---|
| `pid`, `offsets`, `mask`; `tl.load(x_ptr + offsets, mask=mask)` | covered | SEMANTICS "Index arithmetic", "tl.load" |
| `tl.rand(seed, offsets)` (Philox 4x32-10, then `uint_to_uniform_float`) | covered | SEMANTICS "Random numbers"; `TVA.tl.rand`, `Philox.randint4x` (a seed of any integer type, `seed.to(uint64)`); `safe_rand` |
| `seed`, `n_elements` (`int32`, the constexpr `1`, or `int64`) and `p` (`float32`) | covered | SEMANTICS "Scalar arguments" |
| `x_keep = random > p`: an ordered `float32` comparison; no addition consumes the product | covered | SEMANTICS "Operators and promotion", "Random numbers"; `Tensor.flt` |
| `tl.where(x_keep, x / (1 - p), 0.0)` with a boolean condition | covered | `TVA.tl.where`; SEMANTICS "Constants", "/ on floats" (`DivFull`) |
| `tl.store(output_ptr + offsets, output, mask=mask)` | covered | SEMANTICS "tl.store" |
| launch: grid `(cdiv(n_elements, 1024),)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 05-layer-norm.py: _layer_norm_fwd_fused

| Construct | Status | Where |
|---|---|---|
| `Y += row * stride`, `X += row * stride`: pointer arguments advanced by an `int32` | covered | SEMANTICS "Pointers", "Index arithmetic" |
| `mean = 0` before the loops, overwritten before any use | covered | SEMANTICS "Expressions with no effect"; TRANSCRIPTION §4 `dead` (under L59, open, the name holds an `int32` scalar; it is unused either way) |
| three loops `for off in range(0, N, BLOCK_SIZE)` (runtime bound), carrying `_mean`, `_var` | covered | SEMANTICS "Loops"; `safe_forRange` |
| `tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)`: `float16`, `other` `0`, then an exact widening | covered | SEMANTICS "tl.load", "Constants" (`other=`), "Float conversions" (`tl.cvt`) |
| `_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)`; `_mean += a` (no product feeds it) | covered | `TVA.tl.zeros`; SEMANTICS "Float + and -"; `TVA.tl.add` |
| `tl.sum(_mean, axis=0) / N`: a `float32` scalar by the `int32` `N` (or the constexpr `1`) | covered | SEMANTICS "Operators and promotion" (`BinOp.typing? .div`: `N` converted to `float32` by `tl.itof`), "/ on floats"; rules/num.md `DivFull`; `tl.convert` |
| `x = tl.where(cols < N, x - mean, 0.)` | covered | `TVA.tl.where`, `TVA.tl.sub`; SEMANTICS "Constants" |
| `_var += x * x` (a product feeds the addition) | covered | SEMANTICS "Float + and -" (`tl.mul_add`); `TVA.Lemmas.Fused` |
| `tl.sum(_var, axis=0)`: IEEE (`_var` holds sums, not products) | covered | SEMANTICS reductions; `TVA.tl.sum` |
| `eps`, a Python `float` argument (`float32`), in `var + eps` | covered | SEMANTICS "Scalar arguments", "Float + and -" |
| `tl.sqrt(var + eps)` on `float32` | covered | SEMANTICS "tl.exp, exp2, …, sqrt"; rules/num.md `SqrtF32` (on `sqrtDom`; `var + eps` is at least `eps`, above `2^-126`); `TVA.tl.sqrt` |
| `rstd = 1 / tl.sqrt(…)` | covered | SEMANTICS "Constants", "/ on floats"; rules/num.md `DivFull` |
| `tl.store(Mean + row, mean)`, `tl.store(Rstd + row, rstd)`: scalar stores | covered | SEMANTICS "tl.store"; `TVA.tl.store1` |
| `tl.load(W + cols, mask=mask)`, `tl.load(B + cols, mask=mask)` without `other` | covered | SEMANTICS "tl.load" |
| `x_hat = (x - mean) * rstd`; `y = x_hat * w + b` with `float16` `w`, `b` | covered | SEMANTICS "Operators and promotion" (`w`, `b` converted to `float32`), "Float + and -" (`tl.mul_add`); TRANSCRIPTION §2 (`x * w + b`) |
| `tl.store(Y + cols, y, mask=mask)`: `float32` into `float16` | covered | SEMANTICS "tl.store", "Float conversions" |
| `stride`, `N`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(M,)`, `num_warps`, `num_ctas=1` | covered | SEMANTICS "Launches" (`Launch.numCtas`); rules/compiler-faithful.md Scope 4 |

## 05-layer-norm.py: _layer_norm_bwd_dx_fused

Excluded: L37 (user decision).

## 05-layer-norm.py: _layer_norm_bwd_dwdb

| Construct | Status | Where |
|---|---|---|
| `cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)` | covered | SEMANTICS "Index arithmetic" |
| `tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)`, twice | covered | `TVA.tl.zeros` |
| `for i in range(0, M, BLOCK_SIZE_M)` carrying `dw`, `db` | covered | SEMANTICS "Loops"; `safe_forRange` |
| `mask = (rows[:, None] < M) & (cols[None, :] < N)`; `offs = rows[:, None] * N + cols[None, :]` (`int32`) | covered | SEMANTICS "Operators on tensors", "Index arithmetic" |
| `dw += tl.load(DW + offs, mask=mask, other=0.)`: a `float32` accumulator plus a `float16` block | covered | SEMANTICS "Operators and promotion" (the `float16` block converted to `float32`), "tl.load", "Constants" (`other=`), "Float + and -" (IEEE) |
| `tl.sum(dw, axis=0)`, `tl.sum(db, axis=0)` | covered | SEMANTICS reductions; `TVA.tl.sum0` |
| `tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)`: `float32` into `float16` | covered | SEMANTICS "tl.store", "Float conversions" |
| the inputs `_dw`, `_db`, written by the excluded `_layer_norm_bwd_dx_fused` | covered | SEMANTICS "One launch" (a claim is about one launch, from its `init`) |
| `M`, `N`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(cdiv(N, 128),)`, `num_ctas=1` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 06-fused-attention.py: _attn_fwd_inner

| Construct | Status | Where |
|---|---|---|
| `if STAGE == 1: … elif STAGE == 2: … else:`, `if dtype == tl.float8e5:`, `if STAGE == 2:` on constexprs | covered | SEMANTICS "if, elif, else on a compile-time condition"; TRANSCRIPTION §4 `config` |
| `lo, hi = 0, start_m * BLOCK_M` and `lo, hi = 0, N_CTX`: a Python `0` assigned to a name | covered | SEMANTICS "Constants", "Loops"; under L59 (open) `lo` holds an `int32` scalar, which gives the same `offsetk_y = offset_y + lo` and the same `tl.range` bounds |
| `lo = tl.multiple_of(lo, BLOCK_M)` on the scalar `start_m * BLOCK_M`, before the loop | covered | SEMANTICS "Compiler hints" (a value an operation defines: a check there); `TVA.tl.multiple_of1`; `safe_multiple_of1` |
| `for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize)` | covered | SEMANTICS "Loops", "Performance-only arguments" (`warp_specialize` reorders a program's accesses only); `tl.forRange` |
| `start_n = tl.multiple_of(start_n, BLOCK_N)` on the induction variable | covered | SEMANTICS "Compiler hints: which value" (no effect, the model drops it; L1); TRANSCRIPTION §4 `no-op:hint` |
| `desc_k.load([offsetk_y, 0]).T`, `desc_v.load([offsetv_y, 0])`, and for fp8 `desc_v.load([0, offsetv_y]).T`, on host descriptors | covered | SEMANTICS "Tensor descriptors" (`tl.desc_load`, zero padding, `TensorDesc.Access`), "Shape changes" (`x.T`); `safe_desc_load` |
| the same on descriptors built in the kernel (`_maybe_make_tensor_desc` given pointers: the H200 with `warp_specialize=True`) | covered | SEMANTICS "Tensor descriptors" (`tl.make_tensor_descriptor`, pure); rules/compiler-faithful.md Scope 5. L5 (open) adds `0 < shape` for an access; every shape entry is at least `1` when a program runs and `Z · H · N_CTX < 2^31`, which the claim's `Inst.ok` needs for its `int32` arithmetic anyway |
| parameters annotated `tl.constexpr` that receive tensors (`offs_m`, `offs_n`, `N_CTX`) or a dtype (`dtype`) | covered | SEMANTICS "Helpers are inlined" (the body with the call's arguments) |
| `qk = tl.dot(q, k)`, `acc = tl.dot(p, v, acc)` with `float16` operands | covered | SEMANTICS "tl.dot"; rules/num.md `DotF16`; `TVA.tl.dot` |
| the same with `float8e5` operands (`FP8_OUTPUT`) | covered | rules/num.md `DotE5m2` |
| `mask = offs_m[:, None] >= (start_n + offs_n[None, :])` | covered | SEMANTICS "Operators on tensors", "Operators and promotion" |
| `qk * qk_scale + tl.where(mask, 0, -1.0e6)`: both values of `tl.where` Python numbers | covered | SEMANTICS "Operators and promotion" (`whereTyping?` gives `float32`), "Constants", "Float + and -" (`tl.mul_add`) |
| `tl.max(qk, 1)`; `tl.maximum(m_i, …)` on `float32` | covered | SEMANTICS reductions, "tl.maximum, tl.minimum"; `TVA.tl.max1`, `TVA.tl.maximum`, `FloatV.MaxRel` |
| `qk -= m_ij[:, None]` (IEEE: `qk` is a sum there) and `qk * qk_scale - m_ij[:, None]` (fusable) | covered | SEMANTICS "Float + and -"; `TVA.tl.sub`, `TVA.tl.subc` |
| `tl.math.exp2(qk)`, `tl.math.exp2(m_i - m_ij)` (`m_i = -inf` at the first step) | covered | rules/num.md `Exp2F32`, `Exp2F32NegInf`; `TVA.tl.exp2` |
| `l_ij = tl.sum(p, 1)`: IEEE (`p` is not a product) | covered | SEMANTICS reductions; `TVA.tl.sum1` |
| the branch `not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128` (not taken on the H200): `BM: tl.constexpr = acc.shape[0]`, `reshape`, `permute`, `split`, `tl.join` | covered | SEMANTICS "Order" (a compile-time `and`), "Shape changes"; `Tensor.reshape`, `Tensor.permute`, `Tensor.split`, `Tensor.join` |
| `acc = acc * alpha[:, None]` | covered | SEMANTICS "Operators on tensors"; `TVA.tl.mul` |
| `p.to(dtype)` into `float16` | covered | SEMANTICS "Float conversions" (`tl.cvt`) |
| `p.to(tl.float8e5)`, `p` in `[0, 1]` | covered | SEMANTICS "Float conversions" (`tl.cvt8`); rules/num.md `CvtF32E5m2` |
| `l_i = l_i * alpha + l_ij`; `m_i = m_ij`; `offsetk_y += BLOCK_N` | covered | SEMANTICS "Float + and -" (`tl.mul_add`), "Index arithmetic" |
| `return acc, l_i, m_i` at the end | covered | SEMANTICS "Helpers are inlined" |

## 06-fused-attention.py: _maybe_make_tensor_desc

| Construct | Status | Where |
|---|---|---|
| `if isinstance(desc_or_ptr, tl.tensor_descriptor):`: Python's builtin `isinstance` on a kernel value | gap | NEW: as in tutorial 02, SEMANTICS does not say what a builtin call gives; here a compile-time `bool` from the argument's Triton type (`True` for a host `TensorDescriptor`, `False` for a pointer), so only one branch is visited (work/probes.out P2) |
| a host `TensorDescriptor` returned unchanged | covered | SEMANTICS "Tensor descriptors" (the host's shape, strides, block shape and padding) |
| `tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)` with the caller's lists (`int32` scalars and constexprs) | covered | SEMANTICS "Tensor descriptors"; `TVA.tl.make_tensor_descriptor` (shape `int32`, strides `int64`); `safe_make_tensor_descriptor`; rules/compiler-faithful.md Scope 5 (L5's open zero-shape case does not arise, see `_attn_fwd_inner`) |
| a `return` in each branch of a compile-time `if` | covered | SEMANTICS "if, elif, else on a compile-time condition", "Helpers are inlined" |

## 06-fused-attention.py: _attn_fwd

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, prune_configs_by={'early_config_prune': prune_invalid_configs})` over the configs `keep` leaves | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md "Which launch" |
| each config's `pre_hook=_host_descriptor_pre_hook`, which sets the host descriptors' `block_shape` before the launch | gap | L61 (SEMANTICS "One launch" and Faithful's "Which launch" speak of the memory after the last hook, not of the arguments a hook changes) |
| `num_warps`, `num_stages` of each config | covered | rules/compiler-faithful.md Scope 4 |
| the AMD options `waves_per_eu`, `allow_flush_denorm`, and Blackwell's `maxnreg` path | excluded | exclusions.md (targets other than the H200) |
| `dtype = tl.float8e5 if FP8_OUTPUT else tl.float16` | covered | SEMANTICS "Order" (`x if c else y` on a compile-time `c`) |
| `tl.static_assert(BLOCK_N <= HEAD_DIM)` | covered | SEMANTICS "tl.static_assert"; TRANSCRIPTION §4 `compile-check` |
| `tl.program_id(0)`, `tl.program_id(1)` on a 2-D grid | covered | SEMANTICS "Launches" (`program_id = (i % gx, i / gx % gy, …)`) |
| `off_hz // H`, `off_hz % H`, `Z * H * N_CTX`, `offset_y` (`int32`; `Z = 1`, a constexpr, in the tests) | covered | SEMANTICS "Integer //, %, <<, >>", "Index arithmetic", "Scalar arguments"; `safe_idiv1`, `safe_irem1` |
| helper `_maybe_make_tensor_desc` (its `isinstance`), and `m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")` (Python's `float`) | gap | NEW: Python's builtins in a kernel, as in tutorial 02 (the subtraction itself is SEMANTICS "Constants", "Float + and -"; `Softmax.sub_zero_posInf`) |
| descriptors built in the kernel: `triton.set_allocator(alloc_fn)` returns a fresh buffer | covered | SEMANTICS "Tensor descriptors"; rules/compiler-faithful.md Scope 5 |
| `l_i = tl.zeros(…) + 1.0`, `acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)` | covered | SEMANTICS "Constants" (`FloatV.litF32`), "Float + and -"; `TVA.tl.zeros` |
| `sm_scale`, a Python `float` argument (`float32`); `qk_scale = sm_scale`, `qk_scale *= 1.44269504` | covered | SEMANTICS "Scalar arguments", "Constants"; `TVA.tl.mul` |
| `q = desc_q.load([qo_offset_y, 0])` | covered | SEMANTICS "Tensor descriptors"; `safe_desc_load` |
| `if STAGE & 1:`, `if STAGE & 2:` (a constexpr `&`) | covered | SEMANTICS "Expressions with no effect", "if, elif, else on a compile-time condition" |
| helper `_attn_fwd_inner`, called with `4 - STAGE` and with `2` | covered | SEMANTICS "Helpers are inlined" (its section above) |
| `m_i += tl.math.log2(l_i)` on `float32` (`l_i` at least `1`) | covered | rules/num.md `Log2F32`; SEMANTICS "Float + and -" (IEEE: neither operand is a product); `TVA.tl.log2` |
| `acc = acc / l_i[:, None]` | covered | SEMANTICS "/ on floats"; rules/num.md `DivFull` |
| `tl.store(m_ptrs, m_i)` with `m_ptrs = M + off_hz * N_CTX + offs_m` | covered | SEMANTICS "tl.store" |
| `desc_o.store([qo_offset_y, 0], acc.to(tl.float16))` | covered | SEMANTICS "Tensor descriptors" (`tl.desc_store`), "Float conversions"; `safe_desc_store` |
| `acc.to(tl.float8e5)` for the fp8 output (`acc` may exceed `maxVal`: `p` is rounded to fp8 after `l_i` is summed) | gap | L64 (saturation beyond `maxVal` is not stated) |
| `Z`, `H`, `N_CTX`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(cdiv(N_CTX, BLOCK_M), Z · H, 1)`, keyword constexprs (`IS_HOPPER`, `warp_specialize`, …) | covered | SEMANTICS "Launches" (`y < 2^16`); `Launch.Valid` |

## 06-fused-attention.py: _attn_bwd_preprocess

| Construct | Status | Where |
|---|---|---|
| `off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)`, `off_hz = tl.program_id(1)` | covered | SEMANTICS "Launches", "Index arithmetic" |
| unmasked loads `tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])`, and the same of `DO` with `.to(tl.float32)` | covered | SEMANTICS "tl.load", "Pointers", "Float conversions" |
| `o * do`: a `float16` block times a `float32` block | covered | SEMANTICS "Operators and promotion" (`o` converted to `float32`); `tl.convert`, `TVA.tl.mul` |
| `delta = tl.sum(o * do, axis=1)`: a sum of products | covered | SEMANTICS reductions (`tl.sum1c`); `TVA.Lemmas.SumFused` (L33) |
| `tl.store(Delta + off_hz * N_CTX + off_m, delta)` | covered | SEMANTICS "tl.store" |
| `N_CTX`: `int32`, or the constexpr `1` (`Z`, `H` unused) | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(N_CTX // 128, BATCH · N_HEAD)` | covered | SEMANTICS "Launches" |

## 06-fused-attention.py: _attn_bwd_dkdv

| Construct | Status | Where |
|---|---|---|
| `Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d`, on a pointer the caller advanced by an `int64`; `stride_d = 1`, a constexpr | covered | SEMANTICS "Pointers", "Index arithmetic", "Scalar arguments" |
| `tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)` | covered | SEMANTICS "tl.static_assert" |
| `curr_m = start_m`, `step_m = BLOCK_M1` (a constexpr assigned to a name), `curr_m` carried by the loop | covered | SEMANTICS "Loops"; under L59 (open) both names hold `int32` scalars, which gives the same values here (nonnegative, below `2^31`) |
| `for blk_idx in range(num_steps)` (`num_steps` an `int32` scalar) | covered | SEMANTICS "Loops"; `safe_forRange` |
| unmasked loads `tl.load(qT_ptrs)`, `tl.load(M + offs_m)`, `tl.load(do_ptrs)`, `tl.load(D + offs_m)` | covered | SEMANTICS "tl.load" |
| `tl.dot(k, qT)`, `tl.dot(v, tl.trans(do))`, `float16` into `float32` | covered | rules/num.md `DotF16` |
| `qkT - m[None, :]`, `dpT - Di[None, :]` (a dot feeds them), with `.to(tl.float32)` of a `float32` dot | covered | SEMANTICS "Float + and -" (`tl.subc`), "Float conversions" |
| `tl.math.exp2(qkT - m[None, :])` | covered | rules/num.md `Exp2F32` |
| `if MASK:` (a constexpr): `tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)` | covered | SEMANTICS "if, elif, else on a compile-time condition", "Constants"; `TVA.tl.where` |
| `ppT.to(tl.float16)`, `dsT.to(tl.float16)`; `dsT = pT * (dpT - Di[None, :])` | covered | SEMANTICS "Float conversions"; `TVA.tl.mul` |
| `dv += tl.dot(ppT, do)`, `dk += tl.dot(dsT, tl.trans(qT))` | covered | SEMANTICS "The combine pass's rewrites" (`tl.dot_add`); rules/num.md `DotF16`; `safe_dot_add` |
| `tl.trans(do)`, `tl.trans(qT)` on rank 2 | covered | SEMANTICS "Shape changes"; `Tensor.trans` |
| `curr_m += step_m`, `qT_ptrs += step_m * stride_tok`, `do_ptrs += step_m * stride_tok` | covered | SEMANTICS "Index arithmetic", "Pointers" |
| `return dk, dv` at the end | covered | SEMANTICS "Helpers are inlined" |

## 06-fused-attention.py: _attn_bwd_dq

| Construct | Status | Where |
|---|---|---|
| `K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d`, the same for `V` | covered | SEMANTICS "Pointers", "Index arithmetic" |
| `Di = tl.load(D + offs_m)` before the loop; `tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)` | covered | SEMANTICS "tl.load", "tl.static_assert" |
| `curr_n = start_n`, `step_n = BLOCK_N2`; the loop carries `dq`, `curr_n`, `kT_ptrs`, `vT_ptrs` and `offs_n` (assigned in the `if MASK:` branch) | covered | SEMANTICS "Loops", "if, elif, else on a compile-time condition"; under L59 (open) the names hold `int32` scalars, with the same values |
| `tl.dot(q, kT)`, `tl.dot(do, vT)` | covered | rules/num.md `DotF16` |
| `qk - m` with `m` of shape `[BLOCK_M2, 1]` broadcast (a dot feeds it) | covered | SEMANTICS "Operators on tensors" (`bcastCols`), "Float + and -" (`tl.subc`) |
| `tl.math.exp2(qk - m)` | covered | rules/num.md `Exp2F32` |
| `tl.where(mask, p, 0.0)`; `p * (dp - Di[:, None])`; `.to(tl.float16)` | covered | `TVA.tl.where`; SEMANTICS "Float + and -", "Float conversions" |
| `dq += tl.dot(ds, tl.trans(kT))` | covered | SEMANTICS "The combine pass's rewrites" (`tl.dot_add`); rules/num.md `DotF16` |
| the pointer increments; `return dq` | covered | SEMANTICS "Pointers", "Helpers are inlined" |

## 06-fused-attention.py: _attn_bwd

| Construct | Status | Where |
|---|---|---|
| `LN2: tl.constexpr = 0.6931471824645996`; `dq *= LN2` | covered | SEMANTICS "Constants" (an annotated constexpr stays a Python number; `FloatV.litF32`) |
| `tl.program_id(2)`, `tl.program_id(0)` on the grid `(N_CTX // BLOCK_N1, 1, BATCH · N_HEAD)` | covered | SEMANTICS "Launches" (`z < 2^16`) |
| `(bhid * N_CTX).to(tl.int64)`, `(stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)` | covered | SEMANTICS "Index arithmetic" (`int32` until the cast); `Tensor.castInt` |
| `Q += adj`, …, `D += off_chz`: pointers advanced by an `int64` | covered | SEMANTICS "Pointers" |
| `MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR`, `MASK_BLOCK_N2` | covered | SEMANTICS "Expressions with no effect" (`//` on constexprs) |
| unmasked loads of `k`, `v`, `q`, `do`, `m`; `m = m[:, None]` | covered | SEMANTICS "tl.load", "Expressions with no effect" |
| `start_m = 0`, `start_n = 0`, `num_steps = BLOCK_N1 // MASK_BLOCK_M1` (Python numbers assigned to names), passed to the helpers, whose loops carry them | covered | SEMANTICS "Loops", "Constants"; under L59 (open) these names hold `int32` scalars, which gives the same values here |
| the `if CAUSAL:` blocks | covered | SEMANTICS "if, elif, else on a compile-time condition" |
| `(N_CTX - start_m) // BLOCK_M1`, `N_CTX // BLOCK_N2`, `end_n // BLOCK_N2` on nonnegative `int32` values | covered | SEMANTICS "Integer //, %, <<, >>" (`tl.idiv`, truncation) |
| helpers `_attn_bwd_dkdv`, `_attn_bwd_dq`, with the keyword `MASK=` | covered | SEMANTICS "Helpers are inlined", "Every argument is evaluated before the call" (their sections above) |
| `dk *= sm_scale`, with `sm_scale` a Python `float` argument (`float32`) | covered | SEMANTICS "Scalar arguments"; `TVA.tl.mul` |
| `tl.store(dv_ptrs, dv)`, `tl.store(dk_ptrs, dk)`, `tl.store(dq_ptrs, dq)`: `float32` into `float16` | covered | SEMANTICS "tl.store", "Float conversions" |
| the strides (`stride_d = 1`, a constexpr), `H`, `N_CTX` | covered | SEMANTICS "Scalar arguments" |
| launch: `num_warps=4`, `num_stages=5`, keyword constexprs | covered | rules/compiler-faithful.md Scope 4; SEMANTICS "Performance-only arguments" |

## 08-grouped-gemm.py: grouped_matmul_kernel

Excluded: L37 (user decision).

## 08-grouped-gemm.py: grouped_matmul_tma_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(tma_configs, key=['group_size'])` (`num_stages`, `num_warps`) | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md Scope 4 |
| `dtype = tl.float8e4nv if FP8 else tl.float16` | covered | SEMANTICS "Order" |
| `tile_idx = tl.program_id(0)`; `last_problem_end = 0`, carried by `for g in range(group_size)` | covered | SEMANTICS "Loops", "Constants"; under L59 (open) the name holds an `int32` `0`, which gives the same comparisons and sums |
| scalar loads `tl.load(group_gemm_sizes + g * 3)`, `tl.load(g_lds + g * 3 + 1)`, … | covered | SEMANTICS "tl.load"; `TVA.tl.load1`; `safe_load1_buf` |
| `tl.cdiv(gm, BLOCK_SIZE_M)` on a loaded `int32`; `num_tiles = num_m_tiles * num_n_tiles` | covered | SEMANTICS "Helpers are inlined", "Integer //, %, <<, >>", "Index arithmetic" |
| `if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:` (a runtime `if`; `and` of two `int1` scalars) | covered | SEMANTICS "A runtime `if`" (L28), "Order", "Operators and promotion" (`boolOp?`) |
| `tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))` (an `int64` address) | covered | SEMANTICS "Pointers" (`tl.intToPtr`) |
| C written through addresses read from memory (the claim's `Pre` places A, B, C and the tables) | covered | `TVA.independent` (footprints from the initial memory, which the kernel does not write), `Buf` |
| three `tl.make_tensor_descriptor(…, shape=[gm, gk], strides=[lda, 1], block_shape=[…])` inside the runtime `if`, scratch from `triton.set_allocator` (a fresh buffer) | covered | SEMANTICS "Tensor descriptors"; `TVA.tl.make_tensor_descriptor`; rules/compiler-faithful.md Scope 5. L5's open zero-shape case: a zero entry (`gk = 0`) builds descriptors that are then never accessed |
| `while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles)` carrying `tile_idx` | covered | SEMANTICS "Loops" (`tl.whileLoop`); `safe_while` |
| `tile_idx_in_gemm // num_n_tiles`, `% num_n_tiles` | covered | SEMANTICS "Integer //, %, <<, >>"; `safe_idiv1`, `safe_irem1` |
| `for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K))` nested in the `while` | covered | SEMANTICS "Loops" |
| `a_desc.load([offs_am, kk * BLOCK_SIZE_K])`, `b_desc.load(…)`, `c_desc.store([offs_cm, offs_cn], c)` | covered | SEMANTICS "Tensor descriptors" (`tl.desc_load`, `tl.desc_store`, `TensorDesc.Access`); `safe_desc_load`, `safe_desc_store` |
| `accumulator += tl.dot(a, b.T)`, `float16` | covered | SEMANTICS "The combine pass's rewrites" (`tl.dot_add`), "Shape changes"; rules/num.md `DotF16` |
| the same with `float8e4nv` operands (`FP8`) | covered | rules/num.md `DotE4m3` |
| `accumulator.to(dtype)` into `float16` | covered | SEMANTICS "Float conversions" |
| `accumulator.to(tl.float8e4nv)` (`FP8`): an unbounded `float32` accumulator | gap | L64 (saturation beyond `maxVal` is not stated) |
| `tile_idx += NUM_SM`; `last_problem_end = last_problem_end + num_tiles` | covered | SEMANTICS "Index arithmetic", "Loops" |
| `group_size`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(META['NUM_SM'],)`, with `NUM_SM=num_sms()` and `FP8=` as constexprs | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 09-persistent-matmul.py: matmul_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])` (`num_stages`, `num_warps`) | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md Scope 4 |
| `@triton.jit(launch_metadata=_matmul_launch_metadata)` | covered | SEMANTICS "Launches" (`launch_metadata=`); TRANSCRIPTION §4 `decl` |
| the benchmark's launches after `proton.start("matmul", hook="triton")` | excluded | exclusions.md (a hook) |
| builtin `min(num_pid_m - first_pid_m, GROUP_SIZE_M)` | covered | SEMANTICS "Operators and promotion" (`tl.iminimum`) |
| `tl.cdiv`, `pid // num_pid_in_group`, `pid % group_size_m` | covered | SEMANTICS "Helpers are inlined", "Integer //, %, <<, >>" |
| `offs_am = tl.where(offs_am < M, offs_am, 0)` (the `int32` constant `0`) | covered | `TVA.tl.where`; SEMANTICS "Constants" |
| `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` (and `offs_bn`) on a full tile: `M`, `N` multiples of the block sizes | covered | SEMANTICS "Compiler hints: what it promises" (`FirstsDiv` with `contig = BLOCK_SIZE_M`, `ContigIs`); `TVA.tl.multiple_of`, `TVA.tl.max_contiguous`; TRANSCRIPTION §2 |
| the same on a partial tile (the tutorial's `validate(32, 32, 32)`) | excluded | L37 (user decision: full tiles only) |
| `a_ptrs`, `b_ptrs` blocks; masked loads with `other=0.0` on `float16` | covered | SEMANTICS "Pointers", "tl.load", "Constants" |
| masked loads of `float8e4nv` operands (`other=0.0` through `tl.cvt8`) | covered | rules/num.md `CvtF32E4m3` |
| `tl.dot(a, b, accumulator)`, `float16` | covered | rules/num.md `DotF16` |
| `tl.dot` with `float8e4nv` operands | covered | rules/num.md `DotE4m3` |
| `if (c_ptr.dtype.element_ty == tl.float8e4nv):` | covered | SEMANTICS "Expressions with no effect" (attributes of a dtype), "if, elif, else on a compile-time condition" |
| `accumulator.to(tl.float16)`; `c_mask`; masked `tl.store` | covered | SEMANTICS "Float conversions", "tl.store" |
| `accumulator.to(tl.float8e4nv)` | gap | L64 (saturation beyond `maxVal` is not stated) |
| `M`, `N`, `K`, the strides: `int32`, or the constexpr `1` (`stride_bk = 1`: the host passes `b.T`) | covered | SEMANTICS "Scalar arguments" |
| launch: 1-D grid `cdiv(M, BM) · cdiv(N, BN)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 09-persistent-matmul.py: matmul_kernel_tma

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, key=["M", "N", "K", "WARP_SPECIALIZE"])` (`num_stages`, `num_warps`) | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md Scope 4 |
| each config's `pre_hook=matmul_tma_set_block_size_hook`, which replaces the host descriptors' dummy `block_shape` `[1, 1]` | gap | L61 (the claimed launch has the arguments as the last hook leaves them; SEMANTICS and Faithful say so only of memory) |
| `@triton.jit(launch_metadata=…)` | covered | SEMANTICS "Launches" (`launch_metadata=`) |
| the benchmark's launches under the `proton` hook | excluded | exclusions.md (a hook) |
| builtin `min` in the grouping; `tl.cdiv`; scalar `//`, `%` | covered | SEMANTICS "Operators and promotion", "Helpers are inlined", "Integer //, %, <<, >>" |
| `dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16` | covered | SEMANTICS "Order" |
| `for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE)` | covered | SEMANTICS "Loops", "Performance-only arguments" |
| `a_desc.load([offs_am, offs_k])`, `b_desc.load([offs_bn, offs_k])` on `TensorDescriptor.from_tensor` descriptors (zero padding on partial tiles) | covered | SEMANTICS "Tensor descriptors"; `safe_desc_load` |
| `tl.dot(a, b.T, accumulator)`, `float16` and `float8e4nv` | covered | rules/num.md `DotF16`, `DotE4m3` |
| `accumulator.to(tl.float16)`; `c_desc.store([offs_cm, offs_cn], c)` (out-of-bounds lanes dropped) | covered | SEMANTICS "Float conversions", "Tensor descriptors" (`tl.desc_store`) |
| `accumulator.to(tl.float8e4nv)` (`FP8_OUTPUT`) | gap | L64 (saturation beyond `maxVal` is not stated) |
| `M`, `N`, `K`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: 1-D grid, keyword constexprs | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 09-persistent-matmul.py: _compute_pid

| Construct | Status | Where |
|---|---|---|
| `tile_id // num_pid_in_group`, `tile_id % group_size_m`, `(tile_id % num_pid_in_group) // group_size_m` | covered | SEMANTICS "Integer //, %, <<, >>"; `safe_idiv1`, `safe_irem1` |
| builtin `min(num_pid_m - first_pid_m, GROUP_SIZE_M)` | covered | SEMANTICS "Operators and promotion" (Python's `min` folds `tl.minimum`; `tl.iminimum`) |
| constexprs (`GROUP_SIZE_M`, `NUM_SMS`) passed to parameters without annotation; a tuple return | covered | SEMANTICS "Helpers are inlined" |

## 09-persistent-matmul.py: matmul_kernel_persistent

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])` | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md Scope 4 |
| `@triton.jit(launch_metadata=…)` | covered | SEMANTICS "Launches" (`launch_metadata=`) |
| the benchmark's launches under the `proton` hook | excluded | exclusions.md (a hook) |
| `tile_id_c = start_pid - NUM_SMS`, carried and advanced by `NUM_SMS` | covered | SEMANTICS "Loops", "Index arithmetic" |
| `for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)` | covered | SEMANTICS "Loops" (`tl.forRange`), "Performance-only arguments" (`flatten`) |
| helper `_compute_pid`, twice | covered | SEMANTICS "Helpers are inlined" (its section above) |
| the hints `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` on values the body defines, full tiles | covered | SEMANTICS "Compiler hints" (which value: each iteration's; what it promises: `FirstsDiv`, `ContigIs`) |
| the same on partial tiles | excluded | L37 (user decision: full tiles only) |
| `for ki in range(k_tiles)`; masked loads with `other=0.0` on `float16` | covered | SEMANTICS "Loops", "tl.load", "Constants" |
| masked loads of `float8e4nv` operands (`other=0.0` through `tl.cvt8`) | covered | rules/num.md `CvtF32E4m3` |
| `tl.dot(a, b, accumulator)`, `float16` and `float8e4nv` | covered | rules/num.md `DotF16`, `DotE4m3` |
| `if (c_ptr.dtype.element_ty == tl.float8e4nv):`, `.to(tl.float16)`, masked store | covered | SEMANTICS "if, elif, else on a compile-time condition", "Float conversions", "tl.store" |
| `accumulator.to(tl.float8e4nv)` | gap | L64 (saturation beyond `maxVal` is not stated) |
| `M`, `N`, `K`, the strides: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(min(NUM_SMS, tiles),)`, `NUM_SMS=` a constexpr | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 09-persistent-matmul.py: matmul_kernel_tma_persistent

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, key=["M", "N", "K", "WARP_SPECIALIZE"])` | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md Scope 4 |
| each config's `pre_hook=matmul_tma_set_block_size_hook` (with `EPILOGUE_SUBTILE` the `c_desc` block becomes `[BM, BN // 2]`) | gap | L61 (hooks change the launch's arguments, which SEMANTICS and Faithful do not say) |
| `@triton.jit(launch_metadata=…)` | covered | SEMANTICS "Launches" (`launch_metadata=`) |
| the benchmark's launches under the `proton` hook | excluded | exclusions.md (a hook) |
| `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE)` | covered | SEMANTICS "Loops", "Performance-only arguments" |
| helper `_compute_pid`, twice | covered | SEMANTICS "Helpers are inlined" |
| `a_desc.load`, `b_desc.load` on host descriptors; `b.T` | covered | SEMANTICS "Tensor descriptors", "Shape changes" |
| `tl.dot(a, b.T, accumulator)`, `float16` and `float8e4nv` | covered | rules/num.md `DotF16`, `DotE4m3` |
| `if EPILOGUE_SUBTILE:` `tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))`, `tl.permute(acc, (0, 2, 1))`, `tl.split(acc)`, two `c_desc.store` | covered | SEMANTICS "Shape changes" (`reshape`, `permute`, `split`), "Tensor descriptors" (`tl.desc_store`) |
| `.to(dtype)` into `float16` | covered | SEMANTICS "Float conversions" |
| `.to(tl.float8e4nv)` (`FP8_OUTPUT`) | gap | L64 (saturation beyond `maxVal` is not stated) |
| `M`, `N`, `K`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(min(NUM_SMS, tiles),)` | covered | SEMANTICS "Launches" |

## 09-persistent-matmul.py: matmul_kernel_descriptor_persistent

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, prune_configs_by={'early_config_prune': prune_invalid_configs})`, no hook | covered | SEMANTICS "One launch" (L21); rules/compiler-faithful.md Scope 4 |
| `@triton.jit(launch_metadata=…)` | covered | SEMANTICS "Launches" (`launch_metadata=`) |
| the benchmark's launches under the `proton` hook | excluded | exclusions.md (a hook) |
| `dtype = c_ptr.dtype.element_ty` | covered | SEMANTICS "Expressions with no effect" |
| `tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[…])` (three; the block `BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2`), scratch from `triton.set_allocator` (a fresh buffer) | covered | SEMANTICS "Tensor descriptors", "Order"; `TVA.tl.make_tensor_descriptor`; rules/compiler-faithful.md Scope 5. L5's open zero-shape case: with `K = 0` the descriptors of A and B are never accessed |
| the loads and stores through those descriptors | covered | SEMANTICS "Tensor descriptors"; `safe_desc_load`, `safe_desc_store` |
| `tl.range(…, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE)` | covered | SEMANTICS "Loops", "Performance-only arguments" |
| helper `_compute_pid`, twice | covered | SEMANTICS "Helpers are inlined" |
| `tl.dot(a, b.T, accumulator)`, `float16` and `float8e4nv` | covered | rules/num.md `DotF16`, `DotE4m3` |
| epilogue subtiling: `tl.reshape`, `tl.permute`, `tl.split`, two `c_desc.store` | covered | SEMANTICS "Shape changes", "Tensor descriptors" |
| `.to(dtype)` into `float16` | covered | SEMANTICS "Float conversions" |
| `.to(dtype)` into `float8e4nv` | gap | L64 (saturation beyond `maxVal` is not stated) |
| `M`, `N`, `K`: `int32`, or the constexpr `1` | covered | SEMANTICS "Scalar arguments" |
| launch: grid `(min(NUM_SMS, tiles),)`; `FLATTEN`, `WARP_SPECIALIZE` constexprs | covered | SEMANTICS "Launches" |

## 10-block-scaled-matmul.py: block_scaled_matmul_kernel

| Construct | Status | Where |
|---|---|---|
| the target: the H200, where `tl.dot_scaled` runs through Triton's emulation (the tutorial's own `__main__` runs only with block-scaling hardware; its host functions run on the H200) | covered | exclusions.md (targets other than the H200; L37 user decision); rules/num.md "`tl.dot_scaled` on sm_90" |
| `@triton.jit(launch_metadata=_matmul_launch_metadata)` | covered | SEMANTICS "Launches" (`launch_metadata=`) |
| the benchmark's launches under `proton.start(…, hook="triton")` | excluded | exclusions.md (a hook) |
| `if output_type == 0: … elif …` | covered | SEMANTICS "if, elif, else on a compile-time condition" |
| `M`, `N`, `K` constexprs; `tl.cdiv(M, BLOCK_M)` in Python; `pid % num_pid_m`, `pid // num_pid_m` | covered | SEMANTICS "Values" (constexprs), "Integer //, %, <<, >>" |
| `offs_k_a = 0`, `offs_k_b = 0`, `offs_scale_k = 0`, carried by the loop | covered | SEMANTICS "Loops", "Constants"; under L59 (open) the names hold `int32` scalars, which gives the same offsets |
| `MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2` | covered | SEMANTICS "Order" |
| `tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, disallow_acc_multi_buffer=…)` | covered | SEMANTICS "Loops", "Performance-only arguments" |
| `a_desc.load(…)`, `b_desc.load(…)`, the rank-5 `a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])` on `TensorDescriptor.from_tensor` descriptors | covered | SEMANTICS "Tensor descriptors" (rank 1 to 5); `safe_desc_load` |
| `scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)` | covered | SEMANTICS "Shape changes" (row-major `reshape`; `trans` with `dims` is `permute`) |
| `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)` (mxfp8: `e8m0` scales as `uint8`, groups of 32) | covered | SEMANTICS "tl.dot_scaled"; `TVA.tl.dot_scaled` (`ScaledCfg`); rules/num.md `DotScaledE4m3` (on `scaledDomBF16`, `e8m0Dom`); `safe_dot_scaled`; work/probes.out P1 (runs, tutorial tolerance met) |
| the mixed form `"e4m3"` by `"e2m1"` (`b.T` packed fp4 along `K`) | covered | rules/num.md `DotScaledE4m3E2m1`; work/probes.out P1 |
| `"e2m1"` by `"e2m1"` (mxfp4; nvfp4 with `float8e4nv` scales, groups of 16) | covered | SEMANTICS "Launches" (a launch rejected on the host has no outcome; `TVA.Claim.Behaviour`); L37 user decision; work/probes.out P1 (both rejected on the H200 for shared memory with the tutorial's tiles) |
| `offs_k_a += BLOCK_K // ELEM_PER_BYTE_A`, `offs_k_b += …`, `offs_scale_k += rep_k` | covered | SEMANTICS "Index arithmetic" |
| `accumulator.to(output_dtype)` with `float32` (no conversion) or `float16`; `c_desc.store` | covered | SEMANTICS "Float conversions", "Tensor descriptors" |
| `accumulator.to(tl.float8e4nv)` (`output_type == 2`) | gap | L64 (saturation beyond `maxVal` is not stated) |
| launch: grid `(cdiv(M, BM) · cdiv(N, BN), 1)`; the block sizes, `NUM_STAGES` and `disallow_acc_multi_buffer` as constexpr arguments | covered | SEMANTICS "Launches"; rules/compiler-faithful.md "What it says" |

## 10-block-scaled-matmul.py: block_scaled_matmul_kernel_cdna4

Excluded: L37 (user decision).

## 11-programmatic-dependent-launch.py: add_kernel

Excluded: L37 (user decision).

## New gaps

- **NEW** gap — 02 `softmax_kernel` (`-float('inf')`): SEMANTICS does not say what a call of one of Python's builtins in a kernel gives. The front end calls `float`, `isinstance`, `list`, `getattr`, `hasattr` as Python's own on the unwrapped arguments (`code-generator.py:357-365, 1444-1448`), and `int`, `len` by its static executor (`:1639-1659`, a `constexpr` result), at compile time, emitting no operation (`range`, `min`, `max`, `print` it maps to Triton's own, which SEMANTICS does treat), while "Expressions with no effect" lists every call as possibly effectful and "Order" does not count such a result among the compile-time values. work/probes.out P2: `tl.static_assert(float("inf") > 1e308)` and `tl.static_assert(isinstance(out_ptr, tl.tensor))` compile, `-float('inf')` is stored as `0xFF800000`, and only the branch `isinstance` picks is emitted. One sentence in SEMANTICS "How the front end evaluates the source" would close it.
- **NEW** gap — 06 `_maybe_make_tensor_desc` (`isinstance(desc_or_ptr, tl.tensor_descriptor)`): the same missing statement, here deciding which branch exists: a host `TensorDescriptor` argument is a `tl.tensor_descriptor` value, a pointer is not (work/probes.out P2: stores `1.0` for a descriptor, `2.0` for a pointer).
- **NEW** gap — 06 `_attn_fwd`: the same, through `m_i = tl.zeros(…) - float("inf")` and the helper `_maybe_make_tensor_desc` (library lemma `Softmax.sub_zero_posInf` already reads `float("inf")` as `+∞`, but no fact page states it).
