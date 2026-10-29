# Tutorial coverage, round 6

## 01-vector-add.py: add_kernel

| Construct | Status | Where |
|---|---|---|
| `tl.program_id(axis=0)` on a 1-D grid | covered | SEMANTICS "Launches" (`program_id`), "Index arithmetic"; `TVA.tl.program_id`; TRANSCRIPTION §2 |
| `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` (`int32`, wrapping) | covered | SEMANTICS "Index arithmetic"; `TVA.tl.arange`; `TVA.Block` (`offs`, `Fits`, `fits_of_grid`) |
| `mask = offsets < n_elements` | covered | SEMANTICS "Operators on tensors"; `Tensor.ltS`; TRANSCRIPTION §2 |
| `n_elements`, an integer argument in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" (a scalar integer argument below `2^31` is `int32`) |
| `n_elements` equal to `1` (Triton passes it as a constexpr) or at least `2^31` (`int64`) | gap | L29 (work/scalar_specialisation.out; SEMANTICS's "below `2^31` is passed as `int32`" does not except the `1`) |
| `tl.load(x_ptr + offsets, mask=mask)` without `other`, `float32` | covered | SEMANTICS "tl.load" (masked-off lanes unspecified); `TVA.tl.load`; `safe_load_block` |
| `x + y` on `float32`, no product feeding it | covered | SEMANTICS "Float + and -: IEEE or fusable"; `TVA.tl.add`; `FloatV.add_err` |
| `tl.store(output_ptr + offsets, output, mask=mask)`, same dtype | covered | SEMANTICS "tl.store"; `TVA.tl.store`; `safe_store_block` |
| pointer arguments (`p + i` advances `4·i` bytes); the output disjoint from the inputs | covered | SEMANTICS "Values" (`Ptr`, "Pointers"); `TVA.independent`, `Buf` |
| `BLOCK_SIZE: tl.constexpr` (`1024`) | covered | SEMANTICS "Values" (constexpr values are Lean values); `Launch.blocks`, `Launch.Valid` |
| launch `add_kernel[grid](…, BLOCK_SIZE=1024)`, grid `(cdiv(n_elements, BLOCK_SIZE),)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1`; rules/compiler-faithful.md "Scope" |

## 02-fused-softmax.py: softmax_kernel

| Construct | Status | Where |
|---|---|---|
| `tl.program_id(0)`, `tl.num_programs(0)` | covered | SEMANTICS "Launches"; `TVA.tl.program_id`, `TVA.tl.num_programs` |
| `for row_idx in tl.range(row_start, n_rows, row_step, …)`: a runtime start and step | covered | SEMANTICS "Loops" (`tl.forRange`; `tl.RangeOk`: the step `num_programs` is at least `1`); `safe_forRange` |
| `num_stages=num_stages` of `tl.range`, from the constexpr argument `num_stages` | covered | SEMANTICS "Loops" (scheduling keywords change no outcome), "Programs" (L15); TRANSCRIPTION §4 `no-op:range-kw` |
| `input_ptr + row_idx * input_row_stride + col_offsets` (`int32` offsets) | covered | SEMANTICS "Index arithmetic", "Pointers" |
| `tl.load(input_ptrs, mask=mask, other=-float('inf'))` | covered | SEMANTICS "tl.load", "Constants" (`other=`: a `float32` tensor, `FloatV.litF32` of `-inf`) |
| `tl.max(row, axis=0)` of a rank-1 `float32` tensor | covered | SEMANTICS reductions; `TVA.tl.maxAll`; `safe_maxAll` |
| `row - tl.max(row, axis=0)`: a scalar broadcast, IEEE | covered | SEMANTICS "Operators on tensors" (`bc`), "Float + and -"; `TVA.tl.sub` |
| `tl.exp(row_minus_max)` on `float32` (`ex2.approx.f32` after a multiplication, work/math_lowering.out): its accuracy, and `exp(-inf) = +0` on the masked lanes | gap | L4 (rules/num.md states no `MathAcc` or `MathAt` yet) |
| `tl.sum(numerator, axis=0)`: an IEEE tree (the result of `tl.exp` is not a product) | covered | SEMANTICS reductions, "Float + and -"; `TVA.tl.sum`; `sum_err_covers` |
| `numerator / denominator` on `float32` | covered | SEMANTICS "/ on floats"; `TVA.tl.div`; rules/num.md `DivFull` |
| `tl.store(output_ptrs, softmax_output, mask=mask)` | covered | SEMANTICS "tl.store"; `TVA.tl.store` |
| scalar arguments (the strides, `n_rows`, `n_cols`) in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| the same equal to `1` (a constexpr) or at least `2^31` (`int64`) | gap | L29 |
| launch through a precompiled kernel: `softmax_kernel.warmup(…)`, then `kernel[(num_programs, 1, 1)](…)` on the `CompiledKernel` | gap | NEW: (N6) rules/compiler-faithful.md describes launches through the JIT; a `CompiledKernel` launch skips the JIT's specialisation of the arguments (`1` as a constexpr, `tt.divisibility` 16 for integers divisible by 16 and for 16-byte-aligned pointers, work/scalar_specialisation.out), so the scope must say it covers such a launch only with arguments that match the compiled specialisation (here the warm-up's own arguments) |
| grid `(min(NUM_SM · occupancy, n_rows), 1, 1)` from the device's occupancy (possibly 0), `num_warps=8` | covered | SEMANTICS "Launches" (any grid; a zero grid ends `ok` with memory unchanged); rules/compiler-faithful.md "Scope" point 4 (`num_warps`) |

## 03-matrix-multiplication.py: matmul_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])`: benchmark runs on the real arguments, then one config | gap | L21 |
| `num_warps`, `num_stages` in each `triton.Config` | covered | rules/compiler-faithful.md "Scope" point 4; SEMANTICS "Loops" (pipelining matters only for a program that reuses its own data, which this kernel does not) |
| `tl.cdiv(M, BLOCK_SIZE_M)`: Triton's `@jit` helper `(x + div - 1) // div` on an `int32` | covered | SEMANTICS "Helpers are inlined", "Integer //, %" (`tl.idiv`); TRANSCRIPTION §3 (index arithmetic that wraps) |
| scalar `pid // num_pid_in_group`, `pid % num_pid_in_group`, `% group_size_m` | covered | SEMANTICS "Integer //, %" (`tl.idiv`, `tl.irem`, undefined for a zero divisor); `safe_idiv1`, `safe_irem1` |
| builtin `min(num_pid_m - first_pid_m, GROUP_SIZE_M)` on an `int32` scalar | gap | NEW: (N3) SEMANTICS does not say that Python's `min` in a kernel is `tl.minimum` (`core.builtin_min`, `code_generator.py:361-364`), and Python's own `min` when every operand is a compile-time value, nor what integer `tl.minimum` computes (only the docstring of `IntV.min` in `TVA.Lang.Scalar` does) |
| `tl.assume(pid_m >= 0)` … `tl.assume(stride_cn > 0)` | covered | SEMANTICS "Compiler hints" (`tl.assume` is undefined behaviour if false); `TVA.tl.assume` |
| `(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M` | covered | SEMANTICS "Integer //, %" (`tl.irem`) |
| `a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)`: 2-D pointer blocks | covered | SEMANTICS "Operators on tensors" (`x[:, None]`, `.col`/`.row`), "Index arithmetic", "Pointers" |
| `tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)` | covered | `TVA.tl.zeros` (SEMANTICS "The tl.* operations": its docstring) |
| `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))` carrying `accumulator`, `a_ptrs`, `b_ptrs` | covered | SEMANTICS "Loops" (`tl.forRange`, the carried tuple); `safe_forRange` |
| `tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)`: a `[1, BK]` mask broadcast; `other` converted to `float16` | covered | SEMANTICS "tl.load", "Constants" (`other=`), "Float conversions"; `safe_load_buf` |
| the same loads with `float8e5` operands (the tutorial's fp8 run): `other=0.0` is `tl.cvt8` of `0.0`, the platform's conversion | gap | L4 (no `CvtAcc` fact, so a proof cannot show the masked lanes hold `0`) |
| `tl.dot(a, b, accumulator)`, `float16` operands into `float32` | gap | L4 (`DotAcc` is not stated; `DotTotal` is) |
| `tl.dot` with `float8e5` operands (`maxImpreciseAcc = 2^30` on sm_90) | gap | L4 (its list names `float16` and `bfloat16` dots only; the fp8 dot must be added) |
| `a_ptrs += BLOCK_SIZE_K * stride_ak` | covered | SEMANTICS "Pointers" |
| `if ACTIVATION == "leaky_relu":` on a constexpr string | covered | SEMANTICS "if, elif, else on a compile-time condition"; TRANSCRIPTION §4 `config` |
| helper `leaky_relu(accumulator)` | covered | SEMANTICS "Helpers are inlined" (its section below) |
| `accumulator.to(tl.float16)` | covered | SEMANTICS "Float conversions" (`tl.cvt`, nearest even); `FloatV.cvt_err` |
| `c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)`; `tl.store(c_ptrs, c, mask=c_mask)` | covered | SEMANTICS "Operators on tensors" (`Tensor.and`), "tl.store" |
| scalar arguments (`M`, `N`, `K`, the strides) in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| the same equal to `1` (the tutorial's unit strides are, and become constexprs) or at least `2^31` | gap | L29 |
| launch: 1-D grid `cdiv(M, BM) · cdiv(N, BN)`, `ACTIVATION=` a constexpr | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 03-matrix-multiplication.py: leaky_relu

| Construct | Status | Where |
|---|---|---|
| `x >= 0` on a `float32` tensor with the literal `0` | covered | SEMANTICS "Constants" (built in `float32`), "Operators on tensors"; `Tensor.fle` (ordered comparisons) |
| `0.01 * x` (`FloatV.litF32` of `0.01`) | covered | SEMANTICS "Constants"; `TVA.tl.mul` |
| `tl.where(x >= 0, x, 0.01 * x)` | covered | `TVA.tl.where` (both sides computed); SEMANTICS "Expressions with no effect" |
| `return` as the last statement | covered | SEMANTICS "Helpers are inlined" |

## 04-low-memory-dropout.py: _dropout

| Construct | Status | Where |
|---|---|---|
| `pid`, `offsets`, `mask` as in tutorial 01 | covered | SEMANTICS "Index arithmetic"; `TVA.Block` |
| `tl.load(x_ptr + offsets, mask=mask)`, `tl.load(x_keep_ptr + offsets, mask=mask)` (`float32`, `int32`) | covered | SEMANTICS "tl.load"; `safe_load_block` |
| `p`, a Python `float` argument (Triton passes `fp32`) | gap | L29 (work/scalar_specialisation.out) |
| `1 - p`: the constant in `float32`, an IEEE subtraction | covered | SEMANTICS "Constants", "Float + and -"; `TVA.tl.sub` |
| `x / (1 - p)`: `float32` `/`, scalar broadcast | covered | SEMANTICS "/ on floats"; rules/num.md `DivFull` |
| `tl.where(x_keep, …, 0.0)` with the `int32` tensor `x_keep` as the condition | gap | NEW: (N2) Triton casts a condition that is not `int1` to `int1` as `x_keep != 0` (`semantic.py:1638-1643`, `:874-877`); SEMANTICS states neither this implicit cast nor the integer-to-`int1` conversion (only the docstring of `Tensor.castInt` does) |
| `0.0` in `tl.where` (`float32` `+0`) | covered | SEMANTICS "Constants" |
| `tl.store(output_ptr + offsets, output, mask=mask)` | covered | SEMANTICS "tl.store" |
| `n_elements` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `n_elements` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(cdiv(n_elements, 1024),)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 04-low-memory-dropout.py: _seeded_dropout

| Construct | Status | Where |
|---|---|---|
| `pid`, `offsets`, `mask`; `tl.load(x_ptr + offsets, mask=mask)` | covered | SEMANTICS "Index arithmetic", "tl.load" |
| `tl.rand(seed, offsets)` (Philox 4x32-10, then `uint_to_uniform_float`) | covered | SEMANTICS "Random numbers"; `TVA.tl.rand` (a seed of any integer type, `seed.to(uint64)`); `safe_rand` |
| `n_elements`, `seed` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `n_elements`, `seed` equal to `1` (a constexpr) or at least `2^31` (`int64`) | gap | L29 |
| `p`, a Python `float` argument (`fp32`) | gap | L29 |
| `x_keep = random > p`: an ordered `float32` comparison (the product `random` is compared, not added) | covered | SEMANTICS "Operators on tensors", "Random numbers"; `Tensor.flt` |
| `tl.where(x_keep, x / (1 - p), 0.0)` with a boolean condition | covered | `TVA.tl.where`; SEMANTICS "Constants", "/ on floats" (`DivFull`) |
| `tl.store(output_ptr + offsets, output, mask=mask)` | covered | SEMANTICS "tl.store" |
| launch: grid `(cdiv(n_elements, 1024),)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 05-layer-norm.py: _layer_norm_fwd_fused

| Construct | Status | Where |
|---|---|---|
| `Y += row * stride`, `X += row * stride`: pointer arguments advanced by an `int32` | covered | SEMANTICS "Pointers", "Index arithmetic" |
| `mean = 0` before the loops, overwritten before any use | covered | TRANSCRIPTION §4 `dead`; SEMANTICS "Expressions with no effect" |
| three loops `for off in range(0, N, BLOCK_SIZE)` (runtime bound), carrying `_mean`, `_var` | covered | SEMANTICS "Loops"; `safe_forRange` |
| `tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)`: `float16` with `other` `0`, then an exact widening | covered | SEMANTICS "tl.load", "Constants" (`other=`), "Float conversions" (`tl.cvt`) |
| `tl.zeros([BLOCK_SIZE], dtype=tl.float32)`; `_mean += a` (no product feeds it: IEEE) | covered | `TVA.tl.zeros`; SEMANTICS "Float + and -"; `TVA.tl.add` |
| mixed-dtype operands: `tl.sum(_mean, axis=0) / N` (`float32` and the `int32` `N`), `x_hat * w + b` (`float32` and the `float16` `w`, `b`) | gap | NEW: (N1) SEMANTICS says the model writes Triton's dtype promotion, but not what it is for a binary operator on two tensors of different dtypes (`semantic.py` `computation_type_impl`, `integer_promote_impl`: an integer with a float gives the float, two floats the wider, `float16` with `bfloat16` gives `float32`, two integers by width and signedness); it states only `/` and `%` on `float16`/`bfloat16` and the reductions' widenings |
| `tl.where(cols < N, x - mean, 0.)` | covered | `TVA.tl.where`, `TVA.tl.sub`; SEMANTICS "Constants" |
| `_var += x * x` (a product feeds it) | covered | SEMANTICS "Float + and -" (`tl.mul_add`); `FloatV.mulAdd_err` |
| `tl.sum(_var, axis=0)`: an IEEE tree (`_var` holds sums, not products) | covered | SEMANTICS reductions; `TVA.tl.sum`; `sum_err_covers` |
| `eps`, a Python `float` argument (`fp32`), in `var + eps` | gap | L29 |
| `tl.sqrt(var + eps)` on `float32` (`sqrt.approx.ftz.f32` on sm_90, work/math_lowering.out) | gap | L4 (no `MathAcc` for `sqrt`) |
| `1 / tl.sqrt(…)` | covered | SEMANTICS "/ on floats", "Constants"; rules/num.md `DivFull` |
| `tl.store(Mean + row, mean)`, `tl.store(Rstd + row, rstd)`: scalar stores | covered | SEMANTICS "tl.store"; `TVA.tl.store1` |
| `tl.load(W + cols, mask=mask)`, `tl.load(B + cols, mask=mask)` without `other` | covered | SEMANTICS "tl.load" |
| `x_hat = (x - mean) * rstd`; `y = x_hat * w + b` as `tl.mul_add` (after the promotion above) | covered | SEMANTICS "Float + and -"; `TVA.tl.mul_add` |
| `tl.store(Y + cols, y, mask=mask)`: `float32` converted to `float16` | covered | SEMANTICS "tl.store", "Float conversions" |
| `stride`, `N` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `stride`, `N` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(M,)`, `num_warps`, `num_ctas=1` | covered | SEMANTICS "Launches" (`Launch.numCtas`); rules/compiler-faithful.md "Scope" point 4 |

## 05-layer-norm.py: _layer_norm_bwd_dx_fused

Excluded: L37 (user decision).

## 05-layer-norm.py: _layer_norm_bwd_dwdb

| Construct | Status | Where |
|---|---|---|
| `cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)` | covered | SEMANTICS "Index arithmetic" |
| `tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)`, twice | covered | `TVA.tl.zeros` |
| `for i in range(0, M, BLOCK_SIZE_M)` carrying `dw`, `db` | covered | SEMANTICS "Loops"; `safe_forRange` |
| `mask = (rows[:, None] < M) & (cols[None, :] < N)`; `offs = rows[:, None] * N + cols[None, :]` (`int32`) | covered | SEMANTICS "Operators on tensors", "Index arithmetic" |
| `tl.load(DW + offs, mask=mask, other=0.)` (`float16`, `other` `0`) | covered | SEMANTICS "tl.load", "Constants" |
| `dw += tl.load(…)`: a `float32` accumulator plus a `float16` block | gap | NEW: (N1) the promotion of the `float16` block to `float32` is not stated (the addition itself is IEEE, SEMANTICS "Float + and -") |
| `tl.sum(dw, axis=0)`, `tl.sum(db, axis=0)` | covered | SEMANTICS reductions; `TVA.tl.sum0`; `sum_err` |
| `tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)`: `float32` to `float16` | covered | SEMANTICS "tl.store", "Float conversions" |
| `M`, `N` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `M`, `N` equal to `1` or at least `2^31` | gap | L29 |
| the inputs `_dw`, `_db`, written by the excluded `_layer_norm_bwd_dx_fused`: plain memory at this launch | covered | SEMANTICS "Launches" (a claim is about one launch, from its `init`) |
| launch: grid `(cdiv(N, 128),)`, `num_ctas=1` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 06-fused-attention.py: _attn_fwd_inner

| Construct | Status | Where |
|---|---|---|
| `if STAGE == 1: … elif STAGE == 2: … else:` and `if dtype == tl.float8e5:` (constexprs) | covered | SEMANTICS "if, elif, else on a compile-time condition"; TRANSCRIPTION §4 `config` |
| tuple assignments `lo, hi = 0, start_m * BLOCK_M` | covered | SEMANTICS "Index arithmetic", "Constants" |
| `lo = tl.multiple_of(lo, BLOCK_M)` on an `int32` scalar, before the loop | covered | SEMANTICS "Compiler hints"; `TVA.tl.multiple_of1` |
| `for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize)` | covered | SEMANTICS "Loops" (`tl.forRange`; scheduling keywords, `warp_specialize` among them) |
| `start_n = tl.multiple_of(start_n, BLOCK_N)` on the induction variable | gap | L1 (and the docstring of the hints' section in `TVA.Lang.Ops` still says a hint on a block argument has no effect and is dropped, the reading SEMANTICS withdraws) |
| `desc_k.load([offsetk_y, 0]).T`, `desc_v.load([offsetv_y, 0])`, `desc_v.load([0, offsetv_y]).T` on host descriptors | covered | SEMANTICS "Tensor descriptors" (`tl.desc_load`, `TensorDesc.Access`), "Shape changes" (`x.T`); `safe_desc_load` |
| the same on descriptors built in the kernel (`_maybe_make_tensor_desc` given pointers) | gap | L5 |
| parameters annotated `tl.constexpr` (`offs_m`, `offs_n`, `N_CTX`) that receive tensors | covered | SEMANTICS "Helpers are inlined" (the body with the call's arguments; `code_generator.py:1358-1365` binds them as given) |
| `tl.dot(q, k)`, `tl.dot(p, v, acc)` with `float16` operands | gap | L4 (`DotAcc`) |
| the same with `float8e5` operands (`FP8_OUTPUT`) | gap | L4 (fp8 dots are not on its list) |
| `mask = offs_m[:, None] >= (start_n + offs_n[None, :])` | covered | SEMANTICS "Operators on tensors" |
| `qk * qk_scale + tl.where(mask, 0, -1.0e6)` (`tl.mul_add`; both constants built in `float32`) | covered | SEMANTICS "Float + and -", "Constants" |
| `tl.max(qk, 1)`; `tl.maximum(m_i, …)` on `float32` | covered | SEMANTICS reductions, "tl.maximum"; `TVA.tl.max1`, `FloatV.MaxRel` |
| `qk -= m_ij[:, None]` (IEEE) and `qk * qk_scale - m_ij[:, None]` (`tl.subc`) | covered | SEMANTICS "Float + and -" |
| `tl.math.exp2(qk)`, `tl.math.exp2(m_i - m_ij)` (`ex2.approx.ftz.f32`; `m_i = -inf` at the first step, so `exp2(-inf) = +0` is needed) | gap | L4 |
| `tl.sum(p, 1)`: IEEE (`p` is not a product) | covered | SEMANTICS reductions; `TVA.tl.sum1` |
| the branch `not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128`: `BM: tl.constexpr = acc.shape[0]`, `acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()`, `tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])` | covered | SEMANTICS "Order" (a compile-time `and`), "Shape changes" (`Tensor.reshape`, `permute`, `split`, `join`) |
| `acc * alpha[:, None]` | covered | `TVA.tl.mul`; SEMANTICS "Operators on tensors" |
| `p.to(dtype)` to `float16` | covered | SEMANTICS "Float conversions" (`tl.cvt`) |
| `p.to(tl.float8e5)` | gap | L4 (no `CvtAcc` fact) |
| `l_i * alpha + l_ij` (`tl.mul_add`); `m_i = m_ij`; `offsetk_y += BLOCK_N` | covered | SEMANTICS "Float + and -", "Index arithmetic" |
| `return acc, l_i, m_i` at the end | covered | SEMANTICS "Helpers are inlined" |

## 06-fused-attention.py: _maybe_make_tensor_desc

| Construct | Status | Where |
|---|---|---|
| `if isinstance(desc_or_ptr, tl.tensor_descriptor):`, a Python `bool`, with a `return` in each branch | covered | SEMANTICS "Order" (a `bool` is a compile-time value), "if, elif, else on a compile-time condition" |
| a host `TensorDescriptor` returned unchanged | covered | SEMANTICS "Tensor descriptors" (the host's shape, strides, block shape and padding) |
| `tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)` | gap | L5 |

## 06-fused-attention.py: _attn_fwd

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, prune_configs_by={'early_config_prune': …})`, each config with `pre_hook=_host_descriptor_pre_hook`, which sets the host descriptors' `block_shape` | gap | L21 |
| `num_warps`, `num_stages` in each config | covered | rules/compiler-faithful.md "Scope" point 4 |
| `dtype = tl.float8e5 if FP8_OUTPUT else tl.float16` | covered | SEMANTICS "Order" (`x if c else y` on a compile-time `c`) |
| `tl.static_assert(BLOCK_N <= HEAD_DIM)` | covered | SEMANTICS "tl.static_assert"; TRANSCRIPTION §4 `compile-check` |
| `tl.program_id(0)`, `tl.program_id(1)` on a 2-D grid | covered | SEMANTICS "Launches" (`program_id = (i % gx, i / gx % gy, …)`) |
| `off_hz // H`, `off_hz % H`, `Z * H * N_CTX`, `offset_y` (`int32`) | covered | SEMANTICS "Integer //, %", "Index arithmetic"; `safe_idiv1`, `safe_irem1` |
| helper `_maybe_make_tensor_desc` on host descriptors (`warp_specialize=False` on Hopper) | covered | SEMANTICS "Helpers are inlined", "Tensor descriptors" |
| the same on raw pointers (on Hopper with the forward's default `warp_specialize=True`): descriptors built in the kernel, scratch from `triton.set_allocator` | gap | L5 |
| `tl.zeros(…) - float("inf")`, `tl.zeros(…) + 1.0`, `tl.zeros([BLOCK_M, HEAD_DIM], …)` | covered | SEMANTICS "Constants" (`FloatV.litF32`), "Float + and -" |
| `sm_scale`, a Python `float` argument (`fp32`) | gap | L29 |
| `qk_scale *= 1.44269504` (the constant in `float32`) | covered | SEMANTICS "Constants"; `TVA.tl.mul` |
| `q = desc_q.load([qo_offset_y, 0])` | covered | SEMANTICS "Tensor descriptors"; `safe_desc_load` |
| `if STAGE & 1:`, `if STAGE & 2:` (a constexpr `&`) | covered | SEMANTICS "Expressions with no effect", "if, elif, else on a compile-time condition" |
| helper `_attn_fwd_inner`, called with `4 - STAGE` and with `2` | gap | L1 (and L4, L5: inherited, its section above) |
| `m_i += tl.math.log2(l_i)` on `float32` (libdevice's polynomial on sm_90, work/math_log2_ptx.out) | gap | L4 (no `MathAcc` for `log2`) |
| the addition `m_i + log2(l_i)`: IEEE (no product feeds it; the PTX ends `log2` in a select, then a plain `add.f32`) | covered | SEMANTICS "Float + and -" |
| `acc / l_i[:, None]` | covered | SEMANTICS "/ on floats"; rules/num.md `DivFull` |
| `tl.store(m_ptrs, m_i)` with `m_ptrs = M + off_hz * N_CTX + offs_m` | covered | SEMANTICS "tl.store" |
| `desc_o.store([qo_offset_y, 0], acc.to(tl.float16))` | covered | SEMANTICS "Tensor descriptors" (`tl.desc_store`), "Float conversions"; `safe_desc_store` |
| `acc.to(tl.float8e5)` for the fp8 output | gap | L4 (no `CvtAcc` fact) |
| `Z`, `H`, `N_CTX` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `Z`, `H`, `N_CTX` equal to `1` (the tutorial's `Z = 1`) or at least `2^31` | gap | L29 |
| launch: grid `(cdiv(N_CTX, BLOCK_M), Z · H, 1)`, keyword constexprs | covered | SEMANTICS "Launches" (`y < 2^16`) |

## 06-fused-attention.py: _attn_bwd_preprocess

| Construct | Status | Where |
|---|---|---|
| `tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)`, `tl.program_id(1)` on a 2-D grid | covered | SEMANTICS "Launches", "Index arithmetic" |
| unmasked loads `tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])`, `.to(tl.float32)` | covered | SEMANTICS "tl.load", "Float conversions" |
| `o * do`: a `float16` block times a `float32` block | gap | NEW: (N1) the promotion of `o` to `float32` is not stated |
| `tl.sum(o * do, axis=1)`, a sum of products (`tl.sum1c`): its error bound | gap | L33 |
| `tl.store(Delta + off_hz * N_CTX + off_m, delta)` | covered | SEMANTICS "tl.store" |
| `N_CTX` in `[-2^31, 2^31)` other than `1`: `int32` (`Z`, `H` unused) | covered | SEMANTICS "Index arithmetic" |
| `N_CTX` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(N_CTX // 128, BATCH · N_HEAD)` | covered | SEMANTICS "Launches" |

## 06-fused-attention.py: _attn_bwd_dkdv

| Construct | Status | Where |
|---|---|---|
| `Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d`, on a pointer the caller advanced by an `int64` | covered | SEMANTICS "Pointers" (`HAdd (Ptr d) (IntV sg w)`), "Index arithmetic" |
| `tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)` | covered | SEMANTICS "tl.static_assert" |
| `for blk_idx in range(num_steps)` (`num_steps` an `int32` or a constexpr), carrying `dk`, `dv`, `curr_m`, `offs_m`, `qT_ptrs`, `do_ptrs` | covered | SEMANTICS "Loops"; `safe_forRange` |
| unmasked loads `tl.load(qT_ptrs)`, `tl.load(M + offs_m)`, `tl.load(do_ptrs)`, `tl.load(D + offs_m)` | covered | SEMANTICS "tl.load" |
| `tl.dot(k, qT)`, `tl.dot(v, tl.trans(do))`, `float16` into `float32` | gap | L4 (`DotAcc`) |
| `qkT - m[None, :]` and `dpT - Di[None, :]` (a dot feeds them: `tl.subc`); `pT * (…)` | covered | SEMANTICS "Float + and -" |
| `tl.math.exp2(qkT - m[None, :])` | gap | L4 |
| `if MASK:` (a constexpr): `tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)` | covered | SEMANTICS "if, elif, else on a compile-time condition", "Constants"; `TVA.tl.where` |
| `ppT.to(tl.float16)`, `dsT.to(tl.float16)`, and `.to(tl.float32)` of a `float32` dot (exact) | covered | SEMANTICS "Float conversions" (`tl.cvt`) |
| `dv += tl.dot(ppT, do)`, `dk += tl.dot(dsT, tl.trans(qT))` (the combine pass's `tl.dot_add`) | gap | L4 (the meaning is SEMANTICS "The combine pass's rewrites", `TVA.tl.dot_add`; the accuracy needs `DotAcc`) |
| `tl.trans(do)`, `tl.trans(qT)` on rank 2 | covered | SEMANTICS "Shape changes" |
| `curr_m += step_m`, `qT_ptrs += step_m * stride_tok` | covered | SEMANTICS "Index arithmetic", "Pointers" |
| `return dk, dv` at the end | covered | SEMANTICS "Helpers are inlined" |

## 06-fused-attention.py: _attn_bwd_dq

| Construct | Status | Where |
|---|---|---|
| `K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d` and the same for `V` | covered | SEMANTICS "Pointers", "Index arithmetic" |
| `Di = tl.load(D + offs_m)` before the loop; `tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)` | covered | SEMANTICS "tl.load", "tl.static_assert" |
| `for blk_idx in range(num_steps)` carrying `dq`, `curr_n`, `offs_n` (assigned in the `if MASK:` branch), `kT_ptrs`, `vT_ptrs` | covered | SEMANTICS "Loops", "if, elif, else on a compile-time condition" |
| `tl.dot(q, kT)`, `tl.dot(do, vT)` | gap | L4 |
| `qk - m` with `m` of shape `[BLOCK_M2, 1]` broadcast (`tl.subc`) | covered | SEMANTICS "Operators on tensors" (`bcastCols`), "Float + and -" |
| `tl.math.exp2(qk - m)` | gap | L4 |
| `tl.where(mask, p, 0.0)`; `p * (dp - Di[:, None])`; `.to(tl.float16)` | covered | `TVA.tl.where`; SEMANTICS "Float + and -", "Float conversions" |
| `dq += tl.dot(ds, tl.trans(kT))` (`tl.dot_add`) | gap | L4 |
| the pointer increments; `return dq` | covered | SEMANTICS "Pointers", "Helpers are inlined" |

## 06-fused-attention.py: _attn_bwd

| Construct | Status | Where |
|---|---|---|
| `LN2: tl.constexpr = 0.6931471824645996`; `dq *= LN2` (`FloatV.litF32`) | covered | SEMANTICS "Constants" |
| `tl.program_id(2)`, `tl.program_id(0)` on the grid `(N_CTX // 128, 1, BATCH · N_HEAD)` | covered | SEMANTICS "Launches" (`z < 2^16`) |
| `(bhid * N_CTX).to(tl.int64)`, `(stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)`: `int32` arithmetic, then a widening | covered | SEMANTICS "Index arithmetic" (wraps until the cast); `Tensor.castInt` |
| `Q += adj`, …, `D += off_chz`: pointers advanced by an `int64` | covered | SEMANTICS "Pointers" |
| `MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR` | covered | SEMANTICS "Expressions with no effect" (`//` on constexprs) |
| unmasked loads of `k`, `v`, `q`, `do`, `m`; `m[:, None]` | covered | SEMANTICS "tl.load", "Expressions with no effect" |
| the `if CAUSAL:` blocks; `start_m` a Python `0` or an `int32` after them | covered | SEMANTICS "if, elif, else on a compile-time condition", "Constants" |
| `(N_CTX - start_m) // BLOCK_M1`, `N_CTX // BLOCK_N2`, `end_n // BLOCK_N2` (`tl.idiv`) | covered | SEMANTICS "Integer //, %" |
| helpers `_attn_bwd_dkdv`, `_attn_bwd_dq`, with the keyword `MASK=` | gap | L4 (inherited: the dots and `exp2`, their sections above) |
| `dk *= sm_scale`, with `sm_scale` a Python `float` argument (`fp32`) | gap | L29 |
| `tl.store(dv_ptrs, dv)`, `tl.store(dk_ptrs, dk)`, `tl.store(dq_ptrs, dq)`: `float32` to `float16` | covered | SEMANTICS "tl.store", "Float conversions" |
| the strides, `H`, `N_CTX` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| the same equal to `1` (`stride_d = q.stride(3) = 1` in every tutorial run) or at least `2^31` | gap | L29 |
| launch: `num_warps=4`, `num_stages=5`, keyword constexprs | covered | rules/compiler-faithful.md "Scope" point 4 |

## 08-grouped-gemm.py: grouped_matmul_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=…, key=['group_size'])`, with `NUM_SM` from the config and the grid `(META['NUM_SM'],)` | gap | L21 |
| `last_problem_end = 0` and `tile_idx` carried by `for g in range(group_size)` and by the `while` (a Python `0` becomes `int32`) | covered | SEMANTICS "Loops", "Constants" (elsewhere an `int` is `int32`) |
| scalar loads `tl.load(group_gemm_sizes + g * 3)`, `tl.load(g_lds + g * 3 + 1)`, … | covered | SEMANTICS "tl.load"; `TVA.tl.load1`; `safe_load1_buf` |
| `tl.cdiv(gm, BLOCK_SIZE_M)` on a loaded `int32` | covered | SEMANTICS "Helpers are inlined", "Integer //, %" |
| `while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles)`: `and` on two `int1` tensors | covered | SEMANTICS "Loops" (`tl.whileLoop`), "Order" (both sides evaluated); `safe_while` |
| `tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))` (an `int64` address) | covered | SEMANTICS "Values" ("Pointers": `tl.intToPtr`) |
| C written through addresses read from memory (the claim's `Pre` places A, B, C and the tables) | covered | `TVA.independent`, `Buf` (footprints from the initial memory, which the kernel does not write) |
| `tile_idx_in_gemm // num_n_tiles`, `% num_n_tiles` | covered | SEMANTICS "Integer //, %"; `safe_idiv1`, `safe_irem1` |
| `a_ptr + offs_am[:, None] * lda + offs_k[None, :]` | covered | SEMANTICS "Pointers", "Index arithmetic" |
| `for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K))` nested in the `while` | covered | SEMANTICS "Loops" |
| `tl.multiple_of(a_ptrs, [16, 16])`, `tl.multiple_of(b_ptrs, [16, 16])` as statements in the loop body, on the loop-carried rank-2 pointer blocks | gap | L1 (and L2 for the per-dimension values; the docstring of the hints' section in `TVA.Lang.Ops` still states the withdrawn drop) |
| unmasked `tl.load(a_ptrs)`, `tl.load(b_ptrs)` ("assume full tile") | covered | SEMANTICS "tl.load" |
| `accumulator += tl.dot(a, b)` (`tl.dot_add`), `float16` | gap | L4 (`DotAcc`) |
| `accumulator.to(tl.float16)`; unmasked `tl.store(c_ptrs, c)` | covered | SEMANTICS "Float conversions", "tl.store" |
| `a_ptrs += BLOCK_SIZE_K`, `tile_idx += NUM_SM` | covered | SEMANTICS "Pointers", "Index arithmetic" |
| `group_size` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `group_size` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(NUM_SM,)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 08-grouped-gemm.py: grouped_matmul_tma_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(tma_configs, key=['group_size'])` (`num_stages`, `num_warps`) | gap | L21 |
| `dtype = tl.float8e4nv if FP8 else tl.float16` | covered | SEMANTICS "Order" |
| `for g in range(group_size)`, scalar loads, `tl.cdiv` | covered | SEMANTICS "Loops", "tl.load", "Helpers are inlined" |
| `if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:` (a runtime `if`) | gap | L28 |
| `tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))` | covered | SEMANTICS "Values" (`tl.intToPtr`) |
| `tl.make_tensor_descriptor(a_ptr, shape=[gm, gk], strides=[lda, 1], block_shape=[…])` (three), scratch from `triton.set_allocator` | gap | L5 |
| `a_desc.load([offs_am, kk * BLOCK_SIZE_K])`, `b_desc.load(…)`, `c_desc.store([offs_cm, offs_cn], c)` on those descriptors | gap | L5 |
| the `while` loop and `tile_idx += NUM_SM` | covered | SEMANTICS "Loops" (`tl.whileLoop`) |
| `accumulator += tl.dot(a, b.T)` (`tl.dot_add`, `b.T` on rank 2), `float16` | gap | L4 |
| the same with `float8e4nv` operands (`FP8`) | gap | L4 (fp8 dots are not on its list) |
| `accumulator.to(dtype)` to `float16` | covered | SEMANTICS "Float conversions" |
| `accumulator.to(tl.float8e4nv)` | gap | L4 (no `CvtAcc`) |
| `group_size` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `group_size` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(META['NUM_SM'],)`, with `NUM_SM=num_sms()` and `FP8=` as constexprs | covered | SEMANTICS "Launches" |

## 09-persistent-matmul.py: matmul_kernel

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])` | gap | L21 |
| `@triton.jit(launch_metadata=_matmul_launch_metadata)` | gap | NEW: (N5) the library does not say whether `launch_metadata` is within the scope of `Faithful`; Triton calls it only when a launch hook is installed (`CompiledKernel.launch_metadata` in `compiler.py`), which point 3 of the scope already excludes, so it is inert, but nothing states it |
| the benchmark's launches under `proton.start("matmul", hook="triton")` | excluded | exclusions.md (a hook: `Compiler.Faithful` covers launches with no hook installed) |
| builtin `min(num_pid_m - first_pid_m, GROUP_SIZE_M)` | gap | NEW: (N3) builtin `min` on an `int32` scalar |
| `tl.cdiv`, `pid // num_pid_in_group`, `pid % group_size_m` | covered | SEMANTICS "Helpers are inlined", "Integer //, %" |
| `offs_am = tl.where(offs_am < M, offs_am, 0)` (the `int32` constant `0`) | covered | `TVA.tl.where`; SEMANTICS "Constants" |
| `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` on a full tile: `multiple_of` holds only with `contig = BLOCK_SIZE_M` (with the sound `contig = 1`, `offs_am[1]` is not a multiple) | gap | L2 |
| the same on a partial tile (`M` or `N` not a multiple of the block, as in the tutorial's own `validate(32, 32, 32)`) | gap | NEW: (N4) after the clamp, `offs_am` is `[0, …, M-1, 0, …]`, so `max_contiguous` is false and the library makes the launch undefined (`TVA.tl.max_contiguous`), while the device computes the product correctly (work/t09_partial_tiles.out, also at `M, N = 200, 136`); no claim can cover the partial tiles the tutorial runs |
| `a_ptrs`, `b_ptrs` blocks; masked loads with `other=0.0` on `float16` | covered | SEMANTICS "Pointers", "tl.load", "Constants" |
| masked loads of `float8e4nv` operands (`other=0.0` is `tl.cvt8`) | gap | L4 (no `CvtAcc`) |
| `tl.dot(a, b, accumulator)`, `float16` | gap | L4 |
| `tl.dot` with `float8e4nv` operands | gap | L4 (fp8 dots are not on its list) |
| `if (c_ptr.dtype.element_ty == tl.float8e4nv):` | covered | SEMANTICS "Expressions with no effect" (attributes of a dtype), "if, elif, else on a compile-time condition" |
| `accumulator.to(tl.float16)`; masked `tl.store` | covered | SEMANTICS "Float conversions", "tl.store" |
| `accumulator.to(tl.float8e4nv)` | gap | L4 (no `CvtAcc`) |
| `M`, `N`, `K`, the strides in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| the same equal to `1` (the tutorial's unit strides are) or at least `2^31` | gap | L29 |
| launch: 1-D grid `cdiv(M, BM) · cdiv(N, BN)` | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 09-persistent-matmul.py: matmul_kernel_tma

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, key=["M", "N", "K", "WARP_SPECIALIZE"])`, each config with `pre_hook=matmul_tma_set_block_size_hook` (sets the host descriptors' `block_shape`) | gap | L21 |
| `@triton.jit(launch_metadata=…)` | gap | NEW: (N5) `launch_metadata` and `Faithful`'s scope |
| the benchmark's launches under `proton` hooks | excluded | exclusions.md (a hook) |
| builtin `min` in the grouping | gap | NEW: (N3) builtin `min` on an `int32` scalar |
| `dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16`; `tl.cdiv`; scalar `//`, `%` | covered | SEMANTICS "Order", "Helpers are inlined", "Integer //, %" |
| `for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE)` | covered | SEMANTICS "Loops" |
| `a_desc.load([offs_am, offs_k])`, `b_desc.load([offs_bn, offs_k])` on `TensorDescriptor.from_tensor` descriptors (zero padding) | covered | SEMANTICS "Tensor descriptors"; `safe_desc_load` |
| `tl.dot(a, b.T, accumulator)`, `float16` and `float8e4nv` | gap | L4 (the fp8 dot must also be added to its list) |
| `accumulator.to(tl.float16)`; `c_desc.store([offs_cm, offs_cn], c)` | covered | SEMANTICS "Float conversions", "Tensor descriptors" (`tl.desc_store`) |
| `accumulator.to(tl.float8e4nv)` | gap | L4 |
| `M`, `N`, `K` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `M`, `N`, `K` equal to `1` or at least `2^31` | gap | L29 |
| launch: 1-D grid, keyword constexprs | covered | SEMANTICS "Launches" |

## 09-persistent-matmul.py: _compute_pid

| Construct | Status | Where |
|---|---|---|
| `tile_id // num_pid_in_group`, `tile_id % group_size_m`, `(tile_id % num_pid_in_group) // group_size_m` | covered | SEMANTICS "Integer //, %"; `safe_idiv1`, `safe_irem1` |
| builtin `min(num_pid_m - first_pid_m, GROUP_SIZE_M)` | gap | NEW: (N3) builtin `min` on an `int32` scalar |
| constexprs (`GROUP_SIZE_M`, `NUM_SMS`) passed to parameters without annotation; a tuple return | covered | SEMANTICS "Helpers are inlined" |

## 09-persistent-matmul.py: matmul_kernel_persistent

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])` | gap | L21 |
| `@triton.jit(launch_metadata=…)` | gap | NEW: (N5) `launch_metadata` and `Faithful`'s scope |
| the benchmark's launches under `proton` hooks | excluded | exclusions.md (a hook) |
| `tile_id_c = start_pid - NUM_SMS`, carried and advanced by `NUM_SMS` | covered | SEMANTICS "Loops", "Index arithmetic" |
| `for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)` | covered | SEMANTICS "Loops" (`tl.forRange`; `flatten` is a scheduling keyword) |
| helper `_compute_pid`, twice | gap | NEW: (N3) inherited from `_compute_pid` (builtin `min`) |
| the hints `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` on values the body defines (not carried), full tiles | gap | L2 |
| the same on partial tiles | gap | NEW: (N4) the clamped offsets make `max_contiguous` false on a partial tile |
| `for ki in range(k_tiles)`; masked loads with `other=0.0` on `float16` | covered | SEMANTICS "Loops", "tl.load", "Constants" |
| masked loads of `float8e4nv` operands (`other=0.0` is `tl.cvt8`) | gap | L4 |
| `tl.dot(a, b, accumulator)`, `float16` and `float8e4nv` | gap | L4 |
| `if (c_ptr.dtype.element_ty == tl.float8e4nv):`, `.to(tl.float16)`, masked store | covered | SEMANTICS "if, elif, else on a compile-time condition", "Float conversions", "tl.store" |
| `accumulator.to(tl.float8e4nv)` | gap | L4 |
| `M`, `N`, `K`, the strides in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| the same equal to `1` (the unit strides) or at least `2^31` | gap | L29 |
| launch: grid `(min(NUM_SMS, tiles),)`, `NUM_SMS=` a constexpr | covered | SEMANTICS "Launches"; `Launch.valid_grid1` |

## 09-persistent-matmul.py: matmul_kernel_tma_persistent

| Construct | Status | Where |
|---|---|---|
| autotune with `pre_hook=matmul_tma_set_block_size_hook` (the `c_desc` block is `[BM, BN // 2]` with `EPILOGUE_SUBTILE`) | gap | L21 |
| `@triton.jit(launch_metadata=…)` | gap | NEW: (N5) `launch_metadata` and `Faithful`'s scope |
| the benchmark's launches under `proton` hooks | excluded | exclusions.md (a hook) |
| `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE)` | covered | SEMANTICS "Loops" |
| helper `_compute_pid` | gap | NEW: (N3) inherited from `_compute_pid` (builtin `min`) |
| `a_desc.load`, `b_desc.load` on host descriptors; `b.T` | covered | SEMANTICS "Tensor descriptors", "Shape changes" |
| `tl.dot(a, b.T, accumulator)`, `float16` and `float8e4nv` | gap | L4 |
| `if EPILOGUE_SUBTILE:` `tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))`, `tl.permute(acc, (0, 2, 1))`, `tl.split(acc)`, two `c_desc.store` | covered | SEMANTICS "Shape changes" (`reshape`, `permute`, `split`), "Tensor descriptors" (`tl.desc_store`) |
| `.to(dtype)` to `float16` | covered | SEMANTICS "Float conversions" |
| `.to(tl.float8e4nv)` | gap | L4 |
| `M`, `N`, `K` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `M`, `N`, `K` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(min(NUM_SMS, tiles),)` | covered | SEMANTICS "Launches" |

## 09-persistent-matmul.py: matmul_kernel_descriptor_persistent

| Construct | Status | Where |
|---|---|---|
| `@triton.autotune(…, prune_configs_by={'early_config_prune': prune_invalid_configs})` | gap | L21 |
| `@triton.jit(launch_metadata=…)` | gap | NEW: (N5) `launch_metadata` and `Faithful`'s scope |
| the benchmark's launches under `proton` hooks | excluded | exclusions.md (a hook) |
| `dtype = c_ptr.dtype.element_ty` | covered | SEMANTICS "Expressions with no effect" |
| `tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[…])` (three; the block `BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2`), scratch from `triton.set_allocator` | gap | L5 |
| the loads and stores through those descriptors | gap | L5 |
| `tl.range(…, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE)` | covered | SEMANTICS "Loops" |
| helper `_compute_pid` | gap | NEW: (N3) inherited from `_compute_pid` (builtin `min`) |
| `tl.dot(a, b.T, accumulator)`, `float16` and `float8e4nv` | gap | L4 |
| epilogue subtiling: `tl.reshape`, `tl.permute`, `tl.split` | covered | SEMANTICS "Shape changes" |
| `.to(dtype)` to `float8e4nv` | gap | L4 |
| `M`, `N`, `K` in `[-2^31, 2^31)` other than `1`: `int32` | covered | SEMANTICS "Index arithmetic" |
| `M`, `N`, `K` equal to `1` or at least `2^31` | gap | L29 |
| launch: grid `(min(NUM_SMS, tiles),)`; `FLATTEN`, `WARP_SPECIALIZE` constexprs | covered | SEMANTICS "Launches" |

## 10-block-scaled-matmul.py: block_scaled_matmul_kernel

| Construct | Status | Where |
|---|---|---|
| the target: the tutorial runs this kernel only with block-scaling hardware (sm_100 or sm_110, `supports_block_scaling`) | gap | NEW: (N9) the library's device facts and evidence are for sm_90; there the kernel runs for mxfp8 and the mixed format through Triton's emulation and meets the tutorial's tolerance (work/t10_on_sm90.out), while the fp4 formats with the tutorial's `num_stages = 4` are rejected on the host (shared memory); a decision is needed: prove it for the H200, or add sm_100 facts (native scaled MMA) |
| `@triton.jit(launch_metadata=_matmul_launch_metadata)` | gap | NEW: (N5) `launch_metadata` and `Faithful`'s scope |
| the benchmark's launches under `proton.start(…, hook="triton")` | excluded | exclusions.md (a hook) |
| `if output_type == 0: … elif …` | covered | SEMANTICS "if, elif, else on a compile-time condition" |
| `pid % num_pid_m`, `pid // num_pid_m` (a constexpr divisor) | covered | SEMANTICS "Integer //, %" |
| `offs_k_a = 0`, `offs_k_b = 0`, `offs_scale_k = 0` carried by the loop (a Python `0` becomes `int32`) | covered | SEMANTICS "Loops", "Constants" |
| `MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2` | covered | SEMANTICS "Order" |
| `tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, disallow_acc_multi_buffer=…)` | covered | SEMANTICS "Loops" (scheduling keywords) |
| `a_desc.load(…)`, `b_desc.load(…)` and the rank-5 `a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])` on `TensorDescriptor.from_tensor` descriptors | covered | SEMANTICS "Tensor descriptors" (rank 1 to 5); `safe_desc_load` |
| `scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)` | covered | SEMANTICS "Shape changes" (row-major `reshape`; `trans` with `dims` is `permute`) |
| `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)` and its `"e2m1"`/`"e2m1"` and `"e4m3"`/`"e4m3"` forms: packed fp4 as `uint8`, `e8m0` (`uint8`) or `float8e4nv` scales, groups of 16 or 32 | covered | SEMANTICS "tl.dot_scaled"; `TVA.Lang.Scaled` (`ScaledCfg.TypesOk`, `scaledValue`); `safe_dot_scaled` |
| the accuracy of `tl.dot_scaled` (on sm_90, Triton's emulation through `bfloat16`/`float16`) | gap | L4 (`DotScaledAcc` is not stated) |
| `accumulator.to(output_dtype)` to `float32` (exact) or `float16`; `c_desc.store` | covered | SEMANTICS "Float conversions", "Tensor descriptors" |
| `accumulator.to(tl.float8e4nv)` (`output_type == 2`) | gap | L4 |
| launch: grid `(cdiv(M, BM) · cdiv(N, BN), 1)`; `M`, `N`, `K`, the block sizes and `NUM_STAGES` as constexpr arguments | covered | SEMANTICS "Launches"; rules/compiler-faithful.md "What it says" (a launch rejected on the host has no outcome) |

## 10-block-scaled-matmul.py: block_scaled_matmul_kernel_cdna4

| Construct | Status | Where |
|---|---|---|
| the kernel as a whole: written for AMD CDNA4, and launched by its host only there (`is_hip_cdna4`) | gap | NEW: (N8) the library targets NVIDIA sm_90 only, and no decision covers or excludes an AMD-only kernel; without the AMD option its body runs on the H200 and equals the tutorial's reference exactly (work/t10_cdna4_on_sm90.out), so it could be covered there |
| `matrix_instr_nonkdim=` at the launch (an AMD option): the CUDA launcher rejects it on the host ("unrecognised" keyword) | excluded | exclusions.md (a compile or launch option `TVA.Launch` does not record) |
| `num_warps=8`, `num_stages=2` | covered | rules/compiler-faithful.md "Scope" point 4 |
| `pid // num_pid_n`, `pid % num_pid_n`; `num_k_iter = tl.cdiv(K, BLOCK_K // 2)` | covered | SEMANTICS "Integer //, %", "Helpers are inlined" |
| offsets `(… + tl.arange(0, BLOCK_M)) % M` and `% N`; unmasked loads of pointer blocks | covered | SEMANTICS "Integer //, %", "Pointers", "tl.load" |
| `if mfma_nonkdim == 32: … elif mfma_nonkdim == 16:` | covered | SEMANTICS "if, elif, else on a compile-time condition" |
| scale loads reshaped to rank 6 and 7, then `permute(0, 3, 1, 4, 2, 5)` or `permute(0, 5, 3, 1, 4, 2, 6)` and `reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)` | covered | SEMANTICS "Shape changes" (any rank) |
| `tl.load(b_ptrs, cache_modifier=None)` | covered | SEMANTICS "tl.load" (cache hints); TRANSCRIPTION §4 `no-op:cache` |
| `accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")` (`tl.dot_scaled_add`) | covered | SEMANTICS "The combine pass's rewrites" (`tl.dot_scaled_add`); `safe_dot_scaled_add` |
| the accuracy of `tl.dot_scaled` | gap | L4 |
| `accumulator.to(c_ptr.type.element_ty)` (`float32`: exact) | covered | SEMANTICS "Float conversions" |
| `int32` with `int64`: `pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)`, `stride_cm * offs_cm[:, None]`, `offs_cm[:, None] < M` | gap | NEW: (N1) the promotion of the `int32` operands to `int64` (sign extension) is not stated |
| `tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")` | gap | NEW: (N7) SEMANTICS says cache, eviction and `volatile` hints change no result only for `tl.load`; nothing says so for the `cache_modifier` of `tl.store` (TRANSCRIPTION §4's `no-op:cache` cites the load) |
| the strides and `M`, `N`, `K` in `[-2^31, 2^31)` other than `1`: `int32` (`stride_ck = 0`, unused) | covered | SEMANTICS "Index arithmetic" |
| the same equal to `1` (the unit strides) or at least `2^31` | gap | L29 |
| launch: grid `(cdiv(M, BM) · cdiv(N, BN), 1)` | covered | SEMANTICS "Launches" |

## 11-programmatic-dependent-launch.py: add_kernel

Excluded: L37 (user decision).

## New gaps

- **NEW** gap — N6, 02 `softmax_kernel`: a launch through a precompiled `CompiledKernel` (`warmup`, then `kernel[grid](…)`) is outside what rules/compiler-faithful.md describes. It skips the JIT's specialisation of the arguments (`1` as a constexpr, `tt.divisibility` 16 for integers divisible by 16 and for 16-byte-aligned pointers), so `Faithful`'s scope should cover it only with arguments that match the specialisation the kernel was compiled with.
- **NEW** gap — N3, 03 `matmul_kernel`: SEMANTICS does not state that Python's builtin `min` (and `max`) in a kernel is `tl.minimum` (`tl.maximum`) on tensors and Python's own on compile-time values (`core.builtin_min`, `code_generator.py:361-364`), nor what integer `tl.minimum` and `tl.maximum` compute (`IntV.min`, `IntV.max` by signedness; only their Lean docstrings say so).
- **NEW** gap — N2, 04 `_dropout`: `tl.where` casts a condition that is not `int1` to `int1` as `x != 0` (`semantic.py:1638-1643`, `:874-877`); SEMANTICS states neither this implicit cast nor the integer-to-`int1` conversion.
- **NEW** gap — N1, 05 `_layer_norm_fwd_fused`: SEMANTICS says the model writes Triton's dtype promotion but does not state it for a binary operator on two tensors of different dtypes (`computation_type_impl`, `integer_promote_impl`): an integer with a float, `float16` with `float32` (and with `bfloat16`), `int32` with `int64`. The fact page should give the rule, as it does for `/` and `%` on `float16` and the reductions.
- **NEW** gap — N1, 05 `_layer_norm_bwd_dwdb`: the same promotion (`float32` plus `float16`).
- **NEW** gap — N1, 06 `_attn_bwd_preprocess`: the same promotion (`float16` times `float32`).
- **NEW** wording — N5, 09 `matmul_kernel`: `@triton.jit(launch_metadata=…)` is not mentioned by `Faithful`'s scope. Triton calls it only when a launch hook is installed (`CompiledKernel.launch_metadata`), which point 3 of the scope already excludes; one sentence saying it is inert would settle it.
- **NEW** gap — N3, 09 `matmul_kernel`: builtin `min` (as above).
- **NEW** gap — N4, 09 `matmul_kernel`: on a partial tile (`M` or `N` not a multiple of the block, as in the tutorial's own `validate(32, 32, 32)`), the offsets clamped by `tl.where(…, 0)` make `tl.max_contiguous` false, so the library makes the launch undefined, though the device computes it correctly (work/t09_partial_tiles.out). Covering these launches needs a reading of a false hint that is weaker than undefined behaviour for such lanes; otherwise claims must exclude partial tiles.
- **NEW** wording — N5, 09 `matmul_kernel_tma`: `launch_metadata` (as above).
- **NEW** gap — N3, 09 `matmul_kernel_tma`: builtin `min` (as above).
- **NEW** gap — N3, 09 `_compute_pid`: builtin `min` (as above).
- **NEW** wording — N5, 09 `matmul_kernel_persistent`: `launch_metadata` (as above).
- **NEW** gap — N3, 09 `matmul_kernel_persistent`: builtin `min`, inherited from `_compute_pid`.
- **NEW** gap — N4, 09 `matmul_kernel_persistent`: partial tiles (as above).
- **NEW** wording — N5, 09 `matmul_kernel_tma_persistent`: `launch_metadata` (as above).
- **NEW** gap — N3, 09 `matmul_kernel_tma_persistent`: builtin `min`, inherited from `_compute_pid`.
- **NEW** wording — N5, 09 `matmul_kernel_descriptor_persistent`: `launch_metadata` (as above).
- **NEW** gap — N3, 09 `matmul_kernel_descriptor_persistent`: builtin `min`, inherited from `_compute_pid`.
- **NEW** gap — N9, 10 `block_scaled_matmul_kernel`: the tutorial runs it only on sm_100 or sm_110, while the library's device facts are for sm_90. On the H200 it runs for mxfp8 and the mixed format through Triton's emulation, and the fp4 formats with the tutorial's `num_stages = 4` are rejected on the host (work/t10_on_sm90.out). Someone must decide whether its proof is about the H200 or needs sm_100 facts.
- **NEW** wording — N5, 10 `block_scaled_matmul_kernel`: `launch_metadata` (as above).
- **NEW** gap — N8, 10 `block_scaled_matmul_kernel_cdna4`: an AMD-only kernel (its launch keyword `matrix_instr_nonkdim` is rejected by the CUDA launcher); the library targets NVIDIA only and no decision covers or excludes it, though its body runs on the H200 without that option and matches the tutorial's reference exactly (work/t10_cdna4_on_sm90.out).
- **NEW** gap — N1, 10 `block_scaled_matmul_kernel_cdna4`: the promotion of `int32` operands to `int64` (as above).
- **NEW** gap — N7, 10 `block_scaled_matmul_kernel_cdna4`: SEMANTICS states that cache hints change no result only for `tl.load`, not for the `cache_modifier` of `tl.store` (`".wt"` here).
