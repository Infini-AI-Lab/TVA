import TVA

/-!
# `ssd_chunk_scan_bf16`: the prepared claim

Source: `source/ssd_chunk_scan.py`. The kernel is `_chunk_scan_fwd_kernel` (lines 92–488); it reads
the module-level names `triton` and `tl` (lines 88–89). The host `chunk_scan_fwd` (lines 491–644)
launches it as SGLang's Mamba-2 prefill does: `bfloat16` `x`, `C` and `D`, `float32` `cb`, `dt`,
`dA_cumsum`, `states` and `initial_states`, `int32` `seq_idx`, `chunk_indices`, `chunk_offsets`,
`batch = 1`, `z = None`, with `HAS_SEQ_IDX = HAS_D = IS_CAUSAL = IS_TRITON_22 = True`,
`D_HAS_HDIM = HAS_Z = False`, `HAS_INITSTATES` either value (`initial_states` given or `None`),
`BLOCK_SIZE_DSTATE = max(next_power_of_2(dstate), 16) ≤ 128`, `BLOCK_SIZE_M = BLOCK_SIZE_N = 64`,
`BLOCK_SIZE_K = 32`, `num_warps = 4`, `num_stages = 1`, on the grid `(cdiv(chunk_size, 64)·
cdiv(headdim, 64), N, nheads)` (`N = nchunks` without initial states); the metadata are what the
host builds from `query_start_loc` (`query_start_loc_to_chunk_indices_offsets`, lines 647–679,
and `make_inputs`, lines 682–732).

* The model: `chunk_scan_kernel` (`HAS_INITSTATES = True`), `chunk_scan_kernel_noinit`
  (`HAS_INITSTATES = False`), `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre` (with the host's metadata `reqOf`, `hostN`, `hostIdx`,
  `hostOff`, and `cb`'s zeros across requests, `CbZero`).
* The specification: `Post`, `Spec`, with the exact output `yRef` computed from the host's inputs
  (the request boundaries `query_start_loc`, `cb`, `x`, `dt`, `dA_cumsum`, `C`, the chunk states,
  the initial states and `D`), its tolerance `tol`, and the value conditions `Vals`.
* The assumptions: `Assumptions`.
* The checks: `consistent`, `satisfiable`, and one covered instance per test case of the source
  (`setup_*`).

The claim to prove is `Statement`.
-/

namespace Claims.SsdChunkScan

open TVA

/-! ## Constants of the source -/

/-- The `float32` `+0` (the Python `0.0` of `other=0.0`, of line 269 and of `tl.where`). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- The `tl.dot` configuration of lines 361 and 420: `bfloat16` operands, the default `float32`
accumulator, the default `input_precision` (`tf32`; no effect on 16-bit operands) and the default
`max_num_imprecise_acc` (`0`, not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-- `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, fixed by the host (line 544). -/
def BM : Nat := 64
def BN : Nat := 64
def BK : Nat := 32

/-! ## The model -/

/-- `_chunk_scan_fwd_kernel` with `HAS_INITSTATES = True` (source lines 92–488). `cb`, `dt`,
`dA_cumsum`, `states`, `initial_states` are `*fp32` pointers, `x`, `out`, `C`, `D` `*bf16`,
`seq_idx`, `chunk_indices`, `chunk_offsets` `*i32`; `chunk_meta_num`, `chunk_size`, `hdim`,
`dstate`, `seqlen`, `nheads_ngroups_ratio` and the strides are `int32` scalar arguments (`Inst.ok` keeps each below
`2^31`; an integer argument equal to `1` is the compile-time constant `1`, and each of its uses in
the kernel, an operand of `+`, `-`, `*`, `//`, a comparison, a `min`, an `other=`, or a value
assigned to a name, builds the constant in the computation dtype with the same value, so the
model's `IntV` is the same: see `model`). `batch` is `1` on every covered launch, the compile-time
constant `1`; its two uses (lines 173–174) build it in `int64`. `z_ptr` and `out_x_ptr` are `None`
and the four `z` strides `0`; no line the launch runs reads them (`HAS_Z = False`), so the model
has no parameter for them. `BLOCK_SIZE_DSTATE`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` are
Lean parameters.

`IS_CAUSAL`, `HAS_D`, `HAS_SEQ_IDX`, `IS_TRITON_22`, `HAS_INITSTATES` are `True` and `D_HAS_HDIM`,
`HAS_Z` `False` on the launches this model covers (`HAS_INITSTATES = False`:
`chunk_scan_kernel_noinit`), and `BLOCK_SIZE_DSTATE ≤ 128` (`SETUP.md`), so each
compile-time `if` visits only its taken branch; the model writes that branch, and the model map
drops the others (`config`). -/
def chunk_scan_kernel
    (cb_ptr : Ptr .f32) (x_ptr out_ptr : Ptr .bf16) (dt_ptr dA_cumsum_ptr : Ptr .f32)
    (seq_idx_ptr : Ptr .i32) (C_ptr : Ptr .bf16) (states_ptr : Ptr .f32) (D_ptr : Ptr .bf16)
    (initstates_ptr : Ptr .f32) (chunk_indices_ptr chunk_offsets_ptr : Ptr .i32)
    (chunk_meta_num chunk_size hdim dstate seqlen nheads_ngroups_ratio : I32)
    (stride_cb_batch stride_cb_chunk stride_cb_head stride_cb_csize_m stride_cb_csize_k
      stride_x_batch stride_x_seqlen stride_x_head stride_x_hdim
      stride_out_batch stride_out_seqlen stride_out_head stride_out_hdim
      stride_dt_batch stride_dt_chunk stride_dt_head stride_dt_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      stride_seq_idx_batch stride_seq_idx_seqlen
      stride_C_batch stride_C_seqlen stride_C_head stride_C_dstate
      stride_states_batch stride_states_chunk stride_states_head stride_states_hdim
      stride_states_dstate
      stride_init_states_batch stride_init_states_head stride_init_states_hdim
      stride_init_states_dstate
      stride_D_head : I32)
    (BLOCK_SIZE_DSTATE BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Kernel where
  body ctx := do
    -- line 172: axis 1 of the grid, an `int32`, converted to `int64` (sign extension)
    let pid_bc : I64 := IntV.cast (tl.program_id ctx 1)
    -- line 173: `pid_bc // batch`, `batch` the constant `1` built in `int64`; the `int64` `//`
    let pid_c : I64 :=
      (← tl.idiv (Tensor.scalar pid_bc) (Tensor.scalar (IntV.ofNat 1 : I64))).item
    -- line 174: `int64` (wrapping)
    let pid_b : I64 := pid_bc - pid_c * (IntV.ofNat 1 : I64)
    -- lines 175–178: `if not HAS_INITSTATES:` not taken; lines 179–180, the `else` branch:
    -- masked scalar loads at the `int64` offset `pid_c`; the mask `pid_c > -1` compares in
    -- `int64` (the Python `-1` built in `int32`, then promoted); `other=0` is the `int32` `0`
    let c_idx : I32 := (← tl.load (Tensor.scalar (chunk_indices_ptr + pid_c))
      (some (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c)))
      (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
    let c_off : I32 := (← tl.load (Tensor.scalar (chunk_offsets_ptr + pid_c))
      (some (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c)))
      (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
    -- line 182: axis 2 of the grid
    let pid_h := tl.program_id ctx 2
    -- line 183: `tl.cdiv(hdim, BLOCK_SIZE_N)` with a runtime `hdim`: `(hdim + 63) // 64`, a
    -- wrapping `int32` addition and the `int32` `//`
    let num_pid_n : I32 := (← tl.idiv
      (Tensor.scalar (hdim + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 184: the `int32` `//`
    let pid_m : I32 :=
      (← tl.idiv (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
    -- line 185: the `int32` `%`
    let pid_n : I32 :=
      (← tl.irem (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
    -- lines 186–190: `int64` product `pid_b * stride` (the stride sign-extended), `int32`
    -- products (wrapping), the `int32` `//`, each `int32` term sign-extended into the `int64`
    -- sum, then a pointer `+`
    let g_cb : I32 :=
      (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar nheads_ngroups_ratio)).item
    let cb_ptr := cb_ptr + (pid_b * (IntV.cast stride_cb_batch : I64) +
      (IntV.cast (c_idx * stride_cb_chunk) : I64) + (IntV.cast (g_cb * stride_cb_head) : I64))
    -- lines 191–195
    let x_ptr := x_ptr + (pid_b * (IntV.cast stride_x_batch : I64) +
      (IntV.cast (c_idx * chunk_size * stride_x_seqlen) : I64) +
      (IntV.cast (pid_h * stride_x_head) : I64))
    -- line 196: the right-hand side, then the pointer `+`
    let dt_ptr := dt_ptr + (pid_b * (IntV.cast stride_dt_batch : I64) +
      (IntV.cast (c_idx * stride_dt_chunk) : I64) + (IntV.cast (pid_h * stride_dt_head) : I64))
    -- lines 197–201
    let dA_cumsum_ptr := dA_cumsum_ptr + (pid_b * (IntV.cast stride_dA_cs_batch : I64) +
      (IntV.cast (c_idx * stride_dA_cs_chunk) : I64) +
      (IntV.cast (pid_h * stride_dA_cs_head) : I64))
    -- lines 202–206 (the second `pid_h // nheads_ngroups_ratio`)
    let g_C : I32 :=
      (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar nheads_ngroups_ratio)).item
    let C_ptr := C_ptr + (pid_b * (IntV.cast stride_C_batch : I64) +
      (IntV.cast (c_idx * chunk_size * stride_C_seqlen) : I64) +
      (IntV.cast (g_C * stride_C_head) : I64))
    -- line 210: `int32` products and sums (wrapping)
    let offs_m : Tensor [BLOCK_SIZE_M] I32 :=
      (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + c_off) + tl.arange 0 BLOCK_SIZE_M
    -- lines 211–216: no parentheses: three pointer `+`s, left to right (the first by an
    -- `int64`, the others by `int32` products)
    let prev_states_ptr := states_ptr + pid_b * (IntV.cast stride_states_batch : I64) +
      c_idx * stride_states_chunk + pid_h * stride_states_head
    -- lines 217–218: names assigned `int32` arguments hold `int32` scalar tensors (an argument
    -- equal to `1` included)
    let prev_states_hdim : I32 := stride_states_hdim
    let prev_states_dstate : I32 := stride_states_dstate
    -- line 220: Python's `min` with a tensor operand is `tl.minimum`, on `int32` the exact
    -- `tl.iminimum`
    let chunk_size_limit : I32 := (tl.iminimum (Tensor.scalar chunk_size)
      (Tensor.scalar (seqlen - c_idx * chunk_size))).item
    -- lines 221–224: `if HAS_SEQ_IDX:` taken
    let seq_idx_ptr := seq_idx_ptr + (pid_b * (IntV.cast stride_seq_idx_batch : I64) +
      (IntV.cast (c_idx * chunk_size * stride_seq_idx_seqlen) : I64))
    -- lines 227–229: `ptr - k` is `ptr + (-k)` (`semantic.py:265-267`); the mask `c_idx >= 1`
    -- (signed `int32`); `other=0` the `int32` `0`
    let seq_idx_prev : I32 := (← tl.load
      (Tensor.scalar (seq_idx_ptr + (- stride_seq_idx_seqlen)))
      (some (Tensor.scalar (IntV.le (IntV.ofInt 1 : I32) c_idx)))
      (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
    -- line 231: `if HAS_INITSTATES:` taken; line 236: a runtime `if` on the signed `int32`
    -- comparison; it merges `prev_states_ptr`, `prev_states_hdim`, `prev_states_dstate`
    -- (`seq_idx_m` belongs to the branch)
    let (prev_states_ptr, prev_states_hdim, prev_states_dstate) ←
      if IntV.lt (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + c_off) chunk_size_limit then do
        -- lines 237–240: an unmasked scalar load
        let seq_idx_m : I32 ← tl.load1
          (seq_idx_ptr + (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + c_off) * stride_seq_idx_seqlen)
        -- lines 245–251: `(A and B) or C` on `int1` scalars, every operand evaluated
        if (IntV.eq c_off (IntV.ofInt 0) && !(IntV.eq seq_idx_prev seq_idx_m)) ||
            IntV.lt (IntV.ofInt 0) c_off then
          -- lines 253–259: two pointer `+`s by `int32` products; the strides
          pure (initstates_ptr + seq_idx_m * stride_init_states_batch +
              pid_h * stride_init_states_head,
            stride_init_states_hdim, stride_init_states_dstate)
        else
          pure (prev_states_ptr, prev_states_hdim, prev_states_dstate)
      else
        pure (prev_states_ptr, prev_states_hdim, prev_states_dstate)
    -- line 261
    let offs_n : Tensor [BLOCK_SIZE_N] I32 :=
      pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- lines 262–264: `other=0.0` the `float32` `+0`; `.to(tl.float32)` the identity
    let dA_cs_m : Tensor [BLOCK_SIZE_M] F32 ← tl.load (dA_cumsum_ptr + offs_m * stride_dA_cs_csize)
      (some (offs_m.ltS chunk_size)) (some (bc zf32))
    -- line 267: `if HAS_INITSTATES:` taken; line 269: the Python `0.0` assigned to a name, a
    -- `float32` scalar tensor holding `+0`
    let dA_cs_m_boundary₀ : F32 := zf32
    -- lines 272–276: a masked scalar load at the `int64` offset `pid_c + 1`; the mask, `and` of
    -- two `int1` scalars (both evaluated; `chunk_meta_num` sign-extended to `int64`);
    -- `other=-1` the `int32` `-1`
    let c_idx_n : I32 := (← tl.load
      (Tensor.scalar (chunk_indices_ptr + (pid_c + (IntV.ofInt 1 : I64))))
      (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c))
        (Tensor.scalar (IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast chunk_meta_num : I64)))))
      (some (Tensor.scalar (IntV.ofInt (-1) : I32)))).item
    -- line 287: a runtime `if` on `(c_idx == c_idx_n) or c_off > 0` (both evaluated); it
    -- merges `chunk_size_limit` and `dA_cs_m_boundary`
    let (chunk_size_limit, dA_cs_m_boundary) ←
      if IntV.eq c_idx c_idx_n || IntV.lt (IntV.ofInt 0) c_off then do
        -- lines 289–293: as lines 272–276; `other=chunk_size`, the `int32` argument
        let c_off_n : I32 := (← tl.load
          (Tensor.scalar (chunk_offsets_ptr + (pid_c + (IntV.ofInt 1 : I64))))
          (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c))
            (Tensor.scalar (IntV.lt (pid_c + (IntV.ofInt 1 : I64))
              (IntV.cast chunk_meta_num : I64)))))
          (some (Tensor.scalar chunk_size))).item
        -- lines 296–297: a runtime `if`; `min` is the exact `int32` `tl.iminimum`
        let chunk_size_limit : I32 :=
          if IntV.eq c_idx c_idx_n then
            (tl.iminimum (Tensor.scalar c_off_n) (Tensor.scalar chunk_size_limit)).item
          else chunk_size_limit
        -- lines 304–308: a masked scalar load; the mask `((c_off - 1) > -1) and (c_off <
        -- chunk_size)` (`int32`, both evaluated); `other=0.0` the `float32` `+0`;
        -- `.to(tl.float32)` the identity
        let dA_cs_m_boundary : F32 := (← tl.load
          (Tensor.scalar (dA_cumsum_ptr + (c_off - IntV.ofInt 1) * stride_dA_cs_csize))
          (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1)) (c_off - IntV.ofInt 1)))
            (Tensor.scalar (IntV.lt c_off chunk_size))))
          (some (Tensor.scalar zf32))).item
        pure (chunk_size_limit, dA_cs_m_boundary)
      else
        pure (chunk_size_limit, dA_cs_m_boundary₀)
    -- lines 310–317: `if not HAS_INITSTATES:` not taken
    -- line 319: overwritten below before any use (line 361)
    let _acc := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
    -- line 324: `IS_TRITON_22 or …`: the compile-time `True` ends the `or`; the branch is taken
    -- lines 326–328: `BLOCK_SIZE_DSTATE ≤ 128`, so `tl.arange(0, BLOCK_SIZE_DSTATE)`
    let offs_k_dstate : Tensor [BLOCK_SIZE_DSTATE] I32 := tl.arange 0 BLOCK_SIZE_DSTATE
    -- lines 329–331: `int32` products and sums, broadcast (`x[:, None]` is `.col`, `x[None, :]`
    -- is `.row`), then a pointer `+`
    let C_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] (Ptr .bf16) :=
      C_ptr + ((offs_m.col BLOCK_SIZE_DSTATE) * stride_C_seqlen +
        (offs_k_dstate.row BLOCK_SIZE_M) * stride_C_dstate)
    -- lines 333–336
    let prev_states_ptrs : Tensor [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] (Ptr .f32) :=
      prev_states_ptr + ((offs_n.row BLOCK_SIZE_DSTATE) * prev_states_hdim +
        (offs_k_dstate.col BLOCK_SIZE_N) * prev_states_dstate)
    -- lines 337–344: `if HAS_SEQ_IDX:` taken, `if not HAS_INITSTATES:` not taken: the IEEE `-`
    -- (two loaded values, no product feeds it; the scalar broadcast), then `tl.exp`
    let scale_m : Tensor [BLOCK_SIZE_M] F32 ←
      tl.exp (← tl.sub dA_cs_m (bc dA_cs_m_boundary))
    -- line 347: `if BLOCK_SIZE_DSTATE <= 128:` taken
    -- lines 348–353: `other=0.0`, the `float32` `+0` converted to the pointer's `bfloat16`
    let C : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] BF16 ← tl.load C_ptrs
      (some (((offs_m.col BLOCK_SIZE_DSTATE).ltS chunk_size_limit).and
        ((offs_k_dstate.row BLOCK_SIZE_M).ltS dstate)))
      (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] zf32)))
    -- lines 355–359
    let prev_states : Tensor [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] F32 ← tl.load prev_states_ptrs
      (some (((offs_k_dstate.col BLOCK_SIZE_N).ltS dstate).and
        ((offs_n.row BLOCK_SIZE_DSTATE).ltS hdim)))
      (some (bc zf32))
    -- line 360: `.to(bfloat16)`, one rounding to nearest even
    let prev_states₁ : Tensor [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] BF16 ← tl.cvt prev_states
    -- line 361: `tl.dot` with the default zero accumulator, then the IEEE `*` by the broadcast
    -- `scale_m[:, None]`
    let acc₀ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.mul
      (← tl.dot cfgBF16 C prev_states₁ (tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32))
      (scale_m.col BLOCK_SIZE_N)
    -- line 383
    let offs_k : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K + c_off
    -- lines 384–386
    let cb_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] (Ptr .f32) :=
      cb_ptr + ((offs_m.col BLOCK_SIZE_K) * stride_cb_csize_m +
        (offs_k.row BLOCK_SIZE_M) * stride_cb_csize_k)
    -- lines 387–389
    let x_ptrs : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] (Ptr .bf16) :=
      x_ptr + ((offs_k.col BLOCK_SIZE_N) * stride_x_seqlen +
        (offs_n.row BLOCK_SIZE_K) * stride_x_hdim)
    -- line 390
    let dt_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .f32) := dt_ptr + offs_k * stride_dt_csize
    -- line 391
    let dA_cumsum_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .f32) :=
      dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    -- lines 392–396: a conditional expression on the constexpr `IS_CAUSAL = True`: only the
    -- `min` is evaluated, the exact `int32` `tl.iminimum`
    let K_MAX : I32 := (tl.iminimum
      (Tensor.scalar ((pid_m + IntV.ofInt 1) * (IntV.ofNat BLOCK_SIZE_M : I32)))
      (Tensor.scalar chunk_size_limit)).item
    -- line 397: `for k in range(0, K_MAX, BLOCK_SIZE_K)` (`int32` bounds); the loop carries
    -- `acc`, `cb_ptrs`, `x_ptrs`, `dt_ptrs`, `dA_cumsum_ptrs`
    let (acc, _cb_ptrs, _x_ptrs, _dt_ptrs, _dA_cumsum_ptrs) ←
      tl.forRange (IntV.ofInt 0 : I32) K_MAX (IntV.ofNat BLOCK_SIZE_K : I32)
        (acc₀, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs)
        fun k (acc, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs) => do
      -- lines 398–402: `other=0.0` the `float32` `+0`; `.to(tl.float32)` the identity
      let cb : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 ← tl.load cb_ptrs
        (some (((offs_m.col BLOCK_SIZE_K).ltS chunk_size).and
          ((offs_k.row BLOCK_SIZE_M).ltS (chunk_size - k))))
        (some (bc zf32))
      -- lines 403–405
      let dA_cs_k : Tensor [BLOCK_SIZE_K] F32 ← tl.load dA_cumsum_ptrs
        (some (offs_k.ltS (chunk_size - k))) (some (bc zf32))
      -- line 408: the IEEE `-` (loaded values) broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_K]`,
      -- `tl.exp`, then the IEEE `*` (its result feeds a product, not an addition)
      let cb₁ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 ← tl.mul cb
        (← tl.exp (← tl.sub (dA_cs_m.col BLOCK_SIZE_K) (dA_cs_k.row BLOCK_SIZE_M)))
      -- line 409
      let dt_k : Tensor [BLOCK_SIZE_K] F32 ← tl.load dt_ptrs
        (some (offs_k.ltS (chunk_size - k))) (some (bc zf32))
      -- line 410: the IEEE `*` by `dt_k` broadcast along axis 0 (its result is converted, not
      -- added)
      let cb₂ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 ← tl.mul cb₁ (dt_k.row BLOCK_SIZE_M)
      -- line 411: `if IS_CAUSAL:` taken; line 412: `offs_m[:, None] >= k + offs_k[None, :]`,
      -- signed `int32`
      let mask : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] Bool :=
        Tensor.le ((k + offs_k).row BLOCK_SIZE_M) (offs_m.col BLOCK_SIZE_K)
      -- line 413: `0.0` built in `float32`
      let cb₃ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 := tl.where mask cb₂ (bc zf32)
      -- line 414: `.to(bfloat16)`, one rounding to nearest even
      let cb₄ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] BF16 ← tl.cvt cb₃
      -- lines 415–419: `other=0.0` converted to `bfloat16`
      let x : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] BF16 ← tl.load x_ptrs
        (some (((offs_k.col BLOCK_SIZE_N).ltS (chunk_size_limit - k)).and
          ((offs_n.row BLOCK_SIZE_K).ltS hdim)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] zf32)))
      -- line 420: `acc += tl.dot(cb, x)`: a zero default accumulator, the product used once:
      -- the combine pass's `tl.dot_add`, with `acc` as the addend
      let acc : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.dot_add cfgBF16 cb₄ x acc
      -- lines 421–424: the constexpr `BLOCK_SIZE_K` built in `int32` times the stride (an
      -- argument equal to `1`: the Python product, the same `int32` value), then a pointer `+`
      -- on each lane (64-bit addresses)
      let cb_ptrs := cb_ptrs.map (fun (q : Ptr .f32) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_cb_csize_k)
      let x_ptrs := x_ptrs.map (fun (q : Ptr .bf16) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_x_seqlen)
      let dt_ptrs := dt_ptrs.map (fun (q : Ptr .f32) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_dt_csize)
      let dA_cumsum_ptrs := dA_cumsum_ptrs.map (fun (q : Ptr .f32) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_dA_cs_csize)
      pure (acc, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs)
    -- line 426
    let offs_out_m : Tensor [BLOCK_SIZE_M] I32 :=
      (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + c_off) + tl.arange 0 BLOCK_SIZE_M
    -- line 427
    let offs_out_n : Tensor [BLOCK_SIZE_N] I32 :=
      pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- line 429: `if HAS_D:` taken; line 430: `if D_HAS_HDIM:` not taken; line 435: an unmasked
    -- scalar load, `.to(tl.float32)` exact
    let D₀ : BF16 ← tl.load1 (D_ptr + pid_h * stride_D_head)
    let D : F32 := (← tl.cvt (Tensor.scalar D₀)).item
    -- lines 436–441: `other=0.0` converted to `bfloat16`; `.to(tl.float32)` exact
    let x_residual₀ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] BF16 ← tl.load
      (x_ptr + ((offs_m.col BLOCK_SIZE_N) * stride_x_seqlen +
        (offs_n.row BLOCK_SIZE_M) * stride_x_hdim))
      (some (((offs_m.col BLOCK_SIZE_N).ltS chunk_size_limit).and
        ((offs_n.row BLOCK_SIZE_M).ltS hdim)))
      (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_N] zf32)))
    let x_residual : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.cvt x_residual₀
    -- line 442: `acc + x_residual * D`: a product feeds the addition, `tl.mul_add` (the two
    -- IEEE operations, or one `fma`)
    let acc ← tl.mul_add x_residual (bc D) acc
    -- line 444: `if HAS_Z:` not taken
    -- lines 476–480
    let out_ptr := out_ptr + (pid_b * (IntV.cast stride_out_batch : I64) +
      (IntV.cast (c_idx * chunk_size * stride_out_seqlen) : I64) +
      (IntV.cast (pid_h * stride_out_head) : I64))
    -- lines 481–483
    let out_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .bf16) :=
      out_ptr + (stride_out_seqlen * (offs_out_m.col BLOCK_SIZE_N) +
        (offs_out_n.row BLOCK_SIZE_M) * stride_out_hdim)
    -- lines 484–488: the store's conversion to the pointer's `bfloat16` (nearest even)
    let out16 : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] BF16 ← tl.cvt acc
    tl.store out_ptrs out16
      (some (((offs_out_m.col BLOCK_SIZE_N).ltS chunk_size_limit).and
        ((offs_out_n.row BLOCK_SIZE_M).ltS hdim)))

/-- `_chunk_scan_fwd_kernel` with `HAS_INITSTATES = False` (source lines 92–488): the host is given
`initial_states = None` and so drops `chunk_indices` and `chunk_offsets` (`None`, with
`chunk_meta_num = 0` and the four initial-state strides `0`), which no line this launch runs reads,
so the model has no parameter for them; the grid's `y` is `batch·nchunks = nchunks`. Line 176 makes
`c_idx` the `int64` `pid_c`: every product with it is `int64`, and `chunk_size_limit` (line 220),
`K_MAX` (lines 392–396) and the loop variable of line 397 are `int64`; a comparison of an `int32`
offset with them sign-extends the offset (`offs_m64`, `offs_k64`, `offs_out_m64`). The other
parameters and flags are as in `chunk_scan_kernel`. -/
def chunk_scan_kernel_noinit
    (cb_ptr : Ptr .f32) (x_ptr out_ptr : Ptr .bf16) (dt_ptr dA_cumsum_ptr : Ptr .f32)
    (seq_idx_ptr : Ptr .i32) (C_ptr : Ptr .bf16) (states_ptr : Ptr .f32) (D_ptr : Ptr .bf16)
    (chunk_size hdim dstate seqlen nheads_ngroups_ratio : I32)
    (stride_cb_batch stride_cb_chunk stride_cb_head stride_cb_csize_m stride_cb_csize_k
      stride_x_batch stride_x_seqlen stride_x_head stride_x_hdim
      stride_out_batch stride_out_seqlen stride_out_head stride_out_hdim
      stride_dt_batch stride_dt_chunk stride_dt_head stride_dt_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      stride_seq_idx_batch stride_seq_idx_seqlen
      stride_C_batch stride_C_seqlen stride_C_head stride_C_dstate
      stride_states_batch stride_states_chunk stride_states_head stride_states_hdim
      stride_states_dstate
      stride_D_head : I32)
    (BLOCK_SIZE_DSTATE BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Kernel where
  body ctx := do
    -- lines 172–174: as in `chunk_scan_kernel`
    let pid_bc : I64 := IntV.cast (tl.program_id ctx 1)
    let pid_c : I64 :=
      (← tl.idiv (Tensor.scalar pid_bc) (Tensor.scalar (IntV.ofNat 1 : I64))).item
    let pid_b : I64 := pid_bc - pid_c * (IntV.ofNat 1 : I64)
    -- line 175: `if not HAS_INITSTATES:` taken; line 176: `c_idx` holds the `int64` `pid_c`;
    -- line 177: the Python `0` assigned to a name, an `int32` scalar tensor
    let c_idx : I64 := pid_c
    let c_off : I32 := IntV.ofInt 0
    -- lines 182–185: as in `chunk_scan_kernel`
    let pid_h := tl.program_id ctx 2
    let num_pid_n : I32 := (← tl.idiv
      (Tensor.scalar (hdim + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    let pid_m : I32 :=
      (← tl.idiv (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
    let pid_n : I32 :=
      (← tl.irem (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
    -- lines 186–190: `c_idx * stride_cb_chunk` an `int64` product (the stride sign-extended)
    let g_cb : I32 :=
      (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar nheads_ngroups_ratio)).item
    let cb_ptr := cb_ptr + (pid_b * (IntV.cast stride_cb_batch : I64) +
      c_idx * (IntV.cast stride_cb_chunk : I64) + (IntV.cast (g_cb * stride_cb_head) : I64))
    -- lines 191–195: `c_idx * chunk_size * stride_x_seqlen`, `int64` products
    let x_ptr := x_ptr + (pid_b * (IntV.cast stride_x_batch : I64) +
      c_idx * (IntV.cast chunk_size : I64) * (IntV.cast stride_x_seqlen : I64) +
      (IntV.cast (pid_h * stride_x_head) : I64))
    -- line 196
    let dt_ptr := dt_ptr + (pid_b * (IntV.cast stride_dt_batch : I64) +
      c_idx * (IntV.cast stride_dt_chunk : I64) + (IntV.cast (pid_h * stride_dt_head) : I64))
    -- lines 197–201
    let dA_cumsum_ptr := dA_cumsum_ptr + (pid_b * (IntV.cast stride_dA_cs_batch : I64) +
      c_idx * (IntV.cast stride_dA_cs_chunk : I64) +
      (IntV.cast (pid_h * stride_dA_cs_head) : I64))
    -- lines 202–206
    let g_C : I32 :=
      (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar nheads_ngroups_ratio)).item
    let C_ptr := C_ptr + (pid_b * (IntV.cast stride_C_batch : I64) +
      c_idx * (IntV.cast chunk_size : I64) * (IntV.cast stride_C_seqlen : I64) +
      (IntV.cast (g_C * stride_C_head) : I64))
    -- line 210: `int32` (`c_off` the `int32` `0`)
    let offs_m : Tensor [BLOCK_SIZE_M] I32 :=
      (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + c_off) + tl.arange 0 BLOCK_SIZE_M
    -- the sign extension each comparison of `offs_m` with an `int64` value makes
    let offs_m64 : Tensor [BLOCK_SIZE_M] I64 := offs_m.castInt
    -- lines 211–216: three pointer `+`s, left to right (the second by an `int64` product)
    let prev_states_ptr := states_ptr + pid_b * (IntV.cast stride_states_batch : I64) +
      c_idx * (IntV.cast stride_states_chunk : I64) + pid_h * stride_states_head
    -- lines 217–218
    let prev_states_hdim : I32 := stride_states_hdim
    let prev_states_dstate : I32 := stride_states_dstate
    -- line 220: `seqlen - c_idx * chunk_size` is `int64`; `min` promotes `chunk_size` to
    -- `int64` (sign extension), the exact `tl.iminimum`
    let chunk_size_limit : I64 := (tl.iminimum (Tensor.scalar (IntV.cast chunk_size : I64))
      (Tensor.scalar ((IntV.cast seqlen : I64) - c_idx * (IntV.cast chunk_size : I64)))).item
    -- lines 221–224
    let seq_idx_ptr := seq_idx_ptr + (pid_b * (IntV.cast stride_seq_idx_batch : I64) +
      c_idx * (IntV.cast chunk_size : I64) * (IntV.cast stride_seq_idx_seqlen : I64))
    -- lines 227–229: the mask `c_idx >= 1` compares in `int64`
    let seq_idx_prev : I32 := (← tl.load
      (Tensor.scalar (seq_idx_ptr + (- stride_seq_idx_seqlen)))
      (some (Tensor.scalar (IntV.le (IntV.ofInt 1 : I64) c_idx)))
      (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
    -- line 231: `if HAS_INITSTATES:` not taken
    -- line 261
    let offs_n : Tensor [BLOCK_SIZE_N] I32 :=
      pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- lines 262–264
    let dA_cs_m : Tensor [BLOCK_SIZE_M] F32 ← tl.load (dA_cumsum_ptr + offs_m * stride_dA_cs_csize)
      (some (offs_m.ltS chunk_size)) (some (bc zf32))
    -- line 267: `if HAS_INITSTATES:` not taken
    -- lines 310–317: `if HAS_SEQ_IDX:` and `if not HAS_INITSTATES:` taken: a masked block load
    -- (the mask in `int64`); `other=-1` the `int32` `-1`
    let seq_idx_m : Tensor [BLOCK_SIZE_M] I32 ← tl.load
      (seq_idx_ptr + offs_m * stride_seq_idx_seqlen)
      (some (offs_m64.ltS chunk_size_limit)) (some (bc (IntV.ofInt (-1) : I32)))
    -- line 319
    let _acc := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
    -- lines 324–336: as in `chunk_scan_kernel`
    let offs_k_dstate : Tensor [BLOCK_SIZE_DSTATE] I32 := tl.arange 0 BLOCK_SIZE_DSTATE
    let C_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] (Ptr .bf16) :=
      C_ptr + ((offs_m.col BLOCK_SIZE_DSTATE) * stride_C_seqlen +
        (offs_k_dstate.row BLOCK_SIZE_M) * stride_C_dstate)
    let prev_states_ptrs : Tensor [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] (Ptr .f32) :=
      prev_states_ptr + ((offs_n.row BLOCK_SIZE_DSTATE) * prev_states_hdim +
        (offs_k_dstate.col BLOCK_SIZE_N) * prev_states_dstate)
    -- lines 337–340: `tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)`: the `int32`
    -- equality with the broadcast scalar; `tl.exp` runs on every lane; `0.0` built in `float32`
    let scale_m : Tensor [BLOCK_SIZE_M] F32 :=
      tl.where (Tensor.eq seq_idx_m (bc seq_idx_prev)) (← tl.exp dA_cs_m) (bc zf32)
    -- lines 347–353: the mask compares in `int64`
    let C : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] BF16 ← tl.load C_ptrs
      (some (((offs_m64.col BLOCK_SIZE_DSTATE).ltS chunk_size_limit).and
        ((offs_k_dstate.row BLOCK_SIZE_M).ltS dstate)))
      (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE] zf32)))
    -- lines 355–361: as in `chunk_scan_kernel`
    let prev_states : Tensor [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] F32 ← tl.load prev_states_ptrs
      (some (((offs_k_dstate.col BLOCK_SIZE_N).ltS dstate).and
        ((offs_n.row BLOCK_SIZE_DSTATE).ltS hdim)))
      (some (bc zf32))
    let prev_states₁ : Tensor [BLOCK_SIZE_DSTATE, BLOCK_SIZE_N] BF16 ← tl.cvt prev_states
    let acc₀ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.mul
      (← tl.dot cfgBF16 C prev_states₁ (tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32))
      (scale_m.col BLOCK_SIZE_N)
    -- line 383 (`c_off` the `int32` `0`), and the sign extension of its comparisons
    let offs_k : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K + c_off
    let offs_k64 : Tensor [BLOCK_SIZE_K] I64 := offs_k.castInt
    -- lines 384–391: as in `chunk_scan_kernel`
    let cb_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] (Ptr .f32) :=
      cb_ptr + ((offs_m.col BLOCK_SIZE_K) * stride_cb_csize_m +
        (offs_k.row BLOCK_SIZE_M) * stride_cb_csize_k)
    let x_ptrs : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] (Ptr .bf16) :=
      x_ptr + ((offs_k.col BLOCK_SIZE_N) * stride_x_seqlen +
        (offs_n.row BLOCK_SIZE_K) * stride_x_hdim)
    let dt_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .f32) := dt_ptr + offs_k * stride_dt_csize
    let dA_cumsum_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .f32) :=
      dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    -- lines 392–396: `min` of the `int32` product (sign-extended) and the `int64` limit
    let K_MAX : I64 := (tl.iminimum
      (Tensor.scalar (IntV.cast ((pid_m + IntV.ofInt 1) * (IntV.ofNat BLOCK_SIZE_M : I32)) : I64))
      (Tensor.scalar chunk_size_limit)).item
    -- line 397: `range(0, K_MAX, BLOCK_SIZE_K)` with an `int64` bound: an `int64` loop variable
    let (acc, _cb_ptrs, _x_ptrs, _dt_ptrs, _dA_cumsum_ptrs) ←
      tl.forRange (IntV.ofInt 0 : I64) K_MAX (IntV.ofNat BLOCK_SIZE_K : I64)
        (acc₀, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs)
        fun k (acc, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs) => do
      -- lines 398–402: `chunk_size - k` is `int64`
      let cb : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 ← tl.load cb_ptrs
        (some (((offs_m.col BLOCK_SIZE_K).ltS chunk_size).and
          ((offs_k64.row BLOCK_SIZE_M).ltS ((IntV.cast chunk_size : I64) - k))))
        (some (bc zf32))
      -- lines 403–405
      let dA_cs_k : Tensor [BLOCK_SIZE_K] F32 ← tl.load dA_cumsum_ptrs
        (some (offs_k64.ltS ((IntV.cast chunk_size : I64) - k))) (some (bc zf32))
      -- line 408
      let cb₁ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 ← tl.mul cb
        (← tl.exp (← tl.sub (dA_cs_m.col BLOCK_SIZE_K) (dA_cs_k.row BLOCK_SIZE_M)))
      -- line 409
      let dt_k : Tensor [BLOCK_SIZE_K] F32 ← tl.load dt_ptrs
        (some (offs_k64.ltS ((IntV.cast chunk_size : I64) - k))) (some (bc zf32))
      -- line 410
      let cb₂ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 ← tl.mul cb₁ (dt_k.row BLOCK_SIZE_M)
      -- lines 411–412: `int64`
      let mask : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] Bool :=
        Tensor.le ((k + offs_k64).row BLOCK_SIZE_M) (offs_m64.col BLOCK_SIZE_K)
      -- lines 413–414
      let cb₃ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] F32 := tl.where mask cb₂ (bc zf32)
      let cb₄ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] BF16 ← tl.cvt cb₃
      -- lines 415–419: the mask in `int64`
      let x : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] BF16 ← tl.load x_ptrs
        (some (((offs_k64.col BLOCK_SIZE_N).ltS (chunk_size_limit - k)).and
          ((offs_n.row BLOCK_SIZE_K).ltS hdim)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] zf32)))
      -- line 420
      let acc : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.dot_add cfgBF16 cb₄ x acc
      -- lines 421–424
      let cb_ptrs := cb_ptrs.map (fun (q : Ptr .f32) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_cb_csize_k)
      let x_ptrs := x_ptrs.map (fun (q : Ptr .bf16) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_x_seqlen)
      let dt_ptrs := dt_ptrs.map (fun (q : Ptr .f32) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_dt_csize)
      let dA_cumsum_ptrs := dA_cumsum_ptrs.map (fun (q : Ptr .f32) =>
        q + (IntV.ofNat BLOCK_SIZE_K : I32) * stride_dA_cs_csize)
      pure (acc, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs)
    -- lines 426–427
    let offs_out_m : Tensor [BLOCK_SIZE_M] I32 :=
      (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + c_off) + tl.arange 0 BLOCK_SIZE_M
    let offs_out_m64 : Tensor [BLOCK_SIZE_M] I64 := offs_out_m.castInt
    let offs_out_n : Tensor [BLOCK_SIZE_N] I32 :=
      pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- lines 429–435
    let D₀ : BF16 ← tl.load1 (D_ptr + pid_h * stride_D_head)
    let D : F32 := (← tl.cvt (Tensor.scalar D₀)).item
    -- lines 436–441: the mask in `int64`
    let x_residual₀ : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] BF16 ← tl.load
      (x_ptr + ((offs_m.col BLOCK_SIZE_N) * stride_x_seqlen +
        (offs_n.row BLOCK_SIZE_M) * stride_x_hdim))
      (some (((offs_m64.col BLOCK_SIZE_N).ltS chunk_size_limit).and
        ((offs_n.row BLOCK_SIZE_M).ltS hdim)))
      (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_N] zf32)))
    let x_residual : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.cvt x_residual₀
    -- line 442
    let acc ← tl.mul_add x_residual (bc D) acc
    -- lines 476–480: `int64` products with `c_idx`
    let out_ptr := out_ptr + (pid_b * (IntV.cast stride_out_batch : I64) +
      c_idx * (IntV.cast chunk_size : I64) * (IntV.cast stride_out_seqlen : I64) +
      (IntV.cast (pid_h * stride_out_head) : I64))
    -- lines 481–483
    let out_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .bf16) :=
      out_ptr + (stride_out_seqlen * (offs_out_m.col BLOCK_SIZE_N) +
        (offs_out_n.row BLOCK_SIZE_M) * stride_out_hdim)
    -- lines 484–488: the mask in `int64`
    let out16 : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] BF16 ← tl.cvt acc
    tl.store out_ptrs out16
      (some (((offs_out_m64.col BLOCK_SIZE_N).ltS chunk_size_limit).and
        ((offs_out_n.row BLOCK_SIZE_M).ltS hdim)))


/-! ## The conditions -/

/-- What the host chooses: one launch of `chunk_scan_fwd` with `seq_idx` and `D` given, `z = None`,
`batch = 1`, and `initial_states` with `chunk_indices`, `chunk_offsets` given or not (`hasInit`,
`SETUP.md`). The request
boundaries `query_start_loc` (`qsl`) are the host's input from which it builds the metadata; the
buffers are given by their storage (`Buf`). `x` and `C` are row slices of a wider buffer (SGLang's
convolution output) with their own token strides `xT`, `cT`; every other tensor is contiguous, so
the strides the host passes are the shapes' products. -/
structure Inst where
  /-- `HAS_INITSTATES`: `initial_states` (and the metadata) given; `False` when SGLang passes
  `None` (no request of the batch has a cached prefix). -/
  hasInit : Bool
  /-- `nheads` (`x.shape[2]`): the grid's `z`. -/
  H : Nat
  /-- `headdim` (`x.shape[3]`, `hdim`). -/
  P : Nat
  /-- `dstate` (`C.shape[3]`). -/
  N : Nat
  /-- `ngroups` (`C.shape[2]`). -/
  G : Nat
  /-- `chunk_size` (`dt.shape[3]`). -/
  cs : Nat
  /-- `seqlen` (`x.shape[1]`, the packed tokens). -/
  T : Nat
  /-- `num_requests` (`initial_states.shape[0]`). -/
  nreq : Nat
  /-- `query_start_loc`: request `r` holds the tokens `qsl r ≤ t < qsl (r + 1)`. -/
  qsl : Nat → Nat
  /-- The grid's `y`: `N = len(chunk_offsets) = len(chunk_indices)` (passed as `chunk_meta_num`)
  with initial states, `nchunks` without. -/
  nmeta : Nat
  /-- `BLOCK_SIZE_DSTATE = max(next_power_of_2(dstate), 16)`. -/
  BSD : Nat
  /-- `x.stride(1)`: the token stride of `x` (`x.stride(2) = headdim`, `x.stride(3) = 1`). -/
  xT : Nat
  /-- `C.stride(1)`: the token stride of `C` (`C.stride(2) = dstate`, `C.stride(3) = 1`). -/
  cT : Nat
  /-- The storage of `cb` (`float32`, `[1, nchunks, ngroups, chunk_size, chunk_size]`). -/
  cb : Buf .f32
  /-- The storage of `x` (`bfloat16`). -/
  x : Buf .bf16
  /-- The storage of `dt` (`float32`, `[1, nheads, nchunks, chunk_size]`). -/
  dt : Buf .f32
  /-- The storage of `dA_cumsum` (`float32`, `[1, nheads, nchunks, chunk_size]`). -/
  dA : Buf .f32
  /-- The storage of `seq_idx` (`int32`, `[1, seqlen]`). -/
  seqIdx : Buf .i32
  /-- The storage of `C` (`bfloat16`). -/
  C : Buf .bf16
  /-- The storage of `states` (`float32`, `[1, nchunks, nheads, headdim, dstate]`). -/
  states : Buf .f32
  /-- The storage of `D` (`bfloat16`, `[nheads]`). -/
  D : Buf .bf16
  /-- The storage of `initial_states` (`float32`, `[num_requests, nheads, headdim, dstate]`). -/
  initst : Buf .f32
  /-- The storage of `chunk_indices` (`int32`, `[N]`). -/
  chunkIdx : Buf .i32
  /-- The storage of `chunk_offsets` (`int32`, `[N]`). -/
  chunkOffs : Buf .i32
  /-- The storage of `out` (`bfloat16`, `[1, seqlen, nheads, headdim]`, given by the caller). -/
  out : Buf .bf16

/-- `nchunks = cdiv(seqlen, chunk_size)` (`dt.shape[2]`). -/
def nc (d : Inst) : Nat := (d.T + d.cs - 1) / d.cs

/-- `nheads // ngroups` (line 588), passed as `nheads_ngroups_ratio`. -/
def ratio (d : Inst) : Nat := d.H / d.G

/-- The grid's `x`: `cdiv(chunk_size, 64) · cdiv(headdim, 64)` (lines 545–549). -/
def gx (d : Inst) : Nat := ((d.cs + BM - 1) / BM) * ((d.P + BN - 1) / BN)

/-- `headdim · dstate`, one head's state. -/
def PN (d : Inst) : Nat := d.P * d.N

/-- Element `(0, c, g, i, j)` of `cb`. -/
def cbIdx (d : Inst) (c g i j : Nat) : Nat := c * (d.G * (d.cs * d.cs)) + g * (d.cs * d.cs) + i * d.cs + j

/-- Element `(0, t, h, p)` of `x`. -/
def xIdx (d : Inst) (t h p : Nat) : Nat := t * d.xT + h * d.P + p

/-- Element `(0, t, h, p)` of `out` (contiguous). -/
def outIdx (d : Inst) (t h p : Nat) : Nat := t * (d.H * d.P) + h * d.P + p

/-- Element `(0, h, c, i)` of `dt` and of `dA_cumsum`. -/
def dtIdx (d : Inst) (h c i : Nat) : Nat := h * (nc d * d.cs) + c * d.cs + i

/-- Element `(0, t, g, n)` of `C`. -/
def cIdx (d : Inst) (t g n : Nat) : Nat := t * d.cT + g * d.N + n

/-- Element `(0, c, h, p, n)` of `states`. -/
def stIdx (d : Inst) (c h p n : Nat) : Nat := c * (d.H * PN d) + h * PN d + p * d.N + n

/-- Element `(r, h, p, n)` of `initial_states`. -/
def inIdx (d : Inst) (r h p n : Nat) : Nat := r * (d.H * PN d) + h * PN d + p * d.N + n

/-! ### The host's metadata, from `query_start_loc` -/

/-- The requests after the first, `1, …, nreq − 1`: the host's loop over
`zip(cu_seqlens[:-1], cu_seqlens[1:])` (line 666) visits their starts `qsl r` and ends
`qsl (r + 1)`. -/
def laterReqs (d : Inst) : List Nat := List.range' 1 (d.nreq - 1)

/-- The request of token `t`: the number of later requests that start at or before `t`
(`seq_idx[0, t]`, `torch.repeat_interleave(arange(nreq), seqlens)`, line 700). -/
def reqOf (d : Inst) (t : Nat) : Nat :=
  ((laterReqs d).filter (fun r => decide (d.qsl r ≤ t))).length

/-- The host's `N` (lines 656–659): `ceil(seqlen / chunk_size)` plus one per later request whose
start is not a multiple of `chunk_size`. -/
def hostN (d : Inst) : Nat :=
  (d.T + d.cs - 1) / d.cs + ((laterReqs d).filter (fun r => decide (d.qsl r % d.cs ≠ 0))).length

/-- One trip of the host's loop (lines 666–677) on request `r`, `s = qsl r`, `e = qsl (r + 1)`:
`p` counts the insertions, `_s = s // chunk_size + p`, `_e = e // chunk_size + p + (e % chunk_size
> 0)`, `chunk_indices[_s:_e] -= p`, `chunk_offsets[_s] = s % chunk_size`. The state is `p`, the
indices and the offsets. -/
def metaStep (d : Inst) (st : Nat × (Nat → Nat) × (Nat → Nat)) (r : Nat) :
    Nat × (Nat → Nat) × (Nat → Nat) :=
  let s := d.qsl r
  let e := d.qsl (r + 1)
  let p := st.1 + (if s % d.cs > 0 then 1 else 0)
  let s' := s / d.cs + p
  let e' := e / d.cs + p + (if e % d.cs > 0 then 1 else 0)
  (p, fun k => if s' ≤ k ∧ k < e' then st.2.1 k - p else st.2.1 k,
    Function.update st.2.2 s' (s % d.cs))

/-- The host's metadata (lines 660–677): `chunk_indices` starts as `arange(N)`, `chunk_offsets` as
zeros, then the loop's updates in order. -/
def hostMeta (d : Inst) : Nat × (Nat → Nat) × (Nat → Nat) :=
  (laterReqs d).foldl (metaStep d) (0, fun k => k, fun _ => 0)

/-- `chunk_indices` as the host builds it. -/
def hostIdx (d : Inst) : Nat → Nat := (hostMeta d).2.1

/-- `chunk_offsets` as the host builds it. -/
def hostOff (d : Inst) : Nat → Nat := (hostMeta d).2.2

/-- The instances covered (README, Conditions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- At least one head; the grid's `z` within CUDA's limit `65535`. -/
  H_pos : 0 < d.H
  H_lt : d.H < 2 ^ 16
  /-- At least one channel and one state element. -/
  P_pos : 0 < d.P
  N_pos : 0 < d.N
  /-- `ngroups ≥ 1` divides `nheads` (asserted, line 510). -/
  G_pos : 0 < d.G
  G_dvd : d.G ∣ d.H
  /-- `BLOCK_SIZE_DSTATE = max(next_power_of_2(dstate), 16)` (line 633), at most `128`: a power of
  two with `16 ≤ BSD`, `dstate ≤ BSD`, and `BSD = 16` or `BSD < 2·dstate` (SETUP.md: `dstate ≤
  128`). -/
  bsd_pow2 : ∃ a, d.BSD = 2 ^ a
  bsd_ge : 16 ≤ d.BSD
  bsd_le : d.BSD ≤ 128
  N_le : d.N ≤ d.BSD
  bsd_min : d.BSD = 16 ∨ d.BSD < 2 * d.N
  /-- `chunk_size` is a power of two (the code base asserts it, `ssd_combined.py:77`), at most
  `2^14` (SETUP.md: 8 to 256). -/
  cs_pow2 : ∃ a, d.cs = 2 ^ a
  cs_le : d.cs ≤ 2 ^ 14
  /-- At least one request; `query_start_loc` starts at `0`, every request has at least one token,
  and the last ends at `seqlen` (the packed batch). -/
  nreq_pos : 0 < d.nreq
  qsl_zero : d.qsl 0 = 0
  qsl_lt : ∀ r, r < d.nreq → d.qsl r < d.qsl (r + 1)
  qsl_end : d.qsl d.nreq = d.T
  /-- The grid's `y`: the host's `N = len(chunk_offsets)` with initial states, `nchunks` without
  (lines 545–549), within CUDA's limit. -/
  nmeta_eq : d.nmeta = if d.hasInit then hostN d else nc d
  nmeta_lt : d.nmeta < 2 ^ 16
  /-- The grid has fewer than `2^31` programs. -/
  grid_lt : gx d * d.nmeta * d.H < 2 ^ 31
  /-- `x`'s rows hold the heads and `C`'s rows the groups (the slices of a wider row). -/
  xT_ge : d.H * d.P ≤ d.xT
  cT_ge : d.G * d.N ≤ d.cT
  /-- `x.stride(0) = seqlen·xT`, `C.stride(0) = seqlen·cT` and `out.stride(0) = seqlen·H·P` are
  `int32` arguments, the class the model takes (README, Exclusions). Then line 193's `int32`
  product `c_idx * chunk_size * stride_x_seqlen` (with initial states; `int64` without), at most
  `(seqlen − 1)·xT`, does not wrap, nor do lines 204 and 478's. -/
  x_arg : d.T * d.xT < 2 ^ 31
  c_arg : d.T * d.cT < 2 ^ 31
  out_arg : d.T * (d.H * d.P) < 2 ^ 31
  /-- The in-chunk `int32` offsets (lines 329–331, 387–389, 436–438, 481–483: rows below
  `2·chunk_size + 64` times the token stride, plus the channels or state elements) do not
  wrap. -/
  x_off : (2 * d.cs + 64) * d.xT + 64 < 2 ^ 31
  c_off : (2 * d.cs + 64) * d.cT + 2 * d.BSD < 2 ^ 31
  out_off : (2 * d.cs + 64) * (d.H * d.P) + 64 < 2 ^ 31
  /-- `cb.stride(0) = nchunks·ngroups·chunk_size²` (the `int32` product `c_idx * stride_cb_chunk`,
  line 188), `dt.stride(0) = dA_cumsum.stride(0) = nheads·nchunks·chunk_size` (lines 196, 200) and
  `states.stride(0) = nchunks·nheads·headdim·dstate` (line 214) are `int32` arguments, so these
  products do not wrap. -/
  cb_i32 : nc d * (d.G * (d.cs * d.cs)) < 2 ^ 31
  dt_i32 : d.H * (nc d * d.cs) < 2 ^ 31
  st_i32 : nc d * (d.H * PN d) < 2 ^ 31
  /-- **A limit of the kernel** (README, Exclusions): the `int32` product `seq_idx_m *
  stride_init_states_batch` (line 255) does not wrap: `(nreq − 1)·nheads·headdim·dstate <
  2^31`. -/
  init_i32 : d.hasInit = true → (d.nreq - 1) * (d.H * PN d) < 2 ^ 31
  /-- Every element the indices name lies in its tensor's storage. -/
  cb_len : nc d * (d.G * (d.cs * d.cs)) ≤ d.cb.len
  x_len : (d.T - 1) * d.xT + d.H * d.P ≤ d.x.len
  dt_len : d.H * (nc d * d.cs) ≤ d.dt.len
  dA_len : d.H * (nc d * d.cs) ≤ d.dA.len
  seq_len : d.T ≤ d.seqIdx.len
  C_len : (d.T - 1) * d.cT + d.G * d.N ≤ d.C.len
  states_len : nc d * (d.H * PN d) ≤ d.states.len
  D_len : d.H ≤ d.D.len
  init_len : d.hasInit = true → d.nreq * (d.H * PN d) ≤ d.initst.len
  idx_len : d.hasInit = true → d.nmeta ≤ d.chunkIdx.len
  offs_len : d.hasInit = true → d.nmeta ≤ d.chunkOffs.len
  out_len : d.T * (d.H * d.P) ≤ d.out.len
  /-- `out` shares no byte with the inputs (SGLang's preallocated output, `mamba.py:549-565`). -/
  disj : Disjoint d.out.bytes d.cb.bytes ∧ Disjoint d.out.bytes d.x.bytes ∧
    Disjoint d.out.bytes d.dt.bytes ∧ Disjoint d.out.bytes d.dA.bytes ∧
    Disjoint d.out.bytes d.seqIdx.bytes ∧ Disjoint d.out.bytes d.C.bytes ∧
    Disjoint d.out.bytes d.states.bytes ∧ Disjoint d.out.bytes d.D.bytes ∧
    Disjoint d.out.bytes d.initst.bytes ∧ Disjoint d.out.bytes d.chunkIdx.bytes ∧
    Disjoint d.out.bytes d.chunkOffs.bytes

/-- The model on an instance: the launch of lines 544–643, with the strides the host passes
(contiguous tensors, and `x`, `C` with the token strides `xT`, `cT`; the launch passes `dt.stride(0),
stride(2), stride(1), stride(3)` and the same for `dA_cumsum`). An integer argument equal to `1` is
the compile-time constant `1`; each use of it in the kernel builds the constant in the computation
dtype with the same value, so the model's `IntV.ofNat 1` is the same. Without initial states the
host passes `None` for `initial_states`, `chunk_indices`, `chunk_offsets`, `0` for
`chunk_meta_num` and the initial-state strides, none of which `chunk_scan_kernel_noinit` reads. -/
def model (d : Inst) : Kernel :=
  if d.hasInit then chunk_scan_kernel (d.cb.ptr 0) (d.x.ptr 0) (d.out.ptr 0) (d.dt.ptr 0) (d.dA.ptr 0)
    (d.seqIdx.ptr 0) (d.C.ptr 0) (d.states.ptr 0) (d.D.ptr 0) (d.initst.ptr 0)
    (d.chunkIdx.ptr 0) (d.chunkOffs.ptr 0)
    (IntV.ofNat d.nmeta) (IntV.ofNat d.cs) (IntV.ofNat d.P) (IntV.ofNat d.N) (IntV.ofNat d.T)
    (IntV.ofNat (ratio d))
    (IntV.ofNat (nc d * (d.G * (d.cs * d.cs)))) (IntV.ofNat (d.G * (d.cs * d.cs)))
    (IntV.ofNat (d.cs * d.cs)) (IntV.ofNat d.cs) (IntV.ofNat 1)
    (IntV.ofNat (d.T * d.xT)) (IntV.ofNat d.xT) (IntV.ofNat d.P) (IntV.ofNat 1)
    (IntV.ofNat (d.T * (d.H * d.P))) (IntV.ofNat (d.H * d.P)) (IntV.ofNat d.P) (IntV.ofNat 1)
    (IntV.ofNat (d.H * (nc d * d.cs))) (IntV.ofNat d.cs) (IntV.ofNat (nc d * d.cs)) (IntV.ofNat 1)
    (IntV.ofNat (d.H * (nc d * d.cs))) (IntV.ofNat d.cs) (IntV.ofNat (nc d * d.cs)) (IntV.ofNat 1)
    (IntV.ofNat d.T) (IntV.ofNat 1)
    (IntV.ofNat (d.T * d.cT)) (IntV.ofNat d.cT) (IntV.ofNat d.N) (IntV.ofNat 1)
    (IntV.ofNat (nc d * (d.H * PN d))) (IntV.ofNat (d.H * PN d)) (IntV.ofNat (PN d))
    (IntV.ofNat d.N) (IntV.ofNat 1)
    (IntV.ofNat (d.H * PN d)) (IntV.ofNat (PN d)) (IntV.ofNat d.N) (IntV.ofNat 1)
    (IntV.ofNat 1)
    d.BSD BM BN BK
  else chunk_scan_kernel_noinit (d.cb.ptr 0) (d.x.ptr 0) (d.out.ptr 0) (d.dt.ptr 0) (d.dA.ptr 0)
    (d.seqIdx.ptr 0) (d.C.ptr 0) (d.states.ptr 0) (d.D.ptr 0)
    (IntV.ofNat d.cs) (IntV.ofNat d.P) (IntV.ofNat d.N) (IntV.ofNat d.T) (IntV.ofNat (ratio d))
    (IntV.ofNat (nc d * (d.G * (d.cs * d.cs)))) (IntV.ofNat (d.G * (d.cs * d.cs)))
    (IntV.ofNat (d.cs * d.cs)) (IntV.ofNat d.cs) (IntV.ofNat 1)
    (IntV.ofNat (d.T * d.xT)) (IntV.ofNat d.xT) (IntV.ofNat d.P) (IntV.ofNat 1)
    (IntV.ofNat (d.T * (d.H * d.P))) (IntV.ofNat (d.H * d.P)) (IntV.ofNat d.P) (IntV.ofNat 1)
    (IntV.ofNat (d.H * (nc d * d.cs))) (IntV.ofNat d.cs) (IntV.ofNat (nc d * d.cs)) (IntV.ofNat 1)
    (IntV.ofNat (d.H * (nc d * d.cs))) (IntV.ofNat d.cs) (IntV.ofNat (nc d * d.cs)) (IntV.ofNat 1)
    (IntV.ofNat d.T) (IntV.ofNat 1)
    (IntV.ofNat (d.T * d.cT)) (IntV.ofNat d.cT) (IntV.ofNat d.N) (IntV.ofNat 1)
    (IntV.ofNat (nc d * (d.H * PN d))) (IntV.ofNat (d.H * PN d)) (IntV.ofNat (PN d))
    (IntV.ofNat d.N) (IntV.ofNat 1)
    (IntV.ofNat 1)
    d.BSD BM BN BK

/-- The host's launch (lines 544–643): the grid `(cdiv(chunk_size, 64)·cdiv(headdim, 64), N,
nheads)` (`N = nchunks` without initial states); the source's block shapes (`[64]`: `offs_m`,
`offs_n`, `dA_cs_m`, `scale_m`, `offs_out_m`, `offs_out_n`; `[BSD]`: `offs_k_dstate`; `[32]`: `offs_k`, `dt_k`, `dA_cs_k`; `[64,
BSD]`: `C`; `[BSD, 64]`: `prev_states`; `[64, 64]`: `acc`, `x_residual`, the output tile; `[64,
32]`: `cb`; `[32, 64]`: `x`; and the expanded `[64, 1]`, `[1, BSD]`, `[1, 64]`, `[BSD, 1]`, `[1,
32]`, `[32, 1]`); `enable_fp_fusion` at Triton's default (`true`); `num_ctas = 1`, no
`launch_pdl`, `debug = false`. `num_warps = 4` and `num_stages = 1` change no meaning. -/
def launch (d : Inst) : Launch :=
  { grid := ![gx d, d.nmeta, d.H], fusion := true,
    blocks := [[BM], [d.BSD], [BK], [BM, d.BSD], [d.BSD, BN], [BM, BN], [BM, BK], [BK, BN],
      [BM, 1], [1, d.BSD], [1, BN], [d.BSD, 1], [1, BK], [BK, 1]] }

/-- The `int32` value at element `i` of an `int32` buffer in memory `m` (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some v => v.toInt
  | none => 0

/-- The value of element `i` of a `float32` buffer in memory `m` (`0` when absent). -/
def valF (m : Mem) (b : Buf .f32) (i : Nat) : ℚ :=
  match m.read .f32 (b.ptr (i : Int)).addr with
  | some v => v.toRat
  | none => 0

/-- The `bfloat16` value at element `i` of a buffer, if present. -/
def rdB (m : Mem) (b : Buf .bf16) (i : Nat) : Option BF16 := m.read .bf16 (b.ptr (i : Int)).addr

/-- The value of element `i` of a `bfloat16` buffer (`0` when absent). -/
def valB (m : Mem) (b : Buf .bf16) (i : Nat) : ℚ :=
  match rdB m b i with
  | some v => v.toRat
  | none => 0

/-- Element `i` of a `float32` buffer is a finite value. -/
def finF (m : Mem) (b : Buf .f32) (i : Nat) : Prop :=
  ∃ v : F32, m.read .f32 (b.ptr (i : Int)).addr = some v ∧ v.isFinite = true

/-- Element `i` of a `bfloat16` buffer is a finite value. -/
def finB (m : Mem) (b : Buf .bf16) (i : Nat) : Prop :=
  ∃ v : BF16, rdB m b i = some v ∧ v.isFinite = true

/-- The metadata are the host's: `seq_idx[0, t]` is the request of token `t`, and
with initial states `chunk_indices`, `chunk_offsets` are what
`query_start_loc_to_chunk_indices_offsets` builds (lines 647–679), all from `query_start_loc`. -/
def MetaOk (d : Inst) (m : Mem) : Prop :=
  (∀ t, t < d.T → m.read .i32 (d.seqIdx.ptr (t : Int)).addr = some (IntV.ofNat (reqOf d t))) ∧
  (d.hasInit = true → ∀ k, k < d.nmeta →
    m.read .i32 (d.chunkIdx.ptr (k : Int)).addr = some (IntV.ofNat (hostIdx d k))) ∧
  (d.hasInit = true → ∀ k, k < d.nmeta →
    m.read .i32 (d.chunkOffs.ptr (k : Int)).addr = some (IntV.ofNat (hostOff d k)))

/-- Without initial states, `cb` is zero across requests: for tokens `s ≤ t` of one physical chunk
that belong to different requests, `cb[0, c, g, t mod chunk, s mod chunk]` is a zero (SGLang's
`_bmm_chunk_fwd` with `seq_idx` writes `+0` there, `ssd_combined.py:165`). The kernel's loop then
sums over the whole physical chunk (lines 397–420), and these entries add nothing. -/
def CbZero (d : Inst) (m : Mem) : Prop :=
  d.hasInit = false → ∀ t s g, t < d.T → t / d.cs * d.cs ≤ s → s ≤ t → reqOf d s ≠ reqOf d t →
    g < d.G → ∃ v : F32,
      m.read .f32 (d.cb.ptr (cbIdx d (t / d.cs) g (t % d.cs) (s - t / d.cs * d.cs) : Int)).addr =
        some v ∧ v.isFinite = true ∧ v.toRat = 0

/-- The inputs covered: the twelve buffers are valid (aligned, inside `[0, 2^64)`, every byte
present; without initial states the host's `None`s are empty buffers), the metadata are the host's
(`MetaOk`), and without initial states `cb` is zero across requests (`CbZero`). Any other `float32`
and `bfloat16` values: `Post` says what holds when the values are covered (`Vals`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.cb.Valid init ∧ d.x.Valid init ∧ d.dt.Valid init ∧ d.dA.Valid init ∧ d.seqIdx.Valid init ∧
    d.C.Valid init ∧ d.states.Valid init ∧ d.D.Valid init ∧ d.initst.Valid init ∧
    d.chunkIdx.Valid init ∧ d.chunkOffs.Valid init ∧ d.out.Valid init ∧ MetaOk d init ∧
    CbZero d init

/-! ## The specification -/

/-! ### The reference, from the host's inputs -/

/-- The physical chunk of token `t`. -/
def chk (d : Inst) (t : Nat) : Nat := t / d.cs

/-- The first token of `t`'s logical chunk: the later of its chunk's first token and its request's
first token. -/
def lTok (d : Inst) (t : Nat) : Nat := max (chk d t * d.cs) (d.qsl (reqOf d t))

/-- Its offset in the physical chunk. -/
def lOff (d : Inst) (t : Nat) : Nat := lTok d t - chk d t * d.cs

/-- Token `t`'s logical chunk starts from its request's initial state: the request is not the
first and starts in `t`'s chunk. (The first request starts at token `0`; in chunk `0` its state is
`states[0, 0]`, which the host's state passing sets to its initial state.) -/
def fromInit (d : Inst) (t : Nat) : Bool :=
  decide (1 ≤ reqOf d t) && decide (chk d t * d.cs ≤ d.qsl (reqOf d t))

/-- The group of head `h`: `h // (nheads // ngroups)`. -/
def grp (d : Inst) (h : Nat) : Nat := h / ratio d

/-- `x[0, s, h, p]` as a real. -/
noncomputable def xv (m₀ : Mem) (d : Inst) (s h p : Nat) : ℝ := (valB m₀ d.x (xIdx d s h p) : ℝ)

/-- `C[0, t, g, n]` as a real. -/
noncomputable def Cv (m₀ : Mem) (d : Inst) (t g n : Nat) : ℝ := (valB m₀ d.C (cIdx d t g n) : ℝ)

/-- `dA_cumsum[0, h, c, i]` as a real. -/
noncomputable def dAv (m₀ : Mem) (d : Inst) (h c i : Nat) : ℝ := (valF m₀ d.dA (dtIdx d h c i) : ℝ)

/-- `dt[0, h, c, i]` as a real. -/
noncomputable def dtv (m₀ : Mem) (d : Inst) (h c i : Nat) : ℝ := (valF m₀ d.dt (dtIdx d h c i) : ℝ)

/-- `D[h]` as a real. -/
noncomputable def Dv (m₀ : Mem) (d : Inst) (h : Nat) : ℝ := (valB m₀ d.D h : ℝ)

/-- `cb[0, c, g, i, j]` as a real. -/
noncomputable def cbv (m₀ : Mem) (d : Inst) (c g i j : Nat) : ℝ := (valF m₀ d.cb (cbIdx d c g i j) : ℝ)

/-- **The state entering `t`'s logical chunk**, element `(p, n)` for head `h`: when its logical
chunk starts `t`'s request (`fromInit`), the request's initial state (`0` without initial
states: the request starts from the zero state); otherwise `states[0, c]`, the state entering the
physical chunk `c` of the request running there, state passing's output (its `out[0]` is the first
request's initial state, `0` without initial states). -/
noncomputable def Sv (m₀ : Mem) (d : Inst) (t h p n : Nat) : ℝ :=
  if fromInit d t then
    (if d.hasInit then (valF m₀ d.initst (inIdx d (reqOf d t) h p n) : ℝ) else 0)
  else (valF m₀ d.states (stIdx d (chk d t) h p n) : ℝ)

/-- Without initial states the kernel zeroes the state term of a row whose logical chunk starts
its request (line 340's `tl.where`), after reading out `states[0, c]` all the same. -/
def gated (d : Inst) (t : Nat) : Bool := !d.hasInit && fromInit d t

/-- The state the kernel reads out at token `t` (lines 355–361): `Sv`, or `states[0, c]` on a
`gated` row. -/
noncomputable def SvL (m₀ : Mem) (d : Inst) (t h p n : Nat) : ℝ :=
  if gated d t then (valF m₀ d.states (stIdx d (chk d t) h p n) : ℝ) else Sv m₀ d t h p n

/-- The decay exponent of the entering state at token `t`: `dA_cumsum` at `t`'s position, less
its value just before the logical chunk's start when that start is inside the chunk. -/
noncomputable def dSt (m₀ : Mem) (d : Inst) (t h : Nat) : ℝ :=
  dAv m₀ d h (chk d t) (t % d.cs) -
    (if 0 < lOff d t then dAv m₀ d h (chk d t) (lOff d t - 1) else 0)

/-- The decay exponent from token `s` to token `t` (same chunk). -/
noncomputable def dS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  dAv m₀ d h (chk d t) (t % d.cs) - dAv m₀ d h (chk d t) (s - chk d t * d.cs)

/-- `cb[0, c, g, t mod chunk, s mod chunk]` for tokens `t`, `s` of chunk `c`. -/
noncomputable def cbTS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  cbv m₀ d (chk d t) (grp d h) (t % d.cs) (s - chk d t * d.cs)

/-- `dt` of token `s` (in `t`'s chunk). -/
noncomputable def dtS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  dtv m₀ d h (chk d t) (s - chk d t * d.cs)

/-- The tokens of `t`'s logical chunk up to `t`: `lTok t ≤ s ≤ t`. -/
def toks (d : Inst) (t : Nat) : Finset Nat := Finset.Ico (lTok d t) (t + 1)

/-- The tokens `s ≤ t` whose weight the kernel computes for row `t` (lines 397–413): those of the
logical chunk with initial states, those of the whole physical chunk without. -/
def kToks (d : Inst) (t : Nat) : Finset Nat :=
  if d.hasInit then toks d t else Finset.Ico (chk d t * d.cs) (t + 1)

/-- `Σ_n C[0, t, g, n]·S[p, n]`, the entering state read out by `C`. -/
noncomputable def sig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N, Cv m₀ d t (grp d h) n * Sv m₀ d t h p n

/-- The weight of token `s` in `t`'s output: `cb·exp(dA_t − dA_s)·dt_s`. -/
noncomputable def wTS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  cbTS m₀ d t h s * Real.exp (dS m₀ d t h s) * dtS m₀ d t h s

/-- **The exact output** at token `t`, head `h`, channel `p`:
`exp(dSt)·Σ_n C·S + Σ_{s ∈ toks t} cb·exp(dA_t − dA_s)·dt_s·x_s + D·x_t`. -/
noncomputable def yRef (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
    (∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p) + Dv m₀ d h * xv m₀ d t h p

/-! ### The tolerance

From the library's facts (README, Tolerances): `FloatV.sub_err`, `mul_err`, `add_err`, `fma_err`,
`cvt_err` (`float32`: `u = 2^-24`, `η = 2^-150`; `bfloat16`: `u = 2^-8`, `η = 2^-134`),
`Num.ExpF32` at `X = |d| + δ` (relative `2^-22 + 2^-23·X`, absolute `2^-140`), and `Num.DotBF16`
(`relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`, relative to lifted magnitudes). -/

/-- `bfloat16`'s unit roundoff `2^-8`. -/
noncomputable def uB : ℝ := 2 ^ (-8 : ℤ)
/-- `float32`'s unit roundoff `2^-24`. -/
noncomputable def uF : ℝ := 2 ^ (-24 : ℤ)
/-- The smallest normal `bfloat16`, `2^-126` (`liftMag`'s floor). -/
noncomputable def mB : ℝ := 2 ^ (-126 : ℤ)
/-- One `float32` rounding's absolute term. -/
noncomputable def etaF : ℝ := 2 ^ (-150 : ℤ)
/-- One `bfloat16` rounding's absolute term. -/
noncomputable def etaB : ℝ := 2 ^ (-134 : ℤ)
/-- `tl.mul_add`'s relative constant (two roundings, or one). -/
noncomputable def rho : ℝ := 2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ)
/-- `Num.DotBF16`'s constants, as reals. -/
noncomputable def relD (K : Nat) : ℝ := (Rules.Num.relDot16 K : ℝ)
noncomputable def absD (K : Nat) : ℝ := (Rules.Num.absDot16 K : ℝ)

/-- `Num.ExpF32`'s relative constant at `X`. -/
noncomputable def expRel (X : ℝ) : ℝ := 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X

/-- The error of a computed `tl.exp(x̃)` against `exp(x)` when `|x̃ − x| ≤ δ`:
`(1 + rel)·exp(x + δ) − exp(x) + 2^-140`, `rel` at `X = |x| + δ`. -/
noncomputable def errE (x δ : ℝ) : ℝ :=
  (1 + expRel (|x| + δ)) * Real.exp (x + δ) - Real.exp x + 2 ^ (-140 : ℤ)

/-- A bound on the computed `tl.exp`'s magnitude. -/
noncomputable def eHat (x δ : ℝ) : ℝ := Real.exp x + errE x δ

/-- The error of the computed decay exponent of the entering state (line 344's IEEE `-`; exact
when the boundary is the `+0` of line 269). -/
noncomputable def dErrSt (m₀ : Mem) (d : Inst) (t h : Nat) : ℝ :=
  if 0 < lOff d t then uF * |dSt m₀ d t h| + etaF else 0

/-- The error of the computed decay exponent from `s` to `t` (line 408's IEEE `-`). -/
noncomputable def dErrS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ := uF * |dS m₀ d t h s| + etaF

/-- `Σ_n |C|·|S|`. -/
noncomputable def mSig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N, |Cv m₀ d t (grp d h) n| * |Sv m₀ d t h p n|

/-- `Σ_n |C|`. -/
noncomputable def cSum (m₀ : Mem) (d : Inst) (t h : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N, |Cv m₀ d t (grp d h) n|

/-- A bound on the lifted magnitude of line 361's `tl.dot` (`liftMag C · liftMag bf16(S)`), for
the reference's state. -/
noncomputable def lSig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N,
    (|Cv m₀ d t (grp d h) n| + mB) * ((1 + uB) * |Sv m₀ d t h p n| + etaB + mB)

/-- The same for the state the kernel reads (`SvL`): the room of line 361's `tl.dot`. -/
noncomputable def lSigL (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N,
    (|Cv m₀ d t (grp d h) n| + mB) * ((1 + uB) * |SvL m₀ d t h p n| + etaB + mB)

/-- The error of line 361's `tl.dot` against `Σ C·S`: the `bfloat16` rounding of `S` (line 360)
and the dot's own error. -/
noncomputable def eSig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  uB * mSig m₀ d t h p + etaB * cSum m₀ d t h + relD d.BSD * lSig m₀ d t h p + absD d.BSD

/-- The computed scale's error and magnitude bound (line 344). -/
noncomputable def eSt (m₀ : Mem) (d : Inst) (t h : Nat) : ℝ := errE (dSt m₀ d t h) (dErrSt m₀ d t h)
noncomputable def hSt (m₀ : Mem) (d : Inst) (t h : Nat) : ℝ := eHat (dSt m₀ d t h) (dErrSt m₀ d t h)

/-- The error of `acc₀ = dot·scale` (line 361) against `exp(dSt)·Σ C·S`. -/
noncomputable def errSig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  (1 + uF) * eSig m₀ d t h p * hSt m₀ d t h + uF * mSig m₀ d t h p * hSt m₀ d t h +
    mSig m₀ d t h p * eSt m₀ d t h + etaF

/-- A bound on `|acc₀|`. -/
noncomputable def aSig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  (1 + uF) * (mSig m₀ d t h p + eSig m₀ d t h p) * hSt m₀ d t h + etaF

/-- `|cb|·eHat·|dt|`, the magnitude of token `s`'s computed weight before its roundings. -/
noncomputable def wMag (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  |cbTS m₀ d t h s| * eHat (dS m₀ d t h s) (dErrS m₀ d t h s) * |dtS m₀ d t h s|

/-- The absolute terms of the weight's three roundings. -/
noncomputable def wAbs (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  2 ^ (-133 : ℤ) + 2 ^ (-148 : ℤ) * |dtS m₀ d t h s|

/-- The error of token `s`'s computed `bfloat16` weight (lines 408–414) against `wTS`: the
`tl.exp` error, two `float32` products and one `bfloat16` rounding. -/
noncomputable def ewS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  |cbTS m₀ d t h s| * |dtS m₀ d t h s| * errE (dS m₀ d t h s) (dErrS m₀ d t h s) +
    (uB + 3 * uF) * wMag m₀ d t h s + wAbs m₀ d t h s

/-- A bound on the computed weight's magnitude. -/
noncomputable def wHat (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  (1 + uB + 3 * uF) * wMag m₀ d t h s + wAbs m₀ d t h s

/-- `Σ_s` of the weights' errors times `|x_s|`. -/
noncomputable def errW (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ s ∈ toks d t, ewS m₀ d t h s * |xv m₀ d s h p|

/-- `Σ_s |w̃_s|·|x_s|`. -/
noncomputable def mW (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ s ∈ toks d t, wHat m₀ d t h s * |xv m₀ d s h p|

/-- A bound on the lifted magnitudes of line 420's `tl.dot`s (`liftMag w̃ · liftMag x`). -/
noncomputable def lW (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ s ∈ toks d t, (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB)

/-- The trips of the loop of line 397, at most `cdiv(chunk_size, 32)`. -/
def nIt (d : Inst) : Nat := (d.cs + BK - 1) / BK

/-- The accumulated error of line 420's `tl.dot_add`s over the trips (each within
`(relDot16 32 + 2^-21)·(|acc| + lifted) + absDot16 32 + 2^-140`: the dot into the accumulator, or
the zero-accumulator dot followed by the fusable addition, README step 4). -/
noncomputable def errDot (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  2 * (nIt d : ℝ) * ((relD BK + 2 ^ (-21 : ℤ)) * (aSig m₀ d t h p + lW m₀ d t h p) + absD BK +
    2 ^ (-140 : ℤ))

/-- A bound on `|acc|` after the loop. -/
noncomputable def aAcc (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  aSig m₀ d t h p + mW m₀ d t h p + errDot m₀ d t h p

/-- The error of line 442's `tl.mul_add`. -/
noncomputable def errMA (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  rho * (|xv m₀ d t h p * Dv m₀ d h| + aAcc m₀ d t h p) + 3 * etaF

/-- The error of the `float32` value before the store. -/
noncomputable def errPre (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  errSig m₀ d t h p + errW m₀ d t h p + errDot m₀ d t h p + errMA m₀ d t h p

/-- **The tolerance** of `out[0, t, h, p]`: the error before the store, and the store's `bfloat16`
rounding of the result (`2^-8` of `|yRef|` plus the error). -/
noncomputable def tol (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  (1 + uB) * errPre m₀ d t h p + uB * |yRef m₀ d t h p| + etaB

/-- Element `i` of a `float32` buffer is a finite value of magnitude at most `2^120` (below
`bfloat16`'s overflow threshold, so its rounding to `bfloat16` is finite). -/
def finF120 (m : Mem) (b : Buf .f32) (i : Nat) : Prop :=
  ∃ v : F32, m.read .f32 (b.ptr (i : Int)).addr = some v ∧ v.isFinite = true ∧
    |v.toRat| ≤ 2 ^ 120

/-- The inputs' values covered: every value the reference reads is finite (`cb` on the pairs of
tokens `s ≤ t` of one logical chunk); every chunk state and initial state the kernel reads, and
rounds to `bfloat16` at line 360, at most `2^120` in magnitude; every `dA_cumsum` entry at most `0`
(real use: the cumulative sums of `A·dt` with `A < 0`, `dt ≥ 0`); and, a limit of the kernel's
`float32` arithmetic, every intermediate the kernel computes for an output in range: the decay
exponents at most `64`, and the products `|cb|·exp` (line 408) and `|cb|·exp·|dt|` (line 410) on
every pair the loop computes (`kToks`), the read-out of the state the kernel reads (`lSigL`), the
state term, the accumulator and the `D` term below `2^120` (README, Tolerances and Exclusions). -/
def Vals (m₀ : Mem) (d : Inst) : Prop :=
  (∀ t g n, t < d.T → g < d.G → n < d.N → finB m₀ d.C (cIdx d t g n)) ∧
  (∀ t h p, t < d.T → h < d.H → p < d.P → finB m₀ d.x (xIdx d t h p)) ∧
  (∀ h, h < d.H → finB m₀ d.D h) ∧
  (∀ c h p n, c < nc d → h < d.H → p < d.P → n < d.N → finF120 m₀ d.states (stIdx d c h p n)) ∧
  (d.hasInit = true → ∀ r h p n, r < d.nreq → h < d.H → p < d.P → n < d.N →
    finF120 m₀ d.initst (inIdx d r h p n)) ∧
  (∀ h c i, h < d.H → c < nc d → i < d.cs → finF m₀ d.dt (dtIdx d h c i)) ∧
  (∀ h c i, h < d.H → c < nc d → i < d.cs →
    ∃ v : F32, m₀.read .f32 (d.dA.ptr (dtIdx d h c i : Int)).addr = some v ∧
      v.isFinite = true ∧ v.toRat ≤ 0) ∧
  (∀ t g s, t < d.T → g < d.G → s ∈ toks d t →
    finF m₀ d.cb (cbIdx d (chk d t) g (t % d.cs) (s - chk d t * d.cs))) ∧
  (∀ t h p, t < d.T → h < d.H → p < d.P →
    dSt m₀ d t h ≤ 64 ∧
    (∀ s ∈ kToks d t, dS m₀ d t h s ≤ 64 ∧
      |cbTS m₀ d t h s| * eHat (dS m₀ d t h s) (dErrS m₀ d t h s) ≤ 2 ^ 120 ∧
      wMag m₀ d t h s ≤ 2 ^ 120) ∧
    lSigL m₀ d t h p ≤ 2 ^ 120 ∧
    hSt m₀ d t h * (mSig m₀ d t h p + eSig m₀ d t h p) ≤ 2 ^ 120 ∧
    aSig m₀ d t h p + lW m₀ d t h p ≤ 2 ^ 120 ∧
    aAcc m₀ d t h p + |xv m₀ d t h p * Dv m₀ d h| ≤ 2 ^ 120)

/-- What the launch leaves:
* every `out[0, t, h, p]` (`t < seqlen`, `h < nheads`, `p < headdim`) holds a `bfloat16` value;
  when the inputs' values are covered (`Vals`), it is finite and within `tol` of `yRef`;
* every byte outside `out`'s `seqlen·nheads·headdim` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t h p, t < d.T → h < d.H → p < d.P → ∃ v : BF16,
      rdB fin d.out (outIdx d t h p) = some v ∧
      (Vals init d → v.isFinite = true ∧ |v.toReal - yRef init d t h p| ≤ tol init d t h p)) ∧
  ∀ z, z ∉ d.out.elems {j | j < d.T * (d.H * d.P)} → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16`: `tl.dot` of `bfloat16` operands into `float32` (lines 361, 420). -/
  dot : Rules.Num.DotBF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` has a result for all operands. -/
  dot_total : Rules.Num.DotTotal P cfgBF16
  /-- [rule] `Num.ExpF32`: `tl.exp` on `float32` (lines 344, 408). -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input. -/
  exp_total : Rules.Num.MathTotal P .exp .binary32

/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## The checks -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotBF16,
    Platform.ref.dotTotal _, Platform.sm90ref.expF32, Platform.ref.mathTotal⟩⟩

/-- `query_start_loc` from the request lengths: `qsl r = Σ_{i<r} lens[i]`. -/
def qslOf (lens : List Nat) (r : Nat) : Nat := (lens.take r).sum

/-- The base address of buffer `k`: `k·2^34`. -/
def baseOf (k : Nat) : Int := (k : Int) * 17179869184

/-- An instance of the host's launch from `HAS_INITSTATES`, the request lengths, `nheads`,
`headdim`, `dstate`, `ngroups`, `chunk_size` and `BLOCK_SIZE_DSTATE`: the shapes the host derives;
`x` and `C` with the token stride `conv_dim = nheads·headdim + 2·ngroups·dstate` of the source's
`make_inputs`; the grid's `y` (`hostN`, or `nchunks` without initial states); the twelve tensors at
the 16-byte-aligned addresses `k·2^34` (`k = 1 … 12`), each exactly as long as its tensor (the
initial states and the metadata empty without initial states). -/
def mk (hasInit : Bool) (lens : List Nat) (H P N G C BSD : Nat) : Inst :=
  let L := lens.sum
  let NC := (L + C - 1) / C
  let conv := H * P + 2 * G * N
  let base : Inst :=
    { hasInit := hasInit, H := H, P := P, N := N, G := G, cs := C, T := L, nreq := lens.length,
      qsl := qslOf lens,
      nmeta := 0, BSD := BSD, xT := conv, cT := conv,
      cb := ⟨baseOf 1, NC * (G * (C * C))⟩, x := ⟨baseOf 2, (L - 1) * conv + H * P⟩,
      dt := ⟨baseOf 3, H * (NC * C)⟩, dA := ⟨baseOf 4, H * (NC * C)⟩, seqIdx := ⟨baseOf 5, L⟩,
      C := ⟨baseOf 6, (L - 1) * conv + G * N⟩, states := ⟨baseOf 7, NC * (H * (P * N))⟩,
      D := ⟨baseOf 8, H⟩,
      initst := ⟨baseOf 9, if hasInit then lens.length * (H * (P * N)) else 0⟩,
      chunkIdx := ⟨baseOf 10, 0⟩, chunkOffs := ⟨baseOf 11, 0⟩,
      out := ⟨baseOf 12, L * (H * P)⟩ }
  let n := if hasInit then hostN base else NC
  let m := if hasInit then n else 0
  { base with nmeta := n, chunkIdx := ⟨baseOf 10, m⟩, chunkOffs := ⟨baseOf 11, m⟩ }

/-- Two buffers at disjoint byte ranges share no byte. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- Two buffers at the addresses `i·2^34` and `j·2^34`, `i ≠ j`, each at most `2^34` bytes long,
share no byte. -/
theorem disj_far {d1 d2 : DType} (b : Buf d1) (c : Buf d2) (i j : Nat)
    (hb : b.base = baseOf i) (hc : c.base = baseOf j)
    (hbl : b.len * d1.bytes ≤ 17179869184) (hcl : c.len * d2.bytes ≤ 17179869184) (hij : i ≠ j) :
    Disjoint b.bytes c.bytes := by
  apply bytes_disj
  simp only [baseOf] at hb hc
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · left
    have : (i : Int) * 17179869184 + 17179869184 ≤ (j : Int) * 17179869184 := by
      have : (i : Int) + 1 ≤ j := by exact_mod_cast h
      nlinarith
    rw [hb, hc]; omega
  · right
    have : (j : Int) * 17179869184 + 17179869184 ≤ (i : Int) * 17179869184 := by
      have : (j : Int) + 1 ≤ i := by exact_mod_cast h
      nlinarith
    rw [hb, hc]; omega

/-- The instances `mk` builds are covered when their sizes are (each condition decided). -/
theorem mk_ok (hi : Bool) (lens : List Nat) (H P N G C BSD a b : Nat) (hB : BSD = 2 ^ a)
    (hC : C = 2 ^ b)
    (h1 : 0 < H ∧ H < 2 ^ 16 ∧ 0 < P ∧ 0 < N ∧ 0 < G ∧ G ∣ H ∧ 16 ≤ BSD ∧ BSD ≤ 128 ∧ N ≤ BSD ∧
      (BSD = 16 ∨ BSD < 2 * N) ∧ C ≤ 2 ^ 14)
    (h2 : 0 < lens.length ∧ ∀ r, r < lens.length → qslOf lens r < qslOf lens (r + 1))
    (h3 : (mk hi lens H P N G C BSD).nmeta < 2 ^ 16 ∧
      gx (mk hi lens H P N G C BSD) * (mk hi lens H P N G C BSD).nmeta * H < 2 ^ 31 ∧
      H * P ≤ (mk hi lens H P N G C BSD).xT ∧ G * N ≤ (mk hi lens H P N G C BSD).cT ∧
      lens.sum * (mk hi lens H P N G C BSD).xT < 2 ^ 31 ∧
      lens.sum * (mk hi lens H P N G C BSD).cT < 2 ^ 31 ∧
      lens.sum * (H * P) < 2 ^ 31 ∧
      (2 * C + 64) * (mk hi lens H P N G C BSD).xT + 64 < 2 ^ 31 ∧
      (2 * C + 64) * (mk hi lens H P N G C BSD).cT + 2 * BSD < 2 ^ 31 ∧
      (2 * C + 64) * (H * P) + 64 < 2 ^ 31 ∧
      nc (mk hi lens H P N G C BSD) * (G * (C * C)) < 2 ^ 31 ∧
      H * (nc (mk hi lens H P N G C BSD) * C) < 2 ^ 31 ∧
      nc (mk hi lens H P N G C BSD) * (H * (P * N)) < 2 ^ 31 ∧
      (hi = true → (lens.length - 1) * (H * (P * N)) < 2 ^ 31))
    (h4 : (mk hi lens H P N G C BSD).cb.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).x.len * 2 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).dt.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).dA.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).seqIdx.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).C.len * 2 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).states.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).D.len * 2 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).initst.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).chunkIdx.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).chunkOffs.len * 4 ≤ 17179869184 ∧
      (mk hi lens H P N G C BSD).out.len * 2 ≤ 17179869184)
    (h5 : (hi = true → lens.length * (H * (P * N)) ≤ (mk hi lens H P N G C BSD).initst.len) ∧
      (hi = true → (mk hi lens H P N G C BSD).nmeta ≤ (mk hi lens H P N G C BSD).chunkIdx.len) ∧
      (hi = true → (mk hi lens H P N G C BSD).nmeta ≤ (mk hi lens H P N G C BSD).chunkOffs.len)) :
    (mk hi lens H P N G C BSD).ok := by
  obtain ⟨h1a, h1b, h1c, h1d, h1e, h1f, h1g, h1h, h1i, h1j, h1k⟩ := h1
  obtain ⟨h2a, h2b⟩ := h2
  obtain ⟨h3a, h3b, h3c, h3d, h3e, h3f, h3g, h3h, h3i, h3j, h3k, h3l, h3m, h3n⟩ := h3
  obtain ⟨h4a, h4b, h4c, h4d, h4e, h4f, h4g, h4h, h4i, h4j, h4k, h4l⟩ := h4
  obtain ⟨h5a, h5b, h5c⟩ := h5
  refine
    { H_pos := h1a, H_lt := h1b, P_pos := h1c, N_pos := h1d, G_pos := h1e, G_dvd := h1f,
      bsd_pow2 := ⟨a, hB⟩, bsd_ge := h1g, bsd_le := h1h, N_le := h1i, bsd_min := h1j,
      cs_pow2 := ⟨b, hC⟩, cs_le := h1k, nreq_pos := h2a, qsl_zero := by simp [mk, qslOf],
      qsl_lt := h2b, qsl_end := by simp [mk, qslOf], nmeta_eq := ?_, nmeta_lt := h3a,
      grid_lt := h3b, xT_ge := h3c, cT_ge := h3d, x_arg := h3e, c_arg := h3f, out_arg := h3g,
      x_off := h3h, c_off := h3i, out_off := h3j,
      cb_i32 := h3k, dt_i32 := h3l, st_i32 := h3m, init_i32 := h3n,
      cb_len := le_refl _, x_len := le_refl _, dt_len := le_refl _, dA_len := le_refl _,
      seq_len := le_refl _, C_len := le_refl _, states_len := le_refl _, D_len := le_refl _,
      init_len := h5a, idx_len := h5b, offs_len := h5c, out_len := le_refl _,
      disj := ?_ }
  · cases hi <;> rfl
  exact ⟨disj_far _ _ 12 1 rfl rfl h4l h4a (by decide), disj_far _ _ 12 2 rfl rfl h4l h4b (by decide),
    disj_far _ _ 12 3 rfl rfl h4l h4c (by decide), disj_far _ _ 12 4 rfl rfl h4l h4d (by decide),
    disj_far _ _ 12 5 rfl rfl h4l h4e (by decide), disj_far _ _ 12 6 rfl rfl h4l h4f (by decide),
    disj_far _ _ 12 7 rfl rfl h4l h4g (by decide), disj_far _ _ 12 8 rfl rfl h4l h4h (by decide),
    disj_far _ _ 12 9 rfl rfl h4l h4i (by decide), disj_far _ _ 12 10 rfl rfl h4l h4j (by decide),
    disj_far _ _ 12 11 rfl rfl h4l h4k (by decide)⟩

/-- The source's test cases (lines 781–793): (request lengths, `nheads`, `headdim`, `dstate`,
`ngroups`, `chunk_size`), with `BLOCK_SIZE_DSTATE = max(next_power_of_2(dstate), 16)`; the test
runs each with initial states (`setup_1` … `setup_11`) and without (`setup_12` … `setup_22`). -/
def setup_1 : Inst := mk true [1] 8 64 128 8 256 128
def setup_2 : Inst := mk true [256] 8 64 128 8 256 128
def setup_3 : Inst := mk true [16, 2, 8, 13] 16 64 64 16 8 64
def setup_4 : Inst := mk true [270, 88, 212, 203] 16 64 64 16 256 64
def setup_5 : Inst := mk true [16, 20] 4 32 32 4 8 32
def setup_6 : Inst := mk true [5, 1, 1, 1] 13 5 5 13 256 16
def setup_7 : Inst := mk true [138, 225, 138, 225] 8 21 21 8 128 32
def setup_8 : Inst := mk true [64, 32, 16, 8, 8] 4 16 16 4 8 16
def setup_9 : Inst := mk true [1000, 1, 2049, 64, 64, 300] 32 80 128 8 128 128
def setup_10 : Inst := mk true [4096, 4096] 64 64 128 8 256 128
def setup_11 : Inst := mk true (List.replicate 40 3 ++ [500]) 16 64 128 1 256 128
def setup_12 : Inst := mk false [1] 8 64 128 8 256 128
def setup_13 : Inst := mk false [256] 8 64 128 8 256 128
def setup_14 : Inst := mk false [16, 2, 8, 13] 16 64 64 16 8 64
def setup_15 : Inst := mk false [270, 88, 212, 203] 16 64 64 16 256 64
def setup_16 : Inst := mk false [16, 20] 4 32 32 4 8 32
def setup_17 : Inst := mk false [5, 1, 1, 1] 13 5 5 13 256 16
def setup_18 : Inst := mk false [138, 225, 138, 225] 8 21 21 8 128 32
def setup_19 : Inst := mk false [64, 32, 16, 8, 8] 4 16 16 4 8 16
def setup_20 : Inst := mk false [1000, 1, 2049, 64, 64, 300] 32 80 128 8 128 128
def setup_21 : Inst := mk false [4096, 4096] 64 64 128 8 256 128
def setup_22 : Inst := mk false (List.replicate 40 3 ++ [500]) 16 64 128 1 256 128

/-- Two model ranks the GPU check runs at 128 heads and 8192 tokens, with initial states and
without: a Nemotron-H rank (`headdim` 64) and a Nemotron Nano 2 rank (`headdim` 80), `dstate`
128, 8 groups, chunks of 128. -/
def setup_23 : Inst := mk true [3000, 2000, 3192] 128 64 128 8 128 128
def setup_24 : Inst := mk false [3000, 2000, 3192] 128 64 128 8 128 128
def setup_25 : Inst := mk true [5000, 3192] 128 80 128 8 128 128
def setup_26 : Inst := mk false [5000, 3192] 128 80 128 8 128 128

theorem setup_1_ok : setup_1.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_2_ok : setup_2.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_3_ok : setup_3.ok :=
  mk_ok _ _ _ _ _ _ _ _ 6 3 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_4_ok : setup_4.ok :=
  mk_ok _ _ _ _ _ _ _ _ 6 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_5_ok : setup_5.ok :=
  mk_ok _ _ _ _ _ _ _ _ 5 3 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_6_ok : setup_6.ok :=
  mk_ok _ _ _ _ _ _ _ _ 4 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_7_ok : setup_7.ok :=
  mk_ok _ _ _ _ _ _ _ _ 5 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_8_ok : setup_8.ok :=
  mk_ok _ _ _ _ _ _ _ _ 4 3 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_9_ok : setup_9.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_10_ok : setup_10.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_11_ok : setup_11.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_12_ok : setup_12.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_13_ok : setup_13.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_14_ok : setup_14.ok :=
  mk_ok _ _ _ _ _ _ _ _ 6 3 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_15_ok : setup_15.ok :=
  mk_ok _ _ _ _ _ _ _ _ 6 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_16_ok : setup_16.ok :=
  mk_ok _ _ _ _ _ _ _ _ 5 3 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_17_ok : setup_17.ok :=
  mk_ok _ _ _ _ _ _ _ _ 4 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_18_ok : setup_18.ok :=
  mk_ok _ _ _ _ _ _ _ _ 5 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_19_ok : setup_19.ok :=
  mk_ok _ _ _ _ _ _ _ _ 4 3 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_20_ok : setup_20.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_21_ok : setup_21.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_22_ok : setup_22.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 8 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_23_ok : setup_23.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_24_ok : setup_24.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_25_ok : setup_25.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)
theorem setup_26_ok : setup_26.ok :=
  mk_ok _ _ _ _ _ _ _ _ 7 7 rfl rfl (by decide) (by decide) (by decide) (by decide) (by decide)

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

/-- Every aligned `int32` read of `zeroMem` gives `0`. -/
theorem zero_i32 {a : Int} (ha : Mem.Aligned .i32 a) : zeroMem.read .i32 a = some (IntV.ofNat 0) := by
  have hb : zeroMem.readBytes a 4 = some [0, 0, 0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.i32.bytes = 4 from rfl]
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

theorem valid_zero {dt : DType} (b : Buf dt) (hb : 0 ≤ b.base)
    (h2 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (ha : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨hb, h2, ha, fun _ _ => by simp [zeroMem]⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_1, zeroMem, setup_1_ok, ?_⟩
  refine ⟨valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), ⟨?_, ?_, ?_⟩, ?_⟩
  · intro t ht
    have ht0 : t = 0 := by
      have : setup_1.T = 1 := by decide
      omega
    subst ht0
    rw [show reqOf setup_1 0 = 0 by decide]
    exact zero_i32 (by decide)
  · intro _ k hk
    have hk0 : k = 0 := by
      have : setup_1.nmeta = 1 := by decide
      omega
    subst hk0
    rw [show hostIdx setup_1 0 = 0 by decide]
    exact zero_i32 (by decide)
  · intro _ k hk
    have hk0 : k = 0 := by
      have : setup_1.nmeta = 1 := by decide
      omega
    subst hk0
    rw [show hostOff setup_1 0 = 0 by decide]
    exact zero_i32 (by decide)
  · intro h
    exact absurd h (by decide)

end Claims.SsdChunkScan
