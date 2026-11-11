import TVA

/-!
# `ssd_chunk_state_bf16`: Mamba-2 SSD chunk states of a packed prefill batch, SGLang

The kernel is `_chunk_state_fwd_kernel` of `source/ssd_chunk_state.py` (lines 77–224; it reads
the module-level names `triton` and `tl`, lines 73–74). The host `chunk_state_fwd` (lines 227–303)
launches it with `HAS_SEQ_IDX = True`, `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`,
`BLOCK_SIZE_K = 64`, `num_stages = 2`, `num_warps = 4`, on the grid
`(cdiv(headdim, 64)·cdiv(dstate, 128), batch·nchunks, nheads)`; `seq_idx` is the request of each
token, from the request boundaries (`make_inputs`, lines 306–332). Each program computes one
`64 × 128` tile of one chunk's state for one head, with a loop of `tl.dot` steps of `64` tokens.

What is claimed: every covered launch ends normally; every element `states[0, c, h, p, n]` holds a
`float32` value, and when the input values are covered (`Vals`), it is finite and within `tol` of
the exact chunk state `sRef`, computed from the host's inputs (the request boundaries
`query_start_loc`, `x`, `B`, `dt`, `dA_cumsum`): the sum over the chunk's tokens of the last
request in the chunk of `x·B·exp(dA_last − dA_t)·dt`; every other byte is unchanged.

The conditions (`Inst.ok`): positive sizes, `ngroups` dividing `nheads`; `chunk_size` a power of
two, at most `2^16`; the request boundaries increase from `0` to `seqlen`, at most `8192` requests
and `seqlen ≤ 2^22`; the grid within CUDA's limits; the strides `int32` (the batch strides
`int32` or `int64`) and the kernel's `int32` offsets not wrapping; the buffers hold every element
the indices name, and `states` shares no byte with an input. `Pre`: the buffers are valid and
`seq_idx` holds the host's metadata (`reqOf`); any input values are covered.

The tolerance rests on the device facts for `tl.exp` on `float32` and `tl.dot` of `bfloat16`
operands into `float32` (`Device`).
-/

namespace Claims.SsdChunkState

open TVA

/-! ## The model -/

/-- `_chunk_state_fwd_kernel` (source lines 77–224). `x` and `B` are `*bf16` pointers, `states`,
`dt` and `dA_cumsum` `*fp32`, `seq_idx` `*i32`; `hdim`, `dstate`, `chunk_size`, `batch`, `seqlen`,
`nheads_ngroups_ratio` and eighteen of the strides are `int32` scalar arguments (`Inst.ok` keeps
each below `2^31`). The five batch strides `stride_x_batch`, `stride_b_batch`,
`stride_states_batch`, `stride_dt_batch`, `stride_dA_cs_batch` are `int32` arguments below `2^31`
and `int64` ones from `2^31` to `2^63` (`Inst.ok.batch_lt`); the kernel uses each only as an
operand of `*` with the `int64` `pid_b`, so an `int32` one is widened (sign extension, the same
value) and an `int64` one used as it is: in either class the operand is the `int64` value of the
argument, the model's `I64` parameter. An integer argument equal to `1` (here `batch`, and on many launches
`stride_x_hdim`, `stride_b_dstate`, `stride_states_dstate`, `stride_dt_csize`,
`stride_dA_cs_csize`, `stride_seq_idx_seqlen`, `nheads_ngroups_ratio`) is the compile-time
constant `1`; each of its uses in the kernel (an operand of `+`, `-`, `*`, `//`, a comparison, a
`min`) builds the constant in the operation's computation dtype with the same value, or is
Python's arithmetic on compile-time values with the same value, so the model's `IntV` is the same
(see `model`). `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K : tl.constexpr` are Lean parameters.

`HAS_SEQ_IDX` is `True` on every covered launch (`SETUP.md`), so each compile-time `if` on it visits
only its taken branch; the model writes that branch and not the others. -/
def ssd_chunk_state_kernel
    (x_ptr b_ptr : Ptr .bf16) (states_ptr dt_ptr dA_cumsum_ptr : Ptr .f32) (seq_idx_ptr : Ptr .i32)
    (hdim dstate chunk_size batch seqlen nheads_ngroups_ratio : I32)
    (stride_x_batch : I64) (stride_x_seqlen stride_x_head stride_x_hdim : I32)
    (stride_b_batch : I64) (stride_b_seqlen stride_b_head stride_b_dstate : I32)
    (stride_states_batch : I64)
    (stride_states_chunk stride_states_head stride_states_hdim stride_states_dstate : I32)
    (stride_dt_batch : I64) (stride_dt_chunk stride_dt_head stride_dt_csize : I32)
    (stride_dA_cs_batch : I64) (stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : I32)
    (stride_seq_idx_batch stride_seq_idx_seqlen : I32)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Kernel where
  body ctx := do
    -- line 123: axis 1 of the grid, `.to(tl.int64)`: the exact widening
    let pid_bc : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 1))).item
    -- line 124: `pid_bc // batch`: the `int32` `batch` widened to `int64` (the other operand is),
    -- then the truncating `int64` `//` (`tl.idiv`)
    let batch64 : I64 := (← tl.convert .i32 .i64 (Tensor.scalar batch)).item
    let pid_c : I64 := (← tl.idiv (Tensor.scalar pid_bc) (Tensor.scalar batch64)).item
    -- line 125: wrapping `int64` product and difference
    let pid_b : I64 := pid_bc - pid_c * batch64
    -- line 126: axis 2 of the grid, an `int32`
    let pid_h := tl.program_id ctx 2
    -- line 127: `tl.cdiv(dstate, BLOCK_SIZE_N)` is `(dstate + (BLOCK_SIZE_N - 1)) // BLOCK_SIZE_N`:
    -- the constexpr `BLOCK_SIZE_N - 1` computed by Python, built in `int32`; the wrapping `+`, then
    -- the truncating `int32` `//`
    let num_pid_n : I32 := (← tl.idiv (Tensor.scalar (dstate + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- lines 128–129: axis 0 of the grid, `int32` `//` and `%` (`tl.idiv`, `tl.irem`)
    let pid_m : I32 := (← tl.idiv (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
    let pid_n : I32 := (← tl.irem (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
    -- lines 130–134: `pid_b * stride_b_batch` and `pid_c * chunk_size * stride_b_seqlen` are
    -- `int64` (each `int32` operand widened, the batch stride an `int64` value in either argument
    -- class, wrapping products); `(pid_h // nheads_ngroups_ratio) *
    -- stride_b_head` is `int32` (truncating `//`, wrapping `*`), widened for the `int64` sum; the
    -- pointer `+=`
    let head_b : I32 := (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar nheads_ngroups_ratio)).item
    let b_ptr := b_ptr +
      (pid_b * stride_b_batch +
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar chunk_size)).item *
          (← tl.convert .i32 .i64 (Tensor.scalar stride_b_seqlen)).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (head_b * stride_b_head))).item)
    -- lines 135–139: the same shape; `pid_h * stride_x_head` is an `int32` product
    let x_ptr := x_ptr +
      (pid_b * stride_x_batch +
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar chunk_size)).item *
          (← tl.convert .i32 .i64 (Tensor.scalar stride_x_seqlen)).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (pid_h * stride_x_head))).item)
    -- line 140
    let dt_ptr := dt_ptr +
      (pid_b * stride_dt_batch +
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar stride_dt_chunk)).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (pid_h * stride_dt_head))).item)
    -- lines 141–145
    let dA_cumsum_ptr := dA_cumsum_ptr +
      (pid_b * stride_dA_cs_batch +
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar stride_dA_cs_chunk)).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (pid_h * stride_dA_cs_head))).item)
    -- lines 146–149: `if HAS_SEQ_IDX:` taken; `int64` products and sum
    let seq_idx_ptr := seq_idx_ptr +
      (pid_b * (← tl.convert .i32 .i64 (Tensor.scalar stride_seq_idx_batch)).item +
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar chunk_size)).item *
          (← tl.convert .i32 .i64 (Tensor.scalar stride_seq_idx_seqlen)).item)
    -- lines 151–153: `int32` products and sums (wrapping); the constexprs built in `int32`
    let offs_m : Tensor [BLOCK_SIZE_M] I32 := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M
    let offs_n : Tensor [BLOCK_SIZE_N] I32 := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    let offs_k : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K
    -- lines 154–156: `int32` products and sum, then the pointer `+` (a block of pointers)
    let x_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] (Ptr .bf16) :=
      x_ptr + (offs_m.col BLOCK_SIZE_K * stride_x_hdim + offs_k.row BLOCK_SIZE_M * stride_x_seqlen)
    -- lines 157–159
    let b_ptrs : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] (Ptr .bf16) :=
      b_ptr + (offs_n.row BLOCK_SIZE_K * stride_b_dstate + offs_k.col BLOCK_SIZE_N * stride_b_seqlen)
    -- line 160
    let dt_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .f32) := dt_ptr + offs_k * stride_dt_csize
    -- lines 161–163: a scalar load at the `int32` offset `(chunk_size - 1) * stride_dA_cs_csize`;
    -- `.to(tl.float32)` is the identity on `float32`
    let dA_cs_last : F32 ← tl.load1 (dA_cumsum_ptr + (chunk_size - IntV.ofInt 1) * stride_dA_cs_csize : Ptr .f32)
    -- line 164
    let dA_cumsum_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .f32) := dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    -- lines 165–166: `if HAS_SEQ_IDX:` taken
    let seq_idx_ptrs : Tensor [BLOCK_SIZE_K] (Ptr .i32) := seq_idx_ptr + offs_k * stride_seq_idx_seqlen
    -- line 168: Python's `min` with a tensor operand is `tl.minimum`; `pid_c * chunk_size` is
    -- `int64` (the `int32` operand widened), so is `seqlen - …`, and `chunk_size` is widened to
    -- `int64`: the exact `tl.iminimum` on `int64`
    let chunk_size_limit : I64 := (tl.iminimum
      (Tensor.scalar (← tl.convert .i32 .i64 (Tensor.scalar chunk_size)).item)
      (Tensor.scalar ((← tl.convert .i32 .i64 (Tensor.scalar seqlen)).item -
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar chunk_size)).item))).item
    -- lines 169–172: `if HAS_SEQ_IDX:` taken; a scalar load at an `int64` offset
    let seq_idx_last : I32 ← tl.load1 (seq_idx_ptr +
      (chunk_size_limit - IntV.ofInt 1) * (← tl.convert .i32 .i64 (Tensor.scalar stride_seq_idx_seqlen)).item :
        Ptr .i32)
    -- line 174: `float32` `+0.0` in every element
    let acc := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
    -- line 175: `for k in range(0, chunk_size_limit, BLOCK_SIZE_K)`: the bounds' promoted type is
    -- `int64` (`chunk_size_limit` is), so `0` and the constexpr step are `int64` and so is `k`;
    -- the loop carries `x_ptrs`, `b_ptrs`, `dt_ptrs`, `dA_cumsum_ptrs`, `seq_idx_ptrs`, `acc`
    -- (assigned in the body, defined before it)
    let (_x_ptrs, _b_ptrs, _dt_ptrs, _dA_cumsum_ptrs, _seq_idx_ptrs, acc) ←
      tl.forRange (IntV.ofInt 0 : I64) chunk_size_limit (IntV.ofNat BLOCK_SIZE_K : I64)
        (x_ptrs, b_ptrs, dt_ptrs, dA_cumsum_ptrs, seq_idx_ptrs, acc)
        fun k (x_ptrs, b_ptrs, dt_ptrs, dA_cumsum_ptrs, seq_idx_ptrs, acc) => do
      -- `offs_k < chunk_size_limit - k` (lines 178, 183, 187, 191, 193): the `int32` `offs_k`
      -- widened to `int64` (the other operand is), the wrapping `int64` difference, the signed
      -- comparison; broadcast where the source indexes `offs_k[None, :]` or `offs_k[:, None]`
      let offs_k64 : Tensor [BLOCK_SIZE_K] I64 ← tl.convert .i32 .i64 offs_k
      let k_mask : Tensor [BLOCK_SIZE_K] Bool := offs_k64.ltS (chunk_size_limit - k)
      -- lines 176–180: the mask `(offs_m[:, None] < hdim) & (offs_k[None, :] < …)`; `other=0.0`:
      -- the `float32` `+0.0` converted to the pointer's dtype `bfloat16`
      let x : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] BF16 ← tl.load x_ptrs
        (some (((offs_m.col BLOCK_SIZE_K).ltS hdim).and (k_mask.row BLOCK_SIZE_M)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_K] (FloatV.litF32 (FloatV.ofPat 0)))))
      -- lines 181–185: the mask `(offs_k[:, None] < …) & (offs_n[None, :] < dstate)`; `other`
      -- as above; `.to(tl.float32)`: the exact widening (`tl.cvt`)
      let b0 : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] BF16 ← tl.load b_ptrs
        (some ((k_mask.col BLOCK_SIZE_N).and ((offs_n.row BLOCK_SIZE_K).ltS dstate)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] (FloatV.litF32 (FloatV.ofPat 0)))))
      let b : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] F32 ← tl.cvt b0
      -- lines 186–188: `other=0.0` is the `float32` `+0.0`; `.to(tl.float32)` the identity
      let dA_cs_k : Tensor [BLOCK_SIZE_K] F32 ← tl.load dA_cumsum_ptrs (some k_mask)
        (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
      -- lines 189–192: `if HAS_SEQ_IDX:` taken; `other=-1` is the `int32` `-1` (the pointer's
      -- dtype)
      let seq_idx_k : Tensor [BLOCK_SIZE_K] I32 ← tl.load seq_idx_ptrs (some k_mask)
        (some (bc (IntV.ofInt (-1) : I32)))
      -- lines 193–195: as lines 186–188
      let dt_k : Tensor [BLOCK_SIZE_K] F32 ← tl.load dt_ptrs (some k_mask)
        (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
      -- lines 196–197: `if not HAS_SEQ_IDX:` not taken; lines 198–201, the `else` branch:
      -- `dA_cs_last - dA_cs_k` is the IEEE `-` (two loaded values: no product feeds it), the
      -- scalar broadcast; `tl.exp`; the IEEE `*` by `dt_k` (a product feeding no addition); the
      -- `int32` `==` with the scalar `seq_idx_last` broadcast; `tl.where` with `0.0` built in
      -- `float32`. Both operands of `tl.where` are computed
      let dA_diff ← tl.sub (bc dA_cs_last) dA_cs_k
      let decay ← tl.exp dA_diff
      let w ← tl.mul decay dt_k
      let scale : Tensor [BLOCK_SIZE_K] F32 :=
        tl.where (seq_idx_k.eq (bc seq_idx_last)) w (bc (FloatV.litF32 (FloatV.ofPat 0)))
      -- line 202: `b *= scale[:, None]` is `b = b * scale[:, None]`, the IEEE `float32` product
      -- (it feeds no addition: its only use is the conversion below)
      let b ← tl.mul b (scale.col BLOCK_SIZE_N)
      -- line 203: `b.to(x_ptr.dtype.element_ty)`: `float32` to `bfloat16`, one rounding to nearest
      -- even (`tl.cvt`)
      let b : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] BF16 ← tl.cvt b
      -- line 204: `acc += tl.dot(x, b)`: a source `c + tl.dot(a, b)` whose dot has the zero
      -- accumulator (the default), so the combine pass may rewrite it to `tl.dot(a, b, c)`
      -- (`tl.dot_add`); no `input_precision` (the default `tf32`) and no
      -- `max_num_imprecise_acc` (`0`); `bfloat16` operands, a `float32` result
      let acc ← tl.dot_add
        { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } x b acc
      -- lines 205–210: `x += y` is `x = x + y`: the constexpr `BLOCK_SIZE_K` built in `int32` as an
      -- operand of `*` with the `int32` stride, the wrapping product added to every pointer; line
      -- 209: `if HAS_SEQ_IDX:` taken
      let x_ptrs := x_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_x_seqlen) : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] I32)
      let b_ptrs := b_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_b_seqlen) : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] I32)
      let dt_ptrs := dt_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_dt_csize) : Tensor [BLOCK_SIZE_K] I32)
      let dA_cumsum_ptrs := dA_cumsum_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_dA_cs_csize) : Tensor [BLOCK_SIZE_K] I32)
      let seq_idx_ptrs := seq_idx_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_seq_idx_seqlen) : Tensor [BLOCK_SIZE_K] I32)
      pure (x_ptrs, b_ptrs, dt_ptrs, dA_cumsum_ptrs, seq_idx_ptrs, acc)
    -- line 211: `acc.to(states_ptr.dtype.element_ty)`: `float32` to `float32`, the identity
    let states := acc
    -- lines 213–217: `int64` products (the `int32` stride widened, the batch stride `int64`), the
    -- `int32` product
    -- `pid_h * stride_states_head` widened, the `int64` sum, the pointer `+=`
    let states_ptr := states_ptr +
      (pid_b * stride_states_batch +
        pid_c * (← tl.convert .i32 .i64 (Tensor.scalar stride_states_chunk)).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (pid_h * stride_states_head))).item)
    -- lines 218–219: as lines 151–152
    let offs_m : Tensor [BLOCK_SIZE_M] I32 := pid_m * (IntV.ofNat BLOCK_SIZE_M : I32) + tl.arange 0 BLOCK_SIZE_M
    let offs_n : Tensor [BLOCK_SIZE_N] I32 := pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
    -- lines 220–222: `int32` products and sum, the pointer `+`
    let states_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .f32) :=
      states_ptr + (offs_m.col BLOCK_SIZE_N * stride_states_hdim + offs_n.row BLOCK_SIZE_M * stride_states_dstate)
    -- line 223: two signed `int32` comparisons, `&`
    let c_mask := ((offs_m.col BLOCK_SIZE_N).ltS hdim).and ((offs_n.row BLOCK_SIZE_M).ltS dstate)
    -- line 224: `states` is already `float32`, the pointer's dtype
    tl.store states_ptrs states (some c_mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `chunk_state_fwd` with `seq_idx` given (`SETUP.md` fixes
`HAS_SEQ_IDX = True`, `states_in_fp32 = True` and the configuration), `batch = 1`. The request
boundaries `query_start_loc` (`qsl`) are the host's input from which it builds `seq_idx`; the
buffers are given by their storage (`Buf`). `x` and `B` have any strides (the host passes
`x.stride(0…3)` and `B.stride(0…2)`, `B.stride(-1)`); `dt`, `dA_cumsum`, `seq_idx` are
contiguous and `states` is allocated fresh by the host, so their strides are the shapes'
products (`sDb`, …). -/
structure Inst where
  /-- `seqlen` (`x.shape[1]`, the packed tokens). -/
  seqlen : Nat
  /-- `nheads` (`x.shape[2]`): the grid's `z`. -/
  nheads : Nat
  /-- `headdim` (`x.shape[3]`, the kernel's `hdim`). -/
  headdim : Nat
  /-- `dstate` (`B.shape[3]`). -/
  dstate : Nat
  /-- `ngroups` (`B.shape[2]`). -/
  ngroups : Nat
  /-- `chunk_size` (`dt.shape[3]`). -/
  chunk : Nat
  /-- The number of requests packed into the batch. -/
  nreq : Nat
  /-- `query_start_loc`: request `r` holds the tokens `qsl r ≤ t < qsl (r + 1)`. -/
  qsl : Nat → Nat
  /-- `x.stride(0…3)`, in elements. -/
  sXb : Nat
  sXs : Nat
  sXh : Nat
  sXp : Nat
  /-- `B.stride(0…2)` and `B.stride(-1)`, in elements. -/
  sBb : Nat
  sBs : Nat
  sBg : Nat
  sBn : Nat
  /-- The storage of `x` (`bfloat16`): `x[0, t, h, p]` is element `xIdx t h p`. -/
  x : Buf .bf16
  /-- The storage of `B` (`bfloat16`): `B[0, t, g, n]` is element `bIdx t g n`. -/
  b : Buf .bf16
  /-- The storage of `dt` (`float32`, `[1, nheads, nchunks, chunk_size]`, contiguous). -/
  dt : Buf .f32
  /-- The storage of `dA_cumsum` (`float32`, as `dt`). -/
  dA : Buf .f32
  /-- The storage of `seq_idx` (`int32`, `[1, seqlen]`, contiguous). -/
  seqIdx : Buf .i32
  /-- The storage of `states` (`float32`, `[1, nchunks, nheads, headdim, dstate]`, allocated by
  the host). -/
  states : Buf .f32

/-- `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, fixed by the host (line 253). -/
def BM : Nat := 64
def BN : Nat := 128
def BK : Nat := 64

/-- `nchunks = cdiv(seqlen, chunk_size)` (`dt.shape[2]`, as `_chunk_cumsum_fwd` allocates it). -/
def nc (d : Inst) : Nat := (d.seqlen + d.chunk - 1) / d.chunk

/-- The grid's `x`: `cdiv(headdim, 64)·cdiv(dstate, 128)` (lines 254–258). -/
def gx (d : Inst) : Nat := (d.headdim + BM - 1) / BM * ((d.dstate + BN - 1) / BN)

/-- `nheads // ngroups` (line 272). -/
def ratio (d : Inst) : Nat := d.nheads / d.ngroups

/-- The strides of the contiguous tensors, as the host passes them (lines 273–295): `states`
`[1, nchunks, nheads, headdim, dstate]` has `stride(0…4)` `nchunks·nheads·headdim·dstate`,
`nheads·headdim·dstate`, `headdim·dstate`, `dstate`, `1`; for `dt` and `dA_cumsum`
`[1, nheads, nchunks, chunk_size]` the launch passes `stride(0), stride(2), stride(1),
stride(3)`: `nheads·nchunks·chunk_size`, `chunk_size`, `nchunks·chunk_size`, `1`; `seq_idx`
`[1, seqlen]` has `seqlen`, `1`. -/
def sSb (d : Inst) : Nat := nc d * (d.nheads * (d.headdim * d.dstate))

def sSc (d : Inst) : Nat := d.nheads * (d.headdim * d.dstate)

def sSh (d : Inst) : Nat := d.headdim * d.dstate

def sDb (d : Inst) : Nat := d.nheads * (nc d * d.chunk)

def sDh (d : Inst) : Nat := nc d * d.chunk

/-- Element `(0, t, h, p)` of `x`. -/
def xIdx (d : Inst) (t h p : Nat) : Nat := t * d.sXs + h * d.sXh + p * d.sXp

/-- Element `(0, t, g, n)` of `B`. -/
def bIdx (d : Inst) (t g n : Nat) : Nat := t * d.sBs + g * d.sBg + n * d.sBn

/-- Element `(0, h, c, i)` of `dt` and of `dA_cumsum`. -/
def dIdx (d : Inst) (h c i : Nat) : Nat := h * (nc d * d.chunk) + c * d.chunk + i

/-- Element `(0, c, h, p, n)` of `states`. -/
def stIdx (d : Inst) (c h p n : Nat) : Nat :=
  c * (d.nheads * (d.headdim * d.dstate)) + h * (d.headdim * d.dstate) + p * d.dstate + n

/-! ### The host's metadata, from `query_start_loc` -/

/-- The requests after the first, `1, …, nreq − 1`. -/
def laterReqs (d : Inst) : List Nat := List.range' 1 (d.nreq - 1)

/-- The request of token `t`: the number of later requests that start at or before `t`
(`seq_idx[0, t]`, `torch.repeat_interleave(arange(nreq), seqlens)`, line 330). -/
def reqOf (d : Inst) (t : Nat) : Nat := ((laterReqs d).filter (fun r => decide (d.qsl r ≤ t))).length

/-- The instances covered (README, Conditions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- Positive sizes; `ngroups` divides `nheads` (asserted, line 234). -/
  seqlen_pos : 0 < d.seqlen
  heads_pos : 0 < d.nheads
  headdim_pos : 0 < d.headdim
  dstate_pos : 0 < d.dstate
  groups_pos : 0 < d.ngroups
  groups_dvd : d.ngroups ∣ d.nheads
  /-- `chunk_size` is a power of two (the code base asserts it, `ssd_combined.py:77`). -/
  chunk_pow2 : ∃ k, d.chunk = 2 ^ k
  /-- `chunk_size ≤ 2^16`, so the loop runs at most `1024` `tl.dot` steps (README, Exclusions). -/
  chunk_le : d.chunk ≤ 2 ^ 16
  /-- The grid `(gx, nchunks, nheads)` is within CUDA's limits (`y, z < 2^16`) and has fewer than
  `2^31` programs. -/
  grid_y : nc d < 2 ^ 16
  grid_z : d.nheads < 2 ^ 16
  grid_lt : gx d * nc d * d.nheads < 2 ^ 31
  /-- The packed batch: at least one request; `query_start_loc` starts at `0`, every request has
  at least one token, and the last ends at `seqlen`. -/
  nreq_pos : 0 < d.nreq
  qsl_zero : d.qsl 0 = 0
  qsl_lt : ∀ r, r < d.nreq → d.qsl r < d.qsl (r + 1)
  qsl_end : d.qsl d.nreq = d.seqlen
  /-- The lanes `pid_m·64 + arange(64)` and `pid_n·128 + arange(128)`, and `cdiv`'s sums, stay
  below `2^31`; the chunk's token positions are `int32` values (so are `seqlen` and
  `chunk_size`). -/
  headdim_lt : d.headdim + 63 < 2 ^ 31
  dstate_lt : d.dstate + 127 < 2 ^ 31
  tok_lt : nc d * d.chunk < 2 ^ 31
  /-- The strides of `x` and `B` are `int32` arguments, and the `int32` offsets the kernel
  computes from them do not wrap: `pid_h·stride_x_head`, `(pid_h // ratio)·stride_b_head`, every
  lane of the tiles `offs_m·stride_x_hdim + offs_k·stride_x_seqlen` and
  `offs_n·stride_b_dstate + offs_k·stride_b_seqlen` (`offs_m < headdim + 64`,
  `offs_n < dstate + 128`, `offs_k < 64`), and the steps `64·stride` (README, Exclusions). -/
  sX_lt : d.sXs < 2 ^ 31 ∧ d.sXh < 2 ^ 31 ∧ d.sXp < 2 ^ 31
  sB_lt : d.sBs < 2 ^ 31 ∧ d.sBg < 2 ^ 31 ∧ d.sBn < 2 ^ 31
  /-- The batch strides `x.stride(0)`, `B.stride(0)`, `states.stride(0)`, `dt.stride(0)` and
  `dA_cumsum.stride(0)` are signed integer arguments, `int32` or `int64` (below `2^63`). -/
  batch_lt : d.sXb < 2 ^ 63 ∧ d.sBb < 2 ^ 63 ∧ sSb d < 2 ^ 63 ∧ sDb d < 2 ^ 63
  xhead_lt : (d.nheads - 1) * d.sXh < 2 ^ 31
  bhead_lt : (d.ngroups - 1) * d.sBg < 2 ^ 31
  xtile_lt : (d.headdim + 63) * d.sXp + 64 * d.sXs < 2 ^ 31
  btile_lt : (d.dstate + 127) * d.sBn + 64 * d.sBs < 2 ^ 31
  /-- The head offset `pid_h·stride_dt_head = pid_h·nchunks·chunk_size` (an `int32` product, for
  `dt` and `dA_cumsum`) does not wrap (a limit of the kernel, README, Exclusions). -/
  dt_lt : (d.nheads - 1) * (nc d * d.chunk) < 2 ^ 31
  /-- `states.stride(1) = nheads·headdim·dstate` is an `int32` argument (so are `stride(2…3)`, and
  the head offset `pid_h·headdim·dstate` does not wrap); the tile offsets `offs_m·dstate + offs_n`
  stay below `2^31`. -/
  st_lt : sSc d < 2 ^ 31
  sttile_lt : (d.headdim + 63) * d.dstate + d.dstate + 127 < 2 ^ 31
  /-- Every element the indices name lies in its tensor's storage. -/
  x_len : xIdx d (d.seqlen - 1) (d.nheads - 1) (d.headdim - 1) < d.x.len
  b_len : bIdx d (d.seqlen - 1) (d.ngroups - 1) (d.dstate - 1) < d.b.len
  dt_len : sDb d ≤ d.dt.len
  dA_len : sDb d ≤ d.dA.len
  seq_len : d.seqlen ≤ d.seqIdx.len
  states_len : sSb d ≤ d.states.len
  /-- The output's storage shares no byte with the inputs' (the host allocates it, line 244). -/
  disj : Disjoint d.states.bytes d.x.bytes ∧ Disjoint d.states.bytes d.b.bytes ∧
    Disjoint d.states.bytes d.dt.bytes ∧ Disjoint d.states.bytes d.dA.bytes ∧
    Disjoint d.states.bytes d.seqIdx.bytes
  /-- `seqlen` is the number of tokens of the packed prefill batch: a context length, at most
  `2^22` (so is every request's length, `qsl (r + 1) - qsl r ≤ seqlen`). -/
  seqlen_ctx : d.seqlen ≤ 2 ^ 22
  /-- `num_requests` is the number of requests packed into the batch: a batch size, at most
  `8192`. -/
  nreq_le : d.nreq ≤ 8192

/-- The model on an instance: the launch of lines 253–302, with the strides the host passes and
`BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 64`. An integer argument equal to `1`
(`batch` always) is the compile-time constant `1`; each use of it in the kernel builds the constant
in the computation dtype with the same value (or is Python's arithmetic on compile-time values
with the same value), so the model's `IntV.ofNat 1` is the same. The five batch strides are the
`int64` values of their arguments, whichever integer class Triton gives them (see
`ssd_chunk_state_kernel`). -/
def model (d : Inst) : Kernel :=
  ssd_chunk_state_kernel (d.x.ptr 0) (d.b.ptr 0) (d.states.ptr 0) (d.dt.ptr 0) (d.dA.ptr 0)
    (d.seqIdx.ptr 0)
    (IntV.ofNat d.headdim) (IntV.ofNat d.dstate) (IntV.ofNat d.chunk) (IntV.ofNat 1)
    (IntV.ofNat d.seqlen) (IntV.ofNat (ratio d))
    (IntV.ofNat d.sXb : I64) (IntV.ofNat d.sXs) (IntV.ofNat d.sXh) (IntV.ofNat d.sXp)
    (IntV.ofNat d.sBb : I64) (IntV.ofNat d.sBs) (IntV.ofNat d.sBg) (IntV.ofNat d.sBn)
    (IntV.ofNat (sSb d) : I64) (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat d.dstate)
    (IntV.ofNat 1)
    (IntV.ofNat (sDb d) : I64) (IntV.ofNat d.chunk) (IntV.ofNat (sDh d)) (IntV.ofNat 1)
    (IntV.ofNat (sDb d) : I64) (IntV.ofNat d.chunk) (IntV.ofNat (sDh d)) (IntV.ofNat 1)
    (IntV.ofNat d.seqlen) (IntV.ofNat 1)
    BM BN BK

/-- The host's launch (lines 253–302): the grid `(cdiv(headdim, 64)·cdiv(dstate, 128),
nchunks, nheads)` (`batch = 1`); the source's block shapes (the `tl.arange`s, the `x` tile, the
`B` tile and the accumulator); `enable_fp_fusion` at Triton's default (`true`); `num_ctas = 1`, no
`launch_pdl`, `debug = false`. `num_warps = 4` and `num_stages = 2` are performance-only. -/
def launch (d : Inst) : Launch :=
  { grid := ![gx d, nc d, d.nheads], fusion := true,
    blocks := [[BM], [BN], [BK], [BM, BK], [BK, BN], [BM, BN]] }

/-- The value of element `i` of a `float32` buffer in memory `m` (`0` when absent). -/
def valF (m : Mem) (b : Buf .f32) (i : Nat) : ℚ :=
  match m.read .f32 (b.ptr (i : Int)).addr with
  | some v => v.toRat
  | none => 0

/-- The value of element `i` of a `bfloat16` buffer in memory `m` (`0` when absent). -/
def valB (m : Mem) (b : Buf .bf16) (i : Nat) : ℚ :=
  match m.read .bf16 (b.ptr (i : Int)).addr with
  | some v => v.toRat
  | none => 0

/-- The metadata are the host's: `seq_idx[0, t]` is the request of token `t`, from
`query_start_loc`. -/
def MetaOk (d : Inst) (m : Mem) : Prop :=
  ∀ t, t < d.seqlen → m.read .i32 (d.seqIdx.ptr (t : Int)).addr = some (IntV.ofNat (reqOf d t))

/-- The inputs covered: the six buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and `seq_idx` is the host's (`MetaOk`). Any `bfloat16` and `float32` values: `Post` says
what holds when the values are covered (`Vals`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.b.Valid init ∧ d.dt.Valid init ∧ d.dA.Valid init ∧ d.seqIdx.Valid init ∧
    d.states.Valid init ∧ MetaOk d init

/-! ## The specification -/

/-! ### The reference, from the host's inputs -/

/-- The last token of physical chunk `c`: `min((c + 1)·chunk_size, seqlen) − 1`. -/
def endTok (d : Inst) (c : Nat) : Nat := min ((c + 1) * d.chunk) d.seqlen - 1

/-- The request running at the end of chunk `c`. -/
def eReq (d : Inst) (c : Nat) : Nat := reqOf d (endTok d c)

/-- The tokens chunk `c` sums: its own tokens (from `c·chunk_size` to its last) of the request
running at its end. -/
def toks (d : Inst) (c : Nat) : Finset Nat :=
  (Finset.Icc (c * d.chunk) (endTok d c)).filter (fun t => reqOf d t = eReq d c)

/-- The group of head `h`: `h // (nheads // ngroups)`. -/
def grp (d : Inst) (h : Nat) : Nat := h / ratio d

/-- `x[0, t, h, p]`. -/
noncomputable def xV (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ := (valB m₀ d.x (xIdx d t h p) : ℝ)

/-- `B[0, t, grp h, n]`. -/
noncomputable def bV (m₀ : Mem) (d : Inst) (t h n : Nat) : ℝ :=
  (valB m₀ d.b (bIdx d t (grp d h) n) : ℝ)

/-- `dt[0, h, c, t − c·chunk_size]`. -/
noncomputable def dtV (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  (valF m₀ d.dt (dIdx d h c (t - c * d.chunk)) : ℝ)

/-- The decay exponent of token `t` in chunk `c`, head `h`: `dA_cumsum` at the chunk's end
(position `chunk_size − 1`, the chunk's cumulative total) less `dA_cumsum` at the token. -/
noncomputable def dRef (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  (valF m₀ d.dA (dIdx d h c (d.chunk - 1)) : ℝ) - (valF m₀ d.dA (dIdx d h c (t - c * d.chunk)) : ℝ)

/-- The exact weight of token `t`: `exp(dRef)·dt`. -/
noncomputable def wRef (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  Real.exp (dRef m₀ d c h t) * dtV m₀ d c h t

/-- **The exact chunk state** for chunk `c`, head `h`, `p < headdim`, `n < dstate`: the sum over the
chunk's tokens of the request at its end of `x·B·exp(dRef)·dt`. `exp` is the real exponential and
the arithmetic is exact. -/
noncomputable def sRef (m₀ : Mem) (d : Inst) (c h p n : Nat) : ℝ :=
  ∑ t ∈ toks d c, xV m₀ d t h p * bV m₀ d t h n * wRef m₀ d c h t

/-! ### The tolerance

From the device facts, per token `t` of `toks c` (README, Tolerances): the IEEE subtraction of
line 200 (`2^-24·|D| + 2^-150`); `Device.ExpF32` at `X = |D| + δ` (relative `2^-22 + 2^-23·X`,
absolute `2^-140`); the IEEE products by `dt` and by `B` (`2^-24·|p| + 2^-150`); the rounding to
`bfloat16` (`2^-8·|v| + 2^-134`); then `Device.DotBF16` for each `tl.dot` step of `K = 64`,
accumulated over the loop's steps. -/

/-- The error of the computed decay exponent (line 200's rounding). -/
noncomputable def dErr (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  |dRef m₀ d c h t| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)

/-- `Device.ExpF32`'s relative constant at `X = |dRef| + dErr`, which bounds the computed input. -/
noncomputable def expRel (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|dRef m₀ d c h t| + dErr m₀ d c h t)

/-- The error of the computed `tl.exp(dA_cs_last − dA_cs_k)` against `exp(dRef)`:
`(1 + rel)·exp(dRef + δ) − exp(dRef) + 2^-140`. -/
noncomputable def errE (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  (1 + expRel m₀ d c h t) * Real.exp (dRef m₀ d c h t + dErr m₀ d c h t) - Real.exp (dRef m₀ d c h t) +
    2 ^ (-140 : ℤ)

/-- A bound on the computed exponential's magnitude. -/
noncomputable def eHat (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  Real.exp (dRef m₀ d c h t) + errE m₀ d c h t

/-- The error of the computed `scale = exp(…)·dt` (line 200) against `wRef`. -/
noncomputable def errS (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ :=
  |dtV m₀ d c h t| * errE m₀ d c h t + 2 ^ (-24 : ℤ) * (eHat m₀ d c h t * |dtV m₀ d c h t|) +
    2 ^ (-150 : ℤ)

/-- A bound on the computed scale's magnitude. -/
noncomputable def sHat (m₀ : Mem) (d : Inst) (c h t : Nat) : ℝ := |wRef m₀ d c h t| + errS m₀ d c h t

/-- The error of the computed `float32` product `B·scale` (line 202) against `B·wRef`. -/
noncomputable def errP (m₀ : Mem) (d : Inst) (c h t n : Nat) : ℝ :=
  |bV m₀ d t h n| * errS m₀ d c h t + 2 ^ (-24 : ℤ) * (|bV m₀ d t h n| * sHat m₀ d c h t) +
    2 ^ (-150 : ℤ)

/-- A bound on that product's magnitude. -/
noncomputable def pHat (m₀ : Mem) (d : Inst) (c h t n : Nat) : ℝ :=
  |bV m₀ d t h n * wRef m₀ d c h t| + errP m₀ d c h t n

/-- The error of the `bfloat16` operand `b` (line 203) against `B·wRef`: the product's error and
the rounding to `bfloat16`. -/
noncomputable def errB (m₀ : Mem) (d : Inst) (c h t n : Nat) : ℝ :=
  errP m₀ d c h t n + 2 ^ (-8 : ℤ) * pHat m₀ d c h t n + 2 ^ (-134 : ℤ)

/-- A bound on the `bfloat16` operand's magnitude. -/
noncomputable def bHat (m₀ : Mem) (d : Inst) (c h t n : Nat) : ℝ :=
  |bV m₀ d t h n * wRef m₀ d c h t| + errB m₀ d c h t n

/-- The tensor core's lifted magnitude of a `bfloat16` value (`Device.liftMag .bf16`: `0` for
`0`, otherwise `|q|` or the smallest normal `2^-126`, whichever is larger). -/
noncomputable def magB (q : ℚ) : ℝ := (Device.liftMag .bf16 q : ℝ)

/-- The products' lifted magnitudes over the chunk: `Σ liftMag x · max(bHat, 2^-126)` over
`toks c` (the other lanes of the dot hold `±0` operands). -/
noncomputable def dotMag (m₀ : Mem) (d : Inst) (c h p n : Nat) : ℝ :=
  ∑ t ∈ toks d c, magB (valB m₀ d.x (xIdx d t h p)) * max (bHat m₀ d c h t n) (2 ^ (-126 : ℤ))

/-- The number of `tl.dot` steps of the loop (line 175): `cdiv(chunk_size_limit, 64)`, with
`chunk_size_limit = endTok c + 1 − c·chunk_size`. -/
def steps (d : Inst) (c : Nat) : Nat := (endTok d c + 1 - c * d.chunk + BK - 1) / BK

/-- The inputs' magnitudes over the chunk's tokens, the scale of the absolute floor:
`Σ_{t ∈ toks c} (1 + |x|)·(1 + |B|)·(1 + |dt|)` (at least `1`: the chunk's last token is in
`toks c`). -/
noncomputable def inMag (m₀ : Mem) (d : Inst) (c h p n : Nat) : ℝ :=
  ∑ t ∈ toks d c, (1 + |xV m₀ d t h p|) * (1 + |bV m₀ d t h n|) * (1 + |dtV m₀ d c h t|)

/-- **The tolerance of `states[0, c, h, p, n]`**: `2^-8·(1 + steps·2^-6)` of the products' lifted
magnitudes `dotMag` (the `bfloat16` rounding of each scaled `B`, with `2^-6` of it per `tl.dot` step
of the loop for the accumulation and the other roundings), and an absolute floor `2^-100·inMag`
(README, Tolerances). -/
noncomputable def tol (m₀ : Mem) (d : Inst) (c h p n : Nat) : ℝ :=
  2 ^ (-8 : ℤ) * (1 + (steps d c : ℝ) * 2 ^ (-6 : ℤ)) * dotMag m₀ d c h p n +
    2 ^ (-100 : ℤ) * inMag m₀ d c h p n

/-- The inputs' values covered: every `x` and `B` entry of the packed tokens is a finite
`bfloat16`; every `dt` and `dA_cumsum` entry is finite, and every `dA_cumsum` entry is at most `0`
(real use: cumulative sums of `A·dt` with `A < 0`, `dt ≥ 0`); and, a limit of the kernel's `float32`
arithmetic, for every summed token the decay exponent is at most `64`, the scale and the scaled
`B` stay below `2^126`, and the products' lifted magnitudes stay below `2^120` (README,
Exclusions). -/
def Vals (m₀ : Mem) (d : Inst) : Prop :=
  (∀ t h p, t < d.seqlen → h < d.nheads → p < d.headdim →
    ∃ v, m₀.read .bf16 (d.x.ptr (xIdx d t h p : Int)).addr = some v ∧ v.isFinite = true) ∧
  (∀ t g n, t < d.seqlen → g < d.ngroups → n < d.dstate →
    ∃ v, m₀.read .bf16 (d.b.ptr (bIdx d t g n : Int)).addr = some v ∧ v.isFinite = true) ∧
  (∀ h c i, h < d.nheads → c < nc d → i < d.chunk →
    (∃ v, m₀.read .f32 (d.dt.ptr (dIdx d h c i : Int)).addr = some v ∧ v.isFinite = true) ∧
    ∃ v, m₀.read .f32 (d.dA.ptr (dIdx d h c i : Int)).addr = some v ∧ v.isFinite = true ∧
      v.toRat ≤ 0) ∧
  (∀ c h t, c < nc d → h < d.nheads → t ∈ toks d c →
    dRef m₀ d c h t ≤ 64 ∧ eHat m₀ d c h t * |dtV m₀ d c h t| ≤ 2 ^ 126 ∧
    ∀ n, n < d.dstate → |bV m₀ d t h n| * sHat m₀ d c h t ≤ 2 ^ 126) ∧
  (∀ c h p n, c < nc d → h < d.nheads → p < d.headdim → n < d.dstate →
    dotMag m₀ d c h p n ≤ 2 ^ 120)

/-- What the launch leaves:
* every element `states[0, c, h, p, n]` holds a `float32` value; when the values are covered
  (`Vals`), it is finite and within `tol` of the exact chunk state `sRef`;
* every byte outside `states`' `nchunks·nheads·headdim·dstate` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ c h p n, c < nc d → h < d.nheads → p < d.headdim → n < d.dstate → ∃ v,
      fin.read .f32 (d.states.ptr (stIdx d c h p n : Int)).addr = some v ∧
      (Vals init d → v.isFinite = true ∧ |v.toReal - sRef init d c h p n| ≤ tol init d c h p n)) ∧
  ∀ z, z ∉ d.states.elems {j | j < sSb d} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_chunk_state_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous, and the script's launches -/

/-- `query_start_loc` from the request lengths: `qsl r = Σ_{i<r} lens[i]`. -/
def qslOf (lens : List Nat) (r : Nat) : Nat := (lens.take r).sum

/-- The base address of buffer `k` in the covered instances below: `k·2^36`, 16-byte aligned,
the buffers far apart. -/
def slot (k : Nat) : Int := (k : Int) * 68719476736

/-- An instance of the host's launch from the request lengths, `nheads`, `headdim`, `dstate`,
`ngroups`, `chunk_size` and the layout of `x` and `B`: with `rows`, both are views of one
`bfloat16` row of `conv_dim = nheads·headdim + 2·ngroups·dstate` elements per token (SGLang's
layer: `x` its first `nheads·headdim` elements, `B` the next `ngroups·dstate`, so `x.stride` is
`(seqlen·conv_dim, conv_dim, headdim, 1)` and `B.stride` `(seqlen·conv_dim, conv_dim, dstate, 1)`);
otherwise contiguous tensors (SGLang's tests). The other tensors are contiguous; `seq_idx`, `dt`,
`dA_cumsum` and `states` are each exactly as long as the tensor. -/
def mk (lens : List Nat) (H P N G C : Nat) (rows : Bool) : Inst :=
  let L := lens.sum
  let NC := (L + C - 1) / C
  let conv := H * P + 2 * G * N
  { seqlen := L, nheads := H, headdim := P, dstate := N, ngroups := G, chunk := C,
    nreq := lens.length, qsl := qslOf lens,
    sXb := if rows then L * conv else L * (H * P), sXs := if rows then conv else H * P,
    sXh := P, sXp := 1,
    sBb := if rows then L * conv else L * (G * N), sBs := if rows then conv else G * N,
    sBg := N, sBn := 1,
    x := if rows then ⟨slot 1, L * conv⟩ else ⟨slot 1, L * (H * P)⟩,
    b := if rows then ⟨slot 1 + 2 * (H * P : Nat), L * conv - H * P⟩ else ⟨slot 2, L * (G * N)⟩,
    dt := ⟨slot 3, H * (NC * C)⟩, dA := ⟨slot 4, H * (NC * C)⟩, seqIdx := ⟨slot 5, L⟩,
    states := ⟨slot 6, NC * (H * (P * N))⟩ }

/-- Two buffers at disjoint byte ranges share no byte. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The source's test cases (lines 364–375): (request lengths, `nheads`, `headdim`, `dstate`,
`ngroups`, `chunk_size`, rows layout). -/
def setup_1 : Inst := mk [112] 3 5 5 3 16 false
def setup_2 : Inst := mk [128] 32 128 128 32 32 false
def setup_3 : Inst := mk [16, 20] 16 64 64 16 8 false
def setup_4 : Inst := mk [270, 88, 212, 203] 16 64 64 16 256 false
def setup_5 : Inst := mk [138, 225] 13 32 32 13 128 false
def setup_6 : Inst := mk [5, 30] 4 21 21 4 256 false
def setup_7 : Inst := mk [1000, 1, 2049, 64, 64, 300] 80 64 128 1 256 true
def setup_8 : Inst := mk [8192] 128 64 128 8 256 true
def setup_9 : Inst := mk (List.replicate 40 3 ++ [500]) 128 80 128 8 256 true
def setup_10 : Inst := mk [300, 1700] 64 64 256 1 256 true

/-- Edges `check_gpu.py` runs (`SETUP.md`, The launches): the Mamba-2 layer at 32768 tokens with
128 heads (Nemotron-H's shapes at tensor parallelism 1); 200 one-token requests; chunks of `1`
and `2`; one head of `headdim = dstate = 1`; `headdim = 65`, `dstate = 129` (one more than a
block). -/
def setup_e1 : Inst := mk [8000, 24768] 128 64 128 8 256 true
def setup_e2 : Inst := mk (List.replicate 200 1) 8 64 128 1 256 true
def setup_e3 : Inst := mk [3, 2, 4] 4 16 16 2 1 true
def setup_e4 : Inst := mk [7, 1, 8] 4 16 16 4 2 false
def setup_e5 : Inst := mk [37] 1 1 1 1 16 false
def setup_e6 : Inst := mk [100, 157] 6 65 129 3 64 true
/-- One long request of 180,000 tokens at 128 heads × 80 × 128 with eight groups (Nemotron Nano 2's
shapes, chunked prefill disabled): `x.stride(0) = B.stride(0) = seqlen·conv_dim` is above `2^31`,
an `int64` argument. Its tensors exceed the GPU check's memory budget; `check_gpu.py` runs the
`int64` batch strides on a smaller launch. -/
def setup_e7 : Inst := mk [180000] 128 80 128 8 256 true

/-- `Inst.ok` of each instance, every condition decided (`k` the exponent of `chunk_size`); the
five disjointness facts from the far-apart slots. -/
theorem setup_1_ok : setup_1.ok := by
  constructor <;> first | decide | exact ⟨4, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_2_ok : setup_2.ok := by
  constructor <;> first | decide | exact ⟨5, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_3_ok : setup_3.ok := by
  constructor <;> first | decide | exact ⟨3, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_4_ok : setup_4.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_5_ok : setup_5.ok := by
  constructor <;> first | decide | exact ⟨7, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_6_ok : setup_6.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_7_ok : setup_7.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_8_ok : setup_8.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_9_ok : setup_9.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_10_ok : setup_10.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_e1_ok : setup_e1.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

set_option maxRecDepth 20000 in
theorem setup_e2_ok : setup_e2.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_e3_ok : setup_e3.ok := by
  constructor <;> first | decide | exact ⟨0, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_e4_ok : setup_e4.ok := by
  constructor <;> first | decide | exact ⟨1, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_e5_ok : setup_e5.ok := by
  constructor <;> first | decide | exact ⟨4, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_e6_ok : setup_e6.ok := by
  constructor <;> first | decide | exact ⟨6, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

theorem setup_e7_ok : setup_e7.ok := by
  constructor <;> first | decide | exact ⟨8, by decide⟩ |
    exact ⟨bytes_disj _ _ (by decide), bytes_disj _ _ (by decide), bytes_disj _ _ (by decide),
      bytes_disj _ _ (by decide), bytes_disj _ _ (by decide)⟩

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

/-- `setup_1` (one request) in a memory of zeros: every buffer is valid, and `seq_idx` (all `0`)
is the host's, since every token belongs to request `0`. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_1, zeroMem, setup_1_ok, ?_⟩
  refine ⟨valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide), ?_⟩
  intro t ht
  have h0 : reqOf setup_1 t = 0 := by simp [reqOf, laterReqs, setup_1, mk]
  rw [h0]
  have ht' : t < 112 := by simpa [setup_1, mk] using ht
  refine zero_i32 ?_
  simp only [Mem.Aligned, Buf.ptr, setup_1, mk, slot, show DType.i32.bytes = 4 from rfl]
  omega

end Claims.SsdChunkState
