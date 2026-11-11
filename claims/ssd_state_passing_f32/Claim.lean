import TVA

/-!
# `ssd_state_passing_f32`: Mamba-2 SSD state passing, SGLang

The kernel is `_state_passing_fwd_kernel` of `source/ssd_state_passing.py` (lines 79–231; it reads
the module-level names `triton` and `tl`, lines 75–76). The host `state_passing_fwd` (lines 234–333)
launches it with `HAS_SEQ_IDX = IS_CONT_BATCHED = True`, `HAS_INITSTATES` either value
(`initial_states` and `chunk_offsets` given, or both `None`), `BLOCK_SIZE = 1024`, `num_warps = 4`,
`num_stages = 3`, on the grid `(cdiv(dim, 1024), 1, nheads)`; the metadata `seq_idx` and
`chunk_offsets` are what the host builds from `query_start_loc`
(`query_start_loc_to_chunk_indices_offsets`, lines 336–368, and `make_inputs`, lines 375–380).
Program `(pid_m, ·, pid_h)` carries the state of `1024` lanes of head `pid_h` across the physical
chunks of a packed prefill batch, restarting it where a request starts.

What is claimed: every covered launch ends normally; for every head `h` and state element `m`,
`out[0, 0, h, m]` is `initial_states[0, h, m]` up to NaN payload (the `float32` `+0` without initial
states); for `1 ≤ c < nchunks`, `out[0, c, h, m]` holds a `float32` value, and when the inputs'
values are covered (`Vals`) it is finite and within `tol c` of the exact recurrence `sRef c`,
computed in the reals from `query_start_loc`, the chunk states, the initial states (or `0`) and
`dA_cumsum`; `final_states[0, h, m]` the same at `c = nchunks`; every other byte is unchanged.
`tol` is decay-aware: a recursion over the chunks of the kernel's own roundings.

The conditions: `batch = 1`, contiguous tensors with the host's strides; at least one head, fewer
than `2^16`; `chunk_size` a power of two; a packed batch of `1` to `8192` requests, each with a
token, at most `2^22` tokens; the metadata are the host's; the `int32` arguments and products do
not wrap; the buffers hold their tensors, and the two outputs share no byte with the inputs or each
other. `Vals`: finite inputs, `dA_cumsum ≤ 0`, each chunk's decay exponent at most `64`, each
update below `2^126`.

The tolerance rests on the device fact for `tl.exp` on `float32` (`Device.ExpF32`).
-/

namespace Claims.SsdStatePassing

open TVA

/-! ## The model -/

/-- `_state_passing_fwd_kernel` with `HAS_INITSTATES = True` (source lines 79–231). The five `float32` tensors are `*fp32`
pointers, `seq_idx` and `chunk_offsets` `*i32` pointers; `chunk_meta_num`, `dim`, `nchunks`,
`seqlen`, `chunk_size` and the twenty strides are `int32` scalar arguments (`Inst.ok` keeps each
below `2^31`; an integer argument equal to `1` is the compile-time constant `1`, and each of its uses
in the kernel, an operand of `+`, `-` or `*`, a comparison, a `min`, a range bound, builds the
constant in the computation dtype `int32` with the same value, so the model's `IntV` is the same:
see `model`). `BLOCK_SIZE : tl.constexpr` is a Lean parameter.

`HAS_INITSTATES`, `HAS_SEQ_IDX` and `IS_CONT_BATCHED` are `True` on the launches this model covers
(`HAS_INITSTATES = False`: `ssd_state_passing_kernel_noinit`), so each compile-time `if` visits
only its taken branch; the model writes that branch. -/
def ssd_state_passing_kernel
    (states_ptr out_ptr final_states_ptr dA_cs_ptr initstates_ptr : Ptr .f32)
    (seq_idx_ptr chunk_offsets_ptr : Ptr .i32)
    (chunk_meta_num dim nchunks seqlen chunk_size : I32)
    (stride_states_batch stride_states_chunk stride_states_head stride_states_dim
      stride_out_batch stride_out_chunk stride_out_head stride_out_dim
      stride_final_states_batch stride_final_states_head stride_final_states_dim
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      stride_initstates_batch stride_initstates_head stride_initstates_dim
      stride_seq_idx_batch stride_seq_idx_seqlen : I32)
    (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 122: axis 1 of the grid, an `int32`
    let pid_b := tl.program_id ctx 1
    -- line 123: axis 2 of the grid
    let pid_h := tl.program_id ctx 2
    -- line 124: axis 0 of the grid
    let pid_m := tl.program_id ctx 0
    -- line 125: `int32` products and sum (wrapping), then a pointer `+`
    let states_ptr := states_ptr + (pid_b * stride_states_batch + pid_h * stride_states_head)
    -- lines 126–130
    let dA_end_off : I32 := pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
      + (chunk_size - IntV.ofInt 1) * stride_dA_cs_csize
    let dA_cs_ptr := dA_cs_ptr + dA_end_off
    -- line 131
    let out_ptr := out_ptr + (pid_b * stride_out_batch + pid_h * stride_out_head)
    -- lines 132–134
    let final_states_ptr := final_states_ptr +
      (pid_b * stride_final_states_batch + pid_h * stride_final_states_head)
    -- lines 135–138: `if HAS_INITSTATES:` taken; `not IS_CONT_BATCHED` is false, so line 138 is
    -- not visited
    let initstates_ptr := initstates_ptr + pid_h * stride_initstates_head
    -- lines 140–141: `if HAS_SEQ_IDX:` taken
    let seq_idx_ptr := seq_idx_ptr + pid_b * stride_seq_idx_batch
    -- line 143: `int32` products and sums (wrapping)
    let offs_m := pid_m * (IntV.ofNat BLOCK_SIZE : I32) + tl.arange 0 BLOCK_SIZE
    -- line 144: a block of pointers
    let states_ptrs := states_ptr + offs_m * stride_states_dim
    -- line 145
    let out_ptrs := out_ptr + offs_m * stride_out_dim
    -- line 146
    let final_states_ptrs := final_states_ptr + offs_m * stride_final_states_dim
    -- lines 149–151: `if not HAS_INITSTATES:` not taken; lines 152–153, the `else` branch
    let initstates_ptr := initstates_ptr + offs_m * stride_initstates_dim
    let initstates_ptrs := initstates_ptr
    -- line 156: `other=0.0` is the `float32` `+0`; `.to(tl.float32)` is the identity on `float32`
    let states : Tensor [BLOCK_SIZE] F32 ← tl.load initstates_ptrs (some (offs_m.ltS dim))
      (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    -- line 158: `states` is already `float32`, the pointer's dtype
    tl.store out_ptrs states (some (offs_m.ltS dim))
    -- line 159: a pointer `+` on each lane (64-bit addresses)
    let out_ptrs := out_ptrs.map (fun (q : Ptr .f32) => q + stride_out_chunk)
    -- lines 160–161: names assigned the Python `0`: `int32` scalar tensors holding `0`
    let prev_seq_idx_chunk_end : I32 := IntV.ofInt 0
    let logical_chunk_idx : I32 := IntV.ofInt 0
    -- line 162: `for c in range(nchunks)` (a plain name: nothing to evaluate twice); the loop
    -- carries every name defined before it that the body assigns: `states`, `initstates_ptrs`,
    -- `states_ptrs`, `dA_cs_ptr`, `out_ptrs`, `prev_seq_idx_chunk_end`, `logical_chunk_idx`
    let (_states, _initstates_ptrs, _states_ptrs, _dA_cs_ptr, _out_ptrs, _prev, _logical) ←
      tl.forRange (IntV.ofInt 0 : I32) nchunks (IntV.ofInt 1 : I32)
        (states, initstates_ptrs, states_ptrs, dA_cs_ptr, out_ptrs, prev_seq_idx_chunk_end,
          logical_chunk_idx)
        fun c (states, initstates_ptrs, states_ptrs, dA_cs_ptr, out_ptrs, prev_seq_idx_chunk_end,
          logical_chunk_idx) => do
      -- line 163: `other` the `float32` `+0`; `.to(tl.float32)` the identity
      let new_states : Tensor [BLOCK_SIZE] F32 ← tl.load states_ptrs (some (offs_m.ltS dim))
        (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
      -- line 164: a scalar load; `.to(tl.float32)` the identity
      let dA_cs : F32 ← tl.load1 dA_cs_ptr
      -- line 165: the Python `True` assigned to a name: an `int1` scalar tensor holding `true`
      -- (under `HAS_INITSTATES` nothing reassigns it)
      let scale_mask : Tensor [] Bool := Tensor.scalar true
      -- line 166: `if HAS_SEQ_IDX:` taken
      -- lines 170–173: Python's `min` with a tensor operand is `tl.minimum`, on `int32` the exact
      -- `tl.iminimum`; the offset is `int32` (wrapping); a scalar load
      let seq_end_off : I32 := (tl.iminimum (Tensor.scalar ((c + IntV.ofInt 1) * chunk_size))
        (Tensor.scalar seqlen)).item - IntV.ofInt 1
      let seq_idx_chunk_end : I32 ← tl.load1 (seq_idx_ptr +
        seq_end_off * stride_seq_idx_seqlen)
      -- lines 174–175: `if HAS_INITSTATES:` taken; `IS_CONT_BATCHED and …` is the right operand,
      -- the runtime `int32` comparison (the constexpr `True` does not decide `and`): a Lean `if`
      -- on its value, each branch's operations in its branch
      let (states, initstates_ptrs, logical_chunk_idx, dA_cs) ←
        if !(IntV.eq prev_seq_idx_chunk_end seq_idx_chunk_end) then do
          -- lines 180–182: an `int32` product, then a pointer `+` on each lane
          let initstates_ptrs :=
            initstates_ptr.map (fun (q : Ptr .f32) => q + seq_idx_chunk_end * stride_initstates_batch)
          -- lines 185–187: as line 156
          let states : Tensor [BLOCK_SIZE] F32 ← tl.load initstates_ptrs (some (offs_m.ltS dim))
            (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
          -- lines 194–197: as lines 170–173, without the `- 1`
          let seq_start_off : I32 := (tl.iminimum (Tensor.scalar (c * chunk_size))
            (Tensor.scalar seqlen)).item
          let seq_idx_chunk_start : I32 ← tl.load1 (seq_idx_ptr +
            seq_start_off * stride_seq_idx_seqlen)
          -- line 198: `int32` (wrapping)
          let logical_chunk_idx := logical_chunk_idx +
            (seq_idx_chunk_end - seq_idx_chunk_start)
          -- lines 200–204: a masked scalar load; the mask is the signed `int32` comparison with
          -- `chunk_meta_num`; `other=0` is the `int32` `0`
          let c_off : I32 := (← tl.load (Tensor.scalar (chunk_offsets_ptr + logical_chunk_idx))
            (some ((Tensor.scalar logical_chunk_idx).ltS chunk_meta_num))
            (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
          -- lines 206–215: `if c_off > 0:` a runtime `if` on the loaded scalar
          let dA_cs ←
            if IntV.lt (IntV.ofInt 0) c_off then do
              -- lines 208–214: a masked scalar load at `dA_cs_ptr - a + b`, written
              -- `dA_cs_ptr + (b - a)` (the same address: the `int32` difference does not wrap);
              -- the mask is `(c_off - 1) > -1 and c_off < chunk_size`, an `and` of two `int1`
              -- scalars (both evaluated; the first written as the equivalent signed
              -- `-1 < c_off - 1`); `other=0.0` is the `float32` `+0`
              let dA_cs_boundary : F32 := (← tl.load
                (Tensor.scalar (dA_cs_ptr + ((c_off - IntV.ofInt 1) * stride_dA_cs_csize
                  - (chunk_size - IntV.ofInt 1) * stride_dA_cs_csize)))
                (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1))
                      (c_off - IntV.ofInt 1)))
                  ((Tensor.scalar c_off).ltS chunk_size)))
                (some (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0))))).item
              -- line 215: the IEEE `-` (two loaded values; no product feeds it)
              let dA_subbed ← tl.sub (Tensor.scalar dA_cs) (Tensor.scalar dA_cs_boundary)
              pure dA_subbed.item
            else
              pure dA_cs
          pure (states, initstates_ptrs, logical_chunk_idx, dA_cs)
        else
          pure (states, initstates_ptrs, logical_chunk_idx, dA_cs)
      -- line 218: `int32` (wrapping)
      let logical_chunk_idx := logical_chunk_idx + IntV.ofInt 1
      -- line 219: the `else` of `if HAS_INITSTATES:` is not visited
      -- line 221
      let prev_seq_idx_chunk_end := seq_idx_chunk_end
      -- line 223: `tl.where(scale_mask, tl.exp(dA_cs), 0.0)`: the `tl.exp` runs; the condition
      -- is the `int1` `true`, and `0.0` is built in `float32`
      let e ← tl.exp (Tensor.scalar dA_cs)
      let scale : F32 := (tl.where scale_mask e
        (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0)))).item
      -- line 224: `scale * states + new_states`: a product feeds the addition, so `tl.mul_add`
      -- (the two IEEE operations, or one `fma`)
      let states ← tl.mul_add (bc scale) states new_states
      -- lines 225–228: a runtime `if` on the `int32` scalars `c` and `nchunks - 1`
      if IntV.lt c (nchunks - IntV.ofInt 1) then
        -- line 226
        tl.store out_ptrs states (some (offs_m.ltS dim))
      else
        -- line 228
        tl.store final_states_ptrs states (some (offs_m.ltS dim))
      -- lines 229–231: pointer `+`s (64-bit addresses)
      let states_ptrs := states_ptrs.map (fun (q : Ptr .f32) => q + stride_states_chunk)
      let dA_cs_ptr := dA_cs_ptr + stride_dA_cs_chunk
      let out_ptrs := out_ptrs.map (fun (q : Ptr .f32) => q + stride_out_chunk)
      pure (states, initstates_ptrs, states_ptrs, dA_cs_ptr, out_ptrs, prev_seq_idx_chunk_end,
        logical_chunk_idx)
    pure ()

/-- `_state_passing_fwd_kernel` with `HAS_INITSTATES = False` (source lines 79–231): the host is
given `initial_states = None` and `chunk_offsets = None` (SGLang builds no `chunk_offsets` then), and
passes `chunk_meta_num = 0` and the three initial-state strides `0`, which no line this launch runs
reads, so the model has no parameter for them. `HAS_SEQ_IDX` and `IS_CONT_BATCHED` are `True` (the
latter only matters under `HAS_INITSTATES`). The other parameters are as in
`ssd_state_passing_kernel`. -/
def ssd_state_passing_kernel_noinit
    (states_ptr out_ptr final_states_ptr dA_cs_ptr : Ptr .f32) (seq_idx_ptr : Ptr .i32)
    (dim nchunks seqlen chunk_size : I32)
    (stride_states_batch stride_states_chunk stride_states_head stride_states_dim
      stride_out_batch stride_out_chunk stride_out_head stride_out_dim
      stride_final_states_batch stride_final_states_head stride_final_states_dim
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      stride_seq_idx_batch stride_seq_idx_seqlen : I32)
    (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- lines 122–134: as in `ssd_state_passing_kernel`
    let pid_b := tl.program_id ctx 1
    let pid_h := tl.program_id ctx 2
    let pid_m := tl.program_id ctx 0
    let states_ptr := states_ptr + (pid_b * stride_states_batch + pid_h * stride_states_head)
    let dA_end_off : I32 := pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
      + (chunk_size - IntV.ofInt 1) * stride_dA_cs_csize
    let dA_cs_ptr := dA_cs_ptr + dA_end_off
    let out_ptr := out_ptr + (pid_b * stride_out_batch + pid_h * stride_out_head)
    let final_states_ptr := final_states_ptr +
      (pid_b * stride_final_states_batch + pid_h * stride_final_states_head)
    -- lines 135–138: `if HAS_INITSTATES:` not taken
    -- lines 140–141: `if HAS_SEQ_IDX:` taken
    let seq_idx_ptr := seq_idx_ptr + pid_b * stride_seq_idx_batch
    -- lines 143–146: as in `ssd_state_passing_kernel`
    let offs_m := pid_m * (IntV.ofNat BLOCK_SIZE : I32) + tl.arange 0 BLOCK_SIZE
    let states_ptrs := states_ptr + offs_m * stride_states_dim
    let out_ptrs := out_ptr + offs_m * stride_out_dim
    let final_states_ptrs := final_states_ptr + offs_m * stride_final_states_dim
    -- lines 149–150: `if not HAS_INITSTATES:` taken: a block of `float32` `+0`
    let states : Tensor [BLOCK_SIZE] F32 := tl.zeros [BLOCK_SIZE] .f32
    -- line 158
    tl.store out_ptrs states (some (offs_m.ltS dim))
    -- line 159
    let out_ptrs := out_ptrs.map (fun (q : Ptr .f32) => q + stride_out_chunk)
    -- lines 160–161: `int32` scalar tensors holding `0` (`logical_chunk_idx` is not used again)
    let prev_seq_idx_chunk_end : I32 := IntV.ofInt 0
    let _logical_chunk_idx : I32 := IntV.ofInt 0
    -- line 162: the loop carries the names defined before it that the body assigns: `states`,
    -- `states_ptrs`, `dA_cs_ptr`, `out_ptrs`, `prev_seq_idx_chunk_end` (no visited line assigns
    -- `logical_chunk_idx`, and `initstates_ptrs` is not defined)
    let (_states, _states_ptrs, _dA_cs_ptr, _out_ptrs, _prev) ←
      tl.forRange (IntV.ofInt 0 : I32) nchunks (IntV.ofInt 1 : I32)
        (states, states_ptrs, dA_cs_ptr, out_ptrs, prev_seq_idx_chunk_end)
        fun c (states, states_ptrs, dA_cs_ptr, out_ptrs, prev_seq_idx_chunk_end) => do
      -- lines 163–164: as in `ssd_state_passing_kernel`
      let new_states : Tensor [BLOCK_SIZE] F32 ← tl.load states_ptrs (some (offs_m.ltS dim))
        (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
      let dA_cs : F32 ← tl.load1 dA_cs_ptr
      -- line 165: the `int1` `true`, reassigned at line 220 before any use
      let _scale_mask : Tensor [] Bool := Tensor.scalar true
      -- line 166: `if HAS_SEQ_IDX:` taken; lines 170–173: as in `ssd_state_passing_kernel`
      let seq_end_off : I32 := (tl.iminimum (Tensor.scalar ((c + IntV.ofInt 1) * chunk_size))
        (Tensor.scalar seqlen)).item - IntV.ofInt 1
      let seq_idx_chunk_end : I32 ← tl.load1 (seq_idx_ptr +
        seq_end_off * stride_seq_idx_seqlen)
      -- line 174: `if HAS_INITSTATES:` not taken; lines 219–220, the `else` branch: the `int32`
      -- comparison `==`, an `int1` scalar
      let scale_mask : Tensor [] Bool :=
        Tensor.scalar (IntV.eq seq_idx_chunk_end prev_seq_idx_chunk_end)
      -- line 221
      let prev_seq_idx_chunk_end := seq_idx_chunk_end
      -- line 223: the `tl.exp` runs; `0.0` is built in `float32`; the scale is `+0` where the
      -- request at the chunk's end changes
      let e ← tl.exp (Tensor.scalar dA_cs)
      let scale : F32 := (tl.where scale_mask e
        (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0)))).item
      -- line 224: `tl.mul_add`, as in `ssd_state_passing_kernel`
      let states ← tl.mul_add (bc scale) states new_states
      -- lines 225–228
      if IntV.lt c (nchunks - IntV.ofInt 1) then
        tl.store out_ptrs states (some (offs_m.ltS dim))
      else
        tl.store final_states_ptrs states (some (offs_m.ltS dim))
      -- lines 229–231
      let states_ptrs := states_ptrs.map (fun (q : Ptr .f32) => q + stride_states_chunk)
      let dA_cs_ptr := dA_cs_ptr + stride_dA_cs_chunk
      let out_ptrs := out_ptrs.map (fun (q : Ptr .f32) => q + stride_out_chunk)
      pure (states, states_ptrs, dA_cs_ptr, out_ptrs, prev_seq_idx_chunk_end)
    pure ()

/-! ## The conditions -/

/-- What the host chooses: one launch of `state_passing_fwd` with `seq_idx` given, and
`initial_states` with `chunk_offsets` given or both `None` (`hasInit`; `SETUP.md` fixes the other
flags and `out_dtype = float32`), `batch = 1`. The request boundaries `query_start_loc` (`qsl`) are
the host's input from which it builds the metadata; the buffers are given by their storage (`Buf`;
without initial states `initst` and `chunkOffs` stand for the host's `None`s and are empty). All
tensors are contiguous, so the strides the host passes are the shapes' products (`sSb`, …). -/
structure Inst where
  /-- `HAS_INITSTATES`: `initial_states` and `chunk_offsets` given; `False` when SGLang passes
  `None` (no request of the batch has a cached prefix). -/
  hasInit : Bool
  /-- `nheads` (`states.shape[2]`): the grid's `z`. -/
  nheads : Nat
  /-- `dim` (`states.shape[3]`, `headdim · dstate` flattened). -/
  dim : Nat
  /-- `nchunks` (`states.shape[1]`, the physical chunks). -/
  nchunks : Nat
  /-- `seqlen` (`seq_idx.shape[1]`, the packed tokens). -/
  seqlen : Nat
  /-- `chunk_size` (`dA_cumsum.shape[3]`). -/
  chunk : Nat
  /-- `num_requests` (`initial_states.shape[0]`). -/
  nreq : Nat
  /-- `query_start_loc`: request `r` holds the tokens `qsl r ≤ t < qsl (r + 1)`. -/
  qsl : Nat → Nat
  /-- `N = len(chunk_offsets)`, passed as `chunk_meta_num` (`0` without initial states). -/
  nmeta : Nat
  /-- The storage of `states` (`float32`). -/
  states : Buf .f32
  /-- The storage of `dA_cumsum` (`float32`). -/
  dA : Buf .f32
  /-- The storage of `initial_states` (`float32`). -/
  initst : Buf .f32
  /-- The storage of `seq_idx` (`int32`). -/
  seqIdx : Buf .i32
  /-- The storage of `chunk_offsets` (`int32`). -/
  chunkOffs : Buf .i32
  /-- The storage of `out` (`float32`, allocated by the host). -/
  out : Buf .f32
  /-- The storage of `final_states` (`float32`, allocated by the host). -/
  finalst : Buf .f32

/-- `BLOCK_SIZE`, fixed by the host (line 290). -/
def BL : Nat := 1024

/-- The grid's `x`: `triton.cdiv(dim, BLOCK_SIZE)` (line 291). -/
def gx (d : Inst) : Nat := (d.dim + BL - 1) / BL

/-- The strides the host passes for contiguous tensors with `batch = 1`: `states.stride(0…3)` and
`out.stride(0…3)` are `nchunks·nheads·dim`, `nheads·dim`, `dim`, `1`; `final_states.stride(0…2)`
are `nheads·dim`, `dim`, `1`; the launch passes `dA_cumsum.stride(0), stride(2), stride(1),
stride(3)` (lines 317–320), for `[1, nheads, nchunks, chunk_size]` `nheads·nchunks·chunk`,
`chunk`, `nchunks·chunk`, `1`; `initial_states.stride(0…2)` are `nheads·dim`, `dim`, `1`;
`seq_idx.stride(0…1)` are `seqlen`, `1`. -/
def sSb (d : Inst) : Nat := d.nchunks * (d.nheads * d.dim)

def sSc (d : Inst) : Nat := d.nheads * d.dim

def sSh (d : Inst) : Nat := d.dim

def sDb (d : Inst) : Nat := d.nheads * (d.nchunks * d.chunk)

def sDc (d : Inst) : Nat := d.chunk

def sDh (d : Inst) : Nat := d.nchunks * d.chunk

def sQb (d : Inst) : Nat := d.seqlen

/-- Element `(0, c, h, m)` of `states` and of `out`. -/
def stIdx (d : Inst) (c h m : Nat) : Nat := c * (d.nheads * d.dim) + h * d.dim + m

/-- Element `(0, h, c, p)` of `dA_cumsum`. -/
def dAIdx (d : Inst) (h c p : Nat) : Nat := h * (d.nchunks * d.chunk) + c * d.chunk + p

/-- Element `(r, h, m)` of `initial_states`. -/
def inIdx (d : Inst) (r h m : Nat) : Nat := r * (d.nheads * d.dim) + h * d.dim + m

/-- Element `(0, h, m)` of `final_states`. -/
def finIdx (d : Inst) (h m : Nat) : Nat := h * d.dim + m

/-! ### The host's metadata, from `query_start_loc` -/

/-- The requests after the first, `1, …, nreq − 1`: the host's loop over
`zip(cu_seqlens[:-1], cu_seqlens[1:])` (line 355) visits their starts `qsl r`. -/
def laterReqs (d : Inst) : List Nat := List.range' 1 (d.nreq - 1)

/-- The request of token `t`: the number of later requests that start at or before `t`
(`seq_idx[0, t]`, `torch.repeat_interleave(arange(nreq), seqlens)`, line 378). -/
def reqOf (d : Inst) (t : Nat) : Nat := ((laterReqs d).filter (fun r => decide (d.qsl r ≤ t))).length

/-- The host's `N` (lines 345–348): `ceil(seqlen / chunk_size)` plus one per later request whose
start is not a multiple of `chunk_size`. -/
def hostN (d : Inst) : Nat :=
  (d.seqlen + d.chunk - 1) / d.chunk +
    ((laterReqs d).filter (fun r => decide (d.qsl r % d.chunk ≠ 0))).length

/-- One trip of the host's loop (lines 355–366) on the request starting at `s = qsl r`: `p` counts
the insertions, `_s = s // chunk_size + p`, and `chunk_offsets[_s] = s % chunk_size`. -/
def offStep (d : Inst) (st : Nat × (Nat → Nat)) (r : Nat) : Nat × (Nat → Nat) :=
  let s := d.qsl r
  let p := st.1 + (if s % d.chunk > 0 then 1 else 0)
  (p, Function.update st.2 (s / d.chunk + p) (s % d.chunk))

/-- `chunk_offsets` as the host builds it (lines 350–366): zeros, then the loop's writes in order. -/
def hostOff (d : Inst) : Nat → Nat := ((laterReqs d).foldl (offStep d) (0, fun _ => 0)).2

/-- The instances covered (README, Conditions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- At least one head; the grid's `z` within CUDA's limit `65535`. -/
  heads_pos : 0 < d.nheads
  heads_lt : d.nheads < 2 ^ 16
  /-- At least one state element; the lanes `pid_m·1024 + arange(1024)` stay below `2^31`
  (`cdiv(dim, 1024)·1024 ≤ dim + 1023`). -/
  dim_pos : 0 < d.dim
  dim_lt : d.dim + 1023 < 2 ^ 31
  /-- The grid `(cdiv(dim, 1024), 1, nheads)` has fewer than `2^31` programs. -/
  grid_lt : gx d * d.nheads < 2 ^ 31
  /-- `chunk_size` is a power of two (the code base asserts it, `ssd_combined.py:80`). -/
  chunk_pow2 : ∃ k, d.chunk = 2 ^ k
  /-- At least one request; `query_start_loc` starts at `0`, every request has at least one token,
  and the last ends at `seqlen` (the packed batch). -/
  nreq_pos : 0 < d.nreq
  qsl_zero : d.qsl 0 = 0
  qsl_lt : ∀ r, r < d.nreq → d.qsl r < d.qsl (r + 1)
  qsl_end : d.qsl d.nreq = d.seqlen
  /-- `seqlen`, the packed batch's tokens, is a context length: at most `2^22`. -/
  seqlen_ctx : d.seqlen ≤ 2 ^ 22
  /-- `num_requests`, the requests of the batch, is a batch size: at most `8192`. -/
  nreq_le : d.nreq ≤ 8192
  /-- `nchunks` is the host's `cdiv(seqlen, chunk_size)` (`states.shape[1]`). -/
  nchunks_eq : d.nchunks = (d.seqlen + d.chunk - 1) / d.chunk
  /-- `len(chunk_offsets)` is the host's `N` (`0` without initial states), an `int32`. -/
  nmeta_eq : d.nmeta = if d.hasInit then hostN d else 0
  nmeta_lt : d.nmeta < 2 ^ 31
  /-- The token positions `(c + 1)·chunk_size` are `int32` values (so are `seqlen`, `chunk_size`
  and `nchunks`). -/
  tok_lt : d.nchunks * d.chunk < 2 ^ 31
  /-- `dA_cumsum.stride(0) = nheads·nchunks·chunk_size` is an `int32` argument, and so the head
  offset `pid_h·stride_dA_cs_head + chunk_size − 1` (an `int32` sum) does not wrap. -/
  dA_lt : d.nheads * (d.nchunks * d.chunk) < 2 ^ 31
  /-- `states.stride(0) = out.stride(0) = nchunks·nheads·dim` is an `int32` argument (and so are
  the other strides, and the head offsets `pid_h·dim`). -/
  st_lt : d.nchunks * (d.nheads * d.dim) < 2 ^ 31
  /-- With initial states, the `int32` product `seq_idx_chunk_end · stride_initstates_batch`
  (line 181) does not wrap: `(nreq − 1)·nheads·dim < 2^31` (a limit of the kernel, README,
  Exclusions). -/
  init_lt : d.hasInit = true → (d.nreq - 1) * (d.nheads * d.dim) < 2 ^ 31
  /-- Every element the indices name lies in its tensor's storage. -/
  states_len : d.nchunks * (d.nheads * d.dim) ≤ d.states.len
  dA_len : d.nheads * (d.nchunks * d.chunk) ≤ d.dA.len
  init_len : d.hasInit = true → d.nreq * (d.nheads * d.dim) ≤ d.initst.len
  seq_len : d.seqlen ≤ d.seqIdx.len
  offs_len : d.nmeta ≤ d.chunkOffs.len
  out_len : d.nchunks * (d.nheads * d.dim) ≤ d.out.len
  fin_len : d.nheads * d.dim ≤ d.finalst.len
  /-- The two outputs' storage shares no byte with the inputs' or with each other (both are
  allocated by the host, lines 274–279). -/
  disj : Disjoint d.out.bytes d.states.bytes ∧ Disjoint d.out.bytes d.dA.bytes ∧
    Disjoint d.out.bytes d.initst.bytes ∧ Disjoint d.out.bytes d.seqIdx.bytes ∧
    Disjoint d.out.bytes d.chunkOffs.bytes ∧ Disjoint d.out.bytes d.finalst.bytes ∧
    Disjoint d.finalst.bytes d.states.bytes ∧ Disjoint d.finalst.bytes d.dA.bytes ∧
    Disjoint d.finalst.bytes d.initst.bytes ∧ Disjoint d.finalst.bytes d.seqIdx.bytes ∧
    Disjoint d.finalst.bytes d.chunkOffs.bytes

/-- The model on an instance: the launch of lines 293–332, with the contiguous strides and
`BLOCK_SIZE = 1024`. An integer argument equal to `1` is the compile-time constant `1`; each use of
it in the kernel builds the constant in the computation dtype `int32` with the same value, so the
model's `IntV.ofNat 1` is the same. Without initial states the host passes `None` for
`initial_states` and `chunk_offsets`, `0` for `chunk_meta_num` and the initial-state strides, none
of which `ssd_state_passing_kernel_noinit` reads. -/
def model (d : Inst) : Kernel :=
  if d.hasInit then ssd_state_passing_kernel (d.states.ptr 0) (d.out.ptr 0) (d.finalst.ptr 0) (d.dA.ptr 0)
    (d.initst.ptr 0) (d.seqIdx.ptr 0) (d.chunkOffs.ptr 0)
    (IntV.ofNat d.nmeta) (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen)
    (IntV.ofNat d.chunk)
    (IntV.ofNat (sSb d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sSb d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sDb d)) (IntV.ofNat (sDc d)) (IntV.ofNat (sDh d)) (IntV.ofNat 1)
    (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sQb d)) (IntV.ofNat 1)
    BL
  else ssd_state_passing_kernel_noinit (d.states.ptr 0) (d.out.ptr 0) (d.finalst.ptr 0) (d.dA.ptr 0)
    (d.seqIdx.ptr 0)
    (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen) (IntV.ofNat d.chunk)
    (IntV.ofNat (sSb d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sSb d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sSc d)) (IntV.ofNat (sSh d)) (IntV.ofNat 1)
    (IntV.ofNat (sDb d)) (IntV.ofNat (sDc d)) (IntV.ofNat (sDh d)) (IntV.ofNat 1)
    (IntV.ofNat (sQb d)) (IntV.ofNat 1)
    BL

/-- The host's launch (lines 290–332): the grid `(cdiv(dim, 1024), 1, nheads)`; the source's
block shape `[BLOCK_SIZE]` (`tl.arange`, the block loads and stores); `enable_fp_fusion` at
Triton's default (`true`); `num_ctas = 1`, no `launch_pdl`, `debug = false`. `num_warps = 4` and
`num_stages = 3` are performance-only. -/
def launch (d : Inst) : Launch :=
  { grid := ![gx d, 1, d.nheads], fusion := true, blocks := [[BL]] }

/-- The `int32` value at element `i` of an `int32` buffer in memory `m` (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some v => v.toInt
  | none => 0

/-- The value of element `i` of a `float32` buffer in memory `m` (`0` when absent). -/
def valRat (m : Mem) (b : Buf .f32) (i : Nat) : ℚ :=
  match m.read .f32 (b.ptr (i : Int)).addr with
  | some v => v.toRat
  | none => 0

/-- The metadata are the host's: `seq_idx[0, t]` is the request of token `t`, and `chunk_offsets`
is what `query_start_loc_to_chunk_indices_offsets` builds (lines 336–368), both from
`query_start_loc` (without initial states `nmeta = 0` and the second clause is empty). -/
def MetaOk (d : Inst) (m : Mem) : Prop :=
  (∀ t, t < d.seqlen → m.read .i32 (d.seqIdx.ptr (t : Int)).addr = some (IntV.ofNat (reqOf d t))) ∧
  (∀ k, k < d.nmeta → m.read .i32 (d.chunkOffs.ptr (k : Int)).addr = some (IntV.ofNat (hostOff d k)))

/-- The inputs covered: the seven buffers are valid (aligned, inside `[0, 2^64)`, every byte
present; without initial states the host's `None`s are empty buffers), and the metadata are the host's (`MetaOk`). Any `float32` values: `Post` says what holds
when the values are covered (`Vals`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.states.Valid init ∧ d.dA.Valid init ∧ d.initst.Valid init ∧ d.seqIdx.Valid init ∧
    d.chunkOffs.Valid init ∧ d.out.Valid init ∧ d.finalst.Valid init ∧ MetaOk d init

/-! ## The specification -/

/-! ### The reference, from the host's inputs -/

/-- The last token of physical chunk `c`: `min((c + 1)·chunk_size, seqlen) − 1`. -/
def endTok (d : Inst) (c : Nat) : Nat := min ((c + 1) * d.chunk) d.seqlen - 1

/-- The request running at the end of chunk `c` (whose state `out[0, c + 1]` holds). -/
def eReq (d : Inst) (c : Nat) : Nat := reqOf d (endTok d c)

/-- Chunk `c` starts that request: its first token `qsl (eReq c)` lies in chunk `c`, so its state
starts from its initial state there. -/
def isReset (d : Inst) (c : Nat) : Bool := decide (c * d.chunk ≤ d.qsl (eReq d c))

/-- The offset in chunk `c` of that request's first token (on a reset chunk). -/
def offR (d : Inst) (c : Nat) : Nat := d.qsl (eReq d c) - c * d.chunk

/-- On a reset chunk whose request starts after the chunk's first token, the decay counts only the
request's own tokens: `dA_cumsum[…, chunk_size − 1] − dA_cumsum[…, off − 1]`. -/
def hasSub (d : Inst) (c : Nat) : Bool := isReset d c && decide (0 < offR d c)

/-- `initial_states[r, h, m]`. -/
noncomputable def initV (m₀ : Mem) (d : Inst) (r h m : Nat) : ℝ :=
  (valRat m₀ d.initst (inIdx d r h m) : ℝ)

/-- `states[0, c, h, m]`, chunk `c`'s own contribution. -/
noncomputable def newV (m₀ : Mem) (d : Inst) (c h m : Nat) : ℝ :=
  (valRat m₀ d.states (stIdx d c h m) : ℝ)

/-- The decay exponent of chunk `c` for head `h`: the chunk's cumulative `dA` at its end (the
chunk's total, `dA_cumsum`'s meaning), less the part before the request's start on a reset chunk
with a positive offset. -/
noncomputable def dRef (m₀ : Mem) (d : Inst) (h c : Nat) : ℝ :=
  (valRat m₀ d.dA (dAIdx d h c (d.chunk - 1)) : ℝ) -
    (if hasSub d c then (valRat m₀ d.dA (dAIdx d h c (offR d c - 1)) : ℝ) else 0)

/-- The state request `r` starts from: `initial_states[r, h, m]`, or `0` without initial states. -/
noncomputable def startV (m₀ : Mem) (d : Inst) (r h m : Nat) : ℝ :=
  if d.hasInit then initV m₀ d r h m else 0

/-- **The exact recurrence** for head `h`, element `m`: the state entering chunk `c` of the
request running there. `sRef 0 = startV 0` (request `0` starts at token `0`); then
`sRef (c + 1) = exp(dRef c) · xRef c + states[0, c]`, where `xRef c` is the state the chunk
continues: the start state of `eReq c` on a reset chunk (`0` without initial states), `sRef c`
otherwise. -/
noncomputable def sRef (m₀ : Mem) (d : Inst) (h m : Nat) : Nat → ℝ
  | 0 => startV m₀ d 0 h m
  | c + 1 =>
      Real.exp (dRef m₀ d h c) *
        (if isReset d c then startV m₀ d (eReq d c) h m else sRef m₀ d h m c) +
        newV m₀ d c h m

/-- The state chunk `c` continues. -/
noncomputable def xRef (m₀ : Mem) (d : Inst) (h m c : Nat) : ℝ :=
  if isReset d c then startV m₀ d (eReq d c) h m else sRef m₀ d h m c

/-! ### The tolerance

From `Device.ExpF32` at `X = |dRef| + δ` (relative `2^-22 + 2^-23·X`, absolute `2^-140`); the
IEEE subtraction of line 215 (`2^-24·|d| + 2^-150`); and `tl.mul_add` (an IEEE product then sum, or
one `fma`: at most `(2^-23 + 2^-47)·(|p| + |n|) + 3·2^-150` for the product `p` and the addend
`n`). README, Tolerances. -/

/-- The error of the computed decay exponent (line 215's rounding, when it runs). -/
noncomputable def dErr (m₀ : Mem) (d : Inst) (h c : Nat) : ℝ :=
  if hasSub d c then |dRef m₀ d h c| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) else 0

/-- `Device.ExpF32`'s relative constant at `X = |dRef| + dErr`, which bounds the computed input. -/
noncomputable def expRel (m₀ : Mem) (d : Inst) (h c : Nat) : ℝ :=
  2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|dRef m₀ d h c| + dErr m₀ d h c)

/-- The error of the computed `scale = tl.exp(dA_cs)` against `exp(dRef c)`:
`(1 + rel)·exp(dRef + δ) − exp(dRef) + 2^-140`. -/
noncomputable def errE (m₀ : Mem) (d : Inst) (h c : Nat) : ℝ :=
  (1 + expRel m₀ d h c) * Real.exp (dRef m₀ d h c + dErr m₀ d h c) - Real.exp (dRef m₀ d h c) +
    2 ^ (-140 : ℤ)

/-- A bound on the computed scale's magnitude. -/
noncomputable def eHat (m₀ : Mem) (d : Inst) (h c : Nat) : ℝ :=
  Real.exp (dRef m₀ d h c) + errE m₀ d h c

/-- `tl.mul_add`'s relative constant (two roundings, or one). -/
noncomputable def rho : ℝ := 2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ)

/-- **The tolerance of the state entering chunk `c`** (per head and element): `tol 0 = 0` (an exact
copy, or the exact `0`); a reset chunk starts from an exact initial state (or `0`); otherwise the carried error is scaled by
the computed decay. Each chunk adds the decay's error on the state it continues and `tl.mul_add`'s
roundings. -/
noncomputable def tol (m₀ : Mem) (d : Inst) (h m : Nat) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      let tc : ℝ := if isReset d c then 0 else tol m₀ d h m c
      eHat m₀ d h c * tc + errE m₀ d h c * |xRef m₀ d h m c| +
        rho * (eHat m₀ d h c * (|xRef m₀ d h m c| + tc) + |newV m₀ d c h m|) +
        3 * 2 ^ (-150 : ℤ)

/-- The carried error entering chunk `c`'s update: `0` on a reset chunk. -/
noncomputable def tolIn (m₀ : Mem) (d : Inst) (h m c : Nat) : ℝ :=
  if isReset d c then 0 else tol m₀ d h m c

/-- The inputs' values covered: every `states` entry and (with initial states) every
`initial_states` entry is a finite `float32`;
every `dA_cumsum` entry is finite and at most `0` (real use: the cumulative sums of `A·dt` with
`A < 0`, `dt ≥ 0`); and, a limit of the kernel's `float32` arithmetic, no chunk's decay exponent
exceeds `64` and every chunk's update stays below `2^126` (README, Exclusions). -/
def Vals (m₀ : Mem) (d : Inst) : Prop :=
  (∀ c h m, c < d.nchunks → h < d.nheads → m < d.dim →
    ∃ v, m₀.read .f32 (d.states.ptr (stIdx d c h m : Int)).addr = some v ∧ v.isFinite = true) ∧
  (d.hasInit = true → ∀ r h m, r < d.nreq → h < d.nheads → m < d.dim →
    ∃ v, m₀.read .f32 (d.initst.ptr (inIdx d r h m : Int)).addr = some v ∧ v.isFinite = true) ∧
  (∀ h c p, h < d.nheads → c < d.nchunks → p < d.chunk →
    ∃ v, m₀.read .f32 (d.dA.ptr (dAIdx d h c p : Int)).addr = some v ∧ v.isFinite = true ∧
      v.toRat ≤ 0) ∧
  (∀ h m c, h < d.nheads → m < d.dim → c < d.nchunks →
    dRef m₀ d h c ≤ 64 ∧
    eHat m₀ d h c * (|xRef m₀ d h m c| + tolIn m₀ d h m c) + |newV m₀ d c h m| ≤ 2 ^ 126)

/-- What `Post` asks of the state entering chunk `c`, held as `v`: when the values are covered,
`v` is finite and within `tol c` of `sRef c`. -/
def OutO (m₀ : Mem) (d : Inst) (h m c : Nat) (v : F32) : Prop :=
  Vals m₀ d → v.isFinite = true ∧ |v.toReal - sRef m₀ d h m c| ≤ tol m₀ d h m c

/-- What the launch leaves:
* `out[0, 0, h, m]` is `initial_states[0, h, m]` (up to NaN payload), for every input; without
  initial states it is the `float32` `+0`;
* for `1 ≤ c < nchunks`, `out[0, c, h, m]` holds a `float32` value, finite and within `tol c` of
  the exact recurrence `sRef c` when the values are covered (`Vals`);
* `final_states[0, h, m]` the same at `c = nchunks`;
* every byte outside `out`'s and `final_states`' elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ h m, h < d.nheads → m < d.dim →
      (∃ v, fin.read .f32 (d.out.ptr (stIdx d 0 h m : Int)).addr = some v ∧
        if d.hasInit then
          ∃ a, init.read .f32 (d.initst.ptr (inIdx d 0 h m : Int)).addr = some a ∧ FloatV.NaNEq a v
        else v = FloatV.ofPat 0) ∧
      (∀ c, 1 ≤ c → c < d.nchunks → ∃ v,
        fin.read .f32 (d.out.ptr (stIdx d c h m : Int)).addr = some v ∧ OutO init d h m c v)) ∧
  (∀ h m, h < d.nheads → m < d.dim → ∃ v,
      fin.read .f32 (d.finalst.ptr (finIdx d h m : Int)).addr = some v ∧
        OutO init d h m d.nchunks v) ∧
  ∀ z, z ∉ d.out.elems {j | j < d.nchunks * (d.nheads * d.dim)} ∪
      d.finalst.elems {j | j < d.nheads * d.dim} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_state_passing_fwd_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous, and the coverage of the source's tests -/

/-- `query_start_loc` from the request lengths: `qsl r = Σ_{i<r} lens[i]`. -/
def qslOf (lens : List Nat) (r : Nat) : Nat := (lens.take r).sum

/-- An instance of the host's launch from `HAS_INITSTATES`, the request lengths, the heads, `dim`
and `chunk_size`: the shapes the host derives, the metadata length `hostN` (`0` without initial
states), and the seven tensors at the 16-byte-aligned addresses `k·2^34` (`k = 1 … 7`), each exactly
as long as its tensor (`initial_states` and `chunk_offsets` empty without initial states). -/
def mk (hasInit : Bool) (lens : List Nat) (H D C : Nat) : Inst :=
  let L := lens.sum
  let NC := (L + C - 1) / C
  let base : Inst :=
    { hasInit := hasInit, nheads := H, dim := D, nchunks := NC, seqlen := L, chunk := C,
      nreq := lens.length, qsl := qslOf lens, nmeta := 0,
      states := ⟨1 * 17179869184, NC * (H * D)⟩, dA := ⟨2 * 17179869184, H * (NC * C)⟩,
      initst := ⟨3 * 17179869184, if hasInit then lens.length * (H * D) else 0⟩,
      seqIdx := ⟨4 * 17179869184, L⟩,
      chunkOffs := ⟨5 * 17179869184, 0⟩, out := ⟨6 * 17179869184, NC * (H * D)⟩,
      finalst := ⟨7 * 17179869184, H * D⟩ }
  let n := if hasInit then hostN base else 0
  { base with nmeta := n, chunkOffs := ⟨5 * 17179869184, n⟩ }

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
    (hb : b.base = i * 17179869184) (hc : c.base = j * 17179869184)
    (hbl : b.len * d1.bytes ≤ 17179869184) (hcl : c.len * d2.bytes ≤ 17179869184) (hij : i ≠ j) :
    Disjoint b.bytes c.bytes := by
  apply bytes_disj
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · left
    have : i * 17179869184 + 17179869184 ≤ j * 17179869184 := by
      have := Nat.mul_le_mul_right 17179869184 (Nat.succ_le_of_lt h); simpa [Nat.succ_mul] using this
    rw [hb, hc]; omega
  · right
    have : j * 17179869184 + 17179869184 ≤ i * 17179869184 := by
      have := Nat.mul_le_mul_right 17179869184 (Nat.succ_le_of_lt h); simpa [Nat.succ_mul] using this
    rw [hb, hc]; omega

/-- The instances `mk` builds are covered when their sizes are (each condition decided; at most
`2^22` tokens and `8192` requests). -/
theorem mk_ok (hi : Bool) (lens : List Nat) (H D C k : Nat) (hC : C = 2 ^ k)
    (h1 : 0 < H ∧ H < 2 ^ 16 ∧ 0 < D ∧ D + 1023 < 2 ^ 31 ∧ gx (mk hi lens H D C) * H < 2 ^ 31)
    (h2 : 0 < lens.length ∧ ∀ r, r < lens.length → qslOf lens r < qslOf lens (r + 1))
    (h3 : (mk hi lens H D C).nmeta < 2 ^ 31 ∧ (mk hi lens H D C).nchunks * C < 2 ^ 31 ∧
      H * ((mk hi lens H D C).nchunks * C) < 2 ^ 31 ∧
      (mk hi lens H D C).nchunks * (H * D) < 2 ^ 31 ∧
      (hi = true → (lens.length - 1) * (H * D) < 2 ^ 31))
    (h4 : (mk hi lens H D C).states.len * 4 ≤ 17179869184 ∧
      (mk hi lens H D C).dA.len * 4 ≤ 17179869184 ∧
      (mk hi lens H D C).initst.len * 4 ≤ 17179869184 ∧
      (mk hi lens H D C).seqIdx.len * 4 ≤ 17179869184 ∧
      (mk hi lens H D C).chunkOffs.len * 4 ≤ 17179869184 ∧
      (mk hi lens H D C).out.len * 4 ≤ 17179869184 ∧
      (mk hi lens H D C).finalst.len * 4 ≤ 17179869184) (h5 : lens.sum ≤ 2 ^ 22)
    (h6 : lens.length ≤ 8192) :
    (mk hi lens H D C).ok := by
  obtain ⟨h1a, h1b, h1c, h1d, h1e⟩ := h1
  obtain ⟨h2a, h2b⟩ := h2
  obtain ⟨h3a, h3b, h3c, h3d, h3e⟩ := h3
  obtain ⟨h4a, h4b, h4c, h4d, h4e, h4f, h4g⟩ := h4
  refine
    { heads_pos := h1a, heads_lt := h1b, dim_pos := h1c, dim_lt := h1d, grid_lt := h1e,
      chunk_pow2 := ⟨k, hC⟩, nreq_pos := h2a, qsl_zero := by simp [mk, qslOf],
      qsl_lt := h2b, qsl_end := by simp [mk, qslOf], seqlen_ctx := h5, nreq_le := h6, nchunks_eq := rfl,
      nmeta_eq := by cases hi <;> rfl,
      nmeta_lt := h3a, tok_lt := h3b, dA_lt := h3c, st_lt := h3d, init_lt := h3e,
      states_len := le_refl _, dA_len := le_refl _,
      init_len := fun h => by subst h; exact le_refl _, seq_len := le_refl _,
      offs_len := le_refl _, out_len := le_refl _, fin_len := le_refl _, disj := ?_ }
  exact ⟨disj_far _ _ 6 1 rfl rfl h4f h4a (by decide), disj_far _ _ 6 2 rfl rfl h4f h4b (by decide),
    disj_far _ _ 6 3 rfl rfl h4f h4c (by decide), disj_far _ _ 6 4 rfl rfl h4f h4d (by decide),
    disj_far _ _ 6 5 rfl rfl h4f h4e (by decide), disj_far _ _ 6 7 rfl rfl h4f h4g (by decide),
    disj_far _ _ 7 1 rfl rfl h4g h4a (by decide), disj_far _ _ 7 2 rfl rfl h4g h4b (by decide),
    disj_far _ _ 7 3 rfl rfl h4g h4c (by decide), disj_far _ _ 7 4 rfl rfl h4g h4d (by decide),
    disj_far _ _ 7 5 rfl rfl h4g h4e (by decide)⟩

/-- The source's test cases (lines 432–441): (request lengths, `nheads`, `dim`, `chunk_size`),
each with initial states (`setup_1 … setup_9`) and without (`setup_10 … setup_18`). -/
def setup_1 : Inst := mk true [1] 8 8192 256
def setup_2 : Inst := mk true [256] 8 8192 256
def setup_3 : Inst := mk true [16, 20] 4 512 8
def setup_4 : Inst := mk true [270, 88, 212, 203] 4 80 8
def setup_5 : Inst := mk true [270, 88, 212, 203] 16 8192 256
def setup_6 : Inst := mk true [5, 1, 1, 1, 300, 7, 128, 129] 16 8192 128
def setup_7 : Inst := mk true [1000, 1, 2049, 64, 64, 300] 32 10240 128
def setup_8 : Inst := mk true [4096, 4096] 128 8192 256
def setup_9 : Inst := mk true (List.replicate 40 3 ++ [500]) 8 8192 64
def setup_10 : Inst := mk false [1] 8 8192 256
def setup_11 : Inst := mk false [256] 8 8192 256
def setup_12 : Inst := mk false [16, 20] 4 512 8
def setup_13 : Inst := mk false [270, 88, 212, 203] 4 80 8
def setup_14 : Inst := mk false [270, 88, 212, 203] 16 8192 256
def setup_15 : Inst := mk false [5, 1, 1, 1, 300, 7, 128, 129] 16 8192 128
def setup_16 : Inst := mk false [1000, 1, 2049, 64, 64, 300] 32 10240 128
def setup_17 : Inst := mk false [4096, 4096] 128 8192 256
def setup_18 : Inst := mk false (List.replicate 40 3 ++ [500]) 8 8192 64

theorem setup_1_ok : setup_1.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_2_ok : setup_2.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_3_ok : setup_3.ok :=
  mk_ok _ _ _ _ _ 3 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_4_ok : setup_4.ok :=
  mk_ok _ _ _ _ _ 3 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_5_ok : setup_5.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_6_ok : setup_6.ok :=
  mk_ok _ _ _ _ _ 7 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_7_ok : setup_7.ok :=
  mk_ok _ _ _ _ _ 7 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_8_ok : setup_8.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_9_ok : setup_9.ok :=
  mk_ok _ _ _ _ _ 6 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_10_ok : setup_10.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_11_ok : setup_11.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_12_ok : setup_12.ok :=
  mk_ok _ _ _ _ _ 3 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_13_ok : setup_13.ok :=
  mk_ok _ _ _ _ _ 3 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_14_ok : setup_14.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_15_ok : setup_15.ok :=
  mk_ok _ _ _ _ _ 7 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_16_ok : setup_16.ok :=
  mk_ok _ _ _ _ _ 7 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_17_ok : setup_17.ok :=
  mk_ok _ _ _ _ _ 8 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem setup_18_ok : setup_18.ok :=
  mk_ok _ _ _ _ _ 6 rfl (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)

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

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_1, zeroMem, setup_1_ok, ?_⟩
  refine ⟨valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), ?_, ?_⟩
  · intro t ht
    have ht0 : t = 0 := by
      have : setup_1.seqlen = 1 := by decide
      omega
    subst ht0
    rw [show reqOf setup_1 0 = 0 by decide]
    exact zero_i32 (by decide)
  · intro k hk
    have hk0 : k = 0 := by
      have : setup_1.nmeta = 1 := by decide
      omega
    subst hk0
    rw [show hostOff setup_1 0 = 0 by decide]
    exact zero_i32 (by decide)

end Claims.SsdStatePassing
