import Claim
import TVA.Proof
import Proof.Meta

/-!
# The chunk loop's trip, as a program

The body of the loop `tl.forRange` of `_state_passing_fwd_kernel`, written out as a function of the
loop state, so that a lemma can speak of one trip: `tripI` is the launch with initial states
(source lines 163-231), `tripN` the launch without. Each ends in `restI` / `restN`, the lines after
the choice of the state and the decay. `Proof.Program` shows they are the model's loop bodies.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The loop state with initial states: `states`, `initstates_ptrs`, `states_ptrs`, `dA_cs_ptr`,
`out_ptrs`, `prev_seq_idx_chunk_end`, `logical_chunk_idx`. -/
abbrev StI (B : Nat) : Type :=
  Tensor [B] F32 × Tensor [B - 0] (Ptr .f32) × Tensor [B - 0] (Ptr .f32) × Ptr .f32 ×
    Tensor [B - 0] (Ptr .f32) × I32 × I32

/-- The loop state without initial states: `states`, `states_ptrs`, `dA_cs_ptr`, `out_ptrs`,
`prev_seq_idx_chunk_end`. -/
abbrev StN (B : Nat) : Type :=
  Tensor [B] F32 × Tensor [B - 0] (Ptr .f32) × Ptr .f32 × Tensor [B - 0] (Ptr .f32) × I32

/-- The lines of a trip with initial states after the state and the decay are chosen
(source lines 218-231). -/
def restI (nchunks dim stride_states_chunk stride_out_chunk stride_dA_cs_chunk : I32)
    (BLOCK_SIZE : Nat) (offs_m : Tensor [BLOCK_SIZE - 0] I32)
    (final_states_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) (c : I32) (scale_mask : Tensor [] Bool)
    (new_states : Tensor [BLOCK_SIZE] F32) (seq_idx_chunk_end : I32) (states : Tensor [BLOCK_SIZE] F32)
    (initstates_ptrs states_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) (dA_cs_ptr : Ptr .f32)
    (out_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) (logical_chunk_idx : I32) (dA_cs : F32) :
    Prog (StI BLOCK_SIZE) := do
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

/-- **One trip with initial states** (source lines 163-231). -/
def tripI (chunk_meta_num dim nchunks seqlen chunk_size stride_states_chunk stride_out_chunk
    stride_dA_cs_chunk stride_dA_cs_csize stride_initstates_batch stride_seq_idx_seqlen : I32)
    (BLOCK_SIZE : Nat) (offs_m : Tensor [BLOCK_SIZE - 0] I32) (seq_idx_ptr : Ptr .i32)
    (chunk_offsets_ptr : Ptr .i32)
    (initstates_ptr final_states_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) :
    I32 → StI BLOCK_SIZE → Prog (StI BLOCK_SIZE) :=
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
      restI nchunks dim stride_states_chunk stride_out_chunk stride_dA_cs_chunk BLOCK_SIZE offs_m
        final_states_ptrs c scale_mask new_states seq_idx_chunk_end states initstates_ptrs
        states_ptrs dA_cs_ptr out_ptrs logical_chunk_idx dA_cs

/-- The lines of a trip without initial states after the request at the chunk's end is known
(source lines 221-231). -/
def restN (nchunks dim stride_states_chunk stride_out_chunk stride_dA_cs_chunk : I32)
    (BLOCK_SIZE : Nat) (offs_m : Tensor [BLOCK_SIZE - 0] I32)
    (final_states_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) (c : I32) (scale_mask : Tensor [] Bool)
    (new_states : Tensor [BLOCK_SIZE] F32) (seq_idx_chunk_end : I32) (states : Tensor [BLOCK_SIZE] F32)
    (states_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) (dA_cs_ptr : Ptr .f32)
    (out_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) (dA_cs : F32) : Prog (StN BLOCK_SIZE) := do
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

/-- **One trip without initial states** (source lines 163-231). -/
def tripN (dim nchunks seqlen chunk_size stride_states_chunk stride_out_chunk stride_dA_cs_chunk
    stride_seq_idx_seqlen : I32)
    (BLOCK_SIZE : Nat) (offs_m : Tensor [BLOCK_SIZE - 0] I32) (seq_idx_ptr : Ptr .i32)
    (final_states_ptrs : Tensor [BLOCK_SIZE - 0] (Ptr .f32)) :
    I32 → StN BLOCK_SIZE → Prog (StN BLOCK_SIZE) :=
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
      restN nchunks dim stride_states_chunk stride_out_chunk stride_dA_cs_chunk BLOCK_SIZE offs_m
        final_states_ptrs c scale_mask new_states seq_idx_chunk_end states states_ptrs dA_cs_ptr
        out_ptrs dA_cs

end Claims.SsdStatePassing
