import Proof.Basic

/-!
# `token_bitmask_f32`: the program with its loop body named

The loop body as `tripProg` (the row, the block offset, the index) and `tailProg` (the block), with
the two `tl.arange` blocks at the shapes `[4096]` and `[128]`; `body_eq` (by `rfl`) is the form the
proof walks.
-/

namespace Claims.TokenBitmask

open TVA

/-- `tl.arange(0, 4096)`, at the shape `[4096]`. -/
def ar4096 : Tensor [4096] I32 := tl.arange 0 4096

/-- `tl.arange(0, 128)`, at the shape `[128]`. -/
def ar128 : Tensor [128] I32 := tl.arange 0 128

/-- The loop body after `block_offset` and `batch_id` (source lines 109-124). -/
def tailProg (d : Inst) (block_offset batch_id : I32) : Prog Unit := do
  let offsets := block_offset + ar4096
  let bitmask_offsets :=
    (← tl.idiv (Tensor.scalar block_offset) (Tensor.scalar (IntV.ofNat 32 : I32))).item +
      ar128
  let vocab_mask := offsets.ltS (IntV.ofNat d.vocab_size : I32)
  let packed_bitmask_mask := bitmask_offsets.ltS (IntV.ofNat d.bitmask_strides : I32)
  let packed_bitmask : Tensor [128] I32 ←
    tl.load (d.bitmask.ptr 0 + batch_id * (IntV.ofNat d.bitmask_strides : I32) + bitmask_offsets)
      (some packed_bitmask_mask)
  let shifted : Tensor [128, 32] I32 ←
    tl.ashr (packed_bitmask.col 32) ((tl.arange 0 32 : Tensor [32] I32).row 128)
  let bitmask : Tensor [128, 32] Bool :=
    (Tensor.map2 IntV.and shifted (bc (IntV.ofInt 1 : I32))).eq (bc (IntV.ofInt 0 : I32))
  let bitmask : Tensor [4096] Bool := bitmask.reshape [4096] (by decide)
  tl.store (d.logits.ptr 0 + batch_id * (IntV.ofNat d.logits_strides : I32) + offsets)
    (tl.full [4096] negInfC) (some (vocab_mask.and bitmask))

/-- The loop body (source lines 106-124) for the work item `work_id`. -/
def tripProg (d : Inst) (num_blocks work_id : I32) : Prog Unit := do
  let row_id := (← tl.idiv (Tensor.scalar work_id) (Tensor.scalar num_blocks)).item
  let block_offset :=
    (← tl.irem (Tensor.scalar work_id) (Tensor.scalar num_blocks)).item * (IntV.ofNat 4096 : I32)
  let batch_id : I32 ←
    match (if d.hasIdx then some (d.indices.ptr 0) else none) with
    | none => pure row_id
    | some indices_ptr => tl.load1 (indices_ptr + row_id)
  tailProg d block_offset batch_id

/-- The program, with the loop body named. -/
theorem body_eq (d : Inst) (c : Ctx) : (model d).body c =
    (tl.cdiv (Tensor.scalar (IntV.ofNat d.vocab_size : I32))
      (Tensor.scalar (IntV.ofNat 4096 : I32)) >>= fun nbT =>
      tl.forRange (tl.program_id c 0) ((IntV.ofNat d.num_rows : I32) * nbT.item)
        (IntV.ofNat d.NUM_SMS : I32) () fun w _ => tripProg d nbT.item w) := rfl

end Claims.TokenBitmask
