import VecAdd

/-! r-vecadd2, round 2: small checks behind the review.

1. The map note of line 44 says `IntV.ofNat B` is the constant `IntV.ofInt B` that SEMANTICS.md
   ("Constants") gives for a Python integer operand of `*` with an `int32` tensor.
2. The model's body is the library's 1-D block pattern (`Block.offs`, `Block.mask`), for every
   instance and program, by definition (the proof's `body_eq` states it by `rfl`).
3. At the extremes of `Inst.ok` (`n = 2^31 - 1`, `BLOCK_SIZE = 1` and `2^20`) the grid has fewer
   than `2^31` programs and the last program's largest offset is below `2^31`. -/

open TVA

-- 1. `IntV.ofNat` and `IntV.ofInt` agree on natural numbers (here for the block sizes covered).
example (B : Nat) : (IntV.ofNat B : I32) = IntV.ofInt (B : Int) := by
  simp [IntV.ofNat, IntV.ofInt]

-- 2. The model's offsets and mask are `Block.offs` and `Block.mask` of the program's id.
example (d : Kernels.VecAdd.Inst) (c : Ctx) :
    (Kernels.VecAdd.model d).body c =
      (do
        let x ← tl.load (d.x.ptr 0 + Block.offs (c.pid 0) d.BLOCK_SIZE)
          (some (Block.mask (c.pid 0) d.BLOCK_SIZE d.n))
        let y ← tl.load (d.y.ptr 0 + Block.offs (c.pid 0) d.BLOCK_SIZE)
          (some (Block.mask (c.pid 0) d.BLOCK_SIZE d.n))
        let output ← tl.add x y
        tl.store (d.output.ptr 0 + Block.offs (c.pid 0) d.BLOCK_SIZE) output
          (some (Block.mask (c.pid 0) d.BLOCK_SIZE d.n))) := rfl

-- 3. The extremes: grid size and the last program's largest offset `p * B + B - 1`.
def gridOf (n B : Nat) : Nat := (n + B - 1) / B
#eval (gridOf (2^31 - 1) 1, decide (gridOf (2^31 - 1) 1 < 2^31))
#eval (gridOf (2^31 - 1) (2^20), decide ((gridOf (2^31 - 1) (2^20) - 1) * 2^20 + 2^20 - 1 < 2^31))
#eval ((gridOf (2^31 - 1) 1 - 1) * 1 + 1 - 1 < 2^31)
-- The host's own sizes, with `BLOCK_SIZE = 1024`.
#eval (gridOf 98432 1024, (List.range 16).map (fun i => gridOf (2^(12 + i)) 1024))
