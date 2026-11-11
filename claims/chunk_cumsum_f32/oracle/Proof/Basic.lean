import Claim
import TVA.Proof

/-!
# The model's body, unfolded, and the launch

`body_eq` writes the model's body out line by line; `launch_size`, `launch_pid` and `launch_valid`
say the launch is a valid one-dimensional grid of `gridSize d` programs whose program `i` has
`program_id(0) = i`.
-/

namespace Claims.ChunkCumsum

open TVA

/-- The model's body, unfolded. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let pid := tl.program_id c 0
        let pid := (← tl.convert .i32 .i64 (Tensor.scalar pid)).item
        let i_bh := (← tl.irem (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat (d.B * d.H) : I64))).item
        let i_t := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat (d.B * d.H) : I64))).item
        let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.H : I64))).item
        let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.H : I64))).item
        let (i_t, bos, T) ← if d.varlen then do
            let i_n := (← tl.convert d.idxD .i32 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64))))).item
            let i_t := (← tl.convert d.idxD .i64 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
            let bos := (← tl.convert d.idxD .i64 (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n)))).item
            let eos := (← tl.convert d.idxD .i64
              (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))))).item
            pure (i_t, bos, eos - bos)
          else do
            let T64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.T))).item
            pure (i_t, i_b * T64, T64)
        let o_t := bc (i_t * (IntV.ofNat d.BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BT))
        let m_t := o_t.ltS T
        let p_s := d.s.ptr 0 + bos * (IntV.ofNat d.H : I64) + i_h + o_t * (IntV.ofNat d.H : I64)
        let p_o := d.o.ptr 0 + bos * (IntV.ofNat d.H : I64) + i_h + o_t * (IntV.ofNat d.H : I64)
        let b_s ← tl.load p_s (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
        let b_o ← tl.cumsum b_s 0 d.reverse
        let b_o ← if d.hasScale then tl.mul b_o (bc (sigma32 d)) else pure b_o
        tl.store p_o b_o (some m_t)) := rfl

/-- The grid has `gridSize d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = gridSize d := Launch.size_grid1 _ _ _

/-- Program number `i`'s first grid coordinate. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < gridSize d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- `BT` is positive. -/
theorem bt_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  obtain ⟨e, he⟩ := hd.bt_pow2
  rw [he]
  exact Nat.two_pow_pos e

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact hd.bt_pow2
  · simpa using hd.bt_le

end Claims.ChunkCumsum
