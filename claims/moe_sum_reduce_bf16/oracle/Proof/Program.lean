import Proof.Addr
import Proof.Value

/-!
# The program of one block

The kernel's body, with the loop body factored out (`stepBody`) so that the accumulator's step can
be reasoned about on its own. `body_eq` says that the model's program is exactly this one: the
factorisation is definitional.
-/

namespace Claims.MoeSumReduce

open TVA

/-- One iteration of the accumulation loop: load the `k`-th plane of the tile, widen it, add it. -/
def stepBody (d : Inst) (base_ptrs : Tensor [d.BLOCK_M, d.BLOCK_DIM] (Ptr .bf16))
    (s1 : I64) (mt : Tensor [d.BLOCK_M] Bool) (md : Tensor [d.BLOCK_DIM] Bool)
    (i : I32) (accumulator : Tensor [d.BLOCK_M, d.BLOCK_DIM] F32) :
    Prog (Tensor [d.BLOCK_M, d.BLOCK_DIM] F32) := do
  let i64 := (← tl.convert .i32 .i64 (Tensor.scalar i)).item
  let tile ← tl.load (base_ptrs + (bc (i64 * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64))
    (some ((mt.col d.BLOCK_DIM).and (md.row d.BLOCK_M)))
    (some (← tl.convert .f32 .bf16 (tl.full [d.BLOCK_M, d.BLOCK_DIM] (FloatV.litF32 (FloatV.ofPat 0)))))
  let tile32 : Tensor [d.BLOCK_M, d.BLOCK_DIM] F32 ← tl.cvt tile
  let acc ← tl.add accumulator tile32
  return acc

/-- The program of a block, with its own context (its two program ids). -/
def progBody (d : Inst) (c : Ctx) : Prog Unit := do
  let s0 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat (d.K * d.D) : I32))).item
  let s1 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.D : I32))).item
  let os0 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.D : I32))).item
  let token_block_id := tl.program_id c 0
  let dim_block_id := tl.program_id c 1
  let offs_token := token_block_id * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M
  let offs_dim := dim_block_id * (IntV.ofNat d.BLOCK_DIM : I32) + tl.arange 0 d.BLOCK_DIM
  let mask_token := offs_token.ltS (IntV.ofNat d.T : I32)
  let mask_dim := offs_dim.ltS (IntV.ofNat d.D : I32)
  let offs0 : Tensor [d.BLOCK_M] I64 ← tl.convert .i32 .i64 offs_token
  let offs1 : Tensor [d.BLOCK_DIM] I64 ← tl.convert .i32 .i64 offs_dim
  let base_ptrs := d.input.ptr 0 + (offs0.col d.BLOCK_DIM * s0 + offs1.row d.BLOCK_M)
  let accumulator := tl.zeros [d.BLOCK_M, d.BLOCK_DIM] .f32
  let accumulator ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.K : I32) (IntV.ofInt 1 : I32)
    accumulator (stepBody d base_ptrs s1 mask_token mask_dim)
  let acc2 ← tl.mul accumulator (bc (sigma32 d))
  let out : Tensor [d.BLOCK_M, d.BLOCK_DIM] BF16 ← tl.cvt acc2
  let store_ptrs := d.output.ptr 0 + (offs0.col d.BLOCK_DIM * os0 + offs1.row d.BLOCK_M)
  tl.store store_ptrs out (some ((mask_token.col d.BLOCK_DIM).and (mask_dim.row d.BLOCK_M)))

/-- The model's program is `progBody`: only the loop body is factored out. -/
theorem body_eq (d : Inst) (c : Ctx) : (model d).body c = progBody d c := rfl

/-- The accumulation loop is `K` trips of `stepBody` at the constant index `t`. -/
theorem forRange_K (d : Inst) (hd : d.ok) {σ : Type} (acc : σ) (body : I32 → σ → Prog σ) :
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.K : I32) (IntV.ofInt 1 : I32) acc body
      = tl.forN d.K (fun t => body (IntV.ofInt (t : Int))) acc := by
  have hK31 : ((d.K : Int)) < (2 : Int) ^ 31 := by exact_mod_cast hd.K_lt
  have hK : (IntV.ofNat d.K : I32).toInt = (d.K : Int) :=
    IntV.I32.toInt_ofNat (by norm_num; omega)
  have h0 : (IntV.ofInt (0 : Int) : I32).toInt = 0 :=
    IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; norm_num)
  have h1 : (IntV.ofInt (1 : Int) : I32).toInt = 1 :=
    IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; norm_num)
  have hrok : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat d.K : I32) (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [hK, h0, h1]
    unfold IntV.maxInt
    norm_num
    omega
  rw [tl.forRange, ite_eq_left hrok]
  have htc : tl.tripCount ((IntV.ofInt (0 : Int) : I32).toInt) ((IntV.ofNat d.K : I32).toInt)
      ((IntV.ofInt (1 : Int) : I32).toInt) = d.K := by
    unfold tl.tripCount
    rw [hK, h0, h1]
    norm_num
  rw [htc]
  apply congrArg (fun g => tl.forN d.K g acc)
  funext t
  congr 1
  rw [h0, h1]
  congr 1
  omega

end Claims.MoeSumReduce
