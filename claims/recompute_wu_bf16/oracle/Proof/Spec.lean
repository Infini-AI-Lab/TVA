import Claim
import TVA.Proof
import Proof.Geo
import Proof.Cell

/-!
# What each loaded or computed tile holds

The lane predicates of the walk: what `b_A`, `b_b`, `b_v`, `b_vb`, `l_g`, `b_g`, `b_k`, `b_kbh`
hold, lane by lane, for the chunk `it` of the sequence at `s0` (length `len`), head `h`. A lane
of a masked row holds `0`; a lane of an enabled row holds the input's element (or the operand
computed from it). Also the facts `Pre` gives about the inputs' elements.
-/

namespace Claims.RecomputeWU

open TVA

/-! ## Input elements -/

theorem bfV_of {b : Buf .bf16} {m : Mem} {j : Nat} {x : BF16} (h : bfAt b m j = some x) :
    bfV b m j = x := by
  unfold bfV; rw [h]; rfl

/-- A present `float32` element is `f32V`. -/
theorem f32V_of {b : Buf .f32} {m : Mem} {j : Nat} {x : F32} (h : f32At b m j = some x) :
    f32V b m j = x := by
  unfold f32V; rw [h]; rfl

/-- A covered `k` element is finite and at most `2^32`. -/
theorem k_ok {d : Inst} {init : Mem} (hpre : Pre d init) {f h c : Nat} (hf : f < nTok d)
    (hh : h < d.H) (hc : c < d.K) : KVOk (bfV d.k init (kIdx d f h c)) := by
  obtain ⟨x, hx, hok⟩ := hpre.k_vals f h c hf hh hc
  rw [bfV_of hx]; exact hok

/-- A covered `v` element is finite and at most `2^32`. -/
theorem v_ok {d : Inst} {init : Mem} (hpre : Pre d init) {f h c : Nat} (hf : f < nTok d)
    (hh : h < d.HV) (hc : c < d.V) : KVOk (bfV d.v init (vIdx d f h c)) := by
  obtain ⟨x, hx, hok⟩ := hpre.v_vals f h c hf hh hc
  rw [bfV_of hx]; exact hok

/-- A covered `A` element is finite and at most `2^64`. -/
theorem A_ok {d : Inst} {init : Mem} (hpre : Pre d init) {f h j : Nat} (hf : f < nTok d)
    (hh : h < d.HV) (hj : j < 64) : AOk (bfV d.A init (aIdx d f h j)) := by
  obtain ⟨x, hx, hok⟩ := hpre.A_vals f h j hf hh hj
  rw [bfV_of hx]; exact hok

/-- A covered `float32` `beta` element is finite and at most `2^8`. -/
theorem beta_ok {d : Inst} {init : Mem} (hpre : Pre d init) (hb : d.bbf = false) {f h : Nat}
    (hf : f < nTok d) (hh : h < d.HV) : BetaOk (f32V d.beta init (hIdx d f h)) := by
  obtain ⟨x, hx, hok⟩ := hpre.beta_vals hb f h hf hh
  rw [f32V_of hx]; exact hok

/-- A covered `bfloat16` `beta` element is finite and at most `2^8`. -/
theorem betab_ok {d : Inst} {init : Mem} (hpre : Pre d init) (hb : d.bbf = true) {f h : Nat}
    (hf : f < nTok d) (hh : h < d.HV) : BetaOkB (bfV d.betab init (hIdx d f h)) := by
  obtain ⟨x, hx, hok⟩ := hpre.betab_vals hb f h hf hh
  rw [bfV_of hx]; exact hok

/-- A covered `g` element is finite and at most `8`. -/
theorem g_ok {d : Inst} {init : Mem} (hpre : Pre d init) {f h : Nat} (hf : f < nTok d)
    (hh : h < d.HV) : GOk (f32V d.g init (hIdx d f h)) := by
  obtain ⟨x, hx, hok⟩ := hpre.g_vals f h hf hh
  rw [f32V_of hx]; exact hok

/-! ## The lanes -/

/-- The flat token of row `r` of the chunk. -/
def tokOf (s0 it r : Nat) : Nat := s0 + it * 64 + r

/-- `b_A`: row `r` is row `s0 + it·64 + r` of `A` when enabled, `0` otherwise. -/
def ALane (d : Inst) (init : Mem) (s0 len it h : Nat) (b_A : Tensor [64, 64] BF16) : Prop :=
  ∀ r j : Fin 64,
    (rowEn it len r = true → b_A.get2 r j = bfV d.A init (aIdx d (tokOf s0 it r) h j)) ∧
    (rowEn it len r = false → b_A.get2 r j = FloatV.ofPat 0)

/-- `b_b` for a `float32` `beta`. -/
def BLane32 (d : Inst) (init : Mem) (s0 len it h : Nat) (b_b : Tensor [64] F32) : Prop :=
  ∀ r : Fin 64,
    (rowEn it len r = true → b_b.get1 r = f32V d.beta init (hIdx d (tokOf s0 it r) h)) ∧
    (rowEn it len r = false → b_b.get1 r = FloatV.ofPat 0)

/-- `b_b` for a `bfloat16` `beta`. -/
def BLane16 (d : Inst) (init : Mem) (s0 len it h : Nat) (b_b : Tensor [64] BF16) : Prop :=
  ∀ r : Fin 64,
    (rowEn it len r = true → b_b.get1 r = bfV d.betab init (hIdx d (tokOf s0 it r) h)) ∧
    (rowEn it len r = false → b_b.get1 r = FloatV.ofPat 0)

/-- `b_v` of block `t`: the lanes of the block's columns below `V`. -/
def VLane (d : Inst) (init : Mem) (s0 len it h t : Nat) (b_v : Tensor [64, 64] BF16) : Prop :=
  ∀ r c : Fin 64, colEn t d.V c = true →
    (rowEn it len r = true →
      b_v.get2 r c = bfV d.v init (vIdx d (tokOf s0 it r) h (t * 64 + c))) ∧
    (rowEn it len r = false → b_v.get2 r c = FloatV.ofPat 0)

/-- `b_vb` of block `t`: finite and small, equal to `vbV` on enabled rows, `0` on masked ones. -/
def VbLane (d : Inst) (init : Mem) (s0 len it h t : Nat) (b_vb : Tensor [64, 64] BF16) : Prop :=
  ∀ r c : Fin 64, colEn t d.V c = true →
    (b_vb.get2 r c).isFinite = true ∧ |(b_vb.get2 r c).toRat| ≤ 2 ^ 41 ∧
    (rowEn it len r = true →
      b_vb.get2 r c = vbV d init (tokOf s0 it r) h (t * 64 + c)) ∧
    (rowEn it len r = false → (b_vb.get2 r c).toRat = 0)

/-- `l_g`: the loaded `g`. -/
def GLane (d : Inst) (init : Mem) (s0 len it h : Nat) (l_g : Tensor [64] F32) : Prop :=
  ∀ r : Fin 64,
    (rowEn it len r = true → l_g.get1 r = f32V d.g init (hIdx d (tokOf s0 it r) h)) ∧
    (rowEn it len r = false → l_g.get1 r = FloatV.ofPat 0)

/-- `b_g = exp2(l_g)`: finite, within the device's bound of `2^g`, with `g ≤ 8`. -/
def ELane (l_g b_g : Tensor [64] F32) : Prop :=
  ∀ r : Fin 64, (b_g.get1 r).isFinite = true ∧ (l_g.get1 r).toReal ≤ 8 ∧
    |(b_g.get1 r).toReal - 2 ^ (l_g.get1 r).toReal| ≤
      2 ^ (-22 : ℤ) * 2 ^ (l_g.get1 r).toReal + 2 ^ (-126 : ℤ)

/-- `b_k` of block `t`. -/
def KLane (d : Inst) (init : Mem) (s0 len it h t : Nat) (b_k : Tensor [64, 64] BF16) : Prop :=
  ∀ r c : Fin 64, colEn t d.K c = true →
    (rowEn it len r = true →
      b_k.get2 r c = bfV d.k init (kIdx d (tokOf s0 it r) (kHead d h) (t * 64 + c))) ∧
    (rowEn it len r = false → b_k.get2 r c = FloatV.ofPat 0)

/-- `b_kbh` of block `t`: finite; near `kbX · gE` on enabled rows, `0` on masked ones. -/
def KbLane (d : Inst) (init : Mem) (s0 len it h t : Nat) (b_kbh : Tensor [64, 64] BF16) : Prop :=
  ∀ r c : Fin 64, colEn t d.K c = true →
    (b_kbh.get2 r c).isFinite = true ∧
    (rowEn it len r = true →
      KbNear (b_kbh.get2 r c).toReal (kbX d init (tokOf s0 it r) h (t * 64 + c))
        (gE d init (tokOf s0 it r) h)) ∧
    (rowEn it len r = false → (b_kbh.get2 r c).toRat = 0)

end Claims.RecomputeWU
