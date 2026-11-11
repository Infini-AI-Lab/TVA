import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.NumEntry
import Proof.NumFinal
import Proof.NumDot
import Proof.VTrip
import Proof.KLoop

/-!
# The `V` loop

Lines 142–149: for each block of `BV` columns, load `do`, multiply by the rounded matrix `A` with
`tl.dot`, round to `bfloat16` and store into `dv`. `safe_vLoop` shows the loop is safe, writes
only the program's elements of `dv`, and leaves `Q`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The number of trips of the `V` loop is `cdiv(V, BV)`, and a trip's columns start below `V`. -/
theorem vtrips_lt {d : Inst} (hd : d.ok) : (d.V + d.BV - 1) / d.BV < 2 ^ 31 := by
  have h1 := hd.V_le; have h2 := hd.bv_le
  have : (d.V + d.BV - 1) / d.BV ≤ d.V + d.BV - 1 := Nat.div_le_self _ _
  omega

/-- **The `V` loop is safe and leaves `Q`.** -/
theorem safe_vLoop {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T) {A : Tensor [d.BT, d.BT] BF16}
    (hA : AOk d init r bh A) :
    Safe ((launch d).env P) (R d r bh) (W d r bh) (fun _ m => Q d init r bh m) init
      (vLoop d do_ dv o_t (o_t.ltS T) A) := by
  have hK31 := vtrips_lt hd
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  have hstp : (IntV.ofInt 1 : I32).toInt = 1 := by decide
  have hhi : (IntV.ofNat ((d.V + d.BV - 1) / d.BV) : I32).toInt =
      (((d.V + d.BV - 1) / d.BV : Nat) : Int) := IntV.I32.toInt_ofNat hK31
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat ((d.V + d.BV - 1) / d.BV) : I32)
      (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [hlo, hstp, hhi]
    have : IntV.maxInt true 32 = 2 ^ 31 - 1 := by decide
    rw [this]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    have : (((d.V + d.BV - 1) / d.BV : Nat) : Int) < 2 ^ 31 := by exact_mod_cast hK31
    omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt
      (IntV.ofNat ((d.V + d.BV - 1) / d.BV) : I32).toInt (IntV.ofInt 1 : I32).toInt =
      (d.V + d.BV - 1) / d.BV := by
    rw [hlo, hstp, hhi]
    unfold tl.tripCount
    generalize (d.V + d.BV - 1) / d.BV = N
    omega
  have hBV0 : 0 < d.BV := by have := hd.bv_ge; omega
  unfold vLoop
  refine safe_last ?_
  refine Safe.mono (safe_forRange (E := (launch d).env P) (R := R d r bh)
    (Q := fun _ m => Q d init r bh m) (m := init) (Wk := ∅) hrange
    (fun t _ m => Agree d init r bh m ∧ d.dv.Valid m ∧ QCols d init r bh (t * d.BV) m)
    (fun t => d.dv.elems (tileSet d r bh t)) ?_ ?_ ?_ ?_ ?_) (fun _ _ h => h) ?_
  · -- the trips write different elements
    intro t u htu hu
    exact Buf.elems_disjoint d.dv (tileSet_disjoint hd r bh htu)
  · exact Set.disjoint_empty _
  · refine ⟨fun x _ => rfl, hpre.2.2.2.1, ?_⟩
    intro l _ _ e he
    omega
  · -- one trip
    intro t ht st m₁ ⟨hag, hdv, hQ⟩
    rw [htc] at ht
    have htV : t * d.BV < d.V := by
      have h1 : (t + 1) * d.BV ≤ d.V + d.BV - 1 := (Nat.le_div_iff_mul_le hBV0).mp ht
      have h2 : (t + 1) * d.BV = t * d.BV + d.BV := by ring
      omega
    have hik : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt
        = (t : Int) := by
      rw [hlo, hstp]
      rw [IntV.toInt_ofInt (by norm_num) (by constructor <;> omega)]
      ring
    exact safe_vBody hP hd hpre hr hbh S hA htV hik hag hdv hQ
  · -- the end of the loop
    intro st m₁ ⟨hag, hdv, hQ⟩
    rw [htc] at hQ
    refine safe_ret (Q_of_QCols ?_ hQ)
    have h1 := Nat.div_add_mod (d.V + d.BV - 1) d.BV
    have h2 := Nat.mod_lt (d.V + d.BV - 1) hBV0
    rw [Nat.mul_comm] at h1
    omega
  · -- the footprint
    intro x hx
    rcases hx with ⟨t, ht, hx⟩ | hx
    · exact Buf.elems_mono d.dv (tileSet_sub d r bh t) hx
    · exact absurd hx (Set.notMem_empty x)

end Claims.ChunkBwdDvLocal
