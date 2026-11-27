import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Lane
import Proof.TripLanes
import Proof.Trip
import Proof.TripWalk

/-!
# The loop over the blocks

`safe_loop`: `for i_c in range(NT)` with `NT = nb` runs `nb` trips by `safe_forRange` with the
invariant `Inv`, trip `t` writing block `bt t`; after the last trip every position of the
sequence holds a good value (`Q`).
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst} {i : Nat}

/-- The loop's bounds are within `tl.RangeOk`. -/
theorem rangeOk {NT : I64} {nb : Nat} (hNT : NT.toInt = (nb : Int)) (hnb : nb < 2 ^ 62) :
    tl.RangeOk (IntV.ofInt 0 : I64) NT (IntV.ofInt 1 : I64) := by
  have h1 : (IntV.ofInt 1 : I64).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have h0 : (IntV.ofInt 0 : I64).toInt = 0 := IntV.toInt_ofInt (by norm_num) (by
    unfold IntV.SFits; norm_num)
  unfold tl.RangeOk
  rw [hNT, h0, h1]
  simp only [IntV.maxInt]
  refine ⟨by norm_num, by push_cast; omega, by norm_num, by norm_num⟩

/-- The loop runs `nb` trips. -/
theorem tripCount_eq {NT : I64} {nb : Nat} (hNT : NT.toInt = (nb : Int)) :
    tl.tripCount (IntV.ofInt 0 : I64).toInt NT.toInt (IntV.ofInt 1 : I64).toInt = nb := by
  have h1 : (IntV.ofInt 1 : I64).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have h0 : (IntV.ofInt 0 : I64).toInt = 0 := IntV.toInt_ofInt (by norm_num) (by
    unfold IntV.SFits; norm_num)
  unfold tl.tripCount
  rw [hNT, h0, h1]
  simp

/-- The loop's variable at trip `t` is `t`. -/
theorem loopVar_toInt (t : Nat) (ht : t < 2 ^ 62) :
    (IntV.ofInt ((IntV.ofInt 0 : I64).toInt + (t : Int) * (IntV.ofInt 1 : I64).toInt) : I64).toInt =
      (t : Int) := by
  have h1 : (IntV.ofInt 1 : I64).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have h0 : (IntV.ofInt 0 : I64).toInt = 0 := IntV.toInt_ofInt (by norm_num) (by
    unfold IntV.SFits; norm_num)
  rw [h0, h1, zero_add, mul_one]
  exact IntV.I64.toInt_ofInt ⟨by omega, by omega⟩

/-- Distinct trips write distinct blocks. -/
theorem Wt_disjoint (hd : d.ok) {t u : Nat} (htu : t < u) (hu : u < nbOf d i) :
    Disjoint (Wt d i t) (Wt d i u) := by
  unfold Wt
  refine Buf.elems_disjoint d.o ?_
  rw [Set.disjoint_left]
  rintro e ⟨l, hl, -, rfl⟩ ⟨l', hl', -, he⟩
  have hne : bt d i t ≠ bt d i u := by unfold bt; split <;> omega
  have hH := hd.H_pos
  have h1 : (loOf d i + bt d i t * d.BT + l) * d.H = (loOf d i + bt d i u * d.BT + l') * d.H := by
    omega
  have h2 := Nat.eq_of_mul_eq_mul_right hH h1
  have hb := bt_pos hd
  have k1 : (bt d i t * d.BT + l) / d.BT = bt d i t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hl, Nat.zero_add]
  have k2 : (bt d i u * d.BT + l') / d.BT = bt d i u := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hl', Nat.zero_add]
  have : bt d i t * d.BT + l = bt d i u * d.BT + l' := by omega
  rw [this, k2] at k1
  exact hne k1.symm

/-- The trips write the program's elements. -/
theorem iterW_sub_W (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) (n : Nat) :
    iterW (Wt d i) n ⊆ W d i := by
  rintro x ⟨u, -, hx⟩
  unfold Wt at hx
  refine d.o.elems_mono ?_ hx
  rintro e ⟨l, hl, hen, rfl⟩
  have := lo_add_len hT hi
  exact ⟨loOf d i + bt d i u * d.BT + l, by omega, by omega, rfl⟩

/-- After all trips every position of the sequence holds a good value. -/
theorem Q_of_done (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {m : Mem}
    (h : Done d i (nbOf d i) m) : Q d i m := by
  intro p hp1 hp2
  have hb := bt_pos hd
  have hla := lo_add_len hT hi
  have hlen := len_le_nb (i := i) hd
  obtain ⟨b, hbdef⟩ : ∃ b, b = (p - loOf d i) / d.BT := ⟨_, rfl⟩
  obtain ⟨l, hldef⟩ : ∃ l, l = (p - loOf d i) % d.BT := ⟨_, rfl⟩
  have hl : l < d.BT := by rw [hldef]; exact Nat.mod_lt _ hb
  have hpl : p - loOf d i = b * d.BT + l := by
    rw [hbdef, hldef, Nat.mul_comm]; exact (Nat.div_add_mod _ _).symm
  have hbnb : b < nbOf d i := by
    by_contra hc
    have : nbOf d i * d.BT ≤ b * d.BT := Nat.mul_le_mul_right _ (by omega)
    omega
  -- the trip that handles block `b`
  obtain ⟨u, hudef⟩ : ∃ u, u = (if d.reverse then nbOf d i - 1 - b else b) := ⟨_, rfl⟩
  have hu : u < nbOf d i := by rw [hudef]; split <;> omega
  have hbt : bt d i u = b := by
    rw [hudef]; unfold bt
    cases hr : d.reverse
    · simp only [Bool.false_eq_true, ite_false]
    · simp only [ite_true]; omega
  have hen : bt d i u * d.BT + l < lenOf d i := by rw [hbt]; omega
  obtain ⟨v, hv, hg⟩ := h u hu l hl hen
  have hp : base d i u + l = p := by unfold base; rw [hbt]; omega
  rw [hp] at hv hg
  exact ⟨v, hv, hg⟩

/-- **The loop** from its start: safe, it writes the program's elements and ends with `Q`. -/
theorem safe_loop (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform} (hP : Device P)
    (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    {NT TT bos ih : I64} (hNT : NT.toInt = (nbOf d i : Int)) (hTT : TT.toInt = (lenOf d i : Int))
    (hbos : bos.toInt = (loOf d i : Int)) (hih : ih.toInt = (headOf d i : Int)) :
    Safe ((launch d).env P) (R d) (W d i) (fun _ m => Q d i m) init
      (tl.forRange (IntV.ofInt 0 : I64) NT (IntV.ofInt 1 : I64) (FloatV.ofPat 0 : F32)
        (tripBody d NT TT bos ih) >>= fun _ => pure ()) := by
  have hnb : nbOf d i < 2 ^ 62 := by
    have := nb_le_len (i := i) hd; have := lenOf_le hT hi; have := hd.T_le; omega
  have hcount := tripCount_eq hNT
  have hsafe := safe_forRange (E := (launch d).env P) (R := R d) (m := init)
    (Q := fun (_ : Unit) m' => Q d i m') (init := (FloatV.ofPat 0 : F32)) (body := tripBody d NT TT bos ih)
    (k := fun _ => pure ()) (rangeOk hNT hnb) (fun t z m' => Inv d init i t z m')
    (Wt d i) (Wk := ∅) ?_ (Set.disjoint_empty _) ?_ ?_ ?_
  · rw [hcount, Set.union_empty] at hsafe
    exact hsafe.mono (fun _ _ h => h) (iterW_sub_W hT hi _)
  · intro t u htu hu
    rw [hcount] at hu
    exact Wt_disjoint hd htu hu
  · exact ⟨fun _ _ => rfl, fun _ h => h, ztree_zero, fun u hu => absurd hu (Nat.not_lt_zero u)⟩
  · intro t ht st m₁ hI
    rw [hcount] at ht
    exact safe_trip hd hpre hP hT hi hNT hTT hbos hih ht (loopVar_toInt t (by omega)) hI
  · intro st m₁ hI
    rw [hcount] at hI
    exact safe_ret (Q_of_done hd hT hi hI.2.2.2)

end Claims.ChunkGlobalCumsum
