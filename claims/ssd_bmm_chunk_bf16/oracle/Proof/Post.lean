import Claim
import TVA.Proof
import Proof.Numeric

/-!
# From the programs' results to `Post`

Program `i` leaves its tile of `out` as `Q d init i` says, and touches nothing else. Every element
of `out`'s `nchunks × G × C × C` range belongs to exactly one program's tile, and every byte
outside it is written by no program, so the assembled memory is `Post`.
-/

namespace Claims.SsdBmm

open TVA

set_option maxHeartbeats 800000

/-! ## Small arithmetic helpers -/

/-- `BLOCK_SIZE_M` is positive. -/
theorem BM_pos : 0 < BM := by simp [BM]

/-- `BLOCK_SIZE_N` is positive. -/
theorem BN_pos : 0 < BN := by simp [BN]

/-- There is at least one column block. -/
theorem numPidN_pos {d : Inst} (hd : d.ok) : 0 < numPidN d := by
  unfold numPidN
  apply Nat.div_pos _ BN_pos
  have h := hd.C_ge
  simp only [BN]
  omega

/-- The grid's `x` is positive. -/
theorem gx_pos {d : Inst} (hd : d.ok) : 0 < gx d := by
  unfold gx
  have h1 : 0 < (d.C + BM - 1) / BM := by
    apply Nat.div_pos _ BM_pos
    have h := hd.C_ge
    simp only [BM]
    omega
  exact Nat.mul_pos h1 (numPidN_pos hd)

/-- `(a·b + c) / b = a` for `c < b`. -/
theorem add_mul_div {a b c : Nat} (hb : 0 < b) (hc : c < b) : (a * b + c) / b = a := by
  rw [Nat.mul_comm a b, Nat.mul_add_div hb, Nat.div_eq_of_lt hc, Nat.add_zero]

/-- `(a·b + c) % b = c` for `c < b`. -/
theorem add_mul_mod {a b c : Nat} (hc : c < b) : (a * b + c) % b = c := by
  rw [Nat.mul_comm a b, Nat.mul_add_mod, Nat.mod_eq_of_lt hc]

/-- A row `m` of the chunk is in a row block below `cdiv(C, 64)`. -/
theorem div_BM_lt {C m : Nat} (h : m < C) : m / BM < (C + BM - 1) / BM := by
  rw [Nat.div_lt_iff_lt_mul BM_pos]
  have h1 := Nat.div_add_mod (C + BM - 1) BM
  have h2 := Nat.mod_lt (C + BM - 1) BM_pos
  simp only [BM] at h1 h2 ⊢
  omega

/-- A column `n` of the chunk is in a column block below `num_pid_n`. -/
theorem div_BN_lt {d : Inst} {n : Nat} (h : n < d.C) : n / BN < numPidN d := by
  unfold numPidN
  rw [Nat.div_lt_iff_lt_mul BN_pos]
  have h1 := Nat.div_add_mod (d.C + BN - 1) BN
  have h2 := Nat.mod_lt (d.C + BN - 1) BN_pos
  simp only [BN] at h1 h2 ⊢
  omega

/-- The program of chunk `c`, group `g` and tile `(pm, pn)`. -/
def progOf (d : Inst) (c g pm pn : Nat) : Nat :=
  (c * d.G + g) * gx d + (pm * numPidN d + pn)

/-- A tile of a chunk and group inside the launch lies in the first `oB` elements of `out`. -/
theorem tileElems_sub_oB {d : Inst} {c g pm pn : Nat} (hc : c < nch d) (hg : g < d.G) :
    tileElems d c g pm pn ⊆ {j | j < oB d} := by
  rintro j ⟨m, n, hm, hn, h1, h2, rfl⟩
  have hmn : (pm * BM + m) * d.C + (pn * BN + n) < oH d := by
    have h3 : (pm * BM + m) * d.C + (pn * BN + n) < (pm * BM + m + 1) * d.C := by
      conv_rhs => rw [Nat.add_mul, Nat.one_mul]
      exact Nat.add_lt_add_left h2 _
    have h4 : (pm * BM + m + 1) * d.C ≤ d.C * d.C :=
      Nat.mul_le_mul_right _ (by omega)
    have h5 := lt_of_lt_of_le h3 h4
    simpa only [oH] using h5
  have hsum1 : (g + 1) * oH d = g * oH d + oH d := by rw [Nat.add_mul, Nat.one_mul]
  have hsum2 : (c + 1) * oC d = c * oC d + oC d := by rw [Nat.add_mul, Nat.one_mul]
  have hoC : oC d = d.G * oH d := rfl
  have hs1 : g * oH d + ((pm * BM + m) * d.C + (pn * BN + n)) < oC d := by
    have a1 : g * oH d + ((pm * BM + m) * d.C + (pn * BN + n)) < g * oH d + oH d :=
      Nat.add_lt_add_left hmn _
    have a2 : g * oH d + oH d = (g + 1) * oH d := hsum1.symm
    have a3 : (g + 1) * oH d ≤ d.G * oH d := Nat.mul_le_mul_right _ (by omega)
    have a4 : d.G * oH d = oC d := hoC.symm
    calc g * oH d + ((pm * BM + m) * d.C + (pn * BN + n))
        < g * oH d + oH d := a1
      _ = (g + 1) * oH d := a2
      _ ≤ d.G * oH d := a3
      _ = oC d := a4
  have hs2 : c * oC d + (g * oH d + ((pm * BM + m) * d.C + (pn * BN + n))) < nch d * oC d := by
    have a1 : c * oC d + (g * oH d + ((pm * BM + m) * d.C + (pn * BN + n))) <
        c * oC d + oC d := Nat.add_lt_add_left hs1 _
    have a2 : c * oC d + oC d = (c + 1) * oC d := hsum2.symm
    have a3 : (c + 1) * oC d ≤ nch d * oC d := Nat.mul_le_mul_right _ (by omega)
    calc c * oC d + (g * oH d + ((pm * BM + m) * d.C + (pn * BN + n)))
        < c * oC d + oC d := a1
      _ = (c + 1) * oC d := a2
      _ ≤ nch d * oC d := a3
  have hgoal : c * oC d + g * oH d + (pm * BM + m) * d.C + (pn * BN + n) < nch d * oC d := by
    rw [Nat.add_assoc (c * oC d + g * oH d) ((pm * BM + m) * d.C) (pn * BN + n),
      Nat.add_assoc (c * oC d) (g * oH d) ((pm * BM + m) * d.C + (pn * BN + n))]
    exact hs2
  rw [oIdx]
  rw [show oB d = nch d * oC d from rfl]
  exact hgoal

/-- A program's chunk is below `nchunks`. -/
theorem cOf_lt {d : Inst} (i : Fin (launch d).size) : cOf d i < nch d := by
  have hi : (i : Nat) < gx d * gz d := by
    rw [← launch_size']
    exact i.isLt
  have h1 : (i : Nat) / gx d < gz d := Nat.div_lt_of_lt_mul hi
  have h2 : (i : Nat) / gx d < d.G * nch d := by
    rw [gz] at h1
    rw [Nat.mul_comm]
    exact h1
  unfold cOf
  exact Nat.div_lt_of_lt_mul h2

/-- A program's group is below `ngroups`. -/
theorem hOf_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : hOf d i < d.G := by
  unfold hOf
  exact Nat.mod_lt _ hd.G_pos

/-! ## The assembly -/

/-- **The assembly**: memories that agree with the programs' `Q`-memories on their footprints, and with `init` where no program writes, satisfy `Post`. -/
theorem post_of_Q {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size,
      ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init i) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_⟩
  · intro c g m n hc hg hm hn
    have hgx : 0 < gx d := gx_pos hd
    have hnp : 0 < numPidN d := numPidN_pos hd
    have hpm : m / BM < (d.C + BM - 1) / BM := div_BM_lt hm
    have hpn : n / BN < numPidN d := div_BN_lt hn
    have hp : m / BM * numPidN d + n / BN < gx d := by
      have h1 : (m / BM + 1) * numPidN d ≤ (d.C + BM - 1) / BM * numPidN d :=
        Nat.mul_le_mul_right _ (by omega)
      have h2 : m / BM * numPidN d + n / BN < (m / BM + 1) * numPidN d := by
        conv_rhs => rw [Nat.add_mul, Nat.one_mul]
        exact Nat.add_lt_add_left hpn _
      have h3 : (d.C + BM - 1) / BM * numPidN d = gx d := rfl
      calc m / BM * numPidN d + n / BN < (m / BM + 1) * numPidN d := h2
        _ ≤ (d.C + BM - 1) / BM * numPidN d := h1
        _ = gx d := h3
    have hq : c * d.G + g + 1 ≤ nch d * d.G := by
      have h1 : (c + 1) * d.G ≤ nch d * d.G := Nat.mul_le_mul_right _ (by omega)
      have h2 : c * d.G + g + 1 ≤ (c + 1) * d.G := by
        conv_rhs => rw [Nat.add_mul, Nat.one_mul]
        omega
      exact le_trans h2 h1
    have hlt : progOf d c g (m / BM) (n / BN) < (launch d).size := by
      rw [launch_size', progOf, gz]
      have h1 : (c * d.G + g + 1) * gx d ≤ (nch d * d.G) * gx d :=
        Nat.mul_le_mul_right _ hq
      have h2 : (c * d.G + g) * gx d + (m / BM * numPidN d + n / BN) <
          (c * d.G + g + 1) * gx d := by
        conv_rhs => rw [Nat.add_mul, Nat.one_mul]
        exact Nat.add_lt_add_left hp _
      have h3 := lt_of_lt_of_le h2 h1
      rw [Nat.mul_comm (gx d) (nch d * d.G)]
      exact h3
    set i : Fin (launch d).size := ⟨progOf d c g (m / BM) (n / BN), hlt⟩ with hi_def
    have hival : (i : Nat) = progOf d c g (m / BM) (n / BN) := by rw [hi_def]
    obtain ⟨m₀, hQ, hagree⟩ := hloc i
    have hcOf : cOf d i = c := by
      unfold cOf
      rw [hival, progOf, add_mul_div hgx hp, add_mul_div hd.G_pos hg]
    have hhOf : hOf d i = g := by
      unfold hOf
      rw [hival, progOf, add_mul_div hgx hp, add_mul_mod hg]
    have hmOf : mOf d i = m / BM := by
      unfold mOf
      rw [hival, progOf, add_mul_mod hp, add_mul_div hnp hpn]
    have hnOf : nOf d i = n / BN := by
      unfold nOf
      rw [hival, progOf, add_mul_mod hp, add_mul_mod hpn]
    have hmsum : m / BM * BM + m % BM = m := by
      rw [Nat.mul_comm, Nat.div_add_mod]
    have hnsum : n / BN * BN + n % BN = n := by
      rw [Nat.mul_comm, Nat.div_add_mod]
    have hQ' : OutElem d init m₀ c g m n := by
      have h := hQ (m % BM) (n % BN) (Nat.mod_lt _ BM_pos) (Nat.mod_lt _ BN_pos)
        (by rw [hmOf, hmsum]; exact hm)
        (by rw [hnOf, hnsum]; exact hn)
      rwa [hcOf, hhOf, hmOf, hnOf, hmsum, hnsum] at h
    obtain ⟨v, hread, hrest⟩ := hQ'
    refine ⟨v, ?_, hrest⟩
    have hcongr : m₀.read .f32 (d.out.ptr (oIdx d c g m n : Int)).addr
        = fin.read .f32 (d.out.ptr (oIdx d c g m n : Int)).addr := by
      refine Mem.read_congr (m₁ := m₀) (m₂ := fin) (fun x hx => ?_)
      have hj : oIdx d c g m n ∈ tileElems d (cOf d i) (hOf d i) (mOf d i) (nOf d i) := by
        refine ⟨m % BM, n % BN, Nat.mod_lt _ BM_pos, Nat.mod_lt _ BN_pos, ?_, ?_, ?_⟩
        · rw [hmOf, hmsum]; exact hm
        · rw [hnOf, hnsum]; exact hn
        · rw [hcOf, hhOf, hmOf, hnOf, hmsum, hnsum]
      have hxW : x ∈ W d init i := d.out.span_sub_elems hj hx
      exact (hagree x (Or.inr hxW)).symm
    exact hcongr.symm.trans hread
  · intro z hz
    exact hfr z fun i hzi =>
      hz (d.out.elems_mono (tileElems_sub_oB (cOf_lt i) (hOf_lt hd i)) hzi)

end Claims.SsdBmm



