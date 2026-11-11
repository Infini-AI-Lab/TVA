import Claim
import TVA.Proof
import Proof.Facts
import Proof.Lanes

/-!
# The program's scalars

`geoOf d i`: the scalars (`Geo`) program `i` of the grid computes in lines 108–121, from its
program ids and, on a variable-length launch, the tables. `geoOf_ok`: they are those of chunk
`itg` of the claim.
-/

open TVA
namespace Claims.ChunkFwdO

/-- The number of value blocks, `tl.cdiv(V, BV)`. -/
def nVB (d : Inst) : Nat := (d.V + d.bk - 1) / d.bk

/-- The number of chunk rows of the grid, `NT` or `cdiv(T, 64)`. -/
def nCh (d : Inst) : Nat := if d.varlen then d.NT else NTpad d

/-- Program `i`'s `program_id(0)`. -/
def pid0 (d : Inst) (i : Nat) : Nat := i % grid0 d

/-- Program `i`'s `program_id(1)`. -/
def pid1 (d : Inst) (i : Nat) : Nat := i / grid0 d % (d.B * d.HV)

/-- The sequence of chunk row `c` (variable length). -/
def seqOf (d : Inst) (c : Nat) : Nat := (seqAt d c).toNat

/-- The scalars of program `i` (lines 108–121). -/
def geoOf (d : Inst) (i : Nat) : Geo :=
  if d.varlen then
    { iv := pid0 d i % nVB d, ih := pid1 d i % d.HV, itg := pid0 d i / nVB d,
      it := (chkAt d (pid0 d i / nVB d)).toNat,
      bos := (cuAt d (seqOf d (pid0 d i / nVB d))).toNat,
      Tn := (cuAt d (seqOf d (pid0 d i / nVB d) + 1) - cuAt d (seqOf d (pid0 d i / nVB d))).toNat }
  else
    { iv := pid0 d i % nVB d, ih := pid1 d i % d.HV,
      itg := pid1 d i / d.HV * NTpad d + pid0 d i / nVB d, it := pid0 d i / nVB d,
      bos := pid1 d i / d.HV * d.T, Tn := d.T }

/-- `nVB` in the form of the kernel's `tl.cdiv`. -/
theorem nVB_eq (d : Inst) (hb : 0 < d.bk) : nVB d = (d.V + (d.bk - 1)) / d.bk := by
  unfold nVB; congr 1; omega

/-- The grid's `x` is `nVB · nCh`. -/
theorem grid0_eq (d : Inst) : grid0 d = nVB d * nCh d := rfl

/-- The tables are nondecreasing up to `N`. -/
theorem cu_mono (d : Inst) (h : TablesOk d) {a b : Nat} (hab : a ≤ b) (hb : b ≤ d.N) :
    cuAt d a ≤ cuAt d b := by
  induction b, hab using Nat.le_induction with
  | base => exact le_rfl
  | succ b hab ih => exact (ih (by omega)).trans (h.2.1 b (by omega))

/-- Quotient and remainder of `ib · Y + c`, `c < Y`. -/
theorem div_mod_chunk {ib c Y : Nat} (hc : c < Y) : (ib * Y + c) / Y = ib ∧ (ib * Y + c) % Y = c := by
  have hY : 0 < Y := by omega
  have h1 : (ib * Y + c) / Y = ib :=
    Nat.div_eq_of_lt_le (by nlinarith) (by nlinarith)
  refine ⟨h1, ?_⟩
  have := Nat.div_add_mod (ib * Y + c) Y
  rw [h1] at this
  nlinarith

/-- There is a value block. -/
theorem nVB_pos {d : Inst} (hd : d.ok) : 0 < nVB d := by
  have := bk_pos hd
  have := hd.V_pos
  unfold nVB
  exact Nat.div_pos (by omega) (bk_pos hd)

/-- A value block starts inside `V`. -/
theorem iv_bound {d : Inst} (hd : d.ok) {x : Nat} (hx : x < nVB d) : x * d.bk < d.V := by
  have hb := bk_pos hd
  have h1 : (x + 1) * d.bk ≤ nVB d * d.bk := Nat.mul_le_mul_right _ hx
  have h2 : nVB d * d.bk ≤ d.V + d.bk - 1 := Nat.div_mul_le_self _ _
  have h3 : (x + 1) * d.bk = x * d.bk + d.bk := by ring
  omega

/-- The scalars of a program are those of chunk `itg`. -/
theorem geoOf_ok {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < grid0 d * (d.B * d.HV)) : Geo.Ok d (geoOf d i) (geoOf d i).itg := by
  have hg0 : 0 < grid0 d * (d.B * d.HV) := lt_of_le_of_lt (Nat.zero_le _) hi
  have hgpos : 0 < grid0 d := by
    rcases Nat.eq_zero_or_pos (grid0 d) with h | h
    · rw [h] at hg0; simp at hg0
    · exact h
  have hp0 : pid0 d i < grid0 d := Nat.mod_lt _ hgpos
  have hBHV : 0 < d.B * d.HV := Nat.mul_pos hd.B_pos hd.HV_pos
  have hp1 : pid1 d i < d.B * d.HV := Nat.mod_lt _ hBHV
  have hX := nVB_pos hd
  have hc : pid0 d i / nVB d < nCh d := by
    have := hp0; rw [grid0_eq] at this
    exact (Nat.div_lt_iff_lt_mul hX).mpr (by rw [Nat.mul_comm]; exact this)
  have hiv : pid0 d i % nVB d < nVB d := Nat.mod_lt _ hX
  have hih : pid1 d i % d.HV < d.HV := Nat.mod_lt _ hd.HV_pos
  by_cases hv : d.varlen = true
  · have htab := ht hv
    obtain ⟨hcu0, hmono, hcuT, hrows⟩ : 0 ≤ cuAt d 0 ∧ (∀ i, i < d.N → cuAt d i ≤ cuAt d (i + 1)) ∧
      cuAt d d.N ≤ d.T ∧ (∀ r, r < d.NT → 0 ≤ seqAt d r ∧ seqAt d r < d.N ∧ 0 ≤ chkAt d r ∧
        chkAt d r * BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) :=
      ⟨htab.1, htab.2.1, htab.2.2.1, htab.2.2.2.1⟩
    have hcNT : pid0 d i / nVB d < d.NT := by simpa [nCh, hv] using hc
    obtain ⟨hs0, hsN, hk0, hkl⟩ := hrows _ hcNT
    have hN := hd.N_lt hv
    have hB1 := hd.varlen_B hv
    have hm1 : cuAt d (seqOf d (pid0 d i / nVB d)) ≤ cuAt d (seqOf d (pid0 d i / nVB d) + 1) :=
      hmono _ (by unfold seqOf; omega)
    have hm2 : cuAt d (seqOf d (pid0 d i / nVB d) + 1) ≤ cuAt d d.N :=
      cu_mono d htab (by unfold seqOf; omega) le_rfl
    have hm0 : 0 ≤ cuAt d (seqOf d (pid0 d i / nVB d)) := by
      have := cu_mono d htab (Nat.zero_le (seqOf d (pid0 d i / nVB d)))
        (by unfold seqOf; omega)
      exact hcu0.trans this
    have hTT := hd.T_le
    refine
      { r_lt := by simpa [nRows, geoOf, hv] using hcNT
        itg_eq := rfl
        lo_eq := ?_, hi_eq := ?_, start_lt := ?_, ih_lt := by simpa [geoOf, hv] using hih,
        iv_lt := ?_, end_le := ?_ }
    · simp [rowLo, geoOf, Geo.lo, hv, seqOf]
    · simp only [rowHi, geoOf, Geo.lo, hv, ↓reduceIte, rowLo, seqOf, BT] at *
      omega
    · simp only [geoOf, hv, ↓reduceIte, seqOf, BT] at *
      omega
    · simp only [geoOf, hv, ↓reduceIte]; exact iv_bound hd hiv
    · simp only [geoOf, hv, ↓reduceIte, seqOf] at *
      rw [hB1]; omega
  · have hvf : d.varlen = false := by simpa using hv
    have hcN : pid0 d i / nVB d < NTpad d := by simpa [nCh, hvf] using hc
    have hib : pid1 d i / d.HV < d.B := (Nat.div_lt_iff_lt_mul hd.HV_pos).mpr hp1
    obtain ⟨hdv, hmd⟩ := div_mod_chunk (ib := pid1 d i / d.HV) hcN
    have hT : d.T < 2 ^ 31 := hd.T_lt
    refine
      { r_lt := ?_, itg_eq := rfl, lo_eq := ?_, hi_eq := ?_, start_lt := ?_,
        ih_lt := by simpa [geoOf, hvf] using hih, iv_lt := ?_, end_le := ?_ }
    · simp only [geoOf, hvf, nRows]
      simp only [Bool.false_eq_true, ↓reduceIte]
      nlinarith
    · simp only [rowLo, geoOf, Geo.lo, hvf, Bool.false_eq_true, ↓reduceIte, hdv, hmd]
    · simp only [rowHi, rowLo, geoOf, Geo.lo, hvf, Bool.false_eq_true, ↓reduceIte, hdv, hmd]
    · simp only [geoOf, hvf, Bool.false_eq_true, ↓reduceIte, NTpad, BT] at *
      omega
    · simp only [geoOf, hvf, Bool.false_eq_true, ↓reduceIte]; exact iv_bound hd hiv
    · simp only [geoOf, hvf, Bool.false_eq_true, ↓reduceIte]
      nlinarith

end Claims.ChunkFwdO
