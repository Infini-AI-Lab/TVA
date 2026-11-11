import Proof.Program

/-!
# Footprints of the programs

Program `t` reads `R d t` (inside the inputs `inBytes d`) and writes `W d t` (inside the outputs
`outBytes d`); the writes of two programs are disjoint, and so are a program's reads and writes.
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- The rows `i ≠ j` of an `[M, N]` tensor share no element. -/
theorem rowEls_disjoint {i j : Nat} (hij : i ≠ j) : Disjoint (rowEls d i) (rowEls d j) := by
  rw [Set.disjoint_left]
  rintro k ⟨a, ha, rfl⟩ ⟨b, hb, hab⟩
  unfold idx at hab
  rcases lt_or_gt_of_ne hij with h | h
  · have : (i + 1) * d.N ≤ j * d.N := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul] at this
    omega
  · have : (j + 1) * d.N ≤ i * d.N := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul] at this
    omega

/-- Program `t` writes only outputs. -/
theorem W_sub_out {t : Nat} (ht : t < d.M) : W d t ⊆ outBytes d := by
  intro z hz
  rcases hz with (hz | hz) | hz
  · exact Or.inl (Or.inl (y_row_sub ht hz))
  · exact Or.inl (Or.inr (mean_els_sub ht hz))
  · exact Or.inr (rstd_els_sub ht hz)

/-- Program `t` reads only inputs. -/
theorem R_sub_in {t : Nat} (ht : t < d.M) : R d t ⊆ inBytes d := by
  intro z hz
  rcases hz with (hz | hz) | hz
  · exact Or.inl (Or.inl (x_row_sub ht hz))
  · exact Or.inl (Or.inr hz)
  · exact Or.inr hz

/-- No program writes what another reads. -/
theorem W_disjoint_R (hd : d.ok) {i j : Nat} (hi : i < d.M) (hj : j < d.M) :
    Disjoint (W d i) (R d j) :=
  hd.out_in.mono (W_sub_out hi) (R_sub_in hj)

/-- Two programs write different bytes. -/
theorem W_disjoint_W (hd : d.ok) {i j : Nat} (hi : i < d.M) (hj : j < d.M) (hij : i ≠ j) :
    Disjoint (W d i) (W d j) := by
  have hy : Disjoint (d.y.elems (rowEls d i)) (d.y.elems (rowEls d j)) :=
    d.y.elems_disjoint (rowEls_disjoint hij)
  have hm : Disjoint (d.mean.elems {i}) (d.mean.elems {j}) :=
    d.mean.elems_disjoint (by simpa using hij)
  have hr : Disjoint (d.rstd.elems {i}) (d.rstd.elems {j}) :=
    d.rstd.elems_disjoint (by simpa using hij)
  have yi_mj := hd.y_mean.mono (y_row_sub hi) (mean_els_sub hj)
  have yi_rj := hd.y_rstd.mono (y_row_sub hi) (rstd_els_sub hj)
  have yj_mi := hd.y_mean.mono (y_row_sub hj) (mean_els_sub hi)
  have yj_ri := hd.y_rstd.mono (y_row_sub hj) (rstd_els_sub hi)
  have mi_rj := hd.mean_rstd.mono (mean_els_sub hi) (rstd_els_sub hj)
  have mj_ri := hd.mean_rstd.mono (mean_els_sub hj) (rstd_els_sub hi)
  refine Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_ <;>
    refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_
  all_goals first | assumption | exact Disjoint.symm ‹_›

/-- A program reads nothing it writes. -/
theorem R_disjoint_W (hd : d.ok) {t : Nat} (ht : t < d.M) : Disjoint (R d t) (W d t) :=
  (W_disjoint_R hd ht ht).symm

end Claims.LayerNormFwdF16
