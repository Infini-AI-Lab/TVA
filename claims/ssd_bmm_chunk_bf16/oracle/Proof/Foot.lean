import Claim
import TVA.Proof
import Proof.Post

/-!
# Disjoint footprints

The output index `oIdx` is injective on the covered `(chunk, group, row, column)`, so distinct
programs' tiles are disjoint (`tile_disjoint`); every program writes inside `out` and none reads
it (`w_disjoint`, `rw_disjoint`).
-/

namespace Claims.SsdBmm

open TVA

/-! ## Arithmetic of the output index -/

/-- `oIdx` written with the strides `oC`, `oH` expanded. -/
theorem oIdx_eq (d : Inst) (c g m n : Nat) :
    oIdx d c g m n = d.C * (d.C * (d.G * c + g) + m) + n := by
  unfold oIdx oC oH
  ring

/-- An index within the output tensor. -/
theorem oIdx_lt_oB {d : Inst} {c g m n : Nat}
    (hc : c < nch d) (hg : g < d.G) (hm : m < d.C) (hn : n < d.C) :
    oIdx d c g m n < oB d := by
  have step1 : d.G * c + g < nch d * d.G := by
    have h1 : d.G * c + g < d.G * c + d.G := by omega
    have h2 : d.G * c + d.G = (c + 1) * d.G := by ring
    have h3 : (c + 1) * d.G ≤ nch d * d.G := Nat.mul_le_mul_right _ (by omega)
    omega
  have step2 : d.C * (d.G * c + g) + m < d.C * (nch d * d.G) := by
    have h1 : d.C * (d.G * c + g) + m < d.C * (d.G * c + g) + d.C := by omega
    have h2 : d.C * (d.G * c + g) + d.C = d.C * (d.G * c + g + 1) := by ring
    have h3 : d.G * c + g + 1 ≤ nch d * d.G := by omega
    have h4 : d.C * (d.G * c + g + 1) ≤ d.C * (nch d * d.G) := Nat.mul_le_mul_left _ h3
    omega
  have step3 : d.C * (d.C * (d.G * c + g) + m) + n < d.C * (d.C * (nch d * d.G)) := by
    have h1 : d.C * (d.C * (d.G * c + g) + m) + n <
        d.C * (d.C * (d.G * c + g) + m) + d.C := by omega
    have h2 : d.C * (d.C * (d.G * c + g) + m) + d.C =
        d.C * (d.C * (d.G * c + g) + m + 1) := by ring
    have h3 : d.C * (d.G * c + g) + m + 1 ≤ d.C * (nch d * d.G) := by omega
    have h4 : d.C * (d.C * (d.G * c + g) + m + 1) ≤ d.C * (d.C * (nch d * d.G)) :=
      Nat.mul_le_mul_left _ h3
    omega
  rw [oIdx_eq]
  have heq : oB d = d.C * (d.C * (nch d * d.G)) := by
    unfold oB
    ring
  rw [heq]
  exact step3

/-- The output index determines the four program coordinates. -/
theorem oIdx_inj {d : Inst} (hd : d.ok) {c g m n c' g' m' n' : Nat}
    (hg : g < d.G) (hg' : g' < d.G) (hm : m < d.C) (hn : n < d.C)
    (hm' : m' < d.C) (hn' : n' < d.C)
    (h : oIdx d c g m n = oIdx d c' g' m' n') :
    c = c' ∧ g = g' ∧ m = m' ∧ n = n' := by
  have hC : 0 < d.C := by have := hd.C_ge; omega
  have hG : 0 < d.G := hd.G_pos
  have hn_eq : n = n' := by
    have e1 : (d.C * (d.C * (d.G * c + g) + m) + n) % d.C = n := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hn]
    have e2 : (d.C * (d.C * (d.G * c' + g') + m') + n') % d.C = n' := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hn']
    rw [oIdx_eq, oIdx_eq] at h
    rw [h] at e1
    exact e1.symm.trans e2
  have hX : d.C * (d.G * c + g) + m = d.C * (d.G * c' + g') + m' := by
    have e1 : (d.C * (d.C * (d.G * c + g) + m) + n) / d.C =
        d.C * (d.G * c + g) + m := by
      rw [Nat.mul_add_div hC, Nat.div_eq_of_lt hn, Nat.add_zero]
    have e2 : (d.C * (d.C * (d.G * c' + g') + m') + n') / d.C =
        d.C * (d.G * c' + g') + m' := by
      rw [Nat.mul_add_div hC, Nat.div_eq_of_lt hn', Nat.add_zero]
    rw [oIdx_eq, oIdx_eq] at h
    rw [h] at e1
    exact e1.symm.trans e2
  have hm_eq : m = m' := by
    have e1 : (d.C * (d.G * c + g) + m) % d.C = m := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hm]
    have e2 : (d.C * (d.G * c' + g') + m') % d.C = m' := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hm']
    rw [hX] at e1
    exact e1.symm.trans e2
  have hY : d.G * c + g = d.G * c' + g' := by
    have e1 : (d.C * (d.G * c + g) + m) / d.C = d.G * c + g := by
      rw [Nat.mul_add_div hC, Nat.div_eq_of_lt hm, Nat.add_zero]
    have e2 : (d.C * (d.G * c' + g') + m') / d.C = d.G * c' + g' := by
      rw [Nat.mul_add_div hC, Nat.div_eq_of_lt hm', Nat.add_zero]
    rw [hX] at e1
    exact e1.symm.trans e2
  have hc_eq : c = c' := by
    have e1 : (d.G * c + g) / d.G = c := by
      rw [Nat.mul_add_div hG, Nat.div_eq_of_lt hg, Nat.add_zero]
    have e2 : (d.G * c' + g') / d.G = c' := by
      rw [Nat.mul_add_div hG, Nat.div_eq_of_lt hg', Nat.add_zero]
    rw [hY] at e1
    exact e1.symm.trans e2
  have hg_eq : g = g' := by
    have e1 : (d.G * c + g) % d.G = g := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hg]
    have e2 : (d.G * c' + g') % d.G = g' := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hg']
    rw [hY] at e1
    exact e1.symm.trans e2
  exact ⟨hc_eq, hg_eq, hm_eq, hn_eq⟩

/-! ## The program coordinates -/

/-- The four program coordinates determine the program. -/
theorem quad_inj {d : Inst} {i j : Fin (launch d).size}
    (hc : cOf d i = cOf d j) (hh : hOf d i = hOf d j)
    (hm : mOf d i = mOf d j) (hn : nOf d i = nOf d j) : i = j := by
  have h1 : (i : Nat) / gx d = (j : Nat) / gx d := by
    unfold cOf at hc
    unfold hOf at hh
    have e1 : d.G * ((i : Nat) / gx d / d.G) + (i : Nat) / gx d % d.G = (i : Nat) / gx d :=
      Nat.div_add_mod _ _
    have e2 : d.G * ((j : Nat) / gx d / d.G) + (j : Nat) / gx d % d.G = (j : Nat) / gx d :=
      Nat.div_add_mod _ _
    rw [hc, hh] at e1
    exact e1.symm.trans e2
  have h2 : (i : Nat) % gx d = (j : Nat) % gx d := by
    unfold mOf at hm
    unfold nOf at hn
    have e1 : numPidN d * ((i : Nat) % gx d / numPidN d) + (i : Nat) % gx d % numPidN d =
        (i : Nat) % gx d := Nat.div_add_mod _ _
    have e2 : numPidN d * ((j : Nat) % gx d / numPidN d) + (j : Nat) % gx d % numPidN d =
        (j : Nat) % gx d := Nat.div_add_mod _ _
    rw [hm, hn] at e1
    exact e1.symm.trans e2
  have hi : gx d * ((i : Nat) / gx d) + (i : Nat) % gx d = (i : Nat) :=
    Nat.div_add_mod _ _
  have hj : gx d * ((j : Nat) / gx d) + (j : Nat) % gx d = (j : Nat) :=
    Nat.div_add_mod _ _
  rw [h1, h2] at hi
  exact Fin.eq_of_val_eq (by omega)

/-! ## The tiles -/

/-- A tile of a chunk and group inside the launch lies below `oB`. -/
theorem tileElems_lt_oB {d : Inst} {c g pm pn : Nat}
    (hc : c < nch d) (hg : g < d.G) :
    ∀ j ∈ tileElems d c g pm pn, j < oB d := by
  rintro j ⟨m, n, hm, hn, hpm, hpn, rfl⟩
  exact oIdx_lt_oB hc hg hpm hpn

/-- Two distinct programs' tiles are disjoint. -/
theorem tile_disjoint {d : Inst} (hd : d.ok) {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (tileElems d (cOf d i) (hOf d i) (mOf d i) (nOf d i))
      (tileElems d (cOf d j) (hOf d j) (mOf d j) (nOf d j)) := by
  rw [Set.disjoint_left]
  rintro x ⟨m, n, hm, hn, hpm, hpn, rfl⟩ ⟨m', n', hm', hn', hpm', hpn', heq⟩
  have hgi : hOf d i < d.G := hOf_lt hd i
  have hgj : hOf d j < d.G := hOf_lt hd j
  obtain ⟨hc, hgg, hmm, hnn⟩ := oIdx_inj hd hgi hgj hpm hpn hpm' hpn' heq
  have hmOf : mOf d i = mOf d j := by
    simp only [BM] at hm hm' hmm
    omega
  have hnOf : nOf d i = nOf d j := by
    simp only [BN] at hn hn' hnn
    omega
  exact hij (quad_inj hc hgg hmOf hnOf)

/-- A program's writes are within `out`. -/
theorem W_sub_out {d : Inst} (hd : d.ok) (init : Mem) (i : Fin (launch d).size) :
    W d init i ⊆ d.out.bytes := by
  intro x hx
  exact Buf.elems_sub_bytes d.out
    (fun j hj => lt_of_lt_of_le (tileElems_lt_oB (cOf_lt i) (hOf_lt hd i) j hj) hd.out_len)
    hx

/-! ## The footprints -/

/-- No program writes what another program reads or writes. -/
theorem w_disjoint {d : Inst} (hd : d.ok) {init : Mem} {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hWi : W d init i ⊆ d.out.bytes := W_sub_out hd init i
  have hR : Disjoint d.out.bytes (R d init j) := by
    have h1 : Disjoint d.out.bytes (d.a.bytes ∪ d.b.bytes) :=
      Disjoint.union_right hd.out_a hd.out_b
    have h2 : Disjoint d.out.bytes ((d.a.bytes ∪ d.b.bytes) ∪ d.seqIdx.bytes) :=
      Disjoint.union_right h1 hd.out_s
    simpa [R] using h2
  have h3 : Disjoint (W d init i) (R d init j) := hR.mono_left hWi
  have h4 : Disjoint (W d init i) (W d init j) :=
    Buf.elems_disjoint d.out (tile_disjoint hd hij)
  exact Disjoint.union_right h3 h4

/-- A program's reads are disjoint from its own writes. -/
theorem rw_disjoint {d : Inst} (hd : d.ok) {init : Mem} {i : Fin (launch d).size} :
    Disjoint (R d init i) (W d init i) := by
  have hWi : W d init i ⊆ d.out.bytes := W_sub_out hd init i
  have h : Disjoint (R d init i) d.out.bytes := by
    have h1 : Disjoint (d.a.bytes ∪ d.b.bytes) d.out.bytes :=
      Disjoint.union_left hd.out_a.symm hd.out_b.symm
    have h2 : Disjoint ((d.a.bytes ∪ d.b.bytes) ∪ d.seqIdx.bytes) d.out.bytes :=
      Disjoint.union_left h1 hd.out_s.symm
    simpa [R] using h2
  exact h.mono_right hWi

end Claims.SsdBmm



