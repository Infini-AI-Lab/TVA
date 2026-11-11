import Claim
import TVA.Proof
import Proof.Basic

/-!
# The footprints: disjointness of the programs' tiles of `c` and the frame
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The lanes of one program that pass the mask lie inside the matrix of `c`. -/
theorem cImg_sub (d : Inst) (i : Fin (launch d).size) :
    cImg d i ⊆ {e | e < d.M * d.N} := by
  rintro j ⟨l, hl, rfl⟩
  simp only [Set.mem_ofPred_eq]
  rw [laneMask_eq] at hl
  simp only [decide_eq_true_eq] at hl
  obtain ⟨h0, h1, h2, h3⟩ := hl
  rw [laneCoord_zero] at h0 h1
  rw [laneCoord_one] at h2 h3
  have h0' : 0 ≤ (laneRow d i l : Int) := h0
  have h1' : (laneRow d i l : Int) < (d.M : Int) := h1
  have h2' : 0 ≤ (laneCol d i l : Int) := h2
  have h3' : (laneCol d i l : Int) < (d.N : Int) := h3
  have hM : laneRow d i l < d.M := by exact_mod_cast h1'
  have hN : laneCol d i l < d.N := by exact_mod_cast h3'
  unfold laneIdx
  calc laneRow d i l * d.N + laneCol d i l
      < laneRow d i l * d.N + d.N := by omega
    _ = (laneRow d i l + 1) * d.N := by ring
    _ ≤ d.M * d.N := Nat.mul_le_mul_right _ (by omega)

/-- An enabled lane of program `i` lies inside `c`: its row is a row of `c` and its column a
column of `c`. -/
theorem laneMask_bounds {d : Inst} {i : Fin (launch d).size} {l : Idx [128, 256]}
    (h : laneMask d i l = true) :
    laneRow d i l < d.M ∧ laneCol d i l < d.N := by
  rw [laneMask_eq d i l] at h
  simp only [decide_eq_true_eq] at h
  obtain ⟨-, h1, -, h3⟩ := h
  rw [laneCoord_zero] at h1
  rw [laneCoord_one] at h3
  exact ⟨by exact_mod_cast h1, by exact_mod_cast h3⟩

/-- Two different programs write disjoint tiles of `c`: a `c` element is at most one lane of one
program. -/
theorem cImg_disjoint {d : Inst} (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (cImg d i) (cImg d j) := by
  rw [Set.disjoint_left]
  intro k hki hkj
  obtain ⟨l, hl, rfl⟩ := hki
  obtain ⟨l', hl', he⟩ := hkj
  obtain ⟨-, hNi⟩ := laneMask_bounds hl
  obtain ⟨-, hNj⟩ := laneMask_bounds hl'
  have he' : laneIdx d i l = laneIdx d j l' := he.symm
  have hrow : laneRow d i l = laneRow d j l' := by
    have hNpos : 0 < d.N := by omega
    have h := congrArg (fun x => x / d.N) he'
    unfold laneIdx at h
    rw [show laneRow d i l * d.N + laneCol d i l = d.N * laneRow d i l + laneCol d i l from by ring,
      show laneRow d j l' * d.N + laneCol d j l' = d.N * laneRow d j l' + laneCol d j l' from by ring,
      Nat.mul_add_div hNpos, Nat.mul_add_div hNpos, Nat.div_eq_of_lt hNi, Nat.div_eq_of_lt hNj,
      Nat.add_zero, Nat.add_zero] at h
    exact h
  have hcol : laneCol d i l = laneCol d j l' := by
    have h := congrArg (fun x => x % d.N) he'
    unfold laneIdx at h
    rw [show laneRow d i l * d.N + laneCol d i l = d.N * laneRow d i l + laneCol d i l from by ring,
      show laneRow d j l' * d.N + laneCol d j l' = d.N * laneRow d j l' + laneCol d j l' from by ring,
      Nat.mul_add_mod, Nat.mul_add_mod, Nat.mod_eq_of_lt hNi, Nat.mod_eq_of_lt hNj] at h
    exact h
  have hmod : (i : Nat) % gridM d = (j : Nat) % gridM d := by
    have h0 : l.coord 0 < 128 := l.1.2
    have h0' : l'.coord 0 < 128 := l'.1.2
    unfold laneRow at hrow
    omega
  have hdiv : (i : Nat) / gridM d = (j : Nat) / gridM d := by
    have h1 : l.coord 1 < 256 := l.2.1.2
    have h1' : l'.coord 1 < 256 := l'.2.1.2
    unfold laneCol at hcol
    omega
  refine hij (Fin.ext ?_)
  rw [← Nat.div_add_mod (i : Nat) (gridM d), ← Nat.div_add_mod (j : Nat) (gridM d), hmod, hdiv]

/-- No program writes another program's footprint. -/
theorem prog_disj {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d i) (R d j ∪ W d j) := by
  show Disjoint (d.c.elems (cImg d i)) (inEl d ∪ d.c.elems (cImg d j))
  rw [Set.disjoint_union_right]
  exact ⟨Disjoint.mono (Buf.elems_mono d.c (cImg_sub d i)) (le_refl _) hd.c_in,
    Buf.elems_disjoint d.c (cImg_disjoint i j hij)⟩

/-- No program reads what it writes. -/
theorem prog_rw {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    Disjoint (R d i) (W d i) := by
  show Disjoint (inEl d) (d.c.elems (cImg d i))
  exact Disjoint.mono (le_refl _) (Buf.elems_mono d.c (cImg_sub d i)) hd.c_in.symm

/-- The programs' results give `Post`. -/
theorem prog_post {d : Inst} (hd : d.ok) {init : Mem} (_hpre : Pre d init) (fin : Mem)
    (hloc : ∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d i ∪ W d i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ W d i) → fin x = init x) : Post d init fin := by
  constructor
  · intro i j hi hj
    have hMdiv : d.M = 128 * (d.M / 128) := M_div d hd
    have hNdiv : d.N = 128 * (d.N / 128) := N_div d hd
    have hgM : gridM d = d.M / 128 := gridM_eq hd
    have hgN : gridN d = (d.N + 255) / 256 := gridN_eq hd
    have hpm : i / 128 < gridM d := by omega
    have hpn : j / 256 < gridN d := by omega
    have hp_lt : j / 256 * gridM d + i / 128 < (launch d).size := by
      rw [launch_size]
      show j / 256 * gridM d + i / 128 < gridM d * gridN d
      calc j / 256 * gridM d + i / 128 < j / 256 * gridM d + gridM d := Nat.add_lt_add_left hpm _
        _ = (j / 256 + 1) * gridM d := (Nat.succ_mul _ _).symm
        _ ≤ gridN d * gridM d := Nat.mul_le_mul_right _ (by omega)
        _ = gridM d * gridN d := Nat.mul_comm _ _
    let p : Fin (launch d).size := ⟨j / 256 * gridM d + i / 128, hp_lt⟩
    obtain ⟨m, hQ, hagree⟩ := hloc p
    let l : Idx [128, 256] := (⟨i % 128, Nat.mod_lt _ (by norm_num)⟩,
      (⟨j % 256, Nat.mod_lt _ (by norm_num)⟩, ()))
    have hl0 : l.coord 0 = i % 128 := rfl
    have hl1 : l.coord 1 = j % 256 := rfl
    have hrow : laneRow d p l = i := by
      have hmod : (p : Nat) % gridM d = i / 128 := by
        show (j / 256 * gridM d + i / 128) % gridM d = i / 128
        rw [Nat.mul_comm (j / 256) (gridM d), Nat.mul_add_mod, Nat.mod_eq_of_lt hpm]
      unfold laneRow
      rw [hmod, hl0]
      omega
    have hcol : laneCol d p l = j := by
      have hdiv : (p : Nat) / gridM d = j / 256 := by
        show (j / 256 * gridM d + i / 128) / gridM d = j / 256
        rw [Nat.mul_comm (j / 256) (gridM d), Nat.mul_add_div (gridM_pos hd),
          Nat.div_eq_of_lt hpm, Nat.add_zero]
      unfold laneCol
      rw [hdiv, hl1]
      omega
    have hmask : laneMask d p l = true := by
      rw [laneMask_eq d p l]
      simp only [decide_eq_true_eq]
      rw [laneCoord_zero, laneCoord_one, hrow, hcol]
      exact ⟨by omega, by exact_mod_cast hi, by omega, by exact_mod_cast hj⟩
    have hspec : OutSpec d init m i j := by
      have h := hQ l hmask
      rw [hrow, hcol] at h
      exact h
    have hmem : i * d.N + j ∈ cImg d p := by
      refine ⟨l, hmask, ?_⟩
      show laneRow d p l * d.N + laneCol d p l = i * d.N + j
      rw [hrow, hcol]
    have hread : fin.read .f16 (d.c.ptr (↑(i * d.N + j))).addr
        = m.read .f16 (d.c.ptr (↑(i * d.N + j))).addr :=
      Mem.read_congr fun x hx => hagree x (Or.inr (Buf.span_sub_elems d.c hmem hx))
    obtain ⟨c, hc, v, hv1, hv2, hv3, hv4⟩ := hspec
    exact ⟨c, by rw [hread]; exact hc, v, hv1, hv2, hv3, hv4⟩
  · intro z hz
    exact hfr z fun k hk => hz (Buf.elems_mono d.c (cImg_sub d k) hk)

end Claims.BlockScaledMxfp8
