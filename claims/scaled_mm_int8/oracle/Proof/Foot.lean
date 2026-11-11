import Proof.Basic

/-!
# `scaled_mm_int8`: the footprint of one program

What program `p` of the grid reads (its tiles of `A`, `B`, the two scales and the bias) and
writes (its tile of `C`), that each is inside the corresponding matrix of the claim's
conditions, that distinct programs write no byte in common, and that every element of `C` lies
in exactly one program's tile.
-/

namespace Claims.ScaledMmInt8

open TVA

/-- The tile has at least two rows. -/
theorem BM_ge2 {d : Inst} (hd : d.ok) : 2 ≤ d.BM := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (have h1 := congrArg Prod.fst h; omega)

/-- The tile has at least two columns. -/
theorem BN_ge2 {d : Inst} (hd : d.ok) : 2 ≤ d.BN := by
  rcases tile_cases hd with h | h | h | h | h <;>
    (have h1 : d.BN = 64 ∨ d.BN = 128 ∨ d.BN = 32 := by
      first
        | exact Or.inl (congrArg (fun q : Nat × Nat × Nat => q.2.1) h)
        | exact Or.inr (Or.inl (congrArg (fun q : Nat × Nat × Nat => q.2.1) h))
        | exact Or.inr (Or.inr (congrArg (fun q : Nat × Nat × Nat => q.2.1) h))
     rcases h1 with h1 | h1 | h1 <;> omega)

/-- `(BLOCK_SIZE_SCALE_A > 1)` is `scale_a` being per row: `1 < BSA d ↔ d.saVec`. -/
theorem oneLtBSA {d : Inst} (hd : d.ok) :
    (if 1 < BSA d then 1 else 0) = if d.saVec then 1 else 0 := by
  have hB := BM_ge2 hd
  cases hs : d.saVec <;> simp [BSA, hs, show 1 < d.BM by omega]

/-- `1 < BSB` exactly when the `scale_b` is per-channel. -/
theorem oneLtBSB {d : Inst} (hd : d.ok) :
    (if 1 < BSB d then 1 else 0) = if d.sbVec then 1 else 0 := by
  have hB := BN_ge2 hd
  cases hs : d.sbVec <;> simp [BSB, hs, show 1 < d.BN by omega]

/-- The element of `scale_a`'s storage that lane `j` of program `p` loads (lines 109–112). -/
def saOff (d : Inst) (p j : Nat) : Nat :=
  j + (if 1 < BSA d then 1 else 0) * (pm d p * d.BM)

/-- The element of `scale_b`'s storage that lane `j` of program `p` loads (lines 115–118). -/
def sbOff (d : Inst) (p j : Nat) : Nat :=
  j + (if 1 < BSB d then 1 else 0) * (pn d p * d.BN)

/-- The `scale_a` element index of lane `j` of program `p`. -/
theorem saOff_eq {d : Inst} (hd : d.ok) (p j : Nat) :
    saOff d p j = if d.saVec then j + pm d p * d.BM else j := by
  rw [saOff, oneLtBSA hd]
  cases hs : d.saVec <;> simp

/-- The `scale_b` element index of lane `j` of program `p`. -/
theorem sbOff_eq {d : Inst} (hd : d.ok) (p j : Nat) :
    sbOff d p j = if d.sbVec then j + pn d p * d.BN else j := by
  rw [sbOff, oneLtBSB hd]
  cases hs : d.sbVec <;> simp

/-- The elements of `A`'s storage that program `p` loads over the whole loop: its row tile times
every column below `K`. -/
def aTile (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ i k, i < d.BM ∧ k < d.K ∧ pm d p * d.BM + i < d.M ∧
    e = (pm d p * d.BM + i) * d.stride_am + k * d.stride_ak}

/-- The elements of `B`'s storage that program `p` loads: every row below `K` times its column
tile. -/
def bTile (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ k j, k < d.K ∧ j < d.BN ∧ pn d p * d.BN + j < d.N ∧
    e = k * d.stride_bk + (pn d p * d.BN + j) * d.stride_bn}

/-- The elements of `scale_a`'s storage that program `p` loads. -/
def saTile (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ j, j < BSA d ∧ saOff d p j < d.M ∧ e = saIdx d (saOff d p j)}

/-- The elements of `scale_b`'s storage that program `p` loads. -/
def sbTile (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ j, j < BSB d ∧ sbOff d p j < d.N ∧ e = sbIdx d (sbOff d p j)}

/-- The elements of the bias's storage that program `p` loads (none without a bias). -/
def biasTile (d : Inst) (p : Nat) : Set Nat :=
  if d.hasBias then
    {e | ∃ j, j < d.BN ∧ pn d p * d.BN + j < d.N ∧ e = pn d p * d.BN + j}
  else ∅

/-- The elements of `C`'s storage that program `p` stores. -/
def cTile (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ i j, i < d.BM ∧ j < d.BN ∧ pm d p * d.BM + i < d.M ∧ pn d p * d.BN + j < d.N ∧
    e = (pm d p * d.BM + i) * d.stride_cm + (pn d p * d.BN + j) * d.stride_cn}

/-- The read footprint of program `p`. -/
def Rfp (d : Inst) (p : Nat) : Set Int :=
  d.a.elems (aTile d p) ∪ d.b.elems (bTile d p) ∪ d.sa.elems (saTile d p) ∪
    d.sb.elems (sbTile d p) ∪ d.bias.elems (biasTile d p)

/-- The write footprint of program `p`. -/
def Wfp (d : Inst) (p : Nat) : Set Int := d.c.elems (cTile d p)

/-- What program `p` leaves in its tile of `C`: every covered element is `OutSpec`. -/
def Qfp (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ i j, i < d.BM → j < d.BN → pm d p * d.BM + i < d.M → pn d p * d.BN + j < d.N →
    OutSpec d init m (pm d p * d.BM + i) (pn d p * d.BN + j)

/-- The `A` tile lies in the matrix `A`. -/
theorem aTile_sub (d : Inst) (p : Nat) : aTile d p ⊆ aIdx d := by
  rintro e ⟨i, k, hi, hk, hm, rfl⟩
  exact ⟨_, _, hm, hk, rfl⟩

/-- The `B` tile lies in the matrix `B`. -/
theorem bTile_sub (d : Inst) (p : Nat) : bTile d p ⊆ bIdx d := by
  rintro e ⟨k, j, hk, hj, hn, rfl⟩
  exact ⟨_, _, hk, hn, rfl⟩

/-- The `C` tile lies in the matrix `C`. -/
theorem cTile_sub (d : Inst) (p : Nat) : cTile d p ⊆ cIdx d := by
  rintro e ⟨i, j, hi, hj, hm, hn, rfl⟩
  exact ⟨_, _, hm, hn, rfl⟩

/-- The `scale_a` tile lies in the scale vector. -/
theorem saTile_sub {d : Inst} (hd : d.ok) (p : Nat) : saTile d p ⊆ {e | e < saN d} := by
  rintro e ⟨j, hj, hm, rfl⟩
  by_cases hs : d.saVec = true
  · rw [saOff_eq hd, ite_eq_left hs] at hm
    rw [saOff_eq hd, ite_eq_left hs, saIdx, ite_eq_left hs, saN, ite_eq_left hs, hd.scales_contig.1 hs,
      Nat.mul_one]
    exact hm
  · rw [saOff_eq hd, ite_eq_right hs] at hm
    rw [saOff_eq hd, ite_eq_right hs, saIdx, ite_eq_right hs, saN, ite_eq_right hs]
    exact Nat.zero_lt_one

/-- The `scale_b` tile lies in the scale vector. -/
theorem sbTile_sub {d : Inst} (hd : d.ok) (p : Nat) : sbTile d p ⊆ {e | e < sbN d} := by
  rintro e ⟨j, hj, hm, rfl⟩
  by_cases hs : d.sbVec = true
  · rw [sbOff_eq hd, ite_eq_left hs] at hm
    rw [sbOff_eq hd, ite_eq_left hs, sbIdx, ite_eq_left hs, sbN, ite_eq_left hs,
      hd.scales_contig.2.1 hs, Nat.mul_one]
    exact hm
  · rw [sbOff_eq hd, ite_eq_right hs] at hm
    rw [sbOff_eq hd, ite_eq_right hs, sbIdx, ite_eq_right hs, sbN, ite_eq_right hs]
    exact Nat.zero_lt_one

/-- The bias tile lies in the bias vector. -/
theorem biasTile_sub (d : Inst) (p : Nat) : biasTile d p ⊆ {e | e < d.N} := by
  unfold biasTile
  by_cases hb : d.hasBias = true
  · rw [ite_eq_left hb]
    rintro e ⟨j, hj, hn, rfl⟩
    exact hn
  · rw [ite_eq_right hb]
    exact Set.empty_subset _

/-- A program number is its two tile indices. -/
theorem eq_of_pm_pn {d : Inst} {p q : Nat} (h1 : p / gridN d = q / gridN d)
    (h2 : p % gridN d = q % gridN d) : p = q := by
  have e1 := Nat.div_add_mod p (gridN d)
  have e2 := Nat.div_add_mod q (gridN d)
  rw [h1, h2] at e1
  exact e1.symm.trans e2

/-- Two programs write disjoint tiles of `C`: distinct elements of the matrix are distinct
elements of its storage. -/
theorem cTile_disjoint {d : Inst} (hd : d.ok) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (cTile d p) (cTile d q) := by
  rw [Set.disjoint_left]
  rintro e ⟨i, j, hi, hj, him, hjn, he⟩ ⟨i', j', hi', hj', him', hjn', he'⟩
  have h1 : (pm d p * d.BM + i) * d.stride_cm + (pn d p * d.BN + j) * d.stride_cn =
      (pm d q * d.BM + i') * d.stride_cm + (pn d q * d.BN + j') * d.stride_cn := by
    rw [← he, ← he']
  obtain ⟨hii, hjj⟩ := hd.c_inj _ _ _ _ him hjn him' hjn' h1
  have hB := BM_pos hd
  have hN := BN_pos hd
  have hpm : pm d p = pm d q := by
    have h2 : (pm d p * d.BM + i) / d.BM = (pm d q * d.BM + i') / d.BM := by rw [hii]
    rw [Nat.mul_comm (pm d p) d.BM, Nat.mul_comm (pm d q) d.BM] at h2
    rw [Nat.mul_add_div hB, Nat.mul_add_div hB, Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hi',
      Nat.add_zero, Nat.add_zero] at h2
    exact h2
  have hpn : pn d p = pn d q := by
    have h2 : (pn d p * d.BN + j) / d.BN = (pn d q * d.BN + j') / d.BN := by rw [hjj]
    rw [Nat.mul_comm (pn d p) d.BN, Nat.mul_comm (pn d q) d.BN] at h2
    rw [Nat.mul_add_div hN, Nat.mul_add_div hN, Nat.div_eq_of_lt hj, Nat.div_eq_of_lt hj',
      Nat.add_zero, Nat.add_zero] at h2
    exact h2
  exact hpq (eq_of_pm_pn hpm hpn)

/-- `j / BN < cdiv(N, BN)` for every covered column `j`. -/
theorem div_lt_gridN {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.N) : j / d.BN < gridN d := by
  have hB := BN_pos hd
  show j / d.BN < (d.N + d.BN - 1) / d.BN
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (d.N + d.BN - 1) d.BN
  have h2 := Nat.mod_lt (d.N + d.BN - 1) hB
  rw [Nat.mul_comm] at h1
  omega

/-- `i / BM < cdiv(M, BM)` for every covered row `i`. -/
theorem div_lt_gridM {d : Inst} (hd : d.ok) {i : Nat} (hi : i < d.M) :
    i / d.BM < (d.M + d.BM - 1) / d.BM := by
  have hB := BM_pos hd
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (d.M + d.BM - 1) d.BM
  have h2 := Nat.mod_lt (d.M + d.BM - 1) hB
  rw [Nat.mul_comm] at h1
  omega

/-- Every element of the matrix `C` is in the tile of exactly the program whose `pid_m`,
`pid_n` are its row and column tiles. -/
theorem cover {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < d.M) (hj : j < d.N) :
    i / d.BM * gridN d + j / d.BN < grid d ∧
      pm d (i / d.BM * gridN d + j / d.BN) = i / d.BM ∧
      pn d (i / d.BM * gridN d + j / d.BN) = j / d.BN := by
  have hB := BN_pos hd
  have hA : 0 < (d.M + d.BM - 1) / d.BM :=
    Nat.div_pos (by omega) (BM_pos hd)
  have hBpos := gridN_pos hd
  have him := div_lt_gridM hd hi
  have hjn := div_lt_gridN hd hj
  have hgrid : i / d.BM * gridN d + j / d.BN <
      (d.M + d.BM - 1) / d.BM * gridN d := by
    have : i / d.BM + 1 ≤ (d.M + d.BM - 1) / d.BM := him
    have hle : i / d.BM * gridN d + gridN d ≤
        (d.M + d.BM - 1) / d.BM * gridN d := by
      rw [← Nat.add_one_mul]
      exact Nat.mul_le_mul_right _ this
    omega
  refine ⟨hgrid, ?_, ?_⟩
  · show (i / d.BM * gridN d + j / d.BN) / gridN d = i / d.BM
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBpos, Nat.div_eq_of_lt hjn, Nat.zero_add]
  · show (i / d.BM * gridN d + j / d.BN) % gridN d = j / d.BN
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hjn]

end Claims.ScaledMmInt8
