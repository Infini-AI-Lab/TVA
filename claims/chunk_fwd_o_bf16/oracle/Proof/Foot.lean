import Claim
import TVA.Proof
import Proof.GeoOf
import Proof.Prelude

/-!
# The launch and the programs' footprints

* `launch_valid`: the grid is within CUDA's limits and the block shapes are powers of two;
* `progOf`: the program of chunk row `r`, value head `j` and value block `v`, the inverse of
  `geoOf` (`geoOf_progOf`, `progOf_geoOf`);
* distinct programs write distinct elements of `o`, and none writes what any program reads.
-/

open TVA
namespace Claims.ChunkFwdO

/-- The size of the grid. -/
theorem launch_size_eq (d : Inst) : (launch d).size = grid0 d * (d.B * d.HV) * 1 := rfl

/-- A program index is below the grid size. -/
theorem lt_size {d : Inst} (i : Fin (launch d).size) : (i : Nat) < grid0 d * (d.B * d.HV) := by
  have h : (i : Nat) < (launch d).size := i.isLt
  have e : (launch d).size = grid0 d * (d.B * d.HV) := by rw [launch_size_eq, Nat.mul_one]
  omega

/-- The launch is valid: the grid is within CUDA's limits and every block shape is a power of two. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hsz : (launch d).size = grid0 d * (d.B * d.HV) := by rw [launch_size_eq, Nat.mul_one]
  refine ⟨by simpa [launch] using hd.grid0_lt, by simpa [launch] using hd.grid1_lt,
    by simp [launch], ?_, rfl, rfl, rfl, ?_⟩
  · rw [hsz]; exact hd.size_lt
  · have hbk : d.bk = 32 ∨ d.bk = 64 ∨ d.bk = 128 := hd.bk_ok
    have hp2 : ∀ k, (k = 1 ∨ k = 32 ∨ k = 64 ∨ k = 128) → ∃ e, k = 2 ^ e := by
      rintro k (rfl | rfl | rfl | rfl)
      exacts [⟨0, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩]
    intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    have hBT : BT = 64 := rfl
    refine ⟨fun k hk => hp2 k ?_, ?_⟩
    · rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hk <;>
        rcases hbk with h | h | h <;> rcases hk with rfl | rfl <;> simp [h, hBT]
    · rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        rcases hbk with h | h | h <;> simp [h, hBT]

/-! ## Programs and chunks -/

/-- The program of chunk row `r`, value head `j` and value block `v`. -/
def progOf (d : Inst) (r j v : Nat) : Nat :=
  if d.varlen then (r * nVB d + v) + grid0 d * j
  else ((r % NTpad d) * nVB d + v) + grid0 d * ((r / NTpad d) * d.HV + j)

/-- The program of a chunk row, head and block is in the grid, with those program ids. -/
theorem progOf_lt {d : Inst} (hd : d.ok) {r j v : Nat} (hr : r < nRows d) (hj : j < d.HV)
    (hv : v < nVB d) : progOf d r j v < grid0 d * (d.B * d.HV) ∧
      pid0 d (progOf d r j v) = (if d.varlen then r * nVB d + v else (r % NTpad d) * nVB d + v) ∧
      pid1 d (progOf d r j v) = (if d.varlen then j else (r / NTpad d) * d.HV + j) := by
  by_cases hvl : d.varlen = true
  · have hB := hd.varlen_B hvl
    have hr' : r < d.NT := by simpa [nRows, hvl] using hr
    have hp0 : r * nVB d + v < grid0 d := by
      rw [grid0_eq]; simp only [nCh, hvl, ↓reduceIte]
      calc r * nVB d + v < r * nVB d + nVB d := by omega
        _ = (r + 1) * nVB d := by ring
        _ ≤ d.NT * nVB d := Nat.mul_le_mul_right _ hr'
        _ = nVB d * d.NT := Nat.mul_comm _ _
    have hp1 : j < d.B * d.HV := by rw [hB]; omega
    have e0 : (r * nVB d + v + grid0 d * j) % grid0 d = r * nVB d + v := by
      rw [Nat.add_mul_mod_self_left]; exact Nat.mod_eq_of_lt hp0
    have e1 : (r * nVB d + v + grid0 d * j) / grid0 d = j := by
      rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hp0]; simp
    refine ⟨?_, ?_, ?_⟩
    · unfold progOf; simp only [hvl, ↓reduceIte]
      calc _ < grid0 d * (j + 1) := by nlinarith
        _ ≤ grid0 d * (d.B * d.HV) := Nat.mul_le_mul_left _ hp1
    · simp only [pid0, progOf, hvl, ↓reduceIte]; exact e0
    · simp only [pid1, progOf, hvl, ↓reduceIte]; rw [e1]; exact Nat.mod_eq_of_lt hp1
  · have hvf : d.varlen = false := by simpa using hvl
    have hr' : r < d.B * NTpad d := by simpa [nRows, hvf] using hr
    have hY : 0 < NTpad d := by
      rcases Nat.eq_zero_or_pos (NTpad d) with h | h
      · rw [h] at hr'; simp at hr'
      · exact h
    have hrY : r / NTpad d < d.B := (Nat.div_lt_iff_lt_mul hY).mpr hr'
    have hp0 : (r % NTpad d) * nVB d + v < grid0 d := by
      rw [grid0_eq]; simp only [nCh, hvf, Bool.false_eq_true, ↓reduceIte]
      have := Nat.mod_lt r hY
      calc (r % NTpad d) * nVB d + v < (r % NTpad d) * nVB d + nVB d := by omega
        _ = (r % NTpad d + 1) * nVB d := by ring
        _ ≤ NTpad d * nVB d := Nat.mul_le_mul_right _ this
        _ = nVB d * NTpad d := Nat.mul_comm _ _
    have hp1 : (r / NTpad d) * d.HV + j < d.B * d.HV := by
      calc (r / NTpad d) * d.HV + j < (r / NTpad d) * d.HV + d.HV := by omega
        _ = (r / NTpad d + 1) * d.HV := by ring
        _ ≤ d.B * d.HV := Nat.mul_le_mul_right _ hrY
    have e0 : ((r % NTpad d) * nVB d + v + grid0 d * ((r / NTpad d) * d.HV + j)) % grid0 d =
        (r % NTpad d) * nVB d + v := by
      rw [Nat.add_mul_mod_self_left]; exact Nat.mod_eq_of_lt hp0
    have e1 : ((r % NTpad d) * nVB d + v + grid0 d * ((r / NTpad d) * d.HV + j)) / grid0 d =
        (r / NTpad d) * d.HV + j := by
      rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hp0]; simp
    refine ⟨?_, ?_, ?_⟩
    · unfold progOf; simp only [hvf, Bool.false_eq_true, ↓reduceIte]
      calc _ < grid0 d * ((r / NTpad d) * d.HV + j + 1) := by nlinarith
        _ ≤ grid0 d * (d.B * d.HV) := Nat.mul_le_mul_left _ hp1
    · simp only [pid0, progOf, hvf, Bool.false_eq_true, ↓reduceIte]; exact e0
    · simp only [pid1, progOf, hvf, Bool.false_eq_true, ↓reduceIte]; rw [e1]; exact Nat.mod_eq_of_lt hp1

/-- `geoOf` inverts `progOf`. -/
theorem geoOf_progOf {d : Inst} (hd : d.ok) {r j v : Nat} (hr : r < nRows d) (hj : j < d.HV)
    (hv : v < nVB d) : (geoOf d (progOf d r j v)).iv = v ∧ (geoOf d (progOf d r j v)).ih = j ∧
      (geoOf d (progOf d r j v)).itg = r := by
  obtain ⟨-, h0, h1⟩ := progOf_lt hd hr hj hv
  have hX := nVB_pos hd
  by_cases hvl : d.varlen = true
  · simp only [hvl, ↓reduceIte] at h0 h1
    have hB := hd.varlen_B hvl
    simp only [geoOf, hvl, ↓reduceIte, h0, h1]
    refine ⟨?_, Nat.mod_eq_of_lt hj, ?_⟩
    · rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hv
    · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hX, Nat.div_eq_of_lt hv]; simp
  · have hvf : d.varlen = false := by simpa using hvl
    simp only [hvf, Bool.false_eq_true, ↓reduceIte] at h0 h1
    have hr' : r < d.B * NTpad d := by simpa [nRows, hvf] using hr
    have hY : 0 < NTpad d := by
      rcases Nat.eq_zero_or_pos (NTpad d) with h | h
      · rw [h] at hr'; simp at hr'
      · exact h
    have hrm := Nat.mod_lt r hY
    simp only [geoOf, hvf, Bool.false_eq_true, ↓reduceIte, h0, h1]
    refine ⟨?_, ?_, ?_⟩
    · rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hv
    · rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hj
    · have e1 : ((r / NTpad d) * d.HV + j) / d.HV = r / NTpad d := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hd.HV_pos, Nat.div_eq_of_lt hj]; simp
      have e2 : ((r % NTpad d) * nVB d + v) / nVB d = r % NTpad d := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hX, Nat.div_eq_of_lt hv]; simp
      rw [e1, e2]
      exact Nat.div_add_mod' r (NTpad d)

/-- A program index is `pid0 + grid0 · pid1`. -/
theorem pid_decomp {d : Inst} (_hd : d.ok) {i : Nat} (hi : i < grid0 d * (d.B * d.HV)) :
    i = pid0 d i + grid0 d * pid1 d i ∧ pid1 d i < d.B * d.HV := by
  have hgpos : 0 < grid0 d := by
    rcases Nat.eq_zero_or_pos (grid0 d) with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have hdiv : i / grid0 d < d.B * d.HV := (Nat.div_lt_iff_lt_mul hgpos).mpr (by
    rw [Nat.mul_comm]; exact hi)
  have : pid1 d i = i / grid0 d := Nat.mod_eq_of_lt hdiv
  refine ⟨?_, by rw [this]; exact hdiv⟩
  rw [this]; unfold pid0
  have := Nat.mod_add_div i (grid0 d)
  omega

/-- A program is the program of its chunk row, value head and value block. -/
theorem progOf_geoOf {d : Inst} (hd : d.ok) {i : Nat} (hi : i < grid0 d * (d.B * d.HV)) :
    progOf d (geoOf d i).itg (geoOf d i).ih (geoOf d i).iv = i := by
  obtain ⟨hdec, hp1⟩ := pid_decomp hd hi
  have hX := nVB_pos hd
  by_cases hvl : d.varlen = true
  · have hB := hd.varlen_B hvl
    have hp1' : pid1 d i < d.HV := by rw [hB] at hp1; omega
    simp only [geoOf, progOf, hvl, ↓reduceIte]
    rw [Nat.mod_eq_of_lt hp1', Nat.div_add_mod' (pid0 d i) (nVB d)]
    omega
  · have hvf : d.varlen = false := by simpa using hvl
    have hp0 : pid0 d i < grid0 d := Nat.mod_lt _ (by
      rcases Nat.eq_zero_or_pos (grid0 d) with h | h
      · rw [h] at hi; simp at hi
      · exact h)
    have hc : pid0 d i / nVB d < NTpad d := (Nat.div_lt_iff_lt_mul hX).mpr (by
      have := hp0; rw [grid0_eq] at this
      simpa [nCh, hvf, Nat.mul_comm] using this)
    obtain ⟨e1, e2⟩ := div_mod_chunk (ib := pid1 d i / d.HV) hc
    simp only [geoOf, progOf, hvf, Bool.false_eq_true, ↓reduceIte, e1, e2]
    rw [Nat.div_add_mod' (pid0 d i) (nVB d), Nat.div_add_mod' (pid1 d i) d.HV]
    omega

/-- Two programs with the same chunk row, head and value block are the same program. -/
theorem geoOf_inj {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < grid0 d * (d.B * d.HV))
    (hj : j < grid0 d * (d.B * d.HV)) (h1 : (geoOf d i).iv = (geoOf d j).iv)
    (h2 : (geoOf d i).ih = (geoOf d j).ih) (h3 : (geoOf d i).itg = (geoOf d j).itg) : i = j := by
  rw [← progOf_geoOf hd hi, ← progOf_geoOf hd hj, h1, h2, h3]

/-- Two different chunks of the batch have no token in common. -/
theorem chunk_disjoint {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {r r' t : Nat}
    (hr : r < nRows d) (hr' : r' < nRows d) (hne : r ≠ r')
    (h1 : rowLo d r ≤ t ∧ t < rowHi d r) (h2 : rowLo d r' ≤ t ∧ t < rowHi d r') : False := by
  by_cases hvl : d.varlen = true
  · have htab := ht hvl
    have hNT : r < d.NT := by simpa [nRows, hvl] using hr
    have hNT' : r' < d.NT := by simpa [nRows, hvl] using hr'
    obtain ⟨hs0, hsN, hk0, hkl⟩ := htab.2.2.2.1 r hNT
    obtain ⟨hs0', hsN', hk0', hkl'⟩ := htab.2.2.2.1 r' hNT'
    have hu := htab.2.2.2.2.1 r r' hNT hNT' hne
    simp only [rowLo, rowHi, hvl, ↓reduceIte, BT] at h1 h2
    have hmono : ∀ {a b : Nat}, a ≤ b → b ≤ d.N → cuAt d a ≤ cuAt d b :=
      fun h1 h2 => cu_mono d htab h1 h2
    have hN := hd.N_lt hvl
    have hcu0 := htab.1
    have hn0 : 0 ≤ cuAt d (seqAt d r).toNat := (hcu0.trans (hmono (Nat.zero_le _) (by omega)))
    have hn0' : 0 ≤ cuAt d (seqAt d r').toNat := (hcu0.trans (hmono (Nat.zero_le _) (by omega)))
    rcases lt_trichotomy (seqAt d r).toNat (seqAt d r').toNat with h | h | h
    · have := hmono (show (seqAt d r).toNat + 1 ≤ (seqAt d r').toNat by omega) (by omega)
      omega
    · have hseq : seqAt d r = seqAt d r' := by omega
      rw [h] at h1
      have hk : chkAt d r = chkAt d r' := by omega
      exact hu (by rw [hseq, hk])
    · have := hmono (show (seqAt d r').toNat + 1 ≤ (seqAt d r).toNat by omega) (by omega)
      omega
  · have hvf : d.varlen = false := by simpa using hvl
    have hr1 : r < d.B * NTpad d := by simpa [nRows, hvf] using hr
    have hr2 : r' < d.B * NTpad d := by simpa [nRows, hvf] using hr'
    have hY : 0 < NTpad d := by
      rcases Nat.eq_zero_or_pos (NTpad d) with h | h
      · rw [h] at hr1; simp at hr1
      · exact h
    simp only [rowLo, rowHi, hvf, Bool.false_eq_true, ↓reduceIte, BT] at h1 h2
    have hm1 := Nat.mod_lt r hY
    have hm2 := Nat.mod_lt r' hY
    have hT : 0 < d.T := by
      rcases Nat.eq_zero_or_pos d.T with h | h
      · rw [h] at h1; simp at h1
      · exact h
    have hdiv := Nat.div_add_mod r (NTpad d)
    have hdiv' := Nat.div_add_mod r' (NTpad d)
    generalize r / NTpad d = a at *
    generalize r' / NTpad d = a' at *
    generalize r % NTpad d = c at *
    generalize r' % NTpad d = c' at *
    have haa : a = a' := by
      by_contra hne'
      rcases Nat.lt_or_gt_of_ne hne' with h | h
      · have : (a + 1) * d.T ≤ a' * d.T := Nat.mul_le_mul_right _ h
        have e : (a + 1) * d.T = a * d.T + d.T := by ring
        omega
      · have : (a' + 1) * d.T ≤ a * d.T := Nat.mul_le_mul_right _ h
        have e : (a' + 1) * d.T = a' * d.T + d.T := by ring
        omega
    subst haa
    generalize a * d.T = K at *
    have hcc : c = c' := by omega
    subst hcc
    exact hne (by omega)

/-! ## The tiles -/

theorem oIdx_lt {d : Inst} {n t j e : Nat} (ht : t < n) (hj : j < d.HV) (he : e < d.V) :
    oIdx d t j e < n * d.HV * d.V := by
  unfold oIdx
  have h1 : (t + 1) * d.HV ≤ n * d.HV := Nat.mul_le_mul_right _ ht
  have h2 : ((t + 1) * d.HV) * d.V ≤ n * d.HV * d.V := Nat.mul_le_mul_right _ h1
  have e1 : (t + 1) * d.HV * d.V = (t * d.HV) * d.V + d.HV * d.V := by ring
  have h3 : (j + 1) * d.V ≤ d.HV * d.V := Nat.mul_le_mul_right _ hj
  have e2 : (j + 1) * d.V = j * d.V + d.V := by ring
  have e3 : (t * d.HV + j) * d.V = t * d.HV * d.V + j * d.V := by ring
  omega

/-- An element of `o` determines its token, head and column. -/
theorem oIdx_inj {d : Inst} {t t' j j' e e' : Nat} (hj : j < d.HV) (hj' : j' < d.HV)
    (he : e < d.V) (he' : e' < d.V) (h : oIdx d t j e = oIdx d t' j' e') :
    t = t' ∧ j = j' ∧ e = e' := by
  unfold oIdx at h
  have hV : 0 < d.V := by omega
  have hH : 0 < d.HV := by omega
  have d1 : ((t * d.HV + j) * d.V + e) / d.V = t * d.HV + j := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hV, Nat.div_eq_of_lt he]; simp
  have d1' : ((t' * d.HV + j') * d.V + e') / d.V = t' * d.HV + j' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hV, Nat.div_eq_of_lt he']; simp
  have m1 : ((t * d.HV + j) * d.V + e) % d.V = e := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt he
  have m1' : ((t' * d.HV + j') * d.V + e') % d.V = e' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt he'
  rw [h] at d1 m1
  have e_eq : e = e' := m1.symm.trans m1'
  have tj : t * d.HV + j = t' * d.HV + j' := d1.symm.trans d1'
  have d2 : (t * d.HV + j) / d.HV = t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hH, Nat.div_eq_of_lt hj]; simp
  have d2' : (t' * d.HV + j') / d.HV = t' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hH, Nat.div_eq_of_lt hj']; simp
  have m2 : (t * d.HV + j) % d.HV = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hj
  have m2' : (t' * d.HV + j') % d.HV = j' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt hj'
  rw [tj] at d2 m2
  exact ⟨d2.symm.trans d2', m2.symm.trans m2', e_eq⟩

/-- Tokens of a program's tile are tokens of its chunk. -/
theorem tile_tok {d : Inst} {g : Geo} {r : Nat} (hg : Geo.Ok d g r) {k : Nat} (hk : k < BT)
    (hrow : g.rowOk k) : rowLo d r ≤ g.lo + k ∧ g.lo + k < rowHi d r := by
  rw [hg.lo_eq, hg.hi_eq]
  unfold Geo.rowOk at hrow; unfold Geo.lo
  simp only [BT] at *
  omega

/-- A tile's elements are written elements of the claim. -/
theorem tileS_written {d : Inst} {g : Geo} {r : Nat} (hg : Geo.Ok d g r) :
    tileS d g ⊆ written d := by
  rintro x ⟨k, c, hk, hc, hrow, hcol, rfl⟩
  exact ⟨r, g.lo + k, g.ih, g.iv * d.bk + c, hg.r_lt, (tile_tok hg hk hrow).1,
    (tile_tok hg hk hrow).2, hg.ih_lt, hcol, rfl⟩

/-- A tile lies inside `o`. -/
theorem tileS_lt {d : Inst} {g : Geo} {r : Nat} (hd : d.ok) (hg : Geo.Ok d g r) {x : Nat}
    (hx : x ∈ tileS d g) : x < d.o.len := by
  obtain ⟨k, c, hk, hc, hrow, hcol, rfl⟩ := hx
  have h1 : g.lo + k < d.B * d.T := by
    have := hg.end_le; unfold Geo.rowOk at hrow; unfold Geo.lo; simp only [BT] at *; omega
  have := oIdx_lt (d := d) h1 hg.ih_lt hcol
  have h2 := hd.o_len
  exact lt_of_lt_of_le this h2

/-- Different programs have disjoint tiles. -/
theorem tileS_disjoint {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {i j : Nat}
    (hi : i < grid0 d * (d.B * d.HV)) (hj : j < grid0 d * (d.B * d.HV)) (hne : i ≠ j) :
    Disjoint (tileS d (geoOf d i)) (tileS d (geoOf d j)) := by
  rw [Set.disjoint_left]
  rintro x ⟨k, c, hk, hc, hrow, hcol, rfl⟩ ⟨k', c', hk', hc', hrow', hcol', h⟩
  have hgi := geoOf_ok hd ht hi
  have hgj := geoOf_ok hd ht hj
  set gi := geoOf d i
  set gj := geoOf d j
  obtain ⟨htt, hjj, hee⟩ := oIdx_inj hgi.ih_lt hgj.ih_lt hcol hcol' h
  have hb := bk_pos hd
  have hv : gi.iv = gj.iv := by
    have : gi.iv * d.bk + c = gj.iv * d.bk + c' := hee
    have e1 : (gi.iv * d.bk + c) / d.bk = gi.iv := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hc]; simp
    have e2 : (gj.iv * d.bk + c') / d.bk = gj.iv := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hc']; simp
    rw [← e1, ← e2, this]
  by_cases hr : gi.itg = gj.itg
  · exact hne (geoOf_inj hd hi hj hv hjj hr)
  · exact chunk_disjoint hd ht hgi.r_lt hgj.r_lt hr (tile_tok hgi hk hrow) (by rw [htt]; exact tile_tok hgj hk' hrow')

/-! ## Footprints -/

theorem o_disjoint_Rs {d : Inst} (hd : d.ok) : Disjoint d.o.bytes (Rs d) := by
  rw [Set.disjoint_left]
  intro x hx hR
  rcases hR with h | h | h | h | ⟨hG, h⟩ | ⟨hG, h⟩ | ⟨hv, h | h⟩
  · exact Set.disjoint_left.mp hd.o_q hx h
  · exact Set.disjoint_left.mp hd.o_k hx h
  · exact Set.disjoint_left.mp hd.o_v hx h
  · exact Set.disjoint_left.mp hd.o_h hx h
  · exact Set.disjoint_left.mp (hd.o_g hG) hx h
  · exact Set.disjoint_left.mp (hd.o_gg hG) hx h
  · exact Set.disjoint_left.mp (hd.o_cu hv) hx h
  · exact Set.disjoint_left.mp (hd.o_ci hv) hx h

/-- A program writes inside `o`. -/
theorem Wg_sub_o {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < grid0 d * (d.B * d.HV)) : Wg d (geoOf d i) ⊆ d.o.bytes :=
  d.o.elems_sub_bytes fun _ hx => tileS_lt hd (geoOf_ok hd ht hi) hx

/-- No program writes what another reads or writes. -/
theorem w_disjoint {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {i j : Nat}
    (hi : i < grid0 d * (d.B * d.HV)) (hj : j < grid0 d * (d.B * d.HV)) (hne : i ≠ j) :
    Disjoint (Wg d (geoOf d i)) (Rs d ∪ Wg d (geoOf d j)) := by
  rw [Set.disjoint_union_right]
  exact ⟨(o_disjoint_Rs hd).mono_left (Wg_sub_o hd ht hi),
    d.o.elems_disjoint (tileS_disjoint hd ht hi hj hne)⟩

/-- No program reads what it writes. -/
theorem rw_disjoint {d : Inst} (hd : d.ok) (ht : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < grid0 d * (d.B * d.HV)) : Disjoint (Rs d) (Wg d (geoOf d i)) :=
  ((o_disjoint_Rs hd).mono_left (Wg_sub_o hd ht hi)).symm

end Claims.ChunkFwdO
