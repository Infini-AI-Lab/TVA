import Proof.Meta

/-!
# Footprint disjointness

Distinct programs write disjoint memory, and no program reads what it writes.
-/

namespace Claims.RotaryVarlen

open TVA

/-! ## Auxiliary arithmetic and index lemmas -/

private theorem g0_pos_of_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) : 0 < g0 d := by
  have h := hi
  rw [launch_size] at h
  rcases Nat.eq_zero_or_pos (g0 d) with h0 | hpos
  · rw [h0] at h; simp at h
  · exact hpos

private theorem B_pos_of_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) : 0 < d.B := by
  have h := hi
  rw [launch_size] at h
  rcases Nat.eq_zero_or_pos d.B with h0 | hpos
  · rw [h0] at h; simp at h
  · exact hpos

/-- A grid index is the mixed-radix number of its three axes. -/
private theorem ax_decomp (d : Inst) (i : Nat) :
    i = ax2 d i * (g0 d * d.B) + ax1 d i * g0 d + ax0 d i := by
  have h2 : g0 d * (i / g0 d) + i % g0 d = i := by
    rw [Nat.add_comm]
    exact Nat.mod_add_div i (g0 d)
  have h1 : d.B * ((i / g0 d) / d.B) + (i / g0 d) % d.B = i / g0 d :=
    Nat.div_add_mod (i / g0 d) d.B
  rw [Nat.div_div_eq_div_mul] at h1
  simp only [ax0, ax1, ax2]
  calc i = g0 d * (i / g0 d) + i % g0 d := h2.symm
    _ = g0 d * (d.B * (i / (g0 d * d.B)) + (i / g0 d) % d.B) + i % g0 d := by
          nth_rewrite 1 [← h1]
          rfl
    _ = i / (g0 d * d.B) * (g0 d * d.B) + i / g0 d % d.B * g0 d + i % g0 d := by ring

/-- Splitting a product decomposition: the quotient and the remainder both agree. -/
private theorem split_eq {M a b r s : Nat} (hr : r < M) (hs : s < M)
    (h : a * M + r = b * M + s) : a = b ∧ r = s := by
  have hab : a = b := by
    rcases Nat.lt_trichotomy a b with hlt | heq | hgt
    · exfalso
      have hle : a * M + M ≤ b * M := by
        have h1 : a + 1 ≤ b := hlt
        calc a * M + M = (a + 1) * M := by ring
          _ ≤ b * M := Nat.mul_le_mul_right M h1
      omega
    · exact heq
    · exfalso
      have hle : b * M + M ≤ a * M := by
        have h1 : b + 1 ≤ a := hgt
        calc b * M + M = (b + 1) * M := by ring
          _ ≤ a * M := Nat.mul_le_mul_right M h1
      omega
  exact ⟨hab, by rw [hab] at h; omega⟩

/-- The three components of a token index determine it. -/
private theorem tokIdx_inj {d : Inst} {t t' h h' c c' : Nat}
    (hh : h < d.H) (hh' : h' < d.H) (hc : c < d.D) (hc' : c' < d.D)
    (heq : tokIdx d t h c = tokIdx d t' h' c') : c = c' ∧ h = h' ∧ t = t' := by
  have hr : h * d.D + c < d.H * d.D := by
    have h1 : h + 1 ≤ d.H := hh
    calc h * d.D + c < h * d.D + d.D := by omega
      _ = (h + 1) * d.D := by ring
      _ ≤ d.H * d.D := Nat.mul_le_mul_right d.D h1
  have hr' : h' * d.D + c' < d.H * d.D := by
    have h1 : h' + 1 ≤ d.H := hh'
    calc h' * d.D + c' < h' * d.D + d.D := by omega
      _ = (h' + 1) * d.D := by ring
      _ ≤ d.H * d.D := Nat.mul_le_mul_right d.D h1
  have e1 : tokIdx d t h c = t * (d.H * d.D) + (h * d.D + c) := by
    simp only [tokIdx]; ring
  have e2 : tokIdx d t' h' c' = t' * (d.H * d.D) + (h' * d.D + c') := by
    simp only [tokIdx]; ring
  rw [e1, e2] at heq
  obtain ⟨ht, hrEq⟩ := split_eq hr hr' heq
  obtain ⟨hhEq, hcEq⟩ := split_eq hc hc' hrEq
  exact ⟨hcEq, hhEq, ht⟩

/-! ## Token indices of distinct lanes -/

/-- On the padded path, two equal lane tokens come from the same program. -/
private theorem padded_tok_eq {d : Inst} (hv : d.isVarlen = false) {init : Mem} {i j l1 l2 : Nat}
    (hl1 : l1 < d.BT) (hl2 : l2 < d.BT)
    (hon1 : laneOn d init i l1) (hon2 : laneOn d init j l2)
    (heq : tokI d init i l1 = tokI d init j l2) :
    ax0 d i = ax0 d j ∧ ax1 d i = ax1 d j := by
  have hbos1 : bosI d init i = ax1 d i * d.T := by simp [bosI, bosAt, hv, seqI_false hv]
  have hbos2 : bosI d init j = ax1 d j * d.T := by simp [bosI, bosAt, hv, seqI_false hv]
  have hloc1 : locI d init i l1 = ax0 d i * d.BT + l1 := by simp [locI, chunkI_false hv]
  have hloc2 : locI d init j l2 = ax0 d j * d.BT + l2 := by simp [locI, chunkI_false hv]
  have hlen1 : lenI d init i = d.T := by simp [lenI, lenAt, hv]
  have hlen2 : lenI d init j = d.T := by simp [lenI, lenAt, hv]
  have hlt1 : ax0 d i * d.BT + l1 < d.T := by
    have h := hon1.1
    rw [hlen1, hloc1] at h
    exact Int.ofNat_lt.mp h
  have hlt2 : ax0 d j * d.BT + l2 < d.T := by
    have h := hon2.1
    rw [hlen2, hloc2] at h
    exact Int.ofNat_lt.mp h
  have he : ax1 d i * d.T + (ax0 d i * d.BT + l1)
      = ax1 d j * d.T + (ax0 d j * d.BT + l2) := by
    have h := heq
    simp only [tokI, hbos1, hbos2, hloc1, hloc2] at h
    exact h
  obtain ⟨hax1, hr⟩ := split_eq hlt1 hlt2 he
  obtain ⟨hax0, -⟩ := split_eq hl1 hl2 hr
  exact ⟨hax0, hax1⟩

/-- On the packed path, two equal lane tokens come from the same program's axes. -/
private theorem packed_tok_eq {d : Inst} (_hd : d.ok) (hv : d.isVarlen = true) {init : Mem}
    (hs : SeqOk d init) {i j l1 l2 : Nat}
    (hi : i < (launch d).size) (hj : j < (launch d).size)
    (hl1 : l1 < d.BT) (hl2 : l2 < d.BT)
    (hon1 : laneOn d init i l1) (hon2 : laneOn d init j l2)
    (heq : tokI d init i l1 = tokI d init j l2) :
    ax0 d i = ax0 d j := by
  have hg0 : g0 d = d.NT := by simp [g0, hv]
  have hax1 : ax0 d i < d.NT := by rw [← hg0]; exact ax0_lt hi
  have hax2 : ax0 d j < d.NT := by rw [← hg0]; exact ax0_lt hj
  have hci1 := hs.ci_ok (ax0 d i) hax1
  have hci2 := hs.ci_ok (ax0 d j) hax2
  have hseq1 : seqI d init i = chSeq d init (ax0 d i) := seqI_true hv init i
  have hseq2 : seqI d init j = chSeq d init (ax0 d j) := seqI_true hv init j
  have hbos1 : bosI d init i = cuN d init (chSeq d init (ax0 d i)) := by
    simp [bosI, bosAt, hv, hseq1]
  have hbos2 : bosI d init j = cuN d init (chSeq d init (ax0 d j)) := by
    simp [bosI, bosAt, hv, hseq2]
  have hlen1 : lenI d init i = seqLen d init (chSeq d init (ax0 d i)) := by
    simp [lenI, lenAt, hv, hseq1]
  have hlen2 : lenI d init j = seqLen d init (chSeq d init (ax0 d j)) := by
    simp [lenI, lenAt, hv, hseq2]
  have hloc1 : locI d init i l1 = chChunk d init (ax0 d i) * d.BT + l1 := by
    simp [locI, chunkI_true hv]
  have hloc2 : locI d init j l2 = chChunk d init (ax0 d j) * d.BT + l2 := by
    simp [locI, chunkI_true hv]
  have hlt1 : chChunk d init (ax0 d i) * d.BT + l1
      < seqLen d init (chSeq d init (ax0 d i)) := by
    have h := hon1.1
    rw [hlen1, hloc1] at h
    exact Int.ofNat_lt.mp h
  have hlt2 : chChunk d init (ax0 d j) * d.BT + l2
      < seqLen d init (chSeq d init (ax0 d j)) := by
    have h := hon2.1
    rw [hlen2, hloc2] at h
    exact Int.ofNat_lt.mp h
  have hmono1 : cuN d init (chSeq d init (ax0 d i))
      ≤ cuN d init (chSeq d init (ax0 d i) + 1) := hs.cu_mono _ hci1.1
  have hmono2 : cuN d init (chSeq d init (ax0 d j))
      ≤ cuN d init (chSeq d init (ax0 d j) + 1) := hs.cu_mono _ hci2.1
  have htok1 : tokI d init i l1
      = cuN d init (chSeq d init (ax0 d i)) + (chChunk d init (ax0 d i) * d.BT + l1) := by
    simp only [tokI, hbos1, hloc1]
  have htok2 : tokI d init j l2
      = cuN d init (chSeq d init (ax0 d j)) + (chChunk d init (ax0 d j) * d.BT + l2) := by
    simp only [tokI, hbos2, hloc2]
  have hb1 : cuN d init (chSeq d init (ax0 d i)) ≤ tokI d init i l1 := by
    rw [htok1]; omega
  have hb2 : tokI d init i l1 < cuN d init (chSeq d init (ax0 d i) + 1) := by
    rw [htok1]
    have hsL : seqLen d init (chSeq d init (ax0 d i))
        = cuN d init (chSeq d init (ax0 d i) + 1)
          - cuN d init (chSeq d init (ax0 d i)) := rfl
    omega
  have hb3 : cuN d init (chSeq d init (ax0 d j)) ≤ tokI d init j l2 := by
    rw [htok2]; omega
  have hb4 : tokI d init j l2 < cuN d init (chSeq d init (ax0 d j) + 1) := by
    rw [htok2]
    have hsL : seqLen d init (chSeq d init (ax0 d j))
        = cuN d init (chSeq d init (ax0 d j) + 1)
          - cuN d init (chSeq d init (ax0 d j)) := rfl
    omega
  have hn : chSeq d init (ax0 d i) = chSeq d init (ax0 d j) := by
    rcases Nat.lt_trichotomy (chSeq d init (ax0 d i)) (chSeq d init (ax0 d j)) with
      hlt | heq' | hgt
    · exfalso
      have hle : chSeq d init (ax0 d i) + 1 ≤ chSeq d init (ax0 d j) := hlt
      have hm : cuN d init (chSeq d init (ax0 d i) + 1)
          ≤ cuN d init (chSeq d init (ax0 d j)) :=
        cuN_mono_le hs hle (by omega)
      omega
    · exact heq'
    · exfalso
      have hle : chSeq d init (ax0 d j) + 1 ≤ chSeq d init (ax0 d i) := hgt
      have hm : cuN d init (chSeq d init (ax0 d j) + 1)
          ≤ cuN d init (chSeq d init (ax0 d i)) :=
        cuN_mono_le hs hle (by omega)
      omega
  have hloc_eq : chChunk d init (ax0 d i) * d.BT + l1
      = chChunk d init (ax0 d j) * d.BT + l2 := by
    have h := heq
    simp only [tokI, hbos1, hbos2, hloc1, hloc2] at h
    rw [hn] at h
    omega
  obtain ⟨hchunk, -⟩ := split_eq hl1 hl2 hloc_eq
  exact hs.ci_once (ax0 d i) (ax0 d j) hax1 hax2 hn hchunk

/-- Two lanes of two distinct programs never write the same `y` element. -/
private theorem lane_ne {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j l1 l2 c1 c2 : Nat} (hi : i < (launch d).size) (hj : j < (launch d).size)
    (hij : i ≠ j) (hl1 : l1 < d.BT) (hl2 : l2 < d.BT)
    (hc1 : c1 < 2 * d.R) (hc2 : c2 < 2 * d.R)
    (hon1 : laneOn d init i l1) (hon2 : laneOn d init j l2)
    (heq : tokIdx d (tokI d init i l1) (ax2 d i) c1 =
           tokIdx d (tokI d init j l2) (ax2 d j) c2) : False := by
  have hHi : ax2 d i < d.H := ax2_lt hi
  have hHj : ax2 d j < d.H := ax2_lt hj
  have hcD1 : c1 < d.D := lt_of_lt_of_le hc1 hd.r_d
  have hcD2 : c2 < d.D := lt_of_lt_of_le hc2 hd.r_d
  obtain ⟨-, hH, hT⟩ := tokIdx_inj hHi hHj hcD1 hcD2 heq
  have hax : ax0 d i = ax0 d j ∧ ax1 d i = ax1 d j := by
    by_cases hv : d.isVarlen = true
    · have hs : SeqOk d init := hpre.seqs hv
      have h0 : ax0 d i = ax0 d j := packed_tok_eq hd hv hs hi hj hl1 hl2 hon1 hon2 hT
      refine ⟨h0, ?_⟩
      have hB1 : d.B = 1 := hd.varlen_b hv
      simp [ax1, hB1, Nat.mod_one]
    · have hv2 : d.isVarlen = false := by cases h : d.isVarlen <;> simp_all
      exact padded_tok_eq hv2 hl1 hl2 hon1 hon2 hT
  obtain ⟨hax0, hax1⟩ := hax
  apply hij
  have e1 : i = ax2 d i * (g0 d * d.B) + ax1 d i * g0 d + ax0 d i := ax_decomp d i
  have e2 : j = ax2 d j * (g0 d * d.B) + ax1 d j * g0 d + ax0 d j := ax_decomp d j
  have hF : ax2 d i * (g0 d * d.B) + ax1 d i * g0 d + ax0 d i
      = ax2 d j * (g0 d * d.B) + ax1 d j * g0 d + ax0 d j := by rw [hax0, hax1, hH]
  calc i = ax2 d i * (g0 d * d.B) + ax1 d i * g0 d + ax0 d i := e1
    _ = ax2 d j * (g0 d * d.B) + ax1 d j * g0 d + ax0 d j := hF
    _ = j := e2.symm

/-! ## Footprints -/

/-- A program only writes inside the output buffer. -/
private theorem W_sub_y {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) : W d init i ⊆ d.y.bytes := by
  apply Buf.elems_sub_bytes
  rintro k ⟨l, c, hl, hc, hon, rfl⟩
  have htok := tokI_lt hd hpre hi hon
  have hH := ax2_lt hi
  have hcD : c < d.D := lt_of_lt_of_le hc hd.r_d
  have h1 : tokIdx d (tokI d init i l) (ax2 d i) c < nTok d * d.H * d.D := by
    have hh : tokI d init i l * d.H + ax2 d i < nTok d * d.H := by
      calc tokI d init i l * d.H + ax2 d i < tokI d init i l * d.H + d.H := by omega
        _ = (tokI d init i l + 1) * d.H := by ring
        _ ≤ nTok d * d.H := Nat.mul_le_mul_right d.H (by omega)
    calc tokIdx d (tokI d init i l) (ax2 d i) c
        = (tokI d init i l * d.H + ax2 d i) * d.D + c := rfl
      _ < (tokI d init i l * d.H + ax2 d i) * d.D + d.D := by omega
      _ = (tokI d init i l * d.H + ax2 d i + 1) * d.D := by ring
      _ ≤ nTok d * d.H * d.D := Nat.mul_le_mul_right d.D (by omega)
  exact lt_of_lt_of_le h1 hd.y_len

/-- The output buffer is disjoint from every input buffer. -/
private theorem y_disj_inputs {d : Inst} (hd : d.ok) :
    Disjoint (d.x.bytes ∪ d.cos.bytes ∪ d.sin.bytes ∪ d.cu.bytes ∪ d.ci.bytes ∪ d.offs.bytes)
      d.y.bytes := by
  rw [Set.disjoint_left]
  intro z hz hy
  simp only [Set.mem_union] at hz
  rcases hz with ((((hx | hcos) | hsin) | hcu) | hci) | hoffs
  · exact Set.disjoint_left.mp hd.y_x hy hx
  · exact Set.disjoint_left.mp hd.y_cos hy hcos
  · exact Set.disjoint_left.mp hd.y_sin hy hsin
  · exact Set.disjoint_left.mp hd.y_cu hy hcu
  · exact Set.disjoint_left.mp hd.y_ci hy hci
  · exact Set.disjoint_left.mp hd.y_offs hy hoffs

/-- Distinct programs write disjoint memory. -/
theorem foot_disj {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Nat} (hi : i < (launch d).size) (hj : j < (launch d).size) (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hyW : Disjoint
      (d.x.bytes ∪ d.cos.bytes ∪ d.sin.bytes ∪ d.cu.bytes ∪ d.ci.bytes ∪ d.offs.bytes)
      (W d init i) :=
    (y_disj_inputs hd).mono_right (W_sub_y hd hpre hi)
  have hWR : Disjoint (W d init i) (R d init j) := hyW.symm
  have hWW : Disjoint (W d init i) (W d init j) := by
    apply Buf.elems_disjoint
    rw [Set.disjoint_left]
    rintro k ⟨l1, c1, hl1, hc1, hon1, rfl⟩ ⟨l2, c2, hl2, hc2, hon2, hk2⟩
    exact lane_ne hd hpre hi hj hij hl1 hl2 hc1 hc2 hon1 hon2 hk2
  rw [Set.disjoint_left] at hWR hWW ⊢
  intro z hz hzu
  rcases hzu with hzR | hzW
  · exact hWR hz hzR
  · exact hWW hz hzW

/-- A program does not read what it writes. -/
theorem foot_rw {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) :
    Disjoint (R d init i) (W d init i) :=
  (y_disj_inputs hd).mono_right (W_sub_y hd hpre hi)

end Claims.RotaryVarlen
