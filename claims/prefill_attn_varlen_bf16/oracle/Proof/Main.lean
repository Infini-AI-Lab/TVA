import Proof.Program

section
/-!
# The proof of `prefill_attn_varlen_bf16`

Program `i` of the grid owns the tile `tile d init (bOf i) (hOf i) (mOf i)` of `o`: the rows of
query block `mOf` of sequence `bOf`, head `hOf`. Distinct programs own distinct triples, and
distinct triples own disjoint tiles (the sequences are packed, `o`'s elements are distinct), so
the footprints are disjoint and `correct_of_safe` composes `safe_program` over the grid.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-! ## Layout of `o` -/

private theorem divmod_inj {n t t' r r' : Nat} (hr : r < n) (hr' : r' < n)
    (h : t * n + r = t' * n + r') : t = t' ∧ r = r' := by
  have h1 : (t * n + r) / n = t := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hr]; rfl
  have h2 : (t' * n + r') / n = t' := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hr']; rfl
  have ht : t = t' := by rw [← h1, ← h2, h]
  subst ht
  exact ⟨rfl, by omega⟩

/-- **`o`'s elements are distinct**: two in-range elements with the same index are equal. -/
theorem oIdx_inj {d : Inst} (hd : d.ok) {t t' h h' e e' : Nat} (ht : t < d.total)
    (ht' : t' < d.total) (hh : h < d.H) (hh' : h' < d.H) (he : e < d.Lk) (he' : e' < d.Lk)
    (heq : oIdx d t h e = oIdx d t' h' e') : t = t' ∧ h = h' ∧ e = e' := by
  simp only [oIdx] at heq
  rcases hd.o_layout with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · have hs : h * d.soh + e < d.sot := by
      have : (h + 1) * d.soh ≤ d.H * d.soh := Nat.mul_le_mul_right _ hh
      rw [Nat.add_mul] at this
      omega
    have hs' : h' * d.soh + e' < d.sot := by
      have : (h' + 1) * d.soh ≤ d.H * d.soh := Nat.mul_le_mul_right _ hh'
      rw [Nat.add_mul] at this
      omega
    obtain ⟨rfl, hr⟩ := divmod_inj (t := t) (t' := t') hs hs' (by linarith)
    obtain ⟨rfl, rfl⟩ := divmod_inj (n := d.soh) (t := h) (t' := h') (he.trans_le h1) (he'.trans_le h1) hr
    exact ⟨rfl, rfl, rfl⟩
  · have hs : t * d.sot + e < d.soh := by
      have : (t + 1) * d.sot ≤ d.total * d.sot := Nat.mul_le_mul_right _ ht
      rw [Nat.add_mul] at this
      omega
    have hs' : t' * d.sot + e' < d.soh := by
      have : (t' + 1) * d.sot ≤ d.total * d.sot := Nat.mul_le_mul_right _ ht'
      rw [Nat.add_mul] at this
      omega
    obtain ⟨rfl, hr⟩ := divmod_inj (t := h) (t' := h') hs hs' (by linarith)
    obtain ⟨rfl, rfl⟩ := divmod_inj (n := d.sot) (t := t) (t' := t') (he.trans_le h1)
      (he'.trans_le h1) hr
    exact ⟨rfl, rfl, rfl⟩

/-- An in-range element of `o` lies inside its storage. -/
theorem oIdx_lt {d : Inst} (hd : d.ok) {t h e : Nat} (ht : t < d.total) (hh : h < d.H)
    (he : e < d.Lk) : oIdx d t h e < d.o.len := by
  have hl := hd.o_len (by omega)
  have h1 : t * d.sot ≤ (d.total - 1) * d.sot := Nat.mul_le_mul_right _ (by omega)
  have h2 : h * d.soh ≤ (d.H - 1) * d.soh := Nat.mul_le_mul_right _ (by omega)
  simp only [oIdx]
  omega

/-! ## The sequences -/

/-- A sequence of a packed batch lies inside the tokens and within `max_input_len`. -/
theorem seq_bounds {d : Inst} {init : Mem} (hs : SeqOk d init) {b : Nat} (hb : b < d.batch) :
    startOf d init b + lenOf d init b ≤ d.total ∧ lenOf d init b ≤ d.maxLen := by
  obtain ⟨s, n, h1, h2, h3, h4, h5, h6⟩ := hs.1 b hb
  simp only [startOf, lenOf, val32, h1, h2]
  omega

/-- Different query rows of different sequences, or of one sequence, are different tokens. -/
theorem token_inj {d : Inst} {init : Mem} (hs : SeqOk d init) {b b' p p' : Nat} (hb : b < d.batch)
    (hb' : b' < d.batch) (hp : p < lenOf d init b) (hp' : p' < lenOf d init b')
    (heq : startOf d init b + p = startOf d init b' + p') : b = b' ∧ p = p' := by
  by_cases hbb : b = b'
  · subst hbb; exact ⟨rfl, by omega⟩
  · have := hs.2 b b' hb hb' hbb (by omega) (by omega)
    omega

/-! ## The grid -/

/-- A program of the grid is determined by its triple `(bOf, hOf, mOf)`. -/
theorem grid_inj {d : Inst} {i j : Nat} (hb : bOf d i = bOf d j) (hh : hOf d i = hOf d j)
    (hm : mOf d i = mOf d j) : i = j := by
  simp only [bOf, hOf, mOf] at hb hh hm
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul] at hm
  have hq : i / d.batch = j / d.batch :=
    calc i / d.batch = i / d.batch % d.H + d.H * (i / d.batch / d.H) := (Nat.mod_add_div _ _).symm
      _ = j / d.batch % d.H + d.H * (j / d.batch / d.H) := by rw [hh, hm]
      _ = j / d.batch := Nat.mod_add_div _ _
  calc i = i % d.batch + d.batch * (i / d.batch) := (Nat.mod_add_div _ _).symm
    _ = j % d.batch + d.batch * (j / d.batch) := by rw [hb, hq]
    _ = j := Nat.mod_add_div _ _

/-- A program of the grid has a sequence and a head. -/
theorem grid_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) : bOf d i < d.batch ∧ hOf d i < d.H := by
  rw [launch_size] at hi
  have hb : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hi; simp at hi
    · exact h
  exact ⟨Nat.mod_lt _ hb, Nat.mod_lt _ hH⟩

/-- The program that owns query `p` of sequence `b`, head `h`. -/
theorem grid_of {d : Inst} {b h p : Nat} (hb : b < d.batch) (hh : h < d.H)
    (hp : p < d.maxLen) :
    ∃ i : Nat, i < (launch d).size ∧ bOf d i = b ∧ hOf d i = h ∧ mOf d i = p / BLOCK := by
  set m := p / BLOCK with hm
  have hmn : m < nBlocks d := by
    simp only [hm, nBlocks, BLOCK]; omega
  refine ⟨b + d.batch * (h + d.H * m), ?_, ?_, ?_, ?_⟩
  · rw [launch_size]
    have h1 : h + d.H * m < d.H * nBlocks d := by
      have : d.H * (m + 1) ≤ d.H * nBlocks d := Nat.mul_le_mul_left _ hmn
      rw [Nat.mul_add] at this
      omega
    have h2 : d.batch * (h + d.H * m + 1) ≤ d.batch * (d.H * nBlocks d) :=
      Nat.mul_le_mul_left _ h1
    rw [Nat.mul_add, Nat.mul_one] at h2
    rw [Nat.mul_assoc]
    omega
  · simp only [bOf]; rw [Nat.add_mul_mod_self_left]; exact Nat.mod_eq_of_lt hb
  · have : (b + d.batch * (h + d.H * m)) / d.batch = h + d.H * m := by
      rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hb, Nat.zero_add]
    simp only [hOf, this]
    rw [Nat.add_mul_mod_self_left]; exact Nat.mod_eq_of_lt hh
  · have : (b + d.batch * (h + d.H * m)) / d.batch = h + d.H * m := by
      rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hb, Nat.zero_add]
    simp only [mOf]
    rw [← Nat.div_div_eq_div_mul, this, Nat.add_mul_div_left _ _ (by omega),
      Nat.div_eq_of_lt hh, Nat.zero_add]

/-! ## Footprints -/

/-- Every element of a tile is a token of the batch: it lies inside `o`. -/
theorem tile_lt {d : Inst} (hd : d.ok) {init : Mem} (hs : SeqOk d init) {b h sm : Nat}
    (hb : b < d.batch) (hh : h < d.H) {x : Nat} (hx : x ∈ tile d init b h sm) : x < d.o.len := by
  obtain ⟨r, e, hr, he, hl, rfl⟩ := hx
  have := seq_bounds hs hb
  exact oIdx_lt hd (by omega) hh he

/-- A tile only holds elements the launch writes. -/
theorem tile_sub {d : Inst} {init : Mem} {b h sm : Nat} (hb : b < d.batch) (hh : h < d.H) :
    tile d init b h sm ⊆ outElems d init := by
  rintro x ⟨r, e, hr, he, hl, rfl⟩
  exact ⟨b, sm * BLOCK + r, h, e, hb, hl, hh, he, rfl⟩

/-- The tiles of different triples are disjoint. -/
theorem tile_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hs : SeqOk d init) {b b' h h' sm sm' : Nat}
    (hb : b < d.batch) (hb' : b' < d.batch) (hh : h < d.H) (hh' : h' < d.H)
    (hne : ¬ (b = b' ∧ h = h' ∧ sm = sm')) :
    Disjoint (tile d init b h sm) (tile d init b' h' sm') := by
  rw [Set.disjoint_left]
  rintro x ⟨r, e, hr, he, hl, rfl⟩ ⟨r', e', hr', he', hl', hx⟩
  have s1 := seq_bounds hs hb
  have s2 := seq_bounds hs hb'
  obtain ⟨ht, rfl, rfl⟩ := oIdx_inj hd (by omega) (by omega) hh hh' he he' hx
  obtain ⟨rfl, hp⟩ := token_inj hs hb hb' hl hl' ht
  simp only [BLOCK] at hr hr' hp
  exact hne ⟨rfl, rfl, by omega⟩

/-- Programs `i ≠ j` of the grid have different triples. -/
theorem triple_ne {d : Inst} {i j : Nat} (hij : i ≠ j) :
    ¬ (bOf d i = bOf d j ∧ hOf d i = hOf d j ∧ mOf d i = mOf d j) :=
  fun ⟨h1, h2, h3⟩ => hij (grid_inj h1 h2 h3)

/-- The writes of a program lie inside `o`, and so are disjoint from every input. -/
theorem Wf_disjoint_Rf {d : Inst} (hd : d.ok) {init : Mem} (hs : SeqOk d init)
    (i : Fin (launch d).size) : Disjoint (Wf d init i) (Rf d) := by
  obtain ⟨hb, hh⟩ := grid_lt i.isLt
  have hsub : Wf d init i ⊆ d.o.bytes :=
    d.o.elems_sub_bytes fun _ hx => tile_lt hd hs hb hh hx
  obtain ⟨h1, h2, h3, h4, h5⟩ := hd.disj
  simp only [Rf, Set.disjoint_union_right]
  exact ⟨⟨⟨⟨(h1.mono_left hsub), (h2.mono_left hsub)⟩, (h3.mono_left hsub)⟩,
    (h4.mono_left hsub)⟩, (h5.mono_left hsub)⟩

/-- The writes of different programs are disjoint. -/
theorem Wf_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hs : SeqOk d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) : Disjoint (Wf d init i) (Wf d init j) := by
  obtain ⟨hb, hh⟩ := grid_lt i.isLt
  obtain ⟨hb', hh'⟩ := grid_lt j.isLt
  refine d.o.elems_disjoint (tile_disjoint hd hs hb hb' hh hh' fun ⟨h1, h2, h3⟩ => ?_)
  exact triple_ne (fun h => hij (Fin.ext h)) ⟨h1, h2, h3⟩

/-! ## The claim -/

/-- **The claim.** Each program is safe on its own tile (`safe_program`); the tiles' footprints
are disjoint, and every output element lies in the tile of the program of its sequence, head and
query block. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hs := hpre.2.2.2.2.2.2.1
  refine correct_of_safe (fun _ => Rf d) (fun i => Wf d init i) (fun i => Q d init i) (launch_valid hd)
    (fun i => safe_program hP hd hpre i) ?_ ?_ ?_
  · intro i j hij
    exact Disjoint.union_right (Wf_disjoint_Rf hd hs i) (Wf_disjoint hd hs hij)
  · intro i
    exact (Wf_disjoint_Rf hd hs i).symm
  · intro fin hloc hfr
    refine ⟨fun b p h e hb hp hh he => ?_, fun z hz => hfr z fun i hzi => ?_⟩
    · have hbd := seq_bounds hs hb
      obtain ⟨i, hi, hbi, hhi, hmi⟩ := grid_of hb hh (p := p) (by omega)
      obtain ⟨m, hQ, hag⟩ := hloc ⟨i, hi⟩
      have hpos : p / BLOCK * BLOCK + p % BLOCK = p := by simp only [BLOCK]; omega
      obtain ⟨c, hc, hfin, hbound⟩ := hQ (p % BLOCK) e (by simp only [BLOCK]; omega) he (by
        show mOf d i * BLOCK + _ < lenOf d init (bOf d i)
        rw [hmi, hbi, hpos]; exact hp)
      refine ⟨c, ?_, hfin, ?_⟩
      · rw [← hc]
        simp only [rdB, hbi, hhi, hmi, hpos] at hc ⊢
        refine Mem.read_congr fun x hx => hag x (Or.inr ?_)
        refine d.o.span_sub_elems ?_ hx
        refine ⟨p % BLOCK, e, by simp only [BLOCK]; omega, he, ?_, ?_⟩
        · show mOf d i * BLOCK + _ < lenOf d init (bOf d i)
          rw [hmi, hbi, hpos]; exact hp
        · show _ = oIdx d (startOf d init (bOf d i) + (mOf d i * BLOCK + p % BLOCK)) (hOf d i) e
          rw [hmi, hbi, hhi, hpos]
      · simpa only [hbi, hhi, hmi, hpos] using hbound
    · obtain ⟨hb, hh⟩ := grid_lt i.isLt
      exact hz (d.o.elems_mono (tile_sub hb hh) hzi)

end Claims.PrefillAttnVarlen

end
