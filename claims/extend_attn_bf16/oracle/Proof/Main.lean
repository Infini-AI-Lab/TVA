import Proof.Program

/-!
# The proof of `extend_attn_bf16`

Program `i` of the grid owns the tile `tile d init (bOf i) (hOf i) (mOf i)` of `o_extend`: the rows
of query block `mOf` of request `bOf`, head `hOf`. Distinct programs own distinct triples, and
distinct triples own disjoint tiles (the requests' tokens are disjoint, `o_extend`'s elements are
distinct), so the footprints are disjoint and `correct_of_safe` composes `safe_program` over the
grid.
-/

namespace Claims.ExtendAttn

open TVA

/-! ## The requests' tokens -/

/-- Request `b` has `0 ≤ qo_indptr[b] ≤ qo_indptr[b + 1]`. -/
theorem valQo_step {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.batch) :
    0 ≤ valQo d init b ∧ valQo d init b ≤ valQo d init (b + 1) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hreq, -⟩ := hpre
  obtain ⟨q0, q1, -, -, hq0, hq1, -, -, h0, h01, -⟩ := hreq b hb
  have e0 : valQo d init b = q0 := by simp [valQo, hq0]
  have e1 : valQo d init (b + 1) = q1 := by simp [valQo, hq1]
  rw [e0, e1]; exact ⟨h0, h01⟩

/-- `qo_indptr` is monotone along the requests. -/
theorem valQo_mono {d : Inst} {init : Mem} (hpre : Pre d init) {b b' : Nat} (hbb : b ≤ b')
    (hb' : b' ≤ d.batch) : valQo d init b ≤ valQo d init b' := by
  induction b', hbb using Nat.le_induction with
  | base => exact le_rfl
  | succ n hn ih =>
    exact (ih (by omega)).trans (valQo_step hpre (by omega)).2

/-- Request `b`'s tokens end where its `qo_indptr[b + 1]` says. -/
theorem qs_ext {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.batch) :
    (qsOf d init b : Int) + extOf d init b = valQo d init (b + 1) := by
  have := valQo_step hpre hb
  unfold qsOf extOf; omega

/-- Different tokens of different requests (or of one request) are different. -/
theorem token_inj {d : Inst} {init : Mem} (hpre : Pre d init) {b b' p p' : Nat}
    (hb : b < d.batch) (hb' : b' < d.batch) (hp : p < extOf d init b)
    (hp' : p' < extOf d init b') (heq : qsOf d init b + p = qsOf d init b' + p') :
    b = b' ∧ p = p' := by
  by_cases hbb : b = b'
  · subst hbb; exact ⟨rfl, by omega⟩
  · exfalso
    rcases Nat.lt_or_gt_of_ne hbb with h | h
    · have h1 := qs_ext hpre hb
      have h2 := valQo_mono hpre (show b + 1 ≤ b' by omega) hb'.le
      have h3 := (valQo_step hpre hb').1
      have : qsOf d init b' = (valQo d init b').toNat := rfl
      omega
    · have h1 := qs_ext hpre hb'
      have h2 := valQo_mono hpre (show b' + 1 ≤ b by omega) hb.le
      have h3 := (valQo_step hpre hb).1
      have : qsOf d init b = (valQo d init b).toNat := rfl
      omega

/-! ## The grid -/

/-- A program of the grid is determined by its triple `(bOf, hOf, mOf)`. -/
theorem grid_inj {d : Inst} {i j : Nat} (hb : bOf d i = bOf d j) (hh : hOf d i = hOf d j)
    (hm : mOf d i = mOf d j) : i = j := by
  simp only [bOf, hOf, mOf] at hb hh hm
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul] at hm
  have hq : i / d.batch = j / d.batch :=
    calc i / d.batch = i / d.batch % d.H + d.H * (i / d.batch / d.H) :=
          (Nat.mod_add_div _ _).symm
      _ = j / d.batch % d.H + d.H * (j / d.batch / d.H) := by rw [hh, hm]
      _ = j / d.batch := Nat.mod_add_div _ _
  calc i = i % d.batch + d.batch * (i / d.batch) := (Nat.mod_add_div _ _).symm
    _ = j % d.batch + d.batch * (j / d.batch) := by rw [hb, hq]
    _ = j := Nat.mod_add_div _ _

/-- The program that owns block `m` of request `b`, head `h`. -/
theorem grid_of {d : Inst} {b h m : Nat} (hb : b < d.batch) (hh : h < d.H) (hmn : m < nBlocks d) :
    ∃ i : Nat, i < (launch d).size ∧ bOf d i = b ∧ hOf d i = h ∧ mOf d i = m := by
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

/-- An element of `o_extend` that is a token of a request lies inside its storage. -/
theorem oIdx_lt {d : Inst} (hd : d.ok) {t h e : Nat} (ht : t < d.T) (hh : h < d.H)
    (he : e < d.D) : oIdx d t h e < d.o.len := by
  have hl := hd.o_len (by omega)
  have h1 : t * d.sot ≤ (d.T - 1) * d.sot := Nat.mul_le_mul_right _ (by omega)
  have h2 : h * d.soh ≤ (d.H - 1) * d.soh := Nat.mul_le_mul_right _ (by omega)
  simp only [oIdx]
  omega

/-- Every element of a tile lies inside `o_extend`. -/
theorem tile_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b h m : Nat}
    (hb : b < d.batch) (hh : h < d.H) {x : Nat} (hx : x ∈ tile d init b h m) :
    x < d.o.len := by
  obtain ⟨r, e, hr, he, hl, rfl⟩ := hx
  have := (req_sizes hpre hb).1
  exact oIdx_lt hd (by omega) hh he

/-- A tile only holds elements the launch writes. -/
theorem tile_sub {d : Inst} {init : Mem} {b h m : Nat} (hb : b < d.batch) (hh : h < d.H) :
    tile d init b h m ⊆ outElems d init := by
  rintro x ⟨r, e, hr, he, hl, rfl⟩
  exact ⟨b, m * d.BM + r, h, e, hb, hl, hh, he, rfl⟩

/-- The tiles of different triples are disjoint. -/
theorem tile_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {b b' h h' m m' : Nat} (hb : b < d.batch) (hb' : b' < d.batch) (hh : h < d.H)
    (hh' : h' < d.H) (hne : ¬ (b = b' ∧ h = h' ∧ m = m')) :
    Disjoint (tile d init b h m) (tile d init b' h' m') := by
  rw [Set.disjoint_left]
  rintro x ⟨r, e, hr, he, hl, rfl⟩ ⟨r', e', hr', he', hl', hx⟩
  have s1 := (req_sizes hpre hb).1
  have s2 := (req_sizes hpre hb').1
  obtain ⟨ht, rfl, rfl⟩ := epi_oIdx_inj hd (by omega) (by omega) hh hh' he he' hx
  obtain ⟨rfl, hp⟩ := token_inj hpre hb hb' hl hl' ht
  have hbm : 0 < d.BM := by rw [hd.bm]; split_ifs <;> omega
  have : m = m' := by
    have e1 : (m * d.BM + r) / d.BM = m := by
      rw [Nat.mul_comm, Nat.mul_add_div hbm, Nat.div_eq_of_lt hr]; rfl
    have e2 : (m' * d.BM + r') / d.BM = m' := by
      rw [Nat.mul_comm, Nat.mul_add_div hbm, Nat.div_eq_of_lt hr']; rfl
    rw [← e1, ← e2, hp]
  exact hne ⟨rfl, rfl, this⟩

/-- A program of the grid has a request and a head. -/
theorem grid_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) :
    bOf d i < d.batch ∧ hOf d i < d.H := by
  obtain ⟨h1, h2, -⟩ := prog_range hi
  exact ⟨h1, h2⟩

/-- The writes of a program lie inside `o_extend`, so are disjoint from every input it reads. -/
theorem Wf_disjoint_Rf {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) : Disjoint (Wf d init i) (Rf d) := by
  obtain ⟨hb, hh⟩ := grid_lt i.isLt
  have hsub : Wf d init i ⊆ d.o.bytes :=
    d.o.elems_sub_bytes fun _ hx => tile_lt hd hpre hb hh hx
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := hd.disj
  simp only [Rf, Set.disjoint_union_right]
  refine ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1.mono_left hsub, h2.mono_left hsub⟩, h3.mono_left hsub⟩, h4.mono_left hsub⟩,
    h5.mono_left hsub⟩, h6.mono_left hsub⟩, h7.mono_left hsub⟩, h8.mono_left hsub⟩, ?_⟩, ?_⟩
  · split_ifs with hc
    · obtain ⟨c1, c2, -⟩ := h9 hc
      exact Set.disjoint_union_right.2 ⟨c1.mono_left hsub, c2.mono_left hsub⟩
    · simp
  · split_ifs with hc
    · exact ((h9 hc.1).2.2 hc.2).mono_left hsub
    · simp

/-- The writes of different programs are disjoint. -/
theorem Wf_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) : Disjoint (Wf d init i) (Wf d init j) := by
  obtain ⟨hb, hh⟩ := grid_lt i.isLt
  obtain ⟨hb', hh'⟩ := grid_lt j.isLt
  refine d.o.elems_disjoint (tile_disjoint hd hpre hb hb' hh hh' fun ⟨h1, h2, h3⟩ => ?_)
  exact hij (Fin.ext (grid_inj h1 h2 h3))

/-! ## The claim -/

/-- **The claim.** Each program is safe on its own tile (`safe_program`); the tiles' footprints
are disjoint, and every output element lies in the tile of the program of its request, head and
query block. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => Rf d) (fun i => Wf d init i) (fun i => Q d init i)
    (launch_valid hd) (fun i => safe_program hP hd hpre i) ?_ ?_ ?_
  · intro i j hij
    exact Disjoint.union_right (Wf_disjoint_Rf hd hpre i) (Wf_disjoint hd hpre hij)
  · intro i
    exact (Wf_disjoint_Rf hd hpre i).symm
  · intro fin hloc hfr
    refine ⟨fun b p h e hb hp hh he => ?_, fun z hz => hfr z fun i hzi => ?_⟩
    · have hbd := (req_sizes hpre hb).2.1
      have hbm : 0 < d.BM := by rw [hd.bm]; split_ifs <;> omega
      have hmn : p / d.BM < nBlocks d := by
        unfold nBlocks
        rw [Nat.div_lt_iff_lt_mul hbm]
        have := hd.maxExt_le
        have : p < d.maxExt := by omega
        have h2 : d.maxExt ≤ (d.maxExt + d.BM - 1) / d.BM * d.BM := by
          have h3 := Nat.div_add_mod (d.maxExt + d.BM - 1) d.BM
          have h4 := Nat.mod_lt (d.maxExt + d.BM - 1) hbm
          generalize (d.maxExt + d.BM - 1) / d.BM = c at *
          generalize (d.maxExt + d.BM - 1) % d.BM = r at *
          rw [Nat.mul_comm]
          omega
        omega
      obtain ⟨i, hi, hbi, hhi, hmi⟩ := grid_of hb hh hmn
      obtain ⟨m, hQ, hag⟩ := hloc ⟨i, hi⟩
      have hpos : p / d.BM * d.BM + p % d.BM = p := by
        rw [Nat.mul_comm]; exact Nat.div_add_mod p d.BM
      have hr : p % d.BM < d.BM := Nat.mod_lt _ hbm
      have hvr : mOf d i * d.BM + p % d.BM < extOf d init (bOf d i) := by
        rw [hmi, hbi, hpos]; exact hp
      obtain ⟨c, hc, hfin, hbound⟩ := hQ (p % d.BM) e hr he hvr
      refine ⟨c, ?_, hfin, ?_⟩
      · rw [← hc]
        simp only [rdB, hbi, hhi, hmi, hpos] at hc ⊢
        refine Mem.read_congr fun x hx => hag x (Or.inr ?_)
        refine d.o.span_sub_elems ?_ hx
        refine ⟨p % d.BM, e, hr, he, hvr, ?_⟩
        show _ = oIdx d (qsOf d init (bOf d i) + (mOf d i * d.BM + p % d.BM)) (hOf d i) e
        rw [hmi, hbi, hhi, hpos]
      · simpa only [hbi, hhi, hmi, hpos] using hbound
    · obtain ⟨hb, hh⟩ := grid_lt i.isLt
      exact hz (d.o.elems_mono (tile_sub hb hh) hzi)

end Claims.ExtendAttn
