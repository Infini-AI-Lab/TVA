import Proof.Post

/-!
# `lora_shrink_bf16`: `Post` from the programs' invariants and the replay

Given the programs' invariants `Q` on their parts of the interleaved log and the replay of the
whole log on the cells, every output element satisfies `OutSpec`, and every byte outside the
output is unchanged.
-/

namespace Claims.LoraShrink

open TVA

/-- Every step of the interleaved log is a step on some output element that a program targets. -/
theorem steps_on_cells {d : Inst} {init : Mem} {G : List (Fin (launch d).size × AtomicStep)}
    (hQ : ∀ i : Fin (launch d).size, ∃ m, Q d init i (proj G i) m) :
    ∀ st ∈ G.map Prod.snd, ∃ i : Fin (launch d).size, ∃ s t j, Tgt d i s t j ∧ st.d = .f32 ∧
      st.a = cellAddr d s t j := by
  intro st hst
  obtain ⟨⟨i, st'⟩, hG, rfl⟩ := List.mem_map.mp hst
  obtain ⟨m, hl, -⟩ := hQ i
  have : st' ∈ proj G i := List.mem_map.mpr ⟨(i, st'), List.mem_filter.mpr ⟨hG, by simp⟩, rfl⟩
  obtain ⟨s, t, j, h1, h2, h3⟩ := hl st' this
  exact ⟨i, s, t, j, h1, h2, h3⟩

/-- **The specification.** -/
theorem post_of_decomp {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (fin : Mem) (G : List (Fin (launch d).size × AtomicStep))
    (hQ : ∀ i : Fin (launch d).size, ∃ m, Q d init i (proj G i) m)
    (hrep : ∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes (Cells d), fin x = mA x)
    (hfr : ∀ x, x ∉ cellBytes (Cells d) → fin x = init x) : Post d init fin := by
  obtain ⟨mA, hrep, hagree⟩ := hrep
  refine ⟨fun s t j hs ht hj => ?_, fun z hz => ?_⟩
  swap
  · apply hfr z
    rw [cellBytes_eq]; exact hz
  have hcell : (DType.f32, cellAddr d s t j) ∈ Cells d := ⟨s, t, j, hs, ht, hj, rfl⟩
  have hfin_eq : fin.read .f32 (d.out.ptr (outAt d s t j)).addr = mA.read .f32 (cellAddr d s t j) :=
    Mem.read_congr (fun x hx => hagree x (span_sub_cellBytes hcell hx))
  have hx0 : init.read .f32 (cellAddr d s t j) = some (FloatV.ofPat 0) :=
    hpre.2.2.2.2.2.2.2.2.2.2.2.2 s t j hs ht hj
  have hsteps := steps_on_cells hQ
  -- every step is on the cell itself or on bytes disjoint from it
  have hl : ∀ st ∈ G.map Prod.snd, st.d = .f32 ∧
      (st.a = cellAddr d s t j ∨ Disjoint (Mem.span .f32 st.a) (Mem.span .f32 (cellAddr d s t j))) := by
    intro st hst
    obtain ⟨i, s', t', j', hT, hd1, ha1⟩ := hsteps st hst
    obtain ⟨hs', ht', hj', -⟩ := tgt_range d hd i.isLt hT
    refine ⟨hd1, ?_⟩
    by_cases heq : s' = s ∧ t' = t ∧ j' = j
    · obtain ⟨rfl, rfl, rfl⟩ := heq
      exact Or.inl ha1
    · right
      rw [ha1]
      exact cell_disjoint d ht hj ht' hj' heq
  obtain ⟨y0, hy0⟩ := replay_readable hrep hl hx0
  refine ⟨y0, hfin_eq.trans hy0, ?_⟩
  by_cases hneg : d.adapter t < 0
  · left
    refine ⟨hneg, ?_⟩
    have hl' : ∀ st ∈ G.map Prod.snd, MapsCell .f32 (cellAddr d s t j) (fun x => x) st := by
      intro st hst
      obtain ⟨i, s', t', j', hT, hd1, ha1⟩ := hsteps st hst
      obtain ⟨hs', ht', hj', hnn⟩ := tgt_range d hd i.isLt hT
      have hne : ¬ (s' = s ∧ t' = t ∧ j' = j) := by
        rintro ⟨-, rfl, -⟩; omega
      have := cell_disjoint d ht hj ht' hj' hne
      refine mapsCell_other (d := .f32) (a := cellAddr d s t j) (s := st) ?_
      rw [hd1, ha1]; exact this
    have := replay_maps (g := fun _ x => x) hrep hl' hx0
    have hf : ∀ (l : List AtomicStep) (x : F32), l.foldl (fun x _ => x) x = x := by
      intro l; induction l <;> simp_all
    rw [hf] at this
    rw [hy0] at this
    exact (Option.some.inj this)
  · right
    refine ⟨not_lt.mp hneg, ?_⟩
    intro va vb h1 h2 h3 h4
    have hdat : Dat d init s t j va vb := ⟨not_lt.mp hneg, h1, h2, h3, h4⟩
    obtain ⟨f, hf1, hf2⟩ := tgt_unique d hd hs ht hj (not_lt.mp hneg)
    -- each program's operands for this element
    have hQ' : ∀ i : Fin (launch d).size, ∃ ops : Multiset ℚ,
        Paired (AccE d s t j) (proj G i) ops ∧ (¬ Tgt d i s t j → ops = 0) ∧
        (Tgt d i s t j → ∃ v : F32, ops = {v.toRat} ∧ PartOk d va vb (part d i) v) := by
      intro i
      obtain ⟨m, -, ops, hpair, hno, hyes⟩ := hQ i
      refine ⟨ops s t j, hpair s t j hs ht hj va vb hdat, hno s t j, fun htg => ?_⟩
      obtain ⟨v, hv, hp⟩ := hyes s t j htg
      exact ⟨v, hv, hp va vb hdat⟩
    choose ops hpair hno hyes using hQ'
    have : Nonempty F32 := ⟨FloatV.ofPat 0⟩
    have hex : ∀ p, p < sk d → ∃ v : F32,
        (∀ h : f p < (launch d).size, ops ⟨f p, h⟩ = {v.toRat}) ∧ PartOk d va vb p v := by
      intro p hp
      obtain ⟨v, hv, hpo⟩ := hyes ⟨f p, (hf1 p hp).1⟩ (hf1 p hp).2.1
      refine ⟨v, fun h => hv, ?_⟩
      simpa [(hf1 p hp).2.2] using hpo
    choose! vv hvv hpo using hex
    have hfinj : ∀ p, p < sk d → ∀ q, q < sk d → f p = f q → p = q := fun p hp q hq h => by
      rw [← (hf1 p hp).2.2, ← (hf1 q hq).2.2, h]
    have hh0 : ∀ i : Fin (launch d).size, (∀ p, p < sk d → (i : Nat) ≠ f p) → ops i = 0 :=
      fun i hi => hno i fun htg => by
        obtain ⟨p, hp, e⟩ := hf2 i i.isLt htg
        exact hi p hp e
    have hfl : ∀ p, p < sk d → f p < (launch d).size := fun p hp => (hf1 p hp).1
    have hS1 : ∑ i, (ops i).sum = ∑ p ∈ Finset.range (sk d), (vv p).toRat := by
      rw [sum_fin_reindex f hfl hfinj (fun i => (ops i).sum) (fun i hi => by simp [hh0 i hi])]
      refine Finset.sum_congr rfl fun p hp => ?_
      have hp' := Finset.mem_range.mp hp
      simp [hfl p hp', hvv p hp' (hfl p hp')]
    have hS2 : ∑ i, ((ops i).map fun w => |w|).sum = ∑ p ∈ Finset.range (sk d), |(vv p).toRat| := by
      rw [sum_fin_reindex f hfl hfinj (fun i => ((ops i).map fun w => |w|).sum)
        (fun i hi => by simp [hh0 i hi])]
      refine Finset.sum_congr rfl fun p hp => ?_
      have hp' := Finset.mem_range.mp hp
      simp [hfl p hp', hvv p hp' (hfl p hp')]
    have hS3 : ∑ i, Multiset.card (ops i) = sk d := by
      rw [sum_fin_reindex f hfl hfinj (fun i => Multiset.card (ops i))
        (fun i hi => by simp [hh0 i hi])]
      have : ∀ p ∈ Finset.range (sk d), (if h : f p < (launch d).size then
          Multiset.card (ops ⟨f p, h⟩) else 0) = 1 := by
        intro p hp
        have hp' := Finset.mem_range.mp hp
        simp [hfl p hp', hvv p hp' (hfl p hp')]
      rw [Finset.sum_congr rfl this]
      simp
    have hvb : ∀ p, p < sk d →
        |(vv p).toRat - s32 d * partSum d va vb p| ≤ |s32 d| * errPart d va vb p + mulErr d va vb p ∧
        |(vv p).toRat| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p :=
      fun p hp => ⟨(hpo p hp).2.1, (hpo p hp).2.2⟩
    obtain ⟨hsumle, hroom⟩ := atom_room d va vb (fun p => (vv p).toRat) (fun p hp => (hvb p hp).2) h4
    have hx : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero Format.binary32_regular
    obtain ⟨y, hy, hgood, herr⟩ := replay_err_shared (d := .f32) (a := cellAddr d s t j)
      FloatV.toRat (fun x : F32 => x.isFinite = true) (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ))
      (B := Format.binary32.maxVal) (by positivity) (by positivity) (ops := ops) hrep
      (fun i => hpair i) hx0 hx (by rw [hS2, hS3]; simpa using hroom)
    rw [hy0] at hy
    obtain rfl := Option.some.inj hy
    rw [hS1, hS2, hS3] at herr
    simp only [FloatV.toRat_ofPat_zero, abs_zero, zero_add] at herr
    exact ⟨hgood, total_bound d hd va vb h3 (fun p => (vv p).toRat) hvb _ herr⟩

end Claims.LoraShrink
