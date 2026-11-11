import TVA.Proof.Shared.Acc

/-!
# The launch of programs that share cells through atomics: the invariant

The invariant `InvA` that `shared` (`TVA.Proof.Shared.Correct`) follows along every interleaving,
as `Inv` is for `independent` (`TVA.Proof.Compose`). Each program is `SafeA` in its own view
(`localMem`: the launch's memory on its private footprint, the initial memory elsewhere), with its
part of the shared steps so far as its log; the shared cells hold the **replay** of all programs'
shared steps, in the order they were performed, from the initial memory.

* A private step changes only the stepping program's private bytes (`SafeA.stepT`).
* A shared step changes only a shared cell: it appends to the interleaved log `G` and to the
  stepping program's part of it, and the cell now holds the replay extended by that step.
* No two programs race (`no_interRaceA`): private footprints do not conflict; two shared steps on
  overlapping bytes are on the same cell (the cells are equal or disjoint), and both are `acq_rel`
  atomics at `gpu` or `sys` scope, so `Event.Strong` excuses them. Within a program, two shared
  steps on one cell are `Event.SameAtomic`.
-/
namespace TVA

open Classical

section
variable {N : Nat} (E : Env) (R W : Fin N → Set Int) (C : Set (DType × Int))
  (Q : Fin N → List AtomicStep → Mem → Prop) (init : Mem)

/-- The invariant of a launch whose programs share the cells `C` through atomics, after the trace
`tr` in state `c`. -/
structure InvA (tr : List (Fin N × Event)) (c : Config N) : Prop where
  /-- Some interleaving `G` of the programs' shared steps so far: program `i` is `SafeA` in its
  own view with its part of `G` as log and the private bytes it has not yet written as write
  footprint, and the launch's cells hold the replay of `G` from the initial memory. -/
  safe : ∃ G : List (Fin N × AtomicStep),
    (∀ i, ∃ Wc, Wc ⊆ W i ∧ (∀ (n : Nat) (e : Event), tr[n]? = some (i, e) → Disjoint e.writes Wc) ∧
      SafeA E (R i) Wc C (fun _ log m => Q i log m) (proj G i) (localMem R W init c i) (c.progs i)) ∧
    ∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes C, c.mem x = mA x
  /-- Bytes outside the cells that no program may write still hold their initial value. -/
  frame : ∀ x, x ∉ cellBytes C → (∀ i, x ∉ W i) → c.mem x = init x
  /-- The cells' bytes stay present. -/
  present : ∀ x ∈ cellBytes C, c.mem x ≠ none
  /-- Every event of the trace is a private access inside its program's footprint, or a shared
  `acq_rel` atomic at `gpu` or `sys` scope on a whole cell. -/
  good : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) →
    (e.reads ⊆ R j ∪ e.writes ∧ e.writes ⊆ W j) ∨
      ((∃ c ∈ C, e.reads = Mem.span c.1 c.2 ∧ e.writes = Mem.span c.1 c.2) ∧
        ∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta)
  /-- No conflict within a program so far. -/
  clean : ¬ IntraRace tr

variable {E R W C Q init}

/-- A property of every event of a trace, extended by one more event that has it. -/
theorem good_snoc {tr : List (Fin N × Event)} {P : Fin N → Event → Prop}
    (hold : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) → P j e) {j : Fin N} {e : Event}
    (hnew : P j e) : ∀ (n : Nat) (i : Fin N) (e₀ : Event), (tr ++ [(j, e)])[n]? = some (i, e₀) → P i e₀ := by
  intro n i e₀ hn
  by_cases hlt : n < tr.length
  · rw [List.getElem?_append_left hlt] at hn; exact hold n i e₀ hn
  · have hn' : n = tr.length := by
      rcases Nat.lt_or_ge n (tr.length + 1) with h | h
      · omega
      · rw [List.getElem?_eq_none (by simp; omega)] at hn; cases hn
    subst hn'
    rw [List.getElem?_append_right le_rfl] at hn
    simp at hn
    obtain ⟨rfl, rfl⟩ := hn
    exact hnew

/-- The history condition of program `i` after one more event of program `j`. -/
theorem hist_snoc {tr : List (Fin N × Event)} {i j : Fin N} {e : Event} {Wc Wc' : Set Int}
    (hold : ∀ (n : Nat) (f : Event), tr[n]? = some (i, f) → Disjoint f.writes Wc) (hsub : Wc' ⊆ Wc)
    (hnew : i = j → Disjoint e.writes Wc') :
    ∀ (n : Nat) (f : Event), (tr ++ [(j, e)])[n]? = some (i, f) → Disjoint f.writes Wc' := by
  intro n f hn
  by_cases hlt : n < tr.length
  · rw [List.getElem?_append_left hlt] at hn; exact (hold n f hn).mono_right hsub
  · have hn' : n = tr.length := by
      rcases Nat.lt_or_ge n (tr.length + 1) with h | h
      · omega
      · rw [List.getElem?_eq_none (by simp; omega)] at hn; cases hn
    subst hn'
    rw [List.getElem?_append_right le_rfl] at hn
    simp at hn
    obtain ⟨rfl, rfl⟩ := hn
    exact hnew rfl

/-- **One step keeps the invariant**: the stepping program's own view takes a step of its own
(`SafeA.Next`), and the other programs' views do not change. -/
theorem invA_step (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i)) (hRW : ∀ i, Disjoint (R i) (W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    {tr : List (Fin N × Event)} {c c' : Config N} {j : Fin N}
    {e : Event} (hinv : InvA E R W C Q init tr c) (hs : LStep E c (j, e) c') :
    InvA E R W C Q init (tr ++ [(j, e)]) c' ∧
    SafeA.Next E C (localMem R W init c' j, c'.progs j) (localMem R W init c j, c.progs j) ∧
    ∀ i, i ≠ j → localMem R W init c' i = localMem R W init c i ∧ c'.progs i = c.progs i := by
  cases hs with
  | step _ hstep =>
    rename_i m' p'
    obtain ⟨G, hsafeG, mA, hrep, hagA⟩ := hinv.safe
    obtain ⟨Wc, hWc, hhist, hsj⟩ := hsafeG j
    have hag : ∀ x ∈ R j ∪ Wc, c.mem x = localMem R W init c j x := fun x hx => by
      have : x ∈ R j ∪ W j := hx.elim Or.inl (fun h => Or.inr (hWc h))
      simp [localMem, this]
    have hdA : Disjoint (cellBytes C) (R j ∪ Wc) :=
      (hA j).mono_right (Set.union_subset_union_right _ hWc)
    have hframe := Step.frame hstep
    -- earlier events of `j` are private, inside the footprint, or shared, on the cells
    have hold : ∀ (a : Nat) (f : Event), tr[a]? = some (j, f) →
        (f.reads ∪ f.writes ⊆ R j ∪ W j) ∨ (f.reads ∪ f.writes ⊆ cellBytes C) := by
      intro a f ha
      rcases hinv.good a j f ha with ⟨hfr, hfw⟩ | ⟨⟨cf, hcf, hfr, hfw⟩, -⟩
      · left
        rintro z (hz | hz)
        · exact (hfr hz).elim Or.inl (fun h => Or.inr (hfw h))
        · exact Or.inr (hfw hz)
      · right
        rintro z (hz | hz)
        · rw [hfr] at hz; exact span_sub_cellBytes hcf hz
        · rw [hfw] at hz; exact span_sub_cellBytes hcf hz
    rcases hsj.stepT (hQ j) hdA hag hstep with
      ⟨hR, hW, hs2, hsafe'⟩ | ⟨s, hsC, hsc, hsr, hsw, hsre, hswe, hsg, hsafe'⟩
    · -- a private step
      have hloc : localMem R W init ⟨m', Function.update c.progs j p'⟩ j =
          Mem.mix (localMem R W init c j) m' e.writes := by
        funext x
        by_cases hx : x ∈ R j ∪ W j
        · simp only [localMem, hx, ite_true]
          by_cases hxe : x ∈ e.writes
          · rw [Mem.mix_in hxe]
          · rw [Mem.mix_out hxe, hframe x hxe]; simp [localMem, hx]
        · have hxe : x ∉ e.writes := fun h => hx (Or.inr (hWc (hW h)))
          simp only [localMem, hx, ite_false]
          rw [Mem.mix_out hxe]; simp [localMem, hx]
      have hother : ∀ i, i ≠ j → localMem R W init ⟨m', Function.update c.progs j p'⟩ i =
          localMem R W init c i := by
        intro i hij
        funext x
        by_cases hx : x ∈ R i ∪ W i
        · simp only [localMem, hx, ite_true]
          exact hframe x fun hw => Set.disjoint_left.mp (hdisj j i hij.symm) (hWc (hW hw)) hx
        · simp [localMem, hx]
      have hnotA : ∀ x ∈ cellBytes C, x ∉ e.writes := fun x hx hw =>
        Set.disjoint_left.mp (hA j) hx (Or.inr (hWc (hW hw)))
      have heRW : e.reads ∪ e.writes ⊆ R j ∪ W j := by
        rintro z (hz | hz)
        · exact (hR hz).elim Or.inl (fun h => Or.inr (hWc (hW h)))
        · exact Or.inr (hWc (hW hz))
      refine ⟨⟨⟨G, fun i => ?_, mA, hrep, fun x hx => ?_⟩, fun x hx hw => ?_, fun x hx => ?_,
        good_snoc hinv.good (Or.inl ⟨hR, hW.trans hWc⟩), ?_⟩, ?_, ?_⟩
      · by_cases hij : i = j
        · subst hij
          refine ⟨Wc \ e.writes, Set.sdiff_subset.trans hWc,
            hist_snoc hhist Set.sdiff_subset (fun _ => Set.disjoint_sdiff_right), ?_⟩
          rw [hloc]; simpa using hsafe'
        · obtain ⟨Wi, hWi, hhi, hsi⟩ := hsafeG i
          refine ⟨Wi, hWi, hist_snoc hhi le_rfl (fun h => absurd h hij), ?_⟩
          rw [hother i hij]; simpa [Function.update_of_ne hij] using hsi
      · simp only; rw [hframe x (hnotA x hx)]; exact hagA x hx
      · simp only; rw [hframe x fun h => hw j (hWc (hW h))]; exact hinv.frame x hx hw
      · simp only; rw [hframe x (hnotA x hx)]; exact hinv.present x hx
      · intro hrace
        rcases intraRace_snoc hrace with h | ⟨a, f, ha, hcon, -⟩
        · exact hinv.clean h
        · rcases hold a f ha with hf | hf
          · rcases hinv.good a j f ha with ⟨hfr, hfw⟩ | ⟨⟨cf, hcf, hfr, hfw⟩, -⟩
            · have hfe : Disjoint f.writes e.writes := (hhist a f ha).mono_right hW
              have hfR : Disjoint f.writes (R j) := ((hRW j).symm).mono_left hfw
              have heR : Disjoint e.writes (R j) := ((hRW j).symm).mono_left (hW.trans hWc)
              rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
              · rcases hze with hze | hze
                · rcases hR hze with hz | hz
                  · exact Set.disjoint_left.mp hfR hzf hz
                  · exact Set.disjoint_left.mp hfe hzf hz
                · exact Set.disjoint_left.mp hfe hzf hze
              · rcases hfr hzf with hz | hz
                · exact Set.disjoint_left.mp heR hze hz
                · exact Set.disjoint_left.mp hfe hz hze
            · obtain ⟨x, hx⟩ := span_nonempty cf.1 cf.2
              exact Set.disjoint_left.mp (hA j) (span_sub_cellBytes hcf hx) (hf (Or.inr (hfw ▸ hx)))
          · rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
            · exact Set.disjoint_left.mp (hA j) (hf (Or.inr hzf)) (heRW hze)
            · exact Set.disjoint_left.mp (hA j) (hf (Or.inl hzf)) (heRW (Or.inr hze))
      · refine ⟨localMem R W init c j, Mem.mix (localMem R W init c j) m' e.writes, e,
          fun _ _ => rfl, ?_, ?_⟩
        · simpa using hs2
        · rw [hloc]
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx, Mem.mix_out (hnotA x hx)]
          · rw [Mem.mix_out hx]
      · intro i hij; exact ⟨hother i hij, Function.update_of_ne hij _ _⟩
    · -- a shared atomic
      subst hsw
      have hloc : ∀ i, localMem R W init ⟨c.mem.write s.d s.a s.new, Function.update c.progs j p'⟩ i =
          localMem R W init c i := by
        intro i
        funext x
        by_cases hx : x ∈ R i ∪ W i
        · simp only [localMem, hx, ite_true]
          exact Mem.write_frame _ _ _ _ fun hsp =>
            Set.disjoint_left.mp (hA i) (span_sub_cellBytes hsC hsp) hx
        · simp [localMem, hx]
      have hrA : mA.read s.d s.a = some s.old := by
        rw [← hsr]; exact Mem.read_congr fun x hx => (hagA x (span_sub_cellBytes hsC hx)).symm
      have heA : e.reads ∪ e.writes ⊆ cellBytes C := by
        rintro z (hz | hz)
        · rw [hsre] at hz; exact span_sub_cellBytes hsC hz
        · rw [hswe] at hz; exact span_sub_cellBytes hsC hz
      refine ⟨⟨⟨G ++ [(j, s)], fun i => ?_, mA.write s.d s.a s.new, ?_,
        Mem.write_agree hagA _ _ _⟩, fun x hx hw => ?_, fun x hx => Mem.write_present _ (hinv.present x hx),
        good_snoc hinv.good (Or.inr ⟨⟨(s.d, s.a), hsC, hsre, hswe⟩, hsg⟩), ?_⟩, ?_, ?_⟩
      · rw [hloc i]
        by_cases hij : i = j
        · subst hij
          refine ⟨Wc, hWc, hist_snoc hhist le_rfl (fun _ => ?_), ?_⟩
          · rw [hswe]
            exact Set.disjoint_left.mpr fun x hx hxW => Set.disjoint_left.mp (hA i)
              (span_sub_cellBytes hsC hx) (Or.inr (hWc hxW))
          · rw [proj_snoc_self]; simpa using hsafe'
        · obtain ⟨Wi, hWi, hhi, hsi⟩ := hsafeG i
          refine ⟨Wi, hWi, hist_snoc hhi le_rfl (fun h => absurd h hij), ?_⟩
          rw [proj_snoc_other _ hij]; simpa [Function.update_of_ne hij] using hsi
      · rw [List.map_append]; exact hrep.snoc hrA
      · simp only
        rw [Mem.write_frame _ _ _ _ fun hsp => hx (span_sub_cellBytes hsC hsp)]
        exact hinv.frame x hx hw
      · intro hrace
        rcases intraRace_snoc hrace with h | ⟨a, f, ha, hcon, hsame⟩
        · exact hinv.clean h
        · rcases hinv.good a j f ha with ⟨hfr, hfw⟩ | ⟨⟨cf, hcf, hfr, hfw⟩, hfg⟩
          · have hf : f.reads ∪ f.writes ⊆ R j ∪ W j := by
              rintro z (hz | hz)
              · exact (hfr hz).elim Or.inl (fun h => Or.inr (hfw h))
              · exact Or.inr (hfw hz)
            rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
            · exact Set.disjoint_left.mp (hA j) (heA hze) (hf (Or.inr hzf))
            · exact Set.disjoint_left.mp (hA j) (heA (Or.inr hze)) (hf (Or.inl hzf))
          · -- two shared atomics on overlapping cells: the same cell
            have hsame' : cf = (s.d, s.a) := by
              refine hcells cf hcf _ hsC ?_
              rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
              · rw [hfw] at hzf
                rcases hze with hze | hze
                · rw [hsre] at hze; exact ⟨z, hzf, hze⟩
                · rw [hswe] at hze; exact ⟨z, hzf, hze⟩
              · rw [hfr] at hzf; rw [hswe] at hze; exact ⟨z, hzf, hze⟩
            subst hsame'
            obtain ⟨sc₁, hf1, -⟩ := hfg
            obtain ⟨sc₂, he1, -⟩ := hsg
            exact hsame ⟨by rw [hf1]; rfl, by rw [he1]; rfl, hfr.trans hsre.symm, hfw.trans hswe.symm⟩
      · refine ⟨Mem.mix (localMem R W init c j) c.mem (cellBytes C),
          Mem.mix (Mem.mix (localMem R W init c j) c.mem (cellBytes C)) (c.mem.write s.d s.a s.new) e.writes,
          e, fun x hx => Mem.mix_out hx, ?_, ?_⟩
        · simpa using Step.transfer hstep fun x hx => Mem.mix_in (heA hx)
        · rw [hloc j]
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx]
          · rw [Mem.mix_out hx, Mem.mix_out (fun h => hx (heA (Or.inr h))), Mem.mix_out hx]
      · intro i hij; exact ⟨hloc i, Function.update_of_ne hij _ _⟩

/-- The invariant holds along every trace from a state where it holds. -/
theorem invA_reach (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i)) (hRW : ∀ i, Disjoint (R i) (W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    {c₀ c : Config N} {tr : List (Fin N × Event)}
    (h0 : InvA E R W C Q init [] c₀) (hr : Reach E c₀ tr c) : InvA E R W C Q init tr c := by
  induction hr with
  | refl => exact h0
  | @tail c₂ c₃ tr₁ ev _ hs ih =>
    obtain ⟨j, e⟩ := ev
    exact (invA_step hdisj hA hRW hcells hQ ih hs).1

/-- No two programs race. Private events do not conflict across programs or with the shared
cells. Two shared events that conflict touch the same cell, since cells are equal or disjoint,
and both are `acq_rel` at `gpu`/`sys` scope, so they are `Event.Strong`. -/
theorem no_interRaceA (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    {tr : List (Fin N × Event)} {c : Config N}
    (hinv : InvA E R W C Q init tr c) : ¬ InterRace tr := by
  rintro ⟨a, b, ⟨i, x⟩, ⟨j, y⟩, -, ha, hb, hij, hcon, hstrong, -⟩
  have hAi := hA i
  have hAj := hA j
  rcases hinv.good a i x ha with ⟨hxr, hxw⟩ | ⟨⟨cx, hcx, hxr, hxw⟩, hxg⟩ <;>
    rcases hinv.good b j y hb with ⟨hyr, hyw⟩ | ⟨⟨cy, hcy, hyr, hyw⟩, hyg⟩
  · have hy : y.reads ∪ y.writes ⊆ R j ∪ W j := by
      rintro z (hz | hz)
      · exact (hyr hz).elim Or.inl (fun h => Or.inr (hyw h))
      · exact Or.inr (hyw hz)
    have hx : x.reads ⊆ R i ∪ W i := fun z hz => (hxr hz).elim Or.inl (fun h => Or.inr (hxw h))
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (hy hzy)
    · exact Set.disjoint_left.mp (hdisj j i (Ne.symm hij)) (hyw hzy) (hx hzx)
  · have hyA : ∀ z, z ∈ y.reads ∪ y.writes → z ∈ cellBytes C := by
      rintro z (hz | hz)
      · rw [hyr] at hz; exact span_sub_cellBytes (d := cy.1) (a := cy.2) hcy hz
      · rw [hyw] at hz; exact span_sub_cellBytes (d := cy.1) (a := cy.2) hcy hz
    have hx : x.reads ∪ x.writes ⊆ R i ∪ W i := by
      rintro z (hz | hz)
      · exact (hxr hz).elim Or.inl (fun h => Or.inr (hxw h))
      · exact Or.inr (hxw hz)
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp hAi (hyA z hzy) (hx (Or.inr hzx))
    · exact Set.disjoint_left.mp hAi (hyA z (Or.inr hzy)) (hx (Or.inl hzx))
  · have hxA : ∀ z, z ∈ x.reads ∪ x.writes → z ∈ cellBytes C := by
      rintro z (hz | hz)
      · rw [hxr] at hz; exact span_sub_cellBytes (d := cx.1) (a := cx.2) hcx hz
      · rw [hxw] at hz; exact span_sub_cellBytes (d := cx.1) (a := cx.2) hcx hz
    have hy : y.reads ∪ y.writes ⊆ R j ∪ W j := by
      rintro z (hz | hz)
      · exact (hyr hz).elim Or.inl (fun h => Or.inr (hyw h))
      · exact Or.inr (hyw hz)
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp hAj (hxA z (Or.inr hzx)) (hy hzy)
    · exact Set.disjoint_left.mp hAj (hxA z (Or.inl hzx)) (hy (Or.inr hzy))
  · -- both shared: the same cell
    have hsame : cx = cy := by
      refine hcells cx hcx cy hcy ?_
      rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
      · rw [hxw] at hzx
        rcases hzy with hzy | hzy
        · rw [hyr] at hzy; exact ⟨z, hzx, hzy⟩
        · rw [hyw] at hzy; exact ⟨z, hzx, hzy⟩
      · rw [hxr] at hzx; rw [hyw] at hzy; exact ⟨z, hzx, hzy⟩
    subst hsame
    exact hstrong ⟨hxg, hyg, hxr.trans hyr.symm, hxw.trans hyw.symm⟩

end

end TVA
