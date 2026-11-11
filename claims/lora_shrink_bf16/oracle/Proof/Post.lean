import Proof.Foot
import Proof.Numeric

/-!
# `lora_shrink_bf16`: the specification from the programs' logs

The final output cells are the replay of the programs' atomic steps. A cell of a token without an
adapter is never touched; a cell of a token with an adapter is the sum of the `SPLIT_K` operands of
the programs that target it, up to the error of the atomic additions.
-/

namespace Claims.LoraShrink

open TVA

/-- The log of a program is order-insensitive. -/
theorem Q_permInv (d : Inst) (init : Mem) (i : Nat) :
    PermInv (fun (_ : Unit) log m => Q d init i log m) := by
  intro _ l l' m hp h
  obtain ⟨h1, ops, h2, h3, h4⟩ := h
  exact ⟨fun st hst => h1 st (hp.mem_iff.mpr hst), ops,
    fun s t j hs ht hj va vb hD => (h2 s t j hs ht hj va vb hD).perm hp, h3, h4⟩

/-- Cells that overlap are equal. -/
theorem cells_distinct (d : Inst) :
    ∀ c ∈ Cells d, ∀ c' ∈ Cells d, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c' := by
  rintro c ⟨s, t, j, hs, ht, hj, rfl⟩ c' ⟨s', t', j', hs', ht', hj', rfl⟩ ⟨x, hx1, hx2⟩
  by_contra hne
  have hne' : ¬ (s' = s ∧ t' = t ∧ j' = j) := by
    rintro ⟨rfl, rfl, rfl⟩; exact hne rfl
  exact Set.disjoint_left.mp (cell_disjoint d ht hj ht' hj' hne') hx2 hx1

/-- The cells are present initially. -/
theorem cells_present (d : Inst) (hd : d.ok) {init : Mem} (hpre : Pre d init) :
    ∀ x ∈ cellBytes (Cells d), init x ≠ none := by
  intro x hx
  exact hpre.2.2.1.2.2.2 x (cellBytes_sub d hd hx)

/-- A replay that only writes whole cells leaves a readable cell readable. -/
theorem replay_readable {m m' : Mem} {l : List AtomicStep} {a : Int} (hrep : Replay m l m')
    (hl : ∀ s ∈ l, s.d = .f32 ∧ (s.a = a ∨ Disjoint (Mem.span .f32 s.a) (Mem.span .f32 a)))
    {x : F32} (h0 : m.read .f32 a = some x) : ∃ y, m'.read .f32 a = some y := by
  induction hrep generalizing x with
  | nil => exact ⟨x, h0⟩
  | @cons m₀ m₁ s rest hr _ ih =>
    obtain ⟨dd, aa, o, n⟩ := s
    obtain ⟨hdd, hcase⟩ := hl ⟨dd, aa, o, n⟩ List.mem_cons_self
    simp only at hdd hcase
    subst hdd
    refine ih (fun s' hs' => hl s' (List.mem_cons_of_mem _ hs')) (x := ?_) ?_
    · exact (if aa = a then n else x)
    · by_cases haa : aa = a
      · subst haa
        simp only [ite_true]
        exact Mem.read_write_self (Mem.writable_of_read h0) n
      · have hdis : Disjoint (Mem.span .f32 aa) (Mem.span .f32 a) := by
          rcases hcase with h | h
          · exact absurd h haa
          · exact h
        simp only [haa, ite_false]
        rw [Mem.read_write_other n hdis]
        exact h0

/-- Re-indexing a sum over the programs to a sum over the parts, when the summand vanishes off the
programs `f p`. -/
theorem sum_fin_reindex {N k : Nat} {β : Type} [AddCommMonoid β] (f : Nat → Nat)
    (hf : ∀ p, p < k → f p < N) (hinj : ∀ p, p < k → ∀ q, q < k → f p = f q → p = q)
    (F : Fin N → β) (h0 : ∀ i : Fin N, (∀ p, p < k → (i : Nat) ≠ f p) → F i = 0) :
    ∑ i, F i = ∑ p ∈ Finset.range k, (if h : f p < N then F ⟨f p, h⟩ else 0) := by
  classical
  let S : Finset (Fin N) := Finset.univ.filter fun i => ∃ p, p < k ∧ (i : Nat) = f p
  have h1 : ∑ i, F i = ∑ i ∈ S, F i := by
    symm
    refine Finset.sum_subset (Finset.subset_univ _) fun i _ hi => h0 i fun p hp hne => hi ?_
    simp only [S, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨p, hp, hne⟩
  rw [h1]
  symm
  refine Finset.sum_bij (fun p hp => (⟨f p, hf p (Finset.mem_range.mp hp)⟩ : Fin N)) ?_ ?_ ?_ ?_
  · intro p hp
    simp only [S, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨p, Finset.mem_range.mp hp, rfl⟩
  · intro p hp q hq h
    exact hinj p (Finset.mem_range.mp hp) q (Finset.mem_range.mp hq) (Fin.mk.inj h)
  · intro i hi
    simp only [S, Finset.mem_filter, Finset.mem_univ, true_and] at hi
    obtain ⟨p, hp, hpi⟩ := hi
    exact ⟨p, Finset.mem_range.mpr hp, Fin.ext hpi.symm⟩
  · intro p hp
    simp [hf p (Finset.mem_range.mp hp)]

end Claims.LoraShrink
