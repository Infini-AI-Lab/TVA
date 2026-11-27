import Proof.Basic

/-!
# `ep_src2dst_i32`: the search for `first_nonnegative`

Lines 42–51 read only `sorted_topk_ids` and end with `Found`: the number of leading negative ids.
The binary search keeps `0 ≤ left ≤ right ≤ n`, every id before `left` negative and every id from
`right` on not negative (the negative ids come first, `Pre`); `right - left` decreases.
-/

namespace Claims.EpSrc2Dst

open TVA

/-- The loop's invariant from memory `m`: the memory is unchanged and the state brackets the
first non-negative id. -/
def Inv (d : Inst) (m : Mem) (st : I32 × I32) (m' : Mem) : Prop :=
  m' = m ∧ ∃ l r : Nat, st.1.toInt = l ∧ st.2.toInt = r ∧ l ≤ r ∧ r ≤ d.n ∧
    (∀ j, j < l → sid d m j < 0) ∧ (∀ j, r ≤ j → j < d.n → ¬ sid d m j < 0)

/-- The loop's variant, `right - left`. -/
def μ (st : I32 × I32) : Nat := (st.2.toInt - st.1.toInt).toNat

variable {E : Env} {R : Set Int}

/-- One trip of the search: it reads one sorted id and shrinks the bracket. -/
theorem safe_step {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m)
    (hR : d.sorted.elems {j | j < d.n} ⊆ R) (st : I32 × I32) (hI : Inv d m st m)
    (hlt : st.1.toInt < st.2.toInt) :
    Safe E R ∅ (fun st' m' => Inv d m st' m' ∧ μ st' < μ st) m (searchStep (d.sorted.ptr 0) st) := by
  obtain ⟨a, b⟩ := st
  obtain ⟨-, l, r, h1, h2, hlr, hrn, hl, hr⟩ := hI
  simp only at h1 h2 hlt
  have hn := hd.n_lt
  have hlr' : l < r := by omega
  simp only [searchStep]
  have hsub : (b - a).toInt = ((r - l : Nat) : Int) := IntV.I32.sub_nat h2 h1 hlr (by omega)
  have h2c : (IntV.ofInt 2 : I32).toInt = 2 := IntV.I32.toInt_ofInt (by norm_num)
  refine safe_idiv1 (by rw [hsub]; omega) (by rw [h2c]; norm_num) fun q hq => ?_
  rw [hsub, h2c] at hq
  have hq' : q.item.toInt = (((r - l) / 2 : Nat) : Int) := by rw [hq]; omega
  have hmid : (a + q.item).toInt = ((l + (r - l) / 2 : Nat) : Int) :=
    IntV.I32.add_nat h1 hq' (by omega)
  set mid := l + (r - l) / 2 with hmid_def
  have hmr : mid < r := by omega
  have hptr : d.sorted.ptr 0 + (a + q.item) = d.sorted.ptr (mid : Int) := by
    rw [Ptr.add_int, Buf.ptr_offset, hmid, zero_add]
  obtain ⟨v, hv⟩ := Buf.read_some d.sorted hpre.1 (i := mid) (by have := hd.sorted_len; omega)
  refine safe_load1 (v0 := v) (by rw [hptr]; exact hv) ?_ ?_
  · rw [hptr]
    exact (d.sorted.span_sub_elems (show mid ∈ {j | j < d.n} by simp only [Set.mem_ofPred_eq]; omega)).trans hR
  have hsv : sid d m mid = v.toInt := sid_of_read hv
  rw [IntV.lt_eq, IntV.toInt_ofInt_zero]
  by_cases hneg : v.toInt < 0
  · rw [ite_eq_left (by simpa using hneg)]
    refine safe_ret ⟨⟨rfl, mid + 1, r, ?_, h2, by omega, hrn, ?_, hr⟩, ?_⟩
    · exact IntV.I32.add_nat hmid (by simpa using IntV.toInt_ofInt_one (w := 32) (by norm_num))
        (by omega)
    · intro j hj
      exact hpre.2.2.2.1 j mid (by omega) (by omega) (by rw [hsv]; exact hneg)
    · have e : (a + q.item + IntV.ofInt 1 : I32).toInt = ((mid + 1 : Nat) : Int) :=
        IntV.I32.add_nat hmid (by simpa using IntV.toInt_ofInt_one (w := 32) (by norm_num))
          (by omega)
      simp only [μ, e, h2, h1]
      omega
  · rw [ite_eq_right (by simpa using hneg)]
    refine safe_ret ⟨⟨rfl, l, mid, h1, hmid, by omega, by omega, hl, ?_⟩, ?_⟩
    · intro j hj hjn hs
      exact hneg (hsv ▸ hpre.2.2.2.1 mid j hj hjn hs)
    · simp only [μ, hmid, h2, h1]
      omega

/-- **Lines 42–51** from the loaded `s0 = sorted_topk_ids[0]`: they read only sorted ids and end
with `Found`. -/
theorem safe_searchIte {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) (hn0 : 0 < d.n)
    (hR : d.sorted.elems {j | j < d.n} ⊆ R) (s0 : I32) (hs0 : sid d m 0 = s0.toInt) :
    Safe E R ∅ (fun f m' => m' = m ∧ Found d m f) m
      (searchIte (d.sorted.ptr 0) (IntV.ofNat d.n) s0) := by
  have hn := hd.n_lt
  unfold searchIte
  rw [IntV.lt_eq, IntV.toInt_ofInt_zero]
  by_cases h0 : s0.toInt < 0
  · rw [ite_eq_left (by simpa using h0)]
    simp only [Prog.bind_eq]
    have hnv : (IntV.ofNat d.n : I32).toInt = (d.n : Int) := IntV.I32.toInt_ofNat hn
    have hw := safe_while (E := E) (R := R) (m := m) (fun st : I32 × I32 => pure (IntV.lt st.1 st.2))
      (searchStep (d.sorted.ptr 0)) (IntV.ofInt 0, IntV.ofNat d.n)
      (Q' := fun st m' => m' = m ∧ Found d m st.1) (Inv d m) μ (fun _ => ∅)
      (fun _ _ _ => Set.disjoint_empty _)
      ⟨rfl, 0, d.n, IntV.toInt_ofInt_zero, hnv, by omega, le_refl _, fun j hj => by omega,
        fun j hj hjn => by omega⟩
      (fun st m₁ hI => by
        obtain ⟨rfl, l, r, h1, h2, hlr, hrn, hl, hr⟩ := hI
        refine safe_ret ?_
        rw [IntV.lt_eq]
        by_cases hlt : st.1.toInt < st.2.toInt
        · rw [ite_eq_left (by simpa using hlt)]
          exact safe_step hd hpre hR st ⟨rfl, l, r, h1, h2, hlr, hrn, hl, hr⟩ hlt
        · rw [ite_eq_right (by simpa using hlt)]
          refine ⟨rfl, l, h1, by omega, fun j hj => ⟨fun hs => ?_, fun hjl => hl j hjl⟩⟩
          by_contra hjl
          exact hr j (by omega) hj hs)
    rw [iterW_empty] at hw
    refine Safe.bind_ro (hw.mono (fun st m' hq => ?_) le_rfl)
    obtain ⟨a, b⟩ := st
    exact safe_ret hq
  · rw [ite_eq_right (by simpa using h0)]
    refine safe_ret ⟨rfl, 0, IntV.toInt_ofInt_zero, by omega, fun j hj => ⟨fun hs => ?_, fun h => by omega⟩⟩
    exact (h0 (hs0 ▸ hpre.2.2.2.1 0 j (by omega) hj hs)).elim

end Claims.EpSrc2Dst
