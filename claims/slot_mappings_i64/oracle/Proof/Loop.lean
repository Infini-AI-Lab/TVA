import Proof.Basic

/-!
# `slot_mappings_i64`: the token loops

Both loops of the kernel run `for i in range(lo, hi, 1024)` over a token range `[L, H)` of one
group's row, trip `t` storing the elements `(g, x)` for `x` in `[L + 1024 t, min H (L + 1024 t +
1024))`. `safe_tokLoop` turns a proof of one trip into one of the loop.
-/

namespace Claims.SlotMappings

open TVA

/-- The number of trips of `range(L, H, 1024)`. -/
theorem lt_tripCount_iff {L H : Int} (t : Nat) :
    t < tl.tripCount L H 1024 ↔ L + 1024 * t < H := by
  unfold tl.tripCount
  constructor
  · intro h
    have h1 : (t : Int) < (H - L + 1024 - 1) / 1024 := by
      have : ((t : Nat) : Int) < (((H - L + 1024 - 1) / 1024).toNat : Int) := by exact_mod_cast h
      rw [Int.toNat_of_nonneg (by omega)] at this
      exact this
    omega
  · intro h
    have h1 : (t : Int) < (H - L + 1024 - 1) / 1024 := by omega
    have h2 : ((t : Nat) : Int) < (((H - L + 1024 - 1) / 1024).toNat : Int) := by
      rw [Int.toNat_of_nonneg (by omega)]; exact h1
    exact_mod_cast h2

/-- Trip `t` writes the elements of `[L + 1024 t, min H (L + 1024 t + 1024))`. -/
def tripW (d : Inst) (g : Nat) (L H : Int) (t : Nat) : Set Int :=
  d.slot_mappings.elems (tok d g (L + 1024 * t) (min H (L + 1024 * t + 1024)))

/-- Two different tokens of one group are different elements. -/
theorem span_ne {d : Inst} {g x y : Nat} (h : x ≠ y) :
    Disjoint (Mem.span .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr)
      (Mem.span .i64 (d.slot_mappings.ptr ((g * d.S + y : Nat) : Int)).addr) :=
  d.slot_mappings.span_disjoint (by push_cast; omega)

/-- An element outside `tok d g L H` keeps its value through a store with that frame. -/
theorem read_frame {d : Inst} {g x : Nat} {L H : Int} {m m' : Mem}
    (hx : ¬ (L ≤ (x : Int) ∧ (x : Int) < H))
    (hfr : ∀ z, z ∉ d.slot_mappings.elems (tok d g L H) → m' z = m z) :
    m'.read .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr =
      m.read .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr := by
  refine Mem.read_congr fun z hz => hfr z ?_
  rintro ⟨j, ⟨y, hy1, hy2, rfl⟩, hzj⟩
  have hxy : x ≠ y := by rintro rfl; exact hx ⟨hy1, hy2⟩
  exact Set.disjoint_left.mp (span_ne hxy) hz hzj

/-- **The token loop.** If each trip `t` stores `v x` at every element `(g, x)` of its range
(from any memory of `Inv`), changes nothing else and keeps every byte present, then the loop
`for i in range(lo, hi, 1024)` over `[L, H)` leaves `v x` at every element `(g, x)` of `[L, H)`. -/
theorem safe_tokLoop {d : Inst} (hd : d.ok) {init : Mem} {E : Env} {Rr : Set Int} {g : Nat}
    (hg : g < d.G) {lo hi : I32} {L H : Int} (hlo : lo.toInt = L) (hhi : hi.toInt = H)
    (hL : 0 ≤ L) (hH : H ≤ d.N) (v : Nat → I64) {body : I32 → Unit → Prog Unit}
    (htrip : ∀ t : Nat, L + 1024 * t < H → ∀ m, Inv d init m →
      Safe E Rr (tripW d g L H t)
        (fun _ m' => (∀ x : Nat, L + 1024 * t ≤ x → (x : Int) < min H (L + 1024 * t + 1024) →
            m'.read .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr = some (v x)) ∧
          (∀ z, z ∉ tripW d g L H t → m' z = m z) ∧ (∀ z, m z ≠ none → m' z ≠ none))
        m (body (IntV.ofInt (lo.toInt + (t : Int) * (IntV.ofNat 1024 : I32).toInt)) ()))
    {m : Mem} (hm : Inv d init m) :
    Safe E Rr (d.slot_mappings.elems (tok d g L H))
      (fun _ m' => ∀ x : Nat, L ≤ x → (x : Int) < H →
        m'.read .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr = some (v x)) m
      (tl.forRange lo hi (IntV.ofNat 1024 : I32) () body >>= fun _ => pure ()) := by
  have hN := hd.N_le
  have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by norm_num [IntV.maxInt]
  have hrange : tl.RangeOk lo hi (IntV.ofNat 1024 : I32) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [toInt_1024]; norm_num
    · rw [hhi, toInt_1024, hmax]
      have : (d.N : Int) ≤ 2 ^ 22 := by exact_mod_cast hN
      omega
    · rw [hmax]
      have : lo.toInt < 2 ^ 31 := (IntV.sfits_toInt lo).2
      omega
    · rw [toInt_1024, hmax]; norm_num
  have hn : ∀ t, t < tl.tripCount lo.toInt hi.toInt (IntV.ofNat 1024 : I32).toInt ↔
      L + 1024 * t < H := by
    intro t; rw [hlo, hhi, toInt_1024]; exact lt_tripCount_iff t
  set n := tl.tripCount lo.toInt hi.toInt (IntV.ofNat 1024 : I32).toInt with hn_def
  let I : Nat → Unit → Mem → Prop := fun t _ m₁ =>
    Inv d init m₁ ∧ ∀ x : Nat, L ≤ x → (x : Int) < min H (L + 1024 * t) →
      m₁.read .i64 (d.slot_mappings.ptr ((g * d.S + x : Nat) : Int)).addr = some (v x)
  have hsub : ∀ t, tripW d g L H t ⊆ d.slot_mappings.bytes := fun t =>
    Buf.elems_sub_bytes _ (tok_lt hd hg (le_trans (min_le_left _ _) hH))
  refine Safe.mono (W := iterW (tripW d g L H) n ∪ ∅) ?_ (fun _ _ h => h) ?_
  · refine safe_forRange (k := fun _ => pure ()) hrange I (tripW d g L H) ?_
      (Set.disjoint_empty _) ⟨hm, fun x h1 h2 => absurd h2 (by omega)⟩ ?_ ?_
    · intro t u htu _
      refine Buf.elems_disjoint _ (Set.disjoint_left.mpr ?_)
      rintro j ⟨x, hx1, hx2, rfl⟩ ⟨y, hy1, hy2, hxy⟩
      have : x = y := by omega
      subst this
      have := lt_of_lt_of_le hx2 (min_le_right _ _)
      have : (t : Int) + 1 ≤ u := by exact_mod_cast htu
      omega
    · intro t ht _ m₁ ⟨hinv, hprev⟩
      refine Safe.mono (htrip t ((hn t).1 ht) m₁ hinv) ?_ (le_refl _)
      rintro _ m₂ ⟨hnew, hfr, hpres⟩
      refine ⟨⟨fun z hz => ?_, Buf.valid_of_agree hinv.2 fun z hz => hpres z (hinv.2.2.2.2 z hz)⟩, ?_⟩
      · rw [hfr z (fun h => hz (hsub t h))]; exact hinv.1 z hz
      · intro x h1 h2
        by_cases hx : L + 1024 * t ≤ (x : Int)
        · exact hnew x hx (by rw [lt_min_iff] at h2 ⊢; omega)
        · rw [read_frame (by omega) hfr]
          exact hprev x h1 (by rw [lt_min_iff] at h2 ⊢; omega)
    · intro _ m₁ ⟨_, hall⟩
      refine Safe.ret ?_
      intro x h1 h2
      have hnn : ¬ n < n := lt_irrefl n
      rw [hn] at hnn
      exact hall x h1 (by rw [lt_min_iff]; omega)
  · intro z hz
    rcases hz with ⟨t, -, ⟨j, ⟨x, hx1, hx2, rfl⟩, hzj⟩⟩ | hz
    · refine ⟨g * d.S + x, ⟨x, by omega, lt_of_lt_of_le hx2 (min_le_left _ _), rfl⟩, hzj⟩
    · exact absurd hz (Set.notMem_empty z)

end Claims.SlotMappings
