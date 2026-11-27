import Proof.Float

/-!
# Footprints, the post-condition, and facts about the lengths

The one program reads `seq_lens[0 … num_seq)` (`Rd`), writes `num_kv_splits[0 … num_seq·num_group)`
(`Wout`) and, with `uneven_batch`, its element `0` (`Wu`). `Qf` is what the final memory holds.
The lengths' width `wOf d.dL` is `32` or `64`; `toW` and `ofW` are identities there.
-/

namespace Claims.NumKvSplits

open TVA

/-! ## The lengths' dtype -/

/-- The width is at least `32`. -/
theorem w_ge {dt : DType} (h : dt = .i64 ∨ dt = .i32) : 32 ≤ wOf dt := by
  rcases h with rfl | rfl <;> decide

/-- `2^31 ≤ 2^(w-1)` for the lengths' width. -/
theorem pow_w {dt : DType} (h : dt = .i64 ∨ dt = .i32) : (2 : ℕ) ^ 31 ≤ 2 ^ (wOf dt - 1) :=
  Nat.pow_le_pow_right (by norm_num) (by have := w_ge h; omega)

/-- `2^31 ≤ 2^(w-1)` in `ℤ`. -/
theorem pow_wZ {dt : DType} (h : dt = .i64 ∨ dt = .i32) :
    (2 : ℤ) ^ 31 ≤ 2 ^ (wOf dt - 1) := by exact_mod_cast pow_w h

/-- `toW` undoes `ofW`. -/
theorem toW_ofW {dt : DType} (h : dt = .i64 ∨ dt = .i32) (x : IntV true (wOf dt)) :
    toW dt (ofW dt x) = x := by
  rcases h with rfl | rfl <;> rfl

/-- The value of an element is that of its signed integer. -/
theorem ivOf_eq {dt : DType} (h : dt = .i64 ∨ dt = .i32) (x : dt.Val) :
    ivOf x = (toW dt x).toInt := by
  rcases h with rfl | rfl <;> rfl

/-- The value a buffer element holds, read from memory. -/
theorem valOf_of_read {dt : DType} {m : Mem} {b : Buf dt} {i : Nat} {x : dt.Val}
    (h : m.read dt (b.ptr (i : Int)).addr = some x) : valOf m b i = ivOf x := by
  unfold valOf; rw [h]

/-- A value that fits `int32` fits the lengths' width. -/
theorem sfits_w {dt : DType} (h : dt = .i64 ∨ dt = .i32) {x : ℤ} (h0 : 0 ≤ x)
    (h1 : x < 2 ^ 31) : IntV.SFits (wOf dt) x := by
  have := pow_wZ h
  rw [IntV.sfits_iff]; constructor <;> linarith

/-! ## Footprints and the post-condition -/

/-- The program reads the batch's lengths. -/
def Rd (d : Inst) : Set Int := d.seq.elems {j | j < d.S}

/-- The output elements it writes. -/
def Wout (d : Inst) : Set Int := d.out.elems {j | j < d.S * d.G}

/-- `uneven_batch`'s element `0`, when given. -/
def Wu (d : Inst) : Set Int := if d.hasUneven then d.uneven.elems {0} else ∅

/-- The outputs `Post` states, in a memory `m`. -/
def Qout (d : Inst) (init m : Mem) : Prop :=
  ∃ E : Int, CoreOk d init E ∧ ∀ j : Nat, j < d.S * d.G → ∃ v : I32,
    m.read .i32 (d.out.ptr (j : Int)).addr = some v ∧ v.toInt = numSplits d init E (j / d.G)

/-- The uneven flag `Post` states, in a memory `m`. -/
def Qu (d : Inst) (init m : Mem) : Prop :=
  d.hasUneven = true → ∃ u : I32, m.read .i32 (d.uneven.ptr 0).addr = some u ∧
    u.toInt = if minLen d init < maxLen d init then 1 else 0

/-- What the program leaves. -/
def Qf (d : Inst) (init m : Mem) : Prop := Qout d init m ∧ Qu d init m

/-! ## Folds of `max` and `min` -/

section Folds
variable {α : Type} {sg : Bool} {w : Nat}

/-- A fold of `max` is at least its start and every element, and is one of them. -/
theorem foldl_max_spec (f : α → ℤ) : ∀ (l : List α) (a : ℤ),
    a ≤ l.foldl (fun b x => max b (f x)) a ∧ (∀ x ∈ l, f x ≤ l.foldl (fun b x => max b (f x)) a) ∧
      (l.foldl (fun b x => max b (f x)) a = a ∨ ∃ x ∈ l, l.foldl (fun b x => max b (f x)) a = f x)
  | [], a => ⟨le_rfl, by simp, Or.inl rfl⟩
  | y :: l, a => by
    obtain ⟨h1, h2, h3⟩ := foldl_max_spec f l (max a (f y))
    simp only [List.foldl_cons, List.mem_cons, forall_eq_or_imp, exists_eq_or_imp]
    refine ⟨le_trans (le_max_left _ _) h1, ⟨le_trans (le_max_right _ _) h1, h2⟩, ?_⟩
    rcases h3 with h | h
    · rcases le_total a (f y) with hy | hy
      · exact Or.inr (Or.inl (by rw [h, max_eq_right hy]))
      · exact Or.inl (by rw [h, max_eq_left hy])
    · exact Or.inr (Or.inr h)

/-- A fold of `min` is at most its start and every element, and is one of them. -/
theorem foldl_min_spec (f : α → ℤ) : ∀ (l : List α) (a : ℤ),
    l.foldl (fun b x => min b (f x)) a ≤ a ∧ (∀ x ∈ l, l.foldl (fun b x => min b (f x)) a ≤ f x) ∧
      (l.foldl (fun b x => min b (f x)) a = a ∨ ∃ x ∈ l, l.foldl (fun b x => min b (f x)) a = f x)
  | [], a => ⟨le_rfl, by simp, Or.inl rfl⟩
  | y :: l, a => by
    obtain ⟨h1, h2, h3⟩ := foldl_min_spec f l (min a (f y))
    simp only [List.foldl_cons, List.mem_cons, forall_eq_or_imp, exists_eq_or_imp]
    refine ⟨le_trans h1 (min_le_left _ _), ⟨le_trans h1 (min_le_right _ _), h2⟩, ?_⟩
    rcases h3 with h | h
    · rcases le_total a (f y) with hy | hy
      · exact Or.inl (by rw [h, min_eq_left hy])
      · exact Or.inr (Or.inl (by rw [h, min_eq_right hy]))
    · exact Or.inr (Or.inr h)

/-- The value of an `IntV.max` fold is the fold of `max` on values. -/
theorem foldl_imax_toInt : ∀ (l : List (IntV sg w)) (a : IntV sg w),
    (l.foldl IntV.max a).toInt = l.foldl (fun b x => max b x.toInt) a.toInt
  | [], _ => rfl
  | y :: l, a => by
    simp only [List.foldl_cons]
    rw [foldl_imax_toInt l (a.max y), IntV.toInt_max]

/-- The value of an `IntV.min` fold is the fold of `min` on values. -/
theorem foldl_imin_toInt : ∀ (l : List (IntV sg w)) (a : IntV sg w),
    (l.foldl IntV.min a).toInt = l.foldl (fun b x => min b x.toInt) a.toInt
  | [], _ => rfl
  | y :: l, a => by
    simp only [List.foldl_cons]
    rw [foldl_imin_toInt l (a.min y), IntV.toInt_min]

/-- **`tl.max` of a rank-1 integer block**: at least its start and every lane, and one of them. -/
theorem foldAll_max_spec {n : Nat} (x : Tensor [n] (IntV sg w)) (a : IntV sg w) :
    a.toInt ≤ (Tensor.foldAll IntV.max a x).toInt ∧
      (∀ i : Fin n, (x.get1 i).toInt ≤ (Tensor.foldAll IntV.max a x).toInt) ∧
      ((Tensor.foldAll IntV.max a x).toInt = a.toInt ∨
        ∃ i : Fin n, (Tensor.foldAll IntV.max a x).toInt = (x.get1 i).toInt) := by
  unfold Tensor.foldAll
  rw [foldl_imax_toInt]
  obtain ⟨h1, h2, h3⟩ := foldl_max_spec (fun y : IntV sg w => y.toInt) x.toList a.toInt
  refine ⟨h1, fun i => h2 _ (List.mem_map.mpr ⟨(i, ()), Idx.mem_all _ _, rfl⟩), ?_⟩
  rcases h3 with h | ⟨y, hy, h⟩
  · exact Or.inl h
  · obtain ⟨⟨i, ⟨⟩⟩, -, rfl⟩ := List.mem_map.mp hy
    exact Or.inr ⟨i, h⟩

/-- **`tl.min` of a rank-1 integer block**: at most its start and every lane, and one of them. -/
theorem foldAll_min_spec {n : Nat} (x : Tensor [n] (IntV sg w)) (a : IntV sg w) :
    (Tensor.foldAll IntV.min a x).toInt ≤ a.toInt ∧
      (∀ i : Fin n, (Tensor.foldAll IntV.min a x).toInt ≤ (x.get1 i).toInt) ∧
      ((Tensor.foldAll IntV.min a x).toInt = a.toInt ∨
        ∃ i : Fin n, (Tensor.foldAll IntV.min a x).toInt = (x.get1 i).toInt) := by
  unfold Tensor.foldAll
  rw [foldl_imin_toInt]
  obtain ⟨h1, h2, h3⟩ := foldl_min_spec (fun y : IntV sg w => y.toInt) x.toList a.toInt
  refine ⟨h1, fun i => h2 _ (List.mem_map.mpr ⟨(i, ()), Idx.mem_all _ _, rfl⟩), ?_⟩
  rcases h3 with h | ⟨y, hy, h⟩
  · exact Or.inl h
  · obtain ⟨⟨i, ⟨⟩⟩, -, rfl⟩ := List.mem_map.mp hy
    exact Or.inr ⟨i, h⟩

end Folds

/-! ## The lengths -/

/-- `maxLen` is at least every length and is `0` or one of them. -/
theorem maxLen_spec (d : Inst) (m : Mem) :
    (∀ s, s < d.S → len d m s ≤ maxLen d m) ∧ 0 ≤ maxLen d m ∧
      (maxLen d m = 0 ∨ ∃ s, s < d.S ∧ maxLen d m = len d m s) := by
  obtain ⟨h1, h2, h3⟩ := foldl_max_spec (len d m) (List.range d.S) 0
  refine ⟨fun s hs => h2 s (List.mem_range.mpr hs), h1, ?_⟩
  rcases h3 with h | ⟨s, hs, h⟩
  · exact Or.inl h
  · exact Or.inr ⟨s, List.mem_range.mp hs, h⟩

/-- `minLen0` is at most every length and `maxLen`, and is `maxLen` or one of the lengths. -/
theorem minLen0_spec (d : Inst) (m : Mem) :
    (∀ s, s < d.S → minLen0 d m ≤ len d m s) ∧ minLen0 d m ≤ maxLen d m ∧
      (minLen0 d m = maxLen d m ∨ ∃ s, s < d.S ∧ minLen0 d m = len d m s) := by
  obtain ⟨h1, h2, h3⟩ := foldl_min_spec (len d m) (List.range d.S) (maxLen d m)
  refine ⟨fun s hs => h2 s (List.mem_range.mpr hs), h1, ?_⟩
  rcases h3 with h | ⟨s, hs, h⟩
  · exact Or.inl h
  · exact Or.inr ⟨s, List.mem_range.mp hs, h⟩

/-- With the covered lengths, `maxLen` is a length in `[1, 2^22]`. -/
theorem maxLen_bounds {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) :
    1 ≤ maxLen d m ∧ maxLen d m ≤ 2 ^ 22 ∧ ∃ s, s < d.S ∧ maxLen d m = len d m s := by
  obtain ⟨h1, -, h3⟩ := maxLen_spec d m
  have h0 := (hpre.2.2.2 0 hd.S_pos).1
  have hge : 1 ≤ maxLen d m := le_trans h0 (h1 0 hd.S_pos)
  rcases h3 with h | ⟨s, hs, h⟩
  · omega
  · exact ⟨hge, by rw [h]; exact (hpre.2.2.2 s hs).2, s, hs, h⟩

/-- With the covered lengths, `minLen0` is in `[1, maxLen]`. -/
theorem minLen0_bounds {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) :
    1 ≤ minLen0 d m ∧ minLen0 d m ≤ maxLen d m := by
  obtain ⟨-, h2, h3⟩ := minLen0_spec d m
  obtain ⟨hM, -, -⟩ := maxLen_bounds hd hpre
  refine ⟨?_, h2⟩
  rcases h3 with h | ⟨s, hs, h⟩
  · rw [h]; exact hM
  · rw [h]; exact (hpre.2.2.2 s hs).1

/-- `minLen` is in `[1, maxLen]`. -/
theorem minLen_bounds {d : Inst} (hd : d.ok) {m : Mem} (hpre : Pre d m) :
    1 ≤ minLen d m ∧ minLen d m ≤ maxLen d m := by
  obtain ⟨h1, h2⟩ := minLen0_bounds hd hpre
  obtain ⟨hM, -, -⟩ := maxLen_bounds hd hpre
  unfold minLen
  split_ifs <;> constructor <;> omega

end Claims.NumKvSplits
