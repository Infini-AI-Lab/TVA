import Proof.Ints

/-!
# The two loads of `seq_lens` and the folds `tl.max`, `tl.min`

Lane `i` of `seq_lens_ptr + offs_seq` is element `i`, enabled for `i < num_seq` (`seq_addrs`).
The first load's lanes hold the lengths and `0` beyond the batch; its `tl.max` is `maxLen`
(`max_val`). The second load's lanes hold the lengths and `maxLen` beyond; its `tl.min` is
`minLen0` (`min_val`).
-/

namespace Claims.NumKvSplits

open TVA

variable {α : Type} {E : Env} {R W : Set Int} {Q : α → Mem → Prop}

/-- Lane `i` of `tl.arange(0, MAX_NUM_SEQ)` has value `i`. -/
theorem arange_val {d : Inst} (hd : d.ok) (i : Idx [d.MAX - 0]) :
    ((tl.arange 0 d.MAX).get i).toInt = (i.1.val : ℤ) := by
  rcases i with ⟨i, ⟨⟩⟩
  exact IntV.I32.toInt_tl_arange (le_trans hd.MAX_le (by norm_num)) i

/-- `num_seq` as an `int32`. -/
theorem S_val {d : Inst} (hd : d.ok) : (IntV.ofNat d.S : I32).toInt = d.S :=
  IntV.I32.toInt_ofNat (by have := hd.S_le; omega)

/-- **The load's lanes**: lane `i` reads element `i` of `seq_lens`, enabled for `i < num_seq`. -/
theorem seq_addrs {d : Inst} (hd : d.ok) :
    d.seq.Addrs (tl.addrs (d.seq.ptr 0 + tl.arange 0 d.MAX)
      (some ((tl.arange 0 d.MAX).ltS (IntV.ofNat d.S)))) (fun i => i.1.val)
      (fun i => decide (i.1.val < d.S)) := by
  refine Buf.addrs_of_get d.seq (fun i => ?_) (fun i _ => ?_)
  · rw [Tensor.get_ltS, IntV.lt_eq, arange_val hd i, S_val hd]
    simp only [Nat.cast_lt]
  · show (d.seq.ptr 0).offset _ = _
    rw [arange_val hd i, Buf.ptr_offset, zero_add]

/-- **A load of `seq_lens`** (source lines 73 and 75), with `other = o`: the batch's lanes hold
their elements of the memory, the others `o`. -/
theorem safe_load_seq {d : Inst} (hd : d.ok) {m : Mem} (hv : d.seq.Valid m) (hR : Rd d ⊆ R)
    (o : d.dL.Val) {k : Tensor [d.MAX - 0] d.dL.Val → Prog α}
    (hk : ∀ v : Tensor [d.MAX - 0] d.dL.Val,
      (∀ i : Idx [d.MAX - 0], i.1.val < d.S →
        m.read d.dL (d.seq.ptr (i.1.val : ℤ)).addr = some (v.get i)) →
      (∀ i : Idx [d.MAX - 0], d.S ≤ i.1.val → v.get i = o) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load (d.seq.ptr 0 + tl.arange 0 d.MAX)
      (some ((tl.arange 0 d.MAX).ltS (IntV.ofNat d.S))) (some (Tensor.full o)) >>= k) := by
  refine safe_load_buf d.seq (seq_addrs hd) (fun i hi => ?_) hv ?_ fun v h1 h2 => hk v ?_ ?_
  · simp only [decide_eq_true_eq] at hi
    exact lt_of_lt_of_le hi hd.seq_len
  · rintro x ⟨j, ⟨i, hi, rfl⟩, hx⟩
    simp only [decide_eq_true_eq] at hi
    exact hR ⟨i.1.val, hi, hx⟩
  · intro i hi
    exact h1 i (by simpa using hi)
  · intro i hi
    exact h2 i (by simpa using hi) _ rfl

/-- The value of a batch lane: the request's length. -/
theorem lane_len {d : Inst} (hd : d.ok) {m : Mem} {i : Nat} {x : d.dL.Val}
    (h : m.read d.dL (d.seq.ptr (i : ℤ)).addr = some x) : (toW d.dL x).toInt = len d m i := by
  rw [len, valOf_of_read h, ivOf_eq hd.dL_ok]

/-- `wMin` has the smallest value. -/
theorem wMin_val {dt : DType} (h : dt = .i64 ∨ dt = .i32) :
    (wMin dt).toInt = -2 ^ (wOf dt - 1) := by
  have hw := w_ge h
  exact IntV.toInt_ofInt (by omega) (by rw [IntV.sfits_iff]; constructor <;> norm_num)

/-- `wMax` has the largest value. -/
theorem wMax_val {dt : DType} (h : dt = .i64 ∨ dt = .i32) :
    (wMax dt).toInt = 2 ^ (wOf dt - 1) - 1 := by
  have hw := w_ge h
  have : (0 : ℤ) < 2 ^ (wOf dt - 1) := by positivity
  exact IntV.toInt_ofInt (by omega) (by rw [IntV.sfits_iff]; constructor <;> omega)

/-- **Line 74**: the first load's `tl.max` is `maxLen`. -/
theorem max_val {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {v : Tensor [d.MAX - 0] d.dL.Val}
    (h1 : ∀ i : Idx [d.MAX - 0], i.1.val < d.S →
      init.read d.dL (d.seq.ptr (i.1.val : ℤ)).addr = some (v.get i))
    (h2 : ∀ i : Idx [d.MAX - 0], d.S ≤ i.1.val → v.get i = ofW d.dL (IntV.ofInt 0)) :
    (Tensor.foldAll IntV.max (wMin d.dL) (v.map (toW d.dL))).toInt = maxLen d init := by
  have hlane : ∀ i : Fin (d.MAX - 0), ((v.map (toW d.dL)).get1 i).toInt =
      if i.val < d.S then len d init i.val else 0 := by
    intro i
    show (toW d.dL (v.get (i, ()))).toInt = _
    split_ifs with hi
    · exact lane_len hd (h1 (i, ()) hi)
    · rw [h2 (i, ()) (Nat.le_of_not_lt hi), toW_ofW hd.dL_ok]; exact IntV.toInt_ofInt_zero
  obtain ⟨-, hge, hone⟩ := foldAll_max_spec (v.map (toW d.dL)) (wMin d.dL)
  obtain ⟨hM1, -, s, hs, hMs⟩ := maxLen_bounds hd hpre
  obtain ⟨hle, -, -⟩ := maxLen_spec d init
  have hsM : s < d.MAX - 0 := by have := hd.S_MAX; omega
  apply le_antisymm
  · rcases hone with h | ⟨i, h⟩
    · rw [h, wMin_val hd.dL_ok]
      have : (0 : ℤ) < 2 ^ (wOf d.dL - 1) := by positivity
      omega
    · rw [h, hlane i]
      split_ifs with hi
      · exact hle _ hi
      · omega
  · have := hge ⟨s, hsM⟩
    rw [hlane ⟨s, hsM⟩, ite_eq_left hs, ← hMs] at this
    exact this

/-- **Line 76**: the second load's `tl.min` is `minLen0`. -/
theorem min_val {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {v : Tensor [d.MAX - 0] d.dL.Val} {mx : IntV true (wOf d.dL)}
    (hmx : mx.toInt = maxLen d init)
    (h1 : ∀ i : Idx [d.MAX - 0], i.1.val < d.S →
      init.read d.dL (d.seq.ptr (i.1.val : ℤ)).addr = some (v.get i))
    (h2 : ∀ i : Idx [d.MAX - 0], d.S ≤ i.1.val → v.get i = ofW d.dL mx) :
    (Tensor.foldAll IntV.min (wMax d.dL) (v.map (toW d.dL))).toInt = minLen0 d init := by
  have hlane : ∀ i : Fin (d.MAX - 0), ((v.map (toW d.dL)).get1 i).toInt =
      if i.val < d.S then len d init i.val else maxLen d init := by
    intro i
    show (toW d.dL (v.get (i, ()))).toInt = _
    split_ifs with hi
    · exact lane_len hd (h1 (i, ()) hi)
    · rw [h2 (i, ()) (Nat.le_of_not_lt hi), toW_ofW hd.dL_ok, hmx]
  obtain ⟨-, hle, hone⟩ := foldAll_min_spec (v.map (toW d.dL)) (wMax d.dL)
  obtain ⟨-, hM2, s, hs, hMs⟩ := maxLen_bounds hd hpre
  obtain ⟨hge, hmM, hmone⟩ := minLen0_spec d init
  have hlb := (minLen0_bounds hd hpre).1
  have hsM : s < d.MAX - 0 := by have := hd.S_MAX; omega
  apply le_antisymm
  · rcases hmone with h | ⟨t, ht, h⟩
    · have := hle ⟨s, hsM⟩
      rw [hlane ⟨s, hsM⟩, ite_eq_left hs, ← hMs] at this
      rw [h]; exact this
    · have htM : t < d.MAX - 0 := by have := hd.S_MAX; omega
      have := hle ⟨t, htM⟩
      rw [hlane ⟨t, htM⟩, ite_eq_left ht, ← h] at this
      exact this
  · rcases hone with h | ⟨i, h⟩
    · rw [h, wMax_val hd.dL_ok]
      have := pow_wZ hd.dL_ok
      omega
    · rw [h, hlane i]
      split_ifs with hi
      · exact hge _ hi
      · exact hmM

end Claims.NumKvSplits
