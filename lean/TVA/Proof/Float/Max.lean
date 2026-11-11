import TVA.Proof.Float.Error

/-!
# The float maximum and `−∞`

Facts about `tl.maximum`, `tl.minimum`, the `tl.max`/`tl.min` reductions, and a running maximum
that starts at `-inf` (an online softmax):

* Comparisons by value: `FloatV.le_iff`, `lt_iff`, `eq_iff` (operands not NaN), `le_refl'`,
  `le_trans'`.
* The reference `max`/`min` of two non-NaN values is one of them (`max_mem`, `min_mem`) with the
  larger (smaller) value (`toRat_max`, `toRat_min`).
* Every result `tl.maximum`/`tl.minimum` may give (`MaxRel`) on non-NaN operands is one of them
  (`MaxRel.mem`), not below (`MaxRel.le`) or not above (`MaxRel.ge`) either, with value
  `max`/`min` of theirs (`MaxRel.toRat_eq_max`, `MaxRel.toReal_eq_max`, …).
* A `tl.max`/`tl.min` reduction along any tree of non-NaN leaves is one of its leaves, and no leaf
  is larger (smaller): `Tree.eval_maxRel`, `Tree.eval_maxRel_toReal`, `Tree.eval_minRel_toReal`.
* **The maximum of approximations approximates the maximum**: if `|xᵢ − Aᵢ| ≤ δ` for each operand
  or leaf, the result is within `δ` of the largest `Aᵢ` (`abs_max_sub_max_le_of`,
  `FloatV.max_approx`, `MaxRel.max_approx`, `Tree.eval_maxRel_approx`,
  `Tree.eval_maxRel_approx_sup'`, and the `min` forms).
* `−∞` (`FloatV.negInf`) of a format with infinities: below every non-NaN value
  (`negInf_toRat_le`), `max(−∞, x) = x` (`max_negInf_left`, `MaxRel.negInf_left`), `−∞ − x = −∞`
  and `x − (−∞) = +∞` for finite `x` (`sub_negInf_left`, `sub_negInf_right`), `−∞ − (−∞)` is NaN
  (`sub_negInf_negInf`).

Every lemma assumes its operands are not NaN: a NaN operand of `tl.maximum` is dropped (or, with
`propagate_nan`, propagated), which a specification states separately.
-/
namespace TVA

namespace FloatV

variable {f : Format}

/-! ## Comparisons by value -/

/-- `x <= y` on non-NaN values is the order of their values. -/
theorem le_iff {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    le x y = true ↔ x.toRat ≤ y.toRat := by
  unfold le; rw [Format.cmp_of_not_nan hx hy]
  change _ ↔ f.toRat x.pat ≤ f.toRat y.pat
  by_cases h1 : f.toRat x.pat < f.toRat y.pat
  · simp [h1, h1.le]
  · by_cases h2 : f.toRat x.pat = f.toRat y.pat
    · simp [h2]
    · have h3 : ¬ (f.toRat x.pat ≤ f.toRat y.pat) := not_le.mpr (lt_of_le_of_ne (not_lt.mp h1) (Ne.symm h2))
      rw [ite_eq_right h1, ite_eq_right h2]
      exact ⟨fun h => absurd h (by decide), fun h => absurd h h3⟩

/-- `x < y` on non-NaN values is the strict order of their values. -/
theorem lt_iff {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    lt x y = true ↔ x.toRat < y.toRat := by
  unfold lt; rw [Format.cmp_of_not_nan hx hy]
  unfold toRat
  split_ifs with h1 h2 <;> simp_all

/-- `x == y` on non-NaN values is the equality of their values (`-0 == +0`). -/
theorem eq_iff {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    eq x y = true ↔ x.toRat = y.toRat := by
  unfold eq; rw [Format.cmp_of_not_nan hx hy]
  unfold toRat
  split_ifs with h1 h2 <;> simp_all
  exact ne_of_lt h1

/-- `x <= x` holds for a non-NaN `x`. -/
theorem le_refl' {x : FloatV f} (hx : x.isNaN = false) : le x x = true := (le_iff hx hx).mpr le_rfl

/-- `<=` is transitive on non-NaN values. -/
theorem le_trans' {x y z : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false)
    (hz : z.isNaN = false) (h1 : le x y = true) (h2 : le y z = true) : le x z = true :=
  (le_iff hx hz).mpr (((le_iff hx hy).mp h1).trans ((le_iff hy hz).mp h2))

/-! ## The reference `max` and `min` -/

/-- The reference `max` of two non-NaN values is one of them. -/
theorem max_mem {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    max x y = x ∨ max x y = y := by
  unfold max Format.maxNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  split_ifs <;> simp [ofPat_pat]

/-- The reference `min` of two non-NaN values is one of them. -/
theorem min_mem {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    min x y = x ∨ min x y = y := by
  unfold min Format.minNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  split_ifs <;> simp [ofPat_pat]

/-- The reference `max` of two non-NaN values has the larger value. -/
theorem toRat_max {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    (max x y).toRat = Max.max x.toRat y.toRat := by
  unfold max Format.maxNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  unfold toRat
  split_ifs with h1 h2 h3 <;> simp only [ofPat_pat]
  · exact (max_eq_right h1.le).symm
  · exact (max_eq_left h2.le).symm
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, max_self]
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, max_self]

/-- The reference `min` of two non-NaN values has the smaller value. -/
theorem toRat_min {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    (min x y).toRat = Min.min x.toRat y.toRat := by
  unfold min Format.minNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  unfold toRat
  split_ifs with h1 h2 h3 <;> simp only [ofPat_pat]
  · exact (min_eq_left h1.le).symm
  · exact (min_eq_right h2.le).symm
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, min_self]
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, min_self]

/-- A value equal to one of two non-NaN values is not NaN. -/
theorem isNaN_of_mem {x y r : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false)
    (h : r = x ∨ r = y) : r.isNaN = false := by rcases h with rfl | rfl <;> assumption

/-! ## What `tl.maximum` and `tl.minimum` may return -/

/-- A result of `tl.maximum`/`tl.minimum` on non-NaN operands is one of them. -/
theorem MaxRel.mem {isMax : Bool} {x y r : FloatV f} (h : MaxRel isMax x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r = x ∨ r = y := by
  rcases h with h | ⟨-, -, h⟩
  · have hm : (if isMax then max x y else min x y) = x ∨ (if isMax then max x y else min x y) = y := by
      cases isMax
      · exact min_mem hx hy
      · exact max_mem hx hy
    have hn := isNaN_of_mem hx hy hm
    rw [h.eq hn]; exact hm
  · exact h

/-- A result of `tl.maximum` on non-NaN operands is not below either. -/
theorem MaxRel.le {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : FloatV.le x r = true ∧ FloatV.le y r = true := by
  have hmem := h.mem hx hy
  have hr := isNaN_of_mem hx hy hmem
  rcases h with h | ⟨hzx, hzy, -⟩
  · have hm := max_mem hx hy
    have hn := isNaN_of_mem hx hy hm
    simp only [ite_true] at h
    rw [h.eq hn]
    have := toRat_max hx hy
    exact ⟨(le_iff hx hn).mpr (by rw [this]; exact le_max_left _ _),
      (le_iff hy hn).mpr (by rw [this]; exact le_max_right _ _)⟩
  · have h0x := toRat_of_isZero (x := x) hzx
    have h0y := toRat_of_isZero (x := y) hzy
    have h0r : r.toRat = 0 := by rcases hmem with rfl | rfl <;> assumption
    exact ⟨(le_iff hx hr).mpr (by rw [h0x, h0r]), (le_iff hy hr).mpr (by rw [h0y, h0r])⟩

/-- A result of `tl.minimum` on non-NaN operands is not above either. -/
theorem MaxRel.ge {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : FloatV.le r x = true ∧ FloatV.le r y = true := by
  have hmem := h.mem hx hy
  have hr := isNaN_of_mem hx hy hmem
  rcases h with h | ⟨hzx, hzy, -⟩
  · have hm := min_mem hx hy
    have hn := isNaN_of_mem hx hy hm
    simp only [Bool.false_eq_true, ite_false] at h
    rw [h.eq hn]
    have := toRat_min hx hy
    exact ⟨(le_iff hn hx).mpr (by rw [this]; exact min_le_left _ _),
      (le_iff hn hy).mpr (by rw [this]; exact min_le_right _ _)⟩
  · have h0x := toRat_of_isZero (x := x) hzx
    have h0y := toRat_of_isZero (x := y) hzy
    have h0r : r.toRat = 0 := by rcases hmem with rfl | rfl <;> assumption
    exact ⟨(le_iff hr hx).mpr (by rw [h0x, h0r]), (le_iff hr hy).mpr (by rw [h0y, h0r])⟩

end FloatV

/-- **A float max (min) reduction along any tree**, with no NaN leaf: the result is one of the
leaves, and every leaf is at most (at least) it. -/
theorem Tree.eval_maxRel {f : Format} {isMax : Bool} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel isMax) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false) :
    (∃ i ∈ t.leaves, r = x i) ∧
      ∀ i ∈ t.leaves, (if isMax then FloatV.le (x i) r else FloatV.le r (x i)) = true := by
  induction h with
  | leaf i =>
    refine ⟨⟨i, by simp [Tree.leaves], rfl⟩, fun j hj => ?_⟩
    simp only [Tree.leaves, List.mem_singleton] at hj
    rw [hj]
    have := hx i (by simp [Tree.leaves])
    cases isMax <;> simp [FloatV.le_refl' this]
  | @node l rt a b c _ _ hc ihl ihr =>
    have hxl : ∀ i ∈ l.leaves, (x i).isNaN = false := fun i hi => hx i (by simp [Tree.leaves, hi])
    have hxr : ∀ i ∈ rt.leaves, (x i).isNaN = false := fun i hi => hx i (by simp [Tree.leaves, hi])
    obtain ⟨⟨ia, hia, rfl⟩, hla⟩ := ihl hxl
    obtain ⟨⟨ib, hib, rfl⟩, hlb⟩ := ihr hxr
    have hna := hxl ia hia
    have hnb := hxr ib hib
    have hnc := FloatV.isNaN_of_mem hna hnb (hc.mem hna hnb)
    refine ⟨?_, fun j hj => ?_⟩
    · rcases hc.mem hna hnb with h | h
      · exact ⟨ia, by simp [Tree.leaves, hia], h⟩
      · exact ⟨ib, by simp [Tree.leaves, hib], h⟩
    · simp only [Tree.leaves, List.mem_append] at hj
      cases isMax with
      | true =>
        obtain ⟨h1, h2⟩ := hc.le hna hnb
        simp only [ite_true] at hla hlb ⊢
        rcases hj with hj | hj
        · exact FloatV.le_trans' (hxl j hj) hna hnc (hla j hj) h1
        · exact FloatV.le_trans' (hxr j hj) hnb hnc (hlb j hj) h2
      | false =>
        obtain ⟨h1, h2⟩ := hc.ge hna hnb
        simp only [Bool.false_eq_true, ite_false] at hla hlb ⊢
        rcases hj with hj | hj
        · exact FloatV.le_trans' hnc hna (hxl j hj) h1 (hla j hj)
        · exact FloatV.le_trans' hnc hnb (hxr j hj) h2 (hlb j hj)

namespace FloatV

variable {f : Format}

/-! ## `−∞` -/

/-- The pattern of `−∞`: the sign bit and `infMag`. -/
theorem negInf_pat : (negInf : FloatV f).pat = f.setSign true f.infMag :=
  pat_ofPat (Format.setSign_lt _ _)

/-- The pattern of `+∞`: `infMag`. -/
theorem posInf_pat : (posInf : FloatV f).pat = f.setSign false f.infMag :=
  pat_ofPat (Format.setSign_lt _ _)

/-- `−∞` is infinite (a format with infinities). -/
theorem negInf_isInf (hs : f.specials = .ieee) : (negInf : FloatV f).isInf = true := by
  unfold isInf; rw [negInf_pat, Format.isInf_setSign, Format.isInf_infMag hs]

/-- `+∞` is infinite (a format with infinities). -/
theorem posInf_isInf (hs : f.specials = .ieee) : (posInf : FloatV f).isInf = true := by
  unfold isInf; rw [posInf_pat, Format.isInf_setSign, Format.isInf_infMag hs]

/-- `−∞` is not NaN (a format with infinities). -/
theorem negInf_isNaN (hs : f.specials = .ieee) : (negInf : FloatV f).isNaN = false := by
  unfold isNaN; rw [negInf_pat, Format.isNaN_setSign]
  have := Format.isInf_infMag (f := f) hs
  unfold Format.isInf at this; unfold Format.isNaN
  rw [hs] at this ⊢
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  simp [this.1, this.2]

/-- `−∞` is negative. -/
theorem negInf_sign : (negInf : FloatV f).sign = true := by
  unfold sign; rw [negInf_pat, Format.sign_setSign]

/-- The magnitude of an infinity is positive. -/
theorem infMag_pos (hf : f.Regular) : 0 < f.infMag := by
  unfold Format.infMag
  exact Nat.mul_pos (by have := Format.two_pow_e_ge hf; omega) (by positivity)

/-- The magnitude of an infinity is below the sign bit. -/
theorem infMag_lt_signBit : f.infMag < f.signBit := by
  rw [Format.signBit_eq]; unfold Format.infMag
  exact Nat.mul_lt_mul_of_pos_right (by have := Nat.one_le_two_pow (n := f.e); omega) (by positivity)

/-- Every magnitude that is not NaN is at most the infinity's (`.ieee`). -/
theorem mag_le_infMag (hs : f.specials = .ieee) {n : ℕ} (hn : f.isNaN n = false) :
    f.mag n ≤ f.infMag := by
  have hm := f.mag_lt n
  rw [Format.signBit_eq] at hm
  unfold Format.isNaN Format.expField Format.fracField at hn
  rw [hs] at hn
  have hdm := Nat.div_add_mod (f.mag n) (2 ^ f.m)
  have hF : f.mag n % 2 ^ f.m < 2 ^ f.m := Nat.mod_lt _ (by positivity)
  have hE : f.mag n / 2 ^ f.m < 2 ^ f.e := by
    rw [Nat.div_lt_iff_lt_mul (by positivity)]; exact hm
  unfold Format.infMag
  by_cases h1 : f.mag n / 2 ^ f.m = 2 ^ f.e - 1
  · have h0 : f.mag n % 2 ^ f.m = 0 := by simp [h1] at hn; exact hn
    rw [h1, h0] at hdm
    rw [← hdm]; rw [mul_comm]; omega
  · have h2 : f.mag n / 2 ^ f.m + 1 ≤ 2 ^ f.e - 1 := by omega
    have h3 : 2 ^ f.m * (f.mag n / 2 ^ f.m + 1) ≤ 2 ^ f.m * (2 ^ f.e - 1) := Nat.mul_le_mul_left _ h2
    rw [mul_add, mul_one] at h3
    rw [mul_comm]
    omega

/-- `−∞` is below every value that is not NaN, and only `−∞` is equal to it. -/
theorem negInf_toRat_le (hf : f.Regular) (hs : f.specials = .ieee) {x : FloatV f} (hx : x.isNaN = false) :
    (negInf : FloatV f).toRat ≤ x.toRat ∧ ((negInf : FloatV f).toRat = x.toRat → x = negInf) := by
  have hk : f.orderKey (negInf : FloatV f).pat = -(f.infMag : ℤ) := by
    unfold Format.orderKey; rw [negInf_pat, Format.sign_setSign, Format.mag_setSign,
      Format.mag_of_lt infMag_lt_signBit]; rfl
  have hxk : -(f.infMag : ℤ) ≤ f.orderKey x.pat := by
    have := mag_le_infMag hs (n := x.pat) hx
    unfold Format.orderKey; split_ifs <;> omega
  constructor
  · unfold toRat
    rcases (lt_or_eq_of_le (hk ▸ hxk : f.orderKey (negInf : FloatV f).pat ≤ f.orderKey x.pat)) with h | h
    · exact ((Format.toRat_lt_iff_orderKey _ _).mpr h).le
    · exact ((Format.toRat_eq_iff_orderKey _ _).mpr h).le
  · intro heq
    have hkeq := (Format.toRat_eq_iff_orderKey _ _).mp heq
    rw [hk] at hkeq
    -- the key `-infMag` is only `−∞`'s
    have hmag := mag_le_infMag hs (n := x.pat) hx
    have hsx : f.sign x.pat = true := by
      unfold Format.orderKey at hkeq
      by_contra h
      simp only [Bool.not_eq_true] at h
      rw [h] at hkeq; simp at hkeq
      have : 0 < f.infMag := infMag_pos hf
      omega
    have hmx : f.mag x.pat = f.infMag := by
      unfold Format.orderKey at hkeq; rw [hsx] at hkeq; simp at hkeq; omega
    have : x.pat = (negInf : FloatV f).pat := by
      rw [negInf_pat, ← Format.eq_setSign (pat_lt x), hsx]
      unfold Format.setSign; rw [hmx, Format.mag_of_lt infMag_lt_signBit]
    exact ext_pat this

/-- **`max(−∞, x) = x`** for `x` not NaN. -/
theorem max_negInf_left (hf : f.Regular) (hs : f.specials = .ieee) {x : FloatV f}
    (hx : x.isNaN = false) : max negInf x = x := by
  rcases max_mem (negInf_isNaN hs) hx with h | h
  · obtain ⟨hle, heq⟩ := negInf_toRat_le hf hs hx
    have := toRat_max (negInf_isNaN hs) hx
    rw [h] at this
    have hval : (negInf : FloatV f).toRat = x.toRat := by
      rw [max_eq_right hle] at this; exact this
    rw [h, heq hval]
  · exact h

/-- **`tl.maximum(−∞, x)` is `x`** for `x` not NaN. -/
theorem MaxRel.negInf_left (hf : f.Regular) (hs : f.specials = .ieee) {x r : FloatV f}
    (h : MaxRel true negInf x r) (hx : x.isNaN = false) : r = x := by
  rcases h with h | ⟨hz, -, -⟩
  · simp only [ite_true] at h
    rw [max_negInf_left hf hs hx] at h
    exact h.eq hx
  · exfalso
    have := negInf_isInf (f := f) hs
    unfold isZero at hz; unfold isInf at this
    rw [negInf_pat] at hz this
    unfold Format.isZero at hz
    rw [Format.mag_setSign, Format.mag_of_lt infMag_lt_signBit] at hz
    have : 0 < f.infMag := infMag_pos hf
    simp at hz; omega

/-- **`−∞ − x = −∞`** for finite `x`. -/
theorem sub_negInf_left (hs : f.specials = .ieee) {x : FloatV f} (hx : x.isFinite = true) :
    sub negInf x = negInf := by
  have hxn := (isFinite_iff.mp hx)
  unfold sub Format.sub Format.add
  have h1 : f.isNaN (negInf : FloatV f).pat = false := negInf_isNaN hs
  have h2 : f.isInf (negInf : FloatV f).pat = true := negInf_isInf hs
  rw [h1, h2, Format.isNaN_neg, Format.isInf_neg]
  have h3 : f.isNaN x.pat = false := hxn.1
  have h4 : f.isInf x.pat = false := hxn.2
  simp only [h3, h4, Bool.or_false, Bool.false_eq_true, ite_false, Bool.false_and, ite_true]
  unfold Format.infPat; rw [hs]
  have := negInf_sign (f := f); unfold sign at this; rw [this]
  rfl

/-- **`x − (−∞) = +∞`** for finite `x`. -/
theorem sub_negInf_right (hs : f.specials = .ieee) {x : FloatV f} (hx : x.isFinite = true) :
    sub x negInf = posInf := by
  have hxn := (isFinite_iff.mp hx)
  unfold sub Format.sub Format.add
  have h1 : f.isNaN (negInf : FloatV f).pat = false := negInf_isNaN hs
  have h2 : f.isInf (negInf : FloatV f).pat = true := negInf_isInf hs
  have h3 : f.isNaN x.pat = false := hxn.1
  have h4 : f.isInf x.pat = false := hxn.2
  rw [Format.isNaN_neg, Format.isInf_neg, h1, h2, h3, h4]
  simp only [Bool.or_false, Bool.false_eq_true, ite_false, ite_true]
  unfold Format.infPat; rw [hs, Format.sign_neg]
  have := negInf_sign (f := f); unfold sign at this; rw [this]
  rfl

/-- **`−∞ − (−∞)` is NaN.** -/
theorem sub_negInf_negInf (hs : f.specials = .ieee) (hf : f.Regular) :
    (sub negInf negInf : FloatV f).isNaN = true := by
  have h1 : f.isNaN (negInf : FloatV f).pat = false := negInf_isNaN hs
  have h2 : f.isInf (negInf : FloatV f).pat = true := negInf_isInf hs
  have hsg : f.sign (negInf : FloatV f).pat = true := negInf_sign
  have hadd : f.add .rn (negInf : FloatV f).pat (f.neg (negInf : FloatV f).pat) = f.nanMag := by
    unfold Format.add
    rw [Format.isNaN_neg, Format.isInf_neg, h1, h2, Format.sign_neg, hsg]
    simp
  have hlt : f.nanMag < 2 ^ f.width := by
    have h3 := Format.signBit_lt (f := f)
    have h4 := Nat.one_le_two_pow (n := f.e + f.m)
    unfold Format.signBit at h3
    show 2 ^ (f.e + f.m) - 1 < 2 ^ f.width
    omega
  unfold sub Format.sub isNaN
  rw [hadd, pat_ofPat hlt]
  exact Format.isNaN_nanMag hf (by rw [hs]; decide)

end FloatV

section MaxApprox

variable {α : Type*} [AddCommGroup α] [LinearOrder α] [IsOrderedAddMonoid α]

/-- **The maximum of two approximations approximates the maximum**: from `|a − A| ≤ δ` and
`|b − B| ≤ δ`, `|max a b − max A B| ≤ δ`. -/
theorem abs_max_sub_max_le_of {a b A B δ : α} (ha : |a - A| ≤ δ) (hb : |b - B| ≤ δ) :
    |max a b - max A B| ≤ δ :=
  (abs_max_sub_max_le_max a b A B).trans (max_le ha hb)

/-- The same for the minimum: `|min a b − min A B| ≤ δ`. -/
theorem abs_min_sub_min_le_of {a b A B δ : α} (ha : |a - A| ≤ δ) (hb : |b - B| ≤ δ) :
    |min a b - min A B| ≤ δ :=
  (abs_min_sub_min_le_max a b A B).trans (max_le ha hb)

end MaxApprox

namespace FloatV

variable {f : Format}

/-- **The reference `max` of two approximations** (operands not NaN): within `δ` of `max A B`. -/
theorem max_approx {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) {A B δ : ℝ}
    (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) : |(max x y).toReal - Max.max A B| ≤ δ := by
  have e : (max x y).toReal = Max.max x.toReal y.toReal := by
    unfold toReal; rw [toRat_max hx hy, Rat.cast_max]
  rw [e]; exact abs_max_sub_max_le_of ha hb

/-- **The reference `min` of two approximations** (operands not NaN): within `δ` of `min A B`. -/
theorem min_approx {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) {A B δ : ℝ}
    (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) : |(min x y).toReal - Min.min A B| ≤ δ := by
  have e : (min x y).toReal = Min.min x.toReal y.toReal := by
    unfold toReal; rw [toRat_min hx hy, Rat.cast_min]
  rw [e]; exact abs_min_sub_min_le_of ha hb

/-- **What `tl.maximum` gives has the larger value** (operands not NaN). -/
theorem MaxRel.toRat_eq_max {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toRat = Max.max x.toRat y.toRat := by
  have hmem := h.mem hx hy
  have hr := isNaN_of_mem hx hy hmem
  obtain ⟨h1, h2⟩ := h.le hx hy
  have a := (le_iff hx hr).mp h1
  have b := (le_iff hy hr).mp h2
  rcases hmem with rfl | rfl
  · exact (max_eq_left b).symm
  · exact (max_eq_right a).symm

/-- **What `tl.minimum` gives has the smaller value** (operands not NaN). -/
theorem MaxRel.toRat_eq_min {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toRat = Min.min x.toRat y.toRat := by
  have hmem := h.mem hx hy
  have hr := isNaN_of_mem hx hy hmem
  obtain ⟨h1, h2⟩ := h.ge hx hy
  have a := (le_iff hr hx).mp h1
  have b := (le_iff hr hy).mp h2
  rcases hmem with rfl | rfl
  · exact (min_eq_left b).symm
  · exact (min_eq_right a).symm

/-- `MaxRel.toRat_eq_max` in `ℝ`. -/
theorem MaxRel.toReal_eq_max {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toReal = Max.max x.toReal y.toReal := by
  unfold toReal; rw [h.toRat_eq_max hx hy, Rat.cast_max]

/-- `MaxRel.toRat_eq_min` in `ℝ`. -/
theorem MaxRel.toReal_eq_min {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toReal = Min.min x.toReal y.toReal := by
  unfold toReal; rw [h.toRat_eq_min hx hy, Rat.cast_min]

/-- **`tl.maximum` of two approximations**: from `|x − A| ≤ δ`, `|y − B| ≤ δ` (operands not NaN),
any result is within `δ` of `max A B`. -/
theorem MaxRel.max_approx {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℝ} (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) :
    |r.toReal - Max.max A B| ≤ δ := by
  rw [h.toReal_eq_max hx hy]; exact abs_max_sub_max_le_of ha hb

/-- `MaxRel.max_approx` in `ℚ`. -/
theorem MaxRel.max_approx_rat {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℚ} (ha : |x.toRat - A| ≤ δ) (hb : |y.toRat - B| ≤ δ) :
    |r.toRat - Max.max A B| ≤ δ := by
  rw [h.toRat_eq_max hx hy]; exact abs_max_sub_max_le_of ha hb

/-- **`tl.minimum` of two approximations**: any result is within `δ` of `min A B`. -/
theorem MaxRel.min_approx {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℝ} (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) :
    |r.toReal - Min.min A B| ≤ δ := by
  rw [h.toReal_eq_min hx hy]; exact abs_min_sub_min_le_of ha hb

/-- `MaxRel.min_approx` in `ℚ`. -/
theorem MaxRel.min_approx_rat {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℚ} (ha : |x.toRat - A| ≤ δ) (hb : |y.toRat - B| ≤ δ) :
    |r.toRat - Min.min A B| ≤ δ := by
  rw [h.toRat_eq_min hx hy]; exact abs_min_sub_min_le_of ha hb

end FloatV

/-- **A `tl.max` reduction (any tree) by value**: the result is one of the leaves, and no leaf has
a larger real value (leaves not NaN). -/
theorem Tree.eval_maxRel_toReal {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel true) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false) :
    (∃ i ∈ t.leaves, r = x i) ∧ ∀ i ∈ t.leaves, (x i).toReal ≤ r.toReal := by
  obtain ⟨hmem, hle⟩ := Tree.eval_maxRel h hx
  obtain ⟨i0, hi0, rfl⟩ := hmem
  refine ⟨⟨i0, hi0, rfl⟩, fun i hi => ?_⟩
  have := hle i hi
  simp only [ite_true] at this
  unfold FloatV.toReal
  exact_mod_cast (FloatV.le_iff (hx i hi) (hx i0 hi0)).mp this

/-- **A `tl.min` reduction (any tree) by value**: one of the leaves, and no leaf is smaller. -/
theorem Tree.eval_minRel_toReal {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel false) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false) :
    (∃ i ∈ t.leaves, r = x i) ∧ ∀ i ∈ t.leaves, r.toReal ≤ (x i).toReal := by
  obtain ⟨hmem, hle⟩ := Tree.eval_maxRel h hx
  obtain ⟨i0, hi0, rfl⟩ := hmem
  refine ⟨⟨i0, hi0, rfl⟩, fun i hi => ?_⟩
  have := hle i hi
  simp only [Bool.false_eq_true, ite_false] at this
  unfold FloatV.toReal
  exact_mod_cast (FloatV.le_iff (hx i0 hi0) (hx i hi)).mp this

/-- **A `tl.max` reduction of approximations approximates the maximum.** If every leaf `x i` is
within `δ` of `A i`, and `M` is the largest `A i` over the leaves (at least each, equal to one),
the result is within `δ` of `M`. -/
theorem Tree.eval_maxRel_approx {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel true) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false)
    {A : Nat → ℝ} {M δ : ℝ} (hA : ∀ i ∈ t.leaves, |(x i).toReal - A i| ≤ δ)
    (hM : ∀ i ∈ t.leaves, A i ≤ M) (hM' : ∃ i ∈ t.leaves, A i = M) :
    |r.toReal - M| ≤ δ := by
  obtain ⟨⟨i0, hi0, rfl⟩, hle⟩ := Tree.eval_maxRel_toReal h hx
  obtain ⟨j, hj, rfl⟩ := hM'
  rw [abs_le]
  have h0 := abs_le.mp (hA i0 hi0)
  have h1 := abs_le.mp (hA j hj)
  have h2 := hM i0 hi0
  have h3 := hle j hj
  constructor <;> linarith [h0.1, h0.2, h1.1, h1.2]

/-- **A `tl.min` reduction of approximations approximates the minimum**: as
`Tree.eval_maxRel_approx`, with `M` the smallest `A i`. -/
theorem Tree.eval_minRel_approx {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel false) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false)
    {A : Nat → ℝ} {M δ : ℝ} (hA : ∀ i ∈ t.leaves, |(x i).toReal - A i| ≤ δ)
    (hM : ∀ i ∈ t.leaves, M ≤ A i) (hM' : ∃ i ∈ t.leaves, A i = M) :
    |r.toReal - M| ≤ δ := by
  obtain ⟨⟨i0, hi0, rfl⟩, hle⟩ := Tree.eval_minRel_toReal h hx
  obtain ⟨j, hj, rfl⟩ := hM'
  rw [abs_le]
  have h0 := abs_le.mp (hA i0 hi0)
  have h1 := abs_le.mp (hA j hj)
  have h2 := hM i0 hi0
  have h3 := hle j hj
  constructor <;> linarith [h0.1, h0.2, h1.1, h1.2]

/-- **`Tree.eval_maxRel_approx` for a reduction of `n` elements** (`t.Covers n`): the result is
within `δ` of `(Finset.range n).sup' hn A`, the largest `A i` for `i < n`. -/
theorem Tree.eval_maxRel_approx_sup' {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    {n : Nat} (hc : t.Covers n) (hn : (Finset.range n).Nonempty)
    (h : t.Eval (FloatV.MaxRel true) x r) (hx : ∀ i < n, (x i).isNaN = false)
    {A : Nat → ℝ} {δ : ℝ} (hA : ∀ i < n, |(x i).toReal - A i| ≤ δ) :
    |r.toReal - (Finset.range n).sup' hn A| ≤ δ := by
  have hmem : ∀ i, i ∈ t.leaves ↔ i < n := fun i => by
    rw [hc.mem_iff, List.mem_range]
  refine Tree.eval_maxRel_approx h (fun i hi => hx i ((hmem i).mp hi))
    (fun i hi => hA i ((hmem i).mp hi))
    (fun i hi => Finset.le_sup' A (Finset.mem_range.mpr ((hmem i).mp hi))) ?_
  obtain ⟨j, hj, e⟩ := Finset.exists_mem_eq_sup' hn A
  exact ⟨j, (hmem j).mpr (Finset.mem_range.mp hj), e.symm⟩

/-- **`Tree.eval_minRel_approx` for a reduction of `n` elements** (`t.Covers n`): the result is
within `δ` of `(Finset.range n).inf' hn A`, the smallest `A i` for `i < n`. -/
theorem Tree.eval_minRel_approx_inf' {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    {n : Nat} (hc : t.Covers n) (hn : (Finset.range n).Nonempty)
    (h : t.Eval (FloatV.MaxRel false) x r) (hx : ∀ i < n, (x i).isNaN = false)
    {A : Nat → ℝ} {δ : ℝ} (hA : ∀ i < n, |(x i).toReal - A i| ≤ δ) :
    |r.toReal - (Finset.range n).inf' hn A| ≤ δ := by
  have hmem : ∀ i, i ∈ t.leaves ↔ i < n := fun i => by
    rw [hc.mem_iff, List.mem_range]
  refine Tree.eval_minRel_approx h (fun i hi => hx i ((hmem i).mp hi))
    (fun i hi => hA i ((hmem i).mp hi))
    (fun i hi => Finset.inf'_le A (Finset.mem_range.mpr ((hmem i).mp hi))) ?_
  obtain ⟨j, hj, e⟩ := Finset.exists_mem_eq_inf' hn A
  exact ⟨j, (hmem j).mpr (Finset.mem_range.mp hj), e.symm⟩

end TVA
