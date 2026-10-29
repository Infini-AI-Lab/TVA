import TVA.Lemmas.FloatErr

/-!
# The float maximum and `−∞`

Facts an online softmax needs about `tl.maximum`, `tl.max` and a running maximum that starts at
`-inf`:

* comparisons by value: `FloatV.le_iff`, `lt_iff`, `eq_iff` (for operands that are not NaN),
  with `le_refl'`, `le_trans'`, and `ext_pat` (two `FloatV`s with one pattern are equal);
* the reference `max`/`min` of two non-NaN values: `FloatV.max_mem`, `min_mem` (it is one of them),
  `toRat_max`, `toRat_min` (its value is the larger or smaller value, so it is not below or above
  either), `isNaN_of_mem`; every result `tl.maximum`/`tl.minimum` may give (`MaxRel`):
  `MaxRel.mem` (one of the operands), `MaxRel.le` (not below either operand, for `tl.maximum`),
  `MaxRel.ge` (not above either, for `tl.minimum`);
* `Tree.eval_maxRel`: a `tl.max`/`tl.min` reduction (any tree) of non-NaN leaves is one of its
  leaves, and not below (above) any of them;
* `−∞` (`FloatV.negInf`) and `+∞` (`FloatV.posInf`) of an IEEE format: `−∞` is below every
  value that is not NaN (`negInf_toRat_le`), `max(−∞, x) = x` (`max_negInf_left`,
  `MaxRel.negInf_left`), `−∞ − x = −∞` and `x − (−∞) = +∞` for finite `x` (`sub_negInf_left`,
  `sub_negInf_right`), `−∞ − (−∞)` is NaN (`sub_negInf_negInf`).
-/
namespace TVA

open TVA.Math.Float

namespace FloatV

variable {f : Format}

private theorem ofPat_pat' (x : FloatV f) : ofPat x.pat = x := by
  cases x; simp [ofPat, pat]

theorem ext_pat {x y : FloatV f} (h : x.pat = y.pat) : x = y := by
  cases x; cases y; simp only [pat] at h; congr; exact BitVec.eq_of_toNat_eq h

/-! ## Comparisons by value -/

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

theorem lt_iff {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    lt x y = true ↔ x.toRat < y.toRat := by
  unfold lt; rw [Format.cmp_of_not_nan hx hy]
  unfold toRat
  split_ifs with h1 h2 <;> simp_all

theorem eq_iff {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    eq x y = true ↔ x.toRat = y.toRat := by
  unfold eq; rw [Format.cmp_of_not_nan hx hy]
  unfold toRat
  split_ifs with h1 h2 <;> simp_all
  exact ne_of_lt h1

theorem le_refl' {x : FloatV f} (hx : x.isNaN = false) : le x x = true := (le_iff hx hx).mpr le_rfl

theorem le_trans' {x y z : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false)
    (hz : z.isNaN = false) (h1 : le x y = true) (h2 : le y z = true) : le x z = true :=
  (le_iff hx hz).mpr (((le_iff hx hy).mp h1).trans ((le_iff hy hz).mp h2))

/-! ## The reference `max` and `min` -/

theorem max_mem {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    max x y = x ∨ max x y = y := by
  unfold max Format.maxNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  split_ifs <;> simp [ofPat_pat']

theorem min_mem {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    min x y = x ∨ min x y = y := by
  unfold min Format.minNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  split_ifs <;> simp [ofPat_pat']

theorem toRat_max {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    (max x y).toRat = Max.max x.toRat y.toRat := by
  unfold max Format.maxNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  unfold toRat
  split_ifs with h1 h2 h3 <;> simp only [ofPat_pat']
  · exact (max_eq_right h1.le).symm
  · exact (max_eq_left h2.le).symm
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, max_self]
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, max_self]

theorem toRat_min {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) :
    (min x y).toRat = Min.min x.toRat y.toRat := by
  unfold min Format.minNum
  have hx' : f.isNaN x.pat = false := hx
  have hy' : f.isNaN y.pat = false := hy
  simp only [hx', hy', Bool.false_eq_true, ite_false]
  unfold toRat
  split_ifs with h1 h2 h3 <;> simp only [ofPat_pat']
  · exact (min_eq_left h1.le).symm
  · exact (min_eq_right h2.le).symm
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, min_self]
  · have := le_antisymm (not_lt.mp h2) (not_lt.mp h1); rw [this, min_self]

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

/-- `−∞` of an IEEE format (`0xff800000` in `float32`). -/
def negInf : FloatV f := ofPat (f.setSign true f.infMag)

/-- `+∞` of an IEEE format (`0x7f800000` in `float32`). -/
def posInf : FloatV f := ofPat (f.setSign false f.infMag)

theorem negInf_pat : (negInf : FloatV f).pat = f.setSign true f.infMag :=
  pat_ofPat (Format.setSign_lt _ _)

theorem posInf_pat : (posInf : FloatV f).pat = f.setSign false f.infMag :=
  pat_ofPat (Format.setSign_lt _ _)

theorem negInf_isInf (hs : f.specials = .ieee) : (negInf : FloatV f).isInf = true := by
  unfold isInf; rw [negInf_pat, Format.isInf_setSign, Format.isInf_infMag hs]

theorem posInf_isInf (hs : f.specials = .ieee) : (posInf : FloatV f).isInf = true := by
  unfold isInf; rw [posInf_pat, Format.isInf_setSign, Format.isInf_infMag hs]

theorem negInf_isNaN (hs : f.specials = .ieee) : (negInf : FloatV f).isNaN = false := by
  unfold isNaN; rw [negInf_pat, Format.isNaN_setSign]
  have := Format.isInf_infMag (f := f) hs
  unfold Format.isInf at this; unfold Format.isNaN
  rw [hs] at this ⊢
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  simp [this.1, this.2]

theorem negInf_sign : (negInf : FloatV f).sign = true := by
  unfold sign; rw [negInf_pat, Format.sign_setSign]

theorem infMag_pos (hf : f.Regular) : 0 < f.infMag := by
  unfold Format.infMag
  exact Nat.mul_pos (by have := Format.two_pow_e_ge hf; omega) (by positivity)

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

end TVA
