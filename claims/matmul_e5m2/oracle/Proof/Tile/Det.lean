import Proof.Basic

/-!
# Deterministic forms of the integer rules

`tl.idiv` and `tl.irem` are `pick`s: the library's rules leave the result to a hypothesis on it.
When the dividend is nonnegative and the divisor positive the result is known: `IntV.ofInt` of
the integer quotient (or remainder).  The key fact is `IntV.ofInt_toInt`: two integer values
with the same `toInt` are equal, so a result whose value is known *is* `IntV.ofInt` of it.

The file also holds `IntV.toInt_min`/`toInt_max` (`tl.iminimum`/`tl.imaximum`), elementary facts
about rank-0 tensors and `bc`, and the `val?` of finite `float8e5` values.
-/

namespace TVA

/-- A rank-0 tensor is the scalar of its item. -/
theorem tensor0_eq {β : Type} (x : Tensor [] β) : Tensor.scalar x.item = x := by
  apply Tensor.ext
  funext i
  cases i
  rfl

/-- The value of a broadcast tensor. -/
theorem bc_get {s : Shape} {β : Type} (a : β) (i : Idx s) : (bc a : Tensor s β).get i = a := rfl

/-- The rational value of a finite `float8e5` element. -/
theorem val?_f8e5 {x : F8E5} {v : ℚ} : DType.val? .f8e5 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  show (if x.isFinite = true then some x.toRat else none) = some v ↔ _
  split_ifs with h
  · simp [h]
  · simp [h]

/-- A finite `float8e5` value is at most `e5m2`'s largest in magnitude. -/
theorem f8e5_abs_toRat {x : F8E5} (hx : x.isFinite = true) : |x.toRat| ≤ 57344 := by
  have h := FloatV.abs_toRat_le_maxVal (f := Format.e5m2)
    Format.e5m2_regular hx
  rwa [Format.e5m2_maxVal] at h

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- `x // y` of two `int32` scalars, `x ≥ 0`, `y > 0`: the result is `IntV.ofInt (x / y)`. -/
theorem safe_idiv1' {x y : I32} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] I32 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt (x.toInt / y.toInt))))) :
    Safe E R W Q m (tl.idiv (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  refine safe_idiv1 (w := 32) hx hy fun r hr => ?_
  have hval : r = Tensor.scalar (IntV.ofInt (x.toInt / y.toInt)) := by
    rw [← tensor0_eq r]
    congr 1
    rw [← IntV.ofInt_toInt r.item, hr]
  rw [hval]
  exact hk

/-- `x % y` of two `int32` scalars, `x ≥ 0`, `y > 0`: the result is `IntV.ofInt (x % y)`. -/
theorem safe_irem1' {x y : I32} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] I32 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt (x.toInt % y.toInt))))) :
    Safe E R W Q m (tl.irem (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  refine safe_irem1 (w := 32) hx hy fun r hr => ?_
  have hval : r = Tensor.scalar (IntV.ofInt (x.toInt % y.toInt)) := by
    rw [← tensor0_eq r]
    congr 1
    rw [← IntV.ofInt_toInt r.item, hr]
  rw [hval]
  exact hk

/-- `x % y` of two `int32` tensors of one shape, with a nonnegative dividend and a positive
divisor in every lane: the result is `IntV.ofInt (x % y)` lane by lane. -/
theorem safe_irem' {s : Shape} {x y : Tensor s I32}
    (hok : ∀ i, 0 ≤ (x.get i).toInt ∧ 0 < (y.get i).toInt)
    {k : Tensor s I32 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.map2 (fun a b : I32 => IntV.ofInt (a.toInt % b.toInt)) x y))) :
    Safe E R W Q m (tl.irem x y >>= k) := by
  refine safe_irem (fun i => ?_) fun r hr => ?_
  · obtain ⟨q, hq, -⟩ := IntV.rem?_of_nonneg (hok i).1 (hok i).2
    exact ⟨q, hq⟩
  · have hre : r = Tensor.map2 (fun a b : I32 => IntV.ofInt (a.toInt % b.toInt)) x y := by
      apply Tensor.ext
      funext i
      have hri : r.get i = IntV.ofInt ((x.get i).toInt % (y.get i).toInt) := by
        obtain ⟨q, hq, hqv⟩ := IntV.rem?_of_nonneg (hok i).1 (hok i).2
        have h2 := hr i
        rw [hq] at h2
        have h3 : r.get i = q := (Option.some.inj h2).symm
        rw [h3, ← hqv, IntV.ofInt_toInt]
      rw [hri]
      rfl
    rw [hre]
    exact hk

end TVA
