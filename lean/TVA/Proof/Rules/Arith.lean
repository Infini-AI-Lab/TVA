import TVA.Proof.Safe
import TVA.Proof.Int.Arith

/-!
# Rules: float and integer arithmetic

The proof rules for `TVA.Triton.Arith`. Each rule is `Safe … (op >>= k)`; its last premise is the
rest `k`, for every result the operation may give.

* **Exact float operations, up to NaN payload** (`tl.upToNaN`): `safe_upToNaN`, and `safe_exact`
  when no NaN arises; `safe_add`, `safe_sub`, `safe_mul`, `safe_fma`, `safe_div_rn`,
  `safe_sqrt_rn`, `safe_neg`, `safe_abs`.
* **Operations a product may fuse into**: `safe_mul_add` (`a * b + c`), `safe_addc`, `safe_subc`
  (`AddRel`).
* **Platform operations** (`Platform.div`, `Platform.math`): `safe_div`, `safe_fdiv`, `safe_math`.
  A result must exist (the device facts say so); the rest runs for every result the platform
  allows.
* `safe_maximum`, `safe_minimum` (`FloatV.MaxRel`).
* **Integers**: `safe_idiv`, `safe_irem`, `safe_shl`, `safe_ashr`, `safe_lshr`: no lane may divide
  by zero, overflow or over-shift. For scalars with `x ≥ 0`, `y > 0`: `safe_idiv1`, `safe_irem1`
  (with `IntV.div?_of_nonneg`, `IntV.rem?_of_nonneg`).

How this fits: model → `Safe` per program (the rules of `TVA.Proof.Rules`) → `correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

section Float
variable {s : Shape} {f : Format}

/-- **An exact float operation** (`tl.upToNaN v`): the rest must be safe for every tensor equal to
the exact result `v` up to NaN payload. -/
theorem safe_upToNaN {v : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (v.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.upToNaN v >>= k) := by
  simp only [tl.upToNaN, bind_eq, Prog.bind]
  exact .pick ⟨v, fun _ => Or.inl rfl⟩ hk

/-- An exact float operation whose exact result `v` has no NaN gives exactly `v`. -/
theorem safe_exact {v : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hnan : ∀ i, (v.get i).isNaN = false) (hk : Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.upToNaN v >>= k) :=
  safe_upToNaN fun r hr => by
    have : r = v := by
      ext i
      rcases hr i with h | ⟨h, -⟩
      · exact h.symm
      · rw [hnan i] at h; cases h
    subst this; exact hk

/-- `x + y` (no product feeds it): the IEEE sum, up to NaN payload. -/
theorem safe_add {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.add (x.get i) (y.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.add x y >>= k) := safe_upToNaN hk

/-- `x - y` (no product feeds it): the IEEE difference, up to NaN payload. -/
theorem safe_sub {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.sub (x.get i) (y.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sub x y >>= k) := safe_upToNaN hk

/-- `x * y`: the IEEE product, up to NaN payload. -/
theorem safe_mul {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.mul (x.get i) (y.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.mul x y >>= k) := safe_upToNaN hk

/-- `tl.fma(x, y, z)`: one rounding of `x * y + z`, up to NaN payload. -/
theorem safe_fma {x y z : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.fma (x.get i) (y.get i) (z.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.fma x y z >>= k) := safe_upToNaN hk

/-- `tl.div_rn(x, y)`: the IEEE quotient, up to NaN payload. -/
theorem safe_div_rn {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.divRn (x.get i) (y.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.div_rn x y >>= k) := safe_upToNaN hk

/-- `tl.sqrt_rn(x)`: the IEEE square root, up to NaN payload. -/
theorem safe_sqrt_rn {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.sqrtRn (x.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sqrt_rn x >>= k) := safe_upToNaN hk

/-- `-x`: exact, up to NaN payload. -/
theorem safe_neg {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.neg (x.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.neg x >>= k) := safe_upToNaN hk

/-- `tl.abs(x)`: exact, up to NaN payload. -/
theorem safe_abs {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.abs (x.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.abs x >>= k) := safe_upToNaN hk

/-- `a * b + c`: the two IEEE operations, or (with fusion) one `fma`, lane by lane. -/
theorem safe_mul_add {a b c : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i,
      FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
      (E.fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i))) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.mul_add a b c >>= k) := by
  simp only [tl.mul_add, bind_eq, Prog.bind]
  exact .env (.pick ⟨⟨fun i => FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)⟩,
    fun _ => Or.inl (Or.inl rfl)⟩ hk)

/-- `x + y` that a product may feed: any result `AddRel` allows. -/
theorem safe_addc {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, AddRel E.fusion (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.addc x y >>= k) := by
  simp only [tl.addc, bind_eq, Prog.bind]
  exact .env (.pick ⟨Tensor.map2 FloatV.add x y, fun _ => AddRel.ieee _ _ _⟩ hk)

/-- `x - y` that a product may feed: `x + (-y)` under `AddRel`. -/
theorem safe_subc {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, AddRel E.fusion (x.get i) (FloatV.neg (y.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.subc x y >>= k) := safe_addc hk

/-- `x / y` (`tl.div`): a platform quotient exists for every lane (`hex`, from the device facts);
the rest must be safe for every quotient the platform allows. -/
theorem safe_div {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hex : ∀ i, ∃ r, E.P.div f (x.get i) (y.get i) r)
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, E.P.div f (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.div x y >>= k) := by
  simp only [tl.div, bind_eq, Prog.bind]
  exact .plat ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk

/-- `tl.fdiv(x, y, ieee_rounding)`: `safe_div_rn` with `ieee_rounding`, `safe_div` without. -/
theorem safe_fdiv {x y : Tensor s (FloatV f)} {ieee : Bool} {k : Tensor s (FloatV f) → Prog α}
    (hex : ieee = false → ∀ i, ∃ r, E.P.div f (x.get i) (y.get i) r)
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, if ieee then FloatV.NaNEq (FloatV.divRn (x.get i) (y.get i)) (r.get i)
        else E.P.div f (x.get i) (y.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.fdiv x y ieee >>= k) := by
  cases ieee with
  | true => exact safe_div_rn fun r hr => hk r fun i => by simpa using hr i
  | false => exact safe_div (hex rfl) fun r hr => hk r fun i => by simpa using hr i

/-- An approximate function (`tl.exp`, `tl.log`, `tl.sqrt`, …, all `tl.math fn`): a platform
result exists for every lane (`hex`, from the device facts); the rest must be safe for every
result the platform allows. -/
theorem safe_math {fn : MathFn} {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hex : ∀ i, ∃ r, E.P.math fn f (x.get i) r)
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, E.P.math fn f (x.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.math fn x >>= k) := by
  simp only [tl.math, bind_eq, Prog.bind]
  exact .plat ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk

/-- `tl.maximum(x, y)` on floats: any result `MaxRel true` allows. -/
theorem safe_maximum {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.MaxRel true (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.maximum x y >>= k) := by
  simp only [tl.maximum, bind_eq, Prog.bind]
  exact .pick ⟨Tensor.map2 FloatV.max x y, fun _ => Or.inl (Or.inl rfl)⟩ hk

/-- `tl.minimum(x, y)` on floats: any result `MaxRel false` allows. -/
theorem safe_minimum {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.MaxRel false (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.minimum x y >>= k) := by
  simp only [tl.minimum, bind_eq, Prog.bind]
  exact .pick ⟨Tensor.map2 FloatV.min x y, fun _ => Or.inl (Or.inl rfl)⟩ hk

end Float

/-! ## Integers -/

section Int
variable {s : Shape} {sg : Bool} {w : Nat}

/-- `x // y`: no lane divides by zero or overflows; the result is `div?` lane by lane. -/
theorem safe_idiv {x y : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, ∃ q, (x.get i).div? (y.get i) = some q)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).div? (y.get i) = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.idiv x y >>= k) := by
  simp only [tl.idiv, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => Classical.choose (hok i)⟩, fun i => Classical.choose_spec (hok i)⟩ hk

/-- `x % y`: no lane divides by zero or overflows; the result is `rem?` lane by lane. -/
theorem safe_irem {x y : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, ∃ q, (x.get i).rem? (y.get i) = some q)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).rem? (y.get i) = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.irem x y >>= k) := by
  simp only [tl.irem, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => Classical.choose (hok i)⟩, fun i => Classical.choose_spec (hok i)⟩ hk

/-- `x << n`: every shift amount is below the width. -/
theorem safe_shl {x n : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, (n.get i).bits.toNat < w)
    (hk : ∀ r : Tensor s (IntV sg w),
      (∀ i, (x.get i).shl? (n.get i).bits.toNat = some (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.shl x n >>= k) := by
  simp only [tl.shl, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => ⟨(x.get i).bits <<< (n.get i).bits.toNat⟩⟩,
    fun i => by simp [IntV.shl?, hok i]⟩ hk

/-- `x >> n`, arithmetic: every shift amount is below the width. -/
theorem safe_ashr {x n : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, (n.get i).bits.toNat < w)
    (hk : ∀ r : Tensor s (IntV sg w),
      (∀ i, (x.get i).ashr? (n.get i).bits.toNat = some (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.ashr x n >>= k) := by
  simp only [tl.ashr, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => ⟨(x.get i).bits.sshiftRight (n.get i).bits.toNat⟩⟩,
    fun i => by simp [IntV.ashr?, hok i]⟩ hk

/-- `x >> n`, logical: every shift amount is below the width. -/
theorem safe_lshr {x n : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, (n.get i).bits.toNat < w)
    (hk : ∀ r : Tensor s (IntV sg w),
      (∀ i, (x.get i).lshr? (n.get i).bits.toNat = some (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.lshr x n >>= k) := by
  simp only [tl.lshr, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => ⟨(x.get i).bits >>> (n.get i).bits.toNat⟩⟩,
    fun i => by simp [IntV.lshr?, hok i]⟩ hk

end Int

/-- **`x // y` of scalars** (rank-0 tensors) with `x ≥ 0`, `y > 0`: the integer quotient. -/
theorem safe_idiv1 {w : Nat} {x y : IntV true w} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = x.toInt / y.toInt →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.idiv (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  obtain ⟨q, hq, hqv⟩ := IntV.div?_of_nonneg hx hy
  refine safe_idiv (fun _ => ⟨q, hq⟩) fun r hr => hk r ?_
  have := hr ()
  simp only [Tensor.scalar] at this
  rw [hq] at this
  have e : r.item = q := (Option.some.inj this).symm
  rw [e, hqv]

/-- **`x % y` of scalars** with `x ≥ 0`, `y > 0`: the integer remainder. -/
theorem safe_irem1 {w : Nat} {x y : IntV true w} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = x.toInt % y.toInt →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.irem (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  obtain ⟨q, hq, hqv⟩ := IntV.rem?_of_nonneg hx hy
  refine safe_irem (fun _ => ⟨q, hq⟩) fun r hr => hk r ?_
  have := hr ()
  simp only [Tensor.scalar] at this
  rw [hq] at this
  have e : r.item = q := (Option.some.inj this).symm
  rw [e, hqv]

end TVA
