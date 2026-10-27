import TVA.Lemmas.Ops
import TVA.Lemmas.Int
import TVA.Lemmas.Access

/-!
# Scalar loads and integer division

* `safe_load1`, `safe_load1_buf`: `tl.load(p)` of one element.
* `IntV.div?_of_nonneg`, `IntV.rem?_of_nonneg`: signed `//` and `%` of a nonnegative value by
  a positive one are the integer quotient and remainder (`sdiv`/`srem` truncate, which agrees
  with floor division there).
* `safe_idiv1`, `safe_irem1`: the same for rank-0 tensors, as `tl.cdiv` and tile coordinates
  use them.
-/

namespace TVA

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.load` of one element** whose address memory can read. -/
theorem safe_load1 {d : DType} {p : Ptr d} {v0 : d.Val} (hv : m.read d p.addr = some v0)
    (hR : Mem.span d p.addr ⊆ R) {k : d.Val → Prog α} (hk : Safe E R W Q m (k v0)) :
    Safe E R W Q m (tl.load1 p >>= k) := by
  simp only [tl.load1, Prog.bind_eq, Prog.bind]
  refine .load ?_ ?_ fun v hok => ?_
  · rintro ⟨_, a, ha, hnone⟩
    cases ha
    rw [hv] at hnone; cases hnone
  · rintro x ⟨_, a, ha, hx⟩
    cases ha
    exact hR hx
  · have : v.item = v0 := by
      have := hok () p.addr rfl
      rw [hv] at this
      exact (Option.some.inj this).symm
    show Safe E R W Q m (k v.item)
    rw [this]
    exact hk

/-- **`tl.load` of element `i` of a buffer.** -/
theorem safe_load1_buf {d : DType} (b : Buf d) {i : Nat} (hi : i < b.len) (hv : b.Valid m)
    (hR : Mem.span d (b.ptr i).addr ⊆ R) {k : d.Val → Prog α}
    (hk : ∀ v, m.read d (b.ptr i).addr = some v → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load1 (b.ptr i) >>= k) := by
  obtain ⟨v, hv'⟩ := b.read_some hv hi
  exact safe_load1 hv' hR (hk v hv')

namespace IntV

variable {w : Nat}

theorem toInt_signed (a : IntV true w) : a.toInt = a.bits.toInt := by simp [toInt]

/-- `//` of a nonnegative value by a positive one: the integer quotient. -/
theorem div?_of_nonneg {a b : IntV true w} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt) :
    ∃ q, a.div? b = some q ∧ q.toInt = a.toInt / b.toInt := by
  rw [toInt_signed] at ha hb
  have hb0 : b.bits ≠ 0 := fun h => by rw [h] at hb; simp at hb
  have hb1 : b.bits ≠ -1 := fun h => by
    rw [h, show (-1 : BitVec w) = BitVec.allOnes w from BitVec.neg_one_eq_allOnes,
      BitVec.toInt_allOnes] at hb
    split_ifs at hb <;> omega
  refine ⟨⟨a.bits.sdiv b.bits⟩, ?_, ?_⟩
  · simp only [div?, hb0, hb1, and_false, ↓reduceIte]
  · rw [toInt_signed, toInt_signed, toInt_signed b]
    show (a.bits.sdiv b.bits).toInt = _
    rw [BitVec.toInt_sdiv_of_ne_or_ne _ _ (Or.inr hb1), Int.tdiv_eq_ediv_of_nonneg ha]

/-- `%` of a nonnegative value by a positive one: the integer remainder. -/
theorem rem?_of_nonneg {a b : IntV true w} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt) :
    ∃ q, a.rem? b = some q ∧ q.toInt = a.toInt % b.toInt := by
  rw [toInt_signed] at ha hb
  have hb0 : b.bits ≠ 0 := fun h => by rw [h] at hb; simp at hb
  have hb1 : b.bits ≠ -1 := fun h => by
    rw [h, show (-1 : BitVec w) = BitVec.allOnes w from BitVec.neg_one_eq_allOnes,
      BitVec.toInt_allOnes] at hb
    split_ifs at hb <;> omega
  refine ⟨⟨a.bits.srem b.bits⟩, ?_, ?_⟩
  · simp only [rem?, hb0, hb1, and_false, ↓reduceIte]
  · rw [toInt_signed, toInt_signed, toInt_signed b]
    show (a.bits.srem b.bits).toInt = _
    rw [BitVec.toInt_srem, Int.tmod_eq_emod_of_nonneg ha]

/-- `int32` arithmetic on natural values that stays below `2^31` is natural arithmetic. -/
theorem I32.add_nat {a b : I32} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (h : x + y < 2 ^ 31) :
    (a + b).toInt = ((x + y : Nat) : Int) := by
  rw [toInt_add (by norm_num) (by rw [ha, hb]; constructor <;> push_cast <;> omega), ha, hb]; push_cast; rfl

theorem I32.mul_nat {a b : I32} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (h : x * y < 2 ^ 31) :
    (a * b).toInt = ((x * y : Nat) : Int) := by
  have h' : ((x * y : Nat) : Int) < 2 ^ 31 := by exact_mod_cast h
  rw [toInt_mul (by norm_num) (by rw [ha, hb]; constructor <;> push_cast at h' ⊢ <;> nlinarith), ha, hb]
  push_cast; rfl

theorem I32.sub_nat {a b : I32} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (hxy : y ≤ x)
    (h : x < 2 ^ 31) : (a - b).toInt = ((x - y : Nat) : Int) := by
  rw [toInt_sub (by norm_num) (by rw [ha, hb]; constructor <;> push_cast <;> omega), ha, hb]; push_cast [hxy]; rfl

theorem I32.ofNat_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = (n : Int) := I32.toInt_ofNat h

end IntV

/-- **`x // y` of scalars** (rank-0 tensors), `x ≥ 0`, `y > 0`. -/
theorem safe_idiv1 {w : Nat} {x y : IntV true w} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = x.toInt / y.toInt → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.idiv (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  obtain ⟨q, hq, hqv⟩ := IntV.div?_of_nonneg hx hy
  refine safe_idiv (fun _ => ⟨q, hq⟩) fun r hr => hk r ?_
  have := hr ()
  simp only [Tensor.scalar] at this
  rw [hq] at this
  have e : r.item = q := (Option.some.inj this).symm
  rw [e, hqv]

/-- **`x % y` of scalars**, `x ≥ 0`, `y > 0`. -/
theorem safe_irem1 {w : Nat} {x y : IntV true w} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = x.toInt % y.toInt → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.irem (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  obtain ⟨q, hq, hqv⟩ := IntV.rem?_of_nonneg hx hy
  refine safe_irem (fun _ => ⟨q, hq⟩) fun r hr => hk r ?_
  have := hr ()
  simp only [Tensor.scalar] at this
  rw [hq] at this
  have e : r.item = q := (Option.some.inj this).symm
  rw [e, hqv]

end TVA
