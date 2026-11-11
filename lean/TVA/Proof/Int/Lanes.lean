import TVA.Proof.Int.Arith
import TVA.Triton

/-!
# Lanes of integer vectors

An offset vector such as `pid * BLOCK + tl.arange(0, BLOCK)` is a rank-1 tensor of `IntV`. A
proof follows the value of each lane `i` (`x.get1 i`, which is `x.get (i, ())`):

* `Tensor.get1_add`, `get1_sub`, `get1_mul`, `get1_neg`, `get1_sadd`, `get1_adds`, `get1_smul`,
  `get1_muls`, `get1_lt`, `get1_ltS`, `get1_le`, `get1_eq`, `get1_castInt` (`simp`): the
  `get1` forms of `Tensor`'s `get_*` lemmas, so that a lane of an expression is the expression of
  the lanes; `TVA.Proof.Int.Arith` then gives its value.
* `I32.toInt_arange`, `I32.toInt_tl_arange`, `I32.toInt_sadd_arange`: the lanes of `tl.arange` and
  of `start + tl.arange(0, n)`.
* `idiv_ok`, `irem_ok`: every lane of `x // M`, `x % M` is defined when the lanes of `x` are
  natural and `M > 0` (the premise of the rule for `tl.idiv`, `tl.irem`); `idiv_toInt`,
  `irem_toInt`: the lane values of a result (`(r0 + tl.arange(0, BM)) % M` has lanes
  `(r0 + i) % M`).
-/
namespace TVA

namespace Tensor

variable {n : Nat} {sg : Bool} {w : Nat}

/-- Lane `i` of `x + y` is the sum of the lanes `i`. -/
@[simp] theorem get1_add (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x + y).get1 i = x.get1 i + y.get1 i := rfl

/-- Lane `i` of `x - y` is the difference of the lanes `i`. -/
@[simp] theorem get1_sub (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x - y).get1 i = x.get1 i - y.get1 i := rfl

/-- Lane `i` of `x * y` is the product of the lanes `i`. -/
@[simp] theorem get1_mul (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x * y).get1 i = x.get1 i * y.get1 i := rfl

/-- Lane `i` of `-x` is `-(x.get1 i)`. -/
@[simp] theorem get1_neg (x : Tensor [n] (IntV sg w)) (i : Fin n) : (-x).get1 i = -x.get1 i := rfl

/-- Lane `i` of `a + x`, for a scalar `a` (`block_start + offs`), is `a + x.get1 i`. -/
@[simp] theorem get1_sadd (a : IntV sg w) (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (a + x).get1 i = a + x.get1 i := rfl

/-- Lane `i` of `x + a`, for a scalar `a`, is `x.get1 i + a`. -/
@[simp] theorem get1_adds (x : Tensor [n] (IntV sg w)) (a : IntV sg w) (i : Fin n) :
    (x + a).get1 i = x.get1 i + a := rfl

/-- Lane `i` of `a * x`, for a scalar `a`, is `a * x.get1 i`. -/
@[simp] theorem get1_smul (a : IntV sg w) (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (a * x).get1 i = a * x.get1 i := rfl

/-- Lane `i` of `x * a`, for a scalar `a` (`offs * stride`), is `x.get1 i * a`. -/
@[simp] theorem get1_muls (x : Tensor [n] (IntV sg w)) (a : IntV sg w) (i : Fin n) :
    (x * a).get1 i = x.get1 i * a := rfl

/-- Lane `i` of `x < y` is `(x.get1 i).lt (y.get1 i)`. -/
@[simp] theorem get1_lt (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.lt y).get1 i = (x.get1 i).lt (y.get1 i) := rfl

/-- Lane `i` of `x < a`, for a scalar `a` (the mask `offs < n`), is `(x.get1 i).lt a`. -/
@[simp] theorem get1_ltS (x : Tensor [n] (IntV sg w)) (a : IntV sg w) (i : Fin n) :
    (x.ltS a).get1 i = (x.get1 i).lt a := rfl

/-- Lane `i` of `x <= y` is `(x.get1 i).le (y.get1 i)`. -/
@[simp] theorem get1_le (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.le y).get1 i = (x.get1 i).le (y.get1 i) := rfl

/-- Lane `i` of `x == y` is `(x.get1 i).eq (y.get1 i)`. -/
@[simp] theorem get1_eq (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.eq y).get1 i = (x.get1 i).eq (y.get1 i) := rfl

/-- Lane `i` of `x.castInt` (`x.to(tl.int64)` of an integer vector) is the cast of lane `i`. -/
@[simp] theorem get1_castInt {sg' : Bool} {w' : Nat} (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.castInt : Tensor [n] (IntV sg' w')).get1 i = IntV.cast (x.get1 i) := rfl

end Tensor

namespace IntV

/-- Lane `i` of `Tensor.arange start n` (`tl.arange(start, start + n)`, `int32`) has value
`start + i` when the whole range is in `int32`. -/
theorem I32.toInt_arange {start : Int} {n : Nat} (hlo : -(2 ^ 31 : Int) ≤ start)
    (hhi : start + n ≤ 2 ^ 31) (i : Fin n) :
    ((Tensor.arange start n).get1 i).toInt = start + i := by
  rw [Tensor.get1_arange]
  have := i.isLt
  exact I32.toInt_ofInt ⟨by omega, by omega⟩

/-- Lane `i` of `tl.arange(0, n)` has value `i` when `n ≤ 2^31`. -/
theorem I32.toInt_tl_arange {n : Nat} (hn : n ≤ 2 ^ 31) (i : Fin (n - 0)) :
    ((tl.arange 0 n).get1 i).toInt = (i : Nat) := by
  have := I32.toInt_arange (start := ((0 : Nat) : Int)) (n := n - 0) (by norm_num)
    (by push_cast; omega) i
  exact this.trans (by simp)

/-- Lane `i` of `c + tl.arange(0, n)`, for an `int32` scalar `c` of natural value `r0`, has value
`r0 + i` when `r0 + n ≤ 2^31` (`pid * BLOCK + tl.arange(0, BLOCK)`). -/
theorem I32.toInt_sadd_arange {c : I32} {r0 n : Nat} (hc : c.toInt = r0) (h : r0 + n ≤ 2 ^ 31)
    (i : Fin (n - 0)) : ((c + tl.arange 0 n).get1 i).toInt = ((r0 + i : Nat) : Int) := by
  have hi := i.isLt
  exact IntV.add_nat hc (I32.toInt_tl_arange (by omega) i)
    (by simp only [Nat.sub_zero] at hi; omega)

end IntV

variable {n M w : Nat} {f : Fin n → Nat}

/-- Every lane of `x // y` is defined when the lanes of `x` have natural values `f i` and every
lane of `y` the value `M > 0` (the premise of the rule for `tl.idiv`). -/
theorem idiv_ok {x y : Tensor [n] (IntV true w)}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M) :
    ∀ i, ∃ q, (x.get i).div? (y.get i) = some q := by
  rintro ⟨i, ⟨⟩⟩
  obtain ⟨q, hq, -⟩ := IntV.div?_nat (hx i) (hy i) hM
  exact ⟨q, hq⟩

/-- Lane `i` of a result `r` of `tl.idiv x y` (as in `idiv_ok`) has value `f i / M`. -/
theorem idiv_toInt {x y r : Tensor [n] (IntV true w)}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hr : ∀ i, (x.get i).div? (y.get i) = some (r.get i)) (i : Fin n) :
    (r.get1 i).toInt = ((f i / M : Nat) : Int) :=
  IntV.toInt_of_div?_nat (hr (i, ())) (hx i) (hy i) hM

/-- Every lane of `x % y` is defined when the lanes of `x` have natural values `f i` and every
lane of `y` the value `M > 0` (the premise of the rule for `tl.irem`). -/
theorem irem_ok {x y : Tensor [n] (IntV true w)}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M) :
    ∀ i, ∃ q, (x.get i).rem? (y.get i) = some q := by
  rintro ⟨i, ⟨⟩⟩
  obtain ⟨q, hq, -⟩ := IntV.rem?_nat (hx i) (hy i) hM
  exact ⟨q, hq⟩

/-- Lane `i` of a result `r` of `tl.irem x y` (as in `irem_ok`) has value `f i % M`, below `M`:
rows wrapped by `% M` stay in the matrix. -/
theorem irem_toInt {x y r : Tensor [n] (IntV true w)}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hr : ∀ i, (x.get i).rem? (y.get i) = some (r.get i)) (i : Fin n) :
    (r.get1 i).toInt = ((f i % M : Nat) : Int) :=
  IntV.toInt_of_rem?_nat (hr (i, ())) (hx i) (hy i) hM

end TVA
