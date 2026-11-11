import Proof.Defs
import Proof.Tile.Det

/-!
# The integer values the lanes compute

Every address the kernel uses is `ptr + int32 offset`, and every offset is built from the
program id by the scalar chain `pid_m`/`pid_n`.  This file fixes, in `Nat`, what each lane's
offset is (`om`, `on`, `aOff`, `bOff`, `cOff`), and proves that the `int32` lane expressions of
the model compute exactly those values (no wrap-around anywhere, from the `fit` hypotheses of
`Inst.ok`).
-/

namespace Claims.MatmulE5m2

open TVA

/-! ## The lanes -/

/-- The row of `a` (and of `c`) that lane `x` of program `p` handles. -/
def om (d : Inst) (p x : Nat) : Nat := ocm d p x % d.M

/-- The column of `b` (and of `c`) that lane `y` of program `p` handles. -/
def on (d : Inst) (p y : Nat) : Nat := ocn d p y % d.N

/-- The offset into `a` of the element `(om, e)`. -/
def aOff (d : Inst) (p x e : Nat) : Nat := om d p x * d.stride_am + e * d.stride_ak

/-- The offset into `b` of the element `(e, on)`. -/
def bOff (d : Inst) (p y e : Nat) : Nat := e * d.stride_bk + on d p y * d.stride_bn

/-- The offset into `c` of the element `(ocm, ocn)`. -/
def cOff (d : Inst) (p x y : Nat) : Nat := ocm d p x * d.stride_cm + ocn d p y * d.stride_cn

/-- Row `x`'s value at `e`, as the initial memory holds it. -/
def vaOf (d : Inst) (init : Mem) (p x : Nat) : Nat → F8E5 :=
  fun e => (init.read DType.f8e5 (d.a.ptr (aOff d p x e)).addr).getD default

/-- Column `y`'s value at `e`, as the initial memory holds it. -/
def vbOf (d : Inst) (init : Mem) (p y : Nat) : Nat → F8E5 :=
  fun e => (init.read DType.f8e5 (d.b.ptr (bOff d p y e)).addr).getD default

/-! ## `int32` arithmetic -/

/-- `SFits` with the exponent evaluated. -/
theorem sfits_iff {x : ℤ} : IntV.SFits 32 x ↔ -(2 ^ 31) ≤ x ∧ x < 2 ^ 31 := by
  unfold IntV.SFits
  have e : (2 : ℤ) ^ (32 - 1) = 2 ^ 31 := by norm_num
  rw [e]

/-- The value of an unsigned literal below `2^31`. -/
theorem toInt_ofNat' {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = (n : ℤ) := by
  exact IntV.I32.toInt_ofNat h

/-- The value of a signed literal in range. -/
theorem toInt_ofInt' {a : ℤ} (h0 : 0 ≤ a) (h : a < 2 ^ 31) : (IntV.ofInt a : I32).toInt = a := by
  refine IntV.toInt_ofInt (w := 32) (by norm_num) ?_
  rw [sfits_iff]
  exact ⟨by linarith, h⟩

/-- The value of a sum that does not wrap. -/
theorem toInt_add' {a b : I32} (h0 : 0 ≤ a.toInt + b.toInt) (h : a.toInt + b.toInt < 2 ^ 31) :
    (a + b).toInt = a.toInt + b.toInt := by
  refine IntV.toInt_add (w := 32) rfl rfl ?_
  rw [sfits_iff]
  exact ⟨by linarith, h⟩

/-- The value of a difference that does not wrap. -/
theorem toInt_sub' {a b : I32} (h0 : 0 ≤ a.toInt - b.toInt) (h : a.toInt - b.toInt < 2 ^ 31) :
    (a - b).toInt = a.toInt - b.toInt := by
  refine IntV.toInt_sub (w := 32) rfl rfl ?_
  rw [sfits_iff]
  exact ⟨by linarith, h⟩

/-- The value of a product that does not wrap. -/
theorem toInt_mul' {a b : I32} (h0 : 0 ≤ a.toInt * b.toInt) (h : a.toInt * b.toInt < 2 ^ 31) :
    (a * b).toInt = a.toInt * b.toInt := by
  refine IntV.toInt_mul (w := 32) rfl rfl ?_
  rw [sfits_iff]
  exact ⟨by linarith, h⟩

/-- A sum, from the two values. -/
theorem add_toInt {a b : I32} {m n : ℤ} (ha : a.toInt = m) (hb : b.toInt = n)
    (h0 : 0 ≤ m + n) (h : m + n < 2 ^ 31) : (a + b).toInt = m + n := by
  rw [toInt_add' (by rw [ha, hb]; exact h0) (by rw [ha, hb]; exact h), ha, hb]

/-- A difference, from the two values. -/
theorem sub_toInt {a b : I32} {m n : ℤ} (ha : a.toInt = m) (hb : b.toInt = n)
    (h0 : 0 ≤ m - n) (h : m - n < 2 ^ 31) : (a - b).toInt = m - n := by
  rw [toInt_sub' (by rw [ha, hb]; exact h0) (by rw [ha, hb]; exact h), ha, hb]

/-- A product, from the two values. -/
theorem mul_toInt {a b : I32} {m n : ℤ} (ha : a.toInt = m) (hb : b.toInt = n)
    (h0 : 0 ≤ m * n) (h : m * n < 2 ^ 31) : (a * b).toInt = m * n := by
  rw [toInt_mul' (by rw [ha, hb]; exact h0) (by rw [ha, hb]; exact h), ha, hb]

/-- A bound of naturals, in `ℤ`. -/
theorem natCast_lt' {a b : Nat} (h : a < b) : (a : ℤ) < b := by exact_mod_cast h

/-- `Nat.min` of naturals, in `ℤ`. -/
theorem min_cast' {a b : Nat} : ((min a b : ℕ) : ℤ) = Min.min (a : ℤ) (b : ℤ) := by
  rcases le_total a b with h | h
  · rw [Nat.min_eq_left h, min_eq_left (by exact_mod_cast h)]
  · rw [Nat.min_eq_right h, min_eq_right (by exact_mod_cast h)]

/-! ## The bounds `Inst.ok` gives -/

/-- `M` is positive. -/
theorem M_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < d.M := by
  have h := tM_pos hp
  unfold tM at h
  omega

/-- `N` is positive. -/
theorem N_pos {d : Inst} (hd : d.ok) : 0 < d.N := hd.N_dim.1

/-- The number of row tiles is below `2^31`. -/
theorem tM_lt {d : Inst} (hd : d.ok) : tM d < 2 ^ 31 := by
  have := hd.M_le
  unfold tM
  omega

/-- The number of column tiles is below `2^31`. -/
theorem tN_lt {d : Inst} (hd : d.ok) : tN d < 2 ^ 31 := by
  have := hd.N_le
  unfold tN
  omega

/-- The group size `8·tN` fits. -/
theorem npig_lt {d : Inst} (hd : d.ok) : npig d < 2 ^ 31 := by
  have h := hd.N_dim.2.2
  unfold npig tN
  omega

/-- The lane's row is below `M`. -/
theorem om_lt {d : Inst} {p : Nat} (hp : p < grid d) (x : Nat) : om d p x < d.M :=
  Nat.mod_lt _ (M_pos hp)

/-- The lane's column is below `N`. -/
theorem on_lt {d : Inst} (hd : d.ok) (p y : Nat) : on d p y < d.N :=
  Nat.mod_lt _ (N_pos hd)

/-- The lane's row before reduction, when it is in range. -/
theorem om_eq {d : Inst} {p x : Nat} (h : ocm d p x < d.M) : om d p x = ocm d p x :=
  Nat.mod_eq_of_lt h

/-- A column inside the matrix is its own `% N`. -/
theorem on_eq {d : Inst} {p y : Nat} (h : ocn d p y < d.N) : on d p y = ocn d p y :=
  Nat.mod_eq_of_lt h

/-- `pidm·128 + x` fits in `int32`. -/
theorem ocm_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {x : Nat} (hx : x < 128) :
    ocm d p x < 2 ^ 31 := by
  have h1 : pidm d p < tM d := pidm_lt hp
  have h2 := hd.M_le
  unfold ocm
  have : tM d * 128 ≤ d.M + 127 := by
    unfold tM
    omega
  nlinarith

/-- `pidn·256 + y` fits in `int32`. -/
theorem ocn_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {y : Nat} (hy : y < 256) :
    ocn d p y < 2 ^ 31 := by
  have h1 : pidn d p < tN d := pidn_lt hp
  have h2 := hd.N_le
  unfold ocn
  have : tN d * 256 ≤ d.N + 255 := by
    unfold tN
    omega
  nlinarith

/-- `M ≤ 2^31`; `M` itself is an `int32` literal. -/
theorem M_lt {d : Inst} (hd : d.ok) : d.M < 2 ^ 31 := by
  have := hd.M_le; omega

/-- `N` is below `2^31`. -/
theorem N_lt {d : Inst} (hd : d.ok) : d.N < 2 ^ 31 := by
  have := hd.N_le; omega

/-- `pidm` fits. -/
theorem pidm_lim {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : pidm d p < 2 ^ 31 :=
  lt_trans (pidm_lt hp) (tM_lt hd)

/-- `pidn` fits. -/
theorem pidn_lim {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) : pidn d p < 2 ^ 31 :=
  lt_trans (pidn_lt hp) (tN_lt hd)

/-- The offset of a row lane into `a` fits in `int32`. -/
theorem aOff_lim {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {x e : Nat}
    (he : e < 64) : aOff d p x e < 2 ^ 31 := by
  have h1 : om d p x < d.M := om_lt hp x
  have h2 : om d p x * d.stride_am ≤ (d.M - 1) * d.stride_am :=
    Nat.mul_le_mul_right _ (by omega)
  have h3 : e * d.stride_ak ≤ 64 * d.stride_ak := Nat.mul_le_mul_right _ (by omega)
  have h4 := hd.a_fit
  unfold aOff
  omega

/-- The offset of a column lane into `b` fits in `int32`. -/
theorem bOff_lim {d : Inst} (hd : d.ok) {p : Nat} {y e : Nat}
    (he : e < 64) : bOff d p y e < 2 ^ 31 := by
  have h1 : on d p y < d.N := on_lt hd p y
  have h2 : on d p y * d.stride_bn ≤ (d.N - 1) * d.stride_bn :=
    Nat.mul_le_mul_right _ (by omega)
  have h3 : e * d.stride_bk ≤ 64 * d.stride_bk := Nat.mul_le_mul_right _ (by omega)
  have h4 := hd.b_fit
  unfold bOff
  omega

/-- A row lane of `a` is inside the buffer. -/
theorem aOff_lt_len {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {x e : Nat}
    (he : e < d.K) : aOff d p x e < d.a.len :=
  hd.a_len _ _ (om_lt hp x) he

/-- A column lane of `b` is inside the buffer. -/
theorem bOff_lt_len {d : Inst} (hd : d.ok) {p : Nat} {y e : Nat}
    (he : e < d.K) : bOff d p y e < d.b.len :=
  hd.b_len _ _ he (on_lt hd p y)

/-- A `c` lane in range is inside the buffer. -/
theorem cOff_lt_len {d : Inst} (hd : d.ok) {p x y : Nat} (hx : ocm d p x < d.M)
    (hy : ocn d p y < d.N) : cOff d p x y < d.c.len :=
  hd.c_len _ _ hx hy

/-- Distinct `c` lanes in range are distinct offsets. -/
theorem cOff_inj {d : Inst} (hd : d.ok) {p x y x' y' : Nat} (hx : ocm d p x < d.M)
    (hy : ocn d p y < d.N) (hx' : ocm d p x' < d.M) (hy' : ocn d p y' < d.N)
    (h : cOff d p x y = cOff d p x' y') : x = x' ∧ y = y' := by
  unfold cOff at h
  obtain ⟨h1, h2⟩ := hd.c_inj (ocm d p x) (ocn d p y) (ocm d p x') (ocn d p y')
    hx hy hx' hy' h
  have hxeq : x = x' := by
    unfold ocm at h1
    omega
  have hyeq : y = y' := by
    unfold ocn at h2
    omega
  exact ⟨hxeq, hyeq⟩

/-! ## The scalar chain, on values -/

/-- `num_pid_m`. -/
theorem numPidM_val (d : Inst) (hd : d.ok) :
    (IntV.ofInt (((IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32)).toInt /
      (IntV.ofNat 128 : I32).toInt) : I32) = IntV.ofInt (tM d) := by
  have hM_lt : d.M < 2 ^ 31 := by have := hd.M_le; omega
  have hMv : (IntV.ofNat d.M : I32).toInt = (d.M : ℤ) := toInt_ofNat' hM_lt
  have h127 : (IntV.ofNat 127 : I32).toInt = (127 : ℤ) := toInt_ofNat' (by norm_num)
  have hsum : ((IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32)).toInt = (d.M : ℤ) + 127 :=
    add_toInt hMv h127 (by positivity) (by have := hd.M_le; omega)
  have h128 : (IntV.ofNat 128 : I32).toInt = (128 : ℤ) := toInt_ofNat' (by norm_num)
  have hdiv : ((d.M : ℤ) + 127) / 128 = ((tM d : ℕ) : ℤ) := by
    have h1 : ((d.M : ℤ) + 127) = ((d.M + 127 : ℕ) : ℤ) := by push_cast; ring
    rw [h1, show (128 : ℤ) = ((128 : ℕ) : ℤ) from by norm_num, ← Int.natCast_ediv]
  rw [hsum, h128, hdiv]

/-- `num_pid_n`. -/
theorem numPidN_val (d : Inst) (hd : d.ok) :
    (IntV.ofInt (((IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32)).toInt /
      (IntV.ofNat 256 : I32).toInt) : I32) = IntV.ofInt (tN d) := by
  have hN_lt : d.N < 2 ^ 31 := by have := hd.N_le; omega
  have hNv : (IntV.ofNat d.N : I32).toInt = (d.N : ℤ) := toInt_ofNat' hN_lt
  have h255 : (IntV.ofNat 255 : I32).toInt = (255 : ℤ) := toInt_ofNat' (by norm_num)
  have hsum : ((IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32)).toInt = (d.N : ℤ) + 255 :=
    add_toInt hNv h255 (by positivity) (by have := hd.N_le; omega)
  have h256 : (IntV.ofNat 256 : I32).toInt = (256 : ℤ) := toInt_ofNat' (by norm_num)
  have hdiv : ((d.N : ℤ) + 255) / 256 = ((tN d : ℕ) : ℤ) := by
    have h1 : ((d.N : ℤ) + 255) = ((d.N + 255 : ℕ) : ℤ) := by push_cast; ring
    rw [h1, show (256 : ℤ) = ((256 : ℕ) : ℤ) from by norm_num, ← Int.natCast_ediv]
  rw [hsum, h256, hdiv]

/-- The group width `8·tN`. -/
theorem npig_val (d : Inst) (hd : d.ok) :
    ((IntV.ofNat 8 : I32) * IntV.ofInt (tN d)).toInt = 8 * (tN d : ℤ) := by
  have h8 : (IntV.ofNat 8 : I32).toInt = (8 : ℤ) := toInt_ofNat' (by norm_num)
  have htN : (IntV.ofInt (tN d) : I32).toInt = (tN d : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (tN_lt hd))
  have hlt : (8 : ℤ) * (tN d : ℤ) < 2 ^ 31 := by
    have h := npig_lt hd
    have h' : ((8 * tN d : ℕ) : ℤ) < ((2 ^ 31 : ℕ) : ℤ) := by exact_mod_cast h
    push_cast at h'
    simpa using h'
  rw [mul_toInt h8 htN (by positivity) hlt]

/-- `group_id`. -/
theorem groupId_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (IntV.ofInt ((IntV.ofNat p : I32).toInt /
      ((IntV.ofNat 8 : I32) * IntV.ofInt (tN d)).toInt) : I32) = IntV.ofInt (grp d p) := by
  have hpv : (IntV.ofNat p : I32).toInt = (p : ℤ) :=
    toInt_ofNat' (lt_trans hp (grid_lt hd))
  have hdv : ((IntV.ofNat 8 : I32) * IntV.ofInt (tN d)).toInt = 8 * (tN d : ℤ) :=
    npig_val d hd
  have hdiv : (p : ℤ) / (8 * (tN d : ℤ)) = ((grp d p : ℕ) : ℤ) := by
    have hnp : (8 * (tN d : ℤ)) = ((npig d : ℕ) : ℤ) := by
      unfold npig
      push_cast
      ring
    rw [hnp, ← Int.natCast_ediv]
  rw [hpv, hdv, hdiv]

/-- `first_pid_m`. -/
theorem first_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32) : I32) = IntV.ofInt (first d p) := by
  have hgN : grp d p < 2 ^ 31 := lt_trans (grp_lt hp) (tM_lt hd)
  have hgv : (IntV.ofInt (grp d p) : I32).toInt = (grp d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' hgN)
  have h8 : (IntV.ofNat 8 : I32).toInt = (8 : ℤ) := toInt_ofNat' (by norm_num)
  have hfirst : (first d p : ℤ) = (grp d p : ℤ) * 8 := by
    unfold first
    push_cast
    ring
  have hprod : (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32) : I32).toInt = (first d p : ℤ) := by
    rw [mul_toInt hgv h8 (by positivity)
      (by rw [← hfirst]; exact natCast_lt' (lt_trans (first_lt hp) (tM_lt hd)))]
    exact hfirst.symm
  rw [← IntV.ofInt_toInt (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32)), hprod]

/-- `num_pid_m - first_pid_m`. -/
theorem numPidM_sub_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32)).toInt
      = (tM d : ℤ) - (first d p : ℤ) := by
  have hA : (IntV.ofInt (tM d) : I32).toInt = (tM d : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (tM_lt hd))
  have hB : (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32) : I32).toInt = (first d p : ℤ) := by
    rw [first_val d hd hp]
    exact toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (first_lt hp) (tM_lt hd)))
  have h0 : 0 ≤ (tM d : ℤ) - (first d p : ℤ) := by
    have h := natCast_lt' (first_lt hp)
    have h2 : (tM d : ℤ) < 2 ^ 31 := by
      have h3 := tM_lt hd
      exact_mod_cast h3
    linarith
  have hlt : (tM d : ℤ) - (first d p : ℤ) < 2 ^ 31 := by
    have h := natCast_lt' (first_lt hp)
    have h2 : (tM d : ℤ) < 2 ^ 31 := by
      have h3 := tM_lt hd
      exact_mod_cast h3
    linarith
  exact sub_toInt hA hB h0 hlt

/-- `group_size_m`. -/
theorem gsm_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (IntV.min ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32))
      (IntV.ofNat 8 : I32)) = IntV.ofInt (gsz d p) := by
  have hA : ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32)).toInt
      = (tM d : ℤ) - (first d p : ℤ) := numPidM_sub_val d hd hp
  have hB : (IntV.ofNat 8 : I32).toInt = (8 : ℤ) := toInt_ofNat' (by norm_num)
  have hmin : (IntV.min ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32))
      (IntV.ofNat 8 : I32)).toInt
      = Min.min ((tM d : ℤ) - (first d p : ℤ)) 8 := by
    rw [IntV.toInt_min, hA, hB]
  have hgsz : Min.min ((tM d : ℤ) - (first d p : ℤ)) (8 : ℤ) = (gsz d p : ℤ) := by
    have hle : first d p ≤ tM d := le_of_lt (first_lt hp)
    have hdef : gsz d p = min (tM d - first d p) 8 := rfl
    rw [hdef, min_cast']
    push_cast [Nat.cast_sub hle]
    ring
  rw [← IntV.ofInt_toInt (IntV.min ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d p) *
    (IntV.ofNat 8 : I32)) (IntV.ofNat 8 : I32)), hmin, hgsz]

/-- `pid % num_pid_in_group`. -/
theorem pidRmd_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (IntV.ofInt ((IntV.ofNat p : I32).toInt %
      ((IntV.ofNat 8 : I32) * IntV.ofInt (tN d)).toInt) : I32) = IntV.ofInt (rmd d p) := by
  have hpv : (IntV.ofNat p : I32).toInt = (p : ℤ) :=
    toInt_ofNat' (lt_trans hp (grid_lt hd))
  have hdv : ((IntV.ofNat 8 : I32) * IntV.ofInt (tN d)).toInt = 8 * (tN d : ℤ) :=
    npig_val d hd
  have hmod : (p : ℤ) % (8 * (tN d : ℤ)) = ((rmd d p : ℕ) : ℤ) := by
    have hnp : (8 * (tN d : ℤ)) = ((npig d : ℕ) : ℤ) := by
      unfold npig
      push_cast
      ring
    rw [hnp, ← Int.natCast_emod]
  rw [hpv, hdv, hmod]

/-- `pid_m`, the reduction of the remainder by the group size. -/
theorem rmdMod_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (IntV.ofInt ((IntV.ofInt (rmd d p) : I32).toInt / (IntV.ofInt (gsz d p) : I32).toInt) : I32)
      = IntV.ofInt (pidn d p) := by
  have hrv : (IntV.ofInt (rmd d p) : I32).toInt = (rmd d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (rmd_lt hp) (npig_lt hd)))
  have hgv : (IntV.ofInt (gsz d p) : I32).toInt = (gsz d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (lt_of_le_of_lt (gsz_le_eight d p) (by norm_num)))
  have hdiv : (rmd d p : ℤ) / (gsz d p : ℤ) = ((pidn d p : ℕ) : ℤ) := by
    rw [← Int.natCast_ediv]
  rw [hrv, hgv, hdiv]

/-- `pid_m`. -/
theorem pidm_val (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32) +
      IntV.ofInt ((IntV.ofInt (rmd d p) : I32).toInt % (IntV.ofInt (gsz d p) : I32).toInt) : I32)
      = IntV.ofInt (pidm d p) := by
  have hA : (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32) : I32).toInt = (first d p : ℤ) := by
    rw [first_val d hd hp]
    exact toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (first_lt hp) (tM_lt hd)))
  have hrv : (IntV.ofInt (rmd d p) : I32).toInt = (rmd d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (rmd_lt hp) (npig_lt hd)))
  have hgv : (IntV.ofInt (gsz d p) : I32).toInt = (gsz d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (lt_of_le_of_lt (gsz_le_eight d p) (by norm_num)))
  have hlt : rmd d p % gsz d p < 2 ^ 31 := by
    have h1 : rmd d p % gsz d p < gsz d p := Nat.mod_lt _ (gsz_pos hp)
    have h2 := gsz_le_eight d p
    omega
  have hB : (IntV.ofInt ((IntV.ofInt (rmd d p) : I32).toInt %
      (IntV.ofInt (gsz d p) : I32).toInt) : I32).toInt = ((rmd d p % gsz d p : ℕ) : ℤ) := by
    rw [hrv, hgv, ← Int.natCast_emod]
    exact toInt_ofInt' (by positivity) (natCast_lt' hlt)
  have hmn : (first d p : ℤ) + ((rmd d p % gsz d p : ℕ) : ℤ) = (pidm d p : ℤ) := by
    have he : pidm d p = first d p + rmd d p % gsz d p := rfl
    rw [he]
    push_cast
    ring
  have h0 : 0 ≤ (first d p : ℤ) + ((rmd d p % gsz d p : ℕ) : ℤ) := by
    rw [hmn]; positivity
  have hlt2 : (first d p : ℤ) + ((rmd d p % gsz d p : ℕ) : ℤ) < 2 ^ 31 := by
    rw [hmn]
    exact natCast_lt' (lt_trans (pidm_lt hp) (tM_lt hd))
  rw [← IntV.ofInt_toInt (IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32) +
    IntV.ofInt ((IntV.ofInt (rmd d p) : I32).toInt % (IntV.ofInt (gsz d p) : I32).toInt))]
  rw [add_toInt hA hB h0 hlt2, hmn]

end Claims.MatmulE5m2
