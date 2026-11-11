import Claim
import TVA.Proof
import Proof.Geometry
import Proof.Compute
import Proof.Numeric

/-!
# The lanes of the tile: row and column offsets, pointer lanes, masks and hints of the model's
lines, each as a statement about its element.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-! ### Lane values of the arithmetic -/

/-- A lane of `tl.where` picks `x` or `y` by the condition. -/
theorem get1_where {n : Nat} (c : Tensor [n] Bool) (x y : Tensor [n] I32) (i : Fin n) :
    (Tensor.where_ c x y).get1 i = if c.get1 i then x.get1 i else y.get1 i := rfl

/-- A lane of `tl.where` with a true condition is `x`. -/
theorem where_of_true {n : Nat} {c : Tensor [n] Bool} {x y : Tensor [n] I32} (i : Fin n)
    (h : c.get1 i = true) : (Tensor.where_ c x y).get1 i = x.get1 i := by
  rw [get1_where, h]; rfl

/-- A lane of a splat is the splatted value. -/
theorem get1_full {n : Nat} {α : Type} (a : α) (i : Fin n) :
    (Tensor.full a : Tensor [n] α).get1 i = a := rfl

/-- A lane of `x < a` is the comparison of the lane's value. -/
theorem ltS_get {n : Nat} (x : Tensor [n] I32) (a : I32) (i : Fin n) :
    (x.ltS a).get1 i = decide ((x.get1 i).toInt < a.toInt) := by
  show IntV.lt (x.get1 i) a = decide ((x.get1 i).toInt < a.toInt)
  exact IntV.lt_eq _ _

/-- Lane `i` of `tl.arange(0, n)` has value `i`. -/
theorem arange_get1_toInt {n : Nat} (hn : n < 2 ^ 31) (i : Fin n) :
    ((tl.arange 0 n).get1 i).toInt = (i.val : Int) := by
  show ((Tensor.arange (0 : Int) n).get1 i).toInt = (i.val : Int)
  rw [Tensor.get1_arange]
  simp only [Int.zero_add]
  refine IntV.toInt_ofInt (by norm_num) ⟨?_, ?_⟩
  · have h0 : (0 : Int) ≤ (i.val : Int) := by exact_mod_cast Nat.zero_le i.val
    omega
  · have h1 : (i.val : Int) < (n : Int) := by exact_mod_cast i.isLt
    have h2 : (n : Int) < 2 ^ 31 := by exact_mod_cast hn
    omega

/-! ### The tile geometry -/

/-- The element of `a` that lane `(i, j)` of block `kk` of tile `T` reads. -/
def aG (d : Inst) (T kk i j : Nat) : Nat :=
  (pm d T * 128 + i) * d.stride_am + (kk * 64 + j) * d.stride_ak

/-- The element of `b` that lane `(i, j)` of block `kk` of tile `T` reads. -/
def bG (d : Inst) (T kk i j : Nat) : Nat :=
  (kk * 64 + i) * d.stride_bk + (pn d T * 256 + j) * d.stride_bn

/-- The element of `c` that lane `(i, j)` of tile `T` holds. -/
def cG (d : Inst) (T i j : Nat) : Nat :=
  (pm d T * 128 + i) * d.stride_cm + (pn d T * 256 + j) * d.stride_cn

/-- The element of `a` that lane `(i, k)` of the loop step `kk` of tile `T` loads. -/
def aGs (d : Inst) (T kk : Nat) : Idx [128, 64] → Nat :=
  fun x => aG d T kk x.1.val x.2.1.val

/-- The element of `b` that lane `(k, j)` of the loop step `kk` of tile `T` loads. -/
def bGs (d : Inst) (T kk : Nat) : Idx [64, 256] → Nat :=
  fun x => bG d T kk x.1.val x.2.1.val

/-- The element of `c` that lane `(i, j)` of tile `T` stores. -/
def cGs (d : Inst) (T : Nat) : Idx [128, 256] → Nat :=
  fun x => cG d T x.1.val x.2.1.val

/-! ### Bounds on the integer offsets -/

/-- A row of `a` times its stride fits in `int32`. -/
theorem a_mul_lt {d : Inst} (hd : d.ok) {a : Nat} (ha : a ≤ d.M - 1) :
    a * d.stride_am < 2 ^ 31 := by
  have h1 : a * d.stride_am ≤ (d.M - 1) * d.stride_am := Nat.mul_le_mul_right _ ha
  have h2 : (d.M - 1) * d.stride_am ≤ (d.M - 1) * d.stride_am + (d.K - 1) * d.stride_ak := by
    omega
  exact Nat.lt_of_le_of_lt (Nat.le_trans h1 h2) hd.a_fit

/-- A column of `a` times its stride fits in `int32`. -/
theorem ak_mul_lt {d : Inst} (hd : d.ok) {a : Nat} (ha : a ≤ d.K - 1) :
    a * d.stride_ak < 2 ^ 31 := by
  have h1 : a * d.stride_ak ≤ (d.K - 1) * d.stride_ak := Nat.mul_le_mul_right _ ha
  have h2 : (d.K - 1) * d.stride_ak ≤ (d.M - 1) * d.stride_am + (d.K - 1) * d.stride_ak := by
    omega
  exact Nat.lt_of_le_of_lt (h1.trans h2) hd.a_fit

/-- The offset of an element of `a` fits in `int32`. -/
theorem a_sum_lt {d : Inst} (hd : d.ok) {a b : Nat} (ha : a ≤ d.M - 1) (hb : b ≤ d.K - 1) :
    a * d.stride_am + b * d.stride_ak < 2 ^ 31 := by
  have h1 : a * d.stride_am ≤ (d.M - 1) * d.stride_am := Nat.mul_le_mul_right _ ha
  have h2 : b * d.stride_ak ≤ (d.K - 1) * d.stride_ak := Nat.mul_le_mul_right _ hb
  exact Nat.lt_of_le_of_lt (Nat.add_le_add h1 h2) hd.a_fit

/-- A row of `b` times its stride fits in `int32`. -/
theorem bk_mul_lt {d : Inst} (hd : d.ok) {a : Nat} (ha : a ≤ d.K - 1) :
    a * d.stride_bk < 2 ^ 31 := by
  have h1 : a * d.stride_bk ≤ (d.K - 1) * d.stride_bk := Nat.mul_le_mul_right _ ha
  have h2 : (d.K - 1) * d.stride_bk ≤ (d.K - 1) * d.stride_bk + (d.N - 1) * d.stride_bn := by
    omega
  exact Nat.lt_of_le_of_lt (h1.trans h2) hd.b_fit

/-- A column of `b` times its stride fits in `int32`. -/
theorem bn_mul_lt {d : Inst} (hd : d.ok) {b : Nat} (hb : b ≤ d.N - 1) :
    b * d.stride_bn < 2 ^ 31 := by
  have h1 : b * d.stride_bn ≤ (d.N - 1) * d.stride_bn := Nat.mul_le_mul_right _ hb
  have h2 : (d.N - 1) * d.stride_bn ≤ (d.K - 1) * d.stride_bk + (d.N - 1) * d.stride_bn := by
    omega
  exact Nat.lt_of_le_of_lt (h1.trans h2) hd.b_fit

/-- The offset of an element of `b` fits in `int32`. -/
theorem b_sum_lt {d : Inst} (hd : d.ok) {a b : Nat} (ha : a ≤ d.K - 1) (hb : b ≤ d.N - 1) :
    a * d.stride_bk + b * d.stride_bn < 2 ^ 31 := by
  have h1 : a * d.stride_bk ≤ (d.K - 1) * d.stride_bk := Nat.mul_le_mul_right _ ha
  have h2 : b * d.stride_bn ≤ (d.N - 1) * d.stride_bn := Nat.mul_le_mul_right _ hb
  exact Nat.lt_of_le_of_lt (Nat.add_le_add h1 h2) hd.b_fit

/-- A row of `c` times its stride fits in `int32`. -/
theorem cm_mul_lt {d : Inst} (hd : d.ok) {a : Nat} (ha : a ≤ d.M - 1) :
    a * d.stride_cm < 2 ^ 31 := by
  have h1 : a * d.stride_cm ≤ (d.M - 1) * d.stride_cm := Nat.mul_le_mul_right _ ha
  exact Nat.lt_of_le_of_lt h1 hd.cm_fit

/-- A column of `c` times its stride fits in `int32`. -/
theorem cn_mul_lt {d : Inst} (hd : d.ok) {a : Nat} (ha : a ≤ d.N - 1) :
    a * d.stride_cn < 2 ^ 31 := by
  have h1 : a * d.stride_cn ≤ (d.N - 1) * d.stride_cn := Nat.mul_le_mul_right _ ha
  exact Nat.lt_of_le_of_lt h1 hd.cn_fit

/-! ### The tile geometry: the indices are in range -/

/-- A row `i` of tile row `pm d T` is below `M`. -/
theorem pm_mul_bound {d : Inst} (hd : d.ok) {T : Nat} (hT : T < nT d) {i : Nat} (hi : i < 128) :
    pm d T * 128 + i < d.M := by
  have h1 := pm_lt hd hT
  have h3 : d.M / 128 * 128 = d.M :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd.M_tile)
  simp only [nPM] at h1
  omega

/-- A tile column's first column, plus a column of the tile, is below `N`. -/
theorem pn_mul_bound {d : Inst} (hd : d.ok) {T : Nat} (hT : T < nT d) {j : Nat} (hj : j < 256) :
    pn d T * 256 + j < d.N := by
  have h1 := pn_lt hd hT
  have h3 : d.N / 256 * 256 = d.N :=
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd.N_tile)
  simp only [nPN] at h1
  omega

/-- Column `j` of loop step `t` is below `K`. -/
theorem kk_bound {d : Inst} (hd : d.ok) {t : Nat} (ht : t < steps d.K) {j : Nat} (hj : j < 64) :
    t * 64 + j < d.K := by
  have h1 : steps d.K = d.K / 64 := steps_eq hd
  have h2 : d.K / 64 * 64 = d.K := Nat.div_mul_cancel (K_dvd hd)
  omega

/-- The first column of a loop step is below `K`. -/
theorem kk_mul_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < steps d.K) : t * 64 < d.K := by
  have h1 : steps d.K = d.K / 64 := steps_eq hd
  have h2 : d.K / 64 * 64 ≤ d.K := Nat.div_mul_le_self _ _
  have h3 : t * 64 < steps d.K * 64 := by
    rw [Nat.mul_lt_mul_right (by norm_num)]
    exact ht
  calc t * 64 < steps d.K * 64 := h3
    _ = d.K / 64 * 64 := by rw [h1]
    _ ≤ d.K := h2

/-- A loaded element of `a` is in the footprint `aIdx`. -/
theorem aG_mem {d : Inst} (hd : d.ok) {T kk i j : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hi : i < 128) (hj : j < 64) : aG d T kk i j ∈ aIdx d := by
  exact ⟨pm d T * 128 + i, kk * 64 + j, pm_mul_bound hd hT hi, kk_bound hd hkk hj, rfl⟩

/-- A loaded element of `b` is in the footprint `bIdx`. -/
theorem bG_mem {d : Inst} (hd : d.ok) {T kk i j : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hi : i < 64) (hj : j < 256) : bG d T kk i j ∈ bIdx d := by
  exact ⟨kk * 64 + i, pn d T * 256 + j, kk_bound hd hkk hi, pn_mul_bound hd hT hj, rfl⟩

/-- A stored element of `c` is in the tile. -/
theorem cG_mem {d : Inst} (hd : d.ok) {T i j : Nat} (hT : T < nT d) (hi : i < 128)
    (hj : j < 256) : cG d T i j ∈ cIdx d := by
  exact ⟨pm d T * 128 + i, pn d T * 256 + j, pm_mul_bound hd hT hi, pn_mul_bound hd hT hj, rfl⟩

/-- The index of a loaded element of `a` fits in `int32`. -/
theorem aG_lt {d : Inst} (hd : d.ok) {T kk i j : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hi : i < 128) (hj : j < 64) : aG d T kk i j < 2 ^ 31 := by
  have hx : pm d T * 128 + i ≤ d.M - 1 := by have := pm_mul_bound hd hT hi; omega
  have hy : kk * 64 + j ≤ d.K - 1 := by have := kk_bound hd hkk hj; omega
  exact a_sum_lt hd hx hy

/-- A loaded element of `b` is a valid position of `b`. -/
theorem bG_lt {d : Inst} (hd : d.ok) {T kk i j : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hi : i < 64) (hj : j < 256) : bG d T kk i j < 2 ^ 31 := by
  have hx : kk * 64 + i ≤ d.K - 1 := by have := kk_bound hd hkk hi; omega
  have hy : pn d T * 256 + j ≤ d.N - 1 := by have := pn_mul_bound hd hT hj; omega
  exact b_sum_lt hd hx hy

/-- A loaded element of `a` is inside the buffer. -/
theorem aG_len {d : Inst} (hd : d.ok) {T kk i j : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hi : i < 128) (hj : j < 64) : aG d T kk i j < d.a.len :=
  hd.a_len _ _ (pm_mul_bound hd hT hi) (kk_bound hd hkk hj)

/-- A loaded element of `b` is inside the buffer. -/
theorem bG_len {d : Inst} (hd : d.ok) {T kk i j : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hi : i < 64) (hj : j < 256) : bG d T kk i j < d.b.len :=
  hd.b_len _ _ (kk_bound hd hkk hi) (pn_mul_bound hd hT hj)

/-- A stored element of `c` is inside the buffer. -/
theorem cG_len {d : Inst} (hd : d.ok) {T i j : Nat} (hT : T < nT d) (hi : i < 128)
    (hj : j < 256) : cG d T i j < d.c.len :=
  hd.c_len _ _ (pm_mul_bound hd hT hi) (pn_mul_bound hd hT hj)

/-- The lanes of an `a` load lie in `aIdx`. -/
theorem aGs_mem {d : Inst} (hd : d.ok) {T kk : Nat} (hT : T < nT d) (hkk : kk < steps d.K) :
    Buf.img (aGs d T kk) (fun _ => true) ⊆ aIdx d := by
  intro x hx
  simp only [Buf.img, Set.mem_ofPred_eq] at hx
  obtain ⟨i, -, hix⟩ := hx
  subst hix
  exact aG_mem hd hT hkk i.1.isLt i.2.1.isLt

/-- The lanes of a `b` load lie in `bIdx`. -/
theorem bGs_mem {d : Inst} (hd : d.ok) {T kk : Nat} (hT : T < nT d) (hkk : kk < steps d.K) :
    Buf.img (bGs d T kk) (fun _ => true) ⊆ bIdx d := by
  intro x hx
  simp only [Buf.img, Set.mem_ofPred_eq] at hx
  obtain ⟨i, -, hix⟩ := hx
  subst hix
  exact bG_mem hd hT hkk i.1.isLt i.2.1.isLt

/-- The lanes of a `c` store are the tile. -/
theorem cGs_img {d : Inst} (T : Nat) :
    Buf.img (cGs d T) (fun _ => true) = tileOf d T := by
  ext e
  constructor
  · intro hx
    simp only [Buf.img, Set.mem_ofPred_eq] at hx
    obtain ⟨x, -, hix⟩ := hx
    subst hix
    exact ⟨x.1.val, x.2.1.val, x.1.isLt, x.2.1.isLt, rfl⟩
  · intro hx
    obtain ⟨i, j, hi, hj, he⟩ := hx
    subst he
    exact ⟨(⟨i, hi⟩, ⟨j, hj⟩, ()), rfl, rfl⟩

/-- The lanes of a `c` tile are distinct elements of `c`. -/
theorem cGs_inj {d : Inst} (hd : d.ok) {T : Nat} (hT : T < nT d) :
    ∀ x y : Idx [128, 256], (fun _ => true) x = true → (fun _ => true) y = true →
      cGs d T x = cGs d T y → x = y := by
  rintro ⟨x1, x2, xu⟩ ⟨y1, y2, yu⟩ - - h
  have h1 := pm_mul_bound hd hT x1.isLt
  have h2 := pn_mul_bound hd hT x2.isLt
  have h3 := pm_mul_bound hd hT y1.isLt
  have h4 := pn_mul_bound hd hT y2.isLt
  obtain ⟨e1, e2⟩ := hd.c_inj _ _ _ _ h1 h2 h3 h4 h
  have hx : x1.val = y1.val := by omega
  have hy : x2.val = y2.val := by omega
  have hf1 : x1 = y1 := Fin.ext hx
  have hf2 : x2 = y2 := Fin.ext hy
  rw [hf1, hf2]
  cases xu
  cases yu
  rfl

/-! ### The pointer lanes -/

/-- Lane `(i, k)` of the model's `a` pointers is the element `aGs` of `a`. -/
theorem aptrs_lane {d : Inst} (hd : d.ok) {offs_am : Tensor [128] I32} {offs_k : Tensor [64] I32}
    {T kk : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (ham : ∀ i : Fin 128, (offs_am.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int))
    (hok : ∀ j : Fin 64, (offs_k.get1 j).toInt = ((kk * 64 + j.val : Nat) : Int))
    (x : Idx [128, 64]) :
    ((d.a.ptr 0 + (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) +
        offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))).get x).addr =
      (d.a.ptr (aG d T kk x.1.val x.2.1.val)).addr := by
  have hx1 : pm d T * 128 + x.1.val ≤ d.M - 1 := by
    have := pm_mul_bound hd hT x.1.isLt; omega
  have hx2 : kk * 64 + x.2.1.val ≤ d.K - 1 := by
    have := kk_bound hd hkk x.2.1.isLt; omega
  have hP : ((offs_am.col 64 * (IntV.ofNat d.stride_am : I32)).get x).toInt
      = (((pm d T * 128 + x.1.val) * d.stride_am : Nat) : Int) := by
    show ((offs_am.get1 x.1) * (IntV.ofNat d.stride_am : I32)).toInt = _
    exact IntV.I32.mul_nat (ham x.1) (IntV.I32.toInt_ofNat (by have := hd.am; omega))
      (a_mul_lt hd hx1)
  have hQ : ((offs_k.row 128 * (IntV.ofNat d.stride_ak : I32)).get x).toInt
      = (((kk * 64 + x.2.1.val) * d.stride_ak : Nat) : Int) := by
    show ((offs_k.get1 x.2.1) * (IntV.ofNat d.stride_ak : I32)).toInt = _
    exact IntV.I32.mul_nat (hok x.2.1) (IntV.I32.toInt_ofNat (by have := hd.ak; omega))
      (ak_mul_lt hd hx2)
  have hs : ((pm d T * 128 + x.1.val) * d.stride_am
      + (kk * 64 + x.2.1.val) * d.stride_ak) < 2 ^ 31 := a_sum_lt hd hx1 hx2
  have hSum : ((offs_am.col 64 * (IntV.ofNat d.stride_am : I32) +
      offs_k.row 128 * (IntV.ofNat d.stride_ak : I32)).get x).toInt
      = (aG d T kk x.1.val x.2.1.val : Int) := by
    show (IntV.toInt (((offs_am.col 64 * (IntV.ofNat d.stride_am : I32)).get x) +
      ((offs_k.row 128 * (IntV.ofNat d.stride_ak : I32)).get x))) = _
    rw [IntV.toInt_add rfl rfl (by
      rw [hP, hQ]
      constructor
      · have h0 : (0 : Int) ≤ (((pm d T * 128 + x.1.val) * d.stride_am : Nat) : Int) := by
          exact_mod_cast Nat.zero_le _
        omega
      · have h2 : (((pm d T * 128 + x.1.val) * d.stride_am : Nat) : Int)
            + (((kk * 64 + x.2.1.val) * d.stride_ak : Nat) : Int) < 2 ^ 31 := by
          exact_mod_cast hs
        omega), hP, hQ]
    simp only [aG, Nat.cast_add, Nat.cast_mul]
  have e0 : ((d.a.ptr 0 + (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) +
      offs_k.row 128 * (IntV.ofNat d.stride_ak : I32))).get x).addr =
      ((d.a.ptr 0).offset (((offs_am.col 64 * (IntV.ofNat d.stride_am : I32) +
        offs_k.row 128 * (IntV.ofNat d.stride_ak : I32)).get x).toInt)).addr := rfl
  rw [e0, hSum]
  simp only [Buf.ptr_addr, Ptr.offset_addr]
  ring

/-- Lane `(k, j)` of the model's `b` pointers is the element `bGs` of `b`. -/
theorem bptrs_lane {d : Inst} (hd : d.ok) {offs_bn : Tensor [256] I32} {offs_k : Tensor [64] I32}
    {T kk : Nat} (hT : T < nT d) (hkk : kk < steps d.K)
    (hbn : ∀ j : Fin 256, (offs_bn.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int))
    (hok : ∀ i : Fin 64, (offs_k.get1 i).toInt = ((kk * 64 + i.val : Nat) : Int))
    (x : Idx [64, 256]) :
    ((d.b.ptr 0 + (offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) +
        offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32))).get x).addr =
      (d.b.ptr (bG d T kk x.1.val x.2.1.val)).addr := by
  have hx1 : kk * 64 + x.1.val ≤ d.K - 1 := by
    have := kk_bound hd hkk x.1.isLt; omega
  have hx2 : pn d T * 256 + x.2.1.val ≤ d.N - 1 := by
    have := pn_mul_bound hd hT x.2.1.isLt; omega
  have hP : ((offs_k.col 256 * (IntV.ofNat d.stride_bk : I32)).get x).toInt
      = (((kk * 64 + x.1.val) * d.stride_bk : Nat) : Int) := by
    show ((offs_k.get1 x.1) * (IntV.ofNat d.stride_bk : I32)).toInt = _
    exact IntV.I32.mul_nat (hok x.1) (IntV.I32.toInt_ofNat (by have := hd.bk; omega))
      (bk_mul_lt hd hx1)
  have hQ : ((offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32)).get x).toInt
      = (((pn d T * 256 + x.2.1.val) * d.stride_bn : Nat) : Int) := by
    show ((offs_bn.get1 x.2.1) * (IntV.ofNat d.stride_bn : I32)).toInt = _
    exact IntV.I32.mul_nat (hbn x.2.1) (IntV.I32.toInt_ofNat (by have := hd.bn; omega))
      (bn_mul_lt hd hx2)
  have hs : ((kk * 64 + x.1.val) * d.stride_bk
      + (pn d T * 256 + x.2.1.val) * d.stride_bn) < 2 ^ 31 := b_sum_lt hd hx1 hx2
  have hSum : ((offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) +
      offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32)).get x).toInt
      = (bG d T kk x.1.val x.2.1.val : Int) := by
    show (IntV.toInt (((offs_k.col 256 * (IntV.ofNat d.stride_bk : I32)).get x) +
      ((offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32)).get x))) = _
    rw [IntV.toInt_add rfl rfl (by
      rw [hP, hQ]
      constructor
      · have h0 : (0 : Int) ≤ (((kk * 64 + x.1.val) * d.stride_bk : Nat) : Int) := by
          exact_mod_cast Nat.zero_le _
        omega
      · have h2 : (((kk * 64 + x.1.val) * d.stride_bk : Nat) : Int)
            + (((pn d T * 256 + x.2.1.val) * d.stride_bn : Nat) : Int) < 2 ^ 31 := by
          exact_mod_cast hs
        omega), hP, hQ]
    simp only [bG, Nat.cast_add, Nat.cast_mul]
  have e0 : ((d.b.ptr 0 + (offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) +
      offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32))).get x).addr =
      ((d.b.ptr 0).offset (((offs_k.col 256 * (IntV.ofNat d.stride_bk : I32) +
        offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32)).get x).toInt)).addr := rfl
  rw [e0, hSum]
  simp only [Buf.ptr_addr, Ptr.offset_addr]
  ring

/-- Lane `(i, j)` of the model's `c` pointers is the element `cGs` of `c`. -/
theorem cptrs_lane {d : Inst} (hd : d.ok) {offs_cm : Tensor [128] I32}
    {offs_cn : Tensor [256] I32} {T : Nat} (hT : T < nT d)
    (hcm : ∀ i : Fin 128, (offs_cm.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int))
    (hcn : ∀ j : Fin 256, (offs_cn.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int))
    (x : Idx [128, 256]) :
    ((d.c.ptr 0 + (IntV.ofNat d.stride_cm : I32) * offs_cm.col 256 +
        (IntV.ofNat d.stride_cn : I32) * offs_cn.row 128).get x).addr =
      (d.c.ptr (cG d T x.1.val x.2.1.val)).addr := by
  have hx1 : pm d T * 128 + x.1.val ≤ d.M - 1 := by
    have := pm_mul_bound hd hT x.1.isLt; omega
  have hx2 : pn d T * 256 + x.2.1.val ≤ d.N - 1 := by
    have := pn_mul_bound hd hT x.2.1.isLt; omega
  have hP : (((IntV.ofNat d.stride_cm : I32) * offs_cm.col 256).get x).toInt
      = (((pm d T * 128 + x.1.val) * d.stride_cm : Nat) : Int) := by
    show ((IntV.ofNat d.stride_cm : I32) * (offs_cm.get1 x.1)).toInt = _
    have hm := IntV.I32.mul_nat (a := (IntV.ofNat d.stride_cm : I32))
      (b := offs_cm.get1 x.1)
      (IntV.I32.toInt_ofNat (by have := hd.cm; omega)) (hcm x.1)
      (by rw [Nat.mul_comm]; exact cm_mul_lt hd hx1)
    rw [hm, Nat.mul_comm]
  have hQ : (((IntV.ofNat d.stride_cn : I32) * offs_cn.row 128).get x).toInt
      = (((pn d T * 256 + x.2.1.val) * d.stride_cn : Nat) : Int) := by
    show ((IntV.ofNat d.stride_cn : I32) * (offs_cn.get1 x.2.1)).toInt = _
    have hm := IntV.I32.mul_nat (a := (IntV.ofNat d.stride_cn : I32))
      (b := offs_cn.get1 x.2.1)
      (IntV.I32.toInt_ofNat (by have := hd.cn; omega)) (hcn x.2.1)
      (by rw [Nat.mul_comm]; exact cn_mul_lt hd hx2)
    rw [hm, Nat.mul_comm]
  have e0 : ((d.c.ptr 0 + (IntV.ofNat d.stride_cm : I32) * offs_cm.col 256 +
      (IntV.ofNat d.stride_cn : I32) * offs_cn.row 128).get x).addr =
      ((((d.c.ptr 0).offset (((IntV.ofNat d.stride_cm : I32) * offs_cm.col 256).get x).toInt)).offset
        ((((IntV.ofNat d.stride_cn : I32) * offs_cn.row 128).get x).toInt)).addr := rfl
  rw [e0, hP, hQ]
  simp only [Buf.ptr_addr, Ptr.offset_addr]
  rw [show cG d T x.1.val x.2.1.val
      = ((pm d T * 128 + x.1.val) * d.stride_cm
        + (pn d T * 256 + x.2.1.val) * d.stride_cn) from rfl]
  push_cast
  ring

/-! ### The masks -/

/-- Lane `(i, k)` of the `a` mask holds when the column is below `K`. -/
theorem amask_get {d : Inst} (hd : d.ok) {offs_k_for_mask : Tensor [64] I32}
    (hkm : ∀ j : Fin 64, (offs_k_for_mask.get1 j).toInt = (j.val : Int))
    {ki : I32} {kk : Nat} (hki : ki.toInt = (kk : Int)) (hkk : kk < steps d.K)
    (x : Idx [128, 64]) :
    ((offs_k_for_mask.row 128).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32))).get x
      = true := by
  have hK : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have hb : ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32)).toInt
      = ((d.K - kk * 64 : Nat) : Int) := by
    have hm : (ki * (IntV.ofNat 64 : I32)).toInt = ((kk * 64 : Nat) : Int) :=
      IntV.I32.mul_nat hki (IntV.I32.toInt_ofNat (by norm_num))
        (by have h := kk_mul_lt hd hkk; omega)
    have hxy : kk * 64 ≤ d.K := by have := kk_bound hd hkk x.2.1.isLt; omega
    exact IntV.I32.sub_nat (IntV.I32.toInt_ofNat hK) hm hxy hK
  show IntV.lt ((offs_k_for_mask.row 128).get x)
    ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32)) = true
  rw [IntV.lt_eq]
  rw [show (offs_k_for_mask.row 128).get x = offs_k_for_mask.get1 x.2.1 from rfl, hkm x.2.1, hb]
  apply decide_eq_true
  have h := kk_bound hd hkk x.2.1.isLt
  have h2 : (x.2.1.val : Int) < ((d.K - kk * 64 : Nat) : Int) := by
    have hle : x.2.1.val < d.K - kk * 64 := by omega
    exact_mod_cast hle
  exact h2

/-- Lane `(k, j)` of the `b` mask holds when the row is below `K`. -/
theorem bmask_get {d : Inst} (hd : d.ok) {offs_k_for_mask : Tensor [64] I32}
    (hkm : ∀ j : Fin 64, (offs_k_for_mask.get1 j).toInt = (j.val : Int))
    {ki : I32} {kk : Nat} (hki : ki.toInt = (kk : Int)) (hkk : kk < steps d.K)
    (x : Idx [64, 256]) :
    ((offs_k_for_mask.col 256).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32))).get x
      = true := by
  have hK : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have hb : ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32)).toInt
      = ((d.K - kk * 64 : Nat) : Int) := by
    have hm : (ki * (IntV.ofNat 64 : I32)).toInt = ((kk * 64 : Nat) : Int) :=
      IntV.I32.mul_nat hki (IntV.I32.toInt_ofNat (by norm_num))
        (by have h := kk_mul_lt hd hkk; omega)
    have hxy : kk * 64 ≤ d.K := by have := kk_bound hd hkk x.1.isLt; omega
    exact IntV.I32.sub_nat (IntV.I32.toInt_ofNat hK) hm hxy hK
  show IntV.lt ((offs_k_for_mask.col 256).get x)
    ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32)) = true
  rw [IntV.lt_eq]
  rw [show (offs_k_for_mask.col 256).get x = offs_k_for_mask.get1 x.1 from rfl, hkm x.1, hb]
  apply decide_eq_true
  have h := kk_bound hd hkk x.1.isLt
  have h2 : (x.1.val : Int) < ((d.K - kk * 64 : Nat) : Int) := by
    have hle : x.1.val < d.K - kk * 64 := by omega
    exact_mod_cast hle
  exact h2

/-- Every lane of the `c` mask holds. -/
theorem cmask_get {d : Inst} (hd : d.ok) {offs_cm : Tensor [128] I32}
    {offs_cn : Tensor [256] I32} {T : Nat} (hT : T < nT d)
    (hcm : ∀ i : Fin 128, (offs_cm.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int))
    (hcn : ∀ j : Fin 256, (offs_cn.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int))
    (x : Idx [128, 256]) :
    (((offs_cm.col 256).ltS (IntV.ofNat d.M : I32)).and
      ((offs_cn.row 128).ltS (IntV.ofNat d.N : I32))).get x = true := by
  show (IntV.lt ((offs_cm.col 256).get x) (IntV.ofNat d.M : I32) &&
    IntV.lt ((offs_cn.row 128).get x) (IntV.ofNat d.N : I32)) = true
  rw [Bool.and_eq_true]
  constructor
  · rw [IntV.lt_eq,
      show (offs_cm.col 256).get x = offs_cm.get1 x.1 from rfl, hcm x.1,
      IntV.I32.toInt_ofNat (by have := hd.M_le; omega)]
    apply decide_eq_true
    have h := pm_mul_bound hd hT x.1.isLt
    exact_mod_cast h
  · rw [IntV.lt_eq,
      show (offs_cn.row 128).get x = offs_cn.get1 x.2.1 from rfl, hcn x.2.1,
      IntV.I32.toInt_ofNat (by have := hd.N_le; omega)]
    apply decide_eq_true
    have h := pn_mul_bound hd hT x.2.1.isLt
    exact_mod_cast h

/-! ### The hints of lines 334–335 -/

/-- Offsets `B + i` are contiguous along the block (the hint `tl.max_contiguous`). -/
theorem contigArange {n C B : Nat} {x : Tensor [n] I32}
    (hB : ∀ i : Fin n, (x.get1 i).toInt = ((B + i.val : Nat) : Int)) :
    ContigAlong x 0 C := by
  intro i j _
  obtain ⟨i1, iu⟩ := i
  obtain ⟨j1, ju⟩ := j
  cases iu
  cases ju
  show (x.get1 j1).toInt = (x.get1 i1).toInt + ((j1.val : Int) - (i1.val : Int)) * 1
  rw [hB i1, hB j1]
  push_cast
  ring

/-- Offsets `B + i`, with `C ∣ B` and `n ≤ C`, satisfy the hint `tl.multiple_of`. -/
theorem firstsArange {n C B : Nat} {x : Tensor [n] I32} (hCn : n ≤ C)
    (hB : ∀ i : Fin n, (x.get1 i).toInt = ((B + i.val : Nat) : Int))
    (hdvd : (C : Int) ∣ (B : Int)) : FirstsDiv x 0 C C := by
  intro i hi
  show (C : Int) ∣ (x.get1 i.1).toInt
  have hi' : i.1.val % C = 0 := hi
  rw [hB i.1]
  have hlt : i.1.val < C := lt_of_lt_of_le i.1.isLt hCn
  have h0 : i.1.val = 0 := Nat.eq_zero_of_dvd_of_lt (Nat.dvd_of_mod_eq_zero hi') hlt
  simp only [h0, Nat.add_zero]
  exact hdvd

/-- The lane values of the inner loop's `ki * 64 + tl.arange 0 64`. -/
theorem offsk_get1 {d : Inst} (hd : d.ok) {ki : I32} {t : Nat}
    (hki : ki.toInt = (t : Int)) (ht : t < steps d.K) (j : Fin 64) :
    ((ki * (IntV.ofNat 64 : I32) + tl.arange 0 64).get1 j).toInt = ((t * 64 + j.val : Nat) : Int) := by
  have hstep : t * 64 < d.K := kk_mul_lt hd ht
  have hmul : (ki * (IntV.ofNat 64 : I32)).toInt = ((t * 64 : Nat) : Int) :=
    IntV.I32.mul_nat hki (IntV.I32.toInt_ofNat (by norm_num)) (by
      have := hd.K_le
      omega)
  have harange : ((tl.arange 0 64).get1 j).toInt = (j.val : Int) :=
    arange_get1_toInt (by norm_num) j
  have hS : IntV.SFits 32 (((t * 64 : Nat) : Int) + (j.val : Int)) := by
    constructor
    · have h1 : (0 : Int) ≤ ((t * 64 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
      have h2 : (0 : Int) ≤ (j.val : Int) := by exact_mod_cast Nat.zero_le j.val
      omega
    · have h1 : t * 64 < 2 ^ 13 := by have := hd.K_le; omega
      have h1' : ((t * 64 : Nat) : Int) < 2 ^ 13 := by exact_mod_cast h1
      have h2 : (j.val : Int) < 64 := by exact_mod_cast j.isLt
      have h3 : (2 : Int) ^ 31 = 2147483648 := by norm_num
      omega
  show (IntV.toInt ((ki * (IntV.ofNat 64 : I32)) + (tl.arange 0 64).get1 j))
      = ((t * 64 + j.val : Nat) : Int)
  rw [IntV.toInt_add rfl rfl (by rw [hmul, harange]; exact hS), hmul, harange]
  push_cast
  ring

/-- The lane values of `pid * c + tl.arange 0 n`, for a `pid` that represents `pm`. -/
theorem start_arange_get1 {pm : Nat} {pid : I32} {c n : Nat}
    (hp : pid.toInt = (pm : Int)) (hc : pm * c + n < 2 ^ 31) (hn : n < 2 ^ 31) (i : Fin n) :
    (((pid * (IntV.ofNat c : I32)) + tl.arange 0 n).get1 i).toInt = ((pm * c + i.val : Nat) : Int) := by
  have hmul : (pid * (IntV.ofNat c : I32)).toInt = ((pm * c : Nat) : Int) := by
    rcases Nat.eq_zero_or_pos pm with hpm | hpm
    · subst hpm
      have h0 : pid.toInt = 0 := by rw [hp]; rfl
      rw [IntV.toInt_mul rfl rfl (by rw [h0, Int.zero_mul]; norm_num [IntV.SFits]), h0]
      simp
    · have hpc : pm * c < 2 ^ 31 := by omega
      have hc31 : c < 2 ^ 31 := by
        have : c ≤ pm * c := Nat.le_mul_of_pos_left c hpm
        omega
      exact IntV.I32.mul_nat hp (IntV.I32.toInt_ofNat hc31) hpc
  have harange : ((tl.arange 0 n).get1 i).toInt = (i.val : Int) := arange_get1_toInt hn i
  have hS : IntV.SFits 32 (((pm * c : Nat) : Int) + (i.val : Int)) := by
    constructor
    · have h1 : (0 : Int) ≤ ((pm * c : Nat) : Int) := by exact_mod_cast Nat.zero_le _
      have h2 : (0 : Int) ≤ (i.val : Int) := by exact_mod_cast Nat.zero_le i.val
      omega
    · have h1 : pm * c + i.val < 2 ^ 31 := by have hi := i.isLt; omega
      have h1' : ((pm * c + i.val : Nat) : Int) < 2 ^ 31 := by exact_mod_cast h1
      have h3 : (2 : Int) ^ 31 = 2147483648 := by norm_num
      push_cast at h1' ⊢
      omega
  show (IntV.toInt ((pid * (IntV.ofNat c : I32)) + (tl.arange 0 n).get1 i))
      = ((pm * c + i.val : Nat) : Int)
  rw [IntV.toInt_add rfl rfl (by rw [hmul, harange]; exact hS), hmul, harange]
  push_cast
  ring

/-- The same after the bounds mask `tl.where` of `trip` (every lane of a full tile is below `M`). -/
theorem start_arange_where_get1 {pm : Nat} {pid : I32} {c n M : Nat}
    (hp : pid.toInt = (pm : Int)) (hlt : pm * c + n ≤ M)
    (hc : pm * c + n < 2 ^ 31) (hn : n < 2 ^ 31) (hM : M < 2 ^ 31) (i : Fin n) :
    ((tl.where (((pid * (IntV.ofNat c : I32)) + tl.arange 0 n).ltS (IntV.ofNat M : I32))
        ((pid * (IntV.ofNat c : I32)) + tl.arange 0 n) (tl.full [n] (IntV.ofInt 0 : I32))).get1 i).toInt
      = ((pm * c + i.val : Nat) : Int) := by
  have hx : (((pid * (IntV.ofNat c : I32)) + tl.arange 0 n).get1 i).toInt
      = ((pm * c + i.val : Nat) : Int) := start_arange_get1 hp hc hn i
  have htrue : (((pid * (IntV.ofNat c : I32)) + tl.arange 0 n).ltS (IntV.ofNat M : I32)).get1 i
      = true := by
    rw [ltS_get, IntV.I32.toInt_ofNat hM, hx]
    apply decide_eq_true
    have h1 : pm * c + i.val < M := by have hi := i.isLt; omega
    exact_mod_cast h1
  simp only [tl.where, tl.full]
  rw [where_of_true i htrue]
  exact hx

end Claims.PersistentMatmul
