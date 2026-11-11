import Claim
import TVA.Proof
import Proof.Basic
import Proof.Arith
import Proof.Off
import Proof.ScaleOff
import Proof.ScaleIdx
import Proof.ScaledVal
import Proof.Lane

/-!
# Facts about the loaded tiles, for `body_safe`

The element bounds and the per-lane read-back of the four `tl.desc_load`s of the loop's body: a
loaded tile's lane holds the element (or the scale byte) the kernel's value functions read.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## Index arithmetic -/

theorem sIdx_shift {K r t k : Nat} (hk : k < 128) :
    sIdx K r (128 * t + 32 * (k / 32)) = sIdx K r (128 * t + k) := by
  unfold sIdx
  have h1 : (128 * t + 32 * (k / 32)) / 128 = t := by omega
  have h2 : (128 * t + k) / 128 = t := by omega
  have h3 : (128 * t + 32 * (k / 32)) % 128 / 32 = k / 32 := by omega
  have h4 : (128 * t + k) % 128 / 32 = k / 32 := by omega
  rw [h1, h2, h3, h4]

/-- The `a`-scale of element `(i, 128 t + k)` is the one of `(i, 128 t + 32 (k / 32))`. -/
theorem sa_shift {d : Inst} {m : Mem} {i k t : Nat} (hk : k < 128) :
    sa d m i (128 * t + 32 * (k / 32)) = sa d m i (128 * t + k) := by
  unfold sa
  rw [sIdx_shift (K := d.K) (r := i) (t := t) (k := k) hk]

/-- The `b`-scale of element `(j, 128 t + k)` is the one of `(j, 128 t + 32 (k / 32))`. -/
theorem sb_shift {d : Inst} {m : Mem} {j k t : Nat} (hk : k < 128) :
    sb d m j (128 * t + 32 * (k / 32)) = sb d m j (128 * t + k) := by
  unfold sb
  rw [sIdx_shift (K := d.K) (r := j) (t := t) (k := k) hk]

/-- `K / 2 = 64 * (K / 128)`. -/
theorem K_div2 {d : Inst} (hd : d.ok) : d.K / 2 = 64 * (d.K / 128) := by
  conv_lhs => rw [K_div d hd]
  rw [show 128 * (d.K / 128) = 2 * (64 * (d.K / 128)) by ring]
  exact Nat.mul_div_cancel_left (64 * (d.K / 128)) (n := 2) (by norm_num)

/-- The `b` tile's column of step `t` is inside the half-width row of `b`. -/
theorem bcol_lt {d : Inst} (hd : d.ok) {t j : Nat} (ht : t < d.K / 128) (hj : j < 64) :
    64 * t + j < d.K / 2 := by
  rw [K_div2 hd]
  omega

/-- The scale row of a row of `b` in range is below `N / 128`. -/
theorem bRow_lt {d : Inst} (hd : d.ok) {pn n : Nat} (h : pn * 256 + n < d.N) :
    2 * pn + n / 128 < d.N / 128 := by
  have hN := N_div d hd
  have h1 : pn * 256 + n < 128 * (d.N / 128) := by rw [← hN]; exact h
  have h2 : (pn * 256 + n) / 128 < d.N / 128 := Nat.div_lt_of_lt_mul h1
  have h3 : (pn * 256 + n) / 128 = 2 * pn + n / 128 := by
    rw [show pn * 256 = (2 * pn) * 128 by ring]
    omega
  rw [h3] at h2
  exact h2

/-- `getD2` of an in-range position is `get2`. -/
theorem getD2_eq {R C : Nat} {α : Type} [Inhabited α] (x : Tensor [R, C] α) {r c : Nat}
    (hr : r < R) (hc : c < C) : x.getD2 r c = x.get2 ⟨r, hr⟩ ⟨c, hc⟩ := by
  rw [Tensor.getD2, dite_eq_left ⟨hr, hc⟩]

/-! ## Bits of the read-backs -/

theorem toBits_rd8 {m : Mem} {b : Buf .f8e4nv} {i : Nat} :
    DType.toBits .f8e4nv (rd8 m b i) = rdBits m b i := by
  unfold rd8 rdBits
  cases h : m.read .f8e4nv (b.ptr (i : Int)).addr <;> rfl

/-- The bits of a byte read are the bits `rdBits` reads. -/
theorem toBits_rdU8 {m : Mem} {b : Buf .u8} {i : Nat} :
    DType.toBits .u8 (rdU8 m b i) = rdBits m b i := by
  unfold rdU8 rdBits
  cases h : m.read .u8 (b.ptr (i : Int)).addr <;> rfl

/-- An `int32` plus a natural that keeps the sum inside `int32`. -/
theorem addN_toInt {a : I32} {x N : Nat} (ha : a.toInt = (x : Int)) (h : x + N < 2 ^ 31) :
    (a + (IntV.ofNat N : I32)).toInt = ((x + N : Nat) : Int) := by
  refine IntV.I32.add_nat ha ?_ h
  exact IntV.I32.toInt_ofNat (by omega)

/-! ## The loaded elements' bounds -/

/-- The `a` block's elements are inside `a`. -/
theorem aDesc_elemAt_lt {d : Inst} (hd : d.ok) {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) (l : Idx [128, 128]) :
    ((aDesc d).elemAt
        (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
      d.M * d.K := by
  rw [aDesc_elemAt hd pm t l hpm ht]
  have hl0 : l.coord 0 < 128 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 128 := l.coord_lt (by simp)
  have hA : pm * 128 + l.coord 0 < d.M := off_mul_add_lt_of_lt_div hpm hl0
  have hB : 128 * t + l.coord 1 < d.K := by
    have h := off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := d.K) (l := l.coord 1) ht hl1
    omega
  exact off_mul_add_lt_mul hA hB

/-- The element of `b` that lane `l` of a `256 x 64` block addresses (`mixed`). -/
theorem bDesc_elemAt_mixed' {d : Inst} (hm : d.mixed = true) (pn t : Nat) (l : Idx [256, 64]) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat =
      (pn * 256 + l.coord 0) * (d.K / 2) + (64 * t + l.coord 1) := by
  have hep : epbB true = 2 := by simp [epbB]
  have key : ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l))
      = (((pn * 256 + l.coord 0) * (d.K / 2) + (64 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bDesc, hm, hep, List.length_cons,
      List.length_nil, List.range_succ, List.range_zero, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ,
      List.nil_append, List.cons_append]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- A lane of the `256 x 64` block of `b` in bounds has its row and column below the tensor's (`mixed`). -/
theorem bDesc_inbAt_bnds_mixed {d : Inst} (hm : d.mixed = true) {pn t : Nat} (l : Idx [256, 64])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true) :
    pn * 256 + l.coord 0 < d.N ∧ 64 * t + l.coord 1 < d.K / 2 := by
  have hep : epbB true = 2 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff] at hl
  have h0 := hl 0 (by simp [TensorDesc.rank, bDesc, hm, hep])
  have h1 := hl 1 (by simp [TensorDesc.rank, bDesc, hm, hep])
  simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ] at h0 h1
  push_cast at h0 h1
  omega

/-- A lane of the `256 x 64` block of `b` with row and column below the tensor's is in bounds (`mixed`). -/
theorem bDesc_inbAt_of_lt_mixed {d : Inst} (hm : d.mixed = true) {pn t : Nat} {l : Idx [256, 64]}
    (hrow : pn * 256 + l.coord 0 < d.N) (hcol : 64 * t + l.coord 1 < d.K / 2) :
    (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true := by
  have hep : epbB true = 2 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, bDesc, hm, hep] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> push_cast <;> omega

/-- An in-bounds lane of the `256 x 64` block of `b` addresses an element below `N * K / 2` (`mixed`). -/
theorem bDesc_elemAt_lt_mixed_inb {d : Inst} (hm : d.mixed = true) {pn t : Nat}
    (l : Idx [256, 64])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat <
      d.N * (d.K / 2) := by
  rw [bDesc_elemAt_mixed' hm pn t l]
  obtain ⟨hA, hB⟩ := bDesc_inbAt_bnds_mixed hm l hl
  exact off_mul_add_lt_mul hA hB

/-- The element of `b` that lane `l` of a `256 x 128` block addresses (`mxfp8`). -/
theorem bDesc_elemAt_mx' {d : Inst} (hm : d.mixed = false) (pn t : Nat) (l : Idx [256, 128]) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat =
      (pn * 256 + l.coord 0) * d.K + (128 * t + l.coord 1) := by
  have hep : epbB false = 1 := by simp [epbB]
  have key : ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l))
      = (((pn * 256 + l.coord 0) * d.K + (128 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bDesc, hm, hep, List.length_cons,
      List.length_nil, List.range_succ, List.range_zero, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ,
      List.nil_append, List.cons_append, Nat.div_one]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- A lane of the `256 x 128` block of `b` in bounds has its row and column below the tensor's (`mxfp8`). -/
theorem bDesc_inbAt_bnds_mx {d : Inst} (hm : d.mixed = false) {pn t : Nat} (l : Idx [256, 128])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true) :
    pn * 256 + l.coord 0 < d.N ∧ 128 * t + l.coord 1 < d.K := by
  have hep : epbB false = 1 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff] at hl
  have h0 := hl 0 (by simp [TensorDesc.rank, bDesc, hm, hep])
  have h1 := hl 1 (by simp [TensorDesc.rank, bDesc, hm, hep])
  simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ] at h0 h1
  push_cast at h0 h1
  omega

/-- A lane of the `256 x 128` block of `b` with row and column below the tensor's is in bounds (`mxfp8`). -/
theorem bDesc_inbAt_of_lt_mx {d : Inst} (hm : d.mixed = false) {pn t : Nat} {l : Idx [256, 128]}
    (hrow : pn * 256 + l.coord 0 < d.N) (hcol : 128 * t + l.coord 1 < d.K) :
    (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
  have hep : epbB false = 1 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, bDesc, hm, hep] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ,
      Nat.div_one]
    constructor <;> push_cast <;> omega

/-- An in-bounds lane of the `256 x 128` block of `b` addresses an element below `N * K` (`mxfp8`). -/
theorem bDesc_elemAt_lt_mx_inb {d : Inst} (hm : d.mixed = false) {pn t : Nat}
    (l : Idx [256, 128])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
      d.N * d.K := by
  rw [bDesc_elemAt_mx' hm pn t l]
  obtain ⟨hA, hB⟩ := bDesc_inbAt_bnds_mx hm l hl
  exact off_mul_add_lt_mul hA hB

/-- Every lane of `a_scale`'s block at a tile of the grid addresses an element inside the tensor. -/
theorem aScaleDesc_elemAt_lt {d : Inst} (hd : d.ok) {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) (l : Idx [1, 1, 1, 2, 256]) :
    ((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat <
      d.M / 128 * (d.K / 128) * 512 := by
  rw [aScaleDesc_elemAt hd pm t l hpm ht]
  have hK : 0 < d.K / 128 := scaleOff_Kpos hd
  have c3 : l.coord 3 < 2 := l.coord_lt (k := 3) (by simp)
  have c4 : l.coord 4 < 256 := l.coord_lt (k := 4) (by simp)
  set A := (d.K / 128) * 512 with hA
  have h1 : t * 512 + l.coord 3 * 256 + l.coord 4 < A := by
    rw [hA]; omega
  have h2 : pm * A + (t * 512 + l.coord 3 * 256 + l.coord 4) < (pm + 1) * A := by
    rw [Nat.succ_mul]
    exact Nat.add_lt_add_left h1 (pm * A)
  have h3 : (pm + 1) * A ≤ d.M / 128 * A := Nat.mul_le_mul_right A (by omega)
  have h4 : d.M / 128 * A = d.M / 128 * (d.K / 128) * 512 := by
    rw [hA, Nat.mul_assoc]
  have heq : pm * (d.K / 128) * 512 + t * 512 + l.coord 3 * 256 + l.coord 4
      = pm * A + (t * 512 + l.coord 3 * 256 + l.coord 4) := by
    rw [hA, Nat.mul_assoc]
    omega
  calc pm * (d.K / 128) * 512 + t * 512 + l.coord 3 * 256 + l.coord 4
      = pm * A + (t * 512 + l.coord 3 * 256 + l.coord 4) := heq
    _ < (pm + 1) * A := h2
    _ ≤ d.M / 128 * A := h3
    _ = d.M / 128 * (d.K / 128) * 512 := h4

/-- The element of `b_scale` that lane `i` of its block addresses. -/
theorem bScaleDesc_elemAt' {d : Inst} (pn t : Nat) (i : Idx [1, 2, 1, 2, 256]) :
    ((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i)).toNat =
      (2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4 := by
  have c0 : i.coord 0 < 1 := i.coord_lt (k := 0) (by simp)
  have c2 : i.coord 2 < 1 := i.coord_lt (k := 2) (by simp)
  have e0 : i.coord 0 = 0 := by omega
  have e2 : i.coord 2 = 0 := by omega
  have h : (bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i)
      = (((2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 +
          i.coord 4 : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bScaleDesc, List.length_cons, List.length_nil,
      List.range_succ, List.range_zero, List.map_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, List.sum_append, List.nil_append, TensorDesc.coord,
      List.getD_cons_zero, List.getD_cons_succ, e0, e2]
    push_cast
    ring
  rw [h, Int.toNat_natCast]

/-- A lane of `b_scale`'s block in bounds has its scale row and group below the tensor's. -/
theorem bScaleDesc_inbAt_bnds {d : Inst} {pn t : Nat} (i : Idx [1, 2, 1, 2, 256])
    (hl : (bScaleDesc d).inbAt
      (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i) = true) :
    2 * pn + i.coord 1 < d.N / 128 ∧ t + i.coord 2 < d.K / 128 := by
  rw [TensorDesc.inbAt, decide_eq_true_iff] at hl
  have h1 := hl 1 (by simp [TensorDesc.rank, bScaleDesc])
  have h2 := hl 2 (by simp [TensorDesc.rank, bScaleDesc])
  simp only [TensorDesc.coord, bScaleDesc, List.getD_cons_zero, List.getD_cons_succ] at h1 h2
  push_cast at h1 h2
  omega

/-- An in-bounds lane of `b_scale`'s block addresses an element inside the tensor. -/
theorem bScaleDesc_elemAt_lt_inb {d : Inst} (_hd : d.ok) {pn t : Nat}
    (i : Idx [1, 2, 1, 2, 256])
    (hl : (bScaleDesc d).inbAt
      (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i) = true) :
    ((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i)).toNat <
      d.N / 128 * (d.K / 128) * 512 := by
  rw [bScaleDesc_elemAt' pn t i]
  obtain ⟨hA, hB⟩ := bScaleDesc_inbAt_bnds i hl
  have c3 : i.coord 3 < 2 := i.coord_lt (k := 3) (by simp)
  have c4 : i.coord 4 < 256 := i.coord_lt (k := 4) (by simp)
  set A := (d.K / 128) * 512 with hA'
  have h1 : t * 512 + i.coord 3 * 256 + i.coord 4 < A := by rw [hA']; omega
  have h2 : (2 * pn + i.coord 1) * A + (t * 512 + i.coord 3 * 256 + i.coord 4) <
      (2 * pn + i.coord 1 + 1) * A := by
    have hs : (2 * pn + i.coord 1 + 1) * A = (2 * pn + i.coord 1) * A + A := by
      rw [Nat.add_mul, one_mul]
    rw [hs]
    exact Nat.add_lt_add_left h1 _
  have h3 : (2 * pn + i.coord 1 + 1) * A ≤ d.N / 128 * A := Nat.mul_le_mul_right A (by omega)
  have h4 : d.N / 128 * A = d.N / 128 * (d.K / 128) * 512 := by rw [hA', Nat.mul_assoc]
  have heq : (2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4
      = (2 * pn + i.coord 1) * A + (t * 512 + i.coord 3 * 256 + i.coord 4) := by
    rw [hA', Nat.mul_assoc]; omega
  calc (2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4
      = (2 * pn + i.coord 1) * A + (t * 512 + i.coord 3 * 256 + i.coord 4) := heq
    _ < (2 * pn + i.coord 1 + 1) * A := h2
    _ ≤ d.N / 128 * A := h3
    _ = d.N / 128 * (d.K / 128) * 512 := h4

/-- A lane of `b_scale`'s block with row and group below the tensor's is in bounds. -/
theorem bScaleDesc_inbAt_of_lt {d : Inst} {pn t : Nat} {i : Idx [1, 2, 1, 2, 256]}
    (hrow : 2 * pn + i.coord 1 < d.N / 128) (hcol : t + i.coord 2 < d.K / 128) :
    (bScaleDesc d).inbAt
      (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i) = true := by
  have c0 : i.coord 0 < 1 := i.coord_lt (k := 0) (by simp)
  have c2 : i.coord 2 < 1 := i.coord_lt (k := 2) (by simp)
  have c3 : i.coord 3 < 2 := i.coord_lt (k := 3) (by simp)
  have c4 : i.coord 4 < 256 := i.coord_lt (k := 4) (by simp)
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk5 : k < 5 := by simpa [bScaleDesc, TensorDesc.rank] using hk
  interval_cases k
  · simp only [TensorDesc.coord, bScaleDesc, List.getD_cons_zero]
    omega
  · simp only [TensorDesc.coord, bScaleDesc, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> omega
  · simp only [TensorDesc.coord, bScaleDesc, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> omega
  · simp only [TensorDesc.coord, bScaleDesc, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> omega
  · simp only [TensorDesc.coord, bScaleDesc, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> omega

/-- The lane of `b_scale`'s block holding scale `(n, g)` of the tile is in bounds. -/
theorem bScaleDesc_inbAt_ofPos {d : Inst} {pn t : Nat} {n g : Nat} (hn : n < 256) (hg : g < 4)
    (hr : 2 * pn + n / 128 < d.N / 128) (ht : t < d.K / 128) :
    (bScaleDesc d).inbAt
      (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0]
        (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg))) = true := by
  have hp : bPos n g < 1024 := by have := bPos_lt hn hg; simpa using this
  have hb := bU_lt hn hg
  refine bScaleDesc_inbAt_of_lt (i := Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg))
    ?_ ?_
  · rw [eB1 (bPos n g) hp]
    show 2 * pn + bPos n g / 512 < d.N / 128
    have h1 : bPos n g / 512 = n / 128 := by unfold bPos; omega
    rw [h1]; exact hr
  · rw [eB2 (bPos n g) hp]
    show t + 0 < d.K / 128
    omega

/-! ## The read-back of the loaded tiles -/

/-- **The `a` tile's lanes are `xa`.** -/
theorem aTile_read {d : Inst} (hd : d.ok) {m : Mem} {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) {av : Tensor [128, 128] F8E4}
    (hav : ∀ i : Idx [128, 128],
      (aDesc d).inbAt
        (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] i) = true →
      m.read .f8e4nv (d.a.ptr (((aDesc d).elemAt
        (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] i)).toNat : Int)).addr
        = some (av.get i)) :
    ∀ (r c : Fin 128), av.get2 r c = xa d m (pm * 128 + (r : Nat)) (128 * t + (c : Nat)) := by
  intro r c
  have hinb := aDesc_inbAt hd pm t ((r, c, ()) : Idx [128, 128]) hpm ht
  have hread := hav ((r, c, ()) : Idx [128, 128]) hinb
  have helem : ((aDesc d).elemAt
        (TensorDesc.coord (bs := [128, 128]) [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)]
          ((r, c, ()) : Idx [128, 128]))).toNat =
      (pm * 128 + (r : Nat)) * d.K + (128 * t + (c : Nat)) := by
    simpa only [Idx.coord] using
      aDesc_elemAt hd pm t ((r, c, ()) : Idx [128, 128]) hpm ht
  rw [helem] at hread
  simp only [xa, rd8]
  rw [hread]
  rfl

/-- **The `b` tile's lanes are `xbBits`** (`"mixed"`), for a row inside `b`. -/
theorem bTile_read_mixed {d : Inst} (hd : d.ok) (hm : d.mixed = true) {m : Mem} {pn t : Nat}
    (ht : t < d.K / 128) {bv : Tensor [256, 64] ((bDT d.mixed).Val)}
    (hbv : ∀ i : Idx [256, 64],
      (bDesc d).inbAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] i) = true →
      m.read (bDT d.mixed) (d.b.ptr (((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] i)).toNat : Int)).addr
        = some (bv.get i)) :
    ∀ (n : Fin 256), pn * 256 + (n : Nat) < d.N → ∀ j : Fin 64,
      DType.toBits (bDT d.mixed) (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat))) := by
  intro n hrow j
  have hrow' : pn * 256 + Idx.coord (s := [256, 64]) ((n, j, ()) : Idx [256, 64]) 0 < d.N := by
    simpa only [Idx.coord] using hrow
  have hcol : 64 * t + Idx.coord (s := [256, 64]) ((n, j, ()) : Idx [256, 64]) 1 < d.K / 2 := by
    simpa only [Idx.coord] using bcol_lt hd ht j.isLt
  have hinb := bDesc_inbAt_of_lt_mixed (d := d) hm (l := ((n, j, ()) : Idx [256, 64])) hrow' hcol
  have hread := hbv ((n, j, ()) : Idx [256, 64]) hinb
  have helem : ((bDesc d).elemAt
        (TensorDesc.coord (bs := [256, 64]) [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)]
          ((n, j, ()) : Idx [256, 64]))).toNat =
      (pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat)) := by
    simpa only [Idx.coord] using
      bDesc_elemAt_mixed' (d := d) hm pn t ((n, j, ()) : Idx [256, 64])
  rw [helem] at hread
  unfold rdBits
  rw [hread]
  rfl

/-- **The `b` tile's lanes are `xbBits`** (`"mxfp8"`), for a row inside `b`. -/
theorem bTile_read_mx {d : Inst} (_hd : d.ok) (hm : d.mixed = false) {m : Mem} {pn t : Nat}
    (ht : t < d.K / 128) {bv : Tensor [256, 128] ((bDT d.mixed).Val)}
    (hbv : ∀ i : Idx [256, 128],
      (bDesc d).inbAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] i) = true →
      m.read (bDT d.mixed) (d.b.ptr (((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] i)).toNat : Int)).addr
        = some (bv.get i)) :
    ∀ (n : Fin 256), pn * 256 + (n : Nat) < d.N → ∀ j : Fin 128,
      DType.toBits (bDT d.mixed) (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat))) := by
  intro n hrow j
  have hrow' : pn * 256 + Idx.coord (s := [256, 128]) ((n, j, ()) : Idx [256, 128]) 0 < d.N := by
    simpa only [Idx.coord] using hrow
  have hcol : 128 * t + Idx.coord (s := [256, 128]) ((n, j, ()) : Idx [256, 128]) 1 < d.K := by
    rw [Nat.mul_comm 128 t]
    simpa only [Idx.coord] using
      off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := d.K) (l := (j : Nat)) ht j.isLt
  have hinb := bDesc_inbAt_of_lt_mx (d := d) hm (l := ((n, j, ()) : Idx [256, 128])) hrow' hcol
  have hread := hbv ((n, j, ()) : Idx [256, 128]) hinb
  have helem : ((bDesc d).elemAt
        (TensorDesc.coord (bs := [256, 128]) [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)]
          ((n, j, ()) : Idx [256, 128]))).toNat =
      (pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat)) := by
    simpa only [Idx.coord] using
      bDesc_elemAt_mx' (d := d) hm pn t ((n, j, ()) : Idx [256, 128])
  rw [helem] at hread
  unfold rdBits
  rw [hread]
  rfl

/-! ## The read-back of the loaded scale tiles -/

/-- The `a` scale tile's lanes are in bounds wherever the descriptor says so. -/
theorem aScale_read {d : Inst} (hd : d.ok) {m : Mem} {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) {av : Tensor [1, 1, 1, 2, 256] U8}
    (hav : ∀ l : Idx [1, 1, 1, 2, 256],
      (aScaleDesc d).inbAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.a_scale.ptr (((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (av.get l)) :
    ∀ l : Idx [1, 1, 1, 2, 256],
      av.get l = rdU8 m d.a_scale
        ((aScaleDesc d).elemAt
          (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat := by
  intro l
  have hinb := aScaleDesc_inbAt hd pm t l hpm ht
  have hread := hav l hinb
  simp only [rdU8]
  rw [hread]
  rfl

/-- **`a`'s scale of a lane** from a guarded read. -/
theorem aScale_val_at {d : Inst} (hd : d.ok) {m : Mem} {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) {av : Tensor [1, 1, 1, 2, 256] U8}
    (hav : ∀ l : Idx [1, 1, 1, 2, 256],
      (aScaleDesc d).inbAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.a_scale.ptr (((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (av.get l)) :
    ∀ (r g : Nat) (hr : r < 128) (hg : g < 4),
      DType.toBits .u8 (av.get (Idx.ofPos [1, 1, 1, 2, 256] (aU r g) (aU_lt hr hg))) =
        DType.toBits .u8 (sa d m (pm * 128 + r) (128 * t + 32 * g)) := by
  intro r g hr hg
  exact aScale_val hd (aScale_read hd hpm ht hav) r g hr hg

/-- **`b`'s scale of a lane** from a guarded read, for a row inside `b`. -/
theorem bScale_val_at {d : Inst} (hd : d.ok) {m : Mem} {pn t : Nat} (ht : t < d.K / 128)
    {bv : Tensor [1, 2, 1, 2, 256] U8}
    (hbv : ∀ l : Idx [1, 2, 1, 2, 256],
      (bScaleDesc d).inbAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.b_scale.ptr (((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (bv.get l)) :
    ∀ (n g : Nat) (hn : n < 256) (hg : g < 4) (_hr : 2 * pn + n / 128 < d.N / 128),
      DType.toBits .u8 (bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg))) =
        DType.toBits .u8 (sb d m (pn * 256 + n) (128 * t + 32 * g)) := by
  intro n g hn hg hr
  have hinb := bScaleDesc_inbAt_ofPos (d := d) (pn := pn) (t := t) hn hg hr ht
  have hread := hbv _ hinb
  have hval : bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg)) =
      rdU8 m d.b_scale ((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0]
          (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg)))).toNat := by
    simp only [rdU8]
    rw [hread]
    rfl
  rw [hval, bScale_elemAt (d := d) pn t hn hg]
  show DType.toBits .u8 (rdU8 m d.b_scale
      ((2 * pn + n / 128) * (d.K / 128) * 512 + t * 512 + bU n g)) =
    DType.toBits .u8 (rdU8 m d.b_scale (sIdx d.K (pn * 256 + n) (128 * t + 32 * g)))
  rw [sIdx_b hd pn t n g hn hg]

/-- **The `"mixed"` `lhs` scales of a lane**, as `sa`'s bytes. -/
theorem scalesL_at {d : Inst} (hd : d.ok) {m : Mem} {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) {sc : Tensor [128, 4] U8} {av : Tensor [1, 1, 1, 2, 256] U8}
    (hsc : ∀ (r : Fin 128) (g : Fin 4),
      sc.get2 r g = av.get (Idx.ofPos [1, 1, 1, 2, 256] (aU (r : Nat) (g : Nat))
        (aU_lt r.isLt g.isLt)))
    (hav : ∀ l : Idx [1, 1, 1, 2, 256],
      (aScaleDesc d).inbAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.a_scale.ptr (((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (av.get l)) :
    ∀ (i : Fin 128) (k : Fin 128),
      (ScaledCfg.scalesL (ds := .u8) cfgMixed 128 128 sc).get2 i k =
        DType.toBits .u8 (sa d m (pm * 128 + (i : Nat)) (128 * t + (k : Nat))) := by
  intro i k
  have hk : (k : Nat) / 32 < 4 := by have := k.isLt; omega
  have h1 : sc.getD2 (i : Nat) ((k : Nat) / 32) = sc.get2 i ⟨(k : Nat) / 32, hk⟩ :=
    getD2_eq sc i.isLt hk
  refine Eq.trans (b := DType.toBits .u8 (sc.getD2 (i : Nat) ((k : Nat) / 32))) rfl ?_
  rw [h1, hsc i ⟨(k : Nat) / 32, hk⟩]
  refine Eq.trans
    (b := DType.toBits .u8 (sa d m (pm * 128 + (i : Nat)) (128 * t + 32 * ((k : Nat) / 32))))
    (aScale_val_at hd hpm ht hav (i : Nat) ((k : Nat) / 32) i.isLt hk) ?_
  exact congrArg (DType.toBits .u8)
    (sa_shift (d := d) (m := m) (i := pm * 128 + (i : Nat)) (t := t) (k := (k : Nat)) k.isLt)

/-- **The `"mixed"` `rhs` scales of a lane**, as `sb`'s bytes. -/
theorem scalesR_at {d : Inst} (hd : d.ok) {m : Mem} {pn t : Nat} (ht : t < d.K / 128)
    {sc : Tensor [256, 4] U8} {bv : Tensor [1, 2, 1, 2, 256] U8}
    (hsc : ∀ (n : Fin 256) (g : Fin 4),
      sc.get2 n g = bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos (n : Nat) (g : Nat))
        (bPos_lt n.isLt g.isLt)))
    (hbv : ∀ l : Idx [1, 2, 1, 2, 256],
      (bScaleDesc d).inbAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.b_scale.ptr (((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (bv.get l))
    (n : Fin 256) (hr : 2 * pn + (n : Nat) / 128 < d.N / 128) :
    ∀ (k : Fin 128),
      (ScaledCfg.scalesR (ds := .u8) cfgMixed 128 256 sc).get2 k n =
        DType.toBits .u8 (sb d m (pn * 256 + (n : Nat)) (128 * t + (k : Nat))) := by
  intro k
  have hk : (k : Nat) / 32 < 4 := by have := k.isLt; omega
  have h1 : sc.getD2 (n : Nat) ((k : Nat) / 32) = sc.get2 n ⟨(k : Nat) / 32, hk⟩ :=
    getD2_eq sc n.isLt hk
  refine Eq.trans (b := DType.toBits .u8 (sc.getD2 (n : Nat) ((k : Nat) / 32))) rfl ?_
  rw [h1, hsc n ⟨(k : Nat) / 32, hk⟩]
  refine Eq.trans
    (b := DType.toBits .u8 (sb d m (pn * 256 + (n : Nat)) (128 * t + 32 * ((k : Nat) / 32))))
    (bScale_val_at hd ht hbv (n : Nat) ((k : Nat) / 32) n.isLt hk hr) ?_
  exact congrArg (DType.toBits .u8)
    (sb_shift (d := d) (m := m) (j := pn * 256 + (n : Nat)) (t := t) (k := (k : Nat)) k.isLt)

/-- Column `n` of the loaded `b` tile (`mixed`), unpacked, holds the `e2m1` values `xb` of `b`. -/
theorem bTile_mixed_col {d : Inst} (hm : d.mixed = true) {m : Mem} {pn t : Nat}
    {bv : Tensor [256, 64] U8} (n : Fin 256)
    (hbv : ∀ (j : Fin 64), DType.toBits .u8 (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat))))
    (kk : Nat) (hkk : kk < 128) :
    ScaledCfg.rhsBits (d := .u8) cfgMixed bv.trans kk (n : Nat) =
      xbBits d m (pn * 256 + (n : Nat)) (128 * t + kk) := by
  have h1 : kk / 2 < 64 := by omega
  have h2 : (n : Nat) < 256 := n.isLt
  have hdiv : (128 * t + kk) / 2 = 64 * t + kk / 2 := by omega
  have hmod : (128 * t + kk) % 2 = kk % 2 := by omega
  simp [ScaledCfg.rhsBits, cfgMixed, Tensor.getD2, h1, h2, hbv, xbBits, hm, hdiv, hmod]

/-- Column `n` of the loaded `b` tile (`mxfp8`) holds the values `xb` of `b`. -/
theorem bTile_mx_col {d : Inst} (hm : d.mixed = false) {m : Mem} {pn t : Nat}
    {bv : Tensor [256, 128] F8E4} (n : Fin 256)
    (hbv : ∀ (j : Fin 128), (bv.get2 n j).pat =
        rdBits m d.b ((pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat))))
    (kk : Nat) (hkk : kk < 128) :
    ScaledCfg.rhsBits (d := .f8e4nv) cfgMx bv.trans kk (n : Nat) =
      xbBits d m (pn * 256 + (n : Nat)) (128 * t + kk) := by
  have h2 : (n : Nat) < 256 := n.isLt
  have hbit : ∀ x : F8E4, DType.toBits .f8e4nv x = x.pat := fun _ => rfl
  simp [ScaledCfg.rhsBits, cfgMx, Tensor.getD2, hkk, h2, hbit, hbv, xbBits, hm]

end Claims.BlockScaledMxfp8
