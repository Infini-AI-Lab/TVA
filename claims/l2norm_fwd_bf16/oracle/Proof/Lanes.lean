import Claim
import TVA.Proof

/-!
# `l2norm_fwd_bf16`: the geometry of one program

Program `p` of the grid works on the row block `p·BT + [0, BT)` intersected with `[0, T)`, and on
every element `t·D + j` of `x`/`y` with `t` in that block and `j < D`; it also writes `rstd` on
the rows of the block. This file names those sets (`tileRow`, `tileIdx`) and the lane maps the
kernel computes (`gx`, `enx`, `gT`, `enT`), and proves the facts the `Safe` walk needs: the lane
image is the tile, tiles of distinct programs are disjoint, and the tiles sit inside the launch's
footprint.
-/

namespace Claims.L2NormFwd

open TVA

/-! ## Row blocks and tiles -/

/-- The rows program `p` handles: `p·BT ≤ t < p·BT + BT`, and `t < T`. -/
def tileRow (d : Inst) (p : Nat) : Set Nat :=
  {t | t < d.T ∧ p * d.BT ≤ t ∧ t < p * d.BT + d.BT}

/-- The elements `t·D + j` of the tile of program `p` (`t` in its row block, `j < D`). -/
def tileIdx (d : Inst) (p : Nat) : Set Nat :=
  {k | ∃ t ∈ tileRow d p, ∃ j, j < d.D ∧ k = t * d.D + j}

/-- The rows of a block are below `T·D` as element indices (each row has at least one element). -/
theorem tileRow_subset_rowSet {d : Inst} (hd : d.ok) {p : Nat} : tileRow d p ⊆ rowSet d := by
  rintro t ⟨hT, -, -⟩
  show t < d.T * d.D
  have h1 : t + 1 ≤ d.T := hT
  have h2 : t < t + 1 := by omega
  calc t < t + 1 := h2
    _ ≤ d.T := h1
    _ = d.T * 1 := (Nat.mul_one _).symm
    _ ≤ d.T * d.D := Nat.mul_le_mul_left _ hd.d_pos

/-- The elements of a program's tile lie among the first `T·D` elements. -/
theorem tileIdx_subset_rowSet {d : Inst} (_hd : d.ok) {p : Nat} : tileIdx d p ⊆ rowSet d := by
  rintro k ⟨t, ht, j, hj, rfl⟩
  have hT := ht.1
  show t * d.D + j < d.T * d.D
  have h1 : t + 1 ≤ d.T := hT
  have h2 : t * d.D + j < (t + 1) * d.D := by
    rw [Nat.succ_mul]; omega
  exact lt_of_lt_of_le h2 (Nat.mul_le_mul_right _ h1)

/-- A row of the block of program `p` is `t = p·BT + i` with `i < BT`. -/
theorem mem_tileRow_iff {d : Inst} {p t : Nat} :
    t ∈ tileRow d p ↔ t < d.T ∧ ∃ i, i < d.BT ∧ t = p * d.BT + i := by
  constructor
  · rintro ⟨ht, hlo, hhi⟩
    exact ⟨ht, t - p * d.BT, by omega, (Nat.add_sub_cancel' hlo).symm⟩
  · rintro ⟨ht, i, hi, rfl⟩
    exact ⟨ht, by omega, by omega⟩

/-! ## The lane maps -/

/-- The element of `x`/`y` that lane `(i, j)` of a `[BT, BD]` tile addresses. -/
def gx (d : Inst) (p : Nat) (i : Idx [d.BT, d.BD]) : Nat :=
  (p * d.BT + i.1.val) * d.D + i.2.1.val

/-- Whether lane `(i, j)` of a `[BT, BD]` tile is enabled: its row is in `[0, T)` and its column
in `[0, D)`. -/
def enx (d : Inst) (p : Nat) (i : Idx [d.BT, d.BD]) : Bool :=
  decide (p * d.BT + i.1.val < d.T) && decide (i.2.1.val < d.D)

/-- The row of lane `i` of a `[BT]` tile. -/
def gT (d : Inst) (p : Nat) (i : Idx [d.BT]) : Nat := p * d.BT + i.1.val

/-- Whether row `i` of a `[BT]` tile is enabled: it is in `[0, T)`. -/
def enT (d : Inst) (p : Nat) (i : Idx [d.BT]) : Bool := decide (p * d.BT + i.1.val < d.T)

/-- The enabled lanes of the `[BT, BD]` tile address exactly the tile's elements. -/
theorem img_x {d : Inst} (hd : d.ok) (p : Nat) : Buf.img (gx d p) (enx d p) = tileIdx d p := by
  ext k
  constructor
  · rintro ⟨i, hi, hk⟩
    simp only [enx, Bool.and_eq_true, decide_eq_true_eq] at hi
    obtain ⟨h1, h2⟩ := hi
    exact ⟨p * d.BT + i.1.val, ⟨h1, by omega, by have := i.1.isLt; omega⟩,
      i.2.1.val, h2, hk.symm⟩
  · rintro ⟨t, ⟨ht, hlo, hhi⟩, j, hj, hk⟩
    refine ⟨(⟨t - p * d.BT, by omega⟩, ⟨j, lt_of_lt_of_le hj hd.d_bd⟩, ()), ?_, ?_⟩
    · simp only [enx, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨by show p * d.BT + (t - p * d.BT) < d.T; omega, hj⟩
    · show (p * d.BT + (t - p * d.BT)) * d.D + j = k
      rw [hk, show p * d.BT + (t - p * d.BT) = t by omega]

/-- The enabled lanes of the `[BT]` block address exactly the block's rows of `rstd`. -/
theorem img_T (d : Inst) (p : Nat) : Buf.img (gT d p) (enT d p) = tileRow d p := by
  ext t
  constructor
  · rintro ⟨i, hi, hk⟩
    simp only [gT] at hk
    rw [← hk]
    simp only [enT, decide_eq_true_eq] at hi
    exact ⟨hi, by omega, by have := i.1.isLt; omega⟩
  · rintro ⟨ht, hlo, hhi⟩
    refine ⟨(⟨t - p * d.BT, by omega⟩, ()), ?_, ?_⟩
    · simp only [enT, decide_eq_true_eq]
      show p * d.BT + (t - p * d.BT) < d.T
      omega
    · show p * d.BT + (t - p * d.BT) = t
      omega

/-! ## Distinct programs have disjoint tiles -/

/-- Row blocks of distinct programs are disjoint. -/
theorem tileRow_disjoint {d : Inst} (_hBT : 0 < d.BT) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (tileRow d p) (tileRow d q) := by
  rw [Set.disjoint_left]
  intro t ht hq
  obtain ⟨-, hlo, hhi⟩ := ht
  obtain ⟨-, hlo', hhi'⟩ := hq
  rcases lt_or_gt_of_ne hpq with h | h
  · have hmul : (p + 1) * d.BT ≤ q * d.BT := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul] at hmul
    omega
  · have hmul : (q + 1) * d.BT ≤ p * d.BT := Nat.mul_le_mul_right _ h
    rw [Nat.succ_mul] at hmul
    omega

/-- Tiles of distinct programs are disjoint. -/
theorem tileIdx_disjoint {d : Inst} (hd : d.ok) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (tileIdx d p) (tileIdx d q) := by
  rw [Set.disjoint_left]
  intro k hk hk'
  obtain ⟨t, ht, j, hj, rfl⟩ := hk
  obtain ⟨t', ht', j', hj', he⟩ := hk'
  have hD := hd.d_pos
  have ht_eq : t = t' := by
    have hmod : j = j' := by
      have := congrArg (fun x => x % d.D) he
      simpa [Nat.add_mod, Nat.mul_mod_right, Nat.mod_eq_of_lt hj, Nat.mod_eq_of_lt hj'] using this
    have hmul : t * d.D = t' * d.D := by omega
    exact Nat.eq_of_mul_eq_mul_right hD hmul
  subst ht_eq
  obtain ⟨e, he2⟩ := hd.bt_pow2
  exact Set.disjoint_left.1 (tileRow_disjoint (by rw [he2]; exact Nat.two_pow_pos e) hpq) ht ht'

/-! ## The pointers and masks of the tile -/

/-- `x + o_t[:, None] * D + o_d[None, :]`: the `x` pointers of program `p`. -/
def pX (d : Inst) (o_t : Tensor [d.BT] I64) : Tensor [d.BT, d.BD] (Ptr .bf16) :=
  d.x.ptr 0 + o_t.col d.BD * (IntV.ofNat d.D : I64) + (tl.arange 0 d.BD).row d.BT

/-- The same pointers into a `bfloat16` `y`. -/
def pY16 (y : Buf .bf16) (d : Inst) (o_t : Tensor [d.BT] I64) : Tensor [d.BT, d.BD] (Ptr .bf16) :=
  y.ptr 0 + o_t.col d.BD * (IntV.ofNat d.D : I64) + (tl.arange 0 d.BD).row d.BT

/-- The same pointers into a `float32` `y`. -/
def pY32 (y : Buf .f32) (d : Inst) (o_t : Tensor [d.BT] I64) : Tensor [d.BT, d.BD] (Ptr .f32) :=
  y.ptr 0 + o_t.col d.BD * (IntV.ofNat d.D : I64) + (tl.arange 0 d.BD).row d.BT

/-- `rstd + o_t`: the `rstd` pointers of program `p`. -/
def pR (d : Inst) (o_t : Tensor [d.BT] I64) : Tensor [d.BT] (Ptr .f32) :=
  d.rstd.ptr 0 + o_t

/-- `m_t[:, None] & (o_d[None, :] < D)`: the `[BT, BD]` mask of program `p`. -/
def mX (d : Inst) (o_t : Tensor [d.BT] I64) (T64 : I64) : Tensor [d.BT, d.BD] Bool :=
  ((o_t.ltS T64).col d.BD).and (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT)

/-- `o_t < T64`: the `[BT]` mask of program `p`. -/
def mT (d : Inst) (o_t : Tensor [d.BT] I64) (T64 : I64) : Tensor [d.BT] Bool := o_t.ltS T64

/-! ## Lane arithmetic -/

/-- Lane `i` of `tl.arange(0, n)` has value `i`. -/
theorem arange_toInt (n : Nat) (hn : n < 2 ^ 31) (i : Fin n) :
    IntV.toInt ((tl.arange 0 n).get1 i) = (i.val : Int) := by
  have h16 : (i.val : Int) < 2 ^ 31 := by
    have h : i.val < 2 ^ 31 := lt_of_lt_of_le i.isLt hn.le
    exact_mod_cast h
  have hge : -(2 ^ 31 : Int) ≤ (i.val : Int) := by
    have h0 : (0 : Int) ≤ (i.val : Int) := by exact_mod_cast Nat.zero_le i.val
    linarith
  simp only [tl.arange, Nat.sub_zero, Tensor.get1_arange, Nat.cast_zero, Int.zero_add,
    IntV.I32.toInt_ofInt ⟨hge, h16⟩]

/-- The offset block `o_t[:, None] * D`, lane by lane. -/
theorem get_colprod {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64}
    (hD : d.D ≤ 512)
    (hrow : ∀ i : Idx [d.BT], 0 ≤ IntV.toInt (o_t.get i) ∧ IntV.toInt (o_t.get i) < 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (i : Idx [d.BT, d.BD]) :
    IntV.toInt ((o_t.col d.BD * (IntV.ofNat d.D : I64)).get i)
      = ((p * d.BT + i.1.val : Nat) : Int) * (d.D : Int) := by
  have hD63 : (d.D : Int) < 2 ^ 63 := by
    have h : d.D < 2 ^ 63 := lt_of_le_of_lt hD (by norm_num)
    exact_mod_cast h
  have hDcast : (d.D : Int) ≤ 512 := by exact_mod_cast hD
  have hoti : IntV.toInt (o_t.get (i.1, ())) = ((p * d.BT + i.1.val : Nat) : Int) := hot (i.1, ())
  have hDval : IntV.toInt (IntV.ofNat d.D : I64) = (d.D : Int) :=
    IntV.toInt_ofNat (by exact_mod_cast hD63)
  have hsfit : IntV.SFits 64
      (IntV.toInt (o_t.get (i.1, ())) * IntV.toInt (IntV.ofNat d.D : I64)) := by
    obtain ⟨h0, h2⟩ := hrow (i.1, ())
    rw [hoti] at h0 h2
    rw [hoti, hDval]
    have hD0 : (0 : Int) ≤ (d.D : Int) := by exact_mod_cast Nat.zero_le d.D
    refine ⟨?_, ?_⟩
    · have hp := mul_nonneg h0 hD0
      have hnn : -(2 : Int) ^ (64 - 1) ≤ 0 := by norm_num
      linarith
    · have hb := mul_le_mul h2.le hDcast hD0 (by norm_num : (0 : Int) ≤ 2 ^ 51)
      have h63 : (2 : Int) ^ 51 * 512 < (2 : Int) ^ (64 - 1) := by norm_num
      linarith
  have hcol : (o_t.col d.BD).get i = o_t.get (i.1, ()) := rfl
  rw [Tensor.get_muls, hcol, IntV.toInt_mul rfl rfl hsfit, hoti, hDval]

/-- The `[BT, BD]` pointer block of an arbitrary base pointer, lane by lane. -/
theorem get_tile_ptr {d' : DType} {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64}
    (hD : d.D ≤ 512) (hBD : d.BD < 2 ^ 31)
    (hrow : ∀ i : Idx [d.BT], 0 ≤ IntV.toInt (o_t.get i) ∧ IntV.toInt (o_t.get i) < 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (q : Ptr d') (b : Buf d') (hq : q = b.ptr 0) (i : Idx [d.BT, d.BD]) :
    ((q + o_t.col d.BD * (IntV.ofNat d.D : I64) + (tl.arange 0 d.BD).row d.BT).get i)
      = b.ptr (gx d p i : Int) := by
  have hrowr : ((tl.arange 0 d.BD).row d.BT).get i = (tl.arange 0 d.BD).get1 i.2.1 := rfl
  have hrowi : IntV.toInt (((tl.arange 0 d.BD).row d.BT).get i) = (i.2.1.val : Int) := by
    rw [hrowr]
    exact arange_toInt d.BD hBD i.2.1
  rw [Tensor.get_ptrAddT, Tensor.get_ptrAdd, Ptr.offset_offset,
    get_colprod hD hrow hot i, hrowi, hq, Buf.ptr_offset]
  have hidx : (0 : Int) +
      (((p * d.BT + i.1.val : Nat) : Int) * (d.D : Int) + (i.2.1.val : Int))
      = ((gx d p i : Nat) : Int) := by
    simp only [gx]
    push_cast
    ring
  rw [hidx]

/-- The `[BT]` `rstd` pointer block, lane by lane. -/
theorem get_row_ptr {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64}
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (i : Idx [d.BT]) :
    (d.rstd.ptr 0 + o_t).get i = d.rstd.ptr (gT d p i : Int) := by
  have hoti : IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int) := hot i
  rw [Tensor.get_ptrAdd, hoti, Buf.ptr_offset]
  have hidx : (0 : Int) + ((p * d.BT + i.1.val : Nat) : Int) = ((gT d p i : Nat) : Int) := by
    simp only [gT]
    push_cast
    ring
  rw [hidx]

/-- The `[BT, BD]` mask, lane by lane. -/
theorem get_tile_mask {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hD : d.D < 2 ^ 31) (hBD : d.BD < 2 ^ 31) (hT64 : IntV.toInt T64 = (d.T : Int))
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (i : Idx [d.BT, d.BD]) :
    (mX d o_t T64).get i = enx d p i := by
  have h1 : ((o_t.ltS T64).col d.BD).get i = IntV.lt (o_t.get (i.1, ())) T64 := rfl
  have h2 : (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT).get i
      = IntV.lt ((tl.arange 0 d.BD).get1 i.2.1) (IntV.ofNat d.D : I32) := rfl
  have hoti : IntV.toInt (o_t.get (i.1, ())) = ((p * d.BT + i.1.val : Nat) : Int) := hot (i.1, ())
  have hari : IntV.toInt ((tl.arange 0 d.BD).get1 i.2.1) = (i.2.1.val : Int) :=
    arange_toInt d.BD hBD i.2.1
  have hDval : IntV.toInt (IntV.ofNat d.D : I32) = (d.D : Int) := IntV.I32.toInt_ofNat hD
  simp only [mX, enx, Tensor.and, Tensor.get_map2, h1, h2, IntV.lt_eq]
  have hp : decide (IntV.toInt (o_t.get (i.1, ())) < IntV.toInt T64)
      = decide (p * d.BT + i.1.val < d.T) := by
    rw [decide_eq_decide, hoti, hT64]
    norm_cast
  have hq : decide (IntV.toInt ((tl.arange 0 d.BD).get1 i.2.1)
        < IntV.toInt (IntV.ofNat d.D : I32)) = decide (i.2.1.val < d.D) := by
    rw [decide_eq_decide, hari, hDval]
    norm_cast
  rw [hp, hq]

/-- The `[BT]` mask, lane by lane. -/
theorem get_row_mask {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hT64 : IntV.toInt T64 = (d.T : Int))
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (i : Idx [d.BT]) :
    (mT d o_t T64).get i = enT d p i := by
  have h1 : (mT d o_t T64).get i = IntV.lt (o_t.get i) T64 := rfl
  have hoti : IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int) := hot i
  rw [enT, h1, IntV.lt_eq]
  have hp : decide (IntV.toInt (o_t.get i) < IntV.toInt T64)
      = decide (p * d.BT + i.1.val < d.T) := by
    rw [decide_eq_decide, hoti, hT64]
    norm_cast
  rw [hp]

/-! ## The accesses of one program -/

/-- The pointers and mask of the `x` load address the tile's elements of `x` lane by lane. -/
theorem addrs_x {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hD : d.D < 2 ^ 31) (hD512 : d.D ≤ 512) (hBD : d.BD < 2 ^ 31)
    (hrow : ∀ i : Idx [d.BT], 0 ≤ IntV.toInt (o_t.get i) ∧ IntV.toInt (o_t.get i) < 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int)) :
    d.x.Addrs (tl.addrs (pX d o_t) (some (mX d o_t T64))) (gx d p) (enx d p) := by
  refine Buf.addrs_of_get d.x ?_ ?_
  · intro i
    exact get_tile_mask hD hBD hT64 hot i
  · intro i _
    exact get_tile_ptr hD512 hBD hrow hot (d.x.ptr 0) d.x rfl i

/-- The pointers and mask of a `bfloat16` `y` store address the tile's elements of `y`. -/
theorem addrs_y16 {d : Inst} {p : Nat} {y : Buf .bf16} {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hD : d.D < 2 ^ 31) (hD512 : d.D ≤ 512) (hBD : d.BD < 2 ^ 31)
    (hrow : ∀ i : Idx [d.BT], 0 ≤ IntV.toInt (o_t.get i) ∧ IntV.toInt (o_t.get i) < 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int)) :
    y.Addrs (tl.addrs (pY16 y d o_t) (some (mX d o_t T64))) (gx d p) (enx d p) := by
  refine Buf.addrs_of_get y ?_ ?_
  · intro i
    exact get_tile_mask hD hBD hT64 hot i
  · intro i _
    exact get_tile_ptr hD512 hBD hrow hot (y.ptr 0) y rfl i

/-- The pointers and mask of a `float32` `y` store address the tile's elements of `y`. -/
theorem addrs_y32 {d : Inst} {p : Nat} {y : Buf .f32} {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hD : d.D < 2 ^ 31) (hD512 : d.D ≤ 512) (hBD : d.BD < 2 ^ 31)
    (hrow : ∀ i : Idx [d.BT], 0 ≤ IntV.toInt (o_t.get i) ∧ IntV.toInt (o_t.get i) < 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int)) :
    y.Addrs (tl.addrs (pY32 y d o_t) (some (mX d o_t T64))) (gx d p) (enx d p) := by
  refine Buf.addrs_of_get y ?_ ?_
  · intro i
    exact get_tile_mask hD hBD hT64 hot i
  · intro i _
    exact get_tile_ptr hD512 hBD hrow hot (y.ptr 0) y rfl i

/-- The pointers and mask of the `rstd` store address the block's rows of `rstd`. -/
theorem addrs_rstd {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64} {T64 : I64}
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int)) :
    d.rstd.Addrs (tl.addrs (pR d o_t) (some (mT d o_t T64))) (gT d p) (enT d p) := by
  refine Buf.addrs_of_get d.rstd ?_ ?_
  · intro i
    exact get_row_mask hT64 hot i
  · intro i _
    exact get_row_ptr hot i

/-! ## The offset block `o_t = i_t · BT + o_a`, lane by lane -/

/-- The rows the grid covers stay inside `int64`: `T < 2^51`. -/
theorem t_lt {d : Inst} (hd : d.ok) : d.T < 2 ^ 51 := by
  have hB : 0 < d.BT := by
    obtain ⟨e, he⟩ := hd.bt_pow2
    rw [he]
    exact Nat.two_pow_pos e
  have hB20 : d.BT ≤ 2 ^ 20 := by
    have h1 : d.BT * d.BD ≤ 2 ^ 20 := hd.block_le
    have hb : 1 ≤ d.BD := by
      obtain ⟨e, he⟩ := hd.bd_pow2
      rw [he]
      exact Nat.one_le_two_pow
    have h2 : d.BT * 1 ≤ d.BT * d.BD := Nat.mul_le_mul_left _ hb
    rw [Nat.mul_one] at h2
    omega
  have hg := hd.grid_lt
  rw [Nat.div_lt_iff_lt_mul hB] at hg
  have h1 : d.T ≤ d.T + d.BT - 1 := by omega
  have h2 : 2 ^ 31 * d.BT ≤ 2 ^ 51 := by
    have h3 : (2 : Nat) ^ 31 * 2 ^ 20 = 2 ^ 51 := by norm_num
    rw [← h3]
    exact Nat.mul_le_mul_left _ hB20
  omega

/-- `o_t = i_t·BT + o_a` has, in lane `i`, the row `p·BT + i`. -/
theorem get_o_t {d : Inst} {p : Nat} {i_t : I64} {o_a : Tensor [d.BT] I64}
    (hi_t : IntV.toInt i_t = (p : Int))
    (hoa : ∀ i : Idx [d.BT], IntV.toInt (o_a.get i) = (i.1.val : Int))
    (hp : p * d.BT + d.BT ≤ 2 ^ 51) :
    ∀ i : Idx [d.BT],
      IntV.toInt ((i_t * (IntV.ofNat d.BT : I64) + o_a).get i) = ((p * d.BT + i.1.val : Nat) : Int) := by
  intro i
  have hB51 : d.BT ≤ 2 ^ 51 := by omega
  have hBv : IntV.toInt (IntV.ofNat d.BT : I64) = (d.BT : Int) :=
    IntV.toInt_ofNat (by exact_mod_cast lt_of_le_of_lt hB51 (by norm_num))
  have hval : ((p * d.BT + d.BT : Nat) : Int) ≤ 2 ^ 51 := by exact_mod_cast hp
  have hp0 : (0 : Int) ≤ (p : Int) := by exact_mod_cast Nat.zero_le p
  have hB0 : (0 : Int) ≤ (d.BT : Int) := by exact_mod_cast Nat.zero_le d.BT
  have hcast : ((p * d.BT + i.1.val : Nat) : Int) = (p : Int) * (d.BT : Int) + (i.1.val : Int) := by
    push_cast; ring
  have hsf_mul : IntV.SFits 64 (IntV.toInt i_t * IntV.toInt (IntV.ofNat d.BT : I64)) := by
    rw [hi_t, hBv]
    refine ⟨?_, ?_⟩
    · have hnn : (0 : Int) ≤ (p : Int) * (d.BT : Int) := mul_nonneg hp0 hB0
      have h0 : -(2 : Int) ^ (64 - 1) ≤ 0 := by norm_num
      linarith
    · have hple : (p : Int) * (d.BT : Int) ≤ ((p * d.BT + d.BT : Nat) : Int) := by
        have hexp : ((p * d.BT + d.BT : Nat) : Int) = (p : Int) * (d.BT : Int) + (d.BT : Int) := by
          push_cast; ring
        rw [hexp]; linarith
      exact lt_of_le_of_lt (le_trans hple hval) (by norm_num)
  have hmul : IntV.toInt (i_t * (IntV.ofNat d.BT : I64)) = (p : Int) * (d.BT : Int) := by
    rw [IntV.toInt_mul rfl rfl hsf_mul, hi_t, hBv]
  rw [Tensor.get_sadd]
  have hsf_add :
      IntV.SFits 64 (IntV.toInt (i_t * (IntV.ofNat d.BT : I64)) + IntV.toInt (o_a.get i)) := by
    rw [hmul, hoa i, ← hcast]
    refine ⟨?_, ?_⟩
    · have h0 : (0 : Int) ≤ ((p * d.BT + i.1.val : Nat) : Int) := by exact_mod_cast Nat.zero_le _
      have h1 : -(2 : Int) ^ (64 - 1) ≤ 0 := by norm_num
      linarith
    · have h2 : p * d.BT + i.1.val < 2 ^ 51 := by
        have h3 : i.1.val < d.BT := i.1.isLt
        omega
      exact_mod_cast lt_trans h2 (by norm_num)
  rw [IntV.toInt_add rfl rfl hsf_add, hmul, hoa i, hcast]

/-- The `hot` form the lane arithmetic wants. -/
theorem hrow_of_hot {d : Inst} {p : Nat} {o_t : Tensor [d.BT] I64}
    (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int)) :
    ∀ i : Idx [d.BT], 0 ≤ IntV.toInt (o_t.get i) ∧ IntV.toInt (o_t.get i) < 2 ^ 51 := by
  intro i
  rw [hot i]
  refine ⟨by exact_mod_cast Nat.zero_le _, ?_⟩
  have h1 : p * d.BT + i.1.val < 2 ^ 51 := by
    have h3 : i.1.val < d.BT := i.1.isLt
    omega
  exact_mod_cast h1


/-! ## Unpacking masks, pointers, injectivity and the lane of an element -/

/-! ## Unpacking the masks -/

/-- An enabled lane of the tile is in a row below `T`. -/
theorem enx_row {d : Inst} {p : Nat} {i : Idx [d.BT, d.BD]} (h : enx d p i = true) :
    p * d.BT + i.1.val < d.T := by
  simp only [enx, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1

/-- An enabled lane of the tile is in a column below `D`. -/
theorem enx_col {d : Inst} {p : Nat} {i : Idx [d.BT, d.BD]} (h : enx d p i = true) :
    i.2.1.val < d.D := by
  simp only [enx, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

/-- An enabled lane of the `[BT]` block is in a row below `T`. -/
theorem enT_row {d : Inst} {p : Nat} {i : Idx [d.BT]} (h : enT d p i = true) :
    p * d.BT + i.1.val < d.T := by
  simpa only [enT, decide_eq_true_eq] using h

/-- A lane in a row below `T` and a column below `D` is enabled. -/
theorem enx_of {d : Inst} {p : Nat} {i : Idx [d.BT, d.BD]}
    (h1 : p * d.BT + i.1.val < d.T) (h2 : i.2.1.val < d.D) : enx d p i = true := by
  simp only [enx, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨h1, h2⟩

/-- A lane in a row below `T` is enabled. -/
theorem enT_of {d : Inst} {p : Nat} {i : Idx [d.BT]} (h1 : p * d.BT + i.1.val < d.T) :
    enT d p i = true := by
  simpa only [enT, decide_eq_true_eq] using h1

/-! ## Where the enabled lanes point -/

/-- An enabled tile lane addresses an element below `T·D`. -/
theorem gx_lt_rowSet {d : Inst} (hd : d.ok) {p : Nat} {i : Idx [d.BT, d.BD]}
    (h : enx d p i = true) : gx d p i ∈ rowSet d := by
  have h1 := enx_row h
  have h2 := enx_col h
  show gx d p i < d.T * d.D
  have hD := hd.d_pos
  have h3 : p * d.BT + i.1.val + 1 ≤ d.T := h1
  simp only [gx]
  calc (p * d.BT + i.1.val) * d.D + i.2.1.val
      < (p * d.BT + i.1.val) * d.D + d.D := by omega
    _ = (p * d.BT + i.1.val + 1) * d.D := by ring
    _ ≤ d.T * d.D := Nat.mul_le_mul_right _ h3

/-- An enabled `[BT]` lane addresses a row below `T`. -/
theorem gT_lt_rSet {d : Inst} {p : Nat} {i : Idx [d.BT]} (h : enT d p i = true) :
    gT d p i ∈ rSet d := by
  have h1 := enT_row h
  show gT d p i < d.T
  simpa only [gT] using h1

/-- An enabled tile lane addresses an element inside `x`. -/
theorem gx_lt_xLen {d : Inst} (hd : d.ok) {p : Nat} {i : Idx [d.BT, d.BD]}
    (h : enx d p i = true) : gx d p i < d.x.len :=
  lt_of_lt_of_le (lt_of_lt_of_le (gx_lt_rowSet hd h) (le_of_eq rfl)) hd.x_len

/-- An enabled tile lane addresses an element inside `y`. -/
theorem gx_lt_yLen {d : Inst} (hd : d.ok) {p : Nat} {i : Idx [d.BT, d.BD]}
    (h : enx d p i = true) : gx d p i < yLen d :=
  lt_of_lt_of_le (gx_lt_rowSet hd h) hd.y_len

/-- An enabled `[BT]` lane addresses an element inside `rstd`. -/
theorem gT_lt_rLen {d : Inst} (hd : d.ok) {p : Nat} {i : Idx [d.BT]}
    (h : enT d p i = true) : gT d p i < d.rstd.len :=
  lt_of_lt_of_le (gT_lt_rSet h) hd.r_len

/-! ## Injectivity -/

/-- Distinct enabled tile lanes address distinct elements. -/
theorem gx_inj {d : Inst} (hd : d.ok) {p : Nat} {i j : Idx [d.BT, d.BD]}
    (hi : enx d p i = true) (hj : enx d p j = true) (h : gx d p i = gx d p j) : i = j := by
  have hDi := enx_col hi
  have hDj := enx_col hj
  have hD := hd.d_pos
  have hmod : ((p * d.BT + i.1.val) * d.D + i.2.1.val) % d.D
      = ((p * d.BT + j.1.val) * d.D + j.2.1.val) % d.D := congrArg (· % d.D) h
  have hval : i.2.1.val = j.2.1.val := by
    simpa [Nat.add_mod, Nat.mul_mod_right, Nat.mod_eq_of_lt hDi, Nat.mod_eq_of_lt hDj] using hmod
  have hcol : i.2.1 = j.2.1 := Fin.ext hval
  have hmul : (p * d.BT + i.1.val) * d.D = (p * d.BT + j.1.val) * d.D := by
    simp only [gx] at h
    omega
  have hrow : i.1 = j.1 := Fin.ext (by have := Nat.eq_of_mul_eq_mul_right hD hmul; omega)
  exact Prod.ext hrow (Prod.ext hcol rfl)

/-- Distinct enabled `[BT]` lanes address distinct rows. -/
theorem gT_inj {d : Inst} {p : Nat} {i j : Idx [d.BT]} (hi : enT d p i = true)
    (hj : enT d p j = true) (h : gT d p i = gT d p j) : i = j := by
  have h1 := enT_row hi
  have h2 := enT_row hj
  refine Prod.ext (Fin.ext ?_) rfl
  simp only [gT] at h
  omega

/-! ## The lane of an element of the tile -/

/-- The lane of row `t` and column `j` addresses element `t·D + j`. -/
theorem lane_gx {d : Inst} {p t j : Nat} (ht : t ∈ tileRow d p) {h1 : t - p * d.BT < d.BT}
    {h2 : j < d.BD} :
    gx d p (⟨t - p * d.BT, h1⟩, (⟨j, h2⟩, ())) = xIdx d t j := by
  obtain ⟨_, hlo, _⟩ := ht
  simp only [gx, xIdx]
  rw [show p * d.BT + (t - p * d.BT) = t by omega]

/-- The lane of an in-range row `t` and column `j < D` is enabled. -/
theorem lane_enx {d : Inst} {p t j : Nat} (ht : t ∈ tileRow d p) (hj : j < d.D)
    {h1 : t - p * d.BT < d.BT} {h2 : j < d.BD} :
    enx d p (⟨t - p * d.BT, h1⟩, (⟨j, h2⟩, ())) = true := by
  apply enx_of
  · obtain ⟨hT, hlo, _⟩ := ht
    show p * d.BT + (t - p * d.BT) < d.T
    omega
  · simpa using hj

/-- The `[BT]` lane of row `t` addresses row `t`. -/
theorem lane_gT {d : Inst} {p t j : Nat} (ht : t ∈ tileRow d p) {h1 : t - p * d.BT < d.BT}
    {_h2 : j < d.BD} :
    gT d p (⟨t - p * d.BT, h1⟩, ()) = t := by
  obtain ⟨_, hlo, _⟩ := ht
  simp only [gT]
  omega

/-- The `[BT]` lane of an in-range row `t` is enabled. -/
theorem lane_enT {d : Inst} {p t j : Nat} (ht : t ∈ tileRow d p)
    {h1 : t - p * d.BT < d.BT} {_h2 : j < d.BD} :
    enT d p (⟨t - p * d.BT, h1⟩, ()) = true := by
  apply enT_of
  obtain ⟨hT, hlo, _⟩ := ht
  show p * d.BT + (t - p * d.BT) < d.T
  omega

/-- A row of the block has lane index `t - p·BT < BT`. -/
theorem row_sub_lt {d : Inst} (_hd : d.ok) {p t : Nat} (ht : t ∈ tileRow d p) :
    t - p * d.BT < d.BT := by
  obtain ⟨_, hlo, hhi⟩ := ht
  omega

/-- A column below `D` is below `BD`. -/
theorem col_lt_bd {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.D) : j < d.BD :=
  lt_of_lt_of_le hj hd.d_bd

end Claims.L2NormFwd
