import Proof.Common
import Proof.Basic
import Proof.Meta
import Proof.Tol

/-!
# The non-interleaved tail of one program

The rotation of the diagonal blocks: channels `r < R` and `r + R` of each token, with the
`R`-wide tables. The hypotheses are what the prologue of `body_eq` establishes.
-/

namespace Claims.RotaryVarlen

open TVA

set_option maxHeartbeats 1000000

/-! ## Element access at a concrete index

The library's `[simp]` lemmas for elementwise operations are stated with an `Idx s` argument
whose shape is implicit. At a concrete tuple index such as `(l, c, ())` the elaborator cannot
see through `Idx` at `implicit` transparency, so `rw` and `simp` reject them. The lemmas below
restate them with the index components explicit; each is `rfl`. -/

/-! ## The program's tensors, as abbreviations

The `Safe` rules for memory access take the index map `g` and the enabling predicate `en`
explicitly. The abbreviations below name the program's tensors, so that the rules' side
conditions (addresses, bounds) are stated once, per tensor. -/

/-- The tile's token indices, `i_t * BT + arange(0, BT)`. -/
private abbrev ne_oT (d : Inst) (i_t : I64) : Tensor [d.BT] I64 :=
  i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)

/-- The tile's channel indices, `arange(0, BD / 2)`. -/
private abbrev ne_oR (d : Inst) : Tensor [d.BD / 2] I32 := tl.arange 0 (d.BD / 2)

/-- The token part of the tile's mask, as the kernel computes it. -/
private abbrev ne_mT (d : Inst) (i_t : I64) (T_n : I64) (o_cs : Tensor [d.BT] I64) :
    Tensor [d.BT] Bool :=
  ((((ne_oT d i_t).ltS (IntV.ofInt 0 : I64)).not.and ((ne_oT d i_t).ltS T_n)).and
    ((o_cs.ltS (IntV.ofInt 0 : I64)).not)).and (o_cs.ltS (IntV.ofNat d.TR : I64))

/-- The tile's mask as the kernel computes it: the token part broadcast down the columns, anded with the channel test. -/
private abbrev ne_mask (d : Inst) (i_t : I64) (T_n : I64) (o_cs : Tensor [d.BT] I64) :
    Tensor [d.BT, d.BD / 2] Bool :=
  ((ne_mT d i_t T_n o_cs).col (d.BD / 2)).and
    (((ne_oR d).ltS (IntV.ofNat d.R : I32)).row d.BT)

/-- The `H * D` row stride of a token. -/
private abbrev ne_hd (d : Inst) (i_t : I64) : Tensor [d.BT, d.BD / 2] I64 :=
  (ne_oT d i_t).col (d.BD / 2) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)

/-- The `cos`/`sin` table offset of a lane. -/
private abbrev ne_csOff (d : Inst) (o_cs : Tensor [d.BT] I64) :
    Tensor [d.BT, d.BD / 2] I64 :=
  o_cs.col (d.BD / 2) * (IntV.ofNat d.R : I64)
    + (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64)

/-- The pointers of the first-half `x` operand. -/
private abbrev ne_pX (d : Inst) (x_h : Ptr .bf16) (i_t : I64) :
    Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
  (x_h + ne_hd d i_t) + (ne_oR d).row d.BT

/-- The pointers of the stored output. -/
private abbrev ne_pY (d : Inst) (y_h : Ptr .bf16) (i_t : I64) :
    Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
  y_h + (ne_hd d i_t + (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))

/-- The pointers of the `cos` operand. -/
private abbrev ne_pCos (d : Inst) (o_cs : Tensor [d.BT] I64) :
    Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
  d.cos.ptr 0 + ne_csOff d o_cs

/-- The pointers of the `sin` operand. -/
private abbrev ne_pSin (d : Inst) (o_cs : Tensor [d.BT] I64) :
    Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
  d.sin.ptr 0 + ne_csOff d o_cs

/-- The `cos`/`sin` element of a lane. -/
private def ne_gCS (d : Inst) (init : Mem) (i : Nat) (k : Idx [d.BT, d.BD / 2]) : Nat :=
  (posI d init i k.1.val * (d.R : Int) + (k.2.1.val : Int)).toNat

/-- The `x` (and `y`) element of a lane. -/
private def ne_gX0 (d : Inst) (init : Mem) (i : Nat) (k : Idx [d.BT, d.BD / 2]) : Nat :=
  tokIdx d (tokI d init i k.1.val) (ax2 d i) k.2.1.val

/-- The `x` (and `y`) element of a lane's partner, channel `c + R`. -/
private def ne_gX1 (d : Inst) (init : Mem) (i : Nat) (k : Idx [d.BT, d.BD / 2]) : Nat :=
  tokIdx d (tokI d init i k.1.val) (ax2 d i) (k.2.1.val + d.R)

/-- The program's mask, as the enabling predicate of the `Buf` kit. -/
private def ne_en (d : Inst) (i_t : I64) (T_n : I64) (o_cs : Tensor [d.BT] I64)
    (k : Idx [d.BT, d.BD / 2]) : Bool :=
  (ne_mask d i_t T_n o_cs).get k

/-! ## The mask says `laneOn` and `c < R` -/

private theorem mt_get {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get k).toInt
          = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63)
    (k : Idx [d.BT]) :
    (((((i_t * (IntV.ofNat d.BT : I64)
          + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).ltS (IntV.ofInt 0 : I64)).not.and
        ((i_t * (IntV.ofNat d.BT : I64)
          + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).ltS T_n)).and
      (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64))).get k = true
      ↔ laneOn d init i k.1.val := by
  simp only [Tensor.get_and, Tensor.get_not, Tensor.get_ltS, Bool.and_eq_true, Bool.not_eq_true',
    IntV.lt_eq]
  rw [hot k, hT, hocs k]
  rw [ofNat_toInt (w := 64) (n := d.TR)  (by
    have h1 : (d.TR : ℤ) < (2 : ℤ) ^ 63 := by exact_mod_cast hTR
    omega)]
  rw [IntV.toInt_ofInt (w := 64) (x := 0) (by norm_num) (by
    simp only [IntV.SFits]; constructor <;> norm_num)]
  simp only [decide_eq_true_eq, decide_eq_false_iff_not]
  constructor
  · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩
    exact ⟨h2, by omega, h4⟩
  · rintro ⟨h1, h2, h3⟩
    exact ⟨⟨⟨by omega, h1⟩, by omega⟩, h3⟩

private theorem or_lt (d : Inst) (c : Fin (d.BD / 2)) (hBD : d.BD ≤ 2 ^ 20) (hR : d.R < 2 ^ 31) :
    ((tl.arange 0 (d.BD / 2)).ltS (IntV.ofNat d.R : I32)).get (c, ())
      = decide ((c.val : ℤ) < (d.R : ℤ)) := by
  have h1 : ((tl.arange 0 (d.BD / 2)).ltS (IntV.ofNat d.R : I32)).get (c, ())
      = ((tl.arange 0 (d.BD / 2)).get (c, ())).lt (IntV.ofNat d.R : I32) :=
    Tensor.get_ltS _ _ _
  rw [h1]
  have har : (tl.arange 0 (d.BD / 2)).get (c, ())
      = (Tensor.arange (0 : ℤ) (d.BD / 2)).get (c, ()) := rfl
  rw [har, Tensor.get_arange, IntV.lt_eq]
  rw [IntV.toInt_ofInt (w := 32) (by norm_num) (by
    simp only [IntV.SFits]; constructor <;> push_cast <;> omega)]
  rw [ofNat_toInt (w := 32) (n := d.R)  (by
    have h1 : (d.R : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast hR
    have h2 : (2 : ℤ) ^ 31 < (2 : ℤ) ^ 63 := by norm_num
    omega)]
  norm_num

private theorem mask_get {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get k).toInt
          = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hR : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20)
    (k : Idx [d.BT, d.BD / 2]) :
    (let o_t : Tensor [d.BT] I64 :=
        i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)
     let m_t : Tensor [d.BT] Bool :=
        (((o_t.ltS (IntV.ofInt 0 : I64)).not.and (o_t.ltS T_n)).and
          (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64))
     let o_r : Tensor [d.BD / 2] I32 := tl.arange 0 (d.BD / 2)
     let mask : Tensor [d.BT, d.BD / 2] Bool :=
        (m_t.col (d.BD / 2)).and ((o_r.ltS (IntV.ofNat d.R : I32)).row d.BT)
     mask.get k = true ↔ laneOn d init i k.1.val ∧ k.2.1.val < d.R) := by
  dsimp only
  set mt : Tensor [d.BT] Bool :=
    ((((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).ltS (IntV.ofInt 0 : I64)).not.and
      ((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).ltS T_n)).and
      (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64)) with hmt
  set o_r : Tensor [d.BD / 2] I32 := tl.arange 0 (d.BD / 2) with horr
  have hA : ((mt.col (d.BD / 2)).and ((o_r.ltS (IntV.ofNat d.R : I32)).row d.BT)).get k
      = ((mt.col (d.BD / 2)).get k && ((o_r.ltS (IntV.ofNat d.R : I32)).row d.BT).get k) :=
    Tensor.get_and _ _ k
  rw [hA, col_get' mt k, row_get' (o_r.ltS (IntV.ofNat d.R : I32)) k]
  rw [hmt, horr, Bool.and_eq_true]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨(mt_get hot hT hocs hTR (k.1, ())).1 h1, ?_⟩
    rw [or_lt d k.2.1 hBD hR] at h2
    simp only [decide_eq_true_eq] at h2
    omega
  · rintro ⟨h1, h2⟩
    refine ⟨(mt_get hot hT hocs hTR (k.1, ())).2 h1, ?_⟩
    rw [or_lt d k.2.1 hBD hR]
    simp only [decide_eq_true_eq]
    omega

private theorem ne_mask_get {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hR : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20)
    (k : Idx [d.BT, d.BD / 2]) :
    (ne_mask d i_t T_n o_cs).get k = true ↔ laneOn d init i k.1.val ∧ k.2.1.val < d.R :=
  mask_get hot hT hocs hTR hR hBD k

/-! ## Bounds -/

private theorem ne_R_le {d : Inst} (hd : d.ok) : d.R ≤ 128 := by
  have h1 := hd.r_d
  have h2 := hd.d_le
  omega

private theorem ne_R_lt {d : Inst} (hd : d.ok) : d.R < 2 ^ 31 := by
  have h := ne_R_le hd
  omega

private theorem ne_TR_lt63 {d : Inst} (hd : d.ok) : d.TR < 2 ^ 63 := by
  have h1 := hd.tr_lt
  have h2 := hd.r_pos
  have h3 : d.TR ≤ d.TR * d.R := Nat.le_mul_of_pos_right _ h2
  omega

private theorem ofNat_R_I64 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.R : I64).toInt = (d.R : Int) := by
  refine ofNat_toInt  ?_
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [h2]
  have h1 : (d.R : Int) ≤ 128 := by exact_mod_cast ne_R_le hd
  have h3 : (2 : Int) ^ 7 < 2 ^ 63 := by norm_num
  linarith

/-! ## The value of each tensor at a lane -/

/-- The `H * D` stride tensor takes the token's flat index times `H * D`. -/
private theorem ne_hd_val {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    ((ne_hd d i_t).get (l, c, ())).toInt
      = (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) := by
  have hH : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := ofNat_H_I64 hd
  have hD : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := ofNat_D_I64 hd
  have hloc : ((ne_oT d i_t).get (l, ())).toInt = (locI d init i l.val : Int) :=
    oT_toInt hd hpre hi hit l
  have hloc0 : (0 : Int) ≤ (locI d init i l.val : Int) := by exact_mod_cast Nat.zero_le _
  have hloc32 : (locI d init i l.val : Int) < 2 ^ 32 := by
    have hcbt := chunkI_mul_BT_lt hd hpre hi
    have hcbt' : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 31 := by
      have h2 : ((chunkI d init i * d.BT : Nat) : Int) < (2 : Int) ^ 31 := by exact_mod_cast hcbt
      push_cast at h2 ⊢
      linarith
    have hl : (l.val : Int) < 2 ^ 20 := by
      have h1 : (l.val : Int) < (d.BT : Int) := by exact_mod_cast l.isLt
      have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast bt_le hd
      linarith
    simp only [locI]
    push_cast
    linarith
  have hH0 : (0 : Int) ≤ (d.H : Int) := by exact_mod_cast Nat.zero_le d.H
  have hD0 : (0 : Int) ≤ (d.D : Int) := by exact_mod_cast Nat.zero_le d.D
  have hH16 : (d.H : Int) < 2 ^ 16 := by exact_mod_cast hd.grid_h
  have hD256 : (d.D : Int) ≤ 256 := by exact_mod_cast hd.d_le
  have hb1 : IntV.SFits 64 ((locI d init i l.val : Int) * (d.H : Int)) := by
    refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · nlinarith
  have hb2 : IntV.SFits 64 ((locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)) := by
    have hp1 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) := mul_nonneg hloc0 hH0
    have hp1' : (locI d init i l.val : Int) * (d.H : Int) < 2 ^ 32 * 2 ^ 16 := by nlinarith
    have hle : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
        ≤ (2 ^ 32 * 2 ^ 16) * 256 := by
      have h2 : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
          ≤ (2 ^ 32 * 2 ^ 16) * (d.D : Int) := mul_le_mul_of_nonneg_right hp1'.le hD0
      have h3 : (2 ^ 32 * 2 ^ 16) * (d.D : Int) ≤ (2 ^ 32 * 2 ^ 16) * 256 :=
        mul_le_mul_of_nonneg_left hD256 (by norm_num)
      linarith
    have hp0 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) :=
      mul_nonneg hp1 hD0
    have h3 : (2 : Int) ^ 32 * 2 ^ 16 * 256 < 2 ^ 63 := by norm_num
    exact sfits64_of_bounds (by linarith) (by linarith)
  have hs1 : IntV.SFits 64
      (((ne_oT d i_t).get (l, ())).toInt * (IntV.ofNat d.H : I64).toInt) := by
    rw [hloc, hH]; exact hb1
  have hs2 : IntV.SFits 64
      ((((ne_oT d i_t).get (l, ())) * (IntV.ofNat d.H : I64)).toInt
        * (IntV.ofNat d.D : I64).toInt) := by
    rw [IntV.toInt_mul rfl rfl hs1, hloc, hH, hD]; exact hb2
  have hg : ((ne_hd d i_t).get (l, c, ()))
      = ((ne_oT d i_t).get (l, ()) * (IntV.ofNat d.H : I64)) * (IntV.ofNat d.D : I64) := by
    simp only [ne_hd, g2_muls, Tensor.get_col]
  rw [hg, IntV.toInt_mul rfl rfl hs2,
    IntV.toInt_mul rfl rfl hs1, hloc, hH, hD]

/-- The lane's channel, as an `I32` value. -/
private theorem ne_or_val {d : Inst} (hd : d.ok) (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    (((ne_oR d).row d.BT).get (l, c, ())).toInt = (c.val : Int) := by
  have hg : ((ne_oR d).row d.BT).get (l, c, ()) = IntV.ofInt (c.val : Int) := by
    rw [ne_oR, g_row1, g_arange]
    norm_num
  rw [hg]
  refine IntV.I32.toInt_ofInt ⟨?_, ?_⟩
  · have hc0 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
    linarith
  · have hcbd : c.val < d.BD := lt_of_lt_of_le c.isLt (Nat.div_le_self _ 2)
    have hc20 : c.val < 2 ^ 20 := lt_of_lt_of_le hcbd (bd_le hd)
    have hc : (c.val : Int) < 2 ^ 31 := by
      have h1 : (c.val : Int) < (2 : Int) ^ 20 := by exact_mod_cast hc20
      have h2 : (2 : Int) ^ 20 < 2 ^ 31 := by norm_num
      linarith
    linarith

/-- The `cos`/`sin` table offset of a lane is `posI * R + c`. -/
private theorem ne_csOff_val {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    {o_cs : Tensor [d.BT] I64}
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    {l : Fin d.BT} {c : Fin (d.BD / 2)}
    (hlan : laneOn d init i l.val) (hc : c.val < d.R) :
    ((ne_csOff d o_cs).get (l, c, ())).toInt
      = posI d init i l.val * (d.R : Int) + (c.val : Int) := by
  have hR64 : (IntV.ofNat d.R : I64).toInt = (d.R : Int) := ofNat_R_I64 hd
  have hcast : ((((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64).get
      (l, c, ())).toInt = (c.val : Int) := by
    rw [show (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64).get (l, c, ())
        = IntV.cast (((ne_oR d).row d.BT).get (l, c, ())) from rfl]
    rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num), ne_or_val hd]
  have hpos0 : (0 : Int) ≤ posI d init i l.val := hlan.2.1
  have hposTR : posI d init i l.val < (d.TR : Int) := hlan.2.2
  have hTR62 : (d.TR : Int) * (d.R : Int) < 2 ^ 62 := by exact_mod_cast hd.tr_lt
  have hc0 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
  have hclt : (c.val : Int) < (d.R : Int) := by exact_mod_cast hc
  have hR0 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
  have hb1 : IntV.SFits 64 (posI d init i l.val * (d.R : Int)) := by
    refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · nlinarith
  have hb2 : IntV.SFits 64 (posI d init i l.val * (d.R : Int) + (c.val : Int)) := by
    refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · nlinarith
  have hs1 : IntV.SFits 64
      ((o_cs.get (l, ())).toInt * (IntV.ofNat d.R : I64).toInt) := by
    rw [hocs l, hR64]; exact hb1
  have hs2 : IntV.SFits 64
      (((o_cs.get (l, ())) * (IntV.ofNat d.R : I64)).toInt
        + ((((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64).get (l, c, ())).toInt) := by
    rw [IntV.toInt_mul rfl rfl hs1, hocs l, hR64, hcast]; exact hb2
  have hg : ((ne_csOff d o_cs).get (l, c, ()))
      = (o_cs.get (l, ())) * (IntV.ofNat d.R : I64)
        + ((((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64)).get (l, c, ()) := by
    simp only [ne_csOff, g2_add, g2_muls, g_col1]
  rw [hg, IntV.toInt_add rfl rfl hs2,
    IntV.toInt_mul rfl rfl hs1, hocs l, hR64, hcast]

/-- The value of the `x` pointer's index, as an `Int`. -/
private theorem ne_gX0_cast {d : Inst} {init : Mem} {i : Nat} (l : Fin d.BT)
    (c : Fin (d.BD / 2)) :
    ((ne_gX0 d init i (l, c, ()) : Nat) : Int)
      = (((bosI d init i * d.H + ax2 d i) * d.D : Nat) : Int)
        + (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (c.val : Int) := by
  simp only [ne_gX0, tokIdx, tokI]
  push_cast
  ring

/-- The partner channel's index is the lane's index plus `R`. -/
private theorem ne_gX1_eq {d : Inst} {init : Mem} {i : Nat} (l : Fin d.BT)
    (c : Fin (d.BD / 2)) :
    ne_gX1 d init i (l, c, ()) = ne_gX0 d init i (l, c, ()) + d.R := by
  simp only [ne_gX1, ne_gX0, tokIdx]
  omega

/-- The value of the partner pointer's index, as an `Int`. -/
private theorem ne_gX1_cast {d : Inst} {init : Mem} {i : Nat} (l : Fin d.BT)
    (c : Fin (d.BD / 2)) :
    ((ne_gX1 d init i (l, c, ()) : Nat) : Int)
      = (((bosI d init i * d.H + ax2 d i) * d.D : Nat) : Int)
        + (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (c.val : Int)
        + (d.R : Int) := by
  have h := ne_gX0_cast (d := d) (init := init) (i := i) l c
  have he := ne_gX1_eq (d := d) (init := init) (i := i) l c
  rw [he]
  push_cast
  rw [h]
  push_cast
  ring

/-! ## Pointer addresses -/

/-- The `cos` pointer of a lane. -/
private theorem ne_pCos_get {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    {o_cs : Tensor [d.BT] I64}
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    {l : Fin d.BT} {c : Fin (d.BD / 2)} (hlan : laneOn d init i l.val) (hc : c.val < d.R) :
    (ne_pCos d o_cs).get (l, c, ()) = d.cos.ptr (ne_gCS d init i (l, c, ())) := by
  have h0 : (0 : Int) ≤ posI d init i l.val * (d.R : Int) + (c.val : Int) := by
    have h2 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
    have h3 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
    nlinarith [hlan.2.1]
  have hcast : ((ne_gCS d init i (l, c, ()) : Nat) : Int)
      = posI d init i l.val * (d.R : Int) + (c.val : Int) := by
    simp only [ne_gCS]
    exact Int.toNat_of_nonneg h0
  refine ptr_eq_of_addr ?_
  have hg : (ne_pCos d o_cs).get (l, c, ())
      = (d.cos.ptr 0).offset (((ne_csOff d o_cs).get (l, c, ())).toInt) := by
    simp only [ne_pCos, g2_ptrAdd]
  rw [hg, Ptr.offset_addr, ne_csOff_val hd hocs hlan hc, Buf.ptr_addr, hcast,
    Buf.ptr_addr]
  ring

/-- The `sin` pointer of a lane. -/
private theorem ne_pSin_get {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    {o_cs : Tensor [d.BT] I64}
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    {l : Fin d.BT} {c : Fin (d.BD / 2)} (hlan : laneOn d init i l.val) (hc : c.val < d.R) :
    (ne_pSin d o_cs).get (l, c, ()) = d.sin.ptr (ne_gCS d init i (l, c, ())) := by
  have h0 : (0 : Int) ≤ posI d init i l.val * (d.R : Int) + (c.val : Int) := by
    have h2 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
    have h3 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
    nlinarith [hlan.2.1]
  have hcast : ((ne_gCS d init i (l, c, ()) : Nat) : Int)
      = posI d init i l.val * (d.R : Int) + (c.val : Int) := by
    simp only [ne_gCS]
    exact Int.toNat_of_nonneg h0
  refine ptr_eq_of_addr ?_
  have hg : (ne_pSin d o_cs).get (l, c, ())
      = (d.sin.ptr 0).offset (((ne_csOff d o_cs).get (l, c, ())).toInt) := by
    simp only [ne_pSin, g2_ptrAdd]
  rw [hg, Ptr.offset_addr, ne_csOff_val hd hocs hlan hc, Buf.ptr_addr, hcast,
    Buf.ptr_addr]
  ring

/-- The `x` pointer of a lane. -/
private theorem ne_pX_get {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {x_h : Ptr .bf16}
    (hx : x_h.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    (ne_pX d x_h i_t).get (l, c, ()) = d.x.ptr (ne_gX0 d init i (l, c, ())) := by
  refine ptr_eq_of_addr ?_
  have hg : (ne_pX d x_h i_t).get (l, c, ())
      = (x_h.offset (((ne_hd d i_t).get (l, c, ())).toInt)).offset
          ((((ne_oR d).row d.BT).get (l, c, ())).toInt) := by
    simp only [ne_pX, g2_ptrAdd, g2_ptrAddT]
  rw [hg, Ptr.offset_addr, Ptr.offset_addr, hx, Buf.ptr_addr, ne_hd_val hd hpre hi hit l c,
    ne_or_val hd l c, ne_gX0_cast (d := d) (init := init) (i := i) l c,
    Buf.ptr_addr]
  push_cast
  ring_nf

/-- The `x` pointer of a lane's partner channel `c + R`. -/
private theorem ne_pX1_get {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {x_h : Ptr .bf16}
    (hx : x_h.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    ((ne_pX d x_h i_t) + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)).get (l, c, ())
      = d.x.ptr (ne_gX1 d init i (l, c, ())) := by
  have hp := ne_pX_get hd hpre hi hit hx l c
  have hR : ((bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32).get (l, c, ())).toInt
      = (d.R : Int) := by
    rw [g_full2]; exact ofNat_R_I32 hd
  refine ptr_eq_of_addr ?_
  have hg : ((ne_pX d x_h i_t)
        + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)).get (l, c, ())
      = ((ne_pX d x_h i_t).get (l, c, ())).offset (d.R : Int) := by
    simp only [g2_ptrAddT, hR]
  rw [hg, hp, Buf.ptr_offset]
  have hc : ((ne_gX0 d init i (l, c, ()) : Nat) : Int) + (d.R : Int)
      = ((ne_gX1 d init i (l, c, ()) : Nat) : Int) := by
    rw [ne_gX1_eq (d := d) (init := init) (i := i) l c]
    push_cast
    ring
  rw [hc]

/-- The `H * D` stride of a lane is nonnegative and below `2^63`. -/
private theorem ne_val_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) (l : Fin d.BT) :
    (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
      ∧ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) < 2 ^ 62 := by
  have hloc32 : (locI d init i l.val : Int) < 2 ^ 32 := by
    have hcbt : chunkI d init i * d.BT < 2 ^ 31 := chunkI_mul_BT_lt hd hpre hi
    have hcbt' : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 31 := by
      have h2 : ((chunkI d init i * d.BT : Nat) : Int) < (2 : Int) ^ 31 := by exact_mod_cast hcbt
      push_cast at h2 ⊢
      linarith
    have hl : (l.val : Int) < 2 ^ 20 := by
      have h1 : (l.val : Int) < (d.BT : Int) := by exact_mod_cast l.isLt
      have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast bt_le hd
      linarith
    simp only [locI]
    push_cast
    linarith
  have hloc0 : (0 : Int) ≤ (locI d init i l.val : Int) := by exact_mod_cast Nat.zero_le _
  have hH0 : (0 : Int) ≤ (d.H : Int) := by exact_mod_cast Nat.zero_le d.H
  have hD0 : (0 : Int) ≤ (d.D : Int) := by exact_mod_cast Nat.zero_le d.D
  have hH16 : (d.H : Int) < 2 ^ 16 := by exact_mod_cast hd.grid_h
  have hD256 : (d.D : Int) ≤ 256 := by exact_mod_cast hd.d_le
  have hp1 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) := mul_nonneg hloc0 hH0
  have hp1' : (locI d init i l.val : Int) * (d.H : Int) < 2 ^ 32 * 2 ^ 16 := by nlinarith
  have hle : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
      ≤ (2 ^ 32 * 2 ^ 16) * 256 := by
    have h2 : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
        ≤ (2 ^ 32 * 2 ^ 16) * (d.D : Int) := mul_le_mul_of_nonneg_right hp1'.le hD0
    have h3 : (2 ^ 32 * 2 ^ 16) * (d.D : Int) ≤ (2 ^ 32 * 2 ^ 16) * 256 :=
      mul_le_mul_of_nonneg_left hD256 (by norm_num)
    linarith
  have hp0 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) :=
    mul_nonneg hp1 hD0
  have h3 : (2 : Int) ^ 32 * 2 ^ 16 * 256 < 2 ^ 62 := by norm_num
  exact ⟨hp0, by linarith⟩

/-- The `H * D` row stride tensor plus the channel: its value. -/
private theorem ne_hdR_val {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    (((ne_hd d i_t)
        + (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64)).get (l, c, ())).toInt
      = (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (c.val : Int) := by
  have hhd := ne_hd_val hd hpre hi hit l c
  have hor : (((((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64)).get (l, c, ())).toInt
      = (c.val : Int) := by
    rw [show (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64).get (l, c, ())
        = IntV.cast (((ne_oR d).row d.BT).get (l, c, ())) from rfl]
    rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num), ne_or_val hd l c]
  obtain ⟨hL0, hL63⟩ := ne_val_lt hd hpre hi l
  have hc0 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
  have hc20 : (c.val : Int) < 2 ^ 20 := by
    have h1 : (c.val : Int) < (d.BD : Int) := by
      exact_mod_cast lt_of_lt_of_le c.isLt (Nat.div_le_self _ 2)
    have h2 : (d.BD : Int) ≤ 2 ^ 20 := by exact_mod_cast bd_le hd
    linarith
  have hs : IntV.SFits 64 (((ne_hd d i_t).get (l, c, ())).toInt
      + ((((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64).get (l, c, ())).toInt) := by
    rw [hhd, hor]
    refine sfits64_of_bounds ?_ ?_
    · have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
      rw [h2]; linarith
    · linarith
  have hg : ((ne_hd d i_t)
        + (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64)).get (l, c, ())
      = (ne_hd d i_t).get (l, c, ())
        + (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64).get (l, c, ()) :=
    g2_add _ _ l c
  rw [hg, IntV.toInt_add rfl rfl hs, hhd, hor]

/-- The `y` pointer of a lane. -/
private theorem ne_pY_get {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {y_h : Ptr .bf16}
    (hy : y_h.addr = (d.y.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    (ne_pY d y_h i_t).get (l, c, ()) = d.y.ptr (ne_gX0 d init i (l, c, ())) := by
  refine ptr_eq_of_addr ?_
  have hg : (ne_pY d y_h i_t).get (l, c, ())
      = y_h.offset (((ne_hd d i_t
          + (((ne_oR d).row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64)).get (l, c, ())).toInt) := by
    simp only [ne_pY, g2_ptrAdd]
  rw [hg, Ptr.offset_addr, hy, Buf.ptr_addr, ne_hdR_val hd hpre hi hit l c,
    ne_gX0_cast (d := d) (init := init) (i := i) l c, Buf.ptr_addr]
  push_cast
  ring_nf

/-- The `y` pointer of a lane's partner channel `c + R`. -/
private theorem ne_pY1_get {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {y_h : Ptr .bf16}
    (hy : y_h.addr = (d.y.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (l : Fin d.BT) (c : Fin (d.BD / 2)) :
    ((ne_pY d y_h i_t) + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)).get (l, c, ())
      = d.y.ptr (ne_gX1 d init i (l, c, ())) := by
  have hp := ne_pY_get hd hpre hi hit hy l c
  have hR : ((bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32).get (l, c, ())).toInt
      = (d.R : Int) := by
    rw [g_full2]; exact ofNat_R_I32 hd
  refine ptr_eq_of_addr ?_
  have hg : ((ne_pY d y_h i_t)
        + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)).get (l, c, ())
      = ((ne_pY d y_h i_t).get (l, c, ())).offset (d.R : Int) := by
    simp only [g2_ptrAddT, hR]
  rw [hg, hp, Buf.ptr_offset]
  have hc : ((ne_gX0 d init i (l, c, ()) : Nat) : Int) + (d.R : Int)
      = ((ne_gX1 d init i (l, c, ()) : Nat) : Int) := by
    rw [ne_gX1_eq (d := d) (init := init) (i := i) l c]
    push_cast
    ring
  rw [hc]

/-! ## Index bounds -/

/-- The `cos`/`sin` element of a lane is inside the tables. -/
private theorem ne_gCS_lt {d : Inst} (_hd : d.ok) {init : Mem} {i : Nat}
    {len : Nat} (hlen : d.TR * d.R ≤ len)
    {l : Fin d.BT} {c : Fin (d.BD / 2)} (hlan : laneOn d init i l.val) (hc : c.val < d.R) :
    ne_gCS d init i (l, c, ()) < len := by
  have hR0 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
  have hclt : (c.val : Int) < (d.R : Int) := by exact_mod_cast hc
  have hpos0 := hlan.2.1
  have hposTR := hlan.2.2
  have hsum : posI d init i l.val * (d.R : Int) + (c.val : Int) < (d.TR : Int) * (d.R : Int) := by
    nlinarith
  have h0 : (0 : Int) ≤ posI d init i l.val * (d.R : Int) + (c.val : Int) := by nlinarith
  have hg : ((ne_gCS d init i (l, c, ()) : Nat) : Int)
      = posI d init i l.val * (d.R : Int) + (c.val : Int) := by
    simp only [ne_gCS]
    exact Int.toNat_of_nonneg h0
  have hlen' : ((d.TR * d.R : Nat) : Int) ≤ ((len : Nat) : Int) := by exact_mod_cast hlen
  have hlt : ((ne_gCS d init i (l, c, ()) : Nat) : Int) < ((len : Nat) : Int) := by
    rw [hg]
    push_cast at hlen' ⊢
    linarith
  exact_mod_cast hlt

/-- A lane's `x`/`y` element is inside the token's rows. -/
private theorem ne_gX0_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {len : Nat} (hlen : nTok d * d.H * d.D ≤ len)
    {l : Fin d.BT} {c : Fin (d.BD / 2)} (hlan : laneOn d init i l.val) (hc : c.val < d.R) :
    ne_gX0 d init i (l, c, ()) < len := by
  have ht := tokI_lt hd hpre hi hlan
  have hH := ax2_lt hi
  have hD : c.val < d.D := by
    have h2 := hd.r_d
    omega
  have hDpos : 0 < d.D := by
    have h1 := hd.r_pos; have h2 := hd.r_d; omega
  have hstep : tokI d init i l.val * d.H + ax2 d i + 1 ≤ nTok d * d.H := by
    have h1 : tokI d init i l.val * d.H + (ax2 d i + 1)
        ≤ tokI d init i l.val * d.H + d.H :=
      Nat.add_le_add_left (Nat.succ_le_of_lt hH) _
    have h2 : tokI d init i l.val * d.H + d.H = (tokI d init i l.val + 1) * d.H := by
      rw [Nat.add_mul, Nat.one_mul]
    have h3 : (tokI d init i l.val + 1) * d.H ≤ nTok d * d.H :=
      Nat.mul_le_mul_right _ ht
    have h4 : tokI d init i l.val * d.H + (ax2 d i + 1) ≤ nTok d * d.H :=
      le_trans (le_trans h1 (le_of_eq h2)) h3
    rw [Nat.add_assoc]
    exact h4
  have hlt : ne_gX0 d init i (l, c, ()) < nTok d * d.H * d.D := by
    show (tokI d init i l.val * d.H + ax2 d i) * d.D + c.val < nTok d * d.H * d.D
    have hmul := Nat.mul_le_mul_right d.D hstep
    have hone : (tokI d init i l.val * d.H + ax2 d i + 1) * d.D
        = (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D := by
      rw [Nat.add_mul, Nat.one_mul]
    have h6 : (tokI d init i l.val * d.H + ax2 d i) * d.D + c.val
        < (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := by
      calc (tokI d init i l.val * d.H + ax2 d i) * d.D + c.val
          < (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D :=
            Nat.add_lt_add_left hD _
        _ = (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := hone.symm
    exact lt_of_lt_of_le h6 hmul
  exact lt_of_lt_of_le hlt hlen

/-- A lane's partner element `c + R` is inside the token's rows. -/
private theorem ne_gX1_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {len : Nat} (hlen : nTok d * d.H * d.D ≤ len)
    {l : Fin d.BT} {c : Fin (d.BD / 2)} (hlan : laneOn d init i l.val) (hc : c.val < d.R) :
    ne_gX1 d init i (l, c, ()) < len := by
  have ht := tokI_lt hd hpre hi hlan
  have hH := ax2_lt hi
  have hRD : c.val + d.R < d.D := by
    have h2 := hd.r_d
    omega
  have hDpos : 0 < d.D := by
    have h1 := hd.r_pos; have h2 := hd.r_d; omega
  have hstep : tokI d init i l.val * d.H + ax2 d i + 1 ≤ nTok d * d.H := by
    have h1 : tokI d init i l.val * d.H + (ax2 d i + 1)
        ≤ tokI d init i l.val * d.H + d.H :=
      Nat.add_le_add_left (Nat.succ_le_of_lt hH) _
    have h2 : tokI d init i l.val * d.H + d.H = (tokI d init i l.val + 1) * d.H := by
      rw [Nat.add_mul, Nat.one_mul]
    have h3 : (tokI d init i l.val + 1) * d.H ≤ nTok d * d.H :=
      Nat.mul_le_mul_right _ ht
    have h4 : tokI d init i l.val * d.H + (ax2 d i + 1) ≤ nTok d * d.H :=
      le_trans (le_trans h1 (le_of_eq h2)) h3
    rw [Nat.add_assoc]
    exact h4
  have hlt : ne_gX1 d init i (l, c, ()) < nTok d * d.H * d.D := by
    show (tokI d init i l.val * d.H + ax2 d i) * d.D + (c.val + d.R)
      < nTok d * d.H * d.D
    have hmul := Nat.mul_le_mul_right d.D hstep
    have hone : (tokI d init i l.val * d.H + ax2 d i + 1) * d.D
        = (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D := by
      rw [Nat.add_mul, Nat.one_mul]
    have h6 : (tokI d init i l.val * d.H + ax2 d i) * d.D + (c.val + d.R)
        < (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := by
      calc (tokI d init i l.val * d.H + ax2 d i) * d.D + (c.val + d.R)
          < (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D :=
            Nat.add_lt_add_left hRD _
        _ = (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := hone.symm
    exact lt_of_lt_of_le h6 hmul
  exact lt_of_lt_of_le hlt hlen

/-- A token index is `c` modulo `D`. -/
private theorem tokIdx_mod {d : Inst} {t h c : Nat} (hc : c < d.D) :
    tokIdx d t h c % d.D = c := by
  show ((t * d.H + h) * d.D + c) % d.D = c
  rw [Nat.add_comm ((t * d.H + h) * d.D) c, Nat.add_mul_mod_self_right,
    Nat.mod_eq_of_lt hc]

/-! ## Lemmas for `safe_tail_ne` -/

/-- The `x` buffer's bytes are in the read set. -/
private theorem ne_x_sub_R (d : Inst) (init : Mem) (i : Nat) : d.x.bytes ⊆ R d init i := by
  intro z hz
  simp only [R, Set.mem_union]
  tauto

/-- The `cos` buffer's bytes are in the read set. -/
private theorem ne_cos_sub_R (d : Inst) (init : Mem) (i : Nat) : d.cos.bytes ⊆ R d init i := by
  intro z hz
  simp only [R, Set.mem_union]
  tauto

/-- The `sin` buffer's bytes are in the read set. -/
private theorem ne_sin_sub_R (d : Inst) (init : Mem) (i : Nat) : d.sin.bytes ⊆ R d init i := by
  intro z hz
  simp only [R, Set.mem_union]
  tauto

/-- The `cos`/`sin` index of a lane, in the form the table hypotheses use. -/
private theorem ne_gCS_eq {d : Inst} {init : Mem} {i : Nat} {k : Idx [d.BT, d.BD / 2]}
    (hpos : 0 ≤ posI d init i k.1.val) :
    (posI d init i k.1.val).toNat * d.R + k.2.1.val = ne_gCS d init i k := by
  have hc0 : (0 : Int) ≤ (k.2.1.val : Int) := by exact_mod_cast Nat.zero_le k.2.1.val
  have hR0 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
  have h0 : (0 : Int) ≤ posI d init i k.1.val * (d.R : Int) + (k.2.1.val : Int) := by
    nlinarith
  have h1 : (posI d init i k.1.val).toNat * d.R + k.2.1.val
      = (posI d init i k.1.val * (d.R : Int) + (k.2.1.val : Int)).toNat := by
    apply Nat.cast_injective (R := ℤ)
    push_cast
    rw [Int.toNat_of_nonneg hpos, Int.toNat_of_nonneg h0]
  rw [h1]
  rfl

/-- A lane's own element is `c` modulo `D`. -/
private theorem ne_gX0_mod {d : Inst} {init : Mem} {i : Nat} {k : Idx [d.BT, d.BD / 2]}
    (hc : k.2.1.val < d.D) : ne_gX0 d init i k % d.D = k.2.1.val := by
  show tokIdx d (tokI d init i k.1.val) (ax2 d i) k.2.1.val % d.D = k.2.1.val
  exact tokIdx_mod hc

/-- A lane's partner element is `c + R` modulo `D`. -/
private theorem ne_gX1_mod {d : Inst} {init : Mem} {i : Nat} {k : Idx [d.BT, d.BD / 2]}
    (hc : k.2.1.val + d.R < d.D) : ne_gX1 d init i k % d.D = k.2.1.val + d.R := by
  show tokIdx d (tokI d init i k.1.val) (ax2 d i) (k.2.1.val + d.R) % d.D = k.2.1.val + d.R
  exact tokIdx_mod hc

/-- An enabled lane is an active lane with `c < R`. -/
private theorem ne_lane_of_en {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20)
    {k : Idx [d.BT, d.BD / 2]} (hk : ne_en d i_t T_n o_cs k = true) :
    laneOn d init i k.1.val ∧ k.2.1.val < d.R :=
  (ne_mask_get hot hT hocs hTR hRlt hBD k).mp hk

/-- An active lane with `c < R` is enabled. -/
private theorem ne_en_of_lane {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20)
    {k : Idx [d.BT, d.BD / 2]} (hlan : laneOn d init i k.1.val) (hc : k.2.1.val < d.R) :
    ne_en d i_t T_n o_cs k = true :=
  (ne_mask_get hot hT hocs hTR hRlt hBD k).mpr ⟨hlan, hc⟩

/-- The token index separates lanes and channels. -/
private theorem ne_tokI_inj {d : Inst} (hd : d.ok) (hH : 0 < d.H) {init : Mem} {i : Nat}
    {l l' : Fin d.BT} {c c' : Nat} (hc : c < d.D) (hc' : c' < d.D)
    (h : tokIdx d (tokI d init i l.val) (ax2 d i) c
      = tokIdx d (tokI d init i l'.val) (ax2 d i) c') : l = l' ∧ c = c' := by
  have hDpos : 0 < d.D := by have h1 := hd.r_pos; have h2 := hd.r_d; omega
  have hcEq : c = c' := by
    have h1 : tokIdx d (tokI d init i l.val) (ax2 d i) c % d.D = c := tokIdx_mod hc
    rw [h] at h1
    rw [← h1, tokIdx_mod hc']
  have hstep : tokI d init i l.val = tokI d init i l'.val := by
    have hh : (tokI d init i l.val * d.H + ax2 d i) * d.D
        = (tokI d init i l'.val * d.H + ax2 d i) * d.D := by
      rw [hcEq] at h
      exact Nat.add_right_cancel h
    have hh2 : tokI d init i l.val * d.H + ax2 d i
        = tokI d init i l'.val * d.H + ax2 d i := Nat.mul_right_cancel hDpos hh
    have hh3 : tokI d init i l.val * d.H = tokI d init i l'.val * d.H :=
      Nat.add_right_cancel hh2
    exact Nat.mul_right_cancel hH hh3
  have hl : l = l' := by
    have h1 : locI d init i l.val = locI d init i l'.val := by
      have h2 : bosI d init i + locI d init i l.val = bosI d init i + locI d init i l'.val :=
        hstep
      exact Nat.add_left_cancel h2
    have h3 : l.val = l'.val := by
      have h4 : chunkI d init i * d.BT + l.val = chunkI d init i * d.BT + l'.val := h1
      exact Nat.add_left_cancel h4
    exact Fin.ext h3
  exact ⟨hl, hcEq⟩

/-- Two enabled lanes with the same own element are the same lane. -/
private theorem ne_gX0_inj {d : Inst} (hd : d.ok) (hH : 0 < d.H) {init : Mem} {i : Nat}
    {i_t : I64} {T_n : I64} {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20)
    {k k' : Idx [d.BT, d.BD / 2]} (hk : ne_en d i_t T_n o_cs k = true)
    (hk' : ne_en d i_t T_n o_cs k' = true) (he : ne_gX0 d init i k = ne_gX0 d init i k') :
    k = k' := by
  obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk
  obtain ⟨hlan', hcR'⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk'
  have hD1 : k.2.1.val < d.D := by have h2 := hd.r_d; omega
  have hD2 : k'.2.1.val < d.D := by have h2 := hd.r_d; omega
  obtain ⟨hl, hc⟩ := ne_tokI_inj hd hH hD1 hD2 he
  have hc2 : k.2.1 = k'.2.1 := Fin.ext hc
  rw [show k = (k.1, k.2.1, ()) from rfl, show k' = (k'.1, k'.2.1, ()) from rfl, hl, hc2]

/-- Two enabled lanes with the same partner element are the same lane. -/
private theorem ne_gX1_inj {d : Inst} (hd : d.ok) (hH : 0 < d.H) {init : Mem} {i : Nat}
    {i_t : I64} {T_n : I64} {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20)
    {k k' : Idx [d.BT, d.BD / 2]} (hk : ne_en d i_t T_n o_cs k = true)
    (hk' : ne_en d i_t T_n o_cs k' = true) (he : ne_gX1 d init i k = ne_gX1 d init i k') :
    k = k' := by
  obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk
  obtain ⟨hlan', hcR'⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk'
  have hD1 : k.2.1.val + d.R < d.D := by have h2 := hd.r_d; omega
  have hD2 : k'.2.1.val + d.R < d.D := by have h2 := hd.r_d; omega
  obtain ⟨hl, hc⟩ := ne_tokI_inj hd hH hD1 hD2 he
  have hc2 : k.2.1 = k'.2.1 := Fin.ext (Nat.add_right_cancel hc)
  rw [show k = (k.1, k.2.1, ()) from rfl, show k' = (k'.1, k'.2.1, ()) from rfl, hl, hc2]

/-- The own elements the program touches are in the output set. -/
private theorem ne_imgX0_sub {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20) :
    Buf.img (ne_gX0 d init i) (ne_en d i_t T_n o_cs)
      ⊆ {k | ∃ l c, l < d.BT ∧ c < 2 * d.R ∧ laneOn d init i l ∧
          k = tokIdx d (tokI d init i l) (ax2 d i) c} := by
  rintro z ⟨k, hk, rfl⟩
  obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk
  exact ⟨k.1.val, k.2.1.val, k.1.isLt, by omega, hlan, rfl⟩

/-- The partner elements the program touches are in the output set. -/
private theorem ne_imgX1_sub {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20) :
    Buf.img (ne_gX1 d init i) (ne_en d i_t T_n o_cs)
      ⊆ {k | ∃ l c, l < d.BT ∧ c < 2 * d.R ∧ laneOn d init i l ∧
          k = tokIdx d (tokI d init i l) (ax2 d i) c} := by
  rintro z ⟨k, hk, rfl⟩
  obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk
  refine ⟨k.1.val, k.2.1.val + d.R, k.1.isLt, by omega, hlan, ?_⟩
  show tokIdx d (tokI d init i k.1.val) (ax2 d i) (k.2.1.val + d.R)
    = tokIdx d (tokI d init i k.1.val) (ax2 d i) (k.2.1.val + d.R)
  rfl

/-- The own and partner element sets of one token are disjoint. -/
private theorem ne_img_disjoint {d : Inst} (hd : d.ok) {init : Mem} {i : Nat} {i_t : I64}
    {T_n : I64} {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (hRlt : d.R < 2 ^ 31) (hBD : d.BD ≤ 2 ^ 20) :
    Disjoint (Buf.img (ne_gX1 d init i) (ne_en d i_t T_n o_cs))
      (Buf.img (ne_gX0 d init i) (ne_en d i_t T_n o_cs)) := by
  refine Set.disjoint_left.mpr ?_
  rintro z ⟨k, hk, rfl⟩ ⟨k', hk', he⟩
  obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk
  obtain ⟨hlan', hcR'⟩ := ne_lane_of_en hot hT hocs hTR hRlt hBD hk'
  have hD1 : k.2.1.val + d.R < d.D := by have h2 := hd.r_d; omega
  have hD2 : k'.2.1.val < d.D := by have h2 := hd.r_d; omega
  have e1 : ne_gX0 d init i k' % d.D = k'.2.1.val := ne_gX0_mod hD2
  have e2 : ne_gX1 d init i k % d.D = k.2.1.val + d.R := ne_gX1_mod hD1
  rw [← he] at e2
  omega

/-- A read from a buffer is unchanged by a write that avoids its elements. -/
private theorem ne_read_frame {b : Buf .bf16} {m m1 : Mem} {S : Set Nat} {j : Nat}
    (hdisj : ∀ x, x ∉ b.elems S → m1 x = m x) (hj : j ∉ S) :
    bfAt b m1 j = bfAt b m j := by
  refine Mem.read_congr ?_
  intro x hx
  have hx1 : x ∈ b.elems ({j} : Set Nat) :=
    Buf.span_sub_elems b (S := ({j} : Set Nat)) (by simp) hx
  have hdis : Disjoint (b.elems ({j} : Set Nat)) (b.elems S) :=
    Buf.elems_disjoint b (Set.disjoint_singleton_left.mpr hj)
  exact hdisj x (fun hc => Set.disjoint_left.mp hdis hx1 hc)

/-- A `bf16` value converted to `f32` keeps its value. -/
private theorem ne_bf2f {x : BF16} {y : F32} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) y)
    (hx : x.isFinite = true) : y.isFinite = true ∧ y.toRat = x.toRat := by
  have hw := FloatV.toRat_cvt_of_wider (g := Format.binary32)
    Format.bfloat16_regular Format.binary32_regular
    Format.wider_bfloat16_binary32 .rn hx
  have hnan : (FloatV.cvtWith .rn x : FloatV Format.binary32).isNaN = false :=
    (FloatV.isFinite_iff.mp hw.1).1
  have heq : y = (FloatV.cvtWith .rn x : FloatV Format.binary32) := h.eq hnan
  rw [heq]
  exact hw

/-- **The non-interleaved tail is safe**: from the prologue's state and the offset line's `o_cs`, the loads, the rotation and the store of the half-split layout end with `Q`, reading within `R` and writing within `W`. -/
theorem safe_tail_ne {P : Platform} {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size)
    {i_t : I64} {x_h y_h : Ptr .bf16} {T_n : I64} {o_cs : Tensor [d.BT] I64}
    (hit : i_t.toInt = (chunkI d init i : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hx : x_h.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (hy : y_h.addr = (d.y.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    (hint : d.interleaved = false) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      (do
        let o_t : Tensor [d.BT] I64 := i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)
        let m_t : Tensor [d.BT] Bool :=
          (((o_t.ltS (IntV.ofInt 0 : I64)).not.and (o_t.ltS T_n)).and
            (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64))
        let o_r : Tensor [d.BD / 2] I32 := tl.arange 0 (d.BD / 2)
        let p_x : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
          x_h + o_t.col (d.BD / 2) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + o_r.row d.BT
        let p_cos : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
          d.cos.ptr 0 + (o_cs.col (d.BD / 2) * (IntV.ofNat d.R : I64) + ((o_r.row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))
        let p_sin : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
          d.sin.ptr 0 + (o_cs.col (d.BD / 2) * (IntV.ofNat d.R : I64) + ((o_r.row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))
        let mask : Tensor [d.BT, d.BD / 2] Bool :=
          (m_t.col (d.BD / 2)).and ((o_r.ltS (IntV.ofNat d.R : I32)).row d.BT)
        let one_bf : Tensor [d.BT, d.BD / 2] BF16 ←
          tl.convert .f32 .bf16 (tl.full [d.BT, d.BD / 2] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
        let l_cos ← tl.load p_cos (some mask) (some one_bf)
        let b_cos : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_cos
        let zero_bf : Tensor [d.BT, d.BD / 2] BF16 ←
          tl.convert .f32 .bf16 (tl.full [d.BT, d.BD / 2] (FloatV.litF32 (FloatV.ofPat 0)))
        let l_sin ← tl.load p_sin (some mask) (some zero_bf)
        let b_sin : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_sin
        let l_x0 ← tl.load p_x (some mask) (some zero_bf)
        let b_x0 : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_x0
        let l_x1 ← tl.load (p_x + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)) (some mask) (some zero_bf)
        let b_x1 : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_x1
        let b_o0 ← tl.subc (← tl.mul b_x0 b_cos) (← tl.mul b_x1 b_sin)
        let b_o1 ← tl.addc (← tl.mul b_x0 b_sin) (← tl.mul b_x1 b_cos)
        let p_y : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
          y_h + (o_t.col (d.BD / 2) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) +
            ((o_r.row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))
        let s_o0 : Tensor [d.BT, d.BD / 2] BF16 ← tl.cvt b_o0
        tl.store p_y s_o0 (some mask)
        let s_o1 : Tensor [d.BT, d.BD / 2] BF16 ← tl.cvt b_o1
        tl.store (p_y + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)) s_o1 (some mask)) := by
  have hDpos : 0 < d.D := by
    have h1 := hd.r_pos
    have h2 := hd.r_d
    omega
  have hHpos : 0 < d.H := by
    have hsize : 0 < (launch d).size := lt_of_le_of_lt (Nat.zero_le i) hi
    rw [launch_size] at hsize
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h, Nat.mul_zero] at hsize
      exact absurd hsize (Nat.lt_irrefl 0)
    · exact h
  have hot : ∀ k : Idx [d.BT],
      ((ne_oT d i_t).get k).toInt = (locI d init i k.1.val : Int) := by
    intro k
    obtain ⟨l, u⟩ := k
    cases u
    exact oT_toInt hd hpre hi hit l
  have hocs' : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val := by
    intro k
    obtain ⟨l, u⟩ := k
    cases u
    exact hocs l
  have hTR : d.TR < 2 ^ 63 := ne_TR_lt63 hd
  have hRlt : d.R < 2 ^ 31 := ne_R_lt hd
  have hBD : d.BD ≤ 2 ^ 20 := bd_le hd
  refine safe_convert ?_ ?_
  · intro idx
    refine ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)), ?_⟩
    have hg : (tl.full [d.BT, d.BD / 2]
        (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))).get idx
        = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := rfl
    rw [hg]
    show FloatRel P Format.binary32 Format.bfloat16
      (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))
      (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
    exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩
  · intro one_bf hone
    refine safe_load_buf (g := ne_gCS d init i) (en := ne_en d i_t T_n o_cs) d.cos ?_ ?_ hpre.cosv ?_ ?_
    · refine Buf.addrs_of_get d.cos ?_ ?_
      · intro k
        rfl
      · intro k hk
        obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
        rw [show k = (k.1, k.2.1, ()) from rfl]
        exact ne_pCos_get hd hocs hlan hcR
    · intro k hk
      obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
      exact ne_gCS_lt hd hd.cos_len hlan hcR
    · refine Set.Subset.trans (Buf.elems_sub_bytes d.cos ?_) (ne_cos_sub_R d init i)
      intro j hj
      obtain ⟨k, hk, rfl⟩ := hj
      obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
      exact ne_gCS_lt hd hd.cos_len hlan hcR
    · intro l_cos hcos_read hcos_off
      refine safe_cvt ?_ ?_
      · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
      · intro b_cos hb_cos
        refine safe_convert ?_ ?_
        · intro idx
          refine ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0)), ?_⟩
          have hg : (tl.full [d.BT, d.BD / 2] (FloatV.litF32 (FloatV.ofPat 0))).get idx
              = FloatV.litF32 (FloatV.ofPat 0) := rfl
          rw [hg]
          show FloatRel P Format.binary32 Format.bfloat16
            (FloatV.litF32 (FloatV.ofPat 0))
            (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0)))
          exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩
        · intro zero_bf hzero
          refine safe_load_buf (g := ne_gCS d init i) (en := ne_en d i_t T_n o_cs) d.sin ?_ ?_ hpre.sinv ?_ ?_
          · refine Buf.addrs_of_get d.sin ?_ ?_
            · intro k
              rfl
            · intro k hk
              obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
              rw [show k = (k.1, k.2.1, ()) from rfl]
              exact ne_pSin_get hd hocs hlan hcR
          · intro k hk
            obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
            exact ne_gCS_lt hd hd.sin_len hlan hcR
          · refine Set.Subset.trans (Buf.elems_sub_bytes d.sin ?_) (ne_sin_sub_R d init i)
            intro j hj
            obtain ⟨k, hk, rfl⟩ := hj
            obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
            exact ne_gCS_lt hd hd.sin_len hlan hcR
          · intro l_sin hsin_read hsin_off
            refine safe_cvt ?_ ?_
            · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
            · intro b_sin hb_sin
              refine safe_load_buf (g := ne_gX0 d init i) (en := ne_en d i_t T_n o_cs) d.x ?_ ?_ hpre.xv ?_ ?_
              · refine Buf.addrs_of_get d.x ?_ ?_
                · intro k
                  rfl
                · intro k hk
                  obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                  rw [show k = (k.1, k.2.1, ()) from rfl]
                  exact ne_pX_get hd hpre hi hit hx k.1 k.2.1
              · intro k hk
                obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                exact ne_gX0_lt hd hpre hi hd.x_len hlan hcR
              · refine Set.Subset.trans (Buf.elems_sub_bytes d.x ?_) (ne_x_sub_R d init i)
                intro j hj
                obtain ⟨k, hk, rfl⟩ := hj
                obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                exact ne_gX0_lt hd hpre hi hd.x_len hlan hcR
              · intro l_x0 hx0_read hx0_off
                refine safe_cvt ?_ ?_
                · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
                · intro b_x0 hb_x0
                  refine safe_load_buf (g := ne_gX1 d init i) (en := ne_en d i_t T_n o_cs) d.x ?_ ?_ hpre.xv ?_ ?_
                  · refine Buf.addrs_of_get d.x ?_ ?_
                    · intro k
                      rfl
                    · intro k hk
                      obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                      rw [show k = (k.1, k.2.1, ()) from rfl]
                      exact ne_pX1_get hd hpre hi hit hx k.1 k.2.1
                  · intro k hk
                    obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                    exact ne_gX1_lt hd hpre hi hd.x_len hlan hcR
                  · refine Set.Subset.trans (Buf.elems_sub_bytes d.x ?_) (ne_x_sub_R d init i)
                    intro j hj
                    obtain ⟨k, hk, rfl⟩ := hj
                    obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                    exact ne_gX1_lt hd hpre hi hd.x_len hlan hcR
                  · intro l_x1 hx1_read hx1_off
                    refine safe_cvt ?_ ?_
                    · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
                    · intro b_x1 hb_x1
                      refine safe_mul ?_
                      · intro pa hpa
                        refine safe_mul ?_
                        · intro pb hpb
                          refine safe_subc ?_
                          · intro b_o0 hb_o0
                            refine safe_mul ?_
                            · intro pc hpc
                              refine safe_mul ?_
                              · intro pd hpd
                                refine safe_addc ?_
                                · intro b_o1 hb_o1
                                  refine safe_cvt ?_ ?_
                                  · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩
                                  · intro s_o0 hs_o0
                                    refine safe_store_buf (g := ne_gX0 d init i) (en := ne_en d i_t T_n o_cs) d.y ?_ ?_ ?_ hpre.yv ?_ ?_
                                    · refine Buf.addrs_of_get d.y ?_ ?_
                                      · intro k
                                        rfl
                                      · intro k hk
                                        obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                                        rw [show k = (k.1, k.2.1, ()) from rfl]
                                        exact ne_pY_get hd hpre hi hit hy k.1 k.2.1
                                    · intro k hk
                                      obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                                      exact ne_gX0_lt hd hpre hi hd.y_len hlan hcR
                                    · intro k1 k2 hk1 hk2 he
                                      exact ne_gX0_inj hd hHpos hot hT hocs' hTR hRlt hBD hk1 hk2 he
                                    · refine Set.Subset.trans (Buf.elems_mono d.y (ne_imgX0_sub hot hT hocs' hTR hRlt hBD)) ?_
                                      intro x hx
                                      exact hx
                                    · intro m' hread' hdisj' hpres'
                                      refine safe_cvt ?_ ?_
                                      · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩
                                      · intro s_o1 hs_o1
                                        refine safe_last ?_
                                        refine safe_store_buf (g := ne_gX1 d init i) (en := ne_en d i_t T_n o_cs) d.y ?_ ?_ ?_ ?_ ?_ ?_
                                        · refine Buf.addrs_of_get d.y ?_ ?_
                                          · intro k
                                            rfl
                                          · intro k hk
                                            obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                                            rw [show k = (k.1, k.2.1, ()) from rfl]
                                            exact ne_pY1_get hd hpre hi hit hy k.1 k.2.1
                                        · intro k hk
                                          obtain ⟨hlan, hcR⟩ := ne_lane_of_en hot hT hocs' hTR hRlt hBD hk
                                          exact ne_gX1_lt hd hpre hi hd.y_len hlan hcR
                                        · intro k1 k2 hk1 hk2 he
                                          exact ne_gX1_inj hd hHpos hot hT hocs' hTR hRlt hBD hk1 hk2 he
                                        · exact Buf.valid_of_agree hpre.yv
                                            (fun x hx => hpres' x (hpre.yv.2.2.2 x hx))
                                        · refine Set.subset_sdiff.mpr ⟨Buf.elems_mono d.y (ne_imgX1_sub hot hT hocs' hTR hRlt hBD), ?_⟩
                                          exact Buf.elems_disjoint d.y (ne_img_disjoint hd hot hT hocs' hTR hRlt hBD)
                                        · intro m'' hread'' hdisj'' hpres''
                                          refine safe_ret ?_
                                          intro l c hl hc2R hlan
                                          have hseq : seqI d init i < nSeq d := seqI_lt hd hpre hi
                                          have hloc : locI d init i l < lenAt d init (seqI d init i) := by
                                            have h := hlan.1
                                            simp only [lenI] at h
                                            exact_mod_cast h
                                          have hoff : 0 ≤ (locI d init i l : Int) + offAt d init (seqI d init i) :=
                                            hlan.2.1
                                          have hD : c < d.D := by
                                            have h2 := hd.r_d
                                            omega
                                          by_cases hcR : c < d.R
                                          · have hpair : pairOf d c = c := by simp [pairOf, hint, hcR]
                                            have hpart : partner d c = c + d.R := by
                                              simp [partner, hint, hcR]
                                            have hisF : isFirst d c = true := by
                                              simp [isFirst, hint, hcR]
                                            have hcRD : c + d.R < d.D := by
                                              have h2 := hd.r_d
                                              omega
                                            have hcBD : c < d.BD / 2 := by
                                              have h2 := hd.r_bd
                                              omega
                                            let kk : Idx [d.BT, d.BD / 2] := (⟨l, hl⟩, ⟨c, hcBD⟩, ())
                                            have hen : ne_en d i_t T_n o_cs kk = true :=
                                              ne_en_of_lane hot hT hocs' hTR hRlt hBD hlan hcR
                                            have hcosr := hcos_read kk hen
                                            have hsinr := hsin_read kk hen
                                            have hx0r := hx0_read kk hen
                                            have hx1r := hx1_read kk hen
                                            have hmod0 : ne_gX0 d init i kk % d.D = c := by
                                              show tokIdx d (tokI d init i l) (ax2 d i) c % d.D = c
                                              exact tokIdx_mod hD
                                            have hmod1 : ne_gX1 d init i kk % d.D = c + d.R := by
                                              show tokIdx d (tokI d init i l) (ax2 d i) (c + d.R) % d.D
                                                = c + d.R
                                              exact tokIdx_mod hcRD
                                            obtain ⟨a, ha_mem, ha_ok⟩ :=
                                              hpre.x_vals (ne_gX0 d init i kk)
                                                (ne_gX0_lt hd hpre hi (len := nTok d * d.H * d.D)
                                                  le_rfl hlan hcR)
                                                (by rw [hmod0]; exact hc2R)
                                            obtain ⟨b, hb_mem, hb_ok⟩ :=
                                              hpre.x_vals (ne_gX1 d init i kk)
                                                (ne_gX1_lt hd hpre hi (len := nTok d * d.H * d.D)
                                                  le_rfl hlan hcR)
                                                (by rw [hmod1]; omega)
                                            obtain ⟨cs, hcs_mem, hcs_ok⟩ :=
                                              hpre.cos_vals (seqI d init i) (locI d init i l) c
                                                hseq hloc hoff hcR
                                            obtain ⟨sn, hsn_mem, hsn_ok⟩ :=
                                              hpre.sin_vals (seqI d init i) (locI d init i l) c
                                                hseq hloc hoff hcR
                                            have hcsidx : (posI d init i l).toNat * d.R + c
                                                = ne_gCS d init i kk :=
                                              ne_gCS_eq (d := d) (init := init) (i := i) (k := kk)
                                                hlan.2.1
                                            have hcsm : bfAt d.cos init (ne_gCS d init i kk) = some cs := by
                                              rw [← hcsidx]
                                              exact hcs_mem
                                            have hsnm : bfAt d.sin init (ne_gCS d init i kk) = some sn := by
                                              rw [← hcsidx]
                                              exact hsn_mem
                                            have hal : l_x0.get kk = a :=
                                              Option.some.inj (hx0r.symm.trans ha_mem)
                                            have hbl : l_x1.get kk = b :=
                                              Option.some.inj (hx1r.symm.trans hb_mem)
                                            have hcl : l_cos.get kk = cs :=
                                              Option.some.inj (hcosr.symm.trans hcsm)
                                            have hsl : l_sin.get kk = sn :=
                                              Option.some.inj (hsinr.symm.trans hsnm)
                                            have hx0fin : (l_x0.get kk).isFinite = true := by
                                              rw [hal]; exact ha_ok.1
                                            have hx1fin : (l_x1.get kk).isFinite = true := by
                                              rw [hbl]; exact hb_ok.1
                                            have hcosfin : (l_cos.get kk).isFinite = true := by
                                              rw [hcl]; exact hcs_ok.1
                                            have hsinfin : (l_sin.get kk).isFinite = true := by
                                              rw [hsl]; exact hsn_ok.1
                                            have haf := ne_bf2f (x := l_x0.get kk) (y := b_x0.get kk)
                                              (hb_x0 kk) hx0fin
                                            have hbf := ne_bf2f (x := l_x1.get kk) (y := b_x1.get kk)
                                              (hb_x1 kk) hx1fin
                                            have hcf := ne_bf2f (x := l_cos.get kk) (y := b_cos.get kk)
                                              (hb_cos kk) hcosfin
                                            have hsf := ne_bf2f (x := l_sin.get kk) (y := b_sin.get kk)
                                              (hb_sin kk) hsinfin
                                            have ha_ok' : XOk (l_x0.get kk) := by
                                              rw [hal]; exact ha_ok
                                            have hb_ok' : XOk (l_x1.get kk) := by
                                              rw [hbl]; exact hb_ok
                                            have hcs_ok' : TabOk (l_cos.get kk) := by
                                              rw [hcl]; exact hcs_ok
                                            have hsn_ok' : TabOk (l_sin.get kk) := by
                                              rw [hsl]; exact hsn_ok
                                            have hres := tol_sub (a := l_x0.get kk) (b := l_x1.get kk)
                                              (cs := l_cos.get kk) (sn := l_sin.get kk)
                                              (af := b_x0.get kk) (bf := b_x1.get kk)
                                              (cf := b_cos.get kk) (sf := b_sin.get kk)
                                              (p := pa.get kk) (q := pb.get kk) (r := b_o0.get kk)
                                              (v := s_o0.get kk)
                                              haf.2 hbf.2 hcf.2 hsf.2 haf.1 hbf.1 hcf.1 hsf.1
                                              (hpa kk) (hpb kk) (hb_o0 kk) (hs_o0 kk)
                                              ha_ok' hb_ok' hcs_ok' hsn_ok'
                                            rw [hal, hbl, hcl, hsl] at hres
                                            have ha_g : bfAt d.x init (tokIdx d (tokI d init i l) (ax2 d i) c)
                                                = some a := ha_mem
                                            have hb_g : bfAt d.x init
                                                (tokIdx d (tokI d init i l) (ax2 d i) (partner d c))
                                                = some b := by
                                              rw [hpart]
                                              exact hb_mem
                                            have hcs_g : bfAt d.cos init
                                                ((posI d init i l).toNat * d.R + pairOf d c) = some cs := by
                                              rw [hpair, hcsidx]
                                              exact hcsm
                                            have hsn_g : bfAt d.sin init
                                                ((posI d init i l).toNat * d.R + pairOf d c) = some sn := by
                                              rw [hpair, hcsidx]
                                              exact hsnm
                                            have hy_g : bfAt d.y m''
                                                (tokIdx d (tokI d init i l) (ax2 d i) c)
                                                = some (s_o0.get kk) := by
                                              have hjm : ne_gX0 d init i kk ∉
                                                  Buf.img (ne_gX1 d init i) (ne_en d i_t T_n o_cs) := by
                                                intro hmem
                                                exact (Set.disjoint_left.mp
                                                  (ne_img_disjoint hd hot hT hocs' hTR hRlt hBD) hmem)
                                                  ⟨kk, hen, rfl⟩
                                              exact (ne_read_frame (b := d.y) (m := m') (m1 := m'')
                                                (S := Buf.img (ne_gX1 d init i) (ne_en d i_t T_n o_cs))
                                                (j := ne_gX0 d init i kk) hdisj'' hjm).trans (hread' kk hen)
                                            have hres2 : |(s_o0.get kk).toRat -
                                                rotRef (isFirst d c) a.toRat b.toRat cs.toRat sn.toRat|
                                                ≤ tol (isFirst d c) a.toRat b.toRat cs.toRat sn.toRat := by
                                              rw [hisF]
                                              exact hres.2
                                            unfold CellOk
                                            exact ⟨a, b, cs, sn, s_o0.get kk, ha_g, hb_g, hcs_g,
                                              hsn_g, hy_g, hres.1, hres2⟩
                                          · have hRle : d.R ≤ c := le_of_not_gt hcR
                                            have hpair : pairOf d c = c - d.R := by
                                              simp [pairOf, hint, hcR]
                                            have hpart : partner d c = c - d.R := by
                                              simp [partner, hint, hcR]
                                            have hisF : isFirst d c = false := by
                                              simp [isFirst, hint, hcR]
                                            have hcsub : c - d.R < d.R := by omega
                                            have hc2D : c - d.R < d.D := by
                                              have h2 := hd.r_d
                                              omega
                                            have hcBD : c - d.R < d.BD / 2 := by
                                              have h2 := hd.r_bd
                                              omega
                                            let kk : Idx [d.BT, d.BD / 2] :=
                                              (⟨l, hl⟩, ⟨c - d.R, hcBD⟩, ())
                                            have hen : ne_en d i_t T_n o_cs kk = true :=
                                              ne_en_of_lane hot hT hocs' hTR hRlt hBD hlan hcsub
                                            have hcosr := hcos_read kk hen
                                            have hsinr := hsin_read kk hen
                                            have hx0r := hx0_read kk hen
                                            have hx1r := hx1_read kk hen
                                            have hmod0 : ne_gX0 d init i kk % d.D = c - d.R := by
                                              show tokIdx d (tokI d init i l) (ax2 d i) (c - d.R)
                                                % d.D = c - d.R
                                              exact tokIdx_mod hc2D
                                            have hmod1 : ne_gX1 d init i kk % d.D = c := by
                                              show tokIdx d (tokI d init i l) (ax2 d i) (c - d.R + d.R)
                                                % d.D = c
                                              rw [Nat.sub_add_cancel hRle]
                                              exact tokIdx_mod hD
                                            obtain ⟨a, ha_mem, ha_ok⟩ :=
                                              hpre.x_vals (ne_gX1 d init i kk)
                                                (ne_gX1_lt hd hpre hi (len := nTok d * d.H * d.D)
                                                  le_rfl hlan hcsub)
                                                (by rw [hmod1]; exact hc2R)
                                            obtain ⟨b, hb_mem, hb_ok⟩ :=
                                              hpre.x_vals (ne_gX0 d init i kk)
                                                (ne_gX0_lt hd hpre hi (len := nTok d * d.H * d.D)
                                                  le_rfl hlan hcsub)
                                                (by rw [hmod0]; omega)
                                            obtain ⟨cs, hcs_mem, hcs_ok⟩ :=
                                              hpre.cos_vals (seqI d init i) (locI d init i l) (c - d.R)
                                                hseq hloc hoff hcsub
                                            obtain ⟨sn, hsn_mem, hsn_ok⟩ :=
                                              hpre.sin_vals (seqI d init i) (locI d init i l) (c - d.R)
                                                hseq hloc hoff hcsub
                                            have hcsidx : (posI d init i l).toNat * d.R + (c - d.R)
                                                = ne_gCS d init i kk :=
                                              ne_gCS_eq (d := d) (init := init) (i := i) (k := kk)
                                                hlan.2.1
                                            have hcsm : bfAt d.cos init (ne_gCS d init i kk) = some cs := by
                                              rw [← hcsidx]
                                              exact hcs_mem
                                            have hsnm : bfAt d.sin init (ne_gCS d init i kk) = some sn := by
                                              rw [← hcsidx]
                                              exact hsn_mem
                                            have hal : l_x1.get kk = a :=
                                              Option.some.inj (hx1r.symm.trans ha_mem)
                                            have hbl : l_x0.get kk = b :=
                                              Option.some.inj (hx0r.symm.trans hb_mem)
                                            have hcl : l_cos.get kk = cs :=
                                              Option.some.inj (hcosr.symm.trans hcsm)
                                            have hsl : l_sin.get kk = sn :=
                                              Option.some.inj (hsinr.symm.trans hsnm)
                                            have hx1fin : (l_x1.get kk).isFinite = true := by
                                              rw [hal]; exact ha_ok.1
                                            have hx0fin : (l_x0.get kk).isFinite = true := by
                                              rw [hbl]; exact hb_ok.1
                                            have hcosfin : (l_cos.get kk).isFinite = true := by
                                              rw [hcl]; exact hcs_ok.1
                                            have hsinfin : (l_sin.get kk).isFinite = true := by
                                              rw [hsl]; exact hsn_ok.1
                                            have haf := ne_bf2f (x := l_x1.get kk) (y := b_x1.get kk)
                                              (hb_x1 kk) hx1fin
                                            have hbf := ne_bf2f (x := l_x0.get kk) (y := b_x0.get kk)
                                              (hb_x0 kk) hx0fin
                                            have hcf := ne_bf2f (x := l_cos.get kk) (y := b_cos.get kk)
                                              (hb_cos kk) hcosfin
                                            have hsf := ne_bf2f (x := l_sin.get kk) (y := b_sin.get kk)
                                              (hb_sin kk) hsinfin
                                            have ha_ok' : XOk (l_x1.get kk) := by
                                              rw [hal]; exact ha_ok
                                            have hb_ok' : XOk (l_x0.get kk) := by
                                              rw [hbl]; exact hb_ok
                                            have hcs_ok' : TabOk (l_cos.get kk) := by
                                              rw [hcl]; exact hcs_ok
                                            have hsn_ok' : TabOk (l_sin.get kk) := by
                                              rw [hsl]; exact hsn_ok
                                            have hres := tol_add (a := l_x1.get kk) (b := l_x0.get kk)
                                              (cs := l_cos.get kk) (sn := l_sin.get kk)
                                              (af := b_x1.get kk) (bf := b_x0.get kk)
                                              (cf := b_cos.get kk) (sf := b_sin.get kk)
                                              (p := pc.get kk) (q := pd.get kk) (r := b_o1.get kk)
                                              (v := s_o1.get kk)
                                              haf.2 hbf.2 hcf.2 hsf.2 haf.1 hbf.1 hcf.1 hsf.1
                                              (hpc kk) (hpd kk) (hb_o1 kk) (hs_o1 kk)
                                              ha_ok' hb_ok' hcs_ok' hsn_ok'
                                            rw [hal, hbl, hcl, hsl] at hres
                                            have hidx : tokIdx d (tokI d init i l) (ax2 d i) c
                                                = ne_gX1 d init i kk := by
                                              show tokIdx d (tokI d init i l) (ax2 d i) c
                                                = tokIdx d (tokI d init i l) (ax2 d i) (c - d.R + d.R)
                                              rw [Nat.sub_add_cancel hRle]
                                            have ha_g : bfAt d.x init (tokIdx d (tokI d init i l) (ax2 d i) c)
                                                = some a := by
                                              rw [hidx]
                                              exact ha_mem
                                            have hb_g : bfAt d.x init
                                                (tokIdx d (tokI d init i l) (ax2 d i) (partner d c))
                                                = some b := by
                                              rw [hpart]
                                              exact hb_mem
                                            have hcs_g : bfAt d.cos init
                                                ((posI d init i l).toNat * d.R + pairOf d c) = some cs := by
                                              rw [hpair, hcsidx]
                                              exact hcsm
                                            have hsn_g : bfAt d.sin init
                                                ((posI d init i l).toNat * d.R + pairOf d c) = some sn := by
                                              rw [hpair, hcsidx]
                                              exact hsnm
                                            have hy_g : bfAt d.y m''
                                                (tokIdx d (tokI d init i l) (ax2 d i) c)
                                                = some (s_o1.get kk) := by
                                              rw [hidx]
                                              exact hread'' kk hen
                                            have hres2 : |(s_o1.get kk).toRat -
                                                rotRef (isFirst d c) a.toRat b.toRat cs.toRat sn.toRat|
                                                ≤ tol (isFirst d c) a.toRat b.toRat cs.toRat sn.toRat := by
                                              rw [hisF]
                                              exact hres.2
                                            unfold CellOk
                                            exact ⟨a, b, cs, sn, s_o1.get kk, ha_g, hb_g, hcs_g,
                                              hsn_g, hy_g, hres.1, hres2⟩

end Claims.RotaryVarlen
