import Claim
import Proof.Basic

/-!
# `attn_bwd_pre_f16`: the addresses

`off_m`, `off_n`, `off_hz` are the model's offsets with the program ids substituted
(`launch_pid0`, `launch_pid1`), `ptrO`, `ptrDO`, `ptrD` the three pointers of the body, and
`readIdx`, `writeIdx` the element each lane touches.
-/

namespace Claims.AttnBwdPre

open TVA


/-- `off_m` of program `i`. -/
def offm (d : Inst) (i : Nat) : Tensor [d.BLOCK_M] I32 :=
  IntV.ofNat (pm d i) * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M

/-- `off_n` (the same for every program). -/
def offn (d : Inst) : Tensor [d.HEAD_DIM] I32 := tl.arange 0 d.HEAD_DIM

/-- `off_hz` of program `i`. -/
def offhz (d : Inst) (i : Nat) : I32 := IntV.ofNat (bh d i)

/-- `off_hz · HEAD_DIM · N_CTX`. -/
def hzrow (d : Inst) (i : Nat) : I32 :=
  offhz d i * (IntV.ofNat d.HEAD_DIM : I32) * IntV.ofNat d.N_CTX

/-- `off_m · HEAD_DIM` as a `[BLOCK_M, HEAD_DIM]` offset. -/
def mcol (d : Inst) (i : Nat) : Tensor [d.BLOCK_M, d.HEAD_DIM] I32 :=
  (offm d i).col d.HEAD_DIM * (IntV.ofNat d.HEAD_DIM : I32)

/-- `off_n` broadcast down the rows. -/
def nrow (d : Inst) : Tensor [d.BLOCK_M, d.HEAD_DIM] I32 := (offn d).row d.BLOCK_M

/-- The pointer `O` lane `(l, k)` of program `i` reads. -/
def ptrO (d : Inst) (i : Nat) : Tensor [d.BLOCK_M, d.HEAD_DIM] (Ptr .f16) :=
  d.O.ptr 0 + hzrow d i + mcol d i + nrow d

/-- The pointer `DO` lane `(l, k)` of program `i` reads. -/
def ptrDO (d : Inst) (i : Nat) : Tensor [d.BLOCK_M, d.HEAD_DIM] (Ptr .f16) :=
  d.DO.ptr 0 + hzrow d i + mcol d i + nrow d

/-- The pointer `Delta` lane `l` of program `i` writes. -/
def ptrD (d : Inst) (i : Nat) : Tensor [d.BLOCK_M] (Ptr .f32) :=
  d.Delta.ptr 0 + offhz d i * IntV.ofNat d.N_CTX + offm d i

/-- The body with the program ids substituted. -/
theorem body_eq (d : Inst) (i : Nat) :
    (model d).body ((launch d).ctx i) =
      (do
        let o ← tl.load (ptrO d i)
        let «do» : Tensor [d.BLOCK_M, d.HEAD_DIM] F32 ← tl.cvt (← tl.load (ptrDO d i))
        let delta ← tl.sum1c (← tl.mul (← tl.convert .f16 .f32 o) «do»)
        tl.store (ptrD d i) delta) := rfl

/-- The row of `O`, `DO` and `Delta` that lane `l` of program `i` reads (resp. writes). -/
def rowAt (d : Inst) (i : Nat) (l : Nat) : Nat := rowOf d i + l

/-- The `O`/`DO` element lane `(l, k)` of program `i` reads. -/
def readIdx (d : Inst) (i : Nat) (p : Idx [d.BLOCK_M, d.HEAD_DIM]) : Nat :=
  rowAt d i p.1.val * d.HEAD_DIM + p.2.1.val

/-- The `Delta` element lane `l` of program `i` writes. -/
def writeIdx (d : Inst) (i : Nat) (p : Idx [d.BLOCK_M]) : Nat := rowAt d i p.1.val

/-! ### `int32` arithmetic: the products do not wrap -/

theorem i32_ofNat_toInt {n : Nat} (hn : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = (n : Int) :=
  IntV.I32.toInt_ofNat hn

/-- A nonnegative integer below `2^31` as an `int32` has its own value. -/
theorem i32_ofInt_toInt {z : Int} (h0 : 0 ≤ z) (hz : z < 2 ^ 31) :
    (IntV.ofInt z : I32).toInt = z := by
  rw [IntV.I32.toInt_ofInt]
  constructor
  · have : (0 : Int) ≤ 2 ^ 31 := by norm_num
    linarith
  · simpa using hz

/-- A `int32` product with a literal, given the other factor's value. -/
theorem i32_mul {a : I32} {x : Int} {y : Nat} (ha : a.toInt = x) (hx : 0 ≤ x)
    (hy : y < 2 ^ 31) (hxy : x * (y : Int) < 2 ^ 31) :
    (a * (IntV.ofNat y : I32)).toInt = x * (y : Int) := by
  have hyv : ((IntV.ofNat y : I32)).toInt = (y : Int) := i32_ofNat_toInt hy
  have hy0 : (0 : ℤ) ≤ (y : ℤ) := by positivity
  have hsf : IntV.SFits 32 (x * (y : Int)) := by
    constructor
    · have h0 : (0 : ℤ) ≤ x * (y : ℤ) := mul_nonneg hx hy0
      omega
    · omega
  exact IntV.toInt_mul ha hyv hsf

/-- An addition of `int32`s, given both values. -/
theorem i32_add {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y < 2 ^ 31) : (a + b).toInt = x + y := by
  have hsf : IntV.SFits 32 (x + y) := by
    constructor <;> omega
  exact IntV.toInt_add ha hb hsf

/-- `BLOCK_M < 2^31`. -/
theorem block_lt {d : Inst} (hd : d.ok) : d.BLOCK_M < 2 ^ 31 := by
  have hp := head_pos hd
  have := hd.block_le
  nlinarith

/-- `HEAD_DIM < 2^31`. -/
theorem head_lt {d : Inst} (hd : d.ok) : d.HEAD_DIM < 2 ^ 31 := by
  have := hd.head_le; omega

/-- `N_CTX > 0` whenever the grid is nonempty. -/
theorem ctx_pos (d : Inst) (hd : d.ok) (i : Fin (launch d).size) : 0 < d.N_CTX := by
  have h := pm_block_lt d hd i 0 (block_pos hd)
  omega

/-- `bh < 2^31`. -/
theorem bh_lt_31 (d : Inst) (hd : d.ok) (i : Fin (launch d).size) : bh d i < 2 ^ 31 := by
  have h : bh d i < d.Z * d.H := by
    have := bh_lt d i
    simpa [gy] using this
  have hZH := hd.bh_lt
  omega

/-- A program's block offset times `HEAD_DIM` is below `2^31`. -/
theorem blockrow_mul_head_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Nat)
    (hl : l < d.BLOCK_M) : (pm d i * d.BLOCK_M + l) * d.HEAD_DIM < 2 ^ 31 := by
  have h1 := pm_block_lt d hd i l hl
  have h2 : pm d i * d.BLOCK_M + l ≤ d.N_CTX - 1 := by omega
  have h3 := hd.row_fit
  have hle : (pm d i * d.BLOCK_M + l) * d.HEAD_DIM ≤ (d.N_CTX - 1) * d.HEAD_DIM :=
    Nat.mul_le_mul_right _ h2
  omega

/-- `off_hz · HEAD_DIM` does not wrap. -/
theorem hz_head_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    bh d i * d.HEAD_DIM < 2 ^ 31 := by
  have hb : bh d i < d.Z * d.H := by
    have := bh_lt d i; simpa [gy] using this
  have hZH : 1 ≤ d.Z * d.H := by omega
  have hbh : bh d i ≤ d.Z * d.H - 1 := by omega
  have h1 : bh d i * d.HEAD_DIM ≤ (d.Z * d.H - 1) * d.HEAD_DIM := Nat.mul_le_mul_right _ hbh
  have hN : 1 ≤ d.N_CTX := ctx_pos d hd i
  have h2 : (d.Z * d.H - 1) * d.HEAD_DIM ≤ (d.Z * d.H - 1) * d.HEAD_DIM * d.N_CTX := by
    calc (d.Z * d.H - 1) * d.HEAD_DIM = (d.Z * d.H - 1) * d.HEAD_DIM * 1 := by ring
      _ ≤ (d.Z * d.H - 1) * d.HEAD_DIM * d.N_CTX := Nat.mul_le_mul_left _ hN
  have h3 := hd.head_fit
  omega

/-- `off_hz · HEAD_DIM · N_CTX` does not wrap. -/
theorem hz_head_ctx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    bh d i * d.HEAD_DIM * d.N_CTX < 2 ^ 31 := by
  have hb : bh d i < d.Z * d.H := by
    have := bh_lt d i; simpa [gy] using this
  have hbh : bh d i ≤ d.Z * d.H - 1 := by omega
  have h1 : bh d i * d.HEAD_DIM * d.N_CTX ≤ (d.Z * d.H - 1) * d.HEAD_DIM * d.N_CTX := by
    calc bh d i * d.HEAD_DIM * d.N_CTX
        = (bh d i * d.HEAD_DIM) * d.N_CTX := by ring
      _ ≤ ((d.Z * d.H - 1) * d.HEAD_DIM) * d.N_CTX :=
          Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hbh)
  have h3 := hd.head_fit
  omega

/-- `off_hz · N_CTX` does not wrap. -/
theorem hz_ctx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    bh d i * d.N_CTX < 2 ^ 31 := by
  have hp := head_pos hd
  have h1 : bh d i * d.N_CTX ≤ bh d i * d.HEAD_DIM * d.N_CTX := by
    calc bh d i * d.N_CTX = bh d i * 1 * d.N_CTX := by ring
      _ ≤ bh d i * d.HEAD_DIM * d.N_CTX := Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ hp)
  have h2 := hz_head_ctx_lt d hd i
  omega

/-- `off_hz · HEAD_DIM · N_CTX` as an integer. -/
theorem hzrow_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    (hzrow d i).toInt = (bh d i : Int) * (d.HEAD_DIM : Int) * (d.N_CTX : Int) := by
  have h1 : (offhz d i * (IntV.ofNat d.HEAD_DIM : I32)).toInt =
      (bh d i : Int) * (d.HEAD_DIM : Int) :=
    i32_mul (i32_ofNat_toInt (bh_lt_31 d hd i)) (by positivity) (head_lt hd) (by
      push_cast
      exact_mod_cast hz_head_lt d hd i)
  unfold hzrow
  rw [i32_mul h1 (by positivity) (by have := hd.ctx_lt; omega) (by
    push_cast
    exact_mod_cast hz_head_ctx_lt d hd i)]

/-- The `off_m` element of lane `l` as an integer. -/
theorem offm_get (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) :
    (offm d i).get (l, ()) =
      (IntV.ofNat (pm d i) * IntV.ofNat d.BLOCK_M) + IntV.ofInt ((0 : Int) + (l.val : Int)) := rfl

/-- The `off_m` element of lane `l` as an integer. -/
theorem offm_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M) :
    ((offm d i).get (l, ())).toInt = ((pm d i * d.BLOCK_M + l.val : Nat) : Int) := by
  have hB := block_pos hd
  have hval : pm d i * d.BLOCK_M + l.val < 2 ^ 31 := by
    have h := pm_block_lt d hd i l.val l.isLt
    have := hd.ctx_lt; omega
  have hplt : pm d i < 2 ^ 31 := by
    have h1 : pm d i * 1 ≤ pm d i * d.BLOCK_M := Nat.mul_le_mul_left _ (by omega)
    have h2 := hval
    omega
  have hll : (0 : Int) + (l.val : Int) < 2 ^ 31 := by
    have h := l.isLt; have := block_lt hd; omega
  have hprod_lt : pm d i * d.BLOCK_M < 2 ^ 31 := by omega
  have hprod : ((IntV.ofNat (pm d i) : I32) * IntV.ofNat d.BLOCK_M).toInt =
      (pm d i : Int) * (d.BLOCK_M : Int) :=
    i32_mul (i32_ofNat_toInt hplt) (by positivity) (block_lt hd) (by
      push_cast; exact_mod_cast hprod_lt)
  have hx : (pm d i : Int) * (d.BLOCK_M : Int) + ((0 : Int) + (l.val : Int)) < 2 ^ 31 := by
    have h2 : ((pm d i * d.BLOCK_M + l.val : Nat) : Int) < (2 : Int) ^ 31 := by
      exact_mod_cast hval
    push_cast at h2 ⊢
    linarith
  rw [offm_get,
    i32_add hprod (i32_ofInt_toInt (by positivity) hll) (by positivity) (by positivity) hx]
  push_cast
  ring

/-- The `off_m · HEAD_DIM` element of lane `(l, k)` as an integer. -/
theorem mcol_get (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    (mcol d i).get (l, (k, ())) = (offm d i).get (l, ()) * IntV.ofNat d.HEAD_DIM := rfl

/-- The `off_m · HEAD_DIM` element of lane `(l, k)` as an integer. -/
theorem mcol_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M)
    (k : Fin d.HEAD_DIM) :
    ((mcol d i).get (l, (k, ()))).toInt =
      ((pm d i * d.BLOCK_M + l.val : Nat) : Int) * (d.HEAD_DIM : Int) := by
  have hrow := blockrow_mul_head_lt d hd i l.val l.isLt
  have hoff := offm_toInt d hd i l
  rw [mcol_get]
  exact i32_mul hoff (by positivity) (head_lt hd) (by push_cast; exact_mod_cast hrow)

/-- The `off_n` element of lane `(l, k)` as an integer. -/
theorem nrow_get (d : Inst) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    (nrow d).get (l, (k, ())) = IntV.ofInt ((0 : Int) + (k.val : Int)) := rfl

/-- The `off_n` element of lane `(l, k)` as an integer. -/
theorem nrow_toInt (d : Inst) (hd : d.ok) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    ((nrow d).get (l, (k, ()))).toInt = (k.val : Int) := by
  have hk : (0 : Int) + (k.val : Int) < 2 ^ 31 := by
    have := k.isLt; have := head_lt hd; omega
  rw [nrow_get, i32_ofInt_toInt (by positivity) hk]
  ring_nf

/-! ### The pointer lanes -/

/-- The pointer `O` lane `(l, k)` of program `i`, unwound. -/
theorem ptrO_get0 (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    (ptrO d i).get (l, (k, ())) =
      (((d.O.ptr 0).offset (hzrow d i).toInt).offset
        ((mcol d i).get (l, (k, ()))).toInt).offset ((nrow d).get (l, (k, ()))).toInt := rfl

/-- The pointer `DO` lane `(l, k)` of program `i`, unwound. -/
theorem ptrDO_get0 (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    (ptrDO d i).get (l, (k, ())) =
      (((d.DO.ptr 0).offset (hzrow d i).toInt).offset
        ((mcol d i).get (l, (k, ()))).toInt).offset ((nrow d).get (l, (k, ()))).toInt := rfl

/-- The pointer `Delta` lane `l` of program `i`, unwound. -/
theorem ptrD_get0 (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) :
    (ptrD d i).get (l, ()) =
      ((d.Delta.ptr 0).offset (offhz d i * IntV.ofNat d.N_CTX).toInt).offset
        ((offm d i).get (l, ())).toInt := rfl

/-- The element `O` lane `(l, k)` of program `i` reads. -/
theorem ptrO_get (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M)
    (k : Fin d.HEAD_DIM) :
    (ptrO d i).get (l, (k, ())) = d.O.ptr ((readIdx d i (l, (k, ())) : Nat) : Int) := by
  have hh := hzrow_toInt d hd i
  have hm := mcol_toInt d hd i l k
  have hn := nrow_toInt d hd l k
  rw [ptrO_get0, Ptr.offset_offset, Ptr.offset_offset, Buf.ptr_offset, hh, hm, hn]
  have hread : ((readIdx d i (l, (k, ())) : Nat) : Int) =
      (bh d i : Int) * (d.N_CTX : Int) * (d.HEAD_DIM : Int) +
        ((pm d i * d.BLOCK_M + l.val : Nat) : Int) * (d.HEAD_DIM : Int) + (k.val : Int) := by
    unfold readIdx rowAt rowOf
    push_cast
    ring
  congr 1
  rw [hread]
  ring

/-- The element `DO` lane `(l, k)` of program `i` reads. -/
theorem ptrDO_get (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M)
    (k : Fin d.HEAD_DIM) :
    (ptrDO d i).get (l, (k, ())) = d.DO.ptr ((readIdx d i (l, (k, ())) : Nat) : Int) := by
  have hh := hzrow_toInt d hd i
  have hm := mcol_toInt d hd i l k
  have hn := nrow_toInt d hd l k
  rw [ptrDO_get0, Ptr.offset_offset, Ptr.offset_offset, Buf.ptr_offset, hh, hm, hn]
  have hread : ((readIdx d i (l, (k, ())) : Nat) : Int) =
      (bh d i : Int) * (d.N_CTX : Int) * (d.HEAD_DIM : Int) +
        ((pm d i * d.BLOCK_M + l.val : Nat) : Int) * (d.HEAD_DIM : Int) + (k.val : Int) := by
    unfold readIdx rowAt rowOf
    push_cast
    ring
  congr 1
  rw [hread]
  ring

/-- The `off_hz · N_CTX` value of program `i`. -/
theorem offhz_ctx_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    (offhz d i * IntV.ofNat d.N_CTX).toInt = (bh d i : Int) * (d.N_CTX : Int) :=
  i32_mul (i32_ofNat_toInt (bh_lt_31 d hd i)) (by positivity) (by have := hd.ctx_lt; omega) (by
    push_cast
    exact_mod_cast hz_ctx_lt d hd i)

/-- The element `Delta` lane `l` of program `i` writes. -/
theorem ptrD_get (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M) :
    (ptrD d i).get (l, ()) = d.Delta.ptr ((writeIdx d i (l, ()) : Nat) : Int) := by
  have hh := offhz_ctx_toInt d hd i
  have hm := offm_toInt d hd i l
  rw [ptrD_get0, Ptr.offset_offset, Buf.ptr_offset, hh, hm]
  have hwrite : ((writeIdx d i (l, ()) : Nat) : Int) =
      (bh d i : Int) * (d.N_CTX : Int) + (pm d i : Int) * (d.BLOCK_M : Int) + (l.val : Int) := by
    unfold writeIdx rowAt rowOf
    push_cast
    ring
  congr 1
  rw [hwrite]
  push_cast
  ring

end Claims.AttnBwdPre
