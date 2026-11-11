import Proof.BClause
import Proof.Defs
import Proof.Loop
import Proof.Num
import Proof.Ptr
import Proof.SafeProg

/-!
# SafeMain

One program walked line by line (`safe_program`).
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000

/-!
# `ssd_chunk_state_bf16`: the program is safe

The prelude computes the five buffer pointers, the tile offsets and the loop bound; the loop loads
the tiles and accumulates `float32` products; the last line stores the tile of `states`.
Nothing wraps, every load is inside the buffers and the store is inside the tile, so each step is
`Safe` with the read footprint `R` and the write footprint `W`.
-/

set_option maxHeartbeats 800000


open TVA

/-- `ratio d > 0`. -/
theorem ratio_pos {d : Inst} (hd : d.ok) : 0 < ratio d := by
  obtain ⟨k, hk⟩ := hd.groups_dvd
  have hk0 : k ≠ 0 := by
    rintro rfl
    have h0 : d.nheads = 0 := by simpa using hk
    exact (Nat.pos_iff_ne_zero.mp hd.heads_pos) h0
  have hk1 : 0 < k := Nat.pos_of_ne_zero hk0
  have hgn : 0 < d.ngroups := hd.groups_pos
  have hle : d.ngroups ≤ d.ngroups * k := Nat.le_mul_of_pos_right d.ngroups hk1
  unfold ratio
  rw [hk]
  exact Nat.div_pos hle hgn

/-- `nheads / ngroups` is below `2^31`. -/
theorem ratio_lt {d : Inst} (hd : d.ok) : ratio d < 2 ^ 31 := by
  have := hd.grid_z
  have h1 : ratio d ≤ d.nheads := Nat.div_le_self _ _
  omega

/-- The `int32` group of head `pid_h // ratio`: `pid_h·stride_b_head` does not wrap. -/
theorem grp_mul_lt {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.nheads) :
    grp d h * d.sBg < 2 ^ 31 := by
  obtain ⟨k, hk⟩ := hd.groups_dvd
  have hk1 : 0 < k := by
    have hk0 : k ≠ 0 := by
      rintro rfl
      have h0 : d.nheads = 0 := by simpa using hk
      exact (Nat.pos_iff_ne_zero.mp hd.heads_pos) h0
    exact Nat.pos_of_ne_zero hk0
  have hgl : grp d h < d.ngroups := by
    have hrk : ratio d = k := by
      unfold ratio
      rw [hk, Nat.mul_comm d.ngroups k]
      exact Nat.mul_div_left k hd.groups_pos
    unfold grp
    rw [hrk, Nat.div_lt_iff_lt_mul hk1]
    rw [hk] at hh
    exact hh
  have h1 : grp d h * d.sBg ≤ (d.ngroups - 1) * d.sBg :=
    Nat.mul_le_mul_right _ (Nat.le_pred_of_lt hgl)
  exact lt_of_le_of_lt h1 hd.bhead_lt

/-- `qOf d i < gx d`. -/
theorem qOf_lt_gx {d : Inst} (i : Fin (launch d).size) : qOf d i < gx d := by
  have hpos : 0 < gx d * nc d * d.nheads := by
    rw [← launch_size]
    exact Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have h : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h0 | h0
    · rw [h0] at hpos
      simp at hpos
    · exact h0
  unfold qOf
  exact Nat.mod_lt _ h

/-- `qOf d i < 2^31`. -/
theorem qOf_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : qOf d i < 2 ^ 31 :=
  lt_trans (qOf_lt_gx i) (gx_lt hd)


/-! ### `int64` arithmetic on nonnegative values -/

/-- A natural below `2^63` keeps its value in `int64`. -/
theorem I64.ofNat_nat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = (n : Int) :=
  IntV.I64.toInt_ofNat h

/-- The `int64` constant `1`. -/
theorem I64.one : (IntV.ofInt 1 : I64).toInt = 1 := by
  rw [IntV.toInt_ofInt (by norm_num) (by norm_num [IntV.SFits])]

/-- `int64` multiplication of naturals, without wrap, in the form the walk uses. -/
theorem I64.mul_nat' {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ 63) : (a * b).toInt = ((x * y : Nat) : Int) := by
  rw [I64.toInt_mul_nat (by rw [ha]; positivity) (by rw [hb]; positivity)
    (by rw [ha, hb]; push_cast; exact_mod_cast h)]
  rw [ha, hb]; push_cast; ring

/-- `int64` subtraction of naturals, without wrap, in the form the walk uses. -/
theorem I64.sub_nat' {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hxy : y ≤ x) (h : x < 2 ^ 63) : (a - b).toInt = ((x - y : Nat) : Int) := by
  have h' : (x : Int) < 2 ^ 63 := by exact_mod_cast h
  rw [I64.toInt_sub_nat (by rw [ha]; positivity) (by rw [hb]; positivity)
    (by rw [ha, hb]; exact_mod_cast hxy) (by rw [ha, hb]; omega)]
  rw [ha, hb]; push_cast [hxy]; ring

/-- The value of `min` of two integers is the `min` of the values. -/
theorem IntV.min_toInt {w : Nat} (a b : IntV true w) :
    (IntV.min a b).toInt = min a.toInt b.toInt := by
  unfold IntV.min
  rw [IntV.lt_eq]
  by_cases h : a.toInt < b.toInt
  · rw [ite_eq_left (by simpa using h), min_eq_left (le_of_lt h)]
  · rw [ite_eq_right (by simpa using h), min_eq_right (by omega)]

/-- A `dA_cumsum`/`dt`/`seq_idx` element's byte span is readable (not written by the program). -/
theorem span_sub_R (d : Inst) (init : Mem) (i : Fin (launch d).size) {d' : DType} {b : Buf d'}
    (hdisj : Disjoint d.states.bytes b.bytes) {j : Nat} (hj : j < b.len) :
    Mem.span d' (b.ptr (j : Int)).addr ⊆ R d init i := by
  intro x hx
  show x ∉ d.states.bytes
  intro hxs
  exact Set.disjoint_left.mp hdisj hxs (b.span_sub hj hx)

/-- The last token of a chunk is a token of the sequence. -/
theorem endTok_lt_seqlen {d : Inst} (hd : d.ok) {c : Nat} (_hc : c < nc d) :
    endTok d c < d.seqlen := by
  have hck := chunk_pos hd
  have h1 : 1 ≤ (c + 1) * d.chunk := by
    calc 1 = 1 * 1 := by norm_num
      _ ≤ (c + 1) * d.chunk := Nat.mul_le_mul (by omega) hck
  have hsp := hd.seqlen_pos
  unfold endTok
  rcases Nat.le_total ((c + 1) * d.chunk) d.seqlen with h | h
  · rw [Nat.min_eq_left h]
    omega
  · rw [Nat.min_eq_right h]
    omega

/-! `get2`/`get1` versions of the library's `get` lemmas (the library states them for `.get`,
whose index `(i, j, ())` cannot be written in a goal: `Idx []` does not reduce to `Unit`
at implicit transparency).  All are `rfl`. -/
private theorem get2_add' {m n : Nat} {sg : Bool} {w : Nat}
    (x y : Tensor [m, n] (IntV sg w)) (a : Fin m) (j : Fin n) :
    (x + y).get2 a j = x.get2 a j + y.get2 a j := rfl

private theorem get2_muls' {m n : Nat} {sg : Bool} {w : Nat}
    (x : Tensor [m, n] (IntV sg w)) (c : IntV sg w) (a : Fin m) (j : Fin n) :
    (x * c).get2 a j = x.get2 a j * c := rfl

private theorem get2_sadd' {m n : Nat} {sg : Bool} {w : Nat}
    (c : IntV sg w) (x : Tensor [m, n] (IntV sg w)) (a : Fin m) (j : Fin n) :
    (c + x).get2 a j = c + x.get2 a j := rfl

private theorem get2_col' {m n : Nat} {α : Type} (x : Tensor [m] α) (a : Fin m) (j : Fin n) :
    (x.col n).get2 a j = x.get1 a := rfl

private theorem get2_row' {m n : Nat} {α : Type} (x : Tensor [n] α) (a : Fin m) (j : Fin n) :
    (x.row m).get2 a j = x.get1 j := rfl

private theorem get1_muls' {n : Nat} {sg : Bool} {w : Nat}
    (x : Tensor [n] (IntV sg w)) (c : IntV sg w) (j : Fin n) :
    (x * c).get1 j = x.get1 j * c := rfl

private theorem get1_sadd' {n : Nat} {sg : Bool} {w : Nat}
    (c : IntV sg w) (x : Tensor [n] (IntV sg w)) (j : Fin n) :
    (c + x).get1 j = c + x.get1 j := rfl

private theorem qOf_lt_gx' {d : Inst} (i : Fin (launch d).size) : qOf d i < gx d := by
  have hpos : 0 < gx d * nc d * d.nheads := by
    rw [← launch_size]
    exact Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have h : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h0 | h0
    · rw [h0] at hpos
      simp at hpos
    · exact h0
  unfold qOf
  exact Nat.mod_lt _ h

private theorem ptr_off_eq {dd : DType} (B : Buf dd) (base target k : Int)
    (h : base + k = target) : ((B.ptr base).offset k).addr = (B.ptr target).addr := by
  simp only [Ptr.offset_addr, Buf.ptr_addr]
  have : base * dd.bytes + k * dd.bytes = target * dd.bytes := by
    rw [← Int.add_mul, h]
  linarith

private theorem i32_ofInt_add0 (n : Nat) (h : n < 2 ^ 31) :
    (IntV.ofInt (0 + (n : Int)) : I32).toInt = (n : Int) := by
  simp only [Int.zero_add]
  refine IntV.I32.toInt_ofInt (x := (n : Int)) ?_
  constructor
  · have h0 : (0 : Int) ≤ (n : Int) := by exact_mod_cast Nat.zero_le n
    linarith
  · exact_mod_cast h

private theorem pmOf_lt' {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    pmOf d i < (d.headdim + BM - 1) / BM := by
  have hq := qOf_lt_gx' i
  rw [gx_eq] at hq
  unfold pmOf
  rw [Nat.div_lt_iff_lt_mul (npmOf_pos hd)]
  exact hq

private theorem pnOf_lt' {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    pnOf d i < (d.dstate + BN - 1) / BN := by
  have hmod : pnOf d i = qOf d i % ((d.dstate + BN - 1) / BN) := rfl
  rw [hmod]
  have hpos : 0 < (d.dstate + BN - 1) / BN := by
    simpa [npmOf] using npmOf_pos hd
  exact Nat.mod_lt _ hpos

private theorem pmOf_mul_add_le {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (a : Fin BM) :
    pmOf d i * BM + (a : Nat) ≤ d.headdim + 63 := by
  have h1 := pmOf_lt' hd i
  have h2 : pmOf d i * BM + BM ≤ ((d.headdim + BM - 1) / BM) * BM := by
    have h := Nat.mul_le_mul_right BM (Nat.succ_le_of_lt h1)
    calc pmOf d i * BM + BM = (pmOf d i + 1) * BM := by ring
      _ ≤ _ := h
  have h3 : ((d.headdim + BM - 1) / BM) * BM ≤ d.headdim + BM - 1 := Nat.div_mul_le_self _ _
  have ha : (a : Nat) < BM := a.isLt
  have hp := hd.headdim_pos
  simp only [BM] at h2 h3 ha ⊢
  omega

private theorem pnOf_mul_add_le {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (n : Fin BN) :
    pnOf d i * BN + (n : Nat) ≤ d.dstate + 127 := by
  have h1 := pnOf_lt' hd i
  have h2 : pnOf d i * BN + BN ≤ ((d.dstate + BN - 1) / BN) * BN := by
    have h := Nat.mul_le_mul_right BN (Nat.succ_le_of_lt h1)
    calc pnOf d i * BN + BN = (pnOf d i + 1) * BN := by ring
      _ ≤ _ := h
  have h3 : ((d.dstate + BN - 1) / BN) * BN ≤ d.dstate + BN - 1 := Nat.div_mul_le_self _ _
  have ha : (n : Nat) < BN := n.isLt
  have hp := hd.dstate_pos
  simp only [BN] at h2 h3 ha ⊢
  omega

/-! ### Bounds on the tile offsets -/

private theorem x_row_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (a : Fin BM) :
    pmOf d i * BM + (a : Nat) < 2 ^ 31 := by
  have h1 : pmOf d i * BM + (a : Nat) ≤ d.headdim + 63 := pmOf_mul_add_le hd i a
  have h2 := hd.headdim_lt
  omega

private theorem x_lane_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (a : Fin BM) :
    (pmOf d i * BM + (a : Nat)) * d.sXp < 2 ^ 31 := by
  have h1 : pmOf d i * BM + (a : Nat) ≤ d.headdim + 63 := pmOf_mul_add_le hd i a
  have h2 : (pmOf d i * BM + (a : Nat)) * d.sXp ≤ (d.headdim + 63) * d.sXp :=
    Nat.mul_le_mul_right _ h1
  have h3 : (d.headdim + 63) * d.sXp ≤ (d.headdim + 63) * d.sXp + 64 * d.sXs :=
    Nat.le_add_right _ _
  have h4 := hd.xtile_lt
  omega

private theorem k_lane_lt {d : Inst} (hd : d.ok) (j : Fin BK) : (j : Nat) * d.sXs < 2 ^ 31 := by
  have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
  have h1 : (j : Nat) * d.sXs ≤ 63 * d.sXs := Nat.mul_le_mul_right _ (by omega)
  have h2 : 63 * d.sXs ≤ 64 * d.sXs := Nat.mul_le_mul_right _ (by norm_num)
  have h3 : 64 * d.sXs ≤ (d.headdim + 63) * d.sXp + 64 * d.sXs := Nat.le_add_left _ _
  have h4 := hd.xtile_lt
  omega

private theorem b_row_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (n : Fin BN) :
    pnOf d i * BN + (n : Nat) < 2 ^ 31 := by
  have h1 : pnOf d i * BN + (n : Nat) ≤ d.dstate + 127 := pnOf_mul_add_le hd i n
  have h2 := hd.dstate_lt
  omega

private theorem b_lane_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (n : Fin BN) :
    (pnOf d i * BN + (n : Nat)) * d.sBn < 2 ^ 31 := by
  have h1 : pnOf d i * BN + (n : Nat) ≤ d.dstate + 127 := pnOf_mul_add_le hd i n
  have h2 : (pnOf d i * BN + (n : Nat)) * d.sBn ≤ (d.dstate + 127) * d.sBn :=
    Nat.mul_le_mul_right _ h1
  have h3 : (d.dstate + 127) * d.sBn ≤ (d.dstate + 127) * d.sBn + 64 * d.sBs :=
    Nat.le_add_right _ _
  have h4 := hd.btile_lt
  omega

private theorem k_lane_lt_b {d : Inst} (hd : d.ok) (j : Fin BK) : (j : Nat) * d.sBs < 2 ^ 31 := by
  have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
  have h1 : (j : Nat) * d.sBs ≤ 63 * d.sBs := Nat.mul_le_mul_right _ (by omega)
  have h2 : 63 * d.sBs ≤ 64 * d.sBs := Nat.mul_le_mul_right _ (by norm_num)
  have h3 : 64 * d.sBs ≤ (d.dstate + 127) * d.sBn + 64 * d.sBs := Nat.le_add_left _ _
  have h4 := hd.btile_lt
  omega

/-! The `[BK]` offsets `tl.arange 0 BK * 1` of `dt`, `dA_cumsum` and `seq_idx`. -/
private theorem k_off_toInt (j : Fin BK) :
    ((tl.arange 0 BK * (IntV.ofNat 1 : I32)).get1 j).toInt = ((j : Nat) * 1 : Nat) := by
  simp only [tl.arange, get1_muls', Tensor.get1_arange]
  have h1 : (IntV.ofNat 1 : I32).toInt = (1 : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have hj0 : (j : Nat) < 2 ^ 31 := by
    have hj : (j : Nat) < BK := j.isLt
    simp only [BK] at hj ⊢
    omega
  have hofj : (IntV.ofInt (0 + ((j : Nat) : Int)) : I32).toInt = ((j : Nat) : Int) :=
    i32_ofInt_add0 (j : Nat) hj0
  exact IntV.I32.mul_nat hofj h1 (by omega)

/-! x: `offs_m * stride_x_hdim + offs_k * stride_x_seqlen` -/
private theorem x_off_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r : I32}
    (hr : r.toInt = (pmOf d i : Int)) (a : Fin BM) (j : Fin BK) :
    ((((r * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK) * (IntV.ofNat d.sXp : I32) +
       ((tl.arange 0 BK).row BM) * (IntV.ofNat d.sXs : I32)).get2 a j).toInt
      = (((pmOf d i * BM + (a : Nat)) * d.sXp + (j : Nat) * d.sXs : Nat) : Int) := by
  simp only [tl.arange, get2_add', get2_muls', get2_col',
    get1_sadd', Tensor.get1_arange]
  have hBM : (IntV.ofNat BM : I32).toInt = (BM : Int) := IntV.I32.toInt_ofNat (by norm_num [BM])
  have hsXp : (IntV.ofNat d.sXp : I32).toInt = (d.sXp : Int) := IntV.I32.toInt_ofNat hd.sX_lt.2.2
  have hsXs : (IntV.ofNat d.sXs : I32).toInt = (d.sXs : Int) := IntV.I32.toInt_ofNat hd.sX_lt.1
  have hrow : pmOf d i * BM + (a : Nat) < 2 ^ 31 := x_row_lt hd i a
  have hrow2 : pmOf d i * BM < 2 ^ 31 := by
    have hle : pmOf d i * BM ≤ pmOf d i * BM + (a : Nat) := Nat.le_add_right _ _
    omega
  have ha0 : (a : Nat) < 2 ^ 31 := by
    have ha : (a : Nat) < BM := a.isLt
    simp only [BM] at ha ⊢
    omega
  have hof : (IntV.ofInt (0 + ((a : Nat) : Int)) : I32).toInt = ((a : Nat) : Int) :=
    i32_ofInt_add0 (a : Nat) ha0
  have hj0 : (j : Nat) < 2 ^ 31 := by
    have hj : (j : Nat) < BK := j.isLt
    simp only [BK] at hj ⊢
    omega
  have hofj : (IntV.ofInt (0 + ((j : Nat) : Int)) : I32).toInt = ((j : Nat) : Int) :=
    i32_ofInt_add0 (j : Nat) hj0
  have hmul : (r * (IntV.ofNat BM : I32)).toInt = ((pmOf d i * BM : Nat) : Int) :=
    IntV.I32.mul_nat hr hBM hrow2
  have hA : (r * (IntV.ofNat BM : I32) + IntV.ofInt (0 + ((a : Nat) : Int))).toInt
      = ((pmOf d i * BM + (a : Nat) : Nat) : Int) :=
    IntV.I32.add_nat hmul hof hrow
  have hM : ((r * (IntV.ofNat BM : I32) + IntV.ofInt (0 + ((a : Nat) : Int))) *
      (IntV.ofNat d.sXp : I32)).toInt
      = (((pmOf d i * BM + (a : Nat)) * d.sXp : Nat) : Int) :=
    IntV.I32.mul_nat hA hsXp (x_lane_lt hd i a)
  have hK : (IntV.ofInt (0 + ((j : Nat) : Int)) * (IntV.ofNat d.sXs : I32)).toInt
      = (((j : Nat) * d.sXs : Nat) : Int) :=
    IntV.I32.mul_nat hofj hsXs (k_lane_lt hd j)
  have hb : (pmOf d i * BM + (a : Nat)) * d.sXp + (j : Nat) * d.sXs < 2 ^ 31 := by
    have h1 : (pmOf d i * BM + (a : Nat)) * d.sXp ≤ (d.headdim + 63) * d.sXp :=
      Nat.mul_le_mul_right _ (pmOf_mul_add_le hd i a)
    have h2 : (j : Nat) * d.sXs ≤ 63 * d.sXs := by
      have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
      exact Nat.mul_le_mul_right _ (by omega)
    have h3 : 63 * d.sXs ≤ 64 * d.sXs := Nat.mul_le_mul_right _ (by norm_num)
    have h4 : 64 * d.sXs ≤ (d.headdim + 63) * d.sXp + 64 * d.sXs := Nat.le_add_left _ _
    have h5 := hd.xtile_lt
    omega
  exact IntV.I32.add_nat hM hK hb

/-! b: `offs_n * stride_b_hdim + offs_k * stride_b_seqlen` -/
private theorem b_off_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r : I32}
    (hr : r.toInt = (pnOf d i : Int)) (n : Fin BN) (j : Fin BK) :
    ((((r * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK) * (IntV.ofNat d.sBn : I32) +
       ((tl.arange 0 BK).col BN) * (IntV.ofNat d.sBs : I32)).get2 j n).toInt
      = (((pnOf d i * BN + (n : Nat)) * d.sBn + (j : Nat) * d.sBs : Nat) : Int) := by
  simp only [tl.arange, get2_add', get2_muls', get2_row',
    get1_sadd', Tensor.get1_arange]
  have hBN : (IntV.ofNat BN : I32).toInt = (BN : Int) := IntV.I32.toInt_ofNat (by norm_num [BN])
  have hsBn : (IntV.ofNat d.sBn : I32).toInt = (d.sBn : Int) := IntV.I32.toInt_ofNat hd.sB_lt.2.2
  have hsBs : (IntV.ofNat d.sBs : I32).toInt = (d.sBs : Int) := IntV.I32.toInt_ofNat hd.sB_lt.1
  have hrow : pnOf d i * BN + (n : Nat) < 2 ^ 31 := b_row_lt hd i n
  have hrow2 : pnOf d i * BN < 2 ^ 31 := by
    have hle : pnOf d i * BN ≤ pnOf d i * BN + (n : Nat) := Nat.le_add_right _ _
    omega
  have hn0 : (n : Nat) < 2 ^ 31 := by
    have ha : (n : Nat) < BN := n.isLt
    simp only [BN] at ha ⊢
    omega
  have hof : (IntV.ofInt (0 + ((n : Nat) : Int)) : I32).toInt = ((n : Nat) : Int) :=
    i32_ofInt_add0 (n : Nat) hn0
  have hj0 : (j : Nat) < 2 ^ 31 := by
    have hj : (j : Nat) < BK := j.isLt
    simp only [BK] at hj ⊢
    omega
  have hofj : (IntV.ofInt (0 + ((j : Nat) : Int)) : I32).toInt = ((j : Nat) : Int) :=
    i32_ofInt_add0 (j : Nat) hj0
  have hmul : (r * (IntV.ofNat BN : I32)).toInt = ((pnOf d i * BN : Nat) : Int) :=
    IntV.I32.mul_nat hr hBN hrow2
  have hA : (r * (IntV.ofNat BN : I32) + IntV.ofInt (0 + ((n : Nat) : Int))).toInt
      = ((pnOf d i * BN + (n : Nat) : Nat) : Int) :=
    IntV.I32.add_nat hmul hof hrow
  have hM : ((r * (IntV.ofNat BN : I32) + IntV.ofInt (0 + ((n : Nat) : Int))) *
      (IntV.ofNat d.sBn : I32)).toInt
      = (((pnOf d i * BN + (n : Nat)) * d.sBn : Nat) : Int) :=
    IntV.I32.mul_nat hA hsBn (b_lane_lt hd i n)
  have hK : (IntV.ofInt (0 + ((j : Nat) : Int)) * (IntV.ofNat d.sBs : I32)).toInt
      = (((j : Nat) * d.sBs : Nat) : Int) :=
    IntV.I32.mul_nat hofj hsBs (k_lane_lt_b hd j)
  have hb : (pnOf d i * BN + (n : Nat)) * d.sBn + (j : Nat) * d.sBs < 2 ^ 31 := by
    have h1 : (pnOf d i * BN + (n : Nat)) * d.sBn ≤ (d.dstate + 127) * d.sBn :=
      Nat.mul_le_mul_right _ (pnOf_mul_add_le hd i n)
    have h2 : (j : Nat) * d.sBs ≤ 63 * d.sBs := by
      have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
      exact Nat.mul_le_mul_right _ (by omega)
    have h3 : 63 * d.sBs ≤ 64 * d.sBs := Nat.mul_le_mul_right _ (by norm_num)
    have h4 : 64 * d.sBs ≤ (d.dstate + 127) * d.sBn + 64 * d.sBs := Nat.le_add_left _ _
    have h5 := hd.btile_lt
    omega
  exact IntV.I32.add_nat hM hK hb

/-! The index identities, in `Nat`. -/
private theorem idx_x (d : Inst) (i : Fin (launch d).size) (a : Fin BM) (j : Fin BK) :
    xIdx d (chunkOf d i * d.chunk) (headOf d i) 0 +
        ((pmOf d i * BM + (a : Nat)) * d.sXp + (j : Nat) * d.sXs)
      = xIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (headOf d i)
          (pmOf d i * BM + (a : Nat)) := by
  unfold xIdx
  ring

private theorem idx_b (d : Inst) (i : Fin (launch d).size) (n : Fin BN) (j : Fin BK) :
    bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0 +
        ((pnOf d i * BN + (n : Nat)) * d.sBn + (j : Nat) * d.sBs)
      = bIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (grp d (headOf d i))
          (pnOf d i * BN + (n : Nat)) := by
  unfold bIdx
  ring

private theorem idx_dt (d : Inst) (i : Fin (launch d).size) (j : Fin BK) :
    dIdx d (headOf d i) (chunkOf d i) 0 + ((j : Nat) * 1)
      = dIdx d (headOf d i) (chunkOf d i) (0 * BK + (j : Nat)) := by
  unfold dIdx
  ring

private theorem idx_seq (d : Inst) (i : Fin (launch d).size) (j : Fin BK) :
    chunkOf d i * d.chunk + ((j : Nat) * 1)
      = chunkOf d i * d.chunk + (0 * BK + (j : Nat)) := by
  ring

/-- **One program** of the launch is safe from `init`: it reads inside `R`, writes inside `W`, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size)
    (hexp : Device.MathTotal E.P .exp .binary32) (hdot : Device.DotTotal E.P dotCfg)
    (hexpacc : Device.ExpF32 E.P) (hdotacc : Device.DotBF16 E.P) :
    Safe E (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hvx, hvb, hvdt, hvdA, hvq, hvs, hmeta⟩ := hpre
  have hc31 : chunkOf d i < 2 ^ 31 := by
    have := hd.grid_y
    have := chunkOf_lt hd (i : Nat)
    omega
  have hq31 : qOf d i < 2 ^ 31 := qOf_lt hd i
  have hh31 : headOf d i < 2 ^ 31 := by
    have := hd.grid_z
    have := headOf_lt hd i.isLt
    omega
  have hr31 : ratio d < 2 ^ 31 := ratio_lt hd
  have hrp : 0 < ratio d := ratio_pos hd
  have hds : (IntV.ofNat d.dstate + IntV.ofNat (BN - 1) : I32).toInt = (d.dstate + 127 : Nat) := by
    have h := IntV.I32.add_nat (a := (IntV.ofNat d.dstate : I32))
      (b := (IntV.ofNat (BN - 1) : I32)) (x := d.dstate) (y := 127)
      (by exact IntV.I32.toInt_ofNat (by have := hd.dstate_lt; omega))
      (by exact IntV.I32.toInt_ofNat (by norm_num [BN])) (by exact hd.dstate_lt)
    simpa [BN] using h
  have hbn : (IntV.ofNat BN : I32).toInt = (128 : Int) := by
    rw [IntV.I32.toInt_ofNat (by norm_num [BN])]
    norm_num [BN]
  -- line 158: `pid_bc = tl.program_id(1).to(tl.int64)`
  refine safe_conv64 (z := tl.program_id ((launch d).ctx i) 1) ?_
  intro r1 h1
  have H1 : r1.item.toInt = (chunkOf d i : Int) := by
    rw [h1, pid1_eq]; exact IntV.I32.toInt_ofNat hc31
  -- line 159: `batch = 1` as an `int64`
  refine safe_conv64 (z := (IntV.ofNat 1 : I32)) ?_
  intro r2 h2
  have H2 : r2.item.toInt = (1 : Int) := by
    rw [h2]; exact IntV.I32.toInt_ofNat (by norm_num)
  -- line 160: `pid_c = pid_bc // batch`
  refine safe_idiv1 (x := r1.item) (y := r2.item) (by rw [H1]; positivity)
    (by rw [H2]; norm_num) ?_
  intro r3 h3
  have H3 : r3.item.toInt = (chunkOf d i : Int) := by
    rw [h3, H1, H2, Int.ediv_one]
  -- line 161: `pid_h = tl.program_id(2)` (an `int32` `have`, no step)
  -- line 162: `num_pid_n = cdiv(dstate, 128)`
  refine safe_idiv1
    (x := (IntV.ofNat d.dstate + IntV.ofNat (BN - 1) : I32)) (y := (IntV.ofNat BN : I32)) ?_ ?_ ?_
  · rw [hds]; positivity
  · rw [hbn]; norm_num
  intro r4 h4
  have H4 : r4.item.toInt = (npmOf d : Int) := by
    rw [h4, hds, hbn]
    have hd1 : d.dstate + 127 = d.dstate + BN - 1 := by rw [BN]; omega
    rw [hd1]
    exact (Int.natCast_ediv (d.dstate + BN - 1) BN).symm
  -- line 163: `pid_m = tl.program_id(0) // num_pid_n`
  refine safe_idiv1 (x := tl.program_id ((launch d).ctx i) 0) (y := r4.item) ?_ ?_ ?_
  · rw [pid0_eq, IntV.I32.toInt_ofNat hq31]; positivity
  · rw [H4]; exact_mod_cast npmOf_pos hd
  intro r5 h5
  have H5 : r5.item.toInt = (pmOf d i : Int) := by
    rw [h5, pid0_eq, IntV.I32.toInt_ofNat hq31, H4]
    exact (Int.natCast_ediv (qOf d i) (npmOf d)).symm
  -- line 164: `pid_n = tl.program_id(0) % num_pid_n`
  refine safe_irem1 (x := tl.program_id ((launch d).ctx i) 0) (y := r4.item) ?_ ?_ ?_
  · rw [pid0_eq, IntV.I32.toInt_ofNat hq31]; positivity
  · rw [H4]; exact_mod_cast npmOf_pos hd
  intro r6 h6
  have H6 : r6.item.toInt = (pnOf d i : Int) := by
    rw [h6, pid0_eq, IntV.I32.toInt_ofNat hq31, H4]
    exact (Int.natCast_emod (qOf d i) (npmOf d)).symm
  -- line 165: `head_b = pid_h // ratio`
  refine safe_idiv1 (x := tl.program_id ((launch d).ctx i) 2)
    (y := (IntV.ofNat (ratio d) : I32)) ?_ ?_ ?_
  · rw [pid2_eq, IntV.I32.toInt_ofNat hh31]; positivity
  · rw [IntV.I32.toInt_ofNat hr31]; exact_mod_cast hrp
  intro r7 h7
  have H7 : r7.item.toInt = (grp d (headOf d i) : Int) := by
    rw [h7, pid2_eq, IntV.I32.toInt_ofNat hh31, IntV.I32.toInt_ofNat hr31]
    exact (Int.natCast_ediv (headOf d i) (ratio d)).symm
  have hch31 : d.chunk < 2 ^ 31 := by
    have := hd.chunk_le
    have := hd.chunk_pow2
    omega
  have hhh : headOf d i < d.nheads := headOf_lt hd i.isLt
  have hxh31 : headOf d i * d.sXh < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.le_pred_of_lt hhh)) hd.xhead_lt
  have hdh31 : headOf d i * sDh d < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.le_pred_of_lt hhh)) hd.dt_lt
  have hsl31 : d.seqlen < 2 ^ 31 := by
    have := hd.seqlen_ctx
    omega
  -- line 166: `b_ptr`'s components
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro c1 hc1
  have HC1 : c1.item.toInt = (d.chunk : Int) := by
    rw [hc1, IntV.I32.toInt_ofNat hch31]
  refine safe_conv64 (z := (IntV.ofNat d.sBs : I32)) ?_
  intro s1 hs1
  have HS1 : s1.item.toInt = (d.sBs : Int) := by
    rw [hs1, IntV.I32.toInt_ofNat hd.sB_lt.1]
  refine safe_conv64 (z := (r7.item * IntV.ofNat d.sBg : I32)) ?_
  intro s2 hs2
  have HS2 : s2.item.toInt = (grp d (headOf d i) * d.sBg : Int) := by
    rw [hs2, IntV.I32.mul_nat H7 (IntV.I32.toInt_ofNat hd.sB_lt.2.1)
      (grp_mul_lt hd hhh)]
    push_cast
    ring
  -- line 167: `x_ptr`'s components
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro c2 hc2
  have HC2 : c2.item.toInt = (d.chunk : Int) := by
    rw [hc2, IntV.I32.toInt_ofNat hch31]
  refine safe_conv64 (z := (IntV.ofNat d.sXs : I32)) ?_
  intro s3 hs3
  have HS3 : s3.item.toInt = (d.sXs : Int) := by
    rw [hs3, IntV.I32.toInt_ofNat hd.sX_lt.1]
  refine safe_conv64 (z := (tl.program_id ((launch d).ctx i) 2 * IntV.ofNat d.sXh : I32)) ?_
  intro s4 hs4
  have HS4 : s4.item.toInt = (headOf d i * d.sXh : Int) := by
    rw [hs4, IntV.I32.mul_nat (by rw [pid2_eq]; exact IntV.I32.toInt_ofNat hh31)
      (IntV.I32.toInt_ofNat hd.sX_lt.2.1) hxh31]
    push_cast
    ring
  -- line 168: `dt_ptr`'s components
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro c3 hc3
  have HC3 : c3.item.toInt = (d.chunk : Int) := by
    rw [hc3, IntV.I32.toInt_ofNat hch31]
  refine safe_conv64 (z := (tl.program_id ((launch d).ctx i) 2 * IntV.ofNat (sDh d) : I32)) ?_
  intro s5 hs5
  have HS5 : s5.item.toInt = (headOf d i * sDh d : Int) := by
    rw [hs5, IntV.I32.mul_nat (by rw [pid2_eq]; exact IntV.I32.toInt_ofNat hh31)
      (IntV.I32.toInt_ofNat (by show nc d * d.chunk < 2 ^ 31; exact hd.tok_lt)) hdh31]
    push_cast
    ring
  -- line 169: `dA_cumsum_ptr`'s components
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro c4 hc4
  have HC4 : c4.item.toInt = (d.chunk : Int) := by
    rw [hc4, IntV.I32.toInt_ofNat hch31]
  refine safe_conv64 (z := (tl.program_id ((launch d).ctx i) 2 * IntV.ofNat (sDh d) : I32)) ?_
  intro s6 hs6
  have HS6 : s6.item.toInt = (headOf d i * sDh d : Int) := by
    rw [hs6, IntV.I32.mul_nat (by rw [pid2_eq]; exact IntV.I32.toInt_ofNat hh31)
      (IntV.I32.toInt_ofNat (by show nc d * d.chunk < 2 ^ 31; exact hd.tok_lt)) hdh31]
    push_cast
    ring
  -- line 170: `seq_idx_ptr`'s components
  refine safe_conv64 (z := (IntV.ofNat d.seqlen : I32)) ?_
  intro s7 hs7
  have HS7 : s7.item.toInt = (d.seqlen : Int) := by
    rw [hs7, IntV.I32.toInt_ofNat hsl31]
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro c5 hc5
  have HC5 : c5.item.toInt = (d.chunk : Int) := by
    rw [hc5, IntV.I32.toInt_ofNat hch31]
  refine safe_conv64 (z := (IntV.ofNat 1 : I32)) ?_
  intro o1 ho1
  have HO1 : o1.item.toInt = 1 := by
    rw [ho1, IntV.I32.toInt_ofNat (by norm_num)]
    norm_num
  have hck : 0 < d.chunk := by
    obtain ⟨k, hk⟩ := hd.chunk_pow2
    rw [hk]; exact Nat.two_pow_pos k
  have hpb : (r1.item - r3.item * r2.item).toInt = 0 := by
    have hbm : (r3.item * r2.item).toInt = (chunkOf d i : Int) := by
      rw [I64.mul_nat' H3 H2 (by have h2 : chunkOf d i * 1 = chunkOf d i := Nat.mul_one _; rw [h2]; have := hc31; omega)]
      simp
    rw [I64.sub_nat' H1 hbm le_rfl (by have := hc31; omega)]
    simp
  -- line 176: `dA_cs_last`, the chunk's last `dA_cumsum` entry (a scalar load)
  have hold : dIdx d (headOf d i) (chunkOf d i) (d.chunk - 1) < d.dA.len :=
    lt_of_lt_of_le (dIdx_lt hhh (chunkOf_lt hd (i : Nat)) (by omega)) hd.dA_len
  obtain ⟨v0, hv0⟩ := d.dA.read_some hvdA hold
  have hdAb : (d.dA.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
      r3.item * c4.item + s6.item)) = d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) 0) :=
    dA_base hd i hpb H3 HC4 (by rw [HS6]; push_cast; ring)
  have hoff : ((IntV.ofNat d.chunk - IntV.ofInt 1 : I32) * IntV.ofNat 1 : I32).toInt =
      ((d.chunk - 1 : Nat) : Int) := by
    have hc1' : (IntV.ofNat d.chunk : I32).toInt = (d.chunk : Int) := IntV.I32.toInt_ofNat hch31
    have hone : (IntV.ofInt 1 : I32).toInt = 1 :=
      IntV.toInt_ofInt (by norm_num) (by norm_num [IntV.SFits])
    have hone' : (IntV.ofNat 1 : I32).toInt = 1 := by
      rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num
    have hsub : (IntV.ofNat d.chunk - IntV.ofInt 1 : I32).toInt = ((d.chunk - 1 : Nat) : Int) :=
      IntV.I32.sub_nat hc1' hone hck (by have := hd.chunk_le; omega)
    rw [IntV.I32.mul_nat hsub hone' (by have h2 : (d.chunk - 1) * 1 = d.chunk - 1 := Nat.mul_one _; rw [h2]; have := hd.chunk_le; omega)]
    simp
  have hidx : (dIdx d (headOf d i) (chunkOf d i) 0 : Int) + ((d.chunk - 1 : Nat) : Int)
      = (dIdx d (headOf d i) (chunkOf d i) (d.chunk - 1) : Int) := by
    simp only [dIdx]
    push_cast
    omega
  have haddr : ((d.dA.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
      r3.item * c4.item + s6.item)) + ((IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32)).addr
      = (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (d.chunk - 1))).addr := by
    have hp : (d.dA.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
        r3.item * c4.item + s6.item)) + ((IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32)
        = d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (d.chunk - 1)) := by
      rw [hdAb]
      show (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) 0)).offset
          (((IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32)).toInt
          = d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (d.chunk - 1))
      rw [hoff, Buf.ptr_offset, hidx]
    exact congrArg Ptr.addr hp
  simp only []
  refine safe_load1 (E := E) (R := R d init i) (W := W d init i) (Q := fun x m => Q d init i m)
    (m := init) (v0 := v0) (p := d.dA.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
      r3.item * c4.item + s6.item) + ((IntV.ofNat d.chunk - IntV.ofInt 1) * IntV.ofNat 1 : I32)) ?_ ?_ ?_
  · rw [haddr]; exact hv0
  · rw [haddr]
    exact span_sub_R d init i hd.disj.2.2.2.1 hold
  -- line 177: the loop bound `min(chunk, seqlen - pid_c·chunk)`
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro k1 hk1
  have HK1 : k1.item.toInt = (d.chunk : Int) := by
    rw [hk1, IntV.I32.toInt_ofNat hch31]
  refine safe_conv64 (z := (IntV.ofNat d.seqlen : I32)) ?_
  intro k2 hk2
  have HK2 : k2.item.toInt = (d.seqlen : Int) := by
    rw [hk2, IntV.I32.toInt_ofNat hsl31]
  refine safe_conv64 (z := (IntV.ofNat d.chunk : I32)) ?_
  intro k3 hk3
  have HK3 : k3.item.toInt = (d.chunk : Int) := by
    rw [hk3, IntV.I32.toInt_ofNat hch31]
  -- `chunk_size_limit` is a pure `let`, no step of the program
  have hcc31 : chunkOf d i * d.chunk < 2 ^ 31 := PtrAux.chunkOf_mul_lt hd (i : Nat)
  have hcsz : (k2.item - r3.item * k3.item).toInt = ((d.seqlen - chunkOf d i * d.chunk : Nat) : Int) := by
    have h1 : (r3.item * k3.item).toInt = ((chunkOf d i * d.chunk : Nat) : Int) := by
      rw [I64.mul_nat' H3 HK3 (by omega)]
    rw [I64.sub_nat' HK2 h1 (le_of_lt (c_chunk_lt hd (chunkOf_lt hd (i : Nat))))
      (by have := hsl31; omega)]
  have hlim : ((tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item).toInt
      = ((endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk : Nat) : Int) := by
    show (IntV.min k1.item (k2.item - r3.item * k3.item)).toInt = _
    rw [IntV.min_toInt, HK1, hcsz]
    rw [chunk_size_eq hd (chunkOf_lt hd (i : Nat)), Nat.cast_min]
  -- line 177: `1` as an `int64`
  refine safe_conv64 (z := (IntV.ofNat 1 : I32)) ?_
  intro o3 ho3
  have HO3 : o3.item.toInt = 1 := by
    rw [ho3, IntV.I32.toInt_ofNat (by norm_num)]
    norm_num
  -- line 178: `seq_idx_last`, the last `seq_idx` entry of the chunk
  have hone64 : (IntV.ofInt 1 : I64).toInt = 1 := I64.one
  have hmin1 : 1 ≤ endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk := by
    rw [chunk_size_eq hd (chunkOf_lt hd (i : Nat))]
    exact le_min hck (by have := c_chunk_lt hd (chunkOf_lt hd (i : Nat)); omega)
  have hge : chunkOf d i * d.chunk ≤ endTok d (chunkOf d i) := by
    have hlt := c_chunk_lt hd (chunkOf_lt hd (i : Nat))
    unfold endTok
    have h5 : chunkOf d i * d.chunk < min ((chunkOf d i + 1) * d.chunk) d.seqlen := by
      apply lt_min
      · have h6 : (chunkOf d i + 1) * d.chunk = chunkOf d i * d.chunk + d.chunk := by ring
        omega
      · exact hlt
    omega
  have hlsz : (((tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
      - IntV.ofInt 1) * o3.item).toInt
      = ((((endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk) - 1) * 1 : Nat) : Int) := by
    have hs := I64.sub_nat' hlim hone64 hmin1 (by
      have h3 : endTok d (chunkOf d i) < d.seqlen := endTok_lt_seqlen hd (chunkOf_lt hd (i : Nat))
      omega)
    rw [I64.mul_nat' hs HO3 (by
      have h3 : endTok d (chunkOf d i) < d.seqlen := endTok_lt_seqlen hd (chunkOf_lt hd (i : Nat))
      have h4 : ((endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk) - 1) * 1
          = (endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk) - 1 := Nat.mul_one _
      have h5 : (endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk) - 1 ≤ endTok d (chunkOf d i) := by
        omega
      rw [h4]
      have hb : endTok d (chunkOf d i) < 2 ^ 63 :=
        lt_of_lt_of_le h3 (le_of_lt (lt_of_lt_of_le hsl31 (show (2 : Nat) ^ 31 ≤ 2 ^ 63 by norm_num)))
      exact lt_of_le_of_lt h5 hb)]
  have hseqb : (d.seqIdx.ptr 0 + ((r1.item - r3.item * r2.item) * s7.item + r3.item * c5.item * o1.item))
      = d.seqIdx.ptr (chunkOf d i * d.chunk) := by
    have t1 : ((r1.item - r3.item * r2.item) * s7.item).toInt = 0 := PtrAux.toInt_mul_zero_left hpb
    have t2 : (r3.item * c5.item).toInt = ((chunkOf d i * d.chunk : Nat) : Int) :=
      IntV.I64.mul_nat H3 HC5 (PtrAux.lt_63_of_lt_31 hcc31)
    have t3 : (r3.item * c5.item * o1.item).toInt = (((chunkOf d i * d.chunk) * 1 : Nat) : Int) :=
      IntV.I64.mul_nat t2 HO1 (by omega)
    have t4 : ((r1.item - r3.item * r2.item) * s7.item + r3.item * c5.item * o1.item).toInt
        = ((0 + (chunkOf d i * d.chunk) * 1 : Nat) : Int) :=
      IntV.I64.add_nat t1 t3 (by omega)
    apply PtrAux.ptr_add_eq_ptr
    rw [show (chunkOf d i * d.chunk : Nat) = (0 + (chunkOf d i * d.chunk) * 1) by simp]
    exact t4
  have hseqidx : (chunkOf d i * d.chunk : Int)
      + (((endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk) - 1) * 1 : Nat)
      = (endTok d (chunkOf d i) : Int) := by
    push_cast
    omega
  have hseqaddr : ((d.seqIdx.ptr 0 + ((r1.item - r3.item * r2.item) * s7.item + r3.item * c5.item * o1.item)) +
      ((tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
        - IntV.ofInt 1) * o3.item).addr
      = (d.seqIdx.ptr (endTok d (chunkOf d i))).addr := by
    have hp : (d.seqIdx.ptr 0 + ((r1.item - r3.item * r2.item) * s7.item + r3.item * c5.item * o1.item)) +
        ((tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
          - IntV.ofInt 1) * o3.item
        = d.seqIdx.ptr (endTok d (chunkOf d i)) := by
      rw [hseqb]
      show (d.seqIdx.ptr (chunkOf d i * d.chunk)).offset
          ((((tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
            - IntV.ofInt 1) * o3.item)).toInt
          = d.seqIdx.ptr (endTok d (chunkOf d i))
      rw [hlsz, Buf.ptr_offset, hseqidx]
    exact congrArg Ptr.addr hp
  have hseqlen : endTok d (chunkOf d i) < d.seqIdx.len :=
    seqIdx_lt hd (endTok_lt_seqlen hd (chunkOf_lt hd (i : Nat)))
  obtain ⟨v1, hv1⟩ := d.seqIdx.read_some hvq hseqlen
  refine safe_load1 (E := E) (R := R d init i) (W := W d init i) (Q := fun x m => Q d init i m)
    (m := init) (v0 := v1) (p := d.seqIdx.ptr 0 +
      ((r1.item - r3.item * r2.item) * s7.item + r3.item * c5.item * o1.item) +
      ((tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
        - IntV.ofInt 1) * o3.item) ?_ ?_ ?_
  · rw [hseqaddr]; exact hv1
  · rw [hseqaddr]
    exact span_sub_R d init i hd.disj.2.2.2.2 hseqlen
  -- ### the accumulation loop
  have hhi_le : (tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item.toInt
      ≤ (d.chunk : Int) := by
    rw [hlim, chunk_size_eq hd (chunkOf_lt hd (i : Nat))]
    exact_mod_cast Nat.min_le_left _ _
  have hbk : (IntV.ofNat BK : I64).toInt = 64 := by
    rw [I64.ofNat_nat (by norm_num [BK])]; norm_num [BK]
  have htop : IntV.maxInt true 64 = (2 : Int) ^ 63 - 1 := by
    norm_num [IntV.maxInt]
  have hz0 : (IntV.ofInt 0 : I64).toInt = 0 := by
    rw [IntV.toInt_ofInt (by norm_num) (by norm_num [IntV.SFits])]
  have hrange : tl.RangeOk (IntV.ofInt 0 : I64)
      (tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
      (IntV.ofNat BK : I64) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hbk]; norm_num
    · rw [htop, hbk]
      have hd31 : (d.chunk : Int) < 2 ^ 31 := by
        have := hch31; exact_mod_cast this
      omega
    · rw [htop, hz0]; norm_num
    · rw [htop, hbk]; norm_num
  refine (safe_forRange (E := E) (R := R d init i) (Q := fun x m => Q d init i m) (m := init)
      (lo := (IntV.ofInt 0 : I64))
      (hi := (tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item)
      (step := (IntV.ofNat BK : I64))
      (I := fun t st m => Inv d init i t st m) (Wt := fun _ => ∅) (Wk := W d init i)
      hrange ?_ ?_ ?_ ?_ ?_).mono (fun _ _ h => h) ?_
  · intro t u _ _; simp
  · rw [iterW_empty]; simp
  · -- h0: the loop invariant holds at the start
    exact h0_init (d := d) (init := init) (i := i) hd H3 H5 H6 HC1 HC2 HC3 HC4 HC5
      HS1 HS2 HS3 HS4 HS5 HS6 HS7 HO1 hpb
  · -- hstep: one iteration of the loop preserves the invariant
    intro t htrip st m₁ hinv
    obtain ⟨rfl, xh, bh, hP1, hP2, hP3, hP4, hP5, hxcl, hbcl, hacccl⟩ := hinv
    have hc_lt : chunkOf d i < nc d := chunkOf_lt hd (i : Nat)
    -- the number of tokens of the chunk
    set Nch : Nat := endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk with hNch
    have hNpos : 0 < Nch := by rw [hNch]; omega
    have hNle : Nch ≤ 65536 := by
      rw [hNch, chunk_size_eq hd hc_lt]
      have h := hd.chunk_le
      omega
    -- the trip count of the loop is `steps`
    have htripeq : tl.tripCount (IntV.ofInt 0 : I64).toInt
        (tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item.toInt
        (IntV.ofNat BK : I64).toInt = steps d (chunkOf d i) := by
      have h := tripCount_eq hd (chunkOf_lt hd (i : Nat))
      rw [show ((BK : Nat) : Int) = 64 from by norm_num [BK]] at h
      have hcast : ((Nch : Nat) : Int)
          = (↑(endTok d (chunkOf d i)) + 1 - ↑(chunkOf d i) * ↑d.chunk) := by
        rw [hNch, Nat.cast_sub (by omega), Nat.cast_add, Nat.cast_one, Nat.cast_mul]
      rw [hz0, hlim, hbk, hcast]
      exact h
    have hts : t < steps d (chunkOf d i) := by rw [← htripeq]; exact htrip
    have hsteps_le : steps d (chunkOf d i) ≤ 1024 := steps_le hd hc_lt
    have ht1024 : t ≤ 1024 := by omega
    have hsteps_def : steps d (chunkOf d i) = (Nch + BK - 1) / BK := by rw [hNch]; rfl
    have htBK : t * BK < Nch := by
      rw [hsteps_def] at hts
      have hb : 0 < BK := by norm_num [BK]
      by_contra hcon
      have hcon' : Nch ≤ t * BK := by omega
      have h2 : (Nch + BK - 1) / BK < t + 1 := by
        rw [Nat.div_lt_iff_lt_mul hb]
        simp only [BK] at hcon' ⊢
        omega
      omega
    -- the loop bound and the iteration counter, as `int64` values
    set HI : I64 :=
      (tl.iminimum (Tensor.scalar k1.item) (Tensor.scalar (k2.item - r3.item * k3.item))).item
      with hHIdef
    have hHI : HI.toInt = (Nch : Int) := by rw [hHIdef, hlim, hNch]
    set K0 : I64 :=
      IntV.ofInt ((IntV.ofInt 0).toInt + (t : Int) * (IntV.ofNat BK : I64).toInt) with hK0def
    have hK0t : K0.toInt = ((t * BK : Nat) : Int) := by
      rw [hK0def]
      have hx : (IntV.ofInt 0 : I64).toInt + (t : Int) * (IntV.ofNat BK : I64).toInt
          = ((t * BK : Nat) : Int) := by
        rw [hz0, hbk]; push_cast; norm_num [BK]
      rw [hx]
      refine IntV.toInt_ofInt (by norm_num) ?_
      have hle : ((t * BK : Nat) : Int) ≤ 65536 := by
        have h1 : t * BK ≤ 1024 * BK := Nat.mul_le_mul_right BK ht1024
        have h2 : 1024 * BK = 65536 := by norm_num [BK]
        omega
      constructor <;> omega
    have hNK : (HI - K0).toInt = ((Nch - t * BK : Nat) : Int) :=
      I64.sub_nat' hHI hK0t (by omega) (by omega)
    -- ### the conversion of `tl.arange` to `int64`
    refine safe_convert (E := E) (R := R d m₁ i) (W := ∅)
      (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) (m := m₁) (d := .i32) (d' := .i64) ?_ ?_
    · intro ix
      exact ⟨IntV.ofInt ((tl.arange 0 BK).get ix).toInt, rfl⟩
    · intro offs_k64 hoffs
      have hOffJ : ∀ j : Fin BK, (offs_k64.get1 j).toInt = ((j : Nat) : Int) := by
        intro j
        have h0 : offs_k64.get1 j = (IntV.ofInt ((tl.arange 0 BK).get1 j).toInt : I64) :=
          hoffs (j, ())
        rw [h0]
        have hj : (((tl.arange 0 BK).get1 j).toInt) = ((j : Nat) : Int) := by
          simp only [tl.arange, Tensor.get1_arange]
          rw [IntV.toInt_ofInt (w := 32) (by norm_num) (by
            have h := j.isLt
            simp only [BK] at h ⊢
            constructor <;> omega)]
          simp
        rw [hj]
        exact IntV.toInt_ofInt (w := 64) (by norm_num) (by
          have h := j.isLt
          simp only [BK] at h ⊢
          constructor <;> omega)
      have hOffG : ∀ j : Fin BK, (offs_k64.get (j, ())).toInt = ((j : Nat) : Int) :=
        fun j => by simpa only [Tensor.get1] using hOffJ j
      -- an enabled lane of the token-axis mask is inside the chunk
      have hkOf : ∀ j : Fin BK,
          (offs_k64.ltS (HI - K0)).get (j, ()) = true → (j : Nat) < Nch - t * BK := by
        intro j hm
        have hm' : (decide ((j : Nat) < Nch - t * BK)) = true := by
          rw [← offsMaskEq hNK hOffG j]
          exact hm
        exact of_decide_eq_true hm'
      let MASKX : Tensor [BM, BK] Bool :=
        (((r5.item * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK).ltS
          (IntV.ofNat d.headdim)).and ((offs_k64.ltS (HI - K0)).row BM)
      let GX : Idx [BM, BK] → Nat := fun ix =>
        xIdx d (chunkOf d i * d.chunk + (t * BK + (ix.2.1 : Nat)))
          (headOf d i) (pmOf d i * BM + (ix.1 : Nat))
      have hfull : ∀ i : Idx [BM, BK],
          ((tl.full [BM, BK] ((FloatV.ofPat (0 : Nat) : F64).litF32)).get i)
            = (FloatV.ofPat (0 : Nat) : F32) := by
        intro ix
        simp only [tl.full, Tensor.get_full, litF32_zero]
      -- ### the `f32` zero tile converted to `bfloat16` (`x`'s masked-off lanes)
      refine safe_convert (E := E) (R := R d m₁ i) (W := ∅)
        (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) (d := .f32) (d' := .bf16) ?_ ?_
      · intro ix
        refine ⟨(FloatV.ofPat (0 : Nat) : BF16), ?_⟩
        rw [hfull ix]
        simp only [ConvRel, ofFloatRel, FloatRel]
        rw [ite_eq_right (by decide : Format.binary32 ≠ Format.bfloat16)]
        exact Or.inl ⟨by simp [Format.Std],
          by simp [Format.Std], Or.inl cvt_f32_bf16_zero⟩
      · intro xb hxb
        have hxb0 : ∀ (a : Fin BM) (j : Fin BK),
            xb.get2 a j = (FloatV.ofPat (0 : Nat) : BF16) := by
          intro a j
          refine convRel_f32_bf16_zero (P := E.P) ?_
          have h := hxb (a, j, ())
          rw [hfull (a, j, ())] at h
          exact h
        have hmaskX : ∀ (a : Fin BM) (j : Fin BK), MASKX.get2 a j =
            (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
              decide ((j : Nat) < Nch - t * BK)) :=
          fun a j => xMaskEq hd H5 hNK hOffG a j
        have hxg : ∀ (a : Fin BM) (j : Fin BK), MASKX.get2 a j = true →
            xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat))) (headOf d i)
              (pmOf d i * BM + (a : Nat)) < d.x.len := by
          intro a j hm
          have hb : (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
              decide ((j : Nat) < Nch - t * BK)) = true := by
            rw [← hmaskX a j]
            exact hm
          rw [Bool.and_eq_true] at hb
          have hcol : pmOf d i * BM + (a : Nat) < d.headdim := of_decide_eq_true hb.1
          have hj : (j : Nat) < Nch - t * BK := of_decide_eq_true hb.2
          have htk : chunkOf d i * d.chunk + (t * BK + (j : Nat)) < d.seqlen := by
            have h1 : t * BK + (j : Nat) < Nch := by omega
            have h2 : t * BK + (j : Nat) < endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk := by
              simpa [hNch] using h1
            have h3 := tok_lt (c := chunkOf d i) (t := t) (k := (j : Nat)) hd hc_lt h2
            omega
          exact xIdx_lt hd htk hhh hcol
        have hP1' : ∀ (a : Fin BM) (j : Fin BK), (st.1.get (a, j, ())).addr =
            (d.x.ptr (xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
              (headOf d i) (pmOf d i * BM + (a : Nat)))).addr := fun a j => hP1 a j
        -- ### the `x` tile
        refine safe_load (E := E) (R := R d m₁ i) (W := ∅)
          (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂)
          (ptrs := st.1) (mask := some MASKX) (other := some xb) ?_ ?_ ?_
        · intro ix a ha
          refine read_some_masked (d' := .bf16) (b := d.x) hvx (g := GX) ?_ ?_ ix a ha
          · intro ix
            obtain ⟨a, j, u⟩ := ix
            cases u
            exact hP1' a j
          · intro ix hm
            obtain ⟨a, j, u⟩ := ix
            cases u
            exact hxg a j hm
        · exact accBytes_R_of_masked (init := m₁) (d' := .bf16) (b := d.x) (ptrs := st.1)
            (mask := MASKX) (g := GX) i hd.disj.1
            (fun ix => by
              obtain ⟨a, j, u⟩ := ix
              cases u
              exact hP1' a j)
            (fun ix hm => by
              obtain ⟨a, j, u⟩ := ix
              cases u
              exact hxg a j hm)
        · intro xt hxt
          have hxtr : ∀ (a : Fin BM) (j : Fin BK), MASKX.get2 a j = true →
              ∃ v, m₁.read .bf16 ((d.x.ptr (xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
                (headOf d i) (pmOf d i * BM + (a : Nat)))).addr) = some v ∧ xt.get2 a j = v := by
            intro a j hb
            refine ⟨xt.get2 a j, ?_, rfl⟩
            have h := load_read hxt (i := (a, j, ())) hb
            rwa [hP1' a j] at h
          have hxt0 : ∀ (a : Fin BM) (j : Fin BK), MASKX.get2 a j = false →
              xt.get2 a j = (FloatV.ofPat (0 : Nat) : BF16) := by
            intro a j hb
            exact (load_mask_false hxt (i := (a, j, ())) hb).trans (hxb0 a j)
          -- ### the extended `x` table: the earlier tokens, then the `BK` tokens of this step
          let xh' : Nat → Fin BM → ℚ := fun s a =>
            if h : s < t * BK then xh s a
            else if h2 : s - t * BK < BK then (xt.get2 a ⟨s - t * BK, h2⟩).toRat else 0
          have hxagree : ∀ (s : Nat) (a : Fin BM), s < t * BK → xh' s a = xh s a := by
            intro s a hs
            simp only [xh']
            rw [dite_eq_left hs]
          have hxnew : ∀ (a : Fin BM) (j : Fin BK),
              xh' (t * BK + (j : Nat)) a = (xt.get2 a j).toRat := by
            intro a j
            have h1 : ¬ (t * BK + (j : Nat) < t * BK) := by omega
            have h2 : t * BK + (j : Nat) - t * BK < BK := by
              have := j.isLt
              omega
            simp only [xh']
            rw [dite_eq_right h1, dite_eq_left h2]
            have hidx : (⟨t * BK + (j : Nat) - t * BK, h2⟩ : Fin BK) = j := by
              ext
              show t * BK + (j : Nat) - t * BK = (j : Nat)
              exact Nat.add_sub_cancel_left _ _
            rw [hidx]
          have hxcl' := xClause (d := d) (init := m₁) hd (i := i) (t := t)
            (xh := xh) (xh' := xh') (xt := xt) (mb := fun a j => MASKX.get2 a j) (Nch := Nch)
            hNch hmaskX hxcl hxagree hxnew hxtr hxt0
          -- ### the `f32` zero tile converted to `bfloat16` (`b`'s masked-off lanes)
          have hfullB : ∀ ix : Idx [BK, BN],
              ((tl.full [BK, BN] ((FloatV.ofPat (0 : Nat) : F64).litF32)).get ix)
                = (FloatV.ofPat (0 : Nat) : F32) := by
            intro ix
            simp only [tl.full, Tensor.get_full, litF32_zero]
          refine safe_convert (E := E) (R := R d m₁ i) (W := ∅)
            (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) (d := .f32) (d' := .bf16) ?_ ?_
          · intro ix
            refine ⟨(FloatV.ofPat (0 : Nat) : BF16), ?_⟩
            rw [hfullB ix]
            simp only [ConvRel, ofFloatRel, FloatRel]
            rw [ite_eq_right (by decide : Format.binary32 ≠ Format.bfloat16)]
            exact Or.inl ⟨by simp [Format.Std],
              by simp [Format.Std], Or.inl cvt_f32_bf16_zero⟩
          · intro bbo hbbo
            have hbo0 : ∀ (j : Fin BK) (n : Fin BN),
                bbo.get2 j n = (FloatV.ofPat (0 : Nat) : BF16) := by
              intro j n
              refine convRel_f32_bf16_zero (P := E.P) ?_
              have h := hbbo (j, n, ())
              rw [hfullB (j, n, ())] at h
              exact h
            -- ### the `b` tile
            let MASKB : Tensor [BK, BN] Bool :=
              ((offs_k64.ltS (HI - K0)).col BN).and
                (((r6.item * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK).ltS
                  (IntV.ofNat d.dstate))
            let GB : Idx [BK, BN] → Nat := fun ix =>
              bIdx d (chunkOf d i * d.chunk + (t * BK + (ix.1 : Nat)))
                (grp d (headOf d i)) (pnOf d i * BN + (ix.2.1 : Nat))
            have hNchle : Nch ≤ d.chunk := by
              rw [hNch, chunk_size_eq hd hc_lt]
              exact Nat.min_le_left _ _
            have hchunkK : ∀ j : Fin BK, (j : Nat) < Nch - t * BK →
                t * BK + (j : Nat) < d.chunk := by
              intro j hj
              omega
            have hgl : grp d (headOf d i) < d.ngroups := by
              obtain ⟨kk, hkk⟩ := hd.groups_dvd
              have hk1 : 0 < kk := by
                rcases Nat.eq_zero_or_pos kk with h0 | h0
                · rw [h0, Nat.mul_zero] at hkk
                  omega
                · exact h0
              have hrk : ratio d = kk := by
                unfold ratio
                rw [hkk, Nat.mul_comm d.ngroups kk]
                exact Nat.mul_div_left kk hd.groups_pos
              unfold grp
              rw [hrk, Nat.div_lt_iff_lt_mul hk1]
              have hhh' := hhh
              rw [hkk] at hhh'
              exact hhh'
            have hmaskB : ∀ (j : Fin BK) (n : Fin BN), MASKB.get2 j n =
                (decide ((j : Nat) < Nch - t * BK) &&
                  decide (pnOf d i * BN + (n : Nat) < d.dstate)) :=
              fun j n => bMaskEq hd H6 hNK hOffG j n
            have hbg : ∀ ix : Idx [BK, BN], MASKB.get ix = true → GB ix < d.b.len := by
              intro ix hm
              obtain ⟨j, n, u⟩ := ix
              cases u
              have hb : (decide ((j : Nat) < Nch - t * BK) &&
                  decide (pnOf d i * BN + (n : Nat) < d.dstate)) = true := by
                rw [← hmaskB j n]
                exact hm
              rw [Bool.and_eq_true] at hb
              have hjk : (j : Nat) < Nch - t * BK := of_decide_eq_true hb.1
              have hn : pnOf d i * BN + (n : Nat) < d.dstate := of_decide_eq_true hb.2
              have ht' : chunkOf d i * d.chunk + (t * BK + (j : Nat)) < d.seqlen := by
                have h := tok_lt (c := chunkOf d i) (t := t) (k := (j : Nat)) hd hc_lt
                  (by simpa [hNch] using (by omega : t * BK + (j : Nat) < Nch))
                omega
              exact bIdx_lt hd ht' hgl hn
            have hP2' : ∀ (j : Fin BK) (n : Fin BN), (st.2.1.get2 j n).addr =
                (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
                  (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr := fun j n => hP2 j n
            refine safe_load (E := E) (R := R d m₁ i) (W := ∅)
              (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂)
              (ptrs := st.2.1) (mask := some MASKB) (other := some bbo) ?_ ?_ ?_
            · intro ix a ha
              refine read_some_masked (d' := .bf16) (b := d.b) hvb (g := GB) ?_ ?_ ix a ha
              · intro ix
                obtain ⟨j, n, u⟩ := ix
                cases u
                exact hP2' j n
              · intro ix hm
                exact hbg ix hm
            · exact accBytes_R_of_masked (init := m₁) (d' := .bf16) (b := d.b) (ptrs := st.2.1)
                (mask := MASKB) (g := GB) i hd.disj.2.1
                (fun ix => by
                  obtain ⟨j, n, u⟩ := ix
                  cases u
                  exact hP2' j n)
                (fun ix hm => hbg ix hm)
            · intro b0f hb0
              -- ### the `b` tile converted to `float32`
              refine safe_cvt (E := E) (R := R d m₁ i) (W := ∅)
                (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) ?_ ?_
              · exact ⟨by simp [Format.Std], by simp [Format.Std],
                  Or.inl rfl⟩
              · intro b32f hb32
                -- ### the three `[BK]` loads (`dA_cumsum`, `seq_idx`, `dt`)
                refine safe_load (E := E) (R := R d m₁ i) (W := ∅)
                  (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂)
                  (ptrs := st.2.2.2.1) (mask := some (offs_k64.ltS (HI - K0)))
                  (other := some (bc (FloatV.litF32 (FloatV.ofPat 0)))) ?_ ?_ ?_
                · intro ix a ha
                  refine read_some_masked (d' := .f32) (b := d.dA) hvdA
                    (g := fun (ix : Idx [BK]) => dIdx d (headOf d i) (chunkOf d i) (t * BK + (ix.1 : Nat)))
                    ?_ ?_ ix a ha
                  · intro ix
                    obtain ⟨j, u⟩ := ix
                    cases u
                    exact hP4 j
                  · intro ix hm
                    obtain ⟨j, u⟩ := ix
                    cases u
                    have hk : (j : Nat) < Nch - t * BK := by
                      exact hkOf j hm
                    exact lt_of_lt_of_le (dIdx_lt hhh hc_lt (hchunkK j hk)) hd.dA_len
                · exact accBytes_R_of_masked (init := m₁) (d' := .f32) (b := d.dA)
                    (ptrs := st.2.2.2.1) (mask := offs_k64.ltS (HI - K0))
                    (g := fun (ix : Idx [BK]) => dIdx d (headOf d i) (chunkOf d i) (t * BK + (ix.1 : Nat)))
                    i hd.disj.2.2.2.1
                    (fun ix => by obtain ⟨j, u⟩ := ix; cases u; exact hP4 j)
                    (fun ix hm => by
                      obtain ⟨j, u⟩ := ix
                      cases u
                      have hk : (j : Nat) < Nch - t * BK := by
                        exact hkOf j hm
                      exact lt_of_lt_of_le (dIdx_lt hhh hc_lt (hchunkK j hk)) hd.dA_len)
                · intro dAk hdAk
                  refine safe_load (E := E) (R := R d m₁ i) (W := ∅)
                    (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂)
                    (ptrs := st.2.2.2.2.1) (mask := some (offs_k64.ltS (HI - K0)))
                    (other := some (bc (IntV.ofInt (-1) : I32))) ?_ ?_ ?_
                  · intro ix a ha
                    refine read_some_masked (d' := .i32) (b := d.seqIdx) hvq
                      (g := fun (ix : Idx [BK]) => chunkOf d i * d.chunk + (t * BK + (ix.1 : Nat)))
                      ?_ ?_ ix a ha
                    · intro ix
                      obtain ⟨j, u⟩ := ix
                      cases u
                      exact hP5 j
                    · intro ix hm
                      obtain ⟨j, u⟩ := ix
                      cases u
                      have hk : (j : Nat) < Nch - t * BK := by
                        exact hkOf j hm
                      have ht' : chunkOf d i * d.chunk + (t * BK + (j : Nat)) < d.seqlen := by
                        have h := tok_lt (c := chunkOf d i) (t := t) (k := (j : Nat)) hd hc_lt
                          (by simpa [hNch] using (by omega : t * BK + (j : Nat) < Nch))
                        omega
                      exact seqIdx_lt hd ht'
                  · exact accBytes_R_of_masked (init := m₁) (d' := .i32) (b := d.seqIdx)
                      (ptrs := st.2.2.2.2.1) (mask := offs_k64.ltS (HI - K0))
                      (g := fun (ix : Idx [BK]) => chunkOf d i * d.chunk + (t * BK + (ix.1 : Nat)))
                      i hd.disj.2.2.2.2
                      (fun ix => by obtain ⟨j, u⟩ := ix; cases u; exact hP5 j)
                      (fun ix hm => by
                        obtain ⟨j, u⟩ := ix
                        cases u
                        have hk : (j : Nat) < Nch - t * BK := by
                          exact hkOf j hm
                        have ht' : chunkOf d i * d.chunk + (t * BK + (j : Nat)) < d.seqlen := by
                          have h := tok_lt (c := chunkOf d i) (t := t) (k := (j : Nat)) hd hc_lt
                            (by simpa [hNch] using (by omega : t * BK + (j : Nat) < Nch))
                          omega
                        exact seqIdx_lt hd ht')
                  · intro sq hsq
                    refine safe_load (E := E) (R := R d m₁ i) (W := ∅)
                      (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂)
                      (ptrs := st.2.2.1) (mask := some (offs_k64.ltS (HI - K0)))
                      (other := some (bc (FloatV.litF32 (FloatV.ofPat 0)))) ?_ ?_ ?_
                    · intro ix a ha
                      refine read_some_masked (d' := .f32) (b := d.dt) hvdt
                        (g := fun (ix : Idx [BK]) => dIdx d (headOf d i) (chunkOf d i) (t * BK + (ix.1 : Nat)))
                        ?_ ?_ ix a ha
                      · intro ix
                        obtain ⟨j, u⟩ := ix
                        cases u
                        exact hP3 j
                      · intro ix hm
                        obtain ⟨j, u⟩ := ix
                        cases u
                        have hk : (j : Nat) < Nch - t * BK := by
                          exact hkOf j hm
                        exact lt_of_lt_of_le (dIdx_lt hhh hc_lt (hchunkK j hk)) hd.dt_len
                    · exact accBytes_R_of_masked (init := m₁) (d' := .f32) (b := d.dt)
                        (ptrs := st.2.2.1) (mask := offs_k64.ltS (HI - K0))
                        (g := fun (ix : Idx [BK]) => dIdx d (headOf d i) (chunkOf d i) (t * BK + (ix.1 : Nat)))
                        i hd.disj.2.2.1
                        (fun ix => by obtain ⟨j, u⟩ := ix; cases u; exact hP3 j)
                        (fun ix hm => by
                          obtain ⟨j, u⟩ := ix
                          cases u
                          have hk : (j : Nat) < Nch - t * BK := by
                            exact hkOf j hm
                          exact lt_of_lt_of_le (dIdx_lt hhh hc_lt (hchunkK j hk)) hd.dt_len)
                    · intro dtk hdtk
                      -- ### `dA_cs_last - dA_cs_k`, `exp`, `* dt_k`
                      refine safe_sub (E := E) (R := R d m₁ i) (W := ∅)
                        (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) ?_
                      intro dsub hsub
                      refine safe_math (E := E) (R := R d m₁ i) (W := ∅)
                        (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) (fn := .exp) ?_ ?_
                      · intro ix
                        exact hexp (dsub.get ix)
                      · intro dexp hdexp
                        refine safe_mul (E := E) (R := R d m₁ i) (W := ∅)
                          (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) ?_
                        intro w hw
                        -- ### `b * scale[:, None]`
                        refine safe_mul (E := E) (R := R d m₁ i) (W := ∅)
                          (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) ?_
                        intro b2f hb2
                        -- ### `b.to(bfloat16)`
                        refine safe_cvt (E := E) (R := R d m₁ i) (W := ∅)
                          (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) ?_ ?_
                        · exact ⟨by simp [Format.Std],
                            by simp [Format.Std], Or.inl rfl⟩
                        · intro bt hbt
                          -- ### the extended `b` table
                          let bh' : Nat → Fin BN → ℚ := fun s n =>
                            if h : s < t * BK then bh s n
                            else if h2 : s - t * BK < BK then
                              (bt.get2 ⟨s - t * BK, h2⟩ n).toRat else 0
                          have hbagree : ∀ (s : Nat) (n : Fin BN),
                              s < t * BK → bh' s n = bh s n := by
                            intro s n hs
                            simp only [bh']
                            rw [dite_eq_left hs]
                          have hbnew : ∀ (j : Fin BK) (n : Fin BN),
                              bh' (t * BK + (j : Nat)) n = (bt.get2 j n).toRat := by
                            intro j n
                            have h1 : ¬ (t * BK + (j : Nat) < t * BK) := by omega
                            have h2 : t * BK + (j : Nat) - t * BK < BK := by
                              have := j.isLt
                              omega
                            simp only [bh']
                            rw [dite_eq_right h1, dite_eq_left h2]
                            have hidx : (⟨t * BK + (j : Nat) - t * BK, h2⟩ : Fin BK) = j := by
                              ext
                              show t * BK + (j : Nat) - t * BK = (j : Nat)
                              exact Nat.add_sub_cancel_left _ _
                            rw [hidx]
                          -- ### the `b` clause and the finiteness of the `b` operand
                          have hbAll := fun (hV : Vals m₁ d) =>
                            bClause (d := d) (hd := hd) (E := E) (m₁ := m₁) (i := i) (t := t)
                              (st := st) (Nch := Nch) (bt := bt) (b0f := b0f) (bo := bbo)
                              (b32f := b32f) (b2f := b2f) (dAk := dAk) (dtk := dtk)
                              (dsub := dsub) (dexp := dexp) (w := w) (sq := sq)
                              (v0 := v0) (v1 := v1) (mb := MASKB) (bh := bh) (bh' := bh')
                              (hNch := hNch) (hseq := endTok_lt_seqlen hd hc_lt)
                              (hmeta := hmeta) (hV := hV) (hchunk := chunk_pos hd)
                              (hhh := hhh) (hgl := hgl) (hv0 := hv0) (hv1 := hv1)
                              (hP2' := hP2') (hP3 := hP3) (hP4 := hP4) (hP5 := hP5)
                              (hchunkK := hchunkK) (hmaskB := hmaskB) (hbo0 := hbo0)
                              (hb0 := hb0)
                              (_hmaskA := fun j hj => by
                                have h := offsMaskEq (t := t) (HI := HI) (K0 := K0)
                                  (offs := offs_k64) (Nch := Nch) hNK hOffG j
                                have h' : (offs_k64.ltS (HI - K0)).get (j, ()) =
                                    decide ((j : Nat) < Nch - t * BK) := h
                                rw [h']
                                exact decide_eq_true hj)
                              (hmaskAo := fun j => by
                                have h := offsMaskEq (t := t) (HI := HI) (K0 := K0)
                                  (offs := offs_k64) (Nch := Nch) hNK hOffG j
                                have h' : (offs_k64.ltS (HI - K0)).get (j, ()) =
                                    decide ((j : Nat) < Nch - t * BK) := h
                                exact h')
                              (_hzo0 := fun ix => (Tensor.get_full _ ix).trans litF32_zero)
                              (hzoi0 := fun ix => Tensor.get_full _ ix)
                              (hdAk := hdAk) (hsq := hsq) (hdtk := hdtk) (hb32 := hb32)
                              (hsub := hsub) (hdexp := hdexp) (hw := hw) (hb2 := hb2)
                              (hbt := hbt) (hbagree := hbagree) (hbnew := hbnew)
                              (hbcl := fun s n hs => hbcl hV s n hs) (hexpacc := hexpacc)
                          -- ### `acc += tl.dot(x, b)`
                          refine safe_dot_add (E := E) (R := R d m₁ i) (W := ∅)
                            (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂)
                            (cfg := dotCfg) (a := xt) (b := bt) (c := st.2.2.2.2.2)
                            (hex := hdot xt bt st.2.2.2.2.2) ?_
                          intro acc' hracc
                          refine safe_ret (E := E) (R := R d m₁ i) (W := ∅)
                            (Q := fun st' m₂ => Inv d m₁ i (t + 1) st' m₂) (m := m₁) ?_
                          refine ⟨rfl, xh', bh', ?_, ?_, ?_, ?_, ?_, hxcl', ?_, ?_⟩
                          · intro a j
                            exact ptrStepX (d := d) (i := i) (t := t) (st := st)
                              (hoff := fun a j => offX_val hd (a, j, ())) hP1 a j
                          · intro j n
                            exact ptrStepB (d := d) (i := i) (t := t) (st := st)
                              (hoff := fun j n => offB_val hd (j, n, ())) hP2 j n
                          · intro j
                            exact ptrStepDt (d := d) (i := i) (t := t) (st := st)
                              (hoff := fun j => offChunk_val hd (j, ())) hP3 j
                          · intro j
                            exact ptrStepDA (d := d) (i := i) (t := t) (st := st)
                              (hoff := fun j => offChunk_val hd (j, ())) hP4 j
                          · intro j
                            exact ptrStepSeq (d := d) (i := i) (t := t) (st := st)
                              (hoff := fun j => offChunk_val hd (j, ())) hP5 j
                          · -- the `b` value clause
                            intro hV
                            exact (hbAll hV).1
                          · -- the accumulator clause
                            intro hV
                            obtain ⟨hbclnew, hbtfin⟩ := hbAll hV
                            have hxtfin : ∀ (a : Fin BM) (k : Fin BK),
                                (xt.get2 a k).isFinite = true := by
                              intro a k
                              by_cases hm : MASKX.get2 a k = true
                              · obtain ⟨v, hread, hval⟩ := hxtr a k hm
                                rw [hval]
                                have hb : (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
                                    decide ((k : Nat) < Nch - t * BK)) = true := by
                                  rw [← hmaskX a k]
                                  exact hm
                                rw [Bool.and_eq_true] at hb
                                have hp : pmOf d i * BM + (a : Nat) < d.headdim :=
                                  of_decide_eq_true hb.1
                                have hklt : (k : Nat) < Nch - t * BK :=
                                  of_decide_eq_true hb.2
                                have ht' : chunkOf d i * d.chunk + (t * BK + (k : Nat)) <
                                    d.seqlen := by
                                  have h := tok_lt (c := chunkOf d i) (t := t) (k := (k : Nat))
                                    hd hc_lt
                                    (by simpa [hNch] using (by omega : t * BK + (k : Nat) < Nch))
                                  omega
                                obtain ⟨v', hread', hv'⟩ := hV.1
                                  (chunkOf d i * d.chunk + (t * BK + (k : Nat)))
                                  (headOf d i) (pmOf d i * BM + (a : Nat)) ht' hhh hp
                                rw [hread] at hread'
                                rw [Option.some.inj hread']
                                exact hv'
                              · have hf : MASKX.get2 a k = false := by
                                  cases h' : MASKX.get2 a k
                                  · rfl
                                  · exact absurd h' hm
                                rw [hxt0 a k hf]
                                exact BF16_zero_finite
                            have ht1 : t + 1 ≤ steps d (chunkOf d i) := by omega
                            exact accClause (d := d) (hd := hd) (E := E) (m₁ := m₁) (i := i) (t := t)
                              (xh := xh) (xh' := xh') (bh := bh) (bh' := bh')
                              (xt := xt) (bt := bt) (acc := st.2.2.2.2.2) (r := acc')
                              (hV := hV) (hdotacc := hdotacc) (hxagree := hxagree)
                              (hbagree := hbagree) (hxnew := hxnew) (hbnew := hbnew)
                              (hxtfin := hxtfin) (hbtfin := hbtfin) (hr := hracc)
                              (hacc := hacccl hV) (hxcl' := hxcl') (hbcl' := hbclnew)
                              (ht1 := ht1)
  · -- hk: the code after the loop
    intro st m₁ hinv
    exact safe_tail (E := E) (init := init) hd hvs hlim hz0 hbk H3 H5 H6 hpb hinv
  · rw [iterW_empty, Set.empty_union]

end Claims.SsdChunkState
