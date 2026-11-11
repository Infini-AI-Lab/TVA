import Claim
import TVA.Proof
import Proof.Geo
import Proof.GeoBound
import Proof.Basic
import Proof.Util

/-!
# The lane arithmetic of one program

The kernel's body computes, from the program id `i`, the integers `i_bh = i % B·H`,
`i_t = i / (B·H)`, `i_b = i_bh / H` and `i_h = i_bh % H` (all in `int64`), the sequence's extent
`[bos, bos + T)` (from the tables on a variable-length launch), the lane offsets
`o_t = i_t·BT + arange(BT)` and the mask `m_t = o_t < T`, and the two pointer tensors. This file
collects the value and bound facts of that arithmetic, in terms of `row`, `batchOf`, `headOf`,
`nOf`, `chkOf`, `pos`, `en` and `elem`.
-/

namespace Claims.ChunkCumsum

open TVA

/-! ## Bounds of the instance's numbers -/

/-- `BT < 2^63`. -/
theorem bt_lt_two_63 {d : Inst} (hd : d.ok) : d.BT < 2 ^ 63 := by
  have := hd.bt_le; omega

/-- `B·H < 2^63`. -/
theorem BH_lt_two_63 {d : Inst} (hd : d.ok) : d.B * d.H < 2 ^ 63 := by
  have := hd.BH_lt; omega

/-- `T < 2^63`. -/
theorem T_lt_two_63 {d : Inst} (hd : d.ok) : d.T < 2 ^ 63 := by
  have := hd.T_lt; omega

/-- `H < 2^31`. -/
theorem H_lt_two_31 {d : Inst} (hd : d.ok) : d.H < 2 ^ 31 :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_left d.H hd.B_pos) hd.BH_lt

/-- `BT·H < 2^63`. -/
theorem BT_mul_H_lt {d : Inst} (hd : d.ok) : d.BT * d.H < 2 ^ 63 := by
  have h1 := hd.bt_le
  have h2 := H_lt_two_31 hd
  nlinarith

/-- `s` has at most `2^62` elements. -/
theorem s_len_le {d : Inst} {init : Mem} (hpre : Pre d init) : d.s.len ≤ 2 ^ 62 := by
  obtain ⟨hv, -, -, -⟩ := hpre
  obtain ⟨hb0, hb1, -, -⟩ := hv
  have hb1' : (d.s.len : Int) * 4 ≤ 2 ^ 64 := by
    have hb : (DType.f32).bytes = 4 := rfl
    rw [hb] at hb1
    omega
  have h2 : (d.s.len : Int) ≤ 2 ^ 62 := by
    have hq : ((2 : Int) ^ 64) / 4 = 2 ^ 62 := by norm_num
    nlinarith
  exact_mod_cast h2

/-- `o` has at most `2^62` elements. -/
theorem o_len_le {d : Inst} {init : Mem} (hpre : Pre d init) : d.o.len ≤ 2 ^ 62 := by
  obtain ⟨-, hv, -, -⟩ := hpre
  obtain ⟨hb0, hb1, -, -⟩ := hv
  have hb1' : (d.o.len : Int) * 4 ≤ 2 ^ 64 := by
    have hb : (DType.f32).bytes = 4 := rfl
    rw [hb] at hb1
    omega
  have h2 : (d.o.len : Int) ≤ 2 ^ 62 := by
    have hq : ((2 : Int) ^ 64) / 4 = 2 ^ 62 := by norm_num
    nlinarith
  exact_mod_cast h2

/-- The flat tensor `[B, T, H]` has at most `2^62` elements. -/
theorem BTH_le {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) :
    d.B * d.T * d.H ≤ 2 ^ 62 := le_trans hd.s_len (s_len_le hpre)

/-- `chunk_indices` rows are below `2^31`. -/
theorem row_lt_two_31 {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gridSize d) :
    row d i < 2 ^ 31 := by
  have h1 := row_lt hd hi
  have hBH : 1 ≤ d.B * d.H := Nat.mul_pos hd.B_pos hd.H_pos
  have h2 : rowsOf d ≤ gridSize d := by
    rw [gridSize_eq, Nat.mul_assoc]
    have := Nat.mul_le_mul_left (rowsOf d) hBH
    simpa using this
  have h3 := hd.grid_lt
  omega

/-- The lane offset `chk·BT + l` is below `2^23`. -/
theorem chkBT_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) {l : Nat} (hl : l < d.BT) :
    chkOf d i * d.BT + l < 2 ^ 23 := by
  have hb := bt_pos hd
  have hrow := row_lt hd hi
  have hTle := hd.T_le
  have hBTle := hd.bt_le
  cases hvv : d.varlen
  · rw [chkOf_false hvv]
    rw [rowsOf, hvv] at hrow
    have hsub : (d.T + d.BT - 1) - (d.BT - 1) = d.T := by omega
    have hq : row d i * d.BT < d.T := by
      have := (Nat.lt_div_iff_mul_lt hb).mp hrow
      omega
    omega
  · rw [chkOf_true hvv]
    rw [rowsOf, hvv] at hrow
    obtain ⟨-, -, -, hrowok, -⟩ := hT hvv
    obtain ⟨hseq0, hseqlt, hchk0, hchklt⟩ := hrowok (row d i) hrow
    set r := row d i with hr
    set q := seqAt d r with hq
    have hqN : q.toNat + 1 ≤ d.N := by omega
    have hmono : cuAt d (q.toNat + 1) ≤ cuAt d d.N := cuAt_mono_le (hT hvv) hqN (le_refl _)
    have hnn : 0 ≤ cuAt d q.toNat := cuAt_nonneg (hT hvv) (by omega)
    have hTN : cuAt d d.N ≤ (d.T : Int) := hT hvv |>.2.2.1
    have hcn : ((chkAt d r).toNat : Int) = chkAt d r := by omega
    have hmain : ((chkAt d r).toNat : Int) * (d.BT : Int) < (d.T : Int) := by
      rw [hcn]
      nlinarith
    have hq' : (chkAt d r).toNat * d.BT < d.T := by exact_mod_cast hmain
    omega

/-- A sequence's start is inside the flat time axis. -/
theorem baseOf_le {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) : baseOf d i ≤ d.B * d.T :=
  le_trans (lo_le_hi hT (nOf_lt hd hT hi)) (hi_le_BT hd hT (nOf_lt hd hT hi))

/-- The flat position of any lane is inside `B·T` plus one chunk. -/
theorem pos_le {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) {l : Nat} (hl : l < d.BT) :
    pos d i l ≤ d.B * d.T + 2 ^ 23 := by
  have h1 := baseOf_le hd hT hi
  have h2 := chkBT_lt hd hT hi hl
  unfold pos
  omega

/-- A sequence's start, times `H`, fits in `int64`. -/
theorem baseOf_mul_H_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) {init : Mem} (hpre : Pre d init) :
    baseOf d i * d.H < 2 ^ 63 := by
  have h1 := baseOf_le hd hT hi
  have h2 := BTH_le hd hpre
  have h3 : baseOf d i * d.H ≤ (d.B * d.T) * d.H := Nat.mul_le_mul_right _ h1
  have h4 : (d.B * d.T) * d.H ≤ 2 ^ 62 := h2
  omega

/-- A lane's offset, times `H`, fits in `int64`. -/
theorem chkBT_mul_H_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) {l : Nat} (hl : l < d.BT) :
    (chkOf d i * d.BT + l) * d.H < 2 ^ 63 := by
  have h1 := chkBT_lt hd hT hi hl
  have h2 := H_lt_two_31 hd
  have h3 : (chkOf d i * d.BT + l) * d.H < 2 ^ 23 * 2 ^ 31 :=
    Nat.mul_lt_mul'' h1 h2
  have h4 : (2 : Nat) ^ 23 * 2 ^ 31 = 2 ^ 54 := by norm_num
  omega

/-- Every element a lane touches is below `2^63`. -/
theorem elem_lt_two_63 {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) {l : Nat} (hl : l < d.BT) {init : Mem}
    (hpre : Pre d init) : elem d i l < 2 ^ 63 := by
  have hp := pos_le hd hT hi hl
  have hBTH := BTH_le hd hpre
  have hH := H_lt_two_31 hd
  have hh := headOf_lt d hd i
  have h1 : pos d i l * d.H ≤ (d.B * d.T + 2 ^ 23) * d.H := Nat.mul_le_mul_right _ hp
  have h2 : (d.B * d.T + 2 ^ 23) * d.H = (d.B * d.T) * d.H + 2 ^ 23 * d.H := by ring
  have h3 : 2 ^ 23 * d.H < 2 ^ 54 := by
    have := Nat.mul_lt_mul_of_pos_left hH (by norm_num : 0 < 2 ^ 23)
    have h4 : (2 : Nat) ^ 23 * 2 ^ 31 = 2 ^ 54 := by norm_num
    omega
  have h4 : (d.B * d.T) * d.H ≤ 2 ^ 62 := hBTH
  unfold elem
  omega

/-! ## The lane offsets -/

/-- `tl.arange 0 n` at lane `l` is `l`. -/
theorem arange_get_toInt {n l : Nat} (hn : n ≤ 2 ^ 20) (hl : l < n) :
    ((tl.arange 0 n).get (⟨l, hl⟩, ())).toInt = (l : Int) := by
  have hrw : (tl.arange 0 n).get (⟨l, hl⟩, ()) = IntV.ofInt (0 + (l : Int)) := rfl
  rw [hrw, IntV.toInt_ofInt (by norm_num) (by constructor <;> omega)]
  omega

/-- Two `int64` values with the same (nat) value are equal. -/
theorem intEq_of_toInt {a b : I64} {x : Nat} (ha : a.toInt = (x : Int))
    (hb : b.toInt = (x : Int)) : a = b := by
  have ha' : a.bits.toInt = (x : Int) := by simpa [IntV.toInt] using ha
  have hb' : b.bits.toInt = (x : Int) := by simpa [IntV.toInt] using hb
  have hbits : a.bits = b.bits := BitVec.eq_of_toInt_eq (by rw [ha', hb'])
  cases a; cases b; simp_all

/-! ## Values of the tables -/

/-- Entries `k < 2·NT` are inside the `chunk_indices` table. -/
theorem ci_len_le {d : Inst} (hd : d.ok) (h : d.varlen = true) {k : Nat} (hk : k < 2 * d.NT) :
    k < d.ci.len := lt_of_lt_of_le hk (hd.ci_len h)

/-- Entries `k < N + 1` are inside the `cu_seqlens` table. -/
theorem cu_len_le {d : Inst} (hd : d.ok) (h : d.varlen = true) {k : Nat} (hk : k < d.N + 1) :
    k < d.cu.len := lt_of_lt_of_le hk (hd.cu_len h)

end Claims.ChunkCumsum
