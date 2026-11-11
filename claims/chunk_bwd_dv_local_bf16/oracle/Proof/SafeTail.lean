import Claim
import TVA.Proof
import Proof.Prologue
import Proof.Setup
import Proof.KLoop
import Proof.Epi
import Proof.VLoop

/-!
# The tail of the body is safe

The tail first forms the head's base pointers (two `int64` divisions and products) and the
token offsets `o_t` of the chunk's lanes; `Setup` records their values. Then it loads the gate
block, runs the `K` loop, rounds the matrix `A` and runs the `V` loop (`safe_gLoadK`,
`safe_kLoopK`, `safe_epiK`, `safe_vLoop`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The base offsets are below `2^63`, and so are the strides. -/
theorem offs_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d) :
    qOff d r bh < 2 ^ 63 ∧ oOff d r bh < 2 ^ 63 := by
  have h1 := qk_tile_lt hd hT hr hbh
  have h2 := o_tile_lt hd hT hr hbh
  have hq := buf_len_le hpre.1
  have hdo := buf_len_le hpre.2.2.1
  have hqlen := hd.q_len
  have hdlen := hd.dO_len
  simp only [DType.bytes] at hq hdo
  omega

/-- A buffer's pointer advanced by `(bos·H + X)·K` is its element `qOff`. -/
theorem qptr_eq {d : Inst} (b : Buf .bf16) {r bh : Nat} {bos X : I64}
    (hq : qOff d r bh < 2 ^ 63) (h2 : bos.toInt = (baseOf d r bh : Int))
    (hX : X.toInt = (hk d (hvOf d bh) : Int)) (hK : 0 < d.K) (hH : d.H < 2 ^ 63) (hKK : d.K < 2 ^ 63) :
    b.ptr 0 + (bos * (IntV.ofNat d.H : I64) + X) * (IntV.ofNat d.K : I64) = b.ptr (qOff d r bh : Int) := by
  have hle : (baseOf d r bh * d.H + hk d (hvOf d bh)) ≤ qOff d r bh := by
    unfold qOff; exact Nat.le_mul_of_pos_right _ hK
  have hle2 : baseOf d r bh * d.H ≤ qOff d r bh := le_trans (Nat.le_add_right _ _) hle
  have e1 : (bos * (IntV.ofNat d.H : I64)).toInt = ((baseOf d r bh * d.H : Nat) : Int) :=
    IntV.I64.mul_nat h2 (IntV.I64.toInt_ofNat hH) (by omega)
  have e2 : (bos * (IntV.ofNat d.H : I64) + X).toInt =
      ((baseOf d r bh * d.H + hk d (hvOf d bh) : Nat) : Int) :=
    IntV.I64.add_nat e1 hX (by omega)
  have e3 : ((bos * (IntV.ofNat d.H : I64) + X) * (IntV.ofNat d.K : I64)).toInt = (qOff d r bh : Int) :=
    IntV.I64.mul_nat e2 (IntV.I64.toInt_ofNat hKK) (by unfold qOff at hq; exact hq)
  rw [Ptr.add_int, e3, Buf.ptr_offset, zero_add]

/-- A buffer's pointer advanced by `(bos·HV + i_h)·V` is its element `oOff`. -/
theorem optr_eq {d : Inst} (b : Buf .bf16) {r bh : Nat} {bos i_h : I64}
    (ho : oOff d r bh < 2 ^ 63) (h2 : bos.toInt = (baseOf d r bh : Int))
    (h4 : i_h.toInt = (hvOf d bh : Int)) (hV : 0 < d.V) (hHV : d.HV < 2 ^ 63) (hVV : d.V < 2 ^ 63) :
    b.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64) =
      b.ptr (oOff d r bh : Int) := by
  have hle : (baseOf d r bh * d.HV + hvOf d bh) ≤ oOff d r bh := by
    unfold oOff; exact Nat.le_mul_of_pos_right _ hV
  have hle2 : baseOf d r bh * d.HV ≤ oOff d r bh := le_trans (Nat.le_add_right _ _) hle
  have e1 : (bos * (IntV.ofNat d.HV : I64)).toInt = ((baseOf d r bh * d.HV : Nat) : Int) :=
    IntV.I64.mul_nat h2 (IntV.I64.toInt_ofNat hHV) (by omega)
  have e2 : (bos * (IntV.ofNat d.HV : I64) + i_h).toInt =
      ((baseOf d r bh * d.HV + hvOf d bh : Nat) : Int) :=
    IntV.I64.add_nat e1 h4 (by omega)
  have e3 : ((bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64)).toInt =
      (oOff d r bh : Int) :=
    IntV.I64.mul_nat e2 (IntV.I64.toInt_ofNat hVV) (by unfold oOff at ho; exact ho)
  rw [Ptr.add_int, e3, Buf.ptr_offset, zero_add]

/-- The gate pointer advanced by `bos·HV + i_h` is its element `gOff`. -/
theorem gptr_eq {d : Inst} (b : Buf .f32) {r bh : Nat} {bos i_h : I64}
    (ho : oOff d r bh < 2 ^ 63) (h2 : bos.toInt = (baseOf d r bh : Int))
    (h4 : i_h.toInt = (hvOf d bh : Int)) (hV : 0 < d.V) (hHV : d.HV < 2 ^ 63) :
    b.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) = b.ptr (gOff d r bh : Int) := by
  have hle : (baseOf d r bh * d.HV + hvOf d bh) ≤ oOff d r bh := by
    unfold oOff; exact Nat.le_mul_of_pos_right _ hV
  have hle2 : baseOf d r bh * d.HV ≤ oOff d r bh := le_trans (Nat.le_add_right _ _) hle
  have e1 : (bos * (IntV.ofNat d.HV : I64)).toInt = ((baseOf d r bh * d.HV : Nat) : Int) :=
    IntV.I64.mul_nat h2 (IntV.I64.toInt_ofNat hHV) (by omega)
  have e2 : (bos * (IntV.ofNat d.HV : I64) + i_h).toInt =
      ((baseOf d r bh * d.HV + hvOf d bh : Nat) : Int) :=
    IntV.I64.add_nat e1 h4 (by omega)
  rw [Ptr.add_int, e2, Buf.ptr_offset, zero_add]
  rfl

/-- **The rest of the tail** (lines 113–149): the gate block, the `K` loop, the rounded matrix
and the `V` loop. -/
theorem safe_tailRest {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T) :
    Safe ((launch d).env P) (R d r bh) (W d r bh) (fun _ m => Q d init r bh m) init
      (tailRest d q k do_ dv gb o_t T) := by
  rw [tailRest_eq]
  refine safe_gLoadK hd hpre hr hbh S fun b_g hG => ?_
  refine safe_kLoopK hP hd hpre hr hbh S fun b_A hK => ?_
  refine safe_epiK hP hd hpre hr hbh S.ot (mt_get S) hG hK fun A hA => ?_
  exact safe_vLoop hP hd hpre hr hbh S hA

/-- **The tail of the body** (lines 108–149) is safe: it reads within `R`, writes within `W`
and ends with `Q`. -/
theorem safe_tail {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {i_t bos T i_h : I64} (h1 : i_t.toInt = (chkOf d r bh : Int))
    (h2 : bos.toInt = (baseOf d r bh : Int)) (h3 : T.toInt = (lenOf d r bh : Int))
    (h4 : i_h.toInt = (hvOf d bh : Int)) :
    Safe ((launch d).env P) (R d r bh) (W d r bh) (fun _ m => Q d init r bh m) init
      (tail d i_t bos T i_h) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  obtain ⟨hqo, hoo⟩ := offs_lt hd hT hpre hr hbh
  have hHV := hd.HV_pos
  have hH := hd.H_pos
  have hdiv : 0 < d.HV / d.H := by
    obtain ⟨m, hm⟩ := hd.hv_dvd
    rw [hm, Nat.mul_div_cancel_left m hH]
    rcases Nat.eq_zero_or_pos m with h | h
    · rw [h] at hm; omega
    · exact h
  have hdiv63 : d.HV / d.H < 2 ^ 63 := by
    have := hd.hvv_lt
    have h' : d.HV ≤ d.HV * d.V := Nat.le_mul_of_pos_right _ hd.V_pos
    have : d.HV / d.H ≤ d.HV := Nat.div_le_self _ _
    omega
  unfold tail
  -- line 108: `i_h // (HV // H)`, for `q` and for `k`
  refine safe_idiv1 (x := i_h) (y := (IntV.ofNat (d.HV / d.H) : I64)) (by rw [h4]; omega)
    (by rw [IntV.I64.toInt_ofNat hdiv63]; exact_mod_cast hdiv) fun X hX => ?_
  refine safe_idiv1 (x := i_h) (y := (IntV.ofNat (d.HV / d.H) : I64)) (by rw [h4]; omega)
    (by rw [IntV.I64.toInt_ofNat hdiv63]; exact_mod_cast hdiv) fun Y hY => ?_
  have hkv : (hk d (hvOf d bh) : Int) = i_h.toInt / (IntV.ofNat (d.HV / d.H) : I64).toInt := by
    rw [h4, IntV.I64.toInt_ofNat hdiv63]; unfold hk; rw [Int.natCast_div]
  have hHV63 : d.HV < 2 ^ 63 := by
    have := hd.hvv_lt
    have h' : d.HV ≤ d.HV * d.V := Nat.le_mul_of_pos_right _ hd.V_pos
    omega
  have hH63 : d.H < 2 ^ 63 := lt_of_le_of_lt (Nat.le_of_dvd hHV hd.hv_dvd) hHV63
  have hK63 : d.K < 2 ^ 63 := by have := hd.K_le; omega
  have hV63 : d.V < 2 ^ 63 := by have := hd.V_le; omega
  have hqp := qptr_eq (d := d) (r := r) (bh := bh) (d.q) hqo h2 (by rw [hX]; exact hkv.symm) hd.K_pos hH63 hK63
  have hkp := qptr_eq (d := d) (r := r) (bh := bh) (d.k) hqo h2 (by rw [hY]; exact hkv.symm) hd.K_pos hH63 hK63
  have hdop := optr_eq (d := d) (r := r) (bh := bh) (d.dO) hoo h2 h4 hd.V_pos hHV63 hV63
  have hdvp := optr_eq (d := d) (r := r) (bh := bh) (d.dv) hoo h2 h4 hd.V_pos hHV63 hV63
  have hgp := gptr_eq (d := d) (r := r) (bh := bh) (d.g) hoo h2 h4 hd.V_pos hHV63
  simp only [hqp, hkp, hdop, hdvp, hgp]
  have hBT63 : d.BT < 2 ^ 63 := by rcases hd.bt_ok with h | h | h <;> omega
  have hBT31 : d.BT ≤ 2 ^ 31 := by rcases hd.bt_ok with h | h | h <;> omega
  have hstart := chunk_start_lt hd hT hr bh
  have hlen_le : lenOf d r bh ≤ d.B * d.T := by
    have := last_tok_lt hd hT hr hbh; omega
  have hBT22 : d.B * d.T ≤ 2 ^ 35 := by
    have h1 := hd.T_ctx; have h2 := hd.B_le
    calc d.B * d.T ≤ 2 ^ 13 * 2 ^ 22 := Nat.mul_le_mul (by omega) h1
      _ = 2 ^ 35 := by norm_num
  refine safe_convert (d := .i32) (d' := .i64) (fun idx => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩) ?_
  intro x hx
  refine safe_tailRest hP hd hpre hr hbh (q := d.q.ptr (qOff d r bh : Int)) (k := d.k.ptr (qOff d r bh : Int))
    (do_ := d.dO.ptr (oOff d r bh : Int)) (dv := d.dv.ptr (oOff d r bh : Int))
    (gb := d.g.ptr (gOff d r bh : Int)) ⟨rfl, rfl, rfl, rfl, rfl, ?_, h3⟩
  intro l
  have hxl := Tile2.arange64_toInt hx hBT31 l
  have hm1 : (i_t * (IntV.ofNat d.BT : I64)).toInt = ((chkOf d r bh * d.BT : Nat) : Int) :=
    IntV.I64.mul_nat h1 (IntV.I64.toInt_ofNat hBT63) (by omega)
  show ((i_t * (IntV.ofNat d.BT : I64)) + x.get1 l).toInt = _
  exact IntV.I64.add_nat hm1 hxl (by have := l.isLt; omega)

end Claims.ChunkBwdDvLocal
