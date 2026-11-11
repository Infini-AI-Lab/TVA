import TVA.Proof
import Claim
import Proof.Grid

/-!
# The `int32` side of the walk

The `int32` side of the walk: literals, bounds, lanes inside the matrices, the lifted-magnitude bounds, and the addresses of the lanes.
-/


/-!
# The scalar address arithmetic of the kernel

The seven `int32` scalars the kernel computes before its tile — `num_pid_m`, `num_pid_n`,
`group_id`, the two remainders, `pid_m`'s offset and `pid_n` — do not wrap, and are the tile map
of `Pid`.
-/

namespace Claims.Matmul

open TVA

/-! ## The literals -/

/-- `M` as an `int32` has value `M`. -/
theorem i32_M (d : Inst) (hd : d.ok) : ((IntV.ofNat d.M : I32)).toInt = (d.M : Int) :=
  IntV.I32.toInt_ofNat (by have := hd.M_le; omega)

/-- `N` as an `int32` has value `N`. -/
theorem i32_N (d : Inst) (hd : d.ok) : ((IntV.ofNat d.N : I32)).toInt = (d.N : Int) :=
  IntV.I32.toInt_ofNat (by have := hd.N_le; omega)

/-- `K` as an `int32` has value `K`. -/
theorem i32_K (d : Inst) (hd : d.ok) : ((IntV.ofNat d.K : I32)).toInt = (d.K : Int) :=
  IntV.I32.toInt_ofNat (by have := hd.K_le; omega)

/-- The `int32` constant `8`. -/
theorem i32_eight : ((IntV.ofNat 8 : I32)).toInt = 8 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `128`. -/
theorem i32_128 : ((IntV.ofNat 128 : I32)).toInt = 128 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `127`. -/
theorem i32_127 : ((IntV.ofNat (128 - 1) : I32)).toInt = 127 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `255`. -/
theorem i32_255 : ((IntV.ofNat (256 - 1) : I32)).toInt = 255 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `256`. -/
theorem i32_256 : ((IntV.ofNat 256 : I32)).toInt = 256 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `64`. -/
theorem i32_64 : ((IntV.ofNat 64 : I32)).toInt = 64 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `63`. -/
theorem i32_63 : ((IntV.ofNat (64 - 1) : I32)).toInt = 63 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-! ## Signed-range side conditions -/

/-- An integer in `[-2^31, 2^31)` fits `int32`. -/
theorem s32 {x : Int} (h1 : -(2 ^ 31 : Int) ≤ x) (h2 : x < 2 ^ 31) : IntV.SFits 32 x := by
  have e : ((2 : Int) ^ (32 - 1)) = 2 ^ 31 := by norm_num
  unfold IntV.SFits
  rw [e]
  exact ⟨h1, h2⟩

/-! ## The derived bounds -/

/-- At most `2^15` row tiles. -/
theorem npm_le (d : Inst) (hd : d.ok) : npm d ≤ 2 ^ 15 := by
  have h1 : (d.M + 127) / 128 ≤ (2 ^ 22 + 127) / 128 :=
    Nat.div_le_div_right (by have := hd.M_ctx; omega)
  have h2 : (2 ^ 22 + 127) / 128 = 2 ^ 15 := by norm_num
  have h3 : npm d = (d.M + 127) / 128 := rfl
  omega

/-- At most `2^9` column tiles. -/
theorem npn_le (d : Inst) (hd : d.ok) : npn d ≤ 2 ^ 9 := by
  have h1 : (d.N + 255) / 256 ≤ (2 ^ 17 + 255) / 256 :=
    Nat.div_le_div_right (by have := hd.N_dim.2.2; omega)
  have h2 : (2 ^ 17 + 255) / 256 = 2 ^ 9 := by norm_num
  have h3 : npn d = (d.N + 255) / 256 := rfl
  omega

/-- A covered instance has a column tile. -/
theorem npn_pos' (d : Inst) (hd : d.ok) : 0 < npn d := npn_pos hd.N_dim.1

/-- `8 · npn` fits `int32`. -/
theorem eight_npn_lt (d : Inst) (hd : d.ok) : 8 * npn d < 2 ^ 31 := by
  have := npn_le d hd
  omega

/-- `8 · npn` is positive. -/
theorem eight_npn_pos (d : Inst) (hd : d.ok) : 0 < 8 * npn d := by
  have := npn_pos' d hd
  omega

/-- The grid has fewer than `2^31` programs. -/
theorem grid_lt' (d : Inst) (hd : d.ok) : grid d < 2 ^ 31 := by
  have h := hd.grid_lt
  simpa only [grid, npm, npn] using h

/-- The program id as an `int32` has value `i`. -/
theorem i32_pid (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((IntV.ofNat (i : Nat) : I32)).toInt = (i : Int) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  exact IntV.I32.toInt_ofNat (by have := grid_lt' d hd; omega)

/-! ## `num_pid_m` and `num_pid_n` -/

/-- The `int32` sum of two small naturals has the sum of their values. -/
theorem toInt_add_ofNat {a b : Nat} (hab : (a : Int) + (b : Int) < 2 ^ 31) :
    (((IntV.ofNat a : I32)) + (IntV.ofNat b : I32)).toInt = (a : Int) + (b : Int) := by
  have ha : a < 2 ^ 31 := by omega
  have hb : b < 2 ^ 31 := by omega
  have hsa : (IntV.ofNat a : I32).toInt = (a : Int) := IntV.I32.toInt_ofNat ha
  have hsb : (IntV.ofNat b : I32).toInt = (b : Int) := IntV.I32.toInt_ofNat hb
  rw [IntV.toInt_add rfl rfl
    (by rw [hsa, hsb]; exact s32 (by omega) (by omega)), hsa, hsb]

/-- `num_pid_m` has value `npm`. -/
theorem num_pid_m_val {d : Inst} (hd : d.ok) {r : Tensor [] I32}
    (h : r.item.toInt
      = ((IntV.ofNat d.M : I32) + (IntV.ofNat (128 - 1) : I32)).toInt
          / (IntV.ofNat 128 : I32).toInt) :
    r.item.toInt = (npm d : Int) := by
  have hsum : (((IntV.ofNat d.M : I32)) + (IntV.ofNat (128 - 1) : I32)).toInt
      = (d.M : Int) + 127 := by
    rw [toInt_add_ofNat (by have := hd.M_le; omega)]; push_cast; ring
  rw [h, hsum, i32_128]
  have hcast : ((d.M + 127 : Nat) : Int) = (d.M : Int) + 127 := by push_cast; ring
  rw [← hcast]
  show ((d.M + 127 : Nat) : Int) / 128 = ((npm d : Nat) : Int)
  unfold npm
  rw [Int.natCast_div]
  norm_num

/-- `num_pid_n` has value `npn`. -/
theorem num_pid_n_val {d : Inst} (hd : d.ok) {r : Tensor [] I32}
    (h : r.item.toInt
      = ((IntV.ofNat d.N : I32) + (IntV.ofNat (256 - 1) : I32)).toInt
          / (IntV.ofNat 256 : I32).toInt) :
    r.item.toInt = (npn d : Int) := by
  have hsum : (((IntV.ofNat d.N : I32)) + (IntV.ofNat (256 - 1) : I32)).toInt
      = (d.N : Int) + 255 := by
    rw [toInt_add_ofNat (by have := hd.N_le; omega)]; push_cast; ring
  rw [h, hsum, i32_256]
  have hcast : ((d.N + 255 : Nat) : Int) = (d.N : Int) + 255 := by push_cast; ring
  rw [← hcast]
  show ((d.N + 255 : Nat) : Int) / 256 = ((npn d : Nat) : Int)
  unfold npn
  rw [Int.natCast_div]
  norm_num

/-- `num_pid_in_group` has value `8 · npn`. -/
theorem eight_mul {d : Inst} (hd : d.ok) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) :
    ((IntV.ofNat 8 : I32) * rn.item).toInt = 8 * (npn d : Int) := by
  rw [IntV.toInt_mul rfl rfl
    (by rw [i32_eight, hn]; exact s32 (by omega) (by have := eight_npn_lt d hd; omega)),
    i32_eight, hn]

/-- `num_pid_in_group` is positive. -/
theorem eight_mul_pos {d : Inst} (hd : d.ok) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) :
    0 < ((IntV.ofNat 8 : I32) * rn.item).toInt := by
  rw [eight_mul hd hn]
  have := npn_pos' d hd
  omega

/-! ## `group_id`, the remainders, and the tile -/

/-- `group_id` has value `pid / (8 · npn)`. -/
theorem group_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {r : Tensor [] I32}
    (h : r.item.toInt
      = (IntV.ofNat (i : Nat) : I32).toInt / ((IntV.ofNat 8 : I32) * rn.item).toInt) :
    r.item.toInt = (grp d i : Int) := by
  rw [h, i32_pid d hd i, eight_mul hd hn]
  have hcast2 : ((8 * npn d : Nat) : Int) = 8 * (npn d : Int) := by push_cast; ring
  rw [← hcast2]
  show (((i : Nat) : Int)) / ((8 * npn d : Nat) : Int) = ((grp d i : Nat) : Int)
  unfold grp
  rw [Int.natCast_div]

/-- `pid % num_pid_in_group` has its value. -/
theorem rem_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {r : Tensor [] I32}
    (h : r.item.toInt
      = (IntV.ofNat (i : Nat) : I32).toInt % ((IntV.ofNat 8 : I32) * rn.item).toInt) :
    r.item.toInt = (((i : Nat) % (8 * npn d) : Nat) : Int) := by
  rw [h, i32_pid d hd i, eight_mul hd hn]
  have hcast2 : ((8 * npn d : Nat) : Int) = 8 * (npn d : Int) := by push_cast; ring
  rw [← hcast2, Int.natCast_mod]

/-- `pid % num_pid_in_group` is nonnegative. -/
theorem rem_pos {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {r : Tensor [] I32}
    (h : r.item.toInt
      = (IntV.ofNat (i : Nat) : I32).toInt % ((IntV.ofNat 8 : I32) * rn.item).toInt) :
    0 ≤ r.item.toInt := by
  rw [rem_val hd i hn h]
  omega

/-- `group_size_m` has value `gsize`. -/
theorem gsize_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rm : Tensor [] I32}
    {rg : Tensor [] I32}
    (hm : rm.item.toInt = (npm d : Int)) (hg : rg.item.toInt = (grp d i : Int)) :
    (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat 8 : I32)))
      (Tensor.scalar (IntV.ofNat 8 : I32))).item.toInt = (gsize d i : Int) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hlt := pid_fpm_lt_npm hi
  have hnpm := npm_le d hd
  have hfpm : ((fpm d i : Nat) : Int) = 8 * (grp d i : Int) := by
    rw [pid_fpm_eq]; push_cast; ring
  have hmul : (rg.item * (IntV.ofNat 8 : I32)).toInt = 8 * (grp d i : Int) := by
    rw [IntV.toInt_mul rfl rfl
      (by rw [hg, i32_eight]; exact s32 (by omega) (by omega)), hg, i32_eight]
    ring
  have hsub : (rm.item - rg.item * (IntV.ofNat 8 : I32)).toInt
      = ((npm d - fpm d i : Nat) : Int) := by
    rw [IntV.toInt_sub rfl rfl (by
      rw [hm, hmul]; exact s32 (by omega) (by omega)), hm, hmul]
    rw [← hfpm]
    have h2 : ((npm d : Nat) : Int) - ((fpm d i : Nat) : Int)
        = ((npm d - fpm d i : Nat) : Int) := by
      push_cast [Nat.cast_sub (le_of_lt hlt)]
      ring
    rw [h2]
  have hmin : (IntV.min (rm.item - rg.item * (IntV.ofNat 8 : I32))
      (IntV.ofNat 8 : I32)).toInt = (gsize d i : Int) := by
    rw [IntV.toInt_min, hsub, i32_eight]
    have hg0 : min ((npm d - fpm d i : Nat) : Int) (8 : Int) = (gsize d i : Int) := by
      rw [gsize, Nat.cast_min]
      norm_num
    exact hg0
  simpa [tl.iminimum, Tensor.item, Tensor.map2, Tensor.scalar] using hmin

/-- `group_size_m` is positive. -/
theorem gsize_pos' {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rm : Tensor [] I32}
    {rg : Tensor [] I32}
    (hm : rm.item.toInt = (npm d : Int)) (hg : rg.item.toInt = (grp d i : Int)) :
    0 < (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat 8 : I32)))
      (Tensor.scalar (IntV.ofNat 8 : I32))).item.toInt := by
  rw [gsize_val hd i hm hg]
  have := gsize_pos (lt_of_lt_of_eq i.isLt (launch_size d))
  omega

/-- The inner remainder of `pid_m` has its value. -/
theorem off_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {rm : Tensor [] I32}
    (hm : rm.item.toInt = (npm d : Int)) {rg : Tensor [] I32}
    (hg : rg.item.toInt = (grp d i : Int)) {r4 r5 : Tensor [] I32}
    (h4 : r4.item.toInt
      = (IntV.ofNat (i : Nat) : I32).toInt % ((IntV.ofNat 8 : I32) * rn.item).toInt)
    (h5 : r5.item.toInt = r4.item.toInt
      % (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat 8 : I32)))
          (Tensor.scalar (IntV.ofNat 8 : I32))).item.toInt) :
    r5.item.toInt = ((pidM d i - fpm d i : Nat) : Int) := by
  rw [h5, rem_val hd i hn h4, gsize_val hd i hm hg]
  have e : ((((i : Nat) % (8 * npn d)) % gsize d i : Nat) : Int)
      = ((i : Nat) % (8 * npn d) : Nat) % (gsize d i : Int) := by
    rw [Int.natCast_mod]
  rw [← e]
  congr 1
  unfold pidM
  have hle : fpm d i ≤ fpm d i + ((i : Nat) % (8 * npn d)) % gsize d i := Nat.le_add_right _ _
  omega

/-- `pid_n` has value `pidN`. -/
theorem pidn_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (npn d : Int)) {rm : Tensor [] I32}
    (hm : rm.item.toInt = (npm d : Int)) {rg : Tensor [] I32}
    (hg : rg.item.toInt = (grp d i : Int)) {r6 r7 : Tensor [] I32}
    (h6 : r6.item.toInt
      = (IntV.ofNat (i : Nat) : I32).toInt % ((IntV.ofNat 8 : I32) * rn.item).toInt)
    (h7 : r7.item.toInt = r6.item.toInt
      / (tl.iminimum (Tensor.scalar (rm.item - rg.item * (IntV.ofNat 8 : I32)))
          (Tensor.scalar (IntV.ofNat 8 : I32))).item.toInt) :
    r7.item.toInt = ((pidN d i : Nat) : Int) := by
  rw [h7, rem_val hd i hn h6, gsize_val hd i hm hg]
  unfold pidN
  rw [Int.natCast_div]

/-! ## The eight assumptions -/

/-- The `int32` constant `0`. -/
theorem ofInt_zero : ((IntV.ofInt 0 : I32)).toInt = 0 :=
  IntV.I32.toInt_ofInt (by norm_num)

/-- The `tl.assume` `0 ≤ pid_m` holds. -/
theorem le_pidm {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {rg : Tensor [] I32}
    {ro : Tensor [] I32}
    (hg : rg.item.toInt = (grp d i : Int))
    (ho : ro.item.toInt = ((pidM d i - fpm d i : Nat) : Int)) :
    ((IntV.ofInt 0).le (rg.item * IntV.ofNat 8 + ro.item)) = true := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hlt := pid_fpm_lt_npm hi
  have hnpm := npm_le d hd
  have hpm := pidM_lt hi
  have hfpm : ((fpm d i : Nat) : Int) = 8 * (grp d i : Int) := by
    rw [pid_fpm_eq]; push_cast; ring
  have hmul : (rg.item * (IntV.ofNat 8 : I32)).toInt = 8 * (grp d i : Int) := by
    rw [IntV.toInt_mul rfl rfl
      (by rw [hg, i32_eight]; exact s32 (by omega) (by omega)), hg, i32_eight]
    ring
  have hsum : (rg.item * IntV.ofNat 8 + ro.item).toInt = (pidM d i : Int) := by
    rw [IntV.toInt_add rfl rfl (by
      rw [hmul, ho]; exact s32 (by omega) (by omega)), hmul, ho]
    rw [← hfpm]
    have hsplit : ((fpm d i : Nat) : Int) + ((pidM d i - fpm d i : Nat) : Int)
        = ((fpm d i + (pidM d i - fpm d i) : Nat) : Int) := by push_cast; ring
    rw [hsplit]
    congr 1
    have hle : fpm d i ≤ pidM d i := by rw [pidM]; exact Nat.le_add_right _ _
    omega
  rw [IntV.le_eq, decide_eq_true_iff, ofInt_zero, hsum]
  omega

/-- The `tl.assume` `0 ≤ pid_n` holds. -/
theorem le_pidn {d : Inst} (_hd : d.ok) (i : Fin (launch d).size) {rn : Tensor [] I32}
    (hn : rn.item.toInt = (pidN d i : Nat)) :
    ((IntV.ofInt 0).le rn.item) = true := by
  rw [IntV.le_eq, decide_eq_true_iff, ofInt_zero, hn]
  omega

/-- The `tl.assume` `0 < stride` holds for a positive `int32` stride. -/
theorem lt_stride (n : Nat) (h : 0 < n) (hn : n < 2 ^ 31) :
    ((IntV.ofInt 0 : I32).lt (IntV.ofNat n : I32)) = true := by
  rw [IntV.lt_eq, decide_eq_true_iff, ofInt_zero, IntV.I32.toInt_ofNat hn]
  exact_mod_cast h

end Claims.Matmul

/- ===== Mag ===== -/

/-!
# Magnitudes of the lifted products

Every finite `float16` value is at most `65504` in magnitude, so its lifted magnitude is below
`2^16`, a product of two of them below `2^32`, and a step of `64` of them below `2^38`. These are
the numbers that keep the accumulator's magnitude, and so `tl.dot`'s room, far from `float32`'s
range.
-/

namespace Claims.Matmul

open TVA

/-- A `float16` operand of magnitude at most `65504` has a lifted magnitude below `2^16`. -/
theorem liftMag_f16_lt (q : ℚ) (h : |q| ≤ 65504) : Device.liftMag .f16 q < 2 ^ 16 := by
  have h1 := Device.liftMag_le DType.f16 q
  rw [Device.liftFloor_f16] at h1
  have h2 : (65504 : ℚ) + 2 ^ (-14 : ℤ) < 2 ^ 16 := by norm_num
  linarith

/-- A product of lifted `float16` magnitudes is below `2^32`. -/
theorem liftMul_lt (a b : ℚ) (ha : |a| ≤ 65504) (hb : |b| ≤ 65504) :
    Device.liftMag .f16 a * Device.liftMag .f16 b < 2 ^ 32 := by
  have h1 := liftMag_f16_lt a ha
  have h2 := liftMag_f16_lt b hb
  have h3 := Device.liftMag_nonneg DType.f16 a
  have h4 := Device.liftMag_nonneg DType.f16 b
  nlinarith

/-- A finite `float16` value is at most `65504` in magnitude. -/
theorem toRat_abs_le (x : F16) (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have h := TVA.FloatV.abs_toRat_le_maxVal (f := Format.binary16)
    Format.binary16_regular hx
  rwa [Format.binary16_maxVal] at h

/-- The lifted magnitude of one product of the loop. -/
theorem liftV_lt {va vb : Nat → F16} {K e : Nat}
    (ha : ∀ e, e < K → (va e).isFinite = true)
    (hb : ∀ e, e < K → (vb e).isFinite = true) :
    liftV va vb K e < 2 ^ 32 := by
  unfold liftV
  split_ifs with hk
  · exact liftMul_lt _ _ (toRat_abs_le _ (ha e hk)) (toRat_abs_le _ (hb e hk))
  · norm_num

/-- One step of the loop, lifted: at most `64 · 2^32`. -/
theorem stepMag_lt {va vb : Nat → F16} {K t : Nat}
    (ha : ∀ e, e < K → (va e).isFinite = true)
    (hb : ∀ e, e < K → (vb e).isFinite = true) :
    stepMag (liftV va vb K) t ≤ 2 ^ 38 := by
  unfold stepMag
  have hsum : (∑ k : Fin 64, liftV va vb K (64 * t + (k : Nat))) ≤ 64 * 2 ^ 32 := by
    calc ∑ k : Fin 64, liftV va vb K (64 * t + (k : Nat))
        ≤ ∑ _k : Fin 64, (2 : ℚ) ^ 32 :=
          Finset.sum_le_sum (fun k _ => le_of_lt (liftV_lt ha hb))
      _ = 64 * 2 ^ 32 := by simp
  have : (64 : ℚ) * 2 ^ 32 = 2 ^ 38 := by norm_num
  linarith

/-- The partial sum of the loop is at most `t · 2^38`. -/
theorem partSum_abs_le {va vb : Nat → F16} {K : Nat}
    (ha : ∀ e, e < K → (va e).isFinite = true)
    (hb : ∀ e, e < K → (vb e).isFinite = true) (t : Nat) :
    |partSum (prodV va vb K) t| ≤ (t : ℚ) * 2 ^ 38 := by
  unfold partSum
  have h1 : |∑ e ∈ Finset.range (64 * t), prodV va vb K e| ≤
      ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| := Finset.abs_sum_le_sum_abs _ _
  have h2 : ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| ≤
      ∑ _e ∈ Finset.range (64 * t), (2 : ℚ) ^ 32 := by
    apply Finset.sum_le_sum
    intro e _
    exact le_of_lt ((prodV_abs_le_liftV va vb K e).trans_lt (liftV_lt ha hb))
  have h3 : (∑ _e ∈ Finset.range (64 * t), (2 : ℚ) ^ 32) = (64 * t : Nat) * 2 ^ 32 := by simp
  have h4 : ((64 * t : Nat) : ℚ) * 2 ^ 32 = (t : ℚ) * 2 ^ 38 := by push_cast; ring
  calc |∑ e ∈ Finset.range (64 * t), prodV va vb K e|
      ≤ ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| := h1
    _ ≤ ∑ _e ∈ Finset.range (64 * t), (2 : ℚ) ^ 32 := h2
    _ = (64 * t : Nat) * 2 ^ 32 := h3
    _ = (t : ℚ) * 2 ^ 38 := h4

end Claims.Matmul

/- ===== Addr ===== -/

/-!
# The addresses of a program's lanes

A program's tile of `a`, `b` and `c`: which element of each matrix a lane touches, and the
`int32` fits that keep the model's pointer arithmetic from wrapping.
-/

namespace Claims.Matmul

open TVA

/-- The row of `a` (and of `c`) that lane `r` of program `i` reads: `offs_am`, the row taken
`% M` (line 282). -/
def aRow (d : Inst) (i r : Nat) : Nat := (pidM d i * 128 + r) % d.M

/-- The column of `b` (and of `c`) of lane `c` of program `i`: `offs_bn`, taken `% N`
(line 283). -/
def bCol (d : Inst) (i c : Nat) : Nat := (pidN d i * 256 + c) % d.N

/-- The element of `a`'s storage lane `(r, kk)` reads at step `t` of the reduction. -/
def aAdr (d : Inst) (i t r kk : Nat) : Nat :=
  aRow d i r * d.stride_am + (64 * t + kk) * d.stride_ak

/-- The element of `b`'s storage lane `(kk, c)` reads at step `t`. -/
def bAdr (d : Inst) (i t kk c : Nat) : Nat :=
  (64 * t + kk) * d.stride_bk + bCol d i c * d.stride_bn

/-- The element of `c`'s storage lane `(r, c)` writes. -/
def cAdr (d : Inst) (i r c : Nat) : Nat := cRow d i r * d.stride_cm + cCol d i c * d.stride_cn

/-! ## The shape is positive -/

/-- A program of the grid has `M > 0`. -/
theorem M_pos {d : Inst} (i : Fin (launch d).size) : 0 < d.M := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have h : 0 < npm d := pid_npm_pos_of_grid hi
  simp only [npm] at h
  have := Nat.div_add_mod (d.M + 127) 128
  omega

/-- A program of the grid has `N > 0`. -/
theorem N_pos {d : Inst} (i : Fin (launch d).size) : 0 < d.N := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have h : 0 < npn d := pid_npn_pos_of_grid hi
  simp only [npn] at h
  have := Nat.div_add_mod (d.N + 255) 256
  omega

/-! ## The lanes are inside the matrices -/

/-- Lane rows of `a` are inside the matrix. -/
theorem aRow_lt {d : Inst} (i : Fin (launch d).size) (r : Nat) : aRow d i r < d.M :=
  Nat.mod_lt _ (M_pos i)

/-- Lane columns of `b` are inside the matrix. -/
theorem bCol_lt {d : Inst} (i : Fin (launch d).size) (c : Nat) : bCol d i c < d.N :=
  Nat.mod_lt _ (N_pos i)

/-- The `K`-step of the loop has `64 ∣ K`, so `t < steps K` and `kk < 64` put the lane's `k`
below `K`. -/
theorem step_lt {d : Inst} (hd : d.ok) {t kk : Nat}
    (ht : t < steps d.K) (hk : kk < 64) : 64 * t + kk < d.K := by
  have hK : d.K % 64 = 0 := by
    have := hd.K_dim.2.1
    omega
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hK
  rw [steps_eq hK] at ht
  omega

/-- The address of an `a` lane is inside `a`'s storage. -/
theorem aAdr_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {t kk r : Nat}
    (ht : t < steps d.K) (hk : kk < 64) : aAdr d i t r kk < d.a.len :=
  hd.a_len _ _ (aRow_lt i r) (step_lt hd ht hk)

/-- The address of a `b` lane is inside `b`'s storage. -/
theorem bAdr_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {t kk c : Nat}
    (ht : t < steps d.K) (hk : kk < 64) : bAdr d i t kk c < d.b.len :=
  hd.b_len _ _ (step_lt hd ht hk) (bCol_lt i c)

/-- The address of a `c` lane is inside `c`'s storage. -/
theorem cAdr_lt {d : Inst} (hd : d.ok) {i r c : Nat}
    (hM : cRow d i r < d.M) (hN : cCol d i c < d.N) : cAdr d i r c < d.c.len :=
  hd.c_len _ _ hM hN

/-- An `a` lane's address is an element of the matrix `a`. -/
theorem aAdr_mem {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {t kk r : Nat}
    (ht : t < steps d.K) (hk : kk < 64) : aAdr d i t r kk ∈ aSet d :=
  ⟨aRow d i r, 64 * t + kk, aRow_lt i r, step_lt hd ht hk, rfl⟩

/-- A `b` lane's address is an element of the matrix `b`. -/
theorem bAdr_mem {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {t kk c : Nat}
    (ht : t < steps d.K) (hk : kk < 64) : bAdr d i t kk c ∈ bSet d :=
  ⟨64 * t + kk, bCol d i c, step_lt hd ht hk, bCol_lt i c, rfl⟩

/-- A `c` lane's address is an element of the matrix `c`. -/
theorem cAdr_mem {d : Inst} (i : Nat) {r c : Nat} (hr : r < 128) (hc : c < 256)
    (hM : cRow d i r < d.M) (hN : cCol d i c < d.N) : cAdr d i r c ∈ tileSet d i :=
  ⟨r, c, hr, hc, hM, hN, rfl⟩

/-! ## The `int32` fits of the lane offsets -/

/-- The `int32` offset of an `a` lane does not wrap. -/
theorem a_fit_lane {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (r kk : Nat)
    (hk : kk < 64) :
    aRow d i r * d.stride_am + kk * d.stride_ak < 2 ^ 31 := by
  have h1 : aRow d i r ≤ d.M - 1 := by have := aRow_lt i r; omega
  have h2 : kk * d.stride_ak ≤ 64 * d.stride_ak := Nat.mul_le_mul_right _ (by omega)
  have h3 : aRow d i r * d.stride_am ≤ (d.M - 1) * d.stride_am :=
    Nat.mul_le_mul_right _ h1
  have := hd.a_fit
  omega

/-- The row part of an `a` offset does not wrap. -/
theorem a_fit_row {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (r : Nat) :
    aRow d i r * d.stride_am < 2 ^ 31 := by
  have := a_fit_lane hd i r 0 (by norm_num)
  omega

/-- The `k` part of an `a` offset does not wrap. -/
theorem a_fit_k {d : Inst} (hd : d.ok) (kk : Nat) (hk : kk < 64) :
    kk * d.stride_ak < 2 ^ 31 := by
  have := hd.a_fit
  have h2 : kk * d.stride_ak ≤ 64 * d.stride_ak := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The `int32` offset of a `b` lane does not wrap. -/
theorem b_fit_lane {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (c kk : Nat)
    (hk : kk < 64) :
    kk * d.stride_bk + bCol d i c * d.stride_bn < 2 ^ 31 := by
  have h1 : bCol d i c ≤ d.N - 1 := by have := bCol_lt i c; omega
  have h2 : bCol d i c * d.stride_bn ≤ (d.N - 1) * d.stride_bn :=
    Nat.mul_le_mul_right _ h1
  have h3 : kk * d.stride_bk ≤ 64 * d.stride_bk := Nat.mul_le_mul_right _ (by omega)
  have := hd.b_fit
  omega

/-- The row part of a `c` offset does not wrap. -/
theorem c_fit_row {d : Inst} (hd : d.ok) {i r : Nat} (hM : cRow d i r < d.M) :
    cRow d i r * d.stride_cm < 2 ^ 31 := by
  have h1 : cRow d i r ≤ d.M - 1 := by omega
  have h2 : cRow d i r * d.stride_cm ≤ (d.M - 1) * d.stride_cm :=
    Nat.mul_le_mul_right _ h1
  have := hd.cm_fit
  omega

/-- The column part of a `c` offset does not wrap. -/
theorem c_fit_col {d : Inst} (hd : d.ok) {i c : Nat} (hN : cCol d i c < d.N) :
    cCol d i c * d.stride_cn < 2 ^ 31 := by
  have h1 : cCol d i c ≤ d.N - 1 := by omega
  have h2 : cCol d i c * d.stride_cn ≤ (d.N - 1) * d.stride_cn :=
    Nat.mul_le_mul_right _ h1
  have := hd.cn_fit
  omega

/-- The step `64 · stride_ak` fits `int32`. -/
theorem stride_step_a {d : Inst} (hd : d.ok) : 64 * d.stride_ak < 2 ^ 31 := by
  have := hd.a_fit
  omega

/-- The step `64 · stride_bk` fits `int32`. -/
theorem stride_step_b {d : Inst} (hd : d.ok) : 64 * d.stride_bk < 2 ^ 31 := by
  have := hd.b_fit
  omega

end Claims.Matmul
