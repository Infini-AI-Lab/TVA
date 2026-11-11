import Claim
import TVA.Proof
import Proof.LaneGet

/-!
# The lane values of the masks, and the loop's trip count

`maskA`, `maskB`, `maskSeqM`, `maskSeqN` and `maskStore` (in `Proof/Peel.lean`) are comparisons
of the lane offsets with `chunk_size_limit`, with `K − k·BLOCK_SIZE_K`, or with `chunk_size`. This
file records each lane value as one of the `enA`/`enB`/`enM`/`enN`/`enS` predicates, the values of
`chunk_size_limit` and of the `k` bound, and the number of turns of the loop.
-/

namespace Claims.SsdBmm

open TVA

set_option maxHeartbeats 800000

/-- A natural below `2^31` fits `int64`. -/
theorem sfits64_of_lt31 {n : Nat} (h : n < 2 ^ 31) : IntV.SFits 64 (n : ℤ) :=
  IntV.I64.sfits_of_sfits32 (IntV.I32.sfits_natCast h)

/-- `chunk_size_limit` (lines 145, 168) is the number of tokens of the chunk, `min(C, T - pid_c·C)`. -/
theorem chunkLimit_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (chunkLimit d i).toInt = (chunkNat d i : ℤ) := by
  set c : Nat := (i : Nat) / gx d / d.G with hcdef
  have hcC : c * d.C < d.T := by rw [hcdef]; exact cOf_mul_C_lt hd i
  have hC31 : d.C < 2 ^ 31 := C_lt_31 hd
  have hc31 : c < 2 ^ 31 := by
    have h1 : c ≤ c * d.C := Nat.le_mul_of_pos_right c (by have := hd.C_ge; omega)
    have := hd.T_lt
    omega
  have hcC31 : c * d.C < 2 ^ 31 := lt_trans hcC hd.T_lt
  have hcc : (c : ℤ) * (d.C : ℤ) = ((c * d.C : ℕ) : ℤ) := by push_cast; ring
  have hprod : ((IntV.ofInt (c : ℤ) : I64) * (IntV.ofInt (d.C : ℤ) : I64))
      = IntV.ofInt ((c : ℤ) * (d.C : ℤ)) :=
    IntV.ofInt_mul_ofInt _ _
  have hTz : (d.T : ℤ) - (c : ℤ) * (d.C : ℤ) = ((d.T - c * d.C : ℕ) : ℤ) := by
    rw [Nat.cast_sub (le_of_lt hcC)]
    push_cast
    ring
  have hsub : ((IntV.ofInt (d.T : ℤ) : I64) - IntV.ofInt ((c : ℤ) * (d.C : ℤ)))
      = IntV.ofInt ((d.T : ℤ) - (c : ℤ) * (d.C : ℤ)) :=
    IntV.ofInt_sub_ofInt _ _
  have hbody : chunkLimit d i =
      IntV.min (IntV.ofInt (d.C : ℤ) : I64)
        ((IntV.ofInt (d.T : ℤ) : I64) - IntV.ofInt (c : ℤ) * (IntV.ofInt (d.C : ℤ) : I64)) := rfl
  have hCv : (IntV.ofInt (d.C : ℤ) : I64).toInt = (d.C : ℤ) :=
    IntV.toInt_ofInt (by norm_num) (sfits64_of_lt31 hC31)
  have hsubv : (((IntV.ofInt (d.T : ℤ) : I64) - IntV.ofInt (c : ℤ) * (IntV.ofInt (d.C : ℤ) : I64))).toInt
      = ((d.T - c * d.C : ℕ) : ℤ) := by
    rw [hprod, hsub]
    rw [IntV.toInt_ofInt (by norm_num) (by rw [hTz]; exact sfits64_of_lt31 (lt_of_le_of_lt (Nat.sub_le _ _) hd.T_lt))]
    exact hTz
  have hchunk : chunkNat d i = min d.C (d.T - c * d.C) := by
    unfold chunkNat cOf
    rw [← hcdef]
  rw [hbody, IntV.toInt_min, hsubv, hCv, hchunk, Nat.cast_min]

/-- The bound `K - k·BLOCK_SIZE_K` of the `k` mask (line 152) has value `K - t·32`. -/
theorem kMask_toInt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < steps d.K) :
    (kMask d t).toInt = (d.K : ℤ) - (t * BK : ℕ) := by
  have hK31 : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have hBK31 : BK < 2 ^ 31 := by simp [BK]
  have ht32 : t < 32 := lt_of_lt_of_le ht (steps_le hd.K_le)
  have htBK : t * BK < d.K := t_mul_BK_lt ht
  have hKv : (IntV.ofNat d.K : I32).toInt = (d.K : ℤ) := IntV.I32.toInt_ofNat hK31
  have htv : (IntV.ofInt (t : ℤ) : I32).toInt = (t : ℤ) := by
    apply IntV.I32.toInt_ofInt
    constructor
    · have : (0 : ℤ) ≤ (t : ℤ) := Int.natCast_nonneg t
      linarith
    · have h : (t : ℤ) < (2 : ℤ) ^ 31 := by
        exact_mod_cast lt_trans ht32 (by norm_num : (32 : Nat) < 2 ^ 31)
      simpa using h
  have hBKv : (IntV.ofNat BK : I32).toInt = (BK : ℤ) := IntV.I32.toInt_ofNat hBK31
  have hprodlt : t * BK < 2 ^ 31 := by
    have h1 : t ≤ 31 := by omega
    have h2 : t * BK ≤ 31 * 32 := by
      calc t * BK ≤ 31 * BK := Nat.mul_le_mul_right _ h1
        _ = 31 * 32 := by rw [BK]
    omega
  have hprodv : (((IntV.ofInt (t : ℤ) : I32) * (IntV.ofNat BK : I32))).toInt
      = (t : ℤ) * (BK : ℤ) := by
    refine IntV.toInt_mul htv hBKv ?_
    have hmul : (t : ℤ) * (BK : ℤ) = ((t * BK : ℕ) : ℤ) := by push_cast; ring
    rw [hmul]; exact IntV.I32.sfits_natCast hprodlt
  have hkn : (d.K : ℤ) - (t : ℤ) * (BK : ℤ) = ((d.K - t * BK : ℕ) : ℤ) := by
    rw [Nat.cast_sub (le_of_lt htBK)]
    push_cast
    ring
  have hsubv : (((IntV.ofNat d.K : I32) - (IntV.ofInt (t : ℤ) : I32) * (IntV.ofNat BK : I32))).toInt
      = (d.K : ℤ) - (t * BK : ℕ) := by
    rw [IntV.toInt_sub hKv hprodv (by
      rw [hkn]
      exact IntV.I32.sfits_natCast (lt_of_le_of_lt (Nat.sub_le _ _) hK31))]
    push_cast
    ring
  have hbody : kMask d t =
      (IntV.ofNat d.K : I32) - (IntV.ofInt (t : ℤ) : I32) * (IntV.ofNat BK : I32) := rfl
  rw [hbody, hsubv]




/-- A natural below `2^31` is an `int32` constant of its own value. -/
theorem lm_I32_ofInt_toInt {n : Nat} (h : n < 2 ^ 31) :
    (IntV.ofInt (n : ℤ) : I32).toInt = (n : ℤ) :=
  IntV.toInt_ofInt (by norm_num) (IntV.I32.sfits_natCast h)

/-- Lane `mll` of `offs_m` is the `int32` constant `pid_m · 64 + mll`. -/
theorem lm_offsM_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) :
    (offsM d i).get (mll, ()) = IntV.ofInt ((mOf d i * BM + mll.val : ℕ) : ℤ) :=
  IntV.ext_toInt ((offsM_toInt hd i mll).trans (lm_I32_ofInt_toInt (mOf_offs_lt hd i mll)).symm)

/-- Lane `nn` of `offs_n` is the `int32` constant `pid_n · 128 + nn`. -/
theorem lm_offsN_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (nn : Fin BN) :
    (offsN d i).get (nn, ()) = IntV.ofInt ((nOf d i * BN + nn.val : ℕ) : ℤ) :=
  IntV.ext_toInt ((offsN_toInt hd i nn).trans (lm_I32_ofInt_toInt (nOf_offs_lt hd i nn)).symm)

/-- Lane `mll` of the `seq_idx` row mask (line 172) is `enM`: the row is below `chunk_size_limit`. -/
theorem maskSeqM_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) :
    (maskSeqM d i).get (mll, ()) = enM d i mll.val := by
  have hred : (maskSeqM d i).get (mll, ()) =
      IntV.lt ((conv64 (offsM d i)).get (mll, ())) (chunkLimit d i) := rfl
  have hconv : (conv64 (offsM d i)).get (mll, ())
      = IntV.ofInt ((offsM d i).get (mll, ())).toInt := rfl
  have hC := C_lt_31 hd
  have hpm : mOf d i * BM < 2 ^ 31 := lt_of_lt_of_le (mOf_mul_BM_lt hd i) (le_of_lt hC)
  have hmll : mll.val < 2 ^ 31 := by have := mll.isLt; simp [BM] at this; omega
  have hsum : mOf d i * BM + mll.val < 2 ^ 31 := by
    have h1 := mOf_mul_BM_lt hd i
    have h2 := mll.isLt
    have h3 := C_le_46340 hd
    have h4 : BM = 64 := rfl
    omega
  rw [hred, hconv, lm_offsM_get hd i mll, lm_I32_ofInt_toInt hsum,
    IntV.lt_eq, IntV.toInt_ofInt (by norm_num) (sfits64_of_lt31 hsum),
    chunkLimit_toInt hd i]
  simp only [enM, Int.ofNat_lt]

/-- Lane `nn` of the `seq_idx` column mask (line 177) is `enN`. -/
theorem maskSeqN_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (nn : Fin BN) :
    (maskSeqN d i).get (nn, ()) = enN d i nn.val := by
  have hred : (maskSeqN d i).get (nn, ()) =
      IntV.lt ((conv64 (offsN d i)).get (nn, ())) (chunkLimit d i) := rfl
  have hconv : (conv64 (offsN d i)).get (nn, ())
      = IntV.ofInt ((offsN d i).get (nn, ())).toInt := rfl
  have hC := C_lt_31 hd
  have hpn : nOf d i * BN < 2 ^ 31 := lt_of_lt_of_le (nOf_mul_BN_lt hd i) (le_of_lt hC)
  have hnn : nn.val < 2 ^ 31 := by have := nn.isLt; simp [BN] at this; omega
  have hsum : nOf d i * BN + nn.val < 2 ^ 31 := by
    have h1 := nOf_mul_BN_lt hd i
    have h2 := nn.isLt
    have h3 := C_le_46340 hd
    have h4 : BN = 128 := rfl
    omega
  rw [hred, hconv, lm_offsN_get hd i nn, lm_I32_ofInt_toInt hsum,
    IntV.lt_eq, IntV.toInt_ofInt (by norm_num) (sfits64_of_lt31 hsum),
    chunkLimit_toInt hd i]
  simp only [enN, Int.ofNat_lt]

/-- Lane `(mll, nn)` of the store mask (lines 186–190) is `enS`: row and column are below `chunk_size`. -/
theorem maskStore_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) (nn : Fin BN) :
    (maskStore d i).get (mll, nn, ()) = enS d i mll.val nn.val := by
  have hred : (maskStore d i).get (mll, nn, ()) =
      (IntV.lt ((offsM d i).get (mll, ())) (IntV.ofNat d.C : I32) &&
       IntV.lt ((offsN d i).get (nn, ())) (IntV.ofNat d.C : I32)) := rfl
  have hC := C_lt_31 hd
  have hCv : (IntV.ofNat d.C : I32).toInt = (d.C : ℤ) := IntV.I32.toInt_ofNat hC
  have hpm : mOf d i * BM < 2 ^ 31 := lt_of_lt_of_le (mOf_mul_BM_lt hd i) (le_of_lt hC)
  have hmll : mll.val < 2 ^ 31 := by have := mll.isLt; simp [BM] at this; omega
  have hsumM : mOf d i * BM + mll.val < 2 ^ 31 := by
    have h1 := mOf_mul_BM_lt hd i
    have h2 := mll.isLt
    have h3 := C_le_46340 hd
    have h4 : BM = 64 := rfl
    omega
  have hpn : nOf d i * BN < 2 ^ 31 := lt_of_lt_of_le (nOf_mul_BN_lt hd i) (le_of_lt hC)
  have hnn : nn.val < 2 ^ 31 := by have := nn.isLt; simp [BN] at this; omega
  have hsumN : nOf d i * BN + nn.val < 2 ^ 31 := by
    have h1 := nOf_mul_BN_lt hd i
    have h2 := nn.isLt
    have h3 := C_le_46340 hd
    have h4 : BN = 128 := rfl
    omega
  have h1 : IntV.lt ((offsM d i).get (mll, ())) (IntV.ofNat d.C : I32)
      = decide (mOf d i * BM + mll.val < d.C) := by
    rw [IntV.lt_eq, lm_offsM_get hd i mll, lm_I32_ofInt_toInt hsumM, hCv]
    simp only [Int.ofNat_lt]
  have h2 : IntV.lt ((offsN d i).get (nn, ())) (IntV.ofNat d.C : I32)
      = decide (nOf d i * BN + nn.val < d.C) := by
    rw [IntV.lt_eq, lm_offsN_get hd i nn, lm_I32_ofInt_toInt hsumN, hCv]
    simp only [Int.ofNat_lt]
  rw [hred, h1, h2]
  simp only [enS]

/-- Lane `(mll, kk)` of the mask of the `a` load (lines 149–154) is `enA`. -/
theorem maskA_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat)
    (ht : t < steps d.K) (mll : Fin BM) (kk : Fin BK) :
    (maskA d i t).get (mll, kk, ()) = enA d i t mll.val kk.val := by
  have hred : (maskA d i t).get (mll, kk, ()) =
      (IntV.lt ((conv64 (offsM d i)).get (mll, ())) (chunkLimit d i) &&
       IntV.lt ((tl.arange 0 BK).get (kk, ())) (kMask d t)) := rfl
  have hconv : (conv64 (offsM d i)).get (mll, ())
      = IntV.ofInt ((offsM d i).get (mll, ())).toInt := rfl
  have hC := C_lt_31 hd
  have hpm : mOf d i * BM < 2 ^ 31 := lt_of_lt_of_le (mOf_mul_BM_lt hd i) (le_of_lt hC)
  have hmll : mll.val < 2 ^ 31 := by have := mll.isLt; simp [BM] at this; omega
  have hsumM : mOf d i * BM + mll.val < 2 ^ 31 := by
    have h1 := mOf_mul_BM_lt hd i
    have h2 := mll.isLt
    have h3 := C_le_46340 hd
    have h4 : BM = 64 := rfl
    omega
  have hkk : kk.val < 2 ^ 31 := by have := kk.isLt; simp [BK] at this; omega
  have h1 : IntV.lt ((conv64 (offsM d i)).get (mll, ())) (chunkLimit d i)
      = decide (mOf d i * BM + mll.val < chunkNat d i) := by
    rw [hconv, lm_offsM_get hd i mll, lm_I32_ofInt_toInt hsumM, IntV.lt_eq,
      IntV.toInt_ofInt (by norm_num) (sfits64_of_lt31 hsumM), chunkLimit_toInt hd i]
    simp only [Int.ofNat_lt]
  have h2 : IntV.lt ((tl.arange 0 BK).get (kk, ())) (kMask d t)
      = decide (t * BK + kk.val < d.K) := by
    have har : (tl.arange 0 BK).get (kk, ()) = (IntV.ofInt (0 + (kk.val : ℤ)) : I32) := rfl
    rw [har]
    have hz : (0 : ℤ) + (kk.val : ℤ) = (kk.val : ℤ) := Int.zero_add _
    rw [hz, IntV.lt_eq, lm_I32_ofInt_toInt hkk, kMask_toInt hd ht]
    rw [decide_eq_decide]
    have hcK : ((t * BK : ℕ) : ℤ) = (t : ℤ) * (BK : ℤ) := by push_cast; ring
    have hck : ((t * BK + kk.val : ℕ) : ℤ) = (t : ℤ) * (BK : ℤ) + (kk.val : ℤ) := by push_cast; ring
    constructor
    · intro h
      rw [hcK] at h
      have h' : ((t * BK + kk.val : ℕ) : ℤ) < (d.K : ℤ) := by rw [hck]; linarith
      exact_mod_cast h'
    · intro h
      have h' : ((t * BK + kk.val : ℕ) : ℤ) < (d.K : ℤ) := by exact_mod_cast h
      rw [hck] at h'
      rw [hcK]
      linarith
  rw [hred, h1, h2]
  simp only [enA]

/-- Lane `(kk, nn)` of the mask of the `b` load (lines 155–160) is `enB`. -/
theorem maskB_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat)
    (ht : t < steps d.K) (kk : Fin BK) (nn : Fin BN) :
    (maskB d i t).get (kk, nn, ()) = enB d i t kk.val nn.val := by
  have hred : (maskB d i t).get (kk, nn, ()) =
      (IntV.lt ((tl.arange 0 BK).get (kk, ())) (kMask d t) &&
       IntV.lt ((conv64 (offsN d i)).get (nn, ())) (chunkLimit d i)) := rfl
  have hconv : (conv64 (offsN d i)).get (nn, ())
      = IntV.ofInt ((offsN d i).get (nn, ())).toInt := rfl
  have hC := C_lt_31 hd
  have hpn : nOf d i * BN < 2 ^ 31 := lt_of_lt_of_le (nOf_mul_BN_lt hd i) (le_of_lt hC)
  have hnn : nn.val < 2 ^ 31 := by have := nn.isLt; simp [BN] at this; omega
  have hsumN : nOf d i * BN + nn.val < 2 ^ 31 := by
    have h1 := nOf_mul_BN_lt hd i
    have h2 := nn.isLt
    have h3 := C_le_46340 hd
    have h4 : BN = 128 := rfl
    omega
  have hkk : kk.val < 2 ^ 31 := by have := kk.isLt; simp [BK] at this; omega
  have h1 : IntV.lt ((tl.arange 0 BK).get (kk, ())) (kMask d t)
      = decide (t * BK + kk.val < d.K) := by
    have har : (tl.arange 0 BK).get (kk, ()) = (IntV.ofInt (0 + (kk.val : ℤ)) : I32) := rfl
    rw [har]
    have hz : (0 : ℤ) + (kk.val : ℤ) = (kk.val : ℤ) := Int.zero_add _
    rw [hz, IntV.lt_eq, lm_I32_ofInt_toInt hkk, kMask_toInt hd ht]
    rw [decide_eq_decide]
    have hcK : ((t * BK : ℕ) : ℤ) = (t : ℤ) * (BK : ℤ) := by push_cast; ring
    have hck : ((t * BK + kk.val : ℕ) : ℤ) = (t : ℤ) * (BK : ℤ) + (kk.val : ℤ) := by push_cast; ring
    constructor
    · intro h
      rw [hcK] at h
      have h' : ((t * BK + kk.val : ℕ) : ℤ) < (d.K : ℤ) := by rw [hck]; linarith
      exact_mod_cast h'
    · intro h
      have h' : ((t * BK + kk.val : ℕ) : ℤ) < (d.K : ℤ) := by exact_mod_cast h
      rw [hck] at h'
      rw [hcK]
      linarith
  have h2 : IntV.lt ((conv64 (offsN d i)).get (nn, ())) (chunkLimit d i)
      = decide (nOf d i * BN + nn.val < chunkNat d i) := by
    rw [hconv, lm_offsN_get hd i nn, lm_I32_ofInt_toInt hsumN, IntV.lt_eq,
      IntV.toInt_ofInt (by norm_num) (sfits64_of_lt31 hsumN), chunkLimit_toInt hd i]
    simp only [Int.ofNat_lt]
  rw [hred, h1, h2]
  simp only [enB]

/-- The loop of line 148 runs `steps K` times. -/
theorem hsteps {d : Inst} (hd : d.ok) :
    tl.tripCount (IntV.ofInt 0 : I32).toInt
        (IntV.ofInt ((d.K + (BK - 1)) / BK : ℕ) : I32).toInt (IntV.ofInt 1 : I32).toInt =
      steps d.K := by
  have h0 : (IntV.ofInt 0 : I32).toInt = (0:ℤ) := by
    have h : (-(2:ℤ)^(32-1) ≤ (0:ℤ) ∧ (0:ℤ) < 2^(32-1)) := by norm_num
    simpa using IntV.I32.toInt_ofInt (x := (0:ℤ)) h
  have h1 : (IntV.ofInt 1 : I32).toInt = (1:ℤ) := by
    have h : (-(2:ℤ)^(32-1) ≤ (1:ℤ) ∧ (1:ℤ) < 2^(32-1)) := by norm_num
    simpa using IntV.I32.toInt_ofInt (x := (1:ℤ)) h
  have hs : steps d.K ≤ 32 := steps_le hd.K_le
  have hn : ((d.K + (BK - 1)) / BK : ℕ) < 2 ^ 31 := by
    have h : ((d.K + (BK - 1)) / BK : ℕ) = steps d.K := rfl
    rw [h]; omega
  have h2 : (IntV.ofInt ((d.K + (BK - 1)) / BK : ℕ) : I32).toInt = (((d.K + (BK - 1)) / BK : ℕ) : ℤ) :=
    lm_I32_ofInt_toInt hn
  rw [h0, h1, h2]
  unfold tl.tripCount
  have hsum : (((d.K + (BK - 1)) / BK : ℕ) : ℤ) - 0 + 1 - 1 = (((d.K + (BK - 1)) / BK : ℕ) : ℤ) := by ring
  rw [hsum]
  rw [Int.ediv_one, Int.toNat_natCast]
  rfl

/-- The loop bounds of line 148 are within `tl.forRange`'s covered range. -/
theorem rangeOk {d : Inst} (hd : d.ok) :
    tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofInt ((d.K + (BK - 1)) / BK : ℕ) : I32)
      (IntV.ofInt 1 : I32) := by
  unfold tl.RangeOk
  have h0 : (IntV.ofInt 0 : I32).toInt = (0:ℤ) := by
    have h : (-(2:ℤ)^(32-1) ≤ (0:ℤ) ∧ (0:ℤ) < 2^(32-1)) := by norm_num
    simpa using IntV.I32.toInt_ofInt (x := (0:ℤ)) h
  have h1 : (IntV.ofInt 1 : I32).toInt = (1:ℤ) := by
    have h : (-(2:ℤ)^(32-1) ≤ (1:ℤ) ∧ (1:ℤ) < 2^(32-1)) := by norm_num
    simpa using IntV.I32.toInt_ofInt (x := (1:ℤ)) h
  have hs : steps d.K ≤ 32 := steps_le hd.K_le
  have hn : ((d.K + (BK - 1)) / BK : ℕ) < 2 ^ 31 := by
    have h : ((d.K + (BK - 1)) / BK : ℕ) = steps d.K := rfl
    rw [h]; omega
  have h2 : (IntV.ofInt ((d.K + (BK - 1)) / BK : ℕ) : I32).toInt = (((d.K + (BK - 1)) / BK : ℕ) : ℤ) :=
    lm_I32_ofInt_toInt hn
  rw [h0, h1, h2]
  simp only [IntV.maxInt, ite_true]
  have hs2 : (((d.K + (BK - 1)) / BK : ℕ) : ℤ) = (steps d.K : ℤ) := by
    have h : ((d.K + (BK - 1)) / BK : ℕ) = steps d.K := rfl
    rw [h]
  rw [hs2]
  have hstep : (steps d.K : ℤ) + 1 - 1 = (steps d.K : ℤ) := by ring
  rw [hstep]
  constructor
  · norm_num
  constructor
  · have : (steps d.K : ℤ) ≤ 32 := by exact_mod_cast hs
    have h2 : (2:ℤ)^(32-1) - 1 = 2147483647 := by norm_num
    omega
  constructor
  · have h2 : (2:ℤ)^(32-1) - 1 = 2147483647 := by norm_num
    omega
  · have h2 : (2:ℤ)^(32-1) - 1 = 2147483647 := by norm_num
    omega

end Claims.SsdBmm



