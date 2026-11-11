import Proof.Tail
import Proof.Zero
import Proof.Loop
import Proof.Masks
import Proof.Dots

/-!
# `scaled_mm_int8`: one program is safe

`safe_program`: walk the kernel body of program `p` line by line. The prologue computes the tile
origins, offsets and masks; the loop runs `loop_safe` for each trip from `inv_zero`; the epilogue
scales the exact `int32` sum, adds the optional bias and stores, with `value_nobias`,
`value_bias` and `tail_safe_cov`.
-/

namespace Claims.ScaledMmInt8
open TVA

set_option maxHeartbeats 2000000
/-- The `scale_a` block `[BLOCK_SIZE_SCALE_A, 1]` broadcasts to `[BLOCK_SIZE_M, 1]` on every covered launch. -/
theorem h_sa_bcast_ok {d : Inst} : Shape.bcastOk [BSA d, 1] [d.BM, 1] = true := by
  cases hs : d.saVec <;> simp [Shape.bcastOk, BSA, hs]

/-- The `scale_b` block `[BLOCK_SIZE_SCALE_B, 1]` broadcasts to `[BLOCK_SIZE_N, 1]` on every covered launch. -/
theorem h_sb_bcast_ok {d : Inst} : Shape.bcastOk [BSB d, 1] [d.BN, 1] = true := by
  cases hs : d.sbVec <;> simp [Shape.bcastOk, BSB, hs]

/-- `tl.arange 0 1 < 1` in its only lane. -/
theorem h_arange_lt_one (k : Fin 1) :
    ((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).get1 k = true := by
  have har : ((tl.arange 0 1).get (k, ())).toInt = ((k.1 : Nat) : ℤ) :=
    m_arange_toInt (n := 1) (by norm_num) k
  show IntV.lt ((tl.arange 0 1).get (k, ())) (IntV.ofInt 1) = true
  rw [IntV.lt_eq, har, IntV.I32.toInt_ofInt (by constructor <;> norm_num)]
  simp only [decide_eq_true_eq]
  omega

/-- The full `A` scale mask in lane `i`. -/
theorem h_mask_sa_full (d : Inst) (hd : d.ok) (p : Nat) (hp : p < grid d)
    (pmT : Tensor [] I32) (hpm : pmT.item.toInt = (pm d p : ℤ)) :
    ∀ i : Idx [BSA d, 1],
      ((((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).ltS (IntV.ofNat d.M : I32)).col 1).and
        ((((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).col 1).bcastRows (BSA d))).get i
      = decide (saOff d p i.1.1 < d.M) := by
  intro i
  have hb : ((((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).col 1).bcastRows (BSA d)).get i
      = ((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).get1 (⟨0, Nat.zero_lt_one⟩ : Fin 1) := rfl
  have ha : (((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).ltS (IntV.ofNat d.M : I32)).col 1).get i
      = ((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).ltS (IntV.ofNat d.M : I32)).get (i.1, ()) := rfl
  rw [Tensor.get_and, hb, ha,
    show ((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).get1
        (⟨0, Nat.zero_lt_one⟩ : Fin 1) = true from h_arange_lt_one _,
    Bool.and_true, mask_sa d hd p hp pmT hpm (i.1, ())]

/-- The full `B` scale mask in lane `i`. -/
theorem h_mask_sb_full (d : Inst) (hd : d.ok) (p : Nat)
    (pnT : Tensor [] I32) (hpn : pnT.item.toInt = (pn d p : ℤ)) :
    ∀ i : Idx [BSB d, 1],
      ((((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).ltS (IntV.ofNat d.N : I32)).col 1).and
        ((((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).row 1).bcastRows (BSB d))).get i
      = decide (sbOff d p i.1.1 < d.N) := by
  intro i
  have hb : ((((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).row 1).bcastRows (BSB d)).get i
      = ((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).get1 i.2.1 := rfl
  have ha : (((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).ltS (IntV.ofNat d.N : I32)).col 1).get i
      = ((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).ltS (IntV.ofNat d.N : I32)).get (i.1, ()) := rfl
  rw [Tensor.get_and, hb, ha,
    show ((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).get1 i.2.1 = true from
      h_arange_lt_one i.2.1,
    Bool.and_true, mask_sb d hd p pnT hpn (i.1, ())]

/-- `saIdx` of an `saOff` of a lane in `[0, BSA)`. -/
theorem h_saIdx_saOff (d : Inst) (hd : d.ok) (p j : Nat) :
    saIdx d (saOff d p j) = if d.saVec then saOff d p j else 0 := by
  by_cases hs : d.saVec = true
  · rw [saIdx, ite_eq_left hs, hd.scales_contig.1 hs, Nat.mul_one, ite_eq_left hs]
  · rw [saIdx, ite_eq_right hs, ite_eq_right hs]

/-- `sbIdx` of an `sbOff` of a lane in `[0, BSB)`. -/
theorem h_sbIdx_sbOff (d : Inst) (hd : d.ok) (p j : Nat) :
    sbIdx d (sbOff d p j) = if d.sbVec then sbOff d p j else 0 := by
  by_cases hs : d.sbVec = true
  · rw [sbIdx, ite_eq_left hs, hd.scales_contig.2.1 hs, Nat.mul_one, ite_eq_left hs]
  · rw [sbIdx, ite_eq_right hs, ite_eq_right hs]

/-- The pointer of a lane of the `A` scale load. -/
theorem h_sa_ptr_get (d : Inst) (hd : d.ok) (p : Nat) (hp : p < grid d)
    (pmT : Tensor [] I32) (hpm : pmT.item.toInt = (pm d p : ℤ)) :
    ∀ a : Idx [BSA d, 1],
      ((d.sa.ptr 0 + (tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).col 1).get a)
      = d.sa.ptr (saIdx d (saOff d p a.1.1) : Int) := by
  intro a
  apply Ptr.eq_of_addr
  have h0 : ((d.sa.ptr 0 + (tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32)).col 1).get a).addr
      = ((d.sa.ptr 0 + (tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32))).get (a.1, ())).addr := rfl
  have h1 : ((d.sa.ptr 0 + (tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32))).get (a.1, ())).addr
      = (d.sa.ptr 0).addr + (((tl.arange 0 (BSA d) +
        (IntV.ofNat (if 1 < BSA d then 1 else 0) : I32) * pmT.item *
          (IntV.ofNat d.BM : I32))).get (a.1, ())).toInt * DType.f32.bytes := rfl
  rw [h0, h1, Buf.ptr_addr, Buf.ptr_addr]
  rw [offs_sa_toInt d hd p hp pmT hpm (a.1, ()),
    h_saIdx_saOff d hd p a.1.1]
  by_cases hs : d.saVec
  · rw [ite_eq_left hs]; ring
  · rw [ite_eq_right hs, saOff_eq hd p a.1.1, ite_eq_right hs]
    have h0 : (a.1.1 : Nat) = 0 := by
      have hBSA : BSA d = 1 := by simp [BSA, hs]
      have := a.1.2
      omega
    rw [h0]; ring

/-- The pointer of a lane of the `B` scale load. -/
theorem h_sb_ptr_get (d : Inst) (hd : d.ok) (p : Nat)
    (pnT : Tensor [] I32) (hpn : pnT.item.toInt = (pn d p : ℤ)) :
    ∀ a : Idx [BSB d, 1],
      ((d.sb.ptr 0 + (tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).col 1).get a)
      = d.sb.ptr (sbIdx d (sbOff d p a.1.1) : Int) := by
  intro a
  apply Ptr.eq_of_addr
  have h0 : ((d.sb.ptr 0 + (tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32)).col 1).get a).addr
      = ((d.sb.ptr 0 + (tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32))).get (a.1, ())).addr := rfl
  have h1 : ((d.sb.ptr 0 + (tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32))).get (a.1, ())).addr
      = (d.sb.ptr 0).addr + (((tl.arange 0 (BSB d) +
        (IntV.ofNat (if 1 < BSB d then 1 else 0) : I32) * pnT.item *
          (IntV.ofNat d.BN : I32))).get (a.1, ())).toInt * DType.f32.bytes := rfl
  rw [h0, h1, Buf.ptr_addr, Buf.ptr_addr]
  rw [offs_sb_toInt d hd p pnT hpn (a.1, ()),
    h_sbIdx_sbOff d hd p a.1.1]
  by_cases hs : d.sbVec
  · rw [ite_eq_left hs]; ring
  · rw [ite_eq_right hs, sbOff_eq hd p a.1.1, ite_eq_right hs]
    have h0 : (a.1.1 : Nat) = 0 := by
      have hBSB : BSB d = 1 := by simp [BSB, hs]
      have := a.1.2
      omega
    rw [h0]; ring

/-- The lane of the `A` scale that the broadcast gives row `i`. -/
def saLane (d : Inst) (i : Fin d.BM) : Fin (BSA d) :=
  if h : BSA d = 1 then ⟨0, by omega⟩ else ⟨i.1, by
    have hBSA : BSA d = d.BM := by
      cases hs : d.saVec
      · simp [BSA, hs] at h
      · simp [BSA, hs]
    have := i.2
    omega⟩

/-- The lane of the `B` scale that the broadcast gives column `j`. -/
def sbLane (d : Inst) (j : Fin d.BN) : Fin (BSB d) :=
  if h : BSB d = 1 then ⟨0, by omega⟩ else ⟨j.1, by
    have hBSB : BSB d = d.BN := by
      cases hs : d.sbVec
      · simp [BSB, hs] at h
      · simp [BSB, hs]
    have := j.2
    omega⟩

/-- The `scale_a` lane that tile row `i` reads: `0` for a per-tensor scale, `i` for a per-token one. -/
theorem h_saLane_val (d : Inst) (i : Fin d.BM) :
    (saLane d i).1 = if BSA d = 1 then 0 else i.1 := by
  unfold saLane
  split <;> rfl

/-- The `scale_b` lane that tile column `j` reads: `0` for a per-tensor scale, `j` for a per-channel one. -/
theorem h_sbLane_val (d : Inst) (j : Fin d.BN) :
    (sbLane d j).1 = if BSB d = 1 then 0 else j.1 := by
  unfold sbLane
  split <;> rfl

/-- A row inside `M` reads a `scale_a` element inside the scale vector. -/
theorem h_saLane_off_lt (d : Inst) (hd : d.ok) (p : Nat) {i : Fin d.BM}
    (hiM : pm d p * d.BM + i.1 < d.M) : saOff d p (saLane d i).1 < d.M := by
  cases hs : d.saVec
  · have hBSA : BSA d = 1 := by simp [BSA, hs]
    rw [saOff_eq hd, ite_eq_right (by simp [hs]), h_saLane_val, ite_eq_left hBSA]
    exact hd.M_pos
  · have hBSA : BSA d = d.BM := by simp [BSA, hs]
    have hne : ¬ (BSA d = 1) := by rw [hBSA]; have := BM_ge2 hd; omega
    rw [saOff_eq hd, ite_eq_left hs, h_saLane_val, ite_eq_right hne]
    omega

/-- A column inside `N` reads a `scale_b` element inside the scale vector. -/
theorem h_sbLane_off_lt (d : Inst) (hd : d.ok) (p : Nat) {j : Fin d.BN}
    (hjN : pn d p * d.BN + j.1 < d.N) : sbOff d p (sbLane d j).1 < d.N := by
  cases hs : d.sbVec
  · have hBSB : BSB d = 1 := by simp [BSB, hs]
    rw [sbOff_eq hd, ite_eq_right (by simp [hs]), h_sbLane_val, ite_eq_left hBSB]
    exact hd.N_pos
  · have hBSB : BSB d = d.BN := by simp [BSB, hs]
    have hne : ¬ (BSB d = 1) := by rw [hBSB]; have := BN_ge2 hd; omega
    rw [sbOff_eq hd, ite_eq_left hs, h_sbLane_val, ite_eq_right hne]
    omega

/-- The memory address of the broadcast `A` scale lane. -/
theorem h_saIdx_saLane (d : Inst) (hd : d.ok) (p : Nat) (i : Fin d.BM) :
    saIdx d (saOff d p (saLane d i).1) = saIdx d (pm d p * d.BM + i.1) := by
  cases hs : d.saVec
  · rw [saIdx, ite_eq_right (by simpa using hs), saIdx, ite_eq_right (by simpa using hs)]
  · have hL : saIdx d (saOff d p (saLane d i).1) = saOff d p (saLane d i).1 := by
      rw [saIdx, ite_eq_left hs, hd.scales_contig.1 hs, Nat.mul_one]
    have hR : saIdx d (pm d p * d.BM + i.1) = pm d p * d.BM + i.1 := by
      rw [saIdx, ite_eq_left hs, hd.scales_contig.1 hs, Nat.mul_one]
    rw [hL, hR, saOff_eq hd, ite_eq_left hs]
    have hne : ¬ (BSA d = 1) := by
      have hBSA : BSA d = d.BM := by simp [BSA, hs]
      rw [hBSA]; have := BM_ge2 hd; omega
    rw [h_saLane_val, ite_eq_right hne]
    omega

/-- The memory address of the broadcast `B` scale lane. -/
theorem h_sbIdx_sbLane (d : Inst) (hd : d.ok) (p : Nat) (j : Fin d.BN) :
    sbIdx d (sbOff d p (sbLane d j).1) = sbIdx d (pn d p * d.BN + j.1) := by
  cases hs : d.sbVec
  · rw [sbIdx, ite_eq_right (by simpa using hs), sbIdx, ite_eq_right (by simpa using hs)]
  · have hL : sbIdx d (sbOff d p (sbLane d j).1) = sbOff d p (sbLane d j).1 := by
      rw [sbIdx, ite_eq_left hs, hd.scales_contig.2.1 hs, Nat.mul_one]
    have hR : sbIdx d (pn d p * d.BN + j.1) = pn d p * d.BN + j.1 := by
      rw [sbIdx, ite_eq_left hs, hd.scales_contig.2.1 hs, Nat.mul_one]
    rw [hL, hR, sbOff_eq hd, ite_eq_left hs]
    have hne : ¬ (BSB d = 1) := by
      have hBSB : BSB d = d.BN := by simp [BSB, hs]
      rw [hBSB]; have := BN_ge2 hd; omega
    rw [h_sbLane_val, ite_eq_right hne]
    omega

/-- What the broadcast `A` scale gives row `i` is what the load put in `saLane d i`. -/
theorem h_sa_bcast_get {d : Inst} {p : Nat} {init : Mem} {sa : Tensor [BSA d, 1] F32}
    (hd : d.ok) (hbc : Shape.bcastOk [BSA d, 1] [d.BM, 1] = true)
    (hload : ∀ a : Idx [BSA d, 1], decide (saOff d p a.1.1 < d.M) = true →
      init.read .f32 (d.sa.ptr (saIdx d (saOff d p a.1.1) : Int)).addr = some (sa.get a))
    (i : Fin d.BM) (j : Fin d.BN) (hiM : pm d p * d.BM + i.1 < d.M) {saL : F32}
    (hL : init.read .f32 (d.sa.ptr (saIdx d (pm d p * d.BM + i.1) : Int)).addr = some saL) :
    ((sa.broadcastTo [d.BM, 1] hbc).bcastCols d.BN).get2 i j = saL := by
  have h1 : ((sa.broadcastTo [d.BM, 1] hbc).bcastCols d.BN).get2 i j
      = sa.get2 (saLane d i) ⟨0, Nat.zero_lt_one⟩ := by
    have h2 : (sa.broadcastTo [d.BM, 1] hbc).get
        ((i, (⟨0, Nat.zero_lt_one⟩ : Fin 1), ()) : Idx [d.BM, 1])
        = sa.get ((saLane d i, (⟨0, Nat.zero_lt_one⟩ : Fin 1), ()) : Idx [BSA d, 1]) := by
      rw [Tensor.get_broadcastTo sa [d.BM, 1] hbc (i, ⟨0, Nat.zero_lt_one⟩, ())
        (saLane d i, ⟨0, Nat.zero_lt_one⟩, ()) ?_]
      intro k hk
      match k, hk with
      | 0, _ =>
        show (saLane d i).1 = if Shape.dim [BSA d, 1] 0 = 1 then 0 else i.1
        rw [h_saLane_val]
        rfl
      | 1, _ => rfl
    exact h2
  rw [h1]
  have h2 := hload ((saLane d i, ⟨0, Nat.zero_lt_one⟩, ()) : Idx [BSA d, 1])
    (by simpa using h_saLane_off_lt d hd p hiM)
  rw [h_saIdx_saLane d hd p i] at h2
  rw [h2] at hL
  exact Option.some.inj hL

/-- What the broadcast `B` scale gives column `j` is what the load put in `sbLane d j`. -/
theorem h_sb_bcast_get {d : Inst} {p : Nat} {init : Mem} {sb : Tensor [BSB d, 1] F32}
    (hd : d.ok) (hbc : Shape.bcastOk [BSB d, 1] [d.BN, 1] = true)
    (hload : ∀ a : Idx [BSB d, 1], decide (sbOff d p a.1.1 < d.N) = true →
      init.read .f32 (d.sb.ptr (sbIdx d (sbOff d p a.1.1) : Int)).addr = some (sb.get a))
    (i : Fin d.BM) (j : Fin d.BN) (hjN : pn d p * d.BN + j.1 < d.N) {sbL : F32}
    (hL : init.read .f32 (d.sb.ptr (sbIdx d (pn d p * d.BN + j.1) : Int)).addr = some sbL) :
    ((sb.broadcastTo [d.BN, 1] hbc).trans.bcastRows d.BM).get2 i j = sbL := by
  have h1 : ((sb.broadcastTo [d.BN, 1] hbc).trans.bcastRows d.BM).get2 i j
      = sb.get2 (sbLane d j) ⟨0, Nat.zero_lt_one⟩ := by
    have h2 : (sb.broadcastTo [d.BN, 1] hbc).get
        ((j, (⟨0, Nat.zero_lt_one⟩ : Fin 1), ()) : Idx [d.BN, 1])
        = sb.get ((sbLane d j, (⟨0, Nat.zero_lt_one⟩ : Fin 1), ()) : Idx [BSB d, 1]) := by
      rw [Tensor.get_broadcastTo sb [d.BN, 1] hbc (j, ⟨0, Nat.zero_lt_one⟩, ())
        (sbLane d j, ⟨0, Nat.zero_lt_one⟩, ()) ?_]
      intro k hk
      match k, hk with
      | 0, _ =>
        show (sbLane d j).1 = if Shape.dim [BSB d, 1] 0 = 1 then 0 else j.1
        rw [h_sbLane_val]
        rfl
      | 1, _ => rfl
    exact h2
  rw [h1]
  have h2 := hload ((sbLane d j, ⟨0, Nat.zero_lt_one⟩, ()) : Idx [BSB d, 1])
    (by simpa using h_sbLane_off_lt d hd p hjN)
  rw [h_sbIdx_sbLane d hd p j] at h2
  rw [h2] at hL
  exact Option.some.inj hL


/-! ### The epilogue's tail, with only the covered lanes specified -/

/-- The tail (convert to the output dtype, build the pointers and mask, store) is safe, given that every conversion of the `float32` accumulator satisfies `LaneQ` in the covered lanes. -/
theorem tail_safe_cov (P : Platform) (d : Inst) (hd : d.ok) (init : Mem) (hpre : Pre d init)
    (p : Nat) (hp : p < grid d) (pmT pnT : I32) (acc : Tensor [d.BM, d.BN] F32)
    (hpm : pmT.toInt = (pm d p : Int)) (hpn : pnT.toInt = (pn d p : Int))
    (hval : ∀ (cc : Tensor [d.BM, d.BN] d.od.Val),
      (∀ i, ConvRel P .f32 d.od (acc.get i) (cc.get i)) →
      ∀ (i : Idx [d.BM, d.BN]), pm d p * d.BM + i.1 < d.M → pn d p * d.BN + i.2.1 < d.N →
        LaneQ d init (pm d p * d.BM + i.1) (pn d p * d.BN + i.2.1) (cc.get i)) :
    Safe ⟨P, true⟩ (Rfp d p) (Wfp d p) (fun _ m => Qfp d init p m) init
      (do
        let c : Tensor [d.BM, d.BN] d.od.Val ← tl.convert .f32 d.od acc
        let offs_cm : Tensor [d.BM] I64 :=
          (← tl.convert .i32 .i64 (Tensor.scalar (pmT * (IntV.ofNat d.BM : I32)))).item +
            (← tl.convert .i32 .i64 (tl.arange 0 d.BM))
        let offs_cn : Tensor [d.BN] I64 :=
          (← tl.convert .i32 .i64 (Tensor.scalar (pnT * (IntV.ofNat d.BN : I32)))).item +
            (← tl.convert .i32 .i64 (tl.arange 0 d.BN))
        let c_ptrs := d.c.ptr 0 +
          ((← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.stride_cm : I32))).item * offs_cm).col d.BN +
          ((← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.stride_cn : I32))).item * offs_cn).row d.BM
        let c_mask := ((offs_cm.col d.BN).ltS
            (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.M : I32))).item).and
          ((offs_cn.row d.BM).ltS
            (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.N : I32))).item)
        tl.store c_ptrs c (some c_mask)) := by
  refine safe_convert (d := .f32) (d' := d.od) (x := acc)
    (fun i => convRel_f32_exists hd.od_ok (acc.get i)) ?_
  intro cc hcc
  refine safe_convert (d := .i32) (d' := .i64)
    (x := Tensor.scalar (pmT * (IntV.ofNat d.BM : I32))) ?_ ?_
  · intro i; exact ⟨_, rfl⟩
  · intro r4 h4
    refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BM) ?_ ?_
    · intro i; exact ⟨_, rfl⟩
    · intro r5 h5
      refine safe_convert (d := .i32) (d' := .i64)
        (x := Tensor.scalar (pnT * (IntV.ofNat d.BN : I32))) ?_ ?_
      · intro i; exact ⟨_, rfl⟩
      · intro r6 h6
        refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BN) ?_ ?_
        · intro i; exact ⟨_, rfl⟩
        · intro r7 h7
          refine safe_convert (d := .i32) (d' := .i64)
            (x := Tensor.scalar (IntV.ofNat d.stride_cm : I32)) ?_ ?_
          · intro i; exact ⟨_, rfl⟩
          · intro r8 h8
            refine safe_convert (d := .i32) (d' := .i64)
              (x := Tensor.scalar (IntV.ofNat d.stride_cn : I32)) ?_ ?_
            · intro i; exact ⟨_, rfl⟩
            · intro r9 h9
              refine safe_convert (d := .i32) (d' := .i64)
                (x := Tensor.scalar (IntV.ofNat d.M : I32)) ?_ ?_
              · intro i; exact ⟨_, rfl⟩
              · intro r10 h10
                refine safe_convert (d := .i32) (d' := .i64)
                  (x := Tensor.scalar (IntV.ofNat d.N : I32)) ?_ ?_
                · intro i; exact ⟨_, rfl⟩
                · intro r11 h11
                  have e4 : r4.item.toInt = ((pm d p * d.BM : Nat) : Int) := by
                    have h := convRel_i32_i64 (h4 ())
                    have h2 : (pmT * (IntV.ofNat d.BM : I32)).toInt
                        = ((pm d p * d.BM : Nat) : Int) :=
                      IntV.I32.mul_nat hpm
                        (IntV.I32.toInt_ofNat (by have := BM_le256 hd; omega))
                        (pmBM_lt31 hd hp)
                    exact h.trans h2
                  have e5 : ∀ k : Fin d.BM, (r5.get (k, ())).toInt = (k.1 : Int) := by
                    intro k
                    have h := convRel_i32_i64 (h5 (k, ()))
                    have h2 : ((tl.arange 0 d.BM).get ((k, ()) : Idx [d.BM])).toInt
                        = (k.1 : Int) := by
                      have hk : (k.1 : Int) < 2 ^ 31 := by
                        have := k.2
                        have := BM_le256 hd
                        omega
                      rw [tl.arange, Tensor.get_arange]
                      simp only [Nat.cast_zero, zero_add]
                      have hsf : IntV.SFits 32 (k.1 : Int) := by
                        simp only [IntV.SFits]
                        push_cast
                        constructor <;> omega
                      rw [IntV.toInt_ofInt (by norm_num) hsf]
                    exact h.trans h2
                  have e6 : r6.item.toInt = ((pn d p * d.BN : Nat) : Int) := by
                    have h := convRel_i32_i64 (h6 ())
                    have h2 : (pnT * (IntV.ofNat d.BN : I32)).toInt
                        = ((pn d p * d.BN : Nat) : Int) :=
                      IntV.I32.mul_nat hpn
                        (IntV.I32.toInt_ofNat (by have := BN_le256 hd; omega))
                        (pnBN_lt31 hd p)
                    exact h.trans h2
                  have e7 : ∀ k : Fin d.BN, (r7.get (k, ())).toInt = (k.1 : Int) := by
                    intro k
                    have h := convRel_i32_i64 (h7 (k, ()))
                    have h2 : ((tl.arange 0 d.BN).get ((k, ()) : Idx [d.BN])).toInt
                        = (k.1 : Int) := by
                      have hk : (k.1 : Int) < 2 ^ 31 := by
                        have := k.2
                        have := BN_le256 hd
                        omega
                      rw [tl.arange, Tensor.get_arange]
                      simp only [Nat.cast_zero, zero_add]
                      have hsf : IntV.SFits 32 (k.1 : Int) := by
                        simp only [IntV.SFits]
                        push_cast
                        constructor <;> omega
                      rw [IntV.toInt_ofInt (by norm_num) hsf]
                    exact h.trans h2
                  have e8 : r8.item.toInt = (d.stride_cm : Int) := by
                    have h := convRel_i32_i64 (h8 ())
                    have h2 : (IntV.ofNat d.stride_cm : I32).toInt = (d.stride_cm : Int) :=
                      IntV.I32.toInt_ofNat (by have := hd.strides.2.2.2.2.1; omega)
                    exact h.trans h2
                  have e9 : r9.item.toInt = (d.stride_cn : Int) := by
                    have h := convRel_i32_i64 (h9 ())
                    have h2 : (IntV.ofNat d.stride_cn : I32).toInt = (d.stride_cn : Int) :=
                      IntV.I32.toInt_ofNat (by have := hd.strides.2.2.2.2.2; omega)
                    exact h.trans h2
                  have e10 : r10.item.toInt = (d.M : Int) := by
                    have h := convRel_i32_i64 (h10 ())
                    have h2 : (IntV.ofNat d.M : I32).toInt = (d.M : Int) :=
                      IntV.I32.toInt_ofNat (by have := hd.M_le; have := BM_pos hd; omega)
                    exact h.trans h2
                  have e11 : r11.item.toInt = (d.N : Int) := by
                    have h := convRel_i32_i64 (h11 ())
                    have h2 : (IntV.ofNat d.N : I32).toInt = (d.N : Int) :=
                      IntV.I32.toInt_ofNat (by have := hd.N_le; have := BN_pos hd; omega)
                    exact h.trans h2
                  refine safe_store_buf_ret (b := d.c)
                    (g := fun i : Idx [d.BM, d.BN] =>
                      (pm d p * d.BM + (i.1 : Nat)) * d.stride_cm +
                        (pn d p * d.BN + (i.2.1 : Nat)) * d.stride_cn)
                    (en := fun i : Idx [d.BM, d.BN] =>
                      decide (pm d p * d.BM + (i.1 : Nat) < d.M) &&
                        decide (pn d p * d.BN + (i.2.1 : Nat) < d.N))
                    ?_ ?_ ?_ ?_ ?_ ?_
                  · refine Buf.addrs_of_get d.c ?_ ?_
                    · intro i
                      exact t_mask_get hd hp r4 r6 r10 r11 r5 r7 e4 e5 e6 e7 e10 e11 i
                    · intro i hi
                      exact t_cptr_get hd hp r4 r6 r8 r9 r5 r7 e4 e5 e6 e7 e8 e9 i
                  · intro i hi
                    simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
                    exact hd.c_len _ _ hi.1 hi.2
                  · intro i j hi hj hg
                    simp only [Bool.and_eq_true, decide_eq_true_eq] at hi hj
                    obtain ⟨i1, i2, u⟩ := i
                    obtain ⟨j1, j2, v⟩ := j
                    have hg' : (pm d p * d.BM + (i1 : Nat)) * d.stride_cm +
                          (pn d p * d.BN + (i2 : Nat)) * d.stride_cn =
                        (pm d p * d.BM + (j1 : Nat)) * d.stride_cm +
                          (pn d p * d.BN + (j2 : Nat)) * d.stride_cn := hg
                    have hinj := hd.c_inj (pm d p * d.BM + (i1 : Nat))
                      (pn d p * d.BN + (i2 : Nat)) (pm d p * d.BM + (j1 : Nat))
                      (pn d p * d.BN + (j2 : Nat)) hi.1 hi.2 hj.1 hj.2 hg'
                    refine Prod.ext (Fin.ext ?_) (Prod.ext (Fin.ext ?_) ?_)
                    · exact Nat.add_left_cancel hinj.1
                    · exact Nat.add_left_cancel hinj.2
                    · cases u; cases v; rfl
                  · exact hpre.2.2.2.2.1
                  · refine Buf.elems_mono d.c ?_
                    intro e he
                    obtain ⟨i, hen, rfl⟩ := he
                    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
                    exact ⟨i.1, i.2.1, i.1.isLt, i.2.1.isLt, hen.1, hen.2, rfl⟩
                  · intro m' hread hframe hpres
                    refine Safe.ret ?_
                    intro i j hi hj hiM hjN
                    have hL := hval cc hcc ((⟨i, hi⟩, ⟨j, hj⟩, ()) : Idx [d.BM, d.BN])
                      (by simpa using hiM) (by simpa using hjN)
                    refine ⟨_, ?_, hL⟩
                    have hen : (fun i' : Idx [d.BM, d.BN] =>
                        decide (pm d p * d.BM + (i'.1 : Nat) < d.M) &&
                          decide (pn d p * d.BN + (i'.2.1 : Nat) < d.N))
                        ((⟨i, hi⟩, ⟨j, hj⟩, ()) : Idx [d.BM, d.BN]) = true := by
                      simp only [Bool.and_eq_true, decide_eq_true_eq]
                      exact ⟨by simpa using hiM, by simpa using hjN⟩
                    have := hread _ hen
                    simpa using this

/-! ### The bias value in `float32` -/

/-- A finite bias of the output dtype converts exactly to `float32`. -/
theorem bias_f32_toRat {P : Platform} {od : DType} (hod : od = .bf16 ∨ od = .f16)
    {x : od.Val} {β : ℚ} {r : F32} (hv : DType.val? od x = some β)
    (hc : ConvRel P od .f32 x r) : r.isFinite = true ∧ r.toRat = β := by
  have h1 : ¬ (Format.bfloat16 = Format.binary32) := by decide
  have h2 : ¬ (Format.binary16 = Format.binary32) := by decide
  have h3 : ¬ Format.binary32.F8 := by
    unfold Format.F8
    rintro (h | h) <;> exact absurd h (by decide)
  have h4 : ¬ Format.bfloat16.F8 := by
    unfold Format.F8
    rintro (h | h) <;> exact absurd h (by decide)
  have h5 : ¬ Format.binary16.F8 := by
    unfold Format.F8
    rintro (h | h) <;> exact absurd h (by decide)
  rcases hod with rfl | rfl
  · have hne : FloatV.NaNEq (FloatV.cvt x : FloatV Format.binary32) r := by
      have h := hc
      unfold ConvRel at h
      simp only [ofFloatRel, FloatRel] at h
      simp only [h1, h3, h4, ite_false, false_and, and_false, or_false] at h
      exact h.2.2
    have hx : x.isFinite = true ∧ x.toRat = β := by
      have h := hv
      change (if x.isFinite = true then some x.toRat else none) = some β at h
      by_cases hf : x.isFinite = true
      · rw [ite_eq_left hf] at h
        exact ⟨hf, Option.some.inj h⟩
      · rw [ite_eq_right hf] at h
        exact absurd h (by simp)
    obtain ⟨hcf, hcr⟩ := FloatV.toRat_cvt_of_wider (f := .bfloat16) (g := .binary32)
      Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32 .rn hx.1
    have hcf' : (FloatV.cvt x : FloatV Format.binary32).isFinite = true := hcf
    have hcr' : (FloatV.cvt x : FloatV Format.binary32).toRat = x.toRat := hcr
    have hnan : (FloatV.cvt x : FloatV Format.binary32).isNaN = false :=
      (FloatV.isFinite_iff.mp hcf').1
    have hr : r = FloatV.cvt x := FloatV.NaNEq.eq hne hnan
    exact ⟨by rw [hr]; exact hcf', by rw [hr, hcr']; exact hx.2⟩
  · have hne : FloatV.NaNEq (FloatV.cvt x : FloatV Format.binary32) r := by
      have h := hc
      unfold ConvRel at h
      simp only [ofFloatRel, FloatRel] at h
      simp only [h2, h3, h5, ite_false, false_and, and_false, or_false] at h
      exact h.2.2
    have hx : x.isFinite = true ∧ x.toRat = β := by
      have h := hv
      change (if x.isFinite = true then some x.toRat else none) = some β at h
      by_cases hf : x.isFinite = true
      · rw [ite_eq_left hf] at h
        exact ⟨hf, Option.some.inj h⟩
      · rw [ite_eq_right hf] at h
        exact absurd h (by simp)
    obtain ⟨hcf, hcr⟩ := FloatV.toRat_cvt_of_wider (f := .binary16) (g := .binary32)
      Format.binary16_regular Format.binary32_regular Format.wider_binary16_binary32 .rn hx.1
    have hcf' : (FloatV.cvt x : FloatV Format.binary32).isFinite = true := hcf
    have hcr' : (FloatV.cvt x : FloatV Format.binary32).toRat = x.toRat := hcr
    have hnan : (FloatV.cvt x : FloatV Format.binary32).isNaN = false :=
      (FloatV.isFinite_iff.mp hcf').1
    have hr : r = FloatV.cvt x := FloatV.NaNEq.eq hne hnan
    exact ⟨by rw [hr]; exact hcf', by rw [hr, hcr']; exact hx.2⟩

/-! ### The read footprint contains the epilogue's loads -/

/-- The `scale_a` elements the kernel loads lie in `saTile`. -/
theorem h_sa_img_sub (d : Inst) (p : Nat) :
    Buf.img (fun a : Idx [BSA d, 1] => saIdx d (saOff d p a.1.1))
      (fun a : Idx [BSA d, 1] => decide (saOff d p a.1.1 < d.M)) ⊆ saTile d p := by
  rintro e ⟨a, ha, rfl⟩
  simp only [decide_eq_true_eq] at ha
  exact ⟨a.1.1, a.1.2, ha, rfl⟩

/-- Their bytes lie in the read footprint. -/
theorem h_sa_elems_sub (d : Inst) (p : Nat) :
    d.sa.elems (Buf.img (fun a : Idx [BSA d, 1] => saIdx d (saOff d p a.1.1))
      (fun a : Idx [BSA d, 1] => decide (saOff d p a.1.1 < d.M))) ⊆ Rfp d p := by
  refine Set.Subset.trans (Buf.elems_mono d.sa (h_sa_img_sub d p)) ?_
  intro e he
  simp only [Rfp]
  exact Set.mem_union_left _ (Set.mem_union_left _ (Set.mem_union_right _ he))

/-- The `scale_b` elements the kernel loads lie in `sbTile`. -/
theorem h_sb_img_sub (d : Inst) (p : Nat) :
    Buf.img (fun a : Idx [BSB d, 1] => sbIdx d (sbOff d p a.1.1))
      (fun a : Idx [BSB d, 1] => decide (sbOff d p a.1.1 < d.N)) ⊆ sbTile d p := by
  rintro e ⟨a, ha, rfl⟩
  simp only [decide_eq_true_eq] at ha
  exact ⟨a.1.1, a.1.2, ha, rfl⟩

/-- Their bytes lie in the read footprint. -/
theorem h_sb_elems_sub (d : Inst) (p : Nat) :
    d.sb.elems (Buf.img (fun a : Idx [BSB d, 1] => sbIdx d (sbOff d p a.1.1))
      (fun a : Idx [BSB d, 1] => decide (sbOff d p a.1.1 < d.N))) ⊆ Rfp d p := by
  refine Set.Subset.trans (Buf.elems_mono d.sb (h_sb_img_sub d p)) ?_
  intro e he
  simp only [Rfp]
  exact Set.mem_union_left _ (Set.mem_union_right _ he)

/-- The bias elements the kernel loads lie in `biasTile`. -/
theorem h_bias_img_sub (d : Inst) (p : Nat) (hb : d.hasBias = true) :
    Buf.img (fun a : Idx [d.BN] => pn d p * d.BN + (a.1 : Nat))
      (fun a : Idx [d.BN] => decide (pn d p * d.BN + (a.1 : Nat) < d.N)) ⊆ biasTile d p := by
  rintro e ⟨a, ha, rfl⟩
  simp only [decide_eq_true_eq] at ha
  rw [biasTile, ite_eq_left hb]
  simp only [Set.mem_ofPred_eq]
  exact ⟨a.1.1, a.1.2, ha, rfl⟩

/-- Their bytes lie in the read footprint. -/
theorem h_bias_elems_sub (d : Inst) (p : Nat) (hb : d.hasBias = true) :
    d.bias.elems (Buf.img (fun a : Idx [d.BN] => pn d p * d.BN + (a.1 : Nat))
      (fun a : Idx [d.BN] => decide (pn d p * d.BN + (a.1 : Nat) < d.N))) ⊆ Rfp d p := by
  refine Set.Subset.trans (Buf.elems_mono d.bias (h_bias_img_sub d p hb)) ?_
  intro e he
  simp only [Rfp]
  exact Set.mem_union_right _ he

/-! ### The bias pointer -/

/-- The pointer of bias lane `j` is the bias base plus `pn·BN + j`. -/
theorem h_bias_ptr_get (d : Inst) (hd : d.ok) (p : Nat)
    (r7 : Tensor [] I64) (r8 : Tensor [d.BN] I64)
    (h7 : r7.item.toInt = ((pn d p * d.BN : Nat) : ℤ))
    (h8 : ∀ j : Fin d.BN, (r8.get1 j).toInt = (j.1 : ℤ)) :
    ∀ i : Idx [d.BN],
      ((d.bias.ptr 0 + (r7.item + r8)).get i)
        = d.bias.ptr (((pn d p * d.BN + (i.1.1 : Nat)) : Nat) : ℤ) := by
  intro i
  apply Ptr.eq_of_addr
  rw [show ((d.bias.ptr 0 + (r7.item + r8)).get i)
      = (d.bias.ptr 0).offset (((r7.item + r8).get i).toInt) from rfl]
  rw [Ptr.offset_addr, Buf.ptr_addr, Buf.ptr_addr]
  rw [show ((r7.item + r8).get i) = ((r7.item + r8).get1 i.1) from rfl,
    offs_bn_toInt d hd p r7 r8 h7 h8 i.1]
  ring

/-! ### The accumulator as the exact dot -/

/-- `K / BLOCK_SIZE_K · BLOCK_SIZE_K = K`. -/
theorem hkmulBK (d : Inst) (hd : d.ok) : d.K / d.BK * d.BK = d.K :=
  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd.K_div)

/-- After all trips, an accumulator element that holds the sum over the first `K` terms is `dotSum` of the row of `A` and the column of `B`. -/
theorem acc_dotSum {d : Inst} {init : Mem} {p : Nat} {acc : Tensor [d.BM, d.BN] I32}
    {va vb : Nat → I8} {i : Fin d.BM} {j : Fin d.BN}
    (hA : ∀ k, k < d.K → init.read .i8
      (d.a.ptr (((pm d p * d.BM + i.1) * d.stride_am + k * d.stride_ak : Nat) : ℤ)).addr = some (va k))
    (hB : ∀ k, k < d.K → init.read .i8
      (d.b.ptr (((k * d.stride_bk + (pn d p * d.BN + j.1) * d.stride_bn : Nat) : ℤ))).addr = some (vb k))
    (hacc : (acc.get2 i j).toInt =
      ∑ k ∈ Finset.range (d.K / d.BK * d.BK),
        rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
    (hK : d.K / d.BK * d.BK = d.K) :
    (acc.get2 i j).toInt = dotSum va vb d.K := by
  rw [hacc, hK]
  show (∑ k ∈ Finset.range d.K,
      rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
    = ∑ k ∈ Finset.range d.K, (va k).toInt * (vb k).toInt
  refine Finset.sum_congr rfl ?_
  intro k hk
  rw [Finset.mem_range] at hk
  rw [rA_eq (hA k hk), rB_eq (hB k hk)]

/-! ### The lane specification from the epilogue's values -/

/-- Without a bias, the epilogue chain leaves `LaneQ` in the lane. -/
theorem laneQ_nobias (P : Platform) (d : Inst) (hd : d.ok) (init : Mem) (p : Nat)
    {acc : Tensor [d.BM, d.BN] I32} {accf acc1 acc2 : Tensor [d.BM, d.BN] F32}
    {bca bcb : Tensor [d.BM, d.BN] F32} {cc : Tensor [d.BM, d.BN] d.od.Val}
    (hb : d.hasBias = false)
    (hacc : ∀ (i : Fin d.BM) (j : Fin d.BN), pm d p * d.BM + i.1 < d.M →
      pn d p * d.BN + j.1 < d.N →
      (acc.get2 i j).toInt =
        ∑ k ∈ Finset.range (d.K / d.BK * d.BK),
          rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
    (haccf : ∀ x : Idx [d.BM, d.BN], tl.ItofRel .binary32 (acc.get x).toInt (accf.get x))
    (hacc1 : ∀ x : Idx [d.BM, d.BN],
      FloatV.NaNEq (FloatV.mul (bca.get x) (accf.get x)) (acc1.get x))
    (hacc2 : ∀ x : Idx [d.BM, d.BN],
      FloatV.NaNEq (FloatV.mul (bcb.get x) (acc1.get x)) (acc2.get x))
    (hcc : ∀ x : Idx [d.BM, d.BN], ConvRel P .f32 d.od (acc2.get x) (cc.get x))
    (hbca : ∀ (i : Fin d.BM) (j : Fin d.BN), pm d p * d.BM + i.1 < d.M → ∀ {saL : F32},
      init.read .f32 (d.sa.ptr (saIdx d (pm d p * d.BM + i.1) : Int)).addr = some saL →
      bca.get2 i j = saL)
    (hbcb : ∀ (i : Fin d.BM) (j : Fin d.BN), pn d p * d.BN + j.1 < d.N → ∀ {sbL : F32},
      init.read .f32 (d.sb.ptr (sbIdx d (pn d p * d.BN + j.1) : Int)).addr = some sbL →
      bcb.get2 i j = sbL) :
    ∀ (i : Idx [d.BM, d.BN]), pm d p * d.BM + i.1 < d.M → pn d p * d.BN + i.2.1 < d.N →
      LaneQ d init (pm d p * d.BM + i.1) (pn d p * d.BN + i.2.1) (cc.get i) := by
  intro i hiM hjN
  obtain ⟨i1, j1, u⟩ := i
  cases u
  intro va vb sa sb β hA hB hsaR hsaF hsbR hsbF hBias hRoom
  have hβ : β = 0 := by
    have h := hBias
    rw [BiasIs, ite_eq_right (by simpa using hb)] at h
    exact h
  subst hβ
  have hsum := acc_dotSum (d := d) (init := init) (p := p) (acc := acc)
    (va := va) (vb := vb) (i := i1) (j := j1) hA hB (hacc i1 j1 hiM hjN) (hkmulBK d hd)
  have h0 : tl.ItofRel .binary32 (dotSum va vb d.K) (accf.get2 i1 j1) := by
    have h := haccf ((i1, j1, ()) : Idx [d.BM, d.BN])
    rw [show acc.get ((i1, j1, ()) : Idx [d.BM, d.BN]) = acc.get2 i1 j1 from rfl] at h
    rw [hsum] at h
    exact h
  have h1 : FloatV.NaNEq (FloatV.mul sa (accf.get2 i1 j1)) (acc1.get2 i1 j1) := by
    have h := hacc1 ((i1, j1, ()) : Idx [d.BM, d.BN])
    have hbc := hbca i1 j1 hiM hsaR
    exact hbc ▸ h
  have h2 : FloatV.NaNEq (FloatV.mul sb (acc1.get2 i1 j1)) (acc2.get2 i1 j1) := by
    have h := hacc2 ((i1, j1, ()) : Idx [d.BM, d.BN])
    have hbc := hbcb i1 j1 hjN hsbR
    exact hbc ▸ h
  exact value_nobias P hd.od_ok (S := dotSum va vb d.K) (dotSum_bound hd va vb)
    h0 h1 h2 (hcc ((i1, j1, ()) : Idx [d.BM, d.BN])) hsaF hsbF hRoom

/-- With a bias, the epilogue chain leaves `LaneQ` in the lane. -/
theorem laneQ_bias (P : Platform) (d : Inst) (hd : d.ok) (init : Mem) (p : Nat)
    {acc : Tensor [d.BM, d.BN] I32} {accf acc1 acc2 acc3 : Tensor [d.BM, d.BN] F32}
    {bca bcb : Tensor [d.BM, d.BN] F32} {bv32 : Tensor [d.BN] F32}
    {cc : Tensor [d.BM, d.BN] d.od.Val}
    (hacc : ∀ (i : Fin d.BM) (j : Fin d.BN), pm d p * d.BM + i.1 < d.M →
      pn d p * d.BN + j.1 < d.N →
      (acc.get2 i j).toInt =
        ∑ k ∈ Finset.range (d.K / d.BK * d.BK),
          rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
    (haccf : ∀ x : Idx [d.BM, d.BN], tl.ItofRel .binary32 (acc.get x).toInt (accf.get x))
    (hacc1 : ∀ x : Idx [d.BM, d.BN],
      FloatV.NaNEq (FloatV.mul (bca.get x) (accf.get x)) (acc1.get x))
    (hacc2 : ∀ x : Idx [d.BM, d.BN],
      FloatV.NaNEq (FloatV.mul (bcb.get x) (acc1.get x)) (acc2.get x))
    (haddc : ∀ x : Idx [d.BM, d.BN],
      AddRel true (acc2.get x) ((bv32.row d.BM).get x) (acc3.get x))
    (hbvalue : ∀ (j : Fin d.BN), pn d p * d.BN + j.1 < d.N → ∀ (β : ℚ),
      BiasIs d init (pn d p * d.BN + j.1) β →
      (bv32.get1 j).isFinite = true ∧ (bv32.get1 j).toRat = β)
    (hcc : ∀ x : Idx [d.BM, d.BN], ConvRel P .f32 d.od (acc3.get x) (cc.get x))
    (hbca : ∀ (i : Fin d.BM) (j : Fin d.BN), pm d p * d.BM + i.1 < d.M → ∀ {saL : F32},
      init.read .f32 (d.sa.ptr (saIdx d (pm d p * d.BM + i.1) : Int)).addr = some saL →
      bca.get2 i j = saL)
    (hbcb : ∀ (i : Fin d.BM) (j : Fin d.BN), pn d p * d.BN + j.1 < d.N → ∀ {sbL : F32},
      init.read .f32 (d.sb.ptr (sbIdx d (pn d p * d.BN + j.1) : Int)).addr = some sbL →
      bcb.get2 i j = sbL) :
    ∀ (i : Idx [d.BM, d.BN]), pm d p * d.BM + i.1 < d.M → pn d p * d.BN + i.2.1 < d.N →
      LaneQ d init (pm d p * d.BM + i.1) (pn d p * d.BN + i.2.1) (cc.get i) := by
  intro i hiM hjN
  obtain ⟨i1, j1, u⟩ := i
  cases u
  intro va vb sa sb β hA hB hsaR hsaF hsbR hsbF hBias hRoom
  have hsum := acc_dotSum (d := d) (init := init) (p := p) (acc := acc)
    (va := va) (vb := vb) (i := i1) (j := j1) hA hB (hacc i1 j1 hiM hjN) (hkmulBK d hd)
  have h0 : tl.ItofRel .binary32 (dotSum va vb d.K) (accf.get2 i1 j1) := by
    have h := haccf ((i1, j1, ()) : Idx [d.BM, d.BN])
    rw [show acc.get ((i1, j1, ()) : Idx [d.BM, d.BN]) = acc.get2 i1 j1 from rfl] at h
    rw [hsum] at h
    exact h
  have h1 : FloatV.NaNEq (FloatV.mul sa (accf.get2 i1 j1)) (acc1.get2 i1 j1) := by
    have h := hacc1 ((i1, j1, ()) : Idx [d.BM, d.BN])
    have hbc := hbca i1 j1 hiM hsaR
    exact hbc ▸ h
  have h2 : FloatV.NaNEq (FloatV.mul sb (acc1.get2 i1 j1)) (acc2.get2 i1 j1) := by
    have h := hacc2 ((i1, j1, ()) : Idx [d.BM, d.BN])
    have hbc := hbcb i1 j1 hjN hsbR
    exact hbc ▸ h
  obtain ⟨hbfin, hbrat⟩ := hbvalue j1 hjN β hBias
  have h3 : AddRel true (acc2.get2 i1 j1) (bv32.get1 j1) (acc3.get2 i1 j1) :=
    haddc ((i1, j1, ()) : Idx [d.BM, d.BN])
  exact value_bias P hd.od_ok (S := dotSum va vb d.K) (dotSum_bound hd va vb)
    h0 h1 h2 hbfin hbrat h3 (hcc ((i1, j1, ()) : Idx [d.BM, d.BN])) hsaF hsbF hRoom


/-- A `bf16`/`f16` value converted back to `f32` has a `ConvRel` result. -/
theorem convRel_to_f32_exists {P : Platform} (od : DType) (hod : od = .bf16 ∨ od = .f16)
    (x : od.Val) : ∃ r : F32, ConvRel P od .f32 x r := by
  rcases hod with h | h
  · cases h
    exact ⟨_, convRel_bf16_f32 x⟩
  · cases h
    exact ⟨_, convRel_f16_f32 x⟩


/-- A converted `int32` literal keeps the natural number's value. -/
theorem convRel_i32_i64_ofNat {P : Platform} {n : Nat} (hn : n < 2 ^ 31)
    {r : Tensor [] I64} (h : ConvRel P .i32 .i64 (IntV.ofNat n : I32) (r.get ())) :
    r.item.toInt = (n : ℤ) := by
  show (r.get ()).toInt = _
  rw [convRel_i32_i64 h, IntV.I32.toInt_ofNat hn]

/-- A converted `tl.arange 0 n` is the index. -/
theorem convRel_i32_i64_arange {P : Platform} {n : Nat} (hn : n < 2 ^ 31) {r : Tensor [n] I64}
    (h : ∀ i : Idx [n], ConvRel P .i32 .i64 ((tl.arange 0 n).get i) (r.get i)) :
    ∀ i : Fin n, (r.get1 i).toInt = (i.1 : ℤ) := by
  intro i
  have h1 : (r.get (i, ())).toInt = ((tl.arange 0 n).get (i, ())).toInt :=
    convRel_i32_i64 (h (i, ()))
  show (r.get (i, ())).toInt = _
  rw [h1]
  show (IntV.ofInt (0 + (i.1 : ℤ)) : I32).toInt = _
  rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
  ring

/-- A converted `int32` product of natural values below `2^31` is the product. -/
theorem convRel_i32_i64_mul {P : Platform} {a b : I32} {x y : Nat} {r : Tensor [] I64}
    (ha : a.toInt = (x : ℤ)) (hb : b.toInt = (y : ℤ)) (hxy : x * y < 2 ^ 31)
    (h : ConvRel P .i32 .i64 (a * b) (r.get ())) :
    r.item.toInt = ((x * y : Nat) : ℤ) := by
  show (r.get ()).toInt = _
  rw [convRel_i32_i64 h, IntV.I32.mul_nat ha hb hxy]

/-- **One program is safe**: from `init`, program `p` of the grid runs without undefined behaviour or race inside its footprints and ends with `Qfp`. -/
theorem safe_program (P : Platform) (d : Inst) (hd : d.ok) (init : Mem) (hpre : Pre d init)
    (hdot : Device.DotI8 P)
    (htot : Device.DotTotal P
      { a := .i8, b := .i8, acc := .i32, precision := .tf32, maxImpreciseAcc := 0 })
    (c : Ctx) (p : Nat) (hc : c.pid 0 = p) (hp : p < grid d) :
    Safe ⟨P, true⟩ (Rfp d p) (Wfp d p) (fun _ m => Qfp d init p m) init ((model d).body c) := by
  simp only [model, scaled_mm_kernel, tl.program_id, hc]
  refine safe_idiv1 (x := IntV.ofNat d.N + (IntV.ofNat (d.BN - 1) : I32))
    (y := (IntV.ofNat d.BN : I32)) ?_ ?_ ?_
  · rw [toInt_Nadd hd]
    exact_mod_cast Nat.zero_le _
  · rw [toInt_BN hd]
    exact_mod_cast BN_pos hd
  · intro r1 h1
    refine safe_idiv1 (x := (IntV.ofNat p : I32)) (y := r1.item) ?_ ?_ ?_
    · rw [IntV.I32.toInt_ofNat (n := p) (by have hg : grid d < 2 ^ 31 := hd.grid_lt; omega)]
      exact_mod_cast Nat.zero_le p
    · rw [h1, toInt_Nadd hd, toInt_BN hd, ← Int.natCast_ediv]
      exact_mod_cast Nat.div_pos (by have := hd.N_pos; omega) (BN_pos hd)
    · intro r2 h2
      refine safe_irem1 (x := (IntV.ofNat p : I32)) (y := r1.item) ?_ ?_ ?_
      · rw [IntV.I32.toInt_ofNat (n := p) (by have hg : grid d < 2 ^ 31 := hd.grid_lt; omega)]
        exact_mod_cast Nat.zero_le p
      · rw [h1, toInt_Nadd hd, toInt_BN hd, ← Int.natCast_ediv]
        exact_mod_cast Nat.div_pos (by have := hd.N_pos; omega) (BN_pos hd)
      · intro r3 h3
        refine safe_convert (d := .i32) (d' := .i64)
          (x := Tensor.scalar (r2.item * (IntV.ofNat d.BM : I32))) ?_ ?_
        · intro i; exact ⟨_, rfl⟩
        · intro r4 h4
          refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BM) ?_ ?_
          · intro i; exact ⟨_, rfl⟩
          · intro r5 h5
            refine safe_convert (d := .i32) (d' := .i64)
              (x := Tensor.scalar (IntV.ofNat d.M : I32)) ?_ ?_
            · intro i; exact ⟨_, rfl⟩
            · intro r6 h6
              refine safe_convert (d := .i32) (d' := .i64)
                (x := Tensor.scalar (r3.item * (IntV.ofNat d.BN : I32))) ?_ ?_
              · intro i; exact ⟨_, rfl⟩
              · intro r7 h7
                refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BN) ?_ ?_
                · intro i; exact ⟨_, rfl⟩
                · intro r8 h8
                  refine safe_convert (d := .i32) (d' := .i64)
                    (x := Tensor.scalar (IntV.ofNat d.N : I32)) ?_ ?_
                  · intro i; exact ⟨_, rfl⟩
                  · intro r9 h9
                    refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BK) ?_ ?_
                    · intro i; exact ⟨_, rfl⟩
                    · intro r10 h10
                      refine safe_convert (d := .i32) (d' := .i64)
                        (x := Tensor.scalar (IntV.ofNat d.stride_am : I32)) ?_ ?_
                      · intro i; exact ⟨_, rfl⟩
                      · intro r11 h11
                        refine safe_convert (d := .i32) (d' := .i64)
                          (x := Tensor.scalar (IntV.ofNat d.stride_ak : I32)) ?_ ?_
                        · intro i; exact ⟨_, rfl⟩
                        · intro r12 h12
                          refine safe_convert (d := .i32) (d' := .i64)
                            (x := Tensor.scalar (IntV.ofNat d.stride_bk : I32)) ?_ ?_
                          · intro i; exact ⟨_, rfl⟩
                          · intro r13 h13
                            refine safe_convert (d := .i32) (d' := .i64)
                              (x := Tensor.scalar (IntV.ofNat d.stride_bn : I32)) ?_ ?_
                            · intro i; exact ⟨_, rfl⟩
                            · intro r14 h14
                              refine safe_idiv1
                                (x := IntV.ofNat d.K + (IntV.ofNat (d.BK - 1) : I32))
                                (y := (IntV.ofNat d.BK : I32)) ?_ ?_ ?_
                              · rw [toInt_Kadd hd]
                                exact_mod_cast Nat.zero_le _
                              · rw [toInt_BK hd]
                                exact_mod_cast BK_pos hd
                              · intro r15 h15
                                have hp31 : p < 2 ^ 31 := by
                                  have hg : grid d < 2 ^ 31 := hd.grid_lt
                                  omega
                                have hBM31 : d.BM < 2 ^ 31 := by
                                  have := BM_pos hd; have := BM_le256 hd; omega
                                have hBN31 : d.BN < 2 ^ 31 := by
                                  have := BN_pos hd; have := BN_le256 hd; omega
                                have hBK31 : d.BK < 2 ^ 31 := by
                                  have := BK_pos hd; have := BK_le hd; omega
                                have hM31 : d.M < 2 ^ 31 := by
                                  have := hd.M_le; have := BM_pos hd; omega
                                have hN31 : d.N < 2 ^ 31 := by
                                  have := hd.N_le; have := BN_pos hd; omega
                                have hr1 : r1.item.toInt
                                    = ((d.N + d.BN - 1 : Nat) : ℤ) / (d.BN : ℤ) := by
                                  rw [h1, toInt_Nadd hd, toInt_BN hd]
                                  have e : d.N + (d.BN - 1) = d.N + d.BN - 1 := by
                                    have := BN_pos hd; omega
                                  rw [e]
                                have h2v : r2.item.toInt = ((pm d p : Nat) : ℤ) := by
                                  rw [h2, IntV.I32.toInt_ofNat hp31, hr1,
                                    ← Int.natCast_ediv, ← Int.natCast_ediv]
                                have h3v : r3.item.toInt = ((pn d p : Nat) : ℤ) := by
                                  rw [h3, IntV.I32.toInt_ofNat hp31, hr1]
                                  simp only [pn, gridN, Int.natCast_mod, Int.natCast_ediv]
                                have h4v : r4.item.toInt = ((pm d p * d.BM : Nat) : ℤ) :=
                                  convRel_i32_i64_mul h2v (IntV.I32.toInt_ofNat hBM31)
                                    (pmBM_lt31 hd hp) (h4 ())
                                have h5v : ∀ i : Fin d.BM, (r5.get1 i).toInt = (i.1 : ℤ) :=
                                  convRel_i32_i64_arange hBM31 h5
                                have h6v : r6.item.toInt = ((d.M : Nat) : ℤ) :=
                                  convRel_i32_i64_ofNat hM31 (h6 ())
                                have h7v : r7.item.toInt = ((pn d p * d.BN : Nat) : ℤ) :=
                                  convRel_i32_i64_mul h3v (IntV.I32.toInt_ofNat hBN31)
                                    (pnBN_lt31 hd p) (h7 ())
                                have h8v : ∀ j : Fin d.BN, (r8.get1 j).toInt = (j.1 : ℤ) :=
                                  convRel_i32_i64_arange hBN31 h8
                                have h9v : r9.item.toInt = ((d.N : Nat) : ℤ) :=
                                  convRel_i32_i64_ofNat hN31 (h9 ())
                                have h10v : ∀ kk : Fin d.BK, (r10.get1 kk).toInt = (kk.1 : ℤ) :=
                                  convRel_i32_i64_arange hBK31 h10
                                have h11v : r11.item.toInt = (d.stride_am : ℤ) :=
                                  convRel_i32_i64_ofNat hd.strides.1 (h11 ())
                                have h12v : r12.item.toInt = (d.stride_ak : ℤ) :=
                                  convRel_i32_i64_ofNat hd.strides.2.1 (h12 ())
                                have h13v : r13.item.toInt = (d.stride_bk : ℤ) :=
                                  convRel_i32_i64_ofNat hd.strides.2.2.1 (h13 ())
                                have h14v : r14.item.toInt = (d.stride_bn : ℤ) :=
                                  convRel_i32_i64_ofNat hd.strides.2.2.2.1 (h14 ())
                                have h15v : r15.item.toInt = ((d.K / d.BK : Nat) : ℤ) := by
                                  rw [h15, toInt_Kadd hd, toInt_BK hd, ← Int.natCast_ediv,
                                    divK_eq hd]
                                refine safe_forRange_ro (lo := (IntV.ofInt 0 : I32)) (hi := r15.item)
                                  (step := (IntV.ofInt 1 : I32))
                                  (I := fun t st m => m = init ∧ Inv d init p t st) ?_ ?_ ?_ ?_
                                · have hR : IntV.maxInt true 32 = 2 ^ 31 - 1 := rfl
                                  unfold tl.RangeOk
                                  refine ⟨?_, ?_, ?_, ?_⟩
                                  · rw [IntV.I32.toInt_ofInt (by norm_num)]
                                    norm_num
                                  · rw [h15, toInt_Kadd hd, toInt_BK hd,
                                      IntV.I32.toInt_ofInt (by norm_num), hR]
                                    have hle := divK_le hd
                                    have hK := hd.K_lt
                                    omega
                                  · rw [IntV.I32.toInt_ofInt (by norm_num), hR]
                                    norm_num
                                  · rw [IntV.I32.toInt_ofInt (by norm_num), hR]
                                    norm_num
                                · exact ⟨rfl, inv_zero d hd init p hp r4 r11 r12 r13 r14 r5 r7 r8 r10
                                    h4v h5v h7v h8v h10v h11v h12v h13v h14v⟩
                                · intro t ht st m1 hI
                                  obtain ⟨hm, hInv⟩ := hI
                                  rw [hm]
                                  obtain ⟨ok, ap, bp, acc⟩ := st
                                  have htK : t * d.BK + d.BK ≤ d.K := by
                                    have ht' : t < d.K / d.BK := by
                                      have e0 : (IntV.ofInt 0 : I32).toInt = 0 :=
                                        IntV.I32.toInt_ofInt (by norm_num)
                                      have e1 : (IntV.ofInt 1 : I32).toInt = 1 :=
                                        IntV.I32.toInt_ofInt (by norm_num)
                                      have hstep : tl.tripCount (IntV.ofInt 0 : I32).toInt
                                          r15.item.toInt (IntV.ofInt 1 : I32).toInt = d.K / d.BK := by
                                        unfold tl.tripCount
                                        simp only [e0, e1, h15v]
                                        have hnum : ((d.K / d.BK : Nat) : ℤ) - 0 + 1 - 1
                                            = ((d.K / d.BK : Nat) : ℤ) := by ring
                                        rw [hnum, Int.ediv_one]
                                        exact Int.toNat_natCast _
                                      rw [hstep] at ht
                                      exact ht
                                    have h1 : (t + 1) * d.BK ≤ d.K := by
                                      have h2 : t + 1 ≤ d.K / d.BK := by omega
                                      calc (t + 1) * d.BK ≤ (d.K / d.BK) * d.BK :=
                                            Nat.mul_le_mul_right _ h2
                                        _ ≤ d.K := Nat.div_mul_le_self _ _
                                    calc t * d.BK + d.BK = (t + 1) * d.BK := by ring
                                      _ ≤ d.K := h1
                                  refine loop_safe P d hd init hpre hdot htot p t
                                    ((r4.item + r5).ltS r6.item)
                                    ((r7.item + r8).ltS r9.item) ?_ ?_ ok ap bp acc htK hInv
                                  · exact mask_am d hd p hp r4 r6 r5 h4v h5v h6v
                                  · exact mask_bn d hd p r7 r9 r8 h7v h8v h9v
                                · intro st m1 hI
                                  obtain ⟨hm, hInv⟩ := hI
                                  rw [hm]
                                  obtain ⟨ok, ap, bp, acc⟩ := st
                                  have htK : tl.tripCount (IntV.ofInt 0 : I32).toInt r15.item.toInt
                                      (IntV.ofInt 1 : I32).toInt = d.K / d.BK := by
                                    have e0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)
                                    have e1 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
                                    unfold tl.tripCount
                                    simp only [e0, e1, h15v]
                                    have hnum : ((d.K / d.BK : Nat) : ℤ) - 0 + 1 - 1 = ((d.K / d.BK : Nat) : ℤ) := by ring
                                    rw [hnum, Int.ediv_one]
                                    exact Int.toNat_natCast _
                                  rw [htK] at hInv
                                  refine safe_load_buf (b := d.sa)
                                    (g := fun i : Idx [BSA d, 1] => saIdx d (saOff d p (i.1 : Nat)))
                                    (en := fun i : Idx [BSA d, 1] => decide (saOff d p (i.1 : Nat) < d.M)) ?_ ?_ ?_ ?_ ?_
                                  · refine Buf.addrs_of_get d.sa ?_ ?_
                                    · intro i
                                      exact h_mask_sa_full d hd p hp r2 h2v i
                                    · intro i hi
                                      exact h_sa_ptr_get d hd p hp r2 h2v i
                                  · intro i hi
                                    simp only [decide_eq_true_eq] at hi
                                    exact Nat.lt_of_lt_of_le (saTile_sub hd p ⟨i.1.1, i.1.2, hi, rfl⟩) hd.sa_len
                                  · exact hpre.2.2.1
                                  · exact h_sa_elems_sub d p
                                  · intro sa hsa hsaO
                                    rw [dite_eq_left (h_sa_bcast_ok (d := d))]
                                    simp only [Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
                                    refine safe_itof (f := .binary32) ?_ ?_
                                    · exact Or.inr (Or.inr (Or.inl rfl))
                                    · intro accf haccf
                                      refine safe_mul ?_
                                      intro acc1 hacc1
                                      refine safe_load_buf (b := d.sb)
                                        (g := fun i : Idx [BSB d, 1] => sbIdx d (sbOff d p (i.1 : Nat)))
                                        (en := fun i : Idx [BSB d, 1] => decide (sbOff d p (i.1 : Nat) < d.N)) ?_ ?_ ?_ ?_ ?_
                                      · refine Buf.addrs_of_get d.sb ?_ ?_
                                        · intro i
                                          exact h_mask_sb_full d hd p r3 h3v i
                                        · intro i hi
                                          exact h_sb_ptr_get d hd p r3 h3v i
                                      · intro i hi
                                        simp only [decide_eq_true_eq] at hi
                                        exact Nat.lt_of_lt_of_le (sbTile_sub hd p ⟨i.1.1, i.1.2, hi, rfl⟩) hd.sb_len
                                      · exact hpre.2.2.2.1
                                      · exact h_sb_elems_sub d p
                                      · intro sb hsb hsbO
                                        rw [dite_eq_left (h_sb_bcast_ok (d := d))]
                                        refine safe_mul ?_
                                        intro acc2 hacc2
                                        by_cases hb : d.hasBias = true
                                        · rw [ite_eq_left hb]
                                          dsimp only
                                          rw [ite_eq_left (by simpa using hd.bias_nz hb)]
                                          simp only [Prog.bind_assoc]
                                          refine safe_convert ?_ ?_
                                          · intro i
                                            exact ⟨_, rfl⟩
                                          · intro r16 h16
                                            have h16v : r16.item.toInt = ((d.N : Nat) : ℤ) :=
                                              convRel_i32_i64_ofNat hN31 (h16 ())
                                            refine safe_load_buf (b := d.bias)
                                              (g := fun i : Idx [d.BN] => pn d p * d.BN + (i.1 : Nat))
                                              (en := fun i : Idx [d.BN] => decide (pn d p * d.BN + (i.1 : Nat) < d.N))
                                              ?_ ?_ ?_ ?_ ?_
                                            · refine Buf.addrs_of_get d.bias ?_ ?_
                                              · intro i
                                                exact mask_bn d hd p r7 r16 r8 h7v h8v h16v i
                                              · intro i hi
                                                exact h_bias_ptr_get d hd p r7 r8 h7v h8v i
                                            · intro i hi
                                              simp only [decide_eq_true_eq] at hi
                                              exact Nat.lt_of_lt_of_le hi (hd.bias_len hb)
                                            · exact hpre.2.2.2.2.2 hb
                                            · exact h_bias_elems_sub d p hb
                                            · intro bv hbv hbvO
                                              refine safe_convert ?_ ?_
                                              · intro i
                                                exact convRel_to_f32_exists d.od hd.od_ok (bv.get i)
                                              · intro bv32 hbv32
                                                refine safe_addc ?_
                                                intro acc3 hacc3
                                                refine tail_safe_cov P d hd init hpre p hp r2.item r3.item acc3 ?_ ?_ ?_
                                                · simpa using h2v
                                                · simpa using h3v
                                                · intro cc hcc i hiM hjN
                                                  obtain ⟨i1, j1, u⟩ := i
                                                  cases u
                                                  exact laneQ_bias P d hd init p
                                                    (fun i j hiM hjN => hInv.acc i j hiM hjN)
                                                    haccf hacc1 hacc2 hacc3
                                                    (by
                                                      intro j hjN β hB
                                                      rw [BiasIs, ite_eq_left hb] at hB
                                                      obtain ⟨x, hxread, hxv⟩ := hB
                                                      have hstride : d.stride_bias = 1 := hd.scales_contig.2.2 hb
                                                      have h1 : init.read d.od (d.bias.ptr
                                                          (((pn d p * d.BN + (j.1 : Nat)) * d.stride_bias : Nat) : Int)).addr
                                                          = some (bv.get1 j) := by
                                                        have hread := hbv ((j, ()) : Idx [d.BN]) (by
                                                          simp only [decide_eq_true_eq]
                                                          exact hjN)
                                                        rw [hstride, Nat.mul_one]
                                                        exact hread
                                                      have h2 : init.read d.od (d.bias.ptr
                                                          (((pn d p * d.BN + (j.1 : Nat)) * d.stride_bias : Nat) : Int)).addr
                                                          = some x := hxread
                                                      have hxj : x = bv.get1 j := Option.some.inj (h2.symm.trans h1)
                                                      rw [hxj] at hxv
                                                      exact bias_f32_toRat hd.od_ok hxv
                                                        (hbv32 ((j, ()) : Idx [d.BN])))
                                                    hcc
                                                    (fun i j hiM saL hL =>
                                                      h_sa_bcast_get hd (h_sa_bcast_ok (d := d)) hsa i j hiM hL)
                                                    (fun i j hjN sbL hL =>
                                                      h_sb_bcast_get hd (h_sb_bcast_ok (d := d)) hsb i j hjN hL)
                                                    ((i1, j1, ()) : Idx [d.BM, d.BN]) hiM hjN
                                        · have hbf : d.hasBias = false := by
                                            cases hbb : d.hasBias
                                            · rfl
                                            · exact absurd hbb hb
                                          rw [ite_eq_right hb]
                                          refine tail_safe_cov P d hd init hpre p hp r2.item r3.item acc2 ?_ ?_ ?_
                                          · simpa using h2v
                                          · simpa using h3v
                                          · intro cc hcc i hiM hjN
                                            obtain ⟨i1, j1, u⟩ := i
                                            cases u
                                            exact laneQ_nobias P d hd init p hbf
                                              (fun i j hiM hjN => hInv.acc i j hiM hjN)
                                              haccf hacc1 hacc2 hcc
                                              (fun i j hiM saL hL =>
                                                h_sa_bcast_get hd (h_sa_bcast_ok (d := d)) hsa i j hiM hL)
                                              (fun i j hjN sbL hL =>
                                                h_sb_bcast_get hd (h_sb_bcast_ok (d := d)) hsb i j hjN hL)
                                              ((i1, j1, ()) : Idx [d.BM, d.BN]) hiM hjN

end Claims.ScaledMmInt8
