import Proof.Tile.PT
import Proof.Tile.Chunk
import Proof.Tile.Acc2
import Proof.Disj
import Proof.Post

/-!
# The kernel body of `matmul_e5m2`

`tile_safe` proves the `Safe` judgement for one program of the launch: the scalar chain that
computes the program id's tile (`pid_m`, `pid_n`), the `tl.forRange` loop over the `K` panels
(the invariant `Inv` records, per lane, the partial dot-product value and the two pointer
tensors), and the final conversion and store of the accumulator.
-/

namespace Claims.MatmulE5m2

open TVA
open scoped BigOperators


/-! ## Small facts -/

/-- The item of a pointwise integer minimum. -/
theorem imin_item (a b : I32) :
    (tl.iminimum (Tensor.scalar a) (Tensor.scalar b)).item = IntV.min a b := rfl

/-- `IntV.le` on `int32` values, from their mathematical values. -/
theorem le_of_le {a b : I32} (h : a.toInt ≤ b.toInt) : a.le b = true := by
  have h' : a.bits.toInt ≤ b.bits.toInt := by simpa [IntV.toInt] using h
  simp only [IntV.le, reduceIte, BitVec.sle_eq_decide, decide_eq_true_eq]
  exact h'

/-- `IntV.lt` on `int32` values, from their mathematical values. -/
theorem lt_of_lt {a b : I32} (h : a.toInt < b.toInt) : a.lt b = true := by
  have h' : a.bits.toInt < b.bits.toInt := by simpa [IntV.toInt] using h
  simp only [IntV.lt, reduceIte, BitVec.slt_eq_decide, decide_eq_true_eq]
  exact h'

/-- `IntV.le` with the left operand zero. -/
theorem zero_le_of {b : I32} (h : 0 ≤ b.toInt) : (IntV.ofInt 0 : I32).le b = true := by
  apply le_of_le
  have h0 : (IntV.ofInt (0 : ℤ) : I32).toInt = 0 := toInt_ofInt' (by norm_num) (by norm_num)
  rw [h0]; exact h

/-- `IntV.lt` with the left operand zero. -/
theorem zero_lt_of {b : I32} (h : 0 < b.toInt) : (IntV.ofInt 0 : I32).lt b = true := by
  apply lt_of_lt
  have h0 : (IntV.ofInt (0 : ℤ) : I32).toInt = 0 := toInt_ofInt' (by norm_num) (by norm_num)
  rw [h0]; exact h

/-- The value of an `f32` zero. -/
theorem zero_f32_val? : DType.val? .f32 (DType.f32.ofBits 0) = some 0 := by
  have hfin : (DType.f32.ofBits 0 : F32).isFinite = true := by
    show (FloatV.ofPat 0 : F32).isFinite = true
    rw [FloatV.isFinite_ofPat (f := Format.binary32)
         (by decide : (0 : Nat) < 2 ^ Format.binary32.width)]
    exact Format.isFinite_zero (by decide)
  have hrat : (DType.f32.ofBits 0 : F32).toRat = 0 := by
    show (FloatV.ofPat 0 : F32).toRat = 0
    rw [FloatV.toRat_ofPat (f := Format.binary32)
         (by decide : (0 : Nat) < 2 ^ Format.binary32.width)]
    exact Format.toRat_of_isZero (f := Format.binary32)
      Format.isZero_zero
  show (if (DType.f32.ofBits 0 : F32).isFinite = true then
      some (DType.f32.ofBits 0 : F32).toRat else none) = some 0
  simp only [hfin, ite_true, hrat]

/-- The lane value of a zero tensor. -/
theorem zeros_get_f32 (x : Fin 128) (y : Fin 256) :
    (tl.zeros [128, 256] .f32).get2 x y = (DType.f32.ofBits 0 : F32) := rfl

/-- The value of a zero `f32` lane of the accumulator tensor. -/
theorem zeros_val (x : Fin 128) (y : Fin 256) :
    DType.val? .f32 ((tl.zeros [128, 256] .f32).get2 x y) = some 0 := by
  rw [zeros_get_f32]; exact zero_f32_val?

/-- `steps` as an `int32` value. -/
theorem stepsVal (d : Inst) (hd : d.ok) :
    (IntV.ofInt (((IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32)).toInt /
      (IntV.ofNat 64 : I32).toInt) : I32) = IntV.ofInt (steps d.K) := by
  have hK_lt : d.K < 2 ^ 31 := by have := hd.K_dim.2.2; omega
  have hKv : (IntV.ofNat d.K : I32).toInt = (d.K : ℤ) := toInt_ofNat' hK_lt
  have h63 : (IntV.ofNat 63 : I32).toInt = (63 : ℤ) := toInt_ofNat' (by norm_num)
  have hsum : ((IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32)).toInt = (d.K : ℤ) + 63 :=
    add_toInt hKv h63 (by positivity) (by have := hd.K_dim.2.2; omega)
  have h64 : (IntV.ofNat 64 : I32).toInt = (64 : ℤ) := toInt_ofNat' (by norm_num)
  have hdiv : ((d.K : ℤ) + 63) / 64 = ((steps d.K : ℕ) : ℤ) := by
    have h1 : ((d.K : ℤ) + 63) = ((d.K + 63 : ℕ) : ℤ) := by push_cast; ring
    rw [h1, show (64 : ℤ) = ((64 : ℕ) : ℤ) from by norm_num, ← Int.natCast_ediv]
    rfl
  rw [hsum, h64, hdiv]

/-- The number of trips fits in `int32` and is positive. -/
theorem steps_lt (d : Inst) (hd : d.ok) : steps d.K < 2 ^ 31 := by
  have h1 : steps d.K ≤ d.K + 63 := by
    unfold steps
    exact Nat.div_le_self _ _
  have h2 : d.K + 63 < 2 ^ 31 := by
    have := hd.K_le
    omega
  omega

/-- The bounds of the `K`-loop satisfy `tl.RangeOk`. -/
theorem rangeOk_steps (d : Inst) (hd : d.ok) :
    tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofInt (steps d.K) : I32) (IntV.ofInt 1 : I32) := by
  have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
    norm_num [IntV.maxInt]
  have h0 : (IntV.ofInt (0 : ℤ) : I32).toInt = 0 := toInt_ofInt' (by norm_num) (by norm_num)
  have h1 : (IntV.ofInt ((steps d.K : ℕ) : ℤ) : I32).toInt = ((steps d.K : ℕ) : ℤ) :=
    toInt_ofInt' (by positivity) (by have := steps_lt d hd; omega)
  have h2 : (IntV.ofInt (1 : ℤ) : I32).toInt = 1 := toInt_ofInt' (by norm_num) (by norm_num)
  unfold tl.RangeOk
  rw [h0, h1, h2, hmax]
  have := steps_lt d hd
  omega

/-- The loop's trip count is `steps d.K`. -/
theorem tripCount_steps (d : Inst) (hd : d.ok) :
    tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofInt (steps d.K) : I32).toInt
      (IntV.ofInt 1 : I32).toInt = steps d.K := by
  have h0 : (IntV.ofInt (0 : ℤ) : I32).toInt = 0 := toInt_ofInt' (by norm_num) (by norm_num)
  have h1 : (IntV.ofInt ((steps d.K : ℕ) : ℤ) : I32).toInt = ((steps d.K : ℕ) : ℤ) :=
    toInt_ofInt' (by positivity) (by have := steps_lt d hd; omega)
  have h2 : (IntV.ofInt (1 : ℤ) : I32).toInt = 1 := toInt_ofInt' (by norm_num) (by norm_num)
  unfold tl.tripCount
  rw [h0, h1, h2]
  simp only [Int.sub_zero, Int.add_sub_cancel, Int.ediv_one, Int.toNat_natCast]

/-- The reduction length is a multiple of `64`. -/
theorem K64 (d : Inst) (hd : d.ok) : 64 ∣ d.K := by
  have h128 : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.2.1
  exact (by norm_num : 64 ∣ 128).trans h128

/-! ## The loop invariant -/

/-- The lane's row and column of memory are all finite values: the hypothesis under which the
partial-value sequence of a lane exists (a lane whose memory holds a `NaN` or an infinity has no
`toRat`, so the claim's error bound says nothing about it, and `OutSpec` is vacuous there). -/
def FinLane (d : Inst) (init : Mem) (p x y : Nat) : Prop :=
  (∀ k, k < d.K → (vaOf d init p x k).isFinite = true) ∧
    ∀ k, k < d.K → (vbOf d init p y k).isFinite = true

/-- The state of the `K`-loop after `t` trips: the two panel pointers and the accumulator, whose
lane `(x, y)` holds the partial dot product `S (2·t)` of the lane's row and column, for every lane
whose row and column values are finite. -/
def Inv (d : Inst) (init : Mem) (p t : Nat)
    (st : Tensor [128, 64] (Ptr .f8e5) × Tensor [64, 256] (Ptr .f8e5) ×
      Tensor [128, 256] F32) (m : Mem) : Prop :=
  st.1 = aPtrT d p t ∧ st.2.1 = bPtrT d p t ∧ m = init ∧
    ∀ x y, FinLane d init p x.val y.val → ∃ S : Nat → ℚ, S 0 = 0 ∧
      Steps (prodV (vaOf d init p x.val) (vbOf d init p y.val) d.K)
        (liftV (vaOf d init p x.val) (vbOf d init p y.val) d.K) S (2 * t) ∧
      DType.val? .f32 (st.2.2.get2 x y) = some (S (2 * t))

/-- The value of `group_size_m` is positive. -/
theorem gsz_toInt_pos (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    0 < (IntV.min ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d p) * (IntV.ofNat 8 : I32))
      (IntV.ofNat 8 : I32)).toInt := by
  rw [gsm_val d hd hp, toInt_ofInt'
    (by positivity)
    (natCast_lt' (lt_of_le_of_lt (gsz_le_eight d p) (by norm_num)))]
  have h := gsz_pos hp
  omega

/-- The value of `group_size_m` as an `int32` is positive. -/
theorem gszVal_toInt_pos (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    0 < (IntV.ofInt (gsz d p) : I32).toInt := by
  rw [← gsm_val d hd hp]
  exact gsz_toInt_pos d hd hp

/-- The value of `pid % num_pid_in_group` is nonnegative. -/
theorem rmd_toInt_nonneg (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    0 ≤ (IntV.ofInt (rmd d p) : I32).toInt := by
  rw [toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (rmd_lt hp) (npig_lt hd)))]
  omega

/-- `pid_m` is nonnegative. -/
theorem pidm_toInt_nonneg (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    0 ≤ (IntV.ofInt (pidm d p) : I32).toInt := by
  rw [toInt_ofInt' (by positivity) (natCast_lt' (pidm_lim hd hp))]
  omega

/-- `pid_n` is nonnegative. -/
theorem pidn_toInt_nonneg (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) :
    0 ≤ (IntV.ofInt (pidn d p) : I32).toInt := by
  rw [toInt_ofInt' (by positivity) (natCast_lt' (pidn_lim hd hp))]
  omega

/-- A positive stride as an `int32` value. -/
theorem stride_toInt_pos {n : Nat} (h0 : 0 < n) (h : n < 2 ^ 31) :
    0 < (IntV.ofNat n : I32).toInt := by
  rw [toInt_ofNat' h]
  exact_mod_cast h0

/-- `pid_m · 128 + x` fits in `int32` and is nonnegative, before the reduction. -/
theorem offsAM_ok (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [128]) :
    0 ≤ ((offsAM d p).get i).toInt := by
  have h : (offsAM d p).get i = IntV.ofInt (ocm d p i.1.val) := by
    rw [show (offsAM d p).get i = (offsAM d p).get1 i.1 from rfl]
    exact offsAM_raw_get d hd hp i.1
  rw [h, toInt_ofInt' (by positivity) (natCast_lt' (ocm_lt hd hp i.1.isLt))]
  omega

/-- `pid_n · 256 + y` fits in `int32` and is nonnegative, before the reduction. -/
theorem offsBN_ok (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [256]) :
    0 ≤ ((offsBN d p).get i).toInt := by
  have h : (offsBN d p).get i = IntV.ofInt (ocn d p i.1.val) := by
    rw [show (offsBN d p).get i = (offsBN d p).get1 i.1 from rfl]
    exact offsBN_raw_get d hd hp i.1
  rw [h, toInt_ofInt' (by positivity) (natCast_lt' (ocn_lt hd hp i.1.isLt))]
  omega

/-- The modulus `M` is positive as an `int32` value. -/
theorem M_toInt_pos (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [128]) :
    0 < ((bc (s := [128]) (IntV.ofNat d.M) : Tensor [128] I32).get i).toInt := by
  rw [bc_get]
  exact stride_toInt_pos (M_pos hp) (M_lt hd)

/-- The modulus `N` is positive as an `int32` value. -/
theorem N_toInt_pos (d : Inst) (hd : d.ok) (i : Idx [256]) :
    0 < ((bc (s := [256]) (IntV.ofNat d.N) : Tensor [256] I32).get i).toInt := by
  rw [bc_get]
  exact stride_toInt_pos (N_pos hd) (N_lt hd)

/-- The `64` columns of trip `t` are below `K`. -/
theorem seg_lt (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K) (e : Nat) (he : e < 64) :
    64 * t + e < d.K := by
  rw [steps_eq (K64 d hd)] at ht
  have h2 : (t + 1) * 64 ≤ d.K := (Nat.le_div_iff_mul_le (by norm_num)).mp ht
  omega

/-- The `a` panel of one trip is part of `a`. -/
theorem aImg_sub_seg {d : Inst} (hd : d.ok) {p t : Nat} (hp : p < grid d) (init : Mem)
    (ht : t < steps d.K) :
    d.a.elems (Buf.img (fun i : Idx [128, 64] => aOff d p i.1 (64 * t + i.2.1))
      (fun _ => true)) ⊆ R d init p := by
  refine (Buf.elems_mono d.a ?_).trans (Set.subset_union_left)
  rintro e ⟨i, -, rfl⟩
  exact ⟨om d p i.1, 64 * t + i.2.1, om_lt hp i.1, seg_lt d hd ht i.2.1 i.2.1.isLt, rfl⟩

/-- The `b` panel of one trip is part of `b`. -/
theorem bImg_sub_seg {d : Inst} (hd : d.ok) {p t : Nat} (init : Mem)
    (ht : t < steps d.K) :
    d.b.elems (Buf.img (fun i : Idx [64, 256] => bOff d p i.2.1 (64 * t + i.1))
      (fun _ => true)) ⊆ R d init p := by
  refine (Buf.elems_mono d.b ?_).trans (Set.subset_union_right)
  rintro e ⟨i, -, rfl⟩
  exact ⟨64 * t + i.1, on d p i.2.1, seg_lt d hd ht i.1 i.1.isLt, on_lt hd p i.2.1, rfl⟩

/-! ## The kernel body -/

set_option maxHeartbeats 1000000 in
/-- **One program is safe.** Program `i` of the launch runs the kernel body without a trap, reading
only `R`, writing only `W`, and leaving `Q`. -/
theorem tile_safe {P : Platform} (d : Inst) (hd : d.ok) (i : Fin (launch d).size)
    (init : Mem) (hpre : Pre d init)
    (hdot : Device.DotE5m2ChunkLoss P)
    (hdotTotal : Device.DotTotal P ⟨.f8e5, .f8e5, .f32, .tf32, 2 ^ 30⟩)
    (hcvtTotal : Device.CvtTotal P .binary32 .e5m2 .rn) :
    Safe ((launch d).env P) (R d init i.val) (W d init i.val) (fun _ m => Q d init i.val m)
      init ((model d).body ((launch d).ctx i)) := by
  have hp : i.val < grid d := by
    have h := launch_size d
    have := i.isLt
    omega
  simp only [model, matmul_kernel, tl.program_id]
  rw [launch_pid]
  simp only [Nat.reduceSub]
  refine safe_idiv1' (x := IntV.ofNat d.M + IntV.ofNat 127) (y := IntV.ofNat 128) ?_ ?_ ?_
  ·
    rw [add_toInt (toInt_ofNat' (by have := hd.M_le; omega)) (toInt_ofNat' (by norm_num)) (by positivity) (by have := hd.M_le; omega)]
    omega
  ·
    rw [toInt_ofNat' (by norm_num)]
    norm_num
  ·
    rw [numPidM_val d hd]
    refine safe_idiv1' (x := IntV.ofNat d.N + IntV.ofNat 255) (y := IntV.ofNat 256) ?_ ?_ ?_
    ·
      rw [add_toInt (toInt_ofNat' (by have := hd.N_le; omega)) (toInt_ofNat' (by norm_num)) (by positivity) (by have := hd.N_le; omega)]
      omega
    ·
      rw [toInt_ofNat' (by norm_num)]
      norm_num
    ·
      simp only [Tensor.item_scalar]
      rw [numPidN_val d hd]
      refine safe_idiv1' (x := IntV.ofNat i.val) (y := IntV.ofNat 8 * IntV.ofInt (tN d)) ?_ ?_ ?_
      ·
        rw [toInt_ofNat' (lt_trans hp (grid_lt hd))]
        omega
      ·
        rw [npig_val d hd]
        have h := tN_pos_of_lt hp
        omega
      ·
        simp only [Tensor.item_scalar]
        rw [groupId_val d hd hp]
        refine safe_irem1' (x := IntV.ofNat i.val) (y := IntV.ofNat 8 * IntV.ofInt (tN d)) ?_ ?_ ?_
        ·
          rw [toInt_ofNat' (lt_trans hp (grid_lt hd))]
          omega
        ·
          rw [npig_val d hd]
          have h := tN_pos_of_lt hp
          omega
        ·
          simp only [Tensor.item_scalar]
          rw [pidRmd_val d hd hp]
          rw [imin_item]
          refine safe_irem1' (x := IntV.ofInt (rmd d i.val)) (y := IntV.min ((IntV.ofInt (tM d) : I32) - IntV.ofInt (grp d i.val) * (IntV.ofNat 8 : I32)) (IntV.ofNat 8 : I32)) ?_ ?_ ?_
          ·
            exact rmd_toInt_nonneg d hd hp
          ·
            exact gsz_toInt_pos d hd hp
          ·
            simp only [Tensor.item_scalar]
            rw [gsm_val d hd hp, pidm_val d hd hp]
            refine safe_irem1' (x := IntV.ofNat i.val) (y := IntV.ofNat 8 * IntV.ofInt (tN d)) ?_ ?_ ?_
            ·
              rw [toInt_ofNat' (lt_trans hp (grid_lt hd))]
              omega
            ·
              rw [npig_val d hd]
              have h := tN_pos_of_lt hp
              omega
            ·
              simp only [Tensor.item_scalar]
              rw [pidRmd_val d hd hp]
              refine safe_idiv1' (x := IntV.ofInt (rmd d i.val)) (y := (IntV.ofInt (gsz d i.val) : I32)) ?_ ?_ ?_
              ·
                exact rmd_toInt_nonneg d hd hp
              ·
                exact gszVal_toInt_pos d hd hp
              ·
                simp only [Tensor.item_scalar]
                rw [rmdMod_val d hd hp]
                -- the assumptions
                refine safe_assume (c := (IntV.ofInt 0).le (IntV.ofInt (pidm d i.val))) ?_ ?_
                · exact zero_le_of (pidm_toInt_nonneg d hd hp)
                refine safe_assume (c := (IntV.ofInt 0).le (IntV.ofInt (pidn d i.val))) ?_ ?_
                · exact zero_le_of (pidn_toInt_nonneg d hd hp)
                refine safe_assume (c := (IntV.ofInt 0).lt (IntV.ofNat d.stride_am)) ?_ ?_
                · exact zero_lt_of (stride_toInt_pos hd.am.1 hd.am.2)
                refine safe_assume (c := (IntV.ofInt 0).lt (IntV.ofNat d.stride_ak)) ?_ ?_
                · exact zero_lt_of (stride_toInt_pos hd.ak.1 hd.ak.2)
                refine safe_assume (c := (IntV.ofInt 0).lt (IntV.ofNat d.stride_bn)) ?_ ?_
                · exact zero_lt_of (stride_toInt_pos hd.bn.1 hd.bn.2)
                refine safe_assume (c := (IntV.ofInt 0).lt (IntV.ofNat d.stride_bk)) ?_ ?_
                · exact zero_lt_of (stride_toInt_pos hd.bk.1 hd.bk.2)
                refine safe_assume (c := (IntV.ofInt 0).lt (IntV.ofNat d.stride_cm)) ?_ ?_
                · exact zero_lt_of (stride_toInt_pos hd.cm.1 hd.cm.2)
                refine safe_assume (c := (IntV.ofInt 0).lt (IntV.ofNat d.stride_cn)) ?_ ?_
                · exact zero_lt_of (stride_toInt_pos hd.cn.1 hd.cn.2)
                -- `offs_am`
                refine safe_irem' (x := offsAM d i.val) (y := (bc (s := [128]) (IntV.ofNat d.M) : Tensor [128] I32)) ?_ ?_
                · intro j
                  exact ⟨offsAM_ok d hd hp j, M_toInt_pos d hd hp j⟩
                -- `offs_bn`
                refine safe_irem' (x := offsBN d i.val) (y := (bc (s := [256]) (IntV.ofNat d.N) : Tensor [256] I32)) ?_ ?_
                · intro j
                  exact ⟨offsBN_ok d hd hp j, N_toInt_pos d hd j⟩
                -- the number of trips
                refine safe_idiv1' (x := IntV.ofNat d.K + IntV.ofNat 63) (y := IntV.ofNat 64) ?_ ?_ ?_
                · rw [add_toInt (toInt_ofNat' (by have := hd.K_dim.2.2; omega)) (toInt_ofNat' (by norm_num)) (by positivity) (by have := hd.K_dim.2.2; omega)]
                  omega
                · rw [toInt_ofNat' (by norm_num)]
                  norm_num
                · simp only [Tensor.item_scalar]
                  rw [stepsVal d hd]
                  refine (safe_forRange (lo := (IntV.ofInt 0 : I32)) (hi := (IntV.ofInt (steps d.K) : I32))
                    (step := (IntV.ofInt 1 : I32)) (I := Inv d init i.val) (Wt := fun _ => ∅)
                    (Wk := W d init i.val) (hrange := rangeOk_steps d hd) ?_ ?_ ?_ ?_ ?_).mono
                    (fun _ _ h => h) ?_
                  · intro t u _ _; exact Set.empty_disjoint _
                  · rw [iterW_empty]; exact Set.empty_disjoint _
                  · exact ⟨(aPtrT_zero d i.val).symm, (bPtrT_zero d i.val).symm, rfl,
                      fun x y _ => ⟨fun _ => 0, rfl, by intro j hj; omega, zeros_val x y⟩⟩
                  · intro t ht st m₁ hI
                    rw [tripCount_steps d hd] at ht
                    obtain ⟨h1, h2, hm, hv⟩ := hI
                    rw [h1, h2, hm]
                    have hz : ((IntV.ofInt (0 : ℤ) : I32)).toInt = 0 :=
                      toInt_ofInt' (by norm_num) (by norm_num)
                    have ho : ((IntV.ofInt (1 : ℤ) : I32)).toInt = 1 :=
                      toInt_ofInt' (by norm_num) (by norm_num)
                    rw [hz, ho]
                    refine safe_convert (d := DType.f32) (d' := DType.f8e5) ?_ ?_
                    · intro j
                      obtain ⟨r, hr⟩ :=
                        hcvtTotal ((tl.full [128, 64] (FloatV.ofPat 0).litF32).get j)
                      exact ⟨r, Or.inr (Or.inl ⟨Or.inr (Or.inr rfl), Or.inl rfl, hr⟩)⟩
                    · intro a0 _
                      refine safe_load_buf (b := d.a)
                        (g := fun j : Idx [128, 64] => aOff d i.val j.1 (64 * t + j.2.1))
                        (en := fun _ => true) ?_ ?_ hpre.1 (aImg_sub_seg hd hp init ht) ?_
                      · refine Buf.addrs_of_get d.a ?_ ?_
                        · intro j
                          show (dotMaskA d t).get j = true
                          exact dotMaskA_get d hd ht j
                        · intro j _
                          exact aPtrT_get d hd hp t j.1 j.2.1
                      · intro j _
                        exact aOff_lt_len hd hp (seg_lt d hd ht j.2.1 j.2.1.isLt)
                      · intro v hread _
                        refine safe_convert (d := DType.f32) (d' := DType.f8e5) ?_ ?_
                        · intro j
                          obtain ⟨r, hr⟩ :=
                            hcvtTotal ((tl.full [64, 256] (FloatV.ofPat 0).litF32).get j)
                          exact ⟨r, Or.inr (Or.inl ⟨Or.inr (Or.inr rfl), Or.inl rfl, hr⟩)⟩
                        · intro b0 _
                          refine safe_load_buf (b := d.b)
                            (g := fun j : Idx [64, 256] => bOff d i.val j.2.1 (64 * t + j.1))
                            (en := fun _ => true) ?_ ?_ hpre.2.1 (bImg_sub_seg hd init ht) ?_
                          · refine Buf.addrs_of_get d.b ?_ ?_
                            · intro j
                              show (dotMaskB d t).get j = true
                              exact dotMaskB_get d hd ht j
                            · intro j _
                              exact bPtrT_get d hd hp t j.1 j.2.1
                          · intro j _
                            exact bOff_lt_len hd (seg_lt d hd ht j.1 j.1.isLt)
                          · intro v2 hread2 _
                            refine safe_dot
                              (cfg := ⟨.f8e5, .f8e5, .f32, .tf32, 2 ^ 30⟩) ?_ ?_
                            · exact hdotTotal v v2 st.2.2
                            · intro r hdr
                              refine Safe.ret ?_
                              refine ⟨(aPtrT_succ d i.val t).symm, (bPtrT_succ d i.val t).symm,
                                rfl, ?_⟩
                              intro x y hfl
                              obtain ⟨S, hS0, hS, hval⟩ := hv x y hfl
                              have hva : ∀ k : Fin 64,
                                  DType.val? .f8e5 (v.get (x, k, ())) =
                                    some ((vaOf d init i.val x.val (64 * t + k.val)).toRat) := by
                                intro k
                                have hrd := hread (x, k, ()) rfl
                                have hEq :
                                    vaOf d init i.val x.val (64 * t + k.val) =
                                      v.get (x, k, ()) := by
                                  simp only [vaOf, Option.getD_some, hrd]
                                rw [← hEq]
                                exact val?_f8e5.mpr
                                  ⟨hfl.1 (64 * t + k.val) (seg_lt d hd ht k.val k.isLt), rfl⟩
                              have hvb : ∀ k : Fin 64,
                                  DType.val? .f8e5 (v2.get (k, y, ())) =
                                    some ((vbOf d init i.val y.val (64 * t + k.val)).toRat) := by
                                intro k
                                have hrd := hread2 (k, y, ()) rfl
                                have hEq :
                                    vbOf d init i.val y.val (64 * t + k.val) =
                                      v2.get (k, y, ()) := by
                                  simp only [vbOf, Option.getD_some, hrd]
                                rw [← hEq]
                                exact val?_f8e5.mpr
                                  ⟨hfl.2 (64 * t + k.val) (seg_lt d hd ht k.val k.isLt), rfl⟩
                              obtain ⟨S', -, hS'0, hSteps, hval'⟩ :=
                                dot_step (P := P) (prec := .tf32) (K := d.K) hd.K_le
                                  (K64 d hd)
                                  (fun k hk => f8e5_abs_toRat (hfl.1 k hk))
                                  (fun k hk => f8e5_abs_toRat (hfl.2 k hk))
                                  (hdot .tf32) hdr x y ht hva hvb hS0 hS hval
                              exact ⟨S', hS'0, hSteps, hval'⟩
                  · intro st m₁ hI
                    rw [tripCount_steps d hd] at hI
                    obtain ⟨-, -, hm, hv⟩ := hI
                    rw [hm]
                    refine safe_cvt (f := .binary32) (g := .binary16) (rnd := .rn) ?_ ?_
                    · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
                    · intro c hne
                      refine safe_last ?_
                      refine safe_store_buf (b := d.c)
                        (g := fun j : Idx [128, 256] => cOff d i.val j.1 j.2.1) (en := cEn d i.val)
                        ?_ ?_ (cImg_inj hd hp) hpre.2.2 (cImg_sub hd hp init) ?_
                      · refine Buf.addrs_of_get d.c ?_ ?_
                        · intro j; rfl
                        · intro j hj; exact cPtrs_eq d hd hp j hj
                      · intro j hj
                        have hh := (cMask_get d hd hp j).mp hj
                        exact cOff_lt_len hd hh.1 hh.2
                      · intro m' hread _ _
                        refine Safe.ret ?_
                        intro x y hx hy hxM hyN
                        refine outSpec_of_acc_cond (d := d) (init := init) (fin := m')
                          (p := i.val) (x := x) (y := y)
                          (acc := st.2.2.get (⟨x, hx⟩, ⟨y, hy⟩, ()))
                          (c := c.get (⟨x, hx⟩, ⟨y, hy⟩, ())) hd hxM hyN ?_ ?_ ?_
                        · exact hread (⟨x, hx⟩, ⟨y, hy⟩, ())
                            ((cMask_get d hd hp _).mpr ⟨hxM, hyN⟩)
                        · simpa only [FloatV.cvtWith_rn] using hne (⟨x, hx⟩, ⟨y, hy⟩, ())
                        · intro hA hB
                          exact hv ⟨x, hx⟩ ⟨y, hy⟩ ⟨hA, hB⟩
                  · intro x hx
                    simp only [iterW_empty, Set.empty_union] at hx
                    exact hx

end Claims.MatmulE5m2
