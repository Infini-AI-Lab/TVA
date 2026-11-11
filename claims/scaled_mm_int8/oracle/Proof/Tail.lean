import Proof.Setup

/-!
# `scaled_mm_int8`: the epilogue's store

`tail_safe`: from the `float32` accumulator, convert to the output dtype, build the `C` pointer
tensor and mask, and store; the store satisfies the lane specification `LaneQ`.
-/

namespace Claims.ScaledMmInt8
open TVA

/-! ### The offsets and the mask of the store -/

/-- The `i`-th row offset, as an integer. -/
theorem t_row_toInt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d)
    (r4 r8 : Tensor [] I64) (r5 : Tensor [d.BM] I64)
    (e4 : r4.item.toInt = ((pm d p * d.BM : Nat) : Int))
    (e5 : ∀ k : Fin d.BM, (r5.get (k, ())).toInt = (k.1 : Int))
    (e8 : r8.item.toInt = (d.stride_cm : Int)) (k : Fin d.BM) :
    ((r8.item * (r4.item + r5)).get (k, ())).toInt
      = (d.stride_cm : Int) * ((pm d p * d.BM + (k.1 : Nat) : Nat) : Int) := by
  have hsum : ((r4.item + r5).get (k, ())).toInt
      = ((pm d p * d.BM + (k.1 : Nat) : Nat) : Int) := by
    rw [show (r4.item + r5).get (k, ()) = r4.item + r5.get (k, ()) from rfl]
    exact IntV.I64.add_nat e4 (e5 k) (by
      have h1 := pmBM_lt31 hd hp
      have h2 := k.2
      have h3 := BM_le256 hd
      omega)
  rw [show (r8.item * (r4.item + r5)).get (k, ()) = r8.item * (r4.item + r5).get (k, ()) from rfl,
    IntV.I64.mul_nat e8 hsum (by
      have h1 := hd.strides.2.2.2.2.1
      have h2 := pmBM_lt31 hd hp
      have h3 := k.2
      have h4 := BM_le256 hd
      have h5 : pm d p * d.BM + (k.1 : Nat) < 2 ^ 31 + 2 ^ 8 := by omega
      calc d.stride_cm * (pm d p * d.BM + (k.1 : Nat))
          ≤ (2 ^ 31 - 1) * (2 ^ 31 + 2 ^ 8 - 1) := Nat.mul_le_mul (by omega) (by omega)
        _ < 2 ^ 63 := by norm_num)]
  push_cast; ring

/-- The `j`-th column offset, as an integer. -/
theorem t_col_toInt {d : Inst} (hd : d.ok) (p : Nat)
    (r6 r9 : Tensor [] I64) (r7 : Tensor [d.BN] I64)
    (e6 : r6.item.toInt = ((pn d p * d.BN : Nat) : Int))
    (e7 : ∀ k : Fin d.BN, (r7.get (k, ())).toInt = (k.1 : Int))
    (e9 : r9.item.toInt = (d.stride_cn : Int)) (k : Fin d.BN) :
    ((r9.item * (r6.item + r7)).get (k, ())).toInt
      = (d.stride_cn : Int) * ((pn d p * d.BN + (k.1 : Nat) : Nat) : Int) := by
  have hsum : ((r6.item + r7).get (k, ())).toInt
      = ((pn d p * d.BN + (k.1 : Nat) : Nat) : Int) := by
    rw [show (r6.item + r7).get (k, ()) = r6.item + r7.get (k, ()) from rfl]
    exact IntV.I64.add_nat e6 (e7 k) (by
      have h1 := pnBN_lt31 hd p
      have h2 := k.2
      have h3 := BN_le256 hd
      omega)
  rw [show (r9.item * (r6.item + r7)).get (k, ()) = r9.item * (r6.item + r7).get (k, ()) from rfl,
    IntV.I64.mul_nat e9 hsum (by
      have h1 := hd.strides.2.2.2.2.2
      have h2 := pnBN_lt31 hd p
      have h3 := k.2
      have h4 := BN_le256 hd
      have h5 : pn d p * d.BN + (k.1 : Nat) < 2 ^ 31 + 2 ^ 8 := by omega
      calc d.stride_cn * (pn d p * d.BN + (k.1 : Nat))
          ≤ (2 ^ 31 - 1) * (2 ^ 31 + 2 ^ 8 - 1) := Nat.mul_le_mul (by omega) (by omega)
        _ < 2 ^ 63 := by norm_num)]
  push_cast; ring

/-- The mask's element at a lane. -/
theorem t_mask_get {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d)
    (r4 r6 r10 r11 : Tensor [] I64) (r5 : Tensor [d.BM] I64) (r7 : Tensor [d.BN] I64)
    (e4 : r4.item.toInt = ((pm d p * d.BM : Nat) : Int))
    (e5 : ∀ k : Fin d.BM, (r5.get (k, ())).toInt = (k.1 : Int))
    (e6 : r6.item.toInt = ((pn d p * d.BN : Nat) : Int))
    (e7 : ∀ k : Fin d.BN, (r7.get (k, ())).toInt = (k.1 : Int))
    (e10 : r10.item.toInt = (d.M : Int)) (e11 : r11.item.toInt = (d.N : Int))
    (i : Idx [d.BM, d.BN]) :
    ((((r4.item + r5).col d.BN).ltS r10.item).and
        (((r6.item + r7).row d.BM).ltS r11.item)).get i =
      (decide (pm d p * d.BM + (i.1 : Nat) < d.M) &&
        decide (pn d p * d.BN + (i.2.1 : Nat) < d.N)) := by
  have hsum1 : ((r4.item + r5).get (i.1, ())).toInt
      = ((pm d p * d.BM + (i.1 : Nat) : Nat) : Int) := by
    rw [show (r4.item + r5).get (i.1, ()) = r4.item + r5.get (i.1, ()) from rfl]
    exact IntV.I64.add_nat e4 (e5 i.1) (by
      have h1 := pmBM_lt31 hd hp
      have h2 := i.1.2
      have h3 := BM_le256 hd
      omega)
  have hsum2 : ((r6.item + r7).get (i.2.1, ())).toInt
      = ((pn d p * d.BN + (i.2.1 : Nat) : Nat) : Int) := by
    rw [show (r6.item + r7).get (i.2.1, ()) = r6.item + r7.get (i.2.1, ()) from rfl]
    exact IntV.I64.add_nat e6 (e7 i.2.1) (by
      have h1 := pnBN_lt31 hd p
      have h2 := i.2.1.2
      have h3 := BN_le256 hd
      omega)
  have hA : (((r4.item + r5).col d.BN).ltS r10.item).get i
      = IntV.lt ((r4.item + r5).get (i.1, ())) r10.item := rfl
  have hB : (((r6.item + r7).row d.BM).ltS r11.item).get i
      = IntV.lt ((r6.item + r7).get (i.2.1, ())) r11.item := rfl
  have h1 : IntV.lt ((r4.item + r5).get (i.1, ())) r10.item
      = decide (pm d p * d.BM + (i.1 : Nat) < d.M) := by
    rw [IntV.lt_eq, hsum1, e10]
    simp only [Nat.cast_lt]
  have h2 : IntV.lt ((r6.item + r7).get (i.2.1, ())) r11.item
      = decide (pn d p * d.BN + (i.2.1 : Nat) < d.N) := by
    rw [IntV.lt_eq, hsum2, e11]
    simp only [Nat.cast_lt]
  rw [Tensor.get_and, hA, hB, h1, h2]

/-- The pointer of a lane. -/
theorem t_cptr_get {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d)
    (r4 r6 r8 r9 : Tensor [] I64) (r5 : Tensor [d.BM] I64) (r7 : Tensor [d.BN] I64)
    (e4 : r4.item.toInt = ((pm d p * d.BM : Nat) : Int))
    (e5 : ∀ k : Fin d.BM, (r5.get (k, ())).toInt = (k.1 : Int))
    (e6 : r6.item.toInt = ((pn d p * d.BN : Nat) : Int))
    (e7 : ∀ k : Fin d.BN, (r7.get (k, ())).toInt = (k.1 : Int))
    (e8 : r8.item.toInt = (d.stride_cm : Int))
    (e9 : r9.item.toInt = (d.stride_cn : Int))
    (i : Idx [d.BM, d.BN]) :
    (d.c.ptr 0 + (r8.item * (r4.item + r5)).col d.BN +
        (r9.item * (r6.item + r7)).row d.BM).get i =
      d.c.ptr ↑((pm d p * d.BM + (i.1 : Nat)) * d.stride_cm +
        (pn d p * d.BN + (i.2.1 : Nat)) * d.stride_cn) := by
  have hr := t_row_toInt hd hp r4 r8 r5 e4 e5 e8 i.1
  have hc := t_col_toInt hd p r6 r9 r7 e6 e7 e9 i.2.1
  have hcol : ((r8.item * (r4.item + r5)).col d.BN).get i
      = (r8.item * (r4.item + r5)).get (i.1, ()) := rfl
  have hrow : ((r9.item * (r6.item + r7)).row d.BM).get i
      = (r9.item * (r6.item + r7)).get (i.2.1, ()) := rfl
  apply Ptr.eq_of_addr
  simp only [Tensor.get_ptrAdd, Tensor.get_ptrAddT, Ptr.offset_addr, Buf.ptr_addr]
  rw [hcol, hrow, hr, hc]
  simp only [Nat.cast_add, Nat.cast_mul]
  ring

/-- The epilogue's tail: convert the `float32` accumulator to the output dtype, build the
`C` pointer tensor and mask, and store. -/
theorem tail_safe (P : Platform) (d : Inst) (hd : d.ok) (init : Mem) (hpre : Pre d init)
    (p : Nat) (hp : p < grid d) (pmT pnT : I32) (acc : Tensor [d.BM, d.BN] F32)
    (hpm : pmT.toInt = (pm d p : Int)) (hpn : pnT.toInt = (pn d p : Int))
    (hval : ∀ (cc : Tensor [d.BM, d.BN] d.od.Val),
      (∀ i, ConvRel P .f32 d.od (acc.get i) (cc.get i)) →
      ∀ (i : Idx [d.BM, d.BN]), LaneQ d init (pm d p * d.BM + i.1) (pn d p * d.BN + i.2.1) (cc.get i)) :
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
                    refine ⟨_, ?_, hL⟩
                    have hen : (fun i' : Idx [d.BM, d.BN] =>
                        decide (pm d p * d.BM + (i'.1 : Nat) < d.M) &&
                          decide (pn d p * d.BN + (i'.2.1 : Nat) < d.N))
                        ((⟨i, hi⟩, ⟨j, hj⟩, ()) : Idx [d.BM, d.BN]) = true := by
                      simp only [Bool.and_eq_true, decide_eq_true_eq]
                      exact ⟨by simpa using hiM, by simpa using hjN⟩
                    have := hread _ hen
                    simpa using this

end Claims.ScaledMmInt8
