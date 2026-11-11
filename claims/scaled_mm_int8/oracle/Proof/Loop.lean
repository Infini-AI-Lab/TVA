import Proof.Setup
import Proof.Dots
import Proof.Masks

/-!
# `scaled_mm_int8`: one trip of the reduction loop

`loop_safe`: under the invariant at trip `t`, the body of the `for` loop (the masked loads of the
`A` and `B` tiles, the `int8` dot into the `int32` accumulator, and the advances of the offset and
pointer tensors) is safe and re-establishes the invariant at trip `t + 1`.
-/

namespace Claims.ScaledMmInt8
open TVA

set_option maxHeartbeats 2000000

/-! ### Local helpers -/

/-- The `int64` widening of a small `int32` constant has the constant's value. -/
theorem lp_convRel_i32_i64_ofNat {P : Platform} {n : Nat} (hn : n < 2 ^ 31)
    {r : Tensor [] I64} (h : ConvRel P .i32 .i64 (IntV.ofNat n : I32) (r.get ())) :
    r.item.toInt = (n : ℤ) := by
  show (r.get ()).toInt = _
  rw [convRel_i32_i64 h, IntV.I32.toInt_ofNat hn]

/-- An `int8` element read is at most `128` in magnitude. -/
theorem lp_rI8_abs (m : Mem) (addr : ℤ) : |rI8 m addr| ≤ 128 := by
  simp only [rI8]
  cases h : m.read .i8 addr with
  | none => simp
  | some v =>
      show |v.toInt| ≤ 128
      exact I8.abs_toInt_le v

/-- One product `A[i, k]·B[k, j]` is at most `16384` in magnitude. -/
theorem lp_rA_mul_rB_abs (m : Mem) (d : Inst) (i k j : Nat) :
    |rA m d i k * rB m d k j| ≤ 16384 := by
  have h1 : |rA m d i k| ≤ 128 := lp_rI8_abs m _
  have h2 : |rB m d k j| ≤ 128 := lp_rI8_abs m _
  rw [abs_mul]
  calc |rA m d i k| * |rB m d k j| ≤ 128 * 128 :=
        mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    _ = 16384 := by norm_num

/-- A partial dot of `n` terms is at most `n·16384` in magnitude. -/
theorem lp_abs_sum_le (init : Mem) (d : Inst) (i j n : Nat) :
    |(∑ k ∈ Finset.range n, rA init d i k * rB init d k j)| ≤ (n : ℤ) * 16384 := by
  calc |∑ k ∈ Finset.range n, rA init d i k * rB init d k j|
      ≤ ∑ k ∈ Finset.range n, |rA init d i k * rB init d k j| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k ∈ Finset.range n, (16384 : ℤ) :=
        Finset.sum_le_sum (fun k _ => lp_rA_mul_rB_abs init d i k j)
    _ = (n : ℤ) * 16384 := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]

/-- The `k` mask `offsets_k < K` is `decide (t·BK + kk < K)` per lane. -/
theorem lp_mask_k {d : Inst} {t : Nat} (offsets_k : Tensor [d.BK] I64) (r : I64)
    (hK : r.toInt = (d.K : ℤ))
    (h1 : ∀ kk : Fin d.BK, (offsets_k.get1 kk).toInt = ((t * d.BK + kk.1 : Nat) : ℤ)) :
    ∀ kk : Idx [d.BK], (offsets_k.ltS r).get kk = decide (t * d.BK + kk.1 < d.K) := by
  intro kk
  obtain ⟨a, u⟩ := kk
  cases u
  show IntV.lt (offsets_k.get (a, ())) r = decide (t * d.BK + a.1 < d.K)
  rw [IntV.lt_eq, show (offsets_k.get (a, ())).toInt = ((t * d.BK + a.1 : Nat) : ℤ) from h1 a,
    hK]
  have hlt : decide (((t * d.BK + a.1 : Nat) : ℤ) < (d.K : ℤ)) = decide (t * d.BK + a.1 < d.K) := by
    simp only [decide_eq_decide]; norm_cast
  rw [hlt]

/-- The mask of the `A` tile is the row mask `&` the `k` mask. -/
theorem lp_masks_a {d : Inst} {p t : Nat} (masks_am : Tensor [d.BM] Bool)
    (k : Tensor [d.BK] Bool)
    (hmam : ∀ i : Idx [d.BM], masks_am.get i = decide (pm d p * d.BM + i.1 < d.M))
    (hk : ∀ kk : Idx [d.BK], k.get kk = decide (t * d.BK + kk.1 < d.K)) :
    ∀ i : Idx [d.BM, d.BK], ((masks_am.col d.BK).and (k.row d.BM)).get i
      = (decide (pm d p * d.BM + i.1 < d.M) && decide (t * d.BK + i.2.1 < d.K)) := by
  intro i
  obtain ⟨a, kk, u⟩ := i
  have h1 : ((masks_am.col d.BK).and (k.row d.BM)).get (a, kk, u)
      = (masks_am.get (a, u) && k.get (kk, u)) := rfl
  have h2 : masks_am.get (a, u) = decide (pm d p * d.BM + a.1 < d.M) := hmam (a, u)
  have h3 : k.get (kk, u) = decide (t * d.BK + kk.1 < d.K) := hk (kk, u)
  rw [h1, h2, h3]

/-- A lane of a tensor plus a scalar is the lane plus the scalar. -/
theorem lp_get1_adds {n : Nat} {sg : Bool} {w : Nat} (x : Tensor [n] (IntV sg w)) (a : IntV sg w)
    (i : Fin n) : (x + a).get1 i = x.get1 i + a := rfl

/-- A lane of a pointer tensor plus an offset tensor is the pointer offset by the lane. -/
theorem lp_get2_ptrAddT {d : DType} {m n : Nat} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr d)) (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (ps + offs).get2 i j = (ps.get2 i j).offset (offs.get2 i j).toInt := rfl

/-- Every lane of a constant tensor is the constant. -/
theorem lp_get2_full {m n : Nat} {α : Type} (a : α) (i : Fin m) (j : Fin n) :
    (Tensor.full a : Tensor [m, n] α).get2 i j = a := rfl

/-- The mask of the `B` tile is the `k` mask `&` the column mask. -/
theorem lp_masks_b {d : Inst} {p t : Nat} (masks_bn : Tensor [d.BN] Bool)
    (k : Tensor [d.BK] Bool)
    (hmbn : ∀ i : Idx [d.BN], masks_bn.get i = decide (pn d p * d.BN + i.1 < d.N))
    (hk : ∀ kk : Idx [d.BK], k.get kk = decide (t * d.BK + kk.1 < d.K)) :
    ∀ i : Idx [d.BK, d.BN], ((k.col d.BN).and (masks_bn.row d.BK)).get i
      = (decide (t * d.BK + i.1 < d.K) && decide (pn d p * d.BN + i.2.1 < d.N)) := by
  intro i
  obtain ⟨kk, j, u⟩ := i
  have h1 : ((k.col d.BN).and (masks_bn.row d.BK)).get (kk, j, u)
      = (k.get (kk, u) && masks_bn.get (j, u)) := rfl
  have h2 : k.get (kk, u) = decide (t * d.BK + kk.1 < d.K) := hk (kk, u)
  have h3 : masks_bn.get (j, u) = decide (pn d p * d.BN + j.1 < d.N) := hmbn (j, u)
  rw [h1, h2, h3]

/-! ### The loop body -/

/-- **One trip of the loop is safe** and re-establishes the invariant at trip `t + 1`. -/
theorem loop_safe (P : Platform) (d : Inst) (hd : d.ok) (init : Mem) (hpre : Pre d init)
    (hdot : Device.DotI8 P)
    (htot : Device.DotTotal P
      { a := .i8, b := .i8, acc := .i32, precision := .tf32, maxImpreciseAcc := 0 })
    (p : Nat) (t : Nat)
    (masks_am : Tensor [d.BM] Bool) (masks_bn : Tensor [d.BN] Bool)
    (hmam : ∀ i : Idx [d.BM], masks_am.get i = decide (pm d p * d.BM + i.1 < d.M))
    (hmbn : ∀ i : Idx [d.BN], masks_bn.get i = decide (pn d p * d.BN + i.1 < d.N))
    (offsets_k : Tensor [d.BK] I64) (a_ptrs : Tensor [d.BM, d.BK] (Ptr .i8))
    (b_ptrs : Tensor [d.BK, d.BN] (Ptr .i8)) (accumulator : Tensor [d.BM, d.BN] I32)
    (ht : t * d.BK + d.BK ≤ d.K)
    (hInv : Inv d init p t (offsets_k, a_ptrs, b_ptrs, accumulator)) :
    Safe ⟨P, true⟩ (Rfp d p) ∅ (fun st' m₂ => m₂ = init ∧ Inv d init p (t + 1) st') init
      (do
        let masks_k : Tensor [d.BK] Bool :=
          offsets_k.ltS (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.K : I32))).item
        let masks_a := (masks_am.col d.BK).and (masks_k.row d.BM)
        let a ← tl.load a_ptrs (some masks_a)
        let masks_b := (masks_k.col d.BN).and (masks_bn.row d.BK)
        let b ← tl.load b_ptrs (some masks_b)
        let accumulator ← tl.dot
          { a := .i8, b := .i8, acc := .i32, precision := .tf32, maxImpreciseAcc := 0 } a b accumulator
        let offsets_k := offsets_k + (IntV.ofNat d.BK : I64)
        let a_ptrs := a_ptrs +
          (bc ((IntV.ofNat d.BK : I32) * (IntV.ofNat d.stride_ak : I32)) : Tensor [d.BM, d.BK] I32)
        let b_ptrs := b_ptrs +
          (bc ((IntV.ofNat d.BK : I32) * (IntV.ofNat d.stride_bk : I32)) : Tensor [d.BK, d.BN] I32)
        return (offsets_k, a_ptrs, b_ptrs, accumulator)) := by
  obtain ⟨h1, h2, h3, h4⟩ := hInv
  have hBKpos : 0 < d.BK := BK_pos hd
  have hBK256 : d.BK ≤ 256 := BK_le hd
  have hBM256 : d.BM ≤ 256 := BM_le256 hd
  have hBN256 : d.BN ≤ 256 := BN_le256 hd
  have hK17 : d.K < 2 ^ 17 := hd.K_lt
  have hBK31 : d.BK < 2 ^ 31 := by omega
  have hBM31 : d.BM < 2 ^ 31 := by omega
  have hK31 : d.K < 2 ^ 31 := by omega
  have hBK64 : (IntV.ofNat d.BK : I64).toInt = (d.BK : ℤ) := by
    exact IntV.I64.toInt_ofNat (by omega)
  have hBKv32 : (IntV.ofNat d.BK : I32).toInt = (d.BK : ℤ) := IntV.I32.toInt_ofNat hBK31
  have hakv32 : (IntV.ofNat d.stride_ak : I32).toInt = (d.stride_ak : ℤ) :=
    IntV.I32.toInt_ofNat hd.strides.2.1
  have hbkv32 : (IntV.ofNat d.stride_bk : I32).toInt = (d.stride_bk : ℤ) :=
    IntV.I32.toInt_ofNat hd.strides.2.2.1
  refine safe_convert (d := .i32) (d' := .i64)
    (x := Tensor.scalar (IntV.ofNat d.K : I32)) ?_ ?_
  · intro i; exact ⟨_, rfl⟩
  · intro r hKr
    have hKv : r.item.toInt = (d.K : ℤ) := lp_convRel_i32_i64_ofNat hK31 (hKr ())
    have hkk : ∀ kk : Idx [d.BK],
        (offsets_k.ltS r.item).get kk = decide (t * d.BK + kk.1 < d.K) :=
      lp_mask_k offsets_k r.item hKv h1
    simp only []
    refine safe_load_buf (b := d.a)
      (g := fun i : Idx [d.BM, d.BK] =>
        (pm d p * d.BM + i.1) * d.stride_am + (t * d.BK + i.2.1) * d.stride_ak)
      (en := fun i : Idx [d.BM, d.BK] =>
        decide (pm d p * d.BM + i.1 < d.M) && decide (t * d.BK + i.2.1 < d.K)) ?_ ?_ ?_ ?_ ?_
    · refine Buf.addrs_of_get d.a ?_ ?_
      · intro i
        exact lp_masks_a masks_am (offsets_k.ltS r.item) hmam hkk i
      · intro i hi
        obtain ⟨a, kk, u⟩ := i
        cases u
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
        obtain ⟨hM, hK⟩ := hi
        show a_ptrs.get (a, kk, ())
          = d.a.ptr (((pm d p * d.BM + a.1) * d.stride_am
              + (t * d.BK + kk.1) * d.stride_ak : Nat) : ℤ)
        exact h2 a kk
    · intro i hi
      obtain ⟨a, kk, u⟩ := i
      cases u
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
      obtain ⟨hM, hK⟩ := hi
      show (pm d p * d.BM + a.1) * d.stride_am + (t * d.BK + kk.1) * d.stride_ak < d.a.len
      exact hd.a_len _ _ hM (by omega)
    · exact hpre.1
    · rintro x hx
      simp only [Buf.elems, Buf.img] at hx
      obtain ⟨j, ⟨i, hen, hgij⟩, hx⟩ := hx
      rw [← hgij] at hx
      obtain ⟨a, kk, u⟩ := i
      cases u
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
      obtain ⟨hM, hK⟩ := hen
      exact Or.inl (Or.inl (Or.inl (Or.inl
        ⟨_, ⟨a.1, t * d.BK + kk.1, a.2, by omega, hM, rfl⟩, hx⟩)))
    · intro av hav havO
      refine safe_load_buf (b := d.b)
        (g := fun i : Idx [d.BK, d.BN] =>
          (t * d.BK + i.1) * d.stride_bk + (pn d p * d.BN + i.2.1) * d.stride_bn)
        (en := fun i : Idx [d.BK, d.BN] =>
          decide (t * d.BK + i.1 < d.K) && decide (pn d p * d.BN + i.2.1 < d.N)) ?_ ?_ ?_ ?_ ?_
      · refine Buf.addrs_of_get d.b ?_ ?_
        · intro i
          exact lp_masks_b masks_bn (offsets_k.ltS r.item) hmbn hkk i
        · intro i hi
          obtain ⟨kk, j, u⟩ := i
          cases u
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
          obtain ⟨hK, hN⟩ := hi
          show b_ptrs.get (kk, j, ())
            = d.b.ptr (((t * d.BK + kk.1) * d.stride_bk
                + (pn d p * d.BN + j.1) * d.stride_bn : Nat) : ℤ)
          exact h3 kk j
      · intro i hi
        obtain ⟨kk, j, u⟩ := i
        cases u
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hi
        obtain ⟨hK, hN⟩ := hi
        show (t * d.BK + kk.1) * d.stride_bk + (pn d p * d.BN + j.1) * d.stride_bn < d.b.len
        exact hd.b_len _ _ (by omega) hN
      · exact hpre.2.1
      · rintro x hx
        simp only [Buf.elems, Buf.img] at hx
        obtain ⟨j, ⟨i, hen, hgij⟩, hx⟩ := hx
        rw [← hgij] at hx
        obtain ⟨kk, jj, u⟩ := i
        cases u
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
        obtain ⟨hK, hN⟩ := hen
        exact Or.inl (Or.inl (Or.inl (Or.inr
          ⟨_, ⟨t * d.BK + kk.1, jj.1, by omega, jj.2, hN, rfl⟩, hx⟩)))
      · intro bv hbv hbvO
        refine safe_dot (cfg := (⟨.i8, .i8, .i32, .tf32, 0⟩ : DotCfg))
          (M := d.BM) (N := d.BN) (K := d.BK)
          (a := av) (b := bv) (acc := accumulator)
          (htot (M := d.BM) (N := d.BN) (K := d.BK) av bv accumulator) ?_
        intro rr hrr
        refine Safe.ret ⟨rfl, ?_⟩
        refine ⟨?_, ?_, ?_, ?_⟩
        · intro kk
          have hoff : (offsets_k.get1 kk).toInt = ((t * d.BK + kk.1 : Nat) : ℤ) := h1 kk
          show ((offsets_k + (IntV.ofNat d.BK : I64)).get1 kk).toInt
            = (((t + 1) * d.BK + kk.1 : Nat) : ℤ)
          rw [lp_get1_adds, IntV.toInt_add rfl rfl (by
                rw [hoff, hBK64]
                exact sfits_add_nat (by omega)), hoff, hBK64]
          push_cast
          ring
        · intro i kk
          have hoff : a_ptrs.get2 i kk
              = d.a.ptr (((pm d p * d.BM + i.1) * d.stride_am
                  + (t * d.BK + kk.1) * d.stride_ak : Nat) : ℤ) := h2 i kk
          have hb : ((IntV.ofNat d.BK : I32) * (IntV.ofNat d.stride_ak : I32)).toInt
              = ((d.BK * d.stride_ak : Nat) : ℤ) := by
            rw [IntV.toInt_mul rfl rfl (by
                  rw [hBKv32, hakv32]
                  exact m_sfits_mul32 hd.steps.1), hBKv32, hakv32]
            push_cast
            ring
          show (a_ptrs + (Tensor.full ((IntV.ofNat d.BK : I32) * (IntV.ofNat d.stride_ak : I32))
              : Tensor [d.BM, d.BK] I32)).get2 i kk
            = d.a.ptr (((pm d p * d.BM + i.1) * d.stride_am
                + ((t + 1) * d.BK + kk.1) * d.stride_ak : Nat) : ℤ)
          rw [lp_get2_ptrAddT, lp_get2_full, hoff, hb, Buf.ptr_offset]
          congr 1
          push_cast
          ring
        · intro kk j
          have hoff : b_ptrs.get2 kk j
              = d.b.ptr (((t * d.BK + kk.1) * d.stride_bk
                  + (pn d p * d.BN + j.1) * d.stride_bn : Nat) : ℤ) := h3 kk j
          have hb : ((IntV.ofNat d.BK : I32) * (IntV.ofNat d.stride_bk : I32)).toInt
              = ((d.BK * d.stride_bk : Nat) : ℤ) := by
            rw [IntV.toInt_mul rfl rfl (by
                  rw [hBKv32, hbkv32]
                  exact m_sfits_mul32 hd.steps.2), hBKv32, hbkv32]
            push_cast
            ring
          show (b_ptrs + (Tensor.full ((IntV.ofNat d.BK : I32) * (IntV.ofNat d.stride_bk : I32))
              : Tensor [d.BK, d.BN] I32)).get2 kk j
            = d.b.ptr ((((t + 1) * d.BK + kk.1) * d.stride_bk
                + (pn d p * d.BN + j.1) * d.stride_bn : Nat) : ℤ)
          rw [lp_get2_ptrAddT, lp_get2_full, hoff, hb, Buf.ptr_offset]
          congr 1
          push_cast
          ring
        · intro i j hi hj
          have hacc := h4 i j hi hj
          have haen : ∀ k : Fin d.BK,
              (decide (pm d p * d.BM + i.1 < d.M)
                && decide (t * d.BK + k.1 < d.K)) = true := by
            intro k
            rw [Bool.and_eq_true]
            exact ⟨decide_eq_true_eq.mpr hi, decide_eq_true_eq.mpr (by omega)⟩
          have hben : ∀ k : Fin d.BK,
              (decide (t * d.BK + k.1 < d.K)
                && decide (pn d p * d.BN + j.1 < d.N)) = true := by
            intro k
            rw [Bool.and_eq_true]
            exact ⟨decide_eq_true_eq.mpr (by omega), decide_eq_true_eq.mpr hj⟩
          let va : Fin d.BK → ℚ := fun k => ((av.get2 i k).toInt : ℚ)
          let vb : Fin d.BK → ℚ := fun k => ((bv.get2 k j).toInt : ℚ)
          let vc : ℚ := ((accumulator.get2 i j).toInt : ℚ)
          have hAZ : |vc| ≤ ((t * d.BK : Nat) : ℚ) * 16384 := by
            show |((accumulator.get2 i j).toInt : ℚ)| ≤ ((t * d.BK : Nat) : ℚ) * 16384
            rw [hacc, ← Int.cast_abs]
            have h := lp_abs_sum_le init d (pm d p * d.BM + i.1) (pn d p * d.BN + j.1) (t * d.BK)
            exact_mod_cast h
          have hSZ : ∑ k : Fin d.BK, |va k * vb k| ≤ (d.BK : ℚ) * 16384 := by
            have h1 : ∀ k : Fin d.BK, |va k * vb k| ≤ (16384 : ℚ) := by
              intro k
              show |((av.get2 i k).toInt : ℚ) * ((bv.get2 k j).toInt : ℚ)| ≤ 16384
              rw [← Int.cast_mul, ← Int.cast_abs]
              exact_mod_cast I8.abs_mul_le (av.get2 i k) (bv.get2 k j)
            calc ∑ k : Fin d.BK, |va k * vb k| ≤ ∑ k : Fin d.BK, (16384 : ℚ) :=
                  Finset.sum_le_sum (fun k _ => h1 k)
              _ = (d.BK : ℚ) * 16384 := by
                  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          have hK1 : ((t * d.BK : Nat) : ℚ) + (d.BK : ℚ) ≤ (d.K : ℚ) := by
            exact_mod_cast ht
          have h6 : |vc| + ∑ k : Fin d.BK, |va k * vb k| ≤ (2 : ℚ) ^ 31 - 1 := by
            have h7 : |vc| + ∑ k : Fin d.BK, |va k * vb k|
                ≤ (((t * d.BK : Nat) : ℚ) + (d.BK : ℚ)) * 16384 := by
              have h8 : ((t * d.BK : Nat) : ℚ) * 16384 + (d.BK : ℚ) * 16384
                  = (((t * d.BK : Nat) : ℚ) + (d.BK : ℚ)) * 16384 := by ring
              linarith
            have h9 : (d.K : ℚ) * 16384 ≤ (2 : ℚ) ^ 31 - 1 := by
              have h : (d.K : ℚ) ≤ 131071 := by
                have h2 : d.K ≤ 131071 := by omega
                exact_mod_cast h2
              nlinarith
            have h10 : (((t * d.BK : Nat) : ℚ) + (d.BK : ℚ)) * 16384
                ≤ (d.K : ℚ) * 16384 := mul_le_mul_of_nonneg_right hK1 (by norm_num)
            linarith
          have hbnd : |vc| + ∑ k : Fin d.BK, |va k * vb k|
              + ((fun _ : Nat => (0 : ℚ)) d.BK * (|vc| + ∑ k : Fin d.BK, |va k * vb k|)
                  + (fun _ : Nat => (0 : ℚ)) d.BK) ≤ DType.bound .i32 := by
            have hb : DType.bound .i32 = (2 : ℚ) ^ 31 - 1 := rfl
            rw [hb]
            simpa using h6
          obtain ⟨vr, hvr, hdist⟩ :=
            hdot .tf32 (a := av) (b := bv) (acc := accumulator) (r := rr) hrr i j va vb vc
              (fun k => rfl) (fun k => rfl) (fun k => ⟨trivial, trivial⟩) rfl hbnd
          have hvrZ : ((rr.get2 i j).toInt : ℚ) = vr := Option.some.inj hvr
          have hdist0 : vr = vc + ∑ k : Fin d.BK, va k * vb k := by
            have h0 : |vr - (vc + ∑ k : Fin d.BK, va k * vb k)| ≤ 0 := by
              simpa using hdist
            have h1 := abs_nonpos_iff.mp h0
            linarith
          have hsum_eq : ∑ k : Fin d.BK, va k * vb k
              = ((∑ k : Fin d.BK,
                  rA init d (pm d p * d.BM + i.1) (t * d.BK + k.1)
                    * rB init d (t * d.BK + k.1) (pn d p * d.BN + j.1) : ℤ) : ℚ) := by
            have step : ∀ k : Fin d.BK, va k * vb k
                = ((rA init d (pm d p * d.BM + i.1) (t * d.BK + k.1)
                    * rB init d (t * d.BK + k.1) (pn d p * d.BN + j.1) : ℤ) : ℚ) := by
              intro k
              have hra : rA init d (pm d p * d.BM + i.1) (t * d.BK + k.1)
                  = (av.get2 i k).toInt := rA_eq (hav (i, k, ()) (haen k))
              have hrb : rB init d (t * d.BK + k.1) (pn d p * d.BN + j.1)
                  = (bv.get2 k j).toInt := rB_eq (hbv (k, j, ()) (hben k))
              rw [hra, hrb]
              show ((av.get2 i k).toInt : ℚ) * ((bv.get2 k j).toInt : ℚ)
                = (((av.get2 i k).toInt * (bv.get2 k j).toInt : ℤ) : ℚ)
              push_cast
              ring
            simp only [step, Int.cast_sum]
          have hZ : (rr.get2 i j).toInt
              = (∑ k ∈ Finset.range (t * d.BK),
                  rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
                + ∑ k : Fin d.BK,
                  rA init d (pm d p * d.BM + i.1) (t * d.BK + k.1)
                    * rB init d (t * d.BK + k.1) (pn d p * d.BN + j.1) := by
            have key : ((rr.get2 i j).toInt : ℚ)
                = (((∑ k ∈ Finset.range (t * d.BK),
                      rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
                    + ∑ k : Fin d.BK,
                      rA init d (pm d p * d.BM + i.1) (t * d.BK + k.1)
                        * rB init d (t * d.BK + k.1) (pn d p * d.BN + j.1) : ℤ) : ℚ) := by
              push_cast
              rw [hvrZ, hdist0]
              have hcastacc : vc = ((∑ k ∈ Finset.range (t * d.BK),
                  rA init d (pm d p * d.BM + i.1) k
                    * rB init d k (pn d p * d.BN + j.1) : ℤ) : ℚ) := by
                show ((accumulator.get2 i j).toInt : ℚ) = _
                rw [hacc]
              rw [hcastacc, hsum_eq]
              simp only [Int.cast_sum, Int.cast_mul]
            exact Int.cast_injective key
          have hsplit : (∑ k ∈ Finset.range ((t + 1) * d.BK),
                rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
              = (∑ k ∈ Finset.range (t * d.BK),
                  rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))
                + ∑ k : Fin d.BK,
                  rA init d (pm d p * d.BM + i.1) (t * d.BK + k.1)
                    * rB init d (t * d.BK + k.1) (pn d p * d.BN + j.1) := by
            rw [show (t + 1) * d.BK = t * d.BK + d.BK from by ring, Finset.sum_range_add,
              ← Fin.sum_univ_eq_sum_range (fun k =>
                rA init d (pm d p * d.BM + i.1) (t * d.BK + k)
                  * rB init d (t * d.BK + k) (pn d p * d.BN + j.1)) d.BK]
          rw [hsplit]
          exact hZ

end Claims.ScaledMmInt8
