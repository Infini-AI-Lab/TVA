import AttnBwd.DkDv

/-!
# `attn_bwd_f16`: `_attn_bwd_dq`, line by line

`safe_dq`: a call of `_attn_bwd_dq` (lines 319–363) from program `p` of head `b`, for the key rows
`n0, …, n0 + T·BN − 1`, is safe: it only loads (rows of `K`, `V` and `delta` of the head), writes
nothing, and moves the invariant `QInv` of the accumulator `dq` to the covered rows. Each trip of
the loop is `q_step`.
-/

namespace Kernels.AttnBwd

open TVA

/-- The loop state of `_attn_bwd_dq`: `dq`, `offs_n`, `curr_n`, `kT_ptrs`, `vT_ptrs`. -/
abbrev QState (d : Inst) (BN : Nat) : Type :=
  Tensor [128, d.HEAD_DIM] F32 × Tensor [BN] I32 × I32 × Tensor [d.HEAD_DIM, BN] (Ptr .f16) ×
    Tensor [d.HEAD_DIM, BN] (Ptr .f16)

/-- The loop invariant before trip `t`: memory unchanged, `curr_n = n0 + t·BN`, the pointer tiles
at rows `n0 + t·BN + j`, and `QInv` over the rows covered so far. -/
def QLoop (d : Inst) (init m : Mem) (b p : Nat) (C : Finset ℕ) (T0 n0 BN : Nat) (t : Nat)
    (st : QState d BN) (m' : Mem) : Prop :=
  m' = m ∧ st.2.2.1.toInt = ((n0 + t * BN : Nat) : Int) ∧
    st.2.2.2.1 = Tensor.of2 (fun (c : Fin d.HEAD_DIM) (j : Fin BN) =>
      d.K.ptr ↑(b * d.N_CTX * d.HEAD_DIM + (n0 + t * BN + j) * d.HEAD_DIM + c)) ∧
    st.2.2.2.2 = Tensor.of2 (fun (c : Fin d.HEAD_DIM) (j : Fin BN) =>
      d.V.ptr ↑(b * d.N_CTX * d.HEAD_DIM + (n0 + t * BN + j) * d.HEAD_DIM + c)) ∧
    QInv d init b p (C ∪ Finset.Ico n0 (n0 + t * BN)) (T0 + t) st.1

/-- **`_attn_bwd_dq`** from program `p` of head `b` (lines 319–363): with `start_m = 128p`,
`start_n = n0`, `num_steps = T` and block `BN ∈ {16, 32}`, every lane `Allowed` exactly when the
mask keeps it, it is safe, writes nothing and leaves `QInv` over the rows `C ∪ [n0, n0 + T·BN)`. -/
theorem safe_dq {E : Env} (hdot : Rules.Num.DotF16 E.P) (hdt : Rules.Num.DotTotal E.P dotCfg)
    (hexp : Rules.Num.Exp2F32 E.P) (hmt : Rules.Num.MathTotal E.P .exp2 .binary32)
    {d : Inst} (hd : d.ok) {init m : Mem} (hmem : ∀ x ∈ inBytes d, m x = init x)
    (hKv : d.K.Valid m) (hVv : d.V.Valid m) (hDv : d.D.Valid m)
    {b p : Nat} (hb : b < G d) (hp : 128 * p + 128 ≤ d.N_CTX)
    {BN : Nat} (hBN : BN = 16 ∨ BN = 32) (MASK : Bool)
    {s0 sn ns : I32} {n0 T : Nat} (hs0 : s0.toInt = ((128 * p : Nat) : Int)) (hsn : sn.toInt = (n0 : Int))
    (hns : ns.toInt = (T : Int)) (hrange : n0 + T * BN ≤ d.N_CTX)
    (hal : ∀ (r : Fin 128) (nn : Nat), n0 ≤ nn → nn < n0 + T * BN →
      decide (Allowed d nn (128 * p + r)) = (!MASK || decide (nn ≤ 128 * p + r)))
    {C : Finset ℕ} (hC : ∀ x ∈ C, x < n0 ∨ n0 + T * BN ≤ x) (hCN : C ⊆ Finset.range d.N_CTX)
    {T0 : Nat} (hT : T0 + T < 1024)
    {q dO : Tensor [128, d.HEAD_DIM] F16} {mv : Tensor [128, 1] F32}
    (hqv : ∀ hv : HeadV, hv.Holds d init b → ∀ (r : Fin 128) (c : Fin d.HEAD_DIM),
        q.get2 r c = hv.q (128 * p + r) c ∧ dO.get2 r c = hv.dO (128 * p + r) c)
    (hm : ∀ hv : HeadV, hv.Holds d init b → ∀ r : Fin 128, mv.get2 r 0 = hv.lse (128 * p + r))
    {dq : Tensor [128, d.HEAD_DIM] F32} (hinv : QInv d init b p C T0 dq) (H' N' : I32) :
    Safe E (inBytes d) ∅
      (fun r m' => m' = m ∧ QInv d init b p (C ∪ Finset.Ico n0 (n0 + T * BN)) (T0 + T) r) m
      (attn_bwd_dq 128 BN d.HEAD_DIM dq q (d.K.ptr ↑(b * d.N_CTX * d.HEAD_DIM))
        (d.V.ptr ↑(b * d.N_CTX * d.HEAD_DIM)) dO mv (d.D.ptr ↑(b * d.N_CTX))
        (IntV.ofNat d.HEAD_DIM) (IntV.ofNat 1) H' N' s0 sn ns MASK) := by
  have hHD := hd_le hd
  have hHD0 := hd_pos hd
  have hNle := hd.n_le
  have hBN32 : BN ≤ 32 := by omega
  unfold attn_bwd_dq
  simp only []
  have hrm : RowsAt (s0 + tl.arange 0 128 : Tensor [128] I32) (128 * p) := rowsAt_add hs0 (by omega)
  -- line 337: `Di`
  refine safe_load_buf d.D (g := fun (ix : Idx [128]) => b * d.N_CTX + (128 * p + (ix.1 : Nat)))
    (en := fun _ => true) (Buf.addrs_of_get_all d.D fun ⟨i, ⟨⟩⟩ =>
      congrArg (fun (x : Tensor [128] (Ptr .f32)) => x.get (i, ())) (vec_r d.D (b * d.N_CTX) hrm)) ?_ hDv ?_
    fun Di hDi _ => ?_
  · rintro ⟨i, ⟨⟩⟩ _
    have := row_lt hb (show 128 * p + (i : Nat) < d.N_CTX by have := i.2; omega)
    have := hd.d_len
    simp only; omega
  · refine (d.D.elems_mono ?_).trans (dd_sub d)
    rintro _ ⟨⟨i, ⟨⟩⟩, -, rfl⟩
    exact row_lt hb (show 128 * p + (i : Nat) < d.N_CTX by have := i.2; omega)
  have hRok : tl.RangeOk (IntV.ofInt 0 : I32) ns (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [IntV.I32.toInt_ofInt (by norm_num), IntV.I32.toInt_ofInt (by norm_num), hns]
    norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt ns.toInt (IntV.ofInt 1 : I32).toInt = T := by
    rw [IntV.I32.toInt_ofInt (by norm_num), IntV.I32.toInt_ofInt (by norm_num), hns]
    simp [tl.tripCount]
  have hr0 : RowsAt (sn + tl.arange 0 BN : Tensor [BN] I32) n0 := rowsAt_add hsn (by omega)
  have hk0 : RowsAt (tl.arange 0 d.HEAD_DIM : Tensor [d.HEAD_DIM] I32) 0 := rowsAt_arange0 (by omega)
  have hfit0 : (n0 + BN) * d.HEAD_DIM < 2 ^ 31 := by
    have h1 : n0 + BN ≤ 2 ^ 14 + 32 := by
      have : n0 ≤ n0 + T * BN := Nat.le_add_right _ _
      omega
    have : (n0 + BN) * d.HEAD_DIM ≤ (2 ^ 14 + 32) * 256 := Nat.mul_le_mul h1 hHD
    omega
  refine Safe.mono (safe_forRange (Wk := ∅) hRok (QLoop d init m b p C T0 n0 BN) (fun _ => ∅)
    (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, ?_, ?_, ?_, ?_⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · show sn.toInt = _; rw [hsn]; simp
  · refine (tile_cr d.K _ hr0 hk0 hfit0).trans ?_
    simp only [zero_mul, add_zero, Nat.sub_zero]
  · refine (tile_cr d.V _ hr0 hk0 hfit0).trans ?_
    simp only [zero_mul, add_zero, Nat.sub_zero]
  · simp only [zero_mul, add_zero, Finset.Ico_self, Finset.union_empty]
    exact hinv
  · -- one trip
    rw [htc]
    rintro t ht ⟨dq1, on, cn, kp, vp⟩ m₁ ⟨rfl, hcn, hkp, hvp, hinv1⟩
    simp only at hcn hkp hvp hinv1
    subst hkp hvp
    simp only []
    have hntT : n0 + t * BN + BN ≤ n0 + T * BN := by
      have : (t + 1) * BN ≤ T * BN := Nat.mul_le_mul_right _ ht
      rw [Nat.add_mul] at this; omega
    have hntN : n0 + t * BN + BN ≤ d.N_CTX := by omega
    set nt := n0 + t * BN with hnt_def
    -- lines 343–344: `kT`, `vT`
    refine safe_load_buf d.K (g := fun (ix : Idx [d.HEAD_DIM, BN]) => b * d.N_CTX * d.HEAD_DIM + (nt + (ix.2.1 : Nat)) * d.HEAD_DIM + (ix.1 : Nat))
      (en := fun _ => true) (Buf.addrs_of_get_all d.K fun ⟨c, j, ⟨⟩⟩ => rfl) ?_ hKv ?_
      fun kT hkT _ => ?_
    · rintro ⟨c, j, ⟨⟩⟩ _
      have := elem_lt hb (show nt + (j : Nat) < d.N_CTX by have := j.2; omega) c.2
      have := hd.k_len
      simp only; omega
    · refine (d.K.elems_mono ?_).trans (k_sub d)
      rintro _ ⟨⟨c, j, ⟨⟩⟩, -, rfl⟩
      exact elem_lt hb (show nt + (j : Nat) < d.N_CTX by have := j.2; omega) c.2
    refine safe_load_buf d.V (g := fun (ix : Idx [d.HEAD_DIM, BN]) => b * d.N_CTX * d.HEAD_DIM + (nt + (ix.2.1 : Nat)) * d.HEAD_DIM + (ix.1 : Nat))
      (en := fun _ => true) (Buf.addrs_of_get_all d.V fun ⟨c, j, ⟨⟩⟩ => rfl) ?_ hVv ?_
      fun vT hvT _ => ?_
    · rintro ⟨c, j, ⟨⟩⟩ _
      have := elem_lt hb (show nt + (j : Nat) < d.N_CTX by have := j.2; omega) c.2
      have := hd.v_len
      simp only; omega
    · refine (d.V.elems_mono ?_).trans (v_sub d)
      rintro _ ⟨⟨c, j, ⟨⟩⟩, -, rfl⟩
      exact elem_lt hb (show nt + (j : Nat) < d.N_CTX by have := j.2; omega) c.2
    -- line 345
    refine safe_dot (hdt _ _ _) fun qk hqk => ?_
    -- line 346
    refine safe_subc fun xs hxs => ?_
    refine safe_math (fun _ => hmt _) fun p0 hp0 => ?_
    -- line 353
    refine safe_dot (hdt _ _ _) fun dp0 hdp0 => ?_
    refine safe_convert (fun _ => conv_same_ex _ _) fun dp hdp => ?_
    -- line 354
    refine safe_subc fun ys hys => ?_
    refine safe_mul fun ds hds => ?_
    -- line 355
    refine safe_cvt std32 fun ds16 hds16 => ?_
    -- line 358
    refine safe_dot_add (hdt _ _ _) fun dq' hdq' => ?_
    -- lines 360–362
    refine safe_ret ⟨rfl, ?_, ?_, ?_, ?_⟩
    · show (cn + IntV.ofNat BN).toInt = _
      rw [IntV.I32.add_nat hcn (i32_nat (by omega)) (by omega)]
      congr 1; rw [Nat.add_mul]; ring
    · show _ + _ = _
      rw [of2_add_bc d.K _ (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega))
        (by nlinarith))]
      congr 1; funext c j; congr 2; rw [Nat.add_mul]; ring
    · show _ + _ = _
      rw [of2_add_bc d.V _ (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega))
        (by nlinarith))]
      congr 1; funext c j; congr 2; rw [Nat.add_mul]; ring
    · show QInv d init b p (C ∪ Finset.Ico n0 (n0 + (t + 1) * BN)) (T0 + (t + 1)) dq'
      rw [cover_succ, ← hnt_def, ← Nat.add_assoc]
      refine q_step hdot hexp hd hBN hntN hp (cover_apart hC ht) (cover_sub hCN (by omega))
        (by omega) (q := q) (dO := dO) (kT := kT) (vT := vT) (mv := mv) (Di := Di)
        ?_ hqk (fun r j => hxs (r, j, ())) (fun r j => hp0 (r, j, ())) (pT := _)
        (fun r j => ?_) hdp0 (fun r j => conv_same (hdp (r, j, ())))
        (fun r j => hys (r, j, ())) (fun r j => hds (r, j, ())) (fun r j => hds16 (r, j, ()))
        hdq' hinv1
      · intro hv hH
        refine ⟨hqv hv hH, fun c j => ⟨?_, ?_⟩, fun r => ⟨hm hv hH r, ?_⟩⟩
        · have hh := ((hH (nt + j) (by have := j.2; omega)).1 c c.2).2.1
          rw [eIdx_eq] at hh
          exact read_agree hmem d.K (k_sub d) (elem_lt hb (show nt + (j : Nat) < d.N_CTX by have := j.2; omega) c.2)
            (hkT (c, j, ()) rfl) hh
        · have hh := ((hH (nt + j) (by have := j.2; omega)).1 c c.2).2.2.1
          rw [eIdx_eq] at hh
          exact read_agree hmem d.V (v_sub d) (elem_lt hb (show nt + (j : Nat) < d.N_CTX by have := j.2; omega) c.2)
            (hvT (c, j, ()) rfl) hh
        · have hh := (hH (128 * p + r) (by have := r.2; omega)).2.2
          rw [rIdx_eq] at hh
          exact read_agree hmem d.D (dd_sub d) (row_lt hb (show 128 * p + (r : Nat) < d.N_CTX by have := r.2; omega))
            (hDi (r, ()) rfl) hh
      · rw [mask_q MASK hs0 hcn (by omega) (by omega) p0 r j,
          hal r (nt + j) (by omega) (by have := j.2; omega)]
  · -- after the loop
    rw [htc]
    rintro ⟨dq1, on, cn, kp, vp⟩ m₁ ⟨rfl, -, -, -, hinv1⟩
    exact safe_ret ⟨rfl, hinv1⟩

end Kernels.AttnBwd
