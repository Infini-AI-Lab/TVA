import AttnBwd.Step

/-!
# `attn_bwd_f16`: `_attn_bwd_dkdv`, line by line

`safe_dkdv`: a call of `_attn_bwd_dkdv` (lines 266–315) from program `p` of head `b`, for the
query rows `m0, …, m0 + T·BM − 1`, is safe: it only loads (rows of `Q`, `dO`, `M`, `delta` of the
head), writes nothing, and moves the invariant `KVInv` of the accumulators `dk`, `dv` to the
covered rows. Each trip of the loop is `kv_step`.
-/

namespace Kernels.AttnBwd

open TVA

/-! ## Footprints and values -/

theorem eIdx_eq (d : Inst) (b r c : Nat) :
    eIdx d b r c = b * d.N_CTX * d.HEAD_DIM + r * d.HEAD_DIM + c := by unfold eIdx; ring

theorem rIdx_eq (d : Inst) (b r : Nat) : rIdx d b r = b * d.N_CTX + r := rfl

theorem elem_lt {d : Inst} {b r c : Nat} (hb : b < G d) (hr : r < d.N_CTX) (hc : c < d.HEAD_DIM) :
    b * d.N_CTX * d.HEAD_DIM + r * d.HEAD_DIM + c < G d * d.N_CTX * d.HEAD_DIM := by
  have h1 : (r + 1) * d.HEAD_DIM ≤ d.N_CTX * d.HEAD_DIM := Nat.mul_le_mul_right _ hr
  have h2 : (b + 1) * d.N_CTX * d.HEAD_DIM ≤ G d * d.N_CTX * d.HEAD_DIM :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hb)
  have e1 : (r + 1) * d.HEAD_DIM = r * d.HEAD_DIM + d.HEAD_DIM := by ring
  have e2 : (b + 1) * d.N_CTX * d.HEAD_DIM = b * d.N_CTX * d.HEAD_DIM + d.N_CTX * d.HEAD_DIM := by ring
  omega

theorem row_lt {d : Inst} {b r : Nat} (hb : b < G d) (hr : r < d.N_CTX) :
    b * d.N_CTX + r < G d * d.N_CTX := by
  have h2 : (b + 1) * d.N_CTX ≤ G d * d.N_CTX := Nat.mul_le_mul_right _ hb
  have e2 : (b + 1) * d.N_CTX = b * d.N_CTX + d.N_CTX := by ring
  omega

theorem q_sub (d : Inst) : d.Q.elems (tSet d) ⊆ inBytes d := fun _ h => by
  unfold inBytes; simp only [Set.mem_union]; tauto
theorem k_sub (d : Inst) : d.K.elems (tSet d) ⊆ inBytes d := fun _ h => by
  unfold inBytes; simp only [Set.mem_union]; tauto
theorem v_sub (d : Inst) : d.V.elems (tSet d) ⊆ inBytes d := fun _ h => by
  unfold inBytes; simp only [Set.mem_union]; tauto
theorem do_sub (d : Inst) : d.DO.elems (tSet d) ⊆ inBytes d := fun _ h => by
  unfold inBytes; simp only [Set.mem_union]; tauto
theorem m_sub (d : Inst) : d.M.elems (rSet d) ⊆ inBytes d := fun _ h => by
  unfold inBytes; simp only [Set.mem_union]; tauto
theorem dd_sub (d : Inst) : d.D.elems (rSet d) ⊆ inBytes d := fun _ h => by
  unfold inBytes; simp only [Set.mem_union]; tauto

/-- A read of an input element agrees with the initial memory's. -/
theorem read_agree {d : Inst} {m init : Mem} (hmem : ∀ x ∈ inBytes d, m x = init x) {dt : DType}
    (Bf : Buf dt) {S : Set Nat} (hS : Bf.elems S ⊆ inBytes d) {e : Nat} (he : e ∈ S) {x y : dt.Val}
    (h1 : m.read dt (Bf.ptr ↑e).addr = some x) (h2 : init.read dt (Bf.ptr ↑e).addr = some y) : x = y := by
  have : m.read dt (Bf.ptr ↑e).addr = init.read dt (Bf.ptr ↑e).addr :=
    Mem.read_congr fun z hz => hmem z (hS (Bf.span_sub_elems he hz))
  rw [h1, h2] at this
  exact Option.some.inj this

/-- `tl.where(mask, p, 0.0)` with `mask = offs_m[None, :] >= offs_n[:, None]` (line 297), or
`p` without `MASK`. -/
theorem mask_kv {R Cc : Nat} (MASK : Bool) {x0 y0 : I32} {a0 b0 : Nat} (hx : x0.toInt = a0)
    (hy : y0.toInt = b0) (hxR : a0 + R < 2 ^ 31) (hyC : b0 + Cc < 2 ^ 31) (p0 : Tensor [R, Cc] F32)
    (r : Fin R) (i : Fin Cc) :
    (if MASK = true then
        tl.where (((x0 + tl.arange 0 R : Tensor [R] I32).col Cc).le ((y0 + tl.arange 0 Cc : Tensor [Cc] I32).row R))
          p0 (tl.full [R, Cc] (FloatV.litF32 (FloatV.ofPat 0)))
      else p0).get2 r i = if (!MASK || decide (a0 + r ≤ b0 + i)) then p0.get2 r i else FloatV.ofPat 0 := by
  cases MASK with
  | false => simp
  | true =>
    simp only [ite_true, Bool.not_true, Bool.false_or]
    show (if IntV.le (x0 + (tl.arange 0 R : Tensor [R] I32).get1 r) (y0 + (tl.arange 0 Cc : Tensor [Cc] I32).get1 i)
      then p0.get2 r i else FloatV.litF32 (FloatV.ofPat 0)) = _
    have h1 : (x0 + (tl.arange 0 R : Tensor [R] I32).get1 r).toInt = ((a0 + r : Nat) : Int) := rowsAt_add hx hxR r
    have h2 : (y0 + (tl.arange 0 Cc : Tensor [Cc] I32).get1 i).toInt = ((b0 + i : Nat) : Int) := rowsAt_add hy hyC i
    rw [IntV.le_eq, h1, h2, f32_zero_lit]
    simp only [Nat.cast_le]

/-- `offs_n = curr_n + tl.arange(0, BLOCK_N2)` and `tl.where(mask, p, 0.0)` with
`mask = offs_m[:, None] >= offs_n[None, :]` (lines 349–351), or `p` without `MASK`. -/
theorem mask_q {R Cc : Nat} (MASK : Bool) {x0 y0 : I32} {on : Tensor [Cc] I32} {a0 b0 : Nat}
    (hx : x0.toInt = a0) (hy : y0.toInt = b0) (hxR : a0 + R < 2 ^ 31) (hyC : b0 + Cc < 2 ^ 31)
    (p0 : Tensor [R, Cc] F32) (r : Fin R) (j : Fin Cc) :
    (if MASK = true then
        tl.where ((((if MASK = true then y0 + tl.arange 0 Cc else on) : Tensor [Cc] I32).row R).le
          ((x0 + tl.arange 0 R : Tensor [R] I32).col Cc)) p0 (tl.full [R, Cc] (FloatV.litF32 (FloatV.ofPat 0)))
      else p0).get2 r j = if (!MASK || decide (b0 + j ≤ a0 + r)) then p0.get2 r j else FloatV.ofPat 0 := by
  cases MASK with
  | false => simp
  | true =>
    simp only [ite_true, Bool.not_true, Bool.false_or]
    show (if IntV.le (y0 + (tl.arange 0 Cc : Tensor [Cc] I32).get1 j) (x0 + (tl.arange 0 R : Tensor [R] I32).get1 r)
      then p0.get2 r j else FloatV.litF32 (FloatV.ofPat 0)) = _
    have h1 : (x0 + (tl.arange 0 R : Tensor [R] I32).get1 r).toInt = ((a0 + r : Nat) : Int) := rowsAt_add hx hxR r
    have h2 : (y0 + (tl.arange 0 Cc : Tensor [Cc] I32).get1 j).toInt = ((b0 + j : Nat) : Int) := rowsAt_add hy hyC j
    rw [IntV.le_eq, h1, h2, f32_zero_lit]
    simp only [Nat.cast_le]

/-- `.to(tl.float32)` of a `float32` tensor (lines 307, 353): the same value. -/
theorem conv_same_ex (P : Platform) (x : F32) : ∃ r, ConvRel P .f32 .f32 x r :=
  ⟨x, by show FloatRel P _ _ x x; unfold FloatRel; rw [if_pos rfl]⟩

theorem conv_same {P : Platform} {x r : F32} (h : ConvRel P .f32 .f32 x r) : r = x := by
  change FloatRel P _ _ x r at h
  unfold FloatRel at h
  rw [if_pos rfl] at h
  exact FloatV.ext_pat h

theorem std32 : Math.Float.Format.binary32.Std ∧ Math.Float.Format.binary16.Std ∧
    tl.RoundOk Math.Float.Format.binary32 Math.Float.Format.binary16 .rn :=
  ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp, Or.inl rfl⟩

/-- The loop state of `_attn_bwd_dkdv`: `dk`, `dv`, `offs_m`, `curr_m`, `qT_ptrs`, `do_ptrs`. -/
abbrev KVState (d : Inst) (BM : Nat) : Type :=
  Tensor [128, d.HEAD_DIM] F32 × Tensor [128, d.HEAD_DIM] F32 × Tensor [BM] I32 × I32 ×
    Tensor [d.HEAD_DIM, BM] (Ptr .f16) × Tensor [BM, d.HEAD_DIM] (Ptr .f16)

/-- The loop invariant before trip `t`: memory unchanged, `curr_m = m0 + t·BM`, the pointer tiles
at rows `m0 + t·BM + i`, and `KVInv` over the rows covered so far. -/
def KVLoop (d : Inst) (init m : Mem) (b p : Nat) (C : Finset ℕ) (T0 m0 BM : Nat) (t : Nat)
    (st : KVState d BM) (m' : Mem) : Prop :=
  m' = m ∧ st.2.2.2.1.toInt = ((m0 + t * BM : Nat) : Int) ∧
    st.2.2.2.2.1 = Tensor.of2 (fun (c : Fin d.HEAD_DIM) (i : Fin BM) =>
      d.Q.ptr ↑(b * d.N_CTX * d.HEAD_DIM + (m0 + t * BM + i) * d.HEAD_DIM + c)) ∧
    st.2.2.2.2.2 = Tensor.of2 (fun (i : Fin BM) (c : Fin d.HEAD_DIM) =>
      d.DO.ptr ↑(b * d.N_CTX * d.HEAD_DIM + (m0 + t * BM + i) * d.HEAD_DIM + c)) ∧
    KVInv d init b p (C ∪ Finset.Ico m0 (m0 + t * BM)) (T0 + t) st.1 st.2.1

/-! ## The helper -/

/-- **`_attn_bwd_dkdv`** from program `p` of head `b` (lines 266–315): with `start_n = 128p`,
`start_m = m0`, `num_steps = T` and block `BM ∈ {16, 32}`, every lane `Allowed` exactly when the
mask keeps it, it is safe, writes nothing and leaves `KVInv` over the rows `C ∪ [m0, m0 + T·BM)`. -/
theorem safe_dkdv {E : Env} (hdot : Rules.Num.DotF16 E.P) (hdt : Rules.Num.DotTotal E.P dotCfg)
    (hexp : Rules.Num.Exp2F32 E.P) (hmt : Rules.Num.MathTotal E.P .exp2 .binary32)
    {d : Inst} (hd : d.ok) {init m : Mem} (hmem : ∀ x ∈ inBytes d, m x = init x)
    (hQv : d.Q.Valid m) (hDOv : d.DO.Valid m) (hMv : d.M.Valid m) (hDv : d.D.Valid m)
    {b p : Nat} (hb : b < G d) (hp : 128 * p + 128 ≤ d.N_CTX)
    {BM : Nat} (hBM : BM = 16 ∨ BM = 32) (MASK : Bool)
    {sn s0 ns : I32} {m0 T : Nat} (hsn : sn.toInt = ((128 * p : Nat) : Int)) (hs0 : s0.toInt = (m0 : Int))
    (hns : ns.toInt = (T : Int)) (hrange : m0 + T * BM ≤ d.N_CTX)
    (hal : ∀ (r : Fin 128) (mm : Nat), m0 ≤ mm → mm < m0 + T * BM →
      decide (Allowed d (128 * p + r) mm) = (!MASK || decide (128 * p + r ≤ mm)))
    {C : Finset ℕ} (hC : ∀ x ∈ C, x < m0 ∨ m0 + T * BM ≤ x) (hCN : C ⊆ Finset.range d.N_CTX)
    {T0 : Nat} (hT : T0 + T < 1024)
    {k v : Tensor [128, d.HEAD_DIM] F16}
    (hkv : ∀ hv : HeadV, hv.Holds d init b → ∀ (r : Fin 128) (c : Fin d.HEAD_DIM),
        k.get2 r c = hv.k (128 * p + r) c ∧ v.get2 r c = hv.v (128 * p + r) c)
    {dk dv : Tensor [128, d.HEAD_DIM] F32} (hinv : KVInv d init b p C T0 dk dv)
    (sm : F32) (H' N' : I32) :
    Safe E (inBytes d) ∅
      (fun r m' => m' = m ∧ KVInv d init b p (C ∪ Finset.Ico m0 (m0 + T * BM)) (T0 + T) r.1 r.2) m
      (attn_bwd_dkdv BM 128 d.HEAD_DIM dk dv (d.Q.ptr ↑(b * d.N_CTX * d.HEAD_DIM)) k v sm
        (d.DO.ptr ↑(b * d.N_CTX * d.HEAD_DIM)) (d.M.ptr ↑(b * d.N_CTX)) (d.D.ptr ↑(b * d.N_CTX))
        (IntV.ofNat d.HEAD_DIM) (IntV.ofNat 1) H' N' sn s0 ns MASK) := by
  have hHD := hd_le hd
  have hHD0 := hd_pos hd
  have hsize := hd.size_lt
  have hNle := hd.n_le
  have hG0 : 0 < G d := by omega
  have hNHD : d.N_CTX * d.HEAD_DIM < 2 ^ 31 := by
    have : d.N_CTX * d.HEAD_DIM ≤ G d * d.N_CTX * d.HEAD_DIM := by
      rw [Nat.mul_assoc]; exact Nat.le_mul_of_pos_left _ hG0
    omega
  have hBM32 : BM ≤ 32 := by omega
  unfold attn_bwd_dkdv
  have hRok : tl.RangeOk (IntV.ofInt 0 : I32) ns (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [IntV.I32.toInt_ofInt (by norm_num), IntV.I32.toInt_ofInt (by norm_num), hns]
    norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt ns.toInt (IntV.ofInt 1 : I32).toInt = T := by
    rw [IntV.I32.toInt_ofInt (by norm_num), IntV.I32.toInt_ofInt (by norm_num), hns]
    simp [tl.tripCount]
  have hr0 : RowsAt (s0 + tl.arange 0 BM : Tensor [BM] I32) m0 := rowsAt_add hs0 (by omega)
  have hk0 : RowsAt (tl.arange 0 d.HEAD_DIM : Tensor [d.HEAD_DIM] I32) 0 := rowsAt_arange0 (by omega)
  have hfit0 : (m0 + BM) * d.HEAD_DIM < 2 ^ 31 := by
    have h1 : m0 + BM ≤ 2 ^ 14 + 32 := by
      have : m0 ≤ m0 + T * BM := Nat.le_add_right _ _
      omega
    have : (m0 + BM) * d.HEAD_DIM ≤ (2 ^ 14 + 32) * 256 := Nat.mul_le_mul h1 hHD
    omega
  refine Safe.mono (safe_forRange (Wk := ∅) hRok (KVLoop d init m b p C T0 m0 BM) (fun _ => ∅)
    (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, ?_, ?_, ?_, ?_⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · show s0.toInt = _; rw [hs0]; simp
  · refine (tile_cr d.Q _ hr0 hk0 hfit0).trans ?_
    simp only [zero_mul, add_zero, Nat.sub_zero]
  · refine (tile_rc d.DO _ hr0 hk0 hfit0).trans ?_
    simp only [zero_mul, add_zero, Nat.sub_zero]
  · simp only [zero_mul, add_zero, Finset.Ico_self, Finset.union_empty]
    exact hinv
  · -- one trip
    rw [htc]
    rintro t ht ⟨dk1, dv1, om, cm, qp, dp⟩ m₁ ⟨rfl, hcm, hqp, hdp, hinv1⟩
    simp only at hcm hqp hdp hinv1
    subst hqp hdp
    simp only []
    have hmtT : m0 + t * BM + BM ≤ m0 + T * BM := by
      have : (t + 1) * BM ≤ T * BM := Nat.mul_le_mul_right _ ht
      rw [Nat.add_mul] at this; omega
    have hmtN : m0 + t * BM + BM ≤ d.N_CTX := by omega
    set mt := m0 + t * BM with hmt_def
    have hrows : RowsAt (cm + tl.arange 0 BM : Tensor [BM] I32) mt := rowsAt_add hcm (by omega)
    -- line 289: `qT`
    refine safe_load_buf d.Q (g := fun (ix : Idx [d.HEAD_DIM, BM]) => b * d.N_CTX * d.HEAD_DIM + (mt + (ix.2.1 : Nat)) * d.HEAD_DIM + (ix.1 : Nat))
      (en := fun _ => true) (Buf.addrs_of_get_all d.Q fun ⟨c, i, ⟨⟩⟩ => rfl) ?_ hQv ?_
      fun qT hqT _ => ?_
    · rintro ⟨c, i, ⟨⟩⟩ _
      have := elem_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega) c.2
      have := hd.q_len
      simp only; omega
    · refine (d.Q.elems_mono ?_).trans (q_sub d)
      rintro _ ⟨⟨c, i, ⟨⟩⟩, -, rfl⟩
      exact elem_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega) c.2
    -- line 292: `m`
    refine safe_load_buf d.M (g := fun (ix : Idx [BM]) => b * d.N_CTX + (mt + (ix.1 : Nat))) (en := fun _ => true)
      (Buf.addrs_of_get_all d.M fun ⟨i, ⟨⟩⟩ =>
        congrArg (fun (x : Tensor [BM] (Ptr .f32)) => x.get (i, ())) (vec_r d.M (b * d.N_CTX) hrows)) ?_ hMv ?_
      fun mv hmv _ => ?_
    · rintro ⟨i, ⟨⟩⟩ _
      have := row_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega)
      have := hd.m_len
      simp only; omega
    · refine (d.M.elems_mono ?_).trans (m_sub d)
      rintro _ ⟨⟨i, ⟨⟩⟩, -, rfl⟩
      exact row_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega)
    -- line 293
    refine safe_dot (hdt _ _ _) fun qkT hqkT => ?_
    -- line 294
    refine safe_subc fun xs hxs => ?_
    refine safe_math (fun _ => hmt _) fun p0 hp0 => ?_
    -- line 299: `do`
    refine safe_load_buf d.DO (g := fun (ix : Idx [BM, d.HEAD_DIM]) => b * d.N_CTX * d.HEAD_DIM + (mt + (ix.1 : Nat)) * d.HEAD_DIM + (ix.2.1 : Nat))
      (en := fun _ => true) (Buf.addrs_of_get_all d.DO fun ⟨i, c, ⟨⟩⟩ => rfl) ?_ hDOv ?_
      fun dO hdO _ => ?_
    · rintro ⟨i, c, ⟨⟩⟩ _
      have := elem_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega) c.2
      have := hd.do_len
      simp only; omega
    · refine (d.DO.elems_mono ?_).trans (do_sub d)
      rintro _ ⟨⟨i, c, ⟨⟩⟩, -, rfl⟩
      exact elem_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega) c.2
    -- line 302
    refine safe_cvt std32 fun ppT hppT => ?_
    -- line 303
    refine safe_dot_add (hdt _ _ _) fun dv' hdv' => ?_
    -- line 305: `Di`
    refine safe_load_buf d.D (g := fun (ix : Idx [BM]) => b * d.N_CTX + (mt + (ix.1 : Nat))) (en := fun _ => true)
      (Buf.addrs_of_get_all d.D fun ⟨i, ⟨⟩⟩ =>
        congrArg (fun (x : Tensor [BM] (Ptr .f32)) => x.get (i, ())) (vec_r d.D (b * d.N_CTX) hrows)) ?_ hDv ?_
      fun Di hDi _ => ?_
    · rintro ⟨i, ⟨⟩⟩ _
      have := row_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega)
      have := hd.d_len
      simp only; omega
    · refine (d.D.elems_mono ?_).trans (dd_sub d)
      rintro _ ⟨⟨i, ⟨⟩⟩, -, rfl⟩
      exact row_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega)
    -- line 307
    refine safe_dot (hdt _ _ _) fun dpT0 hdpT0 => ?_
    refine safe_convert (fun _ => conv_same_ex _ _) fun dpT hdpT => ?_
    -- line 308
    refine safe_subc fun ys hys => ?_
    refine safe_mul fun dsT hdsT => ?_
    -- line 309
    refine safe_cvt std32 fun dsT16 hdsT16 => ?_
    -- line 310
    refine safe_dot_add (hdt _ _ _) fun dk' hdk' => ?_
    -- lines 312–314
    refine safe_ret ⟨rfl, ?_, ?_, ?_, ?_⟩
    · show (cm + IntV.ofNat BM).toInt = _
      rw [IntV.I32.add_nat hcm (i32_nat (by omega)) (by omega)]
      congr 1; rw [Nat.add_mul]; ring
    · show _ + _ = _
      rw [of2_add_bc d.Q _ (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega))
        (by nlinarith))]
      congr 1; funext c i; congr 2; rw [Nat.add_mul]; ring
    · show _ + _ = _
      rw [of2_add_bc d.DO _ (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega))
        (by nlinarith))]
      congr 1; funext i c; congr 2; rw [Nat.add_mul]; ring
    · show KVInv d init b p (C ∪ Finset.Ico m0 (m0 + (t + 1) * BM)) (T0 + (t + 1)) dk' dv'
      rw [cover_succ, ← hmt_def, ← Nat.add_assoc]
      refine kv_step hdot hexp hd hBM hmtN hp (cover_apart hC ht) (cover_sub hCN (by omega))
        (by omega) (k := k) (v := v) (qT := qT) (dO := dO) (mv := mv) (Di := Di)
        ?_ hqkT (fun r i => hxs (r, i, ())) (fun r i => hp0 (r, i, ())) (pT := _)
        (fun r i => ?_) (fun r i => hppT (r, i, ())) hdpT0 (fun r i => conv_same (hdpT (r, i, ())))
        (fun r i => hys (r, i, ())) (fun r i => hdsT (r, i, ())) (fun r i => hdsT16 (r, i, ()))
        hdv' hdk' hinv1
      · intro hv hH
        refine ⟨hkv hv hH, fun c i => ?_, fun i c => ?_, fun i => ⟨?_, ?_⟩⟩
        · have hh := ((hH (mt + i) (by omega)).1 c c.2).1
          rw [eIdx_eq] at hh
          exact read_agree hmem d.Q (q_sub d) (elem_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega) c.2)
            (hqT (c, i, ()) rfl) hh
        · have hh := ((hH (mt + i) (by omega)).1 c c.2).2.2.2
          rw [eIdx_eq] at hh
          exact read_agree hmem d.DO (do_sub d) (elem_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega) c.2)
            (hdO (i, c, ()) rfl) hh
        · have hh := (hH (mt + i) (by omega)).2.1
          rw [rIdx_eq] at hh
          exact read_agree hmem d.M (m_sub d) (row_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega))
            (hmv (i, ()) rfl) hh
        · have hh := (hH (mt + i) (by omega)).2.2
          rw [rIdx_eq] at hh
          exact read_agree hmem d.D (dd_sub d) (row_lt hb (show mt + (i : Nat) < d.N_CTX by have := i.2; omega))
            (hDi (i, ()) rfl) hh
      · rw [mask_kv MASK hsn hcm (by omega) (by omega) p0 r i,
          hal r (mt + i) (by omega) (by have := i.2; omega)]
  · -- after the loop
    rw [htc]
    rintro ⟨dk1, dv1, om, cm, qp, dp⟩ m₁ ⟨rfl, -, -, -, hinv1⟩
    exact safe_ret ⟨rfl, hinv1⟩

end Kernels.AttnBwd
