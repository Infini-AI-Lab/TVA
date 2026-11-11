import Proof.Stages

/-!
# One trip of the loop (lines 55–69)

`safe_trip`: the trip reads block-table entry `t` and the page's keys and values, writes nothing,
and the invariant `Inv` holds after it.
-/
namespace Claims.PagedAttention

open TVA

/-- **One trip of the loop is safe** and keeps the invariant: it reads inside `R`, writes nothing,
and the state after it satisfies `Inv (t+1)` when the one before satisfied `Inv t`. -/
theorem safe_trip {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init m₁ : Mem}
    (hvk : d.k_cache.Valid init) (hvv : d.v_cache.Valid init) (hvb : d.block_tables.Valid init)
    {s h t : Nat} (hs : s < d.num_seqs) (hh : h < d.num_heads)
    (hsok : SeqOk d init s) (ht : t < nblocks d init s)
    {seq kv_head seq_len : I32} (hseq : seq.toInt = s) (hkv : kv_head.toInt = h / qpk d)
    (hsl : seq_len.toInt = seqLen d init s)
    {q : Tensor [d.HEAD_DIM] F32}
    (hq : SeqVals d init s h → ∀ (e : Nat) (he : e < d.HEAD_DIM),
      (q.get1 ⟨e, he⟩).isFinite = true ∧
        (q.get1 ⟨e, he⟩).toReal = (val16 init d.q (qIdx d s h e) : ℝ))
    {j : I32} (hj : j.toInt = t)
    {st : Tensor [d.HEAD_DIM] F32 × F32 × F32} (hI : Inv d init s h t st m₁) :
    Safe E (R d init s h) ∅ (fun st' m' => Inv d init s h (t + 1) st' m') m₁
      (trip d seq kv_head seq_len q j st) := by
  have hm : m₁ = init := hI.1
  subst hm
  obtain ⟨acc, l_i, m_i⟩ := st
  unfold trip
  dsimp only
  obtain ⟨n, hn_read, hn0, hn_le, hn22, hbt⟩ := hsok
  have hnl : seqLen d m₁ s = n.toInt.toNat := by
    unfold seqLen val32
    rw [hn_read]
  have hn_nat : (n.toInt.toNat : Int) = n.toInt := Int.toNat_of_nonneg hn0
  have hP0 : 0 < d.PAGE_SIZE := by obtain ⟨b, hb, he⟩ := hd.page_size; rw [he]; positivity
  have htn : t * d.PAGE_SIZE < seqLen d m₁ s := by
    unfold nblocks at ht
    have h1 := (Nat.le_div_iff_mul_le hP0).mp (Nat.succ_le_of_lt ht)
    rw [Nat.succ_mul] at h1
    omega
  have hbt' := hbt t (by rw [hnl] at htn; omega)
  obtain ⟨pgv, hpg_read, hpg0, hpgN⟩ := hbt'
  have ht_max : t < d.max_pages := by
    have h1 : t * d.PAGE_SIZE < d.max_pages * d.PAGE_SIZE := by
      have := hn_le
      rw [hnl] at htn; omega
    exact Nat.lt_of_mul_lt_mul_right h1
  have hS31 : s < 2 ^ 31 := by have := hd.seqs_lt; omega
  have ht31 : t < 2 ^ 31 := by
    have := hd.max_pages_le; omega
  have hbs : (seq * (IntV.ofNat d.stride_bs : I32)).toInt = (s * d.stride_bs : ℕ) :=
    IntV.I32.mul_nat hseq (IntV.I32.toInt_ofNat (by have := hd.strides_lt; omega)) (by
      have h1 := hd.prods_lt.2.2.2.2.1
      have h2 : s * d.stride_bs ≤ (d.num_seqs - 1) * d.stride_bs :=
        Nat.mul_le_mul_right _ (by omega)
      omega)
  have hptr_bt : d.block_tables.ptr 0 + seq * (IntV.ofNat d.stride_bs : I32) + j =
      d.block_tables.ptr (btIdx d s t : Int) := by
    rw [Ptr.add_int, Ptr.add_int, hbs, hj, Buf.ptr_offset, Buf.ptr_offset]
    congr 1; unfold btIdx; push_cast; ring
  rw [hptr_bt]
  refine safe_load1_buf d.block_tables (i := btIdx d s t) (hd.bt_len s t hs ht_max) hvb ?_
    fun bv hbv => ?_
  · refine Set.Subset.trans ?_ (fun x hx => Or.inl (Or.inl (Or.inr hx)) : d.block_tables.elems (btSet d m₁ s) ⊆ R d m₁ s h)
    intro x hx
    exact ⟨btIdx d s t, ⟨t, by rw [hnl] at htn; omega, rfl⟩, hx⟩
  · have hbv_eq : bv = pgv := by rw [hbv] at hpg_read; exact Option.some.inj hpg_read
    subst hbv_eq
    -- the page, widened
    have hcv : ∀ (x : I32), ∃ r : I64, ConvRel E.P .i32 .i64 x r := fun x =>
      ⟨_, ConvRel.iff_cast_i32_i64.2 rfl⟩
    refine safe_convert (fun _ => hcv _) fun pgt hpgt => ?_
    have hpage : (pgt.item).toInt = bv.toInt := ConvRel.toInt_i32_i64 (hpgt ())
    -- the mask
    have hP31 : d.PAGE_SIZE ≤ 2 ^ 8 := by
      obtain ⟨b', hb', he⟩ := hd.page_size
      rw [he]; exact Nat.pow_le_pow_right (by norm_num) hb'
    have htn' : t * d.PAGE_SIZE < n.toInt.toNat := by rw [← hnl]; exact htn
    have htn22 : t * d.PAGE_SIZE < 2 ^ 22 := by omega
    have hjP : (j * (IntV.ofNat d.PAGE_SIZE : I32)).toInt = (t * d.PAGE_SIZE : ℕ) :=
      IntV.I32.mul_nat (x := t) (y := d.PAGE_SIZE) hj (IntV.I32.toInt_ofNat (by omega)) (by omega)
    have hmaskl : ∀ r : Fin d.PAGE_SIZE,
        ((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE).ltS seq_len).get1 r =
          decide (t * d.PAGE_SIZE + r.val < seqLen d m₁ s) := by
      intro r
      have hl := IntV.I32.toInt_sadd_arange (c := j * (IntV.ofNat d.PAGE_SIZE : I32)) hjP
        (by show t * d.PAGE_SIZE + d.PAGE_SIZE ≤ 2 ^ 31; omega) r
      have hl' : (((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE)).get1 r).toInt =
          ((t * d.PAGE_SIZE + r.val : Nat) : Int) := hl
      show (IntV.lt (((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE)).get1 r) seq_len) = _
      rw [IntV.lt_eq, hl', hsl]
      norm_cast
    -- the widened strides and offsets
    have hcvt0 : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], ∃ r : F16, ConvRel E.P .f32 .f16 ((tl.full [d.PAGE_SIZE, d.HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0)) : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32).get i) r := fun _ => conv_zero_exists E.P
    have hcvt : ∀ {s' : Shape} (x : Tensor s' I32), ∀ i, ∃ r : I64, ConvRel E.P .i32 .i64 (x.get i) r :=
      fun x i => ⟨_, ConvRel.iff_cast_i32_i64.2 rfl⟩
    refine safe_convert (hcvt _) fun At hAt => ?_
    have hA : (At.item).toInt = d.stride_kp := by
      have h1 := ConvRel.toInt_i32_i64 (hAt ())
      refine h1.trans ?_
      exact IntV.I32.toInt_ofNat hd.strides_lt.2.2.1
    refine safe_convert (hcvt _) fun c1t hc1t => ?_
    have hc1 : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], (c1t.get i).toInt = ((i.1.val * d.stride_kt : Nat) : Int) := by
      intro i
      rw [ConvRel.toInt_i32_i64 (hc1t i)]
      rcases i with ⟨r, e, ⟨⟩⟩
      have hr : ((tl.arange 0 d.PAGE_SIZE).get1 r).toInt = r.val :=
        IntV.I32.toInt_tl_arange (le_trans hP31 (by norm_num)) r
      have hrt : r.val * d.stride_kt < 2 ^ 31 := by
        have h1 : (d.PAGE_SIZE - 1) * d.stride_kt < 2 ^ 31 := hd.prods_lt.2.2.2.2.2.1
        have h2 : r.val ≤ d.PAGE_SIZE - 1 := by have := r.isLt; omega
        exact lt_of_le_of_lt (Nat.mul_le_mul_right _ h2) h1
      exact IntV.I32.mul_nat hr (IntV.I32.toInt_ofNat hd.strides_lt.2.2.2.1) hrt
    refine safe_convert (hcvt _) fun Bt hBt => ?_
    have hgk : h / qpk d < d.num_kv_heads := by
      have hH : d.num_heads = d.num_kv_heads * qpk d := by
        unfold qpk; exact (Nat.mul_div_cancel' hd.kv_dvd).symm
      have hq0 : 0 < qpk d := by
        by_contra hq0; have : qpk d = 0 := by omega
        rw [this] at hH; omega
      rw [Nat.div_lt_iff_lt_mul hq0]; rw [hH] at hh; exact hh
    have hkvh : (kv_head * (IntV.ofNat d.stride_kh : I32)).toInt = ((h / qpk d * d.stride_kh : Nat) : Int) := by
      refine IntV.I32.mul_nat (x := h / qpk d) (y := d.stride_kh) (by rw [hkv]; push_cast; rfl)
        (IntV.I32.toInt_ofNat hd.strides_lt.2.2.2.2.1) ?_
      have h1 : (d.num_kv_heads - 1) * d.stride_kh < 2 ^ 31 := hd.prods_lt.2.2.2.2.2.2
      have h2 : h / qpk d * d.stride_kh ≤ (d.num_kv_heads - 1) * d.stride_kh :=
        Nat.mul_le_mul_right _ (by omega)
      omega
    have hB : (Bt.item).toInt = ((h / qpk d * d.stride_kh : Nat) : Int) := by
      have h1 := ConvRel.toInt_i32_i64 (hBt ())
      exact h1.trans hkvh
    refine safe_convert (hcvt _) fun c3t hc3t => ?_
    have hc3 : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], (c3t.get i).toInt = ((i.2.1.val : Nat) : Int) := by
      intro i
      rw [ConvRel.toInt_i32_i64 (hc3t i)]
      rcases i with ⟨r, e, ⟨⟩⟩
      have hD31 : d.HEAD_DIM ≤ 2 ^ 31 := by
        obtain ⟨a, ha, he⟩ := hd.head_dim
        rw [he]; calc 2 ^ a ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) ha
          _ ≤ 2 ^ 31 := by norm_num
      simp only [Tensor.get_row]
      exact IntV.I32.toInt_tl_arange hD31 e
    have hpgval : val32 m₁ d.block_tables (btIdx d s t) = bv.toInt := by
      unfold val32; rw [hpg_read]
    set pgN := bv.toInt.toNat with hpgN_def
    have hpgN : (pgN : Int) = bv.toInt := Int.toNat_of_nonneg hpg0
    have hpgN_lt : pgN < d.num_pages := by omega
    have hpgN31 : pgN < 2 ^ 31 := by
      have := IntV.sfits_toInt bv
      have h2 : bv.toInt < 2 ^ 31 := (IntV.sfits_toInt bv).2
      omega
    refine safe_convert (hcvt0 : ∀ i, ∃ r : F16, ConvRel E.P .f32 .f16 ((tl.full [d.PAGE_SIZE, d.HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0)) : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32).get i) r) fun oth hoth => ?_
    have hoth0 : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], (oth.get i).isFinite = true ∧ (oth.get i).toReal = 0 :=
      fun i => Num.conv_zero_f16 E.P (hoth i)
    have hgk' : h / qpk d < d.num_kv_heads := hgk
    have hbvq : bv.toInt = pgN := hpgN.symm
    have hpage' : (pgt.item).toInt = pgN := by rw [hpage, hbvq]
    -- the lanes of `kv_offs`
    have hlane : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
        ((pgt.item * At.item + c1t + Bt.item + c3t).get i).toInt =
          ((kvIdx d pgN i.1.val (h / qpk d) i.2.1.val : Nat) : Int) := fun i =>
      kv_offs_lane hd hpgN31 hgk' hpage' hA hB hc1 hc3 i
    have hAddr : ∀ (b : Buf .f16), b.len = b.len →
        ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], 
          (b.ptr 0 + (pgt.item * At.item + c1t + Bt.item + c3t)).get i =
            b.ptr (kvIdx d pgN i.1.val (h / qpk d) i.2.1.val : Int) := by
      intro b _ i
      refine (Tensor.get_ptrAdd _ _ _).trans ?_
      rw [hlane i, Buf.ptr_offset]; simp
    have hmask : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
        (((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE).ltS seq_len).col d.HEAD_DIM).get i =
          decide (t * d.PAGE_SIZE + i.1.val < seqLen d m₁ s) := by
      rintro ⟨r, e, ⟨⟩⟩
      exact hmaskl r
    have hRtok : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], t * d.PAGE_SIZE + i.1.val < seqLen d m₁ s →
        kvIdx d pgN i.1.val (h / qpk d) i.2.1.val ∈ tokSet d m₁ s h := by
      intro i hi
      refine ⟨t * d.PAGE_SIZE + i.1.val, i.2.1.val, hi, i.2.1.isLt, ?_⟩
      rw [tokElem_eq i.1.isLt (by rw [hpgval])]
    have hRtok' : d.k_cache.elems (Buf.img (s := [d.PAGE_SIZE, d.HEAD_DIM])
        (fun i => kvIdx d pgN i.1.val (h / qpk d) i.2.1.val)
        (fun i => decide (t * d.PAGE_SIZE + i.1.val < seqLen d m₁ s))) ⊆ R d m₁ s h := by
      refine Set.Subset.trans (d.k_cache.elems_mono ?_) (fun x hx => Or.inl (Or.inr hx))
      rintro _ ⟨i, hi, rfl⟩
      exact hRtok i (of_decide_eq_true hi)
    refine safe_load_buf d.k_cache (g := fun i : Idx [d.PAGE_SIZE, d.HEAD_DIM] => kvIdx d pgN i.1.val (h / qpk d) i.2.1.val)
      (en := fun i => decide (t * d.PAGE_SIZE + i.1.val < seqLen d m₁ s))
      (Buf.addrs_of_get d.k_cache hmask (fun i _ => hAddr d.k_cache rfl i)) ?_ hvk hRtok' fun kraw hkraw hkoth => ?_
    · intro i hi
      exact (hd.kv_len pgN i.1.val (h / qpk d) i.2.1.val hpgN_lt i.1.isLt hgk' i.2.1.isLt).1
    · refine safe_cvt ⟨Format.binary16_std, Format.binary32_std, Or.inl rfl⟩ fun kk hkk => ?_
      refine safe_mul fun prod hprod => ?_
      refine safe_sum1c (by obtain ⟨a, -, he⟩ := hd.head_dim; show 0 < d.HEAD_DIM; rw [he]; positivity) fun sums hsums => ?_
      refine safe_mul fun sc hsc => ?_
      refine safe_maxAll hP0 true fun mx hmx => ?_
      refine safe_maximum fun mn hmn => ?_
      refine safe_sub fun xa hxa => ?_
      refine safe_math (fn := .exp) (fun i => hP.mathTotal _ _ _) fun al hal => ?_
      refine safe_sub fun xl hxl => ?_
      refine safe_math (fn := .exp) (fun i => hP.mathTotal _ _ _) fun pp hpp => ?_
      refine safe_sum hP0 fun r hr => ?_
      refine safe_mul_add fun l' hl' => ?_
      refine safe_convert (hcvt0 : ∀ i, ∃ r : F16, ConvRel E.P .f32 .f16 ((tl.full [d.PAGE_SIZE, d.HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0)) : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32).get i) r) fun oth2 hoth2 => ?_
      have hRtokv : d.v_cache.elems (Buf.img (s := [d.PAGE_SIZE, d.HEAD_DIM])
          (fun i => kvIdx d pgN i.1.val (h / qpk d) i.2.1.val)
          (fun i => decide (t * d.PAGE_SIZE + i.1.val < seqLen d m₁ s))) ⊆ R d m₁ s h := by
        refine Set.Subset.trans (d.v_cache.elems_mono ?_) (fun x hx => Or.inr hx)
        rintro _ ⟨i, hi, rfl⟩
        exact hRtok i (of_decide_eq_true hi)
      refine safe_load_buf d.v_cache (g := fun i : Idx [d.PAGE_SIZE, d.HEAD_DIM] => kvIdx d pgN i.1.val (h / qpk d) i.2.1.val)
        (en := fun i => decide (t * d.PAGE_SIZE + i.1.val < seqLen d m₁ s))
        (Buf.addrs_of_get d.v_cache hmask (fun i _ => hAddr d.v_cache rfl i)) ?_ hvv hRtokv fun vraw hvraw hvoth => ?_
      · intro i hi
        exact (hd.kv_len pgN i.1.val (h / qpk d) i.2.1.val hpgN_lt i.1.isLt hgk' i.2.1.isLt).2
      · refine safe_cvt ⟨Format.binary16_std, Format.binary32_std, Or.inl rfl⟩ fun vv hvv' => ?_
        refine safe_mul fun pr hpr => ?_
        refine safe_sum0c hP0 fun pv hpv => ?_
        refine safe_acc_update d acc pv al.item (fun acc' hupd => ?_)
        refine safe_ret ?_
        refine ⟨rfl, fun hSV X hX0 hX1 hXk => ⟨fun h0 => absurd h0 (Nat.succ_ne_zero t), fun _ => ?_⟩⟩
        have hn1 : 1 ≤ seqLen d m₁ s := hSV.1
        have hn22' : seqLen d m₁ s ≤ 2 ^ 22 := by rw [hnl]; omega
        have hP256 : d.PAGE_SIZE ≤ 256 := by
          obtain ⟨b', hb', he⟩ := hd.page_size
          rw [he]; calc 2 ^ b' ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) hb'
            _ = 256 := by norm_num
        have hc1 : 1 ≤ min d.PAGE_SIZE (seqLen d m₁ s - t * d.PAGE_SIZE) := by omega
        set c := min d.PAGE_SIZE (seqLen d m₁ s - t * d.PAGE_SIZE) with hcdef
        have hcP : c ≤ d.PAGE_SIZE := min_le_left _ _
        have hcv : ∀ r : Fin d.PAGE_SIZE, r.val < c → t * d.PAGE_SIZE + r.val < seqLen d m₁ s := by
          intro r hr; omega
        have hcv' : ∀ r : Fin d.PAGE_SIZE, t * d.PAGE_SIZE + r.val < seqLen d m₁ s → r.val < c := by
          intro r hr; omega
        have hmin : min ((t + 1) * d.PAGE_SIZE) (seqLen d m₁ s) = t * d.PAGE_SIZE + c := by
          rw [Nat.succ_mul]; omega
        have hminN : min (t * d.PAGE_SIZE) (seqLen d m₁ s) = t * d.PAGE_SIZE := min_eq_left (by omega)
        have hσ0 : 0 ≤ (sigma d : ℝ) := by
          have := hd.scale.2.1
          unfold sigma
          have h1 : (0 : ℚ) ≤ d.scale.toRat := le_trans (by norm_num) this
          exact_mod_cast h1
        have hSb : ∀ k, k < seqLen d m₁ s → |Sx d m₁ s h k| ≤ X := by
          intro k hk
          have h1 : |(score d m₁ s h k : ℚ)| ≤ scoreAbs d m₁ s h k := by
            unfold score scoreAbs; exact Finset.abs_sum_le_sum_abs _ _
          have h2 : |(score d m₁ s h k : ℝ)| ≤ (scoreAbs d m₁ s h k : ℝ) := by exact_mod_cast h1
          unfold Sx
          rw [abs_mul, abs_of_nonneg hσ0]
          exact (mul_le_mul_of_nonneg_left h2 hσ0).trans (hXk k hk)
        have hTok : ∀ r e, r < d.PAGE_SIZE → tokElem d m₁ s h (t * d.PAGE_SIZE + r) e =
            kvIdx d pgN r (h / qpk d) e := fun r e hr => tokElem_eq hr (by rw [hpgval])
        have hscore0 := scores_stage hd hP hSV (hq hSV) hkraw hTok hkk hprod hsums hsc hX0 hX1 hXk
        have hneg : (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000) : F32) = FloatV.negInf := by decide
        have hsp : ∀ r : Fin d.PAGE_SIZE,
            (tl.where ((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE).ltS seq_len) sc
              (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)))).get1 r =
              if r.val < c then sc.get1 r else FloatV.negInf := by
          intro r
          show (if ((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE).ltS seq_len).get1 r = true
            then sc.get1 r else FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)) = _
          rw [hmaskl r, hneg]
          by_cases hr : r.val < c
          · rw [ite_eq_left hr]; rw [ite_eq_left (decide_eq_true (by have := hcv r hr; omega))]
          · rw [ite_eq_right hr]; rw [ite_eq_right (by have := hcv' r; simp only [decide_eq_true_eq]; intro h'; exact hr (hcv' r h'))]
        obtain ⟨hst0, hstpos⟩ := hI.2 hSV X hX0 hX1 hXk
        have hm0 : t = 0 → m_i = FloatV.negInf := by
          intro h0
          have := congrArg (fun p => p.2.2) (hst0 h0)
          simp only [st0] at this
          rw [this, hneg]
        have hmi : m_i = FloatV.negInf ∨ m_i.isFinite = true := by
          rcases Nat.eq_zero_or_pos t with h0 | h0
          · exact Or.inl (hm0 h0)
          · exact Or.inr (hstpos h0).1
        have hold : 0 < t → m_i.isFinite = true ∧ -(Num.Kx X) ≤ m_i.toReal ∧ m_i.toReal ≤ Num.Kx X ∧
            (∀ k, k < t * d.PAGE_SIZE → Sx d m₁ s h k ≤ m_i.toReal + Num.dS X) ∧
            ∃ k, k < t * d.PAGE_SIZE ∧ m_i.toReal ≤ Sx d m₁ s h k + Num.dS X := by
          intro h0
          obtain ⟨hmf, -, -, hK1, hK2, hs1, hs2, -⟩ := hstpos h0
          rw [hminN] at hs1 hs2
          exact ⟨hmf, hK1, hK2, hs1, hs2⟩
        have W := weights_stage hd hP hX0 hX1 hSb hc1 hcP hcv (sc := sc)
          (sp := tl.where ((j * (IntV.ofNat d.PAGE_SIZE : I32) + tl.arange 0 d.PAGE_SIZE).ltS seq_len) sc
              (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))))
          (xl := xl) (pp := pp) (m_i := m_i) (mx := mx) (mn := mn.item) (al := al.item) (xa := xa.item)
          (fun r hr => hscore0 r (hcv r hr)) hsp hmx hmi (hmn ()) hold hm0 (hxa ()) (hal ())
          (fun r => hxl (r, ())) (fun r => hpp (r, ()))
        obtain ⟨hmnf, hK1, hK2, hall, hatt, hal_pos, hal0, hlanes, hmasked⟩ := W
        have hN : t * d.PAGE_SIZE + c ≤ 2 ^ 22 := by rw [← hmin]; exact (min_le_right _ _).trans hn22'
        have ht1 : t + 1 ≤ 2 ^ 16 := Nat.succ_le_of_lt (lt_of_lt_of_le ht_max hd.max_pages_le)
        have hPpos : 0 < d.PAGE_SIZE := hP0
        have hz : (FloatV.ofPat 0 : F32).toReal = 0 := by
          unfold FloatV.toReal; rw [Num.zero32_rat]; simp
        -- the sum `l`
        have hlcol := lcol_stage hP hX0 hX1 hPpos hP256 hc1 hcP ht1 hN (l_i := l_i) (r := r)
          (l' := l'.item) (al := al.item) (pp := pp) (μ' := mn.item.toReal) hK1 hK2 hlanes hmasked hr
          (hl' ()) (by
            intro h0
            obtain ⟨hmf, hlf, -, hK1o, hK2o, hs1, -, hL, -⟩ := hstpos h0
            rw [hminN] at hs1 hL
            obtain ⟨hle, haf, hae, ha2⟩ := hal_pos h0
            exact ⟨m_i.toReal, hlf, hL, hle, haf, hae, ha2, hs1, hK1o, hK2o⟩) (by
            intro h0
            have := congrArg (fun p => p.2.1) (hst0 h0)
            simp only [st0] at this
            exact ⟨by rw [this]; exact Num.lit0, hal0 h0⟩)
        have hNle : t * d.PAGE_SIZE + c ≤ seqLen d m₁ s := by rw [← hmin]; exact min_le_right _ _
        -- the values
        have hvst := v_stage hSV hvraw hvoth (fun i => Num.conv_zero_f16 E.P (hoth2 i)) hTok hvv'
        have hVx : ∀ e k, e < d.HEAD_DIM → k < seqLen d m₁ s → |Vx d m₁ s h e k| ≤ 2 ^ 15 := by
          intro e k he hk
          obtain ⟨-, -, hvb⟩ := hSV.2.2.1 k e hk he
          have h3 : ((|val16 m₁ d.v_cache (tokElem d m₁ s h k e)| : ℚ) : ℝ) ≤ ((2 ^ 15 : ℚ) : ℝ) := by
            exact_mod_cast hvb
          push_cast at h3
          unfold Vx; exact h3
        have hpfin : ∀ i : Fin d.PAGE_SIZE, i.val < c → (pp.get1 i).isFinite = true ∧
            |(pp.get1 i).toReal| ≤ 2 := fun i hi => ⟨(hlanes i hi).1, (hlanes i hi).2.1⟩
        have hblock := Num.acc_block hP hPpos hP256 hcP pp vv pr pv hpfin hmasked
          (fun i e => ⟨(hvst i e).1, fun hi => ((hvst i e).2 (hcv i hi)).2⟩)
          (fun i e => hpr (i, e, ())) hpv
        have hacol : ∀ (e : Nat) (he : e < d.HEAD_DIM), (acc'.get1 ⟨e, he⟩).isFinite = true ∧
            Num.InvA X (Sx d m₁ s h) (Vx d m₁ s h e) (t * d.PAGE_SIZE + c) (t + 1)
              mn.item.toReal (acc'.get1 ⟨e, he⟩).toReal := by
          intro e he
          refine acol_stage hP hX0 hX1 hc1 hcP hP256 ht1 hN (x := acc.get1 ⟨e, he⟩)
            (pv := pv.get1 ⟨e, he⟩) (x' := acc'.get1 ⟨e, he⟩) (al := al.item) (pp := pp) (vv := vv)
            (μ' := mn.item.toReal) hK1 hK2 hlanes (fun k hk => hVx e k he (lt_of_lt_of_le hk hNle)) ?_
            (hblock ⟨e, he⟩) (hupd (⟨e, he⟩, ())) ?_ ?_
          · intro j hj
            have hjP : j < d.PAGE_SIZE := lt_of_lt_of_le hj hcP
            have := ((hvst ⟨j, hjP⟩ ⟨e, he⟩).2 (hcv ⟨j, hjP⟩ hj)).1
            simp only [Num.mr, dite_eq_left hjP, dite_eq_left he]
            exact this
          · intro h0
            obtain ⟨hmf, -, hacf, hK1o, hK2o, hs1, -, -, hA⟩ := hstpos h0
            rw [hminN] at hs1 hA
            obtain ⟨hle, haf, hae, ha2⟩ := hal_pos h0
            exact ⟨m_i.toReal, hacf e he, hA e he, hle, haf, hae, ha2, hs1, hK1o, hK2o⟩
          · intro h0
            have := congrArg (fun p => p.1) (hst0 h0)
            simp only [st0] at this
            exact ⟨by rw [show acc.get1 ⟨e, he⟩ = (tl.zeros [d.HEAD_DIM] .f32).get1 ⟨e, he⟩ by rw [this]]; rfl,
              hal0 h0⟩
        unfold StPos
        dsimp only
        rw [hmin]
        exact ⟨hmnf, hlcol.1, fun e he => (hacol e he).1, hK1, hK2, hall, hatt, hlcol.2,
          fun e he => (hacol e he).2⟩
end Claims.PagedAttention
