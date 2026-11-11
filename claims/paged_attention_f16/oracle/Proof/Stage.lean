import Proof.Setup

/-!
# Facts about one page: addresses, the scores, the weights

Pure lemmas about the values one trip of the loop computes, used by the walk.
-/
namespace Claims.PagedAttention

open TVA

/-- A finite `float16` has magnitude at most `65504`. -/
theorem abs_f16_le {x : F16} (hx : x.isFinite = true) : |x.toReal| ≤ 65504 := by
  have h := Format.abs_toRat_le_maxVal Format.binary16_regular hx
  have h2 : ((|x.toRat| : ℚ) : ℝ) ≤ ((Format.binary16.maxVal : ℚ) : ℝ) := by exact_mod_cast h
  push_cast at h2
  rw [Format.binary16_maxVal_real] at h2
  unfold FloatV.toReal
  exact h2

/-- `val16` of an element that reads as `x` is `x.toRat`. -/
theorem val16_eq {m : Mem} {b : Buf .f16} {i : Nat} {x : F16}
    (h : m.read .f16 (b.ptr (i : Int)).addr = some x) : val16 m b i = x.toRat := by
  unfold val16; rw [h]


/-- `FloatV.hostF32` of the (finite) scale. -/
theorem scale_f32 {d : Inst} (hd : d.ok) :
    (FloatV.hostF32 d.scale).isFinite = true ∧
      |(FloatV.hostF32 d.scale).toReal - (sigma d : ℝ)| ≤ (sigma d : ℝ) * Num.u + 1 / 2 ^ 150 := by
  have hfin := hd.scale.1
  have hnan : d.scale.isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  have hs0 : 0 < (sigma d) := lt_of_lt_of_le (by norm_num) hd.scale.2.1
  have hmax : |d.scale.toRat| ≤ Format.binary32.maxVal := by
    have h1 : d.scale.toRat ≤ 2 ^ 64 := hd.scale.2.2
    have h2 : Format.binary32.maxVal = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
    have h3 : 0 ≤ d.scale.toRat := hs0.le
    rw [abs_of_nonneg h3, h2]; refine h1.trans ?_; norm_num
  unfold FloatV.hostF32
  rw [ite_eq_right (by simp [hnan])]
  obtain ⟨hf, he⟩ := FloatV.cvt_err (g := Format.binary32) Format.binary32_regular hfin hmax
  refine ⟨hf, ?_⟩
  have : ((|(FloatV.cvt d.scale : F32).toRat - d.scale.toRat| : ℚ) : ℝ) ≤
      ((|d.scale.toRat| * Format.binary32.u + Format.binary32.eta : ℚ) : ℝ) := by exact_mod_cast he
  push_cast at this
  rw [Num.u32_real, Num.eta32_real, abs_of_nonneg (by exact_mod_cast hs0.le : (0 : ℝ) ≤ (d.scale.toRat : ℝ))] at this
  unfold FloatV.toReal sigma
  linarith

/-- A covered query element is a finite `float16`; its converted value is `val16`. -/
theorem q_facts {d : Inst} {m : Mem} {s h : Nat} (hSV : SeqVals d m s h)
    {q : Tensor [d.HEAD_DIM] F32} 
    (hq : ∀ (e : Nat) (he : e < d.HEAD_DIM), (q.get1 ⟨e, he⟩).isFinite = true ∧
      (q.get1 ⟨e, he⟩).toReal = (val16 m d.q (qIdx d s h e) : ℝ)) (e : Nat) (he : e < d.HEAD_DIM) :
    (q.get1 ⟨e, he⟩).isFinite = true ∧ (q.get1 ⟨e, he⟩).toReal = (val16 m d.q (qIdx d s h e) : ℝ) ∧
      |(val16 m d.q (qIdx d s h e) : ℝ)| ≤ 65504 := by
  obtain ⟨hf, hv⟩ := hq e he
  refine ⟨hf, hv, ?_⟩
  obtain ⟨x, hx, hxf⟩ := hSV.2.1 e he
  have : val16 m d.q (qIdx d s h e) = x.toRat := val16_eq hx
  rw [this]
  exact abs_f16_le hxf

/-- **The scores of a page**: every valid lane's computed scaled score is finite and within `dS X` of the exact one `Sx`. -/
theorem scores_stage {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {m : Mem}
    {s h t pgN : Nat} (hSV : SeqVals d m s h)
    {q : Tensor [d.HEAD_DIM] F32}
    (hq : ∀ (e : Nat) (he : e < d.HEAD_DIM), (q.get1 ⟨e, he⟩).isFinite = true ∧
      (q.get1 ⟨e, he⟩).toReal = (val16 m d.q (qIdx d s h e) : ℝ))
    {kraw : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F16}
    {kk prod : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32} {sums sc : Tensor [d.PAGE_SIZE] F32}
    (hkraw : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
      decide (t * d.PAGE_SIZE + i.1.val < seqLen d m s) = true →
        m.read .f16 (d.k_cache.ptr (kvIdx d pgN i.1.val (h / qpk d) i.2.1.val : Int)).addr =
          some (kraw.get i))
    (hTok : ∀ r e, r < d.PAGE_SIZE → tokElem d m s h (t * d.PAGE_SIZE + r) e =
      kvIdx d pgN r (h / qpk d) e)
    (hkk : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
      (FloatV.cvtWith RoundMode.rn (kraw.get i)).NaNEq (kk.get i))
    (hprod : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
      ((kk.get i).mul ((q.row d.PAGE_SIZE).get i)).NaNEq (prod.get i))
    (hsums : ∀ i : Fin d.PAGE_SIZE, ∃ tr : Tree, tr.Covers d.HEAD_DIM ∧
      tr.depth ≤ tl.sumBound E.P d.HEAD_DIM ∧
      tr.Eval (AddRel E.fusion) (fun j => if hj : j < d.HEAD_DIM then prod.get2 i ⟨j, hj⟩ else default)
        (sums.get1 i))
    (hsc : ∀ i : Idx [d.PAGE_SIZE],
      ((sums.get i).mul ((bc (FloatV.hostF32 d.scale)).get i)).NaNEq (sc.get i))
    {X : ℝ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024)
    (hXk : ∀ k, k < seqLen d m s → (sigma d : ℝ) * (scoreAbs d m s h k : ℝ) ≤ X) :
    ∀ r : Fin d.PAGE_SIZE, t * d.PAGE_SIZE + r.val < seqLen d m s →
      (sc.get1 r).isFinite = true ∧
        |(sc.get1 r).toReal - Sx d m s h (t * d.PAGE_SIZE + r.val)| ≤ Num.dS X := by
  intro r hr
  have hD : d.HEAD_DIM ≤ 256 := by
    obtain ⟨a, ha, he⟩ := hd.head_dim
    rw [he]; calc 2 ^ a ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) ha
      _ = 256 := by norm_num
  have hDpos : 0 < d.HEAD_DIM := by obtain ⟨a, ha, he⟩ := hd.head_dim; rw [he]; positivity
  obtain ⟨hσf, hσe⟩ := scale_f32 hd
  have hk' : ∀ (e : Nat) (he : e < d.HEAD_DIM), (kk.get2 r ⟨e, he⟩).isFinite = true ∧
      (kk.get2 r ⟨e, he⟩).toReal = (val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ) ∧
      |(val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ)| ≤ 65504 := by
    intro e he
    obtain ⟨⟨x, hx, hxf⟩, -, -⟩ := hSV.2.2.1 (t * d.PAGE_SIZE + r.val) e hr he
    have hrd : decide (t * d.PAGE_SIZE + (r : Fin d.PAGE_SIZE).val < seqLen d m s) = true :=
      decide_eq_true hr
    have h1 := hkraw (r, ⟨e, he⟩, ()) hrd
    have h2 : kvIdx d pgN r.val (h / qpk d) e = tokElem d m s h (t * d.PAGE_SIZE + r.val) e :=
      (hTok r.val e r.isLt).symm
    simp only [] at h1
    rw [← h2] at hx
    have hxe : kraw.get (r, ⟨e, he⟩, ()) = x := by
      rw [hx] at h1; exact (Option.some.inj h1).symm
    obtain ⟨hkf, hkv⟩ := Num.cvt16_exact (x := kraw.get (r, ⟨e, he⟩, ())) (by rw [hxe]; exact hxf) RoundMode.rn (hkk (r, ⟨e, he⟩, ()))
    refine ⟨hkf, ?_, ?_⟩
    · show FloatV.toReal (kk.get (r, ⟨e, he⟩, ())) = _
      rw [hkv, hxe]
      have : val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) = x.toRat := by
        rw [← h2]; exact val16_eq hx
      rw [this]; rfl
    · have : val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) = x.toRat := by
        rw [← h2]; exact val16_eq hx
      rw [this]; exact abs_f16_le hxf
  have hσr : (1 : ℝ) / 2 ^ 64 ≤ (sigma d : ℝ) ∧ (sigma d : ℝ) ≤ 2 ^ 64 := by
    have h1 := hd.scale.2.1
    have h2 := hd.scale.2.2
    unfold sigma
    constructor
    · have : (((1 / 2 ^ 64 : ℚ)) : ℝ) ≤ ((d.scale.toRat : ℚ) : ℝ) := by exact_mod_cast h1
      push_cast at this; exact this
    · have : ((d.scale.toRat : ℚ) : ℝ) ≤ (((2 ^ 64 : ℚ)) : ℝ) := by exact_mod_cast h2
      push_cast at this; exact this
  have hXr := hXk (t * d.PAGE_SIZE + r.val) hr
  have hXs : (sigma d : ℝ) * ∑ e ∈ Finset.range d.HEAD_DIM,
      |(val16 m d.q (qIdx d s h e) : ℝ) * (val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ)| ≤ X := by
    have : (scoreAbs d m s h (t * d.PAGE_SIZE + r.val) : ℝ) = ∑ e ∈ Finset.range d.HEAD_DIM,
        |(val16 m d.q (qIdx d s h e) : ℝ) * (val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ)| := by
      unfold scoreAbs; push_cast; rfl
    rw [← this]; exact hXr
  obtain ⟨hfin, herr⟩ := Num.scores_row hP hD hDpos q kk prod sums r (FloatV.hostF32 d.scale) (sc.get1 r)
    (qr := fun e => (val16 m d.q (qIdx d s h e) : ℝ))
    (kr := fun e => (val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ))
    (σr := (sigma d : ℝ)) (X := X)
    (fun e he => q_facts hSV hq e he) hk'
    (fun e => hprod (r, e, ()))
    (hsums r) (hsc (r, ())) ⟨hσf, hσe⟩ hσr hXs hX1
  refine ⟨hfin, ?_⟩
  have hSx : Sx d m s h (t * d.PAGE_SIZE + r.val) = (sigma d : ℝ) * ∑ e ∈ Finset.range d.HEAD_DIM,
      (val16 m d.q (qIdx d s h e) : ℝ) * (val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ) := by
    unfold Sx score; push_cast; rfl
  rw [hSx]
  refine herr.trans ?_
  have hu := Num.u_pos
  have : 28 * Num.u * ((sigma d : ℝ) * ∑ e ∈ Finset.range d.HEAD_DIM,
      |(val16 m d.q (qIdx d s h e) : ℝ) * (val16 m d.k_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e) : ℝ)|) ≤ 28 * Num.u * X :=
    mul_le_mul_of_nonneg_left hXs (by positivity)
  unfold Num.dS
  have h60 : (1 : ℝ) / 2 ^ 60 = (2 : ℝ) ^ (-60 : ℤ) := by norm_num
  rw [← h60]
  linarith

/-- The offsets of a page's keys and values: lane `(r, e)` of `page·stride_kp + r·stride_kt + g·stride_kh + e`. -/
theorem kv_offs_lane {d : Inst} (hd : d.ok) {page A B : I64}
    {c1 c3 : Tensor [d.PAGE_SIZE, d.HEAD_DIM] I64} {b g : Nat} (hb31 : b < 2 ^ 31)
    (hg : g < d.num_kv_heads) (hpage : page.toInt = b) (hA : A.toInt = d.stride_kp)
    (hB : B.toInt = ((g * d.stride_kh : Nat) : Int))
    (hc1 : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], (c1.get i).toInt = ((i.1.val * d.stride_kt : Nat) : Int))
    (hc3 : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], (c3.get i).toInt = ((i.2.1.val : Nat) : Int))
    (i : Idx [d.PAGE_SIZE, d.HEAD_DIM]) :
    ((page * A + c1 + B + c3).get i).toInt =
      ((b * d.stride_kp + i.1.val * d.stride_kt + g * d.stride_kh + i.2.1.val : Nat) : Int) := by
  have h2 := hd.strides_lt
  have hkp := h2.2.2.1
  have hkt : (d.PAGE_SIZE - 1) * d.stride_kt < 2 ^ 31 := hd.prods_lt.2.2.2.2.2.1
  have hkh : (d.num_kv_heads - 1) * d.stride_kh < 2 ^ 31 := hd.prods_lt.2.2.2.2.2.2
  have hri : i.1.val ≤ d.PAGE_SIZE - 1 := by have := i.1.isLt; omega
  have hrt : i.1.val * d.stride_kt < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ hri) hkt
  have hgh : g * d.stride_kh < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (by omega)) hkh
  have hei : i.2.1.val < 2 ^ 8 := by
    obtain ⟨a, ha, he⟩ := hd.head_dim
    have h1 := i.2.1.isLt
    have h2 : d.HEAD_DIM ≤ 2 ^ 8 := by rw [he]; exact Nat.pow_le_pow_right (by norm_num) ha
    omega
  have hbk : b * d.stride_kp < 2 ^ 62 := by
    have : b * d.stride_kp < 2 ^ 31 * 2 ^ 31 := Nat.mul_lt_mul'' hb31 hkp
    simpa using this
  have e : (page * A + c1 + B + c3).get i = ((page * A + c1.get i) + B) + c3.get i := rfl
  have t1 : (page * A).toInt = ((b * d.stride_kp : Nat) : Int) :=
    IntV.I64.mul_nat hpage hA (by omega)
  have t2 : (page * A + c1.get i).toInt = ((b * d.stride_kp + i.1.val * d.stride_kt : Nat) : Int) :=
    IntV.I64.add_nat t1 (hc1 i) (by omega)
  have t3 : (page * A + c1.get i + B).toInt =
      ((b * d.stride_kp + i.1.val * d.stride_kt + g * d.stride_kh : Nat) : Int) :=
    IntV.I64.add_nat t2 hB (by omega)
  rw [e]
  exact IntV.I64.add_nat t3 (hc3 i) (by omega)

/-- The `other = 0` of the masked loads has a `float16` conversion. -/
theorem conv_zero_exists (P : Platform) : ∃ r : F16, ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0)) r := by
  have hne : Format.binary32 ≠ Format.binary16 := by decide
  refine ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0)), ?_⟩
  simp only [ConvRel, ofFloatRel, FloatRel, hne, ite_false]
  exact Or.inl ⟨Format.binary32_std, Format.binary16_std, FloatV.NaNEq.refl _⟩

/-- Token `t·PAGE_SIZE + r` of the sequence lives at the cache element `kvIdx` of the page loaded at trip `t`. -/
theorem tokElem_eq {d : Inst} {m : Mem} {s h t r e pgv : Nat} (hr : r < d.PAGE_SIZE)
    (hpg : (val32 m d.block_tables (btIdx d s t)).toNat = pgv) :
    tokElem d m s h (t * d.PAGE_SIZE + r) e = kvIdx d pgv r (h / qpk d) e := by
  have hP0 : 0 < d.PAGE_SIZE := by omega
  unfold tokElem pageOf
  have h1 : (t * d.PAGE_SIZE + r) / d.PAGE_SIZE = t := by
    rw [Nat.mul_comm, Nat.mul_add_div hP0, Nat.div_eq_of_lt hr]; simp
  have h2 : (t * d.PAGE_SIZE + r) % d.PAGE_SIZE = r := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hr]
  rw [h1, h2, hpg]


end Claims.PagedAttention
