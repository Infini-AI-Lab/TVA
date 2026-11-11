import Proof.Trip

/-!
# The end of a program: the division and the rounding

`out_lane`: from the invariant after the last page, the stored `float16` is within the claim's
tolerance of the exact attention.
-/
namespace Claims.PagedAttention

open TVA

/-- The exact attention is the ratio of the invariant's sums. -/
theorem attn_eq {d : Inst} {m : Mem} {s h e : Nat} (μ : ℝ) :
    attnRef d m s h e = Num.aS (Sx d m s h) (Vx d m s h e) μ (seqLen d m s) /
      Num.lS (Sx d m s h) μ (seqLen d m s) ∧
    attnAbs d m s h e = Num.mS (Sx d m s h) (Vx d m s h e) μ (seqLen d m s) /
      Num.lS (Sx d m s h) μ (seqLen d m s) := by
  constructor
  · rw [Num.aS_div_lS]
    unfold attnRef weight Sx Vx
    rfl
  · rw [Num.mS_div_lS]
    unfold attnAbs weight Sx Vx
    rfl

/-- Bounds on the number of pages: at most `max_pages ≤ 2^16`, and they hold every token. -/
theorem nblocks_le {d : Inst} (hd : d.ok) {m : Mem} {s : Nat} (hsok : SeqOk d m s) :
    nblocks d m s ≤ 2 ^ 16 ∧ seqLen d m s ≤ nblocks d m s * d.PAGE_SIZE ∧ seqLen d m s ≤ 2 ^ 22 := by
  obtain ⟨n, hn_read, hn0, hn_le, hn22, hbt⟩ := hsok
  have hnl : seqLen d m s = n.toInt.toNat := by unfold seqLen val32; rw [hn_read]
  have hP0 : 0 < d.PAGE_SIZE := by obtain ⟨b, hb, he⟩ := hd.page_size; rw [he]; positivity
  have hlt := Nat.lt_div_mul_add (a := seqLen d m s + d.PAGE_SIZE - 1) hP0
  have hnm : seqLen d m s ≤ d.max_pages * d.PAGE_SIZE := by rw [hnl]; omega
  unfold nblocks
  refine ⟨?_, ?_, ?_⟩
  · have h1 : (seqLen d m s + d.PAGE_SIZE - 1) / d.PAGE_SIZE ≤ d.max_pages := by
      refine Nat.lt_succ_iff.mp ((Nat.div_lt_iff_lt_mul hP0).mpr ?_)
      rw [Nat.succ_mul]; omega
    exact h1.trans hd.max_pages_le
  · omega
  · rw [hnl]; omega

/-- With at least one token there is at least one page. -/
theorem nblocks_pos {d : Inst} (hd : d.ok) {m : Mem} {s : Nat} (hn1 : 1 ≤ seqLen d m s) :
    1 ≤ nblocks d m s := by
  have hP0 : 0 < d.PAGE_SIZE := by obtain ⟨b, hb, he⟩ := hd.page_size; rw [he]; positivity
  unfold nblocks
  rw [Nat.le_div_iff_mul_le hP0]; omega

/-- **One output element**: from the invariant after all pages, the division and the rounding to
`float16` give a finite value within the claim's tolerance of the exact attention. -/
theorem out_lane {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {m : Mem} {s h : Nat}
    (hsok : SeqOk d m s) (hSV : SeqVals d m s h) {X : ℚ} (hXle : X ≤ 2 ^ 10)
    (hXk : ∀ k, k < seqLen d m s → sigma d * scoreAbs d m s h k ≤ X)
    {acc : Tensor [d.HEAD_DIM] F32} {l_i m_i : F32}
    (hnum : ∀ Y : ℝ, 0 ≤ Y → Y ≤ 1024 →
      (∀ k, k < seqLen d m s → (sigma d : ℝ) * (scoreAbs d m s h k : ℝ) ≤ Y) →
        StPos d m s h Y (nblocks d m s) (acc, l_i, m_i))
    {e : Nat} (he : e < d.HEAD_DIM) {qd : F32} {c : F16}
    (hdiv : E.P.div .binary32 (acc.get1 ⟨e, he⟩) l_i qd)
    (hc : (FloatV.cvtWith RoundMode.rn qd).NaNEq c) :
    c.isFinite = true ∧ |c.toReal - attnRef d m s h e| ≤
      (tolA (nblocks d m s) X : ℝ) * (attnAbs d m s h e + |attnRef d m s h e|) +
        1 / 2 ^ 11 * |attnRef d m s h e| + 1 / 2 ^ 24 := by
  have hn1 : 1 ≤ seqLen d m s := hSV.1
  obtain ⟨hT2, hnT, hn22⟩ := nblocks_le hd hsok
  have hT1 := nblocks_pos hd (m := m) (s := s) hn1
  have hσ0 : 0 ≤ sigma d := by
    have := hd.scale.2.1
    unfold sigma; exact le_trans (by norm_num) this
  have hX0q : 0 ≤ X := le_trans (mul_nonneg hσ0 (Finset.sum_nonneg fun _ _ => abs_nonneg _)) (hXk 0 hn1)
  have hX0 : (0 : ℝ) ≤ X := by exact_mod_cast hX0q
  have hX1 : (X : ℝ) ≤ 1024 := by exact_mod_cast hXle
  have hXk' : ∀ k, k < seqLen d m s → (sigma d : ℝ) * (scoreAbs d m s h k : ℝ) ≤ X := by
    intro k hk
    have := hXk k hk
    exact_mod_cast this
  have hSb : ∀ k, k < seqLen d m s → |Sx d m s h k| ≤ X := by
    intro k hk
    have h1 : |(score d m s h k : ℚ)| ≤ scoreAbs d m s h k := by
      unfold score scoreAbs; exact Finset.abs_sum_le_sum_abs _ _
    have h2 : |(score d m s h k : ℝ)| ≤ (scoreAbs d m s h k : ℝ) := by exact_mod_cast h1
    have hσ0' : 0 ≤ (sigma d : ℝ) := by exact_mod_cast hσ0
    unfold Sx
    rw [abs_mul, abs_of_nonneg hσ0']
    exact (mul_le_mul_of_nonneg_left h2 hσ0').trans (hXk' k hk)
  obtain ⟨hmf, hlf, hacf, hK1, hK2, hs1, hs2, hL, hA⟩ := hnum (X : ℝ) hX0 hX1 hXk'
  have hminN : min (nblocks d m s * d.PAGE_SIZE) (seqLen d m s) = seqLen d m s := min_eq_right hnT
  rw [hminN] at hs1 hs2 hL hA
  simp only [] at hmf hlf hK1 hK2 hs1 hs2 hL
  have hAe := hA e he
  simp only [] at hAe
  obtain ⟨hlpos, hbound, h40⟩ := Num.final_core hX0 hX1 (Sx d m s h) (Vx d m s h e) (n := seqLen d m s)
    (T := nblocks d m s) hT1 hT2 hn1 hn22 (fun k hk => by
      obtain ⟨-, -, hvb⟩ := hSV.2.2.1 k e hk he
      have h3 : ((|val16 m d.v_cache (tokElem d m s h k e)| : ℚ) : ℝ) ≤ ((2 ^ 15 : ℚ) : ℝ) := by
        exact_mod_cast hvb
      push_cast at h3; unfold Vx; exact h3) hs1 hs2 hK2 hK1 hL hAe
  obtain ⟨hqf, hqb, hqe⟩ := Num.div_ok hP (a := acc.get1 ⟨e, he⟩) (l := l_i) (q := qd)
    (hacf e he) hlf hlpos h40 hdiv
  obtain ⟨hcf, hce⟩ := Num.cvt16_ok hqf hqb hc
  refine ⟨hcf, ?_⟩
  obtain ⟨hR, hΛ⟩ := attn_eq (d := d) (m := m) (s := s) (h := h) (e := e) m_i.toReal
  have hL0 : 0 < Num.lS (Sx d m s h) m_i.toReal (seqLen d m s) := by
    obtain ⟨k, hk, hkle⟩ := hs2
    exact lt_of_lt_of_le (Real.exp_pos _) (Num.lS_ge (Sx d m s h) hk hkle)
  have hAbs : |Num.aS (Sx d m s h) (Vx d m s h e) m_i.toReal (seqLen d m s)| ≤
      Num.mS (Sx d m s h) (Vx d m s h e) m_i.toReal (seqLen d m s) := Num.abs_aS_le _ _ _ _
  have hMV : Num.mS (Sx d m s h) (Vx d m s h e) m_i.toReal (seqLen d m s) ≤
      2 ^ 15 * Num.lS (Sx d m s h) m_i.toReal (seqLen d m s) := by
    unfold Num.mS Num.lS
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun k hk => ?_
    have : |Vx d m s h e k| ≤ 2 ^ 15 := by
      obtain ⟨-, -, hvb⟩ := hSV.2.2.1 k e (Finset.mem_range.mp hk) he
      have h3 : ((|val16 m d.v_cache (tokElem d m s h k e)| : ℚ) : ℝ) ≤ ((2 ^ 15 : ℚ) : ℝ) := by
        exact_mod_cast hvb
      push_cast at h3; unfold Vx; exact h3
    nlinarith [Real.exp_pos (Sx d m s h k - m_i.toReal)]
  have hRΛ : |attnRef d m s h e| ≤ attnAbs d m s h e := by
    rw [hR, hΛ, abs_div, abs_of_pos hL0]
    exact div_le_div_of_nonneg_right hAbs hL0.le
  have hΛ2 : attnAbs d m s h e ≤ 2 ^ 15 := by
    rw [hΛ, div_le_iff₀ hL0]; exact hMV
  have hτ := Num.tauMax_range hX0 hX1 hT2
  have hw : |(acc.get1 ⟨e, he⟩).toReal / l_i.toReal - attnRef d m s h e| ≤
      Num.tauMax (nblocks d m s) X * (attnAbs d m s h e + |attnRef d m s h e|) + 1 / 2 ^ 30 := by
    rw [← hR, ← hΛ] at hbound
    exact hbound
  have := Num.out_tol hRΛ hΛ2 hτ.1 hτ.2 hw hqe hce
  have hτe : Num.tauMax (nblocks d m s) X * (1 + 1 / 2 ^ 10) + 1 / 2 ^ 21 =
      (tolA (nblocks d m s) X : ℝ) := by
    unfold Num.tauMax Num.tolR tolA
    push_cast
    field_simp
    ring
  rw [hτe] at this
  exact this

end Claims.PagedAttention
