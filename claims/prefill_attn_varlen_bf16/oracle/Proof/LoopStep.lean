import Proof.Setup
import Proof.Access
import Proof.Floats
import Proof.NumStep

section
/-!
# One trip of the loop

`trip_safe_proof` walks `loopBody` for one trip: the loads of `k` and `v`, the score dot product
and its mask, the online-softmax update and the second dot product. The walk collects the
relation each operation gives (`ScoreOps`, `SoftOps`) lane by lane; the float lemma of the trip
turns them into the row's state.
-/

namespace Claims.PrefillAttnVarlen

open TVA TVA.Tile2

/-! ## The program's numbers -/

/-- The block of a program, its head and sequence are in range. -/
theorem prog_range {d : Inst} (i : Fin (launch d).size) :
    bOf d i < d.batch ∧ hOf d i < d.H ∧ mOf d i < nBlocks d := by
  have hsize : (i : Nat) < d.batch * d.H * nBlocks d := i.isLt
  have hbatch : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  refine ⟨Nat.mod_lt _ hbatch, Nat.mod_lt _ hH, ?_⟩
  unfold mOf
  rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos hbatch hH)]
  rw [Nat.mul_comm (nBlocks d)]; exact hsize


/-- The sequence of a program has `start + len ≤ total`, `len ≤ max_input_len` and both are
`int32`-sized. -/
theorem seq_facts {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.batch) :
    startOf d init b + lenOf d init b ≤ d.total ∧ lenOf d init b ≤ d.maxLen := by
  refine ⟨seq_le hpre hb, ?_⟩
  obtain ⟨-, -, -, -, -, -, ⟨hseq, -⟩, -⟩ := id hpre
  obtain ⟨s, n, hs, hn, hs0, hn0, hnm, -⟩ := hseq b hb
  have : lenOf d init b = n.toInt.toNat := by unfold lenOf val32; rw [hn]
  omega

/-- A loop trip `t` of program `i` has a key block inside the sequence (`128·t < len`), starts a
query block inside the sequence, and (when causal) is at or below the query block. -/
theorem trip_facts {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) {t : Nat} (ht : t < nTrips d init i) :
    mOf d i * BLOCK < lenOf d init (bOf d i) ∧ 128 * t < lenOf d init (bOf d i) ∧
      (d.causal = true → t ≤ mOf d i) ∧ t < 1024 := by
  obtain ⟨hb, -, -⟩ := prog_range i
  obtain ⟨-, hlm⟩ := seq_facts hpre hb
  have hmax := hd.maxLen_le
  unfold nTrips at ht
  by_cases h1 : mOf d i * BLOCK < lenOf d init (bOf d i)
  · simp only [h1, ite_true] at ht
    unfold BLOCK at ht h1 ⊢
    cases hc : d.causal
    · simp only [hc, Bool.false_eq_true, ite_false] at ht
      refine ⟨h1, ?_, by simp, ?_⟩ <;> omega
    · simp only [hc, ite_true] at ht
      refine ⟨h1, ?_, fun _ => ?_, ?_⟩ <;> omega
  · simp only [h1, ite_false] at ht
    exact absurd ht (Nat.not_lt_zero _)


/-! ## What the walk collects -/

/-- The relations the operations between the scores `qk` and the scaled accumulator `acc₁` give,
lane by lane (the rules' continuations): the row maximum `m_ij`, `p = exp (qk - m_ij)`, the row
sum `l_ij`, the new maximum, `alpha`, `beta`, the new sum, the weights `p₁`, the accumulator's
scale and the scaled accumulator. -/
def SoftOps (P : Platform) (fusion : Bool) {BD : Nat} (m_i l_i : Tensor [BLOCK] F32)
    (acc : Tensor [BLOCK, BD] F32) (qk : Tensor [BLOCK, BLOCK] F32) (m_new lnew : Tensor [BLOCK] F32)
    (p₁ : Tensor [BLOCK, BLOCK] F32) (acc₁ : Tensor [BLOCK, BD] F32) : Prop :=
  ∃ (m_ij : Tensor [BLOCK] F32) (e1 p : Tensor [BLOCK, BLOCK] F32)
    (l_ij e2 alpha e3 beta a1 a2 pscale d1 ascale : Tensor [BLOCK] F32),
    (∀ i : Fin BLOCK, ∃ t : Tree, t.Covers BLOCK ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < BLOCK then qk.get2 i ⟨j, h⟩ else default)
        (m_ij.get1 i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.sub (qk.get i) ((m_ij.col BLOCK).get i)) (e1.get i)) ∧
    (∀ i, P.math .exp .binary32 (e1.get i) (p.get i)) ∧
    (∀ i : Fin BLOCK, ∃ t : Tree, t.Covers BLOCK ∧ t.depth ≤ tl.sumBound P BLOCK ∧
      t.Eval (AddRel false) (fun j => if h : j < BLOCK then p.get2 i ⟨j, h⟩ else default)
        (l_ij.get1 i)) ∧
    (∀ i, FloatV.MaxRel true (m_i.get i) (m_ij.get i) (m_new.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.sub (m_i.get i) (m_new.get i)) (e2.get i)) ∧
    (∀ i, P.math .exp .binary32 (e2.get i) (alpha.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.sub (m_ij.get i) (m_new.get i)) (e3.get i)) ∧
    (∀ i, P.math .exp .binary32 (e3.get i) (beta.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.mul (alpha.get i) (l_i.get i)) (a1.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.mul (beta.get i) (l_ij.get i)) (a2.get i)) ∧
    (∀ i, AddRel fusion (a1.get i) (a2.get i) (lnew.get i)) ∧
    (∀ i, P.div .binary32 (beta.get i) (lnew.get i) (pscale.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.mul (p.get i) ((pscale.col BLOCK).get i)) (p₁.get i)) ∧
    (∀ i, P.div .binary32 (l_i.get i) (lnew.get i) (d1.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.mul (d1.get i) (alpha.get i)) (ascale.get i)) ∧
    (∀ i, FloatV.NaNEq (FloatV.mul (acc.get i) ((ascale.col BD).get i)) (acc₁.get i))


/-- The scores the walk meets, for the row `r` of the block: the attended lanes are finite and within
`δ` of the exact scores, the others are `-∞`. -/
def ScoreRow (d : Inst) (init : Mem) (i : Nat) (t : Nat) (r : Fin BLOCK)
    (qk : Tensor [BLOCK, BLOCK] F32) : Prop :=
  (∀ c : Fin BLOCK, c.val ∈ Num.blk (lastKey d init (bOf d i) (mOf d i * BLOCK + r)) t →
    (qk.get2 r c).isFinite = true ∧
      |(qk.get2 r c).toReal - score d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) (128 * t + c)| ≤
        scoreErr d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r))) ∧
  ∀ c : Fin BLOCK, c.val ∉ Num.blk (lastKey d init (bOf d i) (mOf d i * BLOCK + r)) t →
    qk.get2 r c = FloatV.negInf

/-- The loaded `k` tile: lane `(e, c)` holds `k[start + 128·t + c, g, e]` inside the sequence and
the head dimension, and `+0` outside. -/
def KFacts (d : Inst) (init : Mem) (i : Nat) (t : Nat) (k : Tensor [d.BD, BLOCK] BF16) : Prop :=
  ∀ (e : Fin d.BD) (c : Fin BLOCK), DType.val? .bf16 (k.get2 e c) =
    some (if 128 * t + c < lenOf d init (bOf d i) ∧ e.val < d.Lk then
      kv d init (bOf d i) (hOf d i) (128 * t + c) e else 0)

/-- The loaded `v` tile, as `KFacts`. -/
def VFacts (d : Inst) (init : Mem) (i : Nat) (t : Nat) (v : Tensor [BLOCK, d.BD] BF16) : Prop :=
  ∀ (c : Fin BLOCK) (e : Fin d.BD), DType.val? .bf16 (v.get2 c e) =
    some (if 128 * t + c < lenOf d init (bOf d i) ∧ e.val < d.Lk then
      vv d init (bOf d i) (hOf d i) (128 * t + c) e else 0)

/-- The scores of every valid row of the tensor `qk`. -/
def ScoreFacts (d : Inst) (init : Mem) (i : Nat) (t : Nat) (qk : Tensor [BLOCK, BLOCK] F32) : Prop :=
  ∀ r : Fin BLOCK, validRow d init i r → ScoreRow d init i t r qk

/-- The last key a row attends is inside the sequence. -/
theorem lastKey_lt {d : Inst} {init : Mem} {b p : Nat} (hp : p < lenOf d init b) :
    lastKey d init b p < lenOf d init b := by
  unfold lastKey
  cases d.causal <;> simp <;> omega

/-- The exact score of a pair is at most its magnitude `scoreMag`. -/
theorem abs_score_le_mag {d : Inst} (hd : d.ok) (m : Mem) (b h p j : Nat) :
    |score d m b h p j| ≤ scoreMag d m b h p j := by
  have hsig : 0 ≤ sig d := by
    have := hd.scale.2.1
    have h0 : (0 : ℝ) ≤ (d.sm_scale.toRat : ℝ) := by
      exact_mod_cast (le_trans (by positivity) this : (0 : ℚ) ≤ d.sm_scale.toRat)
    exact h0
  unfold score scoreMag
  rw [abs_mul, abs_of_nonneg hsig]
  refine mul_le_mul_of_nonneg_left ?_ hsig
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e _ => ?_)
  rw [← Rat.cast_mul, ← Rat.cast_abs]
  exact_mod_cast Device.abs_mul_le_liftMag .bf16 .bf16 (qv d m b h p e) (kv d m b h j e)

/-- The bounds of a valid row that a trip needs (`Trip.trip`): the score error is nonnegative, the
computed scores are at most `Y ≤ 5000`, and the value sums are nonnegative, at most `2^138`, with
entries at most `2^120`. -/
theorem row_bounds {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b h p : Nat}
    (hb : b < d.batch) (hh : h < d.H) (hp : p < lenOf d init b) :
    0 ≤ scoreErr d.BD (Xrow d init b h p) ∧
    (∀ j, j ≤ (rowSpec d init b h p).L → |(rowSpec d init b h p).sc j| +
      scoreErr d.BD (Xrow d init b h p) ≤ rowY d.BD (Xrow d init b h p)) ∧
    rowY d.BD (Xrow d init b h p) ≤ 5000 ∧ (∀ e, 0 ≤ rowV d init b h p e) ∧
    (∀ e, ∑ j ∈ Finset.range ((rowSpec d init b h p).L + 1), |(rowSpec d init b h p).vv e j| ≤
      rowV d init b h p e) ∧ (∀ e, rowV d init b h p e ≤ 2 ^ 138) ∧
    (∀ e j, j ≤ (rowSpec d init b h p).L → |(rowSpec d init b h p).vv e j| ≤ 2 ^ 120) := by
  have hvals := (hpre.2.2.2.2.2.2.2 b hb).2.2
  have hLk := lastKey_lt (d := d) (init := init) hp
  obtain ⟨hSL, hlm⟩ := seq_facts hpre hb
  have hmax := hd.maxLen_le
  have hsc : ∀ j, j ≤ lastKey d init b p → |score d init b h p j| ≤ Xrow d init b h p :=
    fun j hj => (abs_score_le_mag hd init b h p j).trans
      (Finset.le_sup' (fun j => scoreMag d init b h p j) (Finset.mem_range.2 (by omega)))
  have hX0 : 0 ≤ Xrow d init b h p := (abs_nonneg _).trans (hsc 0 (Nat.zero_le _))
  have hX : Xrow d init b h p ≤ 2 ^ 12 :=
    Finset.sup'_le _ _ fun j hj => hvals _ _ _ hh hp
      (lt_of_le_of_lt (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)) hLk)
  have hD2 := hd.bd_le
  have hδ0 : 0 ≤ scoreErr d.BD (Xrow d init b h p) := by unfold scoreErr; positivity
  have hδ1 : scoreErr d.BD (Xrow d init b h p) ≤ 1 := by
    have hD : (d.BD : ℝ) ≤ 256 := by exact_mod_cast hD2
    have h1 : ((d.BD : ℝ) + 8) * Xrow d init b h p ≤ 264 * 4096 := by
      nlinarith [mul_nonneg (show (0 : ℝ) ≤ 256 - d.BD by linarith) hX0]
    unfold scoreErr
    have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    have c40 : (2 : ℝ) ^ (-40 : ℤ) ≤ 1e-11 := by norm_num
    rw [mul_assoc, c22]
    nlinarith
  have hvv : ∀ j, j ≤ lastKey d init b p → ∀ e : Nat, e < d.Lk →
      |((vv d init b h j e : ℚ) : ℝ)| ≤ 2 ^ 120 := by
    intro j hj e he
    have := ((hpre.2.2.2.2.2.2.2 b hb).2.1 j (kvh d h) e (by omega) (kvh_lt hd hh) he).2.2
    exact_mod_cast this
  have hVe : ∀ e : Fin d.BD, rowV d init b h p e ≤ 2 ^ 138 := by
    intro e
    unfold rowV
    split_ifs with he
    · unfold vSum
      refine (Finset.sum_le_card_nsmul _ _ ((2 : ℝ) ^ 120) fun j hj => hvv j
        (by have := Finset.mem_range.1 hj; omega) e he).trans ?_
      rw [Finset.card_range, nsmul_eq_mul]
      have : ((lastKey d init b p + 1 : ℕ) : ℝ) ≤ 2 ^ 17 := by exact_mod_cast (by omega)
      calc ((lastKey d init b p + 1 : ℕ) : ℝ) * 2 ^ 120 ≤ 2 ^ 17 * 2 ^ 120 := by gcongr
        _ ≤ 2 ^ 138 := by norm_num
    · norm_num
  refine ⟨hδ0, fun j hj => ?_, ?_, fun e => ?_, fun e => ?_, hVe, fun e j hj => ?_⟩
  · have := hsc j hj
    simp only [rowSpec, rowY]
    linarith
  · unfold rowY; linarith
  · unfold rowV; split_ifs
    · exact Finset.sum_nonneg fun j _ => abs_nonneg _
    · exact le_rfl
  · unfold rowV
    by_cases he : e.val < d.Lk
    · simp [rowSpec, he, vSum]
    · simp [rowSpec, he]
  · by_cases he : e.val < d.Lk
    · simp only [rowSpec, he, ite_true]
      exact hvv j hj e he
    · simp [rowSpec, he]

/-- **The trip's float lemma** for the valid rows, from the operations' relations. -/
theorem trip_conclude {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) {t : Nat} (ht : t < nTrips d init i)
    {fusion : Bool} {m_i l_i : Tensor [BLOCK] F32} {acc : Tensor [BLOCK, d.BD] F32}
    {qk : Tensor [BLOCK, BLOCK] F32} {m_new lnew : Tensor [BLOCK] F32}
    {p₁ : Tensor [BLOCK, BLOCK] F32} {acc₁ acc₂ : Tensor [BLOCK, d.BD] F32}
    {p₂ : Tensor [BLOCK, BLOCK] BF16} {v : Tensor [BLOCK, d.BD] BF16}
    (hI : ∀ r : Fin BLOCK, validRow d init i r →
      RowState (rowSpec d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r))
        (scoreErr d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)))
        (rowY d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)))
        (rowV d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)) t
        (m_i.get1 r) (l_i.get1 r) (fun e => acc.get2 r e))
    (hqk : ScoreFacts d init i t qk) (hv : VFacts d init i t v)
    (hsoft : SoftOps P fusion m_i l_i acc qk m_new lnew p₁ acc₁)
    (hp2 : ∀ x, FloatV.NaNEq (FloatV.cvtWith .rn (p₁.get x)) (p₂.get x))
    (hdot : P.dot cfgBF16 p₂ v acc₁ acc₂ ∨ ∃ d', P.dot cfgBF16 p₂ v
      (Tensor.full (cfgBF16.acc.ofBits 0)) d' ∧
      ∀ x, AccAdd fusion cfgBF16.acc (d'.get x) (acc₁.get x) (acc₂.get x)) :
    ∀ r : Fin BLOCK, validRow d init i r →
      RowState (rowSpec d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r))
        (scoreErr d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)))
        (rowY d.BD (Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)))
        (rowV d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)) (t + 1)
        (m_new.get1 r) (lnew.get1 r) (fun e => acc₂.get2 r e) := by
  obtain ⟨m_ij, e1, p, l_ij, e2, alpha, e3, beta, a1, a2, pscale, d1, ascale, h1, h2, h3, h4,
    h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := hsoft
  let D : Trip.Data d.BD P fusion :=
    { qk := qk, mi := m_i, li := l_i, acc := acc, v := v, mij := m_ij, x1 := e1, p := p,
      lij := l_ij, mnew := m_new, e1 := e2, alpha := alpha, e2 := e3, beta := beta, u1 := a1,
      u2 := a2, lnew := lnew, psc := pscale, p1 := p₁, dq := d1, asc := ascale, acc1 := acc₁,
      p2 := p₂, acc2 := acc₂, hmax := h1, hsub1 := h2, hexp1 := h3, hsum := h4, hmaximum := h5,
      hsub2 := h6, hexp2 := h7, hsub3 := h8, hexp3 := h9, hmul1 := h10, hmul2 := h11,
      haddc := h12, hdiv1 := h13, hmul3 := h14, hdiv2 := h15, hmul4 := h16, hmul5 := h17,
      hcvt := hp2, hdot := hdot }
  intro r hr
  obtain ⟨hb, hh, hmb⟩ := prog_range i
  obtain ⟨hmT, hT, hcaus, ht1024⟩ := trip_facts hd hpre i ht
  obtain ⟨hδ, hY, hY5, hV0, hV, hVu, hvv⟩ := row_bounds hd hpre hb hh hr
  have hL : 128 * t ≤ (rowSpec d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)).L := by
    simp only [rowSpec, lastKey]
    unfold BLOCK at hr ⊢
    split_ifs with hc
    · have := hcaus hc; omega
    · omega
  have hblk : ∀ c : Fin BLOCK, c.val ∈ Num.blk (lastKey d init (bOf d i) (mOf d i * BLOCK + r)) t ↔
      128 * t + c.val ≤ (rowSpec d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)).L := fun c => by
    simp only [Num.blk, Finset.mem_filter, Finset.mem_range, rowSpec]
    exact ⟨fun h => h.2, fun h => ⟨c.isLt, h⟩⟩
  have hLk := lastKey_lt (d := d) (init := init) hr
  exact Trip.trip hP D (rowSpec d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r)) r hδ hY hY5
    hV0 hV hVu hvv hL ht1024 (hI r hr)
    (fun c hc => by
      obtain ⟨hf, he⟩ := (hqk r hr).1 c ((hblk c).mpr hc)
      exact ⟨hf, he⟩)
    (fun c hc => (hqk r hr).2 c (fun h => hc ((hblk c).mp h)))
    (fun c e => (DType.val?_bf16.mp (hv c e)).1)
    (fun c hc e => by
      have hc' : 128 * t + c.val < lenOf d init (bOf d i) := by
        simp only [rowSpec] at hc
        omega
      have hx := (DType.val?_bf16.mp (hv c e)).2
      unfold FloatV.toReal
      show ((FloatV.toRat (v.get2 c e) : ℚ) : ℝ) = _
      by_cases he : e.val < d.Lk
      · simp only [rowSpec, he, hc', true_and, ite_true] at hx ⊢
        rw [hx]
      · simp only [rowSpec, he, hc', ite_false, true_and] at hx ⊢
        rw [hx]; simp)


/-- The `k` tile of the load, from the elements the rule hands over. -/
theorem kFacts_of_load {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size)
    {t : Nat} {k : Tensor [d.BD, BLOCK] BF16}
    (h1 : ∀ (e : Fin d.BD) (c : Fin BLOCK), lane d.BD e < d.Lk →
      keyBlk (128 * t) c < lenOf d init (bOf d i) →
      init.read .bf16 (d.k.ptr ((startOf d init (bOf d i) * d.skt +
        PrefillAttnVarlen.kvh d (hOf d i) * d.skh + lane d.BD e * 1 +
        keyBlk (128 * t) c * d.skt : Nat) : Int)).addr = some (k.get2 e c))
    (h2 : ∀ (e : Fin d.BD) (c : Fin BLOCK),
      ¬ (lane d.BD e < d.Lk ∧ keyBlk (128 * t) c < lenOf d init (bOf d i)) →
      k.get2 e c = FloatV.ofPat 0) :
    KFacts d init i t k := by
  obtain ⟨hb, hh, -⟩ := prog_range i
  have hgN : PrefillAttnVarlen.kvh d (hOf d i) < d.HKV := kvh_lt hd hh
  intro e c
  by_cases hc : 128 * t + c < lenOf d init (bOf d i) ∧ e.val < d.Lk
  · simp only [hc, and_self, ite_true]
    obtain ⟨hfk, -, -⟩ := (hpre.2.2.2.2.2.2.2 (bOf d i) hb).2.1 (128 * t + c) _ e.val hc.1 hgN hc.2
    obtain ⟨x, hx, hxf⟩ := hfk
    have hidx : kIdx d (startOf d init (bOf d i) + (128 * t + c))
        (PrefillAttnVarlen.kvh d (hOf d i)) e.val = startOf d init (bOf d i) * d.skt +
        PrefillAttnVarlen.kvh d (hOf d i) * d.skh + lane d.BD e * 1 + keyBlk (128 * t) c * d.skt := by
      unfold kIdx lane keyBlk; ring
    have hx' := h1 e c hc.2 hc.1
    unfold rdB at hx
    rw [hidx] at hx
    rw [hx'] at hx
    have hxe : x = k.get2 e c := (Option.some.inj hx).symm
    rw [DType.val?_bf16]
    refine ⟨by rw [← hxe]; exact hxf, ?_⟩
    unfold kv valB rdB
    rw [hidx, hx']
  · simp only [hc, ite_false]
    rw [h2 e c (by unfold lane keyBlk; exact fun h => hc ⟨h.2, h.1⟩)]
    simp


/-- The `v` tile of the load. -/
theorem vFacts_of_load {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size)
    {t : Nat} {v : Tensor [BLOCK, d.BD] BF16}
    (h1 : ∀ (c : Fin BLOCK) (e : Fin d.BD), keyBlk (128 * t) c < lenOf d init (bOf d i) →
      lane d.BD e < d.Lk →
      init.read .bf16 (d.v.ptr ((startOf d init (bOf d i) * d.svt +
        PrefillAttnVarlen.kvh d (hOf d i) * d.svh + keyBlk (128 * t) c * d.svt +
        lane d.BD e * 1 : Nat) : Int)).addr = some (v.get2 c e))
    (h2 : ∀ (c : Fin BLOCK) (e : Fin d.BD),
      ¬ (keyBlk (128 * t) c < lenOf d init (bOf d i) ∧ lane d.BD e < d.Lk) →
      v.get2 c e = FloatV.ofPat 0) :
    VFacts d init i t v := by
  obtain ⟨hb, hh, -⟩ := prog_range i
  have hgN : PrefillAttnVarlen.kvh d (hOf d i) < d.HKV := kvh_lt hd hh
  intro c e
  by_cases hc : 128 * t + c < lenOf d init (bOf d i) ∧ e.val < d.Lk
  · simp only [hc, and_self, ite_true]
    obtain ⟨-, hfv, -⟩ := (hpre.2.2.2.2.2.2.2 (bOf d i) hb).2.1 (128 * t + c) _ e.val hc.1 hgN hc.2
    obtain ⟨x, hx, hxf⟩ := hfv
    have hidx : vIdx d (startOf d init (bOf d i) + (128 * t + c))
        (PrefillAttnVarlen.kvh d (hOf d i)) e.val = startOf d init (bOf d i) * d.svt +
        PrefillAttnVarlen.kvh d (hOf d i) * d.svh + keyBlk (128 * t) c * d.svt + lane d.BD e * 1 := by
      unfold vIdx lane keyBlk; ring
    have hx' := h1 c e hc.1 hc.2
    unfold rdB at hx
    rw [hidx] at hx
    rw [hx'] at hx
    have hxe : x = v.get2 c e := (Option.some.inj hx).symm
    rw [DType.val?_bf16]
    refine ⟨by rw [← hxe]; exact hxf, ?_⟩
    unfold vv valB rdB
    rw [hidx, hx']
  · simp only [hc, ite_false]
    rw [h2 c e (by unfold lane keyBlk; exact fun h => hc ⟨h.1, h.2⟩)]
    simp


/-- Lane `(r, c)` of the mask of the score: `0` where the key is in the sequence (and at or
before the row when causal), `-∞` elsewhere. -/
theorem mask_lane {d : Inst} (hd : d.ok) {sm len : I32} {t mN L : Nat} (hsm : sm.toInt = mN)
    (hlen : len.toInt = L) (hmN : mN < nBlocks d) (hL : L ≤ 2 ^ 22) (ht : 128 * t < L)
    {mask : Tensor [BLOCK, BLOCK] F32} (hmask : mask = (if d.causal = true then
          tl.where
            ((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).row BLOCK).ltS len |>.and
              ((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).row BLOCK).le
                ((sm * IntV.ofNat BLOCK + tl.arange 0 BLOCK).col BLOCK)))
              (bc zf32) (bc ninf32)
          else tl.where ((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).row BLOCK).ltS len)
            (bc zf32) (bc ninf32)))
    (r c : Fin BLOCK) :
    mask.get2 r c = if (128 * t + c < L ∧ (d.causal = true → 128 * t + c ≤ mN * BLOCK + r))
      then zf32 else ninf32 := by
  have hsn : (IntV.ofInt (0 + (t : Int) * 128) : I32).toInt = ((128 * t : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (by omega)]; push_cast; ring
  have hc := startN_lane hsn ht (by omega : L ≤ 2 ^ 22) c
  have hr := offsM_lane hd hsm hmN r
  have hc' : ((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK)).get (c, ())).toInt = ((128 * t + c : Nat) : Int) := hc
  have hr' : (((sm * IntV.ofNat BLOCK + tl.arange 0 BLOCK)).get (r, ())).toInt = ((mN * BLOCK + r : Nat) : Int) := hr
  cases hcs : d.causal
  · have e : mask.get2 r c = if IntV.lt (((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).get (c, ())) len = true then zf32 else ninf32 := by
      rw [hmask]; simp only [hcs]; rfl
    rw [e, IntV.lt_eq, hc', hlen]
    split_ifs <;> first | rfl | (exfalso; simp_all; omega)
  · have e : mask.get2 r c = if (IntV.lt (((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).get (c, ())) len && IntV.le (((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).get (c, ())) ((sm * IntV.ofNat BLOCK + tl.arange 0 BLOCK).get (r, ())) ) = true then zf32 else ninf32 := by
      rw [hmask]; simp only [hcs]; rfl
    simp only [e, Bool.and_eq_true, IntV.lt_eq, IntV.le_eq, hc', hr', hlen]
    split_ifs <;> first | rfl | (exfalso; simp_all; omega)

/-- A sum over `Fin BD` of terms that vanish from `Lk` on is the sum over `range Lk`. -/
theorem sum_trunc {BD Lk : ℕ} (h : Lk ≤ BD) (g : ℕ → ℝ) :
    ∑ e : Fin BD, (if e.val < Lk then g e else 0) = ∑ e ∈ Finset.range Lk, g e := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j < Lk then g j else 0) BD]
  rw [← Finset.sum_subset (Finset.range_subset_range.mpr h)]
  · exact Finset.sum_congr rfl fun j hj => by simp [Finset.mem_range.mp hj]
  · intro j _ hj
    have : ¬ j < Lk := by simpa using hj
    simp [this]

/-- The sums of the score lane over the head dimension, for a query row `a` and a key row `b`
that is `0` when the key is masked off (`P` false). -/
theorem qk_sums {BD Lk : ℕ} (h : Lk ≤ BD) (a b : ℕ → ℚ) (P : Prop) [Decidable P] :
    (∑ e : Fin BD, (((if e.val < Lk then a e else 0 : ℚ) : ℝ) *
        ((if P ∧ e.val < Lk then b e else 0 : ℚ) : ℝ)) =
      if P then ∑ e ∈ Finset.range Lk, ((a e : ℚ) : ℝ) * ((b e : ℚ) : ℝ) else 0) ∧
    (∑ e : Fin BD, ((Device.liftMag .bf16 (if e.val < Lk then a e else 0) *
        Device.liftMag .bf16 (if P ∧ e.val < Lk then b e else 0) : ℚ) : ℝ) =
      if P then ∑ e ∈ Finset.range Lk,
        ((Device.liftMag .bf16 (a e) * Device.liftMag .bf16 (b e) : ℚ) : ℝ) else 0) := by
  by_cases hP : P
  · simp only [hP, true_and, ite_true]
    constructor
    · rw [← sum_trunc h (fun e => ((a e : ℚ) : ℝ) * ((b e : ℚ) : ℝ))]
      refine Finset.sum_congr rfl fun e _ => ?_
      by_cases he : e.val < Lk <;> simp [he]
    · rw [← sum_trunc h (fun e => ((Device.liftMag .bf16 (a e) * Device.liftMag .bf16 (b e) : ℚ) : ℝ))]
      refine Finset.sum_congr rfl fun e _ => ?_
      by_cases he : e.val < Lk <;> simp [he]
  · simp [hP]

/-- A key `j` is attended by row `p` of its sequence exactly when it is in the sequence and, when
causal, not after the row. -/
theorem attend_iff {d : Inst} {init : Mem} {b p j : Nat} (hp : p < lenOf d init b) :
    j ≤ lastKey d init b p ↔ (j < lenOf d init b ∧ (d.causal = true → j ≤ p)) := by
  unfold lastKey
  cases d.causal <;> simp <;> omega

/-- The scores of the lanes of `qk = qk₁ * sm_scale + mask`. -/
theorem score_facts {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) {t : Nat} (ht : t < nTrips d init i)
    {sm len : I32} (hsm : sm.toInt = mOf d i) (hlen : len.toInt = lenOf d init (bOf d i))
    {q : Tensor [BLOCK, d.BD] BF16} (hq : QFacts d init i q)
    {k : Tensor [d.BD, BLOCK] BF16} (hk : KFacts d init i t k)
    {z qk₁ qk : Tensor [BLOCK, BLOCK] F32}
    (hz : ∀ x, z.get x = FloatV.ofPat 0)
    (hdot : P.dot cfgBF16 q k (tl.zeros [BLOCK, BLOCK] .f32) qk₁ ∨ ∃ d', P.dot cfgBF16 q k z d' ∧
      ∀ x, AccAdd true .f32 (d'.get x) ((tl.zeros [BLOCK, BLOCK] DType.f32).get x) (qk₁.get x))
    {mask : Tensor [BLOCK, BLOCK] F32} (hmask : mask = (if d.causal = true then
          tl.where
            (((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).row BLOCK).ltS len).and
              ((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).row BLOCK).le
                ((sm * IntV.ofNat BLOCK + tl.arange 0 BLOCK).col BLOCK)))
              (bc zf32) (bc ninf32)
          else tl.where ((((IntV.ofInt (0 + (t : Int) * 128) : I32) + tl.arange 0 BLOCK).row BLOCK).ltS len)
            (bc zf32) (bc ninf32)))
    (hqk : ∀ x, FloatV.NaNEq (FloatV.add (FloatV.mul (qk₁.get x) ((bc (FloatV.hostF32 d.sm_scale) :
        Tensor [BLOCK, BLOCK] F32).get x)) (mask.get x)) (qk.get x) ∨
      (true = true ∧ FloatV.NaNEq (FloatV.fma (qk₁.get x) ((bc (FloatV.hostF32 d.sm_scale) :
        Tensor [BLOCK, BLOCK] F32).get x) (mask.get x)) (qk.get x))) :
    ScoreFacts d init i t qk := by
  intro r hr
  obtain ⟨hb, hh, hmb⟩ := prog_range i
  obtain ⟨hSL, hlm⟩ := seq_facts hpre hb
  obtain ⟨hmT, hT, hcaus, -⟩ := trip_facts hd hpre i ht
  have hvals := (hpre.2.2.2.2.2.2.2 (bOf d i) hb).2.2
  have hmax := hd.maxLen_le
  have hml := fun c => mask_lane hd hsm hlen hmb (by omega) hT hmask r c
  have hLk := lastKey_lt (d := d) (init := init) hr
  have hX1 : Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) ≤ 2 ^ 12 :=
    Finset.sup'_le _ _ fun j hj => hvals _ _ _ hh hr
      (lt_of_le_of_lt (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)) hLk)
  have hq' : ∀ e : Fin d.BD, DType.val? .bf16 (q.get2 r e) = some
      (if e.val < d.Lk then qv d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e else 0) := by
    intro e
    rw [hq r e]
    simp only [hr, true_and]
  obtain ⟨hs1, hs2, hs3⟩ := hd.scale
  have hz0 : ∀ x y : Fin BLOCK, DType.val? .f32 ((tl.zeros [BLOCK, BLOCK] DType.f32).get2 x y) = some 0 := by
    intro x y; simp only [tl.zeros, Tensor.get2]
    exact (by decide +kernel : DType.val? .f32 (DType.f32.ofBits 0) = some 0)
  have hzz : ∀ x y : Fin BLOCK, DType.val? .f32 (z.get2 x y) = some 0 := by
    intro x y; rw [show z.get2 x y = FloatV.ofPat 0 from hz _]; simp
  constructor
  · intro c hc
    simp only [Num.blk, Finset.mem_filter, Finset.mem_range] at hc
    have hatt := (attend_iff hr).mp hc.2
    have hmk := hml c
    have hatt' : 128 * t + c < lenOf d init (bOf d i) ∧ (d.causal = true → 128 * t + c ≤ mOf d i * BLOCK + r) := by
      simpa [BLOCK] using hatt
    have hmk' : mask.get2 r c = zf32 := by
      rw [hmk]; split_ifs
      all_goals rfl
    have hy : FloatV.NaNEq (FloatV.add (FloatV.mul (qk₁.get2 r c) (FloatV.hostF32 d.sm_scale)) zf32)
        (qk.get2 r c) ∨ (true = true ∧ FloatV.NaNEq (FloatV.fma (qk₁.get2 r c)
          (FloatV.hostF32 d.sm_scale) zf32) (qk.get2 r c)) := by
      rw [← hmk']; exact hqk (r, c, ())
    have hk' : ∀ e : Fin d.BD, DType.val? .bf16 (k.get2 e c) = some
        (if 128 * t + c < lenOf d init (bOf d i) ∧ e.val < d.Lk then
          kv d init (bOf d i) (hOf d i) (128 * t + c) e else 0) := fun e => hk e c
    have hsum := qk_sums hd.lk_le (fun e => qv d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e)
      (fun e => kv d init (bOf d i) (hOf d i) (128 * t + c) e) (128 * t + c < lenOf d init (bOf d i))
    beta_reduce at hsum
    have hXj : sig d * ∑ e ∈ Finset.range d.Lk, ((Device.liftMag .bf16
        (qv d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e) * Device.liftMag .bf16
        (kv d init (bOf d i) (hOf d i) (128 * t + c) e) : ℚ) : ℝ) ≤
        Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) :=
      Finset.le_sup' (fun j => scoreMag d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) j)
        (Finset.mem_range.mpr (Nat.lt_succ_of_le hc.2))
    have hres := score_lane hP (fusion := true) (BD := d.BD) hd.bd_le hdot r c (hzz r c) (hz0 r c)
      hq' hk' (sm := d.sm_scale) hs1 hs2 hs3
      (X := Xrow d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r))
      (by rw [hsum.2]; simp only [hatt.1, ite_true]; exact hXj) hX1 hy
    refine ⟨hres.1, ?_⟩
    have := hres.2
    rw [hsum.1] at this
    simp only [hatt.1, ite_true] at this
    exact this
  · intro c hc
    have hnat : ¬ (128 * t + c ≤ lastKey d init (bOf d i) (mOf d i * BLOCK + r)) := by
      intro h; exact hc (by simp only [Num.blk, Finset.mem_filter, Finset.mem_range]; exact ⟨c.isLt, h⟩)
    have hnatt := fun h => hnat ((attend_iff hr).mpr h)
    have hmk := hml c
    have hnatt' : ¬ (128 * t + c < lenOf d init (bOf d i) ∧ (d.causal = true → 128 * t + c ≤ mOf d i * BLOCK + r)) := by
      simpa [BLOCK] using hnatt
    have hmk' : mask.get2 r c = ninf32 := by
      rw [hmk]; split_ifs
      all_goals rfl
    have hy : FloatV.NaNEq (FloatV.add (FloatV.mul (qk₁.get2 r c) (FloatV.hostF32 d.sm_scale)) ninf32)
        (qk.get2 r c) ∨ (true = true ∧ FloatV.NaNEq (FloatV.fma (qk₁.get2 r c)
          (FloatV.hostF32 d.sm_scale) ninf32) (qk.get2 r c)) := by
      rw [← hmk']; exact hqk (r, c, ())
    have hk' : ∀ e : Fin d.BD, DType.val? .bf16 (k.get2 e c) = some
        (if 128 * t + c < lenOf d init (bOf d i) ∧ e.val < d.Lk then
          kv d init (bOf d i) (hOf d i) (128 * t + c) e else 0) := fun e => hk e c
    have hsum := qk_sums hd.lk_le (fun e => qv d init (bOf d i) (hOf d i) (mOf d i * BLOCK + r) e)
      (fun e => kv d init (bOf d i) (hOf d i) (128 * t + c) e) (128 * t + c < lenOf d init (bOf d i))
    beta_reduce at hsum
    refine score_lane_masked hP (fusion := true) (BD := d.BD) hd.bd_le hdot r c (hzz r c) (hz0 r c)
      hq' hk' (sm := d.sm_scale) hs1 hs2 hs3 ?_ hy
    rw [hsum.2]
    by_cases hj : 128 * t + c < lenOf d init (bOf d i)
    · simp only [hj, ite_true]
      exact hvals _ _ _ hh hr hj
    · simp only [hj, ite_false, mul_zero]; norm_num


/-- **One trip of the loop** keeps the invariant and writes nothing. -/
theorem trip_safe_proof {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) {cb ch sm kvh len st : I32}
    (hs : Scal d init i cb ch sm kvh len st) {q : Tensor [BLOCK, d.BD] BF16}
    (hq : QFacts d init i q) {t : Nat} (ht : t < nTrips d init i)
    (st3 : Tensor [BLOCK] F32 × Tensor [BLOCK] F32 × Tensor [BLOCK, d.BD] F32) (m : Mem)
    (hI : LoopInv d init i t st3 m) :
    Safe ((launch d).env P) (Rf d) ∅ (fun st' m' => LoopInv d init i (t + 1) st' m') m
      (loopBodyD d cb ch sm kvh len st q (IntV.ofInt (0 + (t : Int) * 128)) st3) := by
  obtain ⟨m_i, l_i, acc⟩ := st3
  have hm0 := hI.1
  subst hm0
  obtain ⟨hb, hh, hmb⟩ := prog_range i
  obtain ⟨hmT, hT, -, ht1024⟩ := trip_facts hd hpre i ht
  obtain ⟨hSL, -⟩ := seq_facts hpre hb
  have hgN : PrefillAttnVarlen.kvh d (hOf d i) < d.HKV := kvh_lt hd hh
  have hg : kvh.toInt = ((PrefillAttnVarlen.kvh d (hOf d i) : Nat) : Int) := by
    rw [hs.kvh]; unfold PrefillAttnVarlen.kvh; push_cast; rfl
  have hsn : (IntV.ofInt (0 + (t : Int) * 128) : I32).toInt = ((128 * t : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (by omega)]; push_cast; ring
  unfold loopBodyD loopBody
  simp only []
  refine convert_zero_safe fun ok hok => ?_
  refine safe_load_tileAt d.k (k_ptrs hd hs.st hg hsn hSL hgN hT) (k_mask hd hs.len hsn
    (by omega) hT) (k_len hd hSL hgN) hpre.2.1 (k_Rf hd hSL hgN) fun k hk1 hk2 => ?_
  have hKF : KFacts d m i t k := kFacts_of_load hd hpre i hk1
    (fun e c hec => by rw [hk2 e c hec _ rfl]; exact hok _)
  refine safe_dot_add (hP.dotTotal _ _ _ _) fun qk₁ hqk₁ => ?_
  refine safe_mul_add fun qk hqk => ?_
  have hQK : ScoreFacts d m i t qk := score_facts hP hd hpre i ht hs.sm hs.len hq hKF
    (z := tl.zeros [BLOCK, BLOCK] .f32) (fun _ => rfl) hqk₁ rfl hqk
  refine safe_max1 (by decide) true fun m_ij hmij => ?_
  refine safe_sub fun e1 he1 => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun p hp => ?_
  refine safe_sum1 (by decide) fun l_ij hlij => ?_
  refine safe_maximum fun m_new hmnew => ?_
  refine safe_sub fun e2 he2 => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun alpha halpha => ?_
  refine safe_sub fun e3 he3 => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun beta hbeta => ?_
  refine safe_mul fun a1 ha1 => ?_
  refine safe_mul fun a2 ha2 => ?_
  refine safe_addc fun lnew hlnew => ?_
  refine safe_div (fun _ => hP.divTotal _ _ _) fun pscale hps => ?_
  refine safe_mul fun p₁ hp₁ => ?_
  refine safe_div (fun _ => hP.divTotal _ _ _) fun d1 hd1 => ?_
  refine safe_mul fun ascale hasc => ?_
  refine safe_mul fun acc₁ hacc₁ => ?_
  have hsoft : SoftOps P true m_i l_i acc qk m_new lnew p₁ acc₁ :=
    ⟨m_ij, e1, p, l_ij, e2, alpha, e3, beta, a1, a2, pscale, d1, ascale, hmij, he1, hp, hlij,
      hmnew, he2, halpha, he3, hbeta, ha1, ha2, hlnew, hps, hp₁, hd1, hasc, hacc₁⟩
  refine convert_zero_safe fun okv hokv => ?_
  refine safe_load_tileAt d.v (v_ptrs hd hs.st hg hsn hSL hgN hT) (v_mask hd hs.len hsn
    (by omega) hT) (v_len hd hSL hgN) hpre.2.2.1 (v_Rf hd hSL hgN) fun v hv1 hv2 => ?_
  have hVF : VFacts d m i t v := vFacts_of_load hd hpre i hv1
    (fun c e hce => by rw [hv2 c e hce _ rfl]; exact hokv _)
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun p₂ hp₂ => ?_
  refine safe_dot_add (hP.dotTotal _ _ _ _) fun acc₂ hacc₂ => ?_
  exact safe_ret ⟨rfl, trip_conclude hP hd hpre i ht hI.2 hQK hVF hsoft hp₂ hacc₂⟩

end Claims.PrefillAttnVarlen

end
