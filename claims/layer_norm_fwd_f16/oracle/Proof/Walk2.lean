import Proof.Walk1
import Proof.RstdStage

/-!
# Lines 74–81: the row's `rstd`
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- What the walk of lines 74–81 delivers: a finite `rstd`, within its tolerance of the exact one,
with the two finer bounds the standardised values use. -/
def RstdOK (r : RowP) (v : F32) : Prop :=
  v.isFinite = true ∧ 0 < v.toReal ∧ |v.toReal - r.R| ≤ r.rstdTol ∧
    (r.cond ≤ 1 / 2 → |v.toReal - r.R| ≤ r.thetaS * r.R) ∧
    (¬ r.cond ≤ 1 / 2 → v.toReal ≤ (1 + r.gam) / Real.sqrt r.eps) ∧ v.toReal ≤ 2 ^ 65

set_option maxHeartbeats 1600000 in
/-- **`rstd`** (lines 74–81): the second loop's sum of squares, the `tl.sum`, the division by `N`,
`var + eps`, `tl.sqrt` and `1 / ·`. -/
theorem safe_rstd (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat} (ht : t < d.M) {c : Ctx}
    (hc : c.pid 0 = t) {m : Mem} (hm : MemOk d init m) {E : Env} (hP : Device E.P) {W : Set ℤ}
    {α : Type} {Q : α → Mem → Prop} {k : Tensor [] F32 → Prog α} (mq : Tensor [] F32)
    (hmq : mq.item.isFinite = true)
    (hmq' : |mq.item.toReal - (rowP d hd init hpre t ht).mu| ≤
      11 / 20 * (rowP d hd init hpre t ht).Delta)
    (hk : ∀ rs : Tensor [] F32, RstdOK (rowP d hd init hpre t ht) rs.item →
      Safe E (R d t) W Q m (k rs)) :
    Safe E (R d t) W Q m
      ((tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.N) (IntV.ofNat d.BLOCK)
          (tl.zeros [d.BLOCK] DType.f32) fun off _var =>
          (other16 d.BLOCK).bind fun l1 =>
            (tl.load (d.x.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32) +
                  (off + tl.arange 0 d.BLOCK))
                (some ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N))) (some l1)).bind
              fun l2 => (tl.cvt l2).bind fun x =>
                (tl.subc x (bc mq.item)).bind fun dd =>
                  tl.mul_add
                    (tl.where ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N)) dd
                      (bc (FloatV.ofPat 0).litF32))
                    (tl.where ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N)) dd
                      (bc (FloatV.ofPat 0).litF32)) _var).bind
        fun _var => (tl.itof (Tensor.scalar (IntV.ofNat d.N : I32))).bind fun nf2 =>
          (tl.sum _var).bind fun s2 => (tl.div (Tensor.scalar s2) nf2).bind fun v2 =>
            (tl.addc (Tensor.scalar v2.item) (Tensor.scalar (FloatV.hostF32 d.eps))).bind fun s3 =>
              (tl.sqrt s3).bind fun r1 =>
                (tl.div (Tensor.scalar (FloatV.ofPat 4607182418800017408).litF32) r1).bind k) := by
  have hN := hd.n_ge
  have hB := hd.block_le
  have hB' : d.BLOCK ≤ 2 ^ 15 := by omega
  rw [forRange_one (by omega) hd.n_le_block (by omega)]
  simp only [Prog.bind_assoc]
  refine safe_loadX hd hpre ht hc hm fun a ha => ?_
  refine safe_subc fun dd hdd => ?_
  refine safe_mul_add fun v hv => ?_
  refine safe_itof Format.binary32_std fun nf hnf => ?_
  refine safe_sum (by have := hd.n_le_block; omega) fun S hS => ?_
  obtain ⟨tr, hcov, hdep, heval⟩ := hS
  refine safe_div (fun i => hP.divTotal .binary32 _ _) fun q hq => ?_
  refine safe_addc fun s3 hs3 => ?_
  refine safe_math (fun i => hP.mathTotal .sqrt .binary32 _) fun r1 hr1 => ?_
  refine safe_div (fun i => hP.divTotal .binary32 _ _) fun rs hrs => ?_
  set r := rowP d hd init hpre t ht with hr
  set mf := mq.item with hmf
  have hB31 : d.BLOCK ≤ 2 ^ 31 := by omega
  have hN31 : d.N < 2 ^ 31 := by have := hd.n_le; omega
  have hbe : d.BLOCK = 2 ^ r.D := r.block_eq
  have hDle := r.D_le
  have hxl := r.xl_le
  have hmb : |mf.toReal| ≤ 65505 := by
    have h1 := r.mu_le
    have h2 := r.Delta_le
    have h3 : |mf.toReal| ≤ |r.mu| + |mf.toReal - r.mu| := by
      have := abs_add_le r.mu (mf.toReal - r.mu)
      simpa using this
    have h4 : |mf.toReal - r.mu| ≤ 1 := by linarith
    linarith
  -- the masked difference lanes `where(cols < N, x - mean, 0)`
  obtain ⟨dw, hdwdef⟩ : ∃ dw : Tensor [d.BLOCK] F32, dw = tl.where
      (((IntV.ofInt 0 : I32) + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N)) dd
      (bc (FloatV.ofPat 0).litF32) := ⟨_, rfl⟩
  rw [← hdwdef] at hv
  have hdw : ∀ i : Idx [d.BLOCK],
      dw.get i = if (i.1 : Nat) < d.N then dd.get i else FloatV.ofPat 0 := by
    intro i
    rw [hdwdef]
    have hmk := cols_mask hB31 hN31 i
    simp only [tl.where, Tensor.get_where, hmk, Tensor.get_full, FloatV.litF32_ofPat_zero]
    by_cases h : (i.1 : Nat) < d.N <;> simp [h]
  -- the lanes of `dw`: finite, small, and a computed `x j - mean` on the row
  have hlane : ∀ i : Idx [d.BLOCK], (dw.get i).isFinite = true ∧ |(dw.get i).toReal| ≤ 2 ^ 18 ∧
      ((i.1 : Nat) < d.N → LaneErr (r.x i.1) mf.toReal (dw.get i).toReal) := by
    intro i
    by_cases hi : (i.1 : Nat) < d.N
    · rw [hdw i, ite_eq_left hi]
      obtain ⟨hf, hv⟩ := ha i
      have hxi : r.xl i.1 = r.x i.1 := by unfold RowP.xl; exact ite_eq_left hi
      have hxb : |(a.get i).toReal| ≤ 2 ^ 20 := by
        rw [hv]; exact (hxl _).trans (by norm_num)
      obtain ⟨hf2, hl⟩ := lane_sub hf hmq hxb (hmb.trans (by norm_num)) (hdd i)
      rw [hv, hxi] at hl
      refine ⟨hf2, ?_, fun _ => hl⟩
      have hx := r.x_le i.1 hi
      unfold LaneErr u32 eta32 at hl
      have h1 : |(dd.get i).toReal| ≤ |r.x i.1 - mf.toReal| +
          (2 ^ (-24 : ℤ) * |r.x i.1 - mf.toReal| + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) *
            (|r.x i.1| + |mf.toReal|) + 3 * 2 ^ (-150 : ℤ)) := by
        have := abs_sub_abs_le_abs_sub (dd.get i).toReal (r.x i.1 - mf.toReal)
        linarith
      have h2 : |r.x i.1 - mf.toReal| ≤ 131009 := by
        have := abs_sub (r.x i.1) mf.toReal
        linarith
      have h3 : 2 ^ (-24 : ℤ) * |r.x i.1 - mf.toReal| + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) *
          (|r.x i.1| + |mf.toReal|) + 3 * 2 ^ (-150 : ℤ) ≤ 1 := by
        have : (2 : ℝ) ^ (-24 : ℤ) * |r.x i.1 - mf.toReal| ≤ 2 ^ (-24 : ℤ) * 131009 := by gcongr
        have h4 : 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) * (|r.x i.1| + |mf.toReal|) ≤
            2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) * (65504 + 65505) := by gcongr
        have h5 : (2 : ℝ) ^ (-24 : ℤ) * 131009 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) *
            (65504 + 65505) + 3 * 2 ^ (-150 : ℤ) ≤ 1 := by norm_num
        linarith
      linarith
    · rw [hdw i, ite_eq_right hi]
      exact ⟨FloatV.isFinite_ofPat_zero Format.binary32_regular,
        by rw [FloatV.toReal_ofPat_zero]; norm_num, fun h => absurd h hi⟩
  -- the squares
  have hvl : ∀ i : Idx [d.BLOCK], (v.get i).isFinite = true ∧
      |(v.get i).toReal - (dw.get i).toReal ^ 2| ≤ 3 * u32 * (dw.get i).toReal ^ 2 + 4 * eta32 :=
    fun i => muladd_sq (fusion := E.fusion) (hlane i).1 ((hlane i).2.1.trans (by norm_num)) (hv i)
  -- the real lanes
  obtain ⟨dR, hdR⟩ : ∃ dR : ℕ → ℝ, dR = fun j =>
      if h : j < d.BLOCK then (dw.get ((⟨j, h⟩ : Fin d.BLOCK), ())).toReal else 0 := ⟨_, rfl⟩
  obtain ⟨l, hl⟩ : ∃ l : ℕ → F32, l = fun i =>
      if h : i < d.BLOCK then v.get1 ⟨i, h⟩ else default := ⟨_, rfl⟩
  rw [← hl] at heval
  have hlget : ∀ i (h : i < d.BLOCK), l i = v.get ((⟨i, h⟩ : Fin d.BLOCK), ()) := by
    intro i h; rw [hl]; simp [h]; rfl
  have hdRget : ∀ i (h : i < d.BLOCK), dR i = (dw.get ((⟨i, h⟩ : Fin d.BLOCK), ())).toReal := by
    intro i h; rw [hdR]; simp [h]
  have hd1 : ∀ j, j < r.N → LaneErr (r.x j) mf.toReal (dR j) := by
    intro j hj
    have hjB : j < d.BLOCK := lt_of_lt_of_le hj hd.n_le_block
    rw [hdRget j hjB]
    exact (hlane ((⟨j, hjB⟩ : Fin d.BLOCK), ())).2.2 hj
  have hd0 : ∀ j, r.N ≤ j → j < r.B → dR j = 0 := by
    intro j hj hjB
    have hjB' : j < d.BLOCK := hjB
    rw [hdRget j hjB', hdw ((⟨j, hjB'⟩ : Fin d.BLOCK), ()), ite_eq_right (by have : d.N ≤ j := hj; simpa using this),
      FloatV.toReal_ofPat_zero]
  have hv1 : ∀ j, j < r.B → |(l j).toReal - dR j ^ 2| ≤ 3 * u32 * dR j ^ 2 + 4 * eta32 := by
    intro j hj
    rw [hlget j hj, hdRget j hj]
    exact (hvl _).2
  have hlfin : ∀ i, i < d.BLOCK → (l i).isFinite = true := fun i h => by
    rw [hlget i h]; exact (hvl _).1
  have hlb : ∀ i, i < d.BLOCK → |(l i).toReal| ≤ 2 ^ 40 := fun i h => by
    have h1 := hv1 i h
    have h2 : |dR i| ≤ 2 ^ 18 := by
      rw [hdRget i h]; exact (hlane _).2.1
    have h3 : dR i ^ 2 ≤ 2 ^ 36 := by
      have := abs_nonneg (dR i)
      nlinarith [sq_abs (dR i)]
    have h4 : |(l i).toReal| ≤ dR i ^ 2 + (3 * u32 * dR i ^ 2 + 4 * eta32) := by
      have := abs_sub_abs_le_abs_sub (l i).toReal (dR i ^ 2)
      have h6 : |dR i ^ 2| = dR i ^ 2 := abs_of_nonneg (sq_nonneg _)
      linarith
    have h5 : 3 * u32 * dR i ^ 2 + 4 * eta32 ≤ 2 ^ 14 := by
      unfold u32 eta32
      have : (3 : ℝ) * 2 ^ (-24 : ℤ) * dR i ^ 2 ≤ 3 * 2 ^ (-24 : ℤ) * 2 ^ 36 := by gcongr
      have : (3 : ℝ) * 2 ^ (-24 : ℤ) * 2 ^ 36 + 4 * 2 ^ (-150 : ℤ) ≤ 2 ^ 14 := by norm_num
      linarith
    nlinarith
  have hbe : d.BLOCK = 2 ^ r.D := r.block_eq
  have hdep' : tr.depth ≤ r.D := by
    have h1 := hP.sumDepth d.BLOCK
    have h2 : Nat.clog 2 d.BLOCK = r.D := by rw [hbe, Nat.clog_pow _ _ (by norm_num)]
    unfold tl.sumBound at hdep
    omega
  obtain ⟨hSfin, hSerr⟩ := sum_real hDle hB' l hcov hdep' heval hlfin hlb
  obtain ⟨hnfin, hnval⟩ := itof_exact (N := d.N) (by have := hd.n_le; omega) (hnf ())
  have hS57 : |S.toReal| ≤ 2 ^ 57 := by
    have hA : ∑ i ∈ Finset.range d.BLOCK, |(l i).toReal| ≤ 2 ^ 55 := by
      calc _ ≤ ∑ i ∈ Finset.range d.BLOCK, (2 : ℝ) ^ 40 :=
            Finset.sum_le_sum fun i hi => hlb i (Finset.mem_range.mp hi)
        _ = d.BLOCK * 2 ^ 40 := by simp
        _ ≤ 2 ^ 15 * 2 ^ 40 := by
            have : (d.BLOCK : ℝ) ≤ 2 ^ 15 := by exact_mod_cast hB'
            gcongr
        _ = 2 ^ 55 := by norm_num
    have hT : |∑ i ∈ Finset.range d.BLOCK, (l i).toReal| ≤ 2 ^ 55 :=
      (Finset.abs_sum_le_sum_abs _ _).trans hA
    have hp := one_add_u_pow_le hDle
    have hp0 : 0 ≤ (1 + u32) ^ r.D := by unfold u32; positivity
    have hA0 : 0 ≤ ∑ i ∈ Finset.range d.BLOCK, |(l i).toReal| :=
      Finset.sum_nonneg fun i _ => abs_nonneg _
    have hn' : ((d.BLOCK - 1 : ℕ) : ℝ) ≤ 2 ^ 15 := by
      have : d.BLOCK - 1 ≤ 2 ^ 15 := by omega
      exact_mod_cast this
    have he : eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * (1 + u32) ^ r.D ≤ 1 := by
      have : eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * (1 + u32) ^ r.D ≤ eta32 * 2 ^ 15 * 2 := by
        unfold eta32; gcongr
      refine this.trans ?_
      unfold eta32; norm_num
    have hE : ((1 + u32) ^ r.D - 1) * ∑ i ∈ Finset.range d.BLOCK, |(l i).toReal| ≤ 2 ^ 55 := by
      nlinarith
    have := abs_sub_abs_le_abs_sub S.toReal (∑ i ∈ Finset.range d.BLOCK, (l i).toReal)
    nlinarith [abs_nonneg S.toReal]
  have hN2 : (2 : ℝ) ≤ d.N := by exact_mod_cast hd.n_ge
  have hN0 : ((nf.get ()).toReal) ≠ 0 := by
    rw [hnval]; exact_mod_cast (by omega : d.N ≠ 0)
  have hq' : |S.toReal / (nf.get ()).toReal| ≤ 2 ^ 126 := by
    rw [abs_div, hnval, abs_of_nonneg (show (0 : ℝ) ≤ d.N from Nat.cast_nonneg _)]
    rw [div_le_iff₀ (by linarith)]
    nlinarith [abs_nonneg S.toReal]
  obtain ⟨hqfin, hqerr⟩ := div_real hP hSfin hnfin hN0 hq' (hq ())
  rw [hnval] at hqerr
  obtain ⟨var, hvardef⟩ : ∃ var : F32, var = q.item := ⟨_, rfl⟩
  have hqv : q.get () = var := hvardef.symm
  rw [hqv] at hqfin hqerr
  rw [← hvardef] at hs3
  have hvar57 : |var.toReal| ≤ 2 ^ 57 := by
    have h1 : |S.toReal / d.N| ≤ 2 ^ 56 := by
      rw [abs_div, abs_of_nonneg (show (0 : ℝ) ≤ d.N from Nat.cast_nonneg _), div_le_iff₀ (by linarith)]
      nlinarith [abs_nonneg S.toReal]
    have h2 := abs_sub_abs_le_abs_sub var.toReal (S.toReal / d.N)
    have h3 : (2 : ℝ) ^ (-22 : ℤ) * |S.toReal / d.N| + 2 ^ (-148 : ℤ) ≤ 2 ^ 56 := by
      have : (2 : ℝ) ^ (-22 : ℤ) * |S.toReal / d.N| ≤ 2 ^ (-22 : ℤ) * 2 ^ 56 := by gcongr
      have : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 56 + 2 ^ (-148 : ℤ) ≤ 2 ^ 56 := by norm_num
      linarith
    linarith
  -- `eps` as a `float32`
  have hepos : 0 < r.eps := lt_of_lt_of_le (by positivity) r.eps_ge
  have heps_le : r.eps ≤ 2 ^ 126 := by
    have := r.eps_le; rwa [show ((2 : ℝ) ^ (126 : ℤ)) = 2 ^ 126 by norm_num] at this
  have hepsR : r.eps = (d.eps).toReal := rfl
  obtain ⟨hefin, heerr⟩ := hostF32_err (c := d.eps) hd.eps_fin (by
    rw [← hepsR, abs_of_pos hepos]; exact heps_le)
  rw [← hepsR, abs_of_pos hepos] at heerr
  have heb : |(FloatV.hostF32 d.eps).toReal| ≤ 2 ^ 126 + 2 ^ 110 := by
    have h1 := abs_sub_abs_le_abs_sub (FloatV.hostF32 d.eps).toReal r.eps
    rw [abs_of_pos hepos] at h1
    have h3 : u32 * r.eps + eta32 ≤ 2 ^ 110 := by
      unfold u32 eta32
      have : (2 : ℝ) ^ (-24 : ℤ) * r.eps ≤ 2 ^ (-24 : ℤ) * 2 ^ 126 := by gcongr
      have : (2 : ℝ) ^ (-24 : ℤ) * 2 ^ 126 + 2 ^ (-150 : ℤ) ≤ 2 ^ 110 := by norm_num
      linarith
    linarith
  obtain ⟨hsfin, hserr⟩ := addc_real (fusion := E.fusion) hqfin hefin (by
    have : |var.toReal| + |(FloatV.hostF32 d.eps).toReal| ≤ 2 ^ 57 + (2 ^ 126 + 2 ^ 110) :=
      add_le_add hvar57 heb
    have h2 : (2 : ℝ) ^ 57 + 2 ^ 110 ≤ 2 ^ 111 := by norm_num
    linarith) (hs3 ())
  obtain ⟨hs1, hs2⟩ := r.s_stage mf.toReal hmq' dR (fun j => (l j).toReal) S.toReal var.toReal
    (FloatV.hostF32 d.eps).toReal (s3.get ()).toReal hd1 hd0 hv1 hSerr hqerr heerr hserr
  -- `tl.sqrt`
  have hlo : (2 : ℝ) ^ (-126 : ℤ) ≤ (s3.get ()).toReal := by
    have h1 := r.eps_ge
    have h0 : (0 : ℝ) ≤ 1 - 41 / 10 * u32 := by unfold u32; norm_num
    have h2 : (2 : ℝ) ^ (-125 : ℤ) * (1 - 41 / 10 * u32) ≤ r.eps * (1 - 41 / 10 * u32) :=
      mul_le_mul_of_nonneg_right h1 h0
    have h3 : (2 : ℝ) ^ (-126 : ℤ) ≤ 2 ^ (-125 : ℤ) * (1 - 41 / 10 * u32) - 22 * eta32 := by
      unfold u32 eta32; norm_num
    calc (2 : ℝ) ^ (-126 : ℤ) ≤ 2 ^ (-125 : ℤ) * (1 - 41 / 10 * u32) - 22 * eta32 := h3
      _ ≤ r.eps * (1 - 41 / 10 * u32) - 22 * eta32 := by linarith [h2]
      _ = r.eps - 41 / 10 * u32 * r.eps - 22 * eta32 := by ring
      _ ≤ (s3.get ()).toReal := hs1
  have hhi : (s3.get ()).toReal ≤ 2 ^ 127 := by
    have h1 := abs_sub_abs_le_abs_sub (s3.get ()).toReal
      (var.toReal + (FloatV.hostF32 d.eps).toReal)
    have h2 : |var.toReal + (FloatV.hostF32 d.eps).toReal| ≤ 2 ^ 126 + 2 ^ 111 :=
      (abs_add_le _ _).trans (by
        have : |var.toReal| + |(FloatV.hostF32 d.eps).toReal| ≤ 2 ^ 57 + (2 ^ 126 + 2 ^ 110) :=
          add_le_add hvar57 heb
        have h3 : (2 : ℝ) ^ 57 + 2 ^ 110 ≤ 2 ^ 111 := by norm_num
        linarith)
    have h3 : 3 * u32 * (|var.toReal| + |(FloatV.hostF32 d.eps).toReal|) + 4 * eta32 ≤ 2 ^ 111 := by
      have : |var.toReal| + |(FloatV.hostF32 d.eps).toReal| ≤ 2 ^ 126 + 2 ^ 111 :=
        (add_le_add hvar57 heb).trans (by norm_num)
      have h4 : 3 * u32 * (|var.toReal| + |(FloatV.hostF32 d.eps).toReal|) ≤ 3 * u32 * (2 ^ 126 + 2 ^ 111) := by
        unfold u32; gcongr
      have h5 : 3 * u32 * (2 ^ 126 + 2 ^ 111) + 4 * eta32 ≤ 2 ^ 111 := by unfold u32 eta32; norm_num
      linarith
    have h4 := le_abs_self (s3.get ()).toReal
    have h5 : (2 : ℝ) ^ 126 + 2 ^ 111 + 2 ^ 111 ≤ 2 ^ 127 := by norm_num
    linarith
  obtain ⟨hr1fin, hr1err⟩ := sqrt_real hP hsfin hlo hhi (hr1 ())
  -- `1 / ·`
  have hsq63 : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt (s3.get ()).toReal := by
    apply Real.le_sqrt_of_sq_le
    calc ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = 2 ^ (-126 : ℤ) := by norm_num
      _ ≤ _ := hlo
  have hr1lo : (2 : ℝ) ^ (-64 : ℤ) ≤ (r1.get ()).toReal := by
    have h1 := (abs_le.mp hr1err).1
    have h2 := hsq63
    norm_num at h1 h2 ⊢
    linarith
  have hr1pos : 0 < (r1.get ()).toReal := lt_of_lt_of_le (by positivity) hr1lo
  have hq1 : |(FloatV.ofPat 4607182418800017408 : F64).litF32.toReal / (r1.get ()).toReal| ≤ 2 ^ 126 := by
    rw [litF32_one.2, abs_of_pos (by positivity), div_le_iff₀ hr1pos]
    calc (1 : ℝ) ≤ 2 ^ 126 * 2 ^ (-64 : ℤ) := by norm_num
      _ ≤ 2 ^ 126 * (r1.get ()).toReal := by gcongr
  obtain ⟨hrsfin, hrserr⟩ := div_real hP litF32_one.1 hr1fin hr1pos.ne' (by
    simpa using hq1) (hrs ())
  rw [litF32_one.2, one_div, abs_of_pos (inv_pos.mpr hr1pos)] at hrserr
  have hrstd : |(rs.get ()).toReal - 1 / (r1.get ()).toReal| ≤
      2 ^ (-22 : ℤ) * (1 / (r1.get ()).toReal) + 2 ^ (-148 : ℤ) := by
    simpa [one_div] using hrserr
  obtain ⟨hpos, hbig⟩ := r.rstd_big_of_le (s3.get ()).toReal (r1.get ()).toReal (rs.get ()).toReal hs1
    (hhi.trans (by norm_num)) hr1err hrstd
  refine hk rs ⟨hrsfin, hpos, r.rstd_tol _ _ _ hs1 hs2 hr1err hrstd,
    fun hc => (r.rstd_small hc _ _ _ (hs2 hc).1 (hs2 hc).2 hr1err hrstd).2, fun _ => hbig, ?_⟩
  have hinv : 1 / (r1.get ()).toReal ≤ 2 ^ 64 := by
    rw [div_le_iff₀ hr1pos]
    calc (1 : ℝ) ≤ 2 ^ 64 * 2 ^ (-64 : ℤ) := by norm_num
      _ ≤ 2 ^ 64 * (r1.get ()).toReal := by gcongr
  have h1 := (abs_le.mp hrstd).2
  have h2 : (2 : ℝ) ^ (-22 : ℤ) * (1 / (r1.get ()).toReal) ≤ 2 ^ (-22 : ℤ) * 2 ^ 64 := by gcongr
  have h3 : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 64 + 2 ^ (-148 : ℤ) + 2 ^ 64 ≤ 2 ^ 65 := by norm_num
  have h4 : (rs.get ()).toReal ≤ 1 / (r1.get ()).toReal + (2 ^ (-22 : ℤ) * (1 / (r1.get ()).toReal) + 2 ^ (-148 : ℤ)) := by
    linarith
  calc (rs.get ()).toReal ≤ 2 ^ 64 + (2 ^ (-22 : ℤ) * 2 ^ 64 + 2 ^ (-148 : ℤ)) :=
        h4.trans (add_le_add hinv (add_le_add h2 (le_refl _)))
    _ ≤ 2 ^ 65 := by norm_num

end Claims.LayerNormFwdF16
