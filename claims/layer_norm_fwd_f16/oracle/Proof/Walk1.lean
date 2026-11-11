import Proof.LoadX
import Proof.Glue
import Proof.Mean

/-!
# Lines 67–72: the row's mean
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- **The stored mean** (lines 67–72): the first loop's sum of the row, the `tl.sum` and the division
by `N` give a finite `mean` within `11/20 · Delta` of the exact mean. -/
theorem safe_mean (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat} (ht : t < d.M) {c : Ctx}
    (hc : c.pid 0 = t) {m : Mem} (hm : MemOk d init m) {E : Env} (hP : Device E.P) {W : Set ℤ}
    {α : Type} {Q : α → Mem → Prop} {k : Tensor [] F32 → Prog α}
    (hk : ∀ q : Tensor [] F32, q.item.isFinite = true →
      |q.item.toReal - (rowP d hd init hpre t ht).mu| ≤ 11 / 20 * (rowP d hd init hpre t ht).Delta →
      Safe E (R d t) W Q m (k q)) :
    Safe E (R d t) W Q m
      ((tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.N) (IntV.ofNat d.BLOCK)
          (tl.zeros [d.BLOCK] DType.f32) fun off _mean =>
          (other16 d.BLOCK).bind fun l1 =>
            (tl.load (d.x.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32) +
                  (off + tl.arange 0 d.BLOCK))
                (some ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N))) (some l1)).bind
              fun l2 => (tl.cvt l2).bind fun a => tl.add _mean a).bind
        fun _mean => (tl.itof (Tensor.scalar (IntV.ofNat d.N : I32))).bind fun nf1 =>
          (tl.sum _mean).bind fun s => (tl.div (Tensor.scalar s) nf1).bind k) := by
  have hN := hd.n_ge
  have hB := hd.block_le
  have hB' : d.BLOCK ≤ 2 ^ 15 := by omega
  rw [forRange_one (by omega) hd.n_le_block (by omega)]
  simp only [Prog.bind_assoc]
  refine safe_loadX hd hpre ht hc hm fun a ha => ?_
  refine safe_add fun lm hlm => ?_
  refine safe_itof Format.binary32_std fun nf hnf => ?_
  refine safe_sum (by have := hd.n_le_block; omega) fun S hS => ?_
  obtain ⟨tr, hcov, hdep, heval⟩ := hS
  refine safe_div (fun i => hP.divTotal .binary32 _ _) fun q hq => ?_
  set r := rowP d hd init hpre t ht with hr
  have hxl := r.xl_le
  have hzeros : ∀ i, (tl.zeros [d.BLOCK] DType.f32).get i = FloatV.ofPat 0 := fun i => rfl
  -- the lanes after `_mean += a`
  have hlane : ∀ i : Idx [d.BLOCK], (lm.get i).isFinite = true ∧
      |(lm.get i).toReal - r.xl i.1| ≤ u32 * |r.xl i.1| + eta32 := by
    intro i
    obtain ⟨hf, hv⟩ := ha i
    have hb : |(a.get i).toReal| ≤ 2 ^ 20 := by
      rw [hv]; exact (hxl _).trans (by norm_num)
    have := add_zero_lane hf hb (by simpa [hzeros] using hlm i)
    rw [hv] at this
    exact this
  obtain ⟨l, hl⟩ : ∃ l : ℕ → F32, l = fun i => if h : i < d.BLOCK then lm.get1 ⟨i, h⟩ else default :=
    ⟨_, rfl⟩
  rw [← hl] at heval
  have hlget : ∀ i (h : i < d.BLOCK), l i = lm.get ((⟨i, h⟩ : Fin d.BLOCK), ()) := by
    intro i h; rw [hl]; simp [h]; rfl
  have hlfin : ∀ i, i < d.BLOCK → (l i).isFinite = true := fun i h => by
    rw [hlget i h]; exact (hlane _).1
  have hlb : ∀ i, i < d.BLOCK → |(l i).toReal| ≤ 2 ^ 20 := fun i h => by
    rw [hlget i h]
    obtain ⟨-, h2⟩ := hlane ((⟨i, h⟩ : Fin d.BLOCK), ())
    have h3 := hxl i
    have h4 := abs_sub_abs_le_abs_sub (lm.get ((⟨i, h⟩ : Fin d.BLOCK), ())).toReal (r.xl i)
    unfold u32 eta32 at h2
    have h5 : (2 : ℝ) ^ (-24 : ℤ) * |r.xl i| + 2 ^ (-150 : ℤ) ≤ 1 := by
      have : (2 : ℝ) ^ (-24 : ℤ) * |r.xl i| ≤ 2 ^ (-24 : ℤ) * 65504 := by gcongr
      have h6 : (2 : ℝ) ^ (-24 : ℤ) * 65504 + 2 ^ (-150 : ℤ) ≤ 1 := by norm_num
      linarith
    have : |(lm.get ((⟨i, h⟩ : Fin d.BLOCK), ())).toReal| ≤ 65504 + 1 := by linarith
    exact this.trans (by norm_num)
  have hbe : d.BLOCK = 2 ^ r.D := r.block_eq
  have hDle : r.D ≤ 15 := by
    have h1 := hd.block_le
    have h2 : 2 ^ r.D ≤ 2 ^ 15 := by rw [← hbe]; omega
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h2
  have hdep' : tr.depth ≤ r.D := by
    have h1 := hP.sumDepth d.BLOCK
    have h2 : Nat.clog 2 d.BLOCK = r.D := by rw [hbe, Nat.clog_pow _ _ (by norm_num)]
    unfold tl.sumBound at hdep
    omega
  obtain ⟨hSfin, hSerr⟩ := sum_real hDle hB' l hcov hdep' heval hlfin (fun i h => (hlb i h).trans (by norm_num))
  obtain ⟨hnfin, hnval⟩ := itof_exact (N := d.N) (by have := hd.n_le; omega) (hnf ())
  have hS40 : |S.toReal| ≤ 2 ^ 50 := by
    have hA : ∑ i ∈ Finset.range d.BLOCK, |(l i).toReal| ≤ 2 ^ 35 := by
      calc _ ≤ ∑ i ∈ Finset.range d.BLOCK, (2 : ℝ) ^ 20 :=
            Finset.sum_le_sum fun i hi => hlb i (Finset.mem_range.mp hi)
        _ = d.BLOCK * 2 ^ 20 := by simp
        _ ≤ 2 ^ 15 * 2 ^ 20 := by
            have : (d.BLOCK : ℝ) ≤ 2 ^ 15 := by exact_mod_cast hB'
            gcongr
        _ = 2 ^ 35 := by norm_num
    have hT : |∑ i ∈ Finset.range d.BLOCK, (l i).toReal| ≤ 2 ^ 35 :=
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
    have hE : ((1 + u32) ^ r.D - 1) * ∑ i ∈ Finset.range d.BLOCK, |(l i).toReal| ≤ 2 ^ 35 := by
      nlinarith
    have := abs_sub_abs_le_abs_sub S.toReal (∑ i ∈ Finset.range d.BLOCK, (l i).toReal)
    nlinarith [abs_nonneg S.toReal]
  have hN0 : ((nf.get ()).toReal) ≠ 0 := by
    rw [hnval]; exact_mod_cast (by omega : d.N ≠ 0)
  have hq' : |S.toReal / (nf.get ()).toReal| ≤ 2 ^ 126 := by
    rw [abs_div, hnval, abs_of_nonneg (show (0 : ℝ) ≤ d.N from Nat.cast_nonneg _)]
    have : (2 : ℝ) ≤ d.N := by exact_mod_cast hd.n_ge
    rw [div_le_iff₀ (by linarith)]
    nlinarith [abs_nonneg S.toReal]
  obtain ⟨hqfin, hqerr⟩ := div_real hP hSfin hnfin hN0 hq' (hq ())
  refine hk q hqfin ?_
  rw [hnval] at hqerr
  refine r.mean_stage (fun i => (l i).toReal) S.toReal q.item.toReal ?_ ?_ hqerr
  · intro j hj
    rw [hlget j hj]
    exact (hlane _).2
  · exact hSerr

end Claims.LayerNormFwdF16
