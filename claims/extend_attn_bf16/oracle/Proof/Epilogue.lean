import Proof.Spec
import Proof.Lanes
import Proof.RowFacts
import Proof.RowOut

/-!
# The epilogue

After the last trip each valid row's state is a `RowFl` of all its keys. The epilogue divides by
`deno` (or `1.0` when it is `0`), converts to `bfloat16` and stores the tile: `Q` holds
(`row_output`).
-/

namespace Claims.ExtendAttn

open TVA

private theorem epi_divmod_inj {n t t' r r' : Nat} (hr : r < n) (hr' : r' < n)
    (h : t * n + r = t' * n + r') : t = t' ∧ r = r' := by
  have h1 : (t * n + r) / n = t := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hr]; rfl
  have h2 : (t' * n + r') / n = t' := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hr']; rfl
  have ht : t = t' := by rw [← h1, ← h2, h]
  subst ht
  exact ⟨rfl, by omega⟩

/-- `o`'s elements are distinct. -/
theorem epi_oIdx_inj {d : Inst} (hd : d.ok) {t t' h h' e e' : Nat} (ht : t < d.T)
    (ht' : t' < d.T) (hh : h < d.H) (hh' : h' < d.H) (he : e < d.D) (he' : e' < d.D)
    (heq : oIdx d t h e = oIdx d t' h' e') : t = t' ∧ h = h' ∧ e = e' := by
  simp only [oIdx] at heq
  rcases hd.o_layout with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · have hs : h * d.soh + e < d.sot := by
      have : (h + 1) * d.soh ≤ d.H * d.soh := Nat.mul_le_mul_right _ hh
      rw [Nat.add_mul] at this
      omega
    have hs' : h' * d.soh + e' < d.sot := by
      have : (h' + 1) * d.soh ≤ d.H * d.soh := Nat.mul_le_mul_right _ hh'
      rw [Nat.add_mul] at this
      omega
    obtain ⟨rfl, hr⟩ := epi_divmod_inj (t := t) (t' := t') hs hs' (by linarith)
    obtain ⟨rfl, rfl⟩ := epi_divmod_inj (n := d.soh) (t := h) (t' := h') (he.trans_le h1)
      (he'.trans_le h1) hr
    exact ⟨rfl, rfl, rfl⟩
  · have hs : t * d.sot + e < d.soh := by
      have : (t + 1) * d.sot ≤ d.T * d.sot := Nat.mul_le_mul_right _ ht
      rw [Nat.add_mul] at this
      omega
    have hs' : t' * d.sot + e' < d.soh := by
      have : (t' + 1) * d.sot ≤ d.T * d.sot := Nat.mul_le_mul_right _ ht'
      rw [Nat.add_mul] at this
      omega
    obtain ⟨rfl, hr⟩ := epi_divmod_inj (t := h) (t' := h') hs hs' (by linarith)
    obtain ⟨rfl, rfl⟩ := epi_divmod_inj (n := d.sot) (t := t) (t' := t') (he.trans_le h1)
      (he'.trans_le h1) hr
    exact ⟨rfl, rfl, rfl⟩

set_option maxHeartbeats 1600000 in
/-- **One stored element is within the tolerance.** -/
private theorem epi_out {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size)
    {es K : Nat} {acc : Tensor [d.BM, d.BD] F32} {deno e_max : Tensor [d.BM] F32}
    (hE : LoopInvE d init i es K (acc, deno, e_max) init)
    (hcov : blockEnd d init i ≤ es + BN * K) (hKT : K ≤ (blockEnd d init i + BN - 1) / BN)
    (r : Fin d.BM) (e : Fin d.BD) (hv : validRow d init i r) (he : (e : Nat) < d.D)
    {o : F32} {c : BF16}
    (hdiv : P.div .binary32 (acc.get2 r e)
      (if FloatV.eq (deno.get1 r) zf32 = true then one32 else deno.get1 r) o)
    (hcvt : FloatV.NaNEq (FloatV.cvtWith .rn o) c) :
    c.isFinite = true ∧
      |c.toReal - attnRef d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e| ≤
        tolRel d.BD (tripsPre d init (bOf d i)) (tripsExt d init (bOf d i) (mOf d i * d.BM + r))
            (Xrow d init (bOf d i) (hOf d i) (mOf d i * d.BM + r)) *
          attnAbs d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e +
        tolAbs (vSum d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e) := by
  obtain ⟨hX0, hX, hKne, hcard, -, -, hT, hTe⟩ := row_facts hd hpre hi hv
  have hrow0 := hE.2 r hv
  rw [seen_end hd hpre hi hv r.isLt hcov] at hrow0
  have hBM : 0 < d.BM := r.pos
  have hTe_eq : tripsExt d init (bOf d i) (mOf d i * d.BM + r) =
      (blockEnd d init i + BN - 1) / BN := by
    unfold tripsExt blockEnd
    have : (mOf d i * d.BM + r) / d.BM = mOf d i := by
      rw [Nat.mul_comm, Nat.mul_add_div hBM, Nat.div_eq_of_lt r.isLt]; rfl
    rw [this]
  have hbd := hd.bd_le
  have hBD' : (d.BD : ℝ) ≤ 256 := by exact_mod_cast hbd
  have hd0 : 0 ≤ dlt d.BD (rowX d init i r) := by unfold dlt; have := hX0; positivity
  have hd1 : dlt d.BD (rowX d init i r) ≤ 1 / 2 := by
    unfold dlt
    have h1 : ((d.BD : ℝ) + 8) * (1 / 2 ^ 22) * rowX d init i r ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
      apply mul_le_mul (mul_le_mul_of_nonneg_right (by linarith) (by norm_num)) hX hX0 (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hKR : (K : ℝ) ≤ (tripsExt d init (bOf d i) (mOf d i * d.BM + r) : ℕ) := by
    exact_mod_cast hTe_eq ▸ hKT
  have hγ : g2 + tripsPre d init (bOf d i) * g1p + K * g1e ≤
      g2 + tripsPre d init (bOf d i) * g1p +
        tripsExt d init (bOf d i) (mOf d i * d.BM + r) * g1e := by
    have : (0 : ℝ) ≤ g1e := by unfold g1e; norm_num
    nlinarith
  have hrow := RowFl.mono (t' := tripsPre d init (bOf d i) + tripsExt d init (bOf d i) (mOf d i * d.BM + r))
    (γ' := g2 + tripsPre d init (bOf d i) * g1p +
        tripsExt d init (bOf d i) (mOf d i * d.BM + r) * g1e)
    (by norm_num at hX ⊢; linarith) (by exact hcard) (by omega) (by unfold tmax; omega) hγ hrow0
  have hγ0 : 0 ≤ g2 + tripsPre d init (bOf d i) * g1p +
        tripsExt d init (bOf d i) (mOf d i * d.BM + r) * g1e := by
    unfold g2 g1p g1e; positivity
  have key := row_output hP hbd hX0 hX hTe hT hKne hcard hγ0 le_rfl hrow he hdiv hcvt
  have e1 : attnRef d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e =
      wref (keys d init (bOf d i) (mOf d i * d.BM + r))
        (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
        (rowVc d init (bOf d i) (hOf d i) e) := rfl
  have e2 : attnAbs d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e =
      wabs (keys d init (bOf d i) (mOf d i * d.BM + r))
        (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
        (rowVc d init (bOf d i) (hOf d i) e) := rfl
  have e3 : vSum d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e =
      ∑ j ∈ keys d init (bOf d i) (mOf d i * d.BM + r), |rowVc d init (bOf d i) (hOf d i) e j| :=
    rfl
  have e4 : Xrow d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) = rowX d init i r := rfl
  rw [e1, e2, e3, e4]
  exact key

/-- **The epilogue is safe and leaves `Q`.** The new-token loop ended after `K` trips that started at
the new token `es`, having passed the block's end (`hcov`), and ran at most the row's trips
(`hKT`, the number of `64`-blocks of `blockEnd`). -/
theorem epilogue_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    {es K : Nat} {acc : Tensor [d.BM, d.BD] F32} {deno e_max : Tensor [d.BM] F32}
    (hE : LoopInvE d init i es K (acc, deno, e_max) init)
    (hcov : blockEnd d init i ≤ es + BN * K) (hKT : K ≤ (blockEnd d init i + BN - 1) / BN) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (epilogue d s (maskM d s) acc deno) := by
  obtain ⟨hb, hh, hmN⟩ := prog_range hi
  have hbm := bm_le hd
  have hnb := hd.nb_lt
  have hH := hd.H_lt
  have hsz := req_sizes hpre hb
  obtain ⟨hs1, hs2, hs3, hs4, hs5, hs6, hs7, hs8, -⟩ := hd.strides_lt
  unfold epilogue
  simp only [safe_toQ]
  refine safe_div (fun _ => hP.divTotal _ _ _) fun o ho => ?_
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun o16 hcv => ?_
  refine safe_last ?_
  refine safe_store_en d.o
    (fun ix : Idx [d.BM, d.BD] => decide (validRow d init i ix.1 ∧ ix.2.1.val < d.D))
    (fun ix => oIdx d (qsOf d init (bOf d i) + (mOf d i * d.BM + ix.1)) (hOf d i) ix.2.1)
    ?hm ?hp ?hin ?hinj hpre.2.2.2.1 ?hW ?hk
  case hm =>
    rintro ⟨r, e, ⟨⟩⟩
    have h1 := maskM_lane hd hi hs r
    change ((maskM d s).get1 r && IntV.lt ((tl.arange 0 d.BD).get1 e) (IntV.ofNat d.D)) = _
    have hD : d.D < 2 ^ 31 := by have := hd.D_le; have := hd.bd_le; omega
    rw [h1, IntV.lt_eq, IntV.I32.toInt_tl_arange (by have := hd.bd_le; omega),
      IntV.I32.toInt_ofNat (by simpa using hD)]
    simp
  case hp =>
    rintro ⟨r, e, ⟨⟩⟩ hen
    simp only [decide_eq_true_eq] at hen
    obtain ⟨hv, he⟩ := hen
    apply ptr0_add
    erw [Tensor.get_add, Tensor.get_adds, Tensor.get_muls, Tensor.get_sadd, toQv_get, toQv_get]
    simp only [Tensor.get_col, Tensor.get_row, toQv_scalar_item]
    have hTpos : 0 < d.T := by omega
    have hmB : mOf d i * d.BM < 2 ^ 16 * 128 := by
      have := Nat.mul_le_mul (le_of_lt (lt_of_lt_of_le hmN (le_of_lt hnb) : mOf d i < 2 ^ 16)) hbm
      omega
    have hrB : (r : Nat) < 128 := by omega
    have hP' : (castQ d.cm (s.cur_block_m * IntV.ofNat d.BM)).toInt = ((mOf d i * d.BM : Nat) : Int) := by
      rw [castQ_toInt]; exact IntV.mul_ofNat hs.blk (by norm_num; omega)
    have hX : ((tl.arange 0 d.BM : Tensor [d.BM] I32).get (r, ())).toInt = ((r : Nat) : Int) :=
      IntV.I32.toInt_tl_arange (n := d.BM) (by omega) r
    have hS0 : (IntV.ofNat d.sot : I32).toInt = ((d.sot : Nat) : Int) :=
      IntV.toInt_ofNat (by simpa using hs7)
    have hC : (s.cur_head * IntV.ofNat d.soh).toInt = ((hOf d i * d.soh : Nat) : Int) := by
      refine IntV.mul_ofNat hs.head ?_
      have := hd.heads_i32.2.2.2.1
      have h2 : hOf d i * d.soh ≤ (d.H - 1) * d.soh := Nat.mul_le_mul_right _ (by omega)
      simpa using lt_of_le_of_lt h2 this
    have hY : ((tl.arange 0 d.BD : Tensor [d.BD] I32).get (e, ())).toInt = ((e : Nat) : Int) :=
      IntV.I32.toInt_tl_arange (n := d.BD) (by have := hd.bd_le; omega) e
    have ht : qsOf d init (bOf d i) + mOf d i * d.BM + r < d.T := by unfold validRow at hv; omega
    have h1 : qsOf d init (bOf d i) + mOf d i * d.BM + r < 2 ^ (qoW d.cm - 1) :=
      lt_pow_qoW _ (lt_of_lt_of_le ht (le_trans hd.T_le (by norm_num)))
    have h2 := idx_fits hd (t := qsOf d init (bOf d i) + mOf d i * d.BM + r) (g := hOf d i)
      (e := e) (T' := d.T) (G := d.H) (s0 := d.sot) (s1 := d.soh) ht hh he hd.T_le hH.le hs7 hs8
      (fun hcm => (hd.cm_i32 hcm hTpos).2.2.2)
    have := qoff_val d.cm hs.qs hP' hX hS0 hC hY h1 h2
    rw [this]
    simp only [oIdx]
    congr 1
    ring
  case hin =>
    rintro ⟨r, e, ⟨⟩⟩ hen
    simp only [decide_eq_true_eq] at hen
    obtain ⟨hv, he⟩ := hen
    have hTpos : 0 < d.T := by unfold validRow at hv; omega
    have hl := hd.o_len hTpos
    have ht : qsOf d init (bOf d i) + (mOf d i * d.BM + r) ≤ d.T - 1 := by unfold validRow at hv; omega
    have h1 : (qsOf d init (bOf d i) + (mOf d i * d.BM + r)) * d.sot ≤ (d.T - 1) * d.sot :=
      Nat.mul_le_mul_right _ ht
    have h2 : hOf d i * d.soh ≤ (d.H - 1) * d.soh := Nat.mul_le_mul_right _ (by omega)
    simp only [oIdx]
    omega
  case hinj =>
    rintro ⟨r, e, ⟨⟩⟩ ⟨r', e', ⟨⟩⟩ hen hen' hg
    simp only [decide_eq_true_eq] at hen hen'
    obtain ⟨hv, he⟩ := hen
    obtain ⟨hv', he'⟩ := hen'
    have ht : qsOf d init (bOf d i) + (mOf d i * d.BM + r) < d.T := by unfold validRow at hv; omega
    have ht' : qsOf d init (bOf d i) + (mOf d i * d.BM + r') < d.T := by unfold validRow at hv'; omega
    obtain ⟨h1, -, h3⟩ := epi_oIdx_inj hd ht ht' hh hh he he' hg
    have hr : (r : Nat) = r' := by omega
    have hr2 : r = r' := Fin.ext hr
    have he2 : e = e' := Fin.ext h3
    subst hr2; subst he2; rfl
  case hW =>
    refine d.o.elems_mono ?_
    rintro x ⟨⟨r, e, ⟨⟩⟩, hen, rfl⟩
    simp only [decide_eq_true_eq] at hen
    exact ⟨r, e, r.isLt, hen.2, hen.1, rfl⟩
  case hk =>
    intro m' hrd _ _
    refine safe_ret ?_
    intro r e hr he hv
    have heB : e < d.BD := lt_of_lt_of_le he hd.D_le
    have hdiv : ((launch d).env P).P.div .binary32 (acc.get2 ⟨r, hr⟩ ⟨e, heB⟩)
        (if FloatV.eq (deno.get1 ⟨r, hr⟩) zf32 = true then one32 else deno.get1 ⟨r, hr⟩)
        (o.get (⟨r, hr⟩, ⟨e, heB⟩, ())) := ho (⟨r, hr⟩, ⟨e, heB⟩, ())
    obtain ⟨hfin, hbound⟩ := epi_out hP hd hpre hi hE hcov hKT ⟨r, hr⟩ ⟨e, heB⟩ hv he hdiv
      (hcv (⟨r, hr⟩, ⟨e, heB⟩, ()))
    exact ⟨o16.get (⟨r, hr⟩, ⟨e, heB⟩, ()),
      hrd (⟨r, hr⟩, ⟨e, heB⟩, ()) (by simp [hv, he]), hfin, hbound⟩
end Claims.ExtendAttn
