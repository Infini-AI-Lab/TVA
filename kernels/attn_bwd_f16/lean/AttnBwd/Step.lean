import AttnBwd.Basic

/-!
# `attn_bwd_f16`: one step of each loop, by value

* `dot_spec`: a `tl.dot` over `HEAD_DIM` products of a head's rows (lines 293, 307, 345, 353).
* `pair_lane`: one pair `(n, m)`: `lane` in terms of `Spec`'s quantities.
* `kv_step`: one trip of `_attn_bwd_dkdv`'s loop (lines 289–310), by value: the accumulators'
  invariant `KVInv` moves to the block's query rows.
* `q_step`: one trip of `_attn_bwd_dq`'s loop (lines 343–358): `QInv` moves to the block's key
  rows.
-/

namespace Kernels.AttnBwd

open TVA

/-- One element of a `tl.dot` (zero accumulator) of a row `f` and a column `g` of `HEAD_DIM`
finite `float16` values: within `rH·Σ lm f·lm g + aH` of `Σ f·g`. -/
theorem dot_spec {P : Platform} (hdot : Rules.Num.DotF16 P) {d : Inst} (hd : d.ok) {M N : Nat}
    {a : Tensor [M, d.HEAD_DIM] F16} {b : Tensor [d.HEAD_DIM, N] F16} {r : Tensor [M, N] F32}
    (hr : P.dot dotCfg a b (tl.zeros [M, N] .f32) r) (i : Fin M) (j : Fin N) (f g : Nat → F16)
    (hf : ∀ c : Fin d.HEAD_DIM, a.get2 i c = f c) (hg : ∀ c : Fin d.HEAD_DIM, b.get2 c j = g c)
    (hfin : ∀ c, c < d.HEAD_DIM → (f c).isFinite = true ∧ (g c).isFinite = true) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toReal - ∑ c ∈ Finset.range d.HEAD_DIM, (f c).toReal * (g c).toReal| ≤
        rH d * ∑ c ∈ Finset.range d.HEAD_DIM, lm (f c) * lm (g c) + aH d := by
  obtain ⟨h1, h2⟩ := dot0 hdot (hd_le hd) hr i j (fun c => by rw [hf c]; exact (hfin c c.2).1)
    (fun c => by rw [hg c]; exact (hfin c c.2).2)
  refine ⟨h1, ?_⟩
  have e1 : ∑ c : Fin d.HEAD_DIM, (a.get2 i c).toReal * (b.get2 c j).toReal =
      ∑ c ∈ Finset.range d.HEAD_DIM, (f c).toReal * (g c).toReal := by
    rw [← Fin.sum_univ_eq_sum_range (fun c => (f c).toReal * (g c).toReal)]
    exact Finset.sum_congr rfl fun c _ => by rw [hf c, hg c]
  have e2 : ∑ c : Fin d.HEAD_DIM, lm (a.get2 i c) * lm (b.get2 c j) =
      ∑ c ∈ Finset.range d.HEAD_DIM, lm (f c) * lm (g c) := by
    rw [← Fin.sum_univ_eq_sum_range (fun c => lm (f c) * lm (g c))]
    exact Finset.sum_congr rfl fun c _ => by rw [hf c, hg c]
  rw [e1, e2] at h2
  exact h2

theorem sum_mul_comm (K : Nat) (f g : Nat → F16) :
    ∑ c ∈ Finset.range K, (f c).toReal * (g c).toReal = ∑ c ∈ Finset.range K, (g c).toReal * (f c).toReal :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

theorem sum_lm_mul_comm (K : Nat) (f g : Nat → F16) :
    ∑ c ∈ Finset.range K, lm (f c) * lm (g c) = ∑ c ∈ Finset.range K, lm (g c) * lm (f c) :=
  Finset.sum_congr rfl fun _ _ => mul_comm _ _

/-- **One pair** `(n, m)` of a covered head: from the computed `Σ K·Q` and `Σ V·dO` to the
`float16` `P` and `dS`, within `Spec`'s errors and bounds. -/
theorem pair_lane {P : Platform} (hexp : Rules.Num.Exp2F32 P) {fusion : Bool} {d : Inst} (hd : d.ok)
    {hv : HeadV} (hO : hv.Ok d) {n mm : Nat} (hn : n < d.N_CTX) (hmm : mm < d.N_CTX)
    {s dp x p0 y ds : F32} {p16 ds16 : F16}
    (hsf : s.isFinite = true) (hs : |s.toReal - sQK d hv n mm| ≤ rH d * aQK d hv n mm + aH d)
    (hdpf : dp.isFinite = true) (hdp : |dp.toReal - sVD d hv n mm| ≤ rH d * aVD d hv n mm + aH d)
    (hx : AddRel fusion s (FloatV.neg (hv.lse mm)) x) (hp0 : P.math .exp2 .binary32 x p0)
    (hy : AddRel fusion dp (FloatV.neg (hv.dl mm)) y)
    (hp16 : FloatV.NaNEq (FloatV.cvt (if decide (Allowed d n mm) then p0 else FloatV.ofPat 0)) p16)
    (hds : FloatV.NaNEq (FloatV.mul (if decide (Allowed d n mm) then p0 else FloatV.ofPat 0) y) ds)
    (hds16 : FloatV.NaNEq (FloatV.cvt ds) ds16) :
    p16.isFinite = true ∧ ds16.isFinite = true ∧
      |p16.toReal - pE d hv n mm| ≤ p16Err d hv n mm ∧ |p16.toReal| ≤ p16B d hv n mm ∧
      |ds16.toReal - dsE d hv n mm| ≤ ds16Err d hv n mm ∧ |ds16.toReal| ≤ ds16B d hv n mm := by
  obtain ⟨hl1, hl2⟩ := hO.lse mm hmm
  obtain ⟨hd1, hd2⟩ := hO.dl mm hmm
  obtain ⟨hf1, hf2, h0, h1⟩ := lane hexp hd (decide (Allowed d n mm)) (aQK_le hd hO hn hmm)
    (sQK_abs n mm) (aVD_le hd hO hn hmm) (sVD_abs n mm) hsf hs hl1 hl2 hdpf hdp hd1 hd2
    (X := xE d hv n mm) (EX := eX d hv n mm) (Y := yE d hv n mm) (EY := eY d hv n mm) rfl rfl rfl rfl
    hx hp0 hy hp16 hds hds16 (fun h => hO.fits n mm hn hmm (of_decide_eq_true h))
  refine ⟨hf1, hf2, ?_⟩
  by_cases hal : Allowed d n mm
  · obtain ⟨e1, e2, e3, e4⟩ := h1 (decide_eq_true hal)
    simp only [pE, p16Err, p16B, dsE, ds16Err, ds16B, if_pos hal]
    exact ⟨e1, e2, e3, e4⟩
  · obtain ⟨e1, e2⟩ := h0 (decide_eq_false hal)
    simp only [pE, p16Err, p16B, dsE, ds16Err, ds16B, if_neg hal, e1, e2]
    simp

theorem cover_sub1 {C : Finset ℕ} {lo B N : ℕ} (hC : C ⊆ Finset.range N) (h : lo + B ≤ N) :
    C ∪ Finset.Ico lo (lo + B) ⊆ Finset.range N := by
  intro x hx
  rcases Finset.mem_union.mp hx with h' | h'
  · exact hC h'
  · rw [Finset.mem_Ico] at h'; rw [Finset.mem_range]; omega

/-- **One trip of `_attn_bwd_dkdv`'s loop**, by value (lines 289–310), for the block of query
rows `mt, …, mt + BM − 1`: from the loaded tiles (`hval`) and the results of each line, the
invariant `KVInv` moves from the rows `C` to `C ∪ [mt, mt + BM)`. -/
theorem kv_step {P : Platform} (hdot : Rules.Num.DotF16 P) (hexp : Rules.Num.Exp2F32 P) {fusion : Bool}
    {d : Inst} (hd : d.ok) {init : Mem} {b p : Nat} {BM : Nat} (hBM : BM = 16 ∨ BM = 32)
    {mt : Nat} (hmt : mt + BM ≤ d.N_CTX) (hp : 128 * p + 128 ≤ d.N_CTX)
    {C : Finset ℕ} (hC : ∀ x ∈ C, x < mt ∨ mt + BM ≤ x) (hCN : C ⊆ Finset.range d.N_CTX)
    {T0 : Nat} (hT : T0 < 1024)
    {k v : Tensor [128, d.HEAD_DIM] F16} {qT : Tensor [d.HEAD_DIM, BM] F16}
    {dO : Tensor [BM, d.HEAD_DIM] F16} {mv Di : Tensor [BM] F32}
    (hval : ∀ hv : HeadV, hv.Holds d init b →
      (∀ (r : Fin 128) (c : Fin d.HEAD_DIM),
        k.get2 r c = hv.k (128 * p + r) c ∧ v.get2 r c = hv.v (128 * p + r) c) ∧
      (∀ (c : Fin d.HEAD_DIM) (i : Fin BM), qT.get2 c i = hv.q (mt + i) c) ∧
      (∀ (i : Fin BM) (c : Fin d.HEAD_DIM), dO.get2 i c = hv.dO (mt + i) c) ∧
      (∀ i : Fin BM, mv.get1 i = hv.lse (mt + i) ∧ Di.get1 i = hv.dl (mt + i)))
    {qkT xs p0 pT dpT0 dpT ys dsT : Tensor [128, BM] F32} {ppT dsT16 : Tensor [128, BM] F16}
    (hqkT : P.dot dotCfg k qT (tl.zeros [128, BM] .f32) qkT)
    (hxs : ∀ (r : Fin 128) (i : Fin BM), AddRel fusion (qkT.get2 r i) (FloatV.neg (mv.get1 i)) (xs.get2 r i))
    (hp0 : ∀ (r : Fin 128) (i : Fin BM), P.math .exp2 .binary32 (xs.get2 r i) (p0.get2 r i))
    (hpT : ∀ (r : Fin 128) (i : Fin BM),
      pT.get2 r i = if decide (Allowed d (128 * p + r) (mt + i)) then p0.get2 r i else FloatV.ofPat 0)
    (hppT : ∀ (r : Fin 128) (i : Fin BM), FloatV.NaNEq (FloatV.cvt (pT.get2 r i)) (ppT.get2 r i))
    (hdpT0 : P.dot dotCfg v dO.trans (tl.zeros [128, BM] .f32) dpT0)
    (hdpT : ∀ (r : Fin 128) (i : Fin BM), dpT.get2 r i = dpT0.get2 r i)
    (hys : ∀ (r : Fin 128) (i : Fin BM), AddRel fusion (dpT.get2 r i) (FloatV.neg (Di.get1 i)) (ys.get2 r i))
    (hdsT : ∀ (r : Fin 128) (i : Fin BM), FloatV.NaNEq (FloatV.mul (pT.get2 r i) (ys.get2 r i)) (dsT.get2 r i))
    (hdsT16 : ∀ (r : Fin 128) (i : Fin BM), FloatV.NaNEq (FloatV.cvt (dsT.get2 r i)) (dsT16.get2 r i))
    {dk dv dk' dv' : Tensor [128, d.HEAD_DIM] F32}
    (hdv' : P.dot dotCfg ppT dO dv dv' ∨ ∃ e, P.dot dotCfg ppT dO (Tensor.full (dotCfg.acc.ofBits 0)) e ∧
        ∀ ix, AccAdd fusion dotCfg.acc (e.get ix) (dv.get ix) (dv'.get ix))
    (hdk' : P.dot dotCfg dsT16 qT.trans dk dk' ∨
      ∃ e, P.dot dotCfg dsT16 qT.trans (Tensor.full (dotCfg.acc.ofBits 0)) e ∧
        ∀ ix, AccAdd fusion dotCfg.acc (e.get ix) (dk.get ix) (dk'.get ix))
    (hinv : KVInv d init b p C T0 dk dv) :
    KVInv d init b p (C ∪ Finset.Ico mt (mt + BM)) (T0 + 1) dk' dv' := by
  intro r c hv hH hO
  obtain ⟨hkv, hq, hdo, hmD⟩ := hval hv hH
  obtain ⟨hdv, hdk⟩ := hinv r c hv hH hO
  have hn : 128 * p + r < d.N_CTX := by have := r.2; omega
  have hlane : ∀ i : Fin BM, (ppT.get2 r i).isFinite = true ∧ (dsT16.get2 r i).isFinite = true ∧
      |(ppT.get2 r i).toReal - pE d hv (128 * p + r) (mt + i)| ≤ p16Err d hv (128 * p + r) (mt + i) ∧
      |(ppT.get2 r i).toReal| ≤ p16B d hv (128 * p + r) (mt + i) ∧
      |(dsT16.get2 r i).toReal - dsE d hv (128 * p + r) (mt + i)| ≤ ds16Err d hv (128 * p + r) (mt + i) ∧
      |(dsT16.get2 r i).toReal| ≤ ds16B d hv (128 * p + r) (mt + i) := by
    intro i
    have hmm : mt + i < d.N_CTX := by have := i.2; omega
    obtain ⟨hsf, hs⟩ := dot_spec hdot hd hqkT r i (hv.k (128 * p + r)) (fun c => hv.q (mt + i) c)
      (fun c => (hkv r c).1) (fun c => hq c i) (fun c hc => ⟨hO.k _ c hn hc, hO.q _ c hmm hc⟩)
    obtain ⟨hdpf, hdp⟩ := dot_spec hdot hd hdpT0 r i (hv.v (128 * p + r)) (fun c => hv.dO (mt + i) c)
      (fun c => (hkv r c).2) (fun c => hdo i c) (fun c hc => ⟨hO.v _ c hn hc, hO.dO _ c hmm hc⟩)
    have hx := hxs r i
    rw [(hmD i).1] at hx
    have hy := hys r i
    rw [(hmD i).2, hdpT r i] at hy
    have hp16 := hppT r i
    rw [hpT r i] at hp16
    have hds := hdsT r i
    rw [hpT r i] at hds
    exact pair_lane hexp hd hO hn hmm hsf hs hdpf hdp hx (hp0 r i) hy hp16 hds (hdsT16 r i)
  have hsub := cover_sub1 hCN hmt
  have hdOf : ∀ i : Fin BM, (dO.get2 i c).isFinite = true := fun i => by
    rw [hdo i c]; exact hO.dO _ c (by have := i.2; omega) c.2
  have hqf : ∀ i : Fin BM, (qT.trans.get2 i c).isFinite = true := fun i => by
    show (qT.get2 c i).isFinite = true
    rw [hq c i]; exact hO.q _ c (by have := i.2; omega) c.2
  constructor
  · have hcap := sum_bd_le_lm hd hsub (fun m => p16L d hv (128 * p + r) m) (fun m => hv.dO m c)
      (fun m hm => p16L_bd hO hn hm) (fun m hm => hO.dO m c hm c.2)
    rw [sum_cover hC] at hcap
    have h := accI_dotAdd hdot hBM hdv' r c (fun i => (hlane i).1) hdOf hT hdv
      (fun i => pE d hv (128 * p + r) (mt + i) * (hv.dO (mt + i) c).toReal)
      (fun i => p16Err d hv (128 * p + r) (mt + i) * |(hv.dO (mt + i) c).toReal|)
      (fun i => p16L d hv (128 * p + r) (mt + i) * lm (hv.dO (mt + i) c))
      (fun i => by
        rw [hdo i c, ← sub_mul, abs_mul]
        exact mul_le_mul_of_nonneg_right (hlane i).2.2.1 (abs_nonneg _))
      (fun i => by
        rw [hdo i c]
        exact mul_le_mul_of_nonneg_right (lm_le_lift16 (hlane i).2.2.2.1) (lm_nn _))
      (by linarith)
    simp only [sum_cover hC]
    exact h
  · have hcap := sum_bd_le_lm hd hsub (fun m => ds16L d hv (128 * p + r) m) (fun m => hv.q m c)
      (fun m hm => ds16L_bd hd hO hn hm) (fun m hm => hO.q m c hm c.2)
    rw [sum_cover hC] at hcap
    have h := accI_dotAdd hdot hBM hdk' r c (fun i => (hlane i).2.1) hqf hT hdk
      (fun i => dsE d hv (128 * p + r) (mt + i) * (hv.q (mt + i) c).toReal)
      (fun i => ds16Err d hv (128 * p + r) (mt + i) * |(hv.q (mt + i) c).toReal|)
      (fun i => ds16L d hv (128 * p + r) (mt + i) * lm (hv.q (mt + i) c))
      (fun i => by
        show |(dsT16.get2 r i).toReal * (qT.get2 c i).toReal - _| ≤ _
        rw [hq c i, ← sub_mul, abs_mul]
        exact mul_le_mul_of_nonneg_right (hlane i).2.2.2.2.1 (abs_nonneg _))
      (fun i => by
        show lm (dsT16.get2 r i) * lm (qT.get2 c i) ≤ _
        rw [hq c i]
        exact mul_le_mul_of_nonneg_right (lm_le_lift16 (hlane i).2.2.2.2.2) (lm_nn _))
      (by linarith)
    simp only [sum_cover hC]
    exact h

/-- **One trip of `_attn_bwd_dq`'s loop**, by value (lines 343–358), for the block of key rows
`nt, …, nt + BN − 1`: the invariant `QInv` moves from the rows `C` to `C ∪ [nt, nt + BN)`. -/
theorem q_step {P : Platform} (hdot : Rules.Num.DotF16 P) (hexp : Rules.Num.Exp2F32 P) {fusion : Bool}
    {d : Inst} (hd : d.ok) {init : Mem} {b p : Nat} {BN : Nat} (hBN : BN = 16 ∨ BN = 32)
    {nt : Nat} (hnt : nt + BN ≤ d.N_CTX) (hp : 128 * p + 128 ≤ d.N_CTX)
    {C : Finset ℕ} (hC : ∀ x ∈ C, x < nt ∨ nt + BN ≤ x) (hCN : C ⊆ Finset.range d.N_CTX)
    {T0 : Nat} (hT : T0 < 1024)
    {q dO : Tensor [128, d.HEAD_DIM] F16} {kT vT : Tensor [d.HEAD_DIM, BN] F16}
    {mv : Tensor [128, 1] F32} {Di : Tensor [128] F32}
    (hval : ∀ hv : HeadV, hv.Holds d init b →
      (∀ (r : Fin 128) (c : Fin d.HEAD_DIM),
        q.get2 r c = hv.q (128 * p + r) c ∧ dO.get2 r c = hv.dO (128 * p + r) c) ∧
      (∀ (c : Fin d.HEAD_DIM) (j : Fin BN), kT.get2 c j = hv.k (nt + j) c ∧ vT.get2 c j = hv.v (nt + j) c) ∧
      (∀ r : Fin 128, mv.get2 r 0 = hv.lse (128 * p + r) ∧ Di.get1 r = hv.dl (128 * p + r)))
    {qk xs p0 pT dp0 dp ys ds : Tensor [128, BN] F32} {ds16 : Tensor [128, BN] F16}
    (hqk : P.dot dotCfg q kT (tl.zeros [128, BN] .f32) qk)
    (hxs : ∀ (r : Fin 128) (j : Fin BN), AddRel fusion (qk.get2 r j) (FloatV.neg (mv.get2 r 0)) (xs.get2 r j))
    (hp0 : ∀ (r : Fin 128) (j : Fin BN), P.math .exp2 .binary32 (xs.get2 r j) (p0.get2 r j))
    (hpT : ∀ (r : Fin 128) (j : Fin BN),
      pT.get2 r j = if decide (Allowed d (nt + j) (128 * p + r)) then p0.get2 r j else FloatV.ofPat 0)
    (hdp0 : P.dot dotCfg dO vT (tl.zeros [128, BN] .f32) dp0)
    (hdp : ∀ (r : Fin 128) (j : Fin BN), dp.get2 r j = dp0.get2 r j)
    (hys : ∀ (r : Fin 128) (j : Fin BN), AddRel fusion (dp.get2 r j) (FloatV.neg (Di.get1 r)) (ys.get2 r j))
    (hds : ∀ (r : Fin 128) (j : Fin BN), FloatV.NaNEq (FloatV.mul (pT.get2 r j) (ys.get2 r j)) (ds.get2 r j))
    (hds16 : ∀ (r : Fin 128) (j : Fin BN), FloatV.NaNEq (FloatV.cvt (ds.get2 r j)) (ds16.get2 r j))
    {dq dq' : Tensor [128, d.HEAD_DIM] F32}
    (hdq' : P.dot dotCfg ds16 kT.trans dq dq' ∨
      ∃ e, P.dot dotCfg ds16 kT.trans (Tensor.full (dotCfg.acc.ofBits 0)) e ∧
        ∀ ix, AccAdd fusion dotCfg.acc (e.get ix) (dq.get ix) (dq'.get ix))
    (hinv : QInv d init b p C T0 dq) :
    QInv d init b p (C ∪ Finset.Ico nt (nt + BN)) (T0 + 1) dq' := by
  intro r c hv hH hO
  obtain ⟨hqd, hkv, hmD⟩ := hval hv hH
  have hdq := hinv r c hv hH hO
  have hmm : 128 * p + r < d.N_CTX := by have := r.2; omega
  have hlane : ∀ j : Fin BN, (ds16.get2 r j).isFinite = true ∧
      |(ds16.get2 r j).toReal - dsE d hv (nt + j) (128 * p + r)| ≤ ds16Err d hv (nt + j) (128 * p + r) ∧
      |(ds16.get2 r j).toReal| ≤ ds16B d hv (nt + j) (128 * p + r) := by
    intro j
    have hn : nt + j < d.N_CTX := by have := j.2; omega
    obtain ⟨hsf, hs⟩ := dot_spec hdot hd hqk r j (hv.q (128 * p + r)) (fun c => hv.k (nt + j) c)
      (fun c => (hqd r c).1) (fun c => (hkv c j).1) (fun c hc => ⟨hO.q _ c hmm hc, hO.k _ c hn hc⟩)
    obtain ⟨hdpf, hdp'⟩ := dot_spec hdot hd hdp0 r j (hv.dO (128 * p + r)) (fun c => hv.v (nt + j) c)
      (fun c => (hqd r c).2) (fun c => (hkv c j).2) (fun c hc => ⟨hO.dO _ c hmm hc, hO.v _ c hn hc⟩)
    rw [sum_mul_comm, sum_lm_mul_comm] at hs hdp'
    have hx := hxs r j
    rw [(hmD r).1] at hx
    have hy := hys r j
    rw [(hmD r).2, hdp r j] at hy
    have hdsj := hds r j
    rw [hpT r j] at hdsj
    obtain ⟨-, h2, -, -, h5, h6⟩ := pair_lane (p16 := FloatV.cvt (if decide (Allowed d (nt + j) (128 * p + r))
      then p0.get2 r j else FloatV.ofPat 0)) hexp hd hO hn hmm hsf hs hdpf hdp' hx (hp0 r j) hy
      (FloatV.NaNEq.refl _) hdsj (hds16 r j)
    exact ⟨h2, h5, h6⟩
  have hsub := cover_sub1 hCN hnt
  have hkf : ∀ j : Fin BN, (kT.trans.get2 j c).isFinite = true := fun j => by
    show (kT.get2 c j).isFinite = true
    rw [(hkv c j).1]; exact hO.k _ c (by have := j.2; omega) c.2
  have hcap := sum_bd_le_lm hd hsub (fun n => ds16L d hv n (128 * p + r)) (fun n => hv.k n c)
    (fun n hn => ds16L_bd hd hO hn hmm) (fun n hn => hO.k n c hn c.2)
  rw [sum_cover hC] at hcap
  have h := accI_dotAdd hdot hBN hdq' r c (fun j => (hlane j).1) hkf hT hdq
    (fun j => dsE d hv (nt + j) (128 * p + r) * (hv.k (nt + j) c).toReal)
    (fun j => ds16Err d hv (nt + j) (128 * p + r) * |(hv.k (nt + j) c).toReal|)
    (fun j => ds16L d hv (nt + j) (128 * p + r) * lm (hv.k (nt + j) c))
    (fun j => by
      show |(ds16.get2 r j).toReal * (kT.get2 c j).toReal - _| ≤ _
      rw [(hkv c j).1, ← sub_mul, abs_mul]
      exact mul_le_mul_of_nonneg_right (hlane j).2.1 (abs_nonneg _))
    (fun j => by
      show lm (ds16.get2 r j) * lm (kT.get2 c j) ≤ _
      rw [(hkv c j).1]
      exact mul_le_mul_of_nonneg_right (lm_le_lift16 (hlane j).2.2) (lm_nn _))
    (by linarith)
  simp only [sum_cover hC]
  exact h

end Kernels.AttnBwd
