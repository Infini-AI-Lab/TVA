import Proof.Geom
import Proof.Stats
import Proof.LaneAll

/-!
# The `Safe` walk of one program

`safe_program` walks the body of program `t` one `tl.*` operation at a time, from the conversion
of the program id to the store of `output`'s row, collecting each operation's relation. At the
end, `Qp` (each element of row `t` of `output` is finite, within `yTol` of `yRef`, and a zero below
the band) follows from `stats_chain` (the row's cutoff) and `lane_ok` (each lane).
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 4000000

/-- What program `t` leaves in memory `m`: row `t` of `output`, element by element. -/
def Qp (d : Inst) (init : Mem) (t : Nat) (m : Mem) : Prop :=
  ∀ j, j < d.d → ∃ c : BF16, m.read .bf16 (d.out.ptr (oidx d t j : Int)).addr = some c ∧
    c.isFinite = true ∧ |c.toReal - yRef d init t j| ≤ yTol d init t j ∧
    (gv d init t j ≤ cutRef d init t - cutTol d init t → c.toReal = 0)

/-- `bfloat16 → float32` is a conversion Triton compiles. -/
theorem okA : Format.bfloat16.Std ∧ Format.binary32.Std ∧ tl.RoundOk .bfloat16 .binary32 .rn :=
  ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩

/-- `float32 → bfloat16` (round to nearest even) is a conversion Triton compiles. -/
theorem okB : Format.binary32.Std ∧ Format.bfloat16.Std ∧ tl.RoundOk .binary32 .bfloat16 .rn :=
  ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩

/-- The `bfloat16` `+0` (the loads' `other`) is finite and zero. -/
theorem bz : (FloatV.litBF16 (FloatV.ofPat 0) : BF16).isFinite = true ∧
    (FloatV.litBF16 (FloatV.ofPat 0) : BF16).toRat = 0 := by
  decide +kernel

/-- A loaded lane, widened to `float32`: the element when the lane is enabled, `0` otherwise. -/
theorem lane_load {init : Mem} {b : Buf .bf16} {k : Nat} {e : Bool} {v : BF16} {r : F32}
    (hread : e = true → init.read .bf16 (b.ptr (k : Int)).addr = some v)
    (hzero : e = false → v = FloatV.litBF16 (FloatV.ofPat 0))
    (hok : e = true → InOk (rval init b k))
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn v) r) :
    r.isFinite = true ∧ r.toReal = if e = true then (rval init b k).toReal else 0 := by
  cases e with
  | true =>
    have hv : rval init b k = v := by unfold rval; rw [hread rfl]
    have hf := (hok rfl).1
    rw [hv] at hf
    obtain ⟨h1, h2⟩ := cvt32_lane hf hc
    exact ⟨h1, by rw [h2, hv]; rfl⟩
  | false =>
    rw [hzero rfl] at hc
    obtain ⟨h1, h2⟩ := cvt32_lane bz.1 hc
    refine ⟨h1, ?_⟩
    rw [h2, FloatV.toReal_eq, bz.2]
    simp

/-- An element of `x` covered by `Pre` is at most `2^32` as a real. -/
theorem inOk_le {a : BF16} (h : InOk a) : |a.toReal| ≤ 2 ^ 32 := by
  rw [FloatV.toReal_eq]
  have := h.2
  exact_mod_cast this

/-- **The walk of program `t`.** -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hP : Device P) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d i) (W d i) (fun _ m => Qp d init i.val m) init
      ((model d).body ((launch d).ctx i)) := by
  have ht : i.val < d.T := fin_lt i
  obtain ⟨e, he⟩ := hd.b_pow2
  have hBpos : 0 < d.B := by rw [he]; exact Nat.two_pow_pos e
  have hdB := hd.d_le_b
  have he15 : e ≤ 15 := by
    have h1 : 2 ^ e < 2 ^ 16 := by rw [← he]; have := hd.b_lt; have := hd.d_le; omega
    exact Nat.lt_succ_iff.mp ((Nat.pow_lt_pow_iff_right (by norm_num)).mp h1)
  have hsb : tl.sumBound P d.B ≤ e := by
    unfold tl.sumBound
    have h1 := hP.sumDepth d.B
    have h2 : Nat.clog 2 d.B = e := by rw [he]; exact Nat.clog_pow 2 e (by norm_num)
    omega
  have hgidx : ∀ j, j < d.d → gidx d i.val j < d.T * (2 * d.d) := by
    intro j hj; have := rowX_end ht; unfold gidx; omega
  have huidx : ∀ j, j < d.d → uidx d i.val j < d.T * (2 * d.d) := by
    intro j hj; have := rowX_end ht; unfold uidx; omega
  have hgv : ∀ j, j < d.d → |gv d init i.val j| ≤ 2 ^ 32 :=
    fun j hj => inOk_le (hpre.2.2.1 _ (hgidx j hj))
  have hmR : |mR d| ≤ 64 := by
    unfold mR; rw [FloatV.toReal_eq]; have := hd.m_le; exact_mod_cast this
  let r : RowP :=
    { d := d.d, L := e, g := gv d init i.val, m := mR d, d_pos := hd.d_pos,
      d_le := he ▸ hdB, b_lt := he ▸ hd.b_lt, L_le := he15, g_le := hgv, m_le := hmR }
  have hok : r.Ok := hpre.2.2.2 _ ht
  simp only [model, _gelu_and_mul_sparse_kernel, tl.sigmoid, Prog.bind_eq, Prog.bind_assoc]
  refine safe_convert (hex := fun _ => ⟨_, rfl⟩) ?_
  intro rT hr
  have hrow := row_toInt hd i hr
  refine safe_load_buf d.x (addrs_gate hd ht hrow) (gG_lt hd ht) hpre.1
    (Buf.elems_mono _ (imgG_sub d _)) ?_
  intro vg hvg hvg0
  refine safe_cvt okA ?_; intro gate hgate
  refine safe_load_buf d.x (addrs_up hd ht hrow) (gU_lt hd ht) hpre.1
    (Buf.elems_mono _ (imgU_sub d _)) ?_
  intro vu hvu hvu0
  refine safe_cvt okA ?_; intro up hup
  refine safe_sum hBpos ?_; intro S ⟨t1, hc1, hd1, he1⟩
  refine safe_div (hex := fun _ => hP.divTotal _ _ _) ?_; intro mT hmT
  refine safe_subc ?_; intro scT hsc
  refine safe_mul ?_; intro qT hqT
  refine safe_sumc hBpos ?_; intro Qs ⟨t2, hc2, hd2, he2⟩
  refine safe_div (hex := fun _ => hP.divTotal _ _ _) ?_; intro vT hvT
  refine safe_cvt okB ?_; intro mean16 hmean16
  refine safe_cvt okA ?_; intro mean hmean
  refine safe_math (hex := fun _ => hP.mathTotal _ _ _) ?_; intro sqT hsqT
  refine safe_cvt okB ?_; intro std16 hstd16
  refine safe_cvt okA ?_; intro std hstd
  refine safe_mul ?_; intro prT hprT
  refine safe_cvt okB ?_; intro sc16 hsc16
  refine safe_cvt okA ?_; intro scs hscs
  refine safe_add ?_; intro csT hcsT
  refine safe_cvt okB ?_; intro ck16 hck16
  refine safe_cvt okA ?_; intro ckT hckT
  refine safe_sub ?_; intro s12 hs12
  refine safe_cvt okB ?_; intro sg16 hsg16
  refine safe_cvt okA ?_; intro sg0 hsg0
  refine safe_mul ?_; intro p1 hp1
  refine safe_mul ?_; intro p2 hp2
  refine safe_mul_add ?_; intro z hz
  refine safe_mul ?_; intro inn hinn
  refine safe_mul ?_; intro x2 hx2
  refine safe_neg ?_; intro nx hnx
  refine safe_math (hex := fun _ => hP.mathTotal _ _ _) ?_; intro ex hex
  refine safe_add ?_; intro den hden
  refine safe_div (hex := fun _ => hP.divTotal _ _ _) ?_; intro sg hsg
  refine safe_mul ?_; intro tw htw
  refine safe_subc ?_; intro th hth
  refine safe_mul ?_; intro hh hhh
  refine safe_add ?_; intro ot hot
  refine safe_mul ?_; intro act hact
  refine safe_cvt okB ?_; intro actb hactb
  refine safe_cvt okA ?_; intro av hav
  refine safe_mul ?_; intro pr hpr
  refine safe_cvt okB ?_; intro y16 hy16
  refine safe_last ?_
  refine safe_store_buf d.out (addrs_out hd ht hrow) (gO_lt hd ht) (gO_inj _) hpre.2.1
    (by rw [imgO_eq d _ hd]; exact le_refl _) ?_
  intro m' hread _ _
  refine safe_ret ?_
  -- the gate lanes
  have hgl : ∀ (q : Idx [d.B]), (gate.get q).isFinite = true ∧
      (gate.get q).toReal = if en d q = true then gv d init i.val q.1.val else 0 := by
    intro q
    exact lane_load (b := d.x) (k := gidx d i.val q.1.val) (hvg q)
      (fun h => hvg0 q h _ rfl) (fun h => hpre.2.2.1 _ (hgidx _ (lt_of_en h))) (hgate q)
  have hul : ∀ (q : Idx [d.B]), (up.get q).isFinite = true ∧
      (up.get q).toReal = if en d q = true then uv d init i.val q.1.val else 0 := by
    intro q
    exact lane_load (b := d.x) (k := uidx d i.val q.1.val) (hvu q)
      (fun h => hvu0 q h _ rfl) (fun h => hpre.2.2.1 _ (huidx _ (lt_of_en h))) (hup q)
  -- the row's statistics
  have hm32 := hostF32_err hd.m_fin (by
    rw [FloatV.toReal_eq]; have := hd.m_le
    have h64 : |(d.m.toRat : ℝ)| ≤ 64 := by exact_mod_cast this
    linarith)
  obtain ⟨hdvf, hdv⟩ := dconst hd.d_pos hd.d_le
  have hst := stats_chain hP r hok (fusion := ((launch d).env P).fusion)
    (fun j => if h : j < d.B then gate.get1 ⟨j, h⟩ else default)
    (by
      intro j hj
      have hjB : j < d.B := by rw [he]; exact hj
      simp only [hjB, ↓reduceDIte]
      obtain ⟨h1, h2⟩ := hgl (lane d j hjB)
      refine ⟨h1, ?_⟩
      show (gate.get (lane d j hjB)).toReal = r.gl j
      rw [h2]
      simp only [en, lane, RowP.gl, r]
      by_cases hjd : j < d.d <;> simp [hjd])
    (by show Tree.Covers (2 ^ e) t1; rw [← he]; exact hc1) (hd1.trans hsb) he1 hdvf hdv (hmT ())
    (fun j => if h : j < d.B then
      (if ((tl.arange 0 d.B).ltS (IntV.ofNat d.d : I32)).get (lane d j h) then
        scT.get (lane d j h) else c32 0) else default)
    (by
      intro j hj
      have hjB : j < d.B := lt_of_lt_of_le hj hdB
      have hm := mask_get hd (lane d j hjB)
      have hen : en d (lane d j hjB) = true := by simp only [en, lane]; exact decide_eq_true hj
      simp only [hjB, ↓reduceDIte, hm, hen, ite_true]
      exact hsc (lane d j hjB))
    (by
      intro j h1 h2
      have hjB : j < d.B := by rw [he]; exact h2
      have hm := mask_get hd (lane d j hjB)
      have hen : en d (lane d j hjB) = false := by
        simp only [en, lane]; exact decide_eq_false (Nat.not_lt.mpr h1)
      simp only [hjB, ↓reduceDIte, hm, hen]
      rfl)
    (fun j => if h : j < d.B then qT.get1 ⟨j, h⟩ else default)
    (by
      intro j hj
      have hjB : j < d.B := by rw [he]; exact hj
      simp only [hjB, ↓reduceDIte]
      exact hqT (lane d j hjB))
    (by show Tree.Covers (2 ^ e) t2; rw [← he]; exact hc2) (hd2.trans hsb) he2 (hvT ()) (hmean16 ()) (hmean ()) (hsqT ()) (hstd16 ()) (hstd ())
    hm32.1 hm32.2 (hprT ()) (hsc16 ()) (hscs ()) (hcsT ()) (hck16 ()) (hckT ())
  obtain ⟨hckf, hcut, hckb⟩ := hst
  have hcT : 2 ^ (-130 : ℤ) ≤ cutTol d init i.val := r.cutTol_ge
  -- each lane
  intro j hj
  have hjB : j < d.B := lt_of_lt_of_le hj hdB
  set q := lane d j hjB with hq
  have hen : en d q = true := by simp only [hq, en, lane]; exact decide_eq_true hj
  obtain ⟨hgf, hgv'⟩ := hgl q
  obtain ⟨huf, huv'⟩ := hul q
  simp only [hen, ite_true] at hgv' huv'
  replace hgv' : (gate.get q).toReal = gv d init i.val j := hgv'
  replace huv' : (up.get q).toReal = uv d init i.val j := huv'
  have hgb : |(gate.get q).toReal| ≤ 2 ^ 32 := by rw [hgv']; exact hgv j hj
  have hub : |(up.get q).toReal| ≤ 2 ^ 32 := by
    rw [huv']; exact inOk_le (hpre.2.2.1 _ (huidx j hj))
  obtain ⟨hyf, hye, hy0⟩ := lane_ok hP (cR := cutRef d init i.val) (cT := cutTol d init i.val)
    hgf hgb huf hub hckf hckb hcT hcut (hs12 q) (hsg16 q) (hsg0 q) (hp1 q) (hp2 q) (hz q)
    (hinn q) (hx2 q) (hnx q) (hex q) (hden q) (hsg q) (htw q) (hth q) (hhh q) (hot q) (hact q)
    (hactb q) (hav q) (hpr q) (hy16 q)
  refine ⟨y16.get q, hread q hen, hyf, ?_, ?_⟩
  · rw [hgv', huv'] at hye
    exact hye
  · intro hle
    rw [← hgv'] at hle
    exact hy0 hle

end Claims.GeluAndMulSparse
