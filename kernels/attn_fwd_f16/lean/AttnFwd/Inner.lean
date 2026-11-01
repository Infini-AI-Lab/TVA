import AttnFwd.Tile

/-!
# `attn_fwd_f16`: the loop of `_attn_fwd_inner` (lines 69–109)

* `qk_lane`, `pv_lane`: `Num.DotF16` at one element of `tl.dot(q, k)` (line 73) and of
  `tl.dot(p, v, acc)` (line 103);
* `veOf`, `StInv`: the values of `v` a row sums, and every row's `RowSt`;
* `bodyPlain`, `bodyMasked`: the loop's body for `STAGE` 1 or 3 and for `STAGE` 2, as the model
  has them once the compile-time `if` is decided;
* `safe_loop_plain`, `safe_loop_masked`: the loop keeps `StInv`.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

/-! ## One element of a `tl.dot` -/

theorem val16_of {init : Mem} {b : Buf .f16} {i : ℕ} {x : F16}
    (h : init.read .f16 (b.ptr (i : ℤ)).addr = some x) : val16 init b i = x.toRat := by
  unfold val16; rw [h]

theorem fin_of {init : Mem} {b : Buf .f16} {i : ℕ} {x : F16}
    (h : init.read .f16 (b.ptr (i : ℤ)).addr = some x) (hf : fin16 init b i) : x.isFinite = true := by
  obtain ⟨y, hy, hyf⟩ := hf
  rw [h] at hy; cases hy; exact hyf

theorem val?_f16 {x : F16} (h : x.isFinite = true) : DType.val? .f16 x = some x.toRat := by
  show (if x.isFinite = true then some x.toRat else none) = _
  rw [ite_eq_left h]

/-- **An element of `tl.dot(q, k)`** (line 73), for a covered head: finite and within the
`HEAD_DIM` constants of `Num.DotF16` of the exact score, relative to the lifted magnitude
`scoreMag` (library round 25). -/
theorem qk_lane {P : Platform} (hdot : Rules.Num.DotF16 P) {d : Inst} (hd : d.ok) {init : Mem}
    {bh n j : ℕ} (hn : n < d.N_CTX) (hj : j < d.N_CTX) (hH : HeadOk d init bh)
    {M : ℕ} {q : Tensor [M, d.HEAD_DIM] F16} {kv : Tensor [64, d.HEAD_DIM] F16} {S : Tensor [M, 64] F32}
    (hS : P.dot cfg16 q kv.trans (tl.zeros [M, 64] .f32) S) (r : Fin M) (jj : Fin 64)
    (hqr : ∀ e : Fin d.HEAD_DIM, init.read .f16 (d.q.ptr (idx d bh n e : ℕ)).addr = some (q.get2 r e))
    (hkr : ∀ e : Fin d.HEAD_DIM, init.read .f16 (d.k.ptr (idx d bh j e : ℕ)).addr = some (kv.get2 jj e)) :
    (S.get2 r jj).isFinite = true ∧
      |(S.get2 r jj).toReal - (score d init bh n j : ℝ)| ≤
        (Rules.Num.relDot16 d.HEAD_DIM : ℝ) * (scoreMag d init bh n j : ℝ) +
          (Rules.Num.absDot16 d.HEAD_DIM : ℝ) := by
  have hqf : ∀ e : Fin d.HEAD_DIM, (q.get2 r e).isFinite = true := fun e =>
    fin_of (hqr e) (hH.1 n e hn e.2).1
  have hkf : ∀ e : Fin d.HEAD_DIM, (kv.get2 jj e).isFinite = true := fun e =>
    fin_of (hkr e) (hH.1 j e hj e.2).2.1
  have hsc : score d init bh n j = ∑ e : Fin d.HEAD_DIM, (q.get2 r e).toRat * (kv.get2 jj e).toRat := by
    unfold score
    rw [Finset.sum_range (fun e => val16 init d.q (idx d bh n e) * val16 init d.k (idx d bh j e))]
    exact Finset.sum_congr rfl fun e _ => by rw [val16_of (hqr e), val16_of (hkr e)]
  have hsa : scoreMag d init bh n j = ∑ e : Fin d.HEAD_DIM,
      Rules.Num.liftMag .f16 (q.get2 r e).toRat * Rules.Num.liftMag .f16 (kv.get2 jj e).toRat := by
    unfold scoreMag
    rw [Finset.sum_range (fun e => Rules.Num.liftMag .f16 (val16 init d.q (idx d bh n e)) *
      Rules.Num.liftMag .f16 (val16 init d.k (idx d bh j e)))]
    exact Finset.sum_congr rfl fun e _ => by rw [val16_of (hqr e), val16_of (hkr e)]
  have hσ : 1 / 2 ^ 64 ≤ sigma d := hd.scale.2.1
  have hsa1 : scoreMag d init bh n j ≤ 2 ^ 74 := by
    have h1 := hH.2.2 n j hn hj
    have h0 := scoreMag_nonneg d init bh n j
    have : (1 / 2 ^ 64 : ℚ) * scoreMag d init bh n j ≤ 2 ^ 10 := by nlinarith
    nlinarith
  have hD := headDim_le hd
  have hrel : Rules.Num.relDot16 d.HEAD_DIM ≤ 1 := by
    unfold Rules.Num.relDot16
    have : ((d.HEAD_DIM : ℚ) + 2) ≤ 258 := by have : (d.HEAD_DIM : ℚ) ≤ 256 := by exact_mod_cast hD
                                              linarith
    have h2 : (2 : ℚ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
    rw [h2]; nlinarith
  have habs : Rules.Num.absDot16 d.HEAD_DIM ≤ 1 := by
    unfold Rules.Num.absDot16
    have : ((d.HEAD_DIM : ℚ) + 2) ≤ 258 := by have : (d.HEAD_DIM : ℚ) ≤ 256 := by exact_mod_cast hD
                                              linarith
    have h2 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
    rw [h2]; nlinarith
  have hrel0 : 0 ≤ Rules.Num.relDot16 d.HEAD_DIM := by unfold Rules.Num.relDot16; positivity
  obtain ⟨vr, hvr, herr⟩ := hdot .tf32 q kv.trans (tl.zeros [M, 64] .f32) S hS r jj
    (fun e => (q.get2 r e).toRat) (fun e => (kv.get2 jj e).toRat) 0 (fun e => val?_f16 (hqf e))
    (fun e => val?_f16 (hkf e)) DotLoop.zero_val (by
      show _ ≤ Format.binary32.maxVal
      rw [← hsa]
      have : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := by
        have := maxVal32; exact_mod_cast this
      rw [this]
      simp only [abs_zero, zero_add]
      nlinarith)
  obtain ⟨hSf, hSv⟩ := DotLoop.val?_f32.mp hvr
  refine ⟨hSf, ?_⟩
  rw [← hsa, ← hsc, zero_add] at herr
  simp only [abs_zero, zero_add] at herr
  rw [← hSv] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  rw [toReal_def]
  exact this

/-- **An element of `tl.dot(p, v, acc)`** (line 103): `Num.DotF16` (relative to the lifted
magnitudes `|acc| + Σ liftMag p·liftMag v`), in the form `astep` takes. -/
theorem pv_lane {P : Platform} (hdot : Rules.Num.DotF16 P) {D M : ℕ} {q16 : Tensor [M, 64] F16}
    {vv : Tensor [64, D] F16} {a1 acc' : Tensor [M, D] F32} (h : P.dot cfg16 q16 vv a1 acc')
    (r : Fin M) (e : Fin D) (hvf : ∀ k, (vv.get2 k e).isFinite = true) :
    (a1.get2 r e).isFinite = true → (∀ k, (q16.get2 r k).isFinite = true) →
      |(a1.get2 r e).toRat| + ∑ k, Rules.Num.liftMag .f16 (q16.get2 r k).toRat * Rules.Num.liftMag .f16 (vv.get2 k e).toRat +
        (Rules.Num.relDot16 64 * (|(a1.get2 r e).toRat| +
          ∑ k, Rules.Num.liftMag .f16 (q16.get2 r k).toRat * Rules.Num.liftMag .f16 (vv.get2 k e).toRat) +
          Rules.Num.absDot16 64) ≤ Format.binary32.maxVal →
      (acc'.get2 r e).isFinite = true ∧
        |(acc'.get2 r e).toRat - ((a1.get2 r e).toRat + ∑ k, (q16.get2 r k).toRat * (vv.get2 k e).toRat)| ≤
          Rules.Num.relDot16 64 * (|(a1.get2 r e).toRat| +
            ∑ k, Rules.Num.liftMag .f16 (q16.get2 r k).toRat * Rules.Num.liftMag .f16 (vv.get2 k e).toRat) +
            Rules.Num.absDot16 64 := by
  intro ha hq hroom
  obtain ⟨vr, hvr, herr⟩ := hdot .tf32 q16 vv a1 acc' h r e (fun k => (q16.get2 r k).toRat)
    (fun k => (vv.get2 k e).toRat) (a1.get2 r e).toRat (fun k => val?_f16 (hq k)) (fun k => val?_f16 (hvf k))
    (DotLoop.val?_f32.mpr ⟨ha, rfl⟩) hroom
  obtain ⟨hf, hv⟩ := DotLoop.val?_f32.mp hvr
  exact ⟨hf, by rw [hv]; exact herr⟩

/-! ## The rows' state -/

/-- The values of `v` row sums use: key `j`, column `e` of head `bh` (`0` past `N_CTX`). -/
noncomputable def veOf (d : Inst) (init : Mem) (bh : ℕ) : Fin d.HEAD_DIM → ℕ → ℝ :=
  fun e j => if j < d.N_CTX then (val16 init d.v (idx d bh j e) : ℝ) else 0

theorem veOf_le {d : Inst} {init : Mem} {bh : ℕ} (hH : HeadOk d init bh) (e : Fin d.HEAD_DIM) (j : ℕ) :
    |veOf d init bh e j| ≤ 2 ^ 15 := by
  unfold veOf
  split_ifs with hj
  · have := hH.2.1 j e hj e.2; exact_mod_cast this
  · simp

/-- Every row of program `(sm, bh)` after `t` blocks: for a covered head and every bound `X ≤ 2^10`
on the row's scaled score magnitudes, `RowSt` of its context. -/
def StInv (d : Inst) (init : Mem) (c : F32) (sm bh t : ℕ) (acc : Tensor [128, d.HEAD_DIM] F32)
    (l m : Tensor [128] F32) : Prop :=
  ∀ r : Fin 128, HeadOk d init bh → ∀ X : ℚ, X ≤ 2 ^ 10 →
    (∀ j ∈ keys d (128 * sm + r), sigma d * scoreMag d init bh (128 * sm + r) j ≤ X) →
    RowSt (rowCtx d init c.toReal bh (128 * sm + r) X) (veOf d init bh) t (m.get1 r) (l.get1 r)
      (fun e => acc.get2 r e)

/-- `qk_scale`'s facts (`qk_scale_facts`). -/
def CFacts (d : Inst) (c : F32) : Prop :=
  c.isFinite = true ∧ 0 < c.toReal ∧ c.toReal ≤ 2 ^ 65 ∧ c.toReal ≤ 3 / 2 * (sigma d : ℝ) ∧
    |c.toReal * Real.log 2 - (sigma d : ℝ)| ≤ (sigma d : ℝ) / 2 ^ 22

/-- The `q` tile of program `(sm, bh)` holds `q`'s rows `128·sm …` of head `bh` (line 221). -/
def QReads (d : Inst) (init : Mem) (sm bh : ℕ) (q : Tensor [128, d.HEAD_DIM] F16) : Prop :=
  ∀ (r : Fin 128) (e : Fin d.HEAD_DIM), init.read .f16 (d.q.ptr (idx d bh (128 * sm + r) e : ℕ)).addr = some (q.get2 r e)

/-- Every program's read footprint: `q`, `k` and `v`. -/
def Rd (d : Inst) : Set Int :=
  d.q.elems (below (rows d * d.HEAD_DIM)) ∪ d.k.elems (below (rows d * d.HEAD_DIM)) ∪
    d.v.elems (below (rows d * d.HEAD_DIM))

/-- The loop's body for `STAGE` 1 or 3 (lines 72–73, 80–109), the compile-time `if` decided. -/
def bodyPlain (D : ℕ) (q : Tensor [128, D] F16) (dk dv : TensorDesc .f16) (c : F32) (_start_n : I32)
    (x : Tensor [128, D] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32) :
    Prog (Tensor [128, D] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32) := do
  let kk ← tl.desc_load (bs := [64, D]) dk [x.2.2.2.1, IntV.ofInt 0]
  let qk ← tl.dot cfg16 q kk.trans (tl.zeros [128, 64] .f32)
  let mx ← tl.max1 true qk
  let a ← tl.mul mx (bc c)
  let m_ij ← tl.maximum x.2.2.1 a
  let sc ← tl.mul qk (bc c)
  let qk ← tl.subc sc (m_ij.col 64)
  let p ← tl.exp2 qk
  let dm ← tl.sub x.2.2.1 m_ij
  let alpha ← tl.exp2 dm
  let l_ij ← tl.sum1 p
  let acc ← tl.mul x.1 (alpha.col D)
  let v ← tl.desc_load (bs := [64, D]) dv [x.2.2.2.2, IntV.ofInt 0]
  let p : Tensor [128, 64] F16 ← tl.cvt p
  let acc ← tl.dot cfg16 p v acc
  let l_i ← tl.mul_add x.2.1 alpha l_ij
  pure (acc, l_i, m_ij, x.2.2.2.1 + IntV.ofNat 64, x.2.2.2.2 + IntV.ofNat 64)

theorem f32_f16_ok : Format.binary32.Std ∧ Format.binary16.Std ∧ tl.RoundOk Format.binary32 Format.binary16 .rn :=
  ⟨by unfold Format.Std; simp, by unfold Format.Std; simp, Or.inl rfl⟩

theorem idx_eq (d : Inst) (bh n e : ℕ) : idx d bh n e = (bh * d.N_CTX + n) * d.HEAD_DIM + e := rfl

/-- **The loop for `STAGE` 1 or 3**: blocks `t0, …, t0 + n − 1`, every key of which every row
attends to. -/
theorem safe_loop_plain {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotF16 P) (hdtot : Rules.Num.DotTotal P cfg16) (hexp : Rules.Num.Exp2F32 P)
    (hni : Rules.Num.Exp2F32NegInf P) (hext : Rules.Num.MathTotal P .exp2 .binary32)
    {R W : Set Int} {α : Type} {Q : α → Mem → Prop} (hR : Rd d ⊆ R)
    {sm bh : ℕ} (hsm : 128 * sm + 128 ≤ d.N_CTX) (hbh : bh < d.Z * d.H)
    {c : F32} (hc : CFacts d c) {q : Tensor [128, d.HEAD_DIM] F16} (hq : QReads d init sm bh q)
    {lo hi : I32} {t0 n : ℕ} (hlo : lo.toInt = ((64 * t0 : ℕ) : ℤ)) (hhi : hi.toInt = ((64 * (t0 + n) : ℕ) : ℤ))
    (hn : 64 * (t0 + n) ≤ d.N_CTX)
    (hall : ∀ (r : Fin 128) (j : ℕ), j < 64 * (t0 + n) → alOf d (128 * sm + r) j = true)
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32} {oy : I32}
    (hoy : oy.toInt = ((bh * d.N_CTX : ℕ) : ℤ)) (hinv : StInv d init c sm bh t0 acc l m)
    {k : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32 → Prog α}
    (hk : ∀ st : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32,
      StInv d init c sm bh (t0 + n) st.1 st.2.1 st.2.2.1 → Safe ((launch d).env P) R W Q init (k st)) :
    Safe ((launch d).env P) R W Q init
      (tl.forRange lo hi (IntV.ofNat 64) (acc, l, m, oy + lo, oy + lo)
        (bodyPlain d.HEAD_DIM q (descOf d d.k 64) (descOf d d.v 64) c) >>= k) := by
  obtain ⟨hcf, hc0, hc65, hc32, hcl⟩ := hc
  have hrows := hd.rows_lt
  have hN := hd.n_le
  have hbhN : bh * d.N_CTX + d.N_CTX ≤ rows d := by
    unfold rows
    calc bh * d.N_CTX + d.N_CTX = (bh + 1) * d.N_CTX := by ring
      _ ≤ (d.Z * d.H) * d.N_CTX := Nat.mul_le_mul_right _ hbh
  have hRk : d.k.elems (below (rows d * d.HEAD_DIM)) ⊆ R :=
    fun x hx => hR (Or.inl (Or.inr hx))
  have hRv : d.v.elems (below (rows d * d.HEAD_DIM)) ⊆ R := fun x hx => hR (Or.inr hx)
  have h64 : (IntV.ofNat 64 : I32).toInt = ((64 : ℕ) : ℤ) := i32_nat (by norm_num)
  have hrange : tl.RangeOk lo hi (IntV.ofNat 64) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [hlo, hhi, h64]; push_cast; norm_num; omega
  have htc : tl.tripCount lo.toInt hi.toInt (IntV.ofNat 64 : I32).toInt = n := by
    rw [hlo, hhi, h64]; unfold tl.tripCount; push_cast; omega
  have hoff0 : (oy + lo).toInt = ((bh * d.N_CTX + 64 * (t0 + 0) : ℕ) : ℤ) := by
    rw [IntV.I32.add_nat hoy hlo (by omega)]; congr 1
  refine Safe.mono (safe_forRange (Wk := W) hrange
    (fun t st m₁ => m₁ = init ∧ st.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (t0 + t) : ℕ) : ℤ) ∧
      st.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (t0 + t) : ℕ) : ℤ) ∧
      StInv d init c sm bh (t0 + t) st.1 st.2.1 st.2.2.1)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, hoff0, hoff0, hinv⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · rw [htc]
    rintro t ht ⟨sacc, sl, smm, sok, sov⟩ m₁ ⟨hm₁, hok, hov, hinv'⟩
    subst m₁
    simp only at hok hov hinv'
    have htn : 64 * (t0 + t) + 64 ≤ d.N_CTX := by omega
    have hO : bh * d.N_CTX + 64 * (t0 + t) + 64 ≤ rows d := by omega
    simp only [bodyPlain]
    -- line 72
    refine safe_load_rows hd d.k hd.align.2.1 hd.lens.2.1 hpre.2.1 hRk hok hO fun kv hkv => ?_
    -- line 73
    refine safe_dot (hdtot q kv.trans (tl.zeros [128, 64] .f32)) fun S hS => ?_
    -- line 80
    refine safe_max1 (by norm_num) true fun mx hmx => ?_
    refine safe_mul fun a ha => ?_
    refine safe_maximum fun mij hmij => ?_
    -- line 81
    refine safe_mul fun Sc hSc => ?_
    refine safe_subc fun x hx => ?_
    -- line 82
    refine safe_math (fun i => hext _) fun p hp => ?_
    -- line 84
    refine safe_sub fun dm hdm => ?_
    refine safe_math (fun i => hext _) fun al hal => ?_
    -- line 85
    refine safe_sum1 (by norm_num) fun lij hlij => ?_
    -- line 95
    refine safe_mul fun a1 ha1 => ?_
    -- line 100
    refine safe_load_rows hd d.v hd.align.2.2.1 hd.lens.2.2.1 hpre.2.2.1 hRv hov hO fun vv hvv => ?_
    -- line 101
    refine safe_cvt f32_f16_ok fun q16 hq16 => ?_
    -- line 103
    refine safe_dot (hdtot q16 vv a1) fun acc' hacc' => ?_
    -- line 106
    refine safe_mul_add fun l' hl' => ?_
    -- lines 108–109
    refine safe_ret ⟨rfl, ?_, ?_, ?_⟩
    · show (sok + IntV.ofNat 64).toInt = _
      rw [IntV.I32.add_nat hok h64 (by omega)]; congr 1
    · show (sov + IntV.ofNat 64).toInt = _
      rw [IntV.I32.add_nat hov h64 (by omega)]; congr 1
    · intro r hH X hX hXk
      have hnr : 128 * sm + r < d.N_CTX := by omega
      obtain ⟨hRok, -⟩ := rowCtx_ok hd init hc0 hc65 hc32 bh hnr hX hXk
      have hkey : ∀ jj : Fin 64, 64 * (t0 + t) + jj < d.N_CTX := fun jj => by omega
      have hkr : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM),
          init.read .f16 (d.k.ptr (idx d bh (64 * (t0 + t) + jj) e : ℕ)).addr = some (kv.get2 jj e) := by
        intro jj e
        rw [idx_eq, show bh * d.N_CTX + (64 * (t0 + t) + jj) = bh * d.N_CTX + 64 * (t0 + t) + jj by ring]
        exact hkv jj e
      have hvr : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM),
          init.read .f16 (d.v.ptr (idx d bh (64 * (t0 + t) + jj) e : ℕ)).addr = some (vv.get2 jj e) := by
        intro jj e
        rw [idx_eq, show bh * d.N_CTX + (64 * (t0 + t) + jj) = bh * d.N_CTX + 64 * (t0 + t) + jj by ring]
        exact hvv jj e
      have hvf : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM), (vv.get2 jj e).isFinite = true := fun jj e =>
        fin_of (hvr jj e) (hH.1 _ e (hkey jj) e.2).2.2
      have hvt : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM), (vv.get2 jj e).isFinite = true ∧
          (vv.get2 jj e).toReal = veOf d init bh e (64 * (t0 + t) + jj) := fun jj e => by
        refine ⟨hvf jj e, ?_⟩
        unfold veOf; rw [ite_eq_left (hkey jj), val16_of (hvr jj e), toReal_def]
      exact row_step_plain (t := t0 + t) (S := fun jj => S.get2 r jj) (Sc := fun jj => Sc.get2 r jj)
        (x := fun jj => x.get2 r jj) (p := fun jj => p.get2 r jj) (q := fun jj => q16.get2 r jj)
        (vt := fun jj e => vv.get2 jj e) (acc := fun e => sacc.get2 r e) (a1 := fun e => a1.get2 r e)
        (acc' := fun e => acc'.get2 r e) (mx := mx.get1 r) (a := a.get1 r) (m' := mij.get1 r)
        (dm := dm.get1 r) (al := al.get1 r) (lij := lij.get1 r) (l' := l'.get1 r)
        hexp hni hRok (by omega) (veOf_le hH) hcf rfl
        (fun jj => qk_lane hdot hd hnr (hkey jj) hH hS r jj (hq r) (hkr jj))
        (fun jj => hall r _ (by omega)) (hmx r) (ha (r, ())) (hmij (r, ())) (fun jj => hSc (r, jj, ()))
        (fun jj => hx (r, jj, ())) (fun jj => hp (r, jj, ())) (hdm (r, ())) (hal (r, ())) (hlij r)
        (fun jj => hq16 (r, jj, ())) (fun e => ha1 (r, e, ())) hvt
        (fun e => pv_lane hdot hacc' r e (fun k => hvf k e)) (hl' (r, ())) (hinv' r hH X hX hXk)
  · rw [htc]
    rintro st m₁ ⟨hm₁, -, -, hst⟩
    subst m₁
    exact hk st hst

theorem get2_where {α : Type} {m n : ℕ} (c : Tensor [m, n] Bool) (x y : Tensor [m, n] α) (i : Fin m) (j : Fin n) :
    (Tensor.where_ c x y).get2 i j = if c.get2 i j then x.get2 i j else y.get2 i j := rfl

theorem get2_bc {α : Type} {m n : ℕ} (a : α) (i : Fin m) (j : Fin n) : (bc a : Tensor [m, n] α).get2 i j = a := rfl

/-- The loop's body for `STAGE` 2 (lines 72–78, 82–109), the compile-time `if` decided. -/
def bodyMasked (D : ℕ) (q : Tensor [128, D] F16) (dk dv : TensorDesc .f16) (c : F32)
    (offs_m : Tensor [128] I32) (offs_n : Tensor [64] I32) (start_n : I32)
    (x : Tensor [128, D] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32) :
    Prog (Tensor [128, D] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32) := do
  let kk ← tl.desc_load (bs := [64, D]) dk [x.2.2.2.1, IntV.ofInt 0]
  let qk ← tl.dot cfg16 q kk.trans (tl.zeros [128, 64] .f32)
  let qk ← tl.mul_add qk (bc c)
    (((start_n + offs_n.row 128).le (offs_m.col 64)).where_ (bc (FloatV.litF32 (FloatV.ofPat 0)))
      (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000))))
  let mx ← tl.max1 true qk
  let m_ij ← tl.maximum x.2.2.1 mx
  let qk ← tl.sub qk (m_ij.col 64)
  let p ← tl.exp2 qk
  let dm ← tl.sub x.2.2.1 m_ij
  let alpha ← tl.exp2 dm
  let l_ij ← tl.sum1 p
  let acc ← tl.mul x.1 (alpha.col D)
  let v ← tl.desc_load (bs := [64, D]) dv [x.2.2.2.2, IntV.ofInt 0]
  let p : Tensor [128, 64] F16 ← tl.cvt p
  let acc ← tl.dot cfg16 p v acc
  let l_i ← tl.mul_add x.2.1 alpha l_ij
  pure (acc, l_i, m_ij, x.2.2.2.1 + IntV.ofNat 64, x.2.2.2.2 + IntV.ofNat 64)

/-- **The loop for `STAGE` 2** (causal): the two blocks `2·start_m`, `2·start_m + 1` of the
program's own query rows, under the causal mask. -/
theorem safe_loop_masked {d : Inst} (hd : d.ok) (hcausal : d.causal = true) {init : Mem}
    (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotF16 P) (hdtot : Rules.Num.DotTotal P cfg16) (hexp : Rules.Num.Exp2F32 P)
    (hni : Rules.Num.Exp2F32NegInf P) (hext : Rules.Num.MathTotal P .exp2 .binary32)
    {R W : Set Int} {α : Type} {Q : α → Mem → Prop} (hR : Rd d ⊆ R)
    {sm bh : ℕ} (hsm : 128 * sm + 128 ≤ d.N_CTX) (hbh : bh < d.Z * d.H)
    {c : F32} (hc : CFacts d c) {q : Tensor [128, d.HEAD_DIM] F16} (hq : QReads d init sm bh q)
    {offs_m : Tensor [128] I32} {offs_n : Tensor [64] I32}
    (hom : ∀ r : Fin 128, (offs_m.get1 r).toInt = ((128 * sm + r : ℕ) : ℤ))
    (hon : ∀ jj : Fin 64, (offs_n.get1 jj).toInt = ((jj : ℕ) : ℤ))
    {lo hi : I32} (hlo : lo.toInt = ((64 * (2 * sm) : ℕ) : ℤ)) (hhi : hi.toInt = ((64 * (2 * sm + 2) : ℕ) : ℤ))
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32} {oy : I32}
    (hoy : oy.toInt = ((bh * d.N_CTX : ℕ) : ℤ)) (hinv : StInv d init c sm bh (2 * sm) acc l m)
    {k : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32 → Prog α}
    (hk : ∀ st : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32,
      StInv d init c sm bh (2 * sm + 2) st.1 st.2.1 st.2.2.1 → Safe ((launch d).env P) R W Q init (k st)) :
    Safe ((launch d).env P) R W Q init
      (tl.forRange lo hi (IntV.ofNat 64) (acc, l, m, oy + lo, oy + lo)
        (bodyMasked d.HEAD_DIM q (descOf d d.k 64) (descOf d d.v 64) c offs_m offs_n) >>= k) := by
  obtain ⟨hcf, hc0, hc65, hc32, hcl⟩ := hc
  have hrows := hd.rows_lt
  have hN := hd.n_le
  have hbhN : bh * d.N_CTX + d.N_CTX ≤ rows d := by
    unfold rows
    calc bh * d.N_CTX + d.N_CTX = (bh + 1) * d.N_CTX := by ring
      _ ≤ (d.Z * d.H) * d.N_CTX := Nat.mul_le_mul_right _ hbh
  have hRk : d.k.elems (below (rows d * d.HEAD_DIM)) ⊆ R :=
    fun x hx => hR (Or.inl (Or.inr hx))
  have hRv : d.v.elems (below (rows d * d.HEAD_DIM)) ⊆ R := fun x hx => hR (Or.inr hx)
  have h64 : (IntV.ofNat 64 : I32).toInt = ((64 : ℕ) : ℤ) := i32_nat (by norm_num)
  have hrange : tl.RangeOk lo hi (IntV.ofNat 64) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [hlo, hhi, h64]; push_cast; norm_num; omega
  have htc : tl.tripCount lo.toInt hi.toInt (IntV.ofNat 64 : I32).toInt = 2 := by
    rw [hlo, hhi, h64]; unfold tl.tripCount; push_cast; omega
  have hoff0 : (oy + lo).toInt = ((bh * d.N_CTX + 64 * (2 * sm + 0) : ℕ) : ℤ) := by
    rw [IntV.I32.add_nat hoy hlo (by omega)]; congr 1
  refine Safe.mono (safe_forRange (Wk := W) hrange
    (fun t st m₁ => m₁ = init ∧ st.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (2 * sm + t) : ℕ) : ℤ) ∧
      st.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (2 * sm + t) : ℕ) : ℤ) ∧
      StInv d init c sm bh (2 * sm + t) st.1 st.2.1 st.2.2.1)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, hoff0, hoff0, hinv⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · rw [htc]
    rintro t ht ⟨sacc, sl, smm, sok, sov⟩ m₁ ⟨hm₁, hok, hov, hinv'⟩
    subst m₁
    simp only at hok hov hinv'
    have htn : 64 * (2 * sm + t) + 64 ≤ d.N_CTX := by omega
    have hO : bh * d.N_CTX + 64 * (2 * sm + t) + 64 ≤ rows d := by omega
    have hsn : (IntV.ofInt (lo.toInt + ↑t * (IntV.ofNat 64 : I32).toInt) : I32).toInt =
        ((64 * (2 * sm + t) : ℕ) : ℤ) := by
      rw [hlo, h64, IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega)]; push_cast; ring
    generalize (IntV.ofInt (lo.toInt + ↑t * (IntV.ofNat 64 : I32).toInt) : I32) = start_n at hsn ⊢
    simp only [bodyMasked]
    -- line 72
    refine safe_load_rows hd d.k hd.align.2.1 hd.lens.2.1 hpre.2.1 hRk hok hO fun kv hkv => ?_
    -- line 73
    refine safe_dot (hdtot q kv.trans (tl.zeros [128, 64] .f32)) fun S hS => ?_
    -- line 76
    refine safe_mul_add fun x' hx' => ?_
    -- line 77
    refine safe_max1 (by norm_num) true fun mx hmx => ?_
    refine safe_maximum fun mij hmij => ?_
    -- line 78
    refine safe_sub fun x hx => ?_
    -- line 82
    refine safe_math (fun i => hext _) fun p hp => ?_
    -- line 84
    refine safe_sub fun dm hdm => ?_
    refine safe_math (fun i => hext _) fun al hal => ?_
    -- line 85
    refine safe_sum1 (by norm_num) fun lij hlij => ?_
    -- line 95
    refine safe_mul fun a1 ha1 => ?_
    -- line 100
    refine safe_load_rows hd d.v hd.align.2.2.1 hd.lens.2.2.1 hpre.2.2.1 hRv hov hO fun vv hvv => ?_
    -- line 101
    refine safe_cvt f32_f16_ok fun q16 hq16 => ?_
    -- line 103
    refine safe_dot (hdtot q16 vv a1) fun acc' hacc' => ?_
    -- line 106
    refine safe_mul_add fun l' hl' => ?_
    -- lines 108–109
    refine safe_ret ⟨rfl, ?_, ?_, ?_⟩
    · show (sok + IntV.ofNat 64).toInt = _
      rw [IntV.I32.add_nat hok h64 (by omega)]; congr 1
    · show (sov + IntV.ofNat 64).toInt = _
      rw [IntV.I32.add_nat hov h64 (by omega)]; congr 1
    · intro r hH X hX hXk
      have hnr : 128 * sm + r < d.N_CTX := by omega
      obtain ⟨hRok, -⟩ := rowCtx_ok hd init hc0 hc65 hc32 bh hnr hX hXk
      have hkey : ∀ jj : Fin 64, 64 * (2 * sm + t) + jj < d.N_CTX := fun jj => by omega
      have hkr : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM),
          init.read .f16 (d.k.ptr (idx d bh (64 * (2 * sm + t) + jj) e : ℕ)).addr = some (kv.get2 jj e) := by
        intro jj e
        rw [idx_eq, show bh * d.N_CTX + (64 * (2 * sm + t) + jj) = bh * d.N_CTX + 64 * (2 * sm + t) + jj by ring]
        exact hkv jj e
      have hvr : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM),
          init.read .f16 (d.v.ptr (idx d bh (64 * (2 * sm + t) + jj) e : ℕ)).addr = some (vv.get2 jj e) := by
        intro jj e
        rw [idx_eq, show bh * d.N_CTX + (64 * (2 * sm + t) + jj) = bh * d.N_CTX + 64 * (2 * sm + t) + jj by ring]
        exact hvv jj e
      have hvf : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM), (vv.get2 jj e).isFinite = true := fun jj e =>
        fin_of (hvr jj e) (hH.1 _ e (hkey jj) e.2).2.2
      have hvt : ∀ (jj : Fin 64) (e : Fin d.HEAD_DIM), (vv.get2 jj e).isFinite = true ∧
          (vv.get2 jj e).toReal = veOf d init bh e (64 * (2 * sm + t) + jj) := fun jj e => by
        refine ⟨hvf jj e, ?_⟩
        unfold veOf; rw [ite_eq_left (hkey jj), val16_of (hvr jj e), toReal_def]
      -- the mask (line 75) and the values of line 76
      have hal_iff : ∀ jj : Fin 64, alOf d (128 * sm + r) (64 * (2 * sm + t) + jj) =
          decide (64 * (2 * sm + t) + jj ≤ 128 * sm + r) := fun jj => by
        unfold alOf; rw [hcausal]
        exact decide_eq_decide.mpr ⟨fun h => by rcases h.2 with h' | h' <;> simp_all, fun h => ⟨hkey jj, Or.inr h⟩⟩
      have hmask : ∀ jj : Fin 64, ((start_n + offs_n.row 128).le (offs_m.col 64)).get2 r jj =
          decide (64 * (2 * sm + t) + jj ≤ 128 * sm + r) := fun jj => by
        show IntV.le (start_n + offs_n.get1 jj) (offs_m.get1 r) = _
        rw [i32_le (IntV.I32.add_nat hsn (hon jj) (by omega)) (hom r)]
        exact decide_eq_decide.mpr (by push_cast; omega)
      have hw : ∀ jj : Fin 64,
          ((((start_n + offs_n.row 128).le (offs_m.col 64)).where_ (bc (FloatV.litF32 (FloatV.ofPat 0)))
            (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get2 r jj).isFinite = true ∧
          ((rowCtx d init c.toReal bh (128 * sm + r) X).al (64 * (2 * sm + t) + jj) = true →
            ((((start_n + offs_n.row 128).le (offs_m.col 64)).where_ (bc (FloatV.litF32 (FloatV.ofPat 0)))
              (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get2 r jj).toReal = 0) ∧
          ((rowCtx d init c.toReal bh (128 * sm + r) X).al (64 * (2 * sm + t) + jj) = false →
            ((((start_n + offs_n.row 128).le (offs_m.col 64)).where_ (bc (FloatV.litF32 (FloatV.ofPat 0)))
              (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get2 r jj).toReal = -1000000) := by
        intro jj
        show _ ∧ (alOf d (128 * sm + r) (64 * (2 * sm + t) + jj) = true → _) ∧
          (alOf d (128 * sm + r) (64 * (2 * sm + t) + jj) = false → _)
        rw [get2_where, hmask jj, hal_iff jj, get2_bc, get2_bc]
        by_cases hj : 64 * (2 * sm + t) + jj ≤ 128 * sm + r
        · simp only [hj, decide_true, ite_true, true_implies, Bool.true_eq_false, false_implies, and_true]
          exact ⟨w0_facts.1, by rw [toReal_def, w0_facts.2]; simp⟩
        · simp only [hj, decide_false, Bool.false_eq_true, ite_false, false_implies, true_implies, true_and]
          exact ⟨w1_facts.1, by rw [toReal_def, w1_facts.2]; simp⟩
      have hglob : ∀ jj : Fin 64, (rowCtx d init c.toReal bh (128 * sm + r) X).c *
          (rowCtx d init c.toReal bh (128 * sm + r) X).bv (64 * (2 * sm + t) + jj) ≤ 1536 := fun jj => by
        show c.toReal * (scoreMag d init bh (128 * sm + r) (64 * (2 * sm + t) + jj) : ℝ) ≤ 1536
        have h1 := hH.2.2 _ _ hnr (hkey jj)
        have h1' : (sigma d : ℝ) * (scoreMag d init bh (128 * sm + r) (64 * (2 * sm + t) + jj) : ℝ) ≤ 2 ^ 10 := by
          exact_mod_cast h1
        have h2 : (0 : ℝ) ≤ (scoreMag d init bh (128 * sm + r) (64 * (2 * sm + t) + jj) : ℝ) := by
          exact_mod_cast scoreMag_nonneg _ _ _ _ _
        have := mul_le_mul_of_nonneg_right hc32 h2
        nlinarith
      have h0 : 2 * sm + t = 0 → (rowCtx d init c.toReal bh (128 * sm + r) X).al (64 * (2 * sm + t)) = true := by
        intro h
        show alOf d (128 * sm + r) (64 * (2 * sm + t)) = true
        rw [h]; unfold alOf; simp; omega
      exact row_step_masked (t := 2 * sm + t) (S := fun jj => S.get2 r jj)
        (w := fun jj => (((start_n + offs_n.row 128).le (offs_m.col 64)).where_ (bc (FloatV.litF32 (FloatV.ofPat 0)))
          (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get2 r jj)
        (x' := fun jj => x'.get2 r jj) (x := fun jj => x.get2 r jj) (p := fun jj => p.get2 r jj)
        (q := fun jj => q16.get2 r jj)
        (vt := fun jj e => vv.get2 jj e) (acc := fun e => sacc.get2 r e) (a1 := fun e => a1.get2 r e)
        (acc' := fun e => acc'.get2 r e) (mx := mx.get1 r) (m' := mij.get1 r)
        (dm := dm.get1 r) (al := al.get1 r) (lij := lij.get1 r) (l' := l'.get1 r)
        hexp hni hRok (by omega) (veOf_le hH) hcf rfl
        (fun jj => qk_lane hdot hd hnr (hkey jj) hH hS r jj (hq r) (hkr jj))
        hglob hw (fun jj => hx' (r, jj, ())) (hmx r) (hmij (r, ())) (fun jj => hx (r, jj, ()))
        (fun jj => hp (r, jj, ())) (hdm (r, ())) (hal (r, ())) (hlij r)
        (fun jj => hq16 (r, jj, ())) (fun e => ha1 (r, e, ())) hvt
        (fun e => pv_lane hdot hacc' r e (fun k => hvf k e)) (hl' (r, ())) h0 (hinv' r hH X hX hXk)
  · rw [htc]
    rintro st m₁ ⟨hm₁, -, -, hst⟩
    subst m₁
    exact hk st hst

/-! ## The three calls of `_attn_fwd_inner` -/

/-- The common hypotheses of the calls: the program's tile `(sm, bh)`, the values its arguments
hold. -/
structure CallOk (d : Inst) (init : Mem) (sm bh : ℕ) (c : F32) (q : Tensor [128, d.HEAD_DIM] F16)
    (offset_y start_m N_CTX : I32) (offs_m : Tensor [128] I32) (offs_n : Tensor [64] I32) : Prop where
  sm_le : 128 * sm + 128 ≤ d.N_CTX
  bh_lt : bh < d.Z * d.H
  c_ok : CFacts d c
  q_ok : QReads d init sm bh q
  oy : offset_y.toInt = ((bh * d.N_CTX : ℕ) : ℤ)
  start : start_m.toInt = ((sm : ℕ) : ℤ)
  n : N_CTX.toInt = ((d.N_CTX : ℕ) : ℤ)
  om : ∀ r : Fin 128, (offs_m.get1 r).toInt = ((128 * sm + r : ℕ) : ℤ)
  on : ∀ jj : Fin 64, (offs_n.get1 jj).toInt = ((jj : ℕ) : ℤ)

/-- **`_attn_fwd_inner` with `STAGE = 1`** (the causal launch's first call, line 228): the
blocks before the program's query rows. -/
theorem safe_inner1 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotF16 P) (hdtot : Rules.Num.DotTotal P cfg16) (hexp : Rules.Num.Exp2F32 P)
    (hni : Rules.Num.Exp2F32NegInf P) (hext : Rules.Num.MathTotal P .exp2 .binary32)
    {R W : Set Int} {α : Type} {Q : α → Mem → Prop} (hR : Rd d ⊆ R) {sm bh : ℕ} {c : F32}
    {q : Tensor [128, d.HEAD_DIM] F16} {offset_y start_m N_CTX : I32} {offs_m : Tensor [128] I32}
    {offs_n : Tensor [64] I32} (hcall : CallOk d init sm bh c q offset_y start_m N_CTX offs_m offs_n)
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32} (hinv : StInv d init c sm bh 0 acc l m)
    {k : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 → Prog α}
    (hk : ∀ acc' l' m', StInv d init c sm bh (2 * sm) acc' l' m' →
      Safe ((launch d).env P) R W Q init (k (acc', l', m'))) :
    Safe ((launch d).env P) R W Q init
      (attn_fwd_inner (BLOCK_M := 128) (HEAD_DIM := d.HEAD_DIM) (BLOCK_N := 64) acc l m q
        (descOf d d.k 64) (descOf d d.v 64) offset_y start_m c 1 offs_m offs_n N_CTX >>= k) := by
  simp only [attn_fwd_inner]
  simp only [show (1 : Nat) = 2 ↔ False by decide, ite_false, ite_true, bind_assoc, pure_bind]
  show Safe _ R W Q init (tl.forRange (IntV.ofInt 0) (start_m * IntV.ofNat 128) (IntV.ofNat 64)
    (acc, l, m, offset_y + IntV.ofInt 0, offset_y + IntV.ofInt 0)
    (bodyPlain d.HEAD_DIM q (descOf d d.k 64) (descOf d d.v 64) c) >>= fun x => k (x.1, x.2.1, x.2.2.1))
  have hsm := hcall.sm_le
  have hN := hd.n_le
  refine safe_loop_plain (t0 := 0) (n := 2 * sm) hd hpre hdot hdtot hexp hni hext hR hsm hcall.bh_lt hcall.c_ok
    hcall.q_ok (by rw [i32_zero]; rfl) ?_ (by omega) ?_ hcall.oy hinv fun st hst => hk _ _ _ (by simpa using hst)
  · rw [IntV.I32.mul_nat hcall.start (i32_nat (by norm_num)) (by omega)]; congr 1; ring
  · intro r j hj
    unfold alOf; simp only [decide_eq_true_eq]; omega

/-- **`_attn_fwd_inner` with `STAGE = 3`** (the non-causal launch's call, line 228): every key. -/
theorem safe_inner3 {d : Inst} (hd : d.ok) (hcausal : d.causal = false) {init : Mem} (hpre : Pre d init)
    {P : Platform} (hdot : Rules.Num.DotF16 P) (hdtot : Rules.Num.DotTotal P cfg16)
    (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hext : Rules.Num.MathTotal P .exp2 .binary32)
    {R W : Set Int} {α : Type} {Q : α → Mem → Prop} (hR : Rd d ⊆ R) {sm bh : ℕ} {c : F32}
    {q : Tensor [128, d.HEAD_DIM] F16} {offset_y start_m N_CTX : I32} {offs_m : Tensor [128] I32}
    {offs_n : Tensor [64] I32} (hcall : CallOk d init sm bh c q offset_y start_m N_CTX offs_m offs_n)
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32} (hinv : StInv d init c sm bh 0 acc l m)
    {k : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 → Prog α}
    (hk : ∀ acc' l' m', StInv d init c sm bh (d.N_CTX / 64) acc' l' m' →
      Safe ((launch d).env P) R W Q init (k (acc', l', m'))) :
    Safe ((launch d).env P) R W Q init
      (attn_fwd_inner (BLOCK_M := 128) (HEAD_DIM := d.HEAD_DIM) (BLOCK_N := 64) acc l m q
        (descOf d d.k 64) (descOf d d.v 64) offset_y start_m c 3 offs_m offs_n N_CTX >>= k) := by
  simp only [attn_fwd_inner]
  simp only [show (3 : Nat) = 1 ↔ False by decide, show (3 : Nat) = 2 ↔ False by decide, ite_false,
    bind_assoc, pure_bind]
  show Safe _ R W Q init (tl.forRange (IntV.ofInt 0) N_CTX (IntV.ofNat 64)
    (acc, l, m, offset_y + IntV.ofInt 0, offset_y + IntV.ofInt 0)
    (bodyPlain d.HEAD_DIM q (descOf d d.k 64) (descOf d d.v 64) c) >>= fun x => k (x.1, x.2.1, x.2.2.1))
  have hNm := hd.n_mul
  refine safe_loop_plain (t0 := 0) (n := d.N_CTX / 64) hd hpre hdot hdtot hexp hni hext hR hcall.sm_le
    hcall.bh_lt hcall.c_ok hcall.q_ok (by rw [i32_zero]; rfl) ?_ (by omega) ?_ hcall.oy hinv
    fun st hst => hk _ _ _ (by simpa using hst)
  · rw [hcall.n]; congr 1; omega
  · intro r j hj
    unfold alOf; rw [hcausal]; simp only [decide_eq_true_eq, true_or, and_true]; omega

/-- **`_attn_fwd_inner` with `STAGE = 2`** (the causal launch's second call, line 236): the
program's own blocks, under the mask. -/
theorem safe_inner2 {d : Inst} (hd : d.ok) (hcausal : d.causal = true) {init : Mem} (hpre : Pre d init)
    {P : Platform} (hdot : Rules.Num.DotF16 P) (hdtot : Rules.Num.DotTotal P cfg16)
    (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hext : Rules.Num.MathTotal P .exp2 .binary32)
    {R W : Set Int} {α : Type} {Q : α → Mem → Prop} (hR : Rd d ⊆ R) {sm bh : ℕ} {c : F32}
    {q : Tensor [128, d.HEAD_DIM] F16} {offset_y start_m N_CTX : I32} {offs_m : Tensor [128] I32}
    {offs_n : Tensor [64] I32} (hcall : CallOk d init sm bh c q offset_y start_m N_CTX offs_m offs_n)
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32}
    (hinv : StInv d init c sm bh (2 * sm) acc l m)
    {k : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 → Prog α}
    (hk : ∀ acc' l' m', StInv d init c sm bh (2 * sm + 2) acc' l' m' →
      Safe ((launch d).env P) R W Q init (k (acc', l', m'))) :
    Safe ((launch d).env P) R W Q init
      (attn_fwd_inner (BLOCK_M := 128) (HEAD_DIM := d.HEAD_DIM) (BLOCK_N := 64) acc l m q
        (descOf d d.k 64) (descOf d d.v 64) offset_y start_m c 2 offs_m offs_n N_CTX >>= k) := by
  simp only [attn_fwd_inner]
  simp only [show (2 : Nat) = 1 ↔ False by decide, ite_false, ite_true, bind_assoc, pure_bind]
  have hsm := hcall.sm_le
  have hN := hd.n_le
  have hlo : (start_m * IntV.ofNat 128 : I32).toInt = ((64 * (2 * sm) : ℕ) : ℤ) := by
    rw [IntV.I32.mul_nat hcall.start (i32_nat (by norm_num)) (by omega)]; congr 1; ring
  have hdv : ((128 : ℕ) : ℤ) ∣ HintVal.val (start_m * IntV.ofNat 128 : I32) := by
    show ((128 : ℕ) : ℤ) ∣ (start_m * IntV.ofNat 128 : I32).toInt
    rw [hlo]; exact ⟨sm, by push_cast; ring⟩
  refine safe_multiple_of1 hdv ?_
  show Safe _ R W Q init (tl.forRange (start_m * IntV.ofNat 128) ((start_m + IntV.ofInt 1) * IntV.ofNat 128)
    (IntV.ofNat 64) (acc, l, m, offset_y + start_m * IntV.ofNat 128, offset_y + start_m * IntV.ofNat 128)
    (bodyMasked d.HEAD_DIM q (descOf d d.k 64) (descOf d d.v 64) c offs_m offs_n) >>=
      fun x => k (x.1, x.2.1, x.2.2.1))
  refine safe_loop_masked hd hcausal hpre hdot hdtot hexp hni hext hR hsm hcall.bh_lt hcall.c_ok
    hcall.q_ok hcall.om hcall.on hlo ?_ hcall.oy hinv fun st hst => hk _ _ _ hst
  have h1 : (start_m + IntV.ofInt 1 : I32).toInt = ((sm + 1 : ℕ) : ℤ) :=
    IntV.I32.add_nat hcall.start i32_one (by omega)
  rw [IntV.I32.mul_nat h1 (i32_nat (by norm_num)) (by omega)]; congr 1; ring

end Kernels.AttnFwd
