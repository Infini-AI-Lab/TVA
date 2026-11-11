import Claim
import TVA.Proof
import Proof.GateDefs
import Proof.Lanes
import Proof.LoopSafe
import Proof.NumDefs
import Proof.GammaIdx

/-!
# The gates

`safe_gateG` (`if USE_G`, lines 160–165) and `safe_gateGamma` (`if USE_G_GAMMA`, lines 166–170):
the gate is computed with `tl.exp2` and multiplies `b_o` and `b_A`.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {d : Inst} {g : Geo} {r : Nat} {init : Mem}

/-- Row `i` of `g[token, head]`: the gate exponent, or `+0` past the sequence. -/
def gTile (d : Inst) (g : Geo) (i : Nat) : F32 :=
  if g.rowOk i then d.gv ((g.lo + i) * d.HV + g.ih) else FloatV.ofPat 0

/-- **The load of `b_g`** (line 163). -/
theorem safe_load_g (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r) (hG : d.useG = true)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init) {i_h bos : I64}
    (hih : i_h.toInt = (g.ih : Int)) (hbos : bos.toInt = (g.bos : Int))
    {k' : Tensor [BT] F32 → Prog α}
    (hk : ∀ bg : Tensor [BT] F32, (∀ i : Fin BT, bg.get1 i = gTile d g i) →
      Safe E (Rs d) W Q m (k' bg)) :
    Safe E (Rs d) W Q m (tl.load (d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) +
        o_t * (IntV.ofNat d.HV : I64)) (some (o_t.ltS T)) (some (bc zf32)) >>= k') := by
  subst hm
  obtain ⟨hgv, hgh⟩ := hpre.g_valid hG
  have hHV := HV_lt16 hd
  have hBT := BT_le hd
  have hend := hg.end_le
  have hHV' : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) :=
    IntV.toInt_ofNat (w := 64) (by omega)
  have hbH : (bos * (IntV.ofNat d.HV : I64)).toInt = ((g.bos * d.HV : Nat) : Int) :=
    IntV.mul_nat hbos hHV' (by
      have : g.bos * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
      omega)
  have hsum : (bos * (IntV.ofNat d.HV : I64) + i_h).toInt = ((g.bos * d.HV + g.ih : Nat) : Int) :=
    IntV.add_nat hbH hih (by
      have : g.bos * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
      have := hg.ih_lt
      omega)
  have hlane : ∀ i : Fin BT, g.rowOk i →
      ((o_t * (IntV.ofNat d.HV : I64)).get1 i).toInt = (((g.it * BT + i) * d.HV : Nat) : Int) := by
    intro i hi
    rw [Tensor.get1_muls]
    refine IntV.mul_nat (hl.ot i) hHV' ?_
    have h1 : g.it * BT + i < 2 ^ 35 := by unfold Geo.rowOk at hi; omega
    have : (g.it * BT + i) * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
    omega
  have hptr : ∀ i : Idx [BT], decide (g.rowOk i.1) = true →
      (d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) + o_t * (IntV.ofNat d.HV : I64)).get i =
        d.g.ptr (((g.lo + i.1) * d.HV + g.ih : Nat) : Int) := by
    rintro ⟨i, ⟨⟩⟩ hi
    have hi' : g.rowOk i := of_decide_eq_true hi
    show ((d.g.ptr 0).offset (bos * (IntV.ofNat d.HV : I64) + i_h).toInt).offset
      ((o_t * (IntV.ofNat d.HV : I64)).get1 i).toInt = _
    rw [hsum, hlane i hi', Buf.ptr_offset, Buf.ptr_offset]
    congr 1
    unfold Geo.lo; push_cast; ring
  have hlt : ∀ i : Idx [BT], decide (g.rowOk i.1) = true → (g.lo + i.1) * d.HV + g.ih < d.g.len := by
    rintro ⟨i, ⟨⟩⟩ hi
    have hi' : g.rowOk i := of_decide_eq_true hi
    have hl' := hd.g_len hG
    have h0 : g.it * 64 + i < g.Tn := hi'
    have hlo : g.lo = g.bos + 64 * g.it := rfl
    have h1 : g.lo + i + 1 ≤ d.B * d.T := by have := hg.end_le; omega
    have h2 : (g.lo + i + 1) * d.HV ≤ d.B * d.T * d.HV := Nat.mul_le_mul_right _ h1
    have h3 : (g.lo + i + 1) * d.HV = (g.lo + i) * d.HV + d.HV := by ring
    have := hg.ih_lt
    show (g.lo + i) * d.HV + g.ih < d.g.len
    omega
  refine safe_load_buf d.g (Buf.addrs_of_get d.g (fun i => ?_) hptr) hlt hgv ?_ fun bg h1 h2 => ?_
  · rcases i with ⟨i, ⟨⟩⟩
    show (o_t.ltS T).get1 i = _
    rw [mt_get hl]; rfl
  · exact (Buf.elems_sub_bytes d.g (fun j hj => by
      obtain ⟨⟨i, ⟨⟩⟩, hi, rfl⟩ := hj
      exact hlt _ hi)).trans (fun x hx => Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hG, hx⟩)))))
  · refine hk bg fun i => ?_
    by_cases hi : g.rowOk i
    · have := h1 (i, ()) (by simpa using hi)
      rw [hgh _ (hlt (i, ()) (by simpa using hi))] at this
      unfold gTile; simp only [hi, ↓reduceIte]
      exact (Option.some.inj this).symm
    · have := h2 (i, ()) (by simpa using hi) _ rfl
      unfold gTile; simp only [hi, ↓reduceIte]
      exact this

/-! ## The loop's result in the gates' form -/

theorem nK_mul_le {d : Inst} (hd : d.ok) : nK d * d.bk ≤ 256 := by
  have := hd.K_le
  unfold nK
  rcases hd.bk_ok with h | h | h <;> rw [h] <;> omega

/-- The loop's result for the inter-chunk dot, as `QS`, `PQS` with the constants `cD`, `2^-116`. -/
theorem oinv_cD (hd : d.ok) {bo : Tensor [BT, d.bk] F32} (hO : OInv d g r (nK d) bo)
    {i : Fin BT} {c : Fin d.bk} (hrow : g.rowOk i) (hcol : g.colOk d c)
    (hok : OkO d r (g.lo + i) g.ih (g.iv * d.bk + c)) :
    (bo.get2 i c).isFinite = true ∧
      |(bo.get2 i c).toReal - (QS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ)| ≤
        33 / 500000 * (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) + 2 ^ (-116 : ℤ) := by
  obtain ⟨hf, hb⟩ := hO i c hrow hcol hok
  rw [SQ_eq_QS _ _ _ _ (K_le_nK hd), MQ_eq_PQS _ _ _ _ (K_le_nK hd)] at hb
  have hP : 0 ≤ PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) :=
    Finset.sum_nonneg fun x _ => mul_nonneg (L_nonneg _) (L_nonneg _)
  have h2 := hb.trans (loop_to_cD hd.bk_ok (nK_mul_le hd) hP)
  refine ⟨hf, ?_⟩
  have h3 : ((|(bo.get2 i c).toRat - QS d r (g.lo + i) g.ih (g.iv * d.bk + c)| : ℚ) : ℝ) ≤
      ((33 / 500000 * PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) + 2 ^ (-116 : ℤ) : ℚ) : ℝ) := by
    exact_mod_cast h2
  push_cast at h3
  simpa [FloatV.toReal] using h3

/-- The loop's result for a score, as `Aqk`, `PA` with the constants `cD`, `2^-116`. -/
theorem ainv_cD (hd : d.ok) {bA : Tensor [BT, BT] F32} (hA : AInv d g (nK d) bA)
    {i i' : Fin BT} (hrow : g.rowOk i) (hle : i' ≤ i)
    (hok : OkA d (g.lo + i) (g.lo + i') g.ih) :
    (bA.get2 i i').isFinite = true ∧
      |(bA.get2 i i').toReal - (Aqk d (g.lo + i) (g.lo + i') g.ih : ℝ)| ≤
        33 / 500000 * (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) + 2 ^ (-116 : ℤ) := by
  obtain ⟨hf, hb⟩ := hA i i' hrow hle hok
  rw [SA_eq_Aqk _ _ _ (K_le_nK hd), MA_eq_PA _ _ _ (K_le_nK hd)] at hb
  have hP : 0 ≤ PA d (g.lo + i) (g.lo + i') g.ih :=
    Finset.sum_nonneg fun x _ => mul_nonneg (L_nonneg _) (L_nonneg _)
  have h2 := hb.trans (loop_to_cD hd.bk_ok (nK_mul_le hd) hP)
  refine ⟨hf, ?_⟩
  have h3 : ((|(bA.get2 i i').toRat - Aqk d (g.lo + i) (g.lo + i') g.ih| : ℚ) : ℝ) ≤
      ((33 / 500000 * PA d (g.lo + i) (g.lo + i') g.ih + 2 ^ (-116 : ℤ) : ℚ) : ℝ) := by
    exact_mod_cast h2
  push_cast at h3
  simpa [FloatV.toReal] using h3

/-- For a valid instance, `useG` being set forces `useGamma` to be false. -/
theorem useGamma_false_of_useG (hd : d.ok) (hG : d.useG = true) : d.useGamma = false := by
  have := hd.one_gate
  cases h : d.useGamma
  · rfl
  · exact absurd ⟨hG, h⟩ this

/-- When `useGamma` is false, the gamma error `gamErr` is zero. -/
theorem gamErr_zero {j : Nat} (h : d.useGamma = false) : gamErr d j = 0 := by
  simp [gamErr, h]

/-- When `useG` is set, the gate is `2` to the power `gAt d t j`. -/
theorem gate_useG {r t j : Nat} (hG : d.useG = true) :
    gate d r t j = (2 : ℝ) ^ ((gAt d t j : ℚ) : ℝ) := by
  simp [gate, hG]

/-- When `useG` is set, the weight is `2` to the power of the difference `gAt d t j - gAt d s j`. -/
theorem wgt_useG {r t s j : Nat} (hG : d.useG = true) :
    wgt d r t s j = (2 : ℝ) ^ (((gAt d t j : ℚ) : ℝ) - ((gAt d s j : ℚ) : ℝ)) := by
  simp [wgt, hG]

/-- `|QS|` is at most `PQS`. -/
theorem abs_QS_le (d : Inst) (r t j e : Nat) : |QS d r t j e| ≤ PQS d r t j e := by
  have := abs_SQ_le d r t j e d.K
  rwa [SQ_eq_QS _ _ _ _ le_rfl, MQ_eq_PQS _ _ _ _ le_rfl] at this

/-- `|Aqk|` is at most `PA`. -/
theorem abs_Aqk_le (d : Inst) (t s j : Nat) : |Aqk d t s j| ≤ PA d t s j := by
  have := abs_SA_le d t s j d.K
  rwa [SA_eq_Aqk _ _ _ le_rfl, MA_eq_PA _ _ _ le_rfl] at this

/-- `PQS` is nonnegative. -/
theorem PQS_nonneg (d : Inst) (r t j e : Nat) : 0 ≤ PQS d r t j e :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (L_nonneg _) (L_nonneg _)

/-- `PA` is nonnegative. -/
theorem PA_nonneg (d : Inst) (t s j : Nat) : 0 ≤ PA d t s j :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (L_nonneg _) (L_nonneg _)

/-- A rounded difference of a gate exponent stays below `65`. -/
theorem r_le_65 {gt gs r : ℝ} (hd : gt - gs ≤ 64)
    (hr : |r - (gt - gs)| ≤ 2 ^ (-24 : ℤ) * |gt - gs| + 2 ^ (-150 : ℤ)) : r ≤ 65 := by
  have h1 := (abs_le.mp hr).2
  have h2 : (2 : ℝ) ^ (-24 : ℤ) ≤ 1 / 1000 := by norm_num
  have h3 : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 / 100 := by norm_num
  rcases le_or_gt 0 (gt - gs) with h | h
  · rw [abs_of_nonneg h] at h1
    have := mul_le_mul_of_nonneg_right h2 h
    generalize (2:ℝ)^(-24:ℤ) = A at *
    generalize (2:ℝ)^(-150:ℤ) = B at *
    generalize gt - gs = x at *
    linarith
  · rw [abs_of_neg h] at h1
    have := mul_le_mul_of_nonneg_right h2 (neg_nonneg.mpr h.le)
    generalize (2:ℝ)^(-24:ℤ) = A at *
    generalize (2:ℝ)^(-150:ℤ) = B at *
    generalize gt - gs = x at *
    linarith

/-- The exponent facts of row `i` with `g` gates. -/
theorem grow (_hd : d.ok) (hg : Geo.Ok d g r) (hG : d.useG = true) {i : Nat} (hrow : g.rowOk i)
    (hr : RowIn d r (g.lo + i) g.ih) :
    (gTile d g i).isFinite = true ∧ (gTile d g i).toReal = ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) ∧
      ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) ≤ 64 ∧ |((gAt d (g.lo + i) g.ih : ℚ) : ℝ)| ≤ 2 ^ 64 := by
  have hs : g.lo + i ∈ upTo d r (g.lo + i) := by
    simp only [upTo, Finset.mem_Icc, hg.lo_eq]; omega
  obtain ⟨h64, hall⟩ := hr.2.2.2.1 hG
  obtain ⟨hf, habs, -⟩ := hall _ hs
  have : gTile d g i = d.gv ((g.lo + i) * d.HV + g.ih) := by unfold gTile; simp [hrow]
  refine ⟨by rw [this]; exact hf, by rw [this]; rfl, by exact_mod_cast h64, by exact_mod_cast habs⟩

/-- The exponent facts of an earlier key `i' ≤ i` with `g` gates. -/
theorem gcol (hg : Geo.Ok d g r) (hG : d.useG = true) {i i' : Nat} (hrow' : g.rowOk i')
    (hle : i' ≤ i) (hr : RowIn d r (g.lo + i) g.ih) :
    (gTile d g i').isFinite = true ∧ (gTile d g i').toReal = ((gAt d (g.lo + i') g.ih : ℚ) : ℝ) ∧
      |((gAt d (g.lo + i') g.ih : ℚ) : ℝ)| ≤ 2 ^ 64 ∧
      ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) - ((gAt d (g.lo + i') g.ih : ℚ) : ℝ) ≤ 64 := by
  have hs : g.lo + i' ∈ upTo d r (g.lo + i) := by
    simp only [upTo, Finset.mem_Icc, hg.lo_eq]; omega
  obtain ⟨-, hall⟩ := hr.2.2.2.1 hG
  obtain ⟨hf, habs, hdiff⟩ := hall _ hs
  have : gTile d g i' = d.gv ((g.lo + i') * d.HV + g.ih) := by unfold gTile; simp [hrow']
  refine ⟨by rw [this]; exact hf, by rw [this]; rfl, ?_, ?_⟩
  · exact_mod_cast habs
  · exact_mod_cast hdiff

/-- **The `USE_G` block** (lines 160–165). -/
theorem safe_gateG (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init) {i_h bos : I64}
    (hih : i_h.toInt = (g.ih : Int)) (hbos : bos.toInt = (g.bos : Int))
    {bo : Tensor [BT, d.bk] F32} {bA : Tensor [BT, BT] F32}
    (hO : OInv d g r (nK d) bo) (hA : AInv d g (nK d) bA)
    {k' : Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32 → Prog α}
    (hk : ∀ bo' bA', Gated1 d g r bo' bA' → Safe E (Rs d) W Q m (k' (bo', bA'))) :
    Safe E (Rs d) W Q m (gateG d o_t (o_t.ltS T) i_h bos bo bA >>= k') := by
  by_cases hG : d.useG = true
  · simp only [gateG, hG, ↓reduceIte, Prog.bind_eq, Prog.pure_eq, Prog.bind_assoc, Prog.ret_bind]
    refine safe_load_g hd hpre hg hG hl hm hih hbos fun bg hbg => ?_
    refine safe_math (fun i => hP.mathTotal .exp2 .binary32 _) fun Gh hGh => ?_
    refine safe_mul fun bo' hbo' => ?_
    refine safe_sub fun dd hdd => ?_
    refine safe_math (fun i => hP.mathTotal .exp2 .binary32 _) fun Wh hWh => ?_
    refine safe_mul fun bA' hbA' => ?_
    have hGm : d.useGamma = false := useGamma_false_of_useG hd hG
    refine hk bo' bA' ⟨fun _ => ⟨?_, ?_⟩, fun h => by simp [h] at hG⟩
    · intro i c hrow hcol hin
      have hr := hin.rowIn
      obtain ⟨hgf, hgv, hg64, hgabs⟩ := grow hd hg hG hrow hr
      have hbg1 := hbg i
      have hex := hGh (i, ())
      rw [show bg.get (i, ()) = gTile d g i from hbg1] at hex
      obtain ⟨hGf, hGb⟩ := exp2_fact hP hex hgf (by rw [hgv]; linarith)
      rw [hgv] at hGb
      obtain ⟨hobF, hob⟩ := oinv_cD hd hO hrow hcol hin.okO
      have hmul := hbo' (i, c, ())
      rw [gate_useG hG, gamErr_zero hGm]
      have hG0 : (0:ℝ) ≤ (2:ℝ) ^ ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) := by positivity
      have hGP : gate d r (g.lo + i) g.ih * (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) ≤ 2 ^ 100 :=
        hin.2.2.2.2.2.2.1
      rw [gate_useG hG] at hGP
      have hP100 : (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) ≤ 2 ^ 100 := by
        exact_mod_cast hin.2.2.2.2.2.1
      have hG64 : (2:ℝ) ^ ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) ≤ 2 ^ 64 := by
        calc (2:ℝ) ^ ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
              Real.rpow_le_rpow_of_exponent_le (by norm_num) (by push_cast; exact hg64)
          _ = 2 ^ 64 := by norm_num
      have hQ : |(QS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ)| ≤
          (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) := by
        have := abs_QS_le d r (g.lo + i) g.ih (g.iv * d.bk + c)
        exact_mod_cast this
      have hP0 : (0:ℝ) ≤ (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) := by
        exact_mod_cast PQS_nonneg d r (g.lo + i) g.ih (g.iv * d.bk + c)
      refine og_lane hP0 hQ hG0 hG64 hGP hP100 (le_refl _)
        (by norm_num) hobF hob hGf ?_ hmul
      have h22 : (2:ℝ) ^ (-22:ℤ) * (2:ℝ) ^ ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) ≤
          (2 ^ (-21:ℤ) + 0 / 2) * (2:ℝ) ^ ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) :=
        mul_le_mul_of_nonneg_right (by norm_num) hG0
      exact hGb.trans (by gcongr)
    · intro i i' hrow hle hr
      have hrow' : g.rowOk i' := by
        have : (i' : Nat) ≤ i := hle
        unfold Geo.rowOk at *; omega
      have hs : g.lo + i' ∈ upTo d r (g.lo + i) := by
        simp only [upTo, Finset.mem_Icc, hg.lo_eq]
        have : (i' : Nat) ≤ i := hle
        omega
      obtain ⟨hgf, hgv, hg64, hgabs⟩ := grow hd hg hG hrow hr
      obtain ⟨hgf', hgv', habs', hdiff⟩ := gcol hg hG hrow' hle hr
      have hdd' : FloatV.NaNEq (FloatV.sub (bg.get1 i) (bg.get1 i')) (dd.get (i, i', ())) :=
        hdd (i, i', ())
      rw [hbg i, hbg i'] at hdd'
      have hroom : |(gTile d g i).toReal - (gTile d g i').toReal| ≤ 2 ^ 105 := by
        rw [hgv, hgv']
        have h1 := abs_le.mp hgabs
        have h2 := abs_le.mp habs'
        rw [abs_le]
        constructor <;> nlinarith [(by norm_num : (2:ℝ) ^ 64 + 2 ^ 64 ≤ 2 ^ 105)]
      obtain ⟨hddF, hddb⟩ := sub32_lane hgf hgf' hroom hdd'
      rw [hgv, hgv'] at hddb
      have h65 := r_le_65 hdiff hddb
      have hex := hWh (i, i', ())
      obtain ⟨hWf, hWb⟩ := exp2_fact hP hex hddF (by linarith)
      obtain ⟨-, hwb⟩ := w_g_num hg64 habs' hdiff hddb hWb
      obtain ⟨habF, hab⟩ := ainv_cD hd hA hrow hle (hr.okA hs)
      have hmul := hbA' (i, i', ())
      have hw : wgt d r (g.lo + i) (g.lo + i') g.ih =
          (2:ℝ) ^ (((gAt d (g.lo + i) g.ih : ℚ) : ℝ) - ((gAt d (g.lo + i') g.ih : ℚ) : ℝ)) :=
        wgt_useG hG
      obtain ⟨hPA100, hwPA⟩ := hr.2.2.2.2 _ hs
      rw [hw, gamErr_zero hGm]
      rw [hw] at hwPA
      have hw0 : (0:ℝ) ≤ (2:ℝ) ^ (((gAt d (g.lo + i) g.ih : ℚ) : ℝ) -
          ((gAt d (g.lo + i') g.ih : ℚ) : ℝ)) := by positivity
      have hw64 : (2:ℝ) ^ (((gAt d (g.lo + i) g.ih : ℚ) : ℝ) -
          ((gAt d (g.lo + i') g.ih : ℚ) : ℝ)) ≤ 2 ^ 64 := by
        calc _ ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
              Real.rpow_le_rpow_of_exponent_le (by norm_num) (by push_cast; exact hdiff)
          _ = 2 ^ 64 := by norm_num
      have hPA0 : (0:ℝ) ≤ (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) := by
        exact_mod_cast PA_nonneg d _ _ _
      have hA' : |(Aqk d (g.lo + i) (g.lo + i') g.ih : ℝ)| ≤ (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) := by
        exact_mod_cast abs_Aqk_le d _ _ _
      refine ag_lane hPA0 hA' hw0 hw64 hwPA (by exact_mod_cast hPA100) (le_refl _) (by norm_num)
        habF hab hWf ?_ hmul
      have h16 : (2:ℝ) ^ (-16:ℤ) ≤ 2 ^ (-16:ℤ) + 72 / 100 * 0 := by norm_num
      simpa using hwb
  · have hG' : d.useG = false := by simpa using hG
    simp only [gateG, hG', Bool.false_eq_true, ↓reduceIte, Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
    exact hk bo bA ⟨fun h => absurd h hG, fun _ => ⟨hO, hA⟩⟩

/-! ## The `g_gamma` gate -/

theorem gate_gamma {r t j : Nat} (hG : d.useG = false) (hGm : d.useGamma = true) :
    gate d r t j = (2 : ℝ) ^ ((gam d j * ((t - rowLo d r : Nat) + 1 : ℚ) : ℚ) : ℝ) := by
  simp [gate, hG, hGm]

/-- When only `useGamma` is set, the weight is `2` to the power `gam d j * (t - s)`. -/
theorem wgt_gamma {r t s j : Nat} (hG : d.useG = false) (hGm : d.useGamma = true) :
    wgt d r t s j = (2 : ℝ) ^ ((gam d j * ((t : ℚ) - s) : ℚ) : ℝ) := by
  simp [wgt, hG, hGm]

/-- A rounded `γ·n` stays below `65` when `γ·n ≤ 64`. -/
theorem x_le_65 {x r : ℝ} (hx : x ≤ 64)
    (hr : |r - x| ≤ 2 ^ (-24 : ℤ) * |x| + 2 ^ (-150 : ℤ)) : r ≤ 65 := by
  have := r_le_65 (gt := x) (gs := 0) (r := r) (by simpa using hx) (by simpa using hr)
  exact this

/-- `γ·n ≤ 64` for `γ ≤ 1`, `0 ≤ n ≤ 64`. -/
theorem gn_le {γ n : ℝ} (hγ1 : γ ≤ 1) (hn0 : 0 ≤ n) (hn : n ≤ 64) : γ * n ≤ 64 := by
  nlinarith [mul_nonneg (sub_nonneg.2 hγ1) hn0]

/-- `|γ·n| ≤ 4096`. -/
theorem gn_abs {γ n : ℝ} (hγ : |γ| ≤ 64) (hn0 : 0 ≤ n) (hn : n ≤ 64) : |γ * n| ≤ 4096 := by
  rw [abs_mul, abs_of_nonneg hn0]
  nlinarith [abs_nonneg γ]

/-- A rounded `γ·n` is below `2^13` in magnitude. -/
theorem bg_abs {γ n bg : ℝ} (hγ : |γ| ≤ 64) (hn0 : 0 ≤ n) (hn : n ≤ 64)
    (hbg : |bg - γ * n| ≤ 2 ^ (-24 : ℤ) * |γ * n| + 2 ^ (-150 : ℤ)) : |bg| ≤ 2 ^ 13 := by
  have h1 := gn_abs hγ hn0 hn
  have h2 : (2:ℝ) ^ (-24 : ℤ) ≤ 1 / 1000 := by norm_num
  have h3 : (2:ℝ) ^ (-150 : ℤ) ≤ 1 / 100 := by norm_num
  have h4 := abs_sub_abs_le_abs_sub bg (γ * n)
  have h5 := mul_le_mul_of_nonneg_right h2 (abs_nonneg (γ * n))
  generalize (2:ℝ) ^ (-24 : ℤ) = A at *
  generalize (2:ℝ) ^ (-150 : ℤ) = B at *
  generalize |γ * n| = x at *
  have : |bg| ≤ 4200 := by nlinarith
  linarith

/-- The fused difference of two gate exponents stays below `65`. -/
theorem gam_dd_le {γ n n' bi bj r : ℝ} (hγ : |γ| ≤ 64) (hγ1 : γ ≤ 1) (hn' : 1 ≤ n')
    (hnn : n' ≤ n) (hn : n ≤ 64)
    (hbi : |bi - γ * n| ≤ 2 ^ (-24 : ℤ) * |γ * n| + 2 ^ (-150 : ℤ))
    (hbj : |bj - γ * n'| ≤ 2 ^ (-24 : ℤ) * |γ * n'| + 2 ^ (-150 : ℤ))
    (hr : |r - (bi + -bj)| ≤ 3 * 2 ^ (-24 : ℤ) * (|bi| + |-bj|) + 4 * 2 ^ (-150 : ℤ)) : r ≤ 65 := by
  have hbi' := bg_abs hγ (by linarith) hn hbi
  have hbj' := bg_abs hγ (by linarith) (by linarith) hbj
  have h1 := gn_abs hγ (by linarith) hn
  have h1' := gn_abs hγ (by linarith) (by linarith : n' ≤ 64)
  have h6 : γ * (n - n') ≤ 63 := by
    nlinarith [mul_nonneg (sub_nonneg.2 hγ1) (sub_nonneg.2 hnn)]
  have h2 : (2:ℝ) ^ (-24 : ℤ) ≤ 1 / 10000000 := by norm_num
  have h3 : (2:ℝ) ^ (-150 : ℤ) ≤ 1 / 10000000 := by norm_num
  rw [abs_neg] at hr
  have e1 := (abs_le.mp hbi).2
  have e2 := (abs_le.mp hbj).1
  have e3 := (abs_le.mp hr).2
  have m1 := mul_le_mul_of_nonneg_right h2 (abs_nonneg (γ * n))
  have m2 := mul_le_mul_of_nonneg_right h2 (abs_nonneg (γ * n'))
  have m3 : (2:ℝ) ^ (-24 : ℤ) * (|bi| + |bj|) ≤ 1 / 10000000 * (2 ^ 13 + 2 ^ 13) :=
    mul_le_mul h2 (by linarith) (by positivity) (by norm_num)
  generalize (2:ℝ) ^ (-24 : ℤ) = A at *
  generalize (2:ℝ) ^ (-150 : ℤ) = B at *
  nlinarith

/-- The exponent facts of the `g_gamma` gate. -/
theorem gam_row (hGm : d.useGamma = true) {t j : Nat} (hr : RowIn d r t j) :
    (d.ggv j).isFinite = true ∧ |((gam d j : ℚ) : ℝ)| ≤ 64 ∧ ((gam d j : ℚ) : ℝ) ≤ 1 := by
  obtain ⟨hf, h64, h1⟩ := hr.2.2.1 hGm
  exact ⟨hf, by exact_mod_cast h64, by exact_mod_cast h1⟩

/-- **The `USE_G_GAMMA` block** (lines 166–170). -/
theorem safe_gateGamma (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    (hm : m = init) {i_h : I64} (hih : i_h.toInt = (g.ih : Int))
    {bo : Tensor [BT, d.bk] F32} {bA : Tensor [BT, BT] F32} (hgt : Gated1 d g r bo bA)
    {k' : Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32 → Prog α}
    (hk : ∀ bo' bA', OGate d g r bo' → AGate d g r bA' → Safe E (Rs d) W Q m (k' (bo', bA'))) :
    Safe E (Rs d) W Q m (gateGamma d i_h bo bA >>= k') := by
  by_cases hGm : d.useGamma = true
  · have hG' : d.useG = false := by
      have := hd.one_gate
      cases h : d.useG
      · rfl
      · exact absurd ⟨h, hGm⟩ this
    obtain ⟨hO, hA⟩ := hgt.2 hG'
    simp only [gateGamma, hGm, ↓reduceIte, Prog.bind_eq, Prog.pure_eq, Prog.bind_assoc,
      Prog.ret_bind]
    subst hm
    have hptr : d.gg.ptr 0 + i_h = d.gg.ptr ((g.ih : Nat) : Int) := by
      show (d.gg.ptr 0).offset i_h.toInt = _
      rw [Buf.ptr_offset, hih]; simp
    rw [hptr]
    obtain ⟨hggv, hggh⟩ := hpre.gg_valid hGm
    have hih' := hg.ih_lt
    have hgglen := hd.gg_len hGm
    refine safe_load1_buf d.gg (i := g.ih) (by omega) hggv ?_ fun γ hγ => ?_
    · exact ((d.gg.span_sub_elems (S := {g.ih}) (Set.mem_singleton _)).trans
        (d.gg.elems_sub_bytes (fun j hj => by rw [Set.mem_singleton_iff.mp hj]; omega))).trans
        (fun x hx => Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨hGm, hx⟩))))))
    have hγv : γ = d.ggv g.ih := by
      rw [hggh _ (by omega)] at hγ; exact (Option.some.inj hγ).symm
    subst hγv
    refine safe_gamma_idx fun nf hnf => ?_
    refine safe_mul fun bg hbg => ?_
    refine safe_math (fun i => hP.mathTotal .exp2 .binary32 _) fun Gh hGh => ?_
    refine safe_mul fun bo' hbo' => ?_
    refine safe_subc fun dd hdd => ?_
    refine safe_math (fun i => hP.mathTotal .exp2 .binary32 _) fun Wh hWh => ?_
    refine safe_mul fun bA' hbA' => ?_
    -- the exponent of row `i`
    have hbgrow : ∀ i : Fin BT, (d.ggv g.ih).isFinite = true ∧ |((gam d g.ih : ℚ) : ℝ)| ≤ 64 ∧
        ((gam d g.ih : ℚ) : ℝ) ≤ 1 →
        (bg.get1 i).isFinite = true ∧
        |(bg.get1 i).toReal - ((gam d g.ih : ℚ) : ℝ) * ((i : ℝ) + 1)| ≤
          2 ^ (-24 : ℤ) * |((gam d g.ih : ℚ) : ℝ) * ((i : ℝ) + 1)| + 2 ^ (-150 : ℤ) := by
      intro i ⟨hγF, hγ64, hγ1⟩
      obtain ⟨hnF, hnv⟩ := hnf i
      have hnr : (nf.get1 i).toReal = (i : ℝ) + 1 := by
        unfold FloatV.toReal; rw [hnv]; push_cast; rfl
      have hn64 : (i : ℝ) + 1 ≤ 64 := by
        have := i.isLt; simp only [BT] at this; have : (i : ℝ) + 1 ≤ 64 := by exact_mod_cast this
        exact this
      have hn0 : (0 : ℝ) ≤ (i : ℝ) := Nat.cast_nonneg _
      have hγR : (d.ggv g.ih).toReal = ((gam d g.ih : ℚ) : ℝ) := rfl
      have hmax : |(d.ggv g.ih).toReal * (nf.get1 i).toReal| ≤ 2 ^ 105 := by
        rw [hγR, hnr, abs_mul]
        have := abs_nonneg ((gam d g.ih : ℚ) : ℝ)
        rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ (i:ℝ) + 1)]
        nlinarith [(by norm_num : (64:ℝ) * 64 ≤ 2 ^ 105)]
      have := mul32_lane (x := d.ggv g.ih) (y := nf.get1 i) hγF hnF hmax (hbg (i, ()))
      rw [hγR, hnr] at this
      exact this
    have hn_bd : ∀ i : Fin BT, (0:ℝ) ≤ (i:ℝ) + 1 ∧ (i:ℝ) + 1 ≤ 64 := by
      intro i
      have := i.isLt; simp only [BT] at this
      exact ⟨by positivity, by exact_mod_cast this⟩
    have hΓdef : gamErr d g.ih = 2 ^ (-15 : ℤ) * |((gam d g.ih : ℚ) : ℝ)| := by
      simp only [gamErr, hGm, ↓reduceIte]; norm_num
    have hexp : ∀ (i : Fin BT), ((((gam d g.ih * (((g.lo + i - rowLo d r : Nat) : ℚ) + 1)) : ℚ)) : ℝ)
        = ((gam d g.ih : ℚ) : ℝ) * ((i : ℝ) + 1) := by
      intro i
      rw [hg.lo_eq, Nat.add_sub_cancel_left]; push_cast; ring
    refine hk bo' bA' ?_ ?_
    · intro i c hrow hcol hin
      have hr := hin.rowIn
      have hγ := gam_row hGm hr
      obtain ⟨hγF, hγ64, hγ1⟩ := hγ
      obtain ⟨hbgF, hbgb⟩ := hbgrow i ⟨hγF, hγ64, hγ1⟩
      obtain ⟨hn0, hn64⟩ := hn_bd i
      have hle65 := x_le_65 (gn_le hγ1 hn0 hn64) hbgb
      obtain ⟨hGf, hGb⟩ := exp2_fact hP (hGh (i, ())) hbgF (le_trans hle65 (by norm_num))
      have hi0 : (0:ℝ) ≤ (i:ℝ) := Nat.cast_nonneg _
      obtain ⟨-, hGb'⟩ := gam_G_num hγ64 hγ1 (by linarith) hn64 hbgb hGb
      obtain ⟨hobF, hob⟩ := oinv_cD hd hO hrow hcol hin.okO
      have hmul := hbo' (i, c, ())
      have hgate : gate d r (g.lo + i) g.ih = (2:ℝ) ^ (((gam d g.ih : ℚ) : ℝ) * ((i : ℝ) + 1)) := by
        rw [gate_gamma hG' hGm, hexp]
      have hGP : gate d r (g.lo + i) g.ih * (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) ≤ 2 ^ 100 :=
        hin.2.2.2.2.2.2.1
      have hP100 : (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) ≤ 2 ^ 100 := by
        exact_mod_cast hin.2.2.2.2.2.1
      have hG0 : (0:ℝ) ≤ gate d r (g.lo + i) g.ih := by rw [hgate]; positivity
      have hG64 : gate d r (g.lo + i) g.ih ≤ 2 ^ 64 := by
        rw [hgate]
        calc _ ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
              Real.rpow_le_rpow_of_exponent_le (by norm_num) (by push_cast; exact gn_le hγ1 hn0 hn64)
          _ = 2 ^ 64 := by norm_num
      have hQ : |(QS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ)| ≤
          (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) := by
        exact_mod_cast abs_QS_le d r (g.lo + i) g.ih (g.iv * d.bk + c)
      have hP0 : (0:ℝ) ≤ (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) := by
        exact_mod_cast PQS_nonneg d r (g.lo + i) g.ih (g.iv * d.bk + c)
      have hΓ0 : 0 ≤ gamErr d g.ih := by rw [hΓdef]; positivity
      have hΓ9 : gamErr d g.ih ≤ 2 ^ (-9 : ℤ) := by
        rw [hΓdef]
        have : (2:ℝ) ^ (-15 : ℤ) * 64 = 2 ^ (-9 : ℤ) := by norm_num
        nlinarith [zpow_pos (by norm_num : (0:ℝ) < 2) (-15 : ℤ)]
      refine og_lane hP0 hQ hG0 hG64 hGP hP100 hΓ0 hΓ9 hobF hob hGf ?_ hmul
      rw [hΓdef, hgate]
      simpa using hGb'
    · intro i i' hrow hle hr
      have hrow' : g.rowOk i' := by
        have : (i' : Nat) ≤ i := hle
        unfold Geo.rowOk at *; omega
      have hs : g.lo + i' ∈ upTo d r (g.lo + i) := by
        simp only [upTo, Finset.mem_Icc, hg.lo_eq]
        have : (i' : Nat) ≤ i := hle
        omega
      have hγ := gam_row hGm hr
      obtain ⟨hγF, hγ64, hγ1⟩ := hγ
      obtain ⟨hbiF, hbi⟩ := hbgrow i ⟨hγF, hγ64, hγ1⟩
      obtain ⟨hbjF, hbj⟩ := hbgrow i' ⟨hγF, hγ64, hγ1⟩
      obtain ⟨hn0, hn64⟩ := hn_bd i
      obtain ⟨hn0', hn64'⟩ := hn_bd i'
      have hi0 : (0:ℝ) ≤ (i:ℝ) := Nat.cast_nonneg _
      have hi'0 : (0:ℝ) ≤ (i':ℝ) := Nat.cast_nonneg _
      have hi'i : (i' : ℝ) ≤ i := by exact_mod_cast hle
      have hbi_abs := bg_abs hγ64 hn0 hn64 hbi
      have hbj_abs := bg_abs hγ64 hn0' hn64' hbj
      have hdd' : AddRel E.fusion (bg.get1 i) (FloatV.neg (bg.get1 i')) (dd.get (i, i', ())) :=
        hdd (i, i', ())
      have hnegF : (FloatV.neg (bg.get1 i')).isFinite = true := by rw [FloatV.isFinite_neg]; exact hbjF
      have hnegR : (FloatV.neg (bg.get1 i')).toReal = -(bg.get1 i').toReal := FloatV.toReal_neg _
      obtain ⟨hddF, hddb⟩ := addc32_lane hbiF hnegF (by
        rw [hnegR, abs_neg]; linarith [(by norm_num : (2:ℝ) ^ 13 + 2 ^ 13 ≤ 2 ^ 105)]) hdd'
      rw [hnegR] at hddb
      have h65 := gam_dd_le hγ64 hγ1 (by linarith : (1:ℝ) ≤ (i':ℝ) + 1) (by linarith)
        hn64 hbi hbj hddb
      obtain ⟨hWf, hWb⟩ := exp2_fact hP (hWh (i, i', ())) hddF (by linarith)
      obtain ⟨-, hwb⟩ := gam_w_num hγ64 hγ1 (by linarith : (1:ℝ) ≤ (i':ℝ) + 1) (by linarith) hn64
        hbi hbj hddb hWb
      obtain ⟨habF, hab⟩ := ainv_cD hd hA hrow hle (hr.okA hs)
      have hmul := hbA' (i, i', ())
      have hw : wgt d r (g.lo + i) (g.lo + i') g.ih =
          (2:ℝ) ^ (((gam d g.ih : ℚ) : ℝ) * (((i:ℝ) + 1) - ((i':ℝ) + 1))) := by
        rw [wgt_gamma hG' hGm]; push_cast; ring_nf
      obtain ⟨hPA100, hwPA⟩ := hr.2.2.2.2 _ hs
      have hΓ0 : 0 ≤ gamErr d g.ih := by rw [hΓdef]; positivity
      have hΓ9 : gamErr d g.ih ≤ 2 ^ (-9 : ℤ) := by
        rw [hΓdef]
        have : (2:ℝ) ^ (-15 : ℤ) * 64 = 2 ^ (-9 : ℤ) := by norm_num
        nlinarith [zpow_pos (by norm_num : (0:ℝ) < 2) (-15 : ℤ)]
      have hw0 : (0:ℝ) ≤ wgt d r (g.lo + i) (g.lo + i') g.ih := by rw [hw]; positivity
      have hw64 : wgt d r (g.lo + i) (g.lo + i') g.ih ≤ 2 ^ 64 := by
        rw [hw]
        have hd64 : ((gam d g.ih : ℚ) : ℝ) * (((i:ℝ) + 1) - ((i':ℝ) + 1)) ≤ 64 :=
          gn_le hγ1 (by linarith) (by linarith)
        calc _ ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
              Real.rpow_le_rpow_of_exponent_le (by norm_num) (by push_cast; exact hd64)
          _ = 2 ^ 64 := by norm_num
      have hPA0 : (0:ℝ) ≤ (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) := by
        exact_mod_cast PA_nonneg d _ _ _
      have hA' : |(Aqk d (g.lo + i) (g.lo + i') g.ih : ℝ)| ≤ (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) := by
        exact_mod_cast abs_Aqk_le d _ _ _
      refine ag_lane hPA0 hA' hw0 hw64 hwPA (by exact_mod_cast hPA100) hΓ0 hΓ9 habF hab hWf ?_ hmul
      rw [hΓdef, hw]
      have : (i:ℝ) + 1 - ((i':ℝ) + 1) = (i:ℝ) + 1 - ((i':ℝ) + 1) := rfl
      simpa using hwb
  · have hGm' : d.useGamma = false := by simpa using hGm
    simp only [gateGamma, hGm, Prog.bind_eq, Prog.pure_eq,
     ]
    by_cases hG : d.useG = true
    · obtain ⟨hO, hA⟩ := hgt.1 hG
      exact hk bo bA hO hA
    · have hG' : d.useG = false := by simpa using hG
      obtain ⟨hO, hA⟩ := hgt.2 hG'
      refine hk bo bA ?_ ?_
      · intro i c hrow hcol hin
        obtain ⟨hf, hb⟩ := oinv_cD hd hO hrow hcol hin.okO
        have hP0 : (0:ℝ) ≤ (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) := by
          exact_mod_cast PQS_nonneg d _ _ _ _
        have h1 : gate d r (g.lo + i) g.ih = 1 := by simp [gate, hG', hGm']
        rw [h1, gamErr_zero hGm']
        exact ⟨hf, gate_o_none hP0 le_rfl hb⟩
      · intro i i' hrow hle hr
        have hs : g.lo + i' ∈ upTo d r (g.lo + i) := by
          simp only [upTo, Finset.mem_Icc, hg.lo_eq]
          have : (i' : Nat) ≤ i := hle
          omega
        obtain ⟨hf, hb⟩ := ainv_cD hd hA hrow hle (hr.okA hs)
        have hP0 : (0:ℝ) ≤ (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) := by
          exact_mod_cast PA_nonneg d _ _ _
        have h1 : wgt d r (g.lo + i) (g.lo + i') g.ih = 1 := by simp [wgt, hG', hGm']
        rw [h1, gamErr_zero hGm']
        exact ⟨hf, gate_a_none hP0 le_rfl hb⟩

end Claims.ChunkFwdO
