import TVA.Proof
import Claim
import Proof.Pid
import Proof.Foot
import Proof.Defs
import Proof.Tol
import Proof.Store
import Proof.Inv

/-!
# The epilogue

`tail_safe`: `tailOf` (lines 403-420) loads the routing weights (with `MUL_ROUTED_WEIGHT`),
multiplies, rounds to `bfloat16` and stores the tile; from `AccInv` after all trips, every stored
element satisfies `outElem`.
-/

namespace Claims.FusedMoe

open TVA


/-- A nonnegative integer below `n` has a `toNat` below `n`. -/
theorem toNat_lt_of_lt {x : Int} {n : Nat} (h0 : 0 ≤ x) (h : x < (n : Int)) : x.toNat < n := by omega

/-- The routed value without the routing weight: the `bfloat16` rounding of the accumulator. -/
theorem outFin_norw {d : Inst} {init : Mem} {f : Nat} {cv : BF16} {u : F32} {S T : ℚ}
    (hrw : d.rw = false) (hu : u.isFinite = true) (hT : |u.toRat - S| ≤ T)
    (hnan : FloatV.NaNEq (FloatV.cvt u) cv) : outFin d init f cv u S T := by
  unfold outFin
  rw [ite_eq_right (by simp [hrw])]
  refine ⟨hnan, fun hle => ?_⟩
  have hmax : |u.toRat| ≤ Format.bfloat16.maxVal := by
    have := abs_sub_abs_le_abs_sub u.toRat S
    linarith
  obtain ⟨hfin, -⟩ := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular hu hmax
  rw [← hnan.isFinite_eq]; exact hfin

/-- The routed value with the routing weight. -/
theorem outFin_rw {d : Inst} {init : Mem} {f : Nat} {cv : BF16} {u : F32} {S T : ℚ}
    (hrw : d.rw = true) {wv : F32} (hw : init.read .f32 (d.w.ptr (f : Int)).addr = some wv)
    (hwf : wv.isFinite = true) (hu : u.isFinite = true) (hT : |u.toRat - S| ≤ T) {r : F32}
    (hmul : FloatV.NaNEq (FloatV.mul u wv) r) (hcv : FloatV.NaNEq (FloatV.cvt r) cv) :
    outFin d init f cv u S T := by
  unfold outFin
  rw [ite_eq_left hrw]
  refine ⟨wv, hw, hwf, r, hmul, hcv, fun hle => ?_⟩
  have hu' : |u.toRat| ≤ |S| + T := by
    have := abs_sub_abs_le_abs_sub u.toRat S
    linarith
  have h1 : |u.toRat * wv.toRat| ≤ 2 ^ 126 := by
    rw [abs_mul]
    exact le_trans (mul_le_mul_of_nonneg_right hu' (abs_nonneg _)) hle
  have hmax32 : |u.toRat * wv.toRat| ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]; refine h1.trans ?_; norm_num
  obtain ⟨hmfin, hmerr⟩ := FloatV.mul_err Format.binary32_regular hu hwf hmax32
  have hrfin : r.isFinite = true := by rw [← hmul.isFinite_eq]; exact hmfin
  refine ⟨hrfin, ?_⟩
  have hrval : r.toRat = (FloatV.mul u wv).toRat := (hmul.exact_of_isFinite hmfin).2.1
  have hr2 : |r.toRat| ≤ Format.bfloat16.maxVal := by
    rw [Format.binary32_u, Format.binary32_eta] at hmerr
    rw [Format.bfloat16_maxVal, hrval]
    have h2 := abs_sub_abs_le_abs_sub (FloatV.mul u wv).toRat (u.toRat * wv.toRat)
    have h3 : |u.toRat * wv.toRat| * 2 ^ (-24 : ℤ) ≤ 2 ^ 126 * 2 ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right h1 (by norm_num)
    have h4 : (2 : ℚ) ^ 126 * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) + 2 ^ 126 ≤ (2 ^ 8 - 1) * 2 ^ 120 := by norm_num
    linarith
  obtain ⟨hcfin, -⟩ := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular hrfin hr2
  rw [← hcv.isFinite_eq]; exact hcfin


/-- **A stored lane satisfies `outElem`**: the lane's accumulator after all trips is within `tol` of
the exact dot product, and the stored element is the rounding of it (times the routing weight). -/
theorem outElem_of_lane {d : Inst} (hd : d.ok) {init m' : Mem}
    {pmN pnN e l j : Nat} (he : e < d.E) (hE : eOf d init pmN = (e : Int)) (hl : l < d.BM)
    (hj : j < d.BN) (hent : entryOf d init (pmN * d.BM + l) < (nv d : Int)) (hc : pnN * d.BN + j < d.N)
    {acc : Tensor [d.BM, d.BN] F32}
    (hacc : AccInv d init pmN pnN e (steps d.BK d.K) acc) {cv : BF16}
    (hread : m'.read .bf16 (d.c.ptr (((entryOf d init (pmN * d.BM + l)).toNat * d.N + (pnN * d.BN + j) : Nat) : Int)).addr
      = some cv)
    (hrel : (d.rw = false → FloatV.NaNEq (FloatV.cvt (acc.get2 ⟨l, hl⟩ ⟨j, hj⟩)) cv) ∧
      (d.rw = true → ∃ (wv r : F32), init.read .f32 (d.w.ptr (((entryOf d init (pmN * d.BM + l)).toNat : Nat) : Int)).addr = some wv ∧
        wv.isFinite = true ∧ FloatV.NaNEq (FloatV.mul (acc.get2 ⟨l, hl⟩ ⟨j, hj⟩) wv) r ∧
        FloatV.NaNEq (FloatV.cvt r) cv)) :
    outElem d init m' (pmN * d.BM + l) (entryOf d init (pmN * d.BM + l)).toNat (pnN * d.BN + j) := by
  have hBM := (cfg_facts hd).1
  have hBK := (cfg_facts hd).2.2.1
  have hdiv : (pmN * d.BM + l) / d.BM = pmN := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hl, Nat.zero_add]
  refine ⟨cv, hread, ?_⟩
  rw [hdiv, ite_eq_right (by rw [hE]; omega)]
  refine ⟨e, hE, he, fun va vb hva hvb hmag => ?_⟩
  have hvb' : ∀ k, k < d.K → init.read .bf16
      (d.b.ptr ((e * d.stride_be + bcol d pnN j * d.stride_bn + k : Nat) : Int)).addr =
        some (vb k) ∧ (vb k).isFinite = true := by
    intro k hk
    have : bcol d pnN j = pnN * d.BN + j := Nat.mod_eq_of_lt hc
    rw [this]; exact hvb k hk
  obtain ⟨v, hv, hbound⟩ := hacc ⟨l, hl⟩ ⟨j, hj⟩ hent va vb hva hvb' hmag
  obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hv
  have hfin_ := accErr_final (BK := d.BK) (K := d.K) hBK hd.K_le15 va vb
  have hps := preSum_final (BK := d.BK) (K := d.K) hBK va vb
  have hT : |(acc.get2 ⟨l, hl⟩ ⟨j, hj⟩).toRat - dotSum va vb d.K| ≤ tol d.BK d.K va vb := by
    rw [hval]; rw [hps] at hbound; exact hbound.trans hfin_
  refine ⟨acc.get2 ⟨l, hl⟩ ⟨j, hj⟩, hfin, hT, ?_⟩
  by_cases hrw : d.rw = true
  · obtain ⟨wv, r, hw, hwf, hmul, hcv⟩ := hrel.2 hrw
    exact outFin_rw hrw hw hwf hfin hT hmul hcv
  · exact outFin_norw (by simpa using hrw) hfin hT (hrel.1 (by simpa using hrw))

/-- **The epilogue stores the tile and leaves `outElem` in every element.** -/
theorem tail_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {pmN pnN e : Nat} (he : e < d.E) (hE : eOf d init pmN = (e : Int))
    (hact : pmN * d.BM + d.BM ≤ ntpOf d init) {pn : I32} (hpn : pn.toInt = (pnN : Int))
    (hpnb : pnN * d.BN + d.BN ≤ d.N + d.BN - 1)
    {ot : Tensor [d.BM] I64} {tm : Tensor [d.BM] Bool} (hl : TokLanes d init pmN ot tm)
    {acc : Tensor [d.BM, d.BN] F32} (hacc : AccInv d init pmN pnN e (steps d.BK d.K) acc) :
    Safe ((launch d).env P) (Rfp d) (d.c.elems (Wset d init pmN pnN))
      (fun _ m => QT d init pmN pnN m) init (tailOf d pn ot tm acc) := by
  have hBM := (cfg_facts hd).1
  have hnv := hd.nv_le
  have hent0 : ∀ l : Fin d.BM, 0 ≤ entryOf d init (pmN * d.BM + l) := fun l =>
    hpre.s1 (pmN * d.BM + l) (by have := l.isLt; omega)
  have hrowv : ∀ l : Fin d.BM, (ot.get1 l).toInt =
      (((entryOf d init (pmN * d.BM + l)).toNat : Nat) : Int) :=
    fun l => by rw [hl.1 l]; exact (Int.toNat_of_nonneg (hent0 l)).symm
  have hrnd : (Format.binary32.Std ∧ Format.bfloat16.Std ∧ tl.RoundOk Format.binary32 Format.bfloat16 .rn) :=
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩
  unfold tailOf
  by_cases hrw : d.rw = true
  · rw [ite_eq_left hrw]
    refine safe_convert (d := .i32) (d' := .f32) (x := tl.full [d.BM] (IntV.ofInt 0 : I32))
      (fun i => ⟨FloatV.ofRat _, Or.inl rfl⟩) fun za _ => ?_
    refine safe_load_buf d.w
      (g := fun idx : Idx [d.BM] => (entryOf d init (pmN * d.BM + idx.1)).toNat)
      (en := fun idx : Idx [d.BM] => decide (entryOf d init (pmN * d.BM + idx.1) < (nv d : Int)))
      ?_ ?_ hpre.vw ?_ fun mw hmw _ => ?_
    · refine Buf.addrs_of_get d.w (fun i => hl.2 i.1) fun i hi => ?_
      show (d.w.ptr 0).offset (ot.get i).toInt = _
      rw [show (ot.get i).toInt = (((entryOf d init (pmN * d.BM + i.1)).toNat : Nat) : Int) from hrowv i.1,
        Buf.ptr_offset]
      simp
    · intro i hi
      simp only [decide_eq_true_eq] at hi
      have := hent0 i.1
      exact hd.w_len _ (toNat_lt_of_lt this hi)
    · refine Set.Subset.trans (d.w.elems_sub_bytes ?_) (Rfp_w d)
      rintro x ⟨i, hi, rfl⟩
      simp only [decide_eq_true_eq] at hi
      have := hent0 i.1
      exact hd.w_len _ (toNat_lt_of_lt this hi)
    · refine safe_mul fun pr hpr => ?_
      refine safe_cvt (g := .bfloat16) hrnd fun cc hcc => ?_
      refine safe_last (safe_storeC hd hpre hpn hpnb hact hl cc (Set.Subset.refl _)
        fun m' hread hframe => ?_)
      refine safe_ret ?_
      intro l j hl' hj' hpos hent hc
      have hf : (entryOf d init (pmN * d.BM + l)).toNat < nv d :=
        toNat_lt_of_lt (hent0 ⟨l, hl'⟩) hent
      obtain ⟨wv, hwr, hwf⟩ := hpre.wfin hrw _ hf
      have hmwl : mw.get1 ⟨l, hl'⟩ = wv := by
        have h1 := hmw (⟨l, hl'⟩, ()) (by simp [hent])
        exact Option.some.inj (h1.symm.trans hwr)
      refine outElem_of_lane hd he hE hl' hj' hent hc hacc
        (hread ⟨l, hl'⟩ ⟨j, hj'⟩ hent hc)
        ⟨fun h => absurd h (by simp [hrw]), fun _ => ⟨wv, pr.get2 ⟨l, hl'⟩ ⟨j, hj'⟩, hwr, hwf, ?_, ?_⟩⟩
      · have h := hpr (⟨l, hl'⟩, ⟨j, hj'⟩, ())
        rw [show ((mw.col d.BN).get (⟨l, hl'⟩, ⟨j, hj'⟩, ())) = wv from hmwl] at h
        exact h
      · have h := hcc (⟨l, hl'⟩, ⟨j, hj'⟩, ())
        rw [FloatV.cvtWith_rn] at h
        exact h
  · have hrw' : d.rw = false := by simpa using hrw
    rw [ite_eq_right hrw]
    refine safe_cvt (g := .bfloat16) hrnd fun cc hcc => ?_
    refine safe_last (safe_storeC hd hpre hpn hpnb hact hl cc (Set.Subset.refl _)
      fun m' hread hframe => ?_)
    refine safe_ret ?_
    intro l j hl' hj' hpos hent hc
    refine outElem_of_lane hd he hE hl' hj' hent hc hacc
      (hread ⟨l, hl'⟩ ⟨j, hj'⟩ hent hc) ⟨fun _ => ?_, fun h => absurd h hrw⟩
    have h := hcc (⟨l, hl'⟩, ⟨j, hj'⟩, ())
    rw [FloatV.cvtWith_rn] at h
    exact h

end Claims.FusedMoe
