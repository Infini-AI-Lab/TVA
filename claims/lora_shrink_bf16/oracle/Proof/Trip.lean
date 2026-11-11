import Proof.TripN

/-!
# `lora_shrink_bf16`: one trip of the loop, on one accumulator lane

The loop of `mm_k` (line 136) adds, in trip `t` of part `p`, the product of `BLOCK_K` consecutive
`k` (the block `p + SPLIT_K · t`) to each accumulator lane with `tl.dot_add`. The device facts
(`Device.DotBF16`, `Device.DotBF16Chunk`) bound the result against the exact partial sums of the
claim, two ways at once (`LaneInv`).
-/

namespace Claims.LoraShrink

open TVA

/-- The accumulator starts at `+0.0` (line 128). -/
theorem laneInv_zero (d : Inst) (va vb : Nat → BF16) (p : Nat) :
    LaneInv d va vb p 0 (FloatV.ofPat 0 : F32) := by
  have h0 : (FloatV.ofPat 0 : F32).isFinite = true ∧ (FloatV.ofPat 0 : F32).toRat = 0 :=
    DType.val?_f32.mp DType.val?_zero_f32
  have hs : pSum d va vb p (cpb d * 0) = 0 := by simp [pSum]
  have hl : pLift d va vb p 0 = 0 := by simp [pLift]
  refine ⟨h0.1, ?_, ?_⟩
  · rw [hs, h0.2]; simp [accErr]
  · rw [hs, h0.2, hl]; simp [DotLoop.bound]

/-- **One trip of one lane.** Lane `(i, n)` of the accumulator holds a `LaneInv` value after `t`
trips; the trip `t < stepsOf p` (the block `b = p + SPLIT_K · t` starts below `K`) adds the dot of
the tile row `i` and tile column `n`, whose `kk`-th entries are the values at `k = b · BLOCK_K + kk`
(`+0.0` for `k ≥ K`, the masked-out lanes); the device gives `r` in either form of `tl.dot_add`. Then
lane `(i, n)` of `r` is `LaneInv` after `t + 1` trips. -/
theorem lane_trip {P : Platform} (hP : Device P) {fusion : Bool} (d : Inst) (hd : d.ok)
    (va vb : Nat → BF16) (hfa : ∀ k, k < d.K → (va k).isFinite = true)
    (hfb : ∀ k, k < d.K → (vb k).isFinite = true) (hdm : dotMag d va vb ≤ 2 ^ 120)
    (p t : Nat) (_hp : p < sk d) (ht : t < stepsOf d p)
    (a : Tensor [32, bk d] BF16) (b : Tensor [bk d, 16] BF16) (c z r : Tensor [32, 16] F32)
    (i : Fin 32) (n : Fin 16)
    (hr : P.dot dotCfg a b c r ∨ ∃ dd, P.dot dotCfg a b z dd ∧
      ∀ ix, AccAdd fusion .f32 (dd.get ix) (c.get ix) (r.get ix))
    (hz : DType.val? .f32 (z.get2 i n) = some 0)
    (ha : ∀ kk : Fin (bk d), a.get2 i kk =
      if (p + sk d * t) * bk d + kk < d.K then va ((p + sk d * t) * bk d + kk) else FloatV.ofPat 0)
    (hb : ∀ kk : Fin (bk d), b.get2 kk n =
      if (p + sk d * t) * bk d + kk < d.K then vb ((p + sk d * t) * bk d + kk) else FloatV.ofPat 0)
    (hc : LaneInv d va vb p t (c.get2 i n)) :
    LaneInv d va vb p (t + 1) (r.get2 i n) := by
  obtain ⟨hcfin, hcch, hcst⟩ := hc
  have hb0 := bk_pos d
  have hK := hd.K_top
  have hT : t + 1 ≤ 512 := by have := stepsOf_le d hK p; omega
  have hz0 : (FloatV.ofPat 0 : BF16).isFinite = true ∧ (FloatV.ofPat 0 : BF16).toRat = 0 :=
    DType.val?_bf16.mp DType.val?_zero_bf16
  obtain ⟨la, hla_def⟩ : ∃ la : Fin (bk d) → ℚ, ∀ kk, la kk = (a.get2 i kk).toRat :=
    ⟨_, fun _ => rfl⟩
  obtain ⟨lb, hlb_def⟩ : ∃ lb : Fin (bk d) → ℚ, ∀ kk, lb kk = (b.get2 kk n).toRat :=
    ⟨_, fun _ => rfl⟩
  have hfa' : ∀ kk : Fin (bk d), (a.get2 i kk).isFinite = true := by
    intro kk; rw [ha kk]; split_ifs with h
    · exact hfa _ h
    · exact hz0.1
  have hfb' : ∀ kk : Fin (bk d), (b.get2 kk n).isFinite = true := by
    intro kk; rw [hb kk]; split_ifs with h
    · exact hfb _ h
    · exact hz0.1
  have hla : ∀ kk : Fin (bk d), la kk =
      if (p + sk d * t) * bk d + kk < d.K then (va ((p + sk d * t) * bk d + kk)).toRat else 0 := by
    intro kk; rw [hla_def, ha kk]; split_ifs
    · rfl
    · exact hz0.2
  have hlb : ∀ kk : Fin (bk d), lb kk =
      if (p + sk d * t) * bk d + kk < d.K then (vb ((p + sk d * t) * bk d + kk)).toRat else 0 := by
    intro kk; rw [hlb_def, hb kk]; split_ifs
    · rfl
    · exact hz0.2
  have hvA : ∀ kk : Fin (bk d), DType.val? .bf16 (a.get2 i kk) = some (la kk) :=
    fun kk => DType.val?_bf16.mpr ⟨hfa' kk, (hla_def kk).symm⟩
  have hvB : ∀ kk : Fin (bk d), DType.val? .bf16 (b.get2 kk n) = some (lb kk) :=
    fun kk => DType.val?_bf16.mpr ⟨hfb' kk, (hlb_def kk).symm⟩
  have hD := pSum_block d va vb p t la lb hla hlb
  have hLs := pLift_block d va vb p t la lb hla hlb
  obtain ⟨A, hA⟩ : ∃ A, A = ∑ kk, Device.liftMag .bf16 (la kk) * Device.liftMag .bf16 (lb kk) :=
    ⟨_, rfl⟩
  have hA0 : 0 ≤ A := by
    rw [hA]
    exact Finset.sum_nonneg fun _ _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hAeq : A = pLift d va vb p (t + 1) - pLift d va vb p t := by rw [hLs, hA]; ring
  have hpl1 : pLift d va vb p (t + 1) ≤ 2 ^ 120 := (pLift_le_dotMag d va vb p (t + 1)).trans hdm
  have hpl0 := pLift_nonneg d va vb p t
  have hA120 : A ≤ 2 ^ 120 := by linarith
  have hBt120 : pLift d va vb p t ≤ 2 ^ 120 := by
    have := pLift_nonneg d va vb p (t + 1)
    have h := pLift_nonneg d va vb p t
    linarith
  have hPt := abs_pSum_le d va vb p t
  have hbd := bound_small d (t := t) (by omega) hpl0
  obtain ⟨cv, hcv⟩ : ∃ cv, cv = (c.get2 i n).toRat := ⟨_, rfl⟩
  rw [← hcv] at hcch hcst
  have hcabs : |cv| ≤ 2 ^ 121 := by
    have h1 : |cv| ≤ |cv - pSum d va vb p (cpb d * t)| + |pSum d va vb p (cpb d * t)| := by
      have := abs_add_le (cv - pSum d va vb p (cpb d * t)) (pSum d va vb p (cpb d * t))
      simpa using this
    linarith
  have hSabs : ∑ kk, |la kk * lb kk| ≤ A := by
    rw [hA]
    exact Finset.sum_le_sum fun kk _ => Device.abs_mul_le_liftMag _ _ _ _
  have hSA : |∑ kk, la kk * lb kk| ≤ A :=
    (Finset.abs_sum_le_sum_abs _ _).trans (by
      refine hSabs.trans' ?_
      exact le_of_eq rfl)
  have hrel := relDot16_le d
  have habs := absDot16_le d
  have hrel0 := relDot16_nonneg d
  have habs0 := absDot16_nonneg d
  have heta : Format.binary32.eta ≤ 1 := by rw [Format.binary32_eta]; norm_num
  have hmax := two127_le_maxVal
  -- the step-by-step bound
  have hroomS : 2 * ((1 + Device.relDot16 (bk d)) *
      (|cv| + ∑ kk, Device.liftMag .bf16 (la kk) * Device.liftMag .bf16 (lb kk)) +
      Device.absDot16 (bk d) + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    rw [← hA]
    have : (1 + Device.relDot16 (bk d)) * (|cv| + A) ≤ 2 * (|cv| + A) := by
      nlinarith [abs_nonneg cv]
    linarith
  obtain ⟨hrfin, hrerr⟩ := DotLift.dotAdd_step (hP.dotBF16 .tf32) hrel0 habs0 hr i n hz
    (va := la) (vb := lb) hvA hvB hcfin (by rw [← hcv]; exact hroomS)
  rw [← hcv, ← hA] at hrerr
  have hstepb : |(r.get2 i n).toRat - pSum d va vb p (cpb d * (t + 1))| ≤
      DotLoop.bound (rho d) (alpha d) (t + 1) (pLift d va vb p (t + 1)) := by
    have h := DotLoop.accum_step (rho_nonneg d) (alpha_nonneg d) (St := pSum d va vb p (cpb d * t))
      (S := ∑ kk, la kk * lb kk) (Bt := pLift d va vb p t) (A := A) hPt hSA hcst
      (a' := (r.get2 i n).toRat) (a := cv) hrerr
    rw [← hD, hAeq] at h
    simpa using h
  -- the chunk-by-chunk bound
  have hN : Device.chunks16 (bk d) = cpb d := chunks16_bk d
  have e1 : cpb d * t = t * cpb d := Nat.mul_comm _ _
  have e2 : cpb d * (t + 1) = t * cpb d + cpb d := by rw [Nat.mul_succ, e1]
  rw [e1] at hcch hD hPt
  rw [e2] at hD hstepb
  have hcSum : ∀ j, j < cpb d →
      cSum d va vb p (t * cpb d + j) = Device.chunkSum16 la lb j := by
    intro j hj
    unfold cSum Device.chunkSum16
    exact (sum_chunk_eq d p t hj (fun k => (va k).toRat * (vb k).toRat) (fun kk => la kk * lb kk)
      (tile_prod d p t (fun k => (va k).toRat) (fun k => (vb k).toRat) la lb hla hlb)).symm
  have hPstep : ∀ j, j < cpb d → pSum d va vb p (t * cpb d + j + 1) =
      pSum d va vb p (t * cpb d + j) + Device.chunkSum16 la lb j := by
    intro j hj
    unfold pSum
    rw [Finset.sum_range_succ, hcSum j hj]
  have hlift : ∀ j, j < cpb d → ∀ Mj : ℚ, 0 ≤ Mj → cLift d va vb p (t * cpb d + j) ≤ Mj →
      ∀ kk ∈ Device.chunk16 (bk d) j,
        Device.liftMag .bf16 (la kk) * Device.liftMag .bf16 (lb kk) ≤ Mj := by
    intro j hj Mj hM0 hMj kk hkk
    rw [hla kk, hlb kk]
    split_ifs with h
    · refine le_trans ?_ hMj
      unfold cLift
      exact (Finset.le_fold_max _).mpr
        (Or.inr ⟨(p + sk d * t) * bk d + kk, mem_chunkK_tile d p t hj kk hkk h, le_rfl⟩)
    · simpa using hM0
  have hroomC : |cv| + ∑ kk, |la kk * lb kk| + (Device.relChunk16 (bk d) *
      (|cv| + ∑ kk, |la kk * lb kk|) + Device.absChunk16 (bk d)) ≤ DType.bound .f32 := by
    obtain ⟨hr1, ha1⟩ := chunk_room_le d
    have hS0 : 0 ≤ ∑ kk, |la kk * lb kk| := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have : Device.relChunk16 (bk d) * (|cv| + ∑ kk, |la kk * lb kk|) ≤
        (|cv| + ∑ kk, |la kk * lb kk|) := by
      nlinarith [abs_nonneg cv, Device.relChunk16 (bk d)]
    show _ ≤ Format.binary32.maxVal
    linarith
  have hsteps : ∀ (s : Nat → ℚ), (∀ j < cpb d, ∀ Mj : ℚ, |s j| ≤ Mj →
      (∀ k ∈ Device.chunk16 (bk d) j,
        Device.liftMag .bf16 (la k) * Device.liftMag .bf16 (lb k) ≤ Mj) →
      |s (j + 1) - (s j + Device.chunkSum16 la lb j)| ≤
        Device.rhoChunk16 * Mj + Device.sigmaChunk16 * |s j + Device.chunkSum16 la lb j| +
          Device.alphaChunk16) →
      ∀ j < cpb d, ∀ Mj : ℚ, |s j| ≤ Mj → cLift d va vb p (t * cpb d + j) ≤ Mj →
      |s (j + 1) - (s j + Device.chunkSum16 la lb j)| ≤
        Device.rhoChunk16 * Mj + Device.sigmaChunk16 * |s j + Device.chunkSum16 la lb j| +
          Device.alphaChunk16 := by
    intro s hs j hj Mj hsj hMj
    exact hs j hj Mj hsj (hlift j hj Mj ((abs_nonneg _).trans hsj) hMj)
  have hfused : P.dot dotCfg a b c r →
      |(r.get2 i n).toRat - pSum d va vb p (t * cpb d + cpb d)| ≤
        cChain d va vb p (t * cpb d) (pSum d va vb p) (accErr d va vb p t) (cpb d) := by
    intro hd1
    obtain ⟨s, vr, hvr, hs0, hsn, hst⟩ := Device.DotChunk16Loss.step_left (hP.dotBF16Chunk .tf32)
      a b c r hd1 i n la lb cv hvA hvB (DType.val?_f32.mpr ⟨hcfin, hcv.symm⟩) hroomC
    rw [hN] at hsn hst
    have hvr' : (r.get2 i n).toRat = vr := (DType.val?_f32.mp hvr).2
    have h := cchain_bound d va vb p (t * cpb d) (pSum d va vb p) s
      (fun j => Device.chunkSum16 la lb j) (accErr d va vb p t) (cpb d) hPstep
      (by rw [hs0]; exact hcch) (fun j hj Mj h1 h2 => by
        have := hsteps s hst j hj Mj h1 h2
        exact this) (cpb d) le_rfl
    rw [hsn, ← hvr'] at h
    exact h
  have hsep : ∀ dd : Tensor [32, 16] F32, P.dot dotCfg a b z dd →
      (∀ ix, AccAdd fusion .f32 (dd.get ix) (c.get ix) (r.get ix)) →
      |(r.get2 i n).toRat - pSum d va vb p (t * cpb d + cpb d)| ≤
        accErr d va vb p t +
          cChain d va vb p (t * cpb d) (fun g => pSum d va vb p g - pSum d va vb p (t * cpb d)) 0
            (cpb d) +
          3 * Format.binary32.u *
            (|pSum d va vb p (t * cpb d + cpb d) - pSum d va vb p (t * cpb d)| +
              cChain d va vb p (t * cpb d)
                (fun g => pSum d va vb p g - pSum d va vb p (t * cpb d)) 0 (cpb d) +
              |pSum d va vb p (t * cpb d)| + accErr d va vb p t) +
          4 * Format.binary32.eta := by
    intro dd hd1 hadd
    have hroomZ : |(0 : ℚ)| + ∑ kk, |la kk * lb kk| + (Device.relChunk16 (bk d) *
        (|(0 : ℚ)| + ∑ kk, |la kk * lb kk|) + Device.absChunk16 (bk d)) ≤ DType.bound .f32 := by
      obtain ⟨hr1, ha1⟩ := chunk_room_le d
      have hS0 : 0 ≤ ∑ kk, |la kk * lb kk| := Finset.sum_nonneg fun _ _ => abs_nonneg _
      have : Device.relChunk16 (bk d) * (|(0 : ℚ)| + ∑ kk, |la kk * lb kk|) ≤
          (|(0 : ℚ)| + ∑ kk, |la kk * lb kk|) := by
        have := mul_le_mul_of_nonneg_right hr1 (add_nonneg (abs_nonneg (0 : ℚ)) hS0)
        linarith
      show _ ≤ Format.binary32.maxVal
      simp only [abs_zero] at this ⊢
      linarith
    obtain ⟨s, vr, hvr, hs0, hsn, hst⟩ := Device.DotChunk16Loss.step_left (hP.dotBF16Chunk .tf32)
      a b z dd hd1 i n la lb 0 hvA hvB hz hroomZ
    rw [hN] at hsn hst
    obtain ⟨hddfin, hddval⟩ := DType.val?_f32.mp hvr
    have hL := cchain_bound d va vb p (t * cpb d)
      (fun g => pSum d va vb p g - pSum d va vb p (t * cpb d)) s
      (fun j => Device.chunkSum16 la lb j) 0 (cpb d)
      (fun j hj => by
        show pSum d va vb p (t * cpb d + j + 1) - pSum d va vb p (t * cpb d) =
          pSum d va vb p (t * cpb d + j) - pSum d va vb p (t * cpb d) + Device.chunkSum16 la lb j
        rw [hPstep j hj]; ring)
      (by rw [hs0]; simp) (fun j hj Mj h1 h2 => hsteps s hst j hj Mj h1 h2) (cpb d) le_rfl
    rw [hsn, ← hddval] at hL
    -- the zero-accumulator dot, relative to the lifted magnitudes
    have hroom0 : |(0 : ℚ)| + ∑ kk, Device.liftMag .bf16 (la kk) * Device.liftMag .bf16 (lb kk) +
        (Device.relDot16 (bk d) * (|(0 : ℚ)| +
          ∑ kk, Device.liftMag .bf16 (la kk) * Device.liftMag .bf16 (lb kk)) +
          Device.absDot16 (bk d)) ≤ DType.bound .f32 := by
      rw [← hA]
      have : Device.relDot16 (bk d) * (|(0 : ℚ)| + A) ≤ (|(0 : ℚ)| + A) := by
        have := mul_le_mul_of_nonneg_right hrel (add_nonneg (abs_nonneg (0 : ℚ)) hA0)
        linarith
      show _ ≤ Format.binary32.maxVal
      simp only [abs_zero] at this ⊢
      linarith
    obtain ⟨vd', hvd', herrd⟩ := (hP.dotBF16 .tf32) a b z dd hd1 i n la lb 0 hvA hvB hz hroom0
    have hvv : vd' = vr := Option.some.inj (hvd'.symm.trans hvr)
    rw [hvv] at herrd
    have hddabs : |(dd.get2 i n).toRat| ≤ 2 * A + 1 := by
      rw [hddval]
      rw [← hA] at herrd
      have h1 := abs_sub_abs_le_abs_sub vr (0 + ∑ kk, la kk * lb kk)
      have h2 : Device.relDot16 (bk d) * A ≤ A := by
        have := mul_le_mul_of_nonneg_right hrel hA0
        linarith
      simp only [abs_zero, zero_add] at herrd h1
      linarith
    have hu0 : 0 ≤ Format.binary32.u := by rw [Format.binary32_u]; positivity
    have hadd2 : AddRel fusion (dd.get2 i n) (c.get2 i n) (r.get2 i n) := hadd (i, n, ())
    have hM : |(dd.get2 i n).toRat| + |(c.get2 i n).toRat| ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
      rw [← hcv]
      have : Format.binary32.maxVal / 2 ≥ 2 ^ 126 := by linarith
      linarith
    have hres := FloatV.AddRel.err Format.binary32_regular (by decide)
      (by rw [Format.binary32_u]; norm_num) hadd2 hddfin hcfin hM
    rw [← hcv] at hres
    obtain ⟨_, hres⟩ := hres
    have hD' := hD
    have hdd : |(dd.get2 i n).toRat - (pSum d va vb p (t * cpb d + cpb d) -
        pSum d va vb p (t * cpb d))| ≤
        cChain d va vb p (t * cpb d) (fun g => pSum d va vb p g - pSum d va vb p (t * cpb d)) 0
          (cpb d) := by
      have := hL
      simpa using this
    generalize cChain d va vb p (t * cpb d)
      (fun g => pSum d va vb p g - pSum d va vb p (t * cpb d)) 0 (cpb d) = L at hdd ⊢
    generalize accErr d va vb p t = e at hcch ⊢
    generalize pSum d va vb p (t * cpb d + cpb d) = P1 at hdd ⊢
    generalize pSum d va vb p (t * cpb d) = Pg at hdd hcch ⊢
    generalize (r.get2 i n).toRat = rv at hres ⊢
    generalize (dd.get2 i n).toRat = dv at hdd hres hddabs ⊢
    have hdd1 : |dv| ≤ |P1 - Pg| + L := by
      have := abs_sub_abs_le_abs_sub dv (P1 - Pg)
      linarith
    have hcc1 : |cv| ≤ |Pg| + e := by
      have := abs_sub_abs_le_abs_sub cv Pg
      linarith
    have hsum : |dv| + |cv| ≤ |P1 - Pg| + L + |Pg| + e := by linarith
    have hm := mul_le_mul_of_nonneg_left hsum (by linarith : 0 ≤ 3 * Format.binary32.u)
    have htri : |rv - P1| ≤ |rv - (dv + cv)| + |dv - (P1 - Pg)| + |cv - Pg| := by
      have h1 : rv - P1 = (rv - (dv + cv)) + (dv - (P1 - Pg)) + (cv - Pg) := by ring
      have a1 := abs_add_le (rv - (dv + cv) + (dv - (P1 - Pg))) (cv - Pg)
      have a2 := abs_add_le (rv - (dv + cv)) (dv - (P1 - Pg))
      rw [h1]
      linarith
    linarith
  have hacc : accErr d va vb p (t + 1) = tripErr d va vb p t (accErr d va vb p t) := rfl
  have hchunk : |(r.get2 i n).toRat - pSum d va vb p (cpb d * (t + 1))| ≤ accErr d va vb p (t + 1) := by
    rw [hacc, e2]
    unfold tripErr
    rcases hr with hr1 | ⟨dd, hd1, hadd⟩
    · exact (hfused hr1).trans (le_max_left _ _)
    · exact (hsep dd hd1 hadd).trans' (le_of_eq (by ring)) |>.trans (le_max_right _ _)
  unfold LaneInv
  exact ⟨hrfin, hchunk, by rw [e2]; exact hstepb⟩

/-- **After the loop.** A lane that is `LaneInv` after all `stepsOf p` trips is a finite `float32`
within `errPart p` of `partSum p`. -/
theorem laneInv_final (d : Inst) (va vb : Nat → BF16) (p : Nat) {c : F32}
    (h : LaneInv d va vb p (stepsOf d p) c) :
    c.isFinite = true ∧ |c.toRat - partSum d va vb p| ≤ errPart d va vb p := by
  obtain ⟨hf, h1, h2⟩ := h
  refine ⟨hf, ?_⟩
  have e1 : pSum d va vb p (cpb d * stepsOf d p) = partSum d va vb p := rfl
  have e2 : pLift d va vb p (stepsOf d p) = partLift d va vb p := rfl
  rw [e1] at h1 h2
  rw [e2] at h2
  exact le_min h1 h2

end Claims.LoraShrink
