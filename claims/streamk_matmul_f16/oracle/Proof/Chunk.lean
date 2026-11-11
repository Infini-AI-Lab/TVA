import Claim
import TVA.Proof
import Proof.Acc

/-!
# `Proof.Chunk`

The numeric step of one `tl.dot` chunk and its safety proof (`chunkSafe`).
-/

/-!
# One `tl.dot` chunk step, numerically

One iteration of the inner chunk loop adds `64` products to every in-bounds lane of the
accumulator.  This file proves the numeric step on one lane (`stepLane`): from the tight bound

  `|c| ≤ dev (c0 + t) + fineOne c0 t`

on the accumulator lane `c` (and the rough `|c| ≤ 2 ^ 46` that the room hypothesis of the platform
rule needs), the new lane satisfies the same bound one chunk later,

  `|r| ≤ dev (c0 + t + 1) + fineOne c0 (t + 1)`,

using `DotLift.dotAdd_step` for `DotF16` and the exact reindexing of one `64`-chunk.
-/

namespace Claims.StreamKMatmul

open TVA

variable {P : Platform} {d : Inst} {VA VB : Nat → F16}

/-- `dev` is nonnegative (restated for the local `VA VB`). -/
private theorem dev_nonneg' (K t : Nat) : 0 ≤ dev VA VB K t := dev_nonneg VA VB K t

/-- **The numeric step of one chunk, on one lane.** -/
theorem stepLane {c0 t : Nat} {a : Tensor [128, 64] F16} {b : Tensor [64, 128] F16}
    {c r : Tensor [128, 128] F32} {ii jj : Fin 128}
    (hacc : Device.DotLiftAcc P ⟨.f16, .f16, .f32, .tf32, 0⟩
      Device.relDot16 Device.absDot16)
    (hr : P.dot ⟨.f16, .f16, .f32, .tf32, 0⟩ a b c r)
    (hmod : d.K % 64 = 0) (hv : c0 + t < steps d.K)
    (hva : ∀ k : Fin 64, DType.val? .f16 (a.get2 ii k) =
      some ((VA (64 * (c0 + t) + k.val)).toRat))
    (hvb : ∀ k : Fin 64, DType.val? .f16 (b.get2 k jj) =
      some ((VB (64 * (c0 + t) + k.val)).toRat))
    (hcg : (c.get2 ii jj).isFinite = true)
    (hta : |(c.get2 ii jj).toRat -
      (pre VA VB d.K (c0 + t) - pre VA VB d.K c0)| ≤ fineOne VA VB d.K c0 t)
    (htb : |(c.get2 ii jj).toRat| ≤
      dev VA VB d.K (c0 + t) + fineOne VA VB d.K c0 t)
    (hbig : |(c.get2 ii jj).toRat| ≤ 2 ^ 46) :
    ∃ w : F32, r.get2 ii jj = w ∧ good w ∧
      |w.toRat| ≤ dev VA VB d.K (c0 + t + 1) + fineOne VA VB d.K c0 (t + 1) := by
  -- the two per-lane values of the chunk
  set v := c0 + t with hvdef
  have hvq : v = c0 + t := rfl
  -- magnitude of the operands
  have hvaAbs : ∀ k : Fin 64, |((VA (64 * v + k.val)).toRat)| ≤ 65504 := fun k => by
    rw [← (DType.val?_f16.mp (hva k)).2]
    exact f16_toRat_abs_le (DType.val?_f16.mp (hva k)).1
  have hvbAbs : ∀ k : Fin 64, |((VB (64 * v + k.val)).toRat)| ≤ 65504 := fun k => by
    rw [← (DType.val?_f16.mp (hvb k)).2]
    exact f16_toRat_abs_le (DType.val?_f16.mp (hvb k)).1
  -- the lifted magnitude sum is at most `2 ^ 38`
  have hsumAbs : (∑ k : Fin 64,
      Device.liftMag .f16 ((VA (64 * v + k.val)).toRat) *
        Device.liftMag .f16 ((VB (64 * v + k.val)).toRat)) ≤ 2 ^ 38 :=
    sum_liftMag_le hvaAbs hvbAbs
  have hsum0 : (0 : ℚ) ≤ ∑ k : Fin 64,
      Device.liftMag .f16 ((VA (64 * v + k.val)).toRat) *
        Device.liftMag .f16 ((VB (64 * v + k.val)).toRat) :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _)
      (Device.liftMag_nonneg _ _)
  -- the room hypothesis
  have hroom := dot_room hbig hsumAbs hsum0
  have hrel : (0 : ℚ) ≤ Device.relDot16 64 := by unfold Device.relDot16; positivity
  have habs : (0 : ℚ) ≤ Device.absDot16 64 := by unfold Device.absDot16; positivity
  obtain ⟨hfin, herr⟩ :=
    DotLift.dotAdd_step (fusion := false) (a := a) (b := b) (c := c) (r := r)
      (z := tl.zeros [128, 128] .f32) hacc hrel habs
      (Or.inl hr) ii jj DType.val?_zero_f32 hva hvb hcg hroom
  -- the exact chunk sum and lifted magnitude of chunk `c0 + t`
  have hchunk : (∑ k : Fin 64,
      (VA (64 * v + k.val)).toRat * (VB (64 * v + k.val)).toRat) =
      chunkSum VA VB d.K v := by
    rw [Nat.mul_comm 64 v]
    exact sum_fin_chunk (K := d.K) (v := v) (f := fun k => (VA k).toRat * (VB k).toRat)
      hmod hv
  have hmag : (∑ k : Fin 64,
      Device.liftMag .f16 ((VA (64 * v + k.val)).toRat) *
        Device.liftMag .f16 ((VB (64 * v + k.val)).toRat)) = chunkMag VA VB d.K v := by
    rw [Nat.mul_comm 64 v]
    exact sum_fin_chunkMag (va := VA) (vb := VB) (K := d.K) hmod hv
  -- the previous bound, rewritten
  have hdev0 : (0 : ℚ) ≤ dev VA VB d.K v := dev_nonneg' (K := d.K) v
  
  have hcAbs : |(c.get2 ii jj).toRat| ≤ dev VA VB d.K v + fineOne VA VB d.K c0 t := htb
  -- the two error terms
  have hsumEq : ∑ k : Fin 64,
      (VA (64 * v + k.val)).toRat * (VB (64 * v + k.val)).toRat = chunkSum VA VB d.K v := hchunk
  have herr' : |(r.get2 ii jj).toRat -
      ((c.get2 ii jj).toRat + chunkSum VA VB d.K v)| ≤
      rho * (|(c.get2 ii jj).toRat| + chunkMag VA VB d.K v) + alpha := by
    have h := herr
    rw [hsumEq, hmag] at h
    exact h
  -- the running-sum step
  have hpre : pre VA VB d.K (v + 1) = pre VA VB d.K v + chunkSum VA VB d.K v :=
    pre_succ
  -- the new lane is within `fineOne (t + 1)` of the exact value
  have hmain : |(r.get2 ii jj).toRat - (pre VA VB d.K (v + 1) - pre VA VB d.K c0)| ≤
      fineOne VA VB d.K c0 (t + 1) := by
    have hsplit : |(r.get2 ii jj).toRat - (pre VA VB d.K (v + 1) - pre VA VB d.K c0)| ≤
        |(r.get2 ii jj).toRat - ((c.get2 ii jj).toRat + chunkSum VA VB d.K v)| +
          |(c.get2 ii jj).toRat - (pre VA VB d.K v - pre VA VB d.K c0)| := by
      have e : pre VA VB d.K (v + 1) - pre VA VB d.K c0 =
          ((c.get2 ii jj).toRat + chunkSum VA VB d.K v) +
            ((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat) := by
        rw [hpre]; ring
      rw [e]
      have h1 : (r.get2 ii jj).toRat -
          (((c.get2 ii jj).toRat + chunkSum VA VB d.K v) +
            ((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat)) =
          ((r.get2 ii jj).toRat - ((c.get2 ii jj).toRat + chunkSum VA VB d.K v)) +
            (-(((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat))) := by ring
      rw [h1]
      have habs := abs_add_le
        ((r.get2 ii jj).toRat - ((c.get2 ii jj).toRat + chunkSum VA VB d.K v))
        (-(((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat)))
      have hx : |(-(((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat)))| =
          |(c.get2 ii jj).toRat - (pre VA VB d.K v - pre VA VB d.K c0)| := by
        rw [abs_neg, abs_sub_comm]
      rw [hx] at habs
      exact habs
    have hstep : rho * (|(c.get2 ii jj).toRat| + chunkMag VA VB d.K v) + alpha +
        fineOne VA VB d.K c0 t ≤
        (1 + rho) * fineOne VA VB d.K c0 t +
          (rho * (dev VA VB d.K (c0 + t) + chunkMag VA VB d.K (c0 + t)) + alpha) := by
      have hmul := mul_le_mul_of_nonneg_left htb rho_nonneg
      simp only [hvdef] at hmul ⊢
      ring_nf at hmul ⊢
      linarith
    have hrec : (1 + rho) * fineOne VA VB d.K c0 t +
        (rho * (dev VA VB d.K (c0 + t) + chunkMag VA VB d.K (c0 + t)) + alpha) =
        fineOne VA VB d.K c0 (t + 1) := by
      rw [fineOne_step]
    simp only [hvdef] at hrec hstep herr' hsplit hta
    linarith
  -- the exact value itself
  have hD : |pre VA VB d.K (v + 1) - pre VA VB d.K c0| ≤ dev VA VB d.K (v + 1) :=
    le_dev (va := VA) (vb := VB) (K := d.K) (s := c0) (t := v + 1) (by omega)
  have hv1 : v + 1 = c0 + t + 1 := by omega
  refine ⟨r.get2 ii jj, rfl, hfin, ?_⟩
  have htri2 : |(r.get2 ii jj).toRat| ≤
      |(r.get2 ii jj).toRat - (pre VA VB d.K (v + 1) - pre VA VB d.K c0)| +
        |pre VA VB d.K (v + 1) - pre VA VB d.K c0| := by
    have h := abs_sub_abs_le_abs_sub (r.get2 ii jj).toRat
      (pre VA VB d.K (v + 1) - pre VA VB d.K c0)
    linarith [neg_le_abs (r.get2 ii jj).toRat]
  rw [hv1] at hmain hD htri2 ⊢
  linarith

end Claims.StreamKMatmul


/-!
# One chunk iteration of the inner loop

`chunkSafe` is the `Safe` statement of one iteration of the innermost loop: two masked loads of
`a` and `b` blocks and one `tl.dot` into the accumulator.  The memory is only read, and the reads
lie inside `inElems`, where `m` still agrees with `init`, so the loaded values are the ones the
`ReadsA`/`ReadsB` of `Proof/Q.lean` name.  The numeric side is `DotLift.dotAdd_step`, with the
room and magnitude bounds of `Proof/Mag.lean`, and the accumulator update is the recursion
`fineOne_step`.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst} {P : Platform}

/-- **`float32` to `float16` conversion is always related to `cvt`.**  The `FloatRel` clause for a
narrowing of a `float32` to `float16` holds by definition. -/
theorem convRel_cvt_f32_f16 (P : Platform) (x : F32) : ConvRel P .f32 .f16 x (FloatV.cvt x) := by
  show FloatRel P .binary32 .binary16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide : ¬ (Format.binary32 = Format.binary16))]
  exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, FloatV.NaNEq.refl _⟩

variable {VA VB : Nat → F16}

/-- **The numeric step of one chunk, on one lane** (tight form): the new lane is within
`fineOne (t + 1)` of the exact partial sum. -/
theorem stepLaneTight {c0 t : Nat} {a : Tensor [128, 64] F16} {b : Tensor [64, 128] F16}
    {c r : Tensor [128, 128] F32} {ii jj : Fin 128}
    (hacc : Device.DotLiftAcc P ⟨.f16, .f16, .f32, .tf32, 0⟩
      Device.relDot16 Device.absDot16)
    (hr : P.dot ⟨.f16, .f16, .f32, .tf32, 0⟩ a b c r)
    (hmod : d.K % 64 = 0) (hv : c0 + t < steps d.K)
    (hva : ∀ k : Fin 64, DType.val? .f16 (a.get2 ii k) =
      some ((VA (64 * (c0 + t) + k.val)).toRat))
    (hvb : ∀ k : Fin 64, DType.val? .f16 (b.get2 k jj) =
      some ((VB (64 * (c0 + t) + k.val)).toRat))
    (hcg : (c.get2 ii jj).isFinite = true)
    (hta : |(c.get2 ii jj).toRat -
      (pre VA VB d.K (c0 + t) - pre VA VB d.K c0)| ≤ fineOne VA VB d.K c0 t)
    (htb : |(c.get2 ii jj).toRat| ≤
      dev VA VB d.K (c0 + t) + fineOne VA VB d.K c0 t)
    (hbig : |(c.get2 ii jj).toRat| ≤ 2 ^ 46) :
    ∃ w : F32, r.get2 ii jj = w ∧ good w ∧
      |w.toRat - (pre VA VB d.K (c0 + t + 1) - pre VA VB d.K c0)| ≤
        fineOne VA VB d.K c0 (t + 1) := by
  -- the two per-lane values of the chunk
  set v := c0 + t with hvdef
  have hvq : v = c0 + t := rfl
  -- magnitude of the operands
  have hvaAbs : ∀ k : Fin 64, |((VA (64 * v + k.val)).toRat)| ≤ 65504 := fun k => by
    rw [← (DType.val?_f16.mp (hva k)).2]
    exact f16_toRat_abs_le (DType.val?_f16.mp (hva k)).1
  have hvbAbs : ∀ k : Fin 64, |((VB (64 * v + k.val)).toRat)| ≤ 65504 := fun k => by
    rw [← (DType.val?_f16.mp (hvb k)).2]
    exact f16_toRat_abs_le (DType.val?_f16.mp (hvb k)).1
  -- the lifted magnitude sum is at most `2 ^ 38`
  have hsumAbs : (∑ k : Fin 64,
      Device.liftMag .f16 ((VA (64 * v + k.val)).toRat) *
        Device.liftMag .f16 ((VB (64 * v + k.val)).toRat)) ≤ 2 ^ 38 :=
    sum_liftMag_le hvaAbs hvbAbs
  have hsum0 : (0 : ℚ) ≤ ∑ k : Fin 64,
      Device.liftMag .f16 ((VA (64 * v + k.val)).toRat) *
        Device.liftMag .f16 ((VB (64 * v + k.val)).toRat) :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _)
      (Device.liftMag_nonneg _ _)
  -- the room hypothesis
  have hroom := dot_room hbig hsumAbs hsum0
  have hrel : (0 : ℚ) ≤ Device.relDot16 64 := by unfold Device.relDot16; positivity
  have habs : (0 : ℚ) ≤ Device.absDot16 64 := by unfold Device.absDot16; positivity
  obtain ⟨hfin, herr⟩ :=
    DotLift.dotAdd_step (fusion := false) (a := a) (b := b) (c := c) (r := r)
      (z := tl.zeros [128, 128] .f32) hacc hrel habs
      (Or.inl hr) ii jj DType.val?_zero_f32 hva hvb hcg hroom
  -- the exact chunk sum and lifted magnitude of chunk `c0 + t`
  have hchunk : (∑ k : Fin 64,
      (VA (64 * v + k.val)).toRat * (VB (64 * v + k.val)).toRat) =
      chunkSum VA VB d.K v := by
    rw [Nat.mul_comm 64 v]
    exact sum_fin_chunk (K := d.K) (v := v) (f := fun k => (VA k).toRat * (VB k).toRat)
      hmod hv
  have hmag : (∑ k : Fin 64,
      Device.liftMag .f16 ((VA (64 * v + k.val)).toRat) *
        Device.liftMag .f16 ((VB (64 * v + k.val)).toRat)) = chunkMag VA VB d.K v := by
    rw [Nat.mul_comm 64 v]
    exact sum_fin_chunkMag (va := VA) (vb := VB) (K := d.K) hmod hv
  -- the previous bound, rewritten
  have hdev0 : (0 : ℚ) ≤ dev VA VB d.K v := dev_nonneg VA VB d.K v

  have hcAbs : |(c.get2 ii jj).toRat| ≤ dev VA VB d.K v + fineOne VA VB d.K c0 t := htb
  -- the two error terms
  have hsumEq : ∑ k : Fin 64,
      (VA (64 * v + k.val)).toRat * (VB (64 * v + k.val)).toRat = chunkSum VA VB d.K v := hchunk
  have herr' : |(r.get2 ii jj).toRat -
      ((c.get2 ii jj).toRat + chunkSum VA VB d.K v)| ≤
      rho * (|(c.get2 ii jj).toRat| + chunkMag VA VB d.K v) + alpha := by
    have h := herr
    rw [hsumEq, hmag] at h
    exact h
  -- the running-sum step
  have hpre : pre VA VB d.K (v + 1) = pre VA VB d.K v + chunkSum VA VB d.K v :=
    pre_succ
  -- the new lane is within `fineOne (t + 1)` of the exact value
  have hmain : |(r.get2 ii jj).toRat - (pre VA VB d.K (v + 1) - pre VA VB d.K c0)| ≤
      fineOne VA VB d.K c0 (t + 1) := by
    have hsplit : |(r.get2 ii jj).toRat - (pre VA VB d.K (v + 1) - pre VA VB d.K c0)| ≤
        |(r.get2 ii jj).toRat - ((c.get2 ii jj).toRat + chunkSum VA VB d.K v)| +
          |(c.get2 ii jj).toRat - (pre VA VB d.K v - pre VA VB d.K c0)| := by
      have e : pre VA VB d.K (v + 1) - pre VA VB d.K c0 =
          ((c.get2 ii jj).toRat + chunkSum VA VB d.K v) +
            ((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat) := by
        rw [hpre]; ring
      rw [e]
      have h1 : (r.get2 ii jj).toRat -
          (((c.get2 ii jj).toRat + chunkSum VA VB d.K v) +
            ((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat)) =
          ((r.get2 ii jj).toRat - ((c.get2 ii jj).toRat + chunkSum VA VB d.K v)) +
            (-(((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat))) := by ring
      rw [h1]
      have habs := abs_add_le
        ((r.get2 ii jj).toRat - ((c.get2 ii jj).toRat + chunkSum VA VB d.K v))
        (-(((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat)))
      have hx : |(-(((pre VA VB d.K v - pre VA VB d.K c0) - (c.get2 ii jj).toRat)))| =
          |(c.get2 ii jj).toRat - (pre VA VB d.K v - pre VA VB d.K c0)| := by
        rw [abs_neg, abs_sub_comm]
      rw [hx] at habs
      exact habs
    have hstep : rho * (|(c.get2 ii jj).toRat| + chunkMag VA VB d.K v) + alpha +
        fineOne VA VB d.K c0 t ≤
        (1 + rho) * fineOne VA VB d.K c0 t +
          (rho * (dev VA VB d.K (c0 + t) + chunkMag VA VB d.K (c0 + t)) + alpha) := by
      have hmul := mul_le_mul_of_nonneg_left htb rho_nonneg
      simp only [hvdef] at hmul ⊢
      ring_nf at hmul ⊢
      linarith
    have hrec : (1 + rho) * fineOne VA VB d.K c0 t +
        (rho * (dev VA VB d.K (c0 + t) + chunkMag VA VB d.K (c0 + t)) + alpha) =
        fineOne VA VB d.K c0 (t + 1) := by
      rw [fineOne_step]
    simp only [hvdef] at hrec hstep herr' hsplit hta
    linarith
  have hv1 : v + 1 = c0 + t + 1 := by omega
  refine ⟨r.get2 ii jj, rfl, hfin, ?_⟩
  simpa only [hv1] using hmain

/-- **One chunk iteration is safe** and advances the accumulator invariant.  The memory is only
read: the program's reads are inside `inElems`, and `m` agrees with `init` there (the `c` writes
of the program are disjoint from the inputs), so the loaded values are the ones `ReadsA`/`ReadsB`
name. -/
theorem chunkSafe (hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {m : Mem} (hpres : Pres d m) (hfr : ∀ x ∈ inElems d, m x = init x)
    {i u t : Nat} {pm pn kk : I32}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hkk : kk.toInt = ((c0 d i u + t : Nat) : Int))
    (hu : u < T d) (ht : t < 128) (hks : c0 d i u + t < steps d.K)
    {acc : Tensor [128, 128] F32} (hacc : AccI d init i u t acc) :
    Safe ((launch d).env P) (inElems d) ∅ (fun acc' m' => m' = m ∧ AccI d init i u (t + 1) acc')
      m (ChunkBody d (offs1 pm 128) (offs1 pn 128) kk acc) := by
  have hu' : u < tilesM d * tilesN d := hu
  have hmod : d.K % 64 = 0 := by have := hd.K_mul; omega
  dsimp only [ChunkBody, ptrsA, maskA, ptrsB, maskB]
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨FloatV.cvt ((tl.full [128, 64] (FloatV.ofPat 0).litF32).get i),
      convRel_cvt_f32_f16 P _⟩
  · intro l1 hl1
    refine safe_load ?_ ?_ ?_
    · exact read_of_addrs hpre.1 hpres aBytes_sub (aAddrs hd hpm hkk hu' hks)
        (fun i hi => aLt_len hd hu' hks i hi)
    · exact aAccBytes_sub hd hpm hkk hu' hks
    · intro aT haT
      refine safe_convert ?_ ?_
      · intro i
        exact ⟨FloatV.cvt ((tl.full [64, 128] (FloatV.ofPat 0).litF32).get i),
          convRel_cvt_f32_f16 P _⟩
      · intro l2 hl2
        refine safe_load ?_ ?_ ?_
        · exact read_of_addrs hpre.2.1 hpres bBytes_sub
            (bAddrs hd hkk hpn hks (pidNOf_lt' hd hu'))
            (fun i hi => bLt_len hd hks (pidNOf_lt' hd hu') i hi)
        · exact bAccBytes_sub hd hkk hpn hks (pidNOf_lt' hd hu')
        · intro bT hbT
          refine safe_last ?_
          refine safe_dot (cfg := { a := .f16, b := .f16, acc := .f32 }) ?_ ?_
          · exact hP.dotTotal { a := .f16, b := .f16, acc := .f32 } aT bT acc
          · intro r hr
            refine Safe.ret ⟨rfl, ?_⟩
            intro ii jj hi hj va vb hva hvb
            -- the previous lane value and its three bounds
            obtain ⟨v1, hv1, hg1, hta⟩ := accI_cell hacc hi hj hva hvb
            obtain ⟨v2, hv2, hg2, htb⟩ := accI_bnd hacc hi hj hva hvb
            obtain ⟨v3, hv3, hg3, hbig⟩ := accI_mag hd hacc (Nat.le_of_lt ht) hi hj hva hvb
            have h21 : v2 = v1 := by rw [← hv2, hv1]
            have h31 : v3 = v1 := by rw [← hv3, hv1]
            have hcg : (acc.get2 ii jj).isFinite = true := by rw [hv1]; exact hg1
            have hta' : |(acc.get2 ii jj).toRat -
                (pre va vb d.K (c0 d i u + t) - pre va vb d.K (c0 d i u))| ≤
                fineOne va vb d.K (c0 d i u) t := by
              rw [hv1]; exact hta
            have htb' : |(acc.get2 ii jj).toRat| ≤
                dev va vb d.K (c0 d i u + t) + fineOne va vb d.K (c0 d i u) t := by
              rw [hv1]; simpa only [h21] using htb
            have hbig' : |(acc.get2 ii jj).toRat| ≤ 2 ^ 46 := by
              rw [hv1]; simpa only [h31] using hbig
            -- the chunk index is valid
            have hkK : ∀ k : Fin 64, (c0 d i u + t) * 64 + k.val < d.K := by
              intro k
              have hs : c0 d i u + t < (d.K + 63) / 64 := hks
              have hk64 : k.val < 64 := k.isLt
              omega
            -- the values loaded for `a` are the ones `ReadsA` names
            have hvaL : ∀ k : Fin 64, DType.val? .f16 (aT.get2 ii k) =
                some ((va (64 * (c0 d i u + t) + k.val)).toRat) := by
              intro k
              show DType.val? .f16 (aT.get (ii, k, ())) =
                some ((va (64 * (c0 d i u + t) + k.val)).toRat)
              have hk : (c0 d i u + t) * 64 + k.val < d.K := hkK k
              have hAd := aAddrs (d := d) hd hpm hkk hu' hks
              have hen : (decide (pidMOf d u * 128 + ii.val < d.M) &&
                  decide ((c0 d i u + t) * 64 + k.val < d.K)) = true := by
                simp only [Bool.and_eq_true, decide_eq_true_eq]
                exact ⟨hi, hk⟩
              have haddr : tl.addrs
                  (ptrsA (d.a.ptr 0) (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.stride_am)
                    (IntV.ofNat d.stride_ak))
                  (some (maskA (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.M)
                    (IntV.ofNat d.K)))
                  (ii, k, ()) =
                  some (d.a.ptr (((pidMOf d u * 128 + ii.val) * d.stride_am +
                    ((c0 d i u + t) * 64 + k.val) * d.stride_ak : Nat) : Int)).addr := by
                rw [hAd (ii, k, ())]
                simp only [hen, ite_true]
              have hmem : ((pidMOf d u * 128 + ii.val) * d.stride_am +
                  ((c0 d i u + t) * 64 + k.val) * d.stride_ak) ∈ aIdx d :=
                aImg_sub hd hu' hks ⟨(ii, k, ()), hen, rfl⟩
              have hsub := Buf.span_sub_elems d.a hmem
              have hmread : m.read .f16
                  (d.a.ptr (((pidMOf d u * 128 + ii.val) * d.stride_am +
                    ((c0 d i u + t) * 64 + k.val) * d.stride_ak : Nat) : Int)).addr =
                  some (aT.get (ii, k, ())) :=
                (haT (ii, k, ())).1 _ haddr
              have hcongr : m.read .f16
                  (d.a.ptr (((pidMOf d u * 128 + ii.val) * d.stride_am +
                    ((c0 d i u + t) * 64 + k.val) * d.stride_ak : Nat) : Int)).addr =
                  init.read .f16
                  (d.a.ptr (((pidMOf d u * 128 + ii.val) * d.stride_am +
                    ((c0 d i u + t) * 64 + k.val) * d.stride_ak : Nat) : Int)).addr :=
                Mem.read_congr (fun x hx => hfr x (Or.inl (hsub hx)))
              have hinit : init.read .f16
                  (d.a.ptr (((pidMOf d u * 128 + ii.val) * d.stride_am +
                    ((c0 d i u + t) * 64 + k.val) * d.stride_ak : Nat) : Int)).addr =
                  some (va ((c0 d i u + t) * 64 + k.val)) := (hva _ hk).1
              have hsome : some (aT.get (ii, k, ())) =
                  some (va ((c0 d i u + t) * 64 + k.val)) := by
                rw [← hmread, hcongr]
                exact hinit
              have heq : aT.get (ii, k, ()) = va ((c0 d i u + t) * 64 + k.val) :=
                Option.some.inj hsome
              have hfin : (aT.get (ii, k, ())).isFinite = true := by
                rw [heq]; exact (hva _ hk).2
              have hrat : (aT.get (ii, k, ())).toRat =
                  (va (64 * (c0 d i u + t) + k.val)).toRat := by
                rw [heq, Nat.mul_comm (c0 d i u + t) 64]
              exact DType.val?_f16.mpr ⟨hfin, hrat⟩
            -- the values loaded for `b` are the ones `ReadsB` names
            have hvbL : ∀ k : Fin 64, DType.val? .f16 (bT.get2 k jj) =
                some ((vb (64 * (c0 d i u + t) + k.val)).toRat) := by
              intro k
              show DType.val? .f16 (bT.get (k, jj, ())) =
                some ((vb (64 * (c0 d i u + t) + k.val)).toRat)
              have hk : (c0 d i u + t) * 64 + k.val < d.K := hkK k
              have hBd := bAddrs (d := d) hd hkk hpn hks (pidNOf_lt' hd hu')
              have hen : (decide ((c0 d i u + t) * 64 + k.val < d.K) &&
                  decide (pidNOf d u * 128 + jj.val < d.N)) = true := by
                simp only [Bool.and_eq_true, decide_eq_true_eq]
                exact ⟨hk, hj⟩
              have haddr : tl.addrs
                  (ptrsB (d.b.ptr 0) (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.stride_bk)
                    (IntV.ofNat d.stride_bn))
                  (some (maskB (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.K)
                    (IntV.ofNat d.N)))
                  (k, jj, ()) =
                  some (d.b.ptr ((((c0 d i u + t) * 64 + k.val) * d.stride_bk +
                    (pidNOf d u * 128 + jj.val) * d.stride_bn : Nat) : Int)).addr := by
                rw [hBd (k, jj, ())]
                simp only [hen, ite_true]
              have hmem : (((c0 d i u + t) * 64 + k.val) * d.stride_bk +
                  (pidNOf d u * 128 + jj.val) * d.stride_bn) ∈ bIdx d :=
                bImg_sub hd hks (pidNOf_lt' hd hu') ⟨(k, jj, ()), hen, rfl⟩
              have hsub := Buf.span_sub_elems d.b hmem
              have hmread : m.read .f16
                  (d.b.ptr ((((c0 d i u + t) * 64 + k.val) * d.stride_bk +
                    (pidNOf d u * 128 + jj.val) * d.stride_bn : Nat) : Int)).addr =
                  some (bT.get (k, jj, ())) :=
                (hbT (k, jj, ())).1 _ haddr
              have hcongr : m.read .f16
                  (d.b.ptr ((((c0 d i u + t) * 64 + k.val) * d.stride_bk +
                    (pidNOf d u * 128 + jj.val) * d.stride_bn : Nat) : Int)).addr =
                  init.read .f16
                  (d.b.ptr ((((c0 d i u + t) * 64 + k.val) * d.stride_bk +
                    (pidNOf d u * 128 + jj.val) * d.stride_bn : Nat) : Int)).addr :=
                Mem.read_congr (fun x hx => hfr x (Or.inr (hsub hx)))
              have hinit : init.read .f16
                  (d.b.ptr ((((c0 d i u + t) * 64 + k.val) * d.stride_bk +
                    (pidNOf d u * 128 + jj.val) * d.stride_bn : Nat) : Int)).addr =
                  some (vb ((c0 d i u + t) * 64 + k.val)) := (hvb _ hk).1
              have hsome : some (bT.get (k, jj, ())) =
                  some (vb ((c0 d i u + t) * 64 + k.val)) := by
                rw [← hmread, hcongr]
                exact hinit
              have heq : bT.get (k, jj, ()) = vb ((c0 d i u + t) * 64 + k.val) :=
                Option.some.inj hsome
              have hfin : (bT.get (k, jj, ())).isFinite = true := by
                rw [heq]; exact (hvb _ hk).2
              have hrat : (bT.get (k, jj, ())).toRat =
                  (vb (64 * (c0 d i u + t) + k.val)).toRat := by
                rw [heq, Nat.mul_comm (c0 d i u + t) 64]
              exact DType.val?_f16.mpr ⟨hfin, hrat⟩
            -- the numeric step of the chunk
            simpa only [Nat.add_assoc] using
              stepLaneTight (VA := va) (VB := vb) (c0 := c0 d i u) (t := t)
                (a := aT) (b := bT) (c := acc) (r := r) (ii := ii) (jj := jj)
                (hP.dotF16 .tf32) hr hmod hks hvaL hvbL hcg hta' htb' hbig'

end Claims.StreamKMatmul

