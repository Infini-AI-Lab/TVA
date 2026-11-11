import Claim
import Proof.Defs
import Proof.Num
import Proof.Ptr
import TVA.Proof

/-!
# Loop

The loop over the chunk's tokens: the pointers after `t` steps, and the accumulator's invariant, one `tl.dot` step at a time.
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000
/-! ## XVal -/

/-!
# The `x` value clause after one loop iteration

`Inv` carries, for a partial token window, a table `xh` of exact `ℚ` values of the `x` tile with
zeros outside the chunk.  One iteration extends the window by `BK` tokens: this file shows that the
extended table `xh'` (built from the loaded tile `xt`) satisfies the same clause for the window
`(t+1) * BK`.
-/

open TVA

set_option maxHeartbeats 800000

/-- Extending the `x` value table by one step of the loop. -/
theorem xClause {d : Inst} (_hd : d.ok) {init : Mem}
    {i : Fin (launch d).size} {t : Nat}
    {xh xh' : Nat → Fin BM → ℚ} {xt : Tensor [BM, BK] DType.bf16.Val}
    {mb : Fin BM → Fin BK → Bool} {Nch : Nat}
    (hNch : Nch = endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk)
    (hmask : ∀ (a : Fin BM) (j : Fin BK),
      mb a j = (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
        decide ((j : Nat) < Nch - t * BK)))
    (hold : ∀ (s : Nat) (a : Fin BM), s < t * BK →
      ((chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim →
          (xh s a : ℝ) = xV init d (chunkOf d i * d.chunk + s) (headOf d i)
            (pmOf d i * BM + (a : Nat))) ∧
        (¬(chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) → xh s a = 0)))
    (hagree : ∀ (s : Nat) (a : Fin BM), s < t * BK → xh' s a = xh s a)
    (hnew : ∀ (a : Fin BM) (j : Fin BK),
      xh' (t * BK + (j : Nat)) a = (xt.get2 a j).toRat)
    (hread : ∀ (a : Fin BM) (j : Fin BK), mb a j = true →
      ∃ v, init.read .bf16 ((d.x.ptr (xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (headOf d i) (pmOf d i * BM + (a : Nat)))).addr) = some v ∧ xt.get2 a j = v)
    (hzero : ∀ (a : Fin BM) (j : Fin BK), mb a j = false →
      xt.get2 a j = (FloatV.ofPat (0 : Nat) : BF16)) :
    ∀ (s : Nat) (a : Fin BM), s < (t + 1) * BK →
      ((chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim →
          (xh' s a : ℝ) = xV init d (chunkOf d i * d.chunk + s) (headOf d i)
            (pmOf d i * BM + (a : Nat))) ∧
        (¬(chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) → xh' s a = 0)) := by
  intro s a hs
  by_cases hsold : s < t * BK
  · obtain ⟨h1, h2⟩ := hold s a hsold
    have hxh : xh' s a = xh s a := hagree s a hsold
    exact ⟨fun hc => by rw [hxh]; exact h1 hc, fun hc => by rw [hxh]; exact h2 hc⟩
  · -- `s` lies in the new step: `s = t * BK + j` with `j < BK`
    have hmul : (t + 1) * BK = t * BK + BK := by ring
    have hge : t * BK ≤ s := by omega
    have hjlt : s - t * BK < BK := by omega
    have hsj' : s = t * BK + (s - t * BK) := by omega
    obtain ⟨j, hsj⟩ : ∃ j : Fin BK, s = t * BK + (j : Nat) :=
      ⟨⟨s - t * BK, hjlt⟩, by simpa using hsj'⟩
    rw [hsj]
    by_cases hb : mb a j = true
    · -- the lane is enabled: `hread` gives the value the tile holds
      have hm := hmask a j
      rw [hb] at hm
      have hboth : (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
          decide ((j : Nat) < Nch - t * BK)) = true := hm.symm
      rw [Bool.and_eq_true] at hboth
      obtain ⟨hpmd, hjd⟩ := hboth
      have hpm : pmOf d i * BM + (a : Nat) < d.headdim := of_decide_eq_true hpmd
      have hjk : (j : Nat) < Nch - t * BK := of_decide_eq_true hjd
      obtain ⟨v, hvread, hvxt⟩ := hread a j hb
      have hnewj : xh' (t * BK + (j : Nat)) a = v.toRat := by
        rw [hnew a j, hvxt]
      have hxV : xV init d (chunkOf d i * d.chunk + (t * BK + (j : Nat))) (headOf d i)
          (pmOf d i * BM + (a : Nat)) = ((v.toRat : ℚ) : ℝ) := by
        unfold xV valB
        rw [hvread]
      -- the token is a token of the chunk: `j < Nch - t*BK` and `Nch`'s definition
      have hcond : chunkOf d i * d.chunk + (t * BK + (j : Nat)) ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim := by
        refine ⟨?_, hpm⟩
        have hxNch : t * BK + (j : Nat) < Nch := by omega
        have hpos : 0 < Nch := by omega
        have hcle : chunkOf d i * d.chunk ≤ endTok d (chunkOf d i) + 1 := by
          by_contra hcon
          rw [hNch] at hpos
          omega
        have hsub : t * BK + (j : Nat) <
            endTok d (chunkOf d i) + 1 - chunkOf d i * d.chunk := by
          rw [← hNch]
          exact hxNch
        omega
      refine ⟨fun _ => ?_, fun hneg => absurd hcond hneg⟩
      rw [hnewj]
      exact hxV.symm
    · -- the lane is masked off: the tile holds the zero pattern
      have hbf : mb a j = false := by
        cases h : mb a j
        · rfl
        · exact absurd h hb
      have hm := hmask a j
      rw [hbf] at hm
      have hnot : ¬ (pmOf d i * BM + (a : Nat) < d.headdim ∧ (j : Nat) < Nch - t * BK) := by
        intro hcon
        have hA : decide (pmOf d i * BM + (a : Nat) < d.headdim) = true := decide_eq_true hcon.1
        have hB : decide ((j : Nat) < Nch - t * BK) = true := decide_eq_true hcon.2
        rw [hA, hB] at hm
        simp at hm
      have hxh0 : xh' (t * BK + (j : Nat)) a = 0 := by
        rw [hnew a j, hzero a j hbf]
        exact BF16_zero_toRat
      have hcf : ¬ (chunkOf d i * d.chunk + (t * BK + (j : Nat)) ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) := by
        rintro ⟨hle', hpm⟩
        apply hnot
        refine ⟨hpm, ?_⟩
        have hxNch : t * BK + (j : Nat) < Nch := by
          rw [hNch]
          omega
        omega
      exact ⟨fun hc => absurd hc hcf, fun _ => hxh0⟩



/-!
# The accumulation loop's state and invariant

The `for` loop of the kernel accumulates, in `acc : Tensor [BM, BN] f32`, the partial sums of
`Σ_t x[t, p] · b[t, n]` over the tokens `t` of the chunk.  This file fixes

* `LoopSt`, the loop's carried state (five pointer tiles and the accumulator), and
* `Inv`, the loop invariant: the carried pointers address exactly the tiles of iteration `t`,
  the two `ℚ`-valued families `xh`, `bh` hold the operands actually fed to `tl.dot`, and the
  accumulator is within `DotLoop.bound` of the exact partial sums.
-/

open TVA

set_option maxHeartbeats 800000

/-- The loop's carried state: the five pointer tiles (advanced by `BK` along the token axis at
each iteration) and the `float32` accumulator tile. -/
abbrev LoopSt (_d : Inst) : Type :=
  Tensor [BM, BK] (Ptr DType.bf16) × Tensor [BK, BN] (Ptr DType.bf16) × Tensor [BK] (Ptr DType.f32) ×
  Tensor [BK] (Ptr DType.f32) × Tensor [BK] (Ptr DType.i32) × Tensor [BM, BN] (DType.f32.Val)

/-- The exact `tl.dot` product of one iteration, for one accumulator element. -/
def Sf (xh : Nat → Fin BM → ℚ) (bh : Nat → Fin BN → ℚ) (a : Fin BM) (n : Fin BN) (t : Nat) : ℚ :=
  ∑ j : Fin BK, xh (t * BK + (j : Nat)) a * bh (t * BK + (j : Nat)) n

/-- The corresponding sum of the tensor core's lifted magnitudes (the magnitude `tl.dot`'s
error is relative to: `|q|`, or the smallest normal for a subnormal operand — `DotLiftAcc`). -/
def Af (xh : Nat → Fin BM → ℚ) (bh : Nat → Fin BN → ℚ) (a : Fin BM) (n : Fin BN) (t : Nat) : ℚ :=
  ∑ j : Fin BK,
    Device.liftMag .bf16 (xh (t * BK + (j : Nat)) a) *
      Device.liftMag .bf16 (bh (t * BK + (j : Nat)) n)

/-- The loop invariant at iteration `t` (after `t` full iterations). -/
def Inv (d : Inst) (init : Mem) (i : Fin (launch d).size) (t : Nat)
    (st : LoopSt d) (m : Mem) : Prop :=
  m = init ∧
  ∃ xh : Nat → Fin BM → ℚ, ∃ bh : Nat → Fin BN → ℚ,
    -- the carried pointers address the tiles of iteration `t`
    (∀ (a : Fin BM) (j : Fin BK),
      (st.1.get2 a j).addr =
        (d.x.ptr (xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
          (headOf d i) (pmOf d i * BM + (a : Nat)))).addr) ∧
    (∀ (j : Fin BK) (n : Fin BN),
      (st.2.1.get2 j n).addr =
        (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
          (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr) ∧
    (∀ j : Fin BK,
      (st.2.2.1.get1 j).addr =
        (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr) ∧
    (∀ j : Fin BK,
      (st.2.2.2.1.get1 j).addr =
        (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr) ∧
    (∀ j : Fin BK,
      (st.2.2.2.2.1.get1 j).addr =
        (d.seqIdx.ptr (chunkOf d i * d.chunk + (t * BK + (j : Nat)))).addr) ∧
    -- what the `x` operand holds, for the tokens processed so far
    (∀ (s : Nat) (a : Fin BM), s < t * BK →
      ((chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim →
        (xh s a : ℝ) = xV init d (chunkOf d i * d.chunk + s) (headOf d i)
          (pmOf d i * BM + (a : Nat))) ∧
       (¬(chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) → xh s a = 0))) ∧
    -- what the `b` operand holds, for the tokens processed so far
    (Vals init d → ∀ (s : Nat) (n : Fin BN), s < t * BK →
      (((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |(bh s n : ℝ) - bV init d (chunkOf d i * d.chunk + s) (headOf d i)
            (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → bh s n = 0))) ∧
    -- the accumulator is finite, and within `DotLoop.bound` of the exact partial sums
    (Vals init d →
      (∀ (a : Fin BM) (n : Fin BN), (st.2.2.2.2.2.get2 a n).isFinite = true) ∧
      (∀ (a : Fin BM) (n : Fin BN),
        |(((st.2.2.2.2.2.get2 a n).toRat : ℚ) : ℝ) -
            ((∑ t' ∈ Finset.range t, Sf xh bh a n t' : ℚ) : ℝ)| ≤
          ((DotLoop.bound rho alp t (∑ t' ∈ Finset.range t, Af xh bh a n t') : ℚ) : ℝ)))

/-- **The magnitude bound on the accumulated operands.**  The absolute-value sums of the first
`t` iterations are bounded by the claim's `dotMag` (which is `≤ 2^120` by `Vals`). -/
theorem aSum_le_dotMag {d : Inst} {init : Mem} {i : Fin (launch d).size} (_hd : d.ok)
    {t : Nat} (_ht : t ≤ steps d (chunkOf d i))
    (xh : Nat → Fin BM → ℚ) (bh : Nat → Fin BN → ℚ)
    (hx : ∀ (s : Nat) (a : Fin BM), s < t * BK →
      ((chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim →
        (xh s a : ℝ) = xV init d (chunkOf d i * d.chunk + s) (headOf d i)
          (pmOf d i * BM + (a : Nat))) ∧
       (¬(chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) ∧
          pmOf d i * BM + (a : Nat) < d.headdim) → xh s a = 0)))
    (hb : ∀ (s : Nat) (n : Fin BN), s < t * BK →
      (((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate →
        |(bh s n : ℝ) - bV init d (chunkOf d i * d.chunk + s) (headOf d i)
            (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat))) ∧
       (¬((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
          pnOf d i * BN + (n : Nat) < d.dstate) → bh s n = 0)))
    (_hV : Vals init d) :
    ∀ (a : Fin BM) (n : Fin BN),
      ((∑ t' ∈ Finset.range t, Af xh bh a n t' : ℚ) : ℝ) ≤
        dotMag init d (chunkOf d i) (headOf d i)
          (pmOf d i * BM + (a : Nat)) (pnOf d i * BN + (n : Nat)) := by
  intro a n
  -- the sums over the iterations' steps and over the token positions agree
  have hid : (∑ t' ∈ Finset.range t, Af xh bh a n t')
      = ∑ s ∈ Finset.range (t * BK),
          Device.liftMag .bf16 (xh s a) * Device.liftMag .bf16 (bh s n) := by
    unfold Af
    have hfin : ∀ t' : Nat,
        (∑ j : Fin BK, Device.liftMag .bf16 (xh (t' * BK + (j : Nat)) a) *
            Device.liftMag .bf16 (bh (t' * BK + (j : Nat)) n))
        = ∑ j ∈ Finset.range BK, Device.liftMag .bf16 (xh (t' * BK + j) a) *
            Device.liftMag .bf16 (bh (t' * BK + j) n) := by
      intro t'
      exact Fin.sum_univ_eq_sum_range
        (fun k => Device.liftMag .bf16 (xh (t' * BK + k) a) *
          Device.liftMag .bf16 (bh (t' * BK + k) n)) BK
    rw [Finset.sum_congr rfl (fun t' _ => hfin t')]
    have hind : ∀ m : Nat,
        (∑ t' ∈ Finset.range m, ∑ j ∈ Finset.range BK,
          Device.liftMag .bf16 (xh (t' * BK + j) a) *
            Device.liftMag .bf16 (bh (t' * BK + j) n))
        = ∑ s ∈ Finset.range (m * BK),
          Device.liftMag .bf16 (xh s a) * Device.liftMag .bf16 (bh s n) := by
      intro m
      induction m with
      | zero => simp
      | succ k ih =>
        rw [Finset.sum_range_succ, ih, show (k + 1) * BK = k * BK + BK from by ring,
          Finset.sum_range_add]
    exact hind t
  have hidR : ((∑ t' ∈ Finset.range t, Af xh bh a n t' : ℚ) : ℝ)
      = ∑ s ∈ Finset.range (t * BK),
          ((Device.liftMag .bf16 (xh s a) : ℚ) : ℝ) *
            ((Device.liftMag .bf16 (bh s n) : ℚ) : ℝ) := by
    rw [hid, Rat.cast_sum]
    refine Finset.sum_congr rfl (fun s _ => ?_)
    rw [Rat.cast_mul]
  rw [hidR]
  let P : Nat → Prop := fun s => ((chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i) ∧
    pnOf d i * BN + (n : Nat) < d.dstate ∧ pmOf d i * BM + (a : Nat) < d.headdim)
  -- the per-token bound
  have hstep : ∀ s ∈ Finset.range (t * BK),
      ((Device.liftMag .bf16 (xh s a) : ℚ) : ℝ) *
          ((Device.liftMag .bf16 (bh s n) : ℚ) : ℝ) ≤
        if P s then magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) else 0 := by
    intro s hs
    have hst : s < t * BK := Finset.mem_range.mp hs
    obtain ⟨hxA, hxB⟩ := hx s a hst
    obtain ⟨hbA, hbB⟩ := hb s n hst
    split_ifs with hcase
    · obtain ⟨htok, hnlt, hplt⟩ := hcase
      have hle : chunkOf d i * d.chunk + s ≤ endTok d (chunkOf d i) :=
        (Finset.mem_Icc.mp (Finset.mem_filter.mp htok).1).2
      have hxh : (xh s a : ℝ) = xV init d (chunkOf d i * d.chunk + s) (headOf d i)
          (pmOf d i * BM + (a : Nat)) := hxA ⟨hle, hplt⟩
      have hbh : |(bh s n : ℝ) - bV init d (chunkOf d i * d.chunk + s) (headOf d i)
          (pnOf d i * BN + (n : Nat)) *
          wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)|
          ≤ errB init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
            (pnOf d i * BN + (n : Nat)) := hbA ⟨htok, hnlt⟩
      have hxh' : xh s a = valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i)
          (pmOf d i * BM + (a : Nat))) := by
        have e : ((xh s a : ℚ) : ℝ) = ((valB init d.x (xIdx d (chunkOf d i * d.chunk + s)
            (headOf d i) (pmOf d i * BM + (a : Nat))) : ℚ) : ℝ) := by
          rw [hxh]; rfl
        exact_mod_cast e
      have hxle : ((Device.liftMag .bf16 (xh s a) : ℚ) : ℝ) ≤
          magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s)
            (headOf d i) (pmOf d i * BM + (a : Nat)))) := by
        rw [hxh']
        unfold magB
        exact le_rfl
      have hbhle : ((Device.liftMag .bf16 (bh s n) : ℚ) : ℝ) ≤
          max (bHat init d (chunkOf d i) (headOf d i)
          (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) := by
        have htri : |(bh s n : ℝ)| ≤ |(bh s n : ℝ) - bV init d (chunkOf d i * d.chunk + s)
            (headOf d i) (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)| +
            |bV init d (chunkOf d i * d.chunk + s) (headOf d i) (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)| := by
          have h := abs_add_le ((bh s n : ℝ) - bV init d (chunkOf d i * d.chunk + s)
            (headOf d i) (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s))
            (bV init d (chunkOf d i * d.chunk + s) (headOf d i) (pnOf d i * BN + (n : Nat)) *
            wRef init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s))
          simpa using h
        have h1 : |(bh s n : ℝ)| ≤ bHat init d (chunkOf d i) (headOf d i)
            (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat)) := by
          unfold bHat
          linarith
        have hlm : Device.liftMag .bf16 (bh s n) ≤ max |bh s n| (2 ^ (-126 : ℤ)) := by
          unfold Device.liftMag
          split
          · exact le_trans (by norm_num) (le_max_right _ _)
          · rfl
        have hcast : ((max |bh s n| (2 ^ (-126 : ℤ)) : ℚ) : ℝ) ≤
            max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s)
              (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) := by
          have hEq : ((2 ^ (-126 : ℤ) : ℚ) : ℝ) = (2 ^ (-126 : ℤ) : ℝ) := by push_cast; rfl
          rw [Rat.cast_max, Rat.cast_abs, hEq]
          exact max_le (le_trans h1 (le_max_left _ _)) (le_max_right _ _)
        exact le_trans (by exact_mod_cast hlm) hcast
      refine mul_le_mul hxle hbhle
        (by exact_mod_cast Device.liftMag_nonneg .bf16 _) ?_
      unfold magB
      exact_mod_cast Device.liftMag_nonneg .bf16 _
    · have hzero : ((Device.liftMag .bf16 (xh s a) : ℚ) : ℝ) *
          ((Device.liftMag .bf16 (bh s n) : ℚ) : ℝ) = 0 := by
        by_cases hA : (chunkOf d i * d.chunk + s) ∈ toks d (chunkOf d i)
        · by_cases hB : pnOf d i * BN + (n : Nat) < d.dstate
          · have hC : ¬(pmOf d i * BM + (a : Nat) < d.headdim) := by
              intro hC
              exact hcase ⟨hA, hB, hC⟩
            have hz : xh s a = 0 := hxB (fun hh => hC hh.2)
            rw [hz, Device.liftMag_zero .bf16]
            simp
          · have hz : bh s n = 0 := hbB (fun hh => hB hh.2)
            rw [hz, Device.liftMag_zero .bf16]
            simp
        · have hz : bh s n = 0 := hbB (fun hh => hA hh.1)
          rw [hz, Device.liftMag_zero .bf16]
          simp
      rw [hzero]
  have h1 : ∑ s ∈ Finset.range (t * BK),
        ((Device.liftMag .bf16 (xh s a) : ℚ) : ℝ) *
          ((Device.liftMag .bf16 (bh s n) : ℚ) : ℝ)
      ≤ ∑ s ∈ Finset.range (t * BK), (if P s then magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) else 0) :=
    Finset.sum_le_sum hstep
  have h2 : ∑ s ∈ Finset.range (t * BK), (if P s then magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) else 0)
      = ∑ s ∈ (Finset.range (t * BK)).filter P, magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) :=
    (Finset.sum_filter P (fun s => magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)))).symm
  have h3 : ∑ s ∈ (Finset.range (t * BK)).filter P, magB (valB init d.x (xIdx d (chunkOf d i * d.chunk + s) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (chunkOf d i * d.chunk + s) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ))
      = ∑ u ∈ ((Finset.range (t * BK)).filter P).image (fun s => chunkOf d i * d.chunk + s),
          magB (valB init d.x (xIdx d (u) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (u) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) := by
    rw [Finset.sum_image]
    intro x _ y _ hxy
    have hxy' : chunkOf d i * d.chunk + x = chunkOf d i * d.chunk + y := hxy
    omega
  have h4 : ∑ u ∈ ((Finset.range (t * BK)).filter P).image
        (fun s => chunkOf d i * d.chunk + s), magB (valB init d.x (xIdx d (u) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (u) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ))
      ≤ ∑ u ∈ toks d (chunkOf d i), magB (valB init d.x (xIdx d (u) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (u) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) := by
    apply Finset.sum_le_sum_of_subset_of_nonneg
    · intro u hu
      obtain ⟨s, hs, rfl⟩ := Finset.mem_image.mp hu
      exact (Finset.mem_filter.mp hs).2.1
    · intro u _ _
      apply mul_nonneg
      · unfold magB
        exact_mod_cast Device.liftMag_nonneg .bf16 _
      · exact le_trans (by positivity) (le_max_right _ _)
  have hchain : ∑ s ∈ Finset.range (t * BK),
        ((Device.liftMag .bf16 (xh s a) : ℚ) : ℝ) *
          ((Device.liftMag .bf16 (bh s n) : ℚ) : ℝ)
      ≤ ∑ u ∈ toks d (chunkOf d i), magB (valB init d.x (xIdx d (u) (headOf d i) (pmOf d i * BM + (a : Nat)))) *
              max (bHat init d (chunkOf d i) (headOf d i) (u) (pnOf d i * BN + (n : Nat))) (2 ^ (-126 : ℤ)) :=
    le_trans h1 (le_trans (le_of_eq h2) (le_trans (le_of_eq h3) h4))
  unfold dotMag
  exact hchain


/-! ## AccStep -/

/-!
# One accumulation step

The accumulator clause of `Inv` after one more `tl.dot_add`.  This is the numerical heart of the
accumulation loop: if the accumulator is within `DotLoop.bound` of the exact partial sums after
`t` iterations (`hacc`), and the operands of iteration `t` are the `ℚ` values `xh'`/`bh'`
(`hva`, `hvb`), then after the `tl.dot_add` the new accumulator (`r`) is within
`DotLoop.bound` of the exact partial sums over `t+1` iterations.

`hbig` is the room hypothesis: the accumulator's magnitude plus this iteration's lifted
magnitudes is at most `2^121` (the caller obtains it from the claim's `dotMag ≤ 2^120`).
-/

open TVA

set_option maxHeartbeats 800000

/-- An index of an earlier chunk is below the current chunk's start: `t' < t` and `k < BK`
give `t' * BK + k < t * BK`. -/
private lemma idx_lt_BK {t t' : Nat} (k : Fin BK) (ht' : t' < t) :
    t' * BK + (k : Nat) < t * BK := by
  have hk : (k : Nat) < BK := k.isLt
  have h1 : (t' + 1) * BK ≤ t * BK := Nat.mul_le_mul_right BK (by omega)
  have h2 : t' * BK + (k : Nat) < (t' + 1) * BK := by
    rw [Nat.succ_mul]
    omega
  omega

/-- The per-step relative constant is nonnegative. -/
private lemma rho_nonneg' : (0 : ℚ) ≤ rho := by
  unfold rho DotLoop.stepRel Device.relDot16 BK
  rw [Format.binary32_u]
  norm_num

/-- The per-step absolute constant is nonnegative. -/
private lemma alp_nonneg' : (0 : ℚ) ≤ alp := by
  unfold alp DotLoop.stepAbs Device.absDot16 BK
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

/-- **One accumulation step.**  See the file comment. -/
theorem accStep {E : Env} {t : Nat} (hdotacc : Device.DotBF16 E.P)
    {xh : Nat → Fin BM → ℚ} {bh : Nat → Fin BN → ℚ}
    {xh' : Nat → Fin BM → ℚ} {bh' : Nat → Fin BN → ℚ}
    (hxagree : ∀ (s : Nat) (a : Fin BM), s < t * BK → xh' s a = xh s a)
    (hbagree : ∀ (s : Nat) (n : Fin BN), s < t * BK → bh' s n = bh s n)
    {acc r : Tensor [BM, BN] DType.f32.Val}
    (hacc : (∀ (a : Fin BM) (n : Fin BN), (acc.get2 a n).isFinite = true) ∧
      (∀ (a : Fin BM) (n : Fin BN),
        |(acc.get2 a n).toRat - ∑ t' ∈ Finset.range t, Sf xh bh a n t'| ≤
          DotLoop.bound rho alp t (∑ t' ∈ Finset.range t, Af xh bh a n t')))
    {x : Tensor [BM, BK] DType.bf16.Val} {b : Tensor [BK, BN] DType.bf16.Val}
    (hr : E.P.dot dotCfg x b acc r ∨ ∃ d,
      E.P.dot dotCfg x b (Tensor.full (FloatV.ofPat 0)) d ∧
      ∀ i, AccAdd E.fusion DType.f32 (d.get i) (acc.get i) (r.get i))
    (hva : ∀ (a : Fin BM) (k : Fin BK),
      DType.bf16.val? (x.get2 a k) = some (xh' (t * BK + (k : Nat)) a))
    (hvb : ∀ (k : Fin BK) (n : Fin BN),
      DType.bf16.val? (b.get2 k n) = some (bh' (t * BK + (k : Nat)) n))
    (_ht : (t : ℚ) * rho ≤ 1 / 2)
    (hbig : ∀ (a : Fin BM) (n : Fin BN),
      |(acc.get2 a n).toRat| + ∑ k : Fin BK,
          Device.liftMag .bf16 (xh' (t * BK + (k : Nat)) a) *
            Device.liftMag .bf16 (bh' (t * BK + (k : Nat)) n) ≤ 2 ^ (121 : ℤ)) :
    (∀ (a : Fin BM) (n : Fin BN), (r.get2 a n).isFinite = true) ∧
      (∀ (a : Fin BM) (n : Fin BN),
        |(r.get2 a n).toRat - ∑ t' ∈ Finset.range (t + 1), Sf xh' bh' a n t'| ≤
          DotLoop.bound rho alp (t + 1) (∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t')) := by
  have key : ∀ (a : Fin BM) (n : Fin BN),
      (r.get2 a n).isFinite = true ∧
        |(r.get2 a n).toRat - ∑ t' ∈ Finset.range (t + 1), Sf xh' bh' a n t'| ≤
          DotLoop.bound rho alp (t + 1) (∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t') := by
    intro a n
    set S : ℚ := ∑ k : Fin BK, xh' (t * BK + (k : Nat)) a * bh' (t * BK + (k : Nat)) n with hS
    set A : ℚ := ∑ k : Fin BK, Device.liftMag .bf16 (xh' (t * BK + (k : Nat)) a) *
      Device.liftMag .bf16 (bh' (t * BK + (k : Nat)) n) with hA
    set St : ℚ := ∑ t' ∈ Finset.range t, Sf xh bh a n t' with hStdef
    set Bt : ℚ := ∑ t' ∈ Finset.range t, Af xh bh a n t' with hBtdef
    -- the lifted magnitude of this chunk is nonnegative
    have hA0 : (0 : ℚ) ≤ A := by
      rw [hA]
      exact Finset.sum_nonneg fun k _ =>
        mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
    -- room: `|acc| + A ≤ 2^121`
    have hL : |(acc.get2 a n).toRat| + A ≤ (2 : ℚ) ^ (121 : ℤ) := by
      rw [hA]
      exact hbig a n
    have hconst :
        2 * ((1 + Device.relDot16 BK) * (2 : ℚ) ^ (121 : ℤ) + Device.absDot16 BK +
          2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
      unfold Device.relDot16 Device.absDot16 BK
      rw [Format.binary32_eta, Format.binary32_maxVal]
      norm_num
    have hroom :
        2 * ((1 + Device.relDot16 BK) * (|(acc.get2 a n).toRat| + A) +
          Device.absDot16 BK + 2 * Format.binary32.eta) ≤
          Format.binary32.maxVal := by
      have hrel0 : (0 : ℚ) ≤ Device.relDot16 BK := by
        unfold Device.relDot16; positivity
      have hmul : (1 + Device.relDot16 BK) * (|(acc.get2 a n).toRat| + A) ≤
          (1 + Device.relDot16 BK) * (2 : ℚ) ^ (121 : ℤ) :=
        mul_le_mul_of_nonneg_left hL (by linarith)
      linarith
    -- one `tl.dot_add`
    have hr' : (E.P.dot { a := DType.bf16, b := DType.bf16, acc := DType.f32, precision := .tf32, maxImpreciseAcc := 0 } x b acc r) ∨
        (∃ d, (E.P.dot { a := DType.bf16, b := DType.bf16, acc := DType.f32, precision := .tf32, maxImpreciseAcc := 0 } x b (Tensor.full (FloatV.ofPat 0)) d) ∧
          ∀ i, AccAdd E.fusion DType.f32 (d.get i) (acc.get i) (r.get i)) := hr
    have hstep := DotLift.dotAdd_step (P := E.P) (fusion := E.fusion)
      (hacc := hdotacc .tf32) (a := x) (b := b) (c := acc) (r := r)
      (va := fun k => xh' (t * BK + (k : Nat)) a)
      (vb := fun k => bh' (t * BK + (k : Nat)) n)
      (by unfold Device.relDot16; positivity)
      (by unfold Device.absDot16; positivity)
      hr' a n DType.val?_zero_f32 (fun k => hva a k) (fun k => hvb k n) (hacc.1 a n) hroom
    -- `|S| ≤ A`, and the same for the earlier chunks
    have hSA : |S| ≤ A := by
      rw [hS, hA]
      exact (Finset.abs_sum_le_sum_abs _ _).trans
        (Finset.sum_le_sum fun k _ => Device.abs_mul_le_liftMag _ _ _ _)
    have hSt : |St| ≤ Bt := by
      rw [hStdef, hBtdef]
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
      intro t' _
      unfold Sf Af
      exact (Finset.abs_sum_le_sum_abs _ _).trans
        (Finset.sum_le_sum fun k _ => Device.abs_mul_le_liftMag _ _ _ _)
    have he : |(acc.get2 a n).toRat - St| ≤ DotLoop.bound rho alp t Bt := by
      have h := hacc.2 a n
      rwa [← hStdef, ← hBtdef] at h
    have haccb : |(r.get2 a n).toRat - (St + S)| ≤
        DotLoop.bound rho alp (t + 1) (Bt + A) :=
      DotLoop.accum_step rho_nonneg' alp_nonneg' hSt hSA he hstep.2
    -- identify the sums over `t + 1` iterations
    have hprefixS : (∑ t' ∈ Finset.range t, Sf xh' bh' a n t') = St := by
      rw [hStdef]
      refine Finset.sum_congr rfl ?_
      intro t' ht'
      rw [Finset.mem_range] at ht'
      unfold Sf
      refine Finset.sum_congr rfl ?_
      intro k _
      rw [hxagree (t' * BK + (k : Nat)) a (idx_lt_BK k ht'),
        hbagree (t' * BK + (k : Nat)) n (idx_lt_BK k ht')]
    have hprefixA : (∑ t' ∈ Finset.range t, Af xh' bh' a n t') = Bt := by
      rw [hBtdef]
      refine Finset.sum_congr rfl ?_
      intro t' ht'
      rw [Finset.mem_range] at ht'
      unfold Af
      refine Finset.sum_congr rfl ?_
      intro k _
      rw [hxagree (t' * BK + (k : Nat)) a (idx_lt_BK k ht'),
        hbagree (t' * BK + (k : Nat)) n (idx_lt_BK k ht')]
    have hlastS : Sf xh' bh' a n t = S := by
      rw [hS]
      rfl
    have hlastA : Af xh' bh' a n t = A := by
      rw [hA]
      rfl
    have hSsum : (∑ t' ∈ Finset.range (t + 1), Sf xh' bh' a n t') = St + S := by
      rw [Finset.sum_range_succ, hprefixS, hlastS]
    have hAsum : (∑ t' ∈ Finset.range (t + 1), Af xh' bh' a n t') = Bt + A := by
      rw [Finset.sum_range_succ, hprefixA, hlastA]
    refine ⟨hstep.1, ?_⟩
    rw [hSsum, hAsum]
    exact haccb
  exact ⟨fun a n => (key a n).1, fun a n => (key a n).2⟩


/-! ## PtrStep -/

open TVA
set_option maxHeartbeats 800000

/-! # Pointers advance by one token block -/

/-- A pointer tile plus a constant offset tile: the address moves by `k·bytes`. -/
theorem addr_add_off {s : Shape} {dd : DType} (ps : Tensor s (Ptr dd))
    (off : Tensor s I32) (i : Idx s) (k : Nat)
    (ho : (off.get i).toInt = (k : Int)) :
    ((ps + off).get i).addr = (ps.get i).addr + (k : Int) * dd.bytes := by
  rw [Tensor.get_ptrAddT, Ptr.offset_addr, ho]

/-- `b.ptr A + k·bytes = b.ptr (A + k)`. -/
theorem ptr_addr_shift {dd : DType} (b : Buf dd) (A B k : Nat) (h : B = A + k) :
    (b.ptr (A : Int)).addr + (k : Int) * dd.bytes = (b.ptr (B : Int)).addr := by
  subst h
  rw [Buf.ptr_addr, Buf.ptr_addr]
  push_cast
  ring

/-- One step of the loop moves the `x` pointers on by `64` tokens. -/
theorem ptrStepX {d : Inst} {i : Fin (launch d).size} {t : Nat} {st : LoopSt d}
    {off : Tensor [BM, BK] I32} (hoff : ∀ a : Fin BM, ∀ j : Fin BK, (off.get2 a j).toInt = ((BK * d.sXs : Nat) : Int))
    (hP : ∀ (a : Fin BM) (j : Fin BK),
      (st.1.get2 a j).addr =
        (d.x.ptr (xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
          (headOf d i) (pmOf d i * BM + (a : Nat)))).addr) :
    ∀ (a : Fin BM) (j : Fin BK),
      ((st.1 + off).get2 a j).addr =
        (d.x.ptr (xIdx d (chunkOf d i * d.chunk + ((t + 1) * BK + (j : Nat)))
          (headOf d i) (pmOf d i * BM + (a : Nat)))).addr := by
  intro a j
  have h1 : (st.1.get (a, j, ())).addr =
      (d.x.ptr (xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (headOf d i) (pmOf d i * BM + (a : Nat)))).addr := hP a j
  have h2 : (off.get (a, j, ())).toInt = ((BK * d.sXs : Nat) : Int) := hoff a j
  have hidx : xIdx d (chunkOf d i * d.chunk + ((t + 1) * BK + (j : Nat)))
        (headOf d i) (pmOf d i * BM + (a : Nat))
      = xIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (headOf d i) (pmOf d i * BM + (a : Nat)) + BK * d.sXs := by
    unfold xIdx
    ring
  rw [Tensor.get2, addr_add_off st.1 off (a, j, ()) (BK * d.sXs) h2, h1]
  exact ptr_addr_shift d.x _ _ (BK * d.sXs) hidx

/-- One step of the loop moves the `B` pointers on by `64` tokens. -/
theorem ptrStepB {d : Inst} {i : Fin (launch d).size} {t : Nat} {st : LoopSt d}
    {off : Tensor [BK, BN] I32} (hoff : ∀ j : Fin BK, ∀ n : Fin BN, (off.get2 j n).toInt = ((BK * d.sBs : Nat) : Int))
    (hP : ∀ (j : Fin BK) (n : Fin BN),
      (st.2.1.get2 j n).addr =
        (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
          (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr) :
    ∀ (j : Fin BK) (n : Fin BN),
      ((st.2.1 + off).get2 j n).addr =
        (d.b.ptr (bIdx d (chunkOf d i * d.chunk + ((t + 1) * BK + (j : Nat)))
          (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr := by
  intro j n
  have h1 : (st.2.1.get (j, n, ())).addr =
      (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)))).addr := hP j n
  have h2 : (off.get (j, n, ())).toInt = ((BK * d.sBs : Nat) : Int) := hoff j n
  have hidx : bIdx d (chunkOf d i * d.chunk + ((t + 1) * BK + (j : Nat)))
        (grp d (headOf d i)) (pnOf d i * BN + (n : Nat))
      = bIdx d (chunkOf d i * d.chunk + (t * BK + (j : Nat)))
        (grp d (headOf d i)) (pnOf d i * BN + (n : Nat)) + BK * d.sBs := by
    unfold bIdx
    ring
  rw [Tensor.get2, addr_add_off st.2.1 off (j, n, ()) (BK * d.sBs) h2, h1]
  exact ptr_addr_shift d.b _ _ (BK * d.sBs) hidx

/-- One step of the loop moves the `dt` pointers on by `64` entries. -/
theorem ptrStepDt {d : Inst} {i : Fin (launch d).size} {t : Nat} {st : LoopSt d}
    {off : Tensor [BK] I32} (hoff : ∀ j : Fin BK, (off.get1 j).toInt = (BK : Int))
    (hP : ∀ j : Fin BK,
      (st.2.2.1.get1 j).addr =
        (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr) :
    ∀ j : Fin BK,
      ((st.2.2.1 + off).get1 j).addr =
        (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) ((t + 1) * BK + (j : Nat)))).addr := by
  intro j
  have h1 : (st.2.2.1.get (j, ())).addr =
      (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr := hP j
  have h2 : (off.get (j, ())).toInt = (BK : Int) := hoff j
  have hidx : dIdx d (headOf d i) (chunkOf d i) ((t + 1) * BK + (j : Nat))
      = dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)) + BK := by
    unfold dIdx
    ring
  rw [Tensor.get1, addr_add_off st.2.2.1 off (j, ()) BK h2, h1]
  exact ptr_addr_shift d.dt _ _ BK hidx

/-- One step of the loop moves the `dA_cumsum` pointers on by `64` entries. -/
theorem ptrStepDA {d : Inst} {i : Fin (launch d).size} {t : Nat} {st : LoopSt d}
    {off : Tensor [BK] I32} (hoff : ∀ j : Fin BK, (off.get1 j).toInt = (BK : Int))
    (hP : ∀ j : Fin BK,
      (st.2.2.2.1.get1 j).addr =
        (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr) :
    ∀ j : Fin BK,
      ((st.2.2.2.1 + off).get1 j).addr =
        (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) ((t + 1) * BK + (j : Nat)))).addr := by
  intro j
  have h1 : (st.2.2.2.1.get (j, ())).addr =
      (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)))).addr := hP j
  have h2 : (off.get (j, ())).toInt = (BK : Int) := hoff j
  have hidx : dIdx d (headOf d i) (chunkOf d i) ((t + 1) * BK + (j : Nat))
      = dIdx d (headOf d i) (chunkOf d i) (t * BK + (j : Nat)) + BK := by
    unfold dIdx
    ring
  rw [Tensor.get1, addr_add_off st.2.2.2.1 off (j, ()) BK h2, h1]
  exact ptr_addr_shift d.dA _ _ BK hidx

/-- One step of the loop moves the `seq_idx` pointers on by `64` tokens. -/
theorem ptrStepSeq {d : Inst} {i : Fin (launch d).size} {t : Nat} {st : LoopSt d}
    {off : Tensor [BK] I32} (hoff : ∀ j : Fin BK, (off.get1 j).toInt = (BK : Int))
    (hP : ∀ j : Fin BK,
      (st.2.2.2.2.1.get1 j).addr =
        (d.seqIdx.ptr (chunkOf d i * d.chunk + (t * BK + (j : Nat)))).addr) :
    ∀ j : Fin BK,
      ((st.2.2.2.2.1 + off).get1 j).addr =
        (d.seqIdx.ptr (chunkOf d i * d.chunk + ((t + 1) * BK + (j : Nat)))).addr := by
  intro j
  have h1 : (st.2.2.2.2.1.get (j, ())).addr =
      (d.seqIdx.ptr (chunkOf d i * d.chunk + (t * BK + (j : Nat)))).addr := hP j
  have h2 : (off.get (j, ())).toInt = (BK : Int) := hoff j
  have hidx : chunkOf d i * d.chunk + ((t + 1) * BK + (j : Nat))
      = chunkOf d i * d.chunk + (t * BK + (j : Nat)) + BK := by
    ring
  rw [Tensor.get1, addr_add_off st.2.2.2.2.1 off (j, ()) BK h2, h1]
  exact ptr_addr_shift d.seqIdx _ _ BK hidx

/-- The `x` pointer increment `BLOCK_SIZE_K · stride_x_seqlen` has its exact value. -/
theorem offX_val {d : Inst} (hd : d.ok) (a : Idx [BM, BK]) :
    (((bc ((IntV.ofNat BK : I32) * (IntV.ofNat d.sXs : I32)) : Tensor [BM, BK] I32)).get a).toInt
      = ((BK * d.sXs : Nat) : Int) := by
  have hb : BK * d.sXs < 2 ^ 31 := by
    have h := hd.xtile_lt
    simp only [BK] at h ⊢
    omega
  have hm := IntV.I32.mul_nat (a := (IntV.ofNat BK : I32)) (b := (IntV.ofNat d.sXs : I32))
    (x := BK) (y := d.sXs)
    (IntV.I32.toInt_ofNat (by norm_num [BK])) (IntV.I32.toInt_ofNat hd.sX_lt.1) hb
  simpa using hm

/-- The `B` pointer increment `BLOCK_SIZE_K · stride_b_seqlen` has its exact value. -/
theorem offB_val {d : Inst} (hd : d.ok) (a : Idx [BK, BN]) :
    (((bc ((IntV.ofNat BK : I32) * (IntV.ofNat d.sBs : I32)) : Tensor [BK, BN] I32)).get a).toInt
      = ((BK * d.sBs : Nat) : Int) := by
  have hb : BK * d.sBs < 2 ^ 31 := by
    have h := hd.btile_lt
    simp only [BK] at h ⊢
    omega
  have hm := IntV.I32.mul_nat (a := (IntV.ofNat BK : I32)) (b := (IntV.ofNat d.sBs : I32))
    (x := BK) (y := d.sBs)
    (IntV.I32.toInt_ofNat (by norm_num [BK])) (IntV.I32.toInt_ofNat hd.sB_lt.1) hb
  simpa using hm

/-- The `dt` and `dA_cumsum` pointer increment has its exact value. -/
theorem offChunk_val {d : Inst} (_hd : d.ok) (a : Idx [BK]) :
    (((bc ((IntV.ofNat BK : I32) * (IntV.ofNat 1 : I32)) : Tensor [BK] I32)).get a).toInt
      = ((BK : Nat) : Int) := by
  have hb : BK * 1 < 2 ^ 31 := by
    simp only [BK]
    norm_num
  have hm := IntV.I32.mul_nat (a := (IntV.ofNat BK : I32)) (b := (IntV.ofNat 1 : I32))
    (x := BK) (y := 1)
    (IntV.I32.toInt_ofNat (by norm_num [BK])) (IntV.I32.toInt_ofNat (by norm_num)) hb
  simpa [Nat.mul_one] using hm


/-! ## H0 -/

/-!
# The loop invariant at the start of the accumulation loop

Before the loop the kernel has built the five pointer tiles of the chunk's first `BK` tokens.
This file checks that they address exactly the tiles of iteration `0`, so that `Inv` holds
before the first iteration; the operand clauses are vacuous there and the fresh accumulator
`tl.zeros [BM, BN] f32` satisfies the `DotLoop.bound` clause at `t = 0`.
-/

open TVA

set_option maxHeartbeats 800000

/-! `get2`/`get1` versions of the library's `get` lemmas (the library states them for `.get`,
whose index `(i, j, ())` cannot be written in a goal: `Idx []` does not elaborate to `Unit`
at implicit transparency).  All are `rfl`. -/
private theorem get2_add' {m n : Nat} {sg : Bool} {w : Nat}
    (x y : Tensor [m, n] (IntV sg w)) (a : Fin m) (j : Fin n) :
    (x + y).get2 a j = x.get2 a j + y.get2 a j := rfl

private theorem get2_muls' {m n : Nat} {sg : Bool} {w : Nat}
    (x : Tensor [m, n] (IntV sg w)) (c : IntV sg w) (a : Fin m) (j : Fin n) :
    (x * c).get2 a j = x.get2 a j * c := rfl

private theorem get2_sadd' {m n : Nat} {sg : Bool} {w : Nat}
    (c : IntV sg w) (x : Tensor [m, n] (IntV sg w)) (a : Fin m) (j : Fin n) :
    (c + x).get2 a j = c + x.get2 a j := rfl

private theorem get2_col' {m n : Nat} {α : Type} (x : Tensor [m] α) (a : Fin m) (j : Fin n) :
    (x.col n).get2 a j = x.get1 a := rfl

private theorem get2_row' {m n : Nat} {α : Type} (x : Tensor [n] α) (a : Fin m) (j : Fin n) :
    (x.row m).get2 a j = x.get1 j := rfl

private theorem get1_muls' {n : Nat} {sg : Bool} {w : Nat}
    (x : Tensor [n] (IntV sg w)) (c : IntV sg w) (j : Fin n) :
    (x * c).get1 j = x.get1 j * c := rfl

private theorem get1_sadd' {n : Nat} {sg : Bool} {w : Nat}
    (c : IntV sg w) (x : Tensor [n] (IntV sg w)) (j : Fin n) :
    (c + x).get1 j = c + x.get1 j := rfl

/-- A scalar pointer plus a lane offset tensor. -/
private theorem get2_scalarAdd' {m n : Nat} {dd : DType} {sg : Bool} {w : Nat}
    (p : Ptr dd) (offs : Tensor [m, n] (IntV sg w)) (a : Fin m) (j : Fin n) :
    (p + offs).get2 a j = p.offset (offs.get2 a j).toInt := rfl

/-- A scalar pointer plus a lane offset tensor, rank 1. -/
private theorem get1_scalarAdd' {n : Nat} {dd : DType} {sg : Bool} {w : Nat}
    (p : Ptr dd) (offs : Tensor [n] (IntV sg w)) (j : Fin n) :
    (p + offs).get1 j = p.offset (offs.get1 j).toInt := rfl

private theorem qOf_lt_gx' {d : Inst} (i : Fin (launch d).size) : qOf d i < gx d := by
  have hpos : 0 < gx d * nc d * d.nheads := by
    rw [← launch_size]
    exact Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have h : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h0 | h0
    · rw [h0] at hpos
      simp at hpos
    · exact h0
  unfold qOf
  exact Nat.mod_lt _ h

private theorem ptr_off_eq {dd : DType} (B : Buf dd) (base target k : Int)
    (h : base + k = target) : ((B.ptr base).offset k).addr = (B.ptr target).addr := by
  simp only [Ptr.offset_addr, Buf.ptr_addr]
  have : base * dd.bytes + k * dd.bytes = target * dd.bytes := by
    rw [← Int.add_mul, h]
  linarith

private theorem i32_ofInt_add0 (n : Nat) (h : n < 2 ^ 31) :
    (IntV.ofInt (0 + (n : Int)) : I32).toInt = (n : Int) := by
  simp only [Int.zero_add]
  refine IntV.I32.toInt_ofInt (x := (n : Int)) ?_
  constructor
  · have h0 : (0 : Int) ≤ (n : Int) := by exact_mod_cast Nat.zero_le n
    linarith
  · exact_mod_cast h

private theorem pmOf_lt' {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    pmOf d i < (d.headdim + BM - 1) / BM := by
  have hq := qOf_lt_gx' i
  rw [gx_eq] at hq
  unfold pmOf
  rw [Nat.div_lt_iff_lt_mul (npmOf_pos hd)]
  exact hq

private theorem pnOf_lt' {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    pnOf d i < (d.dstate + BN - 1) / BN := by
  have hmod : pnOf d i = qOf d i % ((d.dstate + BN - 1) / BN) := rfl
  rw [hmod]
  have hpos : 0 < (d.dstate + BN - 1) / BN := by
    simpa [npmOf] using npmOf_pos hd
  exact Nat.mod_lt _ hpos

private theorem pmOf_mul_add_le {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (a : Fin BM) :
    pmOf d i * BM + (a : Nat) ≤ d.headdim + 63 := by
  have h1 := pmOf_lt' hd i
  have h2 : pmOf d i * BM + BM ≤ ((d.headdim + BM - 1) / BM) * BM := by
    have h := Nat.mul_le_mul_right BM (Nat.succ_le_of_lt h1)
    calc pmOf d i * BM + BM = (pmOf d i + 1) * BM := by ring
      _ ≤ _ := h
  have h3 : ((d.headdim + BM - 1) / BM) * BM ≤ d.headdim + BM - 1 := Nat.div_mul_le_self _ _
  have ha : (a : Nat) < BM := a.isLt
  have hp := hd.headdim_pos
  simp only [BM] at h2 h3 ha ⊢
  omega

private theorem pnOf_mul_add_le {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (n : Fin BN) :
    pnOf d i * BN + (n : Nat) ≤ d.dstate + 127 := by
  have h1 := pnOf_lt' hd i
  have h2 : pnOf d i * BN + BN ≤ ((d.dstate + BN - 1) / BN) * BN := by
    have h := Nat.mul_le_mul_right BN (Nat.succ_le_of_lt h1)
    calc pnOf d i * BN + BN = (pnOf d i + 1) * BN := by ring
      _ ≤ _ := h
  have h3 : ((d.dstate + BN - 1) / BN) * BN ≤ d.dstate + BN - 1 := Nat.div_mul_le_self _ _
  have ha : (n : Nat) < BN := n.isLt
  have hp := hd.dstate_pos
  simp only [BN] at h2 h3 ha ⊢
  omega

/-! ### Bounds on the tile offsets -/

private theorem x_row_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (a : Fin BM) :
    pmOf d i * BM + (a : Nat) < 2 ^ 31 := by
  have h1 : pmOf d i * BM + (a : Nat) ≤ d.headdim + 63 := pmOf_mul_add_le hd i a
  have h2 := hd.headdim_lt
  omega

private theorem x_lane_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (a : Fin BM) :
    (pmOf d i * BM + (a : Nat)) * d.sXp < 2 ^ 31 := by
  have h1 : pmOf d i * BM + (a : Nat) ≤ d.headdim + 63 := pmOf_mul_add_le hd i a
  have h2 : (pmOf d i * BM + (a : Nat)) * d.sXp ≤ (d.headdim + 63) * d.sXp :=
    Nat.mul_le_mul_right _ h1
  have h3 : (d.headdim + 63) * d.sXp ≤ (d.headdim + 63) * d.sXp + 64 * d.sXs :=
    Nat.le_add_right _ _
  have h4 := hd.xtile_lt
  omega

private theorem k_lane_lt {d : Inst} (hd : d.ok) (j : Fin BK) : (j : Nat) * d.sXs < 2 ^ 31 := by
  have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
  have h1 : (j : Nat) * d.sXs ≤ 63 * d.sXs := Nat.mul_le_mul_right _ (by omega)
  have h2 : 63 * d.sXs ≤ 64 * d.sXs := Nat.mul_le_mul_right _ (by norm_num)
  have h3 : 64 * d.sXs ≤ (d.headdim + 63) * d.sXp + 64 * d.sXs := Nat.le_add_left _ _
  have h4 := hd.xtile_lt
  omega

private theorem b_row_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (n : Fin BN) :
    pnOf d i * BN + (n : Nat) < 2 ^ 31 := by
  have h1 : pnOf d i * BN + (n : Nat) ≤ d.dstate + 127 := pnOf_mul_add_le hd i n
  have h2 := hd.dstate_lt
  omega

private theorem b_lane_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (n : Fin BN) :
    (pnOf d i * BN + (n : Nat)) * d.sBn < 2 ^ 31 := by
  have h1 : pnOf d i * BN + (n : Nat) ≤ d.dstate + 127 := pnOf_mul_add_le hd i n
  have h2 : (pnOf d i * BN + (n : Nat)) * d.sBn ≤ (d.dstate + 127) * d.sBn :=
    Nat.mul_le_mul_right _ h1
  have h3 : (d.dstate + 127) * d.sBn ≤ (d.dstate + 127) * d.sBn + 64 * d.sBs :=
    Nat.le_add_right _ _
  have h4 := hd.btile_lt
  omega

private theorem k_lane_lt_b {d : Inst} (hd : d.ok) (j : Fin BK) : (j : Nat) * d.sBs < 2 ^ 31 := by
  have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
  have h1 : (j : Nat) * d.sBs ≤ 63 * d.sBs := Nat.mul_le_mul_right _ (by omega)
  have h2 : 63 * d.sBs ≤ 64 * d.sBs := Nat.mul_le_mul_right _ (by norm_num)
  have h3 : 64 * d.sBs ≤ (d.dstate + 127) * d.sBn + 64 * d.sBs := Nat.le_add_left _ _
  have h4 := hd.btile_lt
  omega

/-! The `[BK]` offsets `tl.arange 0 BK * 1` of `dt`, `dA_cumsum` and `seq_idx`. -/
private theorem k_off_toInt (j : Fin BK) :
    ((tl.arange 0 BK * (IntV.ofNat 1 : I32)).get1 j).toInt = ((j : Nat) * 1 : Nat) := by
  simp only [tl.arange, get1_muls', Tensor.get1_arange]
  have h1 : (IntV.ofNat 1 : I32).toInt = (1 : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have hj0 : (j : Nat) < 2 ^ 31 := by
    have hj : (j : Nat) < BK := j.isLt
    simp only [BK] at hj ⊢
    omega
  have hofj : (IntV.ofInt (0 + ((j : Nat) : Int)) : I32).toInt = ((j : Nat) : Int) :=
    i32_ofInt_add0 (j : Nat) hj0
  exact IntV.I32.mul_nat hofj h1 (by omega)

/-! x: `offs_m * stride_x_hdim + offs_k * stride_x_seqlen` -/
private theorem x_off_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r : I32}
    (hr : r.toInt = (pmOf d i : Int)) (a : Fin BM) (j : Fin BK) :
    ((((r * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK) * (IntV.ofNat d.sXp : I32) +
       ((tl.arange 0 BK).row BM) * (IntV.ofNat d.sXs : I32)).get2 a j).toInt
      = (((pmOf d i * BM + (a : Nat)) * d.sXp + (j : Nat) * d.sXs : Nat) : Int) := by
  simp only [tl.arange, get2_add', get2_muls', get2_col',
    get1_sadd', Tensor.get1_arange]
  have hBM : (IntV.ofNat BM : I32).toInt = (BM : Int) := IntV.I32.toInt_ofNat (by norm_num [BM])
  have hsXp : (IntV.ofNat d.sXp : I32).toInt = (d.sXp : Int) := IntV.I32.toInt_ofNat hd.sX_lt.2.2
  have hsXs : (IntV.ofNat d.sXs : I32).toInt = (d.sXs : Int) := IntV.I32.toInt_ofNat hd.sX_lt.1
  have hrow : pmOf d i * BM + (a : Nat) < 2 ^ 31 := x_row_lt hd i a
  have hrow2 : pmOf d i * BM < 2 ^ 31 := by
    have hle : pmOf d i * BM ≤ pmOf d i * BM + (a : Nat) := Nat.le_add_right _ _
    omega
  have ha0 : (a : Nat) < 2 ^ 31 := by
    have ha : (a : Nat) < BM := a.isLt
    simp only [BM] at ha ⊢
    omega
  have hof : (IntV.ofInt (0 + ((a : Nat) : Int)) : I32).toInt = ((a : Nat) : Int) :=
    i32_ofInt_add0 (a : Nat) ha0
  have hj0 : (j : Nat) < 2 ^ 31 := by
    have hj : (j : Nat) < BK := j.isLt
    simp only [BK] at hj ⊢
    omega
  have hofj : (IntV.ofInt (0 + ((j : Nat) : Int)) : I32).toInt = ((j : Nat) : Int) :=
    i32_ofInt_add0 (j : Nat) hj0
  have hmul : (r * (IntV.ofNat BM : I32)).toInt = ((pmOf d i * BM : Nat) : Int) :=
    IntV.I32.mul_nat hr hBM hrow2
  have hA : (r * (IntV.ofNat BM : I32) + IntV.ofInt (0 + ((a : Nat) : Int))).toInt
      = ((pmOf d i * BM + (a : Nat) : Nat) : Int) :=
    IntV.I32.add_nat hmul hof hrow
  have hM : ((r * (IntV.ofNat BM : I32) + IntV.ofInt (0 + ((a : Nat) : Int))) *
      (IntV.ofNat d.sXp : I32)).toInt
      = (((pmOf d i * BM + (a : Nat)) * d.sXp : Nat) : Int) :=
    IntV.I32.mul_nat hA hsXp (x_lane_lt hd i a)
  have hK : (IntV.ofInt (0 + ((j : Nat) : Int)) * (IntV.ofNat d.sXs : I32)).toInt
      = (((j : Nat) * d.sXs : Nat) : Int) :=
    IntV.I32.mul_nat hofj hsXs (k_lane_lt hd j)
  have hb : (pmOf d i * BM + (a : Nat)) * d.sXp + (j : Nat) * d.sXs < 2 ^ 31 := by
    have h1 : (pmOf d i * BM + (a : Nat)) * d.sXp ≤ (d.headdim + 63) * d.sXp :=
      Nat.mul_le_mul_right _ (pmOf_mul_add_le hd i a)
    have h2 : (j : Nat) * d.sXs ≤ 63 * d.sXs := by
      have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
      exact Nat.mul_le_mul_right _ (by omega)
    have h3 : 63 * d.sXs ≤ 64 * d.sXs := Nat.mul_le_mul_right _ (by norm_num)
    have h4 : 64 * d.sXs ≤ (d.headdim + 63) * d.sXp + 64 * d.sXs := Nat.le_add_left _ _
    have h5 := hd.xtile_lt
    omega
  exact IntV.I32.add_nat hM hK hb

/-! b: `offs_n * stride_b_hdim + offs_k * stride_b_seqlen` -/
private theorem b_off_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r : I32}
    (hr : r.toInt = (pnOf d i : Int)) (n : Fin BN) (j : Fin BK) :
    ((((r * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK) * (IntV.ofNat d.sBn : I32) +
       ((tl.arange 0 BK).col BN) * (IntV.ofNat d.sBs : I32)).get2 j n).toInt
      = (((pnOf d i * BN + (n : Nat)) * d.sBn + (j : Nat) * d.sBs : Nat) : Int) := by
  simp only [tl.arange, get2_add', get2_muls', get2_row',
    get1_sadd', Tensor.get1_arange]
  have hBN : (IntV.ofNat BN : I32).toInt = (BN : Int) := IntV.I32.toInt_ofNat (by norm_num [BN])
  have hsBn : (IntV.ofNat d.sBn : I32).toInt = (d.sBn : Int) := IntV.I32.toInt_ofNat hd.sB_lt.2.2
  have hsBs : (IntV.ofNat d.sBs : I32).toInt = (d.sBs : Int) := IntV.I32.toInt_ofNat hd.sB_lt.1
  have hrow : pnOf d i * BN + (n : Nat) < 2 ^ 31 := b_row_lt hd i n
  have hrow2 : pnOf d i * BN < 2 ^ 31 := by
    have hle : pnOf d i * BN ≤ pnOf d i * BN + (n : Nat) := Nat.le_add_right _ _
    omega
  have hn0 : (n : Nat) < 2 ^ 31 := by
    have ha : (n : Nat) < BN := n.isLt
    simp only [BN] at ha ⊢
    omega
  have hof : (IntV.ofInt (0 + ((n : Nat) : Int)) : I32).toInt = ((n : Nat) : Int) :=
    i32_ofInt_add0 (n : Nat) hn0
  have hj0 : (j : Nat) < 2 ^ 31 := by
    have hj : (j : Nat) < BK := j.isLt
    simp only [BK] at hj ⊢
    omega
  have hofj : (IntV.ofInt (0 + ((j : Nat) : Int)) : I32).toInt = ((j : Nat) : Int) :=
    i32_ofInt_add0 (j : Nat) hj0
  have hmul : (r * (IntV.ofNat BN : I32)).toInt = ((pnOf d i * BN : Nat) : Int) :=
    IntV.I32.mul_nat hr hBN hrow2
  have hA : (r * (IntV.ofNat BN : I32) + IntV.ofInt (0 + ((n : Nat) : Int))).toInt
      = ((pnOf d i * BN + (n : Nat) : Nat) : Int) :=
    IntV.I32.add_nat hmul hof hrow
  have hM : ((r * (IntV.ofNat BN : I32) + IntV.ofInt (0 + ((n : Nat) : Int))) *
      (IntV.ofNat d.sBn : I32)).toInt
      = (((pnOf d i * BN + (n : Nat)) * d.sBn : Nat) : Int) :=
    IntV.I32.mul_nat hA hsBn (b_lane_lt hd i n)
  have hK : (IntV.ofInt (0 + ((j : Nat) : Int)) * (IntV.ofNat d.sBs : I32)).toInt
      = (((j : Nat) * d.sBs : Nat) : Int) :=
    IntV.I32.mul_nat hofj hsBs (k_lane_lt_b hd j)
  have hb : (pnOf d i * BN + (n : Nat)) * d.sBn + (j : Nat) * d.sBs < 2 ^ 31 := by
    have h1 : (pnOf d i * BN + (n : Nat)) * d.sBn ≤ (d.dstate + 127) * d.sBn :=
      Nat.mul_le_mul_right _ (pnOf_mul_add_le hd i n)
    have h2 : (j : Nat) * d.sBs ≤ 63 * d.sBs := by
      have hj : (j : Nat) < 64 := by simpa [BK] using j.isLt
      exact Nat.mul_le_mul_right _ (by omega)
    have h3 : 63 * d.sBs ≤ 64 * d.sBs := Nat.mul_le_mul_right _ (by norm_num)
    have h4 : 64 * d.sBs ≤ (d.dstate + 127) * d.sBn + 64 * d.sBs := Nat.le_add_left _ _
    have h5 := hd.btile_lt
    omega
  exact IntV.I32.add_nat hM hK hb

/-! The index identities, in `Nat`. -/
private theorem idx_x (d : Inst) (i : Fin (launch d).size) (a : Fin BM) (j : Fin BK) :
    xIdx d (chunkOf d i * d.chunk) (headOf d i) 0 +
        ((pmOf d i * BM + (a : Nat)) * d.sXp + (j : Nat) * d.sXs)
      = xIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (headOf d i)
          (pmOf d i * BM + (a : Nat)) := by
  unfold xIdx
  ring

private theorem idx_b (d : Inst) (i : Fin (launch d).size) (n : Fin BN) (j : Fin BK) :
    bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0 +
        ((pnOf d i * BN + (n : Nat)) * d.sBn + (j : Nat) * d.sBs)
      = bIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (grp d (headOf d i))
          (pnOf d i * BN + (n : Nat)) := by
  unfold bIdx
  ring

private theorem idx_dt (d : Inst) (i : Fin (launch d).size) (j : Fin BK) :
    dIdx d (headOf d i) (chunkOf d i) 0 + ((j : Nat) * 1)
      = dIdx d (headOf d i) (chunkOf d i) (0 * BK + (j : Nat)) := by
  unfold dIdx
  ring

private theorem idx_seq (d : Inst) (i : Fin (launch d).size) (j : Fin BK) :
    chunkOf d i * d.chunk + ((j : Nat) * 1)
      = chunkOf d i * d.chunk + (0 * BK + (j : Nat)) := by
  ring

/-! ### The five pointer clauses -/

private theorem x_clause0 {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {P : Ptr DType.bf16} (hP : P = d.x.ptr (xIdx d (chunkOf d i * d.chunk) (headOf d i) 0))
    {r : I32} (hr : r.toInt = (pmOf d i : Int)) (a : Fin BM) (j : Fin BK) :
    ((P + ((r * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK * (IntV.ofNat d.sXp : I32) +
       (tl.arange 0 BK).row BM * (IntV.ofNat d.sXs : I32))).get2 a j).addr
      = (d.x.ptr (xIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (headOf d i)
          (pmOf d i * BM + (a : Nat)))).addr := by
  rw [get2_scalarAdd', hP]
  refine ptr_off_eq d.x (xIdx d (chunkOf d i * d.chunk) (headOf d i) 0)
    (xIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (headOf d i)
      (pmOf d i * BM + (a : Nat))) _ ?_
  rw [x_off_toInt hd i hr a j, ← Nat.cast_add, idx_x d i a j]

private theorem b_clause0 {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {P : Ptr DType.bf16} (hP : P = d.b.ptr (bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0))
    {r : I32} (hr : r.toInt = (pnOf d i : Int)) (j : Fin BK) (n : Fin BN) :
    ((P + ((r * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK * (IntV.ofNat d.sBn : I32) +
       (tl.arange 0 BK).col BN * (IntV.ofNat d.sBs : I32))).get2 j n).addr
      = (d.b.ptr (bIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (grp d (headOf d i))
          (pnOf d i * BN + (n : Nat)))).addr := by
  rw [get2_scalarAdd', hP]
  refine ptr_off_eq d.b (bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0)
    (bIdx d (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) (grp d (headOf d i))
      (pnOf d i * BN + (n : Nat))) _ ?_
  rw [b_off_toInt hd i hr n j, ← Nat.cast_add, idx_b d i n j]

private theorem dt_clause0 {d : Inst} (_hd : d.ok) (i : Fin (launch d).size)
    {P : Ptr DType.f32} (hP : P = d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) 0)) (j : Fin BK) :
    ((P + tl.arange 0 BK * (IntV.ofNat 1 : I32)).get1 j).addr
      = (d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) (0 * BK + (j : Nat)))).addr := by
  rw [get1_scalarAdd', hP]
  refine ptr_off_eq d.dt (dIdx d (headOf d i) (chunkOf d i) 0)
    (dIdx d (headOf d i) (chunkOf d i) (0 * BK + (j : Nat))) _ ?_
  rw [k_off_toInt j, ← Nat.cast_add, idx_dt d i j]

private theorem dA_clause0 {d : Inst} (_hd : d.ok) (i : Fin (launch d).size)
    {P : Ptr DType.f32} (hP : P = d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) 0)) (j : Fin BK) :
    ((P + tl.arange 0 BK * (IntV.ofNat 1 : I32)).get1 j).addr
      = (d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) (0 * BK + (j : Nat)))).addr := by
  rw [get1_scalarAdd', hP]
  refine ptr_off_eq d.dA (dIdx d (headOf d i) (chunkOf d i) 0)
    (dIdx d (headOf d i) (chunkOf d i) (0 * BK + (j : Nat))) _ ?_
  rw [k_off_toInt j, ← Nat.cast_add, idx_dt d i j]

private theorem seq_clause0 {d : Inst} (_hd : d.ok) (i : Fin (launch d).size)
    {P : Ptr DType.i32} (hP : P = d.seqIdx.ptr (chunkOf d i * d.chunk)) (j : Fin BK) :
    ((P + tl.arange 0 BK * (IntV.ofNat 1 : I32)).get1 j).addr
      = (d.seqIdx.ptr (chunkOf d i * d.chunk + (0 * BK + (j : Nat)))).addr := by
  rw [get1_scalarAdd', hP]
  refine ptr_off_eq d.seqIdx (chunkOf d i * d.chunk)
    (chunkOf d i * d.chunk + (0 * BK + (j : Nat))) _ ?_
  rw [k_off_toInt j]
  push_cast
  ring

/-! `seq`: the base pointer, with the `seqlen` and `chunk` strides of the prelude. -/
private theorem seq_base' {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {pb ps pc c1 o : I64} (hb : pb.toInt = 0) (_hps : ps.toInt = (d.seqlen : Int))
    (hc : pc.toInt = (chunkOf d i : Int)) (hc1 : c1.toInt = (d.chunk : Int))
    (ho : o.toInt = 1) :
    (d.seqIdx.ptr 0 + (pb * ps + pc * c1 * o))
      = d.seqIdx.ptr (chunkOf d i * d.chunk) := by
  have hck : chunkOf d i * d.chunk < 2 ^ 31 := PtrAux.chunkOf_mul_lt hd i
  have t1 : (pb * ps).toInt = 0 := PtrAux.toInt_mul_zero_left hb
  have t2 : (pc * c1).toInt = ((chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.mul_nat hc hc1 (PtrAux.lt_63_of_lt_31 hck)
  have t3 : (pc * c1 * o).toInt = (((chunkOf d i * d.chunk) * 1 : Nat) : Int) :=
    IntV.I64.mul_nat t2 ho (by omega)
  have t4 : (pb * ps + pc * c1 * o).toInt = ((0 + (chunkOf d i * d.chunk) * 1 : Nat) : Int) :=
    IntV.I64.add_nat t1 t3 (by omega)
  apply PtrAux.ptr_add_eq_ptr
  rw [show (chunkOf d i * d.chunk : Nat) = (0 + (chunkOf d i * d.chunk) * 1) by simp]
  exact t4

/-- The fresh accumulator is finite. -/
private theorem zeros_fin (a : Fin BM) (n : Fin BN) :
    ((tl.zeros [BM, BN] DType.f32).get2 a n).isFinite = true := by
  have hz : DType.val? .f32 ((tl.zeros [BM, BN] DType.f32).get2 a n) = some 0 := DType.val?_zero_f32
  exact (DType.val?_f32.mp hz).1

/-- The fresh accumulator, at `t = 0`, is within `DotLoop.bound … 0 …` of the empty sum. -/
private theorem zeros_bound (xh : Nat → Fin BM → ℚ) (bh : Nat → Fin BN → ℚ)
    (a : Fin BM) (n : Fin BN) :
    |(((tl.zeros [BM, BN] DType.f32).get2 a n).toRat : ℝ) -
        ((∑ t' ∈ Finset.range 0, Sf xh bh a n t' : ℚ) : ℝ)|
      ≤ ((DotLoop.bound rho alp 0 (∑ t' ∈ Finset.range 0, Af xh bh a n t') : ℚ) : ℝ) := by
  have hz : DType.val? .f32 ((tl.zeros [BM, BN] DType.f32).get2 a n) = some 0 := DType.val?_zero_f32
  obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hz
  rw [hval]
  simp [DotLoop.bound]

/-- **The invariant holds before the first iteration.** -/
theorem h0_init {d : Inst} {init : Mem} {i : Fin (launch d).size} (hd : d.ok)
    {r1 r2 r3 c1 c2 c3 c4 c5 s1 s2 s3 s4 s5 s6 s7 o1 : Tensor [] I64}
    {r5 r6 : Tensor [] I32}
    (H3 : r3.item.toInt = (chunkOf d i : Int))
    (H5 : r5.item.toInt = (pmOf d i : Int))
    (H6 : r6.item.toInt = (pnOf d i : Int))
    (HC1 : c1.item.toInt = (d.chunk : Int))
    (HC2 : c2.item.toInt = (d.chunk : Int))
    (HC3 : c3.item.toInt = (d.chunk : Int))
    (HC4 : c4.item.toInt = (d.chunk : Int))
    (HC5 : c5.item.toInt = (d.chunk : Int))
    (HS1 : s1.item.toInt = (d.sBs : Int))
    (HS2 : s2.item.toInt = (grp d (headOf d i) * d.sBg : Int))
    (HS3 : s3.item.toInt = (d.sXs : Int))
    (HS4 : s4.item.toInt = (headOf d i * d.sXh : Int))
    (HS5 : s5.item.toInt = (headOf d i * sDh d : Int))
    (HS6 : s6.item.toInt = (headOf d i * sDh d : Int))
    (HS7 : s7.item.toInt = (d.seqlen : Int))
    (HO1 : o1.item.toInt = 1)
    (hpb : (r1.item - r3.item * r2.item).toInt = 0) :
    Inv d init i 0
      (d.x.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat d.sXb : I64) +
              r3.item * c2.item * s3.item + s4.item) +
          ((r5.item * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK * (IntV.ofNat d.sXp : I32) +
            (tl.arange 0 BK).row BM * (IntV.ofNat d.sXs : I32)),
        d.b.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat d.sBb : I64) +
              r3.item * c1.item * s1.item + s2.item) +
          ((r6.item * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK * (IntV.ofNat d.sBn : I32) +
            (tl.arange 0 BK).col BN * (IntV.ofNat d.sBs : I32)),
        d.dt.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
              r3.item * c3.item + s5.item) +
          tl.arange 0 BK * (IntV.ofNat 1 : I32),
        d.dA.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
              r3.item * c4.item + s6.item) +
          tl.arange 0 BK * (IntV.ofNat 1 : I32),
        d.seqIdx.ptr 0 + ((r1.item - r3.item * r2.item) * s7.item + r3.item * c5.item * o1.item) +
          tl.arange 0 BK * (IntV.ofNat 1 : I32),
        tl.zeros [BM, BN] DType.f32)
      init := by
  refine ⟨rfl, ?_⟩
  refine ⟨fun _ _ => 0, fun _ _ => 0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- the `x` tile
    intro a j
    have hxb : d.x.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat d.sXb : I64) +
        r3.item * c2.item * s3.item + s4.item)
        = d.x.ptr (xIdx d (chunkOf d i * d.chunk) (headOf d i) 0) :=
      x_base hd i hpb H3 HC2 HS3 (by rw [HS4]; push_cast; ring)
    exact x_clause0 hd i hxb H5 a j
  · -- the `b` tile
    intro j n
    have hbb : d.b.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat d.sBb : I64) +
        r3.item * c1.item * s1.item + s2.item)
        = d.b.ptr (bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0) :=
      b_base hd i hpb H3 HC1 HS1 (by rw [HS2])
    exact b_clause0 hd i hbb H6 j n
  · -- `dt`
    intro j
    have hdt : d.dt.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
        r3.item * c3.item + s5.item)
        = d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) 0) :=
      dt_base hd i hpb H3 HC3 (by rw [HS5]; push_cast; ring)
    exact dt_clause0 hd i hdt j
  · -- `dA_cumsum`
    intro j
    have hdA : d.dA.ptr 0 + ((r1.item - r3.item * r2.item) * (IntV.ofNat (sDb d) : I64) +
        r3.item * c4.item + s6.item)
        = d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) 0) :=
      dA_base hd i hpb H3 HC4 (by rw [HS6]; push_cast; ring)
    exact dA_clause0 hd i hdA j
  · -- `seq_idx`
    intro j
    have hsq : d.seqIdx.ptr 0 + ((r1.item - r3.item * r2.item) * s7.item +
        r3.item * c5.item * o1.item)
        = d.seqIdx.ptr (chunkOf d i * d.chunk) :=
      seq_base' hd i hpb HS7 H3 HC5 HO1
    exact seq_clause0 hd i hsq j
  · -- the `x` operand, vacuous at `t = 0`
    intro s a hs
    simp only [Nat.zero_mul] at hs
    exact absurd hs (by omega)
  · -- the `b` operand, vacuous at `t = 0`
    intro _ s n hs
    simp only [Nat.zero_mul] at hs
    exact absurd hs (by omega)
  · -- the fresh accumulator
    intro _
    refine ⟨?_, ?_⟩
    · intro a n
      exact zeros_fin a n
    · intro a n
      exact zeros_bound (fun _ _ => 0) (fun _ _ => 0) a n


end Claims.SsdChunkState
