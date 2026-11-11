import Claim
import TVA.Proof
import Proof.Defs

/-!
# The geometry of a program

For a program of the grid, its value head is below `HV`, its chunk starts inside its sequence, and
its sequence lies inside the batch (`geo`). Packed, the sequence comes from `cu_seqlens` and
`chunk_indices` (`SeqOk`).
-/

namespace Claims.RecomputeWU

open TVA

/-- `cu_seqlens` is non-decreasing up to `N`. -/
theorem cuN_mono {d : Inst} {init : Mem} (hS : SeqOk d init) :
    ∀ n n', n ≤ n' → n' ≤ d.N → cuN d init n ≤ cuN d init n' := by
  intro n n' hn
  induction n', hn using Nat.le_induction with
  | base => intro _; exact le_rfl
  | succ n' hn ih =>
    intro h
    exact (ih (by omega)).trans (hS.cu_mono n' (by omega))

/-- The sequence `n < N` ends inside the batch. -/
theorem cuN_succ_le {d : Inst} {init : Mem} (hS : SeqOk d init) {n : Nat} (hn : n < d.N) :
    cuN d init (n + 1) ≤ d.T := by
  have := cuN_mono hS (n + 1) d.N (by omega) le_rfl
  rwa [hS.cu_last] at this

/-- `cu_seqlens[n] + (cu_seqlens[n+1] - cu_seqlens[n]) = cu_seqlens[n+1]`. -/
theorem seqLen_add {d : Inst} {init : Mem} (hS : SeqOk d init) {n : Nat} (hn : n < d.N) :
    cuN d init n + seqLen d init n = cuN d init (n + 1) := by
  have := hS.cu_mono n hn
  unfold seqLen; omega

/-- A padded grid's axis 0 is `cdiv(T, 64)`. -/
theorem g0_padded {d : Inst} (h : d.isVarlen = false) : g0 d = (d.T + 63) / 64 := by
  simp [g0, h]

/-- A packed grid's axis 0 is `len(chunk_indices)`. -/
theorem g0_varlen {d : Inst} (h : d.isVarlen = true) : g0 d = d.NT := by
  simp [g0, h]

/-- A program of the grid: its coordinates are in range. -/
theorem coords {d : Inst} (hd : d.ok) {i : Nat} (hi : i < g0 d * (d.B * d.HV)) :
    p0Of d i < g0 d ∧ bhOf d i < d.B * d.HV ∧ hOf d i < d.HV ∧ bOf d i < d.B := by
  have hg : 0 < g0 d := by
    rcases Nat.eq_zero_or_pos (g0 d) with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have hbh : 0 < d.B * d.HV := by
    rcases Nat.eq_zero_or_pos (d.B * d.HV) with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have hHV : 0 < d.HV := hd.hv_pos
  have hp : p0Of d i < g0 d := Nat.mod_lt _ hg
  have hb : bhOf d i < d.B * d.HV := Nat.mod_lt _ hbh
  refine ⟨hp, hb, Nat.mod_lt _ hHV, ?_⟩
  unfold bOf
  exact (Nat.div_lt_iff_lt_mul hHV).mpr hb

/-- **The geometry**: a program's chunk starts inside its sequence, and the sequence lies in the
batch. -/
theorem geo {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) {i : Nat}
    (hi : i < g0 d * (d.B * d.HV)) :
    itOf d init i * 64 < lenOf d init i ∧ s0Of d init i + lenOf d init i ≤ nTok d := by
  obtain ⟨hp, hb, hh, hbB⟩ := coords hd hi
  by_cases hv : d.isVarlen = true
  · have hS := hpre.seqs hv
    have hNT : p0Of d i < d.NT := by rw [← g0_varlen hv]; exact hp
    obtain ⟨hn, hc⟩ := hS.ci_ok _ hNT
    have hB := hd.varlen_b hv
    simp only [itOf, lenOf, s0Of, hv, ite_true]
    refine ⟨hc, ?_⟩
    have h1 := seqLen_add hS hn
    have h2 := cuN_succ_le hS hn
    unfold nTok; rw [hB]; omega
  · have hv' : d.isVarlen = false := by simpa using hv
    simp only [itOf, lenOf, s0Of, hv', Bool.false_eq_true, ite_false]
    have hp' : p0Of d i < (d.T + 63) / 64 := by rw [← g0_padded hv']; exact hp
    refine ⟨by omega, ?_⟩
    unfold nTok
    have : (bOf d i + 1) * d.T ≤ d.B * d.T := Nat.mul_le_mul_right _ hbB
    nlinarith

end Claims.RecomputeWU
