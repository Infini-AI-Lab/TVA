import Claim
import TVA.Proof
import Proof.Geo
import Proof.GeoBound

/-!
# Every written element is written by a program

For every sequence `n`, token `p` in it and value head `hv`, some program's lane is token `p` of
head `hv`: in a fixed-length launch program `((p - lo n) / BT, n * HV + hv)` at lane
`(p - lo n) % BT`, and in a variable-length launch the row `r` of `chunk_indices` that names
`(n, chunk)` (which exists by `TablesOk`) with `bh = hv`, at lane `(p - lo n) % BT`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- **Coverage, fixed length.** -/
theorem cover_fixed {d : Inst} (hd : d.ok) (hv : d.varlen = false)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.HV) :
    ∃ r bh l, r < nChunks d ∧ bh < nRows d ∧ l < d.BT ∧ en d r bh l = true ∧
      nOf d r bh = n ∧ pos d r bh l = p ∧ hvOf d bh = h := by
  have hlo : lo d n = n * d.T := lo_false hv n
  have hhi : hi d n = n * d.T + d.T := hi_false hv n
  have hB : n < d.B := by simpa [nSeq_false hv] using hn
  have hBTpos : 0 < d.BT := bt_pos hd
  set q := p - lo d n with hq
  have hqT : q < d.T := by omega
  have hbh : n * d.HV + h < nRows d := by
    have h1 : (n + 1) * d.HV ≤ d.B * d.HV := Nat.mul_le_mul_right _ (by omega)
    have h2 : (n + 1) * d.HV = n * d.HV + d.HV := by ring
    unfold nRows
    omega
  have hhv : hvOf d (n * d.HV + h) = h := by
    unfold hvOf
    rw [Nat.mul_comm, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hh]
  have hbo : bOf d (n * d.HV + h) = n := by
    unfold bOf
    rw [Nat.mul_comm, Nat.mul_add_div hd.HV_pos, Nat.div_eq_of_lt hh, Nat.add_zero]
  have hnOf : nOf d (q / d.BT) (n * d.HV + h) = n := by rw [nOf_false hv]; exact hbo
  have hchk : chkOf d (q / d.BT) (n * d.HV + h) = q / d.BT := by rw [chkOf_false hv]
  have hlen : lenOf d (q / d.BT) (n * d.HV + h) = d.T := by
    unfold lenOf
    rw [hnOf, hlo, hhi]
    omega
  have hdm : q / d.BT * d.BT + q % d.BT = q := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod q d.BT
  refine ⟨q / d.BT, n * d.HV + h, q % d.BT, ?_, hbh, Nat.mod_lt _ hBTpos, ?_, hnOf, ?_, hhv⟩
  · have hrw : nChunks d = (d.T + d.BT - 1) / d.BT := by simp [nChunks, hv]
    rw [hrw]
    have hqle : q ≤ d.T - 1 := by omega
    have h1 : q / d.BT ≤ (d.T - 1) / d.BT := Nat.div_le_div_right hqle
    have h2 : (d.T + d.BT - 1) / d.BT = (d.T - 1) / d.BT + 1 := by
      have heq : d.T + d.BT - 1 = (d.T - 1) + d.BT := by omega
      rw [heq]
      have h5 := Nat.add_mul_div_left (d.T - 1) 1 hBTpos
      rw [Nat.mul_one] at h5
      exact h5
    omega
  · rw [en_iff, hchk, hlen]
    omega
  · unfold pos baseOf
    rw [hnOf, hchk, hlo]
    omega

/-- **Coverage, variable length.** -/
theorem cover_varlen {d : Inst} (hd : d.ok) (hv : d.varlen = true) (hT : TablesOk d)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.HV) :
    ∃ r bh l, r < nChunks d ∧ bh < nRows d ∧ l < d.BT ∧ en d r bh l = true ∧
      nOf d r bh = n ∧ pos d r bh l = p ∧ hvOf d bh = h := by
  have hnN : n < d.N := by simpa [nSeq_true hv] using hn
  have hlo : lo d n = (cuAt d n).toNat := lo_true hv n
  have hhi : hi d n = (cuAt d (n + 1)).toNat := hi_true hv n
  have hnonneg : 0 ≤ cuAt d n := cuAt_nonneg hT (by omega)
  have hstep : cuAt d n ≤ cuAt d (n + 1) := cuAt_step hT hnN
  have hBA : (cuAt d n).toNat ≤ (cuAt d (n + 1)).toNat := Int.toNat_le_toNat hstep
  have hBTpos : 0 < d.BT := bt_pos hd
  have hhBH : h < nRows d := by
    have h1 : d.HV ≤ d.B * d.HV := Nat.le_mul_of_pos_left d.HV hd.B_pos
    unfold nRows
    omega
  set q := p - lo d n with hq
  have hq_len : q < hi d n - lo d n := by omega
  have hqInt : (q : Int) < cuAt d (n + 1) - cuAt d n := by
    have h1 : (q : Int) < ((cuAt d (n + 1)).toNat : Int) - ((cuAt d n).toNat : Int) := by
      have hq' : q < (cuAt d (n + 1)).toNat - (cuAt d n).toNat := by
        have h9 := hq_len
        rw [hlo, hhi] at h9
        exact h9
      have hcast : (((cuAt d (n + 1)).toNat - (cuAt d n).toNat : Nat) : Int)
          = ((cuAt d (n + 1)).toNat : Int) - ((cuAt d n).toNat : Int) := Nat.cast_sub hBA
      have h2 : (q : Int) < (((cuAt d (n + 1)).toNat - (cuAt d n).toNat : Nat) : Int) := by
        exact_mod_cast hq'
      rw [hcast] at h2
      exact h2
    rw [Int.toNat_of_nonneg (le_trans hnonneg hstep), Int.toNat_of_nonneg hnonneg] at h1
    exact h1
  have hcInt : (((q / d.BT) * d.BT : Nat) : Int) < cuAt d (n + 1) - cuAt d n := by
    have hle : q / d.BT * d.BT ≤ q := Nat.div_mul_le_self q d.BT
    have hcast : (((q / d.BT) * d.BT : Nat) : Int) ≤ (q : Int) := Int.ofNat_le.mpr hle
    omega
  obtain ⟨r, hrlt, hseq, hchkr⟩ := hT.2.2.2.2.2 n (q / d.BT) hnN hcInt
  have hhv : hvOf d h = h := Nat.mod_eq_of_lt hh
  have hnOf : nOf d r h = n := by
    rw [nOf_true hv, hseq]
    exact Int.toNat_natCast n
  have hchk : chkOf d r h = q / d.BT := by
    rw [chkOf_true hv, hchkr]
    exact Int.toNat_natCast (q / d.BT)
  have hlen : lenOf d r h = hi d n - lo d n := by
    unfold lenOf
    rw [hnOf]
  have hdm : q / d.BT * d.BT + q % d.BT = q := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod q d.BT
  refine ⟨r, h, q % d.BT, ?_, hhBH, Nat.mod_lt _ hBTpos, ?_, hnOf, ?_, hhv⟩
  · simpa [nChunks, hv] using hrlt
  · rw [en_iff, hchk, hlen]
    omega
  · unfold pos baseOf
    rw [hnOf, hchk]
    omega

/-- **Coverage**: every token of every sequence and every value head is some program's enabled
lane. -/
theorem cover_exists {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.HV) :
    ∃ r bh l, r < nChunks d ∧ bh < nRows d ∧ l < d.BT ∧ en d r bh l = true ∧
      nOf d r bh = n ∧ pos d r bh l = p ∧ hvOf d bh = h := by
  cases hvv : d.varlen
  · exact cover_fixed hd hvv hn hp1 hp2 hh
  · exact cover_varlen hd hvv (hT hvv) hn hp1 hp2 hh

end Claims.ChunkBwdDvLocal
