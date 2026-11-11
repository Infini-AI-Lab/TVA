import Claim
import TVA.Proof
import Proof.Geo
import Proof.GeoBound

/-!
# Every written element is written by a program

For every sequence `n`, position `p` in it and head `h`, some program's lane touches element
`p * H + h`: in a fixed-length launch program `((p - lo n) / BT) * B * H + n * H + h` at lane
`(p - lo n) % BT`, and in a variable-length launch the row of `chunk_indices` that names `(n,
chunk)` (which exists by `TablesOk`) at lane `(p - lo n) % BT`.
-/

namespace Claims.ChunkCumsum

open TVA

/-- **Coverage, fixed length**: every position of every sequence and every head is written by some enabled lane of some program. -/
theorem cover_fixed {d : Inst} (hd : d.ok) (hv : d.varlen = false)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.H) :
    ∃ i l, i < gridSize d ∧ l < d.BT ∧ en d i l = true ∧ nOf d i = n ∧
      pos d i l = p ∧ headOf d i = h := by
  have hlo : lo d n = n * d.T := lo_false hv n
  have hhi : hi d n = n * d.T + d.T := hi_false hv n
  have hB : n < d.B := by simpa [nSeq_false hv] using hn
  have hTpos : 0 < d.T := by omega
  have hBTpos : 0 < d.BT := by
    obtain ⟨e, he⟩ := hd.bt_pow2
    rw [he]; positivity
  have hBHpos : 0 < BH d := BH_pos hd
  set q := p - lo d n with hq
  have hqT : q < d.T := by omega
  have hr : n * d.H + h < BH d := by
    have h1 : n * d.H < d.B * d.H := Nat.mul_lt_mul_of_pos_right hB hd.H_pos
    have h3 : n * d.H + h < (n + 1) * d.H := by
      have h4 : (n + 1) * d.H = n * d.H + d.H := by ring
      omega
    have h2 : (n + 1) * d.H ≤ d.B * d.H := Nat.mul_le_mul_right _ (by omega)
    unfold BH
    omega
  set i := BH d * (q / d.BT) + (n * d.H + h) with hieq
  set l := q % d.BT with hl
  have hrow : row d i = q / d.BT := by
    unfold row
    rw [hieq, Nat.mul_add_div hBHpos, Nat.div_eq_of_lt hr, Nat.add_zero]
  have hlin : lin d i = n * d.H + h := by
    unfold lin
    rw [hieq, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hr]
  have hlin' : lin d i = d.H * n + h := by rw [hlin, Nat.mul_comm n d.H]
  have hbatch : batchOf d i = n := by
    unfold batchOf
    rw [hlin', Nat.mul_add_div hd.H_pos, Nat.div_eq_of_lt hh, Nat.add_zero]
  have hhead : headOf d i = h := by
    unfold headOf
    rw [hlin', Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hh]
  have hnOf : nOf d i = n := by rw [nOf_false hv]; exact hbatch
  have hchk : chkOf d i = q / d.BT := by rw [chkOf_false hv]; exact hrow
  have hbase : baseOf d i = n * d.T := by
    unfold baseOf
    rw [hnOf, hlo]
  have hlen : lenOf d i = d.T := by
    unfold lenOf
    rw [hnOf, hlo, hhi]
    omega
  have hdm : q / d.BT * d.BT + q % d.BT = q := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod q d.BT
  refine ⟨i, l, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [gridSize_eq]
    have hrw : rowsOf d = (d.T + d.BT - 1) / d.BT := by simp [rowsOf, hv]
    rw [hrw, hieq]
    have hqrows : q / d.BT < (d.T + d.BT - 1) / d.BT := by
      have hqle : q ≤ d.T - 1 := by omega
      have h1 : q / d.BT ≤ (d.T - 1) / d.BT := Nat.div_le_div_right hqle
      have h2 : (d.T + d.BT - 1) / d.BT = (d.T - 1) / d.BT + 1 := by
        have heq : d.T + d.BT - 1 = (d.T - 1) + d.BT := by omega
        rw [heq]
        have h5 := Nat.add_mul_div_left (d.T - 1) 1 hBTpos
        rw [Nat.mul_one] at h5
        exact h5
      omega
    calc BH d * (q / d.BT) + (n * d.H + h) < BH d * (q / d.BT) + BH d := by omega
      _ = BH d * (q / d.BT + 1) := by ring
      _ ≤ BH d * ((d.T + d.BT - 1) / d.BT) := Nat.mul_le_mul_left _ (by omega)
      _ = (d.T + d.BT - 1) / d.BT * d.B * d.H := by unfold BH; ring
  · rw [hl]; exact Nat.mod_lt _ hBTpos
  · rw [en_iff, hchk, hlen, hl]
    omega
  · exact hnOf
  · unfold pos
    rw [hbase, hchk, hl]
    omega
  · exact hhead

/-- **Coverage, variable length**: every position of every sequence and every head is written by some enabled lane of some program. -/
theorem cover_varlen {d : Inst} (hd : d.ok) (hv : d.varlen = true) (hT : TablesOk d)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.H) :
    ∃ i l, i < gridSize d ∧ l < d.BT ∧ en d i l = true ∧ nOf d i = n ∧
      pos d i l = p ∧ headOf d i = h := by
  have hnN : n < d.N := by simpa [nSeq_true hv] using hn
  have hlo : lo d n = (cuAt d n).toNat := lo_true hv n
  have hhi : hi d n = (cuAt d (n + 1)).toNat := hi_true hv n
  have hnonneg : 0 ≤ cuAt d n := cuAt_nonneg hT (by omega)
  have hstep : cuAt d n ≤ cuAt d (n + 1) := cuAt_step hT hnN
  have hBA : (cuAt d n).toNat ≤ (cuAt d (n + 1)).toNat := Int.toNat_le_toNat hstep
  have hBTpos : 0 < d.BT := by
    obtain ⟨e, he⟩ := hd.bt_pow2
    rw [he]; positivity
  have hBHpos : 0 < BH d := BH_pos hd
  have hhBH : h < BH d := by
    have h1 : d.H ≤ d.B * d.H := Nat.le_mul_of_pos_left d.H hd.B_pos
    unfold BH
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
  set i := BH d * r + h with hieq
  set l := q % d.BT with hl
  have hrow : row d i = r := by
    unfold row
    rw [hieq, Nat.mul_add_div hBHpos, Nat.div_eq_of_lt hhBH, Nat.add_zero]
  have hlin : lin d i = h := by
    unfold lin
    rw [hieq, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hhBH]
  have hhead : headOf d i = h := by
    unfold headOf
    rw [hlin, Nat.mod_eq_of_lt hh]
  have hnOf : nOf d i = n := by
    rw [nOf_true hv, hrow, hseq]
    exact Int.toNat_natCast n
  have hchk : chkOf d i = q / d.BT := by
    rw [chkOf_true hv, hrow, hchkr]
    exact Int.toNat_natCast (q / d.BT)
  have hbase : baseOf d i = lo d n := by
    rw [baseOf_true hv, hnOf, ← hlo]
  have hlen : lenOf d i = hi d n - lo d n := by
    unfold lenOf
    rw [hnOf]
  have hdm : q / d.BT * d.BT + q % d.BT = q := by
    rw [Nat.mul_comm]
    exact Nat.div_add_mod q d.BT
  refine ⟨i, l, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [gridSize_eq]
    have hrw : rowsOf d = d.NT := by simp [rowsOf, hv]
    rw [hrw, hieq]
    calc BH d * r + h < BH d * r + BH d := by omega
      _ = BH d * (r + 1) := by ring
      _ ≤ BH d * d.NT := Nat.mul_le_mul_left _ (by omega)
      _ = d.NT * d.B * d.H := by unfold BH; ring
  · rw [hl]; exact Nat.mod_lt _ hBTpos
  · rw [en_iff, hchk, hlen, hl]
    omega
  · exact hnOf
  · unfold pos
    rw [hbase, hchk, hl]
    omega
  · exact hhead

/-- Every written element is touched by an enabled lane of some program. -/
theorem cover_exists {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.H) :
    ∃ i l, i < gridSize d ∧ l < d.BT ∧ en d i l = true ∧ nOf d i = n ∧
      pos d i l = p ∧ headOf d i = h := by
  cases hvv : d.varlen
  · exact cover_fixed hd hvv hn hp1 hp2 hh
  · exact cover_varlen hd hvv (hT hvv) hn hp1 hp2 hh

/-- The element identity that `Post` needs. -/
theorem cover_elem {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {n p h : Nat} (hn : n < nSeq d) (hp1 : lo d n ≤ p) (hp2 : p < hi d n) (hh : h < d.H) :
    ∃ i l, i < gridSize d ∧ en d i l = true ∧ nOf d i = n ∧ headOf d i = h ∧
      elem d i l = p * d.H + h := by
  obtain ⟨i, l, hi, -, he, hn', hp, hh'⟩ := cover_exists hd hT hn hp1 hp2 hh
  refine ⟨i, l, hi, he, hn', hh', ?_⟩
  unfold elem
  rw [hp, hh']

end Claims.ChunkCumsum
