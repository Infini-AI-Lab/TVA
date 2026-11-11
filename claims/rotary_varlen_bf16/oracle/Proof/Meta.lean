import Proof.Basic

/-!
# Basic bounds on a program's indices

Program `i` of a valid launch has `ax0 d i < g0 d`, `ax1 d i < B`, `ax2 d i < H`; an enabled lane
of program `i` has its flat token inside the launch's tokens.
-/

namespace Claims.RotaryVarlen

open TVA

/-- Grid axis 0 of a program of the launch is below its extent. -/
theorem ax0_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) : ax0 d i < g0 d := by
  have h := hi
  rw [launch_size] at h
  have hg : 0 < g0 d := by
    rcases Nat.eq_zero_or_pos (g0 d) with h0 | hpos
    · rw [h0] at h; simp at h
    · exact hpos
  exact Nat.mod_lt _ hg

/-- Grid axis 1 of a program of the launch is below the batch size. -/
theorem ax1_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) : ax1 d i < d.B := by
  have h := hi
  rw [launch_size] at h
  have hB : 0 < d.B := by
    rcases Nat.eq_zero_or_pos d.B with h0 | hpos
    · rw [h0] at h; simp at h
    · exact hpos
  exact Nat.mod_lt _ hB

/-- Grid axis 2 of a program of the launch is below the number of heads. -/
theorem ax2_lt {d : Inst} {i : Nat} (hi : i < (launch d).size) : ax2 d i < d.H := by
  have h := hi
  rw [launch_size] at h
  have hp : 0 < g0 d * d.B := by
    rcases Nat.eq_zero_or_pos (g0 d * d.B) with h0 | hpos
    · rw [h0] at h; simp at h
    · exact hpos
  rw [ax2, Nat.div_lt_iff_lt_mul hp]
  calc i < g0 d * d.B * d.H := h
    _ = d.H * (g0 d * d.B) := Nat.mul_comm _ _

/-- `cu_seqlens` is non-decreasing. -/
theorem cuN_mono_le {d : Inst} {init : Mem} (hs : SeqOk d init) {n n' : Nat}
    (hnn' : n ≤ n') (hN : n' ≤ d.N) : cuN d init n ≤ cuN d init n' := by
  induction n' with
  | zero =>
      have : n = 0 := by omega
      subst this; exact le_rfl
  | succ k ih =>
      rcases Nat.eq_or_lt_of_le hnn' with he | hlt
      · subst he; exact le_rfl
      · have h1 : cuN d init n ≤ cuN d init k := ih (by omega) (by omega)
        have h2 := hs.cu_mono k (by omega)
        omega

/-- Program `i`'s sequence is one of the launch's. -/
theorem seqI_lt {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) : seqI d init i < nSeq d := by
  by_cases hv : d.isVarlen = true
  · have hs := hpre.seqs hv
    have hg : g0 d = d.NT := by simp [g0, hv]
    have hax : ax0 d i < d.NT := by rw [← hg]; exact ax0_lt hi
    have hci := hs.ci_ok (ax0 d i) hax
    rw [seqI_true hv]
    simp only [nSeq, hv, ite_true]
    exact hci.1
  · have hv2 : d.isVarlen = false := by cases h : d.isVarlen <;> simp_all
    have hax := ax1_lt (d := d) hi
    rw [seqI_false hv2]
    simp only [nSeq, hv2]
    exact hax

/-- An enabled lane's flat token is one of the launch's tokens. -/
theorem tokI_lt {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {i l : Nat}
    (hi : i < (launch d).size) (hl : laneOn d init i l) : tokI d init i l < nTok d := by
  obtain ⟨hlloc, -, -⟩ := hl
  by_cases hv : d.isVarlen = true
  · have hs := hpre.seqs hv
    have hg : g0 d = d.NT := by simp [g0, hv]
    have hax : ax0 d i < d.NT := by rw [← hg]; exact ax0_lt hi
    have hci := hs.ci_ok (ax0 d i) hax
    set n := seqI d init i with hn
    have hnlt : n < d.N := by rw [hn, seqI_true hv]; exact hci.1
    have hlen : lenI d init i = seqLen d init n := by
      simp [lenI, lenAt, hv, hn]
    have hbos : bosI d init i = cuN d init n := by
      simp [bosI, bosAt, hv, hn]
    have hlocN : locI d init i l < seqLen d init n := by
      have := hlloc; rw [hlen] at this; exact Int.ofNat_lt.mp this
    have hmono : cuN d init n ≤ cuN d init (n + 1) := hs.cu_mono n (by omega)
    have hstep : cuN d init (n + 1) ≤ cuN d init d.N := cuN_mono_le hs (by omega) le_rfl
    have hlast : cuN d init d.N = d.T := hs.cu_last
    have hseqLen : seqLen d init n = cuN d init (n + 1) - cuN d init n := rfl
    rw [tokI, hbos]
    simp only [nTok, hv, ite_true]
    omega
  · have hv2 : d.isVarlen = false := by cases h : d.isVarlen <;> simp_all
    have hax := ax1_lt (d := d) hi
    have hlocT : locI d init i l < d.T := by
      have := hlloc
      simp [lenI, lenAt, hv2] at this
      exact this
    have hmul : ax1 d i * d.T + d.T ≤ d.B * d.T := by
      rw [← Nat.succ_mul]
      exact Nat.mul_le_mul_right _ (by omega)
    have hbos : bosI d init i = ax1 d i * d.T := by
      simp [bosI, bosAt, hv2, seqI_false hv2]
    have hntok : nTok d = d.B * d.T := by simp [nTok, hv2]
    rw [tokI, hbos, hntok]
    omega

end Claims.RotaryVarlen
