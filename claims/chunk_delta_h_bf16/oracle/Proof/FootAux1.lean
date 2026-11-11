import Proof.Defs

/-!
# `chunk_delta_h_bf16`: arithmetic of the cells, sequences and chunks

Mixed-radix numbers (the element maps `hIdx`, `uIdx`, `sIdx`), the order of `cu_seqlens` and of
the sequences' first chunks, and the bounds of the cells a program writes.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH
open TVA

/-- Two numbers `x * D + e` with `e < D` determine `x` and `e`. -/
theorem pair_inj {x x' e e' D : Nat} (he : e < D) (he' : e' < D)
    (h : x * D + e = x' * D + e') : x = x' ∧ e = e' := by
  rcases lt_trichotomy x x' with hx | hx | hx
  · have := Nat.mul_le_mul_right D (Nat.succ_le_of_lt hx); rw [Nat.succ_mul] at this; omega
  · subst hx; omega
  · have := Nat.mul_le_mul_right D (Nat.succ_le_of_lt hx); rw [Nat.succ_mul] at this; omega

/-- `x * D + e < X * D` when `x < X` and `e < D`. -/
theorem pair_lt {x X e D : Nat} (hx : x < X) (he : e < D) : x * D + e < X * D := by
  have := Nat.mul_le_mul_right D (Nat.succ_le_of_lt hx); rw [Nat.succ_mul] at this; omega

/-- A four-digit mixed-radix number determines its digits. -/
theorem idx4_inj {B C D a a' b b' c c' e e' : Nat} (hb : b < B) (hb' : b' < B) (hc : c < C)
    (hc' : c' < C) (he : e < D) (he' : e' < D)
    (h : ((a * B + b) * C + c) * D + e = ((a' * B + b') * C + c') * D + e') :
    a = a' ∧ b = b' ∧ c = c' ∧ e = e' := by
  obtain ⟨h1, h2⟩ := pair_inj he he' h
  obtain ⟨h3, h4⟩ := pair_inj hc hc' h1
  obtain ⟨h5, h6⟩ := pair_inj hb hb' h3
  exact ⟨h5, h6, h4, h2⟩

/-- A four-digit mixed-radix number is below the product of the radices. -/
theorem idx4_lt {A B C D a b c e : Nat} (ha : a < A) (hb : b < B) (hc : c < C) (he : e < D) :
    ((a * B + b) * C + c) * D + e < A * B * C * D :=
  pair_lt (pair_lt (pair_lt ha hb) hc) he

/-! ## Sequences and chunks -/

/-- `cu_seqlens` is nondecreasing up to `N`. -/
theorem cuAt_mono {d : Inst} (hd : d.ok) (hv : d.varlen = true) {m n : Nat} (hmn : m ≤ n)
    (hn : n ≤ d.N) : cuAt d m ≤ cuAt d n := by
  induction n, hmn using Nat.le_induction with
  | base => exact le_rfl
  | succ n hmn ih =>
    refine le_trans (ih (by omega)) ?_
    unfold cuAt
    exact Int.toNat_le_toNat (hd.var_mono hv n (by omega))

/-- `cu_seqlens[N] = T`. -/
theorem cuAt_N {d : Inst} (hd : d.ok) (hv : d.varlen = true) : cuAt d d.N = d.T := by
  unfold cuAt; rw [hd.var_end hv]; simp

/-- A sequence's tokens end by `T`. -/
theorem start_add_len_le {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) :
    startOf d n + lenOf d n ≤ d.T := by
  by_cases hv : d.varlen = true
  · simp only [startOf, lenOf, hv, ite_true]
    have h1 := cuAt_mono hd hv (m := n) (n := n + 1) (by omega) (by omega)
    have h2 := cuAt_mono hd hv (m := n + 1) (n := d.N) (by omega) le_rfl
    have h3 := cuAt_N hd hv
    omega
  · simp only [startOf, lenOf, hv]; simp

/-- Token `s` of sequence `n` is a token of its row. -/
theorem tok_lt {d : Inst} (hd : d.ok) {n s : Nat} (hn : n < d.N) (hs : s < lenOf d n) :
    startOf d n + s < d.T := by
  have := start_add_len_le hd hn
  omega

/-- The row of a sequence is a row. -/
theorem row_lt {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) : rowOf d n < d.B := by
  by_cases hv : d.varlen = true
  · simp only [rowOf, hv, ite_true]; rw [hd.var_B hv]; omega
  · simp only [rowOf, hv]; simpa [hd.fixed_N (by simpa using hv)] using hn

/-- A sequence's tokens lie before the first token of a later sequence (as flat positions of the
packed batch or the rows). -/
theorem tok_lt_next {d : Inst} (hd : d.ok) {n n' s : Nat} (hn : n < n') (hn' : n' < d.N)
    (hs : s < lenOf d n) :
    rowOf d n * d.T + (startOf d n + s) < rowOf d n' * d.T + startOf d n' := by
  by_cases hv : d.varlen = true
  · simp only [startOf, lenOf, rowOf, hv, ite_true] at hs ⊢
    have h1 := cuAt_mono hd hv (m := n + 1) (n := n') (by omega) (by omega)
    omega
  · simp only [startOf, lenOf, rowOf, hv] at hs ⊢
    simp only [Bool.false_eq_true, ite_false] at hs ⊢
    have := Nat.mul_le_mul_right d.T (Nat.succ_le_of_lt hn)
    rw [Nat.succ_mul] at this
    omega

/-- A fixed-length sequence has `cdiv(T, 64)` chunks. -/
theorem nch_fixed {d : Inst} (hv : d.varlen = false) (n : Nat) :
    nch d n = (d.T + BT - 1) / BT := by
  simp [nch, lenOf, hv]

/-- The first chunk of the next sequence follows this sequence's chunks. -/
theorem cbase_succ {d : Inst} (n : Nat) : cbase d (n + 1) = cbase d n + nch d n := by
  by_cases hv : d.varlen = true
  · simp [cbase, hv, List.range_succ]
  · have hv' : d.varlen = false := by simpa using hv
    simp only [cbase, hv, Bool.false_eq_true, ite_false]
    rw [nch_fixed hv']; ring

/-- The first chunks of sequences are nondecreasing. -/
theorem cbase_mono {d : Inst} {n n' : Nat} (h : n ≤ n') : cbase d n ≤ cbase d n' := by
  induction n', h using Nat.le_induction with
  | base => exact le_rfl
  | succ n' _ ih => rw [cbase_succ]; omega

/-- The chunks of sequence `n` come before the first chunk of a later sequence. -/
theorem cbase_add_nch_le {d : Inst} {n n' : Nat} (h : n < n') : cbase d n + nch d n ≤ cbase d n' := by
  rw [← cbase_succ]; exact cbase_mono h

/-- A chunk of a sequence is a chunk of `h`. -/
theorem chunk_lt {d : Inst} (hd : d.ok) {n t : Nat} (hn : n < d.N) (ht : t < nch d n) :
    cbase d n + t < d.B * NTt d := by
  by_cases hv : d.varlen = true
  · have h1 := cbase_add_nch_le (d := d) (n := n) (n' := d.N) hn
    have : NTt d = cbase d d.N := by simp [NTt, hv]
    rw [this, hd.var_B hv]; omega
  · have hv' : d.varlen = false := by simpa using hv
    have h1 := cbase_add_nch_le (d := d) (n := n) (n' := d.N) hn
    have h2 : cbase d d.N = d.N * ((d.T + BT - 1) / BT) := by simp [cbase, hv']
    have h3 : NTt d = (d.T + BT - 1) / BT := by simp [NTt, hv']
    rw [h3, ← hd.fixed_N hv']
    omega

/-! ## Bounds of the element maps -/

/-- The cells of `h` are inside its tensor. -/
theorem hIdx_lt {d : Inst} {c₀ hv c j : Nat} (hc₀ : c₀ < d.B * NTt d) (hhv : hv < d.HV)
    (hc : c < d.K) (hj : j < d.V) : hIdx d c₀ hv c j < d.B * NTt d * d.HV * d.K * d.V := by
  unfold hIdx
  split_ifs
  · exact lt_of_lt_of_eq (idx4_lt hc₀ hhv hj hc) (by ring)
  · exact idx4_lt hc₀ hhv hc hj

/-- The cells of `v_new` are inside its tensor. -/
theorem uIdx_lt {d : Inst} {b t hv j : Nat} (hb : b < d.B) (ht : t < d.T) (hhv : hv < d.HV)
    (hj : j < d.V) : uIdx d b t hv j < d.B * d.T * d.HV * d.V :=
  idx4_lt hb ht hhv hj

/-- The cells of `ht` are inside its tensor. -/
theorem sIdx_lt {d : Inst} {n hv c j : Nat} (hn : n < d.N) (hhv : hv < d.HV) (hc : c < d.K)
    (hj : j < d.V) : sIdx d n hv c j < d.N * d.HV * d.K * d.V := by
  unfold sIdx
  split_ifs
  · exact lt_of_lt_of_eq (idx4_lt hn hhv hj hc) (by ring)
  · exact idx4_lt hn hhv hc hj

/-- The tokens of chunk `t` are tokens of the sequence. -/
theorem cl_lt {d : Inst} {n t i : Nat} (hi : i < cl d n t) : BT * t + i < lenOf d n := by
  unfold cl BT at *; omega

end Claims.ChunkDeltaH
