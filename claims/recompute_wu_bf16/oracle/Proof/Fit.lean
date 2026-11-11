import Claim
import TVA.Proof
import Proof.Geo

/-!
# Integer arithmetic of the offsets

The `int64` offsets of the kernel stay far below `2^62` on a program of the grid: row offsets and
strides are bounded through `Inst.ok.size_lt`.
-/

namespace Claims.RecomputeWU

open TVA

/-- `HV < 2^62`. -/
theorem HV_small {d : Inst} (hd : d.ok) : d.HV < 2 ^ 62 := by
  have h := hd.size_lt
  have h1 : 64 ≤ nTok d + 64 := by omega
  have h2 : 64 ≤ d.K + d.V + 64 := by omega
  have h3 : d.HV ≤ (nTok d + 64) * d.HV * (d.K + d.V + 64) := by
    have := Nat.mul_le_mul h1 (le_refl d.HV)
    nlinarith [hd.hv_pos]
  omega

/-- `H ≤ HV`. -/
theorem H_le_HV {d : Inst} (hd : d.ok) : d.H ≤ d.HV := by
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hd.hv_div
  have : 0 < q := by
    rcases Nat.eq_zero_or_pos q with h | h
    · rw [h] at hq; have := hd.hv_pos; omega
    · exact h
  rw [hq]; nlinarith

/-- A row offset times a stride: `(it·64 + 63) · s < 2^62` for a stride `s ≤ HV·(K + V + 64)`. -/
theorem fit_stride {d : Inst} (hd : d.ok) {s0 len it s : Nat} (hs0 : s0 + len ≤ nTok d)
    (hl : it * 64 < len) (hs : s ≤ d.HV * (d.K + d.V + 64)) : (it * 64 + 63) * s < 2 ^ 62 := by
  have h1 : it * 64 + 63 ≤ nTok d + 64 := by omega
  calc (it * 64 + 63) * s ≤ (nTok d + 64) * (d.HV * (d.K + d.V + 64)) := Nat.mul_le_mul h1 hs
    _ = (nTok d + 64) * d.HV * (d.K + d.V + 64) := by ring
    _ < 2 ^ 62 := hd.size_lt

/-- A flat offset `(f·X + h)·Y + c` of a token `f < nTok + 64`. -/
theorem fit_idx {d : Inst} (hd : d.ok) {f X h Y c : Nat} (hf : f < nTok d + 64) (hX : X ≤ d.HV)
    (hh : h < X) (hY : Y ≤ d.K + d.V + 64) (hc : c < Y) : (f * X + h) * Y + c < 2 ^ 62 := by
  have := idx_lt hf hh hc
  calc _ < (nTok d + 64) * X * Y := this
    _ ≤ (nTok d + 64) * d.HV * (d.K + d.V + 64) :=
        Nat.mul_le_mul (Nat.mul_le_mul_left _ hX) hY
    _ < 2 ^ 62 := hd.size_lt


/-- `len · s < 2^62` for a stride `s ≤ HV·(K + V + 64)`. -/
theorem fit_len {d : Inst} (hd : d.ok) {s0 len s : Nat} (hs0 : s0 + len ≤ nTok d)
    (hs : s ≤ d.HV * (d.K + d.V + 64)) : len * s < 2 ^ 62 := by
  calc len * s ≤ (nTok d + 64) * (d.HV * (d.K + d.V + 64)) :=
        Nat.mul_le_mul (by omega) hs
    _ = (nTok d + 64) * d.HV * (d.K + d.V + 64) := by ring
    _ < 2 ^ 62 := hd.size_lt

/-- A tile of the buffer of an `[·, X, Y]` tensor lies in it. -/
theorem tile_lt {nT s0 len X h Y N L : Nat} (hs : s0 + len ≤ nT) (hl : 1 ≤ len) (hh : h < X)
    (hN : 0 < N) (hNY : N ≤ Y) (hL : nT * X * Y ≤ L) :
    (s0 * X + h) * Y + (len - 1) * (X * Y) + (N - 1) * 1 < L := by
  have h1 := idx_lt (f := s0 + len - 1) (X := X) (h := h) (Y := Y) (c := N - 1) (n := nT)
    (by omega) hh (by omega)
  have h2 : (s0 * X + h) * Y + (len - 1) * (X * Y) + (N - 1) * 1 =
      ((s0 + len - 1) * X + h) * Y + (N - 1) := by
    obtain ⟨l, rfl⟩ : ∃ l, len = l + 1 := ⟨len - 1, by omega⟩
    simp only [Nat.add_sub_cancel]
    have : s0 + (l + 1) - 1 = s0 + l := by omega
    rw [this]; ring
  omega


/-- The scalar `(bos·X + i_h) · Y` is `(s0·X + h)·Y`, for `X ≤ HV`, `h < X`, `Y ≤ K + V + 64`. -/
theorem base_val {d : Inst} (hd : d.ok) {bos i_h : I64} {s0 h X Y : Nat} (hs : s0 < nTok d + 64)
    (hX : X ≤ d.HV) (hh : h < X) (hY0 : 0 < Y) (hY : Y ≤ d.K + d.V + 64) (hbos : bos.toInt = s0)
    (hih : i_h.toInt = h) :
    ((bos * (IntV.ofNat X : I64) + i_h) * (IntV.ofNat Y : I64)).toInt = (((s0 * X + h) * Y : Nat) : Int) := by
  have hXs : X < 2 ^ 62 := lt_of_le_of_lt hX (HV_small hd)
  have hYs : Y < 2 ^ 62 := by have := hd.V_lt; have := hd.K_lt; omega
  have f1 := fit_idx hd (f := s0) (X := X) (h := 0) (Y := 1) (c := 0) hs hX (by omega) (by omega)
    (by omega)
  have f2 := fit_idx hd (f := s0) (X := X) (h := h) (Y := 1) (c := 0) hs hX hh (by omega) (by omega)
  have f3 := fit_idx hd (f := s0) (X := X) (h := h) (Y := Y) (c := 0) hs hX hh hY hY0
  simp only [mul_one, add_zero] at f1 f2 f3
  have e1 : (bos * (IntV.ofNat X : I64)).toInt = ((s0 * X : Nat) : Int) :=
    IntV.mul_nat hbos (IntV.I64.toInt_ofNat (by omega)) (by norm_num; omega)
  have e2 : (bos * (IntV.ofNat X : I64) + i_h).toInt = ((s0 * X + h : Nat) : Int) :=
    IntV.add_nat e1 hih (by norm_num; omega)
  exact IntV.mul_nat e2 (IntV.I64.toInt_ofNat (by omega)) (lt_of_lt_of_le f3 (by norm_num))


/-- A stride `s ≤ HV·(K + V + 64)` is below `2^62`. -/
theorem stride_lt {d : Inst} (hd : d.ok) {s : Nat} (hs : s ≤ d.HV * (d.K + d.V + 64)) :
    s < 2 ^ 62 := by
  have h := hd.size_lt
  have : d.HV * (d.K + d.V + 64) ≤ (nTok d + 64) * d.HV * (d.K + d.V + 64) := by
    rw [Nat.mul_assoc]; exact Nat.le_mul_of_pos_left _ (by omega)
  omega

end Claims.RecomputeWU
