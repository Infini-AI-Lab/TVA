import Mathlib

/-!
# The grouped order of the program ids, on bare numbers

Triton's grouped ordering of a `cm × cn` grid of tiles (groups of `8` row tiles, lines 354-361 of
the source): the program `q ∈ [0, cm·cn)` is tile `(gM q, gN q)`, and the map is a bijection
onto `[0, cm) × [0, cn)`, with inverse `gEnc`.
-/

namespace Claims.LoraShrink

/-- `first_pid_m`: eight times the group of `q`. -/
def gFpm (cn q : Nat) : Nat := q / (8 * cn) * 8

/-- `group_size_m`: the group's row tiles, at most `8`. -/
def gGsz (cm cn q : Nat) : Nat := min (cm - gFpm cn q) 8

/-- The row tile of program `q`. -/
def gM (cm cn q : Nat) : Nat := gFpm cn q + (q % (8 * cn)) % gGsz cm cn q

/-- The rank tile of program `q`. -/
def gN (cm cn q : Nat) : Nat := (q % (8 * cn)) / gGsz cm cn q

/-- The program of tile `(a, b)` (the inverse of `(gM, gN)`). -/
def gEnc (cm cn a b : Nat) : Nat :=
  a / 8 * (8 * cn) + b * min (cm - a / 8 * 8) 8 + a % 8

variable {cm cn q a b : Nat}

/-- On a program of the grid the rank tile count is positive. -/
theorem g_cn_pos (hq : q < cm * cn) : 0 < cn := by
  rcases Nat.eq_zero_or_pos cn with h | h
  · rw [h, Nat.mul_zero] at hq; omega
  · exact h

/-- The first row tile of the group is below the number of row tiles. -/
theorem gFpm_lt (hq : q < cm * cn) : gFpm cn q < cm := by
  have hn : 0 < cn := g_cn_pos hq
  have hdiv : 8 * cn * (q / (8 * cn)) + q % (8 * cn) = q := Nat.div_add_mod q (8 * cn)
  have hkey : 8 * (q / (8 * cn)) * cn ≤ q := by
    calc 8 * (q / (8 * cn)) * cn = 8 * cn * (q / (8 * cn)) := by ring
      _ ≤ q := by omega
  unfold gFpm
  by_contra hcon
  have hcon' : cm ≤ q / (8 * cn) * 8 := not_lt.mp hcon
  have h2 : cm * cn ≤ 8 * (q / (8 * cn)) * cn := by
    have := Nat.mul_le_mul_right cn hcon'
    linarith
  omega

/-- The group has at least one row tile. -/
theorem gGsz_pos (hq : q < cm * cn) : 0 < gGsz cm cn q := by
  have := gFpm_lt hq
  unfold gGsz
  exact Nat.lt_min.mpr ⟨by omega, by norm_num⟩

/-- The group has at most eight row tiles. -/
theorem gGsz_le : gGsz cm cn q ≤ 8 := Nat.min_le_right _ _

/-- The group's rows fit in the row tiles. -/
theorem gFpm_add_gGsz_le (hq : q < cm * cn) : gFpm cn q + gGsz cm cn q ≤ cm := by
  have := gFpm_lt hq
  have h : gGsz cm cn q ≤ cm - gFpm cn q := Nat.min_le_left _ _
  omega

/-- The offset inside the group is below `group_size_m · cn`. -/
theorem g_mod_lt (hq : q < cm * cn) : q % (8 * cn) < gGsz cm cn q * cn := by
  have hn : 0 < cn := g_cn_pos hq
  have hG : 0 < 8 * cn := by omega
  have hdiv : 8 * cn * (q / (8 * cn)) + q % (8 * cn) = q := Nat.div_add_mod q (8 * cn)
  have hmodlt : q % (8 * cn) < 8 * cn := Nat.mod_lt q hG
  rcases Nat.lt_or_ge (cm - gFpm cn q) 8 with hpart | hfull
  · have hgs : gGsz cm cn q = cm - gFpm cn q := Nat.min_eq_left (le_of_lt hpart)
    have hf : gFpm cn q ≤ cm := le_of_lt (gFpm_lt hq)
    have hsum : gFpm cn q + gGsz cm cn q = cm := by rw [hgs]; omega
    have hexp : cm * cn = 8 * cn * (q / (8 * cn)) + gGsz cm cn q * cn := by
      calc cm * cn = (gFpm cn q + gGsz cm cn q) * cn := by rw [hsum]
        _ = _ := by simp only [gFpm]; ring
    omega
  · have hgs : gGsz cm cn q = 8 := Nat.min_eq_right hfull
    rw [hgs]; omega

/-- The row tile is in the grid. -/
theorem gM_lt (hq : q < cm * cn) : gM cm cn q < cm := by
  have hgs := gGsz_pos hq
  have hmod : q % (8 * cn) % gGsz cm cn q < gGsz cm cn q := Nat.mod_lt _ hgs
  have hle := gFpm_add_gGsz_le hq
  unfold gM
  omega

/-- The rank tile is in the grid. -/
theorem gN_lt (hq : q < cm * cn) : gN cm cn q < cn :=
  Nat.div_lt_of_lt_mul (g_mod_lt hq)

/-- The row tile lies in the group of eight row tiles of `gFpm`. -/
theorem gM_lt_fpm_add (hq : q < cm * cn) : gM cm cn q < gFpm cn q + 8 := by
  have hgs := gGsz_pos hq
  have h8 : gGsz cm cn q ≤ 8 := gGsz_le
  have hmod : q % (8 * cn) % gGsz cm cn q < gGsz cm cn q := Nat.mod_lt _ hgs
  unfold gM
  omega

/-- The decode of the encoding of a tile is the tile. -/
theorem g_dec_enc (ha : a < cm) (hb : b < cn) :
    gM cm cn (gEnc cm cn a b) = a ∧ gN cm cn (gEnc cm cn a b) = b := by
  have hn : 0 < cn := by omega
  obtain ⟨gs, hgs⟩ : ∃ gs, gs = min (cm - a / 8 * 8) 8 := ⟨_, rfl⟩
  have ha8 : a % 8 < gs := by
    rw [hgs]
    refine Nat.lt_min.mpr ⟨?_, Nat.mod_lt _ (by norm_num)⟩
    have := Nat.div_add_mod a 8
    omega
  have hgs0 : 0 < gs := by omega
  have hgs8 : gs ≤ 8 := by rw [hgs]; exact Nat.min_le_right _ _
  have hlt : b * gs + a % 8 < 8 * cn := by
    have : b * gs + a % 8 < cn * gs := by nlinarith
    nlinarith
  have hx : gEnc cm cn a b = (b * gs + a % 8) + 8 * cn * (a / 8) := by
    unfold gEnc; rw [← hgs]; ring
  have hq1 : gEnc cm cn a b / (8 * cn) = a / 8 := by
    rw [hx, Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hlt, zero_add]
  have hq2 : gEnc cm cn a b % (8 * cn) = b * gs + a % 8 := by
    rw [hx, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]
  have hF : gFpm cn (gEnc cm cn a b) = a / 8 * 8 := by unfold gFpm; rw [hq1]
  have hG : gGsz cm cn (gEnc cm cn a b) = gs := by unfold gGsz; rw [hF, hgs]
  have hmod : (b * gs + a % 8) % gs = a % 8 := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ha8]
  have hdiv : (b * gs + a % 8) / gs = b := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hgs0, Nat.div_eq_of_lt ha8, zero_add]
  constructor
  · unfold gM; rw [hF, hG, hq2, hmod]
    have := Nat.div_add_mod a 8
    omega
  · unfold gN; rw [hG, hq2, hdiv]

/-- The encoding of a tile is a program of the grid. -/
theorem gEnc_lt (ha : a < cm) (hb : b < cn) : gEnc cm cn a b < cm * cn := by
  have hn : 0 < cn := by omega
  obtain ⟨gs, hgs⟩ : ∃ gs, gs = min (cm - a / 8 * 8) 8 := ⟨_, rfl⟩
  have ha8 : a % 8 < gs := by
    rw [hgs]
    refine Nat.lt_min.mpr ⟨?_, Nat.mod_lt _ (by norm_num)⟩
    have := Nat.div_add_mod a 8
    omega
  have hx : gEnc cm cn a b = (b * gs + a % 8) + 8 * cn * (a / 8) := by
    unfold gEnc; rw [← hgs]; ring
  have hfs : a / 8 * 8 + gs ≤ cm := by
    have : gs ≤ cm - a / 8 * 8 := by rw [hgs]; exact Nat.min_le_left _ _
    have : a / 8 * 8 ≤ a := by have := Nat.div_add_mod a 8; omega
    omega
  have h1 : b * gs + a % 8 < cn * gs := by nlinarith
  have h2 : 8 * cn * (a / 8) + cn * gs = (a / 8 * 8 + gs) * cn := by ring
  have : (a / 8 * 8 + gs) * cn ≤ cm * cn := Nat.mul_le_mul_right cn hfs
  omega

/-- **The decode is one-to-one on the grid**: a program is determined by its tile. -/
theorem g_enc_dec (hq : q < cm * cn) : gEnc cm cn (gM cm cn q) (gN cm cn q) = q := by
  have hgs := gGsz_pos hq
  have hn : 0 < cn := g_cn_pos hq
  have hmodlt : q % (8 * cn) % gGsz cm cn q < gGsz cm cn q := Nat.mod_lt _ hgs
  have h8 : gGsz cm cn q ≤ 8 := gGsz_le
  have ha8 : (gM cm cn q) / 8 = q / (8 * cn) ∧ (gM cm cn q) % 8 = q % (8 * cn) % gGsz cm cn q := by
    unfold gM gFpm
    constructor <;> omega
  have hq0 : 8 * cn * (q / (8 * cn)) + q % (8 * cn) = q := Nat.div_add_mod q (8 * cn)
  have hr0 : gGsz cm cn q * (q % (8 * cn) / gGsz cm cn q) + q % (8 * cn) % gGsz cm cn q =
      q % (8 * cn) := Nat.div_add_mod _ _
  unfold gEnc
  rw [ha8.1, ha8.2]
  have hG : min (cm - q / (8 * cn) * 8) 8 = gGsz cm cn q := rfl
  rw [hG]
  unfold gN
  nlinarith

end Claims.LoraShrink
