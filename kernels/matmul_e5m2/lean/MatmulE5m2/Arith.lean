import TVA

/-!
# `matmul_e5m2`: the grouped program order, on natural numbers

Lines 253–260 map program `p` of a grid of `m * n` programs (`m = cdiv(M, 128)` tile rows,
`n = cdiv(N, 256)` tile columns) to the tile `(pmOf m n p, pnOf m n p)`, in groups of `8` tile
rows. On natural numbers that map is a bijection from `[0, m·n)` onto `[0, m) × [0, n)`
(`pm_lt`, `pn_lt`, `pid_inj`, `pid_surj`); `MatmulE5m2.Program` shows that the model's `int32`
arithmetic computes it.
-/

namespace Kernels.MatmulE5m2

/-- The first tile row of program `p`'s group (`first_pid_m`, line 257). -/
def fpmOf (n p : Nat) : Nat := p / (8 * n) * 8

/-- The size of program `p`'s group (`group_size_m`, line 258). -/
def gsOf (m n p : Nat) : Nat := min (m - fpmOf n p) 8

/-- The tile row of program `p` (`pid_m`, line 259). -/
def pmOf (m n p : Nat) : Nat := fpmOf n p + p % (8 * n) % gsOf m n p

/-- The tile column of program `p` (`pid_n`, line 260). -/
def pnOf (m n p : Nat) : Nat := p % (8 * n) / gsOf m n p

theorem n_pos_of {m n p : Nat} (hp : p < m * n) : 0 < n := by
  rcases Nat.eq_zero_or_pos n with h | h
  · subst h; simp at hp
  · exact h

theorem fpm_lt {m n p : Nat} (hp : p < m * n) : fpmOf n p < m := by
  have hn := n_pos_of hp
  unfold fpmOf
  have h1 : p / (8 * n) * (8 * n) ≤ p := Nat.div_mul_le_self p (8 * n)
  have h2 : p / (8 * n) * 8 * n < m * n := by
    calc p / (8 * n) * 8 * n = p / (8 * n) * (8 * n) := by ring
      _ ≤ p := h1
      _ < m * n := hp
  exact Nat.lt_of_mul_lt_mul_right h2

theorem gs_pos {m n p : Nat} (hp : p < m * n) : 0 < gsOf m n p := by
  have := fpm_lt hp
  unfold gsOf; omega

theorem gs_le {m n p : Nat} : gsOf m n p ≤ 8 := by
  unfold gsOf; omega

theorem fpm_add_gs {m n p : Nat} (hp : p < m * n) : fpmOf n p + gsOf m n p ≤ m := by
  have := fpm_lt hp
  unfold gsOf; omega

/-- The rest of `p` in its group is below `group_size_m · n`. -/
theorem rr_lt {m n p : Nat} (hp : p < m * n) : p % (8 * n) < gsOf m n p * n := by
  have hn := n_pos_of hp
  have hf := fpm_lt hp
  have hdm := Nat.div_add_mod p (8 * n)
  have hmod := Nat.mod_lt p (show 0 < 8 * n by omega)
  unfold gsOf
  unfold fpmOf at hf ⊢
  set g := p / (8 * n)
  set r := p % (8 * n)
  by_cases h8 : 8 ≤ m - g * 8
  · rw [min_eq_right h8]; linarith
  · rw [min_eq_left (by omega)]
    obtain ⟨m', hm'⟩ : ∃ m', m = m' + g * 8 := ⟨m - g * 8, by omega⟩
    rw [hm', Nat.add_sub_cancel]
    rw [hm'] at hp
    nlinarith

theorem pm_lt {m n p : Nat} (hp : p < m * n) : pmOf m n p < m := by
  have h1 := Nat.mod_lt (p % (8 * n)) (gs_pos hp)
  have h2 := fpm_add_gs hp
  unfold pmOf; omega

theorem pn_lt {m n p : Nat} (hp : p < m * n) : pnOf m n p < n := by
  have h := rr_lt hp
  have hg := gs_pos hp
  unfold pnOf
  rw [Nat.div_lt_iff_lt_mul hg]
  linarith

/-- The group of program `p` is its tile row divided by `8`. -/
theorem pm_div {m n p : Nat} (hp : p < m * n) : pmOf m n p / 8 = p / (8 * n) := by
  have h1 := Nat.mod_lt (p % (8 * n)) (gs_pos hp)
  have h2 := gs_le (m := m) (n := n) (p := p)
  unfold pmOf
  unfold fpmOf
  omega

/-- **The order is injective**: two programs of the grid with the same tile are the same. -/
theorem pid_inj {m n p q : Nat} (hp : p < m * n) (hq : q < m * n) (hm : pmOf m n p = pmOf m n q)
    (hn : pnOf m n p = pnOf m n q) : p = q := by
  have hg : p / (8 * n) = q / (8 * n) := by rw [← pm_div hp, ← pm_div hq, hm]
  have hf : fpmOf n p = fpmOf n q := by unfold fpmOf; rw [hg]
  have hs : gsOf m n p = gsOf m n q := by unfold gsOf; rw [hf]
  have hrm : p % (8 * n) % gsOf m n p = q % (8 * n) % gsOf m n q := by
    unfold pmOf at hm; omega
  have hrd : p % (8 * n) / gsOf m n p = q % (8 * n) / gsOf m n q := hn
  have hr : p % (8 * n) = q % (8 * n) := by
    rw [← Nat.div_add_mod (p % (8 * n)) (gsOf m n p), ← Nat.div_add_mod (q % (8 * n)) (gsOf m n q),
      hrd, hrm, hs]
  rw [← Nat.div_add_mod p (8 * n), ← Nat.div_add_mod q (8 * n), hg, hr]

/-- **The order reaches every tile**: injective between sets of the same size. -/
theorem pid_surj {m n i j : Nat} (hi : i < m) (hj : j < n) :
    ∃ p, p < m * n ∧ pmOf m n p = i ∧ pnOf m n p = j := by
  let f : Fin (m * n) → Fin m × Fin n := fun p => (⟨pmOf m n p, pm_lt p.2⟩, ⟨pnOf m n p, pn_lt p.2⟩)
  have hinj : Function.Injective f := by
    intro p q h
    simp only [f, Prod.mk.injEq, Fin.mk.injEq] at h
    exact Fin.ext (pid_inj p.2 q.2 h.1 h.2)
  have hbij : Function.Bijective f :=
    (Fintype.bijective_iff_injective_and_card f).mpr ⟨hinj, by simp⟩
  obtain ⟨p, hp⟩ := hbij.2 (⟨i, hi⟩, ⟨j, hj⟩)
  simp only [f, Prod.mk.injEq, Fin.mk.injEq] at hp
  exact ⟨p, p.2, hp.1, hp.2⟩

end Kernels.MatmulE5m2
