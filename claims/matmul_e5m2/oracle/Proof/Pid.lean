import Proof.Basic

/-!
# The launch's program-id mapping, in `Nat`

The model's lines up to `pid_n`: the grid is grouped in runs of `GROUP_SIZE_M = 8` rows of
blocks, and program `p` handles block `(pid_m, pid_n)` with
`pid_m = 8·g + r % size`, `pid_n = r / size`, `g = p / (8·tN)`, `r = p % (8·tN)`,
`size = min (tM − 8·g) 8`.  This file gives that map over `Nat`, its range and its injectivity
(the number of programs is exactly `tM·tN`).
-/

namespace Claims.MatmulE5m2

open TVA

/-- The number of blocks along the rows: `cdiv(M, 128)`. -/
abbrev tM (d : Inst) : Nat := (d.M + 127) / 128

/-- The number of blocks along the columns: `cdiv(N, 256)`. -/
abbrev tN (d : Inst) : Nat := (d.N + 255) / 256

/-- The grid is `tM · tN` tiles. -/
theorem grid_eq_tiles (d : Inst) : grid d = tM d * tN d := by
  rw [grid_eq]

/-- A program exists only if there is a row tile. -/
theorem tM_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < tM d := by
  rcases Nat.eq_zero_or_pos (tM d) with h | h
  · rw [grid_eq_tiles, h, Nat.zero_mul] at hp
    omega
  · exact h

/-- A program exists only if there is a column tile. -/
theorem tN_pos_of_lt {d : Inst} {p : Nat} (hp : p < grid d) : 0 < tN d := by
  rw [grid_eq_tiles] at hp
  rcases Nat.eq_zero_or_pos (tN d) with h | h
  · rw [h, Nat.mul_zero] at hp; omega
  · exact h

/-- The programs of one group of rows: `GROUP_SIZE_M · tN`. -/
abbrev npig (d : Inst) : Nat := 8 * tN d

/-- The group of program `p`. -/
abbrev grp (d : Inst) (p : Nat) : Nat := p / npig d

/-- The first block row of program `p`'s group. -/
abbrev first (d : Inst) (p : Nat) : Nat := 8 * grp d p

/-- The group's height: the rows of blocks that are left, at most `8`. -/
abbrev gsz (d : Inst) (p : Nat) : Nat := min (tM d - first d p) 8

/-- `p`'s offset inside its group. -/
abbrev rmd (d : Inst) (p : Nat) : Nat := p % npig d

/-- The block row of program `p`. -/
abbrev pidm (d : Inst) (p : Nat) : Nat := first d p + rmd d p % gsz d p

/-- The block column of program `p`. -/
abbrev pidn (d : Inst) (p : Nat) : Nat := rmd d p / gsz d p

/-- The number of programs per group is positive. -/
theorem npig_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < npig d := by
  have h := tN_pos_of_lt hp
  unfold npig
  omega

/-- A program's place within its group is below the group's size. -/
theorem rmd_lt {d : Inst} {p : Nat} (hp : p < grid d) : rmd d p < npig d :=
  Nat.mod_lt _ (npig_pos hp)

/-- `p = first p · tN + rmd p`. -/
theorem first_add_rmd {d : Inst} {p : Nat} : first d p * tN d + rmd d p = p := by
  have h := Nat.div_add_mod p (npig d)
  have hf : first d p * tN d = npig d * (p / npig d) := by unfold first grp npig; ring
  rw [hf]
  exact h

/-- A program's group has a first row tile. -/
theorem grp_lt {d : Inst} {p : Nat} (hp : p < grid d) : grp d p < tM d := by
  have hN := tN_pos_of_lt hp
  have hM := tM_pos hp
  have hgrid : p < tM d * npig d := by
    rw [grid_eq_tiles] at hp
    unfold npig
    have : tN d ≤ 8 * tN d := by omega
    calc p < tM d * tN d := hp
      _ ≤ tM d * (8 * tN d) := Nat.mul_le_mul_left _ this
  unfold grp
  rw [Nat.div_lt_iff_lt_mul (npig_pos hp)]
  exact hgrid

/-- The group's first row tile is a row tile. -/
theorem first_lt {d : Inst} {p : Nat} (hp : p < grid d) : first d p < tM d := by
  have hN := tN_pos_of_lt hp
  have hle : first d p * tN d ≤ p := by
    have h := first_add_rmd (d := d) (p := p)
    omega
  have hlt : first d p * tN d < tM d * tN d :=
    lt_of_le_of_lt hle (by rw [← grid_eq_tiles]; exact hp)
  exact Nat.lt_of_mul_lt_mul_right hlt

/-- A group has at least one row tile. -/
theorem gsz_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < gsz d p := by
  have h := first_lt hp
  unfold gsz
  omega

/-- A group has at most `GROUP_SIZE_M = 8` row tiles. -/
theorem gsz_le_eight (d : Inst) (p : Nat) : gsz d p ≤ 8 := by
  unfold gsz; exact Nat.min_le_right _ _

/-- The group ends at the last row tile or after `8` tiles. -/
theorem first_add_gsz {d : Inst} {p : Nat} (hp : p < grid d) :
    first d p + gsz d p = min (tM d) (first d p + 8) := by
  have h := first_lt hp
  unfold gsz
  omega

/-- A program's row tile is a row tile. -/
theorem pidm_lt {d : Inst} {p : Nat} (hp : p < grid d) : pidm d p < tM d := by
  have hsz := gsz_pos hp
  have hlt := Nat.mod_lt (rmd d p) hsz
  have h1 : pidm d p < first d p + gsz d p := by
    unfold pidm; omega
  have h2 := first_add_gsz hp
  omega

/-- The group left in `p`'s group: `r < size·tN`. -/
theorem rmd_lt_gsz {d : Inst} {p : Nat} (hp : p < grid d) : rmd d p < gsz d p * tN d := by
  have hN := tN_pos_of_lt hp
  have hlt := first_lt hp
  have hp' : p < tM d * tN d := by rw [← grid_eq_tiles]; exact hp
  have hle : first d p * tN d ≤ p := by
    have h := first_add_rmd (d := d) (p := p)
    omega
  have hrmdp : rmd d p = p - first d p * tN d := by
    have h := first_add_rmd (d := d) (p := p)
    omega
  by_cases hcase : tM d - first d p ≤ 8
  · have hsz : gsz d p = tM d - first d p := by unfold gsz; omega
    rw [hsz, Nat.sub_mul, hrmdp]
    exact Nat.sub_lt_sub_right hle hp'
  · have hsz : gsz d p = 8 := by unfold gsz; omega
    rw [hsz]
    exact lt_of_lt_of_le (rmd_lt hp) le_rfl

/-- A program's column tile is a column tile. -/
theorem pidn_lt {d : Inst} {p : Nat} (hp : p < grid d) : pidn d p < tN d := by
  have hsz := gsz_pos hp
  have h := rmd_lt_gsz hp
  unfold pidn
  rw [Nat.div_lt_iff_lt_mul hsz, Nat.mul_comm]
  exact h

/-- `pid_m / 8` recovers the group. -/
theorem pidm_div {d : Inst} {p : Nat} (hp : p < grid d) : pidm d p / 8 = grp d p := by
  have hsz := gsz_pos hp
  have hlt : rmd d p % gsz d p < 8 :=
    lt_of_lt_of_le (Nat.mod_lt _ hsz) (gsz_le_eight d p)
  have he : pidm d p = 8 * grp d p + rmd d p % gsz d p := by
    unfold pidm first
    rfl
  rw [he, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hlt, Nat.add_zero]

/-- The launch's program-id map is one-to-one. -/
theorem pid_inj {d : Inst} {p q : Nat} (hp : p < grid d) (hq : q < grid d)
    (hm : pidm d p = pidm d q) (hn : pidn d p = pidn d q) : p = q := by
  have hgp : pidm d p / 8 = grp d p := pidm_div hp
  have hgq : pidm d q / 8 = grp d q := pidm_div hq
  have hg : grp d p = grp d q := by rw [← hgp, ← hgq, hm]
  have hmod : rmd d p % gsz d p = rmd d q % gsz d q := by
    have h1 : pidm d p = first d p + rmd d p % gsz d p := rfl
    have h2 : pidm d q = first d q + rmd d q % gsz d q := rfl
    rw [h1, h2] at hm
    unfold first at hm
    rw [hg] at hm
    omega
  have hsz : gsz d p = gsz d q := by unfold gsz first; rw [hg]
  have hdiv : rmd d p / gsz d p = rmd d q / gsz d q := hn
  have hp' : rmd d p = rmd d p / gsz d p * gsz d p + rmd d p % gsz d p := by
    rw [Nat.mul_comm (rmd d p / gsz d p) (gsz d p)]
    exact (Nat.div_add_mod (rmd d p) (gsz d p)).symm
  have hq' : rmd d q = rmd d q / gsz d q * gsz d q + rmd d q % gsz d q := by
    rw [Nat.mul_comm (rmd d q / gsz d q) (gsz d q)]
    exact (Nat.div_add_mod (rmd d q) (gsz d q)).symm
  have hrmd : rmd d p = rmd d q := by
    rw [hsz] at hp' hmod hdiv
    rw [hp', hq', hdiv, hmod]
  have hp2 : p = grp d p * npig d + rmd d p := by
    have h := Nat.div_add_mod p (npig d)
    unfold grp rmd
    rw [Nat.mul_comm (p / npig d) (npig d)]
    exact h.symm
  have hq2 : q = grp d q * npig d + rmd d q := by
    have h := Nat.div_add_mod q (npig d)
    unfold grp rmd
    rw [Nat.mul_comm (q / npig d) (npig d)]
    exact h.symm
  calc p = grp d p * npig d + rmd d p := hp2
    _ = grp d q * npig d + rmd d q := by rw [hg, hrmd]
    _ = q := hq2.symm

/-- Every block of the grid is some program's. -/
theorem pid_surj {d : Inst} {i j : Nat} (hi : i < tM d) (hj : j < tN d) :
    ∃ p, p < grid d ∧ pidm d p = i ∧ pidn d p = j := by
  set sz := min (tM d - 8 * (i / 8)) 8 with hszdef
  set p := (i / 8) * npig d + j * sz + i % 8 with hpdef
  have hs8 : i % 8 < 8 := Nat.mod_lt _ (by norm_num)
  have hlast : i % 8 < tM d - 8 * (i / 8) := by omega
  have hszp : 0 < sz := by
    rw [hszdef]; exact lt_min (by omega) (by norm_num)
  have hlastsz : i % 8 < sz := by
    rw [hszdef]; exact lt_min hlast hs8
  have hszle : sz ≤ 8 := by rw [hszdef]; exact Nat.min_le_right _ _
  have hnpig : 0 < npig d := by unfold npig; omega
  have hoff : j * sz + i % 8 < npig d := by
    have h1 : j * sz + i % 8 < tN d * sz := by
      calc j * sz + i % 8 < j * sz + sz := Nat.add_lt_add_left hlastsz _
        _ = (j + 1) * sz := by ring
        _ ≤ tN d * sz := Nat.mul_le_mul_right _ hj
    have h2 : tN d * sz ≤ npig d := by
      unfold npig
      rw [Nat.mul_comm (8 : Nat) (tN d)]
      exact Nat.mul_le_mul_left _ hszle
    exact lt_of_lt_of_le h1 h2
  have hgrp : grp d p = i / 8 := by
    unfold grp
    rw [hpdef, Nat.add_assoc, Nat.mul_comm (i / 8) (npig d), Nat.mul_add_div hnpig,
      Nat.div_eq_of_lt hoff, Nat.add_zero]
  have hfirst : first d p = 8 * (i / 8) := by
    unfold first; rw [hgrp]
  have hgsz : gsz d p = sz := by
    unfold gsz; rw [hfirst, hszdef]
  have hrmd : rmd d p = j * sz + i % 8 := by
    unfold rmd
    rw [hpdef, Nat.add_assoc, Nat.mul_comm (i / 8) (npig d),
      Nat.mul_add_mod (npig d) (i / 8) (j * sz + i % 8), Nat.mod_eq_of_lt hoff]
  have hmod : (j * sz + i % 8) % sz = i % 8 := by
    rw [Nat.add_comm (j * sz) (i % 8), Nat.mul_comm j sz,
      Nat.add_mul_mod_self_left (i % 8) sz j, Nat.mod_eq_of_lt hlastsz]
  have hdiv : (j * sz + i % 8) / sz = j := by
    rw [Nat.mul_comm j sz, Nat.mul_add_div hszp, Nat.div_eq_of_lt hlastsz, Nat.add_zero]
  refine ⟨p, ?_, ?_, ?_⟩
  · rw [grid_eq_tiles, hpdef]
    have h1 : (i / 8) * npig d = 8 * (i / 8) * tN d := by unfold npig; ring
    have h2 : j * sz + i % 8 < tN d * sz := by
      calc j * sz + i % 8 < j * sz + sz := Nat.add_lt_add_left hlastsz _
        _ = (j + 1) * sz := by ring
        _ ≤ tN d * sz := Nat.mul_le_mul_right _ hj
    have h3 : 8 * (i / 8) + sz = min (tM d) (8 * (i / 8) + 8) := by
      rw [hszdef]
      have : 8 * (i / 8) < tM d := by omega
      omega
    calc (i / 8) * npig d + j * sz + i % 8
        = 8 * (i / 8) * tN d + (j * sz + i % 8) := by rw [h1]; ring
      _ < 8 * (i / 8) * tN d + tN d * sz := Nat.add_lt_add_left h2 _
      _ = (8 * (i / 8) + sz) * tN d := by ring
      _ = min (tM d) (8 * (i / 8) + 8) * tN d := by rw [h3]
      _ ≤ tM d * tN d := Nat.mul_le_mul_right _ (Nat.min_le_left _ _)
  · unfold pidm
    rw [hfirst, hgsz, hrmd, hmod]
    omega
  · unfold pidn
    rw [hgsz, hrmd, hdiv]

end Claims.MatmulE5m2
