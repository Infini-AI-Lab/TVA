import GroupedMatmulTma.Defs

/-!
# `grouped_matmul_tma_f16`: the schedule and the footprints

The kernel numbers the tiles of all groups one after another: tile `t` of group `g` (row
`t / cdiv(N g, 128)`, column `t % cdiv(N g, 128)` of tiles) has the global index `start g + t`.
Program `p` takes the global indices `p, p + NUM_SM, p + 2·NUM_SM, …` (lines 210, 251, 275): the
tiles whose global index is `p` modulo `NUM_SM`.

* `start_inj`: a global index names one tile.
* `tileIdx`, `tileBytes`: the elements of `C g` a tile covers; `tileBytes_disj`: different tiles
  share no byte.
* The footprints: `R` (the tables and the input matrices), `Wt p g` (the tiles of group `g`
  program `p` owns), `W p` (all of them), `Wv p g v` (the tile a `while` iteration writes, by the
  loop's variant `v`).
* The invariant of a program, `ProgInv p y m`: memory `m` agrees with `init` outside the tiles
  program `p` has written (those of global index below `y`), keeps every byte present, and holds
  what `OutSpec` says in each of those tiles; `ProgInv_step` (one tile more), `ProgInv_start`,
  `ProgInv_final`.
-/

namespace Kernels.GroupedMatmulTma

open TVA

/-! ## Groups and tiles -/

/-- `cdiv(M g, 128)`: the rows of tiles of group `g` (line 217). -/
def nM (d : Inst) (g : Nat) : Nat := (d.M g + 127) / 128

/-- `cdiv(N g, 128)`: the columns of tiles of group `g` (line 218). -/
def nN (d : Inst) (g : Nat) : Nat := (d.N g + 127) / 128

theorem tiles_eq (d : Inst) (g : Nat) : tiles d g = nM d g * nN d g := rfl

theorem start_zero (d : Inst) : start d 0 = 0 := by simp [start]

theorem start_succ (d : Inst) (g : Nat) : start d (g + 1) = start d g + tiles d g := by
  simp [start, Finset.sum_range_succ]

theorem start_mono (d : Inst) {g h : Nat} (hgh : g ≤ h) : start d g ≤ start d h := by
  induction h with
  | zero => rw [Nat.le_zero.mp hgh]
  | succ h ih =>
    rcases Nat.lt_or_ge g (h + 1) with hlt | hge
    · rw [start_succ]; have := ih (by omega); omega
    · rw [show g = h + 1 by omega]

/-- The tiles of a group come before the next group's. -/
theorem start_add_le (d : Inst) {g h : Nat} (hgh : g < h) : start d g + tiles d g ≤ start d h := by
  rw [← start_succ]; exact start_mono d hgh

/-- A global index names one tile. -/
theorem start_inj (d : Inst) {g h t t' : Nat} (ht : t < tiles d g) (ht' : t' < tiles d h)
    (he : start d g + t = start d h + t') : g = h ∧ t = t' := by
  rcases lt_trichotomy g h with hlt | rfl | hgt
  · have := start_add_le d hlt; omega
  · exact ⟨rfl, by omega⟩
  · have := start_add_le d hgt; omega

/-- A group's tiles end by the last group's end. -/
theorem start_add_le_total (d : Inst) {g : Nat} (hg : g < d.group_size) :
    start d g + tiles d g ≤ start d d.group_size := start_add_le d hg

/-- Two numbers equal modulo `S` within a window of `S` are equal. -/
theorem mod_window {S y u : Nat} (hm : u % S = y % S) (h1 : y ≤ u) (h2 : u < y + S) : u = y := by
  have h3 : (u - y) % S = 0 := Nat.sub_mod_eq_zero_of_mod_eq hm
  have h4 : u - y < S := by omega
  rw [Nat.mod_eq_of_lt h4] at h3
  omega

/-! ## The tiles of `C g` -/

/-- The row of tiles of tile `t` (`tile_m_idx`, line 255). -/
def tm (d : Inst) (g t : Nat) : Nat := t / nN d g

/-- The column of tiles of tile `t` (`tile_n_idx`, line 256). -/
def tn (d : Inst) (g t : Nat) : Nat := t % nN d g

/-- The elements of `C g`'s storage that tile `t` covers: rows `tm·128 + i < M g` and columns
`tn·128 + j < N g` (`i, j < 128`). -/
def tileIdx (d : Inst) (g t : Nat) : Set Nat :=
  {e | ∃ i j, i < 128 ∧ j < 128 ∧ tm d g t * 128 + i < d.M g ∧ tn d g t * 128 + j < d.N g ∧
    e = (tm d g t * 128 + i) * d.ldc g + (tn d g t * 128 + j)}

/-- The bytes of tile `t` of `C g`. -/
def tileBytes (d : Inst) (g t : Nat) : Set Int := (d.C g).elems (tileIdx d g t)

theorem tileIdx_sub (d : Inst) (g t : Nat) : tileIdx d g t ⊆ cIdx d g := by
  rintro _ ⟨i, j, -, -, hi, hj, rfl⟩
  exact ⟨_, _, hi, hj, rfl⟩

theorem tileBytes_sub (d : Inst) (g t : Nat) : tileBytes d g t ⊆ cEl d g :=
  (d.C g).elems_mono (tileIdx_sub d g t)

/-- Rows of `C` below `M` and columns below `N ≤ ldc` determine the element. -/
theorem c_inj {ld N r1 r2 c1 c2 : Nat} (hN : N ≤ ld) (hc1 : c1 < N) (hc2 : c2 < N)
    (h : r1 * ld + c1 = r2 * ld + c2) : r1 = r2 ∧ c1 = c2 := by
  have hr : r1 = r2 := by
    rcases lt_trichotomy r1 r2 with hlt | heq | hgt
    · exfalso
      have h1 : (r1 + 1) * ld ≤ r2 * ld := Nat.mul_le_mul_right _ hlt
      nlinarith
    · exact heq
    · exfalso
      have h1 : (r2 + 1) * ld ≤ r1 * ld := Nat.mul_le_mul_right _ hgt
      nlinarith
  subst hr
  exact ⟨rfl, by omega⟩

/-- A tile's position is its row and column of tiles. -/
theorem tile_eq (d : Inst) (g t : Nat) : t = tm d g t * nN d g + tn d g t := by
  unfold tm tn
  rw [Nat.mul_comm]; exact (Nat.div_add_mod t (nN d g)).symm

/-- Different tiles of one group cover different elements. -/
theorem tileIdx_disj (d : Inst) {g t t' : Nat} (hN : d.N g ≤ d.ldc g) (htt : t ≠ t') :
    Disjoint (tileIdx d g t) (tileIdx d g t') := by
  rw [Set.disjoint_left]
  rintro e ⟨i, j, hi, hj, hiM, hjN, rfl⟩ ⟨i', j', hi', hj', hiM', hjN', he⟩
  obtain ⟨hr, hc⟩ := c_inj hN hjN hjN' he
  have h1 : tm d g t' = tm d g t := by omega
  have h2 : tn d g t' = tn d g t := by omega
  apply htt
  rw [tile_eq d g t, tile_eq d g t', h1, h2]

/-- Different tiles share no byte. -/
theorem tileBytes_disj {d : Inst} (hd : d.ok) {g h t t' : Nat} (hg : g < d.group_size)
    (hh : h < d.group_size) (hne : ¬ (g = h ∧ t = t')) : Disjoint (tileBytes d g t) (tileBytes d h t') := by
  by_cases hgh : g = h
  · subst hgh
    have htt : t ≠ t' := fun e => hne ⟨rfl, e⟩
    exact (d.C g).elems_disjoint (tileIdx_disj d (hd.ldc_ok g hg).2.2 htt)
  · exact (hd.c_c g h hg hh hgh).mono (tileBytes_sub d g t) (tileBytes_sub d h t')

/-! ## The footprints -/

/-- Every program's read footprint: the five tables and every input matrix. -/
def R (d : Inst) : Set Int :=
  tabEl d ∪ {x | ∃ h, h < d.group_size ∧ (x ∈ (d.A h).elems (aIdx d h) ∨ x ∈ (d.B h).elems (bIdx d h))}

/-- The tiles of group `g` that program `p` owns: global index `p` modulo `NUM_SM`. -/
def Wt (d : Inst) (p g : Nat) : Set Int :=
  {x | ∃ t, t < tiles d g ∧ (start d g + t) % d.NUM_SM = p ∧ x ∈ tileBytes d g t}

/-- Program `p`'s write footprint: its tiles of every group. -/
def W (d : Inst) (p : Nat) : Set Int := iterW (Wt d p) d.group_size

/-- The tile the `while` loop of group `g` writes when its variant (the tiles left,
`start g + tiles g − tile_idx`) is `v`, if program `p` owns it. -/
def Wv (d : Inst) (p g v : Nat) : Set Int :=
  if 0 < v ∧ v ≤ tiles d g ∧ (start d g + (tiles d g - v)) % d.NUM_SM = p then tileBytes d g (tiles d g - v)
  else ∅

theorem W_sub (d : Inst) (p : Nat) : W d p ⊆ {x | ∃ g, g < d.group_size ∧ x ∈ cEl d g} := by
  rintro x ⟨g, hg, t, -, -, hx⟩
  exact ⟨g, hg, tileBytes_sub d g t hx⟩

theorem R_W {d : Inst} (hd : d.ok) (p : Nat) : Disjoint (R d) (W d p) := by
  refine Set.disjoint_right.mpr fun x hx hR => ?_
  obtain ⟨g, hg, hxc⟩ := W_sub d p hx
  rcases hR with ht | ⟨h, hh, ha | hb⟩
  · exact Set.disjoint_left.mp (hd.c_tab g hg) hxc ht
  · exact Set.disjoint_left.mp (hd.c_ab g h hg hh).1 hxc ha
  · exact Set.disjoint_left.mp (hd.c_ab g h hg hh).2 hxc hb

theorem Wt_disj {d : Inst} (hd : d.ok) (p : Nat) {g h : Nat} (hgh : g < h) (hh : h < d.group_size) :
    Disjoint (Wt d p g) (Wt d p h) := by
  refine Set.disjoint_left.mpr fun x ⟨t, _, _, hx⟩ ⟨t', _, _, hx'⟩ => ?_
  exact Set.disjoint_left.mp (tileBytes_disj hd (by omega) hh (fun e => by omega)) hx hx'

theorem Wv_disj {d : Inst} (hd : d.ok) (p : Nat) {g : Nat} (hg : g < d.group_size) {u v : Nat}
    (huv : u < v) : Disjoint (Wv d p g u) (Wv d p g v) := by
  unfold Wv
  split_ifs with h1 h2 h2
  · exact tileBytes_disj hd hg hg (fun e => by omega)
  · exact Set.disjoint_empty _
  · exact Set.empty_disjoint _
  · exact Set.disjoint_empty _

theorem Wv_sub (d : Inst) (p g n : Nat) : iterW (Wv d p g) n ⊆ Wt d p g := by
  rintro x ⟨v, -, hx⟩
  unfold Wv at hx
  split_ifs at hx with h
  · exact ⟨tiles d g - v, by omega, h.2.2, hx⟩
  · exact absurd hx (Set.notMem_empty x)

/-- Two programs write different bytes. -/
theorem W_disj {d : Inst} (hd : d.ok) {p q : Nat} (hpq : p ≠ q) : Disjoint (W d p) (W d q) := by
  refine Set.disjoint_left.mpr fun x ⟨g, hg, t, ht, hp, hx⟩ ⟨h, hh, t', ht', hq, hx'⟩ => ?_
  by_cases he : g = h ∧ t = t'
  · obtain ⟨rfl, rfl⟩ := he
    exact hpq (hp.symm.trans hq)
  · exact Set.disjoint_left.mp (tileBytes_disj hd hg hh he) hx hx'

/-! ## The invariant of a program -/

/-- Tile `t` of group `g` is one program `p` owns and has written when `tile_idx` is `y`: its
global index is `p` modulo `NUM_SM` and below `y`. -/
def Done (d : Inst) (p y g t : Nat) : Prop :=
  g < d.group_size ∧ t < tiles d g ∧ (start d g + t) % d.NUM_SM = p ∧ start d g + t < y

/-- The bytes of those tiles. -/
def DoneBytes (d : Inst) (p y : Nat) : Set Int := {x | ∃ g t, Done d p y g t ∧ x ∈ tileBytes d g t}

/-- Memory `m` holds, in each in-bounds element of tile `t` of `C g`, what `OutSpec` says. -/
def TileOK (d : Inst) (init : Mem) (g t : Nat) (m : Mem) : Prop :=
  ∀ i j, i < 128 → j < 128 → tm d g t * 128 + i < d.M g → tn d g t * 128 + j < d.N g →
    OutSpec d init m g (tm d g t * 128 + i) (tn d g t * 128 + j)

/-- Program `p`'s memory when `tile_idx` is `y`: as `init` outside the tiles it has written, every
byte of `init` still present, and those tiles as `OutSpec` says. -/
def ProgInv (d : Inst) (init : Mem) (p y : Nat) (m : Mem) : Prop :=
  (∀ x, x ∉ DoneBytes d p y → m x = init x) ∧ (∀ x, init x ≠ none → m x ≠ none) ∧
    ∀ g t, Done d p y g t → TileOK d init g t m

theorem TileOK.of_agree {d : Inst} {init m m' : Mem} {g t : Nat} (h : TileOK d init g t m)
    (hag : ∀ x ∈ tileBytes d g t, m' x = m x) : TileOK d init g t m' := by
  intro i j hi hj hiM hjN
  obtain ⟨c, hc, hrest⟩ := h i j hi hj hiM hjN
  refine ⟨c, ?_, hrest⟩
  rw [← hc]
  refine Mem.read_congr fun x hx => hag x ⟨_, ⟨i, j, hi, hj, hiM, hjN, rfl⟩, ?_⟩
  exact_mod_cast hx

/-- No tile is written before program `p < NUM_SM` starts (`tile_idx = p`). -/
theorem ProgInv_start (d : Inst) (init : Mem) {p : Nat} (hp : p < d.NUM_SM) : ProgInv d init p p init := by
  have hnone : ∀ g t, ¬ Done d p p g t := fun g t ⟨_, _, hm, hlt⟩ => by
    rw [Nat.mod_eq_of_lt (by omega)] at hm; omega
  exact ⟨fun _ _ => rfl, fun _ h => h, fun g t h => absurd h (hnone g t)⟩

/-- **One tile more.** Program `p` at `tile_idx = y`, the global index of tile `t` of group `g`
(which it owns), stores that tile: the memory then holds `ProgInv` at `y + NUM_SM`. -/
theorem ProgInv_step {d : Inst} (hd : d.ok) {init m m' : Mem} {p y g t : Nat} (hinv : ProgInv d init p y m)
    (hg : g < d.group_size) (ht : t < tiles d g) (hy : start d g + t = y) (hp : y % d.NUM_SM = p)
    (hframe : ∀ x, x ∉ tileBytes d g t → m' x = m x) (hpres : ∀ x, m x ≠ none → m' x ≠ none)
    (hok : TileOK d init g t m') : ProgInv d init p (y + d.NUM_SM) m' := by
  have hdone : Done d p (y + d.NUM_SM) g t := ⟨hg, ht, by rw [hy]; exact hp, by have := hd.SM_pos; omega⟩
  refine ⟨fun x hx => ?_, fun x hx => hpres x (hinv.2.1 x hx), fun h t' hdn => ?_⟩
  · have h1 : x ∉ tileBytes d g t := fun h => hx ⟨g, t, hdone, h⟩
    have h2 : x ∉ DoneBytes d p y := by
      rintro ⟨h', t'', ⟨a, b, c, e⟩, hx'⟩
      exact hx ⟨h', t'', ⟨a, b, c, by omega⟩, hx'⟩
    rw [hframe x h1]; exact hinv.1 x h2
  · obtain ⟨hh, ht', hm, hlt⟩ := hdn
    by_cases hb : start d h + t' < y
    · refine (hinv.2.2 h t' ⟨hh, ht', hm, hb⟩).of_agree fun x hx => hframe x fun hx' => ?_
      exact Set.disjoint_left.mp (tileBytes_disj hd hh hg (fun e => by obtain ⟨rfl, rfl⟩ := e; omega)) hx hx'
    · have he : start d h + t' = y := mod_window (by rw [hm, hp]) (by omega) hlt
      obtain ⟨rfl, rfl⟩ := start_inj d ht' ht (he.trans hy.symm)
      exact hok

/-- At the end (`tile_idx` past every group), every tile program `p` owns is written. -/
theorem ProgInv_final {d : Inst} {init m : Mem} {p y : Nat} (hinv : ProgInv d init p y m)
    (hy : start d d.group_size ≤ y) {g t : Nat} (hg : g < d.group_size) (ht : t < tiles d g)
    (hp : (start d g + t) % d.NUM_SM = p) : TileOK d init g t m :=
  hinv.2.2 g t ⟨hg, ht, hp, by have := start_add_le_total d hg; omega⟩

/-- The tiles a program has written are outputs: the program's memory agrees with `init` on the
read footprint. -/
theorem ProgInv_R {d : Inst} (hd : d.ok) {init m : Mem} {p y : Nat} (hinv : ProgInv d init p y m) :
    ∀ x ∈ R d, m x = init x := by
  intro x hx
  refine hinv.1 x fun ⟨g, t, ⟨hg, _, _, _⟩, hxt⟩ => ?_
  have hxc := tileBytes_sub d g t hxt
  rcases hx with ht | ⟨h, hh, ha | hb⟩
  · exact Set.disjoint_left.mp (hd.c_tab g hg) hxc ht
  · exact Set.disjoint_left.mp (hd.c_ab g h hg hh).1 hxc ha
  · exact Set.disjoint_left.mp (hd.c_ab g h hg hh).2 hxc hb

/-- A buffer valid in `init` stays valid in the program's memory (every byte stays present). -/
theorem ProgInv_valid {d : Inst} {init m : Mem} {p y : Nat} (hinv : ProgInv d init p y m) {dt : DType}
    {b : Buf dt} (hb : b.Valid init) : b.Valid m :=
  Buf.valid_of_agree hb fun x hx => hinv.2.1 x (hb.2.2.2 x hx)

end Kernels.GroupedMatmulTma
