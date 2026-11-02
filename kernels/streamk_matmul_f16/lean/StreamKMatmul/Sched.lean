import StreamKMatmul.Defs

/-!
# `streamk_matmul_f16`: the stream-K schedule, on natural numbers

With `T = steps K` iterations per tile and `I = iters_per_program`, program `p` owns the global
iterations `[p·I, min(p·I + I, total))` (`pstart`, `pend`), and tile `t` is the iterations
`[t·T, t·T + T)`. The program's part of tile `t` is `[tlo, thi)`, its intersection with the tile;
the program touches the tile when that part is not empty (`touches`), and owns all of it when it
holds the whole tile (`owns`). A tile some program of the grid owns all of is `whole`; the others
are split. In tile-local iterations, the part is `[kfirst, klast)`.

* `iter_*`: the while loop of lines 41–64 from an iteration value `n` computes these: its tile is
  `n / T`, `k_first = n % T = kfirst`, `k_last = klast`, the next value is `thi`, and the branch of
  line 60 is the store exactly when the program owns the tile, the atomic add exactly when the tile
  is split;
* `sum_parts`: the programs' parts of a tile partition its iterations;
* the tile's lanes (`cG`, `cEn`) and the footprints (`W`, `R`, `cells`, `Wv`).
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## The schedule -/

/-- The number of tiles, `num_tiles` (line 79). -/
def ntiles (d : Inst) : Nat := tilesM d * tilesN d

/-- The first global iteration of program `p` (`it = pid * iters_per_program`, line 39). -/
def pstart (d : Inst) (p : Nat) : Nat := p * d.iters_per_program

/-- The end of program `p`'s iterations (`end`, line 40). -/
def pend (d : Inst) (p : Nat) : Nat := min (p * d.iters_per_program + d.iters_per_program) d.total_iters

/-- The start of program `p`'s part of tile `t`. -/
def tlo (d : Inst) (p t : Nat) : Nat := max (pstart d p) (t * steps d.K)

/-- The end of program `p`'s part of tile `t`. -/
def thi (d : Inst) (p t : Nat) : Nat := min (pend d p) (t * steps d.K + steps d.K)

/-- Program `p` has some iteration of tile `t`. -/
abbrev touches (d : Inst) (p t : Nat) : Prop := tlo d p t < thi d p t

/-- Program `p`'s first iteration of tile `t`, in the tile (`k_first`). -/
def kfirst (d : Inst) (p t : Nat) : Nat := tlo d p t - t * steps d.K

/-- The end of program `p`'s iterations of tile `t`, in the tile (`k_last`). -/
def klast (d : Inst) (p t : Nat) : Nat := thi d p t - t * steps d.K

/-- Program `p` has every iteration of tile `t`. -/
def owns (d : Inst) (p t : Nat) : Prop :=
  pstart d p ≤ t * steps d.K ∧ t * steps d.K + steps d.K ≤ pend d p

/-- Some program of the grid has every iteration of tile `t`: the tile is stored, not split. -/
def whole (d : Inst) (t : Nat) : Prop := ∃ p, p < d.num_programs ∧ owns d p t

theorem steps_pos {d : Inst} (hd : d.ok) : 0 < steps d.K := by
  have := hd.K_pos; unfold steps; omega

theorem ipp_pos {d : Inst} (hd : d.ok) : 0 < d.iters_per_program := by
  rcases Nat.eq_zero_or_pos d.iters_per_program with h | h
  · exfalso
    have h1 := hd.cover
    have h2 := hd.total
    rw [h, Nat.mul_zero] at h1
    have : 0 < tilesM d := by unfold tilesM; have := hd.M_pos; omega
    have : 0 < tilesN d := by unfold tilesN; have := hd.N_pos; omega
    have : 0 < steps d.K := steps_pos hd
    have : 0 < tilesM d * tilesN d * steps d.K := by positivity
    omega
  · exact h

theorem div_eq_iff' {x k y : Nat} (hk : 0 < k) : x / k = y ↔ y * k ≤ x ∧ x < y * k + k := by
  rw [Nat.div_eq_iff hk]; omega

/-- A tile's iterations end by `total_iters`. -/
theorem tile_end {d : Inst} (hd : d.ok) {t : Nat} (ht : t < ntiles d) :
    t * steps d.K + steps d.K ≤ d.total_iters := by
  rw [hd.total]
  have : (t + 1) * steps d.K ≤ ntiles d * steps d.K := Nat.mul_le_mul_right _ ht
  unfold ntiles at this
  linarith [Nat.succ_mul t (steps d.K)]

/-- At most one program has every iteration of a tile. -/
theorem owns_unique {d : Inst} (hd : d.ok) {p q t : Nat} (hp : owns d p t) (hq : owns d q t) : p = q := by
  have hI := ipp_pos hd
  have hT := steps_pos hd
  unfold owns pstart pend at hp hq
  have h1 : t * steps d.K / d.iters_per_program = p := (div_eq_iff' hI).mpr ⟨hp.1, by omega⟩
  have h2 : t * steps d.K / d.iters_per_program = q := (div_eq_iff' hI).mpr ⟨hq.1, by omega⟩
  omega

/-- A tile a program owns is a tile of the grid. -/
theorem owns_lt {d : Inst} (hd : d.ok) {p t : Nat} (h : owns d p t) : t < ntiles d := by
  have hT := steps_pos hd
  by_contra hc
  have h1 : ntiles d * steps d.K ≤ t * steps d.K := Nat.mul_le_mul_right _ (by omega)
  have h2 := hd.total
  unfold owns pend at h
  unfold ntiles at h1
  omega

/-! ## One iteration of the while loop -/

section Iter

variable {d : Inst} {p n : Nat}

/-- The iteration from `n` starts its tile's part: `tlo = n`. -/
theorem iter_lo (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0) :
    tlo d p (n / steps d.K) = n := by
  have hT := steps_pos hd
  have hle := Nat.div_mul_le_self n (steps d.K)
  unfold tlo
  rcases h2 with h | h
  · rw [max_eq_left (by omega)]; exact h.symm
  · have : n / steps d.K * steps d.K = n := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)
    rw [max_eq_right (by omega), this]

theorem iter_kfirst (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0) :
    kfirst d p (n / steps d.K) = n % steps d.K := by
  unfold kfirst
  rw [iter_lo hd h1 h2]
  have := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at this
  omega

theorem iter_klast (hd : d.ok) (h3 : n < pend d p) :
    klast d p (n / steps d.K) = min (steps d.K) (n % steps d.K + (pend d p - n)) := by
  have := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at this
  have hT := steps_pos hd
  have := Nat.mod_lt n hT
  unfold klast thi
  omega

theorem iter_touches (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) : touches d p (n / steps d.K) := by
  have hT := steps_pos hd
  have := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at this
  have := Nat.mod_lt n hT
  unfold touches
  rw [iter_lo hd h1 h2]
  unfold thi
  omega

/-- The next value of `it` (line 64) is the end of the part. -/
theorem iter_next (hd : d.ok) (h3 : n < pend d p) :
    n + (min (steps d.K) (n % steps d.K + (pend d p - n)) - n % steps d.K) = thi d p (n / steps d.K) := by
  have hT := steps_pos hd
  have := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at this
  have := Nat.mod_lt n hT
  unfold thi
  omega

/-- The store branch of line 60 is taken exactly when the program owns the tile. -/
theorem iter_owns (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) :
    (n % steps d.K = 0 ∧ min (steps d.K) (n % steps d.K + (pend d p - n)) = steps d.K) ↔
      owns d p (n / steps d.K) := by
  have hT := steps_pos hd
  have hdm := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at hdm
  have := Nat.mod_lt n hT
  have hlo := iter_lo hd h1 h2
  unfold tlo at hlo
  unfold owns
  constructor
  · rintro ⟨ha, hb⟩
    constructor <;> omega
  · rintro ⟨ha, hb⟩
    have : max (pstart d p) (n / steps d.K * steps d.K) = n / steps d.K * steps d.K := max_eq_right ha
    constructor <;> omega

/-- The atomic branch of line 60 is taken only on a split tile. -/
theorem iter_split (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p)
    (hc : ¬ (n % steps d.K = 0 ∧ min (steps d.K) (n % steps d.K + (pend d p - n)) = steps d.K)) :
    ¬ whole d (n / steps d.K) := by
  rintro ⟨q, -, hq⟩
  have hI := ipp_pos hd
  have hT := steps_pos hd
  have hdm := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at hdm
  have := Nat.mod_lt n hT
  have hpq : p = q := by
    have hq' := hq
    unfold owns pstart pend at hq'
    unfold pstart at h1
    unfold pend at h3
    have e1 : n / d.iters_per_program = p := (div_eq_iff' hI).mpr ⟨h1, by omega⟩
    have e2 : n / d.iters_per_program = q := (div_eq_iff' hI).mpr ⟨by omega, by omega⟩
    omega
  subst hpq
  exact hc ((iter_owns hd h1 h2 h3).mpr hq)

/-- The parts done after the iteration: those done before, and this tile's. -/
theorem iter_done (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) (t : Nat) :
    (touches d p t ∧ thi d p t ≤ thi d p (n / steps d.K)) ↔
      (t = n / steps d.K ∨ (touches d p t ∧ thi d p t ≤ n)) := by
  have hT := steps_pos hd
  have hlo := iter_lo hd h1 h2
  have htc := iter_touches hd h1 h2 h3
  have hle := Nat.div_mul_le_self n (steps d.K)
  unfold touches at htc ⊢
  constructor
  · rintro ⟨ht, hle'⟩
    by_cases he : t = n / steps d.K
    · exact Or.inl he
    · right
      refine ⟨ht, ?_⟩
      rcases Nat.lt_or_gt_of_ne he with hlt | hgt
      · have : t * steps d.K + steps d.K ≤ n / steps d.K * steps d.K := by
          have := Nat.mul_le_mul_right (steps d.K) (show t + 1 ≤ n / steps d.K by omega)
          linarith [Nat.succ_mul t (steps d.K)]
        unfold thi at ht ⊢
        omega
      · exfalso
        have : n / steps d.K * steps d.K + steps d.K ≤ t * steps d.K := by
          have := Nat.mul_le_mul_right (steps d.K) (show n / steps d.K + 1 ≤ t by omega)
          linarith [Nat.succ_mul (n / steps d.K) (steps d.K)]
        simp only [tlo, thi] at ht hle' htc hlo
        omega
  · rintro (rfl | ⟨ht, hle'⟩)
    · exact ⟨htc, le_rfl⟩
    · exact ⟨ht, by omega⟩

/-- This iteration's tile was not done before it. -/
theorem iter_not_done (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) : ¬ thi d p (n / steps d.K) ≤ n := by
  have := iter_touches hd h1 h2 h3
  unfold touches at this
  rw [iter_lo hd h1 h2] at this
  omega

end Iter

/-- No part is done before the first iteration. -/
theorem not_done_start {d : Inst} {p t : Nat} (h : touches d p t) : ¬ thi d p t ≤ pstart d p := by
  unfold touches tlo at h
  omega

/-- Every part is done once `it` reaches `end`. -/
theorem done_end {d : Inst} {p t n : Nat} (hn : pend d p ≤ n) : thi d p t ≤ n := by
  unfold thi; omega

/-! ## The programs' parts of a tile -/

/-- Program `p`'s part of tile `t`, in the tile, is the tile's iterations whose global index
`p` owns. -/
theorem part_eq {d : Inst} (hd : d.ok) {t : Nat} (ht : t < ntiles d) (p : Nat) :
    Finset.Ico (kfirst d p t) (klast d p t) =
      (Finset.range (steps d.K)).filter (fun u => (t * steps d.K + u) / d.iters_per_program = p) := by
  have hI := ipp_pos hd
  have hend := tile_end hd ht
  ext u
  simp only [Finset.mem_Ico, Finset.mem_filter, Finset.mem_range, div_eq_iff' hI]
  unfold kfirst klast tlo thi pstart pend
  omega

/-- **The programs' parts partition the tile**: summed over the programs of the grid, any function
of a tile-local iteration gives its sum over the tile. -/
theorem sum_parts {d : Inst} (hd : d.ok) {t : Nat} (ht : t < ntiles d) {β : Type} [AddCommMonoid β]
    (F : Nat → β) :
    ∑ p ∈ Finset.range d.num_programs, ∑ u ∈ Finset.Ico (kfirst d p t) (klast d p t), F u =
      ∑ u ∈ Finset.range (steps d.K), F u := by
  have hI := ipp_pos hd
  have hend := tile_end hd ht
  simp_rw [part_eq hd ht]
  refine Finset.sum_fiberwise_of_maps_to (fun u hu => ?_) F
  rw [Finset.mem_range] at hu ⊢
  have h1 := hd.cover
  rw [Nat.div_lt_iff_lt_mul hI]
  omega

/-- The parts' lengths add up to `T`. -/
theorem sum_part_len {d : Inst} (hd : d.ok) {t : Nat} (ht : t < ntiles d) :
    ∑ p ∈ Finset.range d.num_programs, (klast d p t - kfirst d p t) = steps d.K := by
  have := sum_parts hd ht (fun _ => (1 : Nat))
  simpa using this

/-- At most `T` programs touch a tile. -/
theorem card_touch_le {d : Inst} (hd : d.ok) {t : Nat} (ht : t < ntiles d) :
    ∑ p ∈ Finset.range d.num_programs, (if touches d p t then 1 else 0) ≤ steps d.K := by
  rw [← sum_part_len hd ht]
  refine Finset.sum_le_sum fun p _ => ?_
  split_ifs with h
  · unfold touches at h; unfold kfirst klast; unfold tlo thi at h ⊢; omega
  · exact Nat.zero_le _

/-- A program's part has its length in the tile. -/
theorem part_lt {d : Inst} (p t : Nat) : klast d p t ≤ steps d.K := by
  unfold klast thi; omega

/-! ## A tile's lanes -/

/-- The tile row of tile `t` (`pid_m = tile // num_pid_n`, line 46). -/
def tm (d : Inst) (t : Nat) : Nat := t / tilesN d

/-- The tile column of tile `t` (`pid_n = tile % num_pid_n`, line 47). -/
def tn (d : Inst) (t : Nat) : Nat := t % tilesN d

/-- The row of `c` (and of `a`) of lane row `i` of tile `t` (`offs_m`, line 48). -/
def rowOf (d : Inst) (t i : Nat) : Nat := tm d t * 128 + i

/-- The column of `c` (and of `b`) of lane column `j` of tile `t` (`offs_n`, line 49). -/
def colOf (d : Inst) (t j : Nat) : Nat := tn d t * 128 + j

/-- The element of `c` lane `(i, j)` of tile `t` stores or adds to. -/
def cG (d : Inst) (t : Nat) : Idx [128, 128] → Nat :=
  fun x => rowOf d t x.1 * d.stride_cm + colOf d t x.2.1 * d.stride_cn

/-- Whether lane `(i, j)` of tile `t` is enabled (`c_mask`, line 59). -/
def cEn (d : Inst) (t : Nat) : Idx [128, 128] → Bool :=
  fun x => decide (rowOf d t x.1 < d.M) && decide (colOf d t x.2.1 < d.N)

theorem tilesN_pos {d : Inst} (hd : d.ok) : 0 < tilesN d := by
  unfold tilesN; have := hd.N_pos; omega

/-- Rows of `c` below `M` and columns below `N` determine the element. -/
theorem c_inj {scm scn N r1 r2 c1 c2 : Nat} (hcn : 0 < scn) (hrows : N * scn ≤ scm) (hc1 : c1 < N)
    (hc2 : c2 < N) (h : r1 * scm + c1 * scn = r2 * scm + c2 * scn) : r1 = r2 ∧ c1 = c2 := by
  have hr : r1 = r2 := by
    rcases lt_trichotomy r1 r2 with hlt | heq | hgt
    · exfalso
      have h1 : (r1 + 1) * scm ≤ r2 * scm := Nat.mul_le_mul_right _ hlt
      have h2 : c1 * scn < N * scn := Nat.mul_lt_mul_of_pos_right hc1 hcn
      nlinarith
    · exact heq
    · exfalso
      have h1 : (r2 + 1) * scm ≤ r1 * scm := Nat.mul_le_mul_right _ hgt
      have h2 : c2 * scn < N * scn := Nat.mul_lt_mul_of_pos_right hc2 hcn
      nlinarith
  subst hr
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hcn (by omega)⟩

/-- Two enabled lanes, of any tiles, that address one element are the same lane of the same
tile. -/
theorem lane_inj {d : Inst} (hd : d.ok) {t t' : Nat} {x x' : Idx [128, 128]}
    (he : cEn d t x = true) (he' : cEn d t' x' = true) (hg : cG d t x = cG d t' x') :
    t = t' ∧ x = x' := by
  simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he he'
  obtain ⟨hr, hc⟩ := c_inj hd.cn.1 hd.c_rows he.2 he'.2 hg
  unfold rowOf at hr
  unfold colOf at hc
  obtain ⟨i, j, ⟨⟩⟩ := x
  obtain ⟨i', j', ⟨⟩⟩ := x'
  have hi := i.2
  have hi' := i'.2
  have hj := j.2
  have hj' := j'.2
  simp only at hr hc
  have h1 : tm d t = tm d t' := by omega
  have h2 : tn d t = tn d t' := by omega
  have h3 : (i : Nat) = i' := by omega
  have h4 : (j : Nat) = j' := by omega
  refine ⟨?_, ?_⟩
  · unfold tm at h1
    unfold tn at h2
    rw [← Nat.div_add_mod t (tilesN d), ← Nat.div_add_mod t' (tilesN d), h1, h2]
  · rw [Fin.ext h3, Fin.ext h4]

/-- The tile of element `(r, q)` of `c`, and its lane. -/
def tileOf (d : Inst) (r q : Nat) : Nat := r / 128 * tilesN d + q / 128

def laneOf (r q : Nat) : Idx [128, 128] := (⟨r % 128, Nat.mod_lt _ (by norm_num)⟩, ⟨q % 128, Nat.mod_lt _ (by norm_num)⟩, ())

theorem tileOf_lane {d : Inst} (hd : d.ok) {r q : Nat} (hr : r < d.M) (hq : q < d.N) :
    rowOf d (tileOf d r q) (laneOf r q).1 = r ∧ colOf d (tileOf d r q) (laneOf r q).2.1 = q ∧
      tileOf d r q < ntiles d := by
  have hn := tilesN_pos hd
  have hq' : q / 128 < tilesN d := by unfold tilesN; omega
  have htm : tm d (tileOf d r q) = r / 128 := by
    unfold tm tileOf
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hq', Nat.zero_add]
  have htn : tn d (tileOf d r q) = q / 128 := by
    unfold tn tileOf
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hq']
  refine ⟨?_, ?_, ?_⟩
  · unfold rowOf laneOf; simp only; rw [htm]; omega
  · unfold colOf laneOf; simp only; rw [htn]; omega
  · unfold ntiles tileOf
    have : r / 128 < tilesM d := by unfold tilesM; omega
    have : (r / 128 + 1) * tilesN d ≤ tilesM d * tilesN d := Nat.mul_le_mul_right _ this
    nlinarith

theorem tileOf_en {d : Inst} (hd : d.ok) {r q : Nat} (hr : r < d.M) (hq : q < d.N) :
    cEn d (tileOf d r q) (laneOf r q) = true ∧
      cG d (tileOf d r q) (laneOf r q) = r * d.stride_cm + q * d.stride_cn := by
  obtain ⟨h1, h2, -⟩ := tileOf_lane hd hr hq
  refine ⟨?_, ?_⟩
  · simp only [cEn, h1, h2, hr, hq, decide_true, Bool.and_self]
  · simp only [cG, h1, h2]

/-! ## Footprints -/

/-- Every program's read footprint: the matrices `a` and `b`. -/
def R (d : Inst) : Set Int := d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-- The elements of `c` program `p` stores: the tiles it owns. -/
def ownIdx (d : Inst) (p : Nat) : Set Nat := {e | ∃ t, owns d p t ∧ e ∈ Buf.img (cG d t) (cEn d t)}

/-- Program `p`'s private write footprint: the tiles it owns. -/
def W (d : Inst) (p : Nat) : Set Int := d.c.elems (ownIdx d p)

/-- The shared cells: the elements of `c` in split tiles, which the programs that touch them
accumulate into with `tl.atomic_add`. -/
def cells (d : Inst) : Set (DType × Int) :=
  {z | ∃ t e, ¬ whole d t ∧ e ∈ Buf.img (cG d t) (cEn d t) ∧ z = (.f32, (d.c.ptr (e : Int)).addr)}

/-- The private write footprint of the iteration of program `p` whose variant `pend − it` is `v`:
the tile it stores when it owns it, nothing otherwise. -/
def Wv (d : Inst) (p v : Nat) : Set Int :=
  if v ≤ pend d p ∧ (pend d p - v) % steps d.K = 0 ∧ steps d.K ≤ v then
    d.c.elems (Buf.img (cG d ((pend d p - v) / steps d.K)) (cEn d ((pend d p - v) / steps d.K)))
  else ∅

/-- The tiles of `c` an element is in are one. -/
theorem img_disj {d : Inst} (hd : d.ok) {t t' : Nat} (h : t ≠ t') :
    Disjoint (Buf.img (cG d t) (cEn d t)) (Buf.img (cG d t') (cEn d t')) := by
  rw [Set.disjoint_left]
  rintro _ ⟨x, hx, rfl⟩ ⟨x', hx', hg⟩
  exact h (lane_inj hd hx hx' hg.symm).1

theorem Wv_disj {d : Inst} (hd : d.ok) (p : Nat) {u v : Nat} (huv : u < v) :
    Disjoint (Wv d p u) (Wv d p v) := by
  unfold Wv
  split_ifs with hu hv
  · refine d.c.elems_disjoint (img_disj hd ?_)
    intro he
    have hT := steps_pos hd
    have e1 := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hu.2.1)
    have e2 := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hv.2.1)
    rw [he] at e1
    omega
  all_goals simp

theorem Wv_sub {d : Inst} (hd : d.ok) {p v : Nat} (hv : v ≤ pend d p - pstart d p) :
    Wv d p v ⊆ W d p := by
  unfold Wv
  split_ifs with h
  · refine d.c.elems_mono ?_
    intro e he
    refine ⟨_, ?_, he⟩
    have hT := steps_pos hd
    have e1 := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.2.1)
    unfold owns
    omega
  · exact Set.empty_subset _

/-- An element of a tile of the matrix `c` is an element of the matrix. -/
theorem img_sub_cIdx {d : Inst} (t : Nat) : Buf.img (cG d t) (cEn d t) ⊆ cIdx d := by
  rintro _ ⟨x, hx, rfl⟩
  simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at hx
  exact ⟨_, _, hx.1, hx.2, rfl⟩

theorem W_sub {d : Inst} (p : Nat) : W d p ⊆ d.c.elems (cIdx d) := by
  refine d.c.elems_mono ?_
  rintro _ ⟨t, -, he⟩
  exact img_sub_cIdx t he

theorem cells_sub {d : Inst} : cellBytes (cells d) ⊆ d.c.elems (cIdx d) := by
  rintro x ⟨_, ⟨t, e, -, he, rfl⟩, hx⟩
  exact ⟨e, img_sub_cIdx t he, hx⟩

end Kernels.StreamKMatmul
