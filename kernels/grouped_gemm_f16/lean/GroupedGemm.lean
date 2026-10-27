import GroupedGemm.Tile

/-!
# Worked example: grouped GEMM (tutorial 08)

The model, the instances, the conditions (`Inst.ok`, `Pre`), the specification (`Post`) and the
assumptions are in `TVAExamples.GroupedGemm.Defs`; the `K` loop and one tile are proved in
`…KLoop` and `…Tile`. This file proves the schedule and the claim:

* program `p` visits the global tiles `p, p + NUM_SM, p + 2·NUM_SM, …` below the total, in the
  problems that hold them (`ForInv`, `WhileInv`);
* different programs write different `C` elements, and no program writes an input
  (`W_disjoint`);
* `main`: `Assumptions K P → Spec K`, by `meets_independent`;
* `consistent`: the assumptions hold together (the reference platform rounds `tl.dot` once).
-/
namespace TVAExamples.GroupedGemm

open TVA TVA.Math.Float

/-! ## Footprints -/

/-- Program `p` writes the `C` tiles `t` with `t % NUM_SM = p`. -/
def Wp (d : Inst) (p : Nat) : Set Int := {x | ∃ t, t % d.NUM_SM = p ∧ x ∈ tileSet d t}

/-- The tiles below `T` that program `p` owns are done in `m`. -/
def Done (d : Inst) (init : Mem) (p : Nat) (m : Mem) (T : Nat) : Prop :=
  ∀ g, g < d.G → ∀ i j, i < (d.gemm g).M → j < (d.gemm g).N →
    d.tileOf g i j % d.NUM_SM = p → d.tileOf g i j < T → Good d init m g i j

/-- The memory facts every tile needs, for every problem. -/
def MemOK (d : Inst) (init m : Mem) : Prop :=
  (∀ x ∈ inputs d, m x = init x) ∧ ∀ g, g < d.G → (d.gemm g).c.Valid m

/-- Before problem `t`: `last_problem_end = start t`, and `tile_idx` is program `p`'s first tile
at or after `start t`. -/
def ForInv (d : Inst) (init : Mem) (p t : Nat) (st : I32 × I32) (m : Mem) : Prop :=
  st.2.toInt = (d.start t : ℤ) ∧
  (∃ ti : ℕ, st.1.toInt = ti ∧ ti % d.NUM_SM = p ∧ d.start t ≤ ti ∧ ti < d.start t + d.NUM_SM) ∧
  MemOK d init m ∧ Done d init p m (d.start t)

/-- In the `while` loop of problem `t`: the tiles of `p` below `tile_idx` (and `start (t+1)`)
are done. -/
def WhileInv (d : Inst) (init : Mem) (p t : Nat) (tiv : I32) (m : Mem) : Prop :=
  ∃ ti : ℕ, tiv.toInt = ti ∧ ti % d.NUM_SM = p ∧ d.start t ≤ ti ∧ ti < d.start (t + 1) + d.NUM_SM ∧
    MemOK d init m ∧ Done d init p m (min ti (d.start (t + 1)))

/-! ## Cells and tiles -/

section Cells
variable {d : Inst}

theorem c_idx_lt (hd : d.ok) {g i j : Nat} (hg : g < d.G) (hi : i < (d.gemm g).M) (hj : j < (d.gemm g).N) :
    i * (d.gemm g).ldc + j < (d.gemm g).c.len := by
  obtain ⟨-, -, hcl, -, -, hNc, -⟩ := ok_gemm hd hg
  rw [hcl]
  calc i * (d.gemm g).ldc + j < i * (d.gemm g).ldc + (d.gemm g).ldc := by omega
    _ = (i + 1) * (d.gemm g).ldc := by ring
    _ ≤ (d.gemm g).M * (d.gemm g).ldc := Nat.mul_le_mul_right _ hi

theorem cell_sub (hd : d.ok) {g i j : Nat} (hg : g < d.G) (hi : i < (d.gemm g).M) (hj : j < (d.gemm g).N) :
    Mem.span .f16 ((d.gemm g).cAt i j) ⊆ (d.gemm g).c.bytes :=
  (d.gemm g).c.span_sub (c_idx_lt hd hg hi hj)

/-- Two `C` cells that share a byte are the same cell. -/
theorem cell_unique (hd : d.ok) {g i j g' i' j' : Nat} (hg : g < d.G) (hi : i < (d.gemm g).M)
    (hj : j < (d.gemm g).N) (hg' : g' < d.G) (hi' : i' < (d.gemm g').M) (hj' : j' < (d.gemm g').N) {x : Int}
    (hx : x ∈ Mem.span .f16 ((d.gemm g).cAt i j)) (hx' : x ∈ Mem.span .f16 ((d.gemm g').cAt i' j')) :
    g = g' ∧ i = i' ∧ j = j' := by
  by_cases hgg : g = g'
  · subst hgg
    obtain ⟨-, -, -, -, -, hNc, -⟩ := ok_gemm hd hg
    by_cases he : i * (d.gemm g).ldc + j = i' * (d.gemm g).ldc + j'
    · obtain ⟨e1, e2⟩ := grid_unique (by omega) (by omega) he
      exact ⟨rfl, e1, e2⟩
    · exact absurd hx' (Set.disjoint_left.mp ((d.gemm g).c.span_disjoint (by exact_mod_cast he)) hx)
  · exact absurd (cell_sub hd hg' hi' hj' hx')
      (Set.disjoint_left.mp ((hd.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2 g hg g' hg').2.2.1 hgg) (cell_sub hd hg hi hj hx))

theorem tileSet_cell {t : Nat} {x : Int} (h : x ∈ tileSet d t) :
    ∃ g i j, g < d.G ∧ i < (d.gemm g).M ∧ j < (d.gemm g).N ∧ d.tileOf g i j = t ∧
      x ∈ Mem.span .f16 ((d.gemm g).cAt i j) := h

/-- A cell outside tile `t` shares no byte with it. -/
theorem cell_not_tile (hd : d.ok) {g i j t : Nat} (hg : g < d.G) (hi : i < (d.gemm g).M)
    (hj : j < (d.gemm g).N) (ht : d.tileOf g i j ≠ t) {x : Int}
    (hx : x ∈ Mem.span .f16 ((d.gemm g).cAt i j)) : x ∉ tileSet d t := by
  rintro ⟨g', i', j', hg', hi', hj', ht', hx'⟩
  obtain ⟨rfl, rfl, rfl⟩ := cell_unique hd hg hi hj hg' hi' hj' hx hx'
  exact ht ht'

/-- `C` shares no byte with the inputs. -/
theorem cell_not_input (hd : d.ok) {g i j : Nat} (hg : g < d.G) (hi : i < (d.gemm g).M)
    (hj : j < (d.gemm g).N) {x : Int} (hx : x ∈ Mem.span .f16 ((d.gemm g).cAt i j)) : x ∉ inputs d := by
  have hc := cell_sub hd hg hi hj hx
  have hD := hd.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2 g hg
  rintro (h | h | h | h | h | ⟨g', hg', h | h⟩)
  · exact Set.disjoint_left.mp (hD 0 (by omega)).2.2.2.2.2.2.1 hc h
  · exact Set.disjoint_left.mp (hD 0 (by omega)).2.2.2.2.2.2.2 hc h
  · exact Set.disjoint_left.mp (hD 0 (by omega)).2.2.2.1 hc h
  · exact Set.disjoint_left.mp (hD 0 (by omega)).2.2.2.2.1 hc h
  · exact Set.disjoint_left.mp (hD 0 (by omega)).2.2.2.2.2.1 hc h
  · exact Set.disjoint_left.mp (hD g' hg').1 hc h
  · exact Set.disjoint_left.mp (hD g' hg').2.1 hc h

theorem tileSet_not_input (hd : d.ok) {t : Nat} {x : Int} (h : x ∈ tileSet d t) : x ∉ inputs d := by
  obtain ⟨g, i, j, hg, hi, hj, -, hx⟩ := h
  exact cell_not_input hd hg hi hj hx

/-- The global tile of a cell is below the next problem's start. -/
theorem tileOf_lt (hd : d.ok) {g i j : Nat} (hg : g < d.G) (hi : i < (d.gemm g).M) (hj : j < (d.gemm g).N) :
    d.start g ≤ d.tileOf g i j ∧ d.tileOf g i j < d.start (g + 1) := by
  obtain ⟨-, -, -, -, -, -, hBMd, hBNd, -⟩ := ok_gemm hd hg
  have h1 : i / d.BM < (d.gemm g).M / d.BM := Nat.div_lt_div_of_lt_of_dvd hBMd hi
  have h2 : j / d.BN < (d.gemm g).N / d.BN := Nat.div_lt_div_of_lt_of_dvd hBNd hj
  have h3 : i / d.BM * ((d.gemm g).N / d.BN) + j / d.BN < (d.gemm g).M / d.BM * ((d.gemm g).N / d.BN) :=
    calc i / d.BM * ((d.gemm g).N / d.BN) + j / d.BN
        < i / d.BM * ((d.gemm g).N / d.BN) + (d.gemm g).N / d.BN := by omega
      _ = (i / d.BM + 1) * ((d.gemm g).N / d.BN) := by ring
      _ ≤ (d.gemm g).M / d.BM * ((d.gemm g).N / d.BN) := Nat.mul_le_mul_right _ h1
  unfold Inst.tileOf
  show d.start g ≤ d.start g + i / d.BM * ((d.gemm g).N / d.BN) + j / d.BN ∧
    d.start g + i / d.BM * ((d.gemm g).N / d.BN) + j / d.BN <
      d.start g + (d.gemm g).M / d.BM * ((d.gemm g).N / d.BN)
  rw [Nat.add_assoc]
  exact ⟨Nat.le_add_right _ _, Nat.add_lt_add_left h3 _⟩

/-- `Good` only depends on the cell's bytes. -/
theorem Good.congr {init m m' : Mem} {g i j : Nat} (h : Good d init m g i j)
    (he : ∀ x ∈ Mem.span .f16 ((d.gemm g).cAt i j), m' x = m x) : Good d init m' g i j := by
  obtain ⟨o, c, hc, rest⟩ := h
  exact ⟨o, c, by rw [Mem.read_congr he]; exact hc, rest⟩

end Cells

/-! ## The schedule -/

section Schedule
variable {E : Env} {R W : Set Int} {d : Inst}

theorem start_lt (hd : d.ok) {t : Nat} (ht : t ≤ d.G) : d.start t + d.NUM_SM < 2 ^ 31 :=
  lt_of_le_of_lt (Nat.add_le_add_right (d.start_le ht) _) hd.2.2.2.2.2.2.2.2.2.1

/-- One problem of the group: the `for g` body. -/
theorem safe_problem (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (hdot : Rules.Num.DotTotal E.P cfg) (hacc : Rules.Num.DotAcc E.P cfg (fun _ => True) rel abs)
    {p t : Nat} (ht : t < d.G) (hR : inputs d ⊆ R) (hW : Wp d p ⊆ W) {st : I32 × I32} {m : Mem}
    (hI : ForInv d init p t st m) {gv : I32} (hgv : gv.toInt = (t : ℤ)) :
    Safe E R W (fun st' m' => ForInv d init p (t + 1) st' m') m
      (problem (d.aptrs.ptr 0) (d.bptrs.ptr 0) (d.cptrs.ptr 0) (d.sizes.ptr 0) (d.lds.ptr 0) d.NUM_SM
        d.BM d.BN d.BK gv st) := by
  obtain ⟨tiv, last⟩ := st
  obtain ⟨hlast, ⟨ti, htiv, htip, htlo, hthi⟩, ⟨hagree, hcv⟩, hdone⟩ := hI
  have hok := ok_gemm hd ht
  obtain ⟨-, -, -, -, -, -, hBMd, hBNd, -, hMfit, hNfit, hKfit, -⟩ := hok
  obtain ⟨hM, hN, hK, -⟩ := hpre t ht
  obtain ⟨-, -, -, -, -, -, -, -, h3G, hstartG, -, -, -, hsizes, -, -⟩ := id hd
  have hBM := BM_pos hd
  have hBN := BN_pos hd
  have hNUM : 0 < d.NUM_SM := hd.2.2.2.2.2.2.2.1
  have hS1 := start_lt hd (show t + 1 ≤ d.G by omega)
  have hsizes_in : d.sizes.bytes ⊆ inputs d := fun _ hx => Or.inl hx
  have h3 : (3 : I32).toInt = 3 := IntV.I32.toInt_ofNat (by norm_num)
  have hg3 : (gv * 3).toInt = ((t * 3 : ℕ) : ℤ) := IntV.I32.mul_nat hgv h3 (by omega)
  have e0 : d.sizes.ptr 0 + gv * 3 = d.sizes.ptr ((3 * t : ℕ) : ℤ) := by
    rw [ptr_add, hg3]; congr 1; push_cast; ring
  have hp0 : ∀ q : Nat, q < 3 → d.sizes.ptr 0 + gv * 3 + (IntV.ofNat q : I32) = d.sizes.ptr ((3 * t + q : ℕ) : ℤ) := by
    intro q hq
    rw [ptr_add, ptr_add, hg3, IntV.I32.ofNat_nat (by omega)]; congr 1; push_cast; ring
  unfold problem
  simp only []
  -- `gm`, `gn`, `gk`
  refine safe_load1 (v0 := IntV.ofNat (d.gemm t).M) (by
      rw [e0, read_input hagree (span_sub_of _ hsizes_in (by omega))]; exact hM)
    (by rw [e0]; exact (span_sub_of (i := 3 * t) _ hsizes_in (by omega)).trans hR) ?_
  refine safe_load1 (v0 := IntV.ofNat (d.gemm t).N) (by
      rw [show (1 : I32) = IntV.ofNat 1 from rfl, hp0 1 (by norm_num),
        read_input hagree (span_sub_of _ hsizes_in (by omega))]; exact hN)
    (by rw [show (1 : I32) = IntV.ofNat 1 from rfl, hp0 1 (by norm_num)]
        exact (span_sub_of (i := 3 * t + 1) _ hsizes_in (by omega)).trans hR) ?_
  refine safe_load1 (v0 := IntV.ofNat (d.gemm t).K) (by
      rw [show (2 : I32) = IntV.ofNat 2 from rfl, hp0 2 (by norm_num),
        read_input hagree (span_sub_of _ hsizes_in (by omega))]; exact hK)
    (by rw [show (2 : I32) = IntV.ofNat 2 from rfl, hp0 2 (by norm_num)]
        exact (span_sub_of (i := 3 * t + 2) _ hsizes_in (by omega)).trans hR) ?_
  -- `num_m_tiles`, `num_n_tiles`, `num_tiles`
  refine safe_cdiv (IntV.I32.ofNat_nat (by omega)) hBM (by omega) fun nm hnm => ?_
  refine safe_cdiv (IntV.I32.ofNat_nat (by omega)) hBN (by omega) fun nn hnn => ?_
  rw [cdiv_dvd hBM hBMd] at hnm
  rw [cdiv_dvd hBN hBNd] at hnn
  have hS : d.start (t + 1) = d.start t + d.tiles t := rfl
  have htiles : d.tiles t = (d.gemm t).M / d.BM * ((d.gemm t).N / d.BN) := rfl
  have hnt : (nm * nn).toInt = ((d.tiles t : ℕ) : ℤ) := by
    rw [htiles]; exact IntV.I32.mul_nat hnm hnn (by rw [← htiles]; omega)
  have hend : (last + nm * nn).toInt = ((d.start (t + 1) : ℕ) : ℤ) := by
    rw [hS]; exact IntV.I32.add_nat hlast hnt (by omega)
  -- the `while` loop
  refine Safe.bind (safe_while _ _ tiv (WhileInv d init p t)
    (fun tv => (((d.start (t + 1) + d.NUM_SM : ℕ) : ℤ) - tv.toInt).toNat)
    ⟨ti, htiv, htip, htlo, by omega, ⟨hagree, hcv⟩, fun g hg i j hi hj hp hlt => ?_⟩ fun tv m₁ hJ => ?_)
  · -- the tiles below `min ti (start (t+1))` are those below `start t`
    refine hdone g hg i j hi hj hp ?_
    by_contra hge
    push Not at hge
    have hlt' : d.tileOf g i j < ti := lt_of_lt_of_le hlt (min_le_left _ _)
    -- a tile of `p` in `[start t, ti)` would be at most `ti - NUM_SM < start t`
    have : d.tileOf g i j + d.NUM_SM ≤ ti := by
      have h1 := Nat.mod_add_div (d.tileOf g i j) d.NUM_SM
      have h2 := Nat.mod_add_div ti d.NUM_SM
      rw [hp] at h1; rw [htip] at h2
      have : d.tileOf g i j / d.NUM_SM < ti / d.NUM_SM := by
        by_contra h; push Not at h
        have := Nat.mul_le_mul_left d.NUM_SM h
        omega
      have := Nat.mul_le_mul_left d.NUM_SM (Nat.succ_le_of_lt this)
      rw [Nat.mul_succ] at this
      omega
    omega
  · -- one test of the condition
    obtain ⟨ti', htv, htp', hlo', hhi', hmem, hdone'⟩ := hJ
    refine .ret ?_
    have hcond : (last.le tv && tv.lt (last + nm * nn)) =
        (decide (d.start t ≤ ti') && decide (ti' < d.start (t + 1))) := by
      rw [IntV.le_eq, IntV.lt_eq, hlast, htv, hend]; simp
    rw [hcond]
    by_cases hin : ti' < d.start (t + 1)
    · -- the tile `ti'` of problem `t`
      simp only [hlo', hin, decide_true, Bool.and_self, ite_true]
      refine safe_tile hd hpre hdot hacc ht hlo' (by rw [← hS]; exact hin) hgv
        (IntV.I32.ofNat_nat (by omega)) hnn hlast htv hR
        (fun x hx => hW ⟨ti', htp', hx⟩) ⟨hmem.1, hmem.2 t ht⟩ fun m' hgood hfr hpres => ?_
      have hNUM31 : (IntV.ofNat d.NUM_SM : I32).toInt = (d.NUM_SM : ℤ) := IntV.I32.ofNat_nat (by omega)
      have hnext : (tv + (IntV.ofNat d.NUM_SM : I32)).toInt = ((ti' + d.NUM_SM : ℕ) : ℤ) :=
        IntV.I32.add_nat htv hNUM31 (by omega)
      refine ⟨⟨ti' + d.NUM_SM, hnext, by rw [Nat.add_mod_right]; exact htp', by omega, by omega,
        ⟨fun x hx => ?_, fun g hg => ?_⟩, fun g hg i j hi hj hp hlt => ?_⟩, ?_⟩
      · rw [hfr x fun h => tileSet_not_input hd h hx]; exact hmem.1 x hx
      · exact Buf.valid_of_agree (hmem.2 g hg) fun x hx => hpres x ((hmem.2 g hg).2.2.2 x hx)
      · -- tiles below `ti' + NUM_SM` owned by `p`: below `ti'` (done before) or `ti'` itself
        by_cases heq : d.tileOf g i j = ti'
        · -- the tile just stored: `g = t`
          have hgt : g = t := by
            obtain ⟨h1, h2⟩ := tileOf_lt hd hg hi hj
            rcases lt_trichotomy g t with hlt' | rfl | hgt'
            · have := d.start_le (show g + 1 ≤ t by omega); omega
            · rfl
            · have := d.start_le (show t + 1 ≤ g by omega); omega
          subst hgt
          exact hgood i j hi hj heq
        · have hlt' : d.tileOf g i j < ti' := by
            have hmin : min (ti' + d.NUM_SM) (d.start (t + 1)) ≤ ti' + d.NUM_SM := min_le_left _ _
            -- a tile of `p` in `(ti', ti' + NUM_SM)` does not exist
            have h1 := Nat.mod_add_div (d.tileOf g i j) d.NUM_SM
            have h2 := Nat.mod_add_div ti' d.NUM_SM
            rw [hp] at h1; rw [htp'] at h2
            by_contra hge; push Not at hge
            have hlt2 : d.tileOf g i j < ti' + d.NUM_SM := lt_of_lt_of_le hlt hmin
            have : ti' / d.NUM_SM < d.tileOf g i j / d.NUM_SM := by
              by_contra h; push Not at h
              have := Nat.mul_le_mul_left d.NUM_SM h
              omega
            have := Nat.mul_le_mul_left d.NUM_SM (Nat.succ_le_of_lt this)
            rw [Nat.mul_succ] at this
            omega
          have hgood' := hdone' g hg i j hi hj hp (lt_min hlt' (lt_of_lt_of_le hlt (min_le_right _ _)))
          exact hgood'.congr fun x hx => hfr x (cell_not_tile hd hg hi hj heq hx)
      · -- the variant decreases
        rw [hnext, htv]
        omega
    · -- the loop ends: `ti' ≥ start (t+1)`
      have hfalse : (decide (d.start t ≤ ti') && decide (ti' < d.start (t + 1))) = false := by simp [hin]
      rw [hfalse]
      simp only [Bool.false_eq_true, ite_false]
      refine .ret ⟨hend, ⟨ti', htv, htp', by omega, by omega⟩, hmem, fun g hg i j hi hj hp hlt => ?_⟩
      exact hdone' g hg i j hi hj hp (lt_min_iff.mpr ⟨by omega, hlt⟩)

/-- **Program `p`.** -/
theorem safe_program (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (hdot : Rules.Num.DotTotal E.P cfg) (hacc : Rules.Num.DotAcc E.P cfg (fun _ => True) rel abs)
    {p : Nat} (hp : p < d.NUM_SM) (hvc : ∀ g, g < d.G → (d.gemm g).c.Valid init) :
    Safe E (inputs d) (Wp d p) (fun _ m => Done d init p m (d.start d.G)) init
      ((model d).body ((launch d).ctx p)) := by
  have hpid : ((launch d).ctx p).pid 0 = p := Nat.mod_eq_of_lt hp
  have hG : d.G < 2 ^ 31 := by have := hd.2.2.2.2.2.2.2.2.1; omega
  have h0 : (0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)
  have h1 : (1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by norm_num)
  have hGv : (IntV.ofNat d.G : I32).toInt = (d.G : ℤ) := IntV.I32.ofNat_nat hG
  have hrange : tl.RangeOk (0 : I32) (IntV.ofNat d.G : I32) 1 := by
    refine ⟨by rw [h1]; norm_num, ?_, ?_, ?_⟩ <;>
      simp only [Format.intRange, ite_true, h0, h1, hGv] <;> omega
  have htrip : tl.tripCount (0 : I32).toInt (IntV.ofNat d.G : I32).toInt (1 : I32).toInt = d.G := by
    rw [h0, h1, hGv]; unfold tl.tripCount
    rw [sub_zero, add_sub_cancel_right, Int.ediv_one, Int.toNat_natCast]
  have hstart := start_lt hd (le_refl d.G)
  simp only [model, grouped_matmul_kernel, tl.program_id, hpid]
  refine safe_forRange hrange (ForInv d init p) ⟨?_, ⟨p, ?_, Nat.mod_eq_of_lt hp, Nat.zero_le _, ?_⟩,
    ⟨fun _ _ => rfl, hvc⟩, fun g _ i j _ _ _ hlt => absurd hlt (Nat.not_lt_zero _)⟩
    (fun t ht st m₁ hI => ?_) (fun st m₁ hI => ?_)
  · exact h0
  · exact IntV.I32.ofNat_nat (by omega)
  · simp only [Inst.start]; omega
  · rw [htrip] at ht
    refine safe_problem hd hpre hdot hacc ht subset_rfl subset_rfl hI ?_
    rw [h0, h1, IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega)]; ring
  · rw [htrip] at hI
    exact .ret hI.2.2.2

end Schedule

/-! ## The claim -/

theorem launch_size (d : Inst) : (launch d).size = d.NUM_SM := by
  simp [launch, Launch.size, Launch.grid1]

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hNUM : d.NUM_SM < 2 ^ 31 := by have := hd.2.2.2.2.2.2.2.2.2.1; omega
  have hBM := BM_le hd
  have hBN := BN_le hd
  have hBK := BK_le hd
  obtain ⟨-, ⟨eM, heM⟩, ⟨eN, heN⟩, ⟨eK, heK⟩, hMK, hKN, hMN, -⟩ := hd
  refine ⟨hNUM, by simp [launch, Launch.grid1], by simp [launch, Launch.grid1], rfl, ?_⟩
  have h1 : ∃ e, (1 : Nat) = 2 ^ e := ⟨0, rfl⟩
  intro s hs
  simp only [launch, Launch.grid1, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    refine ⟨fun k hk => ?_, by simp; omega⟩ <;>
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk <;>
    rcases hk with rfl | rfl <;> first | exact ⟨_, heM⟩ | exact ⟨_, heN⟩ | exact ⟨_, heK⟩ | exact h1

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent (model := model) (launch := launch) hA.faithful (fun _ _ => rfl)
    (fun d hd => launch_valid hd) (fun d _ _ => inputs d) (fun d _ i => Wp d i)
    (fun d init i m => Done d init i m (d.start d.G)) ?_ ?_ ?_
  · -- each program, alone
    intro d hd init hpre i
    have hi : (i : Nat) < d.NUM_SM := lt_of_lt_of_eq i.isLt (launch_size d)
    exact safe_program hd hpre hA.dotTotal hA.dotAcc hi fun g hg => (hpre g hg).2.2.2.2.2.2.2.2.2.2.2.1
  · -- footprints: a program's writes miss the other programs' reads and writes
    intro d hd init _ i j hij
    have hne : (i : Nat) ≠ j := Fin.val_ne_of_ne hij
    have hi : (i : Nat) < d.NUM_SM := lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < d.NUM_SM := lt_of_lt_of_eq j.isLt (launch_size d)
    rw [Set.disjoint_left]
    rintro x ⟨t, ht, hx⟩ (hR | ⟨t', ht', hx'⟩)
    · exact tileSet_not_input hd hx hR
    · obtain ⟨g, a, b, hg, ha, hb, htg, hxc⟩ := hx
      obtain ⟨g', a', b', hg', ha', hb', htg', hxc'⟩ := hx'
      obtain ⟨rfl, rfl, rfl⟩ := cell_unique hd hg ha hb hg' ha' hb' hxc hxc'
      rw [← htg, htg'] at ht
      exact hne (ht.symm.trans ht')
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun g hg i j hi hj => ?_, fun x hx => hfr x fun p ⟨t, _, hxt⟩ => ?_⟩
    · have hNUM : 0 < d.NUM_SM := hd.2.2.2.2.2.2.2.1
      set p := d.tileOf g i j % d.NUM_SM
      have hp : p < (launch d).size := by rw [launch_size]; exact Nat.mod_lt _ hNUM
      obtain ⟨m, hQ, hag⟩ := hloc ⟨p, hp⟩
      have hlt := (tileOf_lt hd hg hi hj).2
      have hgood := hQ g hg i j hi hj rfl (lt_of_lt_of_le hlt (d.start_le (show g + 1 ≤ d.G by omega)))
      have hfin : Good d init fin g i j := hgood.congr fun x hx =>
        hag x (Or.inr ⟨d.tileOf g i j, rfl, g, i, j, hg, hi, hj, rfl, hx⟩)
      exact hfin
    · obtain ⟨g, a, b, hg, ha, hb, -, hxc⟩ := hxt
      exact hx g hg (cell_sub hd hg ha hb hxc)

/-- **The closed form of the bound**: `errC K S ≤ 2T·(ρ·S + α)` with `T = K / BK`, under the
condition `T·ρ ≤ 1/2` of `Gemm.ok`. -/
theorem errC_le (d : Inst) {K : Nat} (h : ((K / d.BK : Nat) : ℚ) * ρ d ≤ 1 / 2) {S : ℚ} (hS : 0 ≤ S) :
    errC d K S ≤ 2 * (K / d.BK : Nat) * (ρ d * S + α d) :=
  DotLoop.bound_le (ρ_nonneg d) (α_nonneg d) h hS

/-- **The assumptions are consistent**: the reference platform (`tl.dot` rounded once from the
exact result) meets `DotTotal` and `DotAcc` with these constants. -/
theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful (Platform.ref (fun _ _ => 0) (fun _ _ => 0)) model launch
  refine ⟨K, Platform.ref (fun _ _ => 0) (fun _ _ => 0), htot, ⟨fun d _ => hf d, Platform.ref.dotTotal cfg,
    Platform.ref.dotAcc (g := Format.binary32) rfl _ (fun K _ => ?_) (fun K _ => ?_)
      (rel_nonneg 0) (abs_nonneg' 0)⟩⟩
  · rw [DotLoop.binary32_u]; unfold rel
    have : (2 : ℚ) ≤ K + 2 := by linarith [(Nat.cast_nonneg K : (0 : ℚ) ≤ K)]
    have e : (2 : ℚ) ^ (-24 : ℤ) = 2 ^ (-22 : ℤ) / 4 := by norm_num
    rw [e]; nlinarith [show (0 : ℚ) < 2 ^ (-22 : ℤ) by positivity]
  · rw [binary32_eta]; unfold abs
    have : (2 : ℚ) ≤ K + 2 := by linarith [(Nat.cast_nonneg K : (0 : ℚ) ≤ K)]
    have e : (2 : ℚ) ^ (-150 : ℤ) = 2 ^ (-126 : ℤ) / 2 ^ 24 := by norm_num
    rw [e]; nlinarith [show (0 : ℚ) < 2 ^ (-126 : ℤ) by positivity]

end TVAExamples.GroupedGemm
