import StreamKMatmul.Program

/-!
# `streamk_matmul_f16`: Stream-K matrix multiplication, `float16` into `float32`

Source: `source/streamk_matmul.py`. The kernel `streamk_matmul_kernel` is lines 26–64; it reads
the module-level names `triton` and `tl` (lines 22–23). Its host function `streamk_matmul` (lines
70–93) launches it on `float16` matrices `a` (`M × K`) and `b` (`K × N`) into a zero `float32`
matrix `c` (`M × N`), on a 1-D grid of `num_programs` programs (by default the device's
streaming multiprocessors) that split the `tiles · cdiv(K, 64)` iterations of the output tiles
evenly; a program that does a whole tile stores it, and the programs that share a tile add their
partial sums into it with `tl.atomic_add`.

The claim: on every covered instance, every launch that runs ends normally; each element `(i, j)`
of `c` is, whatever order the atomic additions ran in, a `float32` value within the explicit
tolerance `tol K A` of the exact dot product of row `i` of `a` and column `j` of `b` (when these are
finite); every byte outside the matrix `c` is unchanged.

* `StreamKMatmul.Defs`: the model, the conditions, the specification, the assumptions.
* `StreamKMatmul.Facts`: `SafeA` rules for a read-only segment and a `while` loop; replays keep
  elements readable; `int32` arithmetic.
* `StreamKMatmul.Sched`: the iterations' schedule: parts of tiles, owned and split tiles, lanes,
  footprints.
* `StreamKMatmul.Basic`: `other=0.0`, one accumulation step, the tolerance's size.
* `StreamKMatmul.Parts`: the model in parts (`body_eq`), the invariants.
* `StreamKMatmul.Chunk`, `.Tail`, `.Program`: one trip of the inner loop, the end of an iteration,
  one iteration and one program (`safe_program`).
* Here: the programs together (`TVA.shared`), `Post` (`post`), `main`, `consistent`,
  `satisfiable`.
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## The programs' footprints -/

theorem W_R {d : Inst} (hd : d.ok) (p : Nat) : Disjoint (W d p) (R d) :=
  (Disjoint.union_right hd.c_a hd.c_b).mono_left (W_sub p)

theorem cells_R {d : Inst} (hd : d.ok) : Disjoint (cellBytes (cells d)) (R d) :=
  (Disjoint.union_right hd.c_a hd.c_b).mono_left cells_sub

/-- Two programs own different tiles (`owns_unique`), so they store different elements. -/
theorem W_disj {d : Inst} (hd : d.ok) {p q : Nat} (hpq : p ≠ q) : Disjoint (W d p) (W d q) := by
  refine d.c.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro _ ⟨t, hot, x, hx, rfl⟩ ⟨t', hot', x', hx', hg⟩
  obtain ⟨rfl, -⟩ := lane_inj hd hx' hx hg
  exact hpq (owns_unique hd hot hot')

/-- The split tiles' elements are not the elements a program of the grid stores. -/
theorem cells_W {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    Disjoint (cellBytes (cells d)) (W d p) := by
  rw [Set.disjoint_left]
  rintro z ⟨_, ⟨t, e, hw, ⟨x, hx, rfl⟩, rfl⟩, hz⟩ ⟨e', ⟨t', hot', x', hx', rfl⟩, hz'⟩
  by_cases he : cG d t x = cG d t' x'
  · obtain ⟨rfl, -⟩ := lane_inj hd hx hx' he
    exact hw ⟨p, hp, hot'⟩
  · exact Set.disjoint_left.mp (d.c.span_disjoint fun h => he (by exact_mod_cast h)) hz hz'

/-- Two cells whose bytes meet are the same cell. -/
theorem cells_eq {d : Inst} :
    ∀ c ∈ cells d, ∀ c' ∈ cells d, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c' := by
  rintro _ ⟨t, e, -, -, rfl⟩ _ ⟨t', e', -, -, rfl⟩ ⟨z, hz, hz'⟩
  by_cases he : e = e'
  · rw [he]
  · exact absurd hz' (Set.disjoint_left.mp (d.c.span_disjoint fun h => he (by exact_mod_cast h)) hz)

theorem cells_present {d : Inst} (hd : d.ok) {init : Mem} (hvc : d.c.Valid init) :
    ∀ x ∈ cellBytes (cells d), init x ≠ none := by
  intro x hx
  refine hvc.2.2.2 x (d.c.elems_sub_bytes ?_ (cells_sub hx))
  rintro _ ⟨i, j, hi, hj, rfl⟩
  exact hd.c_len _ _ hi hj

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (np_lt hd) ?_
  have hpow : ∀ k : Nat, k = 1 ∨ k = 64 ∨ k = 128 → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl | rfl)
    exacts [⟨0, rfl⟩, ⟨6, rfl⟩, ⟨7, rfl⟩]
  intro s hs
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨fun k hk => hpow k (by simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk; omega),
      by simp⟩

/-! ## A split tile's element -/

theorem dotMag_nonneg (va vb : Nat → F16) (K : Nat) : 0 ≤ dotMag va vb K :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (mag16_nonneg _) (mag16_nonneg _)

theorem dotMag_le {va vb : Nat → F16} {K : Nat} (hK : K ≤ 2 ^ 13)
    (hfa : ∀ k, k < K → (va k).isFinite = true) (hfb : ∀ k, k < K → (vb k).isFinite = true) :
    dotMag va vb K ≤ 2 ^ 46 := by
  have h1 : dotMag va vb K ≤ ∑ _k ∈ Finset.range K, (65504 : ℚ) * 65504 := by
    refine Finset.sum_le_sum fun k hk => ?_
    rw [Finset.mem_range] at hk
    exact mul_le_mul (mag16_le (hfa k hk)) (mag16_le (hfb k hk)) (mag16_nonneg _) (by norm_num)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at h1
  have h2 : (K : ℚ) ≤ 2 ^ 13 := by exact_mod_cast hK
  nlinarith

/-- **The programs' operands on an element of a split tile**: each program that touches the tile
adds one operand, within `DotLoop.bound` of its part's exact sum, and the parts partition the
tile's `T` steps. So the operands add up to within `tolDot K A` of the exact sum, their
magnitudes to at most `A + tolDot K A`, and there are at most `T`. -/
theorem split_sums {d : Inst} (hd : d.ok) {t : Nat} (ht : t < ntiles d) {va vb : Nat → F16}
    (ops : Nat → Multiset ℚ)
    (hops : ∀ p, p < d.num_programs → (touches d p t → ∃ w, ops p = {w} ∧
        |w - partS d va vb (kfirst d p t) (klast d p t)| ≤
          DotLoop.bound rhoD alphaD (klast d p t - kfirst d p t) (partA d va vb (kfirst d p t) (klast d p t))) ∧
      (¬ touches d p t → ops p = 0)) :
    |∑ p ∈ Finset.range d.num_programs, (ops p).sum - dotSum va vb d.K| ≤ tolDot d.K (dotMag va vb d.K) ∧
      ∑ p ∈ Finset.range d.num_programs, ((ops p).map fun w => |w|).sum ≤
        dotMag va vb d.K + tolDot d.K (dotMag va vb d.K) ∧
      ∑ p ∈ Finset.range d.num_programs, Multiset.card (ops p) ≤ steps d.K := by
  have hA0 : ∀ p, 0 ≤ partA d va vb (kfirst d p t) (klast d p t) :=
    fun p => Finset.sum_nonneg fun u _ => chunkA_nonneg _ _ _ _
  have hSA : ∀ p, |partS d va vb (kfirst d p t) (klast d p t)| ≤ partA d va vb (kfirst d p t) (klast d p t) :=
    fun p => (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun u _ => abs_chunkS_le _ _ _ _)
  have hper : ∀ p ∈ Finset.range d.num_programs,
      |(ops p).sum - partS d va vb (kfirst d p t) (klast d p t)| ≤
          DotLoop.bound rhoD alphaD (klast d p t - kfirst d p t) (partA d va vb (kfirst d p t) (klast d p t)) ∧
        ((ops p).map fun w => |w|).sum ≤ partA d va vb (kfirst d p t) (klast d p t) +
          DotLoop.bound rhoD alphaD (klast d p t - kfirst d p t) (partA d va vb (kfirst d p t) (klast d p t)) ∧
        Multiset.card (ops p) = if touches d p t then 1 else 0 := by
    intro p hp
    rw [Finset.mem_range] at hp
    obtain ⟨hon, hoff⟩ := hops p hp
    by_cases htc : touches d p t
    · obtain ⟨w, hw, herr⟩ := hon htc
      rw [hw, ite_eq_left htc]
      simp only [Multiset.sum_singleton, Multiset.map_singleton, Multiset.card_singleton]
      refine ⟨herr, ?_, by simp⟩
      have := abs_sub_abs_le_abs_sub w (partS d va vb (kfirst d p t) (klast d p t))
      linarith [hSA p]
    · have hle : klast d p t ≤ kfirst d p t := by unfold touches at htc; unfold kfirst klast; omega
      have hS0 : partS d va vb (kfirst d p t) (klast d p t) = 0 := by
        unfold partS; rw [Finset.Ico_eq_empty_of_le hle, Finset.sum_empty]
      have hA0' : partA d va vb (kfirst d p t) (klast d p t) = 0 := by
        unfold partA; rw [Finset.Ico_eq_empty_of_le hle, Finset.sum_empty]
      rw [hoff htc, ite_eq_right htc, hS0, hA0', Nat.sub_eq_zero_of_le hle]
      simp [DotLoop.bound]
  have hSsum : ∑ p ∈ Finset.range d.num_programs, partS d va vb (kfirst d p t) (klast d p t) =
      dotSum va vb d.K := by
    unfold partS; rw [sum_parts hd ht]; exact (chunks_end va vb d.K).1
  have hAsum : ∑ p ∈ Finset.range d.num_programs, partA d va vb (kfirst d p t) (klast d p t) =
      dotMag va vb d.K := by
    unfold partA; rw [sum_parts hd ht]; exact (chunks_end va vb d.K).2
  have hBsum : ∑ p ∈ Finset.range d.num_programs,
      DotLoop.bound rhoD alphaD (klast d p t - kfirst d p t) (partA d va vb (kfirst d p t) (klast d p t)) ≤
        tolDot d.K (dotMag va vb d.K) := by
    rw [← hAsum]
    exact sum_bound_le relDot16_nonneg absDot16_nonneg _ _ _ (fun p _ => hA0 p) (sum_part_len hd ht)
  refine ⟨?_, ?_, ?_⟩
  · rw [← hSsum, ← Finset.sum_sub_distrib]
    exact (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun p hp => (hper p hp).1).trans hBsum)
  · refine (Finset.sum_le_sum fun p hp => (hper p hp).2.1).trans ?_
    rw [Finset.sum_add_distrib, hAsum]
    linarith [hBsum]
  · rw [Finset.sum_congr rfl fun p hp => (hper p hp).2.2]
    exact card_touch_le hd ht

/-- The element's error: the atomic additions' bound on top of the operands'. -/
theorem split_err {K : Nat} {A S Sw Aw y : ℚ} {n : Nat} (hn : n ≤ steps K)
    (hS : |Sw - S| ≤ tolDot K A) (hAw : Aw ≤ A + tolDot K A) (hAw0 : 0 ≤ Aw)
    (hy : |y - (0 + Sw)| ≤ DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) n (|0| + Aw)) :
    |y - S| ≤ tol K A := by
  rw [zero_add, abs_zero, zero_add] at hy
  have hb := bound_mono (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (by positivity) (by positivity) hn hAw0 hAw
  have := abs_sub_le y Sw S
  unfold tol
  linarith

/-- Room for the atomic additions' bound in `float32`. -/
theorem split_room {K : Nat} (hK : K ≤ 2 ^ 13) {A Aw : ℚ} {n : Nat} (hA : 0 ≤ A) (hA46 : A ≤ 2 ^ 46)
    (hn : n ≤ steps K) (hAw : Aw ≤ A + tolDot K A) (hAw0 : 0 ≤ Aw) :
    |(0 : ℚ)| + Aw + DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) n (|0| + Aw) ≤
      Math.Float.Format.binary32.maxVal := by
  rw [abs_zero, zero_add, Math.Float.Format.binary32_maxVal]
  have hb := bound_mono (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (by positivity) (by positivity) hn hAw0 hAw
  have ht := tol_le hK hA
  unfold tol at ht
  have h1 : A + 21 / 10000 * A + 1 / 10 ^ 33 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by nlinarith
  have h2 := add_le_add hAw hb
  have h3 : A + tolDot K A + DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (steps K) (A + tolDot K A) ≤
      (2 ^ 24 - 1) * 2 ^ 104 := by
    generalize DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (steps K) (A + tolDot K A) = b at ht ⊢
    generalize tolDot K A = t at ht ⊢
    linarith
  exact h2.trans h3

/-! ## The launch's result -/

/-- **`Post` from the theorem `shared`.** An element of a whole tile is what its owner stored
(`TileOut`); an element of a split tile is the replay of the programs' atomic additions, each
paired with its operand, from the zero `c` held (`replay_err_shared`, `split_sums`). -/
theorem post {d : Inst} (hd : d.ok) {init fin : Mem} (hpre : Pre d init)
    (G : List (Fin (launch d).size × AtomicStep))
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init i (proj G i) m ∧ ∀ x ∈ R d ∪ W d i, fin x = m x)
    (hA : ∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes (cells d), fin x = mA x)
    (hfr : ∀ x, x ∉ cellBytes (cells d) → (∀ i : Fin (launch d).size, x ∉ W d i) → fin x = init x) :
    Post d init fin := by
  have hsize := launch_size d
  refine ⟨fun r q hr hq => ?_, fun z hz => hfr z (fun h => hz (cells_sub h)) fun i hi => hz (W_sub i hi)⟩
  obtain ⟨hrow, hcol, htl⟩ := tileOf_lane hd hr hq
  obtain ⟨hen, hg⟩ := tileOf_en hd hr hq
  generalize tileOf d r q = t at hrow hcol htl hen hg
  generalize laneOf r q = x at hrow hcol hen hg
  unfold OutSpec
  rw [← hg]
  have hRA : ∀ va : Nat → F16, (∀ k, k < d.K →
      init.read .f16 (d.a.ptr ↑(r * d.stride_am + k * d.stride_ak)).addr = some (va k) ∧ (va k).isFinite = true) →
      RowA d init (rowOf d t x.1) va := fun va h => by rw [hrow]; exact h
  have hCB : ∀ vb : Nat → F16, (∀ k, k < d.K →
      init.read .f16 (d.b.ptr ↑(k * d.stride_bk + q * d.stride_bn)).addr = some (vb k) ∧ (vb k).isFinite = true) →
      ColB d init (colOf d t x.2.1) vb := fun vb h => by rw [hcol]; exact h
  by_cases hw : whole d t
  · -- a whole tile: its owner's store
    obtain ⟨p, hp, how⟩ := hw
    obtain ⟨m, ⟨hmem, -⟩, hag⟩ := hloc ⟨p, by rw [hsize]; exact hp⟩
    have htc : touches d p t := by unfold owns at how; unfold touches tlo thi; have := steps_pos hd; omega
    obtain ⟨c, hc, hcs⟩ := hmem t x how htc hen
    refine ⟨c, ?_, fun va vb hra hcb => ?_⟩
    · rw [← hc]
      exact Mem.read_congr fun z hz => hag z (Or.inr ⟨cG d t x, ⟨t, how, x, hen, rfl⟩, hz⟩)
    · obtain ⟨hf, he⟩ := hcs va vb (hRA va hra) (hCB vb hcb)
      exact ⟨hf, he.trans (tolDot_le_tol _ (dotMag_nonneg _ _ _))⟩
  · -- a split tile: the replay of the atomic additions
    obtain ⟨mA, hrep, hagA⟩ := hA
    have hcell : ((.f32 : DType), (d.c.ptr ↑(cG d t x)).addr) ∈ cells d := ⟨t, cG d t x, hw, ⟨x, hen, rfl⟩, rfl⟩
    have hfinA : fin.read .f32 (d.c.ptr ↑(cG d t x)).addr = mA.read .f32 (d.c.ptr ↑(cG d t x)).addr :=
      Mem.read_congr fun z hz => hagA z ⟨_, hcell, hz⟩
    obtain ⟨z, hz0, hzf, hzv⟩ := hpre.2.2.2 r q hr hq
    rw [← hg] at hz0
    obtain ⟨y, hy⟩ := replay_read_some hrep hz0
    refine ⟨y, by rw [hfinA, hy], fun va vb hra hcb => ?_⟩
    have hra' := hRA va hra
    have hcb' := hCB vb hcb
    have hops : ∀ i : Fin (launch d).size, ∃ ops : Multiset ℚ, Paired (AccRel d (cG d t x)) (proj G i) ops ∧
        (touches d i t → ∃ w : ℚ, ops = {w} ∧
          |w - partS d va vb (kfirst d i t) (klast d i t)| ≤
            DotLoop.bound rhoD alphaD (klast d i t - kfirst d i t) (partA d va vb (kfirst d i t) (klast d i t))) ∧
        (¬ touches d i t → ops = 0) := by
      intro i
      obtain ⟨m, ⟨-, hlog⟩, -⟩ := hloc i
      exact hlog t x hw hen va vb hra' hcb'
    choose ops hpair hon hoff using hops
    -- the operands, indexed by the program number
    let opsN : Nat → Multiset ℚ := fun p => if h : p < (launch d).size then ops ⟨p, h⟩ else 0
    have hconv : ∀ {β : Type} [AddCommMonoid β] (g : Multiset ℚ → β),
        ∑ i : Fin (launch d).size, g (ops i) = ∑ p ∈ Finset.range d.num_programs, g (opsN p) := by
      intro β _ g
      rw [← hsize, ← Fin.sum_univ_eq_sum_range]
      refine Finset.sum_congr rfl fun i _ => ?_
      simp only [opsN, dite_eq_left i.2]
    have hopsN : ∀ p, p < d.num_programs → (touches d p t → ∃ w, opsN p = {w} ∧
        |w - partS d va vb (kfirst d p t) (klast d p t)| ≤
          DotLoop.bound rhoD alphaD (klast d p t - kfirst d p t) (partA d va vb (kfirst d p t) (klast d p t))) ∧
        (¬ touches d p t → opsN p = 0) := by
      intro p hp
      have hp' : p < (launch d).size := by rw [hsize]; exact hp
      simp only [opsN, dite_eq_left hp']
      exact ⟨hon ⟨p, hp'⟩, hoff ⟨p, hp'⟩⟩
    obtain ⟨hS, hAw, hn⟩ := split_sums hd htl opsN hopsN
    have hfa : ∀ k, k < d.K → (va k).isFinite = true := fun k hk => (hra k hk).2
    have hfb : ∀ k, k < d.K → (vb k).isFinite = true := fun k hk => (hcb k hk).2
    have hAm := dotMag_nonneg va vb d.K
    have hAw0 : 0 ≤ ∑ p ∈ Finset.range d.num_programs, ((opsN p).map fun w => |w|).sum :=
      Finset.sum_nonneg fun p _ => Multiset.sum_nonneg fun w hw => by
        obtain ⟨w', -, rfl⟩ := Multiset.mem_map.mp hw; exact abs_nonneg _
    have hA46 : dotMag va vb d.K ≤ 2 ^ 46 := dotMag_le hd.K_le hfa hfb
    have hroom := split_room hd.K_le hAm hA46 hn hAw hAw0
    rw [← hzv, ← hconv (fun o => Multiset.card o), ← hconv (fun o => (o.map fun w => |w|).sum)] at hroom
    have hρ : (0 : ℚ) ≤ 2 ^ (-23 : ℤ) := by positivity
    have hα : (0 : ℚ) ≤ 2 ^ (-124 : ℤ) := by positivity
    obtain ⟨y', hy', hgy, herr⟩ := replay_err_shared (d := .f32) (a := (d.c.ptr ↑(cG d t x)).addr)
      (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (B := Math.Float.Format.binary32.maxVal)
      FloatV.toRat (fun z : F32 => z.isFinite = true) hρ hα hrep hpair hz0 hzf hroom
    rw [hy] at hy'
    cases hy'
    rw [hzv, hconv (fun o => o.sum), hconv (fun o => Multiset.card o),
      hconv (fun o => (o.map fun w => |w|).sum)] at herr
    exact ⟨hgy, split_err hn hS hAw hAw0 herr⟩

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_of_faithful hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd) fun d hd init hpre => ?_
  have hsize := launch_size d
  obtain ⟨hwb, hout⟩ := shared (k := model d) (L := launch d) (E := (launch d).env P) (init := init)
    (R := fun _ => R d) (W := fun i => W d i) (C := cells d) (Q := fun i log m => Q d init i log m)
    (fun i => safe_program hd hpre hA.dot hA.dot_total hA.atomic hA.atomic_total (lt_of_lt_of_eq i.isLt hsize))
    (fun i => Q_perm d init i)
    (fun i j hij => Disjoint.union_right (W_R hd i) (W_disj hd fun h => hij (Fin.ext h)))
    (fun i => (W_R hd i).symm)
    (fun i => Disjoint.union_right (cells_R hd) (cells_W hd (lt_of_lt_of_eq i.isLt hsize)))
    cells_eq (cells_present hd hpre.2.2.1)
  refine ⟨hwb, fun o ho => ?_⟩
  obtain ⟨fin, rfl, G, hloc, hAt, hfr⟩ := hout o ho
  exact ⟨fin, rfl, post hd hpre G hloc hAt hfr⟩

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotF16, Platform.ref.dotTotal _,
    Platform.sm90ref.atomicAddF32, Platform.ref.atomicAddTotal _⟩⟩

/-! ## A covered instance -/

/-- The source's first test (line 100): `M = N = K = 512`, row-major `a`, `b` and `c` (strides `512`
and `1`), each of `512 · 512` elements, at the aligned addresses `2^21`, `2·2^21` and `3·2^21`, with
the default `num_programs`, the 132 streaming multiprocessors of an H200: `total_iters =
4 · 4 · 8 = 128`, `iters_per_program = cdiv(128, 132) = 1`, so four programs have no iteration. -/
def testInst : Inst where
  M := 512
  N := 512
  K := 512
  stride_am := 512
  stride_ak := 1
  stride_bk := 512
  stride_bn := 1
  stride_cm := 512
  stride_cn := 1
  num_programs := 132
  iters_per_program := 1
  total_iters := 128
  a := ⟨2097152, 262144⟩
  b := ⟨4194304, 262144⟩
  c := ⟨6291456, 262144⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

/-- Every aligned `float32` read of `testMem` gives `+0`. -/
theorem testMem_read {a : Int} (ha : Mem.Aligned .f32 a) : testMem.read .f32 a = some (FloatV.ofPat 0) := by
  have hb : testMem.readBytes a 4 = some [0, 0, 0, 0] := by
    simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.f32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb]
  decide

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hvalid : ∀ {dd : DType} (b : Buf dd), 0 ≤ b.base → b.base + b.len * dd.bytes ≤ 2 ^ 64 →
      b.base % dd.bytes = 0 → b.Valid testMem :=
    fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [testMem]⟩
  refine ⟨testInst, testMem, ⟨by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst, tilesM, tilesN, steps], by norm_num [testInst], by norm_num [testInst],
    fun i k hi hk => by simp only [testInst] at hi hk ⊢; omega,
    fun k j hk hj => by simp only [testInst] at hk hj ⊢; omega,
    fun i j hi hj => by simp only [testInst] at hi hj ⊢; omega, ?_, ?_⟩,
    hvalid _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hvalid _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hvalid _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    fun i j hi hj => ?_⟩
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, testInst, Set.mem_ofPred_eq, DType.bytes] at hx hy
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      simp only [testInst] at hi hj ⊢; omega
    · rintro _ ⟨i, k, hi, hk, rfl⟩
      simp only [testInst] at hi hk ⊢; omega
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, testInst, Set.mem_ofPred_eq, DType.bytes] at hx hy
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      simp only [testInst] at hi hj ⊢; omega
    · rintro _ ⟨k, j, hk, hj, rfl⟩
      simp only [testInst] at hk hj ⊢; omega
  · have h := DotLoop.val?_f32.mp DotLoop.zero_val
    refine ⟨FloatV.ofPat 0, testMem_read ⟨?_, ?_, ?_⟩, h.1, h.2⟩ <;>
      simp only [Buf.ptr, testInst, DType.bytes] at hi hj ⊢ <;> omega

end Kernels.StreamKMatmul
