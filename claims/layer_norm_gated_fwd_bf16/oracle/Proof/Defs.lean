import Claim
import TVA.Proof
import Proof.Lanes

/-!
# `layer_norm_gated_fwd_bf16`: the grid, the footprints and the per-program result

Program `p` of the grid reads the `[BT, D]` tiles of `x` and `g` and the first `D` elements of
`weight`, writes its `[BT, D]` tile of `y` and its `BT` elements of `rstd` (`R`, `W`), and leaves
there the `rstd` and `y` values `Post` asks for on its rows (`Q`). This file proves the grid facts (`launch_size`, `launch_pid`, validity), that
distinct programs have disjoint write footprints and that no program reads what it writes
(`hdisj`, `hRW`), and that the per-program results give `Post` (`hpost`).
-/

namespace Claims.LayerNormGatedFwd

open TVA

/-! ## The grid and the launch -/

/-- The number of programs, `cdiv(T, BT)`. -/
abbrev grid (d : Inst) : Nat := (d.T + d.BT - 1) / d.BT

/-- The launch has one program per `cdiv(T, BT)` block of rows. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

/-- Program `i` of the launch has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch runs with `debug = false`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- `BD` is positive (a power of two). -/
theorem bd_pos {d : Inst} (hd : d.ok) : 0 < d.BD := by
  obtain ⟨e, he⟩ := hd.bd_pow2
  rw [he]
  exact Nat.two_pow_pos e

/-- `BT` is positive (a power of two). -/
theorem bt_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  obtain ⟨e, he⟩ := hd.bt_pow2
  rw [he]
  exact Nat.two_pow_pos e

/-- `BT ≤ 2^20`, from the block bound `BT·BD ≤ 2^20`. -/
theorem bt_le {d : Inst} (hd : d.ok) : d.BT ≤ 2 ^ 20 := by
  have h1 : d.BT * d.BD ≤ 2 ^ 20 := hd.block_le
  have hb : 1 ≤ d.BD := by have := bd_pos hd; omega
  have h2 : d.BT * 1 ≤ d.BT * d.BD := Nat.mul_le_mul_left _ hb
  rw [Nat.mul_one] at h2
  omega

/-- `BD ≤ 2^20`. -/
theorem bd_le20 {d : Inst} (hd : d.ok) : d.BD ≤ 2 ^ 20 := by
  have := hd.bd_le
  omega

/-- The launch is valid: fewer than `2^31` programs, and every block shape is a power of two within `2^20` elements. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt (fun s hs => ?_)
  simp only [List.mem_cons, List.not_mem_nil,
    or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl
  · obtain ⟨e, he⟩ := hd.bt_pow2
    refine ⟨fun k hk => ⟨e, ?_⟩, ?_⟩
    · simp only [List.mem_singleton] at hk
      rw [hk, he]
    · simpa using bt_le hd
  · obtain ⟨e, he⟩ := hd.bd_pow2
    refine ⟨fun k hk => ⟨e, ?_⟩, ?_⟩
    · simp only [List.mem_singleton] at hk
      rw [hk, he]
    · simpa using bd_le20 hd
  · obtain ⟨e, he⟩ := hd.bt_pow2
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
      rcases hk with rfl | rfl
      · exact ⟨e, he⟩
      · exact ⟨0, by norm_num⟩
    · simpa using bt_le hd
  · obtain ⟨e, he⟩ := hd.bd_pow2
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
      rcases hk with rfl | rfl
      · exact ⟨0, by norm_num⟩
      · exact ⟨e, he⟩
    · simpa using bd_le20 hd
  · obtain ⟨ea, hea⟩ := hd.bt_pow2
    obtain ⟨eb, heb⟩ := hd.bd_pow2
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
      rcases hk with rfl | rfl
      · exact ⟨ea, hea⟩
      · exact ⟨eb, heb⟩
    · simpa using hd.block_le

/-- Each row is handled by a program of the grid: `t / BT < grid`. -/
theorem div_lt_grid {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) : t / d.BT < grid d := by
  have hB := bt_pos hd
  have h1 : t / d.BT ≤ (d.T - 1) / d.BT := Nat.div_le_div_right (by omega)
  have h2 : (d.T - 1) / d.BT < ((d.T - 1) + d.BT) / d.BT := by
    have h := Nat.add_mul_div_left (d.T - 1) 1 hB
    rw [Nat.mul_one] at h
    rw [h]
    exact Nat.lt_succ_self _
  have h3 : (d.T - 1) + d.BT = d.T + d.BT - 1 := by
    rw [Nat.add_sub_assoc (by omega : 1 ≤ d.BT)]
    have h4 : d.T - 1 + 1 = d.T := Nat.sub_add_cancel (by omega : 1 ≤ d.T)
    omega
  show t / d.BT < (d.T + d.BT - 1) / d.BT
  rw [← h3]
  exact lt_of_le_of_lt h1 h2

/-- A row of the block of program `p`, with `p` the row's own program. -/
theorem row_mem_tileRow {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    t ∈ tileRow d (t / d.BT) := by
  rw [mem_tileRow_iff]
  refine ⟨ht, t % d.BT, Nat.mod_lt _ (bt_pos hd), ?_⟩
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod t d.BT).symm

/-! ## Footprints and the per-program result -/

/-- The bytes program `p` reads: its tiles of `x` and `g`, and the first `D` elements of
`weight`. -/
def R (d : Inst) (_ : Mem) (p : Fin (launch d).size) : Set Int :=
  d.x.elems (tileIdx d p.val) ∪ d.g.elems (tileIdx d p.val) ∪ d.w.elems (vecSet d)

/-- The bytes program `p` writes: its tile of `y` and its `BT` elements of `rstd`. -/
def W (d : Inst) (_ : Mem) (p : Fin (launch d).size) : Set Int :=
  d.y.elems (tileIdx d p.val) ∪ d.rstd.elems (tileRow d p.val)

/-- What program `p` leaves in memory: each of its rows as `RowDone` says. -/
def Qn (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ t ∈ tileRow d p, RowDone d init m t

/-- `Qn`, indexed by a program of the grid. -/
def Q (d : Inst) (init : Mem) (p : Fin (launch d).size) (m : Mem) : Prop := Qn d init p.val m

/-! ## Disjointness of the footprints -/

/-- The rows of a program's block are among the `T` rows of `rstd`. -/
theorem tileRow_subset_rowSet {d : Inst} {p : Nat} : tileRow d p ⊆ rowSet d := fun _ h => h.1

/-- The bytes a program writes lie in `outBytes`. -/
theorem W_sub_out {d : Inst} (hd : d.ok) (init : Mem) (i : Fin (launch d).size) :
    W d init i ⊆ outBytes d :=
  Set.union_subset_union (Buf.elems_mono _ (tileIdx_subset_matSet hd))
    (Buf.elems_mono _ tileRow_subset_rowSet)

/-- The bytes a program reads lie in `inBytes`. -/
theorem R_sub_in {d : Inst} (hd : d.ok) (init : Mem) (i : Fin (launch d).size) :
    R d init i ⊆ inBytes d :=
  Set.union_subset_union (Set.union_subset_union (Buf.elems_mono _ (tileIdx_subset_matSet hd))
    (Buf.elems_mono _ (tileIdx_subset_matSet hd))) le_rfl

/-- What program `i` writes is disjoint from what any other program `j` reads or writes. -/
theorem hdisj {d : Inst} (hd : d.ok) (init : Mem) (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hij' : i.val ≠ j.val := fun h => hij (Fin.ext h)
  refine Disjoint.union_right (hd.out_in.mono (W_sub_out hd init i) (R_sub_in hd init j)) ?_
  have hxi : tileIdx d i.val ⊆ matSet d := tileIdx_subset_matSet hd
  have hxj : tileIdx d j.val ⊆ matSet d := tileIdx_subset_matSet hd
  have hAA : Disjoint (d.y.elems (tileIdx d i.val)) (d.y.elems (tileIdx d j.val)) :=
    Buf.elems_disjoint d.y (tileIdx_disjoint hd hij')
  have hAB : Disjoint (d.y.elems (tileIdx d i.val)) (d.rstd.elems (tileRow d j.val)) :=
    hd.y_rstd.mono (Buf.elems_mono _ hxi) (Buf.elems_mono _ tileRow_subset_rowSet)
  have hBA : Disjoint (d.rstd.elems (tileRow d i.val)) (d.y.elems (tileIdx d j.val)) :=
    hd.y_rstd.symm.mono (Buf.elems_mono _ tileRow_subset_rowSet) (Buf.elems_mono _ hxj)
  have hBB : Disjoint (d.rstd.elems (tileRow d i.val)) (d.rstd.elems (tileRow d j.val)) :=
    Buf.elems_disjoint d.rstd (tileRow_disjoint (bt_pos hd) hij')
  simp only [W]
  exact Disjoint.union_left (Disjoint.union_right hAA hAB) (Disjoint.union_right hBA hBB)

/-- A program does not read what it writes. -/
theorem hRW {d : Inst} (hd : d.ok) (init : Mem) (i : Fin (launch d).size) :
    Disjoint (R d init i) (W d init i) :=
  hd.out_in.symm.mono (R_sub_in hd init i) (W_sub_out hd init i)

/-! ## The per-program results give `Post` -/

/-- The per-program results `Q`, together with the untouched bytes, give `Post`. -/
theorem hpost {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size,
      ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init i) → fin x = init x) :
    Post d init fin := by
  constructor
  · intro t ht
    have hp : t / d.BT < (launch d).size := by
      rw [launch_size]
      exact div_lt_grid hd ht
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨t / d.BT, hp⟩
    have htrow : t ∈ tileRow d (t / d.BT) := row_mem_tileRow hd ht
    obtain ⟨⟨rc, hrc, hrf, hrb⟩, hy⟩ := hQ t htrow
    refine ⟨⟨rc, ?_, hrf, hrb⟩, fun j hj => ?_⟩
    · rw [Mem.read_congr (d := .f32) (a := (d.rstd.ptr t).addr)
        (fun x hx => hagree x (Or.inr (Or.inr (Buf.span_sub_elems d.rstd htrow hx)))), hrc]
    · obtain ⟨c, hc, hcf, hcb⟩ := hy j hj
      refine ⟨c, ?_, hcf, hcb⟩
      have hk : idx d t j ∈ tileIdx d (t / d.BT) := ⟨t, htrow, j, hj, rfl⟩
      rw [Mem.read_congr (d := .bf16) (a := (d.y.ptr (idx d t j : Int)).addr)
        (fun x hx => hagree x (Or.inr (Or.inl (Buf.span_sub_elems d.y hk hx)))), hc]
  · intro z hz
    apply hfr
    intro i hzi
    exact hz (W_sub_out hd init i hzi)

end Claims.LayerNormGatedFwd
