import Claim
import TVA.Proof
import Proof.Lanes

/-!
# `l2norm_fwd_bf16`: the grid, the footprints and the per-program result

Program `p` of the grid reads the `[BT, D]` tile of `x`, writes its `[BT, D]` tile of `y` and its
`BT` elements of `rstd` (`R`, `W`), and leaves there the `rstd` and `y` values `Post` asks for on
its rows (`Q`). This file proves the grid facts (`launch_size`, `launch_pid`, validity), that
distinct programs have disjoint write footprints and that no program reads what it writes
(`hdisj`, `hRW`), and that the per-program results give `Post` (`hpost`).
-/

namespace Claims.L2NormFwd

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

/-- The bytes program `p` reads: the elements of its tile of `x`. -/
def R (d : Inst) (_ : Mem) (p : Fin (launch d).size) : Set Int :=
  d.x.elems (tileIdx d p.val)

/-- The bytes program `p` writes: its tile of `y` and its `BT` elements of `rstd`. -/
def W (d : Inst) (_ : Mem) (p : Fin (launch d).size) : Set Int :=
  yElems d (tileIdx d p.val) ∪ d.rstd.elems (tileRow d p.val)

/-- What program `p` leaves in memory: on each of its rows, the `rstd` and `y` values of `Post`. -/
def Qn (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ t ∈ tileRow d p, ∃ a : Nat → BF16,
    (∀ j, j < d.D → init.read .bf16 (d.x.ptr (xIdx d t j)).addr = some (a j)) ∧
    (∃ rc : F32, m.read .f32 (d.rstd.ptr t).addr = some rc ∧ rc.isFinite = true ∧
      |rc.toReal - rstdRef a d.D d.eps.toReal| ≤
        (rstdTol : ℝ) * rstdRef a d.D d.eps.toReal) ∧
    (∀ j, j < d.D → YOk d m (xIdx d t j) ((a j).toReal * rstdRef a d.D d.eps.toReal)
      (|(a j).toReal| * rstdRef a d.D d.eps.toReal))

/-- `Qn`, indexed by a program of the grid. -/
def Q (d : Inst) (init : Mem) (p : Fin (launch d).size) (m : Mem) : Prop := Qn d init p.val m

/-! ## Monotonicity of `yElems` -/

/-- `yElems` grows with the element set. -/
theorem yElems_mono {d : Inst} {S T : Set Nat} (h : S ⊆ T) : yElems d S ⊆ yElems d T := by
  cases hy : d.y with
  | inl b => simpa only [yElems, hy] using Buf.elems_mono b h
  | inr b => simpa only [yElems, hy] using Buf.elems_mono b h

/-- Disjoint element sets of `y` have disjoint bytes. -/
theorem yElems_disjoint {d : Inst} {S T : Set Nat} (h : Disjoint S T) :
    Disjoint (yElems d S) (yElems d T) := by
  cases hy : d.y with
  | inl b => simpa only [yElems, hy] using Buf.elems_disjoint b h
  | inr b => simpa only [yElems, hy] using Buf.elems_disjoint b h

/-- The bytes of element `k` of `y`. -/
def ySpan (d : Inst) (k : Nat) : Set Int :=
  match d.y with
  | .inl b => Mem.span .bf16 (b.ptr (k : Int)).addr
  | .inr b => Mem.span .f32 (b.ptr (k : Int)).addr

/-- The bytes of element `k ∈ S` of `y` lie in `yElems d S`. -/
theorem ySpan_sub {d : Inst} {S : Set Nat} {k : Nat} (hk : k ∈ S) : ySpan d k ⊆ yElems d S := by
  cases hy : d.y with
  | inl b => simpa only [ySpan, yElems, hy] using Buf.span_sub_elems b hk
  | inr b => simpa only [ySpan, yElems, hy] using Buf.span_sub_elems b hk

/-- `YOk` for element `k` of `y` depends only on the bytes of that element. -/
theorem YOk.congr_read {d : Inst} {m m' : Mem} {k : Nat} {ref scale : ℝ}
    (h : ∀ x ∈ ySpan d k, m' x = m x) : YOk d m k ref scale → YOk d m' k ref scale := by
  cases hy : d.y with
  | inl b =>
    have hs : ySpan d k = Mem.span .bf16 (b.ptr (k : Int)).addr := by simp only [ySpan, hy]
    simp only [YOk, hy]
    rintro ⟨c, hc, hf, hb⟩
    exact ⟨c, by rw [Mem.read_congr (fun x hx => h x (by rw [hs]; exact hx)), hc], hf, hb⟩
  | inr b =>
    have hs : ySpan d k = Mem.span .f32 (b.ptr (k : Int)).addr := by simp only [ySpan, hy]
    simp only [YOk, hy]
    rintro ⟨c, hc, hf, hb⟩
    exact ⟨c, by rw [Mem.read_congr (fun x hx => h x (by rw [hs]; exact hx)), hc], hf, hb⟩

/-! ## Disjointness of the footprints -/

/-- The rows of a program's block are among the `T` rows of `rstd`. -/
theorem tileRow_subset_rSet {d : Inst} {p : Nat} : tileRow d p ⊆ rSet d := fun _ h => h.1

/-- What program `i` writes is disjoint from what any other program `j` reads or writes. -/
theorem hdisj {d : Inst} (hd : d.ok) (init : Mem) (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hij' : i.val ≠ j.val := fun h => hij (Fin.ext h)
  have hxi : tileIdx d i.val ⊆ rowSet d := tileIdx_subset_rowSet hd
  have hxj : tileIdx d j.val ⊆ rowSet d := tileIdx_subset_rowSet hd
  have hri : tileRow d i.val ⊆ rSet d := tileRow_subset_rSet
  have hrj : tileRow d j.val ⊆ rSet d := tileRow_subset_rSet
  have hAC : Disjoint (yElems d (tileIdx d i.val)) (d.x.elems (tileIdx d j.val)) :=
    hd.y_x.mono (yElems_mono hxi) (Buf.elems_mono _ hxj)
  have hAA : Disjoint (yElems d (tileIdx d i.val)) (yElems d (tileIdx d j.val)) :=
    yElems_disjoint (tileIdx_disjoint hd hij')
  have hAB : Disjoint (yElems d (tileIdx d i.val)) (d.rstd.elems (tileRow d j.val)) :=
    hd.r_y.symm.mono (yElems_mono hxi) (Buf.elems_mono _ hrj)
  have hBC : Disjoint (d.rstd.elems (tileRow d i.val)) (d.x.elems (tileIdx d j.val)) :=
    hd.r_x.mono (Buf.elems_mono _ hri) (Buf.elems_mono _ hxj)
  have hBA : Disjoint (d.rstd.elems (tileRow d i.val)) (yElems d (tileIdx d j.val)) :=
    hd.r_y.mono (Buf.elems_mono _ hri) (yElems_mono hxj)
  have hBB : Disjoint (d.rstd.elems (tileRow d i.val)) (d.rstd.elems (tileRow d j.val)) :=
    Buf.elems_disjoint d.rstd (tileRow_disjoint (bt_pos hd) hij')
  simp only [W, R]
  exact Disjoint.union_left
    (Disjoint.union_right hAC (Disjoint.union_right hAA hAB))
    (Disjoint.union_right hBC (Disjoint.union_right hBA hBB))

/-- A program does not read what it writes. -/
theorem hRW {d : Inst} (hd : d.ok) (init : Mem) (i : Fin (launch d).size) :
    Disjoint (R d init i) (W d init i) := by
  have hxi : tileIdx d i.val ⊆ rowSet d := tileIdx_subset_rowSet hd
  have hri : tileRow d i.val ⊆ rSet d := tileRow_subset_rSet
  simp only [W, R]
  exact Disjoint.union_right
    (hd.y_x.symm.mono (Buf.elems_mono _ hxi) (yElems_mono hxi))
    (hd.r_x.symm.mono (Buf.elems_mono _ hxi) (Buf.elems_mono _ hri))

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
    obtain ⟨a, ha, ⟨rc, hrc, hrf, hrb⟩, hy⟩ := hQ t htrow
    refine ⟨a, ha, ⟨rc, ?_, hrf, hrb⟩, fun j hj => ?_⟩
    · rw [Mem.read_congr (d := .f32) (a := (d.rstd.ptr t).addr)
        (fun x hx => hagree x (Or.inr (Or.inr (Buf.span_sub_elems d.rstd htrow hx)))), hrc]
    · refine YOk.congr_read (fun x hx => hagree x (Or.inr (Or.inl ?_))) (hy j hj)
      exact ySpan_sub (S := tileIdx d (t / d.BT)) ⟨t, htrow, j, hj, rfl⟩ hx
  · intro z hz
    apply hfr
    intro i hzi
    simp only [W] at hzi
    rcases hzi with hzi | hzi
    · exact hz (Or.inl (yElems_mono (tileIdx_subset_rowSet hd) hzi))
    · exact hz (Or.inr (Buf.elems_mono _ tileRow_subset_rSet hzi))

end Claims.L2NormFwd
