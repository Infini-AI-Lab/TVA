import Claim
import TVA.Proof
import Proof.Lanes

/-!
# `l2norm_bwd_bf16`: the grid, the footprints and the per-program result

Program `p` of the grid reads the `[BT, D]` tiles of `y` and `dy` and its `BT` elements of `rstd`,
writes its `[BT, D]` tile of `dx` (`R`, `W`), and leaves there the `dx` values `Post` asks for on
its rows (`Q`). This file proves the grid facts (`launch_size`, `launch_pid`, validity), that
distinct programs have disjoint write footprints and that no program reads what it writes
(`hdisj`, `hRW`), and that the per-program results give `Post` (`hpost`).
-/

namespace Claims.L2NormBwd

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

/-- The bytes program `p` reads: the elements of its tiles of `y` and `dy`, and its rows of
`rstd`. -/
def R (d : Inst) (_ : Mem) (p : Fin (launch d).size) : Set Int :=
  elemsOf d.y (tileIdx d p.val) ∪ elemsOf d.dy (tileIdx d p.val) ∪ d.rstd.elems (tileRow d p.val)

/-- The bytes program `p` writes: its tile of `dx`. -/
def W (d : Inst) (_ : Mem) (p : Fin (launch d).size) : Set Int :=
  elemsOf d.dx (tileIdx d p.val)

/-- What program `p` leaves in memory: on each of its rows, the `dx` values of `Post`. -/
def Qn (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ t ∈ tileRow d p, ∃ (a g : Nat → ℝ) (r : ℝ),
    (∀ j, j < d.D → ValAt d.y init (xIdx d t j) (a j)) ∧
    (∀ j, j < d.D → ValAt d.dy init (xIdx d t j) (g j)) ∧ RstdAt d init t r ∧
    ∀ j, j < d.D → DxOk d m (xIdx d t j)
      (g j * r - rowDot a g d.D * a j * r)
      (|r| * (|g j| + rowAbs a g d.D * |a j|))
      ((1 + |r|) * (1 + |a j|))

/-- `Qn`, indexed by a program of the grid. -/
def Q (d : Inst) (init : Mem) (p : Fin (launch d).size) (m : Mem) : Prop := Qn d init p.val m

/-! ## Element sets of a buffer of either dtype -/

/-- `elemsOf` grows with the element set. -/
theorem elemsOf_mono (b : Sum (Buf .bf16) (Buf .f32)) {S T : Set Nat} (h : S ⊆ T) :
    elemsOf b S ⊆ elemsOf b T := by
  cases b with
  | inl b => exact Buf.elems_mono b h
  | inr b => exact Buf.elems_mono b h

/-- Disjoint element sets of one buffer have disjoint bytes. -/
theorem elemsOf_disjoint (b : Sum (Buf .bf16) (Buf .f32)) {S T : Set Nat} (h : Disjoint S T) :
    Disjoint (elemsOf b S) (elemsOf b T) := by
  cases b with
  | inl b => exact Buf.elems_disjoint b h
  | inr b => exact Buf.elems_disjoint b h

/-- The bytes of element `k` of a buffer of either dtype. -/
def spanOf (b : Sum (Buf .bf16) (Buf .f32)) (k : Nat) : Set Int :=
  match b with
  | .inl b => Mem.span .bf16 (b.ptr (k : Int)).addr
  | .inr b => Mem.span .f32 (b.ptr (k : Int)).addr

/-- The bytes of element `k ∈ S` lie in `elemsOf b S`. -/
theorem spanOf_sub (b : Sum (Buf .bf16) (Buf .f32)) {S : Set Nat} {k : Nat} (hk : k ∈ S) :
    spanOf b k ⊆ elemsOf b S := by
  cases b with
  | inl b => exact Buf.span_sub_elems b hk
  | inr b => exact Buf.span_sub_elems b hk

/-- `DxOk` for element `k` of `dx` depends only on the bytes of that element. -/
theorem DxOk.congr_read {d : Inst} {m m' : Mem} {k : Nat} {ref M F : ℝ}
    (h : ∀ x ∈ spanOf d.dx k, m' x = m x) : DxOk d m k ref M F → DxOk d m' k ref M F := by
  unfold DxOk
  cases hb : d.dx with
  | inl b =>
    have hs : spanOf d.dx k = Mem.span .bf16 (b.ptr (k : Int)).addr := by simp only [spanOf, hb]
    rintro ⟨c, hc, hf, hbd⟩
    exact ⟨c, by rw [Mem.read_congr (fun x hx => h x (by rw [hs]; exact hx)), hc], hf, hbd⟩
  | inr b =>
    have hs : spanOf d.dx k = Mem.span .f32 (b.ptr (k : Int)).addr := by simp only [spanOf, hb]
    rintro ⟨c, hc, hf, hbd⟩
    exact ⟨c, by rw [Mem.read_congr (fun x hx => h x (by rw [hs]; exact hx)), hc], hf, hbd⟩

/-! ## Disjointness of the footprints -/

/-- The rows of a program's block are among the `T` rows of `rstd`. -/
theorem tileRow_subset_rSet {d : Inst} {p : Nat} : tileRow d p ⊆ rSet d := fun _ h => h.1

/-- What a program writes is disjoint from what any program reads. -/
theorem hWR {d : Inst} (hd : d.ok) (i j : Nat) :
    Disjoint (elemsOf d.dx (tileIdx d i))
      (elemsOf d.y (tileIdx d j) ∪ elemsOf d.dy (tileIdx d j) ∪ d.rstd.elems (tileRow d j)) := by
  have hxi : tileIdx d i ⊆ rowSet d := tileIdx_subset_rowSet hd
  have hxj : tileIdx d j ⊆ rowSet d := tileIdx_subset_rowSet hd
  have hrj : tileRow d j ⊆ rSet d := tileRow_subset_rSet
  exact Disjoint.union_right
    (Disjoint.union_right (hd.dx_y.mono (elemsOf_mono _ hxi) (elemsOf_mono _ hxj))
      (hd.dx_dy.mono (elemsOf_mono _ hxi) (elemsOf_mono _ hxj)))
    (hd.dx_r.mono (elemsOf_mono _ hxi) (Buf.elems_mono _ hrj))

/-- What program `i` writes is disjoint from what any other program `j` reads or writes. -/
theorem hdisj {d : Inst} (hd : d.ok) (init : Mem) (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hij' : i.val ≠ j.val := fun h => hij (Fin.ext h)
  simp only [W, R]
  exact Disjoint.union_right (hWR hd i.val j.val)
    (elemsOf_disjoint _ (tileIdx_disjoint hd hij'))

/-- A program does not read what it writes. -/
theorem hRW {d : Inst} (hd : d.ok) (init : Mem) (i : Fin (launch d).size) :
    Disjoint (R d init i) (W d init i) := by
  simp only [W, R]
  exact (hWR hd i.val i.val).symm

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
    obtain ⟨a, g, r, ha, hg, hr, hdx⟩ := hQ t htrow
    refine ⟨a, g, r, ha, hg, hr, fun j hj => ?_⟩
    refine DxOk.congr_read (fun x hx => hagree x (Or.inr ?_)) (hdx j hj)
    exact spanOf_sub _ (S := tileIdx d (t / d.BT)) ⟨t, htrow, j, hj, rfl⟩ hx
  · intro z hz
    apply hfr
    intro i hzi
    simp only [W] at hzi
    exact hz (elemsOf_mono _ (tileIdx_subset_rowSet hd) hzi)

end Claims.L2NormBwd
