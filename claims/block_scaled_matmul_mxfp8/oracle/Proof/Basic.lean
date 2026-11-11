import Claim
import TVA.Proof

/-!
# `block_scaled_matmul_mxfp8`: the grid, the per-program footprints and the launch facts
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## The grid -/

/-- `cdiv(M, 128)`: the programs along rows of `c`. -/
def gridM (d : Inst) : Nat := (d.M + 127) / 128

/-- `cdiv(N, 256)`: the programs along columns of `c`. -/
def gridN (d : Inst) : Nat := (d.N + 255) / 256

/-- The number of programs, `cdiv(M, 128) · cdiv(N, 256)`. -/
def grid (d : Inst) : Nat := gridM d * gridN d

/-- The launch has `grid d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

/-- The launch does not set `debug`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- Program `i` has program id `i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hi' : (i : Nat) < (d.M + 127) / 128 * ((d.N + 255) / 256) := by
    simpa [grid, gridM, gridN] using hi
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi']

/-- `M` is positive. -/
theorem M_pos {d : Inst} (hd : d.ok) : 0 < d.M := hd.M_ok.1

/-- `N` is positive. -/
theorem N_pos {d : Inst} (hd : d.ok) : 0 < d.N := hd.N_ok.1

/-- `K` is positive. -/
theorem K_pos {d : Inst} (hd : d.ok) : 0 < d.K := hd.K_ok.1

/-- `M` is a multiple of `128`. -/
theorem M_div (d : Inst) (hd : d.ok) : d.M = 128 * (d.M / 128) :=
  (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hd.M_ok.2.1)).symm

/-- `N` is a multiple of `128`. -/
theorem N_div (d : Inst) (hd : d.ok) : d.N = 128 * (d.N / 128) :=
  (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hd.N_ok.2.1)).symm

/-- `K` is a multiple of `128`. -/
theorem K_div (d : Inst) (hd : d.ok) : d.K = 128 * (d.K / 128) :=
  (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hd.K_ok.2.1)).symm

/-- `gridM = M / 128`. -/
theorem gridM_eq {d : Inst} (hd : d.ok) : gridM d = d.M / 128 := by
  unfold gridM
  conv => lhs; rw [M_div d hd]
  rw [Nat.mul_add_div (by norm_num : 0 < 128), Nat.div_eq_of_lt (by norm_num : 127 < 128),
    Nat.add_zero]

/-- `gridN = cdiv(N, 256)`. -/
theorem gridN_eq {d : Inst} (_hd : d.ok) : gridN d = (d.N + 255) / 256 := rfl

/-- There is at least one program row. -/
theorem gridM_pos {d : Inst} (hd : d.ok) : 0 < gridM d := by
  have := M_pos hd
  unfold gridM
  omega

/-- There is at least one program column. -/
theorem gridN_pos {d : Inst} (hd : d.ok) : 0 < gridN d := by
  have := N_pos hd
  unfold gridN
  omega

/-- `gridM` is at most `M / 128`. -/
theorem gridM_le {d : Inst} (hd : d.ok) : gridM d ≤ d.M / 128 := by
  rw [gridM_eq hd]

/-- The grid has fewer than `2^31` programs. -/
theorem grid_lt {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := hd.grid_lt

/-- The shapes of `launch` are Triton block shapes. -/
theorem pow2_mem : ∀ k ∈ [128, 256, 64, 32, 16, 8, 4, 2, 1], ∃ e, k = 2 ^ e := by
  intro k hk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨7, by norm_num⟩
  · exact ⟨8, by norm_num⟩
  · exact ⟨6, by norm_num⟩
  · exact ⟨5, by norm_num⟩
  · exact ⟨4, by norm_num⟩
  · exact ⟨3, by norm_num⟩
  · exact ⟨2, by norm_num⟩
  · exact ⟨1, by norm_num⟩
  · exact ⟨0, by norm_num⟩

/-- Every block shape of the launch is one Triton accepts. -/
theorem block_ok : ∀ s ∈ [[128, 256], [128, 128], [256, 128], [128, 256], [256, 64], [64, 256],
      [1, 1, 1, 2, 256], [1, 1, 32, 4, 4], [1, 4, 32, 1, 4], [128, 4], [1, 2, 1, 2, 256],
      [2, 1, 32, 4, 4], [2, 4, 32, 1, 4], [256, 4]], Shape.TritonOk s := by
  intro s hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    refine ⟨fun k hk => ?_, by decide⟩ <;>
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk <;>
    rcases hk with rfl | rfl | rfl | rfl | rfl <;> exact pow2_mem _ (by simp)

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (grid_lt hd) fun s hs => ?_
  -- the two shapes that depend on `epbB`
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with h | h | h | h | h | h | h | h | h | h | h | h <;> subst h
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · -- [256, 128 / epbB]
    by_cases hm : d.mixed
    · have : (128 : Nat) / epbB d.mixed = 64 := by simp [epbB, hm]
      rw [this]; exact block_ok _ (by simp)
    · have : (128 : Nat) / epbB d.mixed = 128 := by simp [epbB, hm]
      rw [this]; exact block_ok _ (by simp)
  · -- [128 / epbB, 256]
    by_cases hm : d.mixed
    · have : (128 : Nat) / epbB d.mixed = 64 := by simp [epbB, hm]
      rw [this]; exact block_ok _ (by simp)
    · have : (128 : Nat) / epbB d.mixed = 128 := by simp [epbB, hm]
      rw [this]; exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)
  · exact block_ok _ (by simp)

/-! ## The lanes of a program's tile of `c` -/

/-- The offsets a program passes to `cDesc`: rows `pid_m · 128`, columns `pid_n · 256`. -/
def laneOffs (d : Inst) (i : Fin (launch d).size) : List Int :=
  [(((i : Nat) % gridM d) * 128 : Nat), (((i : Nat) / gridM d) * 256 : Nat)]

/-- The descriptor coordinates of lane `l` of program `i`'s `c` block. -/
def laneCoord (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) (k : Nat) : Int :=
  TensorDesc.coord (laneOffs d i) l k

/-- The mask of lane `l` of program `i`'s `c` block: the lane lies inside `c`. -/
def laneMask (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) : Bool :=
  ((cDesc d).maskAt (laneCoord d i)).get l

/-- Lane `l` of the tile is in bounds exactly when its row is below `M` and its column below `N`. -/
theorem laneMask_eq (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) :
    laneMask d i l =
      decide (0 ≤ laneCoord d i l 0 ∧ laneCoord d i l 0 < d.M ∧
        0 ≤ laneCoord d i l 1 ∧ laneCoord d i l 1 < d.N) := by
  simp only [laneMask, TensorDesc.maskAt, TensorDesc.inbAt, TensorDesc.rank, cDesc]
  congr 1
  apply propext
  constructor
  · intro h
    have h0 := h 0 (by simp)
    have h1 := h 1 (by simp)
    simpa using ⟨h0.1, h0.2, h1.1, h1.2⟩
  · rintro ⟨a, b, c, e⟩ k hk
    simp only [List.length_cons, List.length_nil] at hk
    interval_cases k
    · exact ⟨a, by simpa using b⟩
    · exact ⟨c, by simpa using e⟩

/-- The row of `c` lane `l` of program `i` lands on. -/
def laneRow (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) : Nat :=
  ((i : Nat) % gridM d) * 128 + l.coord 0

/-- The column of `c` lane `l` of program `i` lands on. -/
def laneCol (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) : Nat :=
  ((i : Nat) / gridM d) * 256 + l.coord 1

/-- The element of `c` lane `l` of program `i` lands on. -/
def laneIdx (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) : Nat :=
  laneRow d i l * d.N + laneCol d i l

/-- Coordinate 0 of lane `l` of the tile is its row. -/
theorem laneCoord_zero (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) :
    laneCoord d i l 0 = ((laneRow d i l : Nat) : Int) := by
  simp [laneCoord, TensorDesc.coord, laneOffs, laneRow]

/-- Coordinate 1 of lane `l` of the tile is its column. -/
theorem laneCoord_one (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) :
    laneCoord d i l 1 = ((laneCol d i l : Nat) : Int) := by
  simp [laneCoord, TensorDesc.coord, laneOffs, laneCol]

/-- The element of `c` that lane `l` addresses is `laneIdx d i l`. -/
theorem elemAt_cDesc (d : Inst) (i : Fin (launch d).size) (l : Idx [128, 256]) :
    (cDesc d).elemAt (laneCoord d i l) = ((laneIdx d i l : Nat) : Int) := by
  simp only [TensorDesc.elemAt, TensorDesc.rank, cDesc, List.length_cons, List.length_nil,
    List.range_succ, List.range_zero, List.map_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, List.sum_append, List.nil_append, TensorDesc.coord, laneCoord,
    laneOffs, laneIdx, laneRow, laneCol, List.getD_cons_zero, List.getD_cons_succ]
  push_cast
  ring

/-- The set of elements of `c` program `i` writes. -/
def cImg (d : Inst) (i : Fin (launch d).size) : Set Nat :=
  Buf.img (fun l : Idx [128, 256] => laneIdx d i l) (laneMask d i)

/-- The read footprint of a program: every input element. -/
def R (d : Inst) (_i : Fin (launch d).size) : Set Int := inEl d

/-- The write footprint of a program: its tile of `c`. -/
def W (d : Inst) (i : Fin (launch d).size) : Set Int := d.c.elems (cImg d i)

/-- What a program leaves behind: every in-bounds lane of its `c` tile is an `OutSpec`
element. -/
def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  ∀ l : Idx [128, 256], laneMask d i l = true → OutSpec d init m (laneRow d i l) (laneCol d i l)

end Claims.BlockScaledMxfp8
