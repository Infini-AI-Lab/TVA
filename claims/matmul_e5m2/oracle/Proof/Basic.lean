import TVA.Proof
import Claim

/-!
# The launch

The grid, the program-id of each program, the valid launch, and the number of trips of the
`K`-loop.
-/

namespace Claims.MatmulE5m2

open TVA

/-- The number of programs of the launch: `cdiv(M, 128) · cdiv(N, 256)`. -/
abbrev grid (d : Inst) : Nat := (d.M + 128 - 1) / 128 * ((d.N + 256 - 1) / 256)

/-- `steps` is the number of trips when `64` divides `K`. -/
theorem steps_eq {K : Nat} (h : 64 ∣ K) : steps K = K / 64 := by
  obtain ⟨m, rfl⟩ := h
  unfold steps
  omega

/-- The launch has `grid d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = grid d := Launch.size_grid1 _ _ _

/-- `grid` with the `cdiv`s written out. -/
theorem grid_eq (d : Inst) : grid d = (d.M + 127) / 128 * ((d.N + 255) / 256) := by
  unfold grid
  rw [show d.M + 128 - 1 = d.M + 127 by omega, show d.N + 256 - 1 = d.N + 255 by omega]

/-- Program `i` of the launch has program id `i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < (d.M + 127) / 128 * ((d.N + 255) / 256) := by
    rw [← grid_eq, ← launch_size]
    exact i.isLt
  simp only [Launch.ctx, launch, Launch.grid1]
  simp [Nat.mod_eq_of_lt hi]

/-- On covered instances the grid is below `2^31` programs. -/
theorem grid_lt {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := by
  unfold grid
  rw [show d.M + 128 - 1 = d.M + 127 by omega, show d.N + 256 - 1 = d.N + 255 by omega]
  exact hd.grid_lt

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk128 : Shape.TritonOk [128] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  subst hk
  exact ⟨7, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk256 : Shape.TritonOk [256] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  subst hk
  exact ⟨8, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk64 : Shape.TritonOk [64] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  subst hk
  exact ⟨6, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk128_1 : Shape.TritonOk [128, 1] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨7, by norm_num⟩
  · exact ⟨0, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk1_64 : Shape.TritonOk [1, 64] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨0, by norm_num⟩
  · exact ⟨6, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk128_64 : Shape.TritonOk [128, 64] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨7, by norm_num⟩
  · exact ⟨6, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk64_1 : Shape.TritonOk [64, 1] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨6, by norm_num⟩
  · exact ⟨0, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk1_256 : Shape.TritonOk [1, 256] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨0, by norm_num⟩
  · exact ⟨8, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk64_256 : Shape.TritonOk [64, 256] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨6, by norm_num⟩
  · exact ⟨8, by norm_num⟩

/-- A block shape of the kernel is accepted by Triton. -/
theorem blockOk128_256 : Shape.TritonOk [128, 256] := by
  refine ⟨?_, by decide⟩
  intro k hk
  simp at hk
  rcases hk with rfl | rfl
  · exact ⟨7, by norm_num⟩
  · exact ⟨8, by norm_num⟩

/-- Every block shape of the kernel is accepted by Triton. -/
theorem blocksOk : ∀ s ∈ ([[128], [256], [64], [128, 1], [1, 64], [128, 64], [64, 1], [1, 256], [64, 256], [128, 256]] : List Shape), Shape.TritonOk s := by
  intro s hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact blockOk128
  · exact blockOk256
  · exact blockOk64
  · exact blockOk128_1
  · exact blockOk1_64
  · exact blockOk128_64
  · exact blockOk64_1
  · exact blockOk1_256
  · exact blockOk64_256
  · exact blockOk128_256

/-- The launch is valid on covered instances. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (n := grid d) (blocks := _) (fusion := true) (grid_lt hd) blocksOk

end Claims.MatmulE5m2
