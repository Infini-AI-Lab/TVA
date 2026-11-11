import Claim
import TVA.Proof
import Proof.Walk

/-!
# The grid

Program `i` of the launch is `(pm, h) = (i % gx, i / gx)`; the launch is valid, distinct programs
own disjoint slots, and the slots of all programs cover `out` and `final_states`.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The grid has at least one block along the state dimension. -/
theorem gx_pos {d : Inst} (hd : d.ok) : 0 < gx d := by
  unfold gx
  have := hd.dim_pos
  have : 0 < 1024 := by norm_num
  have h2 : BL = 1024 := BL_eq
  rw [h2]
  omega

/-- The launch has `gx · 1 · nheads` programs. -/
theorem launch_size_eq (d : Inst) : (launch d).size = gx d * 1 * d.nheads := by
  simp [Launch.size, launch]

/-- The program ids of program `i`. -/
theorem pid_eq (d : Inst) (i : Nat) :
    ((launch d).ctx i).pid 0 = i % gx d ∧ ((launch d).ctx i).pid 1 = 0 ∧
      ((launch d).ctx i).pid 2 = i / gx d := by
  simp [Launch.ctx, launch, Nat.mod_one]

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg := hd.grid_lt
  have hp := gx_pos hd
  have hn := hd.heads_pos
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · show gx d < 2 ^ 31
    have : gx d ≤ gx d * d.nheads := Nat.le_mul_of_pos_right _ hn
    omega
  · show 1 < 2 ^ 16
    norm_num
  · show d.nheads < 2 ^ 16
    exact hd.heads_lt
  · rw [launch_size_eq]; simpa using hg
  · intro s hs
    simp only [launch, List.mem_singleton] at hs
    subst hs
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_singleton] at hk
      subst hk
      exact ⟨10, by rw [BL_eq]; norm_num⟩
    · rw [BL_eq]; norm_num

/-- The head of program `i` is below `nheads`. -/
theorem grid_h_lt {d : Inst} (i : Fin (launch d).size) : (i : Nat) / gx d < d.nheads := by
  have h : (i : Nat) < gx d * 1 * d.nheads := lt_of_lt_of_eq i.isLt (launch_size_eq d)
  rw [Nat.mul_one] at h
  exact Nat.div_lt_of_lt_mul h

/-- Distinct programs write disjoint bytes. -/
theorem Wp_disjoint {d : Inst} (hd : d.ok) {h h' pm pm' : Nat} (hh : h < d.nheads)
    (hh' : h' < d.nheads) (hne : (h, pm) ≠ (h', pm')) : Disjoint (Wp d h pm) (Wp d h' pm') := by
  rw [Set.disjoint_left]
  rintro x ⟨c, hc, hx⟩ ⟨c', hc', hx'⟩
  exact Set.disjoint_left.mp (slotW_prog_disjoint hd hh hh' (by omega) (by omega) hne) hx hx'

/-- A program writes no input. -/
theorem Wp_disjoint_Rin {d : Inst} (hd : d.ok) {h pm : Nat} (hh : h < d.nheads) :
    Disjoint (Wp d h pm) (Rin d) := by
  rw [Set.disjoint_left]
  rintro x ⟨c, hc, hx⟩
  exact Set.disjoint_left.mp (slotW_disjoint_Rin hd hh (by omega)) hx

/-- Everything a program writes is in `out` or `final_states`. -/
theorem Wp_sub {d : Inst} {h pm : Nat} (hh : h < d.nheads) :
    Wp d h pm ⊆ d.out.elems {j | j < d.nchunks * (d.nheads * d.dim)} ∪
      d.finalst.elems {j | j < d.nheads * d.dim} := by
  rintro x ⟨c, hc, hx⟩
  replace hx : x ∈ slotW d h pm c := hx
  by_cases h1 : c < d.nchunks
  · rw [slotW_out h1] at hx
    refine Or.inl (d.out.elems_mono ?_ hx)
    rintro j ⟨m, ⟨-, -, hm⟩, rfl⟩
    exact stIdx_lt h1 hh hm
  · rw [show c = d.nchunks by omega, slotW_fin] at hx
    refine Or.inr (d.finalst.elems_mono ?_ hx)
    rintro j ⟨m, ⟨-, -, hm⟩, rfl⟩
    exact finIdx_lt hh hm

end Claims.SsdStatePassing
