import Claim
import TVA.Proof

/-!
# Definitions for `merge_attn_states_bf16`

The grid is two dimensional, `T × H`; program `i` handles token `i % T` and head
`i / T % H`. Each program reads a few scalar lse elements and two rows of `head_size`
`bfloat16` elements, and writes one row of `output` (and one lse element when the merged
lse is passed).
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## The grid -/

/-- The launch has `T · H` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.T * d.H := by
  simp [Launch.size, launch]

/-- The token of program `i`. -/
def pidT (d : Inst) (i : Fin (launch d).size) : Nat := (i : Nat) % d.T

/-- The head of program `i`. -/
def pidH (d : Inst) (i : Fin (launch d).size) : Nat := (i : Nat) / d.T % d.H

/-- Program `i`'s first grid coordinate is `pidT`. -/
theorem launch_pidT (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = pidT d i := by
  simp [Launch.ctx, launch, pidT]

/-- Program `i`'s second grid coordinate is `pidH`. -/
theorem launch_pidH (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 1 = pidH d i := by
  simp [Launch.ctx, launch, pidH]

/-- The grid's second dimension is `H`. -/
theorem launch_grid1 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).grid 1 = d.H := by
  simp [Launch.ctx, launch]

/-- A launch with a program has `T > 0`. -/
theorem T_pos {d : Inst} (i : Fin (launch d).size) : 0 < d.T := by
  have hlt : (i : Nat) < d.T * d.H := by
    rw [← launch_size d]; exact i.isLt
  rcases Nat.eq_zero_or_pos d.T with h | h
  · rw [h, Nat.zero_mul] at hlt
    exact absurd hlt (Nat.not_lt_zero _)
  · exact h

/-- A launch with a program has `H > 0`. -/
theorem H_pos {d : Inst} (i : Fin (launch d).size) : 0 < d.H := by
  have hlt : (i : Nat) < d.T * d.H := by
    rw [← launch_size d]; exact i.isLt
  rcases Nat.eq_zero_or_pos d.H with h | h
  · rw [h, Nat.mul_zero] at hlt
    exact absurd hlt (Nat.not_lt_zero _)
  · exact h

/-- The token of a program is below `T`. -/
theorem pidT_lt (d : Inst) (i : Fin (launch d).size) : pidT d i < d.T :=
  Nat.mod_lt _ (T_pos i)

/-- The head of a program is below `H`. -/
theorem pidH_lt (d : Inst) (i : Fin (launch d).size) : pidH d i < d.H :=
  Nat.mod_lt _ (H_pos i)

/-- Program `i` is determined by its token and head. -/
theorem pid_pair {d : Inst} {i j : Fin (launch d).size}
    (h1 : pidT d i = pidT d j) (h2 : pidH d i = pidH d j) : i = j := by
  have hi : (i : Nat) < d.T * d.H := by rw [← launch_size d]; exact i.isLt
  have hj : (j : Nat) < d.T * d.H := by rw [← launch_size d]; exact j.isLt
  have hT := T_pos i
  have hTi : (i : Nat) / d.T < d.H := by
    rw [Nat.div_lt_iff_lt_mul hT, Nat.mul_comm]; exact hi
  have hTj : (j : Nat) / d.T < d.H := by
    rw [Nat.div_lt_iff_lt_mul hT, Nat.mul_comm]; exact hj
  have h2' : (i : Nat) / d.T = (j : Nat) / d.T := by
    have h := h2
    simp only [pidH, Nat.mod_eq_of_lt hTi, Nat.mod_eq_of_lt hTj] at h
    exact h
  have h1' : (i : Nat) % d.T = (j : Nat) % d.T := h1
  apply Fin.ext
  calc (i : Nat) = d.T * ((i : Nat) / d.T) + (i : Nat) % d.T := (Nat.div_add_mod _ _).symm
    _ = d.T * ((j : Nat) / d.T) + (j : Nat) % d.T := by rw [h2', h1']
    _ = (j : Nat) := Nat.div_add_mod _ _

/-- The launch is valid: two-dimensional grid inside CUDA's limits, one block shape. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg2 : (1 : Nat) < 2 ^ 16 := by norm_num
  refine ⟨hd.T_lt, ?_, hg2, ?_, rfl, rfl, rfl, ?_⟩
  · simpa [launch] using hd.H_lt
  · rw [launch_size]; exact hd.grid_lt
  · intro s hs
    simp only [launch, List.mem_singleton] at hs
    subst hs
    refine ⟨?_, ?_⟩
    · intro k hk
      simp only [List.mem_singleton] at hk
      subst hk
      exact hd.padded_pow2
    · simpa using hd.padded_le

/-! ## Footprints -/

/-- The `head_size` elements of a buffer starting at element `base`. -/
def row (d : Inst) (base : Nat) : Set Nat := {j | base ≤ j ∧ j < base + d.head_size}

/-- Membership in a row. -/
theorem row_iff {d : Inst} {base j : Nat} : j ∈ row d base ↔ base ≤ j ∧ j < base + d.head_size :=
  Iff.rfl

/-- The scalar lse elements program `(t, h)` reads. The first side's only when the token is
merged (the copy branch of the kernel never touches it). -/
def Rlse (d : Inst) (t h : Nat) : Set Int :=
  (if t < d.ptwc then Mem.span .f32 (d.prefix_lse.ptr (plIdx d t h : Int)).addr else ∅) ∪
    Mem.span .f32 (d.suffix_lse.ptr (slIdx d t h : Int)).addr

/-- The `bfloat16` rows program `(t, h)` reads (the first side's only when the token is merged). -/
def Rin (d : Inst) (t h : Nat) : Set Int :=
  (if t < d.ptwc then d.prefix_output.elems (row d (poIdx d t h 0)) else ∅) ∪
    d.suffix_output.elems (row d (soIdx d t h 0))

/-- The read footprint of program `(t, h)`. -/
def Rf (d : Inst) (t h : Nat) : Set Int := Rlse d t h ∪ Rin d t h

/-- The `output` row program `(t, h)` writes. -/
def Rout (d : Inst) (t h : Nat) : Set Int := d.output.elems (row d (oIdx d t h 0))

/-- The `output_lse` element program `(t, h)` writes, if the merged lse is passed. -/
def RlseOut (d : Inst) (t h : Nat) : Set Int :=
  if d.outLse then Mem.span .f32 (d.output_lse.ptr (olIdx d t h : Int)).addr else ∅

/-- The write footprint of program `(t, h)`. -/
def Wf (d : Inst) (t h : Nat) : Set Int := Rout d t h ∪ RlseOut d t h

/-- What program `(t, h)` leaves in memory: a merged row for `t < ptwc`, a copied row after. -/
def Qm (d : Inst) (init : Mem) (t h : Nat) (m : Mem) : Prop :=
  if t < d.ptwc then MergeOk d init m t h else CopyOk d init m t h

/-! ## The per-program footprints -/

/-- The read footprint of program `i`. -/
def Rq (d : Inst) (_init : Mem) (i : Fin (launch d).size) : Set Int := Rf d (pidT d i) (pidH d i)

/-- The write footprint of program `i`. -/
def Wq (d : Inst) (_init : Mem) (i : Fin (launch d).size) : Set Int := Wf d (pidT d i) (pidH d i)

/-- What program `i` leaves in memory. -/
def Qq (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  Qm d init (pidT d i) (pidH d i) m

end Claims.MergeAttnStates
