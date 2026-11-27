import Claim
import TVA.Proof

/-!
# `compute_position_i64`: shared definitions

The launch, the running sums of the lengths, the footprints of one program of the grid, and the
postcondition of one program.
-/

namespace Claims.ComputePosition

open TVA

/-! ## The launch -/

/-- The grid has one program per request. -/
theorem launch_size (d : Inst) : (launch d).size = d.bs := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid: fewer than `2^31` programs, block shape `[512]`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨9, by norm_num⟩
  · norm_num

/-! ## The running sums -/

/-- `start (b + 1) = start b + len b`. -/
theorem start_succ (d : Inst) (m : Mem) (b : Nat) :
    start d m (b + 1) = start d m b + len d m b := by
  simp only [start, Finset.sum_range_succ]

theorem start_zero (d : Inst) (m : Mem) : start d m 0 = 0 := by simp [start]

/-- The lengths of the batch are nonnegative, so the running sums grow. -/
theorem start_mono {d : Inst} {init : Mem} (hpre : Pre d init) {a c : Nat} (hac : a ≤ c)
    (hc : c ≤ d.bs) : start d init a ≤ start d init c := by
  induction c, hac using Nat.le_induction with
  | base => exact le_refl _
  | succ c hac ih =>
    rw [start_succ]
    have h0 := (hpre.2.2.2.2.1 c (by omega)).1
    have := ih (by omega)
    linarith

theorem start_nonneg {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b ≤ d.bs) :
    0 ≤ start d init b := by
  have := start_mono hpre (Nat.zero_le b) hb
  rwa [start_zero] at this

/-- Request `b`'s positions end before those of any later request start. -/
theorem start_succ_le {d : Inst} {init : Mem} (hpre : Pre d init) {b c : Nat} (hbc : b < c)
    (hc : c ≤ d.bs) : start d init b + len d init b ≤ start d init c := by
  rw [← start_succ]
  exact start_mono hpre hbc hc

/-- Every request's positions lie in `positions`. -/
theorem start_len_le {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.bs) :
    start d init b + len d init b ≤ (d.positions.len : Int) :=
  le_trans (start_succ_le hpre hb (le_refl _)) hpre.2.2.2.2.2

/-! ## The footprints -/

/-- The elements of `positions` program `b` writes: its request's positions. -/
def posElems (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  {j | ∃ p : Nat, (p : Int) < len d init b ∧ (j : Int) = start d init b + (p : Int)}

/-- The read footprint of program `b`: the lengths `0 … b`, and its `extend_prefix_lens[b]`. -/
def R (d : Inst) (b : Nat) : Set Int :=
  d.extend_seq_lens.elems {i | i ≤ b} ∪
    (if d.hasPrefix = true then d.extend_prefix_lens.elems {b} else ∅)

/-- The write footprint of program `b`: its positions and its start. -/
def W (d : Inst) (init : Mem) (b : Nat) : Set Int :=
  d.positions.elems (posElems d init b) ∪ d.extend_start_loc.elems {b}

/-- What program `b`'s positions hold in `m`, for its positions `j < len` with `j < n`. -/
def QPos (d : Inst) (init : Mem) (b : Nat) (n : Int) (m : Mem) : Prop :=
  ∀ j : Nat, (j : Int) < len d init b → (j : Int) < n → ∃ v : I64,
    m.read .i64 (d.positions.ptr (start d init b + (j : Int))).addr = some v ∧
    v.toInt = pre d init b + (j : Int)

/-- What program `b` leaves: its positions and its start. -/
def Q (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  QPos d init b (len d init b) m ∧
    ∃ v : I32, m.read .i32 (d.extend_start_loc.ptr (b : Int)).addr = some v ∧
      v.toInt = start d init b

/-- The elements of `positions` trip `t` of program `b` writes. -/
def tripElems (d : Inst) (init : Mem) (b t : Nat) : Set Nat :=
  {j | ∃ l : Nat, l < 512 ∧ (t : Int) * 512 + (l : Int) < len d init b ∧
    (j : Int) = start d init b + ((t : Int) * 512 + (l : Int))}

theorem tripElems_sub (d : Inst) (init : Mem) (b t : Nat) :
    tripElems d init b t ⊆ posElems d init b := by
  rintro j ⟨l, -, hlt, hj⟩
  refine ⟨t * 512 + l, by push_cast; exact hlt, ?_⟩
  push_cast
  exact hj

/-- Program `b`'s positions are below the sum of all lengths, inside `positions`. -/
theorem posElems_lt {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.bs)
    {j : Nat} (hj : j ∈ posElems d init b) : (j : Int) < start d init d.bs := by
  obtain ⟨p, hp, hjp⟩ := hj
  have := start_succ_le hpre hb (le_refl _)
  linarith

theorem posElems_len {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.bs) :
    ∀ j ∈ posElems d init b, j < d.positions.len := by
  intro j hj
  have h1 := posElems_lt hpre hb hj
  have h2 := hpre.2.2.2.2.2
  have : (j : Int) < (d.positions.len : Int) := by linarith
  exact_mod_cast this

/-! ## Pointers and values -/

/-- `b.ptr 0 + k` for an integer scalar `k` of value `x` is `b.ptr x`. -/
theorem ptr_zero_add {dt : DType} {sg : Bool} {w : Nat} (b : Buf dt) (k : IntV sg w) {x : Int}
    (hk : k.toInt = x) : b.ptr 0 + k = b.ptr x := by
  rw [Ptr.add_int, Buf.ptr_offset, hk, zero_add]

/-- The value an element of a valid `int32` buffer holds is the one `valOf` reads. -/
theorem valOf_of_read {m : Mem} {b : Buf .i32} {i : Nat} {v : I32}
    (h : m.read .i32 (b.ptr (i : Int)).addr = some v) : valOf m b i = v.toInt := by
  simp only [valOf, h]

end Claims.ComputePosition
