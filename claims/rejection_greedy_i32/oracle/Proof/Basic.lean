import Claim
import TVA.Proof

/-!
# `rejection_greedy_i32`: the program of one request

This file fixes the body of one program of the launch, split into the parts the proof works on,
and the footprints of the programs.

Everything here is stated about an instance `d : Inst`, an initial memory `init` and a request
`r < d.B`.
-/

namespace Claims.RejectionGreedy

open TVA

/-! ## The body of program `r` -/

/-- The `int32` offset of row `r` of the output, `req_idx * (max_spec_len + 1)`. -/
def rowOff (d : Inst) (r : Nat) : I32 := (IntV.ofNat r : I32) * (IntV.ofNat d.S + (1 : I32))

/-- One trip of the loop at position `pos`: while `rejected` is `false`, load the draft and the
target there, convert the target to `int32`, store it at column `pos` of row `r`, and report
whether the draft differs. -/
def tripBody (d : Inst) (r : Nat) (start_idx : I32) (pos : I32) (rejected : Bool) : Prog Bool := do
  if !rejected then
    let draft_token_id ← tl.load1 (d := .i32) (d.draft.ptr 0 + start_idx + pos)
    let target64 ← tl.load1 (d := .i64) (d.target.ptr 0 + start_idx + pos)
    let target_argmax_id : I32 := (← tl.convert .i64 .i32 (Tensor.scalar target64)).item
    let token_id := target_argmax_id
    let rejected := !(draft_token_id.eq target_argmax_id)
    tl.store1 (d.out.ptr 0 + rowOff d r + pos) token_id
    pure rejected
  else
    pure rejected

/-- The last step of a greedy request: load `bonus[r]` and store it at column `num`. -/
def bonusStep (d : Inst) (r : Nat) (num : I32) : Prog Unit := do
  let bonus_token_id ← tl.load1 (d := .i32) (d.bonus.ptr 0 + (IntV.ofNat r : I32))
  tl.store1 (d.out.ptr 0 + rowOff d r + num) bonus_token_id

/-- The part of program `r`'s body after `start_idx` is known, when the request is greedy. -/
def afterStart (d : Inst) (r : Nat) (start_idx : I32) : Prog Unit := do
  let end_idx ← tl.load1 (d := .i32) (d.cu.ptr 0 + (IntV.ofNat r : I32))
  let num_draft_tokens := end_idx - start_idx
  let rejected : Bool := false
  let rejected ←
    tl.forRange (0 : I32) num_draft_tokens (1 : I32) rejected (tripBody d r start_idx)
  if !rejected then bonusStep d r num_draft_tokens else pure ()

/-- The part of program `r`'s body that runs when the request is greedy. -/
def rest (d : Inst) (r : Nat) : Prog Unit := do
  let req_idx := (IntV.ofNat r : I32)
  let start_idx : I32 ←
    if req_idx.eq (0 : I32) then
      pure (tl.zeros [] .i32).item
    else
      tl.load1 (d := .i32) (d.cu.ptr 0 + req_idx - (1 : I32))
  afterStart d r start_idx

/-- Program `r` of the launch: the model's body at `req_idx = r`. -/
def prog (d : Inst) (r : Nat) : Prog Unit := do
  let is_greedy : Bool ←
    match d.greedy.map (·.ptr 0) with
    | none => pure true
    | some p => tl.load1 (d := .i1) (p + (IntV.ofNat r : I32))
  if !is_greedy then return () else rest d r

/-- The model's body at program `i` of the launch is `prog d i`. -/
theorem body_eq (d : Inst) (c : Ctx) : (model d).body c = prog d (c.pid 0) := rfl

/-- The grid has `B` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.B := by
  simp [launch, Launch.size_grid1]

/-- Program `i` of the launch has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch has `debug = false`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.hgrid fun s hs => ?_
  simp only [List.not_mem_nil] at hs

/-! ## Rows and footprints -/

/-- The columns of request `r`: `r * (S + 1), …, r * (S + 1) + S`. -/
def rowSet (d : Inst) (r : Nat) : Set Nat := {j | r * (d.S + 1) ≤ j ∧ j < (r + 1) * (d.S + 1)}

/-- The bytes of row `r` of the output: the write footprint of program `r`. -/
def Wf (d : Inst) (r : Nat) : Set Int := d.out.elems (rowSet d r)

/-- The bytes of column `t` of row `r`: what one trip of the loop writes. -/
def Wt (d : Inst) (r t : Nat) : Set Int := Mem.span .i32 (d.out.ptr (oidx d r t)).addr

/-- Everything a program may read. -/
def Rf (d : Inst) : Set Int :=
  d.cu.bytes ∪ d.draft.bytes ∪ d.target.bytes ∪ d.bonus.bytes ∪
    (match d.greedy with | some g => g.bytes | none => ∅)

/-- What program `r` leaves in the memory it touched: `expect`'s row. -/
def Qf (d : Inst) (init : Mem) (r : Nat) (m : Mem) : Prop :=
  ∀ j, j < d.S + 1 →
    match expect d init r j with
    | some v => ∃ w : I32, m.read .i32 (d.out.ptr (oidx d r j)).addr = some w ∧ w.toInt = v
    | none =>
      m.read .i32 (d.out.ptr (oidx d r j)).addr = init.read .i32 (d.out.ptr (oidx d r j)).addr

/-- Column `j` of row `r` is in the row's element set. -/
theorem oidx_mem_rowSet {d : Inst} {r j : Nat} (hj : j < d.S + 1) : oidx d r j ∈ rowSet d r := by
  refine ⟨Nat.le_add_right _ _, ?_⟩
  have : r * (d.S + 1) + j < r * (d.S + 1) + (d.S + 1) := by omega
  simpa [oidx, Nat.succ_mul, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using this

/-- A row of the batch lies in the output's `B * (S + 1)` elements. -/
theorem rowSet_sub {d : Inst} {r : Nat} (hr : r < d.B) : rowSet d r ⊆ {j | j < d.B * (d.S + 1)} := by
  intro j hj
  obtain ⟨-, hj2⟩ := hj
  have h1 : r + 1 ≤ d.B := by omega
  exact lt_of_lt_of_le hj2 (Nat.mul_le_mul_right _ h1)

/-- Different rows are disjoint. -/
theorem rowSet_disjoint {d : Inst} {r r' : Nat} (h : r ≠ r') : Disjoint (rowSet d r) (rowSet d r') := by
  rcases Nat.lt_or_gt_of_ne h with hlt | hlt
  · refine Set.disjoint_left.2 fun j h1 h2 => ?_
    obtain ⟨-, hj1⟩ := h1
    obtain ⟨hj2, -⟩ := h2
    have h3 : (r + 1) * (d.S + 1) ≤ r' * (d.S + 1) := Nat.mul_le_mul_right _ (by omega)
    exact (Nat.not_lt.2 (le_trans h3 hj2)) hj1
  · refine Set.disjoint_left.2 fun j h1 h2 => ?_
    obtain ⟨hj2, -⟩ := h1
    obtain ⟨-, hj1⟩ := h2
    have h3 : (r' + 1) * (d.S + 1) ≤ r * (d.S + 1) := Nat.mul_le_mul_right _ (by omega)
    exact (Nat.not_lt.2 (le_trans h3 hj2)) hj1

/-- Different programs write disjoint bytes. -/
theorem Wf_disjoint {d : Inst} {r r' : Nat} (h : r ≠ r') : Disjoint (Wf d r) (Wf d r') :=
  d.out.elems_disjoint (rowSet_disjoint h)

/-- A trip's column is inside its row. -/
theorem Wt_sub_Wf {d : Inst} {r t : Nat} (ht : t < d.S + 1) : Wt d r t ⊆ Wf d r := by
  intro x hx
  exact d.out.span_sub_elems (oidx_mem_rowSet ht) hx

/-- Different columns of one row share no byte (they are different elements of the output). -/
theorem Wt_disjoint {d : Inst} {r t u : Nat} (h : t ≠ u) : Disjoint (Wt d r t) (Wt d r u) :=
  d.out.span_disjoint (by
    intro hc
    apply h
    simp only [oidx] at hc
    omega)

end Claims.RejectionGreedy
