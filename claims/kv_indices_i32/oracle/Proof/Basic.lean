import Claim
import TVA.Proof

/-!
# `kv_indices_i32`: shared definitions

The launch, the footprints of one program of the grid, and the postcondition of one program.
-/

namespace Claims.KvIndices

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

/-! ## The footprints -/

/-- The elements of `kv_indices` program `b` writes: its request's positions. -/
def outElems (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  {j | ∃ p : Nat, (p : Int) < slen d init b ∧ (j : Int) = ind d init b + (p : Int)}

/-- The elements of `req_to_token` program `b` reads: its request's window. -/
def tabElems (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  {j | ∃ p : Nat, (p : Int) < slen d init b ∧
    (j : Int) = rpi d init b * (d.max_context_len : Int) + sidx d init b + (p : Int)}

/-- The read footprint of program `b`. -/
def R (d : Inst) (init : Mem) (b : Nat) : Set Int :=
  d.req_pool_indices.elems {b} ∪ d.kv_indptr.elems {b} ∪ d.seq_lens.elems {b} ∪
    (if d.hasStart = true then d.kv_start_idx.elems {b} else ∅) ∪
    d.req_to_token.elems (tabElems d init b)

/-- The write footprint of program `b`. -/
def W (d : Inst) (init : Mem) (b : Nat) : Set Int :=
  d.kv_indices.elems (outElems d init b)

/-- What program `b` leaves: each of its elements of `kv_indices` holds the value the table had. -/
def Q (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  ∀ j : Nat, (j : Int) < slen d init b →
    m.read .i32 (d.kv_indices.ptr (ind d init b + (j : Int))).addr =
      init.read .i32
        (d.req_to_token.ptr
          (rpi d init b * (d.max_context_len : Int) + sidx d init b + (j : Int))).addr

/-- The part of `Q` covered by the first `t` trips of the loop. -/
def Qpart (d : Inst) (init : Mem) (b : Nat) (t : Nat) (m : Mem) : Prop :=
  ∀ j : Nat, (j : Int) < slen d init b → (j : Int) < (t : Int) * 512 →
    m.read .i32 (d.kv_indices.ptr (ind d init b + (j : Int))).addr =
      init.read .i32
        (d.req_to_token.ptr
          (rpi d init b * (d.max_context_len : Int) + sidx d init b + (j : Int))).addr

/-- The elements of `kv_indices` trip `t` writes. -/
def tripElems (d : Inst) (init : Mem) (b : Nat) (t : Nat) : Set Nat :=
  {j | ∃ l : Nat, l < 512 ∧ (t : Int) * 512 + (l : Int) < slen d init b ∧
    (j : Int) = ind d init b + (t : Int) * 512 + (l : Int)}

end Claims.KvIndices
