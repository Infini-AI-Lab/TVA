import Claim
import TVA.Proof

/-!
# `prepare_pos_seq_lens_i32`: shared definitions

The launch, the footprints of each program, its postcondition, and small integer and pointer
facts. Program `r < N` serves request `r`; program `N` (the last) zeroes `seq_lens[N .. M)`.
-/

namespace Claims.PreparePosSeqLens

open TVA

/-! ## The launch -/

/-- The grid has `N + 1` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.N + 1 := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.N + 1 := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- Every program sees `num_programs(0) = N + 1`. -/
theorem launch_grid (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).grid 0 = d.N + 1 := by
  simp [Launch.ctx, launch, Launch.grid1]

/-- The launch is valid: fewer than `2^31` programs, block shape `[1024]`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨10, by norm_num⟩
  · norm_num

/-! ## Pointers and constants -/

/-- `b.ptr 0 + k` for an `int32` `k` of natural value is `b.ptr k`. -/
theorem ptr_zero_add {d : DType} (b : Buf d) (k : Nat) (hk : k < 2 ^ 31) :
    (b.ptr 0 + (IntV.ofNat k : I32)).addr = (b.ptr (k : Int)).addr := by
  simp [Buf.ptr, Ptr.offset, IntV.I32.toInt_ofNat hk]

/-- `b.ptr 0 + k` for a scalar integer `k` is `b.ptr k.toInt`. -/
theorem ptr_zero_addI {d : DType} {sg : Bool} {w : Nat} (b : Buf d) (k : IntV sg w) :
    b.ptr 0 + k = b.ptr k.toInt := by
  simp only [Ptr.add_int, Buf.ptr_offset, zero_add]

theorem toInt_0 : (IntV.ofInt 0 : I32).toInt = 0 :=
  IntV.toInt_ofInt (w := 32) (by norm_num) (IntV.I32.sfits (by norm_num) (by norm_num))

theorem toInt_1 : (1 : I32).toInt = 1 := by decide

theorem toInt_1024 : (IntV.ofNat 1024 : I32).toInt = 1024 := IntV.I32.toInt_ofNat (by norm_num)

theorem maxInt32 : IntV.maxInt true 32 = 2 ^ 31 - 1 := by norm_num [IntV.maxInt]

/-- `num_programs(0) - 1` is `N`. -/
theorem toInt_numReqs {N : Nat} (h : N + 1 < 2 ^ 31) :
    ((IntV.ofNat (N + 1) : I32) - (1 : I32)).toInt = (N : Int) := by
  have := IntV.I32.sub_nat (x := N + 1) (y := 1) (IntV.I32.toInt_ofNat h) toInt_1
    (by omega) (by norm_num; omega)
  simpa using this

/-! ## The footprints -/

/-- The elements of `pos` request `r` writes: its tokens. -/
def posElems (d : Inst) (init : Mem) (r : Nat) : Set Nat :=
  {k | qsl d init r ≤ (k : Int) ∧ (k : Int) < qsl d init (r + 1)}

/-- The elements of `seq_lens` the last program writes: `N ≤ j < M`. -/
def padElems (d : Inst) : Set Nat := {j | d.N ≤ j ∧ j < d.M}

/-- The read footprint of program `i`: for a request, its entry of `idx_mapping`, its two entries
of `query_start_loc` and its slot of `num_computed_tokens`; nothing for the last program. -/
def R (d : Inst) (init : Mem) (i : Nat) : Set Int :=
  if i < d.N then
    d.idx_mapping.elems {i} ∪ d.query_start_loc.elems {i, i + 1} ∪
      d.num_computed_tokens.elems {j | (j : Int) = slot d init i}
  else ∅

/-- The write footprint of program `i`: for a request, its entry of `seq_lens` and its tokens of
`pos`; for the last program, `seq_lens[N .. M)`. -/
def W (d : Inst) (init : Mem) (i : Nat) : Set Int :=
  if i < d.N then d.seq_lens.elems {i} ∪ d.pos.elems (posElems d init i)
  else d.seq_lens.elems (padElems d)

/-- What request `r`'s program leaves: `seq_lens[r]` and its tokens of `pos`. -/
def QReq (d : Inst) (init : Mem) (r : Nat) (m : Mem) : Prop :=
  (∃ v : I32, m.read .i32 (d.seq_lens.ptr (r : Int)).addr = some v ∧
      v.toInt = computed d init r + qlen d init r) ∧
    ∀ j : Nat, (j : Int) < qlen d init r → ∃ v : I64,
      m.read .i64 (d.pos.ptr (qsl d init r + (j : Int))).addr = some v ∧
      v.toInt = computed d init r + (j : Int)

/-- What the last program leaves: `seq_lens[N .. M)` hold `0`. -/
def QPad (d : Inst) (m : Mem) : Prop :=
  ∀ r : Nat, d.N ≤ r → r < d.M → ∃ v : I32,
    m.read .i32 (d.seq_lens.ptr (r : Int)).addr = some v ∧ v.toInt = 0

/-- The postcondition of program `i`. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  if i < d.N then QReq d init i m else QPad d m

end Claims.PreparePosSeqLens
