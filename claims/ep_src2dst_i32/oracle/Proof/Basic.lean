import Claim
import TVA.Proof

/-!
# `ep_src2dst_i32`: shared definitions

The launch, the two parts of a program (the search for `first_nonnegative`, then the block), the
footprints of one program and its postcondition.
-/

namespace Claims.EpSrc2Dst

open TVA Block

/-! ## The launch -/

/-- The number of programs, `cdiv(n, 512)`. -/
def G (d : Inst) : Nat := (d.n + 511) / 512

theorem launch_size (d : Inst) : (launch d).size = G d := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < G d := lt_of_lt_of_eq i.isLt (launch_size d)
  simp only [Launch.ctx, launch, Launch.grid1]
  exact Nat.mod_eq_of_lt (by simpa [G] using hi)

/-- The launch is valid: fewer than `2^31` programs, block shape `[512]`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 ?_ ?_
  · have := hd.n_lt
    show (d.n + 511) / 512 < 2 ^ 31
    omega
  · intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    refine ⟨?_, ?_⟩
    · intro k hk
      simp only [List.mem_singleton] at hk
      subst hk
      exact ⟨9, by norm_num⟩
    · norm_num

/-- A program of the grid starts below `n`, and its block fits `int32`. -/
theorem prog_bounds {d : Inst} (hd : d.ok) {p : Nat} (hp : p < G d) : p * 512 < d.n ∧ Fits p 512 := by
  have h1 : p * 512 < d.n := by unfold G at hp; omega
  exact ⟨h1, fits_of_grid_pow2 (e := 9) (by norm_num) (by norm_num) hd.n_lt (by unfold G at hp; omega)⟩

/-! ## The two parts of a program -/

/-- Lines 42–51 after the load of `sorted_topk_ids[0]` (`s0`): the binary search, or `0`. -/
def searchIte (p : Ptr .i32) (num : I32) (s0 : I32) : Prog I32 :=
  if IntV.lt s0 (IntV.ofInt 0 : I32) then do
      let left : I32 := IntV.ofInt 0
      let right : I32 := num
      let (left, _right) ← tl.whileLoop (left, right) (fun st => pure (IntV.lt st.1 st.2))
        (searchStep p)
      let first_nonnegative := left
      pure first_nonnegative
    else pure (IntV.ofInt 0)

/-- Lines 53–59, with `first_nonnegative = f`. -/
def restProg (d : Inst) (c : Ctx) (f : I32) : Prog Unit := do
  let pid := tl.program_id c 0
  let dst_id := pid * (IntV.ofNat 512 : I32) + tl.arange 0 512
  let mask := dst_id.ltS (IntV.ofNat d.n)
  let src_id ← tl.load (d.reorder.ptr 0 + dst_id) (some mask)
  let packed_id := tl.where (Tensor.le (bc f) dst_id)
    (dst_id - bc f) (bc (IntV.ofInt (-1) : I32))
  tl.store (d.src2dst.ptr 0 + src_id) packed_id (some mask)

/-- The model's body, split in its parts. -/
theorem body_eq (d : Inst) (c : Ctx) : (model d).body c =
    (tl.load1 (d.sorted.ptr 0) >>= fun s0 =>
      searchIte (d.sorted.ptr 0) (IntV.ofNat d.n) s0 >>= restProg d c) := rfl

/-! ## Footprints -/

/-- The search's result `f`: the number `k ≤ n` of the leading negative ids. -/
def Found (d : Inst) (m : Mem) (f : I32) : Prop :=
  ∃ k : Nat, f.toInt = k ∧ k ≤ d.n ∧ ∀ j, j < d.n → (sid d m j < 0 ↔ j < k)

/-- The value `Post` asks at sorted position `j`. -/
def pval (d : Inst) (m : Mem) (j : Nat) : Int :=
  if sid d m j < 0 then -1 else (j : Int) - (nneg d m : Int)

/-- The elements of `src2dst` program `p` writes. -/
def outE (d : Inst) (init : Mem) (p : Nat) : Set Nat :=
  {k | ∃ j, Tile p 512 d.n j ∧ rid d init j = (k : Int)}

/-- Program `p` reads all `n` sorted ids (the search) and its block of `reorder_ids`. -/
def R (d : Inst) (p : Nat) : Set Int :=
  d.sorted.elems {j | j < d.n} ∪ d.reorder.tile p 512 d.n

/-- Program `p` writes the elements `reorder_ids[j]` of its block. -/
def W (d : Inst) (init : Mem) (p : Nat) : Set Int := d.src2dst.elems (outE d init p)

/-- What program `p` leaves: each element `reorder_ids[j]` of its block holds `pval j`. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ j, Tile p 512 d.n j →
    m.read .i32 (d.src2dst.ptr (rid d init j)).addr = some (IntV.ofInt (pval d init j))

/-! ## Reading values -/

theorem sid_of_read {d : Inst} {m : Mem} {j : Nat} {v : I32}
    (h : m.read .i32 (d.sorted.ptr (j : Int)).addr = some v) : sid d m j = v.toInt := by
  simp only [sid, val32, h]

theorem rid_of_read {d : Inst} {m : Mem} {j : Nat} {v : I64}
    (h : m.read .i64 (d.reorder.ptr (j : Int)).addr = some v) : rid d m j = v.toInt := by
  simp only [rid, val64, h]

/-- The count of negative ids is the prefix length the search finds. -/
theorem nneg_eq {d : Inst} {m : Mem} {k : Nat} (hk : k ≤ d.n)
    (h : ∀ j, j < d.n → (sid d m j < 0 ↔ j < k)) : nneg d m = k := by
  unfold nneg
  have : (Finset.range d.n).filter (fun j => sid d m j < 0) = Finset.range k := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨hj, hs⟩; exact (h j hj).mp hs
    · intro hj; exact ⟨by omega, (h j (by omega)).mpr hj⟩
  rw [this, Finset.card_range]

end Claims.EpSrc2Dst
