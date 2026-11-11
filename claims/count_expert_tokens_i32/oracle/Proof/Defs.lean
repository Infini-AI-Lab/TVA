import Claim
import TVA.Proof

/-!
# Definitions and arithmetic shared by the proof of `count_expert_tokens_i32`

`numTrips` is the model's `tl.cdiv(numel, B)`; `cnt d e t l` is the number of positions
`x * B + l` (`x < t`) that a program's lane `l` has counted for expert `e` after `t` trips;
`Rin`/`Wout` are a program's read and write footprints; `Qout` is what a program leaves in
memory.
-/

namespace Claims.CountExpertTokens

open TVA

/-- The number of trips of the loop, `cdiv(numel, B)`. -/
def numTrips (d : Inst) : Nat := (d.numel + d.B - 1) / d.B

/-- Positions `x * B + l` for `x < t` that count toward expert `e`, at lane `l`. -/
def cnt (d : Inst) (e t l : Nat) : Nat :=
  ((Finset.range t).filter
    (fun x => x * d.B + l < d.numel ∧ routed d (x * d.B + l) = (e : Int))).card

/-- The bytes any program of the grid may read. -/
def Rin (d : Inst) : Set Int :=
  d.ids.elems {j | j < d.numel} ∪
    (if d.hasMap = true then d.emap.elems {j | j < d.numGlobal} else ∅)

/-- The bytes the program of expert `e` may write: element `e` of `out`. -/
def Wout (d : Inst) (e : Nat) : Set Int := d.out.elems {e}

/-- What the program of expert `e` leaves in memory: element `e` of `out` holds the count of
`e`, and nothing outside it changed. -/
def Qout (d : Inst) (init : Mem) (e : Nat) (m : Mem) : Prop :=
  ∃ c : I32, c.toInt = (countOf d e : Int) ∧
    m.read .i32 (d.out.ptr e).addr = some c ∧
    (∀ x, x ∉ Mem.span .i32 (d.out.ptr e).addr → m x = init x)

/-! ## Bounds from `Inst.ok` -/

/-- `BLOCK_SIZE` is positive. -/
theorem B_pos {d : Inst} (hd : d.ok) : 0 < d.B := by
  obtain ⟨k, hk⟩ := hd.b_pow2
  rw [hk]
  exact Nat.two_pow_pos k

/-- `BLOCK_SIZE` fits in an `int32`. -/
theorem B_lt {d : Inst} (hd : d.ok) : d.B < 2 ^ 31 := by
  have h := hd.b_le
  omega

/-- `numel` fits in an `int32`. -/
theorem numel_lt {d : Inst} (hd : d.ok) : d.numel < 2 ^ 31 := by
  have h1 := hd.numel_b
  have h2 := B_pos hd
  omega

/-- `cdiv(numel, B) ≤ numel`. -/
theorem numTrips_le_numel {d : Inst} (hd : d.ok) : numTrips d ≤ d.numel := by
  have hB := B_pos hd
  have h1 : d.numel ≤ d.numel * d.B := Nat.le_mul_of_pos_right _ hB
  have h2 : (d.numel + 1) * d.B = d.numel * d.B + d.B := by rw [Nat.add_mul, one_mul]
  have h3 : d.numel + d.B - 1 = d.numel + (d.B - 1) := by omega
  have h : numTrips d < d.numel + 1 := by
    rw [numTrips, Nat.div_lt_iff_lt_mul hB, h3, h2]
    omega
  omega

/-- The trips cover `numel`: `T * B ≥ numel`. -/
theorem numel_le_numTrips_mul {d : Inst} (hd : d.ok) : d.numel ≤ numTrips d * d.B := by
  have hB := B_pos hd
  have h3 : d.numel + d.B - 1 = d.numel + (d.B - 1) := by omega
  have h := Nat.div_add_mod' (d.numel + (d.B - 1)) d.B
  have hr := Nat.mod_lt (d.numel + (d.B - 1)) hB
  simp only [numTrips, h3]
  omega

/-- Below the trip count, `t * B` is still inside the ids. -/
theorem numTrips_pred_mul {d : Inst} (hd : d.ok) {t : Nat} (ht : t < numTrips d) :
    t * d.B < d.numel := by
  have hB := B_pos hd
  have h3 : d.numel + d.B - 1 = d.numel + (d.B - 1) := by omega
  have h1 : numTrips d * d.B ≤ d.numel + d.B - 1 := Nat.div_mul_le_self _ _
  rw [h3] at h1
  have ht1 : t + 1 ≤ numTrips d := by omega
  have h4 : (t + 1) * d.B ≤ numTrips d * d.B := Nat.mul_le_mul_right _ ht1
  rw [Nat.add_mul, one_mul] at h4
  omega

/-- The trip count fits in an `int32`. -/
theorem numTrips_lt {d : Inst} (hd : d.ok) : numTrips d < 2 ^ 31 :=
  lt_of_le_of_lt (numTrips_le_numel hd) (numel_lt hd)

/-! ## The per-lane counts -/

/-- No lane has counted anything before the first trip. -/
theorem cnt_zero (d : Inst) (e l : Nat) : cnt d e 0 l = 0 := by
  simp [cnt]

/-- Counting the filter over `range (t+1)` adds `1` when `p t` holds. -/
theorem card_filter_range_succ {p : Nat → Prop} [DecidablePred p] (t : Nat) :
    ((Finset.range (t + 1)).filter p).card =
      ((Finset.range t).filter p).card + (if p t then 1 else 0) := by
  rw [Finset.range_add_one, Finset.filter_insert]
  by_cases hp : p t
  · rw [ite_eq_left hp, Finset.card_insert_of_notMem (by simp), ite_eq_left hp]
  · rw [ite_eq_right hp, ite_eq_right hp, Nat.add_zero]

/-- Trip `t` adds one to lane `l`'s count exactly when its position is in range and routed to `e`. -/
theorem cnt_succ (d : Inst) (e t l : Nat) :
    cnt d e (t + 1) l =
      cnt d e t l +
        (if t * d.B + l < d.numel ∧ routed d (t * d.B + l) = (e : Int) then 1 else 0) := by
  have h := card_filter_range_succ
    (p := fun x => x * d.B + l < d.numel ∧ routed d (x * d.B + l) = (e : Int)) t
  simpa [cnt] using h

/-- A lane counts at most one position per trip. -/
theorem cnt_le (d : Inst) (e t l : Nat) : cnt d e t l ≤ t := by
  rw [cnt]
  exact (Finset.card_filter_le _ _).trans (le_of_eq (Finset.card_range t))

/-- A lane's count stays at most `numel` up to the last trip. -/
theorem cnt_le_numel {d : Inst} (hd : d.ok) (e : Nat) {t : Nat} (ht : t ≤ numTrips d)
    (l : Nat) : cnt d e t l ≤ d.numel :=
  (cnt_le d e t l).trans (ht.trans (numTrips_le_numel hd))

/-! ## The program's body -/

/-- The body of `count_kernel` with the program id as a parameter. -/
def bodyE (d : Inst) (curr_expert : I32) : Prog Unit := do
  let topk_ids_ptrs := d.ids.ptr 0 + tl.arange 0 d.B
  let acc := tl.zeros [d.B] .i32
  let _ ← tl.idiv (Tensor.scalar ((IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1)))
    (Tensor.scalar (IntV.ofNat d.B))
  let trips ← tl.idiv (Tensor.scalar ((IntV.ofNat d.numel : I32) + IntV.ofNat (d.B - 1)))
    (Tensor.scalar (IntV.ofNat d.B))
  let (acc, _) ← tl.forRange (0 : I32) trips.item (1 : I32) (acc, topk_ids_ptrs)
    (loopBody d.dtype d.hasMap d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr_expert)
  if curr_expert.lt (IntV.ofNat d.numLocal) then
    tl.store1 (d.out.ptr 0 + curr_expert) (Tensor.foldAll (· + ·) (0 : I32) acc)
  else pure ()

/-- The grid has `numLocal` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.numLocal := Launch.size_grid1 _ _ _

/-- Program `i` of the launch has program id `i`. -/
theorem pid_eq (d : Inst) (i : Nat) (hi : i < d.numLocal) :
    tl.program_id ((launch d).ctx i) 0 = (IntV.ofNat i : I32) := by
  simp [tl.program_id, Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.numLocal_lt fun s hs => ?_
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk
    exact hd.b_pow2
  · simpa using hd.b_le

end Claims.CountExpertTokens
