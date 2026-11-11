import Claim
import TVA.Proof
import Proof.Basic

/-!
# `Proof.Index`

The kernel's `int32` scalars (`pid`, `it`, `end`, `num_pid_n`, `iters_per_tile`) as natural numbers, and the decomposition of an iteration into tile and offset.
-/

/-!
# The loop indices of one program

One program of the launch runs the iterations `[lo p, hi p)`.  Iteration `it` handles the chunk
`it % T` (`T = steps K`) of the tile `it / T`.  This file records the `int32` values the kernel
computes for those indices, the two-dimensional lanes of the `a`, `b` and `c` accesses, and the
per-tile chunk ranges of a program.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## `int32` values of the loop -/

/-- The literal `127` as an `int32` is `127`. -/
theorem lit127_toInt : (IntV.ofNat 127 : I32).toInt = 127 := IntV.I32.toInt_ofNat (by norm_num)
/-- The literal `128` as an `int32` is `128`. -/
theorem lit128_toInt : (IntV.ofNat 128 : I32).toInt = 128 := IntV.I32.toInt_ofNat (by norm_num)
/-- The literal `64` as an `int32` is `64`. -/
theorem lit64_toInt : (IntV.ofNat 64 : I32).toInt = 64 := IntV.I32.toInt_ofNat (by norm_num)
/-- The literal `63` as an `int32` is `63`. -/
theorem lit63_toInt : (IntV.ofNat 63 : I32).toInt = 63 := IntV.I32.toInt_ofNat (by norm_num)
/-- The literal `0` as an `int32` is `0`. -/
theorem lit0_toInt : (IntV.ofNat 0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)
/-- The literal `1` as an `int32` is `1`. -/
theorem lit1_toInt : (IntV.ofInt 1 : I32).toInt = 1 := by
  rw [IntV.I32.toInt_ofInt]; norm_num

/-- `N < 2^31`. -/
theorem N_lt (hd : d.ok) : d.N < 2 ^ 31 := by have := hd.N_le; omega
/-- `K < 2^31`. -/
theorem K_lt (hd : d.ok) : d.K < 2 ^ 31 := by have := hd.K_le; omega
/-- `M < 2^31`. -/
theorem M_lt (hd : d.ok) : d.M < 2 ^ 31 := by have := hd.M_le; omega

/-- `N + 127` does not wrap in `int32`. -/
theorem N_add127_toInt (hd : d.ok) :
    (IntV.ofNat d.N + (IntV.ofNat 127 : I32)).toInt = ((d.N + 127 : Nat) : Int) :=
  IntV.I32.add_nat (IntV.I32.toInt_ofNat (N_lt hd)) lit127_toInt (by have := hd.N_le; omega)

/-- `K + 63` does not wrap in `int32`. -/
theorem K_add63_toInt (hd : d.ok) :
    (IntV.ofNat d.K + (IntV.ofNat 63 : I32)).toInt = ((d.K + 63 : Nat) : Int) :=
  IntV.I32.add_nat (IntV.I32.toInt_ofNat (K_lt hd)) lit63_toInt (by have := hd.K_le; omega)

/-- `N` as an `int32` has value `N`. -/
theorem N_toInt (hd : d.ok) : (IntV.ofNat d.N : I32).toInt = (d.N : Int) :=
  IntV.I32.toInt_ofNat (N_lt hd)

/-- `K` as an `int32` has value `K`. -/
theorem K_toInt (hd : d.ok) : (IntV.ofNat d.K : I32).toInt = (d.K : Int) :=
  IntV.I32.toInt_ofNat (K_lt hd)

/-- `M` as an `int32` has value `M`. -/
theorem M_toInt (hd : d.ok) : (IntV.ofNat d.M : I32).toInt = (d.M : Int) :=
  IntV.I32.toInt_ofNat (M_lt hd)

/-- The grid is nonempty. -/
theorem num_programs_pos {d : Inst} (hd : d.ok) : 0 < d.num_programs := by
  by_contra h0
  have hz : d.num_programs = 0 := Nat.eq_zero_of_not_pos h0
  have htot := total_pos hd
  have hc := hd.cover
  rw [hz, Nat.zero_mul] at hc
  omega

/-- `iters_per_program < 2^31`. -/
theorem ipp_lt {d : Inst} (hd : d.ok) : d.iters_per_program < 2 ^ 31 := by
  have h := hd.progs_lt
  have hn := num_programs_pos hd
  calc d.iters_per_program ≤ d.num_programs * d.iters_per_program := Nat.le_mul_of_pos_left _ hn
    _ < 2 ^ 31 := h

/-- `iters_per_program` as an `int32` has its value. -/
theorem ipp_toInt (hd : d.ok) :
    (IntV.ofNat d.iters_per_program : I32).toInt = (d.iters_per_program : Int) :=
  IntV.I32.toInt_ofNat (ipp_lt hd)

/-- `total_iters` as an `int32` has its value. -/
theorem total_toInt (hd : d.ok) :
    (IntV.ofNat d.total_iters : I32).toInt = (d.total_iters : Int) :=
  IntV.I32.toInt_ofNat (by have := total_lt hd; omega)

/-- `pid` as an `int32` has the program index as value. -/
theorem pid_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (tl.program_id ((launch d).ctx i) 0).toInt = (i : Nat) := by
  have hi : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hb := num_programs_lt hd
  rw [tl.program_id, launch_pid]
  exact IntV.I32.toInt_ofNat (by omega)

/-- `int32` division of nonnegative by positive is the integer quotient. -/
theorem div?_toInt {a b q : I32} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt)
    (hq : a.div? b = some q) : q.toInt = a.toInt / b.toInt := by
  obtain ⟨q', hq', hv⟩ := IntV.div?_of_nonneg (w := 32) ha hb
  rw [hq'] at hq
  cases hq
  exact hv

/-- `int32` remainder of nonnegative by positive is the integer remainder. -/
theorem rem?_toInt {a b q : I32} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt)
    (hq : a.rem? b = some q) : q.toInt = a.toInt % b.toInt := by
  obtain ⟨q', hq', hv⟩ := IntV.rem?_of_nonneg (w := 32) ha hb
  rw [hq'] at hq
  cases hq
  exact hv

/-- The kernel's `num_pid_n` is `tilesN`. -/
theorem num_pid_n_toInt (hd : d.ok) {q : I32}
    (hq : (IntV.ofNat d.N + (IntV.ofNat 127 : I32)).div? (IntV.ofNat 128 : I32) = some q) :
    q.toInt = ((tilesN d : Nat) : Int) := by
  rw [div?_toInt (by rw [N_add127_toInt hd]; positivity) (by rw [lit128_toInt]; norm_num) hq,
    N_add127_toInt hd, lit128_toInt]
  unfold tilesN
  exact (Int.natCast_div (d.N + 127) 128).symm

/-- The kernel's `iters_per_tile` is `steps K`. -/
theorem iters_per_tile_toInt (hd : d.ok) {q : I32}
    (hq : (IntV.ofNat d.K + (IntV.ofNat 63 : I32)).div? (IntV.ofNat 64 : I32) = some q) :
    q.toInt = ((steps d.K : Nat) : Int) := by
  rw [div?_toInt (by rw [K_add63_toInt hd]; positivity) (by rw [lit64_toInt]; norm_num) hq,
    K_add63_toInt hd, lit64_toInt]
  unfold steps
  exact (Int.natCast_div (d.K + 63) 64).symm

/-- A range start is below `2^31`. -/
theorem lo_lt (hd : d.ok) {p : Nat} (hp : p < d.num_programs) : lo d p < 2 ^ 31 := by
  have h1 : p * d.iters_per_program ≤ d.num_programs * d.iters_per_program :=
    Nat.mul_le_mul_right _ (le_of_lt hp)
  have h2 := hd.progs_lt
  unfold lo
  omega

/-- `lo p + iters_per_program` is below `2^31`. -/
theorem lo_add_ipp_lt (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    lo d p + d.iters_per_program < 2 ^ 31 := by
  have h1 : (p + 1) * d.iters_per_program ≤ d.num_programs * d.iters_per_program :=
    Nat.mul_le_mul_right _ (by omega)
  have h2 := hd.progs_lt
  rw [succ_mul_ipp] at h1
  omega

/-- The kernel's initial `it` is `lo p`. -/
theorem it_val (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    (IntV.ofNat p * IntV.ofNat d.iters_per_program : I32).toInt = ((lo d p : Nat) : Int) := by
  rw [IntV.I32.mul_nat (IntV.I32.toInt_ofNat (by have := num_programs_lt hd; omega))
    (ipp_toInt hd) (lo_lt hd hp)]
  rfl

/-- The kernel's `it + iters_per_program` is `lo p + iters_per_program`. -/
theorem it_add_ipp (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    (IntV.ofNat p * IntV.ofNat d.iters_per_program + IntV.ofNat d.iters_per_program :
      I32).toInt = ((lo d p + d.iters_per_program : Nat) : Int) := by
  rw [IntV.I32.add_nat (it_val hd hp) (ipp_toInt hd) (lo_add_ipp_lt hd hp)]

/-- `«end» = min (it + iters_per_program) total_iters = hi d p`. -/
theorem end_val (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    (tl.iminimum (Tensor.scalar (IntV.ofNat p * IntV.ofNat d.iters_per_program +
        IntV.ofNat d.iters_per_program : I32)) (Tensor.scalar (IntV.ofNat d.total_iters : I32))).item.toInt =
      ((hi d p : Nat) : Int) := by
  have h3 : (tl.iminimum (Tensor.scalar (IntV.ofNat p * IntV.ofNat d.iters_per_program +
        IntV.ofNat d.iters_per_program : I32))
      (Tensor.scalar (IntV.ofNat d.total_iters : I32))).item =
      IntV.min (IntV.ofNat p * IntV.ofNat d.iters_per_program + IntV.ofNat d.iters_per_program : I32)
        (IntV.ofNat d.total_iters : I32) := rfl
  rw [h3, IntV.toInt_min, it_add_ipp hd hp, total_toInt hd]
  unfold hi
  rw [succ_mul_ipp]
  by_cases h : lo d p + d.iters_per_program ≤ d.total_iters
  · rw [min_eq_left h, min_eq_left (by exact_mod_cast h)]
  · rw [min_eq_right (by omega), min_eq_right (by omega)]

/-! ## The per-iteration derived indices -/

/-- The tile of iteration `it`. -/
def tileOf (d : Inst) (it : Nat) : Nat := it / steps d.K
/-- The chunk within the tile of iteration `it`. -/
def kFirstOf (d : Inst) (it : Nat) : Nat := it % steps d.K
/-- One past the last chunk of the tile handled by the iteration range `[it, end)`. -/
def kLastOf (d : Inst) (it end_ : Nat) : Nat := min (steps d.K) (kFirstOf d it + (end_ - it))
/-- The tile row of tile `u`. -/
def pidMOf (d : Inst) (u : Nat) : Nat := u / tilesN d
/-- The tile column of tile `u`. -/
def pidNOf (d : Inst) (u : Nat) : Nat := u % tilesN d

/-- `steps K` is positive. -/
theorem T_pos {d : Inst} (hd : d.ok) : 0 < steps d.K := steps_pos hd.K_pos

/-- `tilesN` is positive. -/
theorem tilesN_pos {d : Inst} (hd : d.ok) : 0 < tilesN d := by
  have := hd.N_pos; unfold tilesN; omega

/-- `tilesM` is positive. -/
theorem tilesM_pos {d : Inst} (hd : d.ok) : 0 < tilesM d := by
  have := hd.M_pos; unfold tilesM; omega

/-- `total_iters = tilesM * tilesN * steps K`. -/
theorem total_eq {d : Inst} (hd : d.ok) : d.total_iters = tilesM d * tilesN d * steps d.K :=
  hd.total

/-- The tile of an iteration below `total_iters` is a tile. -/
theorem tileOf_lt {d : Inst} (hd : d.ok) {it : Nat} (h : it < d.total_iters) :
    tileOf d it < tilesM d * tilesN d := by
  have hT := T_pos hd
  unfold tileOf
  rw [Nat.div_lt_iff_lt_mul hT]
  rw [total_eq hd] at h
  exact h

/-- An iteration is its tile times `steps K` plus its offset in the tile. -/
theorem tile_decomp {d : Inst} {it : Nat} : it = tileOf d it * steps d.K + kFirstOf d it := by
  unfold tileOf kFirstOf
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod it (steps d.K)).symm

/-- The offset in the tile is below `steps K`. -/
theorem kFirstOf_lt {d : Inst} (hd : d.ok) {it : Nat} (_h : it < d.total_iters) :
    kFirstOf d it < steps d.K := by
  have hT := T_pos hd
  unfold kFirstOf
  exact Nat.mod_lt _ hT

/-- The tile row is below `tilesM`. -/
theorem pidMOf_lt {d : Inst} (hd : d.ok) {it : Nat} (h : it < d.total_iters) :
    pidMOf d (tileOf d it) < tilesM d := by
  have hN := tilesN_pos hd
  unfold pidMOf
  rw [Nat.div_lt_iff_lt_mul hN]
  simpa [Nat.mul_comm] using tileOf_lt hd h

/-- The tile column is below `tilesN`. -/
theorem pidNOf_lt {d : Inst} (hd : d.ok) {it : Nat} (_h : it < d.total_iters) :
    pidNOf d (tileOf d it) < tilesN d := by
  have hN := tilesN_pos hd
  unfold pidNOf
  exact Nat.mod_lt _ hN

/-- Reassembling the tile. -/
theorem tile_reassemble {d : Inst} (hd : d.ok) (u : Nat) :
    pidMOf d u * tilesN d + pidNOf d u = u := by
  have hN := tilesN_pos hd
  unfold pidMOf pidNOf
  simpa [Nat.mul_comm] using Nat.div_add_mod u (tilesN d)

/-- `k_last ≤ steps K`. -/
theorem kLastOf_le {d : Inst} {it end_ : Nat} : kLastOf d it end_ ≤ steps d.K := min_le_left _ _

/-- `k_last ≥ k_first`. -/
theorem kLastOf_ge {d : Inst} (hd : d.ok) {it end_ : Nat} (_h : it ≤ end_) :
    kFirstOf d it ≤ kLastOf d it end_ := by
  have hT := T_pos hd
  refine le_min ?_ ?_
  · unfold kFirstOf; exact le_of_lt (Nat.mod_lt _ hT)
  · unfold kFirstOf; exact Nat.le_add_right _ _

/-- `k_last - k_first` is the length of the run. -/
theorem kLastOf_sub {d : Inst} {it end_ : Nat} (h : it ≤ end_) :
    kLastOf d it end_ - kFirstOf d it = min (steps d.K - kFirstOf d it) (end_ - it) := by
  unfold kLastOf
  omega

/-- The chunks of one tile that a program handles are `[c0, c1)`. -/
def c0 (d : Inst) (p u : Nat) : Nat :=
  min (steps d.K) (max 0 (lo d p - u * steps d.K))
/-- The end of the chunk range `[c0, c1)` of one tile that a program handles. -/
def c1 (d : Inst) (p u : Nat) : Nat :=
  min (steps d.K) (max 0 (hi d p - u * steps d.K))

/-- A program's run in a tile is ordered. -/
theorem c0_le_c1 {d : Inst} {p u : Nat} (h : lo d p ≤ hi d p) : c0 d p u ≤ c1 d p u := by
  unfold c0 c1
  omega

/-- A run ends within the tile. -/
theorem c1_le {d : Inst} {p u : Nat} : c1 d p u ≤ steps d.K := min_le_left _ _

/-- Chunk `v` of tile `u` is handled by program `p` iff `c0 p u ≤ v < c1 p u`. -/
theorem mem_c_iff {d : Inst} {p u v : Nat} (hv : v < steps d.K) :
    (c0 d p u ≤ v ∧ v < c1 d p u) ↔ (lo d p ≤ u * steps d.K + v ∧ u * steps d.K + v < hi d p) := by
  unfold c0 c1
  omega

/-- The per-tile chunk ranges of the programs partition the chunks of the tile. -/
theorem chunk_partition {d : Inst} (hd : d.ok) {u : Nat} (hu : u < tilesM d * tilesN d)
    {v : Nat} (hv : v < steps d.K) :
    ∃! p, p < d.num_programs ∧ c0 d p u ≤ v ∧ v < c1 d p u := by
  have hT := T_pos hd
  have hit : u * steps d.K + v < d.total_iters := by
    rw [total_eq hd]
    have h1 : u * steps d.K + v < (u + 1) * steps d.K := by
      rw [Nat.succ_mul]; omega
    have h2 : (u + 1) * steps d.K ≤ tilesM d * tilesN d * steps d.K :=
      Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hu)
    omega
  obtain ⟨p, hp, hlo, hhi⟩ := cover_all hd hit
  refine ⟨p, ⟨hp, (mem_c_iff hv).mpr ⟨hlo, hhi⟩⟩, ?_⟩
  rintro q ⟨hq, hc0, hc1⟩
  obtain ⟨hlo', hhi'⟩ := (mem_c_iff hv).mp ⟨hc0, hc1⟩
  have hkp : hi d p ≤ lo d p + d.iters_per_program := by
    rw [← succ_mul_ipp]; exact hi_le_succ d p
  have hkq : hi d q ≤ lo d q + d.iters_per_program := by
    rw [← succ_mul_ipp]; exact hi_le_succ d q
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
  · have h1 := lo_mono d (Nat.succ_le_of_lt hlt)
    rw [lo_succ] at h1
    omega
  · have h1 := lo_mono d (Nat.succ_le_of_lt hlt)
    rw [lo_succ] at h1
    omega

end Claims.StreamKMatmul

