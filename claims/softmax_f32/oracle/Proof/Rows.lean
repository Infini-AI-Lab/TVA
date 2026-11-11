import Proof.Lanes

/-!
# The `int32` arithmetic of the row loop

Program `p` runs `tl.forRange p n_rows num_programs`: the trip count is
`tl.tripCount p n_rows num_programs`, the `t`-th trip handles row `r = p + t * num_programs`, and
`row_idx * stride` is an `int32` product that must not wrap (`in_off`, `out_off`). Every row below
`n_rows` is handled by exactly one program (`grid_pos`, `rows_grid`).

The lemmas here turn the instance's bounds into the facts the rules ask for: `tl.RangeOk` for the
loop, the row of each trip, the `int32` product `row_idx * stride`, and the in-bounds element index
of each lane (`x_len`, `y_len`).
-/
namespace Claims.Softmax

open TVA 

/-- The number of trips of program `p`: the `int32` loop `tl.forRange p n_rows num_programs`. -/
abbrev trips (d : Inst) (p : Nat) : Nat :=
  tl.tripCount (p : Int) (d.n_rows : Int) (d.num_programs : Int)

/-- With a positive step and `p ≤ n_rows`, the trip count is the ceiling division. -/
private theorem trips_eq {d : Inst} {p : Nat} (hP : 0 < d.num_programs) (hp : p ≤ d.n_rows) :
    trips d p = (d.n_rows - p + d.num_programs - 1) / d.num_programs := by
  unfold trips tl.tripCount
  have h1 : (p : Int) ≤ (d.n_rows : Int) := by exact_mod_cast hp
  have h2 : (1 : Int) ≤ (d.num_programs : Int) := by exact_mod_cast hP
  have hnum : ((d.n_rows : Int) - (p : Int) + (d.num_programs : Int) - 1)
      = ((d.n_rows - p + d.num_programs - 1 : Nat) : Int) := by
    have h3 : ((d.n_rows - p : Nat) : Int) = (d.n_rows : Int) - (p : Int) := by
      rw [Nat.cast_sub hp]
    have h4 : (1 : Nat) ≤ d.n_rows - p + d.num_programs := by omega
    have h5 : ((d.n_rows - p + d.num_programs - 1 : Nat) : Int)
        = ((d.n_rows - p + d.num_programs : Nat) : Int) - 1 := by
      rw [Nat.cast_sub h4]
      norm_num
    rw [h5, Nat.cast_add, h3]
  rw [hnum, ← Int.natCast_ediv]
  exact Int.toNat_natCast _

/-- With a positive step and `n_rows ≤ p`, the trip count is zero. -/
private theorem trips_eq_zero {d : Inst} {p : Nat} (hP : 0 < d.num_programs)
    (hp : d.n_rows ≤ p) : trips d p = 0 := by
  unfold trips tl.tripCount
  have h1 : (d.n_rows : Int) ≤ (p : Int) := by exact_mod_cast hp
  have hk : (0 : Int) < (d.num_programs : Int) := by exact_mod_cast hP
  have hlt : ((d.n_rows : Int) - (p : Int) + (d.num_programs : Int) - 1)
      < (d.num_programs : Int) := by omega
  have hdiv : ((d.n_rows : Int) - (p : Int) + (d.num_programs : Int) - 1)
      / (d.num_programs : Int) ≤ 0 := by
    rw [Int.ediv_le_iff_le_mul hk]
    simpa using hlt
  exact Int.toNat_of_nonpos hdiv

/-- The loop's `tl.RangeOk`: `num_programs` is positive and `n_rows + num_programs - 1` fits an
`int32`. -/
theorem range_ok {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    tl.RangeOk (IntV.ofNat p : I32) (IntV.ofNat d.n_rows : I32) (IntV.ofNat d.num_programs : I32) := by
  have hP0 : 0 < d.num_programs := Nat.lt_of_le_of_lt (Nat.zero_le p) hp
  have hP31 : d.num_programs < 2 ^ 31 := hd.grid_lt
  have hrg : d.n_rows + d.num_programs ≤ 2 ^ 31 := hd.rows_grid
  have hN31 : d.n_rows < 2 ^ 31 := by omega
  have hp31 : p < 2 ^ 31 := by omega
  have hir : IntV.maxInt true 32 = 2 ^ 31 - 1 := by simp [IntV.maxInt]
  unfold tl.RangeOk
  rw [IntV.I32.toInt_ofNat hp31, IntV.I32.toInt_ofNat hN31, IntV.I32.toInt_ofNat hP31, hir]
  refine ⟨by exact_mod_cast hP0, ?_, ?_, ?_⟩ <;> omega

/-- Trip `t` of program `p` handles row `p + t * num_programs`, which is below `n_rows`. -/
theorem row_lt {d : Inst} (hd : d.ok) {p t : Nat} (hp : p < d.num_programs)
    (ht : t < trips d p) : p + t * d.num_programs < d.n_rows := by
  have _ := hd
  have hP : 0 < d.num_programs := Nat.lt_of_le_of_lt (Nat.zero_le p) hp
  rcases Nat.lt_or_ge p d.n_rows with hlt | hge
  · rw [trips_eq hP (le_of_lt hlt)] at ht
    have h := (Nat.lt_div_iff_mul_lt hP).mp ht
    omega
  · rw [trips_eq_zero hP hge] at ht
    omega

/-- The row index of trip `t` as an `int32`: `row_idx.toInt = p + t * num_programs`. -/
theorem row_toInt {d : Inst} (hd : d.ok) {p t : Nat} (hp : p < d.num_programs)
    (ht : t < trips d p) :
    (IntV.ofInt ((p : Int) + (t : Int) * (d.num_programs : Int)) : I32).toInt =
      ((p + t * d.num_programs : Nat) : Int) := by
  have h2 : p + t * d.num_programs < 2 ^ 31 := by
    have hlt := row_lt hd hp ht
    have hrg : d.n_rows + d.num_programs ≤ 2 ^ 31 := hd.rows_grid
    omega
  have hcast : ((p : Int) + (t : Int) * (d.num_programs : Int))
      = ((p + t * d.num_programs : Nat) : Int) := by push_cast; ring
  rw [hcast]
  have hnn : (0 : Int) ≤ ((p + t * d.num_programs : Nat) : Int) := by positivity
  exact IntV.I32.toInt_ofInt ⟨by omega, by exact_mod_cast h2⟩

/-- The `int32` product `row_idx * stride` does not wrap: it is `r * stride`. -/
theorem stride_mul {row : I32} {stride r : Nat} (hstride : stride < 2 ^ 31)
    (hrow : row.toInt = (r : Int)) (hprod : (r : Int) * stride < 2 ^ 31) :
    (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride := by
  have hstr : (IntV.ofNat stride : I32).toInt = (stride : Int) := IntV.I32.toInt_ofNat hstride
  have h := IntV.I32.mul_nat hrow hstr (by exact_mod_cast hprod)
  rw [h]; push_cast; rfl

/-- The `int32` product `row_idx * stride` needs no help from the block size: `in_off` bounds
`(n_rows - 1) * stride` by `2 ^ 31`, and pointer arithmetic is exact in `Int`, so the per-lane
offsets `r * stride + l` (all `l < 2 ^ 14`) never wrap either. A bound of the form
`r * stride + BLOCK_SIZE ≤ 2 ^ 31` is *not* available: `mk (2 ^ 24) 128 14 1 1` is an instance
whose `(n_rows - 1) * stride` is `2 ^ 31 - 128`, so the block size would take it past `2 ^ 31`. -/
theorem stride_mul_bound {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.n_rows) :
    (r : Int) * d.input_row_stride < 2 ^ 31 := by
  have h := hd.in_off
  have hle : r ≤ d.n_rows - 1 := by omega
  have hmul : r * d.input_row_stride ≤ (d.n_rows - 1) * d.input_row_stride :=
    Nat.mul_le_mul_right _ hle
  have hlt : r * d.input_row_stride < 2 ^ 31 := lt_of_le_of_lt hmul h
  exact_mod_cast hlt

/-- The same for the output stride. -/
theorem stride_mul_bound_out {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.n_rows) :
    (r : Int) * d.output_row_stride < 2 ^ 31 := by
  have h := hd.out_off
  have hle : r ≤ d.n_rows - 1 := by omega
  have hmul : r * d.output_row_stride ≤ (d.n_rows - 1) * d.output_row_stride :=
    Nat.mul_le_mul_right _ hle
  have hlt : r * d.output_row_stride < 2 ^ 31 := lt_of_le_of_lt hmul h
  exact_mod_cast hlt

/-- Column `j < n_cols` of row `r < n_rows` is a valid element of `x`. -/
theorem in_lane_lt {d : Inst} (hd : d.ok) {r j : Nat} (hr : r < d.n_rows) (hj : j < d.n_cols) :
    r * d.input_row_stride + j < d.x.len := by
  simpa [inIdx] using hd.x_len r j hr hj

/-- Column `j < n_cols` of row `r < n_rows` is a valid element of `y`. -/
theorem out_lane_lt {d : Inst} (hd : d.ok) {r j : Nat} (hr : r < d.n_rows) (hj : j < d.n_cols) :
    r * d.output_row_stride + j < d.y.len := by
  simpa [outIdx] using hd.y_len r j hr hj

/-- `n_cols` is an `int32`. -/
theorem n_cols_lt_two_pow {d : Inst} (hd : d.ok) : d.n_cols < 2 ^ 31 := by
  have := hd.cols_dim.2.2
  omega

/-- `BLOCK_SIZE` is positive. -/
theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_SIZE := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [he]
  positivity

/-- Every row below `n_rows` is trip `r / num_programs` of program `r % num_programs`. -/
theorem trip_cover {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.n_rows) :
    r % d.num_programs < d.num_programs ∧ r / d.num_programs < trips d (r % d.num_programs) ∧
      r % d.num_programs + r / d.num_programs * d.num_programs = r := by
  have hN : d.n_rows ≠ 0 := by omega
  have hP : 0 < d.num_programs := hd.grid_pos hN
  have hpN : r % d.num_programs ≤ d.n_rows := by
    have := Nat.mod_le r d.num_programs
    omega
  refine ⟨Nat.mod_lt _ hP, ?_, ?_⟩
  · rw [trips_eq hP hpN, Nat.lt_div_iff_mul_lt hP]
    have h := Nat.div_add_mod r d.num_programs
    rw [Nat.mul_comm d.num_programs (r / d.num_programs)] at h
    omega
  · have h := Nat.div_add_mod r d.num_programs
    rw [Nat.mul_comm d.num_programs (r / d.num_programs), Nat.add_comm] at h
    exact h

end Claims.Softmax
