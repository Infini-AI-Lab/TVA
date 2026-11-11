import Proof.Basic

/-!
# The lanes of a tile

Program `(p0, p1)` of the grid owns the token tile `[p0·BLOCK_M, (p0+1)·BLOCK_M)` and the column
tile `[p1·BLOCK_DIM, (p1+1)·BLOCK_DIM)`. A lane `(a, b)` of its `[BLOCK_M, BLOCK_DIM]` tile is
row `laneT = p0·BLOCK_M + a` of the token axis and column `laneJ = p1·BLOCK_DIM + b` of the
hidden axis; it is *enabled* (`enLane`) when both are inside the tensor. This file relates the
kernel's own arithmetic (masks and pointer expressions) to those lane indices.
-/

namespace Claims.MoeSumReduce

open TVA

/-- `a * b` of `int32` or `int64` values has the product as value when it fits. -/
theorem toInt_mul_of_fits {w : Nat} {a b : IntV true w} (_ : 0 < w)
    (h : IntV.SFits w (a.toInt * b.toInt)) : (a * b).toInt = a.toInt * b.toInt :=
  IntV.toInt_mul rfl rfl h

/-- `a + b` of `int32` or `int64` values has the sum as value when it fits. -/
theorem toInt_add_of_fits {w : Nat} {a b : IntV true w} (_ : 0 < w)
    (h : IntV.SFits w (a.toInt + b.toInt)) : (a + b).toInt = a.toInt + b.toInt :=
  IntV.toInt_add rfl rfl h

/-- A lane of a `[BLOCK_M, BLOCK_DIM]` tile. -/
abbrev Lane (d : Inst) := Idx [d.BLOCK_M, d.BLOCK_DIM]

/-- The token of a lane. -/
def laneT (d : Inst) (p0 : Nat) (i : Lane d) : Nat := p0 * d.BLOCK_M + i.1.val

/-- The hidden-dimension column of a lane. -/
def laneJ (d : Inst) (p1 : Nat) (i : Lane d) : Nat := p1 * d.BLOCK_DIM + i.2.1.val

/-- The lane is inside the tensor on both axes: the lane's mask bit. -/
def enLane (d : Inst) (p0 p1 : Nat) (i : Lane d) : Bool :=
  decide (laneT d p0 i < d.T) && decide (laneJ d p1 i < d.D)

/-- The input element read by lane `i` at iteration `k`: `input[t, k, j]` is `t·K·D + k·D + j`. -/
def inIdx (d : Inst) (p0 p1 k : Nat) (i : Lane d) : Nat :=
  laneT d p0 i * d.K * d.D + k * d.D + laneJ d p1 i

/-- The output element written by lane `i`: `output[t, j]` is `t·D + j`. -/
def outIdx (d : Inst) (p0 p1 : Nat) (i : Lane d) : Nat := laneT d p0 i * d.D + laneJ d p1 i

/-- The output elements a program writes: the indices of its enabled lanes. -/
def wtile (d : Inst) (p0 p1 : Nat) : Set Nat := Buf.img (outIdx d p0 p1) (enLane d p0 p1)

/-! ## The extent of a tile -/

/-- `(a * m + b) / m = a + b / m`. -/
theorem nat_mul_add_div {a m b : Nat} (h : 0 < m) : (a * m + b) / m = a + b / m := by
  rw [Nat.mul_comm]
  exact Nat.mul_add_div h a b

/-- `(a * m + b) % m = b % m`. -/
theorem nat_mul_add_mod {a m b : Nat} : (a * m + b) % m = b % m := by
  rw [Nat.mul_comm]
  exact Nat.mul_add_mod m a b

/-- The token blocks cover `T`. -/
theorem gT_mul_lo {d : Inst} (hd : d.ok) : d.T ≤ gT d * d.BLOCK_M := by
  have hB := block_m_pos hd
  have h1 : d.BLOCK_M * ((d.T + d.BLOCK_M - 1) / d.BLOCK_M) + (d.T + d.BLOCK_M - 1) % d.BLOCK_M
      = d.T + d.BLOCK_M - 1 := Nat.div_add_mod _ _
  have h2 : (d.T + d.BLOCK_M - 1) % d.BLOCK_M < d.BLOCK_M := Nat.mod_lt _ hB
  show d.T ≤ (d.T + d.BLOCK_M - 1) / d.BLOCK_M * d.BLOCK_M
  rw [Nat.mul_comm] at h1
  omega

/-- The token blocks overshoot `T` by less than one block. -/
theorem gT_mul_hi {d : Inst} (_hd : d.ok) : gT d * d.BLOCK_M ≤ d.T + d.BLOCK_M - 1 := by
  show (d.T + d.BLOCK_M - 1) / d.BLOCK_M * d.BLOCK_M ≤ d.T + d.BLOCK_M - 1
  exact Nat.div_mul_le_self _ _

/-- The column blocks cover `D`. -/
theorem gD_mul_lo {d : Inst} (hd : d.ok) : d.D ≤ gD d * d.BLOCK_DIM := by
  have hB := block_d_pos hd
  have h1 : d.BLOCK_DIM * ((d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM) + (d.D + d.BLOCK_DIM - 1) % d.BLOCK_DIM
      = d.D + d.BLOCK_DIM - 1 := Nat.div_add_mod _ _
  have h2 : (d.D + d.BLOCK_DIM - 1) % d.BLOCK_DIM < d.BLOCK_DIM := Nat.mod_lt _ hB
  show d.D ≤ (d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM * d.BLOCK_DIM
  rw [Nat.mul_comm] at h1
  omega

/-- The column blocks overshoot `D` by less than one block. -/
theorem gD_mul_hi {d : Inst} (_hd : d.ok) : gD d * d.BLOCK_DIM ≤ d.D + d.BLOCK_DIM - 1 := by
  show (d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM * d.BLOCK_DIM ≤ d.D + d.BLOCK_DIM - 1
  exact Nat.div_mul_le_self _ _

/-- The grid's extent along the hidden dimension is below `2^31`. -/
theorem gD_lt {d : Inst} (hd : d.ok) : gD d < 2 ^ 31 := by
  have hT := gT_pos hd
  have h : gT d * gD d < 2 ^ 31 := hd.grid_size_lt
  have : gD d ≤ gT d * gD d := Nat.le_mul_of_pos_left _ hT
  omega

/-- A tile row of a program of the grid, plus one block, is below `2^23`. -/
theorem tblock_succ_lt {d : Inst} (hd : d.ok) {p0 : Nat} (hp0 : p0 < gT d) :
    p0 * d.BLOCK_M + d.BLOCK_M ≤ 2 ^ 23 := by
  have hB := block_m_pos hd
  have h2 : p0 * d.BLOCK_M + d.BLOCK_M ≤ gT d * d.BLOCK_M := by
    have h := Nat.mul_le_mul_right d.BLOCK_M (Nat.succ_le_of_lt hp0)
    rwa [Nat.succ_mul] at h
  have h3 := gT_mul_hi hd
  have h4 := hd.T_ctx
  have h5 := block_m_le hd
  omega

/-- A tile row of a program of the grid is below `2^23`. -/
theorem tblock_lt {d : Inst} (hd : d.ok) {p0 : Nat} (hp0 : p0 < gT d) :
    p0 * d.BLOCK_M < 2 ^ 23 := by
  have hB := block_m_pos hd
  have h := tblock_succ_lt hd hp0
  omega

/-- A tile column of a program of the grid, plus one block, is below `2^23`. -/
theorem jblock_succ_lt {d : Inst} (hd : d.ok) {p1 : Nat} (hp1 : p1 < gD d) :
    p1 * d.BLOCK_DIM + d.BLOCK_DIM ≤ 2 ^ 23 := by
  have hB := block_d_pos hd
  have h2 : p1 * d.BLOCK_DIM + d.BLOCK_DIM ≤ gD d * d.BLOCK_DIM := by
    have h := Nat.mul_le_mul_right d.BLOCK_DIM (Nat.succ_le_of_lt hp1)
    rwa [Nat.succ_mul] at h
  have h3 := gD_mul_hi hd
  have h4 := hd.D_dim
  have h5 := block_d_le hd
  omega

/-- A tile column of a program of the grid is below `2^23`. -/
theorem jblock_lt {d : Inst} (hd : d.ok) {p1 : Nat} (hp1 : p1 < gD d) :
    p1 * d.BLOCK_DIM < 2 ^ 23 := by
  have hB := block_d_pos hd
  have h := jblock_succ_lt hd hp1
  omega

/-- A lane's token, for a program of the grid, is a small number: below `2^23`. -/
theorem laneT_lt {d : Inst} (hd : d.ok) {p0 : Nat} (hp0 : p0 < gT d) (i : Lane d) :
    laneT d p0 i < 2 ^ 23 := by
  have h := tblock_succ_lt hd hp0
  have ha := i.1.isLt
  unfold laneT
  omega

/-- A lane's column, for a program of the grid, is a small number: below `2^23`. -/
theorem laneJ_lt {d : Inst} (hd : d.ok) {p1 : Nat} (hp1 : p1 < gD d) (i : Lane d) :
    laneJ d p1 i < 2 ^ 23 := by
  have h := jblock_succ_lt hd hp1
  have ha := i.2.1.isLt
  unfold laneJ
  omega

/-- An enabled lane's token is inside the tensor. -/
theorem laneT_lt_T {d : Inst} {p0 p1 : Nat} {i : Lane d} (h : enLane d p0 p1 i = true) :
    laneT d p0 i < d.T := by
  unfold enLane at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1

/-- An enabled lane's column is inside the tensor. -/
theorem laneJ_lt_D {d : Inst} {p0 p1 : Nat} {i : Lane d} (h : enLane d p0 p1 i = true) :
    laneJ d p1 i < d.D := by
  unfold enLane at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

/-! ## The mask -/

/-- `x` in `[0, 2^23)` fits in `int32`. -/
theorem sfits32_of_lt23 {x : Int} (h0 : 0 ≤ x) (h1 : x < ((2 ^ 23 : Nat) : Int)) :
    IntV.SFits 32 x := by
  have he : ((2 : Int) ^ (32 - 1)) = 2147483648 := by norm_num
  have hc : ((2 ^ 23 : Nat) : Int) = 8388608 := by norm_num
  rw [IntV.SFits, he, hc] at *
  omega
/-- `.get (a, ())` form of a scalar plus a rank-1 `int32` tensor. -/
theorem get1_sadd {n : Nat} (A : I32) (x : Tensor [n] I32) (a : Fin n) :
    (A + x).get (a, ()) = A + x.get (a, ()) := rfl

/-- `.get (a, ())` form of a rank-1 `int32` tensor compared with a scalar. -/
theorem get1_ltS {n : Nat} (x : Tensor [n] I32) (A : I32) (a : Fin n) :
    (x.ltS A).get (a, ()) = IntV.lt (x.get (a, ())) A := rfl

/-- `.get (a, b, u)` form of a column broadcast. -/
theorem get2_col {m n : Nat} {α : Type} (x : Tensor [m] α) (a : Fin m) (b : Fin n) (u : Idx []) :
    (x.col n).get (a, b, u) = x.get (a, ()) := rfl

/-- `.get (a, b, u)` form of a row broadcast. -/
theorem get2_row {m n : Nat} {α : Type} (x : Tensor [n] α) (a : Fin m) (b : Fin n) (u : Idx []) :
    (x.row m).get (a, b, u) = x.get (b, ()) := rfl

/-- `.get (a, b, u)` form of `and`. -/
theorem get2_and {m n : Nat} (x y : Tensor [m, n] Bool) (a : Fin m) (b : Fin n) (u : Idx []) :
    (x.and y).get (a, b, u) = (x.get (a, b, u) && y.get (a, b, u)) := rfl

/-- `.get (a, b, u)` form of a tensor product with a scalar. -/
theorem get2_mulS {m n : Nat} (x : Tensor [m, n] I64) (c : I64) (a : Fin m) (b : Fin n)
    (u : Idx []) : (x * c).get (a, b, u) = x.get (a, b, u) * c := rfl

/-- `.get (a, b, u)` form of a tensor sum. -/
theorem get2_addT {m n : Nat} (x y : Tensor [m, n] I64) (a : Fin m) (b : Fin n) (u : Idx []) :
    (x + y).get (a, b, u) = x.get (a, b, u) + y.get (a, b, u) := rfl

/-- `.get (a, b, u)` form of a pointer plus an offset tensor. -/
theorem get2_ptrAdd {m n : Nat} {dt : DType} {sg : Bool} {w : Nat}
    (p : Ptr dt) (o : Tensor [m, n] (IntV sg w)) (a : Fin m) (b : Fin n) (u : Idx []) :
    (p + o).get (a, b, u) = p.offset (o.get (a, b, u)).toInt := rfl

/-- `.get (a, b, u)` form of tensor pointer arithmetic. -/
theorem get2_ptrAddT {m n : Nat} {dt : DType} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr dt)) (o : Tensor [m, n] (IntV sg w))
    (a : Fin m) (b : Fin n) (u : Idx []) :
    (ps + o).get (a, b, u) = (ps.get (a, b, u)).offset (o.get (a, b, u)).toInt := rfl

/-- `.get (a, b, u)` form of a broadcast scalar tensor. -/
theorem get2_bc {m n : Nat} (c : I64) (a : Fin m) (b : Fin n) (u : Idx []) :
    (bc c : Tensor [m, n] I64).get (a, b, u) = c := rfl

/-- The token of a lane, as its two parts. -/
theorem laneT_val {d : Inst} (p0 : Nat) (a : Fin d.BLOCK_M) (b : Fin d.BLOCK_DIM) (u : Idx []) :
    laneT d p0 (a, b, u) = p0 * d.BLOCK_M + a.val := rfl

/-- The column of a lane, as its two parts. -/
theorem laneJ_val {d : Inst} (p1 : Nat) (a : Fin d.BLOCK_M) (b : Fin d.BLOCK_DIM) (u : Idx []) :
    laneJ d p1 (a, b, u) = p1 * d.BLOCK_DIM + b.val := rfl

/-- The enabled bit of a lane, as its two parts. -/
theorem enLane_val {d : Inst} (p0 p1 : Nat) (a : Fin d.BLOCK_M) (b : Fin d.BLOCK_DIM)
    (u : Idx []) :
    enLane d p0 p1 (a, b, u) =
      (decide (p0 * d.BLOCK_M + a.val < d.T) && decide (p1 * d.BLOCK_DIM + b.val < d.D)) := rfl

/-- The output index of a lane, as its two parts. -/
theorem outIdx_val {d : Inst} (p0 p1 : Nat) (a : Fin d.BLOCK_M) (b : Fin d.BLOCK_DIM)
    (u : Idx []) :
    outIdx d p0 p1 (a, b, u) = (p0 * d.BLOCK_M + a.val) * d.D + (p1 * d.BLOCK_DIM + b.val) := rfl

/-- The output index of a lane, as a sum. -/
theorem outIdx_eq (d : Inst) (p0 p1 : Nat) (i : Lane d) :
    outIdx d p0 p1 i = laneT d p0 i * d.D + laneJ d p1 i := rfl

/-- The input index of a lane, as a sum. -/
theorem inIdx_eq (d : Inst) (p0 p1 k : Nat) (i : Lane d) :
    inIdx d p0 p1 k i = laneT d p0 i * d.K * d.D + k * d.D + laneJ d p1 i := rfl

/-- `decide` of a comparison of two naturals, both cast to `int`. -/
theorem decide_nat_lt_intcast {m n : Nat} :
    decide (((m : Nat) : Int) < ((n : Nat) : Int)) = decide (m < n) := by
  rw [decide_eq_decide]
  exact Nat.cast_lt

/-- `x` in `[0, 2^63)` fits in `int64`. -/
theorem sfits64_of_lt {x : Int} (h0 : 0 ≤ x) (h1 : x < ((2 ^ 63 : Nat) : Int)) :
    IntV.SFits 64 x := by
  have he : ((2 : Int) ^ (64 - 1)) = 9223372036854775808 := by norm_num
  have hc : ((2 ^ 63 : Nat) : Int) = 9223372036854775808 := by norm_num
  rw [IntV.SFits, he, hc] at *
  omega

/-- A product of a threshold-sized and a hidden-sized number is below `2^54`. -/
theorem mul_lt_54 {x y : Nat} (hx : x < 2 ^ 23) (hy : y < 2 ^ 31) : x * y < 2 ^ 54 := by
  have hx' : x ≤ 2 ^ 23 - 1 := by omega
  have hy' : y ≤ 2 ^ 31 - 1 := by omega
  have hm : x * y ≤ (2 ^ 23 - 1) * (2 ^ 31 - 1) := Nat.mul_le_mul hx' hy'
  have h2 : (2 ^ 23 - 1) * ((2 : Nat) ^ 31 - 1) < 2 ^ 54 := by norm_num
  omega

/-- That product, in `int`. -/
theorem mul_lt_63_int {A D : Nat} (hA : A < 2 ^ 23) (hD : D < 2 ^ 31) :
    ((A : Int)) * ((D : Int)) < ((2 ^ 63 : Nat) : Int) := by
  have h : A * D < 2 ^ 63 := by
    have h54 := mul_lt_54 hA hD
    omega
  exact_mod_cast h

/-- That product plus a column offset, in `int`. -/
theorem add_lt_63_int {A D B : Nat} (hA : A < 2 ^ 23) (hD : D < 2 ^ 31) (hB : B < 2 ^ 23) :
    ((A : Int)) * ((D : Int)) + ((B : Int)) < ((2 ^ 63 : Nat) : Int) := by
  have h : A * D + B < 2 ^ 63 := by
    have h54 := mul_lt_54 hA hD
    omega
  exact_mod_cast h

/-- The `int32` token offsets of a lane hold the lane's token. -/
theorem offsT_toInt (d : Inst) (hd : d.ok) (c : Ctx) (hp0 : c.pid 0 < gT d) (i : Lane d) :
    ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M).get
        (i.1, ())).toInt = (laneT d (c.pid 0) i : Int) := by
  obtain ⟨a, b, u⟩ := i
  show ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M).get
        (a, ())).toInt = ((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int)
  have hp0lt : c.pid 0 < 2 ^ 31 := lt_trans hp0 (gT_lt hd)
  have hpid : (tl.program_id c 0).toInt = (c.pid 0 : Int) := by
    simp only [tl.program_id]
    exact IntV.I32.toInt_ofNat hp0lt
  have hBM : (IntV.ofNat d.BLOCK_M : I32).toInt = (d.BLOCK_M : Int) :=
    IntV.I32.toInt_ofNat (by have := block_m_le hd; omega)
  have hltA : (((0 : Nat) : Int) + (a.val : Int)) < ((2 ^ 23 : Nat) : Int) := by
    have ha : a.val < d.BLOCK_M := a.isLt
    have hB : d.BLOCK_M ≤ 2 ^ 20 := block_m_le hd
    have h : a.val < 2 ^ 23 := by omega
    have h' : ((a.val : Nat) : Int) < ((2 ^ 23 : Nat) : Int) := by exact_mod_cast h
    push_cast at h'
    simpa using h'
  have har : ((tl.arange 0 d.BLOCK_M).get (a, ())).toInt = (a.val : Int) := by
    have h0 : (0 : Int) ≤ ((0 : Nat) : Int) + (a.val : Int) := by positivity
    simp only [tl.arange, Tensor.get_arange]
    rw [IntV.toInt_ofInt (by norm_num) (sfits32_of_lt23 h0 hltA)]
    omega
  have hprod : (tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)).toInt
      = (c.pid 0 : Int) * (d.BLOCK_M : Int) := by
    have hlt : (c.pid 0 : Int) * (d.BLOCK_M : Int) < ((2 ^ 23 : Nat) : Int) := by
      have h := tblock_lt hd hp0
      exact_mod_cast h
    have hsf : IntV.SFits 32
        ((tl.program_id c 0).toInt * (IntV.ofNat d.BLOCK_M : I32).toInt) := by
      rw [hpid, hBM]
      exact sfits32_of_lt23 (by positivity) hlt
    rw [toInt_mul_of_fits (by norm_num) hsf, hpid, hBM]
  have hsf2 : IntV.SFits 32 ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)).toInt
      + ((tl.arange 0 d.BLOCK_M).get (a, ())).toInt) := by
    rw [hprod, har]
    refine sfits32_of_lt23 (by positivity) ?_
    have hs := tblock_succ_lt hd hp0
    have ha : a.val < d.BLOCK_M := a.isLt
    have hlt : c.pid 0 * d.BLOCK_M + a.val < 2 ^ 23 := by omega
    exact_mod_cast hlt
  rw [get1_sadd, toInt_add_of_fits (by norm_num) hsf2, hprod, har]
  push_cast
  ring

/-- The `int32` column offsets of a lane hold the lane's column. -/
theorem offsJ_toInt (d : Inst) (hd : d.ok) (c : Ctx) (hp1 : c.pid 1 < gD d) (i : Lane d) :
    ((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32) + tl.arange 0 d.BLOCK_DIM).get
        (i.2.1, ())).toInt = (laneJ d (c.pid 1) i : Int) := by
  obtain ⟨a, b, u⟩ := i
  show ((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32) + tl.arange 0 d.BLOCK_DIM).get
        (b, ())).toInt = ((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int)
  have hp1lt : c.pid 1 < 2 ^ 31 := lt_trans hp1 (gD_lt hd)
  have hpid : (tl.program_id c 1).toInt = (c.pid 1 : Int) := by
    simp only [tl.program_id]
    exact IntV.I32.toInt_ofNat hp1lt
  have hBD : (IntV.ofNat d.BLOCK_DIM : I32).toInt = (d.BLOCK_DIM : Int) :=
    IntV.I32.toInt_ofNat (by have := block_d_le hd; omega)
  have hltA : (((0 : Nat) : Int) + (b.val : Int)) < ((2 ^ 23 : Nat) : Int) := by
    have hb : b.val < d.BLOCK_DIM := b.isLt
    have hB : d.BLOCK_DIM ≤ 2 ^ 20 := block_d_le hd
    have h : b.val < 2 ^ 23 := by omega
    have h' : ((b.val : Nat) : Int) < ((2 ^ 23 : Nat) : Int) := by exact_mod_cast h
    push_cast at h'
    simpa using h'
  have har : ((tl.arange 0 d.BLOCK_DIM).get (b, ())).toInt = (b.val : Int) := by
    have h0 : (0 : Int) ≤ ((0 : Nat) : Int) + (b.val : Int) := by positivity
    simp only [tl.arange, Tensor.get_arange]
    rw [IntV.toInt_ofInt (by norm_num) (sfits32_of_lt23 h0 hltA)]
    omega
  have hprod : (tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)).toInt
      = (c.pid 1 : Int) * (d.BLOCK_DIM : Int) := by
    have hlt : (c.pid 1 : Int) * (d.BLOCK_DIM : Int) < ((2 ^ 23 : Nat) : Int) := by
      have h := jblock_lt hd hp1
      exact_mod_cast h
    have hsf : IntV.SFits 32
        ((tl.program_id c 1).toInt * (IntV.ofNat d.BLOCK_DIM : I32).toInt) := by
      rw [hpid, hBD]
      exact sfits32_of_lt23 (by positivity) hlt
    rw [toInt_mul_of_fits (by norm_num) hsf, hpid, hBD]
  have hsf2 : IntV.SFits 32 ((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)).toInt
      + ((tl.arange 0 d.BLOCK_DIM).get (b, ())).toInt) := by
    rw [hprod, har]
    refine sfits32_of_lt23 (by positivity) ?_
    have hs := jblock_succ_lt hd hp1
    have hb : b.val < d.BLOCK_DIM := b.isLt
    have hlt : c.pid 1 * d.BLOCK_DIM + b.val < 2 ^ 23 := by omega
    exact_mod_cast hlt
  rw [get1_sadd, toInt_add_of_fits (by norm_num) hsf2, hprod, har]
  push_cast
  ring

/-- The kernel's mask at lane `i` is the lane's enabled bit. -/
theorem mask_eq (d : Inst) (hd : d.ok) (c : Ctx) (hp0 : c.pid 0 < gT d) (hp1 : c.pid 1 < gD d)
    (i : Lane d) :
    ((((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M).ltS
          (IntV.ofNat d.T : I32)).col d.BLOCK_DIM).and
        (((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32) + tl.arange 0 d.BLOCK_DIM).ltS
          (IntV.ofNat d.D : I32)).row d.BLOCK_M)).get i
      = enLane d (c.pid 0) (c.pid 1) i := by
  obtain ⟨a, b, u⟩ := i
  show ((((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M).ltS
          (IntV.ofNat d.T : I32)).col d.BLOCK_DIM).and
        (((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32) + tl.arange 0 d.BLOCK_DIM).ltS
          (IntV.ofNat d.D : I32)).row d.BLOCK_M)).get (a, b, u)
      = enLane d (c.pid 0) (c.pid 1) (a, b, u)
  have hT : ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)
        + tl.arange 0 d.BLOCK_M).get (a, ())).toInt
      = ((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int) :=
    offsT_toInt d hd c hp0 (a, b, u)
  have hJ : ((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)
        + tl.arange 0 d.BLOCK_DIM).get (b, ())).toInt
      = ((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int) :=
    offsJ_toInt d hd c hp1 (a, b, u)
  have hTlt : (IntV.ofNat d.T : I32).toInt = (d.T : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.T_lt; omega)
  have hDlt : (IntV.ofNat d.D : I32).toInt = (d.D : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.D_lt; omega)
  have h1 : IntV.lt ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)
        + tl.arange 0 d.BLOCK_M).get (a, ())) (IntV.ofNat d.T : I32)
      = decide (((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int) < ((d.T : Nat) : Int)) := by
    rw [IntV.lt_eq, hT, hTlt]
  have h2 : IntV.lt ((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)
        + tl.arange 0 d.BLOCK_DIM).get (b, ())) (IntV.ofNat d.D : I32)
      = decide (((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int) < ((d.D : Nat) : Int)) := by
    rw [IntV.lt_eq, hJ, hDlt]
  rw [get2_and, get2_col, get2_row, get1_ltS, get1_ltS, h1, h2, enLane_val,
    decide_nat_lt_intcast, decide_nat_lt_intcast]

/-! ## The pointers -/

/-- The store's pointer at lane `i` is `output.ptr (outIdx …)`. -/
theorem store_ptrs (d : Inst) (hd : d.ok) {p0 p1 : Nat} (hp0 : p0 < gT d) (hp1 : p1 < gD d)
    (offs0 : Tensor [d.BLOCK_M] I64) (os0 : I64) (offs1 : Tensor [d.BLOCK_DIM] I64)
    (ho0 : ∀ a, (offs0.get1 a).toInt = ((p0 * d.BLOCK_M + a.val : Nat) : Int))
    (ho1 : ∀ b, (offs1.get1 b).toInt = ((p1 * d.BLOCK_DIM + b.val : Nat) : Int))
    (hos0 : os0.toInt = (d.D : Int))
    (i : Lane d) :
    (d.output.ptr 0 + (offs0.col d.BLOCK_DIM * os0 + offs1.row d.BLOCK_M)).get i
      = d.output.ptr ((outIdx d p0 p1 i : Nat) : Int) := by
  obtain ⟨a, b, u⟩ := i
  show (d.output.ptr 0 + (offs0.col d.BLOCK_DIM * os0 + offs1.row d.BLOCK_M)).get (a, b, u)
      = d.output.ptr (((p0 * d.BLOCK_M + a.val) * d.D + (p1 * d.BLOCK_DIM + b.val) : Nat) : Int)
  rw [get2_ptrAdd]
  have h1 : (offs0.get (a, ())).toInt = ((p0 * d.BLOCK_M + a.val : Nat) : Int) := ho0 a
  have h2 : (offs1.get (b, ())).toInt = ((p1 * d.BLOCK_DIM + b.val : Nat) : Int) := ho1 b
  have hTlt : p0 * d.BLOCK_M + a.val < 2 ^ 23 := by
    have h := laneT_lt hd hp0 (a, b, u)
    rw [laneT_val] at h
    exact h
  have hJlt : p1 * d.BLOCK_DIM + b.val < 2 ^ 23 := by
    have h := laneJ_lt hd hp1 (a, b, u)
    rw [laneJ_val] at h
    exact h
  have hm : IntV.SFits 64 ((offs0.get (a, ())).toInt * os0.toInt) := by
    rw [h1, hos0]
    exact sfits64_of_lt (by positivity) (mul_lt_63_int hTlt hd.D_lt)
  have ha : IntV.SFits 64 ((offs0.get (a, ()) * os0).toInt + (offs1.get (b, ())).toInt) := by
    rw [toInt_mul_of_fits (by norm_num) hm, h1, hos0, h2]
    exact sfits64_of_lt (by positivity) (add_lt_63_int hTlt hd.D_lt hJlt)
  have hoff : ((offs0.col d.BLOCK_DIM * os0 + offs1.row d.BLOCK_M).get (a, b, u)).toInt
      = (((p0 * d.BLOCK_M + a.val) * d.D + (p1 * d.BLOCK_DIM + b.val) : Nat) : Int) := by
    rw [get2_addT, get2_mulS, get2_col, get2_row]
    rw [toInt_add_of_fits (by norm_num) ha, toInt_mul_of_fits (by norm_num) hm, h1, hos0, h2]
    push_cast
    ring
  rw [hoff]
  simp only [Buf.ptr, Ptr.offset, Int.zero_mul, Int.add_zero]

/-- The load's pointer at lane `i`, at iteration `k < K`, is `input.ptr (inIdx …)`. -/
theorem load_ptrs_bounded (d : Inst) (hd : d.ok) {p0 p1 : Nat} (hp0 : p0 < gT d)
    (hp1 : p1 < gD d)
    (offs0 : Tensor [d.BLOCK_M] I64) (s0 : I64) (offs1 : Tensor [d.BLOCK_DIM] I64)
    (i64 s1 : I64) (k : Nat)
    (ho0 : ∀ a, (offs0.get1 a).toInt = ((p0 * d.BLOCK_M + a.val : Nat) : Int))
    (hs0 : s0.toInt = ((d.K * d.D : Nat) : Int))
    (ho1 : ∀ b, (offs1.get1 b).toInt = ((p1 * d.BLOCK_DIM + b.val : Nat) : Int))
    (hi : i64.toInt = (k : Int))
    (hs1 : s1.toInt = (d.D : Int))
    (hk : k < d.K)
    (i : Lane d) :
    ((d.input.ptr 0 + (offs0.col d.BLOCK_DIM * s0 + offs1.row d.BLOCK_M)) +
        (bc (i64 * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64)).get i
      = d.input.ptr ((inIdx d p0 p1 k i : Nat) : Int) := by
  obtain ⟨a, b, u⟩ := i
  show ((d.input.ptr 0 + (offs0.col d.BLOCK_DIM * s0 + offs1.row d.BLOCK_M)) +
        (bc (i64 * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64)).get (a, b, u)
      = d.input.ptr (((p0 * d.BLOCK_M + a.val) * d.K * d.D + k * d.D
          + (p1 * d.BLOCK_DIM + b.val) : Nat) : Int)
  rw [get2_ptrAddT, get2_ptrAdd, get2_bc, Ptr.offset_offset]
  have h1 : (offs0.get (a, ())).toInt = ((p0 * d.BLOCK_M + a.val : Nat) : Int) := ho0 a
  have h2 : (offs1.get (b, ())).toInt = ((p1 * d.BLOCK_DIM + b.val : Nat) : Int) := ho1 b
  have hs0' : s0.toInt = ((d.K * d.D : Nat) : Int) := hs0
  have hTlt : p0 * d.BLOCK_M + a.val < 2 ^ 23 := by
    have h := laneT_lt hd hp0 (a, b, u)
    rw [laneT_val] at h
    exact h
  have hJlt : p1 * d.BLOCK_DIM + b.val < 2 ^ 23 := by
    have h := laneJ_lt hd hp1 (a, b, u)
    rw [laneJ_val] at h
    exact h
  have hk23 : k < 2 ^ 23 := by
    have := hd.K_le
    omega
  have hoff : ((offs0.col d.BLOCK_DIM * s0 + offs1.row d.BLOCK_M).get (a, b, u)).toInt
      = (((p0 * d.BLOCK_M + a.val) * d.K * d.D + (p1 * d.BLOCK_DIM + b.val) : Nat) : Int) := by
    rw [get2_addT, get2_mulS, get2_col, get2_row]
    have hm : IntV.SFits 64 ((offs0.get (a, ())).toInt * s0.toInt) := by
      rw [h1, hs0']
      exact sfits64_of_lt (by positivity) (mul_lt_63_int hTlt hd.stride_lt)
    have ha : IntV.SFits 64
        ((offs0.get (a, ()) * s0).toInt + (offs1.get (b, ())).toInt) := by
      rw [toInt_mul_of_fits (by norm_num) hm, h1, hs0', h2]
      exact sfits64_of_lt (by positivity) (add_lt_63_int hTlt hd.stride_lt hJlt)
    rw [toInt_add_of_fits (by norm_num) ha, toInt_mul_of_fits (by norm_num) hm, h1, hs0', h2]
    push_cast
    ring
  have hbc : IntV.toInt (i64 * s1) = ((k * d.D : Nat) : Int) := by
    have hm : IntV.SFits 64 (i64.toInt * s1.toInt) := by
      rw [hi, hs1]
      exact sfits64_of_lt (by positivity) (mul_lt_63_int hk23 hd.D_lt)
    rw [toInt_mul_of_fits (by norm_num) hm, hi, hs1]
    push_cast
    ring
  rw [hoff, hbc]
  have he : (((p0 * d.BLOCK_M + a.val) * d.K * d.D + (p1 * d.BLOCK_DIM + b.val) : Nat) : Int)
      + ((k * d.D : Nat) : Int)
      = (((p0 * d.BLOCK_M + a.val) * d.K * d.D + k * d.D
          + (p1 * d.BLOCK_DIM + b.val) : Nat) : Int) := by
    push_cast
    ring
  rw [he]
  simp only [Buf.ptr, Ptr.offset, Int.zero_mul, Int.add_zero]

/-! ## Indices are inside the buffers -/

/-- An enabled lane reads an element inside the input buffer. -/
theorem inIdx_lt {d : Inst} (hd : d.ok) {p0 p1 : Nat} (i : Lane d) {k : Nat}
    (hen : enLane d p0 p1 i = true) (hk : k < d.K) : inIdx d p0 p1 k i < d.input.len := by
  have hT := laneT_lt_T hen
  have hJ := laneJ_lt_D hen
  have hD := hd.D_pos
  have h1 : laneT d p0 i * d.K * d.D + d.K * d.D ≤ d.T * d.K * d.D := by
    have h2 : laneT d p0 i + 1 ≤ d.T := by omega
    have e : laneT d p0 i * d.K * d.D + d.K * d.D = (laneT d p0 i + 1) * (d.K * d.D) := by ring
    rw [e]
    calc (laneT d p0 i + 1) * (d.K * d.D) = (laneT d p0 i + 1) * d.K * d.D := by ring
      _ ≤ d.T * d.K * d.D := Nat.mul_le_mul_right d.D (Nat.mul_le_mul_right d.K h2)
  have h3 : k * d.D + d.D ≤ d.K * d.D := by
    have h4 : k + 1 ≤ d.K := by omega
    calc k * d.D + d.D = (k + 1) * d.D := by ring
      _ ≤ d.K * d.D := Nat.mul_le_mul_right d.D h4
  have h5 : laneJ d p1 i + 1 ≤ d.D := by omega
  have hlen := hd.input_len
  rw [inIdx_eq]
  omega

/-- An enabled lane writes an element inside the output buffer. -/
theorem outIdx_lt {d : Inst} (hd : d.ok) {p0 p1 : Nat} (i : Lane d)
    (hen : enLane d p0 p1 i = true) : outIdx d p0 p1 i < d.output.len := by
  have hT := laneT_lt_T hen
  have hJ := laneJ_lt_D hen
  have hD := hd.D_pos
  have h1 : laneT d p0 i * d.D + d.D ≤ d.T * d.D := by
    calc laneT d p0 i * d.D + d.D = (laneT d p0 i + 1) * d.D := by ring
      _ ≤ d.T * d.D := Nat.mul_le_mul_right d.D (by omega)
  have h2 : laneJ d p1 i + 1 ≤ d.D := by omega
  have hlen := hd.output_len
  rw [outIdx_eq]
  omega

/-- `(p * BM + a) / BM = p` for `a < BM`. -/
theorem tile_div {p q a a' BM : Nat} (ha : a < BM) (ha' : a' < BM)
    (h : p * BM + a = q * BM + a') : p = q := by
  have hd : (p * BM + a) / BM = (q * BM + a') / BM := by rw [h]
  rw [nat_mul_add_div (show 0 < BM by omega), nat_mul_add_div (show 0 < BM by omega),
    Nat.div_eq_of_lt ha, Nat.div_eq_of_lt ha'] at hd
  omega

/-- Enabled lanes of one program write distinct output elements. -/
theorem outIdx_inj {d : Inst} (hd : d.ok) {p0 p1 : Nat} (_hp0 : p0 < gT d) (_hp1 : p1 < gD d)
    {i j : Lane d} (hi : enLane d p0 p1 i = true) (hj : enLane d p0 p1 j = true)
    (h : outIdx d p0 p1 i = outIdx d p0 p1 j) : i = j := by
  obtain ⟨a, b, u⟩ := i
  obtain ⟨a', b', u'⟩ := j
  have hTe : p0 * d.BLOCK_M + a.val < d.T := by
    have hh := laneT_lt_T hi
    rw [laneT_val] at hh
    exact hh
  have hJe : p1 * d.BLOCK_DIM + b.val < d.D := by
    have hh := laneJ_lt_D hi
    rw [laneJ_val] at hh
    exact hh
  have hTf : p0 * d.BLOCK_M + a'.val < d.T := by
    have hh := laneT_lt_T hj
    rw [laneT_val] at hh
    exact hh
  have hJf : p1 * d.BLOCK_DIM + b'.val < d.D := by
    have hh := laneJ_lt_D hj
    rw [laneJ_val] at hh
    exact hh
  have hD : 0 < d.D := hd.D_pos
  have h' := h
  rw [outIdx_val, outIdx_val] at h'
  have hsum : (p1 * d.BLOCK_DIM + b.val) = (p1 * d.BLOCK_DIM + b'.val) := by
    have hm := congrArg (fun x => x % d.D) h'
    rw [nat_mul_add_mod, nat_mul_add_mod, Nat.mod_eq_of_lt hJe, Nat.mod_eq_of_lt hJf] at hm
    exact hm
  have htop : p0 * d.BLOCK_M + a.val = p0 * d.BLOCK_M + a'.val := by
    rw [hsum] at h'
    exact Nat.mul_right_cancel hD (Nat.add_right_cancel h')
  have hb : b.val = b'.val := by omega
  have haa : a.val = a'.val := by omega
  have e1 : a = a' := Fin.ext haa
  have e2 : b = b' := Fin.ext hb
  rw [e1, e2]
  cases u
  cases u'
  rfl

/-! ## Two programs write disjoint tiles -/

/-- The output tiles of two different programs are disjoint. -/
theorem wtile_disjoint {d : Inst} (hd : d.ok) {p0 p1 q0 q1 : Nat}
    (_hp0 : p0 < gT d) (_hp1 : p1 < gD d) (_hq0 : q0 < gT d) (_hq1 : q1 < gD d)
    (hne : p0 ≠ q0 ∨ p1 ≠ q1) : Disjoint (wtile d p0 p1) (wtile d q0 q1) := by
  rw [Set.disjoint_left]
  intro e he1 he2
  unfold wtile Buf.img at he1 he2
  obtain ⟨i, hi, hie⟩ := he1
  obtain ⟨j, hj, hje⟩ := he2
  obtain ⟨a, b, u⟩ := i
  obtain ⟨a', b', u'⟩ := j
  have hTe : p0 * d.BLOCK_M + a.val < d.T := by
    have hh := laneT_lt_T hi
    rw [laneT_val] at hh
    exact hh
  have hJe : p1 * d.BLOCK_DIM + b.val < d.D := by
    have hh := laneJ_lt_D hi
    rw [laneJ_val] at hh
    exact hh
  have hTf : q0 * d.BLOCK_M + a'.val < d.T := by
    have hh := laneT_lt_T hj
    rw [laneT_val] at hh
    exact hh
  have hJf : q1 * d.BLOCK_DIM + b'.val < d.D := by
    have hh := laneJ_lt_D hj
    rw [laneJ_val] at hh
    exact hh
  have hD : 0 < d.D := hd.D_pos
  have h' : (q0 * d.BLOCK_M + a'.val) * d.D + (q1 * d.BLOCK_DIM + b'.val)
      = (p0 * d.BLOCK_M + a.val) * d.D + (p1 * d.BLOCK_DIM + b.val) := by
    simp only [outIdx_val] at hje hie
    rw [hje, hie]
  have hJ : q1 * d.BLOCK_DIM + b'.val = p1 * d.BLOCK_DIM + b.val := by
    have hm := congrArg (fun x => x % d.D) h'
    rw [nat_mul_add_mod, nat_mul_add_mod, Nat.mod_eq_of_lt hJf, Nat.mod_eq_of_lt hJe] at hm
    exact hm
  have hT : q0 * d.BLOCK_M + a'.val = p0 * d.BLOCK_M + a.val := by
    rw [hJ] at h'
    exact Nat.mul_right_cancel hD (Nat.add_right_cancel h')
  have hq0 : q0 = p0 := tile_div a'.isLt a.isLt hT
  have hq1 : q1 = p1 := tile_div b'.isLt b.isLt hJ
  rcases hne with h | h
  · exact h hq0.symm
  · exact h hq1.symm

/-- Every element a program writes is one of the output's `T·D` elements. -/
theorem wtile_subset {d : Inst} (hd : d.ok) {p0 p1 : Nat} (_hp0 : p0 < gT d) (_hp1 : p1 < gD d) :
    wtile d p0 p1 ⊆ {e | e < d.T * d.D} := by
  intro e he
  unfold wtile Buf.img at he
  obtain ⟨i, hi, hie⟩ := he
  rw [← hie]
  show outIdx d p0 p1 i < d.T * d.D
  have hT := laneT_lt_T hi
  have hJ := laneJ_lt_D hi
  have hD := hd.D_pos
  rw [outIdx_eq]
  have h1 : laneT d p0 i * d.D + d.D ≤ d.T * d.D := by
    calc laneT d p0 i * d.D + d.D = (laneT d p0 i + 1) * d.D := by ring
      _ ≤ d.T * d.D := Nat.mul_le_mul_right d.D (by omega)
  omega

/-! ## The lane that owns an output element -/

/-- The lane of program `(t / BLOCK_M, j / BLOCK_DIM)` that owns output element `(t, j)`. -/
def ownerLane (d : Inst) (t j : Nat) (hM : 0 < d.BLOCK_M) (hD : 0 < d.BLOCK_DIM) : Lane d :=
  (⟨t % d.BLOCK_M, Nat.mod_lt _ hM⟩, ⟨j % d.BLOCK_DIM, Nat.mod_lt _ hD⟩, ())

/-- The owner lane of `(t, j)` has token `t`. -/
theorem owner_T {d : Inst} (hd : d.ok) {t j : Nat} :
    laneT d (t / d.BLOCK_M) (ownerLane d t j (block_m_pos hd) (block_d_pos hd)) = t := by
  have h := Nat.div_add_mod t d.BLOCK_M
  have h1 : laneT d (t / d.BLOCK_M) (ownerLane d t j (block_m_pos hd) (block_d_pos hd))
      = t / d.BLOCK_M * d.BLOCK_M + t % d.BLOCK_M := rfl
  rw [h1, Nat.mul_comm (t / d.BLOCK_M) d.BLOCK_M]
  exact h

/-- The owner lane of `(t, j)` has column `j`. -/
theorem owner_J {d : Inst} (hd : d.ok) {t j : Nat} :
    laneJ d (j / d.BLOCK_DIM) (ownerLane d t j (block_m_pos hd) (block_d_pos hd)) = j := by
  have h := Nat.div_add_mod j d.BLOCK_DIM
  have h1 : laneJ d (j / d.BLOCK_DIM) (ownerLane d t j (block_m_pos hd) (block_d_pos hd))
      = j / d.BLOCK_DIM * d.BLOCK_DIM + j % d.BLOCK_DIM := rfl
  rw [h1, Nat.mul_comm (j / d.BLOCK_DIM) d.BLOCK_DIM]
  exact h

/-- The owner lane of `(t, j)` is enabled in the program `(t / BLOCK_M, j / BLOCK_DIM)`. -/
theorem owner_en {d : Inst} (hd : d.ok) {t j : Nat} (ht : t < d.T) (hj : j < d.D) :
    enLane d (t / d.BLOCK_M) (j / d.BLOCK_DIM)
      (ownerLane d t j (block_m_pos hd) (block_d_pos hd)) = true := by
  have hT := owner_T (d := d) hd (t := t) (j := j)
  have hJ := owner_J (d := d) hd (t := t) (j := j)
  unfold enLane
  rw [hT, hJ]
  simp [ht, hj]

/-- The owner lane writes output element `t * D + j`. -/
theorem owner_out {d : Inst} (hd : d.ok) {t j : Nat} (_ht : t < d.T) (_hj : j < d.D) :
    outIdx d (t / d.BLOCK_M) (j / d.BLOCK_DIM)
      (ownerLane d t j (block_m_pos hd) (block_d_pos hd)) = t * d.D + j := by
  have hT := owner_T (d := d) hd (t := t) (j := j)
  have hJ := owner_J (d := d) hd (t := t) (j := j)
  unfold outIdx
  rw [hT, hJ]

/-- Distinct programs of the grid have distinct identifiers. -/
theorem pid_inj {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < (launch d).size)
    (hj : j < (launch d).size)
    (h0 : ((launch d).ctx i).pid 0 = ((launch d).ctx j).pid 0)
    (h1 : ((launch d).ctx i).pid 1 = ((launch d).ctx j).pid 1) : i = j := by
  have hT : 0 < gT d := gT_pos hd
  have hiT : i % gT d = j % gT d := by
    rw [← launch_pid0, ← launch_pid0]
    exact h0
  have hiD : i / gT d < gD d := by
    rw [launch_size] at hi
    rw [Nat.div_lt_iff_lt_mul hT]
    calc i < gT d * gD d := hi
      _ = gD d * gT d := Nat.mul_comm _ _
  have hjD : j / gT d < gD d := by
    rw [launch_size] at hj
    rw [Nat.div_lt_iff_lt_mul hT]
    calc j < gT d * gD d := hj
      _ = gD d * gT d := Nat.mul_comm _ _
  have hdiv : i / gT d = j / gT d := by
    have e := h1
    rw [launch_pid1, launch_pid1, Nat.mod_eq_of_lt hiD, Nat.mod_eq_of_lt hjD] at e
    exact e
  have hi1 : i = (i / gT d) * gT d + i % gT d := by
    rw [Nat.mul_comm]
    exact (Nat.div_add_mod i (gT d)).symm
  have hj1 : j = (j / gT d) * gT d + j % gT d := by
    rw [Nat.mul_comm]
    exact (Nat.div_add_mod j (gT d)).symm
  rw [hi1, hj1, hiT, hdiv]

end Claims.MoeSumReduce
