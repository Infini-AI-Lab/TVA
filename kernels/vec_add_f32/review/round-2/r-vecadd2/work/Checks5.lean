import VecAdd

/-! r-vecadd2, round 2, at gate 5f65eb1f7a961315 (filing 4): the library-round-23 changes this
claim reaches (L118's docstrings), the conditions against the host's launches, the map notes of
lines 44, 45 and 47 at the extremes of `Inst.ok`, the grid formula, and what `Post` asks of each
element. -/

open TVA Kernels.VecAdd

/-! ## L118: the docstrings of `I8` and `I32` as the library now states them -/

open Lean in
#eval show CoreM (Option String × Option String) from do
  let env ← getEnv
  return (← findDocString? env ``TVA.I8, ← findDocString? env ``TVA.I32)

/-! ## The conditions cover the host's launches

`add(x, y)` allocates a fresh `output = torch.empty_like(x)` and launches with `n = output.numel()`
and `BLOCK_SIZE = 1024`: any `n < 2^31`, three `float32` buffers of at least `n` elements, and an
output whose bytes meet neither input's. `y` may be `x` (`add(x, x)`). -/

theorem host_ok (n : Nat) (hn : n < 2 ^ 31) (x y o : Buf .f32) (hx : n ≤ x.len) (hy : n ≤ y.len)
    (ho : n ≤ o.len) (hox : Disjoint o.bytes x.bytes) (hoy : Disjoint o.bytes y.bytes) :
    Inst.ok ⟨n, 1024, x, y, o⟩ :=
  ⟨⟨10, by norm_num⟩, by norm_num, hn, hx, hy, ho,
    Buf.elems_disjoint_of o x hox (fun j hj => lt_of_lt_of_le hj ho) (fun j hj => lt_of_lt_of_le hj hx),
    Buf.elems_disjoint_of o y hoy (fun j hj => lt_of_lt_of_le hj ho) (fun j hj => lt_of_lt_of_le hj hy)⟩

/-- `add(x, x)`. -/
theorem host_ok_same (n : Nat) (hn : n < 2 ^ 31) (x o : Buf .f32) (hx : n ≤ x.len) (ho : n ≤ o.len)
    (hox : Disjoint o.bytes x.bytes) : Inst.ok ⟨n, 1024, x, x, o⟩ :=
  host_ok n hn x x o hx hx ho hox hox

/-- The script's sizes: `98432` (line 85) and `2^12 … 2^27` (line 110) are below `2^31`. -/
example : 98432 < 2 ^ 31 ∧ ∀ i, 12 ≤ i → i < 28 → 2 ^ i < 2 ^ 31 := by
  refine ⟨by norm_num, fun i _ hi => Nat.pow_lt_pow_right (by norm_num) (by omega)⟩

/-- `n = 2^31` is not covered (the README's narrowing). -/
example (B : Nat) (x y o : Buf .f32) : ¬ Inst.ok ⟨2 ^ 31, B, x, y, o⟩ := fun h => by
  have := h.n_lt; simp at this

/-- An in-place call (`output` the same buffer as `x`) with `n ≥ 1` is not covered. -/
example (n B : Nat) (hn : 1 ≤ n) (x y : Buf .f32) : ¬ Inst.ok ⟨n, B, x, y, x⟩ := fun h => by
  have hd := h.output_x
  have hmem : x.base ∈ x.elems {j | j < n} := ⟨0, by simp only [Set.mem_ofPred_eq]; omega, by
    simp [Mem.span, Buf.ptr, DType.bytes]⟩
  exact Set.disjoint_left.mp hd hmem hmem

/-- A `y` shorter than `n` is not covered (the README's narrowing: the host does not check it). -/
example (n B : Nat) (x y o : Buf .f32) (hy : y.len < n) : ¬ Inst.ok ⟨n, B, x, y, o⟩ := fun h => by
  have := h.y_len; simp at this; omega

/-! ## The map notes of lines 44, 45 and 47 -/

/-- Line 44: the constexpr `BLOCK_SIZE`, built in `int32` modulo `2^32` (`IntV.ofInt`), is the
model's `IntV.ofNat BLOCK_SIZE`. -/
example (B : Nat) : (IntV.ofNat B : I32) = IntV.ofInt (B : Int) := by
  simp [IntV.ofNat, IntV.ofInt]

/-- Line 47 at `n = 1`: the constant `1`, made an `int32` tensor, is the model's `IntV.ofNat 1`. -/
example : (IntV.ofNat 1 : I32) = IntV.ofInt 1 := by decide

-- Lines 44-45 at the largest covered instances: the last program's last lane, computed as the
-- model computes it (int32 product, int32 sum), is the true offset, with no wrap: `2^31 - 1` at
-- B = 2^20, n = 2^31 - 1 (grid 2048), and `2^31 - 2` at B = 1 (grid 2^31 - 1); Python agrees.
def lastLane (n B : Nat) : Int × Nat :=
  let g := (n + B - 1) / B
  (((IntV.ofNat (g - 1) : I32) * IntV.ofNat B + IntV.ofInt ((B : Int) - 1)).toInt, g)

#eval (lastLane (2 ^ 31 - 1) (2 ^ 20), lastLane (2 ^ 31 - 1) 1)

-- Line 47: the signed `int32` comparison on offsets and `n` in `[0, 2^31)` is the comparison of
-- the numbers (true, false, false, true, false, true).
#eval ((IntV.ofNat 5 : I32).lt (IntV.ofNat (2 ^ 31 - 1)), (IntV.ofNat (2 ^ 31 - 1) : I32).lt (IntV.ofNat 5),
  (IntV.ofNat 0 : I32).lt (IntV.ofNat 0), (IntV.ofNat 98431 : I32).lt (IntV.ofNat 98432),
  (IntV.ofNat 98432 : I32).lt (IntV.ofNat 98432), (IntV.ofNat (2 ^ 31 - 2) : I32).lt (IntV.ofNat (2 ^ 31 - 1)))

/-! ## The grid formula is `triton.cdiv(n_elements, BLOCK_SIZE)`

`launch`'s grid is `(n + B - 1) / B` on `Nat`, Python's `(n + B - 1) // B` for `n ≥ 0`, `B ≥ 1`.
Python gives 97, 4, 131072, 0, 1, 2147483647, 2048. -/
#eval (grid ⟨98432, 1024, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩, grid ⟨2 ^ 12, 1024, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩,
  grid ⟨2 ^ 27, 1024, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩, grid ⟨0, 1024, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩,
  grid ⟨1, 1024, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩, grid ⟨2 ^ 31 - 1, 1, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩,
  grid ⟨2 ^ 31 - 1, 2 ^ 20, ⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩⟩)

-- `launch` keeps the defaults the host's launch gives: debug false, fusion true, one CTA, no PDL,
-- and the one block shape `[BLOCK_SIZE]`.
example (d : Inst) : (launch d).debug = false ∧ (launch d).fusion = true ∧ (launch d).numCtas = 1 ∧
    (launch d).pdl = false ∧ (launch d).blocks = [[d.BLOCK_SIZE]] := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-! ## What `Post` asks of each element: the IEEE `float32` sum, up to NaN payload -/

def f (n : Nat) : F32 := FloatV.ofPat n
def addp (a b : Nat) : Nat := (FloatV.add (f a) (f b)).pat

-- 1 + 2^-23 exact (1065353217); a tie 1 + 2^-24 to even (1065353216); 1 + 3·2^-25 rounds up
-- (1065353217); two smallest subnormals (2); the largest finite twice is +inf (2139095040);
-- -0 + -0 is -0 (2147483648); +0 + -0 is +0 (0); 1 + 2^-10 exact (1065361408), which a
-- bfloat16-precision add would round to 1.0.
#eval (addp 0x3f800000 0x34000000, addp 0x3f800000 0x33800000, addp 0x3f800000 0x33c00000,
  addp 1 1, addp 0x7f7fffff 0x7f7fffff, addp 0x80000000 0x80000000, addp 0 0x80000000,
  addp 0x3f800000 0x3a800000)
-- inf + -inf and a NaN operand give NaN; `NaNEq` is `=` or both NaN, so it tells -0 from +0.
#eval ((FloatV.add (f 0x7f800000) (f 0xff800000)).isNaN, (FloatV.add (f 0x7fc00001) (f 0x3f800000)).isNaN,
  FloatV.add (f 0x80000000) (f 0x80000000) == f 0, (f 0).isNaN)
