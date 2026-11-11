import TVA.Proof.Dot.Chunk16

/-!
# Consequences of the sm_90 `tl.dot` facts

The device facts of `TVA.Device.Sm90`, `TVA.Device.Dot8Chunk` and `TVA.Device.Dot16Chunk` are
stated in the shape the hardware is measured in. A proof usually wants them in another shape:
an fp8 dot as a whole-dot bound, a 16-bit dot on normal values as a plain `DotAcc` fact, the
exact `int8` dot as an equation, or Triton's rewrite of `tl.sum(x[:, :, None] * y[None, :, :], 1)`
to an `ieee` `tl.dot` bounded together with the tree form. This file derives them from a
hypothesis `h : Device P`, so a proof writes `h.dotE5m2Lift p`, `h.dotI8_eq …`.

* `Device.dotE4m3Lift`, `dotE5m2Lift`, `dotE4m3E5m2Lift`, `dotE5m2E4m3Lift`: an fp8 `tl.dot` is
  `DotLiftAcc` with `Dot8Chunk.relLift8`, `absLift8` (from the chunk facts, `DotF8Chunk.liftAcc`),
  so `acc += tl.dot(a, b)` over `K` blocks of fp8 uses `DotLift.dotAdd_step` and `DotLoop.accum`.
* `Device.dotF16ChunkLift`, `dotBF16ChunkLift`: the 16-bit chunk facts as `DotLiftAcc` with
  `Dot16Chunk.relLift16`, `absLift16` (tighter than `relDot16` for large `K`).
* `Device.dotF16Acc`, `dotBF16Acc`: on zero and normal operands, `DotF16` and `DotBF16` are
  `DotAcc` facts (`DotLiftAcc.dotAcc`).
* `Device.dotI8_eq`: an `int8` dot in range is exact.
* `sum_bcast_mul_err`: `tl.sum_bcast_mul` on `float32`, whichever form the device runs (the tree
  of fusable additions, `sum_bcast_mul_tree_err`, or the `ieee` `tl.dot`, `Device.DotF32Ieee`),
  is within the larger of the two bounds.
* `Device.relDot16_nonneg`, `absDot16_nonneg`.
-/

namespace TVA

namespace Device

variable {P : Platform}

/-- `relDot16 K ≥ 0`. -/
theorem relDot16_nonneg (K : ℕ) : 0 ≤ relDot16 K := by unfold relDot16; positivity

/-- `absDot16 K ≥ 0`. -/
theorem absDot16_nonneg (K : ℕ) : 0 ≤ absDot16 K := by unfold absDot16; positivity

/-! ## fp8 and 16-bit dots as whole-dot bounds -/

/-- **A `float8e4nv` `tl.dot` as a whole-dot bound**: `DotLiftAcc` with `relLift8`, `absLift8`.
Use it with `DotLift.dotAdd_step` for an fp8 GEMM loop. -/
theorem dotE4m3Lift (h : Device P) (p : InputPrecision) :
    DotLiftAcc P ⟨.f8e4nv, .f8e4nv, .f32, p, 2 ^ 30⟩ Dot8Chunk.relLift8 Dot8Chunk.absLift8 :=
  DotF8Chunk.liftAcc h.dotE4m3Chunk p

/-- **A `float8e5` `tl.dot` as a whole-dot bound**: as `dotE4m3Lift`. -/
theorem dotE5m2Lift (h : Device P) (p : InputPrecision) :
    DotLiftAcc P ⟨.f8e5, .f8e5, .f32, p, 2 ^ 30⟩ Dot8Chunk.relLift8 Dot8Chunk.absLift8 :=
  DotF8Chunk.liftAcc h.dotE5m2Chunk p

/-- **A `float8e4nv` by `float8e5` `tl.dot` as a whole-dot bound**: as `dotE4m3Lift`. -/
theorem dotE4m3E5m2Lift (h : Device P) (p : InputPrecision) :
    DotLiftAcc P ⟨.f8e4nv, .f8e5, .f32, p, 2 ^ 30⟩ Dot8Chunk.relLift8 Dot8Chunk.absLift8 :=
  DotF8Chunk.liftAcc h.dotE4m3E5m2Chunk p

/-- **A `float8e5` by `float8e4nv` `tl.dot` as a whole-dot bound**: as `dotE4m3Lift`. -/
theorem dotE5m2E4m3Lift (h : Device P) (p : InputPrecision) :
    DotLiftAcc P ⟨.f8e5, .f8e4nv, .f32, p, 2 ^ 30⟩ Dot8Chunk.relLift8 Dot8Chunk.absLift8 :=
  DotF8Chunk.liftAcc h.dotE5m2E4m3Chunk p

/-- **A `float16` `tl.dot` from the chunk facts as a whole-dot bound**: `DotLiftAcc` with
`relLift16`, `absLift16`. Use it in place of `h.dotF16` when `K` is large. -/
theorem dotF16ChunkLift (h : Device P) (p : InputPrecision) :
    DotLiftAcc P ⟨.f16, .f16, .f32, p, 0⟩ Dot16Chunk.relLift16 Dot16Chunk.absLift16 :=
  Dot16BitChunk.liftAcc h.dotF16Chunk p

/-- **A `bfloat16` `tl.dot` from the chunk facts as a whole-dot bound**: as `dotF16ChunkLift`. -/
theorem dotBF16ChunkLift (h : Device P) (p : InputPrecision) :
    DotLiftAcc P ⟨.bf16, .bf16, .f32, p, 0⟩ Dot16Chunk.relLift16 Dot16Chunk.absLift16 :=
  Dot16BitChunk.liftAcc h.dotBF16Chunk p

/-- **`DotF16` on zero and normal operands** is a `DotAcc` fact with the same constants, relative
to `|acc| + Σ|a·b|`. Use it with `DotLoop.dotAdd_step` when a claim keeps operands normal. -/
theorem dotF16Acc (h : Device P) (p : InputPrecision) :
    DotAcc P ⟨.f16, .f16, .f32, p, 0⟩ (fun q => q = 0 ∨ liftFloor .f16 ≤ |q|) relDot16 absDot16 :=
  DotLiftAcc.dotAcc (h.dotF16 p)
    (fun q hq => hq.elim (fun h0 => by simp [h0]) liftMag_of_le)
    (fun q hq => hq.elim (fun h0 => by simp [h0]) liftMag_of_le)

/-- **`DotBF16` on zero and normal operands** is a `DotAcc` fact: as `dotF16Acc`. -/
theorem dotBF16Acc (h : Device P) (p : InputPrecision) :
    DotAcc P ⟨.bf16, .bf16, .f32, p, 0⟩ (fun q => q = 0 ∨ liftFloor .bf16 ≤ |q|) relDot16
      absDot16 :=
  DotLiftAcc.dotAcc (h.dotBF16 p)
    (fun q hq => hq.elim (fun h0 => by simp [h0]) liftMag_of_le)
    (fun q hq => hq.elim (fun h0 => by simp [h0]) liftMag_of_le)

/-! ## The exact `int8` dot -/

/-- **An `int8` `tl.dot` in range is exact**: when `|acc| + Σ|a·b|` fits in `int32`, the result
element's value is `acc + Σ a·b`. -/
theorem dotI8_eq (h : Device P) (p : InputPrecision) {M N K : Nat} {a : Tensor [M, K] I8}
    {b : Tensor [K, N] I8} {acc r : Tensor [M, N] I32}
    (hd : P.dot ⟨.i8, .i8, .i32, p, 0⟩ a b acc r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ)
    (vc : ℚ) (hva : ∀ k, DType.val? .i8 (a.get2 i k) = some (va k))
    (hvb : ∀ k, DType.val? .i8 (b.get2 k n) = some (vb k))
    (hvc : DType.val? .i32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| ≤ DType.bound .i32) :
    DType.val? .i32 (r.get2 i n) = some (vc + ∑ k, va k * vb k) := by
  obtain ⟨vr, hvr, herr⟩ := h.dotI8 p a b acc r hd i n va vb vc hva hvb
    (fun _ => ⟨trivial, trivial⟩) hvc (by simpa using hroom)
  simp only [zero_mul, add_zero, abs_nonpos_iff, sub_eq_zero] at herr
  rw [hvr, herr]

end Device

/-! ## `tl.sum_bcast_mul` on `float32` -/

/-- **`tl.sum(x[:, :, None] * y[None, :, :], axis=1)` on `float32`**, for one output element,
whichever form the device runs (`hr`, as `safe_sum_bcast_mul` gives it): the products rounded and
summed along a tree (`sum_bcast_mul_tree_err`), or Triton's rewrite to an `ieee` `tl.dot`
(`Device.DotF32Ieee`, pass `h.dotF32Ieee`). The result is finite and within the larger of the two
bounds, `bcastTreeBound K B` and `(K+1)·2^-23·B + (K+1)·2^-149`, `B = Σ|x·y|`. -/
theorem sum_bcast_mul_err {P : Platform} (hdot : Device.DotF32Ieee P) {fusion : Bool}
    {M K N : Nat} {x : Tensor [M, K] F32} {y : Tensor [K, N] F32} {r : Tensor [M, N] F32}
    (hr : (∀ i j, ∃ p : Nat → F32,
        (∀ k (hk : k < K), tl.MulRel .f32 (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
        ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd fusion .f32) p (r.get2 i j)) ∨
      (DType.bytes .f32 ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
        P.dot { a := .f32, b := .f32, acc := .f32, precision := .ieee } x y
          (Tensor.full (DType.ofBits .f32 0)) r))
    (i : Fin M) (j : Fin N)
    (hx : ∀ k, (x.get2 i k).isFinite = true) (hy : ∀ k, (y.get2 k j).isFinite = true)
    (hroom : 2 * ((1 + Format.binary32.u) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
        K * Format.binary32.eta) +
      2 * Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (K - 1) (K - 1)
        ((1 + Format.binary32.u) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
          K * Format.binary32.eta) + 4 * Format.binary32.eta ≤ Format.binary32.maxVal)
    (hroom' : ((K : ℚ) + 2) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
      (K + 1) * 2 ^ (-149 : ℤ) ≤ Format.binary32.maxVal) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - ∑ k, (x.get2 i k).toRat * (y.get2 k j).toRat| ≤
        max (bcastTreeBound K (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|))
          ((K + 1) * 2 ^ (-23 : ℤ) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
            (K + 1) * 2 ^ (-149 : ℤ)) := by
  have hu0 := Format.u_nonneg (f := Format.binary32)
  have he0 := Format.eta_nonneg (f := Format.binary32)
  set S := ∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat| with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  rcases hr with htree | ⟨-, -, -, hd⟩
  · obtain ⟨p, hp, t, hcov, ht⟩ := htree i j
    let a : Nat → F32 := fun k => if h : k < K then x.get2 i ⟨k, h⟩ else default
    let b : Nat → F32 := fun k => if h : k < K then y.get2 ⟨k, h⟩ j else default
    have hak : ∀ k (hk : k < K), a k = x.get2 i ⟨k, hk⟩ := fun k hk =>
      dite_eq_left_of_eq_true (eq_true hk)
    have hbk : ∀ k (hk : k < K), b k = y.get2 ⟨k, hk⟩ j := fun k hk =>
      dite_eq_left_of_eq_true (eq_true hk)
    have hsumab : ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum =
        ∑ k, (x.get2 i k).toRat * (y.get2 k j).toRat := by
      rw [sum_range_fin]
      exact Finset.sum_congr rfl fun k _ => by rw [hak k k.isLt, hbk k k.isLt]
    have habs : ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum = S := by
      rw [sum_range_fin]
      exact Finset.sum_congr rfl fun k _ => by rw [hak k k.isLt, hbk k k.isLt]
    have hErr0 : 0 ≤ Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (K - 1)
        (K - 1) ((1 + Format.binary32.u) * S + K * Format.binary32.eta) :=
      Tree.errBound_nonneg (by positivity) (by positivity) _ _ (by positivity)
    have hSmax : S ≤ Format.binary32.maxVal := by nlinarith
    have hab : ∀ k, k < K → |(a k).toRat * (b k).toRat| ≤ Format.binary32.maxVal := by
      intro k hk
      rw [hak k hk, hbk k hk]
      have := Finset.single_le_sum
        (f := fun k : Fin K => |(x.get2 i k).toRat * (y.get2 k j).toRat|)
        (fun _ _ => abs_nonneg _) (Finset.mem_univ ⟨k, hk⟩)
      exact this.trans hSmax
    obtain ⟨hfin, herr⟩ := sum_bcast_mul_tree_err a b p
      (fun k hk => by rw [hak k hk, hbk k hk]; exact hp k hk) hcov ht
      (fun k hk => by rw [hak k hk]; exact hx _) (fun k hk => by rw [hbk k hk]; exact hy _) hab
      (by rw [habs]; linarith)
    rw [hsumab, habs] at herr
    exact ⟨hfin, herr.trans (le_max_left _ _)⟩
  · have hz : DType.val? .f32
        ((Tensor.full (DType.ofBits .f32 0) : Tensor [M, N] F32).get2 i j) = some 0 :=
      DType.val?_zero_f32
    obtain ⟨vr, hvr, herr⟩ := hdot x y _ r hd i j (fun k => (x.get2 i k).toRat)
      (fun k => (y.get2 k j).toRat) 0 (fun k => DType.val?_f32.mpr ⟨hx k, rfl⟩)
      (fun k => DType.val?_f32.mpr ⟨hy k, rfl⟩) (fun _ => ⟨trivial, trivial⟩) hz (by
        simp only [abs_zero, zero_add, DType.bound]
        have h1 : ((K : ℚ) + 1) * 2 ^ (-23 : ℤ) ≤ K + 1 := by
          have : (2 : ℚ) ^ (-23 : ℤ) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)
          nlinarith [Nat.cast_nonneg (α := ℚ) K]
        nlinarith)
    obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hvr
    refine ⟨hfin, ?_⟩
    rw [hval]
    simp only [abs_zero, zero_add] at herr
    exact herr.trans (le_max_right _ _)

end TVA
