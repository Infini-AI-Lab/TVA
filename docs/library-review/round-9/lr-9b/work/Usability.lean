import TVA

/-!
lr-9b, library review round 9: can a kernel apply the round's lemmas and facts as stated?
Each `example` is a step a kernel proof would take. Checked with `lake env lean` (no library
edit). Results are in `usability.out`.
-/

open TVA TVA.Rules.Num TVA.Math.Float

namespace LR9b

/-! ## U1. `replay_add_any` at concrete integer dtypes (instance search on `DType.Val`) -/

example {a : Int} {inc : AtomicStep → DType.Val .i32} {m m' : Mem} {l : List AtomicStep}
    {x : DType.Val .i32} (hrep : Replay m l m') (hl : ∀ s ∈ l, MapsCell .i32 a (· + inc s) s)
    (h0 : m.read .i32 a = some x) : m'.read .i32 a = some (x + (l.map inc).sum) :=
  replay_add_any hrep hl h0

example {a : Int} {inc : AtomicStep → DType.Val .u64} {m m' : Mem} {l : List AtomicStep}
    {x : DType.Val .u64} (hrep : Replay m l m') (hl : ∀ s ∈ l, MapsCell .u64 a (· + inc s) s)
    (h0 : m.read .u64 a = some x) : m'.read .u64 a = some (x + (l.map inc).sum) :=
  replay_add_any hrep hl h0

/-! ## U2. The step a `tl.atomic_add` on `int32` logs maps the cell by `+ v` -/

example {E : Env} {a : Int} {old v new : I32} (h : tl.addUpd E .i32 v old new) :
    MapsCell .i32 a (· + v) ⟨.i32, a, old, new⟩ :=
  mapsCell_self (g := (· + v)) (by simp only [tl.addUpd] at h; exact h.symm)

/-! ## U3. A float `atomic_add` step from the fact `AtomicAddF32` -/

example {P : Platform} (hacc : AtomicAddF32 P) {a : Int} {old v new : F32}
    (hnew : P.atomicAdd .binary32 old v new) (hv : v.isFinite = true) :
    StepRel .f32 a (AddsApprox FloatV.toRat (fun x : F32 => x.isFinite = true) (2 ^ (-23 : ℤ))
      (2 ^ (-124 : ℤ)) Format.binary32.maxVal v.toRat) ⟨.f32, a, old, new⟩ :=
  stepRel_atomicAdd hacc hnew hv

/-! ## U4. `replay_err` over a log with the fact's constants -/

example {a : Int} (w : AtomicStep → ℚ) {m m' : Mem} {l : List AtomicStep} {x : F32}
    (hrep : Replay m l m')
    (hl : ∀ s ∈ l, StepRel .f32 a (AddsApprox FloatV.toRat (fun x : F32 => x.isFinite = true)
      (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) Format.binary32.maxVal (w s)) s)
    (h0 : m.read .f32 a = some x) (hx : x.isFinite = true)
    (hroom : |x.toRat| + (l.map fun s => |w s|).sum +
      DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) l.length (|x.toRat| + (l.map fun s => |w s|).sum)
        ≤ Format.binary32.maxVal) :
    ∃ y : F32, m'.read .f32 a = some y ∧ y.isFinite = true ∧
      |y.toRat - (x.toRat + (l.map w).sum)| ≤
        DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) l.length (|x.toRat| + (l.map fun s => |w s|).sum) :=
  replay_err (d := .f32) FloatV.toRat (fun x : F32 => x.isFinite = true) (by positivity) (by positivity)
    w hrep hl h0 hx hroom

/-! ## U5. Two logged steps that are equal as values must get the same weight `w`.
The log records the cell, the old and the new value, not the operand: two programs adding `1`
and `2` to a cell holding `2^30` (both absorbed) log the same step. -/

example (s₁ s₂ : AtomicStep) (h : s₁ = s₂) (w : AtomicStep → ℚ) : w s₁ = w s₂ := by rw [h]

/-! ## U6. `sum_bcast_mul_err` consumes what `safe_sum_bcast_mul` gives -/

example {E : Env} {M K N : Nat} {x : Tensor [M, K] F32} {y : Tensor [K, N] F32} {r : Tensor [M, N] F32}
    (hdot : DotF32Ieee E.P)
    (hr : (∀ i j, ∃ p : Nat → DType.Val .f32,
        (∀ k (hk : k < K), tl.MulRel .f32 (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
        ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd E.fusion .f32) p (r.get2 i j)) ∨
      (DType.bytes .f32 ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
        E.P.dot { a := .f32, b := .f32, acc := .f32, precision := .ieee } x y
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
    (r.get2 i j).isFinite = true := (sum_bcast_mul_err hdot hr i j hx hy hroom hroom').1

/-! ## U7. `sumc_err_covers` consumes what `safe_sumc` gives -/

example {E : Env} {n : Nat} {x : Tensor [n] F32} {r : F32}
    (h : ∃ t : Tree, t.Covers n ∧
      t.Eval (AddRel E.fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r)
    (hx : ∀ i, i < n → ((fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) i : F32).isFinite = true)
    (hM : ((List.range n).map fun i => |((fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) i : F32).toRat|).sum +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (n - 1) (n - 1)
        ((List.range n).map fun i => |((fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) i : F32).toRat|).sum ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta) :
    r.isFinite = true := by
  obtain ⟨t, hcov, ht⟩ := h
  exact (sumc_err_covers Format.binary32_regular rfl (by norm_num [Format.u, Format.binary32])
    E.fusion _ hcov ht hx hM).1

/-! ## U8. One platform satisfies the cited facts and `Compiler.Faithful` together -/

structure Assm {Inst : Type} (model : Inst → Kernel) (launch : Inst → Launch)
    (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  faithful : ∀ d, Rules.Compiler.Faithful P (model d) (launch d) (K d)
  exp2 : Exp2F32 P
  exp2_ninf : Exp2F32NegInf P
  exp : ExpF32 P
  log2 : Log2F32 P
  sqrt : SqrtF32 P
  div : DivFull P
  div_total : DivTotal P .binary32
  dot16 : DotF16 P
  dotbf : DotBF16 P
  dot8 : DotE5m2 P
  dot32 : DotF32Ieee P
  cvt : CvtF32E5m2 P
  scaled : DotScaledE4m3E2m1 P
  atomic : AtomicAddF32 P
  atomic_total : AtomicAddTotal P .binary32
  math_total : MathTotal P .exp2 .binary32

example {Inst : Type} (model : Inst → Kernel) (launch : Inst → Launch) :
    Claim.Consistent (Assm model launch) := by
  obtain ⟨K, hK, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, hK, ⟨hf, Platform.sm90ref.exp2F32, Platform.sm90ref.exp2F32NegInf,
    Platform.sm90ref.expF32, Platform.sm90ref.log2F32, Platform.sm90ref.sqrtF32,
    Platform.sm90ref.divFull, Platform.ref.divTotal _, Platform.sm90ref.dotF16,
    Platform.sm90ref.dotBF16, Platform.sm90ref.dotE5m2, Platform.sm90ref.dotF32Ieee,
    Platform.sm90ref.cvtF32E5m2, Platform.sm90ref.dotScaledE4m3E2m1, Platform.sm90ref.atomicAddF32,
    Platform.ref.atomicAddTotal _, Platform.ref.mathTotal⟩⟩

/-! ## U9. `ExpF32` at a kernel's input bound, weakened with `MathAcc.mono` -/

example {P : Platform} (h : ExpF32 P) :
    MathAcc P .exp .binary32 (Set.Icc (-100) 0) (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 100) (2 ^ (-140 : ℤ)) :=
  (h 100 (by norm_num)).mono (fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩)
    (fun _ _ _ _ _ => le_rfl)

/-! ## U10. `negInfF32` (the facts) and `Softmax.negInf32` (the softmax lemmas) are the same value -/

example : Rules.Num.negInfF32 = Softmax.negInf32 := rfl

/-! ## U11. A `tl.dot` of `float8e5` operands, written with the field defaults, is not the
configuration of `DotE5m2` (whose `maxImpreciseAcc` is `2^30`). -/

example : ({ a := .f8e5, b := .f8e5, acc := .f32 } : DotCfg) ≠ ⟨.f8e5, .f8e5, .f32, .tf32, 2 ^ 30⟩ := by
  decide

end LR9b
