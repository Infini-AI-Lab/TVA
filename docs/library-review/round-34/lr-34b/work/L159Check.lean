import TVA
/-! lr-34b, round 34, L159: `sum_bcast_mul_depth_err` applied as a kernel prover would, from the
pick of `tl.sumWith fusion` (a tree with `t.depth ≤ tl.sumBound P n`) and `Rules.Num.SumDepth P`,
on the products of a model's `tl.mul` held as a tensor `x`; the bound at `D = Nat.clog 2 n`. -/
open TVA TVA.Math.Float

example {P : Platform} (hP : Rules.Num.SumDepth P) {fusion : Bool} {n : Nat}
    (a b : Nat → F32) (x : Tensor [n] F32) {t : Tree} {r : F32}
    (hx : ∀ k (h : k < n), tl.MulRel .f32 (a k) (b k) (x.get1 ⟨k, h⟩))
    (hcov : t.Covers n) (hdep : t.depth ≤ tl.sumBound P n)
    (ht : t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r)
    (ha : ∀ k, k < n → (a k).isFinite = true) (hb : ∀ k, k < n → (b k).isFinite = true)
    (hab : ∀ k, k < n → |(a k).toRat * (b k).toRat| ≤ Format.binary32.maxVal)
    (hM : ((1 + Format.binary32.u) * ((List.range n).map fun k => |(a k).toRat * (b k).toRat|).sum +
        n * Format.binary32.eta) +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (Nat.clog 2 n) (n - 1)
        ((1 + Format.binary32.u) * ((List.range n).map fun k => |(a k).toRat * (b k).toRat|).sum +
          n * Format.binary32.eta) ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun k => (a k).toRat * (b k).toRat).sum| ≤
        bcastDepthBound (Nat.clog 2 n) n ((List.range n).map fun k => |(a k).toRat * (b k).toRat|).sum :=
  sum_bcast_mul_depth_err (D := Nat.clog 2 n) a b (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default)
    (fun k hk => by simpa [hk] using hx k hk) hcov (Rules.Num.SumDepth.depth_le hP hdep) ht ha hb hab hM

/-- At depth `K − 1` it is the old tree bound. -/
example (K : Nat) (B : ℚ) : bcastTreeBound K B = bcastDepthBound (K - 1) K B := bcastTreeBound_eq K B

#print axioms sum_bcast_mul_depth_err
