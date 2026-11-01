import TVA

/-! lr-24b: round 24's lemmas and facts. Axioms of the new theorems, and a kernel-style use of a chunk
fact: a single-chunk fp8 `tl.dot` (`K = 32`) from `DotE5m2Chunk`, with the bound written out. -/

open TVA TVA.Rules.Num TVA.Math.Float

#print axioms TVA.Dot8Chunk.DotF8Chunk.err
#print axioms TVA.Dot8Chunk.DotChunkAcc.err
#print axioms TVA.Dot8Chunk.DotF8Chunk.liftAcc
#print axioms TVA.Dot8Chunk.DotChunkAcc.liftErr
#print axioms TVA.Dot8Chunk.errB_le
#print axioms TVA.Dot8Chunk.chain_err
#print axioms TVA.Dot8Chunk.liftMag_le
#print axioms TVA.Platform.sm90ref.dotE5m2Chunk
#print axioms TVA.Platform.ref.dotChunkAcc
#print axioms TVA.Rules.Num.DotChunkAcc.mono

/-- A single-chunk fp8 dot (`K = 32`, one chunk): the result is within
`2^-7·max(|vc|, m) + 2^-12·|vc + Σ a·b| + 2^-126` of `vc + Σ a·b`, for any `m` bounding the lifted
products, from `DotE5m2Chunk` alone. -/
example (P : Platform) (h : DotE5m2Chunk P) (p : InputPrecision) {M N : ℕ}
    {a : Tensor [M, 32] F8E5} {b : Tensor [32, N] F8E5} {acc r : Tensor [M, N] F32}
    (hd : P.dot ⟨.f8e5, .f8e5, .f32, p, 2 ^ 30⟩ a b acc r) (i : Fin M) (n : Fin N)
    (va vb : Fin 32 → ℚ) (vc : ℚ)
    (hva : ∀ k, DType.val? .f8e5 (a.get2 i k) = some (va k))
    (hvb : ∀ k, DType.val? .f8e5 (b.get2 k n) = some (vb k))
    (hvc : DType.val? .f32 (acc.get2 i n) = some vc)
    (hroom : |vc| + ∑ k, |va k * vb k| + (relDot8 32 * (|vc| + ∑ k, |va k * vb k|) + absDot8 32) ≤
      DType.bound .f32)
    (m : ℚ) (hm : ∀ k, liftMag .f8e5 (va k) * liftMag .f8e5 (vb k) ≤ m) :
    ∃ vr, DType.val? .f32 (r.get2 i n) = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤
        2 ^ (-7 : ℤ) * max |vc| m + 2 ^ (-12 : ℤ) * |vc + ∑ k, va k * vb k| + 2 ^ (-126 : ℤ) := by
  obtain ⟨vr, hvr, herr⟩ := Dot8Chunk.DotF8Chunk.err h p hd i n va vb vc hva hvb hvc hroom
    (fun _ => m) (fun _ _ k _ => hm k)
  refine ⟨vr, hvr, ?_⟩
  have h1 : chunks8 32 = 1 := rfl
  rw [h1] at herr
  have hs : Dot8Chunk.psum vc (chunkSum8 va vb) 1 = vc + ∑ k, va k * vb k := by
    have := Dot8Chunk.psum_chunks8 (K := 32) vc va vb
    rwa [h1] at this
  simp only [Dot8Chunk.errB, Dot8Chunk.psum_zero, hs, mul_zero, zero_add] at herr
  simpa [rhoChunk8, sigmaChunk8, alphaChunk8] using herr
