import TVA.Proof
import Claim

/-!
# The accuracy bookkeeping

`errAt` is the claim's per-chunk error recurrence; `Dot8Chunk.errM` is the library's.
They agree, and the chunk sums of the library agree with the claim's `partSum` and with
the sums over the `32`-element chunks of a sequence on `Nat`.
-/

namespace Claims.MatmulE5m2

open TVA
open scoped BigOperators

/-- The claim's partial sums are the library's `psum` of the chunk sums. -/
theorem psum_chunk (p : Nat → ℚ) (j : Nat) :
    Dot8Chunk.psum 0 (fun i => ∑ k : Fin 32, p (32 * i + k)) j = partSum p j := by
  induction j with
  | zero => simp [Dot8Chunk.psum, partSum]
  | succ j ih => rw [Dot8Chunk.psum_succ, ih, partSum_succ]

/-- The claim's error recurrence is the library's. -/
theorem errM_eq_errAt (p q : Nat → ℚ) (n : Nat) :
    Dot8Chunk.errM Device.rhoChunk8 Device.sigmaChunk8 Device.alphaChunk8 Device.kappaChunk8 Device.rhoLossChunk8 0
        (fun j => ∑ k : Fin 32, p (32 * j + k))
        (fun j => ∑ k : Fin 32, |p (32 * j + k)|) (fun j => chunkMax q j) n =
      errAt p q n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Dot8Chunk.errM, errAt_succ, ih, psum_chunk, chunkAbs]

/-- Once the chunks cover `K`, the claim's partial sum is the dot product. -/
theorem partSum_dotSum {K n : Nat} (va vb : Nat → F8E5) (hn : K ≤ 32 * n) :
    partSum (prodV va vb K) n = dotSum va vb K := by
  unfold partSum dotSum
  have h1 : (∑ e ∈ Finset.range (32 * n), prodV va vb K e) =
      ∑ e ∈ Finset.range K, (va e).toRat * (vb e).toRat := by
    simp only [prodV]
    rw [← Finset.sum_filter]
    congr 1
    ext e
    simp only [Finset.mem_filter, Finset.mem_range]
    omega
  exact h1

end Claims.MatmulE5m2
