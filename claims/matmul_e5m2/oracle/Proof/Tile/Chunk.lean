import Proof.Tile.Acc1
import Proof.Room

/-!
# One `tl.dot` of the `K`-loop

A trip of the loop reads a `128 × 64` panel of `a` and a `64 × 256` panel of `b` and adds its
product into the accumulator.  This file bridges the library's loss law for one `tl.dot`
(`Device.DotChunkLoss`, with the trip's panel `K = 64`, chunked `2` ways of `32`) to the
claim's `Steps` sequence: the partial values extend by the trip's two chunks, and the room the law
needs is `Room.room`.
-/

namespace Claims.MatmulE5m2

open TVA
open scoped BigOperators

/-- **One trip.**  If the accumulator holds the partial value `S (2·t)` before the trip, whose
chunk laws hold, then after the `tl.dot` the accumulator holds `S' (2·(t+1))` for a sequence `S'`
that extends `S` and whose chunk laws hold up to chunk `2·(t+1)`. -/
theorem dot_step {P : Platform} {prec : InputPrecision} {K : Nat}
    (hK : K ≤ 2 ^ 11) (hK64 : 64 ∣ K) {va vb : Nat → F8E5}
    (ha : ∀ k, k < K → |(va k).toRat| ≤ 57344)
    (hb : ∀ k, k < K → |(vb k).toRat| ≤ 57344)
    (hloss : Device.DotChunkLoss P ⟨.f8e5, .f8e5, .f32, prec, 2 ^ 30⟩
      Device.rhoChunk8 Device.sigmaChunk8 Device.alphaChunk8
      Device.kappaChunk8 Device.rhoLossChunk8)
    {a : Tensor [128, 64] F8E5} {b : Tensor [64, 256] F8E5}
    {acc r : Tensor [128, 256] F32}
    (hd : P.dot ⟨.f8e5, .f8e5, .f32, prec, 2 ^ 30⟩ a b acc r)
    (x : Fin 128) (y : Fin 256) {t : Nat} (ht : t < steps K)
    (hva : ∀ k : Fin 64, DType.val? .f8e5 (a.get2 x k) = some ((va (64 * t + k.val)).toRat))
    (hvb : ∀ k : Fin 64, DType.val? .f8e5 (b.get2 k y) = some ((vb (64 * t + k.val)).toRat))
    {S : Nat → ℚ} (hS0 : S 0 = 0)
    (hS : Steps (prodV va vb K) (liftV va vb K) S (2 * t))
    (hacc : DType.val? .f32 (acc.get2 x y) = some (S (2 * t))) :
    ∃ S' : Nat → ℚ, (∀ j ≤ 2 * t, S' j = S j) ∧ S' 0 = 0 ∧
      Steps (prodV va vb K) (liftV va vb K) S' (2 * (t + 1)) ∧
      DType.val? .f32 (r.get2 x y) = some (S' (2 * (t + 1))) := by
  have hch : Device.chunks8 64 = 2 := by unfold Device.chunks8; norm_num
  have key : ∀ v : Nat → F8E5, (∀ k, k < K → |(v k).toRat| ≤ 57344) →
      ∀ k : Fin 64, |(v (64 * t + k.val)).toRat| ≤ 57344 := by
    intro v hv k
    apply hv
    have hj32 : (k : Nat) / 32 < 2 := by omega
    have hr32 : (k : Nat) % 32 < 32 := by omega
    have hseg := lt_of_seg hK64 ht hj32 hr32
    have hk : 64 * t + (k : Nat) = 64 * t + 32 * ((k : Nat) / 32) + (k : Nat) % 32 := by
      have := Nat.div_add_mod (k : Nat) 32
      omega
    omega
  have hroom := room (K := 64) (by norm_num)
    (acc_le hK hK64 ha hb (by omega) hS0 hS) (key va ha) (key vb hb)
  obtain ⟨s, vr, hvr, hs0, hsn, hstep⟩ :=
    hloss (a := a) (b := b) (acc := acc) (r := r) hd x y
      (fun k => (va (64 * t + k.val)).toRat) (fun k => (vb (64 * t + k.val)).toRat)
      (S (2 * t)) hva hvb hacc hroom
  rw [hch] at hstep hsn
  obtain ⟨S', hmono, hlast, hSteps⟩ := trip hK hK64 ht hS hs0 hstep
  exact ⟨S', hmono, by rw [hmono 0 (by omega), hS0], hSteps, by rw [hlast, hsn]; exact hvr⟩

end Claims.MatmulE5m2
