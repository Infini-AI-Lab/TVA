import TVA.Proof
import Claim

/-!
# Crude bounds: the accumulator stays far below `2^128`

The dot product's `maxImpreciseAcc` hypothesis only needs the accumulator and the sum of
absolute products to be small compared with `binary32.maxVal`; with `K ≤ 2^11` and `e5m2`
values they are below `2^45`.
-/

namespace Claims.MatmulE5m2

open TVA
open scoped BigOperators

/-- `liftMag` of a value of magnitude at most `c` is at most `c`, when `c` is at least the dtype's
smallest normal magnitude. -/
private theorem liftMag_le_of {d : DType} {q c : ℚ}
    (hc : Device.liftFloor d ≤ c) (hq : |q| ≤ c) : Device.liftMag d q ≤ c := by
  unfold Device.liftMag
  split_ifs with h
  · exact (abs_nonneg q).trans hq
  · exact max_le hq hc

/-- `DotLoop.bound` is monotone in the number of steps. -/
private theorem bound_mono {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {L : ℚ} (hL : 0 ≤ L) :
    ∀ {t T : Nat}, t ≤ T → DotLoop.bound ρ α t L ≤ DotLoop.bound ρ α T L := by
  have step : ∀ s, DotLoop.bound ρ α s L ≤ DotLoop.bound ρ α (s + 1) L := by
    intro s
    have hp : (0:ℚ) ≤ (1 + ρ) ^ s := pow_nonneg (by linarith) s
    have hsρ : (0:ℚ) ≤ (s : ℚ) * ρ := mul_nonneg (Nat.cast_nonneg s) hρ
    have hbr : (0:ℚ) ≤ (s : ℚ) * ρ + 1 + ρ := by linarith
    have h1 : (0:ℚ) ≤ (1 + ρ) ^ s * ρ * L := mul_nonneg (mul_nonneg hp hρ) hL
    have h2 : (0:ℚ) ≤ (1 + ρ) ^ s * ((s : ℚ) * ρ + 1 + ρ) * α :=
      mul_nonneg (mul_nonneg hp hbr) hα
    have key : DotLoop.bound ρ α (s + 1) L - DotLoop.bound ρ α s L =
        (1 + ρ) ^ s * ρ * L + (1 + ρ) ^ s * ((s : ℚ) * ρ + 1 + ρ) * α := by
      unfold DotLoop.bound
      rw [pow_succ]
      push_cast
      ring
    linarith
  intro t T h
  induction T with
  | zero => rw [Nat.le_zero.mp h]
  | succ T ih =>
    rcases Nat.eq_or_lt_of_le h with he | hlt
    · rw [he]
    · exact (ih (by omega)).trans (step T)

/-- The accumulator and the absolute sum of products are far below `binary32.maxVal`. -/
theorem room {K : Nat} (hK : K ≤ 2 ^ 11) {vc : ℚ} (hvc : |vc| ≤ 2 ^ 44)
    {va vb : Fin K → ℚ} (ha : ∀ k, |va k| ≤ 57344) (hb : ∀ k, |vb k| ≤ 57344) :
    |vc| + (∑ k : Fin K, |va k * vb k|) +
        (Device.relDot8 K * (|vc| + ∑ k : Fin K, |va k * vb k|) + Device.absDot8 K) ≤
      DType.bound .f32 := by
  have hterm : ∀ k : Fin K, |va k * vb k| ≤ (2 ^ 32 : ℚ) := by
    intro k
    rw [abs_mul]
    have h1 := ha k
    have h2 := hb k
    have h3 : |va k| * |vb k| ≤ (57344 : ℚ) * 57344 :=
      mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    have h4 : (57344 : ℚ) * 57344 ≤ 2 ^ 32 := by norm_num
    linarith
  have hS : (∑ k : Fin K, |va k * vb k|) ≤ (2 ^ 43 : ℚ) := by
    have hle : (∑ k : Fin K, |va k * vb k|) ≤ ∑ _k : Fin K, (2 ^ 32 : ℚ) :=
      Finset.sum_le_sum (fun k _ => hterm k)
    have heq : (∑ _k : Fin K, (2 ^ 32 : ℚ)) = (K : ℚ) * 2 ^ 32 := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hKq : (K : ℚ) ≤ 2 ^ 11 := by exact_mod_cast hK
    have h32 : (0:ℚ) ≤ 2 ^ 32 := by norm_num
    have hstep : (K : ℚ) * 2 ^ 32 ≤ (2 ^ 11 : ℚ) * 2 ^ 32 :=
      mul_le_mul_of_nonneg_right hKq h32
    have hfin : (2 ^ 11 : ℚ) * 2 ^ 32 = 2 ^ 43 := by norm_num
    linarith
  have hchunks : Device.chunks8 K ≤ 64 := by unfold Device.chunks8; omega
  have hx7 : (0:ℚ) ≤ 2 ^ (-7 : ℤ) := by norm_num
  have hbase : (1:ℚ) ≤ 1 + 2 ^ (-7 : ℤ) := by norm_num
  have hnum7 : 2 * ((64:ℕ):ℚ) * 2 ^ (-7:ℤ) = 1 := by norm_num
  have hrel : Device.relDot8 K ≤ 1 := by
    unfold Device.relDot8
    have hpl := DotLoop.pow_le hx7 64 (by norm_num)
    have hmono := pow_le_pow_right₀ hbase hchunks
    linarith
  have habs : Device.absDot8 K ≤ 1 := by
    unfold Device.absDot8
    have hpl := DotLoop.pow_le hx7 64 (by norm_num)
    have hmono := pow_le_pow_right₀ hbase hchunks
    have hp2 : (1 + (2:ℚ) ^ (-7:ℤ)) ^ Device.chunks8 K ≤ 2 := by linarith
    have hck : (Device.chunks8 K : ℚ) ≤ 64 := by exact_mod_cast hchunks
    have hp0 : (0:ℚ) ≤ (1 + (2:ℚ) ^ (-7:ℤ)) ^ Device.chunks8 K := by positivity
    have h2 : (Device.chunks8 K : ℚ) * (1 + (2:ℚ) ^ (-7:ℤ)) ^ Device.chunks8 K ≤ 64 * 2 :=
      mul_le_mul hck hp2 hp0 (by norm_num)
    have h3 : (Device.chunks8 K : ℚ) * (1 + (2:ℚ) ^ (-7:ℤ)) ^ Device.chunks8 K *
        2 ^ (-126:ℤ) ≤ (64 * 2) * 2 ^ (-126:ℤ) :=
      mul_le_mul_of_nonneg_right h2 (by positivity)
    have h4 : (64:ℚ) * 2 * 2 ^ (-126:ℤ) ≤ 1 := by norm_num
    linarith
  have hS0 : (0:ℚ) ≤ ∑ k : Fin K, |va k * vb k| := Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  have hmag0 : (0:ℚ) ≤ |vc| + ∑ k : Fin K, |va k * vb k| := by
    have := abs_nonneg vc; linarith
  have hm : |vc| + ∑ k : Fin K, |va k * vb k| ≤ (2 ^ 44 : ℚ) + 2 ^ 43 := by linarith
  have e1 : Device.relDot8 K * (|vc| + ∑ k : Fin K, |va k * vb k|) ≤
      1 * (|vc| + ∑ k : Fin K, |va k * vb k|) := mul_le_mul_of_nonneg_right hrel hmag0
  have hbound : DType.bound .f32 = ((2 ^ 24 - 1) * 2 ^ 104 : ℚ) := by
    show Format.binary32.maxVal = ((2 ^ 24 - 1) * 2 ^ 104 : ℚ)
    exact Format.binary32_maxVal
  have hfin : 2 * ((2 ^ 44 : ℚ) + 2 ^ 43) + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
  rw [hbound]
  nlinarith [hm, e1, habs, hmag0]

/-- The partial sums and their error stay below `2^44`. -/
theorem acc_bound {K : Nat} (hK : K ≤ 2 ^ 11) {va vb : Nat → F8E5}
    (ha : ∀ k, k < K → |(va k).toRat| ≤ 57344) (hb : ∀ k, k < K → |(vb k).toRat| ≤ 57344) :
    ∀ n ≤ 2 * steps K,
      |partSum (prodV va vb K) n| + errAt (prodV va vb K) (liftV va vb K) n ≤ 2 ^ 44 := by
  intro n hn
  have hT : 2 * steps K ≤ 64 := by unfold steps; omega
  have hsteps : steps K ≤ 32 := by unfold steps; omega
  have hterm : ∀ e, |prodV va vb K e| ≤ (2 ^ 32 : ℚ) := by
    intro e
    unfold prodV
    split_ifs with he
    · rw [abs_mul]
      have h1 := ha e he
      have h2 := hb e he
      have h3 : |(va e).toRat| * |(vb e).toRat| ≤ (57344 : ℚ) * 57344 :=
        mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
      have h4 : (57344 : ℚ) * 57344 ≤ 2 ^ 32 := by norm_num
      linarith
    · simp
  have hminN : Device.liftFloor DType.f8e5 ≤ (57344 : ℚ) := by
    have h : Device.liftFloor DType.f8e5 = (2 ^ (-14 : ℤ) : ℚ) := rfl
    rw [h]; norm_num
  have hlift : ∀ e, liftV va vb K e ≤ (2 ^ 32 : ℚ) := by
    intro e
    unfold liftV
    split_ifs with he
    · have h1 : Device.liftMag DType.f8e5 (va e).toRat ≤ 57344 :=
        liftMag_le_of hminN (ha e he)
      have h2 : Device.liftMag DType.f8e5 (vb e).toRat ≤ 57344 :=
        liftMag_le_of hminN (hb e he)
      have h3 : Device.liftMag DType.f8e5 (va e).toRat *
          Device.liftMag DType.f8e5 (vb e).toRat ≤ (57344 : ℚ) * 57344 :=
        mul_le_mul h1 h2 (Device.liftMag_nonneg _ _) (by norm_num)
      have h4 : (57344 : ℚ) * 57344 ≤ 2 ^ 32 := by norm_num
      linarith
    · simp
  have hS : ∀ j, 32 * j ≤ 2048 → |partSum (prodV va vb K) j| ≤ (2 ^ 43 : ℚ) := by
    intro j hj
    unfold partSum
    have h1 : |∑ e ∈ Finset.range (32 * j), prodV va vb K e| ≤
        ∑ e ∈ Finset.range (32 * j), |prodV va vb K e| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ e ∈ Finset.range (32 * j), |prodV va vb K e| ≤
        ∑ _e ∈ Finset.range (32 * j), (2 ^ 32 : ℚ) := Finset.sum_le_sum (fun e _ => hterm e)
    have h3 : (∑ _e ∈ Finset.range (32 * j), (2 ^ 32 : ℚ)) =
        ((32 * j : Nat) : ℚ) * 2 ^ 32 := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    have hq : ((32 * j : Nat) : ℚ) ≤ 2048 := by exact_mod_cast hj
    have h32 : (0:ℚ) ≤ 2 ^ 32 := by norm_num
    have h4 : ((32 * j : Nat) : ℚ) * 2 ^ 32 ≤ (2048 : ℚ) * 2 ^ 32 :=
      mul_le_mul_of_nonneg_right hq h32
    have h5 : (2048 : ℚ) * 2 ^ 32 = 2 ^ 43 := by norm_num
    linarith
  have hq : ∀ j, chunkMax (liftV va vb K) j ≤ (2 ^ 43 : ℚ) := by
    intro j
    apply chunkMax_le
    intro k
    have h := hlift (32 * j + k)
    have h2 : (2 ^ 32 : ℚ) ≤ 2 ^ 43 := by norm_num
    linarith
  have hL0 : (0:ℚ) ≤ 2 ^ 43 := by norm_num
  have hSn : ∀ j ≤ n, |partSum (prodV va vb K) j| ≤ (2 ^ 43 : ℚ) := by
    intro j hj
    exact hS j (by omega)
  have hqn : ∀ j < n, chunkMax (liftV va vb K) j ≤ (2 ^ 43 : ℚ) := fun j _ => hq j
  have hρ : (0:ℚ) ≤ Device.rhoChunk8 + Device.sigmaChunk8 := by
    unfold Device.rhoChunk8 Device.sigmaChunk8; positivity
  have hα : (0:ℚ) ≤ Device.alphaChunk8 := by
    unfold Device.alphaChunk8; positivity
  have e1 := errAt_le_bound (prodV va vb K) (liftV va vb K) hSn hqn n le_rfl
  have e2 : DotLoop.bound (Device.rhoChunk8 + Device.sigmaChunk8) Device.alphaChunk8 n
        (2 ^ 43) ≤
      DotLoop.bound (Device.rhoChunk8 + Device.sigmaChunk8) Device.alphaChunk8
        (2 * steps K) (2 ^ 43) :=
    bound_mono hρ hα hL0 hn
  have e3 : DotLoop.bound (Device.rhoChunk8 + Device.sigmaChunk8) Device.alphaChunk8
        (2 * steps K) (2 ^ 43) ≤ 7 / 10 * (2 ^ 43) + 1 / 10 ^ 35 :=
    tolA_le (K := K) hK (L := (2 ^ 43 : ℚ)) hL0
  have e4 : |partSum (prodV va vb K) n| ≤ (2 ^ 43 : ℚ) := hSn n le_rfl
  have herr : errAt (prodV va vb K) (liftV va vb K) n ≤ 7 / 10 * (2 ^ 43) + 1 / 10 ^ 35 :=
    e1.trans (e2.trans e3)
  have hsum : |partSum (prodV va vb K) n| + errAt (prodV va vb K) (liftV va vb K) n ≤
      (2 ^ 43 : ℚ) + (7 / 10 * 2 ^ 43 + 1 / 10 ^ 35) := add_le_add e4 herr
  have hfin : (2 ^ 43 : ℚ) + (7 / 10 * 2 ^ 43 + 1 / 10 ^ 35) ≤ 2 ^ 44 := by norm_num
  exact hsum.trans hfin

end Claims.MatmulE5m2
