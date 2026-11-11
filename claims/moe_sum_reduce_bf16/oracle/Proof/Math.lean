import Claim
import TVA.Proof
import Proof.Basic

/-!
# The numbers

The rational-number side of the proof: the `float32` unit roundoff and its constants (`u32`,
`eta32`), the error of a chain of `K` additions (`foldF_bound`, `foldF_err`), the scale's
conversion (`sigma32_rel`), and the arithmetic showing the total error is within `tol`
(`Fx_le_tol`). Also the facts on NaN payloads that the final value needs.
-/

namespace Claims.MoeSumReduce

open TVA

set_option maxHeartbeats 1000000

/-- `float32`'s unit roundoff, `2^-24` (`Format.binary32_u`). -/
def u32 : ℚ := 2 ^ (-24 : ℤ)

/-- `float32`'s minimum subnormal, `2^-150` (`Format.binary32_eta`). -/
def eta32 : ℚ := 2 ^ (-150 : ℤ)

/-- `bfloat16`'s unit roundoff, `2^-8` (`Format.bfloat16_u`). -/
def ubf : ℚ := 2 ^ (-8 : ℤ)

/-- `bfloat16`'s minimum subnormal, `2^-134` (`Format.bfloat16_eta`). -/
def etabf : ℚ := 2 ^ (-134 : ℤ)

/-- `float32`'s least normal magnitude, `2^-126`. -/
def mu32 : ℚ := 2 ^ (-126 : ℤ)

/-- `u32` is `binary32`'s unit roundoff. -/
theorem u32_eq : u32 = Format.binary32.u := Format.binary32_u.symm
/-- `eta32` is `binary32`'s smallest subnormal. -/
theorem eta32_eq : eta32 = Format.binary32.eta := Format.binary32_eta.symm
/-- `ubf` is `bfloat16`'s unit roundoff. -/
theorem ubf_eq : ubf = Format.bfloat16.u := Format.bfloat16_u.symm
/-- `etabf` is `bfloat16`'s smallest subnormal. -/
theorem etabf_eq : etabf = Format.bfloat16.eta := Format.bfloat16_eta.symm

/-- The accumulation's error bound: `K` additions from `+0` over leaves of magnitude sum `A`. -/
def E1 (K : Nat) (A : ℚ) : ℚ := ((1 + u32) ^ K - 1) * A + eta32 * (K : ℚ) * (1 + u32) ^ K

/-- The error of the final `float32` multiplication by the scale. -/
def Em (s S E : ℚ) : ℚ := u32 * s * (|S| + E) + eta32

/-- A bound on the magnitude of the `float32` product. -/
def Qb (s S E : ℚ) : ℚ := s * |S| + s * E + Em s S E

/-- The fine bound: the triangle inequality over the accumulation, the product and the
`bfloat16` rounding. -/
def Fb (s S E σ σ32 : ℚ) : ℚ :=
  ubf * Qb s S E + etabf + Em s S E + s * E + |σ - σ32| * |S|

/-- The same, in the expanded form of `README.md`. -/
def Fx (s S E σ σ32 : ℚ) : ℚ :=
  ubf * s * |S| + (1 + ubf) * ((1 + u32) * s * E + u32 * s * |S| + eta32) + etabf +
    |σ - σ32| * |S|

/-- `E1` is nonnegative. -/
theorem E1_nonneg (K : Nat) (A : ℚ) (hA : 0 ≤ A) : 0 ≤ E1 K A := by
  have hu : (0 : ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have h1 : (1 : ℚ) ≤ (1 + u32) ^ K := one_le_pow₀ (by linarith)
  have h2 : (0 : ℚ) ≤ (1 + u32) ^ K - 1 := by linarith
  have h3 : (0 : ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  unfold E1
  have : (0 : ℚ) ≤ eta32 * (K : ℚ) * (1 + u32) ^ K := by positivity
  nlinarith

/-- `E1` is monotone in the number of steps, for a nonnegative magnitude sum. -/
theorem E1_mono {K : Nat} {A : ℚ} (hA : 0 ≤ A) :
    ∀ t, t ≤ K → E1 t A ≤ E1 K A := by
  have hu : (0 : ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have h3 : (0 : ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  intro t ht
  unfold E1
  have hpt : (1 + u32) ^ t ≤ (1 + u32) ^ K := pow_le_pow_right₀ (by linarith) ht
  have h1 : (0 : ℚ) ≤ (1 + u32) ^ t := by positivity
  have hA' : (0 : ℚ) ≤ A := hA
  have : ((1 + u32) ^ t - 1) * A ≤ ((1 + u32) ^ K - 1) * A := by
    have : (1 + u32) ^ t - 1 ≤ (1 + u32) ^ K - 1 := by linarith
    exact mul_le_mul_of_nonneg_right this hA'
  have h2 : eta32 * (t : ℚ) * (1 + u32) ^ t ≤ eta32 * (K : ℚ) * (1 + u32) ^ K := by
    have ht' : (t : ℚ) ≤ (K : ℚ) := by exact_mod_cast ht
    have h0 : (0 : ℚ) ≤ eta32 * (t : ℚ) := by positivity
    calc eta32 * (t : ℚ) * (1 + u32) ^ t ≤ eta32 * (K : ℚ) * (1 + u32) ^ t := by
          apply mul_le_mul_of_nonneg_right _ h1
          nlinarith
      _ ≤ eta32 * (K : ℚ) * (1 + u32) ^ K := by
          apply mul_le_mul_of_nonneg_left hpt
          nlinarith
  linarith

/-- The two forms of the fine bound agree. -/
theorem Fb_eq_Fx (s S E σ σ32 : ℚ) :
    Fb s S E σ σ32 = Fx s S E σ σ32 := by
  unfold Fb Fx Qb Em
  ring

/-- The `float32` left fold of `x 0, …, x (t-1)`, from `+0`. -/
def foldF (x : Nat → F32) : Nat → F32
  | 0 => FloatV.ofPat 0
  | t + 1 => FloatV.add (foldF x t) (x t)

/-- The fold's every partial sum: finite, and within `E1 t A` of the exact partial sum. -/
theorem foldF_bound (x : Nat → F32) (K : Nat) (A : ℚ)
    (_hKle : K ≤ 2 ^ 20)
    (hfin : ∀ k, k < K → (x k).isFinite = true)
    (hA0 : 0 ≤ A)
    (hA : ∀ t, t ≤ K → |∑ k ∈ Finset.range t, (x k).toRat| ≤ A)
    (hM : A + E1 K A ≤ Format.binary32.maxVal) :
    ∀ t, t ≤ K → (foldF x t).isFinite = true ∧
      |(foldF x t).toRat - ∑ k ∈ Finset.range t, (x k).toRat| ≤ E1 t A := by
  have hu : (0 : ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have he : (0 : ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  have hstep : ∀ n : Nat, E1 n A + (A + E1 n A) * u32 + eta32 ≤ E1 (n + 1) A := by
    intro n
    have hp1 : (1 : ℚ) ≤ (1 + u32) ^ n := one_le_pow₀ (by linarith)
    have hd : E1 (n + 1) A - (E1 n A + (A + E1 n A) * u32 + eta32)
        = eta32 * ((1 + u32) ^ n * (1 + u32) - 1) := by
      unfold E1
      rw [pow_succ]
      push_cast
      ring
    have hnn : (0 : ℚ) ≤ eta32 * ((1 + u32) ^ n * (1 + u32) - 1) := by
      have h1 : (0 : ℚ) ≤ (1 + u32) ^ n * (1 + u32) - 1 := by nlinarith
      exact mul_nonneg he h1
    linarith
  have key : ∀ t, t ≤ K → (foldF x t).isFinite = true ∧
      |(foldF x t).toRat - ∑ k ∈ Finset.range t, (x k).toRat| ≤ E1 t A := by
    intro t
    induction t with
    | zero =>
      intro _
      have hf0 : (FloatV.ofPat (f := Format.binary32) 0).isFinite = true := by
        rw [FloatV.isFinite_ofPat (Nat.pow_pos (by norm_num : 0 < 2))]
        exact Format.isFinite_zero Format.binary32_regular
      have ht0 : (FloatV.ofPat (f := Format.binary32) 0).toRat = 0 := by
        rw [FloatV.toRat_ofPat (Nat.pow_pos (by norm_num : 0 < 2))]
        exact Format.toRat_of_isZero Format.isZero_zero
      rw [show foldF x 0 = FloatV.ofPat (f := Format.binary32) 0 from rfl]
      refine ⟨hf0, ?_⟩
      rw [ht0]
      simp [E1]
    | succ n ih =>
      intro hn
      show (foldF x (n + 1)).isFinite = true ∧
        |(foldF x (n + 1)).toRat - ∑ k ∈ Finset.range (n + 1), (x k).toRat| ≤ E1 (n + 1) A
      obtain ⟨ihfin, iherr⟩ := ih (by omega : n ≤ K)
      have hfinN : (x n).isFinite = true := hfin n (by omega)
      have hA' : |∑ k ∈ Finset.range (n + 1), (x k).toRat| ≤ A := hA (n + 1) hn
      have hS_succ : ∑ k ∈ Finset.range (n + 1), (x k).toRat
          = (∑ k ∈ Finset.range n, (x k).toRat) + (x n).toRat :=
        Finset.sum_range_succ (fun k => (x k).toRat) n
      have hb : |(foldF x n).toRat + (x n).toRat| ≤ A + E1 n A := by
        have hsplit : (foldF x n).toRat + (x n).toRat
            = ((foldF x n).toRat - ∑ k ∈ Finset.range n, (x k).toRat)
              + ((∑ k ∈ Finset.range n, (x k).toRat) + (x n).toRat) := by ring
        rw [hsplit]
        calc |((foldF x n).toRat - ∑ k ∈ Finset.range n, (x k).toRat)
                + ((∑ k ∈ Finset.range n, (x k).toRat) + (x n).toRat)|
            ≤ |(foldF x n).toRat - ∑ k ∈ Finset.range n, (x k).toRat|
                + |(∑ k ∈ Finset.range n, (x k).toRat) + (x n).toRat| := abs_add_le _ _
          _ ≤ E1 n A + A := add_le_add iherr (by rw [← hS_succ]; exact hA')
          _ = A + E1 n A := by ring
      have hmax : |(foldF x n).toRat + (x n).toRat| ≤ Format.binary32.maxVal := by
        have hm : E1 n A ≤ E1 K A := E1_mono hA0 n (by omega : n ≤ K)
        linarith
      obtain ⟨haddfin, hadderr⟩ :=
        FloatV.add_err (f := Format.binary32) Format.binary32_regular ihfin hfinN hmax
      rw [← u32_eq, ← eta32_eq] at hadderr
      refine ⟨?_, ?_⟩
      · rw [show foldF x (n + 1) = FloatV.add (foldF x n) (x n) from rfl]
        exact haddfin
      · rw [show foldF x (n + 1) = FloatV.add (foldF x n) (x n) from rfl]
        have htri : |(FloatV.add (foldF x n) (x n)).toRat
              - ∑ k ∈ Finset.range (n + 1), (x k).toRat|
            ≤ E1 n A + (A + E1 n A) * u32 + eta32 := by
          have hd2 : (FloatV.add (foldF x n) (x n)).toRat
                - ∑ k ∈ Finset.range (n + 1), (x k).toRat
              = ((FloatV.add (foldF x n) (x n)).toRat
                  - ((foldF x n).toRat + (x n).toRat))
                + (((foldF x n).toRat + (x n).toRat)
                  - ∑ k ∈ Finset.range (n + 1), (x k).toRat) := by ring
          rw [hd2]
          have hb2 : |((foldF x n).toRat + (x n).toRat)
              - ∑ k ∈ Finset.range (n + 1), (x k).toRat| ≤ E1 n A := by
            have hh : ((foldF x n).toRat + (x n).toRat)
                - ∑ k ∈ Finset.range (n + 1), (x k).toRat
                = (foldF x n).toRat - ∑ k ∈ Finset.range n, (x k).toRat := by
              rw [hS_succ]; ring
            rw [hh]; exact iherr
          have hu' : |(foldF x n).toRat + (x n).toRat| * u32 + eta32
              ≤ (A + E1 n A) * u32 + eta32 := by
            have h1 : |(foldF x n).toRat + (x n).toRat| * u32 ≤ (A + E1 n A) * u32 :=
              mul_le_mul_of_nonneg_right hb hu
            linarith
          calc |((FloatV.add (foldF x n) (x n)).toRat
                  - ((foldF x n).toRat + (x n).toRat))
                + (((foldF x n).toRat + (x n).toRat)
                  - ∑ k ∈ Finset.range (n + 1), (x k).toRat)|
              ≤ |(FloatV.add (foldF x n) (x n)).toRat
                  - ((foldF x n).toRat + (x n).toRat)|
                + |((foldF x n).toRat + (x n).toRat)
                  - ∑ k ∈ Finset.range (n + 1), (x k).toRat| := abs_add_le _ _
            _ ≤ (|(foldF x n).toRat + (x n).toRat| * u32 + eta32) + E1 n A :=
                add_le_add hadderr hb2
            _ ≤ ((A + E1 n A) * u32 + eta32) + E1 n A := by linarith
            _ = E1 n A + (A + E1 n A) * u32 + eta32 := by ring
        have hs := hstep n
        linarith
  exact key

/-- The whole fold: finite, and within `E1 K A` of the exact sum. -/
theorem foldF_err (x : Nat → F32) (K : Nat) (A : ℚ)
    (_hK : 0 < K) (hKle : K ≤ 2 ^ 20)
    (hfin : ∀ k, k < K → (x k).isFinite = true)
    (hA0 : 0 ≤ A)
    (hA : ∀ t, t ≤ K → |∑ k ∈ Finset.range t, (x k).toRat| ≤ A)
    (hM : A + E1 K A ≤ Format.binary32.maxVal) :
    (foldF x K).isFinite = true ∧
      |(foldF x K).toRat - ∑ k ∈ Finset.range K, (x k).toRat| ≤ E1 K A :=
  foldF_bound x K A hKle hfin hA0 hA hM K le_rfl

/-- For `K ≤ 2^20`, `(1+u)^K · (1 - K·u) ≤ 1`, hence `(1+u)^K ≤ 16/15`. -/
theorem pow16 : ∀ K : Nat, K ≤ 2 ^ 20 →
    (1 + u32) ^ K ≤ 1 + (16 / 15 : ℚ) * (K : ℚ) * u32 := by
  intro K
  induction K with
  | zero => intro _; norm_num
  | succ n ih =>
    intro hK
    have hn : n ≤ 2 ^ 20 := by omega
    have hprev := ih hn
    have hu0 : (0:ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
    have hu : u32 = 1/16777216 := by norm_num [u32]
    have hnq : (n:ℚ) ≤ 2^20 - 1 := by
      have h : ((n+1 : Nat) : ℚ) ≤ ((2:ℕ) : ℚ)^20 := by exact_mod_cast hK
      push_cast at h
      linarith
    have hnu : (n:ℚ) * u32 ≤ 1/16 := by nlinarith
    have hstep : (1 + (16/15:ℚ) * (n:ℚ) * u32) * (1 + u32) ≤ 1 + (16/15:ℚ) * ((n:ℚ) + 1) * u32 := by
      nlinarith
    calc (1+u32)^(n+1) = (1+u32)^n * (1+u32) := pow_succ _ _
      _ ≤ (1 + (16/15:ℚ) * (n:ℚ) * u32) * (1+u32) := by
          exact mul_le_mul_of_nonneg_right hprev (by linarith)
      _ ≤ 1 + (16/15:ℚ) * ((n:ℚ)+1) * u32 := hstep
      _ = 1 + (16/15:ℚ) * ((n+1:Nat):ℚ) * u32 := by push_cast; ring

/-- `pow16` at the top of the range: `(1+u)^K < 2` for `K ≤ 2^20`. -/
theorem pow16_le_two (K : Nat) (hK : K ≤ 2^20) : (1 + u32)^K ≤ 2 := by
  have h := pow16 K hK
  have hu : u32 = 1/16777216 := by norm_num [u32]
  have hKq : (K:ℚ) ≤ 2^20 := by exact_mod_cast hK
  have hKq0 : (0:ℚ) ≤ (K:ℚ) := by positivity
  nlinarith

/-- The error of `K` additions at the top of the range: `E1 K A ≤ (16/15)·K·u·A + 2·η·K`. -/
theorem E1_le (K : Nat) (A : ℚ) (hKle : K ≤ 2 ^ 20) (hA0 : 0 ≤ A) :
    E1 K A ≤ (16 / 15 : ℚ) * (K : ℚ) * u32 * A + 2 * eta32 * (K : ℚ) := by
  have he0 : (0:ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  have hKq0 : (0:ℚ) ≤ (K:ℚ) := by positivity
  have hP2 : (1 + u32)^K ≤ 2 := pow16_le_two K hKle
  have hC : (1 + u32)^K - 1 ≤ (16/15)*(K:ℚ)*u32 := by
    have h := pow16 K hKle
    linarith
  unfold E1
  have h1 : ((1+u32)^K - 1) * A ≤ ((16/15)*(K:ℚ)*u32) * A :=
    mul_le_mul_of_nonneg_right hC hA0
  have h2 : eta32*(K:ℚ)*(1+u32)^K ≤ eta32*(K:ℚ)*2 :=
    mul_le_mul_of_nonneg_left hP2 (mul_nonneg he0 hKq0)
  nlinarith

/-- The combined error terms are within `tol`. -/
theorem Fx_le_tol (K : Nat) (S A σ σ32 s : ℚ)
    (hK1 : 1 ≤ K) (hKle : K ≤ 2 ^ 20)
    (hA0 : 0 ≤ A) (hS : |S| ≤ A) (hs0 : 0 ≤ s)
    (hs : s ≤ (1 + u32) * |σ| + eta32)
    (hσ : |σ - σ32| ≤ u32 * |σ| + eta32) :
    Fx s S (E1 K A) σ σ32 ≤ tol K S A σ := by
  have hu0 : (0:ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have he0 : (0:ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  have hv0 : (0:ℚ) ≤ ubf := by rw [ubf_eq]; exact Format.u_nonneg
  have heb0 : (0:ℚ) ≤ etabf := by rw [etabf_eq]; exact Format.eta_nonneg
  have ha0 : (0:ℚ) ≤ |σ| := abs_nonneg σ
  have hb0 : (0:ℚ) ≤ |S| := abs_nonneg S
  have hKq0 : (0:ℚ) ≤ (K:ℚ) := by positivity
  have hKq1 : (1:ℚ) ≤ (K:ℚ) := by exact_mod_cast hK1
  have hKqle : (K:ℚ) ≤ 2^20 := by exact_mod_cast hKle
  have hP2 : (1 + u32)^K ≤ 2 := pow16_le_two K hKle
  have hC : (1 + u32)^K - 1 ≤ (16/15)*(K:ℚ)*u32 := by
    have h := pow16 K hKle
    linarith
  have hE1 : E1 K A ≤ (16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ) := E1_le K A hKle hA0
  have hE10 : (0:ℚ) ≤ E1 K A := E1_nonneg K A hA0
  have hs' : (0:ℚ) ≤ (1 + u32) * |σ| + eta32 := by nlinarith
  have hsS : s * |S| ≤ ((1 + u32) * |σ| + eta32) * |S| :=
    mul_le_mul_of_nonneg_right hs hb0
  have hsE : s * (E1 K A) ≤ ((1 + u32) * |σ| + eta32) * ((16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ)) :=
    mul_le_mul hs hE1 hE10 hs'
  have h4 : |σ - σ32| * |S| ≤ (u32 * |σ| + eta32) * |S| :=
    mul_le_mul_of_nonneg_right hσ hb0
  have hm1 : ubf * (s * |S|) ≤ ubf * (((1 + u32) * |σ| + eta32) * |S|) :=
    mul_le_mul_of_nonneg_left hsS hv0
  have hm2 : (1 + ubf) * ((1 + u32) * (s * (E1 K A)))
      ≤ (1 + ubf) * ((1 + u32) * (((1 + u32) * |σ| + eta32) * ((16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ)))) := by
    have h1u : (0:ℚ) ≤ 1 + u32 := by linarith
    have h : (1 + u32) * (s * (E1 K A))
        ≤ (1 + u32) * (((1 + u32) * |σ| + eta32) * ((16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ))) :=
      mul_le_mul_of_nonneg_left hsE h1u
    exact mul_le_mul_of_nonneg_left h (by linarith)
  have hm3 : (1 + ubf) * (u32 * (s * |S|))
      ≤ (1 + ubf) * (u32 * (((1 + u32) * |σ| + eta32) * |S|)) := by
    have h : u32 * (s * |S|) ≤ u32 * (((1 + u32) * |σ| + eta32) * |S|) :=
      mul_le_mul_of_nonneg_left hsS hu0
    exact mul_le_mul_of_nonneg_left h (by linarith)
  have habA : |σ| * |S| ≤ |σ| * A := mul_le_mul_of_nonneg_left hS ha0
  have ci : ubf*u32 + (16/15)*(1+ubf)*(1+u32)^2*(K:ℚ)*u32 + (1+ubf)*u32*(1+u32) + u32
      ≤ (K:ℚ)*2^(-22:ℤ) := by
    norm_num [u32, eta32, ubf, etabf, mu32]
    nlinarith
  have cii : ubf*eta32 + (16/15)*(1+ubf)*(1+u32)*(K:ℚ)*u32*eta32 + (1+ubf)*u32*eta32 + eta32
      ≤ (K:ℚ)*2^(-22:ℤ)*mu32 := by
    norm_num [u32, eta32, ubf, etabf, mu32]
    nlinarith
  have ciii : 2*eta32*(K:ℚ)*(1+ubf)*(1+u32)^2 ≤ (K:ℚ)*2^(-22:ℤ)*mu32 := by
    norm_num [u32, eta32, ubf, etabf, mu32]
    nlinarith
  have civ : 2*eta32^2*(K:ℚ)*(1+ubf)*(1+u32) + (1+ubf)*eta32 + etabf ≤ 2^(-133:ℤ) := by
    norm_num [u32, eta32, ubf, etabf, mu32]
    nlinarith
  have hAA : (ubf*(1+u32) + (1+ubf)*(1+u32)^2*(16/15)*(K:ℚ)*u32 + (1+ubf)*u32*(1+u32) + u32) * (|σ| * A)
      ≤ (ubf + (K:ℚ)*2^(-22:ℤ)) * (|σ| * A) := by
    apply mul_le_mul_of_nonneg_right _ (mul_nonneg ha0 hA0)
    nlinarith [ci]
  have hAA_A : (ubf*eta32 + (16/15)*(1+ubf)*(1+u32)*(K:ℚ)*u32*eta32 + (1+ubf)*u32*eta32 + eta32) * A
      ≤ ((K:ℚ)*2^(-22:ℤ)*mu32) * A := by
    apply mul_le_mul_of_nonneg_right _ hA0
    nlinarith [cii]
  have hAA_a : (2*eta32*(K:ℚ)*(1+ubf)*(1+u32)^2) * |σ|
      ≤ ((K:ℚ)*2^(-22:ℤ)*mu32) * |σ| :=
    mul_le_mul_of_nonneg_right ciii ha0
  have h133 : (2:ℚ)^(-133:ℤ) = 2 * etabf := by norm_num [etabf]
  have hconst : 2*eta32^2*(K:ℚ)*(1+ubf)*(1+u32) + (1+ubf)*eta32 + etabf ≤ 2 * etabf := by
    linarith [civ, h133]
  have h8 : (2:ℚ)^(-8:ℤ) = ubf := rfl
  have h126 : (2:ℚ)^(-126:ℤ) = mu32 := rfl
  have h22 : (2:ℚ)^(-22:ℤ) = 4 * u32 := by norm_num [u32]
  have e_u : u32 = (1/2:ℚ)^24 := by norm_num [u32]
  have e_e : eta32 = (1/2:ℚ)^150 := by norm_num [eta32]
  have e_v : ubf = (1/2:ℚ)^8 := by norm_num [ubf]
  have e_eb : etabf = (1/2:ℚ)^134 := by norm_num [etabf]
  have e_m : mu32 = (1/2:ℚ)^126 := by norm_num [mu32]
  unfold Fx tol
  rw [abs_mul]
  simp only [h8, h126, h22, h133, e_u, e_e, e_v, e_eb, e_m] at *
  linarith [hm1, hm2, hm3, h4, habA, hS, hAA, hAA_A, hAA_a, hconst, hKq0]

/-- A pattern and its reduction modulo the format width have the same magnitude field. -/
theorem mag_mod_width (f : Format) (n : Nat) : f.mag (n % 2 ^ f.width) = f.mag n := by
  unfold Format.mag
  rw [Nat.mod_mod_of_dvd n ⟨2, by rw [Nat.mul_comm, Format.two_signBit]⟩]

/-- A pattern and its reduction modulo the format width have the same sign. -/
theorem sign_mod_width (f : Format) (n : Nat) : f.sign (n % 2 ^ f.width) = f.sign n := by
  unfold Format.sign
  rw [show 2 ^ f.width = f.signBit * 2 by rw [Nat.mul_comm, Format.two_signBit]]
  rw [Nat.mod_mul_right_div_self, Nat.mod_mod]

/-- A pattern and its reduction modulo the format width have the same value. -/
theorem toRat_mod_width (f : Format) (n : Nat) : f.toRat (n % 2 ^ f.width) = f.toRat n := by
  unfold Format.toRat
  rw [mag_mod_width f n, sign_mod_width f n]

/-- Being a NaN depends on the pattern only modulo the format width. -/
theorem isNaN_mod_width (f : Format) (n : Nat) : f.isNaN (n % 2 ^ f.width) = f.isNaN n := by
  cases hs : f.specials <;>
    simp [Format.isNaN, Format.expField, Format.fracField, mag_mod_width f n]

/-- Being infinite depends on the pattern only modulo the format width. -/
theorem isInf_mod_width (f : Format) (n : Nat) : f.isInf (n % 2 ^ f.width) = f.isInf n := by
  cases hs : f.specials <;>
    simp [Format.isInf, Format.expField, Format.fracField, mag_mod_width f n]

/-- Being finite depends on the pattern only modulo the format width. -/
theorem isFinite_mod_width (f : Format) (n : Nat) :
    f.isFinite (n % 2 ^ f.width) = f.isFinite n := by
  unfold Format.isFinite
  rw [isNaN_mod_width f n, isInf_mod_width f n]

/-- `ofPat` keeps the pattern modulo the format width. -/
theorem pat_ofPat' (f : Format) (n : Nat) : (FloatV.ofPat n : FloatV f).pat = n % 2 ^ f.width := by
  simp [FloatV.pat, FloatV.ofPat]

/-- The value of `ofPat n` is the format's value of the integer `n`. -/
theorem toRat_ofPat' (f : Format) (n : Nat) : (FloatV.ofPat n : FloatV f).toRat = f.toRat n := by
  unfold FloatV.toRat
  rw [pat_ofPat' f n, toRat_mod_width]

/-- The finiteness of `ofPat n` is the format's finiteness of `n`. -/
theorem isFinite_ofPat' (f : Format) (n : Nat) :
    (FloatV.ofPat n : FloatV f).isFinite = f.isFinite n := by
  unfold FloatV.isFinite
  rw [pat_ofPat' f n, isFinite_mod_width]

/-- The `float32` constant Triton builds for the scale is within one rounding of the double. -/
theorem sigma32_rel (d : Inst) (h64 : (sigma d).isFinite = true)
    (h32 : (sigma32 d).isFinite = true) :
    |(sigma d).toRat - (sigma32 d).toRat| ≤ u32 * |(sigma32 d).toRat| + eta32 := by
  have hnan64 : (sigma d).isNaN = false := (FloatV.isFinite_iff.mp h64).1
  by_cases hz : (sigma d).isZero = true
  · have hA : (sigma d).toRat = 0 := FloatV.toRat_of_isZero hz
    have hB : (sigma32 d).toRat = 0 := by
      unfold sigma32 FloatV.litF32
      rw [ite_eq_left hz]
      rw [toRat_ofPat' Format.binary32 0]
      exact Format.toRat_of_isZero (by simp [Format.isZero, Format.mag])
    rw [hA, hB]
    have he : (0:ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
    simp [he]
  · have hreg : Format.binary32.Regular := by decide
    have hspec : Format.binary32.specials ≠ Format.Specials.finite := by decide
    have hs32 : sigma32 d = (FloatV.ofPat (Format.binary32.roundRat .rn false
        (Format.binary64.sign (sigma d).pat) (Format.binary64.toRat (sigma d).pat))
        : F32) := by
      unfold sigma32 FloatV.litF32
      rw [ite_eq_right hz]
      unfold FloatV.hostF32
      rw [ite_eq_right (by simp [hnan64])]
      unfold FloatV.cvt
      rw [Format.cvt_of_finite Format.binary32 .rn false h64]
    have h32r : Format.binary32.isFinite (Format.binary32.roundRat .rn false
        (Format.binary64.sign (sigma d).pat) (Format.binary64.toRat (sigma d).pat)) = true := by
      rw [hs32, isFinite_ofPat'] at h32
      exact h32
    have hB : (sigma32 d).toRat = Format.binary32.toRat (Format.binary32.roundRat .rn false
        (Format.binary64.sign (sigma d).pat) (Format.binary64.toRat (sigma d).pat)) := by
      rw [hs32, toRat_ofPat']
    have herr := Format.roundRat_rn_err_of_finite (f := Format.binary32) hreg hspec (Format.binary64.sign (sigma d).pat) h32r
    rw [← u32_eq, ← eta32_eq] at herr
    rw [hB, abs_sub_comm]
    simpa [FloatV.toRat, mul_comm] using herr

/-- The fine bound with the scale entering through `σ₃₂`: `Fx |σ₃₂| … ≤ tol`. -/
theorem Fx_le_tol_weak (K : Nat) (S A σ σ32 : ℚ)
    (hK1 : 1 ≤ K) (hKle : K ≤ 2 ^ 20)
    (hA0 : 0 ≤ A) (hS : |S| ≤ A)
    (hσ : |σ - σ32| ≤ u32 * |σ32| + eta32) :
    Fx |σ32| S (E1 K A) σ σ32 ≤ tol K S A σ := by
  have hu0 : (0:ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have he0 : (0:ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  have hv0 : (0:ℚ) ≤ ubf := by rw [ubf_eq]; exact Format.u_nonneg
  have heb0 : (0:ℚ) ≤ etabf := by rw [etabf_eq]; exact Format.eta_nonneg
  have ha0 : (0:ℚ) ≤ |σ| := abs_nonneg σ
  have hb0 : (0:ℚ) ≤ |S| := abs_nonneg S
  have hKq0 : (0:ℚ) ≤ (K:ℚ) := by positivity
  have hKq1 : (1:ℚ) ≤ (K:ℚ) := by exact_mod_cast hK1
  have hKqle : (K:ℚ) ≤ 2^20 := by exact_mod_cast hKle
  have hu12 : u32 ≤ 1/2 := by norm_num [u32]
  have hu1 : (0:ℚ) < 1 - u32 := by norm_num [u32]
  have habs : |σ32| ≤ |σ| + u32 * |σ32| + eta32 := by
    have h1 : |σ32| ≤ |σ| + |σ - σ32| := by
      have h := abs_add_le σ (σ32 - σ)
      have h2 : σ + (σ32 - σ) = σ32 := by ring
      rw [h2, abs_sub_comm σ32 σ] at h
      exact h
    linarith
  have hdiv : |σ32| ≤ (|σ| + eta32) / (1 - u32) := by
    rw [le_div_iff₀ hu1]
    nlinarith [habs]
  have hfac : (|σ| + eta32) / (1 - u32) ≤ (1 + 2*u32) * (|σ| + eta32) := by
    rw [div_le_iff₀ hu1]
    have h1m : (0:ℚ) ≤ 1 - 2*u32 := by norm_num [u32]
    have hp : (0:ℚ) ≤ (|σ| + eta32) * (u32 * (1 - 2*u32)) :=
      mul_nonneg (by linarith) (mul_nonneg hu0 h1m)
    nlinarith [hp]
  have hs2 : |σ32| ≤ (1 + 2*u32)*|σ| + 2*eta32 := by
    have h3 : (1 + 2*u32) * (|σ| + eta32) ≤ (1 + 2*u32)*|σ| + 2*eta32 := by
      nlinarith [hu12, he0]
    linarith [hdiv, hfac, h3]
  have hSA : (0:ℚ) ≤ (1 + 2*u32)*|σ| + 2*eta32 := by nlinarith [hu0, ha0, he0]
  have h1u : (0:ℚ) ≤ 1 + u32 := by linarith
  have h1v : (0:ℚ) ≤ 1 + ubf := by linarith
  have hE1 : E1 K A ≤ (16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ) := E1_le K A hKle hA0
  have hE10 : (0:ℚ) ≤ E1 K A := E1_nonneg K A hA0
  have hsS : |σ32| * |S| ≤ ((1 + 2*u32)*|σ| + 2*eta32) * |S| :=
    mul_le_mul_of_nonneg_right hs2 hb0
  have hsE : |σ32| * (E1 K A) ≤
      ((1 + 2*u32)*|σ| + 2*eta32) * ((16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ)) :=
    mul_le_mul hs2 hE1 hE10 hSA
  have e1 : ubf * (|σ32| * |S|) ≤ ubf * (((1 + 2*u32)*|σ| + 2*eta32) * |S|) :=
    mul_le_mul_of_nonneg_left hsS hv0
  have e2 : (1+ubf) * ((1+u32) * (|σ32| * (E1 K A)))
      ≤ (1+ubf) * ((1+u32) * (((1 + 2*u32)*|σ| + 2*eta32) *
          ((16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ)))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hsE h1u) h1v
  have e3 : (1+ubf) * (u32 * (|σ32| * |S|))
      ≤ (1+ubf) * (u32 * (((1 + 2*u32)*|σ| + 2*eta32) * |S|)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hsS hu0) h1v
  have e4 : |σ - σ32| * |S| ≤ (u32 * ((1 + 2*u32)*|σ| + 2*eta32) + eta32) * A := by
    have hσ' : |σ - σ32| ≤ u32 * ((1 + 2*u32)*|σ| + 2*eta32) + eta32 := by
      have h := mul_le_mul_of_nonneg_left hs2 hu0
      linarith
    have hcoef : (0:ℚ) ≤ u32 * ((1 + 2*u32)*|σ| + 2*eta32) + eta32 := by nlinarith [hu0, he0]
    exact (mul_le_mul_of_nonneg_right hσ' hb0).trans (mul_le_mul_of_nonneg_left hS hcoef)
  have hFxeq : Fx |σ32| S (E1 K A) σ σ32
      = ubf * (|σ32| * |S|) + (1+ubf) * ((1+u32) * (|σ32| * (E1 K A)))
        + (1+ubf) * (u32 * (|σ32| * |S|)) + (1+ubf) * eta32 + etabf
        + |σ - σ32| * |S| := by
    unfold Fx
    ring
  have hFx : Fx |σ32| S (E1 K A) σ σ32 ≤
      ubf * (((1 + 2*u32)*|σ| + 2*eta32) * |S|)
      + (1+ubf) * ((1+u32) * (((1 + 2*u32)*|σ| + 2*eta32) *
          ((16/15)*(K:ℚ)*u32*A + 2*eta32*(K:ℚ))))
      + (1+ubf) * (u32 * (((1 + 2*u32)*|σ| + 2*eta32) * |S|))
      + (1+ubf) * eta32 + etabf
      + (u32 * ((1 + 2*u32)*|σ| + 2*eta32) + eta32) * A := by
    rw [hFxeq]
    linarith [e1, e2, e3, e4]
  have ci : (1+ubf)*(1+u32)*(1+2*u32)*(16/15)*(K:ℚ)*u32 + u32*(1+2*u32)
      + 2*ubf*u32 + (1+ubf)*u32*(1+2*u32) ≤ (K:ℚ)*4*u32 := by
    norm_num [u32, ubf]
    nlinarith
  have cii : 2*(1+ubf)*(1+u32)*(1+2*u32)*(K:ℚ) ≤ (K:ℚ)*4 := by
    norm_num [u32, ubf]
    nlinarith
  have ciii : 2*ubf + 2*(1+ubf)*u32 + 2*(16/15)*(1+ubf)*(1+u32)*(K:ℚ)*u32 + 2*u32 + 1
      ≤ (K:ℚ)*4 := by
    norm_num [u32, ubf]
    nlinarith
  have hKeta : (K:ℚ) * eta32^2 ≤ 2^20 * eta32^2 :=
    mul_le_mul_of_nonneg_right hKqle (by positivity)
  have civ : 4*(K:ℚ)*(1+ubf)*(1+u32)*eta32^2 + (1+ubf)*eta32 + etabf ≤ (2:ℚ)^(-133:ℤ) := by
    norm_num [u32, ubf, eta32, etabf]
    nlinarith [hKeta]
  have hAA : ((1+ubf)*(1+u32)*(1+2*u32)*(16/15)*(K:ℚ)*u32 + u32*(1+2*u32)
      + 2*ubf*u32 + (1+ubf)*u32*(1+2*u32)) * (|σ| * A)
      ≤ ((K:ℚ)*4*u32) * (|σ| * A) :=
    mul_le_mul_of_nonneg_right ci (mul_nonneg ha0 hA0)
  have hcoef1 : (0:ℚ) ≤ 2*ubf*u32 + (1+ubf)*u32*(1+2*u32) := by
    have h1 : (0:ℚ) ≤ ubf*u32 := mul_nonneg hv0 hu0
    have h2 : (0:ℚ) ≤ (1+ubf)*u32*(1+2*u32) :=
      mul_nonneg (mul_nonneg (by linarith) hu0) (by linarith)
    linarith
  have hcoef2 : (0:ℚ) ≤ 2*ubf + 2*(1+ubf)*u32 := by
    have h1 : (0:ℚ) ≤ (1+ubf)*u32 := mul_nonneg (by linarith) hu0
    linarith
  have hAAS : (2*ubf*u32 + (1+ubf)*u32*(1+2*u32)) * (|σ| * |S|)
      ≤ (2*ubf*u32 + (1+ubf)*u32*(1+2*u32)) * (|σ| * A) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hS ha0) hcoef1
  have hAAS2 : (2*ubf + 2*(1+ubf)*u32) * (eta32 * |S|)
      ≤ (2*ubf + 2*(1+ubf)*u32) * (eta32 * A) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hS he0) hcoef2
  have hmu : u32 * mu32 = eta32 := by norm_num [u32, mu32, eta32]
  have hm0 : (0:ℚ) ≤ mu32 := by rw [show mu32 = (2:ℚ)^(-126:ℤ) from rfl]; positivity
  have hnn : (0:ℚ) ≤ 4*(K:ℚ)*u32*mu32*mu32 := by
    have h1 : (0:ℚ) ≤ (K:ℚ)*u32 := mul_nonneg hKq0 hu0
    have h2 : (0:ℚ) ≤ mu32*mu32 := mul_nonneg hm0 hm0
    nlinarith [mul_nonneg h1 h2]
  have hconst : 4*(K:ℚ)*(1+ubf)*(1+u32)*eta32^2 + (1+ubf)*eta32 + etabf
      ≤ (2:ℚ)^(-133:ℤ) + 4*(K:ℚ)*u32*mu32*mu32 := by linarith [civ]
  have h4e : (K:ℚ)*4*eta32 = (K:ℚ)*4*u32*mu32 := by rw [← hmu]; ring
  have cii2 : 2*(1+ubf)*(1+u32)*(1+2*u32)*(K:ℚ)*eta32 ≤ (K:ℚ)*4*u32*mu32 := by
    have h := mul_le_mul_of_nonneg_right cii he0
    rwa [h4e] at h
  have ciii2 : (2*ubf + 2*(1+ubf)*u32 + 2*(16/15)*(1+ubf)*(1+u32)*(K:ℚ)*u32 + 2*u32 + 1)*eta32
      ≤ (K:ℚ)*4*u32*mu32 := by
    have h := mul_le_mul_of_nonneg_right ciii he0
    rwa [h4e] at h
  have hAA_a : (2*(1+ubf)*(1+u32)*(1+2*u32)*(K:ℚ)*eta32) * |σ|
      ≤ ((K:ℚ)*4*u32*mu32) * |σ| :=
    mul_le_mul_of_nonneg_right cii2 ha0
  have hAA_A : (2*ubf + 2*(1+ubf)*u32 + 2*(16/15)*(1+ubf)*(1+u32)*(K:ℚ)*u32 + 2*u32 + 1)*eta32 * A
      ≤ ((K:ℚ)*4*u32*mu32) * A :=
    mul_le_mul_of_nonneg_right ciii2 hA0
  have h8 : (2:ℚ)^(-8:ℤ) = ubf := rfl
  have h22 : (2:ℚ)^(-22:ℤ) = 4 * u32 := by norm_num [u32]
  have h126 : (2:ℚ)^(-126:ℤ) = mu32 := rfl
  unfold tol
  rw [abs_mul]
  simp only [h8, h22, h126]
  linear_combination hFx + hAA + hAAS + hAAS2 + hAA_a + hAA_A + hconst

/-- A value `NaNEq` to a non-NaN value is that value. -/
theorem NaNEq.eq_right {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hy : y.isNaN = false) : x = y := by
  rcases h with h | ⟨-, hy'⟩
  · exact h
  · rw [hy] at hy'; cases hy'

/-- A product with a NaN operand is NaN. -/
theorem mul_isNaN {f : Format} {x y : FloatV f} (hx : x.isNaN = true) :
    FloatV.mul x y = FloatV.ofPat f.nanMag := by
  have hx' : f.isNaN x.pat = true := hx
  show FloatV.ofPat (f.mul .rn x.pat y.pat) = FloatV.ofPat f.nanMag
  rw [Format.mul, ite_eq_left (by rw [hx']; rfl)]

/-- `tl.mul` respects `NaNEq` in its left operand. -/
theorem mul_congr_left {f : Format} {a b c : FloatV f} (h : FloatV.NaNEq a b) :
    FloatV.mul a c = FloatV.mul b c := by
  rcases h with h | ⟨ha, hb⟩
  · rw [h]
  · rw [mul_isNaN ha, mul_isNaN hb]

/-- `tl.mul` respects `NaNEq` in its right operand. -/
theorem mul_congr_right {f : Format} {a b c : FloatV f} (h : FloatV.NaNEq a b) :
    FloatV.mul c a = FloatV.mul c b := by
  rcases h with h | ⟨ha, hb⟩
  · rw [h]
  · rw [show FloatV.mul c a = FloatV.ofPat f.nanMag from by
        show FloatV.ofPat (f.mul .rn c.pat a.pat) = FloatV.ofPat f.nanMag
        have ha' : f.isNaN a.pat = true := ha
        rw [Format.mul, ite_eq_left (by rw [ha']; rw [Bool.or_comm]; rfl)],
      show FloatV.mul c b = FloatV.ofPat f.nanMag from by
        show FloatV.ofPat (f.mul .rn c.pat b.pat) = FloatV.ofPat f.nanMag
        have hb' : f.isNaN b.pat = true := hb
        rw [Format.mul, ite_eq_left (by rw [hb']; rw [Bool.or_comm]; rfl)]]

/-- A sum with a NaN operand is NaN. -/
theorem add_isNaN {f : Format} {x y : FloatV f} (hx : x.isNaN = true) :
    FloatV.add x y = FloatV.ofPat f.nanMag := by
  have hx' : f.isNaN x.pat = true := hx
  show FloatV.ofPat (f.add .rn x.pat y.pat) = FloatV.ofPat f.nanMag
  rw [Format.add, ite_eq_left (by rw [hx']; rfl)]

/-- `tl.add` respects `NaNEq` in its left operand. -/
theorem add_congr_left {f : Format} {a b c : FloatV f} (h : FloatV.NaNEq a b) :
    FloatV.add a c = FloatV.add b c := by
  rcases h with h | ⟨ha, hb⟩
  · rw [h]
  · rw [add_isNaN ha, add_isNaN hb]

/-- `tl.add` respects `NaNEq` in its right operand. -/
theorem add_congr_right {f : Format} {a b c : FloatV f} (h : FloatV.NaNEq a b) :
    FloatV.add c a = FloatV.add c b := by
  rcases h with h | ⟨ha, hb⟩
  · rw [h]
  · rw [show FloatV.add c a = FloatV.ofPat f.nanMag from by
        show FloatV.ofPat (f.add .rn c.pat a.pat) = FloatV.ofPat f.nanMag
        have ha' : f.isNaN a.pat = true := ha
        rw [Format.add, ite_eq_left (by rw [ha']; rw [Bool.or_comm]; rfl)],
      show FloatV.add c b = FloatV.ofPat f.nanMag from by
        show FloatV.ofPat (f.add .rn c.pat b.pat) = FloatV.ofPat f.nanMag
        have hb' : f.isNaN b.pat = true := hb
        rw [Format.add, ite_eq_left (by rw [hb']; rw [Bool.or_comm]; rfl)]]

/-- A conversion of a NaN is NaN. -/
theorem cvt_isNaN {f g : Format} {x : FloatV f} (hx : x.isNaN = true) :
    (FloatV.cvt x : FloatV g) = FloatV.ofPat g.nanMag := by
  have hx' : f.isNaN x.pat = true := hx
  show FloatV.ofPat (Format.cvt f g .rn false x.pat) = FloatV.ofPat g.nanMag
  rw [Format.cvt, ite_eq_left hx']

/-- A conversion respects `NaNEq`. -/
theorem cvt_congr {f g : Format} {x y : FloatV f} (h : FloatV.NaNEq x y) :
    (FloatV.cvt x : FloatV g) = FloatV.cvt y := by
  rcases h with h | ⟨hx, hy⟩
  · rw [h]
  · rw [cvt_isNaN hx, cvt_isNaN hy]

end Claims.MoeSumReduce
