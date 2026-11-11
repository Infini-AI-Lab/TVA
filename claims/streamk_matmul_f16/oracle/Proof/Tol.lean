import Claim
import TVA.Proof
import Proof.Lanes

/-!
# `Proof.Tol`

The error constants of one `tl.dot` step and of the atomic additions, and the tolerance `tol` closed from them (`tol_close`).
-/
set_option maxHeartbeats 1000000

/-!
# The tolerance: from the per-program `tl.dot` errors to `Claim.tol`

The kernel's `K` loop is split by Stream-K into contiguous runs of `64`-wide chunks, one run per
program (per tile).  Program `p` accumulates its run from `0` with `tl.dot` steps, so after `n`
steps its value is within `DotLoop.bound`-sharp

  `fineOne c0 n = Σ_{i<n} (1+ρ)^(n-1-i) · (ρ·(dev(c0+i) + chunkMag(c0+i)) + α)`

of the exact partial sum, with `ρ = DotLoop.stepRel relDot16 64` and
`α = DotLoop.stepAbs absDot16 64` (`DotLift.dotAdd_step`).  The `dev` term carries the magnitude of
the running accumulator (each step's error is relative to `|acc| + Σ|a·b|`), and `chunkMag` the
products.

Summed over the programs' chunk runs, which partition `[0, steps K)`, the amplification
`(1+ρ)^(n-1-i) ≤ (1+ρ)^(steps K - 1 - t)` gives the global `tolDotR`.

The atomics that combine the programs' partial sums add a further
`DotLoop.bound 2^-23 2^-124 na (C + e)`, with `na ≤ steps K` additions and `C = Σ|chunkSum|`,
from `Claim.Assumptions.atomic`.

This file is pure rational arithmetic.  `tol_close` is the single inequality the kernel proof
needs.
-/

namespace Claims.StreamKMatmul

open TVA

/-! ## The two constants of one `tl.dot` step -/

/-- `ρ`: one `64`-product `float16` `tl.dot` step, with the separate addition of the accumulator
(`DotLoop.stepRel`, `relDot16 64 = 66·2^-22`, and `Format.binary32.u = 2^-24`). -/
def rho : ℚ := DotLoop.stepRel Device.relDot16 64

/-- `α`: the absolute term of one `64`-product `float16` `tl.dot` step (`DotLoop.stepAbs`). -/
def alpha : ℚ := DotLoop.stepAbs Device.absDot16 64

/-- `rho` is nonnegative. -/
theorem rho_nonneg : 0 ≤ rho := by
  unfold rho DotLoop.stepRel
  have h1 : 0 ≤ Device.relDot16 64 := by unfold Device.relDot16; positivity
  have hu : (0 : ℚ) ≤ Format.binary32.u := Format.u_nonneg (f := Format.binary32)
  have h2 : (0 : ℚ) ≤ 3 * Format.binary32.u * (1 + Device.relDot16 64) :=
    mul_nonneg (mul_nonneg (by norm_num) hu) (by linarith)
  linarith

/-- `alpha` is nonnegative. -/
theorem alpha_nonneg : 0 ≤ alpha := by
  unfold alpha DotLoop.stepAbs
  have h1 : 0 ≤ Device.absDot16 64 := by unfold Device.absDot16; positivity
  have hu : (0 : ℚ) ≤ Format.binary32.u := Format.u_nonneg (f := Format.binary32)
  have he : (0 : ℚ) ≤ Format.binary32.eta := Format.eta_nonneg (f := Format.binary32)
  have h2 : (0 : ℚ) ≤ Device.absDot16 64 * (1 + 3 * Format.binary32.u) :=
    mul_nonneg h1 (by linarith)
  have h3 : (0 : ℚ) ≤ 4 * Format.binary32.eta := mul_nonneg (by norm_num) he
  linarith

/-- `rho ≤ 1/60000`. -/
theorem rho_le : rho ≤ 1/60000 := by
  unfold rho DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  norm_num

/-- `alpha ≤ 2^-119`. -/
theorem alpha_le : alpha ≤ 2 ^ (-119 : ℤ) := by
  unfold alpha DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

/-! ## The per-program fine error -/

/-- The error of `n` `tl.dot` steps over the chunk run starting at `c0`, in closed form. -/
def fineOne (va vb : Nat → F16) (K c0 n : Nat) : ℚ :=
  ∑ i ∈ Finset.range n,
    (1 + rho) ^ (n - 1 - i) * (rho * (dev va vb K (c0 + i) + chunkMag va vb K (c0 + i)) + alpha)

/-- **The recursion of `fineOne`**: one more step. -/
theorem fineOne_step {va vb : Nat → F16} {K c0 n : Nat} :
    fineOne va vb K c0 (n + 1) =
      (1 + rho) * fineOne va vb K c0 n +
        (rho * (dev va vb K (c0 + n) + chunkMag va vb K (c0 + n)) + alpha) := by
  unfold fineOne
  rw [Finset.sum_range_succ, Finset.mul_sum]
  have hlast : (1 + rho) ^ (n + 1 - 1 - n) = 1 := by
    have h0 : n + 1 - 1 - n = 0 := by omega
    rw [h0, pow_zero]
  rw [hlast, one_mul]
  have hsum : ∑ x ∈ Finset.range n,
        (1 + rho) ^ (n + 1 - 1 - x) *
          (rho * (dev va vb K (c0 + x) + chunkMag va vb K (c0 + x)) + alpha)
      = ∑ i ∈ Finset.range n,
        (1 + rho) * ((1 + rho) ^ (n - 1 - i) *
          (rho * (dev va vb K (c0 + i) + chunkMag va vb K (c0 + i)) + alpha)) := by
    refine Finset.sum_congr rfl fun i hi => ?_
    have hi' : i < n := Finset.mem_range.mp hi
    have hexp : n + 1 - 1 - i = (n - 1 - i) + 1 := by omega
    rw [hexp, pow_succ]
    ring
  rw [hsum]

/-- `fineOne` is nonnegative. -/
theorem fineOne_nonneg {va vb : Nat → F16} {K c0 n : Nat} : 0 ≤ fineOne va vb K c0 n := by
  refine Finset.sum_nonneg fun i _ => ?_
  have h1 : (0 : ℚ) ≤ 1 + rho := by linarith [rho_nonneg]
  have h2 : (0 : ℚ) ≤ rho * (dev va vb K (c0 + i) + chunkMag va vb K (c0 + i)) + alpha := by
    have := dev_nonneg va vb K (c0 + i)
    have := chunkMag_nonneg va vb K (c0 + i)
    nlinarith [rho_nonneg, alpha_nonneg]
  exact mul_nonneg (pow_nonneg h1 _) h2

/-- The global fine bound: every chunk of `[0, steps K)` once, at the largest amplification its
position allows. -/
def tolDotR (va vb : Nat → F16) (K : Nat) : ℚ :=
  ∑ t ∈ Finset.range (steps K),
    (1 + rho) ^ (steps K - 1 - t) *
      (rho * (dev va vb K t + chunkMag va vb K t) + alpha)

/-- **The programs' fine errors add up to the global bound.**  The chunk runs `[c0 p, c1 p)` of the
programs partition `[0, steps K)`, so every chunk appears exactly once, and its amplification
`(1+ρ)^(n p - 1 - i)` is at most the global `(1+ρ)^(steps K - 1 - t)`. -/
theorem sum_fineOne_le {va vb : Nat → F16} {K : Nat} {S : Finset Nat} {c0 c1 : Nat → Nat}
    (hS : ∀ p ∈ S, c0 p ≤ c1 p) (hT : ∀ p ∈ S, c1 p ≤ steps K)
    (hpart : ∀ t, t < steps K → ∃! p, p ∈ S ∧ c0 p ≤ t ∧ t < c1 p) :
    ∑ p ∈ S, fineOne va vb K (c0 p) (c1 p - c0 p) ≤ tolDotR va vb K := by
  have hg : ∀ t : Nat, 0 ≤ rho * (dev va vb K t + chunkMag va vb K t) + alpha := by
    intro t
    have h1 := dev_nonneg va vb K t
    have h2 := chunkMag_nonneg va vb K t
    nlinarith [rho_nonneg, alpha_nonneg]
  have hbase : (1 : ℚ) ≤ 1 + rho := by linarith [rho_nonneg]
  set J : Nat → Finset Nat := fun p => (Finset.range (c1 p - c0 p)).image (fun i => c0 p + i) with hJ
  have hLHS : (∑ p ∈ S, fineOne va vb K (c0 p) (c1 p - c0 p))
      = ∑ p ∈ S, ∑ i ∈ Finset.range (c1 p - c0 p),
          (1 + rho) ^ (c1 p - c0 p - 1 - i) *
            (rho * (dev va vb K (c0 p + i) + chunkMag va vb K (c0 p + i)) + alpha) := by
    unfold fineOne
    rfl
  have hstep : (∑ p ∈ S, ∑ i ∈ Finset.range (c1 p - c0 p),
          (1 + rho) ^ (c1 p - c0 p - 1 - i) *
            (rho * (dev va vb K (c0 p + i) + chunkMag va vb K (c0 p + i)) + alpha))
      ≤ ∑ p ∈ S, ∑ i ∈ Finset.range (c1 p - c0 p),
          (1 + rho) ^ (steps K - 1 - (c0 p + i)) *
            (rho * (dev va vb K (c0 p + i) + chunkMag va vb K (c0 p + i)) + alpha) := by
    refine Finset.sum_le_sum fun p hp => Finset.sum_le_sum fun i hi => ?_
    have hlt : i < c1 p - c0 p := Finset.mem_range.mp hi
    have h01 : c0 p ≤ c1 p := hS p hp
    have hleN : c1 p ≤ steps K := hT p hp
    have hexp : c1 p - c0 p - 1 - i = c1 p - 1 - (c0 p + i) := by omega
    have hexp2 : c1 p - 1 - (c0 p + i) ≤ steps K - 1 - (c0 p + i) := by omega
    have hpow : (1 + rho) ^ (c1 p - c0 p - 1 - i)
        ≤ (1 + rho) ^ (steps K - 1 - (c0 p + i)) := by
      rw [hexp]
      exact pow_le_pow_right₀ hbase hexp2
    exact mul_le_mul_of_nonneg_right hpow (hg (c0 p + i))
  have hper : ∀ p ∈ S, (∑ i ∈ Finset.range (c1 p - c0 p),
          (1 + rho) ^ (steps K - 1 - (c0 p + i)) *
            (rho * (dev va vb K (c0 p + i) + chunkMag va vb K (c0 p + i)) + alpha))
      = ∑ t ∈ J p,
          (1 + rho) ^ (steps K - 1 - t) *
            (rho * (dev va vb K t + chunkMag va vb K t) + alpha) := by
    intro p hp
    have hinj : Set.InjOn (fun i => c0 p + i) ↑(Finset.range (c1 p - c0 p)) := by
      intro a _ b _ hab
      exact Nat.add_left_cancel hab
    rw [hJ]
    exact (Finset.sum_image (g := fun i => c0 p + i) (f := fun t => (1 + rho) ^ (steps K - 1 - t) *
      (rho * (dev va vb K t + chunkMag va vb K t) + alpha)) hinj).symm
  have hdisj : ∀ p ∈ S, ∀ q ∈ S, p ≠ q → Disjoint (J p) (J q) := by
    intro p hp q hq hpq
    rw [hJ]
    refine Finset.disjoint_left.mpr fun t ht htm => ?_
    simp only [Finset.mem_image, Finset.mem_range] at ht htm
    obtain ⟨i, hi, rfl⟩ := ht
    obtain ⟨j, hj, hji⟩ := htm
    have h01p : c0 p ≤ c1 p := hS p hp
    have h01q : c0 q ≤ c1 q := hS q hq
    have hleN : c1 p ≤ steps K := hT p hp
    have hltN : c0 p + i < steps K := by omega
    obtain ⟨r, hr, huniq⟩ := hpart (c0 p + i) hltN
    have hpr : p = r := huniq p ⟨hp, by omega, by omega⟩
    have hqr : q = r := huniq q ⟨hq, by omega, by omega⟩
    exact hpq (by omega)
  have hunion : S.biUnion J = Finset.range (steps K) := by
    ext t
    simp only [hJ, Finset.mem_biUnion, Finset.mem_image, Finset.mem_range]
    constructor
    · rintro ⟨p, hpS, i, hi, rfl⟩
      have h01 : c0 p ≤ c1 p := hS p hpS
      have hleN : c1 p ≤ steps K := hT p hpS
      omega
    · intro ht
      obtain ⟨p, hp, -⟩ := hpart t ht
      obtain ⟨hpS, hplt, hpt⟩ := hp
      exact ⟨p, hpS, t - c0 p, by omega, by omega⟩
  calc ∑ p ∈ S, fineOne va vb K (c0 p) (c1 p - c0 p)
      = ∑ p ∈ S, ∑ i ∈ Finset.range (c1 p - c0 p),
          (1 + rho) ^ (c1 p - c0 p - 1 - i) *
            (rho * (dev va vb K (c0 p + i) + chunkMag va vb K (c0 p + i)) + alpha) := hLHS
    _ ≤ ∑ p ∈ S, ∑ i ∈ Finset.range (c1 p - c0 p),
          (1 + rho) ^ (steps K - 1 - (c0 p + i)) *
            (rho * (dev va vb K (c0 p + i) + chunkMag va vb K (c0 p + i)) + alpha) := hstep
    _ = ∑ p ∈ S, ∑ t ∈ J p,
          (1 + rho) ^ (steps K - 1 - t) *
            (rho * (dev va vb K t + chunkMag va vb K t) + alpha) :=
          Finset.sum_congr rfl hper
    _ = ∑ t ∈ S.biUnion J,
          (1 + rho) ^ (steps K - 1 - t) *
            (rho * (dev va vb K t + chunkMag va vb K t) + alpha) := (Finset.sum_biUnion hdisj).symm
    _ = ∑ t ∈ Finset.range (steps K),
          (1 + rho) ^ (steps K - 1 - t) *
            (rho * (dev va vb K t + chunkMag va vb K t) + alpha) := by
          rw [hunion]
    _ = tolDotR va vb K := rfl

/-- The atomic part is bounded by `Σ_t |chunkSum t| ≤ Σ_t chunkMag t`. -/
theorem abs_chunkSum_sum_le {va vb : Nat → F16} {K : Nat} :
    (∑ t ∈ Finset.range (steps K), |chunkSum va vb K t|) ≤
      ∑ t ∈ Finset.range (steps K), chunkMag va vb K t :=
  Finset.sum_le_sum fun t _ => abs_chunkSum_le_chunkMag va vb K t

/-- `DotLoop.bound` with the atomic constants and at most 128 steps is at most `(2^-16 + 2^-31)·B + 2^-116`. -/
theorem bound_atomic_le {na : ℕ} (hna : (na : ℚ) ≤ 128) {B : ℚ} (hB : 0 ≤ B) :
    DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) na B ≤
      (2 ^ (-16 : ℤ) + 2 ^ (-31 : ℤ)) * B + 2 ^ (-116 : ℤ) := by
  have h16 : (2:ℚ)^(-16:ℤ) = 1/65536 := by norm_num
  have h31 : (2:ℚ)^(-31:ℤ) = 1/2147483648 := by norm_num
  have h116 : (2:ℚ)^(-116:ℤ) = 1/(2:ℚ)^116 := by norm_num
  rw [h16, h31, h116]
  unfold DotLoop.bound
  set x : ℚ := 2 ^ (-23 : ℤ) with hx
  set a : ℚ := 2 ^ (-124 : ℤ) with ha
  set u : ℚ := (na : ℚ) * x with hu
  have hx0 : 0 ≤ x := by rw [hx]; positivity
  have ha0 : 0 ≤ a := by rw [ha]; positivity
  have hu0 : 0 ≤ u := by rw [hu]; exact mul_nonneg (Nat.cast_nonneg na) hx0
  have hx128 : (128:ℚ) * x = 1/65536 := by rw [hx]; norm_num
  have hu16 : u ≤ 1/65536 := by
    rw [hu]
    calc (na:ℚ) * x ≤ 128 * x := mul_le_mul_of_nonneg_right hna hx0
      _ = 1/65536 := hx128
  have hu1 : u ≤ 1/2 := by linarith
  have hden : (0:ℚ) < 1 - u := by linarith
  have hp1 : (1 + x)^na * (1 - u) ≤ 1 := by
    rw [hu]
    exact DotLoop.pow_mul_le hx0 na (by rw [← hu]; linarith)
  have hp2 : (1 + x)^na ≤ (1 - u)⁻¹ := by
    rw [inv_eq_one_div, le_div_iff₀ hden]
    exact hp1
  have hp3 : (1 - u)⁻¹ ≤ 1 + u + 2*u^2 := by
    rw [inv_le_iff_one_le_mul₀ hden]
    have h1 : (0:ℚ) ≤ 1 - 2*u := by linarith
    have h2 : 0 ≤ u^2 * (1 - 2*u) := mul_nonneg (sq_nonneg u) h1
    nlinarith [h2]
  have hu2 : u^2 ≤ 1/4294967296 := by nlinarith [hu0, hu16]
  have hP1 : (1 + x)^na - 1 ≤ 1/65536 + 1/2147483648 := by
    nlinarith [hp2, hp3, hu0, hu16, hu2]
  have hP2 : (1 + x)^na ≤ 2 := by nlinarith [hp2, hp3, hu0, hu1]
  have hT1 : ((1 + x)^na - 1) * B ≤ (1/65536 + 1/2147483648) * B :=
    mul_le_mul_of_nonneg_right hP1 hB
  have hT2 : (na:ℚ) * (1 + x)^na * a ≤ 1/(2:ℚ)^116 := by
    have h1 : (1 + x)^na * a ≤ 2 * a := mul_le_mul_of_nonneg_right hP2 ha0
    have hpos : (0:ℚ) ≤ (1 + x)^na * a :=
      mul_nonneg (pow_nonneg (by linarith) na) ha0
    have h2 : (na:ℚ) * ((1 + x)^na * a) ≤ 128 * (2 * a) :=
      mul_le_mul hna h1 hpos (by norm_num)
    have h3 : (128:ℚ) * (2 * a) = 1/(2:ℚ)^116 := by rw [ha]; norm_num
    have h4 : (na:ℚ) * (1 + x)^na * a = (na:ℚ) * ((1 + x)^na * a) := by ring
    rw [h4]
    exact h2.trans_eq h3
  linarith [hT1, hT2]

/-- **The tolerance.**  The fine error `e` of the dot accumulation over one tile's chunk range,
plus the atomic error of combining the programs' partial sums (`na ≤ steps K` additions from `0`,
`DotLoop.bound 2^-23 2^-124`) is within `tol`.  This is the README's "Tolerances" chain: `ρ/2^-15`
and the atomic coefficient `2^-23·steps K` each take about half of `2^-15`, and `stepRel`'s extra
`3u` still fits. -/
theorem tol_close {va vb : Nat → F16} {K na : Nat} (_hK : 0 < K) (hK13 : K ≤ 2 ^ 13)
    (hna : na ≤ steps K) {e : ℚ} (he : 0 ≤ e) (hele : e ≤ tolDotR va vb K) :
    e + DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) na
        ((∑ t ∈ Finset.range (steps K), |chunkSum va vb K t|) + e) ≤ tol va vb K := by
  have h15 : (2:ℚ)^(-15:ℤ) = 1/32768 := by norm_num
  have h100 : (2:ℚ)^(-100:ℤ) = 1/1267650600228229401496703205376 := by norm_num
  have h23 : (2:ℚ)^(-23:ℤ) = 1/8388608 := by norm_num
  have h124 : (2:ℚ)^(-124:ℤ) = 1/21267647932558653966460912964485513216 := by norm_num
  have h16 : (2:ℚ)^(-16:ℤ) = 1/65536 := by norm_num
  have h31 : (2:ℚ)^(-31:ℤ) = 1/2147483648 := by norm_num
  have h116 : (2:ℚ)^(-116:ℤ) = 1/83076749736557242056487941267521536 := by norm_num
  have h112 : (2:ℚ)^(-112:ℤ) = 1/5192296858534827628530496329220096 := by norm_num
  have h119 : (2:ℚ)^(-119:ℤ) = 1/664613997892457936451903530140172288 := by norm_num
  have htol : tol va vb K = (1/32768) *
      ((∑ t ∈ Finset.range (steps K), dev va vb K t) + dev va vb K (steps K) +
        (∑ t ∈ Finset.range (steps K), chunkMag va vb K t)) +
      1/1267650600228229401496703205376 := by
    unfold tol tolMag
    rw [Finset.sum_range_succ, h15, h100]
  rw [htol, h23, h124]
  have hN128 : steps K ≤ 128 := steps_le_128 hK13
  set M : ℚ := ∑ t ∈ Finset.range (steps K), dev va vb K t with hM
  set D : ℚ := dev va vb K (steps K) with hD
  set Kk : ℚ := ∑ t ∈ Finset.range (steps K), chunkMag va vb K t with hKk
  set C : ℚ := ∑ t ∈ Finset.range (steps K), |chunkSum va vb K t| with hC
  have hM0 : 0 ≤ M := by rw [hM]; exact Finset.sum_nonneg fun t _ => dev_nonneg va vb K t
  have hKk0 : 0 ≤ Kk := by rw [hKk]; exact Finset.sum_nonneg fun t _ => chunkMag_nonneg va vb K t
  have hC0 : 0 ≤ C := by rw [hC]; exact Finset.sum_nonneg fun t _ => abs_nonneg _
  have hna128 : (na:ℚ) ≤ 128 := by
    have h1 : (na:ℚ) ≤ (steps K : ℚ) := by exact_mod_cast hna
    have h2 : (steps K : ℚ) ≤ 128 := by exact_mod_cast hN128
    linarith
  -- C ≤ M + D
  have hCD : C ≤ M + D := by
    rw [hC]
    have hle : ∀ t ∈ Finset.range (steps K), |chunkSum va vb K t| ≤ dev va vb K (t+1) := by
      intro t ht
      have h := abs_chunkSum_le_dev (va := va) (vb := vb) (K := K) (t := t+1) (by omega)
      simpa using h
    have h1 := Finset.sum_le_sum hle
    have h2 : (∑ t ∈ Finset.range (steps K), dev va vb K (t+1)) + dev va vb K 0
        = (∑ t ∈ Finset.range (steps K), dev va vb K t) + dev va vb K (steps K) := by
      rw [← Finset.sum_range_succ', Finset.sum_range_succ]
    rw [← hM, ← hD] at h2
    have hdev0 : (0:ℚ) ≤ dev va vb K 0 := dev_nonneg va vb K 0
    linarith
  have hCK : C ≤ Kk := by
    rw [hC, hKk]
    exact abs_chunkSum_sum_le (va := va) (vb := vb) (K := K)
  have h2C : 2*C ≤ M + D + Kk := by linarith
  -- the sum of the per-step bounds
  have hsum_g : (∑ t ∈ Finset.range (steps K),
        (rho * (dev va vb K t + chunkMag va vb K t) + alpha))
      = rho * (M + Kk) + (steps K : ℚ) * alpha := by
    rw [Finset.sum_add_distrib]
    have hdiv : ∑ t ∈ Finset.range (steps K),
          rho * (dev va vb K t + chunkMag va vb K t) = rho * (M + Kk) := by
      rw [← Finset.mul_sum]
      congr 1
      rw [Finset.sum_add_distrib, ← hM, ← hKk]
    have hconst : ∑ t ∈ Finset.range (steps K), alpha = (steps K : ℚ) * alpha := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    rw [hdiv, hconst]
  have hg : ∀ t : Nat, 0 ≤ rho * (dev va vb K t + chunkMag va vb K t) + alpha := by
    intro t
    have h1 := dev_nonneg va vb K t
    have h2 := chunkMag_nonneg va vb K t
    nlinarith [rho_nonneg, alpha_nonneg]
  have hbase : (1:ℚ) ≤ 1 + rho := by linarith [rho_nonneg]
  have hNr : (steps K : ℚ) * rho ≤ 1/100 := by
    have h1 : (steps K : ℚ) ≤ 128 := by exact_mod_cast hN128
    have h2 : rho ≤ 1/60000 := rho_le
    have h3 : (0:ℚ) ≤ rho := rho_nonneg
    have hm := mul_le_mul h1 h2 h3 (by norm_num : (0:ℚ) ≤ 128)
    linarith
  have hNr2 : (steps K : ℚ) * rho ≤ 1/2 := by linarith
  have hNa : (steps K : ℚ) * alpha ≤ 1/5192296858534827628530496329220096 := by
    have h1 : (steps K : ℚ) ≤ 128 := by exact_mod_cast hN128
    have h2 : alpha ≤ 2 ^ (-119 : ℤ) := alpha_le
    have h3 : (0:ℚ) ≤ alpha := alpha_nonneg
    have hm := mul_le_mul h1 h2 h3 (by norm_num : (0:ℚ) ≤ 128)
    have h4 : (128:ℚ) * 2 ^ (-119:ℤ) = 1/5192296858534827628530496329220096 := by
      rw [h119]; norm_num
    exact hm.trans_eq h4
  have hE : e ≤ (1 + 2 * ((steps K:ℚ) * rho)) * (rho * (M + Kk) + (steps K:ℚ) * alpha) := by
    refine le_trans hele ?_
    calc tolDotR va vb K
        = ∑ t ∈ Finset.range (steps K), (1 + rho) ^ (steps K - 1 - t) *
            (rho * (dev va vb K t + chunkMag va vb K t) + alpha) := rfl
      _ ≤ ∑ t ∈ Finset.range (steps K), (1 + 2 * ((steps K:ℚ) * rho)) *
            (rho * (dev va vb K t + chunkMag va vb K t) + alpha) := by
          refine Finset.sum_le_sum fun t ht => ?_
          have hpow : (1 + rho) ^ (steps K - 1 - t) ≤ 1 + 2 * ((steps K:ℚ) * rho) := by
            calc (1 + rho) ^ (steps K - 1 - t) ≤ (1 + rho) ^ (steps K) :=
                  pow_le_pow_right₀ hbase (by omega)
              _ ≤ 1 + 2 * ↑(steps K) * rho := DotLoop.pow_le rho_nonneg (steps K) hNr2
              _ = 1 + 2 * ((steps K:ℚ) * rho) := by ring
          exact mul_le_mul_of_nonneg_right hpow (hg t)
      _ = (1 + 2 * ((steps K:ℚ) * rho)) *
            ∑ t ∈ Finset.range (steps K), (rho * (dev va vb K t + chunkMag va vb K t) + alpha) := by
          rw [Finset.mul_sum]
      _ = (1 + 2 * ((steps K:ℚ) * rho)) * (rho * (M + Kk) + (steps K:ℚ) * alpha) := by
          rw [hsum_g]
  have hElin : e ≤ (17/1000000) * (M + Kk) + 1/2596148429267413814265248164610048 := by
    have hA0 : 0 ≤ rho * (M + Kk) := mul_nonneg rho_nonneg (by linarith)
    have hv0 : 0 ≤ (steps K:ℚ) * alpha := mul_nonneg (Nat.cast_nonneg _) alpha_nonneg
    have huA : ((steps K:ℚ) * rho) * (rho * (M + Kk)) ≤ (1/100) * (rho * (M + Kk)) :=
      mul_le_mul_of_nonneg_right hNr hA0
    have huB : ((steps K:ℚ) * rho) * ((steps K:ℚ) * alpha) ≤
        (1/100) * (1/5192296858534827628530496329220096) :=
      mul_le_mul hNr hNa hv0 (by norm_num)
    have hA : rho * (M + Kk) ≤ (1/60000) * (M + Kk) :=
      mul_le_mul_of_nonneg_right rho_le (by linarith)
    have h2s : (1 + 1/50) * (1/5192296858534827628530496329220096) ≤
        1/2596148429267413814265248164610048 := by norm_num
    nlinarith [hE, huA, huB, hA, h2s, hM0, hKk0, rho_nonneg, alpha_nonneg]
  have hb' : DotLoop.bound (1/8388608) (1/21267647932558653966460912964485513216) na (C + e) ≤
      (1/65536 + 1/2147483648) * (C + e) + 1/83076749736557242056487941267521536 := by
    have h := bound_atomic_le (na := na) hna128 (B := C + e) (by linarith)
    rw [h23, h124, h16, h31, h116] at h
    exact h
  have hD0 : 0 ≤ D := by rw [hD]; exact dev_nonneg va vb K (steps K)
  have hMDK : 0 ≤ M + D + Kk := by linarith
  nlinarith [hb', hElin, h2C, hC0, he, hM0, hKk0, hD0, hMDK]

/-- **The same, with the atomic bound taken at the larger magnitude `S`.**  The fine error `e`
of a program is at most `S`, and the atomic accumulation of the programs' partial sums adds
`DotLoop.bound 2^-23 2^-124 na (C + S)`; both together stay within `tol`. -/
theorem tol_close2 {va vb : Nat → F16} {K na : Nat} (hK : 0 < K) (hK13 : K ≤ 2 ^ 13)
    (hna : na ≤ steps K) {e S : ℚ} (heS : e ≤ S) (hS0 : 0 ≤ S)
    (hS : S ≤ tolDotR va vb K) :
    e + DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) na
        ((∑ t ∈ Finset.range (steps K), |chunkSum va vb K t|) + S) ≤ tol va vb K := by
  have h := tol_close (va := va) (vb := vb) (K := K) (na := na) hK hK13 hna hS0 hS
  linarith

/-- A program that runs every chunk from `0` has `fineOne` the global bound. -/
theorem fineOne_zero {va vb : Nat → F16} {K : Nat} : fineOne va vb K 0 (steps K) = tolDotR va vb K := by
  rw [fineOne, tolDotR]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Nat.zero_add]

/-- `tolDotR` is nonnegative. -/
theorem tolDotR_nonneg {va vb : Nat → F16} {K : Nat} : 0 ≤ tolDotR va vb K := by
  rw [← fineOne_zero]
  exact fineOne_nonneg

/-- One program's fine error over the whole tile, which is what the `tl.store` branch leaves in
the output, is already within `tol`. -/
theorem tolDotR_le_tol {va vb : Nat → F16} {K : Nat} (hK : 0 < K) (hK13 : K ≤ 2 ^ 13) :
    tolDotR va vb K ≤ tol va vb K := by
  have h := tol_close (va := va) (vb := vb) (K := K) (na := steps K)
    hK hK13 le_rfl tolDotR_nonneg le_rfl
  have hb := DotLoop.bound_nonneg (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ))
    (by norm_num) (by norm_num) (steps K)
    (B := (∑ t ∈ Finset.range (steps K), |chunkSum va vb K t|) + tolDotR va vb K)
    (by
      have h1 : (0 : ℚ) ≤ ∑ t ∈ Finset.range (steps K), |chunkSum va vb K t| :=
        Finset.sum_nonneg fun _ _ => abs_nonneg _
      linarith [tolDotR_nonneg (va := va) (vb := vb) (K := K)])
  linarith

end Claims.StreamKMatmul

