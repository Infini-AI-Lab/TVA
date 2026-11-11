import Proof.Tile.Acc1
import Proof.Tile.Val

/-!
# From the loop's accumulator to `OutSpec`

The last step of a tile: the loop leaves, in lane `(x, y)`, a `float32` accumulator value
`S (2·steps K)` whose partial-value sequence obeys the chunk laws (`Acc1.Steps`).  `Acc1.final`
bounds it by `tol`; `cvt` rounds it to `float16`.  This file turns that into the claim's
`OutSpec`, matching the row and column values of `OutSpec` with the ones the lanes read
(`vaOf`, `vbOf`).
-/

namespace Claims.MatmulE5m2

open TVA

/-! ## Agreement below `K` -/

/-- `prodV` only looks at the indices below `K`. -/
theorem prodV_congr {K : Nat} {va vb va' vb' : Nat → F8E5}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    prodV va vb K = prodV va' vb' K := by
  funext e
  unfold prodV
  by_cases h : e < K
  · simp only [h, ite_true, ha e h, hb e h]
  · simp only [h, ite_false]

/-- `liftV` only looks at the indices below `K`. -/
theorem liftV_congr {K : Nat} {va vb va' vb' : Nat → F8E5}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    liftV va vb K = liftV va' vb' K := by
  funext e
  unfold liftV
  by_cases h : e < K
  · simp only [h, ite_true, ha e h, hb e h]
  · simp only [h, ite_false]

/-- `dotSum` only looks at the indices below `K`. -/
theorem dotSum_congr {K : Nat} {va vb va' vb' : Nat → F8E5}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    dotSum va vb K = dotSum va' vb' K := by
  unfold dotSum
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [Finset.mem_range] at hk
  rw [ha k hk, hb k hk]

/-- `tol` only looks at the indices below `K`. -/
theorem tol_congr {K : Nat} {va vb va' vb' : Nat → F8E5}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    tol va vb K = tol va' vb' K := by
  unfold tol
  rw [prodV_congr ha hb, liftV_congr ha hb]

/-! ## The lane's value is `OutSpec` -/

/-- **The lane's contribution to `Post`.**  If the finished memory reads the stored `float16`
value `c` at the lane's element of `c`, `c` is the `float16` rounding of a `float32` accumulator
value `acc` whose partial-value sequence `S` obeys the chunk laws of the reduction, then the
element `(ocm d p x, ocn d p y)` of `Post` holds. -/
theorem outSpec_of_acc {d : Inst} (hd : d.ok) {init fin : Mem} {p x y : Nat}
    (hx : ocm d p x < d.M) (hy : ocn d p y < d.N)
    {acc : F32} {c : F16} {S : Nat → ℚ}
    (hread : fin.read .f16 (d.c.ptr (cOff d p x y)).addr = some c)
    (hne : FloatV.NaNEq (FloatV.cvt acc) c)
    (hval : DType.val? .f32 acc = some (S (2 * steps d.K)))
    (hS0 : S 0 = 0)
    (hSteps : Steps (prodV (vaOf d init p x) (vbOf d init p y) d.K)
      (liftV (vaOf d init p x) (vbOf d init p y) d.K) S (2 * steps d.K)) :
    OutSpec d init fin (ocm d p x) (ocn d p y) := by
  have h128 : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.2.1
  have hK64 : 64 ∣ d.K := (by norm_num : 64 ∣ 128).trans h128
  obtain ⟨hfin, hrat⟩ := DType.val?_f32.mp hval
  have hbound : |acc.toRat - dotSum (vaOf d init p x) (vbOf d init p y) d.K| ≤
      tol (vaOf d init p x) (vbOf d init p y) d.K := by
    rw [hrat]
    exact final hK64 hS0 hSteps
  refine ⟨c, ?_, ?_⟩
  · simpa only [cOff] using hread
  · intro va vb hva hvb
    have hva' : ∀ k, k < d.K → va k = vaOf d init p x k := by
      intro k hk
      have h1 := (hva k hk).1
      have hoff : aOff d p x k = ocm d p x * d.stride_am + k * d.stride_ak := by
        unfold aOff
        rw [om_eq hx]
      have h2 : vaOf d init p x k = va k := by
        simp only [vaOf, hoff, h1, Option.getD_some]
      exact h2.symm
    have hvb' : ∀ k, k < d.K → vb k = vbOf d init p y k := by
      intro k hk
      have h1 := (hvb k hk).1
      have hoff : bOff d p y k = k * d.stride_bk + ocn d p y * d.stride_bn := by
        unfold bOff
        rw [on_eq hy]
      have h2 : vbOf d init p y k = vb k := by
        simp only [vbOf, hoff, h1, Option.getD_some]
      exact h2.symm
    have hds : dotSum va vb d.K = dotSum (vaOf d init p x) (vbOf d init p y) d.K :=
      dotSum_congr hva' hvb'
    have htl : tol va vb d.K = tol (vaOf d init p x) (vbOf d init p y) d.K :=
      tol_congr hva' hvb'
    have hb' : |acc.toRat - dotSum va vb d.K| ≤ tol va vb d.K := by
      rw [hds, htl]
      exact hbound
    refine ⟨acc, hfin, hb', hne, ?_⟩
    intro hroom
    have hle : |acc.toRat| ≤ |dotSum va vb d.K| + tol va vb d.K := by
      have h1 := abs_add_le (acc.toRat - dotSum va vb d.K) (dotSum va vb d.K)
      have h2 : acc.toRat - dotSum va vb d.K + dotSum va vb d.K = acc.toRat := by ring
      rw [h2] at h1
      linarith
    have hmax : |acc.toRat| ≤ Format.binary16.maxVal := by linarith
    obtain ⟨hcfin, hcerr⟩ := FloatV.cvt_err (g := Format.binary16)
      Format.binary16_regular hfin hmax
    have hceq : c = FloatV.cvt acc :=
      hne.eq (FloatV.isFinite_iff.mp hcfin).1
    refine ⟨?_, ?_⟩
    · rw [hceq]
      exact hcfin
    · rw [hceq]
      have hu : (0 : ℚ) ≤ Format.binary16.u :=
        le_of_lt Format.u_pos
      have hmul : |acc.toRat| * Format.binary16.u ≤
          (|dotSum va vb d.K| + tol va vb d.K) * Format.binary16.u :=
        mul_le_mul_of_nonneg_right hle hu
      have htri : |(FloatV.cvt acc : F16).toRat - dotSum va vb d.K| ≤
          |(FloatV.cvt acc : F16).toRat - acc.toRat| + |acc.toRat - dotSum va vb d.K| := by
        have h1 := abs_add_le ((FloatV.cvt acc : F16).toRat - acc.toRat)
          (acc.toRat - dotSum va vb d.K)
        have h2 : (FloatV.cvt acc : F16).toRat - acc.toRat + (acc.toRat - dotSum va vb d.K)
            = (FloatV.cvt acc : F16).toRat - dotSum va vb d.K := by ring
        rw [h2] at h1
        exact h1
      linarith

/-- **The lane's contribution to `Post`, conditionally.**  The same as `outSpec_of_acc`, with the
partial-value sequence promised only for the lanes whose row and column values are all finite:
`OutSpec` is vacuous on the others (its body quantifies over the row and column values *and*
their finiteness). -/
theorem outSpec_of_acc_cond {d : Inst} (hd : d.ok) {init fin : Mem} {p x y : Nat}
    (hx : ocm d p x < d.M) (hy : ocn d p y < d.N)
    {acc : F32} {c : F16}
    (hread : fin.read .f16 (d.c.ptr (cOff d p x y)).addr = some c)
    (hne : FloatV.NaNEq (FloatV.cvt acc) c)
    (hv : (∀ k, k < d.K → (vaOf d init p x k).isFinite = true) →
      (∀ k, k < d.K → (vbOf d init p y k).isFinite = true) →
      ∃ S : Nat → ℚ, S 0 = 0 ∧
        Steps (prodV (vaOf d init p x) (vbOf d init p y) d.K)
          (liftV (vaOf d init p x) (vbOf d init p y) d.K) S (2 * steps d.K) ∧
        DType.val? .f32 acc = some (S (2 * steps d.K))) :
    OutSpec d init fin (ocm d p x) (ocn d p y) := by
  have h128 : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.2.1
  have hK64 : 64 ∣ d.K := (by norm_num : 64 ∣ 128).trans h128
  refine ⟨c, ?_, ?_⟩
  · simpa only [cOff] using hread
  · intro va vb hva hvb
    have hva' : ∀ k, k < d.K → va k = vaOf d init p x k := by
      intro k hk
      have h1 := (hva k hk).1
      have hoff : aOff d p x k = ocm d p x * d.stride_am + k * d.stride_ak := by
        unfold aOff
        rw [om_eq hx]
      have h2 : vaOf d init p x k = va k := by
        simp only [vaOf, hoff, h1, Option.getD_some]
      exact h2.symm
    have hvb' : ∀ k, k < d.K → vb k = vbOf d init p y k := by
      intro k hk
      have h1 := (hvb k hk).1
      have hoff : bOff d p y k = k * d.stride_bk + ocn d p y * d.stride_bn := by
        unfold bOff
        rw [on_eq hy]
      have h2 : vbOf d init p y k = vb k := by
        simp only [vbOf, hoff, h1, Option.getD_some]
      exact h2.symm
    have hfinA : ∀ k, k < d.K → (vaOf d init p x k).isFinite = true := by
      intro k hk
      rw [← hva' k hk]
      exact (hva k hk).2
    have hfinB : ∀ k, k < d.K → (vbOf d init p y k).isFinite = true := by
      intro k hk
      rw [← hvb' k hk]
      exact (hvb k hk).2
    obtain ⟨S, hS0, hSteps, hval⟩ := hv hfinA hfinB
    obtain ⟨hfin, hrat⟩ := DType.val?_f32.mp hval
    have hbound : |acc.toRat - dotSum (vaOf d init p x) (vbOf d init p y) d.K| ≤
        tol (vaOf d init p x) (vbOf d init p y) d.K := by
      rw [hrat]
      exact final hK64 hS0 hSteps
    have hds : dotSum va vb d.K = dotSum (vaOf d init p x) (vbOf d init p y) d.K :=
      dotSum_congr hva' hvb'
    have htl : tol va vb d.K = tol (vaOf d init p x) (vbOf d init p y) d.K :=
      tol_congr hva' hvb'
    have hb' : |acc.toRat - dotSum va vb d.K| ≤ tol va vb d.K := by
      rw [hds, htl]
      exact hbound
    refine ⟨acc, hfin, hb', hne, ?_⟩
    intro hroom
    have hle : |acc.toRat| ≤ |dotSum va vb d.K| + tol va vb d.K := by
      have h1 := abs_add_le (acc.toRat - dotSum va vb d.K) (dotSum va vb d.K)
      have h2 : acc.toRat - dotSum va vb d.K + dotSum va vb d.K = acc.toRat := by ring
      rw [h2] at h1
      linarith
    have hmax : |acc.toRat| ≤ Format.binary16.maxVal := by linarith
    obtain ⟨hcfin, hcerr⟩ := FloatV.cvt_err (g := Format.binary16)
      Format.binary16_regular hfin hmax
    have hceq : c = FloatV.cvt acc :=
      hne.eq (FloatV.isFinite_iff.mp hcfin).1
    refine ⟨?_, ?_⟩
    · rw [hceq]
      exact hcfin
    · rw [hceq]
      have hu : (0 : ℚ) ≤ Format.binary16.u :=
        le_of_lt Format.u_pos
      have hmul : |acc.toRat| * Format.binary16.u ≤
          (|dotSum va vb d.K| + tol va vb d.K) * Format.binary16.u :=
        mul_le_mul_of_nonneg_right hle hu
      have htri : |(FloatV.cvt acc : F16).toRat - dotSum va vb d.K| ≤
          |(FloatV.cvt acc : F16).toRat - acc.toRat| + |acc.toRat - dotSum va vb d.K| := by
        have h1 := abs_add_le ((FloatV.cvt acc : F16).toRat - acc.toRat)
          (acc.toRat - dotSum va vb d.K)
        have h2 : (FloatV.cvt acc : F16).toRat - acc.toRat + (acc.toRat - dotSum va vb d.K)
            = (FloatV.cvt acc : F16).toRat - dotSum va vb d.K := by ring
        rw [h2] at h1
        exact h1
      linarith

end Claims.MatmulE5m2
