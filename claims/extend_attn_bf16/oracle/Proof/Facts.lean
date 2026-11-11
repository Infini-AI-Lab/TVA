import Proof.Scalars

/-!
# What `Pre` gives about a request
-/

namespace Claims.ExtendAttn

open TVA

/-- The sizes of request `b`: its new tokens lie among the `T`, its cached-part slots among the `NI`,
and the lengths are bounded. -/
theorem req_sizes {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.batch) :
    qsOf d init b + extOf d init b ≤ d.T ∧ extOf d init b ≤ d.maxExt ∧
      ksOf d init b + preOf d init b ≤ d.NI ∧ preOf d init b + extOf d init b ≤ 2 ^ 18 := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hreq, -⟩ := hpre
  obtain ⟨q0, q1, k0, k1, hq0, hq1, hk0, hk1, h0, h01, h1T, h1m, hk, hk01, hkNI, hsum, -⟩ :=
    hreq b hb
  have e0 : valQo d init b = q0 := by simp [valQo, hq0]
  have e1 : valQo d init (b + 1) = q1 := by simp [valQo, hq1]
  have f0 : val32 init d.kvp b = k0.toInt := by unfold val32; rw [hk0]
  have f1 : val32 init d.kvp (b + 1) = k1.toInt := by unfold val32; rw [hk1]
  have hqs : qsOf d init b = q0.toNat := by simp [qsOf, e0]
  have hext : extOf d init b = (q1 - q0).toNat := by simp [extOf, e0, e1]
  have hks : ksOf d init b = k0.toInt.toNat := by simp [ksOf, f0]
  have hpr : preOf d init b = (k1.toInt - k0.toInt).toNat := by simp [preOf, f0, f1]
  rw [hqs, hext, hks, hpr]
  refine ⟨by omega, by omega, by omega, by omega⟩

end Claims.ExtendAttn
