import Proof.Defs

/-!
# The tiles' post-states give the specification

If every program's block of `c` holds a correct result and nothing else moved, the
specification holds.
-/

namespace Claims.MatmulE5m2

open TVA

/-- Covering the matrix by blocks: from per-program block correctness, `Post`. -/
theorem post_of_tiles {d : Inst} {init fin : Mem}
    (h : ∀ p, p < grid d → ∃ m, Q d init p m ∧ ∀ x ∈ R d init p ∪ W d init p, fin x = m x)
    (hfr : ∀ z, z ∉ d.c.elems (cIdx d) → fin z = init z) : Post d init fin := by
  refine ⟨?_, hfr⟩
  intro i j hi hj
  have hti : i / 128 < tM d := by
    unfold tM
    have h1 : i / 128 + 1 ≤ (d.M + 127) / 128 := by
      rw [Nat.le_div_iff_mul_le (by norm_num : 0 < 128)]
      have h2 : i / 128 * 128 ≤ i := Nat.div_mul_le_self i 128
      have h3 : i / 128 * 128 + 128 ≤ i + 128 := Nat.add_le_add_right h2 128
      have h4 : i + 128 ≤ d.M + 127 := by omega
      have e : (i / 128 + 1) * 128 = i / 128 * 128 + 128 := by ring
      rw [e]
      exact le_trans h3 h4
    omega
  have htj : j / 256 < tN d := by
    unfold tN
    have h1 : j / 256 + 1 ≤ (d.N + 255) / 256 := by
      rw [Nat.le_div_iff_mul_le (by norm_num : 0 < 256)]
      have h2 : j / 256 * 256 ≤ j := Nat.div_mul_le_self j 256
      have h3 : j / 256 * 256 + 256 ≤ j + 256 := Nat.add_le_add_right h2 256
      have h4 : j + 256 ≤ d.N + 255 := by omega
      have e : (j / 256 + 1) * 256 = j / 256 * 256 + 256 := by ring
      rw [e]
      exact le_trans h3 h4
    omega
  obtain ⟨p, hp, hpm, hpn⟩ := pid_surj hti htj
  obtain ⟨m, hQ, hagree⟩ := h p hp
  have hx : i % 128 < 128 := Nat.mod_lt _ (by norm_num)
  have hy : j % 256 < 256 := Nat.mod_lt _ (by norm_num)
  have hom : ocm d p (i % 128) = i := by
    have h := Nat.div_add_mod i 128
    unfold ocm
    rw [hpm, Nat.mul_comm (i / 128) 128]
    exact h
  have hon : ocn d p (j % 256) = j := by
    have h := Nat.div_add_mod j 256
    unfold ocn
    rw [hpn, Nat.mul_comm (j / 256) 256]
    exact h
  have hs := hQ (i % 128) (j % 256) hx hy (by rw [hom]; exact hi) (by rw [hon]; exact hj)
  rw [hom, hon] at hs
  obtain ⟨c, hc, H⟩ := hs
  have hmem : i * d.stride_cm + j * d.stride_cn ∈ cBlk d p :=
    ⟨i % 128, j % 256, hx, hy, by rw [hom]; exact hi, by rw [hon]; exact hj,
      by rw [hom, hon]⟩
  have hspan : Mem.span .f16 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr ⊆
      d.c.elems (cBlk d p) := Buf.span_sub_elems d.c hmem
  have hW : d.c.elems (cBlk d p) ⊆ R d init p ∪ W d init p := fun x hx' => Or.inr hx'
  have hfin : fin.read .f16 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr =
      m.read .f16 (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr :=
    Mem.read_congr fun x hx' => hagree x (hW (hspan hx'))
  exact ⟨c, by rw [hfin]; exact hc, H⟩

end Claims.MatmulE5m2
