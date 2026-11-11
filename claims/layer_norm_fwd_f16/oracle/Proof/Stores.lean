import Proof.Defs

/-!
# The two scalar stores (lines 83–84)

`mean[t]` and `rstd[t]` are stored with `tl.store1`; the inputs stay as they were
(`MemOk.store1`), and the pointer `Mean + row` is element `t`.
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- A successful read finds every byte of the element present. -/
theorem Mem.present_of_read {m : Mem} {dy : DType} {a : Int} {v : dy.Val}
    (h : m.read dy a = some v) : ∀ z ∈ Mem.span dy a, m z ≠ none := by
  have key : ∀ (n : Nat) (a : Int) (bs : List Bytes.Byte), m.readBytes a n = some bs →
      ∀ i : Nat, i < n → m (a + i) ≠ none := by
    intro n
    induction n with
    | zero => intro a bs _ i hi; omega
    | succ n ih =>
      intro a bs hbs i hi
      unfold Mem.readBytes at hbs
      cases h0 : m a with
      | none => rw [h0] at hbs; simp at hbs
      | some b =>
        rcases i with _ | i
        · simp [h0]
        · cases h1 : m.readBytes (a + 1) n with
          | none => rw [h0, h1] at hbs; simp at hbs
          | some bs' =>
            have := ih (a + 1) bs' h1 i (by omega)
            have e : a + 1 + (i : Int) = a + ((i + 1 : Nat) : Int) := by push_cast; ring
            rwa [e] at this
  intro z hz
  simp only [Mem.read] at h
  split_ifs at h with hal
  · cases hb : m.readBytes a dy.bytes with
    | none => rw [hb] at h; simp at h
    | some bs =>
      have hzs : z = a + ((z - a).toNat : Nat) := by
        simp only [Mem.span, Set.mem_ofPred_eq] at hz
        omega
      rw [hzs]
      exact key dy.bytes a bs hb _ (by simp only [Mem.span, Set.mem_ofPred_eq] at hz; omega)

/-- A store that stays outside the inputs leaves the inputs as they were. -/
theorem MemOk.store1 {init m m1 : Mem} (hm : MemOk d init m) {dy : DType} {a : Int} {v : dy.Val}
    (hread : m1.read dy a = some v) (hframe : ∀ x, x ∉ Mem.span dy a → m1 x = m x)
    (hdis : Disjoint (Mem.span dy a) (inBytes d)) : MemOk d init m1 := by
  refine ⟨fun z hz => ?_, fun z hz => ?_⟩
  · have : z ∉ Mem.span dy a := fun hs => Set.disjoint_left.mp hdis hs hz
    rw [hframe z this, hm.1 z hz]
  · by_cases hs : z ∈ Mem.span dy a
    · exact Mem.present_of_read hread z hs
    · rw [hframe z hs]; exact hm.2 z hz

end Claims.LayerNormFwdF16
