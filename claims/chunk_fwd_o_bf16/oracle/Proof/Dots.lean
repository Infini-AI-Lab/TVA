import Claim
import TVA.Proof
import Proof.Loop

/-!
# The two dots of a trip

`dotO_safe` and `dotA_safe`: `b_o ← tl.dot(b_q, b_h, b_o)` and `b_A ← tl.dot(b_q, b_k, b_A)` keep
the invariants `OInv`, `AInv`, from the lane values of the loaded tiles.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- A sum over `Fin n` equals the sum over `Finset.range n`. -/
theorem sum_fin_eq_range {n : Nat} (f : Nat → ℚ) : ∑ k : Fin n, f k = ∑ k ∈ Finset.range n, f k :=
  Fin.sum_univ_eq_sum_range f n

/-- `(t + 1) * bk = t * bk + bk`. -/
theorem succ_mul_bk (t bk : Nat) : (t + 1) * bk = t * bk + bk := Nat.succ_mul t bk

/-- **The inter-chunk dot of a trip**: `b_o ← tl.dot(b_q, b_h, b_o)`. -/
theorem dotO_safe (hP : Device E.P) {d : Inst} (hd : d.ok) {g : Geo} {r t : Nat}
    (ht : t < nK d) {bo : Tensor [BT, d.bk] F32} (hO : OInv d g r t bo)
    {vq : Tensor [BT, d.bk] BF16} {vh : Tensor [d.bk, d.bk] BF16}
    (hvq : ∀ (i : Fin BT) (j : Fin d.bk), vq.get2 i j = qTile d g t i j)
    (hvh : ∀ (k c : Fin d.bk), vh.get2 k c = hTile d g r t k c)
    {k : Tensor [BT, d.bk] F32 → Prog α}
    (hk : ∀ bo', OInv d g r (t + 1) bo' → Safe E R W Q m (k bo')) :
    Safe E R W Q m (tl.dot cfgBF16 vq vh bo >>= k) := by
  refine safe_dot (hP.dotTotal cfgBF16 vq vh bo) fun r1 hr1 => hk r1 ?_
  intro i c hrow hcol hok
  obtain ⟨hf, hb⟩ := hO i c hrow hcol hok
  have hva : ∀ k : Fin d.bk, DType.val? .bf16 (vq.get2 i k) =
      some (qPad d (g.lo + i) g.ih (t * d.bk + k)) := fun k => by
    rw [hvq]; exact val?_qTile d g t i k hrow (fun x hx => (hok.1 x hx).1)
  have hvb : ∀ k : Fin d.bk, DType.val? .bf16 (vh.get2 k c) =
      some (hPad d r g.ih (g.iv * d.bk + c) (t * d.bk + k)) := fun k => by
    rw [hvh]; exact val?_hTile d g r t k c hcol (fun x hx => (hok.1 x hx).2)
  have hroom : MQ d r (g.lo + i) g.ih (g.iv * d.bk + c) (t * d.bk) +
      ∑ k : Fin d.bk, L (qPad d (g.lo + i) g.ih (t * d.bk + k)) *
        L (hPad d r g.ih (g.iv * d.bk + c) (t * d.bk + k)) ≤ 2 ^ 100 := by
    rw [sum_fin_eq_range (fun k => L (qPad d (g.lo + i) g.ih (t * d.bk + k)) *
      L (hPad d r g.ih (g.iv * d.bk + c) (t * d.bk + k))), ← MQ_split]
    refine (MQ_mono d r _ _ _ (n := t * d.bk + d.bk) (m := nK d * d.bk) ?_).trans ?_
    · rw [← succ_mul_bk]; exact Nat.mul_le_mul_right _ ht
    · rw [MQ_eq_PQS _ _ _ _ (K_le_nK hd)]; exact hok.2
  have key := dotTrip hP (bk_le hd) hr1 i c _ _ hva hvb (trip_rel hd (lt_of_lt_of_le ht le_rfl))
    (le_of_lt (lt_of_lt_of_le ht (nK_le hd))) ⟨hf, hb⟩ (abs_SQ_le d r _ _ _ _) hroom
  rw [sum_fin_eq_range (fun k => qPad d (g.lo + i) g.ih (t * d.bk + k) *
    hPad d r g.ih (g.iv * d.bk + c) (t * d.bk + k)),
    sum_fin_eq_range (fun k => Device.liftMag .bf16 (qPad d (g.lo + i) g.ih (t * d.bk + k)) *
      Device.liftMag .bf16 (hPad d r g.ih (g.iv * d.bk + c) (t * d.bk + k)))] at key
  rw [succ_mul_bk, SQ_split, MQ_split]
  exact key

/-- **The score dot of a trip**: `b_A ← tl.dot(b_q, b_k, b_A)`. -/
theorem dotA_safe (hP : Device E.P) {d : Inst} (hd : d.ok) {g : Geo} {t : Nat}
    (ht : t < nK d) {bA : Tensor [BT, BT] F32} (hA : AInv d g t bA)
    {vq : Tensor [BT, d.bk] BF16} {vk : Tensor [d.bk, BT] BF16}
    (hvq : ∀ (i : Fin BT) (j : Fin d.bk), vq.get2 i j = qTile d g t i j)
    (hvk : ∀ (k : Fin d.bk) (i' : Fin BT), vk.get2 k i' = kTile d g t k i')
    {k : Tensor [BT, BT] F32 → Prog α}
    (hk : ∀ bA', AInv d g (t + 1) bA' → Safe E R W Q m (k bA')) :
    Safe E R W Q m (tl.dot cfgBF16 vq vk bA >>= k) := by
  refine safe_dot (hP.dotTotal cfgBF16 vq vk bA) fun r1 hr1 => hk r1 ?_
  intro i i' hrow hle hok
  obtain ⟨hf, hb⟩ := hA i i' hrow hle hok
  have hrow' : g.rowOk i' := by
    unfold Geo.rowOk at *
    have : (i' : Nat) ≤ i := hle
    omega
  have hva : ∀ k : Fin d.bk, DType.val? .bf16 (vq.get2 i k) =
      some (qPad d (g.lo + i) g.ih (t * d.bk + k)) := fun k => by
    rw [hvq]; exact val?_qTile d g t i k hrow (fun x hx => (hok.1 x hx).1)
  have hvb : ∀ k : Fin d.bk, DType.val? .bf16 (vk.get2 k i') =
      some (kPad d (g.lo + i') g.ih (t * d.bk + k)) := fun k => by
    rw [hvk]; exact val?_kTile d g t k i' hrow' (fun x hx => (hok.1 x hx).2)
  have hroom : MA d (g.lo + i) (g.lo + i') g.ih (t * d.bk) +
      ∑ k : Fin d.bk, L (qPad d (g.lo + i) g.ih (t * d.bk + k)) *
        L (kPad d (g.lo + i') g.ih (t * d.bk + k)) ≤ 2 ^ 100 := by
    rw [sum_fin_eq_range (fun k => L (qPad d (g.lo + i) g.ih (t * d.bk + k)) *
      L (kPad d (g.lo + i') g.ih (t * d.bk + k))), ← MA_split]
    refine (MA_mono d _ _ _ (n := t * d.bk + d.bk) (m := nK d * d.bk) ?_).trans ?_
    · rw [← succ_mul_bk]; exact Nat.mul_le_mul_right _ ht
    · rw [MA_eq_PA _ _ _ (K_le_nK hd)]; exact hok.2
  have key := dotTrip hP (bk_le hd) hr1 i i' _ _ hva hvb (trip_rel hd ht)
    (le_of_lt (lt_of_lt_of_le ht (nK_le hd))) ⟨hf, hb⟩ (abs_SA_le d _ _ _ _) hroom
  rw [sum_fin_eq_range (fun k => qPad d (g.lo + i) g.ih (t * d.bk + k) *
    kPad d (g.lo + i') g.ih (t * d.bk + k)),
    sum_fin_eq_range (fun k => Device.liftMag .bf16 (qPad d (g.lo + i) g.ih (t * d.bk + k)) *
      Device.liftMag .bf16 (kPad d (g.lo + i') g.ih (t * d.bk + k)))] at key
  rw [succ_mul_bk, SA_split, MA_split]
  exact key

end Claims.ChunkFwdO
