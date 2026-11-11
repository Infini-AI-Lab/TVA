import Proof.Defs
import Proof.Facts
import Proof.Loop
import Proof.Steps

/-!
# `lora_shrink_bf16`: one trip of the loop

The four kinds of trip of the loop over `K` (lines 137-180) all preserve `LoopInv`.
-/

namespace Claims.LoraShrink

open TVA

/-- The body of one trip of the loop (lines 137-180), for the trip number `t`. -/
def bodyProg (d : Inst) (i t : Nat) (x : LState d) : Prog (LState d) :=
  if evenK d = true then do
    let tiled_b ← tl.load x.2.2
    let tiled_a ← tl.load x.2.1
    let accumulator ← tl.dot_add dotCfg tiled_a tiled_b x.1 (Tensor.full (DType.f32.ofBits 0))
    Prog.ret (accumulator, x.2.1 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [32, bk d] I32),
      x.2.2 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [bk d, 16] I32))
  else
    if (IntV.ofNat d.K : I32).le (IntV.ofNat t * IntV.ofNat (bk d * sk d) + IntV.ofNat (part d i * bk d)) = true then do
      let accumulator ← Prog.ret x.1
      Prog.ret (accumulator, x.2.1 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [32, bk d] I32),
        x.2.2 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [bk d, 16] I32))
    else
      if (IntV.ofNat t * IntV.ofNat (bk d * sk d) + IntV.ofNat (part d i * bk d) + IntV.ofNat (bk d) : I32).le
          (IntV.ofNat d.K) = true then
        blockFull d x
      else blockMasked d (IntV.ofNat t * IntV.ofNat (bk d * sk d) + IntV.ofNat (part d i * bk d)) x

/-- **One trip of the loop** preserves `LoopInv`. -/
theorem body_step {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre0 : Pre d init) {i : Nat} (hi : i < (launch d).size) (hactive : Active d i)
    (hs : slice d i < d.S) {t : Nat} (ht : t < numIters d) {st : LState d}
    (hI' : LoopInv d init i t st init) :
    Safe ((launch d).env P) (Rall d) ∅ (fun st' m' => LoopInv d init i (t + 1) st' m') init
      (bodyProg d i t st) := by
  unfold bodyProg
  have hbk256 : bk d ≤ 256 := by rcases bk_sk_cases d with ⟨h, -⟩ | ⟨h, -⟩ <;> omega
  have hSTEP : bk d * sk d < 2 ^ 31 := by
    rcases bk_sk_cases d with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> norm_num
  have hSTEP2 : bk d * sk d ≤ 2 ^ 14 := by
    rcases bk_sk_cases d with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> norm_num
  have hSTEPpos : 0 < bk d * sk d := Nat.mul_pos (bk_pos d) (sk_pos d)
  have hKtop := hd.K_top
  have hnum_lt : numIters d < 2 ^ 31 := by
    unfold numIters
    exact lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)
  have hp : part d i < sk d := Nat.mod_lt _ (sk_pos d)
  have htS : t * (bk d * sk d) < d.K := by
    have h1 := (Nat.le_div_iff_mul_le hSTEPpos).mp (show t + 1 ≤ numIters d from ht)
    rw [Nat.succ_mul] at h1
    omega
  have hpb : part d i * bk d ≤ 2 ^ 14 := by
    have h1 : part d i ≤ 63 := by
      have := hp; rcases bk_sk_cases d with ⟨-, h⟩ | ⟨-, h⟩ <;> omega
    have h2 : bk d ≤ 256 := by rcases bk_sk_cases d with ⟨h, -⟩ | ⟨h, -⟩ <;> omega
    calc part d i * bk d ≤ 63 * 256 := Nat.mul_le_mul h1 h2
      _ ≤ 2 ^ 14 := by norm_num
  have hbase : (IntV.ofNat t * IntV.ofNat (bk d * sk d) + IntV.ofNat (part d i * bk d) : I32).toInt =
      ((t * (bk d * sk d) + part d i * bk d : Nat) : Int) := by
    have ht31 : t < 2 ^ 31 := by omega
    exact IntV.I32.add_nat (IntV.I32.mul_nat (IntV.I32.toInt_ofNat ht31)
      (IntV.I32.toInt_ofNat hSTEP) (by omega)) (IntV.I32.toInt_ofNat (by omega)) (by omega)
  have hbk31 : bk d < 2 ^ 31 := by rcases bk_sk_cases d with ⟨h, -⟩ | ⟨h, -⟩ <;> omega
  have hle1 : ((IntV.ofNat d.K : I32).le (IntV.ofNat t * IntV.ofNat (bk d * sk d) +
      IntV.ofNat (part d i * bk d)) = true) ↔ d.K ≤ t * (bk d * sk d) + part d i * bk d := by
    rw [IntV.le_eq, hbase, IntV.I32.toInt_ofNat (by omega)]
    simp only [decide_eq_true_eq]
    exact Int.ofNat_le
  have hle2 : (((IntV.ofNat t * IntV.ofNat (bk d * sk d) + IntV.ofNat (part d i * bk d) +
      IntV.ofNat (bk d) : I32).le (IntV.ofNat d.K)) = true) ↔
        t * (bk d * sk d) + part d i * bk d + bk d ≤ d.K := by
    rw [IntV.le_eq, IntV.I32.add_nat hbase (IntV.I32.toInt_ofNat hbk31) (by omega),
      IntV.I32.toInt_ofNat (by omega)]
    simp only [decide_eq_true_eq]
    exact Int.ofNat_le
  have hsplit : ∀ b : Nat, (part d i + sk d * t) * bk d = t * (bk d * sk d) + part d i * bk d := by
    intro _; ring
  by_cases hev : evenK d = true
  · simp only [hev, ite_true]
    have ht' : t < stepsOf d (part d i) := by rw [evenK_stepsOf hev hp]; exact ht
    exact step_full hP hd hpre0 hi hactive hs hp ht' (evenK_block hev hp ht) hI'
  · simp only [hev]
    by_cases hk1 : d.K ≤ t * (bk d * sk d) + part d i * bk d
    · simp only [hle1.mpr hk1, ite_true]
      have hge : stepsOf d (part d i) ≤ t := by
        by_contra hlt
        have h1 := (lt_blocks_iff d _).mp ((lt_stepsOf_iff d hp t).mp (not_le.mp hlt))
        have := hsplit 0
        omega
      exact step_skip hge hI'
    · have hk1' : ¬ ((IntV.ofNat d.K : I32).le (IntV.ofNat t * IntV.ofNat (bk d * sk d) +
          IntV.ofNat (part d i * bk d)) = true) := fun h => hk1 (hle1.mp h)
      simp only [hk1']
      have ht' : t < stepsOf d (part d i) := by
        refine (lt_stepsOf_iff d hp t).mpr ((lt_blocks_iff d _).mpr ?_)
        have := hsplit 0
        omega
      by_cases hk2 : t * (bk d * sk d) + part d i * bk d + bk d ≤ d.K
      · simp only [hle2.mpr hk2, ite_true]
        refine step_full hP hd hpre0 hi hactive hs hp ht' ?_ hI'
        have := hsplit 0
        nlinarith
      · have hk2' : ¬ (((IntV.ofNat t * IntV.ofNat (bk d * sk d) + IntV.ofNat (part d i * bk d) +
            IntV.ofNat (bk d) : I32).le (IntV.ofNat d.K)) = true) := fun h => hk2 (hle2.mp h)
        simp only [hk2']
        refine step_masked hP hd hpre0 hi hactive hs hp ht' ?_ hbase hI'
        have := hsplit 0
        nlinarith

end Claims.LoraShrink
