import Proof.Pad

/-!
# `slot_mappings_i64`: the context-parallel local position

Lines 125-140 compute each lane's local position and whether the rank stores it. `cpSec` is that
part of the model; `safe_cpSec` shows that on every lane of nonnegative position the results are
`cpPos` and `cpLocal`, and that no `//` or `%` is undefined on any lane (every divisor is
positive).
-/

namespace Claims.SlotMappings

open TVA

/-! ## Positive divisors -/

theorem bits_ne_of_pos' {w : Nat} {b : IntV true w} (hb : 0 < b.toInt) :
    b.bits ≠ 0 ∧ b.bits ≠ -1 := by
  simp only [IntV.toInt, ite_true] at hb
  refine ⟨fun h => by rw [h] at hb; simp at hb, fun h => ?_⟩
  rw [h, show (-1 : BitVec w) = BitVec.allOnes w from BitVec.neg_one_eq_allOnes,
    BitVec.toInt_allOnes] at hb
  split_ifs at hb <;> omega

/-- `a // b` by a positive `b` is defined, whatever `a`. -/
theorem div?_some {w : Nat} (a : IntV true w) {b : IntV true w} (hb : 0 < b.toInt) :
    ∃ q, a.div? b = some q := by
  obtain ⟨h0, h1⟩ := bits_ne_of_pos' hb
  exact ⟨⟨a.bits.sdiv b.bits⟩, by simp only [IntV.div?, h0, h1, and_false, ↓reduceIte]⟩

/-- `a % b` by a positive `b` is defined, whatever `a`. -/
theorem rem?_some {w : Nat} (a : IntV true w) {b : IntV true w} (hb : 0 < b.toInt) :
    ∃ q, a.rem? b = some q := by
  obtain ⟨h0, h1⟩ := bits_ne_of_pos' hb
  exact ⟨⟨a.bits.srem b.bits⟩, by simp only [IntV.rem?, h0, h1, and_false, ↓reduceIte]⟩

/-! ## The local position -/

/-- Lines 125-140: `(local_positions, is_local)` from the `int64` positions. -/
def cpSec (C I : Nat) (kv_block_size : I32) (bs64 : I64) (sharded : Bool) (cp_rank : I32)
    (positions : Tensor [1024] I64) : Prog (Tensor [1024] I64 × Tensor [1024] Bool) :=
  if C = 1 then
    pure (positions, tl.full [1024] true)
  else do
    let virtual_block_size : I32 := kv_block_size * (IntV.ofNat C : I32)
    let virtual_block_indices ← tl.idiv positions (bc (IntV.cast virtual_block_size : I64))
    let virtual_block_offsets ← tl.irem positions (bc (IntV.cast virtual_block_size : I64))
    let q ← tl.idiv virtual_block_offsets (bc (IntV.ofNat I : I64))
    let q ← tl.irem q (bc (IntV.ofNat C : I64))
    let is_local := Tensor.eq q (bc (IntV.cast cp_rank : I64))
    let rounds ← tl.idiv virtual_block_offsets
      (bc (IntV.ofNat (I * C) : I64))
    let remainder ← tl.irem virtual_block_offsets (bc (IntV.ofNat I : I64))
    let local_offsets := rounds * bc (IntV.ofNat I : I64) + remainder
    let local_positions := virtual_block_indices * bc bs64 + local_offsets
    let local_positions := tl.where (bc sharded) local_positions positions
    let is_local := Tensor.or (bc (!sharded)) is_local
    pure (local_positions, is_local)

/-- The local position of a nonnegative position is nonnegative and at most the position. -/
theorem cpPos_range (d : Inst) (g : Nat) (hB : 1 ≤ d.bsz g) (hC : 1 ≤ d.CP_SIZE)
    (hI : 1 ≤ d.CP_INTERLEAVE) {p : Int} (hp : 0 ≤ p) : 0 ≤ cpPos d g p ∧ cpPos d g p ≤ p := by
  unfold cpPos
  split_ifs with h
  · exact ⟨hp, le_refl _⟩
  · set B : Int := (d.bsz g : Int)
    set C : Int := (d.CP_SIZE : Int)
    set I : Int := (d.CP_INTERLEAVE : Int)
    have hB' : (1 : Int) ≤ B := by simp only [B]; exact_mod_cast hB
    have hC' : (1 : Int) ≤ C := by simp only [C]; exact_mod_cast hC
    have hI' : (1 : Int) ≤ I := by simp only [I]; exact_mod_cast hI
    have hV : ((d.bsz g * d.CP_SIZE : Nat) : Int) = B * C := by push_cast; rfl
    have hIC : ((d.CP_INTERLEAVE * d.CP_SIZE : Nat) : Int) = I * C := by push_cast; rfl
    rw [hV, hIC]
    have hBC : 0 < B * C := by positivity
    have hIC0 : 0 < I * C := by positivity
    have e1 := Int.ediv_mul_add_emod p (B * C)
    have q1 : 0 ≤ p / (B * C) := Int.ediv_nonneg hp hBC.le
    have m1 : 0 ≤ p % (B * C) := Int.emod_nonneg _ hBC.ne'
    set r := p % (B * C)
    have e2 := Int.ediv_mul_add_emod r (I * C)
    have q2 : 0 ≤ r / (I * C) := Int.ediv_nonneg m1 hIC0.le
    have m2 : 0 ≤ r % (I * C) := Int.emod_nonneg _ hIC0.ne'
    have hmod : r % I = r % (I * C) % I := (Int.emod_emod_of_dvd r (Dvd.intro C rfl)).symm
    have m3 : 0 ≤ r % I := Int.emod_nonneg _ (by omega)
    have m4 : r % (I * C) % I ≤ r % (I * C) := by
      have := Int.ediv_mul_add_emod (r % (I * C)) I
      have := Int.ediv_nonneg m2 (show (0 : Int) ≤ I by omega)
      nlinarith
    have h1 : p / (B * C) * B ≤ p / (B * C) * (B * C) := by
      apply mul_le_mul_of_nonneg_left _ q1; nlinarith
    have h2 : r / (I * C) * I ≤ r / (I * C) * (I * C) := by
      apply mul_le_mul_of_nonneg_left _ q2; nlinarith
    refine ⟨by positivity, ?_⟩
    rw [hmod]
    nlinarith

/-- `cpLocal` without context parallelism or for an unsharded group. -/
theorem cpLocal_of (d : Inst) (g : Nat) (p : Int) (h : d.CP_SIZE = 1 ∨ d.sh g = false) :
    cpLocal d g p := by
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)

/-- **The local position, lane by lane.** On a lane of nonnegative position `p`, `cpSec` gives
`cpPos p` and `cpLocal p`. -/
theorem safe_cpSec {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {d : Inst} (hd : d.ok) {g : Nat} (hg : g < d.G) {kv : I32} (hkv : kv.toInt = d.bsz g)
    {sharded : Bool} (hsh : d.CP_SIZE ≠ 1 → sharded = d.sh g) {rank : I32}
    (hrank : rank.toInt = d.cp_rank) (positions : Tensor [1024] I64)
    {k : Tensor [1024] I64 × Tensor [1024] Bool → Prog α}
    (hk : ∀ lp il, (∀ l, 0 ≤ (positions.get l).toInt →
        il.get l = decide (cpLocal d g (positions.get l).toInt) ∧
          (lp.get l).toInt = cpPos d g (positions.get l).toInt) →
      Safe E R W Q m (k (lp, il))) :
    Safe E R W Q m (cpSec d.CP_SIZE d.CP_INTERLEAVE kv (IntV.cast kv) sharded rank positions >>= k) := by
  unfold cpSec
  by_cases hC : d.CP_SIZE = 1
  · rw [ite_eq_left hC, pure_bind]
    refine hk _ _ fun l _ => ⟨?_, ?_⟩
    · simp only [tl.full, Tensor.get_full]
      exact (decide_eq_true (cpLocal_of d g _ (Or.inl hC))).symm
    · simp only [cpPos, hC, true_or, ite_true]
  rw [ite_eq_right hC]
  have hsh' := hsh hC
  have hB := hd.bsz_pos g hg
  have hBC := hd.bsz_cp g hg
  have hC1 := hd.cp_pos
  have hI1 := hd.il_pos
  have hIC := hd.il_cp
  have hCl : d.CP_SIZE < 2 ^ 31 := by
    have : d.CP_SIZE ≤ d.CP_INTERLEAVE * d.CP_SIZE := Nat.le_mul_of_pos_left _ (by omega)
    omega
  have hIl : d.CP_INTERLEAVE < 2 ^ 31 := by
    have : d.CP_INTERLEAVE ≤ d.CP_INTERLEAVE * d.CP_SIZE := Nat.le_mul_of_pos_right _ (by omega)
    omega
  -- the divisors
  have hV : (IntV.cast (kv * (IntV.ofNat d.CP_SIZE : I32)) : I64).toInt =
      ((d.bsz g * d.CP_SIZE : Nat) : Int) := by
    rw [toInt_cast64]; exact IntV.I32.mul_nat hkv (toInt_ofNat32 hCl) hBC
  have hV0 : (0 : Int) < ((d.bsz g * d.CP_SIZE : Nat) : Int) := by
    have : 0 < d.bsz g * d.CP_SIZE := Nat.mul_pos (by omega) (by omega)
    exact_mod_cast this
  have hI : (IntV.ofNat d.CP_INTERLEAVE : I64).toInt = (d.CP_INTERLEAVE : Int) :=
    toInt_ofNat64 (by omega)
  have hCv : (IntV.ofNat d.CP_SIZE : I64).toInt = (d.CP_SIZE : Int) := toInt_ofNat64 (by omega)
  have hICv : (IntV.ofNat (d.CP_INTERLEAVE * d.CP_SIZE) : I64).toInt =
      ((d.CP_INTERLEAVE * d.CP_SIZE : Nat) : Int) := toInt_ofNat64 (by omega)
  have hI0 : (0 : Int) < d.CP_INTERLEAVE := by exact_mod_cast hI1
  have hC0 : (0 : Int) < d.CP_SIZE := by exact_mod_cast hC1
  have hIC0 : (0 : Int) < ((d.CP_INTERLEAVE * d.CP_SIZE : Nat) : Int) := by
    have : 0 < d.CP_INTERLEAVE * d.CP_SIZE := Nat.mul_pos (by omega) (by omega)
    exact_mod_cast this
  have hbs : (IntV.cast kv : I64).toInt = (d.bsz g : Int) := by rw [toInt_cast64, hkv]
  simp only [bind_assoc]
  refine safe_idiv (fun l => div?_some _ (by rw [Tensor.get_full, hV]; exact hV0)) fun r1 h1 => ?_
  refine safe_irem (fun l => rem?_some _ (by rw [Tensor.get_full, hV]; exact hV0)) fun r2 h2 => ?_
  refine safe_idiv (fun l => div?_some _ (by rw [Tensor.get_full, hI]; exact hI0)) fun r3 h3 => ?_
  refine safe_irem (fun l => rem?_some _ (by rw [Tensor.get_full, hCv]; exact hC0)) fun r4 h4 => ?_
  refine safe_idiv (fun l => div?_some _ (by rw [Tensor.get_full, hICv]; exact hIC0)) fun r5 h5 => ?_
  refine safe_irem (fun l => rem?_some _ (by rw [Tensor.get_full, hI]; exact hI0)) fun r6 h6 => ?_
  rw [pure_bind]
  refine hk _ _ fun l hp => ?_
  set p := (positions.get l).toInt with hpdef
  -- the lanes' values
  have v1 : (r1.get l).toInt = p / ((d.bsz g * d.CP_SIZE : Nat) : Int) := by
    have := IntV.toInt_of_div? (h1 l) hp (by rw [Tensor.get_full, hV]; exact hV0)
    rw [this, Tensor.get_full, hV]
  have v2 : (r2.get l).toInt = p % ((d.bsz g * d.CP_SIZE : Nat) : Int) := by
    have := IntV.toInt_of_rem? (h2 l) hp (by rw [Tensor.get_full, hV]; exact hV0)
    rw [this, Tensor.get_full, hV]
  have r2nn : 0 ≤ (r2.get l).toInt := by rw [v2]; exact Int.emod_nonneg _ hV0.ne'
  have r2lt : (r2.get l).toInt < ((d.bsz g * d.CP_SIZE : Nat) : Int) := by
    rw [v2]; exact Int.emod_lt_of_pos _ hV0
  have v3 : (r3.get l).toInt = (r2.get l).toInt / (d.CP_INTERLEAVE : Int) := by
    have := IntV.toInt_of_div? (h3 l) r2nn (by rw [Tensor.get_full, hI]; exact hI0)
    rw [this, Tensor.get_full, hI]
  have v4 : (r4.get l).toInt = (r3.get l).toInt % (d.CP_SIZE : Int) := by
    have := IntV.toInt_of_rem? (h4 l) (by rw [v3]; exact Int.ediv_nonneg r2nn hI0.le)
      (by rw [Tensor.get_full, hCv]; exact hC0)
    rw [this, Tensor.get_full, hCv]
  have v5 : (r5.get l).toInt = (r2.get l).toInt / ((d.CP_INTERLEAVE * d.CP_SIZE : Nat) : Int) := by
    have := IntV.toInt_of_div? (h5 l) r2nn (by rw [Tensor.get_full, hICv]; exact hIC0)
    rw [this, Tensor.get_full, hICv]
  have v6 : (r6.get l).toInt = (r2.get l).toInt % (d.CP_INTERLEAVE : Int) := by
    have := IntV.toInt_of_rem? (h6 l) r2nn (by rw [Tensor.get_full, hI]; exact hI0)
    rw [this, Tensor.get_full, hI]
  have hp63 : p < 2 ^ 63 := (IntV.sfits_toInt (positions.get l)).2
  cases hs : sharded
  · -- an unsharded group: the position itself, always the rank's
    rw [hs] at hsh'
    refine ⟨?_, ?_⟩
    · simp only [Tensor.get_or, Tensor.get_full, Bool.not_false, Bool.true_or]
      exact (decide_eq_true (cpLocal_of d g _ (Or.inr hsh'.symm))).symm
    · simp only [tl.where, Tensor.get_where, Tensor.get_full, Bool.false_eq_true, ite_false]
      simp only [cpPos, hsh'.symm, or_true, ite_true]
      rfl
  · rw [hs] at hsh'
    have hnot : ¬ (d.CP_SIZE = 1 ∨ d.sh g = false) := by
      rw [← hsh']; simp [hC]
    have hloc : (r4.get l).toInt = p % ((d.bsz g * d.CP_SIZE : Nat) : Int) /
        (d.CP_INTERLEAVE : Int) % (d.CP_SIZE : Int) := by rw [v4, v3, v2]
    refine ⟨?_, ?_⟩
    · simp only [Tensor.get_or, Tensor.get_full, Bool.not_true, Bool.false_or, Tensor.get_eq,
        IntV.eq_eq, toInt_cast64, hrank, hloc]
      unfold cpLocal
      apply decide_eq_decide.mpr
      constructor
      · intro h; exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact absurd (Or.inl h) hnot
        · exact absurd (Or.inr h) hnot
        · exact h
    · -- `virtual_block_indices * kv_block_size + (rounds * CP_INTERLEAVE + remainder)`
      obtain ⟨cp0, cp1⟩ := cpPos_range d g hB hC1 hI1 hp
      have hval : cpPos d g p = (r1.get l).toInt * (d.bsz g : Int) +
          ((r5.get l).toInt * (d.CP_INTERLEAVE : Int) + (r6.get l).toInt) := by
        simp only [cpPos, hnot, ite_false, v1, v5, v6, v2]
        ring
      have r1nn : 0 ≤ (r1.get l).toInt := by rw [v1]; exact Int.ediv_nonneg hp hV0.le
      have r5nn : 0 ≤ (r5.get l).toInt := by rw [v5]; exact Int.ediv_nonneg r2nn hIC0.le
      have r6nn : 0 ≤ (r6.get l).toInt := by rw [v6]; exact Int.emod_nonneg _ hI0.ne'
      have hB0 : (0 : Int) ≤ d.bsz g := by positivity
      have hI0' : (0 : Int) ≤ d.CP_INTERLEAVE := hI0.le
      have a1 : 0 ≤ (r1.get l).toInt * (d.bsz g : Int) := mul_nonneg r1nn hB0
      have a2 : 0 ≤ (r5.get l).toInt * (d.CP_INTERLEAVE : Int) := mul_nonneg r5nn hI0'
      have m1 : ((r5 * bc (IntV.ofNat d.CP_INTERLEAVE : I64)).get l).toInt =
          (r5.get l).toInt * (d.CP_INTERLEAVE : Int) := by
        rw [Tensor.get_mul, Tensor.get_full]
        exact IntV.toInt_mul rfl hI (IntV.I64.sfits (by omega) (by omega))
      have m2 : ((r5 * bc (IntV.ofNat d.CP_INTERLEAVE : I64) + r6).get l).toInt =
          (r5.get l).toInt * (d.CP_INTERLEAVE : Int) + (r6.get l).toInt := by
        rw [Tensor.get_add]
        exact IntV.toInt_add m1 rfl (IntV.I64.sfits (by omega) (by omega))
      have m3 : ((r1 * bc (IntV.cast kv : I64)).get l).toInt =
          (r1.get l).toInt * (d.bsz g : Int) := by
        rw [Tensor.get_mul, Tensor.get_full]
        exact IntV.toInt_mul rfl hbs (IntV.I64.sfits (by omega) (by omega))
      simp only [tl.where, Tensor.get_where, Tensor.get_full, ite_true]
      rw [Tensor.get_add]
      rw [IntV.toInt_add m3 m2 (IntV.I64.sfits (by omega) (by omega)), hval]

end Claims.SlotMappings
