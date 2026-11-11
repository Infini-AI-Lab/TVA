import Proof.Tgt

/-!
# `kv_cache_write_bf16`: shared facts for the `Safe` proofs of the programs

The program's ids, the read footprint's parts, and the positivity of a slot's block size.
-/

namespace Claims.KvCacheWrite

open TVA

/-- The `x` id of a program is its token. -/
theorem pid0_eq (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 0 = (IntV.ofNat (tk d (i : Nat)) : I32) := by
  simp only [tl.program_id, launch_pid0, tk]

/-- The `y` id of a program is its tile. -/
theorem pid1_eq (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 1 = (IntV.ofNat (ti d (i : Nat)) : I32) := by
  simp only [tl.program_id, launch_pid1, ti]

/-- The slot mapping bytes are among the bytes the kernel reads. -/
theorem R_sub_slot (d : Inst) : d.slot_mapping.bytes ⊆ R d := by
  intro x hx; simp only [R, Set.mem_union]; tauto

/-- The key input bytes are among the bytes the kernel reads. -/
theorem R_sub_key (d : Inst) : d.key.bytes ⊆ R d := by
  intro x hx; simp only [R, Set.mem_union]; tauto

/-- The value input bytes are among the bytes the kernel reads. -/
theorem R_sub_value (d : Inst) : d.value.bytes ⊆ R d := by
  intro x hx; simp only [R, Set.mem_union]; tauto

/-- In fp8 mode, the key scale bytes are among the bytes the kernel reads. -/
theorem R_sub_kscale {d : Inst} (hm : d.mode = .fp8) : d.k_scale.bytes ⊆ R d := by
  intro x hx; simp only [R, hm, Set.mem_union]; tauto

/-- In fp8 mode, the value scale bytes are among the bytes the kernel reads. -/
theorem R_sub_vscale {d : Inst} (hm : d.mode = .fp8) : d.v_scale.bytes ⊆ R d := by
  intro x hx; simp only [R, hm, Set.mem_union]; tauto

/-- A slot in the cache's range makes the cache's block size positive. -/
theorem block_size_pos' {d : Inst} {init : Mem} {t : Nat}
    (hsl0 : 0 ≤ slotAt d init t) (hsl : slotAt d init t < (numSlots d : Int)) :
    0 < d.block_size := by
  rcases Nat.eq_zero_or_pos d.block_size with h0 | h0
  · exfalso
    have hz : numSlots d = 0 := by simp only [numSlots, h0, Nat.mul_zero]
    omega
  · exact h0

/-- The `int64` value of zero. -/
theorem I64_toInt_zero : (IntV.ofInt 0 : I64).toInt = 0 :=
  I64_toInt_ofInt (by norm_num [IntV.SFits])

/-- A negative-slot test that succeeds means the slot is negative. -/
theorem lt_zero_of_true {a : I64} (h : IntV.lt a (IntV.ofInt 0 : I64) = true) : a.toInt < 0 := by
  rw [IntV.lt_eq, I64_toInt_zero, decide_eq_true_eq] at h
  exact h

/-- A negative-slot test that fails means the slot is non-negative. -/
theorem le_zero_of_not_true {a : I64} (h : ¬ IntV.lt a (IntV.ofInt 0 : I64) = true) :
    0 ≤ a.toInt := by
  rw [IntV.lt_eq, I64_toInt_zero, decide_eq_true_eq] at h
  omega

end Claims.KvCacheWrite
