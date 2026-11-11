import Proof.Dkdv
import Proof.Dq

/-!
# `attn_bwd_f16`: the stages of the two loops

With `CAUSAL`, `dk` and `dv` are accumulated in two stages (a masked one over the block, then the
rest) and `dq` in two stages (masked, then the rest); the tables below say which rows each stage
has processed, in how many steps, and where the next stage starts.
-/

namespace Claims.AttnBwd

open TVA

/-- The query rows and steps of the first stage of `dk`, `dv` (lines 415–429): with `CAUSAL`, the
`8` masked steps over the block of `n0, …, n0 + 127`, ending at `n0 + 128`; else nothing. -/
def stage1 (d : Inst) (n0 : Nat) : Finset Nat × Nat × Nat :=
  if d.CAUSAL = true then (Finset.Ico n0 (n0 + 128), 8, n0 + 128) else (∅, 0, 0)

/-- What the first stage of `dk`, `dv` leaves: memory unchanged, the start of the second stage, and
the accumulators. -/
def Post1 (d : Inst) (init : Mem) (b n0 : Nat)
    (a : Tensor [128, d.HEAD_DIM] F32 × Tensor [128, d.HEAD_DIM] F32 × I32) (m : Mem) : Prop :=
  m = init ∧ a.2.2.toInt = ((stage1 d n0).2.2 : Nat) ∧
    DkdvInv d init b n0 (stage1 d n0).1 (stage1 d n0).2.1 a.1 a.2.1


/-- Zero accumulators satisfy `DkdvInv` with no steps. -/
theorem DkdvInv.zero (d : Inst) (init : Mem) (b n0 : Nat) :
    DkdvInv d init b n0 ∅ 0 (tl.zeros [128, d.HEAD_DIM] .f32) (tl.zeros [128, d.HEAD_DIM] .f32) :=
  fun _ _ _ => ⟨AccInv.zero _ _ _ (fun _ _ => DType.val?_zero_f32),
    AccInv.zero _ _ _ (fun _ _ => DType.val?_zero_f32)⟩

/-- A zero accumulator satisfies `DqInv` with no steps. -/
theorem DqInv.zero (d : Inst) (init : Mem) (b m0 : Nat) :
    DqInv d init b m0 ∅ 0 (tl.zeros [128, d.HEAD_DIM] .f32) :=
  fun _ _ _ => AccInv.zero _ _ _ (fun _ _ => DType.val?_zero_f32)

/-- The first key row `dk`, `dv` need: with `CAUSAL`, the block's first row `n0`; else `0`. -/
def lo (d : Inst) (n0 : Nat) : Nat := if d.CAUSAL = true then n0 else 0

/-- The rows of the first stage are below `N_CTX` and below the start of the second. -/
theorem stage1_lt {d : Inst} {n0 : Nat} (hn0 : n0 + 128 ≤ d.N_CTX) :
    ∀ s ∈ (stage1 d n0).1, s < d.N_CTX ∧ s < (stage1 d n0).2.2 := by
  unfold stage1
  split_ifs
  · intro s hs
    have := Finset.mem_Ico.mp hs
    exact ⟨by omega, by simp only; omega⟩
  · intro s hs; simp at hs

/-- The second stage starts at most at `N_CTX`. -/
theorem stage1_le {d : Inst} {n0 : Nat} (hn0 : n0 + 128 ≤ d.N_CTX) :
    (stage1 d n0).2.2 ≤ d.N_CTX := by
  unfold stage1; split_ifs <;> simp only <;> omega

/-- The second stage runs without a mask: not `CAUSAL`, or it starts after the block. -/
theorem stage1_al (d : Inst) (n0 : Nat) :
    d.CAUSAL = false ∨ n0 + 128 ≤ (stage1 d n0).2.2 := by
  unfold stage1
  by_cases h : d.CAUSAL = true
  · simp [h]
  · left; simpa using h

/-- The first stage takes at most `8` steps. -/
theorem stage1_t (d : Inst) (n0 : Nat) : (stage1 d n0).2.1 ≤ 8 := by
  unfold stage1; split_ifs <;> simp

/-- **The two stages cover the rows `lo … N_CTX`, in `stepsKV` steps.** -/
theorem stage_final {d : Inst} (hd : d.ok) {n0 : Nat} (hn0m : n0 % 128 = 0)
    (hn0 : n0 + 128 ≤ d.N_CTX) :
    (stage1 d n0).1 ∪ Finset.Ico (stage1 d n0).2.2
        ((stage1 d n0).2.2 + (d.N_CTX - (stage1 d n0).2.2) / 32 * 32) =
      Finset.Ico (lo d n0) d.N_CTX ∧
    (stage1 d n0).2.1 + (d.N_CTX - (stage1 d n0).2.2) / 32 = stepsKV d n0 := by
  have hn := hd.n_mod
  unfold stage1 lo stepsKV
  obtain h | h : d.CAUSAL = true ∨ d.CAUSAL = false := by cases d.CAUSAL <;> simp
  · simp only [h, ite_true]
    have e1 : (d.N_CTX - (n0 + 128)) / 32 * 32 = d.N_CTX - (n0 + 128) := by omega
    have e2 : n0 / 128 * 128 = n0 := by omega
    refine ⟨?_, ?_⟩
    · rw [e1, Finset.Ico_union_Ico_eq_Ico (by omega) (by omega)]
      have e3 : n0 + 128 + (d.N_CTX - (n0 + 128)) = d.N_CTX := by omega
      rw [e3]
    · have : (d.N_CTX - 128 * (n0 / 128) - 128) = d.N_CTX - (n0 + 128) := by omega
      rw [this]
  · simp only [h, Bool.false_eq_true, ite_false]
    have e1 : d.N_CTX / 32 * 32 = d.N_CTX := by omega
    refine ⟨?_, by simp⟩
    show (∅ : Finset Nat) ∪ Finset.Ico 0 (0 + (d.N_CTX - 0) / 32 * 32) = Finset.Ico 0 d.N_CTX
    rw [Finset.empty_union, Nat.zero_add, Nat.sub_zero, e1]

/-- What the first stage of `dq` (lines 468–487) has processed: with `CAUSAL`, the `8` masked steps
over the key rows of the block of query rows `q0, …, q0 + 127`, then the second stage runs from key
row `0` for `q0 / 32` steps; else nothing, and the second stage runs from `0` for `N_CTX / 32`
steps. The components: the key rows, the steps, the start of the second stage, its number of
steps. -/
def stage2 (d : Inst) (q0 : Nat) : Finset Nat × Nat × Nat × Nat :=
  if d.CAUSAL = true then (Finset.Ico q0 (q0 + 128), 8, 0, q0 / 32)
  else (∅, 0, 0, d.N_CTX / 32)

/-- What the first stage of `dq` leaves: memory unchanged, the start and the number of steps of the
second stage, and the accumulator. -/
def Post2 (d : Inst) (init m0 : Mem) (b q0 : Nat)
    (a : Tensor [128, d.HEAD_DIM] F32 × I32 × I32) (m : Mem) : Prop :=
  m = m0 ∧ a.2.1.toInt = ((stage2 d q0).2.2.1 : Nat) ∧ a.2.2.toInt = ((stage2 d q0).2.2.2 : Nat) ∧
    DqInv d init b q0 (stage2 d q0).1 (stage2 d q0).2.1 a.1

/-- The last key row (exclusive) `dq` needs: with `CAUSAL`, the block's end; else `N_CTX`. -/
def hi (d : Inst) (q0 : Nat) : Nat := if d.CAUSAL = true then q0 + 128 else d.N_CTX

/-- The key rows of the first stage of `dq` are below `N_CTX`. -/
theorem stage2_lt {d : Inst} {q0 : Nat} (hq0 : q0 + 128 ≤ d.N_CTX) :
    ∀ s ∈ (stage2 d q0).1, s < d.N_CTX := by
  unfold stage2
  split_ifs
  · intro s hs
    have := Finset.mem_Ico.mp hs
    omega
  · intro s hs; simp at hs

/-- The second stage of `dq` runs without a mask: not `CAUSAL`, or it ends before the block. -/
theorem stage2_al (d : Inst) {q0 : Nat} (hq0m : q0 % 128 = 0) :
    d.CAUSAL = false ∨ (stage2 d q0).2.2.1 + (stage2 d q0).2.2.2 * 32 ≤ q0 := by
  unfold stage2
  by_cases h : d.CAUSAL = true
  · right; simp only [h, ite_true]; omega
  · left; simpa using h

/-- The key rows of the first stage of `dq` are past the second stage's. -/
theorem stage2_disj (d : Inst) {q0 : Nat} (hq0m : q0 % 128 = 0) :
    ∀ s ∈ (stage2 d q0).1, (stage2 d q0).2.2.1 + (stage2 d q0).2.2.2 * 32 ≤ s := by
  unfold stage2
  split_ifs
  · intro s hs
    have := Finset.mem_Ico.mp hs
    simp only
    omega
  · intro s hs; simp at hs

/-- The first stage of `dq` takes at most `8` steps. -/
theorem stage2_t (d : Inst) (q0 : Nat) : (stage2 d q0).2.1 ≤ 8 := by
  unfold stage2; split_ifs <;> simp

/-- The second stage of `dq` stays inside `N_CTX`. -/
theorem stage2_steps {d : Inst} (hd : d.ok) (q0 : Nat) (hq0 : q0 + 128 ≤ d.N_CTX) :
    (stage2 d q0).2.2.1 + (stage2 d q0).2.2.2 * 32 ≤ d.N_CTX := by
  have := hd.n_mod
  unfold stage2
  split_ifs
  · simp only; omega
  · simp only; omega

/-- **The two stages of `dq` cover the key rows `0 … hi`, in `stepsQ` steps.** -/
theorem stage2_final {d : Inst} (hd : d.ok) {q0 : Nat} (hq0m : q0 % 128 = 0) :
    (stage2 d q0).1 ∪ Finset.Ico (stage2 d q0).2.2.1
        ((stage2 d q0).2.2.1 + (stage2 d q0).2.2.2 * 32) = Finset.Ico 0 (hi d q0) ∧
    (stage2 d q0).2.1 + (stage2 d q0).2.2.2 = stepsQ d q0 := by
  have hn := hd.n_mod
  unfold stage2 hi stepsQ
  obtain h | h : d.CAUSAL = true ∨ d.CAUSAL = false := by cases d.CAUSAL <;> simp
  · simp only [h, ite_true]
    have e1 : q0 / 32 * 32 = q0 := by omega
    refine ⟨?_, ?_⟩
    · rw [zero_add, e1, Finset.union_comm, Finset.Ico_union_Ico_eq_Ico (by omega) (by omega)]
    · have : 128 * (q0 / 128) / 32 = q0 / 32 := by omega
      rw [this]
  · simp only [h, Bool.false_eq_true, ite_false]
    have e1 : d.N_CTX / 32 * 32 = d.N_CTX := by omega
    refine ⟨?_, by simp⟩
    show (∅ : Finset Nat) ∪ Finset.Ico 0 (0 + d.N_CTX / 32 * 32) = Finset.Ico 0 d.N_CTX
    rw [Finset.empty_union, Nat.zero_add, e1]

end Claims.AttnBwd
