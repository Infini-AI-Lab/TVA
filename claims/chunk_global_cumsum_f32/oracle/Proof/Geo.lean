import Claim
import TVA.Proof
import Proof.Basic

/-!
# The geometry of a program

Bounds of the sequence layout (`lo ≤ hi ≤ B·T` from `TablesOk`), of the blocks
(`len ≤ nb·BT < len + BT`), and small facts about `+0` and tensors.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

/-! ## Small facts -/

/-- A rank-0 tensor is the scalar of its item. -/
theorem tensor_scalar_item {α : Type} (r : Tensor [] α) : Tensor.scalar r.item = r := by
  cases r with
  | mk g =>
    have h : (fun _ : Idx [] => g ()) = g := by
      funext i
      exact congrArg g (Subsingleton.elim () i).symm
    simp only [Tensor.scalar, Tensor.item, h]

/-- `+0` holds `0`. -/
theorem ofPat_zero_toRat : (FloatV.ofPat 0 : F32).toRat = 0 := by
  apply FloatV.toRat_of_isZero
  rw [FloatV.pat_ofPat (by decide)]
  exact Format.isZero_zero

/-- `+0` is finite. -/
theorem ofPat_zero_isFinite : (FloatV.ofPat 0 : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  decide

/-- The literal `0` Triton writes is `+0`. -/
theorem litF32_zero : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by
  have h : (FloatV.ofPat 0 : F64).isZero = true := by decide
  simp only [FloatV.litF32, h, ite_true]

/-- `tl.zeros([], tl.float32)` is `+0`. -/
theorem zeros_item : ((tl.zeros [] .f32).item : F32) = FloatV.ofPat 0 := rfl

/-- A lane of a rank-1 tensor moved along axis `0`. -/
theorem get_set0 {α : Type} {BT : Nat} (x : Tensor [BT] α) {l : Nat} (hl : l < BT) {j : Nat}
    (hj : j < BT) : x.get (Idx.set (⟨l, hl⟩, ()) 0 j) = x.get (⟨j, hj⟩, ()) := by
  simp [Idx.set, hj]

/-! ## The table -/

/-- `cu_seqlens` is nonnegative through `N`. -/
theorem cuAt_nonneg {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n ≤ d.N) : 0 ≤ cuAt d n := by
  obtain ⟨h0, hmono, -⟩ := hT
  induction n with
  | zero => exact h0
  | succ k ih => exact le_trans (ih (by omega)) (hmono k (by omega))

/-- `cu_seqlens` is nondecreasing up to `N`. -/
theorem cuAt_mono_le {d : Inst} (hT : TablesOk d) {a b : Nat} (hab : a ≤ b) (hb : b ≤ d.N) :
    cuAt d a ≤ cuAt d b := by
  induction b with
  | zero => simp_all
  | succ k ih =>
    rcases Nat.eq_or_lt_of_le hab with h | h
    · subst h; exact le_rfl
    · exact le_trans (ih (by omega) (by omega)) (hT.2.1 k (by omega))

/-- `cu_seqlens[n] ≤ T` through `N`. -/
theorem cuAt_le_T {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n ≤ d.N) : cuAt d n ≤ d.T :=
  le_trans (cuAt_mono_le hT hn le_rfl) hT.2.2

/-! ## Program, sequence, head -/

/-- Program `i`'s sequence is a sequence. -/
theorem seqOf_lt {d : Inst} {i : Nat} (hi : i < gridSize d) : seqOf d i < nSeq d := by
  unfold seqOf
  unfold gridSize at hi
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hi)

/-- Program `i`'s head is a head. -/
theorem headOf_lt {d : Inst} (hd : d.ok) (i : Nat) : headOf d i < d.H :=
  Nat.mod_lt _ hd.H_pos

/-- Program `i` is `seq·H + head`. -/
theorem seq_head {d : Inst} (i : Nat) : seqOf d i * d.H + headOf d i = i := by
  unfold seqOf headOf
  rw [Nat.mul_comm]; exact Nat.div_add_mod i d.H

/-- On a variable-length launch, the sequence's bounds are table entries. -/
theorem lo_true {d : Inst} (h : d.varlen = true) (n : Nat) : lo d n = (cuAt d n).toNat := by
  simp [lo, h]
theorem hi_true {d : Inst} (h : d.varlen = true) (n : Nat) : hi d n = (cuAt d (n + 1)).toNat := by
  simp [hi, h]
theorem lo_false {d : Inst} (h : d.varlen = false) (n : Nat) : lo d n = n * d.T := by
  simp [lo, h]
theorem hi_false {d : Inst} (h : d.varlen = false) (n : Nat) : hi d n = n * d.T + d.T := by
  simp [hi, h]
theorem nSeq_true {d : Inst} (h : d.varlen = true) : nSeq d = d.N := by simp [nSeq, h]
theorem nSeq_false {d : Inst} (h : d.varlen = false) : nSeq d = d.B := by simp [nSeq, h]

/-- `lo n ≤ hi n`. -/
theorem lo_le_hi {d : Inst} (hT : d.varlen = true → TablesOk d) {n : Nat} (hn : n < nSeq d) :
    lo d n ≤ hi d n := by
  cases hvv : d.varlen
  · rw [lo_false hvv, hi_false hvv]; omega
  · rw [lo_true hvv, hi_true hvv]
    rw [nSeq_true hvv] at hn
    exact Int.toNat_le_toNat ((hT hvv).2.1 n hn)

/-- Every sequence ends within the `B·T` positions. -/
theorem hi_le_BT {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {n : Nat}
    (hn : n < nSeq d) : hi d n ≤ d.B * d.T := by
  cases hvv : d.varlen
  · rw [hi_false hvv]
    rw [nSeq_false hvv] at hn
    calc n * d.T + d.T = (n + 1) * d.T := by ring
      _ ≤ d.B * d.T := Nat.mul_le_mul_right _ hn
  · rw [hi_true hvv]
    rw [nSeq_true hvv] at hn
    have h1 := cuAt_le_T (hT hvv) (n := n + 1) (by omega)
    have h2 : (cuAt d (n + 1)).toNat ≤ d.T := by omega
    have hb : 1 ≤ d.B := hd.B_pos
    have h5 : d.T ≤ d.B * d.T := by nlinarith
    omega

/-- A sequence is at most `T` long. -/
theorem len_le_T {d : Inst} (hT : d.varlen = true → TablesOk d) {n : Nat}
    (hn : n < nSeq d) : hi d n - lo d n ≤ d.T := by
  cases hvv : d.varlen
  · rw [hi_false hvv, lo_false hvv]; omega
  · rw [hi_true hvv, lo_true hvv]
    rw [nSeq_true hvv] at hn
    have h1 := cuAt_le_T (hT hvv) (n := n + 1) (by omega)
    have h0 := cuAt_nonneg (hT hvv) (n := n) (by omega)
    omega

/-- `B·T ≤ 2^35`. -/
theorem BT_le {d : Inst} (hd : d.ok) : d.B * d.T ≤ 2 ^ 35 := by
  have h1 := hd.B_le
  have h2 := hd.T_le
  have : d.B * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul h1 h2
  omega

/-! ## The program's sequence -/

variable {d : Inst} {i : Nat}

theorem loOf_le_hiOf (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) :
    loOf d i ≤ hiOf d i := lo_le_hi hT (seqOf_lt hi)

theorem hiOf_le (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) :
    hiOf d i ≤ d.B * d.T := hi_le_BT hd hT (seqOf_lt hi)

theorem lenOf_le (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) :
    lenOf d i ≤ d.T := len_le_T hT (seqOf_lt hi)

theorem lo_add_len (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) :
    loOf d i + lenOf d i = hiOf d i := by
  have := loOf_le_hiOf hT hi
  unfold lenOf; omega

/-- `nb = cdiv(len, BT)`. -/
theorem nbOf_eq (d : Inst) (i : Nat) : nbOf d i = (lenOf d i + d.BT - 1) / d.BT := rfl

theorem len_le_nb (hd : d.ok) : lenOf d i ≤ nbOf d i * d.BT := by
  have hb := bt_pos hd
  have h1 := Nat.div_add_mod (lenOf d i + d.BT - 1) d.BT
  have h2 := Nat.mod_lt (lenOf d i + d.BT - 1) hb
  rw [nbOf_eq, Nat.mul_comm]
  omega

theorem nb_lt_len (hd : d.ok) : nbOf d i * d.BT < lenOf d i + d.BT := by
  have hb := bt_pos hd
  rw [nbOf_eq]
  have := Nat.div_mul_le_self (lenOf d i + d.BT - 1) d.BT
  omega

theorem nb_le_len (hd : d.ok) : nbOf d i ≤ lenOf d i := by
  have hb := bt_pos hd
  have h1 := nb_lt_len (i := i) hd
  rcases Nat.lt_or_ge (nbOf d i) (lenOf d i + 1) with h | h
  · omega
  · have : (lenOf d i + 1) * d.BT ≤ nbOf d i * d.BT := Nat.mul_le_mul_right _ h
    have : (lenOf d i + 1) * d.BT = lenOf d i * d.BT + d.BT := by ring
    have : lenOf d i ≤ lenOf d i * d.BT := Nat.le_mul_of_pos_right _ hb
    omega

/-- The block trip `t < nb` handles is a block. -/
theorem bt_lt (t : Nat) (ht : t < nbOf d i) : bt d i t < nbOf d i := by
  unfold bt; split <;> omega

/-- A block starts inside the sequence. -/
theorem bt_start_lt (hd : d.ok) {t : Nat} (ht : t < nbOf d i) : bt d i t * d.BT < lenOf d i := by
  have h1 := bt_lt t ht
  have h2 := nb_lt_len (i := i) hd
  have : (bt d i t + 1) * d.BT ≤ nbOf d i * d.BT := Nat.mul_le_mul_right _ h1
  have : (bt d i t + 1) * d.BT = bt d i t * d.BT + d.BT := by ring
  omega

end Claims.ChunkGlobalCumsum
