import Proof.Defs
import Proof.I32
import Proof.Pid

/-!
# `lora_shrink_bf16`: small facts used by the walk of one program
-/

namespace Claims.LoraShrink

open TVA

/-- The pointer to element `0` plus an `int32` scalar `n` is the pointer to element `n`. -/
theorem buf_ptr_add_ofNat {d : DType} (b : Buf d) (n : Nat) (hn : n < 2 ^ 31) :
    b.ptr 0 + (IntV.ofNat n : I32) = b.ptr (n : Int) := by
  show (b.ptr 0).offset (IntV.ofNat n : I32).toInt = _
  rw [IntV.I32.toInt_ofNat hn, Buf.ptr_offset]
  simp

/-- The pointer to element `j` plus an `int32` scalar `n` is the pointer to element `j + n`. -/
theorem buf_ptr_add_ofNat' {d : DType} (b : Buf d) (j n : Nat) (hn : n < 2 ^ 31) :
    b.ptr (j : Int) + (IntV.ofNat n : I32) = b.ptr ((j + n : Nat) : Int) := by
  show (b.ptr (j : Int)).offset (IntV.ofNat n : I32).toInt = _
  rw [IntV.I32.toInt_ofNat hn, Buf.ptr_offset]
  push_cast
  rfl

/-- A program that is not active leaves the empty log. -/
theorem Q_nil {d : Inst} {init : Mem} {i : Nat} {m : Mem} (h : ¬ Active d i) :
    Q d init i [] m := by
  refine ⟨by simp, fun _ _ _ => 0, ?_, fun _ _ _ _ => rfl, ?_⟩
  · intro s t j _ _ _ va vb _
    exact Paired.nil
  · intro s t j htgt
    exact absurd htgt.1 h

/-- The bytes of the elements of `inputs` are read. -/
theorem span_inp_sub {d : Inst} {n : Nat} (h : n < d.inp.len) :
    Mem.span .bf16 (d.inp.ptr n).addr ⊆ Rall d := by
  intro x hx
  have := d.inp.span_sub h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- The bytes of the elements of slice `s`'s weights are read. -/
theorem span_lora_sub {d : Inst} {s n : Nat} (hs : s < d.S) (h : n < (d.lora s).len) :
    Mem.span .bf16 ((d.lora s).ptr n).addr ⊆ Rall d := by
  intro x hx
  have := (d.lora s).span_sub h hx
  unfold Rall
  simp only [Set.mem_union, Set.mem_iUnion]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr ⟨s, hs, this⟩)))))

/-- The bytes of the address table's entries are read. -/
theorem span_tab_sub {d : Inst} {n : Nat} (h1 : 1 < d.S) (h : n < d.tab.len) :
    Mem.span .u64 (d.tab.ptr n).addr ⊆ Rall d := by
  intro x hx
  have := d.tab.span_sub h hx
  unfold Rall
  simp only [Set.mem_union, Set.mem_ofPred_eq]
  tauto

/-- The bytes of the sorted token indices are read. -/
theorem span_tid_sub {d : Inst} {n : Nat} (h : n < d.tIdxBuf.len) :
    Mem.span .i32 (d.tIdxBuf.ptr n).addr ⊆ Rall d := by
  intro x hx
  have := d.tIdxBuf.span_sub h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- The bytes of the group counts are read. -/
theorem span_cnt_sub {d : Inst} {n : Nat} (h : n < d.cntBuf.len) :
    Mem.span .i32 (d.cntBuf.ptr n).addr ⊆ Rall d := by
  intro x hx
  have := d.cntBuf.span_sub h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- The bytes of the group start locations are read. -/
theorem span_start_sub {d : Inst} {n : Nat} (h : n < d.startBuf.len) :
    Mem.span .i32 (d.startBuf.ptr n).addr ⊆ Rall d := by
  intro x hx
  have := d.startBuf.span_sub h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- The bytes of the group ids are read. -/
theorem span_id_sub {d : Inst} {n : Nat} (h : n < d.idBuf.len) :
    Mem.span .i32 (d.idBuf.ptr n).addr ⊆ Rall d := by
  intro x hx
  have := d.idBuf.span_sub h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- A group id fits in `int32`. -/
theorem idAt_sfits {d : Inst} (hd : d.ok) (g : Nat) : IntV.SFits 32 (idAt d g) := by
  unfold idAt
  split_ifs with hg
  · obtain ⟨t, ht, hta⟩ := hd.grp_nonempty g hg
    have h1 := hd.ad_range t ht
    have hw := hd.lora_wrap t ht
    have hRK : 1 ≤ d.R * d.K := Nat.mul_pos hd.R_pos hd.K_pos
    rw [hta] at h1 hw
    refine IntV.I32.sfits (by omega) ?_
    by_cases hneg : d.groupId g < 0
    · omega
    · have h2 := hw (by omega)
      have : (1 : Int) ≤ ((d.R * d.K : Nat) : Int) := by exact_mod_cast hRK
      nlinarith
  · exact IntV.I32.sfits (by norm_num) (by norm_num)

/-- A group's token count is at most the number of tokens. -/
theorem cntAt_le {d : Inst} (hd : d.ok) (g : Nat) : cntAt d g ≤ d.M := by
  unfold cntAt
  split_ifs with hg
  · rw [hd.cnt_def g hg]
    calc _ ≤ (Finset.range d.M).card := Finset.card_filter_le _ _
      _ = d.M := Finset.card_range _
  · exact Nat.zero_le _

/-- Elements of `inputs` below its length are read. -/
theorem inp_elems_sub {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.inp.len) :
    d.inp.elems S ⊆ Rall d := by
  intro x hx
  have := d.inp.elems_sub_bytes h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- Elements of slice `s`'s weights below its length are read. -/
theorem lora_elems_sub {d : Inst} {s : Nat} {S : Set Nat} (hs : s < d.S)
    (h : ∀ j ∈ S, j < (d.lora s).len) : (d.lora s).elems S ⊆ Rall d := by
  intro x hx
  have := (d.lora s).elems_sub_bytes h hx
  unfold Rall
  simp only [Set.mem_union, Set.mem_iUnion]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr ⟨s, hs, this⟩)))))

/-- Elements of the sorted token indices below its length are read. -/
theorem tid_elems_sub {d : Inst} {S : Set Nat} (h : ∀ j ∈ S, j < d.tIdxBuf.len) :
    d.tIdxBuf.elems S ⊆ Rall d := by
  intro x hx
  have := d.tIdxBuf.elems_sub_bytes h hx
  unfold Rall
  simp only [Set.mem_union]
  tauto

/-- A pointer built from an `int64`-range address is that address (`x.to(tl.pointer_type(...))`). -/
theorem intToPtr_ofInt {dt : DType} {x : Int} (h0 : 0 ≤ x) (h1 : x < 2 ^ 64) :
    tl.intToPtr dt (IntV.ofInt x : U64) = ⟨x⟩ := by
  unfold tl.intToPtr
  congr 1
  show (((BitVec.ofInt 64 x).toNat : Nat) : Int) = x
  rw [BitVec.toNat_ofInt]
  have : x % ((2 ^ 64 : Nat) : Int) = x := Int.emod_eq_of_lt h0 (by exact_mod_cast h1)
  push_cast at this ⊢
  rw [this]
  exact Int.toNat_of_nonneg h0

/-- The columns of the program's rank tile are inside the rank. -/
theorem tileN_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < (launch d).size) {n : Nat} (hn : n < 16) :
    16 * tileN d i + n < d.R := by
  have hx := (decode_lt d hd hi).1
  have h1 := (tile_lt d hx).2
  have h2 : tileN d i = pidN d (ax0 d i) := rfl
  have h3 : cn d = (d.R + 15) / 16 := rfl
  have h4 := hd.R_blk
  omega

/-! ## An active program -/

section Active

variable {d : Inst} {i : Nat}

/-- An active program's group is one of the distinct adapters. -/
theorem Active.g_lt (h : Active d i) : grpOf d i < d.n := by
  by_contra hg
  exact h.1 (by unfold idAt; simp [hg])

/-- An active program's group id is its group's adapter. -/
theorem Active.id_eq (h : Active d i) : idAt d (grpOf d i) = d.groupId (grpOf d i) := by
  unfold idAt; simp [h.g_lt]

/-- An active program's adapter is not `-1`, hence a real adapter. -/
theorem Active.id_nonneg (hd : d.ok) (h : Active d i) : 0 ≤ idAt d (grpOf d i) := by
  rw [h.id_eq]
  obtain ⟨t, ht, hta⟩ := hd.grp_nonempty _ h.g_lt
  have h1 := (hd.ad_range t ht).1
  have h2 := h.1
  rw [h.id_eq] at h2
  omega

/-- An active program's count, start and rows. -/
theorem Active.cnt_eq (h : Active d i) : cntAt d (grpOf d i) = d.groupCnt (grpOf d i) := by
  unfold cntAt; simp [h.g_lt]

/-- The start of an active program's group. -/
theorem Active.start_eq (h : Active d i) : startAt d (grpOf d i) = d.groupStart (grpOf d i) := by
  unfold startAt; simp [h.g_lt.le]

/-- The program has at least one row. -/
theorem Active.mlen_pos (h : Active d i) : 0 < mlen d i := by
  have := h.2
  unfold mlen
  omega

/-- The program has at most 32 rows. -/
theorem mlen_le (d : Inst) (i : Nat) : mlen d i ≤ 32 := Nat.min_le_left _ _

/-- The rows of the program are inside its group. -/
theorem mlen_le_cnt (d : Inst) (i : Nat) : tileM d i * 32 + mlen d i ≤ cntAt d (grpOf d i) ∨
    cntAt d (grpOf d i) ≤ tileM d i * 32 := by
  unfold mlen
  omega

/-- The position of row `m < mlen` is in the group's range and the token is valid, with the
group's adapter. -/
theorem Active.row (hd : d.ok) (h : Active d i) {m : Nat} (hm : m < mlen d i) :
    tokOf d i m < d.M ∧ d.adapter (tokOf d i m) = idAt d (grpOf d i) ∧
      startAt d (grpOf d i) + tileM d i * 32 + m < d.M := by
  have hlen : m < cntAt d (grpOf d i) - tileM d i * 32 := lt_of_lt_of_le hm (Nat.min_le_right _ _)
  have hg := h.g_lt
  have hlo : d.groupStart (grpOf d i) ≤ startAt d (grpOf d i) + tileM d i * 32 + m := by
    rw [h.start_eq]; omega
  have hhi : startAt d (grpOf d i) + tileM d i * 32 + m < d.groupStart (grpOf d i) + d.groupCnt (grpOf d i) := by
    rw [h.start_eq, ← h.cnt_eq]; omega
  obtain ⟨h1, h2, h3⟩ := hd.tid_range _ hg _ hlo hhi
  exact ⟨h2, by rw [h.id_eq]; unfold tokOf; exact h3, h1⟩

/-- Distinct rows of the program are distinct tokens. -/
theorem Active.tok_inj (hd : d.ok) (h : Active d i) {m m' : Nat} (hm : m < mlen d i)
    (hm' : m' < mlen d i) (he : tokOf d i m = tokOf d i m') : m = m' := by
  have h1 := (h.row hd hm).2.2
  have h2 := (h.row hd hm').2.2
  have := hd.tid_inj _ _ h1 h2 he
  omega

/-- The adapter of an active program is a valid weight index, and its offset fits `int32`. -/
theorem Active.lora_idx (hd : d.ok) (h : Active d i) :
    (idAt d (grpOf d i)).toNat < d.maxL ∧ (idAt d (grpOf d i)).toNat * (d.R * d.K) < 2 ^ 31 := by
  obtain ⟨t, ht, hta⟩ := hd.grp_nonempty _ h.g_lt
  have h1 := (hd.ad_range t ht).2
  have h2 := hd.lora_wrap t ht
  have h0 := h.id_nonneg hd
  rw [hta, ← h.id_eq] at h1 h2
  have e : ((idAt d (grpOf d i)).toNat : Int) = idAt d (grpOf d i) := Int.toNat_of_nonneg h0
  constructor
  · omega
  · have := h2 (by omega)
    rw [← e] at this
    exact_mod_cast this

/-- The start of an active program's group is a valid token position. -/
theorem Active.start_lt (hd : d.ok) (h : Active d i) : startAt d (grpOf d i) < d.M := by
  have := (h.row hd (m := 0) h.mlen_pos).2.2
  omega

end Active

end Claims.LoraShrink
