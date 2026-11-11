import Proof.Defs
import Proof.Lanes

/-!
# Where the kernel's pointers land

The token pointer of a row, the weight lane of a tile element, and the output lane of a tile
element: each is the buffer element the claim's `Post` talks about.
-/

namespace Claims.VocabEmbedding

open TVA

open Block

/-- Lane `l` of the block's offsets, as a value. -/
theorem lane_val {p B : Nat} (hfit : Fits p B) (l : Fin (B - 0)) :
    ((offs p B).get (l, ())).toInt = ((p * B + l : Nat) : Int) := Block.lane_toInt hfit l

/-- The token pointer of the row `p` (the `int32` row converted to `int64`). -/
theorem input_ptr_lane {d : Inst} {row : I64} {p : Nat} (hrow : row.toInt = (p : Int)) :
    (d.input_.ptr 0 + row) = d.input_.ptr p := by
  simp only [Ptr.add_int, Ptr.offset, Buf.ptr]
  rw [hrow]
  congr 1
  ring

/-- The weight lane `l` of the block of column block `q` reads the element `wOff` names. -/
theorem weight_ptr_lane {d : Inst} (hd : d.ok) {v : Int} {tok : I64} (htok : tok.toInt = v)
    (hTok : TokOk d v) {q : Nat} (hfit : Fits q d.BLOCK_H) (l : Fin (d.BLOCK_H - 0)) :
    ((d.weight.ptr 0 + (localIdT d tok).item * (IntV.ofNat d.weight_stride0 : I64) +
        Block.offs q d.BLOCK_H).get (l, ())) =
      d.weight.ptr (wOff d v (q * d.BLOCK_H + l.val)) := by
  have h1 : ((d.weight.ptr 0 + (localIdT d tok).item * (IntV.ofNat d.weight_stride0 : I64) +
        Block.offs q d.BLOCK_H).get (l, ()))
      = ((d.weight.ptr 0).offset
            ((localIdT d tok).item * (IntV.ofNat d.weight_stride0 : I64)).toInt).offset
          ((Block.offs q d.BLOCK_H).get (l, ())).toInt := rfl
  rw [h1]
  simp only [Ptr.offset, Buf.ptr, lane_val hfit, localId_mul_toInt hd htok hTok, wOff]
  congr 1
  push_cast
  ring

/-- An enabled weight lane names an element of the weight buffer. -/
theorem weight_lane_lt {d : Inst} (hd : d.ok) {v : Int} {tok : I64} (htok : tok.toInt = v)
    (hTok : TokOk d v) (hvalid : (validT d tok).item = true) {q : Nat}
    (l : Fin (d.BLOCK_H - 0))
    (htile : Tile q d.BLOCK_H d.hidden_dim (q * d.BLOCK_H + l.val)) :
    0 ≤ wOff d v (q * d.BLOCK_H + l.val) ∧
      (wOff d v (q * d.BLOCK_H + l.val)).toNat < d.weight.len := by
  have hv : (validT d tok).item =
      decide (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index ∨
        d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) := validT_item hd htok
  have htrue : decide (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index ∨
        d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) = true := by
    rw [← hv]; exact hvalid
  have hcol : q * d.BLOCK_H + l.val < d.hidden_dim := htile.2.2
  have hcolI : ((q * d.BLOCK_H + l.val : Nat) : Int) < (d.hidden_dim : Int) := by
    exact_mod_cast hcol
  have hstride : (0 : Int) ≤ (d.weight_stride0 : Int) := by positivity
  have key : ∀ (s : Int), 0 ≤ s →
      s * (d.weight_stride0 : Int) + (d.hidden_dim : Int) ≤ (d.weight.len : Int) →
      0 ≤ s * (d.weight_stride0 : Int) + ((q * d.BLOCK_H + l.val : Nat) : Int) ∧
        s * (d.weight_stride0 : Int) + ((q * d.BLOCK_H + l.val : Nat) : Int) <
          (d.weight.len : Int) := by
    intro s hs hup
    have hm : 0 ≤ s * (d.weight_stride0 : Int) := Int.mul_nonneg hs hstride
    constructor <;> linarith
  rcases of_decide_eq_true htrue with h | h
  · obtain ⟨hnon, hup⟩ := key (v - d.org_vocab_start_index)
      (by have := hd.org_start.1; omega) (hTok.1 h).1
    exact ⟨by rw [wOff, laneSel_of_org h]; exact hnon,
      (Int.toNat_lt (by rw [wOff, laneSel_of_org h]; exact hnon)).2
        (by rw [wOff, laneSel_of_org h]; exact hup)⟩
  · by_cases horg : d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index
    · obtain ⟨hnon, hup⟩ := key (v - d.org_vocab_start_index)
        (by have := hd.org_start.1; omega) (hTok.1 horg).1
      exact ⟨by rw [wOff, laneSel_of_org horg]; exact hnon,
        (Int.toNat_lt (by rw [wOff, laneSel_of_org horg]; exact hnon)).2
          (by rw [wOff, laneSel_of_org horg]; exact hup)⟩
    · obtain ⟨hadd0, hup, -⟩ := hTok.2 h
      obtain ⟨hnon, hlt⟩ := key (v - addedOffset d) hadd0 hup
      exact ⟨by rw [wOff, laneSel_of_add h horg]; exact hnon,
        (Int.toNat_lt (by rw [wOff, laneSel_of_add h horg]; exact hnon)).2
          (by rw [wOff, laneSel_of_add h horg]; exact hlt)⟩

/-- The output lane `l` of the block of row `p` writes the element the claim talks about. -/
theorem out_ptr_lane {d : Inst} (hd : d.ok) {p : Nat} {row : I64} (hrow : row.toInt = (p : Int))
    (hp : p < d.n_tokens) {q : Nat} (hfit : Fits q d.BLOCK_H) (l : Fin (d.BLOCK_H - 0)) :
    ((d.output.ptr 0 + row * (IntV.ofNat d.hidden_dim : I64) + Block.offs q d.BLOCK_H).get (l, ())) =
      d.output.ptr ((p * d.hidden_dim + (q * d.BLOCK_H + l.val) : Nat) : Int) := by
  have hh : (IntV.ofNat d.hidden_dim : I64).toInt = (d.hidden_dim : Int) := by
    refine IntV.I64.toInt_ofNat ?_
    have := hd.hidden_le
    omega
  have hfit64 : IntV.SFits 64 ((p : Int) * (d.hidden_dim : Int)) := by
    show -(2 : Int) ^ (64 - 1) ≤ (p : Int) * (d.hidden_dim : Int) ∧
      (p : Int) * (d.hidden_dim : Int) < 2 ^ (64 - 1)
    have hpow : (2 : Int) ^ (64 - 1) = (2 : Int) ^ 63 := by norm_num
    rw [hpow]
    have hpI : (p : Int) < 2 ^ 22 := by
      have h1 : (p : Int) < (d.n_tokens : Int) := by exact_mod_cast hp
      have h2 : (d.n_tokens : Int) ≤ 2 ^ 22 := by exact_mod_cast hd.tokens_le
      linarith
    have hdI : (d.hidden_dim : Int) ≤ 2 ^ 17 := by exact_mod_cast hd.hidden_le
    have hp0 : (0 : Int) ≤ (p : Int) := by positivity
    have hd0 : (0 : Int) ≤ (d.hidden_dim : Int) := by positivity
    have hprod0 : (0 : Int) ≤ (p : Int) * (d.hidden_dim : Int) := Int.mul_nonneg hp0 hd0
    have hprod : (p : Int) * (d.hidden_dim : Int) < (2 : Int) ^ 22 * 2 ^ 17 := by nlinarith
    have he : (2 : Int) ^ 22 * 2 ^ 17 = 2 ^ 39 := by norm_num
    have hlt : (2 : Int) ^ 39 < 2 ^ 63 := by norm_num
    constructor
    · have h2 : (-(2 : Int) ^ 63) ≤ 0 := by norm_num
      linarith
    · linarith
  have hmul : (row * (IntV.ofNat d.hidden_dim : I64)).toInt = (p : Int) * (d.hidden_dim : Int) := by
    exact IntV.toInt_mul hrow hh hfit64
  have h1 : ((d.output.ptr 0 + row * (IntV.ofNat d.hidden_dim : I64) +
        Block.offs q d.BLOCK_H).get (l, ()))
      = ((d.output.ptr 0).offset (row * (IntV.ofNat d.hidden_dim : I64)).toInt).offset
          ((Block.offs q d.BLOCK_H).get (l, ())).toInt := rfl
  rw [h1]
  simp only [Ptr.offset, Buf.ptr, lane_val hfit, hmul]
  congr 1
  push_cast
  ring

end Claims.VocabEmbedding
