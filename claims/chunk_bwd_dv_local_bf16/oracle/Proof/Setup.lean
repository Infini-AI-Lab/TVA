import Claim
import TVA.Proof
import Proof.Tail
import Proof.Post
import Proof.Vals

/-!
# What the tail knows about its program

The tail's variables are the base pointers of the head's rows, the token offsets `o_t` of the
chunk's lanes and the sequence length `T`. `Setup` records their values: `q`, `k` point at
`qOff`, `do_`, `dv` at `oOff`, the gate pointer at `gOff`, lane `l` of `o_t` is the offset
`chkOf·BT + l`, and `T` is the sequence length. The lemmas give lane `l`'s token (`pos`), the
mask `m_t` (`en`), the element each pointer lane names, and the room all index arithmetic has.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-! ## The offsets of the head's rows -/

/-- Element offset of program `(r, bh)`'s first token and query/key head in `q` and `k`. -/
def qOff (d : Inst) (r bh : Nat) : Nat := (baseOf d r bh * d.H + hk d (hvOf d bh)) * d.K

/-- Element offset of the first token and value head in `do` and `dv`. -/
def oOff (d : Inst) (r bh : Nat) : Nat := (baseOf d r bh * d.HV + hvOf d bh) * d.V

/-- Element offset of the first token and value head in `g`. -/
def gOff (d : Inst) (r bh : Nat) : Nat := baseOf d r bh * d.HV + hvOf d bh

/-- **What the tail knows**: its pointers, the lanes of `o_t` and the sequence length. -/
structure Setup (d : Inst) (r bh : Nat) (q k do_ dv : Ptr .bf16) (gb : Ptr .f32)
    (o_t : Tensor [d.BT] I64) (T : I64) : Prop where
  q_eq : q = d.q.ptr (qOff d r bh : Int)
  k_eq : k = d.k.ptr (qOff d r bh : Int)
  do_eq : do_ = d.dO.ptr (oOff d r bh : Int)
  dv_eq : dv = d.dv.ptr (oOff d r bh : Int)
  g_eq : gb = d.g.ptr (gOff d r bh : Int)
  ot : ∀ l : Fin d.BT, (o_t.get1 l).toInt = ((chkOf d r bh * d.BT + l : Nat) : Int)
  T_eq : T.toInt = (lenOf d r bh : Int)

/-- **The mask `m_t`**: lane `l` of `o_t < T` is `en`. -/
theorem mt_get {d : Inst} {r bh : Nat} {q k do_ dv : Ptr .bf16} {gb : Ptr .f32}
    {o_t : Tensor [d.BT] I64} {T : I64} (S : Setup d r bh q k do_ dv gb o_t T) (l : Fin d.BT) :
    (o_t.ltS T).get1 l = en d r bh l := by
  show (o_t.get1 l).lt T = _
  rw [IntV.lt_eq, S.ot l, S.T_eq]
  simp only [en, Nat.cast_lt]

/-! ## The geometry of a program of the grid -/

/-- The program's chunk starts inside its sequence: lane `0` is enabled. -/
theorem chunk_start_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r : Nat}
    (hr : r < nChunks d) (bh : Nat) : chkOf d r bh * d.BT < lenOf d r bh := by
  have hBT := bt_pos hd
  cases hvv : d.varlen
  · rw [chkOf_false hvv, lenOf, hi_false hvv, lo_false hvv, nOf_false hvv]
    have hrw : nChunks d = (d.T + d.BT - 1) / d.BT := by simp [nChunks, hvv]
    rw [hrw] at hr
    have h1 : (r + 1) * d.BT ≤ d.T + d.BT - 1 := (Nat.le_div_iff_mul_le hBT).mp hr
    have h2 : (r + 1) * d.BT = r * d.BT + d.BT := by ring
    omega
  · obtain ⟨-, -, -, hseq, -⟩ := hT hvv
    obtain ⟨h1, h2, h3, h4⟩ := hseq r (row_lt_NT hvv hr)
    rw [chkOf_true hvv, lenOf, hi_true hvv, lo_true hvv, nOf_true hvv]
    have hmono := cuAt_step (hT hvv) (n := (seqAt d r).toNat) (by omega)
    have h5 : cuAt d (seqAt d r).toNat ≤ cuAt d ((seqAt d r).toNat + 1) := hmono
    have hn0 := cuAt_nonneg (hT hvv) (n := (seqAt d r).toNat) (by omega)
    obtain ⟨c, hc⟩ : ∃ c : Nat, chkAt d r = (c : Int) := ⟨(chkAt d r).toNat, (Int.toNat_of_nonneg h3).symm⟩
    rw [hc] at h4 ⊢
    have h6 : ((c * d.BT : Nat) : Int) < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat := by
      push_cast; exact h4
    simp only [Int.toNat_natCast]
    omega

/-- The program's value head is below `HV`, and its query/key head is below `H`. -/
theorem hk_lt {d : Inst} (hd : d.ok) (bh : Nat) : hk d (hvOf d bh) < d.H := by
  have hHV := hd.HV_pos
  obtain ⟨m, hm⟩ := hd.hv_dvd
  have hm0 : 0 < m := by
    rcases Nat.eq_zero_or_pos m with h | h
    · rw [h] at hm; omega
    · exact h
  have hdiv : d.HV / d.H = m := by
    rw [hm]; exact Nat.mul_div_cancel_left m hd.H_pos
  unfold hk
  rw [hdiv, Nat.div_lt_iff_lt_mul hm0]
  have := hvOf_lt hd bh
  rw [hm] at this
  exact this

/-- A matrix of `X + 1` rows of `H` heads of `K` columns lies below `B·T·H·K`. -/
theorem mat_lt {X H K h c BTt : Nat} (hX : X < BTt) (hh : h < H) (hc : c < K) :
    (X * H + h) * K + c < BTt * H * K := by
  have h1 : (X * H + h) * K + c < (X * H + h + 1) * K := by nlinarith
  have h2 : (X * H + h + 1) * K ≤ (X + 1) * H * K := by
    apply Nat.mul_le_mul_right
    nlinarith
  have h3 : (X + 1) * H * K ≤ BTt * H * K := by
    apply Nat.mul_le_mul_right
    apply Nat.mul_le_mul_right
    omega
  omega

section Geom
variable {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh : Nat}
  (hr : r < nChunks d) (hbh : bh < nRows d)
include hd hT hr hbh

/-- The last token of the program's chunk is below the flat token count. -/
theorem last_tok_lt : baseOf d r bh + lenOf d r bh - 1 < d.B * d.T := by
  have h1 := pos_lt_hi hT (nOf_lt hd hT hr hbh) (l := 0) (by
    rw [en_iff, Nat.add_zero]; exact chunk_start_lt hd hT hr bh)
  have h2 := hi_le_BT hd hT (nOf_lt hd hT hr hbh)
  have h3 := baseOf_add_len hT r bh (nOf_lt hd hT hr hbh)
  have h4 := chunk_start_lt hd hT hr bh
  omega

end Geom

/-- A valid buffer's length times its element size is at most `2^64`. -/
theorem buf_len_le {d : DType} {b : Buf d} {m : Mem} (h : b.Valid m) : b.len * d.bytes ≤ 2 ^ 64 := by
  have := h.1; have := h.2.1; omega

section Geom2
variable {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh : Nat}
  (hr : r < nChunks d) (hbh : bh < nRows d)
include hd hT hr hbh

/-- The last element of the program's `q`/`k` tile is inside the tensor. -/
theorem qk_tile_lt : qOff d r bh + (lenOf d r bh - 1) * (d.H * d.K) + (d.K - 1) * 1 <
    d.B * d.T * d.H * d.K := by
  have hl := chunk_start_lt hd hT hr bh
  have hX := last_tok_lt hd hT hr hbh
  have h1 : qOff d r bh + (lenOf d r bh - 1) * (d.H * d.K) + (d.K - 1) * 1 =
      ((baseOf d r bh + lenOf d r bh - 1) * d.H + hk d (hvOf d bh)) * d.K + (d.K - 1) := by
    unfold qOff
    have : baseOf d r bh + lenOf d r bh - 1 = baseOf d r bh + (lenOf d r bh - 1) := by omega
    rw [this]
    ring
  rw [h1]
  exact mat_lt hX (hk_lt hd bh) (by have := hd.K_pos; omega)

/-- The last element of the program's `do`/`dv` tile is inside the tensor. -/
theorem o_tile_lt : oOff d r bh + (lenOf d r bh - 1) * (d.HV * d.V) + (d.V - 1) * 1 <
    d.B * d.T * d.HV * d.V := by
  have hl := chunk_start_lt hd hT hr bh
  have hX := last_tok_lt hd hT hr hbh
  have h1 : oOff d r bh + (lenOf d r bh - 1) * (d.HV * d.V) + (d.V - 1) * 1 =
      ((baseOf d r bh + lenOf d r bh - 1) * d.HV + hvOf d bh) * d.V + (d.V - 1) := by
    unfold oOff
    have : baseOf d r bh + lenOf d r bh - 1 = baseOf d r bh + (lenOf d r bh - 1) := by omega
    rw [this]
    ring
  rw [h1]
  exact mat_lt hX (hvOf_lt hd bh) (by have := hd.V_pos; omega)

/-- The last element of the program's `g` column is inside the tensor. -/
theorem g_tile_lt : gOff d r bh + (lenOf d r bh - 1) * d.HV < d.B * d.T * d.HV := by
  have hl := chunk_start_lt hd hT hr bh
  have hX := last_tok_lt hd hT hr hbh
  have h1 : gOff d r bh + (lenOf d r bh - 1) * d.HV =
      (baseOf d r bh + lenOf d r bh - 1) * d.HV + hvOf d bh := by
    unfold gOff
    have : baseOf d r bh + lenOf d r bh - 1 = baseOf d r bh + (lenOf d r bh - 1) := by omega
    rw [this]
    ring
  rw [h1]
  have := mat_lt hX (hvOf_lt hd bh) (by norm_num : 0 < 1) (BTt := d.B * d.T) (K := 1) (c := 0)
  simpa using this

end Geom2

end Claims.ChunkBwdDvLocal
