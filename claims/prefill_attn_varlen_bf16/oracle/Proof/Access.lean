import Proof.Setup

section
/-!
# The four tile accesses of the kernel

The loads of `q` (line 100), `k` (line 124), `v` (line 164) and the store of `o` (line 182) are
`[128, BLOCK_DMODEL]` or `[BLOCK_DMODEL, 128]` tiles of a strided `[total, heads, Lk]` tensor.
For sequence start `S`, length `L`, query head `hN`, key/value head `gN`, query block `mN` and
key block start `T`, this file states, for the model's own pointer and mask tensors, the
hypotheses of `safe_load_tileAt` and `safe_store_tileAt` that do not depend on the continuation:
the pointers on the enabled lanes (`hp`), the mask (`MaskAt`), the extent in the buffer (`hlen`),
the footprint in `Rf` or `Wf` and, for the store, that its lanes write distinct elements.
The `int32` arithmetic of the offsets does not wrap on an enabled lane: every enabled lane
is an element of the tensor, whose index is below `2^31` (`Inst.ok`).
-/

namespace Claims.PrefillAttnVarlen

open TVA TVA.Tile2

/-! ## The tensors of the model -/

/-- `offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)` (line 89). -/
def offsM (sm : I32) : Tensor [BLOCK] I32 := sm * (IntV.ofNat BLOCK : I32) + tl.arange 0 BLOCK

/-- `offs_n = tl.arange(0, BLOCK_N)` (line 87). -/
def offsN : Tensor [BLOCK] I32 := tl.arange 0 BLOCK

/-- `offs_d = tl.arange(0, BLOCK_DMODEL)` (line 88). -/
def offsD (d : Inst) : Tensor [d.BD] I32 := tl.arange 0 d.BD

/-- `mask_d = offs_d < Lk` (line 98). -/
def maskD (d : Inst) : Tensor [d.BD] Bool := (offsD d).ltS (IntV.ofNat d.Lk : I32)

/-- The pointers of the `q` load (lines 90–91, 100): `Q + off_q`. -/
def ptrsQ (d : Inst) (h sm st : I32) : Tensor [BLOCK, d.BD] (Ptr .bf16) :=
  d.q.ptr 0 + ((st + (offsM sm).col d.BD) * (IntV.ofNat d.sqt : I32) +
    bc (h * (IntV.ofNat d.sqh : I32)) + (offsD d).row BLOCK)

/-- The mask of the `q` load and of the `o` store (lines 100–104, 182–184). -/
def maskQ (d : Inst) (len sm : I32) : Tensor [BLOCK, d.BD] Bool :=
  (((offsM sm).col d.BD).ltS len).and ((maskD d).row BLOCK)

/-- The pointers of the `k` load (lines 95, 106, 124–126): `k_ptrs + broadcast scalar offset`. -/
def ptrsK (d : Inst) (g st sn : I32) : Tensor [d.BD, BLOCK] (Ptr .bf16) :=
  (d.k.ptr 0 + (offsN.row d.BD * (IntV.ofNat d.skt : I32) +
    bc (g * (IntV.ofNat d.skh : I32)) + (offsD d).col BLOCK)) +
    (bc ((st + sn) * (IntV.ofNat d.skt : I32)) : Tensor [d.BD, BLOCK] I32)

/-- The mask of the `k` load (lines 124–128). -/
def maskK (d : Inst) (len sn : I32) : Tensor [d.BD, BLOCK] Bool :=
  (((sn + offsN).row d.BD).ltS len).and ((maskD d).col BLOCK)

/-- The pointers of the `v` load (lines 96, 107, 164–166). -/
def ptrsV (d : Inst) (g st sn : I32) : Tensor [BLOCK, d.BD] (Ptr .bf16) :=
  (d.v.ptr 0 + (offsN.col d.BD * (IntV.ofNat d.svt : I32) +
    bc (g * (IntV.ofNat d.svh : I32)) + (offsD d).row BLOCK)) +
    (bc ((st + sn) * (IntV.ofNat d.svt : I32)) : Tensor [BLOCK, d.BD] I32)

/-- The mask of the `v` load (lines 164–168). -/
def maskV (d : Inst) (len sn : I32) : Tensor [BLOCK, d.BD] Bool :=
  (((sn + offsN).col d.BD).ltS len).and ((maskD d).row BLOCK)

/-- The pointers of the `o` store (lines 176–181): `Out + off_o`. -/
def ptrsO (d : Inst) (h sm st : I32) : Tensor [BLOCK, d.BD] (Ptr .bf16) :=
  d.o.ptr 0 + ((st + (offsM sm).col d.BD) * (IntV.ofNat d.sot : I32) +
    bc (h * (IntV.ofNat d.soh : I32)) + (offsD d).row BLOCK)

/-- Row `i` of query block `m`: `m * 128 + i`. -/
def rowBlk (m : Nat) : Fin BLOCK → Nat := fun i => m * BLOCK + i

/-- Key `i` of the block starting at `T`: `T + i`. -/
def keyBlk (T : Nat) : Fin BLOCK → Nat := fun i => T + i

/-- Lane `i` of `n` lanes, as a natural number. -/
def lane (n : Nat) : Fin n → Nat := fun i => i

/-! ## Lane values -/

section Lanes

variable {d : Inst}

/-- Lane `i` of `offs_m`: `m * 128 + i`, for a query block below `cdiv(max_input_len, 128)`. -/
theorem offsM_lane (hd : d.ok) {sm : I32} {mN : Nat} (hsm : sm.toInt = mN)
    (hmN : mN < nBlocks d) (i : Fin BLOCK) :
    ((offsM sm).get1 i).toInt = ((rowBlk mN i : Nat) : Int) := by
  have hb := nBlocks_le hd
  have h1 : (sm * (IntV.ofNat BLOCK : I32)).toInt = ((mN * BLOCK : Nat) : Int) :=
    IntV.mul_ofNat hsm (by simp only [BLOCK]; omega)
  exact IntV.I32.toInt_sadd_arange h1 (by simp only [BLOCK]; omega) i

/-- Lane `i` of `offs_n`: `i`. -/
theorem offsN_lane (i : Fin BLOCK) : ((offsN.get1 i).toInt = ((i : Nat) : Int)) :=
  IntV.I32.toInt_tl_arange (n := BLOCK) (by unfold BLOCK; omega) i

/-- Lane `j` of `offs_d`: `j`. -/
theorem offsD_lane (hd : d.ok) (j : Fin d.BD) :
    ((offsD d).get1 j).toInt = ((lane d.BD j : Nat) : Int) :=
  IntV.I32.toInt_tl_arange (le_trans hd.bd_le (by norm_num)) j

/-- Lane `j` of `mask_d`: `j < Lk`. -/
theorem maskD_lane (hd : d.ok) (j : Fin d.BD) : (maskD d).get1 j = decide (lane d.BD j < d.Lk) := by
  have hLk : d.Lk < 2 ^ 31 := lt_of_le_of_lt (le_trans hd.lk_le hd.bd_le) (by norm_num)
  show IntV.lt ((offsD d).get1 j) (IntV.ofNat d.Lk) = _
  rw [IntV.lt_eq, offsD_lane hd j, IntV.I32.toInt_ofNat hLk]
  simp only [Nat.cast_lt]

/-- Lane `i` of `start + offs_n`: `T + i`, for a block start below `len ≤ 2^22`. -/
theorem startN_lane {sn : I32} {T L : Nat} (hsn : sn.toInt = T) (hT : T < L) (hL : L ≤ 2 ^ 22)
    (i : Fin BLOCK) : ((sn + offsN).get1 i).toInt = ((keyBlk T i : Nat) : Int) :=
  IntV.I32.toInt_sadd_arange (n := BLOCK) hsn (by unfold BLOCK; omega) i

/-- `(st + t) * a + h * c + j` of natural values does not wrap when its value is below `2^31`. -/
theorem off_sum {st t a h c j : I32} {S R A Hn C J : Nat} (hst : st.toInt = S) (ht : t.toInt = R)
    (ha : a.toInt = A) (hh : h.toInt = Hn) (hc : c.toInt = C) (hj : j.toInt = J)
    (hSR : S + R < 2 ^ 31) (hb : (S + R) * A + Hn * C + J < 2 ^ 31) :
    ((st + t) * a + h * c + j).toInt = (((S + R) * A + Hn * C + J : Nat) : Int) := by
  have h1 := IntV.I32.add_nat hst ht hSR
  have h2 := IntV.I32.mul_nat h1 ha (by omega)
  have h3 := IntV.I32.mul_nat hh hc (by omega)
  have h4 := IntV.I32.add_nat h2 h3 (by omega)
  exact IntV.I32.add_nat h4 hj hb

/-- `t * a + h * c + j` of natural values does not wrap when its value is below `2^31`. -/
theorem off_sum' {t a h c j : I32} {R A Hn C J : Nat} (ht : t.toInt = R) (ha : a.toInt = A)
    (hh : h.toInt = Hn) (hc : c.toInt = C) (hj : j.toInt = J)
    (hb : R * A + Hn * C + J < 2 ^ 31) :
    (t * a + h * c + j).toInt = ((R * A + Hn * C + J : Nat) : Int) := by
  have h2 := IntV.I32.mul_nat ht ha (by omega)
  have h3 := IntV.I32.mul_nat hh hc (by omega)
  have h4 := IntV.I32.add_nat h2 h3 (by omega)
  exact IntV.I32.add_nat h4 hj hb

/-- An element of a `[total, H, Lk]` tensor with the given strides has an index below `2^31`
when its last element does (`q_i32` and the like). -/
theorem idx_lt {t T h Hh j L s0 s1 : Nat} (ht : t < T) (hh : h < Hh) (hj : j < L)
    (hb : (T - 1) * s0 + (Hh - 1) * s1 + L ≤ 2 ^ 31) : t * s0 + h * s1 + j < 2 ^ 31 := by
  have h1 : t * s0 ≤ (T - 1) * s0 := Nat.mul_le_mul_right _ (by omega)
  have h2 : h * s1 ≤ (Hh - 1) * s1 := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The last element of the `L` rows from `S` of head `h` is in the buffer. -/
theorem last_lt {S L h T Hh Lk s0 s1 len : Nat} (hL : 0 < L) (hSL : S + L ≤ T) (hh : h < Hh)
    (hLk : 0 < Lk) (hb : (T - 1) * s0 + (Hh - 1) * s1 + Lk ≤ len) :
    S * s0 + h * s1 + (L - 1) * s0 + (Lk - 1) < len := by
  have h1 : (S + (L - 1)) * s0 ≤ (T - 1) * s0 := Nat.mul_le_mul_right _ (by omega)
  have h2 : h * s1 ≤ (Hh - 1) * s1 := Nat.mul_le_mul_right _ (by omega)
  have h3 : (S + (L - 1)) * s0 = S * s0 + (L - 1) * s0 := Nat.add_mul _ _ _
  omega

end Lanes

/-! ## The accesses -/

section Statements

variable {d : Inst} {init : Mem}

/-- `kv_group_num` divides: the key/value head of a query head is a key/value head. -/
theorem kvh_lt (hd : d.ok) {h : Nat} (hh : h < d.H) : kvh d h < d.HKV := by
  have hpos := hd.kv_pos
  have hkv : 0 < kvg d := Nat.div_pos (Nat.le_of_dvd (by omega) hd.kv_dvd) hpos
  unfold kvh
  rw [Nat.div_lt_iff_lt_mul hkv]
  have : d.HKV * kvg d = d.H := Nat.mul_div_cancel' hd.kv_dvd
  omega

/-- The query block of a program is below `cdiv(max_input_len, 128)`. -/
theorem mOf_lt (d : Inst) (i : Fin (launch d).size) : mOf d i < nBlocks d := by
  exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq i.isLt (launch_size d))

/-- A sequence of the batch lies in the tokens: `start + len ≤ total_tokens`. -/
theorem seq_le (hpre : Pre d init) {b : Nat} (hb : b < d.batch) :
    startOf d init b + lenOf d init b ≤ d.total := by
  obtain ⟨s, n, hs, hn, h0, h1, -, h3⟩ := hpre.2.2.2.2.2.2.1.1 b hb
  simp only [startOf, lenOf, val32, hs, hn]
  omega

/-- **The `q` pointers**: on an enabled lane (row `m * 128 + i < L`, component `j < Lk`) the
pointer is element `[S + m * 128 + i, hN, j]` of `q`, with no `int32` wrap. -/
theorem q_ptrs (hd : d.ok) {st h sm : I32} {S L hN mN : Nat} (hst : st.toInt = S)
    (hh : h.toInt = hN) (hsm : sm.toInt = mN) (hSL : S + L ≤ d.total) (hhN : hN < d.H)
    (hmN : mN < nBlocks d) :
    ∀ (i : Fin BLOCK) (j : Fin d.BD), rowBlk mN i < L → lane d.BD j < d.Lk →
      (ptrsQ d h sm st).get2 i j =
        d.q.ptr ((S * d.sqt + hN * d.sqh + rowBlk mN i * d.sqt + lane d.BD j * 1 : Nat) : Int) := by
  intro i j hi hj
  have htot := hd.total_le
  have hq := hd.q_i32 (by omega)
  have hidx := idx_lt (t := S + rowBlk mN i) (T := d.total) (by omega) hhN (L := d.Lk) hj hq
  have hsqt : d.sqt < 2 ^ 31 := hd.strides_lt.1
  have hsqh : d.sqh < 2 ^ 31 := hd.strides_lt.2.1
  have hrow := offsM_lane hd hsm hmN i
  have hoff := off_sum (c := (IntV.ofNat d.sqh : I32)) (j := (offsD d).get1 j) hst hrow
    (IntV.I32.toInt_ofNat hsqt) hh (IntV.I32.toInt_ofNat hsqh) (offsD_lane hd j)
    (by omega) hidx
  have e : (S + rowBlk mN i) * d.sqt = S * d.sqt + rowBlk mN i * d.sqt := Nat.add_mul _ _ _
  change (d.q.ptr 0).offset (((st + (offsM sm).get1 i) * (IntV.ofNat d.sqt : I32) +
    h * (IntV.ofNat d.sqh : I32) + (offsD d).get1 j).toInt) = _
  rw [hoff, Buf.ptr_offset]
  congr 1
  push_cast
  ring

/-- **The `q` mask**: lane `(i, j)` is on exactly when `m * 128 + i < L` and `j < Lk`. -/
theorem q_mask (hd : d.ok) {len sm : I32} {L mN : Nat} (hlen : len.toInt = L)
    (hsm : sm.toInt = mN) (hmN : mN < nBlocks d) :
    MaskAt (some (maskQ d len sm)) (rowBlk mN) (lane d.BD) L d.Lk := by
  intro i j
  show (IntV.lt ((offsM sm).get1 i) len && (maskD d).get1 j) = _
  rw [IntV.lt_eq, offsM_lane hd hsm hmN i, hlen, maskD_lane hd j]
  simp only [Nat.cast_lt]

/-- **The `q` tile is in the buffer**. -/
theorem q_len (hd : d.ok) {S L hN : Nat} (hSL : S + L ≤ d.total) (hhN : hN < d.H) :
    L = 0 ∨ d.Lk = 0 ∨
      S * d.sqt + hN * d.sqh + (L - 1) * d.sqt + (d.Lk - 1) * 1 < d.q.len := by
  rcases Nat.eq_zero_or_pos L with h0 | hL
  · exact Or.inl h0
  rcases Nat.eq_zero_or_pos d.Lk with h1 | hLk
  · exact Or.inr (Or.inl h1)
  refine Or.inr (Or.inr ?_)
  have := last_lt (s0 := d.sqt) (s1 := d.sqh) hL hSL hhN hLk (hd.q_len (by omega))
  omega

/-- **The `q` tile is read**: its bytes are in `Rf`. -/
theorem q_Rf (hd : d.ok) {S L hN mN : Nat} (hSL : S + L ≤ d.total) (hhN : hN < d.H) :
    d.q.tileAt (S * d.sqt + hN * d.sqh) d.sqt 1 (rowBlk mN) (lane d.BD) L d.Lk ⊆ Rf d :=
  (d.q.tileAt_sub_bytes (q_len hd hSL hhN)).trans fun _ hx => Or.inl (Or.inl (Or.inl (Or.inl hx)))

/-- **The `k` pointers**: on an enabled lane (component `i < Lk`, key `T + j < L`) the pointer
is element `[S + T + j, gN, i]` of `k`, with no `int32` wrap. -/
theorem k_ptrs (hd : d.ok) {st g sn : I32} {S L gN T : Nat} (hst : st.toInt = S)
    (hg : g.toInt = gN) (hsn : sn.toInt = T) (hSL : S + L ≤ d.total) (hgN : gN < d.HKV)
    (hT : T < L) :
    ∀ (i : Fin d.BD) (j : Fin BLOCK), lane d.BD i < d.Lk → keyBlk T j < L →
      (ptrsK d g st sn).get2 i j =
        d.k.ptr ((S * d.skt + gN * d.skh + lane d.BD i * 1 + keyBlk T j * d.skt : Nat) : Int) := by
  intro i j hi hj
  have htot := hd.total_le
  have hk := hd.k_i32 (by omega)
  have hidx := idx_lt (t := S + keyBlk T j) (T := d.total) (h := gN) (Hh := d.HKV) (L := d.Lk)
    (by omega) hgN hi hk
  have hskt : d.skt < 2 ^ 31 := hd.strides_lt.2.2.1
  have hskh : d.skh < 2 ^ 31 := hd.strides_lt.2.2.2.1
  have e : (S + keyBlk T j) * d.skt = S * d.skt + keyBlk T j * d.skt := Nat.add_mul _ _ _
  have e2 : (S + T) * d.skt = S * d.skt + T * d.skt := Nat.add_mul _ _ _
  have e3 : keyBlk T j * d.skt = T * d.skt + (j : Nat) * d.skt := by
    unfold keyBlk; exact Nat.add_mul _ _ _
  have hX := off_sum' (a := (IntV.ofNat d.skt : I32)) (c := (IntV.ofNat d.skh : I32))
    (j := (offsD d).get1 i) (offsN_lane j) (IntV.I32.toInt_ofNat hskt) hg
    (IntV.I32.toInt_ofNat hskh) (offsD_lane hd i) (by omega)
  have hY := IntV.I32.mul_nat (IntV.I32.add_nat hst hsn (by omega))
    (IntV.I32.toInt_ofNat hskt) (by omega)
  change ((d.k.ptr 0).offset ((offsN.get1 j * (IntV.ofNat d.skt : I32) +
    g * (IntV.ofNat d.skh : I32) + (offsD d).get1 i).toInt)).offset
    (((st + sn) * (IntV.ofNat d.skt : I32)).toInt) = _
  rw [hX, hY, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  simp only [keyBlk]
  push_cast
  ring

/-- **The `k` mask**: lane `(i, j)` is on exactly when `i < Lk` and `T + j < L`. -/
theorem k_mask (hd : d.ok) {len sn : I32} {L T : Nat} (hlen : len.toInt = L)
    (hsn : sn.toInt = T) (hL : L ≤ d.total) (hT : T < L) :
    MaskAt (some (maskK d len sn)) (lane d.BD) (keyBlk T) d.Lk L := by
  have htot := hd.total_le
  intro i j
  show (IntV.lt ((sn + offsN).get1 j) len && (maskD d).get1 i) = _
  rw [IntV.lt_eq, startN_lane hsn hT (by omega) j, hlen, maskD_lane hd i, Bool.and_comm]
  simp only [Nat.cast_lt]

/-- **The `k` tile is in the buffer**. -/
theorem k_len (hd : d.ok) {S L gN : Nat} (hSL : S + L ≤ d.total) (hgN : gN < d.HKV) :
    d.Lk = 0 ∨ L = 0 ∨
      S * d.skt + gN * d.skh + (d.Lk - 1) * 1 + (L - 1) * d.skt < d.k.len := by
  rcases Nat.eq_zero_or_pos L with h0 | hL
  · exact Or.inr (Or.inl h0)
  rcases Nat.eq_zero_or_pos d.Lk with h1 | hLk
  · exact Or.inl h1
  refine Or.inr (Or.inr ?_)
  have := last_lt (s0 := d.skt) (s1 := d.skh) hL hSL hgN hLk (hd.k_len (by omega))
  omega

/-- **The `k` tile is read**: its bytes are in `Rf`. -/
theorem k_Rf (hd : d.ok) {S L gN T : Nat} (hSL : S + L ≤ d.total) (hgN : gN < d.HKV) :
    d.k.tileAt (S * d.skt + gN * d.skh) 1 d.skt (lane d.BD) (keyBlk T) d.Lk L ⊆ Rf d :=
  (d.k.tileAt_sub_bytes (k_len hd hSL hgN)).trans fun _ hx => Or.inl (Or.inl (Or.inl (Or.inr hx)))

/-- **The `v` pointers**: on an enabled lane (key `T + i < L`, component `j < Lk`) the pointer
is element `[S + T + i, gN, j]` of `v`, with no `int32` wrap. -/
theorem v_ptrs (hd : d.ok) {st g sn : I32} {S L gN T : Nat} (hst : st.toInt = S)
    (hg : g.toInt = gN) (hsn : sn.toInt = T) (hSL : S + L ≤ d.total) (hgN : gN < d.HKV)
    (hT : T < L) :
    ∀ (i : Fin BLOCK) (j : Fin d.BD), keyBlk T i < L → lane d.BD j < d.Lk →
      (ptrsV d g st sn).get2 i j =
        d.v.ptr ((S * d.svt + gN * d.svh + keyBlk T i * d.svt + lane d.BD j * 1 : Nat) : Int) := by
  intro i j hi hj
  have htot := hd.total_le
  have hv := hd.v_i32 (by omega)
  have hidx := idx_lt (t := S + keyBlk T i) (T := d.total) (h := gN) (Hh := d.HKV) (L := d.Lk)
    (by omega) hgN hj hv
  have hsvt : d.svt < 2 ^ 31 := hd.strides_lt.2.2.2.2.1
  have hsvh : d.svh < 2 ^ 31 := hd.strides_lt.2.2.2.2.2.1
  have e : (S + keyBlk T i) * d.svt = S * d.svt + keyBlk T i * d.svt := Nat.add_mul _ _ _
  have e2 : (S + T) * d.svt = S * d.svt + T * d.svt := Nat.add_mul _ _ _
  have e3 : keyBlk T i * d.svt = T * d.svt + (i : Nat) * d.svt := by
    unfold keyBlk; exact Nat.add_mul _ _ _
  have hX := off_sum' (a := (IntV.ofNat d.svt : I32)) (c := (IntV.ofNat d.svh : I32))
    (j := (offsD d).get1 j) (offsN_lane i) (IntV.I32.toInt_ofNat hsvt) hg
    (IntV.I32.toInt_ofNat hsvh) (offsD_lane hd j) (by omega)
  have hY := IntV.I32.mul_nat (IntV.I32.add_nat hst hsn (by omega))
    (IntV.I32.toInt_ofNat hsvt) (by omega)
  change ((d.v.ptr 0).offset ((offsN.get1 i * (IntV.ofNat d.svt : I32) +
    g * (IntV.ofNat d.svh : I32) + (offsD d).get1 j).toInt)).offset
    (((st + sn) * (IntV.ofNat d.svt : I32)).toInt) = _
  rw [hX, hY, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  simp only [keyBlk]
  push_cast
  ring

/-- **The `v` mask**: lane `(i, j)` is on exactly when `T + i < L` and `j < Lk`. -/
theorem v_mask (hd : d.ok) {len sn : I32} {L T : Nat} (hlen : len.toInt = L)
    (hsn : sn.toInt = T) (hL : L ≤ d.total) (hT : T < L) :
    MaskAt (some (maskV d len sn)) (keyBlk T) (lane d.BD) L d.Lk := by
  have htot := hd.total_le
  intro i j
  show (IntV.lt ((sn + offsN).get1 i) len && (maskD d).get1 j) = _
  rw [IntV.lt_eq, startN_lane hsn hT (by omega) i, hlen, maskD_lane hd j]
  simp only [Nat.cast_lt]

/-- **The `v` tile is in the buffer**. -/
theorem v_len (hd : d.ok) {S L gN : Nat} (hSL : S + L ≤ d.total) (hgN : gN < d.HKV) :
    L = 0 ∨ d.Lk = 0 ∨
      S * d.svt + gN * d.svh + (L - 1) * d.svt + (d.Lk - 1) * 1 < d.v.len := by
  rcases Nat.eq_zero_or_pos L with h0 | hL
  · exact Or.inl h0
  rcases Nat.eq_zero_or_pos d.Lk with h1 | hLk
  · exact Or.inr (Or.inl h1)
  refine Or.inr (Or.inr ?_)
  have := last_lt (s0 := d.svt) (s1 := d.svh) hL hSL hgN hLk (hd.v_len (by omega))
  omega

/-- **The `v` tile is read**: its bytes are in `Rf`. -/
theorem v_Rf (hd : d.ok) {S L gN T : Nat} (hSL : S + L ≤ d.total) (hgN : gN < d.HKV) :
    d.v.tileAt (S * d.svt + gN * d.svh) d.svt 1 (keyBlk T) (lane d.BD) L d.Lk ⊆ Rf d :=
  (d.v.tileAt_sub_bytes (v_len hd hSL hgN)).trans fun _ hx => Or.inl (Or.inl (Or.inr hx))

/-- **The `o` pointers**: on an enabled lane the pointer is element `[S + m * 128 + i, hN, j]`
of `o`, with no `int32` wrap. -/
theorem o_ptrs (hd : d.ok) {st h sm : I32} {S L hN mN : Nat} (hst : st.toInt = S)
    (hh : h.toInt = hN) (hsm : sm.toInt = mN) (hSL : S + L ≤ d.total) (hhN : hN < d.H)
    (hmN : mN < nBlocks d) :
    ∀ (i : Fin BLOCK) (j : Fin d.BD), rowBlk mN i < L → lane d.BD j < d.Lk →
      (ptrsO d h sm st).get2 i j =
        d.o.ptr ((S * d.sot + hN * d.soh + rowBlk mN i * d.sot + lane d.BD j * 1 : Nat) : Int) := by
  intro i j hi hj
  have htot := hd.total_le
  have ho := hd.o_i32 (by omega)
  have hidx := idx_lt (t := S + rowBlk mN i) (T := d.total) (by omega) hhN (L := d.Lk) hj ho
  have hsot : d.sot < 2 ^ 31 := hd.strides_lt.2.2.2.2.2.2.1
  have hsoh : d.soh < 2 ^ 31 := hd.strides_lt.2.2.2.2.2.2.2
  have hrow := offsM_lane hd hsm hmN i
  have hoff := off_sum (c := (IntV.ofNat d.soh : I32)) (j := (offsD d).get1 j) hst hrow
    (IntV.I32.toInt_ofNat hsot) hh (IntV.I32.toInt_ofNat hsoh) (offsD_lane hd j)
    (by omega) hidx
  have e : (S + rowBlk mN i) * d.sot = S * d.sot + rowBlk mN i * d.sot := Nat.add_mul _ _ _
  change (d.o.ptr 0).offset (((st + (offsM sm).get1 i) * (IntV.ofNat d.sot : I32) +
    h * (IntV.ofNat d.soh : I32) + (offsD d).get1 j).toInt) = _
  rw [hoff, Buf.ptr_offset]
  congr 1
  push_cast
  ring

/-- **The `o` mask** is the `q` mask. -/
theorem o_mask (hd : d.ok) {len sm : I32} {L mN : Nat} (hlen : len.toInt = L)
    (hsm : sm.toInt = mN) (hmN : mN < nBlocks d) :
    MaskAt (some (maskQ d len sm)) (rowBlk mN) (lane d.BD) L d.Lk :=
  q_mask hd hlen hsm hmN

/-- **The `o` tile is in the buffer**. -/
theorem o_len (hd : d.ok) {S L hN : Nat} (hSL : S + L ≤ d.total) (hhN : hN < d.H) :
    L = 0 ∨ d.Lk = 0 ∨
      S * d.sot + hN * d.soh + (L - 1) * d.sot + (d.Lk - 1) * 1 < d.o.len := by
  rcases Nat.eq_zero_or_pos L with h0 | hL
  · exact Or.inl h0
  rcases Nat.eq_zero_or_pos d.Lk with h1 | hLk
  · exact Or.inr (Or.inl h1)
  refine Or.inr (Or.inr ?_)
  have := last_lt (s0 := d.sot) (s1 := d.soh) hL hSL hhN hLk (hd.o_len (by omega))
  omega

/-- **The `o` tile is written**: its bytes are the program's `Wf`. -/
theorem o_Wf (d : Inst) (init : Mem) (i : Nat) :
    d.o.tileAt (startOf d init (bOf d i) * d.sot + hOf d i * d.soh) d.sot 1
      (rowBlk (mOf d i)) (lane d.BD) (lenOf d init (bOf d i)) d.Lk ⊆ Wf d init i := by
  refine d.o.elems_mono ?_
  rintro x ⟨r, e, hrL, he, rfl⟩
  refine ⟨r, e, r.isLt, he, hrL, ?_⟩
  simp only [oIdx, rowBlk, lane]
  ring

/-- **The `o` layout does not alias**. -/
theorem o_inj (hd : d.ok) {hN L : Nat} (hhN : hN < d.H) : Inj d.sot 1 L d.Lk := by
  refine inj_rowMajor1 ?_
  rcases hd.o_layout with ⟨h1, h2⟩ | ⟨h1, -⟩
  · have : d.soh ≤ d.H * d.soh := Nat.le_mul_of_pos_left _ (by omega)
    omega
  · exact h1

/-- Distinct rows of the `o` tile are distinct lanes (`hρ`). -/
theorem o_rows_inj {mN L : Nat} :
    ∀ i i' : Fin BLOCK, rowBlk mN i < L → rowBlk mN i' < L → rowBlk mN i = rowBlk mN i' →
      i = i' :=
  add_inj

/-- Distinct components of the `o` tile are distinct lanes (`hκ`). -/
theorem o_cols_inj {n N : Nat} :
    ∀ j j' : Fin n, lane n j < N → lane n j' < N → lane n j = lane n j' → j = j' :=
  fun _ _ _ _ h => Fin.ext h

/-- **`tl.convert(0.0, bfloat16)`** (the `other=0.0` of a `bfloat16` load, `tl.full s zf32`
converted) is a tile of `+0`. -/
theorem convert_zero_safe {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {s : Shape} {k : Tensor s BF16 → Prog α}
    (hk : ∀ o : Tensor s BF16, (∀ ix, o.get ix = FloatV.ofPat 0) → Safe E R W Q m (k o)) :
    Safe E R W Q m (tl.convert .f32 .bf16 (tl.full s zf32) >>= k) := by
  refine safe_convert (d := .f32) (d' := .bf16) (fun i => ⟨_, ConvRel.cvt_f32_bf16 _⟩) ?_
  intro z hz
  refine hk z fun i => ?_
  have h := ConvRel.f32_bf16_iff.mp (hz i)
  have h' : (FloatV.cvt ((tl.full s zf32 : Tensor s F32).get i) : BF16) = FloatV.ofPat 0 := by
    show (FloatV.cvt (zf32 : F32) : BF16) = _
    unfold zf32
    rw [FloatV.litF32_ofPat_zero]
    exact FloatV.cvt_ofPat_zero Format.binary32_regular
  rw [h'] at h
  exact h.eq (by decide +kernel)

end Statements

end Claims.PrefillAttnVarlen

end
