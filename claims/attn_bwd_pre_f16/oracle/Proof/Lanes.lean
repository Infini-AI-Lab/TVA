import Claim
import Proof.Basic
import Proof.Index

/-!
# `attn_bwd_pre_f16`: the elements each program touches

The `readIdx` and `writeRow` values of a program's lanes are in bounds, the programs' write rows
are pairwise disjoint, and every row below `Z · H · N_CTX` is written by some program.
-/

namespace Claims.AttnBwdPre

open TVA


/-- The `Delta` row lane `l` of program `i` writes. -/
def writeRow (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) : Nat := writeIdx d i (l, ())

/-- The `Delta` row of lane `l`: `bh · N_CTX + pm · BLOCK_M + l`. -/
theorem writeRow_val (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) :
    writeRow d i l = bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val := rfl

/-- The `O`/`DO` elements program `i` reads. -/
def readSet (d : Inst) (i : Nat) : Set Nat :=
  {e | ∃ p : Idx [d.BLOCK_M, d.HEAD_DIM], readIdx d i p = e}

/-- The `Delta` elements program `i` writes. -/
def writeSet (d : Inst) (i : Nat) : Set Nat :=
  {e | ∃ l : Fin d.BLOCK_M, writeRow d i l = e}

/-- The `O`/`DO` element of lane `(l, k)`: `row · HEAD_DIM + k`. -/
theorem readIdx_val (d : Inst) (i : Nat) (l : Fin d.BLOCK_M) (k : Fin d.HEAD_DIM) :
    readIdx d i (l, (k, ())) =
      (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val) * d.HEAD_DIM + k.val := rfl

/-- `readIdx` is below `Z · H · N_CTX · HEAD_DIM`. -/
theorem readIdx_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M)
    (k : Fin d.HEAD_DIM) :
    readIdx d i (l, (k, ())) < d.Z * d.H * d.N_CTX * d.HEAD_DIM := by
  have hrow := row_lt d hd i l.val l.isLt
  unfold rowOf at hrow
  have hk := k.isLt
  rw [readIdx_val]
  have h1 : (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val) * d.HEAD_DIM + k.val <
      (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val) * d.HEAD_DIM + d.HEAD_DIM :=
    Nat.add_lt_add_left hk _
  have h2 : (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val) * d.HEAD_DIM + d.HEAD_DIM =
      (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val + 1) * d.HEAD_DIM := by ring
  have h3 : (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val + 1) * d.HEAD_DIM ≤
      d.Z * d.H * d.N_CTX * d.HEAD_DIM := Nat.mul_le_mul_right _ (by omega)
  calc (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val) * d.HEAD_DIM + k.val
      < (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val) * d.HEAD_DIM + d.HEAD_DIM := h1
    _ = (bh d i * d.N_CTX + pm d i * d.BLOCK_M + l.val + 1) * d.HEAD_DIM := h2
    _ ≤ d.Z * d.H * d.N_CTX * d.HEAD_DIM := h3

/-- `readIdx` is a valid element of `O`. -/
theorem readIdx_lt_O (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M)
    (k : Fin d.HEAD_DIM) : readIdx d i (l, (k, ())) < d.O.len :=
  lt_of_lt_of_le (readIdx_lt d hd i l k) hd.O_len

/-- `readIdx` is a valid element of `DO`. -/
theorem readIdx_lt_DO (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M)
    (k : Fin d.HEAD_DIM) : readIdx d i (l, (k, ())) < d.DO.len :=
  lt_of_lt_of_le (readIdx_lt d hd i l k) hd.DO_len

/-- `writeRow` is below `Z · H · N_CTX`. -/
theorem writeRow_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M) :
    writeRow d i l < d.Z * d.H * d.N_CTX := by
  have h := row_lt d hd i l.val l.isLt
  simpa [writeRow, writeIdx, rowAt, rowOf] using h

/-- `writeRow` is a valid element of `Delta`. -/
theorem writeRow_lt_Delta (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.BLOCK_M) :
    writeRow d i l < d.Delta.len :=
  lt_of_lt_of_le (writeRow_lt d hd i l) hd.Delta_len

/-- The elements program `i` reads lie below `Z · H · N_CTX · HEAD_DIM`. -/
theorem readSet_sub (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    readSet d i ⊆ inSet d := by
  rintro e ⟨p, rfl⟩
  exact readIdx_lt d hd i p.1 p.2.1

/-- The elements program `i` writes lie below `Z · H · N_CTX`. -/
theorem writeSet_sub (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    writeSet d i ⊆ outSet d := by
  rintro e ⟨l, rfl⟩
  exact writeRow_lt d hd i l

/-- `Buf.img readIdx (fun _ => true)` is `readSet`. -/
theorem img_readSet (d : Inst) (i : Nat) :
    Buf.img (readIdx d i) (fun _ => true) = readSet d i := by
  ext e
  constructor
  · rintro ⟨p, -, rfl⟩
    exact ⟨p, rfl⟩
  · rintro ⟨p, rfl⟩
    exact ⟨p, rfl, rfl⟩

/-- `Buf.img writeIdx (fun _ => true)` is `writeSet`. -/
theorem img_writeSet (d : Inst) (i : Nat) :
    Buf.img (writeIdx d i) (fun _ => true) = writeSet d i := by
  ext e
  constructor
  · rintro ⟨p, -, h⟩
    obtain ⟨l, u⟩ := p
    exact ⟨l, h⟩
  · rintro ⟨l, h⟩
    exact ⟨(l, ()), rfl, h⟩

/-! ### A program is determined by its grid coordinates -/

/-- `i / gx` is `bh d i`. -/
theorem div_gx_eq_bh (d : Inst) (i : Fin (launch d).size) : (i : Nat) / gx d = bh d i := by
  have hsz : (i : Nat) < gx d * gy d := Nat.lt_of_lt_of_eq i.isLt (launch_size d)
  have hpos : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h | h
    · rw [h, Nat.zero_mul] at hsz
      exact absurd hsz (Nat.not_lt_zero _)
    · exact h
  have hlt : (i : Nat) / gx d < gy d :=
    (Nat.div_lt_iff_lt_mul hpos).mpr (by rw [Nat.mul_comm]; exact hsz)
  unfold bh
  rw [Nat.mod_eq_of_lt hlt]

/-- `i % gx` is `pm d i`. -/
theorem mod_gx_eq_pm (d : Inst) (i : Fin (launch d).size) : (i : Nat) % gx d = pm d i := rfl

/-- Two programs with the same coordinates are equal. -/
theorem prog_ext (d : Inst) {i j : Fin (launch d).size}
    (hb : bh d i = bh d j) (hp : pm d i = pm d j) : i = j := by
  apply Fin.ext
  have hi : (i : Nat) = bh d i * gx d + pm d i := by
    have h := (Nat.div_add_mod' (i : Nat) (gx d)).symm
    rw [div_gx_eq_bh, mod_gx_eq_pm] at h
    exact h
  have hj : (j : Nat) = bh d j * gx d + pm d j := by
    have h := (Nat.div_add_mod' (j : Nat) (gx d)).symm
    rw [div_gx_eq_bh, mod_gx_eq_pm] at h
    exact h
  rw [hi, hj, hb, hp]

/-! ### The write rows are pairwise disjoint -/

/-- Two programs writing the same row are equal. -/
theorem writeSet_aux (d : Inst) (hd : d.ok) {i j : Fin (launch d).size} {e : Nat}
    (hi : e ∈ writeSet d i) (hj : e ∈ writeSet d j) : i = j := by
  obtain ⟨li, hi⟩ := hi
  obtain ⟨lj, hj⟩ := hj
  have hBeq : writeRow d i li = writeRow d j lj := hi.trans hj.symm
  have hN : 0 < d.N_CTX := by
    have := pm_block_lt d hd i li.val li.isLt
    omega
  have hB : 0 < d.BLOCK_M := block_pos hd
  have hri : pm d i * d.BLOCK_M + li.val < d.N_CTX :=
    pm_block_lt d hd i li.val li.isLt
  have hrj : pm d j * d.BLOCK_M + lj.val < d.N_CTX :=
    pm_block_lt d hd j lj.val lj.isLt
  rw [writeRow_val, writeRow_val, add_assoc, add_assoc] at hBeq
  -- `N_CTX` separates the block index
  have h1 : pm d i * d.BLOCK_M + li.val =
      (bh d i * d.N_CTX + (pm d i * d.BLOCK_M + li.val)) % d.N_CTX := by
    rw [Nat.mul_add_mod_self_right, Nat.mod_eq_of_lt hri]
  have h2 : pm d j * d.BLOCK_M + lj.val =
      (bh d j * d.N_CTX + (pm d j * d.BLOCK_M + lj.val)) % d.N_CTX := by
    rw [Nat.mul_add_mod_self_right, Nat.mod_eq_of_lt hrj]
  have hmodeq : pm d i * d.BLOCK_M + li.val = pm d j * d.BLOCK_M + lj.val := by
    rw [h1, h2, hBeq]
  -- `BLOCK_M` separates the lane
  have hpeq : pm d i = pm d j := by
    have hdvi : (pm d i * d.BLOCK_M + li.val) / d.BLOCK_M = pm d i := by
      rw [Nat.add_comm (pm d i * d.BLOCK_M) li.val, Nat.mul_comm (pm d i) d.BLOCK_M,
        Nat.add_mul_div_left li.val (pm d i) hB, Nat.div_eq_of_lt li.isLt, Nat.zero_add]
    have hdvj : (pm d j * d.BLOCK_M + lj.val) / d.BLOCK_M = pm d j := by
      rw [Nat.add_comm (pm d j * d.BLOCK_M) lj.val, Nat.mul_comm (pm d j) d.BLOCK_M,
        Nat.add_mul_div_left lj.val (pm d j) hB, Nat.div_eq_of_lt lj.isLt, Nat.zero_add]
    rw [← hdvi, ← hdvj, hmodeq]
  have hbeq : bh d i = bh d j := by
    have hh := hBeq
    rw [hmodeq] at hh
    exact Nat.mul_right_cancel hN (Nat.add_right_cancel hh)
  exact prog_ext d hbeq hpeq

/-- Programs `i ≠ j` write disjoint sets of rows. -/
theorem writeSet_disjoint (d : Inst) (hd : d.ok) {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (writeSet d i) (writeSet d j) := by
  rw [Set.disjoint_left]
  intro e hi hj
  exact hij (writeSet_aux d hd hi hj)

/-- Every row below `Z · H · N_CTX` is written by some program. -/
theorem write_cover (d : Inst) (hd : d.ok) {r : Nat} (hr : r < d.Z * d.H * d.N_CTX) :
    ∃ (i : Fin (launch d).size) (l : Fin d.BLOCK_M), writeRow d i l = r := by
  have hN : 0 < d.N_CTX := by
    rcases Nat.eq_zero_or_pos d.N_CTX with h | h
    · rw [h, Nat.mul_zero] at hr
      exact absurd hr (Nat.not_lt_zero _)
    · exact h
  have hbm := block_mul_gx d hd
  have hgx : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h | h
    · rw [h, Nat.mul_zero] at hbm
      omega
    · exact h
  have hZH : 0 < d.Z * d.H := by
    rcases Nat.eq_zero_or_pos (d.Z * d.H) with h | h
    · rw [h, Nat.zero_mul] at hr
      exact absurd hr (Nat.not_lt_zero _)
    · exact h
  set bh0 := r / d.N_CTX with hbh0
  set m := r % d.N_CTX with hm
  have hm_lt : m < d.N_CTX := Nat.mod_lt _ hN
  have hbh_lt : bh0 < d.Z * d.H := by
    rw [hbh0]
    exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm (d.N_CTX) (d.Z * d.H)]; exact hr)
  set pm0 := m / d.BLOCK_M with hpm0
  set l0 := m % d.BLOCK_M with hl0
  have hl0_lt : l0 < d.BLOCK_M := Nat.mod_lt _ (block_pos hd)
  have hpm_lt : pm0 < gx d := by
    rw [hpm0]
    exact Nat.div_lt_of_lt_mul (by rw [hbm]; exact hm_lt)
  set i0 := bh0 * gx d + pm0 with hi0
  have hi_lt : i0 < (launch d).size := by
    rw [launch_size]
    calc i0 = bh0 * gx d + pm0 := hi0
      _ < bh0 * gx d + gx d := Nat.add_lt_add_left hpm_lt _
      _ = (bh0 + 1) * gx d := by ring
      _ ≤ (d.Z * d.H) * gx d := Nat.mul_le_mul_right _ (by omega)
      _ = gx d * gy d := by rw [gy, Nat.mul_comm]
  have hpm_i : pm d i0 = pm0 := by
    unfold pm
    rw [hi0, Nat.add_comm (bh0 * gx d) pm0, Nat.mul_comm bh0 (gx d),
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hpm_lt]
  have hbh_i : bh d i0 = bh0 := by
    unfold bh
    have hdiv : (i0 : Nat) / gx d = bh0 := by
      rw [hi0, Nat.add_comm (bh0 * gx d) pm0, Nat.mul_comm bh0 (gx d),
        Nat.add_mul_div_left pm0 bh0 hgx, Nat.div_eq_of_lt hpm_lt, Nat.zero_add]
    rw [hdiv]
    exact Nat.mod_eq_of_lt hbh_lt
  let ifin : Fin (launch d).size := ⟨i0, hi_lt⟩
  refine ⟨ifin, ⟨l0, hl0_lt⟩, ?_⟩
  have hif : (ifin : Nat) = i0 := rfl
  have hpm_if : pm d (ifin : Nat) = pm0 := by rw [hif]; exact hpm_i
  have hbh_if : bh d (ifin : Nat) = bh0 := by rw [hif]; exact hbh_i
  show writeRow d (ifin : Nat) ⟨l0, hl0_lt⟩ = r
  rw [writeRow_val, hbh_if, hpm_if]
  have hmdec : d.BLOCK_M * pm0 + l0 = m := by
    rw [hpm0, hl0]
    exact Nat.div_add_mod m d.BLOCK_M
  have hrdec : d.N_CTX * bh0 + m = r := by
    rw [hbh0, hm]
    exact Nat.div_add_mod r d.N_CTX
  rw [Nat.mul_comm d.BLOCK_M pm0] at hmdec
  rw [Nat.mul_comm d.N_CTX bh0] at hrdec
  rw [add_assoc, hmdec, hrdec]

end Claims.AttnBwdPre
