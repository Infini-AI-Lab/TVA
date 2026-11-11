import Proof.Defs
import Proof.Facts
import Proof.Loop

/-!
# `lora_shrink_bf16`: the lanes of the pointer tiles and the rank vector

The values of the lanes of the kernel's index vectors and pointer tiles at the start of the loop
(lines 236-252), and the hints of line 237.
-/

namespace Claims.LoraShrink

open TVA

/-- **The pointers `base + x + y`** of a 2-D tile, a row term `x` and a column term `y` (of any
widths): lane `(i, j)` points at element `e + X i j + Y i j`. -/
theorem ptr_seq2 {dt : DType} (b : Buf dt) {base : Ptr dt} {e : Nat} (hb : base = b.ptr (e : Int))
    {m n : Nat} {sg sg' : Bool} {w w' : Nat} {x : Tensor [m, n] (IntV sg w)}
    {y : Tensor [m, n] (IntV sg' w')} {X Y : Fin m → Fin n → Nat}
    (hx : ∀ i j, (x.get2 i j).toInt = (X i j : Int)) (hy : ∀ i j, (y.get2 i j).toInt = (Y i j : Int))
    (i : Fin m) (j : Fin n) :
    (base + x + y).get2 i j = b.ptr ((e + X i j + Y i j : Nat) : Int) := by
  rw [Tensor.get2_ptrAddT, Tensor.get2_ptrAdd, hx, hy, hb, Buf.ptr_offset, Buf.ptr_offset]
  push_cast
  ring_nf

/-- Consecutive values `B + i`: the promise `tl.max_contiguous`. -/
theorem contigArange {n C B : Nat} {x : Tensor [n] I32}
    (hB : ∀ i : Fin n, (x.get1 i).toInt = ((B + i.val : Nat) : Int)) : ContigAlong x 0 C := by
  intro i j _
  obtain ⟨i1, iu⟩ := i
  obtain ⟨j1, ju⟩ := j
  cases iu
  cases ju
  show (x.get1 j1).toInt = (x.get1 i1).toInt + ((j1.val : Int) - (i1.val : Int)) * 1
  rw [hB i1, hB j1]
  push_cast
  ring

/-- The first value of the group of `C ≥ n` positions is a multiple of `C`: the promise
`tl.multiple_of`. -/
theorem firstsArange {n C B : Nat} {x : Tensor [n] I32} (hCn : n ≤ C)
    (hB : ∀ i : Fin n, (x.get1 i).toInt = ((B + i.val : Nat) : Int))
    (hdvd : (C : Int) ∣ (B : Int)) : FirstsDiv x 0 C C := by
  intro i hi
  show (C : Int) ∣ (x.get1 i.1).toInt
  have hi' : i.1.val % C = 0 := hi
  rw [hB i.1]
  have hlt : i.1.val < C := lt_of_lt_of_le i.1.isLt hCn
  have h0 : i.1.val = 0 := Nat.eq_zero_of_dvd_of_lt (Nat.dvd_of_mod_eq_zero hi') hlt
  simp only [h0, Nat.add_zero]
  exact hdvd

/-- The column term of the `inputs` pointer tile, `(pid_sk · BLOCK_K + tl.arange(0, BLOCK_K)) · 1`
(line 244). -/
abbrev colK (d : Inst) (i : Nat) : Tensor [bk d] I32 :=
  ((IntV.ofNat (part d i * bk d) : I32) + tl.arange 0 (bk d)) * (IntV.ofNat 1 : I32)

/-- Lane `kk` of the column term has value `part · BLOCK_K + kk`. -/
theorem offsetK_lane {d : Inst} {i : Nat} (kk : Fin (bk d)) :
    ((colK d i).get1 kk).toInt = (((part d i * bk d + kk) * 1 : Nat) : Int) := by
  have hbk : bk d ≤ 256 := by rcases bk_sk_cases d with h | h <;> omega
  have hp : part d i < sk d := Nat.mod_lt _ (sk_pos d)
  have hsk : sk d ≤ 64 := by rcases bk_sk_cases d with h | h <;> omega
  have hpb : part d i * bk d ≤ 64 * 256 := Nat.mul_le_mul (by omega) hbk
  show ((((IntV.ofNat (part d i * bk d) : I32) + tl.arange 0 (bk d)) * (IntV.ofNat 1 : I32)).get1 kk).toInt = _
  rw [Tensor.get1_muls, Tensor.get1_sadd]
  exact IntV.I32.mul_nat (IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega))
    (IntV.I32.toInt_tl_arange (n := bk d) (by omega) kk) (by omega)) (IntV.I32.toInt_ofNat (by norm_num))
    (by omega)

/-- `ptr_seq2` for the `inputs` tile at the start of the loop (lines 241-246): the row term is the
`int64` token index times the `int64` row stride `K`, the column term `pid_sk · BLOCK_K + kk`. -/
theorem aPtr_zero {d : Inst} (hd : d.ok) {i : Nat} (hact : Active d i)
    {ram64 : Tensor [32 - 0] I64} {d0 : I64}
    (hr64 : ∀ r : Fin (32 - 0), (ram64.get1 r).toInt = ((tokOf d i (r.val % mlen d i) : Nat) : Int))
    (hd0 : d0.toInt = (d.K : Int)) (r : Fin 32) (kk : Fin (bk d)) :
    (d.inp.ptr 0 + ram64.col (bk d) * d0 +
      (colK d i).row 32).get2 r kk = d.inp.ptr ((aIdx d i r kk 0 : Nat) : Int) := by
  have hrow := hact.row hd (Nat.mod_lt r.val hact.mlen_pos)
  have hK := hd.K_top
  have hM := hd.M_le
  have hx : ∀ (r : Fin 32) (kk : Fin (bk d)), ((ram64.col (bk d) * d0).get2 r kk).toInt =
      ((tokOf d i (r.val % mlen d i) * d.K : Nat) : Int) := by
    intro r kk
    have hrow := hact.row hd (Nat.mod_lt r.val hact.mlen_pos)
    show (ram64.get1 r * d0).toInt = _
    exact IntV.I64.mul_nat (hr64 r) hd0 (by
      have : tokOf d i (r.val % mlen d i) * d.K ≤ 2 ^ 22 * 2 ^ 17 :=
        Nat.mul_le_mul (by omega) hK
      omega)
  have hy : ∀ (r : Fin 32) (kk : Fin (bk d)), (((colK d i).row 32).get2 r kk).toInt =
      (((part d i * bk d + kk) * 1 : Nat) : Int) := fun r kk => offsetK_lane kk
  have := ptr_seq2 d.inp (base := d.inp.ptr 0) (e := 0) rfl hx hy r kk
  rw [this]
  congr 2
  unfold aIdx
  ring

/-- `ptr_seq2` for the weights tile at the start of the loop (lines 247-252). -/
theorem bPtr_zero {d : Inst} (hd : d.ok) {i : Nat} (hact : Active d i) (hi : i < (launch d).size)
    {rbn : Tensor [16 - 0] I32}
    (hrbn : ∀ n : Fin (16 - 0), (rbn.get1 n).toInt = ((16 * tileN d i + n.val : Nat) : Int))
    (kk : Fin (bk d)) (n : Fin 16) :
    ((d.lora (slice d i)).ptr 0 + IntV.ofNat (d.R * d.K) * (IntV.ofInt (idAt d (grpOf d i)) : I32) +
      rbn.row (bk d) * (IntV.ofNat d.K : I32) +
      ((IntV.ofNat (part d i * bk d) : I32) + tl.arange 0 (bk d)).col 16 * (IntV.ofNat 1 : I32)).get2
        kk n = (d.lora (slice d i)).ptr ((bIdx d i kk n 0 : Nat) : Int) := by
  have hl := hact.lora_idx hd
  have hl0 := hact.id_nonneg hd
  have hRK := hd.RK_lt
  have hK := hd.K_top
  have hbk : bk d ≤ 256 := by rcases bk_sk_cases d with h | h <;> omega
  have hp : part d i < sk d := Nat.mod_lt _ (sk_pos d)
  have hsk : sk d ≤ 64 := by rcases bk_sk_cases d with h | h <;> omega
  have hpb : part d i * bk d ≤ 64 * 256 := Nat.mul_le_mul (by omega) hbk
  have hlv : (IntV.ofInt (idAt d (grpOf d i)) : I32).toInt =
      (((idAt d (grpOf d i)).toNat : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (idAt_sfits hd _)]
    exact (Int.toNat_of_nonneg hl0).symm
  have hs1 : (IntV.ofNat (d.R * d.K) * (IntV.ofInt (idAt d (grpOf d i)) : I32)).toInt =
      ((d.R * d.K * (idAt d (grpOf d i)).toNat : Nat) : Int) :=
    IntV.I32.mul_nat (IntV.I32.toInt_ofNat hRK) hlv (by
      have := hl.2
      rw [Nat.mul_comm] at this
      exact this)
  have hbase : (d.lora (slice d i)).ptr 0 + IntV.ofNat (d.R * d.K) * (IntV.ofInt (idAt d (grpOf d i)) : I32)
      = (d.lora (slice d i)).ptr ((d.R * d.K * (idAt d (grpOf d i)).toNat : Nat) : Int) := by
    show ((d.lora (slice d i)).ptr 0).offset _ = _
    rw [hs1, Buf.ptr_offset]
    simp
  have hx : ∀ (kk : Fin (bk d)) (n : Fin 16), ((rbn.row (bk d) * (IntV.ofNat d.K : I32)).get2 kk n).toInt =
      (((16 * tileN d i + n.val) * d.K : Nat) : Int) := by
    intro kk n
    show (rbn.get1 n * (IntV.ofNat d.K : I32)).toInt = _
    refine IntV.I32.mul_nat (hrbn n) (IntV.I32.toInt_ofNat (by omega)) ?_
    have := tileN_lt hd hi n.isLt
    calc (16 * tileN d i + n.val) * d.K ≤ d.R * d.K := Nat.mul_le_mul_right _ this.le
      _ < 2 ^ 31 := hRK
  have hy : ∀ (kk : Fin (bk d)) (n : Fin 16),
      ((((IntV.ofNat (part d i * bk d) : I32) + tl.arange 0 (bk d)).col 16 * (IntV.ofNat 1 : I32)).get2
        kk n).toInt = (((part d i * bk d + kk) * 1 : Nat) : Int) := by
    intro kk n
    show (((IntV.ofNat (part d i * bk d) : I32) + (tl.arange 0 (bk d)).get1 kk) * (IntV.ofNat 1 : I32)).toInt = _
    exact IntV.I32.mul_nat (IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega))
      (IntV.I32.toInt_tl_arange (n := bk d) (by omega) kk) (by omega)) (IntV.I32.toInt_ofNat (by norm_num))
      (by omega)
  have := ptr_seq2 (d.lora (slice d i)) hbase hx hy kk n
  rw [this]
  congr 2
  unfold bIdx
  ring

end Claims.LoraShrink
