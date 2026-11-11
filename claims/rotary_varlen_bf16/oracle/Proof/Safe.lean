import Proof.Common
import Proof.Basic
import Proof.Meta
import Proof.Body
import Proof.State
import Proof.Ocs
import Proof.TailNE
import Proof.TailI

/-!
# One program is safe

`body_eq` decomposes a program into the prologue (metadata, no memory writes), the offset line,
and one of the two tails. This file glues the three pieces: the prologue's state `StOk` feeds the
offset line, the offset line's `o_cs` feeds the tail, and the early return before the offset line
returns on an empty chunk, where `Q` holds vacuously.
-/

namespace Claims.RotaryVarlen

open TVA

/-- The product of a program's chunk and `BT` is far below the `I64` range: it is a token index of
the launch. -/
private theorem chunk_mul_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) : chunkI d init i * d.BT < 2 ^ 62 := by
  have hBT := bt_le hd
  by_cases hv : d.isVarlen = true
  · have hs := hpre.seqs hv
    have hax : ax0 d i < d.NT := by
      have h := ax0_lt hi
      simpa [g0, hv] using h
    obtain ⟨hseqN, hlt⟩ := hs.ci_ok (ax0 d i) hax
    have hseqT : seqLen d init (chSeq d init (ax0 d i)) ≤ d.T := by
      have h1 : cuN d init (chSeq d init (ax0 d i)) ≤ cuN d init (chSeq d init (ax0 d i) + 1) :=
        hs.cu_mono _ hseqN
      have h2 : cuN d init (chSeq d init (ax0 d i) + 1) ≤ cuN d init d.N :=
        cuN_mono_le hs (by omega) le_rfl
      have h3 : cuN d init d.N = d.T := hs.cu_last
      simp only [seqLen]
      omega
    have hc : chunkI d init i = chChunk d init (ax0 d i) := chunkI_true hv init i
    have ht := hd.t_lt
    have h3 : (2 : Nat) ^ 31 < 2 ^ 62 := by norm_num
    rw [hc]
    omega
  · have hv2 : d.isVarlen = false := by cases h : d.isVarlen <;> simp_all
    have hc : chunkI d init i = ax0 d i := chunkI_false hv2 init i
    have h1 : chunkI d init i < 2 ^ 31 := by
      have h := ax0_lt hi
      have hg : g0 d < 2 ^ 31 := hd.grid_t
      rw [hc]
      omega
    have hle : chunkI d init i * d.BT ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul h1.le hBT
    have h2 : (2 : Nat) ^ 31 * 2 ^ 20 = 2 ^ 51 := by norm_num
    have h3 : (2 : Nat) ^ 51 < 2 ^ 62 := by norm_num
    omega

/-- The `I64` value of `BT`. -/
private theorem ofNat_bt_toInt {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.BT : I64).toInt = (d.BT : Int) := by
  refine ofNat_toInt  ?_
  have h := bt_le hd
  have h1 : (2 : Nat) ^ 20 ≤ 2 ^ 63 := by norm_num
  have h2 : d.BT < 2 ^ 63 := by omega
  have h3 : (2 : Int) ^ (64 - 1) = (2 : Int) ^ 63 := by norm_num
  rw [h3]
  exact_mod_cast h2

/-- The chunk and `BT` product, as an `Int`, fits `I64`. -/
private theorem chunk_mul_lt_int {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 63 := by
  have h := chunk_mul_lt hd hpre hi
  have h1 : ((chunkI d init i * d.BT : Nat) : Int) < (2 : Int) ^ 62 := by exact_mod_cast h
  have h2 : (2 : Int) ^ 62 < (2 : Int) ^ 63 := by norm_num
  push_cast at h1
  omega

/-- **One program.** Program `i` of the grid, run alone from `init`, is safe: it reads `R`, writes
`W`, and leaves `Q` there. -/
theorem safe_program (P : Platform) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq d i]
  refine safe_prologue (P := P) (W := W d init i) (Q := Q d init i) hd hpre hi ?_
  intro st hst
  obtain ⟨i_t, x_h, y_h, T_n, i_n⟩ := st
  obtain ⟨hit, hT, hin, hx, hy⟩ := hst
  dsimp only at hit hT hin hx hy ⊢
  by_cases hc : IntV.le T_n (i_t * (IntV.ofNat d.BT : I64)) = true
  · rw [ite_eq_left hc]
    refine safe_ret (a := ()) ?_
    intro l c hl hcR hlan
    exfalso
    have hcle : T_n.toInt ≤ (i_t * (IntV.ofNat d.BT : I64)).toInt := by
      have h := hc
      rw [IntV.le_eq] at h
      exact of_decide_eq_true h
    have hbT := ofNat_bt_toInt hd
    have hX : (i_t * (IntV.ofNat d.BT : I64)).toInt
        = (chunkI d init i : Int) * (d.BT : Int) := by
      have hfits : IntV.SFits 64 (i_t.toInt * (IntV.ofNat d.BT : I64).toInt) := by
        rw [hit, hbT]
        have hlt := chunk_mul_lt_int hd hpre hi
        simp only [IntV.SFits]
        constructor
        · have hnn : (0 : Int) ≤ (chunkI d init i : Int) * (d.BT : Int) := by positivity
          have h2 : (0 : Int) < 2 ^ 63 := by norm_num
          omega
        · exact chunk_mul_lt_int hd hpre hi
      rw [IntV.toInt_mul rfl rfl hfits, hit, hbT]
    have hloc : (locI d init i l : Int) = (chunkI d init i : Int) * (d.BT : Int) + (l : Int) := by
      simp only [locI]
      push_cast
      ring
    have hlan1 : (locI d init i l : Int) < (lenI d init i : Int) := hlan.1
    omega
  · rw [ite_eq_right hc]
    refine safe_ocs (P := P) (W := W d init i) (Q := Q d init i) hd hpre hi hit hin ?_
    intro o_cs hocs
    by_cases hb : d.interleaved = true
    · rw [ite_eq_right (by simp [hb])]
      exact safe_tail_i (P := P) hd hpre hi hit hT hx hy hocs hb
    · have hbf : d.interleaved = false := by cases h : d.interleaved <;> simp_all
      rw [ite_eq_left (by simp [hbf])]
      exact safe_tail_ne (P := P) hd hpre hi hit hT hx hy hocs hbf

end Claims.RotaryVarlen
