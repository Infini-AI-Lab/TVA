import Proof.Common
import Proof.Basic
import Proof.Meta

/-!
# The table position of the program's tokens

`o_cs`, the table row each of the program's `BT` lanes reads: the token's position, read from the
offset tensor (`offT`) or the one `int` offset, and that it is `posI`.
-/

namespace Claims.RotaryVarlen

open TVA

private theorem exists_convRel_i64 {P : Platform} {dt : DType}
    (hdt : dt = .i32 ∨ dt = .i64) (x : dt.Val) :
    ∃ r : I64, ConvRel P dt .i64 x r := by
  rcases hdt with rfl | rfl
  · exact ⟨IntV.ofInt x.toInt, rfl⟩
  · exact ⟨IntV.ofInt x.toInt, rfl⟩

private theorem convRel_i64_idxVal {P : Platform} {dt : DType}
    (hdt : dt = .i32 ∨ dt = .i64) {x : dt.Val} {r : I64}
    (h : ConvRel P dt .i64 x r) : r.toInt = idxVal dt x := by
  rcases hdt with rfl | rfl
  · have h' : r = IntV.ofInt x.toInt := h
    rw [h']
    show (IntV.ofInt x.toInt : I64).toInt = x.toInt
    exact IntV.toInt_ofInt (w := 64) (by norm_num) (sfits64_of_sfits32 (IntV.sfits_toInt x))
  · have h' : r = IntV.ofInt x.toInt := h
    rw [h']
    show (IntV.ofInt x.toInt : I64).toInt = x.toInt
    exact IntV.toInt_ofInt (w := 64) (by norm_num) (IntV.sfits_toInt x)

/-- The `o_cs` line of `body_eq` reads position `posI` of every lane. -/
theorem safe_ocs {P : Platform} {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) {i_t : I64}
    (hit : i_t.toInt = (chunkI d init i : Int))
    {i_n : I32} (hin : d.isVarlen = true → i_n.toInt = (seqI d init i : Int))
    {W : Set Int} {Q : Mem → Prop} {f : Tensor [d.BT] I64 → Prog Unit}
    (hk : ∀ o_cs : Tensor [d.BT] I64,
      (∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val) →
      Safe ((launch d).env P) (R d init i) W (fun _ m => Q m) init (f o_cs)) :
    Safe ((launch d).env P) (R d init i) W (fun _ m => Q m) init
      (do
        let o_t : Tensor [d.BT] I64 := i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)
        let o_cs : Tensor [d.BT] I64 ← (if !d.offT then
            pure (o_t + (IntV.cast (IntV.ofNat d.off : I32) : I64))
          else (do
            let v_o ← tl.load1 (if d.isVarlen then d.offs.ptr 0 + i_n else d.offs.ptr 0 + (IntV.cast (IntV.ofNat (ax1 d i) : I32) : I64))
            let off_n : I64 := (← tl.convert d.offD .i64 (Tensor.scalar v_o)).item
            pure (o_t + off_n) ) : Prog (Tensor [d.BT] I64))
        f o_cs) := by
  by_cases hoff : d.offT = true
  · have hb : (!d.offT) = false := by simp [hoff]
    simp only [hb, Bool.false_eq_true, ite_false, Prog.bind_eq]
    simp only [Prog.bind_assoc]
    set n := seqI d init i with hn
    have hnlt : n < nSeq d := seqI_lt hd hpre hi
    have hnlen : n < d.offs.len := lt_of_lt_of_le hnlt (hd.offs_len hoff)
    have hvalid : d.offs.Valid init := (hpre.offs_vals hoff).1
    obtain ⟨v, hidx, hv_low, hv_high⟩ := (hpre.offs_vals hoff).2 n hnlt
    have hv_ctx : v ≤ 2 ^ 22 := hpre.offs_ctx hoff n hnlt v hidx
    have hdt : d.offD = .i32 ∨ d.offD = .i64 := hd.offD_ok hoff
    have hR : Mem.span d.offD (d.offs.ptr n).addr ⊆ R d init i := by
      intro x hx
      exact Set.mem_union_right _ (Buf.span_sub d.offs hnlen hx)
    have haddr : (if d.isVarlen = true then d.offs.ptr 0 + i_n
        else d.offs.ptr 0 + (IntV.cast (IntV.ofNat (ax1 d i) : I32) : I64)) = d.offs.ptr n := by
      by_cases hv : d.isVarlen = true
      · rw [ite_eq_left hv]
        have hin' : i_n.toInt = (n : Int) := by rw [hn]; exact hin hv
        rw [Ptr.mk.injEq]
        simp only [Ptr.add_int, Ptr.offset_addr, Buf.ptr_addr, hin']
        ring
      · rw [ite_eq_right hv]
        have hv2 : d.isVarlen = false := by simpa using hv
        have hn' : n = ax1 d i := by rw [hn, seqI_false hv2]
        have hax : ax1 d i < 2 ^ 31 := by
          have h1 : ax1 d i < d.B := ax1_lt hi
          have h2 : d.B ≤ 8192 := hd.b_le
          omega
        have hcast : (IntV.cast (IntV.ofNat (ax1 d i) : I32) : I64).toInt = (ax1 d i : Int) := by
          rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num)]
          exact IntV.I32.toInt_ofNat hax
        rw [Ptr.mk.injEq]
        simp only [Ptr.add_int, Ptr.offset_addr, Buf.ptr_addr, hcast, hn']
        ring
    rw [haddr]
    refine safe_load1_buf (b := d.offs) (i := n) hnlen hvalid hR ?_
    intro v0 hv0
    have hidx0 : idxAt d.offs init n = some (idxVal d.offD v0) := by
      simp only [idxAt, hv0, Option.map_some]
    have hv0val : idxVal d.offD v0 = v := by
      rw [hidx0] at hidx
      exact Option.some.inj hidx
    refine safe_convert (d := d.offD) (d' := .i64) (x := Tensor.scalar v0) ?_ ?_
    · intro idx
      exact exists_convRel_i64 hdt v0
    · intro r hr
      simp only [Prog.pure_eq, Prog.ret_bind]
      refine hk (i_t * IntV.ofNat d.BT + (tl.arange 0 d.BT).castInt + r.item) ?_
      intro l
      have hl : l.val < d.BT := l.isLt
      have hBTle : d.BT ≤ 2 ^ 20 := bt_le hd
      have hcbt : chunkI d init i * d.BT < 2 ^ 31 := chunkI_mul_BT_lt hd hpre hi
      have hl31 : (l.val : Int) < 2 ^ 31 := by
        have : (l.val : Int) < (d.BT : Int) := by exact_mod_cast hl
        have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast hBTle
        linarith
      have hro : r.item.toInt = v := by
        have hconv : ConvRel P d.offD .i64 ((Tensor.scalar v0).get ()) ((r.get ())) := hr ()
        have hmain := convRel_i64_idxVal hdt hconv
        simp only [Tensor.scalar] at hmain
        rw [hv0val] at hmain
        exact hmain
      have h1 : IntV.SFits 64 (i_t.toInt * (d.BT : Int)) := by
        rw [hit]
        refine sfits64_of_bounds ?_ ?_
        · nlinarith
        · linarith
      have h2 : IntV.SFits 64 (i_t.toInt * (d.BT : Int) + (0 + (l.val : Int))) := by
        rw [hit]
        refine sfits64_of_bounds ?_ ?_
        · nlinarith
        · push_cast
          nlinarith
      have h3 : IntV.SFits 64 (i_t.toInt * (d.BT : Int) + (0 + (l.val : Int)) + r.item.toInt) := by
        rw [hit, hro]
        refine sfits64_of_bounds ?_ ?_
        · nlinarith
        · push_cast
          nlinarith
      have hBT63 : (d.BT : Int) < 2 ^ 63 := by
        have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast hBTle
        linarith
      have hiv : (0 + (l.val : Int)) < 2 ^ 31 := by
        push_cast
        linarith
      have hmain := toInt_split (n := d.BT) (i_t := i_t) (i := (l, ()))
        (off := r.item) hBT63 hiv h1 h2 h3
      refine hmain.trans ?_
      rw [hro, hit]
      simp only [posI, locI, offI]
      have hoffAt : offAt d init (seqI d init i) = v := by
        rw [← hn]
        simp only [offAt, hoff, ite_true]
        rw [hidx]
        rfl
      rw [hoffAt]
      push_cast
      ring
  · have hoffF : d.offT = false := by simpa using hoff
    have hb : (!d.offT) = true := by simp [hoffF]
    simp only [hb, ite_true, Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
    refine hk _ ?_
    intro l
    have hl : l.val < d.BT := l.isLt
    have hBTpos : 0 < d.BT := bt_pos hd
    have hBTle : d.BT ≤ 2 ^ 20 := bt_le hd
    have hcbt : chunkI d init i * d.BT < 2 ^ 31 := chunkI_mul_BT_lt hd hpre hi
    have hcbt_int : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 31 := by
      have h2 : ((chunkI d init i * d.BT : Nat) : Int) < (2 ^ 31 : Int) := by exact_mod_cast hcbt
      push_cast at h2 ⊢
      linarith
    have hc0 : (0 : Int) ≤ (chunkI d init i : Int) := by exact_mod_cast Nat.zero_le _
    have hl0 : (0 : Int) ≤ (l.val : Int) := by exact_mod_cast Nat.zero_le _
    have hl31 : (l.val : Int) < 2 ^ 31 := by
      have : (l.val : Int) < (d.BT : Int) := by exact_mod_cast hl
      have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast hBTle
      linarith
    have hoffv : ((IntV.cast (IntV.ofNat d.off : I32) : I64)).toInt = (d.off : Int) := by
      rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num)]
      exact IntV.I32.toInt_ofNat hd.off_lt
    have hoff31 : (d.off : Int) < 2 ^ 31 := by exact_mod_cast hd.off_lt
    have hoff0 : (0 : Int) ≤ (d.off : Int) := by exact_mod_cast Nat.zero_le _
    have h1 : IntV.SFits 64 (i_t.toInt * (d.BT : Int)) := by
      rw [hit]
      refine sfits64_of_bounds ?_ ?_
      · nlinarith
      · linarith
    have h2 : IntV.SFits 64 (i_t.toInt * (d.BT : Int) + (0 + (l.val : Int))) := by
      rw [hit]
      refine sfits64_of_bounds ?_ ?_
      · nlinarith
      · push_cast
        nlinarith
    have h3 : IntV.SFits 64 (i_t.toInt * (d.BT : Int) + (0 + (l.val : Int)) +
        ((IntV.cast (IntV.ofNat d.off : I32) : I64)).toInt) := by
      rw [hit, hoffv]
      refine sfits64_of_bounds ?_ ?_
      · nlinarith
      · push_cast
        nlinarith
    have hBT63 : (d.BT : Int) < 2 ^ 63 := by
      have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast hBTle
      linarith
    have hiv : (0 + (l.val : Int)) < 2 ^ 31 := by
      push_cast
      linarith
    have hmain := toInt_split (n := d.BT) (i_t := i_t) (i := (l, ()))
      (off := (IntV.cast (IntV.ofNat d.off : I32) : I64)) hBT63 hiv h1 h2 h3
    refine hmain.trans ?_
    rw [hoffv, hit]
    simp only [posI, locI, offI]
    have hoffAt : offAt d init (seqI d init i) = (d.off : Int) := by simp [offAt, hoffF]
    rw [hoffAt]
    push_cast
    ring

end Claims.RotaryVarlen
