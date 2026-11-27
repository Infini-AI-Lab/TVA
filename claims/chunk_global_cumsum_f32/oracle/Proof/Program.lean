import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.TripLanes
import Proof.Loop

/-!
# One program

`safe_program` walks the prologue (source lines 74–79: the program's sequence and head, the
sequence's bounds from `cu_seqlens` or the rows), then `safe_tail` the rest (lines 80–101: the
length, the block count `NT = nb`, the loop).
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst} {i : Nat}

/-- A pointer advanced by an integer of value `i` is the buffer's element `i`. -/
theorem addr_ptr_add {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {k : IntV sg w} {j : Int}
    (h : k.toInt = j) : (b.ptr 0 + k).addr = (b.ptr j).addr := by
  rw [Ptr.add_int, h]
  simp [Ptr.offset, Buf.ptr]

/-- A pointer advanced by two integers whose values sum to `i` is the buffer's element `i`. -/
theorem addr_ptr_add2 {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {k : IntV sg w}
    {sg' : Bool} {w' : Nat} {k' : IntV sg' w'} {j : Int} (h : k.toInt + k'.toInt = j) :
    ((b.ptr 0 + k) + k').addr = (b.ptr j).addr := by
  rw [Ptr.add_int, Ptr.add_int, Ptr.offset_offset, h]
  simp [Ptr.offset, Buf.ptr]

/-- A scalar load of an `int32`/`int64` table entry, converted to `int64`, is safe; the rest sees
the entry's value. -/
theorem load_conv_i64 {dt : DType} (hh : dt = .i32 ∨ dt = .i64)
    {init : Mem} {b : Buf dt} {v : Nat → dt.Val} (hH : b.Holds init v) {n : Nat} (hn : n < b.len)
    {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {p : Ptr dt} (hp : p.addr = (b.ptr n).addr) (hR : Mem.span dt p.addr ⊆ R)
    {rest : Tensor [] I64 → Prog α}
    (hrest : ∀ a : Tensor [] I64, a.item.toInt = entryInt dt (v n) → Safe E R W Q init (rest a)) :
    Safe E R W Q init (tl.load1 p >>= fun x => tl.convert dt .i64 (Tensor.scalar x) >>= rest) := by
  rcases hh with h | h
  · subst h
    have hv0 : init.read .i32 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · have hs := IntV.sfits_toInt (v n)
        simp only [IntV.SFits] at hs ⊢
        omega
      · rfl
  · subst h
    have hv0 : init.read .i64 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i64) (d' := .i64) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · have hs := IntV.sfits_toInt (v n)
        simp only [IntV.SFits] at hs ⊢
        omega
      · rfl

/-- **The rest of the body** from the sequence's bounds. -/
theorem safe_tail (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform} (hP : Device P)
    (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {bos eos ih : I64}
    (hbos : bos.toInt = (loOf d i : Int)) (heos : eos.toInt = (hiOf d i : Int))
    (hih : ih.toInt = (headOf d i : Int)) :
    Safe ((launch d).env P) (R d) (W d i) (fun _ m => Q d i m) init (tail d bos eos ih) := by
  have hle := loOf_le_hiOf hT hi
  have hhi := hiOf_le hd hT hi
  have hBT := BT_le hd
  have hbt := hd.bt_le
  have hb := bt_pos hd
  have hTT : (eos - bos).toInt = (lenOf d i : Int) :=
    IntV.I64.sub_nat heos hbos hle (by omega)
  have hlen := lenOf_le hT hi
  have hT22 := hd.T_le
  have hx : (eos - bos + (IntV.ofNat (d.BT - 1) : I64)).toInt = ((lenOf d i + (d.BT - 1) : Nat) : Int) :=
    IntV.I64.add_nat hTT (IntV.I64.toInt_ofNat (by omega)) (by omega)
  have hy : (IntV.ofNat d.BT : I64).toInt = (d.BT : Int) := IntV.I64.toInt_ofNat (by omega)
  unfold tail
  refine safe_idiv1 (by rw [hx]; omega) (by rw [hy]; omega) ?_
  intro r hr
  have hNT : r.item.toInt = (nbOf d i : Int) := by
    rw [hr, hx, hy, nbOf_eq, ← Int.natCast_div]
    congr 2
    omega
  exact safe_loop hd hpre hP hT hi hNT hTT hbos hih

/-- **One program** is safe: it reads within `R`, writes within `W` and ends with `Q`. -/
theorem safe_program (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform} (hP : Device P)
    (hi : i < gridSize d) {c : Ctx} (hc : c.pid 0 = i) :
    Safe ((launch d).env P) (R d) (W d i) (fun _ m => Q d i m) init ((model d).body c) := by
  have hT : d.varlen = true → TablesOk d := fun hv => (hpre.2.2.2 hv).2.2
  have hcuH : d.varlen = true → d.cu.Holds init d.cuVal := fun hv => (hpre.2.2.2 hv).2.1
  have hi31 : i < 2 ^ 31 := lt_trans hi (grid_lt hd)
  have hH17 := hd.H_le
  have hHpos := hd.H_pos
  have hpid : tl.program_id c 0 = (IntV.ofNat i : I32) := by simp only [tl.program_id, hc]
  have hH : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := IntV.I64.toInt_ofNat (by omega)
  have hn := seqOf_lt hi
  rw [body_eq, hpid]
  -- line 74: `program_id(0).to(tl.int64)`
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j
    exact ⟨IntV.ofInt ((IntV.ofNat i : I32).toInt), rfl⟩
  intro r1 hr1
  have hv1 : r1.item.toInt = (i : Int) := by
    have h1 : r1.item = IntV.ofInt ((IntV.ofNat i : I32).toInt) := hr1 ()
    rw [h1]
    refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
    · have hs := IntV.sfits_toInt (IntV.ofNat i : I32)
      simp only [IntV.SFits] at hs ⊢
      omega
    · exact IntV.I32.toInt_ofNat hi31
  -- line 75: `i_nh // H`, `i_nh % H`
  refine safe_idiv1 (x := r1.item) (y := (IntV.ofNat d.H : I64)) (by rw [hv1]; omega)
    (by rw [hH]; omega) ?_
  intro r2 hr2
  have hv2 : r2.item.toInt = (seqOf d i : Int) := by
    rw [hr2, hv1, hH]; unfold seqOf; rw [Int.natCast_div]
  refine safe_irem1 (x := r1.item) (y := (IntV.ofNat d.H : I64)) (by rw [hv1]; omega)
    (by rw [hH]; omega) ?_
  intro r3 hr3
  have hv3 : r3.item.toInt = (headOf d i : Int) := by
    rw [hr3, hv1, hH]; unfold headOf; rw [Int.natCast_mod]
  by_cases hvv : d.varlen = true
  · -- variable length: the two table entries
    simp only [hvv, ite_true]
    simp only [Prog.bind_eq]
    rw [nSeq_true hvv] at hn
    have hcu := hd.cu_len hvv
    have hR : ∀ k, k < d.cu.len → Mem.span d.idxD (d.cu.ptr k).addr ⊆ R d := by
      intro k hk x hx
      exact Or.inr (by rw [ite_eq_left hvv]; exact d.cu.span_sub hk hx)
    refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal) (n := seqOf d i)
      (hd.idx_ok hvv) (hcuH hvv) (by omega) (addr_ptr_add d.cu hv2) (by
        rw [addr_ptr_add d.cu hv2]; exact hR _ (by omega)) ?_
    intro b3 hb3
    have h2 : r2.item.toInt + (IntV.ofNat 1 : I32).toInt = ((seqOf d i + 1 : Nat) : Int) := by
      rw [hv2, IntV.I32.toInt_ofNat (by norm_num)]; push_cast; ring
    refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal) (n := seqOf d i + 1)
      (hd.idx_ok hvv) (hcuH hvv) (by omega) (addr_ptr_add2 d.cu h2) (by
        rw [addr_ptr_add2 d.cu h2]; exact hR _ (by omega)) ?_
    intro e4 he4
    have hbos : b3.item.toInt = (loOf d i : Int) := by
      rw [hb3]; unfold loOf; rw [lo_true hvv, Int.toNat_of_nonneg (cuAt_nonneg (hT hvv) (by omega))]
      rfl
    have heos : e4.item.toInt = (hiOf d i : Int) := by
      rw [he4]; unfold hiOf; rw [hi_true hvv, Int.toNat_of_nonneg (cuAt_nonneg (hT hvv) (by omega))]
      rfl
    exact safe_tail hd hpre hP hT hi hbos heos hv3
  · -- fixed length: the `T` conversion
    have hvf : d.varlen = false := by simpa using hvv
    simp only [hvf, Bool.false_eq_true, ite_false]
    simp only [Prog.bind_eq]
    rw [nSeq_false hvf] at hn
    refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((IntV.ofNat d.T : I32).toInt), rfl⟩
    intro t64 ht64
    have hT64 : t64.item.toInt = (d.T : Int) := by
      have ht1 : t64.item = IntV.ofInt ((IntV.ofNat d.T : I32).toInt) := ht64 ()
      rw [ht1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · have hs := IntV.sfits_toInt (IntV.ofNat d.T : I32)
        simp only [IntV.SFits] at hs ⊢
        omega
      · exact IntV.I32.toInt_ofNat hd.T_lt
    have hBT := BT_le hd
    have hnT : seqOf d i * d.T + d.T ≤ d.B * d.T := by
      have : (seqOf d i + 1) * d.T ≤ d.B * d.T := Nat.mul_le_mul_right _ hn
      have e : (seqOf d i + 1) * d.T = seqOf d i * d.T + d.T := by ring
      omega
    have hbos : (r2.item * t64.item).toInt = (loOf d i : Int) := by
      rw [IntV.I64.mul_nat hv2 hT64 (by omega)]
      unfold loOf; rw [lo_false hvf]
    have heos : (r2.item * t64.item + t64.item).toInt = (hiOf d i : Int) := by
      rw [IntV.I64.add_nat (IntV.I64.mul_nat hv2 hT64 (by omega)) hT64 (by omega)]
      unfold hiOf; rw [hi_false hvf]
    exact safe_tail hd hpre hP hT hi hbos heos hv3

end Claims.ChunkGlobalCumsum
