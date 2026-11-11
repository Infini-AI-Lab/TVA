import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.GeoBound
import Proof.GeoSum
import Proof.LaneArith
import Proof.Tail
import Proof.Util

/-!
# The program is safe

The model's body first computes the lane's four `int64` scalars (the chunk, the sequence's base,
the number of positions, the head) and then runs the loop tail (`tail_safe`). Here those first
steps are taken through the safety rules of the library.
-/

namespace Claims.ChunkCumsum

open TVA

/-! ## Reading one table entry -/

/-- A pointer advanced by an integer of value `i` is the buffer's element `i`. -/
theorem addr_ptr_add {d : DType} (b : Buf d) {sg : Bool} {w : Nat} {k : IntV sg w} {i : Int}
    (h : k.toInt = i) : (b.ptr 0 + k).addr = (b.ptr i).addr := by
  rw [Ptr.add_int, h]
  simp [Ptr.offset, Buf.ptr]

/-- A pointer advanced by two integers whose values sum to `i` is the buffer's element `i`. -/
theorem addr_ptr_add2 {d : DType} (b : Buf d) {sg : Bool} {w : Nat} {k : IntV sg w}
    {sg' : Bool} {w' : Nat} {k' : IntV sg' w'} {i : Int} (h : k.toInt + k'.toInt = i) :
    ((b.ptr 0 + k) + k').addr = (b.ptr i).addr := by
  rw [Ptr.add_int, Ptr.add_int, Ptr.offset_offset, h]
  simp [Ptr.offset, Buf.ptr]

/-- A scalar load of an `int32`/`int64` table entry, converted to `int64`, is safe; the rest sees the entry's value. -/
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

/-- The same converted to `int32`, when the entry fits in `int32`. -/
theorem load_conv_i32 {dt : DType} (hh : dt = .i32 ∨ dt = .i64)
    {init : Mem} {b : Buf dt} {v : Nat → dt.Val} (hH : b.Holds init v) {n : Nat} (hn : n < b.len)
    {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {p : Ptr dt} (hp : p.addr = (b.ptr n).addr) (hR : Mem.span dt p.addr ⊆ R)
    (hz : IntV.SFits 32 (entryInt dt (v n)))
    {rest : Tensor [] I32 → Prog α}
    (hrest : ∀ a : Tensor [] I32, a.item.toInt = entryInt dt (v n) → Safe E R W Q init (rest a)) :
    Safe E R W Q init (tl.load1 p >>= fun x => tl.convert dt .i32 (Tensor.scalar x) >>= rest) := by
  rcases hh with h | h
  · subst h
    have hv0 : init.read .i32 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i32) (d' := .i32) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · exact hz
      · rfl
  · subst h
    have hv0 : init.read .i64 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i64) (d' := .i32) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · exact hz
      · rfl

/-! ## The table entries are in the read footprint -/

/-- The first `chunk_indices` entry of the program's row is in its read footprint. -/
theorem R_mem_ci0 {d : Inst} (hvv : d.varlen = true) (i : Nat)
    {sg : Bool} {w : Nat} {k : IntV sg w} (hk : k.toInt = ((2 * row d i : Nat) : Int)) :
    Mem.span d.idxD (d.ci.ptr 0 + k).addr ⊆ R d i := by
  intro x hx
  have hp : (d.ci.ptr 0 + k).addr = (d.ci.ptr ((2 * (row d i : Nat)) : Int)).addr :=
    addr_ptr_add d.ci hk
  rw [hp] at hx
  unfold R extraReads
  simp only [hvv, ite_true, Set.mem_union]
  tauto

/-- The second `chunk_indices` entry of the program's row is in its read footprint. -/
theorem R_mem_ci1 {d : Inst} (hvv : d.varlen = true) (i : Nat)
    {sg : Bool} {w : Nat} {k : IntV sg w} {sg' : Bool} {w' : Nat} {k' : IntV sg' w'}
    (hk : k.toInt + k'.toInt = (((2 * row d i + 1 : Nat)) : Int)) :
    Mem.span d.idxD ((d.ci.ptr 0 + k) + k').addr ⊆ R d i := by
  intro x hx
  have hp : ((d.ci.ptr 0 + k) + k').addr = (d.ci.ptr ((2 * (row d i : Nat) + 1 : Nat))).addr :=
    addr_ptr_add2 d.ci hk
  rw [hp] at hx
  unfold R extraReads
  simp only [hvv, ite_true, Set.mem_union]
  tauto

/-- The sequence's `cu_seqlens` start entry is in the read footprint. -/
theorem R_mem_cu0 {d : Inst} (hvv : d.varlen = true) (i : Nat)
    {sg : Bool} {w : Nat} {k : IntV sg w}
    (hk : k.toInt = (((seqAt d (row d i)).toNat : Nat) : Int)) :
    Mem.span d.idxD (d.cu.ptr 0 + k).addr ⊆ R d i := by
  intro x hx
  have hp : (d.cu.ptr 0 + k).addr = (d.cu.ptr (seqAt d (row d i)).toNat).addr :=
    addr_ptr_add d.cu hk
  rw [hp] at hx
  unfold R extraReads
  simp only [hvv, ite_true, Set.mem_union]
  tauto

/-- The sequence's `cu_seqlens` end entry is in the read footprint. -/
theorem R_mem_cu1 {d : Inst} (hvv : d.varlen = true) (i : Nat)
    {sg : Bool} {w : Nat} {k : IntV sg w} {sg' : Bool} {w' : Nat} {k' : IntV sg' w'}
    (hk : k.toInt + k'.toInt = (((seqAt d (row d i)).toNat + 1 : Nat) : Int)) :
    Mem.span d.idxD ((d.cu.ptr 0 + k) + k').addr ⊆ R d i := by
  intro x hx
  have hp : ((d.cu.ptr 0 + k) + k').addr = (d.cu.ptr ((seqAt d (row d i)).toNat + 1)).addr :=
    addr_ptr_add2 d.cu hk
  rw [hp] at hx
  unfold R extraReads
  simp only [hvv, ite_true, Set.mem_union]
  tauto

/-! ## The first steps -/

/-- **One program** is safe: it reads within `R`, writes within `W` and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {i : Nat} (hi : i < gridSize d) {c : Ctx} (hc : c.pid 0 = i) :
    Safe E (R d i) (W d i) (fun _ m => Q d i m) init ((model d).body c) := by
  have hT : d.varlen = true → TablesOk d := fun hv => (hpre.2.2.2 hv).2.2.2.2
  have hciH : d.varlen = true → d.ci.Holds init d.ciVal := fun hv => (hpre.2.2.2 hv).2.2.2.1
  have hcuH : d.varlen = true → d.cu.Holds init d.cuVal := fun hv => (hpre.2.2.2 hv).2.2.1
  have hi31 : i < 2 ^ 31 := lt_trans hi hd.grid_lt
  have hpid : tl.program_id c 0 = (IntV.ofNat i : I32) := by simp only [tl.program_id, hc]
  rw [body_eq, hpid]
  -- line 92: `program_id(0).to(tl.int64)`
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j
    exact ⟨IntV.ofInt ((IntV.ofNat i : I32).toInt), rfl⟩
  · intro r1 hr1
    have h1 : r1.item = IntV.ofInt ((IntV.ofNat i : I32).toInt) := hr1 ()
    have hv1 : r1.item.toInt = (i : Int) := by
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · have hs := IntV.sfits_toInt (IntV.ofNat i : I32)
        simp only [IntV.SFits] at hs ⊢
        omega
      · exact IntV.I32.toInt_ofNat hi31
    -- line 96: `% (B * H)`
    refine safe_irem1 (x := r1.item) (y := (IntV.ofNat (d.B * d.H) : I64)) ?_ ?_ ?_
    · rw [hv1]; omega
    · rw [IntV.I64.toInt_ofNat (BH_lt_two_63 hd)]
      have := Nat.mul_pos hd.B_pos hd.H_pos
      omega
    · intro r2 hr2
      have hv2 : r2.item.toInt = (lin d i : Int) := by
        have hmod : r2.item.toInt = (i : Int) % ((d.B * d.H : Nat) : Int) := by
          rw [hr2, hv1, IntV.I64.toInt_ofNat (BH_lt_two_63 hd)]
        rw [hmod]
        unfold lin BH
        rw [Int.natCast_mod]
      -- line 97: `// (B * H)`
      refine safe_idiv1 (x := r1.item) (y := (IntV.ofNat (d.B * d.H) : I64)) ?_ ?_ ?_
      · rw [hv1]; omega
      · rw [IntV.I64.toInt_ofNat (BH_lt_two_63 hd)]
        have := Nat.mul_pos hd.B_pos hd.H_pos
        omega
      · intro r3 hr3
        have hv3 : r3.item.toInt = (row d i : Int) := by
          have hdiv : r3.item.toInt = (i : Int) / ((d.B * d.H : Nat) : Int) := by
            rw [hr3, hv1, IntV.I64.toInt_ofNat (BH_lt_two_63 hd)]
          rw [hdiv]
          unfold row BH
          rw [Int.natCast_div]
        -- line 98: `i_b = i_bh // H`
        refine safe_idiv1 (x := r2.item) (y := (IntV.ofNat d.H : I64)) ?_ ?_ ?_
        · rw [hv2]; omega
        · rw [IntV.I64.toInt_ofNat (by have := H_lt_two_31 hd; omega)]
          exact_mod_cast hd.H_pos
        · intro r4 hr4
          have hv4 : r4.item.toInt = (batchOf d i : Int) := by
            have hdiv : r4.item.toInt = (lin d i : Int) / ((d.H : Nat) : Int) := by
              rw [hr4, hv2, IntV.I64.toInt_ofNat (by have := H_lt_two_31 hd; omega)]
            rw [hdiv]
            unfold batchOf
            rw [Int.natCast_div]
          -- line 99: `i_h = i_bh % H`
          refine safe_irem1 (x := r2.item) (y := (IntV.ofNat d.H : I64)) ?_ ?_ ?_
          · rw [hv2]; omega
          · rw [IntV.I64.toInt_ofNat (by have := H_lt_two_31 hd; omega)]
            exact_mod_cast hd.H_pos
          · intro r5 hr5
            have hv5 : r5.item.toInt = (headOf d i : Int) := by
              have hmod : r5.item.toInt = (lin d i : Int) % ((d.H : Nat) : Int) := by
                rw [hr5, hv2, IntV.I64.toInt_ofNat (by have := H_lt_two_31 hd; omega)]
              rw [hmod]
              unfold headOf
              rw [Int.natCast_mod]
            by_cases hvv : d.varlen = true
            · -- variable length: four table reads
              simp only [hvv, ite_true]
              have hrNT : row d i < d.NT := by
                have := row_lt hd hi
                simpa [rowsOf, hvv] using this
              obtain ⟨hseq0, hseqN, hchk0, hchklt⟩ := (hT hvv).2.2.2.1 (row d i) hrNT
              have hsN : (seqAt d (row d i)).toNat < d.N := by omega
              have hsN1 : (seqAt d (row d i)).toNat + 1 ≤ d.N := by omega
              have hz1 : IntV.SFits 32 (entryInt d.idxD (d.ciVal (2 * row d i))) := by
                show IntV.SFits 32 (seqAt d (row d i))
                have hN31 : ((d.N : Nat) : Int) < 2 ^ 31 := by
                  exact_mod_cast hd.N_lt hvv
                simp only [IntV.SFits]
                omega
              have hk1 : (r3.item * (IntV.ofNat 2 : I64)).toInt = ((2 * row d i : Nat) : Int) := by
                have h2 : (IntV.ofNat 2 : I64).toInt = ((2 : Nat) : Int) := IntV.I64.toInt_ofNat (by norm_num)
                have hm := IntV.I64.mul_nat (x := row d i) (y := 2) hv3 h2
                  (by have := row_lt_two_31 hd hi; omega)
                simpa [Nat.mul_comm] using hm
              have hk2 : (r3.item * (IntV.ofNat 2 : I64)).toInt + (IntV.ofNat 1 : I32).toInt
                  = ((2 * row d i + 1 : Nat) : Int) := by
                rw [hk1, IntV.I32.toInt_ofNat (by norm_num)]
                push_cast
                ring
              simp only [Prog.bind_eq]
              -- row of `chunk_indices`: the sequence
              refine load_conv_i32 (dt := d.idxD) (b := d.ci) (v := d.ciVal) (n := 2 * row d i)
                (hd.idx_ok hvv) (hciH hvv) (ci_len_le hd hvv (by omega)) ?_ ?_ ?_ ?_
              · exact addr_ptr_add d.ci hk1
              · exact R_mem_ci0 hvv i hk1
              · exact hz1
              · intro a1 ha1
                have hn1 : a1.item.toInt = (((seqAt d (row d i)).toNat : Nat) : Int) := by
                  rw [ha1, Int.toNat_of_nonneg hseq0]
                  rfl
                -- the chunk
                refine load_conv_i64 (dt := d.idxD) (b := d.ci) (v := d.ciVal)
                  (n := 2 * row d i + 1) (hd.idx_ok hvv) (hciH hvv)
                  (ci_len_le hd hvv (by omega)) ?_ ?_ ?_
                · exact addr_ptr_add2 d.ci hk2
                · exact R_mem_ci1 hvv i hk2
                · intro t2 ht2
                  have hchk : t2.item.toInt = (chkOf d i : Int) := by
                    rw [ht2, chkOf_true hvv, Int.toNat_of_nonneg hchk0]
                    rfl
                  -- the base
                  refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal)
                    (n := (seqAt d (row d i)).toNat) (hd.idx_ok hvv) (hcuH hvv)
                    (cu_len_le hd hvv (by omega)) ?_ ?_ ?_
                  · exact addr_ptr_add d.cu hn1
                  · exact R_mem_cu0 hvv i hn1
                  · intro b3 hb3
                    have hbs : b3.item.toInt = (baseOf d i : Int) := by
                      rw [hb3, baseOf_true hvv, nOf_true hvv, Int.toNat_of_nonneg
                        (cuAt_nonneg (hT hvv) (le_of_lt hsN))]
                      rfl
                    -- the end
                    refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal)
                      (n := (seqAt d (row d i)).toNat + 1) (hd.idx_ok hvv) (hcuH hvv)
                      (cu_len_le hd hvv (by omega)) ?_ ?_ ?_
                    · refine addr_ptr_add2 d.cu ?_
                      rw [hn1, IntV.I32.toInt_ofNat (by norm_num)]
                      push_cast
                      ring
                    · refine R_mem_cu1 hvv i ?_
                      rw [hn1, IntV.I32.toInt_ofNat (by norm_num)]
                      push_cast
                      ring
                    · intro e4 he4
                      have hes : e4.item.toInt = (((cuAt d ((seqAt d (row d i)).toNat + 1)).toNat : Nat) : Int) := by
                        rw [he4, Int.toNat_of_nonneg
                          (cuAt_nonneg (hT hvv) (by omega))]
                        rfl
                      have hlen : (e4.item - b3.item).toInt = (lenOf d i : Int) := by
                        rw [IntV.I64.sub_nat (x := (cuAt d ((seqAt d (row d i)).toNat + 1)).toNat)
                          (y := (cuAt d ((seqAt d (row d i)).toNat)).toNat) hes
                          (by rw [hb3, Int.toNat_of_nonneg (cuAt_nonneg (hT hvv) (le_of_lt hsN))]
                              rfl)
                          (by have := cuAt_step (hT hvv) hsN
                              have h2 : (cuAt d ((seqAt d (row d i)).toNat)).toNat
                                  ≤ (cuAt d ((seqAt d (row d i)).toNat + 1)).toNat :=
                                Int.toNat_le_toNat (by omega)
                              exact h2)
                          (by have hq : cuAt d ((seqAt d (row d i)).toNat + 1) ≤ d.T :=
                                le_trans (cuAt_mono_le (hT hvv) hsN1 le_rfl) (hT hvv).2.2.1
                              have h2 := T_lt_two_63 hd
                              omega)]
                        congr 1
                        rw [lenOf, hi_true hvv, lo_true hvv, nOf_true hvv]
                      exact tail_safe hd hpre E hT hi (it := t2.item) (bos := b3.item)
                        (TT := e4.item - b3.item) (ih := r5.item) hchk hbs hlen hv5
            · -- fixed length: the `T` conversion
              have hvf : d.varlen = false := by cases h : d.varlen <;> simp_all
              simp only [hvf, Bool.false_eq_true, ite_false]
              simp only [Prog.bind_eq]
              refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
              · intro j
                exact ⟨IntV.ofInt ((IntV.ofNat d.T : I32).toInt), rfl⟩
              · intro t64 ht64
                have ht1 : t64.item = IntV.ofInt ((IntV.ofNat d.T : I32).toInt) := ht64 ()
                have hT64 : t64.item.toInt = (d.T : Int) := by
                  rw [ht1]
                  refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
                  · have hs := IntV.sfits_toInt (IntV.ofNat d.T : I32)
                    simp only [IntV.SFits] at hs ⊢
                    omega
                  · exact IntV.I32.toInt_ofNat hd.T_lt
                have hchk : r3.item.toInt = (chkOf d i : Int) := by
                  rw [chkOf_false hvf]
                  exact hv3
                have hbs : (r4.item * t64.item).toInt = (baseOf d i : Int) := by
                  have hm := IntV.I64.mul_nat (x := batchOf d i) (y := d.T) hv4 hT64 (by
                    have h1 : batchOf d i < d.B := batchOf_lt hd i
                    have h2 : d.T ≤ 2 ^ 22 := hd.T_le
                    have h3 : d.B ≤ 8192 := hd.B_le
                    have h4 : batchOf d i * d.T ≤ d.B * 2 ^ 22 := Nat.mul_le_mul h1.le h2
                    have h5 : d.B * 2 ^ 22 ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul_right _ h3
                    omega)
                  rw [hm, baseOf_false hvf, nOf_false hvf]
                have hlen : t64.item.toInt = (lenOf d i : Int) := by
                  rw [hT64]
                  simp only [lenOf, hi_false hvf, lo_false hvf, nOf_false hvf]
                  congr 1
                  omega
                exact tail_safe hd hpre E hT hi (it := r3.item) (bos := r4.item * t64.item)
                  (TT := t64.item) (ih := r5.item) hchk hbs hlen hv5

end Claims.ChunkCumsum
