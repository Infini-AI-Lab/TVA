import Claim
import TVA.Proof
import Proof.Tail
import Proof.Post

/-!
# The prologue: the chunk, the sequence's start and length, the value head

The first lines of the body (99–106) compute four `int64` scalars from the program ids and, on a
variable-length launch, four entries of the two tables. `safe_prologue` walks them and hands the
values (`chkOf`, `baseOf`, `lenOf`, `hvOf`) to any proof about the tail.
-/

namespace Claims.ChunkBwdDvLocal

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

/-- An entry of `chunk_indices` is in the read footprint (variable length). -/
theorem span_ci_sub_R {d : Inst} (hv : d.varlen = true) {r bh n : Nat} (hn : n < d.ci.len) :
    Mem.span d.idxD (d.ci.ptr n).addr ⊆ R d r bh := by
  intro x hx
  have := Buf.span_sub d.ci hn hx
  unfold R
  simp only [hv, ite_true, Set.mem_union]
  tauto

/-- An entry of `cu_seqlens` is in the read footprint (variable length). -/
theorem span_cu_sub_R {d : Inst} (hv : d.varlen = true) {r bh n : Nat} (hn : n < d.cu.len) :
    Mem.span d.idxD (d.cu.ptr n).addr ⊆ R d r bh := by
  intro x hx
  have := Buf.span_sub d.cu hn hx
  unfold R
  simp only [hv, ite_true, Set.mem_union]
  tauto

/-- On a variable-length launch, program `(r, bh)`'s first token is `cu_seqlens[n]`. -/
theorem baseOf_true' {d : Inst} (h : d.varlen = true) (r bh : Nat) :
    baseOf d r bh = (cuAt d (nOf d r bh)).toNat := by
  simp [baseOf, lo_true h]

/-- The `chunk_indices` entry at a pointer advanced by `k` is in the read footprint. -/
theorem span_ci_add {d : Inst} (hv : d.varlen = true) {r bh : Nat} {sg : Bool} {w : Nat}
    {k : IntV sg w} {i : Nat} (hk : k.toInt = (i : Int)) (hi : i < d.ci.len) :
    Mem.span d.idxD (d.ci.ptr 0 + k).addr ⊆ R d r bh := by
  rw [addr_ptr_add d.ci hk]; exact span_ci_sub_R hv hi

/-- The `chunk_indices` entry at a pointer advanced by `k` and `k'` is in the read footprint. -/
theorem span_ci_add2 {d : Inst} (hv : d.varlen = true) {r bh : Nat} {sg sg' : Bool} {w w' : Nat}
    {k : IntV sg w} {k' : IntV sg' w'} {i : Nat} (hk : k.toInt + k'.toInt = (i : Int))
    (hi : i < d.ci.len) : Mem.span d.idxD ((d.ci.ptr 0 + k) + k').addr ⊆ R d r bh := by
  rw [addr_ptr_add2 d.ci hk]; exact span_ci_sub_R hv hi

/-- The `cu_seqlens` entry at a pointer advanced by `k` is in the read footprint. -/
theorem span_cu_add {d : Inst} (hv : d.varlen = true) {r bh : Nat} {sg : Bool} {w : Nat}
    {k : IntV sg w} {i : Nat} (hk : k.toInt = (i : Int)) (hi : i < d.cu.len) :
    Mem.span d.idxD (d.cu.ptr 0 + k).addr ⊆ R d r bh := by
  rw [addr_ptr_add d.cu hk]; exact span_cu_sub_R hv hi

/-- The `cu_seqlens` entry at a pointer advanced by `k` and `k'` is in the read footprint. -/
theorem span_cu_add2 {d : Inst} (hv : d.varlen = true) {r bh : Nat} {sg sg' : Bool} {w w' : Nat}
    {k : IntV sg w} {k' : IntV sg' w'} {i : Nat} (hk : k.toInt + k'.toInt = (i : Int))
    (hi : i < d.cu.len) : Mem.span d.idxD ((d.cu.ptr 0 + k) + k').addr ⊆ R d r bh := by
  rw [addr_ptr_add2 d.cu hk]; exact span_cu_sub_R hv hi

/-! ## The prologue -/

/-- `tl.convert .i32 .i64` of the scalar `n < 2^31`: the rest sees an `int64` of value `n`. -/
theorem safe_conv_nat {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {n : Nat} (hn : n < 2 ^ 31) {k : Tensor [] I64 → Prog α}
    (hk : ∀ r : Tensor [] I64, r.item.toInt = (n : Int) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat n : I32)) >>= k) := by
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j
    exact ⟨IntV.ofInt ((IntV.ofNat n : I32).toInt), rfl⟩
  · intro r hr
    refine hk r ?_
    have h1 : r.item = IntV.ofInt ((IntV.ofNat n : I32).toInt) := hr ()
    rw [h1]
    refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
    · have hs := IntV.sfits_toInt (IntV.ofNat n : I32)
      simp only [IntV.SFits] at hs ⊢
      omega
    · exact IntV.I32.toInt_ofNat hn

/-- Tables of a variable-length launch from the precondition. -/
theorem tables_of_pre {d : Inst} {init : Mem} (hpre : Pre d init) :
    d.varlen = true → TablesOk d :=
  fun hv => (hpre.2.2.2.2.2.1 hv).2.2.2.2

/-- **The prologue** (lines 99–106): whatever the tail does with the chunk index, the sequence's
first token and length and the value head (as `int64` values), the body does. -/
theorem safe_prologue {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d) {c : Ctx} (hc0 : c.pid 0 = r)
    (hc1 : c.pid 1 = bh)
    (htail : ∀ i_t bos T i_h : I64, i_t.toInt = (chkOf d r bh : Int) →
      bos.toInt = (baseOf d r bh : Int) → T.toInt = (lenOf d r bh : Int) →
      i_h.toInt = (hvOf d bh : Int) →
      Safe E (R d r bh) (W d r bh) (fun _ m => Q d init r bh m) init (tail d i_t bos T i_h)) :
    Safe E (R d r bh) (W d r bh) (fun _ m => Q d init r bh m) init ((model d).body c) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hciH : d.varlen = true → d.ci.Holds init d.ciVal := fun hv => (hpre.2.2.2.2.2.1 hv).2.2.2.1
  have hcuH : d.varlen = true → d.cu.Holds init d.cuVal := fun hv => (hpre.2.2.2.2.2.1 hv).2.2.1
  have hr31 : r < 2 ^ 31 := lt_trans hr hd.grid0_lt
  have hbh31 : bh < 2 ^ 31 := lt_trans hbh (lt_trans hd.grid1_lt (by norm_num))
  have hHV31 : d.HV < 2 ^ 31 := by
    have := hd.hvv_lt
    have h1 : d.HV ≤ d.HV * d.V := Nat.le_mul_of_pos_right _ hd.V_pos
    omega
  have hHVpos := hd.HV_pos
  have hHV63 : d.HV < 2 ^ 63 := lt_trans hHV31 (by norm_num)
  have hpid0 : tl.program_id c 0 = (IntV.ofNat r : I32) := by simp only [tl.program_id, hc0]
  have hpid1 : tl.program_id c 1 = (IntV.ofNat bh : I32) := by simp only [tl.program_id, hc1]
  rw [body_eq, hpid0, hpid1]
  simp only [Prog.bind_eq]
  -- lines 99: the program ids
  refine safe_conv_nat hr31 fun r1 hv1 => ?_
  refine safe_conv_nat hbh31 fun r2 hv2 => ?_
  have hHVv : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) := IntV.I64.toInt_ofNat hHV63
  -- line 100: `i_b = i_bh // HV`, `i_h = i_bh % HV`
  refine safe_idiv1 (x := r2.item) (y := (IntV.ofNat d.HV : I64)) (by rw [hv2]; omega)
    (by rw [hHVv]; exact_mod_cast hHVpos) fun r3 hr3 => ?_
  have hv3 : r3.item.toInt = (bOf d bh : Int) := by
    rw [hr3, hv2, hHVv]; unfold bOf; rw [Int.natCast_div]
  refine safe_irem1 (x := r2.item) (y := (IntV.ofNat d.HV : I64)) (by rw [hv2]; omega)
    (by rw [hHVv]; exact_mod_cast hHVpos) fun r4 hr4 => ?_
  have hv4 : r4.item.toInt = (hvOf d bh : Int) := by
    rw [hr4, hv2, hHVv]; unfold hvOf; rw [Int.natCast_mod]
  by_cases hvv : d.varlen = true
  · -- variable length: four table reads
    simp only [hvv, ite_true]
    have hrNT : r < d.NT := row_lt_NT hvv hr
    obtain ⟨hseq0, hseqN, hchk0, hchklt⟩ := (hT hvv).2.2.2.1 r hrNT
    have hsN : (seqAt d r).toNat < d.N := by omega
    have hsN1 : (seqAt d r).toNat + 1 ≤ d.N := by omega
    have hNlt := hd.N_lt hvv
    have hcilen := hd.ci_len hvv
    have hculen := hd.cu_len hvv
    have hz1 : IntV.SFits 32 (entryInt d.idxD (d.ciVal (2 * r))) := by
      show IntV.SFits 32 (seqAt d r)
      have hN31 : ((d.N : Nat) : Int) < 2 ^ 31 := by exact_mod_cast hNlt
      simp only [IntV.SFits]
      omega
    have hk1 : (r1.item * (IntV.ofNat 2 : I64)).toInt = ((2 * r : Nat) : Int) := by
      have h2 : (IntV.ofNat 2 : I64).toInt = ((2 : Nat) : Int) := IntV.I64.toInt_ofNat (by norm_num)
      have hm := IntV.I64.mul_nat (x := r) (y := 2) hv1 h2 (by omega)
      simpa [Nat.mul_comm] using hm
    have hk2 : (r1.item * (IntV.ofNat 2 : I64)).toInt + (IntV.ofNat 1 : I32).toInt
        = ((2 * r + 1 : Nat) : Int) := by
      rw [hk1, IntV.I32.toInt_ofNat (by norm_num)]
      push_cast
      ring
    refine load_conv_i32 (dt := d.idxD) (b := d.ci) (v := d.ciVal) (n := 2 * r)
      (hd.idx_ok hvv) (hciH hvv) ?_ ?_ ?_ ?_ ?_
    · omega
    · exact addr_ptr_add d.ci hk1
    · exact span_ci_add hvv hk1 (by omega)
    · exact hz1
    · intro a1 ha1
      have hn1 : a1.item.toInt = (((seqAt d r).toNat : Nat) : Int) := by
        rw [ha1, Int.toNat_of_nonneg hseq0]
        rfl
      refine load_conv_i64 (dt := d.idxD) (b := d.ci) (v := d.ciVal)
        (n := 2 * r + 1) (hd.idx_ok hvv) (hciH hvv) ?_ ?_ ?_ ?_
      · omega
      · exact addr_ptr_add2 d.ci hk2
      · exact span_ci_add2 hvv hk2 (by omega)
      · intro t2 ht2
        have hchk : t2.item.toInt = (chkOf d r bh : Int) := by
          rw [ht2, chkOf_true hvv, Int.toNat_of_nonneg hchk0]
          rfl
        refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal)
          (n := (seqAt d r).toNat) (hd.idx_ok hvv) (hcuH hvv) ?_ ?_ ?_ ?_
        · omega
        · exact addr_ptr_add d.cu hn1
        · exact span_cu_add hvv hn1 (by omega)
        · intro b3 hb3
          have hbs : b3.item.toInt = (baseOf d r bh : Int) := by
            rw [hb3, baseOf_true' hvv, nOf_true hvv, Int.toNat_of_nonneg
              (cuAt_nonneg (hT hvv) (le_of_lt hsN))]
            rfl
          refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal)
            (n := (seqAt d r).toNat + 1) (hd.idx_ok hvv) (hcuH hvv) ?_ ?_ ?_ ?_
          · omega
          · refine addr_ptr_add2 d.cu ?_
            rw [hn1, IntV.I32.toInt_ofNat (by norm_num)]
            push_cast
            ring
          · refine span_cu_add2 (i := (seqAt d r).toNat + 1) hvv ?_ (by omega)
            rw [hn1, IntV.I32.toInt_ofNat (by norm_num)]
            push_cast
            ring
          · intro e4 he4
            have hes : e4.item.toInt = (((cuAt d ((seqAt d r).toNat + 1)).toNat : Nat) : Int) := by
              rw [he4, Int.toNat_of_nonneg (cuAt_nonneg (hT hvv) (by omega))]
              rfl
            have hlen : (e4.item - b3.item).toInt = (lenOf d r bh : Int) := by
              have hq : cuAt d ((seqAt d r).toNat + 1) ≤ d.T :=
                le_trans (cuAt_mono_le (hT hvv) hsN1 le_rfl) (hT hvv).2.2.1
              rw [IntV.I64.sub_nat (x := (cuAt d ((seqAt d r).toNat + 1)).toNat)
                (y := (cuAt d ((seqAt d r).toNat)).toNat) hes
                (by rw [hb3, Int.toNat_of_nonneg (cuAt_nonneg (hT hvv) (le_of_lt hsN))]; rfl)
                (by
                  have := cuAt_step (hT hvv) hsN
                  exact Int.toNat_le_toNat this)
                (by have := hd.T_lt; omega)]
              congr 1
              rw [lenOf, hi_true hvv, lo_true hvv, nOf_true hvv]
            exact htail t2.item b3.item (e4.item - b3.item) r4.item hchk hbs hlen hv4
  · -- fixed length: the `T` conversion
    have hvf : d.varlen = false := by cases h : d.varlen <;> simp_all
    simp only [hvf, Bool.false_eq_true, ite_false]
    refine safe_conv_nat hd.T_lt fun t64 hT64 => ?_
    have hchk : r1.item.toInt = (chkOf d r bh : Int) := by
      rw [chkOf_false hvf]
      exact hv1
    have hbo := bOf_lt hd hbh
    have hbs : (r3.item * t64.item).toInt = (baseOf d r bh : Int) := by
      have hm := IntV.I64.mul_nat (x := bOf d bh) (y := d.T) hv3 hT64 (by
        have h2 : d.T ≤ 2 ^ 22 := hd.T_ctx
        have h3 : d.B ≤ 8192 := hd.B_le
        have h4 : bOf d bh * d.T ≤ d.B * 2 ^ 22 := Nat.mul_le_mul hbo.le h2
        have h5 : d.B * 2 ^ 22 ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul_right _ h3
        omega)
      rw [hm]
      unfold baseOf
      rw [nOf_false hvf, lo_false hvf]
    have hlen : t64.item.toInt = (lenOf d r bh : Int) := by
      rw [hT64]
      simp only [lenOf, hi_false hvf, lo_false hvf, nOf_false hvf]
      congr 1
      omega
    exact htail r1.item (r3.item * t64.item) t64.item r4.item hchk hbs hlen hv4

end Claims.ChunkBwdDvLocal
