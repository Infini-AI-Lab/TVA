import Proof.Arith

/-!
# The loop invariant

One trip of the loop: while `rejected` is `false`, column `t` of row `r` is stored with the
converted target, and `rejected` becomes `true` exactly at the first position where the draft and
the target differ.  `EV` is what the trip's columns hold after `t` trips, and `Invs` ties the
carried `rejected` and the memory to `matchLen`.
-/

namespace Claims.RejectionGreedy

open TVA

/-! ## Buffers at integer indices -/

/-- Element `i` (an `Int` index in range) of a buffer lies in its bytes. -/
theorem buf_span_sub_int {d' : DType} (b : Buf d') {i : Int} (h0 : 0 ≤ i)
    (hi : i < (b.len : Int)) : Mem.span d' (b.ptr i).addr ⊆ b.bytes := by
  have hc : (i.toNat : Int) = i := Int.toNat_of_nonneg h0
  have hi' : i.toNat < b.len := by omega
  have h := b.span_sub (j := i.toNat) hi'
  rwa [hc] at h

/-- Element `i` (an `Int` index in range) of a valid buffer can be read. -/
theorem buf_read_some_int {d' : DType} (b : Buf d') {m : Mem} (hv : b.Valid m) {i : Int}
    (h0 : 0 ≤ i) (hi : i < (b.len : Int)) : ∃ v, m.read d' (b.ptr i).addr = some v := by
  have hc : (i.toNat : Int) = i := Int.toNat_of_nonneg h0
  have hi' : i.toNat < b.len := by omega
  obtain ⟨v, hv'⟩ := b.read_some hv hi'
  exact ⟨v, by rw [← hc]; exact hv'⟩

/-- Element `i` (an `Int` index in range) of a valid buffer is aligned. -/
theorem buf_aligned_int {d' : DType} (b : Buf d') {m : Mem} (hv : b.Valid m) {i : Int}
    (h0 : 0 ≤ i) (hi : i < (b.len : Int)) : Mem.Aligned d' (b.ptr i).addr := by
  have hc : (i.toNat : Int) = i := Int.toNat_of_nonneg h0
  have hi' : i.toNat < b.len := by omega
  have h := b.aligned hv hi'
  rwa [hc] at h

/-- The low `w` bits of an `int` as a signed value. -/
theorem ofInt_toInt {w : Nat} (z : Int) : (IntV.ofInt z : IntV true w).toInt = Int.bmod z (2 ^ w) := by
  simp only [IntV.toInt, IntV.ofInt, ite_true, BitVec.toInt_ofInt]

/-! ## Footprints -/

/-- A row of the batch lies in the output's bytes. -/
theorem Wf_sub_out_bytes {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    Wf d r ⊆ d.out.bytes :=
  d.out.elems_sub_bytes fun _ hj => lt_of_lt_of_le (rowSet_sub hr hj) hd.hout_len

/-- `cu`'s bytes are in the read footprint. -/
theorem cu_sub_Rf (d : Inst) : d.cu.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, Set.mem_union]; tauto

/-- `draft`'s bytes are in the read footprint. -/
theorem draft_sub_Rf (d : Inst) : d.draft.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, Set.mem_union]; tauto

/-- `target`'s bytes are in the read footprint. -/
theorem target_sub_Rf (d : Inst) : d.target.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, Set.mem_union]; tauto

/-- `bonus`'s bytes are in the read footprint. -/
theorem bonus_sub_Rf (d : Inst) : d.bonus.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, Set.mem_union]; tauto

/-- The mask's bytes are in the read footprint. -/
theorem greedy_sub_Rf {d : Inst} {g : Buf .i1} (hg : d.greedy = some g) : g.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, hg, Set.mem_union]; tauto

/-- A row shares no byte with `cu`. -/
theorem Wf_disjoint_cu {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    Disjoint (Wf d r) d.cu.bytes :=
  Disjoint.mono (Wf_sub_out_bytes hd hr) (le_refl _) hd.hout_cu

/-- A row shares no byte with `draft`. -/
theorem Wf_disjoint_draft {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    Disjoint (Wf d r) d.draft.bytes :=
  Disjoint.mono (Wf_sub_out_bytes hd hr) (le_refl _) hd.hout_draft

/-- A row shares no byte with `target`. -/
theorem Wf_disjoint_target {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    Disjoint (Wf d r) d.target.bytes :=
  Disjoint.mono (Wf_sub_out_bytes hd hr) (le_refl _) hd.hout_target

/-- A row shares no byte with `bonus`. -/
theorem Wf_disjoint_bonus {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    Disjoint (Wf d r) d.bonus.bytes :=
  Disjoint.mono (Wf_sub_out_bytes hd hr) (le_refl _) hd.hout_bonus

/-- A row shares no byte with the mask. -/
theorem Wf_disjoint_greedy {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) {g : Buf .i1}
    (hg : d.greedy = some g) : Disjoint (Wf d r) g.bytes :=
  Disjoint.mono (Wf_sub_out_bytes hd hr) (le_refl _) (hd.hout_greedy g hg)

/-- Reading a buffer whose bytes avoid the rows written so far gives the initial value. -/
theorem read_frame {d : Inst} {init m : Mem} {r : Nat}
    (hfr : ∀ x, x ∉ Wf d r → m x = init x)
    {d' : DType} {b : Buf d'} (hb : Disjoint (Wf d r) b.bytes) {i : Int}
    (h0 : 0 ≤ i) (hi : i < (b.len : Int)) :
    m.read d' (b.ptr i).addr = init.read d' (b.ptr i).addr :=
  Mem.read_congr fun x hx =>
    hfr x fun hxw => Set.disjoint_left.mp hb hxw (buf_span_sub_int b h0 hi hx)

/-! ## The invariant -/

/-- What column `j` of row `r` holds after `t` trips: the converted target at every position up to
`t` that is also up to `matchLen`. -/
def EV (d : Inst) (init : Mem) (r t j : Nat) : Option Int :=
  if (j : Int) ≤ matchLen d init r ∧ (j : Int) < (t : Int) then
    some (rdt64 d.target init (startOf d init r + (j : Int)))
  else none

/-- The state after `t` trips: `rejected`, the columns of row `r`, and the untouched bytes. -/
def Invs (d : Inst) (init : Mem) (r t : Nat) (rej : Bool) (m : Mem) : Prop :=
  (rej = true ↔ matchLen d init r < (t : Int)) ∧
    (∀ j, j < d.S + 1 →
      match EV d init r t j with
      | some v => ∃ w : I32, m.read .i32 (d.out.ptr (oidx d r j)).addr = some w ∧ w.toInt = v
      | none =>
        m.read .i32 (d.out.ptr (oidx d r j)).addr =
          init.read .i32 (d.out.ptr (oidx d r j)).addr) ∧
    (∀ x, x ∉ Wf d r → m x = init x) ∧
    (∀ x, x ∈ Wf d r → m x ≠ none)

/-- `EV` at a column below both `matchLen` (included) and the trip count is the converted target. -/
theorem EV_some {d : Inst} {init : Mem} {r t j : Nat} (h1 : (j : Int) ≤ matchLen d init r)
    (h2 : (j : Int) < (t : Int)) :
    EV d init r t j = some (rdt64 d.target init (startOf d init r + (j : Int))) := by
  unfold EV
  rw [ite_eq_left (⟨h1, h2⟩)]

/-- `EV` elsewhere is `none`. -/
theorem EV_none {d : Inst} {init : Mem} {r t j : Nat}
    (h : ¬ ((j : Int) ≤ matchLen d init r ∧ (j : Int) < (t : Int))) : EV d init r t j = none := by
  unfold EV
  rw [ite_eq_right (h)]

/-- Before the loop, nothing is written and `rejected` is `false`. -/
theorem Invs_init {d : Inst} {init : Mem} {r : Nat} (hpre : Pre d init) (hd : d.ok)
    (hr : r < d.B) : Invs d init r 0 false init := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h : ¬ (matchLen d init r < ((0 : Nat) : Int)) := by
      simpa using not_lt.mpr (matchLen_nonneg d init r)
    show (false = true ↔ matchLen d init r < ((0 : Nat) : Int))
    rw [Bool.false_eq_true, false_iff]
    exact h
  · intro j hj
    have h : ¬ ((j : Int) ≤ matchLen d init r ∧ (j : Int) < ((0 : Nat) : Int)) := by
      rintro ⟨-, h2⟩
      omega
    rw [EV_none h]
  · intro x _; rfl
  · intro x hx
    exact hpre.out_valid.2.2.2 x (Wf_sub_out_bytes hd hr hx)

/-- The reduction of the trip when `rejected` is already `true`. -/
theorem tripBody_true (d : Inst) (r : Nat) (s p : I32) :
    tripBody d r s p true = pure true := by
  unfold tripBody
  rfl

/-- The reduction of the trip when `rejected` is `false`. -/
theorem tripBody_false (d : Inst) (r : Nat) (s p : I32) :
    tripBody d r s p false =
      (do
        let draft_token_id ← tl.load1 (d := .i32) (d.draft.ptr 0 + s + p)
        let target64 ← tl.load1 (d := .i64) (d.target.ptr 0 + s + p)
        let target_argmax_id : I32 := (← tl.convert .i64 .i32 (Tensor.scalar target64)).item
        let token_id := target_argmax_id
        let rejected := !(draft_token_id.eq target_argmax_id)
        tl.store1 (d.out.ptr 0 + rowOff d r + p) token_id
        pure rejected) := by
  unfold tripBody
  rfl

/-- One trip of the loop, at position `t < nOf`. -/
theorem safe_trip {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B)
    {E : Env} {start_idx : I32} (hst : start_idx.toInt = startOf d init r)
    {n : Nat} (hn : n = (nOf d init r).toNat) {t : Nat} (htn : t < n)
    {pos : I32} (hpos : pos.toInt = (t : Int))
    {rej : Bool} {m : Mem} (hI : Invs d init r t rej m) :
    Safe E (Rf d) (Wt d r t) (fun rej' m' => Invs d init r (t + 1) rej' m') m
      (tripBody d r start_idx pos rej) := by
  have hn0 : 0 ≤ nOf d init r := nOf_nonneg hpre hr
  have hnI : (n : Int) = nOf d init r := by
    rw [hn]; exact Int.toNat_of_nonneg hn0
  have htnI : (t : Int) < nOf d init r := by
    have h : (t : Int) < (n : Int) := by exact_mod_cast htn
    omega
  have ht31 : (t : Int) < 2 ^ 31 := lt_trans htnI (nOf_lt_2p31 hd hpre hr)
  have hkl : matchLen d init r ≤ nOf d init r := matchLen_le_nOf d init r hn0
  have hk0 : (0 : Int) ≤ matchLen d init r := matchLen_nonneg d init r
  have hst0 : (0 : Int) ≤ startOf d init r := startOf_nonneg hpre hr
  have ht0 : (0 : Int) ≤ (t : Int) := Int.natCast_nonneg t
  have hstt0 : (0 : Int) ≤ startOf d init r + (t : Int) := by omega
  have hS1 : t < d.S + 1 := by
    have := hpre.nOf_le hr
    omega
  have hfr : ∀ x, x ∉ Wf d r → m x = init x := hI.2.2.1
  cases rej with
  | true =>
    have hk : matchLen d init r < (t : Int) := hI.1.mp rfl
    rw [tripBody_true]
    refine safe_ret ?_
    refine ⟨?_, ?_, ?_, ?_⟩
    · show (true = true) ↔ matchLen d init r < ((t + 1 : Nat) : Int)
      simp
      omega
    · intro j hj
      have hEv : EV d init r (t + 1) j = EV d init r t j := by
        by_cases hc : ((j : Int) ≤ matchLen d init r ∧ (j : Int) < (t : Int))
        · rw [EV_some hc.1 hc.2, EV_some hc.1 (by push_cast; omega)]
        · rw [EV_none hc, EV_none (fun h2 => hc ⟨h2.1, by omega⟩)]
      rw [hEv]
      exact hI.2.1 j hj
    · exact hI.2.2.1
    · exact hI.2.2.2
  | false =>
    have hk : ¬ matchLen d init r < (t : Int) := by
      intro h
      have h' := hI.1.mpr h
      simp at h'
    have hkle : (t : Int) ≤ matchLen d init r := not_lt.mp hk
    have hlenD : startOf d init r + (t : Int) < (d.draft.len : Int) := by
      have h1 := startOf_add_nOf (d := d) init r
      have h2 := cu_le_draft hpre hr
      omega
    have hlenT : startOf d init r + (t : Int) < (d.target.len : Int) := by
      have h1 := startOf_add_nOf (d := d) init r
      have h2 := cu_le_target hpre hr
      omega
    obtain ⟨v0, hv0i⟩ := buf_read_some_int d.draft hpre.draft_valid hstt0 hlenD
    obtain ⟨v1, hv1i⟩ := buf_read_some_int d.target hpre.target_valid hstt0 hlenT
    have hv0 : m.read .i32 (d.draft.ptr (startOf d init r + (t : Int))).addr = some v0 := by
      rw [read_frame (r := r) hfr (Wf_disjoint_draft hd hr) hstt0 hlenD]
      exact hv0i
    have hv1 : m.read .i64 (d.target.ptr (startOf d init r + (t : Int))).addr = some v1 := by
      rw [read_frame (r := r) hfr (Wf_disjoint_target hd hr) hstt0 hlenT]
      exact hv1i
    have hoidx : oidx d r t < d.out.len := by
      have h1 : oidx d r t < (r + 1) * (d.S + 1) := by
        have he : (r + 1) * (d.S + 1) = r * (d.S + 1) + (d.S + 1) := Nat.succ_mul _ _
        rw [he, oidx]
        exact Nat.add_lt_add_left hS1 _
      exact Nat.lt_of_lt_of_le h1
        (le_trans (Nat.mul_le_mul_right _ (by omega : r + 1 ≤ d.B)) hd.hout_len)
    have hw : m.Writable .i32 (d.out.ptr (oidx d r t)).addr := by
      refine ⟨?_, ?_⟩
      · exact buf_aligned_int d.out hpre.out_valid (Int.natCast_nonneg _) (by exact_mod_cast hoidx)
      · intro k hkk
        exact hI.2.2.2 _ (Wt_sub_Wf hS1 (Mem.mem_span hkk))
    rw [tripBody_false,
      draft_ptr_eq (st := startOf d init r) hst hpos,
      target_ptr_eq (st := startOf d init r) hst hpos,
      out_ptr_eq (r := r) (t := t) (rowOff_toInt hd hr) hpos]
    refine safe_load1 (d := .i32) hv0 ?_ ?_
    · intro x hx
      exact draft_sub_Rf d (buf_span_sub_int d.draft hstt0 hlenD hx)
    dsimp only
    refine safe_load1 (d := .i64) hv1 ?_ ?_
    · intro x hx
      exact target_sub_Rf d (buf_span_sub_int d.target hstt0 hlenT hx)
    refine safe_convert (d := .i64) (d' := .i32) (x := Tensor.scalar v1) ?_ ?_
    · intro i
      exact ⟨IntV.ofInt v1.toInt, rfl⟩
    intro rt hrt
    have hconv' : rt.get () = IntV.ofInt v1.toInt := hrt ()
    have hwv : (rt.get ()).toInt = rdt64 d.target init (startOf d init r + (t : Int)) := by
      rw [hconv', ofInt_toInt, rdt64_eq hv1i]
    have hiff : (v0.eq (rt.get ())) = true ↔
        rd32 d.draft init (startOf d init r + (t : Int)) =
          rdt64 d.target init (startOf d init r + (t : Int)) := by
      rw [rd32_eq hv0i, ← hwv]
      exact IntV.cmpKey_eq v0 (rt.get ())
    refine safe_store_any (ptrs := Tensor.scalar (d.out.ptr (oidx d r t)))
      (vals := Tensor.scalar (rt.get ())) (mask := none) ?_ ?_ ?_
    · rw [storeOk_scalar]; exact hw
    · rw [accBytes_scalar]; exact subset_rfl
    intro order hperm
    refine safe_ret ?_
    have hread : (writeAll m .i32 (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r t))) none)
        (Tensor.scalar (rt.get ())) order).read .i32 (d.out.ptr (oidx d r t)).addr
        = some (rt.get ()) := by
      exact writeAll_read (distinctLanes_scalar _) ((storeOk_scalar _).mpr hw) order
        (hperm.nodup_iff.mpr (enabled_nodup _)) () (d.out.ptr (oidx d r t)).addr
        (hperm.mem_iff.mpr (mem_enabled rfl)) rfl
    refine ⟨?_, ?_, ?_, ?_⟩
    · rcases lt_or_eq_of_le hkle with hlt | heq
      · have hdt := matchLen_eq_of_lt d init r ht0 hlt
        have hbe : (v0.eq (rt.get ())) = true := hiff.mpr hdt
        have hrej : (!(v0.eq (rt.get ()))) = false := by rw [hbe, Bool.not_true]
        show (!(v0.eq (rt.get ()))) = true ↔ matchLen d init r < ((t + 1 : Nat) : Int)
        rw [hrej]
        simp
        omega
      · have hne : rd32 d.draft init (startOf d init r + (t : Int)) ≠
            rdt64 d.target init (startOf d init r + (t : Int)) :=
          matchLen_ne_of_lt d init r ht0 htnI hkle (by omega)
        have hbe : (v0.eq (rt.get ())) = false := by
          cases h : (v0.eq (rt.get ()))
          · rfl
          · exact absurd (hiff.mp h) hne
        have hrej : (!(v0.eq (rt.get ()))) = true := by rw [hbe, Bool.not_false]
        show (!(v0.eq (rt.get ()))) = true ↔ matchLen d init r < ((t + 1 : Nat) : Int)
        rw [hrej]
        simp
        omega
    · intro j hj
      by_cases hjt : j = t
      · subst j
        have hEv : EV d init r (t + 1) t = some
            (rdt64 d.target init (startOf d init r + (t : Int))) :=
          EV_some hkle (by push_cast; omega)
        rw [hEv]
        exact ⟨rt.get (), hread, hwv⟩
      · have hEv : EV d init r (t + 1) j = EV d init r t j := by
          by_cases hc : ((j : Int) ≤ matchLen d init r ∧ (j : Int) < (t : Int))
          · rw [EV_some hc.1 hc.2, EV_some hc.1 (by push_cast; omega)]
          · rw [EV_none hc, EV_none (fun h2 => hc ⟨h2.1, by omega⟩)]
        have hfr' : (writeAll m .i32 (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r t))) none)
            (Tensor.scalar (rt.get ())) order).read .i32 (d.out.ptr (oidx d r j)).addr
            = m.read .i32 (d.out.ptr (oidx d r j)).addr := by
          apply Mem.read_congr
          intro x hx
          have hspan : Disjoint (Mem.span .i32 (d.out.ptr (oidx d r j)).addr)
              (accBytes .i32 (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r t))) none)) := by
            rw [accBytes_scalar]
            exact Wt_disjoint hjt
          exact writeAll_frame m _ _ order (Set.disjoint_left.mp hspan hx)
        rw [hEv, hfr']
        exact hI.2.1 j hj
    · intro x hx
      have hxa : x ∉ accBytes .i32 (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r t))) none) := by
        rw [accBytes_scalar]
        exact fun hw' => hx (Wt_sub_Wf hS1 hw')
      rw [writeAll_frame m _ _ order hxa]
      exact hI.2.2.1 x hx
    · intro x hx
      exact writeAll_present _ _ order (hI.2.2.2 x hx)

end Claims.RejectionGreedy
