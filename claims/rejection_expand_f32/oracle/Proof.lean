import Claim
import TVA.Proof

/-!
# The proof of `rejection_expand_f32`

Program `r` of the grid is request `r`. It reads only `cu_num_tokens` and `x` (`R`), writes only
its own positions `start_r … cu[r] - 1` of the output (`W d init r`, the elements `seg d init r`),
and leaves there `expandVal` of `x[r]` (`Q`). The running sums make the segments of different
requests disjoint, so `correct_of_safe` composes the programs:

1. the running sums: `startOf` is nonnegative, nondecreasing, at most `128 · r`;
2. the segments and footprints, and their disjointness;
3. `safe_replaceVal`: line 126 gives `expandVal` (on `float32` the two `int32` conversions round
   once, `tl.itofRel_binary32`);
4. `safe_tail`, `safe_program`: one program, walked line by line;
5. `main`.
-/

namespace Claims.RejectionExpand

open TVA

/-! ## Integer and pointer helpers -/

theorem i32_zero_toInt : (0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)

theorem i32_one_toInt : (1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by norm_num)

/-- `b.ptr 0 + r` is element `r` of `b`. -/
theorem buf_ptr_add {d' : DType} (b : Buf d') {r : Nat} (hr : r < 2 ^ 31) :
    b.ptr 0 + (IntV.ofNat r : I32) = b.ptr (r : Int) := by
  rw [Ptr.add_int, IntV.I32.toInt_ofNat hr, Buf.ptr_offset, Int.zero_add]

/-- `b.ptr 0 + r - 1` is element `r - 1` of `b`, for `r > 0`. -/
theorem buf_ptr_sub (b : Buf .i32) {r : Nat} (hr0 : 0 < r) (hr : r < 2 ^ 31) :
    b.ptr 0 + (IntV.ofNat r : I32) - (1 : I32) = b.ptr ((r - 1 : Nat) : Int) := by
  have ho : (1 : I32).toInt = 1 := i32_one_toInt
  rw [Ptr.sub_int_of_signed (p := b.ptr 0 + (IntV.ofNat r : I32)) (k := (1 : I32))
    (by rw [ho]; norm_num), ho,
    Ptr.add_int, IntV.I32.toInt_ofNat hr, Ptr.offset_offset, Buf.ptr_offset]
  congr 1
  omega

/-- `rd32` of an element that memory reads as `v` is `v.toInt`. -/
theorem rd32_eq {b : Buf .i32} {m : Mem} {i : Int} {v : I32}
    (h : m.read .i32 (b.ptr i).addr = some v) : rd32 b m i = v.toInt := by
  rw [Buf.ptr_addr] at h
  simp [rd32, h]

/-- Element `i` (an `Int` index in range) of a valid buffer can be read. -/
theorem buf_read_some_int {d' : DType} (b : Buf d') {m : Mem} (hv : b.Valid m) {i : Int}
    (h0 : 0 ≤ i) (hi : i < (b.len : Int)) : ∃ v, m.read d' (b.ptr i).addr = some v := by
  have hc : (i.toNat : Int) = i := Int.toNat_of_nonneg h0
  have hi' : i.toNat < b.len := by omega
  obtain ⟨v, hv'⟩ := b.read_some hv hi'
  exact ⟨v, by rw [← hc]; exact hv'⟩

/-- The bytes of element `i` (an `Int` index in range) lie inside the buffer. -/
theorem buf_span_sub_int {d' : DType} (b : Buf d') {i : Int} (h0 : 0 ≤ i)
    (hi : i < (b.len : Int)) : Mem.span d' (b.ptr i).addr ⊆ b.bytes := by
  have hc : (i.toNat : Int) = i := Int.toNat_of_nonneg h0
  have hi' : i.toNat < b.len := by omega
  have h := b.span_sub (j := i.toNat) hi'
  rwa [hc] at h

/-! ## The running sums -/

section Sums

variable {d : Inst} {init : Mem}

theorem startOf_succ (r : Nat) : startOf d init (r + 1) = rd32 d.cu init (r : Int) := by
  simp [startOf]

theorem cu_eq (r : Nat) : rd32 d.cu init (r : Int) = startOf d init r + nOf d init r := by
  simp only [nOf]; ring

/-- The first position of each request is in `[0, 128 · r]`. -/
theorem startOf_bound (hpre : Pre d init) : ∀ r, r ≤ d.B →
    0 ≤ startOf d init r ∧ startOf d init r ≤ 128 * (r : Int)
  | 0, _ => by simp [startOf]
  | r + 1, h => by
    obtain ⟨h0, h1⟩ := startOf_bound hpre r (by omega)
    obtain ⟨h2, h3⟩ := hpre.2.2.2.1 r (by omega)
    rw [startOf_succ, cu_eq]
    push_cast
    constructor <;> linarith

/-- `startOf` is nondecreasing up to `B`. -/
theorem startOf_mono (hpre : Pre d init) {a : Nat} : ∀ b, a ≤ b → b ≤ d.B →
    startOf d init a ≤ startOf d init b
  | 0, hab, _ => by
    have : a = 0 := by omega
    subst this; exact le_refl _
  | b + 1, hab, hb => by
    rcases Nat.eq_or_lt_of_le hab with h | h
    · rw [h]
    · have ih := startOf_mono hpre (a := a) b (by omega) (by omega)
      have hn := (hpre.2.2.2.1 b (by omega)).1
      rw [startOf_succ, cu_eq]
      linarith

/-- The end of request `r < B` is at most the total `cu[B-1]`, and at most `128 · (r + 1)`. -/
theorem cu_le_last (hpre : Pre d init) (hB : 1 ≤ d.B) {r : Nat} (hr : r < d.B) :
    rd32 d.cu init (r : Int) ≤ rd32 d.cu init ((d.B - 1 : Nat) : Int) := by
  have h1 : rd32 d.cu init (r : Int) = startOf d init (r + 1) := (startOf_succ r).symm
  have h2 : rd32 d.cu init ((d.B - 1 : Nat) : Int) = startOf d init (d.B - 1 + 1) :=
    (startOf_succ _).symm
  rw [h1, h2]
  exact startOf_mono hpre _ (by omega) (by omega)

theorem cu_bound (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    0 ≤ rd32 d.cu init (r : Int) ∧ rd32 d.cu init (r : Int) ≤ 128 * ((r : Int) + 1) := by
  have h := startOf_bound hpre (r + 1) (by omega)
  rw [startOf_succ] at h
  push_cast at h
  exact h

end Sums

/-! ## Segments and footprints -/

/-- The output positions of request `r`: `start_r ≤ j < cu[r]`. -/
def seg (d : Inst) (init : Mem) (r : Nat) : Set Nat :=
  {j | startOf d init r ≤ (j : Int) ∧ (j : Int) < rd32 d.cu init (r : Int)}

/-- Every program may read `cu_num_tokens` and `x`. -/
def R (d : Inst) : Set Int := d.cu.bytes ∪ d.inp.bytes

/-- Program `r` writes its request's positions of the output. -/
def W (d : Inst) (init : Mem) (r : Nat) : Set Int := d.out.elems (seg d init r)

/-- What program `r` leaves: each of its positions holds `expandVal` of `x[r]`. -/
def Q (d : Inst) (init : Mem) (r : Nat) (m : Mem) : Prop :=
  ∀ j ∈ seg d init r, ∃ v, init.read d.dt (d.inp.ptr (r : Int)).addr = some v ∧
    m.read d.dt (d.out.ptr (j : Int)).addr = some (expandVal d.dt v d.rf d.rt)

theorem seg_sub_last {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) : seg d init r ⊆ {j | (j : Int) < rd32 d.cu init ((d.B - 1 : Nat) : Int)} :=
  fun _ hj => lt_of_lt_of_le hj.2 (cu_le_last hpre hd.hB hr)

theorem seg_lt_len {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) : ∀ j ∈ seg d init r, j < d.out.len := by
  intro j hj
  have h1 := seg_sub_last hd hpre hr hj
  have h2 := hpre.2.2.2.2
  simp only [Set.mem_ofPred_eq] at h1
  omega

theorem seg_disjoint {d : Inst} {init : Mem} (hpre : Pre d init) {r r' : Nat} (hr : r < d.B)
    (hr' : r' < d.B) (h : r ≠ r') : Disjoint (seg d init r) (seg d init r') := by
  wlog hlt : r < r' generalizing r r'
  · exact (this hr' hr (Ne.symm h) (by omega)).symm
  refine Set.disjoint_left.mpr fun j h1 h2 => ?_
  have hm : rd32 d.cu init (r : Int) ≤ startOf d init r' := by
    rw [← startOf_succ]
    exact startOf_mono hpre _ (by omega) (by omega)
  have := h1.2
  have := h2.1
  omega

theorem W_sub_out {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) : W d init r ⊆ d.out.bytes :=
  d.out.elems_sub_bytes (seg_lt_len hd hpre hr)

theorem W_disjoint_R {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) : Disjoint (W d init r) (R d) :=
  Disjoint.union_right (hd.hout_cu.mono_left (W_sub_out hd hpre hr))
    (hd.hout_inp.mono_left (W_sub_out hd hpre hr))

/-! ## Line 126 -/

/-- **Line 126** gives `expandVal`: on `float32` both `int32` conversions round once to nearest
(an `int32` is exact in `float64`), and the selection is the float comparison's; on `int32` it is
the integer comparison. -/
theorem safe_replaceVal {α : Type} {E : Env} {R W : Set Int} {Q : α → Mem → Prop} {m : Mem}
    {dt : DType} (hdt : dt = .f32 ∨ dt = .i32) (v : dt.Val) {a b : Int}
    (ha : -2 ^ 31 ≤ a ∧ a < 2 ^ 31) (hb : -2 ^ 31 ≤ b ∧ b < 2 ^ 31) {k : dt.Val → Prog α}
    (hk : Safe E R W Q m (k (expandVal dt v a b))) :
    Safe E R W Q m (replaceVal dt v (IntV.ofInt a) (IntV.ofInt b) >>= k) := by
  have hai : (IntV.ofInt a : I32).toInt = a := IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
  have hbi : (IntV.ofInt b : I32).toInt = b := IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
  rcases hdt with rfl | rfl
  · simp only [replaceVal, Prog.bind_eq, Prog.bind_assoc]
    refine safe_itof (f := .binary32) (by decide) fun t ht => ?_
    simp only [Prog.ret_bind, Prog.pure_eq]
    refine safe_itof (f := .binary32) (by decide) fun t2 ht2 => ?_
    have e1 : t.item = (FloatV.ofRat (a : Rat) : F32) := by
      have h := ht ()
      rw [show ((Tensor.scalar (IntV.ofInt a : I32)).get ()).toInt = a from hai] at h
      exact (tl.itofRel_binary32 (z := a) (by
        rw [abs_lt]; constructor <;> linarith [ha.1, ha.2])).mp h
    have e2 : t2.item = (FloatV.ofRat (b : Rat) : F32) := by
      have h := ht2 ()
      rw [show ((Tensor.scalar (IntV.ofInt b : I32)).get ()).toInt = b from hbi] at h
      exact (tl.itofRel_binary32 (z := b) (by
        rw [abs_lt]; constructor <;> linarith [hb.1, hb.2])).mp h
    have e : (tl.where (Tensor.scalar (FloatV.eq v t.item)) (Tensor.scalar t2.item)
        (Tensor.scalar v)).item = expandVal .f32 v a b := by
      rw [e1, e2]
      rfl
    rw [e]
    exact hk
  · have e : (tl.where (Tensor.scalar (v.eq (IntV.ofInt a : I32))) (Tensor.scalar (IntV.ofInt b : I32))
        (Tensor.scalar v)).item = expandVal .i32 v a b := by
      show (if v.eq (IntV.ofInt a : I32) then (IntV.ofInt b : I32) else v) = _
      rw [IntV.eq_eq, hai]
      simp only [expandVal, decide_eq_true_eq]
    show Safe E R W Q m (k (tl.where (Tensor.scalar (v.eq (IntV.ofInt a : I32)))
      (Tensor.scalar (IntV.ofInt b : I32)) (Tensor.scalar v)).item)
    rw [e]
    exact hk

/-! ## One program -/

/-- Program `r`'s body after `start_idx`. -/
def tail (d : Inst) (r : Nat) (start_idx : I32) : Prog Unit := do
  let end_idx ← tl.load1 (d := .i32) (d.cu.ptr 0 + (IntV.ofNat r : I32))
  let num_tokens := end_idx - start_idx
  let src_val ← tl.load1 (d.inp.ptr 0 + (IntV.ofNat r : I32))
  let src_val ← replaceVal d.dt src_val (IntV.ofInt d.rf) (IntV.ofInt d.rt)
  let offset := tl.arange 0 128
  tl.store (d.out.ptr 0 + start_idx + offset) (bc src_val) (some (offset.ltS num_tokens))

/-- Program `r` of the launch. -/
def prog (d : Inst) (r : Nat) : Prog Unit := do
  let req_idx := (IntV.ofNat r : I32)
  let start_idx : I32 ←
    if req_idx.eq (0 : I32) then
      pure (tl.zeros [] .i32).item
    else
      tl.load1 (d := .i32) (d.cu.ptr 0 + req_idx - (1 : I32))
  tail d r start_idx

theorem body_eq (d : Inst) (c : Ctx) : (model d).body c = prog d (c.pid 0) := rfl

/-- The body after `start_idx`, given its value `startOf`. -/
theorem safe_tail {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {r : Nat}
    (hr : r < d.B) {start_idx : I32} (hst : start_idx.toInt = startOf d init r) :
    Safe E (R d) (W d init r) (fun _ m => Q d init r m) init (tail d r start_idx) := by
  obtain ⟨hvo, hvi, hvc, hn, hlast⟩ := hpre
  have hpre' : Pre d init := ⟨hvo, hvi, hvc, hn, hlast⟩
  have hr31 : r < 2 ^ 31 := by have := hd.hB_le; omega
  have hi0 : (0 : Int) ≤ ((r : Nat) : Int) := Int.natCast_nonneg _
  have hcl : ((r : Nat) : Int) < (d.cu.len : Int) := by exact_mod_cast lt_of_lt_of_le hr hd.hcu
  have hil : ((r : Nat) : Int) < (d.inp.len : Int) := by exact_mod_cast lt_of_lt_of_le hr hd.hinp
  obtain ⟨ve, hve⟩ := buf_read_some_int d.cu hvc hi0 hcl
  obtain ⟨sv, hsv⟩ := buf_read_some_int d.inp hvi hi0 hil
  obtain ⟨hs0, hs1⟩ := startOf_bound hpre' r (by omega)
  obtain ⟨hn0, hn1⟩ := hn r hr
  have hcu := cu_bound hpre' hr
  have hBle := hd.hB_le
  have hend : ve.toInt = startOf d init r + nOf d init r := by
    rw [← rd32_eq hve, cu_eq]
  have hnum : (ve - start_idx).toInt = nOf d init r := by
    have hfit : IntV.SFits 32 (ve.toInt - start_idx.toInt) := by
      simp only [IntV.SFits, hend, hst]
      constructor <;> norm_num <;> linarith
    rw [IntV.toInt_sub rfl rfl hfit, hend, hst]
    ring
  unfold tail
  rw [buf_ptr_add d.cu hr31, buf_ptr_add d.inp hr31]
  refine safe_load1 (d := .i32) hve (fun x hx => Or.inl (buf_span_sub_int d.cu hi0 hcl hx)) ?_
  refine safe_load1 hsv (fun x hx => Or.inr (buf_span_sub_int d.inp hi0 hil hx)) ?_
  refine safe_replaceVal hd.hdt sv hd.hrf hd.hrt ?_
  refine safe_last ?_
  -- the store: lane `i` is element `start_r + i`, enabled when `i < n_r`
  let s0 : Nat := (startOf d init r).toNat
  have hs0' : (s0 : Int) = startOf d init r := Int.toNat_of_nonneg hs0
  let g : Idx [128] → Nat := fun i => s0 + i.1.val
  let en : Idx [128] → Bool := fun i => decide ((i.1.val : Int) < nOf d init r)
  have hl : ∀ i : Fin 128, ((tl.arange 0 128).get (i, ())).toInt = ((i : Nat) : Int) :=
    fun i => IntV.I32.toInt_tl_arange (n := 128) (by norm_num) i
  have hmask : ∀ i, ((tl.arange 0 128).ltS (ve - start_idx)).get i = en i := by
    rintro ⟨i, ⟨⟩⟩
    show IntV.lt ((tl.arange 0 128).get (i, ())) (ve - start_idx) = en (i, ())
    rw [IntV.lt_eq, hnum, hl i]
  have hptr : ∀ i, en i = true →
      (d.out.ptr 0 + start_idx + tl.arange 0 128).get i = d.out.ptr (g i : Int) := by
    rintro ⟨i, ⟨⟩⟩ _
    show (d.out.ptr 0 + start_idx).offset (((tl.arange 0 128).get (i, ())).toInt) = _
    rw [Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset, hl i, hst]
    simp only [g]
    push_cast
    rw [hs0']
    ring_nf
  have hseg : ∀ i, en i = true → g i ∈ seg d init r := by
    intro i hi
    simp only [en, decide_eq_true_eq] at hi
    refine ⟨?_, ?_⟩
    · simp only [g]; push_cast; rw [hs0']; omega
    · simp only [g]; push_cast; rw [hs0', cu_eq]; omega
  have himg : Buf.img g en ⊆ seg d init r := by
    rintro _ ⟨i, hi, rfl⟩; exact hseg i hi
  refine safe_store_buf d.out (Buf.addrs_of_get d.out hmask hptr)
    (fun i hi => seg_lt_len hd hpre' hr _ (hseg i hi)) ?_ hvo (d.out.elems_mono himg)
    fun m' hm' _ _ => safe_ret ?_
  · rintro ⟨i, ⟨⟩⟩ ⟨j, ⟨⟩⟩ _ _ h
    have h2 : s0 + (i : Nat) = s0 + (j : Nat) := h
    exact Prod.ext (Fin.ext (by dsimp only; omega)) rfl
  · intro j hj
    obtain ⟨hj1, hj2⟩ := hj
    have hjs : s0 ≤ j := by omega
    have hjl : j - s0 < 128 := by rw [cu_eq] at hj2; omega
    let i : Idx [128] := (⟨j - s0, hjl⟩, ())
    have hgi : g i = j := by simp only [g, i]; omega
    have hei : en i = true := by
      simp only [en, i, decide_eq_true_eq]
      rw [cu_eq] at hj2
      push_cast [Nat.cast_sub hjs]
      omega
    refine ⟨sv, hsv, ?_⟩
    have := hm' i hei
    rw [hgi] at this
    exact this

/-- **One program** of the launch, request `r`, is safe from `init`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {r : Nat}
    (hr : r < d.B) :
    Safe E (R d) (W d init r) (fun _ m => Q d init r m) init (prog d r) := by
  have hr31 : r < 2 ^ 31 := by have := hd.hB_le; omega
  unfold prog
  dsimp only
  split
  case isTrue hcond =>
    have hz : (IntV.ofNat r : I32).toInt = (0 : I32).toInt := by
      rw [IntV.eq_eq] at hcond; exact of_decide_eq_true hcond
    rw [IntV.I32.toInt_ofNat hr31, i32_zero_toInt] at hz
    have hr0 : r = 0 := by omega
    have hst : ((tl.zeros [] .i32).item : I32).toInt = startOf d init r := by
      rw [hr0]
      have h0 : ((IntV.ofNat 0 : I32).toInt) = (0 : Int) := IntV.I32.toInt_ofNat (by norm_num)
      have h1 : startOf d init 0 = 0 := by simp [startOf]
      rw [h1]
      simp [tl.zeros, Tensor.full, Tensor.item, DType.ofBits] at h0 ⊢
    exact safe_tail hd hpre E hr hst
  case isFalse hne =>
    have hr0 : 0 < r := by
      rcases Nat.eq_zero_or_pos r with h | h
      · subst h; exact absurd rfl hne
      · exact h
    have hrne : r ≠ 0 := by omega
    have hi0 : (0 : Int) ≤ ((r - 1 : Nat) : Int) := Int.natCast_nonneg _
    have hcl : ((r - 1 : Nat) : Int) < (d.cu.len : Int) := by
      have h1 : r - 1 < d.B := by omega
      exact_mod_cast lt_of_lt_of_le h1 hd.hcu
    obtain ⟨v, hv⟩ := buf_read_some_int d.cu hpre.2.2.1 hi0 hcl
    have hstart : v.toInt = startOf d init r := by
      rw [← rd32_eq hv]
      simp [startOf, hrne]
    rw [buf_ptr_sub d.cu hr0 hr31]
    exact safe_load1 (d := .i32) hv (fun x hx => Or.inl (buf_span_sub_int d.cu hi0 hcl hx))
      (safe_tail hd hpre E hr hstart)

/-! ## The launch -/

theorem launch_size (d : Inst) : (launch d).size = d.B := Launch.size_grid1 _ _ _

theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (by have := hd.hB_le; omega) ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨7, by norm_num⟩
  · norm_num

/-- **The claim.** Each program is safe on its own segment (`safe_program`); the segments are
disjoint from each other and from the inputs, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P _ (d : Inst) hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hlt : ∀ i : Fin (launch d).size, (i : Nat) < d.B := fun i => lt_of_lt_of_eq i.isLt (launch_size d)
  refine correct_of_safe (fun _ => R d) (fun i => W d init i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · rw [body_eq, launch_pid]
    exact safe_program hd hpre _ (hlt i)
  · intro i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (W_disjoint_R hd hpre (hlt i))
      (d.out.elems_disjoint (seg_disjoint hpre (hlt i) (hlt j) hij'))
  · intro i
    exact (W_disjoint_R hd hpre (hlt i)).symm
  · intro fin hloc hfr
    refine ⟨fun r hr j hj1 hj2 => ?_, fun z hz => ?_⟩
    · have hi : r < (launch d).size := by rw [launch_size]; exact hr
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨r, hi⟩
      have hjs : j ∈ seg d init r := ⟨hj1, hj2⟩
      obtain ⟨v, hv, hm⟩ := hQ j hjs
      refine ⟨v, hv, ?_⟩
      rw [Mem.read_congr fun x hx => hagree x (Or.inr (d.out.span_sub_elems hjs hx))]
      exact hm
    · exact hfr z fun i hzi => hz (d.out.elems_mono (seg_sub_last hd hpre (hlt i)) hzi)

end Claims.RejectionExpand
