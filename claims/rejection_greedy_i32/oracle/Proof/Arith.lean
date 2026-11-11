import Proof.Diff

/-!
# Bounds and addresses

Numeric consequences of `Inst.ok` and `Pre`: the running sums `cu` are nonnegative and
nondecreasing, so `startOf` and `nOf` are nonnegative and `startOf + nOf = cu[r]`; the loop
bounds are in range; and the integer offsets the program builds are the element indices of the
buffers.
-/

namespace Claims.RejectionGreedy

open TVA

/-! ## The parts of `Pre` -/

/-- `Pre` unpacked into its twelve conjuncts. -/
theorem Pre.parts {d : Inst} {init : Mem} (h : Pre d init) :
    d.out.Valid init ∧ d.cu.Valid init ∧ d.draft.Valid init ∧ d.target.Valid init ∧
      d.bonus.Valid init ∧ (∀ g, d.greedy = some g → g.Valid init) ∧
      0 ≤ rd32 d.cu init 0 ∧
      (∀ r, r + 1 < d.B →
        rd32 d.cu init ((r : Nat) : Int) ≤ rd32 d.cu init ((r + 1 : Nat) : Int)) ∧
      rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.draft.len ∧
      rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.target.len ∧
      (∀ r, r < d.B → nOf d init r ≤ d.S) ∧
      rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ 2 ^ 22 := h

/-- The output buffer is valid in the initial memory. -/
theorem Pre.out_valid {d : Inst} {init : Mem} (h : Pre d init) : d.out.Valid init :=
  h.parts.1

/-- `cu_num_draft_tokens` is valid in the initial memory. -/
theorem Pre.cu_valid {d : Inst} {init : Mem} (h : Pre d init) : d.cu.Valid init :=
  h.parts.2.1

/-- `draft_token_ids` is valid in the initial memory. -/
theorem Pre.draft_valid {d : Inst} {init : Mem} (h : Pre d init) : d.draft.Valid init :=
  h.parts.2.2.1

/-- `target_argmax` is valid in the initial memory. -/
theorem Pre.target_valid {d : Inst} {init : Mem} (h : Pre d init) : d.target.Valid init :=
  h.parts.2.2.2.1

/-- `bonus_token_ids` is valid in the initial memory. -/
theorem Pre.bonus_valid {d : Inst} {init : Mem} (h : Pre d init) : d.bonus.Valid init :=
  h.parts.2.2.2.2.1

/-- The `is_greedy` mask, when given, is valid in the initial memory. -/
theorem Pre.greedy_valid {d : Inst} {init : Mem} (h : Pre d init) {g : Buf .i1}
    (hg : d.greedy = some g) : g.Valid init :=
  h.parts.2.2.2.2.2.1 g hg

/-- The first running sum is nonnegative. -/
theorem Pre.cu0 {d : Inst} {init : Mem} (h : Pre d init) : 0 ≤ rd32 d.cu init 0 :=
  h.parts.2.2.2.2.2.2.1

/-- The running sums do not decrease from `r` to `r + 1`. -/
theorem Pre.cu_step {d : Inst} {init : Mem} (h : Pre d init) {r : Nat} (hr : r + 1 < d.B) :
    rd32 d.cu init ((r : Nat) : Int) ≤ rd32 d.cu init ((r + 1 : Nat) : Int) :=
  h.parts.2.2.2.2.2.2.2.1 r hr

/-- The last running sum is within the draft buffer. -/
theorem Pre.cu_draft {d : Inst} {init : Mem} (h : Pre d init) :
    rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.draft.len :=
  h.parts.2.2.2.2.2.2.2.2.1

/-- The last running sum is within the target buffer. -/
theorem Pre.cu_target {d : Inst} {init : Mem} (h : Pre d init) :
    rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.target.len :=
  h.parts.2.2.2.2.2.2.2.2.2.1

/-- Every request has at most `S` drafts. -/
theorem Pre.nOf_le {d : Inst} {init : Mem} (h : Pre d init) {r : Nat} (hr : r < d.B) :
    nOf d init r ≤ d.S :=
  h.parts.2.2.2.2.2.2.2.2.2.2.1 r hr

/-- The last running sum is at most `2^22`. -/
theorem Pre.cu_22 {d : Inst} {init : Mem} (h : Pre d init) :
    rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ 2 ^ 22 :=
  h.parts.2.2.2.2.2.2.2.2.2.2.2

/-! ## Numbers from `Inst.ok` -/

/-- `max_spec_len` is an `int32`. -/
theorem S_lt {d : Inst} (hd : d.ok) : d.S < 2 ^ 31 := by
  have h1 : d.S + 1 ≤ d.B * (d.S + 1) := by
    calc d.S + 1 = 1 * (d.S + 1) := (Nat.one_mul _).symm
      _ ≤ d.B * (d.S + 1) := Nat.mul_le_mul_right _ hd.hB
  have h2 := hd.hmul
  omega

/-- There is at least one request. -/
theorem B_pos {d : Inst} (hd : d.ok) : 0 < d.B := hd.hB

/-! ## `cu` is nonnegative and nondecreasing -/

/-- The running sums are nondecreasing: `cu[a] ≤ cu[a + k]` for requests in the batch. -/
theorem cu_mono {d : Inst} {init : Mem} (hpre : Pre d init) :
    ∀ (i j : Nat), i ≤ j → j < d.B →
      rd32 d.cu init (i : Int) ≤ rd32 d.cu init (j : Int) := by
  intro i j
  induction j with
  | zero =>
    intro hij _
    have hi : i = 0 := Nat.le_zero.mp hij
    subst hi
    exact le_rfl
  | succ k ih =>
    intro hij hk
    rcases Nat.eq_or_lt_of_le hij with h | h
    · subst h
      exact le_rfl
    · have hik : i ≤ k := by omega
      have hkB : k < d.B := by omega
      exact le_trans (ih hik hkB) (hpre.cu_step (r := k) (by omega))

/-- Every running sum is nonnegative. -/
theorem cu_nonneg {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    0 ≤ rd32 d.cu init (r : Int) :=
  le_trans hpre.cu0 (cu_mono hpre 0 r (Nat.zero_le _) hr)

/-- Every running sum is at most the last one. -/
theorem cu_le_last {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    rd32 d.cu init (r : Int) ≤ rd32 d.cu init ((d.B - 1 : Nat) : Int) :=
  cu_mono hpre r (d.B - 1) (by omega) (by omega)

/-- Every running sum is within the draft buffer. -/
theorem cu_le_draft {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    rd32 d.cu init (r : Int) ≤ (d.draft.len : Int) :=
  le_trans (cu_le_last hpre hr) hpre.cu_draft

/-- Every running sum is within the target buffer. -/
theorem cu_le_target {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    rd32 d.cu init (r : Int) ≤ (d.target.len : Int) :=
  le_trans (cu_le_last hpre hr) hpre.cu_target

/-- Every running sum is at most `2^22`. -/
theorem cu_le_22 {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    rd32 d.cu init (r : Int) ≤ 2 ^ 22 :=
  le_trans (cu_le_last hpre hr) hpre.cu_22

/-! ## `startOf` and `nOf` -/

/-- A request starts at a nonnegative position. -/
theorem startOf_nonneg {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    0 ≤ startOf d init r := by
  rcases Nat.eq_zero_or_pos r with h | h
  · subst h
    simp [startOf]
  · have hrw : startOf d init r = rd32 d.cu init ((r - 1 : Nat) : Int) := by
      simp [startOf, Nat.pos_iff_ne_zero.mp h]
    rw [hrw]
    exact le_trans hpre.cu0 (cu_mono hpre 0 (r - 1) (Nat.zero_le _) (by omega))

/-- A request has a nonnegative number of drafts. -/
theorem nOf_nonneg {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    0 ≤ nOf d init r := by
  rcases Nat.eq_zero_or_pos r with h | h
  · subst h
    have h0 : startOf d init 0 = 0 := by simp [startOf]
    simp only [nOf, h0]
    simpa using hpre.cu0
  · have hrw : startOf d init r = rd32 d.cu init ((r - 1 : Nat) : Int) := by
      simp [startOf, Nat.pos_iff_ne_zero.mp h]
    have := cu_mono hpre (r - 1) r (by omega) hr
    simp only [nOf, hrw]
    omega

/-- A request's start plus its draft count is its running sum. -/
theorem startOf_add_nOf {d : Inst} (init : Mem) (r : Nat) :
    startOf d init r + nOf d init r = rd32 d.cu init (r : Int) := by
  simp only [nOf]
  ring

/-- A request's start is at most its start plus its draft count. -/
theorem startOf_le_startOf_add {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) : startOf d init r ≤ rd32 d.cu init (r : Int) := by
  have h := startOf_add_nOf (d := d) init r
  have := nOf_nonneg hpre hr
  omega

/-- A request's start is at most the draft buffer's length. -/
theorem startOf_lt_draft {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    startOf d init r ≤ (d.draft.len : Int) :=
  le_trans (startOf_le_startOf_add hpre hr) (cu_le_draft hpre hr)

/-- A request's start is at most the target buffer's length. -/
theorem startOf_lt_target {d : Inst} {init : Mem} (hpre : Pre d init) {r : Nat} (hr : r < d.B) :
    startOf d init r ≤ (d.target.len : Int) :=
  le_trans (startOf_le_startOf_add hpre hr) (cu_le_target hpre hr)

/-- A request's draft count fits in `int32`. -/
theorem nOf_lt_2p31 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) : nOf d init r < 2 ^ 31 := by
  have h1 := hpre.nOf_le hr
  have h2 := S_lt hd
  omega

/-! ## The loop bounds -/

/-- `range(0, n)` runs `n.toNat` trips. -/
theorem tripCount_zero (n : Int) : tl.tripCount 0 n 1 = n.toNat := by
  simp [tl.tripCount]

/-- The `int32` constant `0` has value `0`. -/
theorem i32_zero_toInt : (0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)

/-- The `int32` constant `1` has value `1`. -/
theorem i32_one_toInt : (1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by norm_num)

/-- `range(num)` for a nonnegative `int32` count is within `tl.RangeOk`. -/
theorem rangeOk_zero_num {num : I32} (h0 : 0 ≤ num.toInt) (hlt : num.toInt < 2 ^ 31) :
    tl.RangeOk (0 : I32) num (1 : I32) := by
  have hz := i32_zero_toInt
  have ho := i32_one_toInt
  have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
    simp [IntV.maxInt]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [hmax, hz, ho] <;> omega

/-! ## Addresses -/

/-- `b.ptr 0 + r` is element `r` of `b`, for an `int32` `r`. -/
theorem buf_ptr_add {d' : DType} (b : Buf d') {r : Nat} (hr : r < 2 ^ 31) :
    b.ptr 0 + (IntV.ofNat r : I32) = b.ptr (r : Int) := by
  rw [Ptr.add_int, IntV.I32.toInt_ofNat hr, Buf.ptr_offset, Int.zero_add]

/-- `cu.ptr 0 + r - 1` is element `r - 1` of `cu`, for `r > 0`. -/
theorem cu_ptr_sub {d : Inst} {r : Nat} (hr0 : 0 < r) (hr : r < 2 ^ 31) :
    d.cu.ptr 0 + (IntV.ofNat r : I32) - (1 : I32) = d.cu.ptr ((r - 1 : Nat) : Int) := by
  have ho : (1 : I32).toInt = 1 := i32_one_toInt
  rw [Ptr.sub_int_of_signed (p := d.cu.ptr 0 + (IntV.ofNat r : I32)) (k := (1 : I32))
    (by rw [ho]; norm_num), ho,
    Ptr.add_int, IntV.I32.toInt_ofNat hr, Ptr.offset_offset, Buf.ptr_offset]
  congr 1
  omega

/-- `draft.ptr 0 + start_idx + pos` is element `start + t` of `draft`. -/
theorem draft_ptr_eq {d : Inst} {st : Int} {start_idx : I32} (hst : start_idx.toInt = st)
    {p : Int} {pos : I32} (hp : pos.toInt = p) :
    d.draft.ptr 0 + start_idx + pos = d.draft.ptr (st + p) := by
  rw [Ptr.add_int, Ptr.add_int, hst, hp, Buf.ptr_offset, Buf.ptr_offset, Int.zero_add]

/-- `target.ptr 0 + start_idx + pos` is element `start + t` of `target`. -/
theorem target_ptr_eq {d : Inst} {st : Int} {start_idx : I32} (hst : start_idx.toInt = st)
    {p : Int} {pos : I32} (hp : pos.toInt = p) :
    d.target.ptr 0 + start_idx + pos = d.target.ptr (st + p) := by
  rw [Ptr.add_int, Ptr.add_int, hst, hp, Buf.ptr_offset, Buf.ptr_offset, Int.zero_add]

/-- The store address `out.ptr 0 + row offset + pos` is element `oidx d r t` of the output. -/
theorem out_ptr_eq {d : Inst} {r t : Nat}
    (hro : (rowOff d r).toInt = ((r * (d.S + 1) : Nat) : Int))
    {pos : I32} (hp : pos.toInt = (t : Int)) :
    d.out.ptr 0 + rowOff d r + pos = d.out.ptr (oidx d r t : Int) := by
  have hidx : ((r * (d.S + 1) : Nat) : Int) + (t : Int) = (oidx d r t : Int) := by
    simp only [oidx]
    push_cast
    ring
  rw [Ptr.add_int, Ptr.add_int, hro, hp, Ptr.offset_offset, hidx, Buf.ptr_offset, Int.zero_add]

/-- `bonus.ptr 0 + r` is element `r` of `bonus`. -/
theorem bonus_ptr_eq {d : Inst} {r : Nat} (hr : r < 2 ^ 31) :
    d.bonus.ptr 0 + (IntV.ofNat r : I32) = d.bonus.ptr (r : Int) :=
  buf_ptr_add d.bonus hr

/-! ## The row offset does not wrap -/

/-- The `int32` row offset `r * (S + 1)` does not wrap. -/
theorem rowOff_toInt {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    (rowOff d r).toInt = ((r * (d.S + 1) : Nat) : Int) := by
  rcases Nat.lt_or_ge (d.S + 1) (2 ^ 31) with hS | hS
  · have hrB : r + 1 ≤ d.B := by omega
    have hmul := hd.hmul
    have hrS : r * (d.S + 1) < 2 ^ 31 := by
      have h3 : r * (d.S + 1) + (d.S + 1) = (r + 1) * (d.S + 1) := by ring
      have h4 : (r + 1) * (d.S + 1) ≤ d.B * (d.S + 1) := Nat.mul_le_mul_right _ hrB
      omega
    have hr31 : r < 2 ^ 31 :=
      lt_of_le_of_lt (Nat.le_mul_of_pos_right r (by omega : 0 < d.S + 1)) hrS
    have hS1 : (IntV.ofNat d.S : I32).toInt = (d.S : Int) := IntV.I32.toInt_ofNat (by omega)
    have h11 : (1 : I32).toInt = 1 := i32_one_toInt
    have hr' : (IntV.ofNat r : I32).toInt = (r : Int) := IntV.I32.toInt_ofNat hr31
    have hsum_fit : IntV.SFits 32 ((d.S : Int) + 1) := by
      simp only [IntV.SFits]
      omega
    have hsum : (IntV.ofNat d.S + (1 : I32)).toInt = (d.S : Int) + 1 := by
      rw [IntV.toInt_add hS1 h11 hsum_fit]
    have hprod_fit :
        IntV.SFits 32 ((r : Int) * ((d.S : Int) + 1)) := by
      simp only [IntV.SFits]
      have hcast : (r : Int) * ((d.S : Int) + 1) = ((r * (d.S + 1) : Nat) : Int) := by
        push_cast
        ring
      rw [hcast]
      omega
    show ((IntV.ofNat r : I32) * (IntV.ofNat d.S + (1 : I32))).toInt = _
    rw [IntV.toInt_mul hr' hsum hprod_fit]
    push_cast
    ring
  · have hB1 : d.B = 1 := by
      have h1 : d.B * (d.S + 1) ≤ 1 * (d.S + 1) := le_trans hd.hmul (by omega)
      exact le_antisymm (Nat.le_of_mul_le_mul_right h1 (Nat.succ_pos d.S)) hd.hB
    have hr0 : r = 0 := by omega
    subst hr0
    have hz : (IntV.ofNat 0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)
    have hfit : IntV.SFits 32 ((0 : Int) * (IntV.ofNat d.S + (1 : I32)).toInt) := by
      simp only [IntV.SFits, Int.zero_mul]
      omega
    show ((IntV.ofNat 0 : I32) * (IntV.ofNat d.S + (1 : I32))).toInt = _
    rw [IntV.toInt_mul hz rfl hfit]
    simp

/-! ## Scalar accesses -/

/-- The bytes a scalar access touches: the element at its address (`TVA.accBytes_scalar`). -/
theorem accBytes_scalar {d' : DType} (p : Ptr d') :
    accBytes d' (tl.addrs (Tensor.scalar p) none) = Mem.span d' p.addr :=
  TVA.accBytes_scalar d' p.addr

/-- A scalar store is allowed exactly when its element is writable. -/
theorem storeOk_scalar {m : Mem} {d' : DType} (p : Ptr d') :
    StoreOk m d' (tl.addrs (Tensor.scalar p) none) ↔ m.Writable d' p.addr := by
  constructor
  · intro h
    exact h () p.addr rfl
  · intro h i a ha
    simp only [tl.addrs, Tensor.scalar, Option.some.injEq] at ha
    rw [← ha]
    exact h

/-- A scalar access has distinct lanes. -/
theorem distinctLanes_scalar {d' : DType} (p : Ptr d') :
    DistinctLanes d' (tl.addrs (Tensor.scalar p) none) := by
  intro i j a b hij _ _
  exact absurd (by cases i; cases j; rfl) hij

/-- After a scalar store the element holds the value and nothing else changed. -/
theorem storeRes_scalar {m m' : Mem} {d' : DType} {p : Ptr d'} {v : d'.Val}
    (h : StoreRes m m' (Tensor.scalar p) (Tensor.scalar v) none) :
    m'.read d' p.addr = some v ∧ ∀ x, x ∉ Mem.span d' p.addr → m' x = m x := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨?_, ?_⟩
  · have hh := h1 () p.addr (by simp [tl.addrs, Tensor.scalar])
    simpa [Tensor.scalar] using hh
  · intro x hx
    exact h2 x (by rw [accBytes_scalar]; exact hx)

/-! ## Memory helpers -/

/-- `rd32` of an element that memory reads as `v` is `v.toInt`. -/
theorem rd32_eq {b : Buf .i32} {m : Mem} {i : Int} {v : I32}
    (h : m.read .i32 (b.ptr i).addr = some v) : rd32 b m i = v.toInt := by
  rw [Buf.ptr_addr] at h
  simp [rd32, h]

/-- `rdt64` of an element that memory reads as `v` is `v.toInt` truncated to `int32`. -/
theorem rdt64_eq {b : Buf .i64} {m : Mem} {i : Int} {v : IntV true 64}
    (h : m.read .i64 (b.ptr i).addr = some v) : rdt64 b m i = Int.bmod v.toInt (2 ^ 32) := by
  rw [Buf.ptr_addr] at h
  simp [rdt64, h]

/-- `greedyAt` of a mask element that memory reads as `v` is `v`. -/
theorem greedyAt_eq {d : Inst} {g : Buf .i1} (hg : d.greedy = some g) {m : Mem} {r : Nat}
    {v : Bool} (h : m.read .i1 (g.ptr (r : Int)).addr = some v) : greedyAt d m r = v := by
  rw [Buf.ptr_addr] at h
  simp [greedyAt, hg, h]

end Claims.RejectionGreedy
