import Claim
import TVA.Proof

/-!
# `combine_draft_tokens_i32`: shared definitions

The launch, the footprints of one program of the grid, its post-condition, and the facts about
the running sums `cu_num_logits` and `query_start_loc` that the footprints need.
-/

namespace Claims.CombineDraftTokens

open TVA

/-! ## The launch -/

/-- The grid has one program per request. -/
theorem launch_size (d : Inst) : (launch d).size = d.B := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid: fewer than `2^31` programs, block shape `[BLOCK_SIZE]`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.hgrid ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  obtain ⟨e, he, hb⟩ := hd.hblock
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨e, hb⟩
  · simp only [List.prod_cons, List.prod_nil, mul_one, hb]
    exact Nat.pow_le_pow_right (by norm_num) he

/-! ## Footprints and post-condition -/

/-- The read footprint of every program: the bytes of the seven inputs. -/
def R (d : Inst) : Set Int :=
  d.idx_mapping.bytes ∪ d.last_sampled.bytes ∪ d.query_start_loc.bytes ∪ d.seq_lens.bytes ∪
    d.prefill_len.bytes ∪ d.draft_tokens.bytes ∪ d.cu_num_logits.bytes

/-- The elements of `logits_indices` program `b` writes. -/
def logE (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  {t | ∃ j : Nat, (j : Int) < nlog d init b ∧ (t : Int) = cuAt d init b + j}

/-- The element of `input_ids` program `b` writes the last sampled token to (when `LastOn`). -/
def lastE (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  {t | LastOn d init b ∧ (t : Int) = qend d init b - nlog d init b}

/-- The elements of `input_ids` program `b` writes the drafts to (when past its prefill). -/
def draftE (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  {t | Decode d init b ∧ ∃ j : Nat, (j : Int) < ndraft d init b ∧
    (t : Int) = qend d init b - ndraft d init b + j}

/-- The write footprint of program `b`. -/
def W (d : Inst) (init : Mem) (b : Nat) : Set Int :=
  d.logits_indices.elems (logE d init b) ∪
    (d.input_ids.elems (lastE d init b) ∪ d.input_ids.elems (draftE d init b))

/-- The logits indices of request `b` in `m`. -/
def QL (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  ∀ j : Nat, (j : Int) < nlog d init b → ∃ w : I64,
    m.read .i64 (d.logits_indices.ptr (cuAt d init b + j)).addr = some w ∧
    w.toInt = qend d init b - nlog d init b + j

/-- The last sampled token of request `b` in `m`. -/
def QA (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  LastOn d init b → ∃ w : I32,
    m.read .i32 (d.input_ids.ptr (qend d init b - nlog d init b)).addr = some w ∧
    w.toInt = rdt64 d.last_sampled init (ridx d init b)

/-- The draft tokens of request `b` in `m`. -/
def QD (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  Decode d init b → ∀ j : Nat, (j : Int) < ndraft d init b → ∃ w : I32,
    m.read .i32 (d.input_ids.ptr (qend d init b - ndraft d init b + j)).addr = some w ∧
    w.toInt = rdt64 d.draft_tokens init (ridx d init b * d.dstride + j)

/-- What program `b` leaves. -/
def Q (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  QL d init b m ∧ QA d init b m ∧ QD d init b m

/-! ## Pointers -/

/-- `b.ptr 0 + x` is the pointer to element `x.toInt`. -/
theorem ptr_add {d : DType} {sg : Bool} {w : Nat} (b : Buf d) (x : IntV sg w) {v : Int}
    (h : x.toInt = v) : b.ptr 0 + x = b.ptr v := by
  simp [Ptr.add_int, Buf.ptr_offset, h]

/-- `(b.ptr 0 + x) + y` is the pointer to element `x.toInt + y.toInt`. -/
theorem ptr_add_add {d : DType} {sg sg' : Bool} {w w' : Nat} (b : Buf d) (x : IntV sg w)
    (y : IntV sg' w') {v u : Int} (hx : x.toInt = v) (hy : y.toInt = u) :
    b.ptr 0 + x + y = b.ptr (v + u) := by
  simp [Ptr.add_int, Buf.ptr_offset, hx, hy]

/-- The value of the `int32` constant `1`. -/
theorem toInt_one32 : ((1 : I32)).toInt = 1 := IntV.toInt_ofNat_lit (by norm_num)

/-- The value of the `int32` constant `0`. -/
theorem toInt_zero32 : ((0 : I32)).toInt = 0 := IntV.toInt_ofNat_lit (by norm_num)

/-- The `int32` truncation of an `int64` value, as `ConvRel` gives it. -/
theorem convRel_i64_i32_bmod {P : Platform} {x : I64} {r : I32} (h : ConvRel P .i64 .i32 x r) :
    r.toInt = Int.bmod x.toInt (2 ^ 32) := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']
  simp [IntV.ofInt, IntV.toInt, BitVec.toInt_ofInt]

/-! ## Reads -/

/-- A read of an element of a buffer whose bytes `m` keeps from `init`. -/
theorem read_keep {d : DType} {b : Buf d} {m init : Mem} (h : ∀ x ∈ b.bytes, m x = init x)
    {j : Nat} (hj : j < b.len) : m.read d (b.ptr (j : Int)).addr = init.read d (b.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx => h x (b.span_sub hj hx)

/-- `rd32` of an element read as `v`. -/
theorem rd32_of {b : Buf .i32} {m : Mem} {i : Int} {v : I32}
    (h : m.read .i32 (b.ptr i).addr = some v) : rd32 b m i = v.toInt := by
  simp only [rd32, h]

/-- `rd64` of an element read as `v`. -/
theorem rd64_of {b : Buf .i64} {m : Mem} {i : Int} {v : I64}
    (h : m.read .i64 (b.ptr i).addr = some v) : rd64 b m i = v.toInt := by
  simp only [rd64, h]

/-- An `int32` element's value fits `int32`. -/
theorem rd32_bounds (b : Buf .i32) (m : Mem) (i : Int) :
    -(2 ^ 31 : Int) ≤ rd32 b m i ∧ rd32 b m i < 2 ^ 31 := by
  unfold rd32
  split
  · rename_i v _
    have := IntV.sfits_toInt v
    simp only [IntV.SFits] at this
    norm_num at this
    exact this
  · norm_num

/-! ## The running sums -/

section Chains

variable {d : Inst} {init : Mem}

/-- `cu_num_logits` is nondecreasing on `[0, B]`. -/
theorem cu_mono (hpre : Pre d init) {b b' : Nat} (hbb : b ≤ b') (hb' : b' ≤ d.B) :
    cuAt d init b ≤ cuAt d init b' := by
  induction b', hbb using Nat.le_induction with
  | base => exact le_refl _
  | succ n hn ih =>
    have h := (hpre.2.2.2.2.2.2.2.2.2.1 n (by omega)).2.2.2.2.1
    have := ih (by omega)
    simp only [nlog] at h
    have hN : (0 : Int) ≤ d.N := by positivity
    linarith

/-- `query_start_loc` is nondecreasing on `[0, B]`. -/
theorem qsl_mono (hpre : Pre d init) {b b' : Nat} (hbb : b ≤ b') (hb' : b' ≤ d.B) :
    qsl d init b ≤ qsl d init b' := by
  induction b', hbb using Nat.le_induction with
  | base => exact le_refl _
  | succ n hn ih =>
    have h1 := (hpre.2.2.2.2.2.2.2.2.2.1 n (by omega)).2.2.2.2.1
    have h2 := (hpre.2.2.2.2.2.2.2.2.2.1 n (by omega)).2.2.2.2.2.2.1
    have := ih (by omega)
    simp only [qend] at h2
    have hN : (0 : Int) ≤ d.N := by positivity
    linarith

/-- The bounds of request `b`'s logits: `0 ≤ cu[b]`, `cu[b + 1] ≤ len(logits_indices)`. -/
theorem cu_bounds (hpre : Pre d init) {b : Nat} (hb : b < d.B) :
    0 ≤ cuAt d init b ∧ cuAt d init (b + 1) ≤ d.logits_indices.len := by
  have h0 := cu_mono hpre (Nat.zero_le b) (by omega)
  have h1 := cu_mono hpre (show b + 1 ≤ d.B by omega) (le_refl _)
  exact ⟨by linarith [hpre.2.2.2.2.2.2.2.2.2.2.1], by linarith [hpre.2.2.2.2.2.2.2.2.2.2.2.1]⟩

/-- The bounds of request `b`'s query rows: `0 ≤ qsl[b]`, `qsl[b + 1] ≤ len(input_ids)`. -/
theorem qsl_bounds (hpre : Pre d init) {b : Nat} (hb : b < d.B) :
    0 ≤ qsl d init b ∧ qend d init b ≤ d.input_ids.len := by
  have h0 := qsl_mono hpre (Nat.zero_le b) (by omega)
  have h1 := qsl_mono hpre (show b + 1 ≤ d.B by omega) (le_refl _)
  exact ⟨by linarith [hpre.2.2.2.2.2.2.2.2.2.2.2.2.1],
    by simp only [qend]; linarith [hpre.2.2.2.2.2.2.2.2.2.2.2.2.2]⟩

end Chains

/-! ## The inputs in a later memory -/

/-- Each input's bytes lie in `R`. -/
theorem sub_R_idx (d : Inst) : d.idx_mapping.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto
theorem sub_R_last (d : Inst) : d.last_sampled.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto
theorem sub_R_qsl (d : Inst) : d.query_start_loc.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto
theorem sub_R_seq (d : Inst) : d.seq_lens.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto
theorem sub_R_pl (d : Inst) : d.prefill_len.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto
theorem sub_R_draft (d : Inst) : d.draft_tokens.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto
theorem sub_R_cu (d : Inst) : d.cu_num_logits.bytes ⊆ R d := fun _ h => by
  simp only [R, Set.mem_union]; tauto

/-- An input valid in `init` is valid in a memory that keeps `R`. -/
theorem valid_keep {dt : DType} {b : Buf dt} {d : Inst} {init m : Mem} (hv : b.Valid init)
    (hsub : b.bytes ⊆ R d) (hR : ∀ x ∈ R d, m x = init x) : b.Valid m :=
  Buf.valid_of_agree hv fun x hx => by rw [hR x (hsub hx)]; exact hv.2.2.2 x hx

/-- Lane `i` of `tl.arange(0, BLOCK)` has value `i`. -/
theorem arange_toInt {n : Nat} (hn : n ≤ 2 ^ 31) (i : Idx [n - 0]) :
    ((tl.arange 0 n).get i).toInt = ((i.1 : Nat) : Int) := by
  rcases i with ⟨i, ⟨⟩⟩
  exact IntV.I32.toInt_tl_arange hn i

/-- `BLOCK_SIZE ≤ 2^20`. -/
theorem block_le {d : Inst} (hd : d.ok) : d.BLOCK ≤ 2 ^ 20 := by
  obtain ⟨e, he, hb⟩ := hd.hblock
  rw [hb]; exact Nat.pow_le_pow_right (by norm_num) he

/-- `SFits 32` from two bounds. -/
theorem sf32 {x : Int} (h0 : -(2 ^ 31 : Int) ≤ x) (h1 : x < 2 ^ 31) : IntV.SFits 32 x :=
  IntV.I32.sfits h0 h1

/-- `SFits 64` from two bounds. -/
theorem sf64 {x : Int} (h0 : -(2 ^ 63 : Int) ≤ x) (h1 : x < 2 ^ 63) : IntV.SFits 64 x :=
  IntV.I64.sfits h0 h1

/-- A valid buffer of `int64` elements has at most `2^61` elements. -/
theorem len_lt_of_valid64 {b : Buf .i64} {m : Mem} (hv : b.Valid m) : (b.len : Int) ≤ 2 ^ 61 := by
  obtain ⟨h0, h1, -, -⟩ := hv
  simp only [DType.bytes] at h1
  push_cast at h1
  norm_num at h1 ⊢
  linarith

/-- A valid buffer of `int32` elements has at most `2^62` elements. -/
theorem len_lt_of_valid32 {b : Buf .i32} {m : Mem} (hv : b.Valid m) : (b.len : Int) ≤ 2 ^ 62 := by
  obtain ⟨h0, h1, -, -⟩ := hv
  simp only [DType.bytes] at h1
  push_cast at h1
  norm_num at h1 ⊢
  linarith

end Claims.CombineDraftTokens
