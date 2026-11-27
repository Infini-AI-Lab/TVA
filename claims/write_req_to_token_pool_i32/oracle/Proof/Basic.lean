import Claim
import TVA.Proof

/-!
# `write_req_to_token_pool_i32`: shared definitions

The launch, the footprints of one program of the grid, its post-condition, and the element sets
of the two copy loops (a window of `L` elements of a source buffer copied, as `int32` casts, to a
window of the table).
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-! ## The launch -/

/-- The grid has one program per request. -/
theorem launch_size (d : Inst) : (launch d).size = d.bs := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid: fewer than `2^31` programs, block shape `[512]`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt ?_
  intro s hs
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨9, by norm_num⟩
  · norm_num

/-! ## Windows -/

/-- The window of `L` elements from element `a`: `a, …, a + L - 1`. -/
def win (a L : Int) : Set Nat := {e | ∃ o : Nat, (o : Int) < L ∧ (e : Int) = a + o}

/-- The elements of the window that trip `t` of a copy loop covers: offsets `t·512 … t·512 + 511`
below `L`. -/
def tripWin (a L : Int) (t : Nat) : Set Nat :=
  {e | ∃ l : Nat, l < 512 ∧ (t : Int) * 512 + l < L ∧ (e : Int) = a + ((t : Int) * 512 + l)}

theorem tripWin_sub (a L : Int) (t : Nat) : tripWin a L t ⊆ win a L := by
  rintro e ⟨l, -, hl, he⟩
  exact ⟨t * 512 + l, by push_cast; exact hl, by push_cast; exact he⟩

/-- Distinct trips cover disjoint elements. -/
theorem tripWin_disjoint {a L : Int} {t u : Nat} (htu : t ≠ u) :
    Disjoint (tripWin a L t) (tripWin a L u) := by
  rw [Set.disjoint_left]
  rintro e ⟨l, hl, -, he⟩ ⟨l', hl', -, he'⟩
  have h : (t : Int) * 512 + l = (u : Int) * 512 + l' := by linarith
  have hl1 : (l : Int) < 512 := by exact_mod_cast hl
  have hl2 : (l' : Int) < 512 := by exact_mod_cast hl'
  have hl3 : (0 : Int) ≤ l := by positivity
  have hl4 : (0 : Int) ≤ l' := by positivity
  rcases Nat.lt_or_gt_of_ne htu with h1 | h1
  · have : (t : Int) + 1 ≤ u := by exact_mod_cast h1
    nlinarith
  · have : (u : Int) + 1 ≤ t := by exact_mod_cast h1
    nlinarith

/-- A window lies in a buffer when each of its elements does. -/
theorem win_lt {a L : Int} {n : Nat} (h : ∀ o : Nat, (o : Int) < L → 0 ≤ a + o ∧ a + o < n) :
    ∀ e ∈ win a L, e < n := by
  rintro e ⟨o, ho, he⟩
  have := (h o ho).2
  rw [← he] at this
  exact_mod_cast this

/-! ## One program's footprints and result -/

/-- The table elements program `b` writes: its row's positions `0 … seq_lens[b] - 1`. -/
def rowElems (d : Inst) (init : Mem) (b : Nat) : Set Nat :=
  win (rpi d init b * (d.max_context_len : Int)) (seq d init b)

/-- The read footprint of program `b`: its entries of the index buffers and of the prefix
pointers, `extend_lens[0 … b - 1]`, its prefix and its window of `out_cache_loc`. -/
def R (d : Inst) (init : Mem) (b : Nat) : Set Int :=
  d.req_pool_indices.elems {b} ∪ d.prefix_pointers.elems {b} ∪ d.prefix_lens.elems {b} ∪
    d.seq_lens.elems {b} ∪ d.extend_lens.elems {i | i < b} ∪
    (d.prefixT b).elems (win 0 (pre d init b)) ∪
    d.out_cache_loc.elems (win (cum d init b) (seq d init b - pre d init b))

/-- The write footprint of program `b`. -/
def W (d : Inst) (init : Mem) (b : Nat) : Set Int := d.req_to_token.elems (rowElems d init b)

/-- `j` of a window copied from `S` at `s0` to the table at `t0`: the table element holds the
cast of the source element's initial value. -/
def Copied (d : Inst) (init m : Mem) (S : Buf .i64) (s0 t0 : Int) (o : Nat) : Prop :=
  ∃ v : I64, init.read .i64 (S.ptr (s0 + o)).addr = some v ∧
    m.read .i32 (d.req_to_token.ptr (t0 + o)).addr = some (IntV.cast v)

/-- What program `b` leaves: its row's prefix positions hold its prefix, the others its new
locations. -/
def Q (d : Inst) (init : Mem) (b : Nat) (m : Mem) : Prop :=
  (∀ o : Nat, (o : Int) < pre d init b →
      Copied d init m (d.prefixT b) 0 (rpi d init b * (d.max_context_len : Int)) o) ∧
    ∀ o : Nat, (o : Int) < seq d init b - pre d init b →
      Copied d init m d.out_cache_loc (cum d init b)
        (rpi d init b * (d.max_context_len : Int) + pre d init b) o

/-! ## Helpers -/

/-- `b.ptr 0 + k` is `b.ptr k`. -/
theorem ptr_zero_add {d : DType} {sg : Bool} {w : Nat} (b : Buf d) (k : IntV sg w) :
    (b.ptr 0 + k) = b.ptr k.toInt := by
  simp [Buf.ptr, Ptr.offset]

/-- The value of an `int64` element read from memory. -/
theorem valOf_of_read {m : Mem} {b : Buf .i64} {i : Nat} {v : I64}
    (h : m.read .i64 (b.ptr (i : Int)).addr = some v) : valOf m b i = v.toInt := by
  simp only [valOf, h]

/-- The value of a `uint64` element read from memory. -/
theorem uvalOf_of_read {m : Mem} {b : Buf .u64} {i : Nat} {v : U64}
    (h : m.read .u64 (b.ptr (i : Int)).addr = some v) : uvalOf m b i = v.toInt := by
  simp only [uvalOf, h]

/-- The bytes of a buffer element below `len` are bytes of the buffer. -/
theorem span_sub_bytes {d : DType} (b : Buf d) {i : Int} (h0 : 0 ≤ i) (h1 : i < b.len) :
    Mem.span d (b.ptr i).addr ⊆ b.bytes := by
  have : i = ((i.toNat : Nat) : Int) := (Int.toNat_of_nonneg h0).symm
  rw [this]
  exact b.span_sub (by omega)

/-- A read from a buffer disjoint from the table is the same in a memory that agrees with `init`
outside the table. -/
theorem read_init {d : Inst} {dt : DType} {m init : Mem} (b : Buf dt)
    (hdisj : Disjoint d.req_to_token.bytes b.bytes)
    (hfr : ∀ z, z ∉ d.req_to_token.bytes → m z = init z) {i : Int} (h0 : 0 ≤ i) (h1 : i < b.len) :
    m.read dt (b.ptr i).addr = init.read dt (b.ptr i).addr :=
  Mem.read_congr fun x hx =>
    hfr x fun hx' => Set.disjoint_left.mp hdisj hx' (span_sub_bytes b h0 h1 hx)

/-- A buffer disjoint from the table stays valid in a memory that agrees with `init` outside the
table. -/
theorem valid_init {d : Inst} {dt : DType} {m init : Mem} (b : Buf dt)
    (hdisj : Disjoint d.req_to_token.bytes b.bytes)
    (hfr : ∀ z, z ∉ d.req_to_token.bytes → m z = init z) (hv : b.Valid init) : b.Valid m :=
  Buf.valid_of_agree hv fun x hx => by
    rw [hfr x fun hx' => Set.disjoint_left.mp hdisj hx' hx]
    exact hv.2.2.2 x hx

end Claims.WriteReqToTokenPool
