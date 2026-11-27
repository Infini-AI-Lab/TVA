import Claim
import TVA.Proof

/-!
# `slot_mappings_i64`: the launch, the footprints, small facts

Program number `i` is `(g, b) = (i % G, i / G)`: group `g`, and request `b < R` or the padding
program `b = R`. Every program reads only input bytes (`Rall`, all of them) and writes the
elements `(g, t)` of its token range `[qsl b, hiB b)` (`Wp`), where they hold `val` (`Qp`).
-/

namespace Claims.SlotMappings

open TVA

/-! ## The launch -/

theorem launch_size (d : Inst) : (launch d).size = d.G * (d.R + 1) := by
  simp [Launch.size, launch]

theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = i % d.G := by
  simp [Launch.ctx, launch]

theorem launch_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = i / d.G % (d.R + 1) := by
  simp [Launch.ctx, launch]

theorem launch_grid1 (d : Inst) (i : Nat) : ((launch d).ctx i).grid 1 = d.R + 1 := by
  simp [Launch.ctx, launch]

/-- The launch is valid: the grid `(G, R + 1, 1)` within CUDA's limits, block shape `[1024]`. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have h1 := hd.G_lt
  have h2 := hd.R_le
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · simpa [launch] using h1
  · simp [launch]; omega
  · simp [launch]
  · rw [launch_size]; exact hd.grid_lt
  · intro s hs
    simp only [launch, List.mem_singleton] at hs
    subst hs
    refine ⟨fun k hk => ?_, by norm_num⟩
    simp only [List.mem_singleton] at hk
    exact ⟨10, by rw [hk]; norm_num⟩

/-- Program `i`'s coordinates: `i % G < G` and `i / G ≤ R`. -/
theorem prog_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (i : Nat) % d.G < d.G ∧ (i : Nat) / d.G < d.R + 1 := by
  have hi : (i : Nat) < d.G * (d.R + 1) := lt_of_lt_of_eq i.isLt (launch_size d)
  have hG := hd.G_pos
  refine ⟨Nat.mod_lt _ (by omega), ?_⟩
  exact Nat.div_lt_of_lt_mul hi

/-! ## Footprints -/

/-- The upper end of program `b`'s token range: `query_start_loc[b + 1]` for a request,
`max_num_tokens` for the padding program. -/
def hiB (d : Inst) (m : Mem) (b : Nat) : Int := if b < d.R then qsl d m (b + 1) else d.N

/-- The output elements `(g, t)` for `L ≤ t < H`. -/
def tok (d : Inst) (g : Nat) (L H : Int) : Set Nat :=
  {j | ∃ t : Nat, L ≤ (t : Int) ∧ (t : Int) < H ∧ j = g * d.S + t}

/-- Every input byte: what every program may read. -/
def Rall (d : Inst) : Set Int :=
  d.idx_mapping.bytes ∪ d.query_start_loc.bytes ∪ d.positions.bytes ∪ d.block_table_ptrs.bytes ∪
    d.block_table_strides.bytes ∪ d.block_sizes.bytes ∪ d.slot_mapping_enabled.bytes ∪
    d.dcp_sharded.bytes ∪ {z | ∃ g, g < d.G ∧ z ∈ (d.bt g).bytes}

theorem rall_idx {d : Inst} {z : Int} (h : z ∈ d.idx_mapping.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_qsl {d : Inst} {z : Int} (h : z ∈ d.query_start_loc.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_pos {d : Inst} {z : Int} (h : z ∈ d.positions.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_ptrs {d : Inst} {z : Int} (h : z ∈ d.block_table_ptrs.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_strides {d : Inst} {z : Int} (h : z ∈ d.block_table_strides.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_sizes {d : Inst} {z : Int} (h : z ∈ d.block_sizes.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_en {d : Inst} {z : Int} (h : z ∈ d.slot_mapping_enabled.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_sh {d : Inst} {z : Int} (h : z ∈ d.dcp_sharded.bytes) : z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; tauto
theorem rall_bt {d : Inst} {g : Nat} (hg : g < d.G) {z : Int} (h : z ∈ (d.bt g).bytes) :
    z ∈ Rall d := by
  simp only [Rall, Set.mem_union, Set.mem_ofPred_eq]; exact Or.inr ⟨g, hg, h⟩

/-- The write footprint of program `(g, b)`. -/
def Wp (d : Inst) (init : Mem) (g b : Nat) : Set Int :=
  d.slot_mappings.elems (tok d g (qsl d init b) (hiB d init b))

/-- What program `(g, b)` stores at token `t`. -/
def val (d : Inst) (init : Mem) (g b t : Nat) : I64 :=
  if b < d.R then IntV.ofInt (slot d init g b t) else PAD

/-- What program `(g, b)` leaves: each element of its range holds `val`. -/
def Qp (d : Inst) (init : Mem) (g b : Nat) (m : Mem) : Prop :=
  ∀ t : Nat, qsl d init b ≤ (t : Int) → (t : Int) < hiB d init b →
    m.read .i64 (d.slot_mappings.ptr ((g * d.S + t : Nat) : Int)).addr = some (val d init g b t)

/-- The memories a program runs in: the initial memory outside the output, the output valid. -/
def Inv (d : Inst) (init : Mem) (m : Mem) : Prop :=
  (∀ z, z ∉ d.slot_mappings.bytes → m z = init z) ∧ d.slot_mappings.Valid m

/-! ## Reading the initial memory -/

/-- A read of an input byte in a memory of `Inv` is the initial one. -/
theorem read_inv {d : Inst} {init m : Mem} (hm : Inv d init m) {dt : DType} {b : Buf dt}
    (hdisj : Disjoint d.slot_mappings.bytes b.bytes) {j : Nat} (hj : j < b.len) :
    m.read dt (b.ptr (j : Int)).addr = init.read dt (b.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx => hm.1 x fun hx' =>
    Set.disjoint_left.mp hdisj hx' (Buf.span_sub b hj hx)

/-- The value `rd64` reads, from a read. -/
theorem rd64_of {m : Mem} {b : Buf .i64} {i : Nat} {v : I64}
    (h : m.read .i64 (b.ptr (i : Int)).addr = some v) : rd64 m b i = v.toInt := by
  simp only [rd64, h]

/-- The value `rd32` reads, from a read. -/
theorem rd32_of {m : Mem} {b : Buf .i32} {i : Nat} {v : I32}
    (h : m.read .i32 (b.ptr (i : Int)).addr = some v) : rd32 m b i = v.toInt := by
  simp only [rd32, h]

/-- `b.ptr 0 + k` is `b.ptr k` for an `int32` scalar `k` of value `x`. -/
theorem ptr_zero_add {dt : DType} (b : Buf dt) {k : I32} {x : Int} (hk : k.toInt = x) :
    (b.ptr 0 + k) = b.ptr x := by
  simp [Buf.ptr, Ptr.offset, hk]

/-- An element below `len` of a buffer lies in its bytes. -/
theorem span_bytes {dt : DType} (b : Buf dt) {j : Nat} (hj : j < b.len) :
    Mem.span dt (b.ptr (j : Int)).addr ⊆ b.bytes := Buf.span_sub b hj

/-! ## `int32` constants -/

theorem toInt_ofNat32 {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n :=
  IntV.I32.toInt_ofNat h

theorem toInt_ofNat64 {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = n :=
  IntV.I64.toInt_ofNat h

theorem toInt_1024 : (IntV.ofNat 1024 : I32).toInt = 1024 := IntV.I32.toInt_ofNat (by norm_num)

theorem toInt_PAD : PAD.toInt = -1 := IntV.I64.toInt_ofInt (by norm_num)

/-- `int32` values widened to `int64` keep their value. -/
theorem toInt_cast64 (a : I32) : (IntV.cast a : I64).toInt = a.toInt := IntV.toInt_cast a (by norm_num)

/-! ## The token ranges -/

/-- `query_start_loc` is nondecreasing over `[0, R]`. -/
theorem qsl_mono {d : Inst} {init : Mem} (hpre : Pre d init) {a c : Nat} (hac : a ≤ c)
    (hc : c ≤ d.R) : qsl d init a ≤ qsl d init c := by
  induction c, hac using Nat.le_induction with
  | base => exact le_refl _
  | succ c hac ih =>
    exact le_trans (ih (by omega)) (hpre.2.2.2.2.2.2.2.2.2.2.2.2.1 c (by omega))

/-- Every `query_start_loc` entry of the batch is in `[0, N]`. -/
theorem qsl_range {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b ≤ d.R) :
    0 ≤ qsl d init b ∧ qsl d init b ≤ d.N :=
  ⟨le_trans hpre.2.2.2.2.2.2.2.2.2.2.2.1 (qsl_mono hpre (Nat.zero_le b) hb),
    le_trans (qsl_mono hpre hb (le_refl _)) hpre.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩

/-- Program `b`'s range lies in `[0, N]`. -/
theorem range_ok {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b ≤ d.R) :
    0 ≤ qsl d init b ∧ hiB d init b ≤ d.N := by
  refine ⟨(qsl_range hpre hb).1, ?_⟩
  unfold hiB
  split_ifs with h
  · exact (qsl_range hpre (by omega)).2
  · exact le_refl _

/-- An output element of group `g < G` and token `t < N` lies in the output. -/
theorem out_lt {d : Inst} (hd : d.ok) {g t : Nat} (hg : g < d.G) (ht : t < d.N) :
    g * d.S + t < d.slot_mappings.len := by
  have h1 := hd.out_len
  have : g * d.S ≤ (d.G - 1) * d.S := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The elements of `tok d g L H`, `0 ≤ L`, `H ≤ N`, lie in the output. -/
theorem tok_lt {d : Inst} (hd : d.ok) {g : Nat} (hg : g < d.G) {L H : Int} (hH : H ≤ d.N) :
    ∀ j ∈ tok d g L H, j < d.slot_mappings.len := by
  rintro j ⟨t, -, ht, rfl⟩
  exact out_lt hd hg (by omega)

end Claims.SlotMappings
