import Claim
import TVA.Proof

/-!
# `moe_router_bf16`: the shared setup

The launch's validity, the programs' read and write footprints (`R`, `W`), what each program
leaves behind (`Q`, the per-token part of `Post`), and how the footprints give `Post`.
-/

namespace Claims.MoeRouter

open TVA

open Block

/-- With the device fact `SumDepth`, a reduction's tree (depth at most `tl.sumBound P n`) has depth
at most `⌈log₂ n⌉`. -/
theorem depth_le_clog {P : Platform} (h : Device.SumDepth P) {n : Nat} {t : Tree}
    (ht : t.depth ≤ tl.sumBound P n) : t.depth ≤ Nat.clog 2 n :=
  ht.trans (max_le (h n) le_rfl)

/-- A store to a scalar pointer is a store of the scalar tensors. -/
theorem store1_eq {d : DType} (p : Ptr d) (v : d.Val) :
    tl.store1 p v = tl.store (Tensor.scalar p) (Tensor.scalar v) none := rfl

/-- The value of a scalar (rank-0) tensor. -/
theorem get_scalar_eq {α : Type} (a : α) (i : Idx []) : (Tensor.scalar a).get i = a := rfl

/-- The elements of a lane map `2 p` are below any `n` above `2 p + 1`. -/
theorem img0_lt {p n : Nat} (hn : 2 * p + 1 < n) :
    ∀ k ∈ Buf.img (fun (_ : Idx []) => 2 * p) (fun _ => true), k < n := by
  rintro k ⟨u, -, rfl⟩
  show 2 * p < n
  omega

/-- The elements of a lane map `2 p + 1` are below any `n` above `2 p + 1`. -/
theorem img1_lt {p n : Nat} (hn : 2 * p + 1 < n) :
    ∀ k ∈ Buf.img (fun (_ : Idx []) => 2 * p + 1) (fun _ => true), k < n := by
  rintro k ⟨u, -, rfl⟩
  show 2 * p + 1 < n
  omega

/-- The grid has one program per token. -/
theorem launch_size (d : Inst) : (launch d).size = d.bs := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The number of tokens fits the grid limit. -/
theorem grid_lt {d : Inst} (hd : d.ok) : d.bs < 2 ^ 31 := by
  have := hd.out_off; omega

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (grid_lt hd) fun s hs => ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  obtain ⟨e, he⟩ := hd.block_pow2
  have hble := hd.block_le
  have he13 : e ≤ 13 := by
    rw [he] at hble
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp hble
  rcases hs with rfl | rfl | rfl
  · refine ⟨?_, ?_⟩
    · intro k hk
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
      rcases hk with rfl | rfl
      · exact ⟨3, by norm_num⟩
      · exact ⟨e, he⟩
    · show 8 * (d.BLOCK * 1) ≤ 2 ^ 20
      rw [he]
      have h16 : 8 * 2 ^ e ≤ 2 ^ 20 := by
        have he' : (2:Nat) ^ e ≤ 2 ^ 13 := Nat.pow_le_pow_right (by norm_num) he13
        calc 8 * 2 ^ e = 2 ^ 3 * 2 ^ e := rfl
          _ = 2 ^ (3 + e) := (Nat.pow_add _ _ _).symm
          _ ≤ 2 ^ 16 := Nat.pow_le_pow_right (by norm_num) (by omega)
          _ ≤ 2 ^ 20 := Nat.pow_le_pow_right (by norm_num) (by norm_num)
      omega
  · refine ⟨?_, ?_⟩
    · intro k hk
      simp only [List.mem_singleton] at hk
      subst hk; exact ⟨e, he⟩
    · show d.BLOCK * 1 ≤ 2 ^ 20
      rw [he]
      have : (2:Nat) ^ e ≤ 2 ^ 13 := Nat.pow_le_pow_right (by norm_num) he13
      omega
  · refine ⟨?_, ?_⟩
    · intro k hk
      simp only [List.mem_singleton] at hk
      subst hk; exact ⟨3, by norm_num⟩
    · show 8 * 1 ≤ 2 ^ 20
      norm_num

/-- Elements `2p` and `2p+1` of a buffer: the two an output element per program holds. -/
def two (p : Nat) : Set Nat := {j | j = 2 * p ∨ j = 2 * p + 1}

/-- The read footprint of program `p`: its row of `x` and all of `w`. -/
def R (d : Inst) (_init : Mem) (p : Fin (launch d).size) : Set Int :=
  d.x.elems {j | p * d.hidden ≤ j ∧ j < p * d.hidden + d.hidden} ∪ d.w.elems {j | j < 8 * d.hidden}

/-- The write footprint of program `p`: elements `2p` and `2p+1` of the two outputs. -/
def W (d : Inst) (_init : Mem) (p : Fin (launch d).size) : Set Int :=
  d.outw.elems (two (p : Nat)) ∪ d.outi.elems (two (p : Nat))

/-- What program `p` leaves behind: `TokenPost`. -/
def Q (d : Inst) (init : Mem) (p : Fin (launch d).size) (m : Mem) : Prop := TokenPost d init m p

/-- The pairs `{2p, 2p+1}` of distinct tokens are disjoint in a buffer. -/
theorem two_disjoint {dt : DType} (b : Buf dt) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (b.elems (two p)) (b.elems (two q)) := by
  refine b.elems_disjoint ?_
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [two, Set.mem_ofPred_eq] at hx hy
  rcases hx with rfl | rfl <;> rcases hy with h | h <;> omega

/-- Elements `2p` and `2p+1` are below `n` when `2p+1` is. -/
theorem two_lt {p n : Nat} (h : 2 * p + 1 < n) : ∀ k ∈ two p, k < n := by
  intro k hk
  simp only [two, Set.mem_ofPred_eq] at hk
  rcases hk with rfl | rfl <;> omega

/-- The pair `{2p, 2p+1}` lies inside the buffer. -/
theorem two_len {dt : DType} {b : Buf dt} {p n : Nat} (h : 2 * p + 1 < n) (hn : n ≤ b.len) :
    ∀ k ∈ two p, k < b.len :=
  fun k hk => lt_of_lt_of_le (two_lt h k hk) hn

/-- Token `p`'s row lies inside `x`. -/
theorem x_row_len {d : Inst} (hd : d.ok) (p : Fin (launch d).size) :
    ∀ j ∈ {j | p * d.hidden ≤ j ∧ j < p * d.hidden + d.hidden}, j < d.x.len := by
  intro j hj
  have hp : (p : Nat) < d.bs := lt_of_lt_of_eq p.isLt (launch_size d)
  have hm : ((p : Nat) + 1) * d.hidden ≤ d.bs * d.hidden := Nat.mul_le_mul_right _ hp
  have hx := hd.x_len
  have hh := hd.hidden_pos
  simp only [Set.mem_ofPred_eq] at hj
  have e : (p : Nat) * d.hidden + d.hidden = ((p : Nat) + 1) * d.hidden := by ring
  omega

/-- All of `w` lies inside its buffer. -/
theorem w_len_all {d : Inst} (hd : d.ok) : ∀ j ∈ {j | j < 8 * d.hidden}, j < d.w.len := by
  intro j hj
  simp only [Set.mem_ofPred_eq] at hj
  have := hd.w_len
  omega

/-- Programs write disjoint bytes: no program touches another's footprint. -/
theorem hdisj {d : Inst} (hd : d.ok) (init : Mem) (_hpre : Pre d init) (i j : Fin (launch d).size)
    (hij : i ≠ j) : Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hne : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hj : (j : Nat) < d.bs := lt_of_lt_of_eq j.isLt (launch_size d)
  have hl : 2 * (i : Nat) + 1 < 2 * d.bs := by omega
  simp only [W, R]
  refine Disjoint.union_left
    (Disjoint.union_right (Disjoint.union_right ?_ ?_) (Disjoint.union_right ?_ ?_))
    (Disjoint.union_right (Disjoint.union_right ?_ ?_) (Disjoint.union_right ?_ ?_))
  · exact Buf.elems_disjoint_of d.outw d.x hd.outw_x (two_len hl hd.outw_len) (x_row_len hd j)
  · exact Buf.elems_disjoint_of d.outw d.w hd.outw_w (two_len hl hd.outw_len) (w_len_all hd)
  · exact two_disjoint d.outw hne
  · exact Buf.elems_disjoint_of d.outw d.outi hd.outw_outi (two_len hl hd.outw_len)
      (two_len (by omega : 2 * (j:Nat) + 1 < 2 * d.bs) hd.outi_len)
  · exact Buf.elems_disjoint_of d.outi d.x hd.outi_x (two_len hl hd.outi_len) (x_row_len hd j)
  · exact Buf.elems_disjoint_of d.outi d.w hd.outi_w (two_len hl hd.outi_len) (w_len_all hd)
  · exact Buf.elems_disjoint_of d.outi d.outw hd.outw_outi.symm (two_len hl hd.outi_len)
      (two_len (by omega : 2 * (j:Nat) + 1 < 2 * d.bs) hd.outw_len)
  · exact two_disjoint d.outi hne

/-- No program reads what it writes. -/
theorem hRW {d : Inst} (hd : d.ok) (init : Mem) (_hpre : Pre d init) (i : Fin (launch d).size) :
    Disjoint (R d init i) (W d init i) := by
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hl : 2 * (i : Nat) + 1 < 2 * d.bs := by omega
  simp only [W, R]
  refine Disjoint.union_left (Disjoint.union_right ?_ ?_) (Disjoint.union_right ?_ ?_)
  · exact (Buf.elems_disjoint_of d.outw d.x hd.outw_x (two_len hl hd.outw_len) (x_row_len hd i)).symm
  · exact (Buf.elems_disjoint_of d.outi d.x hd.outi_x (two_len hl hd.outi_len) (x_row_len hd i)).symm
  · exact (Buf.elems_disjoint_of d.outw d.w hd.outw_w (two_len hl hd.outw_len) (w_len_all hd)).symm
  · exact (Buf.elems_disjoint_of d.outi d.w hd.outi_w (two_len hl hd.outi_len) (w_len_all hd)).symm

/-- A read of a span inside the writes is the same read before and after. -/
theorem read_frame {d : Inst} {init : Mem} {p : Fin (launch d).size} {fin m : Mem}
    (hag : ∀ x ∈ R d init p ∪ W d init p, fin x = m x)
    {dt : DType} {a : Int} (hspan : Mem.span dt a ⊆ R d init p ∪ W d init p) :
    fin.read dt a = m.read dt a :=
  Mem.read_congr fun x hx => hag x (hspan hx)

/-- `Post` follows from the programs' results. -/
theorem hpost {d : Inst} (_hd : d.ok) (init : Mem) (_hpre : Pre d init) (fin : Mem)
    (hloc : ∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ W d init i) → fin x = init x) : Post d init fin := by
  refine ⟨fun t ht => ?_, fun z hz => ?_⟩
  · obtain ⟨m, hQ, hag⟩ := hloc ⟨t, by rw [launch_size]; exact ht⟩
    obtain ⟨C, i0, i1, hcap, hi0, hmax0, hlt0, hi1, hne1, hmax1, hlt1, hr0, hr1, a0, a1,
      ha0, ha1, hf0, hf1, hb0, hb1⟩ := hQ
    have houti : ∀ k, k ∈ two t → Mem.span .i32 (d.outi.ptr (k : Int)).addr ⊆
        R d init ⟨t, by rw [launch_size]; exact ht⟩ ∪
          W d init ⟨t, by rw [launch_size]; exact ht⟩ := by
      intro k hk x hx
      exact Or.inr (by
        simp only [W]
        exact Or.inr (Buf.span_sub_elems d.outi hk hx))
    have houtw : ∀ k, k ∈ two t → Mem.span .f32 (d.outw.ptr (k : Int)).addr ⊆
        R d init ⟨t, by rw [launch_size]; exact ht⟩ ∪
          W d init ⟨t, by rw [launch_size]; exact ht⟩ := by
      intro k hk x hx
      exact Or.inr (by
        simp only [W]
        exact Or.inl (Buf.span_sub_elems d.outw hk hx))
    refine ⟨C, i0, i1, hcap, hi0, hmax0, hlt0, hi1, hne1, hmax1, hlt1, ?_, ?_, a0, a1, ?_, ?_,
      hf0, hf1, hb0, hb1⟩
    · exact (read_frame hag (houti (2 * t) (Or.inl rfl))).trans hr0
    · exact (read_frame hag (houti (2 * t + 1) (Or.inr rfl))).trans hr1
    · exact (read_frame hag (houtw (2 * t) (Or.inl rfl))).trans ha0
    · exact (read_frame hag (houtw (2 * t + 1) (Or.inr rfl))).trans ha1
  · refine hfr z fun i hzi => hz ?_
    have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
    have hl : 2 * (i : Nat) + 1 < 2 * d.bs := by omega
    simp only [W] at hzi
    rcases hzi with h | h
    · exact Or.inl (d.outw.elems_mono (two_lt hl) h)
    · exact Or.inr (d.outi.elems_mono (two_lt hl) h)

end Claims.MoeRouter
