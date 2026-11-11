import Proof.Row

/-!
# The proof of `softmax_f32`

Program `p` of the grid runs `tl.forRange p n_rows num_programs`: trip `t` softmaxes row
`p + t * num_programs` in a block of `BLOCK_SIZE` lanes (masked lanes load `-∞`).

1. `Proof.Lanes`, `Proof.Rows`: the lanes' addresses, masks and `int32` arithmetic, the rows of a
   program (`trips`, `rowIn`, `rowOut`), and that distinct rows have disjoint output elements;
2. `Proof.Numeric`: `row_bound`, the error analysis of one row from the device facts (`tl.exp`,
   `/`, the depth of `tl.sum`'s tree): one factor `sc` per row, within `sumTol` of `1`;
3. `safe_trip`: one trip, walked line by line with the rules of `TVA.Proof.Rules`, reads the
   input and writes only its row, keeping the invariant `Inv` (rows done hold their softmax);
4. `safe_program`: the row loop, by `safe_forRange` with the invariant `Inv`;
5. `main`: the programs' footprints are disjoint (`Wy_disjoint`, `Wy_disjoint_R`) and every row is
   some program's, so `correct_of_safe` gives `Post`.
-/
namespace Claims.Softmax

open TVA 


/-- One trip of the row loop: the softmax of one row of `BLOCK_SIZE` lanes. -/
def trip (d : Inst) (row_idx : I32) : Prog Unit := do
  let row_start_ptr : Ptr .f32 := d.x.ptr 0 + row_idx * (IntV.ofNat d.input_row_stride)
  let col_offsets : Tensor [d.BLOCK_SIZE - 0] I32 := tl.arange 0 d.BLOCK_SIZE
  let input_ptrs : Tensor [d.BLOCK_SIZE - 0] (Ptr .f32) := row_start_ptr + col_offsets
  let mask : Tensor [d.BLOCK_SIZE - 0] Bool := col_offsets.ltS (IntV.ofNat d.n_cols)
  let row : Tensor [d.BLOCK_SIZE - 0] F32 ←
    tl.load input_ptrs (some mask) (some (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))))
  let row_minus_max : Tensor [d.BLOCK_SIZE - 0] F32 ← tl.sub row (bc (← tl.maxAll true row))
  let numerator : Tensor [d.BLOCK_SIZE - 0] F32 ← tl.exp row_minus_max
  let denominator : F32 ← tl.sum numerator
  let softmax_output : Tensor [d.BLOCK_SIZE - 0] F32 ← tl.div numerator (bc denominator)
  let output_row_start_ptr : Ptr .f32 := d.y.ptr 0 + row_idx * (IntV.ofNat d.output_row_stride)
  let output_ptrs : Tensor [d.BLOCK_SIZE - 0] (Ptr .f32) := output_row_start_ptr + col_offsets
  tl.store output_ptrs softmax_output (some mask)

/-- The model's body is the row loop over `trip`. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      tl.forRange (tl.program_id c 0) (IntV.ofNat d.n_rows) (tl.num_programs c 0) ()
        (fun row_idx _ => trip d row_idx) := rfl

/-- The kernel's `other=-float('inf')` literal is the `float32` `-∞`. -/
theorem neginf_lit :
    (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000) : F32) = FloatV.negInf := by decide

/-- Every lane of a broadcast is the broadcast value. -/
theorem bc_get {s : Shape} {α : Type} (a : α) (i : Idx s) :
    ((bc a : Tensor s α).get i) = a := rfl

/-- Every lane of the broadcast `-inf` literal is `-∞`. -/
theorem bc_lit_neg {s : Shape} (i : Idx s) :
    ((bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)) : Tensor s F32).get i) =
      FloatV.negInf :=
  (bc_get _ i).trans neginf_lit

/-- The grid has `num_programs` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.num_programs := Launch.size_grid1 _ _ _

/-- Program `i` has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- Every program sees `num_programs(0) = num_programs`. -/
theorem launch_grid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).grid 0 = d.num_programs := by
  simp [Launch.ctx, launch, Launch.grid1]

/-- The launch is valid: the grid is below `2^31` and the block shape is a power of two. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt fun s hs => ?_
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk
    exact hd.block_pow2
  · have h : d.BLOCK_SIZE ≤ 2 ^ 20 := by
      have := hd.block_le
      omega
    simpa using h

/-- The bytes of `x` the launch reads, unchanged from `init`. -/
def xAgree (d : Inst) (init m : Mem) : Prop := ∀ a ∈ d.x.elems (inSet d), m a = init a

/-- Every byte present at the start is still present. -/
def Presence (init m : Mem) : Prop := ∀ x, init x ≠ none → m x ≠ none

/-- Row `r` of the output holds a softmax of row `r` of the input (as read in `init`). -/
def RowQ (d : Inst) (init : Mem) (r : Nat) (m : Mem) : Prop :=
  ∃ xs : Nat → F32,
    (∀ k, k < d.n_cols → init.read .f32 (d.x.ptr (inIdx d r k)).addr = some (xs k)) ∧
    (∀ j, j < d.n_cols → ∃ c : F32, m.read .f32 (d.y.ptr (outIdx d r j)).addr = some c) ∧
    (RowOk xs d.n_cols → ∃ sc : ℝ, |sc - 1| ≤ (sumTol : ℝ) ∧
      ∀ j, j < d.n_cols → ∃ c : F32, m.read .f32 (d.y.ptr (outIdx d r j)).addr = some c ∧
        c.isFinite = true ∧
        |c.toReal - sc * softmaxRef xs d.n_cols j| ≤
          (elemTol : ℝ) * (softmaxGap xs d.n_cols j + 4) * softmaxRef xs d.n_cols j +
            (floorTol : ℝ))

/-- The invariant of the row loop: the rows already handled hold their softmax. -/
def Inv (d : Inst) (init : Mem) (p t : Nat) (m : Mem) : Prop :=
  (∀ t' < t, RowQ d init (p + t' * d.num_programs) m) ∧ xAgree d init m ∧ Presence init m

/-- Program `p`'s rows. -/
def Qp (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ t, t < trips d p → RowQ d init (p + t * d.num_programs) m

/-- The read footprint of a program: the input elements the launch reads. -/
def R (d : Inst) (_init : Mem) (_i : Fin (launch d).size) : Set Int := d.x.elems (inSet d)

/-- The write footprint of program `p`: the output elements of its rows. -/
def Wy (d : Inst) (p : Nat) : Set Int :=
  iterW (fun t => d.y.elems (rowOut d (p + t * d.num_programs))) (trips d p)

/-- The write footprint of program `i`: the output elements of its rows. -/
def W (d : Inst) (_init : Mem) (i : Fin (launch d).size) : Set Int := Wy d (i : Nat)

/-- The post-condition of program `i`: its rows hold their softmax. -/
def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop := Qp d init (i : Nat) m

/-- Row `p + t * num_programs` belongs to program `p`. -/
theorem row_mod {d : Inst} {p t : Nat} (hp : p < d.num_programs) :
    (p + t * d.num_programs) % d.num_programs = p := by
  rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp]

/-- Equal rows of programs `p` and `q` force `p = q`. -/
theorem rows_ne {d : Inst} {p q t u : Nat} (hp : p < d.num_programs) (hq : q < d.num_programs)
    (h : p + t * d.num_programs = q + u * d.num_programs) : p = q := by
  have h1 := row_mod (d := d) (p := p) (t := t) hp
  have h2 := row_mod (d := d) (p := q) (t := u) hq
  rw [h] at h1
  exact h1.symm.trans h2

/-- Distinct trips of one program handle distinct rows. -/
theorem add_mul_ne {d : Inst} {p t u : Nat} (hP : 0 < d.num_programs) (h : t ≠ u) :
    p + t * d.num_programs ≠ p + u * d.num_programs := by
  intro hc
  exact h (Nat.mul_right_cancel hP (Nat.add_left_cancel hc))

/-- Reading row `r`'s input does not depend on what happens to the output. -/
theorem read_x_eq {d : Inst} {init m : Mem} (hx : xAgree d init m) {r k : Nat}
    (hr : r < d.n_rows) (hk : k < d.n_cols) :
    init.read .f32 (d.x.ptr (inIdx d r k)).addr = m.read .f32 (d.x.ptr (inIdx d r k)).addr := by
  refine (Mem.read_congr (m₁ := m) (m₂ := init) ?_).symm
  intro a ha
  have hspan : a ∈ d.x.elems (inSet d) :=
    Buf.elems_mono d.x (rowIn_subset hr)
      (Buf.span_sub_elems d.x (S := rowIn d r) (j := inIdx d r k) ⟨k, hk, rfl⟩ ha)
  exact hx a hspan

/-- Reading output row `r'` does not see a write confined to the disjoint row `r`. -/
theorem read_row {d : Inst} {m m' : Mem} {r r' j : Nat}
    (hdisj : Disjoint (d.y.elems (rowOut d r')) (d.y.elems (rowOut d r)))
    (hframe : ∀ x, x ∉ d.y.elems (rowOut d r) → m' x = m x) (hj : j < d.n_cols) :
    m'.read .f32 (d.y.ptr (outIdx d r' j)).addr = m.read .f32 (d.y.ptr (outIdx d r' j)).addr := by
  refine Mem.read_congr fun x hx => hframe x ?_
  exact fun hc =>
    Set.disjoint_left.mp hdisj
      (Buf.span_sub_elems d.y (S := rowOut d r') (j := outIdx d r' j) ⟨j, hj, rfl⟩ hx) hc

/-- `RowQ` of a row survives any change confined to a disjoint output row. -/
theorem RowQ.frame {d : Inst} {init : Mem} {r r' : Nat} {m m' : Mem}
    (h : RowQ d init r' m)
    (hdisj : Disjoint (d.y.elems (rowOut d r')) (d.y.elems (rowOut d r)))
    (hframe : ∀ x, x ∉ d.y.elems (rowOut d r) → m' x = m x) :
    RowQ d init r' m' := by
  obtain ⟨xs, hx, hy, hsc⟩ := h
  refine ⟨xs, hx, ?_, ?_⟩
  · intro j hj
    obtain ⟨c, hc⟩ := hy j hj
    exact ⟨c, by rw [read_row hdisj hframe hj]; exact hc⟩
  · intro hok
    obtain ⟨sc, hsc1, hsc2⟩ := hsc hok
    refine ⟨sc, hsc1, fun j hj => ?_⟩
    obtain ⟨c, hc, hfin, herr⟩ := hsc2 j hj
    exact ⟨c, by rw [read_row hdisj hframe hj]; exact hc, hfin, herr⟩

/-- The output elements of row `r` lie in the write footprint of the program that owns it. -/
theorem row_mem_Wy {d : Inst} {r t p' : Nat} (ht : t < trips d p')
    (hpr : p' + t * d.num_programs = r) :
    d.y.elems (rowOut d r) ⊆ Wy d p' := by
  intro x hx
  rw [← hpr] at hx
  exact ⟨t, ht, hx⟩

/-- A program writes only output elements of the launch. -/
theorem Wy_subset {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    Wy d p ⊆ d.y.elems (outSet d) := by
  rintro x ⟨t, ht, hx⟩
  exact Buf.elems_mono d.y (rowOut_subset (row_lt hd hp ht)) hx

/-- Distinct programs write disjoint output elements. -/
theorem Wy_disjoint {d : Inst} (hd : d.ok) {p q : Nat} (hp : p < d.num_programs)
    (hq : q < d.num_programs) (hpq : p ≠ q) : Disjoint (Wy d p) (Wy d q) := by
  rw [Set.disjoint_left]
  rintro x ⟨t, ht, hx⟩ ⟨u, hu, hxu⟩
  have hrp := row_lt hd hp ht
  have hrq := row_lt hd hq hu
  have hrows : p + t * d.num_programs = q + u * d.num_programs := by
    by_contra hne
    exact Set.disjoint_left.mp (rowOut_disjoint hd hrp hrq hne) hx hxu
  exact hpq (rows_ne hp hq hrows)

/-- No program writes an input element the launch reads. -/
theorem Wy_disjoint_R {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_programs) :
    Disjoint (Wy d p) (d.x.elems (inSet d)) := by
  rw [Set.disjoint_left]
  intro x hx hx'
  exact Set.disjoint_left.mp hd.y_x (Wy_subset hd hp hx) hx'

/-- **One trip** of the row loop is safe: it reads the input, writes only its row, and the invariant `Inv` holds after it (`row_bound` gives the row's tolerance). -/
theorem safe_trip {d : Inst} (hd : d.ok) {P : Platform}
    (htot : Device.MathTotal P .exp .binary32) (hexp : Device.ExpF32 P)
    (hninf : Device.ExpF32NegInf P) (hdivt : Device.DivTotal P .binary32)
    (hdiv : Device.DivFull P) (hsum : Device.SumDepth P)
    {init m₁ : Mem} {p t : Nat} {row_idx : I32} (hpre : Pre d init)
    (hp : p < d.num_programs) (ht : t < trips d p)
    (hrow : row_idx.toInt = ((p + t * d.num_programs : Nat) : Int))
    (hx : xAgree d init m₁) (hpres : Presence init m₁)
    (hprev : ∀ t' < t, RowQ d init (p + t' * d.num_programs) m₁) :
    Safe ((launch d).env P) (d.x.elems (inSet d)) (d.y.elems (rowOut d (p + t * d.num_programs)))
      (fun _ m' => Inv d init p (t + 1) m') m₁ (trip d row_idx) := by
  have hPn : 0 < d.num_programs := Nat.lt_of_le_of_lt (Nat.zero_le p) hp
  have hrs : p + t * d.num_programs < d.n_rows := row_lt hd hp ht
  have hB31 : d.BLOCK_SIZE ≤ 2 ^ 31 := by
    have := hd.block_le
    omega
  have hn31 : d.n_cols < 2 ^ 31 := n_cols_lt_two_pow hd
  have hxValid : d.x.Valid m₁ :=
    Buf.valid_of_agree hpre.1 fun x hx' => hpres x (hpre.1.2.2.2 x hx')
  have hyValid : d.y.Valid m₁ :=
    Buf.valid_of_agree hpre.2 fun x hx' => hpres x (hpre.2.2.2.2 x hx')
  have hprodIn : (row_idx * IntV.ofNat d.input_row_stride : I32).toInt =
      ((p + t * d.num_programs : Nat) : Int) * d.input_row_stride :=
    stride_mul hd.in_stride_lt hrow (stride_mul_bound hd hrs)
  have hprodOut : (row_idx * IntV.ofNat d.output_row_stride : I32).toInt =
      ((p + t * d.num_programs : Nat) : Int) * d.output_row_stride :=
    stride_mul hd.out_stride_lt hrow (stride_mul_bound_out hd hrs)
  have himgin : Buf.img (s := [d.BLOCK_SIZE - 0])
      (fun i : Idx [d.BLOCK_SIZE - 0] =>
        (p + t * d.num_programs) * d.input_row_stride + (i.1 : Nat))
      (fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols)) =
      rowIn d (p + t * d.num_programs) := by
    rw [img_row hd.cols_le]
    rfl
  have himgout : Buf.img (s := [d.BLOCK_SIZE - 0])
      (fun i : Idx [d.BLOCK_SIZE - 0] =>
        (p + t * d.num_programs) * d.output_row_stride + (i.1 : Nat))
      (fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols)) =
      rowOut d (p + t * d.num_programs) := by
    rw [img_row hd.cols_le]
    rfl
  rw [trip]
  refine safe_load_buf (b := d.x)
    (g := fun i : Idx [d.BLOCK_SIZE - 0] =>
      (p + t * d.num_programs) * d.input_row_stride + (i.1 : Nat))
    (en := fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols))
    ?_ ?_ hxValid ?_ ?_
  · exact addrs_row' (b := d.x) (row := row_idx) (stride := d.input_row_stride)
      (B := d.BLOCK_SIZE) (n := d.n_cols) (r := p + t * d.num_programs) hprodIn hB31 hn31
  · intro i hi
    exact in_lane_lt (d := d) (r := p + t * d.num_programs) (j := (i.1 : Nat)) hd hrs
      (of_decide_eq_true hi)
  · rw [himgin]
    exact Buf.elems_mono d.x (rowIn_subset hrs)
  · intro v hlane hother
    refine safe_maxAll (N := d.BLOCK_SIZE - 0) ?_ true ?_
    · simpa using block_pos hd
    · intro mv hmv
      refine safe_sub (x := v) (y := bc mv) ?_
      intro rsm hrsm
      refine safe_math (fn := .exp) ?_ ?_
      · intro i
        exact htot (rsm.get i)
      · intro num hnum
        refine safe_sum (n := d.BLOCK_SIZE - 0) ?_ ?_
        · simpa using block_pos hd
        · intro den hden
          refine safe_div (x := num) (y := bc den) ?_ ?_
          · intro i
            exact hdivt (num.get i) den
          · intro q hq
            refine safe_last ?_
            refine safe_store_buf (b := d.y)
              (g := fun i : Idx [d.BLOCK_SIZE - 0] =>
                (p + t * d.num_programs) * d.output_row_stride + (i.1 : Nat))
              (en := fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols))
              ?_ ?_ ?_ hyValid ?_ ?_
            · exact addrs_row' (b := d.y) (row := row_idx) (stride := d.output_row_stride)
                (B := d.BLOCK_SIZE) (n := d.n_cols) (r := p + t * d.num_programs)
                hprodOut hB31 hn31
            · intro i hi
              exact out_lane_lt (d := d) (r := p + t * d.num_programs) (j := (i.1 : Nat)) hd hrs
                (of_decide_eq_true hi)
            · intro i j _ _ heq
              exact Prod.ext (Fin.ext (Nat.add_left_cancel heq)) (by cases i.2; cases j.2; rfl)
            · rw [himgout]
            · intro m' hstore hframe hpresent
              refine safe_ret ?_
              have hrows : ∀ t' < t + 1, RowQ d init (p + t' * d.num_programs) m' := by
                intro t' ht'
                rcases Nat.lt_or_ge t' t with hlt | hge
                · have hne : t' ≠ t := Nat.ne_of_lt hlt
                  refine RowQ.frame (hprev t' hlt)
                    (rowOut_disjoint hd (row_lt hd hp (Nat.lt_trans hlt ht)) hrs
                      (add_mul_ne hPn hne)) ?_
                  intro x hxx
                  exact hframe x (by rw [himgout]; exact hxx)
                · have hte : t' = t := by omega
                  rw [hte]
                  let xs : Nat → F32 :=
                    fun k => if h : k < d.BLOCK_SIZE then v.get1 ⟨k, h⟩ else default
                  have hvrow : ∀ (i : Nat) (hi : i < d.BLOCK_SIZE),
                      v.get1 ⟨i, hi⟩ = if h : i < d.n_cols then xs i else FloatV.negInf := by
                    intro i hi
                    by_cases hin : i < d.n_cols
                    · have hxs : xs i = v.get1 ⟨i, hi⟩ := by
                        change (if h : i < d.BLOCK_SIZE then v.get1 ⟨i, h⟩ else default) =
                          v.get1 ⟨i, hi⟩
                        rw [dite_eq_left hi]
                      rw [dite_eq_left hin, hxs]
                    · rw [dite_eq_right hin]
                      have hf : (fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols))
                          (⟨⟨i, hi⟩, ()⟩ : Idx [d.BLOCK_SIZE - 0]) = false := by
                        show decide (i < d.n_cols) = false
                        exact decide_eq_false hin
                      have hval := hother (⟨⟨i, hi⟩, ()⟩ : Idx [d.BLOCK_SIZE - 0]) hf
                        (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))) rfl
                      exact hval.trans
                        (bc_lit_neg (s := [d.BLOCK_SIZE - 0]) (⟨⟨i, hi⟩, ()⟩))
                  have hrsm' : ∀ (i : Nat) (hi : i < d.BLOCK_SIZE),
                      FloatV.NaNEq (FloatV.sub (v.get1 ⟨i, hi⟩) mv) (rsm.get1 ⟨i, hi⟩) :=
                    fun i hi => hrsm (⟨i, hi⟩, ())
                  have hnum' : ∀ (i : Nat) (hi : i < d.BLOCK_SIZE),
                      P.math .exp .binary32 (rsm.get1 ⟨i, hi⟩) (num.get1 ⟨i, hi⟩) :=
                    fun i hi => hnum (⟨i, hi⟩, ())
                  have hq' : ∀ (i : Nat) (hi : i < d.BLOCK_SIZE),
                      P.div .binary32 (num.get1 ⟨i, hi⟩) den (q.get1 ⟨i, hi⟩) :=
                    fun i hi => hq (⟨i, hi⟩, ())
                  have hreadQ : ∀ (j : Nat) (hj : j < d.n_cols) (hjB : j < d.BLOCK_SIZE),
                      m'.read .f32 (d.y.ptr (outIdx d (p + t * d.num_programs) j)).addr
                        = some (q.get1 ⟨j, hjB⟩) := by
                    intro j hj hjB
                    have hen : (fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols))
                        (⟨⟨j, hjB⟩, ()⟩ : Idx [d.BLOCK_SIZE - 0]) = true := by
                      show decide (j < d.n_cols) = true
                      exact decide_eq_true hj
                    have hread := hstore (⟨⟨j, hjB⟩, ()⟩ : Idx [d.BLOCK_SIZE - 0]) hen
                    simpa only [outIdx, Tensor.get1] using hread
                  refine ⟨xs, ?_, ?_, ?_⟩
                  · intro k hk
                    have hkB : k < d.BLOCK_SIZE := lt_of_lt_of_le hk hd.cols_le
                    have hen : (fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols))
                        (⟨⟨k, hkB⟩, ()⟩ : Idx [d.BLOCK_SIZE - 0]) = true := by
                      show decide (k < d.n_cols) = true
                      exact decide_eq_true hk
                    have hread := hlane (⟨⟨k, hkB⟩, ()⟩ : Idx [d.BLOCK_SIZE - 0]) hen
                    rw [read_x_eq hx hrs hk]
                    rw [show m₁.read .f32
                          (d.x.ptr (inIdx d (p + t * d.num_programs) k)).addr =
                          some (v.get1 ⟨k, hkB⟩) from by
                      simpa only [inIdx, Tensor.get1] using hread]
                    have hxs : xs k = v.get1 ⟨k, hkB⟩ := by
                      change (if h : k < d.BLOCK_SIZE then v.get1 ⟨k, h⟩ else default) =
                        v.get1 ⟨k, hkB⟩
                      rw [dite_eq_left hkB]
                    rw [hxs]
                  · intro j hj
                    have hjB : j < d.BLOCK_SIZE := lt_of_lt_of_le hj hd.cols_le
                    exact ⟨q.get1 ⟨j, hjB⟩, hreadQ j hj hjB⟩
                  · intro hok
                    obtain ⟨sc, hsc1, hsc2⟩ := row_bound (n := d.n_cols) (B := d.BLOCK_SIZE)
                      hd.cols_dim.1 hd.cols_le hd.block_le P hexp hninf hdiv hsum
                      hvrow hmv hrsm' hnum' hden hq' hok
                    refine ⟨sc, hsc1, fun j hj => ?_⟩
                    have hjB : j < d.BLOCK_SIZE := lt_of_lt_of_le hj hd.cols_le
                    obtain ⟨hfin, herr⟩ := hsc2 j hj hjB
                    exact ⟨q.get1 ⟨j, hjB⟩, hreadQ j hj hjB, hfin, herr⟩
              refine ⟨hrows, ?_, fun x h0 => hpresent x (hpres x h0)⟩
              intro a ha
              have hnotin : a ∉ d.y.elems (Buf.img (s := [d.BLOCK_SIZE - 0])
                  (fun i : Idx [d.BLOCK_SIZE - 0] =>
                    (p + t * d.num_programs) * d.output_row_stride + (i.1 : Nat))
                  (fun i : Idx [d.BLOCK_SIZE - 0] => decide ((i.1 : Nat) < d.n_cols))) := by
                rw [himgout]
                intro hc
                exact Set.disjoint_left.mp hd.y_x (Buf.elems_mono d.y (rowOut_subset hrs) hc) ha
              rw [hframe a hnotin]
              exact hx a ha

/-- **One program** is safe: the row loop keeps `Inv` over all its trips and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {P : Platform}
    (htot : Device.MathTotal P .exp .binary32) (hexp : Device.ExpF32 P)
    (hninf : Device.ExpF32NegInf P) (hdivt : Device.DivTotal P .binary32)
    (hdiv : Device.DivFull P) (hsum : Device.SumDepth P)
    {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hp : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hN : (IntV.ofNat d.n_rows : I32).toInt = (d.n_rows : Int) := by
    have hrg := hd.rows_grid
    have hP0 : 0 < d.num_programs := Nat.lt_of_le_of_lt (Nat.zero_le (i : Nat)) hp
    exact IntV.I32.toInt_ofNat (by omega)
  have hP : (IntV.ofNat d.num_programs : I32).toInt = (d.num_programs : Int) :=
    IntV.I32.toInt_ofNat hd.grid_lt
  have hi0 : (IntV.ofNat (i : Nat) : I32).toInt = ((i : Nat) : Int) := by
    have hgl := hd.grid_lt
    exact IntV.I32.toInt_ofNat (by omega)
  rw [body_eq, tl.program_id, tl.num_programs, launch_pid, launch_grid]
  refine safe_last ?_
  refine (safe_forRange (range_ok hd hp) (fun t _ m => Inv d init (i : Nat) t m)
    (fun t => d.y.elems (rowOut d ((i : Nat) + t * d.num_programs))) (Wk := ∅) ?_ ?_ ?_ ?_ ?_).mono
    (fun _ _ h => h) ?_
  · -- different trips write different rows
    intro t u htu hu
    simp only [hN, hP, hi0] at hu
    have ht' : t < trips d (i : Nat) := Nat.lt_trans htu hu
    exact rowOut_disjoint hd (row_lt hd hp ht') (row_lt hd hp hu)
      (add_mul_ne (Nat.lt_of_le_of_lt (Nat.zero_le (i : Nat)) hp) (Nat.ne_of_lt htu))
  · exact Set.disjoint_empty _
  · -- the invariant at the start
    exact ⟨fun t' ht' => absurd ht' (Nat.not_lt_zero t'), fun a _ => rfl, fun x hx => hx⟩
  · -- one trip
    intro t ht st m₁ hI
    simp only [hN, hP, hi0] at ht ⊢
    exact safe_trip hd htot hexp hninf hdivt hdiv hsum hpre hp ht
      (row_toInt hd hp ht) hI.2.1 hI.2.2 hI.1
  · -- the loop is done
    intro st m₁ hI
    simp only [hN, hP, hi0] at hI
    exact safe_ret hI.1
  · -- the loop's rows are the program's rows
    intro x hx
    have hx' : x ∈ iterW (fun t => d.y.elems (rowOut d ((i : Nat) + t * d.num_programs)))
        (trips d (i : Nat)) := by simpa [hN, hP, hi0] using hx
    simpa only [W, Wy] using hx'

/-- **The claim.** Each program is safe on its own rows (`safe_program`); the programs' output rows
are disjoint from each other and from the input, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d init i) (fun i => W d init i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · exact safe_program hd (hP.mathTotal _ _) hP.expF32 hP.expF32NegInf (hP.divTotal _) hP.divFull hP.sumDepth hpre i
  · intro i j hij
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    have hip : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
    have hjp : (j : Nat) < d.num_programs := lt_of_lt_of_eq j.isLt (launch_size d)
    exact (Wy_disjoint_R hd hip).union_right (Wy_disjoint hd hip hjp hij')
  · intro i
    have hip : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
    exact (Wy_disjoint_R hd hip).symm
  · intro fin hloc hfr
    refine ⟨fun r hr => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · obtain ⟨hp, ht, hpr⟩ := trip_cover hd hr
      have hsize : r % d.num_programs < (launch d).size := by
        rw [launch_size d]; exact hp
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨r % d.num_programs, hsize⟩
      have hrow : RowQ d init r m := by
        have h := hQ (r / d.num_programs) ht
        rwa [hpr] at h
      obtain ⟨xs, hxread, hyread, hsc⟩ := hrow
      refine ⟨xs, hxread, ?_, ?_⟩
      · intro j hj
        obtain ⟨c, hc⟩ := hyread j hj
        have hspan : Mem.span .f32 (d.y.ptr (outIdx d r j)).addr ⊆ d.y.elems (rowOut d r) :=
          Buf.span_sub_elems d.y (S := rowOut d r) (j := outIdx d r j) ⟨j, hj, rfl⟩
        have hfin : fin.read .f32 (d.y.ptr (outIdx d r j)).addr =
            m.read .f32 (d.y.ptr (outIdx d r j)).addr :=
          Mem.read_congr fun x hxx => hagree x (Or.inr (row_mem_Wy ht hpr (hspan hxx)))
        exact ⟨c, by rw [hfin]; exact hc⟩
      · intro hok
        obtain ⟨sc, hsc1, hsc2⟩ := hsc hok
        refine ⟨sc, hsc1, fun j hj => ?_⟩
        obtain ⟨c, hc, hfin', herr⟩ := hsc2 j hj
        have hspan : Mem.span .f32 (d.y.ptr (outIdx d r j)).addr ⊆ d.y.elems (rowOut d r) :=
          Buf.span_sub_elems d.y (S := rowOut d r) (j := outIdx d r j) ⟨j, hj, rfl⟩
        have hfin : fin.read .f32 (d.y.ptr (outIdx d r j)).addr =
            m.read .f32 (d.y.ptr (outIdx d r j)).addr :=
          Mem.read_congr fun x hxx => hagree x (Or.inr (row_mem_Wy ht hpr (hspan hxx)))
        exact ⟨c, by rw [hfin]; exact hc, hfin', herr⟩
    · exact Wy_subset hd (lt_of_lt_of_eq i.isLt (launch_size d)) hzi

end Claims.Softmax
