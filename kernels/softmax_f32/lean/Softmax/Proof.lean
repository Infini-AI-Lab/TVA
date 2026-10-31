import Softmax.Err

/-!
# `softmax_f32`: the proof

* `safe_row`: one trip of the loop, row `r`. The masked load (`-∞` on the lanes `≥ n_cols`), the
  max, the subtraction, `exp`, the sum, the division and the masked store; the row is then done
  (`RowDone`): its outputs are within the tolerance of the exact softmax (`row_bound`, `Err.lean`).
* `safe_program`: program `p` runs the rows `p, p + G, …` below `n_rows` (`G = num_programs`), one
  trip each, through `safe_forRange`; trip `t` writes row `p + t·G`'s outputs only.
* `main`: `TVA.meets_independent`, with program `p` reading the input rows `≡ p (mod G)`
  (`progIn`) and writing those output rows (`progOut`); programs write disjoint rows, and no output
  element meets an input element (`Inst.ok.y_x`).
* `consistent`: every assumption holds of `Platform.sm90ref` with a total faithful behaviour.
* `satisfiable`: the source's test instance (`1823 × 781`), in a memory of zeros.
-/

namespace Kernels.Softmax

open TVA TVA.Math.Float

/-! ## Footprints and what a program leaves -/

/-- The input elements of row `r`. -/
def rowIn (d : Inst) (r : Nat) : Set Nat := {k | ∃ j, j < d.n_cols ∧ k = inIdx d r j}

/-- The output elements of row `r`. -/
def rowOut (d : Inst) (r : Nat) : Set Nat := {k | ∃ j, j < d.n_cols ∧ k = outIdx d r j}

/-- The input elements program `p` reads: the rows `r ≡ p (mod num_programs)`. -/
def progIn (d : Inst) (p : Nat) : Set Nat :=
  {k | ∃ r j, r < d.n_rows ∧ r % d.num_programs = p ∧ j < d.n_cols ∧ k = inIdx d r j}

/-- The output elements program `p` writes. -/
def progOut (d : Inst) (p : Nat) : Set Nat :=
  {k | ∃ r j, r < d.n_rows ∧ r % d.num_programs = p ∧ j < d.n_cols ∧ k = outIdx d r j}

/-- Row `r` is done in memory `m`: its outputs are, up to the row's common factor, the softmax of
its inputs in `init`, within the tolerance (`Post`'s clause for one row). -/
def RowDone (d : Inst) (init m : Mem) (r : Nat) : Prop :=
  ∃ xs : Nat → F32,
    (∀ k, k < d.n_cols → init.read .f32 (d.x.ptr (inIdx d r k)).addr = some (xs k)) ∧
    ∃ sc : ℝ, |sc - 1| ≤ (sumTol d.BLOCK_SIZE : ℝ) ∧
      ∀ j, j < d.n_cols → ∃ c : F32, m.read .f32 (d.y.ptr (outIdx d r j)).addr = some c ∧
        c.isFinite = true ∧
        |c.toReal - sc * softmaxRef xs d.n_cols j| ≤
          ((elemTol : ℝ) + (gapTol : ℝ) * softmaxGap xs d.n_cols j) * softmaxRef xs d.n_cols j +
            (floorTol : ℝ) * softmaxMax xs d.n_cols

/-- What program `p` leaves: each of its rows is done. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ r, r < d.n_rows → r % d.num_programs = p → RowDone d init m r

theorem rowOut_sub {d : Inst} {r p : Nat} (hr : r < d.n_rows) (hrp : r % d.num_programs = p) :
    rowOut d r ⊆ progOut d p := by
  rintro k ⟨j, hj, rfl⟩; exact ⟨r, j, hr, hrp, hj, rfl⟩

theorem rowIn_sub {d : Inst} {r p : Nat} (hr : r < d.n_rows) (hrp : r % d.num_programs = p) :
    rowIn d r ⊆ progIn d p := by
  rintro k ⟨j, hj, rfl⟩; exact ⟨r, j, hr, hrp, hj, rfl⟩

theorem progOut_sub (d : Inst) (p : Nat) : progOut d p ⊆ outSet d := by
  rintro k ⟨r, j, hr, -, hj, rfl⟩; exact ⟨r, j, hr, hj, rfl⟩

theorem progIn_sub (d : Inst) (p : Nat) : progIn d p ⊆ inSet d := by
  rintro k ⟨r, j, hr, -, hj, rfl⟩; exact ⟨r, j, hr, hj, rfl⟩

/-- Distinct output elements: one row, or rows at least `n_cols` apart. -/
theorem outIdx_inj {d : Inst} (hd : d.ok) {r r' j j' : Nat} (hr : r < d.n_rows) (hr' : r' < d.n_rows)
    (hj : j < d.n_cols) (hj' : j' < d.n_cols) (h : outIdx d r j = outIdx d r' j') : r = r' ∧ j = j' := by
  unfold outIdx at h
  rcases hd.out_rows with h1 | h1
  · have : r = 0 := by omega
    have : r' = 0 := by omega
    subst r; subst r'
    simp at h; exact ⟨rfl, h⟩
  · have key : r = r' := by
      by_contra hne
      rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
      · have : (r + 1) * d.output_row_stride ≤ r' * d.output_row_stride := Nat.mul_le_mul_right _ hlt
        nlinarith
      · have : (r' + 1) * d.output_row_stride ≤ r * d.output_row_stride := Nat.mul_le_mul_right _ hlt
        nlinarith
    subst key
    exact ⟨rfl, by omega⟩

theorem rowOut_disjoint {d : Inst} (hd : d.ok) {r r' : Nat} (hr : r < d.n_rows) (hr' : r' < d.n_rows)
    (hne : r ≠ r') : Disjoint (d.y.elems (rowOut d r)) (d.y.elems (rowOut d r')) := by
  refine d.y.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro k ⟨j, hj, rfl⟩ ⟨j', hj', h⟩
  exact hne (outIdx_inj hd hr hr' hj hj' h).1

theorem progOut_disjoint {d : Inst} (hd : d.ok) {p q : Nat} (hne : p ≠ q) :
    Disjoint (d.y.elems (progOut d p)) (d.y.elems (progOut d q)) := by
  refine d.y.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro k ⟨r, j, hr, hrp, hj, rfl⟩ ⟨r', j', hr', hrq, hj', h⟩
  obtain ⟨rfl, -⟩ := outIdx_inj hd hr hr' hj hj' h
  exact hne (hrp.symm.trans hrq)

/-- A done row stays done in a memory that agrees on its output elements. -/
theorem rowDone_of_agree {d : Inst} {init m m' : Mem} {r : Nat} (h : RowDone d init m r)
    (hag : ∀ z ∈ d.y.elems (rowOut d r), m' z = m z) : RowDone d init m' r := by
  obtain ⟨xs, hxs, sc, hsc, hout⟩ := h
  refine ⟨xs, hxs, sc, hsc, fun j hj => ?_⟩
  obtain ⟨c, hc, h1, h2⟩ := hout j hj
  refine ⟨c, ?_, h1, h2⟩
  rw [← hc]
  exact Mem.read_congr fun z hz => hag z ⟨outIdx d r j, ⟨j, hj, rfl⟩, hz⟩

/-! ## One trip of the loop -/

/-- The loop body for the row `row_idx`, the model's lines with the names replaced by their
values. -/
def rowProg (d : Inst) (row_idx : I32) : Prog Unit := do
  let row ← tl.load (d.x.ptr 0 + row_idx * IntV.ofNat d.input_row_stride + tl.arange 0 d.BLOCK_SIZE)
    (some ((tl.arange 0 d.BLOCK_SIZE).ltS (IntV.ofNat d.n_cols)))
    (some (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))))
  let row_minus_max ← tl.sub row (bc (← tl.maxAll true row))
  let numerator ← tl.exp row_minus_max
  let denominator ← tl.sum numerator
  let softmax_output ← tl.div numerator (bc denominator)
  tl.store (d.y.ptr 0 + row_idx * IntV.ofNat d.output_row_stride + tl.arange 0 d.BLOCK_SIZE)
    softmax_output (some ((tl.arange 0 d.BLOCK_SIZE).ltS (IntV.ofNat d.n_cols)))

theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c = tl.forRange (tl.program_id c 0) (IntV.ofNat d.n_rows) (tl.num_programs c 0) ()
      (fun row_idx _ => rowProg d row_idx) := rfl

theorem negInf_lit : FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000 : F64) = (FloatV.negInf : F32) := by
  decide

/-- Lane `i` of `tl.arange(0, B) < n`. -/
theorem mask_get {B n : Nat} (hB : B < 2 ^ 31) (hn : n < 2 ^ 31) (i : Idx [B - 0]) :
    ((tl.arange 0 B).ltS (IntV.ofNat n : I32)).get i = decide (i.1.1 < n) := by
  obtain ⟨l, ⟨⟩⟩ := i
  have hl := l.isLt
  simp only [Nat.sub_zero] at hl
  show IntV.lt (IntV.ofInt (((0 : Nat) : Int) + (l : Nat))) (IntV.ofNat n : I32) = _
  rw [IntV.lt_eq, IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega), IntV.I32.toInt_ofNat hn]
  simp

/-- Lane `i` of `b_ptr + row_idx * stride + tl.arange(0, B)`: element `r * stride + i`. -/
theorem ptr_get {B : Nat} (hB : B < 2 ^ 31) (b : Buf .f32) {row_idx : I32} {r s : Nat}
    (hrow : row_idx.toInt = r) (hs : s < 2 ^ 31) (hrs : r * s < 2 ^ 31) (i : Idx [B - 0]) :
    (b.ptr 0 + row_idx * (IntV.ofNat s : I32) + tl.arange 0 B).get i = b.ptr ((r * s + i.1.1 : Nat) : Int) := by
  obtain ⟨l, ⟨⟩⟩ := i
  have hl := l.isLt
  simp only [Nat.sub_zero] at hl
  have hmul : (row_idx * (IntV.ofNat s : I32)).toInt = ((r * s : Nat) : Int) :=
    IntV.I32.mul_nat hrow (IntV.I32.toInt_ofNat hs) hrs
  have hli : (IntV.ofInt (((0 : Nat) : Int) + (l : Nat)) : I32).toInt = l := by
    rw [IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega)]; push_cast; ring
  show ((b.ptr 0).offset (row_idx * (IntV.ofNat s : I32)).toInt).offset
      (IntV.ofInt (((0 : Nat) : Int) + (l : Nat)) : I32).toInt = _
  rw [hmul, hli, Buf.ptr_offset, Buf.ptr_offset]
  push_cast; ring_nf

/-- **One trip**: the row `r` (`row_idx = r`), from a memory that agrees with `init` on the row's
inputs. The program reads the row's inputs, writes the row's outputs (`rowOut`), and leaves the row
done, nothing else changed and every present byte present. -/
theorem safe_row {E : Env} (htot : Rules.Num.MathTotal E.P .exp .binary32) (hexp : Rules.Num.ExpF32 E.P)
    (hninf : Rules.Num.ExpF32NegInf E.P) (hdtot : Rules.Num.DivTotal E.P .binary32)
    (hdiv : Rules.Num.DivFull E.P)
    {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {Rd : Set Int} {r : Nat}
    (hr : r < d.n_rows) {row_idx : I32} (hrow : row_idx.toInt = r)
    (hR : d.x.elems (rowIn d r) ⊆ Rd) {m₁ : Mem} (hpres : ∀ z, init z ≠ none → m₁ z ≠ none)
    (hag : ∀ z ∈ d.x.elems (rowIn d r), m₁ z = init z) {Q' : Unit → Mem → Prop}
    (hk : ∀ m', RowDone d init m' r → (∀ z, z ∉ d.y.elems (rowOut d r) → m' z = m₁ z) →
      (∀ z, m₁ z ≠ none → m' z ≠ none) → Q' () m') :
    Safe E Rd (d.y.elems (rowOut d r)) Q' m₁ (rowProg d row_idx) := by
  obtain ⟨hvx, hvy, hrows⟩ := hpre
  obtain ⟨hxin, hxfin⟩ := hrows r hr
  have hBpos : 0 < d.BLOCK_SIZE := by obtain ⟨e, he⟩ := hd.block_pow2; rw [he]; positivity
  have hB31 : d.BLOCK_SIZE < 2 ^ 31 := by have := hd.block_le; omega
  have hn31 : d.n_cols < 2 ^ 31 := by have := hd.cols_le; have := hd.block_le; omega
  have hrle : r ≤ d.n_rows - 1 := by omega
  have hrs_in : r * d.input_row_stride < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ hrle) hd.in_off
  have hrs_out : r * d.output_row_stride < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ hrle) hd.out_off
  have hvx₁ : d.x.Valid m₁ := Buf.valid_of_agree hvx fun z hz => hpres z (hvx.2.2.2 z hz)
  have hvy₁ : d.y.Valid m₁ := Buf.valid_of_agree hvy fun z hz => hpres z (hvy.2.2.2 z hz)
  unfold rowProg
  -- line 99: the masked load of the row, `-∞` on the disabled lanes
  refine safe_load_buf d.x (g := fun (i : Idx [d.BLOCK_SIZE - 0]) => inIdx d r i.1.1)
    (en := fun (i : Idx [d.BLOCK_SIZE - 0]) => decide (i.1.1 < d.n_cols))
    (Buf.addrs_of_get d.x (fun i => mask_get hB31 hn31 i)
      (fun i _ => ptr_get hB31 d.x hrow hd.in_stride_lt hrs_in i))
    (fun i hi => hd.x_len r _ hr (of_decide_eq_true hi)) hvx₁ ?_ fun row hv1 hv2 => ?_
  · rintro z ⟨k, ⟨i, hi, rfl⟩, hz⟩
    exact hR ⟨_, ⟨i.1.1, of_decide_eq_true hi, rfl⟩, hz⟩
  have hread : ∀ j (hj : j < d.BLOCK_SIZE), j < d.n_cols →
      init.read .f32 (d.x.ptr (inIdx d r j)).addr = some (row.get (⟨j, hj⟩, ())) := by
    intro j hjB hj
    rw [← hv1 (⟨j, hjB⟩, ()) (decide_eq_true hj)]
    exact (Mem.read_congr fun z hz => hag z ⟨_, ⟨j, hj, rfl⟩, hz⟩).symm
  -- lines 101–105
  refine safe_maxAll hBpos true fun mx hmx => ?_
  refine safe_sub fun rmm hrmm => ?_
  refine safe_math (fn := .exp) (fun i => htot _) fun num hnum => ?_
  refine safe_sum hBpos fun den hden => ?_
  refine safe_div (fun i => hdtot _ _) fun out hout => ?_
  -- line 109: the masked store of the row
  refine safe_last (safe_store_buf d.y (g := fun (i : Idx [d.BLOCK_SIZE - 0]) => outIdx d r i.1.1)
    (en := fun (i : Idx [d.BLOCK_SIZE - 0]) => decide (i.1.1 < d.n_cols))
    (Buf.addrs_of_get d.y (fun i => mask_get hB31 hn31 i)
      (fun i _ => ptr_get hB31 d.y hrow hd.out_stride_lt hrs_out i))
    (fun i hi => hd.y_len r _ hr (of_decide_eq_true hi)) ?_ hvy₁ ?_ fun m' hm1 hm2 hm3 => safe_ret ?_)
  · rintro ⟨i, ⟨⟩⟩ ⟨j, ⟨⟩⟩ - - h
    have : i.1 = j.1 := by dsimp only at h; unfold outIdx at h; omega
    rw [Fin.ext this]
  · rintro z ⟨k, ⟨i, hi, rfl⟩, hz⟩
    exact ⟨_, ⟨i.1.1, of_decide_eq_true hi, rfl⟩, hz⟩
  have himg : ∀ z, z ∉ d.y.elems (rowOut d r) →
      z ∉ d.y.elems (Buf.img (fun i : Idx [d.BLOCK_SIZE - 0] => outIdx d r i.1.1)
        (fun i : Idx [d.BLOCK_SIZE - 0] => decide (i.1.1 < d.n_cols))) := by
    rintro z hz ⟨k, ⟨i, hi, rfl⟩, hz'⟩
    exact hz ⟨_, ⟨i.1.1, of_decide_eq_true hi, rfl⟩, hz'⟩
  refine hk m' ?_ (fun z hz => hm2 z (himg z hz)) hm3
  -- the row is done
  set xs : Nat → F32 := fun j => if h : j < d.BLOCK_SIZE then row.get (⟨j, h⟩, ()) else default with hxs
  have hxs_eq : ∀ j (hj : j < d.BLOCK_SIZE), xs j = row.get (⟨j, hj⟩, ()) := fun j hj => by
    rw [hxs]; simp only [hj, dite_true]
  have hreadx : ∀ k, k < d.n_cols → init.read .f32 (d.x.ptr (inIdx d r k)).addr = some (xs k) := by
    intro k hk
    have hkB := lt_of_lt_of_le hk hd.cols_le
    rw [hxs_eq k hkB]; exact hread k hkB hk
  refine ⟨xs, hreadx, ?_⟩
  -- an empty row: nothing to show
  rcases Nat.eq_zero_or_pos d.n_cols with hn0 | hn0
  · refine ⟨1, ?_, fun j hj => absurd hj (by omega)⟩
    rw [sub_self, abs_zero]
    unfold sumTol
    positivity
  have hok : ∀ k, k < d.n_cols → InputOk (xs k) := by
    intro k hk
    obtain ⟨a, ha, hok⟩ := hxin k hk
    rw [hreadx k hk] at ha
    cases ha; exact hok
  have hfin : ∃ k, k < d.n_cols ∧ (xs k).isFinite = true := by
    obtain ⟨k, hk, a, ha, hf⟩ := hxfin hn0
    rw [hreadx k hk] at ha
    cases ha; exact ⟨k, hk, hf⟩
  have hrow_in : ∀ i : Fin d.BLOCK_SIZE, i.1 < d.n_cols → row.get (i, ()) = xs i := fun i _ =>
    (hxs_eq i i.2).symm
  have hrow_out : ∀ i : Fin d.BLOCK_SIZE, d.n_cols ≤ i.1 → row.get (i, ()) = FloatV.negInf := by
    intro i hi
    rw [hv2 (i, ()) (decide_eq_false (show ¬ (i.1 < d.n_cols) by omega)) _ rfl]
    exact negInf_lit
  obtain ⟨sc, hsc, hrows⟩ := row_bound hexp hninf hdiv hd.block_le hd.cols_le xs hok hfin row rmm num
    out mx den hrow_in hrow_out hmx hrmm hnum hden hout
  refine ⟨sc, hsc, fun j hj => ?_⟩
  have hjB := lt_of_lt_of_le hj hd.cols_le
  obtain ⟨hf, hb⟩ := hrows ⟨j, hjB⟩ hj
  exact ⟨out.get (⟨j, hjB⟩, ()), hm1 (⟨j, hjB⟩, ()) (decide_eq_true hj), hf, hb⟩

/-! ## One program -/

/-- The loop of program `p` runs trip `t` (row `p + t·G`) exactly when that row is below `n`. -/
theorem trip_lt_iff {p n G t : Nat} (hG : 0 < G) :
    t < tl.tripCount (p : Int) (n : Int) (G : Int) ↔ p + t * G < n := by
  unfold tl.tripCount
  rw [Int.lt_toNat]
  have hG' : (0 : Int) < G := by exact_mod_cast hG
  constructor
  · intro h
    have h1 : ((t : Int) + 1) ≤ ((n : Int) - p + G - 1) / G := by omega
    rw [Int.le_ediv_iff_mul_le hG'] at h1
    have h2 : ((t : Int) + 1) * G = t * G + G := by ring
    have h3 : ((p + t * G : Nat) : Int) < n := by push_cast; linarith
    exact_mod_cast h3
  · intro h
    have h3 : ((p : Int) + t * G) < n := by exact_mod_cast h
    have h1 : ((t : Int) + 1) ≤ ((n : Int) - p + G - 1) / G := by
      rw [Int.le_ediv_iff_mul_le hG']
      have h2 : ((t : Int) + 1) * G = t * G + G := by ring
      linarith
    omega

theorem rowIn_inSet {d : Inst} {r : Nat} (hr : r < d.n_rows) : rowIn d r ⊆ inSet d := by
  rintro k ⟨j, hj, rfl⟩; exact ⟨r, j, hr, hj, rfl⟩

theorem rowOut_outSet {d : Inst} {r : Nat} (hr : r < d.n_rows) : rowOut d r ⊆ outSet d := by
  rintro k ⟨j, hj, rfl⟩; exact ⟨r, j, hr, hj, rfl⟩

/-- **One program.** Program `p` runs the rows `p, p + G, p + 2G, …` below `n_rows`
(`G = num_programs`), one trip each (`safe_row`); trip `t` writes the outputs of row `p + t·G`,
and different trips write different rows. -/
theorem safe_program {E : Env} (htot : Rules.Num.MathTotal E.P .exp .binary32)
    (hexp : Rules.Num.ExpF32 E.P) (hninf : Rules.Num.ExpF32NegInf E.P)
    (hdtot : Rules.Num.DivTotal E.P .binary32) (hdiv : Rules.Num.DivFull E.P)
    {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} (hp : p < d.num_programs) :
    Safe E (d.x.elems (progIn d p)) (d.y.elems (progOut d p)) (fun _ m => Q d init p m) init
      (tl.forRange (IntV.ofNat p : I32) (IntV.ofNat d.n_rows) (IntV.ofNat d.num_programs) ()
        (fun row_idx _ => rowProg d row_idx)) := by
  have hG : 0 < d.num_programs := by omega
  have hG31 := hd.grid_lt
  have hrg := hd.rows_grid
  have hn31 : d.n_rows < 2 ^ 31 := by omega
  have hlo : (IntV.ofNat p : I32).toInt = p := IntV.I32.toInt_ofNat (by omega)
  have hhi : (IntV.ofNat d.n_rows : I32).toInt = d.n_rows := IntV.I32.toInt_ofNat hn31
  have hst : (IntV.ofNat d.num_programs : I32).toInt = d.num_programs := IntV.I32.toInt_ofNat hG31
  have hrange : tl.RangeOk (IntV.ofNat p : I32) (IntV.ofNat d.n_rows) (IntV.ofNat d.num_programs) := by
    unfold tl.RangeOk
    rw [hlo, hhi, hst]
    simp only [Format.intRange, ite_true]
    refine ⟨by omega, ?_, ?_, ?_⟩ <;> norm_num <;> omega
  have htrip : ∀ t, t < tl.tripCount (IntV.ofNat p : I32).toInt (IntV.ofNat d.n_rows : I32).toInt
      (IntV.ofNat d.num_programs : I32).toInt ↔ p + t * d.num_programs < d.n_rows := by
    intro t; rw [hlo, hhi, hst]; exact trip_lt_iff hG
  set N := tl.tripCount (IntV.ofNat p : I32).toInt (IntV.ofNat d.n_rows : I32).toInt
      (IntV.ofNat d.num_programs : I32).toInt with hN
  set Wt : Nat → Set Int := fun t => d.y.elems (rowOut d (p + t * d.num_programs)) with hWt_def
  have hmod : ∀ t, (p + t * d.num_programs) % d.num_programs = p := by
    intro t; rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp]
  have hrow_lt : ∀ t u, t ≤ u → u < N → p + t * d.num_programs < d.n_rows := by
    intro t u htu hu
    have := (htrip u).mp hu
    have : t * d.num_programs ≤ u * d.num_programs := Nat.mul_le_mul_right _ htu
    omega
  have hWt : ∀ t u, t < u → u < N → Disjoint (Wt t) (Wt u) := by
    intro t u htu hu
    refine rowOut_disjoint hd (hrow_lt t u htu.le hu) ((htrip u).mp hu) fun h => ?_
    have : t * d.num_programs = u * d.num_programs := by omega
    exact htu.ne (Nat.eq_of_mul_eq_mul_right hG this)
  have hWsub : iterW Wt N ⊆ d.y.elems (progOut d p) := by
    rintro z ⟨t, ht, hz⟩
    exact d.y.elems_mono (rowOut_sub ((htrip t).mp ht) (hmod t)) hz
  have hxy : ∀ z ∈ d.x.elems (inSet d), z ∉ d.y.elems (outSet d) := fun z hz hz' =>
    Set.disjoint_left.mp hd.y_x hz' hz
  refine safe_last (Safe.mono (safe_forRange (k := pure) hrange
    (fun t _ m₁ => (∀ z, init z ≠ none → m₁ z ≠ none) ∧ (∀ z, z ∉ iterW Wt t → m₁ z = init z) ∧
      ∀ t', t' < t → RowDone d init m₁ (p + t' * d.num_programs))
    Wt (Wk := ∅) hWt (Set.disjoint_empty _)
    ⟨fun z h => h, fun z _ => rfl, fun t' h => absurd h (Nat.not_lt_zero _)⟩ ?_ ?_)
    (fun _ _ h => h) (Set.union_subset hWsub (Set.empty_subset _)))
  · -- trip `t`: the row `p + t·G`
    intro t ht _ m₁ ⟨hpres, hfr, hdone⟩
    have hr := (htrip t).mp ht
    have hrow : (IntV.ofInt ((IntV.ofNat p : I32).toInt + t * (IntV.ofNat d.num_programs : I32).toInt)
        : I32).toInt = ((p + t * d.num_programs : Nat) : Int) := by
      have hr' : ((p + t * d.num_programs : Nat) : Int) < 2 ^ 31 := by
        have : p + t * d.num_programs < 2 ^ 31 := by omega
        exact_mod_cast this
      have h0 : (0 : Int) ≤ ((p + t * d.num_programs : Nat) : Int) := Int.natCast_nonneg _
      push_cast at hr' h0
      rw [hlo, hst, IntV.I32.toInt_ofInt ⟨by linarith, hr'⟩]
      push_cast; ring
    refine safe_row htot hexp hninf hdtot hdiv hd hpre hr hrow
      (d.x.elems_mono (rowIn_sub hr (hmod t))) hpres ?_ ?_
    · intro z hz
      apply hfr
      rintro ⟨t', ht', hz'⟩
      exact hxy z (d.x.elems_mono (rowIn_inSet hr) hz)
        (d.y.elems_mono (rowOut_outSet (hrow_lt t' t ht'.le ht)) hz')
    · intro m' hdone' hfr' hpres'
      refine ⟨fun z hz => hpres' z (hpres z hz), fun z hz => ?_, fun t' ht' => ?_⟩
      · have h1 : z ∉ iterW Wt t := fun h => hz (iterW_mono (Nat.le_succ t) h)
        have h2 : z ∉ Wt t := fun h => hz ⟨t, Nat.lt_succ_self t, h⟩
        rw [hfr' z h2, hfr z h1]
      · rcases Nat.lt_succ_iff_lt_or_eq.mp ht' with h | rfl
        · exact rowDone_of_agree (hdone t' h) fun z hz => hfr' z fun hz' =>
            Set.disjoint_left.mp (hWt t' t h ht) hz hz'
        · exact hdone'
  · -- after the loop: every row of the program is done
    intro _ m₁ ⟨_, _, hdone⟩
    refine safe_ret fun r hr hrp => ?_
    have hr' : r = p + r / d.num_programs * d.num_programs := by
      have := Nat.mod_add_div r d.num_programs
      rw [hrp, Nat.mul_comm] at this
      omega
    have hlt : r / d.num_programs < N := (htrip _).mpr (by rw [← hr']; exact hr)
    have := hdone (r / d.num_programs) hlt
    rwa [← hr'] at this

/-! ## The launch -/

theorem launch_size (d : Inst) : (launch d).size = d.num_programs := Launch.size_grid1 _ _ _

theorem launch_ctx (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = i ∧ ((launch d).ctx i).grid 0 = d.num_programs := by
  have hi : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.grid_lt fun s hs => ?_
  simp only [List.mem_singleton] at hs
  subst hs
  refine ⟨fun k hk => ?_, ?_⟩
  · simp only [List.mem_singleton] at hk
    subst hk; exact hd.block_pow2
  · simpa using le_trans hd.block_le (by norm_num)

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ i => d.x.elems (progIn d i)) (fun d _ i => d.y.elems (progOut d i))
    (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    obtain ⟨h1, h2⟩ := launch_ctx d i
    rw [body_eq]
    simp only [tl.program_id, tl.num_programs, h1, h2]
    exact safe_program hA.exp_total hA.exp_acc hA.exp_ninf hA.div_total hA.div_full hd hpre
      (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right
      (hd.y_x.mono (d.y.elems_mono (progOut_sub d i)) (d.x.elems_mono (progIn_sub d j)))
      (progOut_disjoint hd hij')
  · -- no program reads what it writes
    intro d hd init _ i
    exact (hd.y_x.mono (d.y.elems_mono (progOut_sub d i)) (d.x.elems_mono (progIn_sub d i))).symm
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun r hr => ?_, fun z hz => hfr z fun i hzi => hz (d.y.elems_mono (progOut_sub d i) hzi)⟩
    have hG : 0 < d.num_programs := hd.grid_pos (by omega)
    have hp : r % d.num_programs < (launch d).size := by
      rw [launch_size]; exact Nat.mod_lt _ hG
    obtain ⟨m, hQ, hag⟩ := hloc ⟨r % d.num_programs, hp⟩
    exact rowDone_of_agree (hQ r hr rfl) fun z hz =>
      hag z (Or.inr (d.y.elems_mono (rowOut_sub hr rfl) hz))

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.ref.mathTotal,
    Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf, Platform.ref.divTotal _,
    Platform.ref.divFull⟩⟩

/-- The source's own test instance (lines 186–188): `x` of shape `(1823, 781)`, contiguous
(`stride(0) = 781`), `BLOCK_SIZE = next_power_of_2(781) = 1024`, `num_stages = 4`, and `132`
programs; `x` and `y` of `1823 · 781` elements at the addresses `2^24` and `2^25`. -/
def testInst : Inst where
  n_rows := 1823
  n_cols := 781
  input_row_stride := 781
  output_row_stride := 781
  BLOCK_SIZE := 1024
  num_stages := 4
  num_programs := 132
  x := ⟨16777216, 1423763⟩
  y := ⟨33554432, 1423763⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

/-- Every aligned `float32` read of `testMem` gives `+0`. -/
theorem testMem_read {a : Int} (ha : Mem.Aligned .f32 a) :
    testMem.read .f32 a = some Rules.Num.zeroF32 := by
  have hb : testMem.readBytes a 4 = some [0, 0, 0, 0] := by
    simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.f32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb]
  decide

theorem testInst_read {r j : Nat} (hr : r < 1823) (hj : j < 781) :
    testMem.read .f32 (testInst.x.ptr (inIdx testInst r j)).addr = some Rules.Num.zeroF32 := by
  refine testMem_read ⟨?_, ?_, ?_⟩ <;>
    simp only [Buf.ptr, testInst, inIdx, DType.bytes] <;> omega

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hinIdx : ∀ r j, r < 1823 → j < 781 → inIdx testInst r j < 1423763 := by
    intro r j hr hj; simp only [inIdx, testInst]; omega
  refine ⟨testInst, testMem, ⟨⟨10, by norm_num [testInst]⟩, by norm_num [testInst],
    by norm_num [testInst], fun _ => by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], hinIdx, hinIdx, Or.inr le_rfl, ?_⟩, ?_, ?_, ?_⟩
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro z hz hz'
      simp only [Buf.bytes, Set.mem_ofPred_eq, testInst, DType.bytes] at hz hz'
      omega
    · rintro k ⟨r, j, hr, hj, rfl⟩; exact hinIdx r j hr hj
    · rintro k ⟨r, j, hr, hj, rfl⟩; exact hinIdx r j hr hj
  · exact ⟨by norm_num [testInst], by norm_num [testInst, DType.bytes],
      by norm_num [testInst, DType.bytes], fun _ _ => by simp [testMem]⟩
  · exact ⟨by norm_num [testInst], by norm_num [testInst, DType.bytes],
      by norm_num [testInst, DType.bytes], fun _ _ => by simp [testMem]⟩
  · intro r hr
    have hr' : r < 1823 := hr
    refine ⟨fun j hj => ⟨_, testInst_read hr' hj, Or.inl ⟨zeroF32_isFinite, ?_⟩⟩,
      fun _ => ⟨0, by norm_num [testInst], _, testInst_read hr' (by norm_num), zeroF32_isFinite⟩⟩
    rw [zeroF32_toRat]; norm_num

end Kernels.Softmax
