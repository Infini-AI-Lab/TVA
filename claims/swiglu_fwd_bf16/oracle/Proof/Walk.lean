import Claim
import TVA.Proof
import Proof.Geo
import Proof.Lane

/-!
# `swiglu_fwd_bf16`: the `Safe` walk of one program

`safe_program`: program `p` of the grid, walked line by line. First the index arithmetic
(`program_id` and `arange` widened to `int64`, `offs`, the promoted integer arguments
`safe_arg64`, `mask`, `row`, `col`), whose lanes are computed by `lane_offs`, `lane_ptr`; then the
two loads, the widenings, `tl.sigmoid`'s four steps, the two products, the rounding and the store;
the stored values satisfy `ZOk` by `lane`.
-/

set_option maxHeartbeats 1000000

namespace Claims.SwigluFwd

open TVA

/-- An integer argument as Triton passes it (`argOf`), promoted to `int64` (`arg64`): its value. -/
theorem safe_arg64 {α : Type} {E : Env} {R W : Set Int} {Qp : α → Mem → Prop} {m : Mem} {n : Nat}
    (hn : n < 2 ^ 63) {k : I64 → Prog α} (hk : ∀ v : I64, v.toInt = n → Safe E R W Qp m (k v)) :
    Safe E R W Qp m (arg64 (argOf n) >>= k) := by
  unfold argOf
  split_ifs with h
  · simp only [arg64, bind_assoc, pure_bind]
    rw [safe_convert_iff (y := Tensor.scalar (IntV.cast (IntV.ofNat n : I32) : I64))
      (fun _ _ => ConvRel.iff_cast_i32_i64)]
    refine hk _ ?_
    show (IntV.cast (IntV.ofNat n : I32) : I64).toInt = n
    rw [IntV.toInt_cast _ (by norm_num), IntV.I32.toInt_ofNat h]
  · simp only [arg64, pure_bind]
    exact hk _ (IntV.I64.toInt_ofNat hn)

/-- The block `offs = i_n·B + o_a` has, in lane `i`, the flat element `p·B + i`. -/
theorem lane_offs {B p : Nat} {i_n : I64} {o_a : Tensor [B] I64}
    (hi_n : i_n.toInt = (p : Int))
    (hoa : ∀ i : Idx [B], (o_a.get i).toInt = (i.1.val : Int))
    (hp : p * B + B ≤ 2 ^ 51) (i : Idx [B]) :
    ((i_n * (IntV.ofNat B : I64) + o_a).get i).toInt = ((p * B + i.1.val : Nat) : Int) := by
  have hB : B < 2 ^ 63 := by omega
  have hBv : (IntV.ofNat B : I64).toInt = (B : Int) := IntV.I64.toInt_ofNat hB
  have hmul : (i_n * (IntV.ofNat B : I64)).toInt = ((p * B : Nat) : Int) :=
    IntV.I64.mul_nat hi_n hBv (by omega)
  rw [Tensor.get_sadd]
  have := i.1.isLt
  exact IntV.I64.add_nat hmul (hoa i) (by omega)

/-- The pointers `q + row * s + col` of an enabled lane: element `(k / D)·s + k % D` of `b`, for
the lane's flat element `k`. -/
theorem lane_ptr {dt : DType} {B D : Nat} (b : Buf dt) {row col : Tensor [B] I64} {sv : I64} {s : Nat}
    {i : Idx [B]} {k : Nat} (hrow : (row.get i).toInt = ((k / D : Nat) : Int))
    (hcol : (col.get i).toInt = ((k % D : Nat) : Int)) (hs : sv.toInt = s)
    (hfit : k / D * s < 2 ^ 63) :
    (b.ptr 0 + row * sv + col).get i = b.ptr ((k / D * s + k % D : Nat) : Int) := by
  have hm : (row.get i * sv).toInt = ((k / D * s : Nat) : Int) :=
    IntV.I64.mul_nat hrow hs hfit
  rw [Tensor.get_ptrAddT, Tensor.get_ptrAdd, Tensor.get_muls, Ptr.offset_offset, hm, hcol,
    Buf.ptr_offset]
  congr 1
  push_cast
  ring

/-- The bounds of program `p`'s block: `p·B + B ≤ 2^51`. -/
theorem p_bound {d : Inst} (hd : d.ok) {p : Nat} (hp : p < G d) : p * d.B + d.B ≤ 2 ^ 51 := by
  have hg : G d < 2 ^ 31 := hd.grid_lt
  have h31 : p + 1 ≤ 2 ^ 31 := by omega
  calc p * d.B + d.B = (p + 1) * d.B := by ring
    _ ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul h31 hd.b_le
    _ = 2 ^ 51 := by norm_num

/-- `T = N·D < 2^51`. -/
theorem t_lt {d : Inst} (hd : d.ok) : d.N * d.D < 2 ^ 51 := by
  have hB := b_pos hd
  have hg := hd.grid_lt
  rw [Nat.div_lt_iff_lt_mul hB] at hg
  have h2 : 2 ^ 31 * d.B ≤ 2 ^ 51 := by
    calc 2 ^ 31 * d.B ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul_left _ hd.b_le
      _ = 2 ^ 51 := by norm_num
  omega

/-- A valid buffer of `bfloat16` holds at most `2^63` elements. -/
theorem len_le {b : Buf .bf16} {m : Mem} (hv : b.Valid m) : b.len ≤ 2 ^ 63 := by
  obtain ⟨h0, h1, -, -⟩ := hv
  have : DType.bytes DType.bf16 = 2 := rfl
  rw [this] at h1
  omega

/-- **One program** of the launch, the one with `program_id(0) = p`, is safe from `init`: it
reads its elements of `x` and `y`, writes its block of `z`, and ends with `Q`. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {p : Nat} (hp : p < G d) (c : Ctx) (hc : c.pid 0 = p) :
    Safe ((launch d).env P) (Rset d p) (Wset d p) (fun _ m => Q d init p m) init
      ((model d).body c) := by
  have hpB := p_bound hd hp
  have hT := t_lt hd
  have hD := hd.d_pos
  obtain ⟨hvx, hvy, hvz, hin⟩ := hpre
  have hxl := len_le hvx
  have hyl := len_le hvy
  have hG : G d < 2 ^ 31 := hd.grid_lt
  have hpid : (tl.program_id c 0).toInt = (p : Int) := by
    rw [tl.program_id, hc]; exact IntV.I32.toInt_ofNat (by omega)
  simp only [model, swiglu_fwd_kernel]
  rw [safe_convert_iff (y := Tensor.scalar (IntV.cast (tl.program_id c 0) : I64))
    (fun _ _ => ConvRel.iff_cast_i32_i64)]
  rw [safe_convert_iff (y := (tl.arange 0 d.B).map (fun v => (IntV.cast v : I64)))
    (fun _ _ => ConvRel.iff_cast_i32_i64)]
  have hi_n : (Tensor.scalar (IntV.cast (tl.program_id c 0) : I64)).item.toInt = (p : Int) := by
    show (IntV.cast (tl.program_id c 0) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), hpid]
  have hoa : ∀ i : Idx [d.B],
      (((tl.arange 0 d.B).map (fun v => (IntV.cast v : I64))).get i).toInt = (i.1.val : Int) := by
    rintro ⟨i, ⟨⟩⟩
    have h1 : (((tl.arange 0 d.B).map (fun v => (IntV.cast v : I64))).get (i, ())) =
        IntV.cast ((tl.arange 0 d.B).get1 i) := rfl
    rw [h1, IntV.toInt_cast _ (by norm_num)]
    exact IntV.I32.toInt_tl_arange (n := d.B) (by have := hd.b_le; omega) i
  have hoffs := lane_offs hi_n hoa hpB
  refine safe_arg64 (n := d.T) (by unfold Inst.T; omega) fun T64 hT64 => ?_
  -- `row`, `col`
  have hx : ∀ i : Fin d.B, (((Tensor.scalar (IntV.cast (tl.program_id c 0) : I64)).item *
      (IntV.ofNat d.B : I64) + (tl.arange 0 d.B).map (fun v => (IntV.cast v : I64))).get1 i).toInt =
      ((p * d.B + i.val : Nat) : Int) := fun i => hoffs (i, ())
  have hy : ∀ i : Fin d.B, ((bc (IntV.ofNat d.D : I64) : Tensor [d.B] I64).get1 i).toInt =
      (d.D : Int) := fun _ => IntV.I64.toInt_ofNat (by have := hd.d_lt; omega)
  refine safe_idiv (idiv_ok hx hy hD) fun row hrow => ?_
  have hrowv := idiv_toInt hx hy hD hrow
  refine safe_irem (irem_ok hx hy hD) fun col hcol => ?_
  have hcolv := irem_toInt hx hy hD hcol
  -- the mask
  have hmask : ∀ i, ((((Tensor.scalar (IntV.cast (tl.program_id c 0) : I64)).item *
      (IntV.ofNat d.B : I64) + (tl.arange 0 d.B).map (fun v => (IntV.cast v : I64)))).ltS T64).get i
      = en d p i := by
    intro i
    rw [Tensor.get_ltS, IntV.lt_eq, hoffs i, hT64, en]
    unfold Inst.T
    simp only [decide_eq_decide]
    norm_cast
  -- lane facts of an enabled lane
  have hen : ∀ i : Idx [d.B], en d p i = true → gk d p i < d.N * d.D := by
    intro i hi; simp only [en, decide_eq_true_eq] at hi; exact hi
  have hrc : ∀ i : Idx [d.B], (row.get i).toInt = ((gk d p i / d.D : Nat) : Int) ∧
      (col.get i).toInt = ((gk d p i % d.D : Nat) : Int) := by
    rintro ⟨i, ⟨⟩⟩; exact ⟨hrowv i, hcolv i⟩
  -- an input with row stride `s` (`x` or `y`): its pointers and bounds
  have hptrIn : ∀ (b : Buf .bf16) (s : Nat) (sv : I64), sv.toInt = s →
      (∀ r, r < d.N → r * s + d.D ≤ b.len) → b.len ≤ 2 ^ 63 →
      (∀ i, en d p i = true → (b.ptr 0 + row * sv + col).get i = b.ptr (gin d s p i : Int)) ∧
      (∀ i, en d p i = true → gin d s p i < b.len) := by
    intro b s sv hs hlen hbl
    refine ⟨fun i hi => ?_, fun i hi => ?_⟩
    · obtain ⟨h1, -⟩ := row_col hd (hen i hi)
      have := hlen _ h1
      exact lane_ptr b (hrc i).1 (hrc i).2 hs (by omega)
    · obtain ⟨h1, h2⟩ := row_col hd (hen i hi)
      have := hlen _ h1
      show gk d p i / d.D * s + gk d p i % d.D < b.len
      omega
  -- line 99: `x`
  refine safe_arg64 (n := d.sx) hd.sx_lt fun sx hsx => ?_
  obtain ⟨hpx, hlx⟩ := hptrIn d.x d.sx sx hsx hd.x_len hxl
  refine safe_load_buf d.x (Buf.addrs_of_get d.x hmask hpx) hlx hvx Set.subset_union_left
    fun b_x16 hx16 _ => ?_
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ fun b_x hbx => ?_
  -- line 100: `y`
  refine safe_arg64 (n := d.sy) hd.sy_lt fun sy hsy => ?_
  obtain ⟨hpy, hly⟩ := hptrIn d.y d.sy sy hsy hd.y_len hyl
  refine safe_load_buf d.y (Buf.addrs_of_get d.y hmask hpy) hly hvy Set.subset_union_right
    fun b_y16 hy16 _ => ?_
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ fun b_y hby => ?_
  -- line 101: `tl.sigmoid`, then the two products
  simp only [tl.sigmoid, bind_assoc]
  refine safe_neg fun nx hnx => ?_
  refine safe_math (fn := .exp) (fun i => hP.mathTotal _ _ _) fun ex hex => ?_
  refine safe_add fun den hden => ?_
  refine safe_div (fun i => hP.divTotal _ _ _) fun sg hsg => ?_
  refine safe_mul fun xs hxs => ?_
  refine safe_mul fun b_z hbz => ?_
  -- line 102: the rounding and the store
  refine safe_arg64 (n := d.D) (by have := hd.d_lt; omega) fun sz hsz => ?_
  refine safe_cvt (g := .bfloat16) (rnd := .rn)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ fun b_z16 hbz16 => ?_
  refine safe_last ?_
  have hpz : ∀ i, en d p i = true → (d.z.ptr 0 + row * sz + col).get i = d.z.ptr (gk d p i : Int) := by
    intro i hi
    have h := lane_ptr d.z (hrc i).1 (hrc i).2 hsz
      (by have := Nat.div_mul_le_self (gk d p i) d.D; have := hen i hi; omega)
    rw [h]
    congr 2
    exact Nat.div_add_mod' (gk d p i) d.D
  refine safe_store_buf d.z (Buf.addrs_of_get d.z hmask hpz)
    (fun i hi => by have := hen i hi; have := hd.z_len; omega)
    (fun i j _ _ h => ?_) hvz (by rw [img_gk]; exact le_refl _) fun m' hm' _ _ => safe_ret ?_
  · rcases i with ⟨i, ⟨⟩⟩; rcases j with ⟨j, ⟨⟩⟩
    change p * d.B + i.val = p * d.B + j.val at h
    exact Prod.ext (Fin.ext (show i.val = j.val by omega)) rfl
  · rintro k ⟨hk1, hk2, hk3⟩
    let i : Idx [d.B] := (⟨k - p * d.B, by omega⟩, ())
    have hgi : gk d p i = k := by simp only [gk, i]; omega
    have hei : en d p i = true := by simp only [en, decide_eq_true_eq, i]; omega
    obtain ⟨h1, h2⟩ := row_col hd hk3
    obtain ⟨a, b, ha, hb, haf, hbf, hz⟩ := hin _ h1 _ h2
    have ein : ∀ s : Nat, ((gin d s p i : Nat) : Int) =
        ((k / d.D : Nat) : Int) * (s : Int) + ((k % d.D : Nat) : Int) := by
      intro s; simp only [gin, inIdx, hgi]; push_cast; ring
    have ein' : ∀ s : Nat, ((inIdx d s k : Nat) : Int) =
        ((k / d.D : Nat) : Int) * (s : Int) + ((k % d.D : Nat) : Int) := by
      intro s; simp only [inIdx]; push_cast; ring
    have hxa := hx16 i hei
    rw [ein, ha] at hxa
    have hyb := hy16 i hei
    rw [ein, hb] at hyb
    have hza := hm' i hei
    rw [hgi] at hza
    refine ⟨a, b, b_z16.get i, by rw [ein']; exact ha, by rw [ein']; exact hb, hza, ?_⟩
    have hax : b_x16.get i = a := (Option.some.inj hxa).symm
    have hby' : b_y16.get i = b := (Option.some.inj hyb).symm
    have hden' := hden i
    have hsg' := hsg i
    simp only [Tensor.get_full] at hden' hsg'
    rw [← hax, ← hby'] at hz ⊢
    rw [← hax] at haf
    rw [← hby'] at hbf
    exact lane hP.expF32 hP.expF32Overflow hP.divFull hP.divF32InfDivisor haf hbf hz
      (hbx i) (hby i) (hnx i) (hex i) hden' hsg' (hxs i) (hbz i) (hbz16 i)

end Claims.SwigluFwd
