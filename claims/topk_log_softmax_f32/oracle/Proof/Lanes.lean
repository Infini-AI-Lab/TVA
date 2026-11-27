import Proof.Out

/-!
# Lanes and addresses

The `int32` lanes `c + tl.arange(0, B)` of a block and their mask against an `int32` bound
(`blk_lane`, `blk_mask`), the block access `b.ptr base + (c + tl.arange(0, B))` as a buffer
relation (`blk_addrs`), the `int64` products `req_idx * s` (`mul64`), the scalar pointers
(`ptr_add64`), and the gather `row_ptr + ids` (`gather_addrs`).
-/
namespace Claims.TopkLogSoftmax

open TVA

/-- Lane `i` of `c + tl.arange(0, B)`, for an `int32` `c` of value `r0` with `r0 + B ≤ 2^31`. -/
lemma blk_lane {c : I32} {r0 B : Nat} (hc : c.toInt = r0) (h : r0 + B ≤ 2 ^ 31)
    (i : Idx [B - 0]) : ((c + tl.arange 0 B).get i).toInt = ((r0 + i.1.1 : Nat) : Int) := by
  obtain ⟨l, ⟨⟩⟩ := i
  exact IntV.I32.toInt_sadd_arange hc h l

/-- Lane `i` of the mask `c + tl.arange(0, B) < n`. -/
lemma blk_mask {c : I32} {r0 B n : Nat} (hc : c.toInt = r0) (h : r0 + B ≤ 2 ^ 31)
    (hn : n < 2 ^ 31) (i : Idx [B - 0]) :
    ((c + tl.arange 0 B).ltS (IntV.ofNat n : I32)).get i = decide (r0 + i.1.1 < n) := by
  rw [Tensor.get_ltS, IntV.lt_eq, blk_lane hc h i, IntV.I32.toInt_ofNat hn]
  simp only [decide_eq_decide]
  norm_cast

/-- The block access `b.ptr base + (c + tl.arange(0, B))` masked by `< n`: lane `i` is element
`base + r0 + i`, enabled when `r0 + i < n`. -/
lemma blk_addrs {dt : DType} (b : Buf dt) (base : Nat) {c : I32} {r0 B n : Nat}
    (hc : c.toInt = r0) (h : r0 + B ≤ 2 ^ 31) (hn : n < 2 ^ 31) :
    b.Addrs (tl.addrs (b.ptr (base : Int) + (c + tl.arange 0 B))
        (some ((c + tl.arange 0 B).ltS (IntV.ofNat n : I32))))
      (fun i => base + (r0 + i.1.1)) (fun i => decide (r0 + i.1.1 < n)) := by
  refine Buf.addrs_of_get b (fun i => blk_mask hc h hn i) fun i _ => ?_
  rw [Tensor.get_ptrAdd, blk_lane hc h i, Buf.ptr_offset]
  congr 1

/-- The elements a masked block access touches. -/
lemma blk_img {B n base r0 : Nat} :
    Buf.img (s := [B - 0]) (fun i => base + (r0 + i.1.1)) (fun i => decide (r0 + i.1.1 < n)) ⊆
      {e | ∃ k, k < n ∧ e = base + k} := by
  rintro e ⟨⟨l, u⟩, hl, rfl⟩
  exact ⟨r0 + l, of_decide_eq_true hl, rfl⟩

/-- An `int64` product of two values below `2^31`. -/
lemma mul64 {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (hx : x < 2 ^ 31)
    (hy : y < 2 ^ 31) : (a * b).toInt = ((x * y : Nat) : Int) := by
  have hfit : IntV.SFits 64 ((x : Int) * (y : Int)) := by
    rw [IntV.sfits_iff]
    have h0 : (0 : Int) ≤ (x : Int) * (y : Int) := by positivity
    have h1 : (x : Int) * (y : Int) < 2 ^ 31 * 2 ^ 31 := by
      have hx' : (x : Int) < 2 ^ 31 := by exact_mod_cast hx
      have hy' : (y : Int) < 2 ^ 31 := by exact_mod_cast hy
      have : (0 : Int) ≤ (y : Int) := by positivity
      nlinarith
    norm_num at h1 ⊢
    constructor <;> linarith
  rw [IntV.toInt_mul ha hb hfit]
  push_cast; ring

/-- `x.to(tl.int64)` of an `int32` keeps its value. -/
lemma cast64 {x : I32} {r : I64} (h : ConvRel P .i32 .i64 x r) : r.toInt = x.toInt := by
  rw [ConvRel.iff_cast_i32_i64.mp h]
  exact IntV.toInt_cast x (by norm_num)

/-- The scalar pointer `b.ptr 0 + x` for an `int64` `x` of value `k`. -/
lemma ptr_add64 {dt : DType} (b : Buf dt) {x : I64} {k : Nat} (hx : x.toInt = k) :
    b.ptr 0 + x = b.ptr (k : Int) := by
  rw [Ptr.add_int, Buf.ptr_offset, hx, zero_add]

/-- The gather `b.ptr base + ids`, masked by `mask`: an enabled lane whose id is a natural number
`v` is element `base + v`. -/
lemma gather_addrs {dt : DType} {s : Shape} (b : Buf dt) (base : Nat) {ids : Tensor s I64}
    {mask : Tensor s Bool} (hnn : ∀ i, mask.get i = true → 0 ≤ (ids.get i).toInt) :
    b.Addrs (tl.addrs (b.ptr (base : Int) + ids) (some mask))
      (fun i => base + (ids.get i).toInt.toNat) (fun i => mask.get i) := by
  refine Buf.addrs_of_get b (fun i => rfl) fun i hi => ?_
  rw [Tensor.get_ptrAdd, Buf.ptr_offset]
  congr 1
  push_cast
  rw [Int.toNat_of_nonneg (hnn i hi)]

/-! ## The launch -/

/-- The grid has `batch` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.batch := Launch.size_grid1 _ _ _

/-- Program `i` has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.batch := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid: the grid is below `2^31` and both block shapes are powers of two. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.batch_lt fun s hs => ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  obtain ⟨e, he, hTB⟩ := hd.tb
  rcases hs with rfl | rfl
  · refine ⟨fun k hk => ?_, by norm_num⟩
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨10, by norm_num⟩
  · refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_singleton] at hk
      subst hk
      exact ⟨e, hTB⟩
    · have : 2 ^ e ≤ 2 ^ 10 := Nat.pow_le_pow_right (by norm_num) he
      simp only [List.prod_cons, List.prod_nil, mul_one, hTB]
      omega

end Claims.TopkLogSoftmax
