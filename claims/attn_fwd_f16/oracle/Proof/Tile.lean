import Proof.Outer

/-!
# The host descriptors' tile accesses

A tile of `BM` rows of a tensor through `hostDesc d b BM`, at the row offset `ro` (and column
`0`): every lane is inside the tensor when the rows `ro, …, ro + BM − 1` are below `rows d`, and
lane `(r, e)` is element `(ro + r)·HEAD_DIM + e` of the buffer.
-/

namespace Claims.AttnFwd

open TVA

variable {d : Inst}

/-- The host descriptor has rank 2. -/
theorem hostDesc_rank (d : Inst) (b : Buf .f16) (BM : Nat) : (hostDesc d b BM).rank = 2 := rfl

/-- The descriptor of a buffer of the instance is valid. -/
theorem hostDesc_valid (hd : d.ok) {b : Buf .f16} (hb : b.base % 16 = 0) (BM : Nat)
    (_hlen : rows d * d.HEAD_DIM ≤ b.len) :
    (hostDesc d b BM).Valid := by
  have hD := hd.head_dim
  have hrows := hd.rows_lt
  have hn := hd.n_mul.1
  have hzh := hd.zh_pos
  have hrpos : 0 < rows d := by
    unfold rows; exact Nat.mul_pos (Nat.mul_pos hzh.1 hzh.2) hn
  have hDpos : 0 < d.HEAD_DIM := by omega
  refine ⟨by simp [TensorDesc.rank, hostDesc], by simp [TensorDesc.rank, hostDesc], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [TensorDesc.rank, hostDesc]
  · simp [TensorDesc.rank, hostDesc]
  · intro k hk
    simp only [TensorDesc.rank, hostDesc, List.length_cons, List.length_nil] at hk
    interval_cases k
    · simp [hostDesc]; omega
    · simp [hostDesc]; omega
  · simp [hostDesc, Buf.ptr, hb]
  · simp [hostDesc]
  · intro k hk
    simp only [TensorDesc.rank, hostDesc, List.length_cons, List.length_nil] at hk
    have hk0 : k = 0 := by omega
    subst hk0
    simp [hostDesc, DType.bytes]; omega
  · simp [hostDesc]
  · intro h; simp [hostDesc] at h

/-- The integer value of `IntV.ofInt 0` as `I32` is 0. -/
theorem ofInt_zero' : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

/-- Mapping `toInt` over the offsets `[ro, 0]` gives `[ro.toInt, 0]`. -/
theorem offsets_map (ro : I32) : ([ro, IntV.ofInt 0] : List I32).map IntV.toInt = [ro.toInt, 0] := by
  simp [ofInt_zero']

/-- The access of a block at the row offset `ro`, column `0`. -/
theorem hostDesc_access (hd : d.ok) {b : Buf .f16} (hb : b.base % 16 = 0) (BM : Nat)
    (hlen : rows d * d.HEAD_DIM ≤ b.len) (ro : I32) :
    (hostDesc d b BM).Access (([ro, IntV.ofInt 0] : List I32).map IntV.toInt) [BM, d.HEAD_DIM] := by
  refine ⟨hostDesc_valid hd hb BM hlen, rfl, rfl, by simp [hostDesc, TensorDesc.rank], ?_⟩
  simp [ofInt_zero']

/-- Coordinate 0 of a tile index is its first component. -/
theorem coord0 {BM : Nat} (i : Idx [BM, d.HEAD_DIM]) : i.coord 0 = (i.1 : Nat) := by
  rcases i with ⟨a, b, ⟨⟩⟩; simp [Idx.coord]

/-- Coordinate 1 of a tile index is its second component. -/
theorem coord1 {BM : Nat} (i : Idx [BM, d.HEAD_DIM]) : i.coord 1 = (i.2.1 : Nat) := by
  rcases i with ⟨a, b, ⟨⟩⟩; simp [Idx.coord]

/-- Lane `(r, e)` of a block at row offset `ro` is element `(ro + r)·HEAD_DIM + e`. -/
theorem hostDesc_lane (b : Buf .f16) (BM : Nat) (ro : I32) (hro : 0 ≤ ro.toInt)
    (i : Idx [BM, d.HEAD_DIM]) :
    (hostDesc d b BM).lane (bs := [BM, d.HEAD_DIM]) [ro, IntV.ofInt 0] i =
      (ro.toInt.toNat + (i.1 : Nat)) * d.HEAD_DIM + (i.2.1 : Nat) := by
  simp [TensorDesc.lane, TensorDesc.elemAt, TensorDesc.coord, hostDesc, TensorDesc.rank, List.range,
    List.range.loop, ofInt_zero', coord0, coord1]
  obtain ⟨n, hn⟩ : ∃ n : ℕ, ro.toInt = n := ⟨ro.toInt.toNat, (Int.toNat_of_nonneg hro).symm⟩
  rw [hn]
  have : ((n : Int) + (i.1 : Nat)) * d.HEAD_DIM + (i.2.1 : Nat) =
      (((n + (i.1 : Nat)) * d.HEAD_DIM + (i.2.1 : Nat) : Nat) : Int) := by
    push_cast; ring
  rw [this, Int.toNat_natCast, Int.toNat_natCast]

/-- Every lane of a block that lies within the rows is inside the tensor. -/
theorem hostDesc_laneIn (b : Buf .f16) (BM : Nat) (ro : I32) (hro : 0 ≤ ro.toInt)
    (hrows : ro.toInt + BM ≤ rows d) (i : Idx [BM, d.HEAD_DIM]) :
    (hostDesc d b BM).laneIn (bs := [BM, d.HEAD_DIM]) [ro, IntV.ofInt 0] i = true := by
  simp only [TensorDesc.laneIn, TensorDesc.inbAt, decide_eq_true_eq]
  intro k hk
  simp only [hostDesc, TensorDesc.rank, List.length_cons, List.length_nil] at hk
  have h0 := i.1.isLt
  have h1 := i.2.1.isLt
  interval_cases k
  · simp [TensorDesc.coord, hostDesc, coord0, ofInt_zero']
    omega
  · simp [TensorDesc.coord, hostDesc, coord1, ofInt_zero']

/-- The element a lane of such a block addresses is inside the buffer. -/
theorem hostDesc_lane_lt (b : Buf .f16) (BM : Nat) (ro : I32) (hro : 0 ≤ ro.toInt)
    (hrows : ro.toInt + BM ≤ rows d)
    (i : Idx [BM, d.HEAD_DIM]) :
    (hostDesc d b BM).lane (bs := [BM, d.HEAD_DIM]) [ro, IntV.ofInt 0] i < rows d * d.HEAD_DIM := by
  rw [hostDesc_lane b BM ro hro i]
  have h0 := i.1.isLt
  have h1 := i.2.1.isLt
  have h2 : ro.toInt.toNat + BM ≤ rows d := by omega
  have h3 : (ro.toInt.toNat + (i.1 : Nat) + 1) * d.HEAD_DIM ≤ rows d * d.HEAD_DIM :=
    Nat.mul_le_mul_right _ (by omega)
  nlinarith

/-- Distinct lanes of such a block address distinct elements. -/
theorem hostDesc_lane_inj (b : Buf .f16) (BM : Nat) (ro : I32) (hro : 0 ≤ ro.toInt)
    (i j : Idx [BM, d.HEAD_DIM])
    (h : (hostDesc d b BM).lane (bs := [BM, d.HEAD_DIM]) [ro, IntV.ofInt 0] i =
      (hostDesc d b BM).lane (bs := [BM, d.HEAD_DIM]) [ro, IntV.ofInt 0] j) : i = j := by
  rw [hostDesc_lane b BM ro hro i, hostDesc_lane b BM ro hro j] at h
  have hi := i.2.1.isLt
  have hj := j.2.1.isLt
  have hD : 0 < d.HEAD_DIM := by omega
  have e1 : ((ro.toInt.toNat + (i.1 : Nat)) * d.HEAD_DIM + (i.2.1 : Nat)) / d.HEAD_DIM =
      ro.toInt.toNat + (i.1 : Nat) := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hi, zero_add]
  have e2 : ((ro.toInt.toNat + (j.1 : Nat)) * d.HEAD_DIM + (j.2.1 : Nat)) / d.HEAD_DIM =
      ro.toInt.toNat + (j.1 : Nat) := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hj, zero_add]
  have hr : (i.1 : Nat) = (j.1 : Nat) := by
    have := congrArg (· / d.HEAD_DIM) h
    simp only [e1, e2] at this; omega
  have hc : (i.2.1 : Nat) = (j.2.1 : Nat) := by
    rw [hr] at h; omega
  rcases i with ⟨a, b', ⟨⟩⟩; rcases j with ⟨a', b'', ⟨⟩⟩
  have ha : a = a' := Fin.ext hr
  have hb : b' = b'' := Fin.ext hc
  subst ha; subst hb; rfl

section Rules
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **A tile load through a host descriptor**: the block of `BM` rows at row offset `ro` (inside the
tensor) holds, lane by lane, the elements `(ro + r)·HEAD_DIM + e` of the buffer. -/
theorem safe_tile_load (hd : d.ok) {b : Buf .f16} (hb : b.base % 16 = 0)
    (hlen : rows d * d.HEAD_DIM ≤ b.len) (hv : b.Valid m) (BM : Nat) (ro : I32)
    (hro : 0 ≤ ro.toInt) (hrows : ro.toInt + BM ≤ rows d)
    (hR : b.elems (below (rows d * d.HEAD_DIM)) ⊆ R)
    {k : Tensor [BM, d.HEAD_DIM] F16 → Prog α}
    (hk : ∀ v : Tensor [BM, d.HEAD_DIM] F16,
      (∀ (r : Fin BM) (e : Fin d.HEAD_DIM),
        m.read .f16 (b.ptr (((ro.toInt.toNat + r) * d.HEAD_DIM + e : Nat) : Int)).addr =
          some (v.get2 r e)) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_load (bs := [BM, d.HEAD_DIM]) (hostDesc d b BM) [ro, IntV.ofInt 0] >>= k) := by
  refine safe_desc_load b (D := hostDesc d b BM) (offsets := [ro, IntV.ofInt 0]) rfl
    (hostDesc_access hd hb BM hlen ro)
    (fun i _ => (hostDesc_lane_lt b BM ro hro hrows i).trans_le hlen) hv ?_ fun v hv1 _ => hk v fun r e => ?_
  · refine (b.elems_mono ?_).trans hR
    rintro _ ⟨i, _, rfl⟩
    exact hostDesc_lane_lt b BM ro hro hrows i
  · have := hv1 (r, e, ()) (hostDesc_laneIn b BM ro hro hrows _)
    have e1 := hostDesc_lane (d := d) b BM ro hro (r, e, ())
    rw [e1] at this
    exact this

/-- **A tile store through a host descriptor**: the block of `BM` rows at row offset `ro` (inside
the tensor) is written, lane by lane, to the elements `(ro + r)·HEAD_DIM + e`; the rest of memory
is unchanged. -/
theorem safe_tile_store (hd : d.ok) {b : Buf .f16} (hb : b.base % 16 = 0)
    (hlen : rows d * d.HEAD_DIM ≤ b.len) (hv : b.Valid m) (BM : Nat) (ro : I32)
    (hro : 0 ≤ ro.toInt) (hrows : ro.toInt + BM ≤ rows d) {vals : Tensor [BM, d.HEAD_DIM] F16}
    (S : Set Nat) (hS : ∀ (r : Fin BM) (e : Fin d.HEAD_DIM),
      (ro.toInt.toNat + r) * d.HEAD_DIM + e ∈ S)
    (hW : b.elems S ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ (r : Fin BM) (e : Fin d.HEAD_DIM),
        m'.read .f16 (b.ptr (((ro.toInt.toNat + r) * d.HEAD_DIM + e : Nat) : Int)).addr =
          some (vals.get2 r e)) →
      (∀ x, x ∉ b.elems S → m' x = m x) → Safe E R (W \ b.elems S) Q m' (k ())) :
    Safe E R W Q m (tl.desc_store (hostDesc d b BM) [ro, IntV.ofInt 0] vals >>= k) := by
  have himg : Buf.img ((hostDesc d b BM).lane (bs := [BM, d.HEAD_DIM]) [ro, IntV.ofInt 0])
      ((hostDesc d b BM).laneIn [ro, IntV.ofInt 0]) ⊆ S := by
    rintro _ ⟨i, _, rfl⟩
    rw [hostDesc_lane b BM ro hro i]
    exact hS i.1 i.2.1
  refine safe_desc_store b (D := hostDesc d b BM) (offsets := [ro, IntV.ofInt 0]) rfl
    (hostDesc_access hd hb BM hlen ro)
    (fun i _ => (hostDesc_lane_lt b BM ro hro hrows i).trans_le hlen)
    (fun i j _ _ h => hostDesc_lane_inj b BM ro hro i j h) hv ((b.elems_mono himg).trans hW)
    fun m' h1 h2 _ => ?_
  refine Safe.mono (hk m' (fun r e => ?_) fun x hx => h2 x fun hx' => hx (b.elems_mono himg hx'))
    (fun _ _ h => h) (fun x hx => ⟨hx.1, fun hx' => hx.2 (b.elems_mono himg hx')⟩)
  have := h1 (r, e, ()) (hostDesc_laneIn b BM ro hro hrows _)
  have e1 := hostDesc_lane (d := d) b BM ro hro (r, e, ())
  rw [e1] at this
  exact this

end Rules

end Claims.AttnFwd

