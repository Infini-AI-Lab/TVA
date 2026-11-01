import AttnFwdE5m2.Cen

/-!
# `attn_fwd_e5m2`: facts about the launch, the integers, the descriptors and the constants

* the grid: program `i` is `(start_m, off_hz) = (i % g0, i / g0)` (`pm`, `pbh`), and the launch is
  valid;
* `int32` arithmetic on small values;
* the descriptors: both kinds of `DescArg` give the same row-major descriptor of `q`, `k`, `o`
  (`descOf`), and the same descriptor of the transposed `v` (`descV`: shape `[HEAD_DIM, y_dim]`,
  strides `[N_CTX, 1]`); their accesses meet Triton's requirements;
* the constants of lines 214, 215, 219 and 76;
* `qk_scale`: `qk_scale·ln 2` is within `2^-22·sm_scale` of `sm_scale`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The grid -/

/-- The grid's `x`, `cdiv(N_CTX, 128)`. -/
def g0 (d : Inst) : Nat := (d.N_CTX + 127) / 128

/-- Program `i`'s `start_m` (line 189). -/
def pm (d : Inst) (i : Nat) : Nat := i % g0 d

/-- Program `i`'s `off_hz` (line 190). -/
def pbh (d : Inst) (i : Nat) : Nat := i / g0 d

theorem g0_eq {d : Inst} (hd : d.ok) : g0 d = d.N_CTX / 128 := by
  unfold g0; have := hd.n_mul; omega

theorem g0_pos {d : Inst} (hd : d.ok) : 0 < g0 d := by
  rw [g0_eq hd]; have := hd.n_mul; omega

theorem launch_size (d : Inst) : (launch d).size = g0 d * (d.Z * d.H) := by
  simp [Launch.size, launch, g0]

theorem pm_lt {d : Inst} (hd : d.ok) (i : Nat) : pm d i < g0 d := Nat.mod_lt _ (g0_pos hd)

theorem pbh_lt {d : Inst} (_hd : d.ok) {i : Nat} (hi : i < (launch d).size) : pbh d i < d.Z * d.H := by
  rw [launch_size] at hi
  unfold pbh
  exact Nat.div_lt_of_lt_mul hi

theorem pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = pm d i := by
  simp [Launch.ctx, launch, pm, g0]

theorem pid1 {d : Inst} (hd : d.ok) {i : Nat} (hi : i < (launch d).size) :
    ((launch d).ctx i).pid 1 = pbh d i := by
  have h := pbh_lt hd hi
  simp only [Launch.ctx, launch]
  show i / ((d.N_CTX + 127) / 128) % (d.Z * d.H) = pbh d i
  exact Nat.mod_eq_of_lt h

/-- Program `i`'s first query row `128·start_m`, below `N_CTX`. -/
theorem pm_row {d : Inst} (hd : d.ok) (i : Nat) : 128 * pm d i + 128 ≤ d.N_CTX := by
  have h1 := pm_lt hd i
  rw [g0_eq hd] at h1
  have := hd.n_mul
  omega

/-- The program of query tile `sm` of head `bh`. -/
theorem prog_of {d : Inst} (hd : d.ok) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < g0 d) :
    bh * g0 d + sm < (launch d).size ∧ pm d (bh * g0 d + sm) = sm ∧ pbh d (bh * g0 d + sm) = bh := by
  have hg := g0_pos hd
  refine ⟨?_, ?_, ?_⟩
  · rw [launch_size]
    calc bh * g0 d + sm < bh * g0 d + g0 d := by omega
      _ = (bh + 1) * g0 d := by ring
      _ ≤ (d.Z * d.H) * g0 d := Nat.mul_le_mul_right _ hbh
      _ = g0 d * (d.Z * d.H) := by ring
  · unfold pm; rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hsm]
  · unfold pbh; rw [Nat.add_comm, Nat.add_mul_div_right _ _ hg, Nat.div_eq_of_lt hsm, zero_add]

theorem headDim_pow {d : Inst} (hd : d.ok) : ∃ e, d.HEAD_DIM = 2 ^ e := by
  rcases hd.head_dim with h | h | h <;> rw [h]
  · exact ⟨6, rfl⟩
  · exact ⟨7, rfl⟩
  · exact ⟨8, rfl⟩

theorem headDim_le {d : Inst} (hd : d.ok) : d.HEAD_DIM ≤ 256 := by
  rcases hd.head_dim with h | h | h <;> omega

theorem headDim_pos {d : Inst} (hd : d.ok) : 64 ≤ d.HEAD_DIM := by
  rcases hd.head_dim with h | h | h <;> omega

/-- The rows of the view are fewer than `2^25` (`elems_lt`, `HEAD_DIM ≥ 64`). -/
theorem rows_lt {d : Inst} (hd : d.ok) : rows d * 64 < 2 ^ 31 := by
  have h1 := hd.elems_lt
  have h2 := headDim_pos hd
  calc rows d * 64 ≤ rows d * d.HEAD_DIM := Nat.mul_le_mul_left _ h2
    _ < 2 ^ 31 := h1

theorem rows_pos {d : Inst} (hd : d.ok) : 0 < rows d := by
  unfold rows; have h1 := hd.zh_pos; have h2 := hd.n_mul
  exact Nat.mul_pos (Nat.mul_pos h1.1 h1.2) h2.1

/-- **The launch is valid**: a grid `(cdiv(N_CTX, 128), Z·H, 1)` within CUDA's limits, fewer than
`2^31` programs, and block shapes of powers of two with at most `2^20` elements. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg : g0 d ≤ d.N_CTX := by rw [g0_eq hd]; omega
  have hrows := rows_lt hd
  have hzh := hd.zh_lt
  have hN := hd.n_le
  have hsize : (launch d).size < 2 ^ 31 := by
    rw [launch_size]
    calc g0 d * (d.Z * d.H) ≤ d.N_CTX * (d.Z * d.H) := Nat.mul_le_mul_right _ hg
      _ = rows d := by unfold rows; ring
      _ < 2 ^ 31 := by omega
  refine ⟨?_, ?_, ?_, hsize, rfl, rfl, ?_⟩
  · show g0 d < 2 ^ 31; omega
  · show d.Z * d.H < 2 ^ 16; exact hzh
  · show 1 < 2 ^ 16; norm_num
  · obtain ⟨e, he⟩ := headDim_pow hd
    have hD := headDim_le hd
    intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact ⟨by simp; exact ⟨7, rfl⟩, by norm_num⟩
    · exact ⟨by simp; exact ⟨6, rfl⟩, by norm_num⟩
    · exact ⟨by simp; exact ⟨⟨7, rfl⟩, ⟨0, rfl⟩⟩, by norm_num⟩
    · exact ⟨by simp; exact ⟨⟨0, rfl⟩, ⟨6, rfl⟩⟩, by norm_num⟩
    · exact ⟨by simp; exact ⟨⟨7, rfl⟩, ⟨6, rfl⟩⟩, by norm_num⟩
    · exact ⟨by simp; exact ⟨⟨e, he⟩, ⟨6, rfl⟩⟩, by simp; omega⟩
    · exact ⟨by simp; exact ⟨⟨6, rfl⟩, ⟨e, he⟩⟩, by simp; omega⟩
    · exact ⟨by simp; exact ⟨⟨7, rfl⟩, ⟨e, he⟩⟩, by simp; omega⟩

/-! ## `int32` arithmetic -/

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

theorem i32_le {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.le a b = decide (x ≤ y) := by
  rw [IntV.le_eq, ha, hb]

theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

theorem i64_nat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = n :=
  IntV.toInt_ofNat (by norm_num) (by push_cast; exact_mod_cast h)

/-! ## The descriptors -/

/-- The row-major `[Z·H·N_CTX, HEAD_DIM]` descriptor of buffer `b` (`q`, `k`, `o`) with block rows
`BM`. -/
def descOf (d : Inst) (b : Buf .f8e5) (BM : Nat) : TensorDesc .f8e5 :=
  ⟨b.ptr 0, [rows d, d.HEAD_DIM], [d.HEAD_DIM, 1], [BM, d.HEAD_DIM], .zero⟩

/-- The descriptor of `v` (lines 198, 530): shape `[HEAD_DIM, Z·H·N_CTX]`, strides `[N_CTX, 1]`,
block `[HEAD_DIM, 64]`. -/
def descV (d : Inst) : TensorDesc .f8e5 :=
  ⟨d.v.ptr 0, [d.HEAD_DIM, rows d], [d.N_CTX, 1], [d.HEAD_DIM, 64], .zero⟩

theorem inbAt2 {dt : DType} (base : Ptr dt) (s0 s1 l : Nat) (bl : Shape) (pad : Padding) (c : Nat → Int) :
    (⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc dt).inbAt c =
      decide ((0 ≤ c 0 ∧ c 0 < s0) ∧ (0 ≤ c 1 ∧ c 1 < s1)) := by
  unfold TensorDesc.inbAt TensorDesc.rank
  refine decide_eq_decide.mpr ⟨fun h => ⟨h 0 (by simp), h 1 (by simp)⟩, fun h k hk => ?_⟩
  simp only [List.length_cons, List.length_nil] at hk
  rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl
  · exact h.1
  · exact h.2

theorem elemAt2 {dt : DType} (base : Ptr dt) (s0 s1 l : Nat) (bl : Shape) (pad : Padding) (c : Nat → Int) :
    (⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc dt).elemAt c = c 0 * l + c 1 := by
  unfold TensorDesc.elemAt TensorDesc.rank
  simp [List.range_succ]

theorem coord2_0 (o0 o1 : Int) {m n : Nat} (i : Fin m) (j : Fin n) :
    TensorDesc.coord [o0, o1] (bs := [m, n]) (i, j, ()) 0 = o0 + i.val := by
  simp [TensorDesc.coord, Idx.coord]

theorem coord2_1 (o0 o1 : Int) {m n : Nat} (i : Fin m) (j : Fin n) :
    TensorDesc.coord [o0, o1] (bs := [m, n]) (i, j, ()) 1 = o1 + j.val := by
  simp [TensorDesc.coord, Idx.coord]

/-- Lane `(i, j)` of a block at rows `O0`, column `0` of `descOf d b BM`: inside when
`O0 + i < rows`, and then element `(O0 + i)·HEAD_DIM + j`. -/
theorem lane2 (d : Inst) (b : Buf .f8e5) (BM : Nat) {m n : Nat} {o0 : I32} {O0 : Nat}
    (h0 : o0.toInt = O0) (hn : n ≤ d.HEAD_DIM) (i : Fin m) (j : Fin n) :
    (descOf d b BM).inbAt (TensorDesc.coord ([o0, (IntV.ofInt 0 : I32)].map IntV.toInt) (bs := [m, n]) (i, j, ())) =
      decide (O0 + i < rows d) ∧
    ((descOf d b BM).elemAt (TensorDesc.coord ([o0, (IntV.ofInt 0 : I32)].map IntV.toInt) (bs := [m, n]) (i, j, ()))).toNat =
      (O0 + i) * d.HEAD_DIM + j := by
  unfold descOf
  simp only [List.map_cons, List.map_nil]
  rw [inbAt2, elemAt2, coord2_0, coord2_1, h0, i32_zero]
  constructor
  · refine decide_eq_decide.mpr ⟨fun h => by omega, fun h => ⟨⟨by omega, by omega⟩, by omega, by omega⟩⟩
  · rw [show ((O0 : Int) + (i : Nat)) * (d.HEAD_DIM : Int) + ((0 : Int) + (j : Nat)) =
      (((O0 + i) * d.HEAD_DIM + j : Nat) : Int) by push_cast; ring, Int.toNat_natCast]

/-- Lane `(e, jj)` of a `[HEAD_DIM, 64]` block at `[0, O]` of `descV d`: inside when
`O + jj < rows`, and then element `e·N_CTX + O + jj` of `v`. -/
theorem laneV (d : Inst) {o1 : I32} {O : Nat} (h1 : o1.toInt = O) (e : Fin d.HEAD_DIM) (jj : Fin 64) :
    (descV d).inbAt (TensorDesc.coord ([(IntV.ofInt 0 : I32), o1].map IntV.toInt) (bs := [d.HEAD_DIM, 64]) (e, jj, ())) =
      decide (O + jj < rows d) ∧
    ((descV d).elemAt (TensorDesc.coord ([(IntV.ofInt 0 : I32), o1].map IntV.toInt) (bs := [d.HEAD_DIM, 64]) (e, jj, ()))).toNat =
      e * d.N_CTX + (O + jj) := by
  unfold descV
  simp only [List.map_cons, List.map_nil]
  rw [inbAt2, elemAt2, coord2_0, coord2_1, h1, i32_zero]
  constructor
  · refine decide_eq_decide.mpr ⟨fun h => by omega, fun h => ⟨⟨by omega, by omega⟩, by omega, by omega⟩⟩
  · rw [show ((0 : Int) + (e : Nat)) * (d.N_CTX : Int) + ((O : Int) + (jj : Nat)) =
      ((e * d.N_CTX + (O + jj) : Nat) : Int) by push_cast; ring, Int.toNat_natCast]

/-- **An access at `[O0, 0]` through `descOf d b BM`** meets Triton's requirements. -/
theorem access_descOf {d : Inst} (hd : d.ok) {b : Buf .f8e5} (hb : b.base % 16 = 0) {BM : Nat}
    (o0 : I32) : (descOf d b BM).Access ([o0, (IntV.ofInt 0 : I32)].map IntV.toInt) [BM, d.HEAD_DIM] := by
  have hrows := rows_lt hd
  have hD := headDim_le hd
  have hD64 := headDim_pos hd
  have hpos := rows_pos hd
  have hb' : (b.ptr 0).addr % 16 = 0 := by simpa [Buf.ptr] using hb
  refine ⟨⟨by simp [descOf, TensorDesc.rank], by simp [descOf, TensorDesc.rank],
    by simp [descOf, TensorDesc.rank], by simp [descOf, TensorDesc.rank], fun k hk => ?_, hb', rfl,
    fun k hk => ?_, by simp [descOf], by simp [descOf]⟩,
    rfl, by simp [descOf, TensorDesc.rank], by simp [descOf, TensorDesc.rank], ?_⟩
  · simp only [descOf, TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl
    · simp only [descOf, List.getD_cons_zero]; exact ⟨hpos, by omega⟩
    · simp only [descOf, List.getD_cons_succ, List.getD_cons_zero]; constructor <;> omega
  · simp only [descOf, TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rw [show k = 0 by omega]
    simp only [descOf, List.getD_cons_zero, DType.bytes]
    rcases hd.head_dim with h | h | h <;> rw [h]
  · simp [i32_zero]

/-- **An access at `[0, O]` through `descV d`**, `O` a multiple of `16`, meets Triton's
requirements. -/
theorem access_descV {d : Inst} (hd : d.ok) (o1 : I32) (ho : o1.toInt % 16 = 0) :
    (descV d).Access ([(IntV.ofInt 0 : I32), o1].map IntV.toInt) [d.HEAD_DIM, 64] := by
  have hrows := rows_lt hd
  have hD := headDim_le hd
  have hD64 := headDim_pos hd
  have hpos := rows_pos hd
  have hb' : (d.v.ptr 0).addr % 16 = 0 := by simpa [Buf.ptr] using hd.align.2.2.1
  have hN := hd.n_mul
  refine ⟨⟨by simp [descV, TensorDesc.rank], by simp [descV, TensorDesc.rank],
    by simp [descV, TensorDesc.rank], by simp [descV, TensorDesc.rank], fun k hk => ?_, hb', rfl,
    fun k hk => ?_, by simp [descV], by simp [descV]⟩,
    rfl, by simp [descV, TensorDesc.rank], by simp [descV, TensorDesc.rank], ?_⟩
  · simp only [descV, TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl
    · simp only [descV, List.getD_cons_zero]; constructor <;> omega
    · simp only [descV, List.getD_cons_succ, List.getD_cons_zero]; exact ⟨hpos, by omega⟩
  · simp only [descV, TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rw [show k = 0 by omega]
    simp only [descV, List.getD_cons_zero, DType.bytes]
    omega
  · simp only [List.map_cons, List.map_nil, List.getLast?_cons_cons, List.getLast?_singleton,
      Option.getD_some, DType.bytes]
    omega

theorem zh_val {d : Inst} (hd : d.ok) : (IntV.ofNat d.Z * IntV.ofNat d.H : I32).toInt = ((d.Z * d.H : Nat) : Int) := by
  have h1 : d.Z ≤ d.Z * d.H := Nat.le_mul_of_pos_right _ hd.zh_pos.2
  have h2 : d.H ≤ d.Z * d.H := Nat.le_mul_of_pos_left _ hd.zh_pos.1
  have := hd.zh_lt
  exact IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega)) (by omega)

/-- `y_dim = Z·H·N_CTX` (line 194), no wrap. -/
theorem ydim_val {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.Z * IntV.ofNat d.H * IntV.ofNat d.N_CTX : I32).toInt = (rows d : Int) := by
  have hrows := rows_lt hd
  have hN := hd.n_le
  exact IntV.I32.mul_nat (zh_val hd) (i32_nat (by omega)) (by unfold rows at hrows; omega)

/-- A host descriptor and one the kernel builds (line 173) are the same descriptor (`q`, `k`, `o`). -/
theorem make_desc {d : Inst} (hd : d.ok) (b : Buf .f8e5) (BM : Nat) :
    (⟨b.ptr 0, ([IntV.ofNat d.Z * IntV.ofNat d.H * IntV.ofNat d.N_CTX, (IntV.ofNat d.HEAD_DIM : I32)] : List I32).map
        (fun x => x.toInt.toNat),
      ([(IntV.ofNat d.HEAD_DIM : I64), (IntV.ofNat 1 : I64)] : List I64).map (fun x => x.toInt.toNat),
      [BM, d.HEAD_DIM], .zero⟩ : TensorDesc .f8e5) = descOf d b BM := by
  have hD := headDim_le hd
  have hy := ydim_val hd
  unfold descOf
  simp only [List.map_cons, List.map_nil, hy, i32_nat (show d.HEAD_DIM < 2 ^ 31 by omega),
    i64_nat (show d.HEAD_DIM < 2 ^ 63 by omega), i64_nat (show 1 < 2 ^ 63 by norm_num), Int.toNat_natCast]

/-- The same for `v` (lines 198–199, 173). -/
theorem make_descV {d : Inst} (hd : d.ok) :
    (⟨d.v.ptr 0, ([(IntV.ofNat d.HEAD_DIM : I32), IntV.ofNat d.Z * IntV.ofNat d.H * IntV.ofNat d.N_CTX] : List I32).map
        (fun x => x.toInt.toNat),
      ([(IntV.cast (IntV.ofNat d.N_CTX : I32) : I64), (IntV.ofNat 1 : I64)] : List I64).map (fun x => x.toInt.toNat),
      [d.HEAD_DIM, 64], .zero⟩ : TensorDesc .f8e5) = descV d := by
  have hD := headDim_le hd
  have hy := ydim_val hd
  have hN := hd.n_le
  have hc : (IntV.cast (IntV.ofNat d.N_CTX : I32) : I64).toInt = d.N_CTX := by
    rw [IntV.toInt_cast _ (by norm_num), i32_nat (by omega)]
  unfold descV
  simp only [List.map_cons, List.map_nil, hy, hc, i32_nat (show d.HEAD_DIM < 2 ^ 31 by omega),
    i64_nat (show 1 < 2 ^ 63 by norm_num), Int.toNat_natCast]

/-- What `_maybe_make_tensor_desc` returns for `descArg d b BM`: `descOf d b BM`, either kind. -/
theorem safe_maybe_desc {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {d : Inst} (hd : d.ok) (b : Buf .f8e5) (BM : Nat) {k : TensorDesc .f8e5 → Prog α}
    (hk : Safe E R W Q m (k (descOf d b BM))) :
    Safe E R W Q m (maybe_make_tensor_desc (descArg d b BM)
      [IntV.ofNat d.Z * IntV.ofNat d.H * IntV.ofNat d.N_CTX, (IntV.ofNat d.HEAD_DIM : I32)]
      [(IntV.ofNat d.HEAD_DIM : I64), (IntV.ofNat 1 : I64)] [BM, d.HEAD_DIM] >>= k) := by
  unfold descArg
  split_ifs with hw
  · simp only [maybe_make_tensor_desc]
    have hrows := rows_lt hd
    have hD := headDim_le hd
    refine safe_make_tensor_descriptor ?_ ?_ ?_
    · intro x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · rw [ydim_val hd]; positivity
      · rw [i32_nat (by omega)]; omega
    · intro x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · rw [i64_nat (by omega)]; omega
      · rw [i64_nat (by norm_num)]; omega
    · rw [make_desc hd b BM]; exact hk
  · simp only [maybe_make_tensor_desc, Prog.pure_eq]
    exact hk

/-- What `_maybe_make_tensor_desc` returns for `descArgV d`: `descV d`, either kind. -/
theorem safe_maybe_descV {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {d : Inst} (hd : d.ok) {k : TensorDesc .f8e5 → Prog α}
    (hk : Safe E R W Q m (k (descV d))) :
    Safe E R W Q m (maybe_make_tensor_desc (descArgV d)
      [(IntV.ofNat d.HEAD_DIM : I32), IntV.ofNat d.Z * IntV.ofNat d.H * IntV.ofNat d.N_CTX]
      [(IntV.cast (IntV.ofNat d.N_CTX : I32) : I64), (IntV.ofNat 1 : I64)] [d.HEAD_DIM, 64] >>= k) := by
  unfold descArgV
  split_ifs with hw
  · simp only [maybe_make_tensor_desc]
    have hrows := rows_lt hd
    have hD := headDim_le hd
    have hN := hd.n_le
    refine safe_make_tensor_descriptor ?_ ?_ ?_
    · intro x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · rw [i32_nat (by omega)]; omega
      · rw [ydim_val hd]; positivity
    · intro x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · rw [IntV.toInt_cast _ (by norm_num), i32_nat (by omega)]; omega
      · rw [i64_nat (by norm_num)]; omega
    · rw [make_descV hd]; exact hk
  · simp only [maybe_make_tensor_desc, Prog.pure_eq]
    exact hk

/-! ## The constants -/

theorem zero32 : (FloatV.ofPat 0 : F32).isFinite = true ∧ (FloatV.ofPat 0 : F32).toRat = 0 :=
  DotLoop.val?_f32.mp DotLoop.zero_val

theorem zero8 : (FloatV.ofPat 0 : F8E5).isFinite = true ∧ (FloatV.ofPat 0 : F8E5).toRat = 0 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- Line 214: `0 - float("inf")` is `-∞`. -/
theorem m_init_eq {r : F32}
    (h : FloatV.NaNEq (FloatV.sub (FloatV.ofPat 0 : F32) (FloatV.litF32 (FloatV.ofPat 0x7FF0000000000000 : F64))) r) :
    r = FloatV.negInf := by
  have e : FloatV.sub (FloatV.ofPat 0 : F32) (FloatV.litF32 (FloatV.ofPat 0x7FF0000000000000 : F64)) =
      FloatV.negInf := by decide +kernel
  rw [e] at h
  exact h.eq negInf32_isNaN

/-- Line 215: `0 + 1.0` is `1.0`. -/
theorem l_init_eq {r : F32}
    (h : FloatV.NaNEq (FloatV.add (FloatV.ofPat 0 : F32) (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000 : F64))) r) :
    r.isFinite = true ∧ r.toReal = 1 := by
  have e : FloatV.add (FloatV.ofPat 0 : F32) (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000 : F64)) =
      FloatV.ofPat 0x3F800000 := by decide +kernel
  have h1 : (FloatV.ofPat 0x3F800000 : F32).isFinite = true := by decide +kernel
  have h2 : (FloatV.ofPat 0x3F800000 : F32).toRat = 1 := by decide +kernel
  rw [e] at h
  rw [h.eq (isNaN_of_fin h1)]
  exact ⟨h1, by rw [toReal_def, h2]; simp⟩

/-- Line 219's constant `1.44269504` in `float32`: `12102203/2^23`. -/
theorem k32_facts : (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F : F64) : F32).isFinite = true ∧
    (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F : F64) : F32).toRat = 12102203 / 8388608 := by
  have e : (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F : F64) : F32) = FloatV.ofPat 0x3FB8AA3B := by
    decide +kernel
  rw [e]
  exact ⟨by decide +kernel, by decide +kernel⟩

/-- Line 76's `0` in `float32`. -/
theorem w0_facts : (FloatV.litF32 (FloatV.ofPat 0 : F64) : F32).isFinite = true ∧
    (FloatV.litF32 (FloatV.ofPat 0 : F64) : F32).toRat = 0 := by
  have e : (FloatV.litF32 (FloatV.ofPat 0 : F64) : F32) = FloatV.ofPat 0 := by decide +kernel
  rw [e]; exact zero32

/-- Line 76's `-1.0e6` in `float32`: exact. -/
theorem w1_facts : (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000 : F64) : F32).isFinite = true ∧
    (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000 : F64) : F32).toRat = -1000000 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-! ## `qk_scale` (line 219) -/

theorem k32_ln2 : 1 - 1 / 2 ^ 26 ≤ (12102203 / 8388608 : ℝ) * Real.log 2 ∧
    (12102203 / 8388608 : ℝ) * Real.log 2 ≤ 1 := by
  have h1 := Real.log_two_gt_d9
  have h2 := Real.log_two_lt_d9
  constructor <;> nlinarith

/-- **`qk_scale`**: `fl(fl(sm_scale)·fl(1.44269504))`, finite, positive, at most `2^65` and
`3/2·sm_scale`, and `qk_scale·ln 2` within `2^-22·sm_scale` of `sm_scale`. -/
theorem qk_scale_facts {d : Inst} (hd : d.ok) {c : F32}
    (hc : FloatV.NaNEq (FloatV.mul (FloatV.hostF32 d.sm_scale)
      (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F : F64))) c) :
    c.isFinite = true ∧ 0 < c.toReal ∧ c.toReal ≤ 2 ^ 65 ∧ c.toReal ≤ 3 / 2 * (sigma d : ℝ) ∧
      |c.toReal * Real.log 2 - (sigma d : ℝ)| ≤ (sigma d : ℝ) / 2 ^ 22 := by
  obtain ⟨hsf, hs1, hs2⟩ := hd.scale
  have hσd : ((sigma d : ℚ) : ℝ) = d.sm_scale.toReal := rfl
  rw [hσd]
  have hσ1 : 1 / 2 ^ 64 ≤ d.sm_scale.toReal := by
    have := (Rat.cast_le (K := ℝ)).mpr hs1; push_cast at this; exact this
  have hσ2 : d.sm_scale.toReal ≤ 2 ^ 64 := by
    have := (Rat.cast_le (K := ℝ)).mpr hs2; push_cast at this; exact this
  have hσ0 : 0 < d.sm_scale.toReal := lt_of_lt_of_le (by positivity) hσ1
  -- `fl(sm_scale)`
  have hnan : d.sm_scale.isNaN = false := isNaN_of_fin hsf
  have hh : FloatV.hostF32 d.sm_scale = FloatV.cvt d.sm_scale := by
    unfold FloatV.hostF32; simp [hnan]
  have hmax : |d.sm_scale.toRat| ≤ Format.binary32.maxVal := by
    have : ((|d.sm_scale.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, abs_of_pos hσ0]; nlinarith
    exact_mod_cast this
  obtain ⟨hs32f, hs32e⟩ := FloatV.cvt_err (g := Format.binary32) Format.binary32_regular hsf hmax
  rw [← hh] at hs32f hs32e
  have he2 : |(FloatV.hostF32 d.sm_scale).toReal - d.sm_scale.toReal| ≤
      d.sm_scale.toReal / 2 ^ 24 + 1 / 2 ^ 150 := by
    rw [u32_eq, eta32_eq] at hs32e
    have := (Rat.cast_le (K := ℝ)).mpr hs32e
    push_cast at this
    rw [← toReal_def, ← toReal_def, abs_of_pos hσ0] at this
    norm_num at this ⊢; linarith
  -- the constant
  obtain ⟨hkf, hkv⟩ := k32_facts
  have hkr : (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F : F64) : F32).toReal = 12102203 / 8388608 := by
    rw [toReal_def, hkv]; push_cast; ring
  obtain ⟨hkl1, hkl2⟩ := k32_ln2
  have hl1 := Real.log_two_gt_d9
  have hl2 := Real.log_two_lt_d9
  generalize FloatV.hostF32 d.sm_scale = s32 at hs32f he2 hc
  generalize (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F : F64) : F32) = k32 at hkf hkr hc
  generalize d.sm_scale.toReal = σ at hσ1 hσ2 hσ0 he2 ⊢
  -- the product
  have hs32b : |s32.toReal| ≤ σ * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub s32.toReal σ
    rw [abs_of_pos hσ0] at this; linarith
  have hroom : |s32.toReal * k32.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul, hkr, abs_of_pos (by norm_num : (0 : ℝ) < 12102203 / 8388608)]
    nlinarith
  obtain ⟨hcf, hce⟩ := f32_mul hc hs32f hkf hroom
  refine ⟨hcf, ?_⟩
  rw [abs_mul, hkr, abs_of_pos (by norm_num : (0 : ℝ) < 12102203 / 8388608)] at hce
  generalize Real.log 2 = L at hl1 hl2 hkl1 hkl2 ⊢
  have hL0 : 0 < L := by linarith
  generalize c.toReal = cr at hce ⊢
  generalize s32.toReal = sr at hce hs32b he2
  have e : cr * L - σ = (cr - sr * (12102203 / 8388608)) * L + (sr - σ) * (12102203 / 8388608 * L) +
      σ * (12102203 / 8388608 * L - 1) := by ring
  have hb1 : |(cr - sr * (12102203 / 8388608)) * L| ≤ (|sr| * (12102203 / 8388608) / 2 ^ 24 + 1 / 2 ^ 150) * L := by
    rw [abs_mul, abs_of_pos hL0]; exact mul_le_mul_of_nonneg_right hce hL0.le
  have hb2 : |(sr - σ) * (12102203 / 8388608 * L)| ≤ (σ / 2 ^ 24 + 1 / 2 ^ 150) * 1 := by
    rw [abs_mul, abs_of_pos (mul_pos (by norm_num : (0 : ℝ) < 12102203 / 8388608) hL0)]
    exact mul_le_mul he2 hkl2 (mul_pos (by norm_num : (0 : ℝ) < 12102203 / 8388608) hL0).le (by positivity)
  have hb3 : |σ * (12102203 / 8388608 * L - 1)| ≤ σ * (1 / 2 ^ 26) := by
    rw [abs_mul, abs_of_pos hσ0]
    refine mul_le_mul_of_nonneg_left ?_ hσ0.le
    rw [abs_le]; constructor <;> linarith
  have hb1' : (|sr| * (12102203 / 8388608) / 2 ^ 24 + 1 / 2 ^ 150) * L ≤
      (σ * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150) / 2 ^ 24 + 1 / 2 ^ 150 := by
    have h1 : |sr| * (12102203 / 8388608) * L ≤ |sr| := by
      have := mul_le_mul_of_nonneg_left hkl2 (abs_nonneg sr); linarith
    have h2 : (1 : ℝ) / 2 ^ 150 * L ≤ 1 / 2 ^ 150 := by nlinarith
    have : (|sr| * (12102203 / 8388608) / 2 ^ 24 + 1 / 2 ^ 150) * L =
        |sr| * (12102203 / 8388608) * L / 2 ^ 24 + 1 / 2 ^ 150 * L := by ring
    rw [this]
    have := div_le_div_of_nonneg_right (h1.trans hs32b) (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
    linarith
  have hmain : |cr * L - σ| ≤ σ / 2 ^ 22 := by
    rw [e]
    have t1 := abs_add_le ((cr - sr * (12102203 / 8388608)) * L + (sr - σ) * (12102203 / 8388608 * L))
      (σ * (12102203 / 8388608 * L - 1))
    have t2 := abs_add_le ((cr - sr * (12102203 / 8388608)) * L) ((sr - σ) * (12102203 / 8388608 * L))
    have hσt : (1 : ℝ) / 2 ^ 148 ≤ σ / 2 ^ 84 := by
      rw [div_le_div_iff₀ (by norm_num) (by norm_num)]; nlinarith
    have hnum : (σ * (1 + 1 / 2 ^ 24) + 1 / 2 ^ 150) / 2 ^ 24 + 1 / 2 ^ 150 + (σ / 2 ^ 24 + 1 / 2 ^ 150) * 1 +
        σ * (1 / 2 ^ 26) ≤ σ / 2 ^ 22 := by
      norm_num at hσt ⊢; linarith
    linarith
  have hcl := abs_le.mp hmain
  have hc0 : 0 < cr := by
    by_contra h
    replace h := not_lt.mp h
    have : cr * L ≤ 0 := mul_nonpos_of_nonpos_of_nonneg h hL0.le
    norm_num at hcl; linarith
  have hc32 : cr ≤ 3 / 2 * σ := by
    by_contra h; replace h := not_le.mp h
    have : 3 / 2 * σ * L < cr * L := mul_lt_mul_of_pos_right h hL0
    nlinarith
  refine ⟨hc0, ?_, hc32, hmain⟩
  have : (3 : ℝ) / 2 * σ ≤ 2 ^ 65 := by nlinarith
  linarith

/-- `qk_scale` against the exact base-2 scale `w = sm_scale / ln 2`: `|c − w| ≤ w·2^-22`. -/
theorem qk_scale_w {σ c : ℝ} (h : |c * Real.log 2 - σ| ≤ σ / 2 ^ 22) :
    |c - σ / Real.log 2| ≤ σ / Real.log 2 / 2 ^ 22 := by
  have hl : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have e : c - σ / Real.log 2 = (c * Real.log 2 - σ) / Real.log 2 := by field_simp
  rw [e, abs_div, abs_of_pos hl, show σ / Real.log 2 / 2 ^ 22 = σ / 2 ^ 22 / Real.log 2 by ring]
  exact div_le_div_of_nonneg_right h hl.le

end Kernels.AttnFwdE5m2
