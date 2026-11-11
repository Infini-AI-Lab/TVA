import Claim
import TVA.Proof
import Proof.Lane

/-!
# Peeling the kernel's scalar computations

The model's body is a chain of scalar `int32`/`int64` computations (`program_id`, `wide`,
`tl.idiv`, `tl.irem`, `batchOff`) whose values are known from the instance's bounds. Each lemma
here peels one such step of a `Safe` goal, given the concrete value it computes.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 800000

/-- A rank-0 tensor is the scalar of its item. -/
theorem tensor_scalar_item {α : Type} (r : Tensor [] α) : r = Tensor.scalar r.item := by
  ext j; rfl

/-- The one result of `tl.convert .i32 .i64` of a scalar is the sign extension of its value. -/
theorem conv_scalar_iff {E : Env} (n : I32) (i : Idx []) (r : I64) :
    ConvRel E.P .i32 .i64 ((Tensor.scalar n).get i) r ↔
      r = (Tensor.scalar (IntV.ofInt n.toInt) : Tensor [] I64).get i := by
  rw [ConvRel.iff_cast_i32_i64]
  exact Iff.of_eq (congrArg (r = ·) (IntV.cast_eq_ofInt _))

/-- Peel `tl.convert .i32 .i64` of a scalar. -/
theorem safe_conv_i32_i64_scalar {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (n : I32) {k : Tensor [] I64 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt n.toInt)))) :
    Safe E R W Q m (tl.convert .i32 .i64 (Tensor.scalar n) >>= k) :=
  (safe_convert_iff (conv_scalar_iff n)).2 hk

/-- Peel `wide` of a natural below `2^31`. -/
theorem safe_wide_nat {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (n : Nat) (hn : n < 2 ^ 31) {k : I64 → Prog α}
    (hk : Safe E R W Q m (k (IntV.ofInt (n : ℤ)))) :
    Safe E R W Q m (wide (IntV.ofNat n : I32) >>= k) := by
  simp only [wide]
  refine (safe_convert_iff (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat n : I32)) (y := Tensor.scalar (IntV.ofInt (IntV.ofNat n : I32).toInt)) (conv_scalar_iff (E := E) (IntV.ofNat n : I32))).2 ?_
  simp only [Tensor.item, Tensor.scalar]
  rw [IntV.toInt_ofNat hn]
  exact hk

/-- `(ofNat 0) * x = ofNat 0` on a signed integer. -/
theorem IntV.ofNat_zero_mul {w : Nat} (x : IntV true w) :
    (IntV.ofNat 0 : IntV true w) * x = IntV.ofNat 0 := by
  cases x with
  | mk b =>
    show (⟨(0#w) * b⟩ : IntV true w) = ⟨0#w⟩
    rw [BitVec.zero_mul]

/-- `tl.idiv` of two scalars with known values. -/
theorem safe_idiv_val {w : Nat} {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {x y : IntV true w} (a b : Nat) (hw : 0 < w) (hx : x.toInt = (a : ℤ)) (hy : y.toInt = (b : ℤ))
    (hb : 0 < b) (hq : ((a / b : Nat) : ℤ) < 2 ^ (w - 1))
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt ((a / b : Nat) : ℤ))))) :
    Safe E R W Q m (tl.idiv (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  have ha0 : 0 ≤ x.toInt := by rw [hx]; exact Int.natCast_nonneg a
  have hb0 : 0 < y.toInt := by rw [hy]; exact_mod_cast hb
  refine safe_idiv1 ha0 hb0 ?_
  intro r hr
  have h1 : r.item = IntV.ofInt ((a / b : Nat) : ℤ) := by
    apply IntV.ext_toInt
    rw [hr, hx, hy, IntV.toInt_ofInt hw (IntV.sfits_of_nonneg (Int.natCast_nonneg _) hq), Int.natCast_ediv]
  rw [tensor_scalar_item r, h1]; exact hk

/-- `tl.irem` of two scalars with known values. -/
theorem safe_irem_val {w : Nat} {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {x y : IntV true w} (a b : Nat) (hw : 0 < w) (hx : x.toInt = (a : ℤ)) (hy : y.toInt = (b : ℤ))
    (hb : 0 < b) (hq : ((a % b : Nat) : ℤ) < 2 ^ (w - 1))
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt ((a % b : Nat) : ℤ))))) :
    Safe E R W Q m (tl.irem (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  have ha0 : 0 ≤ x.toInt := by rw [hx]; exact Int.natCast_nonneg a
  have hb0 : 0 < y.toInt := by rw [hy]; exact_mod_cast hb
  refine safe_irem1 ha0 hb0 ?_
  intro r hr
  have h1 : r.item = IntV.ofInt ((a % b : Nat) : ℤ) := by
    apply IntV.ext_toInt
    rw [hr, hx, hy, IntV.toInt_ofInt hw (IntV.sfits_of_nonneg (Int.natCast_nonneg _) hq), Int.natCast_emod]
  rw [tensor_scalar_item r, h1]; exact hk

/-- A natural below `2^31` is an `int64` constant of its own value. -/
theorem I64_toInt_ofNat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofInt (n : ℤ) : I64).toInt = n :=
  IntV.I64.toInt_ofInt ⟨by omega, by omega⟩

/-- `ofInt 0` times anything is `ofInt 0`. -/
theorem IntV.ofInt_zero_mul {w : Nat} (x : IntV true w) :
    (IntV.ofInt (0 : ℤ) : IntV true w) * x = IntV.ofInt 0 := by
  cases x with
  | mk b =>
    show (⟨BitVec.ofInt w (0 : ℤ) * b⟩ : IntV true w) = ⟨BitVec.ofInt w 0⟩
    rw [show BitVec.ofInt w (0 : ℤ) = 0#w from rfl, BitVec.zero_mul]

/-- `batchOff` of `pid_b = 0`. -/
theorem safe_batchOff_zero {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (s : Nat) {k : I64 → Prog α}
    (hk : Safe E R W Q m (k (IntV.ofInt 0))) :
    Safe E R W Q m (batchOff (IntV.ofNat 0 : I32) (intArg s) >>= k) := by
  unfold intArg
  by_cases h : s < 2 ^ 31
  · simp only [h, ite_true, batchOff]
    rw [IntV.ofNat_zero_mul]
    exact safe_wide_nat (n := 0) (by norm_num) hk
  · simp only [h, ite_false, batchOff, wide, Prog.bind_eq, Prog.pure_eq, Prog.bind_assoc,
      Prog.ret_bind]
    refine (safe_convert_iff (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat 0 : I32)) (y := Tensor.scalar (IntV.ofInt (IntV.ofNat 0 : I32).toInt)) (conv_scalar_iff (E := E) (IntV.ofNat 0 : I32))).2 ?_
    rw [IntV.toInt_ofNat (by norm_num)]
    show Safe E R W Q m (k ((IntV.ofInt ((0 : Nat) : ℤ) : I64) * (IntV.ofNat s : I64)))
    rw [Nat.cast_zero, IntV.ofInt_zero_mul]
    exact hk

/-- `d.G < 2^31`, from `gz d < 2^16`. -/
theorem G_lt_31 {d : Inst} (hd : d.ok) : d.G < 2 ^ 31 := by
  have h := hd.gz_lt
  unfold gz at h
  have h1 : 0 < nch d := nch_pos hd
  have h2 : d.G ≤ nch d * d.G := by
    calc d.G = 1 * d.G := (Nat.one_mul _).symm
      _ ≤ nch d * d.G := Nat.mul_le_mul_right _ (by omega)
  omega

/-- A program index is below `2^31`. -/
theorem prog_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : (i : Nat) < 2 ^ 31 := by
  have h1 : (i : Nat) < gx d * 1 * gz d := i.isLt
  have h2 : gx d * 1 * gz d = gx d * gz d := by rw [Nat.mul_one]
  have h3 := hd.grid_lt
  omega

/-! ## Zero-pattern facts -/

/-- The `float32` `+0` is finite. -/
theorem f32_zero_isFinite : (FloatV.ofPat 0 : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (Nat.two_pow_pos 32)]
  exact Format.isFinite_zero Format.binary32_regular

/-- The `bfloat16` `+0` is finite. -/
theorem bf16_zero_isFinite : (FloatV.ofPat 0 : BF16).isFinite = true := by
  rw [FloatV.isFinite_ofPat (Nat.two_pow_pos 16)]
  exact Format.isFinite_zero Format.bfloat16_regular

/-- The `float32` `+0` is the rational `0`. -/
theorem f32_zero_toRat : (FloatV.ofPat 0 : F32).toRat = 0 :=
  FloatV.toRat_of_isZero (x := (FloatV.ofPat 0 : F32)) Format.isZero_zero

/-- The `bfloat16` `+0` is the rational `0`. -/
theorem bf16_zero_toRat : (FloatV.ofPat 0 : BF16).toRat = 0 :=
  FloatV.toRat_of_isZero (x := (FloatV.ofPat 0 : BF16)) Format.isZero_zero

/-- The `float32` `+0` has the value `0`. -/
theorem f32_zero_val : DType.val? .f32 (FloatV.ofPat 0) = some 0 := by
  show (if (FloatV.ofPat 0 : F32).isFinite = true then some (FloatV.ofPat 0).toRat else none) = some 0
  rw [f32_zero_isFinite, f32_zero_toRat]
  rfl

/-- The `bfloat16` `+0` has the value `0`. -/
theorem bf16_zero_val : DType.val? .bf16 (FloatV.ofPat 0) = some 0 := by
  show (if (FloatV.ofPat 0 : BF16).isFinite = true then some (FloatV.ofPat 0).toRat else none) = some 0
  rw [bf16_zero_isFinite, bf16_zero_toRat]
  rfl

/-- The `float32` with bit pattern `0` is `+0`. -/
theorem f32_ofBits_zero : (DType.ofBits .f32 0 : F32) = FloatV.ofPat 0 := rfl

/-- `tl.zeros [BM,BN] .f32` is the `+0` matrix. -/
theorem zeros_f32_get2 (mll : Fin BM) (nn : Fin BN) :
    (tl.zeros [BM, BN] .f32).get2 mll nn = (FloatV.ofPat 0 : F32) := rfl

/-- The rounding of `+0` from `float32` into `bfloat16` is `+0`. -/
theorem bf16_cvt_zero_fmt : Format.binary32.cvt Format.bfloat16 .rn false 0 = 0 := by
  rw [Format.cvt_of_finite (g := .bfloat16) (rnd := .rn) (sat := false) (hx := by rfl)]
  have h0 : Format.binary32.toRat 0 = 0 := Format.toRat_of_isZero Format.isZero_zero
  rw [h0, Format.roundRat_zero]
  show (if Format.binary32.sign 0 then Format.bfloat16.signBit else 0) +
    Format.bfloat16.mag 0 = 0
  rfl

/-- Converting the `float32` `+0` to `bfloat16` gives `+0`. -/
theorem bf16_cvt_zero : FloatV.cvt (g := .bfloat16) (FloatV.ofPat 0 : F32) = (FloatV.ofPat 0 : BF16) := by
  show (FloatV.ofPat (Format.binary32.cvt Format.bfloat16 .rn false 0) : BF16) = FloatV.ofPat 0
  rw [bf16_cvt_zero_fmt]

/-- The `bfloat16` `+0` is not a NaN. -/
theorem bf16_zero_notNaN : (FloatV.ofPat 0 : BF16).isNaN = false := rfl

/-- The conversion of `+0` from `float32` into `bfloat16` is not a NaN. -/
theorem bf16_cvt_zero_notNaN :
    (FloatV.cvt (g := .bfloat16) (FloatV.ofPat 0 : F32)).isNaN = false := by
  show (FloatV.ofPat (Format.binary32.cvt Format.bfloat16 .rn false 0) : BF16).isNaN = false
  rw [bf16_cvt_zero_fmt]
  exact bf16_zero_notNaN

/-- `tl.convert .f32 .bf16` sends `+0` to `+0`. -/
theorem conv_bf16_zero {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 (FloatV.ofPat 0) r) : r = FloatV.ofPat 0 := by
  have hF : FloatRel P .binary32 .bfloat16 (FloatV.ofPat 0) r := h
  unfold FloatRel at hF
  rw [ite_eq_right (by decide)] at hF
  rcases hF with ⟨-, -, h1⟩ | ⟨-, h2, -⟩ | ⟨h3, -⟩
  · exact (h1.eq bf16_cvt_zero_notNaN).trans bf16_cvt_zero
  · exact absurd h2 (by
      intro hh; rcases hh with hh | hh <;> exact absurd hh (by decide))
  · exact absurd h3 (by
      intro hh; rcases hh with hh | hh <;> exact absurd hh (by decide))

/-! ## A loop rule with no writes in the body -/

/-- The rule for `tl.forRange` with a loop-carried state and a body that writes nothing: if an invariant holds at each turn and the body keeps it, it holds at the end. -/
theorem safe_forRange_ne {σ : Type} {α : Type} {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init : σ}
    {body : IntV sg w → σ → Prog σ} {k : σ → Prog α} {E : Env} {R : Set Int}
    {Q : α → Mem → Prop} {m : Mem}
    (hrange : tl.RangeOk lo hi step)
    (I : Nat → σ → Mem → Prop) {Wk : Set Int}
    (h0 : I 0 init m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R ∅ (fun st' m₂ => I (t + 1) st' m₂) m₁
        (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ → Safe E R Wk Q m₁ (k st)) :
    Safe E R Wk Q m (tl.forRange lo hi step init body >>= k) := by
  have hWt : ∀ t u, t < u → u < tl.tripCount lo.toInt hi.toInt step.toInt →
      Disjoint ((fun _ => (∅ : Set Int)) t) ((fun _ => (∅ : Set Int)) u) := by
    intro t u _ _
    simp
  have h := safe_forRange (E := E) (R := R) (Q := Q) (m := m) hrange I (fun _ => ∅) (Wk := Wk)
    hWt (by rw [iterW_empty]; exact Set.empty_disjoint Wk) h0 hstep hk
  rw [iterW_empty, Set.empty_union] at h
  exact h

/-! ## The loop invariant's ingredients -/

/-- The step's relative error bound. -/
def rho : ℚ := DotLoop.stepRel Device.relDot16 32

/-- The step's absolute error bound. -/
def alpha : ℚ := DotLoop.stepAbs Device.absDot16 32

/-- `ρ = stepRel relDot16 32` is nonnegative. -/
theorem rho_nonneg' : 0 ≤ rho := by
  unfold rho; exact rho_nonneg

/-- `α = stepAbs absDot16 32` is nonnegative. -/
theorem alpha_nonneg : 0 ≤ alpha := by
  unfold alpha DotLoop.stepAbs Device.absDot16
  norm_num [Format.binary32_u, Format.binary32_eta]

/-- The state of the loop: the two pointer tiles and the accumulator. -/
abbrev StT : Type :=
  Tensor [BM, BK] (Ptr .bf16) × Tensor [BK, BN] (Ptr .bf16) × Tensor [BM, BN] F32

/-! ## The masks of the loads and the store -/

/-- The sign-extended row offsets. -/
def conv64 {n : Nat} (x : Tensor [n] I32) : Tensor [n] I64 := x.map (fun v => IntV.ofInt v.toInt)

/-- `chunk_size_limit`, as a natural. -/
def chunkNat (d : Inst) (i : Fin (launch d).size) : Nat :=
  min d.C (d.T - cOf d i * d.C)

/-- The `k` bound of the loop step: `K - k·BLOCK_SIZE_K`. -/
def kMask (d : Inst) (t : Nat) : I32 :=
  IntV.ofNat d.K - IntV.ofInt (t : ℤ) * (IntV.ofNat BK : I32)

/-- The mask of the `a` load. -/
def maskA (d : Inst) (i : Fin (launch d).size) (t : Nat) : Tensor [BM, BK] Bool :=
  (((conv64 (offsM d i)).col BK).ltS (chunkLimit d i)).and
    (((tl.arange 0 BK).row BM).ltS (kMask d t))

/-- The mask of the `b` load. -/
def maskB (d : Inst) (i : Fin (launch d).size) (t : Nat) : Tensor [BK, BN] Bool :=
  ((((tl.arange 0 BK).col BN).ltS (kMask d t)).and
    (((conv64 (offsN d i)).row BK).ltS (chunkLimit d i)))

/-- The mask of the `seq_idx` load of the `m` rows. -/
def maskSeqM (d : Inst) (i : Fin (launch d).size) : Tensor [BM] Bool :=
  (conv64 (offsM d i)).ltS (chunkLimit d i)

/-- The mask of the `seq_idx` load of the `n` columns. -/
def maskSeqN (d : Inst) (i : Fin (launch d).size) : Tensor [BN] Bool :=
  (conv64 (offsN d i)).ltS (chunkLimit d i)

/-- The mask of the `out` store. -/
def maskStore (d : Inst) (i : Fin (launch d).size) : Tensor [BM, BN] Bool :=
  ((((offsM d i).col BN).ltS (IntV.ofNat d.C : I32)).and
    (((offsN d i).row BM).ltS (IntV.ofNat d.C : I32)))

/-- A lane of the `a` tile is enabled. -/
def enA (d : Inst) (i : Fin (launch d).size) (t : Nat) (mll kk : Nat) : Bool :=
  decide (mOf d i * BM + mll < chunkNat d i) && decide (t * BK + kk < d.K)

/-- A lane of the `b` tile is enabled. -/
def enB (d : Inst) (i : Fin (launch d).size) (t : Nat) (kk nn : Nat) : Bool :=
  decide (t * BK + kk < d.K) && decide (nOf d i * BN + nn < chunkNat d i)

/-- A row of the `seq_idx` load is enabled. -/
def enM (d : Inst) (i : Fin (launch d).size) (mll : Nat) : Bool :=
  decide (mOf d i * BM + mll < chunkNat d i)

/-- A column of the `seq_idx` load is enabled. -/
def enN (d : Inst) (i : Fin (launch d).size) (nn : Nat) : Bool :=
  decide (nOf d i * BN + nn < chunkNat d i)

/-- A lane of the `out` store is enabled. -/
def enS (d : Inst) (i : Fin (launch d).size) (mll nn : Nat) : Bool :=
  decide (mOf d i * BM + mll < d.C) && decide (nOf d i * BN + nn < d.C)

end Claims.SsdBmm



