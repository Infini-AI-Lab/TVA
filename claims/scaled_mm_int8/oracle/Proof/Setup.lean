import Proof.Foot
import Proof.Value

/-!
# `scaled_mm_int8`: shared definitions

The lane specification `LaneQ` (the conclusion of `OutSpec` for one lane), the conversions
between the formats of the tail, and the rule for a read-only `for` loop.
-/

namespace Claims.ScaledMmInt8
open TVA

/-- `float32` to `bfloat16` conversion is the rounding `FloatV.cvt`. -/
theorem convRel_f32_bf16 {P : Platform} (x : F32) :
    ConvRel P .f32 .bf16 x (FloatV.cvt (g := .bfloat16) x) := by
  show FloatRel P .binary32 .bfloat16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩

/-- `float32` to `float16` conversion is the rounding `FloatV.cvt`. -/
theorem convRel_f32_f16 {P : Platform} (x : F32) :
    ConvRel P .f32 .f16 x (FloatV.cvt (g := .binary16) x) := by
  show FloatRel P .binary32 .binary16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, FloatV.NaNEq.refl _⟩

/-- `bfloat16` to `float32` conversion is the exact `FloatV.cvt`. -/
theorem convRel_bf16_f32 {P : Platform} (x : BF16) :
    ConvRel P .bf16 .f32 x (FloatV.cvt (g := .binary32) x) := by
  show FloatRel P .bfloat16 .binary32 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), FloatV.NaNEq.refl _⟩

/-- `float16` to `float32` conversion is the exact `FloatV.cvt`. -/
theorem convRel_f16_f32 {P : Platform} (x : F16) :
    ConvRel P .f16 .f32 x (FloatV.cvt (g := .binary32) x) := by
  show FloatRel P .binary16 .binary32 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨Or.inl rfl, Or.inr (Or.inr (Or.inl rfl)), FloatV.NaNEq.refl _⟩

/-- A `float32` converts to the output dtype (some result is allowed). -/
theorem convRel_f32_exists {P : Platform} {od : DType} (hod : od = .bf16 ∨ od = .f16) (x : F32) :
    ∃ r : od.Val, ConvRel P .f32 od x r := by
  rcases hod with h | h
  · rw [h]; exact ⟨_, convRel_f32_bf16 x⟩
  · rw [h]; exact ⟨_, convRel_f32_f16 x⟩

/-- An `int32`-to-`int64` conversion keeps the value. -/
theorem convRel_i32_i64 {P : Platform} {x : I32} {r : I64} (h : ConvRel P .i32 .i64 x r) :
    r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  subst h'
  refine IntV.toInt_ofInt (by norm_num) ?_
  have hx := IntV.sfits_toInt x
  simp only [IntV.SFits] at hx ⊢
  omega

/-- The inner conclusion of `OutSpec` for one lane, with the stored value still a variable. -/
def LaneQ (d : Inst) (init : Mem) (i j : Nat) (c : d.od.Val) : Prop :=
  ∀ (va vb : Nat → I8) (sa sb : F32) (β : ℚ),
    (∀ k, k < d.K → init.read .i8 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k)) →
    (∀ k, k < d.K → init.read .i8 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k)) →
    init.read .f32 (d.sa.ptr ↑(saIdx d i)).addr = some sa → sa.isFinite = true →
    init.read .f32 (d.sb.ptr ↑(sbIdx d j)).addr = some sb → sb.isFinite = true →
    BiasIs d init j β →
    Room32 sa.toRat sb.toRat (dotSum va vb d.K) β →
      ∃ v : F32, v.isFinite = true ∧
        |v.toRat - (sa.toRat * sb.toRat * dotSum va vb d.K + β)| ≤
          tol sa.toRat sb.toRat (dotSum va vb d.K) β ∧
        OutIs d.od v c ∧
        (|sa.toRat * sb.toRat * dotSum va vb d.K + β| + tol sa.toRat sb.toRat (dotSum va vb d.K) β ≤
            outMax d.od →
          ∃ q : ℚ, d.od.val? c = some q ∧
            |q - (sa.toRat * sb.toRat * dotSum va vb d.K + β)| ≤
              outTol d.od sa.toRat sb.toRat (dotSum va vb d.K) β)

/-- The program of `scaled_mm_kernel` on the parameters of `d` (statement only, for the walk). -/
abbrev St (d : Inst) : Type :=
  Tensor [d.BK] I64 × Tensor [d.BM, d.BK] (Ptr .i8) × Tensor [d.BK, d.BN] (Ptr .i8) ×
    Tensor [d.BM, d.BN] (IntV true 32)

/-- The `int8` value stored at `addr` in memory `m` (as an integer). -/
def rI8 (m : Mem) (addr : Int) : ℤ := ((m.read .i8 addr).map IntV.toInt).getD 0

/-- Row `i`, column `k` of matrix `A` in memory `m`. -/
def rA (m : Mem) (d : Inst) (i k : Nat) : ℤ :=
  rI8 m (d.a.ptr ((i * d.stride_am + k * d.stride_ak : Nat) : ℤ)).addr

/-- Row `k`, column `j` of matrix `B` in memory `m`. -/
def rB (m : Mem) (d : Inst) (k j : Nat) : ℤ :=
  rI8 m (d.b.ptr ((k * d.stride_bk + j * d.stride_bn : Nat) : ℤ)).addr

/-- The loop invariant: the induction offset, the two pointer tiles and the accumulator valued as
the exact dot of the columns below `t * BK` (on the lanes the output covers). -/
def Inv (d : Inst) (init : Mem) (p t : Nat) (st : St d) : Prop :=
  (∀ kk : Fin d.BK, (st.1.get1 kk).toInt = ((t * d.BK + kk.1 : Nat) : ℤ)) ∧
  (∀ (i : Fin d.BM) (kk : Fin d.BK),
    st.2.1.get2 i kk =
      d.a.ptr (((pm d p * d.BM + i.1) * d.stride_am + (t * d.BK + kk.1) * d.stride_ak : Nat) : ℤ)) ∧
  (∀ (kk : Fin d.BK) (j : Fin d.BN),
    st.2.2.1.get2 kk j =
      d.b.ptr (((t * d.BK + kk.1) * d.stride_bk + (pn d p * d.BN + j.1) * d.stride_bn : Nat) : ℤ)) ∧
  (∀ (i : Fin d.BM) (j : Fin d.BN), pm d p * d.BM + i.1 < d.M → pn d p * d.BN + j.1 < d.N →
    (st.2.2.2.get2 i j).toInt =
      ∑ k ∈ Finset.range (t * d.BK), rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1))

/-- Two pointers with the same address are equal. -/
theorem Ptr.eq_of_addr {d : DType} {p q : Ptr d} (h : p.addr = q.addr) : p = q := by
  cases p; cases q; simp_all

/-- The accumulator at a covered lane, as a sum over the loop trips. -/
theorem Inv.acc {d : Inst} {init : Mem} {p t : Nat} {st : St d} (h : Inv d init p t st)
    (i : Fin d.BM) (j : Fin d.BN) (hi : pm d p * d.BM + i.1 < d.M)
    (hj : pn d p * d.BN + j.1 < d.N) :
    (st.2.2.2.get2 i j).toInt =
      ∑ k ∈ Finset.range (t * d.BK), rA init d (pm d p * d.BM + i.1) k * rB init d k (pn d p * d.BN + j.1) :=
  h.2.2.2 i j hi hj

/-- `safe_store_buf` with the trailing bind absorbed, so it applies to a bare `tl.store`. -/
theorem safe_store_buf_ret {E : Env} {R W : Set Int} {Q : Unit → Mem → Prop} {d : DType}
    {s : Shape} {m : Mem} (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {vals : Tensor s d.Val} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j)
    (hv : b.Valid m) (hW : b.elems (Buf.img g en) ⊆ W)
    (hk : ∀ m', (∀ i, en i = true → m'.read d (b.ptr (g i : Int)).addr = some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img g en)) Q m' (Prog.ret ())) :
    Safe E R W Q m (tl.store ptrs vals mask) := by
  change Safe E R W Q m (tl.store ptrs vals mask >>= Prog.ret)
  exact safe_store_buf (b := b) (g := g) (en := en) (k := Prog.ret) hA hin hinj hv hW hk

/-- A `for` loop whose body only reads memory, with invariant `I`, is safe when the range is accepted and `I` is kept. -/
theorem safe_forRange_ro {E : Env} {R Wk : Set Int} {α : Type} {σ : Type} {sg : Bool} {w : Nat}
    {lo hi step : IntV sg w} {init : σ} {body : IntV sg w → σ → Prog σ} {k : σ → Prog α}
    {Q : α → Mem → Prop} {m : Mem}
    (hrange : tl.RangeOk lo hi step) (I : Nat → σ → Mem → Prop) (h0 : I 0 init m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R ∅ (fun st' m₂ => I (t + 1) st' m₂) m₁ (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ → Safe E R Wk Q m₁ (k st)) :
    Safe E R Wk Q m (tl.forRange lo hi step init body >>= k) := by
  have h := safe_forRange (E := E) (R := R) (Q := Q) (m := m) (Wt := fun _ => ∅) hrange I
    (by intro t u _ _; simp) (by simp [iterW_empty]) h0 hstep hk
  simpa [iterW_empty] using h

end Claims.ScaledMmInt8
