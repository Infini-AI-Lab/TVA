import TVA

/-! lr-19b, library review round 19: the batch's constants and lemmas used as a kernel proof
would use them (`lake env lean` on this file, from `lean/`). -/

open TVA TVA.Math.Float

/-! ## L98: the NaN constants, evaluated by the kernel (`decide +kernel`) -/

-- `float('nan')`, `-float('nan')`, a signalling NaN, a payload NaN
example : (FloatV.litF32 (FloatV.ofPat 0x7ff8000000000000 : F64)).pat = 0x7fc00000 := by decide +kernel
example : (FloatV.litF32 (FloatV.ofPat 0xfff8000000000000 : F64)).pat = 0xffc00000 := by decide +kernel
example : (FloatV.litF16 (FloatV.ofPat 0x7ff8000000000000 : F64)).pat = 0x7e00 := by decide +kernel
example : (FloatV.litF16 (FloatV.ofPat 0xfff8000000000000 : F64)).pat = 0xfe00 := by decide +kernel
example : (FloatV.litBF16 (FloatV.ofPat 0x7ff8000000000000 : F64)).pat = 0x7fc0 := by decide +kernel
example : (FloatV.litBF16 (FloatV.ofPat 0xfff8000000000000 : F64)).pat = 0xffc0 := by decide +kernel
example : (FloatV.litF64 (FloatV.ofPat 0xfff8000000000000 : F64)).pat = 0xfff8000000000000 := by decide +kernel
example : (FloatV.litF32 (FloatV.ofPat 0x7ff0000000000001 : F64)).pat = 0x7fc00000 := by decide +kernel
example : (FloatV.litF32 (FloatV.ofPat 0xfff0000020000000 : F64)).pat = 0xffc00001 := by decide +kernel
example : (FloatV.litF16 (FloatV.ofPat 0x7ff4000000000000 : F64)).pat = 0x7f00 := by decide +kernel
example : (FloatV.hostF32 (FloatV.ofPat 0xfff80000a0000000 : F64)).pat = 0xffc00005 := by decide +kernel
-- through `Operand.constIn?`, as SEMANTICS "Constants" tells a prover to evaluate it
example : ((Operand.float (FloatV.ofPat 0xfff8000000000000)).constIn? .f32).map FloatV.pat =
    some 0xffc00000 := by decide +kernel
example : ((Operand.float (FloatV.ofPat 0x7ff8000000000000)).constIn? .bf16).map FloatV.pat =
    some 0x7fc0 := by decide +kernel
-- numbers are unchanged by the round: 0.1 in float32, and the bfloat16 text cut
example : (FloatV.litF32 (FloatV.ofPat 0x3fb999999999999a : F64)).pat = 0x3dcccccd := by decide +kernel
example : (FloatV.litBF16 (FloatV.ofPat 0x3ee9e3fe580f5494 : F64)).pat = 0x3749 := by decide +kernel
-- the constants are NaNs where the device gives NaNs, and not the canonical one
example : (FloatV.litF32 (FloatV.ofPat 0x7ff8000000000000 : F64)).isNaN = true := by decide +kernel
example : FloatV.litF32 (FloatV.ofPat 0x7ff8000000000000 : F64) ≠ FloatV.nan := by decide +kernel

/-! ## L99: the new theorems, applied -/

-- `toRat` of `+∞` in `float32` is `2^128`, beyond `maxVal`
example : (FloatV.ofPat 0x7f800000 : F32).toRat = 2 ^ 128 := by
  have h := Format.toRat_of_isInf (f := Format.binary32) Format.binary32_regular (n := 0x7f800000) (by decide)
  rw [show (if Format.binary32.sign 0x7f800000 then -(2 : ℚ) ^ (Format.binary32.bias + 1)
      else (2 : ℚ) ^ (Format.binary32.bias + 1)) = 2 ^ 128 by
    rw [ite_eq_right (by decide)]; unfold Format.bias Format.binary32; norm_num] at h
  unfold FloatV.toRat; rw [FloatV.pat_ofPat (by decide)]; exact h

example : Format.binary32.maxVal < 2 ^ 128 := by
  have h := Format.maxVal_lt_inf (f := Format.binary32) Format.binary32_regular rfl
  have hb : Format.binary32.bias + 1 = 128 := by unfold Format.bias Format.binary32; norm_num
  rw [hb] at h; exact_mod_cast h

-- `tl.sqrt_rn` on `float32`: the bound with the format's constants
example (x : F32) (hx : x.isFinite = true) (h0 : 0 ≤ x.toRat) :
    |(FloatV.sqrtRn x).toReal - Real.sqrt x.toReal| ≤
      Real.sqrt x.toReal * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
  obtain ⟨-, -, h⟩ := FloatV.sqrtRn_err Format.binary32_regular hx h0
  rw [Format.binary32_u, Format.binary32_eta] at h
  push_cast at h
  exact h

-- toward zero, saturation changes nothing on a finite pattern or a NaN
example : Format.cvt Format.binary32 Format.e5m2 .rz true 0x7fc00000 =
    Format.cvt Format.binary32 Format.e5m2 .rz false 0x7fc00000 :=
  Format.cvt_rz_sat _ (by decide)
-- and on `+∞` it does change the result (so "only `±∞`" is exact)
example : Format.cvt Format.binary32 Format.e5m2 .rz true 0x7f800000 ≠
    Format.cvt Format.binary32 Format.e5m2 .rz false 0x7f800000 := by decide +kernel

/-! ## L100: the integer casts -/

-- index arithmetic widened from `int32` to `int64`
example (a : I32) : (a.cast : I64).toInt = a.toInt := IntV.toInt_cast a (by decide)
example (a : U32) : (a.cast : U64).toInt = a.toInt := IntV.toInt_cast_unsigned a (by decide)

/-! ## L100: `tl.atomic_add` through `safe_atomic_buf` (unfold, `Safe.env`, then the rule) -/

example {d : DType} {s : Shape} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (b : Buf d) {ptrs : Tensor s (Ptr d)} {vals : Tensor s d.Val} {mask : Option (Tensor s Bool)}
    {sem : MemSem} {sc : Scope} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j)
    (htot : ∀ i o, ∃ n, tl.addUpd E d (vals.get i) o n) (hv : b.Valid m)
    (hW : b.elems (Buf.img g en) ⊆ W) (hRd : Disjoint (b.elems (Buf.img g en)) R)
    {k : Tensor s d.Val → Prog α}
    (hk : ∀ (old : Tensor s d.Val) (m' : Mem),
      (∀ i, en i = true → ∃ n, m.read d (b.ptr (g i : Int)).addr = some (old.get i) ∧
        tl.addUpd E d (vals.get i) (old.get i) n ∧ m'.read d (b.ptr (g i : Int)).addr = some n) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img g en)) Q m' (k old)) :
    Safe E R W Q m (tl.atomic_add ptrs vals mask sem sc >>= k) := by
  simp only [tl.atomic_add, Prog.bind_eq, Prog.bind]
  refine Safe.env ?_
  have := safe_atomic_buf (upd := fun i => tl.addUpd E d (vals.get i)) (sem := sem) (sc := sc)
    b hA hin hinj htot hv hW hRd hk
  simpa only [Prog.bind_eq] using this

/-! ## L100: `tl.atomic_add` on the shared region (unfold, `SafeA.env`, then `SafeA.atomic`) -/

example {E : Env} {R W : Set Int} {C : Set (DType × Int)} {α : Type}
    {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep} {m : Mem} {s : Shape} {d : DType}
    {ptrs : Tensor s (Ptr d)} {vals : Tensor s d.Val} {mask : Option (Tensor s Bool)} {sc : Scope}
    {k : Tensor s d.Val → Prog α}
    (hcell : ∀ i, (tl.addrs ptrs mask i).isSome = true →
      Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C)
    (hsc : sc ≠ .cta) (htot : ∀ i old, ∃ new, tl.addUpd E d (vals.get i) old new)
    (hk : ∀ order : List (Idx s), order.Perm ((Idx.all s).filter fun i => (tl.addrs ptrs mask i).isSome) →
      ∀ junk olds news : Idx s → d.Val, (∀ i ∈ order, tl.addUpd E d (vals.get i) (olds i) (news i)) →
      SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m
        (k ⟨fun i => if i ∈ order then olds i else junk i⟩)) :
    SafeA E R W C Q log m (tl.atomic_add ptrs vals mask .acqRel sc >>= k) := by
  simp only [tl.atomic_add, Prog.bind_eq, Prog.bind]
  refine SafeA.env ?_
  have := SafeA.atomic (upd := fun i => tl.addUpd E d (vals.get i)) (sem := .acqRel)
    hcell rfl hsc htot hk
  simpa only [Prog.bind_eq] using this

/-! The batch's theorems use the standard axioms only. -/
#print axioms TVA.Math.Float.Format.toRat_of_isInf
#print axioms TVA.Math.Float.Format.maxVal_lt_inf
#print axioms TVA.Math.Float.Format.cvt_rz_sat
#print axioms TVA.Math.Float.Format.sqrtMag_rn_spec
#print axioms TVA.Math.Float.Format.sqrt_rn_err
#print axioms TVA.FloatV.sqrtRn_err
#print axioms TVA.IntV.toInt_cast
#print axioms TVA.IntV.toInt_cast_unsigned
