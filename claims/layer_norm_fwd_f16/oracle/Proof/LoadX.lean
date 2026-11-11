import Proof.Defs

/-!
# The loads of the kernel

`other16 >>= load >>= cvt` of `X + cols` (lines 70, 77, 91) is one rule, `safe_loadX`: lane `i` of the
`float32` block is the row's `x[t, i]` (exactly) for `i < N` and `0` otherwise. The loads of
`weight` and `bias` (lines 89–90) are `safe_loadWB`.
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- An element of `x`, `weight` or `bias` below the launch's extent read in a memory that still
holds the inputs gives the initial memory's element. -/
theorem MemOk.read_x {init m : Mem} (hm : MemOk d init m) {k : Nat} (hk : k < d.M * d.N) :
    m.read .f16 (d.x.ptr k).addr = init.read .f16 (d.x.ptr k).addr :=
  Mem.read_congr fun _ hz => hm.1 _ (Or.inl (Or.inl ⟨k, hk, hz⟩))

/-- Reading an element of `weight` in a memory that still holds the inputs. -/
theorem MemOk.read_w {init m : Mem} (hm : MemOk d init m) {k : Nat} (hk : k < d.N) :
    m.read .f16 (d.w.ptr k).addr = init.read .f16 (d.w.ptr k).addr :=
  Mem.read_congr fun _ hz => hm.1 _ (Or.inl (Or.inr ⟨k, hk, hz⟩))

/-- Reading an element of `bias` in a memory that still holds the inputs. -/
theorem MemOk.read_b {init m : Mem} (hm : MemOk d init m) {k : Nat} (hk : k < d.N) :
    m.read .f16 (d.b.ptr k).addr = init.read .f16 (d.b.ptr k).addr :=
  Mem.read_congr fun _ hz => hm.1 _ (Or.inr ⟨k, hk, hz⟩)

/-- A buffer valid in `init` is valid in a memory that still holds the inputs. -/
theorem MemOk.valid {init m : Mem} (hm : MemOk d init m) {dy : DType} {B : Buf dy}
    (hB : B.Valid init) : B.Valid m :=
  Buf.valid_of_agree hB fun z hz => hm.2 z (hB.2.2.2 z hz)

/-- `rval` of an element that `init` holds. -/
theorem rval_of_read {init : Mem} {B : Buf .f16} {k : Nat} {v : F16}
    (h : init.read .f16 (B.ptr (k : Int)).addr = some v) : rval init B k = v := by
  unfold rval; rw [h]

/-- **`other=0.`, the load of `X + cols`, and the conversion to `float32`** (lines 70, 77, 91):
lane `i` is the row's `x[t, i]` as a `float32`, exactly, for `i < N`, and `0` beyond. -/
theorem safe_loadX (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat} (ht : t < d.M) {c : Ctx}
    (hc : c.pid 0 = t) {m : Mem} (hm : MemOk d init m) {E : Env} {W : Set ℤ} {α : Type}
    {Q : α → Mem → Prop}
    {k : Tensor [d.BLOCK - 0] F32 → Prog α}
    (hk : ∀ a : Tensor [d.BLOCK - 0] F32, (∀ i : Idx [d.BLOCK - 0], (a.get i).isFinite = true ∧
        (a.get i).toReal = (rowP d hd init hpre t ht).xl (i.1 : Nat)) → Safe E (R d t) W Q m (k a)) :
    Safe E (R d t) W Q m ((other16 d.BLOCK).bind fun o =>
      (tl.load (d.x.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32) +
          ((IntV.ofInt 0 : I32) + tl.arange 0 d.BLOCK))
        (some (((IntV.ofInt 0 : I32) + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N))) (some o)).bind
        fun l => (tl.cvt l).bind k) := by
  have hB31 : d.BLOCK ≤ 2 ^ 31 := by have := hd.block_le; omega
  have hN31 : d.N < 2 ^ 31 := by have := hd.n_le; omega
  refine safe_other16 fun o ho => ?_
  refine safe_load_buf d.x (g := fun i : Idx [d.BLOCK - 0] => idx d t i.1.val)
    (en := fun i : Idx [d.BLOCK - 0] => decide (i.1.val < d.N))
    (Buf.addrs_of_get d.x (fun i => cols_mask hB31 hN31 i) fun i _ => row_ptr hd ht hc d.x i)
    (fun i hi => ?_) (hm.valid hpre.1) ?_ fun l hl hl' => ?_
  · exact lt_of_lt_of_le (idx_lt ht (by simpa using hi)) hd.x_len
  · rintro _ ⟨_, ⟨i, hi, rfl⟩, hz⟩
    exact Or.inl (Or.inl ⟨_, ⟨i.1.val, by simpa using hi, rfl⟩, hz⟩)
  refine safe_cvt ⟨Format.binary16_std, Format.binary32_std, Or.inl rfl⟩ fun a ha => hk a fun i => ?_
  by_cases hi : (i.1 : Nat) < d.N
  · have hen : decide ((i.1 : Nat) < d.N) = true := by simpa using hi
    have hread := hl i hen
    rw [hm.read_x (idx_lt ht hi)] at hread
    have hl_eq : l.get i = xv d init t i.1 := (rval_of_read hread).symm
    have hfin : (xv d init t i.1).isFinite = true := hpre.2.2.2.2.2.2.1 t ht _ hi
    obtain ⟨hf1, -, hf3⟩ := FloatV.cvt_f16_f32_exact hfin
    have ha' := ha i
    rw [FloatV.cvtWith_rn, hl_eq] at ha'
    rw [ha'.eq_of_isFinite hf1]
    refine ⟨hf1, ?_⟩
    rw [hf3]
    simp [RowP.xl, rowP, hi, xR]
  · have hen : decide ((i.1 : Nat) < d.N) = false := by simpa using hi
    have hl0 : l.get i = FloatV.ofPat 0 := (hl' i hen o rfl).trans (ho i)
    have ha' := ha i
    rw [FloatV.cvtWith_rn, hl0, FloatV.cvt_ofPat_zero Format.binary16_regular] at ha'
    rw [ha'.eq (FloatV.isNaN_ofPat_zero Format.binary32_regular)]
    refine ⟨FloatV.isFinite_ofPat_zero Format.binary32_regular, ?_⟩
    simp [RowP.xl, rowP, hi, FloatV.toReal_ofPat_zero]

end Claims.LayerNormFwdF16
