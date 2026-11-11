import Claim
import TVA.Proof
import Proof.Meta

/-!
# The trip's `int32` arithmetic

The values of the scalar expressions of a trip: the last and first token of the chunk, and the
helper facts about the constants `1`, `chunk_size`, `nchunks` as `int32`.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The `int32` constant `1` has value `1`. -/
theorem toInt_one : (IntV.ofInt 1 : I32).toInt = ((1 : Nat) : Int) := by
  simpa using IntV.toInt_ofInt_one (w := 32) (by norm_num)

/-- The `int32` constant `0` has value `0`. -/
theorem toInt_zero : (IntV.ofInt 0 : I32).toInt = ((0 : Nat) : Int) := by
  simpa using IntV.toInt_ofInt_zero (w := 32)

/-- The constant `IntV.ofNat 1` (a stride of `1`) has value `1`. -/
theorem toInt_ofNat1 : (IntV.ofNat 1 : I32).toInt = ((1 : Nat) : Int) :=
  IntV.I32.toInt_ofNat (by norm_num)

/-- `nchunks` is an `int32` value. -/
theorem nchunks_lt {d : Inst} (hd : d.ok) : d.nchunks < 2 ^ 31 := by
  have h1 := hd.tok_lt
  have : d.nchunks ≤ d.nchunks * d.chunk := Nat.le_mul_of_pos_right _ (chunk_pos hd)
  omega

/-- `chunk_size` is an `int32` value. -/
theorem chunk_lt {d : Inst} (hd : d.ok) : d.chunk < 2 ^ 31 := by
  have h1 := hd.tok_lt
  have : d.chunk ≤ d.nchunks * d.chunk := Nat.le_mul_of_pos_left _ (nchunks_pos hd)
  omega

/-- `seqlen` is an `int32` value. -/
theorem seqlen_lt {d : Inst} (hd : d.ok) : d.seqlen < 2 ^ 31 := by
  have := hd.seqlen_ctx
  have : (2 : Nat) ^ 22 < 2 ^ 31 := by norm_num
  omega

/-- The last token of the chunk, as the trip computes it (source lines 170-173). -/
theorem end_off_toInt {d : Inst} (hd : d.ok) {c : I32} {t : Nat} (hc : c.toInt = t)
    (ht : t < d.nchunks) :
    ((tl.iminimum (Tensor.scalar ((c + IntV.ofInt 1) * IntV.ofNat d.chunk))
        (Tensor.scalar (IntV.ofNat d.seqlen))).item - IntV.ofInt 1 : I32).toInt =
      (endTok d t : Nat) := by
  have hch := chunk_pos hd
  have hlt : (t + 1) * d.chunk < 2 ^ 31 := by
    have h1 : (t + 1) * d.chunk ≤ d.nchunks * d.chunk := Nat.mul_le_mul_right _ ht
    have := hd.tok_lt
    omega
  have h1 : (c + IntV.ofInt 1 : I32).toInt = ((t + 1 : Nat) : Int) :=
    IntV.I32.add_nat hc toInt_one (by have := hd.tok_lt; have : t < d.nchunks * d.chunk := by nlinarith
                                      omega)
  have h2 : ((c + IntV.ofInt 1) * IntV.ofNat d.chunk : I32).toInt = (((t + 1) * d.chunk : Nat) : Int) :=
    IntV.I32.mul_nat h1 (IntV.I32.toInt_ofNat (chunk_lt hd)) hlt
  have h3 : (tl.iminimum (Tensor.scalar ((c + IntV.ofInt 1) * IntV.ofNat d.chunk))
      (Tensor.scalar (IntV.ofNat d.seqlen))).item.toInt =
      ((min ((t + 1) * d.chunk) d.seqlen : Nat) : Int) := by
    show (IntV.min _ _).toInt = _
    rw [IntV.toInt_min]
    show Min.min (((c + IntV.ofInt 1) * IntV.ofNat d.chunk : I32).toInt)
      ((IntV.ofNat d.seqlen : I32).toInt) = _
    rw [h2, IntV.I32.toInt_ofNat (seqlen_lt hd)]
    omega
  have hpos : 1 ≤ min ((t + 1) * d.chunk) d.seqlen := by
    have := seqlen_pos hd
    have : 1 ≤ (t + 1) * d.chunk := Nat.mul_pos (by omega) hch
    omega
  rw [IntV.I32.sub_nat h3 toInt_one hpos (by omega)]
  rfl

/-- The first token of the chunk, as the trip computes it (source lines 194-197). -/
theorem start_off_toInt {d : Inst} (hd : d.ok) {c : I32} {t : Nat} (hc : c.toInt = t)
    (ht : t < d.nchunks) :
    (tl.iminimum (Tensor.scalar (c * IntV.ofNat d.chunk))
        (Tensor.scalar (IntV.ofNat d.seqlen))).item.toInt = ((t * d.chunk : Nat) : Int) := by
  have hlt : t * d.chunk < 2 ^ 31 := by
    have h1 : t * d.chunk ≤ d.nchunks * d.chunk := Nat.mul_le_mul_right _ ht.le
    have := hd.tok_lt
    omega
  have h2 : (c * IntV.ofNat d.chunk : I32).toInt = ((t * d.chunk : Nat) : Int) :=
    IntV.I32.mul_nat hc (IntV.I32.toInt_ofNat (chunk_lt hd)) hlt
  show (IntV.min _ _).toInt = _
  rw [IntV.toInt_min]
  show Min.min ((c * IntV.ofNat d.chunk : I32).toInt) ((IntV.ofNat d.seqlen : I32).toInt) = _
  rw [h2, IntV.I32.toInt_ofNat (seqlen_lt hd)]
  have := chunk_start_lt hd ht
  omega

/-- The chunk stride `nheads · dim` is an `int32` value. -/
theorem sSc_lt {d : Inst} (hd : d.ok) : sSc d < 2 ^ 31 := by
  have h1 := hd.st_lt
  have : sSc d ≤ d.nchunks * (d.nheads * d.dim) :=
    Nat.le_mul_of_pos_left _ (nchunks_pos hd)
  unfold sSc at *
  omega

/-- The chunk stride as an `int32` constant. -/
theorem sSc_toInt {d : Inst} (hd : d.ok) : (IntV.ofNat (sSc d) : I32).toInt = (sSc d : Int) :=
  IntV.I32.toInt_ofNat (sSc_lt hd)

/-- The `dA_cumsum` chunk stride (`chunk_size`) as an `int32` constant. -/
theorem sDc_toInt {d : Inst} (hd : d.ok) : (IntV.ofNat (sDc d) : I32).toInt = (sDc d : Int) :=
  IntV.I32.toInt_ofNat (chunk_lt hd)

/-- A lane of a pointer block moved by a scalar. -/
theorem get1_map_add {n : Nat} {dt : DType} (ps : Tensor [n] (Ptr dt)) (k : I32) (i : Fin n) :
    (ps.map (fun (q : Ptr dt) => q + k)).get1 i = (ps.get1 i).offset k.toInt := rfl

/-- With initial states, the host's `N` is an `int32` value. -/
theorem hostN_lt {d : Inst} (hd : d.ok) (hinit : d.hasInit = true) : hostN d < 2 ^ 31 := by
  have := hd.nmeta_lt
  rw [hd.nmeta_eq, ite_eq_left hinit] at this
  exact this

end Claims.SsdStatePassing
