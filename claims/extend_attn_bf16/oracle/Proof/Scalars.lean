import Proof.Setup
import Proof.Load

/-!
# The scalars and the tensors common to the tiles

`ScOk`: the scalars of a program (`Sc`) have the values of its request. The tensors `offsD`,
`offsM`, `offsN`, `maskD`, `maskM` of the model and their lanes.
-/

namespace Claims.ExtendAttn

open TVA

/-! ## The tensors of the model -/

/-- `offs_d` and `offs_dv = tl.arange(0, BLOCK_DMODEL)` (lines 221-222). -/
def offsD (d : Inst) : Tensor [d.BD] I32 := tl.arange 0 d.BD

/-- `offs_m = tl.arange(0, BLOCK_M)` (line 223). -/
def offsM (d : Inst) : Tensor [d.BM] I32 := tl.arange 0 d.BM

/-- `offs_n = offs_n_prefix = tl.arange(0, 64)` (lines 259, 261). -/
def offsN : Tensor [BN] I32 := tl.arange 0 BN

/-- `mask_d = mask_dv = offs_d < Lq` (lines 226-227). -/
def maskD (d : Inst) : Tensor [d.BD] Bool := (offsD d).ltS (IntV.ofNat d.D : I32)

/-- `mask_m` (line 224): the block's rows inside the request's new tokens. -/
def maskM (d : Inst) (s : Sc d) : Tensor [d.BM] Bool :=
  (toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offsM d)).ltS s.cur_seq_len_extend

/-! ## The scalars -/

/-- Row `r` of program `i`'s block lies inside the request's new tokens. -/
@[reducible] def validRow (d : Inst) (init : Mem) (i : Nat) (r : Nat) : Prop :=
  mOf d i * d.BM + r < extOf d init (bOf d i)


/-- The scalars of program `i` have the values of its request `b`, head `h` and block `m`. -/
structure ScOk (d : Inst) (init : Mem) (i : Nat) (s : Sc d) : Prop where
  seq : s.cur_seq.toInt = bOf d i
  head : s.cur_head.toInt = hOf d i
  blk : s.cur_block_m.toInt = mOf d i
  kvh : s.cur_kv_head.toInt = kvh d (hOf d i)
  qs : s.cur_seq_extend_start_idx.toInt = qsOf d init (bOf d i)
  ext : s.cur_seq_len_extend.toInt = extOf d init (bOf d i)
  ks : s.cur_seq_kv_start_idx.toInt = ksOf d init (bOf d i)
  pre : s.cur_seq_len_prefix.toInt = preOf d init (bOf d i)
  len : s.cur_seq_len.toInt = preOf d init (bOf d i) + extOf d init (bOf d i)

/-- `BLOCK_M` is at most `128`. -/
theorem bm_le {d : Inst} (hd : d.ok) : d.BM ≤ 128 := by rw [hd.bm]; split_ifs <;> omega

/-- `mask_m`'s lanes: row `r` is on exactly when it is inside the request's new tokens. -/
theorem maskM_lane {d : Inst} (hd : d.ok) {init : Mem} {i : Nat} (hi : i < (launch d).size)
    {s : Sc d} (hs : ScOk d init i s) (r : Fin d.BM) :
    (maskM d s).get1 r = decide (validRow d init i r) := by
  obtain ⟨-, -, hm⟩ := prog_range hi
  have hnb := hd.nb_lt
  have hbm := bm_le hd
  have hB : (IntV.ofNat d.BM : I32).toInt = d.BM := IntV.I32.toInt_ofNat (by omega)
  have hmb : mOf d i * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
  have h1 : (s.cur_block_m * (IntV.ofNat d.BM : I32)).toInt = ((mOf d i * d.BM : Nat) : Int) :=
    IntV.mul_ofNat hs.blk (by omega)
  have h2 := IntV.I32.toInt_sadd_arange (n := d.BM) h1 (by omega) r
  show IntV.lt ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offsM d)).get1 r)
    s.cur_seq_len_extend = _
  have e : (toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offsM d)).get1 r =
      castQ d.cm ((s.cur_block_m * (IntV.ofNat d.BM : I32) + offsM d).get1 r) :=
    toQv_get d.cm _ _
  rw [e, IntV.lt_eq, castQ_toInt]
  change decide (((s.cur_block_m * (IntV.ofNat d.BM : I32) + tl.arange 0 d.BM).get1 r).toInt < _) = _
  rw [h2, hs.ext]
  simp only [validRow, Nat.cast_lt]

/-- The mask data `cm` the kernel loads (lines 213-219): none without the custom mask; with it the
request's `mask_indptr` entry and, with a window, its `window_kv_offsets` entry. -/
def CmOk (d : Inst) (init : Mem) (i : Nat) (cm : Option (I64 × Option I64)) : Prop :=
  (d.cm = false ∧ cm = none) ∨
  (d.cm = true ∧ ∃ (msi : I64) (wo : Option I64), cm = some (msi, wo) ∧
    msi.toInt = mStart d init (bOf d i) ∧
    ((d.sw = 0 ∧ wo = none) ∨
      (0 < d.sw ∧ ∃ w : I64, wo = some w ∧ w.toInt = wOff d init (bOf d i))))

end Claims.ExtendAttn
