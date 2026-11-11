import Proof.Inv
import Proof.RowStat
import Proof.ColTrip

/-!
# What the walk of one trip collects

`PrefData` and `ExtData`: the tensors of one trip of the cached-part loop and of the new-token loop,
with the relation each operation gives (the continuations of the proof rules), lane by lane. They
are all the float glue needs to know about the program.
-/

namespace Claims.ExtendAttn

open TVA

/-- **One cached-part trip**, `BM` rows, `BD` components, 64 keys: the scale `ss` (the product of the
`float32` `sm_scale` and `k_scale`), the score `tl.dot` `qk0` and its product `qk1` by the scale;
the masked scores `tl.where FM qk1 (-∞)`; the row maximum, the new `e_max`, `re_scale = re`, `p`,
the row sum `sm`, the new `deno`; the conversion `p16` of `p`; the weights' `tl.dot` `dotv`, the
scaled accumulator `a1`, the scaled dot `a2` and the new accumulator `acc'`. -/
structure PrefData (P : Platform) (fusion : Bool) (BM BD : Nat) (sm32 : F32) where
  q16 : Tensor [BM, BD] BF16
  k : Tensor [BD, 64] BF16
  v : Tensor [64, BD] BF16
  FM : Tensor [BM, 64] Bool
  acc : Tensor [BM, BD] F32
  deno : Tensor [BM] F32
  e_max : Tensor [BM] F32
  ss : F32
  qk0 : Tensor [BM, 64] F32
  qk1 : Tensor [BM, 64] F32
  row_max : Tensor [BM] F32
  n_e_max : Tensor [BM] F32
  x1 : Tensor [BM] F32
  re : Tensor [BM] F32
  x2 : Tensor [BM, 64] F32
  p : Tensor [BM, 64] F32
  sm : Tensor [BM] F32
  deno' : Tensor [BM] F32
  p16 : Tensor [BM, 64] BF16
  dotv : Tensor [BM, BD] F32
  a1 : Tensor [BM, BD] F32
  a2 : Tensor [BM, BD] F32
  acc' : Tensor [BM, BD] F32
  hss : FloatV.NaNEq (FloatV.mul sm32 (FloatV.hostF32 one64)) ss
  hdot1 : P.dot cfgBF16 q16 k (tl.zeros [BM, 64] .f32) qk0
  hmul1 : ∀ x, FloatV.NaNEq (FloatV.mul (qk0.get x) ss) (qk1.get x)
  hmax : ∀ i : Fin BM, ∃ t : Tree, t.Covers 64 ∧
    t.Eval (FloatV.MaxRel true) (fun j => if h : j < 64 then
      (tl.where FM qk1 (bc ninf32)).get2 i ⟨j, h⟩ else default) (row_max.get1 i)
  hnm : ∀ i, FloatV.MaxRel true
    ((tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max).get i) (e_max.get i)
    (n_e_max.get i)
  hsub1 : ∀ i, FloatV.NaNEq (FloatV.sub (e_max.get i) (n_e_max.get i)) (x1.get i)
  hexp1 : ∀ i, P.math .exp .binary32 (x1.get i) (re.get i)
  hsub2 : ∀ x, FloatV.NaNEq (FloatV.sub ((tl.where FM qk1 (bc ninf32)).get x)
    ((n_e_max.col 64).get x)) (x2.get x)
  hexp2 : ∀ x, P.math .exp .binary32 (x2.get x) (p.get x)
  hsum : ∀ i : Fin BM, ∃ t : Tree, t.Covers 64 ∧ t.depth ≤ tl.sumBound P 64 ∧
    t.Eval (AddRel false) (fun j => if h : j < 64 then p.get2 i ⟨j, h⟩ else default) (sm.get1 i)
  hden : ∀ i, FloatV.NaNEq (FloatV.add (FloatV.mul (deno.get i) (re.get i)) (sm.get i))
      (deno'.get i) ∨
    (fusion = true ∧ FloatV.NaNEq (FloatV.fma (deno.get i) (re.get i) (sm.get i)) (deno'.get i))
  hcvt : ∀ x, FloatV.NaNEq (FloatV.cvtWith .rn (p.get x)) (p16.get x)
  hdot2 : P.dot cfgBF16 p16 v (tl.zeros [BM, BD] .f32) dotv
  hmul2 : ∀ x, FloatV.NaNEq (FloatV.mul (acc.get x) ((re.col BD).get x)) (a1.get x)
  hmul3 : ∀ x, FloatV.NaNEq (FloatV.mul (dotv.get x)
    ((bc (FloatV.hostF32 one64) : Tensor [BM, BD] F32).get x)) (a2.get x)
  hadd : ∀ x, AddRel fusion (a1.get x) (a2.get x) (acc'.get x)

/-- **One new-token trip**: as `PrefData`, with `q` itself (no conversion), the scale `sm32` alone,
and the weights' product accumulated by `tl.dot_add` into the scaled accumulator `a1`. -/
structure ExtData (P : Platform) (fusion : Bool) (BM BD : Nat) (sm32 : F32) where
  q : Tensor [BM, BD] BF16
  k : Tensor [BD, 64] BF16
  v : Tensor [64, BD] BF16
  FM : Tensor [BM, 64] Bool
  acc : Tensor [BM, BD] F32
  deno : Tensor [BM] F32
  e_max : Tensor [BM] F32
  qk0 : Tensor [BM, 64] F32
  qk1 : Tensor [BM, 64] F32
  row_max : Tensor [BM] F32
  n_e_max : Tensor [BM] F32
  x1 : Tensor [BM] F32
  re : Tensor [BM] F32
  x2 : Tensor [BM, 64] F32
  p : Tensor [BM, 64] F32
  sm : Tensor [BM] F32
  deno' : Tensor [BM] F32
  p16 : Tensor [BM, 64] BF16
  a1 : Tensor [BM, BD] F32
  acc' : Tensor [BM, BD] F32
  hdot1 : P.dot cfgBF16 q k (tl.zeros [BM, 64] .f32) qk0
  hmul1 : ∀ x, FloatV.NaNEq (FloatV.mul (qk0.get x) ((bc sm32 : Tensor [BM, 64] F32).get x))
    (qk1.get x)
  hmax : ∀ i : Fin BM, ∃ t : Tree, t.Covers 64 ∧
    t.Eval (FloatV.MaxRel true) (fun j => if h : j < 64 then
      (tl.where FM qk1 (bc ninf32)).get2 i ⟨j, h⟩ else default) (row_max.get1 i)
  hnm : ∀ i, FloatV.MaxRel true
    ((tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max).get i) (e_max.get i)
    (n_e_max.get i)
  hsub1 : ∀ i, FloatV.NaNEq (FloatV.sub (e_max.get i) (n_e_max.get i)) (x1.get i)
  hexp1 : ∀ i, P.math .exp .binary32 (x1.get i) (re.get i)
  hsub2 : ∀ x, FloatV.NaNEq (FloatV.sub ((tl.where FM qk1 (bc ninf32)).get x)
    ((n_e_max.col 64).get x)) (x2.get x)
  hexp2 : ∀ x, P.math .exp .binary32 (x2.get x) (p.get x)
  hsum : ∀ i : Fin BM, ∃ t : Tree, t.Covers 64 ∧ t.depth ≤ tl.sumBound P 64 ∧
    t.Eval (AddRel false) (fun j => if h : j < 64 then p.get2 i ⟨j, h⟩ else default) (sm.get1 i)
  hden : ∀ i, FloatV.NaNEq (FloatV.add (FloatV.mul (deno.get i) (re.get i)) (sm.get i))
      (deno'.get i) ∨
    (fusion = true ∧ FloatV.NaNEq (FloatV.fma (deno.get i) (re.get i) (sm.get i)) (deno'.get i))
  hcvt : ∀ x, FloatV.NaNEq (FloatV.cvtWith .rn (p.get x)) (p16.get x)
  hmul2 : ∀ x, FloatV.NaNEq (FloatV.mul (acc.get x) ((re.col BD).get x)) (a1.get x)
  hdot2 : P.dot cfgBF16 p16 v a1 acc' ∨ ∃ d', P.dot cfgBF16 p16 v (Tensor.full (cfgBF16.acc.ofBits 0)) d' ∧
    ∀ x, AccAdd fusion .f32 (d'.get x) (a1.get x) (acc'.get x)

end Claims.ExtendAttn
