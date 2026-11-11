import Claim
import TVA.Proof
import Proof.Lane

/-!
# The kernel's body, split for the proof

The two models `recompute_w_u_fwd_kernel_b32` and `_b16` differ only in how `beta` is loaded and in
the operations on `b_b` (lines 104, 115, 129): everything else is the same text. `tailProg` is the
body after line 99 with those three places as parameters, so that one walk serves both dtypes of
`beta`. `body32_eq`, `body16_eq` say the models' bodies are `tailProg` with each dtype's parts.
-/

namespace Claims.RecomputeWU

open TVA

/-- Lines 103–104 for a `float32` `beta`: the pointer tile and the masked load, `other=0.0`. -/
def loadBeta32 (beta : Ptr .f32) (bos i_h : I64) (o_t : Tensor [64] I64) (m_t : Tensor [64] Bool)
    (HV : Nat) (k : Tensor [64] F32 → Prog Unit) : Prog Unit := do
  let p_b : Tensor [64] (Ptr .f32) :=
    beta + bos * (IntV.ofNat HV : I64) + i_h + o_t * (IntV.ofNat HV : I64)
  let b_b : Tensor [64] F32 ← tl.load p_b (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
  k b_b

/-- Lines 103–104 for a `bfloat16` `beta`. -/
def loadBeta16 (beta : Ptr .bf16) (bos i_h : I64) (o_t : Tensor [64] I64) (m_t : Tensor [64] Bool)
    (HV : Nat) (k : Tensor [64] BF16 → Prog Unit) : Prog Unit := do
  let p_b : Tensor [64] (Ptr .bf16) :=
    beta + bos * (IntV.ofNat HV : I64) + i_h + o_t * (IntV.ofNat HV : I64)
  let zb : Tensor [64] BF16 ← tl.convert .f32 .bf16 (tl.full [64] (FloatV.litF32 (FloatV.ofPat 0)))
  let b_b ← tl.load p_b (some m_t) (some zb)
  k b_b

/-- Line 115 for a `float32` `beta`. -/
def vbStep32 (b_b : Tensor [64] F32) (b_v : Tensor [64, 64] BF16)
    (k : Tensor [64, 64] BF16 → Prog Unit) : Prog Unit := do
  let b_vf : Tensor [64, 64] F32 ← tl.cvt b_v
  let b_vb : Tensor [64, 64] BF16 ← tl.cvt (← tl.mul b_vf (b_b.col 64))
  k b_vb

/-- Line 115 for a `bfloat16` `beta`. -/
def vbStep16 (b_b : Tensor [64] BF16) (b_v : Tensor [64, 64] BF16)
    (k : Tensor [64, 64] BF16 → Prog Unit) : Prog Unit := do
  let b_vb : Tensor [64, 64] BF16 ← tl.mul b_v (b_b.col 64)
  k b_vb

/-- Lines 129–132 for a `float32` `beta`: `b_k · b_b · b_g`, rounded to `bfloat16`. -/
def kbStep32 (b_b : Tensor [64] F32) (b_g : Tensor [64] F32) (b_k : Tensor [64, 64] BF16)
    (k : Tensor [64, 64] BF16 → Prog Unit) : Prog Unit := do
  let b_kf : Tensor [64, 64] F32 ← tl.cvt b_k
  let b_kb ← tl.mul b_kf (b_b.col 64)
  let b_kb ← tl.mul b_kb (b_g.col 64)
  let b_kbh : Tensor [64, 64] BF16 ← tl.cvt b_kb
  k b_kbh

/-- Lines 129–132 for a `bfloat16` `beta`. -/
def kbStep16 (b_b : Tensor [64] BF16) (b_g : Tensor [64] F32) (b_k : Tensor [64, 64] BF16)
    (k : Tensor [64, 64] BF16 → Prog Unit) : Prog Unit := do
  let b_kb16 : Tensor [64, 64] BF16 ← tl.mul b_k (b_b.col 64)
  let b_kbf : Tensor [64, 64] F32 ← tl.cvt b_kb16
  let b_kb ← tl.mul b_kbf (b_g.col 64)
  let b_kbh : Tensor [64, 64] BF16 ← tl.cvt b_kb
  k b_kbh


/-- The `u` loop's body (lines 110–117) for the trip `i_v`, given line 115's step `vbK`. -/
def uTrip (v u : Ptr .bf16) (HV V : Nat) (bos i_h : I64) (o_t : Tensor [64] I64)
    (m_t : Tensor [64] Bool) (b_A : Tensor [64, 64] BF16)
    (vbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit)
    (i_v : I32) : Prog Unit := do
  let o_v : Tensor [64] I32 := i_v * (IntV.ofNat 64 : I32) + tl.arange 0 64
  let m_v : Tensor [64, 64] Bool := (m_t.col 64).and ((o_v.ltS (IntV.ofNat V : I32)).row 64)
  let p_v : Tensor [64, 64] (Ptr .bf16) :=
    v + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64) +
      o_t.col 64 * (IntV.ofNat (HV * V) : I64) + o_v.row 64
  let p_u : Tensor [64, 64] (Ptr .bf16) :=
    u + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64) +
      o_t.col 64 * (IntV.ofNat (HV * V) : I64) + o_v.row 64
  let zV : Tensor [64, 64] BF16 ← tl.convert .f32 .bf16 (tl.full [64, 64] (FloatV.litF32 (FloatV.ofPat 0)))
  let b_v ← tl.load p_v (some m_v) (some zV)
  vbK b_v fun b_vb => do
    let b_u ← tl.dot cfgU b_A b_vb (tl.zeros [64, 64] .f32)
    let s_u : Tensor [64, 64] BF16 ← tl.cvt b_u
    tl.store p_u s_u (some m_v)

/-- The `w` loop's body (lines 124–133) for the trip `i_k`, given lines 129–132's step `kbK`. -/
def wTrip (k w : Ptr .bf16) (H HV K : Nat) (bos i_h : I64) (o_t : Tensor [64] I64)
    (m_t : Tensor [64] Bool) (b_A : Tensor [64, 64] BF16)
    (kbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit)
    (i_k : I32) : Prog Unit := do
  let o_k : Tensor [64] I32 := i_k * (IntV.ofNat 64 : I32) + tl.arange 0 64
  let m_k : Tensor [64, 64] Bool := (m_t.col 64).and ((o_k.ltS (IntV.ofNat K : I32)).row 64)
  let p_k : Tensor [64, 64] (Ptr .bf16) :=
    k + (bos * (IntV.ofNat H : I64) +
        (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
        (IntV.ofNat K : I64) +
      o_t.col 64 * (IntV.ofNat (H * K) : I64) + o_k.row 64
  let p_w : Tensor [64, 64] (Ptr .bf16) :=
    w + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64) +
      o_t.col 64 * (IntV.ofNat (HV * K) : I64) + o_k.row 64
  let zK : Tensor [64, 64] BF16 ← tl.convert .f32 .bf16 (tl.full [64, 64] (FloatV.litF32 (FloatV.ofPat 0)))
  let b_k ← tl.load p_k (some m_k) (some zK)
  kbK b_k fun b_kbh => do
    let b_w ← tl.dot cfgW b_A b_kbh (tl.zeros [64, 64] .f32)
    let s_w : Tensor [64, 64] BF16 ← tl.cvt b_w
    tl.store p_w s_w (some m_k)

/-- The body after line 99 (`o_t`) and line 101 (`m_t`), with `beta`'s dtype `β` and the three
places that depend on it as parameters. -/
def tailProg {β : DType} (v u A : Ptr .bf16) (k w : Ptr .bf16) (g : Ptr .f32) (H HV K V : Nat)
    (bos i_h : I64) (o_t : Tensor [64] I64) (m_t : Tensor [64] Bool)
    (loadβ : (Tensor [64] β.Val → Prog Unit) → Prog Unit)
    (vbK : Tensor [64] β.Val → Tensor [64, 64] BF16 →
      (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit)
    (kbK : Tensor [64] β.Val → Tensor [64] F32 → Tensor [64, 64] BF16 →
      (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit) : Prog Unit := do
  let o_A : Tensor [64] I32 := tl.arange 0 64
  let m_A : Tensor [64, 64] Bool := (m_t.col 64).and ((o_A.ltS (IntV.ofNat 64 : I32)).row 64)
  loadβ fun b_b => do
    let p_A : Tensor [64, 64] (Ptr .bf16) :=
      A + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat 64 : I64) +
        o_t.col 64 * (IntV.ofNat (HV * 64) : I64) + o_A.row 64
    let zA : Tensor [64, 64] BF16 ← tl.convert .f32 .bf16 (tl.full [64, 64] (FloatV.litF32 (FloatV.ofPat 0)))
    let b_A ← tl.load p_A (some m_A) (some zA)
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((V + (64 - 1)) / 64) : I32) (IntV.ofInt 1 : I32) ()
      fun i_v _ => uTrip v u HV V bos i_h o_t m_t b_A (vbK b_b) i_v
    let p_g : Tensor [64] (Ptr .f32) :=
      g + (bos * (IntV.ofNat HV : I64) + i_h) + o_t * (IntV.ofNat HV : I64)
    let l_g : Tensor [64] F32 ← tl.load p_g (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
    let b_g : Tensor [64] F32 ← tl.exp2 l_g
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((K + (64 - 1)) / 64) : I32) (IntV.ofInt 1 : I32) ()
      fun i_k _ => wTrip k w H HV K bos i_h o_t m_t b_A (kbK b_b b_g) i_k


/-- Lines 91–98: the program's head `i_h`, chunk `i_t`, sequence start `bos` and length `T`, given
to the rest of the body. -/
def preProg {sD : DType} (cu chunk_indices : Ptr sD) (T : I32) (HV : Nat) (IS_VARLEN : Bool)
    (ctx : Ctx) (rest : I64 → I64 → I64 → I64 → Prog Unit) : Prog Unit := do
  let i_t0 : I64 := IntV.cast (tl.program_id ctx 0)
  let i_bh : I64 := IntV.cast (tl.program_id ctx 1)
  let i_b : I64 := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
  let i_h : I64 := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat HV : I64))).item
  let st ← (if IS_VARLEN then (do
      let v_n ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64))
      let i_n : I32 := (← tl.convert sD .i32 (Tensor.scalar v_n)).item
      let v_t ← tl.load1 (chunk_indices + i_t0 * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))
      let i_t : I64 := (← tl.convert sD .i64 (Tensor.scalar v_t)).item
      let v_b ← tl.load1 (cu + i_n)
      let bos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_b)).item
      let v_e ← tl.load1 (cu + i_n + (IntV.ofNat 1 : I32))
      let eos : I64 := (← tl.convert sD .i64 (Tensor.scalar v_e)).item
      let T_n : I64 := eos - bos
      pure (i_t, bos, T_n))
    else (do
      let bos : I64 := i_b * (IntV.cast T : I64)
      let _eos : I64 := i_b * (IntV.cast T : I64) + (IntV.cast T : I64)
      pure (i_t0, bos, (IntV.cast T : I64))) : Prog (I64 × I64 × I64))
  let (i_t, bos, T_n) := st
  rest i_h i_t bos T_n

/-- Line 99: `i_t * BT + tl.arange(0, BT)` in `int64`. -/
def oT (i_t : I64) : Tensor [64] I64 :=
  i_t * (IntV.ofNat 64 : I64) + ((tl.arange 0 64).castInt : Tensor [64] I64)

/-- Line 101: `o_t < T`, the signed `int64` comparison. -/
def mT (o_t : Tensor [64] I64) (T_n : I64) : Tensor [64] Bool := o_t.ltS T_n

/-- The model for a `float32` `beta` is `preProg` and `tailProg` with that dtype's parts. -/
theorem body32_eq {sD : DType} (k v : Ptr .bf16) (beta : Ptr .f32) (w u A : Ptr .bf16)
    (g : Ptr .f32) (cu chunk_indices : Ptr sD) (T : I32) (H HV K V : Nat) (IS_VARLEN : Bool)
    (c : Ctx) :
    (recompute_w_u_fwd_kernel_b32 k v beta w u A g cu chunk_indices T H HV K V 64 64 64 IS_VARLEN).body c =
      preProg cu chunk_indices T HV IS_VARLEN c fun i_h i_t bos T_n =>
        tailProg (β := .f32) v u A k w g H HV K V bos i_h (oT i_t) (mT (oT i_t) T_n)
          (loadBeta32 beta bos i_h (oT i_t) (mT (oT i_t) T_n) HV) vbStep32 kbStep32 := rfl

/-- The model for a `bfloat16` `beta`. -/
theorem body16_eq {sD : DType} (k v : Ptr .bf16) (beta : Ptr .bf16) (w u A : Ptr .bf16)
    (g : Ptr .f32) (cu chunk_indices : Ptr sD) (T : I32) (H HV K V : Nat) (IS_VARLEN : Bool)
    (c : Ctx) :
    (recompute_w_u_fwd_kernel_b16 k v beta w u A g cu chunk_indices T H HV K V 64 64 64 IS_VARLEN).body c =
      preProg cu chunk_indices T HV IS_VARLEN c fun i_h i_t bos T_n =>
        tailProg (β := .bf16) v u A k w g H HV K V bos i_h (oT i_t) (mT (oT i_t) T_n)
          (loadBeta16 beta bos i_h (oT i_t) (mT (oT i_t) T_n) HV) vbStep16 kbStep16 := rfl

end Claims.RecomputeWU
