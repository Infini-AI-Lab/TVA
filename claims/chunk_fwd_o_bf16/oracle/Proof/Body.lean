import Claim
import TVA.Proof

/-!
# `chunk_fwd_o_bf16`: the model's body in three parts

The kernel's body is cut where the scalars of lines 108–121 are known (`tail1`, lines 123–127: the
five pointers) and where the pointers are known (`tail2`, lines 129–181: the loop, the gates, the
epilogue). `body_eq` shows the cut changes nothing (`rfl`).
-/

namespace Claims.ChunkFwdO

open TVA

/-- The body of the loop over `K` (lines 136–158): one trip `i_k` from the accumulators
`(b_o, b_A)`. `q`, `k`, `h` are the offset pointers, `o_t`, `m_t`, `o_v` the token offsets, the
token mask and the value-column offsets. -/
def loopBody (d : Inst) (q k h : Ptr .bf16) (o_t : Tensor [BT] I64) (m_t : Tensor [BT] Bool)
    (o_v : Tensor [d.bk] I64) :
    I32 → Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32 →
      Prog (Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32) :=
  fun i_k (b_o, b_A) => do
    -- line 136: `int32`
    let o_k : Tensor [d.bk] I32 := i_k * (IntV.ofNat d.bk : I32) + tl.arange 0 d.bk
    -- line 137
    let m_k := o_k.ltS (IntV.ofNat d.K : I32)
    -- line 138: `q + o_t[:, None] * (H*K)` (`int64`), then `+ o_k[None, :]` (`int32`)
    let p_q : Tensor [BT, d.bk] (Ptr .bf16) :=
      q + o_t.col d.bk * (IntV.ofNat (d.H * d.K) : I64) + o_k.row BT
    -- line 139
    let p_k : Tensor [d.bk, BT] (Ptr .bf16) :=
      k + o_k.col BT + o_t.row d.bk * (IntV.ofNat (d.H * d.K) : I64)
    -- lines 147–149: `other=0.0`, the `float32` `+0` converted to `bfloat16`
    let b_q ← tl.load p_q (some ((m_t.col d.bk).and (m_k.row BT)))
      (some (← tl.convert .f32 .bf16 (tl.full [BT, d.bk] zf32)))
    let b_k ← tl.load p_k (some ((m_k.col BT).and (m_t.row d.bk)))
      (some (← tl.convert .f32 .bf16 (tl.full [d.bk, BT] zf32)))
    -- lines 140–145, 150, 155–158: the state tile, either way round
    let b_o ← if d.svf then do
        let p_h : Tensor [d.bk, d.bk] (Ptr .bf16) :=
          h + o_v.col d.bk * (IntV.ofNat d.K : I64) + o_k.row d.bk
        let m_h := ((o_v.col d.bk).ltS (IntV.ofNat d.V : I64)).and (m_k.row d.bk)
        let b_h ← tl.load p_h (some m_h)
          (some (← tl.convert .f32 .bf16 (tl.full [d.bk, d.bk] zf32)))
        tl.dot cfgBF16 b_q b_h.trans b_o
      else do
        let p_h : Tensor [d.bk, d.bk] (Ptr .bf16) :=
          h + o_k.col d.bk * (IntV.ofNat d.V : I32) + o_v.row d.bk
        let m_h := (m_k.col d.bk).and ((o_v.row d.bk).ltS (IntV.ofNat d.V : I64))
        let b_h ← tl.load p_h (some m_h)
          (some (← tl.convert .f32 .bf16 (tl.full [d.bk, d.bk] zf32)))
        tl.dot cfgBF16 b_q b_h b_o
    -- line 158
    let b_A ← tl.dot cfgBF16 b_q b_k b_A
    pure (b_o, b_A)

/-- Lines 160–165, `if USE_G:`. -/
def gateG (d : Inst) (o_t : Tensor [BT] I64) (m_t : Tensor [BT] Bool) (i_h bos : I64)
    (b_o : Tensor [BT, d.bk] F32) (b_A : Tensor [BT, BT] F32) :
    Prog (Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32) :=
  if d.useG then do
    let g := d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h)
    let p_g : Tensor [BT] (Ptr .f32) := g + o_t * (IntV.ofNat d.HV : I64)
    let b_g ← tl.load p_g (some m_t) (some (bc zf32))
    let b_o ← tl.mul b_o ((← tl.exp2 b_g).col d.bk)
    let b_A ← tl.mul b_A (← tl.exp2 (← tl.sub (b_g.col BT) (b_g.row BT)))
    pure (b_o, b_A)
  else pure (b_o, b_A)

/-- Lines 166–170, `if USE_G_GAMMA:`. -/
def gateGamma (d : Inst) (i_h : I64) (b_o : Tensor [BT, d.bk] F32) (b_A : Tensor [BT, BT] F32) :
    Prog (Tensor [BT, d.bk] F32 × Tensor [BT, BT] F32) :=
  if d.useGamma then do
    let b_gamma ← tl.load1 (d.gg.ptr 0 + i_h)
    let b_g : Tensor [BT] F32 ← tl.mul (bc b_gamma)
      (← tl.convert .i32 .f32 ((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32)))
    let b_o ← tl.mul b_o ((← tl.exp2 b_g).col d.bk)
    let b_A ← tl.mul b_A (← tl.exp2 (← tl.subc (b_g.col BT) (b_g.row BT)))
    pure (b_o, b_A)
  else pure (b_o, b_A)

/-- Lines 171–181: the causal mask, the scores' rounding, the second dot, the scale, the store. -/
def epilogue (d : Inst) (v o : Ptr .bf16) (o_t : Tensor [BT] I64) (m_t : Tensor [BT] Bool)
    (o_v : Tensor [d.bk] I64) (b_o : Tensor [BT, d.bk] F32) (b_A : Tensor [BT, BT] F32) :
    Prog Unit := do
  let m_A := (Tensor.le (o_t.row BT) (o_t.col BT)).and ((m_t.col BT).and (m_t.row BT))
  let b_A := tl.where m_A b_A (bc zf32)
  let p_v : Tensor [BT, d.bk] (Ptr .bf16) :=
    v + o_t.col d.bk * (IntV.ofNat (d.HV * d.V) : I64) + o_v.row BT
  let p_o : Tensor [BT, d.bk] (Ptr .bf16) :=
    o + o_t.col d.bk * (IntV.ofNat (d.HV * d.V) : I64) + o_v.row BT
  let m_o := (m_t.col d.bk).and ((o_v.ltS (IntV.ofNat d.V : I64)).row BT)
  let b_v ← tl.load p_v (some m_o) (some (← tl.convert .f32 .bf16 (tl.full [BT, d.bk] zf32)))
  let b_o ← tl.addc (← tl.mul b_o (bc (FloatV.hostF32 d.scale)))
    (← tl.mul (← tl.dot cfgBF16 (← tl.cvt b_A) b_v (tl.zeros [BT, d.bk] .f32))
      (bc (FloatV.hostF32 d.scale)))
  let o16 : Tensor [BT, d.bk] BF16 ← tl.cvt b_o
  tl.store p_o o16 (some m_o)

/-- Lines 129–181 of the kernel, from the offset pointers `q k v h o` and the scalars `i_v`,
`i_h`, `i_t`, `bos`, `T`, written as the model writes it. -/
def tail2 (d : Inst) (q k v h o : Ptr .bf16) (i_v i_h i_t bos T : I64) : Prog Unit := do
    let b_o := tl.zeros [BT, d.bk] .f32
    let b_A := tl.zeros [BT, BT] .f32
    let o_t : Tensor [BT] I64 :=
      bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
    let m_t := o_t.ltS T
    let o_v : Tensor [d.bk] I64 :=
      bc (i_v * (IntV.ofNat d.bk : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.bk))
    let (b_o, b_A) ← tl.forRange (IntV.ofInt 0 : I32)
      (IntV.ofNat ((d.K + (d.bk - 1)) / d.bk) : I32)
      (IntV.ofInt 1 : I32) (b_o, b_A) (loopBody d q k h o_t m_t o_v)
    let (b_o, b_A) ← if d.useG then do
        let g := d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h)
        let p_g : Tensor [BT] (Ptr .f32) := g + o_t * (IntV.ofNat d.HV : I64)
        let b_g ← tl.load p_g (some m_t) (some (bc zf32))
        let b_o ← tl.mul b_o ((← tl.exp2 b_g).col d.bk)
        let b_A ← tl.mul b_A (← tl.exp2 (← tl.sub (b_g.col BT) (b_g.row BT)))
        pure (b_o, b_A)
      else pure (b_o, b_A)
    let (b_o, b_A) ← if d.useGamma then do
        let b_gamma ← tl.load1 (d.gg.ptr 0 + i_h)
        let b_g : Tensor [BT] F32 ← tl.mul (bc b_gamma)
          (← tl.convert .i32 .f32 ((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32)))
        let b_o ← tl.mul b_o ((← tl.exp2 b_g).col d.bk)
        let b_A ← tl.mul b_A (← tl.exp2 (← tl.subc (b_g.col BT) (b_g.row BT)))
        pure (b_o, b_A)
      else pure (b_o, b_A)
    epilogue d v o o_t m_t o_v b_o b_A

/-- `tail2` with its two `if USE_G` blocks as programs of their own. -/
def tail2s (d : Inst) (q k v h o : Ptr .bf16) (i_v i_h i_t bos T : I64) : Prog Unit := do
    let b_o := tl.zeros [BT, d.bk] .f32
    let b_A := tl.zeros [BT, BT] .f32
    let o_t : Tensor [BT] I64 :=
      bc (i_t * (IntV.ofNat BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 BT))
    let m_t := o_t.ltS T
    let o_v : Tensor [d.bk] I64 :=
      bc (i_v * (IntV.ofNat d.bk : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.bk))
    let (b_o, b_A) ← tl.forRange (IntV.ofInt 0 : I32)
      (IntV.ofNat ((d.K + (d.bk - 1)) / d.bk) : I32)
      (IntV.ofInt 1 : I32) (b_o, b_A) (loopBody d q k h o_t m_t o_v)
    let (b_o, b_A) ← gateG d o_t m_t i_h bos b_o b_A
    let (b_o, b_A) ← gateGamma d i_h b_o b_A
    epilogue d v o o_t m_t o_v b_o b_A

/-- The two forms of the tail are the same program. -/
theorem tail2_eq (d : Inst) (q k v h o : Ptr .bf16) (i_v i_h i_t bos T : I64) :
    tail2 d q k v h o i_v i_h i_t bos T = tail2s d q k v h o i_v i_h i_t bos T := by
  unfold tail2 tail2s gateG gateGamma
  rcases Bool.eq_false_or_eq_true d.useG with h1 | h1 <;>
    rcases Bool.eq_false_or_eq_true d.useGamma with h2 | h2 <;>
    simp [h1, h2, Prog.bind_assoc, Prog.ret_bind]

/-- Lines 123–127 of the kernel: the five offset pointers; then `tail2`. -/
def tail1 (d : Inst) (i_v i_h i_tg i_t bos T : I64) : Prog Unit := do
    let q := d.q.ptr 0 + (bos * (IntV.ofNat d.H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (d.HV / d.H) : I64))).item) *
        (IntV.ofNat d.K : I64)
    let k := d.k.ptr 0 + (bos * (IntV.ofNat d.H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (d.HV / d.H) : I64))).item) *
        (IntV.ofNat d.K : I64)
    let v := d.v.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64)
    let o := d.o.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64)
    let h := d.h.ptr 0 + (i_tg * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.K : I64) *
      (IntV.ofNat d.V : I64)
    tail2 d q k v h o i_v i_h i_t bos T

set_option maxHeartbeats 1000000 in
/-- The kernel's body on program `ctx`: lines 108–121 (the scalars), then `tail1`. -/
theorem body_eq (d : Inst) (ctx : Ctx) :
    (model d).body ctx =
      (do
        let pid := tl.program_id ctx 0
        let pid := (← tl.convert .i32 .i64 (Tensor.scalar pid)).item
        let i_v := (← tl.irem (Tensor.scalar pid)
          (Tensor.scalar (IntV.ofNat ((d.V + (d.bk - 1)) / d.bk) : I64))).item
        let i_t := (← tl.idiv (Tensor.scalar pid)
          (Tensor.scalar (IntV.ofNat ((d.V + (d.bk - 1)) / d.bk) : I64))).item
        let i_bh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 1))).item
        let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.HV : I64))).item
        let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.HV : I64))).item
        let (i_tg, i_t, bos, T) ← if d.varlen then do
            let i_tg := i_t
            let i_n := (← tl.convert d.idxD .i32 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64))))).item
            let i_t := (← tl.convert d.idxD .i64 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
            let bos := (← tl.convert d.idxD .i64
              (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n)))).item
            let eos := (← tl.convert d.idxD .i64
              (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))))).item
            pure (i_tg, i_t, bos, eos - bos)
          else do
            let NT := (← tl.idiv (Tensor.scalar ((IntV.ofNat d.T : I32) + (IntV.ofNat (BT - 1) : I32)))
              (Tensor.scalar (IntV.ofNat BT : I32))).item
            let NT64 := (← tl.convert .i32 .i64 (Tensor.scalar NT)).item
            let T64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.T : I32))).item
            pure (i_b * NT64 + i_t, i_t, i_b * T64, T64)
        tail1 d i_v i_h i_tg i_t bos T) := rfl

end Claims.ChunkFwdO
