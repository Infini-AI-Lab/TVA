import Claim
import TVA.Proof
import Proof.Util

/-!
# The body, split into the prologue and the tail

The model's body first computes four `int64` scalars (the chunk index, the sequence's first token,
its length and the value head: lines 99–106) and then runs the tail (lines 108–149) on them.
`tail` is the second part; `body_eq` says the body is the prologue followed by `tail`. The tail
forms the head's base pointers and the token offsets, then runs `tailRest`, which `tailRest_eq`
cuts into the gate block, the `K` loop, the epilogue of the matrix and the `V` loop. Each piece
takes the rest of the program as its continuation, so that the nesting of the binds is the
source's.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The rest of the tail once the pointers and the token offsets `o_t` are known (source lines
113–149): `q`, `k`, `do_`, `dv` the head's base pointers, `gb` the gate's, `o_t` the chunk's token
offsets and `T` the sequence length. -/
def tailRest (d : Inst) (q k do_ dv : Ptr .bf16) (gb : Ptr .f32) (o_t : Tensor [d.BT] I64)
    (T : I64) : Prog Unit := do
    let m_t := o_t.ltS T
    let b_g : Tensor [d.BT] F32 ← if d.useG then do
        let p_g := gb + o_t * (IntV.ofNat d.HV : I64)
        tl.load p_g (some m_t) (some (bc zf32))
      else pure (bc zf32)
    let b_A := tl.zeros [d.BT, d.BT] .f32
    let b_A ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + d.BK - 1) / d.BK) : I32)
      (IntV.ofInt 1 : I32) b_A fun i_k b_A => do
        let o_k : Tensor [d.BK] I32 := i_k * (IntV.ofNat d.BK : I32) + tl.arange 0 d.BK
        let m_k := o_k.ltS (IntV.ofNat d.K : I32)
        let p_k := k + (o_t * (IntV.ofNat (d.H * d.K) : I64)).col d.BK + o_k.row d.BT
        let p_q := q + o_k.col d.BT + (o_t * (IntV.ofNat (d.H * d.K) : I64)).row d.BK
        let b_k : Tensor [d.BT, d.BK] BF16 ← tl.load p_k (some ((m_t.col d.BK).and (m_k.row d.BT)))
          (some (← tl.convert .f32 .bf16 (tl.full [d.BT, d.BK] zf32)))
        let b_q : Tensor [d.BK, d.BT] BF16 ← tl.load p_q (some ((m_k.col d.BT).and (m_t.row d.BK)))
          (some (← tl.convert .f32 .bf16 (tl.full [d.BK, d.BT] zf32)))
        let b_kq : Tensor [d.BT, d.BT] F32 ← tl.dot cfgBF16 b_k b_q (tl.zeros [d.BT, d.BT] .f32)
        tl.mul_add b_kq (bc (FloatV.hostF32 d.scale)) b_A
    let b_A ← if d.useG then do
        let b_dg ← tl.sub (b_g.row d.BT) (b_g.col d.BT)
        tl.mul b_A (← tl.exp2 b_dg)
      else pure b_A
    let m_A := (Tensor.le (o_t.col d.BT) (o_t.row d.BT)).and ((m_t.col d.BT).and (m_t.row d.BT))
    let b_A : Tensor [d.BT, d.BT] BF16 ← tl.cvt (tl.where m_A b_A (bc zf32))
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.V + d.BV - 1) / d.BV) : I32)
      (IntV.ofInt 1 : I32) () fun i_v _ => do
        let o_v : Tensor [d.BV] I32 := i_v * (IntV.ofNat d.BV : I32) + tl.arange 0 d.BV
        let m_v := o_v.ltS (IntV.ofNat d.V : I32)
        let p_do := do_ + (o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV + o_v.row d.BT
        let p_dv := dv + (o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV + o_v.row d.BT
        let b_do : Tensor [d.BT, d.BV] BF16 ← tl.load p_do (some ((m_t.col d.BV).and (m_v.row d.BT)))
          (some (← tl.convert .f32 .bf16 (tl.full [d.BT, d.BV] zf32)))
        let b_dv : Tensor [d.BT, d.BV] F32 ← tl.dot cfgBF16 b_A b_do (tl.zeros [d.BT, d.BV] .f32)
        let o16 : Tensor [d.BT, d.BV] BF16 ← tl.cvt b_dv
        tl.store p_dv o16 (some ((m_t.col d.BV).and (m_v.row d.BT)))

/-- The tail of the body (source lines 108–149) after the prologue's scalars: the chunk index
`i_t`, the sequence's first token `bos`, its length `T` and the value head `i_h` (all `int64`). -/
def tail (d : Inst) (i_t bos T i_h : I64) : Prog Unit := do
    let q := d.q.ptr 0 + (bos * (IntV.ofNat d.H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (d.HV / d.H) : I64))).item) *
        (IntV.ofNat d.K : I64)
    let k := d.k.ptr 0 + (bos * (IntV.ofNat d.H : I64) +
      (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (d.HV / d.H) : I64))).item) *
        (IntV.ofNat d.K : I64)
    let do_ := d.dO.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64)
    let dv := d.dv.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64)
    let o_t : Tensor [d.BT] I64 :=
      bc (i_t * (IntV.ofNat d.BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BT))
    tailRest d q k do_ dv (d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h)) o_t T

/-! ### The tail in pieces

The pieces take their continuation, so that the nesting of the binds is the source's. -/

/-- The gate block (lines 119–122), then `k` with it. -/
def gLoadK {α : Type} (d : Inst) (gb : Ptr .f32) (o_t : Tensor [d.BT] I64)
    (m_t : Tensor [d.BT] Bool) (k : Tensor [d.BT] F32 → Prog α) : Prog α := do
  let b_g : Tensor [d.BT] F32 ← if d.useG then do
      let p_g := gb + o_t * (IntV.ofNat d.HV : I64)
      tl.load p_g (some m_t) (some (bc zf32))
    else pure (bc zf32)
  k b_g

/-- One trip of the `K` loop (lines 129–136). -/
def kBody (d : Inst) (q k : Ptr .bf16) (o_t : Tensor [d.BT] I64) (m_t : Tensor [d.BT] Bool)
    (i_k : I32) (b_A : Tensor [d.BT, d.BT] F32) : Prog (Tensor [d.BT, d.BT] F32) := do
  let o_k : Tensor [d.BK] I32 := i_k * (IntV.ofNat d.BK : I32) + tl.arange 0 d.BK
  let m_k := o_k.ltS (IntV.ofNat d.K : I32)
  let p_k := k + (o_t * (IntV.ofNat (d.H * d.K) : I64)).col d.BK + o_k.row d.BT
  let p_q := q + o_k.col d.BT + (o_t * (IntV.ofNat (d.H * d.K) : I64)).row d.BK
  let b_k : Tensor [d.BT, d.BK] BF16 ← tl.load p_k (some ((m_t.col d.BK).and (m_k.row d.BT)))
    (some (← tl.convert .f32 .bf16 (tl.full [d.BT, d.BK] zf32)))
  let b_q : Tensor [d.BK, d.BT] BF16 ← tl.load p_q (some ((m_k.col d.BT).and (m_t.row d.BK)))
    (some (← tl.convert .f32 .bf16 (tl.full [d.BK, d.BT] zf32)))
  let b_kq : Tensor [d.BT, d.BT] F32 ← tl.dot cfgBF16 b_k b_q (tl.zeros [d.BT, d.BT] .f32)
  tl.mul_add b_kq (bc (FloatV.hostF32 d.scale)) b_A

/-- The `K` loop (lines 127–136), then `k` with the accumulated `scale * k @ qᵀ`. -/
def kLoopK {α : Type} (d : Inst) (q k : Ptr .bf16) (o_t : Tensor [d.BT] I64)
    (m_t : Tensor [d.BT] Bool) (cont : Tensor [d.BT, d.BT] F32 → Prog α) : Prog α := do
  let b_A ← tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + d.BK - 1) / d.BK) : I32)
    (IntV.ofInt 1 : I32) (tl.zeros [d.BT, d.BT] .f32) (kBody d q k o_t m_t)
  cont b_A

/-- The epilogue of the matrix (lines 137–140): the gate, the causal mask, the rounding to
`bfloat16`; then `k` with the rounded matrix. -/
def epiK {α : Type} (d : Inst) (o_t : Tensor [d.BT] I64) (m_t : Tensor [d.BT] Bool)
    (b_g : Tensor [d.BT] F32) (b_A : Tensor [d.BT, d.BT] F32)
    (cont : Tensor [d.BT, d.BT] BF16 → Prog α) : Prog α := do
  let b_A ← if d.useG then do
      let b_dg ← tl.sub (b_g.row d.BT) (b_g.col d.BT)
      tl.mul b_A (← tl.exp2 b_dg)
    else pure b_A
  let m_A := (Tensor.le (o_t.col d.BT) (o_t.row d.BT)).and ((m_t.col d.BT).and (m_t.row d.BT))
  let b_A : Tensor [d.BT, d.BT] BF16 ← tl.cvt (tl.where m_A b_A (bc zf32))
  cont b_A

/-- One trip of the `V` loop (lines 143–149). -/
def vBody (d : Inst) (do_ dv : Ptr .bf16) (o_t : Tensor [d.BT] I64) (m_t : Tensor [d.BT] Bool)
    (b_A : Tensor [d.BT, d.BT] BF16) (i_v : I32) (_ : Unit) : Prog Unit := do
  let o_v : Tensor [d.BV] I32 := i_v * (IntV.ofNat d.BV : I32) + tl.arange 0 d.BV
  let m_v := o_v.ltS (IntV.ofNat d.V : I32)
  let p_do := do_ + (o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV + o_v.row d.BT
  let p_dv := dv + (o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV + o_v.row d.BT
  let b_do : Tensor [d.BT, d.BV] BF16 ← tl.load p_do (some ((m_t.col d.BV).and (m_v.row d.BT)))
    (some (← tl.convert .f32 .bf16 (tl.full [d.BT, d.BV] zf32)))
  let b_dv : Tensor [d.BT, d.BV] F32 ← tl.dot cfgBF16 b_A b_do (tl.zeros [d.BT, d.BV] .f32)
  let o16 : Tensor [d.BT, d.BV] BF16 ← tl.cvt b_dv
  tl.store p_dv o16 (some ((m_t.col d.BV).and (m_v.row d.BT)))

/-- The `V` loop (lines 142–149): `dv = A @ do` a column block at a time. -/
def vLoop (d : Inst) (do_ dv : Ptr .bf16) (o_t : Tensor [d.BT] I64) (m_t : Tensor [d.BT] Bool)
    (b_A : Tensor [d.BT, d.BT] BF16) : Prog Unit :=
  tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.V + d.BV - 1) / d.BV) : I32)
    (IntV.ofInt 1 : I32) () (vBody d do_ dv o_t m_t b_A)

/-- `tailRest` in pieces. -/
theorem tailRest_eq (d : Inst) (q k do_ dv : Ptr .bf16) (gb : Ptr .f32) (o_t : Tensor [d.BT] I64)
    (T : I64) :
    tailRest d q k do_ dv gb o_t T =
      gLoadK d gb o_t (o_t.ltS T) fun b_g =>
        kLoopK d q k o_t (o_t.ltS T) fun b_A =>
          epiK d o_t (o_t.ltS T) b_g b_A fun b_A =>
            vLoop d do_ dv o_t (o_t.ltS T) b_A := rfl

/-- The model's body, unfolded into the prologue (lines 99–106) and `tail`. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let i_t := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 0))).item
        let i_bh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 1))).item
        let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.HV : I64))).item
        let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.HV : I64))).item
        let (i_t, bos, T) ← if d.varlen then do
            let i_n := (← tl.convert d.idxD .i32 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64))))).item
            let i_t := (← tl.convert d.idxD .i64 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
            let bos := (← tl.convert d.idxD .i64 (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n)))).item
            let eos := (← tl.convert d.idxD .i64
              (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))))).item
            pure (i_t, bos, eos - bos)
          else do
            let T64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.T : I32))).item
            pure (i_t, i_b * T64, T64)
        tail d i_t bos T i_h) := rfl

end Claims.ChunkBwdDvLocal
