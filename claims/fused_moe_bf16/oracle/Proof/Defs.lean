import TVA.Proof
import Claim
import Proof.Pid

/-!
# The model's body, cut into the pieces the proof walks

`fused_moe_kernel` on an instance, after the `pid` opening (`pidPrefix`), as `rest1` (the block
offsets and the early return), `rest2` (the block's entries, the mask and the expert), `rest3` (the
`-1` expert or the `K` loop), with the loop's trip `tripBody` and the epilogue `tailOf` apart. Each
is the text of the model at the instance's parameters; `body_eq` is `rfl`.
-/

namespace Claims.FusedMoe

open TVA

/-- The loop's carried values: `a`'s pointers, `b`'s pointers and the accumulator. -/
abbrev St (d : Inst) :=
  Tensor [d.BM, d.BK] (Ptr .bf16) × Tensor [d.BK, d.BN] (Ptr .bf16) × Tensor [d.BM, d.BN] F32

/-- One trip of the `K` loop (lines 333-377). -/
def tripBody (d : Inst) (token_mask : Tensor [d.BM] Bool) (k : I32) (st : St d) : Prog (St d) :=
  (fun k (a_ptrs, b_ptrs, accumulator) => do
    let offs_k : Tensor [d.BK] I32 := tl.arange 0 d.BK
    let a ← tl.load a_ptrs
      (some ((token_mask.col d.BK).and
        ((offs_k.row d.BM).ltS (IntV.ofNat d.K - k * (IntV.ofNat d.BK : I32)))))
      (some (← tl.convert .f32 .bf16 (tl.full [d.BM, d.BK] (FloatV.litF32 (FloatV.ofPat 0)))))
    let b ← tl.load b_ptrs
      (some ((offs_k.col d.BN).ltS (IntV.ofNat d.K - k * (IntV.ofNat d.BK : I32))))
      (some (← tl.convert .f32 .bf16 (tl.full [d.BK, d.BN] (FloatV.litF32 (FloatV.ofPat 0)))))
    let accumulator ← tl.dot_add
      { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
      a b accumulator
    let a_ptrs := a_ptrs +
      (bc ((IntV.ofNat d.BK : I32) * IntV.ofNat 1) : Tensor [d.BM, d.BK] I32)
    let b_ptrs := b_ptrs +
      (bc ((IntV.ofNat d.BK : I32) * IntV.ofNat 1) : Tensor [d.BK, d.BN] I32)
    return (a_ptrs, b_ptrs, accumulator)) k st

/-- The store of a `BM × BN` tile of `bfloat16` to the rows `offs_token` of `C` (lines 417-420,
and `write_zeros_to_output`'s lines 103-106). -/
def storeC (d : Inst) (pn : I32) (offs_token : Tensor [d.BM] I64) (token_mask : Tensor [d.BM] Bool)
    (c : Tensor [d.BM, d.BN] BF16) : Prog Unit := do
  let offs_cn : Tensor [d.BN] I32 := pn * (IntV.ofNat d.BN : I32) + tl.arange 0 d.BN
  let c_ptrs : Tensor [d.BM, d.BN] (Ptr .bf16) :=
    d.c.ptr 0 + (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat (stride_cm d) : I32))).item *
        offs_token.col d.BN +
      (← tl.convert .i32 .i64 (offs_cn.row d.BM * (IntV.ofNat 1 : I32)))
  let c_mask := (token_mask.col d.BN).and ((offs_cn.row d.BM).ltS (IntV.ofNat d.N : I32))
  tl.store c_ptrs c (some c_mask)

/-- The epilogue (lines 403-420): the routing weight, the rounding to `bfloat16`, the store. -/
def tailOf (d : Inst) (pn : I32) (offs_token : Tensor [d.BM] I64) (token_mask : Tensor [d.BM] Bool)
    (accumulator : Tensor [d.BM, d.BN] F32) : Prog Unit :=
  if d.rw then
    do
      let moe_weight : Tensor [d.BM] F32 ←
        tl.load (d.w.ptr 0 + offs_token) (some token_mask)
          (some (← tl.convert .i32 .f32 (tl.full [d.BM] (IntV.ofInt 0))))
      let accumulator ← tl.mul accumulator (moe_weight.col d.BN)
      let c : Tensor [d.BM, d.BN] BF16 ← tl.cvt accumulator
      storeC d pn offs_token token_mask c
  else
    do
      let c : Tensor [d.BM, d.BN] BF16 ← tl.cvt accumulator
      storeC d pn offs_token token_mask c

/-- The main branch (lines 252-420): the expert `off_experts` is not `-1`. -/
def rest3 (d : Inst) (pn : I32) (offs_token : Tensor [d.BM] I64) (token_mask : Tensor [d.BM] Bool)
    (off_experts : I64) : Prog Unit := do
  let n64 : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.N : I32))).item
  let offs_bn : Tensor [d.BN] I64 ←
    tl.irem ((← tl.convert .i32 .i64 (Tensor.scalar (pn * (IntV.ofNat d.BN : I32)))).item +
      (← tl.convert .i32 .i64 (tl.arange 0 d.BN)))
      (bc n64)
  let offs_k : Tensor [d.BK] I32 := tl.arange 0 d.BK
  let a_ptrs : Tensor [d.BM, d.BK] (Ptr .bf16) :=
    d.a.ptr 0 + (((← tl.idiv offs_token (bc (IntV.ofNat d.top_k : I64))) *
        (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.K : I32))).item).col d.BK +
      (← tl.convert .i32 .i64 (offs_k.row d.BM * (IntV.ofNat 1 : I32))))
  let b_ptrs : Tensor [d.BK, d.BN] (Ptr .bf16) :=
    d.b.ptr 0 + off_experts * (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.stride_be : I32))).item +
      ((← tl.convert .i32 .i64 (offs_k.col d.BN * (IntV.ofNat 1 : I32))) +
        offs_bn.row d.BK *
          (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.stride_bn : I32))).item)
  let accumulator := tl.zeros [d.BM, d.BN] .f32
  let (_a_ptrs, _b_ptrs, accumulator) ← tl.forRange (IntV.ofInt 0 : I32)
      (← tl.idiv (Tensor.scalar (IntV.ofNat d.K + (IntV.ofNat (d.BK - 1) : I32)))
        (Tensor.scalar (IntV.ofNat d.BK : I32))).item
      (IntV.ofInt 1 : I32) (a_ptrs, b_ptrs, accumulator) (tripBody d token_mask)
  tailOf d pn offs_token token_mask accumulator

/-- Lines 227-250: widen the entries, the mask, the block's expert; the zeros branch for expert
`-1`, else `rest3`. -/
def restTok (d : Inst) (pm pn : I32) (tok : Tensor [d.BM] I32) : Prog Unit := do
  let offs_token : Tensor [d.BM] I64 ← tl.convert .i32 .i64 tok
  let token_mask : Tensor [d.BM] Bool :=
    offs_token.ltS (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat (nv d) : I32))).item
  let e32 : I32 ← tl.load1 (d.eid.ptr 0 + pm)
  let off_experts : I64 := (← tl.convert .i32 .i64 (Tensor.scalar e32)).item
  if IntV.eq off_experts (IntV.ofInt (-1 : Int) : I64) then
    write_zeros_to_output (d.c.ptr 0) (IntV.ofNat (stride_cm d)) (IntV.ofNat 1) pn (IntV.ofNat d.N)
      (BLOCK_SIZE_M := d.BM) (BLOCK_SIZE_N := d.BN) offs_token token_mask
    return ()
  else rest3 d pn offs_token token_mask off_experts

/-- Lines 218-226: the block's entries (loaded, or built for the naive assignment), then
`restTok`. -/
def rest2 (d : Inst) (pm pn : I32) (offs : Tensor [d.BM] I64) : Prog Unit := do
  let offs_token : Tensor [d.BM] I32 ← if !d.naive then do
      let offs_token_id : Tensor [d.BM] I64 :=
        (← tl.convert .i32 .i64 (Tensor.scalar (pm * (IntV.ofNat d.BM : I32)))).item + offs
      tl.load (d.s.ptr 0 + offs_token_id)
    else
      pure (tl.where (offs.eq (bc (IntV.ofInt 0 : I64))) (bc pm) (bc (IntV.ofNat (nv d) : I32)))
  restTok d pm pn offs_token

/-- Lines 214-217: the block offsets, `num_tokens_post_padded`, and the early return. -/
def rest1 (d : Inst) (pm pn : I32) : Prog Unit := do
  let offs : Tensor [d.BM] I64 ← tl.convert .i32 .i64 (tl.arange 0 d.BM)
  let num_tokens_post_padded : I32 ← tl.load1 (d.ntpb.ptr 0)
  if IntV.le num_tokens_post_padded (pm * (IntV.ofNat d.BM : I32)) then
    return ()
  else rest2 d pm pn offs

/-- `write_zeros_to_output` is the store of a tile of `bfloat16` zeros (`storeC`). -/
theorem zero_eq (d : Inst) (pn : I32) (offs_token : Tensor [d.BM] I64)
    (token_mask : Tensor [d.BM] Bool) :
    write_zeros_to_output (d.c.ptr 0) (IntV.ofNat (stride_cm d)) (IntV.ofNat 1) pn (IntV.ofNat d.N)
      (BLOCK_SIZE_M := d.BM) (BLOCK_SIZE_N := d.BN) offs_token token_mask =
    storeC d pn offs_token token_mask (tl.zeros [d.BM, d.BN] .bf16) := rfl

/-- The model's body is the opening `pidPrefix` followed by `rest1`. -/
theorem body_eq (d : Inst) (ctx : Ctx) : (model d).body ctx = pidPrefix d ctx (rest1 d) := rfl

end Claims.FusedMoe
