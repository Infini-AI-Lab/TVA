import Claim
import TVA

/-!
# The kernel body, cut into named pieces

The model's text, cut where a proof needs a name: `elseProg` (after the early exit), `afterCm`
(offsets, `mask_m`, the load of `q`), `afterQ` (the cached-part loop), `afterPref` (the new-token loop),
`epilogue` (the store) and the loop bodies `prefBody`, `extBody`. The scalars computed before the
early exit are the fields of `Sc`. `body_eq` is `rfl`: walking a piece is a lemma about a name
rather than a copy of the model.
-/

namespace Claims.ExtendAttn

open TVA

/-- The scalars the kernel computes before its early exit (lines 195-207), in the model's types:
the grid axes, `cur_head // kv_group_num`, `qo_indptr[cur_seq]` and the number of new tokens,
`kv_indptr[cur_seq]` and the cached-part length, and their sum. -/
structure Sc (d : Inst) where
  cur_seq : I32
  cur_head : I32
  cur_block_m : I32
  cur_kv_head : I32
  cur_seq_extend_start_idx : QI d.cm
  cur_seq_len_extend : QI d.cm
  cur_seq_kv_start_idx : I32
  cur_seq_len_prefix : I32
  cur_seq_len : QI d.cm

/-- What the loops carry: the accumulator, `deno` and `e_max`. -/
abbrev LoopSt (d : Inst) : Type :=
  Tensor [d.BM, d.BD] F32 × Tensor [d.BM] F32 × Tensor [d.BM] F32

/-- `mask_n` of the cached-part loop (line 270): the trip's keys inside the cached-part. -/
def maskNP (d : Inst) (s : Sc d) (start_n : I32) : Tensor [BN] Bool :=
  (start_n + tl.arange 0 BN).ltS s.cur_seq_len_prefix

/-- `final_mask` of the cached-part loop (lines 272, 285-289): the block's rows, the trip's keys inside
the cached-part, and with a window the keys within it. -/
def finalMaskP (d : Inst) (s : Sc d) (mask_m : Tensor [d.BM] Bool) (start_n : I32) :
    Tensor [d.BM, BN] Bool :=
  let final_mask : Tensor [d.BM, BN] Bool := (mask_m.col BN).and ((maskNP d s start_n).row d.BM)
  if 0 < d.sw then
    final_mask.and (Tensor.le
      (s.cur_seq_len_prefix + s.cur_block_m * (IntV.ofNat d.BM : I32) +
        (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)
      (start_n + (tl.arange 0 BN : Tensor [BN] I32).row d.BM + (IntV.ofNat d.sw : I32)))
  else final_mask

/-- The epilogue (lines 607-638): the divisor `deno` (`1.0` when it is `0`), `acc / deno`, and the
`bfloat16` store of the tile. -/
def epilogue (d : Inst) (s : Sc d) (mask_m : Tensor [d.BM] Bool)
    (acc : Tensor [d.BM, d.BD] F32) (deno : Tensor [d.BM] F32) : Prog Unit := do
  let offs_dv : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let offs_m : Tensor [d.BM] I32 := tl.arange 0 d.BM
  let mask_dv : Tensor [d.BD] Bool := offs_dv.ltS (IntV.ofNat d.D : I32)
  let no_kv := deno.feq (bc zf32)
  let offs_o : Tensor [d.BM, d.BD] (QI d.cm) :=
    ((s.cur_seq_extend_start_idx +
          (← toQ d.cm (Tensor.scalar (s.cur_block_m * (IntV.ofNat d.BM : I32)))).item) +
        (← toQ d.cm (offs_m.col d.BD))) *
        (← toQ d.cm (Tensor.scalar (IntV.ofNat d.sot : I32))).item +
      (← toQ d.cm (Tensor.scalar (s.cur_head * (IntV.ofNat d.soh : I32)))).item +
      (← toQ d.cm (offs_dv.row d.BM))
  let deno_safe := tl.where no_kv (bc one32) deno
  let o ← tl.div acc (deno_safe.col d.BD)
  let o16 : Tensor [d.BM, d.BD] BF16 ← tl.cvt o
  tl.store ((d.o.ptr 0) + offs_o) o16 (some ((mask_m.col d.BD).and (mask_dv.row d.BM)))

/-- One trip of the cached-part loop (lines 270-440). -/
def prefBody (d : Inst) (s : Sc d) (mask_m : Tensor [d.BM] Bool) (q : Tensor [d.BM, d.BD] BF16)
    (start_n : I32) (st : LoopSt d) : Prog (LoopSt d) := do
  let offs_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let offs_dv : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let mask_d : Tensor [d.BD] Bool := offs_d.ltS (IntV.ofNat d.D : I32)
  let mask_dv : Tensor [d.BD] Bool := offs_dv.ltS (IntV.ofNat d.D : I32)
  let offs_n_prefix : Tensor [BN] I32 := tl.arange 0 BN
  let (acc, deno, e_max) := st
  let mask_n : Tensor [BN] Bool := maskNP d s start_n
  let final_mask : Tensor [d.BM, BN] Bool := finalMaskP d s mask_m start_n
  let SKIP_TILE : Tensor [] Bool ←
    if 0 < d.sw then do
      let fm : Tensor [d.BM, BN] I32 ← tl.convert .i1 .i32 final_mask
      pure (Tensor.scalar (IntV.eq
        (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 fm))
        (IntV.ofInt 0)))
    else pure (Tensor.scalar false)
  if (Tensor.not SKIP_TILE).item then do
    let offs_kv_loc : Tensor [BN] I64 ←
      tl.load ((d.kvi.ptr 0) + s.cur_seq_kv_start_idx + start_n + offs_n_prefix) (some mask_n)
        (some (← tl.convert .i32 .i64 (tl.full [BN] (IntV.ofInt 0 : I32))))
    let offs_buf_k : Tensor [d.BD, BN] I64 :=
      offs_kv_loc.row d.BD *
          (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.sbkt : I32))).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.sbkh : I32)))).item +
        (← tl.convert .i32 .i64 (offs_d.col BN))
    let k : Tensor [d.BD, BN] BF16 ←
      tl.load ((d.kb.ptr 0) + offs_buf_k)
        (some ((mask_n.row d.BD).and (mask_d.col BN)))
        (some (← tl.convert .f32 .bf16 (tl.full [d.BD, BN] zf32)))
    let q16 : Tensor [d.BM, d.BD] BF16 ← tl.cvt q
    let qk : Tensor [d.BM, BN] F32 ←
      tl.dot cfgBF16 q16 k (tl.zeros [d.BM, BN] .f32)
    let qk ← tl.mul qk (bc (← tl.mul (Tensor.scalar (FloatV.hostF32 d.sm_scale)) (Tensor.scalar (FloatV.hostF32 one64))).item)
    let qk := tl.where final_mask qk (bc ninf32)
    let row_max : Tensor [d.BM] F32 ← tl.max1 true qk
    let row_max_fixed := tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max
    let n_e_max : Tensor [d.BM] F32 ← tl.maximum row_max_fixed e_max
    let re_scale : Tensor [d.BM] F32 ← tl.exp (← tl.sub e_max n_e_max)
    let p : Tensor [d.BM, BN] F32 ←
      tl.exp (← tl.sub qk (n_e_max.col BN))
    let deno ← tl.mul_add deno re_scale (← tl.sum1 p)
    let offs_buf_v : Tensor [BN, d.BD] I64 :=
      offs_kv_loc.col d.BD *
          (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.sbvt : I32))).item +
        (← tl.convert .i32 .i64 (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.sbvh : I32)))).item +
        (← tl.convert .i32 .i64 (offs_dv.row BN))
    let v : Tensor [BN, d.BD] BF16 ←
      tl.load ((d.vb.ptr 0) + offs_buf_v)
        (some ((mask_n.col d.BD).and (mask_dv.row BN)))
        (some (← tl.convert .f32 .bf16 (tl.full [BN, d.BD] zf32)))
    let p16 : Tensor [d.BM, BN] BF16 ← tl.cvt p
    let dot : Tensor [d.BM, d.BD] F32 ←
      tl.dot cfgBF16 p16 v (tl.zeros [d.BM, d.BD] .f32)
    let acc ← tl.addc (← tl.mul acc (re_scale.col d.BD)) (← tl.mul dot (bc (FloatV.hostF32 one64)))
    let e_max := n_e_max
    pure (acc, deno, e_max)
  else
    pure (acc, deno, e_max)

/-- One trip of the new-token loop (lines 461-596), for the block's end `cur_block_m_end` and the
mask data `cm`. -/
def extBody (d : Inst) (s : Sc d) (cm : Option (I64 × Option I64)) (mask_m : Tensor [d.BM] Bool)
    (q : Tensor [d.BM, d.BD] BF16) (cur_block_m_end : QI d.cm) (start_n : QI d.cm) (st : LoopSt d) :
    Prog (LoopSt d) := do
  let offs_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let offs_dv : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let offs_m : Tensor [d.BM] I32 := tl.arange 0 d.BM
  let mask_d : Tensor [d.BD] Bool := offs_d.ltS (IntV.ofNat d.D : I32)
  let mask_dv : Tensor [d.BD] Bool := offs_dv.ltS (IntV.ofNat d.D : I32)
  let offs_n : Tensor [BN] I32 := tl.arange 0 BN
  let (acc, deno, e_max) := st
  let mask_n : Tensor [BN] Bool :=
    (start_n + (← toQ d.cm offs_n)).ltS cur_block_m_end
  let final_mask : Tensor [d.BM, BN] Bool :=
    (mask_m.col BN).and (mask_n.row d.BM)
  let final_mask : Tensor [d.BM, BN] Bool ← match cm with
    | some (cur_seq_mask_start_idx, window_kv_offset) => do
      let ptrs : Tensor [d.BM, BN] (Ptr .i1) ← match window_kv_offset with
        | none => do
          let wko : I32 := IntV.ofInt 0
          let row_len : QI d.cm := s.cur_seq_len + (← toQ d.cm (Tensor.scalar wko)).item
          let row_off : Tensor [d.BM, BN] (QI d.cm) :=
            (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN)) *
              row_len
          pure ((d.cmask.ptr 0) + cur_seq_mask_start_idx + row_off +
            (bc wko : Tensor [d.BM, BN] _) +
            (bc s.cur_seq_len_prefix : Tensor [d.BM, BN] _) +
            (bc start_n : Tensor [d.BM, BN] _) + offs_n.row d.BM)
        | some wko => do
          let row_len : I64 := (← qToI64 d.cm (Tensor.scalar s.cur_seq_len)).item + wko
          let row_off : Tensor [d.BM, BN] I64 :=
            (← tl.convert .i32 .i64
              (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN)) * row_len
          pure ((d.cmask.ptr 0) + cur_seq_mask_start_idx + row_off +
            (bc wko : Tensor [d.BM, BN] _) +
            (bc s.cur_seq_len_prefix : Tensor [d.BM, BN] _) +
            (bc start_n : Tensor [d.BM, BN] _) + offs_n.row d.BM)
      let custom_mask : Tensor [d.BM, BN] Bool ←
        tl.load ptrs (some ((mask_m.col BN).and (mask_n.row d.BM)))
          (some (← tl.convert .i32 .i1 (tl.full [d.BM, BN] (IntV.ofInt 0 : I32))))
      let cm32 : Tensor [d.BM, BN] I32 ← tl.convert .i1 .i32 custom_mask
      let custom_mask : Tensor [d.BM, BN] Bool :=
        ((cm32.map (fun x => !(IntV.eq x (IntV.ofInt 0)))).and (mask_m.col BN)).and
          (mask_n.row d.BM)
      pure (final_mask.and custom_mask)
    | none => do
      let mask_causual : Tensor [d.BM, BN] Bool :=
        Tensor.le (start_n + (← toQ d.cm (offs_n.row d.BM)))
          (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN))
      let mask_causual := mask_causual.and ((mask_m.col BN).and (mask_n.row d.BM))
      pure (final_mask.and mask_causual)
  let final_mask : Tensor [d.BM, BN] Bool ←
    if 0 < d.sw then do
      let window_mask : Tensor [d.BM, BN] Bool :=
        Tensor.le
          (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN))
          (start_n + (← toQ d.cm (offs_n.row d.BM)) +
            (IntV.ofNat d.sw : QI d.cm))
      pure (final_mask.and window_mask)
    else pure final_mask
  let SKIP_TILE : Tensor [] Bool ←
    if d.cm || decide (0 < d.sw) then do
      let fm : Tensor [d.BM, BN] I32 ← tl.convert .i1 .i32 final_mask
      pure (Tensor.scalar (IntV.eq
        (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 fm))
        (IntV.ofInt 0)))
    else pure (Tensor.scalar false)
  if (Tensor.not SKIP_TILE).item then do
    let offs_k : Tensor [d.BD, BN] (QI d.cm) :=
      ((s.cur_seq_extend_start_idx + start_n) +
            (← toQ d.cm (offs_n.row d.BD))) *
          (← toQ d.cm (Tensor.scalar (IntV.ofNat d.skt : I32))).item +
        (← toQ d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.skh : I32)))).item +
        (← toQ d.cm (offs_d.col BN))
    let k : Tensor [d.BD, BN] BF16 ←
      tl.load ((d.k.ptr 0) + offs_k) (some ((mask_n.row d.BD).and (mask_d.col BN)))
        (some (← tl.convert .f32 .bf16 (tl.full [d.BD, BN] zf32)))
    let qk : Tensor [d.BM, BN] F32 ←
      tl.dot cfgBF16 q k (tl.zeros [d.BM, BN] .f32)
    let qk ← tl.mul qk (bc (FloatV.hostF32 d.sm_scale))
    let qk := tl.where final_mask qk (bc ninf32)
    let row_max : Tensor [d.BM] F32 ← tl.max1 true qk
    let row_max_fixed := tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max
    let n_e_max : Tensor [d.BM] F32 ← tl.maximum row_max_fixed e_max
    let re_scale : Tensor [d.BM] F32 ← tl.exp (← tl.sub e_max n_e_max)
    let p : Tensor [d.BM, BN] F32 ← tl.exp (← tl.sub qk (n_e_max.col BN))
    let deno ← tl.mul_add deno re_scale (← tl.sum1 p)
    let offs_v : Tensor [BN, d.BD] (QI d.cm) :=
      ((s.cur_seq_extend_start_idx + start_n) +
            (← toQ d.cm (offs_n.col d.BD))) *
          (← toQ d.cm (Tensor.scalar (IntV.ofNat d.svt : I32))).item +
        (← toQ d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.svh : I32)))).item +
        (← toQ d.cm (offs_dv.row BN))
    let v : Tensor [BN, d.BD] BF16 ←
      tl.load ((d.v.ptr 0) + offs_v) (some ((mask_n.col d.BD).and (mask_dv.row BN)))
        (some (← tl.convert .f32 .bf16 (tl.full [BN, d.BD] zf32)))
    let p16 : Tensor [d.BM, BN] BF16 ← tl.cvt p
    let acc ← tl.dot_add cfgBF16 p16 v (← tl.mul acc (re_scale.col d.BD))
    let e_max := n_e_max
    pure (acc, deno, e_max)
  else
    pure (acc, deno, e_max)

/-- Everything after the cached-part loop: the new-token loop, then the epilogue. -/
def afterPref (d : Inst) (s : Sc d) (cm : Option (I64 × Option I64)) (mask_m : Tensor [d.BM] Bool)
    (q : Tensor [d.BM, d.BD] BF16) (st : LoopSt d) : Prog Unit := do
  let (acc, deno, e_max) := st
  let cur_block_m_end : QI d.cm :=
    (tl.iminimum (Tensor.scalar s.cur_seq_len_extend)
      (← toQ d.cm (Tensor.scalar ((s.cur_block_m + (IntV.ofInt 1 : I32)) *
        (IntV.ofNat d.BM : I32))))).item
  let extend_end := cur_block_m_end
  let extend_start : I32 ←
    if 0 < d.sw then do
      let lo : Tensor [] I32 :=
        tl.imaximum (Tensor.scalar (s.cur_block_m * (IntV.ofNat d.BM : I32) -
          (IntV.ofNat d.sw : I32))) (Tensor.scalar (IntV.ofInt 0 : I32))
      let lo ← tl.idiv lo (Tensor.scalar (IntV.ofNat BN : I32))
      pure (lo.item * (IntV.ofNat BN : I32))
    else pure (IntV.ofInt 0)
  let (acc, deno, _) ← tl.forRange
      (← toQ d.cm (Tensor.scalar extend_start)).item extend_end
      (IntV.ofNat BN : QI d.cm) (acc, deno, e_max)
    fun start_n st => extBody d s cm mask_m q cur_block_m_end start_n st
  epilogue d s mask_m acc deno

/-- Everything after the load of `q`: the cached-part loop, then `afterPref`. -/
def afterQ (d : Inst) (s : Sc d) (cm : Option (I64 × Option I64)) (mask_m : Tensor [d.BM] Bool)
    (q : Tensor [d.BM, d.BD] BF16) : Prog Unit := do
  let acc := tl.zeros [d.BM, d.BD] .f32
  let deno := tl.zeros [d.BM] .f32
  let e_max : Tensor [d.BM] F32 ← tl.sub (tl.zeros [d.BM] .f32) (bc pinf32)
  let st ← tl.forRange (IntV.ofInt 0 : I32) s.cur_seq_len_prefix
      (IntV.ofNat BN : I32) (acc, deno, e_max)
    fun start_n st => prefBody d s mask_m q start_n st
  afterPref d s cm mask_m q st

/-- Everything after the mask data: the offsets, the load of `q`, then `afterQ`. -/
def afterCm (d : Inst) (s : Sc d) (cm : Option (I64 × Option I64)) : Prog Unit := do
  let offs_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let offs_m : Tensor [d.BM] I32 := tl.arange 0 d.BM
  let mask_m : Tensor [d.BM] Bool :=
    (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m)).ltS s.cur_seq_len_extend
  let mask_d : Tensor [d.BD] Bool := offs_d.ltS (IntV.ofNat d.D : I32)
  let offs_q : Tensor [d.BM, d.BD] (QI d.cm) :=
    ((s.cur_seq_extend_start_idx +
          (← toQ d.cm (Tensor.scalar (s.cur_block_m * (IntV.ofNat d.BM : I32)))).item) +
        (← toQ d.cm (offs_m.col d.BD))) *
        (← toQ d.cm (Tensor.scalar (IntV.ofNat d.sqt : I32))).item +
      (← toQ d.cm (Tensor.scalar (s.cur_head * (IntV.ofNat d.sqh : I32)))).item +
      (← toQ d.cm (offs_d.row d.BM))
  let q : Tensor [d.BM, d.BD] BF16 ←
    tl.load ((d.q.ptr 0) + offs_q) (some ((mask_m.col d.BD).and (mask_d.row d.BM)))
      (some (← tl.convert .f32 .bf16 (tl.full [d.BM, d.BD] zf32)))
  afterQ d s cm mask_m q

/-- The program after the early exit: the mask data, then `afterCm`. -/
def elseProg (d : Inst) (s : Sc d) : Prog Unit := do
  let cm : Option (I64 × Option I64) ←
    if d.cm then do
      let cur_seq_mask_start_idx : I64 ← tl.load1 ((d.mip.ptr 0) + s.cur_seq)
      let window_kv_offset : Option I64 ←
        if 0 < d.sw then do
          let w : I64 ← tl.load1 ((d.wko.ptr 0) + s.cur_seq)
          pure (some w)
        else pure none
      pure (some (cur_seq_mask_start_idx, window_kv_offset))
    else pure none
  afterCm d s cm

/-- The model's body is the preamble, the early exit, and `elseProg`. -/
theorem body_eq (d : Inst) (ctx : Ctx) :
    (model d).body ctx = (do
      let cur_seq := tl.program_id ctx 0
      let cur_head := tl.program_id ctx 1
      let cur_block_m := tl.program_id ctx 2
      let cur_kv_head := (← tl.idiv (Tensor.scalar cur_head) (Tensor.scalar (IntV.ofNat (kvg d) : I32))).item
      let cur_seq_extend_start_idx : QI d.cm ← loadQo d.cm ((d.qo.ptr 0) + cur_seq)
      let cur_seq_len_extend : QI d.cm :=
        (← loadQo d.cm ((d.qo.ptr 0) + cur_seq + (IntV.ofInt 1 : I32))) - cur_seq_extend_start_idx
      let cur_seq_kv_start_idx : I32 ← tl.load1 ((d.kvp.ptr 0) + cur_seq)
      let cur_seq_len_prefix : I32 :=
        (← tl.load1 ((d.kvp.ptr 0) + cur_seq + (IntV.ofInt 1 : I32))) - cur_seq_kv_start_idx
      let cur_seq_len : QI d.cm :=
        (← toQ d.cm (Tensor.scalar cur_seq_len_prefix)).item + cur_seq_len_extend
      if IntV.le cur_seq_len_extend
          (← toQ d.cm (Tensor.scalar (cur_block_m * (IntV.ofNat d.BM : I32)))).item then
        return ()
      else
        elseProg d ⟨cur_seq, cur_head, cur_block_m, cur_kv_head, cur_seq_extend_start_idx,
          cur_seq_len_extend, cur_seq_kv_start_idx, cur_seq_len_prefix, cur_seq_len⟩) := rfl

end Claims.ExtendAttn
