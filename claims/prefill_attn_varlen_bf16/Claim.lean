import TVA

/-!
# `prefill_attn_varlen_bf16`: the prepared claim

Source: `source/prefill_attention.py`. The kernel is `_fwd_kernel`
(lines 51–184); it reads the module-level names `triton` and `tl` (lines 47–48). The host
`context_attention_fwd` (lines 187–243) launches it on `bfloat16` `q` (`[total_tokens, num_heads,
head_dim]`), `k`, `v` (`[total_tokens, num_kv_heads, head_dim]`), the `int32` `b_start_loc`,
`b_seq_len` (`[batch]`), and writes `o = torch.empty_like(q)`, on the grid
`(batch, num_heads, cdiv(max_input_len, 128))`, with `BLOCK_M = BLOCK_N = 128` (an H200,
capability 9 > 8), `BLOCK_DMODEL = next_power_of_2(Lk)` and the caller's `IS_CAUSAL`, `True` or
`False` (`SETUP.md`).

* The model: `fwd_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the keys a query attends (`j ≤ lastKey`), the
  reference `attnRef`, its scale `attnAbs`, the row's score bound `Xrow` and the tolerances
  `tolRel`, `tolAbs`.
* The assumptions: `Assumptions`.
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `setup_*_ok`.
-/

namespace Claims.PrefillAttnVarlen

open TVA TVA.Math.Float

/-! ## Constants of the source -/

/-- The `float32` `+0` (`other=0.0`, the `tl.where` value `0` beside `float("-inf")`). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- The `float32` `+∞` (`float("inf")` of line 110, in an arithmetic operation on `float32`). -/
def pinf32 : F32 := FloatV.litF32 (FloatV.ofPat 0x7FF0000000000000)

/-- The `float32` `−∞` (`float("-inf")` of line 140, a `tl.where` value). -/
def ninf32 : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)

/-- `BLOCK = 128`, the wrapper's choice on an H200 (line 204): `BLOCK_M` and `BLOCK_N`. -/
def BLOCK : Nat := 128

/-- The `tl.dot` configuration of lines 132 and 171: `bfloat16` operands, the default `float32`
accumulator, the default `input_precision` (`tf32`; no effect on 16-bit operands) and the default
`max_num_imprecise_acc` (`0`, not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## The model -/

/-- `_fwd_kernel` (source lines 51–184). `Q`, `K`, `V`, `Out` are `*bf16`, `B_Start_Loc`,
`B_Seqlen` `*i32`; `sm_scale` is the runtime `float32` scalar (a Python `float` argument:
`FloatV.hostF32` of the double); the eight strides are `int32` arguments (one equal to `1` is the
compile-time constant `1`, which each use, an operand of `*` with an `int32`, builds in `int32`:
the same model); `kv_group_num`, `BLOCK_M`, `BLOCK_DMODEL`, `BLOCK_N`, `Lk` are compile-time
values, and so is `IS_CAUSAL` (a `bool`): the compile-time conditional expression of lines
116–120 and the compile-time `if` of lines 135–145 visit only the side it picks. -/
def fwd_kernel (Q K V Out : Ptr .bf16) (sm_scale : F32)
    (B_Start_Loc B_Seqlen : Ptr .i32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh : I32)
    (kv_group_num BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_CAUSAL : Bool) (Lk : Nat) : Kernel where
  body ctx := do
    -- line 75: axis 0 of the grid, an `int32`
    let cur_batch := tl.program_id ctx 0
    -- line 76: axis 1
    let cur_head := tl.program_id ctx 1
    -- line 77: axis 2
    let start_m := tl.program_id ctx 2
    -- line 79: `int32` `//` by the constexpr built in `int32` (truncating; the divisor is at
    -- least `1` on a covered instance)
    let cur_kv_head :=
      (← tl.idiv (Tensor.scalar cur_head) (Tensor.scalar (IntV.ofNat kv_group_num : I32))).item
    -- line 81: the `*i32` pointer advanced by `cur_batch`; one `int32` element
    let cur_batch_seq_len ← tl.load1 (B_Seqlen + cur_batch)
    -- line 82
    let cur_batch_in_all_start_index ← tl.load1 (B_Start_Loc + cur_batch)
    -- line 84: the constexpr `BLOCK_M` built in `int32`, times `start_m` (a wrapping product)
    let block_start_loc := (IntV.ofNat BLOCK_M : I32) * start_m
    -- lines 87–89
    let offs_n : Tensor [BLOCK_N] I32 := tl.arange 0 BLOCK_N
    let offs_d : Tensor [BLOCK_DMODEL] I32 := tl.arange 0 BLOCK_DMODEL
    let offs_m : Tensor [BLOCK_M] I32 :=
      start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    -- lines 90–94: the `int32` products and sums in the source's order, each term broadcast to
    -- the final shape (`x[:, None]` is `.col`, `x[None, :]` is `.row`, a scalar `bc`)
    let off_q : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_qbs +
        bc (cur_head * stride_qh) + offs_d.row BLOCK_M
    -- lines 95–96
    let off_k : Tensor [BLOCK_DMODEL, BLOCK_N] I32 :=
      offs_n.row BLOCK_DMODEL * stride_kbs +
        bc (cur_kv_head * stride_kh) + offs_d.col BLOCK_N
    let off_v : Tensor [BLOCK_N, BLOCK_DMODEL] I32 :=
      offs_n.col BLOCK_DMODEL * stride_vbs +
        bc (cur_kv_head * stride_vh) + offs_d.row BLOCK_N
    -- line 98: the constexpr `Lk` built in `int32`, signed comparison
    let mask_d : Tensor [BLOCK_DMODEL] Bool := offs_d.ltS (IntV.ofNat Lk : I32)
    -- lines 100–104: `other=0.0` is the `float32` `+0`, converted to the pointer's dtype
    let q : Tensor [BLOCK_M, BLOCK_DMODEL] BF16 ←
      tl.load (Q + off_q)
        (some (((offs_m.col BLOCK_DMODEL).ltS cur_batch_seq_len).and (mask_d.row BLOCK_M)))
        (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_M, BLOCK_DMODEL] zf32)))
    -- lines 106–107
    let k_ptrs := K + off_k
    let v_ptrs := V + off_v
    -- line 110: `tl.zeros(...) - float("inf")`: the Python number built in `float32`, IEEE `-`
    let m_i : Tensor [BLOCK_M] F32 ← tl.sub (tl.zeros [BLOCK_M] .f32) (bc pinf32)
    -- line 111
    let l_i := tl.zeros [BLOCK_M] .f32
    -- line 112
    let acc := tl.zeros [BLOCK_M, BLOCK_DMODEL] .f32
    -- line 114: the Python `1` and `0` built in `int32`; a scalar `int32` tensor
    let block_mask : Tensor [] I32 :=
      tl.where (Tensor.lt (Tensor.scalar block_start_loc) (Tensor.scalar cur_batch_seq_len))
        (bc (IntV.ofInt 1 : I32)) (bc (IntV.ofInt 0 : I32))
    -- lines 116–120: a conditional expression on the constexpr `not IS_CAUSAL`: only the side it
    -- picks is evaluated: `cur_batch_seq_len`, or `tl.minimum((start_m + 1) * BLOCK_M,
    -- cur_batch_seq_len)`, an exact `int32` minimum
    let end_n : I32 :=
      if !IS_CAUSAL then cur_batch_seq_len
      else
        (tl.iminimum (Tensor.scalar ((start_m + (IntV.ofInt 1 : I32)) * (IntV.ofNat BLOCK_M : I32)))
          (Tensor.scalar cur_batch_seq_len)).item
    -- line 121: `range(0, block_mask * end_n, BLOCK_N)`, the bound the `int32` product,
    -- evaluated once; the loop carries `m_i`, `l_i`, `acc`
    let (_, _, acc) ← tl.forRange (IntV.ofInt 0 : I32)
      ((block_mask * Tensor.scalar end_n).item) (IntV.ofNat BLOCK_N : I32) (m_i, l_i, acc)
      fun start_n (m_i, l_i, acc) => do
        -- line 122: a hint on the loop's induction variable: no effect (not modelled)
        -- lines 124–128: the scalar `int32` offset broadcast; `other=0.0` converted
        let k : Tensor [BLOCK_DMODEL, BLOCK_N] BF16 ←
          tl.load (k_ptrs + (bc ((cur_batch_in_all_start_index + start_n) *
              stride_kbs) : Tensor [BLOCK_DMODEL, BLOCK_N] I32))
            (some ((((start_n + offs_n).row BLOCK_DMODEL).ltS cur_batch_seq_len).and
              (mask_d.col BLOCK_N)))
            (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_DMODEL, BLOCK_N] zf32)))
        -- line 131
        let qk₀ := tl.zeros [BLOCK_M, BLOCK_N] .f32
        -- line 132: `qk += tl.dot(q, k)`: a zero default accumulator, the product used once:
        -- the combine pass's `tl.dot_add`
        let qk₁ : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.dot_add cfgBF16 q k qk₀
        -- lines 133–145: `qk *= sm_scale` feeds `qk += tl.where(cond, 0, float("-inf"))` of the
        -- branch the compile-time `if IS_CAUSAL:` (line 135) visits: one fusable multiply-add;
        -- the `tl.where` values built in `float32`. Lines 136–141: the key mask and the causal
        -- mask; lines 143–145: the key mask alone, a `[1, BLOCK_N]` select that the `+=`
        -- broadcasts to `[BLOCK_M, BLOCK_N]` (written at the broadcast shape: the same values)
        let qk : Tensor [BLOCK_M, BLOCK_N] F32 ←
          tl.mul_add qk₁ (bc sm_scale)
            (if IS_CAUSAL then
              tl.where
                ((((start_n + offs_n).row BLOCK_M).ltS cur_batch_seq_len).and
                  (Tensor.le ((start_n + offs_n).row BLOCK_M) (offs_m.col BLOCK_N)))
                (bc zf32) (bc ninf32)
            else
              tl.where (((start_n + offs_n).row BLOCK_M).ltS cur_batch_seq_len)
                (bc zf32) (bc ninf32))
        -- line 148: the row maximum
        let m_ij : Tensor [BLOCK_M] F32 ← tl.max1 true qk
        -- line 149: the IEEE `-` (no product feeds it), then `tl.exp`
        let p : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.exp (← tl.sub qk (m_ij.col BLOCK_N))
        -- line 150: `p` is not a product: the IEEE `tl.sum1`
        let l_ij : Tensor [BLOCK_M] F32 ← tl.sum1 p
        -- line 152
        let m_i_new : Tensor [BLOCK_M] F32 ← tl.maximum m_i m_ij
        -- line 153: the IEEE `-`, then `tl.exp`
        let alpha : Tensor [BLOCK_M] F32 ← tl.exp (← tl.sub m_i m_i_new)
        -- line 154
        let beta : Tensor [BLOCK_M] F32 ← tl.exp (← tl.sub m_ij m_i_new)
        -- line 155: `alpha * l_i + beta * l_ij`: a product on each side: `tl.addc`
        let l_i_new : Tensor [BLOCK_M] F32 ←
          tl.addc (← tl.mul alpha l_i) (← tl.mul beta l_ij)
        -- line 158: the platform's `float32` division
        let p_scale : Tensor [BLOCK_M] F32 ← tl.div beta l_i_new
        -- line 159: the IEEE `*`
        let p₁ : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.mul p (p_scale.col BLOCK_N)
        -- line 161: `(l_i / l_i_new) * alpha`
        let acc_scale : Tensor [BLOCK_M] F32 ← tl.mul (← tl.div l_i l_i_new) alpha
        -- line 162: the IEEE `*`
        let acc₁ : Tensor [BLOCK_M, BLOCK_DMODEL] F32 ← tl.mul acc (acc_scale.col BLOCK_DMODEL)
        -- lines 164–168
        let v : Tensor [BLOCK_N, BLOCK_DMODEL] BF16 ←
          tl.load (v_ptrs + (bc ((cur_batch_in_all_start_index + start_n) *
              stride_vbs) : Tensor [BLOCK_N, BLOCK_DMODEL] I32))
            (some ((((start_n + offs_n).col BLOCK_DMODEL).ltS cur_batch_seq_len).and
              (mask_d.row BLOCK_N)))
            (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_N, BLOCK_DMODEL] zf32)))
        -- line 170: `p.to(v.dtype)`, one rounding to nearest even
        let p₂ : Tensor [BLOCK_M, BLOCK_N] BF16 ← tl.cvt p₁
        -- line 171: `acc += tl.dot(p, v)`: `tl.dot_add`, the accumulator `acc₁`
        let acc₂ : Tensor [BLOCK_M, BLOCK_DMODEL] F32 ← tl.dot_add cfgBF16 p₂ v acc₁
        -- lines 173–174
        pure (m_i_new, l_i_new, acc₂)
    -- lines 176–180
    let off_o : Tensor [BLOCK_M, BLOCK_DMODEL] I32 :=
      (cur_batch_in_all_start_index + offs_m.col BLOCK_DMODEL) * stride_obs +
        bc (cur_head * stride_oh) + offs_d.row BLOCK_M
    -- line 181
    let out_ptrs := Out + off_o
    -- lines 182–184: the store's conversion to the pointer's `bfloat16` (nearest even)
    let o16 : Tensor [BLOCK_M, BLOCK_DMODEL] BF16 ← tl.cvt acc
    tl.store out_ptrs o16
      (some (((offs_m.col BLOCK_DMODEL).ltS cur_batch_seq_len).and (mask_d.row BLOCK_M)))

/-! ## The conditions -/

/-- What the host chooses: one launch of `_fwd_kernel` through `context_attention_fwd(q, k, v,
b_start_loc, b_seq_len, max_input_len, is_causal, sm_scale)`. Each tensor is its storage
(`Buf`) and its own strides in elements, the last dimension contiguous. -/
structure Inst where
  /-- `b_seq_len.shape[0]`: the grid's `x`. -/
  batch : Nat
  /-- `q.shape[0]` (`= k.shape[0] = v.shape[0]`): `total_tokens`. -/
  total : Nat
  /-- `q.shape[1]`: `num_heads`, the grid's `y`. -/
  H : Nat
  /-- `k.shape[1]` (`= v.shape[1]`): `num_kv_heads`. -/
  HKV : Nat
  /-- `Lk = k.shape[-1]` (`= q.shape[-1] = v.shape[-1]`): the head dimension. -/
  Lk : Nat
  /-- `BLOCK_DMODEL = triton.next_power_of_2(Lk)`. -/
  BD : Nat
  /-- `max_input_len`. -/
  maxLen : Nat
  /-- `sm_scale`, the Python `float` (a double; by default `1/sqrt(Lq)`, line 211). -/
  sm_scale : F64
  /-- `is_causal`, passed as the constexpr `IS_CAUSAL` (line 238): `True` (the wrapper's default)
  or `False` (the vision encoder's call). -/
  causal : Bool
  /-- `q.stride(0)`, `q.stride(1)`. -/
  sqt : Nat
  sqh : Nat
  /-- `k.stride(0)`, `k.stride(1)`. -/
  skt : Nat
  skh : Nat
  /-- `v.stride(0)`, `v.stride(1)`. -/
  svt : Nat
  svh : Nat
  /-- `o.stride(0)`, `o.stride(1)`. -/
  sot : Nat
  soh : Nat
  /-- The storages of `q`, `k`, `v`, `o`. -/
  q : Buf .bf16
  k : Buf .bf16
  v : Buf .bf16
  o : Buf .bf16
  /-- The storages of `b_start_loc`, `b_seq_len`. -/
  b_start_loc : Buf .i32
  b_seq_len : Buf .i32

/-- Element `[t, h, e]` of `q`, by its own strides. -/
def qIdx (d : Inst) (t h e : Nat) : Nat := t * d.sqt + h * d.sqh + e
/-- Element `[t, g, e]` of `k`. -/
def kIdx (d : Inst) (t g e : Nat) : Nat := t * d.skt + g * d.skh + e
/-- Element `[t, g, e]` of `v`. -/
def vIdx (d : Inst) (t g e : Nat) : Nat := t * d.svt + g * d.svh + e
/-- Element `[t, h, e]` of `o`. -/
def oIdx (d : Inst) (t h e : Nat) : Nat := t * d.sot + h * d.soh + e

/-- `kv_group_num = num_heads // num_kv_heads` (line 213). -/
def kvg (d : Inst) : Nat := d.H / d.HKV

/-- The key/value head of query head `h`: `h // kv_group_num` (the grouped-query attention of
the host's reference, `repeat_interleave(group)`). -/
def kvh (d : Inst) (h : Nat) : Nat := h / kvg d

/-- The number of query blocks, `cdiv(max_input_len, 128)` (line 215). -/
def nBlocks (d : Inst) : Nat := (d.maxLen + BLOCK - 1) / BLOCK

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid `(batch, num_heads, cdiv(max_input_len, 128))` within CUDA's limits and fewer than
  `2^31` programs (`Launch.Valid`). -/
  batch_lt : d.batch < 2 ^ 31
  H_lt : d.H < 2 ^ 16
  grid_lt : d.batch * d.H * nBlocks d < 2 ^ 31
  /-- `num_kv_heads ≥ 1` divides `num_heads` (SETUP.md). -/
  kv_pos : 0 < d.HKV
  kv_dvd : d.HKV ∣ d.H
  /-- The head dimension: `1 ≤ Lk`, and `BLOCK_DMODEL` a power of two with
  `Lk ≤ BLOCK_DMODEL ≤ 256` (`next_power_of_2(Lk)` for `Lk ≤ 256`; SETUP.md's head dims are 13 to
  128). -/
  lk_pos : 0 < d.Lk
  bd_pow2 : ∃ a, d.BD = 2 ^ a
  lk_le : d.Lk ≤ d.BD
  bd_le : d.BD ≤ 256
  /-- `max_input_len ≤ 2^17` (prompts of up to 131072 tokens; SETUP.md: 1 to thousands). -/
  maxLen_le : d.maxLen ≤ 2 ^ 17
  /-- `sm_scale` is a finite double in `[2^-64, 2^64]` (by default `1/sqrt(Lq) ∈ [1/16, 1]`). -/
  scale : d.sm_scale.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ d.sm_scale.toRat ∧
    d.sm_scale.toRat ≤ 2 ^ 64
  /-- The eight strides are `int32` arguments. -/
  strides_lt : d.sqt < 2 ^ 31 ∧ d.sqh < 2 ^ 31 ∧ d.skt < 2 ^ 31 ∧ d.skh < 2 ^ 31 ∧
    d.svt < 2 ^ 31 ∧ d.svh < 2 ^ 31 ∧ d.sot < 2 ^ 31 ∧ d.soh < 2 ^ 31
  /-- **A limit of the kernel**: its `int32` element offsets (lines 90–96, 125, 165, 176–180) do
  not wrap: every element of `q`, `k`, `v`, `o` has an index below `2^31`. -/
  q_i32 : 0 < d.total → (d.total - 1) * d.sqt + (d.H - 1) * d.sqh + d.Lk ≤ 2 ^ 31
  k_i32 : 0 < d.total → (d.total - 1) * d.skt + (d.HKV - 1) * d.skh + d.Lk ≤ 2 ^ 31
  v_i32 : 0 < d.total → (d.total - 1) * d.svt + (d.HKV - 1) * d.svh + d.Lk ≤ 2 ^ 31
  o_i32 : 0 < d.total → (d.total - 1) * d.sot + (d.H - 1) * d.soh + d.Lk ≤ 2 ^ 31
  /-- Each storage holds its tensor's last element. -/
  q_len : 0 < d.total → (d.total - 1) * d.sqt + (d.H - 1) * d.sqh + d.Lk ≤ d.q.len
  k_len : 0 < d.total → (d.total - 1) * d.skt + (d.HKV - 1) * d.skh + d.Lk ≤ d.k.len
  v_len : 0 < d.total → (d.total - 1) * d.svt + (d.HKV - 1) * d.svh + d.Lk ≤ d.v.len
  o_len : 0 < d.total → (d.total - 1) * d.sot + (d.H - 1) * d.soh + d.Lk ≤ d.o.len
  bsl_len : d.batch ≤ d.b_start_loc.len
  bsq_len : d.batch ≤ d.b_seq_len.len
  /-- `o`'s elements are distinct: it is row-major in one order or the other
  (`torch.empty_like(q)`: `q`'s strides when `q` is dense, else contiguous). -/
  o_layout : (d.Lk ≤ d.soh ∧ d.H * d.soh ≤ d.sot) ∨ (d.Lk ≤ d.sot ∧ d.total * d.sot ≤ d.soh)
  /-- `o` shares no byte with the five inputs (a fresh allocation, line 197). -/
  disj : Disjoint d.o.bytes d.q.bytes ∧ Disjoint d.o.bytes d.k.bytes ∧
    Disjoint d.o.bytes d.v.bytes ∧ Disjoint d.o.bytes d.b_start_loc.bytes ∧
    Disjoint d.o.bytes d.b_seq_len.bytes

/-- The model on an instance: the launch of lines 218–242. -/
def model (d : Inst) : Kernel :=
  fwd_kernel (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (FloatV.hostF32 d.sm_scale)
    (d.b_start_loc.ptr 0) (d.b_seq_len.ptr 0)
    (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
    (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
    (kvg d) BLOCK d.BD BLOCK d.causal d.Lk

/-- The host's launch (line 215): the grid `(batch, num_heads, cdiv(max_input_len, 128))`; the
source's block shapes (`[128]`: `offs_m`, `offs_n`, `m_i`, `l_i`, …; `[BD]`: `offs_d`, `mask_d`;
`[128, BD]`: `q`, `acc`, `v`, `o`; `[BD, 128]`: `k`; `[128, 128]`: `qk`, `p`; and the expanded
`[128, 1]`, `[1, 128]`, `[1, BD]`, `[BD, 1]`); `enable_fp_fusion` Triton's default (`true`);
`num_ctas = 1`, no `launch_pdl`, `debug = false`. `num_warps` and `num_stages = 1` change no
meaning. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.batch, d.H, nBlocks d], fusion := true,
    blocks := [[BLOCK], [d.BD], [BLOCK, d.BD], [d.BD, BLOCK], [BLOCK, BLOCK], [BLOCK, 1],
      [1, BLOCK], [1, d.BD], [d.BD, 1]] }

/-- The `bfloat16` value at element `i` of a buffer, if present. -/
def rdB (m : Mem) (b : Buf .bf16) (i : Nat) : Option BF16 := m.read .bf16 (b.ptr (i : Int)).addr

/-- Element `i` of a `bfloat16` buffer as a rational (`0` when absent). -/
def valB (m : Mem) (b : Buf .bf16) (i : Nat) : ℚ :=
  match rdB m b i with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `bfloat16` buffer is a finite value. -/
def finB (m : Mem) (b : Buf .bf16) (i : Nat) : Prop :=
  ∃ x : BF16, rdB m b i = some x ∧ x.isFinite = true

/-- Element `i` of an `int32` buffer as an integer (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- `b_start_loc[b]`. -/
def startOf (d : Inst) (m : Mem) (b : Nat) : Nat := (val32 m d.b_start_loc b).toNat
/-- `b_seq_len[b]`. -/
def lenOf (d : Inst) (m : Mem) (b : Nat) : Nat := (val32 m d.b_seq_len b).toNat

/-- `sm_scale` as a real. -/
noncomputable def sig (d : Inst) : ℝ := (d.sm_scale.toRat : ℝ)

/-- Component `e` of the query of local position `p` of sequence `b`, head `h`. -/
def qv (d : Inst) (m : Mem) (b h p e : Nat) : ℚ := valB m d.q (qIdx d (startOf d m b + p) h e)
/-- Component `e` of the key of local position `j` of sequence `b`, for query head `h`. -/
def kv (d : Inst) (m : Mem) (b h j e : Nat) : ℚ := valB m d.k (kIdx d (startOf d m b + j) (kvh d h) e)
/-- Component `e` of the value of local position `j` of sequence `b`, for query head `h`. -/
def vv (d : Inst) (m : Mem) (b h j e : Nat) : ℚ := valB m d.v (vIdx d (startOf d m b + j) (kvh d h) e)

/-- **The exact score** of query `p` against key `j` (sequence `b`, head `h`):
`sm_scale · Σ_e q·k`. -/
noncomputable def score (d : Inst) (m : Mem) (b h p j : Nat) : ℝ :=
  sig d * ∑ e ∈ Finset.range d.Lk, ((qv d m b h p e : ℚ) : ℝ) * ((kv d m b h j e : ℚ) : ℝ)

/-- The score's magnitude, `sm_scale · Σ_e liftMag q · liftMag k` (`Rules.Num.liftMag`: a
subnormal counted at `2^-126`), the magnitude `Rules.Num.DotBF16`'s error is relative to. It is
at least `|score|`. -/
noncomputable def scoreMag (d : Inst) (m : Mem) (b h p j : Nat) : ℝ :=
  sig d * ∑ e ∈ Finset.range d.Lk,
    ((Rules.Num.liftMag .bf16 (qv d m b h p e) * Rules.Num.liftMag .bf16 (kv d m b h j e) : ℚ) : ℝ)

/-- The last key query `p` of sequence `b` attends: `p` when causal (the keys `j ≤ p`), else
`b_seq_len[b] − 1` (every key of the sequence, `j < b_seq_len[b]`). -/
def lastKey (d : Inst) (m : Mem) (b p : Nat) : Nat :=
  if d.causal then p else lenOf d m b - 1

/-- The row's score bound: the largest `scoreMag` over the keys it attends, `j ≤ lastKey`. -/
noncomputable def Xrow (d : Inst) (m : Mem) (b h p : Nat) : ℝ :=
  (Finset.range (lastKey d m b p + 1)).sup' ⟨0, by simp⟩ fun j => scoreMag d m b h p j

/-- The packed sequences (SETUP.md): each `b_start_loc[b]`, `b_seq_len[b]` an `int32` with
`0 ≤ start`, `0 ≤ len ≤ max_input_len`, `start + len ≤ total_tokens`; no two sequences share a
token. -/
def SeqOk (d : Inst) (m : Mem) : Prop :=
  (∀ b, b < d.batch → ∃ s n : I32,
    m.read .i32 (d.b_start_loc.ptr (b : Int)).addr = some s ∧
    m.read .i32 (d.b_seq_len.ptr (b : Int)).addr = some n ∧
    0 ≤ s.toInt ∧ 0 ≤ n.toInt ∧ n.toInt ≤ d.maxLen ∧ s.toInt + n.toInt ≤ d.total) ∧
  ∀ b b', b < d.batch → b' < d.batch → b ≠ b' → 0 < lenOf d m b → 0 < lenOf d m b' →
    startOf d m b + lenOf d m b ≤ startOf d m b' ∨ startOf d m b' + lenOf d m b' ≤ startOf d m b

/-- The input values covered: in every sequence, every query, key and value component is a
finite `bfloat16`, every `|v| ≤ 2^120`, and every query–key pair of the sequence (masked or not)
has `scoreMag ≤ 2^12`. -/
def ValsOk (d : Inst) (m : Mem) : Prop :=
  ∀ b, b < d.batch →
    (∀ p h e, p < lenOf d m b → h < d.H → e < d.Lk → finB m d.q (qIdx d (startOf d m b + p) h e)) ∧
    (∀ j g e, j < lenOf d m b → g < d.HKV → e < d.Lk →
      finB m d.k (kIdx d (startOf d m b + j) g e) ∧ finB m d.v (vIdx d (startOf d m b + j) g e) ∧
      |valB m d.v (vIdx d (startOf d m b + j) g e)| ≤ 2 ^ 120) ∧
    (∀ h p j, h < d.H → p < lenOf d m b → j < lenOf d m b → scoreMag d m b h p j ≤ 2 ^ 12)

/-- The inputs covered: the six buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), the sequences are packed (`SeqOk`), and their values covered (`ValsOk`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k.Valid init ∧ d.v.Valid init ∧ d.o.Valid init ∧
    d.b_start_loc.Valid init ∧ d.b_seq_len.Valid init ∧ SeqOk d init ∧ ValsOk d init

/-! ## The specification -/

/-- **The exact attention** of query `p` of sequence `b`, head `h`, component `e`, over the keys
it attends (`L = lastKey`: `j ≤ p` when causal, every `j < b_seq_len[b]` otherwise): the
softmax-weighted mean `Σ_{j ≤ L} e^{score_j}·v_j / Σ_{j ≤ L} e^{score_j}`. -/
noncomputable def attnRef (d : Inst) (m : Mem) (b h p e : Nat) : ℝ :=
  (∑ j ∈ Finset.range (lastKey d m b p + 1),
      Real.exp (score d m b h p j) * ((vv d m b h j e : ℚ) : ℝ)) /
    ∑ j ∈ Finset.range (lastKey d m b p + 1), Real.exp (score d m b h p j)

/-- The same weighted mean of `|v|`, the scale of the tolerance (`≥ |attnRef|`). -/
noncomputable def attnAbs (d : Inst) (m : Mem) (b h p e : Nat) : ℝ :=
  (∑ j ∈ Finset.range (lastKey d m b p + 1),
      Real.exp (score d m b h p j) * |((vv d m b h j e : ℚ) : ℝ)|) /
    ∑ j ∈ Finset.range (lastKey d m b p + 1), Real.exp (score d m b h p j)

/-- `Σ_{j ≤ lastKey} |v_j|`, the scale of the absolute floor. -/
noncomputable def vSum (d : Inst) (m : Mem) (b h p e : Nat) : ℝ :=
  ∑ j ∈ Finset.range (lastKey d m b p + 1), |((vv d m b h j e : ℚ) : ℝ)|

/-- The loop trips that reach key `L`, the trips of query `p`'s program when `L = lastKey`:
`L // 128 + 1` (`start_m + 1` when causal, `cdiv(b_seq_len[b], 128)` otherwise). -/
def trips (L : Nat) : Nat := L / BLOCK + 1

/-! ### The tolerances (README, "Tolerances")

`BD = BLOCK_DMODEL`, `T` the row's loop trips, `X` the row's score bound. -/

/-- The bound on each computed score's error: the score `tl.dot` (`(BD+2)·2^-22` of `scoreMag`),
the `float32` rounding of `sm_scale`, the product by it and the `tl.dot_add`'s addition of `+0`,
and absolute floors. -/
noncomputable def dlt (BD : Nat) (X : ℝ) : ℝ := ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X + 1 / 2 ^ 40

/-- The bound on the relative error of each key's coefficient and of the normaliser `l_i`: one
`bfloat16` rounding of `p` (`2^-8`), the `float32` roundings and `tl.exp`/`/` errors of each trip
(`2^-18` per trip), and `tl.exp`'s error that grows with its argument (`2^-17·(X + dlt)`). -/
noncomputable def rho (BD T : Nat) (X : ℝ) : ℝ :=
  Real.exp (1 / 2 ^ 8 + ((T : ℝ) + 8) * (1 / 2 ^ 18) + (X + dlt BD X) * (1 / 2 ^ 17)) - 1

/-- The bound on the accumulated `p·v` `tl.dot` errors, relative to the coefficients' weighted
`|v|`: `(BLOCK_N + 8)·2^-22` per trip, compounded. -/
noncomputable def gam (T : Nat) : ℝ := Real.exp ((T : ℝ) * 136 * (1 / 2 ^ 22)) - 1

/-- The bound on `|acc − attnRef| / attnAbs` before the store. -/
noncomputable def errE (BD T : Nat) (X : ℝ) : ℝ :=
  Real.exp (2 * dlt BD X) * (rho BD T X + gam T * (1 + rho BD T X)) / (1 - rho BD T X) +
    (Real.exp (2 * dlt BD X) - 1)

/-- **The output's tolerance relative to `attnAbs`**: the store's `bfloat16` rounding and
`errE`. -/
noncomputable def tolRel (BD T : Nat) (X : ℝ) : ℝ := 1 / 2 ^ 8 + (1 + 1 / 2 ^ 8) * errE BD T X

/-- **The output's absolute floor**: `2^-80·(1 + Σ|v|)` (subnormal and absolute terms). -/
noncomputable def tolAbs (vs : ℝ) : ℝ := (1 / 2 ^ 80) * (1 + vs)

/-- The elements of `o` the launch writes. -/
def outElems (d : Inst) (m : Mem) : Set Nat :=
  {x | ∃ b p h e, b < d.batch ∧ p < lenOf d m b ∧ h < d.H ∧ e < d.Lk ∧
    x = oIdx d (startOf d m b + p) h e}

/-- What the launch leaves: every `o[start_b + p, h, e]` (`b < batch`, `p < b_seq_len[b]`,
`h < num_heads`, `e < Lk`) is a finite `bfloat16` within
`tolRel(BD, lastKey//128 + 1, Xrow)·attnAbs + tolAbs(Σ|v|)` of the exact attention `attnRef`
(causal or not, as `is_causal`), over the initial inputs; every byte outside these elements is
unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b p h e, b < d.batch → p < lenOf d init b → h < d.H → e < d.Lk →
    ∃ c : BF16, rdB fin d.o (oIdx d (startOf d init b + p) h e) = some c ∧ c.isFinite = true ∧
      |c.toReal - attnRef d init b h p e| ≤
        tolRel d.BD (trips (lastKey d init b p)) (Xrow d init b h p) * attnAbs d init b h p e +
          tolAbs (vSum d init b h p e)) ∧
  ∀ z, z ∉ d.o.elems (outElems d init) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16`: `tl.dot` of `bfloat16` operands into `float32` (lines 132, 171). -/
  dot : Rules.Num.DotBF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` has a result for all operands. -/
  dot_total : Rules.Num.DotTotal P cfgBF16
  /-- [rule] `Num.ExpF32`: `tl.exp` on `float32` (lines 149, 153, 154). -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.ExpF32NegInf`: `tl.exp(-∞) = +0` on `float32` (masked lanes; the first trip's
  `alpha`). -/
  exp_ninf : Rules.Num.ExpF32NegInf P
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input. -/
  exp_total : Rules.Num.MathTotal P .exp .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32` (lines 158, 161). -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs. -/
  div_total : Rules.Num.DivTotal P .binary32

/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotBF16,
    Platform.ref.dotTotal _, Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf,
    Platform.ref.mathTotal, Platform.ref.divFull, Platform.ref.divTotal _⟩⟩

/-! ## The host's layouts

`canon batch total H HKV Lk BD maxLen sm causal` is the host's launch for those sizes: contiguous `q`,
`k`, `v` (`[total, H, Lk]`, `[total, HKV, Lk]`) and `o = empty_like(q)`, the six storages at
`2^40`, `2·2^40`, …, `6·2^40`. -/

def canon (batch total H HKV Lk BD maxLen : Nat) (sm : F64) (causal : Bool) : Inst where
  batch := batch
  total := total
  H := H
  HKV := HKV
  Lk := Lk
  BD := BD
  maxLen := maxLen
  sm_scale := sm
  causal := causal
  sqt := H * Lk
  sqh := Lk
  skt := HKV * Lk
  skh := Lk
  svt := HKV * Lk
  svh := Lk
  sot := H * Lk
  soh := Lk
  q := ⟨2 ^ 40, total * H * Lk⟩
  k := ⟨2 * 2 ^ 40, total * HKV * Lk⟩
  v := ⟨3 * 2 ^ 40, total * HKV * Lk⟩
  o := ⟨4 * 2 ^ 40, total * H * Lk⟩
  b_start_loc := ⟨5 * 2 ^ 40, batch⟩
  b_seq_len := ⟨6 * 2 ^ 40, batch⟩

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The last element of a contiguous `[t, a, l]` tensor. -/
theorem last_idx (t a l : Nat) (ht : 0 < t) (ha : 0 < a) :
    (t - 1) * (a * l) + (a - 1) * l + l = t * a * l := by
  obtain ⟨t', rfl⟩ : ∃ t', t = t' + 1 := ⟨t - 1, by omega⟩
  obtain ⟨a', rfl⟩ : ∃ a', a = a' + 1 := ⟨a - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  ring

/-- **Every host launch at the code base's sizes is covered**: up to `2^14` sequences and
`2^17` tokens, `1 … 64` heads with any `num_kv_heads` dividing them, head dims up to `256`,
`max_input_len ≤ 2^17`, an `sm_scale` in `[2^-64, 2^64]`, causal or not. -/
theorem canon_ok (batch total H HKV Lk BD maxLen a : Nat) (sm : F64) (causal : Bool)
    (hb : batch ≤ 2 ^ 14) (ht : total ≤ 2 ^ 17) (hH0 : 0 < H) (hH : H ≤ 64) (hkv0 : 0 < HKV)
    (hkv : HKV ∣ H) (hl0 : 0 < Lk) (hl : Lk ≤ BD) (hbd : BD = 2 ^ a) (hbd256 : BD ≤ 256)
    (hm : maxLen ≤ 2 ^ 17)
    (hsm : sm.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm.toRat ∧ sm.toRat ≤ 2 ^ 64) :
    (canon batch total H HKV Lk BD maxLen sm causal).ok := by
  have hkvH : HKV ≤ H := Nat.le_of_dvd hH0 hkv
  have hl256 : Lk ≤ 256 := le_trans hl hbd256
  have hHL : H * Lk ≤ 64 * 256 := Nat.mul_le_mul hH hl256
  have hKL : HKV * Lk ≤ 64 * 256 := Nat.mul_le_mul (le_trans hkvH hH) hl256
  have hqs : total * H * Lk ≤ 2 ^ 17 * 64 * 256 := Nat.mul_le_mul (Nat.mul_le_mul ht hH) hl256
  have hks : total * HKV * Lk ≤ 2 ^ 17 * 64 * 256 :=
    Nat.mul_le_mul (Nat.mul_le_mul ht (le_trans hkvH hH)) hl256
  have hnb : (maxLen + BLOCK - 1) / BLOCK ≤ 2 ^ 10 := by
    simp only [BLOCK]; omega
  have hbH : batch * H ≤ 2 ^ 14 * 64 := Nat.mul_le_mul hb hH
  have hgrid : batch * H * ((maxLen + BLOCK - 1) / BLOCK) ≤ 2 ^ 14 * 64 * 2 ^ 10 :=
    Nat.mul_le_mul hbH hnb
  refine
    { batch_lt := by simp only [canon]; omega
      H_lt := by simp only [canon]; omega
      grid_lt := by simp only [canon, nBlocks]; omega
      kv_pos := hkv0
      kv_dvd := hkv
      lk_pos := hl0
      bd_pow2 := ⟨a, hbd⟩
      lk_le := hl
      bd_le := hbd256
      maxLen_le := hm
      scale := hsm
      strides_lt := by simp only [canon]; omega
      q_i32 := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]; omega
      k_i32 := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]; omega
      v_i32 := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]; omega
      o_i32 := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]; omega
      q_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]
      k_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]
      v_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]
      o_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]
      bsl_len := le_rfl
      bsq_len := le_rfl
      o_layout := Or.inl ⟨le_rfl, le_rfl⟩
      disj := by
        refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
          exact bytes_disj _ _ (by simp only [canon, DType.bytes]; omega) }

/-! ## Coverage: the host's test cases (source lines 270–279) and SETUP.md's model shapes -/

/-- `1/sqrt(128)`, the host's default `sm_scale` at head dim 128 (the double `0x3fb6a09e667f3bcc`). -/
def sm128 : F64 := FloatV.ofPat 0x3fb6a09e667f3bcc
/-- `1/sqrt(96)`. -/
def sm96 : F64 := FloatV.ofPat 0x3fba20bd700c2c3f
/-- `1/sqrt(80)`. -/
def sm80 : F64 := FloatV.ofPat 0x3fbc9f25c5bfedd9
/-- `1/sqrt(13)`. -/
def sm13 : F64 := FloatV.ofPat 0x3fd1c01aa03be896
/-- `1/sqrt(64) = 0.125`. -/
def sm64 : F64 := FloatV.ofPat 0x3fc0000000000000

theorem sm128_ok : sm128.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm128.toRat ∧ sm128.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm96_ok : sm96.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm96.toRat ∧ sm96.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm80_ok : sm80.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm80.toRat ∧ sm80.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm13_ok : sm13.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm13.toRat ∧ sm13.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm64_ok : sm64.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm64.toRat ∧ sm64.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- `(4, 4, 128, [8, 12])`. -/
def setup_h128 : Inst := canon 2 20 4 4 128 128 12 sm128 true
theorem setup_h128_ok : setup_h128.ok :=
  canon_ok 2 20 4 4 128 128 12 7 sm128 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) sm128_ok

/-- `(4, 4, 96, [8, 12])`. -/
def setup_h96 : Inst := canon 2 20 4 4 96 128 12 sm96 true
theorem setup_h96_ok : setup_h96.ok :=
  canon_ok 2 20 4 4 96 128 12 7 sm96 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) sm96_ok

/-- `(4, 4, 80, [8, 12])`. -/
def setup_h80 : Inst := canon 2 20 4 4 80 128 12 sm80 true
theorem setup_h80_ok : setup_h80.ok :=
  canon_ok 2 20 4 4 80 128 12 7 sm80 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) sm80_ok

/-- `(4, 4, 13, [8, 12])`. -/
def setup_h13 : Inst := canon 2 20 4 4 13 16 12 sm13 true
theorem setup_h13_ok : setup_h13.ok :=
  canon_ok 2 20 4 4 13 16 12 4 sm13 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) sm13_ok

/-- `(32, 8, 128, [1, 127, 128, 129, 1000])` (Llama-3-8B's heads). -/
def setup_llama : Inst := canon 5 1385 32 8 128 128 1000 sm128 true
theorem setup_llama_ok : setup_llama.ok :=
  canon_ok 5 1385 32 8 128 128 1000 7 sm128 true (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num)
    (by norm_num) sm128_ok

/-- `(28, 4, 128, [255, 3, 2048])` (Qwen2.5-7B's heads). -/
def setup_qwen : Inst := canon 3 2306 28 4 128 128 2048 sm128 true
theorem setup_qwen_ok : setup_qwen.ok :=
  canon_ok 3 2306 28 4 128 128 2048 7 sm128 true (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num)
    (by norm_num) sm128_ok

/-- `(16, 16, 80, [1024, 64, 1])` (a vision encoder). -/
def setup_vision : Inst := canon 3 1089 16 16 80 128 1024 sm80 true
theorem setup_vision_ok : setup_vision.ok :=
  canon_ok 3 1089 16 16 80 128 1024 7 sm80 true (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) sm80_ok

/-- `(8, 1, 64, [300, 513])`. -/
def setup_h64 : Inst := canon 2 813 8 1 64 64 513 sm64 true
theorem setup_h64_ok : setup_h64.ok :=
  canon_ok 2 813 8 1 64 64 513 6 sm64 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) sm64_ok

/-- SETUP.md's model shapes at their largest: 64 heads per rank with grouped KV heads (8), head
dim 128, prompts of thousands of tokens (two sequences of 8192). -/
def setup_large : Inst := canon 2 16384 64 8 128 128 8192 sm128 true
theorem setup_large_ok : setup_large.ok :=
  canon_ok 2 16384 64 8 128 128 8192 7 sm128 true (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num)
    (by norm_num) sm128_ok

/-- SETUP.md's strided inputs: a fused `qkv` projection, `q`, `k`, `v` the views
`qkv[:, :H·D]`, `qkv[:, H·D:(H+Hkv)·D]`, `qkv[:, (H+Hkv)·D:]` of one `[total, (H + 2·Hkv)·D]`
tensor (token stride `(32 + 2·8)·128 = 6144`), at Llama-3-8B's heads (32/8, head dim 128) and the
host's lengths `[1, 127, 128, 129, 1000]`; `o = empty_like(q)` is contiguous (`q` is not dense). The
three input buffers are the views' spans of the one storage at `2^40`. -/
def setup_fused : Inst where
  batch := 5
  total := 1385
  H := 32
  HKV := 8
  Lk := 128
  BD := 128
  maxLen := 1000
  sm_scale := sm128
  causal := true
  sqt := 6144
  sqh := 128
  skt := 6144
  skh := 128
  svt := 6144
  svh := 128
  sot := 4096
  soh := 128
  q := ⟨2 ^ 40, 1385 * 6144⟩
  k := ⟨2 ^ 40 + 4096 * 2, 1385 * 6144 - 4096⟩
  v := ⟨2 ^ 40 + 5120 * 2, 1385 * 6144 - 5120⟩
  o := ⟨4 * 2 ^ 40, 1385 * 4096⟩
  b_start_loc := ⟨5 * 2 ^ 40, 5⟩
  b_seq_len := ⟨6 * 2 ^ 40, 5⟩

theorem setup_fused_ok : setup_fused.ok where
  batch_lt := by norm_num [setup_fused]
  H_lt := by norm_num [setup_fused]
  grid_lt := by norm_num [setup_fused, nBlocks, BLOCK]
  kv_pos := by norm_num [setup_fused]
  kv_dvd := by norm_num [setup_fused]
  lk_pos := by norm_num [setup_fused]
  bd_pow2 := ⟨7, by norm_num [setup_fused]⟩
  lk_le := by norm_num [setup_fused]
  bd_le := by norm_num [setup_fused]
  maxLen_le := by norm_num [setup_fused]
  scale := sm128_ok
  strides_lt := by norm_num [setup_fused]
  q_i32 := fun _ => by norm_num [setup_fused]
  k_i32 := fun _ => by norm_num [setup_fused]
  v_i32 := fun _ => by norm_num [setup_fused]
  o_i32 := fun _ => by norm_num [setup_fused]
  q_len := fun _ => by norm_num [setup_fused]
  k_len := fun _ => by norm_num [setup_fused]
  v_len := fun _ => by norm_num [setup_fused]
  o_len := fun _ => by norm_num [setup_fused]
  bsl_len := by norm_num [setup_fused]
  bsq_len := by norm_num [setup_fused]
  o_layout := Or.inl ⟨by norm_num [setup_fused], by norm_num [setup_fused]⟩
  disj := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
      exact bytes_disj _ _ (by simp only [setup_fused, DType.bytes]; omega)

/-! ## Coverage: the vision encoders' launches (`is_causal=False`)

The vision encoder (`python/sglang/srt/layers/attention/vision.py:510`) calls the wrapper with
`is_causal=False`, `q`, `k`, `v` the views of its fused `qkv` projection (`vision.py:1425-1432`,
`STRIDED_QKV_BACKENDS`: no copy), `o = empty_like(q)` (contiguous: `q` is not dense),
`b_start_loc = cu_seqlens`, `b_seq_len` its differences (`int32`), `sm_scale` the default.
`fusedQKV batch total H HKV Lk BD maxLen sm causal` is that launch: `q`, `k`, `v` the views
`qkv[:, :H·Lk]`, `qkv[:, H·Lk:(H+HKV)·Lk]`, `qkv[:, (H+HKV)·Lk:]` of one
`[total, (H + 2·HKV)·Lk]` storage at `2^40` (each input buffer the view's span), `o` at `4·2^40`,
`b_start_loc`, `b_seq_len` at `5·2^40`, `6·2^40`. -/

def fusedQKV (batch total H HKV Lk BD maxLen : Nat) (sm : F64) (causal : Bool) : Inst where
  batch := batch
  total := total
  H := H
  HKV := HKV
  Lk := Lk
  BD := BD
  maxLen := maxLen
  sm_scale := sm
  causal := causal
  sqt := (H + 2 * HKV) * Lk
  sqh := Lk
  skt := (H + 2 * HKV) * Lk
  skh := Lk
  svt := (H + 2 * HKV) * Lk
  svh := Lk
  sot := H * Lk
  soh := Lk
  q := ⟨2 ^ 40, total * ((H + 2 * HKV) * Lk)⟩
  k := ⟨2 ^ 40 + H * Lk * 2, total * ((H + 2 * HKV) * Lk) - H * Lk⟩
  v := ⟨2 ^ 40 + (H * Lk + HKV * Lk) * 2, total * ((H + 2 * HKV) * Lk) - (H * Lk + HKV * Lk)⟩
  o := ⟨4 * 2 ^ 40, total * H * Lk⟩
  b_start_loc := ⟨5 * 2 ^ 40, batch⟩
  b_seq_len := ⟨6 * 2 ^ 40, batch⟩

/-- A row index below the last row's start plus one row. -/
theorem row_le (t W x : Nat) (ht : 0 < t) (hx : x ≤ W) : (t - 1) * W + x ≤ t * W := by
  obtain ⟨t', rfl⟩ : ∃ t', t = t' + 1 := ⟨t - 1, by omega⟩
  simp only [Nat.add_sub_cancel, Nat.succ_mul]
  omega

/-- The last head's end. -/
theorem head_end (a l : Nat) (ha : 0 < a) : (a - 1) * l + l = a * l := by
  obtain ⟨a', rfl⟩ : ∃ a', a = a' + 1 := ⟨a - 1, by omega⟩
  simp only [Nat.add_sub_cancel, Nat.succ_mul]

/-- **Every vision-encoder launch at the code base's sizes is covered**: up to `2^14` sequences,
`1 … 64` heads with any `num_kv_heads` dividing them, head dims up to `256`,
`max_input_len ≤ 2^17`, a fused `qkv` of fewer than `2^31` elements, causal or not. -/
theorem fusedQKV_ok (batch total H HKV Lk BD maxLen a : Nat) (sm : F64) (causal : Bool)
    (hb : batch ≤ 2 ^ 14) (hH0 : 0 < H) (hH : H ≤ 64) (hkv0 : 0 < HKV)
    (hkv : HKV ∣ H) (hl0 : 0 < Lk) (hl : Lk ≤ BD) (hbd : BD = 2 ^ a) (hbd256 : BD ≤ 256)
    (hm : maxLen ≤ 2 ^ 17)
    (hsm : sm.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm.toRat ∧ sm.toRat ≤ 2 ^ 64)
    (hW : total * ((H + 2 * HKV) * Lk) ≤ 2 ^ 31) :
    (fusedQKV batch total H HKV Lk BD maxLen sm causal).ok := by
  have hkvH : HKV ≤ H := Nat.le_of_dvd hH0 hkv
  have hl256 : Lk ≤ 256 := le_trans hl hbd256
  have hWe : (H + 2 * HKV) * Lk = H * Lk + 2 * (HKV * Lk) := by ring
  have hHL : H * Lk ≤ 64 * 256 := Nat.mul_le_mul hH hl256
  have hKL : HKV * Lk ≤ 64 * 256 := Nat.mul_le_mul (le_trans hkvH hH) hl256
  have hoW : total * (H * Lk) ≤ total * ((H + 2 * HKV) * Lk) :=
    Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ (by omega))
  have hoA : total * H * Lk = total * (H * Lk) := Nat.mul_assoc _ _ _
  have hnb : (maxLen + BLOCK - 1) / BLOCK ≤ 2 ^ 10 := by
    simp only [BLOCK]; omega
  have hbH : batch * H ≤ 2 ^ 14 * 64 := Nat.mul_le_mul hb hH
  have hgrid : batch * H * ((maxLen + BLOCK - 1) / BLOCK) ≤ 2 ^ 14 * 64 * 2 ^ 10 :=
    Nat.mul_le_mul hbH hnb
  refine
    { batch_lt := by simp only [fusedQKV]; omega
      H_lt := by simp only [fusedQKV]; omega
      grid_lt := by simp only [fusedQKV, nBlocks]; omega
      kv_pos := hkv0
      kv_dvd := hkv
      lk_pos := hl0
      bd_pow2 := ⟨a, hbd⟩
      lk_le := hl
      bd_le := hbd256
      maxLen_le := hm
      scale := hsm
      strides_lt := by simp only [fusedQKV]; omega
      q_i32 := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        have := row_le total ((H + 2 * HKV) * Lk) (H * Lk) h0 (by omega)
        rw [Nat.add_assoc, head_end H Lk hH0]; omega
      k_i32 := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        have := row_le total ((H + 2 * HKV) * Lk) (HKV * Lk) h0 (by omega)
        rw [Nat.add_assoc, head_end HKV Lk hkv0]; omega
      v_i32 := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        have := row_le total ((H + 2 * HKV) * Lk) (HKV * Lk) h0 (by omega)
        rw [Nat.add_assoc, head_end HKV Lk hkv0]; omega
      o_i32 := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        rw [last_idx _ _ _ h0 hH0]; omega
      q_len := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        have := row_le total ((H + 2 * HKV) * Lk) (H * Lk) h0 (by omega)
        rw [Nat.add_assoc, head_end H Lk hH0]; omega
      k_len := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        have := row_le total ((H + 2 * HKV) * Lk) (HKV * Lk + H * Lk) h0 (by omega)
        rw [Nat.add_assoc, head_end HKV Lk hkv0]; omega
      v_len := fun h0 => by
        simp only [fusedQKV] at h0 ⊢
        have := row_le total ((H + 2 * HKV) * Lk) (HKV * Lk + (H * Lk + HKV * Lk)) h0 (by omega)
        rw [Nat.add_assoc, head_end HKV Lk hkv0]; omega
      o_len := fun h0 => by
        simp only [fusedQKV] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]
      bsl_len := le_rfl
      bsq_len := le_rfl
      o_layout := Or.inl ⟨le_rfl, le_rfl⟩
      disj := by
        refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
          exact bytes_disj _ _ (by simp only [fusedQKV, DType.bytes]; omega) }

/-- `1/sqrt(72)`, the default `sm_scale` at head dim 72 (the double `0x3fbe2b7dddfefa67`). -/
def sm72 : F64 := FloatV.ofPat 0x3fbe2b7dddfefa67
/-- `1.0`, an explicit `sm_scale` (the Gemma 4 vision tower's `softmax_scale=1.0`,
`python/sglang/srt/models/gemma4_vision.py:194`). -/
def sm1 : F64 := FloatV.ofPat 0x3ff0000000000000

theorem sm72_ok : sm72.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm72.toRat ∧ sm72.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm1_ok : sm1.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm1.toRat ∧ sm1.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- Qwen2.5-VL's vision encoder (16 heads of 80), a full-attention layer on one image of
`74 × 74 = 5476` patches. -/
def setup_vit_full : Inst := fusedQKV 1 5476 16 16 80 128 5476 sm80 false
theorem setup_vit_full_ok : setup_vit_full.ok :=
  fusedQKV_ok 1 5476 16 16 80 128 5476 7 sm80 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) sm80_ok (by norm_num)

/-- The same image in a window-attention layer: `100` windows of at most `64` patches. -/
def setup_vit_window : Inst := fusedQKV 100 5476 16 16 80 128 64 sm80 false
theorem setup_vit_window_ok : setup_vit_window.ok :=
  fusedQKV_ok 100 5476 16 16 80 128 64 7 sm80 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) sm80_ok (by norm_num)

/-- The largest image SGLang's Qwen-VL processor admits (`SGLANG_IMAGE_MAX_PIXELS = 16384·28·28`:
`65536` patches) in a full-attention layer, 16 heads of 80. -/
def setup_vit_max : Inst := fusedQKV 1 65536 16 16 80 128 65536 sm80 false
theorem setup_vit_max_ok : setup_vit_max.ok :=
  fusedQKV_ok 1 65536 16 16 80 128 65536 7 sm80 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) sm80_ok (by norm_num)

/-- The same image with the heads split over 8 ranks: 2 heads of 80 per rank. -/
def setup_vit_max_tp8 : Inst := fusedQKV 1 65536 2 2 80 128 65536 sm80 false
theorem setup_vit_max_tp8_ok : setup_vit_max_tp8.ok :=
  fusedQKV_ok 1 65536 2 2 80 128 65536 7 sm80 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) sm80_ok (by norm_num)

/-- A video: 16 frame pairs of `3072` patches each (`VIDEO_MAX_PIXELS = 768·28·28`), one sequence
per frame pair, 16 heads of 80. -/
def setup_vit_video : Inst := fusedQKV 16 49152 16 16 80 128 3072 sm80 false
theorem setup_vit_video_ok : setup_vit_video.ok :=
  fusedQKV_ok 16 49152 16 16 80 128 3072 7 sm80 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) sm80_ok (by norm_num)

/-- Qwen3-VL's vision encoder (16 heads of 72, `BLOCK_DMODEL = 128`), two images of `4096` and
`1024` patches. -/
def setup_vit_qwen3 : Inst := fusedQKV 2 5120 16 16 72 128 4096 sm72 false
theorem setup_vit_qwen3_ok : setup_vit_qwen3.ok :=
  fusedQKV_ok 2 5120 16 16 72 128 4096 7 sm72 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) sm72_ok (by norm_num)

/-- MiMo-VL's vision encoder (32 heads of 64 on 8 KV heads, `python/sglang/srt/models/mimo_vl.py:40-56`),
two images of `4096` and `1024` patches. -/
def setup_vit_mimo : Inst := fusedQKV 2 5120 32 8 64 64 4096 sm64 false
theorem setup_vit_mimo_ok : setup_vit_mimo.ok :=
  fusedQKV_ok 2 5120 32 8 64 64 4096 6 sm64 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num)
    (by norm_num) sm64_ok (by norm_num)

/-- An explicit `sm_scale = 1.0` on contiguous `q`, `k`, `v` (normalised per head before the
call), 16 heads of 64, two images of `1024` and `256` patches. -/
def setup_vit_scale1 : Inst := canon 2 1280 16 16 64 64 1024 sm1 false
theorem setup_vit_scale1_ok : setup_vit_scale1.ok :=
  canon_ok 2 1280 16 16 64 64 1024 6 sm1 false (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num)
    (by norm_num) sm1_ok

/-! ## Satisfiable

One sequence of one token, one head, head dim 16 (`BLOCK_DMODEL = 16`, the smallest that
compiles: `tl.dot` needs `K ≥ 16`), with every input `0` and `b_seq_len[0] = 1`. -/

def satInst : Inst := canon 1 1 1 1 16 16 1 sm64 true

theorem satInst_ok : satInst.ok :=
  canon_ok 1 1 1 1 16 16 1 4 sm64 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) sm64_ok

/-- Every byte `0`, except the low byte of `b_seq_len[0]`, which is `1`. -/
def satMem : Mem := fun x => if x = 6 * 2 ^ 40 then some 1 else some 0

theorem satMem_read_i32_len :
    satMem.read .i32 (satInst.b_seq_len.ptr ((0 : Nat) : Int)).addr = some (IntV.ofInt 1) := by
  decide +kernel

theorem satMem_read_i32_start :
    satMem.read .i32 (satInst.b_start_loc.ptr ((0 : Nat) : Int)).addr = some (IntV.ofInt 0) := by
  decide +kernel

theorem satMem_read_bf16 {i : Int} (hi : 0 ≤ i) (hi' : i < 2 ^ 39) (b : Int)
    (hb : b = 2 ^ 40 ∨ b = 2 * 2 ^ 40 ∨ b = 3 * 2 ^ 40) :
    satMem.read .bf16 (b + i * 2) = some (FloatV.ofPat 0 : BF16) := by
  have h1 : ¬ (b + i * 2 = 6 * 2 ^ 40) := by rcases hb with rfl | rfl | rfl <;> omega
  have h2 : ¬ (b + i * 2 + 1 = 6 * 2 ^ 40) := by rcases hb with rfl | rfl | rfl <;> omega
  have hbytes : satMem.readBytes (b + i * 2) 2 = some [0, 0] := by
    simp only [Mem.readBytes, satMem, h1, h2, ite_false]
  have hal : Mem.Aligned .bf16 (b + i * 2) := by
    refine ⟨by rcases hb with rfl | rfl | rfl <;> omega, ?_, ?_⟩
    · simp only [DType.bytes]; rcases hb with rfl | rfl | rfl <;> omega
    · simp only [DType.bytes]; omega
  unfold Mem.read
  rw [show DType.bf16.bytes = 2 from rfl]
  simp only [hal, ite_true, hbytes]
  decide

theorem satB {b : Buf .bf16} (hb : b = satInst.q ∨ b = satInst.k ∨ b = satInst.v) {i : Nat}
    (hi : i < 16) : rdB satMem b i = some (FloatV.ofPat 0) := by
  simp only [rdB, Buf.ptr_addr]
  refine satMem_read_bf16 (by omega) (by omega) _ ?_
  rcases hb with rfl | rfl | rfl <;> simp [satInst, canon]

theorem satV {b : Buf .bf16} (hb : b = satInst.q ∨ b = satInst.k ∨ b = satInst.v) {i : Nat}
    (hi : i < 16) : valB satMem b i = 0 := by
  simp only [valB, satB hb hi]
  decide +kernel

theorem satLen : lenOf satInst satMem 0 = 1 := by
  simp only [lenOf, val32, satMem_read_i32_len]; decide

theorem satStart : startOf satInst satMem 0 = 0 := by
  simp only [startOf, val32, satMem_read_i32_start]; decide

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨satInst, satMem, satInst_ok, ?_⟩
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid satMem := fun b h1 h2 h3 =>
    ⟨h1, h2, h3, fun x _ => by simp only [satMem]; split <;> simp⟩
  refine ⟨hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes])
      (by simp [satInst, canon, DType.bytes]), ⟨?_, ?_⟩, ?_⟩
  · intro b hb
    have hb0 : b = 0 := by simp [satInst, canon] at hb; omega
    subst hb0
    refine ⟨IntV.ofInt 0, IntV.ofInt 1, satMem_read_i32_start, satMem_read_i32_len, by decide,
      by decide, by decide, by decide⟩
  · intro b b' hb hb' hne
    simp [satInst, canon] at hb hb'
    omega
  · intro b hb
    have hb0 : b = 0 := by simp [satInst, canon] at hb; omega
    subst hb0
    rw [satLen, satStart]
    have hLk : satInst.Lk = 16 := rfl
    refine ⟨fun p h e hp hh he => ?_, fun j g e hj hg he => ?_, fun h p j hh hp hj => ?_⟩
    · have : p = 0 ∧ h = 0 ∧ e < 16 := by simp [satInst, canon] at hp hh he; omega
      obtain ⟨rfl, rfl, he'⟩ := this
      have hi : qIdx satInst (0 + 0) 0 e = e := by simp [qIdx]
      exact ⟨_, by rw [hi]; exact satB (Or.inl rfl) he', by decide⟩
    · have : j = 0 ∧ g = 0 ∧ e < 16 := by simp [satInst, canon] at hj hg he; omega
      obtain ⟨rfl, rfl, he'⟩ := this
      have hik : kIdx satInst (0 + 0) 0 e = e := by simp [kIdx]
      have hiv : vIdx satInst (0 + 0) 0 e = e := by simp [vIdx]
      refine ⟨⟨_, by rw [hik]; exact satB (Or.inr (Or.inl rfl)) he', by decide⟩,
        ⟨_, by rw [hiv]; exact satB (Or.inr (Or.inr rfl)) he', by decide⟩, ?_⟩
      rw [hiv, satV (Or.inr (Or.inr rfl)) he']; norm_num
    · have : h = 0 ∧ p = 0 ∧ j = 0 := by simp [satInst, canon] at hh hp hj; omega
      obtain ⟨rfl, rfl, rfl⟩ := this
      have hqv : ∀ e ∈ Finset.range satInst.Lk, qv satInst satMem 0 0 0 e = 0 := by
        intro e he
        rw [hLk, Finset.mem_range] at he
        simp only [qv, satStart]
        rw [show qIdx satInst (0 + 0) 0 e = e by simp [qIdx]]
        exact satV (Or.inl rfl) he
      have hsum : scoreMag satInst satMem 0 0 0 0 = 0 := by
        rw [scoreMag, Finset.sum_eq_zero fun e he => by
          rw [hqv e he]; simp [Rules.Num.liftMag]]
        ring
      rw [hsum]; norm_num

end Claims.PrefillAttnVarlen
