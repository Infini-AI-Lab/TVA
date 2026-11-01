import TVA

/-!
# `attn_fwd_f16`: the model, the conditions, the specification and the assumptions

Source: `source/06-fused-attention.py` (tutorial 06, fused attention, with the config of
`SETUP.md`: `BLOCK_M = 128`, `BLOCK_N = 64`, `num_stages = 2`, `num_warps = 4`, the autotune
decorator removed and its `pre_hook` done by the host before the launch). The kernel is `_attn_fwd`
(lines 176–245) with its helpers `_attn_fwd_inner` (lines 47–110) and `_maybe_make_tensor_desc`
(lines 168–173), in the `float16` run (`FP8_OUTPUT = False`). It reads the module-level names
`triton` and `tl` (lines 20–21) and its helpers.

The host (`_attention.forward`, lines 505–580) launches it on `float16` tensors `q`, `k`, `v` of
shape `[Z, H, N_CTX, HEAD_DIM]` (contiguous), writes the `float16` output `o` (same shape) and the
`float32` log-sum-exp `M` (`[Z, H, N_CTX]`). On the H200 (`is_hopper()`, `supports_host_descriptor()`)
the descriptors are host `TensorDescriptor`s when `warp_specialize` is false, and the raw tensors
(descriptors built in the kernel) when it is true.

* The model: `attn_fwd_inner`, `maybe_make_tensor_desc`, `attn_fwd`, `model`, `launch`. The Lean
  names drop the source's leading underscore.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `attnRef`, `lseRef` and the tolerances
  `tolO`, `tolS`, `tolF`, `tolM`.
* The assumptions: `Assumptions`.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

/-! ## The model -/

/-- What a descriptor parameter of `_attn_fwd` receives: a host `TensorDescriptor` (`host`), or a
`float16` tensor, which the kernel receives as a pointer (`ptr`). -/
inductive DescArg where
  | host (D : TensorDesc .f16)
  | ptr (p : Ptr .f16)

/-- `_maybe_make_tensor_desc` (source lines 168–173), inlined at each call. Line 170's
`isinstance(desc_or_ptr, tl.tensor_descriptor)` is evaluated by Python at compile time: true of a host
`TensorDescriptor`, false of a pointer (SEMANTICS.md, Python's builtins), so only one branch is
visited: line 171 returns the host descriptor; line 173 builds one in the kernel, its `shape` entries
`int32` and its `strides` `int64`, its `block_shape` the call's, the default padding `"zero"`. -/
def maybe_make_tensor_desc (desc_or_ptr : DescArg) (shape : List I32) (strides : List I64)
    (block_shape : Shape) : Prog (TensorDesc .f16) :=
  match desc_or_ptr with
  | .host D => pure D
  | .ptr p => tl.make_tensor_descriptor p shape strides block_shape

/-- The `tl.dot` configuration of lines 73 and 103: `float16` operands into `float32`, no
`input_precision` (the default `tf32`, which Triton ignores for `float16` operands) and no
`max_num_imprecise_acc` (`0`). -/
def cfg16 : DotCfg := { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-- `_attn_fwd_inner` (source lines 47–110), inlined at each call (SEMANTICS.md, helpers are
inlined), with `dtype = tl.float16` (the only value `_attn_fwd` passes on covered launches, line
187), `IS_HOPPER = True` (the H200) and the call's `STAGE`. `offs_m`, `offs_n` and `N_CTX` are
annotated `tl.constexpr` but receive tensors (`_attn_fwd`'s `int32` blocks and its runtime `int32`
`N_CTX`), which they hold (an annotation on a helper's parameter changes nothing). `qk_scale` is the
`float32` scalar of line 219. `BLOCK_M`, `HEAD_DIM`, `BLOCK_N`, `STAGE` are compile-time values
(Lean parameters); `warp_specialize` is used only as `tl.range`'s scheduling keyword (line 69) and
in line 87's condition, whose `not IS_HOPPER` is false, so the model has no parameter for it. -/
def attn_fwd_inner {BLOCK_M HEAD_DIM BLOCK_N : Nat}
    (acc : Tensor [BLOCK_M, HEAD_DIM] F32) (l_i m_i : Tensor [BLOCK_M] F32)
    (q : Tensor [BLOCK_M, HEAD_DIM] F16) (desc_k desc_v : TensorDesc .f16)
    (offset_y start_m : I32) (qk_scale : F32) (STAGE : Nat)
    (offs_m : Tensor [BLOCK_M] I32) (offs_n : Tensor [BLOCK_N] I32) (N_CTX : I32) :
    Prog (Tensor [BLOCK_M, HEAD_DIM] F32 × Tensor [BLOCK_M] F32 × Tensor [BLOCK_M] F32) := do
  -- lines 55–62: a compile-time `if` on `STAGE`; a Python `0` assigned to `lo` holds the `int32`
  -- `+0`; `start_m * BLOCK_M` and `(start_m + 1) * BLOCK_M` are `int32` (the constexprs built in
  -- `int32`), wrapping; line 59's hint checks `lo` where it is defined (line 58)
  let (lo, hi) ← if STAGE = 1 then
      pure ((IntV.ofInt 0 : I32), start_m * (IntV.ofNat BLOCK_M : I32))
    else if STAGE = 2 then do
      let lo := start_m * (IntV.ofNat BLOCK_M : I32)
      let hi := (start_m + (IntV.ofInt 1 : I32)) * (IntV.ofNat BLOCK_M : I32)
      let lo ← tl.multiple_of1 lo BLOCK_M
      pure (lo, hi)
    else
      pure ((IntV.ofInt 0 : I32), N_CTX)
  -- line 63
  let offsetk_y := offset_y + lo
  -- lines 64–67: `dtype == tl.float8e5` is false (compile time): line 67
  let offsetv_y := offset_y + lo
  -- line 69: `tl.range(lo, hi, BLOCK_N)` with `int32` bounds and step (the constexpr), an `int32`
  -- induction variable; the loop carries `acc`, `l_i`, `m_i`, `offsetk_y`, `offsetv_y`
  let (acc, l_i, m_i, _offsetk_y, _offsetv_y) ← tl.forRange lo hi (IntV.ofNat BLOCK_N : I32)
      (acc, l_i, m_i, offsetk_y, offsetv_y) fun start_n (acc, l_i, m_i, offsetk_y, offsetv_y) => do
    -- line 70: a hint on the induction variable, dropped
    -- line 72: the `[BLOCK_N, HEAD_DIM]` block at `[offsetk_y, 0]` (the Python `0` an `int32`),
    -- transposed
    let k : Tensor [HEAD_DIM, BLOCK_N] F16 :=
      (← tl.desc_load (bs := [BLOCK_N, HEAD_DIM]) desc_k [offsetk_y, (IntV.ofInt 0 : I32)]).trans
    -- line 73: `tl.dot(q, k)` with no accumulator: the zero `float32` accumulator
    let qk ← tl.dot cfg16 q k (tl.zeros [BLOCK_M, BLOCK_N] .f32)
    -- lines 74–81: a compile-time `if` on `STAGE`
    let (qk, m_ij) ← if STAGE = 2 then do
        -- line 75: `offs_m[:, None] >= start_n + offs_n[None, :]`, `int32`, signed, broadcast to
        -- `[BLOCK_M, BLOCK_N]`
        let mask := (start_n + offs_n.row BLOCK_M).le (offs_m.col BLOCK_N)
        -- line 76: `qk * qk_scale + tl.where(mask, 0, -1.0e6)`: an addition the product feeds
        -- (`tl.mul_add`); the values of `tl.where` are built in its `float32` dtype (`+0` and
        -- `float32(-1e6)`)
        let qk ← tl.mul_add qk (bc qk_scale)
          (Tensor.where_ mask (bc (FloatV.litF32 (FloatV.ofPat 0)))
            (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000))))
        -- line 77: the row maximum of the `float32` block (no widening), then `tl.maximum`
        let m_ij ← tl.maximum m_i (← tl.max1 true qk)
        -- line 78: `qk = qk - m_ij[:, None]`: no product feeds it (the IEEE `-`)
        let qk ← tl.sub qk (m_ij.col BLOCK_N)
        pure (qk, m_ij)
      else do
        -- line 80: the row maximum times the scalar (an IEEE product: it feeds `tl.maximum`)
        let m_ij ← tl.maximum m_i (← tl.mul (← tl.max1 true qk) (bc qk_scale))
        -- line 81: `qk * qk_scale - m_ij[:, None]`: a subtraction the product feeds (`tl.subc`)
        let qk ← tl.subc (← tl.mul qk (bc qk_scale)) (m_ij.col BLOCK_N)
        pure (qk, m_ij)
    -- line 82
    let p ← tl.exp2 qk
    -- line 84: the IEEE `-` (no product feeds it), then `tl.math.exp2`
    let alpha ← tl.exp2 (← tl.sub m_i m_ij)
    -- line 85: `tl.sum(p, 1)` of a block that is not a product: IEEE additions
    let l_ij ← tl.sum1 p
    -- lines 87–95: `not IS_HOPPER` is false (compile time): line 95, the IEEE product
    let acc ← tl.mul acc (alpha.col HEAD_DIM)
    -- lines 97–100: `dtype == tl.float8e5` is false: line 100, the `[BLOCK_N, HEAD_DIM]` block
    let v ← tl.desc_load (bs := [BLOCK_N, HEAD_DIM]) desc_v [offsetv_y, (IntV.ofInt 0 : I32)]
    -- line 101: `float32` to `float16`, rounded to nearest even
    let p : Tensor [BLOCK_M, BLOCK_N] F16 ← tl.cvt p
    -- line 103: `tl.dot(p, v, acc)`
    let acc ← tl.dot cfg16 p v acc
    -- line 106: `l_i * alpha + l_ij`: an addition the product feeds (`tl.mul_add`)
    let l_i ← tl.mul_add l_i alpha l_ij
    -- line 107
    let m_i := m_ij
    -- lines 108–109: `int32` additions of the constexpr (built in `int32`)
    let offsetk_y := offsetk_y + (IntV.ofNat BLOCK_N : I32)
    let offsetv_y := offsetv_y + (IntV.ofNat BLOCK_N : I32)
    pure (acc, l_i, m_i, offsetk_y, offsetv_y)
  -- line 110
  pure (acc, l_i, m_i)

set_option linter.unusedVariables false in
/-- `_attn_fwd` (source lines 176–245) with `FP8_OUTPUT = False` and `IS_HOPPER = True`, the values
the host passes on covered launches. `sm_scale` is the runtime `float32` scalar (a Python `float`
argument, `FloatV.hostF32` of the double); `M` is `*fp32`; `Z`, `H`, `N_CTX` are `int32` (an
integer argument other than `1`; at `1` the compile-time constant `1`, which gives the same model:
see `model`); the four descriptor parameters are `DescArg`s; `HEAD_DIM`, `BLOCK_M`, `BLOCK_N`,
`STAGE` are compile-time values (Lean parameters); `warp_specialize` reaches only
`_attn_fwd_inner`, whose model does not use it (see there), so `attn_fwd` has no parameter for it. -/
def attn_fwd (sm_scale : F32) (M : Ptr .f32) (Z H : I32)
    (desc_q desc_k desc_v desc_o : DescArg) (N_CTX : I32)
    (HEAD_DIM BLOCK_M BLOCK_N STAGE : Nat) : Kernel where
  body ctx := do
    -- line 187: `dtype = tl.float16` (compile time); line 188: `tl.static_assert(BLOCK_N <= HEAD_DIM)`
    -- holds on covered launches and is dropped
    -- lines 189–190
    let start_m := tl.program_id ctx 0
    let off_hz := tl.program_id ctx 1
    -- lines 191–192: `int32` `//` and `%` (`tl.idiv`, `tl.irem`; undefined on a zero divisor)
    let off_z := (← tl.idiv (Tensor.scalar off_hz) (Tensor.scalar H)).item
    let off_h := (← tl.irem (Tensor.scalar off_hz) (Tensor.scalar H)).item
    -- line 194: `int32` products (wrapping)
    let y_dim := Z * H * N_CTX
    -- lines 195–206: the four calls, in the source's order (`FP8_OUTPUT` false: line 201)
    let desc_q ← maybe_make_tensor_desc desc_q [y_dim, (IntV.ofNat HEAD_DIM : I32)]
      [(IntV.ofNat HEAD_DIM : I64), (IntV.ofNat 1 : I64)] [BLOCK_M, HEAD_DIM]
    let desc_v ← maybe_make_tensor_desc desc_v [y_dim, (IntV.ofNat HEAD_DIM : I32)]
      [(IntV.ofNat HEAD_DIM : I64), (IntV.ofNat 1 : I64)] [BLOCK_N, HEAD_DIM]
    let desc_k ← maybe_make_tensor_desc desc_k [y_dim, (IntV.ofNat HEAD_DIM : I32)]
      [(IntV.ofNat HEAD_DIM : I64), (IntV.ofNat 1 : I64)] [BLOCK_N, HEAD_DIM]
    let desc_o ← maybe_make_tensor_desc desc_o [y_dim, (IntV.ofNat HEAD_DIM : I32)]
      [(IntV.ofNat HEAD_DIM : I64), (IntV.ofNat 1 : I64)] [BLOCK_M, HEAD_DIM]
    -- lines 208–209: `int32` arithmetic (the constexpr `BLOCK_M` built in `int32`)
    let offset_y := off_z * (N_CTX * H) + off_h * N_CTX
    let qo_offset_y := offset_y + start_m * (IntV.ofNat BLOCK_M : I32)
    -- lines 211–212
    let offs_m := start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    let offs_n := tl.arange 0 BLOCK_N
    -- line 214: `float("inf")` is the Python `+∞`, built in `float32` as an operand of `-`
    let m_i ← tl.sub (tl.zeros [BLOCK_M] .f32) (bc (FloatV.litF32 (FloatV.ofPat 0x7FF0000000000000)))
    -- line 215: `1.0` built in `float32`
    let l_i ← tl.add (tl.zeros [BLOCK_M] .f32) (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
    -- line 216
    let acc := tl.zeros [BLOCK_M, HEAD_DIM] .f32
    -- line 218: the name holds the `float32` scalar
    let qk_scale := sm_scale
    -- line 219: `qk_scale * 1.44269504`, the constant built in `float32` (an IEEE product)
    let qk_scale := (← tl.mul (Tensor.scalar qk_scale)
      (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F)))).item
    -- line 221: the `[BLOCK_M, HEAD_DIM]` block at `[qo_offset_y, 0]`
    let q ← tl.desc_load (bs := [BLOCK_M, HEAD_DIM]) desc_q [qo_offset_y, (IntV.ofInt 0 : I32)]
    -- lines 225–231: `STAGE & 1` is computed at compile time; the helper with `STAGE = 4 - STAGE`
    let (acc, l_i, m_i) ← if STAGE &&& 1 ≠ 0 then
        attn_fwd_inner acc l_i m_i q desc_k desc_v offset_y start_m qk_scale (4 - STAGE) offs_m offs_n N_CTX
      else pure (acc, l_i, m_i)
    -- lines 233–239: `STAGE & 2`, the helper with `STAGE = 2`
    let (acc, l_i, m_i) ← if STAGE &&& 2 ≠ 0 then
        attn_fwd_inner acc l_i m_i q desc_k desc_v offset_y start_m qk_scale 2 offs_m offs_n N_CTX
      else pure (acc, l_i, m_i)
    -- line 241: `m_i + tl.math.log2(l_i)`: no product feeds it (the IEEE `+`)
    let m_i ← tl.add m_i (← tl.log2 l_i)
    -- line 242: `float32 / float32`, the platform's division
    let acc ← tl.div acc (l_i.col HEAD_DIM)
    -- line 243: `(M + off_hz * N_CTX) + offs_m`, the `int32` product
    let m_ptrs := M + off_hz * N_CTX + offs_m
    -- line 244: `m_i` is already `float32`, the pointer's dtype
    tl.store m_ptrs m_i
    -- line 245: `acc.to(dtype)`, `float32` to `float16` rounded to nearest even, evaluated before the
    -- store; the block at `[qo_offset_y, 0]`
    let o16 : Tensor [BLOCK_M, HEAD_DIM] F16 ← tl.cvt acc
    tl.desc_store desc_o [qo_offset_y, (IntV.ofInt 0 : I32)] o16

/-! ## The conditions -/

/-- What the host chooses: one launch of `_attn_fwd` through `attention(q, k, v, causal, sm_scale,
warp_specialize)` (`_attention.forward`, lines 505–580), on `float16` tensors. -/
structure Inst where
  /-- `Z` (`q.shape[0]`, the batch). -/
  Z : Nat
  /-- `H` (`q.shape[1]`, the heads). -/
  H : Nat
  /-- `N_CTX` (`q.shape[2]`, the sequence length). -/
  N_CTX : Nat
  /-- `HEAD_DIM` (`k.shape[-1]`, which line 511 asserts equal to `q`'s and `v`'s). -/
  HEAD_DIM : Nat
  /-- `sm_scale`, the Python `float` (a double). -/
  sm_scale : F64
  /-- `causal`: the host passes `STAGE = 3` when true, `1` otherwise (line 514). -/
  causal : Bool
  /-- `warp_specialize`: on the H200 the host passes host `TensorDescriptor`s when it is false and
  the raw tensors when it is true (line 523). -/
  warp_specialize : Bool
  /-- The storage of `q`, contiguous: element `(z, h, n, e)` is element
  `((z·H + h)·N_CTX + n)·HEAD_DIM + e`. -/
  q : Buf .f16
  /-- The storage of `k`, as `q`. -/
  k : Buf .f16
  /-- The storage of `v`, as `q`. -/
  v : Buf .f16
  /-- The storage of `o` (`torch.empty_like(q)`, line 513), as `q`. -/
  o : Buf .f16
  /-- The storage of `M` (`float32`, `[Z, H, N_CTX]`, line 521): element `(z, h, n)` is element
  `(z·H + h)·N_CTX + n`. -/
  M : Buf .f32

/-- The rows of the `[Z·H·N_CTX, HEAD_DIM]` view the descriptors address (line 194's `y_dim`). -/
def rows (d : Inst) : Nat := d.Z * d.H * d.N_CTX

/-- Element `(bh, n, e)` of `q`, `k`, `v` or `o`, for `bh = z·H + h`: `(bh·N_CTX + n)·HEAD_DIM + e`. -/
def idx (d : Inst) (bh n e : Nat) : Nat := (bh * d.N_CTX + n) * d.HEAD_DIM + e

/-- The elements below `n`. -/
def below (n : Nat) : Set Nat := {i | i < n}

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `HEAD_DIM` is `64`, `128` or `256`. The wrapper also allows `16` and `32` (line 512), which line
  188's `tl.static_assert(BLOCK_N <= HEAD_DIM)` rejects on the host (no outcome). -/
  head_dim : d.HEAD_DIM = 64 ∨ d.HEAD_DIM = 128 ∨ d.HEAD_DIM = 256
  /-- `N_CTX` is a positive multiple of `BLOCK_M = 128`: otherwise the last query tile of a head runs
  into the next head's rows (a race on `o`), and the last key block reads the next head's keys. -/
  n_mul : 0 < d.N_CTX ∧ d.N_CTX % 128 = 0
  /-- `N_CTX ≤ 2^16`, which keeps the tolerance meaningful (the host's largest is `2^14`). -/
  n_le : d.N_CTX ≤ 2 ^ 16
  /-- At least one batch and head. -/
  zh_pos : 0 < d.Z ∧ 0 < d.H
  /-- `Z·H < 2^16`: the grid's `y` is `Z·H` (line 549), below CUDA's limit. -/
  zh_lt : d.Z * d.H < 2 ^ 16
  /-- `Z·H·N_CTX < 2^31`: `y_dim` (line 194) and every row offset are `int32` values that do not
  wrap, and the descriptors' shape entries are below `2^31`. -/
  rows_lt : rows d < 2 ^ 31
  /-- `sm_scale` is a finite double in `[2^-64, 2^64]` (the host passes `0.5` and `1.3`). -/
  scale : d.sm_scale.isFinite = true ∧ 1 / 2 ^ 64 ≤ d.sm_scale.toRat ∧
    d.sm_scale.toRat ≤ 2 ^ 64
  /-- The tensors hold `Z·H·N_CTX·HEAD_DIM` elements each, and `M` holds `Z·H·N_CTX`. -/
  lens : rows d * d.HEAD_DIM ≤ d.q.len ∧ rows d * d.HEAD_DIM ≤ d.k.len ∧
    rows d * d.HEAD_DIM ≤ d.v.len ∧ rows d * d.HEAD_DIM ≤ d.o.len ∧ rows d ≤ d.M.len
  /-- The four tensors' storage is 16-byte aligned (TMA's requirement; PyTorch's allocations are). -/
  align : d.q.base % 16 = 0 ∧ d.k.base % 16 = 0 ∧ d.v.base % 16 = 0 ∧ d.o.base % 16 = 0
  /-- `o`'s elements share no byte with `q`'s, `k`'s, `v`'s or `M`'s, and `M`'s none with `q`'s,
  `k`'s or `v`'s (separate tensors). -/
  disj : Disjoint (d.o.elems (below (rows d * d.HEAD_DIM))) (d.q.elems (below (rows d * d.HEAD_DIM))) ∧
    Disjoint (d.o.elems (below (rows d * d.HEAD_DIM))) (d.k.elems (below (rows d * d.HEAD_DIM))) ∧
    Disjoint (d.o.elems (below (rows d * d.HEAD_DIM))) (d.v.elems (below (rows d * d.HEAD_DIM))) ∧
    Disjoint (d.o.elems (below (rows d * d.HEAD_DIM))) (d.M.elems (below (rows d))) ∧
    Disjoint (d.M.elems (below (rows d))) (d.q.elems (below (rows d * d.HEAD_DIM))) ∧
    Disjoint (d.M.elems (below (rows d))) (d.k.elems (below (rows d * d.HEAD_DIM))) ∧
    Disjoint (d.M.elems (below (rows d))) (d.v.elems (below (rows d * d.HEAD_DIM)))

/-- The descriptor parameter for the tensor `b` with block rows `BM`: a host `TensorDescriptor`
(lines 528, 533, 535, 536: `shape = [y_dim, HEAD_DIM]`, `strides = [HEAD_DIM, 1]`, the default
padding `"zero"`, the block shape `[BM, HEAD_DIM]` that `_host_descriptor_pre_hook`, run by the host
at line 557, sets: `[128, HEAD_DIM]` for `q` and `o`, `[64, HEAD_DIM]` for `k` and `v`) when
`warp_specialize` is false, and the tensor itself (lines 538–541) when it is true. -/
def descArg (d : Inst) (b : Buf .f16) (BM : Nat) : DescArg :=
  if d.warp_specialize then .ptr (b.ptr 0)
  else .host ⟨b.ptr 0, [rows d, d.HEAD_DIM], [d.HEAD_DIM, 1], [BM, d.HEAD_DIM], .zero⟩

/-- The model on an instance: the launch of lines 562–574 with `BLOCK_M = 128`, `BLOCK_N = 64`,
`FP8_OUTPUT = False`, `STAGE = 3` when causal and `1` otherwise, `IS_HOPPER = True`. An integer
argument equal to `1` (`Z = 1` in the test) is the compile-time constant `1`: line 194's `Z * H`
builds it in `int32` (an operand of `*` with the `int32` `H`), so the model is the same; likewise
`H = 1` (`off_hz // H`, `off_hz % H`, `N_CTX * H`, `Z * H`, each with an `int32` operand). -/
def model (d : Inst) : Kernel :=
  attn_fwd (FloatV.hostF32 d.sm_scale) (d.M.ptr 0) (IntV.ofNat d.Z) (IntV.ofNat d.H)
    (descArg d d.q 128) (descArg d d.k 64) (descArg d d.v 64) (descArg d d.o 128)
    (IntV.ofNat d.N_CTX) d.HEAD_DIM 128 64 (if d.causal then 3 else 1)

/-- The host's launch (lines 548–549, 562–574): the grid `(cdiv(N_CTX, 128), Z·H, 1)`; the
source's block shapes; `enable_fp_fusion` is Triton's default (`true`); `num_ctas = 1`, no
`launch_pdl`, `debug = false`. `num_warps = 4` and `num_stages = 2` are not recorded
(performance-only and harmless to a well-behaved model); `maxnreg` is given only on Blackwell. -/
def launch (d : Inst) : Launch :=
  { grid := ![(d.N_CTX + 127) / 128, d.Z * d.H, 1], fusion := true,
    blocks := [[128], [64], [128, 1], [1, 64], [128, 64], [d.HEAD_DIM, 64], [64, d.HEAD_DIM],
      [128, d.HEAD_DIM]] }

/-- The inputs covered: the five buffers are valid (aligned, inside `[0, 2^64)`, every byte
present). Any input values: `Post` says what holds for heads whose values are covered (`HeadOk`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k.Valid init ∧ d.v.Valid init ∧ d.o.Valid init ∧ d.M.Valid init

/-! ## The specification -/

/-- The value of element `i` of a `float16` buffer in memory `m` (`0` when absent). -/
def val16 (m : Mem) (b : Buf .f16) (i : Nat) : ℚ :=
  match m.read .f16 (b.ptr (i : Int)).addr with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `float16` buffer holds a finite value in memory `m`. -/
def fin16 (m : Mem) (b : Buf .f16) (i : Nat) : Prop :=
  ∃ x : F16, m.read .f16 (b.ptr (i : Int)).addr = some x ∧ x.isFinite = true

/-- `sm_scale` as a rational number. -/
def sigma (d : Inst) : ℚ := d.sm_scale.toRat

/-- The keys query `n` attends to: `j < N_CTX`, and `j ≤ n` when causal. -/
def keys (d : Inst) (n : Nat) : Finset Nat :=
  (Finset.range d.N_CTX).filter (fun j => d.causal = false ∨ j ≤ n)

/-- The exact score `q[bh, n] · k[bh, j]` of the input values. -/
def score (d : Inst) (m : Mem) (bh n j : Nat) : ℚ :=
  ∑ e ∈ Finset.range d.HEAD_DIM, val16 m d.q (idx d bh n e) * val16 m d.k (idx d bh j e)

/-- The score's magnitude, `Σ_e liftMag q[bh, n, e] · liftMag k[bh, j, e]`: the magnitude the error
of the score's `tl.dot` is relative to (`Rules.Num.DotF16`, `Rules.Num.liftMag`). `liftMag` is `|x|`
for a zero or normal `float16` value and the smallest normal magnitude `2^-14` for a subnormal one
(the tensor core aligns a subnormal operand at it), so for zero and normal values this is
`Σ_e |q·k|`, and in general it is at most `Σ_e |q·k| + 2^-14·(Σ_e |q| + Σ_e |k|) + HEAD_DIM·2^-28`
(`DotLift.sum_liftMag_mul_le`). -/
def scoreMag (d : Inst) (m : Mem) (bh n j : Nat) : ℚ :=
  ∑ e ∈ Finset.range d.HEAD_DIM,
    Rules.Num.liftMag .f16 (val16 m d.q (idx d bh n e)) * Rules.Num.liftMag .f16 (val16 m d.k (idx d bh j e))

/-- The values of head `bh` are covered: every element of `q`, `k`, `v` in the head is a finite
`float16`, every `|v| ≤ 2^15`, and every scaled score magnitude `sm_scale · scoreMag ≤ 2^10`. On the
host's inputs, `sm_scale·Σ_e |q·k| = sm_scale·s²·Σ_e |X_e·Y_e|` with `X`, `Y` independent standard
normals (the test: `s = 0.5`, `sm_scale = 0.5`; the benchmark: `s = 1`, `sm_scale = 1.3`); a Chernoff
bound on the sum of `HEAD_DIM = 128` terms, with a union bound over every pair of the largest shape,
keeps it below `22.7` (test) and `239.9` (benchmark) except with probability below `10^-6`, the means
being about `10` and `106`. The lift adds at most `sm_scale·(2^-14·(Σ_e |q| + Σ_e |k|) + 128·2^-28)`;
with `Σ_e |q| ≤ √128·|q|` and the `χ²` tail of `|q|²` (every row's `|q|² ≤ 288·s²` except with
probability below `3·10^-6`) that is below `0.006` (test) and `0.031` (benchmark). So every
`sm_scale · scoreMag` is below `23` (test) and `240` (benchmark), except with probability below
`4·10^-6`. The bound on `v` leaves room for the
output's tolerance in `float16`; the bound on the scores keeps the causal mask's `-1e6` below every
unmasked score and the tolerance's linear bounds valid. -/
def HeadOk (d : Inst) (m : Mem) (bh : Nat) : Prop :=
  (∀ n e, n < d.N_CTX → e < d.HEAD_DIM →
    fin16 m d.q (idx d bh n e) ∧ fin16 m d.k (idx d bh n e) ∧ fin16 m d.v (idx d bh n e)) ∧
  (∀ n e, n < d.N_CTX → e < d.HEAD_DIM → |val16 m d.v (idx d bh n e)| ≤ 2 ^ 15) ∧
  (∀ n j, n < d.N_CTX → j < d.N_CTX → sigma d * scoreMag d m bh n j ≤ 2 ^ 10)

/-- The exact softmax weight of key `j` for query `n`, unnormalised: `exp(sm_scale · score)`. -/
noncomputable def weight (d : Inst) (m : Mem) (bh n j : Nat) : ℝ :=
  Real.exp ((sigma d : ℝ) * (score d m bh n j : ℝ))

/-- The exact attention output: `Σ_j w_j · v[bh, j, e] / Σ_j w_j` over `keys d n`. -/
noncomputable def attnRef (d : Inst) (m : Mem) (bh n e : Nat) : ℝ :=
  (∑ j ∈ keys d n, weight d m bh n j * (val16 m d.v (idx d bh j e) : ℝ)) /
    ∑ j ∈ keys d n, weight d m bh n j

/-- The same weighted mean of `|v|`: the scale of the output's tolerance. -/
noncomputable def attnAbs (d : Inst) (m : Mem) (bh n e : Nat) : ℝ :=
  (∑ j ∈ keys d n, weight d m bh n j * |(val16 m d.v (idx d bh j e) : ℝ)|) /
    ∑ j ∈ keys d n, weight d m bh n j

/-- `Σ_j |v[bh, j, e]|` over `keys d n`. -/
def vAbsSum (d : Inst) (m : Mem) (bh n e : Nat) : ℚ :=
  ∑ j ∈ keys d n, |val16 m d.v (idx d bh j e)|

/-- The exact log-sum-exp in base 2: `log₂ Σ_j exp(sm_scale · score)`. -/
noncomputable def lseRef (d : Inst) (m : Mem) (bh n : Nat) : ℝ :=
  Real.logb 2 (∑ j ∈ keys d n, weight d m bh n j)

/-! ### The tolerances

`X` bounds the row's scaled score magnitudes (`sm_scale · scoreMag ≤ X` over its keys), and `D`,
`N` are `HEAD_DIM` and `N_CTX`. The constants are the library's: `relDot16` (`Rules.Num.DotF16`),
`2^-22` (`Exp2F32`, `DivFull`, `Log2F32`), and one rounding of `float32` (`2^-24`) and of `float16`
(`2^-11`). -/

/-- One rounding of `float32`, relative: `2^-24`. -/
def u32 : ℚ := 1 / 2 ^ 24

/-- The bound `3X/2` on `qk_scale · scoreMag`, in the base-2 exponent (`qk_scale` is
`sm_scale · log₂ e` up to `2^-22` relative). -/
def ybound (X : ℚ) : ℚ := 3 / 2 * X

/-- The error of each computed exponent `qk·qk_scale − m` against the exact one: the score's
`tl.dot` error (`relDot16 D` of `scoreMag`) and the roundings. -/
def expErr (D : Nat) (X : ℚ) : ℚ := (Rules.Num.relDot16 D + 1 / 2 ^ 20) * ybound X + 1 / 2 ^ 50

/-- The relative error of each `p = exp2(·)` against its exact weight. -/
def psiP (D : Nat) (X : ℚ) : ℚ := (1 + expErr D X) * (1 + 1 / 2 ^ 22) - 1

/-- The same after `p.to(float16)`. -/
def psi16 (D : Nat) (X : ℚ) : ℚ := (1 + psiP D X) * (1 + 1 / 2 ^ 11) - 1

/-- The relative error of a block's row sum `l_ij` (a tree of `63` additions over the `p`s). -/
def psiL (D : Nat) (X : ℚ) : ℚ := (1 + u32) ^ 63 * (1 + psiP D X) - 1

/-- The growth of the running maximum's rounding over the loop: `exp(u·(m + Y + E))` with
`|m| ≤ Y + E`, at most `1 + 4u·(Y + E)`. -/
def drift (D : Nat) (X : ℚ) : ℚ := 1 + 4 * u32 * (ybound X + expErr D X)

/-- The factor by which each step can scale the accumulator's earlier content: `tl.dot`'s
`relDot16 64`, `2^-150` from alpha's input, alpha's `2^-22` and the rounding of `acc * alpha`. -/
def stepAcc : ℚ := (1 + Rules.Num.relDot16 64) * (1 + 1 / 2 ^ 150) * ((1 + 1 / 2 ^ 22) * (1 + u32))

/-- The same for `l_i`: `tl.mul_add`'s `3u`, alpha's input and alpha's `2^-22`. -/
def stepL : ℚ := (1 + 3 * u32) * (1 + 1 / 2 ^ 150) * (1 + 1 / 2 ^ 22)

/-- The number of key blocks a row goes through, at most: `N / 64`. -/
def nblocks (N : Nat) : Nat := N / 64

/-- The relative bound on the accumulator after the loop. -/
def facc (D N : Nat) (X : ℚ) : ℚ := (1 + psi16 D X) * drift D X * stepAcc ^ nblocks N

/-- The relative bound on `l_i` after the loop. -/
def fl (D N : Nat) (X : ℚ) : ℚ := (1 + psiL D X) * drift D X * stepL ^ nblocks N

/-- The absolute error of `l_i`, per block (the masked lanes, the `2^-126` of each `exp2`, the
roundings near zero). -/
def absL : ℚ := 1 / 2 ^ 100

/-- The absolute error of the accumulator per block, besides `omega0·|v|` per key and `kapA` per
unit of weight: the `p·v` `tl.dot`'s `relDot16 64 = 66·2^-22` times the rest of the lift of a
block's operands (`2^-14·(2^-25 + 2^-125) + 2^-28` per key, at most `2^-22 + 2^-32` over the block's
`64` keys), about `0.52·2^-37`, with the roundings near zero. -/
def absA : ℚ := 1 / 2 ^ 37

/-- The absolute error of the accumulator per unit of weight: the `p·v` `tl.dot`'s `relDot16 64`
times the lift `2^-14·|p|` of a subnormal `v` (counted at `2^-14`), with
`|p| ≤ (1 + psi16)·weight + 2^-25 + 2^-125` and `psi16 ≤ 1/7`: `66·2^-36·8/7 ≤ 2^-29`. Over the
keys it is `kapA` times `l_i`'s exact value, so it adds about `kapA` to `acc / l_i`. -/
def kapA : ℚ := 1 / 2 ^ 29

/-- The absolute error per key, times `|v|`: half `float16`'s smallest subnormal from
`p.to(float16)` (`2^-25`, and more), and the `p·v` `tl.dot`'s `relDot16 64` times the lift `2^-14` of
a subnormal `p` (`66·2^-36`). -/
def omega0 : ℚ := 1 / 2 ^ 25 + 1 / 2 ^ 125 + 66 / 2 ^ 36

/-- The lower bound `1 − E` on the sum of a row's weights against its maximum. -/
def wlow (D : Nat) (X : ℚ) : ℚ := 1 - expErr D X

/-- `l_i`'s relative error after the loop, the absolute part folded in. -/
def betaL (D N : Nat) (X : ℚ) : ℚ := (fl D N X - 1) + fl D N X * (nblocks N * absL) / wlow D X

/-- The denominator `1 − betaL` of the division's error. -/
def denQ (D N : Nat) (X : ℚ) : ℚ := 1 - betaL D N X

/-- The relative error of `acc / l_i` against the kernel's own weighted mean. -/
def relQ (D N : Nat) (X : ℚ) : ℚ := (facc D N X - 1 + betaL D N X) / denQ D N X

/-- The coefficient of `Σ|v|` in the error of `acc / l_i`. -/
def absQS (D N : Nat) (X : ℚ) : ℚ := facc D N X * omega0 / (wlow D X * denQ D N X)

/-- The absolute error of `acc / l_i`: `absA` per block, and `kapA` per unit of weight. -/
def absQ0 (D N : Nat) (X : ℚ) : ℚ :=
  facc D N X * (nblocks N * absA) / (wlow D X * denQ D N X) + facc D N X * kapA / denQ D N X

/-- The relative change of the weights from `qk_scale · ln 2` to `sm_scale`: `4θ` with
`θ = 2^-22·X`. -/
def thetaW (X : ℚ) : ℚ := 1 / 2 ^ 20 * X

/-- The last two roundings (`/` and `.to(float16)`), relative. -/
def lastRel : ℚ := 1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22

/-- **The output's tolerance relative to `attnAbs`**, the weighted mean of `|v|`. -/
def tolO (D N : Nat) (X : ℚ) : ℚ :=
  lastRel * (1 + relQ D N X) * (1 + thetaW X) + relQ D N X * (1 + thetaW X) + thetaW X

/-- **The output's tolerance relative to `vAbsSum`** (`Σ|v|` over the row's keys): the `float16`
rounding of small `p`s and the lift of subnormal `p`s in the `p·v` `tl.dot` (`omega0`, about
`2^-25` per key). -/
def tolS (D N : Nat) (X : ℚ) : ℚ := absQS D N X * (1 + lastRel)

/-- **The output's absolute tolerance**: mainly the output's `float16` rounding near zero (`2^-25`),
and `absQ0`, the lift of a subnormal `v` in the `p·v` `tl.dot` (`kapA`, about `2^-29`, and `absA`
per block over `N/64` blocks). -/
def tolF (D N : Nat) (X : ℚ) : ℚ :=
  absQ0 D N X * (1 + lastRel) + 1 / 2 ^ 11 * (1 / 2 ^ 148) + 1 / 2 ^ 25 + 1 / 2 ^ 148

/-- **`M`'s tolerance**, besides `2^-24·|lseRef|`: `tl.math.log2`'s `2^-22·|log₂ l_i|` with
`|log₂ l_i| ≤ 17`, `log₂` of `l_i`'s relative error (at most twice it), the change from `qk_scale`
to `sm_scale·log₂ e` (`2^-21·X`), `log2`'s absolute `2^-149`, and one rounding. -/
def tolM (D N : Nat) (X : ℚ) : ℚ :=
  (1 + u32) * (17 / 2 ^ 22 + 2 * betaL D N X + 1 / 2 ^ 21 * X + 1 / 2 ^ 149) + 1 / 2 ^ 150

/-- The bytes the launch may write: the elements of `o` and of `M`. -/
def outBytes (d : Inst) : Set Int :=
  d.o.elems (below (rows d * d.HEAD_DIM)) ∪ d.M.elems (below (rows d))

/-- What the launch leaves:
* every element `(bh, n, e)` of `o` holds a `float16` value `c`; when head `bh`'s values are covered
  (`HeadOk`), `c` is finite and, for every bound `X ≤ 2^10` on the row's scaled score magnitudes,
  within `tolO·attnAbs + tolS·vAbsSum + tolF` of the exact attention `attnRef`;
* every element `(bh, n)` of `M` holds a `float32` value `c`; for a covered head, `c` is finite and
  within `tolM + 2^-24·|lseRef|` of the exact log-sum-exp `lseRef` (base 2);
* every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ bh n e, bh < d.Z * d.H → n < d.N_CTX → e < d.HEAD_DIM →
    ∃ c : F16, fin.read .f16 (d.o.ptr (idx d bh n e : Int)).addr = some c ∧
      (HeadOk d init bh → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
        (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
        |c.toReal - attnRef d init bh n e| ≤
          (tolO d.HEAD_DIM d.N_CTX X : ℝ) * attnAbs d init bh n e +
            (tolS d.HEAD_DIM d.N_CTX X : ℝ) * (vAbsSum d init bh n e : ℝ) + (tolF d.HEAD_DIM d.N_CTX X : ℝ))) ∧
  (∀ bh n, bh < d.Z * d.H → n < d.N_CTX →
    ∃ c : F32, fin.read .f32 (d.M.ptr ((bh * d.N_CTX + n : Nat) : Int)).addr = some c ∧
      (HeadOk d init bh → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
        (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
        |c.toReal - lseRef d init bh n| ≤
          (tolM d.HEAD_DIM d.N_CTX X : ℝ) + (u32 : ℝ) * |lseRef d init bh n|)) ∧
  ∀ z, z ∉ outBytes d → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotF16`: `tl.dot` of `float16` operands into `float32` (lines 73 and 103). -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (lines 73, 103), of configuration `cfg16` (written
  out), always has a result. -/
  dot_total : Rules.Num.DotTotal P
    { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
  /-- [rule] `Num.Exp2F32`: `tl.math.exp2` on `float32` (lines 82, 84). -/
  exp2 : Rules.Num.Exp2F32 P
  /-- [rule] `Num.Exp2F32NegInf`: `tl.math.exp2(-inf) = +0` on `float32` (line 84, first block). -/
  exp2_ninf : Rules.Num.Exp2F32NegInf P
  /-- [rule] `Num.MathTotal`: `tl.math.exp2` on `float32` has a result for every input. -/
  exp2_total : Rules.Num.MathTotal P .exp2 .binary32
  /-- [rule] `Num.Log2F32`: `tl.math.log2` on `float32` (line 241). -/
  log2 : Rules.Num.Log2F32 P
  /-- [rule] `Num.MathTotal`: `tl.math.log2` on `float32` has a result for every input. -/
  log2_total : Rules.Num.MathTotal P .log2 .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32` (line 242). -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs. -/
  div_total : Rules.Num.DivTotal P .binary32

end Kernels.AttnFwd
