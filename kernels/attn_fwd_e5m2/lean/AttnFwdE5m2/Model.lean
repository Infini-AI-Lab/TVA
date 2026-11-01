import TVA

/-!
# `attn_fwd_e5m2`: the model, the conditions, the specification and the assumptions

Source: `source/06-fused-attention.py` (tutorial 06, fused attention, with the config of
`SETUP.md`: `BLOCK_M = 128`, `BLOCK_N = 64`, `num_stages = 2`, `num_warps = 4`, the autotune
decorator removed and its `pre_hook` done by the host before the launch). The kernel is `_attn_fwd`
(lines 176–245) with its helpers `_attn_fwd_inner` (lines 47–110) and `_maybe_make_tensor_desc`
(lines 168–173), in the `float8e5` run (`FP8_OUTPUT = True`). It reads the module-level names
`triton` and `tl` (lines 20–21) and its helpers.

The host (`_attention.forward`, lines 505–580) launches it on `float8e5` tensors `q`, `k` of shape
`[Z, H, N_CTX, HEAD_DIM]` (contiguous) and `v`, the `[Z, H, N_CTX, HEAD_DIM]` view of a contiguous
`[Z, H, HEAD_DIM, N_CTX]` tensor (the test's and the benchmark's `v.permute(0, 1, 3, 2).contiguous()
.permute(0, 1, 3, 2)`); it writes the `float8e5` output `o` (`torch.empty_like(q)`) and the
`float32` log-sum-exp `M` (`[Z, H, N_CTX]`). On the H200 the descriptors are host
`TensorDescriptor`s when `warp_specialize` is false, and the raw tensors (descriptors built in the
kernel) when it is true.

* The model: `attn_fwd_inner`, `maybe_make_tensor_desc`, `attn_fwd`, `model`, `launch`. The Lean
  names drop the source's leading underscore.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `attnRef`, `lseRef` and the tolerances
  `tolO`, `tolM`.
* The assumptions: `Assumptions`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The model -/

/-- What a descriptor parameter of `_attn_fwd` receives: a host `TensorDescriptor` (`host`), or a
`float8e5` tensor, which the kernel receives as a pointer (`ptr`). -/
inductive DescArg where
  | host (D : TensorDesc .f8e5)
  | ptr (p : Ptr .f8e5)

/-- `_maybe_make_tensor_desc` (source lines 168–173), inlined at each call. Line 170's
`isinstance(desc_or_ptr, tl.tensor_descriptor)` is evaluated by Python at compile time: true of a host
`TensorDescriptor`, false of a pointer (SEMANTICS.md, Python's builtins), so only one branch is
visited: line 171 returns the host descriptor; line 173 builds one in the kernel, its `shape` entries
`int32` and its `strides` `int64`, its `block_shape` the call's, the default padding `"zero"`. -/
def maybe_make_tensor_desc (desc_or_ptr : DescArg) (shape : List I32) (strides : List I64)
    (block_shape : Shape) : Prog (TensorDesc .f8e5) :=
  match desc_or_ptr with
  | .host D => pure D
  | .ptr p => tl.make_tensor_descriptor p shape strides block_shape

/-- The `tl.dot` configuration of lines 73 and 103: `float8e5` operands into `float32`, no
`input_precision` (the default `tf32`) and no `max_num_imprecise_acc` (for an fp8 × fp8 dot on
sm_90 the default is `2^30`, SEMANTICS.md, `tl.dot`). -/
def cfg8 : DotCfg :=
  { a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32, maxImpreciseAcc := 2 ^ 30 }

/-- `_attn_fwd_inner` (source lines 47–110), inlined at each call (SEMANTICS.md, helpers are
inlined), with `dtype = tl.float8e5` (the only value `_attn_fwd` passes on covered launches, line
187), `IS_HOPPER = True` (the H200) and the call's `STAGE`. `offs_m`, `offs_n` and `N_CTX` are
annotated `tl.constexpr` but receive tensors (`_attn_fwd`'s `int32` blocks and its runtime `int32`
`N_CTX`), which they hold (an annotation on a helper's parameter changes nothing). `qk_scale` is the
`float32` scalar of line 219. `BLOCK_M`, `HEAD_DIM`, `BLOCK_N`, `STAGE` are compile-time values
(Lean parameters); `warp_specialize` is used only as `tl.range`'s scheduling keyword (line 69) and
in line 87's condition, whose `not IS_HOPPER` is false, so the model has no parameter for it. -/
def attn_fwd_inner {BLOCK_M HEAD_DIM BLOCK_N : Nat}
    (acc : Tensor [BLOCK_M, HEAD_DIM] F32) (l_i m_i : Tensor [BLOCK_M] F32)
    (q : Tensor [BLOCK_M, HEAD_DIM] F8E5) (desc_k desc_v : TensorDesc .f8e5)
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
  -- lines 64–65: `dtype == tl.float8e5` is true (compile time): line 65, the `int32` product with
  -- the constexpr (built in `int32`), then the `int32` `+`, wrapping
  let offsetv_y := offset_y * (IntV.ofNat HEAD_DIM : I32) + lo
  -- line 69: `tl.range(lo, hi, BLOCK_N)` with `int32` bounds and step (the constexpr), an `int32`
  -- induction variable; the loop carries `acc`, `l_i`, `m_i`, `offsetk_y`, `offsetv_y`
  let (acc, l_i, m_i, _offsetk_y, _offsetv_y) ← tl.forRange lo hi (IntV.ofNat BLOCK_N : I32)
      (acc, l_i, m_i, offsetk_y, offsetv_y) fun start_n (acc, l_i, m_i, offsetk_y, offsetv_y) => do
    -- line 70: a hint on the induction variable, dropped
    -- line 72: the `[BLOCK_N, HEAD_DIM]` block at `[offsetk_y, 0]` (the Python `0` an `int32`),
    -- transposed
    let k : Tensor [HEAD_DIM, BLOCK_N] F8E5 :=
      (← tl.desc_load (bs := [BLOCK_N, HEAD_DIM]) desc_k [offsetk_y, (IntV.ofInt 0 : I32)]).trans
    -- line 73: `tl.dot(q, k)` with no accumulator: the zero `float32` accumulator
    let qk ← tl.dot cfg8 q k (tl.zeros [BLOCK_M, BLOCK_N] .f32)
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
    -- lines 97–98: `dtype == tl.float8e5` is true: line 98, the `[HEAD_DIM, BLOCK_N]` block (the
    -- descriptor's block shape) at `[0, offsetv_y]` (the Python `0` an `int32`), transposed
    let v : Tensor [BLOCK_N, HEAD_DIM] F8E5 :=
      (← tl.desc_load (bs := [HEAD_DIM, BLOCK_N]) desc_v [(IntV.ofInt 0 : I32), offsetv_y]).trans
    -- line 101: `p.to(tl.float8e5)`, `float32` to `float8e5` with the default rounding (`"rtne"`),
    -- the platform's conversion
    let p : Tensor [BLOCK_M, BLOCK_N] F8E5 ← tl.cvt8 p
    -- line 103: `tl.dot(p, v, acc)`
    let acc ← tl.dot cfg8 p v acc
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
/-- `_attn_fwd` (source lines 176–245) with `FP8_OUTPUT = True` and `IS_HOPPER = True`, the values
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
    -- line 187: `dtype = tl.float8e5` (compile time); line 188: `tl.static_assert(BLOCK_N <= HEAD_DIM)`
    -- holds on covered launches and is dropped
    -- lines 189–190
    let start_m := tl.program_id ctx 0
    let off_hz := tl.program_id ctx 1
    -- lines 191–192: `int32` `//` and `%` (`tl.idiv`, `tl.irem`; undefined on a zero divisor)
    let off_z := (← tl.idiv (Tensor.scalar off_hz) (Tensor.scalar H)).item
    let off_h := (← tl.irem (Tensor.scalar off_hz) (Tensor.scalar H)).item
    -- line 194: `int32` products (wrapping)
    let y_dim := Z * H * N_CTX
    -- lines 195–206: the four calls, in the source's order (`FP8_OUTPUT` true: lines 198–199)
    let desc_q ← maybe_make_tensor_desc desc_q [y_dim, (IntV.ofNat HEAD_DIM : I32)]
      [(IntV.ofNat HEAD_DIM : I64), (IntV.ofNat 1 : I64)] [BLOCK_M, HEAD_DIM]
    let desc_v ← maybe_make_tensor_desc desc_v [(IntV.ofNat HEAD_DIM : I32), y_dim]
      [(IntV.cast N_CTX : I64), (IntV.ofNat 1 : I64)] [HEAD_DIM, BLOCK_N]
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
    -- line 245: `acc.to(dtype)`, `float32` to `float8e5` with the default rounding, the platform's
    -- conversion, evaluated before the store; the block at `[qo_offset_y, 0]`
    let o8 : Tensor [BLOCK_M, HEAD_DIM] F8E5 ← tl.cvt8 acc
    tl.desc_store desc_o [qo_offset_y, (IntV.ofInt 0 : I32)] o8

/-! ## The conditions -/

/-- What the host chooses: one launch of `_attn_fwd` through `attention(q, k, v, causal, sm_scale,
warp_specialize)` (`_attention.forward`, lines 505–580), on `float8e5` tensors. -/
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
  q : Buf .f8e5
  /-- The storage of `k`, as `q`. -/
  k : Buf .f8e5
  /-- The storage of `v`, transposed: element `(z, h, n, e)` of the `[Z, H, N_CTX, HEAD_DIM]` view is
  element `((z·H + h)·HEAD_DIM + e)·N_CTX + n` of the contiguous `[Z, H, HEAD_DIM, N_CTX]` tensor. -/
  v : Buf .f8e5
  /-- The storage of `o` (`torch.empty_like(q)`, line 513), as `q`. -/
  o : Buf .f8e5
  /-- The storage of `M` (`float32`, `[Z, H, N_CTX]`, line 521): element `(z, h, n)` is element
  `(z·H + h)·N_CTX + n`. -/
  M : Buf .f32

/-- The rows of the `[Z·H·N_CTX, HEAD_DIM]` view the descriptors of `q`, `k`, `o` address (line
194's `y_dim`). -/
def rows (d : Inst) : Nat := d.Z * d.H * d.N_CTX

/-- Element `(bh, n, e)` of `q`, `k` or `o`, for `bh = z·H + h`: `(bh·N_CTX + n)·HEAD_DIM + e`. -/
def idx (d : Inst) (bh n e : Nat) : Nat := (bh * d.N_CTX + n) * d.HEAD_DIM + e

/-- Element `(bh, n, e)` of `v` (transposed storage): `(bh·HEAD_DIM + e)·N_CTX + n`. -/
def vIdx (d : Inst) (bh n e : Nat) : Nat := (bh * d.HEAD_DIM + e) * d.N_CTX + n

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
  /-- `N_CTX ≤ 2^16` (the host's largest is `2^14`). -/
  n_le : d.N_CTX ≤ 2 ^ 16
  /-- At least one batch and head. -/
  zh_pos : 0 < d.Z ∧ 0 < d.H
  /-- `Z·H < 2^16`: the grid's `y` is `Z·H` (line 549), below CUDA's limit. -/
  zh_lt : d.Z * d.H < 2 ^ 16
  /-- `Z·H·N_CTX·HEAD_DIM < 2^31`: line 65's `offsetv_y = offset_y·HEAD_DIM + lo` (and its steps,
  line 109) is an `int32` value that does not wrap; with it `y_dim` (line 194) and every row offset
  (lines 208–209, 243) do not wrap, and the descriptors' shape entries are below `2^31`. -/
  elems_lt : rows d * d.HEAD_DIM < 2 ^ 31
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

/-- The descriptor parameter for the row-major tensor `b` (`q`, `k`, `o`) with block rows `BM`: a
host `TensorDescriptor` (lines 528, 535, 536: `shape = [y_dim, HEAD_DIM]`, `strides = [HEAD_DIM, 1]`,
the default padding `"zero"`, the block shape `[BM, HEAD_DIM]` that `_host_descriptor_pre_hook`, run
by the host at line 557, sets: `[128, HEAD_DIM]` for `q` and `o`, `[64, HEAD_DIM]` for `k`) when
`warp_specialize` is false, and the tensor itself (lines 538–541) when it is true. -/
def descArg (d : Inst) (b : Buf .f8e5) (BM : Nat) : DescArg :=
  if d.warp_specialize then .ptr (b.ptr 0)
  else .host ⟨b.ptr 0, [rows d, d.HEAD_DIM], [d.HEAD_DIM, 1], [BM, d.HEAD_DIM], .zero⟩

/-- The descriptor parameter for `v`: the host `TensorDescriptor` of line 530 (`q.dtype` is
`float8_e5m2`: `shape = [HEAD_DIM, y_dim]`, `strides = [N_CTX, 1]`, the default padding `"zero"`,
the block shape `[HEAD_DIM, 64]` that the pre-hook sets when `FP8_OUTPUT`, line 121) when
`warp_specialize` is false, and the tensor `v` itself (line 539) when it is true. -/
def descArgV (d : Inst) : DescArg :=
  if d.warp_specialize then .ptr (d.v.ptr 0)
  else .host ⟨d.v.ptr 0, [d.HEAD_DIM, rows d], [d.N_CTX, 1], [d.HEAD_DIM, 64], .zero⟩

/-- The model on an instance: the launch of lines 562–574 with `BLOCK_M = 128`, `BLOCK_N = 64`,
`FP8_OUTPUT = True`, `STAGE = 3` when causal and `1` otherwise, `IS_HOPPER = True`. An integer
argument equal to `1` (`Z = 1` in the test) is the compile-time constant `1`: line 194's `Z * H`
builds it in `int32` (an operand of `*` with the `int32` `H`), so the model is the same; likewise
`H = 1` (`off_hz // H`, `off_hz % H`, `N_CTX * H`, `Z * H`, each with an `int32` operand). -/
def model (d : Inst) : Kernel :=
  attn_fwd (FloatV.hostF32 d.sm_scale) (d.M.ptr 0) (IntV.ofNat d.Z) (IntV.ofNat d.H)
    (descArg d d.q 128) (descArg d d.k 64) (descArgV d) (descArg d d.o 128)
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

/-- The value of element `i` of a `float8e5` buffer in memory `m` (`0` when absent). -/
def val8 (m : Mem) (b : Buf .f8e5) (i : Nat) : ℚ :=
  match m.read .f8e5 (b.ptr (i : Int)).addr with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `float8e5` buffer holds a finite value in memory `m`. -/
def fin8 (m : Mem) (b : Buf .f8e5) (i : Nat) : Prop :=
  ∃ x : F8E5, m.read .f8e5 (b.ptr (i : Int)).addr = some x ∧ x.isFinite = true

/-- `sm_scale` as a rational number. -/
def sigma (d : Inst) : ℚ := d.sm_scale.toRat

/-- The keys query `n` attends to: `j < N_CTX`, and `j ≤ n` when causal. -/
def keys (d : Inst) (n : Nat) : Finset Nat :=
  (Finset.range d.N_CTX).filter (fun j => d.causal = false ∨ j ≤ n)

/-- The exact score `q[bh, n] · k[bh, j]` of the input values. -/
def score (d : Inst) (m : Mem) (bh n j : Nat) : ℚ :=
  ∑ e ∈ Finset.range d.HEAD_DIM, val8 m d.q (idx d bh n e) * val8 m d.k (idx d bh j e)

/-- Row `n` of head `bh` of `q`: the score's `tl.dot` operand values. -/
def qrow (d : Inst) (m : Mem) (bh n : Nat) : Fin d.HEAD_DIM → ℚ := fun e => val8 m d.q (idx d bh n e)

/-- Row `j` of head `bh` of `k`. -/
def krow (d : Inst) (m : Mem) (bh j : Nat) : Fin d.HEAD_DIM → ℚ := fun e => val8 m d.k (idx d bh j e)

/-- **The lifted score magnitude** `Σ_e liftMag(q[bh, n, e])·liftMag(k[bh, j, e])`: `Σ_e |q·k|`
with a subnormal operand counted at the smallest normal `2^-14` (`Rules.Num.liftMag`, the magnitude
the tensor core aligns it by). For zero and normal values it is `Σ_e |q·k|`. -/
def scoreLift (d : Inst) (m : Mem) (bh n j : Nat) : ℚ :=
  ∑ e, Rules.Num.liftMag .f8e5 (qrow d m bh n e) * Rules.Num.liftMag .f8e5 (krow d m bh j e)

/-- **The score's error bound** from the chunk facts (`Num.DotE5m2Chunk`): `Dot8Chunk.errB` over
the `⌈HEAD_DIM/32⌉` chunks of `q·k` from a zero accumulator, from each chunk's largest lifted
product (`Dot8Chunk.chunkLift8`) and the exact partial sums (`Dot8Chunk.DotF8Chunk.err`). -/
def scoreErr (d : Inst) (m : Mem) (bh n j : Nat) : ℚ :=
  Dot8Chunk.errB Rules.Num.rhoChunk8 Rules.Num.sigmaChunk8 Rules.Num.alphaChunk8 0
    (Rules.Num.chunkSum8 (qrow d m bh n) (krow d m bh j))
    (Dot8Chunk.chunkLift8 .f8e5 .f8e5 (qrow d m bh n) (krow d m bh j)) (Rules.Num.chunks8 d.HEAD_DIM)

/-- **Head `bh`'s `v` is read**: `bh·HEAD_DIM < Z·H`. Line 98 loads `v` through `desc_v`, whose shape
is `[HEAD_DIM, y_dim]` with `y_dim = Z·H·N_CTX` (lines 198, 530), at the column
`offsetv_y = bh·N_CTX·HEAD_DIM + lo` (line 65, `offset_y = bh·N_CTX`). The columns of head `bh`'s keys,
`bh·N_CTX·HEAD_DIM + j` (`j < N_CTX`), lie inside the descriptor exactly when `bh·HEAD_DIM < Z·H`;
otherwise every lane of every `v` block of the head is outside it and holds the padding `+0`
(SEMANTICS.md, tensor descriptors). -/
def vIn (d : Inst) (bh : Nat) : Prop := bh * d.HEAD_DIM < d.Z * d.H

/-- **The values of `v` the kernel uses** for key `j`, column `e` of head `bh`: `v[bh, j, e]` when
head `bh`'s `v` is read (`vIn`), and `0` otherwise. -/
def vEff (d : Inst) (m : Mem) (bh j e : Nat) : ℚ :=
  if bh * d.HEAD_DIM < d.Z * d.H then val8 m d.v (vIdx d bh j e) else 0

/-- The bound on a covered head's scaled lifted score magnitudes: `2^10 / ⌈HEAD_DIM/32⌉` (`512`,
`256`, `128` at `HEAD_DIM = 64, 128, 256`), which keeps every exponent's error bound `expErr` below
`13`. -/
def Xh (D : Nat) : ℚ := 2 ^ 10 / Rules.Num.chunks8 D

/-- The values of head `bh` are covered: every element of `q` and `k` in the head is a finite
`float8e5`, so is every element of `v` in the head when the head's `v` is read (`vIn`), and every
scaled lifted score magnitude `sm_scale · scoreLift` is at most `Xh HEAD_DIM` (the host's inputs:
`normal(0, 0.5)` cast to `float8e5` with `sm_scale = 0.5`, about `16` at most at `HEAD_DIM = 128`,
and `randn` with `sm_scale = 1.3`, about `200` at most). -/
def HeadOk (d : Inst) (m : Mem) (bh : Nat) : Prop :=
  (∀ n e, n < d.N_CTX → e < d.HEAD_DIM → fin8 m d.q (idx d bh n e) ∧ fin8 m d.k (idx d bh n e)) ∧
  (∀ n e, n < d.N_CTX → e < d.HEAD_DIM → vIn d bh → fin8 m d.v (vIdx d bh n e)) ∧
  (∀ n j, n < d.N_CTX → j < d.N_CTX → sigma d * scoreLift d m bh n j ≤ Xh d.HEAD_DIM)

/-- The exact softmax weight of key `j` for query `n`, unnormalised: `exp(sm_scale · score)`. -/
noncomputable def weight (d : Inst) (m : Mem) (bh n j : Nat) : ℝ :=
  Real.exp ((sigma d : ℝ) * (score d m bh n j : ℝ))

/-- The sum of the weights of query `n`'s keys. -/
noncomputable def wsum (d : Inst) (m : Mem) (bh n : Nat) : ℝ :=
  ∑ j ∈ keys d n, weight d m bh n j

/-- **The exact attention output of the values the kernel uses**: `Σ_j w_j · vEff[bh, j, e] / Σ_j w_j`
over `keys d n`. For a head whose `v` is read (`vIn`) it is the attention of `q`, `k`, `v`; for the
other heads it is `0`. -/
noncomputable def attnRef (d : Inst) (m : Mem) (bh n e : Nat) : ℝ :=
  (∑ j ∈ keys d n, weight d m bh n j * (vEff d m bh j e : ℝ)) / wsum d m bh n

/-- The exact log-sum-exp in base 2: `log₂ Σ_j exp(sm_scale · score)`. -/
noncomputable def lseRef (d : Inst) (m : Mem) (bh n : Nat) : ℝ :=
  Real.logb 2 (wsum d m bh n)

/-! ### The order of the loop

Row `n` goes through the key blocks `t = 0, 1, …, nblk d n − 1` (keys `64t … 64t + 63`), in this
order: all `N_CTX/64` blocks, or, when causal, the `2·(n/128)` blocks below its query tile and the
tile's own two blocks. -/

/-- The number of key blocks row `n` goes through. -/
def nblk (d : Inst) (n : Nat) : Nat := if d.causal then 2 * (n / 128) + 2 else d.N_CTX / 64

/-- The keys of row `n` in the blocks before block `t`. -/
def keysBelow (d : Inst) (n t : Nat) : Finset Nat := (keys d n).filter (fun j => j < 64 * t)

/-- The keys of row `n` in block `t`. -/
def keysBlk (d : Inst) (n t : Nat) : Finset Nat :=
  (keys d n).filter (fun j => 64 * t ≤ j ∧ j < 64 * t + 64)

/-- The exact weights' sum over the keys of the blocks before block `t`. -/
noncomputable def partW (d : Inst) (m : Mem) (bh n t : Nat) : ℝ :=
  ∑ j ∈ keysBelow d n t, weight d m bh n j

/-- **The centred partial sum** `Σ w_j·(vEff_j − attnRef)` over the keys of the blocks before block
`t`. After the last block it is `0`. -/
noncomputable def centC (d : Inst) (m : Mem) (bh n e t : Nat) : ℝ :=
  ∑ j ∈ keysBelow d n t, weight d m bh n j * ((vEff d m bh j e : ℝ) - attnRef d m bh n e)

/-- `Σ w_j·|vEff_j − attnRef|` over the keys of block `t`. -/
noncomputable def blkC (d : Inst) (m : Mem) (bh n e t : Nat) : ℝ :=
  ∑ j ∈ keysBlk d n t, weight d m bh n j * |(vEff d m bh j e : ℝ) - attnRef d m bh n e|

/-- `Σ |vEff_j − attnRef|` over the keys of block `t`. -/
noncomputable def blkA (d : Inst) (m : Mem) (bh n e t : Nat) : ℝ :=
  ∑ j ∈ keysBlk d n t, |(vEff d m bh j e : ℝ) - attnRef d m bh n e|

/-- Over the two chunks of block `t` (keys `64t + 32c … 64t + 32c + 31`, `c = 0, 1`: the `p·v`
`tl.dot`'s chunks), the sum of each chunk's largest `w_j·liftMag(vEff_j)` over its keys. -/
noncomputable def blkLW (d : Inst) (m : Mem) (bh n e t : Nat) : ℝ :=
  ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0 fun k : Fin 64 =>
    if 64 * t + (k : ℕ) ∈ keys d n then
      weight d m bh n (64 * t + (k : ℕ)) * (Rules.Num.liftMag .f8e5 (vEff d m bh (64 * t + (k : ℕ)) e) : ℝ)
    else 0

/-- The same without the weights: the sum of each chunk's largest `liftMag(vEff_j)` over its keys. -/
noncomputable def blkVL (d : Inst) (m : Mem) (bh n e t : Nat) : ℝ :=
  ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0 fun k : Fin 64 =>
    if 64 * t + (k : ℕ) ∈ keys d n then (Rules.Num.liftMag .f8e5 (vEff d m bh (64 * t + (k : ℕ)) e) : ℝ)
    else 0

/-! ### The constants

`X` bounds the row's scaled lifted score magnitudes (`sm_scale · scoreLift ≤ X` over its keys), `Z`
its scaled score errors (`sm_scale · scoreErr ≤ Z`); `D` and `N` are `HEAD_DIM` and `N_CTX`. The
constants are the library's: the chunk facts' `2^-7`, `2^-12`, `2^-126` (`Rules.Num.DotE5m2Chunk`,
for the score's `tl.dot` and for the `p·v` `tl.dot` of `64` products in two chunks of `32`), their
whole-dot bound `relLift8` (`Dot8Chunk.DotF8Chunk.liftAcc`), `2^-22` (`Exp2F32`, `DivFull`,
`Log2F32`), and the roundings of `float32` (`2^-24`, `2^-150`) and of `float8e5` (`2^-3` relative,
`2^-17` absolute: `CvtSatF32E5m2`, one rounding). -/

/-- One rounding of `float32`, relative: `2^-24`. -/
def u32 : ℚ := 1 / 2 ^ 24

/-- The bound `3X/2` on `qk_scale · scoreLift` and on `sm_scale·log₂e · scoreLift`, the base-2
exponents' magnitudes (`qk_scale` is `sm_scale · log₂ e` up to `2^-22` relative). -/
def ybound (X : ℚ) : ℚ := 3 / 2 * X

/-- **The bound on `qk_scale` times a key's score error**: `3Z/2`, or, from the whole-dot bound,
`relLift8 D · 3X/2 + 2^-55`, whichever is smaller. -/
def scoreE (D : Nat) (X Z : ℚ) : ℚ := min (3 / 2 * Z) (Dot8Chunk.relLift8 D * ybound X + 1 / 2 ^ 55)

/-- The error of each computed exponent `qk·qk_scale − m` against the exact one
`sm_scale·log₂e·score − m`: the score's `tl.dot` error (`scoreE`), the change from `qk_scale` to
`sm_scale·log₂e`, and the roundings. -/
def expErr (D : Nat) (X Z : ℚ) : ℚ := scoreE D X Z + ybound X / 2 ^ 19 + 1 / 2 ^ 50

/-- `2^expErr`: the factor by which each `p` may differ from its exact weight. -/
noncomputable def pE (D : Nat) (X Z : ℚ) : ℝ := (2 : ℝ) ^ ((expErr D X Z : ℚ) : ℝ)

/-- The relative change of `alpha` and of each running maximum's rescaling per block, besides
`exp2`'s `2^-22`: `(m_ij − m_i)·2^-24 + 2^-150` with `|m| ≤ 3X/2 + expErr`. -/
def dmax (D : Nat) (X Z : ℚ) : ℚ := 2 * (ybound X + expErr D X Z) / 2 ^ 24 + 1 / 2 ^ 150

/-- The relative error of each `p = exp2(·)` against its exact weight, upwards. -/
noncomputable def psiP (D : Nat) (X Z : ℚ) : ℝ := pE D X Z * (1 + 1 / 2 ^ 22) - 1

/-- The number of key blocks of a row, at most: `N / 64`. -/
def nblocks (N : Nat) : Nat := N / 64

/-- The relative error of a block's sum `l_ij` (63 roundings). -/
def gam63 : ℚ := (1 + u32) ^ 63 - 1

/-- `l_i`'s per-block factor (`tl.mul_add`'s `3·2^-24` and `alpha`'s error). -/
def stepL (D : Nat) (X Z : ℚ) : ℚ := (1 + 3 * u32) * ((1 + dmax D X Z) * (1 + 1 / 2 ^ 22))

/-- `l_i`'s per-block factor from below. -/
def qL (D : Nat) (X Z : ℚ) : ℚ := (1 - 3 * u32) * (2 - (1 + dmax D X Z) * (1 + 1 / 2 ^ 22))

/-- **`l_i` after `t` blocks against the sum of the exact weights, from above**: `l_i ≤ FhT t·Σ`
(up to an absolute `t·2^-80·FhT t`), where `Σ` is the sum of the exact weights in the running
maximum's units, `Σ_j 2^(sm_scale·log₂e·score_j − m_i)` over the row's keys so far. -/
noncomputable def FhT (D : Nat) (X Z : ℚ) (t : Nat) : ℝ :=
  (1 + (gam63 : ℝ)) * (1 + psiP D X Z) * (stepL D X Z : ℝ) ^ t

/-- `FhT` after all the blocks: `l_i ≤ Fhi·Σ + aL`. -/
noncomputable def Fhi (D N : Nat) (X Z : ℚ) : ℝ := FhT D X Z (nblocks N)

/-- **The same from below**: `l_i ≥ Flo·Σ − aL` (`Σ` as for `FhT`). -/
noncomputable def Flo (D N : Nat) (X Z : ℚ) : ℝ :=
  (1 - 3 * (u32 : ℝ)) * (1 - (gam63 : ℝ)) * (1 - 1 / 2 ^ 22) / pE D X Z * (qL D X Z : ℝ) ^ nblocks N

/-- `l_i`'s absolute error per block. -/
def absL : ℚ := 1 / 2 ^ 64

/-- `l_i`'s absolute error after the loop. -/
def aL (N : Nat) : ℚ := nblocks N * absL

/-- The lower bound `Flo − aL·2^expErr` on `l_i` relative to the sum of the weights (the sum is at
least `2^-expErr` against the running maximum). -/
noncomputable def Fp (D N : Nat) (X Z : ℚ) : ℝ := Flo D N X Z - (aL N : ℝ) * pE D X Z

/-- **The `p·v` `tl.dot`'s constant on its accumulator.** Over its two chunks (`Dot8Chunk.errB` at
`2` chunks, `ρ = 2^-7`, `σ = 2^-12`, `α = 2^-126`) the result is within
`cS8·|acc| + cC8·Σ|p·v| + cP8·(m₀ + m₁) + cA8` of `acc + Σ p·v`, `m_c` chunk `c`'s largest lifted
product; `cS8 = (ρ + σ)(2 + ρ + σ)`. -/
def cS8 : ℚ :=
  (Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) * (2 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8)

/-- The `p·v` `tl.dot`'s constant on its chunks' exact sums (as `cS8`): `ρ + σ(2 + ρ + σ)`. -/
def cC8 : ℚ := Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8 * (2 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8)

/-- The `p·v` `tl.dot`'s constant on its chunks' largest lifted products (as `cS8`):
`ρ(1 + ρ + σ)`. -/
def cP8 : ℚ := Rules.Num.rhoChunk8 * (1 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8)

/-- The `p·v` `tl.dot`'s absolute constant (as `cS8`): `(2 + ρ + σ)·α`. -/
def cA8 : ℚ := (2 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) * Rules.Num.alphaChunk8

/-- The absolute error of a key's `p.to(float8e5)` against its weight: `float8e5`'s `2^-17` and
`exp2`'s `2^-126` (`9/8` of it, at most `2^-125`). -/
def eps8 : ℚ := 1 / 2 ^ 17 + 1 / 2 ^ 125

/-! ### The output's tolerance

The tolerance follows `acc − r·l_i` for `r = attnRef`, centred: line 106 rescales `l_i` by the
`alpha` that line 95 applies to `acc`, and each key's fp8 `p` lies between `a8` and `b8` times its
exact weight (in the running maximum's units), up to `eps8`. With `kap` the middle of `[a8, b8]` and
`hw` its half-width, `acc − r·l_i` stays within `ε_C + A·l_i` of `kap·Σ_j w_j (vEff_j − attnRef)`
(scaled), where `ε_C` (`errC`, in the units of the exact weights) collects each block's spread
`hw·Σ w_j |vEff_j − attnRef|` and the `p·v` `tl.dot`'s error, compounded block by block, and `A`
(`accA`) the errors relative to `l_i` and the absolute ones (the fp8 rounding of `p` against `r`,
`l_ij`'s and `l_i`'s roundings, `2^-17` per key). After the last block the centred sum is `0`
(`attnRef` is the weights' mean), so `acc / l_i` is within `ε_C/(Σw·Fp) + A` of `attnRef`. The
functions below take `pe = pE` and `dm = dmax`. -/

/-- `alpha`'s relative error: `(1 + dmax)(1 + 2^-22) − 1`. -/
noncomputable def etaA (dm : ℝ) : ℝ := (1 + dm) * (1 + 1 / 2 ^ 22) - 1

/-- **The growth per block** of what `acc` carries: `acc * alpha` (`alpha`'s error and one
rounding) and the `p·v` `tl.dot`'s `cS8` on it. -/
noncomputable def gA (dm : ℝ) : ℝ := (1 + (cS8 : ℝ)) * ((1 + dm) * (1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) - 1

/-- An attended key's fp8 `p` is at least `a8` times its exact weight (in the running maximum's
units, up to `eps8`): `exp2`'s `2^-expErr` and `2^-22`, and the fp8 rounding's `7/8`. -/
noncomputable def a8 (pe : ℝ) : ℝ := 7 / 8 * (1 - 1 / 2 ^ 22) / pe

/-- An attended key's fp8 `p` is at most `b8` times its exact weight (up to `eps8`):
`2^expErr·(1 + 2^-22)` and `9/8`. -/
noncomputable def b8 (pe : ℝ) : ℝ := 9 / 8 * pe * (1 + 1 / 2 ^ 22)

/-- The middle of `[a8, b8]`. -/
noncomputable def kap (pe : ℝ) : ℝ := (a8 pe + b8 pe) / 2

/-- The half-width of `[a8, b8]`: the spread of the keys' ratios of fp8 `p` to their exact weights. -/
noncomputable def hw (pe : ℝ) : ℝ := (b8 pe - a8 pe) / 2

/-- `l_i`'s new value from below, per unit of its rescaled old value. -/
noncomputable def denX (dm : ℝ) : ℝ := 1 - etaA dm - 3 * (1 + etaA dm) / 2 ^ 24

/-- `l_i`'s new value from below, per unit of the block's `Σ|p|`. -/
noncomputable def denS : ℝ := 1 - (gam63 : ℝ) - 3 * (1 + (gam63 : ℝ)) / 2 ^ 24

/-- The cost of a block's `Σ|p|` against `|r|`: the fp8 rounding of `p` (`1/8`), the `tl.dot`'s `cC8`
on `9/8·|r|·Σ|p|`, `l_ij`'s and `l_i`'s roundings. -/
noncomputable def coefS : ℝ := 1 / 8 + 9 / 8 * (cC8 : ℝ) + (gam63 : ℝ) + 3 * (1 + (gam63 : ℝ)) / 2 ^ 24

/-- **One block of `A`**: the old `A` (and `|r|`'s share of `l_i`'s rounding) against `l_i`'s new
value, the block's `Σ|p|` at `|r|·coefS`, both through `l_i`'s new value from below; then, against
`l_i ≥ 1/2`, the block's absolute errors: `eps8` per key on `|vEff − r|` (`Ba`) and on the
`tl.dot`'s chunks' largest lifted values (`Bv`, with `liftMag`'s `2^-14`), `2^-17` per key on
`|r|`, and the roundings near zero. `R = |r|`. -/
noncomputable def accNext (dm R A Ba Bv : ℝ) : ℝ :=
  (1 + 1 / 2 ^ 50) * max ((A + 3 * (1 + etaA dm) * R / 2 ^ 24) / denX dm) (R * coefS / denS) +
    2 * ((1 + (cC8 : ℝ)) * (eps8 : ℝ) * Ba + (cP8 : ℝ) * ((eps8 : ℝ) + 1 / 2 ^ 14) * Bv +
      R * (1 + (cC8 : ℝ)) / 2 ^ 11 + (1 + R + A) / 2 ^ 50)

/-- **One block of `ε_C`**: the old `ε_C` grows by `gA`; the old centred sum `C` costs `gA·kap·|C|`
and the old `r·l_i + A·l_i` (`l_i ≤ Lw`, in the exact weights' units) `(gA + etaA)·(R + A)·Lw` (the
`tl.dot`'s and `alpha`'s errors on what `acc` carries); the block costs the spread `hw` and the
`tl.dot`'s `cC8·b8` on `Bc = Σ w|vEff − r|`, and the `tl.dot`'s `cP8·b8` on its chunks' largest
`w·liftMag(vEff)` (`Bl`). -/
noncomputable def errNext (pe dm EC C R A Lw Bc Bl : ℝ) : ℝ :=
  (1 + gA dm) * EC + gA dm * kap pe * |C| + (gA dm + etaA dm) * (R + A) * Lw +
    (hw pe + (cC8 : ℝ) * b8 pe) * Bc + (cP8 : ℝ) * b8 pe * Bl

/-- `A` after `t` blocks, for element `(bh, n, e)` (`r = attnRef`). -/
noncomputable def accA (d : Inst) (m : Mem) (bh n e : Nat) (X Z : ℚ) : Nat → ℝ
  | 0 => 0
  | t + 1 => accNext (dmax d.HEAD_DIM X Z) |attnRef d m bh n e| (accA d m bh n e X Z t)
      (blkA d m bh n e t) (blkVL d m bh n e t)

/-- `ε_C` after `t` blocks, for element `(bh, n, e)`, in the units of the exact weights. -/
noncomputable def errC (d : Inst) (m : Mem) (bh n e : Nat) (X Z : ℚ) : Nat → ℝ
  | 0 => 0
  | t + 1 => errNext (pE d.HEAD_DIM X Z) (dmax d.HEAD_DIM X Z) (errC d m bh n e X Z t)
      (centC d m bh n e t) |attnRef d m bh n e| (accA d m bh n e X Z t)
      (FhT d.HEAD_DIM X Z t * partW d m bh n t) (blkC d m bh n e t) (blkLW d m bh n e t)

/-- **The output's tolerance**: `acc / l_i` is within `errC / (Σw·Fp) + accA` of the exact
attention (`l_i ≥ Fp` times the weights' sum), then the division (`DivFull`: `2^-22` relative,
`2^-148`) and the rounding to `float8e5` (`2^-3` relative, `2^-17` absolute). -/
noncomputable def tolO (d : Inst) (m : Mem) (bh n e : Nat) (X Z : ℚ) : ℝ :=
  9 / 8 * ((errC d m bh n e X Z (nblk d n) / wsum d m bh n / Fp d.HEAD_DIM d.N_CTX X Z +
      accA d m bh n e X Z (nblk d n)) * (1 + 1 / 2 ^ 22) + |attnRef d m bh n e| / 2 ^ 22 + 1 / 2 ^ 148) +
    |attnRef d m bh n e| / 8 + 1 / 2 ^ 17

/-- The bound on `|log₂(l_i / Σw)|` (`l_i` against the weights in the running maximum's units). -/
noncomputable def lgB (D N : Nat) (X Z : ℚ) : ℝ :=
  max (Real.logb 2 (Fhi D N X Z + (aL N : ℝ) * pE D X Z)) (-Real.logb 2 (Fp D N X Z))

/-- **`M`'s tolerance**, besides `2^-24·|lseRef|`: `log₂ l_i`'s error `lgB`, `tl.math.log2`'s
`2^-22·|log₂ l_i|` (with `|log₂ l_i| ≤ lgB + expErr + 16`) and `2^-149`, and the rounding of
`m_i + log2(l_i)`. -/
noncomputable def tolM (D N : Nat) (X Z : ℚ) : ℝ :=
  (1 + (u32 : ℝ)) * (lgB D N X Z + (lgB D N X Z + (expErr D X Z : ℝ) + 16) / 2 ^ 22 + 1 / 2 ^ 149) +
    1 / 2 ^ 150

/-! ### The second bound on the output

`tolO` compares the kernel's `p` with the exact weights, so each weight's factor `2^±expErr` enters
it. The second bound, `tolV`, compares `acc` with `r·l_i` for the reference `r = attnRef`, over the
weights the kernel computes (lines 95 and 106 rescale `acc` and `l_i` by the same `alpha`): the
values enter only through `K`, a bound on `|vEff_j − r|` over the row's keys, and `R = |r|`. Its
constants are the library's: the `p·v` `tl.dot`'s `cS8` (with `cP8` on `liftMag`'s `2^-14`, which
`iotaV` drops when every value of the row is `0`), the fp8
rounding of `p` (`2^-3`, `2^-17`), the `float32` roundings and `l_ij`'s `63` roundings (below
`2^-17`, `gam63`). -/

/-- `min 1 (2^16·(K + R))`: at most `1`, and `0` when `K = R = 0`. A nonzero `float8e5` value is at
least `2^-16` in magnitude, so when `K + R < 2^-16` every value of the row is `0` and so is every
lifted product of the `p·v` `tl.dot` (`liftMag 0 = 0`); the factor keeps `liftMag`'s `2^-14` out of
`tolV` then (a head without `vIn`). -/
noncomputable def iotaV (K R : ℝ) : ℝ := min 1 (2 ^ 16 * (K + R))

/-- The cost of a block's `Σ|p|` in the second bound: the fp8 rounding of `p` against `v − r`
(`9/8·K`, with the `tl.dot`'s `cS8`) and against `r` (`R/8`), `l_ij`'s and `l_i`'s roundings, and
the `tl.dot`'s `cP8` on the lifted `p`'s `2^-14` (times `iotaV`). -/
noncomputable def magCp (K R : ℝ) : ℝ :=
  9 / 8 * K * (1 + (cS8 : ℝ)) + R * (9 / 8 * (cS8 : ℝ) + 1 / 8 + 1 / 2 ^ 17 + 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) +
    9 / 8 * (cP8 : ℝ) * iotaV K R / 2 ^ 14

/-- One block of the relative part `A` of `|acc − r·l_i| ≤ A·l_i + D`: the old part, rescaled by
`alpha`, grows by the `tl.dot`'s `cS8` and the roundings; the block's new part costs `magCp`;
`l_i`'s new value bounds both. -/
noncomputable def magNext (K R A : ℝ) : ℝ :=
  max (((1 + (cS8 : ℝ)) * (1 + 1 / 2 ^ 24) * A + R * (4 / 2 ^ 24 + (cS8 : ℝ) * (1 + 1 / 2 ^ 24))) /
      (1 - 3 / 2 ^ 24))
    (magCp K R / (1 - 1 / 2 ^ 17 - 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24))

/-- The relative part after `t` blocks. -/
noncomputable def magA (K R : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => magNext K R (magA K R t)

/-- The absolute part after `t` blocks: the fp8 rounding's `2^-17` per key (`64` per block), the
`tl.dot`'s `cP8` on the lifted values' `2^-14` (times `iotaV`), and the roundings near zero. -/
noncomputable def magD (K R : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => (1 + (cS8 : ℝ)) * (1 + 1 / 2 ^ 24) * (1 + 1 / 2 ^ 12) * magD K R t +
      (K + R) * (1 + (cS8 : ℝ)) / 2 ^ 11 + (cP8 : ℝ) * iotaV K R * (K + R + 1 / 2 ^ 13) / 2 ^ 8 +
      (magA K R (t + 1) + R + 1) / 2 ^ 80

/-- **The output's second tolerance**, for a bound `K` on `|vEff_j − attnRef|` over the row's keys:
`acc / l_i` is within `magA + D/l_i` of `attnRef` (`l_i ≥ 1/2`), then the division (`DivFull`) and
the rounding to `float8e5`. -/
noncomputable def tolV (d : Inst) (m : Mem) (bh n e : Nat) (K : ℝ) : ℝ :=
  9 / 8 * ((magA K |attnRef d m bh n e| (nblk d n) + 2 * magD K |attnRef d m bh n e| (nblk d n)) *
      (1 + 1 / 2 ^ 22) + |attnRef d m bh n e| / 2 ^ 22 + 1 / 2 ^ 148) +
    |attnRef d m bh n e| / 8 + 1 / 2 ^ 17

/-- The bytes the launch may write: the elements of `o` and of `M`. -/
def outBytes (d : Inst) : Set Int :=
  d.o.elems (below (rows d * d.HEAD_DIM)) ∪ d.M.elems (below (rows d))

/-- What the launch leaves:
* every element `(bh, n, e)` of `o` holds a `float8e5` value `c`; when head `bh`'s values are
  covered (`HeadOk`), `c` is finite and, for every bound `X ≤ Xh HEAD_DIM` on the row's scaled lifted
  score magnitudes and every bound `Z` on its scaled score errors, within `tolO` of the exact
  attention of the values the kernel uses, `attnRef`, and, for every bound `K` on
  `|vEff_j − attnRef|` over the row's keys, within `tolV`. Neither tolerance need be below
  `|attnRef|` (README, Specification);
* every element `(bh, n)` of `M` holds a `float32` value `c`; for a covered head, `c` is finite and
  within `tolM + 2^-24·|lseRef|` of the exact log-sum-exp `lseRef` (base 2);
* every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ bh n e, bh < d.Z * d.H → n < d.N_CTX → e < d.HEAD_DIM →
    ∃ c : F8E5, fin.read .f8e5 (d.o.ptr (idx d bh n e : Int)).addr = some c ∧
      (HeadOk d init bh → c.isFinite = true ∧
        (∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
          (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
          (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
          |c.toReal - attnRef d init bh n e| ≤ tolO d init bh n e X Z) ∧
        ∀ K : ℝ, (∀ j ∈ keys d n, |(vEff d init bh j e : ℝ) - attnRef d init bh n e| ≤ K) →
          |c.toReal - attnRef d init bh n e| ≤ tolV d init bh n e K)) ∧
  (∀ bh n, bh < d.Z * d.H → n < d.N_CTX →
    ∃ c : F32, fin.read .f32 (d.M.ptr ((bh * d.N_CTX + n : Nat) : Int)).addr = some c ∧
      (HeadOk d init bh → c.isFinite = true ∧ ∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
        (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
        (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
        |c.toReal - lseRef d init bh n| ≤
          tolM d.HEAD_DIM d.N_CTX X Z + (u32 : ℝ) * |lseRef d init bh n|)) ∧
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
  /-- [rule] `Num.DotE5m2Chunk`: `tl.dot` of `float8e5` operands into `float32` with
  `maxImpreciseAcc = 2^30` (the default on sm_90), any `input_precision` (lines 73 and 103), chunk by
  chunk: chunks of `32` consecutive `k` in increasing order, each step within
  `2^-7·Mj + 2^-12·|s j + c_j| + 2^-126`. -/
  dot : Rules.Num.DotE5m2Chunk P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (lines 73, 103), of configuration `cfg8` (written
  out), always has a result. -/
  dot_total : Rules.Num.DotTotal P
    { a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32, maxImpreciseAcc := 2 ^ 30 }
  /-- [rule] `Num.CvtSatF32E5m2`: `x.to(tl.float8e5)` of a `float32` with the default rounding
  (lines 101 and 245) is the saturating rounding to nearest even, up to NaN payload, on every
  input. -/
  cvt : Rules.Num.CvtSatF32E5m2 P
  /-- [rule] `Num.CvtTotal`: the conversion from `float32` to `float8e5` with the default rounding
  (lines 101, 245) always has a result. -/
  cvt_total : Rules.Num.CvtTotal P Math.Float.Format.binary32 Math.Float.Format.e5m2 .rn
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

end Kernels.AttnFwdE5m2
