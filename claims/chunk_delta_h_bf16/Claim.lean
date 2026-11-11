import TVA

/-!
# `chunk_delta_h_bf16`: the chunked gated delta rule's state recurrence, forward, fla

The kernel is `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` of `source/chunk_delta_h.py`
(lines 58–349), which calls the `@triton.jit` helper `exp2` (lines 47–48); it reads the
module-level names `triton` and `tl` (lines 43–44). The host `chunk_gated_delta_rule_fwd_h`
(lines 398–467) launches it on `bfloat16` `k` (`[B, T, H, K]`), `w` (`[B, T, HV, K]`), `u`
(`[B, T, HV, V]`), an optional `float32` gate `g` (`[B, T, HV]`), an optional `float32` initial state
`h0` (`[N, HV, K, V]`), and writes `h` (`[B, NT, HV, K, V]`, `bfloat16`), `v_new` (like `u`) and,
when asked, the `float32` final state `ht` (`[N, HV, K, V]`), over fixed-length rows or a packed
variable-length batch (`cu_seqlens`), with `BT = 64`, `BV = 32` (`SETUP.md`). One program takes one
sequence, one value head and one block of `32` value columns, and runs over the sequence's chunks
of `64` tokens, carrying the `float32` state.

What is claimed: every covered launch ends normally; for every sequence `n`, value head `hv` and
value column `j`, every `v_new` element and every element of the stored states `h[t]` holds a
`bfloat16` value, and there are `float32` states `Y 0, …, Y (nch n)` such that `Y 0` is the initial
state (`0` without one), each stored state `h[t]` is the exact `bfloat16` rounding of `Y t`, `ht` is
the last of them, and each `Y (t + 1)` is within `errStep t` of one exact step `Step t` of the
recurrence from `Y t` (the new values computed from `h[t]`, which the kernel's `w` dots read); each
new value is within `tolV` of `vrefS`, `u − w·h[t]`. The numeric clauses hold when the inputs meet
`Vals` and the stored states of the chunk and of every earlier chunk meet `StOk`. Every byte outside
`h`, `v_new` and (when stored) `ht` is unchanged.

The conditions: `H` divides `HV`; `K ≤ 256`; the sizes, the grid and the `int32` arguments and
products within their ranges; `T ≤ 2^22` tokens and at most `8192` sequences and rows; fixed length
with `N = B`, or variable length with `B = 1` and `cu_seqlens` (`int32` or `int64`) from `0`,
nondecreasing, ending at `T`; the buffers hold their tensors; the outputs share no byte with each
other or with the inputs; on a variable-length launch `chunk_offsets` is what the host builds.

The tolerances rest on the device facts for `tl.dot` of `bfloat16` operands and for `tl.math.exp2`
on `float32` (`Device`).

* The model: `chunk_step`, `body_rest`, `fwd_kernel_h`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre` (with the host's `chunk_offsets`, `cbase`).
* The specification: `Post`.
* The claim: `claim`; the checks: `claim_satisfiable` and the covered instances of the setup
  (`setup_*_ok`).
-/

namespace Claims.ChunkDeltaH

open TVA

/-! ## Constants of the source -/

/-- The `float32` `+0` (`other=0.0`, the `tl.where` value `0` beside a `float32` tensor). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- The `tl.dot` configuration of lines 214–236 and 295–316: `bfloat16` operands, the default
`float32` accumulator, the default `input_precision` (`tf32`; no effect on 16-bit operands) and the
default `max_num_imprecise_acc` (`0`, not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-- `BT = chunk_size = 64` (the host's default, line 406) and `BV = 32` (the host's
configuration, line 436). -/
def BT : Nat := 64
def BV : Nat := 32

/-- The four `[64, BV]` blocks `b_h1 … b_h4` of the state. A block whose `if K > …` is not taken is
not a name of the source; the model carries a zero block in its place, which no operation reads. -/
abbrev St (bv : Nat) : Type :=
  Tensor [64, bv] F32 × Tensor [64, bv] F32 × Tensor [64, bv] F32 × Tensor [64, bv] F32

/-! ## The model -/

/-- One trip of the main loop (source lines 176–316), `STATE_V_FIRST = False`, `USE_G` either way
(the compile-time branch of line 246), `USE_GK = False`, `SAVE_NEW_VALUE = True`. The pointers are the offset ones of lines 118–123;
`T` is the sequence length as an `int64` (the comparison of line 178 promotes an `int32` `T`);
`i_t_int64` is line 176's value and `g_last_off` the offset of line 247, which the caller computes
in the loop's own integer type (lines 245, 247). -/
def chunk_step (k w v v_new h : Ptr .bf16) (g : Ptr .f32) (H HV K V bt bv : Nat) (USE_G : Bool)
    (bos i_h T : I64) (o_v : Tensor [bv] I64) (m_v : Tensor [bv] Bool)
    (o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32) (m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool)
    (i_t_int64 g_last_off : I64) (st : St bv) : Prog (St bv) := do
  let (b_h1, b_h2, b_h3, b_h4) := st
  -- line 177: the `int64` scalar `i_t_int64 * BT` plus `tl.arange(0, BT)` (`int32`, promoted)
  let o_t : Tensor [bt] I64 :=
    i_t_int64 * (IntV.ofNat bt : I64) + (← tl.convert .i32 .i64 (tl.arange 0 bt))
  -- line 178: signed `int64` comparison
  let m_t := o_t.ltS T
  -- lines 179–185 (`STATE_V_FIRST` false: lines 183–185): the scalar `int64` offset, then the
  -- `int32` `o_k1[:, None] * V` and the `int64` `o_v[None, :]`, each broadcast to `[64, BV]`; the
  -- store converts to the pointer's `bfloat16` (nearest even)
  let p_h1 : Tensor [64, bv] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
    o_k1.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
  let m_h1 := (m_k1.col bv).and (m_v.row 64)
  tl.store p_h1 (← tl.cvt b_h1) (some m_h1)
  -- lines 186–209: the same for the blocks that exist
  if 64 < K then do
    let p_h2 : Tensor [64, bv] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
      o_k2.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
    tl.store p_h2 (← tl.cvt b_h2) (some ((m_k2.col bv).and (m_v.row 64)))
  else pure ()
  if 128 < K then do
    let p_h3 : Tensor [64, bv] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
      o_k3.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
    tl.store p_h3 (← tl.cvt b_h3) (some ((m_k3.col bv).and (m_v.row 64)))
  else pure ()
  if 192 < K then do
    let p_h4 : Tensor [64, bv] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
      o_k4.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
    tl.store p_h4 (← tl.cvt b_h4) (some ((m_k4.col bv).and (m_v.row 64)))
  else pure ()
  -- lines 211–212: `o_t[:, None] * (HV*K)` (`int64`, the constexpr product built in `int64`) plus
  -- `o_k1[None, :]` (`int32`, promoted); `other=0.0` is the `float32` `+0` converted to `bfloat16`
  let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k1.row bt
  let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k1.row bt)))
    (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
  -- line 216: `tl.dot(b_w, b_h1.to(b_w.dtype))`: the state block rounded to `bfloat16`, the default
  -- accumulator (`+0`); no addition follows the product (line 239 subtracts it)
  let b_v : Tensor [bt, bv] F32 ← tl.dot cfgBF16 b_w (← tl.cvt b_h1) (tl.zeros [bt, bv] .f32)
  -- lines 217–237: one more `tl.dot` per existing block, with `b_v` as the accumulator
  let b_v ← if 64 < K then do
      let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k2.row bt
      let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k2.row bt)))
        (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
      tl.dot cfgBF16 b_w (← tl.cvt b_h2) b_v
    else pure b_v
  let b_v ← if 128 < K then do
      let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k3.row bt
      let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k3.row bt)))
        (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
      tl.dot cfgBF16 b_w (← tl.cvt b_h3) b_v
    else pure b_v
  let b_v ← if 192 < K then do
      let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k4.row bt
      let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k4.row bt)))
        (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
      tl.dot cfgBF16 b_w (← tl.cvt b_h4) b_v
    else pure b_v
  -- lines 238–239: the `bfloat16` load promoted to `float32` (exact), then `-`: a `tl.dot` result
  -- feeds it, so the fusable `tl.subc`
  let p_v : Tensor [bt, bv] (Ptr .bf16) := v + o_t.col bv * bc (IntV.ofNat (HV * V) : I64) + o_v.row bt
  let b_u : Tensor [bt, bv] BF16 ← tl.load p_v (some ((m_t.col bv).and (m_v.row bt)))
    (some (← tl.convert .f32 .bf16 (tl.full [bt, bv] zf32)))
  let b_v ← tl.subc (← tl.convert .bf16 .f32 b_u) b_v
  -- lines 241–243 (`SAVE_NEW_VALUE` true): stored as `bfloat16` (nearest even)
  let p_v : Tensor [bt, bv] (Ptr .bf16) := v_new + o_t.col bv * bc (IntV.ofNat (HV * V) : I64) + o_v.row bt
  tl.store p_v (← tl.cvt b_v) (some ((m_t.col bv).and (m_v.row bt)))
  -- line 246: the compile-time `if USE_G:` (lines 247–258); without it `b_v` and the state
  -- blocks pass unchanged
  let (b_v, b_h1, b_h2, b_h3, b_h4) ← if USE_G then do
      -- line 247: a scalar `float32` load; `.to(tl.float32)` is the identity
      let b_g_last : F32 ← tl.load1 (g + g_last_off)
      -- lines 248–249: `other=0.0` the `float32` `+0`
      let p_g : Tensor [bt] (Ptr .f32) := g + (bos * (IntV.ofNat HV : I64) + i_h) + o_t * bc (IntV.ofNat HV : I64)
      let b_g : Tensor [bt] F32 ← tl.load p_g (some m_t) (some (bc zf32))
      -- line 250: `exp2(b_g_last - b_g)` (the helper, lines 47–48: `tl.math.exp2(x.to(tl.float32))`,
      -- the `.to` the identity), the IEEE `-` (two loaded values, no product feeds it); the
      -- `tl.where` value `0` built in `float32`; `[:, None]`, then the IEEE `*`
      let e_g ← tl.exp2 (← tl.sub (bc b_g_last) b_g)
      let b_v ← tl.mul b_v ((tl.where m_t e_g (bc zf32)).col bv)
      -- line 251: the helper on the scalar
      let b_g_last := (← tl.exp2 (Tensor.scalar b_g_last)).item
      -- lines 252–258: IEEE `*`
      let b_h1 ← tl.mul b_h1 (bc b_g_last)
      let b_h2 ← if 64 < K then tl.mul b_h2 (bc b_g_last) else pure b_h2
      let b_h3 ← if 128 < K then tl.mul b_h3 (bc b_g_last) else pure b_h3
      let b_h4 ← if 192 < K then tl.mul b_h4 (bc b_g_last) else pure b_h4
      pure (b_v, b_h1, b_h2, b_h3, b_h4)
    else pure (b_v, b_h1, b_h2, b_h3, b_h4)
  -- line 288: `b_v.to(k.dtype.element_ty)`, `bfloat16` (nearest even)
  let b_vb : Tensor [bt, bv] BF16 ← tl.cvt b_v
  -- lines 290–295: `o_k1[:, None]` (`int32`) plus `o_t[None, :] * (H*K)` (`int64`); `tl.dot` with
  -- the state block as its accumulator
  let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k1.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
  let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k1.col bt).and (m_t.row 64)))
    (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
  let b_h1 ← tl.dot cfgBF16 b_k b_vb b_h1
  -- lines 296–316
  let b_h2 ← if 64 < K then do
      let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k2.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
      let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k2.col bt).and (m_t.row 64)))
        (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
      tl.dot cfgBF16 b_k b_vb b_h2
    else pure b_h2
  let b_h3 ← if 128 < K then do
      let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k3.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
      let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k3.col bt).and (m_t.row 64)))
        (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
      tl.dot cfgBF16 b_k b_vb b_h3
    else pure b_h3
  let b_h4 ← if 192 < K then do
      let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k4.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
      let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k4.col bt).and (m_t.row 64)))
        (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
      tl.dot cfgBF16 b_k b_vb b_h4
    else pure b_h4
  pure (b_h1, b_h2, b_h3, b_h4)

/-- Source lines 100–175 and 318–349, after the branch of line 90: the state blocks, the offset
pointers, the initial state, the main loop (`runLoop`, which the branch of line 90 gives: its
iterator, its variable's type and lines 176 and 245, applied to one trip `chunk_step`), and the
final state. `T` is the sequence length as an `int64`. -/
def body_rest (k v w v_new h : Ptr .bf16) (g h0 ht : Ptr .f32) (H HV K V bt bv : Nat)
    (USE_G USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (i_v i_nh i_h bos boh T : I64)
    (runLoop : (I64 → I64 → St bv → Prog (St bv)) → St bv → Prog (St bv)) : Prog Unit := do
  -- lines 100–115 (`STATE_V_FIRST` false: lines 109–115)
  let b_h1 := tl.zeros [64, bv] .f32
  let b_h2 := tl.zeros [64, bv] .f32
  let b_h3 := tl.zeros [64, bv] .f32
  let b_h4 := tl.zeros [64, bv] .f32
  -- line 118: `int64` products and sums (`.to(tl.int64)` the identity; `x * K*V` is `(x * K) * V`)
  let h := h + (boh * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
  -- line 119
  let v := v + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
  -- line 120: `i_h // (HV // H)`, the compile-time `HV // H` built in `int64`, truncating
  let k := k + (bos * (IntV.ofNat H : I64) +
    (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
      (IntV.ofNat K : I64)
  -- line 121
  let w := w + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64)
  -- lines 122–123 (`SAVE_NEW_VALUE` true)
  let v_new := v_new + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
  -- lines 125–128: pointer arithmetic only (nothing reads `h0`, `ht` when their flag is false)
  let h0 := h0 + i_nh * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
  let ht := ht + i_nh * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
  -- line 131: the `int64` `i_v * BV` plus `tl.arange(0, BV)` (promoted)
  let o_v : Tensor [bv] I64 := i_v * (IntV.ofNat bv : I64) + (← tl.convert .i32 .i64 (tl.arange 0 bv))
  -- line 132
  let m_v := o_v.ltS (IntV.ofNat V : I64)
  -- lines 133–140: `int32`
  let o_k1 : Tensor [64] I32 := tl.arange 0 64
  let m_k1 := o_k1.ltS (IntV.ofNat K : I32)
  let o_k2 : Tensor [64] I32 := (IntV.ofNat 64 : I32) + o_k1
  let m_k2 := o_k2.ltS (IntV.ofNat K : I32)
  let o_k3 : Tensor [64] I32 := (IntV.ofNat 128 : I32) + o_k1
  let m_k3 := o_k3.ltS (IntV.ofNat K : I32)
  let o_k4 : Tensor [64] I32 := (IntV.ofNat 192 : I32) + o_k1
  let m_k4 := o_k4.ltS (IntV.ofNat K : I32)
  -- lines 141–172 (`STATE_V_FIRST` false: lines 146–147, …): `other=0.0` the `float32` `+0`;
  -- `.to(tl.float32)` the identity; `+=` the IEEE `+` (no product feeds it)
  let (b_h1, b_h2, b_h3, b_h4) ← if USE_INITIAL_STATE then do
      let p_h0_1 : Tensor [64, bv] (Ptr .f32) := h0 + o_k1.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
      let m_h0_1 := (m_k1.col bv).and (m_v.row 64)
      let b_h1 ← tl.add b_h1 (← tl.load p_h0_1 (some m_h0_1) (some (bc zf32)))
      let b_h2 ← if 64 < K then do
          let p_h0_2 : Tensor [64, bv] (Ptr .f32) := h0 + o_k2.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
          tl.add b_h2 (← tl.load p_h0_2 (some ((m_k2.col bv).and (m_v.row 64))) (some (bc zf32)))
        else pure b_h2
      let b_h3 ← if 128 < K then do
          let p_h0_3 : Tensor [64, bv] (Ptr .f32) := h0 + o_k3.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
          tl.add b_h3 (← tl.load p_h0_3 (some ((m_k3.col bv).and (m_v.row 64))) (some (bc zf32)))
        else pure b_h3
      let b_h4 ← if 192 < K then do
          let p_h0_4 : Tensor [64, bv] (Ptr .f32) := h0 + o_k4.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
          tl.add b_h4 (← tl.load p_h0_4 (some ((m_k4.col bv).and (m_v.row 64))) (some (bc zf32)))
        else pure b_h4
      pure (b_h1, b_h2, b_h3, b_h4)
    else pure (b_h1, b_h2, b_h3, b_h4)
  -- line 175: the loop carries `b_h1 … b_h4`
  let (b_h1, b_h2, b_h3, b_h4) ← runLoop
    (fun i_t_int64 g_last_off st => chunk_step k w v v_new h g H HV K V bt bv USE_G bos i_h T o_v m_v
      o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4 i_t_int64 g_last_off st)
    (b_h1, b_h2, b_h3, b_h4)
  -- lines 318–349 (`STATE_V_FIRST` false: lines 323–324, …): `float32` stores (no conversion)
  if STORE_FINAL_STATE then do
    let p_ht : Tensor [64, bv] (Ptr .f32) := ht + o_k1.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
    tl.store p_ht b_h1 (some ((m_k1.col bv).and (m_v.row 64)))
    if 64 < K then do
      let p_ht : Tensor [64, bv] (Ptr .f32) := ht + o_k2.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
      tl.store p_ht b_h2 (some ((m_k2.col bv).and (m_v.row 64)))
    else pure ()
    if 128 < K then do
      let p_ht : Tensor [64, bv] (Ptr .f32) := ht + o_k3.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
      tl.store p_ht b_h3 (some ((m_k3.col bv).and (m_v.row 64)))
    else pure ()
    if 192 < K then do
      let p_ht : Tensor [64, bv] (Ptr .f32) := ht + o_k4.col bv * bc (IntV.ofNat V : I32) + o_v.row 64
      tl.store p_ht b_h4 (some ((m_k4.col bv).and (m_v.row 64)))
    else pure ()
  else pure ()

/-- The four `[BV, 64]` blocks of the state with `STATE_V_FIRST = True` (`St`'s blocks transposed),
under the same convention. -/
abbrev StV (bv : Nat) : Type :=
  Tensor [bv, 64] F32 × Tensor [bv, 64] F32 × Tensor [bv, 64] F32 × Tensor [bv, 64] F32

/-- One trip of the main loop (source lines 176–316) with `STATE_V_FIRST = True` (the state blocks
`[BV, 64]`, rows the value columns), `USE_G` either way (the compile-time branch of line 246),
`USE_GK = False`, `SAVE_NEW_VALUE = True`. The pointers are the offset ones of lines 118–123;
`T` is the sequence length as an `int64` (the comparison of line 178 promotes an `int32` `T`);
`i_t_int64` is line 176's value and `g_last_off` the offset of line 247, which the caller computes
in the loop's own integer type (lines 245, 247). -/
def chunk_step_vf (k w v v_new h : Ptr .bf16) (g : Ptr .f32) (H HV K V bt bv : Nat) (USE_G : Bool)
    (bos i_h T : I64) (o_v : Tensor [bv] I64) (m_v : Tensor [bv] Bool)
    (o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32) (m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool)
    (i_t_int64 g_last_off : I64) (st : StV bv) : Prog (StV bv) := do
  let (b_h1, b_h2, b_h3, b_h4) := st
  -- line 177: the `int64` scalar `i_t_int64 * BT` plus `tl.arange(0, BT)` (`int32`, promoted)
  let o_t : Tensor [bt] I64 :=
    i_t_int64 * (IntV.ofNat bt : I64) + (← tl.convert .i32 .i64 (tl.arange 0 bt))
  -- line 178: signed `int64` comparison
  let m_t := o_t.ltS T
  -- lines 179–181, 185 (`STATE_V_FIRST` true): the scalar `int64` offset, then the `int64`
  -- `o_v[:, None] * K` and the `int32` `o_k1[None, :]`, each broadcast to `[BV, 64]`; the store
  -- converts to the pointer's `bfloat16` (nearest even)
  let p_h1 : Tensor [bv, 64] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
    o_v.col 64 * bc (IntV.ofNat K : I64) + o_k1.row bv
  let m_h1 := (m_v.col 64).and (m_k1.row bv)
  tl.store p_h1 (← tl.cvt b_h1) (some m_h1)
  -- lines 186–209 (lines 187–189, 193, …): the same for the blocks that exist
  if 64 < K then do
    let p_h2 : Tensor [bv, 64] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
      o_v.col 64 * bc (IntV.ofNat K : I64) + o_k2.row bv
    tl.store p_h2 (← tl.cvt b_h2) (some ((m_v.col 64).and (m_k2.row bv)))
  else pure ()
  if 128 < K then do
    let p_h3 : Tensor [bv, 64] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
      o_v.col 64 * bc (IntV.ofNat K : I64) + o_k3.row bv
    tl.store p_h3 (← tl.cvt b_h3) (some ((m_v.col 64).and (m_k3.row bv)))
  else pure ()
  if 192 < K then do
    let p_h4 : Tensor [bv, 64] (Ptr .bf16) := h + i_t_int64 * (IntV.ofNat HV : I64) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64) +
      o_v.col 64 * bc (IntV.ofNat K : I64) + o_k4.row bv
    tl.store p_h4 (← tl.cvt b_h4) (some ((m_v.col 64).and (m_k4.row bv)))
  else pure ()
  -- lines 211–212: `o_t[:, None] * (HV*K)` (`int64`, the constexpr product built in `int64`) plus
  -- `o_k1[None, :]` (`int32`, promoted); `other=0.0` is the `float32` `+0` converted to `bfloat16`
  let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k1.row bt
  let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k1.row bt)))
    (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
  -- lines 213–214: `tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))`: the block transposed, rounded to
  -- `bfloat16`; the default accumulator (`+0`); no addition follows the product (line 239)
  let b_v : Tensor [bt, bv] F32 ← tl.dot cfgBF16 b_w (← tl.cvt b_h1.trans) (tl.zeros [bt, bv] .f32)
  -- lines 217–237 (lines 221, 228, 235): one more `tl.dot` per existing block, accumulator `b_v`
  let b_v ← if 64 < K then do
      let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k2.row bt
      let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k2.row bt)))
        (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
      tl.dot cfgBF16 b_w (← tl.cvt b_h2.trans) b_v
    else pure b_v
  let b_v ← if 128 < K then do
      let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k3.row bt
      let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k3.row bt)))
        (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
      tl.dot cfgBF16 b_w (← tl.cvt b_h3.trans) b_v
    else pure b_v
  let b_v ← if 192 < K then do
      let p_w : Tensor [bt, 64] (Ptr .bf16) := w + o_t.col 64 * bc (IntV.ofNat (HV * K) : I64) + o_k4.row bt
      let b_w : Tensor [bt, 64] BF16 ← tl.load p_w (some ((m_t.col 64).and (m_k4.row bt)))
        (some (← tl.convert .f32 .bf16 (tl.full [bt, 64] zf32)))
      tl.dot cfgBF16 b_w (← tl.cvt b_h4.trans) b_v
    else pure b_v
  -- lines 238–239: the `bfloat16` load promoted to `float32` (exact), then `-`: a `tl.dot` result
  -- feeds it, so the fusable `tl.subc`
  let p_v : Tensor [bt, bv] (Ptr .bf16) := v + o_t.col bv * bc (IntV.ofNat (HV * V) : I64) + o_v.row bt
  let b_u : Tensor [bt, bv] BF16 ← tl.load p_v (some ((m_t.col bv).and (m_v.row bt)))
    (some (← tl.convert .f32 .bf16 (tl.full [bt, bv] zf32)))
  let b_v ← tl.subc (← tl.convert .bf16 .f32 b_u) b_v
  -- lines 241–243 (`SAVE_NEW_VALUE` true): stored as `bfloat16` (nearest even)
  let p_v : Tensor [bt, bv] (Ptr .bf16) := v_new + o_t.col bv * bc (IntV.ofNat (HV * V) : I64) + o_v.row bt
  tl.store p_v (← tl.cvt b_v) (some ((m_t.col bv).and (m_v.row bt)))
  -- line 246: the compile-time `if USE_G:` (lines 247–258); without it `b_v` and the state
  -- blocks pass unchanged
  let (b_v, b_h1, b_h2, b_h3, b_h4) ← if USE_G then do
      -- line 247: a scalar `float32` load; `.to(tl.float32)` is the identity
      let b_g_last : F32 ← tl.load1 (g + g_last_off)
      -- lines 248–249: `other=0.0` the `float32` `+0`
      let p_g : Tensor [bt] (Ptr .f32) := g + (bos * (IntV.ofNat HV : I64) + i_h) + o_t * bc (IntV.ofNat HV : I64)
      let b_g : Tensor [bt] F32 ← tl.load p_g (some m_t) (some (bc zf32))
      -- line 250: `exp2(b_g_last - b_g)` (the helper, lines 47–48: `tl.math.exp2(x.to(tl.float32))`,
      -- the `.to` the identity), the IEEE `-` (two loaded values, no product feeds it); the
      -- `tl.where` value `0` built in `float32`; `[:, None]`, then the IEEE `*`
      let e_g ← tl.exp2 (← tl.sub (bc b_g_last) b_g)
      let b_v ← tl.mul b_v ((tl.where m_t e_g (bc zf32)).col bv)
      -- line 251: the helper on the scalar
      let b_g_last := (← tl.exp2 (Tensor.scalar b_g_last)).item
      -- lines 252–258: IEEE `*`
      let b_h1 ← tl.mul b_h1 (bc b_g_last)
      let b_h2 ← if 64 < K then tl.mul b_h2 (bc b_g_last) else pure b_h2
      let b_h3 ← if 128 < K then tl.mul b_h3 (bc b_g_last) else pure b_h3
      let b_h4 ← if 192 < K then tl.mul b_h4 (bc b_g_last) else pure b_h4
      pure (b_v, b_h1, b_h2, b_h3, b_h4)
    else pure (b_v, b_h1, b_h2, b_h3, b_h4)
  -- line 288: `b_v.to(k.dtype.element_ty)`, `bfloat16` (nearest even)
  let b_vb : Tensor [bt, bv] BF16 ← tl.cvt b_v
  -- lines 290–293: `o_k1[:, None]` (`int32`) plus `o_t[None, :] * (H*K)` (`int64`); line 293:
  -- `b_h1 += tl.trans(tl.dot(b_k, b_v))`: the dot from the default accumulator, transposed, then
  -- `+`, which a `tl.dot` result feeds: the fusable `tl.addc` (the combine pass leaves it: the
  -- addition's operand is the transpose, not the dot). Its other operand is line 252's product, and
  -- the backend contracts the `*` and the `+` into an `fma`; `tl.addc`'s `AddRel` allows that
  -- result (the rounding of `p + q`, `p` the exact product, which rounds to `tl.mul`'s result)
  let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k1.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
  let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k1.col bt).and (m_t.row 64)))
    (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
  let d_h1 : Tensor [64, bv] F32 ← tl.dot cfgBF16 b_k b_vb (tl.zeros [64, bv] .f32)
  let b_h1 ← tl.addc b_h1 d_h1.trans
  -- lines 296–316 (lines 300, 307, 314)
  let b_h2 ← if 64 < K then do
      let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k2.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
      let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k2.col bt).and (m_t.row 64)))
        (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
      let d_h2 : Tensor [64, bv] F32 ← tl.dot cfgBF16 b_k b_vb (tl.zeros [64, bv] .f32)
      tl.addc b_h2 d_h2.trans
    else pure b_h2
  let b_h3 ← if 128 < K then do
      let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k3.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
      let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k3.col bt).and (m_t.row 64)))
        (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
      let d_h3 : Tensor [64, bv] F32 ← tl.dot cfgBF16 b_k b_vb (tl.zeros [64, bv] .f32)
      tl.addc b_h3 d_h3.trans
    else pure b_h3
  let b_h4 ← if 192 < K then do
      let p_k : Tensor [64, bt] (Ptr .bf16) := k + o_k4.col bt + o_t.row 64 * bc (IntV.ofNat (H * K) : I64)
      let b_k : Tensor [64, bt] BF16 ← tl.load p_k (some ((m_k4.col bt).and (m_t.row 64)))
        (some (← tl.convert .f32 .bf16 (tl.full [64, bt] zf32)))
      let d_h4 : Tensor [64, bv] F32 ← tl.dot cfgBF16 b_k b_vb (tl.zeros [64, bv] .f32)
      tl.addc b_h4 d_h4.trans
    else pure b_h4
  pure (b_h1, b_h2, b_h3, b_h4)

/-- As `body_rest`, with `STATE_V_FIRST = True`: the state blocks `[BV, 64]`, the `h0` and `ht`
layouts `[V, K]` (source lines 100–107, 143–144, …, 320–321, …). Source lines 100–175 and 318–349,
after the branch of line 90: the state blocks, the offset
pointers, the initial state, the main loop (`runLoop`, which the branch of line 90 gives: its
iterator, its variable's type and lines 176 and 245, applied to one trip `chunk_step`), and the
final state. `T` is the sequence length as an `int64`. -/
def body_rest_vf (k v w v_new h : Ptr .bf16) (g h0 ht : Ptr .f32) (H HV K V bt bv : Nat)
    (USE_G USE_INITIAL_STATE STORE_FINAL_STATE : Bool) (i_v i_nh i_h bos boh T : I64)
    (runLoop : (I64 → I64 → StV bv → Prog (StV bv)) → StV bv → Prog (StV bv)) : Prog Unit := do
  -- lines 100–107 (`STATE_V_FIRST` true)
  let b_h1 := tl.zeros [bv, 64] .f32
  let b_h2 := tl.zeros [bv, 64] .f32
  let b_h3 := tl.zeros [bv, 64] .f32
  let b_h4 := tl.zeros [bv, 64] .f32
  -- line 118: `int64` products and sums (`.to(tl.int64)` the identity; `x * K*V` is `(x * K) * V`)
  let h := h + (boh * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
  -- line 119
  let v := v + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
  -- line 120: `i_h // (HV // H)`, the compile-time `HV // H` built in `int64`, truncating
  let k := k + (bos * (IntV.ofNat H : I64) +
    (← tl.idiv (Tensor.scalar i_h) (Tensor.scalar (IntV.ofNat (HV / H) : I64))).item) *
      (IntV.ofNat K : I64)
  -- line 121
  let w := w + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat K : I64)
  -- lines 122–123 (`SAVE_NEW_VALUE` true)
  let v_new := v_new + (bos * (IntV.ofNat HV : I64) + i_h) * (IntV.ofNat V : I64)
  -- lines 125–128: pointer arithmetic only (nothing reads `h0`, `ht` when their flag is false)
  let h0 := h0 + i_nh * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
  let ht := ht + i_nh * (IntV.ofNat K : I64) * (IntV.ofNat V : I64)
  -- line 131: the `int64` `i_v * BV` plus `tl.arange(0, BV)` (promoted)
  let o_v : Tensor [bv] I64 := i_v * (IntV.ofNat bv : I64) + (← tl.convert .i32 .i64 (tl.arange 0 bv))
  -- line 132
  let m_v := o_v.ltS (IntV.ofNat V : I64)
  -- lines 133–140: `int32`
  let o_k1 : Tensor [64] I32 := tl.arange 0 64
  let m_k1 := o_k1.ltS (IntV.ofNat K : I32)
  let o_k2 : Tensor [64] I32 := (IntV.ofNat 64 : I32) + o_k1
  let m_k2 := o_k2.ltS (IntV.ofNat K : I32)
  let o_k3 : Tensor [64] I32 := (IntV.ofNat 128 : I32) + o_k1
  let m_k3 := o_k3.ltS (IntV.ofNat K : I32)
  let o_k4 : Tensor [64] I32 := (IntV.ofNat 192 : I32) + o_k1
  let m_k4 := o_k4.ltS (IntV.ofNat K : I32)
  -- lines 141–172 (`STATE_V_FIRST` true: lines 143–144, …: `o_v[:, None] * K` in `int64`,
  -- `o_k1[None, :]` in `int32`): `other=0.0` the `float32` `+0`;
  -- `.to(tl.float32)` the identity; `+=` the IEEE `+` (no product feeds it)
  let (b_h1, b_h2, b_h3, b_h4) ← if USE_INITIAL_STATE then do
      let p_h0_1 : Tensor [bv, 64] (Ptr .f32) := h0 + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k1.row bv
      let m_h0_1 := (m_v.col 64).and (m_k1.row bv)
      let b_h1 ← tl.add b_h1 (← tl.load p_h0_1 (some m_h0_1) (some (bc zf32)))
      let b_h2 ← if 64 < K then do
          let p_h0_2 : Tensor [bv, 64] (Ptr .f32) := h0 + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k2.row bv
          tl.add b_h2 (← tl.load p_h0_2 (some ((m_v.col 64).and (m_k2.row bv))) (some (bc zf32)))
        else pure b_h2
      let b_h3 ← if 128 < K then do
          let p_h0_3 : Tensor [bv, 64] (Ptr .f32) := h0 + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k3.row bv
          tl.add b_h3 (← tl.load p_h0_3 (some ((m_v.col 64).and (m_k3.row bv))) (some (bc zf32)))
        else pure b_h3
      let b_h4 ← if 192 < K then do
          let p_h0_4 : Tensor [bv, 64] (Ptr .f32) := h0 + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k4.row bv
          tl.add b_h4 (← tl.load p_h0_4 (some ((m_v.col 64).and (m_k4.row bv))) (some (bc zf32)))
        else pure b_h4
      pure (b_h1, b_h2, b_h3, b_h4)
    else pure (b_h1, b_h2, b_h3, b_h4)
  -- line 175: the loop carries `b_h1 … b_h4`
  let (b_h1, b_h2, b_h3, b_h4) ← runLoop
    (fun i_t_int64 g_last_off st => chunk_step_vf k w v v_new h g H HV K V bt bv USE_G bos i_h T o_v m_v
      o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4 i_t_int64 g_last_off st)
    (b_h1, b_h2, b_h3, b_h4)
  -- lines 318–349 (`STATE_V_FIRST` true: lines 320–321, …): `float32` stores (no conversion)
  if STORE_FINAL_STATE then do
    let p_ht : Tensor [bv, 64] (Ptr .f32) := ht + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k1.row bv
    tl.store p_ht b_h1 (some ((m_v.col 64).and (m_k1.row bv)))
    if 64 < K then do
      let p_ht : Tensor [bv, 64] (Ptr .f32) := ht + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k2.row bv
      tl.store p_ht b_h2 (some ((m_v.col 64).and (m_k2.row bv)))
    else pure ()
    if 128 < K then do
      let p_ht : Tensor [bv, 64] (Ptr .f32) := ht + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k3.row bv
      tl.store p_ht b_h3 (some ((m_v.col 64).and (m_k3.row bv)))
    else pure ()
    if 192 < K then do
      let p_ht : Tensor [bv, 64] (Ptr .f32) := ht + o_v.col 64 * bc (IntV.ofNat K : I64) + o_k4.row bv
      tl.store p_ht b_h4 (some ((m_v.col 64).and (m_k4.row bv)))
    else pure ()
  else pure ()

/-- The main loop of a variable-length launch (source lines 175–176, 245, 247): `range(NT)` over the
`int64` `NT` (a plain name), the `int64` variable `i_t`; one trip is `step i_t_int64 g_last_off`. -/
def loopVar {σ : Type} (HV bt : Nat) (NT T bos i_h : I64) (step : I64 → I64 → σ → Prog σ)
    (st0 : σ) : Prog σ :=
  tl.forRange (IntV.ofInt 0 : I64) NT (IntV.ofInt 1 : I64) st0 fun i_t st => do
    -- line 176: `.to(tl.int64)` of an `int64`, the identity
    let i_t_int64 := i_t
    -- line 245: Python's `min` of two `int64` tensors (`tl.minimum`, exact), then `- 1`
    let last_idx : I64 := (tl.iminimum
      (Tensor.scalar ((i_t + IntV.ofInt 1) * (IntV.ofNat bt : I64))) (Tensor.scalar T)).item -
        IntV.ofInt 1
    -- line 247's offset, in `int64` (integer arithmetic only; read by the trip only when `USE_G`)
    step i_t_int64 (bos * (IntV.ofNat HV : I64) + last_idx * (IntV.ofNat HV : I64) + i_h) st

/-- The main loop of a fixed-length launch: `range(NT)` over the `int32` `NT`, the `int32` variable
`i_t`, the `int32` argument `T`. -/
def loopFix {σ : Type} (HV bt : Nat) (NT T : I32) (bos i_h : I64) (step : I64 → I64 → σ → Prog σ)
    (st0 : σ) : Prog σ :=
  tl.forRange (IntV.ofInt 0 : I32) NT (IntV.ofInt 1 : I32) st0 fun i_t st => do
    -- line 176: `.to(tl.int64)`
    let i_t_int64 := (← tl.convert .i32 .i64 (Tensor.scalar i_t)).item
    -- line 245: `int32` (`(i_t + 1) * BT`, Python's `min` as `tl.minimum`, `- 1`)
    let last_idx : I32 := (tl.iminimum
      (Tensor.scalar ((i_t + IntV.ofInt 1) * (IntV.ofNat bt : I32))) (Tensor.scalar T)).item -
        IntV.ofInt 1
    -- line 247's offset: `bos * HV` (`int64`) plus the `int32` `last_idx * HV` (promoted), plus `i_h`
    -- (integer arithmetic only; read by the trip only when `USE_G`)
    let lh := (← tl.convert .i32 .i64 (Tensor.scalar (last_idx * (IntV.ofNat HV : I32)))).item
    step i_t_int64 (bos * (IntV.ofNat HV : I64) + lh + i_h) st

/-- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (source lines 58–349). `k`, `v` (the host's
`u`), `w`, `v_new`, `h` are `*bf16` pointers, `g`, `h0`, `ht` `*fp32`; `cu_seqlens` a pointer of
the dtype `cuD` (`int32` or `int64`) and `chunk_offsets` an `*i64` (both `None`, never read, on a
fixed-length launch; `h0`, `ht` likewise when their flag is false); `gk` is `None` and never read;
`T` is the runtime `int32` (`do_not_specialize=['T']`: never a compile-time constant). `H`, `HV`,
`K`, `V`, `BT`, `BV` and the flags are `tl.constexpr`, Lean parameters; `SAVE_NEW_VALUE` is `True`
and `USE_GK` is `False` on every covered launch, and their branches are written as taken (the
others dropped, `config`); `USE_G` is either way (`g` is `None`, never read, when it is `False`);
`STATE_V_FIRST` chooses `body_rest_vf` or `body_rest`. -/
def fwd_kernel_h (k v w v_new : Ptr .bf16) (g : Ptr .f32) (h : Ptr .bf16) (h0 ht : Ptr .f32)
    (cuD : DType) (cu_seqlens : Ptr cuD) (chunk_offsets : Ptr .i64) (T : I32)
    (H HV K V bt bv : Nat) (USE_G USE_INITIAL_STATE STORE_FINAL_STATE STATE_V_FIRST IS_VARLEN : Bool) :
    Kernel where
  body ctx := do
    -- line 86: axis 0 of the grid, `.to(tl.int64)`
    let pid := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 87: `tl.cdiv(V, BV)` of two compile-time values, Python's `(V + (BV - 1)) // BV`
    let NV : Nat := (V + (bv - 1)) / bv
    -- line 88: `int64` `%` and `//` by `NV` built in `int64` (`.to(tl.int64)` the identity)
    let i_v := (← tl.irem (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat NV : I64))).item
    let i_nh := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar (IntV.ofNat NV : I64))).item
    -- line 89
    let i_n := (← tl.idiv (Tensor.scalar i_nh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    let i_h := (← tl.irem (Tensor.scalar i_nh) (Tensor.scalar (IntV.ofNat HV : I64))).item
    -- line 90: the compile-time `if IS_VARLEN:`
    if IS_VARLEN then do
      -- line 91: two scalar loads, `.to(tl.int64)`; `cu_seqlens + i_n + 1` adds the `int32` `1`
      let bos := (← tl.convert cuD .i64 (Tensor.scalar (← tl.load1 (cu_seqlens + i_n)))).item
      let eos := (← tl.convert cuD .i64
        (Tensor.scalar (← tl.load1 (cu_seqlens + i_n + (IntV.ofInt 1 : I32))))).item
      -- line 92: the wrapping `int64` `-`
      let T := eos - bos
      -- line 93: `tl.cdiv(T, BT)`, `(T + (BT - 1)) // BT` in `int64`
      let NT := (← tl.idiv (Tensor.scalar (T + (IntV.ofNat (bt - 1) : I64)))
        (Tensor.scalar (IntV.ofNat bt : I64))).item
      -- line 94: an `int64` load (`.to(tl.int64)` the identity)
      let boh ← tl.load1 (chunk_offsets + i_n)
      -- lines 100–349, the layout chosen by the compile-time `if STATE_V_FIRST:` of lines 100, 142, …
      if STATE_V_FIRST then
        body_rest_vf k v w v_new h g h0 ht H HV K V bt bv USE_G USE_INITIAL_STATE STORE_FINAL_STATE i_v i_nh
          i_h bos boh T (loopVar HV bt NT T bos i_h)
      else
        body_rest k v w v_new h g h0 ht H HV K V bt bv USE_G USE_INITIAL_STATE STORE_FINAL_STATE i_v i_nh i_h
          bos boh T (loopVar HV bt NT T bos i_h)
    else do
      -- line 96: `i_n * T` with the `int32` `T` promoted to `int64` (`eos` is not used)
      let T64 := (← tl.convert .i32 .i64 (Tensor.scalar T)).item
      let bos := i_n * T64
      -- line 97: `tl.cdiv(T, BT)` in `int32` (the addition wraps; `Inst.ok` keeps it below `2^31`)
      let NT := (← tl.idiv (Tensor.scalar (T + (IntV.ofNat (bt - 1) : I32)))
        (Tensor.scalar (IntV.ofNat bt : I32))).item
      -- line 98: `i_n * NT`, `NT` promoted to `int64`
      let boh := i_n * (← tl.convert .i32 .i64 (Tensor.scalar NT)).item
      if STATE_V_FIRST then
        body_rest_vf k v w v_new h g h0 ht H HV K V bt bv USE_G USE_INITIAL_STATE STORE_FINAL_STATE i_v i_nh
          i_h bos boh T64 (loopFix HV bt NT T bos i_h)
      else
        body_rest k v w v_new h g h0 ht H HV K V bt bv USE_G USE_INITIAL_STATE STORE_FINAL_STATE i_v i_nh i_h
          bos boh T64 (loopFix HV bt NT T bos i_h)

/-! ## The conditions -/

/-- What the host chooses: one launch of `chunk_gated_delta_rule_fwd_h(k, w, u, g,
initial_state, output_final_state, cu_seqlens=…)` with `gk = None`, `chunk_size = 64`,
`save_new_value = True`, `g` given or `None`, `state_v_first` either way (`SETUP.md`). The tensors are contiguous, given
by their storage (`Buf`); `cu_seqlens`' entries are part of the instance. -/
structure Inst where
  /-- `IS_VARLEN = cu_seqlens is not None`. -/
  varlen : Bool
  /-- `USE_G = g is not None` (the gated delta rule passes the gate; the delta rule passes
  `g = None`). -/
  useG : Bool
  /-- `USE_INITIAL_STATE = initial_state is not None`. -/
  useH0 : Bool
  /-- `STORE_FINAL_STATE = output_final_state`. -/
  storeHt : Bool
  /-- `STATE_V_FIRST = state_v_first`: the state, `h`, `h0` and `ht` in the `[V, K]` layout. -/
  vf : Bool
  /-- `B, T, H, K = k.shape`; `HV = u.shape[2]`, `V = u.shape[3]`. -/
  B : Nat
  T : Nat
  H : Nat
  HV : Nat
  K : Nat
  V : Nat
  /-- Variable length: the number of sequences `N` (`cu_seqlens` has `N + 1` entries); fixed
  length: `N = B`. -/
  N : Nat
  /-- Variable length: `cu_seqlens`' dtype (`int32` or `int64`) and its entries. -/
  cuD : DType
  cuVal : Nat → cuD.Val
  /-- The storages: `k`, `w`, `u` (`bfloat16`), `g` (`float32`, when given), `h`, `v_new` (`bfloat16`,
  allocated by the host), `h0` (`float32`, when given), `ht` (`float32`, allocated by the host when
  asked), `cu_seqlens` and `chunk_offsets` (`int64`, built by the host) on a variable-length
  launch. -/
  k : Buf .bf16
  w : Buf .bf16
  u : Buf .bf16
  g : Buf .f32
  h : Buf .bf16
  vn : Buf .bf16
  h0 : Buf .f32
  ht : Buf .f32
  cu : Buf cuD
  co : Buf .i64

/-- An integer table entry (`int32`/`int64`) as an integer. -/
def entryInt : (dt : DType) → dt.Val → Int
  | .i32, x => x.toInt
  | .i64, x => x.toInt
  | _, _ => 0

/-- `cu_seqlens[i]`. -/
def cuAt (d : Inst) (i : Nat) : Nat := (entryInt d.cuD (d.cuVal i)).toNat

/-- Sequence `n`'s batch row, its first token in that row, and its length: row `n`, from `0`,
`T` tokens (fixed length); row `0`, from `cu_seqlens[n]` to `cu_seqlens[n+1]` (variable length). -/
def rowOf (d : Inst) (n : Nat) : Nat := if d.varlen then 0 else n
def startOf (d : Inst) (n : Nat) : Nat := if d.varlen then cuAt d n else 0
def lenOf (d : Inst) (n : Nat) : Nat := if d.varlen then cuAt d (n + 1) - cuAt d n else d.T

/-- The number of chunks of sequence `n`, `cdiv(len, 64)`. -/
def nch (d : Inst) (n : Nat) : Nat := (lenOf d n + BT - 1) / BT

/-- Sequence `n`'s first chunk in `h`: `n·cdiv(T, 64)` (fixed length), or the chunks of the
sequences before it (`prepare_chunk_offsets`, lines 391–395). -/
def cbase (d : Inst) (n : Nat) : Nat :=
  if d.varlen then ((List.range n).map (nch d)).sum else n * ((d.T + BT - 1) / BT)

/-- `h.shape[1]`: `cdiv(T, 64)` (fixed length) or `len(chunk_indices)`, all the chunks of the
batch (variable length). -/
def NTt (d : Inst) : Nat := if d.varlen then cbase d d.N else (d.T + BT - 1) / BT

/-- The grid's size, `triton.cdiv(V, BV) · N · HV` (line 437). -/
def gridN (d : Inst) : Nat := (d.V + BV - 1) / BV * d.N * d.HV

/-- Contiguous element indices: `k[b, t, hk, c]`, `w[b, t, hv, c]`, `u`/`v_new[b, t, hv, j]`,
`g[b, t, hv]`, `h[c₀, hv, c, j]` with `c₀` the flat chunk index (`b·NT + chunk`), and
`h0`/`ht[n, hv, c, j]`, for state row `c < K` and value column `j < V`; with `STATE_V_FIRST` the
last two axes of `h`, `h0`, `ht` are `[V, K]` (`h[c₀, hv, j, c]`). -/
def kIdx (d : Inst) (b t hk c : Nat) : Nat := ((b * d.T + t) * d.H + hk) * d.K + c
def wIdx (d : Inst) (b t hv c : Nat) : Nat := ((b * d.T + t) * d.HV + hv) * d.K + c
def uIdx (d : Inst) (b t hv j : Nat) : Nat := ((b * d.T + t) * d.HV + hv) * d.V + j
def gIdx (d : Inst) (b t hv : Nat) : Nat := (b * d.T + t) * d.HV + hv
def hIdx (d : Inst) (c₀ hv c j : Nat) : Nat :=
  if d.vf then ((c₀ * d.HV + hv) * d.V + j) * d.K + c else ((c₀ * d.HV + hv) * d.K + c) * d.V + j
def sIdx (d : Inst) (n hv c j : Nat) : Nat :=
  if d.vf then ((n * d.HV + hv) * d.V + j) * d.K + c else ((n * d.HV + hv) * d.K + c) * d.V + j

/-- The instances covered (README, Conditions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- Heads: `1 ≤ H`, `H` divides `HV ≥ 1` (the code base's grouped value heads), the grid within
  CUDA's limits. -/
  H_pos : 0 < d.H
  HV_pos : 0 < d.HV
  H_dvd : d.H ∣ d.HV
  HV_lt : d.HV < 2 ^ 16
  /-- `1 ≤ K ≤ 256` (the host asserts `K ≤ 256`), `1 ≤ V < 2^16`. -/
  K_pos : 0 < d.K
  K_le : d.K ≤ 256
  V_pos : 0 < d.V
  V_lt : d.V < 2 ^ 16
  /-- The compile-time products `HV*V` (lines 238, 242) are `int32` constants (real use: at most
  `32·128`). -/
  HVV_lt : d.HV * d.V < 2 ^ 31
  /-- At least one sequence; the grid `(cdiv(V, 32)·N·HV,)` has fewer than `2^31` programs. -/
  N_pos : 0 < d.N
  grid_lt : gridN d < 2 ^ 31
  /-- `T` is an `int32` argument, and line 97's `T + 63` does not wrap. -/
  T_lt : d.T + BT < 2 ^ 31
  /-- `T` is a context length (a row's tokens, or a packed batch's): at most `2^22` (README,
  Exclusions). -/
  T_ctx : d.T ≤ 2 ^ 22
  /-- The number of sequences `N` and of batch rows `B` are batch sizes: at most `8192` (README,
  Exclusions). -/
  N_batch : d.N ≤ 8192 ∧ d.B ≤ 8192
  /-- With the gate, line 247's `int32` `last_idx·HV` does not wrap (without it line 247 does not
  run). -/
  THV_lt : d.useG = true → d.T * d.HV < 2 ^ 31
  /-- Fixed length: `N = B ≥ 1`. -/
  fixed_N : d.varlen = false → d.N = d.B
  /-- Variable length: `B = 1`, `int32` or `int64` `cu_seqlens`, starting at `0`, nondecreasing,
  ending at `T` (every token belongs to a sequence). -/
  var_B : d.varlen = true → d.B = 1
  var_cuD : d.varlen = true → d.cuD = .i32 ∨ d.cuD = .i64
  var_cu0 : d.varlen = true → entryInt d.cuD (d.cuVal 0) = 0
  var_mono : d.varlen = true → ∀ n, n < d.N →
    entryInt d.cuD (d.cuVal n) ≤ entryInt d.cuD (d.cuVal (n + 1))
  var_end : d.varlen = true → entryInt d.cuD (d.cuVal d.N) = d.T
  /-- The storages hold the tensors. -/
  k_len : d.B * d.T * d.H * d.K ≤ d.k.len
  w_len : d.B * d.T * d.HV * d.K ≤ d.w.len
  u_len : d.B * d.T * d.HV * d.V ≤ d.u.len
  g_len : d.useG = true → d.B * d.T * d.HV ≤ d.g.len
  h_len : d.B * NTt d * d.HV * d.K * d.V ≤ d.h.len
  vn_len : d.B * d.T * d.HV * d.V ≤ d.vn.len
  h0_len : d.useH0 = true → d.N * d.HV * d.K * d.V ≤ d.h0.len
  ht_len : d.storeHt = true → d.N * d.HV * d.K * d.V ≤ d.ht.len
  cu_len : d.varlen = true → d.N + 1 ≤ d.cu.len
  co_len : d.varlen = true → d.N + 1 ≤ d.co.len
  /-- The outputs `h`, `v_new`, `ht` (fresh allocations, lines 432–435) share no byte with each
  other or with the inputs. -/
  disj_h : Disjoint d.h.bytes d.k.bytes ∧ Disjoint d.h.bytes d.w.bytes ∧
    Disjoint d.h.bytes d.u.bytes ∧ Disjoint d.h.bytes d.g.bytes ∧ Disjoint d.h.bytes d.vn.bytes ∧
    Disjoint d.h.bytes d.h0.bytes ∧ Disjoint d.h.bytes d.ht.bytes ∧
    Disjoint d.h.bytes d.cu.bytes ∧ Disjoint d.h.bytes d.co.bytes
  disj_vn : Disjoint d.vn.bytes d.k.bytes ∧ Disjoint d.vn.bytes d.w.bytes ∧
    Disjoint d.vn.bytes d.u.bytes ∧ Disjoint d.vn.bytes d.g.bytes ∧
    Disjoint d.vn.bytes d.h0.bytes ∧ Disjoint d.vn.bytes d.ht.bytes ∧
    Disjoint d.vn.bytes d.cu.bytes ∧ Disjoint d.vn.bytes d.co.bytes
  disj_ht : Disjoint d.ht.bytes d.k.bytes ∧ Disjoint d.ht.bytes d.w.bytes ∧
    Disjoint d.ht.bytes d.u.bytes ∧ Disjoint d.ht.bytes d.g.bytes ∧
    Disjoint d.ht.bytes d.h0.bytes ∧ Disjoint d.ht.bytes d.cu.bytes ∧
    Disjoint d.ht.bytes d.co.bytes

/-- The model on an instance: the launch of lines 437–466 (`BT = 64`, `BV = 32`). -/
def model (d : Inst) : Kernel :=
  fwd_kernel_h (d.k.ptr 0) (d.u.ptr 0) (d.w.ptr 0) (d.vn.ptr 0) (d.g.ptr 0) (d.h.ptr 0)
    (d.h0.ptr 0) (d.ht.ptr 0) d.cuD (d.cu.ptr 0) (d.co.ptr 0) (IntV.ofNat d.T)
    d.H d.HV d.K d.V BT BV d.useG d.useH0 d.storeHt d.vf d.varlen

/-- The host's launch (lines 437–466): the grid `(cdiv(V, 32)·N·HV,)`; the source's block shapes
(`[32]`: `o_v`, `m_v`; `[64]`: `o_k*`, `m_k*`, `o_t`, `m_t`, `b_g`; `[64, 32]`: the state blocks,
`b_v`, the `h`, `h0`, `ht`, `u`, `v_new` tiles; `[32, 64]`: the state blocks and tiles with
`STATE_V_FIRST`; `[64, 64]`: `b_w`, `b_k`; and the expanded `[64, 1]`, `[1, 32]`, `[32, 1]`, `[1, 64]`);
`enable_fp_fusion` Triton's default (`true`); `num_ctas = 1`, no `launch_pdl`, `debug = false`.
`num_warps = 4` and `num_stages = 3` are performance-only. -/
def launch (d : Inst) : Launch :=
  { grid := ![gridN d, 1, 1], fusion := true,
    blocks := [[BV], [BT], [BT, BV], [BV, BT], [BT, BT], [BT, 1], [1, BV], [BV, 1], [1, BT]] }

/-- The inputs covered: the buffers the launch reads or writes are valid (aligned, present; `g`
when given), and on a variable-length launch `cu_seqlens` holds its entries and `chunk_offsets` is what the host
builds (`prepare_chunk_offsets`: `chunk_offsets[n] = cbase n`, `n ≤ N`). Any float values:
`Post` says what holds when they are covered (`Vals`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.k.Valid init ∧ d.w.Valid init ∧ d.u.Valid init ∧ (d.useG = true → d.g.Valid init) ∧ d.h.Valid init ∧
    d.vn.Valid init ∧ (d.useH0 = true → d.h0.Valid init) ∧ (d.storeHt = true → d.ht.Valid init) ∧
    (d.varlen = true → d.cu.Valid init ∧ d.co.Valid init ∧ d.cu.Holds init d.cuVal ∧
      ∀ n, n ≤ d.N → init.read .i64 (d.co.ptr (n : Int)).addr = some (IntV.ofNat (cbase d n)))

/-! ## The specification -/

/-! ### The inputs' values -/

/-- Element `i` of a `bfloat16` buffer, if present, and its value (`0` when absent). -/
def rdB (m : Mem) (b : Buf .bf16) (i : Nat) : Option BF16 := m.read .bf16 (b.ptr (i : Int)).addr
def valB (m : Mem) (b : Buf .bf16) (i : Nat) : ℚ :=
  match rdB m b i with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `float32` buffer, if present, and its value (`0` when absent). -/
def rdF (m : Mem) (b : Buf .f32) (i : Nat) : Option F32 := m.read .f32 (b.ptr (i : Int)).addr
def valF (m : Mem) (b : Buf .f32) (i : Nat) : ℚ :=
  match rdF m b i with
  | some x => x.toRat
  | none => 0

/-- The key head of value head `hv`: `hv // (HV // H)`. -/
def kHead (d : Inst) (hv : Nat) : Nat := hv / (d.HV / d.H)

/-- Token `s` of sequence `n` (`s < len n`): `k[s, c]` of the key head of `hv`, `w[s, c]`,
`u[s, j]` and `g[s]` of value head `hv`, as exact values of the initial memory. -/
def kq (m : Mem) (d : Inst) (n hv s c : Nat) : ℚ :=
  valB m d.k (kIdx d (rowOf d n) (startOf d n + s) (kHead d hv) c)
def wq (m : Mem) (d : Inst) (n hv s c : Nat) : ℚ :=
  valB m d.w (wIdx d (rowOf d n) (startOf d n + s) hv c)
def uq (m : Mem) (d : Inst) (n hv s j : Nat) : ℚ :=
  valB m d.u (uIdx d (rowOf d n) (startOf d n + s) hv j)
def gq (m : Mem) (d : Inst) (n hv s : Nat) : ℚ :=
  valF m d.g (gIdx d (rowOf d n) (startOf d n + s) hv)

/-! ### Chunks -/

/-- The number of tokens of chunk `t` of sequence `n`: `min(64, len − 64t)`. -/
def cl (d : Inst) (n t : Nat) : Nat := min BT (lenOf d n - BT * t)

/-- The gate at chunk `t`'s last token, `g_last` (line 247); `0` without the gate. -/
noncomputable def glast (m : Mem) (d : Inst) (n hv t : Nat) : ℝ :=
  if d.useG then (gq m d n hv (BT * t + cl d n t - 1) : ℝ) else 0

/-- The decay exponent of token `i` of chunk `t`, `g_last − g` (line 250); `0` without the
gate. -/
noncomputable def dgv (m : Mem) (d : Inst) (n hv t i : Nat) : ℝ :=
  if d.useG then glast m d n hv t - (gq m d n hv (BT * t + i) : ℝ) else 0

/-- The chunk's decay `2^g_last` and token `i`'s factor `2^(g_last − g)` (both `1` without the
gate). -/
noncomputable def gam (m : Mem) (d : Inst) (n hv t : Nat) : ℝ := Real.rpow 2 (glast m d n hv t)
noncomputable def Gam (m : Mem) (d : Inst) (n hv t i : Nat) : ℝ := Real.rpow 2 (dgv m d n hv t i)

/-- Token `i` of chunk `t`: `k`, `w`, `u` as reals. -/
noncomputable def kc (m : Mem) (d : Inst) (n hv t i c : Nat) : ℝ := (kq m d n hv (BT * t + i) c : ℝ)
noncomputable def wc (m : Mem) (d : Inst) (n hv t i c : Nat) : ℝ := (wq m d n hv (BT * t + i) c : ℝ)
noncomputable def uc (m : Mem) (d : Inst) (n hv t i j : Nat) : ℝ := (uq m d n hv (BT * t + i) j : ℝ)

/-! ### The stored states, read back -/

/-- Row `c` of column `j` of the stored state `h[cbase n + t, hv]` (chunk `t` of sequence `n`), as
an exact value of the final memory (`0` when absent). -/
noncomputable def Hs (fin : Mem) (d : Inst) (n hv j t c : Nat) : ℝ :=
  (valB fin d.h (hIdx d (cbase d n + t) hv c j) : ℝ)

/-- The initial state's column: `h0[n, hv, :, j]`, or `0` without an initial state. -/
noncomputable def h0v (m : Mem) (d : Inst) (n hv j c : Nat) : ℝ :=
  if d.useH0 then (valF m d.h0 (sIdx d n hv c j) : ℝ) else 0

/-! ### The reference: one exact step of the state -/

/-- The exact new value of token `i` of chunk `t` from the stored state `h[t]`:
`u[i, j] − Σ_c w[i, c]·h[t][c, j]`. -/
noncomputable def vrefS (m fin : Mem) (d : Inst) (n hv j t i : Nat) : ℝ :=
  uc m d n hv t i j - ∑ c ∈ Finset.range d.K, wc m d n hv t i c * Hs fin d n hv j t c

/-- **One exact step of the gated delta rule's chunked recurrence** of chunk `t` (sequence `n`,
value head `hv`, value column `j`) from the state `y` entering it, with the new values from the
stored state `h[t]`: row `c` of `2^g_last·y + Σ_i k[i]ᵀ·2^(g_last − g_i)·(u[i, j] − w[i]·h[t])` over
the chunk's tokens `i`. The arithmetic is exact, `2^x` the real power; the inputs are exact values
of the initial memory and `h[t]` an exact value of the final memory. -/
noncomputable def Step (m fin : Mem) (d : Inst) (n hv j t : Nat) (y : Nat → ℝ) (c : Nat) : ℝ :=
  gam m d n hv t * y c +
    ∑ i ∈ Finset.range (cl d n t), kc m d n hv t i c * Gam m d n hv t i * vrefS m fin d n hv j t i

/-! ### Norms -/

/-- The Euclidean norm of the first `n` entries of `x`. -/
noncomputable def colNorm (n : Nat) (x : Nat → ℝ) : ℝ := Real.sqrt (∑ i ∈ Finset.range n, x i ^ 2)

/-- The spectral norm of the `m × n` matrix `A` (the largest singular value): the least `c ≥ 0`
with `colNorm m (A x) ≤ c · colNorm n x` for every `x`. -/
noncomputable def specNorm (m n : Nat) (A : Nat → Nat → ℝ) : ℝ :=
  sInf {c : ℝ | 0 ≤ c ∧ ∀ x : Nat → ℝ,
    colNorm m (fun a => ∑ b ∈ Finset.range n, A a b * x b) ≤ c * colNorm n x}

/-! ### The tolerances

Simple bounds (README, Tolerances: derived from the device facts as a fine bound, then relaxed):
the new value's tolerance is `2^-8` of `vAbs`, its magnitude `|u − w·h[t]|` with `2^-5` of its
operands' magnitudes `|u| + ‖w‖·‖h[t]‖`; a step's tolerance is `2^-8` of the gated `vAbs` through
`‖kᵀ‖₂`, `2^-15` of the same through `‖k‖_F` and of the decayed state `2^g_last·‖h[t]‖`, and an
absolute floor. Each quantity is a function of the inputs of chunk `t` and of the stored state
`h[t]`; none is carried from one chunk to the next. -/

/-- The constants of the padded rows' bound (`padB`): `float32`'s unit and floor, `2^-126`, the
dot constants at `64`. -/
noncomputable def u32 : ℝ := 2 ^ (-24 : ℤ)
noncomputable def e32 : ℝ := 2 ^ (-150 : ℤ)
noncomputable def mu : ℝ := 2 ^ (-126 : ℤ)
noncomputable def rD : ℝ := (Device.relDot16 64 : ℝ)
noncomputable def aD : ℝ := (Device.absDot16 64 : ℝ)

section Tol

variable (m fin : Mem) (d : Inst) (n hv j t : Nat)

/-- `‖h[t]‖` and `‖liftMag h[t]‖` (column `j`; `liftMag`: a subnormal counted at `2^-126`). -/
noncomputable def hn : ℝ := colNorm d.K (Hs fin d n hv j t)
noncomputable def hl : ℝ :=
  colNorm d.K (fun c => (Device.liftMag .bf16 (valB fin d.h (hIdx d (cbase d n + t) hv c j)) : ℝ))

/-- `‖w[i]‖` under `liftMag`. -/
noncomputable def wl (i : Nat) : ℝ :=
  colNorm d.K (fun c => (Device.liftMag .bf16 (wq m d n hv (BT * t + i) c) : ℝ))

/-- The magnitude of token `i`'s new value for the tolerances: `|vrefS| + 2^-5·(|u| +
‖liftMag w‖·‖liftMag h[t]‖)`, the new value with `2^-5` of its operands' magnitudes. -/
noncomputable def vAbs (i : Nat) : ℝ :=
  |vrefS m fin d n hv j t i| +
    2 ^ (-5 : ℤ) * (|uc m d n hv t i j| + wl m d n hv t i * hl fin d n hv j t)

/-- **The tolerance of `v_new`** at token `i` of chunk `t`: `2^-8·vAbs + 2^-100` (the `w` dots,
the `-` of line 239 and the store's `bfloat16` rounding, line 243). -/
noncomputable def tolV (i : Nat) : ℝ := 2 ^ (-8 : ℤ) * vAbs m fin d n hv j t i + 2 ^ (-100 : ℤ)

/-- The decay `exp2(g_last)`'s error (line 251, `Device.Exp2F32`) and a bound on its result (`0` and
`2^g_last = 1` without the gate); `padB` uses them. -/
noncomputable def errE : ℝ := if d.useG then 2 ^ (-22 : ℤ) * gam m d n hv t + mu else 0
noncomputable def eB : ℝ := gam m d n hv t + errE m d n hv t

/-- `‖liftMag k‖_F` over the chunk's tokens, and `‖kᵀ‖₂` (`K × cl`). -/
noncomputable def kl : ℝ :=
  Real.sqrt (∑ i ∈ Finset.range (cl d n t), ∑ c ∈ Finset.range d.K,
    (Device.liftMag .bf16 (kq m d n hv (BT * t + i) c) : ℝ) ^ 2)
noncomputable def sKn : ℝ := specNorm d.K (cl d n t) (fun a i => kc m d n hv t i a)

/-- **The tolerance of one step of the kernel's `float32` state** (`Y t` entering chunk `t` to
`Y (t + 1)`, the state entering chunk `t + 1`, or `ht` after the last chunk):
`‖Y (t + 1) − Step t (Y t)‖ ≤ errStep t`, with `G = (2^(g_last − g_i)·vAbs i)_i` the gated
magnitudes: `2^-8·‖kᵀ‖₂·‖G‖` (the gated values' roundings through `kᵀ`), `2^-15·(‖k‖_F·‖G‖ +
2^g_last·‖h[t]‖)` (the `k` dot's error and the decay's and the scaling's roundings), and the floor
`2^-36`. -/
noncomputable def errStep : ℝ :=
  2 ^ (-8 : ℤ) * sKn m d n hv t *
      colNorm (cl d n t) (fun i => Gam m d n hv t i * vAbs m fin d n hv j t i) +
    2 ^ (-15 : ℤ) * (kl m d n hv t *
      colNorm (cl d n t) (fun i => Gam m d n hv t i * vAbs m fin d n hv j t i) +
        gam m d n hv t * hn fin d n hv j t) +
    2 ^ (-36 : ℤ)

end Tol

/-- **The bound on the padded state rows** (`K ≤ c < 64·cdiv(K, 64)`, when `64` does not divide
`K`), from the inputs alone: their true value is `0`, their initial value `0` (the masked load),
and each chunk scales them (line 252) and adds a `k` dot whose `k` rows are `0`, which the dot's
facts allow an absolute error `absDot16 64` (and with `STATE_V_FIRST` the fusable `+`'s rounding).
They are never stored, but feed the `w` dot (their `w` columns are `0`), whose fact needs them
finite. The kernel's padded rows entering chunk `t` are within `padB t` when every chunk before
`t` has its stored state in range (`StOk`, which keeps that chunk's gated values finite, so the
`k` dot's fact applies to the padded rows). -/
noncomputable def padB (m : Mem) (d : Inst) (n hv : Nat) : Nat → ℝ
  | 0 => 0
  | t + 1 => (1 + rD + 3 * u32) * (eB m d n hv t * (1 + u32) * padB m d n hv t + e32 + aD) + 4 * e32

/-- The inputs' values covered (README, Conditions; a hypothesis of `Post`'s numeric clauses):
every element of `k`, `w`, `u` (and of `h0` when given) is finite with magnitude at most `2^16`,
every element of `g` (when given) finite with magnitude at most `2^24`; in every chunk `g_last ≤ 8` and
`g_last − g ≤ 8` for each token (real use: `g` is a chunk-local cumulative sum of log decays,
nonincreasing and `≤ 0` up to rounding); and the padded state rows stay in range,
`padB t ≤ 2^100`. -/
def Vals (m : Mem) (d : Inst) : Prop :=
  (∀ i, i < d.B * d.T * d.H * d.K → ∃ x, rdB m d.k i = some x ∧ x.isFinite = true ∧
    |x.toRat| ≤ 2 ^ 16) ∧
  (∀ i, i < d.B * d.T * d.HV * d.K → ∃ x, rdB m d.w i = some x ∧ x.isFinite = true ∧
    |x.toRat| ≤ 2 ^ 16) ∧
  (∀ i, i < d.B * d.T * d.HV * d.V → ∃ x, rdB m d.u i = some x ∧ x.isFinite = true ∧
    |x.toRat| ≤ 2 ^ 16) ∧
  (d.useG = true → ∀ i, i < d.B * d.T * d.HV → ∃ x, rdF m d.g i = some x ∧ x.isFinite = true ∧
    |x.toRat| ≤ 2 ^ 24) ∧
  (d.useH0 = true → ∀ i, i < d.N * d.HV * d.K * d.V → ∃ x, rdF m d.h0 i = some x ∧
    x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 16) ∧
  (∀ n hv t, n < d.N → hv < d.HV → t < nch d n →
    glast m d n hv t ≤ 8 ∧ ∀ i, i < cl d n t → dgv m d n hv t i ≤ 8) ∧
  (∀ n hv t, n < d.N → hv < d.HV → t ≤ nch d n → padB m d n hv t ≤ 2 ^ 100)

/-- **The stored state `h[t]` is in range** (column `j`; chunk `t`'s clauses assume it for `t` and
every earlier chunk): every element is a finite `bfloat16` value and `‖h[t]‖ ≤ 2^40`. -/
def StOk (fin : Mem) (d : Inst) (n hv j t : Nat) : Prop :=
  (∀ c, c < d.K → ∃ x, rdB fin d.h (hIdx d (cbase d n + t) hv c j) = some x ∧
    x.isFinite = true) ∧
  colNorm d.K (Hs fin d n hv j t) ≤ 2 ^ 40

/-- What the launch leaves, for every sequence `n`, value head `hv` and value column `j`:
* `v_new[token s of n, hv, j]` (`s < len n`, chunk `t = s / 64`, token `i = s % 64`): a `bfloat16`
  value; when `Vals` and `h[0]`, …, `h[t]` are in range (`StOk t'` for every `t' ≤ t`), finite
  and within `tolV t i` of `vrefS t i`;
* every element of `h[cbase n + t, hv, :, j]` (`t < nch n`) holds a `bfloat16` value;
* **the states**: there are `float32` states `Y 0, Y 1, …, Y (nch n)` (rows `c < K`; the kernel's
  `b_h`) such that
  - when `Vals`: `Y 0` is the initial state (`0` without one), finite;
  - **each stored state is the rounding of its state**: for `t < nch n`, a finite `Y t c` gives
    `h[t][c] = bf16(Y t c)` exactly (`FloatV.cvt`, nearest even);
  - **each state is one step from the previous one**: for `t < nch n`, when `Vals` and `StOk t'`
    for every `t' ≤ t`, `Y t` and `Y (t + 1)` are finite and
    `‖Y (t + 1) − Step t (Y t)‖ ≤ errStep t`;
  - with `STORE_FINAL_STATE`: `ht[n, hv, :, j]` is `Y (nch n)` exactly;
* every byte outside `h`'s, `v_new`'s and (with `STORE_FINAL_STATE`) `ht`'s elements is
  unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ n hv j, n < d.N → hv < d.HV → j < d.V →
    (∀ s, s < lenOf d n →
      ∃ x, rdB fin d.vn (uIdx d (rowOf d n) (startOf d n + s) hv j) = some x ∧
        (Vals init d → (∀ t', t' ≤ s / BT → StOk fin d n hv j t') → x.isFinite = true ∧
          |x.toReal - vrefS init fin d n hv j (s / BT) (s % BT)| ≤
            tolV init fin d n hv j (s / BT) (s % BT))) ∧
    (∀ t, t < nch d n → ∀ c, c < d.K → ∃ x, rdB fin d.h (hIdx d (cbase d n + t) hv c j) = some x) ∧
    ∃ Y : Nat → Nat → F32,
      (Vals init d → ∀ c, c < d.K → (Y 0 c).isFinite = true ∧ (Y 0 c).toReal = h0v init d n hv j c) ∧
      (∀ t, t < nch d n → ∀ c, c < d.K → (Y t c).isFinite = true →
        rdB fin d.h (hIdx d (cbase d n + t) hv c j) = some (FloatV.cvt (Y t c))) ∧
      (∀ t, t < nch d n → Vals init d → (∀ t', t' ≤ t → StOk fin d n hv j t') →
        (∀ c, c < d.K → (Y t c).isFinite = true ∧ (Y (t + 1) c).isFinite = true) ∧
        colNorm d.K (fun c => (Y (t + 1) c).toReal -
          Step init fin d n hv j t (fun c' => (Y t c').toReal) c) ≤ errStep init fin d n hv j t) ∧
      (d.storeHt = true → ∀ c, c < d.K → rdF fin d.ht (sIdx d n hv c j) = some (Y (nch d n) c))) ∧
  ∀ z, z ∉ d.h.elems {i | i < d.B * NTt d * d.HV * d.K * d.V} ∪
      d.vn.elems {i | i < d.B * d.T * d.HV * d.V} ∪
      (if d.storeHt then d.ht.elems {i | i < d.N * d.HV * d.K * d.V} else ∅) →
    fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` ends
normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Coverage of the setup

`mkF` is a fixed-length launch (`B` rows of `T` tokens) and `mkV` a variable-length one (the
sequences' lengths `lens`, `cu_seqlens` of dtype `dt`), each with `H`, `HV`, `K`, `V` and the two
flags; the ten storages are contiguous, at the byte addresses `i·2^40`, `i = 1 … 10` (`k`, `w`,
`u`, `g`, `h`, `v_new`, `h0`, `ht`, `cu_seqlens`, `chunk_offsets`), each exactly as long as its
tensor. The contents are not part of `Inst.ok`. -/

/-- `cu_seqlens[i] = Σ_{r<i} lens[r]`. -/
def cuOf (lens : List Nat) (i : Nat) : Nat := (lens.take i).sum

/-- The batch's chunks, `Σ cdiv(len, 64)` (`len(chunk_indices)`). -/
def nchSum (lens : List Nat) : Nat := (lens.map (fun l => (l + 63) / 64)).sum

def mkF (B T H HV K V : Nat) (useH0 storeHt vf : Bool) : Inst where
  varlen := false
  useG := true
  useH0 := useH0
  storeHt := storeHt
  vf := vf
  B := B
  T := T
  H := H
  HV := HV
  K := K
  V := V
  N := B
  cuD := .i64
  cuVal := fun _ => DType.i64.ofBits 0
  k := ⟨1 * 2 ^ 40, B * T * H * K⟩
  w := ⟨2 * 2 ^ 40, B * T * HV * K⟩
  u := ⟨3 * 2 ^ 40, B * T * HV * V⟩
  g := ⟨4 * 2 ^ 40, B * T * HV⟩
  h := ⟨5 * 2 ^ 40, B * ((T + 63) / 64) * HV * K * V⟩
  vn := ⟨6 * 2 ^ 40, B * T * HV * V⟩
  h0 := ⟨7 * 2 ^ 40, B * HV * K * V⟩
  ht := ⟨8 * 2 ^ 40, B * HV * K * V⟩
  cu := ⟨9 * 2 ^ 40, 0⟩
  co := ⟨10 * 2 ^ 40, 0⟩

def mkV (lens : List Nat) (H HV K V : Nat) (dt : DType) (useH0 storeHt vf : Bool) : Inst where
  varlen := true
  useG := true
  useH0 := useH0
  storeHt := storeHt
  vf := vf
  B := 1
  T := lens.sum
  H := H
  HV := HV
  K := K
  V := V
  N := lens.length
  cuD := dt
  cuVal := fun i => dt.ofBits (cuOf lens i)
  k := ⟨1 * 2 ^ 40, lens.sum * H * K⟩
  w := ⟨2 * 2 ^ 40, lens.sum * HV * K⟩
  u := ⟨3 * 2 ^ 40, lens.sum * HV * V⟩
  g := ⟨4 * 2 ^ 40, lens.sum * HV⟩
  h := ⟨5 * 2 ^ 40, nchSum lens * HV * K * V⟩
  vn := ⟨6 * 2 ^ 40, lens.sum * HV * V⟩
  h0 := ⟨7 * 2 ^ 40, lens.length * HV * K * V⟩
  ht := ⟨8 * 2 ^ 40, lens.length * HV * K * V⟩
  cu := ⟨9 * 2 ^ 40, lens.length + 1⟩
  co := ⟨10 * 2 ^ 40, lens.length + 1⟩

/-- Two buffers at disjoint byte ranges share no byte. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- Two buffers at `i·2^40` and `j·2^40`, `i ≠ j`, each at most `2^40` bytes long, share no
byte. -/
theorem disj_far {d1 d2 : DType} (b : Buf d1) (c : Buf d2) (i j : Nat)
    (hb : b.base = i * 2 ^ 40) (hc : c.base = j * 2 ^ 40)
    (hbl : b.len * d1.bytes ≤ 2 ^ 40) (hcl : c.len * d2.bytes ≤ 2 ^ 40) (hij : i ≠ j) :
    Disjoint b.bytes c.bytes := by
  apply bytes_disj
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · left
    have : i * 2 ^ 40 + 2 ^ 40 ≤ j * 2 ^ 40 := by
      have := Nat.mul_le_mul_right (2 ^ 40) (Nat.succ_le_of_lt h); simpa [Nat.succ_mul] using this
    rw [hb, hc]; omega
  · right
    have : j * 2 ^ 40 + 2 ^ 40 ≤ i * 2 ^ 40 := by
      have := Nat.mul_le_mul_right (2 ^ 40) (Nat.succ_le_of_lt h); simpa [Nat.succ_mul] using this
    rw [hb, hc]; omega

/-- An instance whose sizes and tables satisfy the decidable conditions, with its ten storages at
`i·2^40` (in the order above) and each at most `2^40` bytes, is covered. -/
theorem ok_of (d : Inst)
    (hA : 0 < d.H ∧ 0 < d.HV ∧ d.H ∣ d.HV ∧ d.HV < 2 ^ 16 ∧ 0 < d.K ∧ d.K ≤ 256 ∧ 0 < d.V ∧
      d.V < 2 ^ 16 ∧ d.HV * d.V < 2 ^ 31 ∧ 0 < d.N ∧ gridN d < 2 ^ 31 ∧ d.T + BT < 2 ^ 31 ∧
      (d.useG = true → d.T * d.HV < 2 ^ 31) ∧ d.T ≤ 2 ^ 22 ∧ d.N ≤ 8192 ∧ d.B ≤ 8192)
    (hB : (d.varlen = false → d.N = d.B) ∧ (d.varlen = true → d.B = 1) ∧
      (d.varlen = true → d.cuD = .i32 ∨ d.cuD = .i64) ∧
      (d.varlen = true → entryInt d.cuD (d.cuVal 0) = 0) ∧
      (d.varlen = true → ∀ n, n < d.N →
        entryInt d.cuD (d.cuVal n) ≤ entryInt d.cuD (d.cuVal (n + 1))) ∧
      (d.varlen = true → entryInt d.cuD (d.cuVal d.N) = d.T))
    (hC : d.B * d.T * d.H * d.K ≤ d.k.len ∧ d.B * d.T * d.HV * d.K ≤ d.w.len ∧
      d.B * d.T * d.HV * d.V ≤ d.u.len ∧ (d.useG = true → d.B * d.T * d.HV ≤ d.g.len) ∧
      d.B * NTt d * d.HV * d.K * d.V ≤ d.h.len ∧ d.B * d.T * d.HV * d.V ≤ d.vn.len ∧
      (d.useH0 = true → d.N * d.HV * d.K * d.V ≤ d.h0.len) ∧
      (d.storeHt = true → d.N * d.HV * d.K * d.V ≤ d.ht.len) ∧
      (d.varlen = true → d.N + 1 ≤ d.cu.len) ∧ (d.varlen = true → d.N + 1 ≤ d.co.len))
    (hD : d.k.base = 1 * 2 ^ 40 ∧ d.w.base = 2 * 2 ^ 40 ∧ d.u.base = 3 * 2 ^ 40 ∧
      d.g.base = 4 * 2 ^ 40 ∧ d.h.base = 5 * 2 ^ 40 ∧ d.vn.base = 6 * 2 ^ 40 ∧
      d.h0.base = 7 * 2 ^ 40 ∧ d.ht.base = 8 * 2 ^ 40 ∧ d.cu.base = 9 * 2 ^ 40 ∧
      d.co.base = 10 * 2 ^ 40)
    (hE : d.k.len * 2 ≤ 2 ^ 40 ∧ d.w.len * 2 ≤ 2 ^ 40 ∧ d.u.len * 2 ≤ 2 ^ 40 ∧
      d.g.len * 4 ≤ 2 ^ 40 ∧ d.h.len * 2 ≤ 2 ^ 40 ∧ d.vn.len * 2 ≤ 2 ^ 40 ∧
      d.h0.len * 4 ≤ 2 ^ 40 ∧ d.ht.len * 4 ≤ 2 ^ 40 ∧ d.cu.len * d.cuD.bytes ≤ 2 ^ 40 ∧
      d.co.len * 8 ≤ 2 ^ 40) : d.ok := by
  obtain ⟨a1, a0, a2, a3, a4, a5, a6, a7, a12, a8, a9, a10, a11, a13, a14, a15⟩ := hA
  obtain ⟨b1, b2, b3, b4, b5, b6⟩ := hB
  obtain ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10⟩ := hC
  obtain ⟨dk, dw, du, dg, dh, dvn, dh0, dht, dcu, dco⟩ := hD
  obtain ⟨ek, ew, eu, eg, eh, evn, eh0, eht, ecu, eco⟩ := hE
  exact
    { H_pos := a1, HV_pos := a0, H_dvd := a2, HV_lt := a3, K_pos := a4, K_le := a5, V_pos := a6,
      V_lt := a7, HVV_lt := a12,
      N_pos := a8, grid_lt := a9, T_lt := a10, T_ctx := a13, N_batch := ⟨a14, a15⟩,
      THV_lt := a11, fixed_N := b1, var_B := b2,
      var_cuD := b3, var_cu0 := b4, var_mono := b5, var_end := b6,
      k_len := c1, w_len := c2, u_len := c3, g_len := c4, h_len := c5, vn_len := c6,
      h0_len := c7, ht_len := c8, cu_len := c9, co_len := c10,
      disj_h := ⟨disj_far _ _ 5 1 dh dk eh ek (by decide), disj_far _ _ 5 2 dh dw eh ew (by decide),
        disj_far _ _ 5 3 dh du eh eu (by decide), disj_far _ _ 5 4 dh dg eh eg (by decide),
        disj_far _ _ 5 6 dh dvn eh evn (by decide), disj_far _ _ 5 7 dh dh0 eh eh0 (by decide),
        disj_far _ _ 5 8 dh dht eh eht (by decide), disj_far _ _ 5 9 dh dcu eh ecu (by decide),
        disj_far _ _ 5 10 dh dco eh eco (by decide)⟩,
      disj_vn := ⟨disj_far _ _ 6 1 dvn dk evn ek (by decide),
        disj_far _ _ 6 2 dvn dw evn ew (by decide), disj_far _ _ 6 3 dvn du evn eu (by decide),
        disj_far _ _ 6 4 dvn dg evn eg (by decide), disj_far _ _ 6 7 dvn dh0 evn eh0 (by decide),
        disj_far _ _ 6 8 dvn dht evn eht (by decide), disj_far _ _ 6 9 dvn dcu evn ecu (by decide),
        disj_far _ _ 6 10 dvn dco evn eco (by decide)⟩,
      disj_ht := ⟨disj_far _ _ 8 1 dht dk eht ek (by decide),
        disj_far _ _ 8 2 dht dw eht ew (by decide), disj_far _ _ 8 3 dht du eht eu (by decide),
        disj_far _ _ 8 4 dht dg eht eg (by decide), disj_far _ _ 8 7 dht dh0 eht eh0 (by decide),
        disj_far _ _ 8 9 dht dcu eht ecu (by decide),
        disj_far _ _ 8 10 dht dco eht eco (by decide)⟩ }

/-! The covered instances. Fixed length (`mkF B T H HV K V useH0 storeHt`): the shapes of fla's
`test_chunk` (`tests/ops/test_gdn.py:149-169`) and `test_chunk_with_chunk_size` at chunk size 64,
in `bfloat16`, with an initial state and the final state (as the tests run them); Qwen3-Next's
heads (`H = 16`, `HV = 32`, `K = V = 128`) with the four flag combinations; fla's default
GatedDeltaNet layer (`H = HV = 6`, `K = 256`, `V = 512`). Variable length (`mkV lens H HV K V dt
useH0 storeHt`): the sequences of `test_chunk_varlen` (`test_gdn.py:752-771`) and
`test_chunk_varlen_prefill` (`test_gdn.py:848-863`), `int64` `cu_seqlens` (the tests') and
`int32` (fla's layers); Qwen3-Next's heads on a packed batch, with the four flag combinations. -/

def setup_f1 : Inst := mkF 4 1024 4 4 128 128 true true false
def setup_f2 : Inst := mkF 2 75 4 4 64 64 true true false
def setup_f3 : Inst := mkF 2 500 3 3 60 60 true true false
def setup_f4 : Inst := mkF 2 1000 3 3 64 64 true true false
def setup_f5 : Inst := mkF 3 1024 4 4 100 100 true true false
def setup_f6 : Inst := mkF 2 1500 4 4 128 128 true true false
def setup_f7 : Inst := mkF 4 2048 8 8 64 64 true true false
def setup_f8 : Inst := mkF 2 256 2 4 64 64 true true false
def setup_f9 : Inst := mkF 2 512 2 8 64 64 true true false
def setup_f10 : Inst := mkF 2 1024 4 8 128 128 true true false
def setup_f11 : Inst := mkF 1 64 2 4 32 32 true true false
def setup_qf_00 : Inst := mkF 2 4096 16 32 128 128 false false false
def setup_qf_01 : Inst := mkF 2 4096 16 32 128 128 false true false
def setup_qf_10 : Inst := mkF 2 4096 16 32 128 128 true false false
def setup_qf_11 : Inst := mkF 2 4096 16 32 128 128 true true false
def setup_gdn_f : Inst := mkF 2 2048 6 6 256 512 true true false

def setup_v1 : Inst := mkV [15] 4 4 60 60 .i64 true true false
def setup_v2 : Inst := mkV [256, 244, 500] 4 4 64 64 .i64 true true false
def setup_v3 : Inst := mkV [15, 85, 200, 900, 800] 4 4 100 100 .i64 true true false
def setup_v4 : Inst := mkV [256, 244, 500] 2 4 64 64 .i64 true true false
def setup_v5 : Inst := mkV [256, 244, 500] 2 8 64 64 .i64 true true false
def setup_v6 : Inst := mkV [8192] 4 4 60 60 .i64 true true false
def setup_v7 : Inst := mkV [256, 244, 500] 4 4 64 64 .i32 true true false
def setup_qv_00 : Inst := mkV [1000, 3000, 0, 1, 4096] 16 32 128 128 .i32 false false false
def setup_qv_01 : Inst := mkV [1000, 3000, 0, 1, 4096] 16 32 128 128 .i32 false true false
def setup_qv_10 : Inst := mkV [1000, 3000, 0, 1, 4096] 16 32 128 128 .i64 true false false
def setup_qv_11 : Inst := mkV [1000, 3000, 0, 1, 4096] 16 32 128 128 .i64 true true false
def setup_gdn_v : Inst := mkV [2000, 48, 2048] 6 6 256 512 .i32 true true false

/-! `STATE_V_FIRST` (`mkF … true`, `mkV … true`): `test_chunk_state_v_first`'s shapes
(`test_gdn.py:414-425`: `(B, T, H, D)` = `(1, 63, 1, 64)`, `(2, 500, 3, 60)`, `(3, 1024, 4, 128)`,
`(4, 2048, 8, 64)`); fla's GatedDeltaNet layer (`fla/layers/gated_deltanet.py:302-334`: 6 heads,
`K = 256`, `V = 512`) on a packed batch with the four flag combinations and on a fixed-length one;
Qwen3-Next's heads on a packed batch. -/
def setup_vf1 : Inst := mkF 1 63 1 1 64 64 true true true
def setup_vf2 : Inst := mkF 2 500 3 3 60 60 true true true
def setup_vf3 : Inst := mkF 3 1024 4 4 128 128 true true true
def setup_vf4 : Inst := mkF 4 2048 8 8 64 64 true true true
def setup_vf_gdn_00 : Inst := mkV [2000, 48, 2048] 6 6 256 512 .i32 false false true
def setup_vf_gdn_01 : Inst := mkV [2000, 48, 2048] 6 6 256 512 .i32 false true true
def setup_vf_gdn_10 : Inst := mkV [2000, 48, 2048] 6 6 256 512 .i32 true false true
def setup_vf_gdn_11 : Inst := mkV [2000, 48, 2048] 6 6 256 512 .i32 true true true
def setup_vf_gdn_f : Inst := mkF 2 2048 6 6 256 512 false true true
def setup_vf_qv : Inst := mkV [1000, 3000, 0, 1, 4096] 16 32 128 128 .i64 true true true

/-! Without the gate (`USE_G` false, `g = None`: the delta rule's forward,
`fla/ops/delta_rule/chunk.py:39-49`): `noG` of an instance. The shapes of fla's delta-rule
`test_chunk` (`tests/ops/test_delta.py:21-26`: `(B, T, H, D)` = `(1, 63, 1, 64)`, `(2, 100, 4, 60)`,
`(2, 1000, 3, 128)`, `(2, 1024, 4, 128)`, `(3, 2000, 4, 128)`, `(4, 2048, 8, 64)`) and
`test_chunk_with_chunk_size` at chunk size 64 (`(1, 64, 2, 32)`, `test_delta.py:95`); the sequences
of `test_chunk_varlen` (`test_delta.py:167-170`, `int32` `cu_seqlens`); fla's default DeltaNet
layer (`fla/models/delta_net/configuration_delta_net.py:26-35`: 16 heads, `K = V = 128`) fixed and
packed with the four flag combinations; grouped value heads; the `[V, K]` layout. -/
def noG (d : Inst) : Inst := { d with useG := false }

def setup_d1 : Inst := noG (mkF 1 63 1 1 64 64 true true false)
def setup_d2 : Inst := noG (mkF 2 100 4 4 60 60 true true false)
def setup_d3 : Inst := noG (mkF 2 1000 3 3 128 128 true true false)
def setup_d4 : Inst := noG (mkF 2 1024 4 4 128 128 true true false)
def setup_d5 : Inst := noG (mkF 3 2000 4 4 128 128 true true false)
def setup_d6 : Inst := noG (mkF 4 2048 8 8 64 64 true true false)
def setup_d7 : Inst := noG (mkF 1 64 2 2 32 32 true true false)
def setup_dv1 : Inst := noG (mkV [15] 2 2 64 64 .i32 true true false)
def setup_dv2 : Inst := noG (mkV [111, 389] 3 3 60 60 .i32 true true false)
def setup_dv3 : Inst := noG (mkV [256, 244, 400, 100] 3 3 64 64 .i32 true true false)
def setup_dv4 : Inst := noG (mkV [15, 85, 200, 900, 399, 201, 200] 4 4 100 100 .i32 true true false)
def setup_dn_f : Inst := noG (mkF 2 4096 16 16 128 128 false false false)
def setup_dn_00 : Inst := noG (mkV [1000, 3000, 0, 1, 4096] 16 16 128 128 .i32 false false false)
def setup_dn_01 : Inst := noG (mkV [1000, 3000, 0, 1, 4096] 16 16 128 128 .i32 false true false)
def setup_dn_10 : Inst := noG (mkV [1000, 3000, 0, 1, 4096] 16 16 128 128 .i64 true false false)
def setup_dn_11 : Inst := noG (mkV [1000, 3000, 0, 1, 4096] 16 16 128 128 .i32 true true false)
def setup_d_gv : Inst := noG (mkV [256, 244, 500] 2 4 64 64 .i64 true true false)
def setup_d_vf : Inst := noG (mkV [256, 244, 500] 2 4 64 64 .i64 true true true)

theorem setup_f1_ok : setup_f1.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f2_ok : setup_f2.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f3_ok : setup_f3.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f4_ok : setup_f4.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f5_ok : setup_f5.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f6_ok : setup_f6.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f7_ok : setup_f7.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f8_ok : setup_f8.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f9_ok : setup_f9.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f10_ok : setup_f10.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_f11_ok : setup_f11.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qf_00_ok : setup_qf_00.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qf_01_ok : setup_qf_01.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qf_10_ok : setup_qf_10.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qf_11_ok : setup_qf_11.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_gdn_f_ok : setup_gdn_f.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v1_ok : setup_v1.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v2_ok : setup_v2.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v3_ok : setup_v3.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v4_ok : setup_v4.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v5_ok : setup_v5.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v6_ok : setup_v6.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_v7_ok : setup_v7.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qv_00_ok : setup_qv_00.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qv_01_ok : setup_qv_01.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qv_10_ok : setup_qv_10.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_qv_11_ok : setup_qv_11.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_gdn_v_ok : setup_gdn_v.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)

theorem setup_vf1_ok : setup_vf1.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf2_ok : setup_vf2.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf3_ok : setup_vf3.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf4_ok : setup_vf4.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf_gdn_00_ok : setup_vf_gdn_00.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf_gdn_01_ok : setup_vf_gdn_01.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf_gdn_10_ok : setup_vf_gdn_10.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf_gdn_11_ok : setup_vf_gdn_11.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf_gdn_f_ok : setup_vf_gdn_f.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_vf_qv_ok : setup_vf_qv.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)

theorem setup_d1_ok : setup_d1.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d2_ok : setup_d2.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d3_ok : setup_d3.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d4_ok : setup_d4.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d5_ok : setup_d5.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d6_ok : setup_d6.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d7_ok : setup_d7.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dv1_ok : setup_dv1.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dv2_ok : setup_dv2.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dv3_ok : setup_dv3.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dv4_ok : setup_dv4.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dn_f_ok : setup_dn_f.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dn_00_ok : setup_dn_00.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dn_01_ok : setup_dn_01.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dn_10_ok : setup_dn_10.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_dn_11_ok : setup_dn_11.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d_gv_ok : setup_d_gv.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
theorem setup_d_vf_ok : setup_d_vf.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem valid_zero {dt : DType} (b : Buf dt) (hb : 0 ≤ b.base)
    (h2 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (ha : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨hb, h2, ha, fun _ _ => by simp [zeroMem]⟩

/-- The smallest fixed-length launch: one row of one token, one head, `K = V = 1`, no initial or
final state. -/
def setup_min : Inst := mkF 1 1 1 1 1 1 false false false

theorem setup_min_ok : setup_min.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_min, zeroMem, setup_min_ok, ?_⟩
  show Pre setup_min zeroMem
  refine ⟨valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), fun _ => valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    fun h => by simp [setup_min, mkF] at h, fun h => by simp [setup_min, mkF] at h,
    fun h => by simp [setup_min, mkF] at h⟩

end Claims.ChunkDeltaH
