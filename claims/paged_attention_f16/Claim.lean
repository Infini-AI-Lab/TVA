import TVA

/-!
# `paged_attention_f16`: the prepared claim

Source: `source/paged_attention.py`. The kernel is `paged_attention_kernel` (lines 30–72); it reads
the module-level names `triton` and `tl` (lines 26–27). The host (`paged_attention`, lines 75–97)
launches it on `float16` tensors `q` (`[num_seqs, num_heads, head_dim]`), `k_cache` and `v_cache`
(`[num_pages, PAGE, num_kv_heads, head_dim]`, the same strides), the `int32` tensors `block_tables`
(`[num_seqs, max_pages]`) and `seq_lens` (`[num_seqs]`), and writes the `float16` output `out =
torch.empty_like(q)`, with the grid `(num_seqs, num_heads)`. The launches to cover are in
`SETUP.md`.

* The model: `paged_attention_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `attnRef`, its scale `attnAbs` and the
  tolerance `tolA`.
* The assumptions: `Assumptions`.
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `canon_ok`, `setup_*_ok`.
-/

namespace Claims.PagedAttention

open TVA TVA.Math.Float


/-! ## The model -/

/-- `paged_attention_kernel` (source lines 30–72). The four `float16` tensors are `*fp16`
pointers, `block_tables` and `seq_lens` `*i32` pointers; `scale` is the runtime `float32` scalar
(a Python `float` argument, `FloatV.hostF32` of the double); the eight strides are `int32` (an
integer argument in `[0, 2^31)` other than `1`; at `1` the compile-time constant `1`, which each
use builds in the dtype of the value it multiplies, with the same value `1`, so the model is the
same: see `model`); `QUERIES_PER_KV`,
`PAGE_SIZE`, `HEAD_DIM` are compile-time values (Lean parameters). -/
def paged_attention_kernel (out_ptr q_ptr k_cache_ptr v_cache_ptr : Ptr .f16)
    (block_tables_ptr seq_lens_ptr : Ptr .i32) (scale : F32)
    (stride_qs stride_qh stride_kp stride_kt stride_kh stride_bs stride_os stride_oh : I32)
    (QUERIES_PER_KV PAGE_SIZE HEAD_DIM : Nat) : Kernel where
  body ctx := do
    -- line 42: axis 0 of the grid, an `int32`
    let seq := tl.program_id ctx 0
    -- line 43: axis 1 of the grid
    let head := tl.program_id ctx 1
    -- line 44: `int32` `//` by the constexpr, built in `int32` (`tl.idiv`, truncating)
    let kv_head := (← tl.idiv (Tensor.scalar head) (Tensor.scalar (IntV.ofNat QUERIES_PER_KV : I32))).item
    -- line 45: the `*i32` pointer advanced by `seq`; one `int32` element
    let seq_len ← tl.load1 (seq_lens_ptr + seq)
    -- lines 47–48
    let offs_t : Tensor [PAGE_SIZE] I32 := tl.arange 0 PAGE_SIZE
    let offs_d : Tensor [HEAD_DIM] I32 := tl.arange 0 HEAD_DIM
    -- line 49: two wrapping `int32` products added to the pointer, then the block of `HEAD_DIM`
    -- lanes; no mask; `float16` to `float32` (exact)
    let q : Tensor [HEAD_DIM] F32 ← tl.cvt (← tl.load (q_ptr + seq * stride_qs + head * stride_qh + offs_d))
    -- line 51: a name assigned the Python `float` `-inf` holds the `float32` scalar `-∞`
    let m_i : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)
    -- line 52: the `float32` `+0`
    let l_i : F32 := FloatV.litF32 (FloatV.ofPat 0)
    -- line 53
    let acc := tl.zeros [HEAD_DIM] .f32
    -- line 54: `range(0, tl.cdiv(seq_len, PAGE_SIZE))`: the bound `(seq_len + (PAGE_SIZE - 1)) //
    -- PAGE_SIZE` in `int32`, evaluated once before the loop; the `int32` induction variable `j`;
    -- the loop carries `acc`, `l_i`, `m_i`
    let (acc, l_i, _m_i) ← tl.forRange (IntV.ofInt 0 : I32)
        (← tl.idiv (Tensor.scalar (seq_len + (IntV.ofNat (PAGE_SIZE - 1) : I32)))
          (Tensor.scalar (IntV.ofNat PAGE_SIZE : I32))).item
        (IntV.ofInt 1 : I32) (acc, l_i, m_i) fun j (acc, l_i, m_i) => do
      -- line 57: one `int32` element of the block table, widened to `int64` (`tl.convert`)
      let page := (← tl.convert .i32 .i64 (Tensor.scalar (← tl.load1 (block_tables_ptr + seq * stride_bs + j)))).item
      -- line 58: signed `int32` comparison with the scalar `seq_len`
      let mask := (j * (IntV.ofNat PAGE_SIZE : I32) + offs_t).ltS seq_len
      -- line 59: `page * stride_kp` is `int64` (`stride_kp` widened); `offs_t[:, None] * stride_kt`
      -- and `kv_head * stride_kh` are `int32` products widened to `int64` for the additions; the
      -- result broadcast to `[PAGE_SIZE, HEAD_DIM]`
      let kv_offs : Tensor [PAGE_SIZE, HEAD_DIM] I64 :=
        page * (← tl.convert .i32 .i64 (Tensor.scalar stride_kp)).item
          + (← tl.convert .i32 .i64 (offs_t.col HEAD_DIM * stride_kt))
          + (← tl.convert .i32 .i64 (Tensor.scalar (kv_head * stride_kh))).item
          + (← tl.convert .i32 .i64 (offs_d.row PAGE_SIZE))
      -- line 60: `other=0.0` is the `float32` `+0`, converted to `float16`; then to `float32`
      let k : Tensor [PAGE_SIZE, HEAD_DIM] F32 ← tl.cvt (← tl.load (k_cache_ptr + kv_offs) (some (mask.col HEAD_DIM))
        (some (← tl.convert .f32 .f16 (tl.full [PAGE_SIZE, HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0))))))
      -- line 61: the product feeds the row sum (`tl.sum1c`); then the IEEE product by `scale`
      let s ← tl.mul (← tl.sum1c (← tl.mul k (q.row PAGE_SIZE))) (bc scale)
      -- line 62: `-inf` built in `float32`, `s`'s dtype
      let s := tl.where mask s (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)))
      -- line 63: `tl.max(s, axis=0)` of the 1-D block, then `tl.maximum` of two scalars
      let m_new := (← tl.maximum (Tensor.scalar m_i) (Tensor.scalar (← tl.maxAll true s))).item
      -- line 64: the IEEE `-` (no product feeds it), then `tl.exp`
      let alpha := (← tl.exp (← tl.sub (Tensor.scalar m_i) (Tensor.scalar m_new))).item
      -- line 65: the IEEE `-` (no product feeds it), then `tl.exp`
      let p ← tl.exp (← tl.sub s (bc m_new))
      -- line 66: `l_i * alpha + tl.sum(p, axis=0)`: the IEEE `tl.sum`, and the addition the
      -- product feeds (`tl.mul_add`)
      let l_i := (← tl.mul_add (Tensor.scalar l_i) (Tensor.scalar alpha) (Tensor.scalar (← tl.sum p))).item
      -- line 67: as line 60
      let v : Tensor [PAGE_SIZE, HEAD_DIM] F32 ← tl.cvt (← tl.load (v_cache_ptr + kv_offs) (some (mask.col HEAD_DIM))
        (some (← tl.convert .f32 .f16 (tl.full [PAGE_SIZE, HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0))))))
      -- line 68: the column sum of the products (`tl.sum0c`); with `PAGE_SIZE ≥ 2` its value is an
      -- addition, and `acc * alpha + …` is `tl.mul_add`; with `PAGE_SIZE = 1` the sum has one lane,
      -- its value is the product `p·v` itself, and the `+` adds two products: `tl.addc` of the IEEE
      -- product `acc * alpha` and the sum
      let pv ← tl.sum0c (← tl.mul (p.col HEAD_DIM) v)
      let acc ← if PAGE_SIZE = 1 then do tl.addc (← tl.mul acc (bc alpha)) pv
        else tl.mul_add acc (bc alpha) pv
      -- line 69
      let m_i := m_new
      pure (acc, l_i, m_i)
    -- line 71: `float32 / float32`, the platform's division
    let out ← tl.div acc (bc l_i)
    -- line 72: `out.to(tl.float16)` (one rounding to nearest even), stored with no mask
    let o16 : Tensor [HEAD_DIM] F16 ← tl.cvt out
    tl.store (out_ptr + seq * stride_os + head * stride_oh + offs_d) o16

/-! ## The conditions -/

/-- What the host chooses: one launch of `paged_attention_kernel` through
`paged_attention(q, k_cache, v_cache, block_tables, seq_lens)` (lines 75–97). The tensors are
given by their storage (`Buf`) and their strides in elements; the last stride of `q`, `k_cache`,
`v_cache`, `block_tables` and `out` is `1` (line 82 asserts it of `q`, `k_cache` and
`block_tables`; `v_cache`'s strides are `k_cache`'s, line 79; `out = torch.empty_like(q)`).
`seq_lens` is taken contiguous: the kernel reads element `s` of its storage (line 45), so the claim
is about `num_seqs` consecutive `int32` values from `seq_lens`'s data pointer, which are the lengths
`seq_lens[s]` when `seq_lens` is contiguous (the test's `torch.tensor`, line 115); the wrapper does
not check it. -/
structure Inst where
  /-- `num_seqs` (`q.shape[0]`): the grid's `x`. -/
  num_seqs : Nat
  /-- `num_heads` (`q.shape[1]`): the grid's `y`. -/
  num_heads : Nat
  /-- `num_kv_heads` (`k_cache.shape[2]`). -/
  num_kv_heads : Nat
  /-- `HEAD_DIM` (`q.shape[2]`, equal to `k_cache.shape[3]`, line 80). -/
  HEAD_DIM : Nat
  /-- `PAGE_SIZE` (`k_cache.shape[1]`). -/
  PAGE_SIZE : Nat
  /-- `num_pages` (`k_cache.shape[0]`). -/
  num_pages : Nat
  /-- `max_pages` (`block_tables.shape[1]`). -/
  max_pages : Nat
  /-- `scale`, the Python `float` (a double; the host passes `head_dim ** -0.5`). -/
  scale : F64
  /-- `q.stride(0)`, `q.stride(1)`. -/
  stride_qs : Nat
  stride_qh : Nat
  /-- `k_cache.stride(0)`, `k_cache.stride(1)`, `k_cache.stride(2)` (also `v_cache`'s). -/
  stride_kp : Nat
  stride_kt : Nat
  stride_kh : Nat
  /-- `block_tables.stride(0)`. -/
  stride_bs : Nat
  /-- `out.stride(0)`, `out.stride(1)`. -/
  stride_os : Nat
  stride_oh : Nat
  /-- The storage of `q`, `k_cache`, `v_cache` and `out` (`float16`). -/
  q : Buf .f16
  k_cache : Buf .f16
  v_cache : Buf .f16
  out : Buf .f16
  /-- The storage of `block_tables` and of `seq_lens` (`int32`; `seq_lens`'s element `s` is the
  length of sequence `s`: contiguous). -/
  block_tables : Buf .i32
  seq_lens : Buf .i32

/-- `QUERIES_PER_KV = num_heads // num_kv_heads` (line 92). -/
def qpk (d : Inst) : Nat := d.num_heads / d.num_kv_heads

/-- Element `(s, h, e)` of `q`. -/
def qIdx (d : Inst) (s h e : Nat) : Nat := s * d.stride_qs + h * d.stride_qh + e

/-- Element `(s, h, e)` of `out`. -/
def outIdx (d : Inst) (s h e : Nat) : Nat := s * d.stride_os + h * d.stride_oh + e

/-- Element `(p, r, g, e)` of `k_cache` and of `v_cache`: page `p`, row `r`, key/value head `g`. -/
def kvIdx (d : Inst) (p r g e : Nat) : Nat := p * d.stride_kp + r * d.stride_kt + g * d.stride_kh + e

/-- Element `(s, j)` of `block_tables`. -/
def btIdx (d : Inst) (s j : Nat) : Nat := s * d.stride_bs + j

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `num_heads % num_kv_heads == 0` (line 80), `num_kv_heads ≥ 1`. -/
  kv_pos : 0 < d.num_kv_heads
  kv_dvd : d.num_kv_heads ∣ d.num_heads
  /-- The grid `(num_seqs, num_heads, 1)` within CUDA's limits (axis 0 below `2^31` also when
  `num_heads = 0`), fewer than `2^31` programs. -/
  heads_lt : d.num_heads < 2 ^ 16
  grid_lt : d.num_seqs * d.num_heads < 2 ^ 31
  seqs_lt : d.num_seqs < 2 ^ 31
  /-- `num_seqs`, the decode batch's sequences, is a batch size: at most `8192`. -/
  seqs_le : d.num_seqs ≤ 8192
  /-- `HEAD_DIM` and `PAGE_SIZE` are powers of two (`tl.arange` rejects anything else on the host),
  at most `256`. -/
  head_dim : ∃ a, a ≤ 8 ∧ d.HEAD_DIM = 2 ^ a
  page_size : ∃ b, b ≤ 8 ∧ d.PAGE_SIZE = 2 ^ b
  /-- At most `2^16` pages per sequence (`2^24` tokens): keeps the tolerance meaningful. -/
  max_pages_le : d.max_pages ≤ 2 ^ 16
  /-- The strides are `int32` arguments (below `2^31`), and the `int32` products of lines 49, 57,
  59 and 72 do not wrap. -/
  strides_lt : d.stride_qs < 2 ^ 31 ∧ d.stride_qh < 2 ^ 31 ∧ d.stride_kp < 2 ^ 31 ∧
    d.stride_kt < 2 ^ 31 ∧ d.stride_kh < 2 ^ 31 ∧ d.stride_bs < 2 ^ 31 ∧ d.stride_os < 2 ^ 31 ∧
    d.stride_oh < 2 ^ 31
  prods_lt : (d.num_seqs - 1) * d.stride_qs < 2 ^ 31 ∧ (d.num_heads - 1) * d.stride_qh < 2 ^ 31 ∧
    (d.num_seqs - 1) * d.stride_os < 2 ^ 31 ∧ (d.num_heads - 1) * d.stride_oh < 2 ^ 31 ∧
    (d.num_seqs - 1) * d.stride_bs < 2 ^ 31 ∧ (d.PAGE_SIZE - 1) * d.stride_kt < 2 ^ 31 ∧
    (d.num_kv_heads - 1) * d.stride_kh < 2 ^ 31
  /-- Every element the indices name lies in its tensor's storage. -/
  q_len : ∀ s h e, s < d.num_seqs → h < d.num_heads → e < d.HEAD_DIM → qIdx d s h e < d.q.len
  out_len : ∀ s h e, s < d.num_seqs → h < d.num_heads → e < d.HEAD_DIM → outIdx d s h e < d.out.len
  kv_len : ∀ p r g e, p < d.num_pages → r < d.PAGE_SIZE → g < d.num_kv_heads → e < d.HEAD_DIM →
    kvIdx d p r g e < d.k_cache.len ∧ kvIdx d p r g e < d.v_cache.len
  bt_len : ∀ s j, s < d.num_seqs → j < d.max_pages → btIdx d s j < d.block_tables.len
  sl_len : d.num_seqs ≤ d.seq_lens.len
  /-- `out`'s elements are distinct (`torch.empty_like(q)` has no overlapping elements). -/
  out_inj : ∀ s h e s' h' e', s < d.num_seqs → h < d.num_heads → e < d.HEAD_DIM →
    s' < d.num_seqs → h' < d.num_heads → e' < d.HEAD_DIM →
    outIdx d s h e = outIdx d s' h' e' → s = s' ∧ h = h' ∧ e = e'
  /-- `out`'s storage shares no byte with the inputs' (a new allocation). -/
  disj : Disjoint d.out.bytes d.q.bytes ∧ Disjoint d.out.bytes d.k_cache.bytes ∧
    Disjoint d.out.bytes d.v_cache.bytes ∧ Disjoint d.out.bytes d.block_tables.bytes ∧
    Disjoint d.out.bytes d.seq_lens.bytes
  /-- `scale` is a finite double in `[2^-64, 2^64]` (the host passes `head_dim ** -0.5`). -/
  scale : d.scale.isFinite = true ∧ 1 / 2 ^ 64 ≤ d.scale.toRat ∧ d.scale.toRat ≤ 2 ^ 64

/-- The model on an instance: the launch of lines 85–96, `QUERIES_PER_KV = num_heads //
num_kv_heads`. An integer argument equal to `1` is the compile-time constant `1`; each use of a
stride (lines 49, 57, 59, 72) is an operand of `*`, which builds the constant in the dtype of the
other operand: `int32` (`seq`, `head`, `offs_t[:, None]`, `kv_head`), or `int64` for line 59's
`page * stride_kp`. Either way it is the value `1`, which the model's `IntV.ofNat 1` (converted to
`int64` for `stride_kp`) also is, so the model is the same. -/
def model (d : Inst) : Kernel :=
  paged_attention_kernel (d.out.ptr 0) (d.q.ptr 0) (d.k_cache.ptr 0) (d.v_cache.ptr 0)
    (d.block_tables.ptr 0) (d.seq_lens.ptr 0) (FloatV.hostF32 d.scale)
    (IntV.ofNat d.stride_qs) (IntV.ofNat d.stride_qh) (IntV.ofNat d.stride_kp) (IntV.ofNat d.stride_kt)
    (IntV.ofNat d.stride_kh) (IntV.ofNat d.stride_bs) (IntV.ofNat d.stride_os) (IntV.ofNat d.stride_oh)
    (qpk d) d.PAGE_SIZE d.HEAD_DIM

/-- The host's launch (lines 84–96): the grid `(num_seqs, num_heads, 1)`; the source's block
shapes; `enable_fp_fusion` is Triton's default (`true`); `num_ctas = 1`, no `launch_pdl`,
`debug = false`. `num_warps = 4` is performance-only and not recorded. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.num_seqs, d.num_heads, 1], fusion := true,
    blocks := [[d.PAGE_SIZE], [d.HEAD_DIM], [d.PAGE_SIZE, 1], [1, d.HEAD_DIM],
      [d.PAGE_SIZE, d.HEAD_DIM]] }

/-- Sequence `s`'s entries in memory `m`: its length `n` (`seq_lens[s]`) is in
`[0, max_pages·PAGE_SIZE]` and is a context length, at most `2^22` tokens (real model shapes,
README, Exclusions), and each block-table entry `block_tables[s, j]` the kernel reads
(`j < cdiv(n, PAGE_SIZE)`, that is `j·PAGE_SIZE < n`) is a page number below `num_pages`. Pages
need not be distinct, and entries the kernel does not read may hold anything. -/
def SeqOk (d : Inst) (m : Mem) (s : Nat) : Prop :=
  ∃ n : I32, m.read .i32 (d.seq_lens.ptr (s : Int)).addr = some n ∧ 0 ≤ n.toInt ∧
    n.toInt ≤ ((d.max_pages * d.PAGE_SIZE : Nat) : Int) ∧ n.toInt ≤ 2 ^ 22 ∧
    ∀ j : Nat, ((j * d.PAGE_SIZE : Nat) : Int) < n.toInt →
      ∃ b : I32, m.read .i32 (d.block_tables.ptr (btIdx d s j : Int)).addr = some b ∧
        0 ≤ b.toInt ∧ b.toInt < (d.num_pages : Int)

/-- The inputs covered: the six buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and every sequence's length and block-table entries are as `SeqOk` says. Any `float16`
values: `Post` says what holds for sequences whose values are covered (`SeqVals`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k_cache.Valid init ∧ d.v_cache.Valid init ∧ d.out.Valid init ∧
    d.block_tables.Valid init ∧ d.seq_lens.Valid init ∧ ∀ s, s < d.num_seqs → SeqOk d init s

/-! ## The specification -/

/-- The value of element `i` of a `float16` buffer in memory `m` (`0` when absent). -/
def val16 (m : Mem) (b : Buf .f16) (i : Nat) : ℚ :=
  match m.read .f16 (b.ptr (i : Int)).addr with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `float16` buffer holds a finite value in memory `m`. -/
def fin16 (m : Mem) (b : Buf .f16) (i : Nat) : Prop :=
  ∃ x : F16, m.read .f16 (b.ptr (i : Int)).addr = some x ∧ x.isFinite = true

/-- The value of element `i` of an `int32` buffer in memory `m` (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- Sequence `s`'s length, `seq_lens[s]`. -/
def seqLen (d : Inst) (m : Mem) (s : Nat) : Nat := (val32 m d.seq_lens s).toNat

/-- The number of pages the kernel visits for sequence `s`: `cdiv(seq_lens[s], PAGE_SIZE)`. -/
def nblocks (d : Inst) (m : Mem) (s : Nat) : Nat := (seqLen d m s + d.PAGE_SIZE - 1) / d.PAGE_SIZE

/-- The page that holds token `t` of sequence `s`: `block_tables[s, t // PAGE_SIZE]`. -/
def pageOf (d : Inst) (m : Mem) (s t : Nat) : Nat :=
  (val32 m d.block_tables (btIdx d s (t / d.PAGE_SIZE))).toNat

/-- The element of `k_cache` and `v_cache` that holds component `e` of token `t` of sequence `s`
for query head `h`: page `pageOf`, row `t % PAGE_SIZE`, key/value head `h // QUERIES_PER_KV`. -/
def tokElem (d : Inst) (m : Mem) (s h t e : Nat) : Nat :=
  kvIdx d (pageOf d m s t) (t % d.PAGE_SIZE) (h / qpk d) e

/-- `scale` as a rational number. -/
def sigma (d : Inst) : ℚ := d.scale.toRat

/-- The exact score `q[s, h] · k[token t]` of the input values. -/
def score (d : Inst) (m : Mem) (s h t : Nat) : ℚ :=
  ∑ e ∈ Finset.range d.HEAD_DIM, val16 m d.q (qIdx d s h e) * val16 m d.k_cache (tokElem d m s h t e)

/-- The score's magnitude `Σ_e |q·k|`, which the score's rounding errors are relative to. -/
def scoreAbs (d : Inst) (m : Mem) (s h t : Nat) : ℚ :=
  ∑ e ∈ Finset.range d.HEAD_DIM, |val16 m d.q (qIdx d s h e) * val16 m d.k_cache (tokElem d m s h t e)|

/-- The values of sequence `s` and head `h` are covered: the sequence has at least one token,
`q`'s row and every key and value of its tokens are finite `float16`, every `|v| ≤ 2^15`, and
every scaled score magnitude `scale · scoreAbs ≤ 2^10`. On the test's inputs (`randn` values,
`scale = head_dim^-0.5`), `scale · scoreAbs` is about `0.64·√head_dim` on average (`7.2` at
`head_dim = 128`) and at most `10.96` on every pair. -/
def SeqVals (d : Inst) (m : Mem) (s h : Nat) : Prop :=
  1 ≤ seqLen d m s ∧
  (∀ e, e < d.HEAD_DIM → fin16 m d.q (qIdx d s h e)) ∧
  (∀ t e, t < seqLen d m s → e < d.HEAD_DIM →
    fin16 m d.k_cache (tokElem d m s h t e) ∧ fin16 m d.v_cache (tokElem d m s h t e) ∧
      |val16 m d.v_cache (tokElem d m s h t e)| ≤ 2 ^ 15) ∧
  (∀ t, t < seqLen d m s → sigma d * scoreAbs d m s h t ≤ 2 ^ 10)

/-- The exact softmax weight of token `t`, unnormalised: `exp(scale · score)`. -/
noncomputable def weight (d : Inst) (m : Mem) (s h t : Nat) : ℝ :=
  Real.exp ((sigma d : ℝ) * (score d m s h t : ℝ))

/-- **The exact attention output**: `Σ_t w_t · v[token t, e] / Σ_t w_t` over the sequence's
tokens `t < seq_lens[s]`, with the keys and values read through the block table. -/
noncomputable def attnRef (d : Inst) (m : Mem) (s h e : Nat) : ℝ :=
  (∑ t ∈ Finset.range (seqLen d m s), weight d m s h t * (val16 m d.v_cache (tokElem d m s h t e) : ℝ)) /
    ∑ t ∈ Finset.range (seqLen d m s), weight d m s h t

/-- The same weighted mean of `|v|`: the scale of the output's tolerance. -/
noncomputable def attnAbs (d : Inst) (m : Mem) (s h e : Nat) : ℝ :=
  (∑ t ∈ Finset.range (seqLen d m s), weight d m s h t * |(val16 m d.v_cache (tokElem d m s h t e) : ℝ)|) /
    ∑ t ∈ Finset.range (seqLen d m s), weight d m s h t

/-! ### The tolerance

`T` is the number of pages the sequence spans (`cdiv(seq_len, PAGE_SIZE)`, `nblocks`) and `X` a
bound on the sequence's scaled score magnitudes (`scale · scoreAbs ≤ X` on its tokens). The
output is within `tolA T X · (attnAbs + |attnRef|) + 2^-11·|attnRef| + 2^-24` of `attnRef`:
`2^-11·|attnRef|` is the output's own `float16` rounding; `tolA` grows with the pages (each
online-softmax update rescales the earlier pages' sums by `alpha`, `T` sequential roundings) and
with the score bound (the exponents' roundings). README, Tolerances, derives it from the library's
facts. -/

/-- **The output's tolerance relative to `attnAbs + |attnRef|`**: `(9·T/8 + 8·X + 8)·2^-21`. -/
def tolA (T : Nat) (X : ℚ) : ℚ := (9 * (T : ℚ) / 8 + 8 * X + 8) / 2 ^ 21

/-- The elements of `out` the launch writes. -/
def outElems (d : Inst) : Set Nat :=
  {x | ∃ s h e, s < d.num_seqs ∧ h < d.num_heads ∧ e < d.HEAD_DIM ∧ x = outIdx d s h e}

/-- What the launch leaves:
* every element `(s, h, e)` of `out` holds a `float16` value `c`; when the sequence's values are
  covered (`SeqVals`), `c` is finite and, for every bound `X ≤ 2^10` on the sequence's scaled score
  magnitudes, within `tolA T X · (attnAbs + |attnRef|) + 2^-11·|attnRef| + 2^-24` of the exact
  attention `attnRef` over the sequence's tokens, read through `block_tables` in `init`
  (`T = nblocks`);
* every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ s h e, s < d.num_seqs → h < d.num_heads → e < d.HEAD_DIM →
    ∃ c : F16, fin.read .f16 (d.out.ptr (outIdx d s h e : Int)).addr = some c ∧
      (SeqVals d init s h → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
        (∀ t, t < seqLen d init s → sigma d * scoreAbs d init s h t ≤ X) →
        |c.toReal - attnRef d init s h e| ≤
          (tolA (nblocks d init s) X : ℝ) * (attnAbs d init s h e + |attnRef d init s h e|) +
            1 / 2 ^ 11 * |attnRef d init s h e| + 1 / 2 ^ 24)) ∧
  ∀ z, z ∉ d.out.elems (outElems d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.ExpF32`: `tl.exp` on `float32` (lines 64, 65). -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.ExpF32NegInf`: `tl.exp(-inf) = +0` on `float32` (line 64 on the first page,
  line 65 on the masked lanes). -/
  exp_ninf : Rules.Num.ExpF32NegInf P
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input. -/
  exp_total : Rules.Num.MathTotal P .exp .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32` (line 71). -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs. -/
  div_total : Rules.Num.DivTotal P .binary32
  /-- [rule] `Num.SumDepth`: a float `tl.sum` over `n` positions combines along a tree of depth
  at most `⌈log₂ n⌉` (lines 61, 66, 68). -/
  sum_depth : Rules.Num.SumDepth P


/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.expF32,
    Platform.sm90ref.expF32NegInf, Platform.ref.mathTotal, Platform.ref.divFull, Platform.ref.divTotal _,
    Platform.sm90ref.sumDepth⟩⟩

/-! ## The host's layout

`canon S H KV D P NP MP sc` is the host's launch (lines 75–97) on contiguous tensors: `q` and
`out` `[S, H, D]`, `k_cache` and `v_cache` `[NP, P, KV, D]`, `block_tables` `[S, MP]`, `seq_lens`
`[S]`, the six storages at `2^40`, `2·2^40`, …, `6·2^40`, and `scale = sc` (the host's
`head_dim ** -0.5`). -/

def canon (S H KV D P NP MP : Nat) (sc : F64) : Inst where
  num_seqs := S
  num_heads := H
  num_kv_heads := KV
  HEAD_DIM := D
  PAGE_SIZE := P
  num_pages := NP
  max_pages := MP
  scale := sc
  stride_qs := H * D
  stride_qh := D
  stride_kp := P * KV * D
  stride_kt := KV * D
  stride_kh := D
  stride_bs := MP
  stride_os := H * D
  stride_oh := D
  q := ⟨2 ^ 40, S * H * D⟩
  k_cache := ⟨2 * 2 ^ 40, NP * P * KV * D⟩
  v_cache := ⟨3 * 2 ^ 40, NP * P * KV * D⟩
  out := ⟨4 * 2 ^ 40, S * H * D⟩
  block_tables := ⟨5 * 2 ^ 40, S * MP⟩
  seq_lens := ⟨6 * 2 ^ 40, S⟩

/-- A mixed-radix index is below its range. -/
theorem mix_lt {a A b B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B :=
  calc a * B + b < a * B + B := by omega
    _ = (a + 1) * B := by ring
    _ ≤ A * B := Nat.mul_le_mul_right _ ha

/-- A mixed-radix index determines its digits. -/
theorem mix_inj {a a' b b' B : Nat} (hb : b < B) (hb' : b' < B) (h : a * B + b = a' * B + b') :
    a = a' ∧ b = b' := by
  have hB : 0 < B := by omega
  have e1 : (a * B + b) / B = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hb, Nat.zero_add]
  have e2 : (a' * B + b') / B = a' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hb', Nat.zero_add]
  have f1 : (a * B + b) % B = b := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have f2 : (a' * B + b') % B = b' := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb']
  exact ⟨by rw [← e1, ← e2, h], by rw [← f1, ← f2, h]⟩

/-- Disjoint byte ranges. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (hbc : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- **The host's launches are covered**: any shapes the wrapper admits (`num_kv_heads ≥ 1`
dividing `num_heads`, head dims and pages powers of two up to `256`),
within CUDA's grid limits, the kernel's `int32` products, up to `8192` sequences and up to `2^16`
pages per sequence, with
contiguous tensors of fewer than `2^38` elements each (the cache may hold more than `2^31`), and a
`scale` in `[2^-64, 2^64]`. The numeric hypotheses are checked by `norm_num` on each launch. -/
theorem canon_ok (S H KV D P NP MP : Nat) (sc : F64)
    (hkv : 0 < KV) (hdvd : KV ∣ H) (hH : H < 2 ^ 16) (hgrid : S * H < 2 ^ 31)
    (hS : S < 2 ^ 31)
    (hD : ∃ a, a ≤ 8 ∧ D = 2 ^ a) (hP : ∃ b, b ≤ 8 ∧ P = 2 ^ b) (hMP : MP ≤ 2 ^ 16)
    (hstr : H * D < 2 ^ 31 ∧ D < 2 ^ 31 ∧ P * KV * D < 2 ^ 31 ∧ KV * D < 2 ^ 31 ∧ D < 2 ^ 31 ∧
      MP < 2 ^ 31 ∧ H * D < 2 ^ 31 ∧ D < 2 ^ 31)
    (hprod : (S - 1) * (H * D) < 2 ^ 31 ∧ (H - 1) * D < 2 ^ 31 ∧ (S - 1) * (H * D) < 2 ^ 31 ∧
      (H - 1) * D < 2 ^ 31 ∧ (S - 1) * MP < 2 ^ 31 ∧ (P - 1) * (KV * D) < 2 ^ 31 ∧
      (KV - 1) * D < 2 ^ 31)
    (hsz : S * H * D ≤ 2 ^ 38 ∧ NP * P * KV * D ≤ 2 ^ 38 ∧ S * MP ≤ 2 ^ 38 ∧ S ≤ 2 ^ 38)
    (hsc : sc.isFinite = true ∧ 1 / 2 ^ 64 ≤ sc.toRat ∧ sc.toRat ≤ 2 ^ 64) (hS13 : S ≤ 8192) :
    (canon S H KV D P NP MP sc).ok := by
  obtain ⟨hs1, hs2, hs3, hs4⟩ := hsz
  have hqi : ∀ s h e, s < S → h < H → e < D → s * (H * D) + h * D + e < S * H * D := by
    intro s h e hs hh he
    have : s * (H * D) + h * D + e = (s * H + h) * D + e := by ring
    rw [this]; exact mix_lt (mix_lt hs hh) he
  refine
    { kv_pos := hkv
      kv_dvd := hdvd
      heads_lt := hH
      grid_lt := hgrid
      seqs_lt := hS
      seqs_le := hS13
      head_dim := hD
      page_size := hP
      max_pages_le := hMP
      strides_lt := hstr
      prods_lt := hprod
      q_len := fun s h e hs hh he => hqi s h e hs hh he
      out_len := fun s h e hs hh he => hqi s h e hs hh he
      kv_len := fun p r g e hp hr hg he => by
        have : p * (P * KV * D) + r * (KV * D) + g * D + e = ((p * P + r) * KV + g) * D + e := by ring
        have h1 := mix_lt (mix_lt (mix_lt hp hr) hg) he
        simp only [kvIdx, canon]; rw [this]; exact ⟨h1, h1⟩
      bt_len := fun s j hs hj => mix_lt hs hj
      sl_len := le_rfl
      out_inj := fun s h e s' h' e' hs hh he hs' hh' he' heq => by
        simp only [outIdx, canon] at heq
        have r1 : s * (H * D) + h * D + e = (s * H + h) * D + e := by ring
        have r2 : s' * (H * D) + h' * D + e' = (s' * H + h') * D + e' := by ring
        rw [r1, r2] at heq
        obtain ⟨h1, h2⟩ := mix_inj he he' heq
        obtain ⟨h3, h4⟩ := mix_inj hh hh' h1
        exact ⟨h3, h4, h2⟩
      disj := by
        refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> exact bytes_disj _ _ (by simp only [canon, DType.bytes]; omega)
      scale := hsc }

/-! ## Coverage: the launches of `SETUP.md`

The test (lines 137–155): `(num_heads, num_kv_heads, head_dim, page_size, seq_lens)` of
`(32, 8, 128, 16, [1, 15, 16, 17, 1000, 4096, 333])`, `(8, 8, 64, 16, [5, 64, 2049])` and
`(16, 2, 128, 32, [31, 32, 33, 8191])`; `num_pages` is the sequences' pages plus `50`,
`max_pages` the largest `cdiv(seq_len, page_size)` (`make_cache`), `scale = head_dim ** -0.5`. -/

/-- `128 ** -0.5` (the double `0x3fb6a09e667f3bcd`). -/
def sc128 : F64 := FloatV.ofPat 0x3FB6A09E667F3BCD
/-- `64 ** -0.5 = 0.125`. -/
def sc64 : F64 := FloatV.ofPat 0x3FC0000000000000

theorem sc128_ok : sc128.isFinite = true ∧ 1 / 2 ^ 64 ≤ sc128.toRat ∧ sc128.toRat ≤ 2 ^ 64 := by
  refine ⟨by decide +kernel, ?_, ?_⟩
  · have : (1 / 2 ^ 64 : ℚ) ≤ sc128.toRat := by decide +kernel
    exact this
  · have : sc128.toRat ≤ 2 ^ 64 := by decide +kernel
    exact this

theorem sc64_ok : sc64.isFinite = true ∧ 1 / 2 ^ 64 ≤ sc64.toRat ∧ sc64.toRat ≤ 2 ^ 64 := by
  refine ⟨by decide +kernel, ?_, ?_⟩
  · have : (1 / 2 ^ 64 : ℚ) ≤ sc64.toRat := by decide +kernel
    exact this
  · have : sc64.toRat ≤ 2 ^ 64 := by decide +kernel
    exact this

/-- The test's first configuration: 7 sequences, 32/8 heads, head dim 128, pages of 16
(`1 + 1 + 1 + 2 + 63 + 256 + 21 = 345` pages, `max_pages = 256`, `num_pages = 395`). -/
def setup_t1 : Inst := canon 7 32 8 128 16 395 256 sc128
theorem setup_t1_ok : setup_t1.ok :=
  canon_ok 7 32 8 128 16 395 256 sc128 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
    ⟨7, by norm_num, by norm_num⟩ ⟨4, by norm_num, by norm_num⟩ (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) sc128_ok (by norm_num)

/-- The second: 3 sequences, 8/8 heads, head dim 64, pages of 16 (`1 + 4 + 129 = 134` pages,
`max_pages = 129`, `num_pages = 184`). -/
def setup_t2 : Inst := canon 3 8 8 64 16 184 129 sc64
theorem setup_t2_ok : setup_t2.ok :=
  canon_ok 3 8 8 64 16 184 129 sc64 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
    ⟨6, by norm_num, by norm_num⟩ ⟨4, by norm_num, by norm_num⟩ (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) sc64_ok (by norm_num)

/-- The third: 4 sequences, 16/2 heads, head dim 128, pages of 32 (`1 + 1 + 2 + 256 = 260` pages,
`max_pages = 256`, `num_pages = 310`). -/
def setup_t3 : Inst := canon 4 16 2 128 32 310 256 sc128
theorem setup_t3_ok : setup_t3.ok :=
  canon_ok 4 16 2 128 32 310 256 sc128 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
    ⟨7, by norm_num, by norm_num⟩ ⟨5, by norm_num, by norm_num⟩ (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) sc128_ok (by norm_num)

/-- A cache of more than `2^31` elements (`2^17 + 64` pages of `16·8·128`), read through `int64`
page offsets, at the first configuration's heads. -/
def setup_bigcache : Inst := canon 7 32 8 128 16 (2 ^ 17 + 64) 256 sc128
theorem setup_bigcache_ok : setup_bigcache.ok :=
  canon_ok 7 32 8 128 16 (2 ^ 17 + 64) 256 sc128 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
    ⟨7, by norm_num, by norm_num⟩ ⟨4, by norm_num, by norm_num⟩ (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) sc128_ok (by norm_num)

/-- Pages of one token (`2^b` with `b = 0`): 4 sequences of `[1, 2, 1000, 4097]` tokens, 8/2
heads, head dim 128 (`5100` pages plus `50`, `max_pages = 4097`). -/
def setup_page1 : Inst := canon 4 8 2 128 1 5150 4097 sc128
theorem setup_page1_ok : setup_page1.ok :=
  canon_ok 4 8 2 128 1 5150 4097 sc128 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
    ⟨7, by norm_num, by norm_num⟩ ⟨0, by norm_num, by norm_num⟩ (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) sc128_ok (by norm_num)

/-! ## Satisfiable

One sequence, one head, `HEAD_DIM = 128`, `PAGE_SIZE = 16`, one page, in a memory of zeros
(`seq_lens[0] = 0`). -/

/-- The instance. -/
def testInst : Inst := canon 1 1 1 128 16 1 1 sc128

theorem testInst_ok : testInst.ok :=
  canon_ok 1 1 1 128 16 1 1 sc128 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)
    ⟨7, by norm_num, by norm_num⟩ ⟨4, by norm_num, by norm_num⟩ (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) sc128_ok (by norm_num)

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

/-- Every aligned `int32` read of `testMem` gives `0`. -/
theorem testMem_read {a : Int} (ha : Mem.Aligned .i32 a) :
    ∃ v : I32, testMem.read .i32 a = some v ∧ v.toInt = 0 := by
  have hb : testMem.readBytes a 4 = some [0, 0, 0, 0] := by
    simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.i32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb, Option.map_some]
  exact ⟨_, rfl, by decide⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid testMem := fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [testMem]⟩
  refine ⟨testInst, testMem, testInst_ok, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals try exact hv _ (by simp [testInst, canon]) (by simp [testInst, canon, DType.bytes]) (by simp [testInst, canon, DType.bytes])
  intro s hs
  have hs0 : s = 0 := by simp only [testInst, canon] at hs; omega
  subst hs0
  obtain ⟨v, hrd, hv0⟩ := testMem_read (a := (testInst.seq_lens.ptr ((0 : ℕ) : ℤ)).addr)
    ⟨by simp [testInst, canon, Buf.ptr, DType.bytes], by simp [testInst, canon, Buf.ptr, DType.bytes],
      by simp [testInst, canon, Buf.ptr, DType.bytes]⟩
  refine ⟨v, hrd, by omega, by rw [hv0]; positivity, by rw [hv0]; norm_num, fun j hj => ?_⟩
  rw [hv0] at hj
  omega

end Claims.PagedAttention
