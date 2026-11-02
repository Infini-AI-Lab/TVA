import TVA

/-!
# `paged_attention_f16`: the model, the conditions, the specification and the assumptions

Source: `source/paged_attention.py`. The kernel is `paged_attention_kernel` (lines 30–72); it reads
the module-level names `triton` and `tl` (lines 26–27). The host (`paged_attention`, lines 75–97)
launches it on `float16` tensors `q` (`[num_seqs, num_heads, head_dim]`), `k_cache` and `v_cache`
(`[num_pages, PAGE, num_kv_heads, head_dim]`, the same strides), the `int32` tensors `block_tables`
(`[num_seqs, max_pages]`) and `seq_lens` (`[num_seqs]`), and writes the `float16` output `out =
torch.empty_like(q)`, with the grid `(num_seqs, num_heads)`.

* The model: `paged_attention_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `attnRef`, its scale `attnAbs` and the
  tolerances `tolO`, `tolF`.
* The assumptions: `Assumptions`.
-/

namespace Kernels.PagedAttention

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
      -- line 68: the column sum of the products (`tl.sum0c`), and `tl.mul_add`
      let acc ← tl.mul_add acc (bc alpha) (← tl.sum0c (← tl.mul (p.col HEAD_DIM) v))
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
  /-- At least one sequence and one head (a grid with a zero extent launches no program). -/
  seqs_pos : 0 < d.num_seqs
  heads_pos : 0 < d.num_heads
  /-- `num_heads % num_kv_heads == 0` (line 80), `num_kv_heads ≥ 1`. -/
  kv_pos : 0 < d.num_kv_heads
  kv_dvd : d.num_kv_heads ∣ d.num_heads
  /-- The grid `(num_seqs, num_heads, 1)` within CUDA's limits, fewer than `2^31` programs. -/
  heads_lt : d.num_heads < 2 ^ 16
  grid_lt : d.num_seqs * d.num_heads < 2 ^ 31
  /-- `HEAD_DIM` and `PAGE_SIZE` are powers of two (`tl.arange` rejects anything else on the host),
  at most `256`; `PAGE_SIZE` is at least `2`. At `PAGE_SIZE = 1` the reduction of line 68 has one
  lane, so its value is the product `p·v` itself and line 68 adds two products (`tl.addc`), which
  the model's `tl.mul_add` does not cover; the test's pages are `16` and `32` tokens. -/
  head_dim : ∃ a, a ≤ 8 ∧ d.HEAD_DIM = 2 ^ a
  page_size : ∃ b, 1 ≤ b ∧ b ≤ 8 ∧ d.PAGE_SIZE = 2 ^ b
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
`[0, max_pages·PAGE_SIZE]`, and each block-table entry `block_tables[s, j]` the kernel reads
(`j < cdiv(n, PAGE_SIZE)`, that is `j·PAGE_SIZE < n`) is a page number below `num_pages`. Pages
need not be distinct, and entries the kernel does not read may hold anything. -/
def SeqOk (d : Inst) (m : Mem) (s : Nat) : Prop :=
  ∃ n : I32, m.read .i32 (d.seq_lens.ptr (s : Int)).addr = some n ∧ 0 ≤ n.toInt ∧
    n.toInt ≤ ((d.max_pages * d.PAGE_SIZE : Nat) : Int) ∧
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

/-! ### The tolerances

`D`, `P` are `HEAD_DIM` and `PAGE_SIZE`, `T` the number of pages the sequence spans
(`cdiv(seq_len, PAGE_SIZE)`), and `X` a bound on the sequence's scaled score magnitudes
(`scale · scoreAbs ≤ X` on its tokens). The constants are the library's: one rounding of
`float32` (`2^-24` relative, `2^-150` absolute) and of `float16` (`2^-11`, `2^-25`), `Num.ExpF32`
(`2^-22 + 2^-23·|x|` relative, `2^-140` absolute) and `Num.DivFull` (`2^-22`, `2^-148`). -/

/-- One rounding of `float32`, relative: `2^-24`. -/
def u32 : ℚ := 1 / 2 ^ 24

/-- The relative error of a score's row sum (`D` rounded products added along a tree of
fusable additions, `3·2^-24` each). -/
def rhoD (D : Nat) : ℚ := (1 + u32) * (1 + 3 * u32) ^ (D - 1) - 1

/-- The error of each exponent `s − m_new` (line 65) against the exact `scale·score − m_new`. -/
def expErr (D : Nat) (X : ℚ) : ℚ := (rhoD D + 8 * u32) * X + 1 / 2 ^ 50

/-- `Num.ExpF32`'s relative error on the exponents, which lie in `[-(2X + 1), 2X + 1]`. -/
def relExp (X : ℚ) : ℚ := 1 / 2 ^ 22 + (2 * X + 1) / 2 ^ 23

/-- The relative error of each `p` (line 65) against its exact weight. -/
def psiP (D : Nat) (X : ℚ) : ℚ := (1 + expErr D X + expErr D X ^ 2) * (1 + relExp X) - 1

/-- The relative error of a page's `tl.sum(p)` (line 66): `P − 1` IEEE additions. -/
def psiL (D P : Nat) (X : ℚ) : ℚ := (1 + u32) ^ (P - 1) * (1 + psiP D X) - 1

/-- The relative error of a page's `tl.sum(p[:, None] * v, axis=0)` (line 68). -/
def psiA (D P : Nat) (X : ℚ) : ℚ := (1 + 3 * u32) ^ (P - 1) * (1 + u32) * (1 + psiP D X) - 1

/-- The factor by which a page can scale the earlier pages' error: `tl.mul_add`'s `3·2^-24`,
`alpha`'s `2^-22` and its input's `2^-140`. -/
def step : ℚ := (1 + 3 * u32) * (1 + 1 / 2 ^ 140) * (1 + 1 / 2 ^ 22)

/-- The growth of `alpha`'s input-dependent error over the loop: `exp(2^-22·(m + X + E))` with
`|m| ≤ X + E`, at most `1 + 4·2^-22·(X + E)`. -/
def drift (D : Nat) (X : ℚ) : ℚ := 1 + 4 / 2 ^ 22 * (X + expErr D X)

/-- The relative bound on `l_i` after `T` pages. -/
def fL (D P T : Nat) (X : ℚ) : ℚ := (1 + psiL D P X) * drift D X * step ^ T

/-- The relative bound on the accumulator after `T` pages. -/
def fA (D P T : Nat) (X : ℚ) : ℚ := (1 + psiA D P X) * drift D X * step ^ T

/-- The absolute error of `l_i` per page. -/
def absL : ℚ := 1 / 2 ^ 100

/-- The absolute error of the accumulator per page. -/
def absA : ℚ := 1 / 2 ^ 90

/-- The lower bound `1 − E` on the sum of the kernel's weights against its maximum. -/
def wlow (D : Nat) (X : ℚ) : ℚ := 1 - expErr D X

/-- `l_i`'s relative error after the loop, the absolute part folded in. -/
def betaL (D P T : Nat) (X : ℚ) : ℚ := (fL D P T X - 1) + fL D P T X * (T * absL) / wlow D X

/-- The denominator `1 − betaL`. -/
def denQ (D P T : Nat) (X : ℚ) : ℚ := 1 - betaL D P T X

/-- The relative error of `acc / l_i` against the exact weighted mean. -/
def relQ (D P T : Nat) (X : ℚ) : ℚ := (fA D P T X - 1 + betaL D P T X) / denQ D P T X

/-- The absolute error of `acc / l_i`. -/
def absQ (D P T : Nat) (X : ℚ) : ℚ := fA D P T X * (T * absA) / (wlow D X * denQ D P T X)

/-- The last two roundings (`/` and `.to(float16)`), relative. -/
def lastRel : ℚ := 1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22

/-- **The output's tolerance relative to `attnAbs`**, the weighted mean of `|v|`. -/
def tolO (D P T : Nat) (X : ℚ) : ℚ := lastRel * (1 + relQ D P T X) + relQ D P T X

/-- **The output's absolute tolerance**: mainly the output's `float16` rounding near zero
(`2^-25`). -/
def tolF (D P T : Nat) (X : ℚ) : ℚ :=
  absQ D P T X * (1 + lastRel) + 1 / 2 ^ 11 * (1 / 2 ^ 148) + 1 / 2 ^ 148 + 1 / 2 ^ 25

/-- The elements of `out` the launch writes. -/
def outElems (d : Inst) : Set Nat :=
  {x | ∃ s h e, s < d.num_seqs ∧ h < d.num_heads ∧ e < d.HEAD_DIM ∧ x = outIdx d s h e}

/-- What the launch leaves:
* every element `(s, h, e)` of `out` holds a `float16` value `c`; when the sequence's values are
  covered (`SeqVals`), `c` is finite and, for every bound `X ≤ 2^10` on the sequence's scaled score
  magnitudes, within `tolO·attnAbs + tolF` of the exact attention `attnRef` over the sequence's
  tokens, read through `block_tables` in `init`;
* every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ s h e, s < d.num_seqs → h < d.num_heads → e < d.HEAD_DIM →
    ∃ c : F16, fin.read .f16 (d.out.ptr (outIdx d s h e : Int)).addr = some c ∧
      (SeqVals d init s h → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
        (∀ t, t < seqLen d init s → sigma d * scoreAbs d init s h t ≤ X) →
        |c.toReal - attnRef d init s h e| ≤
          (tolO d.HEAD_DIM d.PAGE_SIZE (nblocks d init s) X : ℝ) * attnAbs d init s h e +
            (tolF d.HEAD_DIM d.PAGE_SIZE (nblocks d init s) X : ℝ))) ∧
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

end Kernels.PagedAttention
