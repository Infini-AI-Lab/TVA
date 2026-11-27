import TVA

/-!
# `topk_log_softmax_f32`: log-softmax at selected token ids, vLLM, float32

The kernel is `_topk_log_softmax_kernel` of `source/logprob.py` (lines 30–72; it reads the
module-level names `triton` and `tl`, bound by the imports of lines 23–24). Its host
`compute_token_logprobs` (lines 75–99) launches it with one program per row of a `float32` logits
matrix `[batch_size, vocab_size]` and an `int64` token-id matrix `[batch_size, num_logprobs]`, and
`BLOCK_SIZE = 1024`, `TOPK_BLOCK_SIZE = min(next_power_of_2(num_logprobs), 1024)`. Program `r`
takes the max of row `r` of the logits in blocks of `1024`, then the sum of `exp(logit − max)` in
blocks of `1024`, then `lse = log(sum)`, and stores `logits[r, id] − max − lse` for each id of row
`r` of the token ids, in blocks of `TOPK_BLOCK_SIZE`.

What is claimed: every covered launch ends normally; every output element `out[r, j]`,
`r < batch_size`, `j < num_logprobs`, holds a `float32` value; for every logits row whose values
are finite or `-∞`, with one finite and the finite ones within `2^127` of each other (`RowOk`),
there is one shift `sh` for the whole row, `|sh| ≤ lseTol = 2^-34·vocab_size + 2^-16`, such that
every output of the row is `-∞` exactly when its token's logit is `-∞`, and otherwise finite and
within `elemTol·|L| + floorTol` (`elemTol = 2^-22`, `floorTol = 2^-37`) of `L − sh`, where
`L = x_id − log Σ_k e^(x_k)` is the exact log-softmax in the reals; every other byte is unchanged.

The conditions: `0 < vocab_size ≤ 2^19`; `TOPK_BLOCK_SIZE` a power of two at most `1024`;
`num_logprobs + TOPK_BLOCK_SIZE ≤ 2^31`; fewer than `2^31` rows; strides below `2^31`; the buffers
hold every element the launch touches; the output shares no byte with the inputs; every token id
the launch reads is in `[0, vocab_size)`. Any strides, `batch_size = 0`, `num_logprobs = 0` and
every `float32` logit are covered.

The tolerances rest on the device facts for `tl.exp` and `tl.log` on `float32` and on the depth
of a `tl.sum`'s tree (`Device`).
-/

namespace Claims.TopkLogSoftmax

open TVA

/-! ## The model -/

/-- `float("-inf")`, the `float32` constant Triton builds for the Python double `-∞` (`litF32` of
its pattern; `TVA.Triton.Promote`). -/
def negInfC : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)

/-- `0.0`, the `float32` constant of the Python double `+0` (as `se = 0.0`, `other=0.0` and the
value of `tl.where`). -/
def zeroC : F32 := FloatV.litF32 (FloatV.ofPat 0)

set_option linter.unusedVariables false in
/-- `_topk_log_softmax_kernel` (source lines 30–72). `output_ptr` and `logits_ptr` are `*fp32`,
`topk_ids_ptr` is `*int64` (the host passes `token_ids.to(torch.int64)`). `logits_stride`,
`topk_ids_stride`, `topk` and `vocab_size` are `int32`: each is an integer argument in
`[0, 2^31)` (`Inst.ok`), a runtime `int32` except when it equals `1`, where Triton passes the
compile-time constant `1`; each use converts it first (`req_idx * s` builds it in `int64`, the
product's dtype, the same value as the sign-extended `int32`; the bound of `range` and the
comparison `k_offset < topk` make it an `int32` tensor), so the model is the same. `BLOCK_SIZE`
and `TOPK_BLOCK_SIZE` are the `tl.constexpr` parameters, Lean parameters here. The three loops
are `range` loops over `int32` (their bounds are `int32` or Python integers, which Triton makes
`int32` tensors); every name a body assigns, but `max_val` and `se`, is first defined in that
body and not used after it. The reductions are over rank-1 blocks. -/
def topk_log_softmax_kernel (output_ptr logits_ptr : Ptr .f32) (logits_stride : I32)
    (topk_ids_ptr : Ptr .i64) (topk_ids_stride topk vocab_size : I32)
    (BLOCK_SIZE TOPK_BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 42: `tl.program_id(0).to(tl.int64)`, sign extension of the `int32` program id
    let req_idx := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 43: the `int32` stride sign-extended to `int64`, the wrapping `int64` product, then the
    -- scalar pointer moved by its value
    let row_ptr := logits_ptr + req_idx * (← tl.convert .i32 .i64 (Tensor.scalar logits_stride)).item
    -- line 45: `max_val = float("-inf")`, a `float32` scalar (the number's own dtype); lines
    -- 46–49: the loop carries it
    let max_val ← tl.forRange (IntV.ofInt 0 : I32) vocab_size (IntV.ofNat BLOCK_SIZE : I32) negInfC
      fun i max_val => do
        -- line 47: the `int32` scalar plus the `int32` lanes
        let block := i + tl.arange 0 BLOCK_SIZE
        -- line 48: the `int32` lanes against the `int32` scalar `vocab_size`, signed;
        -- `other=float("-inf")` is the `float32` `-∞`, already the pointer's dtype
        let logits ← tl.load (row_ptr + block) (some (block.ltS vocab_size)) (some (bc negInfC))
        -- line 49: `tl.maximum` of the block and the broadcast scalar (`MaxRel`), then `tl.max`
        -- of the `float32` block (no widening)
        tl.maxAll true (← tl.maximum logits (bc max_val))
    -- line 50: `max_val.to(tl.float32)` is already `float32`: `max_val` itself
    -- line 52: `se = 0.0`, a `float32` scalar; lines 53–60: the loop carries it
    let se ← tl.forRange (IntV.ofInt 0 : I32) vocab_size (IntV.ofNat BLOCK_SIZE : I32) zeroC
      fun i se => do
        -- line 54
        let block := i + tl.arange 0 BLOCK_SIZE
        -- line 55: `other=0.0` is the `float32` `+0`, already the pointer's dtype
        let logits ← tl.load (row_ptr + block) (some (block.ltS vocab_size)) (some (bc zeroC))
        -- line 57: `logits.to(tl.float32)` is already `float32`
        -- line 58: the IEEE `-` against the broadcast scalar (no product feeds it), then the
        -- platform's `tl.exp`
        let e ← tl.exp (← tl.sub logits (bc max_val))
        -- line 59: the Python `0.0` is built in `float32`, the other value's dtype
        let e := tl.where (block.ltS vocab_size) e (bc zeroC)
        -- line 60: `se = se + tl.sum(e)`: IEEE additions along the sum's tree (the elements are
        -- `tl.exp` results, not products), then the IEEE `+` of two `float32` scalars
        return (← tl.add (bc se) (bc (← tl.sum e))).item
    -- line 61: the platform's `tl.log` of the `float32` scalar
    let lse := (← tl.log (bc se)).item
    -- lines 63–72: no carried name
    tl.forRange (IntV.ofInt 0 : I32) topk (IntV.ofNat TOPK_BLOCK_SIZE : I32) () fun j _ => do
      -- line 64
      let k_offset := j + tl.arange 0 TOPK_BLOCK_SIZE
      -- line 65: signed `int32` comparison
      let k_mask := k_offset.ltS topk
      -- lines 66–68: `(topk_ids_ptr + req_idx * topk_ids_stride) + k_offset`, left to right: the
      -- `int64` product (the `int32` stride sign-extended), then the `int32` lanes; `other=0`
      -- is the Python `0`, an `int32` tensor converted to the pointer's dtype, the `int64` `0`
      let topk_ids ← tl.load
        (topk_ids_ptr + req_idx * (← tl.convert .i32 .i64 (Tensor.scalar topk_ids_stride)).item
          + k_offset) (some k_mask) (some (bc (IntV.ofInt 0 : I64)))
      -- line 69: the scalar pointer moved by each `int64` id; masked lanes unspecified (no
      -- `other=`)
      let logits ← tl.load (row_ptr + topk_ids) (some k_mask)
      -- line 70: already `float32`; line 71: `(logits - max_val) - lse`, two IEEE `-` against
      -- broadcast scalars (no product feeds them)
      let o ← tl.sub (← tl.sub logits (bc max_val)) (bc lse)
      -- line 72: `(output_ptr + req_idx * topk) + k_offset`: the `int64` product (the `int32`
      -- `topk` sign-extended), then the `int32` lanes; `o` is already `float32`
      tl.store (output_ptr + req_idx * (← tl.convert .i32 .i64 (Tensor.scalar topk)).item
        + k_offset) o (some k_mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `_topk_log_softmax_kernel` through
`compute_token_logprobs`. -/
structure Inst where
  /-- `batch_size` (the host passes `logits.shape[0]`; the grid `(batch_size,)`). -/
  batch : Nat
  /-- `vocab_size` (the host passes `logits.shape[1]`). -/
  vocab : Nat
  /-- `topk` (the host passes `num_logprobs = token_ids.shape[1]`). -/
  topk : Nat
  /-- `logits_stride` (the host passes `logits.stride(0)`), in elements. -/
  logits_stride : Nat
  /-- `topk_ids_stride` (the host passes `token_ids.stride(0)`), in elements. -/
  ids_stride : Nat
  /-- `TOPK_BLOCK_SIZE` (the host passes `min(next_power_of_2(num_logprobs), 1024)`). -/
  TOPK_BLOCK_SIZE : Nat
  /-- `logits_ptr`: the `float32` logits (from the first element of row `0`, unit column
  stride). -/
  logits : Buf .f32
  /-- `topk_ids_ptr`: the `int64` token ids (unit column stride). -/
  ids : Buf .i64
  /-- `output_ptr`: the `float32` output `[batch_size, num_logprobs]`, contiguous. -/
  out : Buf .f32

/-- Logit `k` of row `r`: `logits[r, k]` at `r * logits_stride + k`. -/
def inIdx (d : Inst) (r k : Nat) : Nat := r * d.logits_stride + k

/-- Token id `j` of row `r`: `token_ids[r, j]` at `r * topk_ids_stride + j`. -/
def idIdx (d : Inst) (r j : Nat) : Nat := r * d.ids_stride + j

/-- Output `j` of row `r`: `out[r, j]` at `r * topk + j`. -/
def outIdx (d : Inst) (r j : Nat) : Nat := r * d.topk + j

/-- The logits the launch reads: every row's `vocab_size` values. -/
def inSet (d : Inst) : Set Nat := {e | ∃ r k, r < d.batch ∧ k < d.vocab ∧ e = inIdx d r k}

/-- The token ids the launch reads. -/
def idSet (d : Inst) : Set Nat := {e | ∃ r j, r < d.batch ∧ j < d.topk ∧ e = idIdx d r j}

/-- The output elements the launch writes. -/
def outSet (d : Inst) : Set Nat := {e | ∃ r j, r < d.batch ∧ j < d.topk ∧ e = outIdx d r j}

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- A row has at least one logit. -/
  vocab_pos : 0 < d.vocab
  /-- `vocab_size ≤ 2^19 = 524288`: the vocabularies of real models (the largest in use, Gemma
  3's `262208` and Gemma 3n's `262400`, are above `2^18`); the tolerance `lseTol` grows with it
  (README, Exclusions). -/
  vocab_le : d.vocab ≤ 2 ^ 19
  /-- `TOPK_BLOCK_SIZE` is a power of two of at most `1024` (the host's is). -/
  tb : ∃ e, e ≤ 10 ∧ d.TOPK_BLOCK_SIZE = 2 ^ e
  /-- The `int32` lanes `j + l` and the `int32` loop `range(0, topk, TOPK_BLOCK_SIZE)` do not
  overflow: `topk` is an `int32` and `topk + TOPK_BLOCK_SIZE ≤ 2^31` (a limit of the kernel). -/
  topk_lt : d.topk + d.TOPK_BLOCK_SIZE ≤ 2 ^ 31
  /-- The grid has fewer than `2^31` programs. -/
  batch_lt : d.batch < 2 ^ 31
  /-- `logits_stride` is an `int32`. -/
  stride_lt : d.logits_stride < 2 ^ 31
  /-- `topk_ids_stride` is an `int32`. -/
  ids_stride_lt : d.ids_stride < 2 ^ 31
  /-- `logits` holds every row the launch reads. -/
  logits_len : ∀ r, r < d.batch → r * d.logits_stride + d.vocab ≤ d.logits.len
  /-- `token_ids` holds every id the launch reads. -/
  ids_len : ∀ r j, r < d.batch → j < d.topk → idIdx d r j < d.ids.len
  /-- The output holds `batch_size` rows of `topk`. -/
  out_len : d.batch * d.topk ≤ d.out.len
  /-- The output elements share no byte with the logits the launch reads (the host's output is
  fresh). -/
  out_logits : Disjoint (d.out.elems (outSet d)) (d.logits.elems (inSet d))
  /-- The output elements share no byte with the token ids the launch reads. -/
  out_ids : Disjoint (d.out.elems (outSet d)) (d.ids.elems (idSet d))

/-- The model on an instance: the kernel's arguments are the buffers' first elements, the four
integers as `int32`, `BLOCK_SIZE = 1024` (the host's constant) and `TOPK_BLOCK_SIZE`. -/
def model (d : Inst) : Kernel :=
  topk_log_softmax_kernel (d.out.ptr 0) (d.logits.ptr 0) (IntV.ofNat d.logits_stride)
    (d.ids.ptr 0) (IntV.ofNat d.ids_stride) (IntV.ofNat d.topk) (IntV.ofNat d.vocab) 1024
    d.TOPK_BLOCK_SIZE

/-- The host's launch (source lines 88–98): the grid `(batch_size,)`. The source's block shapes
are `[BLOCK_SIZE]` (`block`, the two passes' `logits`, `e`) and `[TOPK_BLOCK_SIZE]`
(`k_offset`, `k_mask`, `topk_ids`, the gathered `logits`, `o`). `enable_fp_fusion` is Triton's
default (`true`); `grid1` keeps the defaults `num_ctas = 1`, `launch_pdl = False` and
`debug = false`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.batch [[1024], [d.TOPK_BLOCK_SIZE]] (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and every token id the launch reads is a column of the row, in `[0, vocab_size)` (the
host's ids are sampled tokens, top-k indices or ids checked against the vocabulary; SETUP.md).
Every `float32` logit is covered, NaN and infinities included: `Post` constrains a row that is not
`RowOk` less. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.logits.Valid init ∧ d.ids.Valid init ∧ d.out.Valid init ∧
    ∀ r j, r < d.batch → j < d.topk → ∀ v : I64,
      init.read .i64 (d.ids.ptr (idIdx d r j)).addr = some v → 0 ≤ v.toInt ∧ v.toInt < d.vocab

/-! ## The specification -/

/-- A row `xs 0, …, xs (n-1)` whose log-softmax is defined and does not overflow: every value is a
finite `float32` or `-∞`, one value is finite, and any two finite values are within `2^127` of
each other (so `x − max` stays in `float32`'s range). -/
def RowOk (xs : Nat → F32) (n : Nat) : Prop :=
  (∀ k, k < n → (xs k).isFinite = true ∨ xs k = FloatV.negInf) ∧
    (∃ k, k < n ∧ (xs k).isFinite = true) ∧
    ∀ k k', k < n → k' < n → (xs k).isFinite = true → (xs k').isFinite = true →
      |(xs k).toReal - (xs k').toReal| ≤ 2 ^ 127

/-- `e^a` of a logit `a`, with `e^(-∞) = 0`. -/
noncomputable def expIn (a : F32) : ℝ := if a = FloatV.negInf then 0 else Real.exp a.toReal

/-- The log-softmax of the row `xs 0, …, xs (n-1)` at column `j`, exactly, in the reals:
`x_j − log Σ_{k<n} e^(x_k)` (for a finite `x_j`). -/
noncomputable def logSoftmax (xs : Nat → F32) (n j : Nat) : ℝ :=
  (xs j).toReal - Real.log (∑ k ∈ Finset.range n, expIn (xs k))

/-- The tolerance of a row's common shift: `2^-34 · vocab_size + 2^-16`. It covers the error of
`lse`: the sum's roundings, along the `⌈vocab_size/1024⌉` additions of the loop and the tree of
depth at most `⌈log₂ 1024⌉ = 10` of each `tl.sum` (`Device.SumDepth`), the lanes' `tl.exp`
errors summed over the row, and `tl.log`'s error (README, Tolerances). -/
def lseTol (n : Nat) : ℚ := 2 ^ (-34 : ℤ) * n + 2 ^ (-16 : ℤ)

/-- The relative tolerance of each output, per unit of `|L|`: `2^-22` (the roundings of
`x − max` and of `(x − max) − lse`; README, Tolerances). -/
def elemTol : ℚ := 2 ^ (-22 : ℤ)

/-- The absolute floor of each output: `2^-37` (the last rounding's share of the shift, and the
subtractions' absolute errors). -/
def floorTol : ℚ := 2 ^ (-37 : ℤ)

/-- What the launch leaves: for each row `r < batch_size`, with `xs` the row's logits and `tid` the
row's token ids as read in the initial memory, every output element `out[r, j]`, `j < topk`, holds
a `float32` value; when the row is `RowOk`, there is one shift `sh` for the whole row, within
`lseTol vocab_size` of `0`, such that each output element is `-∞` when its token's logit is `-∞`,
and otherwise a finite `float32` within `elemTol · |L| + floorTol` of `L − sh`, where
`L = logSoftmax xs vocab_size id` is the exact log-softmax at the token's id; every byte outside
the output elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r, r < d.batch → ∃ (xs : Nat → F32) (tid : Nat → I64),
      (∀ k, k < d.vocab → init.read .f32 (d.logits.ptr (inIdx d r k)).addr = some (xs k)) ∧
      (∀ j, j < d.topk → init.read .i64 (d.ids.ptr (idIdx d r j)).addr = some (tid j)) ∧
      (∀ j, j < d.topk → ∃ c : F32, fin.read .f32 (d.out.ptr (outIdx d r j)).addr = some c) ∧
      (RowOk xs d.vocab → ∃ sh : ℝ, |sh| ≤ (lseTol d.vocab : ℝ) ∧
        ∀ j, j < d.topk → ∃ c : F32, fin.read .f32 (d.out.ptr (outIdx d r j)).addr = some c ∧
          (xs (tid j).toInt.toNat = FloatV.negInf → c = FloatV.negInf) ∧
          (xs (tid j).toInt.toNat ≠ FloatV.negInf → c.isFinite = true ∧
            |c.toReal - (logSoftmax xs d.vocab (tid j).toInt.toNat - sh)| ≤
              (elemTol : ℝ) * |logSoftmax xs d.vocab (tid j).toInt.toNat| + (floorTol : ℝ)))) ∧
    ∀ z, z ∉ d.out.elems (outSet d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_topk_log_softmax_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- Two buffers whose byte ranges are ordered do not overlap. -/
theorem bytes_disjoint_of_le {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : b.base + b.len * d.bytes ≤ c.base) : Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hcx => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hcx
  have hb := d.bytes_pos
  linarith

/-- The host's layout: a `B × V` logits matrix of row stride `S` at `2^34` (`S = V` contiguous;
`S > V` a column slice of a wider matrix), a contiguous `B × K` id matrix at `2^37` and the fresh
`B × K` output at `2^38`, with `TOPK_BLOCK_SIZE = 2^e`. -/
def mk (B V S K e : Nat) : Inst where
  batch := B
  vocab := V
  topk := K
  logits_stride := S
  ids_stride := K
  TOPK_BLOCK_SIZE := 2 ^ e
  logits := ⟨2 ^ 34, B * S⟩
  ids := ⟨2 ^ 37, B * K⟩
  out := ⟨2 ^ 38, B * K⟩

theorem idx_lt {R C r j : Nat} (hr : r < R) (hj : j < C) : r * C + j < R * C := by
  have : (r + 1) * C ≤ R * C := Nat.mul_le_mul_right _ hr
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

theorem row_le {R C V r : Nat} (hr : r < R) (hV : V ≤ C) : r * C + V ≤ R * C := by
  have : (r + 1) * C ≤ R * C := Nat.mul_le_mul_right _ hr
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- Every such launch is covered when `0 < V ≤ 2^19`, `V ≤ S < 2^31`, `e ≤ 10`,
`K + 2^e ≤ 2^31`, `B < 2^31`, and the matrices fit their gaps (`4·B·S ≤ 2^36`,
`8·B·K ≤ 2^37`). -/
theorem mk_ok {B V S K e : Nat} (hV : 0 < V) (hV19 : V ≤ 2 ^ 19) (hVS : V ≤ S) (hS : S < 2 ^ 31)
    (he : e ≤ 10) (hK : K + 2 ^ e ≤ 2 ^ 31) (hB : B < 2 ^ 31) (hBS : B * S * 4 ≤ 2 ^ 36)
    (hBK : B * K * 8 ≤ 2 ^ 37) :
    (mk B V S K e).ok := by
  have hK31 : K < 2 ^ 31 := by
    have : 0 < 2 ^ e := Nat.two_pow_pos e
    omega
  refine ⟨hV, hV19, ⟨e, he, rfl⟩, hK, hB, hS, hK31, fun r hr => row_le hr hVS,
    fun r j hr hj => idx_lt hr hj, le_refl _, ?_, ?_⟩
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · refine (bytes_disjoint_of_le ?_).symm
      simp only [mk, DType.bytes]
      have : ((B * S : Nat) : Int) * 4 ≤ 2 ^ 36 := by exact_mod_cast hBS
      push_cast at this ⊢
      linarith
    · rintro k ⟨r, j, hr, hj, rfl⟩; exact idx_lt hr hj
    · rintro k ⟨r, j, hr, hj, rfl⟩
      simp only [inIdx, mk] at hr hj ⊢
      have := row_le (C := S) hr hVS
      omega
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · refine (bytes_disjoint_of_le ?_).symm
      simp only [mk, DType.bytes]
      have : ((B * K : Nat) : Int) * 8 ≤ 2 ^ 37 := by exact_mod_cast hBK
      push_cast at this ⊢
      linarith
    · rintro k ⟨r, j, hr, hj, rfl⟩; exact idx_lt hr hj
    · rintro k ⟨r, j, hr, hj, rfl⟩; exact idx_lt hr hj

/-- The memory every byte of which is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

theorem readBytes_zeroMem : ∀ (n : Nat) (a : Int),
    zeroMem.readBytes a n = some (List.replicate n 0) := by
  intro n
  induction n with
  | zero => intro a; rfl
  | succ n ih =>
    intro a
    show (match zeroMem a, zeroMem.readBytes (a + 1) n with
      | some b, some bs => some (b :: bs)
      | _, _ => none) = _
    rw [ih (a + 1)]
    rfl

theorem zeroMem_read {d : DType} {a : Int} (h : Mem.Aligned d a) :
    zeroMem.read d a = some (DType.ofBits d 0) := by
  simp only [Mem.read, h, ↓reduceIte, readBytes_zeroMem, Option.map_some]
  have hlv : ∀ n : Nat, Bytes.leValue (List.replicate n 0) = 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [List.replicate, Bytes.leValue, ih]; norm_num
  rw [hlv]

/-- A covered instance and memory: one row of `1024` logits, one id (`0`), on the all-zero
memory. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1 1024 1024 1 0, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ?_⟩
  rotate_left 9
  · intro r j hr hj v hv
    have hr0 : r = 0 := by simp [mk] at hr; omega
    have hj0 : j = 0 := by simp [mk] at hj; omega
    subst hr0 hj0
    have hal : Mem.Aligned .i64 (2 ^ 37) := by simp only [Mem.Aligned, DType.bytes]; norm_num
    simp only [idIdx, mk, Buf.ptr, DType.bytes] at hv
    norm_num at hv
    rw [zeroMem_read (by simpa using hal), Option.some.injEq] at hv
    subst hv
    have h0 : (DType.ofBits DType.i64 0 : I64).toInt = 0 := by decide
    rw [h0]
    simp [mk]
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## Coverage of the host's launches (`SETUP.md`)

`compute_token_logprobs(logits, token_ids)` on a contiguous `float32` matrix `[B, V]` and a
contiguous `int64` id matrix `[B, K]`: `TOPK_BLOCK_SIZE = min(next_power_of_2(K), 1024)`. -/

/-- Sampling without logprobs (`compute_topk_scores`, `num_logprobs = 0`: the sampled token only),
Qwen's vocabulary `151936`, `1024` requests. -/
def setup_sample_qwen_k1 : Inst := mk 1024 151936 151936 1 0
theorem setup_sample_qwen_k1_ok : setup_sample_qwen_k1.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Sampling with `logprobs=20` (the sampled token and the top 20), Llama 3's vocabulary
`128256`, `256` requests. -/
def setup_sample_llama3_k21 : Inst := mk 256 128256 128256 21 5
theorem setup_sample_llama3_k21_ok : setup_sample_llama3_k21.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Sampling with `logprobs=5`, Gemma 3's vocabulary `262208`, `128` requests. -/
def setup_sample_gemma3_k6 : Inst := mk 128 262208 262208 6 3
theorem setup_sample_gemma3_k6_ok : setup_sample_gemma3_k6.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Custom `logprob_token_ids` (up to `128` per request) next to `logprobs=20`: `1 + 128` columns,
DeepSeek-V3's vocabulary `129280`, `64` requests. -/
def setup_custom_ids_k129 : Inst := mk 64 129280 129280 129 8
theorem setup_custom_ids_k129_ok : setup_custom_ids_k129.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Speculative decoding (the rejection sampler's logits, `1 + 4` rows per request, `1024`
requests) with `logprobs=1`, GPT-2's vocabulary `50257` (not a multiple of `1024`). -/
def setup_spec_gpt2_k2 : Inst := mk 5120 50257 50257 2 1
theorem setup_spec_gpt2_k2_ok : setup_spec_gpt2_k2.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Prompt logprobs over the whole vocabulary (`prompt_logprobs=-1`: `1 + vocab_size` columns), a
chunk of `1024` prompt positions, Llama 2's vocabulary `32000`. -/
def setup_prompt_full_vocab : Inst := mk 1024 32000 32000 32001 10
theorem setup_prompt_full_vocab_ok : setup_prompt_full_vocab.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Prompt-token scoring (`prompt_logprob_token_ids`, `compute_token_logprobs` on a chunk of
`1024` positions with `8` label ids each), Qwen's vocabulary `151936`. -/
def setup_prompt_label_ids_k8 : Inst := mk 1024 151936 151936 8 3
theorem setup_prompt_label_ids_k8_ok : setup_prompt_label_ids_k8.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Logits as a column slice of a wider matrix (`logits[:, :vocab_size]` of the sharded
sampler's padded logits): Qwen's vocabulary `151936` in rows of `152064`, `logprobs=20`, `256`
requests. -/
def setup_slice_qwen_k21 : Inst := mk 256 151936 152064 21 5
theorem setup_slice_qwen_k21_ok : setup_slice_qwen_k21.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- An empty batch: the grid `(0,)`. -/
def setup_empty : Inst := mk 0 151936 151936 21 5
theorem setup_empty_ok : setup_empty.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end Claims.TopkLogSoftmax
