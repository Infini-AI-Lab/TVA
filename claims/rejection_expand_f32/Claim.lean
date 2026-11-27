import TVA

/-!
# `rejection_expand_f32`: per-request values expanded to tokens, vLLM

The kernel is `expand_kernel` of `source/rejection_expand.py` (lines 107-128). The host
`expand_batch_to_tokens` (lines 66-104, through `_expand`, lines 131-148) launches it with one
program per request: program `r` reads the request's bounds `start_r` (`0`, or `cu[r-1]`) and
`end_r = cu[r]` in the `int32` running sums `cu_num_tokens`, reads the request's value `x[r]`,
replaces it by `replace_to` when it equals `replace_from`, and stores it at the output positions
`start_r … end_r - 1` with one store of `MAX_NUM_TOKENS = 128` lanes masked by
`offset < end_r - start_r`. vLLM calls it on `float32` (`temperature`, with `replace_from = 0`,
`replace_to = 1`; `top_p`, with `0` and `0`) and `int32` (`top_k`, with `0` and `0`) values
(`SETUP.md`).

What is claimed: every covered launch ends normally; for every request `r` and every position
`j` with `start_r ≤ j < end_r`, output element `j` holds `x[r]`, or `replace_to` where `x[r] ==
replace_from` (`expandVal`: on `float32` the ordered float comparison with the `float32` nearest
to `replace_from`, so `-0.0` and `+0.0` both match `0` and a NaN matches nothing, and the
`float32` nearest to `replace_to`); every byte outside the output's first `cu[B-1]` elements is
unchanged.

The conditions: the values are `float32` or `int32`; `1 ≤ B ≤ 8192` requests; `replace_from` and
`replace_to` are `int32` arguments; `cu` and `x` hold one entry per request; the output shares no
byte with an input; `cu` holds the inclusive running sums of the per-request token counts, each
count in `[0, 128]` (vLLM's `MAX_SPEC_LEN`, which the host's caller asserts), the total within the
output. Every value of `x` is covered (NaNs, infinities and signed zeros included).

The specification is exact: the comparison and selection are exact, and the conversions of the two
`int32` arguments to `float32` round once to nearest; no device fact.
-/

namespace Claims.RejectionExpand

open TVA

/-! ## Helpers of the transcription

`x` may be `float32` or `int32` (the host's callers pass both). The model is one function of the
dtype, with the dtype-dependent line written explicitly; `Inst.ok` restricts the dtype to the two
covered ones. -/

/-- Line 126, `src_val = tl.where(src_val == replace_from, replace_to, src_val)`, for the dtype `d`
of `input_ptr`, on the loaded scalar `src_val` and the `int32` scalars `replace_from`,
`replace_to`.

- `float32`: `src_val == replace_from` computes in `float32` (`BinOp.typing? .eq` of a `float32`
  and an `int32` tensor), so `replace_from` is converted first (`sitofp`, `tl.itof`); the `==` is
  the ordered float comparison (`FloatV.eq`). `tl.where` types its values by `whereTyping?`: an
  `int32` and a `float32` give `float32`, so `replace_to` is converted (`tl.itof`); the selection
  is exact.
- `int32`: the `==` of two `int32` scalars, and `tl.where` of two `int32` scalars, no conversion.

Every other dtype is outside the claim (`Inst.ok`); the model leaves the value as it is there. -/
def replaceVal : (d : DType) → d.Val → I32 → I32 → Prog d.Val
  | .f32, src_val, replace_from, replace_to => do
    let replace_from_f : F32 := (← tl.itof (f := .binary32) (Tensor.scalar replace_from)).item
    let cond := FloatV.eq src_val replace_from_f
    let replace_to_f : F32 := (← tl.itof (f := .binary32) (Tensor.scalar replace_to)).item
    pure (tl.where (Tensor.scalar cond) (Tensor.scalar replace_to_f) (Tensor.scalar src_val)).item
  | .i32, src_val, replace_from, replace_to =>
    pure (tl.where (Tensor.scalar (src_val.eq replace_from)) (Tensor.scalar replace_to)
      (Tensor.scalar src_val)).item
  | _, src_val, _, _ => pure src_val

/-! ## The model -/

/-- `expand_kernel` (source lines 107-128). `output_ptr` and `input_ptr` point at the output and at
`x`, both of dtype `d`; `cu_num_tokens_ptr` is `*i32`. `replace_from` and `replace_to` are runtime
`int32` arguments: the decorator's `do_not_specialize` keeps Triton from specialising them, so
they are `int32` whatever their value in `[-2^31, 2^31)`, `1` included. `MAX_NUM_TOKENS` is the
`tl.constexpr`, a Lean parameter. -/
def expand_kernel {d : DType} (output_ptr input_ptr : Ptr d) (cu_num_tokens_ptr : Ptr .i32)
    (replace_from replace_to : I32) (MAX_NUM_TOKENS : Nat) : Kernel where
  body ctx := do
    -- line 117: `req_idx = tl.program_id(0)`, an `int32`
    let req_idx := tl.program_id ctx 0
    -- lines 118-121: `if req_idx == 0:` is a runtime `if` (the Python `0` becomes an `int32`
    -- tensor of its own type, `int32 == int32`); each branch assigns `start_idx`, an `int32`
    -- scalar: `tl.zeros([], dtype=cu_num_tokens_ptr.dtype.element_ty)` is the `int32` `0`;
    -- `cu_num_tokens_ptr + req_idx - 1` is `(ptr + req_idx) - 1`, Python's left-to-right order:
    -- the pointer advanced by `req_idx`, then one element back (`Ptr.sub_int_of_signed`)
    let start_idx : I32 ←
      if req_idx.eq (0 : I32) then
        pure (tl.zeros [] .i32).item
      else
        tl.load1 (d := .i32) (cu_num_tokens_ptr + req_idx - (1 : I32))
    -- line 122
    let end_idx ← tl.load1 (d := .i32) (cu_num_tokens_ptr + req_idx)
    -- line 123: the wrapping `int32` difference
    let num_tokens := end_idx - start_idx
    -- line 125: one element of dtype `d`
    let src_val ← tl.load1 (input_ptr + req_idx)
    -- line 126 (`replaceVal`)
    let src_val ← replaceVal d src_val replace_from replace_to
    -- line 127: `int32` lanes `0 … MAX_NUM_TOKENS - 1`
    let offset := tl.arange 0 MAX_NUM_TOKENS
    -- line 128: `output_ptr + start_idx + offset` is `(ptr + start_idx) + offset`, a block of
    -- pointers; the scalar `src_val`, already of the pointer's dtype, is broadcast to the block
    -- (`bc`); the mask is the signed `int32` `offset < num_tokens`
    tl.store (output_ptr + start_idx + offset) (bc src_val) (some (offset.ltS num_tokens))

/-! ## The conditions -/

/-- What the host chooses: one launch of `expand_kernel` through `expand_batch_to_tokens`
(source lines 66-104 and 131-148). -/
structure Inst where
  /-- The dtype of `x` and of the output (`x.new_empty`). -/
  dt : DType
  /-- `batch_size = x.shape[0]`: the grid, one program per request. -/
  B : Nat
  /-- `expanded_x`, the output: a contiguous `[num_tokens]` tensor of dtype `dt`. -/
  out : Buf dt
  /-- `x`: a contiguous `[batch_size]` tensor of dtype `dt`. -/
  inp : Buf dt
  /-- `cu_num_tokens`: a contiguous `[batch_size]` `int32` tensor. -/
  cu : Buf .i32
  /-- `replace_from`. -/
  rf : Int
  /-- `replace_to`. -/
  rt : Int

/-- The `int32` value memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes are
absent). -/
def rd32 (b : Buf .i32) (m : Mem) (i : Int) : Int :=
  match m.read .i32 (b.ptr i).addr with | some v => v.toInt | none => 0

/-- Request `r`'s first position: `0` for the first request, else `cu[r-1]`. -/
def startOf (d : Inst) (m : Mem) (r : Nat) : Int :=
  if r = 0 then 0 else rd32 d.cu m ((r - 1 : Nat) : Int)

/-- Request `r`'s number of tokens `n_r`: `cu[r] - start`. -/
def nOf (d : Inst) (m : Mem) (r : Nat) : Int := rd32 d.cu m ((r : Int)) - startOf d m r

/-- The value the launch writes for a request whose value is `v`, of dtype `dt`, with
`replace_from = a` and `replace_to = b`: on `float32`, the `float32` nearest to `b` (ties to even)
when `v == a'` for the `float32` `a'` nearest to `a` (the ordered comparison: false on NaN, true for
both zeros against `+0`), else `v`; on `int32`, `b` (as an `int32`) when `v` is `a`, else `v`. -/
def expandVal : (dt : DType) → dt.Val → Int → Int → dt.Val
  | .f32, v, a, b =>
    if FloatV.eq v (FloatV.ofRat (a : Rat) : F32) then (FloatV.ofRat (b : Rat) : F32) else v
  | .i32, v, a, b => if v.toInt = a then (IntV.ofInt b : I32) else v
  | _, v, _, _ => v

/-- The instances covered: either dtype, `1 ≤ B ≤ 8192`, `int32` replacement arguments, one `cu`
and one `x` entry per request, and an output that shares no byte with an input. The inputs may
overlap one another (they are only read). -/
structure Inst.ok (d : Inst) : Prop where
  /-- `x` is `float32` (`temperature`, `top_p`) or `int32` (`top_k`). -/
  hdt : d.dt = .f32 ∨ d.dt = .i32
  /-- At least one request. -/
  hB : 1 ≤ d.B
  /-- `batch_size` is a batch size: at most `8192` (the grid then has fewer than `2^31`
  programs). -/
  hB_le : d.B ≤ 8192
  /-- `replace_from` is an `int32` argument (`[-2^31, 2^31)`; a Python `int` outside it would be
  passed as an `int64`). -/
  hrf : -2 ^ 31 ≤ d.rf ∧ d.rf < 2 ^ 31
  /-- `replace_to` is an `int32` argument. -/
  hrt : -2 ^ 31 ≤ d.rt ∧ d.rt < 2 ^ 31
  /-- `cu_num_tokens` has one entry per request. -/
  hcu : d.B ≤ d.cu.len
  /-- `x` has one entry per request. -/
  hinp : d.B ≤ d.inp.len
  /-- The output shares no byte with `cu_num_tokens`. -/
  hout_cu : Disjoint d.out.bytes d.cu.bytes
  /-- The output shares no byte with `x` (`x.new_empty` allocates it). -/
  hout_inp : Disjoint d.out.bytes d.inp.bytes

/-- The model on an instance: the kernel's arguments are the buffers' first elements, the two
replacement values as `int32`, and `MAX_NUM_TOKENS = MAX_SPEC_LEN = 128`. -/
def model (d : Inst) : Kernel :=
  expand_kernel (d.out.ptr 0) (d.inp.ptr 0) (d.cu.ptr 0) (IntV.ofInt d.rf) (IntV.ofInt d.rt) 128

/-- The host's launch (source lines 139-148): a 1-D grid of `batch_size` programs; the block
shape `[128]` (`tl.arange(0, MAX_NUM_TOKENS)`); `enable_fp_fusion` is Triton's default (`true`);
`num_warps = 4` and `num_stages = 3` change no result; `grid1` keeps the defaults `num_ctas = 1`,
`launch_pdl = False` and `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.B [[128]] (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); `cu` holds the inclusive running sums of the per-request token counts, each count in
`[0, 128]` (`MAX_SPEC_LEN`: vLLM's draft counts, `max_spec_len ≤ MAX_SPEC_LEN` asserted by
`RejectionSampler.forward`), with its last entry, the number of tokens, at most the output's
length. The values of `x` and the output's initial contents are arbitrary. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.out.Valid init ∧ d.inp.Valid init ∧ d.cu.Valid init ∧
    (∀ r, r < d.B → 0 ≤ nOf d init r ∧ nOf d init r ≤ 128) ∧
    rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.out.len

/-! ## The specification -/

/-- What the launch leaves: for each request `r < B` and each position `j` of the request
(`start_r ≤ j < cu[r]`), output element `j` holds `expandVal` of the request's initial value
`x[r]`; every byte outside the output's first `cu[B-1]` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r, r < d.B → ∀ j : Nat, startOf d init r ≤ (j : Int) → (j : Int) < rd32 d.cu init (r : Int) →
      ∃ v, init.read d.dt (d.inp.ptr (r : Int)).addr = some v ∧
        fin.read d.dt (d.out.ptr (j : Int)).addr = some (expandVal d.dt v d.rf d.rt)) ∧
    ∀ z, z ∉ d.out.elems {j | (j : Int) < rd32 d.cu init ((d.B - 1 : Nat) : Int)} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `expand_kernel` ends normally with `Post`. -/
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

/-- An instance of dtype `dt` with `B` requests, `N` tokens and the replacement `rf → rt`, with the
buffers laid out one after another from `2^30` with `2^20`-byte gaps (the output first). Each
buffer takes 8 bytes per element in the layout, so every base is a multiple of 8. -/
def mk (dt : DType) (B N : Nat) (rf rt : Int) : Inst :=
  let o : Int := 2 ^ 30
  let c : Int := o + N * 8 + 2 ^ 20
  let x : Int := c + B * 8 + 2 ^ 20
  { dt := dt, B := B, out := ⟨o, N⟩, inp := ⟨x, B⟩, cu := ⟨c, B⟩, rf := rf, rt := rt }

theorem mk_ok (dt : DType) (B N : Nat) (rf rt : Int) (hdt : dt = .f32 ∨ dt = .i32) (h1 : 1 ≤ B)
    (h2 : B ≤ 8192) (h3 : -2 ^ 31 ≤ rf ∧ rf < 2 ^ 31) (h4 : -2 ^ 31 ≤ rt ∧ rt < 2 ^ 31) :
    (mk dt B N rf rt).ok := by
  refine ⟨hdt, h1, h2, h3, h4, le_refl _, le_refl _, ?_, ?_⟩
  · refine bytes_disjoint_of_le ?_
    rcases hdt with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> nlinarith
  · refine bytes_disjoint_of_le ?_
    rcases hdt with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> nlinarith

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

theorem rd32_zeroMem (b : Buf .i32) (i : Int) : rd32 b zeroMem i = 0 := by
  rw [rd32]
  by_cases h : Mem.Aligned .i32 (b.ptr i).addr
  · rw [zeroMem_read h]; decide
  · simp only [Mem.read, h, ↓reduceIte]

/-- A covered instance and memory: the `float32` `temperature` call on 37 requests with no
tokens, every buffer `0` (every count `0`). -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk .f32 37 0 0 1, zeroMem,
    mk_ok _ _ _ _ _ (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ?_, ?_⟩
  rotate_left 9
  · intro r _
    simp only [nOf, startOf, rd32_zeroMem]
    split_ifs <;> norm_num
  · rw [rd32_zeroMem]; simp [mk]
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## The setup's launches

The host's test cases (source lines 160-206) and SETUP.md's serving sizes: dtype, `B` requests,
`N = Σ n_r` tokens (the output's length), `replace_from`, `replace_to`. Each random batch makes the
three calls of `apply_sampling_constraints`: `temperature` (`float32`, `0 → 1`), `top_k`
(`int32`, `0 → 0`) and `top_p` (`float32`, `0 → 0`). The random batches' `N` are those the test's
generator (`torch.Generator().manual_seed(0)`) draws. -/

/-- The docstring's example: `x = [a, b, c]`, `cu_num_tokens = [2, 5, 6]` (`float32`, `0 → 0`). -/
def setup_docstring : Inst := mk .f32 3 6 0 0
theorem setup_docstring_ok : setup_docstring.ok :=
  mk_ok _ _ _ _ _ (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `temperature` of a batch of `B` requests and `N` tokens. -/
def temperature (B N : Nat) : Inst := mk .f32 B N 0 1
/-- `top_k` of a batch of `B` requests and `N` tokens. -/
def top_k (B N : Nat) : Inst := mk .i32 B N 0 0
/-- `top_p` of a batch of `B` requests and `N` tokens. -/
def top_p (B N : Nat) : Inst := mk .f32 B N 0 0

theorem temperature_ok {B N : Nat} (h1 : 1 ≤ B) (h2 : B ≤ 8192) : (temperature B N).ok :=
  mk_ok _ _ _ _ _ (Or.inl rfl) h1 h2 (by norm_num) (by norm_num)
theorem top_k_ok {B N : Nat} (h1 : 1 ≤ B) (h2 : B ≤ 8192) : (top_k B N).ok :=
  mk_ok _ _ _ _ _ (Or.inr rfl) h1 h2 (by norm_num) (by norm_num)
theorem top_p_ok {B N : Nat} (h1 : 1 ≤ B) (h2 : B ≤ 8192) : (top_p B N).ok :=
  mk_ok _ _ _ _ _ (Or.inl rfl) h1 h2 (by norm_num) (by norm_num)

/-- Random batch: 1 request, 1 token: the three calls. -/
def setup_b1_k1_temperature : Inst := temperature 1 1
theorem setup_b1_k1_temperature_ok : setup_b1_k1_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b1_k1_top_k : Inst := top_k 1 1
theorem setup_b1_k1_top_k_ok : setup_b1_k1_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b1_k1_top_p : Inst := top_p 1 1
theorem setup_b1_k1_top_p_ok : setup_b1_k1_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- Random batch: 8 requests, 3 tokens each: the three calls. -/
def setup_b8_k3_temperature : Inst := temperature 8 24
theorem setup_b8_k3_temperature_ok : setup_b8_k3_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b8_k3_top_k : Inst := top_k 8 24
theorem setup_b8_k3_top_k_ok : setup_b8_k3_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b8_k3_top_p : Inst := top_p 8 24
theorem setup_b8_k3_top_p_ok : setup_b8_k3_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- Random batch: 64 requests, 0 to 5 tokens (176 tokens): the three calls. -/
def setup_b64_k5_temperature : Inst := temperature 64 176
theorem setup_b64_k5_temperature_ok : setup_b64_k5_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b64_k5_top_k : Inst := top_k 64 176
theorem setup_b64_k5_top_k_ok : setup_b64_k5_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b64_k5_top_p : Inst := top_p 64 176
theorem setup_b64_k5_top_p_ok : setup_b64_k5_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- Random batch: 256 requests, 0 to 4 tokens (481 tokens): the three calls. -/
def setup_b256_k4_temperature : Inst := temperature 256 481
theorem setup_b256_k4_temperature_ok : setup_b256_k4_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b256_k4_top_k : Inst := top_k 256 481
theorem setup_b256_k4_top_k_ok : setup_b256_k4_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b256_k4_top_p : Inst := top_p 256 481
theorem setup_b256_k4_top_p_ok : setup_b256_k4_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- Random batch: 1024 requests, 0 to 8 tokens (4022 tokens): the three calls. -/
def setup_b1024_k8_temperature : Inst := temperature 1024 4022
theorem setup_b1024_k8_temperature_ok : setup_b1024_k8_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b1024_k8_top_k : Inst := top_k 1024 4022
theorem setup_b1024_k8_top_k_ok : setup_b1024_k8_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b1024_k8_top_p : Inst := top_p 1024 4022
theorem setup_b1024_k8_top_p_ok : setup_b1024_k8_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- Random batch: 37 requests, 0 to 128 tokens (2245 tokens): the three calls. -/
def setup_b37_k128_temperature : Inst := temperature 37 2245
theorem setup_b37_k128_temperature_ok : setup_b37_k128_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b37_k128_top_k : Inst := top_k 37 2245
theorem setup_b37_k128_top_k_ok : setup_b37_k128_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b37_k128_top_p : Inst := top_p 37 2245
theorem setup_b37_k128_top_p_ok : setup_b37_k128_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- Random batch: 16 requests, 2 tokens each: the three calls. -/
def setup_b16_k2_temperature : Inst := temperature 16 32
theorem setup_b16_k2_temperature_ok : setup_b16_k2_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_b16_k2_top_k : Inst := top_k 16 32
theorem setup_b16_k2_top_k_ok : setup_b16_k2_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_b16_k2_top_p : Inst := top_p 16 32
theorem setup_b16_k2_top_p_ok : setup_b16_k2_top_p.ok := top_p_ok (by norm_num) (by norm_num)
/-- SETUP.md's serving sizes, 1024 requests of `MAX_SPEC_LEN = 128` tokens: the three calls. -/
def setup_serving_max_temperature : Inst := temperature 1024 (1024 * 128)
theorem setup_serving_max_temperature_ok : setup_serving_max_temperature.ok := temperature_ok (by norm_num) (by norm_num)
def setup_serving_max_top_k : Inst := top_k 1024 (1024 * 128)
theorem setup_serving_max_top_k_ok : setup_serving_max_top_k.ok := top_k_ok (by norm_num) (by norm_num)
def setup_serving_max_top_p : Inst := top_p 1024 (1024 * 128)
theorem setup_serving_max_top_p_ok : setup_serving_max_top_p.ok := top_p_ok (by norm_num) (by norm_num)

end Claims.RejectionExpand
