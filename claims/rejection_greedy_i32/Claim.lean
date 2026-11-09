import TVA

/-!
# `rejection_greedy_i32`: vLLM's greedy rejection sampler (prepared claim)

Source: `source/rejection_greedy.py`. The kernel
`rejection_greedy_sample_kernel` (lines 67-122) runs one program per request: request `r` walks its
`n_r` packed draft tokens in order and, at each position, writes the target model's argmax token,
stopping after the first position whose draft token differs; if every draft matched it appends the
bonus token at position `n_r`; a request whose `is_greedy` flag is false returns at once. The
covered launches are the host's with `SYNTHETIC_MODE = False` (SETUP.md), both with
`is_greedy = None` (an all-greedy batch) and with an `is_greedy` mask (a mixed batch), on a batch
of `B ≥ 1` requests with at most `S = max_spec_len` drafts each.

The claim (`Statement`): on every covered instance, every launch that runs ends normally; each row
`r` of a greedy request holds, at column `j`, the target's argmax (its `int32` truncation, the
kernel's `.to(tl.int32)`) for the columns up to and including the first mismatch, the bonus token
at column `n_r` when every draft matched, and its initial value (the host's `-1`) everywhere else;
each row of a request that is not greedy keeps its initial value; every byte outside the output's
`B * (S + 1)` elements is unchanged.

This file states the claim and proves only `consistent`, `satisfiable` and the coverage of the
setup's launches (`setup_*_ok`); `Statement` is to be proved.
-/

namespace Claims.RejectionGreedy

open TVA

/-! ## The model -/

/-- `rejection_greedy_sample_kernel` (source lines 67-122). The pointers are `*i32` except
`target_argmax_ptr` (`*i64`) and `is_greedy_ptr` (`*i1`, a `bool` tensor); `max_spec_len` is a
runtime `int32` (the decorator's `do_not_specialize` keeps Triton from specialising it, so it is an
`int32` whatever its value in `[-2^31, 2^31)`, `1` included; SEMANTICS.md, "Scalar arguments").

`is_greedy_ptr` is `none` when the host passes `is_greedy = None` (the parameter holds the
compile-time `None`) and `some p` for a `bool` tensor. The covered launches pass
`synthetic_conditional_rates = None`, so the constexpr `SYNTHETIC_MODE` is `False`; the kernel
below is the instance's program with it fixed (the model map drops the branch it cuts).
`uniform_probs_ptr` and `synthetic_conditional_rates_ptr` are read only in that branch and have no
meaning in the model. -/
def rejection_greedy_sample_kernel
    (output_token_ids_ptr : Ptr .i32)
    (cu_num_draft_tokens_ptr : Ptr .i32)
    (draft_token_ids_ptr : Ptr .i32)
    (target_argmax_ptr : Ptr .i64)
    (bonus_token_ids_ptr : Ptr .i32)
    (is_greedy_ptr : Option (Ptr .i1))
    (max_spec_len : I32) : Kernel where
  body ctx := do
    -- line 80: `req_idx = tl.program_id(0)`, an `int32`
    let req_idx := tl.program_id ctx 0
    -- line 83: `is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)`.
    -- `is` gives a compile-time value (SEMANTICS.md, Order), so only one side is evaluated. With
    -- `None`, the Python `True` assigned to a name becomes the `int1` scalar tensor of `1`
    -- (SEMANTICS.md, "A name assigned a Python number holds a tensor"), the `Bool` `true` here.
    -- With a tensor, the `int1` load of element `req_idx` (`*i1` pointer, one byte per element).
    let is_greedy : Bool ←
      match is_greedy_ptr with
      | none => pure true
      | some is_greedy_ptr => tl.load1 (d := .i1) (is_greedy_ptr + req_idx)
    -- lines 84-86: `if not is_greedy: return`, a runtime `if` (its condition the `int1` tensor
    -- `not is_greedy`); the programs that take it end there (SEMANTICS.md, `return`), the others
    -- run the rest of the body. Only this branch returns, so the statements after it stay in the
    -- subset.
    if !is_greedy then
      return ()
    else
      -- lines 88-92: `start_idx = tl.zeros([], dtype=…) if req_idx == 0 else tl.load(…)`. The
      -- condition `req_idx == 0` is a runtime tensor, so each side is evaluated in its own branch
      -- of a runtime `if` (SEMANTICS.md, Order); `tl.zeros([], dtype=int32)` is the `int32`
      -- scalar `0`. `cu_num_draft_tokens_ptr + req_idx - 1` is `(ptr + req_idx) - 1`, Python's
      -- left-to-right order: the pointer advanced by `req_idx`, then the pointer minus the `int32`
      -- `1` (SEMANTICS.md, Pointers: `p - k`; `Ptr.sub_int_of_signed`: one element back).
      let start_idx : I32 ←
        if req_idx.eq (0 : I32) then
          pure (tl.zeros [] .i32).item
        else
          tl.load1 (d := .i32) (cu_num_draft_tokens_ptr + req_idx - (1 : I32))
      -- line 93
      let end_idx ← tl.load1 (d := .i32) (cu_num_draft_tokens_ptr + req_idx)
      -- line 94: the wrapping `int32` difference
      let num_draft_tokens := end_idx - start_idx
      -- line 96: `rejected = False`: the Python `False` becomes the `int1` scalar tensor `0`
      let rejected : Bool := false
      -- lines 97-114: `for pos in range(num_draft_tokens)`: a runtime `int32` bound (a plain
      -- name, nothing evaluated twice), `tl.forRange` from `0` step `1`; the loop carries
      -- `rejected`
      let rejected ← tl.forRange (0 : I32) num_draft_tokens (1 : I32) rejected
        (fun pos rejected => do
          -- line 98: `if not rejected:`, a runtime `if`
          if !rejected then
            -- line 99: `draft_token_ids_ptr + start_idx + pos` is `(ptr + start_idx) + pos`
            let draft_token_id ← tl.load1 (d := .i32) (draft_token_ids_ptr + start_idx + pos)
            -- line 100: `.to(tl.int32)` of the loaded `int64` (`tl.convert`, which wraps)
            let target64 ← tl.load1 (d := .i64) (target_argmax_ptr + start_idx + pos)
            let target_argmax_id : I32 := (← tl.convert .i64 .i32 (Tensor.scalar target64)).item
            -- lines 101-108: `if SYNTHETIC_MODE:` is a compile-time `if`, `False` on every
            -- covered launch, so only the else side is visited
            -- line 109
            let token_id := target_argmax_id
            -- line 110: the `int32` `!=`, giving `int1`
            let rejected := !(draft_token_id.eq target_argmax_id)
            -- lines 111-114: `token_id` is already `int32`, the pointer's dtype: no conversion;
            -- the address is `(ptr + req_idx * (max_spec_len + 1)) + pos`, `int32` arithmetic
            tl.store1 (output_token_ids_ptr + req_idx * (max_spec_len + (1 : I32)) + pos) token_id
            pure rejected
          else
            pure rejected)
      -- lines 116-122: `if not rejected:`, a runtime `if`; `rejected` holds what the last
      -- iteration left (`false` when the loop ran zero times)
      if !rejected then
        -- line 118
        let bonus_token_id ← tl.load1 (d := .i32) (bonus_token_ids_ptr + req_idx)
        -- lines 119-122
        tl.store1 (output_token_ids_ptr + req_idx * (max_spec_len + (1 : I32)) + num_draft_tokens)
          bonus_token_id
      else
        pure ()

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `rejection_greedy_sample_kernel` with
`synthetic_conditional_rates = None` and `SYNTHETIC_MODE = False` (SETUP.md); `is_greedy` is
`None` or a `bool` tensor; `uniform_probs` (`None`, or the `float32` tensor vLLM passes on a mixed
batch) is not read. -/
structure Inst where
  /-- `batch_size`: the grid, one program per request. -/
  B : Nat
  /-- `max_spec_len`, a runtime `int32` scalar (`do_not_specialize`). -/
  S : Nat
  /-- `output_token_ids_ptr`: a contiguous `[B, S+1]` `int32` tensor. -/
  out : Buf .i32
  /-- `cu_num_draft_tokens_ptr`: a `[B]` `int32` tensor. -/
  cu : Buf .i32
  /-- `draft_token_ids_ptr`: a contiguous `[num_tokens]` `int32` tensor. -/
  draft : Buf .i32
  /-- `target_argmax_ptr`: a `[num_tokens]` `int64` tensor. -/
  target : Buf .i64
  /-- `bonus_token_ids_ptr`: a contiguous `[B, 1]` `int32` tensor (`B` elements). -/
  bonus : Buf .i32
  /-- `is_greedy_ptr`: `none` for `is_greedy = None`, else a contiguous `[B]` `bool` tensor
  (`int1`, one byte per element). -/
  greedy : Option (Buf .i1)

/-- The `int32` value memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes
are absent). -/
def rd32 (b : Buf .i32) (m : Mem) (i : Int) : Int :=
  match m.read .i32 (b.ptr i).addr with | some v => v.toInt | none => 0

/-- The `int32` truncation (two's complement, what `.to(tl.int32)` gives) of the `int64` value
memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes are absent). -/
def rdt64 (b : Buf .i64) (m : Mem) (i : Int) : Int :=
  match m.read .i64 (b.ptr i).addr with
  | some v => Int.bmod v.toInt (2 ^ 32) | none => 0

/-- Whether request `r` is greedy: always with `is_greedy = None`, else the `int1` value memory
`m` holds at element `r` of the mask (a byte reads as `true` when non-zero, `DType.ofBits`;
`true` when the byte is absent). -/
def greedyAt (d : Inst) (m : Mem) (r : Nat) : Bool :=
  match d.greedy with
  | none => true
  | some g => match m.read .i1 (g.ptr r).addr with | some v => v | none => true

/-- Request `r`'s first packed position: `0` for the first request, else `cu[r-1]` (the
inclusive running sums of the draft counts, SETUP.md). -/
def startOf (d : Inst) (m : Mem) (r : Nat) : Int :=
  if r = 0 then 0 else rd32 d.cu m ((r - 1 : Nat) : Int)

/-- Request `r`'s number `n_r` of draft tokens: `cu[r] - start`. -/
def nOf (d : Inst) (m : Mem) (r : Nat) : Int := rd32 d.cu m ((r : Int)) - startOf d m r

/-- The first position `p ≥ p₀` of request `r`'s drafts whose draft token differs from the
target's truncated argmax, walking at most `fuel` positions; the walk stops at `n_r`. -/
def firstDiff (d : Inst) (m : Mem) (r : Nat) : Nat → Int → Int
  | 0, p => p
  | fuel + 1, p =>
      if p < nOf d m r then
        if rd32 d.draft m (startOf d m r + p) = rdt64 d.target m (startOf d m r + p) then
          firstDiff d m r fuel (p + 1)
        else p
      else p

/-- The number `k_r` of request `r`'s leading positions whose draft token equals the target's
truncated argmax; `n_r` when every draft matches. -/
def matchLen (d : Inst) (m : Mem) (r : Nat) : Int := firstDiff d m r (nOf d m r).toNat 0

/-- The flat index of column `j` of request `r`'s row of the output. -/
def oidx (d : Inst) (r j : Nat) : Nat := r * (d.S + 1) + j

/-- The value the launch leaves at column `j` of request `r`'s row: for a greedy request, the
target's truncated argmax for the columns up to the first mismatch (`k_r`, included, and below
`n_r`), the bonus token at column `n_r` when every draft matched, and `none` (the initial value)
everywhere else; for a request that is not greedy, `none` in every column. -/
def expect (d : Inst) (init : Mem) (r j : Nat) : Option Int :=
  let n := nOf d init r
  let k := matchLen d init r
  if greedyAt d init r = false then none
  else if (j : Int) < min (k + 1) n then some (rdt64 d.target init (startOf d init r + j))
  else if (j : Int) = n ∧ k = n then some (rd32 d.bonus init ((r : Int)))
  else none

/-- The instances covered: `B ≥ 1` requests in a grid CUDA accepts, the row offsets
`req_idx * (max_spec_len + 1) + pos` within `int32` (the kernel's limit, README "Exclusions"),
one `cu`, one `bonus` and (with a mask) one `is_greedy` entry per request, an output of
`B * (S + 1)` elements, and an output that shares no byte with an input. The inputs may overlap one
another (they are only read). -/
structure Inst.ok (d : Inst) : Prop where
  /-- At least one request (the host's `batch_size = len(num_draft_tokens) ≥ 1`). -/
  hB : 1 ≤ d.B
  /-- The grid has fewer than `2^31` programs (`Launch.Valid`). -/
  hgrid : d.B < 2 ^ 31
  /-- The `int32` offsets `req_idx * (max_spec_len + 1) + pos` (at most `B * (S + 1) - 1`) do
  not wrap, and `max_spec_len` is an `int32` (`S < 2^31`). -/
  hmul : d.B * (d.S + 1) ≤ 2 ^ 31
  /-- `cu_num_draft_tokens` has one entry per request. -/
  hcu : d.B ≤ d.cu.len
  /-- `bonus_token_ids` has one entry per request. -/
  hbonus : d.B ≤ d.bonus.len
  /-- The output holds `B` rows of `S + 1` columns. -/
  hout_len : d.B * (d.S + 1) ≤ d.out.len
  /-- The output shares no byte with `cu_num_draft_tokens`. -/
  hout_cu : Disjoint d.out.bytes d.cu.bytes
  /-- The output shares no byte with `draft_token_ids`. -/
  hout_draft : Disjoint d.out.bytes d.draft.bytes
  /-- The output shares no byte with `target_argmax`. -/
  hout_target : Disjoint d.out.bytes d.target.bytes
  /-- The output shares no byte with `bonus_token_ids`. -/
  hout_bonus : Disjoint d.out.bytes d.bonus.bytes
  /-- A mask has one entry per request. -/
  hgreedy : ∀ g, d.greedy = some g → d.B ≤ g.len
  /-- A mask shares no byte with the output. -/
  hout_greedy : ∀ g, d.greedy = some g → Disjoint d.out.bytes g.bytes

/-- The model on an instance: the kernel's arguments are the buffers' first elements (the mask's,
or `none`), `max_spec_len` as an `int32`. -/
def model (d : Inst) : Kernel :=
  rejection_greedy_sample_kernel (d.out.ptr 0) (d.cu.ptr 0) (d.draft.ptr 0) (d.target.ptr 0)
    (d.bonus.ptr 0) (d.greedy.map (·.ptr 0)) (IntV.ofNat d.S)

/-- The host's launch (source lines 164-177): a 1-D grid of `batch_size` programs. The kernel
has no block shape (no `tl.arange`). `enable_fp_fusion` is Triton's default (`true`); the
launch's `num_warps = 4` and `num_stages = 3` change no result; `grid1` keeps the defaults
`num_ctas = 1`, `launch_pdl = False` and `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.B [] (fusion := true)

/-- The inputs covered: the five buffers, and the mask when given, are valid (aligned, inside
`[0, 2^64)`, every byte present); `cu` holds the inclusive running sums of the draft counts (nonnegative, nondecreasing)
with its last entry at most the lengths of `draft_token_ids` and `target_argmax`; every request's
draft count is at most `max_spec_len` (the host passes `max_r n_r`). The draft, target and bonus
token values are arbitrary (`-1` drafts and any `int64` target included), and so are the mask's
bytes. The running sums hold for every request, greedy or not (vLLM's `cu_num_draft_tokens`
counts every request's drafts). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.out.Valid init ∧ d.cu.Valid init ∧ d.draft.Valid init ∧ d.target.Valid init ∧
    d.bonus.Valid init ∧ (∀ g, d.greedy = some g → g.Valid init) ∧
    0 ≤ rd32 d.cu init 0 ∧
    (∀ r, r + 1 < d.B → rd32 d.cu init ((r : Nat) : Int) ≤ rd32 d.cu init ((r + 1 : Nat) : Int)) ∧
    rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.draft.len ∧
    rd32 d.cu init ((d.B - 1 : Nat) : Int) ≤ d.target.len ∧
    (∀ r, r < d.B → nOf d init r ≤ d.S)

/-! ## The specification -/

/-- What the launch leaves: for each request `r < B` and each column `j < S + 1`, element
`(r, j)` of the output holds the value `expect` gives, or its initial value where `expect` gives
none; every byte outside the output's first `B * (S + 1)` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r, r < d.B → ∀ j, j < d.S + 1 →
      match expect d init r j with
      | some v => ∃ w : I32, fin.read .i32 (d.out.ptr (oidx d r j)).addr = some w ∧ w.toInt = v
      | none =>
          fin.read .i32 (d.out.ptr (oidx d r j)).addr =
            init.read .i32 (d.out.ptr (oidx d r j)).addr) ∧
    ∀ z, z ∉ d.out.elems {i | i < d.B * (d.S + 1)} → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The
  rule covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-- The claim to prove. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- Two buffers whose byte ranges are ordered do not overlap. -/
theorem bytes_disjoint_of_le {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : b.base + b.len * d.bytes ≤ c.base) : Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hcx => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hcx
  have hb := d.bytes_pos
  linarith

theorem bytes_disjoint_of_ge {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : c.base + c.len * d'.bytes ≤ b.base) : Disjoint b.bytes c.bytes :=
  (bytes_disjoint_of_le h).symm

/-- An instance of `B` requests, `max_spec_len = S` and `N` packed drafts, with the buffers laid
out one after another from `2^30` with `2^20`-byte gaps (the output first); with `masked`, an
`is_greedy` mask of `B` bytes after the bonus tokens, else `is_greedy = None`. -/
def mk (B S N : Nat) (masked : Bool := false) : Inst :=
  let o : Int := 2 ^ 30
  let c : Int := o + B * (S + 1) * 4 + 2 ^ 20
  let dr : Int := c + B * 4 + 2 ^ 20
  let t : Int := dr + N * 4 + 2 ^ 20
  let bo : Int := t + N * 8 + 2 ^ 20
  let g : Int := bo + B * 4 + 2 ^ 20
  { B := B, S := S, out := ⟨o, B * (S + 1)⟩, cu := ⟨c, B⟩, draft := ⟨dr, N⟩,
    target := ⟨t, N⟩, bonus := ⟨bo, B⟩, greedy := if masked then some ⟨g, B⟩ else none }

theorem mk_ok (B S N : Nat) (masked : Bool) (h1 : 1 ≤ B) (h2 : B < 2 ^ 31)
    (h3 : B * (S + 1) ≤ 2 ^ 31) : (mk B S N masked).ok := by
  refine ⟨h1, h2, h3, le_refl _, le_refl _, le_refl _, ?_, ?_, ?_, ?_, ?_, ?_⟩
  rotate_left 4
  · intro g hg
    cases masked <;> simp only [mk, Bool.false_eq_true, ↓reduceIte, reduceCtorEq,
      Option.some.injEq] at hg
    subst hg; exact le_refl _
  · intro g hg
    cases masked <;> simp only [mk, Bool.false_eq_true, ↓reduceIte, reduceCtorEq,
      Option.some.injEq] at hg
    subst hg
    refine bytes_disjoint_of_le ?_; simp only [mk, DType.bytes]; push_cast; nlinarith
  all_goals
    refine bytes_disjoint_of_le ?_; simp only [mk, DType.bytes]; push_cast; nlinarith

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
  have hlv : ∀ n : Nat, Math.Bytes.leValue (List.replicate n 0) = 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [List.replicate, Math.Bytes.leValue, ih]; norm_num
  rw [hlv]

theorem rd32_zeroMem (b : Buf .i32) (i : Int) : rd32 b zeroMem i = 0 := by
  rw [rd32]
  by_cases h : Mem.Aligned .i32 (b.ptr i).addr
  · rw [zeroMem_read h]; decide
  · simp only [Mem.read, h, ↓reduceIte]

theorem valid_zeroMem {d : DType} (b : Buf d) (h0 : 0 ≤ b.base)
    (h1 : b.base + b.len * d.bytes ≤ 2 ^ 64) (h2 : b.base % d.bytes = 0) : b.Valid zeroMem :=
  ⟨h0, h1, h2, fun _ _ => by simp [zeroMem]⟩

/-- The vLLM case "no drafts" (source line 225), `B = 1`, `S = 0`, no draft tokens, on the
all-zero memory (`cu = [0]`). -/
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨mk 1 0 0, zeroMem, mk_ok 1 0 0 false (le_refl _) (by norm_num) (by norm_num), ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals try (apply valid_zeroMem <;> norm_num [mk, DType.bytes])
  · intro g hg; simp [mk] at hg
  · rw [rd32_zeroMem]
  · intro r hr; simp [mk] at hr
  · rw [rd32_zeroMem]; simp [mk]
  · rw [rd32_zeroMem]; simp [mk]
  · intro r hr
    simp only [mk] at hr
    have : r = 0 := by omega
    subst this
    simp [nOf, startOf, rd32_zeroMem, mk]

/-! ## The setup's launches

The host's test cases (source lines 219-265): `B` requests, `max_spec_len = S = max_r n_r`,
`N = Σ n_r` packed drafts. vLLM's hand-written cases pass `is_greedy = None`; each random batch
runs twice, with `is_greedy = None` and with a random `is_greedy` mask (`setup_*_mask`). The random
batches' `S` and `N` are those the test's generator (`torch.Generator().manual_seed(0)`) draws. -/

/-- vLLM's "perfect match" and "early mismatch" (`[[1, 2, 3]]`). -/
def setup_vllm_one_req : Inst := mk 1 3 3
theorem setup_vllm_one_req_ok : setup_vllm_one_req.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- vLLM's `[[1, 2], [3]]`. -/
def setup_vllm_two_req : Inst := mk 2 2 3
theorem setup_vllm_two_req_ok : setup_vllm_two_req.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- vLLM's "single token" (`[[1]]`). -/
def setup_vllm_single : Inst := mk 1 1 1
theorem setup_vllm_single_ok : setup_vllm_single.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- vLLM's "no drafts" (`[[]]`). -/
def setup_vllm_no_drafts : Inst := mk 1 0 0
theorem setup_vllm_no_drafts_ok : setup_vllm_no_drafts.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- vLLM's `[[1, 2, 3], [4, 5, 6]]`. -/
def setup_vllm_multi_mismatch : Inst := mk 2 3 6
theorem setup_vllm_multi_mismatch_ok : setup_vllm_multi_mismatch.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 1 request, 1 draft. -/
def setup_b1_k1 : Inst := mk 1 1 1
theorem setup_b1_k1_ok : setup_b1_k1.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 8 requests, 3 drafts each. -/
def setup_b8_k3 : Inst := mk 8 3 24
theorem setup_b8_k3_ok : setup_b8_k3.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 64 requests, 0 to 5 drafts (`S = 5`, 134 drafts). -/
def setup_b64_k5 : Inst := mk 64 5 134
theorem setup_b64_k5_ok : setup_b64_k5.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 256 requests, 0 to 4 drafts, padded (`-1`) drafts (`S = 4`, 500 drafts). -/
def setup_b256_k4 : Inst := mk 256 4 500
theorem setup_b256_k4_ok : setup_b256_k4.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 1024 requests, 0 to 8 drafts (`S = 8`, 4084 drafts). -/
def setup_b1024_k8 : Inst := mk 1024 8 4084
theorem setup_b1024_k8_ok : setup_b1024_k8.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 37 requests, 0 to 128 drafts (`S = 123`, 2340 drafts). -/
def setup_b37_k128 : Inst := mk 37 123 2340
theorem setup_b37_k128_ok : setup_b37_k128.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 16 requests, 2 drafts each (accept 1.0). -/
def setup_b16_k2_all : Inst := mk 16 2 32
theorem setup_b16_k2_all_ok : setup_b16_k2_all.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch: 16 requests, 0 to 2 drafts (accept 0.0; `S = 2`, 17 drafts). -/
def setup_b16_k2_none : Inst := mk 16 2 17
theorem setup_b16_k2_none_ok : setup_b16_k2_none.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- Random batch with a mask: 1 request, 1 draft. -/
def setup_b1_k1_mask : Inst := mk 1 1 1 true
theorem setup_b1_k1_mask_ok : setup_b1_k1_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 8 requests, 3 drafts each. -/
def setup_b8_k3_mask : Inst := mk 8 3 24 true
theorem setup_b8_k3_mask_ok : setup_b8_k3_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 64 requests, 0 to 5 drafts. -/
def setup_b64_k5_mask : Inst := mk 64 5 134 true
theorem setup_b64_k5_mask_ok : setup_b64_k5_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 256 requests, 0 to 4 drafts, padded drafts. -/
def setup_b256_k4_mask : Inst := mk 256 4 500 true
theorem setup_b256_k4_mask_ok : setup_b256_k4_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 1024 requests, 0 to 8 drafts. -/
def setup_b1024_k8_mask : Inst := mk 1024 8 4084 true
theorem setup_b1024_k8_mask_ok : setup_b1024_k8_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 37 requests, 0 to 128 drafts. -/
def setup_b37_k128_mask : Inst := mk 37 123 2340 true
theorem setup_b37_k128_mask_ok : setup_b37_k128_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 16 requests, 2 drafts each, accept 1.0. -/
def setup_b16_k2_all_mask : Inst := mk 16 2 32 true
theorem setup_b16_k2_all_mask_ok : setup_b16_k2_all_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Random batch with a mask: 16 requests, 0 to 2 drafts, accept 0.0. -/
def setup_b16_k2_none_mask : Inst := mk 16 2 17 true
theorem setup_b16_k2_none_mask_ok : setup_b16_k2_none_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- SETUP.md's serving sizes: `MAX_SPEC_LEN = 128` drafts per request (every one used) and a
batch of 1024 requests. -/
def setup_serving_max : Inst := mk 1024 128 (1024 * 128)
theorem setup_serving_max_ok : setup_serving_max.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- SETUP.md's serving sizes on a mixed batch (an `is_greedy` mask). -/
def setup_serving_max_mask : Inst := mk 1024 128 (1024 * 128) true
theorem setup_serving_max_mask_ok : setup_serving_max_mask.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- The edge of `Inst.ok`: `B * (S + 1) = 2^31` (here `S = 127`, `2^24` requests). -/
def edge_offsets : Inst := mk (2 ^ 24) 127 0
theorem edge_offsets_ok : edge_offsets.ok := mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num)

end Claims.RejectionGreedy
