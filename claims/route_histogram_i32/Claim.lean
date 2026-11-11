import TVA

/-!
# `route_histogram_i32`: LoRA route histogram, SGLang

The kernel is `_route_histogram_kernel` of `source/route_histogram.py` (lines 109–144); it calls
the `@triton.jit` helpers `route_bucket_ids` (lines 62–89) and `_add_counts` (lines 92–106), which
the front end inlines, and it reads the module-level names `triton` and `tl` (lines 58–59). The
host `route_histogram` (lines 154–185) launches it on a 1-D grid of `cdiv(max(num_pairs, 1), 512)`
programs into a zeroed `int32` `counts` of `NUM_BUCKETS` elements; `SETUP.md` lists the launches.

Each pair `p` (a token and one of its groups) has a bucket: its token's adapter slot times
`GROUPS_PER_SLOT` plus its group (or the slot alone), the last bucket when the slot or the group
is invalid. Each program counts its `BLOCK` pairs: with `BINS > 0`, with `tl.histogram` over
`BINS` bins, adding its nonzero bins below `NUM_BUCKETS` into `counts` with one `tl.atomic_add` per
bin; with `BINS == 0`, with one `tl.atomic_add` of `1` per pair.

What is claimed, on both paths: every covered launch ends normally; each element
`b < NUM_BUCKETS` of `counts` is its initial value plus the exact number of pairs whose bucket is
`b` (the wrapping `int32` sum); every other byte is unchanged.

The conditions: `BLOCK` a power of two at most `2^20`; `BINS = 0`, or a power of two at most
`2^20` above `NUM_BUCKETS`; `NUM_BUCKETS = GROUPS_PER_SLOT · MAX_LORAS + 1 < 2^31`;
`1 ≤ GROUPS_PER_SLOT < 2^31`, `1 ≤ WIDTH < 2^31`, `num_pairs < 2^31`; the buffers hold what the
launch touches, and `counts` shares no byte with an input. Every `int32` content is covered.

The kernel is integer-only: the specification uses no device fact.
-/

namespace Claims.RouteHistogram

open TVA

/-! ## The model -/

/-- `route_bucket_ids` (source lines 62–89), inlined at its one call (lines 127–138). The
pointers are `*i32` (without groups `group_ids_ptr` is the host's placeholder, `token_slots`,
never read). The constexprs are Lean parameters; `if HAS_GROUPS:` and `if GROUPS_PER_SLOT == 1:`
are compile-time conditions, written as Lean `if`s on the constexprs. -/
def route_bucket_ids {B : Nat} (group_ids_ptr token_slots_ptr : Ptr .i32)
    (pair_ids : Tensor [B] I32) (pair_mask : Tensor [B] Bool)
    (GROUPS_PER_SLOT MAX_LORAS WIDTH : Nat) (HAS_GROUPS : Bool) : Prog (Tensor [B] I32) := do
  -- line 75: `pair_ids // WIDTH`: the constexpr `WIDTH` built in `int32` (the operator's dtype),
  -- and `//` the truncating `tl.idiv`
  let token_ids ← tl.idiv pair_ids (tl.full [B] (IntV.ofNat WIDTH : I32))
  -- line 76: the masked load; `other=-1` is the `int32` `-1`
  let slots ← tl.load (token_slots_ptr + token_ids) (some pair_mask)
      (some (tl.full [B] (IntV.ofInt (-1) : I32)))
  -- line 77: `(slots >= 0) & (slots < MAX_LORAS)`: signed `int32` comparisons with the
  -- constants, then `&`
  let live := ((tl.full [B] (IntV.ofInt 0 : I32)).le slots).and
    (slots.ltS (IntV.ofNat MAX_LORAS))
  -- line 78: `slots.to(tl.int32)` of an `int32` tensor: the identity (`castInt` to `I32`)
  let slots32 : Tensor [B] I32 := slots.castInt
  if HAS_GROUPS = true then
    -- line 80: the masked load of the pair's group, `other=-1`
    let groups ← tl.load (group_ids_ptr + pair_ids) (some pair_mask)
        (some (tl.full [B] (IntV.ofInt (-1) : I32)))
    -- line 81
    let live := live.and ((tl.full [B] (IntV.ofInt 0 : I32)).le groups)
    if GROUPS_PER_SLOT = 1 then
      -- lines 83 and 89: `buckets = slots`, then `tl.where(live, buckets, -1)`
      pure (tl.where live slots32 (tl.full [B] (IntV.ofInt (-1) : I32)))
    else
      -- line 85
      let live := live.and (groups.ltS (IntV.ofNat GROUPS_PER_SLOT))
      -- line 86: the wrapping `int32` `*` (the constexpr built in `int32`) and `+`;
      -- `groups.to(tl.int32)` is the identity
      let buckets := slots32 * (IntV.ofNat GROUPS_PER_SLOT : I32) + (groups.castInt : Tensor [B] I32)
      -- line 89
      pure (tl.where live buckets (tl.full [B] (IntV.ofInt (-1) : I32)))
  else
    -- lines 88 and 89
    pure (tl.where live slots32 (tl.full [B] (IntV.ofInt (-1) : I32)))

/-- `_add_counts` (source lines 92–106), inlined at its one call. `if BINS == 0:` (line 101) is a
compile-time condition on the constexpr, written as a Lean `if`: the path `BINS == 0` (line 102)
and the `tl.histogram` path (lines 104–106). -/
def add_counts {B : Nat} (counts_ptr : Ptr .i32) (bucket_ids : Tensor [B] I32)
    (mask : Tensor [B] Bool) (NUM_BUCKETS BINS : Nat) : Prog Unit := do
  -- line 100: the constexpr `NUM_BUCKETS - 1` (Python's) built in `int32`; signed `< 0`
  let buckets := tl.where (bucket_ids.ltS (IntV.ofInt 0 : I32))
    (tl.full [B] (IntV.ofNat (NUM_BUCKETS - 1) : I32)) bucket_ids
  if BINS = 0 then
    -- line 102: one `tl.atomic_add` per enabled lane (the caller's `pair_mask`) of the `int32`
    -- `1` (a constant built in `int32`, broadcast to `[B]`; the pointer's dtype already), with the
    -- defaults `sem="acq_rel"`, `scope="gpu"`; its result is unused
    let _ ← tl.atomic_add (counts_ptr + buckets) (tl.full [B] (IntV.ofInt 1 : I32)) (some mask)
    pure ()
  else
    -- line 104: a masked-off lane goes to bin `BINS - 1` (Python's `BINS - 1`, built in `int32`)
    let mine ← tl.histogram (tl.where mask buckets (tl.full [B] (IntV.ofNat (BINS - 1) : I32))) BINS
    -- line 105
    let bins := tl.arange 0 BINS
    -- line 106: `(bins < NUM_BUCKETS) & (mine > 0)` (signed `int32`), one `tl.atomic_add` per
    -- enabled bin with the defaults `sem="acq_rel"`, `scope="gpu"`; its result is unused
    let _ ← tl.atomic_add (counts_ptr + bins) mine
      (some ((bins.ltS (IntV.ofNat NUM_BUCKETS : I32)).and ((tl.full [BINS] (IntV.ofInt 0 : I32)).lt mine)))
    pure ()

/-- `_route_histogram_kernel` (source lines 109–144). The pointers are `*i32`; `num_pairs` is the
`int32` runtime argument (at the value `1` a compile-time constant, which gives the same values:
its one use, `pair_ids < num_pairs`, makes it an `int32` tensor); the constexprs are Lean
parameters. `if USE_PDL:` (line 143) is a compile-time condition, `False` on every covered launch
(the host passes `USE_PDL=False`): lines 143–144 are not visited. -/
def route_histogram_kernel (group_ids_ptr token_slots_ptr counts_ptr : Ptr .i32) (num_pairs : I32)
    (NUM_BUCKETS GROUPS_PER_SLOT MAX_LORAS WIDTH BLOCK BINS : Nat) (HAS_GROUPS : Bool) : Kernel where
  body ctx := do
    -- line 125: the wrapping `int32` product (the constexpr built in `int32`) and sum
    let pair_ids := tl.program_id ctx 0 * (IntV.ofNat BLOCK : I32) + tl.arange 0 BLOCK
    -- line 126: signed `int32` comparison with the scalar
    let pair_mask := pair_ids.ltS num_pairs
    -- lines 127–142: `_add_counts(counts_ptr, route_bucket_ids(...), pair_mask, ...)`
    let bucket_ids ← route_bucket_ids group_ids_ptr token_slots_ptr pair_ids pair_mask
      GROUPS_PER_SLOT MAX_LORAS WIDTH HAS_GROUPS
    add_counts counts_ptr bucket_ids pair_mask NUM_BUCKETS BINS

/-! ## The conditions -/

/-- What the host chooses: one launch of `_route_histogram_kernel`, with the contents of its
buffers. -/
structure Inst where
  /-- `num_pairs` (`groups.numel()`: `group_ids`'s, or `token_slots`'s without groups). -/
  n : Nat
  /-- `WIDTH`: pairs per token (`group_ids.shape[1]`, `1` without groups). -/
  width : Nat
  /-- `GROUPS_PER_SLOT`. -/
  gps : Nat
  /-- `MAX_LORAS`. -/
  maxLoras : Nat
  /-- `NUM_BUCKETS` (the host's `groups_per_slot * max_loras + 1`). -/
  numBuckets : Nat
  /-- `BLOCK` (the host's `_HIST_BLOCK = 512`). -/
  block : Nat
  /-- `BINS` (the host's `_histogram_bins(num_buckets, num_pairs)`). -/
  bins : Nat
  /-- `HAS_GROUPS = group_ids is not None`. -/
  hasGroups : Bool
  /-- `token_slots`. -/
  tokenSlots : Buf .i32
  /-- Its entries. -/
  tsVal : Nat → I32
  /-- `group_ids`, contiguous, read flat (the placeholder `token_slots` without groups). -/
  groupIds : Buf .i32
  /-- Its entries (read only with groups). -/
  giVal : Nat → I32
  /-- `counts`, the output. -/
  counts : Buf .i32
  /-- Its initial entries (the host's `torch.zeros`: `0`; any value is covered). -/
  cntVal : Nat → I32

/-- The instances covered: both paths, `BINS == 0` and the `tl.histogram` path with every bin
count Triton accepts above `NUM_BUCKETS`, every block Triton accepts, the host's bucket count below
`2^31`, every `num_pairs` below `2^31`, buffers holding what the launch touches, and a `counts`
sharing no byte with an input. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BLOCK` is a power of two (`tl.arange`) of at most `2^20` elements. -/
  block_pow2 : ∃ e, d.block = 2 ^ e
  block_le : d.block ≤ 2 ^ 20
  /-- `BINS = 0` (the path of one atomic per pair, the host's `_histogram_bins` when
  `NUM_BUCKETS ≥ 512` or `num_pairs < 16384`), or the `tl.histogram` path: `BINS` a power of two of
  at most `2^20` elements (`tl.arange(0, BINS)`) above `NUM_BUCKETS`, so every lane's value is a
  bin and the masked-off lanes' bin `BINS - 1` is not added (the host's
  `BINS = 1 << NUM_BUCKETS.bit_length()` is above `NUM_BUCKETS`). -/
  bins_ok : d.bins = 0 ∨ ((∃ e, d.bins = 2 ^ e) ∧ d.bins ≤ 2 ^ 20 ∧ d.numBuckets < d.bins)
  /-- The host's bucket count. -/
  buckets_eq : d.numBuckets = d.gps * d.maxLoras + 1
  /-- `NUM_BUCKETS < 2^31`: the host's check (it raises otherwise), so the constexprs
  `NUM_BUCKETS` and `NUM_BUCKETS - 1` are `int32` constants. -/
  buckets_lt : d.numBuckets < 2 ^ 31
  /-- `GROUPS_PER_SLOT ≥ 1` (the host's groups per slot; see README, Exclusions) and an `int32`
  constant. -/
  gps_pos : 0 < d.gps
  gps_lt : d.gps < 2 ^ 31
  /-- `WIDTH ≥ 1` (`//` by `WIDTH`) and an `int32` constant. -/
  width_pos : 0 < d.width
  width_lt : d.width < 2 ^ 31
  /-- `num_pairs < 2^31`: an `int32` argument, and the `int32` `pid * BLOCK + arange` of the grid
  does not wrap (a limit of the kernel: README, Exclusions). -/
  n_lt : d.n < 2 ^ 31
  /-- `token_slots` holds the token of every pair: `cdiv(num_pairs, WIDTH)` elements. -/
  tokenSlots_len : (d.n + d.width - 1) / d.width ≤ d.tokenSlots.len
  /-- With groups, `group_ids` holds every pair. -/
  groupIds_len : d.hasGroups = true → d.n ≤ d.groupIds.len
  /-- `counts` holds the buckets. -/
  counts_len : d.numBuckets ≤ d.counts.len
  /-- `counts` shares no byte with the inputs (the host allocates it). -/
  counts_ts : Disjoint d.counts.bytes d.tokenSlots.bytes
  counts_gi : d.hasGroups = true → Disjoint d.counts.bytes d.groupIds.bytes

/-- The inputs covered: the buffers are valid and hold the instance's entries. Every `int32`
value is covered in `token_slots`, `group_ids` and `counts`. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.tokenSlots.Valid init ∧ d.counts.Valid init ∧
    d.tokenSlots.Holds init d.tsVal ∧ d.counts.Holds init d.cntVal ∧
    (d.hasGroups = true → d.groupIds.Valid init ∧ d.groupIds.Holds init d.giVal)

/-- The model on an instance (`USE_PDL = False`; `num_pairs` as an `int32`). -/
def model (d : Inst) : Kernel :=
  route_histogram_kernel (d.groupIds.ptr 0) (d.tokenSlots.ptr 0) (d.counts.ptr 0) (IntV.ofNat d.n)
    d.numBuckets d.gps d.maxLoras d.width d.block d.bins d.hasGroups

/-- The host's launch (lines 169–184): a 1-D grid of `cdiv(max(num_pairs, 1), BLOCK)` programs;
the block shapes `[BLOCK]`, and `[BINS]` on the `tl.histogram` path; `enable_fp_fusion = true` (the default; no float
operation), `debug = false`, `num_ctas = 1`, no `launch_pdl`. `num_warps = 8` is
performance-only; `num_stages = 3` changes no outcome of a model without a conflict inside a
program. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((max d.n 1 + d.block - 1) / d.block)
    (if d.bins = 0 then [[d.block]] else [[d.block], [d.bins]]) (fusion := true)

/-! ## The specification -/

/-- The bucket of pair `p`, from the host's inputs as the source's docstring defines it: the
pair's token is `p / WIDTH`, its slot `token_slots[p / WIDTH]`, its group `group_ids.view(-1)[p]`;
the bucket is `slot * GROUPS_PER_SLOT + group` with groups and more than one group per slot,
`slot` otherwise, when the slot is in `[0, MAX_LORAS)` and (with groups) the group is `≥ 0` and,
with more than one group per slot, `< GROUPS_PER_SLOT`; `NUM_BUCKETS - 1` otherwise. -/
def bucketOf (d : Inst) (p : Nat) : Nat :=
  let sl := (d.tsVal (p / d.width)).toInt
  let g := (d.giVal p).toInt
  if 0 ≤ sl ∧ sl < d.maxLoras ∧ (d.hasGroups = true → 0 ≤ g ∧ (d.gps = 1 ∨ g < d.gps)) then
    (if d.hasGroups = true ∧ d.gps ≠ 1 then sl * d.gps + g else sl).toNat
  else d.numBuckets - 1

/-- The number of pairs `p < num_pairs` in bucket `b`. -/
def pairsIn (d : Inst) (b : Nat) : Nat :=
  ((Finset.range d.n).filter (fun p => bucketOf d p = b)).card

/-- What the launch leaves: element `b < NUM_BUCKETS` of `counts` is its initial value plus the
number of pairs in bucket `b` (the wrapping `int32` sum; with the host's zero `counts`, exactly
the count); every byte outside those elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b : Nat, b < d.numBuckets →
    fin.read .i32 (d.counts.ptr b).addr = some (d.cntVal b + IntV.ofNat (pairsIn d b))) ∧
  ∀ z, z ∉ d.counts.elems {b | b < d.numBuckets} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_route_histogram_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- An instance of the host's launch with the buffers at fixed, separate addresses
(`token_slots` at `2^32`, `group_ids` at `2^33`, `counts` at `2^34`) and zero contents:
`ntok` tokens of `w` pairs each (`n = ntok·w`), `gps` groups per slot, `ml` adapters,
`hg` with groups, `BLOCK = 2^eb`, and `BINS = 2^ek` with `hist` (the `tl.histogram` path),
`BINS = 0` without. -/
def mk (ntok w gps ml : Nat) (hg : Bool) (eb : Nat) (hist : Bool) (ek : Nat) : Inst where
  n := ntok * w
  width := w
  gps := gps
  maxLoras := ml
  numBuckets := gps * ml + 1
  block := 2 ^ eb
  bins := if hist then 2 ^ ek else 0
  hasGroups := hg
  tokenSlots := ⟨2 ^ 32, ntok⟩
  tsVal := fun _ => 0
  groupIds := ⟨2 ^ 33, ntok * w⟩
  giVal := fun _ => 0
  counts := ⟨2 ^ 34, gps * ml + 1⟩
  cntVal := fun _ => 0

theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {ntok w gps ml : Nat} {hg : Bool} {eb : Nat} {hist : Bool} {ek : Nat}
    (heb : eb ≤ 20) (hbins : hist = true → ek ≤ 20 ∧ gps * ml + 1 < 2 ^ ek)
    (hnb : gps * ml + 1 < 2 ^ 31) (hgps : 0 < gps) (hgps' : gps < 2 ^ 31) (hw : 0 < w)
    (hw' : w < 2 ^ 31) (hn : ntok * w < 2 ^ 31) (hsz : ntok * w * 4 ≤ 2 ^ 32) :
    (mk ntok w gps ml hg eb hist ek).ok where
  block_pow2 := ⟨eb, rfl⟩
  block_le := Nat.pow_le_pow_right (by norm_num) heb
  bins_ok := by
    cases hist with
    | false => exact Or.inl rfl
    | true =>
      obtain ⟨hek, hlt⟩ := hbins rfl
      exact Or.inr ⟨⟨ek, rfl⟩, Nat.pow_le_pow_right (by norm_num) hek, hlt⟩
  buckets_eq := rfl
  buckets_lt := hnb
  gps_pos := hgps
  gps_lt := hgps'
  width_pos := hw
  width_lt := hw'
  n_lt := hn
  tokenSlots_len := by
    show (ntok * w + w - 1) / w ≤ ntok
    rw [Nat.div_le_iff_le_mul_add_pred hw]
    have : ntok * w + w - 1 = w * ntok + (w - 1) := by rw [Nat.mul_comm]; omega
    omega
  groupIds_len := fun _ => le_refl _
  counts_len := le_refl _
  counts_ts := bytes_disj _ _ (Or.inr (by
    simp only [mk, DType.bytes]; push_cast
    have : (ntok : Int) * 4 ≤ 2 ^ 32 := by
      have h1 : ntok * 4 ≤ ntok * w * 4 := by nlinarith
      exact_mod_cast le_trans h1 hsz
    omega))
  counts_gi := fun _ => bytes_disj _ _ (Or.inr (by
    simp only [mk, DType.bytes]; push_cast
    have : ((ntok * w : Nat) : Int) * 4 ≤ 2 ^ 32 := by exact_mod_cast hsz
    push_cast at this; omega))

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1 1 1 1 false 0 true 2, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num), ?_⟩
  have hb4 : (DType.i32.bytes : Int) = 4 := rfl
  have hread : ∀ a : Int, Mem.Aligned .i32 a → zeroMem.read .i32 a = some 0 := by
    intro a ha
    simp only [Mem.read, ha, ite_true]
    simp [zeroMem, Mem.readBytes, DType.bytes]
    rfl
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ?_, ?_, ?_⟩
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp [zeroMem]
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp [zeroMem]
  · intro i hi
    have hi0 : i = 0 := by simp only [mk] at hi; omega
    subst hi0
    exact hread _ (by simp only [mk, Buf.ptr, Mem.Aligned, hb4]; decide)
  · intro i hi
    have hi2 : i < 2 := by simp only [mk] at hi; omega
    interval_cases i <;> exact hread _ (by simp only [mk, Buf.ptr, Mem.Aligned, hb4]; decide)
  · intro hc; simp [mk] at hc

/-! ## Coverage of the host's launches

The host's test launches (source lines 209–246) on both paths, with the host's `BLOCK = 512` and
its `BINS` (`_histogram_bins`: `0` when `NUM_BUCKETS ≥ 512` or `num_pairs < 16384`, otherwise
`1 << NUM_BUCKETS.bit_length()`), and the real routes `SETUP.md` names at their extremes. The
contents are not part of `Inst.ok`. -/

/-- `dense`, 16384 tokens, 1 adapter (`NUM_BUCKETS = 2`, `BINS = 4`). -/
def setup_dense_16384 : Inst := mk 16384 1 1 1 false 9 true 2
theorem setup_dense_16384_ok : setup_dense_16384.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `dense`, 16385 tokens, 4 adapters (`NUM_BUCKETS = 5`, `BINS = 8`). -/
def setup_dense_16385 : Inst := mk 16385 1 1 4 false 9 true 3
theorem setup_dense_16385_ok : setup_dense_16385.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `dense`, 100000 tokens, 8 adapters (`NUM_BUCKETS = 9`, `BINS = 16`). -/
def setup_dense_100000 : Inst := mk 100000 1 1 8 false 9 true 4
theorem setup_dense_100000_ok : setup_dense_100000.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `dense`, 32768 tokens, 16 adapters (`NUM_BUCKETS = 17`, `BINS = 32`). -/
def setup_dense_32768 : Inst := mk 32768 1 1 16 false 9 true 5
theorem setup_dense_32768_ok : setup_dense_32768.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `moe per expert`, 2048×8 pairs, 64 groups per slot, 4 adapters (`NUM_BUCKETS = 257`,
`BINS = 512`). -/
def setup_moe_2048x8 : Inst := mk 2048 8 64 4 true 9 true 9
theorem setup_moe_2048x8_ok : setup_moe_2048x8.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `moe per expert`, 8193×2 pairs, 8 groups per slot, 8 adapters (`NUM_BUCKETS = 65`,
`BINS = 128`). -/
def setup_moe_8193x2 : Inst := mk 8193 2 8 8 true 9 true 7
theorem setup_moe_8193x2_ok : setup_moe_8193x2.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `moe per expert`, 16384×8 pairs, 32 groups per slot, 8 adapters (`NUM_BUCKETS = 257`,
`BINS = 512`). -/
def setup_moe_16384x8 : Inst := mk 16384 8 32 8 true 9 true 9
theorem setup_moe_16384x8_ok : setup_moe_16384x8.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `moe shared outer`, 3000×6 pairs, 1 group per slot, 8 adapters (`NUM_BUCKETS = 9`,
`BINS = 16`). -/
def setup_shared_3000x6 : Inst := mk 3000 6 1 8 true 9 true 4
theorem setup_shared_3000x6_ok : setup_shared_3000x6.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `pairs (heads)`, 1024 tokens × 32 heads as 32768 pairs of width 1, 32 groups per slot,
4 adapters (`NUM_BUCKETS = 129`, `BINS = 256`). -/
def setup_heads_1024x32 : Inst := mk 32768 1 32 4 true 9 true 8
theorem setup_heads_1024x32_ok : setup_heads_1024x32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- A dense route with 64 adapters (`NUM_BUCKETS = 65`, `BINS = 128`) over `2^24` tokens. -/
def setup_dense_large : Inst := mk (2 ^ 24) 1 1 64 false 9 true 7
theorem setup_dense_large_ok : setup_dense_large.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `dense, one atomic per pair`, 1000 tokens, 8 adapters (`NUM_BUCKETS = 9`; fewer than
16384 pairs, so `BINS = 0`). -/
def setup_dense_1000 : Inst := mk 1000 1 1 8 false 9 false 0
theorem setup_dense_1000_ok : setup_dense_1000.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- `moe, one atomic per pair`, 4096×8 pairs, 128 groups per slot, 4 adapters
(`NUM_BUCKETS = 513`, `BINS = 0`). -/
def setup_moe_4096x8_513 : Inst := mk 4096 8 128 4 true 9 false 0
theorem setup_moe_4096x8_513_ok : setup_moe_4096x8_513.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- MoE LoRA per expert at prefill size: 16384 tokens × top-8, 128 local experts, 4 adapters
(`NUM_BUCKETS = 513 ≥ 512`, so `BINS = 0`). -/
def setup_moe_lora_128x4 : Inst := mk 16384 8 128 4 true 9 false 0
theorem setup_moe_lora_128x4_ok : setup_moe_lora_128x4.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- MoE LoRA per expert, 65536 tokens × top-8, 256 local experts, 8 adapters
(`NUM_BUCKETS = 2049`, `BINS = 0`). -/
def setup_moe_lora_256x8 : Inst := mk 65536 8 256 8 true 9 false 0
theorem setup_moe_lora_256x8_ok : setup_moe_lora_256x8.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- MoE LoRA per expert with `384 × 32 ≥ 8192` route buckets (the large builder at any pair
count): 96 tokens × top-8, 384 local experts, 32 adapters (`NUM_BUCKETS = 12289`, `BINS = 0`). -/
def setup_moe_lora_384x32 : Inst := mk 96 8 384 32 true 9 false 0
theorem setup_moe_lora_384x32_ok : setup_moe_lora_384x32.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- A route of fewer than 16384 pairs with one group per slot: 2000 tokens × top-6, 8 adapters
(`NUM_BUCKETS = 9`, `BINS = 0`). -/
def setup_shared_outer_small : Inst := mk 2000 6 1 8 true 9 false 0
theorem setup_shared_outer_small_ok : setup_shared_outer_small.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end Claims.RouteHistogram
