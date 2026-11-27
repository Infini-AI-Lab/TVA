import TVA

/-!
# `num_kv_splits_i32`: KV splits of split-K decode attention, SGLang

The kernel is `get_num_kv_splits_triton` of `source/metadata.py` (lines 56-108; it reads the
module-level names `triton` and `tl`, lines 52-53). The host `get_num_kv_splits` (lines 111-149)
launches it on the grid `(1,)` with the batch's KV lengths `seq_lens` (`[num_seq]`, `int64` or
`int32`), the `int32` output `num_kv_splits` (`[num_seq * num_group]`), the backend's head counts,
its largest number of splits `max_kv_splits` and the device's core count, `MAX_NUM_SEQ` the
power of two `SCHEDULE_SEQ`, and `uneven_batch` either `None` or an `int32` tensor. `SETUP.md`
lists the launches.

What is claimed: every covered launch ends normally, and there is one integer `E`, the kernel's
`ext_device_core_count`, with `C ≤ E`, `E ≤ v + C·2^-17` and `v - C·2^-17 < E + 1`, where
`C = device_core_count` and `v = C · max(log₂(L / 64), 1)` for the largest length `L`, such that
element `j < num_seq * num_group` of `num_kv_splits` is the exact `int32` value
`numSplits E (j / num_group)` (`Post`), the kernel's integer formula; with `uneven_batch`, its
element `0` is `1` when the batch is uneven (the smallest length, after the kernel's 80 % rule, is
below the largest) and `0` otherwise; every other byte is unchanged. `E` is computed in `float32`
(`tl.math.log2`, a division, a product, a truncation), so the device facts bound it only to that
set; every other step is exact.

The conditions: `seq_lens` is `int64` or `int32`; `1 ≤ num_seq ≤ 8192`; `MAX_NUM_SEQ` a power
of two, at least `num_seq`, at most `2^20`, with `MAX_NUM_SEQ * num_group ≤ 2^31`; `num_group ≥ 1`;
`1 ≤ num_kv_head ≤ num_head`; `num_seq * num_group * num_head ≤ 2^30`; `1 ≤ max_kv_splits ≤ 2^22`;
`1 ≤ device_core_count ≤ 2^16`; every length in `[1, 2^22]`; the buffers hold the batch and are
valid, and no output shares a byte with an input or with the other output.

The tolerance rests on the device facts for `/` on `float32` (`Device.DivFull`) and
`tl.math.log2` on `float32` (`Device.Log2F32`); README, "Tolerances".
-/
namespace Claims.NumKvSplits

open TVA

/-! ## Helpers of the transcription

`seq_lens` may be `int64` or `int32`. The model is one function of the dtype `dt`, computing in
the signed integer type of its width, `IntV true (wOf dt)`: for `int64` and `int32` that type is
`dt`'s own (`toW dt` and `ofW dt` are the identity there), so each line computes in the dtype the
source computes in. `Inst.ok` restricts the dtype to these two. -/

/-- The width of the lengths' dtype: `64` for `int64`, `32` otherwise. -/
def wOf : DType → Nat
  | .i64 => 64
  | _ => 32

/-- An element of `seq_lens` as a signed integer of its width (the identity for `int64` and
`int32`; `0` for any other dtype, which no covered launch has). -/
def toW : (dt : DType) → dt.Val → IntV true (wOf dt)
  | .i64, x => x
  | .i32, x => x
  | _, _ => 0

/-- A signed integer of the lengths' width as an element of `seq_lens` (the identity for `int64`
and `int32`). -/
def ofW : (dt : DType) → IntV true (wOf dt) → dt.Val
  | .i64, x => x
  | .i32, x => x
  | dt, _ => dt.ofBits 0

/-- The start of the integer `tl.max` fold: the smallest value of the type, below every element,
so the fold is the maximum of the elements. -/
def wMin (dt : DType) : IntV true (wOf dt) := IntV.ofInt (-2 ^ (wOf dt - 1))

/-- The start of the integer `tl.min` fold: the largest value of the type. -/
def wMax (dt : DType) : IntV true (wOf dt) := IntV.ofInt (2 ^ (wOf dt - 1) - 1)

/-- The `float32` constant Triton builds for the Python float `64.0` (`litF32` of the double
`0x4050000000000000`). -/
def c64 : F32 := FloatV.litF32 (FloatV.ofPat 0x4050000000000000)

/-- The `float32` constant Triton builds for the Python float `1.0` (`litF32` of the double
`0x3FF0000000000000`). -/
def c1 : F32 := FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)

/-! ## The model -/

/-- `get_num_kv_splits_triton` (source lines 57-108). `seq_lens_ptr` is `*dt` (`int64` or
`int32`), `num_kv_splits_ptr` is `*i32`; `uneven_batch_ptr` is `none` when the host passes `None`
(line 79 is then a compile-time `if` whose branch is not visited) and `some p` for a tensor
(`*i32`). The six integer arguments are runtime `int32` values (each in `[1, 2^31)`; an argument
equal to `1` is a compile-time constant, which gives the same values: README, "Modelling
decisions"); `MAX_NUM_SEQ` is the `tl.constexpr`, a Lean parameter. The kernel is one program: it
never reads `tl.program_id`. -/
def get_num_kv_splits_triton {dt : DType} (MAX_NUM_SEQ : Nat) (num_kv_splits_ptr : Ptr .i32)
    (seq_lens_ptr : Ptr dt)
    (num_seq num_group num_head num_kv_head max_kv_splits device_core_count : I32)
    (uneven_batch_ptr : Option (Ptr .i32)) : Kernel where
  body _ := do
    -- lines 70-71: the `int32` lanes and the signed mask `offs_seq < num_seq`
    let offs_seq := tl.arange 0 MAX_NUM_SEQ
    let mask_seq := offs_seq.ltS num_seq
    -- line 73: the masked load; `other=0` is an `int32` `0`, converted to the pointer's dtype
    let seq_lens0 ← tl.load (seq_lens_ptr + offs_seq) (some mask_seq)
      (some (Tensor.full (ofW dt (IntV.ofInt 0))))
    -- line 74: `tl.max` of an integer block (no widening at 32 and 64 bits) is the fold of the
    -- signed `IntV.max` over every lane, masked lanes (`0`) included
    let max_seq_len : IntV true (wOf dt) :=
      Tensor.foldAll IntV.max (wMin dt) (seq_lens0.map (toW dt))
    -- line 75: the same load, `other=max_seq_len` (a scalar of the pointer's dtype, broadcast)
    let seq_lens1 ← tl.load (seq_lens_ptr + offs_seq) (some mask_seq)
      (some (Tensor.full (ofW dt max_seq_len)))
    let seq_lens := seq_lens1.map (toW dt)
    -- line 76: `tl.min`, the fold of the signed `IntV.min`
    let min_seq_len : IntV true (wOf dt) := Tensor.foldAll IntV.min (wMax dt) seq_lens
    -- lines 77-78: a runtime `if`; the Python numbers `8` and `10` are built in the lengths'
    -- dtype (wrapping products), the comparison is signed
    let min_seq_len := if (max_seq_len * IntV.ofInt 8).lt (min_seq_len * IntV.ofInt 10) then
      max_seq_len else min_seq_len
    -- lines 79-80: `uneven_batch_ptr is not None` is a compile-time condition; the `int1`
    -- comparison `.to(tl.int32)`, stored to the scalar pointer
    let _ ← (match uneven_batch_ptr with
      | some p => do
        let u ← tl.convert .i1 .i32 (Tensor.scalar (min_seq_len.lt max_seq_len))
        tl.store1 p u.item
      | none => pure () : Prog Unit)
    -- line 81: `tl.cdiv` of two scalars of the lengths' dtype, then `tl.minimum` with the
    -- `int32` `max_kv_splits` promoted (sign-extended) to that dtype
    let q1 ← tl.cdiv (Tensor.scalar max_seq_len) (Tensor.scalar min_seq_len)
    let max_kv_splits_1 := IntV.min q1.item (IntV.cast max_kv_splits)
    -- line 82
    let kv_chunk_size_1 := (← tl.cdiv (Tensor.scalar max_seq_len)
      (Tensor.scalar max_kv_splits_1)).item
    -- line 85: `tl.cast(max_seq_len, tl.float32)` (`tl.itof`), then the default `float32`
    -- division by the constant `64.0` built in `float32`
    let msl ← tl.itof (f := .binary32) (Tensor.scalar max_seq_len)
    let ext_seq_len ← tl.div msl (Tensor.scalar c64)
    -- lines 86-88: `tl.log2` (`tl.math .log2`); `tl.maximum` with the Python float `1.0` (a
    -- `float32` tensor of its own dtype); the `int32` `device_core_count` times the `float32`
    -- tensor computes in `float32` (`tl.itof`, then the IEEE product, which no addition
    -- follows); `tl.cast(…, tl.int32)` truncates (`tl.ftoi`)
    let lg ← tl.log2 ext_seq_len
    let mx ← tl.maximum lg (Tensor.scalar c1)
    let dcc ← tl.itof (f := .binary32) (Tensor.scalar device_core_count)
    let prod ← tl.mul dcc mx
    let ext_device_core_count : I32 := (← tl.ftoi (sg := true) (w := 32) prod).item
    -- line 89: a tuple assignment; `16` becomes an `int32` scalar tensor, `//` truncates
    let block_h : I32 := IntV.ofInt 16
    let num_kv_group := (← tl.idiv (Tensor.scalar num_head) (Tensor.scalar num_kv_head)).item
    -- lines 90-95: a runtime `if` (the Python `1` is an `int32`); both branches define
    -- `token_grid`, wrapping `int32` products; `block_h` is used in the `else` branch only
    let token_grid ← (if num_kv_group.eq (IntV.ofInt 1) then
        pure (num_seq * num_group * num_head)
      else do
        let block_h := IntV.min block_h num_kv_group
        let c ← tl.cdiv (Tensor.scalar num_head) (Tensor.scalar block_h)
        pure (num_seq * num_group * c.item) : Prog I32)
    -- lines 96-98: `int32` throughout
    let q2 ← tl.cdiv (Tensor.scalar ext_device_core_count) (Tensor.scalar token_grid)
    let max_kv_splits_2 := IntV.min q2.item max_kv_splits
    -- line 99: the `int32` `max_kv_splits_2` promoted to the lengths' dtype in `tl.cdiv`'s
    -- body `(x + div - 1) // div` (both uses)
    let kv_chunk_size_2 := (← tl.cdiv (Tensor.scalar max_seq_len)
      (Tensor.scalar (IntV.cast max_kv_splits_2))).item
    -- lines 101-103: `tl.cdiv` of the block (line 75's load) by the broadcast scalars, then the
    -- signed integer `tl.maximum`
    let n1 ← tl.cdiv seq_lens (Tensor.full kv_chunk_size_1)
    let n2 ← tl.cdiv seq_lens (Tensor.full kv_chunk_size_2)
    let num_kv_splits := tl.imaximum n1 n2
    -- lines 105-106: wrapping `int32` products, the signed mask
    let offs_token := offs_seq * num_group
    let mask_token := offs_token.ltS (num_seq * num_group)
    -- lines 107-108: `for i in range(0, num_group)`, `int32` bounds; the pointer
    -- `(num_kv_splits_ptr + i) + offs_token`; the values converted to the pointer's `int32`
    -- (`Tensor.castInt`, which wraps; the identity for `int32` lengths)
    tl.forRange (0 : I32) num_group (1 : I32) () fun i _ =>
      tl.store (num_kv_splits_ptr + i + offs_token)
        (Tensor.castInt num_kv_splits : Tensor [MAX_NUM_SEQ] I32) (some mask_token)

/-! ## The conditions -/

/-- What the host chooses: one launch through `get_num_kv_splits` (source lines 111-149). The
tensors are given by their storage (`Buf`). -/
structure Inst where
  /-- The dtype of `seq_lens` (`int64`, or `int32`). -/
  dL : DType
  /-- `num_seq = seq_lens.shape[0]`, the batch's requests. -/
  S : Nat
  /-- `num_group = num_kv_splits.shape[0] // num_seq`, the tokens per request. -/
  G : Nat
  /-- `num_head`, the attention heads (of this rank). -/
  H : Nat
  /-- `num_kv_head`, the KV heads (of this rank). -/
  Hkv : Nat
  /-- `max_kv_splits`, the largest number of splits. -/
  K : Nat
  /-- `device_core_count`, the device's streaming multiprocessors. -/
  C : Nat
  /-- `MAX_NUM_SEQ`, the host's `SCHEDULE_SEQ`. -/
  MAX : Nat
  /-- `uneven_batch` is a tensor (`true`) or `None` (`false`). -/
  hasUneven : Bool
  /-- The storage of `seq_lens`. -/
  seq : Buf dL
  /-- The storage of the output `num_kv_splits` (`int32`). -/
  out : Buf .i32
  /-- The storage of `uneven_batch` (`int32`; unused without it). -/
  uneven : Buf .i32

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `seq_lens` is `int64` (the batch's lengths) or `int32` (a difference of `int32` index
  pointers). -/
  dL_ok : d.dL = .i64 ∨ d.dL = .i32
  /-- A batch of at least one request. -/
  S_pos : 1 ≤ d.S
  /-- `num_seq` is a batch size: at most `8192`. -/
  S_le : d.S ≤ 8192
  /-- `MAX_NUM_SEQ` is a power of two (`tl.arange`) … -/
  MAX_pow2 : ∃ e, d.MAX = 2 ^ e
  /-- … at most `2^20` (Triton's largest block) … -/
  MAX_le : d.MAX ≤ 2 ^ 20
  /-- … that holds the batch (the host's `SCHEDULE_SEQ` is `256` or `next_power_of_2(num_seq)`). -/
  S_MAX : d.S ≤ d.MAX
  /-- At least one token per request. -/
  G_pos : 1 ≤ d.G
  /-- `MAX_NUM_SEQ * num_group ≤ 2^31`: the `int32` `offs_seq * num_group` does not wrap, so the
  mask `offs_token < num_seq * num_group` holds exactly for the batch's lanes (a limit of the
  kernel: README, Exclusions). -/
  MAX_G : d.MAX * d.G ≤ 2 ^ 31
  /-- At least one KV head, and no more than the attention heads. -/
  Hkv_pos : 1 ≤ d.Hkv
  Hkv_le : d.Hkv ≤ d.H
  /-- `num_seq * num_group * num_head ≤ 2^30`: the `int32` `token_grid` and the addition in its
  `tl.cdiv` do not wrap (a limit of the kernel: README, Exclusions). -/
  grid_le : d.S * d.G * d.H ≤ 2 ^ 30
  /-- `max_kv_splits` is positive and at most `2^22` (a number of chunks of a context). -/
  K_pos : 1 ≤ d.K
  K_le : d.K ≤ 2 ^ 22
  /-- `device_core_count` is positive (the host does not launch otherwise) and at most `2^16`. -/
  C_pos : 1 ≤ d.C
  C_le : d.C ≤ 2 ^ 16
  /-- The buffers hold the batch: `num_seq` lengths, `num_seq * num_group` outputs, and one
  element of `uneven_batch` when given. -/
  seq_len : d.S ≤ d.seq.len
  out_len : d.S * d.G ≤ d.out.len
  uneven_len : d.hasUneven = true → 1 ≤ d.uneven.len
  /-- The outputs share no byte with `seq_lens` or with each other. -/
  out_seq : Disjoint d.out.bytes d.seq.bytes
  uneven_seq : d.hasUneven = true → Disjoint d.uneven.bytes d.seq.bytes
  uneven_out : d.hasUneven = true → Disjoint d.uneven.bytes d.out.bytes

/-- The model on an instance: the buffers' first elements, the six integer arguments as `int32`
values, and the constexpr. -/
def model (d : Inst) : Kernel :=
  get_num_kv_splits_triton d.MAX (d.out.ptr 0) (d.seq.ptr 0)
    (IntV.ofNat d.S) (IntV.ofNat d.G) (IntV.ofNat d.H) (IntV.ofNat d.Hkv) (IntV.ofNat d.K)
    (IntV.ofNat d.C) (if d.hasUneven then some (d.uneven.ptr 0) else none)

/-- The host's launch (source line 136): the grid `(1,)`, the source's one block shape
`[MAX_NUM_SEQ]`; `enable_fp_fusion` is Triton's default (`true`; no addition follows a product);
`num_warps = 4`, `num_stages = 3` change no result; `num_ctas = 1`, no `launch_pdl`,
`debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 1 [[d.MAX]] (fusion := true)

/-! ## The specification -/

/-- The signed value of an `int64` or `int32` element (`0` for any other dtype, which no covered
buffer has). -/
def ivOf {dt : DType} (x : dt.Val) : Int :=
  match dt, x with
  | .i64, y => y.toInt
  | .i32, y => y.toInt
  | _, _ => 0

/-- The value element `i` of a buffer holds in `m` (`0` when absent). -/
def valOf {dt : DType} (m : Mem) (b : Buf dt) (i : Nat) : Int :=
  match m.read dt (b.ptr (i : Int)).addr with
  | some x => ivOf x
  | none => 0

/-- Request `s`'s KV length, `seq_lens[s]`. -/
def len (d : Inst) (m : Mem) (s : Nat) : Int := valOf m d.seq s

/-- The largest length, `max_seq_len` (the lengths are positive, so the start `0` does not
count). -/
def maxLen (d : Inst) (m : Mem) : Int :=
  (List.range d.S).foldl (fun a s => max a (len d m s)) 0

/-- The smallest length (at most `maxLen`, the fold's start). -/
def minLen0 (d : Inst) (m : Mem) : Int :=
  (List.range d.S).foldl (fun a s => min a (len d m s)) (maxLen d m)

/-- `min_seq_len` after the kernel's rule (source lines 77-78): the largest length when
`8 · max < 10 · min` (the smallest is above 80 % of the largest), else the smallest. -/
def minLen (d : Inst) (m : Mem) : Int :=
  if maxLen d m * 8 < minLen0 d m * 10 then maxLen d m else minLen0 d m

/-- Ceiling division of positive integers, `(a + b - 1) / b` (what `tl.cdiv` computes on them). -/
def cdivZ (a b : Int) : Int := (a + b - 1) / b

/-- `kv_chunk_size_1` (source lines 81-82). -/
def chunk1 (d : Inst) (m : Mem) : Int :=
  cdivZ (maxLen d m) (min (cdivZ (maxLen d m) (minLen d m)) d.K)

/-- `token_grid` (source lines 89-95): with one attention head per KV head,
`num_seq · num_group · num_head`; otherwise `num_seq · num_group · ⌈num_head / block_h⌉` with
`block_h = min(16, num_head // num_kv_head)`. -/
def tokenGrid (d : Inst) : Int :=
  if (d.H : Int) / d.Hkv = 1 then (d.S : Int) * d.G * d.H
  else (d.S : Int) * d.G * cdivZ d.H (min 16 ((d.H : Int) / d.Hkv))

/-- `kv_chunk_size_2` (source lines 96-99) for `ext_device_core_count = E`. -/
def chunk2 (d : Inst) (m : Mem) (E : Int) : Int :=
  cdivZ (maxLen d m) (min (cdivZ E (tokenGrid d)) d.K)

/-- `num_kv_splits` of request `s` (source lines 101-103) for `ext_device_core_count = E`. -/
def numSplits (d : Inst) (m : Mem) (E : Int) (s : Nat) : Int :=
  max (cdivZ (len d m s) (chunk1 d m)) (cdivZ (len d m s) (chunk2 d m E))

/-- The exact value the `float32` lines 85-88 approximate before truncation:
`C · max(log₂(L / 64), 1)`, for `C = device_core_count` and `L` the largest length. -/
noncomputable def coreExact (d : Inst) (m : Mem) : ℝ :=
  (d.C : ℝ) * max (Real.logb 2 ((maxLen d m : ℝ) / 64)) 1

/-- What `ext_device_core_count = E` may be: an integer at least `C`, with `E ≤ v + C·2^-17` and
`v - C·2^-17 < E + 1` for `v = coreExact`: the truncation of a `float32` value at least `C` and
within `C·2^-17` of `v` (README, "Tolerances"). -/
def CoreOk (d : Inst) (m : Mem) (E : Int) : Prop :=
  (d.C : Int) ≤ E ∧ (E : ℝ) ≤ coreExact d m + d.C * 2 ^ (-17 : ℤ) ∧
    coreExact d m - d.C * 2 ^ (-17 : ℤ) < (E : ℝ) + 1

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present;
`uneven_batch` when given), and every request's length is in `[1, 2^22]` (a decode request has at
least one token; contexts up to `2^22` tokens). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.seq.Valid init ∧ d.out.Valid init ∧ (d.hasUneven = true → d.uneven.Valid init) ∧
    ∀ s, s < d.S → 1 ≤ len d init s ∧ len d init s ≤ 2 ^ 22

/-- What the launch leaves: for one `E` with `CoreOk`, element `j < num_seq * num_group` of
`num_kv_splits` holds `numSplits E (j / num_group)` (token `j` belongs to request
`j / num_group`); with `uneven_batch`, its element `0` holds `1` when `minLen < maxLen` and `0`
otherwise; every byte outside those elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∃ E : Int, CoreOk d init E ∧
    ∀ j : Nat, j < d.S * d.G → ∃ v : I32,
      fin.read .i32 (d.out.ptr (j : Int)).addr = some v ∧ v.toInt = numSplits d init E (j / d.G)) ∧
  (d.hasUneven = true → ∃ u : I32, fin.read .i32 (d.uneven.ptr 0).addr = some u ∧
    u.toInt = if minLen d init < maxLen d init then 1 else 0) ∧
  ∀ z, z ∉ d.out.elems {j | j < d.S * d.G} →
    (d.hasUneven = true → z ∉ d.uneven.elems {0}) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `get_num_kv_splits_triton` ends normally with `Post`. -/
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

/-- An instance with the host's layout: `S` requests of dtype `dt`, `G` tokens each, `H` and
`Hkv` heads, `K` splits at most, `C` cores, `MAX_NUM_SEQ = MAX`, and `uneven_batch` when `hu`;
`seq_lens` at `2^32`, the output at `2^33`, `uneven_batch` at `2^34`. -/
def mk (dt : DType) (S G H Hkv K C MAX : Nat) (hu : Bool) : Inst where
  dL := dt
  S := S
  G := G
  H := H
  Hkv := Hkv
  K := K
  C := C
  MAX := MAX
  hasUneven := hu
  seq := ⟨2 ^ 32, S⟩
  out := ⟨2 ^ 33, S * G⟩
  uneven := ⟨2 ^ 34, 1⟩

theorem mk_ok {dt : DType} {S G H Hkv K C MAX : Nat} {hu : Bool} (e : Nat)
    (hdt : dt = .i64 ∨ dt = .i32) (h1 : 1 ≤ S) (h2 : S ≤ 8192) (hM : MAX = 2 ^ e)
    (h3 : MAX ≤ 2 ^ 20) (h4 : S ≤ MAX) (h5 : 1 ≤ G) (h6 : MAX * G ≤ 2 ^ 31) (h7 : 1 ≤ Hkv)
    (h8 : Hkv ≤ H) (h9 : S * G * H ≤ 2 ^ 30) (h10 : 1 ≤ K) (h11 : K ≤ 2 ^ 22) (h12 : 1 ≤ C)
    (h13 : C ≤ 2 ^ 16) :
    (mk dt S G H Hkv K C MAX hu).ok := by
  have hSG : S * G ≤ 2 ^ 30 := le_trans (Nat.le_mul_of_pos_right _ (by omega)) h9
  refine ⟨hdt, h1, h2, ⟨e, hM⟩, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, le_refl _,
    le_refl _, fun _ => le_refl _, ?_, fun _ => ?_, fun _ => ?_⟩
  · refine (bytes_disjoint_of_le ?_).symm
    rcases hdt with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> omega
  · refine (bytes_disjoint_of_le ?_).symm
    rcases hdt with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> omega
  · refine (bytes_disjoint_of_le ?_).symm
    simp only [mk, DType.bytes]; push_cast; omega

/-- The memory in which every byte is present: `1` at `2^32` (the first length's low byte), `0`
elsewhere. -/
def oneMem : Mem := fun a => if a = 2 ^ 32 then some 1 else some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk .i32 1 1 1 1 1 1 256 false, oneMem,
    mk_ok 8 (Or.inr rfl) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) le_rfl
      (by norm_num) le_rfl le_rfl (by norm_num) le_rfl (by norm_num) le_rfl (by norm_num), ?_⟩
  have hb4 : (DType.i32.bytes : Int) = 4 := rfl
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, fun h => absurd h (by simp [mk]), ?_⟩
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp only [oneMem]; split <;> simp
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp only [oneMem]; split <;> simp
  · intro s hs
    have hs0 : s = 0 := by simp only [mk] at hs; omega
    subst hs0
    have hlen : len (mk .i32 1 1 1 1 1 1 256 false) oneMem 0 = 1 := by decide
    rw [hlen]; norm_num

/-! ## Coverage of the host's launches

The launches of the source's host (`SETUP.md`): the backend's call sites, each with the shapes
of its models and batches, and the source's own test cases. The contents are not part of
`Inst.ok`. Names: `<site>_<model>_bs<batch>`. The core count is the H200's `132` unless the name
says otherwise. -/

/-- Eager decode (`triton_backend.py:973`), with `uneven_batch`: Llama-3-8B (32 heads, 8 KV heads), `int64` lengths, the default 8 splits, batch 1. -/
def setup_decode_llama8b_bs1 : Inst := mk .i64 1 1 32 8 8 132 256 true
theorem setup_decode_llama8b_bs1_ok : setup_decode_llama8b_bs1.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Eager decode, Llama-3-8B, batch 7. -/
def setup_decode_llama8b_bs7 : Inst := mk .i64 7 1 32 8 8 132 256 true
theorem setup_decode_llama8b_bs7_ok : setup_decode_llama8b_bs7.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Eager decode, Llama-3-70B at TP 8 (8 heads, 1 KV head per rank), batch 256. -/
def setup_decode_llama70b_tp8_bs256 : Inst := mk .i64 256 1 8 1 8 132 256 true
theorem setup_decode_llama70b_tp8_bs256_ok : setup_decode_llama70b_tp8_bs256.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Eager decode, DeepSeek-V3 MLA at TP 8 (16 heads, 1 KV head), the MLA cap of 256 splits (`triton_backend.py:83-90`), batch 300 (`SCHEDULE_SEQ = 512`). -/
def setup_decode_dsv3_mla_tp8_bs300 : Inst := mk .i64 300 1 16 1 256 132 512 true
theorem setup_decode_dsv3_mla_tp8_bs300_ok : setup_decode_dsv3_mla_tp8_bs300.ok :=
  mk_ok 9 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Eager decode, Qwen2.5-7B (28 heads, 4 KV heads), batch 8192 (`SCHEDULE_SEQ = 8192`). -/
def setup_decode_qwen7b_bs8192 : Inst := mk .i64 8192 1 28 4 8 132 8192 true
theorem setup_decode_qwen7b_bs8192_ok : setup_decode_qwen7b_bs8192.ok :=
  mk_ok 13 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Eager decode with a split tile of 256 over a 131072-token context (`max_kv_splits = 512`, `triton_backend.py:369-372`), batch 64. -/
def setup_decode_tile256_bs64 : Inst := mk .i64 64 1 32 8 512 132 256 true
theorem setup_decode_tile256_bs64_ok : setup_decode_tile256_bs64.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Eager sliding-window lengths (`triton_backend.py:936`, no `uneven_batch`), batch 100. -/
def setup_window_llama8b_bs100 : Inst := mk .i64 100 1 32 8 8 132 256 false
theorem setup_window_llama8b_bs100_ok : setup_window_llama8b_bs100.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Speculative draft window lengths (`triton_backend.py:948`): `int32` differences of `window_kv_indptr`, batch 33. -/
def setup_draft_window_i32_bs33 : Inst := mk .i32 33 1 32 8 8 132 256 false
theorem setup_draft_window_i32_bs33_ok : setup_draft_window_i32_bs33.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Decode into preallocated buffers (`triton_backend.py:1415`, with `uneven_batch`), batch 160. -/
def setup_prealloc_decode_llama8b_bs160 : Inst := mk .i64 160 1 32 8 8 132 256 true
theorem setup_prealloc_decode_llama8b_bs160_ok : setup_prealloc_decode_llama8b_bs160.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Sliding-window lengths into preallocated buffers (`triton_backend.py:1421`), batch 160. -/
def setup_prealloc_window_llama8b_bs160 : Inst := mk .i64 160 1 32 8 8 132 256 false
theorem setup_prealloc_window_llama8b_bs160_ok : setup_prealloc_window_llama8b_bs160.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Multi-step draft (`triton_backend.py:2827`): `num_group = topk = 4`, batch 64. -/
def setup_multistep_topk4_bs64 : Inst := mk .i64 64 4 32 8 8 132 256 false
theorem setup_multistep_topk4_bs64_ok : setup_multistep_topk4_bs64.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Multi-step draft, `topk = 8`, batch 1024. -/
def setup_multistep_topk8_bs1024 : Inst := mk .i64 1024 8 32 8 8 132 1024 false
theorem setup_multistep_topk8_bs1024_ok : setup_multistep_topk8_bs1024.ok :=
  mk_ok 10 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- The source's test case 1: `[1]`, 32/8 heads, 8 splits, `int64`, `uneven_batch`. -/
def setup_test_1 : Inst := mk .i64 1 1 32 8 8 132 256 true
theorem setup_test_1_ok : setup_test_1.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Test case 2: eight lengths 4096, 32/32 heads. -/
def setup_test_2 : Inst := mk .i64 8 1 32 32 8 132 256 true
theorem setup_test_2_ok : setup_test_2.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Test case 3: 64 random lengths, 128/1 heads. -/
def setup_test_3 : Inst := mk .i64 64 1 128 1 8 132 256 true
theorem setup_test_3_ok : setup_test_3.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Test case 4: 300 random lengths, 64/8 heads, 16 splits, no `uneven_batch`. -/
def setup_test_4 : Inst := mk .i64 300 1 64 8 16 132 512 false
theorem setup_test_4_ok : setup_test_4.ok :=
  mk_ok 9 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Test case 5: 7 requests of 4 tokens, `int32` lengths, 32/8 heads. -/
def setup_test_5 : Inst := mk .i32 7 4 32 8 8 132 256 false
theorem setup_test_5_ok : setup_test_5.ok :=
  mk_ok 8 (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

end Claims.NumKvSplits
