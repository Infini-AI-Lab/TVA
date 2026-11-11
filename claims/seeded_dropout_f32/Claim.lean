import TVA

/-!
# `seeded_dropout_f32`: seeded dropout, Triton tutorial 04

The kernel is `_seeded_dropout` of `source/04-low-memory-dropout.py` (lines 110–131; it reads the
module-level names `triton` and `tl`, bound by the imports of lines 38–39). Its host
`seeded_dropout` (lines 134–140) allocates `output = torch.empty_like(x)` and launches a 1-D grid of
`cdiv(n_elements, 1024)` programs with `BLOCK_SIZE=1024`, passing the Python `float` `p` and the
Python `int` `seed`. The file calls it on a `float32` vector of `10` elements with `p = 0.5` and the
seeds `123` (twice) and `512` (lines 143–147); `SETUP.md` lists the launches.

What is claimed: every covered launch ends normally, and for each `j < n` the output element `j` is
* `+0` exactly when the element's random number `u j` is not above `p` (the `float32` the kernel
  receives); `u j` is `tl.rand(seed, j)`: the first word of Philox 4x32-10 on the counter
  `(j, 0, 0, 0)` and the key `(seed mod 2^32, (seed mod 2^64) / 2^32)`, made non-negative and
  scaled to `float32` (`uRef`), decided bit for bit;
* otherwise, for a finite `x[j]` with `|x[j] / dRef| ≤ 2^126`, a finite `float32` within
  `2^-22·|x[j]/dRef| + 2^-148` of the exact quotient `x[j] / dRef`, where `dRef` is the IEEE
  `float32` `1 - p`;
and every other byte is unchanged.

The conditions: `BLOCK_SIZE` is a power of two at most `2^20`; `n < 2^31`; the kernel's `p` is a
finite `float32` in `[0, 1]`; `-2^63 ≤ seed < 2^64`; both buffers hold at least `n` elements; the
first `n` elements of `output` share no byte with the first `n` elements of `x`. Every `float32`
input is covered.

The tolerance of the kept values is the device fact for `/` on `float32` (`Device.DivFull`).
-/

namespace Claims.SeededDropout

open TVA

/-! ## The model -/

/-- The Python `1` of line 130 (`1 - p`), an operand of `-` with the `float32` scalar `p`: built in
`float32` from the double `1.0` (`TVA.Triton.Promote`: a Python number with an arithmetic
operator). -/
def one32 : F32 := FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)

/-- `_seeded_dropout` (source lines 110–131). The pointers are `*fp32`; `n_elements` is an `int32`
(an integer argument in `[2, 2^31)`, or `0`; at `1` it is a compile-time constant, which gives the
same model, see `model`); `p` is the runtime `float32` scalar (a Python `float` argument,
`FloatV.hostF32` of the double, see `model`); `seed` is the runtime integer scalar, of the type
Triton gives the Python `int` (`int32`, `int64` or `uint64`, see `model`), hence the implicit
`sg`, `w`; `BLOCK_SIZE : tl.constexpr` is a Lean parameter. -/
def seeded_dropout {sg : Bool} {w : Nat} (x_ptr output_ptr : Ptr .f32) (n_elements : I32)
    (p : F32) (seed : IntV sg w) (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 120
    let pid := tl.program_id ctx 0
    -- line 121: the constexpr `BLOCK_SIZE` built in `int32`, the wrapping `int32` product
    let block_start := pid * (IntV.ofNat BLOCK_SIZE : I32)
    -- line 122: the `int32` scalar added to every lane of the `int32` `tl.arange`
    let offsets := block_start + tl.arange 0 BLOCK_SIZE
    -- line 124: the signed `int32` comparison with the scalar `n_elements`
    let mask := offsets.ltS n_elements
    -- line 125: masked load without `other`: the disabled lanes are unspecified
    let x ← tl.load (x_ptr + offsets) (some mask)
    -- line 127: `tl.rand(seed, offsets)` on the `int32` offsets, 10 rounds (the default); every
    -- lane, the disabled ones included (their values reach no stored lane)
    let random ← tl.rand seed offsets
    -- line 128: `random > p`: the ordered `float32` comparison of each lane with the scalar `p`
    -- (no promotion: both `float32`), written `p < random`; a comparison, not an addition, so
    -- the product inside `tl.rand` fuses with nothing
    let x_keep := Tensor.flt (bc p) random
    -- line 130: `1 - p`: the scalar `float32` subtraction, IEEE (`tl.sub`: no product feeds it)
    let one_minus_p ← tl.sub (Tensor.scalar one32) (Tensor.scalar p)
    -- line 130: `x / (1 - p)`: the platform's `float32` division of each lane by the scalar
    -- (`tl.div`), on every lane
    let q ← tl.div x (bc one_minus_p.item)
    -- line 130: `tl.where(x_keep, …, 0.0)`: the Python `0.0`, a value of `tl.where`, built in
    -- `float32`, the `+0`
    let output := tl.where x_keep q (bc (FloatV.litF32 (FloatV.ofPat 0)))
    -- line 131: `output` is already `float32`, the pointer's dtype: no conversion
    tl.store (output_ptr + offsets) output (some mask)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `_seeded_dropout`. -/
structure Inst where
  /-- `n_elements` (the host passes `x.numel()`). -/
  n : Nat
  /-- `BLOCK_SIZE` (the host passes `1024`). -/
  BLOCK_SIZE : Nat
  /-- `p`, the Python `float` (a double; the file passes `0.5`). -/
  p : F64
  /-- `seed`, the Python `int` (the file passes `123` and `512`). -/
  seed : Int
  /-- `x_ptr`: a contiguous `float32` tensor. -/
  x : Buf .f32
  /-- `output_ptr`: a contiguous `float32` tensor. -/
  output : Buf .f32

/-- The `float32` the kernel receives for `p`: the double rounded to nearest even
(`FloatV.hostF32`). -/
def p32 (d : Inst) : F32 := FloatV.hostF32 d.p

/-- The instances covered: every block size Triton accepts, every `n` that `n_elements` holds as an
`int32`, every `p` whose `float32` lies in `[0, 1]`, every seed Triton passes as an
integer argument, buffers of at least `n` elements, and an output whose first `n` elements share
no byte with the first `n` elements of `x`. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BLOCK_SIZE` is a power of two (Triton requires it of `tl.arange`). -/
  block_pow2 : ∃ e, d.BLOCK_SIZE = 2 ^ e
  /-- `BLOCK_SIZE` is at most `2^20` elements (Triton's largest block). -/
  block_le : d.BLOCK_SIZE ≤ 2 ^ 20
  /-- `n < 2^31`: Triton passes `n_elements` as an `int32` (or, at `n = 1`, as a constant), and the
  `int32` offsets `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` of the grid `cdiv(n,
  BLOCK_SIZE)` do not wrap. -/
  n_lt : d.n < 2 ^ 31
  /-- The kernel's `p` is finite. -/
  p_fin : (p32 d).isFinite = true
  /-- The kernel's `p` is at least `0` (`-0.0` included): a dropout probability (README,
  Exclusions). -/
  p_ge : 0 ≤ (p32 d).toRat
  /-- The kernel's `p` is at most `1`. At `p = 1` the divisor `1 - p` is `0`, but every random
  number is below `1`, so no stored lane takes a quotient. -/
  p_le : (p32 d).toRat ≤ 1
  /-- `seed ≥ -2^63`: below, Triton rejects the integer argument on the host. -/
  seed_ge : -2 ^ 63 ≤ d.seed
  /-- `seed < 2^64`: from `2^64` on, Triton rejects the integer argument on the host. -/
  seed_lt : d.seed < 2 ^ 64
  /-- `x` has at least `n` elements. -/
  x_len : d.n ≤ d.x.len
  /-- `output` has at least `n` elements. -/
  output_len : d.n ≤ d.output.len
  /-- The first `n` elements of `output` share no byte with the first `n` elements of `x`. -/
  output_x : Disjoint (d.output.elems {j | j < d.n}) (d.x.elems {j | j < d.n})

/-- The kernel on an instance, with the seed of type `IntV sg w`. -/
def modelAt {sg : Bool} {w : Nat} (d : Inst) (seed : IntV sg w) : Kernel :=
  seeded_dropout (d.x.ptr 0) (d.output.ptr 0) (IntV.ofNat d.n) (p32 d) seed d.BLOCK_SIZE

/-- The model on an instance: the buffers' first elements, `n` as an `int32`, `p` as the
launcher's `float32` (`p32`), the seed as Triton passes the Python `int`: an
`int32` in `[-2^31, 2^31)`, an `int64` in `[2^31, 2^63)` and `[-2^63, -2^31)`, a
`uint64` in `[2^63, 2^64)`, each the integer modulo `2^w`. At `seed = 1` Triton passes the
compile-time constant `1`; `tl.rand` makes it a tensor (`tl.to_tensor`, an `int32`) before any use,
so the model is the `int32` one. At `n = 1` Triton passes `n_elements` as the compile-time constant
`1`; its only use, `offsets < n_elements`, makes it an `int32` tensor before comparing
(`TVA.Triton.Promote`), so the model is the same. -/
def model (d : Inst) : Kernel :=
  if -2 ^ 31 ≤ d.seed ∧ d.seed < 2 ^ 31 then modelAt d (IntV.ofInt d.seed : I32)
  else if d.seed < 2 ^ 63 then modelAt d (IntV.ofInt d.seed : I64)
  else modelAt d (IntV.ofInt d.seed : U64)

/-- The host's launch (source lines 138–139): a 1-D grid of `triton.cdiv(n_elements, BLOCK_SIZE)`
programs. The source's only block shape is `[BLOCK_SIZE]` (`tl.arange`, `offsets`, `mask`, `x`,
`random`, `x_keep`, the quotient, `output`). `enable_fp_fusion` is Triton's default (`true`);
`grid1` keeps the defaults `num_ctas = 1`, `launch_pdl = False` and `debug = false`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.n + d.BLOCK_SIZE - 1) / d.BLOCK_SIZE) [[d.BLOCK_SIZE]] (fusion := true)

/-- The inputs covered: the two buffers are valid (aligned, inside `[0, 2^64)`, and every byte
present) in the initial memory. Any `float32` values of `x`, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.output.Valid init

/-! ## The specification

The reference is computed from the host's inputs (`seed`, `p`, and `x[j]`) and the element's index
`j`, by the definition of Triton's generator, not by the kernel's indexing. -/

/-- The first word of the Philox key: `seed mod 2^32`. -/
def key0 (seed : Int) : BitVec 32 := BitVec.ofInt 32 seed

/-- The second word of the Philox key: the high 32 bits of `seed mod 2^64`. -/
def key1 (seed : Int) : BitVec 32 := BitVec.ofNat 32 ((seed % 2 ^ 64).toNat / 2 ^ 32)

/-- Element `j`'s 32 random bits: the first word of Philox 4x32 with 10 rounds (`Philox.impl`) on
the counter `(j mod 2^32, 0, 0, 0)` and the key `(key0 seed, key1 seed)`. -/
def rbits (seed : Int) (j : Nat) : BitVec 32 :=
  (Philox.impl 10 (BitVec.ofNat 32 j) 0 0 0 (key0 seed) (key1 seed)).1

/-- The non-negative integer `uint_to_uniform_float` scales: the bits `r` read as an `int32`,
complemented when negative, so `r` below `2^31` and `2^32 - 1 - r` from `2^31` on. -/
def rint (seed : Int) (j : Nat) : Nat :=
  let r := (rbits seed j).toNat
  if r < 2 ^ 31 then r else 2 ^ 32 - 1 - r

/-- Element `j`'s random number: `rint` rounded to `float32` (nearest even), times the `float32`
constant `4.6566127342e-10` (`Philox.scale32`), one IEEE product. In `[0, 1)`. -/
def uRef (seed : Int) (j : Nat) : F32 :=
  FloatV.mul (FloatV.ofRat (rint seed j : ℚ)) Philox.scale32

/-- Element `j` is kept: its random number is above the kernel's `p` (the ordered comparison). -/
def keep (d : Inst) (j : Nat) : Bool := FloatV.lt (p32 d) (uRef d.seed j)

/-- The divisor: the IEEE `float32` difference `1 - p` of the kernel's `p` (round to nearest even;
exact when `p ≥ 1/2`; `1` at `p = ±0`, `0` at `p = 1`, where no element is kept). -/
def dRef (d : Inst) : F32 := FloatV.sub one32 (p32 d)

/-- What the launch leaves: for each `j < n`, element `j` of `x` can be read in `init` (value `a`)
and element `j` of `output` in `fin` (value `c`); if `j` is not kept, `c` is `+0` (bit pattern
`0`); if `j` is kept and `a` is finite with `|a / dRef| ≤ 2^126`, then `c` is finite and within
`2^-22·|a/dRef| + 2^-148` of the exact `a / dRef`. Every byte outside `output`'s first `n`
elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ j : Nat, j < d.n → ∃ a c : F32,
      init.read .f32 (d.x.ptr j).addr = some a ∧
      fin.read .f32 (d.output.ptr j).addr = some c ∧
      (keep d j = false → c = FloatV.ofPat 0) ∧
      (keep d j = true → a.isFinite = true → |a.toRat / (dRef d).toRat| ≤ 2 ^ 126 →
        c.isFinite = true ∧
        |c.toRat - a.toRat / (dRef d).toRat| ≤
          2 ^ (-22 : ℤ) * |a.toRat / (dRef d).toRat| + 2 ^ (-148 : ℤ))) ∧
    ∀ z, z ∉ d.output.elems {j | j < d.n} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_seeded_dropout` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The double `0.5` (`0x3FE0000000000000`), the `p` of the file's launches. -/
def half : F64 := FloatV.ofPat 0x3FE0000000000000

theorem half_facts : (FloatV.hostF32 half).isFinite = true ∧ 0 ≤ (FloatV.hostF32 half).toRat ∧
    (FloatV.hostF32 half).toRat ≤ 1 := by
  refine ⟨by decide +kernel, ?_, ?_⟩
  · have : (0 : ℚ) ≤ (FloatV.hostF32 (FloatV.ofPat 0x3FE0000000000000 : F64)).toRat := by
      decide +kernel
    exact this
  · have : (FloatV.hostF32 (FloatV.ofPat 0x3FE0000000000000 : F64)).toRat ≤ (1 : ℚ) := by
      decide +kernel
    exact this

/-- The host's launch on `n` elements with `BLOCK_SIZE = 2^e`, `p = 0.5` and the seed `s`: `x` and
a fresh `output` of `n` elements each, at the 16-byte-aligned addresses `2^34` and `2·2^34`. -/
def mk (n e : Nat) (s : Int) : Inst where
  n := n
  BLOCK_SIZE := 2 ^ e
  p := half
  seed := s
  x := ⟨2 ^ 34, n⟩
  output := ⟨2 * 2 ^ 34, n⟩

/-- Two buffers whose byte ranges do not meet have first `n` elements that do not meet. -/
theorem elems_disjoint_of_lt {b c : Buf .f32} {n : Nat} (hb : n ≤ b.len) (hc : n ≤ c.len)
    (h : b.base + b.len * 4 ≤ c.base ∨ c.base + c.len * 4 ≤ b.base) :
    Disjoint (b.elems {j | j < n}) (c.elems {j | j < n}) := by
  refine Buf.elems_disjoint_of b c ?_ (fun j hj => by simp only [Set.mem_ofPred_eq] at hj; omega)
    (fun j hj => by simp only [Set.mem_ofPred_eq] at hj; omega)
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq, DType.bytes] at hx hy
  omega

/-- Every host launch with `p = 0.5` on `n < 2^31` elements, a block size `2^e ≤ 2^20` and a
seed in `[-2^63, 2^64)` is covered. -/
theorem mk_ok {n e : Nat} {s : Int} (hn : n < 2 ^ 31) (he : e ≤ 20) (hs0 : -2 ^ 63 ≤ s)
    (hs1 : s < 2 ^ 64) : (mk n e s).ok where
  block_pow2 := ⟨e, rfl⟩
  block_le := Nat.pow_le_pow_right (by norm_num) he
  n_lt := hn
  p_fin := half_facts.1
  p_ge := half_facts.2.1
  p_le := half_facts.2.2
  seed_ge := hs0
  seed_lt := hs1
  x_len := le_refl _
  output_len := le_refl _
  output_x := elems_disjoint_of_lt (le_refl _) (le_refl _) (Or.inr (by
    simp only [mk]; push_cast; omega))

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 10 10 123, testMem, mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num),
    ?_, ?_⟩
  all_goals
    refine ⟨by norm_num [mk], by norm_num [mk, DType.bytes], by norm_num [mk, DType.bytes],
      fun _ _ => by simp [testMem]⟩

/-! ## Coverage of the file's launches (`SETUP.md`)

`seeded_dropout(x, p=0.5, seed=…)` with `BLOCK_SIZE = 1024 = 2^10` on the `10` elements of line
143: the seed `123` (lines 145 and 146, the same launch twice) and the seed `512` (line 147). -/

def setup_seed123 : Inst := mk 10 10 123
theorem setup_seed123_ok : setup_seed123.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_seed512 : Inst := mk 10 10 512
theorem setup_seed512_ok : setup_seed512.ok :=
  mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.SeededDropout
