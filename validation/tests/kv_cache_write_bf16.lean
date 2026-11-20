import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `kv_cache_write_bf16` run by `execLaunch` on three small instances
(`bfloat16` inputs and caches in vLLM's `flash` layout; the same in the `triton_attn` layout, with a
tile wider than a token's row; `float8e4nv` inputs and caches in the `triton_attn` layout). The
configuration with `bfloat16` inputs and an fp8 cache divides by the scales (a platform operation)
and is not run here. Not a test against the device (phase 4): it checks that `exec` runs the model
end to end, and that the final memory is what `Post` says: each non-padding token's key and value
rows copied, bit for bit, to its slot of the key and value caches, and every other element (the
rest of the shared cache allocation, the inputs) unchanged, with no byte added. -/

open TVA TVA.Validation Claims.KvCacheWrite

/-- A platform that relates nothing: these configurations use no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- One case: the configuration, the cache layout, the shapes, and the slot of each token. -/
structure Case where
  mode : Mode
  layout : Layout
  T : Nat
  nh : Nat
  hs : Nat
  bs : Nat
  nb : Nat
  TILE : Nat
  slots : List Int

def Case.inst (c : Case) : Inst :=
  Claims.KvCacheWrite.mk c.mode c.layout c.T c.nh c.hs c.bs c.nb c.TILE
def Case.n (c : Case) : Nat := c.nh * c.hs
/-- The elements of the `qkv` allocation `key` and `value` are views of (`[T, 3, n]`), and the
`2 n` elements after it that `value`'s storage (from its first element) reaches. -/
def Case.qlen (c : Case) : Nat := 3 * c.T * c.n + 2 * c.n
/-- The elements of the one cache allocation (`[nb, 2, bs, n]` or `[nb, nh, bs, 2 hs]`). -/
def Case.clen (c : Case) : Nat := c.nb * (2 * c.bs * c.n)

/-- The bit pattern of element `j` of the `qkv` allocation: distinct finite values
(`bfloat16`: `2 + j/128`; `float8e4nv`: the positive patterns `1, 2, …`, below `0x7f`, the NaN). -/
def inBits (m : Mode) (j : Nat) : Nat :=
  match m with
  | .fp8in => j + 1
  | _ => 0x4000 + j

/-- The sentinel in every cache element: those no token's slot names must keep it. -/
def sentinel (m : Mode) : Nat :=
  match m with
  | .bf16 => 0xBEEF
  | _ => 0xF0

/-- Element `(b, r, h, e)` of the key cache, as an element of the allocation, from vLLM's layout
(`make_caches`), and the value cache's offset in it. -/
def Case.kv (c : Case) (b r h e : Nat) : Nat :=
  match c.layout with
  | .flash => ((b * 2 + 0) * c.bs + r) * c.n + h * c.hs + e
  | .triton_attn => ((b * c.nh + h) * c.bs + r) * (2 * c.hs) + e
def Case.vOff (c : Case) : Nat :=
  match c.layout with
  | .flash => c.bs * c.n
  | .triton_attn => c.hs

def Case.init (c : Case) : CMem :=
  let d := c.inst
  -- the `qkv` allocation (`key` is `qkv[:, 1]`, so its element `j - n` is the allocation's `j`)
  let m := CMem.ofElems ∅ d.mode.kd ((List.range c.qlen).map fun j : Nat =>
    ((d.key.ptr ((j : Int) - c.n)).addr, d.mode.kd.ofBits (inBits d.mode j)))
  let m := CMem.ofElems m .i64 ((List.range c.T).map fun t : Nat =>
    ((d.slot_mapping.ptr t).addr, (IntV.ofInt (c.slots.getD t (-1)) : I64)))
  CMem.ofElems m d.mode.cd ((List.range c.clen).map fun j : Nat =>
    ((d.key_cache.ptr j).addr, d.mode.cd.ofBits (sentinel d.mode)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 1000 (model c.inst) (launch c.inst) c.init

/-- What `Post` says the allocation's element `j` holds, as `(element of the cache, element of
`qkv` it holds)` pairs: for token `t` with slot `s ≥ 0`, head `h`, component `e`, the key cache's
element `(s / bs, s % bs, h, e)` holds `key[t, h, e]` (`qkv[t, 1, h, e]`), the value cache's
holds `value[t, h, e]` (`qkv[t, 2, h, e]`). -/
def Case.moves (c : Case) : List (Nat × Nat) :=
  (List.range c.T).flatMap fun t : Nat =>
    let s := c.slots.getD t (-1)
    if s < 0 then [] else
      (List.range c.nh).flatMap fun h : Nat => (List.range c.hs).flatMap fun e : Nat =>
        let k := c.kv (s.toNat / c.bs) (s.toNat % c.bs) h e
        let src := t * (3 * c.n) + h * c.hs + e
        [(k, c.n + src), (c.vOff + k, 2 * c.n + src)]

/-- The bits `Post` says element `j` of the cache allocation holds; the sentinel if none. -/
def Case.expect (c : Case) (j : Nat) : Nat :=
  match c.moves.find? (·.1 == j) with
  | some (_, q) => inBits c.mode q
  | none => sentinel c.mode

/-- The final memory against `Post`: every element of the cache allocation (written or not), the
inputs unchanged, and no byte added. -/
def Case.agree (c : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m0 := CMem.toMem c.init
  let m := CMem.toMem cf.mem
  let badCache := (List.range c.clen).filter fun j : Nat =>
    (m.read d.mode.cd (d.key_cache.ptr j).addr).map d.mode.cd.toBits != some (c.expect j)
  let qkvSame := (List.range c.qlen).all fun j : Nat =>
    let a := (d.key.ptr ((j : Int) - c.n)).addr
    (m.read d.mode.kd a).map d.mode.kd.toBits == (m0.read d.mode.kd a).map d.mode.kd.toBits
  let slotsSame := (List.range c.T).all fun t : Nat =>
    (m.read .i64 (d.slot_mapping.ptr t).addr).map DType.i64.toBits ==
      (m0.read .i64 (d.slot_mapping.ptr t).addr).map DType.i64.toBits
  if !badCache.isEmpty then some s!"cache allocation elements {badCache} differ from Post"
  else if !(qkvSame && slotsSame) then some "an input changed"
  else if cf.mem.size != c.init.size then some "bytes outside the buffers were written"
  else none

/-- `Pre`'s slot conditions, and the moves' targets inside the views (as `Inst.ok` makes them). -/
def Case.covered (c : Case) : Bool :=
  c.slots.length == c.T &&
  c.slots.all (fun s => s < (c.nb * c.bs : Nat)) &&
  ((c.slots.filter (0 ≤ ·)).eraseDups.length == (c.slots.filter (0 ≤ ·)).length) &&
  c.moves.all (fun p => p.1 < c.clen && p.2 < c.qlen) &&
  (c.moves.map (·.1)).eraseDups.length == c.moves.length

/-- `bfloat16`, `flash` layout; token 1 is padding; two tiles of four per token. -/
def flashBf16 : Case :=
  { mode := .bf16, layout := .flash, T := 3, nh := 2, hs := 4, bs := 4, nb := 2, TILE := 4,
    slots := [5, -1, 2] }

/-- `bfloat16`, `triton_attn` layout (keys and values interleaved by `head_size`); one tile of 8
for a row of 6 (two lanes masked); token 2 is padding. -/
def tritonBf16 : Case :=
  { mode := .bf16, layout := .triton_attn, T := 3, nh := 3, hs := 2, bs := 2, nb := 3, TILE := 8,
    slots := [4, 1, -1] }

/-- `float8e4nv` inputs and caches (`FP8_KV_CACHE = True`, stored as they are), `triton_attn`
layout; two tiles of four per token. -/
def tritonFp8in : Case :=
  { mode := .fp8in, layout := .triton_attn, T := 2, nh := 2, hs := 4, bs := 2, nb := 2, TILE := 4,
    slots := [3, 0] }

#guard flashBf16.covered && tritonBf16.covered && tritonFp8in.covered
-- `inBits` stays below the fp8 NaN `0x7f` and away from the sentinel.
#guard tritonFp8in.qlen < 0x7e

def verdictFlash : Verdict := Verdict.ofLaunch flashBf16.agree flashBf16.run
def verdictTriton : Verdict := Verdict.ofLaunch tritonBf16.agree tritonBf16.run
def verdictFp8in : Verdict := Verdict.ofLaunch tritonFp8in.agree tritonFp8in.run

#eval IO.println (verdictFlash.line "kv_cache_write_bf16/bf16_flash")
#eval IO.println (verdictTriton.line "kv_cache_write_bf16/bf16_triton_attn")
#eval IO.println (verdictFp8in.line "kv_cache_write_bf16/fp8in_triton_attn")
#guard verdictFlash == .certified
#guard verdictTriton == .certified
#guard verdictFp8in == .certified

-- A wrong expectation is caught: `agree` is not vacuous (the two tokens' slots swapped; the
-- `flash` expectation on the `triton_attn` run).
def swapped : Case := { flashBf16 with slots := [2, -1, 5] }
#guard !(Verdict.ofLaunch swapped.agree flashBf16.run == .certified)
def wrongLayout : Case := { tritonBf16 with layout := .flash }
#guard !(Verdict.ofLaunch wrongLayout.agree tritonBf16.run == .certified)
