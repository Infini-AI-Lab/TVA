import TVA.Triton.Core

/-!
# Memory: `tl.load`, `tl.store`, atomics

Accesses through tensors of pointers (`Ptr d`, `TVA.Memory`). Each is one `Prog` effect
(`Prog.load`, `Prog.store`, `Prog.rmw`), whose meaning (what it reads, what makes it undefined,
which accesses race) is `TVA.Launch`'s. A lane's address is `some a` when the mask enables it and
`none` otherwise (`tl.addrs`); a disabled lane accesses nothing.

The performance-only arguments (`cache_modifier`, `eviction_policy`, `volatile`) change no result
of a well-behaved launch and are omitted. A stored value, an atomic's operand and `other=` are
converted to the pointer's dtype by the source implicitly, and by the model explicitly
(`TVA.Triton.Convert`), after a Python number has become a tensor of its own dtype.

Block pointers are in `TVA.Triton.BlockPtr`, tensor descriptors in `TVA.Triton.Desc`.

## Atomics

A tensor atomic is one atomic read-modify-write per enabled lane (`Prog.rmw`), in an order the
program does not control, each returning its element's old value; disabled lanes return
unspecified values. The defaults are `sem="acq_rel"` and `scope="gpu"`.

Triton accepts `atomic_add` on `int32`, `uint32`, `int64`, `uint64`, `float16`, `bfloat16`,
`float32`, `float64`; `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, `atomic_xor` and
`atomic_xchg` on the 32- and 64-bit integers and `float32`, `float64`; `atomic_cas` on every 16-,
32- and 64-bit element (`semantic.py:1257-1290`). It rejects other element types on the host, so
the update relations' meanings on them are never used. `atomic_and`/`or`/`xor` on floats (the
integer operation on the bit patterns) are not modelled: `bitUpd` has no result there, so the
model's atomic is undefined behaviour. `atomic_cas` converts nothing: its `cmp` and `val` have the
cell's dtype, or the kernel is rejected.

**An atomic whose result is unused.** A `tl.atomic_*` call whose returned value no operation that
runs reads (an expression statement, or a value bound only to names nothing that runs reads) is
written `tl.atomic_*_unused`: the same atomic with order `MemSem.unused sem scope` (`acq_rel` at
`gpu` or `sys` scope, whatever the source's `sem`), returning nothing. Two such atomics on the same
bytes from different programs then do not race (`TVA.Program`, `MemSem.unused`, explains why this
is sound). A launch in which a program *uses* an atomic's result on a cell that another program, or
another atomic operation of the same program, also accesses is outside every claim (`TVA.Launch`).
-/
namespace TVA

namespace tl

variable {s : Shape} {α : Type}

/-! ## Addresses -/

/-- The byte address of each lane the mask enables (all lanes without a mask), `none` for the
others. -/
def addrs {d : DType} (ptrs : Tensor s (Ptr d)) (mask : Option (Tensor s Bool)) :
    Idx s → Option Int :=
  fun i => match mask with
    | none => some (ptrs.get i).addr
    | some m => if m.get i then some (ptrs.get i).addr else none

end tl

namespace Buf

variable {d : DType} {s : Shape}

/-- The elements an access touches: `g i` for the enabled lanes (`en i`). -/
def img (g : Idx s → Nat) (en : Idx s → Bool) : Set Nat := {j | ∃ i, en i = true ∧ g i = j}

/-- An access's lane addresses `addrs` (`tl.addrs ptrs mask`) are elements of `b`: lane `i`
addresses element `g i` when `en i`, and nothing otherwise. Most loads and stores are of this
shape, and the footprint is then `b.elems (img g en)`. -/
def Addrs (b : Buf d) (addrs : Idx s → Option Int) (g : Idx s → Nat) (en : Idx s → Bool) : Prop :=
  ∀ i, addrs i = if en i then some (b.ptr (g i : Int)).addr else none

/-- **From lanes to `Addrs`.** Show, lane by lane, that the mask is `en` and that an enabled lane
points at element `g i`. -/
theorem addrs_of_get (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Tensor s Bool}
    {g : Idx s → Nat} {en : Idx s → Bool} (hm : ∀ i, mask.get i = en i)
    (hp : ∀ i, en i = true → ptrs.get i = b.ptr (g i : Int)) :
    b.Addrs (tl.addrs ptrs (some mask)) g en := by
  intro i
  simp only [tl.addrs, hm i]
  split_ifs with he
  · rw [hp i he]
  · rfl

/-- **From lanes to `Addrs`, without a mask**: every lane is enabled. -/
theorem addrs_of_get_all (b : Buf d) {ptrs : Tensor s (Ptr d)} {g : Idx s → Nat}
    (hp : ∀ i, ptrs.get i = b.ptr (g i : Int)) : b.Addrs (tl.addrs ptrs none) g (fun _ => true) := by
  intro i
  simp only [tl.addrs, hp i, ite_true]

end Buf

/-- `padding_option` of a block-pointer or descriptor load: what a lane outside the bounds holds.
`undef` is an unspecified value (block pointers' default `""`), `zero` the zero bit pattern, `nan`
a NaN (float dtypes only; Triton rejects it for integers). -/
inductive Padding where
  /-- An unspecified value. -/
  | undef
  /-- The zero bit pattern. -/
  | zero
  /-- A NaN. -/
  | nan
  deriving DecidableEq, Repr

/-- A lane outside the bounds holds what the padding says. -/
def Padding.Holds {d : DType} : Padding → d.Val → Prop
  | .undef, _ => True
  | .zero, x => x = d.ofBits 0
  | .nan, x => d.IsNaN x

namespace tl

variable {s : Shape} {α : Type}

/-! ## Loads and stores -/

/-- `tl.load(ptrs, mask, other)`: each enabled lane reads its element. Disabled lanes are not
accessed; they hold `other` when it is given and an unspecified value otherwise ("If `other` is
`None`, the masked-out value is undefined"). `other` is the source's `other` converted to the
pointer's dtype (`semantic.py:1043-1045`). -/
def load {d : DType} (ptrs : Tensor s (Ptr d)) (mask : Option (Tensor s Bool) := none)
    (other : Option (Tensor s d.Val) := none) : Prog (Tensor s d.Val) :=
  .load d s (addrs ptrs mask) fun v =>
    .ret (match mask, other with
      | some m, some o => Tensor.where_ m v o
      | _, _ => v)

/-- `tl.load(p)` of a scalar pointer. -/
def load1 {d : DType} (p : Ptr d) : Prog d.Val :=
  .load d [] (fun _ => some p.addr) fun v => .ret v.item

/-- `tl.store(ptrs, vals, mask)`: each enabled lane writes its value. The model passes values
already of the pointer's dtype (the source's implicit `vals.to(ptr.dtype.element_ty)` with the
default rounding, `semantic.py:1205-1243`, is written explicitly). -/
def store {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) : Prog Unit :=
  .store d s (addrs ptrs mask) vals (.ret ())

/-- `tl.store(p, v)` to a scalar pointer. -/
def store1 {d : DType} (p : Ptr d) (v : d.Val) : Prog Unit :=
  .store d [] (fun _ => some p.addr) (Tensor.scalar v) (.ret ())

/-- The lanes outside the bounds (`inb i = false`) of a loaded block hold what the padding says;
the others keep their loaded value. -/
def pad {d : DType} (p : Padding) (inb : Idx s → Bool) (v : Tensor s d.Val) :
    Prog (Tensor s d.Val) :=
  .pick (fun r : Tensor s d.Val => ∀ i, if inb i then r.get i = v.get i else p.Holds (r.get i)) .ret

/-! ## Atomics -/

/-- The read-modify-writes of the lanes `order`, in that order, collecting the old values. -/
def rmwSeq {d : DType} (ptrs : Tensor s (Ptr d)) (upd : Idx s → d.Val → d.Val → Prop)
    (sem : MemSem) (scope : Scope) : List (Idx s) → (Idx s → d.Val) → Prog (Idx s → d.Val)
  | [], acc => .ret acc
  | i :: rest, acc => .rmw d (ptrs.get i).addr (upd i) sem scope fun old =>
      rmwSeq ptrs upd sem scope rest (Function.update acc i old)

/-- A tensor atomic whose lane `i` updates its element by the relation `upd i old new`: the enabled
lanes in some order, each returning its old value; disabled lanes return unspecified values. -/
def atomicWith {d : DType} (ptrs : Tensor s (Ptr d)) (upd : Idx s → d.Val → d.Val → Prop)
    (mask : Option (Tensor s Bool)) (sem : MemSem) (scope : Scope) : Prog (Tensor s d.Val) :=
  let enabled := (Idx.all s).filter (fun i => (addrs ptrs mask i).isSome)
  .pick (fun (_ : Idx s → d.Val) => True) fun junk =>
  .pick (fun (order : List (Idx s)) => order.Perm enabled) fun order => do
    let old ← rmwSeq ptrs upd sem scope order junk
    return ⟨old⟩

/-- The new value `n` of an `atomic_add` of `v` to a cell holding `o`: integers wrap; floats as the
platform's memory system rounds (`Platform.atomicAdd`). -/
def addUpd (e : Env) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = o + v | .i16, v, o, n => n = o + v | .i32, v, o, n => n = o + v
  | .i64, v, o, n => n = o + v
  | .u8, v, o, n => n = o + v | .u16, v, o, n => n = o + v | .u32, v, o, n => n = o + v
  | .u64, v, o, n => n = o + v
  | .f16, v, o, n => e.P.atomicAdd _ o v n | .bf16, v, o, n => e.P.atomicAdd _ o v n
  | .f32, v, o, n => e.P.atomicAdd _ o v n | .f64, v, o, n => e.P.atomicAdd _ o v n
  | _, _, _, _ => False

/-- The new value `n` of a float `atomic_max` (`isMax`) or `atomic_min` of `v` on a cell holding
`o`: one of the two, and, when neither is NaN, not below (above) either. Which zero wins between
`-0.0` and `+0.0`, and which value with a NaN operand, are left open (Triton implements them with
integer atomics on the bit pattern, `semantic.py:1292-1373`). -/
def fBound {f : Format} (isMax : Bool) (o v n : FloatV f) : Prop :=
  (n = o ∨ n = v) ∧ (o.isNaN = false → v.isNaN = false →
    if isMax then o.le n = true ∧ v.le n = true else n.le o = true ∧ n.le v = true)

/-- The new value of `atomic_max` (`isMax`) or `atomic_min`: integers compare as their signedness
says, floats as `fBound`. -/
def maxUpd (isMax : Bool) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = if isMax then o.max v else o.min v
  | .i16, v, o, n => n = if isMax then o.max v else o.min v
  | .i32, v, o, n => n = if isMax then o.max v else o.min v
  | .i64, v, o, n => n = if isMax then o.max v else o.min v
  | .u8, v, o, n => n = if isMax then o.max v else o.min v
  | .u16, v, o, n => n = if isMax then o.max v else o.min v
  | .u32, v, o, n => n = if isMax then o.max v else o.min v
  | .u64, v, o, n => n = if isMax then o.max v else o.min v
  | .f16, v, o, n => fBound isMax o v n | .bf16, v, o, n => fBound isMax o v n
  | .f32, v, o, n => fBound isMax o v n | .f64, v, o, n => fBound isMax o v n
  | _, _, _, _ => False

/-- The new value of a bitwise atomic (`op` on the bit vectors), on integers only. -/
def bitUpd (op : ∀ {sg : Bool} {w : Nat}, IntV sg w → IntV sg w → IntV sg w) :
    (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = op o v | .i16, v, o, n => n = op o v | .i32, v, o, n => n = op o v
  | .i64, v, o, n => n = op o v
  | .u8, v, o, n => n = op o v | .u16, v, o, n => n = op o v | .u32, v, o, n => n = op o v
  | .u64, v, o, n => n = op o v
  | _, _, _, _ => False

/-- `tl.atomic_add(ptrs, vals, mask, sem, scope)`: adds `vals` lane by lane, returns the old
values. -/
def atomic_add {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  .env fun e => atomicWith ptrs (fun i => addUpd e d (vals.get i)) mask sem scope

/-- `tl.atomic_xchg(ptrs, vals, mask, sem, scope)`: writes `vals`, returns the old values. -/
def atomic_xchg {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i _ n => n = vals.get i) mask sem scope

/-- `tl.atomic_cas(ptrs, cmp, vals, sem, scope)`: the new value is `val` when the old bit pattern
equals `cmp`'s, the old value otherwise; returns the old values. -/
def atomic_cas {d : DType} (ptrs : Tensor s (Ptr d)) (cmp vals : Tensor s d.Val)
    (sem : MemSem := .acqRel) (scope : Scope := .gpu) : Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i o n => n = if o = cmp.get i then vals.get i else o) none sem scope

/-- `tl.atomic_max(ptrs, vals, mask, sem, scope)` (`maxUpd`). -/
def atomic_max {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => maxUpd true d (vals.get i)) mask sem scope

/-- `tl.atomic_min(ptrs, vals, mask, sem, scope)` (`maxUpd`). -/
def atomic_min {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => maxUpd false d (vals.get i)) mask sem scope

/-- `tl.atomic_and(ptrs, vals, mask, sem, scope)` on integers. -/
def atomic_and {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => bitUpd IntV.and d (vals.get i)) mask sem scope

/-- `tl.atomic_or(ptrs, vals, mask, sem, scope)` on integers. -/
def atomic_or {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => bitUpd IntV.or d (vals.get i)) mask sem scope

/-- `tl.atomic_xor(ptrs, vals, mask, sem, scope)` on integers. -/
def atomic_xor {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => bitUpd IntV.xor d (vals.get i)) mask sem scope

/-! ### Atomics whose result is unused -/

/-- `tl.atomic_add(…)` whose result is unused (`MemSem.unused`). -/
def atomic_add_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_add ptrs vals mask (sem.unused scope) scope
  return ()

/-- `tl.atomic_xchg(…)` whose result is unused (`MemSem.unused`). -/
def atomic_xchg_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_xchg ptrs vals mask (sem.unused scope) scope
  return ()

/-- `tl.atomic_cas(…)` whose result is unused (`MemSem.unused`). -/
def atomic_cas_unused {d : DType} (ptrs : Tensor s (Ptr d)) (cmp vals : Tensor s d.Val)
    (sem : MemSem := .acqRel) (scope : Scope := .gpu) : Prog Unit := do
  let _ ← atomic_cas ptrs cmp vals (sem.unused scope) scope
  return ()

/-- `tl.atomic_max(…)` whose result is unused (`MemSem.unused`). -/
def atomic_max_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_max ptrs vals mask (sem.unused scope) scope
  return ()

/-- `tl.atomic_min(…)` whose result is unused (`MemSem.unused`). -/
def atomic_min_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_min ptrs vals mask (sem.unused scope) scope
  return ()

/-- `tl.atomic_and(…)` whose result is unused (`MemSem.unused`). -/
def atomic_and_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_and ptrs vals mask (sem.unused scope) scope
  return ()

/-- `tl.atomic_or(…)` whose result is unused (`MemSem.unused`). -/
def atomic_or_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_or ptrs vals mask (sem.unused scope) scope
  return ()

/-- `tl.atomic_xor(…)` whose result is unused (`MemSem.unused`). -/
def atomic_xor_unused {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog Unit := do
  let _ ← atomic_xor ptrs vals mask (sem.unused scope) scope
  return ()

end tl

/-- At `gpu` scope an unused atomic's order is `acq_rel`, whatever the source's. -/
@[simp] theorem MemSem.unused_gpu (sem : MemSem) : sem.unused .gpu = .acqRel := rfl

/-- At `sys` scope an unused atomic's order is `acq_rel`, whatever the source's. -/
@[simp] theorem MemSem.unused_sys (sem : MemSem) : sem.unused .sys = .acqRel := rfl

/-- At `cta` scope an unused atomic keeps the source's order. -/
@[simp] theorem MemSem.unused_cta (sem : MemSem) : sem.unused .cta = sem := rfl

end TVA
