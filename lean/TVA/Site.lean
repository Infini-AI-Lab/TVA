import TVA.Core

/-!
# Triton's sites: what a Triton program may leave open

The semantics (`TVA.Launch`) is written for any signature `Sig` (`TVA.Program`): the kinds of
value a program cannot predict, and what each allows. This module gives Triton's: the `Sig`
instance that every `tl.*` operation, every model and every claim uses.

* `Site`: where a `tl.*` operation leaves a value to the device or the compiler. `Site.other P`
  is any value of type `β` satisfying `P`.
* `Upd d`: the update of one element of dtype `d` by an atomic. `Upd.other upd` is any new value
  `upd old` allows.

`Prog.plat` is an operation the platform defines (`PlatQ`: `tl.exp`, `/`, `tl.dot`, a conversion
into fp8): read the environment, then any result its relation allows on the platform.
-/
namespace TVA

/-- A site: a value a Triton program cannot predict, and what it may be. -/
inductive Site : Type 1 where
  /-- Any value of type `β` satisfying `P`. -/
  | other (β : Type) (P : β → Prop)

/-- The type of a site's value. -/
@[reducible] def Site.Out : Site → Type
  | .other β _ => β

/-- The values a site allows. -/
@[reducible] def Site.Allows : (s : Site) → s.Out → Prop
  | .other _ P => P

/-- The update an atomic makes to one element of dtype `d`. -/
inductive Upd (d : DType) : Type 1 where
  /-- Any new value `upd old` allows. -/
  | other (upd : d.Val → d.Val → Prop)

/-- The new values an update allows, given the old one. -/
@[reducible] def Upd.Allows {d : DType} : Upd d → d.Val → d.Val → Prop
  | .other upd => upd

/-- **Triton's signature**: its sites and its atomics' updates. -/
instance Site.sig : Sig where
  Site := Site
  Out := Site.Out
  Allows := Site.Allows
  Upd := Upd
  UpdAllows := Upd.Allows

@[simp] theorem Site.allows_other {β : Type} {P : β → Prop} {b : β} :
    Sig.Allows (Site.other β P) b ↔ P b := Iff.rfl

@[simp] theorem Upd.allows_other {d : DType} {upd : d.Val → d.Val → Prop} {old new : d.Val} :
    Sig.UpdAllows (Upd.other upd) old new ↔ upd old new := Iff.rfl

namespace Prog

/-- An operation the platform defines, on concrete operands (`PlatQ`: `tl.exp`, `/`, `tl.dot`, a
conversion into fp8): read the environment, then any result `q.Rel` allows on its platform;
undefined behaviour when there is none. -/
def plat {α : Type} (q : PlatQ) (k : q.Out → Prog α) : Prog α :=
  env fun e => pick (Site.other _ (q.Rel e.P)) k

@[simp] theorem plat_bind {α β : Type} (q : PlatQ) (k : q.Out → Prog α) (f : α → Prog β) :
    (plat q k).bind f = plat q (fun r => (k r).bind f) := rfl

end Prog

end TVA
