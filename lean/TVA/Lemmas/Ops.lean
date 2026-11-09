import TVA.Lemmas.Buf
import TVA.Lemmas.Safe
import TVA.Lang.Promote

/-!
# Proof rules for the `tl.*` operations

One rule per operation, in the shape `Safe … (op >>= k)`: the premises are what the operation
needs (valid addresses, footprint, a platform result existing), and the last premise
is the rest of the program, for every result the operation may produce. A per-program proof is
then a walk down the model's `do` block, applying the rule of each line.

* Memory: `safe_load` (result: `LoadRes`), `safe_store` (for distinct lanes; result
  `StoreRes`), `safe_store_any` (any lanes), `safe_store1` (`tl.store1`, a store through a scalar
  pointer: `tl.store` of the scalar tensors, `tl.store1_eq`).
* Exact float operations, up to NaN payload: `safe_upToNaN` (and `safe_exact` when no NaN
  arises); `safe_add`, `safe_sub`, `safe_mul`, `safe_fma`, `safe_div_rn`, `safe_sqrt_rn`,
  `safe_neg`, `safe_abs`, `safe_cvt`; `safe_maximum`, `safe_minimum` (`MaxRel`).
* Conversions of a dtype: `safe_convert` (`tl.convert`: `x.to(…)` with the default rounding and
  every implicit conversion of an operand, each element as `ConvRel` allows; library issue L113).
* Fusable and platform operations: `safe_mul_add`, `safe_addc`, `safe_subc`, `safe_div`, `safe_math`, `safe_cvt8`, `safe_ftoi`, `safe_dot`, `safe_sum`/`sum1`/`sum0` (IEEE nodes),
  `safe_sumc`/`sum1c`/`sum0c` (fusable nodes), the `…With` forms, `safe_max1`, `safe_dot_add`,
  `safe_sum_bcast_mul`, `safe_maxAll`, `safe_max0`; index reductions: `safe_argmaxAll`,
  `safe_argmax1`, `safe_argmax0`, whose results `tl.ArgRes` describes: `tl.ArgRes.index_lt` (the
  index is in range), `tl.ArgRes.value` (the value is an element), `tl.ArgRes.tree_of_noNaN`
  (without NaN, a tree's pair), with `Tree.eval_argCombine` (the result is a leaf) and
  `Tree.eval_argCombine_first` (with `tie_break_left` on integers, the first index of the
  maximum or minimum; integers have no NaN, `IntV.cmpKey_nan`).
* Integers and hints: `safe_idiv`, `safe_irem`, `safe_shl`, `safe_ashr`, `safe_lshr`, `safe_assume`,
  `safe_multiple_of(1)`, `safe_max_contiguous`, `safe_max_constancy`,
  `safe_max_contiguous_multiple_of` (the pair on one value), and the any-rank `…N` forms;
  `safe_reshape_any`.
* Control: `safe_ret`, `safe_last`, `safe_forN`, `safe_forN_bind`, `safe_forRange`, `safe_while`,
  with per-iteration write footprints (`iterW`).
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

theorem safe_ret {a : α} (h : Q a m) : Safe E R W Q m (pure a) := .ret h

/-- The last line of a program: apply the line's rule with `pure` as the rest. -/
theorem safe_last {p : Prog α} (h : Safe E R W Q m (p >>= pure)) : Safe E R W Q m p := by
  rwa [bind_eq, show (pure : α → Prog α) = Prog.ret from rfl, Prog.bind_ret] at h

/-! ## Memory -/

section Memory
variable {s : Shape} {d : DType}

/-- What `tl.load ptrs mask other` can return from memory `m`: each enabled lane holds its
element; a disabled lane holds `other` when given. -/
def LoadRes (m : Mem) (ptrs : Tensor s (Ptr d)) (mask : Option (Tensor s Bool))
    (other : Option (Tensor s d.Val)) (v : Tensor s d.Val) : Prop :=
  ∀ i, (∀ a, tl.addrs ptrs mask i = some a → m.read d a = some (v.get i)) ∧
    (tl.addrs ptrs mask i = none → ∀ o, other = some o → v.get i = o.get i)

theorem safe_load {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {other : Option (Tensor s d.Val)} {k : Tensor s d.Val → Prog α}
    (hval : ∀ i a, tl.addrs ptrs mask i = some a → ∃ v, m.read d a = some v)
    (hR : accBytes d (tl.addrs ptrs mask) ⊆ R)
    (hk : ∀ v, LoadRes m ptrs mask other v → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mask other >>= k) := by
  simp only [tl.load, bind_eq, Prog.bind]
  refine .load ?_ hR (fun v hok => hk _ ?_)
  · rintro ⟨i, a, ha, hnone⟩
    obtain ⟨v, hv⟩ := hval i a ha
    rw [hv] at hnone; cases hnone
  · intro i
    constructor
    · intro a ha
      have := hok i a ha
      cases mask with
      | none => cases other <;> exact this
      | some mm =>
        cases other with
        | none => exact this
        | some o =>
          have hm : mm.get i = true := by
            simp only [tl.addrs] at ha; split at ha
            · assumption
            · cases ha
          simpa [Tensor.where_, hm] using this
    · intro hnone o ho
      subst ho
      cases mask with
      | none => simp [tl.addrs] at hnone
      | some mm =>
        have hm : mm.get i = false := by
          simp only [tl.addrs] at hnone; split at hnone
          · cases hnone
          · simpa using ‹¬ mm.get i = true›
        simp [Tensor.where_, hm]

/-- What memory is after `tl.store ptrs vals mask` from `m`: the enabled lanes hold their
values; nothing else changed. -/
def StoreRes (m m' : Mem) (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool)) : Prop :=
  (∀ i a, tl.addrs ptrs mask i = some a → m'.read d a = some (vals.get i)) ∧
    (∀ x, x ∉ accBytes d (tl.addrs ptrs mask) → m' x = m x)

theorem enabled_nodup (addrs : Idx s → Option Int) : (enabled addrs).Nodup :=
  (Idx.nodup_all s).filter _

theorem mem_enabled {addrs : Idx s → Option Int} {i : Idx s} {a : Int} (h : addrs i = some a) :
    i ∈ enabled addrs := by
  simp [enabled, Idx.mem_all, h]

/-- `tl.store` to pairwise distinct elements. The rest of the program may not write the stored
bytes again. -/
theorem safe_store {ptrs : Tensor s (Ptr d)} {vals : Tensor s d.Val}
    {mask : Option (Tensor s Bool)} {k : Unit → Prog α}
    (hok : StoreOk m d (tl.addrs ptrs mask))
    (hdist : DistinctLanes d (tl.addrs ptrs mask))
    (hW : accBytes d (tl.addrs ptrs mask) ⊆ W)
    (hk : ∀ m', StoreRes m m' ptrs vals mask →
      Safe E R (W \ accBytes d (tl.addrs ptrs mask)) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mask >>= k) := by
  simp only [tl.store, bind_eq, Prog.bind]
  refine .store hok hW (fun order hperm => hk _ ⟨?_, ?_⟩)
  · intro i a ha
    exact writeAll_read hdist hok order (hperm.nodup_iff.mpr (enabled_nodup _)) i a
      (hperm.mem_iff.mpr (mem_enabled ha)) ha
  · intro x hx
    exact writeAll_frame _ _ _ _ hx

/-- `tl.store` to any lanes (duplicates allowed): every order of the writes. -/
theorem safe_store_any {ptrs : Tensor s (Ptr d)} {vals : Tensor s d.Val}
    {mask : Option (Tensor s Bool)} {k : Unit → Prog α}
    (hok : StoreOk m d (tl.addrs ptrs mask))
    (hW : accBytes d (tl.addrs ptrs mask) ⊆ W)
    (hk : ∀ order, order.Perm (enabled (tl.addrs ptrs mask)) →
      Safe E R (W \ accBytes d (tl.addrs ptrs mask)) Q (writeAll m d (tl.addrs ptrs mask) vals order)
        (k ())) :
    Safe E R W Q m (tl.store ptrs vals mask >>= k) := by
  simp only [tl.store, bind_eq, Prog.bind]
  exact .store hok hW hk

/-- `tl.store1 p v`, a store through a scalar pointer, is `tl.store` of the scalar tensors with no
mask, so `safe_store` and `safe_store_any` apply to it after this rewrite (library issue L113). -/
theorem tl.store1_eq (p : Ptr d) (v : d.Val) :
    tl.store1 p v = tl.store (Tensor.scalar p) (Tensor.scalar v) none := rfl

/-- `tl.store1 p v`: the one lane writes `v` at `p` (a scalar store has no second lane, so no
duplicate), as `safe_store` gives for the scalar tensors (library issue L113). -/
theorem safe_store1 {p : Ptr d} {v : d.Val} {k : Unit → Prog α}
    (hok : StoreOk m d (tl.addrs (Tensor.scalar p) none))
    (hdist : DistinctLanes d (tl.addrs (Tensor.scalar p) none))
    (hW : accBytes d (tl.addrs (Tensor.scalar p) none) ⊆ W)
    (hk : ∀ m', StoreRes m m' (Tensor.scalar p) (Tensor.scalar v) none →
      Safe E R (W \ accBytes d (tl.addrs (Tensor.scalar p) none)) Q m' (k ())) :
    Safe E R W Q m (tl.store1 p v >>= k) := by
  rw [tl.store1_eq]
  exact safe_store hok hdist hW hk

end Memory

/-! ## Arithmetic -/

section Arith
variable {s : Shape} {f : TVA.Math.Float.Format}

/-- **An exact float operation** (`tl.upToNaN v`, which `tl.add`, `sub`, `mul`, `fma`, `div_rn`,
`sqrt_rn`, `neg` and `abs` are): the result is the exact result up to NaN payload. `tl.cvt` is a
`pick` of its format check around `tl.upToNaN`, so its rule is `safe_cvt`. -/
theorem safe_upToNaN {f' : TVA.Math.Float.Format} {v : Tensor s (FloatV f')} {k : Tensor s (FloatV f') → Prog α}
    (hk : ∀ r : Tensor s (FloatV f'), (∀ i, FloatV.NaNEq (v.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.upToNaN v >>= k) := by
  simp only [tl.upToNaN, bind_eq, Prog.bind]
  exact .pick ⟨v, fun i => FloatV.NaNEq.refl _⟩ hk

/-- When the exact result has no NaN, the operation gives exactly it. -/
theorem safe_exact {f' : TVA.Math.Float.Format} {v : Tensor s (FloatV f')} {k : Tensor s (FloatV f') → Prog α}
    (hnan : ∀ i, (v.get i).isNaN = false) (hk : Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.upToNaN v >>= k) :=
  safe_upToNaN fun r hr => by
    have : r = v := by ext i; exact (hr i).eq (hnan i)
    subst this; exact hk

theorem safe_add {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.add (x.get i) (y.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.add x y >>= k) := safe_upToNaN hk

theorem safe_sub {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.sub (x.get i) (y.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sub x y >>= k) := safe_upToNaN hk

theorem safe_mul_add {a b c : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i,
      FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
      (E.fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i))) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.mul_add a b c >>= k) := by
  simp only [tl.mul_add, bind_eq, Prog.bind]
  exact .env (.pick ⟨⟨fun i => FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)⟩,
    fun i => Or.inl (FloatV.NaNEq.refl _)⟩ hk)

theorem safe_addc {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, AddRel E.fusion (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.addc x y >>= k) := by
  simp only [tl.addc, bind_eq, Prog.bind]
  exact .env (.pick ⟨Tensor.map2 FloatV.add x y, fun i => AddRel.ieee _ _ _⟩ hk)

theorem safe_subc {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, AddRel E.fusion (x.get i) (FloatV.neg (y.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.subc x y >>= k) := safe_addc hk

theorem safe_mul {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.mul (x.get i) (y.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.mul x y >>= k) := safe_upToNaN hk

theorem safe_fma {x y z : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f),
      (∀ i, FloatV.NaNEq (FloatV.fma (x.get i) (y.get i) (z.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.fma x y z >>= k) := safe_upToNaN hk

theorem safe_div_rn {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.divRn (x.get i) (y.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.div_rn x y >>= k) := safe_upToNaN hk

theorem safe_sqrt_rn {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.sqrtRn (x.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sqrt_rn x >>= k) := safe_upToNaN hk

theorem safe_neg {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.neg (x.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.neg x >>= k) := safe_upToNaN hk

theorem safe_abs {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.abs (x.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.abs x >>= k) := safe_upToNaN hk

theorem safe_cvt {g : TVA.Math.Float.Format} {x : Tensor s (FloatV f)} {rnd : TVA.Math.Float.RoundMode}
    {k : Tensor s (FloatV g) → Prog α} (hok : f.Std ∧ g.Std ∧ tl.RoundOk f g rnd)
    (hk : ∀ r : Tensor s (FloatV g), (∀ i, FloatV.NaNEq (FloatV.cvtWith rnd (x.get i)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cvt x rnd >>= k) := by
  simp only [tl.cvt, bind_eq, Prog.bind]
  exact .pick ⟨(), hok⟩ fun _ _ => safe_upToNaN hk

/-- **`tl.convert d d' x`** (`x.to(d')` with the default rounding, and every implicit conversion of
an operand: a promotion, the `int1` condition of `tl.where`): when every element has a result
`ConvRel` allows on the platform, the rest of the program runs on every tensor whose elements are
such results (library issue L113). -/
theorem safe_convert {d d' : DType} {x : Tensor s d.Val} {k : Tensor s d'.Val → Prog α}
    (hex : ∀ i, ∃ r, ConvRel E.P d d' (x.get i) r)
    (hk : ∀ r : Tensor s d'.Val, (∀ i, ConvRel E.P d d' (x.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.convert d d' x >>= k) := by
  simp only [tl.convert, bind_eq, Prog.bind]
  refine .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ ?_)
  intro r hr
  simpa [Prog.bind] using hk r hr

/-- **`x.to(dtype)` out of fp8**: finite elements convert exactly. -/
theorem safe_cvtFrom8 {g : TVA.Math.Float.Format} {x : Tensor s (FloatV f)}
    {k : Tensor s (FloatV g) → Prog α}
    (hok : f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32))
    (hk : ∀ r : Tensor s (FloatV g), (∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cvtFrom8 x >>= k) := by
  simp only [tl.cvtFrom8, bind_eq, Prog.bind]
  exact .pick ⟨(), hok⟩ fun _ _ => .pick ⟨x.map FloatV.cvt, fun i _ => by simp [Tensor.get_map]⟩ hk

/-- **`x.to(float)` of an integer tensor**: each element is one of the roundings `tl.ItofRel`
allows. -/
theorem safe_itof {sg : Bool} {w : Nat} {x : Tensor s (IntV sg w)} {k : Tensor s (FloatV f) → Prog α}
    (hstd : f.Std)
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, tl.ItofRel f (x.get i).toInt (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.itof x >>= k) := by
  simp only [tl.itof, bind_eq, Prog.bind]
  exact .pick ⟨(), hstd⟩ fun _ _ =>
    .pick ⟨x.map fun a => FloatV.ofRat (a.toInt : Rat), fun i => Or.inl (by simp [Tensor.get_map])⟩ hk

theorem safe_maximum {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.MaxRel true (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.maximum x y >>= k) := by
  simp only [tl.maximum, bind_eq, Prog.bind]
  exact .pick ⟨Tensor.map2 FloatV.max x y, fun i => Or.inl (FloatV.NaNEq.refl _)⟩ hk

theorem safe_minimum {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.MaxRel false (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.minimum x y >>= k) := by
  simp only [tl.minimum, bind_eq, Prog.bind]
  exact .pick ⟨Tensor.map2 FloatV.min x y, fun i => Or.inl (FloatV.NaNEq.refl _)⟩ hk

/-- An approximate function: some result exists for every element (`hex`, from the function's
rule), and the rest is safe for every result the platform may give. -/
theorem safe_math {fn : MathFn} {x : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hex : ∀ i, ∃ r, E.P.math fn f (x.get i) r)
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, E.P.math fn f (x.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.math fn x >>= k) := by
  simp only [tl.math, bind_eq, Prog.bind]
  exact .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk)

theorem safe_div {x y : Tensor s (FloatV f)} {k : Tensor s (FloatV f) → Prog α}
    (hex : ∀ i, ∃ r, E.P.div f (x.get i) (y.get i) r)
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, E.P.div f (x.get i) (y.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.div x y >>= k) := by
  simp only [tl.div, bind_eq, Prog.bind]
  exact .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk)

theorem safe_cvt8 {g : TVA.Math.Float.Format} {x : Tensor s (FloatV f)} {rnd : TVA.Math.Float.RoundMode}
    {k : Tensor s (FloatV g) → Prog α}
    (hok : (f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ tl.RoundOk f g rnd)
    (hex : ∀ i, ∃ r, E.P.cvt f g rnd (x.get i) r)
    (hk : ∀ r : Tensor s (FloatV g), (∀ i, E.P.cvt f g rnd (x.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cvt8 x rnd >>= k) := by
  simp only [tl.cvt8, bind_eq, Prog.bind]
  exact .pick ⟨(), hok⟩ fun _ _ =>
    .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk)

theorem safe_dot {cfg : DotCfg} {M N K : Nat} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {acc : Tensor [M, N] cfg.acc.Val}
    {k : Tensor [M, N] cfg.acc.Val → Prog α}
    (hex : ∃ r, E.P.dot cfg a b acc r)
    (hk : ∀ r, E.P.dot cfg a b acc r → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot cfg a b acc >>= k) := by
  simp only [tl.dot, bind_eq, Prog.bind]
  exact .env (.pick hex hk)

/-- `tl.dot(a, b) + c`: either combined form, given that the `tl.dot` with accumulator `c` has a
result (`hex`): that one form is enough for the program to go on, so neither the separate product
(with the zero accumulator `z`) nor the addition needs to have one. -/
theorem safe_dot_add {cfg : DotCfg} {M N K : Nat} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {c z : Tensor [M, N] cfg.acc.Val}
    {k : Tensor [M, N] cfg.acc.Val → Prog α}
    (hex : ∃ r, E.P.dot cfg a b c r)
    (hk : ∀ r, (E.P.dot cfg a b c r ∨ ∃ d, E.P.dot cfg a b z d ∧
        ∀ i, AccAdd E.fusion cfg.acc (d.get i) (c.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot_add cfg a b c z >>= k) := by
  simp only [tl.dot_add, bind_eq, Prog.bind]
  obtain ⟨r, hr⟩ := hex
  exact .env (.pick ⟨r, Or.inl hr⟩ hk)

/-! ### Integer division, shifts and hints -/

section IntOps
variable {sg : Bool} {w : Nat}

/-- `x // y`: no lane divides by zero or overflows. -/
theorem safe_idiv {x y : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, ∃ q, (x.get i).div? (y.get i) = some q)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).div? (y.get i) = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.idiv x y >>= k) := by
  simp only [tl.idiv, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => Classical.choose (hok i)⟩, fun i => Classical.choose_spec (hok i)⟩ hk

theorem safe_irem {x y : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, ∃ q, (x.get i).rem? (y.get i) = some q)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).rem? (y.get i) = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.irem x y >>= k) := by
  simp only [tl.irem, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => Classical.choose (hok i)⟩, fun i => Classical.choose_spec (hok i)⟩ hk

theorem safe_shl {x n : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, (n.get i).bits.toNat < w)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).shl? (n.get i).bits.toNat = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.shl x n >>= k) := by
  simp only [tl.shl, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => ⟨(x.get i).bits <<< (n.get i).bits.toNat⟩⟩, fun i => by simp [IntV.shl?, hok i]⟩ hk

theorem safe_ashr {x n : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, (n.get i).bits.toNat < w)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).ashr? (n.get i).bits.toNat = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.ashr x n >>= k) := by
  simp only [tl.ashr, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => ⟨(x.get i).bits.sshiftRight (n.get i).bits.toNat⟩⟩,
    fun i => by simp [IntV.ashr?, hok i]⟩ hk

theorem safe_lshr {x n : Tensor s (IntV sg w)} {k : Tensor s (IntV sg w) → Prog α}
    (hok : ∀ i, (n.get i).bits.toNat < w)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).lshr? (n.get i).bits.toNat = some (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.lshr x n >>= k) := by
  simp only [tl.lshr, bind_eq, Prog.bind]
  exact .pick ⟨⟨fun i => ⟨(x.get i).bits >>> (n.get i).bits.toNat⟩⟩,
    fun i => by simp [IntV.lshr?, hok i]⟩ hk

theorem safe_assume {c : Bool} {k : Unit → Prog α} (hc : c = true) (hk : Safe E R W Q m (k ())) :
    Safe E R W Q m (tl.assume c >>= k) := by
  simp only [tl.assume, bind_eq, Prog.bind]
  exact .pick ⟨(), hc⟩ fun _ _ => hk

theorem safe_multiple_of {n : Nat} {β : Type} [HintVal β] {x : Tensor [n] β} {v contig : Nat}
    {k : Tensor [n] β → Prog α}
    (hv : FirstsDiv x 0 contig v) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.multiple_of x v contig >>= k) := by
  simp only [tl.multiple_of, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hv⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_multiple_of1 {β : Type} [HintVal β] {x : β} {v : Nat} {k : β → Prog α}
    (hv : (v : Int) ∣ HintVal.val x) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.multiple_of1 x v >>= k) := by
  simp only [tl.multiple_of1, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hv⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_max_contiguous {n : Nat} {β : Type} [HintVal β] {x : Tensor [n] β} {C : Nat}
    {k : Tensor [n] β → Prog α}
    (hc : ContigIs x 0 C) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguous x C >>= k) := by
  simp only [tl.max_contiguous, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_max_constancy {n : Nat} {β : Type} {x : Tensor [n] β} {C : Nat}
    {k : Tensor [n] β → Prog α}
    (hc : ConstAlong x 0 C) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_constancy x C >>= k) := by
  simp only [tl.max_constancy, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_multiple_ofN {s : Shape} {β : Type} [HintVal β] {x : Tensor s β} {vs contig : List Nat}
    {k : Tensor s β → Prog α}
    (hv : ∀ k, k < s.length → FirstsDiv x k (contig.getD k 1) (vs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.multiple_ofN x vs contig >>= k) := by
  simp only [tl.multiple_ofN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hv⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_max_contiguousN {s : Shape} {β : Type} [HintVal β] {x : Tensor s β} {Cs : List Nat}
    {k : Tensor s β → Prog α}
    (hc : ∀ k, k < s.length → ContigIs x k (Cs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguousN x Cs >>= k) := by
  simp only [tl.max_contiguousN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_max_contiguous_multiple_of {n : Nat} {β : Type} [HintVal β] {x : Tensor [n] β}
    {C v : Nat} {k : Tensor [n] β → Prog α}
    (hc : ContigAlong x 0 C) (hv : FirstsDiv x 0 C v) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguous_multiple_of x C v >>= k) := by
  simp only [tl.max_contiguous_multiple_of, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc, hv⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_max_contiguous_multiple_ofN {s : Shape} {β : Type} [HintVal β] {x : Tensor s β}
    {Cs vs : List Nat} {k : Tensor s β → Prog α}
    (hcv : ∀ k, k < s.length → ContigAlong x k (Cs.getD k 1) ∧ FirstsDiv x k (Cs.getD k 1) (vs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguous_multiple_ofN x Cs vs >>= k) := by
  simp only [tl.max_contiguous_multiple_ofN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hcv⟩ fun y hy => by rw [hy.1]; exact hk

theorem safe_max_constancyN {s : Shape} {β : Type} {x : Tensor s β} {Cs : List Nat}
    {k : Tensor s β → Prog α}
    (hc : ∀ k, k < s.length → ConstAlong x k (Cs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_constancyN x Cs >>= k) := by
  simp only [tl.max_constancyN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

/-- `reshape` with reordering: some order of the same elements. -/
theorem safe_reshape_any {s : Shape} {β : Type} {x : Tensor s β} {s' : Shape} {k : Tensor s' β → Prog α}
    (hsz : s'.prod = s.prod)
    (hk : ∀ r : Tensor s' β, (∃ e : Idx s' ≃ Idx s, ∀ i, r.get i = x.get (e i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.reshape_any x s' >>= k) := by
  simp only [tl.reshape_any, bind_eq, Prog.bind]
  refine .pick ⟨x.reshape s' hsz, ?_⟩ hk
  refine ⟨⟨fun i => Idx.ofPos s (Idx.pos s' i) (hsz ▸ Idx.pos_lt s' i),
    fun j => Idx.ofPos s' (Idx.pos s j) (hsz.symm ▸ Idx.pos_lt s j), fun i => ?_, fun j => ?_⟩, fun i => rfl⟩
  · simp only
    rw [Idx.ofPos_congr (Idx.pos_ofPos s _ _) _ (Idx.pos_lt s' i), Idx.ofPos_pos]
  · simp only
    rw [Idx.ofPos_congr (Idx.pos_ofPos s' _ _) _ (Idx.pos_lt s j), Idx.ofPos_pos]

end IntOps

/-! ### Reductions -/

theorem Tree.seq_leaves : ∀ n, 0 < n → (Tree.seq n).leaves = List.range n
  | 0, h => absurd h (lt_irrefl 0)
  | 1, _ => rfl
  | n + 2, _ => by
    simp only [Tree.seq, Tree.leaves]
    rw [Tree.seq_leaves (n + 1) (by omega)]
    exact (List.range_succ (n := n + 1)).symm

theorem Tree.seq_covers {n : Nat} (hn : 0 < n) : (Tree.seq n).Covers n := by
  unfold Tree.Covers; rw [Tree.seq_leaves n hn]

/-- With a total combining relation, every tree has a value. -/
theorem Tree.eval_exists {β : Type} {op : β → β → β → Prop} (htot : ∀ a b, ∃ c, op a b c)
    (x : Nat → β) : ∀ t : Tree, ∃ r, t.Eval op x r
  | .leaf i => ⟨x i, .leaf i⟩
  | .node l r => by
    obtain ⟨a, ha⟩ := Tree.eval_exists htot x l
    obtain ⟨b, hb⟩ := Tree.eval_exists htot x r
    obtain ⟨c, hc⟩ := htot a b
    exact ⟨c, .node ha hb hc⟩

theorem addRel_total (fusion : Bool) : ∀ a b : FloatV f, ∃ c, AddRel fusion a b c :=
  fun a b => ⟨_, AddRel.ieee fusion a b⟩

/-- A balanced tree over `lo, …, lo + len − 1` in order: the first half has `⌈len/2⌉` leaves. -/
def Tree.bal (lo : Nat) : Nat → Tree
  | 0 => .leaf lo
  | 1 => .leaf lo
  | len + 2 =>
    .node (Tree.bal lo ((len + 3) / 2)) (Tree.bal (lo + (len + 3) / 2) (len + 2 - (len + 3) / 2))
termination_by len => len
decreasing_by all_goals omega

theorem Tree.bal_leaves (lo : Nat) : ∀ len, 0 < len → (Tree.bal lo len).leaves = List.range' lo len
  | 0, h => absurd h (lt_irrefl 0)
  | 1, _ => by simp [Tree.bal, Tree.leaves]
  | len + 2, _ => by
    rw [Tree.bal, Tree.leaves, Tree.bal_leaves lo ((len + 3) / 2) (by omega),
      Tree.bal_leaves (lo + (len + 3) / 2) (len + 2 - (len + 3) / 2) (by omega)]
    have := List.range'_append (s := lo) (m := (len + 3) / 2) (n := len + 2 - (len + 3) / 2) (step := 1)
    simp only [one_mul] at this
    rw [this]; congr 1; omega
termination_by len => len
decreasing_by all_goals omega

/-- A balanced tree has depth `⌈log₂ len⌉`. -/
theorem Tree.bal_depth (lo : Nat) : ∀ len, (Tree.bal lo len).depth ≤ Nat.clog 2 len
  | 0 => by simp [Tree.bal, Tree.depth]
  | 1 => by simp [Tree.bal, Tree.depth]
  | len + 2 => by
    rw [Tree.bal, Tree.depth, Nat.clog_of_two_le (by norm_num) (by omega)]
    have h1 := Tree.bal_depth lo ((len + 3) / 2)
    have h2 := Tree.bal_depth (lo + (len + 3) / 2) (len + 2 - (len + 3) / 2)
    have hm : Nat.clog 2 (len + 2 - (len + 3) / 2) ≤ Nat.clog 2 ((len + 3) / 2) :=
      Nat.clog_mono_right 2 (by omega)
    have he : (len + 2 + 2 - 1) / 2 = (len + 3) / 2 := by omega
    rw [he]; omega
termination_by len => len
decreasing_by all_goals omega

theorem Tree.bal_covers {n : Nat} (hn : 0 < n) : (Tree.bal 0 n).Covers n := by
  unfold Tree.Covers; rw [Tree.bal_leaves 0 n hn, List.range_eq_range']

/-- Some tree over `0, …, n−1` fits every reduction's depth bound (`tl.sumBound`). -/
theorem Tree.exists_sumBound (P : Platform) {n : Nat} (hn : 0 < n) :
    ∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound P n :=
  ⟨Tree.bal 0 n, Tree.bal_covers hn, (Tree.bal_depth 0 n).trans (le_max_right _ _)⟩

/-- `tl.sumWith`, with the depth bound of the tree (library issue L149). -/
theorem safe_sumWith_depth (fusion : Bool) {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)}
    {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound E.P n ∧
        t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sumWith fusion x >>= k) := by
  simp only [tl.sumWith, bind_eq, Prog.bind]
  refine .env ?_
  obtain ⟨t, hc, hd⟩ := Tree.exists_sumBound E.P hn
  obtain ⟨r, hr⟩ := Tree.eval_exists (addRel_total fusion)
    (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) t
  exact .pick ⟨r, t, hc, hd, hr⟩ hk

theorem safe_sumWith (fusion : Bool) {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)}
    {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧
        t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sumWith fusion x >>= k) :=
  safe_sumWith_depth fusion hn fun r ⟨t, hc, _, he⟩ => hk r ⟨t, hc, he⟩

/-- `tl.sum1With`, with the depth bound of each row's tree (library issue L149). -/
theorem safe_sum1With_depth (fusion : Bool) {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧ t.depth ≤ tl.sumBound E.P N ∧
        t.Eval (AddRel fusion) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1With fusion x >>= k) := by
  simp only [tl.sum1With, bind_eq, Prog.bind]
  refine .env ?_
  obtain ⟨t, hc, hd⟩ := Tree.exists_sumBound E.P hn
  have hex : ∀ i : Fin M, ∃ r, t.Eval (AddRel fusion)
      (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) r :=
    fun i => Tree.eval_exists (addRel_total fusion) _ _
  exact .pick ⟨Tensor.of1 (fun i => Classical.choose (hex i)), fun i =>
    ⟨t, hc, hd, Classical.choose_spec (hex i)⟩⟩ hk

theorem safe_sum1With (fusion : Bool) {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.Eval (AddRel fusion) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1With fusion x >>= k) :=
  safe_sum1With_depth fusion hn fun r hr => hk r fun i =>
    let ⟨t, hc, _, he⟩ := hr i; ⟨t, hc, he⟩

/-- `tl.sum0With`, with the depth bound of each column's tree (library issue L149). -/
theorem safe_sum0With_depth (fusion : Bool) {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧ t.depth ≤ tl.sumBound E.P M ∧
        t.Eval (AddRel fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0With fusion x >>= k) := by
  simp only [tl.sum0With, bind_eq, Prog.bind]
  refine .env ?_
  obtain ⟨t, hc, hd⟩ := Tree.exists_sumBound E.P hm
  have hex : ∀ j : Fin N, ∃ r, t.Eval (AddRel fusion)
      (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) r :=
    fun j => Tree.eval_exists (addRel_total fusion) _ _
  exact .pick ⟨Tensor.of1 (fun j => Classical.choose (hex j)), fun j =>
    ⟨t, hc, hd, Classical.choose_spec (hex j)⟩⟩ hk

theorem safe_sum0With (fusion : Bool) {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.Eval (AddRel fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0With fusion x >>= k) :=
  safe_sum0With_depth fusion hm fun r hr => hk r fun j =>
    let ⟨t, hc, _, he⟩ := hr j; ⟨t, hc, he⟩

/-- `tl.sum` (IEEE nodes). -/
theorem safe_sum {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)} {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧
        t.Eval (AddRel false) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum x >>= k) := safe_sumWith false hn hk

/-- `tl.sum`, with the depth bound of the tree. -/
theorem safe_sum_depth {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)} {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound E.P n ∧
        t.Eval (AddRel false) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum x >>= k) := safe_sumWith_depth false hn hk

theorem safe_sum1 {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.Eval (AddRel false) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1 x >>= k) := safe_sum1With false hn hk

theorem safe_sum1_depth {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧ t.depth ≤ tl.sumBound E.P N ∧
        t.Eval (AddRel false) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1 x >>= k) := safe_sum1With_depth false hn hk

theorem safe_sum0 {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.Eval (AddRel false) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0 x >>= k) := safe_sum0With false hm hk

theorem safe_sum0_depth {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧ t.depth ≤ tl.sumBound E.P M ∧
        t.Eval (AddRel false) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0 x >>= k) := safe_sum0With_depth false hm hk

/-- `tl.sumc` (fusable nodes). -/
theorem safe_sumc {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)} {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sumc x >>= k) := by
  simp only [tl.sumc, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sumWith E.fusion hn hk)

/-- `tl.sumc`, with the depth bound of the tree. -/
theorem safe_sumc_depth {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)} {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound E.P n ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sumc x >>= k) := by
  simp only [tl.sumc, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sumWith_depth E.fusion hn hk)

theorem safe_sum1c {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.Eval (AddRel E.fusion) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1c x >>= k) := by
  simp only [tl.sum1c, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sum1With E.fusion hn hk)

theorem safe_sum1c_depth {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧ t.depth ≤ tl.sumBound E.P N ∧
        t.Eval (AddRel E.fusion) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1c x >>= k) := by
  simp only [tl.sum1c, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sum1With_depth E.fusion hn hk)

/-- `tl.sum0c` (fusable nodes), as `safe_sum1c` along the other axis (library issue L113). -/
theorem safe_sum0c {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0c x >>= k) := by
  simp only [tl.sum0c, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sum0With E.fusion hm hk)

theorem safe_sum0c_depth {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧ t.depth ≤ tl.sumBound E.P M ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0c x >>= k) := by
  simp only [tl.sum0c, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sum0With_depth E.fusion hm hk)

theorem safe_max1 {M N : Nat} (hn : 0 < N) (isMax : Bool) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.max1 isMax x >>= k) := by
  simp only [tl.max1, bind_eq, Prog.bind]
  have hex : ∀ i : Fin M, ∃ r, (Tree.seq N).Eval (FloatV.MaxRel isMax)
      (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) r :=
    fun i => Tree.eval_exists (op := FloatV.MaxRel isMax)
      (fun a b => ⟨_, Or.inl (FloatV.NaNEq.refl (if isMax then FloatV.max a b else FloatV.min a b))⟩) _ _
  exact .pick ⟨Tensor.of1 (fun i => Classical.choose (hex i)), fun i =>
    ⟨Tree.seq N, Tree.seq_covers hn, Classical.choose_spec (hex i)⟩⟩ hk

theorem safe_maxAll {N : Nat} (hn : 0 < N) (isMax : Bool) {x : Tensor [N] (FloatV f)}
    {k : FloatV f → Prog α}
    (hk : ∀ r : FloatV f, (∃ t : Tree, t.Covers N ∧
        t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < N then x.get1 ⟨j, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.maxAll isMax x >>= k) := by
  simp only [tl.maxAll, bind_eq, Prog.bind]
  obtain ⟨r, hr⟩ := Tree.eval_exists (op := FloatV.MaxRel isMax)
    (fun a b => ⟨_, Or.inl (FloatV.NaNEq.refl (if isMax then FloatV.max a b else FloatV.min a b))⟩)
    (fun j => if h : j < N then x.get1 ⟨j, h⟩ else default) (Tree.seq N)
  exact .pick ⟨r, Tree.seq N, Tree.seq_covers hn, hr⟩ hk

theorem safe_max0 {M N : Nat} (hm : 0 < M) (isMax : Bool) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.Eval (FloatV.MaxRel isMax) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.max0 isMax x >>= k) := by
  simp only [tl.max0, bind_eq, Prog.bind]
  have hex : ∀ j : Fin N, ∃ r, (Tree.seq M).Eval (FloatV.MaxRel isMax)
      (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) r :=
    fun j => Tree.eval_exists (op := FloatV.MaxRel isMax)
      (fun a b => ⟨_, Or.inl (FloatV.NaNEq.refl (if isMax then FloatV.max a b else FloatV.min a b))⟩) _ _
  exact .pick ⟨Tensor.of1 (fun j => Classical.choose (hex j)), fun j =>
    ⟨Tree.seq M, Tree.seq_covers hm, Classical.choose_spec (hex j)⟩⟩ hk

/-! ### Index reductions (`tl.argmaxAll`, `tl.argmax1`, `tl.argmax0`) -/

/-- The pair `argCombine` keeps is one of its two operands. -/
theorem tl.argCombine_eq_or {β : Type} [CmpKey β] (isMax tieLeft : Bool) (a b : β × Nat) :
    tl.argCombine isMax tieLeft a b = a ∨ tl.argCombine isMax tieLeft a b = b := by
  simp only [tl.argCombine]
  by_cases h : ((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
      (tieLeft && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true
  · exact Or.inl (if_pos h)
  · exact Or.inr (if_neg h)

/-- An index reduction returns one of its leaves: the index is a leaf of the tree, and the value
is the element at that index. -/
theorem Tree.eval_argCombine {β : Type} [CmpKey β] {isMax tieLeft : Bool} {g : Nat → β} :
    ∀ {t : Tree} {p : β × Nat}, t.Eval (fun a b c => c = tl.argCombine isMax tieLeft a b)
      (fun j => (g j, j)) p → p.2 ∈ t.leaves ∧ p.1 = g p.2
  | .leaf i, _, .leaf _ => ⟨by simp [Tree.leaves], rfl⟩
  | .node l r, _, .node (a := a) (b := b) ha hb hc => by
    obtain ⟨hal, haf⟩ := Tree.eval_argCombine ha
    obtain ⟨hbl, hbf⟩ := Tree.eval_argCombine hb
    subst hc
    rcases tl.argCombine_eq_or isMax tieLeft a b with h | h <;> rw [h]
    · exact ⟨List.mem_append_left _ hal, haf⟩
    · exact ⟨List.mem_append_right _ hbl, hbf⟩

/-- `argCombine` always has a result. -/
theorem tl.argCombine_total {β : Type} [CmpKey β] (isMax tieLeft : Bool) (a b : β × Nat) :
    ∃ c, c = tl.argCombine isMax tieLeft a b := ⟨_, rfl⟩

/-- The pair a tree gives is a result `ArgRes` allows. -/
theorem tl.ArgRes.of_tree {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat} {g : Nat → β}
    {p : β × Nat} {t : Tree} (ht : t.Covers n)
    (hp : t.Eval (fun a b c => c = tl.argCombine isMax tieLeft a b) (fun j => (g j, j)) p) :
    tl.ArgRes isMax tieLeft n g (p.1, IntV.ofNat p.2) :=
  Or.inl ⟨p, t, ht, hp, rfl⟩

/-- Some result exists (a sequential tree's). -/
theorem tl.ArgRes.exists {β : Type} [CmpKey β] (isMax tieLeft : Bool) {n : Nat} (hn : 0 < n)
    (g : Nat → β) : ∃ r, tl.ArgRes isMax tieLeft n g r := by
  obtain ⟨p, hp⟩ := Tree.eval_exists (op := fun a b c => c = tl.argCombine isMax tieLeft a b)
    (tl.argCombine_total isMax tieLeft) (fun j => (g j, j)) (Tree.seq n)
  exact ⟨_, tl.ArgRes.of_tree (Tree.seq_covers hn) hp⟩

/-- **Without NaN** in the slice, the result is the pair some tree gives. -/
theorem tl.ArgRes.tree_of_noNaN {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat}
    {g : Nat → β} {r : β × I32} (hno : ∀ j, j < n → CmpKey.nan (g j) = false)
    (h : tl.ArgRes isMax tieLeft n g r) :
    ∃ p : β × Nat, ∃ t : Tree, t.Covers n ∧
      t.Eval (fun a b c => c = tl.argCombine isMax tieLeft a b) (fun j => (g j, j)) p ∧
      r = (p.1, IntV.ofNat p.2) := by
  rcases h with h | ⟨⟨j, hj, hnan⟩, -⟩
  · exact h
  · rw [hno j hj] at hnan; cases hnan

/-- The index is that of a position. -/
theorem tl.ArgRes.index_lt {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat}
    {g : Nat → β} {r : β × I32} (h : tl.ArgRes isMax tieLeft n g r) :
    ∃ j, j < n ∧ r.2 = IntV.ofNat j := by
  rcases h with ⟨p, t, ht, hp, rfl⟩ | ⟨-, -, hidx⟩
  · have hl := (Tree.eval_argCombine hp).1
    exact ⟨p.2, List.mem_range.mp (ht.mem_iff.mp hl), rfl⟩
  · exact hidx

/-- The value is one of the elements, up to NaN payload. -/
theorem tl.ArgRes.value {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat}
    {g : Nat → β} {r : β × I32} (h : tl.ArgRes isMax tieLeft n g r) :
    ∃ j, j < n ∧ (r.1 = g j ∨ (CmpKey.nan r.1 = true ∧ CmpKey.nan (g j) = true)) := by
  rcases h with ⟨p, t, ht, hp, rfl⟩ | ⟨-, hval, -⟩
  · obtain ⟨hl, hv⟩ := Tree.eval_argCombine hp
    exact ⟨p.2, List.mem_range.mp (ht.mem_iff.mp hl), Or.inl hv⟩
  · exact hval

/-- Integers have no NaN. -/
@[simp] theorem IntV.cmpKey_nan {sg : Bool} {w : Nat} (a : IntV sg w) : CmpKey.nan a = false := rfl

/-- `tl.argmaxAll` (`tl.max`/`tl.min` with `return_indices`, `tl.argmax`/`tl.argmin` of a rank-1
tensor): never undefined; the rest runs for every result `ArgRes` allows. -/
theorem safe_argmaxAll {β : Type} [CmpKey β] [Inhabited β] {N : Nat} (hn : 0 < N)
    (isMax tieLeft : Bool)
    {x : Tensor [N] β} {k : β × I32 → Prog α}
    (hk : ∀ r : β × I32,
      tl.ArgRes isMax tieLeft N (fun j => if h : j < N then x.get1 ⟨j, h⟩ else default) r →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.argmaxAll isMax tieLeft x >>= k) := by
  simp only [tl.argmaxAll, bind_eq, Prog.bind]
  exact .pick (tl.ArgRes.exists isMax tieLeft hn _) hk

/-- `tl.argmax1` (axis 1 of a rank-2 tensor), as `safe_argmaxAll` for each row. -/
theorem safe_argmax1 {β : Type} [CmpKey β] [Inhabited β] {M N : Nat} (hn : 0 < N)
    (isMax tieLeft : Bool)
    {x : Tensor [M, N] β} {k : Tensor [M] β × Tensor [M] I32 → Prog α}
    (hk : ∀ r : Tensor [M] β × Tensor [M] I32, (∀ i, tl.ArgRes isMax tieLeft N
        (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.1.get1 i, r.2.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.argmax1 isMax tieLeft x >>= k) := by
  simp only [tl.argmax1, bind_eq, Prog.bind]
  have hex : ∀ i : Fin M, ∃ r, tl.ArgRes isMax tieLeft N
      (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) r :=
    fun i => tl.ArgRes.exists isMax tieLeft hn _
  refine .pick ⟨(Tensor.of1 fun i => (Classical.choose (hex i)).1,
      Tensor.of1 fun i => (Classical.choose (hex i)).2), fun i => ?_⟩ hk
  simpa using Classical.choose_spec (hex i)

/-- `tl.argmax0` (axis 0 of a rank-2 tensor), as `safe_argmaxAll` for each column. -/
theorem safe_argmax0 {β : Type} [CmpKey β] [Inhabited β] {M N : Nat} (hm : 0 < M)
    (isMax tieLeft : Bool)
    {x : Tensor [M, N] β} {k : Tensor [N] β × Tensor [N] I32 → Prog α}
    (hk : ∀ r : Tensor [N] β × Tensor [N] I32, (∀ j, tl.ArgRes isMax tieLeft M
        (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.1.get1 j, r.2.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.argmax0 isMax tieLeft x >>= k) := by
  simp only [tl.argmax0, bind_eq, Prog.bind]
  have hex : ∀ j : Fin N, ∃ r, tl.ArgRes isMax tieLeft M
      (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) r :=
    fun j => tl.ArgRes.exists isMax tieLeft hm _
  refine .pick ⟨(Tensor.of1 fun j => (Classical.choose (hex j)).1,
      Tensor.of1 fun j => (Classical.choose (hex j)).2), fun j => ?_⟩ hk
  simpa using Classical.choose_spec (hex j)

/-- `CmpKey`'s `<` on integers is the order of their values. -/
theorem IntV.cmpKey_lt {sg : Bool} {w : Nat} (a b : IntV sg w) :
    CmpKey.lt a b = true ↔ a.toInt < b.toInt := by
  show IntV.lt a b = true ↔ _
  cases sg
  · simp only [IntV.lt, IntV.toInt, Bool.false_eq_true, ite_false, BitVec.ult_iff_lt]
    simp [BitVec.lt_def]
  · simp only [IntV.lt, IntV.toInt, ite_true]
    exact BitVec.slt_iff_toInt_lt

/-- `CmpKey`'s `==` on integers is the equality of their values. -/
theorem IntV.cmpKey_eq {sg : Bool} {w : Nat} (a b : IntV sg w) :
    CmpKey.eq a b = true ↔ a.toInt = b.toInt := by
  show IntV.eq a b = true ↔ _
  cases sg
  · simp only [IntV.eq, IntV.toInt, Bool.false_eq_true, ite_false, beq_iff_eq, Nat.cast_inj]
    exact ⟨fun h => h ▸ rfl, fun h => BitVec.eq_of_toNat_eq h⟩
  · simp only [IntV.eq, IntV.toInt, ite_true, beq_iff_eq]
    exact ⟨fun h => h ▸ rfl, fun h => BitVec.eq_of_toInt_eq h⟩

section ArgFirst
variable {β : Type} [CmpKey β] (key : β → Int)
  (hlt : ∀ u v : β, CmpKey.lt u v = true ↔ key u < key v)
  (heq : ∀ u v : β, CmpKey.eq u v = true ↔ key u = key v)

include hlt heq in
/-- With `tie_break_left`, on keys ordered by `key`, `argCombine` keeps the pair whose value is
better (greater for `isMax`, smaller otherwise), and on equal values the smaller index. -/
theorem tl.argCombine_tieLeft (isMax : Bool) (a b : β × Nat) :
    tl.argCombine isMax true a b =
      if (if isMax then key b.1 < key a.1 else key a.1 < key b.1) ∨
          (key a.1 = key b.1 ∧ a.2 < b.2) then a else b := by
  have hw : ((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
      (true && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true ↔
      ((if isMax then key b.1 < key a.1 else key a.1 < key b.1) ∨ (key a.1 = key b.1 ∧ a.2 < b.2)) := by
    cases isMax <;> simp [hlt, heq]
  simp only [tl.argCombine]
  by_cases h : (if isMax then key b.1 < key a.1 else key a.1 < key b.1) ∨ (key a.1 = key b.1 ∧ a.2 < b.2)
  · rw [if_pos (hw.mpr h), if_pos h]
  · have h' : ¬ (((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
        (true && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true) := fun h' => h (hw.mp h')
    rw [if_neg h', if_neg h]

include hlt heq in
/-- **`tl.argmax`/`tl.argmin` with `tie_break_left` on integers.** Over any tree whose leaves are
distinct, the reduction of `(g j, j)` gives a leaf `p.2` whose value is a maximum (`isMax`) or
minimum of the leaves' values, and the smallest such leaf. With `t.Covers n` the leaves are
`0, …, n-1`, so `p.2` is the first index of the maximum (minimum). -/
theorem Tree.eval_argCombine_first (isMax : Bool) (g : Nat → β) :
    ∀ {t : Tree} {p : β × Nat}, t.leaves.Nodup →
      t.Eval (fun a b c => c = tl.argCombine isMax true a b) (fun j => (g j, j)) p →
      p.2 ∈ t.leaves ∧ p.1 = g p.2 ∧ ∀ j ∈ t.leaves,
        (if isMax then key (g j) ≤ key (g p.2) else key (g p.2) ≤ key (g j)) ∧
        (key (g j) = key (g p.2) → p.2 ≤ j)
  | .leaf i, _, _, .leaf _ => by
    refine ⟨by simp [Tree.leaves], rfl, fun j hj => ?_⟩
    simp only [Tree.leaves, List.mem_singleton] at hj
    subst hj; cases isMax <;> simp
  | .node l r, _, hnd, .node (a := ⟨va, ia⟩) (b := ⟨vb, ib⟩) ha hb hc => by
    simp only [Tree.leaves, List.nodup_append] at hnd
    obtain ⟨hndl, hndr, hdisj⟩ := hnd
    obtain ⟨hal, haf, hab⟩ := Tree.eval_argCombine_first isMax g hndl ha
    obtain ⟨hbl, hbf, hbb⟩ := Tree.eval_argCombine_first isMax g hndr hb
    simp only at hal haf hab hbl hbf hbb
    subst haf hbf hc
    rw [tl.argCombine_tieLeft key hlt heq]
    simp only [Tree.leaves, List.mem_append]
    by_cases hw : (if isMax then key (g ib) < key (g ia) else key (g ia) < key (g ib)) ∨
        (key (g ia) = key (g ib) ∧ ia < ib)
    · rw [if_pos hw]
      refine ⟨Or.inl hal, rfl, fun j hj => ?_⟩
      rcases hj with hj | hj
      · exact hab j hj
      · obtain ⟨h1, h2⟩ := hbb j hj
        have hjne : ia ≠ j := hdisj _ hal _ hj
        cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] at hw h1 ⊢ <;>
          exact ⟨by omega, fun e => by omega⟩
    · rw [if_neg hw]
      refine ⟨Or.inr hbl, rfl, fun j hj => ?_⟩
      rcases hj with hj | hj
      · obtain ⟨h1, h2⟩ := hab j hj
        have hne : ia ≠ ib := hdisj _ hal _ hbl
        cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] at hw h1 ⊢ <;>
          exact ⟨by omega, fun e => by omega⟩
      · exact hbb j hj

end ArgFirst

/-- Products of a float dtype exist. -/
theorem tl.MulRel.total {d : DType} (hd : d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64) :
    ∀ a b : d.Val, ∃ c, tl.MulRel d a b c := by
  rcases hd with rfl | rfl | rfl | rfl <;> exact fun _ _ => ⟨_, FloatV.NaNEq.refl _⟩

/-- Additions of a float dtype exist. -/
theorem AccAdd.total {d : DType} (fusion : Bool) (hd : d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64) :
    ∀ a b : d.Val, ∃ c, AccAdd fusion d a b c := by
  rcases hd with rfl | rfl | rfl | rfl <;> exact fun a b => ⟨_, AddRel.ieee fusion a b⟩

/-- **`tl.sum(x[:, :, None] * y[None, :, :], axis=1)`**: the result is, as a whole, either a
tensor each of whose elements is a sum of its products along a tree (each element with its own
tree and products), or Triton's `tl.dot` rewrite (`E.P.dot` with the zero accumulator); the form
is chosen once for the whole tensor, not per element. -/
theorem safe_sum_bcast_mul {d : DType} {M K N : Nat} {x : Tensor [M, K] d.Val} {y : Tensor [K, N] d.Val}
    (hd : d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64) (hK : 0 < K)
    {k : Tensor [M, N] d.Val → Prog α}
    (hk : ∀ r : Tensor [M, N] d.Val,
      ((∀ i j, ∃ p : Nat → d.Val,
        (∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
        ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd E.fusion d) p (r.get2 i j)) ∨
      (d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
        E.P.dot { a := d, b := d, acc := d, precision := .ieee } x y (Tensor.full (d.ofBits 0)) r)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum_bcast_mul d x y >>= k) := by
  simp only [tl.sum_bcast_mul, bind_eq, Prog.bind]
  have hp : ∀ (i : Fin M) (j : Fin N), ∃ p : Nat → d.Val,
      ∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k) := by
    intro i j
    refine ⟨fun k => if hk : k < K then Classical.choose (tl.MulRel.total hd (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j))
      else default, fun k hk => ?_⟩
    simp only [dif_pos hk]
    exact Classical.choose_spec (tl.MulRel.total hd _ _)
  have hex : ∀ (i : Fin M) (j : Fin N), ∃ (v : d.Val) (p : Nat → d.Val) (t : Tree),
      (∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
      t.Covers K ∧ t.Eval (AccAdd E.fusion d) p v := by
    intro i j
    obtain ⟨p, hp⟩ := hp i j
    obtain ⟨v, hv⟩ := Tree.eval_exists (AccAdd.total E.fusion hd) p (Tree.seq K)
    exact ⟨v, p, Tree.seq K, hp, Tree.seq_covers hK, hv⟩
  refine .env (.pick ⟨Tensor.of2 fun i j => Classical.choose (hex i j), Or.inl fun i j => ?_⟩ hk)
  obtain ⟨p, t, h1, h2, h3⟩ := Classical.choose_spec (hex i j)
  exact ⟨p, h1, t, h2, h3⟩

/-- `x.to(int)`: every element must be finite and in range; the result truncates. -/
theorem safe_ftoi {sg : Bool} {w : Nat} {x : Tensor s (FloatV f)} {k : Tensor s (IntV sg w) → Prog α}
    (hstd : f.Std) (hin : ∀ i, (x.get i).isFinite = true ∧
      (TVA.Math.Float.Format.intRange sg w).1 ≤ (x.get i).toRat.num.tdiv (x.get i).toRat.den ∧
      (x.get i).toRat.num.tdiv (x.get i).toRat.den ≤ (TVA.Math.Float.Format.intRange sg w).2)
    (hk : Safe E R W Q m (k ⟨fun i => IntV.ofInt ((x.get i).toRat.num.tdiv (x.get i).toRat.den)⟩)) :
    Safe E R W Q m (tl.ftoi x >>= k) := by
  simp only [tl.ftoi, bind_eq, Prog.bind]
  refine .pick ⟨(), hstd⟩ fun _ _ => .pick ⟨⟨fun i => IntV.ofInt ((x.get i).toRat.num.tdiv (x.get i).toRat.den)⟩,
    fun i => ⟨hin i, rfl⟩⟩ fun r hr => ?_
  have : r = ⟨fun i => IntV.ofInt ((x.get i).toRat.num.tdiv (x.get i).toRat.den)⟩ := by
    ext i; exact (hr i).2
  subst this; exact hk

end Arith

/-! ## Loops -/

section Loops

/-- The bytes that iterations `0, …, n-1` may write, iteration `t` writing inside `Wt t`. -/
def iterW (Wt : Nat → Set Int) (n : Nat) : Set Int := {x | ∃ t, t < n ∧ x ∈ Wt t}

theorem iterW_mono {Wt : Nat → Set Int} {n n' : Nat} (h : n ≤ n') : iterW Wt n ⊆ iterW Wt n' :=
  fun _ ⟨t, ht, hx⟩ => ⟨t, by omega, hx⟩

theorem iterW_zero (Wt : Nat → Set Int) : iterW Wt 0 = ∅ := by
  ext x; simp [iterW]

theorem iterW_empty (n : Nat) : iterW (fun _ => ∅) n = ∅ := by
  ext x; simp [iterW]

/-- **Loop invariant** for `forN` (and hence `for` loops): `I t st m` holds before iteration
`t`; each iteration preserves it and writes only inside its own footprint `Wt t`. Different
iterations write disjoint bytes (a conflict across iterations is a conflict within the program,
library issue L15). A loop that writes nothing takes `Wt = fun _ => ∅` (`iterW_empty`). -/
theorem safe_forN {σ : Type} (n : Nat) (body : Nat → σ → Prog σ) (init : σ)
    (I : Nat → σ → Mem → Prop) (Wt : Nat → Set Int)
    (hWt : ∀ t u, t < u → u < n → Disjoint (Wt t) (Wt u)) (h0 : I 0 init m)
    (hstep : ∀ t, t < n → ∀ st m₁, I t st m₁ →
      Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁ (body t st)) :
    Safe E R (iterW Wt n) (fun st m₁ => I n st m₁) m (tl.forN n body init) := by
  induction n with
  | zero => exact .ret h0
  | succ n ih =>
    simp only [tl.forN]
    have hd : Disjoint (iterW Wt n) (Wt n) := Set.disjoint_left.mpr fun x ⟨t, ht, hx⟩ hx' =>
      Set.disjoint_left.mp (hWt t n ht (by omega)) hx hx'
    have h := Safe.bind ((ih (fun t u htu hu => hWt t u htu (by omega))
      (fun t ht => hstep t (by omega))).mono (fun st m₁ h => hstep n (by omega) st m₁ h) le_rfl) hd
    refine h.mono (fun _ _ h => h) ?_
    rintro x (⟨t, ht, hx⟩ | hx)
    · exact ⟨t, by omega, hx⟩
    · exact ⟨n, by omega, hx⟩

/-- `forN` followed by the rest of the program, which writes outside the loop's footprint. -/
theorem safe_forN_bind {σ : Type} {n : Nat} {body : Nat → σ → Prog σ} {init : σ}
    {k : σ → Prog α} (I : Nat → σ → Mem → Prop) (Wt : Nat → Set Int) {Wk : Set Int}
    (hWt : ∀ t u, t < u → u < n → Disjoint (Wt t) (Wt u)) (hWk : Disjoint (iterW Wt n) Wk)
    (h0 : I 0 init m)
    (hstep : ∀ t, t < n → ∀ st m₁, I t st m₁ →
      Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁ (body t st))
    (hk : ∀ st m₁, I n st m₁ → Safe E R Wk Q m₁ (k st)) :
    Safe E R (iterW Wt n ∪ Wk) Q m (tl.forN n body init >>= k) :=
  Safe.bind ((safe_forN n body init I Wt hWt h0 hstep).mono hk le_rfl) hWk

/-- `for i in range(lo, hi, step)` with `step > 0` and no overflow of the last increment: an
invariant over the trip number `t`, where the induction variable is `lo + t * step`; trip `t`
writes inside `Wt t`. -/
theorem safe_forRange {σ : Type} {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init : σ}
    {body : IntV sg w → σ → Prog σ} {k : σ → Prog α} (hrange : tl.RangeOk lo hi step)
    (I : Nat → σ → Mem → Prop) (Wt : Nat → Set Int) {Wk : Set Int}
    (hWt : ∀ t u, t < u → u < tl.tripCount lo.toInt hi.toInt step.toInt → Disjoint (Wt t) (Wt u))
    (hWk : Disjoint (iterW Wt (tl.tripCount lo.toInt hi.toInt step.toInt)) Wk) (h0 : I 0 init m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁
        (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ → Safe E R Wk Q m₁ (k st)) :
    Safe E R (iterW Wt (tl.tripCount lo.toInt hi.toInt step.toInt) ∪ Wk) Q m
      (tl.forRange lo hi step init body >>= k) := by
  simp only [tl.forRange, if_pos hrange]
  exact safe_forN_bind I Wt hWt hWk h0 hstep hk

/-- **While loop** with an invariant `I` and a variant `μ` that every iteration decreases. The
condition writes nothing; the iteration from a state `st` writes inside `Wv (μ st)`, and
footprints of different variant values are disjoint. -/
theorem safe_while {σ : Type} (cond : σ → Prog Bool) (body : σ → Prog σ) (init : σ)
    {Q' : σ → Mem → Prop} (I : σ → Mem → Prop) (μ : σ → Nat) (Wv : Nat → Set Int)
    (hWv : ∀ u v, u < v → Disjoint (Wv u) (Wv v)) (h0 : I init m)
    (hiter : ∀ st m₁, I st m₁ →
      Safe E R ∅ (fun c m₂ => if c then
          Safe E R (Wv (μ st)) (fun st' m₃ => I st' m₃ ∧ μ st' < μ st) m₂ (body st)
        else Q' st m₂) m₁ (cond st)) :
    Safe E R (iterW Wv (μ init + 1)) Q' m (tl.whileLoop init cond body) := by
  unfold tl.whileLoop
  suffices H : ∀ n, ∀ st m₁, μ st = n → I st m₁ →
      Safe E R (iterW Wv (n + 1)) Q' m₁ (.loop st (fun st => (cond st).bind fun c =>
        if c then (body st).bind (fun st' => .ret (.inl st')) else .ret (.inr st)) .ret) from
    H _ init m rfl h0
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro st m₁ hμ hI
    apply Safe.loop
    unfold Prog.loopStep
    have hd : Disjoint (Wv n) (iterW Wv n) := Set.disjoint_left.mpr fun x hx ⟨t, ht, hx'⟩ =>
      Set.disjoint_left.mp (hWv t n ht) hx' hx
    refine (Safe.bind (W₁ := Wv n) (W₂ := iterW Wv n) ?_ hd).mono (fun _ _ h => h) ?_
    · apply Safe.bind_ro
      refine (hiter st m₁ hI).mono ?_ le_rfl
      intro c m₂ hc
      cases c with
      | true =>
        simp only [ite_true] at hc ⊢
        rw [show Wv n = Wv n ∪ ∅ by simp]
        refine Safe.bind ?_ (Set.disjoint_empty _)
        subst hμ
        refine hc.mono ?_ le_rfl
        rintro st' m₃ ⟨hI', hlt⟩
        have hle : μ st' + 1 ≤ μ st := by omega
        exact .ret ((ih (μ st') hlt st' m₃ rfl hI').mono (fun _ _ h => h) (iterW_mono (Wt := Wv) hle))
      | false =>
        simp only [Bool.false_eq_true, ite_false] at hc ⊢
        exact .ret (.ret hc)
    · rintro x (hx | ⟨t, ht, hx⟩)
      · exact ⟨n, by omega, hx⟩
      · exact ⟨t, by omega, hx⟩

end Loops

end TVA
