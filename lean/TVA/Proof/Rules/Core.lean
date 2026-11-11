import TVA.Proof.Safe

/-!
# Rules: ends of programs, assumptions, hints, reshapes

The proof rules for `TVA.Triton.Core`. A rule has the shape `Safe … (op >>= k)`: its premises
are what the operation needs, and its last premise is the rest of the program `k`, for every
result the operation may give. A per-program proof is a walk down the model's `do` block, applying
the rule of each line:

* `safe_ret` ends the walk; `safe_last` lets the last line of a block use its rule too;
* `safe_assume`, `safe_static_assert`: the condition must hold;
* the hints (`tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy` and their forms): the
  promise must hold, and the value is unchanged;
* `safe_reshape_any`: the result is some reordering of the elements.

How this fits: model → `Safe` per program (these rules, with `TVA.Proof.Rules.Memory`, `Arith`,
`Convert`, `Reduce`, `Dot`, `Loop`) → `correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- The program has ended: show the post-condition. -/
theorem safe_ret {a : α} (h : Q a m) : Safe E R W Q m (pure a) := .ret h

/-- The last line of a program: apply that line's rule with `pure` as the rest. -/
theorem safe_last {p : Prog α} (h : Safe E R W Q m (p >>= pure)) : Safe E R W Q m p := by
  rwa [bind_eq, show (pure : α → Prog α) = Prog.ret from rfl, Prog.bind_ret] at h

/-- `tl.assume(c)`: `c` must hold. -/
theorem safe_assume {c : Bool} {k : Unit → Prog α} (hc : c = true) (hk : Safe E R W Q m (k ())) :
    Safe E R W Q m (tl.assume c >>= k) := by
  simp only [tl.assume, bind_eq, Prog.bind]
  exact .pick ⟨(), hc⟩ fun _ _ => hk

/-- `tl.static_assert(c)`: `c` must hold (the claim's instances satisfy it). -/
theorem safe_static_assert {c : Bool} {k : Unit → Prog α} (hc : c = true)
    (hk : Safe E R W Q m (k ())) : Safe E R W Q m (tl.static_assert c >>= k) := by
  simp only [tl.static_assert, bind_eq, Prog.bind]
  exact .pick ⟨(), hc⟩ fun _ _ => hk

/-! ## Hints -/

section Hints
variable {β : Type}

/-- `tl.multiple_of(x, v)` of a rank-1 tensor: show the promise; the value is `x`. -/
theorem safe_multiple_of {n : Nat} [HintVal β] {x : Tensor [n] β} {v contig : Nat}
    {k : Tensor [n] β → Prog α} (hv : FirstsDiv x 0 contig v) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.multiple_of x v contig >>= k) := by
  simp only [tl.multiple_of, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hv⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.multiple_of(x, v)` of a scalar: show `v ∣ x`; the value is `x`. -/
theorem safe_multiple_of1 [HintVal β] {x : β} {v : Nat} {k : β → Prog α}
    (hv : (v : Int) ∣ HintVal.val x) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.multiple_of1 x v >>= k) := by
  simp only [tl.multiple_of1, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hv⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.max_contiguous(x, C)` of a rank-1 tensor: show the promise; the value is `x`. -/
theorem safe_max_contiguous {n : Nat} [HintVal β] {x : Tensor [n] β} {C : Nat}
    {k : Tensor [n] β → Prog α} (hc : ContigIs x 0 C) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguous x C >>= k) := by
  simp only [tl.max_contiguous, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.max_constancy(x, C)` of a rank-1 tensor: show the promise; the value is `x`. -/
theorem safe_max_constancy {n : Nat} {x : Tensor [n] β} {C : Nat} {k : Tensor [n] β → Prog α}
    (hc : ConstAlong x 0 C) (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_constancy x C >>= k) := by
  simp only [tl.max_constancy, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.max_contiguous(tl.multiple_of(x, v), C)` of a rank-1 tensor: show both promises. -/
theorem safe_max_contiguous_multiple_of {n : Nat} [HintVal β] {x : Tensor [n] β} {C v : Nat}
    {k : Tensor [n] β → Prog α} (hc : ContigAlong x 0 C) (hv : FirstsDiv x 0 C v)
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguous_multiple_of x C v >>= k) := by
  simp only [tl.max_contiguous_multiple_of, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc, hv⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.multiple_of(x, vs)` of any rank: show the promise along every dimension. -/
theorem safe_multiple_ofN {s : Shape} [HintVal β] {x : Tensor s β} {vs contig : List Nat}
    {k : Tensor s β → Prog α}
    (hv : ∀ k, k < s.length → FirstsDiv x k (contig.getD k 1) (vs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.multiple_ofN x vs contig >>= k) := by
  simp only [tl.multiple_ofN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hv⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.max_contiguous(x, Cs)` of any rank: show the promise along every dimension. -/
theorem safe_max_contiguousN {s : Shape} [HintVal β] {x : Tensor s β} {Cs : List Nat}
    {k : Tensor s β → Prog α} (hc : ∀ k, k < s.length → ContigIs x k (Cs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguousN x Cs >>= k) := by
  simp only [tl.max_contiguousN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.max_contiguous(tl.multiple_of(x, vs), Cs)` of any rank: show both promises along every
dimension. -/
theorem safe_max_contiguous_multiple_ofN {s : Shape} [HintVal β] {x : Tensor s β}
    {Cs vs : List Nat} {k : Tensor s β → Prog α}
    (hcv : ∀ k, k < s.length →
      ContigAlong x k (Cs.getD k 1) ∧ FirstsDiv x k (Cs.getD k 1) (vs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_contiguous_multiple_ofN x Cs vs >>= k) := by
  simp only [tl.max_contiguous_multiple_ofN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hcv⟩ fun y hy => by rw [hy.1]; exact hk

/-- `tl.max_constancy(x, Cs)` of any rank: show the promise along every dimension. -/
theorem safe_max_constancyN {s : Shape} {x : Tensor s β} {Cs : List Nat}
    {k : Tensor s β → Prog α} (hc : ∀ k, k < s.length → ConstAlong x k (Cs.getD k 1))
    (hk : Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.max_constancyN x Cs >>= k) := by
  simp only [tl.max_constancyN, bind_eq, Prog.bind]
  exact .pick ⟨x, rfl, hc⟩ fun y hy => by rw [hy.1]; exact hk

end Hints

/-- `tl.reshape(x, s', can_reorder=True)`: the rest must be safe for every reordering of `x`'s
elements. -/
theorem safe_reshape_any {s : Shape} {β : Type} {x : Tensor s β} {s' : Shape}
    {k : Tensor s' β → Prog α} (hsz : s'.prod = s.prod)
    (hk : ∀ r : Tensor s' β, (∃ e : Idx s' ≃ Idx s, ∀ i, r.get i = x.get (e i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.reshape_any x s' >>= k) := by
  simp only [tl.reshape_any, bind_eq, Prog.bind]
  refine .pick ⟨x.reshape s' hsz, ?_⟩ hk
  refine ⟨⟨fun i => Idx.ofPos s (Idx.pos s' i) (hsz ▸ Idx.pos_lt s' i),
    fun j => Idx.ofPos s' (Idx.pos s j) (hsz.symm ▸ Idx.pos_lt s j), fun i => ?_, fun j => ?_⟩,
    fun i => rfl⟩
  · simp only
    rw [Idx.ofPos_congr (Idx.pos_ofPos s _ _) _ (Idx.pos_lt s' i), Idx.ofPos_pos]
  · simp only
    rw [Idx.ofPos_congr (Idx.pos_ofPos s' _ _) _ (Idx.pos_lt s j), Idx.ofPos_pos]

end TVA
