import TVA.Proof.Safe

/-!
# Rules: loops

The proof rules for `TVA.Triton.Loop`: a loop invariant and a write footprint per iteration.

* `safe_forN`, `safe_forN_bind`: a fixed trip count; iteration `t` preserves the invariant `I t`
  and writes inside `Wt t`.
* `safe_forRange`: `for i in range(lo, hi, step)`, the same over the trip number, with the
  induction variable `lo + t * step`.
* `safe_while`: an invariant and a variant that every iteration decreases.

Different iterations write disjoint bytes (a conflict across iterations is a conflict within the
program); the loop's write footprint is `iterW Wt n`. A loop that writes nothing takes
`Wt = fun _ => ∅` (`iterW_empty`).

How this fits: model → `Safe` per program (the rules of `TVA.Proof.Rules`) → `correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- The bytes that iterations `0, …, n-1` may write, iteration `t` writing inside `Wt t`. -/
def iterW (Wt : Nat → Set Int) (n : Nat) : Set Int := {x | ∃ t, t < n ∧ x ∈ Wt t}

/-- More iterations write more. -/
theorem iterW_mono {Wt : Nat → Set Int} {n n' : Nat} (h : n ≤ n') : iterW Wt n ⊆ iterW Wt n' :=
  fun _ ⟨t, ht, hx⟩ => ⟨t, by omega, hx⟩

/-- No iteration writes nothing. -/
@[simp] theorem iterW_zero (Wt : Nat → Set Int) : iterW Wt 0 = ∅ := by
  ext x; simp [iterW]

/-- Iterations that write nothing write nothing. -/
@[simp] theorem iterW_empty (n : Nat) : iterW (fun _ => ∅) n = ∅ := by
  ext x; simp [iterW]

/-- **Loop invariant** for `tl.forN`: `I t st m` holds before iteration `t`; each iteration
preserves it and writes only inside its own footprint `Wt t`, disjoint from the others'. -/
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
      (fun t ht => hstep t (by omega))).mono (fun st m₁ h => hstep n (by omega) st m₁ h)
        le_rfl) hd
    refine h.mono (fun _ _ h => h) ?_
    rintro x (⟨t, ht, hx⟩ | hx)
    · exact ⟨t, by omega, hx⟩
    · exact ⟨n, by omega, hx⟩

/-- `tl.forN` followed by the rest of the program, which writes outside the loop's footprint. -/
theorem safe_forN_bind {σ : Type} {n : Nat} {body : Nat → σ → Prog σ} {init : σ}
    {k : σ → Prog α} (I : Nat → σ → Mem → Prop) (Wt : Nat → Set Int) {Wk : Set Int}
    (hWt : ∀ t u, t < u → u < n → Disjoint (Wt t) (Wt u)) (hWk : Disjoint (iterW Wt n) Wk)
    (h0 : I 0 init m)
    (hstep : ∀ t, t < n → ∀ st m₁, I t st m₁ →
      Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁ (body t st))
    (hk : ∀ st m₁, I n st m₁ → Safe E R Wk Q m₁ (k st)) :
    Safe E R (iterW Wt n ∪ Wk) Q m (tl.forN n body init >>= k) :=
  Safe.bind ((safe_forN n body init I Wt hWt h0 hstep).mono hk le_rfl) hWk

/-- **`for i in range(lo, hi, step)`** within `tl.RangeOk` (a positive step, no overflow of the
last increment): an invariant over the trip number `t`, where the induction variable is
`lo + t * step`; trip `t` writes inside `Wt t`. -/
theorem safe_forRange {σ : Type} {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init : σ}
    {body : IntV sg w → σ → Prog σ} {k : σ → Prog α} (hrange : tl.RangeOk lo hi step)
    (I : Nat → σ → Mem → Prop) (Wt : Nat → Set Int) {Wk : Set Int}
    (hWt : ∀ t u, t < u → u < tl.tripCount lo.toInt hi.toInt step.toInt →
      Disjoint (Wt t) (Wt u))
    (hWk : Disjoint (iterW Wt (tl.tripCount lo.toInt hi.toInt step.toInt)) Wk) (h0 : I 0 init m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁
        (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ →
      Safe E R Wk Q m₁ (k st)) :
    Safe E R (iterW Wt (tl.tripCount lo.toInt hi.toInt step.toInt) ∪ Wk) Q m
      (tl.forRange lo hi step init body >>= k) := by
  simp only [tl.forRange, ite_eq_left hrange]
  exact safe_forN_bind I Wt hWt hWk h0 hstep hk

/-- **`while` loop** (`tl.whileLoop`) with an invariant `I` and a variant `μ` that every
iteration decreases. The condition writes nothing; the iteration from a state `st` writes inside
`Wv (μ st)`, and footprints of different variant values are disjoint. -/
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
        exact .ret ((ih (μ st') hlt st' m₃ rfl hI').mono (fun _ _ h => h)
          (iterW_mono (Wt := Wv) hle))
      | false =>
        simp only [Bool.false_eq_true, ite_false] at hc ⊢
        exact .ret (.ret hc)
    · rintro x (hx | ⟨t, ht, hx⟩)
      · exact ⟨n, by omega, hx⟩
      · exact ⟨t, by omega, hx⟩

end TVA
