import TVA

/-!
# `streamk_matmul_f16`: general facts the proof uses

Lemmas that follow from the library's definitions, proved here for this kernel:

* `safeA_ro`: a private segment that writes nothing (a `Safe` proof with an empty write footprint)
  followed by the rest, for a program that shares memory through atomics (`TVA.SafeA`);
* `safeA_while`: a `while` loop with a pure condition, for `SafeA`, as the library's
  `safe_while` is for `Safe`: an invariant, a variant that every iteration decreases, and write
  footprints per value of the variant, disjoint;
* `replay_read_some`: a replay of atomic steps keeps every readable element readable;
* `int32` arithmetic on small values (`i32_*`).
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## `SafeA` -/

/-- **A private segment that writes nothing**, then the rest. -/
theorem safeA_ro {E : Env} {R W : Set Int} {C : Set (DType × Int)} {α β : Type}
    {Q : β → List AtomicStep → Mem → Prop} {log : List AtomicStep} {m : Mem} {p : Prog α}
    {f : α → Prog β} (h : Safe E R ∅ (fun a m' => SafeA E R W C Q log m' (f a)) m p) :
    SafeA E R W C Q log m (p >>= f) := by
  have := SafeA.of_safe h (Set.empty_disjoint W)
  simpa using this

/-- **`while c(s): s = body(s)`** with a pure condition, for a program with shared atomics: an
invariant `I` over the state, the log and the memory, a variant `μ` that every iteration
decreases, and write footprints `Wv` per value of the variant, disjoint. -/
theorem safeA_while {E : Env} {R : Set Int} {C : Set (DType × Int)} {σ : Type}
    {Q' : σ → List AtomicStep → Mem → Prop} (c : σ → Bool) (body : σ → Prog σ) (init : σ)
    (I : σ → List AtomicStep → Mem → Prop) (μ : σ → Nat) (Wv : Nat → Set Int)
    (hWv : ∀ u v, u < v → Disjoint (Wv u) (Wv v)) {log : List AtomicStep} {m : Mem}
    (h0 : I init log m)
    (hT : ∀ st log m, I st log m → c st = true →
      SafeA E R (Wv (μ st)) C (fun st' log' m' => I st' log' m' ∧ μ st' < μ st) log m (body st))
    (hF : ∀ st log m, I st log m → c st = false → Q' st log m) :
    SafeA E R (iterW Wv (μ init + 1)) C Q' log m (tl.whileLoop init (fun st => pure (c st)) body) := by
  unfold tl.whileLoop
  suffices H : ∀ n, ∀ st log m, μ st = n → I st log m →
      SafeA E R (iterW Wv (n + 1)) C Q' log m (.loop st (fun st => (pure (c st) : Prog Bool).bind fun b =>
        if b then (body st).bind (fun st' => .ret (.inl st')) else .ret (.inr st)) .ret) from
    H _ init log m rfl h0
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro st log m hμ hI
    apply SafeA.loop
    unfold Prog.loopStep
    have hd : Disjoint (Wv n) (iterW Wv n) := Set.disjoint_left.mpr fun x hx ⟨t, ht, hx'⟩ =>
      Set.disjoint_left.mp (hWv t n ht) hx' hx
    refine (SafeA.bind (W₁ := Wv n) (W₂ := iterW Wv n) ?_ hd).mono (fun _ _ _ h => h) ?_
    · show SafeA E R (Wv n) C _ log m ((pure (c st) : Prog Bool).bind fun b =>
        if b = true then (body st).bind (fun st' => Prog.ret (Sum.inl st')) else Prog.ret (Sum.inr st))
      cases hc : c st with
      | true =>
        show SafeA E R (Wv n) C _ log m ((body st).bind fun st' => Prog.ret (Sum.inl st'))
        rw [show Wv n = Wv n ∪ ∅ by simp]
        refine SafeA.bind ?_ (Set.disjoint_empty _)
        subst hμ
        refine (hT st log m hI hc).mono ?_ le_rfl
        rintro st' log' m' ⟨hI', hlt⟩
        have hle : μ st' + 1 ≤ μ st := by omega
        exact .ret ((ih (μ st') hlt st' log' m' rfl hI').mono (fun _ _ _ h => h) (iterW_mono (Wt := Wv) hle))
      | false =>
        show SafeA E R (Wv n) C _ log m (Prog.ret (Sum.inr st))
        exact .ret (.ret (hF st log m hI hc))
    · rintro x (hx | ⟨t, ht, hx⟩)
      · exact ⟨n, by omega, hx⟩
      · exact ⟨t, by omega, hx⟩

/-! ## Replays -/

/-- A replay of atomic steps keeps an element readable: each step writes its own bytes and keeps
every byte present. -/
theorem replay_read_some {m m' : Mem} {l : List AtomicStep} (h : Replay m l m') {d : DType} {a : Int}
    {x : d.Val} (hx : m.read d a = some x) : ∃ y, m'.read d a = some y := by
  induction h generalizing x with
  | nil => exact ⟨x, hx⟩
  | @cons m₀ m₁ s rest _ _ ih =>
    have hw := Mem.writable_of_read hx
    obtain ⟨y, hy⟩ := Mem.read_some_of_writable (m := m₀.write s.d s.a s.new)
      ⟨hw.1, fun i hi => Mem.write_present _ (hw.2 i hi)⟩
    exact ih hy

/-! ## `int32` arithmetic -/

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

theorem i32_lt {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.lt a b = decide (x < y) := by
  rw [IntV.lt_eq, ha, hb]

theorem i32_eq {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.eq a b = decide (x = y) := by
  have e : IntV.eq a b = decide (a = b) := by
    unfold IntV.eq
    by_cases h : a = b
    · subst h; simp
    · have : a.bits ≠ b.bits := fun e => h (by cases a; cases b; simp_all)
      simp [h, this]
  rw [e]
  have hinj : a = b ↔ a.toInt = b.toInt := by
    constructor
    · rintro rfl; rfl
    · intro h
      have h1 : a.bits.toInt = b.bits.toInt := by simpa [IntV.toInt] using h
      cases a; cases b
      simp only [IntV.mk.injEq]
      exact BitVec.toInt_inj.mp h1
  rw [ha, hb] at hinj
  exact decide_eq_decide.mpr hinj

theorem i32_min {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    (IntV.min a b).toInt = min x y := by
  unfold IntV.min
  rw [i32_lt ha hb]
  by_cases h : x < y
  · simp [h, ha]; omega
  · simp [h, hb]; omega

theorem i32_add {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x + y) (h1 : x + y < 2 ^ 31) : (a + b).toInt = x + y := by
  rw [IntV.toInt_add (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_sub {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x - y) (h1 : x - y < 2 ^ 31) : (a - b).toInt = x - y := by
  rw [IntV.toInt_sub (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_mul {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x * y) (h1 : x * y < 2 ^ 31) : (a * b).toInt = x * y := by
  rw [IntV.toInt_mul (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

end Kernels.StreamKMatmul
