import TVA.Proof

/-!
# The `Safe` rules, for a program that ends with a shared atomic

The kernel's operations before its atomic write nothing: each is a `Safe` rule. `SafeA.liftS` turns
such a rule (with the rest of the program as its continuation) into a step of a `SafeA` walk, and the
wrappers below state the rules the walk uses directly in that form.
-/

namespace TVA

section Wrap

variable {E : Env} {R : Set Int} {C : Set (DType × Int)} {α β : Type}
  {Q : β → List AtomicStep → Mem → Prop} {log : List AtomicStep} {m : Mem}

/-- **A `Safe` step of a `SafeA` walk.** An operation `op` that writes nothing, shown `Safe` with
the post-condition "the rest `k` is `SafeA` from here", is a `SafeA` step. -/
theorem SafeA.liftS {op : Prog α} {k : α → Prog β}
    (h : Safe E R ∅ (fun a m' => SafeA E R ∅ C Q log m' (k a)) m (op >>= pure)) :
    SafeA E R ∅ C Q log m (op >>= k) := by
  have := SafeA.of_safe (W₁ := ∅) (W₂ := ∅) (f := k) (p := op >>= pure) (C := C) (log := log)
    (Q := Q) (m := m) (by simpa using h) (Set.disjoint_empty _)
  rw [Prog.bind_eq, Prog.bind_assoc] at this
  simpa [Prog.bind_eq] using this

/-- `x // y` of nonnegative and positive `int32` scalars. -/
theorem SafeA.idiv1 {w : Nat} {x y : IntV true w} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] (IntV true w) → Prog β}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = x.toInt / y.toInt →
      SafeA E R ∅ C Q log m (k r)) :
    SafeA E R ∅ C Q log m (tl.idiv (Tensor.scalar x) (Tensor.scalar y) >>= k) :=
  SafeA.liftS (safe_idiv1 hx hy fun r hr => safe_ret (hk r hr))

/-- `x % y` of nonnegative and positive `int32` scalars. -/
theorem SafeA.irem1 {w : Nat} {x y : IntV true w} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt)
    {k : Tensor [] (IntV true w) → Prog β}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = x.toInt % y.toInt →
      SafeA E R ∅ C Q log m (k r)) :
    SafeA E R ∅ C Q log m (tl.irem (Tensor.scalar x) (Tensor.scalar y) >>= k) :=
  SafeA.liftS (safe_irem1 hx hy fun r hr => safe_ret (hk r hr))

/-- `x % y` of integer tensors. -/
theorem SafeA.irem {s : Shape} {sg : Bool} {w : Nat} {x y : Tensor s (IntV sg w)}
    {k : Tensor s (IntV sg w) → Prog β}
    (hok : ∀ i, ∃ q, (x.get i).rem? (y.get i) = some q)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, (x.get i).rem? (y.get i) = some (r.get i)) →
      SafeA E R ∅ C Q log m (k r)) :
    SafeA E R ∅ C Q log m (tl.irem x y >>= k) :=
  SafeA.liftS (safe_irem hok fun r hr => safe_ret (hk r hr))

/-- A scalar load of element `i` of a valid buffer. -/
theorem SafeA.load1_buf {d : DType} (b : Buf d) {i : Nat} (hi : i < b.len) (hv : b.Valid m)
    (hR : Mem.span d (b.ptr i).addr ⊆ R) {k : d.Val → Prog β}
    (hk : ∀ v, m.read d (b.ptr i).addr = some v → SafeA E R ∅ C Q log m (k v)) :
    SafeA E R ∅ C Q log m (tl.load1 (b.ptr i) >>= k) :=
  SafeA.liftS (safe_load1_buf b hi hv hR fun v hv' => safe_ret (hk v hv'))

/-- A scalar load at a pointer. -/
theorem SafeA.load1 {d : DType} {p : Ptr d} {v0 : d.Val} (hv : m.read d p.addr = some v0)
    (hR : Mem.span d p.addr ⊆ R) {k : d.Val → Prog β}
    (hk : SafeA E R ∅ C Q log m (k v0)) :
    SafeA E R ∅ C Q log m (tl.load1 p >>= k) :=
  SafeA.liftS (safe_load1 hv hR (safe_ret hk))

/-- A tensor load from the elements of a buffer. -/
theorem SafeA.load_buf {d : DType} {s : Shape} (b : Buf d) {ptrs : Tensor s (Ptr d)}
    {mask : Option (Tensor s Bool)} {other : Option (Tensor s d.Val)} {g : Idx s → Nat}
    {en : Idx s → Bool} (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hv : b.Valid m) (hR : b.elems (Buf.img g en) ⊆ R) {k : Tensor s d.Val → Prog β}
    (hk : ∀ v : Tensor s d.Val,
      (∀ i, en i = true → m.read d (b.ptr (g i : Int)).addr = some (v.get i)) →
      (∀ i, en i = false → ∀ o, other = some o → v.get i = o.get i) →
      SafeA E R ∅ C Q log m (k v)) :
    SafeA E R ∅ C Q log m (tl.load ptrs mask other >>= k) :=
  SafeA.liftS (safe_load_buf b hA hin hv hR fun v h1 h2 => safe_ret (hk v h1 h2))

/-- A conversion with a result for every element. -/
theorem SafeA.convert {d d' : DType} {s : Shape} {x : Tensor s d.Val} {k : Tensor s d'.Val → Prog β}
    (hex : ∀ i, ∃ r, ConvRel E.P d d' (x.get i) r)
    (hk : ∀ r : Tensor s d'.Val, (∀ i, ConvRel E.P d d' (x.get i) (r.get i)) →
      SafeA E R ∅ C Q log m (k r)) :
    SafeA E R ∅ C Q log m (tl.convert d d' x >>= k) :=
  SafeA.liftS (safe_convert hex fun r hr => safe_ret (hk r hr))

/-- `tl.max_contiguous(tl.multiple_of(x, v), C)`. -/
theorem SafeA.max_contiguous_multiple_of {γ : Type} {n : Nat} [HintVal γ] {x : Tensor [n] γ} {C' v : Nat}
    {k : Tensor [n] γ → Prog β} (hc : ContigAlong x 0 C') (hv : FirstsDiv x 0 C' v)
    (hk : SafeA E R ∅ C Q log m (k x)) :
    SafeA E R ∅ C Q log m (tl.max_contiguous_multiple_of x C' v >>= k) :=
  SafeA.liftS (safe_max_contiguous_multiple_of hc hv (safe_ret hk))

/-- `tl.mul` of float tensors. -/
theorem SafeA.mul {s : Shape} {f : Format} {x y : Tensor s (FloatV f)}
    {k : Tensor s (FloatV f) → Prog β}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, FloatV.NaNEq (FloatV.mul (x.get i) (y.get i)) (r.get i)) →
      SafeA E R ∅ C Q log m (k r)) :
    SafeA E R ∅ C Q log m (tl.mul x y >>= k) :=
  SafeA.liftS (safe_mul fun r hr => safe_ret (hk r hr))

/-- `acc += tl.dot(a, b)`. -/
theorem SafeA.dot_add {cfg : DotCfg} {M N K : Nat} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {c z : Tensor [M, N] cfg.acc.Val}
    {k : Tensor [M, N] cfg.acc.Val → Prog β} (hex : ∃ r, E.P.dot cfg a b c r)
    (hk : ∀ r, (E.P.dot cfg a b c r ∨ ∃ d, E.P.dot cfg a b z d ∧
        ∀ i, AccAdd E.fusion cfg.acc (d.get i) (c.get i) (r.get i)) →
      SafeA E R ∅ C Q log m (k r)) :
    SafeA E R ∅ C Q log m (tl.dot_add cfg a b c z >>= k) :=
  SafeA.liftS (safe_dot_add hex fun r hr => safe_ret (hk r hr))

/-- **A loop of private reads**, followed by the rest of the program. The loop writes nothing; the
invariant `I` over the trip number carries the state and the memory. -/
theorem SafeA.forRange {σ : Type} {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init : σ}
    {body : IntV sg w → σ → Prog σ} {k : σ → Prog β} (hrange : tl.RangeOk lo hi step)
    (I : Nat → σ → Mem → Prop) (h0 : I 0 init m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R ∅ (fun st' m₂ => I (t + 1) st' m₂) m₁ (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ →
      SafeA E R ∅ C Q log m₁ (k st)) :
    SafeA E R ∅ C Q log m (tl.forRange lo hi step init body >>= k) := by
  refine SafeA.liftS ?_
  have h := safe_forRange (E := E) (R := R) (m := m) (Q := fun st m₁ => SafeA E R ∅ C Q log m₁ (k st))
    (k := pure) hrange I (fun _ => ∅) (Wk := ∅) (by intro t u _ _; simp) (by simp) h0 hstep
    (fun st m₁ h => safe_ret (hk st m₁ h))
  simpa using h

end Wrap

end TVA
