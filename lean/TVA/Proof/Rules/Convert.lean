import TVA.Proof.Safe

/-!
# Rules: conversions

The proof rules for `TVA.Triton.Convert`. Each rule is `Safe … (op >>= k)`; its last premise is
the rest `k`, for every result the conversion may give.

* `safe_convert`: `tl.convert d d' x` (`x.to(d')` with the default rounding, and every implicit
  conversion of an operand): each element as `ConvRel` allows on the platform.
* Between standard float formats: `safe_cvt` (one rounding, up to NaN payload).
* Into and out of fp8: `safe_cvt8` (the platform's conversion), `safe_cvtFrom8` (exact on finite
  elements).
* Between integers and floats: `safe_itof` (`ItofRel`), `safe_ftoi` (`FtoiRel`: truncation, the
  element finite and in range).

How this fits: model → `Safe` per program (the rules of `TVA.Proof.Rules`) → `correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {s : Shape} {f : Format}

/-- **`tl.convert d d' x`**: when every element has a result `ConvRel` allows on the platform, the
rest must be safe for every tensor of such results. -/
theorem safe_convert {d d' : DType} {x : Tensor s d.Val} {k : Tensor s d'.Val → Prog α}
    (hex : ∀ i, ∃ r, ConvRel E.P d d' (x.get i) r)
    (hk : ∀ r : Tensor s d'.Val, (∀ i, ConvRel E.P d d' (x.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.convert d d' x >>= k) := by
  simp only [tl.convert, bind_eq, Prog.bind]
  refine .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩,
    fun i => Classical.choose_spec (hex i)⟩ ?_)
  intro r hr
  simpa [Prog.bind] using hk r hr

/-- **`x.to(dtype, fp_downcast_rounding=rnd)` between standard float formats**: the pair is one
Triton compiles (`hok`); each element is the rounding of `x`, up to NaN payload. -/
theorem safe_cvt {g : Format} {x : Tensor s (FloatV f)} {rnd : RoundMode}
    {k : Tensor s (FloatV g) → Prog α} (hok : f.Std ∧ g.Std ∧ tl.RoundOk f g rnd)
    (hk : ∀ r : Tensor s (FloatV g),
      (∀ i, FloatV.NaNEq (FloatV.cvtWith rnd (x.get i)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cvt x rnd >>= k) := by
  simp only [tl.cvt, tl.upToNaN, bind_eq, Prog.bind]
  exact .pick ⟨(), hok⟩ fun _ _ => .pick ⟨_, fun _ => Or.inl rfl⟩ hk

/-- **`x.to(fp8)`**: the pair is one Triton compiles (`hok`), a platform result exists for every
element (`hex`, from the device facts); the rest must be safe for every result the platform
allows. -/
theorem safe_cvt8 {g : Format} {x : Tensor s (FloatV f)} {rnd : RoundMode}
    {k : Tensor s (FloatV g) → Prog α}
    (hok : (f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ tl.RoundOk f g rnd)
    (hex : ∀ i, ∃ r, E.P.cvt f g rnd (x.get i) r)
    (hk : ∀ r : Tensor s (FloatV g), (∀ i, E.P.cvt f g rnd (x.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cvt8 x rnd >>= k) := by
  simp only [tl.cvt8, bind_eq, Prog.bind]
  exact .pick ⟨(), hok⟩ fun _ _ =>
    .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk)

/-- **`x.to(dtype)` out of fp8**: finite elements convert exactly. -/
theorem safe_cvtFrom8 {g : Format} {x : Tensor s (FloatV f)} {k : Tensor s (FloatV g) → Prog α}
    (hok : f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32))
    (hk : ∀ r : Tensor s (FloatV g),
      (∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cvtFrom8 x >>= k) := by
  simp only [tl.cvtFrom8, bind_eq, Prog.bind]
  exact .pick ⟨(), hok⟩ fun _ _ =>
    .pick ⟨x.map FloatV.cvt, fun i _ => by simp [Tensor.get_map]⟩ hk

/-- **`x.to(float)` of an integer tensor**: each element is one of the roundings `tl.ItofRel`
allows. -/
theorem safe_itof {sg : Bool} {w : Nat} {x : Tensor s (IntV sg w)}
    {k : Tensor s (FloatV f) → Prog α} (hstd : f.Std)
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, tl.ItofRel f (x.get i).toInt (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.itof x >>= k) := by
  simp only [tl.itof, bind_eq, Prog.bind]
  exact .pick ⟨(), hstd⟩ fun _ _ =>
    .pick ⟨x.map fun a => FloatV.ofRat (a.toInt : Rat),
      fun i => Or.inl (by simp [Tensor.get_map])⟩ hk

/-- **`x.to(int)` of a float tensor**: every element is finite and in range (`FtoiRel` has a
result); the result truncates each element toward zero. -/
theorem safe_ftoi {sg : Bool} {w : Nat} {x : Tensor s (FloatV f)}
    {k : Tensor s (IntV sg w) → Prog α} (hstd : f.Std)
    (hex : ∀ i, ∃ r, FtoiRel sg w (x.get i) r)
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, FtoiRel sg w (x.get i) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.ftoi x >>= k) := by
  simp only [tl.ftoi, bind_eq, Prog.bind]
  exact .pick ⟨(), hstd⟩ fun _ _ =>
    .pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk

end TVA
