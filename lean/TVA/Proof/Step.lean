import TVA.Proof.Mem

/-!
# One step, in two memories

Two facts about one `Step` of a program (`TVA.Launch.Step`), whatever the step is:

* `Step.frame`: a step changes only the bytes its event writes;
* `Step.transfer`: a step can be taken the same way in any memory that agrees on the bytes its
  event reads and writes, and leads to that memory with the step's writes (`Mem.mix`).

`Mem.mix m₀ m S` is `m` on the bytes `S` and `m₀` elsewhere: the memory a program would see had
only its own writes `S` happened.

How this fits: `Safe` (`TVA.Proof.Safe`) is proved of a program run alone, from its own memory.
These two facts let `Safe.stepT` replay each step of the program inside a launch, where other
programs change other bytes, and so carry `Safe` along any interleaving (`TVA.Proof.Compose`).
-/
namespace TVA

/-! ## Memory mixes -/

open Classical in
/-- `m` on the bytes `S`, `m₀` elsewhere. -/
noncomputable def Mem.mix (m₀ m : Mem) (S : Set Int) : Mem := fun x => if x ∈ S then m x else m₀ x

namespace Mem

/-- Mixing in nothing changes nothing. -/
@[simp] theorem mix_empty (m₀ m : Mem) : Mem.mix m₀ m ∅ = m₀ := by
  funext x; simp [Mem.mix]

/-- A memory that differs from `m₀` only on `S` is `m₀` mixed with it on `S`. -/
theorem mix_of_frame {m₀ m : Mem} {S : Set Int} (h : ∀ x, x ∉ S → m x = m₀ x) :
    Mem.mix m₀ m S = m := by
  funext x
  by_cases hx : x ∈ S
  · simp [Mem.mix, hx]
  · simp [Mem.mix, hx, h x hx]

/-- Inside `S`, a mix is the second memory. -/
theorem mix_in {m₀ m : Mem} {S : Set Int} {x : Int} (hx : x ∈ S) : Mem.mix m₀ m S x = m x := by
  simp [Mem.mix, hx]

/-- Outside `S`, a mix is the first memory. -/
theorem mix_out {m₀ m : Mem} {S : Set Int} {x : Int} (hx : x ∉ S) :
    Mem.mix m₀ m S x = m₀ x := by
  simp [Mem.mix, hx]

/-- Mixing in a write on its own bytes is that write. -/
theorem mix_write_self (m₀ m : Mem) (d : DType) (a : Int) (v : d.Val) :
    Mem.mix m₀ (m.write d a v) (Mem.span d a) = m₀.write d a v := by
  funext x
  by_cases hx : x ∈ Mem.span d a
  · rw [mix_in hx]; exact write_span_eq _ _ _ _ _ hx
  · rw [mix_out hx]; exact (write_frame _ _ _ _ hx).symm

/-- A write and a mix on disjoint bytes commute. -/
theorem mix_write {m₀ m' : Mem} {T : Set Int} {d : DType} {a : Int} (v : d.Val)
    (hT : Disjoint (Mem.span d a) T) :
    Mem.mix (m₀.write d a v) m' T = (Mem.mix m₀ m' T).write d a v := by
  funext x
  by_cases hx : x ∈ Mem.span d a
  · have hxT : x ∉ T := Set.disjoint_left.mp hT hx
    rw [mix_out hxT]; exact write_span_eq _ _ _ _ _ hx
  · rw [write_frame _ _ _ _ hx]
    by_cases hxT : x ∈ T
    · rw [mix_in hxT, mix_in hxT]
    · rw [mix_out hxT, mix_out hxT, write_frame _ _ _ _ hx]

end Mem

/-- Mixing in a store on its own bytes is that store, when the two memories agree there. -/
theorem writeAll_mix {m₀ m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
    (vals : Tensor s d.Val) (order : List (Idx s)) (h : ∀ x ∈ accBytes d addrs, m₀ x = m x) :
    Mem.mix m₀ (writeAll m d addrs vals order) (accBytes d addrs) =
      writeAll m₀ d addrs vals order := by
  funext x
  by_cases hx : x ∈ accBytes d addrs
  · rw [Mem.mix_in hx]
    exact writeAll_agree (fun y hy => (h y hy).symm) addrs vals order x hx
  · rw [Mem.mix_out hx]; exact (writeAll_frame m₀ addrs vals order hx).symm

/-! ## One step -/

/-- **A step changes only the bytes its event writes.** -/
theorem Step.frame {α : Type} {E : Env} {m m' : Mem} {p p' : Prog α} {e : Event}
    (h : Step E m p (.next m' p' e)) : ∀ x, x ∉ e.writes → m' x = m x := by
  generalize ho : (Out.next m' p' e) = o at h
  induction h generalizing m' p' e with
  | store order _ _ => cases ho; exact fun x hx => writeAll_frame _ _ _ _ hx
  | rmw old new _ _ => cases ho; exact fun x hx => Mem.write_frame _ _ _ _ hx
  | load => cases ho; exact fun _ _ => rfl
  | pick => cases ho; exact fun _ _ => rfl
  | plat => cases ho; exact fun _ _ => rfl
  | loop => cases ho; exact fun _ _ => rfl
  | env => cases ho; exact fun _ _ => rfl
  | defer _ _ ih => cases ho; exact ih default rfl
  | loadUB => cases ho
  | storeUB => cases ho
  | rmwUB => cases ho
  | pickUB => cases ho
  | platUB => cases ho
  | ub => cases ho

/-- **Transfer of one step.** A step can be taken the same way in any memory that agrees on the
bytes its event reads and writes; the result is that memory with the step's writes. -/
theorem Step.transfer {α : Type} {E : Env} {m m₀ m' : Mem} {p p' : Prog α} {e : Event}
    (h : Step E m p (.next m' p' e)) (hag : ∀ x ∈ e.reads ∪ e.writes, m₀ x = m x) :
    Step E m₀ p (.next (Mem.mix m₀ m' e.writes) p' e) := by
  generalize ho : (Out.next m' p' e) = o at h
  induction h generalizing m' p' e with
  | load v hok =>
    cases ho
    simpa using Step.load v ((loadOk_congr (fun x hx => hag x (Or.inl hx))).mpr hok)
  | store order hok hperm =>
    cases ho
    have hag' : ∀ x ∈ accBytes _ _, m₀ x = _ := fun x hx => hag x (Or.inr hx)
    simp only
    rw [writeAll_mix _ _ hag']
    exact .store order ((storeOk_congr hag').mpr hok) hperm
  | rmw old new hr hu =>
    cases ho
    simp only
    rw [Mem.mix_write_self]
    exact .rmw old new ((Mem.read_congr fun x hx => hag x (Or.inl hx)).trans hr) hu
  | pick b hb => cases ho; simpa [Event.silent] using Step.pick b hb
  | plat r hr => cases ho; simpa [Event.silent] using Step.plat r hr
  | loop => cases ho; simpa [Event.silent] using Step.loop
  | env => cases ho; simpa [Event.silent] using Step.env
  | defer _ hc ih => cases ho; exact .defer (fun v => ih v hag rfl) hc
  | loadUB => cases ho
  | storeUB => cases ho
  | rmwUB => cases ho
  | pickUB => cases ho
  | platUB => cases ho
  | ub => cases ho

end TVA
