import TVA.Proof.Shared.Replay
import TVA.Device.Shapes
import TVA.Device.Tolerance
import TVA.Proof.Dot.Loop

/-!
# Float accumulation through shared atomics

Split-K GEMM and stream-K add `float32` partial results into one output element with
`tl.atomic_add`, from several programs, in an order nobody controls. Each addition rounds
(`Device.AtomicAddAcc`: within `rel·|x + w| + abs` of the exact `x + w`), and rounding does not
commute, so the final value depends on the order. It is still within a bound of the exact sum that
holds for every order: after `n` additions of operands `w₁, …, wₙ` to `x₀`, the cell is within
`DotLoop.bound rel abs n (|x₀| + Σ|wᵢ|)` of `x₀ + Σ wᵢ` (`replay_err`), the bound of `n` steps of
an accumulation (`TVA.Device.Tolerance`), whatever the order of the steps.

The operands come from the programs' logs paired with them (`Paired`, `TVA.Proof.Shared.Replay`):
each program pairs its steps on the cell with its own operands (`accStep_atomicAdd`) and its steps
on other cells with none (`accStep_other`); `replay_err_shared` merges them along the interleaving
and gives the bound in terms of every program's operands.
-/
namespace TVA

open Classical

/-- Step `s`, applied to the cell `(d, a)` holding `x`, leaves a value `y` with `r x y`. -/
def StepRel (d : DType) (a : Int) (r : d.Val → d.Val → Prop) (s : AtomicStep) : Prop :=
  ∀ (m : Mem) (x : d.Val), m.read d a = some x → m.read s.d s.a = some s.old →
    ∃ y, (m.write s.d s.a s.new).read d a = some y ∧ r x y

/-- A step on the cell itself relates the cell's values as `r` relates its old and new values.
Use it for a program's own steps on the cell. -/
theorem stepRel_self {d : DType} {a : Int} {old new : d.Val} {r : d.Val → d.Val → Prop}
    (h : r old new) : StepRel d a r ⟨d, a, old, new⟩ := by
  intro m x hx hold
  simp only at hold
  rw [hx] at hold
  cases hold
  exact ⟨new, Mem.read_write_self (Mem.writable_of_read hx) new, h⟩

/-- A step on bytes disjoint from the cell leaves the cell's value, which `r` must allow. -/
theorem stepRel_other {d : DType} {a : Int} {s : AtomicStep} {r : d.Val → d.Val → Prop}
    (hr : ∀ x, r x x) (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) : StepRel d a r s := by
  intro m x hx _
  refine ⟨x, ?_, hr x⟩
  rw [Mem.read_congr (fun y hy => Mem.write_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdis h hy)), hx]

/-- What one accumulation step promises: from a `good` value `x`, when `|x + w| ≤ B`, a `good`
value within `ρ·|x + w| + α` of `x + w`. -/
def AddsApprox {β : Type} (val : β → ℚ) (good : β → Prop) (ρ α B w : ℚ) (x y : β) : Prop :=
  good x → |val x + w| ≤ B → good y ∧ |val y - (val x + w)| ≤ ρ * |val x + w| + α

/-- A step that leaves the value adds `0` exactly (for a step paired with the operand `0`). -/
theorem addsApprox_refl {β : Type} (val : β → ℚ) (good : β → Prop) {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α)
    (x : β) : AddsApprox val good ρ α B 0 x x := fun hx _ =>
  ⟨hx, by rw [add_zero, sub_self, abs_zero]; positivity⟩

/-- One logged step with its operand, for float accumulation into the cell `(d, a)`: with
`some w`, the step adds `w` to the cell up to `ρ·|x + w| + α` (`AddsApprox`; `accStep_atomicAdd`);
with `none`, it leaves the cell alone (`accStep_other`). -/
def AccStep (d : DType) (a : Int) (val : d.Val → ℚ) (good : d.Val → Prop) (ρ α B : ℚ) :
    AtomicStep → Option ℚ → Prop
  | s, none => MapsCell d a id s
  | s, some w => StepRel d a (AddsApprox val good ρ α B w) s

/-- A step on bytes disjoint from the cell is an `AccStep` without operand. Use it for a
program's steps on its other shared cells. -/
theorem accStep_other {d : DType} {a : Int} {val : d.Val → ℚ} {good : d.Val → Prop} {ρ α B : ℚ}
    {s : AtomicStep} (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) :
    AccStep d a val good ρ α B s none :=
  mapsCell_other hdis

/-- `DotLoop.bound` grows with the number of steps and the magnitude. -/
private theorem bound_mono {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t t' : ℕ} {B B' : ℚ} (ht : t ≤ t')
    (hB : 0 ≤ B) (hBB : B ≤ B') : DotLoop.bound ρ α t B ≤ DotLoop.bound ρ α t' B' := by
  unfold DotLoop.bound
  have h1 : 1 ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have h2 : (1 + ρ) ^ t ≤ (1 + ρ) ^ t' := pow_le_pow_right₀ (by linarith) ht
  have ht' : (t : ℚ) ≤ t' := by exact_mod_cast ht
  have e1 : ((1 + ρ) ^ t - 1) * B ≤ ((1 + ρ) ^ t' - 1) * B' := mul_le_mul (by linarith) hBB hB (by linarith)
  have e2 : (t : ℚ) * (1 + ρ) ^ t * α ≤ t' * (1 + ρ) ^ t' * α :=
    mul_le_mul_of_nonneg_right (mul_le_mul ht' h2 (by positivity) (by positivity)) hα
  linarith

/-- `replay_err` from a partial accumulation: `t` steps done, exact sum `St`, magnitude `Bt`. -/
private theorem replay_err_aux {d : DType} {a : Int} (val : d.Val → ℚ) (good : d.Val → Prop)
    {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (Btot : ℚ) (n : ℕ) :
    ∀ (L : List (AtomicStep × Option ℚ)) (m m' : Mem) (x : d.Val) (t : ℕ) (St Bt : ℚ),
      Replay m (L.map Prod.fst) m' → (∀ p ∈ L, AccStep d a val good ρ α B p.1 p.2) →
      m.read d a = some x → good x → |St| ≤ Bt →
      |val x - St| ≤ DotLoop.bound ρ α t Bt → t + (L.filterMap Prod.snd).length ≤ n →
      Bt + ((L.filterMap Prod.snd).map fun w => |w|).sum ≤ Btot → Btot + DotLoop.bound ρ α n Btot ≤ B →
      ∃ y, m'.read d a = some y ∧ good y ∧
        |val y - (St + (L.filterMap Prod.snd).sum)| ≤
          DotLoop.bound ρ α (t + (L.filterMap Prod.snd).length)
            (Bt + ((L.filterMap Prod.snd).map fun w => |w|).sum)
  | [], m, m', x, t, St, Bt, hrep, _, h0, hx, _, he, _, _, _ => by
    cases hrep
    exact ⟨x, h0, hx, by simpa using he⟩
  | (s, none) :: L, m, m', x, t, St, Bt, hrep, hl, h0, hx, hSt, he, hn, hB, hroom => by
    cases hrep with
    | cons hr hrest =>
      have h1 : (m.write s.d s.a s.new).read d a = some x := hl (s, none) List.mem_cons_self m x h0 hr
      have hc : ((s, none) :: L).filterMap Prod.snd = L.filterMap Prod.snd := rfl
      rw [hc] at hn hB ⊢
      exact replay_err_aux val good hρ hα Btot n L _ m' x t St Bt hrest
        (fun p hp => hl p (List.mem_cons_of_mem _ hp)) h1 hx hSt he hn hB hroom
  | (s, some w) :: L, m, m', x, t, St, Bt, hrep, hl, h0, hx, hSt, he, hn, hB, hroom => by
    cases hrep with
    | cons hr hrest =>
      obtain ⟨y, hy, hxy⟩ := (hl (s, some w) List.mem_cons_self : StepRel d a _ s) m x h0 hr
      have hw0 : 0 ≤ ((L.filterMap Prod.snd).map fun w => |w|).sum :=
        List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨w', -, rfl⟩; exact abs_nonneg _)
      have hBt0 : 0 ≤ Bt := (abs_nonneg _).trans hSt
      have hc : ((s, some w) :: L).filterMap Prod.snd = w :: L.filterMap Prod.snd := rfl
      rw [hc] at hB hn ⊢
      simp only [List.map_cons, List.sum_cons, List.length_cons] at hB hn ⊢
      have hxabs : |val x| ≤ Bt + DotLoop.bound ρ α t Bt := by
        have := abs_sub_abs_le_abs_sub (val x) St
        linarith
      have hbt : DotLoop.bound ρ α t Bt ≤ DotLoop.bound ρ α n Btot :=
        bound_mono hρ hα (by omega) hBt0 (by linarith [abs_nonneg w])
      have hroom1 : |val x + w| ≤ B := by
        have := abs_add_le (val x) w
        linarith [abs_nonneg w]
      obtain ⟨hgy, herr⟩ := hxy hx hroom1
      have hstep : |val y - (St + w)| ≤ DotLoop.bound ρ α (t + 1) (Bt + |w|) :=
        DotLoop.accum_step hρ hα hSt le_rfl he (by
          have := abs_add_le (val x) w
          have : ρ * |val x + w| ≤ ρ * (|val x| + |w|) := mul_le_mul_of_nonneg_left this hρ
          linarith)
      have hSt' : |St + w| ≤ Bt + |w| := (abs_add_le _ _).trans (by linarith)
      obtain ⟨z, hz, hgz, hze⟩ := replay_err_aux val good hρ hα Btot n L _ m' y (t + 1) (St + w)
        (Bt + |w|) hrest (fun p hp => hl p (List.mem_cons_of_mem _ hp)) hy hgy hSt' hstep
        (by omega) (by linarith) hroom
      refine ⟨z, hz, hgz, ?_⟩
      have e1 : St + (w + (L.filterMap Prod.snd).sum) = St + w + (L.filterMap Prod.snd).sum := by ring
      have e2 : Bt + (|w| + ((L.filterMap Prod.snd).map fun w => |w|).sum) =
          Bt + |w| + ((L.filterMap Prod.snd).map fun w => |w|).sum := by ring
      rw [e1, e2, show t + ((L.filterMap Prod.snd).length + 1) = t + 1 + (L.filterMap Prod.snd).length by omega]
      exact hze

/-- **Float accumulation through atomics, per position.** If every step of a replay, paired with
its operand, either adds `w` to the cell up to `ρ·|x + w| + α` (`some w`; `accStep_atomicAdd`
from `Device.AtomicAddAcc`) or leaves it alone (`none`; `accStep_other`), the cell ends within
`DotLoop.bound ρ α n (|x₀| + Σ|w|)` of `x₀ + Σ w`, the sums over the operands of the steps on the
cell, `L.filterMap Prod.snd` (the `w` of each step paired with `some w`), and `n` their number, in
whatever order the programs' atomics interleaved, given room for that bound (`… ≤ B`). Across
programs, `replay_err_shared`. -/
theorem replay_err {d : DType} {a : Int} (val : d.Val → ℚ) (good : d.Val → Prop)
    {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {m m' : Mem} {L : List (AtomicStep × Option ℚ)}
    {x : d.Val} (hrep : Replay m (L.map Prod.fst) m')
    (hl : ∀ p ∈ L, AccStep d a val good ρ α B p.1 p.2) (h0 : m.read d a = some x) (hx : good x)
    (hroom : |val x| + ((L.filterMap Prod.snd).map fun w => |w|).sum +
      DotLoop.bound ρ α (L.filterMap Prod.snd).length
        (|val x| + ((L.filterMap Prod.snd).map fun w => |w|).sum) ≤ B) :
    ∃ y, m'.read d a = some y ∧ good y ∧
      |val y - (val x + (L.filterMap Prod.snd).sum)| ≤
        DotLoop.bound ρ α (L.filterMap Prod.snd).length
          (|val x| + ((L.filterMap Prod.snd).map fun w => |w|).sum) := by
  obtain ⟨y, hy, hgy, he⟩ := replay_err_aux val good hρ hα _ (L.filterMap Prod.snd).length L m m' x 0
    (val x) |val x| hrep hl h0 hx le_rfl (by simp [DotLoop.bound]) (by omega) le_rfl hroom
  exact ⟨y, hy, hgy, by simpa using he⟩

/-- **Float accumulation from several programs, in any order.** Suppose each program's part of
the interleaved log `G` (from `correct_of_safeA`) is paired with its operands `ops i` by `AccStep`: its
steps on the cell `(d, a)` each add their operand up to `ρ·|x + w| + α`, its other steps leave the
cell alone. Then the replay leaves the cell within `DotLoop.bound ρ α n (|x₀| + A)` of
`x₀ + S`, where `S = Σᵢ Σ (ops i)` is the sum of every program's operands, `A` the sum of their
magnitudes and `n` their number, given room for that bound. -/
theorem replay_err_shared {d : DType} {a : Int} (val : d.Val → ℚ) (good : d.Val → Prop)
    {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {N : Nat} {G : List (Fin N × AtomicStep)}
    {ops : Fin N → Multiset ℚ} {m m' : Mem} {x : d.Val} (hrep : Replay m (G.map Prod.snd) m')
    (hp : ∀ i, Paired (AccStep d a val good ρ α B) (proj G i) (ops i))
    (h0 : m.read d a = some x) (hx : good x)
    (hroom : |val x| + (∑ i, ((ops i).map fun w => |w|).sum) +
      DotLoop.bound ρ α (∑ i, Multiset.card (ops i))
        (|val x| + ∑ i, ((ops i).map fun w => |w|).sum) ≤ B) :
    ∃ y, m'.read d a = some y ∧ good y ∧
      |val y - (val x + ∑ i, (ops i).sum)| ≤
        DotLoop.bound ρ α (∑ i, Multiset.card (ops i)) (|val x| + ∑ i, ((ops i).map fun w => |w|).sum) := by
  obtain ⟨L, hm, ho, hr⟩ := Paired.merge hp
  have hS : (L.filterMap Prod.snd).sum = ∑ i, (ops i).sum := by
    rw [← Multiset.sum_coe, ho]
    exact map_sum Multiset.sumAddMonoidHom ops Finset.univ
  have hA : ((L.filterMap Prod.snd).map fun w => |w|).sum = ∑ i, ((ops i).map fun w => |w|).sum := by
    rw [← Multiset.sum_coe, ← Multiset.map_coe, ho,
      show Multiset.map (fun w => |w|) (∑ i, ops i) = ∑ i, (ops i).map fun w => |w| from
        map_sum (Multiset.mapAddMonoidHom fun w : ℚ => |w|) ops Finset.univ]
    exact map_sum Multiset.sumAddMonoidHom _ Finset.univ
  have hn : (L.filterMap Prod.snd).length = ∑ i, Multiset.card (ops i) := by
    rw [← Multiset.coe_card, ho, Multiset.card_sum]
  have h := replay_err (a := a) val good hρ hα (hm ▸ hrep) hr h0 hx (by rw [hA, hn]; exact hroom)
  rwa [hS, hA, hn] at h

/-- **A float `atomic_add` step on the cell**, from `Device.AtomicAddAcc`: a `float32` step
`⟨.f32, a, old, new⟩` whose new value the platform's `atomicAdd` gave for the finite operand `v`
adds `v` up to `rel·|x + v| + abs`, with room `maxVal`. -/
theorem stepRel_atomicAdd {P : Platform} {rel abs : ℚ}
    (hacc : Device.AtomicAddAcc P .binary32 rel abs) {a : Int} {old v new : F32}
    (hnew : P.atomicAdd .binary32 old v new) (hv : v.isFinite = true) :
    StepRel .f32 a (AddsApprox FloatV.toRat (fun x : F32 => x.isFinite = true) rel abs
      Format.binary32.maxVal v.toRat) ⟨.f32, a, old, new⟩ :=
  stepRel_self fun ho hroom => hacc old v new hnew ho hv hroom

/-- `stepRel_atomicAdd` as an `AccStep` with the operand `v`: what a program proves of each lane
of its float `tl.atomic_add` on the cell, to pair its log with its operands (`Paired.lanes`). -/
theorem accStep_atomicAdd {P : Platform} {rel abs : ℚ}
    (hacc : Device.AtomicAddAcc P .binary32 rel abs) {a : Int} {old v new : F32}
    (hnew : P.atomicAdd .binary32 old v new) (hv : v.isFinite = true) :
    AccStep .f32 a FloatV.toRat (fun x : F32 => x.isFinite = true) rel abs
      Format.binary32.maxVal ⟨.f32, a, old, new⟩ (some v.toRat) :=
  stepRel_atomicAdd hacc hnew hv

end TVA
