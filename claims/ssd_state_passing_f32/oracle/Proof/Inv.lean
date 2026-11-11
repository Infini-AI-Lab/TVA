import Claim
import TVA.Proof
import Proof.Num
import Proof.Lanes
import Proof.Trip

/-!
# What a program has done

`slotRead` reads the state held in slot `c` of a lane; `Done` says that slots `1..t` hold values
within the claim's tolerance of `sRef` and slot `0` the initial state; `Pres` that no byte has
vanished.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The value in slot `c` of lane `m` of head `h` in memory `mem`: `out[0, c, h, m]`, or
`final_states[0, h, m]` for the last slot. -/
def slotRead (d : Inst) (h c m : Nat) (mem : Mem) : Option F32 :=
  if c < d.nchunks then mem.read .f32 (d.out.ptr (stIdx d c h m : Int)).addr
  else mem.read .f32 (d.finalst.ptr (finIdx d h m : Int)).addr

/-- Slot `0` of every lane holds the initial state (the `float32` `+0` without initial states). -/
def Zero (d : Inst) (init : Mem) (pm h : Nat) (mem : Mem) : Prop :=
  ∀ m ∈ lanes d pm, ∃ v, slotRead d h 0 m mem = some v ∧
    (if d.hasInit then ∃ a, init.read .f32 (d.initst.ptr (inIdx d 0 h m : Int)).addr = some a ∧
        FloatV.NaNEq a v
      else v = FloatV.ofPat 0)

/-- **Slots done**: slot `0` holds the initial state and slots `1..t` hold a value within the
tolerance of the exact recurrence. -/
def Done (d : Inst) (init : Mem) (pm h t : Nat) (mem : Mem) : Prop :=
  Zero d init pm h mem ∧ ∀ c, 1 ≤ c → c ≤ t → ∀ m ∈ lanes d pm,
    ∃ v, slotRead d h c m mem = some v ∧ OutO init d h m c v

/-- No byte present at the start has disappeared. -/
def Pres (init mem : Mem) : Prop := ∀ x, init x ≠ none → mem x ≠ none

/-- The bytes of an element of slot `c` lie in the slot's footprint. -/
theorem slot_span_sub {d : Inst} {h pm c m : Nat} (hm : m ∈ lanes d pm) :
    (if c < d.nchunks then Mem.span DType.f32 (d.out.ptr (stIdx d c h m : Int)).addr
      else Mem.span DType.f32 (d.finalst.ptr (finIdx d h m : Int)).addr) ⊆ slotW d h pm c := by
  unfold slotW
  by_cases h1 : c < d.nchunks
  · simp only [h1, ite_true]
    exact d.out.span_sub_elems ⟨m, hm, rfl⟩
  · simp only [h1, ite_false]
    exact d.finalst.span_sub_elems ⟨m, hm, rfl⟩

/-- Reading an element of slot `c'` is unchanged by a store that changes only the bytes of
another slot `c`. -/
theorem slotRead_frame {d : Inst} (hd : d.ok) {h pm c c' m : Nat} {mem mem' : Mem}
    (hh : h < d.nheads) (hc : c ≤ d.nchunks) (hc' : c' ≤ d.nchunks) (hne : c' ≠ c)
    (hm : m ∈ lanes d pm) (hf : ∀ x, x ∉ slotW d h pm c → mem' x = mem x) :
    slotRead d h c' m mem' = slotRead d h c' m mem := by
  unfold slotRead
  have hs := slot_span_sub (h := h) (c := c') hm
  have hdis := slotW_disjoint hd (pm := pm) hh hc' hc hne
  by_cases h1 : c' < d.nchunks
  · simp only [h1, ite_true] at hs ⊢
    exact Mem.read_congr fun x hx => hf x (Set.disjoint_left.mp hdis (hs hx))
  · simp only [h1, ite_false] at hs ⊢
    exact Mem.read_congr fun x hx => hf x (Set.disjoint_left.mp hdis (hs hx))

/-- **The invariant of the loop with initial states** before trip `t`: the inputs and their bytes
are untouched, the state's lanes are within tolerance of `sRef t`, the pointers and counters are
those of trip `t`, and the slots `0..t` are done. -/
def InvI (d : Inst) (init : Mem) (pm h t : Nat) (st : StI BL) (mem : Mem) : Prop :=
  (∀ x ∈ Rin d, mem x = init x) ∧ Pres init mem ∧
  (∀ k : Fin BL, pm * BL + k < d.dim → OutO init d h (pm * BL + k) t (st.1.get1 k)) ∧
  (∀ k : Fin BL, st.2.2.1.get1 k = d.states.ptr (stIdx d t h (pm * BL + k) : Int)) ∧
  st.2.2.2.1 = d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int) ∧
  (∀ k : Fin BL, st.2.2.2.2.1.get1 k = d.out.ptr (stIdx d (t + 1) h (pm * BL + k) : Int)) ∧
  st.2.2.2.2.2.1.toInt = (prevReq d t : Int) ∧
  st.2.2.2.2.2.2.toInt = ((t + pc d (prevReq d t) : Nat) : Int) ∧
  Done d init pm h t mem

/-- **The invariant of the loop without initial states** before trip `t`. -/
def InvN (d : Inst) (init : Mem) (pm h t : Nat) (st : StN BL) (mem : Mem) : Prop :=
  (∀ x ∈ Rin d, mem x = init x) ∧ Pres init mem ∧
  (∀ k : Fin BL, pm * BL + k < d.dim → OutO init d h (pm * BL + k) t (st.1.get1 k)) ∧
  (∀ k : Fin BL, st.2.1.get1 k = d.states.ptr (stIdx d t h (pm * BL + k) : Int)) ∧
  st.2.2.1 = d.dA.ptr (dAIdx d h t (d.chunk - 1) : Int) ∧
  (∀ k : Fin BL, st.2.2.2.1.get1 k = d.out.ptr (stIdx d (t + 1) h (pm * BL + k) : Int)) ∧
  st.2.2.2.2.toInt = (prevReq d t : Int) ∧
  Done d init pm h t mem

/-- Slot `t + 1` filled in leaves the earlier slots as they were: `Done` extends. -/
theorem Done.succ {d : Inst} (hd : d.ok) {init : Mem} {pm h t : Nat} (hh : h < d.nheads)
    (ht : t < d.nchunks) {m₁ m' : Mem} (hdone : Done d init pm h t m₁)
    (hframe : ∀ x, x ∉ slotW d h pm (t + 1) → m' x = m₁ x)
    (hnew : ∀ m ∈ lanes d pm, ∃ v, slotRead d h (t + 1) m m' = some v ∧ OutO init d h m (t + 1) v) :
    Done d init pm h (t + 1) m' := by
  obtain ⟨hz, hs⟩ := hdone
  refine ⟨fun m hm => ?_, fun c hc1 hc2 m hm => ?_⟩
  · obtain ⟨v, hv, hz'⟩ := hz m hm
    refine ⟨v, ?_, hz'⟩
    rw [slotRead_frame hd hh (by omega) (by omega) (by omega) hm hframe]
    exact hv
  · by_cases hct : c = t + 1
    · subst hct; exact hnew m hm
    · obtain ⟨v, hv, hz'⟩ := hs c hc1 (by omega) m hm
      refine ⟨v, ?_, hz'⟩
      rw [slotRead_frame hd hh (by omega) (by omega) hct hm hframe]
      exact hv

end Claims.SsdStatePassing
