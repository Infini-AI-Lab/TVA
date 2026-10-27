import TVA.Lemmas.Safe
import TVA.Sem.Launch
import Mathlib.Order.WellFounded
import Mathlib.Data.Fintype.Lattice

/-!
# Independent programs

**Theorem `independent`.** Suppose each program of a launch, run *alone* from the initial memory,
is `Safe` with read footprint `R i` and write footprint `W i`, and ends in a memory satisfying
`Q i`. Suppose also that no program writes a byte another program reads or writes. Then:

* the launch is well behaved: no interleaving reaches undefined behaviour or a data race;
* it cannot hang;
* every outcome is a normal end whose final memory agrees, on each program's footprint, with a
  memory satisfying that program's `Q i`, and equals the initial memory wherever no program
  writes.

So for most Triton kernels (each program reads inputs and writes its own output tile) the proof
reasons about one program at a time, sequentially, on the initial memory.
-/
namespace TVA

open Classical

/-- Every step either leaves the program's state alone or is a step of it. -/
theorem Step.frame {α : Type} {E : Env} {m m' : Mem} {p p' : Prog α} {e : Event}
    (h : Step E m p (.next m' p' e)) : ∀ x, x ∉ e.writes → m' x = m x := by
  cases h with
  | store order _ _ => exact fun x hx => writeAll_frame _ _ _ _ hx
  | rmw old new _ _ => exact fun x hx => Mem.write_frame _ _ _ _ hx
  | load => exact fun _ _ => rfl
  | pick => exact fun _ _ => rfl
  | loop => exact fun _ _ => rfl
  | env => exact fun _ _ => rfl

/-- A sequence that, at each step, stays put or descends in a relation that is accessible from
its start, descends only finitely often. -/
theorem eventually_no_descent {β : Type*} {r : β → β → Prop} (x : Nat → β)
    (hstep : ∀ n, x (n + 1) = x n ∨ r (x (n + 1)) (x n)) :
    ∀ a, Acc r a → ∀ n₀, x n₀ = a → ∃ T, n₀ ≤ T ∧ ∀ m, T ≤ m → ¬ r (x (m + 1)) (x m) := by
  intro a ha
  induction ha with
  | intro a _ ih =>
    intro n₀ hn
    by_cases hc : ∃ m, n₀ ≤ m ∧ r (x (m + 1)) (x m)
    · let m := Nat.find hc
      have hm : n₀ ≤ m ∧ r (x (m + 1)) (x m) := Nat.find_spec hc
      have hconst : ∀ k, n₀ ≤ k → k ≤ m → x k = a := by
        intro k hk hkm
        induction k with
        | zero => have : n₀ = 0 := by omega
                  subst this; exact hn
        | succ k ihk =>
          by_cases hk' : n₀ ≤ k
          · have hxk := ihk hk' (by omega)
            have hnr : ¬ r (x (k + 1)) (x k) := fun hr => Nat.find_min hc (show k < m by omega) ⟨hk', hr⟩
            rcases hstep k with h | h
            · rw [h, hxk]
            · exact absurd h hnr
          · have : n₀ = k + 1 := by omega
            subst this; exact hn
      have hxm := hconst m hm.1 le_rfl
      have hr : r (x (m + 1)) a := hxm ▸ hm.2
      obtain ⟨T, hT, hTm⟩ := ih (x (m + 1)) hr (m + 1) rfl
      exact ⟨T, by omega, hTm⟩
    · push Not at hc
      exact ⟨n₀, le_rfl, fun m hm => hc m hm⟩

section
variable {N : Nat} (E : Env) (R W : Fin N → Set Int) (Q : Fin N → Mem → Prop) (init : Mem)

/-- Program `i`'s own view of memory: the launch's memory on its footprint, the initial memory
elsewhere. -/
noncomputable def localMem (c : Config N) (i : Fin N) : Mem :=
  fun x => if x ∈ R i ∪ W i then c.mem x else init x

/-- The invariant of a launch of independent programs. -/
structure Inv (tr : List (Fin N × Event)) (c : Config N) : Prop where
  safe : ∀ i, Safe E (R i) (W i) (fun _ m => Q i m) (localMem R W init c i) (c.progs i)
  frame : ∀ x, (∀ i, x ∉ W i) → c.mem x = init x
  good : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) → e.reads ⊆ R j ∧ e.writes ⊆ W j

variable {E R W Q init}

theorem inv_step (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    {tr : List (Fin N × Event)} {c c' : Config N} {j : Fin N} {e : Event}
    (hinv : Inv E R W Q init tr c) (hs : LStep E c (j, e) c') :
    Inv E R W Q init (tr ++ [(j, e)]) c' ∧
    Safe.Next E (localMem R W init c' j, c'.progs j) (localMem R W init c j, c.progs j) ∧
    ∀ i, i ≠ j → localMem R W init c' i = localMem R W init c i ∧ c'.progs i = c.progs i := by
  cases hs with
  | step _ hstep =>
    rename_i m' p'
    have hsafe := hinv.safe j
    obtain ⟨m₁', m₂', p'', e', ho, hs2, hag, hfr⟩ :=
      hsafe.transfer (m₁ := c.mem) (fun x hx => by simp [localMem, hx]) hstep
    cases ho
    obtain ⟨m3, p3, e3, ho2, hR, hW, hsafe'⟩ := hsafe.step hs2
    simp only [Out.next.injEq] at ho2
    obtain ⟨h1, h2, h3⟩ := ho2
    subst h1 h2 h3
    have hframe2 := Step.frame hs2
    have hloc : localMem R W init ⟨m', Function.update c.progs j p'⟩ j = m₂' := by
      funext x
      by_cases hx : x ∈ R j ∪ W j
      · simp only [localMem, hx, ite_true]; exact hag x hx
      · simp only [localMem, hx, ite_false]
        have hxw : x ∉ e.writes := fun hw => hx (Or.inr (hW hw))
        rw [hframe2 x hxw]; simp [localMem, hx]
    have hother : ∀ i, i ≠ j → localMem R W init ⟨m', Function.update c.progs j p'⟩ i =
        localMem R W init c i := by
      intro i hij
      funext x
      by_cases hx : x ∈ R i ∪ W i
      · simp only [localMem, hx, ite_true]
        have hxw : x ∉ e.writes := fun hw => Set.disjoint_left.mp (hdisj j i hij.symm) (hW hw) hx
        exact hfr x hxw
      · simp [localMem, hx]
    refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
    · intro i
      by_cases hij : i = j
      · subst hij
        rw [hloc]
        simpa using hsafe'
      · rw [hother i hij]
        simpa [Function.update_of_ne hij] using hinv.safe i
    · intro x hx
      have hxw : x ∉ e.writes := fun hw => hx j (hW hw)
      simp only; rw [hfr x hxw]; exact hinv.frame x hx
    · intro n i e₀ hn
      by_cases hlt : n < tr.length
      · rw [List.getElem?_append_left hlt] at hn
        exact hinv.good n i e₀ hn
      · have hn' : n = tr.length := by
          rcases Nat.lt_or_ge n (tr.length + 1) with h | h
          · omega
          · rw [List.getElem?_eq_none (by simp; omega)] at hn; cases hn
        subst hn'
        rw [List.getElem?_append_right le_rfl] at hn
        simp at hn
        obtain ⟨rfl, rfl⟩ := hn
        exact ⟨hR, hW⟩
    · refine ⟨e, ?_⟩
      simp only [Function.update_self]
      rw [hloc]; exact hs2
    · intro i hij
      exact ⟨hother i hij, Function.update_of_ne hij _ _⟩

theorem inv_reach (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)) {c₀ c : Config N}
    {tr : List (Fin N × Event)} (h0 : Inv E R W Q init [] c₀) (hr : Reach E c₀ tr c) :
    Inv E R W Q init tr c := by
  induction hr with
  | refl => exact h0
  | @tail c₂ c₃ tr₁ ev _ hs ih =>
    obtain ⟨j, e⟩ := ev
    exact (inv_step hdisj ih hs).1

/-- No event of a trace satisfying the invariant conflicts across programs. -/
theorem no_interRace (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    {tr : List (Fin N × Event)} {c : Config N} (hinv : Inv E R W Q init tr c) : ¬ InterRace tr := by
  rintro ⟨a, b, ⟨i, x⟩, ⟨j, y⟩, -, ha, hb, hij, hcon, -, -⟩
  obtain ⟨hxr, hxw⟩ := hinv.good a i x ha
  obtain ⟨hyr, hyw⟩ := hinv.good b j y hb
  rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
  · rcases hzy with hzy | hzy
    · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (Or.inl (hyr hzy))
    · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (Or.inr (hyw hzy))
  · exact Set.disjoint_left.mp (hdisj j i (Ne.symm hij)) (hyw hzy) (Or.inl (hxr hzx))

end

/-- **Independent programs.** Programs that are each safe alone, from the initial memory, with
footprints no other program writes into, make a well-behaved launch. Its only outcomes are
normal ends whose memory agrees with each program's own final memory on that program's footprint
and with the initial memory where nobody writes. -/
theorem independent {k : Kernel} {L : Launch} {E : Env} {init : Mem}
    {R W : Fin L.size → Set Int} {Q : Fin L.size → Mem → Prop}
    (hsafe : ∀ i, Safe E (R i) (W i) (fun _ m => Q i m) init (k.body (L.ctx i)))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)) :
    WellBehaved E k L init ∧
    ∀ o, Outcomes E k L init o → ∃ fin, o = .ok fin ∧
      (∀ i, ∃ m, Q i m ∧ ∀ x ∈ R i ∪ W i, fin x = m x) ∧
      (∀ x, (∀ i, x ∉ W i) → fin x = init x) := by
  have hloc0 : ∀ i, localMem R W init (Config.start k L init) i = init := by
    intro i; funext x; simp [localMem, Config.start]
  have h0 : Inv E R W Q init [] (Config.start k L init) := by
    refine ⟨fun i => ?_, fun x _ => rfl, fun n j e h => by simp at h⟩
    rw [hloc0]
    simpa [Config.start] using hsafe i
  have hinv : ∀ tr c, Reach E (Config.start k L init) tr c → Inv E R W Q init tr c :=
    fun tr c hr => inv_reach hdisj h0 hr
  have hnoub : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E := by
    rintro tr c hr ⟨i, hub⟩
    obtain ⟨_, _, _, _, ho, -⟩ :=
      ((hinv tr c hr).safe i).transfer (m₁ := c.mem) (fun x hx => by simp [localMem, hx]) hub
    cases ho
  refine ⟨fun tr c hr => ⟨hnoub tr c hr, no_interRace hdisj (hinv tr c hr)⟩, ?_⟩
  intro o ho
  cases o with
  | ok fin =>
    obtain ⟨tr, c, hr, hdone, hmem⟩ := ho
    have hI := hinv tr c hr
    refine ⟨fin, rfl, fun i => ?_, fun x hx => hmem ▸ hI.frame x hx⟩
    have hs := hI.safe i
    rw [hdone i] at hs
    cases hs with
    | ret hq => exact ⟨_, hq, fun x hx => by simp [localMem, hx, hmem]⟩
  | ub =>
    obtain ⟨tr, c, hr, hub⟩ := ho
    exact absurd hub (hnoub tr c hr)
  | hang =>
    exfalso
    obtain ⟨f, hf0, hf⟩ := ho
    let ev : Nat → Fin L.size × Event := fun n => Classical.choose (hf n)
    have hev : ∀ n, LStep E (f n) (ev n) (f (n + 1)) := fun n => Classical.choose_spec (hf n)
    let tr : Nat → List (Fin L.size × Event) := fun n => (List.range n).map ev
    have htr : ∀ n, tr (n + 1) = tr n ++ [ev n] := by
      intro n; simp [tr, List.range_succ]
    have hreach : ∀ n, Reach E (Config.start k L init) (tr n) (f n) := by
      intro n
      induction n with
      | zero => rw [← hf0]; exact .refl _
      | succ n ih => rw [htr]; exact .tail ih (hev n)
    let S : Fin L.size → Nat → Mem × Prog Unit := fun i n =>
      (localMem R W init (f n) i, (f n).progs i)
    have hstepS : ∀ n, Safe.Next E (S (ev n).1 (n + 1)) (S (ev n).1 n) ∧
        ∀ i, i ≠ (ev n).1 → S i (n + 1) = S i n := by
      intro n
      obtain ⟨-, hnext, hothers⟩ := inv_step (j := (ev n).1) (e := (ev n).2) hdisj
        (hinv _ _ (hreach n)) (by simpa using hev n)
      refine ⟨hnext, fun i hij => ?_⟩
      obtain ⟨h1, h2⟩ := hothers i hij
      simp only [S, h1, h2]
    have hT : ∀ i, ∃ T, ∀ m, T ≤ m → ¬ Safe.Next E (S i (m + 1)) (S i m) := by
      intro i
      have hacc : Acc (Safe.Next E) (S i 0) := by
        have := ((hinv _ _ (hreach 0)).safe i).acc
        simpa [S] using this
      have hmono : ∀ n, S i (n + 1) = S i n ∨ Safe.Next E (S i (n + 1)) (S i n) := by
        intro n
        by_cases hij : i = (ev n).1
        · right; rw [hij]; exact (hstepS n).1
        · left; exact (hstepS n).2 i hij
      have hnd := eventually_no_descent (r := Safe.Next E) (S i) hmono
      obtain ⟨T, -, hT⟩ := hnd (S i 0) hacc 0 rfl
      exact ⟨T, hT⟩
    choose T hTi using hT
    let Tmax := Finset.univ.sup T
    have hle : T (ev Tmax).1 ≤ Tmax := Finset.le_sup (Finset.mem_univ _)
    exact hTi (ev Tmax).1 Tmax hle (hstepS Tmax).1

end TVA
