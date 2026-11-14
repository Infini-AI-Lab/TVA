import TVA.Proof.Safe
import Mathlib.Order.WellFounded
import Mathlib.Data.Fintype.Lattice

/-!
# Composing programs: from `Safe` per program to the launch

**Theorem `independent`.** Suppose each program of a launch, run *alone* from the initial memory,
is `Safe` with read footprint `R i` and write footprint `W i`, and ends in a memory satisfying
`Q i`. Suppose also that no program writes a byte another program reads or writes, and that each
program's read and write footprints are disjoint (it does not read back or update in place what
it writes). Then:

* the launch is well behaved (`WellBehaved`): no interleaving reaches undefined behaviour, a race
  between programs or a conflict within a program, and no program uses an atomic's result;
* it cannot hang;
* every outcome is a normal end whose final memory agrees, on each program's footprint, with a
  memory satisfying that program's `Q i`, and equals the initial memory wherever no program
  writes.

The proof follows every interleaving with an invariant (`Inv`): each program is safe in its own
view of memory (`localMem`: the launch's memory on its footprint, the initial memory elsewhere),
with the bytes it has not yet written as its write footprint. Another program's step changes no
byte of that view, and the program's own step is the step it takes alone (`Safe.stepT`). A hang
would make some program take infinitely many steps, which `Safe.acc` rules out.

How this fits: model → `Safe` per program → `correct_of_safe` (`TVA.Proof.Correct`), which
restates `independent` as `Correct`. Most Triton kernels are covered: each program reads inputs
and writes its own output tile, so the proof reasons about one program at a time, sequentially,
from the initial memory.
-/
namespace TVA

open Classical

/-- A sequence that, at each step, stays put or descends in a relation that is accessible from its
start, descends only finitely often. -/
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
        | zero =>
          have : n₀ = 0 := by omega
          subst this; exact hn
        | succ k ihk =>
          by_cases hk' : n₀ ≤ k
          · have hxk := ihk hk' (by omega)
            have hnr : ¬ r (x (k + 1)) (x k) := fun hr =>
              Nat.find_min hc (show k < m by omega) ⟨hk', hr⟩
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

/-- Extending a trace by one event adds, as possible conflicts within a program, only pairs
ending at the new event. -/
theorem intraRace_snoc {N : Nat} {tr : List (Fin N × Event)} {j : Fin N} {e : Event}
    (h : IntraRace (tr ++ [(j, e)])) :
    IntraRace tr ∨ ∃ (a : Nat) (f : Event), tr[a]? = some (j, f) ∧ f.Conflict e ∧
      ¬ Event.SameAtomic f e := by
  obtain ⟨a, b, x, y, hab, ha, hb, hij, hcon, hsame⟩ := h
  by_cases hb' : b < tr.length
  · left
    rw [List.getElem?_append_left hb'] at hb
    rw [List.getElem?_append_left (by omega)] at ha
    exact ⟨a, b, x, y, hab, ha, hb, hij, hcon, hsame⟩
  · right
    have hbl : b = tr.length := by
      rcases Nat.lt_or_ge b (tr.length + 1) with h | h
      · omega
      · rw [List.getElem?_eq_none (by simp; omega)] at hb; cases hb
    subst hbl
    rw [List.getElem?_append_right le_rfl] at hb
    simp at hb
    subst hb
    rw [List.getElem?_append_left hab] at ha
    obtain ⟨i, f⟩ := x
    simp only at hij
    subst hij
    exact ⟨a, f, ha, hcon, hsame⟩

/-- The last position of a trace extended by one event holds that event. -/
private theorem getElem?_snoc_eq {N : Nat} {tr : List (Fin N × Event)} {ev x : Fin N × Event}
    {n : Nat} (hlt : ¬ n < tr.length) (hn : (tr ++ [ev])[n]? = some x) : n = tr.length ∧ x = ev := by
  have hn' : n = tr.length := by
    rcases Nat.lt_or_ge n (tr.length + 1) with h | h
    · omega
    · rw [List.getElem?_eq_none (by simp; omega)] at hn; cases hn
  subst hn'
  rw [List.getElem?_append_right le_rfl] at hn
  simp at hn
  exact ⟨rfl, hn.symm⟩

section
variable {N : Nat} (E : Env) (R W : Fin N → Set Int) (Q : Fin N → Mem → Prop) (init : Mem)

/-- Program `i`'s own view of memory: the launch's memory on its footprint, the initial memory
elsewhere. -/
noncomputable def localMem (c : Config N) (i : Fin N) : Mem :=
  fun x => if x ∈ R i ∪ W i then c.mem x else init x

/-- The invariant of a launch of independent programs, after the trace `tr` in state `c`. -/
structure Inv (tr : List (Fin N × Event)) (c : Config N) : Prop where
  /-- Program `i` is safe in its own view, with the bytes it has not yet written as write
  footprint; its events so far wrote none of those. -/
  safe : ∀ i, ∃ Wc, Wc ⊆ W i ∧
    (∀ (n : Nat) (e : Event), tr[n]? = some (i, e) → Disjoint e.writes Wc) ∧
    Safe E (R i) Wc (fun _ m => Q i m) (localMem R W init c i) (c.progs i)
  /-- Bytes no program may write still hold their initial value. -/
  frame : ∀ x, (∀ i, x ∉ W i) → c.mem x = init x
  /-- Every event of the trace stays inside its program's footprint. -/
  good : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) →
    e.reads ⊆ R j ∪ e.writes ∧ e.writes ⊆ W j
  /-- No conflict within a program so far. -/
  clean : ¬ IntraRace tr

variable {E R W Q init}

/-- **One step keeps the invariant**: the stepping program's own view takes the step it would
take alone (`Safe.Next`), and the other programs' views do not change. -/
theorem inv_step (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hRW : ∀ i, Disjoint (R i) (W i))
    {tr : List (Fin N × Event)} {c c' : Config N} {j : Fin N} {e : Event}
    (hinv : Inv E R W Q init tr c) (hs : LStep E c (j, e) c') :
    Inv E R W Q init (tr ++ [(j, e)]) c' ∧
    Safe.Next E (localMem R W init c' j, c'.progs j) (localMem R W init c j, c.progs j) ∧
    ∀ i, i ≠ j → localMem R W init c' i = localMem R W init c i ∧ c'.progs i = c.progs i := by
  cases hs with
  | step _ hstep =>
    rename_i m' p'
    obtain ⟨Wc, hWc, hhist, hsafe⟩ := hinv.safe j
    obtain ⟨hR, hW, hs2, hsafe'⟩ := hsafe.stepT (m := c.mem) (fun x hx => by
      have : x ∈ R j ∪ W j := hx.elim Or.inl (fun h => Or.inr (hWc h))
      simp [localMem, this]) hstep
    have hframe := Step.frame hstep
    have hloc : localMem R W init ⟨m', Function.update c.progs j p'⟩ j =
        Mem.mix (localMem R W init c j) m' e.writes := by
      funext x
      by_cases hx : x ∈ R j ∪ W j
      · simp only [localMem, hx, ite_true]
        by_cases hxe : x ∈ e.writes
        · rw [Mem.mix_in hxe]
        · rw [Mem.mix_out hxe, hframe x hxe]; simp [localMem, hx]
      · have hxe : x ∉ e.writes := fun h => hx (Or.inr (hWc (hW h)))
        simp only [localMem, hx, ite_false]
        rw [Mem.mix_out hxe]; simp [localMem, hx]
    have hother : ∀ i, i ≠ j → localMem R W init ⟨m', Function.update c.progs j p'⟩ i =
        localMem R W init c i := by
      intro i hij
      funext x
      by_cases hx : x ∈ R i ∪ W i
      · simp only [localMem, hx, ite_true]
        exact hframe x fun hw => Set.disjoint_left.mp (hdisj j i hij.symm) (hWc (hW hw)) hx
      · simp [localMem, hx]
    refine ⟨⟨fun i => ?_, ?_, ?_, ?_⟩, ?_, ?_⟩
    · by_cases hij : i = j
      · subst hij
        refine ⟨Wc \ e.writes, Set.sdiff_subset.trans hWc, fun n f hn => ?_, ?_⟩
        · by_cases hlt : n < tr.length
          · rw [List.getElem?_append_left hlt] at hn
            exact (hhist n f hn).mono_right Set.sdiff_subset
          · obtain ⟨-, hx⟩ := getElem?_snoc_eq hlt hn
            cases hx
            exact Set.disjoint_sdiff_right
        · rw [hloc]; simpa using hsafe'
      · obtain ⟨Wi, hWi, hhi, hsi⟩ := hinv.safe i
        refine ⟨Wi, hWi, fun n f hn => ?_, ?_⟩
        · by_cases hlt : n < tr.length
          · rw [List.getElem?_append_left hlt] at hn
            exact hhi n f hn
          · obtain ⟨-, hx⟩ := getElem?_snoc_eq hlt hn
            simp only [Prod.mk.injEq] at hx
            exact absurd hx.1 hij
        · rw [hother i hij]; simpa [Function.update_of_ne hij] using hsi
    · intro x hx
      have hxw : x ∉ e.writes := fun hw => hx j (hWc (hW hw))
      simp only; rw [hframe x hxw]; exact hinv.frame x hx
    · intro n i e₀ hn
      by_cases hlt : n < tr.length
      · rw [List.getElem?_append_left hlt] at hn
        exact hinv.good n i e₀ hn
      · obtain ⟨-, hx⟩ := getElem?_snoc_eq hlt hn
        simp only [Prod.mk.injEq] at hx
        obtain ⟨rfl, rfl⟩ := hx
        exact ⟨hR, hW.trans hWc⟩
    · intro hrace
      rcases intraRace_snoc hrace with h | ⟨a, f, ha, hcon, -⟩
      · exact hinv.clean h
      · obtain ⟨hfr, hfw⟩ := hinv.good a j f ha
        have hfe : Disjoint f.writes e.writes := (hhist a f ha).mono_right hW
        have hfR : Disjoint f.writes (R j) := ((hRW j).symm).mono_left hfw
        have heR : Disjoint e.writes (R j) := ((hRW j).symm).mono_left (hW.trans hWc)
        rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
        · rcases hze with hze | hze
          · rcases hR hze with hz | hz
            · exact Set.disjoint_left.mp hfR hzf hz
            · exact Set.disjoint_left.mp hfe hzf hz
          · exact Set.disjoint_left.mp hfe hzf hze
        · rcases hfr hzf with hz | hz
          · exact Set.disjoint_left.mp heR hze hz
          · exact Set.disjoint_left.mp hfe hz hze
    · refine ⟨e, ?_⟩
      simp only [Function.update_self]
      rw [hloc]; exact hs2
    · intro i hij
      exact ⟨hother i hij, Function.update_of_ne hij _ _⟩

/-- The invariant holds along every trace from a state where it holds. -/
theorem inv_reach (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hRW : ∀ i, Disjoint (R i) (W i))
    {c₀ c : Config N} {tr : List (Fin N × Event)} (h0 : Inv E R W Q init [] c₀)
    (hr : Reach E c₀ tr c) : Inv E R W Q init tr c := by
  induction hr with
  | refl => exact h0
  | @tail c₂ c₃ tr₁ ev _ hs ih =>
    obtain ⟨j, e⟩ := ev
    exact (inv_step hdisj hRW ih hs).1

/-- No two events of a trace satisfying the invariant conflict across programs. -/
theorem no_interRace (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    {tr : List (Fin N × Event)} {c : Config N} (hinv : Inv E R W Q init tr c) :
    ¬ InterRace tr := by
  rintro ⟨a, b, ⟨i, x⟩, ⟨j, y⟩, -, ha, hb, hij, hcon, -, -⟩
  obtain ⟨hxr, hxw⟩ := hinv.good a i x ha
  obtain ⟨hyr, hyw⟩ := hinv.good b j y hb
  have hy : y.reads ∪ y.writes ⊆ R j ∪ W j := by
    rintro z (hz | hz)
    · rcases hyr hz with h | h
      · exact Or.inl h
      · exact Or.inr (hyw h)
    · exact Or.inr (hyw hz)
  have hx : x.reads ⊆ R i ∪ W i := fun z hz => (hxr hz).elim Or.inl (fun h => Or.inr (hxw h))
  rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
  · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (hy hzy)
  · exact Set.disjoint_left.mp (hdisj j i (Ne.symm hij)) (hyw hzy) (hx hzx)

end

/-- **Independent programs.** Programs that are each safe alone from the initial memory, with
footprints no other program writes into and read footprints disjoint from their own write
footprints, make a well-behaved launch. Its only outcomes are normal ends whose memory agrees
with a memory satisfying each program's `Q i` on that program's footprint, and with the initial
memory where no program writes. `correct_of_safe` states it as `Correct`. -/
theorem independent {k : Kernel} {L : Launch} {E : Env} {init : Mem}
    {R W : Fin L.size → Set Int} {Q : Fin L.size → Mem → Prop}
    (hsafe : ∀ i, Safe E (R i) (W i) (fun _ m => Q i m) init (k.body (L.ctx i)))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)) (hRW : ∀ i, Disjoint (R i) (W i)) :
    WellBehaved E k L init ∧
    ∀ o, Outcomes E k L init o → ∃ fin, o = .ok fin ∧
      (∀ i, ∃ m, Q i m ∧ ∀ x ∈ R i ∪ W i, fin x = m x) ∧
      (∀ x, (∀ i, x ∉ W i) → fin x = init x) := by
  have hloc0 : ∀ i, localMem R W init (Config.start k L init) i = init := by
    intro i; funext x; simp [localMem, Config.start]
  have h0 : Inv E R W Q init [] (Config.start k L init) := by
    refine ⟨fun i => ⟨W i, le_rfl, fun n e h => by simp at h, ?_⟩, fun x _ => rfl,
      fun n j e h => by simp at h, fun ⟨a, _, _, _, _, ha, _⟩ => by simp at ha⟩
    rw [hloc0]
    simpa [Config.start] using hsafe i
  have hinv : ∀ tr c, Reach E (Config.start k L init) tr c → Inv E R W Q init tr c :=
    fun tr c hr => inv_reach hdisj hRW h0 hr
  have hnoub : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E := by
    rintro tr c hr ⟨i, hub⟩
    obtain ⟨Wc, hWc, -, hs⟩ := (hinv tr c hr).safe i
    exact hs.no_ub (m := c.mem) (fun x hx => by
      have : x ∈ R i ∪ W i := hx.elim Or.inl (fun h => Or.inr (hWc h))
      simp [localMem, this]) hub
  have hatom : ¬ UsesSharedAtomicResult E k L init := by
    rintro ⟨tr, c, i, d, a, upd, sv, sem, sc, kk, hr, hc, ⟨v, w, hne⟩, -⟩
    obtain ⟨Wc, -, -, hs⟩ := (hinv tr c hr).safe i
    rw [hc] at hs
    cases hs with
    | rmw _ _ _ hconst _ => exact hne (hconst v w)
  refine ⟨⟨hnoub, fun tr c hr => no_interRace hdisj (hinv tr c hr),
    fun tr c hr => (hinv tr c hr).clean, hatom⟩, ?_⟩
  intro o ho
  cases o with
  | ok fin =>
    obtain ⟨tr, c, hr, hdone, hmem⟩ := ho
    have hI := hinv tr c hr
    refine ⟨fin, rfl, fun i => ?_, fun x hx => hmem ▸ hI.frame x hx⟩
    obtain ⟨Wc, -, -, hs⟩ := hI.safe i
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
      obtain ⟨-, hnext, hothers⟩ := inv_step (j := (ev n).1) (e := (ev n).2) hdisj hRW
        (hinv _ _ (hreach n)) (by simpa using hev n)
      refine ⟨hnext, fun i hij => ?_⟩
      obtain ⟨h1, h2⟩ := hothers i hij
      simp only [S, h1, h2]
    have hT : ∀ i, ∃ T, ∀ m, T ≤ m → ¬ Safe.Next E (S i (m + 1)) (S i m) := by
      intro i
      have hacc : Acc (Safe.Next E) (S i 0) := by
        obtain ⟨Wc, -, -, hs⟩ := (hinv _ _ (hreach 0)).safe i
        simpa [S] using hs.acc
      have hmono : ∀ n, S i (n + 1) = S i n ∨ Safe.Next E (S i (n + 1)) (S i n) := by
        intro n
        by_cases hij : i = (ev n).1
        · right; rw [hij]; exact (hstepS n).1
        · left; exact (hstepS n).2 i hij
      obtain ⟨T, -, hT⟩ := eventually_no_descent (r := Safe.Next E) (S i) hmono (S i 0) hacc 0 rfl
      exact ⟨T, hT⟩
    choose T hTi using hT
    let Tmax := Finset.univ.sup T
    have hle : T (ev Tmax).1 ≤ Tmax := Finset.le_sup (Finset.mem_univ _)
    exact hTi (ev Tmax).1 Tmax hle (hstepS Tmax).1

end TVA
