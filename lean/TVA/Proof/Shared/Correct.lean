import TVA.Proof.Shared.Launch

/-!
# The endpoint for programs that share cells through atomics

`correct_of_safeA` is `correct_of_safe` (`TVA.Proof.Correct`) for kernels whose programs, besides
their private loads and stores, update shared cells with atomics: split-K GEMM and stream-K add
partial tiles into the output with `tl.atomic_add`, a histogram increments shared bins. It reduces
`Correct` to

1. `L.Valid`;
2. each program `SafeA` alone from `init` (`TVA.Proof.Shared.SafeA`), with private footprints
   `R i`, `W i`, the shared cells `C`, and a post-condition `Q i log m` on its own log of shared
   steps, which does not depend on the log's order (`PermInv`);
3. the footprint conditions of `correct_of_safe`, and: the cells are equal or disjoint, outside
   every private footprint, and present in `init`;
4. the post-conditions and the cells give the specification: for some interleaving `G` of the
   programs' shared steps, each program's private footprint holds a `Q i (proj G i)`-memory, and
   the cells hold the replay of `G` (`Replay`) from `init`.

**Why the result is determined.** Many programs `tl.atomic_add` into one cell; the cell ends at
the replay of their steps in some order. For integer additions every order gives the same value,
the initial value plus the sum of all increments (`replay_add`, `sum_proj`); for integer
`max`/`min`, the largest or smallest operand (`replay_max`, `replay_min`); for a float addition,
the exact sum up to an error bound that holds for every order (`replay_err_shared`). Each program
proves which steps it performed (`Paired`, built from `SafeA.atomic_add`'s log), and step 4
combines them with `Paired.merge` (`TVA.Proof.Shared.Replay`).
-/
namespace TVA

open Classical

/-- **Programs sharing cells through atomics.** Suppose each program is `SafeA` alone from the
initial memory, with private footprints that no other program writes into, its read footprint
disjoint from its own write footprint, and a post-condition that does not depend on the order of
its log. Suppose also that the shared cells `C` are equal or disjoint, are disjoint from every
private footprint, and are present initially. Then the launch is well behaved (no program uses an
atomic's result, since `SafeA` does not) and cannot hang, and
every outcome is a normal end in which, for some interleaving `G` of the programs' shared steps:
- each program's private footprint holds a memory satisfying its post-condition for its own part
  of `G`;
- the cells hold the replay of `G` from the initial memory;
- every other byte keeps its initial value.
`correct_of_safeA` states it as `Correct`. -/
theorem shared {k : Kernel} {L : Launch} {E : Env} {init : Mem}
    {R W : Fin L.size → Set Int} {C : Set (DType × Int)} {Q : Fin L.size → List AtomicStep → Mem → Prop}
    (hsafe : ∀ i, SafeA E (R i) (W i) C (fun _ log m => Q i log m) [] init (k.body (L.ctx i)))
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)) (hRW : ∀ i, Disjoint (R i) (W i))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hpres : ∀ x ∈ cellBytes C, init x ≠ none) :
    WellBehaved E k L init ∧
    ∀ o, Outcomes E k L init o → ∃ fin, o = .ok fin ∧
      ∃ G : List (Fin L.size × AtomicStep),
        (∀ i, ∃ m, Q i (proj G i) m ∧ ∀ x ∈ R i ∪ W i, fin x = m x) ∧
        (∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes C, fin x = mA x) ∧
        (∀ x, x ∉ cellBytes C → (∀ i, x ∉ W i) → fin x = init x) := by
  have hloc0 : ∀ i, localMem R W init (Config.start k L init) i = init := by
    intro i; funext x; simp [localMem, Config.start]
  have h0 : InvA E R W C Q init [] (Config.start k L init) := by
    refine ⟨⟨[], fun i => ⟨W i, le_rfl, fun n e h => by simp at h, ?_⟩, init, .nil _, fun _ _ => rfl⟩,
      fun _ _ _ => rfl, hpres, fun n j e h => by simp at h, fun ⟨a, _, _, _, _, ha, _⟩ => by simp at ha⟩
    rw [hloc0]
    simpa [Config.start, proj] using hsafe i
  have hinv : ∀ tr c, Reach E (Config.start k L init) tr c → InvA E R W C Q init tr c :=
    fun tr c hr => invA_reach hdisj hA hRW hcells hQ h0 hr
  have hnoub : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E := by
    rintro tr c hr ⟨i, hub⟩
    have hI := hinv tr c hr
    obtain ⟨G, hsafeG, -⟩ := hI.safe
    obtain ⟨Wc, hWc, -, hs⟩ := hsafeG i
    exact hs.no_ub (m := c.mem) (fun x hx => by
      have : x ∈ R i ∪ W i := hx.elim Or.inl (fun h => Or.inr (hWc h))
      simp [localMem, this]) hI.present hub
  have hatom : ¬ UsesSharedAtomicResult E k L init := by
    rintro ⟨tr, c, i, d, a, upd, sv, sem, sc, kk, hr, hc, ⟨v, w, hne⟩, -⟩
    obtain ⟨G, hsafeG, -⟩ := (hinv tr c hr).safe
    obtain ⟨Wc, -, -, hs⟩ := hsafeG i
    rw [hc] at hs
    exact hne (hs.rmw_const v w)
  refine ⟨⟨hnoub, fun tr c hr => no_interRaceA hdisj hA hcells (hinv tr c hr),
    fun tr c hr => (hinv tr c hr).clean, hatom⟩, ?_⟩
  intro o ho
  cases o with
  | ok fin =>
    obtain ⟨tr, c, hr, hdone, hmem⟩ := ho
    have hI := hinv tr c hr
    obtain ⟨G, hsafeG, mA, hrep, hagA⟩ := hI.safe
    refine ⟨fin, rfl, G, fun i => ?_, ⟨mA, hrep, fun x hx => hmem ▸ hagA x hx⟩,
      fun x hx hw => hmem ▸ hI.frame x hx hw⟩
    obtain ⟨Wc, -, -, hs⟩ := hsafeG i
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
    have hstepS : ∀ n, SafeA.Next E C (S (ev n).1 (n + 1)) (S (ev n).1 n) ∧
        ∀ i, i ≠ (ev n).1 → S i (n + 1) = S i n := by
      intro n
      obtain ⟨-, hnext, hothers⟩ := invA_step (j := (ev n).1) (e := (ev n).2) hdisj hA hRW hcells hQ
        (hinv _ _ (hreach n)) (by simpa using hev n)
      refine ⟨hnext, fun i hij => ?_⟩
      obtain ⟨h1, h2⟩ := hothers i hij
      simp only [S, h1, h2]
    have hT : ∀ i, ∃ T, ∀ m, T ≤ m → ¬ SafeA.Next E C (S i (m + 1)) (S i m) := by
      intro i
      have hacc : Acc (SafeA.Next E C) (S i 0) := by
        obtain ⟨G, hsafeG, -⟩ := (hinv _ _ (hreach 0)).safe
        obtain ⟨Wc, hWc, -, hs⟩ := hsafeG i
        simpa [S] using hs.acc (hQ i) ((hA i).mono_right (Set.union_subset_union_right _ hWc))
      have hmono : ∀ n, S i (n + 1) = S i n ∨ SafeA.Next E C (S i (n + 1)) (S i n) := by
        intro n
        by_cases hij : i = (ev n).1
        · right; rw [hij]; exact (hstepS n).1
        · left; exact (hstepS n).2 i hij
      obtain ⟨T, -, hT⟩ := eventually_no_descent (r := SafeA.Next E C) (S i) hmono (S i 0) hacc 0 rfl
      exact ⟨T, hT⟩
    choose T hTi using hT
    let Tmax := Finset.univ.sup T
    have hle : T (ev Tmax).1 ≤ Tmax := Finset.le_sup (Finset.mem_univ _)
    exact hTi (ev Tmax).1 Tmax hle (hstepS Tmax).1

/-- **The endpoint of a claim's proof for programs that share cells through atomics.** The launch
`L` of `k` from `init` is `Correct` for `Post` when it is valid; each program `i` is `SafeA` alone
from `init` with private footprints `R i`, `W i`, shared cells `C` and a post-condition `Q i` on
its log that ignores the log's order; private footprints do not conflict (as for
`correct_of_safe`); the cells are equal or disjoint, outside every private footprint and present
in `init`; and any final memory that, for some interleaving `G` of the shared steps, agrees with a
`Q i (proj G i)`-memory on each program's private footprint, holds the replay of `G` on the cells,
and agrees with `init` elsewhere where no program writes, satisfies `Post`.

Use it as the last step of `main`, as `correct_of_safe`. In the last goal, take a cell's final
value from the replay (`replay_add`, `replay_max`, `replay_err_shared`) and combine the programs'
own logs with `Paired.merge` or `sum_proj`. -/
theorem correct_of_safeA {E : Env} {k : Kernel} {L : Launch} {init : Mem} {Post : Mem → Prop}
    (R W : Fin L.size → Set Int) (C : Set (DType × Int))
    (Q : Fin L.size → List AtomicStep → Mem → Prop) (hvalid : L.Valid)
    (hsafe : ∀ i, SafeA E (R i) (W i) C (fun _ log m => Q i log m) [] init (k.body (L.ctx i)))
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)) (hRW : ∀ i, Disjoint (R i) (W i))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hpres : ∀ x ∈ cellBytes C, init x ≠ none)
    (hpost : ∀ fin (G : List (Fin L.size × AtomicStep)),
      (∀ i, ∃ m, Q i (proj G i) m ∧ ∀ x ∈ R i ∪ W i, fin x = m x) →
      (∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes C, fin x = mA x) →
      (∀ x, x ∉ cellBytes C → (∀ i, x ∉ W i) → fin x = init x) → Post fin) :
    Correct E k L init Post := by
  obtain ⟨hwb, hout⟩ := shared hsafe hQ hdisj hRW hA hcells hpres
  refine ⟨hvalid, hwb, fun o ho => ?_⟩
  obtain ⟨fin, rfl, G, hloc, hrep, hfr⟩ := hout o ho
  exact ⟨fin, rfl, hpost fin G hloc hrep hfr⟩

end TVA
