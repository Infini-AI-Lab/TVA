import TVA.Proof.Compose

/-!
# The endpoint: `Correct` from per-program `Safe`

`correct_of_safe` is the theorem a claim's proof ends with. A claim's `Claim.Holds` asks, for each
covered instance and initial memory, for `Correct ((c.launch d).env P) (c.kernel d) (c.launch d)
init (c.Post d init)`: the launch is valid and well behaved, and every way it ends is a normal end
whose memory satisfies the specification. `correct_of_safe` reduces that to

1. `L.Valid` (for a one-dimensional grid: `Launch.valid_grid1`);
2. each program `Safe` alone from `init`, with read footprint `R i`, write footprint `W i` and
   post-condition `Q i` (proved with the rules of `TVA.Proof.Rules`);
3. the footprint conditions: no program writes what another reads or writes, and no program reads
   what it writes;
4. the programs' post-conditions together give the specification.

How this fits: model → `Safe` per program → `correct_of_safe`. The work is in step 2; steps 3 and 4
are usually a few lines about tiles of buffers (`Buf.elems_disjoint`, `Buf.elems_disjoint_of`).
-/
namespace TVA

/-- A one-dimensional grid of `n` programs has `n` programs. -/
@[simp] theorem Launch.size_grid1 (n : Nat) (blocks : List Shape) (fusion : Bool) :
    (Launch.grid1 n blocks fusion).size = n := by
  simp [Launch.size, Launch.grid1]

/-- **A one-dimensional launch is valid** when it has fewer than `2^31` programs and Triton
accepts its block shapes (`grid1` keeps the defaults `num_ctas = 1`, no programmatic dependent
launch and no `debug`). -/
theorem Launch.valid_grid1 {n : Nat} {blocks : List Shape} {fusion : Bool} (hn : n < 2 ^ 31)
    (hb : ∀ s ∈ blocks, s.TritonOk) : (Launch.grid1 n blocks fusion).Valid :=
  ⟨by simpa [Launch.grid1] using hn, by simp [Launch.grid1], by simp [Launch.grid1],
    by rw [Launch.size_grid1]; exact hn, rfl, rfl, rfl, hb⟩

/-- **The endpoint of a claim's proof.** The launch `L` of `k` from `init` is `Correct` for the
specification `Post` when it is valid, each program `i` is `Safe` alone from `init` with footprints
`R i`, `W i` and post-condition `Q i`, no program writes into another's footprint, no program reads
what it writes, and any final memory that agrees with some `Q i`-memory on each program's footprint
and with `init` where no program writes satisfies `Post`.

Use it as the last step of `theorem main : claim.Holds`:
```
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  refine correct_of_safe (R d) (W d) (Q d init) (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · -- program `i`: rewrite the body (`k.body (L.ctx i)`) and walk it with the rules
    exact safe_program hd hpre i
  · -- `∀ i j, i ≠ j → Disjoint (W d i) (R d j ∪ W d j)`
    …
  · -- `∀ i, Disjoint (R d i) (W d i)`
    …
  · -- `∀ fin, (∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d i ∪ W d i, fin x = m x) →`
    -- `  (∀ x, (∀ i, x ∉ W d i) → fin x = init x) → claim.Post d init fin`
    intro fin hloc hfr
    …
```
To read a value back in the last goal, take the program `i` that wrote it, get its memory `m`
from `hloc i`, and move the read with `Mem.read_congr` (the element lies in `W i`). -/
theorem correct_of_safe {E : Env} {k : Kernel} {L : Launch} {init : Mem} {Post : Mem → Prop}
    (R W : Fin L.size → Set Int) (Q : Fin L.size → Mem → Prop) (hvalid : L.Valid)
    (hsafe : ∀ i, Safe E (R i) (W i) (fun _ m => Q i m) init (k.body (L.ctx i)))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hRW : ∀ i, Disjoint (R i) (W i))
    (hpost : ∀ fin, (∀ i, ∃ m, Q i m ∧ ∀ x ∈ R i ∪ W i, fin x = m x) →
      (∀ x, (∀ i, x ∉ W i) → fin x = init x) → Post fin) :
    Correct E k L init Post := by
  obtain ⟨hwb, hout⟩ := independent hsafe hdisj hRW
  refine ⟨hvalid, hwb, fun o ho => ?_⟩
  obtain ⟨fin, rfl, hloc, hfr⟩ := hout o ho
  exact ⟨fin, rfl, hpost fin hloc hfr⟩

end TVA
