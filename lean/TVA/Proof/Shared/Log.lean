import TVA.Proof.Basic

/-!
# Logs of shared atomics and their replay

Some kernels have several programs update the same memory cells, each with `tl.atomic_*`:
split-K GEMM adds partial tiles into one output tile, a histogram increments shared bins,
stream-K adds the partial sums of a tile split across programs. No program can know the values it
finds in such a cell, since the other programs change it in an order nobody controls. What it can
know is what it did: each of its atomic steps on a shared cell found some old value and wrote a
new one. This file is the vocabulary for that record.

* `AtomicStep`: one atomic step as it happened (the cell, its old value and its new value).
* `Replay m l m'`: performing the steps of `l` in order from `m` gives `m'` (each step finds its
  old value). The launch's shared cells end as the replay of every program's steps, interleaved.
* `proj G i`: program `i`'s steps in an interleaved log `G`.
* `cellBytes C`: the bytes of a set of cells.
* `PermInv Q`: a post-condition that does not depend on the order of the log. A program's
  atomics may be performed in an order other than the text's (`Step.defer`), so its own log is
  known only up to order.

How this fits: `TVA.Proof.Shared.SafeA` proves one program with this log, `TVA.Proof.Shared.Launch`
and `TVA.Proof.Shared.Correct` compose the programs, and `TVA.Proof.Shared.Replay` computes what a
replay leaves in a cell (a sum, a maximum, a float sum with its error).
-/
namespace TVA

/-- A successful read means the element's bytes are present at an aligned address. -/
theorem Mem.writable_of_read {m : Mem} {d : DType} {a : Int} {v : d.Val}
    (h : m.read d a = some v) : m.Writable d a := by
  unfold Mem.read at h
  split_ifs at h with hal
  refine ⟨hal, ?_⟩
  have key : ∀ (n : Nat) (a : Int) (bs : List Bytes.Byte), m.readBytes a n = some bs →
      ∀ k : Nat, k < n → m (a + k) ≠ none := by
    intro n
    induction n with
    | zero => intro _ _ _ k hk; omega
    | succ n ih =>
      intro a bs h k hk
      simp only [Mem.readBytes] at h
      cases ha : m a with
      | none => rw [ha] at h; simp at h
      | some b =>
        cases hr : m.readBytes (a + 1) n with
        | none => rw [ha, hr] at h; simp at h
        | some bs' =>
          rcases k with _ | k
          · simp [ha]
          · have := ih (a + 1) bs' hr k (by omega)
            rwa [show a + 1 + (k : Int) = a + ((k + 1 : Nat) : Int) by push_cast; ring] at this
  cases hb : m.readBytes a d.bytes with
  | none => rw [hb] at h; simp at h
  | some bs => exact key _ _ _ hb

/-! ## Atomic steps and their replay -/

/-- One atomic read-modify-write of one element (one lane of a `tl.atomic_*`) on a shared cell,
as it happened: the cell (its dtype `d` and byte address `a`), the old value it found there and
the new value it wrote. The operand is not recorded (`TVA.Proof.Shared.Replay` pairs it with the
step when the new value alone does not determine it). -/
structure AtomicStep where
  /-- The dtype of the cell. -/
  d : DType
  /-- The byte address of the cell. -/
  a : Int
  /-- The value the atomic found. -/
  old : d.Val
  /-- The value the atomic wrote. -/
  new : d.Val

/-- A launch step determines the atomic step it performed: its event names the cell, the memory
before gives the old value, the memory after the new one. Use it to identify two descriptions of
one step. -/
theorem AtomicStep.unique {m m' : Mem} {e : Event} {s s' : AtomicStep}
    (hc : e.cell = some (s.d, s.a)) (hc' : e.cell = some (s'.d, s'.a))
    (hr : m.read s.d s.a = some s.old) (hr' : m.read s'.d s'.a = some s'.old)
    (hw : m' = m.write s.d s.a s.new) (hw' : m' = m.write s'.d s'.a s'.new) : s = s' := by
  obtain ⟨d, a, o, n⟩ := s
  obtain ⟨d', a', o', n'⟩ := s'
  rw [hc] at hc'
  simp only [Option.some.injEq, Prod.mk.injEq] at hc'
  obtain ⟨rfl, rfl⟩ := hc'
  simp only at hr hr' hw hw'
  rw [hr] at hr'
  cases hr'
  have h1 := Mem.read_write_self (Mem.writable_of_read hr) n
  have h2 := Mem.read_write_self (Mem.writable_of_read hr) n'
  rw [← hw] at h1
  rw [← hw', h1] at h2
  cases h2
  rfl

/-- **Replaying atomic steps** from a memory, in the list's order: each step finds its `old`
value in the cell and writes its `new` value. The shared cells of a launch end as the replay of
all programs' steps in the order they were performed. -/
inductive Replay : Mem → List AtomicStep → Mem → Prop where
  /-- No step leaves the memory as it is. -/
  | nil (m : Mem) : Replay m [] m
  /-- The first step finds its old value, writes its new one, and the rest replays from there. -/
  | cons {m m' : Mem} {s : AtomicStep} {rest : List AtomicStep} :
      m.read s.d s.a = some s.old → Replay (m.write s.d s.a s.new) rest m' →
      Replay m (s :: rest) m'

/-- A replay extended by one more step that finds its old value. -/
theorem Replay.snoc {m m' : Mem} {l : List AtomicStep} {s : AtomicStep} (h : Replay m l m')
    (hr : m'.read s.d s.a = some s.old) : Replay m (l ++ [s]) (m'.write s.d s.a s.new) := by
  induction h with
  | nil m => exact .cons hr (.nil _)
  | cons h0 _ ih => exact .cons h0 (ih hr)

/-- Program `i`'s part of an interleaved log `G` (each step tagged with its program), in order. -/
def proj {N : Nat} (G : List (Fin N × AtomicStep)) (i : Fin N) : List AtomicStep :=
  (G.filter (fun g => decide (g.1 = i))).map Prod.snd

/-- A step of program `j` appended to the interleaved log extends `j`'s part. -/
theorem proj_snoc_self {N : Nat} (G : List (Fin N × AtomicStep)) (j : Fin N) (s : AtomicStep) :
    proj (G ++ [(j, s)]) j = proj G j ++ [s] := by
  simp [proj, List.filter_append]

/-- A step of program `j` appended to the interleaved log leaves another program's part. -/
theorem proj_snoc_other {N : Nat} (G : List (Fin N × AtomicStep)) {i j : Fin N} (h : i ≠ j)
    (s : AtomicStep) : proj (G ++ [(j, s)]) i = proj G i := by
  simp [proj, List.filter_append, Ne.symm h]

/-! ## Shared cells -/

/-- The bytes of a set of cells, each a dtype and a byte address. -/
def cellBytes (C : Set (DType × Int)) : Set Int := {x | ∃ c ∈ C, x ∈ Mem.span c.1 c.2}

/-- A cell's bytes are among the bytes of a set containing it. -/
theorem span_sub_cellBytes {C : Set (DType × Int)} {d : DType} {a : Int} (h : (d, a) ∈ C) :
    Mem.span d a ⊆ cellBytes C := fun _ hx => ⟨(d, a), h, hx⟩

/-- An element occupies at least one byte. -/
theorem span_nonempty (d : DType) (a : Int) : (Mem.span d a).Nonempty :=
  ⟨a, by simp only [Mem.span, Set.mem_ofPred_eq]; have := d.bytes_pos; omega⟩

/-- A post-condition (on a result, a log and a memory) that does not depend on the order of the
log. The post-condition of a program with shared atomics must be one: its atomics may be performed
in an order other than the text's (`Step.defer`). Sums, maxima and multisets of the log's steps
are; `Paired.perm` gives it for a post-condition stated with `Paired`. -/
def PermInv {α : Type} (Q : α → List AtomicStep → Mem → Prop) : Prop :=
  ∀ a l l' m, l.Perm l' → Q a l m → Q a l' m

end TVA
