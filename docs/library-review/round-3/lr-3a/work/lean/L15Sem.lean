import TVA

/-! lr-3a, round 3: checks that the L15 definitions mean what SEMANTICS.md says
(`lake env lean`, this file only).

1. `Step.defer` is not vacuous: a one-program launch doing `atomic_xchg(p, 1)` then
   `atomic_xchg(p, 2)` can end with `1` at `p` (the second performed first).
2. `IntraRace`: atomics of one program on partly overlapping bytes (`int32` at 0, `int16` at 0)
   conflict, so a launch doing both is not well behaved.
3. `Launch.Valid` now rejects `pdl = true`, and `valid_grid1` still proves the 1-D case.
-/

open TVA

namespace LR3A

/-- Four present zero bytes at 0..3, nothing else. -/
def m0 : Mem := fun x => if 0 ≤ x ∧ x < 4 then some 0 else none

theorem readBytes_some (m : Mem) : ∀ (n : Nat) (a : Int), (∀ i : Nat, i < n → m (a + i) ≠ none) →
    ∃ bs, m.readBytes a n = some bs := by
  intro n
  induction n with
  | zero => intro a _; exact ⟨[], rfl⟩
  | succ n ih =>
    intro a ha
    have h0 : m a ≠ none := by simpa using ha 0 (by omega)
    obtain ⟨bs, hbs⟩ := ih (a + 1) (fun i hi => by
      have := ha (i + 1) (by omega)
      rwa [show a + ((i + 1 : Nat) : Int) = a + 1 + i by push_cast; ring] at this)
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.mp h0
    exact ⟨b :: bs, by simp [Mem.readBytes, hb, hbs]⟩

theorem read_some (m : Mem) (d : DType) (hw : m.Writable d 0) : ∃ v, m.read d 0 = some v := by
  obtain ⟨bs, hbs⟩ := readBytes_some m d.bytes 0 hw.2
  refine ⟨d.ofBits (TVA.Math.Bytes.leValue bs), ?_⟩
  simp only [Mem.read, if_pos hw.1, hbs, Option.map_some]

theorem w0 (d : DType) (hd : d.bytes ≤ 4) (ha : (0 : Int) % d.bytes = 0) : m0.Writable d 0 := by
  refine ⟨⟨le_rfl, by have := hd; omega, ha⟩, fun i hi => ?_⟩
  simp only [m0]
  rw [if_pos (by omega)]
  simp

/-- `xchg(p, v1)` then `xchg(p, v2)` on `int32` at address 0. -/
def two (v1 v2 : DType.Val .i32) : Prog Unit :=
  .rmw .i32 0 (fun _ n => n = v1) .acqRel .gpu
    (fun _ => .rmw .i32 0 (fun _ n => n = v2) .acqRel .gpu (fun _ => .ret ()))

def K2 (v1 v2 : DType.Val .i32) : Kernel := ⟨fun _ => two v1 v2⟩

def L1 : Launch := Launch.grid1 1 []

theorem L1_size : L1.size = 1 := by simp [L1, Launch.size, Launch.grid1]

/-- 1. The second exchange may be performed first, so the first exchange's value can be final. -/
theorem defer_reorders (E : Env) (v1 v2 : DType.Val .i32) :
    ∃ fin, Outcomes E (K2 v1 v2) L1 m0 (.ok fin) ∧ fin.read .i32 0 = some v1 := by
  have hw : m0.Writable .i32 0 := w0 .i32 (by decide) (by decide)
  obtain ⟨o0, ho0⟩ := read_some m0 .i32 hw
  let m1 := m0.write .i32 0 v2
  have hw1 : m1.Writable .i32 0 := by
    refine ⟨hw.1, fun i hi => ?_⟩
    have := Mem.read_write_self hw v2
    -- bytes of a written element are present
    simp only [m1, Mem.write, Mem.writeBytes, Mem.leBytes_len]
    rw [if_pos (by simp [DType.bytes] at hi ⊢; omega)]
    simp only [ne_eq]
    intro hnone
    have hlen := Mem.leBytes_len .i32 v2
    have : ((0 : Int) + i - 0).toNat < (TVA.Math.Bytes.leBytes (DType.i32.bytes) (DType.i32.toBits v2)).length := by
      rw [hlen]; simp [DType.bytes] at hi ⊢; omega
    rw [List.getElem?_eq_getElem this] at hnone
    exact Option.some_ne_none _ hnone
  obtain ⟨o1, ho1⟩ := read_some m1 .i32 hw1
  -- step 1: the first xchg stays pending while the second is performed (`Step.defer`)
  have s1 : Step E m0 (two v1 v2)
      (.next m1 (.rmw .i32 0 (fun _ n => n = v1) .acqRel .gpu (fun _ => .ret ()))
        { reads := Mem.span .i32 0, writes := Mem.span .i32 0, atomic := some (.acqRel, .gpu),
          cell := some (.i32, 0) }) :=
    Step.defer (k' := fun _ => .ret ()) (fun _ => Step.rmw o0 v2 ho0 rfl)
  -- step 2: the first xchg is performed
  have s2 : Step E m1 (.rmw .i32 0 (fun _ n => n = v1) .acqRel .gpu (fun _ => (.ret () : Prog Unit)))
      (.next (m1.write .i32 0 v1) (.ret ())
        { reads := Mem.span .i32 0, writes := Mem.span .i32 0, atomic := some (.acqRel, .gpu),
          cell := some (.i32, 0) }) :=
    Step.rmw o1 v1 ho1 rfl
  have hsz : L1.size = 1 := L1_size
  let i0 : Fin L1.size := ⟨0, by rw [hsz]; omega⟩
  have hstart : (Config.start (K2 v1 v2) L1 m0).progs i0 = two v1 v2 := rfl
  let c1 : Config L1.size := ⟨m1, Function.update (Config.start (K2 v1 v2) L1 m0).progs i0
    (.rmw .i32 0 (fun _ n => n = v1) .acqRel .gpu (fun _ => .ret ()))⟩
  have l1 : LStep E (Config.start (K2 v1 v2) L1 m0) (i0, _) c1 :=
    LStep.step i0 (by rw [hstart]; exact s1)
  have hc1 : c1.progs i0 = .rmw .i32 0 (fun _ n => n = v1) .acqRel .gpu (fun _ => .ret ()) := by
    simp [c1]
  let c2 : Config L1.size := ⟨m1.write .i32 0 v1, Function.update c1.progs i0 (.ret ())⟩
  have l2 : LStep E c1 (i0, _) c2 := LStep.step i0 (by rw [hc1]; exact s2)
  refine ⟨m1.write .i32 0 v1, ⟨_, c2, ((Reach.refl _).tail l1).tail l2, ?_, rfl⟩,
    Mem.read_write_self hw1 v1⟩
  intro j
  have hj : (j : Nat) < 1 := lt_of_lt_of_eq j.2 hsz
  have : j = i0 := Fin.ext (by simp [i0]; omega)
  subst this
  simp [c2]

/-- `atomic_add` on `int32` at 0, then on `int16` at 0: partly overlapping bytes. -/
def overlap : Prog Unit :=
  .rmw .i32 0 (fun _ _ => True) .acqRel .gpu
    (fun _ => .rmw .i16 0 (fun _ _ => True) .acqRel .gpu (fun _ => .ret ()))

/-- 2. Atomics of one program on partly overlapping bytes are a conflict: not well behaved. -/
theorem overlap_not_wellBehaved (E : Env) : ¬ WellBehaved E ⟨fun _ => overlap⟩ L1 m0 := by
  intro hwb
  have hw : m0.Writable .i32 0 := w0 .i32 (by decide) (by decide)
  obtain ⟨o0, ho0⟩ := read_some m0 .i32 hw
  let m1 := m0.write .i32 0 o0
  have hw16 : m1.Writable .i16 0 := by
    have h16 : m0.Writable .i16 0 := w0 .i16 (by decide) (by decide)
    refine ⟨h16.1, fun i hi => ?_⟩
    have hin : (0 : Int) + i ∈ Mem.span .i32 0 := by
      simp [Mem.span, DType.bytes] at hi ⊢; omega
    -- writing the same value back keeps the bytes present
    have hrw := Mem.read_write_self hw o0
    simp only [m1, Mem.write, Mem.writeBytes, Mem.leBytes_len]
    rw [if_pos (by simp [Mem.span, DType.bytes] at hin ⊢; omega)]
    intro hnone
    have : ((0 : Int) + i - 0).toNat < (TVA.Math.Bytes.leBytes (DType.i32.bytes) (DType.i32.toBits o0)).length := by
      rw [Mem.leBytes_len]; simp [DType.bytes] at hi ⊢; omega
    rw [List.getElem?_eq_getElem this] at hnone
    exact Option.some_ne_none _ hnone
  obtain ⟨o1, ho1⟩ := read_some m1 .i16 hw16
  have hsz : L1.size = 1 := L1_size
  let i0 : Fin L1.size := ⟨0, by rw [hsz]; omega⟩
  let e1 : Event := ⟨Mem.span .i32 0, Mem.span .i32 0, some (.acqRel, .gpu), some (.i32, 0)⟩
  let e2 : Event := ⟨Mem.span .i16 0, Mem.span .i16 0, some (.acqRel, .gpu), some (.i16, 0)⟩
  let c0 := Config.start (⟨fun _ => overlap⟩ : Kernel) L1 m0
  let c1 : Config L1.size := ⟨m1, Function.update c0.progs i0
    (.rmw .i16 0 (fun _ _ => True) .acqRel .gpu (fun _ => .ret ()))⟩
  have l1 : LStep E c0 (i0, e1) c1 := LStep.step i0 (Step.rmw o0 o0 ho0 trivial)
  have hc1 : c1.progs i0 = .rmw .i16 0 (fun _ _ => True) .acqRel .gpu (fun _ => .ret ()) := by
    simp [c1]
  let c2 : Config L1.size := ⟨m1.write .i16 0 o1, Function.update c1.progs i0 (.ret ())⟩
  have l2 : LStep E c1 (i0, e2) c2 := LStep.step i0 (by rw [hc1]; exact Step.rmw o1 o1 ho1 trivial)
  have hr := ((Reach.refl c0).tail l1).tail l2
  apply (hwb _ _ hr).2.2
  refine ⟨0, 1, (i0, e1), (i0, e2), by omega, rfl, rfl, rfl, ?_, ?_⟩
  · left
    exact ⟨0, by simp [e1, e2, Mem.span, DType.bytes]⟩
  · rintro ⟨-, -, hreads, -⟩
    have : (2 : Int) ∈ e1.reads := by simp [e1, Mem.span, DType.bytes]
    rw [hreads] at this
    simp [e2, Mem.span, DType.bytes] at this

/-- 3. `pdl = true` is not valid; the one-dimensional launch still is. -/
example (b : List Shape) : ¬ ({ grid := ![1, 1, 1], pdl := true, blocks := b } : Launch).Valid := by
  intro h; have := h.2.2.2.2.2.1; simp at this

example : (Launch.grid1 (2 ^ 31 - 1) []).Valid := Launch.valid_grid1 (by norm_num) (by simp)

#print axioms defer_reorders
#print axioms overlap_not_wellBehaved

end LR3A
