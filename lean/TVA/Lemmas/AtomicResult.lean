import TVA.Sem.AtomicResult
import TVA.Lemmas.Mem

/-!
# lr-34a's counterexample is excluded (library issue L161)

Library review lr-34a (round 34, `docs/library-review/round-34/lr-34a/work/crosswarp_nowait.py`;
`evidence/atomic-result-order`, launch A) found a well-behaved launch whose device outcome the
semantics lacks. Program 0 runs `r = tl.atomic_add(Y + x, 0)`, `tl.store(OUT + x, r)` and
`tl.atomic_add(C + x, 1, sem=SEM)` over `x = tl.arange(0, 512)`; program 1 runs
`q = tl.atomic_add(C + 511, 0)`, `tl.atomic_add(Y, q)` and `tl.store(Q, q)`. (The source's delay
chains are loads of buffers no program writes, which on the run leave `x = offs` and `j = 0`; the
model writes those values.) Program 0 uses an atomic's result on `Y[0]`, which program 1 also
accesses: `crosswarp_excluded` proves the launch is under `UsesSharedAtomicResult`, for every
order `SEM` and every initial memory in which program 1's two cells are present and distinct.

The proof follows program 0 to its first read-modify-write (lane 0 first), whose continuation
stores the old value (`skipRmw` follows the later read-modify-writes to that store), and
program 1 through its first atomic to its read-modify-write of `Y[0]`.
-/
namespace TVA.AtomicResult

open TVA

/-! ## Following a program's read-modify-writes -/

/-- The program after its leading read-modify-writes, each given the default old value. -/
def skipRmw {α : Type} : Prog α → Prog α
  | .rmw _ _ _ _ _ k => skipRmw (k default)
  | .ret a => .ret a
  | .load d s addrs k => .load d s addrs k
  | .store d s addrs vals k => .store d s addrs vals k
  | .pick P sv k => .pick P sv k
  | .plat q k => .plat q k
  | .ub => .ub
  | .loop init body k => .loop init body k
  | .env k => .env k

theorem foldl_update_other {ι β : Type} [DecidableEq ι] (x : β) :
    ∀ (rest : List ι) (acc : ι → β) (i : ι), i ∉ rest →
      (rest.foldl (fun a j => Function.update a j x) acc) i = acc i
  | [], _, _, _ => rfl
  | j :: rest, acc, i, h => by
    simp only [List.foldl_cons]
    rw [foldl_update_other x rest _ i (fun h' => h (List.mem_cons_of_mem _ h'))]
    exact Function.update_of_ne (by rintro rfl; exact h List.mem_cons_self) _ _

/-- The read-modify-writes of `rmwSeq`, followed with default old values, reach the continuation
with the defaults collected. -/
theorem skipRmw_rmwSeq {s : Shape} {d : DType} {β : Type} (ptrs : Tensor s (Ptr d))
    (upd : Idx s → d.Val → d.Val → Prop) (sem : MemSem) (sc : Scope) (g : (Idx s → d.Val) → Prog β) :
    ∀ (rest : List (Idx s)) (acc : Idx s → d.Val),
      skipRmw ((tl.rmwSeq ptrs upd sem sc rest acc).bind g) =
        skipRmw (g (rest.foldl (fun a j => Function.update a j default) acc))
  | [], _ => rfl
  | i :: rest, acc => by
    show skipRmw ((tl.rmwSeq ptrs upd sem sc rest (Function.update acc i default)).bind g) = _
    rw [skipRmw_rmwSeq ptrs upd sem sc g rest]
    rfl

/-! ## The counterexample's model -/

/-- Program 0 after its first atomic: `tl.store(OUT + x, r)`, then `tl.atomic_add(C + x, 1, sem=SEM)`
(its result unused). -/
def cont0 (C OUT : Ptr .i32) (sem : MemSem) (r : Tensor [512] I32) : Prog Unit := do
  tl.store (OUT + tl.arange 0 512) r
  tl.atomic_add_unused (C + tl.arange 0 512) (tl.full [512] (1 : I32)) (sem := sem)

/-- Program 0: `x = tl.arange(0, 512)`, `r = tl.atomic_add(Y + x, 0)` (its result used), then
`cont0`. -/
def prog0 (Y C OUT : Ptr .i32) (sem : MemSem) : Prog Unit := do
  let r ← tl.atomic_add (Y + tl.arange 0 512) (tl.zeros [512] .i32)
  cont0 C OUT sem r

/-- Program 1 after its first atomic: `tl.atomic_add(Y, q)`, `tl.store(Q, q)`. -/
def cont1 (Y Q : Ptr .i32) (q : Tensor [] I32) : Prog Unit := do
  let _ ← tl.atomic_add (Tensor.scalar Y) q
  tl.store1 Q q.item

/-- Program 1: `q = tl.atomic_add(C + 511, 0)`, then `cont1`. -/
def prog1 (Y C Q : Ptr .i32) : Prog Unit := do
  let q ← tl.atomic_add (Tensor.scalar (C.offset 511)) (Tensor.scalar (0 : I32))
  cont1 Y Q q

/-- The kernel: program 0 and program 1. -/
def kernel (Y C OUT Q : Ptr .i32) (sem : MemSem) : Kernel where
  body ctx := if ctx.pid 0 = 0 then prog0 Y C OUT sem else prog1 Y C Q

/-- The launch: two programs, blocks of 512. -/
def launch : Launch := Launch.grid1 2 [[512]]

theorem launch_size : launch.size = 2 := rfl

/-- The model's program `0`. -/
def p0 : Fin launch.size := ⟨0, by decide⟩
/-- The model's program `1`. -/
def p1 : Fin launch.size := ⟨1, by decide⟩

theorem p1_ne_p0 : p1 ≠ p0 := by decide

/-! ## The exclusion holds -/

theorem reach_snoc {E : Env} {N : Nat} {c c' : Config N} {tr : List (Fin N × Event)} (h : Reach E c tr c')
    (i : Fin N) {m' : Mem} {p : Prog Unit} {e : Event} (hs : Step E c'.mem (c'.progs i) (.next m' p e)) :
    Reach E c (tr ++ [(i, e)]) ⟨m', Function.update c'.progs i p⟩ :=
  Reach.tail h (LStep.step i hs)

theorem reach_snoc_upd {E : Env} {N : Nat} {c : Config N} {tr : List (Fin N × Event)} {mem : Mem}
    {progs : Fin N → Prog Unit} {i : Fin N} {q : Prog Unit} {m' : Mem} {p : Prog Unit} {e : Event}
    (h : Reach E c tr ⟨mem, Function.update progs i q⟩) (hs : Step E mem q (.next m' p e)) :
    Reach E c (tr ++ [(i, e)]) ⟨m', Function.update (Function.update progs i q) i p⟩ :=
  Reach.tail h (LStep.step i (by show Step E mem (Function.update progs i q i) _
                                 rw [Function.update_self]; exact hs))

/-- **lr-34a's counterexample is excluded** (library issue L161): program 0 uses the result of its
atomic on `Y[0]`, which program 1 also accesses. -/
theorem crosswarp_excluded (P : Platform) (Y C OUT Q : Ptr .i32) (sem : MemSem) (init : Mem)
    (hC : ∃ v, init.read .i32 (C.offset 511).addr = some v)
    (hY : ∃ v, init.read .i32 Y.addr = some v)
    (hdisj : Disjoint (Mem.span .i32 (C.offset 511).addr) (Mem.span .i32 Y.addr)) :
    UsesSharedAtomicResult (launch.env P) (kernel Y C OUT Q sem) launch init := by
  obtain ⟨c0v, hC⟩ := hC
  obtain ⟨y0, hY⟩ := hY
  -- program 0, to its first read-modify-write: lane 0 first
  let E := launch.env P
  let ptrs0 : Tensor [512 - 0] (Ptr .i32) := Y + tl.arange 0 512
  let upd0 : Idx [512 - 0] → I32 → I32 → Prop := fun i => tl.addUpd E .i32 ((tl.zeros [512] .i32).get i)
  let junk : Idx [512 - 0] → I32 := fun _ => default
  let lane0 : Idx [512 - 0] := ((0 : Fin 512), ())
  let enab : List (Idx [512 - 0]) :=
    (Idx.all [512 - 0]).filter (fun i => (tl.addrs ptrs0 none i).isSome)
  have hmem : lane0 ∈ enab := by simp [enab, Idx.mem_all, tl.addrs]
  have hnd : enab.Nodup := (Idx.nodup_all _).filter _
  let G0 := cont0 C OUT sem
  let kk : I32 → Prog Unit := fun v =>
    ((tl.rmwSeq ptrs0 upd0 .acqRel .gpu (enab.erase lane0) (Function.update junk lane0 v)).bind
      (fun old => Prog.ret ⟨old⟩)).bind G0
  let q1 : Prog Unit := (tl.atomicWith ptrs0 upd0 none .acqRel .gpu).bind G0
  let q2 : Prog Unit := (Prog.pick (fun order : List (Idx [512 - 0]) => order.Perm enab) Solve.none fun order =>
    ((tl.rmwSeq ptrs0 upd0 .acqRel .gpu order junk).bind (fun old => Prog.ret ⟨old⟩)).bind G0)
  let q3 : Prog Unit := .rmw .i32 (ptrs0.get lane0).addr (upd0 lane0) .acqRel .gpu kk
  have s1 : Step E init (prog0 Y C OUT sem) (.next init q1 .silent) := Step.env
  have s2 : Step E init q1 (.next init q2 .silent) := Step.pick junk trivial
  have s3 : Step E init q2 (.next init q3 .silent) :=
    Step.pick (lane0 :: enab.erase lane0) (List.perm_cons_erase hmem).symm
  let c0 := Config.start (kernel Y C OUT Q sem) launch init
  have h00 : c0.progs p0 = prog0 Y C OUT sem := rfl
  have r1 := reach_snoc (Reach.refl c0) p0 (m' := init) (p := q1) (e := .silent) (by rw [h00]; exact s1)
  have r3 := reach_snoc_upd (reach_snoc_upd r1 s2) s3
  refine ⟨_, _, p0, .i32, (ptrs0.get lane0).addr, upd0 lane0, .acqRel, .gpu, kk, r3,
    (by simp [q3]), ?uses, ?part2⟩
  case uses =>
    refine ⟨(0 : I32), (1 : I32), fun h => ?_⟩
    let fold : I32 → Idx [512 - 0] → I32 := fun v =>
      (enab.erase lane0).foldl (fun a j => Function.update a j default) (Function.update junk lane0 v)
    have key : ∀ v, skipRmw (kk v) = skipRmw (G0 ⟨fold v⟩) := by
      intro v
      show skipRmw (((tl.rmwSeq ptrs0 upd0 .acqRel .gpu (enab.erase lane0)
        (Function.update junk lane0 v)).bind (fun old => Prog.ret ⟨old⟩)).bind G0) = _
      rw [Prog.bind_assoc, skipRmw_rmwSeq]
      rfl
    have h' := congrArg skipRmw h
    rw [key, key] at h'
    have hs : ∀ t : Tensor [512] I32, skipRmw (G0 t) = .store .i32 [512 - 0]
        (tl.addrs (OUT + tl.arange 0 512) none) t ((Prog.ret ()).bind fun _ =>
          tl.atomic_add_unused (C + tl.arange 0 512) (tl.full [512] (1 : I32)) (sem := sem)) :=
      fun t => rfl
    rw [hs, hs] at h'
    have hv : (⟨fold 0⟩ : Tensor [512] I32) = ⟨fold 1⟩ := by
      injection h' with h1 h2 h3 h4 h5
    have h0 := congrArg (fun t : Tensor [512] I32 => t.get lane0) hv
    simp only [fold] at h0
    rw [foldl_update_other _ _ _ _ (hnd.not_mem_erase), foldl_update_other _ _ _ _ (hnd.not_mem_erase),
      Function.update_self, Function.update_self] at h0
    exact absurd h0 (by decide)
  case part2 =>
    -- program 1: its atomic on `C[511]`, then its read-modify-write of `Y[0]`
    let ptrsC : Tensor [] (Ptr .i32) := Tensor.scalar (C.offset 511)
    let updC : Idx [] → I32 → I32 → Prop := fun i => tl.addUpd E .i32 ((Tensor.scalar (0 : I32)).get i)
    let junkC : Idx [] → I32 := fun _ => default
    let enabC : List (Idx []) := (Idx.all []).filter (fun i => (tl.addrs ptrsC none i).isSome)
    let G1 := cont1 Y Q
    let w1 : Prog Unit := (tl.atomicWith ptrsC updC none .acqRel .gpu).bind G1
    let w2 : Prog Unit := Prog.pick (fun order : List (Idx []) => order.Perm enabC) Solve.none fun order =>
      ((tl.rmwSeq ptrsC updC .acqRel .gpu order junkC).bind (fun old => Prog.ret ⟨old⟩)).bind G1
    let w3 : Prog Unit :=
      ((tl.rmwSeq ptrsC updC .acqRel .gpu enabC junkC).bind (fun old => Prog.ret ⟨old⟩)).bind G1
    let qv : Tensor [] I32 := ⟨Function.update junkC () c0v⟩
    let aC : Int := (C.offset 511).addr
    let e4 : Event := ⟨Mem.span .i32 aC, Mem.span .i32 aC, some (.acqRel, .gpu), some (.i32, aC)⟩
    let m4 : Mem := init.write .i32 aC (c0v + 0)
    let ptrsY : Tensor [] (Ptr .i32) := Tensor.scalar Y
    let updY : Idx [] → I32 → I32 → Prop := fun i => tl.addUpd E .i32 (qv.get i)
    let enabY : List (Idx []) := (Idx.all []).filter (fun i => (tl.addrs ptrsY none i).isSome)
    let G2 : Tensor [] I32 → Prog Unit := fun _ => tl.store1 Q qv.item
    let w5 : Prog Unit := (tl.atomicWith ptrsY updY none .acqRel .gpu).bind G2
    let w6 : Prog Unit := Prog.pick (fun order : List (Idx []) => order.Perm enabY) Solve.none fun order =>
      ((tl.rmwSeq ptrsY updY .acqRel .gpu order junkC).bind (fun old => Prog.ret ⟨old⟩)).bind G2
    let w7 : Prog Unit :=
      ((tl.rmwSeq ptrsY updY .acqRel .gpu enabY junkC).bind (fun old => Prog.ret ⟨old⟩)).bind G2
    let e8 : Event := ⟨Mem.span .i32 Y.addr, Mem.span .i32 Y.addr, some (.acqRel, .gpu), some (.i32, Y.addr)⟩
    have hY4 : m4.read .i32 Y.addr = some y0 := by rw [Mem.read_write_other _ hdisj]; exact hY
    have t1 : Step E init (prog1 Y C Q) (.next init w1 .silent) := Step.env
    have t2 : Step E init w1 (.next init w2 .silent) := Step.pick junkC trivial
    have t3 : Step E init w2 (.next init w3 .silent) := Step.pick enabC (List.Perm.refl _)
    have t4 : Step E init w3 (.next m4 (cont1 Y Q qv) e4) := Step.rmw c0v (c0v + 0) hC rfl
    have t5 : Step E m4 (cont1 Y Q qv) (.next m4 w5 .silent) := Step.env
    have t6 : Step E m4 w5 (.next m4 w6 .silent) := Step.pick junkC trivial
    have t7 : Step E m4 w6 (.next m4 w7 .silent) := Step.pick enabY (List.Perm.refl _)
    have t8 : Step E m4 w7 (.next (m4.write .i32 Y.addr (y0 + qv.get ())) (G2 ⟨Function.update junkC () y0⟩) e8) :=
      Step.rmw y0 (y0 + qv.get ()) hY4 rfl
    let c0 := Config.start (kernel Y C OUT Q sem) launch init
    have h01 : c0.progs p1 = prog1 Y C Q := rfl
    have u1 := reach_snoc (Reach.refl c0) p1 (m' := init) (p := w1) (e := .silent) (by rw [h01]; exact t1)
    have u8 := reach_snoc_upd (reach_snoc_upd (reach_snoc_upd (reach_snoc_upd (reach_snoc_upd
      (reach_snoc_upd (reach_snoc_upd u1 t2) t3) t4) t5) t6) t7) t8
    have hYin : Y.addr ∈ Mem.span .i32 Y.addr := by
      show Y.addr ≤ Y.addr ∧ Y.addr < Y.addr + (4 : Nat); omega
    have ha0 : (ptrs0.get lane0).addr = Y.addr := by
      show Y.addr + ((tl.arange 0 512).get lane0).toInt * (4 : Nat) = Y.addr
      have : ((tl.arange 0 512).get lane0).toInt = 0 := by decide
      rw [this]; simp
    refine ⟨_, _, p1, e8, u8, by simp, p1_ne_p0, Y.addr, Or.inl hYin, ?_⟩
    rw [ha0]; exact hYin

end TVA.AtomicResult
