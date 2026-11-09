import TVA.Lemmas.AtomicResult
/-! lr-35a: a tensor atomic whose result the continuation discards has a continuation that is the
same for every old value, so `UsesSharedAtomicResult`'s second conjunct fails for it (the
predicate is not satisfied merely because lanes collect old values). -/
open TVA

theorem rmwSeq_discard {s : Shape} {d : DType} {β : Type} (ptrs : Tensor s (Ptr d))
    (upd : Idx s → d.Val → d.Val → Prop) (sem : MemSem) (sc : Scope) (G : Prog β) :
    ∀ (rest : List (Idx s)) (acc acc' : Idx s → d.Val),
      (tl.rmwSeq ptrs upd sem sc rest acc).bind (fun _ => G) =
        (tl.rmwSeq ptrs upd sem sc rest acc').bind (fun _ => G)
  | [], _, _ => rfl
  | i :: rest, acc, acc' => by
    show Prog.rmw _ _ _ _ _ (fun old => _) = Prog.rmw _ _ _ _ _ (fun old => _)
    congr 1
    funext old
    exact rmwSeq_discard ptrs upd sem sc G rest _ _

example {s : Shape} {d : DType} (ptrs : Tensor s (Ptr d)) (upd : Idx s → d.Val → d.Val → Prop)
    (rest : List (Idx s)) (junk : Idx s → d.Val) (l : Idx s) (G : Prog Unit) :
    ∀ v w : d.Val,
      (((tl.rmwSeq ptrs upd .acqRel .gpu rest (Function.update junk l v)).bind
          (fun old => Prog.ret (⟨old⟩ : Tensor s d.Val))).bind (fun _ => G)) =
      (((tl.rmwSeq ptrs upd .acqRel .gpu rest (Function.update junk l w)).bind
          (fun old => Prog.ret (⟨old⟩ : Tensor s d.Val))).bind (fun _ => G)) := by
  intro v w
  rw [Prog.bind_assoc, Prog.bind_assoc]
  exact rmwSeq_discard ptrs upd _ _ G rest _ _
