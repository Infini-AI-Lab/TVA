import TVA
open TVA

#check @TVA.sum_bcast_mul_depth_err
#check @TVA.bcastTreeBound_eq
#check @TVA.AddRel.wide
#check @TVA.tl.atomic_add_unused
#print axioms TVA.sum_bcast_mul_depth_err
#print axioms TVA.AddRel.wide
#print axioms TVA.MemSem.unused_gpu
#print axioms TVA.bcastTreeBound_eq

-- MemSem.unused is a plain function: nothing ties it to an unused result.
example : (TVA.MemSem.relaxed).unused .gpu = .acqRel := rfl
example : (TVA.MemSem.relaxed).unused .cta = .relaxed := rfl

-- The depth bound at depth K - 1 is the tree bound; at a smaller depth it is no larger (sanity, K = 8).
example : bcastDepthBound 7 8 1 = bcastTreeBound 8 1 := (bcastTreeBound_eq 8 1).symm
