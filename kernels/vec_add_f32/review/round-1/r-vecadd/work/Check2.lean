import VecAdd
open TVA TVAExamples.VecAdd
def d0 : Inst := ⟨98432, 1024, ⟨0, 98432⟩, ⟨2^20, 98432⟩, ⟨2^21, 98432⟩⟩
#eval ((launch d0).debug, (launch d0).numCtas, (launch d0).fusion, (launch d0).grid 0, (launch d0).grid 1, (launch d0).grid 2, (launch d0).blocks)
def d1 : Inst := ⟨2^31 - 1, 1024, ⟨0, 2^31-1⟩, ⟨2^34, 2^31-1⟩, ⟨2^35, 2^31-1⟩⟩
#eval (grid d1, (grid d1 - 1) * 1024 + 1023 < 2^31)
-- the bundle's ok excludes d1 although int32 offsets fit:
example : ¬ d1.ok := by intro h; have := h.2.1; simp [d1] at this
#check @TVA.exists_total_faithful
