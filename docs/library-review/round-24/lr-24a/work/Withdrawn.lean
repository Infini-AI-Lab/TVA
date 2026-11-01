import TVA

/-! lr-24a: the facts withdrawn in round 24 (L125) no longer exist; each `#check` below must fail with an unknown
identifier. Checked with `lake env lean` on this file only. -/

#check @TVA.Rules.Num.DotE5m2
#check @TVA.Rules.Num.DotE4m3
#check @TVA.Rules.Num.DotE4m3E5m2
#check @TVA.Rules.Num.DotE5m2E4m3
#check @TVA.Rules.Num.DotF8
#check @TVA.Platform.sm90ref.dotE5m2
#check @TVA.Dot8Chunk.LiftAcc
