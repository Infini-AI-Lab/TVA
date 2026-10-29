import TVA.Lang.Promote
/-! lr-15a: `whereTyping?` on the three forms TRANSCRIPTION §2's `tl.where` rows give (library issue
L83): an `int32` block with `0.0`, an `int32` block with `0`, and two Python numbers under a block
condition. The rows say `float32`, `int32`, `float32`. -/
open TVA
#eval (whereTyping? (.tensor .i32 true) (.tensor .i32 true) (.float (FloatV.ofPat 0))).map (·.res)
#eval (whereTyping? (.tensor .i1 true) (.tensor .i32 true) (.int 0)).map (·.res)
#eval (whereTyping? (.tensor .i1 true) (.int 0) (.float (FloatV.ofPat 0xC12E848000000000))).map (fun t => (t.res, t.block))
#eval (whereTyping? (.tensor .i1 true) (.tensor .bf16 true) (.float (FloatV.ofPat 0))).map (·.res)
#eval (whereTyping? (.tensor .i1 true) (.tensor .i64 true) (.float (FloatV.ofPat 0))).map (·.res)
