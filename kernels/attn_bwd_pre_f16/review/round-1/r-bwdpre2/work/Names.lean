import AttnBwdPre

/-! Reviewer r-bwdpre2: what the model's names elaborate to, and the promotion of line 260. -/

set_option pp.fullNames true
set_option pp.explicit false

#print Kernels.AttnBwdPre._attn_bwd_preprocess
#print Kernels.AttnBwdPre.model
#print Kernels.AttnBwdPre.launch
#print Kernels.AttnBwdPre.Post
#print Kernels.AttnBwdPre.OutSpec
#print Kernels.AttnBwdPre.Assumptions
#print axioms Kernels.AttnBwdPre.main

-- line 260: a float16 block times a float32 block
#eval TVA.BinOp.typing? .mul (.tensor .f16 true) (.tensor .f32 true)
-- line 254: an int32 scalar times the constexpr 128
#eval TVA.BinOp.typing? .mul (.tensor .i32 false) (.int 128)
-- lines 258-259: an int32 scalar times an int32 scalar
#eval TVA.BinOp.typing? .mul (.tensor .i32 false) (.tensor .i32 false)
-- lines 258-259 when N_CTX = 1 is a constexpr
#eval TVA.BinOp.typing? .mul (.tensor .i32 false) (.int 1)
-- line 258: an int32 block [BLOCK_M, 1] times the constexpr HEAD_DIM
#eval TVA.BinOp.typing? .mul (.tensor .i32 true) (.int 256)

-- every operator of the body with the instance it resolves to
set_option pp.notation false in
set_option pp.instances true in
set_option pp.explicit true in
#print Kernels.AttnBwdPre._attn_bwd_preprocess
