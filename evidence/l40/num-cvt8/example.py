"""TVA library evidence, L40 (sm_89): one input where `x / s` into `float8e4nv` differs from one rounding.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/l40/num-cvt8/example.py
Output: evidence/l40/num-cvt8/example.out.

`x = 1.0625` (`bfloat16`), `s = 1 - 2^-12` (`float32`): `x / s` is about `2.6e-4` above the `float8e4nv` midpoint
`1.0625` between `1` and `1.125`, less than a `float16` ulp, so the truncation into `float16` lands on the midpoint
and ties to even give `1`; one rounding gives `1.125`. This is the shape of `kv_cache_write` with an fp8 cache.
"""
import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, s_ptr, o_ptr):
    x = tl.load(x_ptr)
    o = (x / tl.load(s_ptr)).to(tl.float8e4nv)
    tl.store(o_ptr, o)
x = torch.tensor([1.0625], dtype=torch.bfloat16, device="cuda")
s = torch.tensor([1 - 2.0 ** -12], dtype=torch.float32, device="cuda")
o = torch.empty(1, dtype=torch.float8_e4m3fn, device="cuda")
k[(1,)](x, s, o)
print(torch.cuda.get_device_name(0), "key 1.0625 / k_scale", float(s), "-> fp8", float(o.float()), "; one rounding of", 1.0625 / float(s), "->", float(torch.tensor([1.0625 / float(s)]).to(torch.float8_e4m3fn).float()))
