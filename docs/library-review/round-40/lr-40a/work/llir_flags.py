"""Print the LLVM IR lines of masked_wrap.load_k that carry nsw/nuw/inbounds (lr-40a, round 40)."""
import sys, torch, triton
sys.path.insert(0, ".")
import masked_wrap as mw
X = torch.zeros(1 << 20, dtype=torch.int8, device="cuda"); Y = torch.zeros(1024, 16, dtype=torch.int8, device="cuda")
h = mw.load_k.warmup(X, Y, 1000, 16, 2149000, 1, BM=64, BN=16, grid=(16,))
for l in h.asm["llir"].splitlines():
    if any(f in l for f in (" nsw", " nuw", "inbounds", "getelementptr", " mul ")):
        print(l.strip()[:160])
