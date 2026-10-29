"""lr-18a, round 18: another module rebinds a name of the kernel's module (rebind_mod.py) before the launch.

SEMANTICS "The supported subset" (Names) admits a module-level name of the kernel's module only as its own `import`
or `def` binds it, "and no other statement of the module may bind it"; Compiler.Faithful's scope names no
condition on other modules. Here the kernel module is untouched and in the subset; the caller, a different module,
assigns `mod.double = mod.triple` before the first launch (A), or after a first launch (B). Triton 3.8.0's front end
reads the function's `__globals__`, the module's dict, when it compiles (code_generator.py `global_lookup`), and
checks the globals it recorded at each later launch (jit.py:761-766).

Run on GPU 0 from this folder: CUDA_VISIBLE_DEVICES=0 python -I rebind_ext.py > rebind_ext.out
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
import torch  # noqa: E402
import triton  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "rebind_mod.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


x = torch.arange(1, 17, dtype=torch.float32, device="cuda")
print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")

# A: rebound by the caller before the first launch
a = load("rebind_a")
a.double = a.triple
o = torch.zeros_like(x)
a.k[(1,)](x, o)
torch.cuda.synchronize()
print(f"A, `mod.double = mod.triple` before the first launch: o / x = {sorted(set((o / x).tolist()))}")

# B: rebound after a first launch, then launched again
b = load("rebind_b")
o = torch.zeros_like(x)
b.k[(1,)](x, o)
torch.cuda.synchronize()
print(f"B, first launch (as the source binds it): o / x = {sorted(set((o / x).tolist()))}")
b.double = b.triple
o = torch.zeros_like(x)
try:
    b.k[(1,)](x, o)
    torch.cuda.synchronize()
    print(f"B, launch after `mod.double = mod.triple`: o / x = {sorted(set((o / x).tolist()))}")
except Exception as e:  # noqa: BLE001 - reported
    print(f"B, launch after `mod.double = mod.triple`: {type(e).__name__}: {str(e)[:200]}")
