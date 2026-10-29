"""lr-16a, Part U on the device: an unused function's def statement rebinds a module-level name the kernel reads.

Two sources (`probes_src.py`), each imported from a temporary file (`@triton.jit` reads its function's source
file) and launched once on 16 `float32` values `x = 0..15`:
- SCALE_SRC: the kernel lines bind `SCALE = tl.constexpr(2)` and the kernel stores `x * SCALE`; the unused
  `_unused_helper(x=(SCALE := tl.constexpr(3)))` rebinds `SCALE` when Python executes its `def` at import;
- JIT_SRC: the kernel stores `double(x)`, the `@triton.jit` helper `double` returning `x * 2`; the unused
  `_unused_helper(f=(double := triple))` rebinds `double` to the unused `@triton.jit` function `triple` (`x * 3`).
Triton 3.8.0 resolves the kernel's global names at compile time. What does the kernel store?
Run: `CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I unused_gpu.py > unused_gpu.out`.
"""
import importlib.util
import os
import sys
import tempfile

import torch
import triton

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probes_src import SOURCES  # noqa: E402

print(f"Triton {triton.__version__}, {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
d = tempfile.mkdtemp()
x = torch.arange(16, dtype=torch.float32, device="cuda")
for name, src in SOURCES.items():
    p = os.path.join(d, f"unused_{name.lower()}.py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location(f"unused_{name.lower()}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    o = m.run()
    torch.cuda.synchronize()
    bound = m.SCALE if name == "SCALE" else m.double.fn.__name__
    print(f"- {name}_SRC: after import the module's {'SCALE' if name == 'SCALE' else 'double'} is {bound!r}; "
          f"the kernel stored {o.tolist()}; o == 2·x: {bool(torch.equal(o, 2 * x))}, o == 3·x: {bool(torch.equal(o, 3 * x))}")
