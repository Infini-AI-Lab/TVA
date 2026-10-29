"""lr-20a, round 20 (L104): which objects the in-scope tutorials' hosts change, against scope 10's "no code of
the process ... changes a module, class or object of Triton" (REVIEWER_GUIDE item 10: "assigns to a Triton
module, class or object"). No GPU. Run from the TVA root:

    .venv/bin/python -I docs/library-review/round-20/lr-20a/work/scope10.py

Output: scope10.out beside this file.
"""
import inspect
import re

import triton
from triton.tools.tensor_descriptor import TensorDescriptor

print(f"triton {triton.__version__}")
print(f"TensorDescriptor is a class of Triton: {TensorDescriptor.__module__}.{TensorDescriptor.__name__}, "
      f"fields {list(TensorDescriptor.__dataclass_fields__)}")
print(f"triton.set_allocator ({triton.set_allocator.__module__}) changes Triton's state: "
      f"{inspect.getsource(triton.set_allocator).strip().splitlines()[-1].strip()}")
for path in ("tutorials/06-fused-attention.py", "tutorials/08-grouped-gemm.py", "tutorials/09-persistent-matmul.py"):
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if re.search(r"\]\.block_shape = |set_allocator\(", line):
                print(f"{path}:{n}: {line.strip()}")
