# The order of a call's arguments, runtime conditional expressions, and `launch_metadata=`

Library evidence (not a kernel's), for library issue L46 (round 7). Triton 3.8.0
(`~/fpv/tva/.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28, default launch
options. Script: [run.py](run.py); output: [out.txt](out.txt).

**Questions.** SEMANTICS.md, "How the front end evaluates the source" and "Launches", states three
facts:
1. a call's keyword arguments are evaluated before its positional ones;
2. in `x if c else y` with a runtime `c`, only the taken side runs;
3. `launch_metadata=` on `@triton.jit` changes no result.

**What Triton's sources say** (the pinned files in `primitives/`):
- `visit_Call` (`triton-3.8.0-code-generator.py:1455-1476`) visits the called function's
  expression, then every keyword argument (`kws = dict(self.visit(keyword) …)`), then the positional
  arguments; `static_executor` (`:1639-1650`) does the same for the static calls.
- `visit_IfExp` (`:998-1036`): with a tensor condition, each side is visited into its own block, and
  the blocks become the `then` and `else` regions of one `scf.if`; with a compile-time condition,
  only the side it picks is visited (`:1037-1049`).
- `launch_metadata` is stored on the `JITFunction` (`triton-3.8.0-jit.py:787-797`) and is not part
  of what is compiled. At a launch, `CompiledKernel.launch_metadata`
  (`triton-3.8.0-compiler.py:493-502`) wraps the function, with the grid, the kernel's metadata and
  the arguments, in a `LazyDict` (`:374-387`), which calls it only when someone calls `.get()`. The
  launcher passes that object to `knobs.runtime.launch_enter_hook` and `launch_exit_hook`
  (`triton-3.8.0-jit.py:778-779`; `triton-3.8.0-nvidia-driver.c:1406-1415, 1447, 1519`), which are
  empty `HookChain`s unless a hook is installed (`triton-3.8.0-knobs.py:415-433, 477-478`). So
  without an installed hook the function never runs.

**Method.**
- **K.** Two kernels whose calls hold, as positional and keyword arguments, atomics of different
  kinds (`add`, `xor`, `max`, `and`, `or`, `xchg`); the script prints the order of the
  `tt.atomic_rmw` operations in the TTIR, which is the order the front end emitted them.
- **C.** `v = tl.atomic_add(c, 1) if flag > 0 else tl.atomic_add(c, 100)` with `flag` a runtime
  `int32` (5 or -5, not 1, which Triton would make a constant), 4 programs; the counter shows which
  sides ran.
- **M.** The same function compiled as `triton.jit(body)` and `triton.jit(launch_metadata=meta)(body)`,
  where `meta` records each call; launched without a hook, then with a launch-enter hook that reads
  the metadata, then without it again.

**Result** (`out.txt`).

| Case | Result |
|---|---|
| K1 `tl.store(out, <add>, mask=<xor> >= 0)` | `xor`, then `add`: the keyword argument first |
| K2 `tl.store(out + <max>, v, mask=<and> >= 0, cache_modifier=…)`; `tl.atomic_add(c, <or>, mask=<xchg> >= 0)` | `and`, `max`, then `exch`, `or`, `add`: in each call the keyword arguments first, then the positional ones, then the call itself |
| C `flag = 5` | counter 4 (only the `then` side ran in each program; both sides would give 404), with an `scf.if` in the TTIR |
| C `flag = -5` | counter 400 (only the `else` side) |
| M1 no hook | `meta` ran 0 times |
| M2 | the PTX with and without `launch_metadata` is identical, and so are the results |
| M3 a hook that calls `.get()` | `meta` ran once, and the hook saw its values |
| M4 the hook removed | no further call |

**Conclusions.**
1. The front end evaluates a call's keyword arguments before its positional ones. A model is not
   affected: only which operations are evaluated matters, since a well-behaved model's accesses to
   the same bytes with a write among them are atomics, which the semantics performs in every order
   (SEMANTICS.md, "Order").
2. A conditional expression with a runtime condition runs only the taken side. What such a runtime
   branch means in a model stays open (library issue L28).
3. `launch_metadata=` changes neither the compiled code nor any result, and its function runs only
   for an installed launch hook, which `Compiler.Faithful`'s scope excludes; a model omits it.
