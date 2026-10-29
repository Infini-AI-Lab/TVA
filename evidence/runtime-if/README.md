# Runtime `if`, `elif`, conditional expressions, `and`/`or` and early `return`

**Question.** What does Triton 3.8.0 do with an `if` (or `elif`, or `x if c else y`) whose
condition is a runtime value, with `and`, `or` and `not` on runtime values, and with a `return`
inside a runtime `if`? Which of these are rejected, with what error, and when? Library issue L28
(round 8).

The front end's code: `visit_If` (`primitives/triton-3.8.0-code-generator.py` lines 967-996)
rejects a block condition with more than one element (970-972), takes a one-element block's
element with a warning (973-977), casts the condition to `int1` (978), and emits an unstructured
branch (`cf.cond_br`, `visit_if_top_level`, 893-929) when the `if` contains a `return`
(`ContainsReturnChecker`, 144-229; it does not look into called functions, 152-156, 227-228),
refusing that inside a loop (979-982), and an `scf.if` otherwise (`visit_if_scf`, 932-965). The
branches' names are merged by `visit_then_else_blocks` (834-891). A compile-time condition is
evaluated by Python and only the taken side is visited (986-996). `visit_compound_statement`
stops at a `return` (508-517). The conditional expression is `visit_IfExp` (998-1049);
`and`/`or` are `visit_BoolOp` (1481-1531). The cast to `int1` is `x != 0`
(`primitives/triton-3.8.0-semantic.py` lines 873-877 for integers, 884-887 for floats, with
`not_equal` at 568-577: `cmpi ne`, `cmpf une`); `logical_and`, `logical_or` and `not_` bitcast a
non-`int1` operand to `int1` (440-457), which fails for wider types (807).

**Method.** `run.py` compiles and launches small kernels on GPU 0, one process per case (R2f faults
on purpose), with a fresh Triton cache so that every kernel is compiled and its compile-time
warnings show. Stores go into buffers filled with a sentinel (`-777`) and atomics count into
counters, so the host sees, per program, which operations ran. For each kernel the script reports
markers of the front end's output (`asm['source']`): `scf.if`, `cf.cond_br`, the comparison the
condition became, `tt.atomic_rmw`, `tt.call`. For a rejected kernel it reports the exception, the
compiler's diagnostic where there is one, and whether the watched buffers stayed untouched.

- R1: `if pid % 3 == 0:` with a store and an atomic in each branch, 64 programs.
- R2: an atomic only in the then branch of `if flag == 7:` (`flag` a runtime `int32`), and of
  `if tl.load(x + pid) > 0:`; a store 2^40 elements out of bounds in the untaken branch. R2f: the
  same store in the taken branch (control).
- R3: `if n:` with `n` a runtime `int32` or `int64`, and `if x:` with `x` a runtime `float32`.
- R4: `if v > 0:` with `v` a block of 4, and of 1, elements.
- R5: a name assigned in one branch only and used after the `if`; a live-in name given another
  dtype in one branch; a new name given different dtypes, or shapes, in the two branches; a
  live-in name changed in one branch and a new name defined in both (no error expected).
- R6: `if pid % 2 == 1: return` at the kernel's top level; a `return` in the `else` branch; a
  `return` at top level with code after it; a kernel that returns a value.
- R7: a `return` in an `if` nested in a runtime `if`; in a runtime `if` in a `for`, in a `while`,
  and directly in a `for` body; in a runtime `if` in a `tl.static_range` loop; a helper with an
  early `return` called at top level and called in a `for` loop; a helper with a `return` in a
  runtime `if` in its own `for` loop.
- R8: `tl.atomic_add(a, 1) if pid % 2 == 0 else tl.atomic_add(b, 1)`; a block condition;
  an `int32` side and a `float32` side.
- R9: an `elif` chain whose second condition runs an atomic; `and` and `or` of two runtime
  conditions, each running an atomic; `<runtime condition with an atomic> and False` (a
  constexpr); `n and m` and `not n` on runtime `int32`; `not c` on an `int1`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/runtime-if/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | construct | front end | device |
|---|---|---|---|
| R1 | runtime `if`/`else`, `int1` condition | `scf.if` | each program ran exactly its taken branch (22 then, 42 else) |
| R2 | atomic / out-of-bounds store in the untaken branch | `scf.if` | did not run (count 0; no fault); ran when taken (count 64) |
| R2f | the out-of-bounds store taken | `scf.if` | illegal memory access |
| R3 | `int32`/`int64` condition | `arith.cmpi ne` | taken iff `n != 0` (`256` and `2^32` taken) |
| R3 | `float32` condition | `arith.cmpf une` | taken iff `x != 0` unordered: `0.0`, `-0.0` not taken; `nan`, `1e-45`, `-inf` taken |
| R4 | block of 4 as condition | rejected: `UnsupportedLanguageConstruct: Boolean value of Tensor with more than one value is ambiguous` | nothing ran |
| R4 | block of 1 as condition | warning `If conditional called with multidimensional Tensor instead of scalar ...`; `scf.if` | branch by the element |
| R5 | name assigned in one branch, used after | rejected: `NameError('y is not defined')` | nothing ran |
| R5 | live-in redefined with another dtype | rejected: ``AssertionError('initial value for `y` is of type int32[], but the then block redefines it as fp32[]')`` | nothing ran |
| R5 | new name, different dtypes / shapes | rejected: `AssertionError('Mismatched type for y between then block (int32) and else block (fp32)')` (shapes likewise) | nothing ran |
| R5 | live-in changed in one branch; new name in both | `scf.if` with results | merged values correct |
| R6 | `return` in a top-level runtime `if` (then or else) | `cf.cond_br` | programs that return run nothing after it (counts `[10, 54, 10]`) |
| R6 | `return` at top level, code after it | the code after it is not emitted | counts `[64, 0]` |
| R6 | kernel returns a value | rejected: `RuntimeError: PassManager::run failed` (`failed to legalize operation 'tt.return'`) | nothing ran |
| R7 | `return` in an `if` nested in a runtime `if` | `cf.cond_br` x2 | correct (`[32, 16, 48]`) |
| R7 | `return` in a runtime `if` in `for` or `while` | rejected: ``UnsupportedLanguageConstruct: Cannot have `return` statements inside `while` or `for` statements in triton.`` | nothing ran |
| R7 | `return` directly in a `for` body | rejected: `AssertionError('We use SCF, so the loop body should only have one block')` | nothing ran |
| R7 | `return` in a runtime `if` in `tl.static_range` | unrolled; `cf.cond_br` x4 | correct (`[7, 6, 5, 4]`) |
| R7 | helper with an early `return`, at top level or in a `for` loop | accepted; `tt.call` (not inlined in the loop) | correct |
| R7 | helper with a `return` in a runtime `if` in its own loop | rejected (as for `for` above) | nothing ran |
| R8 | `x if c else y`, runtime scalar `c`, atomics on both sides | `scf.if` | only the taken side ran (32 and 32) |
| R8 | block condition | rejected: `RuntimeError: error encountered during parsing` (`'scf.if' op operand #0 must be 1-bit signless integer, but got 'tensor<4xi1>'`) | nothing ran |
| R8 | `int32` side and `float32` side | rejected: `AssertionError('Ternary expression with dynamic condition has inconsistent types int32 and fp32')` | nothing ran |
| R9 | `elif` chain | nested `scf.if` x3 | an `elif` condition runs only in programs that reach it (62 of 64) |
| R9 | `and` / `or` of runtime conditions | both operands emitted; one `scf.if` | both atomics ran in all 64 programs (no short circuit) |
| R9 | `<runtime> and False` | the runtime operand is emitted; the `if` is compile-time false | atomic ran 64 times; branch never taken |
| R9 | `n and m`, `not n` on `int32` | rejected: `ValueError('Cannot bitcast data-type of size 32 to data-type of size 1')` | nothing ran |
| R9 | `not c` on `int1` | `scf.if` | correct |

Every rejection is raised by the launch call on the host, at compile time, before any program
runs (every watched buffer still holds its sentinel).

**Conclusions**, used by `docs/SEMANTICS.md`:

1. A runtime `if` (scalar condition) runs, in each program, the operations of the taken branch
   only; the untaken branch's operations (loads, stores, atomics) do not run. The condition is
   first converted to `int1` as `c != 0` (for floats, unordered: NaN is true, `-0.0` false).
2. A block condition with more than one element is rejected on the host; a one-element block is
   read as its element (with a warning).
3. After the `if`, a name has the taken branch's value if it was live before the `if` or is
   defined in both branches, with the same type (dtype and shape) in both; otherwise the kernel is
   rejected on the host (a name defined in one branch only is not defined after the `if`).
4. A `return` inside a runtime `if` that is not inside a `for` or `while` loop (of the same
   function) ends the program (or the helper) for the programs that take it; in a loop it is
   rejected on the host. A `tl.static_range` loop is unrolled and does not count as a loop here.
   Code after a top-level `return` is not compiled. A kernel cannot return a value (rejected).
5. `x if c else y` with a runtime scalar `c` evaluates only the taken side; both sides must have
   the same type, and a block condition is rejected.
6. `and` and `or` evaluate every runtime operand (no short circuit at run time); a compile-time
   operand short-circuits at compile time, after the operands before it have been emitted.
   `and`, `or` and `not` on a runtime operand other than `int1` are rejected on the host. An
   `elif` condition is part of the `else` branch: it runs only in programs that reach it.
