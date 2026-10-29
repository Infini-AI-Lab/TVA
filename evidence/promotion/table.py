"""The result type of Triton 3.8.0's operators on every pair of operand kinds, as the front end
computes it (library issue L45, round 8).

The script builds IR the way the front end does (`ir.context`, the CUDA backend's dialects, options,
`codegen_fns` and `module_map`, and a `CodeGenerator` from `triton.compiler.code_generator`), opens a
function body, binds the operands to the names `a`, `b` (and `c` for `tl.where`) in the generator's
local scope, and visits a parsed expression such as `a + b`, `a < b`, `a and b`, `tl.minimum(a, b)`,
`min(a, b)`, `tl.where(c, a, b)` or `-a`. So the dispatch is the front end's own: `visit_BinOp`,
`visit_Compare`, `visit_BoolOp`, `visit_UnaryOp` and `visit_Call`, then the tensor's methods in
`core.py` and `TritonSemantic`. Only types are recorded; no GPU is needed.

Operands (spelling in the table):
  t:<dtype>   a block tensor of shape [4]
  s:<dtype>   a scalar (0-d) tensor, as a scalar kernel argument or a value loaded from one
  int:<v>     the Python int literal v (the front end's `constexpr(v)`)
  float       the Python float literal 1.5
  bool        the Python literal True
Results: t:<dtype>, s:<dtype>, py:<value> (a compile-time Python value; `py:big` for an int of
128 bits or more, which only Python-int shifts produce), or `error` with a fifth
column holding the exception's class and the first line of its message.

`tl.where`: the condition ranges over every operand with the values (t:fp32, t:fp32) and with
(s:fp32, s:fp32) (op `where:<cond>`, columns `<x>`, `<y>`); then the values range over every pair
with the conditions t:int1 and s:int1.

Run: .venv/bin/python -I evidence/promotion/table.py > evidence/promotion/table.tsv  (from TVA's root)
"""
import ast
import sys
import warnings

import triton
import triton.language as tl
from triton import knobs
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.code_generator import CodeGenerator
from triton.compiler.compiler import make_backend

DTYPES = ["int1", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "fp8e5", "fp8e4nv", "fp16", "bf16", "fp32", "fp64"]
INTS = [0, 1, -1, 2**31 - 1, 2**31, -2**31 - 1, 2**63, 2**64 - 1]

BINARY = [("+", "a + b"), ("-", "a - b"), ("*", "a * b"), ("/", "a / b"), ("//", "a // b"),
          ("%", "a % b"), ("&", "a & b"), ("|", "a | b"), ("^", "a ^ b"), ("<<", "a << b"),
          (">>", "a >> b"), ("<", "a < b"), ("<=", "a <= b"), (">", "a > b"), (">=", "a >= b"),
          ("==", "a == b"), ("!=", "a != b"), ("tl.minimum", "tl.minimum(a, b)"),
          ("tl.maximum", "tl.maximum(a, b)"), ("and", "a and b"), ("or", "a or b"),
          ("min", "min(a, b)"), ("max", "max(a, b)")]
UNARY = [("neg", "-a"), ("invert", "~a"), ("not", "not a"), ("pos", "+a")]


@triton.jit
def _probe():
    pass


def operand_names():
    names = [f"t:{d}" for d in DTYPES] + [f"s:{d}" for d in DTYPES]
    names += [f"int:{v}" for v in INTS] + ["float", "bool"]
    return names


class Probe:

    def __init__(self):
        target = GPUTarget("cuda", 90, 32)
        backend = make_backend(target)
        self.context = ir.context()
        ir.load_dialects(self.context)
        backend.load_dialects(self.context)
        options = backend.parse_options({})
        codegen_fns = backend.get_codegen_implementation(options)
        module_map = backend.get_module_map()
        self.gen = CodeGenerator(self.context, None, gscope={"tl": tl}, function_name="probe", jit_fn=_probe,
                                 options=options, codegen_fns=codegen_fns, module_map=module_map,
                                 is_gluon=False, file_name="probe.py")
        b = self.gen.builder
        fn = b.get_or_insert_function(self.gen.module, "probe", b.get_function_ty([], []), "public", False)
        self.gen.module.push_back(fn)
        self.gen.fn = fn
        b.set_insertion_point_to_start(fn.add_entry_block())
        self.sem = self.gen.semantic

    def operand(self, name):
        kind, _, rest = name.partition(":")
        if kind == "t":
            return self.sem.full([4], 0, getattr(tl, _tl_name(rest)))
        if kind == "s":
            return self.sem.scalar_constant(0, getattr(tl, _tl_name(rest)))
        if kind == "int":
            return tl.constexpr(int(rest))
        if name == "float":
            return tl.constexpr(1.5)
        if name == "bool":
            return tl.constexpr(True)
        raise ValueError(name)

    def run(self, expr, **binds):
        self.gen.lscope = {k: self.operand(v) for k, v in binds.items()}
        node = ast.parse(expr, mode="eval").body
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = self.gen.visit(node)
        except Exception as e:  # the front end's rejection
            msg = (str(e).strip().splitlines() or [""])[0][:200].replace("\t", " ")
            return "error\t" + type(e).__name__ + ": " + msg
        return spell(r)


def _tl_name(d):
    return {"fp8e5": "float8e5", "fp8e4nv": "float8e4nv", "fp16": "float16", "bf16": "bfloat16",
            "fp32": "float32", "fp64": "float64"}.get(d, d)


def spell(r):
    if isinstance(r, tl.constexpr):
        r = r.value
    if isinstance(r, tl.tensor):
        return ("t:" if r.type.is_block() else "s:") + r.type.scalar.name
    if isinstance(r, int) and not isinstance(r, bool) and abs(r) >= 2**128:
        return "py:big"   # a compile-time int of more than 128 bits (Python's own arithmetic)
    if isinstance(r, (int, float, bool)):
        return f"py:{r!r}"
    return f"other:{type(r).__name__}"


def main():
    knobs.compilation.front_end_debugging = True   # raise the front end's own exception, unwrapped
    p = Probe()
    ops = operand_names()
    out = sys.stdout
    for op, expr in BINARY:
        for a in ops:
            for b in ops:
                out.write(f"{op}\t{a}\t{b}\t{p.run(expr, a=a, b=b)}\n")
    for op, expr in UNARY:
        for a in ops:
            out.write(f"{op}\t{a}\t-\t{p.run(expr, a=a)}\n")
    for x in ("t:fp32", "s:fp32"):
        for c in ops:
            out.write(f"where:{c}\t{x}\t{x}\t{p.run('tl.where(c, a, b)', c=c, a=x, b=x)}\n")
    for c in ("t:int1", "s:int1"):
        for a in ops:
            for b in ops:
                out.write(f"where:{c}\t{a}\t{b}\t{p.run('tl.where(c, a, b)', c=c, a=a, b=b)}\n")


if __name__ == "__main__":
    main()
