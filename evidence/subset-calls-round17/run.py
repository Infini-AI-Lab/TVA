"""Every call, method and `tl` attribute the in-scope tutorial kernels use, and where TVA gives it meaning.

Library evidence for library issue L88 (round 17): lr-15a found that the supported subset admits
`tl.cdiv`, which the in-scope kernels call, while neither SEMANTICS.md nor `TVA.Lang` gave it a
meaning. This scan lists, for the 23 in-scope functions of `tutorials/inventory.json` (every function
but tutorial 05's `_layer_norm_bwd_dx_fused`, tutorial 08's `grouped_matmul_kernel`, tutorial 10's
`block_scaled_matmul_kernel_cdna4` and tutorial 11, which exclusions.md excludes):
  - every call: of a `tl.*` function (`tl.x`, `tl.math.x`, …), of a method (`<expr>.x(…)`), of a
    Python builtin, of a helper of the same file, with the keywords the calls give;
  - every attribute read that is not a call (`tl.float16`, `x.dtype`, `acc.shape`, …);
  - the decorators, apart (they run on the host: SEMANTICS.md "The supported subset", Functions).
The parameters (annotations, defaults) and the body are scanned.
Each is matched with the place that gives it a meaning, from the table MEANING below (written by
reading SEMANTICS.md and `lean/TVA/Lang`), and the scan prints any name the table lacks.

No GPU. Run from TVA's root: python3 -I evidence/subset-calls-round17/run.py
"""
import ast, collections, json, os

ROOT = os.getcwd()
TUT = os.path.join(ROOT, "tutorials")
EXCLUDED = {("05-layer-norm.py", "_layer_norm_bwd_dx_fused"), ("08-grouped-gemm.py", "grouped_matmul_kernel"),
            ("10-block-scaled-matmul.py", "block_scaled_matmul_kernel_cdna4"),
            ("11-programmatic-dependent-launch.py", "add_kernel")}

# Where each name gets its meaning: a SEMANTICS.md section ("S: …") and/or a TVA.Lang definition.
MEANING = {
    # tl.* functions
    "tl.arange": "S: Values, Index arithmetic; TVA.tl.arange",
    "tl.assume": "S: Compiler hints; TVA.tl.assume",
    "tl.cdiv": "S: tl.cdiv (round 17)",
    "tl.dot": "S: tl.dot; TVA.tl.dot",
    "tl.dot_scaled": "S: tl.dot_scaled; TVA.Lang.Scaled",
    "tl.exp": "S: tl.exp, exp2, …; TVA.tl.exp",
    "tl.math.exp2": "S: tl.exp, exp2, …; TVA.tl.exp2",
    "tl.math.log2": "S: tl.exp, exp2, log, log2, …; TVA.tl.log2",
    "tl.join": "S: Shape changes; Tensor.join",
    "tl.load": "S: tl.load; TVA.tl.load",
    "tl.make_tensor_descriptor": "S: Tensor descriptors; TVA.Lang.Desc",
    "tl.max": "S: reductions, index reductions; TVA.tl.max1/max0/maxAll",
    "tl.max_contiguous": "S: Compiler hints; TVA.tl.max_contiguous(N), max_contiguous_multiple_of(N)",
    "tl.maximum": "S: tl.maximum, tl.minimum; Operators and promotion; TVA.tl.maximum",
    "tl.multiple_of": "S: Compiler hints; TVA.tl.multiple_of(N), multiple_of1",
    "tl.num_programs": "TVA.tl.num_programs",
    "tl.permute": "S: Shape changes; Tensor.permute",
    "tl.program_id": "S: Values, Launches; TVA.tl.program_id",
    "tl.rand": "S: Random numbers; TVA.Lang.Random",
    "tl.range": "S: Loops and unrolling, Loops; TVA.tl.forRange",
    "tl.reshape": "S: Shape changes; Tensor.reshape, TVA.tl.reshape_any",
    "tl.split": "S: Shape changes; Tensor.split",
    "tl.sqrt": "S: tl.exp, …, sqrt; TVA.tl.sqrt",
    "tl.static_assert": "S: tl.static_assert",
    "tl.store": "S: tl.store; TVA.tl.store",
    "tl.sum": "S: reductions; TVA.tl.sum/sum1/sum0 (and the c forms)",
    "tl.trans": "S: Shape changes; Tensor.trans, Tensor.permute",
    "tl.where": "S: Operators and promotion (tl.where); TVA.tl.where",
    "tl.zeros": "TVA.tl.zeros; S: Constants (tl.full)",
    "tl.pointer_type": "S: Values, Pointers (tl.intToPtr)",
    # methods
    ".to": "S: Float conversions, x.to(int), x.to(float), Operators and promotion (tl.convert); Pointers",
    ".reshape": "S: Shape changes",
    ".permute": "S: Shape changes",
    ".split": "S: Shape changes",
    ".trans": "S: Shape changes",
    ".load": "S: Tensor descriptors (tl.desc_load); TVA.Lang.Desc",
    ".store": "S: Tensor descriptors (tl.desc_store); TVA.Lang.Desc",
    # builtins
    "float": "S: Python's builtins",
    "isinstance": "S: Python's builtins",
    "min": "S: Operators and promotion (Python's min and max)",
    "range": "S: Loops and unrolling, Loops",
    "len": "S: Python's builtins",
    "int": "S: Python's builtins",
    "max": "S: Operators and promotion (Python's min and max)",
    # attributes
    "tl.constexpr": "S: The supported subset (annotations, parameters)",
    "tl.float16": "S: Values", "tl.bfloat16": "S: Values", "tl.float32": "S: Values",
    "tl.float8e4nv": "S: Values", "tl.float8e5": "S: Values", "tl.int32": "S: Values",
    "tl.int64": "S: Values", "tl.int8": "S: Values", "tl.uint8": "S: Values", "tl.int16": "S: Values",
    "tl.tensor_descriptor": "S: Python's builtins (isinstance)",
    ".dtype": "S: Expressions with no effect", ".element_ty": "S: Expressions with no effect",
    ".shape": "S: Expressions with no effect", ".T": "S: Shape changes",
}


def dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        d = dotted(n.value)
        return None if d is None else d + "." + n.attr
    return None


def scan():
    inv = json.load(open(os.path.join(TUT, "inventory.json")))
    found = collections.defaultdict(collections.Counter)
    keywords = collections.defaultdict(set)
    decorators = collections.Counter()
    nfun = 0
    for t in inv["tutorials"]:
        src = open(os.path.join(TUT, t["source"])).read()
        tree = ast.parse(src)
        want = {f["name"] for f in t["functions"] if (t["source"], f["name"]) not in EXCLUDED}
        helpers = {f["name"] for f in t["functions"]}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in want:
                continue
            nfun += 1
            where = t["source"] + ": " + node.name
            for dec in node.decorator_list:
                decorators[dotted(dec.func if isinstance(dec, ast.Call) else dec)] += 1
            # the parameters (annotations, defaults) and the body; the decorators run on the host
            parts = [node.args] + node.body
            called, inner = set(), set()
            for part in parts:
                for sub in ast.walk(part):
                    if isinstance(sub, ast.Call):
                        called.add(id(sub.func))
                    if isinstance(sub, ast.Attribute) and (dotted(sub.value) or "").startswith("tl"):
                        inner.add(id(sub.value))
            for part in parts:
                for sub in ast.walk(part):
                    if isinstance(sub, ast.Call):
                        d = dotted(sub.func)
                        if d is not None and (d.startswith("tl.") or d.startswith("triton.")):
                            key = d
                        elif isinstance(sub.func, ast.Attribute):
                            key = "." + sub.func.attr
                        elif isinstance(sub.func, ast.Name):
                            key = "helper " + sub.func.id if sub.func.id in helpers else sub.func.id
                        else:
                            key = "call of " + type(sub.func).__name__
                        found[key][where] += 1
                        keywords[key] |= {kw.arg or "**" for kw in sub.keywords}
                    elif isinstance(sub, ast.Attribute) and id(sub) not in called and id(sub) not in inner:
                        d = dotted(sub)
                        if d is not None and d.startswith("tl."):
                            found[d][where] += 1
                        elif isinstance(sub.ctx, ast.Load):
                            found["." + sub.attr][where] += 1
    return nfun, found, keywords, decorators


def main():
    nfun, found, keywords, decorators = scan()
    print(f"in-scope functions scanned: {nfun}")
    print("decorators (SEMANTICS 'The supported subset', Functions; an autotune's arguments run on the host): "
          + ", ".join(f"{d} x{n}" for d, n in sorted(decorators.items())))
    missing = []
    for name in sorted(found):
        uses = found[name]
        total = sum(uses.values())
        if name.startswith("helper "):
            meaning = "S: Helpers are inlined (a @triton.jit function of the same file)"
        else:
            meaning = MEANING.get(name)
        if meaning is None:
            missing.append(name)
            meaning = "NO MEANING FOUND"
        kws = f"; keywords {sorted(keywords[name])}" if keywords.get(name) else ""
        print(f"{name}: {total} uses in {len(uses)} functions{kws} -- {meaning}")
    print(f"names without a stated meaning: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
