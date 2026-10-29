"""lr-15a: an independent AST scan of the in-scope tutorial kernels against SEMANTICS "The supported
subset" (library issue L88).

For every function of tutorials/inventory.json that is in scope (exclusions.md removes tutorial 05's
_layer_norm_bwd_dx_fused, tutorial 11, tutorial 10's block_scaled_matmul_kernel_cdna4 and tutorial
08's grouped_matmul_kernel), it prints the syntax it uses: statement and expression node kinds,
operators, calls, attributes, subscripts, decorators, parameters (annotations, defaults), annotated
assignments, tuple assignments (names vs elements), loop iterators, names read that the function does
not bind (module globals and builtins), and uses of a for-loop variable after its loop. Run from TVA's
root: python3 -I docs/library-review/round-15/lr-15a/work/subset_scan.py
"""
import ast, collections, json, os, sys

ROOT = os.getcwd()
TUT = os.path.join(ROOT, "tutorials")
EXCLUDED = {("05-layer-norm.py", "_layer_norm_bwd_dx_fused"),
            ("11-programmatic-dependent-launch.py", "add_kernel"),
            ("10-block-scaled-matmul.py", "block_scaled_matmul_kernel_cdna4"),
            ("08-grouped-gemm.py", "grouped_matmul_kernel")}


def dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        d = dotted(n.value)
        return (d + "." if d else "?.") + n.attr
    if isinstance(n, ast.Call):
        return dotted(n.func) + "(...)"
    if isinstance(n, ast.Subscript):
        return dotted(n.value) + "[...]"
    return type(n).__name__


def bound_names(fn):
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def scan(src, fn, out):
    out["decorators"].update(ast.unparse(d)[:90] for d in fn.decorator_list)
    a = fn.args
    if a.vararg or a.kwarg:
        out["varargs"].add(fn.name)
    if a.posonlyargs or a.kwonlyargs:
        out["posonly/kwonly"].add(fn.name)
    nd = len(a.defaults)
    for i, arg in enumerate(a.args):
        ann = ast.unparse(arg.annotation) if arg.annotation is not None else None
        out["param annotations"][ann] += 1
        if i >= len(a.args) - nd:
            out["param defaults"].add(f"{fn.name}.{arg.arg}={ast.unparse(a.defaults[i - (len(a.args) - nd)])}")
    bound = bound_names(fn)
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, ast.stmt):
            out["stmt"][type(node).__name__] += 1
        elif isinstance(node, ast.expr):
            out["expr"][type(node).__name__] += 1
        if isinstance(node, ast.BinOp):
            out["binop"][type(node.op).__name__] += 1
        if isinstance(node, ast.UnaryOp):
            out["unaryop"][type(node.op).__name__] += 1
        if isinstance(node, ast.BoolOp):
            out["boolop"][type(node.op).__name__] += 1
        if isinstance(node, ast.AugAssign):
            out["augassign"][type(node.op).__name__] += 1
        if isinstance(node, ast.Compare):
            out["compare"]["/".join(type(o).__name__ for o in node.ops)] += 1
            if len(node.ops) > 1:
                out["chained compare"].add(ast.unparse(node))
        if isinstance(node, ast.Call):
            out["calls"][dotted(node.func)] += 1
            if any(isinstance(x, ast.Starred) for x in node.args) or any(k.arg is None for k in node.keywords):
                out["star in call"].add(ast.unparse(node)[:80])
            for k in node.keywords:
                out["keywords"][f"{dotted(node.func)}:{k.arg}"] += 1
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node.value
            if not (isinstance(base, ast.Name) and base.id in ("tl", "triton")) and not (
                    isinstance(base, ast.Attribute) and dotted(base).startswith(("tl.", "triton."))):
                out["attributes (not tl.*)"][node.attr] += 1
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            out["attribute store"].add(ast.unparse(node))
        if isinstance(node, ast.Subscript):
            if not isinstance(node.ctx, ast.Load):
                out["subscript store"].add(ast.unparse(node))
            out["subscripts"][ast.unparse(node.slice)[:40]] += 1
        if isinstance(node, ast.Slice) and (node.lower or node.upper or node.step):
            out["slice with bounds"].add(ast.unparse(node))
        if isinstance(node, ast.AnnAssign):
            out["annassign"].add(f"{ast.unparse(node.target)}: {ast.unparse(node.annotation)} = "
                                 f"{ast.unparse(node.value) if node.value is not None else '<bare>'}")
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                out["multi-target assign"].add(ast.unparse(node))
            t = node.targets[0]
            if isinstance(t, (ast.Tuple, ast.List)):
                elts = node.value.elts if isinstance(node.value, (ast.Tuple, ast.List)) else None
                out["tuple assign"].add(f"{ast.unparse(t)} = {ast.unparse(node.value)[:60]}"
                                        + ("" if elts is None or len(elts) == len(t.elts) else "  <COUNT MISMATCH>"))
                if any(not isinstance(e, ast.Name) for e in t.elts):
                    out["tuple assign non-name target"].add(ast.unparse(t))
        if isinstance(node, ast.For):
            out["for iter"][dotted(node.iter.func) if isinstance(node.iter, ast.Call) else type(node.iter).__name__] += 1
            if node.orelse:
                out["for-else"].add(fn.name)
            if not isinstance(node.target, ast.Name):
                out["for non-name target"].add(ast.unparse(node.target))
        if isinstance(node, ast.While) and node.orelse:
            out["while-else"].add(fn.name)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            out["free names"][node.id] += 1
        if isinstance(node, ast.Return):
            out["returns"][f"{fn.name}:{'value' if node.value is not None else 'none'}"] += 1
        if isinstance(node, ast.JoinedStr):
            out["f-strings"].add(ast.unparse(node)[:80])
        if isinstance(node, ast.Constant):
            out["constant kinds"][type(node.value).__name__] += 1
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp,
                             ast.NamedExpr, ast.Starred, ast.Dict, ast.Set, ast.Await, ast.Yield)):
            out["other exprs"].add(type(node).__name__ + ": " + ast.unparse(node)[:60])
        if isinstance(node, (ast.Break, ast.Continue, ast.With, ast.Try, ast.Raise, ast.Global, ast.Nonlocal,
                             ast.Delete, ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            out["other stmts"].add(type(node).__name__)
        if isinstance(node, ast.IfExp):
            out["ifexp"].add(ast.unparse(node)[:80])
    # a for-loop variable read after its loop, in the same statement list
    def walk_lists(stmts):
        for i, s in enumerate(stmts):
            if isinstance(s, ast.For) and isinstance(s.target, ast.Name):
                v = s.target.id
                for later in stmts[i + 1:]:
                    reads = [n for n in ast.walk(later) if isinstance(n, ast.Name) and n.id == v and isinstance(n.ctx, ast.Load)]
                    stores = [n for n in ast.walk(later) if isinstance(n, ast.Name) and n.id == v and isinstance(n.ctx, ast.Store)]
                    if reads and not stores:
                        out["for var used after loop"].add(f"{fn.name}:{v} line {later.lineno}")
                        break
                    if stores:
                        break
            for field in ("body", "orelse"):
                sub = getattr(s, field, None)
                if isinstance(sub, list):
                    walk_lists(sub)
    walk_lists(fn.body)


def main():
    inv = json.load(open(os.path.join(TUT, "inventory.json")))
    out = collections.defaultdict(lambda: collections.Counter())
    for k in ["decorators", "varargs", "posonly/kwonly", "param defaults", "chained compare", "star in call",
              "attribute store", "subscript store", "slice with bounds", "annassign", "multi-target assign",
              "tuple assign", "tuple assign non-name target", "for-else", "for non-name target", "while-else",
              "f-strings", "other exprs", "other stmts", "ifexp", "for var used after loop"]:
        out[k] = set()
    count = 0
    for t in inv["tutorials"]:
        src = open(os.path.join(TUT, t["source"])).read()
        mod = ast.parse(src)
        want = {f["name"]: f for f in t["functions"] if (t["source"], f["name"]) not in EXCLUDED}
        for node in ast.walk(mod):
            if isinstance(node, ast.FunctionDef) and node.name in want and node.lineno <= want[node.name]["line"] <= node.end_lineno + 1:
                count += 1
                scan(src, node, out)
    print(f"in-scope functions scanned: {count}")
    for k, v in out.items():
        if isinstance(v, set):
            print(f"\n## {k} ({len(v)})")
            for x in sorted(v):
                print("  " + x)
        else:
            print(f"\n## {k}")
            for x, c in sorted(v.items(), key=lambda kv: str(kv[0])):
                print(f"  {x}: {c}")


if __name__ == "__main__":
    main()
