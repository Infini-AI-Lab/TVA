"""lr-15b: an independent AST scan of the in-scope tutorial kernels against SEMANTICS "The supported subset".

For each function of tutorials/inventory.json that the library keeps in scope (not tutorial 05's
_layer_norm_bwd_dx_fused, not tutorial 08's grouped_matmul_kernel, not tutorial 10's
block_scaled_matmul_kernel_cdna4, not tutorial 11), list:
  - every AST node type in the body (decorators and signature separately);
  - every statement kind, assignment target shape, annotation;
  - every call's callee text, with its keyword names and its number of positional arguments;
  - every attribute name read, every subscript's slice shape;
  - every Name read that the function does not bind (parameters, assigned names, loop targets):
    these resolve to module-level values or builtins;
  - every comparison with more than one operator, every BoolOp, IfExp, Starred, Lambda, etc.
Run: python3 -I work/subset_scan.py  (from TVA's root; reads tutorials/ only)
"""
import ast, collections, json, os, sys

ROOT = os.getcwd()
TUT = os.path.join(ROOT, "tutorials")
EXCLUDED = {("05-layer-norm.py", "_layer_norm_bwd_dx_fused"), ("08-grouped-gemm.py", "grouped_matmul_kernel"),
            ("10-block-scaled-matmul.py", "block_scaled_matmul_kernel_cdna4")}
EXCLUDED_FILES = {"11-programmatic-dependent-launch.py"}

inv = json.load(open(os.path.join(TUT, "inventory.json")))

node_types = collections.Counter()
stmt_types = collections.Counter()
calls = collections.defaultdict(set)          # callee text -> {(npos, kw names, starred)}
attrs = collections.Counter()
subscripts = collections.Counter()
free_names = collections.defaultdict(set)
annassigns = []
assign_targets = collections.Counter()
augops = collections.Counter()
binops = collections.Counter()
unaryops = collections.Counter()
cmpops = collections.Counter()
boolops = collections.Counter()
ifexps = []
decorators = collections.Counter()
params = []
constants = collections.Counter()
for_iters = []
returns = collections.Counter()
helper_calls_kw = []
loopvar_after = []

for t in inv["tutorials"]:
    src_name = t["source"]
    if src_name in EXCLUDED_FILES:
        continue
    src = open(os.path.join(TUT, src_name)).read()
    tree = ast.parse(src)
    module_names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tg in node.targets:
                for n in ast.walk(tg):
                    if isinstance(n, ast.Name):
                        module_names.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                module_names.add((a.asname or a.name).split(".")[0])
    funcs = {f["name"]: f for f in t["functions"]}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in funcs:
            continue
        if (src_name, node.name) in EXCLUDED:
            continue
        if node.lineno > funcs[node.name]["end_line"] or node.end_lineno < funcs[node.name]["line"]:
            continue
        where = f"{src_name}:{node.name}"
        for d in node.decorator_list:
            decorators[ast.unparse(d)[:90]] += 1
        a = node.args
        for p in a.posonlyargs + a.args + a.kwonlyargs:
            params.append((where, p.arg, ast.unparse(p.annotation) if p.annotation else None))
        if a.defaults or a.kw_defaults or a.vararg or a.kwarg:
            params.append((where, "DEFAULTS/VARARGS", ast.unparse(node.args)))
        bound = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
        for stmt in node.body:
            for n in ast.walk(stmt):
                node_types[type(n).__name__] += 1
                if isinstance(n, ast.stmt):
                    stmt_types[type(n).__name__] += 1
                if isinstance(n, ast.Assign):
                    assign_targets[type(n.targets[0]).__name__ + ("" if len(n.targets) == 1 else " chained")] += 1
                    if isinstance(n.targets[0], ast.Tuple):
                        nv = len(n.value.elts) if isinstance(n.value, ast.Tuple) else "?"
                        assign_targets[f"tuple {len(n.targets[0].elts)} names <- {type(n.value).__name__} {nv}"] += 1
                if isinstance(n, ast.AnnAssign):
                    annassigns.append((where, n.lineno, ast.unparse(n)))
                if isinstance(n, ast.AugAssign):
                    augops[type(n.op).__name__ + " target " + type(n.target).__name__] += 1
                if isinstance(n, ast.BinOp):
                    binops[type(n.op).__name__] += 1
                if isinstance(n, ast.UnaryOp):
                    unaryops[type(n.op).__name__] += 1
                if isinstance(n, ast.Compare):
                    cmpops[" ".join(type(o).__name__ for o in n.ops)] += 1
                if isinstance(n, ast.BoolOp):
                    boolops[type(n.op).__name__ + f" x{len(n.values)}"] += 1
                if isinstance(n, ast.IfExp):
                    ifexps.append((where, n.lineno, ast.unparse(n)[:100]))
                if isinstance(n, ast.Call):
                    calls[ast.unparse(n.func)].add((len(n.args), tuple(k.arg for k in n.keywords),
                                                    any(isinstance(x, ast.Starred) for x in n.args)))
                if isinstance(n, ast.Attribute):
                    attrs[n.attr] += 1
                if isinstance(n, ast.Subscript):
                    subscripts[ast.unparse(n.slice)[:60]] += 1
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
                    free_names[n.id].add(where)
                if isinstance(n, ast.Constant):
                    constants[type(n.value).__name__] += 1
                if isinstance(n, ast.For):
                    for_iters.append((where, n.lineno, ast.unparse(n.target), ast.unparse(n.iter)))
                if isinstance(n, ast.Return):
                    returns[(where, "value" if n.value is not None else "none")] += 1

def show(title, obj):
    print(f"== {title}")
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            v = obj[k]
            if isinstance(v, set):
                v = sorted(v, key=str)
            print(f"  {k}: {v}")
    else:
        for x in obj:
            print(f"  {x}")

show("node types", node_types)
show("statement types", stmt_types)
show("assign targets", assign_targets)
show("annotated assignments", annassigns)
show("augmented assignments", augops)
show("binary operators", binops)
show("unary operators", unaryops)
show("comparisons", cmpops)
show("bool ops", boolops)
show("conditional expressions", ifexps)
show("calls (callee: {(n positional, keywords, starred)})", calls)
show("attributes read", attrs)
show("subscript slices", subscripts)
show("free names (module level or builtins)", free_names)
show("constant kinds", constants)
show("for loops", for_iters)
show("returns", returns)
show("decorators", decorators)
show("parameters (where, name, annotation)", params)
