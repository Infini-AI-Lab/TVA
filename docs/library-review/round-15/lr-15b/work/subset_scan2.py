"""lr-15b: three more syntactic checks of the in-scope tutorial kernels against the supported subset.

  A. a `for` loop's variable read after the loop (lexically after it, in the same function);
  B. a name first assigned inside a `for`/`while` body and read after the loop;
  C. every assignment of a non-tensor-looking value (None, a dtype `tl.<type>`, a conditional
     expression of dtypes, a string), and every `if` whose branches assign such a name;
  D. expression statements that are not calls.
Run: python3 -I work/subset_scan2.py  (from TVA's root; reads tutorials/ only)
"""
import ast, json, os

ROOT = os.getcwd()
TUT = os.path.join(ROOT, "tutorials")
EXCLUDED = {("05-layer-norm.py", "_layer_norm_bwd_dx_fused"), ("08-grouped-gemm.py", "grouped_matmul_kernel"),
            ("10-block-scaled-matmul.py", "block_scaled_matmul_kernel_cdna4")}
inv = json.load(open(os.path.join(TUT, "inventory.json")))
DTYPES = {"float16", "float32", "bfloat16", "float8e5", "float8e4nv", "int32", "int64", "int8", "int16",
          "uint8", "uint32", "float64", "int1"}


def nonvalue(e):
    if isinstance(e, ast.Constant) and (e.value is None or isinstance(e.value, str)):
        return True
    if isinstance(e, ast.Attribute) and e.attr in DTYPES:
        return True
    if isinstance(e, ast.IfExp):
        return nonvalue(e.body) and nonvalue(e.orelse)
    return False


def names_read(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def names_stored(nodes):
    out = set()
    for s in nodes:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
    return out


for t in inv["tutorials"]:
    src_name = t["source"]
    if src_name.startswith("11-"):
        continue
    tree = ast.parse(open(os.path.join(TUT, src_name)).read())
    funcs = {f["name"]: f for f in t["functions"]}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in funcs or (src_name, fn.name) in EXCLUDED:
            continue
        if not (funcs[fn.name]["line"] <= fn.lineno <= funcs[fn.name]["end_line"]):
            continue
        where = f"{src_name}:{fn.name}"
        params = {a.arg for a in fn.args.args}
        # Walk statement lists to find statements following each loop at the same or an outer level.
        def walk_block(stmts, before_names, after_stmts_outer):
            for i, s in enumerate(stmts):
                rest = stmts[i + 1:] + after_stmts_outer
                if isinstance(s, (ast.For, ast.While)):
                    later_reads = set().union(*[names_read(r) for r in rest]) if rest else set()
                    if isinstance(s, ast.For) and isinstance(s.target, ast.Name) and s.target.id in later_reads:
                        # read after the loop before any re-assignment? report and let a human check
                        print(f"A {where}:{s.lineno} loop variable {s.target.id} read later (check reassignment)")
                    body_new = names_stored(s.body) - before_names - params
                    if isinstance(s, ast.For):
                        body_new -= {s.target.id}
                    for nm in sorted(body_new & later_reads):
                        print(f"B {where}:{s.lineno} name {nm} first defined in the loop body, read later (check)")
                if isinstance(s, ast.If):
                    for branch in (s.body, s.orelse):
                        for b in branch:
                            for n in ast.walk(b):
                                if isinstance(n, ast.Assign):
                                    for tg in n.targets:
                                        if isinstance(tg, ast.Name) and tg.id in nonvalue_names:
                                            print(f"C {where}:{n.lineno} branch of if at {s.lineno} assigns {tg.id}, "
                                                  f"which held a non-tensor value earlier")
                if isinstance(s, ast.Assign) and nonvalue(s.value):
                    for tg in s.targets:
                        if isinstance(tg, ast.Name):
                            nonvalue_names.add(tg.id)
                            print(f"C {where}:{s.lineno} {ast.unparse(s)[:90]}")
                if isinstance(s, ast.Expr) and not isinstance(s.value, ast.Call):
                    print(f"D {where}:{s.lineno} expression statement {ast.unparse(s)[:80]}")
                before_names = before_names | names_stored([s])
                for field in ("body", "orelse"):
                    sub = getattr(s, field, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk_block(sub, before_names, rest)
        nonvalue_names = set()
        walk_block(fn.body, set(), [])
print("done")
