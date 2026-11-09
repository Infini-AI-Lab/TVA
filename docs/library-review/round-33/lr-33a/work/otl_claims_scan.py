"""lr-33a, round 33, L149: list every float reduction (tl.sum/max/min/reduce, x.sum(...) forms) that sits inside a
runtime `for` loop (range / tl.range) of a claim's source, wherever nested (if, static_range, helpers not followed),
so the broadened pattern can be read by eye. Static scan; reads claims/*/source only.
Run from the TVA root: python3 -I <this file>"""
import ast
import glob

RED = {"sum", "max", "min", "reduce"}


def is_runtime_for(n):
    it = n.iter
    f = it.func if isinstance(it, ast.Call) else None
    name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
    return name in ("range",) and not (isinstance(f, ast.Attribute) and f.attr == "static_range")


class V(ast.NodeVisitor):
    def __init__(self, path):
        self.path, self.depth, self.hits = path, 0, []

    def visit_For(self, n):
        rt = is_runtime_for(n)
        self.depth += rt
        self.generic_visit(n)
        self.depth -= rt

    def visit_Call(self, n):
        f = n.func
        if self.depth and isinstance(f, ast.Attribute) and f.attr in RED:
            self.hits.append((n.lineno, ast.unparse(n)[:110]))
        self.generic_visit(n)


for p in sorted(glob.glob("claims/*/source/*.py")):
    try:
        t = ast.parse(open(p).read())
    except SyntaxError:
        continue
    v = V(p)
    v.visit(t)
    for ln, s in v.hits:
        print(f"{p}:{ln}: {s}")
