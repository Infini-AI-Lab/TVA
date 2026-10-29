"""The promotion rules of README.md ("Conclusions"), written out independently of Triton, and checked
line by line against `table.tsv` (the front end's own answers, from `table.py`).

It prints the number of lines compared and every line where the rules and the front end disagree
in the result column (`t:<dtype>`, `s:<dtype>`, `py:<value>` or `error`) or in the error's class.
Python-only cases (both operands Python values) are Python's own arithmetic; the rules only say
"Python's operator" and take the result from Python.

Run: .venv/bin/python -I evidence/promotion/rules.py  (from TVA's root)
"""
import math
import os
import sys
import warnings

warnings.simplefilter("ignore", DeprecationWarning)   # Python's own `~True`

HERE = os.path.dirname(os.path.abspath(__file__))

SINT = ["int8", "int16", "int32", "int64"]
UINT = ["int1", "uint8", "uint16", "uint32", "uint64"]          # int1 counts as unsigned
FP8 = ["fp8e5", "fp8e4nv"]
FLOAT = FP8 + ["fp16", "bf16", "fp32", "fp64"]
WIDTH = {"int1": 1, "int8": 8, "int16": 16, "int32": 32, "int64": 64, "uint8": 8, "uint16": 16,
         "uint32": 32, "uint64": 64}
BITS = dict(WIDTH, fp8e5=8, fp8e4nv=8, fp16=16, bf16=16, fp32=32, fp64=64)
MANT = {"fp8e5": 2, "fp8e4nv": 3, "fp16": 10, "bf16": 7, "fp32": 23, "fp64": 52}


class Reject(Exception):
    pass


def is_int(d):
    return d in SINT or d in UINT


def kind(d):                     # BOOLEAN < INTEGRAL < FLOATING (core.py dtype.kind)
    return 0 if d == "int1" else 1 if is_int(d) else 2


def int_range(d):
    w = WIDTH[d]
    return (-(1 << (w - 1)), (1 << (w - 1)) - 1) if d in SINT else (0, (1 << w) - 1)


def py_type(v):                  # semantic.to_tensor_type
    if isinstance(v, bool):
        return "int1"
    if isinstance(v, int):
        if -2**31 <= v < 2**31:
            return "int32"
        if 2**31 <= v < 2**32:
            return "uint32"
        if -2**63 <= v < 2**63:
            return "int64"
        if 2**63 <= v < 2**64:
            return "uint64"
        raise Reject("ValueError")
    a = abs(v)
    if a == math.inf or a == 0.0 or v != v or 2**-126 <= a <= (2 - 2**-23) * 2**127:
        return "fp32"
    return "fp64"


def int_promote(a, b):           # semantic.integer_promote_impl
    sa, sb = a in SINT, b in SINT
    if sa == sb:
        return a if WIDTH[a] > WIDTH[b] else b
    if not sa:                   # a unsigned
        return a if WIDTH[a] >= WIDTH[b] else b
    return b if WIDTH[b] >= WIDTH[a] else a


def computation_type(a, a_py, b, b_py, dm):   # semantic.computation_type_impl
    if a_py != b_py:
        s, t = (a, b) if a_py else (b, a)
        if kind(s) <= kind(t):
            return "fp32" if dm and t in ("fp16", "bf16") else t
    if "fp64" in (a, b):
        return "fp64"
    if "fp32" in (a, b):
        return "fp32"
    if "fp16" in (a, b):
        return "fp32" if dm else "fp16"
    if a == "bf16" and b == "bf16":
        return "fp32" if dm else "bf16"
    if "bf16" in (a, b):
        return "fp32"
    if a in FP8 and b in FP8:
        return a if a == b else "fp16"
    if not is_int(a) or not is_int(b):
        raise Reject("TypeError")
    if dm and (a in SINT) != (b in SINT):
        raise Reject("TypeError")
    return int_promote(a, b)


class Op:
    """An operand: a tensor (dtype, block?) or a Python value."""

    def __init__(self, dtype=None, block=False, py=None, is_py=False):
        self.dtype, self.block, self.py, self.is_py = dtype, block, py, is_py


def parse(s):
    if s.startswith("t:"):
        return Op(s[2:], True)
    if s.startswith("s:"):
        return Op(s[2:], False)
    if s.startswith("int:"):
        return Op(py=int(s[4:]), is_py=True)
    if s == "float":
        return Op(py=1.5, is_py=True)
    if s == "bool":
        return Op(py=True, is_py=True)
    raise ValueError(s)


def to_tensor(x):                # a Python value becomes a scalar tensor of its own type
    return x if not x.is_py else Op(py_type(x.py), False)


def common(x, y, dm=False):      # semantic.binary_op_type_checking_impl, then broadcasting
    xt = py_type(x.py) if x.is_py else x.dtype
    yt = py_type(y.py) if y.is_py else y.dtype
    r = computation_type(xt, x.is_py, yt, y.is_py, dm)
    for o in (x, y):
        if o.is_py and o.py < 0 and r in UINT:
            raise Reject("ValueError")
        if o.is_py and is_int(r):
            lo, hi = int_range(r)
            if not lo <= o.py <= hi:
                raise Reject("ValueError")
    return r, (x.block or y.block)


def res(d, block):
    return ("t:" if block else "s:") + d


PYOPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
         "/": lambda a, b: a / b, "//": lambda a, b: a // b, "%": lambda a, b: a % b,
         "&": lambda a, b: a & b, "|": lambda a, b: a | b, "^": lambda a, b: a ^ b,
         "<<": lambda a, b: _huge(a, b) if isinstance(a, int) and isinstance(b, int) and b >= 2**62 else a << b,
         ">>": lambda a, b: a >> b,
         "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
         ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
         "min": min, "max": max}


def _huge(a, b):
    if a == 0:
        return 0
    raise MemoryError()


def spell_py(v):
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 2**128:
        return "py:big"
    return f"py:{v!r}"


def binary(op, x, y):
    if x.is_py and y.is_py and op in PYOPS:
        try:
            return spell_py(PYOPS[op](x.py, y.py))
        except ValueError:
            raise Reject("ValueError")
        except ZeroDivisionError:
            raise Reject("ZeroDivisionError")
        except TypeError:
            raise Reject("TypeError")
        except MemoryError:
            raise Reject("MemoryError")
    if op in ("+", "-", "*"):
        return res(*common(x, y))
    if op == "/":
        r, blk = common(x, y, dm=True)
        return res(r if r in FLOAT else "fp32", blk)
    if op == "//":
        r, blk = common(x, y, dm=True)
        if not is_int(r):
            raise Reject("TypeError")
        return res(r, blk)
    if op == "%":
        return res(*common(x, y, dm=True))
    if op in ("&", "|", "^", "<<", ">>"):
        r, blk = common(x, y)
        if not is_int(r):
            raise Reject("IncompatibleTypeErrorImpl")
        return res(r, blk)
    if op in ("<", "<=", ">", ">=", "==", "!="):
        r, blk = common(to_tensor(x), to_tensor(y))
        return res("int1", blk)
    if op in ("tl.minimum", "tl.maximum", "min", "max"):
        x, y = to_tensor(x), to_tensor(y)
        x = Op("fp32" if x.dtype == "bf16" else x.dtype, x.block)
        y = Op("fp32" if y.dtype == "bf16" else y.dtype, y.block)
        return res(*common(x, y))
    if op in ("and", "or"):
        vals = []
        last = None
        for v in (x, y):
            last = v
            if v.is_py:
                if (not v.py) and op == "and" or bool(v.py) and op == "or":
                    return spell_py(v.py)
            else:
                vals.append(v)
        if not vals:
            return spell_py(last.py)
        if len(vals) == 1:
            return res(vals[0].dtype, vals[0].block)
        for v in vals:
            if v.dtype != "int1" and BITS[v.dtype] != 1:
                raise Reject("ValueError")      # bitcast to int1 of a wider type
        return res("int1", vals[0].block or vals[1].block)
    raise ValueError(op)


def unary(op, x):
    if x.is_py:
        v = x.py
        try:
            return spell_py({"neg": lambda: -v, "invert": lambda: ~v, "not": lambda: not v,
                             "pos": lambda: +v}[op]())
        except TypeError:
            raise Reject("TypeError")
    if op == "neg":
        return res(x.dtype, x.block)
    if op == "invert":
        if x.dtype in FLOAT:
            raise Reject("ValueError")
        return res(x.dtype, x.block)
    if op == "not":
        if x.dtype != "int1":
            raise Reject("ValueError")
        return res("int1", x.block)
    if op == "pos":
        raise Reject("AttributeError")          # a tensor has no __pos__
    raise ValueError(op)


def where(c, x, y):
    c = to_tensor(c)
    if c.dtype in FP8:
        raise Reject("AssertionError")          # no cast from fp8 to int1
    r, blk = common(x, y)
    return res(r, blk or c.block)


def predict(op, a, b):
    try:
        if op.startswith("where:"):
            return where(parse(op[6:]), parse(a), parse(b))
        if b == "-":
            return unary(op, parse(a))
        return binary(op, parse(a), parse(b))
    except Reject as e:
        return "error\t" + str(e)


def main():
    n = bad = 0
    with open(os.path.join(HERE, "table.tsv")) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            got = cols[3] if cols[3] != "error" else "error\t" + cols[4].split(":")[0]
            want = predict(*cols[:3])
            n += 1
            if got != want:
                bad += 1
                if bad <= 40:
                    print("DIFFER", "\t".join(cols[:3]), "| front end:", got.replace("\t", " "),
                          "| rules:", want.replace("\t", " "))
    print(f"{n} lines compared, {bad} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
