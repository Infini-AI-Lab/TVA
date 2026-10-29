"""lr-20e: is a `bfloat16` constant the host's C locale's?

SEMANTICS ("Constants") and `FloatV.litBF16` say a Python number built in `bfloat16` is the `bfloat16` nearest
to the six-decimal text `std::to_string` prints for its `float32` (`APFloat(BFloat, std::to_string(v))`,
`primitives/triton-3.8.0-ir.cc`). `std::to_string` is `%f`, which follows the process's `LC_NUMERIC`. This
script compiles `tl.full([1], c, tl.bfloat16)` and `x * c` (a `bfloat16` `x = 1`) in two fresh processes, each
with an empty Triton cache: one with the default locale, one after `locale.setlocale(LC_NUMERIC, ...)` to a
locale whose decimal point is `,` (`en_DK.utf8`, present on this machine), and prints the stored bits.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I bf16_locale.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

VALUES = ["0.1", "1.5", "2.75", "-3.25", "100.5", "0.001"]


def child(loc):
    import locale
    if loc:
        locale.setlocale(locale.LC_NUMERIC, loc)
    import importlib.util
    import torch
    import triton  # noqa: F401
    tmp = tempfile.mkdtemp(prefix="lr20e-loc-")
    body = ["i = tl.arange(0, 1)", "x = tl.load(x_ptr + i)"]
    for k, v in enumerate(VALUES):
        body.append(f"tl.store(o_ptr + {2 * k} + i, tl.full([1], {v}, tl.bfloat16))")
        body.append(f"tl.store(o_ptr + {2 * k + 1} + i, x * {v})")
    src = "\n".join(["import triton", "import triton.language as tl", "", "@triton.jit",
                     "def kloc(x_ptr, o_ptr):"] + ["    " + b for b in body]) + "\n"
    path = os.path.join(tmp, "kloc.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("kloc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.ones(1, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(2 * len(VALUES), dtype=torch.bfloat16, device="cuda")
    mod.kloc[(1,)](x, o)
    torch.cuda.synchronize()
    print("RESULT " + json.dumps({"decimal_point": locale.localeconv()["decimal_point"],
                                  "bits": [b & 0xFFFF for b in o.view(torch.int16).tolist()]}))


def run(loc):
    env = dict(os.environ)
    env["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr20e-loc-cache-")
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", loc], capture_output=True,
                       text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    err = r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "no output"
    # keep no absolute path in the output (LLVM's abort message names its source file)
    return {"error": re.sub(r"/\S*/llvm(?=/lib/)", "<llvm>", err)}


def main():
    import torch
    import triton
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    a = run("")
    b = run("en_DK.utf8")
    print(f"default locale: decimal point {a.get('decimal_point')!r}; LC_NUMERIC=en_DK.utf8: decimal point "
          f"{b.get('decimal_point')!r}")
    if "error" in a or "error" in b:
        print("error:", a.get("error"), b.get("error"))
        return
    print("| c | tl.full, default | tl.full, en_DK | x*c, default | x*c, en_DK |")
    print("|---|---|---|---|---|")
    differ = 0
    for k, v in enumerate(VALUES):
        fa, ma, fb, mb = a["bits"][2 * k], a["bits"][2 * k + 1], b["bits"][2 * k], b["bits"][2 * k + 1]
        differ += (fa != fb) + (ma != mb)
        print(f"| {v} | `{fa:04x}` | `{fb:04x}` | `{ma:04x}` | `{mb:04x}` |")
    print(f"stored values that differ between the two locales: {differ}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
