"""What an autotuned call runs on the caller's buffers, around the launch a claim is about.

Library evidence for library issue L21 (round 8). The kernel adds 1 in place to an int32 buffer,
so after a call each element holds the number of times the kernel ran on it. Each case runs in its
own process (the autotuner's cache is per process, the disk cache per cache folder):

  A1  three configs, no hooks: the first call; then the same call again;
  A2  `restore_value=['x_ptr']`; `reset_to_zero=['x_ptr']` (first and second call);
  A3  a user `pre_hook`/`post_hook` on the autotuner and a `pre_hook` on a config: which run when,
      and that they write the arguments;
  A4  a second call with the same key; a call with a new key;
  A5  one config; `prune_configs_by` leaving one config and leaving two;
  A6  `TRITON_CACHE_AUTOTUNING=1`: a second process with the same cache folder.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/autotune-launch/run.py  (from TVA's root)
"""
import json, os, subprocess, sys, tempfile

import torch
import triton
import triton.language as tl

N = 1 << 16
EVENTS = []   # host hooks record here
CFGS = [triton.Config({"BLOCK": 128}), triton.Config({"BLOCK": 256}), triton.Config({"BLOCK": 512})]


def body():
    @triton.jit
    def inc(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1, mask=m)
    return inc


def grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def counts(x):
    v = x.unique().tolist()
    return v[0] if len(v) == 1 else v


def call(k, x):
    k[grid](x, x.numel())
    torch.cuda.synchronize()
    return counts(x)


def case_A1():
    k = triton.autotune(configs=CFGS, key=["n"])(body())
    x = torch.zeros(N, dtype=torch.int32, device="cuda")
    first = call(k, x)
    x.zero_()
    second = call(k, x)
    return {"runs on x at the first call": first, "runs on x at the second call": second,
            "best config": str(k.best_config)}


def case_A2():
    k = triton.autotune(configs=CFGS, key=["n"], restore_value=["x_ptr"])(body())
    x = torch.full((N,), 7, dtype=torch.int32, device="cuda")
    r1 = call(k, x)
    k2 = triton.autotune(configs=CFGS, key=["n"], reset_to_zero=["x_ptr"])(body())
    y = torch.full((N,), 7, dtype=torch.int32, device="cuda")
    z1 = call(k2, y)
    y.fill_(7)
    z2 = call(k2, y)
    return {"restore_value: x = 7 before the first call, after": r1,
            "reset_to_zero: x = 7 before the first call, after": z1,
            "reset_to_zero: x = 7 before the second call, after": z2}


def case_A3():
    def user_pre(nargs, reset_only=False):
        EVENTS.append("autotuner pre_hook" + (" (reset_only)" if reset_only else ""))
        nargs["x_ptr"].fill_(100)

    def user_post(nargs, exception):
        EVENTS.append("autotuner post_hook")
        nargs["x_ptr"].fill_(-5)

    def cfg_pre(nargs):
        EVENTS.append("config pre_hook BLOCK=%d" % nargs["BLOCK"])
        nargs["x_ptr"].fill_(1000)

    cfgs = [triton.Config({"BLOCK": 128}, pre_hook=cfg_pre), triton.Config({"BLOCK": 256}, pre_hook=cfg_pre)]
    k = triton.autotune(configs=cfgs, key=["n"], pre_hook=user_pre, post_hook=user_post)(body())
    x = torch.zeros(N, dtype=torch.int32, device="cuda")
    EVENTS.clear()
    v1 = call(k, x)
    ev1 = list(EVENTS)
    EVENTS.clear()
    x.zero_()
    v2 = call(k, x)
    ev2 = list(EVENTS)

    def summary(ev):
        out = {}
        for e in ev:
            out[e] = out.get(e, 0) + 1
        return {"hook calls": out, "last three": ev[-3:]}

    return {"first call: x after (x = 0 before)": v1, "first call": summary(ev1),
            "second call: x after (x = 0 before)": v2, "second call": summary(ev2)}


def case_A4():
    k = triton.autotune(configs=CFGS, key=["n"])(body())
    x = torch.zeros(N, dtype=torch.int32, device="cuda")
    a = call(k, x)
    x.zero_()
    b = call(k, x)
    x2 = torch.zeros(N // 2, dtype=torch.int32, device="cuda")
    c = call(k, x2)
    x3 = torch.zeros(N, dtype=torch.int32, device="cuda")[1:]  # same key n? no: n = N - 1
    d = call(k, x3)
    return {"first call, key n=%d" % N: a, "second call, same key": b,
            "call with n=%d (new key)" % (N // 2): c, "call with n=%d (new key)" % (N - 1): d,
            "cache keys": [str(key) for key in k.cache]}


def case_A5():
    k1 = triton.autotune(configs=CFGS[:1], key=["n"])(body())
    x = torch.zeros(N, dtype=torch.int32, device="cuda")
    one = call(k1, x)
    k2 = triton.autotune(configs=CFGS, key=["n"],
                         prune_configs_by={"early_config_prune": lambda cfgs, nargs, **kw: cfgs[:1]})(body())
    x.zero_()
    pruned1 = call(k2, x)
    k3 = triton.autotune(configs=CFGS, key=["n"],
                         prune_configs_by={"early_config_prune": lambda cfgs, nargs, **kw: cfgs[:2]})(body())
    x.zero_()
    pruned2 = call(k3, x)
    return {"one config": one, "pruned to one config": pruned1, "pruned to two configs": pruned2}


def case_A6():
    k = triton.autotune(configs=CFGS, key=["n"])(body())
    x = torch.zeros(N, dtype=torch.int32, device="cuda")
    return {"runs on x at the first call in this process": call(k, x)}


CASES = {
    "A1": "three configs, no hooks",
    "A2": "restore_value and reset_to_zero",
    "A3": "user pre_hook/post_hook and a config pre_hook, all writing x",
    "A4": "same key, new key",
    "A5": "one config; pruned to one; pruned to two",
    "A6-first": "TRITON_CACHE_AUTOTUNING=1, empty cache folder",
    "A6-second": "TRITON_CACHE_AUTOTUNING=1, the same cache folder, a new process",
}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        name = sys.argv[2]
        fn = {"A6-first": case_A6, "A6-second": case_A6}.get(name) or globals()["case_" + name]
        try:
            res = fn()
        except Exception as e:
            res = {"error": f"{type(e).__name__}: " + str(e).splitlines()[0][:200]}
        print("RESULT " + json.dumps(res, default=str))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print(f"x: int32 buffer of {N} elements; each run of the kernel adds 1 to every element")
    a6dir = tempfile.mkdtemp(prefix="tva-a6-")   # an empty cache folder shared by A6's two processes
    for name, desc in CASES.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        if name.startswith("A6"):
            env["TRITON_CACHE_AUTOTUNING"] = "1"
            env["TRITON_CACHE_DIR"] = a6dir
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        print(f"\n{name}: {desc}")
        if line is None:
            print("  process failed: " + ((p.stderr.strip().splitlines() or ["(no output)"])[-1])[:200])
            continue
        for k, v in json.loads(line[len("RESULT "):]).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
