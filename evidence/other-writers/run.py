"""Writes by others to bytes next to a launch's bytes, while the launch runs.

Library evidence for library issue L49 (round 8). A claimed launch repeatedly read-modify-writes
its own bytes of a buffer (`y[2i] += 1`, R times); at the same time another kernel, on another
stream, repeatedly writes the bytes in between (`y[2i + 1]`), which share every 32-byte sector and
128-byte line with the claimed ones. Element (int32) and byte (uint8) interleavings. Both kernels
record `%globaltimer` at their start and end, so the overlap is measured, not assumed.

  W1  int32 interleave: the claimed launch alone; with the neighbour writer; 20 trials each;
  W2  uint8 interleave (neighbours inside the same 4-byte word): the same;
  W3  control: the other writer writes the claimed launch's own bytes (`y[2i]`).

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/other-writers/run.py  (from TVA's root)
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl

PROGRAMS, BLOCK = 16, 256
NCLAIM = PROGRAMS * BLOCK          # claimed elements, at the even positions of y
R_CLAIM, R_OTHER = 4000, 120000


@triton.jit
def now():
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def claimed(y_ptr, t_ptr, R, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t0 = now()
    offs = 2 * (pid * BLOCK + tl.arange(0, BLOCK))     # this launch's bytes: the even elements
    for r in range(R):
        v = tl.load(y_ptr + offs, volatile=True)
        tl.store(y_ptr + offs, v + 1)
    t1 = now()
    if pid == 0:
        tl.store(t_ptr + 0, t0)
        tl.store(t_ptr + 1, t1)


@triton.jit
def other(y_ptr, t_ptr, R, PARITY: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    t0 = now()
    offs = 2 * (pid * BLOCK + tl.arange(0, BLOCK)) + PARITY   # odd: the neighbours; even: the same bytes
    for r in range(R):
        tl.store(y_ptr + offs, (r * 7 + offs).to(y_ptr.dtype.element_ty))
    t1 = now()
    if pid == 0:
        tl.store(t_ptr + 0, t0)
        tl.store(t_ptr + 1, t1)


def trial(dtype, with_other, parity):
    y = torch.zeros(2 * NCLAIM, dtype=dtype, device="cuda")
    tc = torch.zeros(2, dtype=torch.int64, device="cuda")
    to = torch.zeros(2, dtype=torch.int64, device="cuda")
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(sa):
        claimed[(PROGRAMS,)](y, tc, R_CLAIM, BLOCK=BLOCK)
    if with_other:
        with torch.cuda.stream(sb):
            other[(PROGRAMS,)](y, to, R_OTHER, PARITY=parity, BLOCK=BLOCK)
    torch.cuda.synchronize()
    expect = R_CLAIM % 256 if dtype == torch.uint8 else R_CLAIM
    mine = y[0::2]
    ok = bool((mine == expect).all())
    a0, a1 = tc.tolist()
    b0, b1 = to.tolist()
    overlap = with_other and b0 < a1 and a0 < b1
    cover = max(0, min(a1, b1) - max(a0, b0)) / (a1 - a0) if with_other else 0.0
    return ok, int((mine != expect).sum()), overlap, (a1 - a0) / 1e6, ((b1 - b0) / 1e6 if with_other else 0.0), cover


def case(dtype, parity, trials=20):
    lone = [trial(dtype, False, parity) for _ in range(trials)]
    both = [trial(dtype, True, parity) for _ in range(trials)]
    return {
        "alone: trials with every claimed element correct": f"{sum(t[0] for t in lone)} of {trials}",
        "with the other writer: trials with every claimed element correct": f"{sum(t[0] for t in both)} of {trials}",
        "with the other writer: claimed elements wrong (max over trials)": max(t[1] for t in both),
        "trials in which the two launches overlapped in time": f"{sum(t[2] for t in both)} of {trials}",
        "claimed launch duration ms (median)": sorted(t[3] for t in both)[trials // 2],
        "other launch duration ms (median)": sorted(t[4] for t in both)[trials // 2],
        "fraction of the claimed launch's time (program 0) during the other's (median)":
            round(sorted(t[5] for t in both)[trials // 2], 3),
    }


CASES = {
    "W1": ("int32 elements; the other writer writes the odd elements (same sectors and lines)",
           lambda: case(torch.int32, 1)),
    "W2": ("uint8 elements; the other writer writes the odd bytes (same 4-byte words)",
           lambda: case(torch.uint8, 1)),
    "W3": ("control, int32: the other writer writes the claimed launch's own (even) elements",
           lambda: case(torch.int32, 0)),
}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = CASES[sys.argv[2]][1]()
        except Exception as e:
            res = {"error": f"{type(e).__name__}: " + str(e).splitlines()[0][:200]}
        print("RESULT " + json.dumps(res, default=str))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print(f"claimed launch: {PROGRAMS} programs x {BLOCK} elements, y[2i] += 1 repeated {R_CLAIM} times; "
          f"other launch: {PROGRAMS} programs, {R_OTHER} stores to each of its elements")
    for name, (desc, _) in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        print(f"\n{name}: {desc}")
        if line is None:
            print("  process failed: " + ((p.stderr.strip().splitlines() or ["(no output)"])[-1])[:200])
            continue
        for k, v in json.loads(line[len("RESULT "):]).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
