# Library batch, round 27: author's notes

Issue: L135 ([ISSUES.md](../../ISSUES.md)). The user's goal (2026-11-02): "we should commit and push
the bench. it should be a part of tva. actually I do not think it is a bench." The bench, built that
day outside TVA, moves into TVA as `sandbox/` (the user chose the name), and the user chose to exempt
the folder in `tvr` rather than keep it outside. No library review follows unless the user asks for
one. Checked by `tvr audit`, `tvr check` and `tvr status` on every kernel, and `tvr lint library 27
author`. No Lean file changed, so `lake build TVA` and `tvr library-check` were not re-run.

## L135: the sandbox's files skip the text scan's generic path tests

- **What changed.**
  - `tools/tvr.py`: a constant `SANDBOX` (the folder `sandbox/` under TVA's root); `foreign_refs`
    takes `home` (true by default), and `home=False` skips the home-path test; `scan_files` calls it
    with `absolute=False`, `relative=False` and `home=False` for a file under `sandbox/`. The name
    test and the `fpv/<name>` test apply to those files as to every other file, and a symbolic link
    there must still stay inside TVA.
  - `docs/REVIEW_PROTOCOL.md` §2 (the text scan) says which tests the sandbox's files skip, and why.
  - New folder `sandbox/` (its `README.md`, `PROTOCOL.md` and `HARNESS.md` say what it holds);
    TVA's `README.md` points to it.
- **Why.** The sandbox's files run in containers and name those containers' paths: the test
  harness's folders, the review service's folder, the agents' homes and scratch folders, the
  toolchain's folder. The scan read 60 of them, in 14 of the folder's 26 files, as paths out of TVA,
  so with the folder in the tree every kernel's gate failed its independence check. Writing those
  paths so that the lexical scan misses them (computed at run time) would hide them rather than check
  them. The exemption says what the folder is, as Triton's pinned files are exempt from the generic
  tests, and keeps the tests that find another project: its name and a path into a sibling of TVA.
- **Evidence.** `tvr audit` prints `independent` with `sandbox/` in the tree. A planted path into a
  sibling of TVA (`fpv/<name>/…`) in a file under `sandbox/` is still found, and a home path in a file of
  `docs/` still is (both removed after the probe). `tvr check` re-ran on each of the twelve kernels
  (`attn_bwd_f16`, `attn_bwd_pre_f16`, `attn_fwd_e5m2`, `attn_fwd_f16`, `block_scaled_matmul_mxfp8`,
  `grouped_matmul_tma_f16`, `matmul_e5m2`, `matmul_f16`, `paged_attention_f16`, `softmax_f32`,
  `streamk_matmul_f16`, `vec_add_f32`): each passed with the digest it had, and `tvr status` grants
  each certificate again. The kernel files that changed are the gates' records (`gate.json`: its time
  and the sha256 of `tools/tvr.py`; `claim.md` and `REVIEW.md`: the gate's time;
  `review/certificate.json`: its time).
- **Constants whose meaning changed.** None: no Lean file changed. The gate's code changed, so every
  kernel's gate was stale until re-run; item digests do not cover `tools/tvr.py`, so no acceptance
  changed.
