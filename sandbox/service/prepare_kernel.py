#!/usr/bin/env python3
"""Reset a kernel of the TVA tree to the state a prover starts from (run once, at image build).

  prepare_kernel.py TASK_JSON

The kernel's folder keeps its source (and SETUP.md, the coordinator's notes on the launch, when it has
one); everything the prover and the reviewers wrote goes: the Lean, the write-up, the model map, the
line classes of kernel.json, the review record, and the generated pages. kernel.json keeps what the task
fixes (the protocol, the Lean module and namespace, the source file and its sha256) and its title (which
names the launched kernel and the dtype). The tree's other
kernels stay as they are: they are the library's worked examples.
"""
import json
import os
import shutil
import sys

ROOT = os.environ.get("TVA_ROOT", "/home/tva/fpv/tva")


def main(task_path):
    task = json.load(open(task_path))
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import tvr  # the tree's own tool: its README template and its page renderers

    k = task["kernel"]
    kd = os.path.join(ROOT, "kernels", k)
    man = json.load(open(os.path.join(kd, "kernel.json")))
    src = man["source"]
    if (man["lean"]["module"], man["lean"]["namespace"], src["file"], src["sha256"]) != \
            (task["module"], task["namespace"], task["source"], task["source_sha256"]):
        sys.exit(f"kernels/{k}/kernel.json does not match the task")
    if tvr.file_sha(os.path.join(kd, src["file"])) != src["sha256"]:
        sys.exit(f"kernels/{k}/{src['file']} does not match its sha256")
    keep = {src["file"].split("/")[0], "SETUP.md"}
    for name in os.listdir(kd):
        if name not in keep:
            p = os.path.join(kd, name)
            shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    # the title stays: it names the launched kernel and the dtype, which is the task's statement
    tvr.dump_json(os.path.join(kd, "kernel.json"), {
        "protocol": man["protocol"], "title": man.get("title", k), "lean": man["lean"],
        "source": {"file": src["file"], "sha256": src["sha256"], "ranges": [], "host": []},
        "doc": "README.md"})
    tvr.dump_json(os.path.join(kd, "model-map.json"), {"entries": []})
    tvr.write(os.path.join(kd, "README.md"), tvr.README_TEMPLATE.format(title=k, src=src["file"]))
    for d in ("lean", "review"):
        os.makedirs(os.path.join(kd, d))
    # the kernel's own page and the index, as the tree would show a kernel just created
    tvr.render_review(k)
    tvr.render_index()
    tvr.render_reports()
    if os.geteuid() == 0:  # what this wrote belongs to the tree's owner, as the rest of the tree does
        st = os.stat(ROOT)
        pages = [os.path.join(ROOT, "kernels", "README.md"), os.path.join(ROOT, "docs", "library-review", "REPORTS.md")]
        for dirpath, dirs, files in os.walk(kd):
            for n in [dirpath] + [os.path.join(dirpath, f) for f in files]:
                os.chown(n, st.st_uid, st.st_gid)
        for p in pages:
            os.chown(p, st.st_uid, st.st_gid)
    print(f"kernels/{k} reset to its skeleton: " + ", ".join(sorted(os.listdir(kd))))


if __name__ == "__main__":
    main(sys.argv[1])
