# tva-sandbox-base: the TVA library (Lean 4 + Mathlib, built) and its protocol tool `tvr`.
#
# Built by build_base.sh from the files TVA ships (no git history, so no proof the history
# holds). Every image of a task (prover, reviewer, verifier) starts from this one.
#
# TVA lives at /home/tva/fpv/tva, owned by the user `tva`: its prompts and documents name it
# `~/fpv/tva`, and its text scan refuses other absolute paths in a record. A role that runs as another
# user reaches it as its own `~/fpv/tva` (a link to this tree, or a copy of it whose `lean/.lake` links
# to this tree's), and `~/.elan` (a link to /opt/elan).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    ELAN_HOME=/opt/elan \
    PATH=/opt/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TVA_ROOT=/home/tva/fpv/tva \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git python3 python3-pip procps xz-utils zstd tini sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir numpy \
    && git config --system --add safe.directory '*'
# (Lake runs git in Mathlib's package folders; a user other than their owner must be allowed to.)

# elan and the Lean toolchain lean/lean-toolchain pins, readable by every user.
COPY tva/lean/lean-toolchain /tmp/lean-toolchain
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o /tmp/elan-init.sh \
    && sh /tmp/elan-init.sh -y --default-toolchain none --no-modify-path \
    && elan toolchain install "$(cat /tmp/lean-toolchain)" \
    && elan default "$(cat /tmp/lean-toolchain)" \
    && chmod -R a+rX /opt/elan && rm /tmp/elan-init.sh /tmp/lean-toolchain

RUN useradd -m -u 1001 -s /bin/bash tva && ln -s /opt/elan /home/tva/.elan
COPY --chown=tva:tva tva /home/tva/fpv/tva

# Mathlib's prebuilt cache for the pinned commit, then the TVA library itself, built as `tva` so that
# no layer copies the build to change its owner. (The cache unpacks thousands of files at once: the
# default soft limit of 1024 open files is too low.)
USER tva
WORKDIR /home/tva/fpv/tva/lean
RUN ulimit -n 65536 && lake exe cache get && lake build TVA && rm -rf /home/tva/.cache

# Smoke test: the protocol tool runs here, and a shipped certificate still holds against this tree (the
# library files and the packages' versions its gate recorded). It builds no kernel: a task's prover must
# find no kernel's build output in the image.
WORKDIR /home/tva/fpv/tva
RUN python3 -I tools/tvr.py status vec_add_f32 | grep -q "Certificate: GRANTED" \
    && test ! -e lean/.lake/build/lib/lean/VecAdd.olean
USER root
