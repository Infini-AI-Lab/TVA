# The claim base image tva-claim-base:<commit> (build_base.sh), shared by every claim task. Its /app:
#   /app/claim/                the task's claim (each task's image adds it) and the prover's proof
#   /app/examples/vec_add_f32/ a fixed claim with its complete proof (Proof.lean), checked at build
#   /app/lean/                 the TVA library (TVA/) with Mathlib, built; lakefile: TVA and the claim (ClaimTask)
#   /app/docs/                 TVA's semantics and proof guide
# /app/CLAUDE.md (how to work here) and /app/.claude/ (the prover's Claude Code settings: hooks, and its subagent) are
# the prover's harness: the prover starts in /app, so Claude Code loads both as the project's, for the prover and its
# subagents (Harbor points CLAUDE_CONFIG_DIR elsewhere, so ~/.claude is not
# read). /process_safety.md (what never to kill) is the prover's too, and each task adds /instruction.md. One tool, /usr/local/bin/tva-claim (check, verify, prepare, hooks). The verifier is the same image.
ARG BASE
FROM ${BASE} AS tva
USER root
COPY lakefile.toml /stage/lean/lakefile.toml
RUN cd /home/tva/fpv/tva \
    && cp -a lean/TVA lean/TVA.lean lean/lake-manifest.json lean/lean-toolchain lean/.lake /stage/lean/ \
    && rm -rf /stage/lean/.lake/build/lib/lean/VecAdd* \
    && mkdir -p /stage/docs && cp -a docs/SEMANTICS.md docs/PROVER_GUIDE.md docs/LIBRARY.md docs/TRANSCRIPTION.md \
       docs/rules /stage/docs/ \
    && chown -R root:root /stage && chmod -R a+rX,go-w /stage /opt/elan

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 ELAN_HOME=/opt/elan \
    PATH=/opt/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git python3 procps tini \
    && rm -rf /var/lib/apt/lists/* && git config --system --add safe.directory '*'
COPY --from=tva --chown=root:root /opt/elan /opt/elan
COPY --from=tva /stage /app
COPY --from=tva --chown=root:root /home/tva/fpv/tva/tools/tvr.py /usr/local/lib/tva-claim/tvr.py
COPY tva-claim /usr/local/bin/tva-claim
COPY lake /tmp/tva-lake
COPY settings.json /app/.claude/settings.json
COPY lean-fixer.md /app/.claude/agents/lean-fixer.md
COPY CLAUDE.md /app/CLAUDE.md
COPY process_safety.md /
COPY example/ /app/examples/vec_add_f32/
# every `lake` waits its turn (the wrapper: builds alone, `lake env` alongside other readers)
RUN mv /opt/elan/bin/lake /opt/elan/bin/lake.elan && install -m 755 /tmp/tva-lake /opt/elan/bin/lake \
    && rm /tmp/tva-lake && touch /tmp/tva-lake.turn /tmp/tva-lake.lock && chmod 666 /tmp/tva-lake.turn /tmp/tva-lake.lock \
    && chmod 755 /usr/local/bin/tva-claim
WORKDIR /app/lean
# the library is built (no rebuild after the move), and the example's proof passes the check; then its
# build is removed, so a task's claim (the same module names) starts clean
RUN timeout 900 lake build TVA \
    && cp -a /app/examples/vec_add_f32 /app/claim && tva-claim prepare && tva-claim check /app/claim \
    && rm -rf /app/claim /app/lean/.lake/build/lib/lean/Claim.* /app/lean/.lake/build/lib/lean/Proof.* \
       /app/lean/.lake/build/lib/lean/Proof /app/lean/.lake/build/ir/Claim.* /app/lean/.lake/build/ir/Proof.* \
       /app/lean/.lake/build/ir/Proof /tmp/.tva-claim-stop.json
# The prover runs as `prover` (task.toml's [agent] user). Everything here is root's and read-only to it, except
# where its builds and its proof go: /tmp, and the build folders of /app/lean (sticky, so a build adds the claim's
# and the proof's outputs but never replaces or removes root's). Each task's image opens /app/claim the same way.
RUN useradd -m -u 2000 -s /bin/bash prover \
    && chmod -R go-w /app/.claude /app/CLAUDE.md /process_safety.md /usr/local/lib/tva-claim /app/examples \
    && chmod 1777 /app/lean/.lake \
    && find /app/lean/.lake/build -type d -exec chmod 1777 {} + \
    && test -z "$(find /app /opt/elan /process_safety.md /usr/local/bin/tva-claim \
       /usr/local/lib/tva-claim -not -user root -print -quit)"
WORKDIR /app
