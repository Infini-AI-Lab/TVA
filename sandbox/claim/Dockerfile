# tva3-claim:<tag>: the image every claim task starts from (claim/build.sh). Its /app:
#   /app/claim/                 the task's claim (each task's image adds it) and the prover's proof
#   /app/examples/vec_add_f32/  a fixed claim with its complete proof (Proof.lean), checked at build
#   /app/lean/                  the TVA library with Mathlib, built; its lakefile adds the claim's library
#                               (ClaimTask: /app/claim's Claim and Proof)
#   /app/docs/                  TVA's docs
#   /app/CLAUDE.md, /app/.claude/   the prover's harness: how to work here, the hooks, the lean-fixer subagent.
#                               The prover starts in /app, so Claude Code loads both as the project's.
#   /process_safety.md          what never to kill; each task adds /instruction.md
#   /usr/local/bin/tva-check    the one tool (check, grade, verify, prepare, time, hooks)
#   /usr/local/lib/tva/replay.lean  its last step: the kernel re-checks each module, one at a time
#   /usr/local/bin/lake         the build lock in front of the toolchain's lake
# The verifier's image is a task's image too, with its own copy of the claim.
ARG BASE
FROM ${BASE}
COPY tva-check /usr/local/bin/tva-check
COPY replay.lean /usr/local/lib/tva/replay.lean
COPY lake /usr/local/bin/lake
COPY settings.json /app/.claude/settings.json
COPY lean-fixer.md /app/.claude/agents/lean-fixer.md
COPY CLAUDE.md /app/CLAUDE.md
COPY process_safety.md /process_safety.md
COPY docs/ /app/docs/
COPY example/ /app/examples/vec_add_f32/
WORKDIR /app/lean
# The claim's library; every `lake` waits its turn (/usr/local/bin/lake); the library is built (no rebuild after
# the lakefile change); the example's proof passes the check; then its build is removed, so a task's claim (the same
# module names) starts clean.
RUN printf '\n# the claim (/app/claim): the fixed Claim.lean and the proof, Proof.lean and Proof/\n[[lean_lib]]\nname = "ClaimTask"\nsrcDir = "../claim"\nroots = ["Claim", "Proof"]\n' >> lakefile.toml \
    && chmod 755 /usr/local/bin/tva-check /usr/local/bin/lake && chmod -R a+rX,go-w /usr/local/lib/tva \
    && touch /tmp/tva-lake.turn /tmp/tva-lake.lock && chmod 666 /tmp/tva-lake.turn /tmp/tva-lake.lock \
    && timeout 600 lake build TVA \
    && cp -a /app/examples/vec_add_f32 /app/claim && tva-check prepare && tva-check check /app/claim \
    && rm -rf /app/claim /app/lean/.lake/build/lib/lean/Claim.* /app/lean/.lake/build/lib/lean/Proof.* \
       /app/lean/.lake/build/lib/lean/Proof /app/lean/.lake/build/ir/Claim.* /app/lean/.lake/build/ir/Proof.* \
       /app/lean/.lake/build/ir/Proof /tmp/.tva-check-*
# The prover runs as `prover` (task.toml's [agent] user). Everything here is root's and read-only to it, except
# where its builds and its proof go: /tmp, and the build folders of /app/lean (sticky, so a build adds the claim's
# and the proof's outputs but never replaces or removes root's). Each task's image opens /app/claim the same way.
RUN useradd -m -u 2000 -s /bin/bash prover \
    && chmod -R a+rX,go-w /app/.claude /app/CLAUDE.md /app/docs /app/examples /app/lean/lakefile.toml \
       /process_safety.md \
    && chmod 1777 /app/lean/.lake \
    && find /app/lean/.lake/build -type d -exec chmod 1777 {} + \
    && test -z "$(find /app /opt/lean /process_safety.md /usr/local/bin/tva-check /usr/local/bin/lake /usr/local/lib/tva \
       -not -user root -print -quit)"
WORKDIR /app
