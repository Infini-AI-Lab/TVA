# tva3-base:<hash>: the Lean toolchain, Mathlib and the TVA library, built (sandbox/build_base.sh).
#
# Built from three contexts, all copied, nothing downloaded but Ubuntu's packages:
#   lean       the library's source (lakefile.toml, lake-manifest.json, lean-toolchain, TVA.lean, TVA/), a
#              snapshot taken by build_base.sh (its content hash is the tag)
#   packages   lean/.lake/packages: Mathlib and its dependencies, built
#   toolchain  the Lean toolchain lean-toolchain names (from the host's elan)
#
# /app/lean and /opt/lean are root's and readable by every user (the files are made so in a first stage, so that no
# layer of the image copies them again to change a mode). Every module of TVA/ is built: TVA and the proof kit.
FROM ubuntu:24.04 AS stage
COPY --from=toolchain . /stage/opt/lean
COPY --from=packages . /stage/app/lean/.lake/packages
COPY --from=lean . /stage/app/lean
RUN chown -R root:root /stage && chmod -R a+rX,go-w /stage

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/lean/bin
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates git python3 procps tini util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
# (Lake runs git in the packages' folders; a user other than their owner must be allowed to.)
COPY --from=stage /stage/opt/lean /opt/lean
COPY --from=stage /stage/app /app
WORKDIR /app/lean
# Mathlib is already built (no rebuild: the toolchain and the sources are the same); then every TVA module.
RUN lean --version \
    && mods="$(find TVA -name '*.lean' | sed 's/\.lean$//; s#/#.#g' | sort)" \
    && timeout 5400 lake build TVA $mods \
    && chmod -R a+rX,go-w /app/lean/.lake/build \
    && test -z "$(find /app /opt/lean ! -perm -a+r -print -quit)"
WORKDIR /app
