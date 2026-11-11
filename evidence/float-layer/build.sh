#!/bin/sh
# Compile the library's float definitions (lean/TVA/Math/Defs.lean) and Driver.lean to one native
# executable, with the Lean toolchain that lean/lean-toolchain names.
# Usage, from this folder: sh build.sh <build folder>
# The toolchain is found through lake, elan's proxy, run in lean/: it runs the toolchain that
# lean/lean-toolchain names, and its `lake env` puts that toolchain's bin folder first on PATH.
# LAKE names the lake to use (default: lake on PATH, else elan's).
set -e
OUT=$(realpath "$1")
HERE=$(pwd)
LAKE=${LAKE:-$(command -v lake || echo "$HOME/.elan/bin/lake")}
cd ../../lean
BIN=$("$LAKE" env sh -c 'dirname "$(command -v lean)"')
echo "toolchain $(cat lean-toolchain): $BIN"
mkdir -p "$OUT/TVA/Math"
env -u LEAN_PATH "$BIN/lean" -R . -o "$OUT/TVA/Math/Defs.olean" -c "$OUT/Defs.c" TVA/Math/Defs.lean
cd "$HERE"
LEAN_PATH="$OUT" "$BIN/lean" -R . -o "$OUT/Driver.olean" -c "$OUT/Driver.c" Driver.lean
"$BIN/leanc" -O3 -o "$OUT/driver" "$OUT/Driver.c" "$OUT/Defs.c"
echo "built $OUT/driver"
