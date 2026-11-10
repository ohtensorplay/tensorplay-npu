#!/usr/bin/env bash
# Recover a freshly created Ascend lab container to a testable state.
#
# The instance lives for two hours and the whole tree is rewritten when it is
# recreated, so every step here is idempotent and every phase is selectable:
# a build that runs out of the window can be resumed with `build` alone.
#
#   ./lab_setup.sh fetch    # sources + pinned third-party checkouts
#   ./lab_setup.sh build    # configure and compile the core
#   ./lab_setup.sh env      # the offline-converter environment
#   ./lab_setup.sh all
set -uo pipefail

ROOT=${ROOT:-/opt/atomgit}
CORE=$ROOT/tp
BACKEND=$ROOT/tensorplay-npu
CORE_PROJECT=${CORE_PROJECT:-lexing-2026/TensorPlay}
CORE_REF=${CORE_REF:-refs/heads/main}
BACKEND_PROJECT=${BACKEND_PROJECT:-ohtensorplay/tensorplay-npu}
BACKEND_REF=${BACKEND_REF:-refs/heads/main}
JOBS=${JOBS:-4}
LOGS=$ROOT/drv
mkdir -p "$LOGS"

archive() { echo "https://codeload.github.com/$1/tar.gz/$2"; }

untar_to() { # <url> <directory>
    local url=$1 directory=$2
    # The phase rewrites its target, so keep it inside the working root.
    case "$directory/" in
        "$ROOT"/*|"$HOME"/*) ;;
        *) echo "refusing to rewrite $directory"; return 1 ;;
    esac
    rm -rf "$directory" && mkdir -p "$directory" || return 1
    # smart-HTTP clones reset on this box; the archive endpoint does not.
    curl -fsSL --retry 3 --connect-timeout 20 "$url" \
        | tar -xz --strip-components=1 -C "$directory"
}

fetch() {
    echo "== sources =="
    untar_to "$(archive "$CORE_PROJECT" "$CORE_REF")" "$CORE" || return 1
    untar_to "$(archive "$BACKEND_PROJECT" "$BACKEND_REF")" "$BACKEND" || return 1
    echo "== pinned checkouts =="
    python3 "$BACKEND/scripts/lab_submodules.py" \
        --root "$CORE" --project "$CORE_PROJECT" --ref "$(basename "$CORE_REF")" || return 1
}

build() {
    echo "== configure =="
    # The optional math kit and the RPC transport are off: their checkouts are
    # not fetched, and the distributed transport pulls a header gloo provides.
    cmake -S "$CORE" -B "$CORE/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_MPS=OFF -DUSE_VULKAN=OFF \
        -DUSE_MKLDNN=OFF -DUSE_NNPACK=OFF \
        -DTP_USE_TENSORPIPE=OFF -DTP_USE_MPI=OFF \
        -DBUILD_TESTS=OFF || return 1
    echo "== compile =="
    # A single object can take several minutes here, so the build is left in
    # the foreground of a `nohup` wrapper that logs.
    make -C "$CORE/build" "-j$JOBS" _C || return 1
}

env_setup() {
    echo "== converter environment =="
    local bindir config cann
    bindir=$(dirname "$(command -v python3)")
    config=$(ls "$bindir"/python3.*-config 2>/dev/null | head -1)
    if [ -n "$config" ] && [ ! -x "$HOME/.local/bin/python3-config" ]; then
        # The converter asks for a prefix through `python3-config` and loads
        # the matching interpreter library; without a pair that agrees it
        # reports an invalid interpreter version and the tile back end fails
        # to initialise.
        mkdir -p "$HOME/.local/bin"
        printf '#!/bin/sh\nexec %s "$@"\n' "$config" > "$HOME/.local/bin/python3-config"
        chmod +x "$HOME/.local/bin/python3-config"
        echo "WRAPPER $HOME/.local/bin/python3-config -> $config"
    fi
    cann=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | sort | tail -1)
    if [ -n "$cann" ]; then
        cat > "$ROOT/lab_env.sh" <<EOF
export PATH="$HOME/.local/bin:\$PATH"
export PYTHONPATH="$CORE:$BACKEND:$cann/python/site-packages:$cann/opp/built-in/op_impl/ai_core/tbe:\${PYTHONPATH:-}"
EOF
        echo "ENV $ROOT/lab_env.sh"
    fi
}

main() {
    local phase=${1:-all}
    case "$phase" in
        fetch) fetch ;;
        build) build ;;
        env) env_setup ;;
        all) fetch && build && env_setup ;;
        *) echo "usage: $0 [fetch|build|env|all]"; return 2 ;;
    esac
    echo "PHASE_$phase_RC=$?"
}

main "$@"
