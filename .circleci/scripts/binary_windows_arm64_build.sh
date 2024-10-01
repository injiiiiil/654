#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

export CUDA_VERSION="cpu"
export USE_SCCACHE=1
export SCCACHE_BUCKET=ossci-compiler-cache
export SCCACHE_IGNORE_SERVER_IO_ERROR=1
export VC_YEAR=2022

echo "Free space on filesystem before build:"
df -h

pushd "$BUILDER_ROOT"
elif [[ "$PACKAGE_TYPE" == 'wheel' || "$PACKAGE_TYPE" == 'libtorch' ]]; then
    export NIGHTLIES_PYTORCH_ROOT="$PYTORCH_ROOT"
    ./windows/internal/arm64/build_wheels.bat
fi

echo "Free space on filesystem after build:"
df -h
