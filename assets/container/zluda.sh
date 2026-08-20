#!/usr/bin/env bash

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
# Distributed under the terms of the GNU General Public License, Version 3.0.

# Author  : Alejandro Gonzales-Irribarren, 2026
# Github  : alejandrogzi
# Contact : alejandrxgzi@gmail.com

# ZLUDA runtime shim: put ZLUDA's libcuda.so.1 ahead of everything, then exec hspZ.
#
# ZLUDA ships libnvcuda.so / libcuda.so.1 implemented on HIP, so the same driver-API binary
# that runs on NVIDIA runs on AMD with no rebuild. This is a correctness backend, not a
# performance reference.
set -euo pipefail

ZLUDA_DIR=${ZLUDA_DIR:-/opt/zluda}
if [ ! -d "$ZLUDA_DIR" ]; then
    echo "hspZ: no ZLUDA at $ZLUDA_DIR — this is the ZLUDA image, rebuild it" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$ZLUDA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# A missing /dev/kfd is the single most likely operator error, and the CUDA-shaped error that
# comes out of it otherwise is unhelpful. `--gpus all` is the NVIDIA flag and does nothing here.
if [ ! -e /dev/kfd ]; then
    echo "hspZ: /dev/kfd is not present — the ZLUDA image needs the AMD KFD device." >&2
    echo "      Run with: --device=/dev/kfd --device=/dev/dri --group-add video --group-add render" >&2
    echo "      (--gpus all is the NVIDIA flag and has no effect on this image.)" >&2
    exit 1
fi

# Pass-through dispatch: nextflow launches the container as `bash -c "<task>"`, operators
# invoke `hspZ run ...` or `run ...`, and the CI checks `--help` — all three must reach the
# right binary without hspZ being the ENTRYPOINT (an hspZ entrypoint would eat nextflow's
# `bash` argument and die on "unrecognized subcommand"). LD_LIBRARY_PATH above is inherited
# by whatever is exec'd, so the task shell gets ZLUDA's libcuda.so.1 too.
case "${1:-}" in
    hspZ)                      shift; exec hspZ "$@" ;;
    run|benchmark|compare|--help|--version|-h|-V)
                               exec hspZ "$@" ;;
    *)                         exec "$@" ;;
esac
