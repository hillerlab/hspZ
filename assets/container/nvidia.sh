#!/usr/bin/env bash

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
# Distributed under the terms of the GNU General Public License, Version 3.0.

# Author  : Alejandro Gonzales-Irribarren, 2026
# Github  : alejandrogzi
# Contact : alejandrxgzi@gmail.com

# NVIDIA runtime shim. Its only job is to turn a missing driver into a sentence.
#
# hspZ has libcuda.so.1 as a DT_NEEDED, and the NVIDIA Container Toolkit injects that library
# from the host driver only when the container is started with --gpus. Without it the dynamic
# loader kills the process before main(), so even `--help` dies with
# "error while loading shared libraries: libcuda.so.1" — accurate, but it does not tell an
# operator what to do. We cannot make --help work driver-less without dlopen'ing the driver,
# so the honest fix is a clear message rather than a stub libcuda (a stub that ever shadowed
# the real driver would make every GPU run fail as "no CUDA device", which is far worse).
set -euo pipefail

if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1' \
   && [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    cat >&2 <<MSG
hspZ: no NVIDIA driver in this container (libcuda.so.1 is absent).

  The NVIDIA Container Toolkit injects the host driver only when you pass --gpus:

      docker run --rm --gpus all -v "\$PWD:/data" <image> run -r REF.fa -q QRY.fa -o out/

  This image is the NVIDIA build. For AMD via ZLUDA use the :zluda tag, which instead wants
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render.
MSG
    exit 1
fi
exec hspZ "$@"
