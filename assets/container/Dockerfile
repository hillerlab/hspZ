# syntax=docker/dockerfile:1

# Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
# Distributed under the terms of the GNU General Public License, Version 3.0.

# Author  : Alejandro Gonzales-Irribarren, 2026
# Github  : alejandrogzi
# Contact : alejandrxgzi@gmail.com

# hspZ container images. One builder, two runtimes.
#
#   docker build -f assets/container/Dockerfile --target nvidia -t hspz:test  .
#   docker build -f assets/container/Dockerfile --target zluda  -t hspz:zluda .
#
# Context is the repository root, not this directory (Cargo.lock pins the cuda-oxide SHA
# the build must use, and .cargo/cuda-oxide.toml carries default-arch).
#
# Why the CUDA 12.4 redistributables instead of a CUDA devel base (plan amendment A1):
# cuda.h aliases cuEventElapsedTime to cuEventElapsedTime_v2 from CUDA 12.8 on, and a host
# driver older than 570 does not export it — round 76 lost a rented pod to exactly that on
# driver 550.163.01. Building against 12.4 headers puts the driver floor at ~550 instead of
# ~570, byte-identically (round 79). Every other _v2 symbol hspZ needs dates to CUDA 4.0.
# `smoke.sh` re-checks this on the built artifact so the guarantee cannot rot.
#
# Canonical A gave the same digest on every device:
#
#   T4          sm_75 native              4,248 HSPs  852ce647a1f9cd74
#   L4          sm_75 PTX JIT'd to Ada    4,248 HSPs  852ce647a1f9cd74
#   L4          sm_80                     4,248 HSPs  852ce647a1f9cd74
#   RX 6500 XT  sm_75 via ZLUDA/HIP       byte-identical to sm_80

# ---------------------------------------------------------------------------- builder
FROM ubuntu:24.04 AS builder

ARG RUST_NIGHTLY=nightly-2026-04-03
ARG CUDA_REDIST_VER=12.4.1
# Both architectures are always built; the runtime stages choose. sm_75 is Turing (T4) and
# is forward-compatible upward by PTX JIT; sm_80 is what ZLUDA's bundled ptxas expects.
ARG HSPZ_ARCHS="sm_75 sm_80"

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

# clang/libclang 18 is what cuda-oxide actually resolves (doctor reports resource dir
# /usr/lib/clang/18) — the plan's LLVM 21 requirement does not apply, and `llc` comes from
# rustup's llvm-tools component, not from a system LLVM (amendment A2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential pkg-config python3 \
        clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.4 headers, libNVVM, libdevice, ptxas and nvJitLink. cuda_cudart gives cuda.h and
# the libcuda stub to link against; cuda_nvcc gives the rest. No GPU is needed to build.
ENV CUDA_HOME=/opt/cuda CUDA_TOOLKIT_PATH=/opt/cuda
RUN set -eux; \
    base=https://developer.download.nvidia.com/compute/cuda/redist; \
    curl -fsSL "$base/redistrib_${CUDA_REDIST_VER}.json" \
        -o /tmp/redist.json; \
    python3 -c "import json;i=json.load(open('/tmp/redist.json'));\
[print(i[p]['linux-x86_64']['relative_path']) for p in ('cuda_cudart','cuda_nvcc')]" \
        > /tmp/paths.txt; \
    mkdir -p /tmp/cuda "$CUDA_HOME"; \
    cd /tmp/cuda; \
    while read -r rel; do \
        curl -fsSL "$base/$rel" -o pkg.tar.xz; \
        tar -xf pkg.tar.xz; \
        rm pkg.tar.xz; \
    done < /tmp/paths.txt; \
    for a in *-archive; do cp -r "$a"/. "$CUDA_HOME"/; done; \
    cd /; \
    rm -rf /tmp/cuda /tmp/redist.json /tmp/paths.txt; \
    test -f "$CUDA_HOME/include/cuda.h"; \
    grep -q "define CUDA_VERSION 12040" "$CUDA_HOME/include/cuda.h"; \
    test -f "$CUDA_HOME/lib/stubs/libcuda.so" \
        || test -f "$CUDA_HOME/lib64/stubs/libcuda.so"

ENV PATH=$CUDA_HOME/bin:$PATH \
    LIBRARY_PATH=$CUDA_HOME/lib/stubs:$CUDA_HOME/lib64/stubs \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64

# Toolchain before source, so a source edit does not re-download Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain "$RUST_NIGHTLY" --profile minimal \
          -c rust-src -c rustc-dev -c llvm-tools \
    && rustc --version && test -x "$(rustup which llc 2>/dev/null || echo /usr/local/rustup/toolchains/$RUST_NIGHTLY-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llc)"

WORKDIR /src

# cargo-oxide is pinned to the SHA in Cargo.lock, never to upstream HEAD (plan Phase 1.3).
COPY Cargo.lock Cargo.toml rust-toolchain.toml ./
RUN set -eux; \
    sha=$(grep -m1 -o 'cuda-oxide\.git#[0-9a-f]\{40\}' Cargo.lock \
        | cut -d'#' -f2); \
    test -n "$sha"; \
    echo "cuda-oxide pin: $sha" | tee /cuda-oxide.sha; \
    cargo install cargo-oxide --git https://github.com/NVlabs/cuda-oxide.git \
        --rev "$sha" --locked --root /usr/local

COPY . .
# One target tree per architecture so artifacts cannot leak between them.
RUN set -eux; \
    mkdir -p /build/bin; \
    for arch in $HSPZ_ARCHS; do \
        CARGO_TARGET_DIR=/build/target-$arch \
            cargo oxide build --arch "$arch" -- --release --locked; \
        cp "/build/target-$arch/release/hspZ" "/build/bin/hspZ-$arch"; \
        strings -n 6 "/build/bin/hspZ-$arch" \
            | grep -q "\.target $arch" \
            || { echo "FATAL: hspZ-$arch does not carry .target $arch PTX" >&2; exit 1; }; \
        bad=$(nm -D --undefined-only "/build/bin/hspZ-$arch" \
                | grep -oE 'cu[A-Za-z]+_v[0-9]+' \
                | grep -vE \
                    'cuDevicePrimaryCtxRelease_v2|cuEventDestroy_v2|cuMemAllocHost_v2|cuMemAlloc_v2|cuMemcpyDtoHAsync_v2|cuMemcpyHtoDAsync_v2|cuMemcpyHtoD_v2|cuMemFree_v2|cuMemGetInfo_v2|cuStreamDestroy_v2' \
                || true); \
        test -z "$bad" \
            || { echo "FATAL: hspZ-$arch needs post-CUDA-4.0 driver symbols: $bad" >&2; exit 1; }; \
        ldd "/build/bin/hspZ-$arch" || true; \
    done; \
    ls -la /build/bin

# Deterministic fixtures (fixed PRNG seed) baked in, so smoke needs no python at runtime and
# no external data. Their expected answers are the ones assets/tests/parity.sh already validates
# against the C++ oracle (amendment A4) — the container is not allowed to invent its own.
RUN mkdir -p /build/fixtures && python3 assets/tests/gen_fixtures.py /build/fixtures \
    && ls /build/fixtures | wc -l

# ---------------------------------------------------------------------------- nvidia
# `base` and not `runtime`: hspZ links only libcuda.so.1, which the NVIDIA container toolkit
# injects from the host driver, and the embedded PTX is JIT'd by that driver — libNVVM is a
# build-time dependency only (amendment A5, proven by rounds 76-79 on a bare Ubuntu pod).
# Base image glibc must match the builder's: the builder is ubuntu 24.04 (glibc 2.39), so a
# 22.04 runtime (2.35) would refuse the binary. The runtime base's own CUDA version is
# irrelevant — it ships no CUDA library hspZ uses, only the container-toolkit labels — while
# the *driver floor* is set by the 12.4 headers the builder compiled against (amendment A1).
FROM nvidia/cuda:12.6.3-base-ubuntu24.04 AS nvidia
ARG HSPZ_NVIDIA_ARCH=sm_75
ARG GIT_SHA=unknown
ARG HSPZ_VERSION=0.0.2
LABEL org.opencontainers.image.source="https://github.com/hillerlab/hspZ" \
      org.opencontainers.image.revision="$GIT_SHA" \
      org.opencontainers.image.version="$HSPZ_VERSION" \
      hspZ.backend="nvidia" \
      hspZ.arch="$HSPZ_NVIDIA_ARCH" \
      hspZ.cuda_headers="12.4.1" \
      hspZ.driver_floor="550"
COPY --from=builder /build/bin/hspZ-${HSPZ_NVIDIA_ARCH} /usr/local/bin/hspZ
COPY --from=builder /cuda-oxide.sha /usr/local/share/hspZ/cuda-oxide.sha
COPY --from=builder /build/fixtures /opt/hspZ/fixtures
COPY assets/container/nvidia.sh /usr/local/bin/nvidia.sh
RUN chmod +x /usr/local/bin/nvidia.sh
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/nvidia.sh"]
CMD ["--help"]

# ---------------------------------------------------------------------------- zluda
# Always the sm_80 artifact: ZLUDA's bundled ptxas expects sm_80. Never sm_75 here.
# 24.04 for the glibc match with the builder. The plan asks for a ROCm 6.1.x runtime to track
# the host (rocm-runtime 6.1.2 here), but ROCm publishes 6.1.2 only for ubuntu 22.04 (glibc
# 2.35) and starts at 6.2 on 24.04 — so this is the closest 6.x that can run the builder's
# binary. Container userspace is 6.2 against a 6.1.2 host kernel driver, which is the one
# assumption the local RX 6500 XT smoke exists to test; if it fails, move the builder to 22.04
# (glibc 2.35) and use rocm/dev-ubuntu-22.04:6.1.2 instead. Still 6.x: do not jump to 7.
FROM rocm/dev-ubuntu-24.04:6.2 AS zluda
# Pinned by tag AND asset, because ZLUDA release assets carry a build hash in the filename
# (zluda-linux-<sha>.tar.gz) — a guessed "zluda-linux-x86_64.tar.gz" 404s. v6 is the last
# non-preview release (2026-06-29); everything newer is v7-preview, and the plan says do not
# jump to 7 until 6.1 works.
ARG ZLUDA_VERSION=v6
ARG ZLUDA_ASSET=zluda-linux-3fe12063.tar.gz
ARG GIT_SHA=unknown
ARG HSPZ_VERSION=0.0.1

LABEL org.opencontainers.image.source="https://github.com/hillerlab/hspZ" \
      org.opencontainers.image.revision="$GIT_SHA" \
      org.opencontainers.image.version="$HSPZ_VERSION" \
      hspZ.backend="zluda" \
      hspZ.arch="sm_80" \
      hspZ.zluda="$ZLUDA_VERSION"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -fsSL -o /tmp/zluda.tar.gz \
        "https://github.com/vosen/ZLUDA/releases/download/${ZLUDA_VERSION}/${ZLUDA_ASSET}"; \
    mkdir -p /opt/zluda; \
    tar -xzf /tmp/zluda.tar.gz -C /opt/zluda --strip-components=1 \
        || tar -xzf /tmp/zluda.tar.gz -C /opt/zluda; \
    rm /tmp/zluda.tar.gz; \
    test -e /opt/zluda/libcuda.so.1 \
        || { echo "FATAL: ZLUDA tarball has no libcuda.so.1" >&2; \
             find /opt/zluda -maxdepth 2 -name "libcuda*" >&2; exit 1; }; \
    ls /opt/zluda | head

COPY --from=builder /build/bin/hspZ-sm_80 /usr/local/bin/hspZ
COPY --from=builder /cuda-oxide.sha /usr/local/share/hspZ/cuda-oxide.sha
COPY --from=builder /build/fixtures /opt/hspZ/fixtures
COPY assets/container/zluda.sh /usr/local/bin/zluda.sh

RUN chmod +x /usr/local/bin/zluda.sh

WORKDIR /data

ENTRYPOINT ["/usr/local/bin/zluda.sh"]
CMD ["--help"]
