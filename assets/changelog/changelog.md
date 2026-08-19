<p align="center">
  <p align="center">
    <img width=100 align="center" src="./assets/figures/hz.png" >
  </p>

<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="./assets/figures/hillerlab-dark.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="./assets/figures/hillerlab-light.png"
    >
    <img
      width="200"
      alt="Hiller Lab"
      src="./assets/figures/hillerlab-light.png"
    >
  </picture>
</p>

  <span>
    <h1 align="center">
        CHANGELOG
    </h1>
  </span>

  <p align="center">
    <a href="https://github.com/hillerlab/hspZ" reference="_blank">
      <img alt="GitHub License" src="https://img.shields.io/github/license/hillerlab/hspZ?color=blue">
    </a>
  </p>
</p>

All notable changes to `hspZ` are documented here, newest first.

## [0.0.1] — 2026-08-19

First preview release. hspZ is a standalone, GPU-accelerated replacement for
the HSP-finding half of KegAlign (C++/CUDA), reimplemented in Rust on top of
NVIDIA's cuda-oxide. It seeds spaced k-mers, extends them with X-drop, applies
the entropy-aware score gate, and emits `.segments` files under KegAlign's own
naming scheme.

Highlights of what's in the box:

- **Three subcommands** — `run` (seed + filter + emit), `benchmark` (timed
  cold/warm iterations with JSON records), and `compare` (runs the C++ oracle
  and this implementation back to back and diffs runtime + output).
- **Input flexibility** — FASTA, FASTA.gz, and 2bit files, detected by magic
  bytes and decoded to one packed representation.
- **Planning** — whole chromosomes binned by an LPT-balanced planner with
  independent reference/query block targets; `--kegalign-bins` reproduces
  KegAlign's sequential fill for matched-granularity benchmarking.
- **Multi-GPU execution** — reference bins split across workers by
  deterministic LPT, output replayed in ordinal order so results never depend
  on which GPU finished first. Above one worker, seeds are generated on the
  device and uploaded on a second stream, overlapped with the previous batch's
  compute.
- **Output options** — in-memory diagonal partitioning (`-D`) and reproducible
  `.tar.gz` archives (`-Z`) with pinned mtimes; both byte-identical 1-GPU vs
  N-GPU.
- **Parity with the oracle** — byte-identical output wherever the plan and
  `MAX_HITS` match (whole hg38 × mm39: ~15.16 M HSPs), 2.13–2.23x faster HSP
  generation than KegAlign on one NVIDIA L4, and 89.7% multi-GPU scaling
  efficiency on 4x RTX 4090.
- **Friendly CI** — GitHub Actions workflows for building, testing, and
  shipping the crate (check `.github/` for details).

Warnings and known limits for this preview:

- Uses a preview release of cuda-oxide; driver support is limited to NVIDIA
  and ZLUDA, so native validation variants are kept behind opt-in features
  (`nvidia-*`, `device-seeds`, `warp-score-gate`, `dense-anchors`,
  `left-pair-tile`, `simd-prelude`).
- Block layout and `--max-hits` set the dedup scope: different layouts produce
  different (not wrong) HSP sets, so keep a frozen digest per layout.
- LASTZ, gapped extension, chaining, and AXT/PSL output are out of scope for
  now — this is the seed + filter stage, period.
- Preview builds are stripped (`strip = true`), so panic backtraces carry no
  symbol names.
