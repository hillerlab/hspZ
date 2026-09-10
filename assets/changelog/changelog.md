<p align="center">
  <p align="center">
    <img width=100 align="center" src="../figures/hz.png" >
  </p>

<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="../figures/hillerlab-dark.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="../figures/hillerlab-light.png"
    >
    <img
      width="200"
      alt="Hiller Lab"
      src="../figures/hillerlab-light.png"
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

## [0.0.5] — 2026-09-10

Unit-level static partitioning behind a device-class guard, with a per-unit ledger
and emitter-side receiver checks. Output bytes are unchanged.

- **Unit-level static partition (default on for matching GPUs).** With two or more
  GPUs the frozen plan's work units are now split across workers by count quotas
  (whole bins first, then contiguous query slices of as few bins as possible, one
  extra reference build per split bin) instead of whole-bin ownership, when every
  device the run uses is the same class (equal SM count and L2, nominal clocks
  within 10%) and each worker has its own device. Output bytes are unchanged.
  Measured on canonical hg38 × mm39 at `MAX_HITS=16,711,680`: seed-and-filter
  wall **−11.9%** on 2× RTX 4090 (1,400 → 1,233 s) and **−5.1%** on 4× RTX 4090
  (718 → 681 s), identical output. `HSPZ_UNIT_PARTITION=0` restores whole-bin
  ownership, `1` forces the partition; W=1, time-sliced workers and mixed device
  classes keep whole-bin ownership. The static-attribute guard cannot detect
  same-model cards with different sustained clocks (on such a pair the balance
  depends on device order).
- **Per-unit ledger.** `--time` prints one `unit ledger:` row per work unit
  (worker, device, bins, bp, GPU ms, host start/end, pack/swap ms, seeds, hits,
  raw/final HSPs) plus per-worker finish/busy lines, and every run prints
  `schedule:` lines with the policy, its reason, and each worker's visits.
- **Receiver checks.** The emitter now rejects duplicate or out-of-plan work units
  and verifies completeness after joining the workers, reporting a worker's own
  error first.
- **Automatic layout (`-B 0`, opt-in measurement tool).** `-B 0` resolves a
  count-balanced layout for the current machine instead of the fixed 500 Mbp
  default; it changes the HSP set (a different frozen plan), so measure per machine
  before adopting it (a 2× T4 screen was 17–22% slower than the default). At `--gpus W ≥ 2` the reference
  takes `R = min(records, ceil(R_default/W)·W)` LPT-balanced bins, and the query
  collapses to one bin when the whole query fits both the per-worker device budget
  and the host preflight (an explicit `--query-block-size` is honoured); `W=1`
  resolves to exactly the 500 Mbp plan, and a candidate that fails the final fit
  falls back to the default policy rather than erroring. Resolved sizes are stored
  in the manifest/`--dump-plan`, and one `layout: auto W=…` line prints the flags
  that reproduce them. `-B 0` with `--kegalign-bins` is an error.
  `HSPZ_LAYOUT_FORCE_WORKERS` exercises the W≥2 path on one device.
- **Diagnostics.** The ref-buckets autotune drops and restarts an OFF block that
  reaches twice the block span with fewer than four settled chunks (an
  inconclusive ON block decides OFF, with a ledger line). `--time` attributes
  gap pairs per worker, counts the >1 s transitions the gap accounting discards,
  and totals the host phases (query pack, swap, chunk prep, seed table,
  standalone reference prep).

## [0.0.4] — 2026-09-09

Two exact speed-ups, both byte-identical to 0.0.3 on every device tested (NVIDIA L4,
RTX 4090, AMD via ZLUDA), plus the foundation repairs behind them.

- **Reference-locality bucketing (`ref-loc-buckets`, default feature).** Between
  `find_hits` and the score gate, each hit chunk is stably reordered into
  reference-address buckets so consecutive warps gather from an L2-resident
  reference window; survivors are restored to their original order before
  `find_hsps`, so the HSP set and file bytes are unchanged. On a 72 W L4 the
  lower DRAM traffic lets the SM clock rise (1142 → 1370 MHz) and the whole-genome
  wall falls **6.5%** (hg38 × mm39, one GPU, disjoint reversed pairs). Cards that
  already run near their boost clock pay the bucketing pass instead (RTX 4090:
  +3.7%), so the choice is made at runtime: `HSPZ_REF_BUCKETS` unset means `auto`
  — each engine alternates both paths in blocks of ≥3 s of gate time, discards
  each block's first second, and commits to the faster settled path (L4 → on,
  4090 → off, measured). `1`/`0` force a path; `HSPZ_REF_BUCKET_SHIFT=<16..31>`
  pins the window size; cards whose L2 cannot hold the smallest window (T4) stay
  off. `--time` prints the per-block measurements and the decision per engine.
  The window budget is three quarters of the L2: on an L4 that selects the
  32 MiB window, measured 2.8–3.0% less GPU time than 16 MiB on two whole-genome
  work units and 2.75% less wall on the full hg38 × mm39 run (9,877 vs 10,156 s),
  with identical output.
- **Sparse chunk walk (default).** Every seed batch that exceeds `--max-hits`
  used to copy its whole cumulative hit array to the host (~6.8 MB, pageable) to
  locate two or three chunk boundaries. The walk now fetches only the ≤1 KB block
  holding each boundary, producing the same chunks. On two RTX 4090s at whole
  genome this is **−9.6%** wall (1,634 → 1,480 s, disjoint reversed pairs); on
  an L4 the copy was hidden under the gate, so nothing changes.
  `HSPZ_CHUNK_WALK=full` restores the old path.
- **Foundations.** The `--max-hits` target (semantic, decides chunk boundaries and
  therefore output) is now separated from the physical hit capacity derived from
  free VRAM (decides success or failure only); the historical over-cap tail that
  aborted whole-genome runs is admitted without changing any chunk. `--dump-plan`
  / `--dump-manifest` / `--from-manifest` freeze and replay a plan (a second node
  validates fit and fails rather than replanning). The `--time` ledger gained
  query-pack / swap timers, per-worker GPU-busy and stage-gap notes, and an
  explicit host-memory budget line. Multi-GPU output is byte-identical to the
  one-GPU output at the same plan and cap.
- **Measured but not shipped** (recorded so nobody repeats them): certified
  query-context rejection (0.19% of hits), nibble-packed reference at genome
  scale (+0.1%), a 64 M hit cap (+1%), per-chunk and paired-chunk autotuning of
  the bucketing pass (both misread the clock-mediated L4 win).

## [0.0.3] — 2026-08-21

Warp-per-seed `find_hits` for dense launches, on by default.

- **`find-hits-warp`** — when a `MAX_HITS` launch has at least 16 hits per
  seed, one warp owns that seed's `pos_table` walk so adjacent lanes read
  adjacent positions and write adjacent packed anchors. Sparse launches keep
  the thread-per-seed kernel. Destination indices, `MAX_HITS` chunking, and
  the HSP set are unchanged. This is a default feature alongside
  `simd-prelude`; disable with `--no-default-features` if you need the old
  compile-out.
- **`--hit-stats`** — the distribution table now reports hit-weighted
  bucket shares and lane utilisation under both mappings, so a repeat-rich
  tail is visible even when mean hits/seed looks modest.
- On an NVIDIA L4 high-density proxy (hg38 × a 250 kb slice of mm39 chr19),
  the warp path cut `find_hits` by 56% and whole wall by 3.8–4.0% with
  disjoint reversed rounds and exact hashes. Canonical A/B (sparse) does not
  regress: those launches stay on the thread path.

## [0.0.2] — 2026-08-20

Container and nextflow integration fixes.

- **Pass-through entrypoint shims** — `nvidia.sh` and `zluda.sh` no longer
  `exec hspZ` unconditionally. They keep the driver/KFD checks, then dispatch:
  a leading `hspZ` is stripped, known hspZ invocations (`run`, `benchmark`,
  `compare`, `--help`, `--version`) go straight to the binary, and anything
  else — nextflow's `bash -c "<task>"` launcher, `sh`, ad-hoc shells — is
  exec'd as-is. hspZ stays a plain `/usr/local/bin/hspZ` binary and is not the
  container ENTRYPOINT, so pipelines that call `hspZ run ...` inside the
  container work without `--entrypoint ''` overrides.
- **smoke.sh** — new `hspZ run` arm on the frozen `repeat` fixture (57 HSPs,
  digest `d01edd2118cf5fa5`), locking the nextflow-style invocation to the
  same answer as the bare `run` form.
- **Formatting** — `assets/container/*` re-wrapped for readability; the
  Dockerfile change is whitespace-only.

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
