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

## [0.0.4] — 2026-09-13

Exact speed-ups for the GPU pass, a work-unit scheduler for multi-GPU runs, and a
new way to run one reference against many queries. Every arm of every measurement
below produced byte-identical output to 0.0.3 at the same plan and cap, on every
device tested (NVIDIA L4, RTX 4090, Tesla T4, AMD via ZLUDA).

- **One reference, many queries: `--query-list`.** `hspZ run --reference ref.fa --query-list queries.txt
  --output OUT` runs every listed query FASTA in one process: each reference bin is built and uploaded
  once for the whole list instead of once per query, while every query keeps its own plan, cap, seed
  batches, dedup scopes and `-D` history, so its output under `OUT/000001/`, `OUT/000002/`, … is
  byte-identical to its standalone run. `-Z` writes one archive per job; `OUT/queries.tsv` maps job →
  path, size, sha256, blocks, units, HSPs, completion. With `--gpus W ≥ 2` the whole batch's work units
  are spread over the workers by the same count partition a single query uses, and the receiver reorders
  each job's results, so the per-job output is unchanged. Measured against hg38 with 100 queries of 5 Mbp:
  on one L4, 10,400–11,600 s as separate runs versus 1,890–1,900 s batched (**−82%**); on two RTX 4090s,
  497/482 s at `--gpus 1` versus 255/255 s at `--gpus 2` (**−48%**, workers balanced to 1.2%, all 100 jobs
  identical), and −49% on two T4s. Both arms of each comparison ran the same pinned settings (device
  seeder; bucketing on for the L4, off for the RTX 4090s; partition forced at `--gpus 2`), not the
  `--gpus 1` defaults. Three 5 Mbp queries are −62%; whole-chromosome queries (61–121 Mbp) −2.6%;
  whole-genome queries are expected to gain under 1% (not measured in batch mode). All jobs must plan to identical reference bins; `-B 0`,
  `--kegalign-bins` and `--from-manifest` are rejected in batch mode; a failing job aborts the batch
  (partial outputs may remain); dumps become per-job siblings (`out.manifest.000001`). Bin-major order
  delays the first result, so amortized time per query is not request latency.
- **On-disk reference index: `hspZ index` and `run --index DIR`.** `hspZ index --reference hg38.fa --index DIR`
  builds every reference bin's seed tables and encoded bases once and writes them as raw arrays with a text
  manifest, published atomically. A later `hspZ run … --index DIR` loads them instead of rebuilding, inside the
  same prefetch, and the output is byte-identical. Before any upload the run proves the index matches: the parsed
  reference records must hash to the recorded value (a same-size edit is caught), the resolved plan must have
  exactly the index's bins, and every array must match its recorded length and checksum. Measured on an L4 with
  hg38 and 5 Mbp queries run as separate processes: three queries 333 → 87 s (**−74%**) and 100 queries
  10,883/10,789 → 2,683/2,678 s (**−75%**), after a one-off index build of 119–123 s producing 8.93 GB.
  Measured with bucketing pinned on, the device seeder and the index on local NVMe with a warm page cache.
  Loading the seven hg38 bins costs ~12.6 s per run; on an L4 all but the first load (~1.7 s) hide behind the
  GPU, each run still re-reads and re-hashes the reference (~4.7 s), and on faster GPUs a small query's loads
  set the pace. `-B 0` is rejected on both
  commands; put the index on local storage (a load above 10 s warns).
- **Unit-level static partition across GPUs (default on for matching devices).** With two or more GPUs the frozen
  plan's work units are split by count quotas (whole bins first, then contiguous query slices of as few bins as
  possible, one extra reference build per split bin) instead of whole-bin ownership, when every device the run
  uses is the same class (equal SM count and L2, nominal clocks within 10%) and each worker has its own device.
  Output is unchanged. On canonical hg38 × mm39 at `MAX_HITS=16,711,680`: seed-and-filter wall **−11.9%** on
  2× RTX 4090 (1,400 → 1,233 s) and **−5.1%** on 4× RTX 4090 (718 → 681 s); other worker counts and device
  classes are unmeasured. `HSPZ_UNIT_PARTITION=0` restores
  whole-bin ownership, `1` forces the partition; W=1, time-sliced workers and mixed device classes keep whole-bin
  ownership. The static-attribute guard cannot detect same-model cards with different sustained clocks, and on
  such a pair the balance depends on device order.
- **Reference-locality bucketing (`ref-loc-buckets`, default feature).** Between `find_hits` and the score gate,
  each hit chunk is stably reordered into reference-address buckets so consecutive warps gather from an
  L2-resident reference window; survivors are restored to their original order before `find_hsps`, so the HSP set
  and file bytes are unchanged. On a 72 W L4 the lower DRAM traffic lets the SM clock rise (1142 → 1370 MHz) and
  the whole-genome wall falls **6.5%** (hg38 × mm39, one GPU, disjoint reversed pairs). Cards that already run near
  their boost clock pay the bucketing pass instead (RTX 4090: +3.7%), so the choice is made at runtime:
  `HSPZ_REF_BUCKETS` unset means `auto` — each engine alternates both paths in blocks of ≥3 s of gate time,
  discards each block's first second, and commits to the faster settled path (L4 → on, 4090 → off, measured).
  An engine needs about 20 s of gate time to decide, so short engines (e.g. 5 Mbp queries run as separate
  processes) stay on the off path; pin `HSPZ_REF_BUCKETS=1` on L4-class cards for those.
  `1`/`0` force a path; `HSPZ_REF_BUCKET_SHIFT=<16..31>` pins the window size; cards whose L2 cannot hold the
  smallest window (T4) stay off. `--time` prints the per-block measurements and the decision per engine. The
  window budget is three quarters of the L2: on an L4 that selects the 32 MiB window, measured 2.8–3.0% less GPU
  time than 16 MiB on two whole-genome work units and 2.75% less wall on the full run (9,877 vs 10,156 s).
- **Sparse chunk walk (default).** Every seed batch that exceeds `--max-hits` used to copy its whole cumulative hit
  array to the host (~6.8 MB, pageable) to locate two or three chunk boundaries. The walk now fetches only the
  ≤1 KB block holding each boundary, producing the same chunks. On two RTX 4090s at whole genome this is **−9.6%**
  wall (1,634 → 1,480 s, disjoint reversed pairs); on an L4 no regression was detected (one overlapping unit-0 pair).
  `HSPZ_CHUNK_WALK=full` restores the old path.
- **`hspZ hits-estimate` (CPU only).** Prints, before any GPU work, the exact number of seed hits each work unit
  (reference bin × query block) will produce, from dense k-mer histograms with the transition variants folded in;
  it matches the `--time` ledger exactly (whole genome: 42 units, 6.12e12 hits). GPU time per unit is ~0.37 ns per
  hit on a 4090, so the rows are the unit costs of a multi-GPU schedule. 56–62 s on 32 cores for hg38 × mm39
  (10 GiB RSS); `--stride N` samples the query side. A planning and diagnostic tool; `run` does not call it.
- **Automatic layout (`-B 0`, opt-in measurement tool).** `-B 0` resolves a count-balanced layout for the current
  machine instead of the fixed 500 Mbp default; it changes the HSP set (a different frozen plan), so measure per
  machine before adopting it (a 2× T4 screen was 17–22% slower than the default). At `--gpus W ≥ 2` the reference
  takes `R = min(records, ceil(R_default/W)·W)` LPT-balanced bins, and the query collapses to one bin when the whole
  query fits both the per-worker device budget and the host preflight (an explicit `--query-block-size` is honoured);
  `W=1` resolves to exactly the 500 Mbp plan, and a candidate that fails the final fit falls back to the default
  policy rather than erroring. Resolved sizes are stored in the manifest and `--dump-plan`, and one `layout: auto W=…`
  line prints the flags that reproduce them. `-B 0` with `--kegalign-bins` is an error;
  `HSPZ_LAYOUT_FORCE_WORKERS` exercises the W ≥ 2 path on one device.
- **Ledger and receiver checks.** Every run prints `schedule:` lines with the scheduling policy, its reason and each
  worker's visits. `--time` prints one `unit ledger:` row per work unit (worker, device, bins, bp, GPU ms, host
  start/end, pack/swap ms, seeds, hits, raw and final HSPs) plus per-worker finish and busy lines, attributes gap
  pairs per worker, counts the >1 s transitions the gap accounting discards, and totals the host phases. The emitter
  rejects duplicate or out-of-plan work units and verifies completeness after joining the workers, reporting a
  worker's own error first. The ref-buckets autotune drops and restarts an inconclusive block rather than guessing.
- **Foundations.** The `--max-hits` target (semantic: it decides chunk boundaries and therefore output) is separated
  from the physical hit capacity derived from free VRAM (which decides success or failure only); the historical
  over-cap tail that aborted whole-genome runs is admitted without changing any chunk. `--dump-plan`,
  `--dump-manifest` and `--from-manifest` freeze and replay a plan, and a second node validates fit and fails rather
  than replanning. Multi-GPU output is byte-identical to one-GPU output at the same plan and cap.
- **Fixes.** A seed batch whose hits exceed 2^32 is refused by name: its 32-bit hit total used to wrap,
  either aborting with a misleading "not monotonic" error or, when the wrapped value fell under the cap,
  writing past the anchor buffer (reachable with unmasked, closely related inputs and ~1 Gbp bins; KegAlign's
  scan wraps too). `hspZ index` refuses record names its manifest cannot store and bins over 4,294,967,295 bp
  before building. `-I 0` and `-C 0` are argument errors (`-I 0` used to loop forever). With `-Z`, the `--time`
  footer reports the archive's size on disk (it read the size before the gzip trailer was written and fell back
  to the formatted byte count); `--dump-raw` files are flushed explicitly, so a failed final write is an error
  instead of a silently truncated dump; frozen and index manifests list the `ref-loc-buckets` feature; the
  `--time` line for `HSPZ_REF_BUCKETS=0` names the override; `--help` for `-B` and `--gpus` describes the
  shipped multi-GPU policy instead of recommending one reference bin per worker.
- **Measured but not shipped** (recorded so nobody repeats them): certified query-context rejection (0.19% of hits),
  nibble-packed reference at genome scale (+0.1%), a 64 M hit cap (+1%), per-chunk and paired-chunk autotuning of the
  bucketing pass (both misread the clock-mediated L4 win), overlapping the next chunk's `find_hits` with the current
  score gate (a wash on both an L4 and a 4090: the gate loses what the expansion saves), computing hit-count weights
  inside a run to balance the schedule (exact, but 56 s of host work against a 7.6 s budget), and a 7 × 12 query
  layout for four GPUs (−4.3%, below the bar).

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
