// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Command-line surface: `Cli`, the per-subcommand argument structs, and the
//! `Tuning` group that `CompareArgs` flattens. Every argument carries a short
//! flag where a letter survives clap's reserved `-h`/`-V`, so all three
//! subcommands keep a terse form.

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "hspZ",
    bin_name = "hspZ",
    version = env!("CARGO_PKG_VERSION"),
    about = "GPU-accelerated high-scoring ungapped alignment pair backend",
    author = env!("CARGO_PKG_AUTHORS")
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// The five subcommands hspZ accepts.
#[derive(Subcommand)]
pub(crate) enum Command {
    /// Seed, extend and filter one reference/query pair.
    Run(RunArgs),
    /// Write one reference's per-bin seed tables to an on-disk index.
    Index(IndexArgs),
    /// Cold/warm benchmark: timed runs against wall clock.
    Benchmark(BenchArgs),
    /// Run the C++ CUDA reference and this implementation back to back and
    /// compare runtime and HSP output.
    Compare(CompareArgs),
    /// Exact per-unit seed-hit estimates before any GPU work (CPU-only).
    HitsEstimate(HitsEstimateArgs),
}

#[derive(Args, Clone)]
pub(crate) struct RunArgs {
    #[arg(short, long)]
    pub(crate) reference: PathBuf,
    #[arg(
        short,
        long,
        required_unless_present = "query_list",
        conflicts_with = "query_list"
    )]
    pub(crate) query: Option<PathBuf>,
    /// One query FASTA path per line (blank lines and `#` comments ignored).
    /// Batch: one reference × many queries, spread over `--gpus` workers by the
    /// same unit partition as single queries (a slot position never becomes a
    /// query-bin id). All-or-nothing: a failing job aborts the batch, so
    /// partial per-job outputs (and a truncated `-Z` archive) may remain on disk.
    #[arg(long, conflicts_with = "query")]
    pub(crate) query_list: Option<PathBuf>,
    /// Directory to write `tmp<n>.block<q>.r<r>.{plus,minus}.segments` into.
    #[arg(short, long, default_value = ".")]
    pub(crate) output: PathBuf,

    /// plus / minus / both
    #[arg(short = 'S', long, default_value = "both")]
    pub(crate) strand: String,
    /// 12of19, 14of22, or an arbitrary pattern of 1s, 0s and Ts.
    #[arg(short, long, default_value = "12of19")]
    pub(crate) seed: String,
    #[arg(short = 'e', long, default_value_t = 1)]
    pub(crate) step: u32,
    /// Don't allow one transition in a seed hit.
    #[arg(short, long)]
    pub(crate) notransition: bool,

    #[arg(short, long, default_value_t = 910)]
    pub(crate) xdrop: i32,
    #[arg(short = 'H', long, default_value_t = 3000)]
    pub(crate) hspthresh: i32,
    /// Don't apply the entropy correction to low-scoring segment pairs.
    #[arg(short = 'E', long)]
    pub(crate) noentropy: bool,
    /// Ambiguous nucleotide handling: `n`, `iupac`, or `<field>,<reward>,<penalty>`.
    #[arg(short, long, default_value = "")]
    pub(crate) ambiguous: String,
    /// Substitution matrix in LASTZ score-set format.
    #[arg(short = 'c', long)]
    pub(crate) scoring: Option<PathBuf>,

    #[arg(short = 'T', long, default_value = "")]
    pub(crate) target_prefix: String,
    #[arg(short = 'Q', long, default_value = "")]
    pub(crate) query_prefix: String,

    #[arg(short = 'C', long, default_value_t = 250_000)]
    pub(crate) wga_chunk_size: u32,
    #[arg(short = 'I', long, default_value_t = 10_000_000)]
    pub(crate) lastz_interval_size: u32,
    /// Target bin size in bases (not a hard cut; chromosomes stay atomic).
    /// Default 500 Mbp is the KegAlign-matched digest. For `--gpus W` wall,
    /// about `total_reference_bp / W` (one ref bin per worker) is faster and
    /// a *different* HSP set. Never overwritten from `--gpus`. See
    /// `assets/guidance/guidance.md`.
    #[arg(short = 'B', long, default_value_t = 500_000_000)]
    pub(crate) seq_block_size: u32,
    /// Bin target for the *query* side; defaults to `--seq-block-size`.
    /// Splitting the reference is what balances workers — splitting the query only
    /// multiplies work units (`R × Q`), and each unit costs a query swap and its own
    /// `MAX_HITS` chunk boundary on every worker. Raise this above `-B` to pick a
    /// reference layout without paying for a query one. Changes the plan, so it
    /// changes the HSP set: keep a digest per layout.
    #[arg(long)]
    pub(crate) query_block_size: Option<u32>,
    /// Semantic hits per GPU chunk (target H); 0 derives KegAlign's `4194304 * GiB` default. Physical capacity C>=H is derived from free VRAM and only affects success/failure, never successful output bytes.
    #[arg(short, long, default_value_t = 0)]
    pub(crate) max_hits: u32,
    /// Worker threads for seed generation; 0 uses available parallelism.
    #[arg(short, long, default_value_t = 0)]
    pub(crate) threads: usize,
    /// N8: `find_hsps` grid blocks; 0 uses the compiled-in optimum (16384).
    #[arg(short = 'b', long, default_value_t = 0)]
    pub(crate) hsp_blocks: u32,
    /// Stage seeds in pageable memory. The default stages them in pinned host
    /// memory, which measured -0.5..-0.8% on an L4 and is what makes the H->D
    /// copy a DMA; backends without `cuMemHostAlloc` fall back automatically.
    #[arg(short = 'P', long)]
    pub(crate) no_pinned_seeds: bool,
    /// Reallocate `d_seeds` / `d_hit_num` per batch. The default reuses them,
    /// worth -3.5% (A) / -3.8% (B) on an L4.
    #[arg(long)]
    pub(crate) no_persistent_seed_buffers: bool,
    /// Bin records the way KegAlign does — sequential fill in input order, closing
    /// a block once it exceeds `--seq-block-size` — instead of the balanced planner.
    /// For matched-granularity benchmarking only: it makes block membership, and so
    /// the dedup scope, identical on both sides. The planner will not shrink the
    /// block size to fit the GPU in this mode; it errors instead, because shrinking
    /// would unmatch the granularity.
    #[arg(long)]
    pub(crate) kegalign_bins: bool,
    /// Write the plan's bin membership to this file and continue: one
    /// `side<TAB>bin<TAB>record_name<TAB>bp` line per record. Matched-granularity
    /// benchmarking compares it against KegAlign's `{ref,query}_block*.name`, which
    /// is the only way to *show* both tools binned the input the same way.
    #[arg(long)]
    pub(crate) dump_plan: Option<PathBuf>,
    /// Write a frozen executable plan (GPU `run` only, same binary required for replay).
    /// A second node given `--from-manifest` will *validate fit and fail*, not replan.
    #[arg(long)]
    pub(crate) dump_manifest: Option<PathBuf>,
    /// Replay a frozen plan from `--dump-manifest` (GPU `run` only). Requires the
    /// same binary, inputs, resolved scoring matrix, strand and record prefixes.
    /// The planner will not shrink bins or the hit cap.
    #[arg(long)]
    pub(crate) from_manifest: Option<PathBuf>,
    /// GPUs to run on. Reference bins are split across that many
    /// workers by deterministic LPT, each owning its bins end to end; output still
    /// follows `WorkUnit.ordinal`, so it does not depend on which GPU finished
    /// first. More workers than devices time-slices one GPU: a correctness
    /// configuration, not a performance one. Batch mode (`--query-list`) uses
    /// the same workers over unit-partitioned `(job, unit)` slots.
    #[arg(short = 'G', long, default_value_t = 1)]
    pub(crate) gpus: usize,
    /// Wait for the GPU after every stage instead of enqueueing the whole
    /// per-batch chain. The default enqueues and waits only where the host needs a
    /// device result; on its own that was neutral (round 29), but together with the
    /// overlapped seed upload it is worth -1.51% on an L4 with disjoint ranges over
    /// six paired rounds (round 30).
    #[arg(long)]
    pub(crate) no_async_stages: bool,
    /// Upload each batch's seeds with a blocking copy at the point of use. The
    /// default uploads on a second stream one batch ahead, so the DMA overlaps the
    /// previous batch's kernels: exposed `H->D seeds` 5,734 -> 72 ms on an L4.
    /// The overlap needs pinned staging and the persistent seed buffers, and turns
    /// itself off when either is missing (ZLUDA has no `cuMemHostAlloc`, and an
    /// async copy from pageable memory blocks anyway).
    #[arg(long)]
    pub(crate) no_async_seed_copy: bool,
    /// Build each reference bin only when its turn comes. The default builds the
    /// next bin's pack + seed table on a worker thread while the current bin's
    /// work units run on the GPU; that build is the largest host cost the GPU
    /// cannot otherwise hide (7.1% of an L4 multi5 run).
    #[arg(long)]
    pub(crate) no_ref_prefetch: bool,

    /// Load per-bin reference tables from an on-disk index written by
    /// `hspZ index` instead of rebuilding them. The run still parses the
    /// reference and plans as usual, then requires the planned reference bins
    /// to equal the index's, or it errors naming the mismatch.
    #[arg(long)]
    pub(crate) index: Option<PathBuf>,

    /// Report the full wall-time accounting.
    #[arg(short = 'y', long)]
    pub(crate) time: bool,
    /// Report the hits-per-seed distribution.
    #[arg(short = 'd', long)]
    pub(crate) hit_stats: bool,
    /// Stop after CPU preprocessing and report seed/hit counts. Needs no GPU.
    #[arg(short = 'u', long)]
    pub(crate) cpu_only: bool,
    /// Write every pre-dedup HSP to this file as
    /// `strand ref_start query_start len score`.
    #[arg(long)]
    pub(crate) dump_raw: Option<PathBuf>,

    /// Split each output file along its diagonal, KegAlign-style, emitting
    /// `*.split1.segments` onward instead of one file. Partitions the HSP
    /// structs directly — no unsplit file is written first.
    #[arg(short = 'D', long)]
    pub(crate) diagonal_partition: bool,
    /// Write output into one `.tar.gz` instead of a directory. Without a path,
    /// `<output>.tar.gz` is used. Archived directly from the formatted bytes.
    /// In batch mode (`--query-list`) one archive per job is written
    /// (`OUT/000001.tar.gz` …) and a custom path value is ignored (a note is
    /// printed).
    #[arg(short = 'Z', long, num_args = 0..=1, default_missing_value = "-", require_equals = false)]
    pub(crate) tarball: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct IndexArgs {
    /// Reference FASTA (or FASTA.gz / 2bit) to index.
    #[arg(short, long)]
    pub(crate) reference: PathBuf,
    /// Directory to write the index into. Required; must not exist yet
    /// (no --force: delete it or pick another path).
    #[arg(long)]
    pub(crate) index: PathBuf,
    /// 12of19, 14of22, or an arbitrary pattern of 1s, 0s and Ts.
    #[arg(short, long, default_value = "12of19")]
    pub(crate) seed: String,
    #[arg(short = 'e', long, default_value_t = 1)]
    pub(crate) step: u32,
    /// Target bin size in bases, as in `run` (`-B 0` rejected: the automatic
    /// layout is device-dependent and cannot be frozen into an index).
    #[arg(short = 'B', long, default_value_t = 500_000_000)]
    pub(crate) seq_block_size: u32,
    /// Bin records the way KegAlign does — sequential fill in input order —
    /// instead of the balanced planner. Must match the `run` side.
    #[arg(long)]
    pub(crate) kegalign_bins: bool,
    /// Worker threads for seed generation; 0 uses available parallelism.
    #[arg(short, long, default_value_t = 0)]
    pub(crate) threads: usize,
    /// Report per-bin build/write timings and payload sizes.
    #[arg(short = 'y', long)]
    pub(crate) time: bool,
    /// Rejected: the index stores unprefixed names; pass --target-prefix on run.
    #[arg(short = 'T', long, hide = true)]
    pub(crate) target_prefix: Option<String>,
}

#[derive(Args)]
pub(crate) struct BenchArgs {
    #[command(flatten)]
    pub(crate) run: RunArgs,
    /// Timed Seed + Filter iterations after the warmup.
    #[arg(short, long, default_value_t = 20)]
    pub(crate) iterations: u32,
    /// Untimed iterations first.
    #[arg(short, long, default_value_t = 3)]
    pub(crate) warmup: u32,
    /// Launches used to price ZLUDA's per-launch overhead.
    #[arg(short = 'l', long, default_value_t = 200)]
    pub(crate) launch_probe: u32,

    /// Append one JSON record per warm iteration to this file.
    #[arg(short = 'j', long)]
    pub(crate) json: Option<PathBuf>,
    /// Variant label recorded in each JSON record (e.g. `nvidia-segment-align16`).
    #[arg(short = 'v', long, default_value = "baseline")]
    pub(crate) variant: String,
    /// Workload label recorded in each JSON record (e.g. `A`).
    #[arg(short = 'k', long, default_value = "unnamed")]
    pub(crate) workload: String,
    /// A JSON object merged into each record's `environment`. The runner fills
    /// this with the things a shell knows and this binary should not shell out
    /// for: git commit, rustc/cargo versions, GPU name, Slurm ids.
    #[arg(short = 'f', long, default_value = "{}")]
    pub(crate) env_json: String,
}

#[derive(Args)]
pub(crate) struct CompareArgs {
    #[arg(short, long)]
    pub(crate) reference: PathBuf,
    #[arg(short, long)]
    pub(crate) query: PathBuf,
    /// The C++ CUDA oracle.
    #[arg(short = 'k', long, default_value = "/tmp/kegalign/build/kegalign")]
    pub(crate) kegalign: PathBuf,
    /// Legacy-stream shim the Thrust/CUB reference needs under ZLUDA.
    #[arg(
        short = 'L',
        long,
        default_value = "/home/alejandro/opt/zluda-guide/hipfix.so"
    )]
    pub(crate) ld_preload: String,
    #[arg(
        short = 'l',
        long,
        default_value = "/home/alejandro/opt/zluda:/home/alejandro/opt/cudaconda/lib"
    )]
    pub(crate) ld_library_path: String,
    /// Where to keep both runs' output. Defaults to a fresh temp directory.
    #[arg(short = 'w', long)]
    pub(crate) workdir: Option<PathBuf>,
    #[command(flatten)]
    pub(crate) tuning: Tuning,
}

/// CPU-only exact seed-hit estimator (R92 PR1): same planning inputs as
/// `run` (bins/blocks identical to what `run` would freeze), no CUDA.
#[derive(Args, Clone)]
pub(crate) struct HitsEstimateArgs {
    #[arg(short, long)]
    pub(crate) reference: PathBuf,
    #[arg(short, long)]
    pub(crate) query: PathBuf,
    /// Target bin size in bases, as in `run` (`-B 0` resolves to the default).
    #[arg(short = 'B', long, default_value_t = 500_000_000)]
    pub(crate) seq_block_size: u32,
    /// Bin target for the query side; defaults to `--seq-block-size`.
    #[arg(long)]
    pub(crate) query_block_size: Option<u32>,
    /// Bin records the way KegAlign does (sequential fill, not LPT).
    #[arg(long)]
    pub(crate) kegalign_bins: bool,
    /// 12of19, 14of22, or an arbitrary pattern of 1s, 0s and Ts (k<=12).
    #[arg(long, default_value = "12of19")]
    pub(crate) seed: String,
    #[arg(short = 'e', long, default_value_t = 1)]
    pub(crate) step: u32,
    /// Don't allow one transition in a seed hit.
    #[arg(long)]
    pub(crate) notransition: bool,
    /// plus / minus / both
    #[arg(short = 'S', long, default_value = "both")]
    pub(crate) strand: String,
    /// Worker threads for histogram passes; 0 uses available parallelism.
    #[arg(short, long, default_value_t = 0)]
    pub(crate) threads: usize,
    /// LASTZ interval size, as in `run` (`-I`): interval boundaries decide
    /// which shared window starts the pipeline seeds once and which twice.
    #[arg(short = 'I', long, default_value_t = 10_000_000)]
    pub(crate) lastz_interval_size: u32,
    /// Query seed chunk size, as in `run` (`-C`): the interval schedule
    /// (`seed::chunks`) is part of the count, so this must match `run`.
    #[arg(short = 'C', long, default_value_t = 250_000)]
    pub(crate) wga_chunk_size: u32,
    /// Count every S-th query window start and scale by S (approximation).
    #[arg(long, default_value_t = 1)]
    pub(crate) stride: u32,
    /// Write the plan's bin membership to this file and continue.
    #[arg(long)]
    pub(crate) dump_plan: Option<PathBuf>,
}

#[derive(Args, Clone)]
pub(crate) struct Tuning {
    #[arg(short, long, default_value = "12of19")]
    pub(crate) seed: String,
    #[arg(short, long, default_value_t = 910)]
    pub(crate) xdrop: i32,
    #[arg(short = 'H', long, default_value_t = 3000)]
    pub(crate) hspthresh: i32,
    #[arg(short = 'c', long)]
    pub(crate) scoring: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The short flags must parse without collisions — clap errors at parse
    /// time if two args in one subcommand share a letter, so this exercises
    /// every grouped flag set at least once.
    #[test]
    fn short_flags_parse() {
        match Cli::try_parse_from(["hspz", "run", "-r", "r.fa", "-q", "q.fa", "-o", "out"])
            .unwrap()
            .command
        {
            Command::Run(args) => {
                assert_eq!(args.reference, PathBuf::from("r.fa"));
                assert_eq!(args.query, Some(PathBuf::from("q.fa")));
                assert_eq!(args.output, PathBuf::from("out"));
            }
            _ => panic!("wrong subcommand"),
        }
        match Cli::try_parse_from(["hspz", "benchmark", "-r", "r.fa", "-q", "q.fa", "-i", "5"])
            .unwrap()
            .command
        {
            Command::Benchmark(args) => assert_eq!(args.iterations, 5),
            _ => panic!("wrong subcommand"),
        }
        match Cli::try_parse_from([
            "hspZ", "compare", "-r", "r.fa", "-q", "q.fa", "-k", "keg", "-w", "wd",
        ])
        .unwrap()
        .command
        {
            Command::Compare(args) => {
                assert_eq!(args.kegalign, PathBuf::from("keg"));
                assert_eq!(args.workdir, Some(PathBuf::from("wd")));
            }
            _ => panic!("wrong subcommand"),
        }
        // Bare `-Z` must parse (optional value, clap's `-` sentinel), and
        // `-Z <path>` must still take the path.
        match Cli::try_parse_from(["hspz", "run", "-r", "r.fa", "-q", "q.fa", "-Z", "o.tgz"])
            .unwrap()
            .command
        {
            Command::Run(args) => assert_eq!(args.tarball, Some(PathBuf::from("o.tgz"))),
            _ => panic!("wrong subcommand"),
        }
        // `--query-list` is mutually exclusive with `-q/--query`, and `-q`
        // becomes optional so one of the two must be present.
        assert!(
            Cli::try_parse_from(["hspz", "run", "-r", "r.fa", "-o", "out"]).is_err(),
            "one of -q/--query-list is required"
        );
        match Cli::try_parse_from([
            "hspz",
            "run",
            "-r",
            "r.fa",
            "--query-list",
            "q.txt",
            "-o",
            "out",
        ])
        .unwrap()
        .command
        {
            Command::Run(args) => {
                assert_eq!(args.query, None);
                assert_eq!(args.query_list, Some(PathBuf::from("q.txt")));
            }
            _ => panic!("wrong subcommand"),
        }
        assert!(
            Cli::try_parse_from([
                "hspz",
                "run",
                "-r",
                "r.fa",
                "-q",
                "q.fa",
                "--query-list",
                "q.txt",
            ])
            .is_err(),
            "-q and --query-list must conflict"
        );
    }
}
