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
    version = env!("CARGO_PKG_VERSION"),
    about = "GPU-accelerated high-scoring ungapped alignment pair backend",
    author = env!("CARGO_PKG_AUTHORS")
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// The three subcommands hspZ accepts.
#[derive(Subcommand)]
pub(crate) enum Command {
    /// Seed, extend and filter one reference/query pair.
    Run(RunArgs),
    /// Cold/warm benchmark: timed runs against wall clock.
    Benchmark(BenchArgs),
    /// Run the C++ CUDA reference and this implementation back to back and
    /// compare runtime and HSP output.
    Compare(CompareArgs),
}

#[derive(Args, Clone)]
pub(crate) struct RunArgs {
    #[arg(short, long)]
    pub(crate) reference: PathBuf,
    #[arg(short, long)]
    pub(crate) query: PathBuf,
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
    /// Hits per GPU chunk; 0 derives KegAlign's `4194304 * GiB` default.
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
    /// GPUs to run on. Reference bins are split across that many
    /// workers by deterministic LPT, each owning its bins end to end; output still
    /// follows `WorkUnit.ordinal`, so it does not depend on which GPU finished
    /// first. More workers than devices time-slices one GPU: a correctness
    /// configuration, not a performance one.
    #[arg(short = 'G', long, default_value_t = 1)]
    pub(crate) gpus: usize,
    /// Wait for the GPU after every stage instead of enqueueing the whole
    /// per-batch chain. The default enqueues and waits only where the host needs a
    /// device result; on its own that was neutral, but together with the
    /// overlapped seed upload it is worth -1.51% on an L4 with disjoint ranges.
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
    #[arg(short = 'Z', long, num_args = 0..=1, default_missing_value = "-", require_equals = false)]
    pub(crate) tarball: Option<PathBuf>,
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
    /// `LD_PRELOAD` for the oracle process. Empty (the default) inherits this
    /// process's environment. Running the Thrust/CUB reference under ZLUDA needs a
    /// legacy-stream shim here, so a ZLUDA host must pass its own path.
    #[arg(short = 'L', long, default_value = "")]
    pub(crate) ld_preload: String,
    /// `LD_LIBRARY_PATH` for the oracle process. Empty (the default) inherits this
    /// process's environment rather than clearing it.
    #[arg(short = 'l', long, default_value = "")]
    pub(crate) ld_library_path: String,
    /// Where to keep both runs' output. Defaults to a fresh temp directory.
    #[arg(short = 'w', long)]
    pub(crate) workdir: Option<PathBuf>,
    #[command(flatten)]
    pub(crate) tuning: Tuning,
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
        match Cli::try_parse_from(["hspZ", "run", "-r", "r.fa", "-q", "q.fa", "-o", "out"])
            .unwrap()
            .command
        {
            Command::Run(args) => {
                assert_eq!(args.reference, PathBuf::from("r.fa"));
                assert_eq!(args.query, PathBuf::from("q.fa"));
                assert_eq!(args.output, PathBuf::from("out"));
            }
            _ => panic!("wrong subcommand"),
        }
        match Cli::try_parse_from(["hspZ", "benchmark", "-r", "r.fa", "-q", "q.fa", "-i", "5"])
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
        match Cli::try_parse_from(["hspZ", "run", "-r", "r.fa", "-q", "q.fa", "-Z", "o.tgz"])
            .unwrap()
            .command
        {
            Command::Run(args) => assert_eq!(args.tarball, Some(PathBuf::from("o.tgz"))),
            _ => panic!("wrong subcommand"),
        }
    }
}
