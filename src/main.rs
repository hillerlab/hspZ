// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! hspZ — KegAlign's GPU Seed + Filter stage, reimplemented in Rust on
//! cuda-oxide.
//!
//! # What hspZ is
//!
//! A standalone, drop-in replacement for the HSP-finding half of KegAlign
//! (C++/CUDA). It seeds spaced k-mers, extends seed hits with X-drop, applies
//! the entropy-aware score gate, and emits `.segments` files under KegAlign's
//! own naming scheme (`tmp<n>.block<q>.r<r>.{plus,minus}.segments`). At a
//! matched plan and `MAX_HITS` the output is byte-identical to the C++ oracle;
//! LASTZ, gapped extension, chaining and AXT/PSL remain out of scope.
//!
//! The port grew well beyond the original single-block stage:
//!
//! * **Inputs** — FASTA, FASTA.gz and 2bit, detected by magic bytes, all
//!   decoding to the same packed representation.
//! * **Planning** — whole chromosomes are binned by the planner (`plan.rs`);
//!   reference and query block targets are independent (`-B` /
//!   `--query-block-size`), and `--kegalign-bins` reproduces KegAlign's
//!   sequential fill for matched-granularity benchmarking.
//! * **Execution** — one or more GPU workers (`--gpus W`), reference bins
//!   assigned by deterministic LPT, output replayed in ordinal order so it
//!   never depends on which GPU finished first. Above one worker the seeds are
//!   generated on the device, and the seed upload runs on a second stream
//!   overlapped with the previous batch's compute.
//! * **Output** — in-memory diagonal partitioning (`-D`, a port of
//!   `diagonal_partition.py` with exact counts instead of byte estimates) and
//!   reproducible `.tar.gz` archives (`-Z`, mtime pinned); both are
//!   byte-identical 1-GPU vs N-GPU.
//!
//! # CLI
//!
//! ```text
//! hspZ run        -r REF -q QRY [-o OUT] [flags]    seed + filter + emit
//! hspZ benchmark  -r REF -q QRY [run flags] [-i N]  timed cold/warm oracle
//! hspZ compare    -r REF -q QRY [-k KEGALIGN]       vs the C++ oracle
//! ```
//!
//! Notable `run` flags and defaults (the full surface is `cli.rs`): `-B`
//! block target 500 Mbp (the KegAlign-matched digest layout),
//! `--query-block-size` (defaults to `-B`), `--max-hits` derived as
//! `4194304 * device GiB`, `-G/--gpus 1`, `-S` strand, `-x` X-drop 910,
//! `-H` HSP threshold 3000, `-D`, `-Z`, `-y/--time` (wall-time accounting with
//! CUDA events), `-u/--cpu-only`, `--threads 0` = available parallelism, split
//! across workers when `--gpus > 1`. `HSPZ_DEVICE_SEEDS` forces the seeder so
//! 1-GPU vs N-GPU wall is a matched scaling number.
//!
//! # Quick benchmark data
//!
//! * **Parity** — byte-identical to the C++ oracle wherever plan and
//!   `MAX_HITS` match: apple/orange 1,226 HSPs, A 4,248, B 8,908, chr1
//!   155,195, multi5 726,559, whole hg38 x mm39 ~15.16 M HSPs.
//! * **vs KegAlign** — 2.13–2.23x faster HSP generation on one NVIDIA L4
//!   (whole-genome pipeline 37.3 s vs 17.5 s), ~2.2x end to end.
//! * **Multi-GPU** — whole hg38 x mm39 in 11.9 min on 4x RTX 4090
//!   (`-B 400 Mbp`, R=8); matched-seeder scaling 3.587x of 4 =
//!   89.7% efficiency.
//! * **Single-GPU native** — L4 A 447 ms / B 386 ms (6.8x over the original
//!   ZLUDA baseline); score gate at 1.23 ns/hit.
//! * **Exactness caveat** — block layout and `--max-hits` set the dedup
//!   scope, so different layouts produce different (not wrong) HSP sets;
//!   keep a frozen digest per layout.

mod benchmark;
mod census;
mod cli;
mod compare;
mod gpu;
mod hsp;
mod partition;
mod plan;
mod run;
mod scoring;
mod seed;
mod sequence;
mod sink;
mod timing;

use clap::Parser;
use cli::{Cli, Command};
use std::time::Instant;

/// The error type every command returns: anything boxed, printed as `error: {e}`.
pub(crate) type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    let started = Instant::now();
    let pre_main_ms = timing::since_process_start_ms().unwrap_or(0.0);
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Run(args) => run::run(&args, pre_main_ms, started).map(|_| ()),
        Command::Benchmark(args) => benchmark::benchmark(&args, pre_main_ms),
        Command::Compare(args) => compare::compare(&args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
