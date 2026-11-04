// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! `run --paired`: record i of `-r` against record i of `-q` only.
//!
//! Built for chain gap filling, where every gap is one small, independent
//! (target region, query region) pair and an all-vs-all over millions of
//! unmasked regions would be dominated by cross-pair repeat hits.
//!
//! **Exactness claim:** the lines naming pair i are byte-identical to what a
//! standalone `run -S plus` of pair i's two records writes. Batching many
//! pairs into one engine keeps that because:
//!
//! * pairs are packed in order and each seed only walks the bucket slice of its
//!   own pair (`find_num_hits_paired` / `find_hits_dense_paired`), which is the
//!   same positions, in the same order, as the standalone pair's whole bucket;
//! * a standalone table never indexes its buffer's position 0, so every target
//!   record's first base is masked for the table build ([`mask_starts`]);
//! * a query record exactly one seed span long seeds nothing standalone (no
//!   interval), so its first base is masked too;
//! * each record is exactly one standalone segment (one interval, one chunk),
//!   checked up front ([`validate_pairs`]), and a call is never split by
//!   `MAX_HITS` unless it holds a single pair: an over-cap call with several
//!   pairs stops before `find_hits` ([`crate::gpu::FilterOutput::oversized`])
//!   and is re-issued one pair per call — exactly the standalone call;
//! * `&` separators score -10·xdrop, so x-drop stops at record edges just as it
//!   does at a standalone buffer edge, and dedup cannot join pairs (disjoint
//!   reference ranges).
//!
//! `assets/tests/paired-identity.sh` is the oracle for this claim.

use crate::Fallible;
use crate::cli::RunArgs;
use crate::gpu::{Engine, EngineConfig, ExecutionContract};
use crate::hsp::{self, SegmentPair};
use crate::plan::{self, PackedBin};
use crate::run::{Stats, report_counts, resolve_threads};
use crate::scoring;
use crate::seed::{self, SeedTable, Shape};
use crate::sequence::{self, Chr, N_NT};
use crate::timing::{self, Phases};
use cuda_core::CudaContext;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::time::Instant;

/// Rejects every flag `--paired` does not implement, before any input is read.
pub(crate) fn validate_args(args: &RunArgs) -> Result<(), String> {
    if !cfg!(feature = "dense-anchors") {
        return Err("--paired needs a build with the dense-anchors feature".into());
    }
    if args.strand != "plus" {
        return Err(format!(
            "--paired compares plus strands only: pass -S plus (got -S {}); \
             reverse-complement minus-strand query records before the run",
            args.strand
        ));
    }
    if args.step != 1 {
        return Err(format!(
            "--paired needs --step 1 (got {}): strided seeds depend on the packed offset",
            args.step
        ));
    }
    if args.xdrop <= 0 {
        return Err(format!(
            "--paired needs --xdrop > 0 (got {}): the record separator scores -10·xdrop, \
             which only stops extension at record edges when xdrop is positive",
            args.xdrop
        ));
    }
    if args.reference.is_none() || args.query.is_none() {
        return Err("--paired needs both -r and -q".into());
    }
    let rejected: [(bool, &str); 14] = [
        (args.query_list.is_some(), "--query-list"),
        (args.index.is_some(), "--index"),
        (args.from_manifest.is_some(), "--from-manifest"),
        (args.dump_manifest.is_some(), "--dump-manifest"),
        (args.dump_plan.is_some(), "--dump-plan"),
        (args.kegalign_bins, "--kegalign-bins"),
        (args.query_block_size.is_some(), "--query-block-size"),
        (args.seq_block_size == 0, "-B 0"),
        (args.diagonal_partition, "-D"),
        (args.tarball.is_some(), "-Z"),
        (args.cpu_only, "--cpu-only"),
        (args.dump_raw.is_some(), "--dump-raw"),
        (args.hit_stats, "--hit-stats"),
        (args.gpus > 1, "--gpus > 1"),
    ];
    match rejected.iter().find(|(set, _)| *set) {
        Some((_, flag)) => Err(format!("{flag} is not supported with --paired")),
        None => Ok(()),
    }
}

/// Checks that the two record lists pair up (same count, names and order;
/// unique names) and that every query record is exactly one standalone segment:
/// one LASTZ interval and one seed chunk, so the batched chunk plan never splits
/// a pair where a standalone run would not.
pub(crate) fn validate_pairs(
    targets: &[(String, Vec<u8>)],
    queries: &[(String, Vec<u8>)],
    span: usize,
    wga_chunk_size: u32,
    lastz_interval_size: u32,
) -> Result<(), String> {
    if targets.len() != queries.len() {
        return Err(format!(
            "--paired: -r has {} records but -q has {}",
            targets.len(),
            queries.len()
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(targets.len());
    for (i, ((t, _), (q, qs))) in targets.iter().zip(queries).enumerate() {
        if t != q {
            return Err(format!(
                "--paired: record {} is '{t}' in -r but '{q}' in -q (names must match in order)",
                i + 1
            ));
        }
        if !seen.insert(t.as_str()) {
            return Err(format!(
                "--paired: record name '{t}' appears more than once"
            ));
        }
        // One interval, seeded by one chunk that reaches its last window: at
        // `len - span == C` standalone makes one chunk but drops that window
        // (`seed::chunks` stops at `i < end`), which a merged range would seed.
        let intervals = sequence::intervals(qs.len(), span, lastz_interval_size);
        let one_segment = match intervals.as_slice() {
            [] => true,
            [(start, end)] => seed::chunks(*start, *end, wga_chunk_size) == [(*start, *end + 1)],
            _ => false,
        };
        if !one_segment {
            return Err(format!(
                "--paired: query record '{q}' ({} bp) spans more than one seed chunk \
                 (-C {wga_chunk_size}) or LASTZ interval (-I {lastz_interval_size}); \
                 each pair must be one standalone segment",
                qs.len()
            ));
        }
    }
    Ok(())
}

/// Consecutive pair ranges. `fits(ref_packed, query_packed, pairs)` decides
/// whether a batch of that size fits; a single pair that does not fit still
/// gets a batch of its own.
pub(crate) fn plan_batches(
    lens: &[(u64, u64)],
    fits: impl Fn(u64, u64, usize) -> bool,
) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut r, mut q) = (0usize, 0u64, 0u64);
    for (i, &(t_len, q_len)) in lens.iter().enumerate() {
        // Packed lengths: every record is followed by one `&`.
        let (nr, nq) = (r + t_len + 1, q + q_len + 1);
        if i > start && !fits(nr, nq, i + 1 - start) {
            out.push(start..i);
            start = i;
            r = t_len + 1;
            q = q_len + 1;
        } else {
            r = nr;
            q = nq;
        }
    }
    if start < lens.len() {
        out.push(start..lens.len());
    }
    out
}

/// One `seed_and_filter` call: the half-open seed range and the pairs (indices
/// into the batch) whose records it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub(crate) range: (u32, u32),
    pub(crate) pairs: Range<usize>,
}

/// Packs whole query records into calls of at most `wga_chunk_size` seed
/// positions. A record of `len > span` seeds `[start, start + len - span]`,
/// exactly its standalone range; shorter records seed nothing. One contiguous
/// range per call is enough because windows crossing `&` are invalid k-mers.
pub(crate) fn plan_chunks(q_chrs: &[Chr], span: usize, wga_chunk_size: u32) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut cur: Option<Chunk> = None;
    for (k, c) in q_chrs.iter().enumerate() {
        let len = c.len as usize;
        if len <= span {
            continue;
        }
        let (s, e) = (c.start as u32, (c.start + len - span) as u32);
        cur = match cur.take() {
            Some(mut ch)
                if u64::from(e) + 1 - u64::from(ch.range.0) <= u64::from(wga_chunk_size) =>
            {
                ch.range.1 = e + 1;
                ch.pairs.end = k + 1;
                Some(ch)
            }
            prev => {
                out.extend(prev);
                Some(Chunk {
                    range: (s, e + 1),
                    pairs: k..k + 1,
                })
            }
        };
    }
    out.extend(cur);
    out
}

/// Sets the first base of every record `pick` selects to `N`, so no seed window
/// starts there (any other window containing it already crosses `&`).
fn mask_starts(buf: &mut [u8], chrs: &[Chr], pick: impl Fn(&Chr) -> bool) {
    for c in chrs.iter().filter(|c| c.len > 0 && pick(c)) {
        buf[c.start] = b'N';
    }
}

/// Device bytes one batch needs: the planner's per-unit model plus the paired
/// bounds (12 B per pair) and the per-seed slice start (4 B per seed).
#[allow(clippy::too_many_arguments)]
fn batch_bytes(
    r: u64,
    q: u64,
    pairs: usize,
    shape: &Shape,
    max_hits: u32,
    wga_chunk_size: u32,
    transitions: bool,
) -> Result<u64, String> {
    let unit = plan::unit_device_bytes(
        r,
        q,
        shape.kmer_size,
        1,
        max_hits,
        wga_chunk_size,
        transitions,
    )?;
    let max_seeds = seed::max_seeds(wga_chunk_size, shape, transitions) as u64;
    Ok(unit + 12 * pairs as u64 + 4 * max_seeds)
}

/// One call for the `--time` report.
struct Call {
    hits: u32,
    batch: usize,
    pairs: Range<usize>,
    reissued: bool,
}

pub(crate) fn run(args: &RunArgs, pre_main_ms: f64, started: Instant) -> Fallible<Stats> {
    validate_args(args)?;
    let mut phases = Phases::new();
    phases.add_ms("process startup", pre_main_ms);

    let shape = Shape::parse(&args.seed)?;
    let span = shape.size;
    let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, args.scoring.as_deref())?;
    let transitions = !args.notransition;
    let threads = resolve_threads(args.threads);

    let t = Instant::now();
    let (_, targets, _) = sequence::read_records(args.reference.as_ref().expect("validated"))?;
    phases.add("input.reference", t.elapsed());
    let t = Instant::now();
    let (_, queries, _) = sequence::read_records(args.query.as_ref().expect("validated"))?;
    phases.add("input.query", t.elapsed());
    validate_pairs(
        &targets,
        &queries,
        span,
        args.wga_chunk_size,
        args.lastz_interval_size,
    )?;

    let t = Instant::now();
    let ctx = CudaContext::new(0)?;
    phases.add("CUDA context init", t.elapsed());
    let mut contract = ExecutionContract::resolve(&ctx, args.max_hits, args.hsp_blocks);
    let budget = plan::worker_device_budget(
        crate::gpu::min_free_bytes(1)?,
        1,
        crate::gpu::device_count().max(1),
    );
    // Size at 9/8·H: the walk's final sub-chunk can overshoot H, and a
    // standalone pair (tiny reference) always has that headroom.
    let sizing_hits = u32::try_from(u64::from(contract.max_hits) * 9 / 8).unwrap_or(u32::MAX);
    // Packed lengths address with u32; `-B` caps a batch like a bin target.
    let cap = u64::from(args.seq_block_size).min(u64::from(u32::MAX) - 1);
    let lens: Vec<(u64, u64)> = targets
        .iter()
        .zip(&queries)
        .map(|((_, t), (_, q))| (t.len() as u64, q.len() as u64))
        .collect();
    let fits = |r: u64, q: u64, n: usize| {
        r <= cap
            && q <= cap
            && batch_bytes(
                r,
                q,
                n,
                &shape,
                sizing_hits,
                args.wga_chunk_size,
                transitions,
            )
            .is_ok_and(|b| b <= budget)
    };
    let batches = plan_batches(&lens, fits);

    std::fs::create_dir_all(&args.output)?;
    let final_path = args.output.join("paired.segments");
    let tmp_path = args.output.join("paired.segments.tmp");
    let mut out = BufWriter::new(std::fs::File::create(&tmp_path)?);

    let mut stats = Stats::default();
    let mut calls: Vec<Call> = Vec::new();
    for (b, range) in batches.iter().enumerate() {
        let t = Instant::now();
        let mut packed_ref = PackedBin::build(
            targets[range.clone()]
                .iter()
                .map(|(n, s)| (n.as_str(), s.as_slice())),
            &args.target_prefix,
            false,
        );
        // `enc` is already built from the unmasked bases, so scoring is untouched.
        mask_starts(&mut packed_ref.buf, &packed_ref.chrs, |_| true);
        let table =
            SeedTable::build_parallel(&packed_ref.buf[..packed_ref.block_len], &shape, 1, threads);
        let mut packed_q = PackedBin::build(
            queries[range.clone()]
                .iter()
                .map(|(n, s)| (n.as_str(), s.as_slice())),
            &args.query_prefix,
            false,
        );
        mask_starts(&mut packed_q.buf, &packed_q.chrs, |c| {
            c.len as usize == span
        });
        for c in packed_q.chrs.iter().filter(|c| c.len as usize == span) {
            packed_q.enc[c.start] = N_NT;
        }
        phases.add("paired pack + seed table", t.elapsed());

        // Physical capacity for this batch: the largest H' whose model fits.
        let (r, q, n) = (
            packed_ref.buf.len() as u64,
            packed_q.buf.len() as u64,
            range.len(),
        );
        let model = |h: u32| batch_bytes(r, q, n, &shape, h, args.wga_chunk_size, transitions);
        let (mut lo, mut hi) = (contract.max_hits, u32::MAX);
        if model(lo).is_ok_and(|v| v <= budget) {
            while lo < hi {
                let mid = (u64::from(lo) + (u64::from(hi) - u64::from(lo)).div_ceil(2)) as u32;
                if model(mid).is_ok_and(|v| v <= budget) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
        }
        contract.hit_capacity =
            crate::gpu::clamp_hit_capacity(contract.max_hits, lo, contract.hsp_blocks)?;

        let cfg = EngineConfig {
            index_table: &table.index_table,
            pos_table: &table.pos_table,
            ref_seq: &packed_ref.enc,
            sub_mat: &sub_mat,
            seed_size: span as u32,
            xdrop: args.xdrop,
            hspthresh: args.hspthresh,
            noentropy: args.noentropy,
            max_hits: contract.max_hits,
            hit_capacity: contract.hit_capacity,
            timing: args.time,
            hsp_blocks: contract.hsp_blocks,
        };
        let mut engine = Engine::new(&ctx, cfg, &mut phases)?;
        engine.persistent_seed_buffers = !args.no_persistent_seed_buffers;
        engine.async_stages = !args.no_async_stages;
        // One slot, blocking uploads, so a re-issue never races an
        // upload in flight.
        engine.async_seed_copy = false;
        let q_start: Vec<u32> = packed_q.chrs.iter().map(|c| c.start as u32).collect();
        let t_lo: Vec<u32> = packed_ref.chrs.iter().map(|c| c.start as u32).collect();
        let t_hi: Vec<u32> = packed_ref
            .chrs
            .iter()
            .map(|c| (c.start + c.len as usize) as u32)
            .collect();
        engine.set_pairs(&q_start, &t_lo, &t_hi)?;
        engine.swap_query(&packed_q.enc, &[])?;
        let ref_chrs = std::mem::take(&mut packed_ref.chrs);
        drop(packed_ref);
        drop(table);

        let qbuf = &packed_q.buf[..packed_q.block_len];
        let pair_of = |pos: u32| q_start.partition_point(|&s| s <= pos) - 1;
        let mut hsps: Vec<SegmentPair> = Vec::new();
        for chunk in plan_chunks(&packed_q.chrs, span, args.wga_chunk_size) {
            let t = Instant::now();
            let parts = seed::chunk_seeds_parts(qbuf, &shape, transitions, chunk.range, threads);
            let mut seeds = vec![0u64; parts.iter().map(Vec::len).sum()];
            seed::concat_parts(&parts, &mut seeds);
            phases.add("seed generation", t.elapsed());
            let (Some(&first), Some(&last)) = (seeds.first(), seeds.last()) else {
                continue;
            };
            stats.seeds += seeds.len() as u64;
            engine.upload_seeds(0, &seeds)?;
            engine.paired_multi = pair_of(first as u32) != pair_of(last as u32);
            let o = engine.seed_and_filter(0, false)?;
            if !o.oversized {
                stats.seed_hits += u64::from(o.num_hits);
                stats.raw_hsps += u64::from(o.raw_hsps);
                hsps.extend_from_slice(&o.hsps);
                calls.push(Call {
                    hits: o.num_hits,
                    batch: b,
                    pairs: chunk.pairs,
                    reissued: false,
                });
                continue;
            }
            // Over the cap with several pairs: one call per pair, which is the
            // standalone call. A pair's seeds are a contiguous sub-slice
            // (seeds ascend by query position).
            for p in chunk.pairs {
                let lo = q_start[p];
                let hi = lo + packed_q.chrs[p].len;
                let a = seeds.partition_point(|&s| (s as u32) < lo);
                let z = seeds.partition_point(|&s| (s as u32) < hi);
                if a == z {
                    continue;
                }
                engine.upload_seeds(0, &seeds[a..z])?;
                engine.paired_multi = false;
                let o = engine.seed_and_filter(0, false)?;
                stats.seed_hits += u64::from(o.num_hits);
                stats.raw_hsps += u64::from(o.raw_hsps);
                hsps.extend_from_slice(&o.hsps);
                calls.push(Call {
                    hits: o.num_hits,
                    batch: b,
                    pairs: p..p + 1,
                    reissued: true,
                });
            }
        }

        let t = Instant::now();
        stats.hsps += hsps.len() as u64;
        let recs = hsp::records(&hsps, &ref_chrs, &packed_q.chrs, '+');
        out.write_all(hsp::render_records(&recs, &ref_chrs, &packed_q.chrs, '+').as_bytes())?;
        phases.add("render + write", t.elapsed());
        phases.merge(&engine.phases);
    }
    out.flush()?;
    drop(out);
    std::fs::rename(&tmp_path, &final_path)?;

    report_counts(&stats);
    eprintln!(
        "paired: {} pairs in {} batches, {} calls ({} re-issued one pair per call)",
        lens.len(),
        batches.len(),
        calls.len(),
        calls.iter().filter(|c| c.reissued).count()
    );
    if args.time {
        // Per-pair hit counts exist only for one-pair calls; a merged
        // call reports its whole pair range.
        calls.sort_by_key(|c| std::cmp::Reverse(c.hits));
        eprintln!("paired: top calls by seed hits (batch, pairs, hits)");
        for c in calls.iter().take(10) {
            let names = |p: usize| queries[batches[c.batch].start + p].0.as_str();
            eprintln!(
                "  batch {} pairs {}..={} ({}..={}) hits {}{}",
                c.batch,
                c.pairs.start,
                c.pairs.end - 1,
                names(c.pairs.start),
                names(c.pairs.end - 1),
                c.hits,
                if c.reissued { " [re-issued]" } else { "" }
            );
        }
        let wall = pre_main_ms + started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("\nWALL-TIME ACCOUNTING\n{}", phases.report(wall));
        eprintln!("  peak RSS: {:>10} KiB", timing::peak_rss_kib());
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    fn rec(name: &str, seq: &str) -> (String, Vec<u8>) {
        (name.to_string(), seq.as_bytes().to_vec())
    }

    fn chrs(lens: &[usize]) -> Vec<Chr> {
        let recs: Vec<(String, Vec<u8>)> = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| rec(&format!("g{i}"), &"A".repeat(l)))
            .collect();
        sequence::pack(recs.iter().map(|(n, s)| (n.as_str(), s.as_slice())), "").1
    }

    fn paired_args(extra: &[&str]) -> Result<(), String> {
        let mut argv = vec![
            "hspz", "run", "-r", "t.fa", "-q", "q.fa", "--paired", "-S", "plus",
        ];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Run(a) => validate_args(&a),
            _ => unreachable!(),
        }
    }

    #[test]
    fn flags_paired_does_not_implement_are_rejected() {
        assert!(paired_args(&[]).is_ok());
        for bad in [
            &["-D"][..],
            &["-Z"],
            &["--cpu-only"],
            &["--hit-stats"],
            &["--dump-raw", "x"],
            &["--gpus", "2"],
            &["-B", "0"],
            &["--step", "2"],
            &["--xdrop", "0"],
            &["--kegalign-bins"],
            &["--dump-plan", "x"],
            &["--query-block-size", "10"],
        ] {
            assert!(paired_args(bad).is_err(), "{bad:?} must be rejected");
        }
        let both = Cli::try_parse_from(["hspz", "run", "-r", "t", "-q", "q", "--paired"]).unwrap();
        let Command::Run(a) = both.command else {
            unreachable!()
        };
        assert!(validate_args(&a).unwrap_err().contains("-S plus"));
    }

    #[test]
    fn pairs_must_match_by_count_name_order_and_be_unique() {
        let ok = [rec("a", "ACGT"), rec("b", "ACGT")];
        assert!(validate_pairs(&ok, &ok, 19, 250_000, 10_000_000).is_ok());
        assert!(validate_pairs(&ok, &ok[..1], 19, 250_000, 10_000_000).is_err());
        let swapped = [rec("b", "ACGT"), rec("a", "ACGT")];
        assert!(
            validate_pairs(&ok, &swapped, 19, 250_000, 10_000_000)
                .unwrap_err()
                .contains("record 1")
        );
        let dup = [rec("a", "ACGT"), rec("a", "ACGT")];
        assert!(
            validate_pairs(&dup, &dup, 19, 250_000, 10_000_000)
                .unwrap_err()
                .contains("more than once")
        );
    }

    #[test]
    fn a_pair_must_be_one_standalone_segment() {
        // len - span == C - 1 is one chunk over every window; at len - span ==
        // C standalone still makes one chunk but drops the last window.
        let (span, c) = (19usize, 100u32);
        let one = [rec("a", &"A".repeat(span + c as usize - 1))];
        assert!(validate_pairs(&one, &one, span, c, 10_000_000).is_ok());
        let two = [rec("a", &"A".repeat(span + c as usize))];
        assert!(validate_pairs(&two, &two, span, c, 10_000_000).is_err());
        let interval = [rec("a", &"A".repeat(span + 51))];
        assert!(validate_pairs(&interval, &interval, span, 1_000, 50).is_err());
        let short = [rec("a", ""), rec("b", &"A".repeat(span))];
        assert!(validate_pairs(&short, &short, span, c, 10_000_000).is_ok());
    }

    #[test]
    fn batches_are_greedy_and_an_oversized_pair_gets_its_own() {
        let lens = [(10, 10), (10, 10), (100, 5), (10, 10)];
        let b = plan_batches(&lens, |r, q, _| r <= 30 && q <= 30);
        assert_eq!(b, vec![0..2, 2..3, 3..4]);
        assert_eq!(
            plan_batches(&[], |_, _, _| true),
            Vec::<Range<usize>>::new()
        );
        assert_eq!(plan_batches(&lens, |_, _, _| true), vec![0..4]);
    }

    /// The batched seed stream for a chunk plan equals the concatenation of
    /// every record's standalone stream (its single-record intervals × chunks),
    /// shifted to the record's packed start — including records of exactly one
    /// span (masked), shorter and empty records, and a record at `len - span ==
    /// C - 1`.
    #[test]
    fn chunk_seed_streams_equal_per_pair_standalone_streams() {
        let shape = Shape::parse("12of19").unwrap();
        let span = shape.size;
        let c = 64u32;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut base = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b"ACGT"[(state % 4) as usize]
        };
        let lens = [40usize, 0, 19, 5, 82, 30, 19, 25, 60, 1];
        let recs: Vec<(String, Vec<u8>)> = lens
            .iter()
            .enumerate()
            .map(|(i, &l)| (format!("g{i}"), (0..l).map(|_| base()).collect()))
            .collect();
        let packed = PackedBin::build(
            recs.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
            "",
            false,
        );
        let mut buf = packed.buf.clone();
        mask_starts(&mut buf, &packed.chrs, |c| c.len as usize == span);
        let qbuf = &buf[..packed.block_len];

        let mut batched = Vec::new();
        for ch in plan_chunks(&packed.chrs, span, c) {
            assert!(ch.range.1 - ch.range.0 <= c);
            batched.extend(seed::chunk_seeds(qbuf, &shape, true, ch.range));
        }
        let mut standalone = Vec::new();
        for (k, (_, s)) in recs.iter().enumerate() {
            let mut one = s.clone();
            one.push(sequence::SEP);
            let off = packed.chrs[k].start as u64;
            for (start, end) in sequence::intervals(s.len(), span, 10_000_000) {
                for r in seed::chunks(start, end, c) {
                    standalone.extend(
                        seed::chunk_seeds(&one, &shape, true, r)
                            .into_iter()
                            .map(|x| x + off),
                    );
                }
            }
        }
        assert!(!standalone.is_empty());
        assert_eq!(batched, standalone);
    }

    /// The batched table equals the union of single-record tables, each shifted
    /// to its record's packed start: masking every record's first base
    /// reproduces the standalone "position 0 is never indexed" rule.
    #[test]
    fn masked_batch_table_is_the_union_of_standalone_tables() {
        let shape = Shape::parse("12of19").unwrap();
        let recs = [
            rec("a", "ACGTACGTACGTACGTACGTACGTT"),
            rec("b", "ACGTACGTACGTACGTACGTACGTT"),
            rec("c", ""),
        ];
        let packed = PackedBin::build(
            recs.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
            "",
            false,
        );
        let mut buf = packed.buf.clone();
        mask_starts(&mut buf, &packed.chrs, |_| true);
        let batch = SeedTable::build(&buf[..packed.block_len], &shape, 1);
        let mut want: Vec<(usize, u32)> = Vec::new();
        for (k, (_, s)) in recs.iter().enumerate() {
            let t = SeedTable::build(s, &shape, 1);
            for kmer in 0..t.index_table.len() {
                let lo = if kmer == 0 {
                    0
                } else {
                    t.index_table[kmer - 1]
                } as usize;
                for &p in &t.pos_table[lo..t.index_table[kmer] as usize] {
                    want.push((kmer, p + packed.chrs[k].start as u32));
                }
            }
        }
        let mut got = Vec::new();
        for kmer in 0..batch.index_table.len() {
            let lo = if kmer == 0 {
                0
            } else {
                batch.index_table[kmer - 1]
            } as usize;
            for &p in &batch.pos_table[lo..batch.index_table[kmer] as usize] {
                got.push((kmer, p));
            }
        }
        want.sort_unstable();
        assert!(!want.is_empty());
        assert_eq!(got, want);
    }

    #[test]
    fn chunks_pack_whole_records_up_to_c() {
        // Records 40, 19 (no seeds), 5 (no seeds), 82: ranges 22, -, -, 64.
        let ch = plan_chunks(&chrs(&[40, 19, 5, 82]), 19, 64);
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].pairs, 0..1);
        assert_eq!(ch[1].pairs, 3..4);
        let ch = plan_chunks(&chrs(&[25, 25, 25]), 19, 64);
        assert_eq!(ch.len(), 1);
        assert_eq!(ch[0].pairs, 0..3);
    }
}
