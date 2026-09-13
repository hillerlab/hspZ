// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The `hits-estimate` command (R92 PR1): exact per-unit seed-hit counts,
//! computed on the host before any GPU work.
//!
//! CPU-only: no CUDA calls anywhere on this path, so it runs on a node
//! without a GPU. Planning reuses `plan::plan_with` with the same flags and
//! defaults as `run`, so bins/blocks are identical to what `run` would freeze
//! whenever `run` does not shrink the plan against a device budget (all PR1
//! tests and the default whole-genome layout on a fitting GPU; see report.md
//! for the documented deviations).

use crate::Fallible;
use crate::cli::HitsEstimateArgs;
use crate::plan::{self, PackedBin};
use crate::seed::{self, Shape};
use crate::sequence;
use crate::timing;
use std::io::Write;
use std::time::Instant;

/// What one `estimate` call produced, for the stderr line.
pub(crate) struct EstimateReport {
    pub(crate) r: usize,
    pub(crate) q: usize,
    pub(crate) threads: usize,
    pub(crate) stride: u32,
    pub(crate) load_ms: f64,
    pub(crate) ref_hist_ms: f64,
    pub(crate) qry_hist_ms: f64,
    pub(crate) fold_dot_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) peak_rss_mib: u64,
}

/// Loads the inputs, plans like `run`, histograms every bin and dots to
/// row-major `r*Q+q` predictions (`--stride` scaled).
pub(crate) fn estimate(
    args: &HitsEstimateArgs,
) -> Fallible<(plan::Plan, Vec<u64>, EstimateReport)> {
    let total = Instant::now();
    if args.stride == 0 {
        return Err("--stride must be >= 1".into());
    }
    if args.lastz_interval_size == 0 {
        return Err("--lastz-interval-size must be >= 1".into());
    }
    if args.wga_chunk_size == 0 {
        return Err("--wga-chunk-size must be >= 1".into());
    }
    let plus = args.strand == "plus" || args.strand == "both";
    let minus = args.strand == "minus" || args.strand == "both";
    if !plus && !minus {
        return Err(format!("--strand must be plus, minus or both, got {}", args.strand).into());
    }
    let shape = Shape::parse(&args.seed)?;
    let transitions = !args.notransition;
    let threads = crate::run::resolve_threads(args.threads);

    let t = Instant::now();
    let (_, ref_records, _) = sequence::read_records(&args.reference)?;
    let (_, qry_records, _) = sequence::read_records(&args.query)?;
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;

    if args.seq_block_size == 0 && args.kegalign_bins {
        return Err(plan::AUTO_KEGALIGN_ERROR.into());
    }
    // Same resolution as `run` at one worker: `-B 0` means the default target.
    let ref_target = if args.seq_block_size == 0 {
        plan::DEFAULT_BLOCK_TARGET
    } else {
        u64::from(args.seq_block_size)
    };
    let qry_target = args.query_block_size.map(u64::from).unwrap_or(ref_target);
    let plan = plan::plan_with(
        &crate::run::record_meta(&ref_records),
        &crate::run::record_meta(&qry_records),
        ref_target.max(1),
        qry_target.max(1),
        args.kegalign_bins,
    );

    if let Some(path) = &args.dump_plan {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for (side, bins, recs) in [
            ("reference", &plan.reference_bins, &ref_records),
            ("query", &plan.query_bins, &qry_records),
        ] {
            for b in bins {
                for &id in &b.record_ids {
                    let (name, seq) = &recs[id as usize];
                    writeln!(f, "{side}\t{}\t{name}\t{}", b.id, seq.len())?;
                }
            }
        }
        f.flush()?;
    }

    // Histograms over exactly what the executor consumes: packed bins from
    // the shared `PackedBin::build` (byte-identical to `run`).
    let t = Instant::now();
    let mut ref_hists = Vec::with_capacity(plan.reference_bins.len());
    for b in &plan.reference_bins {
        let packed = PackedBin::build(
            b.record_ids.iter().map(|&id| {
                let (n, s) = &ref_records[id as usize];
                (n.as_str(), s.as_slice())
            }),
            "",
            false,
        );
        ref_hists.push(seed::count_ref_block_parallel(
            &packed.buf[..packed.block_len],
            &shape,
            args.step,
            threads,
        )?);
    }
    let ref_hist_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut qry_hists = Vec::with_capacity(plan.query_bins.len());
    for b in &plan.query_bins {
        let packed = PackedBin::build(
            b.record_ids.iter().map(|&id| {
                let (n, s) = &qry_records[id as usize];
                (n.as_str(), s.as_slice())
            }),
            "",
            true,
        );
        qry_hists.push(seed::count_query_block(
            &packed.buf[..packed.block_len],
            &packed.rc,
            packed.block_len,
            &shape,
            plus,
            minus,
            args.lastz_interval_size,
            args.wga_chunk_size,
            args.stride,
            threads,
        )?);
    }
    let qry_hist_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let hits = seed::unit_hits(&ref_hists, &qry_hists, &shape, transitions);
    let fold_dot_ms = t.elapsed().as_secs_f64() * 1000.0;

    // `--stride` S counts every S-th query window start: scale back up.
    let scale = u64::from(args.stride);
    let mut pred = Vec::with_capacity(hits.len());
    for &h in &hits {
        pred.push(
            h.checked_mul(scale)
                .ok_or("hits-estimate: hit count overflow")?,
        );
    }
    let report = EstimateReport {
        r: plan.reference_bins.len(),
        q: plan.query_bins.len(),
        threads,
        stride: args.stride,
        load_ms,
        ref_hist_ms,
        qry_hist_ms,
        fold_dot_ms,
        total_ms: total.elapsed().as_secs_f64() * 1000.0,
        peak_rss_mib: timing::peak_rss_kib() / 1024,
    };
    Ok((plan, pred, report))
}

/// TSV rows to stdout, one stderr summary line (sufficient for whole-genome
/// timing on the 24-thread node).
pub(crate) fn run(args: &HitsEstimateArgs) -> Fallible<()> {
    let (plan, pred, rep) = estimate(args)?;
    let q = rep.q;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (r, rb) in plan.reference_bins.iter().enumerate() {
        for (qq, qb) in plan.query_bins.iter().enumerate() {
            writeln!(
                out,
                "{r}\t{qq}\t{}\t{}\t{}",
                rb.total_bp,
                qb.total_bp,
                pred[r * q + qq]
            )?;
        }
    }
    out.flush()?;
    eprintln!(
        "hits-estimate: R={} Q={} threads={} stride={} load_ms={:.0} ref_hist_ms={:.0} \
         qry_hist_ms={:.0} fold_dot_ms={:.0} total_ms={:.0} peak_rss_mib={}",
        rep.r,
        rep.q,
        rep.threads,
        rep.stride,
        rep.load_ms,
        rep.ref_hist_ms,
        rep.qry_hist_ms,
        rep.fold_dot_ms,
        rep.total_ms,
        rep.peak_rss_mib,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use crate::seed::{SeedTable, chunk_seeds};
    use clap::Parser;

    /// Independent oracle for one packed (ref, query) pair: the existing
    /// `cpu_stats` logic — `SeedTable::hit_count` summed over `chunk_seeds`
    /// on both strands — which the estimator must reproduce exactly. The
    /// window starts come from [`seed::chunks`], the same batch walk the
    /// device seeder and `cpu_stats` both execute.
    fn oracle_hits(
        ref_buf: &[u8],
        fwd: &[u8],
        rc: &[u8],
        block_len: usize,
        shape: &Shape,
        step: u32,
        transitions: bool,
        plus: bool,
        minus: bool,
        lastz: u32,
        wga_chunk: u32,
    ) -> u64 {
        let table = SeedTable::build(ref_buf, shape, step);
        let mut hits = 0u64;
        let qbl = (block_len - shape.size) as u32;
        for &(s, e) in &sequence::intervals(block_len, shape.size, lastz) {
            if plus {
                for (lo, hi) in seed::chunks(s, e, wga_chunk) {
                    for seed in chunk_seeds(fwd, shape, transitions, (lo, hi)) {
                        hits += table.hit_count((seed >> 32) as u32) as u64;
                    }
                }
            }
            if minus {
                for (lo, hi) in seed::chunks(qbl - e, qbl - s, wga_chunk) {
                    for seed in chunk_seeds(rc, shape, transitions, (lo, hi)) {
                        hits += table.hit_count((seed >> 32) as u32) as u64;
                    }
                }
            }
        }
        hits
    }

    /// Hand-counted toy: short ACGT records with one N run and one lowercase
    /// run per side, both strands, transitions on and off, steps 1-3, against
    /// the `SeedTable` + `chunk_seeds` oracle. Runs in the normal suite.
    #[test]
    fn toy_estimator_matches_pipeline_oracle() {
        let shape = Shape::parse("TTT0T").unwrap(); // size 5, k 4, all-transition
        let lastz = 7; // force several intervals on ~20 bp blocks
        let ref_recs = vec![
            ("r1", b"ACGTACGTACGTACGT".as_slice()),
            ("r2", b"AAAANNNNAAAACCCC".as_slice()),
            ("r3", b"acgtACGTacgtACGT".as_slice()),
        ];
        let qry_recs = vec![
            ("q1", b"TTTTGGGGAAAACCCCGGGG".as_slice()),
            ("q2", b"ACnnACGTACGTACGTAC".as_slice()),
        ];
        let (ref_buf, _, ref_len) = sequence::pack(ref_recs.iter().map(|(n, s)| (*n, *s)), "");
        let (fwd, fwd_chrs, block_len) = sequence::pack(qry_recs.iter().map(|(n, s)| (*n, *s)), "");
        let (rc, _) = sequence::reverse_complement(&fwd, &fwd_chrs, block_len);
        assert!(ref_buf.iter().any(|&c| c == b'N'));
        assert!(ref_buf.iter().any(|&c| c == b'a'));

        for transitions in [true, false] {
            for (plus, minus) in [(true, true), (true, false), (false, true)] {
                for step in [1u32, 2, 3] {
                    let want = oracle_hits(
                        &ref_buf[..ref_len],
                        &fwd[..block_len],
                        &rc,
                        block_len,
                        &shape,
                        step,
                        transitions,
                        plus,
                        minus,
                        lastz,
                        250_000,
                    );
                    for threads in [1usize, 3] {
                        let rh = seed::count_ref_block_parallel(
                            &ref_buf[..ref_len],
                            &shape,
                            step,
                            threads,
                        )
                        .unwrap();
                        // Serial and parallel reference counts agree.
                        assert_eq!(
                            rh,
                            seed::count_ref_block(&ref_buf[..ref_len], &shape, step).unwrap()
                        );
                        let qh = seed::count_query_block(
                            &fwd[..block_len],
                            &rc,
                            block_len,
                            &shape,
                            plus,
                            minus,
                            lastz,
                            250_000,
                            1,
                            threads,
                        )
                        .unwrap();
                        let got = seed::unit_hits(&[rh], &[qh], &shape, transitions);
                        assert_eq!(
                            got,
                            vec![want],
                            "transitions={transitions} plus={plus} minus={minus} step={step} threads={threads}"
                        );
                    }
                }
            }
        }
    }

    /// What `seed::chunks` does at a shared endpoint is the entire boundary
    /// question. Consecutive intervals share their endpoint (`intervals()`
    /// emits `(0,7),(7,11),...`); the left interval seeds it only when its
    /// chunk walk does not land exactly on it — i.e. when its length is not a
    /// multiple of `wga_chunk` — otherwise the right interval seeds it, once.
    /// With a chunk larger than the interval (the test's 250,000 against 7)
    /// every endpoint lands inside the single chunk and is seeded twice.
    /// Packed-bin counts therefore do NOT equal summed per-record counts in
    /// general — the boundaries differ — which is why the estimator histograms
    /// packed bins, not records.
    #[test]
    fn interval_boundary_windows_are_counted_twice_like_the_pipeline() {
        let shape = Shape::parse("TTT0T").unwrap();
        // All-valid 16 bp record, lastz 7 -> intervals (0,7),(7,11), default
        // chunk: start 7 is seeded in both intervals, 13 seeds from 12 windows.
        let seq = b"ACGTACGTACGTACGT".as_slice();
        let bin = PackedBin::build([("r1", seq)], "", true);
        let h = seed::count_query_block(
            &bin.buf[..bin.block_len],
            &bin.rc,
            bin.block_len,
            &shape,
            true,
            false,
            7,
            250_000,
            1,
            1,
        )
        .unwrap();
        let total: u32 = h.iter().sum();
        assert_eq!(total, 13, "boundary start 7 must be double-counted");
        // And the oracle agrees (it walks the same chunks).
        let want = oracle_hits(
            &bin.buf[..bin.block_len],
            &bin.buf[..bin.block_len],
            &bin.rc,
            bin.block_len,
            &shape,
            1,
            true,
            true,
            false,
            7,
            250_000,
        );
        let got = seed::unit_hits(
            &[seed::count_ref_block(&bin.buf[..bin.block_len], &shape, 1).unwrap()],
            &[h],
            &shape,
            true,
        );
        assert_eq!(got, vec![want]);
    }

    /// Multi-interval exactness on a 200 bp record, CPU-only. With `-I 40`
    /// and `-C 8` (`40 % 8 == 0`) the shared endpoints are seeded once and the
    /// estimator must NOT double them; with the default `-C 250000` the single
    /// chunk spans each interval and they are seeded twice. `-I 45` is not a
    /// multiple of 8, so both chunk sizes seed every endpoint twice. The oracle
    /// is the `cpu_stats`/device batch walk (`seed::chunks` + `chunk_seeds` +
    /// `hit_count`); the chunk sizes are also shown to disagree at `-I 40`.
    #[test]
    fn multi_interval_endpoints_follow_wga_chunking() {
        let shape = Shape::parse("TTT0T").unwrap();
        let seq: Vec<u8> = (0..200u32)
            .map(|i| b"ACGT"[(i.wrapping_mul(7) % 4) as usize])
            .collect();
        let bin = PackedBin::build([("q1", seq.as_slice())], "", true);
        let mut totals = Vec::new();
        for &(lastz, chunk) in &[(40u32, 8u32), (40, 250_000), (45, 8), (45, 250_000)] {
            let want = oracle_hits(
                &bin.buf[..bin.block_len],
                &bin.buf[..bin.block_len],
                &bin.rc,
                bin.block_len,
                &shape,
                1,
                true,
                true,
                true,
                lastz,
                chunk,
            );
            totals.push(want);
            for threads in [1usize, 3] {
                let rh =
                    seed::count_ref_block_parallel(&bin.buf[..bin.block_len], &shape, 1, threads)
                        .unwrap();
                let qh = seed::count_query_block(
                    &bin.buf[..bin.block_len],
                    &bin.rc,
                    bin.block_len,
                    &shape,
                    true,
                    true,
                    lastz,
                    chunk,
                    1,
                    threads,
                )
                .unwrap();
                let got = seed::unit_hits(&[rh], &[qh], &shape, true);
                assert_eq!(
                    got,
                    vec![want],
                    "lastz={lastz} chunk={chunk} threads={threads}"
                );
            }
        }
        assert!(
            totals[0] < totals[1],
            "C=8 must drop the duplicated endpoints: {totals:?}"
        );
        assert_eq!(
            totals[2], totals[3],
            "I=45 endpoint rule is chunk-independent: {totals:?}"
        );
    }

    #[test]
    fn wide_seeds_error_cleanly() {
        let shape = Shape::parse("14of22").unwrap();
        assert!(seed::count_ref_block(b"ACGTACGT", &shape, 1).is_err());
        assert!(
            seed::count_query_block(
                b"ACGTACGT",
                b"ACGTACGT",
                8,
                &shape,
                true,
                true,
                7,
                250_000,
                1,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn hits_estimate_flags_parse() {
        match Cli::try_parse_from(["hspz", "hits-estimate", "-r", "r.fa", "-q", "q.fa"])
            .unwrap()
            .command
        {
            Command::HitsEstimate(a) => {
                assert_eq!(a.seq_block_size, 500_000_000);
                assert_eq!(a.query_block_size, None);
                assert!(!a.kegalign_bins);
                assert_eq!(a.seed, "12of19");
                assert_eq!(a.step, 1);
                assert!(!a.notransition);
                assert_eq!(a.strand, "both");
                assert_eq!(a.threads, 0);
                assert_eq!(a.lastz_interval_size, 10_000_000);
                assert_eq!(a.wga_chunk_size, 250_000);
                assert_eq!(a.stride, 1);
                assert_eq!(a.dump_plan, None);
            }
            _ => panic!("wrong subcommand"),
        }
        // `-I`/`-C` must be real flags with the same names as `run`.
        match Cli::try_parse_from([
            "hspz",
            "hits-estimate",
            "-r",
            "r.fa",
            "-q",
            "q.fa",
            "-I",
            "1000000",
            "-C",
            "8",
        ])
        .unwrap()
        .command
        {
            Command::HitsEstimate(a) => {
                assert_eq!(a.lastz_interval_size, 1_000_000);
                assert_eq!(a.wga_chunk_size, 8);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    /// A zero interval size or chunk size would loop forever in
    /// `sequence::intervals` / `seed::chunks`; the estimator must reject both
    /// before touching any input.
    #[test]
    fn zero_interval_or_chunk_errors() {
        let mut args = chr20_args();
        args.lastz_interval_size = 0;
        assert!(estimate(&args).is_err());
        args.lastz_interval_size = 10_000_000;
        args.wga_chunk_size = 0;
        assert!(estimate(&args).is_err());
    }

    fn chr20_args() -> HitsEstimateArgs {
        HitsEstimateArgs {
            reference: "/tmp/hspz-cycle3-chr20/ref.fa".into(),
            query: "/tmp/hspz-cycle3-chr20/qry.fa".into(),
            seq_block_size: 500_000_000,
            query_block_size: None,
            kegalign_bins: false,
            seed: "12of19".into(),
            step: 1,
            notransition: false,
            strand: "both".into(),
            threads: 0,
            lastz_interval_size: 10_000_000,
            wga_chunk_size: 250_000,
            stride: 1,
            dump_plan: None,
        }
    }

    /// chr20 pair, default flags: the single predicted unit equals `#seed
    /// hits` = 192,899,566 (round-90 ledger; re-derived with
    /// `hspz run --time --gpus 1` — see report.md). Needs the input files;
    /// the GPU re-derivation itself is manual.
    #[test]
    #[ignore]
    fn chr20_single_unit_matches_seed_hits() {
        let (plan, pred, _) = estimate(&chr20_args()).unwrap();
        assert_eq!((plan.reference_bins.len(), plan.query_bins.len()), (1, 1));
        assert_eq!(pred, vec![192_899_566]);
    }

    /// chr20, multi-interval: with `-I 1000000` (4 intervals) the pipeline
    /// reports `#seed hits` = 192,899,602 in all three paths — device seeder,
    /// CPU seeder and `--cpu-only` (report2.md) — which the estimator with the
    /// same `-I`/`-C` must reproduce exactly. The shared endpoints are seeded
    /// once because 1,000,000 is a multiple of the default chunk 250,000.
    #[test]
    #[ignore]
    fn chr20_interval_1000000_matches_seed_hits() {
        let args = HitsEstimateArgs {
            lastz_interval_size: 1_000_000,
            ..chr20_args()
        };
        let (plan, pred, _) = estimate(&args).unwrap();
        assert_eq!((plan.reference_bins.len(), plan.query_bins.len()), (1, 1));
        assert_eq!(pred, vec![192_899_602]);
    }

    /// chr20, `-I 300000`: the interval length is not a multiple of the
    /// default chunk, so both seeders double-seed every shared endpoint and
    /// `#seed hits` = 192,900,034 (report2.md). The estimator must agree.
    #[test]
    #[ignore]
    fn chr20_interval_300000_matches_seed_hits() {
        let args = HitsEstimateArgs {
            lastz_interval_size: 300_000,
            ..chr20_args()
        };
        let (plan, pred, _) = estimate(&args).unwrap();
        assert_eq!((plan.reference_bins.len(), plan.query_bins.len()), (1, 1));
        assert_eq!(pred, vec![192_900_034]);
    }

    /// The 5x3 synthetic plan (`-B 10000000 --query-block-size 1000000`):
    /// all 15 predictions equal the ledger's 15 `hits` values (from
    /// `hspz run --time --gpus 2 --max-hits 300000` on ZLUDA — see
    /// report.md). Row-major `r*3+q`.
    #[test]
    #[ignore]
    fn synthetic_5x3_plan_matches_ledger_hits() {
        let args = HitsEstimateArgs {
            reference: "/tmp/opencode/r90/ref5.fa".into(),
            query: "/tmp/opencode/r90/qry3.fa".into(),
            seq_block_size: 10_000_000,
            query_block_size: Some(1_000_000),
            kegalign_bins: false,
            seed: "12of19".into(),
            step: 1,
            notransition: false,
            strand: "both".into(),
            threads: 0,
            lastz_interval_size: 10_000_000,
            wga_chunk_size: 250_000,
            stride: 1,
            dump_plan: None,
        };
        let (plan, pred, _) = estimate(&args).unwrap();
        assert_eq!((plan.reference_bins.len(), plan.query_bins.len()), (5, 3));
        assert_eq!(
            pred,
            vec![
                14_652_192, 14_903_352, 14_303_585, //
                15_314_304, 15_665_330, 15_130_480, //
                8_010_334, 8_393_551, 8_339_378, //
                11_633_793, 12_239_693, 12_206_053, //
                13_491_008, 14_274_597, 14_340_512,
            ]
        );
    }

    /// Private-tables vs key-sharded reference counting on chr20: identical
    /// results, with wall times for the strategy comparison (run with
    /// `-- --nocapture`).
    #[test]
    #[ignore]
    fn chr20_private_vs_sharded_strategy() {
        let shape = Shape::parse("12of19").unwrap();
        let (_, records, _) = sequence::read_records(chr20_args().reference.as_path()).unwrap();
        let (buf, _, block_len) =
            sequence::pack(records.iter().map(|(n, s)| (n.as_str(), s.as_slice())), "");
        let threads = crate::run::resolve_threads(0);
        let t = Instant::now();
        let a = seed::count_ref_block_parallel(&buf[..block_len], &shape, 1, threads).unwrap();
        let ms_a = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let b = seed::count_ref_block_sharded(&buf[..block_len], &shape, 1, threads).unwrap();
        let ms_b = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a, b);
        eprintln!(
            "strategy: private-tables {ms_a:.0} ms vs key-sharded {ms_b:.0} ms ({threads} threads)"
        );
    }
}
