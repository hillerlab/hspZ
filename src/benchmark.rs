// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The `benchmark` command: the optimization oracle. Prepares input and the
//! CUDA context once, warms up, then times repeated Seed + Filter passes,
//! checking after every one that the HSP hash is unchanged. The FNV hashes
//! here are the parity check the whole optimization loop is built on.

use crate::Fallible;
use crate::cli::BenchArgs;
use crate::gpu::{self, Engine};
use crate::hsp::{self, SegmentPair};
use crate::run::{
    OutputReport, Pass, Prepared, device_seeds_for, prepare, resolve_threads, seed_and_filter_all,
    write_outputs,
};
use crate::timing::{self, Phases};
use cuda_core::CudaContext;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// benchmark — PLAN.md Milestone 3

/// One timed warm iteration: its wall elapsed and that same iteration's phase
/// rows, kept together so a stage table always describes the pass it was
/// recorded for. `warms` is never reordered in place — statistics select the
/// median through an index view ([`sorted_sample_indices`]) and JSON reads
/// `elapsed_ms` and `phases` from the same element, so the ordinal and the
/// stages stay paired.
struct WarmSample {
    elapsed_ms: f64,
    phases: Phases,
}

/// Original-order indices sorted by ascending `elapsed_ms`, so statistics and
/// the test read a sample's stages from the same element its duration came
/// from. The middle element of the returned order keeps the existing
/// upper-middle median convention. `warms` itself is never reordered, so JSON
/// ordinals stay the real execution order.
fn sorted_sample_indices(warms: &[WarmSample]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..warms.len()).collect();
    order.sort_by(|&a, &b| warms[a].elapsed_ms.total_cmp(&warms[b].elapsed_ms));
    order
}

pub(crate) fn benchmark(args: &BenchArgs, pre_main_ms: f64) -> Fallible<()> {
    // The benchmark always wants CUDA-event durations alongside host time.
    let mut run_args = args.run.clone();
    run_args.time = true;
    let run_args = &run_args;
    let mut cold = Phases::new();
    cold.add_ms("process startup", pre_main_ms);

    let p = prepare(run_args, &mut cold)?;

    let t = Instant::now();
    let ctx = CudaContext::new(0)?;
    cold.add("CUDA context init", t.elapsed());

    let mut contract =
        gpu::ExecutionContract::resolve(&ctx, run_args.max_hits, run_args.hsp_blocks);
    // Physical capacity for this prepared single-bin pair under a one-device
    // budget; same helper and kernel clamp as production. Never replans.
    {
        let plan = p.capacity_plan();
        let budget = gpu::min_free_bytes(1)?;
        let (shape, transitions) = p.seeding();
        let candidate = crate::plan::max_hit_capacity(
            &plan,
            budget,
            shape.kmer_size,
            run_args.step,
            contract.max_hits,
            run_args.wga_chunk_size,
            transitions,
        )?;
        contract.hit_capacity =
            gpu::clamp_hit_capacity(contract.max_hits, candidate, contract.hsp_blocks)?;
    }
    let mut engine = Engine::new(&ctx, p.engine_config(run_args, &contract), &mut cold)?;
    // Single-worker driver, so this is the env override or false — but it must go
    // through the same function the executor uses, or the two disagree silently.
    engine.device_seeds = device_seeds_for(1);
    // AM-B1: reference-only construction, so the query enters via swap_query
    // exactly as it will for every bin in a multi-bin plan.
    let (fwd, rc) = p.encoded_query();
    engine.swap_query(fwd, rc)?;
    engine.collect_hit_stats = true;
    engine.persistent_seed_buffers = !run_args.no_persistent_seed_buffers;
    engine.async_stages = !run_args.no_async_stages;

    // -D / -Z put partition/format/archive on the critical path (PLAN.md §15/§22);
    // without them the benchmark never touches the output layer, so the core
    // numbers stay identical to before.
    let output_mode = run_args.tarball.is_some() || run_args.diagonal_partition;

    // Cold: first pass, paying every one-time cost.
    let t = Instant::now();
    let first = {
        let (sh, tr) = p.seeding();
        seed_and_filter_all(
            &mut engine,
            &p.query_pass(),
            sh,
            tr,
            run_args,
            resolve_threads(run_args.threads),
        )?
    };
    let cold_pass_ms = t.elapsed().as_secs_f64() * 1000.0;
    cold.merge(&engine.phases);
    let cold_total = cold.total_ms();
    let mut hit_stats = std::mem::take(&mut engine.hit_stats);
    engine.collect_hit_stats = false;
    let cold_out = if output_mode {
        write_outputs(run_args, &p, &first, &mut Phases::new())?
    } else {
        OutputReport::default()
    };

    let launch_ms = engine.launch_overhead_ms(args.launch_probe)?;
    let launches_per_pass = engine.launches;

    let reference_hash = hash_pass(&first);
    let segments_hash = hash_segments(&p, &first);

    for _ in 0..args.warmup {
        let pass = {
            let (sh, tr) = p.seeding();
            seed_and_filter_all(
                &mut engine,
                &p.query_pass(),
                sh,
                tr,
                run_args,
                resolve_threads(run_args.threads),
            )?
        };
        if hash_pass(&pass) != reference_hash {
            return Err("warmup iteration produced different HSPs".into());
        }
        if output_mode {
            write_outputs(run_args, &p, &pass, &mut Phases::new())?;
        }
    }

    // Warm: one record per iteration so the median elapsed is paired with *that*
    // iteration's stages, not the last pass vs a sorted elapsed sample. The
    // record type lives at module scope ([`WarmSample`]), so the no-GPU
    // regression test drives the same selection/serialization path.
    let mut warms = Vec::with_capacity(args.iterations as usize);
    let mut out_sum = OutputReport::default();
    let mut out_iters = 0usize;
    for _ in 0..args.iterations {
        engine.phases = Phases::new();
        // Round 71: the gap accumulator is per-pass, like `phases`, or it is quoted
        // against a wall it does not cover.
        engine.reset_gaps();
        let t = Instant::now();
        let pass = {
            let (sh, tr) = p.seeding();
            seed_and_filter_all(
                &mut engine,
                &p.query_pass(),
                sh,
                tr,
                run_args,
                resolve_threads(run_args.threads),
            )?
        };
        if hash_pass(&pass) != reference_hash {
            return Err("benchmark iteration produced different HSPs".into());
        }
        if output_mode {
            let r = write_outputs(run_args, &p, &pass, &mut engine.phases)?;
            out_sum.partition_ms += r.partition_ms;
            out_sum.format_ms += r.format_ms;
            out_sum.archive_ms += r.archive_ms;
            out_sum.files += r.files;
            out_sum.bytes_in += r.bytes_in;
            out_sum.bytes_out += r.bytes_out;
            out_iters += 1;
        }
        warms.push(WarmSample {
            elapsed_ms: t.elapsed().as_secs_f64() * 1000.0,
            phases: std::mem::take(&mut engine.phases),
        });
    }
    // Statistics sort an *index* view so each sample's elapsed stays paired with
    // its own stages; `warms` keeps original execution order for JSON ordinals.
    let order = sorted_sample_indices(&warms);
    let sorted: Vec<f64> = order.iter().map(|&i| warms[i].elapsed_ms).collect();
    let mid = sorted.len() / 2;
    let median = sorted[mid];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p95 = sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1)];
    // The sample whose *own* elapsed is the median — its stages, not a sorted
    // neighbour's, are what the WARM STAGES table and the JSON median show.
    let median_idx = order[mid];
    let median_iteration = median_idx + 1;

    println!("PARITY");
    println!("  final HSP hash    {reference_hash:016x}");
    println!("  segments hash     {segments_hash:016x}");
    println!("\nTIMING (ms)");
    println!("  cold total        {cold_total:9.2}");
    println!("  cold pass         {cold_pass_ms:9.2}");
    println!("  warm median       {median:9.2}");
    println!("  warm mean         {mean:9.2}");
    println!("  warm p95          {p95:9.2}");
    println!(
        "  warm min/max      {:9.2} / {:.2}",
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!("  iterations        {:9}", args.iterations);
    println!("\nCOLD BREAKDOWN\n{}", cold.report(cold_total));
    println!(
        "WARM STAGES (median iteration {median_iteration} of {})\n{}",
        args.iterations,
        warms[median_idx].phases.report(median)
    );
    println!("LAUNCH OVERHEAD");
    println!(
        "  per launch        {launch_ms:9.4} ms  ({} probes)",
        args.launch_probe
    );
    println!("  launches/pass     {launches_per_pass:9}");
    println!(
        "  launch total      {:9.2} ms  ({:.1}% of warm median)",
        launch_ms * launches_per_pass as f64,
        launch_ms * launches_per_pass as f64 / median * 100.0
    );
    if let Some(path) = &args.json {
        write_json_records(
            args,
            path,
            &warms,
            &first,
            &engine,
            launch_ms,
            segments_hash,
        )?;
    }

    if output_mode {
        let n = out_iters.max(1) as f64;
        println!("\nOUTPUT (mean per warm iteration)");
        println!("  partition   {:9.2} ms", out_sum.partition_ms / n);
        println!("  format      {:9.2} ms", out_sum.format_ms / n);
        println!("  archive     {:9.2} ms", out_sum.archive_ms / n);
        println!(
            "  output total {:9.2} ms",
            (out_sum.partition_ms + out_sum.format_ms + out_sum.archive_ms) / n
        );
        println!("  files       {:9}", out_sum.files / out_iters.max(1));
        let wi = out_iters.max(1) as u64;
        println!(
            "  bytes       {:9} formatted, {} written/iter",
            out_sum.bytes_in / wi,
            if out_sum.bytes_out > 0 {
                out_sum.bytes_out / wi
            } else {
                out_sum.bytes_in / wi
            }
        );
        println!(
            "  cold output {:9.2} ms (first pass only)",
            cold_out.partition_ms + cold_out.format_ms + cold_out.archive_ms
        );
    }

    let (_, total_mem) = gpu::device_memory();
    println!("\nGPU");
    println!(
        "  host syncs        {:9} stage, {} pipeline (Phase 1 §12)",
        engine.stage_syncs(),
        engine.pipeline_syncs()
    );
    for ((a, c), ms, n) in engine.gap_pairs().into_iter().take(6) {
        println!("    gap {a} -> {c}: {ms:.2} ms over {n}");
    }
    let (gap_ms, gap_n, gap_max, (ga, gb)) = engine.stage_gaps();
    if gap_n > 0 {
        // Round 71: GPU-timeline idle between one stage's end and the next stage's
        // start, which is the only place a host round trip inside the chunk loop can
        // show up. The phase rows cannot recover it.
        println!(
            "  stage gaps        {gap_ms:9.2} ms over {gap_n} pairs, largest {gap_max:.3} ms \
             ({ga} -> {gb})"
        );
    }
    let (uploads, stalls) = engine.seed_copy_stats();
    println!("  seed uploads      {uploads:9} ({stalls} stalled)");
    println!(
        "  peak allocated    {:9.1} MiB",
        engine.peak_used as f64 / 1048576.0
    );
    println!(
        "  device total      {:9.1} MiB",
        total_mem as f64 / 1048576.0
    );
    println!("  host peak RSS     {:9} KiB", timing::peak_rss_kib());
    println!("\nDATA");
    println!("  seeds             {:9}", first.stats.seeds);
    println!("  seed hits         {:9}", first.stats.seed_hits);
    println!("  raw HSPs          {:9}", first.stats.raw_hsps);
    println!("  final HSPs        {:9}", first.stats.hsps);
    println!("\nHITS PER SEED\n{}", hit_stats.report());
    if let Some(c) = engine.census.as_mut() {
        println!("\n{}", c.report());
    }
    Ok(())
}

/// One JSON record per warm iteration (PLAN.md §6), appended so a whole matrix
/// lands in a single `results.jsonl`. Hand-rolled rather than pulling in serde:
/// this is the only JSON the binary emits and every value is a number, a bool,
/// or one of our own label strings.
///
/// The wrapper is the only code that touches GPU/run state: it gathers the
/// engine facts and device memory, folds them into `common_fields` once, and
/// delegates the per-iteration emission to the GPU-free [`write_json_samples`].
/// Every record still derives its `whole_ms`, `stages`, and `counts.batches`
/// from the *same* [`WarmSample`] it is numbered after, and `first` is
/// run-level — every pass is hash-checked identical.
#[allow(clippy::too_many_arguments)]
fn write_json_records(
    args: &BenchArgs,
    path: &Path,
    warms: &[WarmSample],
    first: &Pass,
    engine: &Engine,
    launch_ms: f64,
    segments_hash: u64,
) -> Fallible<()> {
    // This function is only reached when --json is requested, so the device
    // query belongs here rather than in the caller where it would run even for
    // a plain benchmark.
    let (_, total_mem) = gpu::device_memory();
    let parity = format!(
        "{{\"hsp_hash\":\"{:016x}\",\"segments_hash\":\"{:016x}\"}}",
        hash_pass(first),
        segments_hash,
    );
    // PLAN.md §1.2: every variant must prove its mechanism activated before any
    // timing delta is read. Job 49359833 produced a full performance table from
    // six arms that had all silently run the baseline config, so these are
    // recorded per run and `report.py` marks a mismatch VOID rather than
    // PASS/FAIL/INCONCLUSIVE.
    let mechanisms = format!(
        "{{\"pinned_seeds\":{},\"persistent_seed_buffers\":{},\"entropy_unpacked\":false,\
         \"segment_align16\":{},\"find_num_unchecked\":{},\"async_seed_upload\":{},\
         \"device_seeds\":{},\"device_seed_check\":{},\"warp_score_gate\":{},\"dense_anchors\":{},\
         \"left_pair_tile\":{},\"simd_prelude\":{},\"hsp_blocks\":{},\"hit_capacity\":{},\
         \"stage_syncs\":{},\"pipeline_syncs\":{}}}",
        engine.pinned_seeds_active(),
        !args.run.no_persistent_seed_buffers,
        align_of::<SegmentPair>() == 16,
        cfg!(feature = "nvidia-find-num-unchecked"),
        engine.async_seed_copy,
        engine.device_seeds,
        cfg!(feature = "device-seeds-check"),
        cfg!(feature = "warp-score-gate"),
        cfg!(feature = "dense-anchors"),
        cfg!(feature = "left-pair-tile"),
        cfg!(feature = "simd-prelude"),
        engine.hsp_blocks(),
        engine.hit_capacity(),
        engine.stage_syncs(),
        engine.pipeline_syncs(),
    );

    let env = format!(
        "{{\"hostname\":\"{}\",\"slurm_job_id\":\"{}\",\"date_utc\":\"{}\",\
         \"vram_mib\":{:.0},\"max_hits\":{},\"hsp_blocks\":{},\"threads\":{},\
         \"hspz_version\":\"{}\",\"extra\":{}}}",
        timing::json_key(&hostname()),
        timing::json_key(&std::env::var("SLURM_JOB_ID").unwrap_or_default()),
        timing::json_key(&utc_now()),
        total_mem as f64 / 1048576.0,
        engine.max_hits(),
        engine.hsp_blocks(),
        resolve_threads(args.run.threads),
        env!("CARGO_PKG_VERSION"),
        // Verbatim object from the runner; it is the only caller.
        args.env_json.trim(),
    );

    // Run-level record fields, once, without outer braces: variant, workload,
    // launch overhead, peak VRAM, and the three run-level objects.
    let common_fields = format!(
        "\"variant\":\"{}\",\"workload\":\"{}\",\"launch_ms\":{:.6},\"peak_vram_mib\":{:.1},\
         \"parity\":{},\"mechanisms\":{},\"environment\":{}",
        timing::json_key(&args.variant),
        timing::json_key(&args.workload),
        launch_ms,
        engine.peak_used as f64 / 1048576.0,
        parity,
        mechanisms,
        env,
    );

    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    write_json_samples(&mut f, warms, first, &common_fields)?;
    Ok(())
}

/// Emits one complete JSON object per warm iteration in original execution
/// order onto `out`. All run-level fields arrive pre-rendered in
/// `common_fields` — a leading field list without outer braces — and each
/// record appends its own `iteration`, `whole_ms`, `stages`, and `counts`, all
/// read from the same [`WarmSample`], so the ordinal and the stages stay
/// paired. GPU-free: the benchmark writes to the JSON file, the regression test
/// to a `Vec<u8>`.
fn write_json_samples(
    out: &mut impl std::io::Write,
    samples: &[WarmSample],
    first: &Pass,
    common_fields: &str,
) -> std::io::Result<()> {
    for (i, s) in samples.iter().enumerate() {
        let stages = s.phases.json_stages();
        let counts = format!(
            "{{\"seeds\":{},\"seed_hits\":{},\"raw_hsps\":{},\"hsps\":{},\"batches\":{}}}",
            first.stats.seeds,
            first.stats.seed_hits,
            first.stats.raw_hsps,
            first.stats.hsps,
            s.phases.calls("find_hsps"),
        );
        writeln!(
            out,
            "{{{common_fields},\"iteration\":{},\"whole_ms\":{:.4},\"stages\":{},\"counts\":{}}}",
            i + 1,
            s.elapsed_ms,
            stages,
            counts,
        )?;
    }
    Ok(())
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Seconds-resolution UTC from the clock, without pulling in `chrono`.
fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since epoch -> civil date, Howard Hinnant's algorithm.
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = era * 400 + yoe + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn hash_pass(pass: &Pass) -> u64 {
    let mut h = FNV_OFFSET;
    for (fw, rc) in &pass.intervals {
        for s in fw.iter().chain(rc.iter()) {
            for v in [s.ref_start, s.query_start, s.len, s.score as u32] {
                h = fnv1a(&v.to_le_bytes(), h);
            }
        }
    }
    h
}

/// Hashes the rendered `.segments` text without writing it, so the benchmark
/// checks the same bytes the parity suite diffs.
fn hash_segments(p: &Prepared, pass: &Pass) -> u64 {
    let mut h = FNV_OFFSET;
    for (fw, rc) in &pass.intervals {
        for (hsps, chrs, strand) in [(fw, &p.query.chrs, '+'), (rc, &p.rc_chrs, '-')] {
            h = fnv1a(
                hsp::render_segments(hsps, &p.reference.chrs, chrs, strand).as_bytes(),
                h,
            );
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn fnv_distinguishes_records() {
        let a = fnv1a(b"abc", FNV_OFFSET);
        assert_ne!(a, fnv1a(b"abd", FNV_OFFSET));
        assert_eq!(a, fnv1a(b"abc", FNV_OFFSET), "stable");
    }

    /// F5 regression: durations are out of order and each sample carries
    /// distinct phase timings and `find_hsps` call counts, so any cross-
    /// iteration mispair — a wrong sample's elapsed, stages, or batch count —
    /// fails its line's assertions. Drives the real selection
    /// ([`sorted_sample_indices`]) and serialization ([`write_json_samples`])
    /// with no GPU and no temporary file.
    #[test]
    fn json_records_keep_each_iterations_stages_paired_with_its_elapsed() {
        let mk = |elapsed: f64, stage: &str, stage_ms: u64, find_calls: u32| -> WarmSample {
            let mut p = Phases::new();
            p.add(stage, Duration::from_millis(stage_ms));
            for _ in 0..find_calls {
                p.add_gpu("find_hsps", Duration::from_millis(3), 2.0);
            }
            WarmSample {
                elapsed_ms: elapsed,
                phases: p,
            }
        };
        // Original order 100 / 50 / 75 ms: the 75 ms median is the *last*
        // sample (index 2), not the middle position, so a naive middle-sample
        // selection (50 ms) is caught.
        let warms = vec![
            mk(100.0, "alpha", 20, 1),
            mk(50.0, "beta", 10, 5),
            mk(75.0, "gamma", 30, 9),
        ];
        let order = sorted_sample_indices(&warms);
        assert_eq!(order, vec![1usize, 2, 0]);
        assert_eq!(order[warms.len() / 2], 2, "median is the 75 ms sample");

        // Run-level fields arrive pre-rendered; the helper only appends each
        // iteration's own fields, so this checks the wiring, not metadata.
        let common = "\"variant\":\"bench\",\"workload\":\"hsps\",\"launch_ms\":0.000000,\
                      \"peak_vram_mib\":0.0,\"parity\":{},\"mechanisms\":{},\"environment\":{}";
        let mut out = Vec::new();
        write_json_samples(&mut out, &warms, &Pass::default(), common).unwrap();
        let lines: Vec<&str> = std::str::from_utf8(&out).unwrap().lines().collect();
        assert_eq!(lines.len(), warms.len(), "one record per warm iteration");
        for (i, s) in warms.iter().enumerate() {
            let line = lines[i];
            let stages = s.phases.json_stages();
            assert!(line.contains(&format!("\"iteration\":{}", i + 1)), "{line}");
            assert!(
                line.contains(&format!("\"whole_ms\":{:.4}", s.elapsed_ms)),
                "{line}"
            );
            assert!(line.contains(&format!("\"stages\":{stages}")), "{line}");
            assert!(
                line.contains(&format!("\"batches\":{}", s.phases.calls("find_hsps"))),
                "{line}"
            );
        }
    }
}
