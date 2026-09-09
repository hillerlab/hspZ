// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The `run` command: prepare the reference/query pair, drive the whole Seed +
//! Filter pass, and write the `.segments` files. Also hosts the host-side
//! preparation types that `benchmark` reuses to hold input across iterations.
//!
//! With `--gpus W` the reference bins are distributed over `W` worker threads
//! (deterministic LPT via `plan::assign_bins`), each owning its bins end to
//! end with one context per device; output is replayed in ordinal order, so it
//! never depends on completion order (round 31). `device_seeds_for` selects
//! the device seeder above one worker, `--threads` is the machine-wide budget
//! divided across workers (round 35), and the host-memory preflight gates the
//! run before any CUDA allocation (rounds 32–33).

use crate::Fallible;
use crate::cli::RunArgs;
use crate::gpu::{Engine, EngineConfig, HitStats, Lifecycle};
use crate::hsp::{self, SegmentPair};
use crate::partition::{Partitioner, Plan};
use crate::plan::{self, PackedBin, RecordMeta};
use crate::scoring;
use crate::seed::{self, SeedTable, Shape};
use crate::sequence::{self, Chr, Genome, encode};
use crate::sink::{DirectorySink, OutputSink, TarGzSink};
use crate::timing::{self, Phases};
use cuda_core::CudaContext;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Counts KegAlign prints under `--debug`, plus the pre-dedup count it does not.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub(crate) struct Stats {
    pub(crate) seeds: u64,
    pub(crate) seed_hits: u64,
    pub(crate) raw_hsps: u64,
    pub(crate) hsps: u64,
}

// ---------------------------------------------------------------------------
// Preparation — everything that does not depend on the GPU and is done once

/// Input in the form Seed + Filter consumes, so a benchmark can hold it across
/// iterations instead of re-reading and re-indexing every time.
pub(crate) struct Prepared {
    shape: Shape,
    sub_mat: Vec<i32>,
    pub(crate) reference: Genome,
    pub(crate) query: Genome,
    pub(crate) rc_chrs: Vec<Chr>,
    query_rc: Vec<u8>,
    table: SeedTable,
    intervals: Vec<(u32, u32)>,
    q_block_len: u32,
    transitions: bool,
    plus: bool,
    minus: bool,
    enc_ref: Vec<u8>,
    enc_query: Vec<u8>,
    enc_query_rc: Vec<u8>,
}

/// Reads and indexes both FASTA files and builds the encoded inputs the GPU
/// pipeline consumes. GPU-free, so `benchmark` can hold the result across
/// iterations.
pub(crate) fn prepare(args: &RunArgs, phases: &mut Phases) -> Fallible<Prepared> {
    // Frozen-plan replay/dump is only implemented by the GPU `run` executor
    // (multi-bin planning, fit checks, ordinal replay). `benchmark` and
    // `--cpu-only` both reach the GPU-free path through this function, which
    // has no plan and nothing to dump — so silently accepting the flags here
    // would just drop them on the floor. Reject before either input is read.
    if args.from_manifest.is_some() || args.dump_manifest.is_some() {
        return Err(
            "--from-manifest/--dump-manifest are only supported by `run`, not benchmark or \
             --cpu-only"
                .into(),
        );
    }

    let plus = args.strand == "plus" || args.strand == "both";
    let minus = args.strand == "minus" || args.strand == "both";
    if !plus && !minus {
        return Err(format!("--strand must be plus, minus or both, got {}", args.strand).into());
    }

    let shape = Shape::parse(&args.seed)?;
    let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, args.scoring.as_deref())?;

    let t = Instant::now();
    // PLAN.md §1: reference and query input are timed separately and kept out
    // of `core`, so a format change can never be confused with a core change.
    let query = Genome::load(&args.query, &args.query_prefix, args.seq_block_size)?;
    phases.add("input.query", t.elapsed());
    let t = Instant::now();
    let reference = Genome::load(&args.reference, &args.target_prefix, args.seq_block_size)?;
    phases.add("input.reference", t.elapsed());
    if reference.block_len <= shape.size || query.block_len <= shape.size {
        return Err("reference and query blocks must be longer than the seed".into());
    }

    let t = Instant::now();
    let (query_rc, rc_chrs) = query.reverse_complement();
    let enc_ref = encode(&reference.buf[..reference.block_len]);
    let enc_query = encode(&query.buf[..query.block_len]);
    let enc_query_rc = encode(&query_rc);
    phases.add("revcomp + encode", t.elapsed());

    let t = Instant::now();
    // PLAN.md M7/M9.1: the reference index is the largest CPU stage (10.1% of a
    // chr1 run), and it uses the one existing --threads budget rather than a
    // knob of its own.
    let table = SeedTable::build_parallel(
        &reference.buf[..reference.block_len],
        &shape,
        args.step,
        resolve_threads(args.threads),
    );
    phases.add("seed table build", t.elapsed());

    let q_block_len = (query.block_len - shape.size) as u32;
    let intervals = sequence::intervals(query.block_len, shape.size, args.lastz_interval_size);

    Ok(Prepared {
        shape,
        sub_mat,
        reference,
        query,
        rc_chrs,
        query_rc,
        table,
        intervals,
        q_block_len,
        transitions: !args.notransition,
        plus,
        minus,
        enc_ref,
        enc_query,
        enc_query_rc,
    })
}

impl Prepared {
    /// The seed shape and transition setting every pass shares.
    pub(crate) fn seeding(&self) -> (&Shape, bool) {
        (&self.shape, self.transitions)
    }

    /// This input as the single query bin of a 1x1 plan (AM-B2).
    pub(crate) fn query_pass(&self) -> QueryPass<'_> {
        QueryPass {
            fwd: self.enc_query_source(),
            rc: &self.query_rc,
            intervals: &self.intervals,
            q_block_len: self.q_block_len,
        }
    }

    /// The encoded query strands `Engine::swap_query` consumes (AM-B1).
    ///
    /// A single-block run is a 1x1 plan, so this is that plan's only query bin.
    pub(crate) fn encoded_query(&self) -> (&[u8], &[u8]) {
        (&self.enc_query, &self.enc_query_rc)
    }

    /// The device-facing configuration: tables, sequences, and the tuning
    /// knobs that land in kernel constants.
    pub(crate) fn engine_config<'a>(
        &'a self,
        args: &RunArgs,
        contract: &crate::gpu::ExecutionContract,
    ) -> EngineConfig<'a> {
        EngineConfig {
            index_table: &self.table.index_table,
            pos_table: &self.table.pos_table,
            ref_seq: &self.enc_ref,
            sub_mat: &self.sub_mat,
            seed_size: self.shape.size as u32,
            xdrop: args.xdrop,
            hspthresh: args.hspthresh,
            noentropy: args.noentropy,
            max_hits: contract.max_hits,
            hit_capacity: contract.hit_capacity,
            timing: args.time,
            hsp_blocks: contract.hsp_blocks,
        }
    }

    /// Single-bin plan for this prepared pair, for physical capacity sizing.
    /// Builds `RecordMeta` from the resident `chrs` names/lengths and bins once
    /// at `u64::MAX`; no new genome reads or packing. Semantic plan unchanged.
    pub(crate) fn capacity_plan(&self) -> plan::Plan {
        let ref_meta: Vec<plan::RecordMeta> = self
            .reference
            .chrs
            .iter()
            .enumerate()
            .map(|(i, c)| plan::RecordMeta {
                id: i as u32,
                name: c.name.clone(),
                len: u64::from(c.len),
                ordinal: i as u32,
            })
            .collect();
        let qry_meta: Vec<plan::RecordMeta> = self
            .query
            .chrs
            .iter()
            .enumerate()
            .map(|(i, c)| plan::RecordMeta {
                id: i as u32,
                name: c.name.clone(),
                len: u64::from(c.len),
                ordinal: i as u32,
            })
            .collect();
        plan::plan(&ref_meta, &qry_meta, u64::MAX)
    }
}

// ---------------------------------------------------------------------------
// The Seed + Filter pass itself

#[derive(Default)]
pub(crate) struct Pass {
    pub(crate) stats: Stats,
    /// Per interval: (plus HSPs, minus HSPs).
    pub(crate) intervals: Vec<(Vec<SegmentPair>, Vec<SegmentPair>)>,
    raw: Vec<(char, Vec<SegmentPair>)>,
    /// Env-gated AL3 input; production runs keep this empty.
    audit: Vec<(char, crate::census::AcceptedHsp)>,
}

/// One `seed_and_filter` call: a wga_chunk of one strand of one interval.
///
/// The interval/strand/chunk nest is flattened into a flat list (PLAN.md §3) so
/// the seed worker can always run exactly one batch ahead, including across
/// strand and interval boundaries. The order is identical to the original
/// nesting — plus strand then minus strand, chunks ascending — because that
/// order fixes the `MAX_HITS` chunking and therefore the final HSP set.
struct Batch {
    interval: usize,
    rev: bool,
    range: (u32, u32),
}

/// Host staging for one batch's seeds (PLAN.md N1).
///
/// The GPU path only ever sees `&[u64]`, so whether the pages are pinned is
/// invisible to it — that is what makes N1 a buffer-placement experiment rather
/// than a pipeline change. Pinned slots are allocated once at the worst-case
/// size so the seed worker never allocates mid-pass.
enum SeedSlot {
    Paged(Vec<u64>),
    Pinned {
        buf: cuda_core::PinnedHostBuffer<u64>,
        len: usize,
    },
}

impl SeedSlot {
    /// Reclaims the pinned buffer so the engine can keep it for the next pass.
    fn into_pinned(self) -> Option<cuda_core::PinnedHostBuffer<u64>> {
        match self {
            SeedSlot::Pinned { buf, .. } => Some(buf),
            SeedSlot::Paged(_) => None,
        }
    }

    fn seeds(&self) -> &[u64] {
        match self {
            SeedSlot::Paged(v) => v,
            SeedSlot::Pinned { buf, len } => &buf.as_slice()[..*len],
        }
    }

    /// Concatenates the per-worker pieces into this slot. Identical work in both
    /// variants — the `Vec` path is the concatenation `chunk_seeds_parallel`
    /// already did, so pinning adds no copy.
    fn fill(&mut self, parts: &[Vec<u64>]) {
        let total: usize = parts.iter().map(Vec::len).sum();
        match self {
            SeedSlot::Paged(v) => {
                v.clear();
                v.resize(total, 0);
                seed::concat_parts(parts, v);
            }
            SeedSlot::Pinned { buf, len } => {
                *len = seed::concat_parts(parts, buf.as_mut_slice());
                debug_assert_eq!(*len, total);
            }
        }
    }
}

/// Runs every Seed + Filter batch for one prepared pair, overlapping seed
/// generation with GPU work across two host slots.
/// Everything one (reference, query-bin) pass needs from the query side
/// (PLAN.md §3 / AM-B2).
///
/// Seeding reads **raw** bytes, not the device alphabet: `fwd` is the raw forward
/// block and `rc` the raw reverse complement. `Engine::swap_query` handles the
/// *encoded* device buffers separately, so a pass needs both halves supplied and
/// the two must describe the same block.
///
/// `intervals` and `q_block_len` are per query bin. A multi-bin executor derives
/// them from that bin's own `block_len`; reusing whole-genome intervals here would
/// silently seed the wrong ranges, which is the trap AM-B2 names.
pub(crate) struct QueryPass<'a> {
    pub fwd: &'a [u8],
    pub rc: &'a [u8],
    pub intervals: &'a [(u32, u32)],
    pub q_block_len: u32,
}

pub(crate) fn seed_and_filter_all(
    engine: &mut Engine,
    q: &QueryPass<'_>,
    shape: &Shape,
    transitions: bool,
    args: &RunArgs,
    threads: usize,
) -> Fallible<Pass> {
    let plus = args.strand == "plus" || args.strand == "both";
    let minus = args.strand == "minus" || args.strand == "both";
    let mut pass = Pass::default();

    let mut batches = Vec::new();
    for (i, &(start, end)) in q.intervals.iter().enumerate() {
        if plus {
            for range in seed::chunks(start, end, args.wga_chunk_size) {
                batches.push(Batch {
                    interval: i,
                    rev: false,
                    range,
                });
            }
        }
        if minus {
            let r = (q.q_block_len - end, q.q_block_len - start);
            for range in seed::chunks(r.0, r.1, args.wga_chunk_size) {
                batches.push(Batch {
                    interval: i,
                    rev: true,
                    range,
                });
            }
        }
    }

    let mut out: Vec<(Vec<SegmentPair>, Vec<SegmentPair>)> =
        vec![(Vec::new(), Vec::new()); q.intervals.len()];

    // Round 69: chosen at runtime, not at compile time. The device seeder wins on
    // several GPUs and loses on one, so one binary has to be able to do both.
    if engine.device_seeds {
        let _ = (q.fwd, q.rc, threads);
        engine.async_seed_copy = false;
        for b in &batches {
            let n_seeds = engine.generate_seeds(0, b.rev, b.range, shape, transitions)?;
            #[cfg(feature = "device-seeds-check")]
            {
                let seq = if b.rev { q.rc } else { q.fwd };
                let expected = seed::chunk_seeds(seq, shape, transitions, b.range);
                engine.check_seed_bytes(0, &expected)?;
            }
            if n_seeds > 0 {
                pass.stats.seeds += n_seeds as u64;
                let o = engine.seed_and_filter(0, b.rev).map_err(|e| {
                    format!(
                        "interval {} range {}-{} rev {}: {e}",
                        b.interval, b.range.0, b.range.1, b.rev
                    )
                })?;
                pass.stats.seed_hits += o.num_hits as u64;
                pass.stats.raw_hsps += o.raw_hsps as u64;
                let dst = &mut out[b.interval];
                if b.rev {
                    dst.1.extend_from_slice(&o.hsps);
                } else {
                    dst.0.extend_from_slice(&o.hsps);
                }
                if engine.dump_raw {
                    pass.raw.push((if b.rev { '-' } else { '+' }, o.raw));
                }
                pass.audit.extend(
                    o.audit
                        .into_iter()
                        .map(|h| (if b.rev { '-' } else { '+' }, h)),
                );
            }
        }
        for (fw, rc) in out {
            pass.stats.hsps += (fw.len() + rc.len()) as u64;
            pass.intervals.push((fw, rc));
        }
        return Ok(pass);
    }

    {
        let fwd = q.fwd;
        let rc = q.rc;
        // Identical call to the one the serial loop made — PLAN.md §3 forbids
        // touching the seed-generation algorithm in this experiment, so the seed
        // sequence stays bit-identical.
        let parts = |b: &Batch| -> Vec<Vec<u64>> {
            let seq: &[u8] = if b.rev { rc } else { fwd };
            seed::chunk_seeds_parts(seq, shape, transitions, b.range, threads)
        };

        // N1: two host slots, allocated once. The pinned pair is sized from the
        // widest chunk any batch will ask for, so the worker never allocates.
        let widest = batches
            .iter()
            .map(|b| b.range.1 - b.range.0)
            .max()
            .unwrap_or(0);
        let cap = seed::max_seeds(widest, shape, transitions);
        // Pinned staging is the default, but `cuMemHostAlloc` is not universally
        // available — ZLUDA returns DriverError(801) for it at any size — so a
        // refusal falls back to a pageable `Vec` rather than failing the run.
        let new_slot = |engine: &mut Engine| -> SeedSlot {
            if args.no_pinned_seeds {
                return SeedSlot::Paged(Vec::new());
            }
            match engine.take_pinned(cap) {
                Ok(buf) => SeedSlot::Pinned { buf, len: 0 },
                Err(_) => SeedSlot::Paged(Vec::new()),
            }
        };
        // Phase 3: how many batches the host runs ahead of the one computing.
        //
        // One is enough to hide seed *generation* behind GPU work (N1). Overlapping
        // the *upload* needs two: batch N+1's seeds must already be in host memory
        // when batch N's kernels are enqueued, or there is no compute left for the DMA
        // to hide behind — which is exactly why round 15's same-stream async copy
        // bought nothing. Slots: one being consumed, one being uploaded, one being
        // generated.
        //
        // Two things turn the overlap off, both because it cannot work without them:
        // pageable staging (an async copy from unpinned memory blocks until staged, so
        // ZLUDA never overlaps), and reallocated seed buffers (an upload in flight into
        // a freed buffer is AM-B's use-after-free).
        let first = new_slot(engine);
        let overlap = !args.no_async_seed_copy
            && !args.no_persistent_seed_buffers
            && matches!(first, SeedSlot::Pinned { .. });
        engine.async_seed_copy = overlap;
        let lead = if overlap { 2 } else { 1 };
        let nslots = lead + 1;
        let mut ring: Vec<SeedSlot> = std::iter::once(first)
            .chain((1..nslots).map(|_| new_slot(engine)))
            .collect();

        // Standalone generation time summed across workers, versus the part of it
        // that the GPU could not cover. `exposed` is measured as the time the main
        // thread actually blocks in `join` after its own GPU work finished, which
        // is exactly `max(0, seed_end[N+1] - gpu_end[N])` (PLAN.md §4).
        let (mut standalone, mut exposed) = (Duration::ZERO, Duration::ZERO);

        std::thread::scope(|scope| -> Fallible<()> {
            if batches.is_empty() {
                return Ok(());
            }
            // The first `lead` batches have no GPU work to hide behind; they are
            // exposed by construction and are the only ones that must be.
            for (i, b) in batches.iter().enumerate().take(lead) {
                let t = Instant::now();
                ring[i % nslots].fill(&parts(b));
                standalone += t.elapsed();
                exposed += t.elapsed();
            }
            // Batch 0's upload, so the loop can always be one upload ahead.
            if overlap {
                engine.upload_seeds(0, ring[0].seeds())?;
            }

            for i in 0..batches.len() {
                // Issue batch N+1's upload before touching the GPU: it runs on the
                // copy stream while this batch's kernels run on the compute stream.
                if overlap {
                    if batches.get(i + 1).is_some() {
                        engine.upload_seeds((i + 1) % 2, ring[(i + 1) % nslots].seeds())?;
                    }
                } else {
                    engine.upload_seeds(0, ring[i % nslots].seeds())?;
                }

                // Hand batch N+lead to a worker before touching the GPU, so it runs
                // while the main thread is blocked on this batch's kernels. The slot
                // moves into the worker and comes back filled, which keeps exactly
                // `nslots` host buffers alive without borrowing one across the scope.
                //
                // Reuse is safe without an extra wait: this slot last held batch
                // N-1 (or older), whose compute has finished, and whose compute could
                // only finish after its upload completed.
                let worker = batches.get(i + lead).map(|next| {
                    let k = (i + lead) % nslots;
                    let mut slot = std::mem::replace(&mut ring[k], SeedSlot::Paged(Vec::new()));
                    let handle = scope.spawn(move || {
                        let t = Instant::now();
                        slot.fill(&parts(next));
                        (slot, t.elapsed())
                    });
                    (k, handle)
                });

                let b = &batches[i];
                let n_seeds = ring[i % nslots].seeds().len();
                if n_seeds > 0 {
                    pass.stats.seeds += n_seeds as u64;
                    let o = engine
                        .seed_and_filter(if overlap { i % 2 } else { 0 }, b.rev)
                        .map_err(|e| {
                            format!(
                                "interval {} range {}-{} rev {}: {e}",
                                b.interval, b.range.0, b.range.1, b.rev
                            )
                        })?;
                    pass.stats.seed_hits += o.num_hits as u64;
                    pass.stats.raw_hsps += o.raw_hsps as u64;
                    let dst = &mut out[b.interval];
                    if b.rev {
                        dst.1.extend_from_slice(&o.hsps)
                    } else {
                        dst.0.extend_from_slice(&o.hsps)
                    };
                    if engine.dump_raw {
                        pass.raw.push((if b.rev { '-' } else { '+' }, o.raw));
                    }
                    pass.audit.extend(
                        o.audit
                            .into_iter()
                            .map(|h| (if b.rev { '-' } else { '+' }, h)),
                    );
                }

                if let Some((k, worker)) = worker {
                    let t = Instant::now();
                    let (slot, dur) = worker.join().expect("seed worker panicked");
                    exposed += t.elapsed();
                    standalone += dur;
                    ring[k] = slot;
                }
            }
            Ok(())
        })?;

        // Give the pinned buffers back so the next pass reuses them rather than
        // paying cuMemHostAlloc again (PLAN.md N1).
        for slot in ring {
            if let Some(buf) = slot.into_pinned() {
                engine.give_pinned(buf);
            }
        }

        engine.phases.add("seed generation (exposed)", exposed);
        engine
            .phases
            .add_overlapped("seed generation (standalone)", standalone);

        for (fw, rc) in out {
            pass.stats.hsps += (fw.len() + rc.len()) as u64;
            pass.intervals.push((fw, rc));
        }
        Ok(pass)
    }
}

impl Prepared {
    /// Seeding reads the *raw* query bytes, not the device encoding.
    fn enc_query_source(&self) -> &[u8] {
        &self.query.buf[..self.query.block_len]
    }
}

/// Resolves `-Z`'s optional path: bare `-Z` (clap's `-` sentinel, since empty
/// strings are rejected) derives `<output>.tar.gz`.
fn tarball_path(args: &RunArgs) -> Option<PathBuf> {
    args.tarball.as_ref().map(|p| {
        if p.as_os_str().is_empty() || p.as_os_str() == "-" {
            let mut d = args.output.clone();
            d.as_mut_os_string().push(".tar.gz");
            d
        } else {
            p.clone()
        }
    })
}

/// Formats and emits every logical output file (PLAN.md §9, §17, §19).
///
/// What a `write_outputs` call produced, for the benchmark's per-iteration
/// `-D`/`-Z` accounting and the `--time` report.
#[derive(Default)]
pub(crate) struct OutputReport {
    pub partition_ms: f64,
    pub format_ms: f64,
    pub archive_ms: f64,
    pub files: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// One output pass: a sink and a `Partitioner` hoisted out of the old
/// `write_outputs` so the multi-bin executor emits every work unit into the
/// same archive (`-Z`) and shares one `-D` history (PLAN.md §9.10 / AM-A4).
pub(crate) struct Emitter {
    sink: Box<dyn OutputSink>,
    part: Partitioner,
    diagonal: bool,
    partition_ms: f64,
    format_ms: f64,
    archive_ms: f64,
    files: usize,
    bytes_in: u64,
    audit: Option<BufWriter<std::fs::File>>,
}

impl Emitter {
    pub(crate) fn new(args: &RunArgs) -> Fallible<Self> {
        let sink: Box<dyn OutputSink> = match tarball_path(args) {
            Some(path) => Box::new(TarGzSink::new(&path)?),
            None => Box::new(DirectorySink::new(&args.output)?),
        };
        let mut audit = crate::census::SurvivorAudit::dump_path()
            .map(|path| std::fs::File::create(path).map(BufWriter::new))
            .transpose()?;
        if let Some(out) = audit.as_mut() {
            writeln!(
                out,
                "class\treference\tquery\tstrand\tref_start\tquery_start\tspan\tscore\tframe"
            )?;
        }
        Ok(Emitter {
            sink,
            part: Partitioner::default(),
            diagonal: args.diagonal_partition,
            partition_ms: 0.0,
            format_ms: 0.0,
            archive_ms: 0.0,
            files: 0,
            bytes_in: 0,
            audit,
        })
    }

    /// Emits every logical file for one reference-bin × query-bin work unit.
    /// `ref_bin`/`query_bin` become KegAlign's block indices in the filename;
    /// coordinates stay chromosome-relative via the bin-local `Chr` tables.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_unit(
        &mut self,
        ref_bin: u32,
        query_bin: u32,
        ref_chrs: &[Chr],
        query_chrs: &[Chr],
        rc_chrs: &[Chr],
        pass: &Pass,
    ) -> Fallible<()> {
        if let Some(out) = self.audit.as_mut() {
            for &(strand, accepted) in &pass.audit {
                let q_chrs = if strand == '-' { rc_chrs } else { query_chrs };
                let rec = hsp::record(&accepted.hsp, ref_chrs, q_chrs);
                writeln!(
                    out,
                    "{}\t{}\t{}\t{strand}\t{}\t{}\t{}\t{}\toriented",
                    if accepted.common { "common" } else { "rare" },
                    ref_chrs[rec.r_chr as usize].name,
                    q_chrs[rec.q_chr as usize].name,
                    rec.r_start - 1,
                    rec.q_start - 1,
                    accepted.hsp.len as usize + 1,
                    rec.score,
                )?;
            }
        }
        for (n, (fw, rc)) in pass.intervals.iter().enumerate() {
            // `segment_printer.cpp` names files by 1-based interval index,
            // query block index and reference block index (PLAN.md §5 / AM-A3).
            let base = format!("tmp{}.block{}.r{}", n + 1, query_bin, ref_bin);
            for (hsps, q_chrs, strand) in [(fw, query_chrs, '+'), (rc, rc_chrs, '-')] {
                if hsps.is_empty() {
                    continue;
                }
                let t = Instant::now();
                let recs = hsp::records(hsps, ref_chrs, q_chrs, strand);
                let plan = if self.diagonal {
                    self.part.plan(recs, strand)
                } else {
                    Plan::Whole(recs)
                };
                self.partition_ms += t.elapsed().as_secs_f64() * 1000.0;

                let stem = format!("{base}.{}", if strand == '-' { "minus" } else { "plus" });
                // One counter per logical file, pre-incremented, so names start
                // at `.split1` and cover both per-pair splits and skip aggregates.
                let mut ctr = 0usize;
                match plan {
                    Plan::Whole(recs) => {
                        self.files += 1;
                        self.emit_file(
                            ref_chrs,
                            q_chrs,
                            strand,
                            &recs,
                            &format!("{stem}.segments"),
                        )?;
                    }
                    Plan::Split(parts) => {
                        for chunk in &parts {
                            ctr += 1;
                            self.files += 1;
                            self.emit_file(
                                ref_chrs,
                                q_chrs,
                                strand,
                                chunk,
                                &format!("{stem}.split{ctr}.segments"),
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_file(
        &mut self,
        ref_chrs: &[Chr],
        q_chrs: &[Chr],
        strand: char,
        recs: &[hsp::Record],
        name: &str,
    ) -> Fallible<()> {
        let t = Instant::now();
        let text = hsp::render_records(recs, ref_chrs, q_chrs, strand);
        self.format_ms += t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        self.sink.write_entry(name, text.as_bytes())?;
        self.archive_ms += t.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    pub(crate) fn finish(mut self, phases: &mut Phases) -> Fallible<OutputReport> {
        if let Some(out) = self.audit.as_mut() {
            out.flush()?;
        }
        self.bytes_in = self.sink.bytes_in();
        let t = Instant::now();
        let bytes_out = self.sink.bytes_out().unwrap_or(0);
        Box::new(self.sink).finish()?;
        self.archive_ms += t.elapsed().as_secs_f64() * 1000.0;
        phases.add_ms("partition", self.partition_ms);
        phases.add_ms("format", self.format_ms);
        phases.add_ms("archive", self.archive_ms);
        Ok(OutputReport {
            partition_ms: self.partition_ms,
            format_ms: self.format_ms,
            archive_ms: self.archive_ms,
            files: self.files,
            bytes_in: self.bytes_in,
            bytes_out,
        })
    }
}

/// Builds the planner's metadata from raw records: `id` and `ordinal` are the
/// input index, so `bin.record_ids` indexes straight back into `records`.
fn record_meta(records: &[(String, Vec<u8>)]) -> Vec<RecordMeta> {
    records
        .iter()
        .enumerate()
        .map(|(i, (name, seq))| RecordMeta {
            id: i as u32,
            name: name.clone(),
            len: seq.len() as u64,
            ordinal: i as u32,
        })
        .collect()
}

/// Single-block convenience for the benchmark's output-mode timing path. A
/// single block is a 1×1 plan, so this emits one unit with bin ids 0/0 and
/// reuses the executor's `Emitter` (PLAN.md §9: one output path, not two).
pub(crate) fn write_outputs(
    args: &RunArgs,
    p: &Prepared,
    pass: &Pass,
    phases: &mut Phases,
) -> Fallible<OutputReport> {
    let mut emitter = Emitter::new(args)?;
    emitter.emit_unit(0, 0, &p.reference.chrs, &p.query.chrs, &p.rc_chrs, pass)?;
    emitter.finish(phases)
}

// ---------------------------------------------------------------------------
// Multi-GPU execution (Phase 5)

/// One finished work unit on its way to the emitter (§19).
///
/// Workers complete units in whatever order their GPU gets to them; the emitter
/// replays them by ordinal, so `-D` history, file names and tar entry order never
/// depend on completion order.
struct UnitOutput {
    ordinal: u32,
    reference_bin: u32,
    query_bin: u32,
    ref_chrs: Vec<Chr>,
    query_chrs: Vec<Chr>,
    rc_chrs: Vec<Chr>,
    pass: Pass,
}

/// What one worker reports at join. Everything the serial executor used to
/// accumulate inline, now per worker and summed by the caller.
#[derive(Default)]
struct WorkerReport {
    stats: Stats,
    phases: Phases,
    lifecycle: Lifecycle,
    launches: u64,
    stage_syncs: u64,
    pipeline_syncs: u64,
    uploads: u64,
    copy_stalls: u64,
    seed_table_ms: Duration,
    /// Round 72: GPU-timeline idle between stages, summed for this worker. Each worker
    /// drives one GPU, so this is that GPU's idle and therefore the per-GPU ceiling on
    /// any batch-level pipelining.
    gap_ms: f32,
    gap_n: u64,
    prefetched_ms: Duration,
    hit_stats: HitStats,
    audit: Option<crate::census::SurvivorAudit>,
    peak_used: usize,
}

/// `sub_mat` is the actual resolved 64-cell matrix (`scoring::build_sub_mat`
/// on this command line), compared by value: two different `--ambiguous`/
/// `--scoring` invocations that resolve to the same matrix must be allowed to
/// replay, so the text itself is never compared. Strand and both prefixes are
/// frozen too — they land in the emitted record names (`sequence::pack`'s
/// `prefix` argument), not the output filenames, so a mismatch would silently
/// change `.segments` byte content rather than fail loudly. `--seq-block-size`/
/// `--query-block-size` are deliberately not compared: the frozen plan's bin
/// membership is authoritative regardless of what this command line's B/Q
/// defaults would have produced.
fn check_manifest_params(
    m: &plan::PlanManifest,
    args: &RunArgs,
    sub_mat: &[i32],
) -> Result<(), String> {
    let mut bad = Vec::new();
    if m.seed != args.seed {
        bad.push(format!("seed {} vs {}", m.seed, args.seed));
    }
    if m.step != args.step {
        bad.push(format!("step {} vs {}", m.step, args.step));
    }
    if m.transitions != !args.notransition {
        bad.push("transitions".into());
    }
    if m.xdrop != args.xdrop {
        bad.push(format!("xdrop {} vs {}", m.xdrop, args.xdrop));
    }
    if m.hspthresh != args.hspthresh {
        bad.push(format!("hspthresh {} vs {}", m.hspthresh, args.hspthresh));
    }
    if m.noentropy != args.noentropy {
        bad.push("noentropy".into());
    }
    if m.wga_chunk_size != args.wga_chunk_size {
        bad.push("wga_chunk_size".into());
    }
    if m.lastz_interval_size != args.lastz_interval_size {
        bad.push("lastz_interval_size".into());
    }
    if m.kegalign_bins != args.kegalign_bins {
        bad.push("kegalign_bins".into());
    }
    if m.sub_mat.as_slice() != sub_mat {
        bad.push("substitution matrix".into());
    }
    if m.strand != args.strand {
        bad.push(format!("strand {} vs {}", m.strand, args.strand));
    }
    if m.target_prefix != args.target_prefix {
        bad.push(format!(
            "target_prefix {:?} vs {:?}",
            m.target_prefix, args.target_prefix
        ));
    }
    if m.query_prefix != args.query_prefix {
        bad.push(format!(
            "query_prefix {:?} vs {:?}",
            m.query_prefix, args.query_prefix
        ));
    }
    // A nonzero CLI cap is an explicit pin and must agree with the frozen
    // plan; `0` (unset) silently adopts the manifest's resolved cap via
    // `ExecutionContract::from_resolved`, never a fresh device derivation.
    if args.max_hits > 0 && args.max_hits != m.max_hits {
        bad.push(format!(
            "max_hits {} vs manifest {}",
            args.max_hits, m.max_hits
        ));
    }
    if args.hsp_blocks > 0 && args.hsp_blocks != m.hsp_blocks {
        bad.push(format!(
            "hsp_blocks {} vs manifest {}",
            args.hsp_blocks, m.hsp_blocks
        ));
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "manifest parameters do not match this command line: {}",
            bad.join(", ")
        ))
    }
}

/// Whether to generate the query seed stream on the device (round 69).
///
/// The trade reverses with GPU count. On one GPU the device seeder adds ~165 s of
/// device work against a host tail that pinned async H->D already hides (r39, r51); on
/// two T4s that tail is 397 s and exposed, and moving it is worth -20.3% of wall (r68).
/// So the worker count is the whole decision. `HSPZ_DEVICE_SEEDS=0|1` overrides it,
/// which is how the device path gets exercised on a one-GPU box.
pub fn device_seeds_for(workers: usize) -> bool {
    match std::env::var("HSPZ_DEVICE_SEEDS").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => workers > 1,
    }
}

/// Runs one worker's reference bins on `device`, streaming finished units to the
/// emitter (§Phase 5: build/upload each reference once, reuse it across its
/// queries; no GPU is shared for performance).
///
/// This is the serial executor, parameterised by which bins it owns: with one
/// worker it is exactly the old path, which is what makes `serial == multi-GPU`
/// (§20) a property of the assignment rather than of two code paths.
#[allow(clippy::too_many_arguments)]
fn run_bins(
    device: usize,
    bins: &[u32],
    plan: &plan::Plan,
    ref_records: &[(String, Vec<u8>)],
    qry_records: &[(String, Vec<u8>)],
    shape: &Shape,
    sub_mat: &[i32],
    transitions: bool,
    args: &RunArgs,
    prefetch: bool,
    threads: usize,
    // Round 69: the device seeder is worth -20.3% of 2-GPU wall and a loss on one GPU,
    // so the executor decides per run rather than the build deciding once.
    device_seeds: bool,
    contract: crate::gpu::ExecutionContract,
    tx: &std::sync::mpsc::SyncSender<UnitOutput>,
) -> Fallible<WorkerReport> {
    let mut rep = WorkerReport::default();
    if bins.is_empty() {
        return Ok(rep);
    }
    let t = Instant::now();
    let ctx = CudaContext::new(device)?;
    rep.phases.add("CUDA context init", t.elapsed());

    // A reference bin's pack + seed table is host work the GPU cannot hide: it
    // all happens before that bin's first kernel (3.0 s of a 196 s L4 multi5 run
    // on 16 CPUs, 15.4 s of the same run on 4). Build bin k+1 on a worker thread
    // while bin k's work units run, so only the first build stays exposed.
    // Execution order, bin identity and the lifecycle counts are untouched: this
    // moves *when* the data is built, not what runs or in which order.
    let build_ref_bin = |rbin: &plan::Bin| -> (PackedBin, SeedTable) {
        let packed = PackedBin::build(
            rbin.record_ids.iter().map(|&id| {
                let (n, s) = &ref_records[id as usize];
                (n.as_str(), s.as_slice())
            }),
            &args.target_prefix,
            false,
        );
        let table =
            SeedTable::build_parallel(&packed.buf[..packed.block_len], shape, args.step, threads);
        (packed, table)
    };
    let mut pending: Option<(PackedBin, SeedTable)> = None;

    for (bin_index, bin_id) in bins.iter().enumerate() {
        let rbin = &plan.reference_bins[*bin_id as usize];
        let t = Instant::now();
        let (mut packed_ref, table) = match pending.take() {
            Some(built) => built,
            None => build_ref_bin(rbin),
        };
        rep.seed_table_ms += t.elapsed();
        rep.lifecycle.seed_table_builds += 1;

        let cfg = EngineConfig {
            index_table: &table.index_table,
            pos_table: &table.pos_table,
            ref_seq: &packed_ref.enc,
            sub_mat,
            seed_size: shape.size as u32,
            xdrop: args.xdrop,
            hspthresh: args.hspthresh,
            noentropy: args.noentropy,
            // Resolved once for the run (device 0 or the frozen manifest). Not
            // re-derived from this worker's device.
            max_hits: contract.max_hits,
            hit_capacity: contract.hit_capacity,
            timing: args.time,
            hsp_blocks: contract.hsp_blocks,
        };
        let mut engine = Engine::new(&ctx, cfg, &mut rep.phases)?;
        rep.lifecycle.engine_creations += 1;
        engine.dump_raw = args.dump_raw.is_some();
        engine.device_seeds = device_seeds;
        engine.collect_hit_stats = args.hit_stats;
        engine.persistent_seed_buffers = !args.no_persistent_seed_buffers;
        engine.async_stages = !args.no_async_stages;

        // The device owns the tables and the encoded reference now, and nothing
        // host-side reads them again — only the bin-local `Chr` table is still
        // needed, for coordinate mapping at emit. Release the rest before the GPU
        // phase: `pos_table` alone is 4 B/bp (300 MB for a 75 Mbp bin), and it is
        // what the prefetch would otherwise hold twice.
        let ref_chrs = std::mem::take(&mut packed_ref.chrs);
        drop(packed_ref);
        drop(table);

        // The next bin *this worker owns* rides along with this bin's GPU work.
        let next_bin = bins
            .get(bin_index + 1)
            .map(|id| &plan.reference_bins[*id as usize])
            .filter(|_| prefetch);
        std::thread::scope(|scope| -> Fallible<()> {
            let prefetch = next_bin.map(|nb| {
                scope.spawn(|| {
                    let t = Instant::now();
                    (build_ref_bin(nb), t.elapsed())
                })
            });

            for unit in plan.units.iter().filter(|u| u.reference_bin == rbin.id) {
                let qbin = &plan.query_bins[unit.query_bin as usize];
                let t = Instant::now();
                let packed_q = PackedBin::build(
                    qbin.record_ids.iter().map(|&id| {
                        let (n, s) = &qry_records[id as usize];
                        (n.as_str(), s.as_slice())
                    }),
                    &args.query_prefix,
                    true,
                );
                rep.phases.add("query pack", t.elapsed());
                // Per-bin intervals + q_block_len (AM-B2). `intervals` is empty for a
                // block <= seed, and `q_block_len` is then never read; saturating
                // avoids the underflow the single-block path guards with an error.
                let intervals =
                    sequence::intervals(packed_q.block_len, shape.size, args.lastz_interval_size);
                let q_block_len = packed_q.block_len.saturating_sub(shape.size) as u32;

                let t = Instant::now();
                engine.swap_query(&packed_q.enc, &packed_q.enc_rc)?;
                rep.phases.add("swap_query", t.elapsed());
                let qpass = QueryPass {
                    fwd: &packed_q.buf[..packed_q.block_len],
                    rc: &packed_q.rc,
                    intervals: &intervals,
                    q_block_len,
                };
                let pass =
                    seed_and_filter_all(&mut engine, &qpass, shape, transitions, args, threads)
                        .map_err(|e| {
                            format!(
                                "unit {} ref_bin {} query_bin {}: {e}",
                                unit.ordinal, rbin.id, qbin.id
                            )
                        })?;

                rep.stats.seeds += pass.stats.seeds;
                rep.stats.seed_hits += pass.stats.seed_hits;
                rep.stats.raw_hsps += pass.stats.raw_hsps;
                rep.stats.hsps += pass.stats.hsps;
                rep.lifecycle.work_units_executed += 1;

                // A dead emitter means the run is already failing; propagate rather
                // than block forever on a channel nobody drains.
                tx.send(UnitOutput {
                    ordinal: unit.ordinal,
                    reference_bin: rbin.id,
                    query_bin: qbin.id,
                    ref_chrs: ref_chrs.clone(),
                    query_chrs: packed_q.chrs.clone(),
                    rc_chrs: packed_q.rc_chrs.clone(),
                    pass,
                })
                .map_err(|_| "emitter stopped receiving work units")?;
            }

            // Whatever is left of the prefetch after the GPU work is the exposed
            // part, and it is charged to `seed table build` like an inline build.
            if let Some(prefetch) = prefetch {
                let t = Instant::now();
                let (built, standalone) = prefetch.join().expect("reference prefetch panicked");
                rep.seed_table_ms += t.elapsed();
                rep.prefetched_ms += standalone;
                pending = Some(built);
            }
            Ok(())
        })?;

        rep.lifecycle.reference_uploads += engine.reference_uploads();
        rep.lifecycle.query_swaps += engine.query_swaps();
        {
            let (g, n, _, _) = engine.stage_gaps();
            rep.gap_ms += g;
            rep.gap_n += n;
        }
        rep.launches += engine.launches;
        rep.stage_syncs += engine.stage_syncs();
        rep.pipeline_syncs += engine.pipeline_syncs();
        let (u, st) = engine.seed_copy_stats();
        rep.uploads += u;
        rep.copy_stalls += st;
        rep.peak_used = rep.peak_used.max(engine.peak_used);
        if args.hit_stats {
            rep.hit_stats.merge(&engine.hit_stats);
        }
        if let Some(c) = engine.census.as_mut() {
            eprintln!("\n(reference bin) {}", c.report());
            rep.audit
                .get_or_insert_with(crate::census::SurvivorAudit::default)
                .merge(c);
        }
        rep.phases.merge(&engine.phases);
        #[cfg(feature = "counters")]
        {
            eprintln!(
                "\nFIND_HSPS COUNTERS (reference bin {})\n{}",
                rbin.id,
                engine.hsp_stats.report()
            );
            eprintln!(
                "\nRAW HSP PROVENANCE (reference bin {})\n{}",
                rbin.id,
                engine.groups.report()
            );
            let share = std::env::var("HSPZ_FIND_HSPS_SHARE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.55);
            eprintln!(
                "\nDIAGONAL STRUCTURE / REPEATED WORK\n{}",
                engine.hsp_stats.diagonal_report(share)
            );
        }
    }
    Ok(rep)
}

// ---------------------------------------------------------------------------
// run

pub(crate) fn run(args: &RunArgs, pre_main_ms: f64, started: Instant) -> Fallible<Stats> {
    let mut phases = Phases::new();
    phases.add_ms("process startup", pre_main_ms);

    if args.cpu_only {
        return run_cpu_only(args, &mut phases, pre_main_ms, started);
    }

    // Shared config, parsed once.
    let shape = Shape::parse(&args.seed)?;
    let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, args.scoring.as_deref())?;
    let transitions = !args.notransition;

    // Load both sides as raw records — no block-size guard (§10).
    let t = Instant::now();
    let (_, qry_records, _) = sequence::read_records(&args.query)?;
    phases.add("input.query", t.elapsed());
    let t = Instant::now();
    let (_, ref_records, _) = sequence::read_records(&args.reference)?;
    phases.add("input.reference", t.elapsed());

    let ref_meta = record_meta(&ref_records);
    let qry_meta = record_meta(&qry_records);

    // Manifest replay: every check that does not need the GPU — format,
    // software fingerprint (now including the executable hash), input
    // hashes, command-line parameters and record topology — runs before any
    // CUDA context, fit estimation or bin indexing, so a bad replay fails
    // before it costs a context or a device probe. The loaded manifest is
    // kept so a `--dump-manifest` on a replay re-emits it unchanged rather
    // than rebuilding it from this command line's (possibly different) B/Q
    // defaults; frozen plan membership, ordinals, targets and caps stay
    // authoritative and are never replanned.
    let loaded_manifest: Option<plan::PlanManifest> = args
        .from_manifest
        .as_ref()
        .map(|path| -> Fallible<plan::PlanManifest> {
            let text = std::fs::read_to_string(path)?;
            let m = plan::PlanManifest::read(&text)?;
            m.check_software()?;
            m.check_inputs(&ref_records, &qry_records)?;
            check_manifest_params(&m, args, &sub_mat)?;
            m.validate_records(&ref_meta, &qry_meta)?;
            Ok(m)
        })
        .transpose()?;

    let t = Instant::now();
    let ctx = CudaContext::new(0)?;
    phases.add("CUDA context init", t.elapsed());

    let t = Instant::now();
    let devices = crate::gpu::device_count().max(1);
    let probe = args.gpus.max(1).min(devices);
    let free = crate::gpu::min_free_bytes(probe)?;
    // ponytail: min(gpus, n_records) overcharges when bins < records; upgrade on measured false rejections
    let workers_upper = args.gpus.max(1).min(ref_meta.len().max(1));
    let budget = plan::worker_device_budget(free, workers_upper, devices);
    let q_target = args.query_block_size.unwrap_or(args.seq_block_size) as u64;
    let (plan, _worst, mut contract) = if let Some(m) = &loaded_manifest {
        let worst = m.check_fit(budget, shape.kmer_size)?;
        let contract = crate::gpu::ExecutionContract::from_resolved(m.max_hits, m.hsp_blocks);
        (m.plan.clone(), worst, contract)
    } else {
        let contract = crate::gpu::ExecutionContract::resolve(&ctx, args.max_hits, args.hsp_blocks);
        let (plan, worst) = plan::plan_within_budget(
            &ref_meta,
            &qry_meta,
            args.seq_block_size as u64,
            q_target,
            budget,
            shape.kmer_size,
            args.step,
            contract.max_hits,
            args.kegalign_bins,
            args.wga_chunk_size,
            transitions,
        )?;
        (plan, worst, contract)
    };
    // Physical capacity from the frozen/chosen plan at H under the same
    // per-worker budget; never replans. Clamped to the kernel-safe ceiling.
    {
        let candidate = plan::max_hit_capacity(
            &plan,
            budget,
            shape.kmer_size,
            args.step,
            contract.max_hits,
            args.wga_chunk_size,
            transitions,
        )?;
        contract.hit_capacity =
            crate::gpu::clamp_hit_capacity(contract.max_hits, candidate, contract.hsp_blocks)?;
    }
    phases.add("plan", t.elapsed());

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
    if let Some(path) = &args.dump_manifest {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        match &loaded_manifest {
            // Replay + dump: re-emit exactly what was loaded and validated,
            // never a reconstruction from this run's B/Q CLI defaults.
            Some(m) => m.write(&mut f)?,
            None => {
                // Fresh dump only: this is the one place `plan::executable_hash`
                // runs, so an ordinary run pays no hashing overhead.
                let m = plan::PlanManifest {
                    version: plan::PlanManifest::FORMAT,
                    hspz_version: env!("CARGO_PKG_VERSION").into(),
                    features: plan::compiled_features(),
                    max_hits: contract.max_hits,
                    hsp_blocks: contract.hsp_blocks,
                    seed: args.seed.clone(),
                    step: args.step,
                    transitions: !args.notransition,
                    xdrop: args.xdrop,
                    hspthresh: args.hspthresh,
                    noentropy: args.noentropy,
                    wga_chunk_size: args.wga_chunk_size,
                    lastz_interval_size: args.lastz_interval_size,
                    kegalign_bins: args.kegalign_bins,
                    seq_block_size: args.seq_block_size as u64,
                    query_block_size: q_target,
                    ref_hash: plan::records_hash(&ref_records),
                    qry_hash: plan::records_hash(&qry_records),
                    executable_hash: plan::executable_hash()?,
                    sub_mat: sub_mat.clone(),
                    strand: args.strand.clone(),
                    target_prefix: args.target_prefix.clone(),
                    query_prefix: args.query_prefix.clone(),
                    plan: plan.clone(),
                };
                m.write(&mut f)?;
            }
        }
        // `BufWriter::drop` discards a failed final flush, which would turn a
        // truncated manifest into a silently "successful" dump.
        f.flush()?;
    }

    // §18: reference bins to workers, deterministic LPT. Every query bin runs
    // against every reference bin, so `cost(R) = reference_bp x total_query_bp` is
    // monotone in the bin's own bp — LPT on `total_bp` is the same schedule with
    // less arithmetic (plan::assign_bins).
    let devices = crate::gpu::device_count().max(1);
    let workers = args.gpus.max(1).min(plan.reference_bins.len().max(1));
    let assignment = plan::assign_bins(&plan.reference_bins, workers);
    if workers > devices {
        eprintln!(
            "note: {workers} workers over {devices} device(s) — they time-slice one GPU. \
             That is a correctness configuration (§20), not a performance one."
        );
    }

    // Phase 1: host-memory preflight. The GPU preflight only sized the device
    // side; here we size host RAM for `workers` each building their own
    // (prefetched) reference state, falling back to no-prefetch before failing.
    // Runs after plan_within_budget so it sees the accepted (possibly shrunk)
    // bin set (AM-B of the review).
    let ref_bp_total: u64 = ref_meta.iter().map(|r| r.len).sum();
    let qry_bp_total: u64 = qry_meta.iter().map(|r| r.len).sum();
    let est = plan::host_estimate(
        &plan,
        ref_bp_total,
        qry_bp_total,
        shape.kmer_size,
        args.step,
        resolve_threads(args.threads),
        seed::max_seeds(args.wga_chunk_size, &shape, transitions),
    );
    // Phase 1 §6: never budget to 100% — reserve 10% for runtime/allocator/output
    // overhead the model does not see.
    let prefetch_requested = !args.no_ref_prefetch;
    let mut prefetch = prefetch_requested;
    let mut host_budget = None;
    let mut host_status = "unknown (no cgroup/meminfo reading)";
    if let Some(available) = timing::available_host_bytes().map(|b| b * 9 / 10) {
        host_budget = Some(available);
        let fits = plan::host_preflight(&est, &assignment, available)?;
        host_status = if fits {
            "fits with prefetch"
        } else {
            "fits without prefetch"
        };
        if prefetch && !fits {
            eprintln!(
                "note: host preflight disabled reference prefetch for {workers} worker(s) \
                 (Phase 1 §7)"
            );
        }
        prefetch &= fits;
    }
    // §9: the estimate that the decision was made on, so a run can be checked
    // against its own measured RSS (§11) without re-deriving the model.
    let host_peak_est = plan::host_peak(&est, &assignment, prefetch);

    let mut emitter = Emitter::new(args)?;
    let mut raw_all: Vec<(char, Vec<SegmentPair>)> = Vec::new();
    // §19: the emitter consumes units strictly in `WorkUnit.ordinal` order, so
    // `-D` history, file names and tar entry order never depend on which GPU
    // finished first. Workers push completed units into a bounded channel; the
    // main thread replays them in order, buffering whatever arrives early.
    // The thread budget is the machine's, not each worker's: `--threads` (or the
    // available parallelism) is divided once here. Resolving it inside every worker
    // gave a 2-GPU run 2x the host threads it asked for, which is exactly the
    // CPU starvation a small box would then be blamed for (Kaggle plan, amendment A).
    let per_worker = (resolve_threads(args.threads) / workers).max(1);
    // Round 68 measured the device seeder at -20.3% of 2-GPU wall and round 51 measured
    // the 1-GPU case as already hidden, so the worker count is the whole decision.
    // `HSPZ_DEVICE_SEEDS=0|1` overrides it, which is how the 1-GPU path gets tested.
    let device_seeds = device_seeds_for(workers);
    if device_seeds {
        eprintln!("note: generating query seeds on the device ({workers} workers)");
    }
    if workers > 1 {
        eprintln!(
            "note: {} host thread(s) per worker ({} workers)",
            per_worker, workers
        );
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<UnitOutput>(2 * workers);
    let reports = std::thread::scope(|scope| -> Fallible<Vec<WorkerReport>> {
        let mut handles = Vec::new();
        for (w, bins) in assignment.iter().enumerate() {
            let tx = tx.clone();
            let (plan, ref_records, qry_records, shape, sub_mat) =
                (&plan, &ref_records, &qry_records, &shape, &sub_mat);
            // `Box<dyn Error>` is not `Send`, so a worker reports failure as a
            // string and the caller turns it back into an error.
            handles.push(scope.spawn(move || {
                run_bins(
                    w % devices,
                    bins,
                    plan,
                    ref_records,
                    qry_records,
                    shape,
                    sub_mat,
                    transitions,
                    args,
                    prefetch,
                    per_worker,
                    device_seeds,
                    contract,
                    &tx,
                )
                .map_err(|e| e.to_string())
            }));
        }
        drop(tx);

        let mut buffered: std::collections::BTreeMap<u32, UnitOutput> =
            std::collections::BTreeMap::new();
        let mut next = 0u32;
        for unit in rx {
            buffered.insert(unit.ordinal, unit);
            while let Some(u) = buffered.remove(&next) {
                emitter.emit_unit(
                    u.reference_bin,
                    u.query_bin,
                    &u.ref_chrs,
                    &u.query_chrs,
                    &u.rc_chrs,
                    &u.pass,
                )?;
                if args.dump_raw.is_some() {
                    raw_all.extend_from_slice(&u.pass.raw);
                }
                next += 1;
            }
        }
        if !buffered.is_empty() {
            return Err(format!(
                "emitter has {} unit(s) it can never reach: expected ordinal {next}, \
                 hold {:?} — a worker died without sending (§19)",
                buffered.len(),
                buffered.keys().collect::<Vec<_>>()
            )
            .into());
        }
        let mut out = Vec::new();
        for h in handles {
            out.push(h.join().expect("gpu worker panicked")?);
        }
        Ok(out)
    })?;

    // Round 70: per-worker load, because the aggregate cannot show imbalance. Units are
    // assigned by reference bin, and a bin count that does not divide the worker count
    // leaves one worker holding the tail while the others idle — 7 bins over 2 workers is
    // 4/3, over 4 workers 2/2/2/1. The wall is set by the slowest worker, so the spread
    // here is the ceiling on any further multi-GPU scaling.
    if workers > 1 {
        let busy: Vec<f64> = reports.iter().map(|r| r.phases.gpu_ms()).collect();
        let units: Vec<u32> = reports
            .iter()
            .map(|r| r.lifecycle.work_units_executed)
            .collect();
        let (lo, hi) = (
            busy.iter().cloned().fold(f64::MAX, f64::min),
            busy.iter().cloned().fold(0.0, f64::max),
        );
        let gaps: Vec<f32> = reports.iter().map(|r| r.gap_ms).collect();
        let gapn: Vec<u64> = reports.iter().map(|r| r.gap_n).collect();
        eprintln!("note: per-worker stage-gap ms {gaps:?} over {gapn:?} pairs");
        eprintln!(
            "note: per-worker gpu-busy ms {busy:?} over units {units:?}; \
             spread {:.1}% of the busiest worker",
            if hi > 0.0 {
                100.0 * (hi - lo) / hi
            } else {
                0.0
            }
        );
    }

    let mut stats = Stats::default();
    let mut lifecycle = Lifecycle::default();
    let mut hit_stats = HitStats::default();
    let mut audit: Option<crate::census::SurvivorAudit> = None;
    let mut launches = 0u64;
    let (mut stage_syncs, mut pipeline_syncs) = (0u64, 0u64);
    let (mut uploads, mut copy_stalls) = (0u64, 0u64);
    let mut seed_table_ms = Duration::ZERO;
    let mut prefetched_ms = Duration::ZERO;
    let mut peak_used = 0usize;
    for r in &reports {
        stats.seeds += r.stats.seeds;
        stats.seed_hits += r.stats.seed_hits;
        stats.raw_hsps += r.stats.raw_hsps;
        stats.hsps += r.stats.hsps;
        lifecycle.seed_table_builds += r.lifecycle.seed_table_builds;
        lifecycle.engine_creations += r.lifecycle.engine_creations;
        lifecycle.reference_uploads += r.lifecycle.reference_uploads;
        lifecycle.query_swaps += r.lifecycle.query_swaps;
        lifecycle.work_units_executed += r.lifecycle.work_units_executed;
        launches += r.launches;
        stage_syncs += r.stage_syncs;
        pipeline_syncs += r.pipeline_syncs;
        uploads += r.uploads;
        copy_stalls += r.copy_stalls;
        seed_table_ms += r.seed_table_ms;
        prefetched_ms += r.prefetched_ms;
        peak_used = peak_used.max(r.peak_used);
        hit_stats.merge(&r.hit_stats);
        if let Some(a) = r.audit.as_ref() {
            audit
                .get_or_insert_with(crate::census::SurvivorAudit::default)
                .merge(a);
        }
        // With more than one worker these phase sums overlap in wall time: the
        // table is then "summed across workers", not a timeline.
        phases.merge(&r.phases);
    }
    let _ = peak_used;

    phases.add("seed table build", seed_table_ms);
    if prefetched_ms > Duration::ZERO {
        phases.add_overlapped("reference bin prep (standalone)", prefetched_ms);
    }
    lifecycle.check(plan.reference_bins.len() as u32, plan.units.len() as u32)?;

    if let Some(path) = &args.dump_raw {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for (strand, raw) in &raw_all {
            for h in raw {
                writeln!(
                    f,
                    "{strand}\t{}\t{}\t{}\t{}",
                    h.ref_start, h.query_start, h.len, h.score
                )?;
            }
        }
    }

    let out = emitter.finish(&mut phases)?;
    report_counts(&stats);
    if args.time {
        let wall = pre_main_ms + started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("\nWALL-TIME ACCOUNTING\n{}", phases.report(wall));
        eprintln!("  kernel launches: {}", launches);
        // Phase 1 §12: the mechanism gate. `stage` waits are the ones stream
        // ordering makes unnecessary and are 0 with --async-stages.
        eprintln!(
            "  host syncs: {} stage, {} pipeline ({} per launch)",
            stage_syncs,
            pipeline_syncs,
            if launches > 0 {
                format!(
                    "{:.3}",
                    (stage_syncs + pipeline_syncs) as f64 / launches as f64
                )
            } else {
                "-".into()
            }
        );
        eprintln!(
            "  max_hits: {} (target; resolved once; pinned unless --max-hits 0)\n  hit_capacity: {} (physical; success/failure only, never output bytes)\n  lifecycle: {} ref bins, \
{} work units, {} builds, {} engines, {} ref uploads, {} query swaps",
            contract.max_hits,
            contract.hit_capacity,
            plan.reference_bins.len(),
            plan.units.len(),
            lifecycle.seed_table_builds,
            lifecycle.engine_creations,
            lifecycle.reference_uploads,
            lifecycle.query_swaps,
        );
        // Phase 3 mechanism: a stalled upload is one that had not finished when
        // its compute needed it, i.e. overlap that did not happen.
        eprintln!(
            "  seed uploads: {} ({} stalled{})",
            uploads,
            copy_stalls,
            if uploads > 0 {
                format!(", {:.2}%", copy_stalls as f64 / uploads as f64 * 100.0)
            } else {
                String::new()
            }
        );
        eprintln!(
            "  output: {} files, {} bytes formatted, {} bytes written{}",
            out.files,
            out.bytes_in,
            if out.bytes_out > 0 {
                out.bytes_out
            } else {
                out.bytes_in
            },
            if args.diagonal_partition { " (-D)" } else { "" }
        );
        eprintln!("  peak RSS: {:>10} KiB", timing::peak_rss_kib());
        // Phase 1 §9: the host-budget decision, in the same units as the line
        // above so §11's validation is a subtraction.
        let mib = |b: u64| b as f64 / 1048576.0;
        eprintln!(
            "  host budget: estimated peak {:.0} MiB (shared {:.0} + {} worker(s), \
             {:.0} prefetching / {:.0} not), budget {}, status {}, \
             prefetch requested {} effective {}",
            mib(host_peak_est),
            mib(est.shared),
            workers,
            mib(est.per_worker_prefetch),
            mib(est.per_worker_no_prefetch),
            host_budget.map_or("unknown".to_string(), |b| format!("{:.0} MiB", mib(b))),
            host_status,
            prefetch_requested,
            prefetch,
        );
    }
    if let Some(a) = audit.as_ref() {
        eprintln!("\n(ALL REFERENCE BINS) {}", a.report());
    }
    if args.hit_stats {
        eprintln!("\nHITS PER SEED\n{}", hit_stats.report());
    }
    Ok(stats)
}

/// The `--cpu-only` path: reader-attributable digests and seed/hit counts, no
/// GPU. Kept on `prepare`, so it still rejects oversized single-block input —
/// the whole-genome reader check is a separate concern from the executor.
fn run_cpu_only(
    args: &RunArgs,
    phases: &mut Phases,
    pre_main_ms: f64,
    started: Instant,
) -> Fallible<Stats> {
    let p = prepare(args, phases)?;
    eprintln!(
        "reference: {:16x}  {} ({} bytes, {} records)",
        p.reference.digest(),
        p.reference.format.label(),
        p.reference.bytes_read,
        p.reference.chrs.len()
    );
    eprintln!(
        "query:     {:16x}  {} ({} bytes, {} records)",
        p.query.digest(),
        p.query.format.label(),
        p.query.bytes_read,
        p.query.chrs.len()
    );
    let stats = cpu_stats(&p, args);
    report_counts(&stats);
    if args.time {
        let wall = pre_main_ms + started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("\nWALL-TIME ACCOUNTING\n{}", phases.report(wall));
        eprintln!("  peak RSS: {:>10} KiB", timing::peak_rss_kib());
    }
    Ok(stats)
}

/// `--threads 0` means "as many as the machine will usefully give us".
pub(crate) fn resolve_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn report_counts(stats: &Stats) {
    eprintln!("#seeds: {}", stats.seeds);
    eprintln!("#seed hits: {}", stats.seed_hits);
    eprintln!("#raw HSPs: {}", stats.raw_hsps);
    eprintln!("#HSPs: {}", stats.hsps);
}

/// Seed and hit counts without a GPU: `find_num_hits` is just a lookup into the
/// reference index table.
fn cpu_stats(p: &Prepared, args: &RunArgs) -> Stats {
    let mut stats = Stats::default();
    let mut tally = |seq: &[u8], range: (u32, u32)| {
        for chunk in seed::chunks(range.0, range.1, args.wga_chunk_size) {
            let seeds = seed::chunk_seeds(seq, &p.shape, p.transitions, chunk);
            stats.seeds += seeds.len() as u64;
            stats.seed_hits += seeds
                .iter()
                .map(|s| p.table.hit_count((s >> 32) as u32) as u64)
                .sum::<u64>();
        }
    };
    for &(start, end) in &p.intervals {
        if p.plus {
            tally(p.enc_query_source(), (start, end));
        }
        if p.minus {
            tally(&p.query_rc, (p.q_block_len - end, p.q_block_len - start));
        }
    }
    // Diagnostic only: exact production counts above are untouched. When
    // HSPZ_N1_CENSUS is set, a second CPU-only pass reports the hit-weighted
    // fraction of seeds whose query END is N1-certified.
    if std::env::var("HSPZ_N1_CENSUS").is_ok() {
        let c = n1_census(p, args);
        eprintln!("{}", n1_report_line(&c));
    }
    stats
}

/// Per-column maximum of the resolved substitution matrix over all rows.
///
/// `sub_mat` is the `NUC x NUC` matrix from `scoring::build_sub_mat` on this
/// command line; `colmax[q] = max_r SUB[r * NUC + q]`.
pub(crate) fn n1_colmax(sub_mat: &[i32]) -> [i32; crate::sequence::NUC] {
    let mut colmax = [i32::MIN; crate::sequence::NUC];
    for r in 0..crate::sequence::NUC {
        for q in 0..crate::sequence::NUC {
            let v = sub_mat[r * crate::sequence::NUC + q];
            if v > colmax[q] {
                colmax[q] = v;
            }
        }
    }
    colmax
}

/// N1 eligibility bitset over query ENDs for one orientation.
///
/// `enc` is the encoded whole query block (`E_NT` separators included) in the
/// same coordinate frame the seeds use. STOP iff `colmax < -xdrop` over the
/// actual matrix; positive mass per position is `max(0, colmax)`. A legal END
/// `e` belongs to the maximal stop-free gap containing `e - 1`; it is
/// certified iff that gap's mass is strictly below `hspthresh`. Returned
/// `eligible[e]` is indexed by END (`0..=n`); index 0 is always false.
pub(crate) fn n1_eligibility(enc: &[u8], sub_mat: &[i32], xdrop: i32, hspthresh: i32) -> Vec<bool> {
    let colmax = n1_colmax(sub_mat);
    let neg_xdrop = xdrop.checked_neg().unwrap_or(i32::MIN);
    let mut is_stop = [false; crate::sequence::NUC];
    let mut posmass = [0i64; crate::sequence::NUC];
    for q in 0..crate::sequence::NUC {
        is_stop[q] = colmax[q] < neg_xdrop;
        posmass[q] = colmax[q].max(0) as i64;
    }
    let n = enc.len();
    let mut eligible = vec![false; n.checked_add(1).expect("query block too large")];
    let mut lo: usize = 0;
    let mut mass: i64 = 0;
    for i in 0..n {
        let q = enc[i] as usize;
        let (stop, pm) = if q < crate::sequence::NUC {
            (is_stop[q], posmass[q])
        } else {
            (true, 0)
        };
        if stop {
            if mass < hspthresh as i64 && lo.checked_add(1).unwrap_or(usize::MAX) <= i {
                eligible[lo + 1..=i].fill(true);
            }
            lo = i.checked_add(1).expect("query block too large");
            mass = 0;
        } else {
            mass = mass.checked_add(pm).expect("N1 gap mass overflow");
        }
    }
    if mass < hspthresh as i64 && lo.checked_add(1).unwrap_or(usize::MAX) <= n {
        eligible[lo + 1..=n].fill(true);
    }
    eligible
}

/// Hit-weighted N1 census counts, per orientation and total.
#[derive(Default, Debug)]
pub(crate) struct N1Census {
    pub seeds_plus: u64,
    pub elig_plus: u64,
    pub hits_plus: u64,
    pub hits_elig_plus: u64,
    pub seeds_minus: u64,
    pub elig_minus: u64,
    pub hits_minus: u64,
    pub hits_elig_minus: u64,
}

/// Second CPU-only pass over the same seeds `cpu_stats` counts. Production
/// totals are recomputed identically here only for the report denominators;
/// `cpu_stats` itself is untouched.
pub(crate) fn n1_census(p: &Prepared, args: &RunArgs) -> N1Census {
    let mut c = N1Census::default();
    let plus_elig = if p.plus {
        Some(n1_eligibility(
            &p.enc_query,
            &p.sub_mat,
            args.xdrop,
            args.hspthresh,
        ))
    } else {
        None
    };
    let minus_elig = if p.minus {
        Some(n1_eligibility(
            &p.enc_query_rc,
            &p.sub_mat,
            args.xdrop,
            args.hspthresh,
        ))
    } else {
        None
    };
    for &(start, end) in &p.intervals {
        if p.plus {
            let elig = plus_elig.as_ref().expect("plus eligibility built");
            for chunk in seed::chunks(start, end, args.wga_chunk_size) {
                let seeds = seed::chunk_seeds(p.enc_query_source(), &p.shape, p.transitions, chunk);
                for s in &seeds {
                    let hc = p.table.hit_count((s >> 32) as u32) as u64;
                    c.seeds_plus = c.seeds_plus.checked_add(1).expect("seed count overflow");
                    c.hits_plus = c.hits_plus.checked_add(hc).expect("hit count overflow");
                    let pos = (*s & 0xffff_ffff) as usize;
                    let e = pos.checked_add(p.shape.size).expect("END overflow");
                    if *elig.get(e).unwrap_or(&false) {
                        c.elig_plus = c.elig_plus.checked_add(1).expect("seed count overflow");
                        c.hits_elig_plus = c
                            .hits_elig_plus
                            .checked_add(hc)
                            .expect("hit count overflow");
                    }
                }
            }
        }
        if p.minus {
            let elig = minus_elig.as_ref().expect("minus eligibility built");
            for chunk in seed::chunks(
                p.q_block_len - end,
                p.q_block_len - start,
                args.wga_chunk_size,
            ) {
                let seeds = seed::chunk_seeds(&p.query_rc, &p.shape, p.transitions, chunk);
                for s in &seeds {
                    let hc = p.table.hit_count((s >> 32) as u32) as u64;
                    c.seeds_minus = c.seeds_minus.checked_add(1).expect("seed count overflow");
                    c.hits_minus = c.hits_minus.checked_add(hc).expect("hit count overflow");
                    let pos = (*s & 0xffff_ffff) as usize;
                    let e = pos.checked_add(p.shape.size).expect("END overflow");
                    if *elig.get(e).unwrap_or(&false) {
                        c.elig_minus = c.elig_minus.checked_add(1).expect("seed count overflow");
                        c.hits_elig_minus = c
                            .hits_elig_minus
                            .checked_add(hc)
                            .expect("hit count overflow");
                    }
                }
            }
        }
    }
    c
}

/// One stderr line for the env-gated census. Percentages are eligible/total.
pub(crate) fn n1_report_line(c: &N1Census) -> String {
    let t_seeds = c
        .seeds_plus
        .checked_add(c.seeds_minus)
        .expect("seed count overflow");
    let t_elig = c
        .elig_plus
        .checked_add(c.elig_minus)
        .expect("seed count overflow");
    let t_hits = c
        .hits_plus
        .checked_add(c.hits_minus)
        .expect("hit count overflow");
    let t_ehits = c
        .hits_elig_plus
        .checked_add(c.hits_elig_minus)
        .expect("hit count overflow");
    let pct = |a: u64, b: u64| {
        if b > 0 {
            a as f64 / b as f64 * 100.0
        } else {
            0.0
        }
    };
    format!(
        "#n1 census: eligible seeds {} of {} ({:.4}%), eligible hits {} of {} ({:.4}%) \
[plus: {} of {} ({:.4}%), {} of {} ({:.4}%); minus: {} of {} ({:.4}%), {} of {} ({:.4}%)]",
        t_elig,
        t_seeds,
        pct(t_elig, t_seeds),
        t_ehits,
        t_hits,
        pct(t_ehits, t_hits),
        c.elig_plus,
        c.seeds_plus,
        pct(c.elig_plus, c.seeds_plus),
        c.hits_elig_plus,
        c.hits_plus,
        pct(c.hits_elig_plus, c.hits_plus),
        c.elig_minus,
        c.seeds_minus,
        pct(c.elig_minus, c.seeds_minus),
        c.hits_elig_minus,
        c.hits_minus,
        pct(c.hits_elig_minus, c.hits_minus),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use crate::plan::{Bin, PlanManifest, WorkUnit};
    use clap::Parser;

    /// Real CLI defaults, not a hand-maintained struct literal: the parser is
    /// the actual entry point every flag test below claims to exercise, so a
    /// new field gets its real default here instead of a second, driftable copy.
    fn base_args() -> RunArgs {
        match Cli::try_parse_from(["hspz", "run", "-r", "ref.fa", "-q", "qry.fa"])
            .unwrap()
            .command
        {
            Command::Run(args) => args,
            _ => unreachable!("parsed `run` subcommand"),
        }
    }

    /// A minimal, otherwise-valid format2 manifest fixture — one reference bin,
    /// one query bin, one work unit — just enough for `check_manifest_params`,
    /// which never looks past the scalar fields it compares.
    fn base_manifest(sub_mat: Vec<i32>) -> PlanManifest {
        PlanManifest {
            version: PlanManifest::FORMAT,
            hspz_version: env!("CARGO_PKG_VERSION").into(),
            features: plan::compiled_features(),
            max_hits: 16_711_680,
            hsp_blocks: 16384,
            seed: "12of19".into(),
            step: 1,
            transitions: true,
            xdrop: 910,
            hspthresh: 3000,
            noentropy: false,
            wga_chunk_size: 250_000,
            lastz_interval_size: 10_000_000,
            kegalign_bins: false,
            seq_block_size: 500_000_000,
            query_block_size: 500_000_000,
            ref_hash: 1,
            qry_hash: 2,
            executable_hash: 0xdead_beef_cafe_1234,
            sub_mat,
            strand: "both".into(),
            target_prefix: String::new(),
            query_prefix: String::new(),
            plan: plan::Plan {
                reference_bins: vec![Bin {
                    id: 0,
                    record_ids: vec![0],
                    total_bp: 100,
                }],
                query_bins: vec![Bin {
                    id: 0,
                    record_ids: vec![0],
                    total_bp: 50,
                }],
                units: vec![WorkUnit {
                    ordinal: 0,
                    reference_bin: 0,
                    query_bin: 0,
                }],
            },
        }
    }

    /// The frozen identity is the actual 64-cell matrix, not the `--ambiguous`/
    /// `--scoring` text that produced it: an equal matrix must replay even from
    /// different flags, and a matrix that differs only because of ambiguous
    /// handling must still be rejected.
    #[test]
    fn matrix_identity_not_scoring_text_decides_replay() {
        let args = base_args();
        let plus = scoring::build_sub_mat(&args.ambiguous, args.xdrop, None).unwrap();
        let mut other = args.clone();
        other.ambiguous = "iupac".into();
        let iupac = scoring::build_sub_mat(&other.ambiguous, other.xdrop, None).unwrap();
        assert_ne!(plus, iupac, "fixture must actually differ");

        let m = base_manifest(plus.clone());
        assert!(check_manifest_params(&m, &args, &plus).is_ok());

        let err = check_manifest_params(&m, &args, &iupac).unwrap_err();
        assert!(err.contains("substitution matrix"), "{err}");
    }

    #[test]
    fn strand_and_prefix_mismatches_are_rejected() {
        let args = base_args();
        let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, None).unwrap();
        let m = base_manifest(sub_mat.clone());
        assert!(check_manifest_params(&m, &args, &sub_mat).is_ok());

        let mut strand = args.clone();
        strand.strand = "plus".into();
        let err = check_manifest_params(&m, &strand, &sub_mat).unwrap_err();
        assert!(err.contains("strand"), "{err}");

        let mut target = args.clone();
        target.target_prefix = "chrT_".into();
        let err = check_manifest_params(&m, &target, &sub_mat).unwrap_err();
        assert!(err.contains("target_prefix"), "{err}");

        let mut query = args.clone();
        query.query_prefix = "chrQ_".into();
        let err = check_manifest_params(&m, &query, &sub_mat).unwrap_err();
        assert!(err.contains("query_prefix"), "{err}");
    }

    /// A nonzero CLI cap is a pin and must agree with the frozen plan; `0`
    /// (unset) silently adopts the manifest's resolved cap instead.
    #[test]
    fn nonzero_caps_must_match_zero_adopts_the_manifest() {
        let args = base_args();
        let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, None).unwrap();
        let m = base_manifest(sub_mat.clone());
        assert!(check_manifest_params(&m, &args, &sub_mat).is_ok());

        let mut hits_conflict = args.clone();
        hits_conflict.max_hits = 1;
        let err = check_manifest_params(&m, &hits_conflict, &sub_mat).unwrap_err();
        assert!(err.contains("max_hits"), "{err}");

        let mut hits_agree = args.clone();
        hits_agree.max_hits = m.max_hits;
        assert!(check_manifest_params(&m, &hits_agree, &sub_mat).is_ok());

        let mut blocks_conflict = args.clone();
        blocks_conflict.hsp_blocks = 1;
        let err = check_manifest_params(&m, &blocks_conflict, &sub_mat).unwrap_err();
        assert!(err.contains("hsp_blocks"), "{err}");
    }

    /// `iupac` and the equivalent `<field>,<reward>,<penalty>` spelling resolve
    /// to the same matrix, which is the property `matrix_identity_not_scoring_
    /// text_decides_replay` above relies on to allow replay across flag spellings.
    #[test]
    fn equal_matrix_from_different_ambiguity_spellings() {
        let bare = scoring::build_sub_mat("iupac", 910, None).unwrap();
        let spelled_out = scoring::build_sub_mat("iupac,0,0", 910, None).unwrap();
        assert_eq!(bare, spelled_out);
    }

    /// `prepare` is the GPU-free path `benchmark` and `--cpu-only` share; neither
    /// implements frozen-plan replay or manifest dumps, so both flags must fail
    /// fast here rather than being silently dropped. Both paths point at inputs
    /// that do not exist, so a pass would only be possible by never reading them.
    #[test]
    fn prepare_rejects_manifest_flags_before_input_reads() {
        let mut phases = Phases::new();

        let mut from_manifest = base_args();
        from_manifest.reference = PathBuf::from("/nonexistent/ref.fa");
        from_manifest.query = PathBuf::from("/nonexistent/qry.fa");
        from_manifest.from_manifest = Some(PathBuf::from("/nonexistent/plan.manifest"));
        let err = prepare(&from_manifest, &mut phases).err().unwrap();
        assert!(err.to_string().contains("--from-manifest"), "{err}");

        let mut dump_manifest = base_args();
        dump_manifest.reference = PathBuf::from("/nonexistent/ref.fa");
        dump_manifest.query = PathBuf::from("/nonexistent/qry.fa");
        dump_manifest.dump_manifest = Some(PathBuf::from("/nonexistent/out.manifest"));
        let err = prepare(&dump_manifest, &mut phases).err().unwrap();
        assert!(err.to_string().contains("--dump-manifest"), "{err}");
    }

    /// §19's replay buffers units by ordinal so a worker's completion order
    /// never changes what lands on disk: drives a hand-built `Emitter` (not
    /// `Emitter::new`, which would resolve `HSPZ_ANCHOR_CENSUS` from the
    /// inherited environment) through all `3! = 6` orderings of three
    /// `UnitOutput`s and diffs each permutation's filename -> bytes map
    /// against the first one. `is_identity_permutation`'s four negative cases
    /// prove a count-only check (rejected by a prior review) would miss a
    /// same-count duplicate/miss, a dropped empty-output unit, a swapped bin,
    /// or an unknown ordinal.
    #[test]
    fn emitter_output_is_independent_of_unit_emission_order() {
        fn seg(ref_start: u32, query_start: u32, len: u32, score: i32) -> SegmentPair {
            SegmentPair {
                ref_start,
                query_start,
                len,
                score,
            }
        }

        // Two bin-local chromosomes per table, boundary at offset 100, so
        // fixtures below can freely land HSPs in either half.
        fn chrs(names: &[&str]) -> Vec<Chr> {
            names
                .iter()
                .enumerate()
                .map(|(i, &name)| Chr {
                    name: name.into(),
                    start: i * 100,
                    len: 100,
                })
                .collect()
        }

        let units = vec![
            // Empty unit: a real work unit with zero HSPs on both strands, the
            // completeness gap a count-only check would miss.
            UnitOutput {
                ordinal: 7,
                reference_bin: 3,
                query_bin: 8,
                ref_chrs: Vec::new(),
                query_chrs: Vec::new(),
                rc_chrs: Vec::new(),
                pass: Pass {
                    intervals: Vec::new(),
                    ..Pass::default()
                },
            },
            UnitOutput {
                ordinal: 2,
                reference_bin: 11,
                query_bin: 4,
                ref_chrs: chrs(&["b_r0", "b_r1"]),
                query_chrs: chrs(&["b_q0", "b_q1"]),
                rc_chrs: chrs(&["b_rc0", "b_rc1"]),
                pass: Pass {
                    intervals: vec![
                        (vec![seg(5, 5, 10, 100)], vec![seg(150, 150, 5, 50)]),
                        (vec![seg(120, 20, 8, 77)], vec![seg(30, 130, 12, 33)]),
                    ],
                    ..Pass::default()
                },
            },
            UnitOutput {
                ordinal: 5,
                reference_bin: 6,
                query_bin: 13,
                ref_chrs: chrs(&["c_r0", "c_r1"]),
                query_chrs: chrs(&["c_q0", "c_q1"]),
                rc_chrs: chrs(&["c_rc0", "c_rc1"]),
                pass: Pass {
                    intervals: vec![
                        (vec![seg(2, 60, 6, 11)], vec![seg(70, 3, 4, 22)]),
                        (vec![seg(65, 65, 9, 44)], vec![seg(5, 175, 3, 66)]),
                    ],
                    ..Pass::default()
                },
            },
        ];
        let expected_identity: Vec<(u32, u32, u32)> = units
            .iter()
            .map(|u| (u.ordinal, u.reference_bin, u.query_bin))
            .collect();

        /// Test-only specification oracle, not a runtime guard: true only if
        /// `got` is an exact rearrangement of `expected` — same length,
        /// unique ordinals, and every (ordinal, reference_bin, query_bin)
        /// tuple actually belongs to `expected`. `expected` is itself
        /// ordinal-unique, so matching count plus that membership already
        /// forces full coverage; there is nothing left to check.
        fn is_identity_permutation(expected: &[(u32, u32, u32)], got: &[(u32, u32, u32)]) -> bool {
            if got.len() != expected.len() {
                return false;
            }
            let mut seen = std::collections::BTreeSet::new();
            for &(ordinal, reference_bin, query_bin) in got {
                if !seen.insert(ordinal) {
                    return false;
                }
                match expected.iter().find(|&&(o, ..)| o == ordinal) {
                    Some(&(_, rb, qb)) if (rb, qb) == (reference_bin, query_bin) => {}
                    _ => return false,
                }
            }
            true
        }

        // Negative cases the prior review flagged as unexecuted: each must be
        // rejected for a distinct reason.
        assert!(
            !is_identity_permutation(&expected_identity, &[(7, 3, 8), (7, 3, 8), (2, 11, 4)]),
            "duplicate ordinal at the expected count must still fail identity"
        );
        assert!(
            !is_identity_permutation(&expected_identity, &[(2, 11, 4), (5, 6, 13)]),
            "dropping the empty-output unit must fail identity"
        );
        assert!(
            !is_identity_permutation(&expected_identity, &[(7, 3, 8), (2, 999, 4), (5, 6, 13)]),
            "wrong bin tuple must fail identity"
        );
        assert!(
            !is_identity_permutation(&expected_identity, &[(7, 3, 8), (2, 11, 4), (99, 6, 13)]),
            "unknown ordinal must fail identity"
        );

        const PERMS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        const EXPECTED_FILES: [&str; 8] = [
            "tmp1.block4.r11.plus.segments",
            "tmp1.block4.r11.minus.segments",
            "tmp2.block4.r11.plus.segments",
            "tmp2.block4.r11.minus.segments",
            "tmp1.block13.r6.plus.segments",
            "tmp1.block13.r6.minus.segments",
            "tmp2.block13.r6.plus.segments",
            "tmp2.block13.r6.minus.segments",
        ];

        let mut baseline: Option<std::collections::BTreeMap<String, Vec<u8>>> = None;
        for (p, order) in PERMS.iter().enumerate() {
            let permuted_identity: Vec<(u32, u32, u32)> = order
                .iter()
                .map(|&i| (units[i].ordinal, units[i].reference_bin, units[i].query_bin))
                .collect();
            assert!(is_identity_permutation(
                &expected_identity,
                &permuted_identity
            ));

            // Never pre-deleted: a leftover dir from a crashed prior run must
            // fail loudly here, not get silently wiped.
            let dir =
                std::env::temp_dir().join(format!("hspz-emitter-order-{}-{p}", std::process::id()));
            std::fs::create_dir(&dir).unwrap();

            let mut emitter = Emitter {
                sink: Box::new(DirectorySink::new(&dir).unwrap()),
                part: Partitioner::default(),
                diagonal: false,
                partition_ms: 0.0,
                format_ms: 0.0,
                archive_ms: 0.0,
                files: 0,
                bytes_in: 0,
                audit: None,
            };

            for &i in order {
                let u = &units[i];
                emitter
                    .emit_unit(
                        u.reference_bin,
                        u.query_bin,
                        &u.ref_chrs,
                        &u.query_chrs,
                        &u.rc_chrs,
                        &u.pass,
                    )
                    .unwrap();
            }
            let mut phases = Phases::new();
            emitter.finish(&mut phases).unwrap();

            let mut got = std::collections::BTreeMap::new();
            for entry in std::fs::read_dir(&dir).unwrap() {
                let entry = entry.unwrap();
                got.insert(
                    entry.file_name().to_string_lossy().into_owned(),
                    std::fs::read(entry.path()).unwrap(),
                );
            }
            std::fs::remove_dir_all(&dir).unwrap();

            match &baseline {
                None => {
                    assert_eq!(
                        got.len(),
                        8,
                        "expected file count changed under the fixtures above"
                    );
                    for name in EXPECTED_FILES {
                        assert!(got.contains_key(name), "missing expected file {name}");
                    }
                    // Formatted strand/chromosome values actually present,
                    // not just an empty file of the right name.
                    let plus_b =
                        String::from_utf8(got["tmp1.block4.r11.plus.segments"].clone()).unwrap();
                    assert!(
                        plus_b.contains("b_r0")
                            && plus_b.contains("b_q0")
                            && plus_b.contains("\t+\t")
                    );
                    let minus_c =
                        String::from_utf8(got["tmp2.block13.r6.minus.segments"].clone()).unwrap();
                    assert!(minus_c.contains("c_rc1") && minus_c.contains("\t-\t"));
                    baseline = Some(got);
                }
                Some(base) => assert_eq!(
                    &got, base,
                    "permutation {order:?} produced a different filename->bytes map than permutation 0"
                ),
            }
        }
    }

    /// Env-gated N1 census: a 300 bp query with a 25-bp uppercase island
    /// (mass 25*91 = 2275 < 3000, eligible) bounded by lowercase stops inside
    /// two long uppercase runs (masses 9100 and 14105, not eligible).
    /// Reference `N + A*20 + N + T*20` (step 1, transitions off) indexes
    /// exactly two pure-A and two pure-T windows, so every plus (all-A) and
    /// minus (all-T) seed hits twice. Hand-derived, both orientations:
    /// 82 + 7 + 137 = 226 seeds, 7 eligible; 452 hits, 14 eligible hits.
    #[test]
    fn n1_census_island_eligible_long_runs_not() {
        use crate::seed::SeedTable;
        use crate::sequence::{self, Genome};

        let mut q = Vec::new();
        q.extend_from_slice(&vec![b'A'; 100]);
        q.extend_from_slice(&vec![b'a'; 10]);
        q.extend_from_slice(&vec![b'A'; 25]);
        q.extend_from_slice(&vec![b'a'; 10]);
        q.extend_from_slice(&vec![b'A'; 155]);
        assert_eq!(q.len(), 300);

        let r = [b"N".as_slice(), &vec![b'A'; 20], b"N", &vec![b'T'; 20]].concat();
        assert_eq!(r.len(), 42);

        let shape = Shape::parse("12of19").unwrap();
        assert_eq!(shape.size, 19);
        let sub_mat = scoring::build_sub_mat("", 910, None).unwrap();
        // Default colmax: A/T 91, C/G 100, L/N stop, X zero-mass non-stop.
        let cm = n1_colmax(&sub_mat);
        assert_eq!([cm[0], cm[1], cm[2], cm[3]], [91, 100, 100, 91]);
        assert!(cm[4] < -910 && cm[5] < -910 && cm[7] < -910);
        assert!(!(cm[6] < -910));

        let (qbuf, qchrs, qblock) = sequence::pack([("q", q.as_slice())], "");
        let (rbuf, rchrs, rblock) = sequence::pack([("r", r.as_slice())], "");
        let query = Genome {
            buf: qbuf,
            chrs: qchrs,
            block_len: qblock,
            format: sequence::Format::Fasta,
            bytes_read: 0,
        };
        let reference = Genome {
            buf: rbuf,
            chrs: rchrs,
            block_len: rblock,
            format: sequence::Format::Fasta,
            bytes_read: 0,
        };
        let (query_rc, rc_chrs) = query.reverse_complement();
        let enc_query = sequence::encode(&query.buf[..query.block_len]);
        let enc_query_rc = sequence::encode(&query_rc);
        let table = SeedTable::build(&reference.buf[..reference.block_len], &shape, 1);
        let intervals = sequence::intervals(query.block_len, shape.size, 10_000_000);
        assert_eq!(intervals, vec![(0, 281)]);
        let q_block_len = (query.block_len - shape.size) as u32;

        let mut args = base_args();
        args.xdrop = 910;
        args.hspthresh = 3000;
        args.wga_chunk_size = 250_000;

        let p = Prepared {
            shape,
            sub_mat,
            reference,
            query,
            rc_chrs,
            query_rc,
            table,
            intervals,
            q_block_len,
            transitions: false,
            plus: true,
            minus: true,
            enc_ref: Vec::new(),
            enc_query,
            enc_query_rc,
        };

        // Spot-check the bitsets: island ENDs eligible, long-run ENDs not.
        let pe = n1_eligibility(&p.enc_query, &p.sub_mat, args.xdrop, args.hspthresh);
        assert_eq!(pe.len(), 301);
        assert!(!pe[19], "pos 0 in the 100-bp run");
        assert!(!pe[100], "END on the run/stop boundary");
        assert!(pe[129], "island pos 110");
        assert!(
            pe[135],
            "island END on the stop boundary stays with its gap"
        );
        assert!(!pe[164], "tail pos 145");
        assert!(!pe[300], "tail END");
        let me = n1_eligibility(&p.enc_query_rc, &p.sub_mat, args.xdrop, args.hspthresh);
        assert_eq!(me.len(), 301);
        assert!(!me[19], "RC head run");
        assert!(me[184], "RC island");
        assert!(!me[300], "RC tail run");

        let c = n1_census(&p, &args);
        assert_eq!((c.seeds_plus, c.elig_plus), (226, 7), "plus seeds");
        assert_eq!((c.hits_plus, c.hits_elig_plus), (452, 14), "plus hits");
        assert_eq!((c.seeds_minus, c.elig_minus), (226, 7), "minus seeds");
        assert_eq!((c.hits_minus, c.hits_elig_minus), (452, 14), "minus hits");

        // Production counting agrees with the census denominators.
        let stats = cpu_stats(&p, &args);
        assert_eq!(stats.seeds, 452);
        assert_eq!(stats.seed_hits, 904);

        // Report line carries eligible/total with 4 decimals.
        let line = n1_report_line(&c);
        assert!(line.starts_with("#n1 census: eligible seeds 14 of 452 (3.0973%)"));
        assert!(line.contains("eligible hits 28 of 904 (3.0973%)"));
    }
}
