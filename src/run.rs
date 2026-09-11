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
use crate::gpu::{DeviceProfile, Engine, EngineConfig, HitStats, Lifecycle};
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
    if args.query_list.is_some() {
        return Err("--query-list is only supported by `run`, not benchmark or --cpu-only".into());
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
    // `-B 0` (automatic layout) is a planner decision this single-block path
    // never makes: resolve to the default target so the guard below keeps
    // today's behavior. `Genome::load` packs the whole input as one block and
    // uses the target only to reject multi-block input, so no per-bin target
    // applies here (the GPU `run` path below does not call `Genome::load`).
    if args.seq_block_size == 0 && args.kegalign_bins {
        return Err(plan::AUTO_KEGALIGN_ERROR.into());
    }
    let seq_block_size = if args.seq_block_size == 0 {
        plan::DEFAULT_BLOCK_TARGET as u32
    } else {
        args.seq_block_size
    };
    let query_path = args
        .query
        .as_ref()
        .ok_or("--query is required without --query-list")?;
    let query = Genome::load(query_path, &args.query_prefix, seq_block_size)?;
    phases.add("input.query", t.elapsed());
    let t = Instant::now();
    let reference = Genome::load(&args.reference, &args.target_prefix, seq_block_size)?;
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
        Self::new_at(
            &args.output,
            tarball_path(args).as_deref(),
            args.diagonal_partition,
            crate::census::SurvivorAudit::dump_path().as_deref(),
        )
    }

    /// One emitter rooted at `output`: a directory sink, or a tarball sink at
    /// `tarball` when set. `audit_path` is the `HSPZ_ANCHOR_CENSUS` dump file;
    /// batch jobs pass a per-job path so emitters never share one file.
    pub(crate) fn new_at(
        output: &std::path::Path,
        tarball: Option<&std::path::Path>,
        diagonal: bool,
        audit_path: Option<&std::path::Path>,
    ) -> Fallible<Self> {
        let sink: Box<dyn OutputSink> = match tarball {
            Some(path) => Box::new(TarGzSink::new(path)?),
            None => Box::new(DirectorySink::new(output)?),
        };
        let mut audit = audit_path
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
            diagonal,
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

/// Writes a `--dump-plan` membership table: one `side<TAB>bin<TAB>record<TAB>bp`
/// line per record, reference side first. Shared by the single-query path and
/// the batch executor, which calls it once per job at a per-job sibling path.
fn write_plan_dump(
    path: &std::path::Path,
    plan: &plan::Plan,
    ref_records: &[(String, Vec<u8>)],
    qry_records: &[(String, Vec<u8>)],
) -> Fallible<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (side, bins, recs) in [
        ("reference", &plan.reference_bins, ref_records),
        ("query", &plan.query_bins, qry_records),
    ] {
        for b in bins {
            for &id in &b.record_ids {
                let (name, seq) = &recs[id as usize];
                writeln!(f, "{side}\t{}\t{name}\t{}", b.id, seq.len())?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

/// Builds the manifest `--dump-manifest` writes for a fresh (non-replay) run.
/// `executable_hash` is passed in because the batch computes it once for every
/// job, while the single-query path calls [`plan::executable_hash`] only here
/// so an ordinary run pays no hashing overhead.
#[allow(clippy::too_many_arguments)]
fn plan_manifest(
    args: &RunArgs,
    contract: &crate::gpu::ExecutionContract,
    sub_mat: &[i32],
    res_seq: u64,
    res_qry: u64,
    ref_records: &[(String, Vec<u8>)],
    qry_records: &[(String, Vec<u8>)],
    plan: &plan::Plan,
    executable_hash: u64,
) -> plan::PlanManifest {
    plan::PlanManifest {
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
        seq_block_size: res_seq,
        query_block_size: res_qry,
        ref_hash: plan::records_hash(ref_records),
        qry_hash: plan::records_hash(qry_records),
        executable_hash,
        sub_mat: sub_mat.to_vec(),
        strand: args.strand.clone(),
        target_prefix: args.target_prefix.clone(),
        query_prefix: args.query_prefix.clone(),
        plan: plan.clone(),
    }
}

/// Writes one `--dump-manifest` file and flushes it: `BufWriter::drop`
/// discards a failed final flush, turning a truncated manifest into a
/// silently "successful" dump.
fn write_manifest_dump(path: &std::path::Path, m: &plan::PlanManifest) -> Fallible<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    m.write(&mut f)?;
    f.flush()?;
    Ok(())
}

/// Everything the `--time` footer reports after the executor returns. One
/// printer for the single-query and batch paths, so the two can never drift;
/// the callers pass the totals that apply to them.
struct TimeFooter<'a> {
    phases: &'a Phases,
    wall_ms: f64,
    launches: u64,
    stage_syncs: u64,
    pipeline_syncs: u64,
    contract: &'a crate::gpu::ExecutionContract,
    ref_bins: usize,
    units: usize,
    lifecycle: &'a Lifecycle,
    uploads: u64,
    copy_stalls: u64,
    files: usize,
    bytes_in: u64,
    bytes_out: u64,
    diagonal: bool,
    host_peak_est: u64,
    est_shared: u64,
    workers: usize,
    est_prefetch: u64,
    est_no_prefetch: u64,
    host_budget: Option<u64>,
    host_status: &'a str,
    prefetch_requested: bool,
    prefetch: bool,
}

impl TimeFooter<'_> {
    fn print(&self) {
        eprintln!(
            "\nWALL-TIME ACCOUNTING\n{}",
            self.phases.report(self.wall_ms)
        );
        eprintln!("  kernel launches: {}", self.launches);
        // Phase 1 §12: the mechanism gate. `stage` waits are the ones stream
        // ordering makes unnecessary and are 0 with --async-stages.
        eprintln!(
            "  host syncs: {} stage, {} pipeline ({} per launch)",
            self.stage_syncs,
            self.pipeline_syncs,
            if self.launches > 0 {
                format!(
                    "{:.3}",
                    (self.stage_syncs + self.pipeline_syncs) as f64 / self.launches as f64
                )
            } else {
                "-".into()
            }
        );
        eprintln!(
            "  max_hits: {} (target; resolved once; pinned unless --max-hits 0)\n  hit_capacity: {} (physical; success/failure only, never output bytes)\n  lifecycle: {} ref bins, \
 {} work units, {} builds, {} engines, {} ref uploads, {} query swaps",
            self.contract.max_hits,
            self.contract.hit_capacity,
            self.ref_bins,
            self.units,
            self.lifecycle.seed_table_builds,
            self.lifecycle.engine_creations,
            self.lifecycle.reference_uploads,
            self.lifecycle.query_swaps,
        );
        // Phase 3 mechanism: a stalled upload is one that had not finished when
        // its compute needed it, i.e. overlap that did not happen.
        eprintln!(
            "  seed uploads: {} ({} stalled{})",
            self.uploads,
            self.copy_stalls,
            if self.uploads > 0 {
                format!(
                    ", {:.2}%",
                    self.copy_stalls as f64 / self.uploads as f64 * 100.0
                )
            } else {
                String::new()
            }
        );
        eprintln!(
            "  output: {} files, {} bytes formatted, {} bytes written{}",
            self.files,
            self.bytes_in,
            self.bytes_out,
            if self.diagonal { " (-D)" } else { "" }
        );
        eprintln!("  peak RSS: {:>10} KiB", timing::peak_rss_kib());
        // Phase 1 §9: the host-budget decision, in the same units as the line
        // above so §11's validation is a subtraction.
        let mib = |b: u64| b as f64 / 1048576.0;
        eprintln!(
            "  host budget: estimated peak {:.0} MiB (shared {:.0} + {} worker(s), \
             {:.0} prefetching / {:.0} not), budget {}, status {}, \
             prefetch requested {} effective {}",
            mib(self.host_peak_est),
            mib(self.est_shared),
            self.workers,
            mib(self.est_prefetch),
            mib(self.est_no_prefetch),
            self.host_budget
                .map_or("unknown".to_string(), |b| format!("{:.0} MiB", mib(b))),
            self.host_status,
            self.prefetch_requested,
            self.prefetch,
        );
    }
}

/// Builds the planner's metadata from raw records: `id` and `ordinal` are the
/// input index, so `bin.record_ids` indexes straight back into `records`.
pub(crate) fn record_meta(records: &[(String, Vec<u8>)]) -> Vec<RecordMeta> {
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

/// One executed work unit for the `--time` unit ledger (round 90). GPU busy
/// is the `engine.phases.gpu_ms()` delta across the unit's own
/// `seed_and_filter_all` — no new device syncs, events resolve at the
/// existing pipeline boundaries. Shared reference setup is not charged here.
pub(crate) struct UnitLedgerRow {
    ordinal: u32,
    worker: usize,
    device: usize,
    reference_bin: u32,
    query_bin: u32,
    reference_bp: u64,
    query_bp: u64,
    gpu_ms: f64,
    host_start_ms: f64,
    host_end_ms: f64,
    pack_ms: f64,
    swap_ms: f64,
    seeds: u64,
    hits: u64,
    raw_hsps: u64,
    hsps: u64,
}

/// What one worker reports at join. Everything the serial executor used to
/// accumulate inline, now per worker and summed by the caller.
#[derive(Default)]
struct WorkerReport {
    stats: Stats,
    phases: Phases,
    lifecycle: Lifecycle,
    ledger: Vec<UnitLedgerRow>,
    /// Wall ms since run start when this worker returned (after its last send and
    /// engine teardown) — the per-worker critical path, unlike the last unit's end.
    finished_ms: f64,
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
    /// Per-pair gap totals merged across this worker's engines (round 72),
    /// plus the discarded >1,000 ms intervals (unit-transition work) as a
    /// separate count and sum so they stay visible instead of silent.
    gap_pairs: Vec<((&'static str, &'static str), f32, u64)>,
    gap_long_n: u64,
    gap_long_ms: f32,
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

/// Layout worker count for `-B 0`: requested GPUs clamped to visible devices,
/// so one device always resolves to the default layout. `forced` (the env hook
/// below) overrides the clamp for testing the W>=2 path on a one-GPU box.
pub(crate) fn layout_workers(gpus: usize, devices: usize, forced: Option<usize>) -> usize {
    if let Some(w) = forced.filter(|&w| w >= 1) {
        return w;
    }
    gpus.max(1).min(devices.max(1))
}

/// Debug/test hook: forces the `-B 0` layout worker count without touching
/// execution (worker threads, device ids, budgets are all derived from
/// `--gpus` as before). Deliberately ungated (no `cfg`): the release binary
/// must exercise the W>=2 path on a one-GPU box, and the layout line always
/// prints when it fires, so a stray setting is visible. Invalid values are
/// ignored.
fn forced_layout_workers() -> Option<usize> {
    std::env::var("HSPZ_LAYOUT_FORCE_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&w| w >= 1)
}

/// Round 90b `HSPZ_UNIT_PARTITION` policy: unset is auto, `0` forces whole-bin,
/// `1` forces the unit partition, anything else is a startup error.
///
/// Auto partitions iff `workers >= 2` and every device the run will use
/// reports the same [`DeviceProfile`] (SM count, clock, L2). The guard exists
/// because a heterogeneous pair pays -10.8% or +12.7% depending on which
/// device got which units, so the count-balanced partition must not silently
/// turn on there. Pure over its inputs so the decision is unit-testable with
/// no env access and no device; `profiles` holds one entry per used device
/// (worker `w` runs on `w % devices`), in ordinal order.
pub(crate) fn partition_policy(
    env: Option<&str>,
    workers: usize,
    profiles: &[DeviceProfile],
) -> Result<(bool, String), String> {
    let forced = match env {
        Some("1") => Some((true, "forced by HSPZ_UNIT_PARTITION=1")),
        Some("0") => Some((false, "forced by HSPZ_UNIT_PARTITION=0")),
        Some(v) => {
            return Err(format!(
                "HSPZ_UNIT_PARTITION must be unset, 0 or 1, got {v:?}"
            ));
        }
        None => None,
    };
    if let Some((on, why)) = forced {
        return Ok((on, why.to_string()));
    }
    if workers <= 1 {
        return Ok((false, "auto: W=1".to_string()));
    }
    let first = match profiles.first() {
        Some(p) => p,
        None => return Ok((false, "auto: no device profiles".to_string())),
    };
    // Workers sharing one device gain nothing from balancing and pay the replicas.
    if workers > profiles.len() {
        return Ok((
            false,
            format!(
                "auto: {workers} workers time-slice {} device{}",
                profiles.len(),
                if profiles.len() == 1 { "" } else { "s" }
            ),
        ));
    }
    // Same class = same SM count and L2 (the architecture/model); clocks within 10%,
    // because otherwise-identical cards from different vendors report different
    // nominal boost clocks (RTX 4090: 2520-2610 MHz) and the partition is worth
    // -12% on such a pair. A same-model card that is slower under load is not
    // detectable here (HSPZ_UNIT_PARTITION=0 restores whole-bin ownership).
    let same_class = |p: &DeviceProfile| {
        p.sms == first.sms
            && p.l2_bytes == first.l2_bytes
            && (p.clock_khz - first.clock_khz).abs() * 10 <= first.clock_khz.abs()
    };
    if let Some((i, p)) = profiles.iter().enumerate().find(|(_, p)| !same_class(p)) {
        return Ok((
            false,
            format!(
                "auto: device {i} differs from device 0 ({} vs {} SMs, {} vs {} MHz, {} vs {} MiB L2)",
                p.sms,
                first.sms,
                p.clock_khz / 1000,
                first.clock_khz / 1000,
                p.l2_bytes >> 20,
                first.l2_bytes >> 20,
            ),
        ));
    }
    Ok((
        true,
        format!(
            "auto: {} matching device{} ({} SMs, {} MHz, {} MiB L2; static attributes only, clocks within 10%)",
            profiles.len(),
            if profiles.len() == 1 { "" } else { "s" },
            first.sms,
            first.clock_khz / 1000,
            first.l2_bytes >> 20,
        ),
    ))
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

/// Packs one reference bin's records and builds its seed table — the host half
/// of a reference build. The single-query worker and the batch executor both
/// call it inline for the first bin and from the prefetch thread for the next,
/// so the two paths cannot drift.
fn build_ref_bin(
    rbin: &plan::Bin,
    ref_records: &[(String, Vec<u8>)],
    target_prefix: &str,
    shape: &Shape,
    step: u32,
    threads: usize,
) -> (PackedBin, SeedTable) {
    let packed = PackedBin::build(
        rbin.record_ids.iter().map(|&id| {
            let (n, s) = &ref_records[id as usize];
            (n.as_str(), s.as_slice())
        }),
        target_prefix,
        false,
    );
    let table = SeedTable::build_parallel(&packed.buf[..packed.block_len], shape, step, threads);
    (packed, table)
}

/// Runs one worker's visits on `device`, streaming finished units to the
/// emitter (§Phase 5: build/upload each visited reference once, reuse it
/// across its slice; no GPU is shared for performance).
///
/// This is the serial executor, parameterised by which visits it owns: with one
/// worker (or the whole-bin policy) every visit is a whole bin, which is what
/// makes `serial == multi-GPU` (§20) a property of the assignment rather than
/// of two code paths. One engine per visit, reused across the slice.
#[allow(clippy::too_many_arguments)]
fn run_bins(
    device: usize,
    worker: usize,
    visits: &[plan::Visit],
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
    run_started: Instant,
) -> Fallible<WorkerReport> {
    let mut rep = WorkerReport::default();
    if visits.is_empty() {
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
    let mut pending: Option<(PackedBin, SeedTable)> = None;

    for (visit_index, visit) in visits.iter().enumerate() {
        let rbin = &plan.reference_bins[visit.bin];
        let t = Instant::now();
        let (mut packed_ref, table) = match pending.take() {
            Some(built) => built,
            None => build_ref_bin(
                rbin,
                ref_records,
                &args.target_prefix,
                shape,
                args.step,
                threads,
            ),
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

        // The next visit *this worker owns* rides along with this visit's GPU work.
        let next_bin = visits
            .get(visit_index + 1)
            .map(|v| &plan.reference_bins[v.bin])
            .filter(|_| prefetch);
        std::thread::scope(|scope| -> Fallible<()> {
            let prefetch = next_bin.map(|nb| {
                scope.spawn(|| {
                    let t = Instant::now();
                    let built = build_ref_bin(
                        nb,
                        ref_records,
                        &args.target_prefix,
                        shape,
                        args.step,
                        threads,
                    );
                    (built, t.elapsed())
                })
            });

            for unit in plan.units.iter().filter(|u| {
                u.reference_bin == rbin.id && visit.queries.contains(&(u.query_bin as usize))
            }) {
                let qbin = &plan.query_bins[unit.query_bin as usize];
                let host_start_ms = run_started.elapsed().as_secs_f64() * 1000.0;
                let t = Instant::now();
                let packed_q = PackedBin::build(
                    qbin.record_ids.iter().map(|&id| {
                        let (n, s) = &qry_records[id as usize];
                        (n.as_str(), s.as_slice())
                    }),
                    &args.query_prefix,
                    true,
                );
                let pack_dur = t.elapsed();
                rep.phases.add("query pack", pack_dur);
                // Per-bin intervals + q_block_len (AM-B2). `intervals` is empty for a
                // block <= seed, and `q_block_len` is then never read; saturating
                // avoids the underflow the single-block path guards with an error.
                let intervals =
                    sequence::intervals(packed_q.block_len, shape.size, args.lastz_interval_size);
                let q_block_len = packed_q.block_len.saturating_sub(shape.size) as u32;

                let t = Instant::now();
                engine.swap_query(&packed_q.enc, &packed_q.enc_rc)?;
                let swap_dur = t.elapsed();
                rep.phases.add("swap_query", swap_dur);
                let qpass = QueryPass {
                    fwd: &packed_q.buf[..packed_q.block_len],
                    rc: &packed_q.rc,
                    intervals: &intervals,
                    q_block_len,
                };
                let gpu_before = engine.phases.gpu_ms();
                let pass =
                    seed_and_filter_all(&mut engine, &qpass, shape, transitions, args, threads)
                        .map_err(|e| {
                            format!(
                                "unit {} ref_bin {} query_bin {}: {e}",
                                unit.ordinal, rbin.id, qbin.id
                            )
                        })?;
                let busy_ms = engine.phases.gpu_ms() - gpu_before;
                let host_end_ms = run_started.elapsed().as_secs_f64() * 1000.0;

                rep.stats.seeds += pass.stats.seeds;
                rep.stats.seed_hits += pass.stats.seed_hits;
                rep.stats.raw_hsps += pass.stats.raw_hsps;
                rep.stats.hsps += pass.stats.hsps;
                rep.lifecycle.work_units_executed += 1;
                if args.time {
                    rep.ledger.push(UnitLedgerRow {
                        ordinal: unit.ordinal,
                        worker,
                        device,
                        reference_bin: rbin.id,
                        query_bin: qbin.id,
                        reference_bp: rbin.total_bp,
                        query_bp: qbin.total_bp,
                        gpu_ms: busy_ms,
                        host_start_ms,
                        host_end_ms,
                        pack_ms: pack_dur.as_secs_f64() * 1000.0,
                        swap_ms: swap_dur.as_secs_f64() * 1000.0,
                        seeds: pass.stats.seeds,
                        hits: pass.stats.seed_hits,
                        raw_hsps: pass.stats.raw_hsps,
                        hsps: pass.stats.hsps,
                    });
                }

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
            for ((a, b), ms, c) in engine.gap_pairs() {
                match rep.gap_pairs.iter_mut().find(|e| e.0 == (a, b)) {
                    Some(e) => {
                        e.1 += ms;
                        e.2 += c;
                    }
                    None => rep.gap_pairs.push(((a, b), ms, c)),
                }
            }
            let (ln, lms) = engine.discarded_gaps();
            rep.gap_long_n += ln;
            rep.gap_long_ms += lms;
        }
        // Engine-end autotune ledger: still in Auto means the chunks ran
        // out before a decision, so say so under --time.
        engine.finish_bucket_autotune();
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
    rep.finished_ms = run_started.elapsed().as_secs_f64() * 1000.0;
    Ok(rep)
}

// ---------------------------------------------------------------------------
// run

pub(crate) fn run(args: &RunArgs, pre_main_ms: f64, started: Instant) -> Fallible<Stats> {
    // Round 90b gate, read once here and validated before the `--cpu-only`
    // branch returns (a bad value is a startup error there too). The auto arm
    // is resolved with the other scheduling decisions below, where W and the
    // per-device profiles are known; until then the env value is just carried.
    let unit_env = std::env::var("HSPZ_UNIT_PARTITION").ok();
    partition_policy(unit_env.as_deref(), 1, &[])?;
    let mut phases = Phases::new();
    phases.add_ms("process startup", pre_main_ms);

    if args.cpu_only {
        return run_cpu_only(args, &mut phases, pre_main_ms, started);
    }
    if args.query_list.is_some() {
        return run_batch(args, &mut phases, pre_main_ms, started);
    }

    // Shared config, parsed once.
    let shape = Shape::parse(&args.seed)?;
    let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, args.scoring.as_deref())?;
    let transitions = !args.notransition;

    // Load both sides as raw records — no block-size guard (§10).
    let query_path = args
        .query
        .as_ref()
        .ok_or("--query is required without --query-list")?;
    let t = Instant::now();
    let (_, qry_records, _) = sequence::read_records(query_path)?;
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
    // `-B 0` is the automatic layout, resolved here before planning. It needs
    // LPT bins, so `--kegalign-bins` (sequential fill) is rejected in every
    // mode, including replay, where `-B` itself is otherwise ignored.
    if args.seq_block_size == 0 && args.kegalign_bins {
        return Err(plan::AUTO_KEGALIGN_ERROR.into());
    }
    let (plan, _worst, mut contract, res_seq, res_qry) = if let Some(m) = &loaded_manifest {
        let worst = m.check_fit(budget, shape.kmer_size)?;
        let contract = crate::gpu::ExecutionContract::from_resolved(m.max_hits, m.hsp_blocks);
        (
            m.plan.clone(),
            worst,
            contract,
            m.seq_block_size,
            m.query_block_size,
        )
    } else {
        let contract = crate::gpu::ExecutionContract::resolve(&ctx, args.max_hits, args.hsp_blocks);
        if args.seq_block_size == 0 {
            let w = layout_workers(args.gpus, devices, forced_layout_workers());
            if w <= 1 {
                // Trivial resolution: exactly today's default call, so a
                // one-worker `-B 0` run is byte-identical to `-B 500000000`.
                let q = args
                    .query_block_size
                    .map(u64::from)
                    .unwrap_or(plan::DEFAULT_BLOCK_TARGET);
                let (plan, worst) = plan::plan_within_budget(
                    &ref_meta,
                    &qry_meta,
                    plan::DEFAULT_BLOCK_TARGET,
                    q,
                    budget,
                    shape.kmer_size,
                    args.step,
                    contract.max_hits,
                    args.kegalign_bins,
                    args.wga_chunk_size,
                    transitions,
                )?;
                (plan, worst, contract, plan::DEFAULT_BLOCK_TARGET, q)
            } else {
                let auto = plan::auto_layout(
                    &ref_meta,
                    &qry_meta,
                    w,
                    args.gpus.max(1).min(ref_meta.len().max(1)),
                    budget,
                    timing::available_host_bytes().map(|b| b * 9 / 10),
                    args.query_block_size.map(u64::from),
                    &plan::AutoCtx {
                        kmer_size: shape.kmer_size,
                        step: args.step,
                        max_hits: contract.max_hits,
                        wga_chunk_size: args.wga_chunk_size,
                        transitions,
                        threads: resolve_threads(args.threads),
                        max_seeds: seed::max_seeds(args.wga_chunk_size, &shape, transitions),
                    },
                )?;
                match auto.fell_back {
                    Some((r, q)) => eprintln!(
                        "layout: auto W={w} -> default --seq-block-size {} --query-block-size {} \
                         (candidate R={r} Q={q} did not fit the device budget)",
                        plan::DEFAULT_BLOCK_TARGET,
                        plan::DEFAULT_BLOCK_TARGET
                    ),
                    None => eprintln!(
                        "{}",
                        plan::auto_layout_line(w, auto.seq_target, auto.query_target, &auto.plan)
                    ),
                }
                let (seq, qry) = (auto.seq_target, auto.query_target);
                (auto.plan, 0, contract, seq, qry)
            }
        } else {
            let q_target = args
                .query_block_size
                .map(u64::from)
                .unwrap_or(args.seq_block_size as u64);
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
            (plan, worst, contract, args.seq_block_size as u64, q_target)
        }
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
        write_plan_dump(path, &plan, &ref_records, &qry_records)?;
    }
    if let Some(path) = &args.dump_manifest {
        match &loaded_manifest {
            // Replay + dump: re-emit exactly what was loaded and validated,
            // never a reconstruction from this run's B/Q CLI defaults.
            Some(m) => write_manifest_dump(path, m)?,
            None => {
                // Fresh dump only: this is the one place `plan::executable_hash`
                // runs, so an ordinary run pays no hashing overhead.
                let m = plan_manifest(
                    args,
                    &contract,
                    &sub_mat,
                    res_seq,
                    res_qry,
                    &ref_records,
                    &qry_records,
                    &plan,
                    plan::executable_hash()?,
                );
                write_manifest_dump(path, &m)?;
            }
        }
    }

    // §18: reference bins to workers, deterministic LPT (whole-bin), or the
    // round-90 unit partition. `cost(R) = reference_bp x total_query_bp` is
    // monotone in the bin's own bp — LPT on `total_bp` is the same schedule with
    // less arithmetic (plan::assign_bins). W = 1 always takes today's path.
    let devices = crate::gpu::device_count().max(1);
    let workers = args.gpus.max(1).min(plan.reference_bins.len().max(1));
    // Round 90b: query each device the run will use (worker `w` maps to
    // `w % devices`, so the used ordinals are `0..min(workers, devices)`) once,
    // before any worker spawns. Only auto needs the profiles; a forced value
    // short-circuits in `partition_policy` without them.
    let profiles: Vec<DeviceProfile> = if unit_env.is_none() && workers > 1 {
        (0..workers.min(devices))
            .map(|d| crate::gpu::device_profile(d as i32))
            .collect::<Result<_, _>>()?
    } else {
        Vec::new()
    };
    let (unit_enabled, reason) = partition_policy(unit_env.as_deref(), workers, &profiles)?;
    let use_units = unit_enabled && workers > 1;
    let part: Vec<Vec<plan::Visit>> = if use_units {
        plan::unit_partition(&plan, workers)
    } else {
        let q = plan.query_bins.len();
        plan::assign_bins(&plan.reference_bins, workers)
            .into_iter()
            .map(|bins| {
                bins.into_iter()
                    .map(|id| plan::Visit {
                        bin: id as usize,
                        queries: 0..q,
                    })
                    .collect()
            })
            .collect()
    };
    // Distinct reference bins per worker: the host peak counts one bin build at
    // a time, never multiplied by the replica count.
    let visit_bins: Vec<Vec<u32>> = part
        .iter()
        .map(|visits| {
            let mut bins: Vec<u32> = visits.iter().map(|t| t.bin as u32).collect();
            bins.sort_unstable();
            bins.dedup();
            bins
        })
        .collect();
    let total_visits: usize = part.iter().map(Vec::len).sum();
    eprintln!(
        "schedule: policy={} W={workers} visits={total_visits} extra_replicas={} ({reason})",
        if use_units {
            "unit-partition"
        } else {
            "whole-bin"
        },
        total_visits.saturating_sub(plan.reference_bins.len())
    );
    for (x, visits) in part.iter().enumerate() {
        let units: usize = visits.iter().map(|t| t.queries.len()).sum();
        let desc = visits
            .iter()
            .map(|t| format!("R{}[{}..{})", t.bin, t.queries.start, t.queries.end))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "schedule: worker {x} device {}: {desc} units={units} visits={}",
            x % devices,
            visits.len()
        );
    }
    eprintln!(
        "schedule: device mapping: {}",
        (0..workers)
            .map(|x| format!("worker{x}->device{}", x % devices))
            .collect::<Vec<_>>()
            .join(" ")
    );
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
        let fits = plan::host_preflight(&est, &visit_bins, available)?;
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
    let host_peak_est = plan::host_peak(&est, &visit_bins, prefetch);

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
        for (w, visits) in part.iter().enumerate() {
            let tx = tx.clone();
            let (plan, ref_records, qry_records, shape, sub_mat) =
                (&plan, &ref_records, &qry_records, &shape, &sub_mat);
            // `Box<dyn Error>` is not `Send`, so a worker reports failure as a
            // string and the caller turns it back into an error.
            handles.push(scope.spawn(move || {
                run_bins(
                    w % devices,
                    w,
                    visits,
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
                    started,
                )
                .map_err(|e| e.to_string())
            }));
        }
        drop(tx);

        let mut buffered: std::collections::BTreeMap<u32, UnitOutput> =
            std::collections::BTreeMap::new();
        let mut next = 0u32;
        for unit in rx {
            let ord = unit.ordinal;
            // The ordinal alone is not identity: the unit must be the plan's unit.
            let pu = plan.units.get(ord as usize).ok_or_else(|| {
                format!(
                    "work unit ordinal {ord} is outside the plan ({} units)",
                    plan.units.len()
                )
            })?;
            if (pu.reference_bin, pu.query_bin) != (unit.reference_bin, unit.query_bin) {
                return Err(format!(
                    "work unit ordinal {ord} arrived as R{} Q{} but the plan has R{} Q{}",
                    unit.reference_bin, unit.query_bin, pu.reference_bin, pu.query_bin
                )
                .into());
            }
            if ord < next || buffered.insert(ord, unit).is_some() {
                return Err(format!("duplicate work unit ordinal {ord} from a worker").into());
            }
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
        let mut out = Vec::new();
        for h in handles {
            out.push(h.join().expect("gpu worker panicked")?);
        }
        // Join first so a worker's own error (OOM, CUDA failure) is reported
        // instead of the completeness check it also trips (codex review, round 90).
        if !buffered.is_empty() {
            return Err(format!(
                "emitter has {} unit(s) it can never reach: expected ordinal {next}, \
                 hold {:?} — a worker died without sending (§19)",
                buffered.len(),
                buffered.keys().collect::<Vec<_>>()
            )
            .into());
        }
        if next != plan.units.len() as u32 {
            return Err(format!(
                "emitter reached ordinal {next} of {} work units — a unit ran twice or never",
                plan.units.len()
            )
            .into());
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
    // Residual attribution: per-worker notes that keep every existing line
    // above unchanged (cycle-4/5 report scripts still parse). Printed under
    // --time for any worker count, so a single-worker run still shows them.
    if args.time {
        let pairs: Vec<String> = reports
            .iter()
            .map(|r| {
                let mut v = r.gap_pairs.clone();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                let top: Vec<String> = v
                    .iter()
                    .take(5)
                    .map(|((a, b), ms, n)| format!("({a}->{b}, {ms:.1}, {n})"))
                    .collect();
                format!("[{}]", top.join(", "))
            })
            .collect();
        eprintln!("note: per-worker gap pairs: {pairs:?}");
        let long_n: Vec<u64> = reports.iter().map(|r| r.gap_long_n).collect();
        let long_ms: Vec<f32> = reports.iter().map(|r| r.gap_long_ms).collect();
        eprintln!("note: per-worker long transitions: n={long_n:?} sum={long_ms:?} ms");
        let qp: Vec<f64> = reports.iter().map(|r| r.phases.ms("query pack")).collect();
        let sq: Vec<f64> = reports.iter().map(|r| r.phases.ms("swap_query")).collect();
        let cp: Vec<f64> = reports
            .iter()
            .map(|r| r.phases.ms("chunk prep (lower_bound)"))
            .collect();
        let st: Vec<f64> = reports
            .iter()
            .map(|r| r.seed_table_ms.as_secs_f64() * 1000.0)
            .collect();
        // Standalone (non-overlapped) reference prep lives in the worker report,
        // not in the worker Phases (that name is added globally after this note).
        let rp: Vec<f64> = reports
            .iter()
            .map(|r| r.prefetched_ms.as_secs_f64() * 1000.0)
            .collect();
        eprintln!(
            "note: per-worker host phases ms: [query pack={qp:?}, swap_query={sq:?}, \
             chunk prep (lower_bound)={cp:?}, seed table build={st:?}, \
             reference bin prep (standalone)={rp:?}]"
        );
        // Round 90 unit ledger: one tab-separated line per executed unit in
        // ordinal order, then per-worker totals. The summed unit busy must agree
        // with the per-worker gpu-busy note within 1% (warning only).
        let mut rows: Vec<&UnitLedgerRow> = reports.iter().flat_map(|r| r.ledger.iter()).collect();
        rows.sort_by_key(|r| r.ordinal);
        eprintln!(
            "unit ledger:\tordinal\tworker\tdevice\tref_bin\tquery_bin\tref_bp\tquery_bp\t\
             gpu_ms\thost_start_ms\thost_end_ms\tpack_ms\tswap_ms\tseeds\thits\traw_hsps\thsps"
        );
        for r in &rows {
            eprintln!(
                "unit ledger:\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}",
                r.ordinal,
                r.worker,
                r.device,
                r.reference_bin,
                r.query_bin,
                r.reference_bp,
                r.query_bp,
                r.gpu_ms,
                r.host_start_ms,
                r.host_end_ms,
                r.pack_ms,
                r.swap_ms,
                r.seeds,
                r.hits,
                r.raw_hsps,
                r.hsps
            );
        }
        for (x, rep) in reports.iter().enumerate() {
            let busy: f64 = rep.ledger.iter().map(|r| r.gpu_ms).sum();
            let host: f64 = rep
                .ledger
                .iter()
                .map(|r| r.host_end_ms - r.host_start_ms)
                .sum();
            let end: f64 = rep.ledger.iter().map(|r| r.host_end_ms).fold(0.0, f64::max);
            let gpu = rep.phases.gpu_ms();
            eprintln!(
                "unit ledger: worker {x} finished at {:.2} ms (last unit end {end:.2}), busy {busy:.2} ms, unit-host {host:.2} ms",
                rep.finished_ms
            );
            eprintln!(
                "unit ledger: worker {x} self-check unit_busy_sum={busy:.2} ms gpu_busy={gpu:.2} ms"
            );
            let tol = 0.01 * gpu.max(1e-9);
            if (busy - gpu).abs() > tol {
                eprintln!(
                    "unit ledger: worker {x} WARNING unit busy and gpu-busy differ by over 1%"
                );
            }
            debug_assert!((busy - gpu).abs() <= tol, "unit ledger self-check");
        }
    }

    for (x, r) in reports.iter().enumerate() {
        let vw = part[x].len() as u32;
        let uw: u32 = part[x].iter().map(|t| t.queries.len() as u32).sum();
        r.lifecycle
            .check(vw, uw)
            .map_err(|e| format!("worker {x}: {e}"))?;
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
    lifecycle.check(total_visits as u32, plan.units.len() as u32)?;

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
        TimeFooter {
            phases: &phases,
            wall_ms: wall,
            launches,
            stage_syncs,
            pipeline_syncs,
            contract: &contract,
            ref_bins: plan.reference_bins.len(),
            units: plan.units.len(),
            lifecycle: &lifecycle,
            uploads,
            copy_stalls,
            files: out.files,
            bytes_in: out.bytes_in,
            bytes_out: if out.bytes_out > 0 {
                out.bytes_out
            } else {
                out.bytes_in
            },
            diagonal: args.diagonal_partition,
            host_peak_est,
            est_shared: est.shared,
            workers,
            est_prefetch: est.per_worker_prefetch,
            est_no_prefetch: est.per_worker_no_prefetch,
            host_budget,
            host_status,
            prefetch_requested,
            prefetch,
        }
        .print();
    }
    if let Some(a) = audit.as_ref() {
        eprintln!("\n(ALL REFERENCE BINS) {}", a.report());
    }
    if args.hit_stats {
        eprintln!("\nHITS PER SEED\n{}", hit_stats.report());
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Bin-major batch: one reference × many queries (`--query-list`), W=1 only.
//
// Every job is planned independently exactly as `run -q` would (same
// `plan_within_budget` call, same once-resolved `--max-hits`), and the batch
// aborts before any GPU work unless all jobs share identical reference bins.
// Execution is bin-major: each reference bin is built and uploaded once (with
// the existing one-bin-ahead prefetch), then every job's query blocks for that
// bin run through the unchanged `seed_and_filter_all` on their original units.
// Each job owns one `Emitter` (+ `-D` history) fed in its standalone ordinal
// order via a per-job cursor in `JobRouter`.
//
// A batch is all-or-nothing: any job's failure (`?`, OOM, CUDA, router, pack)
// unwinds `run_batch`. Per-job directories are left with whatever files were
// already written, a `-Z` archive is left truncated (`TarGzSink::finish` never
// runs), later jobs on later bins never run, and `OUT/queries.tsv` is only
// written after every emitter finishes. This is documented in the
// `--query-list` CLI help as well.

/// Batch-mode flag validation, pure so it is unit-testable without a GPU.
pub(crate) fn validate_batch_args(args: &RunArgs) -> Result<(), String> {
    if args.gpus > 1 {
        return Err("batch mode is W=1 in this release".into());
    }
    if args.seq_block_size == 0 {
        return Err("batch mode rejects -B 0 (automatic layout): give -B explicitly".into());
    }
    if args.kegalign_bins {
        return Err("batch mode rejects --kegalign-bins".into());
    }
    if args.from_manifest.is_some() {
        return Err("batch mode rejects --from-manifest".into());
    }
    Ok(())
}

/// Reads a `--query-list` file: one query FASTA path per line, blank lines and
/// `#` comments ignored. Relative paths resolve against the list file's dir.
pub(crate) fn read_query_list(path: &std::path::Path) -> Fallible<Vec<PathBuf>> {
    let text = std::fs::read_to_string(path)?;
    let base = path.parent().filter(|p| !p.as_os_str().is_empty());
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let p = PathBuf::from(line);
        out.push(match base {
            Some(dir) if p.is_relative() => dir.join(p),
            _ => p,
        });
    }
    if out.is_empty() {
        return Err(format!("query list {} has no queries", path.display()).into());
    }
    Ok(out)
}

/// Reference-bin compatibility across batch jobs: identical count, ids,
/// membership and sizes. Anything else would silently replan one job into a
/// different dedup scope, so the batch aborts naming the job instead.
pub(crate) fn check_reference_bins_compatible(
    first: &[plan::Bin],
    job: &[plan::Bin],
    job_index: usize,
    job_path: &std::path::Path,
) -> Result<(), String> {
    if first.len() != job.len()
        || first
            .iter()
            .zip(job.iter())
            .any(|(a, b)| a.id != b.id || a.total_bp != b.total_bp || a.record_ids != b.record_ids)
    {
        return Err(format!(
            "batch: job {:06} ({}): reference bins differ from job 000001; \
             batch only admits jobs with identical reference bins",
            job_index + 1,
            job_path.display()
        ));
    }
    Ok(())
}

/// Output router: one next-ordinal cursor per job. A single global cursor
/// would buffer every later job while bin-major execution waits on the first,
/// so each job advances in its own standalone ordinal order.
pub(crate) struct JobRouter {
    next: Vec<u32>,
    totals: Vec<u32>,
}

impl JobRouter {
    pub(crate) fn new(totals: Vec<u32>) -> Self {
        Self {
            next: vec![0; totals.len()],
            totals,
        }
    }

    /// Admits `(job, ordinal)` iff it is that job's next expected unit *and*
    /// the bin tuple matches the job plan's unit at that ordinal.
    pub(crate) fn accept(
        &mut self,
        job: usize,
        ordinal: u32,
        reference_bin: u32,
        query_bin: u32,
        plan: &plan::Plan,
    ) -> Result<(), String> {
        if job >= self.next.len() {
            return Err(format!("batch router: unknown job {job}"));
        }
        let want = self.next[job];
        if ordinal < want {
            return Err(format!(
                "batch router: duplicate work unit ordinal {ordinal} for job {:06}",
                job + 1
            ));
        }
        if ordinal > want {
            return Err(format!(
                "batch router: missing work unit for job {:06}: expected ordinal {want}, got {ordinal}",
                job + 1
            ));
        }
        if ordinal >= self.totals[job] {
            return Err(format!(
                "batch router: work unit ordinal {ordinal} is outside job {:06}'s plan ({} units)",
                job + 1,
                self.totals[job]
            ));
        }
        let pu = &plan.units[ordinal as usize];
        if (pu.reference_bin, pu.query_bin) != (reference_bin, query_bin) {
            return Err(format!(
                "batch router: job {:06} ordinal {ordinal} arrived as R{reference_bin} Q{query_bin} \
                 but the plan has R{} Q{}",
                job + 1,
                pu.reference_bin,
                pu.query_bin
            ));
        }
        self.next[job] += 1;
        Ok(())
    }

    /// True once every job's cursor reached its plan length.
    pub(crate) fn complete(&self) -> bool {
        self.next == self.totals
    }
}

/// Per-job sibling of a batch dump path: append `.{job:06}` to the full file
/// name, so `--dump-manifest out.manifest` writes `out.manifest.000001` and
/// `--dump-plan`/`--dump-raw`/the `HSPZ_ANCHOR_CENSUS` dump all agree. Never
/// `Path::with_extension`, which would replace `.manifest` with `.000001`.
/// `job` is the zero-based loop index.
pub(crate) fn job_sibling(path: &std::path::Path, job: usize) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(format!(".{:06}", job + 1));
    PathBuf::from(name)
}

/// SHA-256 of a file via the existing `sha256sum` binary (no new crates).
fn sha256_file(path: &std::path::Path) -> Fallible<String> {
    let out = std::process::Command::new("sha256sum").arg(path).output()?;
    if !out.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| format!("sha256sum gave no digest for {}", path.display()).into())
}

#[allow(clippy::too_many_lines)]
fn run_batch(
    args: &RunArgs,
    phases: &mut Phases,
    pre_main_ms: f64,
    started: Instant,
) -> Fallible<Stats> {
    validate_batch_args(args).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let list_path = args
        .query_list
        .as_ref()
        .ok_or("--query-list is required for batch mode")?;
    let job_paths = read_query_list(list_path)?;

    let shape = Shape::parse(&args.seed)?;
    let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, args.scoring.as_deref())?;
    let transitions = !args.notransition;

    let t = Instant::now();
    let (_, ref_records, ref_bytes) = sequence::read_records(&args.reference)?;
    phases.add("input.reference", t.elapsed());
    let ref_meta = record_meta(&ref_records);
    let ref_bp_total: u64 = ref_meta.iter().map(|r| r.len).sum();

    // Resolve `--max-hits 0` ONCE for the whole batch (design: never from
    // whichever worker happens to run a unit — here there is one device).
    let t = Instant::now();
    let ctx = CudaContext::new(0)?;
    phases.add("CUDA context init", t.elapsed());
    // `plan` mirrors the single-query path exactly: device probe, worker budget
    // and contract resolution, then each job's own `plan_within_budget`, then
    // the shared hit capacity. Never the CUDA context or the query loads — the
    // query reads are charged to `input.query` (once per job), as standalone.
    let t = Instant::now();
    let devices = crate::gpu::device_count().max(1);
    let free = crate::gpu::min_free_bytes(1)?;
    let workers_upper = 1usize.min(ref_meta.len().max(1));
    let budget = plan::worker_device_budget(free, workers_upper, devices);
    let mut contract = crate::gpu::ExecutionContract::resolve(&ctx, args.max_hits, args.hsp_blocks);
    let resolved_max_hits = contract.max_hits;
    let q_target = args
        .query_block_size
        .map(u64::from)
        .unwrap_or(args.seq_block_size as u64);
    let res_seq = args.seq_block_size as u64;
    let mut plan_ms = t.elapsed();

    // Plan EVERY job independently with the same call the single path uses.
    let mut qry_records_list: Vec<Vec<(String, Vec<u8>)>> = Vec::with_capacity(job_paths.len());
    let mut plans: Vec<plan::Plan> = Vec::with_capacity(job_paths.len());
    let mut worsts: Vec<u64> = Vec::with_capacity(job_paths.len());
    for path in &job_paths {
        let tq = Instant::now();
        let (_, recs, _) = sequence::read_records(path)?;
        phases.add("input.query", tq.elapsed());
        let tp = Instant::now();
        let meta = record_meta(&recs);
        let (p, worst) = plan::plan_within_budget(
            &ref_meta,
            &meta,
            res_seq,
            q_target,
            budget,
            shape.kmer_size,
            args.step,
            resolved_max_hits,
            false,
            args.wga_chunk_size,
            transitions,
        )
        .map_err(|e| {
            format!(
                "batch: job {} ({}): {e}",
                qry_records_list.len() + 1,
                path.display()
            )
        })?;
        qry_records_list.push(recs);
        plans.push(p);
        worsts.push(worst);
        plan_ms += tp.elapsed();
    }
    let tp = Instant::now();
    // The reference bins must be identical across all jobs — abort before any
    // GPU work, naming the job.
    for (j, (p, path)) in plans.iter().zip(&job_paths).enumerate().skip(1) {
        check_reference_bins_compatible(&plans[0].reference_bins, &p.reference_bins, j, path)?;
    }

    // Physical capacity shared by the batch's engines: per-plan candidates
    // never lower the semantic cap, so take the minimum that still fits every
    // job (success/failure only, never output bytes).
    {
        let mut shared: Option<u32> = None;
        for p in &plans {
            let candidate = plan::max_hit_capacity(
                p,
                budget,
                shape.kmer_size,
                args.step,
                resolved_max_hits,
                args.wga_chunk_size,
                transitions,
            )?;
            shared = Some(shared.map_or(candidate, |s: u32| s.min(candidate)));
        }
        contract.hit_capacity = crate::gpu::clamp_hit_capacity(
            contract.max_hits,
            shared.unwrap_or(contract.max_hits),
            contract.hsp_blocks,
        )?;
    }
    plan_ms += tp.elapsed();
    phases.add("plan", plan_ms);

    if let Some(path) = &args.dump_plan {
        for (j, p) in plans.iter().enumerate() {
            write_plan_dump(&job_sibling(path, j), p, &ref_records, &qry_records_list[j])?;
        }
    }
    if let Some(path) = &args.dump_manifest {
        // One executable hash for every job's manifest. Batch rejects
        // `--from-manifest`, so every manifest here is a fresh dump.
        let executable_hash = plan::executable_hash()?;
        for (j, p) in plans.iter().enumerate() {
            let m = plan_manifest(
                args,
                &contract,
                &sub_mat,
                res_seq,
                q_target,
                &ref_records,
                &qry_records_list[j],
                p,
                executable_hash,
            );
            write_manifest_dump(&job_sibling(path, j), &m)?;
        }
    }

    // Host preflight over the WHOLE batch: all queries' records stay in RAM,
    // packed per unit from the held records (never re-read per bin).
    let qry_bp_sum: u64 = qry_records_list
        .iter()
        .map(|recs| recs.iter().map(|(_, s)| s.len() as u64).sum::<u64>())
        .sum();
    let largest_ref = plans[0]
        .reference_bins
        .iter()
        .map(|b| b.total_bp)
        .max()
        .unwrap_or(0);
    let largest_qry = plans
        .iter()
        .flat_map(|p| p.query_bins.iter().map(|b| b.total_bp))
        .max()
        .unwrap_or(0);
    let threads = resolve_threads(args.threads);
    let max_seeds = seed::max_seeds(args.wga_chunk_size, &shape, transitions);
    // Shared formula with the single-query path; the two maxima span every
    // job here, because the batch holds all of them in RAM at once.
    let est = plan::host_estimate_sizes(
        largest_ref,
        largest_qry,
        ref_bp_total,
        qry_bp_sum,
        shape.kmer_size,
        args.step,
        threads,
        max_seeds,
    );
    let ref_bins_all: Vec<u32> = plans[0].reference_bins.iter().map(|b| b.id).collect();
    let assignment: Vec<Vec<u32>> = vec![ref_bins_all];
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
            eprintln!("note: host preflight disabled reference prefetch (batch W=1)");
        }
        prefetch &= fits;
    }
    let host_peak_est = plan::host_peak(&est, &assignment, prefetch);

    // One emitter (+ `-D` history) per job, each starting empty.
    std::fs::create_dir_all(&args.output)?;
    let batch_tarball = args.tarball.is_some();
    if let Some(p) = &args.tarball {
        // Bare `-Z` (clap's `-` sentinel) and an empty value mean the default
        // per-job archive layout; any other path cannot be honoured because a
        // batch writes one archive per job.
        if !p.as_os_str().is_empty() && p.as_os_str() != "-" {
            eprintln!(
                "note: batch mode writes one archive per job (OUT/000001.tar.gz …); \
                 the -Z path value is ignored"
            );
        }
    }
    let audit_base = crate::census::SurvivorAudit::dump_path();
    let mut emitters: Vec<Emitter> = Vec::with_capacity(job_paths.len());
    for (j, _) in job_paths.iter().enumerate() {
        let tag = format!("{:06}", j + 1);
        let (dir, tar) = if batch_tarball {
            (
                args.output.join(&tag),
                Some(args.output.join(format!("{tag}.tar.gz"))),
            )
        } else {
            (args.output.join(&tag), None)
        };
        let audit_path = audit_base.as_ref().map(|p| job_sibling(p, j));
        emitters.push(Emitter::new_at(
            &dir,
            tar.as_deref(),
            args.diagonal_partition,
            audit_path.as_deref(),
        )?);
    }
    let mut router = JobRouter::new(plans.iter().map(|p| p.units.len() as u32).collect());
    let mut job_stats: Vec<Stats> = vec![Stats::default(); job_paths.len()];
    let mut job_raw: Vec<Vec<(char, Vec<SegmentPair>)>> =
        (0..job_paths.len()).map(|_| Vec::new()).collect();
    let mut job_first_ms: Vec<Option<f64>> = vec![None; job_paths.len()];
    let mut job_done_ms: Vec<f64> = vec![0.0; job_paths.len()];
    let mut job_hit_stats: Vec<crate::gpu::HitStats> = (0..job_paths.len())
        .map(|_| crate::gpu::HitStats::default())
        .collect();
    let mut job_audits: Vec<Option<crate::census::SurvivorAudit>> =
        (0..job_paths.len()).map(|_| None).collect();

    let total_units: usize = plans.iter().map(|p| p.units.len()).sum();
    let r_bins = plans[0].reference_bins.len();
    eprintln!(
        "schedule: policy=whole-bin W=1 visits={r_bins} extra_replicas=0 (batch W=1; auto: W=1)"
    );
    {
        let desc = plans[0]
            .reference_bins
            .iter()
            .map(|b| format!("R{}(Qbatch)", b.id))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("schedule: worker 0 device 0: {desc} units={total_units} visits={r_bins}");
    }
    eprintln!("schedule: device mapping: worker0->device0");

    let device_seeds = device_seeds_for(1);
    if device_seeds {
        eprintln!("note: generating query seeds on the device (1 worker)");
    }

    let mut pending: Option<(plan::PackedBin, SeedTable)> = None;
    let mut lifecycle = Lifecycle::default();
    let mut launches = 0u64;
    let (mut stage_syncs, mut pipeline_syncs) = (0u64, 0u64);
    let (mut uploads, mut copy_stalls) = (0u64, 0u64);
    let mut seed_table_ms = Duration::ZERO;
    let mut prefetched_ms = Duration::ZERO;

    for (visit_index, rbin) in plans[0].reference_bins.iter().enumerate() {
        let t = Instant::now();
        let (mut packed_ref, table) = match pending.take() {
            Some(built) => built,
            None => build_ref_bin(
                rbin,
                &ref_records,
                &args.target_prefix,
                &shape,
                args.step,
                threads,
            ),
        };
        seed_table_ms += t.elapsed();
        lifecycle.seed_table_builds += 1;

        let cfg = EngineConfig {
            index_table: &table.index_table,
            pos_table: &table.pos_table,
            ref_seq: &packed_ref.enc,
            sub_mat: &sub_mat,
            seed_size: shape.size as u32,
            xdrop: args.xdrop,
            hspthresh: args.hspthresh,
            noentropy: args.noentropy,
            max_hits: contract.max_hits,
            hit_capacity: contract.hit_capacity,
            timing: args.time,
            hsp_blocks: contract.hsp_blocks,
        };
        let mut engine = Engine::new(&ctx, cfg, &mut *phases)?;
        lifecycle.engine_creations += 1;
        engine.dump_raw = args.dump_raw.is_some();
        engine.device_seeds = device_seeds;
        engine.collect_hit_stats = args.hit_stats;
        engine.persistent_seed_buffers = !args.no_persistent_seed_buffers;
        engine.async_stages = !args.no_async_stages;

        let ref_chrs = std::mem::take(&mut packed_ref.chrs);
        drop(packed_ref);
        drop(table);

        // The next reference bin builds on a worker thread while this bin's
        // units run on the GPU — the existing one-bin-ahead prefetch, kept at
        // W=1 (`std::thread::scope` so the closure can borrow the records).
        let next_bin = plans[0]
            .reference_bins
            .get(visit_index + 1)
            .filter(|_| prefetch);
        std::thread::scope(|scope| -> Fallible<()> {
            let prefetch = next_bin.map(|nb| {
                scope.spawn(|| {
                    let t = Instant::now();
                    let built = build_ref_bin(
                        nb,
                        &ref_records,
                        &args.target_prefix,
                        &shape,
                        args.step,
                        threads,
                    );
                    (built, t.elapsed())
                })
            });

            // Bin-major: for this reference bin, every job's units at this bin
            // in that job's ordinal order (ref-outer ordinals make this
            // per-job ordered, so each job's `-D` history matches standalone).
            for (j, plan_j) in plans.iter().enumerate() {
                let units: Vec<plan::WorkUnit> = plan_j
                    .units
                    .iter()
                    .filter(|u| u.reference_bin == rbin.id)
                    .copied()
                    .collect();
                for unit in units {
                    let qbin = &plan_j.query_bins[unit.query_bin as usize];
                    let t = Instant::now();
                    let packed_q = plan::PackedBin::build(
                        qbin.record_ids.iter().map(|&id| {
                            let (n, s) = &qry_records_list[j][id as usize];
                            (n.as_str(), s.as_slice())
                        }),
                        &args.query_prefix,
                        true,
                    );
                    phases.add("query pack", t.elapsed());
                    let intervals = sequence::intervals(
                        packed_q.block_len,
                        shape.size,
                        args.lastz_interval_size,
                    );
                    let q_block_len = packed_q.block_len.saturating_sub(shape.size) as u32;

                    let t = Instant::now();
                    engine.swap_query(&packed_q.enc, &packed_q.enc_rc)?;
                    phases.add("swap_query", t.elapsed());
                    let qpass = QueryPass {
                        fwd: &packed_q.buf[..packed_q.block_len],
                        rc: &packed_q.rc,
                        intervals: &intervals,
                        q_block_len,
                    };
                    let pass = seed_and_filter_all(
                        &mut engine,
                        &qpass,
                        &shape,
                        transitions,
                        args,
                        threads,
                    )
                    .map_err(|e| {
                        format!(
                            "batch: job {:06} unit {} ref_bin {} query_bin {}: {e}",
                            j + 1,
                            unit.ordinal,
                            rbin.id,
                            qbin.id
                        )
                    })?;
                    // Per-job diagnostic attribution (byte-neutral): drain what
                    // this unit added into its job's accumulators.
                    if args.hit_stats {
                        job_hit_stats[j].merge(&std::mem::take(&mut engine.hit_stats));
                    }
                    if let Some(c) = engine.census.as_mut() {
                        job_audits[j]
                            .get_or_insert_with(crate::census::SurvivorAudit::default)
                            .merge(&std::mem::take(c));
                    }

                    router
                        .accept(j, unit.ordinal, unit.reference_bin, unit.query_bin, plan_j)
                        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                    emitters[j].emit_unit(
                        unit.reference_bin,
                        unit.query_bin,
                        &ref_chrs,
                        &packed_q.chrs,
                        &packed_q.rc_chrs,
                        &pass,
                    )?;
                    if args.dump_raw.is_some() {
                        job_raw[j].extend_from_slice(&pass.raw);
                    }
                    let now_ms = started.elapsed().as_secs_f64() * 1000.0;
                    if job_first_ms[j].is_none() {
                        job_first_ms[j] = Some(now_ms);
                    }
                    job_done_ms[j] = now_ms;
                    job_stats[j].seeds += pass.stats.seeds;
                    job_stats[j].seed_hits += pass.stats.seed_hits;
                    job_stats[j].raw_hsps += pass.stats.raw_hsps;
                    job_stats[j].hsps += pass.stats.hsps;
                    lifecycle.work_units_executed += 1;
                    lifecycle.query_swaps += 1;
                }
            }

            if let Some(prefetch) = prefetch {
                let t = Instant::now();
                let (built, standalone) = prefetch.join().expect("reference prefetch panicked");
                seed_table_ms += t.elapsed();
                prefetched_ms += standalone;
                pending = Some(built);
            }
            Ok(())
        })?;

        let got_swaps = engine.query_swaps();
        let want_swaps: u32 = plans
            .iter()
            .map(|p| {
                p.units
                    .iter()
                    .filter(|u| u.reference_bin == rbin.id)
                    .count() as u32
            })
            .sum();
        if got_swaps != want_swaps {
            return Err(format!(
                "batch: reference bin {}: engine query_swaps = {got_swaps}, expected {want_swaps}",
                rbin.id
            )
            .into());
        }
        lifecycle.reference_uploads += engine.reference_uploads();
        engine.finish_bucket_autotune();
        launches += engine.launches;
        stage_syncs += engine.stage_syncs();
        pipeline_syncs += engine.pipeline_syncs();
        let (u, st) = engine.seed_copy_stats();
        uploads += u;
        copy_stalls += st;
        if let Some(c) = engine.census.as_mut() {
            eprintln!("\n(reference bin {}) {}", rbin.id, c.report());
        }
        phases.merge(&engine.phases);
    }
    if !router.complete() {
        return Err("batch: emitter did not reach every job's final ordinal".into());
    }
    lifecycle.check(r_bins as u32, total_units as u32)?;

    phases.add("seed table build", seed_table_ms);
    if prefetched_ms > Duration::ZERO {
        phases.add_overlapped("reference bin prep (standalone)", prefetched_ms);
    }

    if let Some(path) = &args.dump_raw {
        for (j, raw) in job_raw.iter().enumerate() {
            let mut f = std::io::BufWriter::new(std::fs::File::create(job_sibling(path, j))?);
            for (strand, hsps) in raw {
                for h in hsps {
                    writeln!(
                        f,
                        "{strand}\t{}\t{}\t{}\t{}",
                        h.ref_start, h.query_start, h.len, h.score
                    )?;
                }
            }
        }
    }

    let mut total_files = 0usize;
    let mut total_bytes_in = 0u64;
    let mut total_bytes_out = 0u64;
    for emitter in emitters {
        let out = emitter.finish(&mut *phases)?;
        total_files += out.files;
        total_bytes_in += out.bytes_in;
        total_bytes_out += out.bytes_out.max(out.bytes_in);
    }

    // OUT/queries.tsv: per-job identity and completion.
    {
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(args.output.join("queries.tsv"))?);
        writeln!(
            f,
            "job\tpath\tbytes\tsha256\tblocks\tunits\thsps\tcompletion_ms"
        )?;
        for (j, path) in job_paths.iter().enumerate() {
            let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let sha = sha256_file(path)?;
            writeln!(
                f,
                "{:06}\t{}\t{bytes}\t{sha}\t{}\t{}\t{}\t{:.2}",
                j + 1,
                path.display(),
                plans[j].query_bins.len(),
                plans[j].units.len(),
                job_stats[j].hsps,
                job_done_ms[j],
            )?;
        }
        f.flush()?;
    }

    let mut stats = Stats::default();
    for s in &job_stats {
        stats.seeds += s.seeds;
        stats.seed_hits += s.seed_hits;
        stats.raw_hsps += s.raw_hsps;
        stats.hsps += s.hsps;
    }
    report_counts(&stats);
    for (j, s) in job_stats.iter().enumerate() {
        eprintln!(
            "job {:06}: blocks={} units={} hsps={} first_output_ms={:.2} completed_ms={:.2}",
            j + 1,
            plans[j].query_bins.len(),
            plans[j].units.len(),
            s.hsps,
            job_first_ms[j].unwrap_or(job_done_ms[j]),
            job_done_ms[j],
        );
    }
    eprintln!(
        "batch: jobs={} units={total_units} wall_ms={:.2}",
        job_paths.len(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    if args.time {
        let wall = pre_main_ms + started.elapsed().as_secs_f64() * 1000.0;
        TimeFooter {
            phases,
            wall_ms: wall,
            launches,
            stage_syncs,
            pipeline_syncs,
            contract: &contract,
            ref_bins: r_bins,
            units: total_units,
            lifecycle: &lifecycle,
            uploads,
            copy_stalls,
            files: total_files,
            bytes_in: total_bytes_in,
            bytes_out: total_bytes_out,
            diagonal: args.diagonal_partition,
            host_peak_est,
            est_shared: est.shared,
            workers: 1,
            est_prefetch: est.per_worker_prefetch,
            est_no_prefetch: est.per_worker_no_prefetch,
            host_budget,
            host_status,
            prefetch_requested,
            prefetch,
        }
        .print();
        for (j, a) in job_audits.iter().flatten().enumerate() {
            eprintln!("\n(job {:06}, ALL REFERENCE BINS) {}", j + 1, a.report());
        }
        if args.hit_stats {
            let mut merged = crate::gpu::HitStats::default();
            for h in &job_hit_stats {
                merged.merge(h);
            }
            eprintln!("\nHITS PER SEED\n{}", merged.report());
        }
    }
    let _ = (ref_bytes, worsts);
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

    /// `-B 0` needs LPT bins, so `--kegalign-bins` (sequential fill) is
    /// rejected before any input is read; the GPU `run` path repeats the
    /// check before planning.
    #[test]
    fn prepare_rejects_kegalign_with_auto_layout() {
        let mut phases = Phases::new();
        let mut auto = base_args();
        auto.reference = PathBuf::from("/nonexistent/ref.fa");
        auto.query = Some(PathBuf::from("/nonexistent/qry.fa"));
        auto.seq_block_size = 0;
        auto.kegalign_bins = true;
        let err = prepare(&auto, &mut phases).err().unwrap();
        assert!(err.to_string().contains("automatic layout"), "{err}");
    }

    #[test]
    fn layout_workers_clamps_to_devices_with_env_override() {
        assert_eq!(layout_workers(0, 0, None), 1);
        assert_eq!(layout_workers(1, 1, None), 1);
        assert_eq!(layout_workers(2, 1, None), 1);
        assert_eq!(layout_workers(2, 2, None), 2);
        assert_eq!(layout_workers(8, 2, None), 2);
        assert_eq!(layout_workers(1, 4, Some(2)), 2);
        assert_eq!(layout_workers(2, 2, Some(0)), 2, "zero override is ignored");
    }

    /// Batch validation is pure: W>1, `-B 0`, `--kegalign-bins` and
    /// `--from-manifest` are all rejected with clear messages.
    #[test]
    fn batch_rejects_unsupported_combinations() {
        let mut bad_gpus = base_args();
        bad_gpus.query = None;
        bad_gpus.query_list = Some(PathBuf::from("q.txt"));
        bad_gpus.gpus = 2;
        assert_eq!(
            validate_batch_args(&bad_gpus).unwrap_err(),
            "batch mode is W=1 in this release"
        );

        let mut bad_b = base_args();
        bad_b.query_list = Some(PathBuf::from("q.txt"));
        bad_b.seq_block_size = 0;
        assert!(validate_batch_args(&bad_b).unwrap_err().contains("-B 0"));

        let mut bad_k = base_args();
        bad_k.query_list = Some(PathBuf::from("q.txt"));
        bad_k.kegalign_bins = true;
        assert!(
            validate_batch_args(&bad_k)
                .unwrap_err()
                .contains("kegalign")
        );

        let mut bad_m = base_args();
        bad_m.query_list = Some(PathBuf::from("q.txt"));
        bad_m.from_manifest = Some(PathBuf::from("m.txt"));
        assert!(
            validate_batch_args(&bad_m)
                .unwrap_err()
                .contains("from-manifest")
        );

        let ok = base_args();
        assert!(validate_batch_args(&ok).is_ok());
    }

    /// `--query-list` parsing: blank lines and `#` comments are ignored;
    /// relative paths resolve against the list file's directory.
    #[test]
    fn query_list_ignores_blanks_comments_and_resolves_relative() {
        let dir = std::env::temp_dir().join(format!("hspz-qlist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let list = dir.join("list.txt");
        std::fs::write(&list, "# comment\n\na.fa\n  \n# another\nsub/b.fa\n").unwrap();
        let got = read_query_list(&list).unwrap();
        assert_eq!(got, vec![dir.join("a.fa"), dir.join("sub/b.fa")]);
        std::fs::remove_dir_all(&dir).unwrap();

        let dir2 = std::env::temp_dir().join(format!("hspz-qlist-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir2).unwrap();
        let empty = dir2.join("empty.txt");
        std::fs::write(&empty, "# only comments\n\n").unwrap();
        assert!(read_query_list(&empty).is_err());
        std::fs::remove_dir_all(&dir2).unwrap();
    }

    /// Per-job dump siblings append `.{job:06}` to the full file name, so all
    /// four batch dump paths agree. The old `Path::with_extension` form dropped
    /// `.manifest`/`.raw` and wrote `out.000001` instead of `out.manifest.000001`.
    #[test]
    fn batch_dump_siblings_append_to_the_full_file_name() {
        assert_eq!(
            job_sibling(PathBuf::from("out.manifest").as_path(), 0),
            PathBuf::from("out.manifest.000001")
        );
        assert_eq!(
            job_sibling(PathBuf::from("/tmp/out.plan").as_path(), 41),
            PathBuf::from("/tmp/out.plan.000042")
        );
        assert_eq!(
            job_sibling(PathBuf::from("dump.raw").as_path(), 2),
            PathBuf::from("dump.raw.000003")
        );
        // The HSPZ_ANCHOR_CENSUS dump uses the same helper, not `format!("{}.{}")`.
        assert_eq!(
            job_sibling(PathBuf::from("out/queries.tsv").as_path(), 9),
            PathBuf::from("out/queries.tsv.000010")
        );
    }

    /// Reference-bin compatibility: identical bins pass; any difference in
    /// count, membership or size aborts naming the job.
    #[test]
    fn reference_bin_compatibility_names_the_job() {
        use crate::plan::Bin;
        let bins = vec![
            Bin {
                id: 0,
                record_ids: vec![0, 1],
                total_bp: 100,
            },
            Bin {
                id: 1,
                record_ids: vec![2],
                total_bp: 50,
            },
        ];
        assert!(
            check_reference_bins_compatible(
                &bins,
                &bins.clone(),
                1,
                PathBuf::from("b.fa").as_path()
            )
            .is_ok()
        );
        let fewer = vec![Bin {
            id: 0,
            record_ids: vec![0, 1],
            total_bp: 100,
        }];
        let err =
            check_reference_bins_compatible(&bins, &fewer, 1, PathBuf::from("b.fa").as_path())
                .unwrap_err();
        assert!(err.contains("job 000002") && err.contains("b.fa"), "{err}");
        let mut moved = bins.clone();
        moved[0].record_ids = vec![0, 2];
        let err =
            check_reference_bins_compatible(&bins, &moved, 2, PathBuf::from("c.fa").as_path())
                .unwrap_err();
        assert!(err.contains("job 000003"), "{err}");
        let mut resized = bins.clone();
        resized[1].total_bp = 51;
        assert!(
            check_reference_bins_compatible(&bins, &resized, 1, PathBuf::from("b.fa").as_path())
                .is_err()
        );
    }

    /// The batch router keeps one next-ordinal cursor per job: a bin-major
    /// interleave (R0J0, R0J1, R1J0, R1J1) advances each job in its standalone
    /// order, while duplicates, skips, swapped bins and unknown jobs fail.
    #[test]
    fn job_router_routes_by_job_and_rejects_bad_identity() {
        use crate::plan::{Bin, Plan, WorkUnit};
        // Two jobs sharing R=2 bins: job 0 has Q=1 (2 units), job 1 has Q=2
        // (4 units). Ordinals are ref-outer per job.
        let bins = || {
            vec![
                Bin {
                    id: 0,
                    record_ids: vec![0],
                    total_bp: 10,
                },
                Bin {
                    id: 1,
                    record_ids: vec![1],
                    total_bp: 10,
                },
            ]
        };
        let plan0 = Plan {
            reference_bins: bins(),
            query_bins: vec![Bin {
                id: 0,
                record_ids: vec![0],
                total_bp: 5,
            }],
            units: vec![
                WorkUnit {
                    ordinal: 0,
                    reference_bin: 0,
                    query_bin: 0,
                },
                WorkUnit {
                    ordinal: 1,
                    reference_bin: 1,
                    query_bin: 0,
                },
            ],
        };
        let plan1 = Plan {
            reference_bins: bins(),
            query_bins: vec![
                Bin {
                    id: 0,
                    record_ids: vec![0],
                    total_bp: 5,
                },
                Bin {
                    id: 1,
                    record_ids: vec![1],
                    total_bp: 5,
                },
            ],
            units: vec![
                WorkUnit {
                    ordinal: 0,
                    reference_bin: 0,
                    query_bin: 0,
                },
                WorkUnit {
                    ordinal: 1,
                    reference_bin: 0,
                    query_bin: 1,
                },
                WorkUnit {
                    ordinal: 2,
                    reference_bin: 1,
                    query_bin: 0,
                },
                WorkUnit {
                    ordinal: 3,
                    reference_bin: 1,
                    query_bin: 1,
                },
            ],
        };
        let mut r = JobRouter::new(vec![2, 4]);
        // Bin-major traversal the executor uses.
        r.accept(0, 0, 0, 0, &plan0).unwrap();
        r.accept(1, 0, 0, 0, &plan1).unwrap();
        r.accept(1, 1, 0, 1, &plan1).unwrap();
        r.accept(0, 1, 1, 0, &plan0).unwrap();
        r.accept(1, 2, 1, 0, &plan1).unwrap();
        r.accept(1, 3, 1, 1, &plan1).unwrap();
        assert!(r.complete());

        // Duplicate: job 0 ordinal 1 was already consumed.
        let mut r = JobRouter::new(vec![2, 4]);
        r.accept(0, 0, 0, 0, &plan0).unwrap();
        r.accept(0, 1, 1, 0, &plan0).unwrap();
        let err = r.accept(0, 1, 1, 0, &plan0).unwrap_err();
        assert!(err.contains("duplicate"), "{err}");

        // Missing: skipping ordinal 0 for job 1.
        let mut r = JobRouter::new(vec![2, 4]);
        let err = r.accept(1, 1, 0, 1, &plan1).unwrap_err();
        assert!(err.contains("missing"), "{err}");

        // Swapped bin tuple at the expected ordinal.
        let mut r = JobRouter::new(vec![2, 4]);
        let err = r.accept(0, 0, 1, 0, &plan0).unwrap_err();
        assert!(err.contains("R1 Q0") && err.contains("R0 Q0"), "{err}");

        // Unknown ordinal past the plan end, and unknown job.
        let mut r = JobRouter::new(vec![2, 4]);
        r.accept(0, 0, 0, 0, &plan0).unwrap();
        r.accept(0, 1, 1, 0, &plan0).unwrap();
        let err = r.accept(0, 2, 0, 0, &plan0).unwrap_err();
        assert!(err.contains("outside"), "{err}");
        assert!(
            r.accept(7, 0, 0, 0, &plan0)
                .unwrap_err()
                .contains("unknown job")
        );
        assert!(!r.complete());
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
        from_manifest.query = Some(PathBuf::from("/nonexistent/qry.fa"));
        from_manifest.from_manifest = Some(PathBuf::from("/nonexistent/plan.manifest"));
        let err = prepare(&from_manifest, &mut phases).err().unwrap();
        assert!(err.to_string().contains("--from-manifest"), "{err}");

        let mut dump_manifest = base_args();
        dump_manifest.reference = PathBuf::from("/nonexistent/ref.fa");
        dump_manifest.query = Some(PathBuf::from("/nonexistent/qry.fa"));
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

/// Round 90b: the `HSPZ_UNIT_PARTITION` auto arm and its forced overrides are
/// pure, so every branch (W=1, identical, one attribute differing, forced
/// either way, invalid) is covered without CUDA or env access.
#[cfg(test)]
mod partition_policy_tests {
    use super::partition_policy;
    use crate::gpu::DeviceProfile;

    fn p(sms: i32, mhz: i32, mib: u64) -> DeviceProfile {
        DeviceProfile {
            sms,
            clock_khz: mhz * 1000,
            l2_bytes: mib << 20,
        }
    }

    #[test]
    fn w1_stays_whole_bin() {
        assert_eq!(
            partition_policy(None, 1, &[p(128, 2520, 72)]),
            Ok((false, "auto: W=1".to_string()))
        );
    }

    #[test]
    fn identical_devices_partition() {
        assert_eq!(
            partition_policy(None, 2, &[p(128, 2520, 72), p(128, 2520, 72)]),
            Ok((
                true,
                "auto: 2 matching devices (128 SMs, 2520 MHz, 72 MiB L2; static attributes only, clocks within 10%)".to_string()
            ))
        );
        // Vendor OC variants of one model report different nominal clocks; within 10% is the same class.
        assert!(partition_policy(None, 2, &[p(128, 2520, 72), p(128, 2610, 72)]).unwrap().0);
    }

    #[test]
    fn time_sliced_workers_stay_whole_bin() {
        let (on, why) = partition_policy(None, 2, &[p(128, 2520, 72)]).unwrap();
        assert!(!on);
        assert_eq!(why, "auto: 2 workers time-slice 1 device");
    }

    #[test]
    fn any_differing_attribute_forces_whole_bin() {
        let base = p(40, 1590, 4);
        for other in [p(48, 1590, 4), p(40, 1200, 4), p(40, 1590, 8)] {
            let (on, why) = partition_policy(None, 2, &[base, other]).unwrap();
            assert!(!on);
            assert!(
                why.starts_with("auto: device 1 differs from device 0 ("),
                "{why}"
            );
        }
        // The tuple is (differing device) vs (device 0), differing first.
        let (_, why) = partition_policy(None, 2, &[base, p(48, 1500, 8)]).unwrap();
        assert_eq!(
            why,
            "auto: device 1 differs from device 0 \
             (48 vs 40 SMs, 1500 vs 1590 MHz, 8 vs 4 MiB L2)"
        );
    }

    #[test]
    fn forced_values_override_the_profiles() {
        let same = [p(128, 2520, 72), p(128, 2520, 72)];
        let diff = [p(40, 1590, 4), p(48, 1500, 8)];
        assert_eq!(
            partition_policy(Some("1"), 2, &diff),
            Ok((true, "forced by HSPZ_UNIT_PARTITION=1".to_string()))
        );
        assert_eq!(
            partition_policy(Some("0"), 2, &same),
            Ok((false, "forced by HSPZ_UNIT_PARTITION=0".to_string()))
        );
    }

    #[test]
    fn invalid_value_is_an_error() {
        assert_eq!(
            partition_policy(Some("2"), 2, &[]).unwrap_err(),
            "HSPZ_UNIT_PARTITION must be unset, 0 or 1, got \"2\""
        );
    }
}
