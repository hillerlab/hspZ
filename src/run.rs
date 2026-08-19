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
//! never depends on completion order. `device_seeds_for` selects the device
//! seeder above one worker, `--threads` is the machine-wide budget divided
//! across workers, and the host-memory preflight gates the run before any CUDA
//! allocation.

use crate::Fallible;
use crate::cli::RunArgs;
use crate::gpu::{Engine, EngineConfig, HitStats, Lifecycle, device_memory, resolve_max_hits};
use crate::hsp::{self, SegmentPair};
use crate::partition::{Partitioner, Plan};
use crate::plan::{self, PackedBin, RecordMeta};
use crate::scoring;
use crate::seed::{self, SeedTable, Shape};
use crate::sequence::{self, Chr, Genome, encode};
use crate::sink::{DirectorySink, OutputSink, TarGzSink};
use crate::timing::{self, Phases};
use cuda_core::CudaContext;
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
    let plus = args.strand == "plus" || args.strand == "both";
    let minus = args.strand == "minus" || args.strand == "both";
    if !plus && !minus {
        return Err(format!("--strand must be plus, minus or both, got {}", args.strand).into());
    }

    let shape = Shape::parse(&args.seed)?;
    let sub_mat = scoring::build_sub_mat(&args.ambiguous, args.xdrop, args.scoring.as_deref())?;

    let t = Instant::now();
    // Reference and query input are timed separately and kept out of `core`,
    // so a format change can never be confused with a core change.
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
    // The reference index is the largest CPU stage (10.1% of a chr1 run), and
    // it uses the one existing --threads budget rather than a knob of its own.
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

    /// This input as the single query bin of a 1x1 plan.
    pub(crate) fn query_pass(&self) -> QueryPass<'_> {
        QueryPass {
            fwd: self.enc_query_source(),
            rc: &self.query_rc,
            intervals: &self.intervals,
            q_block_len: self.q_block_len,
        }
    }

    /// The encoded query strands `Engine::swap_query` consumes.
    ///
    /// A single-block run is a 1x1 plan, so this is that plan's only query bin.
    pub(crate) fn encoded_query(&self) -> (&[u8], &[u8]) {
        (&self.enc_query, &self.enc_query_rc)
    }

    /// The device-facing configuration: tables, sequences, and the tuning
    /// knobs that land in kernel constants.
    pub(crate) fn engine_config(&self, args: &RunArgs) -> EngineConfig<'_> {
        EngineConfig {
            index_table: &self.table.index_table,
            pos_table: &self.table.pos_table,
            ref_seq: &self.enc_ref,
            sub_mat: &self.sub_mat,
            seed_size: self.shape.size as u32,
            xdrop: args.xdrop,
            hspthresh: args.hspthresh,
            noentropy: args.noentropy,
            max_hits: args.max_hits,
            timing: args.time,
            hsp_blocks: args.hsp_blocks,
        }
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
}

/// One `seed_and_filter` call: a wga_chunk of one strand of one interval.
///
/// The interval/strand/chunk nest is flattened into a flat list so
/// the seed worker can always run exactly one batch ahead, including across
/// strand and interval boundaries. The order is identical to the original
/// nesting — plus strand then minus strand, chunks ascending — because that
/// order fixes the `MAX_HITS` chunking and therefore the final HSP set.
struct Batch {
    interval: usize,
    rev: bool,
    range: (u32, u32),
}

/// Host staging for one batch's seeds.
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
/// Everything one (reference, query-bin) pass needs from the query side.
///
/// Seeding reads **raw** bytes, not the device alphabet: `fwd` is the raw forward
/// block and `rc` the raw reverse complement. `Engine::swap_query` handles the
/// *encoded* device buffers separately, so a pass needs both halves supplied and
/// the two must describe the same block.
///
/// `intervals` and `q_block_len` are per query bin. A multi-bin executor derives
/// them from that bin's own `block_len`; reusing whole-genome intervals here would
/// silently seed the wrong ranges.
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

    // Chosen at runtime, not at compile time. The device seeder wins on
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
                let o = engine.seed_and_filter(0, b.rev)?;
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
        // Identical call to the one the serial loop made — the seed-generation
        // algorithm is untouched here, so the seed sequence stays bit-identical.
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
        // How many batches the host runs ahead of the one computing.
        //
        // One is enough to hide seed *generation* behind GPU work (N1). Overlapping
        // the *upload* needs two: batch N+1's seeds must already be in host memory
        // when batch N's kernels are enqueued, or there is no compute left for the DMA
        // to hide behind — which is exactly why a same-stream async copy
        // bought nothing. Slots: one being consumed, one being uploaded, one
        // being generated.
        //
        // Two things turn the overlap off, both because it cannot work without them:
        // pageable staging (an async copy from unpinned memory blocks until staged, so
        // ZLUDA never overlaps), and reallocated seed buffers (an upload in flight into
        // a freed buffer is a use-after-free).
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
        // is exactly `max(0, seed_end[N+1] - gpu_end[N])`.
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
                    let o = engine.seed_and_filter(if overlap { i % 2 } else { 0 }, b.rev)?;
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
        // paying cuMemHostAlloc again.
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

/// Formats and emits every logical output file.
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
/// same archive (`-Z`) and shares one `-D` history.
pub(crate) struct Emitter {
    sink: Box<dyn OutputSink>,
    part: Partitioner,
    diagonal: bool,
    partition_ms: f64,
    format_ms: f64,
    archive_ms: f64,
    files: usize,
    bytes_in: u64,
}

impl Emitter {
    pub(crate) fn new(args: &RunArgs) -> Fallible<Self> {
        let sink: Box<dyn OutputSink> = match tarball_path(args) {
            Some(path) => Box::new(TarGzSink::new(&path)?),
            None => Box::new(DirectorySink::new(&args.output)?),
        };
        Ok(Emitter {
            sink,
            part: Partitioner::default(),
            diagonal: args.diagonal_partition,
            partition_ms: 0.0,
            format_ms: 0.0,
            archive_ms: 0.0,
            files: 0,
            bytes_in: 0,
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
        for (n, (fw, rc)) in pass.intervals.iter().enumerate() {
            // `segment_printer.cpp` names files by 1-based interval index,
            // query block index and reference block index.
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
/// reuses the executor's `Emitter` — one output path, not two.
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
// Multi-GPU execution

/// One finished work unit on its way to the emitter.
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
    /// GPU-timeline idle between stages, summed for this worker. Each worker
    /// drives one GPU, so this is that GPU's idle and therefore the per-GPU
    /// ceiling on any batch-level pipelining.
    gap_ms: f32,
    gap_n: u64,
    prefetched_ms: Duration,
    hit_stats: HitStats,
    peak_used: usize,
}

/// Runs one worker's reference bins on `device`, streaming finished units to the
/// emitter (build/upload each reference once, reuse it across its queries; no
/// GPU is shared for performance).
///
/// This is the serial executor, parameterised by which bins it owns: with one
/// worker it is exactly the old path, which is what makes `serial == multi-GPU`
/// a property of the assignment rather than of two code paths.
#[allow(clippy::too_many_arguments)]
/// Whether to generate the query seed stream on the device.
///
/// The trade reverses with GPU count. On one GPU the device seeder adds ~165 s of
/// device work against a host tail that pinned async H->D already hides; on two
/// T4s that tail is 397 s and exposed, and moving it is worth -20.3% of wall.
/// So the worker count is the whole decision. `HSPZ_DEVICE_SEEDS=0|1` overrides it,
/// which is how the device path gets exercised on a one-GPU box.
pub fn device_seeds_for(workers: usize) -> bool {
    match std::env::var("HSPZ_DEVICE_SEEDS").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => workers > 1,
    }
}

// one worker's whole execution context, threaded rather than shared
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
    // The device seeder measured -20.3% of 2-GPU wall and a loss on one GPU, so
    // the executor decides per run rather than the build deciding once.
    device_seeds: bool,
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
            // Pinned by the caller from device 0, so two devices with different
            // VRAM cannot chunk differently and make output depend on which GPU
            // ran a bin.
            max_hits: args.max_hits,
            timing: args.time,
            hsp_blocks: args.hsp_blocks,
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
                let packed_q = PackedBin::build(
                    qbin.record_ids.iter().map(|&id| {
                        let (n, s) = &qry_records[id as usize];
                        (n.as_str(), s.as_slice())
                    }),
                    &args.query_prefix,
                    true,
                );
                // Per-bin intervals + q_block_len. `intervals` is empty for a
                // block <= seed, and `q_block_len` is then never read; saturating
                // avoids the underflow the single-block path guards with an error.
                let intervals =
                    sequence::intervals(packed_q.block_len, shape.size, args.lastz_interval_size);
                let q_block_len = packed_q.block_len.saturating_sub(shape.size) as u32;

                engine.swap_query(&packed_q.enc, &packed_q.enc_rc)?;
                let qpass = QueryPass {
                    fwd: &packed_q.buf[..packed_q.block_len],
                    rc: &packed_q.rc,
                    intervals: &intervals,
                    q_block_len,
                };
                let pass =
                    seed_and_filter_all(&mut engine, &qpass, shape, transitions, args, threads)?;

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
            eprintln!("\n{}", c.report());
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

    // Load both sides as raw records — no block-size guard.
    let t = Instant::now();
    let (_, qry_records, _) = sequence::read_records(&args.query)?;
    phases.add("input.query", t.elapsed());
    let t = Instant::now();
    let (_, ref_records, _) = sequence::read_records(&args.reference)?;
    phases.add("input.reference", t.elapsed());

    let ref_meta = record_meta(&ref_records);
    let qry_meta = record_meta(&qry_records);

    let t = Instant::now();
    let ctx = CudaContext::new(0)?;
    phases.add("CUDA context init", t.elapsed());

    let t = Instant::now();
    let max_hits = resolve_max_hits(&ctx, args.max_hits);
    let (free, _) = device_memory();
    let (plan, _worst) = plan::plan_within_budget(
        &ref_meta,
        &qry_meta,
        args.seq_block_size as u64,
        // The query side takes its own target, defaulting to the reference one so
        // an unset flag reproduces every existing plan bit for bit.
        args.query_block_size.unwrap_or(args.seq_block_size) as u64,
        free as u64,
        shape.kmer_size,
        args.step,
        max_hits,
        args.kegalign_bins,
    )?;
    phases.add("plan", t.elapsed());

    if let Some(path) = &args.dump_plan {
        use std::io::Write;
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
    }

    // Reference bins to workers, deterministic LPT. Every query bin runs
    // against every reference bin, so `cost(R) = reference_bp x total_query_bp` is
    // monotone in the bin's own bp — LPT on `total_bp` is the same schedule with
    // less arithmetic (plan::assign_bins).
    let devices = crate::gpu::device_count().max(1);
    let workers = args.gpus.max(1).min(plan.reference_bins.len().max(1));
    let assignment = plan::assign_bins(&plan.reference_bins, workers);
    if workers > devices {
        eprintln!(
            "note: {workers} workers over {devices} device(s) — they time-slice one GPU. \
             That is a correctness configuration, not a performance one."
        );
    }

    // Host-memory preflight. The GPU preflight only sized the device side;
    // here we size host RAM for `workers` each building their own (prefetched)
    // reference state, falling back to no-prefetch before failing. Runs after
    // plan_within_budget so it sees the accepted (possibly shrunk) bin set.
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
    // Never budget to 100% — reserve 10% for runtime/allocator/output overhead
    // the model does not see.
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
            eprintln!("note: host preflight disabled reference prefetch for {workers} worker(s)");
        }
        prefetch &= fits;
    }
    // The estimate that the decision was made on, so a run can be checked
    // against its own measured RSS without re-deriving the model.
    let host_peak_est = plan::host_peak(&est, &assignment, prefetch);

    let mut emitter = Emitter::new(args)?;
    let mut raw_all: Vec<(char, Vec<SegmentPair>)> = Vec::new();
    // The emitter consumes units strictly in `WorkUnit.ordinal` order, so
    // `-D` history, file names and tar entry order never depend on which GPU
    // finished first. Workers push completed units into a bounded channel; the
    // main thread replays them in order, buffering whatever arrives early.
    // The thread budget is the machine's, not each worker's: `--threads` (or the
    // available parallelism) is divided once here. Resolving it inside every worker
    // gave a 2-GPU run 2x the host threads it asked for, which is exactly the
    // CPU starvation a small box would then be blamed for.
    let per_worker = (resolve_threads(args.threads) / workers).max(1);
    // The device seeder measured -20.3% of 2-GPU wall, and the 1-GPU case is
    // already hidden behind pinned async copies, so the worker count is the
    // whole decision.
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
                 hold {:?} — a worker died without sending",
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

    // Per-worker load, because the aggregate cannot show imbalance. Units are
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
        use std::io::Write;
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
        // `stage` waits are the ones stream ordering makes unnecessary and are
        // 0 with --async-stages.
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
            "  max_hits: {} (device-derived unless --max-hits)\n  lifecycle: {} ref bins, \
{} work units, {} builds, {} engines, {} ref uploads, {} query swaps",
            max_hits,
            plan.reference_bins.len(),
            plan.units.len(),
            lifecycle.seed_table_builds,
            lifecycle.engine_creations,
            lifecycle.reference_uploads,
            lifecycle.query_swaps,
        );
        // A stalled upload is one that had not finished when its compute needed
        // it, i.e. overlap that did not happen.
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
        // The host-budget decision, in the same units as the line above so the
        // validation is a subtraction.
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
    stats
}
