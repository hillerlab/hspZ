// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Seed shapes, k-mer indexing and the reference seed-position table.
//!
//! Port of `ntcoding.cpp` + `seed_pos_table.cu`. The device-side table is a
//! pair of arrays:
//!   * `index_table[k]` — number of reference k-mers with index `<= k`, so
//!     bucket `k` spans `index_table[k-1] .. index_table[k]`;
//!   * `pos_table` — the reference positions, grouped by bucket.
//!
//! `SeedTable::build_parallel` is the two-pass, lock-free parallel builder
//! (rounds 19–20): workers count into private tables, a two-level scan turns
//! them into disjoint write cursors, and pass 2 rescans and scatters, so
//! within-bucket order — which fixes `MAX_HITS` chunking — is preserved
//! bit-for-bit. Query-side seeds are generated per work unit; the device-side
//! seed kernels live in `gpu/kernels.rs` (`seed_kmers`/`scatter_seeds`,
//! round 68).

use crate::sequence::{N_NT, nt_char_to_int};

/// `ntcoding.h: INVALID_KMER`. Doubles as the "no seed here" sentinel that
/// `seeder.cpp` tests against.
pub const INVALID_KMER: u32 = 1 << 31;

/// `parameters.h: TRANSITION_MASK`.
const TRANSITION_MASK: u64 = 2;

/// A spaced seed pattern (`GenerateShapePos`).
#[derive(Debug, Clone)]
pub struct Shape {
    /// Width of the pattern in bases — `cfg.seed.size`.
    pub size: usize,
    /// Number of care positions — `cfg.seed.kmer_size`.
    pub kmer_size: usize,
    /// Indices of the care positions, in pattern order.
    pub pos: Vec<usize>,
    /// Whether each care position admits a transition.
    pub transition: Vec<bool>,
}

impl Shape {
    /// `main.cpp`: the two named patterns, or an arbitrary 1/0/T string. Note
    /// that KegAlign rewrites every care position of a custom pattern to `T`,
    /// so custom shapes always allow transitions at every position.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let pattern = match spec {
            "12of19" => "TTT0T00TT00T0T0TTTT".to_string(),
            "14of22" => "TTT0T0TT00TT00T0T0TTTT".to_string(),
            other => other
                .bytes()
                .map(|c| if c == b'1' || c == b'T' { 'T' } else { '0' })
                .collect(),
        };

        let mut pos = Vec::new();
        let mut transition = Vec::new();
        for (i, c) in pattern.bytes().enumerate() {
            if c == b'1' || c == b'T' {
                pos.push(i);
                transition.push(c == b'T');
            }
        }
        if !(4..=15).contains(&pos.len()) {
            return Err(format!(
                "seed {spec} has {} care positions; GenerateSeedPosTable requires 4..=15",
                pos.len()
            ));
        }
        Ok(Shape {
            size: pattern.len(),
            kmer_size: pos.len(),
            pos,
            transition,
        })
    }

    /// `GetKmerIndexAtPos`. Reads `size` bases and returns `INVALID_KMER` if any
    /// of them is not uppercase ACGT.
    ///
    /// KegAlign decodes into a `uint32_t nt[64]` scratch array, which silently
    /// stops validating past 64 bases; validating the window directly has no
    /// measurable cost and no such ceiling.
    #[inline]
    pub fn kmer_at(&self, seq: &[u8], pos: usize) -> u32 {
        let window = &seq[pos..pos + self.size];
        // KegAlign rejects the whole window, not just the care positions.
        for &c in window {
            if nt_char_to_int(c) == N_NT {
                return INVALID_KMER;
            }
        }
        let mut kmer = 0u32;
        for &p in &self.pos {
            kmer = (kmer << 2) + nt_char_to_int(window[p]) as u32;
        }
        kmer
    }

    /// The seed offsets `seeder.cpp` pushes for one query position: the k-mer
    /// itself plus one single-transition variant per transition position.
    /// Returns the number of offsets appended.
    #[inline]
    pub fn push_seeds(&self, kmer: u32, j: u32, transitions: bool, out: &mut Vec<u64>) -> usize {
        let base = (kmer as u64) << 32;
        out.push(base | j as u64);
        if !transitions {
            return 1;
        }
        let mut n = 1;
        for t in 0..self.kmer_size {
            if self.transition[t] {
                let idx = (kmer as u64) ^ (TRANSITION_MASK << (2 * t));
                out.push((idx << 32) | j as u64);
                n += 1;
            }
        }
        n
    }
}

/// The reference seed-position table (`GenerateSeedPosTable`).
pub struct SeedTable {
    /// Device-side `index_table`: `4^kmer_size` entries, already shifted by one
    /// (`SendSeedPosTable(index_table + 1, ...)`).
    pub index_table: Vec<u32>,
    pub pos_table: Vec<u32>,
}

impl SeedTable {
    /// `seq` is the reference block; `step` is `--step`.
    pub fn build(seq: &[u8], shape: &Shape, step: u32) -> Self {
        let ref_len = seq.len();
        let step = step.max(1) as usize;
        let offset = (shape.size + 1) % step;
        let start_offset = step - offset;

        let table_size = (1usize << (2 * shape.kmer_size)) + 1;
        let mut counts = vec![0u32; table_size];

        // `num_steps = (ref_length - shape_size + offset) / step`, verbatim.
        let num_steps = (ref_len + offset).saturating_sub(shape.size) / step;

        let mut idxs = vec![INVALID_KMER; num_steps];
        for (i, slot) in idxs.iter_mut().enumerate() {
            let idx = shape.kmer_at(seq, start_offset + i * step);
            *slot = idx;
            if idx != INVALID_KMER {
                counts[idx as usize + 1] += 1;
            }
        }

        for i in 1..table_size {
            counts[i] += counts[i - 1];
        }

        // buckets are filled in ascending position order; KegAlign
        // fills them from a racy tbb::parallel_for so its within-bucket order
        // varies run to run. Order does not reach the output — every hit is
        // expanded independently and the HSP set is sorted before dedup — so a
        // parallel fill is only worth it if seed-table build shows up in the
        // per-stage timings.
        let mut cursor: Vec<u32> = counts.clone();
        let mut pos_table = vec![0u32; counts[table_size - 1] as usize];
        for (i, &idx) in idxs.iter().enumerate() {
            if idx != INVALID_KMER {
                let slot = &mut cursor[idx as usize];
                pos_table[*slot as usize] = (start_offset + i * step) as u32;
                *slot += 1;
            }
        }

        SeedTable {
            index_table: counts[1..].to_vec(),
            pos_table,
        }
    }

    /// [`build`](Self::build) by partitions that fit a cache (round 112),
    /// byte-identical to it.
    ///
    /// The two-pass builder below makes three random DRAM accesses per position:
    /// a private 4^k count table, the cursor table and `pos_table` itself. With
    /// eight builds at once (eight workers starting on an A100 node) that is what
    /// they all wait for: 7 to 9 s per reference bin.
    ///
    /// Here every position is first written as a `(low k-mer bits, position)`
    /// pair into the stream of its partition, the k-mer's high bits: at most 256
    /// sequential streams per thread. Each partition is then sorted on its own:
    /// a count table of the low bits (256 KiB for 12 care positions) and the few
    /// MiB of `pos_table` that the partition owns both stay in cache.
    ///
    /// The pairs are 8 bytes per position, so they are not all held at once:
    /// the partitions are taken in rounds of consecutive partitions whose pairs
    /// fit `max_pairs` (a partition larger than that is a round of its own), and
    /// every round scans the sequence again for its own partitions. The scan is
    /// the cheap part.
    ///
    /// The order inside a k-mer's bucket is ascending position, as `build`
    /// leaves it: thread ranges ascend, a partition's stream is laid out thread
    /// by thread, each thread appends in position order, and the counting sort
    /// is stable.
    fn build_partitioned(
        seq: &[u8],
        shape: &Shape,
        step: usize,
        threads: usize,
        start_offset: usize,
        num_steps: usize,
        max_pairs: usize,
    ) -> Self {
        let kbits = 2 * shape.kmer_size as u32;
        // At most 256 partitions; at least 16 low bits when the k-mer has them.
        let low_bits = kbits.min(16).max(kbits.saturating_sub(8));
        let parts = 1usize << (kbits - low_bits);
        let low_mask = (1u32 << low_bits) - 1;
        let nk = 1usize << kbits;

        let per = num_steps.div_ceil(threads);
        let ranges: Vec<(usize, usize)> = (0..threads)
            .map(|t| (t * per, ((t + 1) * per).min(num_steps)))
            .collect();

        // Pass A1: how many positions each thread sends to each partition.
        let counts: Vec<Vec<u64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .map(|&(lo, hi)| {
                    scope.spawn(move || {
                        let mut c = vec![0u64; parts];
                        for_each_kmer(seq, shape, start_offset, step, lo, hi, |kmer, _| {
                            c[(kmer >> low_bits) as usize] += 1;
                        });
                        c
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("seed count worker panicked"))
                .collect()
        });

        // Where each partition's positions go in `pos_table`.
        let mut part_start = vec![0u64; parts + 1];
        for p in 0..parts {
            part_start[p + 1] = part_start[p] + counts.iter().map(|c| c[p]).sum::<u64>();
        }
        let total_u64 = part_start[parts];
        assert!(
            total_u64 <= u64::from(u32::MAX),
            "more than u32::MAX indexed positions in one reference bin"
        );
        let total = total_u64 as usize;

        // Rounds of consecutive partitions whose pairs fit `max_pairs`.
        let mut rounds: Vec<(usize, usize)> = Vec::new();
        let mut first = 0;
        while first < parts {
            let mut end = first + 1;
            while end < parts && part_start[end + 1] - part_start[first] <= max_pairs as u64 {
                end += 1;
            }
            rounds.push((first, end));
            first = end;
        }
        let widest = rounds
            .iter()
            .map(|&(a, b)| (part_start[b] - part_start[a]) as usize)
            .max()
            .unwrap_or(0);

        let mut index_table = vec![0u32; nk];
        let mut pos_table = vec![0u32; total];
        // `low bits << 32 | position`, one round's worth.
        let mut pairs = vec![0u64; widest];
        let (idx, pos) = (
            index_table.as_mut_ptr() as usize,
            pos_table.as_mut_ptr() as usize,
        );
        for &(pa, pb) in &rounds {
            let round_base = part_start[pa];

            // Pass A2: this round's pairs. A partition's stream is one slice of
            // `pairs`, thread after thread.
            let mut cursors: Vec<Vec<u64>> = vec![vec![0u64; pb - pa]; threads];
            for p in pa..pb {
                let mut at = part_start[p] - round_base;
                for (t, cursor) in cursors.iter_mut().enumerate() {
                    cursor[p - pa] = at;
                    at += counts[t][p];
                }
            }
            {
                let base = pairs.as_mut_ptr() as usize;
                std::thread::scope(|scope| {
                    for (&(lo, hi), mut cur) in ranges.iter().zip(cursors) {
                        scope.spawn(move || {
                            for_each_kmer(seq, shape, start_offset, step, lo, hi, |kmer, at| {
                                let p = (kmer >> low_bits) as usize;
                                if p < pa || p >= pb {
                                    return;
                                }
                                let slot = &mut cur[p - pa];
                                // SAFETY: `*slot` walks this thread's own run of
                                // its partition's stream. The runs were laid out
                                // from the pass-A1 counts, so they are disjoint,
                                // they tile the round's pairs, and this pass meets
                                // the same positions.
                                unsafe {
                                    *(base as *mut u64).add(*slot as usize) =
                                        u64::from(kmer & low_mask) << 32 | u64::from(at);
                                }
                                *slot += 1;
                            });
                        });
                    }
                });
            }

            // Pass B: a stable counting sort of every partition of the round on
            // its low bits.
            let (pairs, part_start) = (&pairs, &part_start);
            let next = std::sync::atomic::AtomicUsize::new(pa);
            std::thread::scope(|scope| {
                for _ in 0..threads.min(pb - pa) {
                    let next = &next;
                    scope.spawn(move || {
                        let mut slot = vec![0u32; 1usize << low_bits];
                        loop {
                            let p = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if p >= pb {
                                return;
                            }
                            let stream = &pairs[(part_start[p] - round_base) as usize
                                ..(part_start[p + 1] - round_base) as usize];
                            slot.fill(0);
                            for &pair in stream {
                                slot[(pair >> 32) as usize] += 1;
                            }
                            // Counts become bucket starts; the index keeps every
                            // bucket's end, cumulative over the whole table.
                            let mut run = part_start[p] as u32;
                            for (low, s) in slot.iter_mut().enumerate() {
                                let n = *s;
                                *s = run;
                                run += n;
                                // SAFETY: k-mer `p << low_bits | low` belongs to
                                // partition `p` alone, and one thread owns `p`.
                                unsafe {
                                    *(idx as *mut u32).add((p << low_bits) | low) = run;
                                }
                            }
                            for &pair in stream {
                                let s = &mut slot[(pair >> 32) as usize];
                                // SAFETY: the buckets of partition `p` tile its
                                // own span of `pos_table`, which this thread
                                // owns; each slot is written once.
                                unsafe {
                                    *(pos as *mut u32).add(*s as usize) = pair as u32;
                                }
                                *s += 1;
                            }
                        }
                    });
                }
            });
        }

        SeedTable {
            index_table,
            pos_table,
        }
    }

    /// [`build`](Self::build) across `threads` workers, byte-identical to it.
    ///
    /// The serial builder is 11.4 s on hg38 chr1 — 10.1% of that
    /// run's wall time and its largest CPU stage — which is what earned this.
    ///
    /// Two passes over the same strided seed-start sequence, no atomics, no
    /// locks, no final sort:
    ///
    /// 1. each worker counts k-mers over a contiguous range of the *step index*
    ///    `i`, into a private count table;
    /// 2. the per-worker tables are prefix-summed across workers to give every
    ///    worker a disjoint write cursor per k-mer, and each worker rescans its
    ///    own range and scatters positions directly.
    ///
    /// Why the bucket order comes out identical (byte-identity is required,
    /// not just the same multiset — hit order feeds `MAX_HITS` chunk boundaries,
    /// so a reordering would move HSPs without changing any count): worker
    /// ranges are contiguous and ascending in `i`, and for every k-mer worker
    /// `t`'s cursor region precedes worker `t+1`'s. Positions therefore land in
    /// ascending genomic order within each bucket, exactly as the single-cursor
    /// serial walk produces them.
    ///
    /// Pass 2 recomputes `kmer_at` rather than storing the k-mer of every
    /// position, which also drops the serial builder's `idxs` temporary — 1 GB
    /// on chr1 at `--step 1`.
    #[allow(clippy::needless_range_loop)]
    pub fn build_parallel(seq: &[u8], shape: &Shape, step: u32, threads: usize) -> Self {
        let step = step.max(1) as usize;
        // Same strided start sequence as the serial builder; at --step 1 this
        // makes start_offset 1, so reference position 0 is never indexed.
        let offset = (shape.size + 1) % step;
        let start_offset = step - offset;
        let table_size = (1usize << (2 * shape.kmer_size)) + 1;
        let num_steps = (seq.len() + offset).saturating_sub(shape.size) / step;
        let threads = threads.max(1).min(num_steps.max(1));
        if threads == 1 || num_steps < 1 << 16 {
            return Self::build(seq, shape, step as u32);
        }
        if partitioned_build(shape) {
            let pairs = if pairs_at_once() {
                usize::MAX
            } else {
                (build_pairs_bytes(threads, shape.kmer_size) / 8) as usize
            };
            return Self::build_partitioned(
                seq,
                shape,
                step,
                threads,
                start_offset,
                num_steps,
                pairs,
            );
        }

        let per = num_steps.div_ceil(threads);
        let ranges: Vec<(usize, usize)> = (0..threads)
            .map(|t| (t * per, ((t + 1) * per).min(num_steps)))
            .collect();

        // Pass 1: private counts per worker, indexed by kmer (no +1 shift yet).
        let counts: Vec<Vec<u32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .map(|&(lo, hi)| {
                    scope.spawn(move || {
                        let mut c = vec![0u32; table_size - 1];
                        for i in lo..hi {
                            let idx = shape.kmer_at(seq, start_offset + i * step);
                            if idx != INVALID_KMER {
                                c[idx as usize] += 1;
                            }
                        }
                        c
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("seed count worker panicked"))
                .collect()
        });

        // Prefix across workers, two-level so it does not become the new
        // bottleneck. The obvious single loop is `for k { for t { .. } }` =
        // 4^kmer_size x threads serial additions striding across T separate
        // 67 MB tables; measured on hg38 chr20 that capped speedup at 1.84x and
        // made 16 threads slower than 8, because the serial term grows with T.
        //
        // Instead: sum each k-chunk in parallel, scan the T chunk totals
        // serially, then convert counts to absolute cursors in parallel.
        let nk = table_size - 1;
        let kper = nk.div_ceil(threads);
        let kranges: Vec<(usize, usize)> = (0..threads)
            .map(|c| (c * kper, ((c + 1) * kper).min(nk)))
            .collect();

        let chunk_totals: Vec<u64> = std::thread::scope(|scope| {
            let handles: Vec<_> = kranges
                .iter()
                .map(|&(lo, hi)| {
                    let counts = &counts;
                    scope.spawn(move || {
                        let mut sum = 0u64;
                        for c in counts.iter() {
                            for k in lo..hi {
                                sum += c[k] as u64;
                            }
                        }
                        sum
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("prefix worker panicked"))
                .collect()
        });

        let mut chunk_base = Vec::with_capacity(kranges.len());
        let mut acc = 0u64;
        for t in &chunk_totals {
            chunk_base.push(acc);
            acc += t;
        }
        let total = acc as u32;

        let mut index_table = vec![0u32; nk];
        let mut cursors = counts;
        {
            // SAFETY: each worker touches only k in its own `[lo, hi)`, and the
            // chunks tile `0..nk` without overlap, so no two workers write the
            // same `index_table[k]` or the same `cursors[t][k]`. The raw
            // pointers exist only because the disjointness is by k while the
            // owning containers are indexed by t first.
            let idx_ptr = index_table.as_mut_ptr() as usize;
            let cur_ptrs: Vec<usize> = cursors
                .iter_mut()
                .map(|c| c.as_mut_ptr() as usize)
                .collect();
            std::thread::scope(|scope| {
                for (ci, &(lo, hi)) in kranges.iter().enumerate() {
                    let cur_ptrs = &cur_ptrs;
                    let base = chunk_base[ci];
                    scope.spawn(move || {
                        let mut running = base as u32;
                        for k in lo..hi {
                            for &cp in cur_ptrs.iter() {
                                let slot = unsafe { &mut *(cp as *mut u32).add(k) };
                                let n = *slot;
                                *slot = running;
                                running += n;
                            }
                            unsafe { *(idx_ptr as *mut u32).add(k) = running };
                        }
                    });
                }
            });
        }

        // Pass 2: disjoint scatter. Each worker owns a distinct set of slots by
        // construction of the cursors above, so the writes cannot overlap.
        let mut pos_table = vec![0u32; total as usize];
        let base = pos_table.as_mut_ptr() as usize;
        std::thread::scope(|scope| {
            for (&(lo, hi), mut cur) in ranges.iter().zip(cursors) {
                scope.spawn(move || {
                    for i in lo..hi {
                        let pos = start_offset + i * step;
                        let idx = shape.kmer_at(seq, pos);
                        if idx == INVALID_KMER {
                            continue;
                        }
                        let slot = &mut cur[idx as usize];
                        // SAFETY: `*slot` is this worker's next slot in bucket
                        // `idx`. Cursors were built by prefix-summing the pass-1
                        // counts, so worker t's region for a bucket is exactly
                        // the count it recorded there and is disjoint from every
                        // other worker's; the debug_assert above checks the
                        // regions tile each bucket exactly. `pos_table` has
                        // `total` slots and every cursor stays below it.
                        unsafe {
                            *(base as *mut u32).add(*slot as usize) = pos as u32;
                        }
                        *slot += 1;
                    }
                });
            }
        });

        SeedTable {
            index_table,
            pos_table,
        }
    }

    /// Number of reference hits for a seed — what `find_num_hits` computes.
    #[inline]
    pub fn hit_count(&self, seed: u32) -> u32 {
        let end = self.index_table[seed as usize];
        if seed > 0 {
            end - self.index_table[seed as usize - 1]
        } else {
            end
        }
    }
}

/// Collects the seed offsets for one WGA chunk of a query block, matching the
/// `for (j = i; j < e; j++)` loop in `seeder.cpp`.
///
/// # Example
/// ```
/// let shape = Shape::parse("12of19").unwrap();
/// let seq = b"ACGTACGTACGTACGTACGTACGT"; // 24 bases
/// let seeds = chunk_seeds(seq, &shape, false, (0, 24));
/// assert_eq!(seeds.len(), 24 - 19 + 1); // one seed per valid 19-mer start
/// ```
pub fn chunk_seeds(seq: &[u8], shape: &Shape, transitions: bool, range: (u32, u32)) -> Vec<u64> {
    // Every position emits at most one seed per care position plus the exact
    // k-mer, and a WGA chunk is 250k positions — growing from empty means
    // reallocating and copying tens of megabytes per chunk.
    let per_pos = if transitions { 1 + shape.kmer_size } else { 1 };
    let mut out = Vec::with_capacity((range.1 - range.0) as usize * per_pos);
    for j in range.0..range.1 {
        let kmer = shape.kmer_at(seq, j as usize);
        if kmer != INVALID_KMER {
            shape.push_seeds(kmer, j, transitions, &mut out);
        }
    }
    out
}

/// Bytes of `(k-mer, position)` pairs the partitioned builder holds at once:
/// 512 MiB, or what the two-pass builder's count tables would take with this
/// many threads if that is more. So a build never holds more beside its tables
/// than it did before round 112, except on hosts with fewer than eight
/// threads, where it may hold 512 MiB instead of 64 MiB per thread.
pub fn build_pairs_bytes(threads: usize, kmer_size: usize) -> u64 {
    (threads as u64 * (1u64 << (2 * kmer_size)) * 4).max(512 << 20)
}

/// `HSPZ_SEED_TABLE`: `two-pass` keeps the builder of before round 112
/// (rollback, and the control arm of a measurement); `rounds` keeps the
/// partitioned builder's pairs within [`build_pairs_bytes`] whatever memory the
/// host has. Unset or `partitioned`: the partitioned builder, all pairs at once
/// where the host budget allows. Any other value is reported once and ignored.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SeedTableRequest {
    Partitioned,
    Rounds,
    TwoPass,
}

fn seed_table_request() -> SeedTableRequest {
    static REQUEST: std::sync::OnceLock<SeedTableRequest> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| match std::env::var("HSPZ_SEED_TABLE").as_deref() {
        Err(_) | Ok("partitioned") => SeedTableRequest::Partitioned,
        Ok("rounds") => SeedTableRequest::Rounds,
        Ok("two-pass") => SeedTableRequest::TwoPass,
        Ok(other) => {
            eprintln!(
                "warning: HSPZ_SEED_TABLE={other:?} is not partitioned, rounds or two-pass; ignored"
            );
            SeedTableRequest::Partitioned
        }
    })
}

fn two_pass_forced() -> bool {
    seed_table_request() == SeedTableRequest::TwoPass
}

/// Whether the partitioned builder holds all the pairs of a bin at once. Off
/// until the run's host budget says they fit ([`set_pairs_at_once`]).
static PAIRS_AT_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The run's decision on [`build_pairs_at_once_bytes`], taken once, before any
/// worker builds a table. Rounds only cost time (one more pass over the bases
/// each: 1.0 against 1.4 s per build of a human reference bin with eight builds
/// at once), so a host that cannot hold the pairs loses nothing else.
///
/// Returns what the builds will do: `HSPZ_SEED_TABLE=rounds` or `two-pass`
/// overrides a budget that allows it.
pub fn set_pairs_at_once(on: bool) -> bool {
    PAIRS_AT_ONCE.store(on, std::sync::atomic::Ordering::Relaxed);
    pairs_at_once()
}

fn pairs_at_once() -> bool {
    PAIRS_AT_ONCE.load(std::sync::atomic::Ordering::Relaxed)
        && seed_table_request() == SeedTableRequest::Partitioned
}

/// Whether [`SeedTable::build_parallel`] builds by cache-sized partitions: the
/// sliding window it reads k-mers with holds shapes of up to 32 bases.
pub fn partitioned_build(shape: &Shape) -> bool {
    shape.size <= 32 && !two_pass_forced()
}

/// Bytes a reference build may hold beside its two tables while it runs, for a
/// bin of at most `positions` indexed positions. The shape's length, which
/// picks the builder too, is not known to the host budget, so this is the
/// larger of the two builders' needs. A single partition with more pairs than
/// [`build_pairs_bytes`] is held whole and is not counted here: with 256
/// partitions that takes one leading four-base word in front of more than 67
/// million positions of a bin.
pub fn build_transient_bytes(threads: usize, kmer_size: usize, positions: u64) -> u64 {
    let two_pass = threads as u64 * (1u64 << (2 * kmer_size)) * 4;
    // One thread builds serially, with one count table and no pairs.
    if two_pass_forced() || threads <= 1 {
        two_pass
    } else {
        two_pass.max((8 * positions).min(build_pairs_bytes(threads, kmer_size)))
    }
}

/// Bytes a build holds on top of [`build_transient_bytes`] when it keeps all the
/// pairs of a bin of `positions` indexed positions at once; nothing where the
/// request or the thread count rules that out.
pub fn build_pairs_at_once_bytes(threads: usize, kmer_size: usize, positions: u64) -> u64 {
    if seed_table_request() != SeedTableRequest::Partitioned || threads <= 1 {
        return 0;
    }
    (8 * positions).saturating_sub(build_transient_bytes(threads, kmer_size, positions))
}

/// Calls `emit(kmer, position)` for every indexed position
/// `start_offset + i * step`, `i` in `lo..hi`, whose window is all uppercase
/// `ACGT`, in ascending order: what [`Shape::kmer_at`] returns there, from one
/// window that slides over the bases in between (as in [`chunk_seeds_into`]).
/// The shape must be at most 32 bases long.
fn for_each_kmer(
    seq: &[u8],
    shape: &Shape,
    start_offset: usize,
    step: usize,
    lo: usize,
    hi: usize,
    mut emit: impl FnMut(u32, u32),
) {
    if lo >= hi {
        return;
    }
    let size = shape.size;
    debug_assert!(size <= 32 && size > 0);
    let runs = care_runs(shape);
    let mut code = [N_NT; 256];
    for (c, slot) in code.iter_mut().enumerate() {
        *slot = nt_char_to_int(c as u8);
    }
    let keep = if size == 32 {
        u64::MAX
    } else {
        (1u64 << (2 * size)) - 1
    };
    let first = start_offset + lo * step;
    let last = start_offset + (hi - 1) * step;
    let (mut window, mut run) = (0u64, 0usize);
    for &c in &seq[first..first + size - 1] {
        let v = code[c as usize];
        if v == N_NT {
            run = 0;
        } else {
            run += 1;
            window = ((window << 2) | u64::from(v)) & keep;
        }
    }
    let mut until = 0usize;
    for pos in first..=last {
        let v = code[seq[pos + size - 1] as usize];
        if v == N_NT {
            run = 0;
        } else {
            run += 1;
            window = ((window << 2) | u64::from(v)) & keep;
        }
        if until == 0 {
            until = step;
            if run >= size {
                let mut kmer = 0u64;
                for &(down, mask, up) in &runs {
                    kmer |= ((window >> down) & mask) << up;
                }
                emit(kmer as u32, pos as u32);
            }
        }
        until -= 1;
    }
}

/// The care positions of a shape as runs of adjacent positions, for pulling a
/// k-mer out of a sliding window: window index `p` (0 is the oldest base) sits
/// at bits `2 * (size - 1 - p)`, and a run of `r` adjacent care positions
/// `pos[i]..=pos[i + r - 1]` moves as one field to bits `2 * (k - i - r)` of the
/// k-mer, where `kmer = (kmer << 2) + code` over the care positions in order
/// puts it. Each entry is `(shift down, mask, shift up)`.
fn care_runs(shape: &Shape) -> Vec<(u32, u64, u32)> {
    let (size, k) = (shape.size, shape.kmer_size);
    let mut runs = Vec::with_capacity(k);
    let mut i = 0;
    while i < k {
        let mut r = 1;
        while i + r < k && shape.pos[i + r] == shape.pos[i + r - 1] + 1 {
            r += 1;
        }
        let last = shape.pos[i + r - 1];
        runs.push((
            (2 * (size - 1 - last)) as u32,
            (1u64 << (2 * r)) - 1,
            (2 * (k - i - r)) as u32,
        ));
        i += r;
    }
    runs
}

/// [`chunk_seeds`] written straight into `dst`, returning how many seeds were
/// written: the same values in the same order, without the `Vec` and the copy
/// out of it (round 107). `dst` must hold [`max_seeds`] of the range's span;
/// a shorter one is a caller bug and panics on the first write past its end.
///
/// The window slides instead of being decoded again at every position: one
/// 2-bit code enters a `u64` of the last `shape.size` bases, a counter says how
/// many of them in a row were codeable, and the k-mer is the care positions
/// pulled out of that word run by run. [`Shape::kmer_at`] reads `size` bases and
/// `kmer_size` of them again for every position, which is most of what a
/// single-threaded batch cost. A shape longer than 32 bases does not fit the
/// word and takes the plain loop.
pub fn chunk_seeds_into(
    seq: &[u8],
    shape: &Shape,
    transitions: bool,
    range: (u32, u32),
    dst: &mut [u64],
) -> usize {
    if shape.size > 32 || shape.size == 0 {
        return chunk_seeds_into_plain(seq, shape, transitions, range, dst);
    }
    if range.0 >= range.1 {
        return 0;
    }
    let size = shape.size;
    let k = shape.kmer_size;
    let runs = care_runs(shape);
    let flips: Vec<u64> = (0..k)
        .filter(|&t| transitions && shape.transition[t])
        .map(|t| TRANSITION_MASK << (2 * t))
        .collect();
    let mut code = [N_NT; 256];
    for (c, slot) in code.iter_mut().enumerate() {
        *slot = nt_char_to_int(c as u8);
    }
    let keep = if size == 32 {
        u64::MAX
    } else {
        (1u64 << (2 * size)) - 1
    };

    // `run` counts the codeable bases in a row that end at the newest one. An
    // uncodeable base leaves the word alone: the `size` codeable bases that must
    // follow before a window is valid again push every older bit out of it.
    let (mut window, mut run) = (0u64, 0usize);
    let start = range.0 as usize;
    for &c in &seq[start..start + size - 1] {
        let v = code[c as usize];
        if v == N_NT {
            run = 0;
        } else {
            run += 1;
            window = ((window << 2) | u64::from(v)) & keep;
        }
    }
    let mut at = 0;
    for j in range.0..range.1 {
        let v = code[seq[j as usize + size - 1] as usize];
        if v == N_NT {
            run = 0;
            continue;
        }
        run += 1;
        window = ((window << 2) | u64::from(v)) & keep;
        if run < size {
            continue;
        }
        let mut kmer = 0u64;
        for &(down, mask, up) in &runs {
            kmer |= ((window >> down) & mask) << up;
        }
        let low = u64::from(j);
        dst[at] = (kmer << 32) | low;
        at += 1;
        for &flip in &flips {
            dst[at] = ((kmer ^ flip) << 32) | low;
            at += 1;
        }
    }
    at
}

/// [`chunk_seeds_into`] with [`Shape::kmer_at`] at every position, for a shape
/// the sliding word cannot hold.
fn chunk_seeds_into_plain(
    seq: &[u8],
    shape: &Shape,
    transitions: bool,
    range: (u32, u32),
    dst: &mut [u64],
) -> usize {
    let mut at = 0;
    for j in range.0..range.1 {
        let kmer = shape.kmer_at(seq, j as usize);
        if kmer == INVALID_KMER {
            continue;
        }
        dst[at] = ((kmer as u64) << 32) | j as u64;
        at += 1;
        if transitions {
            for t in 0..shape.kmer_size {
                if shape.transition[t] {
                    let idx = (kmer as u64) ^ (TRANSITION_MASK << (2 * t));
                    dst[at] = (idx << 32) | j as u64;
                    at += 1;
                }
            }
        }
    }
    at
}

/// [`chunk_seeds`] over `threads` scoped workers.
///
/// Query positions split into contiguous ranges, each worker runs the exact
/// same [`chunk_seeds`] over its range, and the parts are concatenated in
/// original position order — so the seed sequence is bit-identical to the
/// single-threaded result, which the output hash depends on. No shared state
/// and no locking in the hot loop.
#[cfg(test)]
pub fn chunk_seeds_parallel(
    seq: &[u8],
    shape: &Shape,
    transitions: bool,
    range: (u32, u32),
    threads: usize,
) -> Vec<u64> {
    let parts = chunk_seeds_parts(seq, shape, transitions, range, threads);
    let mut out = vec![0u64; parts.iter().map(Vec::len).sum()];
    concat_parts(&parts, &mut out);
    out
}

/// The per-worker pieces of [`chunk_seeds_parallel`], before concatenation.
///
/// Split out so a caller can concatenate straight into pinned host memory
/// instead of into a `Vec` it would then have to copy again. The
/// pieces are in original position order, so concatenating them reproduces
/// `chunk_seeds` exactly.
pub fn chunk_seeds_parts(
    seq: &[u8],
    shape: &Shape,
    transitions: bool,
    range: (u32, u32),
    threads: usize,
) -> Vec<Vec<u64>> {
    let span = range.1.saturating_sub(range.0);
    // Below this a worker does less work than it costs to spawn one.
    if threads <= 1 || span < 4096 {
        return vec![chunk_seeds(seq, shape, transitions, range)];
    }

    let threads = threads.min(span as usize);
    let per = span.div_ceil(threads as u32);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let lo = range.0 + t as u32 * per;
                let hi = (lo.saturating_add(per)).min(range.1);
                scope.spawn(move || {
                    if lo < hi {
                        chunk_seeds(seq, shape, transitions, (lo, hi))
                    } else {
                        Vec::new()
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("seed worker panicked"))
            .collect()
    })
}

/// Copies `parts` end to end into `dst`, returning how many seeds were written.
///
/// Panics if `dst` is too small — the caller sizes it from the worst case, so a
/// short buffer is a bug, not a runtime condition.
pub fn concat_parts(parts: &[Vec<u64>], dst: &mut [u64]) -> usize {
    let mut at = 0;
    for part in parts {
        dst[at..at + part.len()].copy_from_slice(part);
        at += part.len();
    }
    at
}

/// Worst-case seeds one chunk of `span` query positions can emit: every
/// position yields the exact k-mer plus, with transitions on, one variant per
/// care position. Used to size the pinned staging buffers once up front so the
/// seed worker never allocates.
pub fn max_seeds(span: u32, shape: &Shape, transitions: bool) -> usize {
    let per_pos = if transitions { 1 + shape.kmer_size } else { 1 };
    span as usize * per_pos
}

/// Look-ahead budget for host seed slots (round 107): each producer of the
/// host seed stream holds one worst-case batch.
pub const SEED_AHEAD_BYTES: usize = 512 << 20;
/// More producers than this only add slots. One thread writes a 250 kbp batch
/// in 3 to 5 ms, and the fastest card measured (an A100 on hg38 x mm39) takes a
/// batch every 4 ms on average, so six stay well ahead of it.
pub const SEED_PRODUCERS_MAX: usize = 6;

/// Producers of the host seed stream for a worker with `threads` threads and
/// worst-case batches of `slot_bytes`: one per thread, at most
/// [`SEED_PRODUCERS_MAX`], and no more than [`SEED_AHEAD_BYTES`] of slots.
/// The host budget counts the same number.
pub fn producers_auto(threads: usize, slot_bytes: usize) -> usize {
    threads
        .min(SEED_PRODUCERS_MAX)
        .min(SEED_AHEAD_BYTES / slot_bytes.max(1))
        .max(1)
}

// ---------------------------------------------------------------------------
// R92 PR1: exact per-unit seed-hit estimator (host histograms, no GPU).
//
// `hits(r,q) = Σ_key refcount_r[key] × qrycount_q[key]` over the pipeline's
// compact seeds, equal to the `unit ledger:` hits column EXACTLY (u64). The
// reference counts use the same strided starts as `SeedTable::build`; the
// query counts use the same `kmer_at` validity over the same chunked LASTZ
// interval schedule (`seed::chunks`) on both strands, counting base k-mers
// only. Transition variants are folded into the reference side
// (`fold_transitions`): each valid query window emits
// its base k-mer plus one single-transition variant per transition care
// position (`push_seeds`, identical XOR mapping), so dotting base query
// counts against folded reference weights is exact.

/// Dense-table guard: one `u32` table is `4^k` entries; `k > 12` needs 1 GiB+
/// per table and is out of scope (12of19 default).
pub fn estimator_table_size(shape: &Shape) -> Result<usize, String> {
    if shape.kmer_size > 12 {
        return Err(format!(
            "hits-estimate needs k<=12 (dense 4^k table); seed has k={}",
            shape.kmer_size
        ));
    }
    Ok(1usize << (2 * shape.kmer_size))
}

/// Strided start sequence shared with `SeedTable::build`: positions
/// `start + i*step` for `i in 0..num` over a block of `len` bases.
fn ref_starts(len: usize, shape: &Shape, step: u32) -> (usize, usize) {
    let step = step.max(1) as usize;
    let offset = (shape.size + 1) % step;
    let num = (len + offset).saturating_sub(shape.size) / step;
    (step - offset, num)
}

/// Serial reference histogram over a packed block.
pub fn count_ref_block(buf: &[u8], shape: &Shape, step: u32) -> Result<Vec<u32>, String> {
    let nkeys = estimator_table_size(shape)?;
    let (start, num) = ref_starts(buf.len(), shape, step);
    let step = step.max(1) as usize;
    let mut hist = vec![0u32; nkeys];
    for i in 0..num {
        let k = shape.kmer_at(buf, start + i * step);
        if k != INVALID_KMER {
            hist[k as usize] += 1;
        }
    }
    Ok(hist)
}

/// Adds `part` into `acc` (the histogram merge; counts commute).
fn merge_hist(acc: &mut [u32], part: &[u32]) {
    for (a, b) in acc.iter_mut().zip(part.iter()) {
        *a += *b;
    }
}

/// Parallel reference histogram: the step-index range is split into `threads`
/// contiguous pieces, each counted into a private table, then summed.
/// Exact — the same positions, the same `kmer_at`, addition commutes.
pub fn count_ref_block_parallel(
    buf: &[u8],
    shape: &Shape,
    step: u32,
    threads: usize,
) -> Result<Vec<u32>, String> {
    let nkeys = estimator_table_size(shape)?;
    let (start, num) = ref_starts(buf.len(), shape, step);
    let step = step.max(1) as usize;
    let threads = threads.max(1).min(num.max(1));
    if threads <= 1 || num < 1 << 16 {
        return count_ref_block(buf, shape, step as u32);
    }
    let per = num.div_ceil(threads);
    let parts: Vec<Vec<u32>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (lo, hi) = (t * per, ((t + 1) * per).min(num));
                s.spawn(move || {
                    let mut h = vec![0u32; nkeys];
                    for i in lo..hi {
                        let k = shape.kmer_at(buf, start + i * step);
                        if k != INVALID_KMER {
                            h[k as usize] += 1;
                        }
                    }
                    h
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("ref count worker panicked"))
            .collect()
    });
    let mut out = vec![0u32; nkeys];
    for p in &parts {
        merge_hist(&mut out, p);
    }
    Ok(out)
}

/// Key-sharded reference histogram (the alternative strategy, test-only):
/// the key space is split into `threads` shards and every worker scans all
/// positions, counting only keys in its shard — one final table, no merge,
/// but `kmer_at` is recomputed per worker. Used by the strategy-comparison
/// test; production uses the private-tables path above.
#[cfg(test)]
pub fn count_ref_block_sharded(
    buf: &[u8],
    shape: &Shape,
    step: u32,
    threads: usize,
) -> Result<Vec<u32>, String> {
    let nkeys = estimator_table_size(shape)?;
    let (start, num) = ref_starts(buf.len(), shape, step);
    let step = step.max(1) as usize;
    let threads = threads.max(1).min(nkeys);
    if threads <= 1 || num < 1 << 16 {
        return count_ref_block(buf, shape, step as u32);
    }
    let kper = nkeys.div_ceil(threads);
    let mut out = vec![0u32; nkeys];
    {
        let ptr = out.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for t in 0..threads {
                let (lo, hi) = (t * kper, ((t + 1) * kper).min(nkeys));
                s.spawn(move || {
                    for i in 0..num {
                        let k = shape.kmer_at(buf, start + i * step);
                        if k != INVALID_KMER && (k as usize) >= lo && (k as usize) < hi {
                            // SAFETY: shard `[lo,hi)` is disjoint across workers.
                            unsafe { *(ptr as *mut u32).add(k as usize) += 1 };
                        }
                    }
                });
            }
        });
    }
    Ok(out)
}

/// Base-k-mer query histogram over one packed query bin: `fwd[..block_len]`
/// plus its reverse complement `rc`, over the pipeline's LASTZ intervals on
/// the strands `plus`/`minus` select. Counts base k-mers only (variants are
/// folded on the reference side). `stride` S counts window starts `j % S == 0`
/// only (S=1 is exact); `lastz_interval` is the pipeline's `-I` tiling and
/// `wga_chunk` its `-C` chunk size. Interval ranges are split across `threads`
/// workers with private tables, then merged.
///
/// The window starts are exactly the ones `seed_and_filter_all` batches: each
/// interval is cut with [`chunks`] (whose exclusive upper bound includes the
/// interval end only when the chunk walk does not land on it — an interval
/// whose length is a multiple of `wga_chunk` leaves its end to the next
/// interval), and the minus strand chunks its RC mirror
/// `(qbl - end, qbl - start)`, whose own chunk alignment decides the mirror's
/// duplicate windows. Counting raw interval ends as inclusive, as the first
/// version of this estimator did, double-counts every shared endpoint.
#[allow(clippy::too_many_arguments)]
pub fn count_query_block(
    fwd: &[u8],
    rc: &[u8],
    block_len: usize,
    shape: &Shape,
    plus: bool,
    minus: bool,
    lastz_interval: u32,
    wga_chunk: u32,
    stride: u32,
    threads: usize,
) -> Result<Vec<u32>, String> {
    let nkeys = estimator_table_size(shape)?;
    if block_len <= shape.size {
        return Ok(vec![0u32; nkeys]);
    }
    let stride = stride.max(1);
    let qbl = (block_len - shape.size) as u32;
    let ivs = crate::sequence::intervals(block_len, shape.size, lastz_interval);
    // (is_rc, lo, hi-exclusive) window-start ranges, exactly the pipeline's.
    let mut ranges: Vec<(bool, u32, u32)> = Vec::new();
    if plus {
        for &(s, e) in &ivs {
            ranges.extend(
                chunks(s, e, wga_chunk)
                    .into_iter()
                    .map(|(lo, hi)| (false, lo, hi)),
            );
        }
    }
    if minus {
        for &(s, e) in &ivs {
            let (lo, hi) = (qbl - e, qbl - s);
            ranges.extend(
                chunks(lo, hi, wga_chunk)
                    .into_iter()
                    .map(|(l, h)| (true, l, h)),
            );
        }
    }
    let threads = threads.max(1).min(ranges.len().max(1));
    if threads <= 1 {
        let mut hist = vec![0u32; nkeys];
        count_query_ranges(&mut hist, fwd, rc, &ranges, shape, stride);
        return Ok(hist);
    }
    let chunk = ranges.len().div_ceil(threads);
    let parts: Vec<Vec<u32>> = std::thread::scope(|s| {
        let hs: Vec<_> = ranges
            .chunks(chunk)
            .map(|ch| {
                s.spawn(move || {
                    let mut h = vec![0u32; nkeys];
                    count_query_ranges(&mut h, fwd, rc, ch, shape, stride);
                    h
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("query count worker panicked"))
            .collect()
    });
    let mut out = vec![0u32; nkeys];
    for p in &parts {
        merge_hist(&mut out, p);
    }
    Ok(out)
}

/// Counts base k-mers over `ranges` (exclusive upper bounds, as [`chunks`]
/// returns them), skipping window starts `j % stride != 0`.
fn count_query_ranges(
    hist: &mut [u32],
    fwd: &[u8],
    rc: &[u8],
    ranges: &[(bool, u32, u32)],
    shape: &Shape,
    stride: u32,
) {
    for &(is_rc, lo, hi) in ranges {
        let seq = if is_rc { rc } else { fwd };
        for j in lo..hi {
            if j % stride != 0 {
                continue;
            }
            let k = shape.kmer_at(seq, j as usize);
            if k != INVALID_KMER {
                hist[k as usize] += 1;
            }
        }
    }
}

/// Folds transition variants into reference weights (exact): each valid query
/// window with base k-mer K scores `ref[K] + Σ_t ref[K ^ (2 << 2t)]` over the
/// transition care positions — the same XOR mapping `push_seeds` emits — so
/// the dot product over base query counts reproduces the pipeline's compact
/// seed stream. `transitions=false` (`--notransition`) is the identity. u64:
/// a 3 Gbp bin holds < 2^32 positions per key, but the 13-term sum can exceed
/// u32 (up to 13× the bin length in the adversarial case).
pub fn fold_transitions(ref_hist: &[u32], shape: &Shape, transitions: bool) -> Vec<u64> {
    let mut out: Vec<u64> = ref_hist.iter().map(|&c| c as u64).collect();
    if !transitions {
        return out;
    }
    let n = out.len();
    for t in 0..shape.kmer_size {
        if !shape.transition[t] {
            continue;
        }
        let flip = 2u32 << (2 * t);
        for k in 0..n {
            out[k] += ref_hist[(k as u32 ^ flip) as usize] as u64;
        }
    }
    out
}

/// Row-major `r*Q+q` exact hit counts: each reference histogram is folded
/// once, then dotted against every query histogram (base counts). One scoped
/// worker per reference bin (R is small); the histogram passes above carry
/// the thread budget.
pub fn unit_hits(
    ref_hists: &[Vec<u32>],
    qry_hists: &[Vec<u32>],
    shape: &Shape,
    transitions: bool,
) -> Vec<u64> {
    let q = qry_hists.len();
    if ref_hists.is_empty() || q == 0 {
        return Vec::new();
    }
    let rows: Vec<Vec<u64>> = std::thread::scope(|s| {
        let hs: Vec<_> = ref_hists
            .iter()
            .map(|rh| {
                s.spawn(move || {
                    let w = fold_transitions(rh, shape, transitions);
                    qry_hists
                        .iter()
                        .map(|qh| w.iter().zip(qh.iter()).map(|(&a, &b)| a * b as u64).sum())
                        .collect::<Vec<u64>>()
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("dot worker panicked"))
            .collect()
    });
    rows.into_iter().flatten().collect()
}

/// The `[i, e)` chunk bounds `seeder.cpp` walks for one interval. `end` is
/// inclusive there, hence the `+ 1`.
pub fn chunks(start: u32, end: u32, wga_chunk: u32) -> Vec<(u32, u32)> {
    // u64 so `i + wga_chunk` and `end + 1` cannot wrap near 2^32; the bounds
    // pushed are exactly the u32 values the seeder loop walks.
    let mut out = Vec::new();
    let step = u64::from(wga_chunk.max(1));
    let end = u64::from(end);
    let mut i = u64::from(start);
    while i < end {
        let hi = (i + step).min(end + 1).min(u64::from(u32::MAX));
        out.push((i as u32, hi as u32));
        i += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round 107: the direct writer is `chunk_seeds` without the `Vec`. Random
    /// sequence with soft-masked runs, N runs and separators, both shapes, with
    /// and without transitions, ranges that start and end inside invalid runs.
    #[test]
    fn chunk_seeds_into_equals_chunk_seeds() {
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut seq = Vec::with_capacity(60_000);
        while seq.len() < 60_000 {
            let run = (next() % 400 + 1) as usize;
            let kind = next() % 10;
            for _ in 0..run {
                let b = b"ACGT"[(next() >> 33) as usize & 3];
                seq.push(match kind {
                    0 => b.to_ascii_lowercase(),
                    1 => b'N',
                    2 => b'n',
                    _ => b,
                });
            }
        }
        seq.truncate(60_000);
        // 19 and 22 bases, a short one, exactly 32 bases (the word is full), and
        // 36 bases (the plain loop).
        let wide32 = "1100000000000000000000000000011";
        let wide36 = "110000000000000000000000000000000011";
        assert_eq!((wide32.len() + 1, wide36.len()), (32, 36));
        let wide32 = format!("1{wide32}");
        for spec in ["12of19", "14of22", "1101011", wide32.as_str(), wide36] {
            let shape = Shape::parse(spec).unwrap();
            let last = (seq.len() - shape.size) as u32;
            for transitions in [false, true] {
                for range in [(0, last), (0, 1), (17, 17), (123, 9_876), (40_000, last)] {
                    let want = chunk_seeds(&seq, &shape, transitions, range);
                    let mut dst = vec![u64::MAX; max_seeds(range.1 - range.0, &shape, transitions)];
                    let n = chunk_seeds_into(&seq, &shape, transitions, range, &mut dst);
                    assert_eq!(n, want.len(), "{spec} {transitions} {range:?}");
                    assert_eq!(&dst[..n], &want[..], "{spec} {transitions} {range:?}");
                    assert!(
                        dst[n..].iter().all(|&v| v == u64::MAX),
                        "wrote past its count"
                    );
                }
            }
        }
    }

    #[test]
    fn shape_12of19_matches_kegalign() {
        let s = Shape::parse("12of19").unwrap();
        assert_eq!(s.size, 19);
        assert_eq!(s.kmer_size, 12);
        assert_eq!(s.pos, vec![0, 1, 2, 4, 7, 8, 11, 13, 15, 16, 17, 18]);
        assert!(s.transition.iter().all(|&t| t), "12of19 is all-T");
    }

    #[test]
    fn kmer_packs_care_positions_only() {
        assert!(
            Shape::parse("101").is_err(),
            "3 care positions is below the k-mer floor"
        );
        let s = Shape::parse("1111").unwrap();
        assert_eq!(s.kmer_at(b"ACGT", 0), 0b00_01_10_11);
        // A separator anywhere in the window invalidates it.
        assert_eq!(s.kmer_at(b"ACG&", 0), INVALID_KMER);
        // ... including at a don't-care position.
        let spaced = Shape::parse("11011").unwrap();
        assert_eq!(spaced.kmer_at(b"AC&GT", 0), INVALID_KMER);
        assert_eq!(spaced.kmer_at(b"ACAGT", 0), 0b00_01_10_11);
    }

    #[test]
    fn transition_seeds_flip_one_purine_pyrimidine_bit() {
        let s = Shape::parse("1111").unwrap();
        let mut out = Vec::new();
        let n = s.push_seeds(0, 7, true, &mut out);
        assert_eq!(n, 5, "1 exact + 4 transition variants");
        assert_eq!(out[0], 7, "kmer 0 at position 7");
        assert_eq!(out[1] >> 32, 2, "TRANSITION_MASK << 0");
        assert_eq!(out[4] >> 32, 2 << 6, "TRANSITION_MASK << 2*3");
        assert!(out.iter().all(|o| o & 0xFFFF_FFFF == 7));
    }

    /// The parallel builder must be byte-identical to the
    /// serial one for every thread count, on both tables.
    ///
    /// `pos_table` order matters as much as its contents: within-bucket order is
    /// hit order, hit order sets `MAX_HITS` chunk boundaries, and those decide
    /// the final HSP set. A bucket-order drift would change output with no
    /// change in any count, so this compares the vectors, not multisets.
    /// Round 112: the partitioned builder against the serial one, over masked
    /// runs, Ns and record separators, for shapes with one, four and 256
    /// partitions, strides, thread counts on both sides of the partition
    /// count, and pair budgets from one round to a round per partition.
    #[test]
    fn partitioned_seed_table_is_byte_identical_to_serial() {
        let mut seq = Vec::new();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..150_000u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            seq.push(match (x >> 11) % 64 {
                0..=15 => b'A',
                16..=31 => b'C',
                32..=47 => b'G',
                48..=58 => b'T',
                59 => b'N',
                60 => b'a',
                61 => b'c',
                62 => b'g',
                _ => b't',
            });
            if i % 9_973 == 0 || i % 40_001 == 0 {
                seq.push(crate::sequence::SEP);
            }
        }
        // 4, 9 and 12 care positions: 8, 18 and 24 k-mer bits.
        for spec in ["1111", "1101011011011", "12of19"] {
            let shape = Shape::parse(spec).unwrap();
            for step in [1usize, 2, 3] {
                let want = SeedTable::build(&seq, &shape, step as u32);
                let offset = (shape.size + 1) % step;
                let start_offset = step - offset;
                let num_steps = (seq.len() + offset).saturating_sub(shape.size) / step;
                // All pairs at once, a few rounds, and a round per partition.
                for (threads, max_pairs) in [
                    (1usize, usize::MAX),
                    (2, 40_000),
                    (3, usize::MAX),
                    (8, 1),
                    (16, 9_000),
                ] {
                    let got = SeedTable::build_partitioned(
                        &seq,
                        &shape,
                        step,
                        threads,
                        start_offset,
                        num_steps,
                        max_pairs,
                    );
                    assert!(
                        got.index_table == want.index_table,
                        "index_table differs: {spec} step {step} threads {threads} pairs {max_pairs}"
                    );
                    assert!(
                        got.pos_table == want.pos_table,
                        "pos_table differs: {spec} step {step} threads {threads} pairs {max_pairs}"
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_seed_table_is_byte_identical_to_serial() {
        // Deterministic pseudo-random DNA with the awkward cases mixed in:
        // Ns (kill a window), soft-masked lowercase (also killed by kmer_at),
        // record separators, and short inter-separator runs.
        let mut seq = Vec::new();
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for i in 0..400_000u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            seq.push(match (x >> 11) % 64 {
                0..=15 => b'A',
                16..=31 => b'C',
                32..=47 => b'G',
                48..=58 => b'T',
                59 => b'N',
                60 => b'a',
                61 => b'c',
                62 => b'g',
                _ => b't',
            });
            // Record boundaries, including some very short records, so
            // window validation across SEP is exercised.
            if i % 9_973 == 0 || i % 40_001 == 0 {
                seq.push(crate::sequence::SEP);
            }
        }

        // 12of19 only: 14of22 needs a 4^14 count table (1 GB per build) and adds
        // nothing here, since the builder's behaviour varies with `step` and
        // window validation, not with the number of care positions. Transitions
        // are a `chunk_seeds` concern and never reach this table.
        for shape_str in ["12of19"] {
            let shape = Shape::parse(shape_str).unwrap();
            for step in [1u32, 3] {
                let want = SeedTable::build(&seq, &shape, step);
                for threads in [1usize, 2, 3, 4, 8] {
                    let got = SeedTable::build_parallel(&seq, &shape, step, threads);
                    assert_eq!(
                        got.index_table, want.index_table,
                        "index_table differs: {shape_str} step {step} threads {threads}"
                    );
                    assert_eq!(
                        got.pos_table, want.pos_table,
                        "pos_table differs (order matters): {shape_str} step {step} \
                         threads {threads}"
                    );
                }
            }
        }
    }

    /// A tiny input must take the same path and stay identical — the builder
    /// short-circuits below a size threshold, and that branch needs covering.
    #[test]
    fn parallel_seed_table_handles_tiny_and_empty_input() {
        let shape = Shape::parse("12of19").unwrap();
        for seq in [b"".to_vec(), b"ACGT".to_vec(), b"AAAAACGTA".to_vec()] {
            let want = SeedTable::build(&seq, &shape, 1);
            for threads in [1usize, 4] {
                let got = SeedTable::build_parallel(&seq, &shape, 1, threads);
                assert_eq!(got.index_table, want.index_table, "len {}", seq.len());
                assert_eq!(got.pos_table, want.pos_table, "len {}", seq.len());
            }
        }
    }

    #[test]
    fn seed_table_buckets_are_contiguous_and_skip_position_zero() {
        let shape = Shape::parse("1111").unwrap();
        // step=1 -> start_offset=1, so position 0 is never indexed.
        let t = SeedTable::build(b"AAAAACGTA", &shape, 1);
        assert_eq!(t.pos_table.len(), 5, "positions 1..=5 of a 9-base ref");
        let bucket_0 = &t.pos_table[..t.hit_count(0) as usize];
        assert_eq!(bucket_0, &[1], "only pos 1 is AAAA");
        assert_eq!(t.index_table[t.index_table.len() - 1], 5);
    }

    #[test]
    fn parallel_seed_generation_is_bit_identical() {
        let shape = Shape::parse("12of19").unwrap();
        let seq: Vec<u8> = (0..30_000u32)
            .map(|i| {
                // Deterministic pseudo-sequence with N and soft-masked runs, so
                // workers hit invalid windows and empty sub-ranges too.
                match (i.wrapping_mul(2654435761) >> 11) % 64 {
                    0 => b'N',
                    1 => b'a',
                    n => b"ACGT"[(n % 4) as usize],
                }
            })
            .collect();
        let range = (0, seq.len() as u32 - shape.size as u32);
        let one = chunk_seeds(&seq, &shape, true, range);
        assert!(!one.is_empty());
        for threads in [1, 2, 3, 4, 8, 16] {
            let many = chunk_seeds_parallel(&seq, &shape, true, range, threads);
            assert_eq!(many, one, "{threads} threads changed the seed sequence");
        }
    }

    #[test]
    fn chunking_matches_seeder_loop() {
        assert_eq!(
            chunks(0, 376657, 250_000),
            vec![(0, 250_000), (250_000, 376_658)]
        );
        assert_eq!(chunks(0, 10, 250_000), vec![(0, 11)]);
    }
}
