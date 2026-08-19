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
//! `SeedTable::build_parallel` is the two-pass, lock-free parallel builder:
//! workers count into private tables, a two-level scan turns
//! them into disjoint write cursors, and pass 2 rescans and scatters, so
//! within-bucket order — which fixes `MAX_HITS` chunking — is preserved
//! bit-for-bit. Query-side seeds are generated per work unit; the device-side
//! seed kernels live in `gpu/kernels.rs` (`seed_kmers`/`scatter_seeds`).

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

    /// [`build`](Self::build) across `threads` workers, byte-identical to it.
    ///
    /// The serial builder is 11.4 s on hg38 chr1 — 10.1% of that run's wall
    /// time and its largest CPU stage — which is what earned this.
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
    /// Why the bucket order comes out identical (byte-identity, not just the
    /// same multiset — hit order feeds `MAX_HITS` chunk boundaries,
    /// so a reordering would move HSPs without changing any count): worker
    /// ranges are contiguous and ascending in `i`, and for every k-mer worker
    /// `t`'s cursor region precedes worker `t+1`'s. Positions therefore land in
    /// ascending genomic order within each bucket, exactly as the single-cursor
    /// serial walk produces them.
    ///
    /// Pass 2 recomputes `kmer_at` rather than storing the k-mer of every
    /// position, which also drops the serial builder's `idxs` temporary — 1 GB
    /// on chr1 at `--step 1`.
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
                            // the index is the k-mer position being accumulated
                            #[allow(clippy::needless_range_loop)]
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
/// (into pinned host memory) instead of into a `Vec` it would then have to
/// copy again. The
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

/// The `[i, e)` chunk bounds `seeder.cpp` walks for one interval. `end` is
/// inclusive there, hence the `+ 1`.
pub fn chunks(start: u32, end: u32, wga_chunk: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut i = start;
    while i < end {
        out.push((i, (i + wga_chunk).min(end + 1)));
        i += wga_chunk;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The parallel builder must be byte-identical to the serial one for every
    /// thread count, on both tables.
    ///
    /// `pos_table` order matters as much as its contents: within-bucket order is
    /// hit order, hit order sets `MAX_HITS` chunk boundaries, and those decide
    /// the final HSP set. A bucket-order drift would change output with no
    /// change in any count, so this compares the vectors, not multisets.
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
